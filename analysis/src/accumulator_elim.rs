//! Accumulator elimination: drop loop-carried *accumulator* state from a
//! function's parameter list by relocating it into a returned tuple.
//!
//! A counted loop in SSA threads its whole state `S` forward through every
//! iteration. After [`loop_to_recursion`](crate::loop_to_recursion) that state
//! becomes the parameter list of a tail-recursive lambda. But not every slot of
//! `S` needs to be a *parameter*: a slot is only a genuine parameter if the
//! recursion branches on it (a **driver**, e.g. the counter) or if a driver's
//! next value is computed from it. Every other slot is an **accumulator** — its
//! next value depends only on other accumulators — and can be eliminated from
//! the call interface by computing it on the way *out* of the recursion instead
//! of threading it *in*.
//!
//! Concretely, the iterative Fibonacci loop
//!
//! ```text
//! f(hn, a, b) = if hn == 0 then a else f(hn-1, b, a+b)   // 3 parameters
//! ```
//!
//! is rewritten to a tuple-returning, non-tail recursion over the *driver only*:
//!
//! ```text
//! g(hn) = if hn == 0 then (0, 1)                          // 1 parameter
//!         else let (a, b) = g(hn-1) in (b, a+b)
//! f(n)  = g(n).0                                          // project the result
//! ```
//!
//! The accumulators `a, b` migrate from forward-threaded arguments into the
//! returned pair; the host projects the original return value out of it. This
//! trades O(1) loop stack for O(n) recursion depth — the cost of fewer
//! parameters.
//!
//! # Soundness
//!
//! The tuple is built on the way *up*, which feeds driver values to the
//! accumulator updates in the *reverse* of their forward order. The rewrite is
//! therefore only valid when each accumulator update is independent of the
//! drivers (`stepA` reads only accumulators) — which the classifier enforces by
//! demoting any "accumulator" whose update reads a driver back into the driver
//! set, and refusing to fire when no accumulators remain. The base case returns
//! the loop's initial accumulator values, so those must be constants.

use std::borrow::Cow;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use qcode::{
    builder::Builder,
    context::Context,
    value::{
        BasicBlock, Function, FunctionId, Instruction, InstructionRef, Renameable, ValueId,
        block::BlockId,
        block_param::BlockParamId,
        insn::{CBranch, Mnemonic},
    },
};

use crate::loop_to_recursion::recognize_loop;
use crate::{FunctionPass, PipelineEnv};

#[derive(Default)]
pub struct AccumulatorElim;

impl FunctionPass for AccumulatorElim {
    const NAME: &'static str = "accumulator_elim";

    fn description(&self) -> &'static str {
        "Eliminate loop-carried accumulators by returning them as a tuple"
    }

    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        Ok(accumulator_elim(ctx, fun_id))
    }
}

crate::register_function_pass!(AccumulatorElim);

pub fn accumulator_elim(ctx: &mut Context, host: FunctionId) -> bool {
    let Some(model) = recognize_loop(ctx, host) else {
        return false;
    };
    // v1 handles the canonical single-latch counted loop.
    if model.back_edges.len() != 1 {
        return false;
    }
    let head = model.head;
    let (latch, next_args) = model.back_edges[0].clone();

    let Some(cbranch) = header_cbranch(ctx, head) else {
        return false;
    };
    // The loop *continues* through whichever header edge enters the latch
    // directly; the other edge exits. (Multi-block bodies are out of scope.)
    let (cont_args, exit_block, exit_args, cond_true_is_exit) =
        if cbranch.success_block == latch {
            (
                cbranch.success_args.clone(),
                cbranch.failure_block,
                cbranch.failure_args.clone(),
                false,
            )
        } else if cbranch.failure_block == latch {
            (
                cbranch.failure_args.clone(),
                cbranch.success_block,
                cbranch.success_args.clone(),
                true,
            )
        } else {
            return false;
        };

    // State slots = header parameters.
    let head_info: Vec<(BlockParamId, usize)> = BasicBlock::from_id(ctx, head)
        .params()
        .map(|p| (p.id, p.size()))
        .collect();
    let m = head_info.len();
    if m == 0 || next_args.len() != m || model.init_args.len() != m {
        return false;
    }
    let head_params: Vec<BlockParamId> = head_info.iter().map(|&(id, _)| id).collect();
    let head_sizes: Vec<usize> = head_info.iter().map(|&(_, sz)| sz).collect();
    let head_index: HashMap<BlockParamId, usize> = head_params
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i))
        .collect();

    // Resolve the body's next-state expressions (latch scope) back to header
    // params by binding each latch param to the value the header passed for it.
    let latch_params: Vec<BlockParamId> = BasicBlock::from_id(ctx, latch)
        .params()
        .map(|p| p.id)
        .collect();
    if latch_params.len() != cont_args.len() {
        return false;
    }
    let body_bindings: HashMap<BlockParamId, ValueId> = latch_params
        .iter()
        .copied()
        .zip(cont_args.iter().copied())
        .collect();

    // The exit returns a single value; we project it from the result tuple.
    let Some(ret_val) = block_return_value(ctx, exit_block) else {
        return false;
    };
    let exit_params: Vec<BlockParamId> = BasicBlock::from_id(ctx, exit_block)
        .params()
        .map(|p| p.id)
        .collect();
    if exit_params.len() != exit_args.len() {
        return false;
    }
    let exit_bindings: HashMap<BlockParamId, ValueId> = exit_params
        .iter()
        .copied()
        .zip(exit_args.iter().copied())
        .collect();

    // --- Classify drivers vs accumulators --------------------------------
    let deps: Vec<HashSet<usize>> = next_args
        .iter()
        .map(|&arg| head_param_deps(ctx, arg, &body_bindings, &head_index))
        .collect();
    let cond_deps = head_param_deps(ctx, cbranch.condition, &body_bindings, &head_index);

    let mut is_driver = vec![false; m];
    for &i in &cond_deps {
        is_driver[i] = true;
    }
    loop {
        let mut changed = false;
        for i in 0..m {
            // Drivers are closed under their own next-state dependencies.
            if is_driver[i] {
                for &k in &deps[i] {
                    if !is_driver[k] {
                        is_driver[k] = true;
                        changed = true;
                    }
                }
            }
        }
        for i in 0..m {
            // An "accumulator" whose update reads a driver cannot be eliminated.
            if !is_driver[i] && deps[i].iter().any(|&k| is_driver[k]) {
                is_driver[i] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let d_slots: Vec<usize> = (0..m).filter(|&i| is_driver[i]).collect();
    let a_slots: Vec<usize> = (0..m).filter(|&i| !is_driver[i]).collect();
    if a_slots.is_empty() {
        return false; // nothing to eliminate
    }

    // Gate: the returned value must be a function of accumulators only.
    let ret_deps = head_param_deps(ctx, ret_val, &exit_bindings, &head_index);
    if ret_deps.iter().any(|&i| is_driver[i]) {
        return false;
    }
    // Gate: accumulator initial values become the base case, so must be constant.
    for &i in &a_slots {
        if !is_const_literal(ctx, model.init_args[i]) {
            return false;
        }
    }

    transform(
        ctx,
        host,
        &model,
        &Plan {
            head_params,
            head_sizes,
            next_args,
            cbranch,
            cond_true_is_exit,
            body_bindings,
            exit_bindings,
            ret_val,
            d_slots,
            a_slots,
        },
    );
    true
}

struct Plan {
    head_params: Vec<BlockParamId>,
    head_sizes: Vec<usize>,
    next_args: Vec<ValueId>,
    cbranch: CBranch,
    cond_true_is_exit: bool,
    body_bindings: HashMap<BlockParamId, ValueId>,
    exit_bindings: HashMap<BlockParamId, ValueId>,
    ret_val: ValueId,
    d_slots: Vec<usize>,
    a_slots: Vec<usize>,
}

fn transform(ctx: &mut Context, host: FunctionId, model: &crate::loop_to_recursion::LoopModel, p: &Plan) {
    let host_name = Function::from_id(ctx, host).name().to_owned();
    let g_name = ctx.get_unique_name(Cow::Owned(format!("{host_name}_acc")));
    let g = Function::make_lambda(ctx, g_name.clone())
        .expect("accumulator lambda name was deduplicated")
        .id;

    // Three fresh blocks: header (root, drivers in), base case, recursive case.
    let g_head = BasicBlock::make(ctx).id;
    let base = BasicBlock::make(ctx).id;
    let rec = BasicBlock::make(ctx).id;
    for &b in &[g_head, base, rec] {
        Function::from_id_mut(ctx, g).add_block(b);
    }
    let base_name = ctx_unique(ctx, format!("{g_name}_base"));
    let _ = BasicBlock::from_id_mut(ctx, base).rename(base_name);
    let rec_name = ctx_unique(ctx, format!("{g_name}_rec"));
    let _ = BasicBlock::from_id_mut(ctx, rec).rename(rec_name);
    Function::from_id_mut(ctx, g)
        .set_root(g_head)
        .expect("fresh header has no conflicting address");

    // g's parameters: the drivers only. Map each driver header-param to its new
    // counterpart so cloned `cond`/`stepD` expressions read g's params.
    let mut driver_subst: HashMap<ValueId, ValueId> = HashMap::default();
    {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, g_head));
        unsafe { b.dont_finalize() };
        for &i in &p.d_slots {
            let pid = b.push_param(p.head_sizes[i]).id;
            driver_subst.insert(
                ValueId::BlockParam(p.head_params[i]),
                ValueId::BlockParam(pid),
            );
        }
    }

    // Header: recompute the loop condition over g's drivers, then branch.
    let cond = clone_value(ctx, p.cbranch.condition, &mut driver_subst, &p.body_bindings, g_head);
    {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, g_head));
        if p.cond_true_is_exit {
            b.push_cbranch(cond, base, rec);
        } else {
            b.push_cbranch(cond, rec, base);
        }
    }

    // Base case: return the initial accumulator tuple (constants).
    {
        let base_fields: Vec<ValueId> = p
            .a_slots
            .iter()
            .map(|&i| const_at_size(ctx, model.init_args[i], p.head_sizes[i]))
            .collect();
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, base));
        let tuple = b.push_tuple(base_fields).id();
        b.push_return_value(tuple);
    }

    // Recursive case: recurse on the drivers' next values, unpack the deeper
    // accumulators, apply the accumulator update, return the new tuple.
    {
        // stepD: driver-next values (read only drivers).
        let driver_next: Vec<ValueId> = p
            .d_slots
            .iter()
            .map(|&i| clone_value(ctx, p.next_args[i], &mut driver_subst, &p.body_bindings, rec))
            .collect();

        let (deep, acc_vals) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, rec));
            unsafe { b.dont_finalize() };
            let deep = b.push_apply(g, driver_next).id();
            let acc_vals: Vec<ValueId> = (0..p.a_slots.len())
                .map(|pos| b.push_extract(deep, pos).id())
                .collect();
            (deep, acc_vals)
        };
        let _ = deep;

        // stepA: accumulator-next, reading the unpacked accumulators.
        let mut acc_subst: HashMap<ValueId, ValueId> = HashMap::default();
        for (pos, &i) in p.a_slots.iter().enumerate() {
            acc_subst.insert(ValueId::BlockParam(p.head_params[i]), acc_vals[pos]);
        }
        let new_acc: Vec<ValueId> = p
            .a_slots
            .iter()
            .map(|&i| clone_value(ctx, p.next_args[i], &mut acc_subst, &p.body_bindings, rec))
            .collect();

        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, rec));
        let tuple = b.push_tuple(new_acc).id();
        b.push_return_value(tuple);
    }

    // --- Host: seed g and project the original return value out of the tuple.
    let root = model.root;
    if let Some(term) = block_terminator(ctx, root) {
        ctx.remove_instruction(term);
    }
    let driver_init: Vec<ValueId> = p.d_slots.iter().map(|&i| model.init_args[i]).collect();
    let result = {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, root));
        unsafe { b.dont_finalize() };
        let t = b.push_apply(g, driver_init).id();
        let mut host_subst: HashMap<ValueId, ValueId> = HashMap::default();
        for (pos, &i) in p.a_slots.iter().enumerate() {
            let field = b.push_extract(t, pos).id();
            host_subst.insert(ValueId::BlockParam(p.head_params[i]), field);
        }
        // Drop the builder before the raw clone, then reconstruct `ret`.
        drop(b);
        clone_value(ctx, p.ret_val, &mut host_subst, &p.exit_bindings, root)
    };
    {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, root));
        b.push_return_value(result);
    }

    // The original loop region is now unreachable from the host; delete it.
    for &b in &model.region {
        BasicBlock::from_id_mut(ctx, b).delete(host);
    }
}

/// Head-param slot indices that `val` transitively reads, resolving block-param
/// references (latch/exit params) through `bindings`.
fn head_param_deps(
    ctx: &Context,
    val: ValueId,
    bindings: &HashMap<BlockParamId, ValueId>,
    head_index: &HashMap<BlockParamId, usize>,
) -> HashSet<usize> {
    let mut out = HashSet::default();
    let mut seen = HashSet::default();
    collect_deps(ctx, val, bindings, head_index, &mut out, &mut seen);
    out
}

fn collect_deps(
    ctx: &Context,
    val: ValueId,
    bindings: &HashMap<BlockParamId, ValueId>,
    head_index: &HashMap<BlockParamId, usize>,
    out: &mut HashSet<usize>,
    seen: &mut HashSet<ValueId>,
) {
    if !seen.insert(val) {
        return;
    }
    match val {
        ValueId::BlockParam(p) => {
            if let Some(&idx) = head_index.get(&p) {
                out.insert(idx);
            } else if let Some(&bound) = bindings.get(&p) {
                collect_deps(ctx, bound, bindings, head_index, out, seen);
            }
        }
        ValueId::Instruction(id) => {
            for op in Instruction::from_id(ctx, id).mnemonic().args() {
                collect_deps(ctx, op, bindings, head_index, out, seen);
            }
        }
        // Literals, byte blobs, varnodes, function/block refs read no state.
        _ => {}
    }
}

/// Rebuilds the expression DAG rooted at `val` into `target`, substituting
/// seeded leaves via `subst` and resolving block-param references via
/// `bindings`. Returns the value in the new graph. Memoizes into `subst`.
fn clone_value(
    ctx: &mut Context,
    val: ValueId,
    subst: &mut HashMap<ValueId, ValueId>,
    bindings: &HashMap<BlockParamId, ValueId>,
    target: BlockId,
) -> ValueId {
    if let Some(&v) = subst.get(&val) {
        return v;
    }
    let result = match val {
        ValueId::BlockParam(p) => match bindings.get(&p) {
            // Resolve a latch/exit param to the value its predecessor passed.
            Some(&bound) => clone_value(ctx, bound, subst, bindings, target),
            // Gates guarantee every reachable header param is seeded in `subst`.
            None => val,
        },
        ValueId::Instruction(id) => {
            let (mnemonic, type_id) = {
                let insn = Instruction::from_id(ctx, id);
                (insn.mnemonic().clone(), insn.type_id())
            };
            let mut remapped = mnemonic.clone();
            for op in mnemonic.args() {
                let new_op = clone_value(ctx, op, subst, bindings, target);
                if new_op != op {
                    remapped.replace_value(op, new_op);
                }
            }
            let new_id = InstructionRef::from_mnemonic_with_type(ctx, remapped, type_id).id;
            let idx = BasicBlock::from_id(ctx, target).instruction_ids().len();
            BasicBlock::from_id_mut(ctx, target).insert_insn_at_index(idx, new_id);
            ValueId::Instruction(new_id)
        }
        // Literals, byte blobs, varnodes, function/block refs are context-global.
        _ => val,
    };
    subst.insert(val, result);
    result
}

fn header_cbranch(ctx: &Context, header: BlockId) -> Option<CBranch> {
    let term = block_terminator(ctx, header)?;
    match Instruction::from_id(ctx, term).mnemonic() {
        Mnemonic::CBranch(cbranch) => Some(cbranch.clone()),
        _ => None,
    }
}

fn block_return_value(ctx: &Context, block: BlockId) -> Option<ValueId> {
    let term = block_terminator(ctx, block)?;
    match Instruction::from_id(ctx, term).mnemonic() {
        Mnemonic::ReturnValue(rv) => Some(rv.value),
        _ => None,
    }
}

fn block_terminator(ctx: &Context, block: BlockId) -> Option<qcode::value::insn::InstructionId> {
    let &id = BasicBlock::from_id(ctx, block).instruction_ids().last()?;
    Instruction::from_id(ctx, id).is_terminator().then_some(id)
}

fn is_const_literal(ctx: &Context, val: ValueId) -> bool {
    matches!(val, ValueId::Literal(id) if ctx.values.literals[id].symbolic.is_none())
}

/// Reinterprets a constant literal at `size` bytes, so a base-case accumulator
/// matches its slot width; non-literals pass through unchanged.
fn const_at_size(ctx: &mut Context, val: ValueId, size: usize) -> ValueId {
    if let ValueId::Literal(id) = val {
        let lit = ctx.values.literals[id].clone();
        if lit.symbolic.is_none() {
            return ctx.get_const(lit.value, size).id();
        }
    }
    val
}

fn ctx_unique<'str>(ctx: &mut Context<'str>, name: String) -> Cow<'str, str> {
    ctx.get_unique_name(Cow::Owned(name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode_emulator::{SizedValue, StandaloneEmulator};
    use qcode_macro::qcode;

    fn run(ctx: &Context, fun: FunctionId, n: u64) -> Option<u64> {
        let root = Function::from_id(ctx, fun).root().expect("root").id;
        let ret = match Instruction::from_id(ctx, block_terminator(ctx, root)?).mnemonic() {
            Mnemonic::ReturnValue(r) => r.value,
            _ => return None,
        };
        let mut emu = StandaloneEmulator::new(root);
        emu.run_pure(ctx, fun, &[SizedValue::new(n, 8)], 100_000)
            .expect("rewritten recursion runs");
        emu.get_value(ctx, ret)
    }

    #[test]
    fn eliminates_fibonacci_accumulators() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda fib_loop:
            <entry @input:i64>
                goto <head @hn=@input @a=0 @b=1>;

            <head @hn:i64 @a:i64 @b:i64>
                %done = @hn == 0;
                if %done goto <exit @r=@a> else goto <body @m=@hn @x=@a @y=@b>;

            <body @m:i64 @x:i64 @y:i64>
                %next = @x + @y;
                %m1 = @m - 1;
                goto <head @hn=%m1 @a=@y @b=%next>;

            <exit @r:i64>
                return @r;
            "
        );

        assert!(accumulator_elim(&mut ctx, fib_loop));

        let g = Function::from_name(&ctx, "fib_loop_acc").expect("accumulator lambda exists");
        assert!(g.is_lambda());
        // Driver-only interface: one parameter instead of three.
        let g_root = g.root().expect("g root");
        assert_eq!(g_root.params().count(), 1);

        // Numerically equivalent to the original loop.
        assert_eq!(run(&ctx, fib_loop, 0), Some(0));
        assert_eq!(run(&ctx, fib_loop, 1), Some(1));
        assert_eq!(run(&ctx, fib_loop, 6), Some(8));
        assert_eq!(run(&ctx, fib_loop, 10), Some(55));
    }

    #[test]
    fn leaves_driver_dependent_accumulator_loop_untouched() {
        // sum of 0..n: the `s += i` update reads the driver `i`, so the sound
        // classifier demotes `s` to a driver and there is nothing to eliminate.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda sum_up:
            <entry @input:i64>
                goto <head @i=0 @s=0 @bound=@input>;

            <head @i:i64 @s:i64 @bound:i64>
                %done = @i == @bound;
                if %done goto <exit @r=@s> else goto <body @j=@i @t=@s @bd=@bound>;

            <body @j:i64 @t:i64 @bd:i64>
                %s2 = @t + @j;
                %j1 = @j + 1;
                goto <head @i=%j1 @s=%s2 @bound=@bd>;

            <exit @r:i64>
                return @r;
            "
        );

        assert!(!accumulator_elim(&mut ctx, sum_up));
    }

    #[test]
    fn leaves_non_loop_lambda_untouched() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda straight:
            <entry @input:i64>
                %x = @input + 1;
                return %x;
            "
        );
        assert!(!accumulator_elim(&mut ctx, straight));
    }
}
