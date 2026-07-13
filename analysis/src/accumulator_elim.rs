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
    types::TypeId,
    value::{
        FunctionId, FunctionKind, ValueId,
        block::BlockId,
        block_param::{BlockParam, BlockParamId},
        insn::{Apply, CBranch, Extract, Mnemonic},
        util::{
            base_ref::{BaseRef, HostRef},
            host_mut::PassBacking,
        },
    },
};

use crate::loop_to_recursion::{recognize_loop, substitute_operands};
use crate::pipeline::{ContextView, FunctionBody, Minted, Outcome};
use crate::{FunctionPass, register_function_pass};

#[derive(Default)]
pub struct AccumulatorElim;

impl FunctionPass for AccumulatorElim {
    const NAME: &'static str = "accumulator_elim";
    const MINTS: bool = true;

    fn description(&self) -> &'static str {
        "Eliminate loop-carried accumulators by returning them as a tuple"
    }

    fn run<'str>(
        &self,
        f: &mut FunctionBody<'_, 'str>,
        m: ContextView<'_, 'str>,
    ) -> Result<Outcome<'str>, String> {
        let mut minted = Vec::new();
        let changed = accumulator_elim(m, f, &mut minted);
        Ok(Outcome {
            changed,
            rename: None,
            minted,
        })
    }
}

register_function_pass!(AccumulatorElim);

pub fn accumulator_elim<'str>(
    m: ContextView<'_, 'str>,
    body: &mut FunctionBody<'_, 'str>,
    minted: &mut Vec<Minted<'str>>,
) -> bool {
    let host = body.id();
    let Some((model, plan)) = classify(body.read_host(m), host) else {
        return false;
    };
    transform(m, body, minted, &model, &plan);
    true
}

/// Recognize the accumulator-elimination shape in `host` and, if it fires, return
/// the loop model and the rewrite plan. Reads only — the mutation is
/// [`transform`]'s.
fn classify(
    host: HostRef,
    host_fid: FunctionId,
) -> Option<(crate::loop_to_recursion::LoopModel, Plan)> {
    let ctx = host;
    let model = recognize_loop(host, host_fid)?;
    // v1 handles the canonical single-latch counted loop.
    if model.back_edges.len() != 1 {
        return None;
    }
    let head = model.head;
    let (latch, next_args) = model.back_edges[0].clone();

    let cbranch = header_cbranch(ctx, head)?;
    // The loop *continues* through whichever header edge enters the latch
    // directly; the other edge exits. (Multi-block bodies are out of scope.)
    // The header CBranch's targets are body-local indices in the header's arena.
    let q = |t| BlockId::new(head.func, t);
    let qa = |args: &[qcode::value::LocalValueId]| -> Vec<ValueId> {
        args.iter().map(|a| a.qualify(head.func)).collect()
    };
    let (cont_args, exit_block, exit_args, cond_true_is_exit) = if q(cbranch.success_block) == latch
    {
        (
            qa(&cbranch.success_args),
            q(cbranch.failure_block),
            qa(&cbranch.failure_args),
            false,
        )
    } else if q(cbranch.failure_block) == latch {
        (
            qa(&cbranch.failure_args),
            q(cbranch.success_block),
            qa(&cbranch.success_args),
            true,
        )
    } else {
        return None;
    };

    // State slots = header parameters.
    let head_info: Vec<(BlockParamId, usize)> = ctx
        .block_ref(head)
        .params()
        .map(|p| (p.id, p.size()))
        .collect();
    let m = head_info.len();
    if m == 0 || next_args.len() != m || model.init_args.len() != m {
        return None;
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
    let latch_params: Vec<BlockParamId> = ctx.block_ref(latch).params().map(|p| p.id).collect();
    if latch_params.len() != cont_args.len() {
        return None;
    }
    let body_bindings: HashMap<BlockParamId, ValueId> = latch_params
        .iter()
        .copied()
        .zip(cont_args.iter().copied())
        .collect();

    // The exit returns a single value; we project it from the result tuple.
    let ret_val = block_return_value(ctx, exit_block)?;
    let exit_params: Vec<BlockParamId> = ctx.block_ref(exit_block).params().map(|p| p.id).collect();
    if exit_params.len() != exit_args.len() {
        return None;
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
    let cond_deps = head_param_deps(
        ctx,
        cbranch.condition.qualify(head.func),
        &body_bindings,
        &head_index,
    );

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
        return None; // nothing to eliminate
    }

    // Gate: the returned value must be a function of accumulators only.
    let ret_deps = head_param_deps(ctx, ret_val, &exit_bindings, &head_index);
    if ret_deps.iter().any(|&i| is_driver[i]) {
        return None;
    }
    // Gate: accumulator initial values become the base case, so must be constant.
    for &i in &a_slots {
        if !is_const_literal(ctx, model.init_args[i]) {
            return None;
        }
    }

    Some((
        model,
        Plan {
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
    ))
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

fn transform<'str>(
    m: ContextView<'_, 'str>,
    body: &mut FunctionBody<'_, 'str>,
    minted_out: &mut Vec<Minted<'str>>,
    model: &crate::loop_to_recursion::LoopModel,
    p: &Plan,
) {
    let host_fid = body.id();
    let base_name = format!("{}_acc", body.read_host(m).function_ref(host_fid).name());
    // Mint the driver-only recursive lambda (name buffered raw; the driver
    // uniquifies it at the barrier). `None` (pool exhausted) leaves the loop alone.
    let Some(g) = body.mint_function(
        minted_out,
        Cow::Owned(base_name.clone()),
        FunctionKind::Lambda,
        true,
    ) else {
        return;
    };

    // --- Build the lambda body: read the host expressions, write the minted one.
    let tuple_ty = {
        let (own, mut minted) = body.host_with_minted(minted_out, m, g);

        // Three fresh blocks: header (root, drivers in), base case, recursive case.
        let g_head = minted.make_block(g);
        let base = minted.make_block(g);
        let rec = minted.make_block(g);
        minted.function_mut(g).set_root_id(Some(g_head));
        let _ = BaseRef::new(minted.reborrow(), base)
            .rename_local(Cow::Owned(format!("{base_name}_base")));
        let _ = BaseRef::new(minted.reborrow(), rec)
            .rename_local(Cow::Owned(format!("{base_name}_rec")));

        // g's parameters: the drivers only. Map each driver header-param to its new
        // counterpart so cloned `cond`/`stepD` expressions read g's params.
        let mut driver_subst: HashMap<ValueId, ValueId> = HashMap::default();
        for &i in &p.d_slots {
            let ty = minted.shr().types.get_or_make_int(p.head_sizes[i]);
            let pid = push_param(&mut minted, g_head, ty);
            driver_subst.insert(ValueId::BlockParam(p.head_params[i]), pid);
        }

        // Header: recompute the loop condition over g's drivers, then branch.
        let cond = clone_cross(
            own,
            &mut minted,
            p.cbranch.condition.qualify(model.head.func),
            &mut driver_subst,
            &p.body_bindings,
            g_head,
        );
        {
            let mut b = Builder::from_block(BaseRef::new(minted.reborrow(), g_head));
            if p.cond_true_is_exit {
                b.push_cbranch(cond, base, rec);
            } else {
                b.push_cbranch(cond, rec, base);
            }
        }

        // Base case: return the initial accumulator tuple (constants). Its type is
        // the recursion's return type — reused for the (minted, uninstalled)
        // `apply g` nodes below and in the host, which `push_apply` cannot type.
        let tuple_ty = {
            let base_fields: Vec<ValueId> = p
                .a_slots
                .iter()
                .map(|&i| const_at_size(minted.shr(), model.init_args[i], p.head_sizes[i]))
                .collect();
            let mut b = Builder::from_block(BaseRef::new(minted.reborrow(), base));
            let tuple = b.push_tuple(base_fields);
            let ty = tuple.type_id();
            let tuple = tuple.id();
            b.push_return_value(tuple);
            ty
        };

        // Recursive case: recurse on the drivers' next values, unpack the deeper
        // accumulators, apply the accumulator update, return the new tuple.
        {
            // stepD: driver-next values (read only drivers).
            let driver_next: Vec<ValueId> = p
                .d_slots
                .iter()
                .map(|&i| {
                    clone_cross(
                        own,
                        &mut minted,
                        p.next_args[i],
                        &mut driver_subst,
                        &p.body_bindings,
                        rec,
                    )
                })
                .collect();

            // `apply g(driver_next)` — g is the minted (uninstalled) lambda, so
            // type the node explicitly with the tuple type; the extracts then read
            // that type off the arena.
            let deep = push_typed(
                &mut minted,
                rec,
                Mnemonic::Apply(Apply {
                    target: g,
                    args: driver_next.into_iter().map(|a| a.localize(g)).collect(),
                }),
                tuple_ty,
            );
            let acc_vals: Vec<ValueId> = {
                let mut b = Builder::from_block(BaseRef::new(minted.reborrow(), rec));
                (0..p.a_slots.len())
                    .map(|pos| b.push_extract(deep, pos).id())
                    .collect()
            };

            // stepA: accumulator-next, reading the unpacked accumulators.
            let mut acc_subst: HashMap<ValueId, ValueId> = HashMap::default();
            for (pos, &i) in p.a_slots.iter().enumerate() {
                acc_subst.insert(ValueId::BlockParam(p.head_params[i]), acc_vals[pos]);
            }
            let new_acc: Vec<ValueId> = p
                .a_slots
                .iter()
                .map(|&i| {
                    clone_cross(
                        own,
                        &mut minted,
                        p.next_args[i],
                        &mut acc_subst,
                        &p.body_bindings,
                        rec,
                    )
                })
                .collect();

            let mut b = Builder::from_block(BaseRef::new(minted.reborrow(), rec));
            let tuple = b.push_tuple(new_acc).id();
            b.push_return_value(tuple);
        }

        tuple_ty
    };

    // --- Host: seed g and project the original return value out of the tuple.
    let root = model.root;
    if let Some(term) = block_terminator(body.read_host(m), root) {
        body.remove_instruction(m, term);
    }
    let driver_init: Vec<ValueId> = p.d_slots.iter().map(|&i| model.init_args[i]).collect();
    // TODO(5b-ii): the seeding below runs on a temporary host because
    // `push_typed`/`clone_self` stay generic for the minted `PassBacking` path.
    let mut host = body.host(m);
    // `apply g(driver_init)` typed explicitly (g uninstalled), then unpack each
    // accumulator field the original return reads.
    let t = push_typed(
        &mut host,
        root,
        Mnemonic::Apply(Apply {
            target: g,
            args: driver_init
                .into_iter()
                .map(|a| a.localize(root.func))
                .collect(),
        }),
        tuple_ty,
    );
    let mut host_subst: HashMap<ValueId, ValueId> = HashMap::default();
    for (pos, &i) in p.a_slots.iter().enumerate() {
        let field_ty = m
            .shr()
            .types
            .field_type(tuple_ty, pos)
            .expect("accumulator tuple field");
        let field = push_typed(
            &mut host,
            root,
            Mnemonic::Extract(Extract {
                agg: t.localize(root.func),
                index: pos,
            }),
            field_ty,
        );
        host_subst.insert(ValueId::BlockParam(p.head_params[i]), field);
    }
    let result = clone_self(
        &mut host,
        p.ret_val,
        &mut host_subst,
        &p.exit_bindings,
        root,
    );
    {
        let mut b = Builder::from_block(BaseRef::new(host.reborrow(), root));
        b.push_return_value(result);
    }

    // The original loop region is now unreachable from the host; delete it.
    for &b in &model.region {
        body.delete_block(m, b);
    }
}

/// Push a driver param typed `ty` onto `block` in the minted host, returning its
/// value (host-routed `BasicBlock::push_param` + the `type_id` write).
fn push_param<'str>(host: &mut PassBacking<'_, 'str>, block: BlockId, ty: TypeId) -> ValueId {
    let index = host.block_ref(block).num_params();
    let pid = host.push_block_param(block.func, BlockParam::new(index, ty, block));
    host.block_mut(block).params.push(pid.localize(block.func));
    ValueId::BlockParam(pid)
}

/// Mint an instruction with an explicit result type and append it to `block`.
fn push_typed<'str>(
    host: &mut PassBacking<'_, 'str>,
    block: BlockId,
    mnemonic: Mnemonic,
    ty: TypeId,
) -> ValueId {
    let id = host.push_mnemonic_with_type(block.func, mnemonic, ty);
    BaseRef::new(host.reborrow(), block).push_insn(id);
    ValueId::Instruction(id)
}

/// Head-param slot indices that `val` transitively reads, resolving block-param
/// references (latch/exit params) through `bindings`.
fn head_param_deps(
    host: HostRef,
    val: ValueId,
    bindings: &HashMap<BlockParamId, ValueId>,
    head_index: &HashMap<BlockParamId, usize>,
) -> HashSet<usize> {
    let mut out = HashSet::default();
    let mut seen = HashSet::default();
    collect_deps(host, val, bindings, head_index, &mut out, &mut seen);
    out
}

fn collect_deps(
    host: HostRef,
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
                collect_deps(host, bound, bindings, head_index, out, seen);
            }
        }
        ValueId::Instruction(id) => {
            for op in host.insn_ref(id).operands() {
                collect_deps(host, op, bindings, head_index, out, seen);
            }
        }
        // Literals, byte blobs, varnodes, function/block refs read no state.
        _ => {}
    }
}

/// Rebuild the expression DAG rooted at `val` (read from `read`, the *host*
/// function) into `target` (a block of the *minted* lambda `write`), substituting
/// seeded leaves via `subst` and resolving block-param references via `bindings`.
fn clone_cross<'str>(
    read: HostRef<'_, 'str>,
    write: &mut PassBacking<'_, 'str>,
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
            Some(&bound) => clone_cross(read, write, bound, subst, bindings, target),
            None => val,
        },
        ValueId::Instruction(id) => {
            let (mnemonic, type_id) = {
                let r = read.insn_ref(id);
                (r.mnemonic().clone(), r.type_id())
            };
            let mut remapped = mnemonic.clone();
            let pairs: Vec<_> = mnemonic
                .args()
                .into_iter()
                .filter_map(|op| {
                    // The operand is bare-local in the *read* function's arena; the
                    // rebuilt value lives in the minted function's arena.
                    let q = op.qualify(id.func);
                    let new_op = clone_cross(read, write, q, subst, bindings, target);
                    (new_op != q).then(|| (op, new_op.localize(target.func)))
                })
                .collect();
            substitute_operands(&mut remapped, &pairs);
            push_typed(write, target, remapped, type_id)
        }
        _ => val,
    };
    subst.insert(val, result);
    result
}

/// Like [`clone_cross`] but source and target are the *same* (host) function;
/// reads route through the host's own read view.
fn clone_self<'str>(
    host: &mut PassBacking<'_, 'str>,
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
            Some(&bound) => clone_self(host, bound, subst, bindings, target),
            None => val,
        },
        ValueId::Instruction(id) => {
            let (mnemonic, type_id) = {
                let r = host.insn_ref(id);
                (r.mnemonic().clone(), r.type_id())
            };
            let mut remapped = mnemonic.clone();
            let pairs: Vec<_> = mnemonic
                .args()
                .into_iter()
                .filter_map(|op| {
                    let q = op.qualify(id.func);
                    let new_op = clone_self(host, q, subst, bindings, target);
                    (new_op != q).then(|| (op, new_op.localize(target.func)))
                })
                .collect();
            substitute_operands(&mut remapped, &pairs);
            push_typed(host, target, remapped, type_id)
        }
        _ => val,
    };
    subst.insert(val, result);
    result
}

fn header_cbranch(host: HostRef, header: BlockId) -> Option<CBranch> {
    let term = block_terminator(host, header)?;
    match host.insn_ref(term).mnemonic() {
        Mnemonic::CBranch(cbranch) => Some(cbranch.clone()),
        _ => None,
    }
}

fn block_return_value(host: HostRef, block: BlockId) -> Option<ValueId> {
    let term = block_terminator(host, block)?;
    match host.insn_ref(term).mnemonic() {
        Mnemonic::ReturnValue(rv) => Some(rv.value.qualify(term.func)),
        _ => None,
    }
}

fn block_terminator(host: HostRef, block: BlockId) -> Option<qcode::value::insn::InstructionId> {
    let &id = host.block_ref(block).instruction_ids().last()?;
    host.insn_ref(id).is_terminator().then_some(id)
}

fn is_const_literal(host: HostRef, val: ValueId) -> bool {
    matches!(val, ValueId::Literal(id) if host.shr().values.literals[id].symbolic.is_none())
}

/// Reinterprets a constant literal at `size` bytes, so a base-case accumulator
/// matches its slot width; non-literals pass through unchanged.
fn const_at_size(shared: &qcode::context::Shared, val: ValueId, size: usize) -> ValueId {
    if let ValueId::Literal(id) = val {
        let lit = shared.values.literals[id].clone();
        if lit.symbolic.is_none() {
            return shared.get_const(lit.value, size);
        }
    }
    val
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::context::Context;
    use qcode::value::{Function, Instruction};
    use qcode_emulator::{SizedValue, StandaloneEmulator};
    use qcode_macro::qcode;

    use crate::test_util::run_function_pass;

    fn run(ctx: &Context, fun: FunctionId, n: u64) -> Option<u64> {
        let root = Function::from_id(ctx, fun).root().expect("root").id;
        let term = block_terminator(HostRef::Module(ctx), root)?;
        let ret = match Instruction::from_id(ctx, term).mnemonic() {
            Mnemonic::ReturnValue(r) => r.value.qualify(term.func),
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

        assert!(run_function_pass::<AccumulatorElim>(&mut ctx, fib_loop).unwrap());

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

        assert!(!run_function_pass::<AccumulatorElim>(&mut ctx, sum_up).unwrap());
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
        assert!(!run_function_pass::<AccumulatorElim>(&mut ctx, straight).unwrap());
    }
}
