//! Recover counted/while loops as explicit tail-recursive lambdas.
//!
//! This is a purely *structural* transform: it does not interpret the loop's
//! arithmetic at all, so it works for any condition, any counter, any state
//! update, multiple carried values, and any return shape (a tuple counts as one
//! value). A natural loop in SSA already factors into three pieces:
//!
//! ```text
//! <entry @args...>            init : () -> S     (the branch into the header)
//!   ... goto <head @s0=...>
//! <head @s...>               latch : S -> bool  (the header's own terminator)
//!   if cond goto <exit ...> else goto <body @s...>
//! <body @s...>               iter  : S -> S      (the back-edge into the header)
//!   ... goto <head @s0=...>
//! <exit @r...>
//!   return @r
//! ```
//!
//! The header `<head>` together with everything reachable from it (the body
//! region and the exit/return blocks) is cloned into a fresh lambda whose root is
//! the cloned `<head>` and whose parameters are the loop-carried state `S`. Each
//! back-edge `goto <head @s0=next...>` becomes `%r = apply rec(next...); return
//! %r` in the clone, turning one loop iteration into one recursive call. The
//! original entry becomes `%r = apply rec(init...); return %r`, and the host copy
//! of the region is deleted.
//!
//! (As a checked-out function pass it *clones* rather than moves the region: a
//! minted lambda may not own blocks stored in the host's arena.) Because the
//! region is reproduced verbatim except for the back-edge rewrite, the transform
//! is correct by construction — there is no recurrence to derive and get wrong.

use std::borrow::Cow;
use std::collections::HashSet;

use rustc_hash::FxHashMap as HashMap;

use qcode::{
    builder::Builder,
    value::{
        FunctionId, FunctionKind, LocalValueId, QCodeView, ValueId, VarnodeId,
        block::BlockId,
        block_param::BlockParam,
        insn::{Branch, InstructionId, Mnemonic},
        util::{base_ref::BaseRef, host_mut::PassBacking},
    },
};

use crate::pipeline::{ContextView, FunctionBody, Minted, Outcome};
use crate::{FunctionPass, register_function_pass};

/// Simultaneously replace operands across arenas without allowing a newly
/// written local id to collide with a source local still awaiting replacement.
pub(crate) fn substitute_operands(mnemonic: &mut Mnemonic, pairs: &[(LocalValueId, LocalValueId)]) {
    let mut occupied: Vec<LocalValueId> = mnemonic
        .args()
        .into_iter()
        .chain(pairs.iter().map(|&(_, new)| new))
        .collect();
    let mut sentinels = Vec::with_capacity(pairs.len());
    let mut next = 0usize;
    for _ in pairs {
        let sentinel = loop {
            let candidate = LocalValueId::Varnode(VarnodeId::from(next));
            next += 1;
            if !occupied.contains(&candidate) {
                occupied.push(candidate);
                break candidate;
            }
        };
        sentinels.push(sentinel);
    }
    for (&(old, _), &sentinel) in pairs.iter().zip(&sentinels) {
        mnemonic.replace_value(old, sentinel);
    }
    for (&(_, new), &sentinel) in pairs.iter().zip(&sentinels) {
        mnemonic.replace_value(sentinel, new);
    }
}

#[derive(Default)]
pub struct LoopToRecursion;

impl FunctionPass for LoopToRecursion {
    const NAME: &'static str = "loop_to_recursion";

    fn description(&self) -> &'static str {
        "Recover counted loops as recursive lambda applications"
    }

    fn run<'str>(
        &self,
        f: &mut FunctionBody<'str>,
        m: ContextView<'_, 'str>,
        next_minted: &mut u32,
    ) -> Result<Outcome<'str>, String> {
        let mut minted = Vec::new();
        let changed = loop_to_recursion(m, f, next_minted, &mut minted);
        Ok(Outcome {
            changed,
            rename: None,
            minted,
        })
    }
}

register_function_pass!(LoopToRecursion);

#[derive(Debug, Clone)]
pub(crate) struct LoopModel {
    /// The host lambda's entry block (stays in the host, performs `init`).
    pub(crate) root: BlockId,
    /// The loop header; becomes the recursive lambda's root.
    pub(crate) head: BlockId,
    /// Arguments forwarded from `root` into `head` — the initial state `S`.
    pub(crate) init_args: Vec<ValueId>,
    /// `head` plus every block reachable from it; moved into the new lambda.
    pub(crate) region: Vec<BlockId>,
    /// Latch blocks: each ends in `goto head(next...)` and is rewritten into a
    /// recursive call. Paired with the state passed on the back-edge.
    pub(crate) back_edges: Vec<(BlockId, Vec<ValueId>)>,
}

pub fn loop_to_recursion<'str>(
    m: ContextView<'_, 'str>,
    body: &mut FunctionBody<'str>,
    next_minted: &mut u32,
    minted: &mut Vec<Minted<'str>>,
) -> bool {
    let Some(model) = recognize_loop(m.body_view(body), body.id()) else {
        return false;
    };
    transform(m, body, next_minted, minted, &model)
}

pub(crate) fn recognize_loop<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    fun_id: FunctionId,
) -> Option<LoopModel> {
    let fun = host.function_ref(fun_id);
    if !fun.is_lambda() {
        return None;
    }
    let root = fun.root()?.id;

    // The entry must unconditionally branch into the header, carrying `init`.
    let (head, init_args) = match terminator_mnemonic(host, root)? {
        // The entry's branch target is a body-local index in `root`'s arena.
        Mnemonic::Branch(Branch { target, args }) => (
            BlockId::new(root.func, *target),
            args.iter()
                .map(|a| a.qualify(root.func))
                .collect::<Vec<_>>(),
        ),
        _ => return None,
    };
    if head == root {
        return None;
    }
    let arity = block_param_count(host, head);
    if init_args.len() != arity {
        return None;
    }

    // The loop is the header and everything it dominates. Since `root`'s only
    // successor is `head`, the region is reachable solely through the header, so
    // moving it out cannot strand any code the entry still needs.
    let region = reachable_from(host, fun_id, head);
    if region.contains(&root) {
        return None;
    }

    // Every predecessor of the header is either the entry or a back-edge. Each
    // back-edge must be a clean latch (`goto head(next...)`) so we can rewrite
    // it into a recursive call without disturbing other control flow.
    let mut back_edges = Vec::new();
    for (_edge, pred) in host.block_ref(head).predecessors() {
        if pred == root {
            continue;
        }
        if !region.contains(&pred) {
            // An entry into the loop from outside the region we are extracting.
            return None;
        }
        match terminator_mnemonic(host, pred)? {
            Mnemonic::Branch(Branch { target, args })
                if BlockId::new(pred.func, *target) == head =>
            {
                if args.len() != arity {
                    return None;
                }
                back_edges.push((pred, args.iter().map(|a| a.qualify(pred.func)).collect()));
            }
            _ => return None,
        }
    }
    if back_edges.is_empty() {
        return None;
    }

    let mut region: Vec<BlockId> = region.into_iter().collect();
    region.sort();

    Some(LoopModel {
        root,
        head,
        init_args,
        region,
        back_edges,
    })
}

fn transform<'str>(
    m: ContextView<'_, 'str>,
    body: &mut FunctionBody<'str>,
    next_minted: &mut u32,
    minted_out: &mut Vec<Minted<'str>>,
    model: &LoopModel,
) -> bool {
    let host_fid = body.id();
    let name = format!("{}_rec", m.body_view(body).function_ref(host_fid).name());
    // Mint the recursive lambda (name buffered raw; the driver uniquifies it at
    // the barrier). Keep the placeholder in both recursive and host references;
    // the install barrier patches it to the materialized function id.
    let rec = crate::pipeline::mint_function(
        body,
        next_minted,
        minted_out,
        Cow::Owned(name),
        FunctionKind::Lambda,
        true,
    );

    // The set of back-edge latch blocks: their `goto head` terminator becomes an
    // `apply rec(next…); return` in the clone, so it is cloned specially.
    let latches: HashSet<BlockId> = model.back_edges.iter().map(|(p, _)| *p).collect();

    // --- Clone the region into the lambda (a checked-out pass may not *move*
    //     host blocks into another function — that would leave `rec` owning
    //     host-stored blocks, which the checked-out invariant forbids — so the
    //     region is reproduced, and the host copy deleted below).
    // TODO(5b-ii): function minting (`host_with_minted`) stays on the host path
    // until the minting chunk lands.
    {
        let (own, mut minted) = crate::pipeline::host_with_minted(body, minted_out, m, rec);

        // Pass 1: a fresh block per region block, with its params cloned. Names
        // and the head-as-root are set here so later passes can reference them.
        let mut block_map: HashMap<BlockId, BlockId> = HashMap::default();
        let mut value_map: HashMap<ValueId, ValueId> = HashMap::default();
        for &ob in &model.region {
            let nb = minted.make_block(host_fid);
            block_map.insert(ob, nb);
            if let Some(name) = own.block_ref(ob).name() {
                let _ =
                    BaseRef::new(minted.reborrow(), nb).rename_local(Cow::Owned(name.to_owned()));
            }
            let params: Vec<(
                qcode::value::block_param::BlockParamId,
                qcode::types::TypeId,
            )> = own
                .block_ref(ob)
                .params()
                .map(|p| {
                    let pid = match p.id() {
                        ValueId::BlockParam(pid) => pid,
                        _ => unreachable!("block params are BlockParam values"),
                    };
                    (pid, own.block_param(pid).type_id)
                })
                .collect();
            for (pid, ty) in params {
                let np = push_param(&mut minted, nb, ty);
                value_map.insert(ValueId::BlockParam(pid), np);
            }
        }
        minted
            .function_mut(host_fid)
            .set_root_id(Some(block_map[&model.head].local));

        // Pass 2: clone every non-terminator instruction (and the terminator of a
        // non-latch block), remapping block targets now (the map is complete) and
        // recording the value map so operands are remapped once all defs exist.
        let mut cloned: Vec<InstructionId> = Vec::new();
        for &ob in &model.region {
            let nb = block_map[&ob];
            let insns: Vec<InstructionId> = own.block_ref(ob).iter().map(|i| i.id).collect();
            let last = insns.last().copied();
            for iid in insns {
                let is_latch_term = latches.contains(&ob) && Some(iid) == last;
                if is_latch_term {
                    continue; // becomes `apply rec(next…); return` in pass 4
                }
                let r = own.insn_ref(iid);
                let mut mn = r.mnemonic().clone();
                let ty = r.type_id();
                remap_block_targets(&mut mn, ob.func, nb.func, &block_map);
                let new_id = minted.push_mnemonic_with_type(host_fid, mn, ty);
                BaseRef::new(minted.reborrow(), nb).push_insn(new_id);
                value_map.insert(ValueId::Instruction(iid), ValueId::Instruction(new_id));
                cloned.push(new_id);
            }
        }

        // Pass 3: remap value operands of the cloned instructions (region params
        // and defs), and add CFG edges for the cloned (non-latch) terminators.
        for &new_id in &cloned {
            let mut mn = minted.insn_ref(new_id).mnemonic().clone();
            let pairs: Vec<_> = mn
                .args()
                .into_iter()
                .filter_map(|a| {
                    // The clone still holds the source body's bare-local operands, so
                    // qualify with the host function for the map lookup and store the
                    // replacement local to the minted lambda's arena.
                    value_map
                        .get(&a.qualify(host_fid))
                        .map(|&n| (a, n.localize(new_id.func)))
                })
                .collect();
            if !pairs.is_empty() {
                substitute_operands(&mut mn, &pairs);
                // Keeps the minted function's reverse-use map in sync.
                minted.replace_instruction_mnemonic(new_id, mn);
            }
        }
        for &ob in &model.region {
            if latches.contains(&ob) {
                continue;
            }
            let nb = block_map[&ob];
            let succs: Vec<BlockId> = own.block_ref(ob).successors().map(|(_, s)| s).collect();
            for s in succs {
                if let Some(&ns) = block_map.get(&s) {
                    minted.add_cfg_edge(nb, ns);
                }
            }
        }

        // Pass 4: each latch's `goto head` becomes `%r = apply rec(next…); return %r`.
        for (pred, next_args) in &model.back_edges {
            let nb = block_map[pred];
            let args: Vec<ValueId> = next_args
                .iter()
                .map(|a| value_map.get(a).copied().unwrap_or(*a))
                .collect();
            let mut b = Builder::from_block(BaseRef::new(minted.reborrow(), nb));
            let out = b.push_apply(rec, args).id();
            b.push_return_value(out);
        }
    }

    // --- Host: the entry seeds the recursion and returns it; delete the region.
    if let Some(term) = terminator_id(m.body_view(body), model.root) {
        body.remove_instruction(term);
    }
    {
        // TODO(5b-ii): `Builder` drives a `BaseRef`, which is not mirrored on
        // `FunctionBody`; go through a temporary host.
        let mut host = m.host(body);
        let mut b = Builder::from_block(BaseRef::new(host.reborrow(), model.root));
        let out = b.push_apply(rec, model.init_args.clone()).id();
        b.push_return_value(out);
    }
    for &blk in &model.region {
        body.delete_block(blk);
    }
    true
}

/// Push a cloned param typed `ty` onto `block`, returning its value (host-routed
/// `BasicBlock::push_param` + the `type_id` write).
fn push_param<'str>(
    host: &mut PassBacking<'_, 'str>,
    block: BlockId,
    ty: qcode::types::TypeId,
) -> ValueId {
    let index = host.block_ref(block).num_params();
    let pid = host.push_block_param(block.func, BlockParam::new(index, ty, block.local));
    host.block_mut(block).params.push(pid.localize(block.func));
    ValueId::BlockParam(pid)
}

/// Rewrite the block targets of a cloned terminator through `block_map` (value
/// operands are remapped separately, once every region def is cloned). Targets are
/// bare body-local indices: a freshly cloned terminator still holds its source
/// block's local target (`old_func`-relative), so qualify with `old_func` for the
/// lookup and re-localize the mapped clone against its new arena `new_func`.
fn remap_block_targets(
    mn: &mut Mnemonic,
    old_func: qcode::value::FunctionId,
    new_func: qcode::value::FunctionId,
    block_map: &HashMap<BlockId, BlockId>,
) {
    let remap = |b: &mut qcode::value::LocalBlockId| {
        if let Some(&nb) = block_map.get(&BlockId::new(old_func, *b)) {
            *b = nb.localize(new_func);
        }
    };
    match mn {
        Mnemonic::Branch(br) => remap(&mut br.target),
        Mnemonic::CBranch(cb) => {
            remap(&mut cb.success_block);
            remap(&mut cb.failure_block);
        }
        _ => {}
    }
}

/// Blocks reachable from `start` within `fun_id`, following CFG successors.
fn reachable_from<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    fun_id: FunctionId,
    start: BlockId,
) -> HashSet<BlockId> {
    let mut seen = HashSet::new();
    let mut stack = vec![start];
    while let Some(block) = stack.pop() {
        if !seen.insert(block) {
            continue;
        }
        for (_edge, succ) in host.block_ref(block).successors() {
            let same_fun = host
                .block_ref(succ)
                .function()
                .is_some_and(|f| f.id == fun_id);
            if same_fun {
                stack.push(succ);
            }
        }
    }
    seen
}

fn block_param_count<'ctx, 'str: 'ctx>(host: impl QCodeView<'ctx, 'str>, block: BlockId) -> usize {
    host.block_ref(block).params().count()
}

/// The id of `block`'s terminator instruction, if it ends in one.
fn terminator_id<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    block: BlockId,
) -> Option<InstructionId> {
    let &id = host.block_ref(block).instruction_ids().last()?;
    host.insn_ref(id).is_terminator().then_some(id)
}

/// A borrow of `block`'s terminator mnemonic, avoiding a full clone.
fn terminator_mnemonic<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    block: BlockId,
) -> Option<&'ctx Mnemonic> {
    let id = terminator_id(host, block)?;
    Some(host.insn_ref(id).mnemonic())
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{context::Context, value::FunctionBody};
    use qcode_emulator::{SizedValue, StandaloneEmulator};
    use qcode_macro::qcode;

    use crate::test_util::run_function_pass;

    fn run(ctx: &Context, fun: FunctionId, n: u64) -> Option<u64> {
        let root = FunctionBody::from_id(ctx, fun).root().expect("root").id;
        let ret = match terminator_mnemonic(qcode::value::ModuleView::new(ctx), root)? {
            Mnemonic::ReturnValue(r) => r.value.qualify(root.func),
            _ => return None,
        };
        let mut emu = StandaloneEmulator::new(root);
        emu.run_pure(ctx, fun, &[SizedValue::new(n, 8)], 100_000)
            .expect("rewritten loop runs");
        emu.get_value(ctx, ret)
    }

    #[test]
    fn rewrites_iterative_fibonacci_loop_to_recursion() {
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

        assert!(run_function_pass::<LoopToRecursion>(&mut ctx, fib_loop).unwrap());

        // The host entry is now just `apply rec(init); return`.
        let fib = FunctionBody::from_id(&ctx, fib_loop);
        let root = fib.root().expect("root");
        let insns = root.instruction_ids();
        assert!(matches!(
            ctx.get_insn(*insns.last().unwrap()).mnemonic(),
            Mnemonic::ReturnValue(_)
        ));
        // The loop blocks were moved out of the host: only the entry remains.
        assert_eq!(fib.blocks().count(), 1);

        let rec = FunctionBody::from_name(&ctx, "fib_loop_rec").expect("recursive lambda exists");
        assert!(rec.is_lambda());
        assert!(rec.to_string().contains("apply fib_loop_rec"));

        assert_eq!(run(&ctx, fib_loop, 6), Some(8));
        assert_eq!(run(&ctx, fib_loop, 10), Some(55));
    }

    /// A *non-linear* recurrence (a running product) that the old linear-algebra
    /// recognizer could never have handled — the structural transform doesn't
    /// care about the arithmetic at all.
    #[test]
    fn rewrites_factorial_loop_to_recursion() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda fact:
            <entry @input:i64>
                goto <head @n=@input @acc=1>;

            <head @n:i64 @acc:i64>
                %done = @n == 0;
                if %done goto <exit @r=@acc> else goto <body @m=@n @a=@acc>;

            <body @m:i64 @a:i64>
                %na = @a * @m;
                %m1 = @m - 1;
                goto <head @n=%m1 @acc=%na>;

            <exit @r:i64>
                return @r;
            "
        );

        assert!(run_function_pass::<LoopToRecursion>(&mut ctx, fact).unwrap());
        assert!(
            FunctionBody::from_name(&ctx, "fact_rec")
                .unwrap()
                .is_lambda()
        );

        assert_eq!(run(&ctx, fact, 5), Some(120));
        assert_eq!(run(&ctx, fact, 0), Some(1));
        assert_eq!(run(&ctx, fact, 6), Some(720));
    }

    /// A loop that counts *up* and compares against a carried bound — a
    /// different counter and a different comparison from the Fibonacci shape.
    #[test]
    fn rewrites_count_up_sum_loop() {
        let mut ctx = Context::new();
        // sum of 0..input via an ascending counter compared to the bound.
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

        assert!(run_function_pass::<LoopToRecursion>(&mut ctx, sum_up).unwrap());
        // 0+1+2+3+4 = 10
        assert_eq!(run(&ctx, sum_up, 5), Some(10));
        assert_eq!(run(&ctx, sum_up, 1), Some(0));
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
        assert!(!run_function_pass::<LoopToRecursion>(&mut ctx, straight).unwrap());
    }
}
