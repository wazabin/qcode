use rustc_hash::FxHashSet as HashSet;

use qcode::{
    context::Context,
    value::{BlockId, FunctionId, InstructionId, ValueId, insn::Mnemonic, util::base_ref::HostRef},
};

use crate::loop_unroll::replace_terminator_with_branch;
#[cfg(test)]
use crate::loop_unroll::replace_terminator_with_branch_generic;

/// This host's users of `v` (its owning function's reverse-use list), or `&[]`
/// for a shared value with no owning function. Mirrors [`Context::users`].
fn host_users<'a, 'str>(host: HostRef<'a, 'str>, v: ValueId) -> &'a [InstructionId] {
    match v.owning_function() {
        Some(f) => host.function(f).users_of(v),
        None => &[],
    }
}

/// Returns instructions in `block_id` that are pure and have no users.
pub fn dead_insns<'a, 'str: 'a>(
    host: impl Into<HostRef<'a, 'str>>,
    block_id: BlockId,
) -> HashSet<InstructionId> {
    let host = host.into();
    let mut dead = HashSet::default();
    let insn_ids: Vec<InstructionId> = host.block_ref(block_id).instruction_ids().to_vec();
    for id in insn_ids {
        let mnemonic = host.instruction(id).mnemonic();
        let side_effects = mnemonic.has_side_effects();

        if !side_effects && host_users(host, ValueId::Instruction(id)).is_empty() {
            dead.insert(id);
        }
    }
    dead
}

/// Removes dead pure instructions from `block_id` iteratively until fixed point,
/// updating the users reverse map after each round.
/// TODO(5b-ii): Takes Context; migrate to FunctionBody/ContextView when public API stabilizes.
pub fn remove_dead_insns(ctx: &mut Context, block_id: BlockId) -> bool {
    remove_dead_insns_module(ctx, block_id)
}

/// Generic host-based version of [`remove_dead_insns`]; see that function.
/// TODO(5b-ii): For backwards compatibility; prefer concrete version for new code.
pub fn remove_dead_insns_module<'str>(host: &mut Context<'str>, block_id: BlockId) -> bool {
    let mut changed = false;
    loop {
        let dead = dead_insns(host.read_host(), block_id);
        if dead.is_empty() {
            break;
        }

        changed = true;
        for id in &dead {
            host.remove_instruction(*id);
        }
    }

    let params_changed = remove_unused_no_pred_block_params_generic(host, block_id);
    changed || params_changed
}

/// Host-generic core of [`remove_dead_insns`]; see that function.
/// This is the concrete version for FunctionBody/ContextView (stage 5b).
pub fn remove_dead_insns_host<'a, 'str>(
    body: &'a mut FunctionBody<'_, 'str>,
    cx: ContextView<'a, 'str>,
    block_id: BlockId,
) -> bool {
    let mut changed = false;
    loop {
        let dead = dead_insns(body.read_host(cx), block_id);
        if dead.is_empty() {
            break;
        }

        changed = true;
        for id in &dead {
            body.remove_instruction(cx, *id);
        }
    }

    let params_changed = remove_unused_no_pred_block_params_host(body, cx, block_id);
    changed || params_changed
}

/// Removes a `Call` terminator whose callee is `is_pure`, whose return value
/// has no users, and whose clobber set is empty: such a call has no observable
/// effect and is dead.
///
/// A call is a terminator with a single CFG fall-through edge to its
/// continuation, so it cannot simply be deleted (that would tear down the edge
/// and orphan the continuation). Instead it is rewritten into an unconditional
/// `Branch` to that fall-through successor, preserving the CFG minus the call.
/// The fall-through has no params fed by the call, so the branch carries no
/// arguments.
#[cfg(test)]
pub fn remove_dead_pure_call(ctx: &mut Context, block_id: BlockId) -> bool {
    remove_dead_pure_call_module(ctx, block_id)
}

/// Generic version of remove_dead_pure_call for test use.
#[cfg(test)]
fn remove_dead_pure_call_module<'str>(host: &mut Context<'str>, block_id: BlockId) -> bool {
    let Some(term_id) = host.block_ref(block_id).instruction_ids().last().copied() else {
        return false;
    };

    let (clobbers_empty, target) = match host.insn_ref(term_id).mnemonic() {
        Mnemonic::Call(call) => (call.clobbers.is_empty(), call.target),
        _ => return false,
    };
    if !clobbers_empty {
        return false;
    }
    if !host.function_ref(target).is_pure() {
        return false;
    }
    if !host_users(host.read_host(), ValueId::Instruction(term_id)).is_empty() {
        return false;
    }

    // A pure call's block has exactly one successor: its fall-through.
    let successors: Vec<BlockId> = host
        .block_ref(block_id)
        .successors()
        .map(|(_, b)| b)
        .collect();
    debug_assert_eq!(
        successors.len(),
        1,
        "pure call block must have a single fall-through successor"
    );
    let Some(&fallthrough) = successors.first() else {
        return false;
    };

    replace_terminator_with_branch_generic(host, block_id, fallthrough, vec![]);
    true
}

/// Host-generic core of [`remove_dead_pure_call`]; see that function.
fn remove_dead_pure_call_host<'a, 'str>(
    body: &'a mut FunctionBody<'_, 'str>,
    cx: ContextView<'a, 'str>,
    block_id: BlockId,
) -> bool {
    let Some(term_id) = body
        .block_ref(cx, block_id)
        .instruction_ids()
        .last()
        .copied()
    else {
        return false;
    };

    let (clobbers_empty, target) = match body.insn_ref(cx, term_id).mnemonic() {
        Mnemonic::Call(call) => (call.clobbers.is_empty(), call.target),
        _ => return false,
    };
    if !clobbers_empty {
        return false;
    }
    if !body.function_ref(cx, target).is_pure() {
        return false;
    }
    if !host_users(body.read_host(cx), ValueId::Instruction(term_id)).is_empty() {
        return false;
    }

    // A pure call's block has exactly one successor: its fall-through.
    let successors: Vec<BlockId> = body
        .block_ref(cx, block_id)
        .successors()
        .map(|(_, b)| b)
        .collect();
    debug_assert_eq!(
        successors.len(),
        1,
        "pure call block must have a single fall-through successor"
    );
    let Some(&fallthrough) = successors.first() else {
        return false;
    };

    replace_terminator_with_branch(body, cx, block_id, fallthrough, vec![]);
    true
}

/// Removes block params that have no users when the block has no incoming
/// control-flow edges. This covers function-entry params introduced for
/// load-before-store registers that later become dead, without touching join
/// blocks whose predecessor terminators carry positional arguments.
/// The `&mut Context` version.
/// TODO(5b-ii): For backwards compatibility; prefer concrete version for new code.
pub fn remove_unused_no_pred_block_params_generic<'str>(
    host: &mut Context<'str>,
    block_id: BlockId,
) -> bool {
    if host.block_ref(block_id).predecessors().next().is_some() {
        return false;
    }

    // A `pure_reg` function's entry params are its canonical interface, aligned
    // index-for-index with `input_regs` and every caller's `Call.args`. Removing
    // one is an interprocedural change that must drop the param, its `input_regs`
    // entry, and the matching argument at every caller in lockstep — that is the
    // job of the `dead_signature` module pass (via `remove_entry_param`), not of
    // this per-function sweep. A function pass must not reach across functions, so
    // leave pure_reg entry params for `dead_signature`; the *local* fallback below
    // would silently drop the param and break the interface alignment.
    let is_pure_reg_entry = host
        .block_ref(block_id)
        .function()
        .is_some_and(|f| f.is_pure_reg() && f.root().map(|b| b.id) == Some(block_id));
    if is_pure_reg_entry {
        return false;
    }

    let params = host.read_host().block(block_id).params.clone();
    let mut kept = Vec::with_capacity(params.len());
    let mut changed = false;
    for param in params {
        if host_users(host.read_host(), ValueId::BlockParam(param)).is_empty()
            && !host.read_host().block_param(param).protected
        {
            host.block_param_mut(param).parent = None;
            changed = true;
        } else {
            host.block_param_mut(param).index = kept.len();
            kept.push(param);
        }
    }

    if changed {
        host.block_mut(block_id).params = kept;
    }
    changed
}

/// Removes block params that have no users when the block has no incoming
/// control-flow edges. This covers function-entry params introduced for
/// load-before-store registers that later become dead, without touching join
/// blocks whose predecessor terminators carry positional arguments.
/// Concrete version for FunctionBody/ContextView (stage 5b).
pub fn remove_unused_no_pred_block_params_host<'a, 'str>(
    body: &'a mut FunctionBody<'_, 'str>,
    cx: ContextView<'a, 'str>,
    block_id: BlockId,
) -> bool {
    if body.block_ref(cx, block_id).predecessors().next().is_some() {
        return false;
    }

    // A `pure_reg` function's entry params are its canonical interface, aligned
    // index-for-index with `input_regs` and every caller's `Call.args`. Removing
    // one is an interprocedural change that must drop the param, its `input_regs`
    // entry, and the matching argument at every caller in lockstep — that is the
    // job of the `dead_signature` module pass (via `remove_entry_param`), not of
    // this per-function sweep. A function pass must not reach across functions, so
    // leave pure_reg entry params for `dead_signature`; the *local* fallback below
    // would silently drop the param and break the interface alignment.
    let is_pure_reg_entry = body
        .block_ref(cx, block_id)
        .function()
        .is_some_and(|f| f.is_pure_reg() && f.root().map(|b| b.id) == Some(block_id));
    if is_pure_reg_entry {
        return false;
    }

    let params = body.read_host(cx).block(block_id).params.clone();
    let mut kept = Vec::with_capacity(params.len());
    let mut changed = false;
    for param in params {
        if host_users(body.read_host(cx), ValueId::BlockParam(param)).is_empty()
            && !body.read_host(cx).block_param(param).protected
        {
            body.block_param_mut(param).parent = None;
            changed = true;
        } else {
            body.block_param_mut(param).index = kept.len();
            kept.push(param);
        }
    }

    if changed {
        body.block_mut(block_id).params = kept;
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    use qcode::{
        builder::Builder,
        context::Context,
        space::SpaceId,
        testing::TestContext,
        value::{
            BasicBlock, BlockId, Function, ValueId,
            insn::{Mnemonic, PCodeOpId},
        },
    };

    fn reg_space(ctx: &Context) -> SpaceId {
        ctx.try_get_space("register").unwrap()
    }

    fn build_block(f: impl FnOnce(&mut Builder<'static, '_>)) -> (Context<'static>, BlockId) {
        let mut ctx = TestContext::new().ctx;
        let block_id = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0x1000, __f)
        };
        let mut builder = Builder::from_context(&mut ctx, 0x1000);
        f(&mut builder);
        unsafe { builder.dont_finalize() };
        drop(builder);
        (ctx, block_id)
    }

    #[test]
    fn test_pure_binop_no_users_removed() {
        let (mut ctx, block_id) = build_block(|b| {
            let a = b.context_mut().get_const(1u64, 8).id();
            let c = b.context_mut().get_const(2u64, 8).id();
            b.push_add(a, c);
        });

        remove_dead_insns(&mut ctx, block_id);
        assert!(
            BasicBlock::from_id(&ctx, block_id)
                .instruction_ids()
                .is_empty(),
            "dead binop should be removed"
        );
    }

    #[test]
    fn test_pure_binop_with_users_kept() {
        let (ctx, block_id) = build_block(|b| {
            let a = b.context_mut().get_const(1u64, 8).id();
            let c = b.context_mut().get_const(2u64, 8).id();
            let sum = b.push_add(a, c).id();
            let d = b.context_mut().get_const(3u64, 8).id();
            b.push_add(sum, d);
        });

        // The top-level add (no users) is dead, but the first add feeds it —
        // after top-level is removed, first add has no users and is then also removed.
        // This test checks the chain case; use dead_insns (single round) for the kept case.
        let dead = dead_insns(&ctx, block_id);
        let insn_ids: Vec<_> = BasicBlock::from_id(&ctx, block_id)
            .instruction_ids()
            .to_vec();
        // Only the outermost add (no users) is dead in the first round.
        assert_eq!(
            dead.len(),
            1,
            "only the unused result should be dead initially"
        );
        assert!(
            !dead.contains(&insn_ids[0]),
            "first add (whose result is used) must not be dead"
        );
        assert!(
            dead.contains(&insn_ids[1]),
            "second add (no users) should be dead"
        );
    }

    #[test]
    fn test_chain_both_removed() {
        // c = a + b, d = c * 2, neither used → both removed after iterating.
        let (mut ctx, block_id) = build_block(|b| {
            let a = b.context_mut().get_const(1u64, 8).id();
            let bv = b.context_mut().get_const(2u64, 8).id();
            let c = b.push_add(a, bv).id();
            let two = b.context_mut().get_const(2u64, 8).id();
            b.push_mul(c, two);
        });

        remove_dead_insns(&mut ctx, block_id);
        assert!(
            BasicBlock::from_id(&ctx, block_id)
                .instruction_ids()
                .is_empty(),
            "entire dead chain should be removed"
        );
    }

    #[test]
    fn test_store_kept() {
        let (mut ctx, block_id) = build_block(|b| {
            let rax_vn = b.context().get_named("r0").unwrap().as_varnode().unwrap();
            let rax_ptr = ValueId::Varnode(rax_vn);
            let space = reg_space(b.context());
            let v = b.context_mut().get_const(42u64, 8).id();
            b.push_store(v, rax_ptr, space);
        });

        let block = BasicBlock::from_id_mut(&mut ctx, block_id);
        let block_len = block.instruction_ids().len();

        remove_dead_insns(&mut ctx, block_id);

        let block = BasicBlock::from_id(&ctx, block_id);

        assert_eq!(
            block.instruction_ids().len(),
            block_len,
            "store must not be removed, block:\n{block}",
        );
    }

    #[test]
    fn test_pcode_op_kept() {
        let dead = {
            let (ctx, block_id) = build_block(|b| {
                let op_id: PCodeOpId = b.context_mut().shared.pcode_ops.push(Box::from("syscall"));
                b.push_pcode_op(op_id, vec![], None, 0);
            });
            dead_insns(&ctx, block_id)
        };
        assert!(dead.is_empty(), "PCodeOp must not be marked dead");
    }

    #[test]
    fn test_terminator_kept() {
        let dead = {
            let (ctx, block_id) = build_block(|b| {
                let zero = b.context_mut().get_const(0u64, 8).id();
                b.push_return(zero);
            });
            dead_insns(&ctx, block_id)
        };
        assert!(dead.is_empty(), "terminator must not be marked dead");
    }

    /// Build a callee (optionally marked `is_pure`) and a caller whose root
    /// block at `0x1000` ends in a `Call` to it, with a fall-through edge to a
    /// continuation block at `0x2000`. If `use_result`, the continuation
    /// `Extract`s the call's return value so it has a user. Returns the caller's
    /// call block and continuation block.
    fn build_call_case(
        ctx: &mut Context<'static>,
        pure: bool,
        use_result: bool,
    ) -> (BlockId, BlockId) {
        let callee = Function::make(ctx, "callee".into()).unwrap().id;
        let callee_block = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0x4000, __f)
        };
        Function::from_id_mut(ctx, callee)
            .set_root(callee_block)
            .unwrap();
        Function::from_id_mut(ctx, callee).set_is_pure(pure);

        // Both blocks must live in the *same* function: the call's result (defined
        // in the call block) is used by the store in the continuation, and users
        // of an SSA value are intra-function. Store both directly in `caller`.
        let caller = Function::make(ctx, "caller".into()).unwrap().id;
        let call_block = ctx.get_or_make_block(0x1000, caller);
        Function::from_id_mut(ctx, caller)
            .set_root(call_block)
            .unwrap();

        let call_id = {
            let mut b = Builder::from_context(ctx, 0x1000);
            let id = b.push_call(callee).id();
            unsafe { b.dont_finalize() };
            id
        };

        let cont = ctx.get_or_make_block(0x2000, caller);
        ctx.add_cfg_edge(call_block, cont);

        if use_result {
            // Give the call's return value a user so it is no longer dead.
            let reg = ctx.try_get_space("register").unwrap();
            let rax = ctx.get_named("r0").unwrap().as_varnode().unwrap();
            let mut b = Builder::from_context(ctx, 0x2000);
            b.push_store(call_id, ValueId::Varnode(rax), reg);
            unsafe { b.dont_finalize() };
        }

        (call_block, cont)
    }

    fn terminator(ctx: &Context, block_id: BlockId) -> Mnemonic {
        let id = *BasicBlock::from_id(ctx, block_id)
            .instruction_ids()
            .last()
            .unwrap();
        ctx.get_insn(id).mnemonic().clone()
    }

    #[test]
    fn unused_pure_call_rewritten_to_branch() {
        let mut ctx = TestContext::new().ctx;
        let (call_block, cont) = build_call_case(&mut ctx, true, false);

        assert!(remove_dead_pure_call(&mut ctx, call_block));

        match terminator(&ctx, call_block) {
            Mnemonic::Branch(br) => assert_eq!(br.target, cont, "branch must target fall-through"),
            other => panic!("expected branch, got {other:?}"),
        }
        // The continuation is still reachable via exactly its one predecessor.
        let preds: Vec<_> = BasicBlock::from_id(&ctx, cont).predecessors().collect();
        assert_eq!(
            preds.len(),
            1,
            "continuation must keep its single predecessor"
        );
        assert_eq!(preds[0].1, call_block);
    }

    #[test]
    fn used_pure_call_kept() {
        let mut ctx = TestContext::new().ctx;
        let (call_block, _) = build_call_case(&mut ctx, true, true);

        assert!(!remove_dead_pure_call(&mut ctx, call_block));
        assert!(matches!(terminator(&ctx, call_block), Mnemonic::Call(_)));
    }

    #[test]
    fn unused_impure_call_kept() {
        let mut ctx = TestContext::new().ctx;
        let (call_block, _) = build_call_case(&mut ctx, false, false);

        assert!(!remove_dead_pure_call(&mut ctx, call_block));
        assert!(matches!(terminator(&ctx, call_block), Mnemonic::Call(_)));
    }

    #[test]
    fn unused_entry_block_param_removed_after_dead_user_is_removed() {
        let (mut ctx, block_id) = build_block(|b| {
            let param = b.push_param(4).id();
            b.push_zext(param, 8);
        });

        assert_eq!(BasicBlock::from_id(&ctx, block_id).num_params(), 1);
        remove_dead_insns(&mut ctx, block_id);
        assert_eq!(
            BasicBlock::from_id(&ctx, block_id).num_params(),
            0,
            "unused entry block param should be pruned"
        );
    }

    /// The canonical dead counted loop `array_promote`/`loop_to_scan` leaves
    /// behind — a pure `iv == N` counter whose carried values are all internal —
    /// is bypassed by rerouting its preheader straight to the exit.
    #[test]
    fn dead_counted_loop_is_bypassed() {
        use qcode_macro::qcode;
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
            <pre @n:i64>
                goto <head @i=0x1 @acc=@n>;
            <head @i:i64 @acc:i64>
                %done = i64 @i == i64 0x5;
                if %done goto <exit> else goto <body>;
            <body>
                %next = i64 @acc + i64 @i;
                %j1 = i64 @i + i64 0x1;
                goto <head @i=%j1 @acc=%next>;
            <exit>
                goto <0x2000>;
            "
        );

        assert!(
            super::remove_dead_counted_loop(&mut ctx, f),
            "the pure, terminating, live-out-free loop should be recognized"
        );

        // The preheader now jumps straight to the exit, orphaning head/body.
        let term = BasicBlock::from_id(&ctx, pre).iter().last().unwrap();
        let Mnemonic::Branch(br) = term.mnemonic() else {
            panic!("preheader should end in an unconditional branch to the exit");
        };
        assert_eq!(br.target, exit);
        assert!(
            BasicBlock::from_id(&ctx, head).predecessors().count() == 1,
            "header keeps only its now-unreachable back-edge"
        );

        // A second call is a no-op: the loop is no longer reachable/matchable.
        assert!(!super::remove_dead_counted_loop(&mut ctx, f));
    }

    /// A loop with a memory side effect must not be removed even if its values
    /// are otherwise unused.
    #[test]
    fn loop_with_side_effect_is_kept() {
        use qcode_macro::qcode;
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
            <pre @n:i64 @p:i64>
                goto <head @i=0x1>;
            <head @i:i64>
                %done = i64 @i == i64 0x5;
                if %done goto <exit> else goto <body>;
            <body>
                store(ram:8, i64 @p <- i64 @i);
                %j1 = i64 @i + i64 0x1;
                goto <head @i=%j1>;
            <exit>
                goto <0x2000>;
            "
        );

        assert!(
            !super::remove_dead_counted_loop(&mut ctx, f),
            "a loop that stores to memory is observable and must be kept"
        );
    }
}

// ----- dead counted loop -----------------------------------------------------

use qcode::value::insn::{Binary, Binop, IntBinop};

/// A recognized dead, provably-terminating counted loop (see
/// [`remove_dead_counted_loop`]).
struct DeadLoop {
    /// The loop's sole preheader block, ending in `goto header(inits…)`.
    preheader: BlockId,
    /// The exit block to reroute the preheader to.
    exit: BlockId,
}

/// `c` if `v` is the integer literal `c`, else `None`.
fn dl_literal(host: HostRef, v: ValueId) -> Option<u64> {
    match v {
        ValueId::Literal(lid) => Some(host.shr().values.literals[lid].value),
        _ => None,
    }
}

/// `true` if `v` is `iv + 1` (either operand order).
fn dl_is_unit_inc(host: HostRef, v: ValueId, iv: ValueId) -> bool {
    let ValueId::Instruction(id) = v else {
        return false;
    };
    let Mnemonic::Binop(Binary { lhs, rhs, op }) = host.instruction(id).mnemonic() else {
        return false;
    };
    let one = |x: ValueId| dl_literal(host, x) == Some(1);
    matches!(op, Binop::Int(IntBinop::Add))
        && ((*lhs == iv && one(*rhs)) || (*rhs == iv && one(*lhs)))
}

/// `true` if `cond` is `iv == <literal>` (either operand order).
fn dl_is_eq_const(host: HostRef, cond: ValueId, iv: ValueId) -> bool {
    let ValueId::Instruction(id) = cond else {
        return false;
    };
    let Mnemonic::Binop(Binary { lhs, rhs, op }) = host.instruction(id).mnemonic() else {
        return false;
    };
    matches!(op, Binop::Int(IntBinop::Equal))
        && ((*lhs == iv && dl_literal(host, *rhs).is_some())
            || (*rhs == iv && dl_literal(host, *lhs).is_some()))
}

/// Distinct predecessor blocks of `b`.
fn dl_preds(host: HostRef, b: BlockId) -> Vec<BlockId> {
    let mut seen = HashSet::default();
    host.block_ref(b)
        .predecessors()
        .map(|(_, p)| p)
        .filter(|&p| seen.insert(p))
        .collect()
}

/// Terminator instruction of `b`, if any.
fn dl_term(host: HostRef, b: BlockId) -> Option<InstructionId> {
    host.block_ref(b).instruction_ids().last().copied()
}

/// `true` if every user of `v` lives in one of `region`'s blocks.
fn dl_users_confined(host: HostRef, v: ValueId, region: &[BlockId]) -> bool {
    host_users(host, v).iter().all(|&u| {
        qcode::value::InstructionRef::new(host, u)
            .parent()
            .is_some_and(|p| region.contains(&p.id))
    })
}

/// Matches the canonical dead counted loop headed by `header`:
///
/// ```text
///   P:  goto H(inits…)
///   H(…iv…):  if iv == N goto E else goto B(…)     // exit on the success arm
///   B:  … (pure) … goto H(…iv+1…)
///   E:  …
/// ```
///
/// with region `{H, B}` side-effect-free, the exit edge `H→E` carrying no
/// arguments, and no value defined in the region used outside it.
fn match_dead_loop(host: HostRef, header: BlockId) -> Option<DeadLoop> {
    let term = dl_term(host, header)?;
    let Mnemonic::CBranch(cb) = host.instruction(term).mnemonic() else {
        return None;
    };
    let (condition, exit, body) = (cb.condition, cb.success_block, cb.failure_block);
    // The exit arm must carry no arguments and land on a param-less block, so the
    // rerouted preheader→exit branch stays well-formed and nothing escapes.
    if !cb.success_args.is_empty() {
        return None;
    }
    if exit == header || body == header || exit == body {
        return None;
    }
    if !host.block(exit).params.is_empty() {
        return None;
    }

    // Body: a single unconditional back-edge to the header, reached only from it.
    let bterm = dl_term(host, body)?;
    let Mnemonic::Branch(bbr) = host.instruction(bterm).mnemonic() else {
        return None;
    };
    if bbr.target != header {
        return None;
    }
    let back_args = bbr.args.clone();
    let bpreds = dl_preds(host, body);
    if bpreds != [header] {
        return None;
    }

    // Header preds: exactly the back-edge plus one external preheader.
    let hpreds = dl_preds(host, header);
    if hpreds.len() != 2 || !hpreds.contains(&body) {
        return None;
    }
    let preheader = *hpreds.iter().find(|&&p| p != body)?;
    let pterm = dl_term(host, preheader)?;
    let Mnemonic::Branch(pbr) = host.instruction(pterm).mnemonic() else {
        return None;
    };
    if pbr.target != header {
        return None;
    }
    let init_args = pbr.args.clone();

    // The region's non-terminator instructions must all be pure.
    let region = [header, body];
    for &blk in &region {
        for id in host.block_ref(blk).instruction_ids().to_vec() {
            let m = host.instruction(id).mnemonic();
            if !m.is_terminator() && m.has_side_effects() {
                return None;
            }
        }
    }

    // No live-out: every region-defined value (instruction results and block
    // params) is used only within the region.
    for &blk in &region {
        for id in host.block_ref(blk).instruction_ids().to_vec() {
            if !dl_users_confined(host, ValueId::Instruction(id), &region) {
                return None;
            }
        }
        for &pid in &host.block(blk).params {
            if !dl_users_confined(host, ValueId::BlockParam(pid), &region) {
                return None;
            }
        }
    }

    // Termination: some header param is a unit-stride induction variable that
    // starts at a literal and drives the `iv == N` exit test, so the loop always
    // reaches the exit within `2^width` iterations.
    let hparams = host.block(header).params.clone();
    let counted = hparams.iter().enumerate().any(|(k, &pid)| {
        let iv = ValueId::BlockParam(pid);
        back_args
            .get(k)
            .is_some_and(|&be| dl_is_unit_inc(host, be, iv))
            && init_args
                .get(k)
                .is_some_and(|&ini| dl_literal(host, ini).is_some())
            && dl_is_eq_const(host, condition, iv)
    });
    if !counted {
        return None;
    }

    Some(DeadLoop { preheader, exit })
}

/// Removes one dead, provably-terminating counted loop from `fun_id` by
/// rerouting its preheader straight to the loop exit, leaving the loop region
/// unreachable for `simplify_cfg` to prune. Returns `true` if one was removed.
///
/// This is the region-level counterpart to the local pure-instruction sweep: a
/// loop-carried value is never locally dead (its back-edge is a self-use), so a
/// dead loop can only be recognized by reasoning over the whole cyclic region.
#[cfg(test)]
fn remove_dead_counted_loop(ctx: &mut Context, fun_id: FunctionId) -> bool {
    remove_dead_counted_loop_module(ctx, fun_id)
}

/// Generic version of remove_dead_counted_loop for test use.
#[cfg(test)]
fn remove_dead_counted_loop_module<'str>(host: &mut Context<'str>, fun_id: FunctionId) -> bool {
    let headers: Vec<BlockId> = host.function_ref(fun_id).blocks().map(|b| b.id).collect();
    for header in headers {
        if let Some(dl) = match_dead_loop(host.read_host(), header) {
            replace_terminator_with_branch_generic(host, dl.preheader, dl.exit, vec![]);
            return true;
        }
    }
    false
}

/// Host-generic core of [`remove_dead_counted_loop`]; see that function.
fn remove_dead_counted_loop_host<'a, 'str>(
    body: &'a mut FunctionBody<'_, 'str>,
    cx: ContextView<'a, 'str>,
    fun_id: FunctionId,
) -> bool {
    let headers: Vec<BlockId> = body
        .function_ref(cx, fun_id)
        .blocks()
        .map(|b| b.id)
        .collect();
    for header in headers {
        if let Some(dl) = match_dead_loop(body.read_host(cx), header) {
            replace_terminator_with_branch(body, cx, dl.preheader, dl.exit, vec![]);
            return true;
        }
    }
    false
}

// ----- pass ------------------------------------------------------------------

use crate::{ContextView, FunctionBody, FunctionPass};

#[derive(Default)]
pub struct Dce;

impl FunctionPass for Dce {
    const NAME: &'static str = "dce";
    fn description(&self) -> &'static str {
        "Remove unused pure instructions"
    }
    fn run<'str>(
        &self,
        f: &mut FunctionBody<'_, 'str>,
        m: ContextView<'_, 'str>,
    ) -> Result<bool, String> {
        let fun_id = f.id();
        Ok(dce_core(f, m, fun_id))
    }
}

/// Host-generic core of the [`Dce`] pass: the fixpoint over per-block dead-pure-call
/// / dead-instruction sweeps, redundant/dead block-argument elimination, and dead
/// counted-loop removal.
fn dce_core<'a, 'str>(
    body: &'a mut FunctionBody<'_, 'str>,
    cx: ContextView<'a, 'str>,
    fun_id: FunctionId,
) -> bool {
    let root = body.function_ref(cx, fun_id).root().map(|b| b.id);
    let block_ids: Vec<_> = body
        .function_ref(cx, fun_id)
        .blocks()
        .map(|b| b.id)
        .collect();
    let mut changed = false;
    // Loop to fixed point: rewriting a dead pure call into a branch can make
    // its argument-producing instructions (Extracts, etc.) unused, which the
    // dead-instruction sweep then removes, and so on. Dropping a useless block
    // argument likewise strips the predecessor's arg-producing value, which
    // can then become dead — so the block-arg sweep joins the same fixpoint.
    loop {
        let mut round = false;
        for &block_id in &block_ids {
            round |= remove_dead_pure_call_host(body, cx, block_id);
            round |= remove_dead_insns_host(body, cx, block_id);
        }
        round |= super::remove_dead_block_args_host(body, cx, &block_ids, root);
        round |= super::remove_dead_block_params_host(body, cx, &block_ids, root);
        round |= remove_dead_counted_loop_host(body, cx, fun_id);
        if !round {
            break;
        }
        changed = true;
    }
    changed
}

crate::register_function_pass!(Dce);
