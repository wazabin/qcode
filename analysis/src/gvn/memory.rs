//! Memory sub-pass: alias-aware store→load forwarding.
//!
//! A thin [`SubPass`] adapter over [`MemoryState`], which holds the byte-level
//! forwarding state. Unlike pure CSE state, forwarded memory does not flow
//! freely down the dominator tree: it is pruned at loop headers (a loop-body
//! store may overwrite it on a later iteration), after calls (clobbered
//! registers), and cleared entirely for blocks shared between walk entries.

use jstd::graph::analysis::DominatorTree;

use crate::AliasResult;
use qcode::value::{
    QCodeView, ValueId,
    block::BlockId,
    insn::{Mnemonic, Zext},
};

use super::affine::Numbering;
use std::any::Any;

use super::walk::{Claim, Editor, InsnCtx, SubPass};
use crate::memory_state::MemoryState;

use crate::{ContextView, FunctionBody};

/// Memory forwarding reasons across a whole function (loop-header pruning,
/// post-call clobbers). Every read goes through the selected function's static
/// view and every rebuild through the body's inherent verbs.
pub(super) struct MemoryForwarding;

/// The function-pass [`SubPass`] impl (body-local): reads route through
/// `cx.body_view(body)`, forwards through [`Editor::replace`], and the
/// [`MemoryState`] rebuild/record helpers mutate the checked-out body.
impl<'str> SubPass<'str> for MemoryForwarding {
    fn init_state(&self) -> Box<dyn Any> {
        Box::new(MemoryState::default())
    }

    fn clone_state(&self, state: &dyn Any) -> Box<dyn Any> {
        Box::new(
            state
                .downcast_ref::<MemoryState>()
                .expect("memory state")
                .clone(),
        )
    }

    fn on_block_entry(
        &self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        state: &mut dyn Any,
        block_id: BlockId,
        tree: &DominatorTree<BlockId>,
        aliases: Option<&AliasResult>,
        numbering: &Numbering,
        is_shared: bool,
    ) {
        let state = state.downcast_mut::<MemoryState>().expect("memory state");
        if is_shared {
            state.clear();
        }
        state.prune_join_paths(cx.body_view(body), block_id, tree, aliases, numbering);
        state.prune_loop_carried(cx.body_view(body), block_id, tree, aliases, numbering);
    }

    fn on_insn(
        &self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        state: &mut dyn Any,
        ic: &InsnCtx,
        ed: &mut Editor,
    ) -> Claim {
        let state = state.downcast_mut::<MemoryState>().expect("memory state");
        match ic.mnemonic {
            Mnemonic::Store(store) => {
                state.record_store(body, cx, ic.insn_id.func, store, ic.aliases, ic.numbering);
                Claim::Done
            }
            Mnemonic::Load(load) => {
                let forwarded = state.try_load(
                    body,
                    cx,
                    ic.block_id,
                    ic.insn_id,
                    load,
                    ic.aliases,
                    ic.numbering,
                );
                match forwarded {
                    Some(value) => {
                        let view = cx.body_view(body);
                        let load_ty = view.type_of(ic.id);
                        let value_ty = view.type_of(value);
                        let types = &view.shared().types;
                        let load_is_bool = types.is_bool(load_ty);
                        let value_is_bool = types.is_bool(value_ty);
                        let same_size = types.size_of(load_ty) == types.size_of(value_ty);

                        let value = if value_is_bool && !load_is_bool && same_size {
                            // A bool store forwarded through an ordinary i8 load
                            // is a semantic bool→integer conversion even though
                            // both occupy one byte. Preserve that boundary with
                            // an explicit cast instead of wiring the bool directly
                            // into the load's integer users.
                            ValueId::Instruction(ed.replace_with_new_insn_typed(
                                body,
                                cx,
                                ic.block_id,
                                ic.insn_id,
                                Mnemonic::Zext(Zext {
                                    src: value.localize(ic.insn_id.func),
                                    size: ic.size,
                                }),
                                load_ty,
                            ))
                        } else if value_is_bool != load_is_bool {
                            // Other bool/non-bool crossings require a cast that
                            // this forwarding path cannot synthesize. Keep the
                            // load as the memory value.
                            state.define_load(
                                cx.body_view(body),
                                ic.insn_id.func,
                                load,
                                ic.id,
                                ic.aliases,
                                ic.numbering,
                            );
                            return Claim::Done;
                        } else {
                            // Preserve historical forwarding among non-bool
                            // semantic types such as pointers, stack addresses,
                            // and same-width integers.
                            ed.replace(body, cx, ic.insn_id, value);
                            value
                        };
                        state.define_load(
                            cx.body_view(body),
                            ic.insn_id.func,
                            load,
                            value,
                            ic.aliases,
                            ic.numbering,
                        );
                    }
                    None => state.define_load(
                        cx.body_view(body),
                        ic.insn_id.func,
                        load,
                        ic.id,
                        ic.aliases,
                        ic.numbering,
                    ),
                }
                Claim::Done
            }
            _ => Claim::Pass,
        }
    }

    fn after_block(
        &self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        state: &mut dyn Any,
        block_id: BlockId,
        aliases: Option<&AliasResult>,
        _numbering: &Numbering,
    ) {
        let state = state.downcast_mut::<MemoryState>().expect("memory state");
        state.prune_clobbered_by_call(cx.body_view(body), block_id, aliases);
    }
}

#[cfg(test)]
mod tests {
    use crate::AliasResult;
    use crate::gvn::{constant_fold_function, gvn, gvn_function};
    use qcode::value::QCodeMut;
    use qcode::{
        context::Context,
        testing::TestContext,
        value::{
            BasicBlock, FunctionBody, ValueId,
            block::BlockId,
            function::FunctionId,
            insn::{Instruction, InstructionId, Mnemonic, Range},
        },
    };
    use wazabin_qcode_macro::qcode;

    #[test]
    fn test_gvn_function_does_not_forward_loads_across_loop_header() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
                varnode i32 A;

                fn loop_load:
                    <entry>
                        store(A:4, &A <- i32 0);
                        goto <header>;

                    <header>
                        %v = load(A:4, &A);
                        if i8 1 goto <body> else goto <exit>;

                    <body>
                        store(A:4, &A <- i32 1);
                        goto <header>;

                    <exit>
                        return at 0x1000;
                "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        gvn_function(&mut ctx, loop_load, Some(&aliases));

        assert!(
            BasicBlock::from_id(&ctx, header)
                .instruction_ids()
                .contains(&v),
            "header load must not be replaced by the entry store; the backedge may overwrite it"
        );
    }

    #[test]
    fn test_gvn_function_forwards_loads_across_loop_header_when_body_stores_do_not_alias() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
                varnode i32 A;
                varnode i32 B;

                fn loop_load:
                    <entry>
                        store(A:4, &A <- i32 7);
                        goto <header>;

                    <header>
                        %v = load(A:4, &A);
                        if i8 1 goto <body> else goto <exit>;

                    <body>
                        store(B:4, &B <- i32 1);
                        goto <header>;

                    <exit>
                        return at 0x1000;
                "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        gvn_function(&mut ctx, loop_load, Some(&aliases));

        assert!(
            !BasicBlock::from_id(&ctx, header)
                .instruction_ids()
                .contains(&v),
            "header load should be replaced by the dominating store when loop stores do not alias"
        );
    }

    /// A header that is its own loop body: the header's own store (after the load,
    /// before the back edge) clobbers the inherited value on iterations >= 2, so
    /// the entry store must not be forwarded into the header's load.
    #[test]
    fn test_gvn_function_self_loop_header_store_blocks_forwarding() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
                varnode i32 A;

                fn self_loop:
                    <entry>
                        store(A:4, &A <- i32 0);
                        goto <header>;

                    <header>
                        %v = load(A:4, &A);
                        store(A:4, &A <- i32 1);
                        if i8 1 goto <header> else goto <exit>;

                    <exit>
                        return at 0x1000;
                "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        gvn_function(&mut ctx, self_loop, Some(&aliases));

        assert!(
            BasicBlock::from_id(&ctx, header)
                .instruction_ids()
                .contains(&v),
            "header load must not be replaced by the entry store; the header's own store \
             overwrites it before the back edge"
        );
    }

    /// Build `caller`: store a value to a pinned stack slot `A`, call `callee`,
    /// then reload `A` (kept live by storing it to `B`). The call block falls
    /// through to the reload block. Returns `(caller_id, reload_block, load_v,
    /// store_to_A_ptr, call_insn)`.
    fn build_store_call_reload(
        mut ctx: &mut Context,
    ) -> (FunctionId, BlockId, InstructionId, ValueId, InstructionId) {
        qcode!(
            ctx,
            "
                varnode i32 A;
                varnode i32 B;

                fn caller:
                    <entry>
                        store(A:4, &A <- i32 7);
                        call <callee>;
                    <reload>
                        %v = load(A:4, &A);
                        store(B:4, &B <- %v);
                        return at 0x1000;
                "
        );

        // The macro does not wire a call's fallthrough edge, so connect the call
        // block to the reload block: the reload now runs after the call.
        ctx.add_cfg_edge(entry, reload);

        let (store_ptr, call_id) = {
            let mut store_ptr = None;
            let mut call_id = None;
            for insn in BasicBlock::from_id(ctx, entry).iter() {
                match insn.mnemonic() {
                    Mnemonic::Store(s) => store_ptr = Some(s.ptr.qualify(caller)),
                    Mnemonic::Call(_) => call_id = Some(insn.id),
                    _ => {}
                }
            }
            (store_ptr.expect("store to A"), call_id.expect("call"))
        };

        (caller, reload, v, store_ptr, call_id)
    }

    /// A call that may write through `&A` (the pointer escaped into the callee,
    /// recorded in the call's clobber set) must block forwarding the pre-call
    /// store of `A` to the post-call reload: the callee may have overwritten it.
    #[test]
    fn test_gvn_does_not_forward_pinned_stack_slot_across_call_that_may_write_it() {
        let mut ctx = Context::new();
        let (caller, reload, v, store_ptr, call_id) = build_store_call_reload(&mut ctx);

        // Record that `&A` escaped into the callee.
        if let Mnemonic::Call(call) = Instruction::from_id_mut(&mut ctx, call_id).mnemonic_mut() {
            call.clobbers.push(store_ptr.localize(call_id.func));
        } else {
            panic!("expected a call");
        }

        let aliases = AliasResult::simple_for_function(&ctx, caller);
        gvn_function(&mut ctx, caller, Some(&aliases));

        assert!(
            BasicBlock::from_id(&ctx, reload)
                .instruction_ids()
                .contains(&v),
            "the reload must not be forwarded: the call may write through &A"
        );
    }

    /// Control: an identical call with an empty clobber set does not endanger the
    /// stack slot, so the pre-call store still forwards to the reload.
    #[test]
    fn test_gvn_forwards_pinned_stack_slot_across_call_that_cannot_write_it() {
        let mut ctx = Context::new();
        let (caller, reload, v, _store_ptr, _call_id) = build_store_call_reload(&mut ctx);

        let aliases = AliasResult::simple_for_function(&ctx, caller);
        gvn_function(&mut ctx, caller, Some(&aliases));

        assert!(
            !BasicBlock::from_id(&ctx, reload)
                .instruction_ids()
                .contains(&v),
            "the reload should be forwarded: the call cannot write through &A"
        );
    }

    /// Constant propagation: store → load → fold chain collapses to a literal.
    #[test]
    fn test_constant_propagation() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
                varnode i64 A;
                varnode i64 B;
                <block>
                    store(A:8, &A <- i64 5);
                    %a = load(A:8, &A);
                    %v1 = %a + 2;
                    %v2 = %v1 + 3;
                    store(B:8, &B <- %v2);
                    goto <0x1001>;"
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);

        let mut block = BasicBlock::from_id_mut(&mut ctx, block);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));

        gvn(&mut block, Some(&aliases));

        assert!(!block.instruction_ids().contains(&v1));
        assert!(!block.instruction_ids().contains(&v2));
        assert!(block.to_string().contains("B <- i64 0xa"));
    }

    #[test]
    fn bool_store_forwarded_through_i8_load_inserts_cast() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i8 A;
                fn f:
                    <entry @a:i8>
                        %cond = @a == 0;
                        store(A:1, &A <- %cond);
                        %loaded = load(A:1, &A);
                        %masked = %loaded & i8 2;
                        return %masked;
            "
        );

        assert!(crate::verify::verify_bool_typing(&ctx).is_empty());
        let aliases = AliasResult::simple_for_function(&ctx, f);
        gvn_function(&mut ctx, f, Some(&aliases));

        assert!(
            crate::verify::verify_bool_typing(&ctx).is_empty(),
            "GVN memory forwarding must preserve bool-to-integer conversion"
        );
        assert!(!ctx.contains_instruction(loaded));
        assert!(FunctionBody::from_id(&ctx, f).blocks().any(|block| {
            block
                .iter()
                .any(|insn| matches!(insn.mnemonic(), Mnemonic::Zext(_)))
        }));
    }

    // -----------------------------------------------------------------------
    // Register store→load forwarding
    // -----------------------------------------------------------------------

    /// Storing to a register and reading it straight back must forward the
    /// stored value, even when overlapping sub-registers (r0/r0_lo32/...) put
    /// the location in a multi-member alias class. Mirrors the post-call
    /// `*[register]:4 EAX = v; %r = *[register]:4 EAX` reload chains the lifter
    /// emits across every fixture.
    #[test]
    fn test_register_store_load_forwarding() {
        let mut tc = TestContext::new();
        let eax = tc.r0_lo32;
        let other = tc.r1;

        qcode!(
            tc.ctx,
            "
                fn func:
                    <block>
                        store(register:4, {eax} <- i32 0x12345678); # EAX = c
                        %r = load(register:4, {eax});         # %r = EAX
                        store(register:4, {other} <- %r);            # use %r (keeps it live)
                        return at 0x1000;
                "
        );

        let aliases = AliasResult::simple_for_function(&tc.ctx, func);
        gvn_function(&mut tc.ctx, func, Some(&aliases));

        assert!(
            !block_contains(&tc.ctx, block, r),
            "register reload should be forwarded to the stored value, got:\n{}",
            BasicBlock::from_id(&tc.ctx, block)
        );
    }

    // -----------------------------------------------------------------------
    // Memory coalesce: rebuilding a wide value from partial writes
    // -----------------------------------------------------------------------

    /// Run const-fold + GVN to a fixpoint, as the real `Gvn` pass does.
    fn optimize(ctx: &mut Context, fun_id: FunctionId) {
        loop {
            let mut changed = constant_fold_function(ctx, fun_id);
            let aliases = AliasResult::simple_for_function(ctx, fun_id);
            changed |= gvn_function(ctx, fun_id, Some(&aliases));
            if !changed {
                break;
            }
        }
    }

    fn store_src(ctx: &Context, store: ValueId) -> ValueId {
        let ValueId::Instruction(id) = store else {
            panic!("expected an instruction value, got {store:?}");
        };
        match ctx.get_insn(id).mnemonic() {
            Mnemonic::Store(s) => s.src.qualify(id.func),
            other => panic!("expected a store, got {other:?}"),
        }
    }

    fn block_contains(ctx: &Context, block: BlockId, id: InstructionId) -> bool {
        BasicBlock::from_id(ctx, block)
            .instruction_ids()
            .contains(&id)
    }

    fn literal_value(ctx: &Context, v: ValueId) -> Option<u64> {
        match v {
            ValueId::Literal(lid) => Some(ctx.shared.values.literals[lid].value),
            _ => None,
        }
    }

    /// Helper: the store-to-r1 instruction in `block` (used to read its src).
    fn load_insn_use(tc: &TestContext, block: BlockId) -> ValueId {
        BasicBlock::from_id(&tc.ctx, block)
            .iter()
            .find(|i| {
                matches!(i.mnemonic(), Mnemonic::Store(s)
                    if matches!(s.ptr.qualify(block.func), ValueId::Varnode(v) if v == tc.r1))
            })
            .expect("store to r1")
            .id()
    }

    /// `xor eax,eax; setnz al; push eax`: the wide read of EAX must be rebuilt as
    /// `zext(%cc)` — low byte from the partial write, upper bytes from the zero.
    #[test]
    fn coalesce_motivating_idiom_symbolic_low_byte() {
        let mut tc = TestContext::new();
        let r0_byte0 = tc.r0_byte0;
        let r0_byte1 = tc.r0_byte1;
        let r0_lo32 = tc.r0_lo32;
        let r1 = tc.r1;

        qcode!(
            tc.ctx,
            "
                fn func:
                    <block>
                        # A symbolic 1-byte value (the `setnz al` result), read before the zero.
                        %cc = load(register:1, {r0_byte1});
                        store(register:4, {r0_lo32} <- i32 0); # xor eax, eax
                        store(register:1, {r0_byte0} <- %cc);  # setnz al
                        %load = load(register:4, {r0_lo32});
                        store(register:4, {r1} <- %load);      # push eax (use)
                        return at 0x1000;
                "
        );

        optimize(&mut tc.ctx, func);

        assert!(
            !block_contains(&tc.ctx, block, load),
            "the wide EAX read should be coalesced away:\n{}",
            BasicBlock::from_id(&tc.ctx, block)
        );

        // After folding, the rebuilt value is exactly `zext(%cc)`.
        let src = store_src(&tc.ctx, load_insn_use(&tc, block));
        let ValueId::Instruction(zid) = src else {
            panic!("expected the rebuilt value to be an instruction, got {src:?}");
        };
        match tc.ctx.get_insn(zid).mnemonic() {
            Mnemonic::Zext(z) => {
                assert_eq!(
                    z.src.qualify(zid.func),
                    ValueId::Instruction(cc),
                    "zext of the setnz byte"
                )
            }
            other => panic!("expected zext(%cc), got {other:?}"),
        }
    }

    /// Same idiom but the low byte is a constant: the whole read folds to a literal.
    #[test]
    fn coalesce_constant_idiom_folds_to_literal() {
        let mut tc = TestContext::new();
        let r0_byte0 = tc.r0_byte0;
        let r0_lo32 = tc.r0_lo32;
        let r1 = tc.r1;

        qcode!(
            tc.ctx,
            "
                fn func:
                    <block>
                        store(register:4, {r0_lo32} <- i32 0);
                        store(register:1, {r0_byte0} <- i8 1);
                        %v = load(register:4, {r0_lo32});
                        store(register:4, {r1} <- %v);
                        return at 0x1000;
                "
        );

        optimize(&mut tc.ctx, func);

        assert_eq!(
            literal_value(&tc.ctx, store_src(&tc.ctx, load_insn_use(&tc, block))),
            Some(1),
            "0x00000000 with low byte 1 folds to 1"
        );
    }

    /// Four distinct constant byte writes coalesce in little-endian order.
    #[test]
    fn coalesce_constant_bytes_are_little_endian() {
        let mut tc = TestContext::new();
        let r0_byte0 = tc.r0_byte0;
        let r0_byte1 = tc.r0_byte1;
        let r0_byte2 = tc.r0_byte2;
        let r0_byte3 = tc.r0_byte3;
        let r0_lo32 = tc.r0_lo32;
        let r1 = tc.r1;

        qcode!(
            tc.ctx,
            "
                fn func:
                    <block>
                        store(register:1, {r0_byte0} <- i8 0xAA);
                        store(register:1, {r0_byte1} <- i8 0xBB);
                        store(register:1, {r0_byte2} <- i8 0xCC);
                        store(register:1, {r0_byte3} <- i8 0xDD);
                        %v = load(register:4, {r0_lo32});
                        store(register:4, {r1} <- %v);
                        return at 0x1000;
                "
        );

        optimize(&mut tc.ctx, func);

        assert_eq!(
            literal_value(&tc.ctx, store_src(&tc.ctx, load_insn_use(&tc, block))),
            Some(0xDDCC_BBAA),
            "byte 0 is least significant"
        );
    }

    /// Two symbolic byte reads coalesce into a halfword built with `zext`/`<<`/`|`.
    #[test]
    fn coalesce_two_symbolic_bytes_into_halfword() {
        let mut tc = TestContext::new();
        let r0_byte0 = tc.r0_byte0;
        let r0_byte1 = tc.r0_byte1;
        let r0_byte2 = tc.r0_byte2;
        let r0_byte3 = tc.r0_byte3;
        let r0_lo16 = tc.r0_lo16;
        let r1 = tc.r1;

        qcode!(
            tc.ctx,
            "
                fn func:
                    <block>
                        # Symbolic sources from non-overlapping bytes (offsets 2 and 3).
                        %x = load(register:1, {r0_byte2});
                        %y = load(register:1, {r0_byte3});
                        store(register:1, {r0_byte0} <- %x);
                        store(register:1, {r0_byte1} <- %y);
                        %load = load(register:2, {r0_lo16});
                        store(register:2, {r1} <- %load);
                        return at 0x1000;
                "
        );

        optimize(&mut tc.ctx, func);

        assert!(
            !block_contains(&tc.ctx, block, load),
            "the halfword read should be coalesced:\n{}",
            BasicBlock::from_id(&tc.ctx, block)
        );
        // The result must not collapse to a constant (both pieces are symbolic).
        assert!(
            literal_value(&tc.ctx, store_src(&tc.ctx, load_insn_use(&tc, block))).is_none(),
            "symbolic coalesce must not fold to a literal"
        );
    }

    /// Four sub-byte reads then a wide read: the wide read coalesces all four.
    #[test]
    fn coalesce_four_byte_reads_into_word() {
        let mut tc = TestContext::new();
        let r0_byte0 = tc.r0_byte0;
        let r0_byte1 = tc.r0_byte1;
        let r0_byte2 = tc.r0_byte2;
        let r0_byte3 = tc.r0_byte3;
        let r0_lo32 = tc.r0_lo32;
        let r1 = tc.r1;

        qcode!(
            tc.ctx,
            "
                fn func:
                    <block>
                        load(register:1, {r0_byte0});
                        load(register:1, {r0_byte1});
                        load(register:1, {r0_byte2});
                        load(register:1, {r0_byte3});
                        %load = load(register:4, {r0_lo32});
                        store(register:4, {r1} <- %load);
                        return at 0x1000;
                "
        );

        optimize(&mut tc.ctx, func);

        assert!(
            !block_contains(&tc.ctx, block, load),
            "the word read should coalesce four byte reads:\n{}",
            BasicBlock::from_id(&tc.ctx, block)
        );
    }

    /// A wide store then a one-byte overwrite: the wide read keeps a `Range` of the
    /// wide store for the untouched upper bytes.
    #[test]
    fn coalesce_wide_store_with_one_byte_overwrite_uses_range() {
        let mut tc = TestContext::new();
        let r0_byte0 = tc.r0_byte0;
        let r0_lo32 = tc.r0_lo32;
        let r1 = tc.r1;

        qcode!(
            tc.ctx,
            "
                fn func:
                    <block>
                        %w = load(register:4, {r0_lo32});
                        store(register:4, {r0_lo32} <- %w); # re-store: %w is the live writer of bytes 1..4
                        store(register:1, {r0_byte0} <- i8 0xAB);
                        %v = load(register:4, {r0_lo32});
                        store(register:4, {r1} <- %v);
                        return at 0x1000;
                "
        );

        optimize(&mut tc.ctx, func);

        // A Range(%w, 1, 3) must have been materialized for the upper three bytes.
        let w = ValueId::Instruction(w);
        let has_upper_range = BasicBlock::from_id(&tc.ctx, block).iter().any(|i| {
            matches!(i.mnemonic(), Mnemonic::Range(Range { src, start: 1, size: 3 }) if src.qualify(i.id.func) == w)
        });
        assert!(
            has_upper_range,
            "expected Range(%w, 1, 3) for the untouched upper bytes:\n{}",
            BasicBlock::from_id(&tc.ctx, block)
        );
    }

    /// A wide write then a narrow read still works (the old `range_table` path):
    /// reading byte 1 of a wide value yields `Range(%w, 1, 1)`.
    #[test]
    fn narrow_read_of_wide_value_extracts_range() {
        let mut tc = TestContext::new();
        let r0_byte1 = tc.r0_byte1;
        let r0_lo32 = tc.r0_lo32;
        let r1 = tc.r1;

        qcode!(
            tc.ctx,
            "
                fn func:
                    <block>
                        %w = load(register:4, {r0_lo32});
                        store(register:4, {r0_lo32} <- %w);
                        %narrow = load(register:1, {r0_byte1});
                        store(register:1, {r1} <- %narrow);
                        return at 0x1000;
                "
        );

        optimize(&mut tc.ctx, func);

        assert!(
            !block_contains(&tc.ctx, block, narrow),
            "the narrow read should be replaced by a Range"
        );
        let w = ValueId::Instruction(w);
        let has_range = BasicBlock::from_id(&tc.ctx, block).iter().any(|i| {
            matches!(i.mnemonic(), Mnemonic::Range(Range { src, start: 1, size: 1 }) if src.qualify(i.id.func) == w)
        });
        assert!(has_range, "expected Range(%w, 1, 1)");
    }

    /// A gap in coverage (byte 1 never written) leaves the wide read intact.
    #[test]
    fn coalesce_bails_on_coverage_gap() {
        let mut tc = TestContext::new();
        let r0_byte0 = tc.r0_byte0;
        let r0_byte2 = tc.r0_byte2;
        let r0_lo32 = tc.r0_lo32;
        let r1 = tc.r1;

        qcode!(
            tc.ctx,
            "
                fn func:
                    <block>
                        store(register:1, {r0_byte0} <- i8 1);
                        store(register:1, {r0_byte2} <- i8 2); # byte 1 and 3 unwritten
                        %load = load(register:4, {r0_lo32});
                        store(register:4, {r1} <- %load);
                        return at 0x1000;
                "
        );

        optimize(&mut tc.ctx, func);

        assert!(
            block_contains(&tc.ctx, block, load),
            "a partially-covered read must not be forwarded"
        );
    }

    /// Partial writes in a dominating block forward into a dominated successor.
    #[test]
    fn coalesce_across_dominating_block() {
        let mut tc = TestContext::new();
        let r0_byte0 = tc.r0_byte0;
        let r0_byte1 = tc.r0_byte1;
        let r0_lo16 = tc.r0_lo16;
        let r1 = tc.r1;

        qcode!(
            tc.ctx,
            "
                fn func:
                    <entry>
                        store(register:1, {r0_byte0} <- i8 0xAA);
                        store(register:1, {r0_byte1} <- i8 0xBB);
                        goto <succ>;

                    <succ>
                        %load = load(register:2, {r0_lo16});
                        store(register:2, {r1} <- %load);
                        return at 0x2000;
                "
        );

        optimize(&mut tc.ctx, func);

        assert!(
            !block_contains(&tc.ctx, succ, load),
            "partial writes in the dominator should coalesce into the successor read:\n{}",
            BasicBlock::from_id(&tc.ctx, succ)
        );
        // Both bytes are constant, so the dominated read folds to 0xBBAA.
        let keep = load_insn_use(&tc, succ);
        assert_eq!(
            literal_value(&tc.ctx, store_src(&tc.ctx, keep)),
            Some(0xBBAA)
        );
    }

    /// With no alias oracle the byte map is inert: only exact opaque matches forward,
    /// so a coalesce-shaped read is left untouched.
    #[test]
    fn no_alias_oracle_disables_coalesce() {
        let mut tc = TestContext::new();

        let r0_byte0 = tc.r0_byte0;
        let r0_byte1 = tc.r0_byte1;
        let r0_lo16 = tc.r0_lo16;
        let r0 = tc.r1;

        qcode!(
            tc.ctx,
            "
                <block>
                    store(register:4, {r0_byte0} <- i32 0xAA);
                    store(register:4, {r0_byte1} <- i32 0xBB);

                    %load = load(register:4, {r0_lo16});
                    store(ram:8, %load <- {r0});
                    goto <0x1000>;"
        );

        let mut bb = BasicBlock::from_id_mut(&mut tc.ctx, block);
        gvn(&mut bb, None);

        assert!(
            block_contains(&tc.ctx, block, load),
            "without an alias oracle the coalesce read must remain"
        );
    }

    // -----------------------------------------------------------------------
    // Symbolic (affine base) RAM forwarding
    // -----------------------------------------------------------------------

    /// `store(p, 4); load(p + 1, 1)` forwards a `Range` of the store: `p` has no
    /// pinned interval, so it is tracked as a symbolic affine base and the inner
    /// byte is covered.
    #[test]
    fn symbolic_base_forwards_inner_byte() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i64 PB;
                varnode i32 OUT;

                fn f:
                    <entry>
                        %p = load(PB:8, &PB);
                        store(ram:4, %p <- i32 0x11223344);
                        %p1 = %p + i64 1;
                        %r = load(ram:1, %p1);
                        store(OUT:1, &OUT <- %r);
                        return at 0x1000;
                "
        );

        optimize(&mut ctx, f);

        assert!(
            !block_contains(&ctx, entry, r),
            "load(p+1, 1) must forward from store(p, 4) via the symbolic base:\n{}",
            BasicBlock::from_id(&ctx, entry)
        );
        // Byte 1 of 0x11223344 (little-endian) is 0x33.
        assert_eq!(
            literal_value(&ctx, store_src(&ctx, out_store(&ctx, entry))),
            Some(0x33)
        );
    }

    /// `store(p, 4); load(p + 3, 2)` reads past the store's last byte (offset 4
    /// is uncovered), so it is not forwarded.
    #[test]
    fn symbolic_base_coverage_gap_not_forwarded() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i64 PB;
                varnode i16 OUT;

                fn f:
                    <entry>
                        %p = load(PB:8, &PB);
                        store(ram:4, %p <- i32 0x11223344);
                        %p3 = %p + i64 3;
                        %r = load(ram:2, %p3);
                        store(OUT:2, &OUT <- %r);
                        return at 0x1000;
                "
        );

        optimize(&mut ctx, f);

        assert!(
            block_contains(&ctx, entry, r),
            "load(p+3, 2) spills past the 4-byte store and must not be forwarded"
        );
    }

    /// Two distinct symbolic bases with no may-alias proof: a store to `q` must
    /// conservatively drop the cells of `p`, so `load(p + 1, 1)` does not forward.
    #[test]
    fn distinct_symbolic_bases_not_forwarded() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i64 PB;
                varnode i64 QB;
                varnode i32 OUT;

                fn f:
                    <entry>
                        %p = load(PB:8, &PB);
                        %q = load(QB:8, &QB);
                        store(ram:4, %p <- i32 0x11223344);
                        store(ram:4, %q <- i32 0x55667788);
                        %p1 = %p + i64 1;
                        %r = load(ram:1, %p1);
                        store(OUT:1, &OUT <- %r);
                        return at 0x1000;
                "
        );

        optimize(&mut ctx, f);

        assert!(
            block_contains(&ctx, entry, r),
            "store(q) may alias p (no disjointness proof), so load(p+1) must remain:\n{}",
            BasicBlock::from_id(&ctx, entry)
        );
    }

    /// `(p + 4) - 4` canonicalizes to the same affine base as `p`, so a whole
    /// store forwards through the reassociated pointer.
    #[test]
    fn affine_base_reassociation_unifies_with_p() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i64 PB;
                varnode i32 OUT;

                fn f:
                    <entry>
                        %p = load(PB:8, &PB);
                        store(ram:4, %p <- i32 0x11223344);
                        %p4 = %p + i64 4;
                        %pm = %p4 - i64 4;
                        %r = load(ram:4, %pm);
                        store(OUT:4, &OUT <- %r);
                        return at 0x1000;
                "
        );

        optimize(&mut ctx, f);

        assert!(
            !block_contains(&ctx, entry, r),
            "load((p+4)-4, 4) must forward the whole store(p, 4):\n{}",
            BasicBlock::from_id(&ctx, entry)
        );
        assert_eq!(
            literal_value(&ctx, store_src(&ctx, out_store(&ctx, entry))),
            Some(0x1122_3344)
        );
    }

    /// The store-to-`&OUT` instruction in `block` (used to read the forwarded src).
    fn out_store(ctx: &Context, block: BlockId) -> ValueId {
        BasicBlock::from_id(ctx, block)
            .iter()
            .find(|i| {
                matches!(i.mnemonic(), Mnemonic::Store(s)
                if matches!(s.ptr.qualify(block.func), ValueId::Varnode(_)))
            })
            .expect("store to &OUT")
            .id()
    }
}
