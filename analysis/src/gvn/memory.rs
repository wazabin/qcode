//! Memory sub-pass: alias-aware store→load forwarding.
//!
//! A thin [`SubPass`] adapter over [`MemForward`], which holds the byte-level
//! forwarding state. Unlike pure CSE state, forwarded memory does not flow
//! freely down the dominator tree: it is pruned at loop headers (a loop-body
//! store may overwrite it on a later iteration), after calls (clobbered
//! registers), and cleared entirely for blocks shared between walk entries.

use jstd::graph::analysis::DominatorTree;

use crate::AliasResult;
use qcode::{
    context::Context,
    value::{block::BlockId, insn::Mnemonic},
};

use super::mem_forward::MemForward;
use super::walk::{Claim, Editor, InsnCtx, SubPass};

pub(super) struct MemoryForwarding;

impl SubPass for MemoryForwarding {
    type State = MemForward;

    fn on_block_entry(
        &self,
        ctx: &mut Context,
        state: &mut MemForward,
        block_id: BlockId,
        tree: &DominatorTree<BlockId>,
        aliases: Option<&AliasResult>,
        is_shared: bool,
    ) {
        if is_shared {
            state.clear();
        }
        state.prune_loop_carried(ctx, block_id, tree, aliases);
    }

    fn on_insn(
        &self,
        ctx: &mut Context,
        state: &mut MemForward,
        ic: &InsnCtx,
        ed: &mut Editor,
    ) -> Claim {
        match ic.mnemonic {
            Mnemonic::Store(store) => {
                state.record_store(ctx, store, ic.aliases);
                Claim::Done
            }
            Mnemonic::Load(load) => {
                match state.try_load(ctx, ic.block_id, ic.insn_id, load, ic.aliases) {
                    Some(value) => {
                        ed.replace(ctx, ic.insn_id, value);
                        state.define_load(load, value, ic.aliases);
                    }
                    None => state.define_load(load, ic.id, ic.aliases),
                }
                Claim::Done
            }
            _ => Claim::Pass,
        }
    }

    // A block that ends in a call clobbers registers: its dominated children
    // run after the call, so register values the call clobbers must not be
    // forwarded into them (e.g. a caller's post-call `RAX` read is the
    // callee's result, not a value computed before the call).
    fn after_block(
        &self,
        ctx: &Context,
        state: &mut MemForward,
        block_id: BlockId,
        aliases: Option<&AliasResult>,
    ) {
        state.prune_clobbered_by_call(ctx, block_id, aliases);
    }
}

#[cfg(test)]
mod tests {
    use crate::AliasResult;
    use crate::gvn::{constant_fold_function, gvn, gvn_function};
    use qcode::builder::Builder;
    use qcode::value::{Function, Value};
    use qcode::{
        context::Context,
        testing::TestContext,
        value::{
            BasicBlock, ValueId,
            block::BlockId,
            function::FunctionId,
            insn::{InstructionId, Mnemonic, Range},
        },
    };
    use qcode_macro::qcode;

    #[test]
    fn test_gvn_function_does_not_forward_loads_across_loop_header() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
                varnode i32 A;

                fn loop_load:
                    <entry>
                        store(&A, i32 0);
                        goto <header>;

                    <header>
                        %v = load(i32, &A);
                        if i8 1 goto <body> else goto <exit>;

                    <body>
                        store(&A, i32 1);
                        goto <header>;

                    <exit>
                        return [0x1000];
                "
        );

        let aliases = AliasResult::simple(&ctx);
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
                        store(&A, i32 7);
                        goto <header>;

                    <header>
                        %v = load(i32, &A);
                        if i8 1 goto <body> else goto <exit>;

                    <body>
                        store(&B, i32 1);
                        goto <header>;

                    <exit>
                        return [0x1000];
                "
        );

        let aliases = AliasResult::simple(&ctx);
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
                        store(&A, i32 0);
                        goto <header>;

                    <header>
                        %v = load(i32, &A);
                        store(&A, i32 1);
                        if i8 1 goto <header> else goto <exit>;

                    <exit>
                        return [0x1000];
                "
        );

        let aliases = AliasResult::simple(&ctx);
        gvn_function(&mut ctx, self_loop, Some(&aliases));

        assert!(
            BasicBlock::from_id(&ctx, header)
                .instruction_ids()
                .contains(&v),
            "header load must not be replaced by the entry store; the header's own store \
             overwrites it before the back edge"
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
                    store(&A, i64 5);
                    %a = load(i64, &A);
                    %v1 = %a + 2;
                    %v2 = %v1 + 3;
                    store(&B, %v2);
                    goto <0x1001>;"
        );

        let aliases = AliasResult::simple(&ctx);

        let mut block = BasicBlock::from_id_mut(&mut ctx, block);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));

        gvn(&mut block, Some(&aliases));

        assert!(!block.instruction_ids().contains(&v1));
        assert!(!block.instruction_ids().contains(&v2));
        assert!(block.to_string().contains("B = 0xa"));
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
        let (fun_id, block_id) = single_block_fn(&mut tc);

        let reg_space = tc.reg_space;
        let eax = ValueId::Varnode(tc.r0_lo32);
        let other = ValueId::Varnode(tc.r1);

        let load_id;
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            let c = b.context_mut().get_const(0x12345678, 4).id();
            b.push_store(c, eax, reg_space); // EAX = c
            let loaded = b.push_load::<false>(eax, 4, reg_space).id(); // %r = EAX
            b.push_store(loaded, other, reg_space); // use %r (keeps it live)
            load_id = loaded;
            unsafe { b.dont_finalize() };
        }

        let aliases = AliasResult::simple(&tc.ctx);
        gvn_function(&mut tc.ctx, fun_id, Some(&aliases));

        let ValueId::Instruction(load_insn) = load_id else {
            panic!("push_load should produce an instruction value");
        };
        assert!(
            !BasicBlock::from_id(&tc.ctx, block_id)
                .instruction_ids()
                .contains(&load_insn),
            "register reload should be forwarded to the stored value, got:\n{}",
            BasicBlock::from_id(&tc.ctx, block_id)
        );
    }

    // -----------------------------------------------------------------------
    // Memory coalesce: rebuilding a wide value from partial writes
    // -----------------------------------------------------------------------

    /// A single-block function rooted at 0x1000.
    fn single_block_fn(tc: &mut TestContext) -> (FunctionId, BlockId) {
        let fun_id = Function::make(&mut tc.ctx, "test".into()).unwrap().id;
        let block_id = tc.ctx.get_or_make_block(0x1000);
        Function::from_id_mut(&mut tc.ctx, fun_id)
            .set_root(block_id)
            .unwrap();
        (fun_id, block_id)
    }

    /// Run const-fold + GVN to a fixpoint, as the real `Gvn` pass does.
    fn optimize(ctx: &mut Context, fun_id: FunctionId) {
        loop {
            let mut changed = constant_fold_function(ctx, fun_id);
            let aliases = AliasResult::simple(ctx);
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
            Mnemonic::Store(s) => s.src,
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
            ValueId::Literal(lid) => Some(ctx.values.literals[lid].value),
            _ => None,
        }
    }

    /// Helper: the store-to-r1 instruction in `block` (used to read its src).
    fn load_insn_use(tc: &TestContext, block: BlockId) -> ValueId {
        BasicBlock::from_id(&tc.ctx, block)
            .iter()
            .find(|i| {
                matches!(i.mnemonic(), Mnemonic::Store(s)
                    if matches!(s.ptr, ValueId::Varnode(v) if v == tc.r1))
            })
            .expect("store to r1")
            .id()
    }

    /// `xor eax,eax; setnz al; push eax`: the wide read of EAX must be rebuilt as
    /// `zext(%cc)` — low byte from the partial write, upper bytes from the zero.
    #[test]
    fn coalesce_motivating_idiom_symbolic_low_byte() {
        let mut tc = TestContext::new();
        let (fun_id, block_id) = single_block_fn(&mut tc);
        let reg = tc.reg_space;

        let (cc, load_id, keep);
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            // A symbolic 1-byte value (the `setnz al` result), read before the zero.
            cc = b
                .push_load::<false>(ValueId::Varnode(tc.r0_byte1), 1, reg)
                .id();
            let zero = b.context_mut().get_const(0, 4).id();
            b.push_store(zero, ValueId::Varnode(tc.r0_lo32), reg); // xor eax, eax
            b.push_store(cc, ValueId::Varnode(tc.r0_byte0), reg); // setnz al
            load_id = b
                .push_load::<false>(ValueId::Varnode(tc.r0_lo32), 4, reg)
                .id();
            keep = b.push_store(load_id, ValueId::Varnode(tc.r1), reg).id(); // push eax (use)
            unsafe { b.dont_finalize() };
        }

        optimize(&mut tc.ctx, fun_id);

        let ValueId::Instruction(load_insn) = load_id else {
            panic!("load is an instruction");
        };
        assert!(
            !block_contains(&tc.ctx, block_id, load_insn),
            "the wide EAX read should be coalesced away:\n{}",
            BasicBlock::from_id(&tc.ctx, block_id)
        );

        // After folding, the rebuilt value is exactly `zext(%cc)`.
        let src = store_src(&tc.ctx, keep);
        let ValueId::Instruction(zid) = src else {
            panic!("expected the rebuilt value to be an instruction, got {src:?}");
        };
        match tc.ctx.get_insn(zid).mnemonic() {
            Mnemonic::Zext(z) => assert_eq!(z.src, cc, "zext of the setnz byte"),
            other => panic!("expected zext(%cc), got {other:?}"),
        }
    }

    /// Same idiom but the low byte is a constant: the whole read folds to a literal.
    #[test]
    fn coalesce_constant_idiom_folds_to_literal() {
        let mut tc = TestContext::new();
        let (fun_id, _block) = single_block_fn(&mut tc);
        let reg = tc.reg_space;

        let keep;
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            let zero = b.context_mut().get_const(0, 4).id();
            let one = b.context_mut().get_const(1, 1).id();
            b.push_store(zero, ValueId::Varnode(tc.r0_lo32), reg);
            b.push_store(one, ValueId::Varnode(tc.r0_byte0), reg);
            let v = b
                .push_load::<false>(ValueId::Varnode(tc.r0_lo32), 4, reg)
                .id();
            keep = b.push_store(v, ValueId::Varnode(tc.r1), reg).id();
            unsafe { b.dont_finalize() };
        }

        optimize(&mut tc.ctx, fun_id);

        assert_eq!(
            literal_value(&tc.ctx, store_src(&tc.ctx, keep)),
            Some(1),
            "0x00000000 with low byte 1 folds to 1"
        );
    }

    /// Four distinct constant byte writes coalesce in little-endian order.
    #[test]
    fn coalesce_constant_bytes_are_little_endian() {
        let mut tc = TestContext::new();
        let (fun_id, _block) = single_block_fn(&mut tc);
        let reg = tc.reg_space;

        let keep;
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            for (byte_vn, val) in [
                (tc.r0_byte0, 0xAA),
                (tc.r0_byte1, 0xBB),
                (tc.r0_byte2, 0xCC),
                (tc.r0_byte3, 0xDD),
            ] {
                let c = b.context_mut().get_const(val, 1).id();
                b.push_store(c, ValueId::Varnode(byte_vn), reg);
            }
            let v = b
                .push_load::<false>(ValueId::Varnode(tc.r0_lo32), 4, reg)
                .id();
            keep = b.push_store(v, ValueId::Varnode(tc.r1), reg).id();
            unsafe { b.dont_finalize() };
        }

        optimize(&mut tc.ctx, fun_id);

        assert_eq!(
            literal_value(&tc.ctx, store_src(&tc.ctx, keep)),
            Some(0xDDCC_BBAA),
            "byte 0 is least significant"
        );
    }

    /// Two symbolic byte reads coalesce into a halfword built with `zext`/`<<`/`|`.
    #[test]
    fn coalesce_two_symbolic_bytes_into_halfword() {
        let mut tc = TestContext::new();
        let (fun_id, block_id) = single_block_fn(&mut tc);
        let reg = tc.reg_space;

        let load_id;
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            // Symbolic sources from non-overlapping bytes (offsets 2 and 3).
            let x = b
                .push_load::<false>(ValueId::Varnode(tc.r0_byte2), 1, reg)
                .id();
            let y = b
                .push_load::<false>(ValueId::Varnode(tc.r0_byte3), 1, reg)
                .id();
            b.push_store(x, ValueId::Varnode(tc.r0_byte0), reg);
            b.push_store(y, ValueId::Varnode(tc.r0_byte1), reg);
            load_id = b
                .push_load::<false>(ValueId::Varnode(tc.r0_lo16), 2, reg)
                .id();
            b.push_store(load_id, ValueId::Varnode(tc.r1), reg);
            unsafe { b.dont_finalize() };
        }

        optimize(&mut tc.ctx, fun_id);

        let ValueId::Instruction(load_insn) = load_id else {
            panic!("load is an instruction");
        };
        assert!(
            !block_contains(&tc.ctx, block_id, load_insn),
            "the halfword read should be coalesced:\n{}",
            BasicBlock::from_id(&tc.ctx, block_id)
        );
        // The result must not collapse to a constant (both pieces are symbolic).
        assert!(
            literal_value(&tc.ctx, store_src(&tc.ctx, load_insn_use(&tc, block_id))).is_none(),
            "symbolic coalesce must not fold to a literal"
        );
    }

    /// Four sub-byte reads then a wide read: the wide read coalesces all four.
    #[test]
    fn coalesce_four_byte_reads_into_word() {
        let mut tc = TestContext::new();
        let (fun_id, block_id) = single_block_fn(&mut tc);
        let reg = tc.reg_space;

        let load_id;
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            for byte_vn in [tc.r0_byte0, tc.r0_byte1, tc.r0_byte2, tc.r0_byte3] {
                b.push_load::<false>(ValueId::Varnode(byte_vn), 1, reg);
            }
            load_id = b
                .push_load::<false>(ValueId::Varnode(tc.r0_lo32), 4, reg)
                .id();
            b.push_store(load_id, ValueId::Varnode(tc.r1), reg);
            unsafe { b.dont_finalize() };
        }

        optimize(&mut tc.ctx, fun_id);

        let ValueId::Instruction(load_insn) = load_id else {
            panic!("load is an instruction");
        };
        assert!(
            !block_contains(&tc.ctx, block_id, load_insn),
            "the word read should coalesce four byte reads:\n{}",
            BasicBlock::from_id(&tc.ctx, block_id)
        );
    }

    /// A wide store then a one-byte overwrite: the wide read keeps a `Range` of the
    /// wide store for the untouched upper bytes.
    #[test]
    fn coalesce_wide_store_with_one_byte_overwrite_uses_range() {
        let mut tc = TestContext::new();
        let (fun_id, block_id) = single_block_fn(&mut tc);
        let reg = tc.reg_space;

        let w;
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            w = b
                .push_load::<false>(ValueId::Varnode(tc.r0_lo32), 4, reg)
                .id();
            // Re-store it so the wide value is the live writer of bytes 1..4.
            b.push_store(w, ValueId::Varnode(tc.r0_lo32), reg);
            let lo = b.context_mut().get_const(0xAB, 1).id();
            b.push_store(lo, ValueId::Varnode(tc.r0_byte0), reg);
            let v = b
                .push_load::<false>(ValueId::Varnode(tc.r0_lo32), 4, reg)
                .id();
            b.push_store(v, ValueId::Varnode(tc.r1), reg);
            unsafe { b.dont_finalize() };
        }

        optimize(&mut tc.ctx, fun_id);

        // A Range(%w, 1, 3) must have been materialized for the upper three bytes.
        let has_upper_range = BasicBlock::from_id(&tc.ctx, block_id).iter().any(|i| {
            matches!(i.mnemonic(), Mnemonic::Range(Range { src, start: 1, size: 3 }) if *src == w)
        });
        assert!(
            has_upper_range,
            "expected Range(%w, 1, 3) for the untouched upper bytes:\n{}",
            BasicBlock::from_id(&tc.ctx, block_id)
        );
    }

    /// A wide write then a narrow read still works (the old `range_table` path):
    /// reading byte 1 of a wide value yields `Range(%w, 1, 1)`.
    #[test]
    fn narrow_read_of_wide_value_extracts_range() {
        let mut tc = TestContext::new();
        let (fun_id, block_id) = single_block_fn(&mut tc);
        let reg = tc.reg_space;

        let (w, narrow);
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            w = b
                .push_load::<false>(ValueId::Varnode(tc.r0_lo32), 4, reg)
                .id();
            b.push_store(w, ValueId::Varnode(tc.r0_lo32), reg);
            narrow = b
                .push_load::<false>(ValueId::Varnode(tc.r0_byte1), 1, reg)
                .id();
            b.push_store(narrow, ValueId::Varnode(tc.r1), reg);
            unsafe { b.dont_finalize() };
        }

        optimize(&mut tc.ctx, fun_id);

        let ValueId::Instruction(narrow_insn) = narrow else {
            panic!("load is an instruction");
        };
        assert!(
            !block_contains(&tc.ctx, block_id, narrow_insn),
            "the narrow read should be replaced by a Range"
        );
        let has_range = BasicBlock::from_id(&tc.ctx, block_id).iter().any(|i| {
            matches!(i.mnemonic(), Mnemonic::Range(Range { src, start: 1, size: 1 }) if *src == w)
        });
        assert!(has_range, "expected Range(%w, 1, 1)");
    }

    /// A gap in coverage (byte 1 never written) leaves the wide read intact.
    #[test]
    fn coalesce_bails_on_coverage_gap() {
        let mut tc = TestContext::new();
        let (fun_id, block_id) = single_block_fn(&mut tc);
        let reg = tc.reg_space;

        let load_id;
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            let c0 = b.context_mut().get_const(1, 1).id();
            let c2 = b.context_mut().get_const(2, 1).id();
            b.push_store(c0, ValueId::Varnode(tc.r0_byte0), reg);
            b.push_store(c2, ValueId::Varnode(tc.r0_byte2), reg); // byte 1 and 3 unwritten
            load_id = b
                .push_load::<false>(ValueId::Varnode(tc.r0_lo32), 4, reg)
                .id();
            b.push_store(load_id, ValueId::Varnode(tc.r1), reg);
            unsafe { b.dont_finalize() };
        }

        optimize(&mut tc.ctx, fun_id);

        let ValueId::Instruction(load_insn) = load_id else {
            panic!("load is an instruction");
        };
        assert!(
            block_contains(&tc.ctx, block_id, load_insn),
            "a partially-covered read must not be forwarded"
        );
    }

    /// Partial writes in a dominating block forward into a dominated successor.
    #[test]
    fn coalesce_across_dominating_block() {
        let mut tc = TestContext::new();
        let fun_id = Function::make(&mut tc.ctx, "test".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let succ = tc.ctx.get_or_make_block(0x2000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fun_id);
            f.set_root(entry).unwrap();
            f.add_block(entry);
            f.add_block(succ);
        }
        let reg = tc.reg_space;

        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let c0 = b.context_mut().get_const(0xAA, 1).id();
            let c1 = b.context_mut().get_const(0xBB, 1).id();
            b.push_store(c0, ValueId::Varnode(tc.r0_byte0), reg);
            b.push_store(c1, ValueId::Varnode(tc.r0_byte1), reg);
            b.push_branch(succ);
            unsafe { b.dont_finalize() };
        }
        let load_id;
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, succ));
            load_id = b
                .push_load::<false>(ValueId::Varnode(tc.r0_lo16), 2, reg)
                .id();
            b.push_store(load_id, ValueId::Varnode(tc.r1), reg);
            unsafe { b.dont_finalize() };
        }

        optimize(&mut tc.ctx, fun_id);

        let ValueId::Instruction(load_insn) = load_id else {
            panic!("load is an instruction");
        };
        assert!(
            !block_contains(&tc.ctx, succ, load_insn),
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
                    store({r0_byte0}, i32 0xAA);
                    store({r0_byte1}, i32 0xBB);

                    %load = load(i32, {r0_lo16});
                    store(%load, {r0});
                    goto <0x1000>;"
        );

        let mut bb = BasicBlock::from_id_mut(&mut tc.ctx, block);
        gvn(&mut bb, None);

        assert!(
            block_contains(&tc.ctx, block, load),
            "without an alias oracle the coalesce read must remain"
        );
    }
}
