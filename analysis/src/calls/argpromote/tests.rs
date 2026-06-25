#[cfg(test)]
mod tests {
    use qcode::builder::Builder;
    use qcode::value::{
        BasicBlock, BlockId, Function, Instruction, Value, Varnode, VarnodeId,
        insn::{Binop, Call, InstructionId, IntBinop, Mnemonic},
    };
    use qcode_macro::qcode;

    use super::super::*;
    use super::super::{mark_pure::*, ram::*, registers::*};

    /// Add a "stack" space (addr_size 4, as `brighten_stack` creates it) and a
    /// nameless stack-passed input varnode at `offset`. Its synthesized argument
    /// name is `stack_<stack_base(4) + offset>` — e.g. offset 4 → `stack_10000004`
    /// — which is what a callee's promoted param name must equal.
    fn stack_input(tc: &mut qcode::testing::TestContext, offset: i64, size: usize) -> VarnodeId {
        use qcode::space::Space;
        let space = tc
            .ctx
            .try_get_space("stack")
            .unwrap_or_else(|| tc.ctx.add_space(Space::new(Some("stack"), 1, 4)));
        Varnode::make(&mut tc.ctx, offset, size, space).id
    }

    /// Find the (single) call instruction in `block` and give it `args`.
    fn set_call(
        tc: &mut qcode::testing::TestContext,
        block: BlockId,
        target: FunctionId,
        args: Vec<ValueId>,
    ) -> InstructionId {
        let call_id = BasicBlock::from_id(&tc.ctx, block)
            .iter()
            .find(|i| matches!(i.mnemonic(), Mnemonic::Call(_)))
            .unwrap()
            .id;
        tc.ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target,
                args,
                clobbers: vec![],
            }),
        );
        call_id
    }

    /// Phase 0 gate for the register channel (see `ARGPROMOTE_REGISTERS.md`): the
    /// early register run will send an **aggregate-typed call result** through
    /// mem2reg/gvn/dce for the first time (today's RAM run is post-mem2reg, so
    /// aggregates never reach those passes). An opaque call result is the riskiest
    /// shape — gvn cannot fold it, so the `extract` must thread through intact.
    /// This builds exactly that and asserts the value passes neither panic nor
    /// mangle it.
    #[test]
    fn aggregate_call_result_survives_value_passes() {
        let mut tc = qcode::testing::TestContext::new();
        let (r0, r1, r3) = (tc.r0, tc.r1, tc.r3);
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry>
                    %v = load(i64, {r1});
                    %s = %v + i64 5;
                    store({r0}, %s);
                    %fin = load(i64, {r0});
                    %agg = (%fin);
                    return [%agg];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = (f, g, r0, r1);

        // A one-field aggregate (the positional register write-set shape).
        let i64_ty = tc.ctx.types.get_or_make_int(8);
        let agg_ty = tc.ctx.types.get_or_make_aggregate(vec![i64_ty]);

        // Make the call produce that aggregate, then replay field 0 into r3.
        let call_id = set_call(&mut tc, g_call, f, vec![]);
        Instruction::from_id_mut(&mut tc.ctx, call_id).set_type(agg_ty);
        tc.ctx.add_cfg_edge(g_call, g_cont);
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, g_cont));
            b.set_insert_point_to_start();
            let out = ValueId::Instruction(b.push_extract(ValueId::Instruction(call_id), 0).id);
            b.push_store(out, ValueId::Varnode(r3), tc.reg_space);
        }

        // The Phase 0 question: do the value passes survive an aggregate-typed,
        // opaque call result? (A panic here fails the test.)
        let aliases = crate::AliasResult::simple(&tc.ctx);
        let _ = crate::mem2reg(&mut tc.ctx, g, &aliases);
        let _ = crate::gvn_function(&mut tc.ctx, g, Some(&aliases));

        // gvn cannot see into the opaque call result, so the extract must remain.
        let has_extract = Function::from_id(&tc.ctx, g).iter().any(|b| {
            b.iter()
                .any(|i| matches!(i.mnemonic(), Mnemonic::Extract(_)))
        });
        assert!(
            has_extract,
            "extract of an aggregate call result must survive mem2reg+gvn"
        );
    }

    #[test]
    fn promotes_single_inout_stack_param() {
        let mut tc = qcode::testing::TestContext::new();
        // Bind the stack slot at offset 4 as f's single input.
        let input = stack_input(&mut tc, 4, 8);

        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64>
                    %v = load(i32, @stack_10000004);
                    store(@stack_10000004, %v);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;

        Function::from_id_mut(&mut tc.ctx, f).set_input_regs(vec![input]);
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let ptr = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, f, vec![ptr]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(
            argpromote(&mut tc.ctx),
            "single in/out param should be promoted"
        );

        // The callee now hands the buffer back through Return::value.
        let returns_value = Function::from_id(&tc.ctx, f).iter().any(|b| {
            b.iter()
                .last()
                .is_some_and(|i| matches!(i.mnemonic(), Mnemonic::Return(r) if r.value.is_some()))
        });
        assert!(returns_value, "callee must return the buffer value");

        // The caller loads the region just before the call (the by-value arg).
        let has_load = BasicBlock::from_id(&tc.ctx, g_call)
            .iter()
            .any(|i| matches!(i.mnemonic(), Mnemonic::Load(_)));
        assert!(
            has_load,
            "caller must load the region value before the call"
        );
        let _ = call_id;
    }

    #[test]
    fn skips_non_pure_reg_function() {
        // The RAM channel keys snapshot args off the `param[i] ↔ Call.args[i]`
        // lockstep, which is an invariant ONLY for `pure_reg` functions. A
        // conventional (non-`pure_reg`) function uses the register ABI and accrues
        // root params no caller passes positionally, so argpromote must leave it
        // untouched rather than read a non-existent argument slot. Identical setup to
        // `promotes_single_inout_stack_param`, but WITHOUT `set_pure_reg` — so nothing
        // is promoted.
        let mut tc = qcode::testing::TestContext::new();
        let input = stack_input(&mut tc, 4, 8);
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64>
                    %v = load(i32, @stack_10000004);
                    store(@stack_10000004, %v);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, f).set_input_regs(vec![input]);
        // NB: deliberately NOT `set_pure_reg(true)`.
        let ptr = tc.ctx.get_const(0x4000, 8).id();
        set_call(&mut tc, g_call, f, vec![ptr]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(
            !argpromote(&mut tc.ctx),
            "a non-pure_reg function must be left untouched (its param↔arg lockstep is not an invariant)"
        );
    }

    #[test]
    fn pure_function_may_call_pure_function() {
        // A `pure_reg` caller whose only non-trivial instruction is a clobber-free
        // call to an `is_pure` callee is itself pure; flip the callee to impure and
        // the caller must no longer qualify.
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn callee:
                <c_entry>
                    return [i64 0];

            fn caller:
                <f_entry>
                    goto <f_call>;
                <f_call>
                    call <callee>;
                <f_cont>
                    return [i64 0];
            "
        );
        let _ = (f_entry, f_cont);

        Function::from_id_mut(&mut tc.ctx, caller).set_pure_reg(true);
        set_call(&mut tc, f_call, callee, vec![]);
        tc.ctx.add_cfg_edge(f_call, f_cont);

        // Callee not yet pure → caller's call is an untracked effect.
        assert!(
            !body_is_pure(&tc.ctx, caller),
            "call to impure callee blocks purity"
        );

        Function::from_id_mut(&mut tc.ctx, callee).set_is_pure(true);
        assert!(
            body_is_pure(&tc.ctx, caller),
            "a clobber-free call to a pure callee is permitted in a pure body"
        );

        // mark_pure flags the caller once the callee is pure, order-independently.
        Function::from_id_mut(&mut tc.ctx, caller).set_pure_reg(true);
        mark_pure_functions(&mut tc.ctx);
        assert!(
            Function::from_id(&tc.ctx, caller).is_pure(),
            "caller of a pure function should be marked pure"
        );
    }

    /// Whether `f`'s root has a by-value snapshot param (a promoted read).
    fn has_val_param(ctx: &Context, f: FunctionId) -> bool {
        Function::from_id(ctx, f).root().is_some_and(|b| {
            b.params()
                .any(|p| p.name().is_some_and(|n| n.contains("_val_")))
        })
    }

    #[test]
    fn unmodelable_access_falls_back_to_partial() {
        // When a real-memory access cannot be captured in the shadow — here a store
        // through an address *loaded from memory* (a multi-level deref the model
        // can't recompute) — the shadow path is unsound (a later store→load forward
        // could collapse a read across the unmodelled store). Instead of bailing,
        // we fall back to *partial* promotion: the clean `@p+0` / `@p+0x30` reads
        // are exposed as by-value snapshot params seeded into REAL ram, and nothing
        // is redirected into shadow — so the unmodelled store is left untouched and
        // there is no forwarder-collapse exposure.
        let mut tc = qcode::testing::TestContext::new();
        let input = stack_input(&mut tc, 4, 8);
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64>
                    %addr = load(i64, @stack_10000004);
                    store(%addr, i32 0);
                    %a = @stack_10000004 + i64 0x30;
                    %v = load(i32, %a);
                    return [%v];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, f).set_input_regs(vec![input]);
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let ptr = tc.ctx.get_const(0x4000, 8).id();
        set_call(&mut tc, g_call, f, vec![ptr]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(
            argpromote(&mut tc.ctx),
            "an unmodelable access must fall back to partial promotion, not bail"
        );
        assert!(
            has_val_param(&tc.ctx, f),
            "the clean reads should be exposed as by-value snapshot params"
        );
        // Partial mode never touches shadow: every load/store stays in real ram, so
        // the unmodelled store cannot be collapsed across.
        let ram = tc.ctx.default_space;
        let all_ram = Function::from_id(&tc.ctx, f).blocks().all(|b| {
            b.iter().all(|i| match i.mnemonic() {
                Mnemonic::Load(l) => l.space == ram,
                Mnemonic::Store(s) => s.space == ram,
                _ => true,
            })
        });
        assert!(
            all_ram,
            "partial mode must not redirect any access into shadow"
        );
        // Idempotent: the snapshots already exist, so a re-visit adds nothing.
        assert!(
            !argpromote(&mut tc.ctx),
            "re-running partial promotion must be a no-op (idempotence)"
        );
    }

    #[test]
    fn partial_mode_is_inputs_only_no_writeset() {
        // A pointer param written through (an in/out write) AND leaked (its address
        // stored to memory) escapes, so the shadow path is unavailable. Partial mode
        // is **inputs-only**: with no *reads* to expose, there is nothing to promote.
        // The write is left in place (correct, just not functionalized) and no
        // write-set is surfaced.
        let mut tc = qcode::testing::TestContext::new();
        let input = stack_input(&mut tc, 4, 8);
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64>
                    store(@stack_10000004, i32 0x41);
                    store(i64 0x9000, @stack_10000004);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, f).set_input_regs(vec![input]);
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let ptr = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, f, vec![ptr]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        // No reads to expose → inputs-only partial mode promotes nothing.
        assert!(
            !argpromote(&mut tc.ctx),
            "a write-only leaked function has no reads to expose, so partial mode is a no-op"
        );

        // The in-place store survives untouched (still in real ram).
        let ram = tc.ctx.default_space;
        assert!(
            Function::from_id(&tc.ctx, f).blocks().any(|b| b.iter().any(
                |i| matches!(i.mnemonic(), Mnemonic::Store(s) if s.space == ram && s.size == 4)
            )),
            "the original in-place store must survive"
        );

        // No write-set was surfaced, so the caller has no replay extract.
        let has_replay = Function::from_id(&tc.ctx, g).iter().any(|b| {
            b.iter().any(
                |i| matches!(i.mnemonic(), Mnemonic::Extract(e) if e.agg == ValueId::Instruction(call_id)),
            )
        });
        assert!(
            !has_replay,
            "inputs-only partial mode surfaces no write-set to replay"
        );
    }

    #[test]
    fn sub_offset_store_is_modelled_and_read_promotes() {
        // A `Sub`-form deref store (`@p-4`, the `[ESP-k]` shape) is now captured and
        // redirected into the shadow, so the footprint is fully modelled and the
        // disjoint `@p+0x30` read still promotes. (The shared shadow + alias-aware
        // gvn is what later keeps an *overlapping* read behind a load; here the
        // store is disjoint, so the read is free to forward.)
        let mut tc = qcode::testing::TestContext::new();
        let input = stack_input(&mut tc, 4, 8);
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64>
                    %s = @stack_10000004 - i64 0x4;
                    store(%s, i32 0);
                    %a = @stack_10000004 + i64 0x30;
                    %v = load(i32, %a);
                    return [%v];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, f).set_input_regs(vec![input]);
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let ptr = tc.ctx.get_const(0x4000, 8).id();
        set_call(&mut tc, g_call, f, vec![ptr]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(
            argpromote(&mut tc.ctx),
            "a fully-modelled footprint (Sub store + disjoint read) must promote"
        );
        assert!(
            has_val_param(&tc.ctx, f),
            "the disjoint read should still be promoted"
        );
    }

    #[test]
    fn argpromote_is_idempotent() {
        // Re-running over already-promoted IR must be a no-op: the seed store and
        // the redirected shadow accesses live in a shadow space, which
        // `analyze_param` now skips. (Without this, the seed `store(snap, ptr)`
        // would look like a fresh write through the pointer and re-promote.)
        let mut tc = qcode::testing::TestContext::new();
        let input = stack_input(&mut tc, 4, 8);
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64>
                    %v = load(i32, @stack_10000004);
                    store(@stack_10000004, %v);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, f).set_input_regs(vec![input]);
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let ptr = tc.ctx.get_const(0x4000, 8).id();
        set_call(&mut tc, g_call, f, vec![ptr]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(argpromote(&mut tc.ctx), "first run promotes");
        assert!(
            !argpromote(&mut tc.ctx),
            "second run over promoted IR must change nothing"
        );
    }

    #[test]
    fn promotes_read_at_large_fixed_offset() {
        // A read at offset 0x30 (a segment-relative `FS:0x30`-style access) used to
        // Escape: the snapshot was a contiguous `[base, base+0x30+4)` = 0x34-byte
        // region failing the 1|2|4|8 width gate. Now the offset is decoupled from
        // the width: a 4-byte scalar keyed by offset 0x30. Read-only, so no return
        // rewrite — only a by-value arg appears.
        let mut tc = qcode::testing::TestContext::new();
        let input = stack_input(&mut tc, 4, 8);

        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64>
                    %a = @stack_10000004 + i64 0x30;
                    %v = load(i32, %a);
                    return [%v];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;

        Function::from_id_mut(&mut tc.ctx, f).set_input_regs(vec![input]);
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let ptr = tc.ctx.get_const(0x4000, 8).id();
        set_call(&mut tc, g_call, f, vec![ptr]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(
            argpromote(&mut tc.ctx),
            "a 4-byte read at fixed offset 0x30 should promote"
        );

        // The callee gains a by-value snapshot param keyed by the offset.
        let has_val_param = Function::from_id(&tc.ctx, f).root().is_some_and(|b| {
            b.params()
                .any(|p| p.name().is_some_and(|n| n.contains("_val_30")))
        });
        assert!(
            has_val_param,
            "callee must gain a `*_val_30` by-value param"
        );

        // The caller computes `arg + 0x30` and loads the scalar before the call.
        let g_block = BasicBlock::from_id(&tc.ctx, g_call);
        assert!(
            g_block
                .iter()
                .any(|i| matches!(i.mnemonic(), Mnemonic::Load(_))),
            "caller must load the scalar at arg + 0x30"
        );
        assert!(
            g_block.iter().any(
                |i| matches!(i.mnemonic(), Mnemonic::Binop(b) if matches!(b.op, Binop::Int(IntBinop::Add)))
            ),
            "caller must compute the arg + 0x30 address"
        );
    }

    #[test]
    fn noop_when_no_promotable_param() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn foo:
                <entry>
                    return [0x1000];
            "
        );
        assert!(Function::from_id(&ctx, foo).root().is_some());
        // No pointer parameters, no callers: nothing to promote, and no panic.
        assert!(!argpromote(&mut ctx));
    }

    // ---- TDD acceptance suite for the shadow-memory / write-set redesign ----
    //
    // These encode the worked examples from the design (see ARGPROMOTE_DESIGN.md).
    // They are `#[ignore]`d until the shadow-memory rewrite lands: the current
    // single-buffer pass does not yet produce the `(real_return, write-set)`
    // aggregate return, nor replay it at the caller. Each test builds the
    // pre-transformation callee + a caller, runs `argpromote`, and asserts the
    // caller-observable contract: one replayed `store` per write-set entry. The
    // exact aggregate nesting is an implementation detail and deliberately not
    // asserted here.

    /// Number of `store` instructions the caller replays after the call — i.e.
    /// in the call block's continuation. The new design emits one per write-set
    /// `(address, value)` pair returned by the callee.
    fn replayed_stores(tc: &qcode::testing::TestContext, call_id: InstructionId) -> usize {
        let Some(call_block) = Instruction::from_id(&tc.ctx, call_id)
            .parent()
            .map(|b| b.id)
        else {
            return 0;
        };
        let Some(cont) = BasicBlock::from_id(&tc.ctx, call_block)
            .successors()
            .next()
            .map(|(_, b)| b)
        else {
            return 0;
        };
        BasicBlock::from_id(&tc.ctx, cont)
            .iter()
            .filter(|i| matches!(i.mnemonic(), Mnemonic::Store(_)))
            .count()
    }

    /// `void f(int *p, int *q) { *q = 100; *p += 1; }`
    /// → `((q_ptr, 100), (p_ptr, p + 1))`. The aliasing example: with both
    /// pointers sharing `shadow_ram`, `f(&x, &x)` stays correct, and the caller
    /// replays two writes. (This is the case v1 *rejects*.)
    #[test]
    fn example_two_inout_pointers() {
        let mut tc = qcode::testing::TestContext::new();
        let in_p = stack_input(&mut tc, 4, 8); // stack_10000004
        let in_q = stack_input(&mut tc, 16, 8); // stack_10000010
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64 @stack_10000010:i64>
                    store(@stack_10000010, i32 100);
                    %v = load(i32, @stack_10000004);
                    %s = %v + i32 1;
                    store(@stack_10000004, %s);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, f).set_input_regs(vec![in_p, in_q]);
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let a = tc.ctx.get_const(0x4000, 8).id();
        let b = tc.ctx.get_const(0x5000, 8).id();
        let call_id = set_call(&mut tc, g_call, f, vec![a, b]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(
            argpromote(&mut tc.ctx),
            "both in/out pointers should be promoted"
        );
        assert_eq!(
            replayed_stores(&tc, call_id),
            2,
            "two writes => two replays"
        );
    }

    /// `void foo(int *p) { *p += 10; }`
    /// → `(intptr, int) foo(intptr p_ptr, int p) { return (p_ptr, p + 10); }`.
    #[test]
    fn example_single_inout_void() {
        let mut tc = qcode::testing::TestContext::new();
        let in_p = stack_input(&mut tc, 4, 8);
        qcode!(
            tc.ctx,
            "
            fn foo:
                <foo_entry @stack_10000004:i64>
                    %v = load(i32, @stack_10000004);
                    %s = %v + i32 10;
                    store(@stack_10000004, %s);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <foo>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, foo).set_input_regs(vec![in_p]);
        Function::from_id_mut(&mut tc.ctx, foo).set_pure_reg(true);
        let a = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, foo, vec![a]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(argpromote(&mut tc.ctx));
        assert_eq!(replayed_stores(&tc, call_id), 1);
    }

    /// `int* foo(int* p) { *p += 10; return p; }`
    /// → `(intptr, (intptr, int)) foo(intptr p_ptr, int p) { return (p_ptr, (p_ptr, p + 10)); }`.
    /// A real return value coexists with the write-set.
    #[test]
    fn example_inout_returns_pointer() {
        let mut tc = qcode::testing::TestContext::new();
        let in_p = stack_input(&mut tc, 4, 8);
        let r0 = tc.r0; // the real-return register
        qcode!(
            tc.ctx,
            "
            fn foo:
                <foo_entry @stack_10000004:i64>
                    %v = load(i32, @stack_10000004);
                    %s = %v + i32 10;
                    store(@stack_10000004, %s);
                    store({r0}, @stack_10000004);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <foo>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, foo).set_input_regs(vec![in_p]);
        Function::from_id_mut(&mut tc.ctx, foo).set_pure_reg(true);
        let a = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, foo, vec![a]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(argpromote(&mut tc.ctx));
        // One write replayed; the returned pointer rides the real-return channel.
        assert_eq!(replayed_stores(&tc, call_id), 1);
    }

    /// `int foo(int* p) { p += 10; *p = 5; }`
    /// → `(intptr, int) foo(intptr p_ptr, int p) { return (p_ptr + 10, 5); }`.
    /// A write through an *advanced* pointer: the returned address is `p + 10`.
    #[test]
    fn example_write_through_advanced_pointer() {
        let mut tc = qcode::testing::TestContext::new();
        let in_p = stack_input(&mut tc, 4, 8);
        qcode!(
            tc.ctx,
            "
            fn foo:
                <foo_entry @stack_10000004:i64>
                    %adv = @stack_10000004 + i64 40;
                    store(%adv, i32 5);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <foo>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, foo).set_input_regs(vec![in_p]);
        Function::from_id_mut(&mut tc.ctx, foo).set_pure_reg(true);
        let a = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, foo, vec![a]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(argpromote(&mut tc.ctx));
        assert_eq!(replayed_stores(&tc, call_id), 1);
    }

    /// `int foo(int *p) { return *p + 10; }`
    /// → `int foo(int p) { return p + 10; }`. A read-only pointer collapses to a
    /// by-value scalar: no write-set, so the caller replays nothing.
    #[test]
    fn example_read_only_pointer_has_no_writeset() {
        let mut tc = qcode::testing::TestContext::new();
        let in_p = stack_input(&mut tc, 4, 8);
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn foo:
                <foo_entry @stack_10000004:i64>
                    %v = load(i32, @stack_10000004);
                    %s = %v + i32 10;
                    store({r0}, %s);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <foo>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, foo).set_input_regs(vec![in_p]);
        Function::from_id_mut(&mut tc.ctx, foo).set_pure_reg(true);
        let a = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, foo, vec![a]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        argpromote(&mut tc.ctx);
        assert_eq!(
            replayed_stores(&tc, call_id),
            0,
            "read-only pointer writes nothing back"
        );
    }

    /// `int* foo(int *p) { return p + 10; }`
    /// → `intptr foo(intptr p) { return p + 10; }`. Pure pointer arithmetic, no
    /// dereference: nothing to mirror, no write-set.
    #[test]
    fn example_pointer_arithmetic_only_has_no_writeset() {
        let mut tc = qcode::testing::TestContext::new();
        let in_p = stack_input(&mut tc, 4, 8);
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn foo:
                <foo_entry @stack_10000004:i64>
                    %adv = @stack_10000004 + i64 40;
                    store({r0}, %adv);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <foo>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, foo).set_input_regs(vec![in_p]);
        Function::from_id_mut(&mut tc.ctx, foo).set_pure_reg(true);
        let a = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, foo, vec![a]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        argpromote(&mut tc.ctx);
        assert_eq!(
            replayed_stores(&tc, call_id),
            0,
            "no dereference => no write-set"
        );
    }

    /// `(int*, int, int) f1(int* p_ptr, int p, int q) { return (p_ptr, p + 1, 100); }`
    /// The flat multi-value illustration: one write through `p` plus a real
    /// return, producing a multi-element functional return. Modeled here as
    /// `*p += 1` with an `int` return of `100`; the caller replays the single
    /// write while the `100` rides the real-return channel.
    #[test]
    fn example_multi_value_return() {
        let mut tc = qcode::testing::TestContext::new();
        let in_p = stack_input(&mut tc, 4, 8);
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn f1:
                <f1_entry @stack_10000004:i64>
                    %v = load(i32, @stack_10000004);
                    %s = %v + i32 1;
                    store(@stack_10000004, %s);
                    store({r0}, i32 100);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f1>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, f1).set_input_regs(vec![in_p]);
        Function::from_id_mut(&mut tc.ctx, f1).set_pure_reg(true);
        let a = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, f1, vec![a]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(argpromote(&mut tc.ctx));
        assert_eq!(replayed_stores(&tc, call_id), 1);
    }

    /// A `pure_reg` callee — empty `input_regs`, a register write-set already on
    /// the return — whose **stack-argument** pointer is written. The memory pass
    /// must (1) still *trigger* (the pointer is found via the root params, not the
    /// empty `input_regs`) and (2) *append* its `(addr, value)` pair after the
    /// register field instead of clobbering it. Regression for the two fixes.
    #[test]
    fn pure_reg_stack_pointer_appends_to_register_writeset() {
        let mut tc = qcode::testing::TestContext::new();
        let _ = stack_input(&mut tc, 4, 8);
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64>
                    %out = i64 7 + i64 0;
                    store(@stack_10000004, i32 5);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        // The shape the register/stack channels leave behind: a functionalized
        // function whose ABI register list is never filled in, with a one-field
        // register write-set already on `Return::value` (set as the register
        // channel does, since `return [..]` only sets the conventional operand).
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let ret_id = Function::from_id(&tc.ctx, f)
            .iter()
            .find_map(|b| {
                let last = b.iter().last()?;
                matches!(last.mnemonic(), Mnemonic::Return(_)).then_some(last.id)
            })
            .unwrap();
        let ret_block = tc.ctx.get_insn(ret_id).parent().map(|b| b.id).unwrap();
        let out = ValueId::Instruction(
            BasicBlock::from_id(&tc.ctx, ret_block)
                .iter()
                .find(|i| matches!(i.mnemonic(), Mnemonic::Binop(_)))
                .unwrap()
                .id,
        );
        let reg_tuple = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, ret_block));
            b.set_insert_point_before(ret_id);
            ValueId::Instruction(b.push_named_tuple(vec![("o0".to_owned(), out)]).id)
        };
        {
            let mut m = tc.ctx.get_insn(ret_id).mnemonic().clone();
            if let Mnemonic::Return(ref mut r) = m {
                r.value = Some(reg_tuple);
            }
            tc.ctx.replace_instruction_mnemonic(ret_id, m);
        }
        assert_eq!(
            register_writeset_len(&tc, f),
            Some(1),
            "precondition: one register output field on the return"
        );

        let a = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, f, vec![a]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(
            argpromote(&mut tc.ctx),
            "the stack-argument pointer must be found and promoted"
        );
        assert_eq!(
            register_writeset_len(&tc, f),
            Some(3),
            "the memory write's flat addr+value fields are appended after the \
             register field, not clobbering it"
        );
        assert_eq!(
            replayed_stores(&tc, call_id),
            1,
            "the caller replays the one appended write"
        );
    }

    /// Build a callee `f(@RSP, @p)` that writes to its own stack frame (`@RSP - 8`)
    /// *and* through the caller-supplied pointer `@p`, plus a caller `g`. `@RSP` is
    /// marked as the incoming stack pointer (origin = `sp_vn`). Returns `(f, sp_vn)`.
    fn build_own_frame_writer(tc: &mut qcode::testing::TestContext) -> (FunctionId, VarnodeId) {
        let sp_vn = tc.r3; // stand-in stack-pointer register varnode
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @RSP:i64 @p:i64>
                    %loc = @RSP - i64 0x8;
                    store(%loc, i32 5);
                    store(@p, i32 9);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        // Mark `@RSP` as the incoming stack pointer (origin = the SP register).
        let pid = {
            let f_ref = Function::from_id(&tc.ctx, f);
            let p = f_ref
                .root()
                .unwrap()
                .params()
                .find(|p| p.name() == Some("RSP"))
                .unwrap();
            match p.id() {
                ValueId::BlockParam(pid) => pid,
                _ => unreachable!(),
            }
        };
        tc.ctx.values.block_params[pid].origin = Some(ValueId::Varnode(sp_vn));
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);

        let sp_arg = tc.ctx.get_const(0x7000, 8).id();
        let p_arg = tc.ctx.get_const(0x4000, 8).id();
        set_call(tc, g_call, f, vec![sp_arg, p_arg]);
        tc.ctx.add_cfg_edge(g_call, g_cont);
        (f, sp_vn)
    }

    /// With the stack pointer supplied, the own-frame write (`@RSP - 8`) is dead on
    /// exit and excluded from the returned write-set — only the caller-pointer write
    /// survives (one flat `(addr, value)` pair → 2 fields). Without it, both writes
    /// are captured (4 fields).
    #[test]
    fn own_frame_write_excluded_from_returned_writeset() {
        let mut tc = qcode::testing::TestContext::new();
        let (f, sp_vn) = build_own_frame_writer(&mut tc);
        assert!(argpromote_with_sp(&mut tc.ctx, Some(sp_vn)));
        assert_eq!(
            register_writeset_len(&tc, f),
            Some(2),
            "only the caller-pointer write is returned; the own-frame write is dead on exit"
        );

        // Control: without the stack pointer, the own-frame write is also captured.
        let mut tc2 = qcode::testing::TestContext::new();
        let (f2, _) = build_own_frame_writer(&mut tc2);
        assert!(argpromote_with_sp(&mut tc2.ctx, None));
        assert_eq!(
            register_writeset_len(&tc2, f2),
            Some(4),
            "without frame awareness, both the caller write and the own-frame write are captured"
        );
    }

    /// Regression: when the register channel has **already typed** the call result
    /// to its (smaller) register-only write-set, appending the memory pairs *grows*
    /// that aggregate. The caller retype must use the resizing setter — a plain
    /// `set_type` panics on the size change (`size N → M`), which is what crashed the
    /// whole pipeline on a real binary. Here we pre-type the call to a 1-field
    /// aggregate so the append must resize it.
    #[test]
    fn appended_writeset_resizes_already_typed_call() {
        let mut tc = qcode::testing::TestContext::new();
        let _ = stack_input(&mut tc, 4, 8);
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64>
                    %out = i64 7 + i64 0;
                    store(@stack_10000004, i32 5);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);

        // Give the return a 1-field register write-set, as the register channel does.
        let ret_id = Function::from_id(&tc.ctx, f)
            .iter()
            .find_map(|b| {
                let last = b.iter().last()?;
                matches!(last.mnemonic(), Mnemonic::Return(_)).then_some(last.id)
            })
            .unwrap();
        let ret_block = tc.ctx.get_insn(ret_id).parent().map(|b| b.id).unwrap();
        let out = ValueId::Instruction(
            BasicBlock::from_id(&tc.ctx, ret_block)
                .iter()
                .find(|i| matches!(i.mnemonic(), Mnemonic::Binop(_)))
                .unwrap()
                .id,
        );
        let reg_tuple = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, ret_block));
            b.set_insert_point_before(ret_id);
            ValueId::Instruction(b.push_named_tuple(vec![("o0".to_owned(), out)]).id)
        };
        let reg_ty = tc.ctx.type_of(reg_tuple);
        {
            let mut m = tc.ctx.get_insn(ret_id).mnemonic().clone();
            if let Mnemonic::Return(ref mut r) = m {
                r.value = Some(reg_tuple);
            }
            tc.ctx.replace_instruction_mnemonic(ret_id, m);
        }

        let a = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, f, vec![a]);
        tc.ctx.add_cfg_edge(g_call, g_cont);
        // Pre-type the call result to the register-only write-set (non-zero size),
        // exactly as the register caller-rewrite leaves it. Without the resizing
        // setter, the append below panics on the size change.
        Instruction::from_id_mut(&mut tc.ctx, call_id).set_type(reg_ty);

        assert!(
            argpromote(&mut tc.ctx),
            "the pointer must be promoted and its write-set appended"
        );
        // The append grew the call result past its original register-only size.
        let new_ty = tc.ctx.type_of(ValueId::Instruction(call_id));
        let new_size = tc.ctx.types.size_of(new_ty);
        let old_size = tc.ctx.types.size_of(reg_ty);
        assert!(
            new_size > old_size,
            "call result must grow ({old_size} -> {new_size}) without panicking"
        );
        assert_eq!(replayed_stores(&tc, call_id), 1);
    }

    /// A callee with **two return blocks** whose single in/out write dominates
    /// both returns (it happens in the entry, before the branch). The write-set is
    /// built at *every* return, and the one caller replays the write once. Exercises
    /// the multi-return path and its dominance gate.
    #[test]
    fn example_multi_return_write_dominates() {
        let mut tc = qcode::testing::TestContext::new();
        let in_p = stack_input(&mut tc, 4, 8);
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn foo:
                <foo_entry @stack_10000004:i64>
                    %v = load(i32, @stack_10000004);
                    %s = %v + i32 1;
                    store(@stack_10000004, %s);
                    %c = load(i8, {r0});
                    if %c goto <ret_a> else goto <ret_b>;
                <ret_a>
                    return [i64 0];
                <ret_b>
                    return [i64 1];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <foo>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = (g, r0, ret_a, ret_b);
        Function::from_id_mut(&mut tc.ctx, foo).set_input_regs(vec![in_p]);
        Function::from_id_mut(&mut tc.ctx, foo).set_pure_reg(true);
        let a = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, foo, vec![a]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(
            argpromote(&mut tc.ctx),
            "a multi-return function whose write dominates both returns must promote"
        );
        // Both returns now carry a write-set.
        let returns_with_value = Function::from_id(&tc.ctx, foo)
            .iter()
            .filter(|b| {
                b.iter().last().is_some_and(
                    |i| matches!(i.mnemonic(), Mnemonic::Return(r) if r.value.is_some()),
                )
            })
            .count();
        assert_eq!(returns_with_value, 2, "every return carries the write-set");
        assert_eq!(
            replayed_stores(&tc, call_id),
            1,
            "the caller replays one write"
        );
    }

    /// A path-dependent write (written on only one arm) is now promotable: the write
    /// target is *seeded* as a by-value input, so the arm that does not write it
    /// reloads that initial value and replays a no-op — sound without any dominance
    /// requirement.
    #[test]
    fn multi_return_path_dependent_write_is_seeded() {
        let mut tc = qcode::testing::TestContext::new();
        let in_p = stack_input(&mut tc, 4, 8);
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn foo:
                <foo_entry @stack_10000004:i64>
                    %c = load(i8, {r0});
                    if %c goto <wr> else goto <skip>;
                <wr>
                    store(@stack_10000004, i32 7);
                    return [i64 0];
                <skip>
                    return [i64 1];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <foo>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = (g, r0, wr, skip);
        Function::from_id_mut(&mut tc.ctx, foo).set_input_regs(vec![in_p]);
        Function::from_id_mut(&mut tc.ctx, foo).set_pure_reg(true);
        let a = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, foo, vec![a]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(
            argpromote(&mut tc.ctx),
            "a path-dependent write is promotable now that write targets are seeded"
        );
        // The caller replays the one surfaced write; on the skip arm the callee
        // returns the seeded initial value, so that replay is a no-op.
        assert_eq!(replayed_stores(&tc, call_id), 1);
        // Idempotent: re-running redirects nothing new (the stores are already in
        // shadow), so the return's write-set is neither rebuilt nor duplicated — a
        // second append would create duplicate `write*` field names and panic.
        assert!(
            !argpromote(&mut tc.ctx),
            "re-running shadow promotion must be a no-op (idempotence)"
        );
    }

    // ---- Phase 1: register channel — callee-side scan + rewrite -------------

    /// The width of the flat register write-set on `fid`'s (first) return: the
    /// field count of the `Tuple` in `Return::value`, or `None` if unset.
    fn register_writeset_len(tc: &qcode::testing::TestContext, fid: FunctionId) -> Option<usize> {
        let ret = Function::from_id(&tc.ctx, fid).iter().find_map(|b| {
            let last = b.iter().last()?;
            matches!(last.mnemonic(), Mnemonic::Return(_)).then_some(last.id)
        })?;
        let val = match tc.ctx.get_insn(ret).mnemonic() {
            Mnemonic::Return(r) => r.value?,
            _ => return None,
        };
        let ValueId::Instruction(tuple_id) = val else {
            return None;
        };
        match tc.ctx.get_insn(tuple_id).mnemonic() {
            Mnemonic::Tuple(t) => Some(t.fields.len()),
            _ => None,
        }
    }

    /// `void f() { r0 = 42; }` — a pure clobber: no inputs, one output, and a
    /// one-field write-set. The return register has no special status.
    #[test]
    fn register_pure_clobber() {
        let mut tc = qcode::testing::TestContext::new();
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry>
                    store({r0}, i64 42);
                    return [i64 0];
            "
        );
        let _ = entry;
        let eff = scan_register_effects(&tc.ctx, f).expect("a written register is promotable");
        assert_eq!(eff.outputs, vec![r0]);
        // The output is over-approximated as an input too (seeded so a no-write
        // path reads the caller's incoming value); the seed is dead here and DCE
        // would prune it.
        assert_eq!(eff.inputs, vec![r0]);

        rewrite_registers(&mut tc.ctx, f, &eff);
        assert_eq!(
            register_writeset_len(&tc, f),
            Some(1),
            "one output → one field"
        );
    }

    /// `r0 += 1` — read-modify-write: r0 is both input and output. The input
    /// becomes a by-value param seeded into register space at entry.
    #[test]
    fn register_read_modify_write() {
        let mut tc = qcode::testing::TestContext::new();
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry>
                    %v = load(i64, {r0});
                    %s = %v + i64 1;
                    store({r0}, %s);
                    return [i64 0];
            "
        );
        let eff = scan_register_effects(&tc.ctx, f).expect("written register");
        assert_eq!(eff.inputs, vec![r0]);
        assert_eq!(eff.outputs, vec![r0]);

        rewrite_registers(&mut tc.ctx, f, &eff);
        // One by-value input param added, and its seed store is the new entry head.
        assert_eq!(BasicBlock::from_id(&tc.ctx, entry).params().count(), 1);
        let first = BasicBlock::from_id(&tc.ctx, entry).iter().next().unwrap();
        assert!(
            matches!(first.mnemonic(), Mnemonic::Store(s) if s.ptr == ValueId::Varnode(r0)),
            "entry must begin by seeding r0 from its input param"
        );
        assert_eq!(register_writeset_len(&tc, f), Some(1));
    }

    /// Writes to `r0_lo32` and `r0` overlap; the write-set canonicalizes to the
    /// coarsest covering register (`r0`), so replay stays order-independent.
    #[test]
    fn register_overlap_canonicalizes_to_coarsest() {
        let mut tc = qcode::testing::TestContext::new();
        let (r0, r0_lo32) = (tc.r0, tc.r0_lo32);
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry>
                    store({r0_lo32}, i32 2);
                    store({r0}, i64 3);
                    return [i64 0];
            "
        );
        let eff = scan_register_effects(&tc.ctx, f).expect("written register");
        assert_eq!(
            eff.outputs,
            vec![r0],
            "overlap group collapses to the 8-byte r0"
        );
        let _ = r0_lo32;
        rewrite_registers(&mut tc.ctx, f, &eff);
        assert_eq!(register_writeset_len(&tc, f), Some(1));
    }

    /// `r0` (the return register) and `r1` (a clobber) are written. Both ride the
    /// single write-set — the return value is just another output register.
    #[test]
    fn register_return_value_folds_into_writeset() {
        let mut tc = qcode::testing::TestContext::new();
        let (r0, r1) = (tc.r0, tc.r1);
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry>
                    store({r0}, i64 7);
                    store({r1}, i64 8);
                    return [i64 0];
            "
        );
        let eff = scan_register_effects(&tc.ctx, f).expect("written register");
        assert_eq!(eff.outputs, vec![r0, r1], "both outputs, sorted by address");
        rewrite_registers(&mut tc.ctx, f, &eff);
        assert_eq!(
            register_writeset_len(&tc, f),
            Some(2),
            "return reg folds in"
        );
    }

    /// A function that only *reads* registers has nothing to functionalize.
    #[test]
    fn register_read_only_is_skipped() {
        let mut tc = qcode::testing::TestContext::new();
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry>
                    %v = load(i64, {r0});
                    return [i64 0];
            "
        );
        assert!(
            scan_register_effects(&tc.ctx, f).unwrap_err() == RegPurityReason::NoRegisterWrites,
            "no write ⇒ nothing to promote"
        );
    }

    // ---- Phase 2: register channel — caller replay + precision --------------

    /// `void f() { r0 = 42; }` called by `g`, which reads r0 afterwards. The
    /// caller replays the returned r0, and the following mem2reg forwards that
    /// replay into the post-call read — precise dataflow, not an opaque reload.
    #[test]
    fn register_caller_replays_output_precisely() {
        let mut tc = qcode::testing::TestContext::new();
        let (r0, r1) = (tc.r0, tc.r1);
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry>
                    store({r0}, i64 42);
                    return [i64 0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    %x = load(i64, {r0});
                    store({r1}, %x);
                    return [i64 0];
            "
        );
        let _ = (f, g, r0);
        set_call(&mut tc, g_call, f, vec![]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(
            argpromote_registers(&mut tc.ctx),
            "f's register clobber should be promoted"
        );

        // The continuation replays the returned register from the call result.
        let has_extract = BasicBlock::from_id(&tc.ctx, g_cont)
            .iter()
            .any(|i| matches!(i.mnemonic(), Mnemonic::Extract(_)));
        assert!(
            has_extract,
            "continuation must extract the returned register value"
        );

        // Precision argpromote delivers: the replay writes the callee's *precise*
        // returned register value — an `Extract` of the call result — straight back into
        // r0. (Forwarding that replay into the subsequent `load(r0)` would additionally
        // require the *caller* to be functionalized/seeded; a conventional no-caller
        // function like `g` keeps the read as a plain load, since mem2reg no longer
        // promotes a live-in register — the call interface is argpromote's.)
        let _ = r1;
        let replays_extract = BasicBlock::from_id(&tc.ctx, g_cont).iter().any(|i| {
            matches!(
                i.mnemonic(),
                Mnemonic::Store(s)
                    if s.ptr == ValueId::Varnode(r0)
                        && matches!(s.src, ValueId::Instruction(id)
                            if matches!(tc.ctx.get_insn(id).mnemonic(), Mnemonic::Extract(_)))
            )
        });
        assert!(
            replays_extract,
            "the caller must replay the precise extracted r0 value back into r0"
        );
    }

    /// An RMW callee's input register is passed as a positional call argument.
    #[test]
    fn register_caller_passes_input_arg() {
        let mut tc = qcode::testing::TestContext::new();
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry>
                    %v = load(i64, {r0});
                    %s = %v + i64 1;
                    store({r0}, %s);
                    return [i64 0];

            fn g:
                <g_entry>
                    store({r0}, i64 10);
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = (f, g, r0);
        let call_id = set_call(&mut tc, g_call, f, vec![]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(argpromote_registers(&mut tc.ctx));

        // f gained exactly one by-value input param (r0).
        let root = Function::from_id(&tc.ctx, f).root().unwrap().id;
        assert_eq!(BasicBlock::from_id(&tc.ctx, root).params().count(), 1);

        // The call passes that input as one positional argument.
        let args_len = match tc.ctx.get_insn(call_id).mnemonic() {
            Mnemonic::Call(c) => c.args.len(),
            _ => unreachable!(),
        };
        assert_eq!(
            args_len, 1,
            "the input register is passed as one call argument"
        );
    }

    /// Every return block gets its own write-set (the design's per-return rule).
    #[test]
    fn register_multiple_returns_each_get_writeset() {
        let mut tc = qcode::testing::TestContext::new();
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry>
                    store({r0}, i64 1);
                    return [i64 0];
                <other>
                    store({r0}, i64 2);
                    return [i64 0];
            "
        );
        let _ = (r0, entry, other);
        let eff = scan_register_effects(&tc.ctx, f).expect("written register");
        rewrite_registers(&mut tc.ctx, f, &eff);
        let with_writeset = Function::from_id(&tc.ctx, f)
            .iter()
            .filter(|b| {
                b.iter().last().is_some_and(
                    |i| matches!(i.mnemonic(), Mnemonic::Return(r) if r.value.is_some()),
                )
            })
            .count();
        assert_eq!(
            with_writeset, 2,
            "every return carries the register write-set"
        );
    }

    /// Phase 3: emulator round-trip. The rewritten module must produce the same
    /// caller-visible register state as the original. `g` calls an RMW `f`
    /// (`r0 += 1`); we pin `f`'s return at `g`'s continuation address so the
    /// nested return works, then emulate before and after the rewrite and compare
    /// `r0`. This exercises the whole functional path: input-param seeding, the
    /// returned aggregate (`Return.value`), and the caller's `extract`/replay.
    #[test]
    fn register_roundtrip_preserves_caller_visible_state() {
        use qcode_emulator::StandaloneEmulator;

        let mut tc = qcode::testing::TestContext::new();
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry>
                    %v = load(i64, {r0});
                    %s = %v + i64 1;
                    store({r0}, %s);
                    return [i64 4096];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = (f, r0);
        // `f` returns to `g`'s continuation: pin it at the return address (4096).
        BasicBlock::from_id_mut(&mut tc.ctx, g_cont)
            .set_address(4096)
            .unwrap();
        set_call(&mut tc, g_call, f, vec![]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        let g_root = Function::from_id(&tc.ctx, g).root().unwrap().id;
        let run = |ctx: &Context<'_>| -> u64 {
            let mut emu = StandaloneEmulator::new(g_root);
            emu.set_varnode(ctx, r0, 10).unwrap();
            emu.run_function(ctx, g).unwrap();
            emu.read_varnode(ctx, r0).unwrap()
        };

        let original = run(&tc.ctx);
        assert!(argpromote_registers(&mut tc.ctx));
        let transformed = run(&tc.ctx);

        assert_eq!(original, 11, "original: r0 = 10 + 1");
        assert_eq!(transformed, original, "rewrite preserves caller-visible r0");
    }

    /// Two registers that overlap but where neither contains the other (no single
    /// covering register) leave the function on the conservative path.
    #[test]
    fn register_partial_overlap_no_cover_bails() {
        let mut tc = qcode::testing::TestContext::new();
        let r0_lo32 = tc.r0_lo32; // bytes [0, 4)
        // A 4-byte register at offset 2 → bytes [2, 6): overlaps r0_lo32 at [2,4)
        // but neither interval contains the other.
        let mid = Varnode::make(&mut tc.ctx, 2, 4, tc.reg_space).id;
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry>
                    store({r0_lo32}, i32 1);
                    store({mid}, i32 2);
                    return [i64 0];
            "
        );
        let _ = (r0_lo32, mid, entry);
        assert!(
            scan_register_effects(&tc.ctx, f).unwrap_err()
                == RegPurityReason::NonCanonicalRegisters,
            "partial overlap with no covering register must bail"
        );
    }

    /// `mark_pure_functions` asserts `is_pure` on a `pure_reg` function with a
    /// side-effect-free body, but not on one that still stores to memory.
    #[test]
    fn mark_pure_flags_only_fully_pure_bodies() {
        let mut tc = qcode::testing::TestContext::new();

        // A clean function: arithmetic over a param, returned through a tuple.
        let clean = Function::make(&mut tc.ctx, "clean".into()).unwrap().id;
        let clean_entry = tc.ctx.get_or_make_block(0x1000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, clean);
            f.set_root(clean_entry).unwrap();
            f.add_block(clean_entry);
        }
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, clean_entry));
            let a = b.push_param(8).id();
            let c = b.context_mut().get_const(1, 8).id();
            let s = b.push_add(a, c).id();
            let _ = b.push_tuple(vec![s]).id();
            let ptr = b.context_mut().get_const(0x2000, 8).id();
            b.push_return(ptr);
            unsafe { b.dont_finalize() };
        }
        Function::from_id_mut(&mut tc.ctx, clean).set_pure_reg(true);

        // A function that still loads from memory (an untracked value source).
        let dirty = Function::make(&mut tc.ctx, "dirty".into()).unwrap().id;
        let dirty_entry = tc.ctx.get_or_make_block(0x3000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, dirty);
            f.set_root(dirty_entry).unwrap();
            f.add_block(dirty_entry);
        }
        let (r0, reg_space) = (tc.r0, tc.reg_space);
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, dirty_entry));
            let _ = b
                .push_load::<false>(ValueId::Varnode(r0), 8, reg_space)
                .id();
            let ptr = b.context_mut().get_const(0x4000, 8).id();
            b.push_return(ptr);
            unsafe { b.dont_finalize() };
        }
        Function::from_id_mut(&mut tc.ctx, dirty).set_pure_reg(true);

        assert!(
            mark_pure_functions(&mut tc.ctx),
            "the clean function is newly pure"
        );
        assert!(Function::from_id(&tc.ctx, clean).is_pure());
        assert!(
            !Function::from_id(&tc.ctx, dirty).is_pure(),
            "a function with a residual load must not be marked pure"
        );

        // Idempotent: a second run flags nothing new.
        assert!(!mark_pure_functions(&mut tc.ctx));
    }

    /// A dynamic-index buffer loop — `for i in 0..20 { buf[i] += 1 }`, where `buf`
    /// is an incoming pointer arg — is the motivating case for the region path. The
    /// induction variable is bounded to `[0,20)` by the loop guard, so the whole
    /// `[buf, buf+20)` span snapshots as ONE `Array` input and returns as ONE wide
    /// write-set entry. Step-1 guarantee: **no real-ram access survives** (every
    /// load/store redirected into shadow), the function is functionalized.
    #[test]
    fn dynamic_index_loop_promotes_as_array_region() {
        let mut tc = qcode::testing::TestContext::new();
        // The buffer *pointer* is f's single stack-passed input.
        let input = stack_input(&mut tc, 4, 8);

        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64>
                    goto <f_head @i=i64 0x0>;
                <f_head @i:i64>
                    %c = @i < i64 0x14;
                    if %c goto <f_body> else goto <f_exit>;
                <f_body>
                    %addr = @stack_10000004 + @i;
                    %b = load(i8, %addr);
                    %nb = %b + i8 0x1;
                    store(%addr, %nb);
                    %ni = @i + i64 0x1;
                    goto <f_head @i=%ni>;
                <f_exit>
                    return [i64 0x0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = (g, f_head, f_body, f_exit);

        Function::from_id_mut(&mut tc.ctx, f).set_input_regs(vec![input]);
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let ptr = tc.ctx.get_const(0x4000, 8).id();
        set_call(&mut tc, g_call, f, vec![ptr]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(
            argpromote(&mut tc.ctx),
            "a bounded dynamic-index buffer loop should region-promote"
        );

        // Step-1 guarantee: no access into the real (default) space remains —
        // every load/store was redirected into the shadow.
        let ram = tc.ctx.default_space;
        let real_access = Function::from_id(&tc.ctx, f).iter().any(|blk| {
            blk.iter().any(|i| match i.mnemonic() {
                Mnemonic::Load(l) => l.space == ram,
                Mnemonic::Store(s) => s.space == ram,
                _ => false,
            })
        });
        assert!(
            !real_access,
            "no real-ram access may survive: the buffer is fully functionalized"
        );

        // The callee gains one `Array`-typed by-value snapshot param for the region.
        let has_array_param = Function::from_id(&tc.ctx, f).root().is_some_and(|b| {
            b.params()
                .any(|p| tc.ctx.types.array_of(p.type_id()).is_some())
        });
        assert!(
            has_array_param,
            "callee must gain one Array-typed region snapshot param"
        );

        // The written region rides back out through the return write-set.
        let returns_value = Function::from_id(&tc.ctx, f).iter().any(|b| {
            b.iter()
                .last()
                .is_some_and(|i| matches!(i.mnemonic(), Mnemonic::Return(r) if r.value.is_some()))
        });
        assert!(
            returns_value,
            "the written region must ride out in the write-set"
        );

        // The caller snapshots the region (a wide load) before the call.
        let has_load = BasicBlock::from_id(&tc.ctx, g_call)
            .iter()
            .any(|i| matches!(i.mnemonic(), Mnemonic::Load(_)));
        assert!(has_load, "caller must snapshot the region before the call");

        // The functionalized callee is now pure: its only residual loads are into
        // the private shadow space (seeded from the Array input), which `mark_pure`
        // treats as deterministic. The loop survives, but no caller-visible effect.
        assert!(
            mark_pure_functions(&mut tc.ctx),
            "the region-promoted function should be marked pure"
        );
        assert!(
            Function::from_id(&tc.ctx, f).is_pure(),
            "f is a deterministic function of its Array input"
        );

        // And the independent verifier agrees — a shadow loop is not a violation.
        let violations = crate::verify::verify_pure_functions(&tc.ctx);
        assert!(
            violations.iter().all(|v| v.function != f),
            "verifier must not flag the shadow loop as impure: {violations:?}"
        );
    }

    /// The real-world shape: the buffer pointer is spilled to `[ESP+4]` and
    /// **reloaded inside the loop**. Under the frame assumptions
    /// (`ArgsDisjointFromCallerFrame` + `LoadedPointerDisjointFromSlot`), GVN
    /// forwards that reload to the by-value pointer param — turning the deref base
    /// into a clean param, which the region path then promotes (covered by
    /// `dynamic_index_loop_becomes_map`). This was the bug: without the
    /// loaded-pointer assumption the reload stays opaque and forwarding is blocked
    /// by the loop's byte store, so the buffer never promotes.
    #[test]
    fn spilled_reloaded_base_forwards_under_assumption() {
        use qcode::assumption::Proposition;

        let mut tc = qcode::testing::TestContext::new();
        let sp_reg = tc.r0;

        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @esp:i64 @bufptr:i64>
                    %slot = @esp + i64 0x4;
                    store(%slot, @bufptr);
                    goto <f_head @i=i64 0x0>;
                <f_head @i:i64>
                    %c = @i < i64 0x14;
                    if %c goto <f_body> else goto <f_exit>;
                <f_body>
                    %cc = load(i64, %slot);
                    %addr = %cc + @i;
                    %b = load(i8, %addr);
                    %nb = %b + i8 0x1;
                    store(%addr, %nb);
                    %ni = @i + i64 0x1;
                    goto <f_head @i=%ni>;
                <f_exit>
                    return [i64 0x0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = (g, f_head, f_body, f_exit);

        // Mark @esp as the stack pointer so frame freshness activates.
        let esp_pid = BasicBlock::from_id(&tc.ctx, f_entry)
            .params()
            .next()
            .unwrap()
            .id();
        if let ValueId::BlockParam(inner) = esp_pid {
            tc.ctx.values.block_params[inner].origin = Some(ValueId::Varnode(sp_reg));
        }
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let espv = tc.ctx.get_const(0x7000, 8).id();
        let bufp = tc.ctx.get_const(0x9000, 8).id();
        set_call(&mut tc, g_call, f, vec![espv, bufp]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        // Record both frame assumptions (as the pipeline's assume_arg_frame does),
        // then forward with a frame-fresh oracle.
        tc.ctx
            .assume_true(Proposition::ArgsDisjointFromCallerFrame(f));
        tc.ctx
            .assume_true(Proposition::LoadedPointerDisjointFromSlot(f));
        let aliases =
            crate::AliasResult::simple(&tc.ctx).with_frame_freshness(&tc.ctx, f, Some(sp_reg));
        crate::gvn::gvn_function(&mut tc.ctx, f, Some(&aliases));

        // The in-loop reload of the buffer pointer is forwarded away: no load of
        // `%slot` survives in the body (the base is now the by-value @bufptr).
        let slot_reload = Function::from_id(&tc.ctx, f).iter().any(|blk| {
            blk.iter().any(|i| {
                matches!(i.mnemonic(), Mnemonic::Load(l)
                    if l.size == 8 && l.space == tc.ctx.default_space)
            })
        });
        assert!(
            !slot_reload,
            "the spilled buffer-pointer reload must be forwarded to @bufptr"
        );

        // The deref base is now the by-value pointer param directly.
        let base_is_param = Function::from_id(&tc.ctx, f).iter().any(|b| {
            b.iter().any(|i| matches!(i.mnemonic(), Mnemonic::Binop(bi)
                if matches!(bi.op, Binop::Int(IntBinop::Add)) && matches!(bi.lhs, ValueId::BlockParam(_))))
        });
        assert!(base_is_param, "the buffer deref base is now a clean param");
    }

    /// The user's real post-forwarding shape: the buffer base is now the by-value
    /// pointer param `@ESP_val_4`, but argpromote's own dead spilled-arg seed stores
    /// to `[ESP]`/`[ESP+4]` (caller-frame slots) still sit in the entry. The region
    /// is through an incoming pointer, disjoint from the whole frame, so those frame
    /// writes must not block region promotion (the relaxed `regions_disjoint`).
    #[test]
    fn region_promotes_with_coexisting_frame_seed_stores() {
        use qcode::assumption::Proposition;

        let mut tc = qcode::testing::TestContext::new();
        let sp_reg = tc.r0;

        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @esp:i64 @esp_val_0:i64 @esp_val_4:i64>
                    store(@esp, @esp_val_0);
                    %slot = @esp + i64 0x4;
                    store(%slot, @esp_val_4);
                    goto <f_head @i=i64 0x0>;
                <f_head @i:i64>
                    %c = @i < i64 0x14;
                    if %c goto <f_body> else goto <f_exit>;
                <f_body>
                    %addr = @i + @esp_val_4;
                    %b = load(i8, %addr);
                    %nb = %b + i8 0x1;
                    store(%addr, %nb);
                    %ni = @i + i64 0x1;
                    goto <f_head @i=%ni>;
                <f_exit>
                    %r = pack(EAX=@esp_val_4);
                    return [@esp_val_0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = (g, f_head, f_body, f_exit, f_entry);

        let esp_pid = BasicBlock::from_id(&tc.ctx, f_entry)
            .params()
            .next()
            .unwrap()
            .id();
        if let ValueId::BlockParam(inner) = esp_pid {
            tc.ctx.values.block_params[inner].origin = Some(ValueId::Varnode(sp_reg));
        }
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let espv = tc.ctx.get_const(0x7000, 8).id();
        let v0 = tc.ctx.get_const(0x10, 8).id();
        let v4 = tc.ctx.get_const(0x9000, 8).id();
        set_call(&mut tc, g_call, f, vec![espv, v0, v4]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        tc.ctx
            .assume_true(Proposition::ArgsDisjointFromCallerFrame(f));

        assert!(
            argpromote_with_sp(&mut tc.ctx, Some(sp_reg)),
            "region must promote despite the caller-frame seed stores"
        );
        // The buffer region snapshots as an Array param.
        let has_array_param = Function::from_id(&tc.ctx, f).root().is_some_and(|b| {
            b.params()
                .any(|p| tc.ctx.types.array_of(p.type_id()).is_some())
        });
        assert!(has_array_param, "the buffer region became an Array input");

        mark_pure_functions(&mut tc.ctx);
        assert!(
            crate::calls::loop_to_map::recognize_total_maps(&mut tc.ctx),
            "the loop should be recognized as a map"
        );
        let has_map = Function::from_id(&tc.ctx, f)
            .iter()
            .any(|b| b.iter().any(|i| matches!(i.mnemonic(), Mnemonic::Map(_))));
        assert!(has_map, "f must contain a map");
    }

    /// End to end: step-1 region promotion then the loop-to-map recognizer turns
    /// the buffer loop's returned write-set value into `map(body_fn, arr)` — the
    /// projectable form. The body is outlined into a fresh pure function.
    #[test]
    fn dynamic_index_loop_becomes_map() {
        let mut tc = qcode::testing::TestContext::new();
        let input = stack_input(&mut tc, 4, 8);

        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64>
                    goto <f_head @i=i64 0x0>;
                <f_head @i:i64>
                    %c = @i < i64 0x14;
                    if %c goto <f_body> else goto <f_exit>;
                <f_body>
                    %addr = @stack_10000004 + @i;
                    %b = load(i8, %addr);
                    %nb = %b + i8 0x1;
                    store(%addr, %nb);
                    %ni = @i + i64 0x1;
                    goto <f_head @i=%ni>;
                <f_exit>
                    return [i64 0x0];

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return [i64 0];
            "
        );
        let _ = (g, f_head, f_body, f_exit);

        Function::from_id_mut(&mut tc.ctx, f).set_input_regs(vec![input]);
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let ptr = tc.ctx.get_const(0x4000, 8).id();
        set_call(&mut tc, g_call, f, vec![ptr]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(argpromote(&mut tc.ctx), "step 1 region promotion");
        mark_pure_functions(&mut tc.ctx);
        assert!(
            crate::calls::loop_to_map::recognize_total_maps(&mut tc.ctx),
            "the total-map loop should be recognized"
        );

        // f now contains a `map` whose source is the Array snapshot param.
        let map_src = Function::from_id(&tc.ctx, f).iter().find_map(|b| {
            b.iter().find_map(|i| match i.mnemonic() {
                Mnemonic::Map(m) => Some(m.src),
                _ => None,
            })
        });
        let map_src = map_src.expect("f must contain a map");
        assert!(
            tc.ctx
                .stored_type_of(map_src)
                .and_then(|t| tc.ctx.types.array_of(t))
                .is_some(),
            "the map source is the Array snapshot"
        );

        // The returned write-set value is now the map result (the wide shadow
        // reload was forwarded away).
        let map_val = Function::from_id(&tc.ctx, f)
            .iter()
            .find_map(|b| {
                b.iter().find_map(|i| match i.mnemonic() {
                    Mnemonic::Map(_) => Some(ValueId::Instruction(i.id)),
                    _ => None,
                })
            })
            .unwrap();
        let writeset_uses_map = Function::from_id(&tc.ctx, f).iter().any(|b| {
            b.iter()
                .any(|i| matches!(i.mnemonic(), Mnemonic::Tuple(t) if t.fields.contains(&map_val)))
        });
        assert!(writeset_uses_map, "write-set value must be the map result");

        // The outlined body is a fresh pure function.
        let has_body = tc
            .ctx
            .function_ids()
            .into_iter()
            .any(|fid| Function::from_id(&tc.ctx, fid).name().contains("_map_body"));
        assert!(has_body, "the per-element body was outlined");

        // The dead loop is gone: only entry + exit remain, and no shadow access
        // (seed store, RMW, or reload) survives.
        assert_eq!(
            Function::from_id(&tc.ctx, f).iter().count(),
            2,
            "loop blocks deleted, leaving entry + exit"
        );
        let any_mem = Function::from_id(&tc.ctx, f).iter().any(|b| {
            b.iter()
                .any(|i| matches!(i.mnemonic(), Mnemonic::Load(_) | Mnemonic::Store(_)))
        });
        assert!(!any_mem, "no shadow load/store should remain in f");

        // The rewritten IR verifies clean (no dangling refs, valid terminators).
        let problems = crate::verify::verify(&tc.ctx);
        assert!(
            problems.is_empty(),
            "post-rewrite IR must verify: {problems:?}"
        );
    }
}
