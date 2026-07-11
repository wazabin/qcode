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
                    %v = load(register:8, {r1});
                    %s = %v + i64 5;
                    store(register:8, {r0} <- %s);
                    %fin = load(register:8, {r0});
                    %agg = (%fin);
                    return at %agg;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
            "
        );
        let _ = (f, g, r0, r1);

        // A one-field aggregate (the positional register write-set shape).
        let i64_ty = tc.ctx.shared.types.get_or_make_int(8);
        let agg_ty = tc.ctx.shared.types.get_or_make_aggregate(vec![i64_ty]);

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
        let aliases = crate::AliasResult::simple_for_function(&tc.ctx, g);
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
                    %v = load(ram:4, @stack_10000004);
                    store(ram:4, @stack_10000004 <- %v);
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
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
                    %v = load(ram:4, @stack_10000004);
                    store(ram:4, @stack_10000004 <- %v);
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
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
                    return at i64 0;

            fn caller:
                <f_entry>
                    goto <f_call>;
                <f_call>
                    call <callee>;
                <f_cont>
                    return at i64 0;
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

    /// A constant real-ram address used as a load/store base is lifted to a
    /// `glob_<addr>` parameter: the access is rewritten to dereference the param,
    /// no constant-address access remains, and every direct caller passes the
    /// address literal as the new argument.
    #[test]
    fn lifts_constant_global_address_to_param() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry>
                    %p = load(ram:4, 0x454df8);
                    store(ram:4, 0x454df8 <- i32 0x270);
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let call_id = set_call(&mut tc, g_call, f, vec![]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        let address_taken = super::super::address_taken_set(&tc.ctx);
        assert!(
            super::super::globals::globalize_constants(&mut tc.ctx, &address_taken, f),
            "constant global address must be lifted to a param"
        );

        // f gains a `glob_454df8` param.
        let pnames: Vec<String> = Function::from_id(&tc.ctx, f)
            .root()
            .unwrap()
            .params()
            .filter_map(|p| p.name().map(str::to_string))
            .collect();
        assert!(
            pnames.iter().any(|n| n == "glob_454df8"),
            "glob param added: {pnames:?}"
        );

        // No constant-address real-ram access remains in f's body.
        let const_access = Function::from_id(&tc.ctx, f)
            .iter()
            .flat_map(|b| b.iter())
            .any(|i| match i.mnemonic() {
                Mnemonic::Load(l) => matches!(l.ptr, ValueId::Literal(_)),
                Mnemonic::Store(s) => matches!(s.ptr, ValueId::Literal(_)),
                _ => false,
            });
        assert!(
            !const_access,
            "every constant-address access is rewritten to deref the param"
        );

        // The caller threads the address literal as the new last argument.
        let Mnemonic::Call(call) = tc.ctx.get_insn(call_id).mnemonic().clone() else {
            panic!("g_call is a call");
        };
        assert_eq!(call.args.len(), 1, "address threaded to the caller");
        assert!(
            matches!(call.args[0], ValueId::Literal(_)),
            "caller passes the address literal"
        );
    }

    /// An address-taken function must not be globalized: an indirect caller this
    /// pass cannot rewrite would be left without the new argument.
    #[test]
    fn skips_address_taken_function_for_globals() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry>
                    %p = load(ram:4, 0x454df8);
                    return at i64 0;

            fn g:
                <g_entry>
                    return at i64 0;
            "
        );
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);

        // Take f's address: store the function value somewhere in g, so f lands
        // in `address_taken_set`.
        let addr = tc.ctx.get_const(0x9000, 8).id();
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, g_entry));
            b.set_insert_point_to_start();
            b.push_store(ValueId::Function(f), addr, tc.reg_space);
        }

        let address_taken = super::super::address_taken_set(&tc.ctx);
        assert!(
            address_taken.contains(&f),
            "test setup: f must be address-taken"
        );
        assert!(
            !super::super::globals::globalize_constants(&mut tc.ctx, &address_taken, f),
            "address-taken function must be left untouched"
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
                    %addr = load(ram:8, @stack_10000004);
                    store(ram:4, %addr <- i32 0);
                    %a = @stack_10000004 + i64 0x30;
                    %v = load(ram:4, %a);
                    return at %v;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
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
        let ram = tc.ctx.shared.default_space;
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

    /// End-to-end globalize: a store to a *constant* global address used to be an
    /// unmodellable access that forced partial (inputs-only) mode. Globalize now
    /// lifts `0x9000` into a `glob_9000` param, so the access is param-relative and
    /// the RAM channel fully functionalizes the write into the returned write-set.
    /// The caller passes the address literal in and replays the global write.
    #[test]
    fn global_store_is_functionalized_and_replayed_at_caller() {
        let mut tc = qcode::testing::TestContext::new();
        let input = stack_input(&mut tc, 4, 8);
        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64>
                    store(ram:4, @stack_10000004 <- i32 0x41);
                    store(ram:8, i64 0x9000 <- @stack_10000004);
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
            "
        );
        let _ = g;
        Function::from_id_mut(&mut tc.ctx, f).set_input_regs(vec![input]);
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let ptr = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, f, vec![ptr]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(argpromote(&mut tc.ctx), "the global write functionalizes");

        // f gained a `glob_9000` param for the lifted constant address, and no
        // constant-address access remains in its body.
        let pnames: Vec<String> = Function::from_id(&tc.ctx, f)
            .root()
            .unwrap()
            .params()
            .filter_map(|p| p.name().map(str::to_string))
            .collect();
        assert!(
            pnames.iter().any(|n| n == "glob_9000"),
            "glob param added: {pnames:?}"
        );
        let const_access = Function::from_id(&tc.ctx, f)
            .iter()
            .flat_map(|b| b.iter())
            .any(|i| match i.mnemonic() {
                Mnemonic::Load(l) => matches!(l.ptr, ValueId::Literal(_)),
                Mnemonic::Store(s) => matches!(s.ptr, ValueId::Literal(_)),
                _ => false,
            });
        assert!(!const_access, "no constant-address access remains in f");

        // The caller threads the address literal `0x9000` as an argument …
        let Mnemonic::Call(call) = tc.ctx.get_insn(call_id).mnemonic().clone() else {
            panic!("g_call is a call");
        };
        let passes_addr =
            call.args
                .iter()
                .any(|&a| match qcode::value::ValueRef::new(a, &tc.ctx) {
                    qcode::value::ValueRef::Literal(l) => l.value() == 0x9000,
                    _ => false,
                });
        assert!(passes_addr, "caller passes the global address literal");

        // … and replays the functionalized write-set out of the call result.
        let has_replay = Function::from_id(&tc.ctx, g).iter().any(|b| {
            b.iter().any(
                |i| matches!(i.mnemonic(), Mnemonic::Extract(e) if e.agg == ValueId::Instruction(call_id)),
            )
        });
        assert!(has_replay, "caller replays the returned global write-set");
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
                    store(ram:4, %s <- i32 0);
                    %a = @stack_10000004 + i64 0x30;
                    %v = load(ram:4, %a);
                    return at %v;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
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
                    %v = load(ram:4, @stack_10000004);
                    store(ram:4, @stack_10000004 <- %v);
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
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
                    %v = load(ram:4, %a);
                    return at %v;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
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

    /// Reproduction of the TEB/PEB bug: a read through a **`gep`** field access
    /// (`load(gep(p.field))`) — the form the struct-typing pass produces — must
    /// promote to a by-value snapshot exactly like the equivalent `load(p + off)`
    /// add form. Otherwise the load is redirected into shadow with no seed and
    /// reads garbage.
    #[test]
    fn promotes_read_through_gep_field() {
        use qcode::types::AggregateField;
        let mut tc = qcode::testing::TestContext::new();
        let input = stack_input(&mut tc, 4, 8);

        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64>
                    %a = @stack_10000004 + i64 0x30;
                    %v = load(ram:4, %a);
                    return at %v;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
            "
        );
        let _ = g;

        // Type the param as a struct pointer with a field at 0x30 and rewrite the
        // `param + 0x30` add into the `gep(param.field)` form the typing pass emits.
        let i32_ty = tc.ctx.shared.types.get_or_make_int(4);
        let s_ty = tc.ctx.shared.types.get_or_make_struct(
            "S",
            0x34,
            vec![AggregateField::new_at("peb", i32_ty, 0x30)],
        );
        let ptr_ty = tc.ctx.shared.types.get_or_make_struct_pointer(8, s_ty);
        let root = Function::from_id(&tc.ctx, f).root().unwrap().id;
        let pid = BasicBlock::from_id(&tc.ctx, root)
            .params()
            .next()
            .unwrap()
            .id();
        let param = pid;
        if let ValueId::BlockParam(bp) = param {
            tc.ctx.block_param_mut(bp).type_id = ptr_ty;
        }
        // Find the add and its load, replace the add with a gep.
        let add_id = BasicBlock::from_id(&tc.ctx, root)
            .iter()
            .find(|i| matches!(i.mnemonic(), Mnemonic::Binop(b) if matches!(b.op, Binop::Int(IntBinop::Add))))
            .unwrap()
            .id;
        let gep = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, root));
            b.set_insert_point_before(add_id);
            b.push_gep(param, 0x30).id()
        };
        tc.ctx
            .replace_all_uses_with(ValueId::Instruction(add_id), gep);

        Function::from_id_mut(&mut tc.ctx, f).set_input_regs(vec![input]);
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let ptr = tc.ctx.get_const(0x4000, 8).id();
        set_call(&mut tc, g_call, f, vec![ptr]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(
            argpromote(&mut tc.ctx),
            "a 4-byte read through gep(p.field) at offset 0x30 should promote"
        );

        assert!(
            has_val_param(&tc.ctx, f),
            "the gep read must be exposed as a by-value snapshot param"
        );

        // The real-pipeline step that exposes the bug: GVN's memory forwarding must
        // forward the redirected *shadow* gep-load from the entry seed store (both
        // resolve to base @param + 0x30). Without gep-aware affine numbering the
        // load reads un-seeded shadow and survives — the reported PEB bug.
        let aliases = crate::AliasResult::simple_for_function(&tc.ctx, f);
        crate::gvn::gvn_function(&mut tc.ctx, f, Some(&aliases));

        let ram = tc.ctx.shared.default_space;
        let surviving_shadow_load = Function::from_id(&tc.ctx, f).iter().any(|blk| {
            blk.iter()
                .any(|i| matches!(i.mnemonic(), Mnemonic::Load(l) if l.space != ram))
        });
        assert!(
            !surviving_shadow_load,
            "the redirected shadow load must forward from the seed (read the snapshot \
             param), not survive reading un-seeded shadow"
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
                    return at 0x1000;
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
                    store(ram:4, @stack_10000010 <- i32 100);
                    %v = load(ram:4, @stack_10000004);
                    %s = %v + i32 1;
                    store(ram:4, @stack_10000004 <- %s);
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
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
                    %v = load(ram:4, @stack_10000004);
                    %s = %v + i32 10;
                    store(ram:4, @stack_10000004 <- %s);
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <foo>;
                <g_cont>
                    return at i64 0;
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
                    %v = load(ram:4, @stack_10000004);
                    %s = %v + i32 10;
                    store(ram:4, @stack_10000004 <- %s);
                    store(register:8, {r0} <- @stack_10000004);
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <foo>;
                <g_cont>
                    return at i64 0;
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
                    store(ram:4, %adv <- i32 5);
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <foo>;
                <g_cont>
                    return at i64 0;
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
                    %v = load(ram:4, @stack_10000004);
                    %s = %v + i32 10;
                    store(register:4, {r0} <- %s);
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <foo>;
                <g_cont>
                    return at i64 0;
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
                    store(register:8, {r0} <- %adv);
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <foo>;
                <g_cont>
                    return at i64 0;
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
                    %v = load(ram:4, @stack_10000004);
                    %s = %v + i32 1;
                    store(ram:4, @stack_10000004 <- %s);
                    store(register:4, {r0} <- i32 100);
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f1>;
                <g_cont>
                    return at i64 0;
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
                    store(ram:4, @stack_10000004 <- i32 5);
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
            "
        );
        let _ = g;
        // The shape the register/stack channels leave behind: a functionalized
        // function whose ABI register list is never filled in, with a one-field
        // register write-set already on `Return::value` (set as the register
        // channel does, since `return at ..` only sets the conventional operand).
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
                    store(ram:4, %loc <- i32 5);
                    store(ram:4, @p <- i32 9);
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
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
        tc.ctx.block_param_mut(pid).origin = Some(ValueId::Varnode(sp_vn));
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
                    store(ram:4, @stack_10000004 <- i32 5);
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
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
        let new_size = tc.ctx.shared.types.size_of(new_ty);
        let old_size = tc.ctx.shared.types.size_of(reg_ty);
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
                    %v = load(ram:4, @stack_10000004);
                    %s = %v + i32 1;
                    store(ram:4, @stack_10000004 <- %s);
                    %c = load(register:1, {r0});
                    if %c goto <ret_a> else goto <ret_b>;
                <ret_a>
                    return at i64 0;
                <ret_b>
                    return at i64 1;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <foo>;
                <g_cont>
                    return at i64 0;
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
                    %c = load(register:1, {r0});
                    if %c goto <wr> else goto <skip>;
                <wr>
                    store(ram:4, @stack_10000004 <- i32 7);
                    return at i64 0;
                <skip>
                    return at i64 1;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <foo>;
                <g_cont>
                    return at i64 0;
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
                    store(register:8, {r0} <- i64 42);
                    return at i64 0;
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
                    %v = load(register:8, {r0});
                    %s = %v + i64 1;
                    store(register:8, {r0} <- %s);
                    return at i64 0;
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
                    store(register:4, {r0_lo32} <- i32 2);
                    store(register:8, {r0} <- i64 3);
                    return at i64 0;
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
                    store(register:8, {r0} <- i64 7);
                    store(register:8, {r1} <- i64 8);
                    return at i64 0;
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
                    %v = load(register:8, {r0});
                    return at i64 0;
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
                    store(register:8, {r0} <- i64 42);
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    %x = load(register:8, {r0});
                    store(register:8, {r1} <- %x);
                    return at i64 0;
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
                    %v = load(register:8, {r0});
                    %s = %v + i64 1;
                    store(register:8, {r0} <- %s);
                    return at i64 0;

            fn g:
                <g_entry>
                    store(register:8, {r0} <- i64 10);
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
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
                    store(register:8, {r0} <- i64 1);
                    return at i64 0;
                <other>
                    store(register:8, {r0} <- i64 2);
                    return at i64 0;
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
                    %v = load(register:8, {r0});
                    %s = %v + i64 1;
                    store(register:8, {r0} <- %s);
                    return at i64 4096;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
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
                    store(register:4, {r0_lo32} <- i32 1);
                    store(register:4, {mid} <- i32 2);
                    return at i64 0;
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
        let clean_entry = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x1000, __f)
        };
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
        let dirty_entry = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x3000, __f)
        };
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
                    %b = load(ram:1, %addr);
                    %nb = %b + i8 0x1;
                    store(ram:1, %addr <- %nb);
                    %ni = @i + i64 0x1;
                    goto <f_head @i=%ni>;
                <f_exit>
                    return at i64 0x0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
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
        let ram = tc.ctx.shared.default_space;
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
                .any(|p| tc.ctx.shared.types.array_of(p.type_id()).is_some())
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

    /// A **strided, nested-offset** buffer write — `store((base + i*4) + 4)`, the
    /// shape an `i32[]` fill compiles to (element scale `*4`, a `+4` skipping a
    /// header field, with `base` buried two adds down) — must still region-promote.
    /// The structural one-add matcher missed this; the affine decomposition recovers
    /// `constant=4, terms=[(base,1),(i,4)]`, so the byte offset is `4 + 4·i` over the
    /// bounded `i ∈ [0,20)`, i.e. a `[i32; 20]` region at byte offset 4.
    #[test]
    fn strided_nested_offset_loop_promotes_as_array_region() {
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
                    %s = @i * i64 0x4;
                    %base = %s + @stack_10000004;
                    %addr = %base + i64 0x4;
                    store(ram:4, %addr <- i32 0x7b);
                    %ni = @i + i64 0x1;
                    goto <f_head @i=%ni>;
                <f_exit>
                    return at i64 0x0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
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
            "a strided nested-offset buffer loop should region-promote"
        );

        // The callee gains one `[i32; 20]` region snapshot param: element width 4
        // (the `*4` stride) and 20 elements (the `[0,20)` index bound).
        let array = Function::from_id(&tc.ctx, f)
            .root()
            .and_then(|b| b.params().find_map(|p| tc.ctx.shared.types.array_of(p.type_id())));
        let (elem_ty, count) = array.expect("callee must gain an Array region param");
        assert_eq!(
            tc.ctx.shared.types.size_of(elem_ty),
            4,
            "element width is the *4 stride"
        );
        assert_eq!(count, 20, "20 elements over the bounded index");

        // No real-ram access survives — the strided store is redirected into shadow.
        let ram = tc.ctx.shared.default_space;
        let real_access = Function::from_id(&tc.ctx, f).iter().any(|blk| {
            blk.iter().any(|i| match i.mnemonic() {
                Mnemonic::Load(l) => l.space == ram,
                Mnemonic::Store(s) => s.space == ram,
                _ => false,
            })
        });
        assert!(
            !real_access,
            "the strided buffer write must be functionalized"
        );
    }

    /// The **single-block self-loop** shape — the store and the loop guard live in
    /// one block (the index is that block's own param, and the guard tests the
    /// *incremented* value at the bottom). This is what a raw-lifted counted loop
    /// looks like before any rotation, e.g. the Mersenne-Twister state fill. The
    /// index range must still be recovered (`@i ∈ [0,20)` from the back-edge guard
    /// on `@i+1 < 20`) so the strided write region-promotes.
    #[test]
    fn single_block_self_loop_promotes_as_array_region() {
        let mut tc = qcode::testing::TestContext::new();
        let input = stack_input(&mut tc, 4, 8);

        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64>
                    goto <f_loop @i=i64 0x0>;
                <f_loop @i:i64>
                    %s = @i * i64 0x4;
                    %base = %s + @stack_10000004;
                    %addr = %base + i64 0x4;
                    store(ram:4, %addr <- i32 0x7b);
                    %ni = @i + i64 0x1;
                    %c = %ni < i64 0x14;
                    if %c goto <f_loop @i=%ni> else goto <f_exit>;
                <f_exit>
                    return at i64 0x0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
            "
        );
        let _ = (g, f_loop, f_exit);

        Function::from_id_mut(&mut tc.ctx, f).set_input_regs(vec![input]);
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let ptr = tc.ctx.get_const(0x4000, 8).id();
        set_call(&mut tc, g_call, f, vec![ptr]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(
            argpromote(&mut tc.ctx),
            "a single-block self-loop buffer fill should region-promote"
        );

        let has_array_param = Function::from_id(&tc.ctx, f).root().is_some_and(|b| {
            b.params()
                .any(|p| tc.ctx.shared.types.array_of(p.type_id()).is_some())
        });
        assert!(
            has_array_param,
            "callee must gain an Array region param for the self-loop fill"
        );
    }

    /// A written dynamic-index region may coexist with a scalar write to the
    /// *same base* at a constant offset that is disjoint from the region's span
    /// (e.g. a `count` field beside an array body). `regions_disjoint` now proves
    /// the two don't overlap by offset, so the function still takes the full shadow
    /// path and the region promotes to an `Array` — previously the extra non-frame
    /// scalar write forced partial mode and no `Array` param was minted.
    #[test]
    fn region_coexists_with_disjoint_scalar_write() {
        let mut tc = qcode::testing::TestContext::new();
        let input = stack_input(&mut tc, 4, 8);

        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64>
                    %said = @stack_10000004 + i64 0x40;
                    store(ram:4, %said <- i32 0xaa);
                    goto <f_head @i=i64 0x0>;
                <f_head @i:i64>
                    %c = @i < i64 0x14;
                    if %c goto <f_body> else goto <f_exit>;
                <f_body>
                    %addr = @stack_10000004 + @i;
                    %b = load(ram:1, %addr);
                    %nb = %b + i8 0x1;
                    store(ram:1, %addr <- %nb);
                    %ni = @i + i64 0x1;
                    goto <f_head @i=%ni>;
                <f_exit>
                    return at i64 0x0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
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
            "the region should still promote alongside a disjoint scalar write"
        );

        // Full shadow path: no real-ram access survives — including the scalar
        // write at offset 0x40, which is redirected into the shadow too.
        let ram = tc.ctx.shared.default_space;
        let real_access = Function::from_id(&tc.ctx, f).iter().any(|blk| {
            blk.iter().any(|i| match i.mnemonic() {
                Mnemonic::Load(l) => l.space == ram,
                Mnemonic::Store(s) => s.space == ram,
                _ => false,
            })
        });
        assert!(
            !real_access,
            "shadow path: the region and the disjoint scalar are both functionalized"
        );

        // The region still becomes an `Array`-typed by-value snapshot param.
        let has_array_param = Function::from_id(&tc.ctx, f).root().is_some_and(|b| {
            b.params()
                .any(|p| tc.ctx.shared.types.array_of(p.type_id()).is_some())
        });
        assert!(
            has_array_param,
            "the coexisting disjoint scalar must not block region→Array promotion"
        );
    }

    /// Regression for the TEB/PEB orphan bug: a bounded dynamic-index read whose
    /// span exceeds `MAX_REGION_BYTES` (4096) makes `RegionAcc::build()` return
    /// `None` — so NO `Array` snapshot is seeded for it. The folded access must then
    /// be left **unmodelled** (dropped), steering the function to partial mode, NOT
    /// committed to `accesses` where it would be redirected into shadow as an
    /// un-seeded orphan load (reading garbage). Here the index is `idx & 0x1fff`, a
    /// bounded `[0, 0x1fff]` range whose 0x2003-byte span exceeds 4096, so the
    /// region enters the fold path (`region.add`) but cannot build — exactly the
    /// shape that orphaned the PEB read into shadow.
    #[test]
    fn oversized_region_does_not_orphan_into_shadow() {
        let mut tc = qcode::testing::TestContext::new();
        let input = stack_input(&mut tc, 4, 8);

        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64 @idx:i64>
                    %m = @idx & i64 0x1fff;
                    %addr = @stack_10000004 + %m;
                    %v = load(ram:4, %addr);
                    return at %v;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
            "
        );
        let _ = g;

        Function::from_id_mut(&mut tc.ctx, f).set_input_regs(vec![input]);
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let ptr = tc.ctx.get_const(0x4000, 8).id();
        let idx = tc.ctx.get_const(0x10, 8).id();
        set_call(&mut tc, g_call, f, vec![ptr, idx]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        // The region can't build, so there's nothing to model: no shadow promotion.
        argpromote(&mut tc.ctx);

        // The crux: the buffer load must NOT have been redirected into a shadow
        // space. It stays a real-ram load (left for a later round once the index is
        // a constant, or genuinely unpromotable) — never an un-seeded shadow orphan.
        let ram = tc.ctx.shared.default_space;
        let shadow_load = Function::from_id(&tc.ctx, f).iter().any(|blk| {
            blk.iter()
                .any(|i| matches!(i.mnemonic(), Mnemonic::Load(l) if l.space != ram))
        });
        assert!(
            !shadow_load,
            "an oversized region must not be redirected into shadow without a seed"
        );

        // No `Array` snapshot param was minted for the failed region.
        let has_array_param = Function::from_id(&tc.ctx, f).root().is_some_and(|b| {
            b.params()
                .any(|p| tc.ctx.shared.types.array_of(p.type_id()).is_some())
        });
        assert!(
            !has_array_param,
            "no Array param for a region that can't build"
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
                    store(ram:8, %slot <- @bufptr);
                    goto <f_head @i=i64 0x0>;
                <f_head @i:i64>
                    %c = @i < i64 0x14;
                    if %c goto <f_body> else goto <f_exit>;
                <f_body>
                    %cc = load(ram:8, %slot);
                    %addr = %cc + @i;
                    %b = load(ram:1, %addr);
                    %nb = %b + i8 0x1;
                    store(ram:1, %addr <- %nb);
                    %ni = @i + i64 0x1;
                    goto <f_head @i=%ni>;
                <f_exit>
                    return at i64 0x0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
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
            tc.ctx.block_param_mut(inner).origin = Some(ValueId::Varnode(sp_reg));
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
        let aliases = crate::AliasResult::simple_for_function(&tc.ctx, f).with_frame_freshness(
            &tc.ctx,
            f,
            Some(sp_reg),
        );
        crate::gvn::gvn_function(&mut tc.ctx, f, Some(&aliases));

        // The in-loop reload of the buffer pointer is forwarded away: no load of
        // `%slot` survives in the body (the base is now the by-value @bufptr).
        let slot_reload = Function::from_id(&tc.ctx, f).iter().any(|blk| {
            blk.iter().any(|i| {
                matches!(i.mnemonic(), Mnemonic::Load(l)
                    if l.size == 8 && l.space == tc.ctx.shared.default_space)
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
    #[ignore = "TODO(array-form-migration step 2): array_promote does not yet \
                absorb argpromote's wide shadow envelope, so recognize_total_maps \
                finds no carried array to fold into a map"]
    fn region_promotes_with_coexisting_frame_seed_stores() {
        use qcode::assumption::Proposition;

        let mut tc = qcode::testing::TestContext::new();
        let sp_reg = tc.r0;

        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @esp:i64 @esp_val_0:i64 @esp_val_4:i64>
                    store(ram:8, @esp <- @esp_val_0);
                    %slot = @esp + i64 0x4;
                    store(ram:8, %slot <- @esp_val_4);
                    goto <f_head @i=i64 0x0>;
                <f_head @i:i64>
                    %c = @i < i64 0x14;
                    if %c goto <f_body> else goto <f_exit>;
                <f_body>
                    %addr = @i + @esp_val_4;
                    %b = load(ram:1, %addr);
                    %nb = %b + i8 0x1;
                    store(ram:1, %addr <- %nb);
                    %ni = @i + i64 0x1;
                    goto <f_head @i=%ni>;
                <f_exit>
                    %r = pack(EAX=@esp_val_4);
                    return at @esp_val_0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
            "
        );
        let _ = (g, f_head, f_body, f_exit, f_entry);

        let esp_pid = BasicBlock::from_id(&tc.ctx, f_entry)
            .params()
            .next()
            .unwrap()
            .id();
        if let ValueId::BlockParam(inner) = esp_pid {
            tc.ctx.block_param_mut(inner).origin = Some(ValueId::Varnode(sp_reg));
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
                .any(|p| tc.ctx.shared.types.array_of(p.type_id()).is_some())
        });
        assert!(has_array_param, "the buffer region became an Array input");

        mark_pure_functions(&mut tc.ctx);
        crate::test_util::run_function_pass::<crate::mem::array_promote::ArrayPromote>(
            &mut tc.ctx,
            f,
        )
        .unwrap();
        assert!(
            crate::test_util::run_function_pass::<crate::calls::loop_to_map::LoopToMap>(
                &mut tc.ctx,
                f
            )
            .unwrap(),
            "the loop should be recognized as a map"
        );
        let has_map = Function::from_id(&tc.ctx, f)
            .iter()
            .any(|b| b.iter().any(|i| matches!(i.mnemonic(), Mnemonic::Map(_))));
        assert!(has_map, "f must contain a map");
    }

    /// A state-init function (MT19937 `init_genrand` shape) writes its buffer
    /// region through the by-value pointer `@gp_val_0` *and* publishes that pointer
    /// into a global slot: `store(@gp, @gp_val_0)`. The slot `@gp` is a distinct
    /// incoming pointer — not a frame slot — so v1 `regions_disjoint` rejected the
    /// whole function to partial mode. Under `LoadedPointerDisjointFromSlot`, the
    /// buffer (`*@gp`, the region) is disjoint from the pointer's storage (`@gp`),
    /// recognized via the `{slot}_val_<off>` snapshot name, so the region promotes.
    #[test]
    fn region_promotes_with_coexisting_global_pointer_publish() {
        use qcode::assumption::Proposition;

        let mut tc = qcode::testing::TestContext::new();
        let sp_reg = tc.r0;

        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @esp:i64 @gp:i64 @gp_val_0:i64>
                    store(ram:8, @gp <- @gp_val_0);
                    goto <f_head @i=i64 0x0>;
                <f_head @i:i64>
                    %c = @i < i64 0x14;
                    if %c goto <f_body> else goto <f_exit>;
                <f_body>
                    %addr = @i + @gp_val_0;
                    %b = load(ram:1, %addr);
                    %nb = %b + i8 0x1;
                    store(ram:1, %addr <- %nb);
                    %ni = @i + i64 0x1;
                    goto <f_head @i=%ni>;
                <f_exit>
                    return at i64 0x0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
            "
        );
        let _ = (g, f_head, f_body, f_exit, f_entry);

        let esp_pid = BasicBlock::from_id(&tc.ctx, f_entry)
            .params()
            .next()
            .unwrap()
            .id();
        if let ValueId::BlockParam(inner) = esp_pid {
            tc.ctx.block_param_mut(inner).origin = Some(ValueId::Varnode(sp_reg));
        }
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let espv = tc.ctx.get_const(0x7000, 8).id();
        let gpv = tc.ctx.get_const(0x454df8, 8).id();
        let bufp = tc.ctx.get_const(0x9000, 8).id();
        set_call(&mut tc, g_call, f, vec![espv, gpv, bufp]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        // Without the loaded-pointer assumption the global-slot write blocks the
        // region: it stays on the partial path (no Array param).
        let array_param = |tc: &qcode::testing::TestContext| {
            Function::from_id(&tc.ctx, f).root().is_some_and(|b| {
                b.params()
                    .any(|p| tc.ctx.shared.types.array_of(p.type_id()).is_some())
            })
        };

        tc.ctx
            .assume_true(Proposition::ArgsDisjointFromCallerFrame(f));
        tc.ctx
            .assume_true(Proposition::LoadedPointerDisjointFromSlot(f));
        assert!(
            argpromote_with_sp(&mut tc.ctx, Some(sp_reg)),
            "the region must promote despite the coexisting global-pointer publish"
        );
        assert!(
            array_param(&tc),
            "the buffer region became an Array input (shadow path, not partial)"
        );
    }

    /// Faithful reproduction of MT19937 `init_genrand` (40b3d0): a **write-only**
    /// strided region `*(buf + i*4 + 4)` with a **loop-carried** stored value, plus
    /// the `mt[0]` index and `mt[1]` seed scalar writes, plus the global-pointer
    /// publish and the caller-frame retaddr/arg writes. Used to pin down which gate
    /// keeps it on the partial path.
    #[test]
    fn mt_init_genrand_shape_promotes() {
        use qcode::assumption::Proposition;

        let mut tc = qcode::testing::TestContext::new();
        let sp_reg = tc.r0;

        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @esp:i32 @gp:i32 @esp_val_0:i32 @esp_val_4:i32 @gp_val_0:i32>
                    store(ram:4, @esp <- @esp_val_0);
                    %a4 = @esp + i32 0x4;
                    store(ram:4, %a4 <- @esp_val_4);
                    store(ram:4, @gp <- @gp_val_0);
                    %seed = @gp_val_0 + i32 0x4;
                    store(ram:4, %seed <- @esp_val_4);
                    goto <f_head @ebx=@esp_val_4 @i=i32 0x1>;
                <f_head @ebx:i32 @i:i32>
                    %v = @ebx + @i;
                    %off = @i * i32 0x4;
                    %a = %off + @gp_val_0;
                    %addr = %a + i32 0x4;
                    store(ram:4, %addr <- %v);
                    %ni = @i + i32 0x1;
                    %c = %ni < i32 0x270;
                    if %c goto <f_head @ebx=%v @i=%ni> else goto <f_exit>;
                <f_exit>
                    store(ram:4, @gp_val_0 <- i32 0x270);
                    %r = pack(EAX=@esp_val_4);
                    return at @esp_val_0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i32 0;
            "
        );
        let _ = (g, f_head, f_exit, f_entry);

        let esp_pid = BasicBlock::from_id(&tc.ctx, f_entry)
            .params()
            .next()
            .unwrap()
            .id();
        if let ValueId::BlockParam(inner) = esp_pid {
            tc.ctx.block_param_mut(inner).origin = Some(ValueId::Varnode(sp_reg));
        }
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let espv = tc.ctx.get_const(0x7000, 4).id();
        let gpv = tc.ctx.get_const(0x454df8, 4).id();
        let v0 = tc.ctx.get_const(0x10, 4).id();
        let v4 = tc.ctx.get_const(0x20, 4).id();
        let bufp = tc.ctx.get_const(0x9000, 4).id();
        set_call(&mut tc, g_call, f, vec![espv, gpv, v0, v4, bufp]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        tc.ctx
            .assume_true(Proposition::ArgsDisjointFromCallerFrame(f));
        tc.ctx
            .assume_true(Proposition::LoadedPointerDisjointFromSlot(f));
        argpromote_with_sp(&mut tc.ctx, Some(sp_reg));

        let array_param = Function::from_id(&tc.ctx, f).root().is_some_and(|b| {
            b.params()
                .any(|p| tc.ctx.shared.types.array_of(p.type_id()).is_some())
        });
        assert!(
            array_param,
            "the MT buffer region must promote to an Array input"
        );
    }

    /// End to end: step-1 region promotion then the loop-to-map recognizer turns
    /// the buffer loop's returned write-set value into `map(body_fn, arr)` — the
    /// projectable form. The body is outlined into a fresh pure function.
    #[test]
    #[ignore = "TODO(array-form-migration step 2): array_promote does not yet \
                absorb argpromote's wide shadow envelope (whole-array seed store + \
                wide exit reload), so the loop never reaches the carried-array form \
                the new recognize_total_maps matcher needs"]
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
                    %b = load(ram:1, %addr);
                    %nb = %b + i8 0x1;
                    store(ram:1, %addr <- %nb);
                    %ni = @i + i64 0x1;
                    goto <f_head @i=%ni>;
                <f_exit>
                    return at i64 0x0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
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
        crate::test_util::run_function_pass::<crate::mem::array_promote::ArrayPromote>(
            &mut tc.ctx,
            f,
        )
        .unwrap();
        eprintln!("AFTER ARRAYPROMOTE:\n{}", Function::from_id(&tc.ctx, f));
        assert!(
            crate::test_util::run_function_pass::<crate::calls::loop_to_map::LoopToMap>(
                &mut tc.ctx,
                f
            )
            .unwrap(),
            "the total-map loop should be recognized"
        );
        eprintln!("AFTER MAP:\n{}", Function::from_id(&tc.ctx, f));

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
                .and_then(|t| tc.ctx.shared.types.array_of(t))
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

    /// Keep-loop extraction: the buffer loop *also* threads a register accumulator
    /// (`@acc`, the obfuscator's clobbered `EDX`-style value) that escapes into the
    /// return envelope. The loop is therefore not deletable, but the array channel
    /// is still a clean total map. The recognizer must extract `map(body, arr)` for
    /// the buffer write-set value while leaving the loop running for `@acc`.
    #[test]
    #[ignore = "TODO(array-form-migration step 2): the new carried-array map \
                recognizer replaces the old shadow-form 'keep loop / reroute wide \
                reload' behavior this test asserts; needs array_promote to absorb \
                the wide shadow envelope first"]
    fn keep_loop_extracts_map_when_register_escapes() {
        let mut tc = qcode::testing::TestContext::new();
        let input = stack_input(&mut tc, 4, 8);

        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64>
                    goto <f_head @i=i64 0x0 @acc=i64 0x0>;
                <f_head @i:i64 @acc:i64>
                    %c = @i < i64 0x14;
                    if %c goto <f_body> else goto <f_exit>;
                <f_body>
                    %addr = @stack_10000004 + @i;
                    %b = load(ram:1, %addr);
                    %nb = %b + i8 0x1;
                    store(ram:1, %addr <- %nb);
                    %ni = @i + i64 0x1;
                    %nacc = @acc + @i;
                    goto <f_head @i=%ni @acc=%nacc>;
                <f_exit>
                    %r = pack(EDX=@acc);
                    return at @acc;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
            "
        );
        let _ = (g, f_head, f_body, f_exit, f_entry, r);

        Function::from_id_mut(&mut tc.ctx, f).set_input_regs(vec![input]);
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let ptr = tc.ctx.get_const(0x4000, 8).id();
        set_call(&mut tc, g_call, f, vec![ptr]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(argpromote(&mut tc.ctx), "step 1 region promotion");
        mark_pure_functions(&mut tc.ctx);
        crate::test_util::run_function_pass::<crate::mem::array_promote::ArrayPromote>(
            &mut tc.ctx,
            f,
        )
        .unwrap();
        assert!(
            crate::test_util::run_function_pass::<crate::calls::loop_to_map::LoopToMap>(
                &mut tc.ctx,
                f
            )
            .unwrap(),
            "the array channel should be recognized as a map even though @acc escapes"
        );

        // f contains a `map` over the Array snapshot for the buffer write-set value.
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
                .and_then(|t| tc.ctx.shared.types.array_of(t))
                .is_some(),
            "the map source is the Array snapshot"
        );

        // The outlined per-element body is a fresh pure function.
        assert!(
            tc.ctx
                .function_ids()
                .into_iter()
                .any(|fid| Function::from_id(&tc.ctx, fid).name().contains("_map_body")),
            "the per-element body was outlined"
        );

        // The loop is *kept*: the header/body survive so @acc keeps being computed.
        // (Contrast `dynamic_index_loop_becomes_map`, where the private loop is
        // deleted down to entry + exit.)
        assert!(
            Function::from_id(&tc.ctx, f).iter().count() > 2,
            "the residual @acc loop must remain"
        );
        let has_cbranch = Function::from_id(&tc.ctx, f).iter().any(|b| {
            b.iter()
                .any(|i| matches!(i.mnemonic(), Mnemonic::CBranch(_)))
        });
        assert!(has_cbranch, "the loop's header branch must remain");

        // The loop's shadow channel is *kept intact*: only the wide reload was
        // rerouted to the map. The seed and per-lane stores remain so the surviving
        // residual loop (which still loads each lane) reads a properly seeded
        // region. (Contrast `dynamic_index_loop_becomes_map`, where the deletable
        // loop is removed and the seed store stripped.)
        let any_store = Function::from_id(&tc.ctx, f).iter().any(|b| {
            b.iter().any(|i| matches!(i.mnemonic(), Mnemonic::Store(s)
                if matches!(qcode::space::Space::from_id(&tc.ctx, s.space).ty, qcode::space::SpaceType::Temporary)))
        });
        assert!(any_store, "the kept loop's shadow stores must remain");

        // The rewritten IR verifies clean (no dangling refs, valid terminators).
        let problems = crate::verify::verify(&tc.ctx);
        assert!(
            problems.is_empty(),
            "post-rewrite IR must verify: {problems:?}"
        );
    }

    /// Keep-loop extraction still applies when the escaping accumulator reads the
    /// *array element* (`@acc += buf[i]`). The map replacement only depends on the
    /// region being touched by exactly the four shadow accesses plus totality — it
    /// does not care that the loaded element also feeds `@acc`. Because the loop is
    /// not deletable, its shadow channel (seed + per-lane store + lane load) is left
    /// fully intact so the surviving accumulator loop keeps reading a seeded region;
    /// only the wide reload is rerouted to `map(body, arr)`.
    #[test]
    #[ignore = "TODO(array-form-migration step 2): the new carried-array map \
                recognizer replaces the old shadow-form 'keep loop / reroute wide \
                reload' behavior this test asserts; needs array_promote to absorb \
                the wide shadow envelope first"]
    fn keep_loop_extracts_map_when_element_leaks_to_escaping_result() {
        let mut tc = qcode::testing::TestContext::new();
        let input = stack_input(&mut tc, 4, 8);

        qcode!(
            tc.ctx,
            "
            fn f:
                <f_entry @stack_10000004:i64>
                    goto <f_head @i=i64 0x0 @acc=i64 0x0>;
                <f_head @i:i64 @acc:i64>
                    %c = @i < i64 0x14;
                    if %c goto <f_body> else goto <f_exit>;
                <f_body>
                    %addr = @stack_10000004 + @i;
                    %b = load(ram:1, %addr);
                    %nb = %b + i8 0x1;
                    store(ram:1, %addr <- %nb);
                    %ni = @i + i64 0x1;
                    %bw = zext(i64, %b);
                    %nacc = @acc + %bw;
                    goto <f_head @i=%ni @acc=%nacc>;
                <f_exit>
                    %r = pack(EDX=@acc);
                    return at @acc;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
            "
        );
        let _ = (g, f_head, f_body, f_exit, f_entry, r);

        Function::from_id_mut(&mut tc.ctx, f).set_input_regs(vec![input]);
        Function::from_id_mut(&mut tc.ctx, f).set_pure_reg(true);
        let ptr = tc.ctx.get_const(0x4000, 8).id();
        set_call(&mut tc, g_call, f, vec![ptr]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(argpromote(&mut tc.ctx), "step 1 region promotion");
        mark_pure_functions(&mut tc.ctx);
        crate::test_util::run_function_pass::<crate::mem::array_promote::ArrayPromote>(
            &mut tc.ctx,
            f,
        )
        .unwrap();
        assert!(
            crate::test_util::run_function_pass::<crate::calls::loop_to_map::LoopToMap>(
                &mut tc.ctx,
                f
            )
            .unwrap(),
            "the array channel is a clean total map even though the element feeds @acc"
        );

        // A map over the Array snapshot is extracted for the buffer write-set value.
        let has_map = Function::from_id(&tc.ctx, f)
            .iter()
            .any(|b| b.iter().any(|i| matches!(i.mnemonic(), Mnemonic::Map(_))));
        assert!(has_map, "the array channel must be extracted as a map");

        // The loop is kept (not deletable — @acc escapes), with its shadow channel
        // left intact so the surviving lane load still reads a seeded region.
        let has_cbranch = Function::from_id(&tc.ctx, f).iter().any(|b| {
            b.iter()
                .any(|i| matches!(i.mnemonic(), Mnemonic::CBranch(_)))
        });
        assert!(has_cbranch, "the residual @acc loop must remain");
        let any_store = Function::from_id(&tc.ctx, f).iter().any(|b| {
            b.iter().any(|i| matches!(i.mnemonic(), Mnemonic::Store(s)
                if matches!(qcode::space::Space::from_id(&tc.ctx, s.space).ty, qcode::space::SpaceType::Temporary)))
        });
        assert!(any_store, "the kept loop's shadow stores must remain");

        // The rewritten IR verifies clean (no dangling refs, valid terminators).
        let problems = crate::verify::verify(&tc.ctx);
        assert!(
            problems.is_empty(),
            "post-rewrite IR must verify: {problems:?}"
        );
    }
}
