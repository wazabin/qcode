//! End-to-end tests for the reasonable-assumption subsystem, driven through the
//! real x64 lifter so block/instruction machine addresses are populated exactly
//! as in production.

use std::collections::HashMap;

use harbinger::arch::{arch_config, x64};
use qcode::address_index::AddressIndex;
use qcode::assumption::{Certainty, Proposition};
use qcode::{
    context::Context,
    value::{BasicBlock, FunctionBody, FunctionId, block::BlockId, insn::Mnemonic},
};
use qcode_analysis::pipeline::{Pipeline, PipelineError, analyze_with_overrides_with_progress};
use qcode_analysis::{analyze_with_assumptions, assume_call_returns, verify_assumptions};

/// `main` at 0x1000 calls `foo` at 0x1006, then returns.
///
/// ```text
/// 0x1000: E8 01 00 00 00   call foo (0x1006)
/// 0x1005: C3               ret            <- main's continuation block
/// 0x1006: <callee bytes>
/// ```
fn lift_caller_and_callee(callee_bytes: &[u8]) -> Context<'static> {
    let mut bytes = vec![0xE8, 0x01, 0x00, 0x00, 0x00, 0xC3];
    bytes.extend_from_slice(callee_bytes);

    let mut disasm = x64::Disassembler::from_bytes(0x1000, &bytes);
    let mut ctx = x64::make_context();
    disasm.lift_from_entries(&mut ctx, [0x1000]);
    ctx
}

/// Returns the block in `func` that ends in a `Call`.
fn call_block_of(ctx: &Context<'_>, func: FunctionId) -> BlockId {
    FunctionBody::from_id(ctx, func)
        .blocks()
        .find(|b| {
            b.iter()
                .last()
                .is_some_and(|insn| matches!(insn.mnemonic(), Mnemonic::Call(_)))
        })
        .expect("function has a call block")
        .id
}

fn function_at(ctx: &Context<'_>, address: u64) -> Option<FunctionId> {
    AddressIndex::analyze(ctx).function_at(address)
}

#[test]
fn returning_callee_confirms_assumption_and_keeps_edge() {
    // callee `foo`: just `ret`.
    let mut ctx = lift_caller_and_callee(&[0xC3]);

    let foo = function_at(&ctx, 0x1006).expect("foo lifted");
    let main = function_at(&ctx, 0x1000).expect("main lifted");
    let call_block = call_block_of(&ctx, main);

    // The lifter already materialized the fall-through edge.
    assert!(
        BasicBlock::from_id(&ctx, call_block)
            .successors()
            .next()
            .is_some(),
        "lifter materialized the call fall-through edge"
    );

    let pruned = assume_call_returns(&mut ctx);
    assert_eq!(pruned, 0, "a returning callee's edge is kept, not pruned");

    // The assumption is recorded under the callee, tagged with the make pass,
    // and the real edge is retained.
    let truth = ctx
        .truth(Proposition::FunctionReturns(foo))
        .expect("assumption recorded");
    assert!(truth.value && truth.certainty == Certainty::Assumed);
    assert_eq!(truth.pass, "assume_call_returns");
    assert!(
        BasicBlock::from_id(&ctx, call_block)
            .successors()
            .next()
            .is_some(),
        "call block gained a continuation successor"
    );

    let novel = verify_assumptions(&mut ctx);
    assert_eq!(
        novel, 0,
        "confirming a matching assumption is not novel (no replay)"
    );
    assert!(
        ctx.violations().is_empty(),
        "a returning callee violates nothing"
    );

    assert_eq!(ctx.known(Proposition::FunctionReturns(foo)), Some(true));
    assert!(
        BasicBlock::from_id(&ctx, call_block)
            .successors()
            .next()
            .is_some(),
        "confirmed assumption keeps its edge"
    );
}

#[test]
fn indirect_tail_callee_is_classified_returning() {
    // Regression: a callee whose tail is an indirect branch (`FF E0` = `jmp rax`,
    // a tail call / jump table) has no `Return` of its own, yet it returns to
    // its caller through the tail target. The old "returns iff body contains a
    // `Return`" rule mis-flagged it noreturn, which pruned the fall-through edge
    // between `main`'s call and the instruction after it.
    let mut ctx = lift_caller_and_callee(&[0xFF, 0xE0]);

    let foo = function_at(&ctx, 0x1006).expect("foo lifted");
    let main = function_at(&ctx, 0x1000).expect("main lifted");
    let call_block = call_block_of(&ctx, main);

    assume_call_returns(&mut ctx);
    verify_assumptions(&mut ctx);

    assert_eq!(
        ctx.known(Proposition::FunctionReturns(foo)),
        Some(true),
        "an indirect-tail callee returns; absence of `Return` is not noreturn"
    );
    assert!(
        ctx.violations().is_empty(),
        "a returning callee violates no assumption"
    );
    assert!(
        BasicBlock::from_id(&ctx, call_block)
            .successors()
            .next()
            .is_some(),
        "the caller keeps its fall-through edge to the post-call instruction"
    );
}

#[test]
fn noreturn_callee_records_violation() {
    // callee `foo`: `EB FE` = `jmp $` (infinite self-loop) — no `Return`.
    let mut ctx = lift_caller_and_callee(&[0xEB, 0xFE]);

    assume_call_returns(&mut ctx);

    let foo = function_at(&ctx, 0x1006).expect("foo lifted");

    verify_assumptions(&mut ctx);

    assert_eq!(
        ctx.known(Proposition::FunctionReturns(foo)),
        Some(false),
        "noreturn callee is a proven fact"
    );
    // The first round assumed `foo` returns (it was not yet known noreturn), so
    // the materialized edge is still present here; the replay driver prunes it
    // once the proven fact is seeded (see the driver test below).
    let [violation] = ctx.violations() else {
        panic!("expected exactly one violation");
    };
    assert_eq!(violation.prop, Proposition::FunctionReturns(foo));
    assert_eq!(violation.assuming_pass, "assume_call_returns");
    assert_eq!(violation.asserting_pass, "verify_assumptions");
}

#[test]
fn driver_replays_and_leaves_no_residue_for_noreturn_call() {
    let baseline = lift_caller_and_callee(&[0xEB, 0xFE]);

    // No dependent passes — we only assert the assumption machinery converges
    // and the wrong guess leaves no continuation edge in the result.
    let result = analyze_with_assumptions(&baseline, |_ctx| {});

    let main = function_at(&result, 0x1000).expect("main lifted");
    let foo = function_at(&result, 0x1006).expect("foo lifted");
    let call_block = call_block_of(&result, main);

    assert!(
        BasicBlock::from_id(&result, call_block)
            .successors()
            .next()
            .is_none(),
        "after replay, the noreturn call has no continuation edge"
    );

    // The converged round was seeded with the proven fact, so the make-pass's
    // assume was refused and no violation re-occurred.
    assert_eq!(result.known(Proposition::FunctionReturns(foo)), Some(false));
    assert!(result.violations().is_empty(), "converged round is clean");
}

#[test]
fn overrides_path_lifts_the_whole_binary() {
    // foo @ 0x1000 calls bar @ 0x100a. Starting from a fresh (un-lifted) context,
    // the override-aware driver must lift the binary itself — not just analyze
    // whatever was already lifted — because the lifter is now injected on the
    // overrides path too.
    let bytes = [
        0xe8, 0x05, 0x00, 0x00, 0x00, // call bar (0x100a)
        0xc3, // ret
        0x90, 0x90, 0x90, 0x90, // padding
        0x31, 0xc0, // bar: xor eax, eax
        0xc3, // ret
    ];
    let mut disas = x64::Disassembler::from_bytes(0x1000, &bytes);
    let mut ctx = x64::make_context();
    let cfg = arch_config(&ctx).expect("arch config");

    // A benign override keeps the bounded (overrides) path active without
    // contradicting analysis.
    let mut overrides = HashMap::new();
    overrides.insert(
        Proposition::ImmutableMemory {
            addr: 0x9999,
            size: 1,
        },
        true,
    );

    let analyzed = disas
        .lift_and_analyze_with_overrides(&mut ctx, &cfg, &overrides, |_| {})
        .expect("benign override converges");

    assert!(
        function_at(&analyzed, 0x1000).is_some(),
        "entry function should be lifted on the overrides path"
    );
    assert!(
        function_at(&analyzed, 0x100a).is_some(),
        "the call target should be discovered and lifted on the overrides path"
    );
}

#[test]
fn override_agreeing_with_analysis_converges() {
    // foo is noreturn (`jmp $`). Forcing FunctionReturns(foo) = false agrees with
    // what analysis would prove, so the overrides driver converges cleanly.
    let baseline = lift_caller_and_callee(&[0xEB, 0xFE]);
    let cfg = arch_config(&baseline).expect("arch config");
    let foo = function_at(&baseline, 0x1006).expect("foo lifted");

    let mut overrides = HashMap::new();
    overrides.insert(Proposition::FunctionReturns(foo), false);

    let result = analyze_with_overrides_with_progress(
        &baseline,
        &cfg,
        &Pipeline::default(),
        None,
        &overrides,
        |_| {},
    )
    .expect("forced value matches analysis, converges");

    assert_eq!(result.known(Proposition::FunctionReturns(foo)), Some(false));
    assert!(result.known_contradictions().is_empty());
}

#[test]
fn override_contradicted_by_analysis_errors() {
    // foo is noreturn, but the user forces FunctionReturns(foo) = true. Analysis
    // disproves it, so the driver returns a Contradiction rather than looping.
    let baseline = lift_caller_and_callee(&[0xEB, 0xFE]);
    let cfg = arch_config(&baseline).expect("arch config");
    let foo = function_at(&baseline, 0x1006).expect("foo lifted");

    let mut overrides = HashMap::new();
    overrides.insert(Proposition::FunctionReturns(foo), true);

    let result = analyze_with_overrides_with_progress(
        &baseline,
        &cfg,
        &Pipeline::default(),
        None,
        &overrides,
        |_| {},
    );

    match result {
        Err(PipelineError::Contradiction(c)) => {
            assert_eq!(c.prop, Proposition::FunctionReturns(foo));
            assert!(c.known && !c.proven);
        }
        Err(other) => panic!("expected Contradiction, got {other:?}"),
        Ok(_) => panic!("expected Contradiction, got Ok"),
    }
}

#[test]
fn idempotent_make_pass_does_not_double_act() {
    let mut ctx = lift_caller_and_callee(&[0xC3]);

    let first = assume_call_returns(&mut ctx);
    let second = assume_call_returns(&mut ctx);

    // A returning callee keeps its edge: nothing is pruned on either run.
    assert_eq!(first, 0);
    assert_eq!(
        second, 0,
        "re-running prunes nothing and re-assumes idempotently"
    );
}
