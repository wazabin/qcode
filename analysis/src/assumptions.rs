//! Reasonable-assumption analysis passes and the checkpoint+replay driver.
//!
//! - [`assume_call_returns`] is the *make* pass: it connects the fall-through of
//!   every returning call with a real CFG edge and records a
//!   [`CallReturns`](qcode::assumption::AssumptionKind::CallReturns) assumption.
//! - [`verify_assumptions`] is the *verify* pass: it checks each recorded
//!   assumption against its callee and reports any newly-proven noreturn facts.
//! - [`analyze_with_assumptions`] wraps a caller-supplied set of dependent
//!   passes in whole-program checkpoint+replay so a violated assumption leaves
//!   no derived residue (see [`qcode::assumption`]).

use std::collections::HashSet;

use qcode::{
    assumption::{
        AssumptionId, AssumptionKind, AssumptionKnowledge, AssumptionStatus, CallReturnsAssumption,
    },
    context::Context,
    value::{
        BasicBlock, Function, FunctionId, Instruction,
        block::BlockId,
        insn::{InstructionId, Mnemonic},
    },
};

/// Callees that, by convention, do not return. Used to classify external stubs
/// (which have no body to inspect) as noreturn.
const NORETURN_NAMES: &[&str] = &[
    "exit",
    "_exit",
    "_Exit",
    "quick_exit",
    "abort",
    "__stack_chk_fail",
    "_Unwind_Resume",
    "longjmp",
    "siglongjmp",
    "__assert_fail",
    "pthread_exit",
    "ExitProcess",
    "ExitThread",
    "RtlExitUserProcess",
    "RtlExitUserThread",
];

/// *Make* pass — record call-return assumptions across the whole program.
///
/// For every block that ends in a direct `Call` whose callee is not already
/// known noreturn, this connects the call block to its fall-through
/// continuation with a CFG edge and records an
/// [`Unverified`](AssumptionStatus::Unverified) assumption. The continuation is
/// the block in the same function with the smallest address strictly greater
/// than the call instruction's address — robustly the instruction after the
/// call, since nothing can start inside the call's bytes.
///
/// Idempotent: a call block that already has a successor edge is skipped.
/// `CallInd` is skipped in v1 (callee unknown).
///
/// Returns the number of assumptions recorded.
pub fn assume_call_returns(ctx: &mut Context, knowledge: &AssumptionKnowledge) -> usize {
    let func_ids: Vec<FunctionId> = ctx.functions().map(|f| f.id).collect();
    let mut count = 0;

    for func_id in func_ids {
        let block_ids: Vec<BlockId> = ctx.values.functions[func_id]
            .blocks
            .iter()
            .copied()
            .collect();

        // Blocks of this function that carry a machine address, sorted ascending,
        // so the continuation lookup is a simple "first address greater than".
        let mut addressed: Vec<(u64, BlockId)> = block_ids
            .iter()
            .filter_map(|&bid| ctx.values.basic_blocks[bid].address.map(|a| (a, bid)))
            .collect();
        addressed.sort_by_key(|(a, _)| *a);

        // (call_block, call_site, callee, continuation) to materialise after the
        // read-only scan releases its borrow of `ctx`.
        let mut edits: Vec<(BlockId, InstructionId, FunctionId, BlockId)> = Vec::new();

        for &call_block in &block_ids {
            let Some(&call_site) = ctx.values.basic_blocks[call_block].instructions.last() else {
                continue;
            };
            let callee = match ctx.values.instructions[call_site].mnemonic() {
                Mnemonic::Call(call) => call.target,
                _ => continue,
            };
            if knowledge.is_noreturn(callee) {
                continue;
            }
            // Idempotency: don't double-connect an already-linked call block.
            if BasicBlock::from_id(ctx, call_block)
                .successors()
                .next()
                .is_some()
            {
                continue;
            }
            let Some(call_addr) = Instruction::from_id(ctx, call_site).address() else {
                continue;
            };
            let Some(&(_, continuation)) = addressed.iter().find(|(a, _)| *a > call_addr) else {
                continue;
            };
            edits.push((call_block, call_site, callee, continuation));
        }

        for (call_block, call_site, callee, continuation) in edits {
            let continuation_edge = ctx.add_cfg_edge(call_block, continuation);
            ctx.add_assumption(AssumptionKind::CallReturns(CallReturnsAssumption {
                callee,
                call_site,
                call_block,
                continuation,
                continuation_edge,
            }));
            count += 1;
        }
    }

    count
}

/// *Verify* pass — check every unverified assumption against its callee.
///
/// Each assumption is flipped to [`Confirmed`](AssumptionStatus::Confirmed) or
/// [`Violated`](AssumptionStatus::Violated). On violation the continuation edge
/// is removed (best-effort, for single-pass use) and the callee is added to the
/// returned set of newly-proven noreturn functions — the facts the
/// checkpoint+replay driver feeds back into the next round.
pub fn verify_assumptions(ctx: &mut Context) -> HashSet<FunctionId> {
    let mut learned = HashSet::new();

    let unverified: Vec<AssumptionId> = ctx
        .assumptions()
        .filter(|(_, a)| a.status == AssumptionStatus::Unverified)
        .map(|(id, _)| id)
        .collect();

    for id in unverified {
        let AssumptionKind::CallReturns(call) = &ctx.assumption(id).kind else {
            continue;
        };
        let callee = call.callee;
        let continuation_edge = call.continuation_edge;

        if function_returns(ctx, callee) {
            ctx.set_assumption_status(id, AssumptionStatus::Confirmed);
        } else {
            ctx.set_assumption_status(id, AssumptionStatus::Violated);
            ctx.remove_cfg_edge(continuation_edge);
            learned.insert(callee);
        }
    }

    learned
}

/// Whether `f` is assumed to return to its caller.
///
/// v1 rule (edge-independent, hence monotone): a function returns iff its body
/// contains a `Return`. External stubs have no body, so they are taken to return
/// unless their name is in [`NORETURN_NAMES`].
fn function_returns(ctx: &Context, f: FunctionId) -> bool {
    let func = Function::from_id(ctx, f);

    if is_noreturn_name(func.name()) {
        return false;
    }
    if func.is_external() {
        return true;
    }

    func.blocks().any(|block| {
        block
            .iter()
            .any(|insn| matches!(insn.mnemonic(), Mnemonic::Return(_)))
    })
}

fn is_noreturn_name(name: &str) -> bool {
    let base = name
        .rsplit("::")
        .next()
        .unwrap_or(name)
        .split('@')
        .next()
        .unwrap_or(name);
    NORETURN_NAMES
        .iter()
        .any(|known| known.eq_ignore_ascii_case(base))
}

/// Run assumption-dependent analysis to a fixpoint with whole-program
/// checkpoint+replay.
///
/// `baseline` is the freshly-lifted IR (no speculation). Each round clones it,
/// pins already-proven facts, records call-return assumptions, runs the
/// caller-supplied `run_dependent` passes (mem2reg, gvn, dce, …) on the assumed
/// edges, then verifies. If verification proves a *new* noreturn callee, the
/// working copy is discarded and the round replays with the fact pinned;
/// knowledge only grows, so the loop terminates. Returns the converged Context.
pub fn analyze_with_assumptions<'str>(
    baseline: &Context<'str>,
    mut run_dependent: impl FnMut(&mut Context<'str>),
) -> Context<'str> {
    let mut knowledge = AssumptionKnowledge::default();

    loop {
        let mut ctx = baseline.clone();
        assume_call_returns(&mut ctx, &knowledge);
        run_dependent(&mut ctx);
        let learned = verify_assumptions(&mut ctx);

        if learned.iter().all(|f| knowledge.noreturn.contains(f)) {
            return ctx;
        }
        knowledge.noreturn.extend(learned);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noreturn_names_match_windows_import_display_names() {
        assert!(is_noreturn_name("KERNEL32.DLL::ExitProcess"));
        assert!(is_noreturn_name("kernel32.dll::exitthread@4"));
        assert!(is_noreturn_name("LIBC::abort"));
        assert!(!is_noreturn_name("KERNEL32.DLL::CreateFileW"));
    }
}
