//! Reasonable-assumption analysis passes and the checkpoint+replay driver.
//!
//! - [`assume_call_returns`] is the *make* pass: it connects the fall-through of
//!   every returning call with a real CFG edge after recording a
//!   [`Proposition::FunctionReturns`] assumption (skipping callees already
//!   known noreturn — the assume itself refuses).
//! - [`verify_assumptions`] is the *verify* pass: it proves each assumed
//!   `FunctionReturns` proposition true or false via
//!   [`Context::set_known`](qcode::context::Context::set_known); a
//!   contradiction records a violation on the context.
//! - [`analyze_with_assumptions`] wraps a caller-supplied set of dependent
//!   passes in whole-program checkpoint+replay so a violated assumption leaves
//!   no derived residue (see [`qcode::assumption`]).

use std::collections::HashMap;

use qcode::{
    assumption::Proposition,
    context::Context,
    pass_scope,
    value::{BasicBlock, Function, FunctionId, Instruction, block::BlockId, insn::Mnemonic},
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
/// For every block that ends in a direct `Call`, this assumes
/// [`Proposition::FunctionReturns`] for the callee and, if the assumption is
/// admissible (not already known/assumed noreturn), connects the call block to
/// its fall-through continuation with a CFG edge. The continuation is the
/// block in the same function with the smallest address strictly greater than
/// the call instruction's address — robustly the instruction after the call,
/// since nothing can start inside the call's bytes.
///
/// Idempotent: a call block that already has a successor edge is skipped.
/// `CallInd` is skipped in v1 (callee unknown).
///
/// Returns the number of continuation edges added.
pub fn assume_call_returns(ctx: &mut Context) -> usize {
    let _scope = pass_scope::enter("assume_call_returns");
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

        // (call_block, callee, continuation) to materialise after the
        // read-only scan releases its borrow of `ctx`.
        let mut edits: Vec<(BlockId, FunctionId, BlockId)> = Vec::new();

        for &call_block in &block_ids {
            let Some(&call_site) = ctx.values.basic_blocks[call_block].instructions.last() else {
                continue;
            };
            let callee = match ctx.values.instructions[call_site].mnemonic() {
                Mnemonic::Call(call) => call.target,
                _ => continue,
            };
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
            edits.push((call_block, callee, continuation));
        }

        for (call_block, callee, continuation) in edits {
            // Refused when the callee is already known (or assumed) noreturn.
            if !ctx.assume_true(Proposition::FunctionReturns(callee)) {
                qcode::pass_log!(trace, "callee {callee:?} is noreturn, not linking");
                continue;
            }
            ctx.add_cfg_edge(call_block, continuation);
            count += 1;
        }
    }

    qcode::stat!("call_return_edges", count as u64);
    count
}

/// *Verify* pass — prove every assumed `FunctionReturns` proposition.
///
/// Each assumed callee is checked with [`function_returns`] and the result
/// recorded via [`Context::set_known`](Context::set_known); a contradiction
/// (an assumed-returning callee proven noreturn) lands in
/// [`Context::violations`](Context::violations), the replay driver's signal.
///
/// Returns the number of newly-proven facts (novel knowledge).
pub fn verify_assumptions(ctx: &mut Context) -> usize {
    let _scope = pass_scope::enter("verify_assumptions");

    let assumed: Vec<FunctionId> = ctx
        .truths()
        .filter(|(_, t)| t.certainty == qcode::assumption::Certainty::Assumed)
        .filter_map(|(p, _)| match p {
            Proposition::FunctionReturns(f) => Some(f),
            _ => None,
        })
        .collect();

    let mut novel = 0;
    for callee in assumed {
        let returns = function_returns(ctx, callee);
        if ctx.set_known(Proposition::FunctionReturns(callee), returns) {
            novel += 1;
            qcode::pass_log!(
                debug,
                "proved {} {}",
                Function::from_id(ctx, callee).name(),
                if returns { "returns" } else { "noreturn" },
            );
        }
    }
    novel
}

/// Re-prove user-forced `FunctionReturns` facts against the function body.
///
/// [`verify_assumptions`] only checks propositions still in the *assumed* state,
/// so a user override (seeded as *known*) is otherwise never validated. This
/// proves each forced `FunctionReturns` override and records a
/// [`KnownContradiction`](qcode::assumption::KnownContradiction) on the context
/// when the forced polarity disagrees with the body — the driver's signal to
/// abort with an error rather than silently honor an impossible override.
pub fn verify_forced_returns(ctx: &mut Context, overrides: &HashMap<Proposition, bool>) {
    let _scope = pass_scope::enter("verify_assumptions");
    for &prop in overrides.keys() {
        if let Proposition::FunctionReturns(f) = prop {
            let returns = function_returns(ctx, f);
            // Same value: no-op. Opposite of the forced known fact: records a
            // known-contradiction (set_known keeps the original known value).
            ctx.set_known(prop, returns);
        }
    }
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
/// seeds the facts proven in earlier rounds, records call-return assumptions,
/// runs the caller-supplied `run_dependent` passes (mem2reg, gvn, dce, …) on
/// the assumed edges, then verifies. If the round recorded a violation or
/// proved a novel fact, the working copy is discarded and the round replays
/// with the knowledge seeded; knowledge only grows, so the loop terminates.
/// Returns the converged Context.
pub fn analyze_with_assumptions<'str>(
    baseline: &Context<'str>,
    mut run_dependent: impl FnMut(&mut Context<'str>),
) -> Context<'str> {
    let mut knowledge: HashMap<Proposition, bool> = HashMap::new();

    loop {
        let mut ctx = baseline.clone();
        for (&prop, &value) in &knowledge {
            ctx.seed_known(prop, value);
        }
        assume_call_returns(&mut ctx);
        run_dependent(&mut ctx);
        let novel = verify_assumptions(&mut ctx);

        if novel == 0 && ctx.violations().is_empty() {
            return ctx;
        }
        knowledge.extend(ctx.known_facts());
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
