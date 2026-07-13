// TODO: make the passes ?

//! Reasonable-assumption analysis passes and the checkpoint+replay driver.
//!
//! - [`assume_call_returns`] is the *make* pass: the clean IR already carries a
//!   materialized fall-through edge for every may-return call, so this records a
//!   [`Proposition::FunctionReturns`] assumption for each callee and *prunes*
//!   that edge for callees already known/assumed noreturn (the assume refuses).
//! - [`verify_assumptions`] is the *verify* pass: it proves each assumed
//!   `FunctionReturns` proposition true or false via
//!   [`Context::set_known`](qcode::context::Context::set_known); a
//!   contradiction records a violation on the context.
//! - [`analyze_with_assumptions`] wraps a caller-supplied set of dependent
//!   passes in whole-program checkpoint+replay so a violated assumption leaves
//!   no derived residue (see [`qcode::assumption`]).

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use qcode::{
    assumption::{PassName, Proposition},
    context::Context,
    pass_scope,
    value::{
        BasicBlock, FunctionBody, FunctionId,
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
/// The clean IR already carries a materialized fall-through edge for every
/// may-return call (wired at lift time). For every block that ends in a direct
/// `Call`, this assumes [`Proposition::FunctionReturns`] for the callee; if the
/// assumption is refused (the callee is already known/assumed noreturn), the
/// materialized fall-through edge is *pruned*. A `Call` is a terminator, so its
/// only successor is that fall-through, making the edge to remove unambiguous.
///
/// Idempotent: a returning callee keeps its edge; a noreturn callee's edge is
/// removed once and a re-run finds nothing left to prune. `CallInd` is left
/// untouched (callee unknown — conservatively assumed returning).
///
/// Returns the number of fall-through edges pruned.
pub fn assume_call_returns(ctx: &mut Context) -> usize {
    let _scope = pass_scope::enter("assume_call_returns");
    let func_ids: Vec<FunctionId> = ctx.functions().map(|f| f.id).collect();
    let mut count = 0;

    for func_id in func_ids {
        let block_ids: Vec<BlockId> = qcode::value::FunctionBody::from_id(ctx, func_id).block_ids();

        // (call_block, callee) to act on after the read-only scan releases its
        // borrow of `ctx`.
        let mut calls: Vec<(BlockId, FunctionId)> = Vec::new();

        for &call_block in &block_ids {
            let Some(&call_site) = ctx.block(call_block).instruction_ids().last() else {
                continue;
            };
            let call_site = InstructionId::new(call_block.func, call_site);
            let callee = match ctx.instruction(call_site).mnemonic() {
                Mnemonic::Call(call) => match call.target.real() {
                    Some(callee) => callee,
                    None => continue,
                },
                _ => continue,
            };
            calls.push((call_block, callee));
        }

        for (call_block, callee) in calls {
            // Admissible (callee may return): keep the materialized edge.
            if ctx.assume_true(Proposition::FunctionReturns(callee)) {
                continue;
            }
            // Refused — the callee is known/assumed noreturn. Prune the call
            // block's fall-through edge(s); a `Call` terminator has no other
            // successor, so this only removes the continuation edge.
            qcode::pass_log!(trace, "callee {callee:?} is noreturn, pruning fall-through");
            let mut edges: Vec<_> = BasicBlock::from_id(ctx, call_block)
                .successors()
                .map(|(edge, _)| edge)
                .collect();
            edges.sort_unstable();
            edges.dedup();
            for edge in edges {
                ctx.remove_cfg_edge(call_block.func, edge);
                count += 1;
            }
        }
    }

    qcode::stat!("call_return_edges_pruned", count as u64);
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
                FunctionBody::from_id(ctx, callee).name(),
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
pub fn verify_forced_returns(
    ctx: &mut Context,
    overrides: &std::collections::HashMap<Proposition, bool>,
) {
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
/// A function returns unless it provably cannot. The *only* way it cannot is if
/// a call to a noreturn callee post-dominates the entry: then every path from
/// entry funnels through that call and control never comes back. Crucially,
/// *absence of a `Return`* is **not** evidence of noreturn — a function whose
/// tail is an indirect call or indirect branch (a tail call / jump table) has
/// no `Return` of its own yet still returns to its caller through the tail
/// target. The old "returns iff body contains a `Return`" rule mis-flagged
/// exactly those functions as noreturn, dropping the fall-through edges between
/// their callers and the instruction after the call.
///
/// External stubs have no body, so they are taken to return unless their name is
/// in [`NORETURN_NAMES`].
fn function_returns(ctx: &Context, f: FunctionId) -> bool {
    let func = FunctionBody::from_id(ctx, f);

    if is_noreturn_name(func.name()) {
        return false;
    }
    if func.is_external() {
        return true;
    }

    let Some(entry) = func.root() else {
        // No body to inspect: conservatively assume it returns.
        return true;
    };

    // f returns iff a *returning exit* is reachable from the entry. A returning
    // exit is an exit block (no successors) whose terminator is anything but a
    // call to a noreturn callee — `Return`, an indirect branch/call tail, etc.
    // The only path to noreturn is for every reachable exit to be a noreturn
    // call (equivalently: such a call post-dominates the entry), so control
    // never flows back. An exitless body (an unconditional infinite loop) has no
    // returning exit and is therefore noreturn.
    let mut seen: HashSet<BlockId> = HashSet::default();
    let mut stack = vec![entry.id];
    while let Some(block) = stack.pop() {
        if !seen.insert(block) {
            continue;
        }
        let mut has_successor = false;
        for (_, succ) in BasicBlock::from_id(ctx, block).successors() {
            has_successor = true;
            stack.push(succ);
        }
        if !has_successor && !is_noreturn_call_block(ctx, block) {
            return true;
        }
    }
    false
}

/// Whether `block`'s terminator is a direct `Call` to a callee already
/// known/assumed noreturn (the fall-through edge having been pruned by
/// [`assume_call_returns`], leaving the block an exit).
fn is_noreturn_call_block(ctx: &Context, block: BlockId) -> bool {
    let Some(&last) = ctx.block(block).instruction_ids().last() else {
        return false;
    };
    let last = InstructionId::new(block.func, last);
    let Mnemonic::Call(call) = ctx.instruction(last).mnemonic() else {
        return false;
    };
    // A recorded `FunctionReturns(callee) = false` (assumed or known) marks the
    // callee noreturn; an unrecorded callee defaults to returning.
    let Some(target) = call.target.real() else {
        return false;
    };
    ctx.truth(Proposition::FunctionReturns(target))
        .is_some_and(|t| !t.value)
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
    let mut knowledge: HashMap<Proposition, (bool, PassName)> = HashMap::default();

    loop {
        let mut ctx = baseline.clone();
        for (&prop, &(value, pass)) in &knowledge {
            ctx.seed_known(prop, value, pass);
        }
        assume_call_returns(&mut ctx);
        run_dependent(&mut ctx);
        let novel = verify_assumptions(&mut ctx);

        if novel == 0 && ctx.violations().is_empty() {
            return ctx;
        }
        knowledge.extend(ctx.known_facts().map(|(p, v, pass)| (p, (v, pass))));
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
