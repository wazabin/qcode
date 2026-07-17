//! Interprocedural call analysis: function summaries (unbounded-stack-read
//! facts), call-argument binding, and the stack-fact learning that backs them.
//! The register effect/interface channel lives in [`FunctionEffects`] on the
//! function interface, populated by the `argpromote` register passes.

use qcode::{
    context::Context,
    value::{
        FunctionId,
        insn::{InstructionId, Mnemonic},
    },
};

mod argpromote;
mod call_graph;
mod carried_array;
mod dead_signature;
mod depipeline;
pub(crate) mod effect_engine;
mod interface;
mod loop_to_map;
mod loop_to_scan;
mod mem_effects;
mod outline;
mod param_attrs;
mod partial_inline;
mod projection;
mod stack_facts;
mod strlen;
mod summaries;

pub use argpromote::{
    RegPurityGates, RegPurityReason, argpromote, argpromote_registers, mark_pure_functions,
    reg_purity,
};
pub use call_graph::{CallEdge, CallEdgeId, CallGraph, CallGraphAnalysis, CallKind, CallTarget};
pub use dead_signature::dead_signature;
pub use interface::{append_caller_arg, append_entry_param, remove_entry_param};
pub use mem_effects::set_all_written_spaces;
pub(crate) use outline::inline_pure_body;
pub use param_attrs::infer_param_attrs;
pub use partial_inline::partial_inline;
pub use projection::{Projection, project_return, return_field};
pub use stack_facts::{learn_stack_facts, seed_stack_facts};
pub use summaries::{set_all_function_summaries, set_function_summaries};

/// Incoming sites backed specifically by a real [`Mnemonic::Call`]. The graph's
/// public `call_sites` query also includes direct-like `Apply`/`Map`/`Scan` and
/// `TailCall` sites; interface-rewrite consumers must not treat those encodings
/// as positional `Call.args`.
pub(crate) fn direct_call_sites(
    ctx: &Context<'_>,
    graph: &CallGraph,
    callee: FunctionId,
) -> Vec<InstructionId> {
    graph
        .call_sites(callee)
        .into_iter()
        .filter(|&site| {
            matches!(
                ctx.get_insn(site).mnemonic(),
                Mnemonic::Call(call) if call.target.real() == Some(callee)
            )
        })
        .collect()
}

/// Build a fresh graph, extract real direct-call sites, then drop the snapshot
/// before the caller mutates IR.
pub(crate) fn fresh_direct_call_sites(ctx: &Context<'_>, callee: FunctionId) -> Vec<InstructionId> {
    direct_call_sites(ctx, &CallGraph::analyze(ctx), callee)
}
