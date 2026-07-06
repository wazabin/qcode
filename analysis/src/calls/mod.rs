//! Interprocedural call analysis: function summaries (input/saved/clobbered
//! registers, stack delta), call-argument binding, the stack-fact learning that
//! backs them, and the clobbered-register computation.

mod argpromote;
mod carried_array;
mod clobbered;
mod dead_signature;
mod depipeline;
mod interface;
mod loop_to_map;
mod loop_to_scan;
mod mem_effects;
mod outline;
mod param_attrs;
mod partial_inline;
mod projection;
mod stack_facts;
mod summaries;

pub use argpromote::{
    RegPurityGates, RegPurityReason, argpromote, argpromote_registers, mark_pure_functions,
    reg_purity,
};
pub use clobbered::{compute_clobbered_regs, set_clobbered_regs};
pub use dead_signature::dead_signature;
pub use interface::{append_caller_arg, append_entry_param, remove_entry_param};
pub use mem_effects::set_all_written_spaces;
pub(crate) use outline::inline_pure_body;
pub use param_attrs::infer_param_attrs;
pub use partial_inline::partial_inline;
pub use projection::{Projection, project_return, return_field};
pub use stack_facts::{learn_stack_facts, seed_stack_facts};
pub use summaries::{
    compute_call_clobbered_regs, compute_input_regs, compute_stack_delta,
    set_all_call_clobbered_regs, set_all_function_summaries, set_function_summaries,
};
