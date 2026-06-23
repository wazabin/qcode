//! Interprocedural call analysis: function summaries (input/saved/clobbered
//! registers, stack delta), call-argument binding, the stack-fact learning that
//! backs them, and the clobbered-register computation.

mod argpromote;
mod argpromote_stack;
mod binding;
mod clobbered;
mod dead_signature;
mod interface;
mod partial_inline;
mod projection;
mod stack_facts;
mod summaries;

pub use argpromote::{
    RegPurityReason, argpromote, argpromote_registers, mark_pure_functions, reg_purity,
};
pub use argpromote_stack::argpromote_stack;
pub use binding::{bind_all_call_args, bind_call_args, resolve_arg_loads};
pub use clobbered::{compute_clobbered_regs, set_clobbered_regs};
pub use dead_signature::dead_signature;
pub use interface::{append_caller_arg, append_entry_param, remove_entry_param};
pub use partial_inline::partial_inline;
pub use projection::{Projection, project_return, return_field};
pub use stack_facts::{learn_stack_facts, seed_stack_facts};
pub use summaries::{
    compute_call_clobbered_regs, compute_input_regs, compute_saved_regs, compute_stack_delta,
    set_all_call_clobbered_regs, set_all_function_summaries, set_function_summaries,
};
