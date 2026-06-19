//! Interprocedural call analysis: function summaries (input/saved/clobbered
//! registers, stack delta), call-argument binding, the stack-fact learning that
//! backs them, and the clobbered-register computation.

mod argpromote;
mod binding;
mod clobbered;
mod dead_signature;
mod stack_facts;
mod summaries;

pub use argpromote::{RegPurityReason, argpromote, reg_purity};
pub use binding::{bind_all_call_args, bind_call_args, resolve_arg_loads};
pub use clobbered::{compute_clobbered_regs, set_clobbered_regs};
pub use dead_signature::dead_signature;
pub use stack_facts::{learn_stack_facts, seed_stack_facts};
pub use summaries::{
    compute_call_clobbered_regs, compute_input_regs, compute_saved_regs, compute_stack_delta,
    set_all_call_clobbered_regs, set_all_function_summaries, set_function_summaries,
};
