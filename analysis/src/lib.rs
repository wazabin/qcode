pub mod assumptions;
pub use assumptions::{
    analyze_with_assumptions, assume_call_returns, verify_assumptions, verify_forced_returns,
};

pub mod brighten;
pub use brighten::brighten_stack;

pub mod lower_stack;
pub use lower_stack::lower_stack;

pub mod clobbered;
pub use clobbered::{compute_clobbered_regs, set_clobbered_regs};

pub mod crt;
pub use crt::discover_libc_main;

pub mod call_summary;
pub use call_summary::{
    bind_all_call_args, bind_call_args, compute_call_clobbered_regs, compute_input_regs,
    compute_saved_regs, compute_stack_delta, learn_stack_facts, resolve_arg_loads,
    seed_stack_facts, set_all_call_clobbered_regs, set_all_function_summaries,
    set_function_summaries,
};

pub mod cfg;

pub mod naming;

#[cfg(test)]
mod test_util;

pub mod external_sig;
pub use external_sig::{apply_all_external_signatures, apply_external_signature};

pub mod cfg_simplify;
pub use cfg_simplify::simplify_cfg;

pub mod dead_load;
pub use dead_load::{dead_load_insns, remove_dead_load_insns, remove_dead_load_insns_block};

pub mod mem_liveness;
pub use mem_liveness::{MemLiveness, compute_memory_liveness};

pub mod dce;
pub use dce::{dead_insns, remove_dead_insns};

pub mod alias;
pub use alias::{AliasResult, alias_analysis};

pub mod gvn;
pub use gvn::{constant_fold_function, gvn, gvn_function};

pub mod mem2reg;
pub use mem2reg::mem2reg;

pub mod value_range;
pub use value_range::{ValueRange, value_range};

pub mod lift;
pub use lift::{discover_addresses_in_binary, lift_new_addresses};

pub mod mem_protections;
pub use mem_protections::{MemoryProtections, establish_memory_protections};

pub mod pipeline;
pub use pipeline::{
    ArchConfig, CallingConvention, DEFAULT_PIPELINE_TOML, DynFunctionPass, DynPass, FunctionPass,
    GpReg, LiftOutcome, LiftSummary, Pass, PassRegistration, Pipeline, PipelineEnv,
    PipelineServices, RegisteredPass, analyze_and_lift_with_progress, analyze_default,
    analyze_with_pipeline,
};
