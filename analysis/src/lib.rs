pub mod assumptions;
pub use assumptions::{
    MemoryProtections, analyze_with_assumptions, apply_all_external_signatures,
    apply_external_signature, assume_args_disjoint_caller_frame, assume_call_returns,
    establish_memory_protections, verify_args_disjoint_caller_frame, verify_assumptions,
    verify_forced_returns,
};

pub mod stack;

pub mod calls;
pub use calls::{
    append_caller_arg, append_entry_param, argpromote, argpromote_registers,
    compute_call_clobbered_regs, compute_clobbered_regs, compute_input_regs,
    compute_stack_delta, learn_stack_facts, remove_entry_param, seed_stack_facts,
    set_all_call_clobbered_regs, set_all_function_summaries, set_clobbered_regs,
    set_function_summaries,
};

pub mod cfg;

pub mod structure;
pub use structure::{
    BlockExit, EdgeCondition, Program, RecoverSwitch, RefineLoops, Structure, TokenKind, TokenLine,
    block_exit, decompile_function, emit_c, emit_tokens, lower_expr, lower_function,
};

pub mod naming;

pub mod structs;
pub use structs::StructTyping;
pub use structs::win32::{WindowsTebSeed, register_teb_structs, seed_teb_register};

pub mod example;

#[cfg(test)]
mod test_util;

pub mod dce;
pub use dce::{
    dead_insns, dead_load_insns, remove_dead_insns, remove_dead_load_insns,
    remove_dead_load_insns_block,
};

pub mod alias;
pub use alias::{AliasResult, RegisterBase, alias_analysis};

pub mod gvn;
pub use gvn::{constant_fold_function, gvn, gvn_function};

pub mod mem;
pub use mem::{MemLiveness, compute_memory_liveness, mem2reg};

pub mod value_range;
pub use value_range::{ValueRange, value_range};

pub mod verify;
pub use verify::{PureRegCallArgsViolation, Verify, verify, verify_ir, verify_pure_reg_call_args};

pub mod loop_unroll;
pub use loop_unroll::{
    RecognizeSimpleLoops, UnrollSimpleLoops, recognize_simple_loops, unroll_simple_loops,
};

pub mod lift;
pub use lift::{discover_addresses_in_binary, lift_new_addresses, split_overlapping_functions};

pub mod pipeline;
pub use pipeline::{
    ArchConfig, CallingConvention, DEFAULT_PIPELINE_TOML, DecompilePass, DynDecompilePass,
    DynFunctionPass, DynPass, FunctionPass, GpReg, LiftOutcome, LiftSummary, Pass,
    PassRegistration, Pipeline, PipelineEnv, PipelineServices, ProgressSink, RegisteredPass,
    YieldSignal, analyze_and_lift_with_progress, analyze_default, analyze_with_pipeline,
};
#[cfg(not(target_arch = "wasm32"))]
pub use pipeline::{
    PipelineFile, create_named_user_pipeline_from_default_in, create_user_pipeline_from_default_in,
    list_user_pipelines_in, load_named_user_pipeline_in,
};

pub mod trace;
pub use trace::Trace;
