mod external_sig;
mod mem_protections;
mod no_return;

pub use external_sig::{apply_all_external_signatures, apply_external_signature};
pub use mem_protections::{MemoryProtections, establish_memory_protections};
pub use no_return::{
    analyze_with_assumptions, assume_call_returns, verify_assumptions, verify_forced_returns,
};
