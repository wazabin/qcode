use crate::value::VarnodeId;

/// Optional ABI description attached to a function.
/// All fields are `Option` — only provided fields affect analysis.
#[derive(Clone)]
pub struct FunctionSignature {
    /// Registers read as inputs (informational; reserved for future passes).
    pub inputs: Option<Vec<VarnodeId>>,
    /// Registers written as outputs / return values. Joined to Unknown in alias analysis.
    pub outputs: Option<Vec<VarnodeId>>,
    /// Registers clobbered by the callee (caller must save). Joined to Unknown in alias analysis.
    pub caller_saved: Option<Vec<VarnodeId>>,
}
