use crate::value::VarnodeId;

/// Optional ABI description attached to a function.
/// All fields are `Option` — only provided fields affect analysis.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct FunctionSignature {
    /// Registers read as inputs (informational; reserved for future passes).
    pub inputs: Option<Vec<VarnodeId>>,
    /// Registers written as outputs / return values. Joined to Unknown in alias analysis.
    pub outputs: Option<Vec<VarnodeId>>,
    /// Registers clobbered by the callee (caller must save). Joined to Unknown in alias analysis.
    pub caller_saved: Option<Vec<VarnodeId>>,
    /// Registers actually written by this function, as computed by analysis.
    /// This is a precise, body-derived subset of `caller_saved`: it only
    /// includes registers the function concretely stores to.
    pub clobbered: Option<Vec<VarnodeId>>,
    /// Registers read on entry and restored unchanged at exit (the
    /// save/restore prologue/epilogue pattern). These are preserved across a
    /// call — neither arguments nor clobbers — so they are excluded from both
    /// `inputs` and `clobbered`.
    pub saved: Option<Vec<VarnodeId>>,
    /// Net change this function applies to the stack pointer between entry and
    /// return, derived from the final `RSP = @stack_base + N` write (so `N` for
    /// the standard x86-64 epilogue that pops the return address is `+8`).
    /// `None` when the delta is unknown or the function's return blocks disagree;
    /// in that case the stack pointer is treated as an ordinary clobber.
    pub stack_delta: Option<i64>,
}
