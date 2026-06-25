use crate::value::VarnodeId;

/// Optional ABI description attached to a function.
/// All fields are `Option` — only provided fields affect analysis.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct FunctionSignature {
    /// Registers read as inputs (informational; reserved for future passes).
    pub inputs: Option<Vec<VarnodeId>>,
    /// Display names for positional call arguments, one per `Call.args` slot, in
    /// order. Populated for functions whose call interface is known by name but
    /// not by a register/param the usual naming (`input_arg_name`) can resolve —
    /// chiefly **external** callees whose stack-passed (`stdcall`/`cdecl`)
    /// arguments come from a C prototype, including a synthesized
    /// `return_address` slot. A `None` entry leaves that slot unnamed.
    #[serde(default)]
    pub input_names: Option<Vec<Option<Box<str>>>>,
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
    /// `true` when this function performs an unresolved/dynamic memory access, or
    /// forwards a stack-typed pointer into a callee that does. A caller that hands
    /// a pointer into its own frame to such a function cannot bound which of its
    /// stack slots the callee reads, so it must keep its whole frame in memory
    /// (no stack promotion). See the stack-escape handling in `mem2reg`.
    #[serde(default)]
    pub reads_unbounded_stack: bool,
    /// `true` when this function passes a pointer into its *own* stack frame to a
    /// callee that may read it unboundedly (a callee with
    /// [`reads_unbounded_stack`](Self::reads_unbounded_stack), an external, or an
    /// indirect call). Such a callee may clobber any of this function's stack
    /// slots, so its frame must stay in memory — `mem2reg` disables all stack
    /// promotion for it. Computed at bind time (while `StackAddress` types are
    /// still present) and seeded across checkpoint+replay rounds.
    #[serde(default)]
    pub frame_escapes_to_unbounded: bool,
    /// `true` once `argpromote_registers` has functionalized this function's
    /// register side effects: every register read is a by-value input param and
    /// every register write rides the returned write-set aggregate, so the body
    /// is a pure function over its params with no register-channel ABI left to
    /// honor. Set only on success (which already implies non-external,
    /// non-address-taken, and direct callers only). `dead_signature` gates on
    /// this — a pure-reg function's args and returned fields can be trimmed
    /// purely from in-IR uses, decoupled from the ABI register lists.
    #[serde(default)]
    pub pure_reg: bool,
    /// `true` once this function's returned values are a deterministic function of
    /// its by-value params, with no value flowing in from outside the SSA graph:
    /// no loads (an untracked memory read), no calls, no architecture p-code ops,
    /// and no raw register/global reads. Stores are permitted — they produce no
    /// value, so they cannot feed a returned field. Strictly stronger than
    /// [`pure_reg`](Self::pure_reg), which only asserts the register channel is
    /// functionalized. Pure-function emulation in constant propagation gates on
    /// this (see `PURE_EMULATION_DESIGN.md`): such a callee may be emulated to
    /// harvest constant return-tuple fields, with the call left in place. Asserted
    /// by argpromote's `mark_pure`; checked by a `verify/` rule.
    #[serde(default)]
    pub is_pure: bool,
    /// `true` when this function's *full* register effect is captured precisely
    /// by its call interface, so the conservative call over-approximation can be
    /// dropped at every call site. Concretely, a resolved callee:
    ///
    /// * **reads no registers** beyond the explicit `Call.args` — the argument
    ///   values are already loaded and passed by the caller, so a call is not an
    ///   implicit use of any register (it may still read memory *through* a
    ///   pointer argument); and
    /// * **writes exactly** the registers in [`clobbered`](Self::clobbered) —
    ///   never the conservative "every register" fallback used for callees with
    ///   no known effect.
    ///
    /// Set for **external** callees whose C prototype is known: their arguments
    /// come from the prototype and their clobber set is the calling convention's
    /// caller-saved (volatile) registers. Consulted by alias analysis and
    /// `mem2reg`'s clobber-aware liveness, which otherwise treat a callee with no
    /// recorded clobber set as reading and writing every register.
    ///
    /// Named for its origin: the resolution comes from *outside* the IR (a C
    /// prototype + calling convention), not from analyzing a body.
    #[serde(default)]
    pub externally_resolved: bool,
}
