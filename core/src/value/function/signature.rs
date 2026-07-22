use crate::value::VarnodeId;

/// Per-parameter pointer attributes, LLVM-style, inferred (or read from a C
/// prototype) and consumed at call sites to relax the default "every pointer
/// argument aliases everything and is written through by the callee" assumption.
///
/// The two bits are deliberately independent: `readonly` proves only that the
/// callee does not write through the pointer *during this call*, which is enough
/// to stop a call from clobbering the cells the argument reaches (see
/// `mem_forward`'s call-kill). It does *not* prove the pointer is safe to reason
/// about across the call: a captured pointer can be written through later, so
/// frame-freshness reasoning additionally requires `nocapture`. Both bits default
/// to `false` (fully conservative); an inference/extern pass sets them.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub struct ParamAttrs {
    /// The callee never writes through this pointer parameter (C `*const`
    /// semantics, one level deep: a store through a pointer *loaded from* the
    /// param does not clear this — only stores whose address is affine-derived
    /// from the param itself do).
    #[serde(default)]
    pub readonly: bool,
    /// The callee does not retain this pointer beyond the call, except by
    /// returning it (capture-by-return does not clear this: the returned pointer
    /// is still tracked by the caller, so it is not an unbounded escape).
    #[serde(default)]
    pub nocapture: bool,
}

impl ParamAttrs {
    /// The fully permissive attribute set (both bits): the optimistic starting
    /// point of the bottom-up inference fixpoint, whittled down as the body walk
    /// finds writes/captures/escapes.
    pub const OPTIMISTIC: Self = Self {
        readonly: true,
        nocapture: true,
    };
}

/// Where one external call argument is loaded from at the call site.
///
/// Planned once by `external_sigs` from the C prototype + calling convention;
/// consumed by `argpromote_external`, which turns each slot into an SSA value at
/// every direct caller (a register reload or an SP-relative stack load).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExternSlot {
    /// A register argument: reload the register (`size` bytes) live at the call.
    Reg(VarnodeId, usize),
    /// A stack argument at `offset` bytes above the call-site stack pointer.
    Stack { offset: i64, size: usize },
}

/// One planned positional argument of an external call.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExternArg {
    /// Where the argument value is loaded from at the call site.
    pub slot: ExternSlot,
    /// The display name (the C prototype parameter name, or `return_address`
    /// for the synthesized `stdcall`/`cdecl` slot). `None` leaves it unnamed.
    #[serde(default)]
    pub name: Option<Box<str>>,
    /// Per-argument pointer attributes (chiefly `readonly` from a `const`
    /// pointee). Defaults fully conservative for non-pointer arguments.
    #[serde(default)]
    pub attrs: ParamAttrs,
}

/// The memory kind of one prototyped-external parameter, as seen by the RAM
/// argmem model. An external can only touch memory *we* model through pointers
/// *we* pass it (its own libc-internal state lives outside the lifted image), so
/// each pointer parameter bounds a whole-object effect on the caller's argument.
///
/// The variants are ordered so the analysis layer needs no C-type data of its
/// own: it reads this per-input kind (kept in lockstep with the materialized
/// register-interface inputs) plus the function-level variadic flag.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum ArgMemKind {
    /// A non-pointer scalar (integer/float): the external cannot reach any memory
    /// we model through it. No effect.
    NonPtr,
    /// A mutable data pointer (`char *`, `void *`, `struct S *`): a whole-object
    /// **read+write** effect on the addressed object — the callee may both read
    /// the pre-call contents and clobber them (`strcat`'s dest, `realloc`). The
    /// read half means a frame-local landing goes ⊤ (freshness), so such a
    /// pointer is *not* memory-free-composable through an uninitialized local.
    MutPtr,
    /// A `const`-qualified data pointer (`const char *`): a whole-object read-only
    /// effect on the addressed object.
    ConstPtr,
    /// A pointer the shallow whole-object model cannot bound: a function/callback
    /// pointer (re-enters our code — `qsort`'s comparator), or a pointer to
    /// another pointer / an unmodeled pointee (a transitive write escapes the
    /// addressed object). Its presence sends the whole external footprint to ⊤.
    Opaque,
    /// A **write-only** destination pointer (`memset`/`memcpy` dest): the callee
    /// never reads the pre-call contents, only clobbers them. A pure whole-object
    /// *write*, so a frame-local landing is contained (the `memset(&local)` fold).
    /// Minted only for symbols the `extern_argmem` write-only table vouches for
    /// (libc semantics guarantee the destination is never read before write).
    ///
    /// Appended last on purpose: bincode encodes a fieldless enum by variant
    /// **index**, so keeping `NonPtr`/`MutPtr`/`ConstPtr`/`Opaque` at their old
    /// indices lets pre-`OutPtr` `.harbinger` snapshots still decode.
    OutPtr,
}

/// C-prototype-derived argmem summary for a prototyped external: the ordered
/// per-parameter [`ArgMemKind`]s (in lockstep with the materialized register
/// interface inputs, so `params[i]` describes the `i`-th positional argument /
/// `Param(i)`) and whether the callee is variadic. Read by the RAM effect
/// channel's `external_leaf` to derive a bounded argmem footprint in place of ⊤.
/// `None` on the signature for non-externals and un-prototyped externals.
/// Serde-defaulted, so older `.harbinger` snapshots load with it absent.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExternArgmem {
    /// Per-input parameter kinds, lockstep with the register-interface inputs.
    pub params: Vec<ArgMemKind>,
    /// Whether the prototype takes a trailing `...` (unknowable pointer args).
    #[serde(default)]
    pub variadic: bool,
}

/// C-prototype-derived call interface for an external (imported, bodyless)
/// function, planned once by `external_sigs` and consumed by
/// `argpromote_external`, which needs neither the binary nor `cabi` afterwards.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExternInterface {
    /// The ordered call slots, one per positional argument.
    pub args: Vec<ExternArg>,
}

/// Optional ABI description attached to a function.
/// All fields are `Option` — only provided fields affect analysis.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct FunctionSignature {
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
    /// `true` once this function's returned values are a deterministic function of
    /// its by-value params, with no value flowing in from outside the SSA graph:
    /// no loads (an untracked memory read), no calls, no architecture p-code ops,
    /// and no raw register/global reads. Stores are permitted — they produce no
    /// value, so they cannot feed a returned field. Strictly stronger than a
    /// materialized register interface (`is_reg_materialized`), which only asserts the
    /// register channel is functionalized. Pure-function emulation in constant propagation gates on
    /// this (see `PURE_EMULATION_DESIGN.md`): such a callee may be emulated to
    /// harvest constant return-tuple fields, with the call left in place. Asserted
    /// by argpromote's `mark_pure`; checked by a `verify/` rule.
    #[serde(default)]
    pub is_pure: bool,
    /// Per-parameter pointer attributes ([`readonly`](ParamAttrs::readonly) /
    /// [`nocapture`](ParamAttrs::nocapture)), indexed like the positional call
    /// arguments (`Call.args`) — which for a functionalized (`pure_reg`) callee
    /// align with its root block params, and for an external align with the
    /// materialized register interface.
    ///
    /// `None` means "not analyzed" (fully conservative — every pointer arg
    /// escapes and is written through). A present vector may still be shorter than
    /// the argument list; a missing entry is also treated conservatively. Set by
    /// the extern C-prototype path ([`readonly`](ParamAttrs::readonly) only) and
    /// the bottom-up `param_attrs` inference pass. Dropped (and re-inferred) when
    /// argpromote/`dead_signature` rewrites the parameter list.
    #[serde(default)]
    pub param_attrs: Option<Vec<ParamAttrs>>,
    /// C-prototype-derived call interface for an **external** callee: the ordered
    /// argument slots (register or stack) and variadic flag, planned once by
    /// `external_sigs`. `argpromote_external` reads this to rewrite call sites
    /// without re-consulting `cabi` or the binary. `None` for non-externals and
    /// externals with no known prototype. Serde-defaulted, so older `.harbinger`
    /// snapshots load with it absent.
    #[serde(default)]
    pub extern_interface: Option<ExternInterface>,
    /// C-prototype-derived argmem summary for a prototyped **external** callee:
    /// the ordered per-input pointer kinds and variadic flag, planned once by
    /// `external_sigs`. The RAM effect channel's `external_leaf` reads this to
    /// bound the external's memory footprint through the pointers we pass it,
    /// instead of treating every external as unbounded (⊤). `None` for
    /// non-externals and externals with no known prototype. Serde-defaulted, so
    /// older `.harbinger` snapshots load with it absent.
    #[serde(default)]
    pub argmem: Option<ExternArgmem>,
}
