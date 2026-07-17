//! Heuristic *assumptions* and proven *knowledge* shared by analysis passes.
//!
//! A [`Proposition`] is a positive statement about the program ("function `f`
//! returns to its caller"). During analysis each proposition can be in one of
//! four states, tracked by a [`Truth`] in a single map on the
//! [`Context`](crate::context::Context):
//!
//! - **assumed true / assumed false** — a pass guessed, recorded via
//!   [`Context::assume_true`](crate::context::Context::assume_true) /
//!   [`assume_false`](crate::context::Context::assume_false). Assuming fails
//!   (returns `false`) if the opposite polarity is already assumed or known.
//! - **known true / known false** — proven by a verification pass via
//!   [`Context::set_known`](crate::context::Context::set_known). Proving the
//!   opposite of an existing assumption records a [`Violation`].
//!
//! Every entry carries the name of the pass that recorded it, picked up
//! automatically from the [`pass_scope`](crate::pass_scope) thread-local set by
//! the pipeline driver.
//!
//! Because analysis passes mutate the [`Context`](crate::context::Context)
//! arena in place, a *violated* assumption leaves behind IR that is now
//! incorrect. The invalidation strategy is **checkpoint + replay**: a
//! freshly-lifted baseline `Context` is cloned before any speculative
//! analysis; after a round, if any violation was recorded (or a novel fact
//! proven), the working copy is discarded, its known facts are seeded into a
//! fresh clone, and the round replays. Knowledge only ever grows, so replay
//! terminates.

use crate::value::VarnodeId;
use crate::value::function::FunctionId;

/// The register-space effect the opt-in [`Proposition::AssumeCallingConvention`]
/// hypothesis assigns to an indirect / unresolved call, precomputed once from the
/// module's calling convention by the `assume_calling_convention` pass and cached
/// on the [`Shared`](crate::context::Shared) context so the mem2reg / alias
/// register classifier can consult it without an ABI in hand.
///
/// `reads` is *all* convention argument registers (integer + SSE) — reads-all-args
/// keeps pre-call argument setup live, since a variadic-arity callee may consume
/// any of them — and `writes` is the convention's caller-saved (volatile) set. The
/// stack- and frame-pointer varnodes are excluded from both, consistent with the
/// rest of the register channel.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssumedCallEffect {
    /// Argument registers the call is assumed to read.
    pub reads: Vec<VarnodeId>,
    /// Caller-saved registers the call is assumed to write (clobber).
    pub writes: Vec<VarnodeId>,
}

/// A positive statement about the program whose truth a pass may assume or
/// prove. Used as the key of the truth map on the
/// [`Context`](crate::context::Context).
///
/// `#[non_exhaustive]` so adding future propositions does not break exhaustive
/// matches in downstream crates.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Proposition {
    /// Function returns normally to the fall-through after a call to it
    /// (false = noreturn: `exit`/`abort`, infinite loops, no `Return` in body).
    FunctionReturns(FunctionId),
    /// Function performs an unresolved/dynamic stack read, or forwards a stack
    /// pointer into one, so callers must keep their frame in memory.
    UnboundedStackReader(FunctionId),
    /// Function hands a pointer into its own frame to an unbounded-reading
    /// callee, so its own frame must stay in memory.
    FrameEscapingCaller(FunctionId),
    /// Every incoming pointer parameter of this function (and any address offset
    /// from one) is disjoint from the function's own *caller-frame* region — the
    /// `@SP + k` (`k ≥ 0`) slots holding the return address and incoming stack
    /// arguments. This lets the frame-freshness alias rule forward a load of a
    /// caller-frame slot across a store through an incoming pointer (the
    /// spilled-pointer reload idiom). Unlike own-frame freshness it is **not**
    /// statically sound on its own — a caller could pass the address of one of its
    /// outgoing-argument slots — so a pass records it `Assumed` and a verifier
    /// (`verify_args_disjoint_caller_frame`) keeps it standing by default,
    /// refuting it (→ replay) only when a direct caller *provably* passes a pointer
    /// argument whose access interval overlaps the callee's argument slots.
    ArgsDisjointFromCallerFrame(FunctionId),
    /// Within this function, a pointer **loaded from a slot** does not alias that
    /// slot: a store through `load(X) + …` cannot clobber `X` itself — the buffer a
    /// pointer addresses does not overlap the storage of the pointer. This lets the
    /// alias rule (see [`AliasResult::provably_disjoint`]) forward a spilled buffer
    /// pointer's in-loop reload across the very store that writes *through* it,
    /// which `argpromote` needs to region-promote a dynamic-index buffer loop.
    /// Like [`ArgsDisjointFromCallerFrame`] it is **not** statically sound on its
    /// own — it fails only for a self-referential pointer (`*pp == &pp`), which real
    /// code does not build — so a pass records it `Assumed`; v1 has no verifier
    /// (nothing currently proves the negation), the checkpoint+replay net catching
    /// any future refutation.
    ///
    /// [`AliasResult::provably_disjoint`]: crate
    LoadedPointerDisjointFromSlot(FunctionId),
    /// The source and destination buffers of a recognized copy/transform loop in
    /// this function do not overlap — the C `strcpy`/`memcpy` contract, where
    /// overlapping buffers are undefined behaviour (that is `memmove`'s job).
    ///
    /// A copy-until-terminator (`while (*src) *dst++ = *src++;`) loop pipelines
    /// its loaded byte through a loop-carried register, and re-deriving that byte
    /// as `load(src - step)` — the move that removes the carry and exposes the
    /// `map(body, take_while(src))` shape — re-reads memory that the body also
    /// writes through `dst`. That re-read is value-preserving only when the two
    /// buffers are disjoint. Like [`ArgsDisjointFromCallerFrame`] it is **not**
    /// statically sound on its own — a caller may pass overlapping pointers — so a
    /// pass records it `Assumed`; v1 has no verifier (the checkpoint+replay net
    /// catches any future refutation).
    CopyBuffersDisjoint(FunctionId),
    /// The `size` bytes at virtual address `addr` (a jump-table entry the
    /// jump-table resolver read out of read-only data) are assumed never
    /// written at runtime; a write would invalidate the resolved jump target.
    ImmutableMemory { addr: u64, size: u8 },
    /// The mapped region `[start, end)` is executable code. The lifter assumes
    /// this for any readable byte (default r/x) until the optional
    /// `memory_protections` pass establishes the binary's real protections; a
    /// target in a region the pass marks non-executable contradicts it. Keyed by
    /// the whole containing segment (not per byte) so the truth map stays small.
    ExecutableMemory { start: u64, end: u64 },
    /// The binary is a Windows target of the given `bitness` (32 or 64), so the
    /// segment-base register (`FS` on x86, `GS` on x64) points at the Thread
    /// Information Block. Recorded `Assumed` by the TEB-seeding pass to justify
    /// retyping that base as `PtrTo<TEB>`; it is an analyst aid and override
    /// hook, with no verifier in v1 (nothing currently proves the negation).
    WindowsTeb { bitness: u8 },
    /// Whole-program, opt-in hypothesis: every indirect (`CallInd`) and
    /// unresolved-direct call obeys the module's calling convention, so instead
    /// of clobbering the entire register file it reads only the convention's
    /// argument registers and writes only its caller-saved set (the effect is
    /// cached as an [`AssumedCallEffect`] on the context). A deliberate,
    /// *controllable unsoundness* — an indirect callee may violate the ABI — off
    /// by default, recorded by the `assume_calling_convention` pass when the user
    /// opts in, and surfaced in the assumptions panel.
    ///
    /// It is a **downstream refinement only**: it sharpens how the mem2reg / alias
    /// register classifier (`classify_call_reg_effect`) clobbers around such
    /// calls, and does *not* feed back into the argpromote effect-summary fixpoint
    /// (which keeps modelling `CallInd` as `Some(empty)`). No verifier in v1 (it is
    /// never proven, so it is not discharged and the checkpoint+replay net never
    /// acts on it).
    AssumeCallingConvention,
}

/// How certain we are about a proposition's recorded value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Certainty {
    /// A pass's heuristic guess; can be contradicted by [`set_known`]
    /// (recording a [`Violation`]).
    ///
    /// [`set_known`]: crate::context::Context::set_known
    Assumed,
    /// Proven by a verification pass; survives checkpoint+replay rounds.
    Known,
}

/// The recorded truth state of one [`Proposition`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Truth {
    /// The polarity recorded for the proposition.
    pub value: bool,
    /// Guess or proven fact.
    pub certainty: Certainty,
    /// The pass that recorded this entry (from [`crate::pass_scope`]).
    pub pass: PassName,
}

/// A proven fact contradicting an earlier assumption — the signal that the
/// checkpoint+replay driver must discard the working copy and replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Violation {
    /// The contradicted proposition.
    pub prop: Proposition,
    /// The polarity that was assumed (the proven value is its negation).
    pub assumed: bool,
    /// The pass that made the wrong assumption.
    pub assuming_pass: PassName,
    /// The verification pass that proved the opposite.
    pub asserting_pass: PassName,
}

/// A proven fact that contradicted an existing *known* fact (as opposed to a
/// mere assumption). Unlike a [`Violation`], this is not a replay signal — it
/// means two verification results, or a user-forced override and a verification
/// result, disagree irreconcilably. The checkpoint+replay driver surfaces it as
/// a hard error rather than looping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct KnownContradiction {
    /// The proposition proven two different ways.
    pub prop: Proposition,
    /// The value already recorded as known (e.g. a user override).
    pub known: bool,
    /// The value a later pass proved (the negation of `known`).
    pub proven: bool,
    /// The pass that recorded the original known fact.
    pub known_pass: PassName,
    /// The pass that proved the contradicting value.
    pub proven_pass: PassName,
}

/// A pass name, as recorded on truth-map entries. A transparent `&'static str`
/// wrapper: serde's derive would otherwise tie the deserializer lifetime to
/// `'static`, so it gets manual impls — serialized as a string, deserialized by
/// leaking. Pass names form a small finite set, so the leak is bounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PassName(pub &'static str);

impl std::ops::Deref for PassName {
    type Target = str;
    fn deref(&self) -> &str {
        self.0
    }
}

impl std::fmt::Display for PassName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl PartialEq<&str> for PassName {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

impl serde::Serialize for PassName {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.0)
    }
}

impl<'de> serde::Deserialize<'de> for PassName {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Ok(PassName(Box::leak(s.into_boxed_str())))
    }
}
