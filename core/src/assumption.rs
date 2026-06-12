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

use crate::value::function::FunctionId;

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
