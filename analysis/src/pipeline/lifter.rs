//! The lifting boundary between analysis and the disassembler.
//!
//! `qcode_analysis` owns the driver, but `harbinger` owns binary formats,
//! decoders, import handling, and p-code emission. The driver therefore invokes a
//! caller-provided mutable lifter service instead of depending on `harbinger`.

use qcode::{
    address_index::AddressIndex,
    context::Context,
    discovery::{Discovery, DiscoveryKey},
};

pub struct PipelineServices<'a> {
    pub lifter: Option<&'a mut dyn Lifter>,
    /// Wall-clock deadline for the analyze/lift driver. Once it passes, no new
    /// discovery or optimization round starts: the driver returns the analysis
    /// of the most-grown IR best-effort (pending discoveries stay queued).
    /// `None` falls back to the fixed round cap (wasm has no monotonic clock).
    pub deadline: Option<std::time::Instant>,
}

impl<'a> PipelineServices<'a> {
    pub fn none() -> Self {
        Self {
            lifter: None,
            deadline: None,
        }
    }

    pub fn with_lifter(lifter: &'a mut dyn Lifter) -> Self {
        Self {
            lifter: Some(lifter),
            deadline: None,
        }
    }

    /// Set the wall-clock deadline (see [`PipelineServices::deadline`]).
    pub fn with_deadline(mut self, deadline: Option<std::time::Instant>) -> Self {
        self.deadline = deadline;
        self
    }
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiftSummary {
    pub lifted: usize,
    pub already_lifted: usize,
    pub failed: usize,
    pub skipped: usize,
    pub enqueued: usize,
}

impl LiftSummary {
    pub fn changed(self) -> bool {
        self.lifted > 0 || self.enqueued > 0
    }

    pub fn merge(&mut self, other: LiftSummary) {
        self.lifted += other.lifted;
        self.already_lifted += other.already_lifted;
        self.failed += other.failed;
        self.skipped += other.skipped;
        self.enqueued += other.enqueued;
    }
}

pub enum LiftOutcome {
    Lifted {
        key: DiscoveryKey,
        successors: Vec<Discovery>,
    },
    AlreadyLifted {
        key: DiscoveryKey,
        successors: Vec<Discovery>,
    },
    Failed {
        key: DiscoveryKey,
        reason: String,
    },
    Skipped {
        key: DiscoveryKey,
        reason: String,
    },
}

/// Injected disassembler/lifter. Implemented by `harbinger`.
pub trait Lifter {
    fn seed_binary(
        &mut self,
        ctx: &mut Context,
        addresses: &mut AddressIndex,
    ) -> Result<Vec<Discovery>, String>;

    /// Pre-register a function stub at `addr` (idempotent). The driver calls this
    /// for every pending `DiscoveryKind::Function` before lifting any block in a
    /// drain, so the tail-call boundary (a direct branch into a known entry stays
    /// out of the caller) is independent of the order discoveries drain in.
    fn ensure_function(&mut self, ctx: &mut Context, addresses: &mut AddressIndex, addr: u64);

    fn ensure_discovered_function(
        &mut self,
        ctx: &mut Context,
        addresses: &mut AddressIndex,
        discovery: &Discovery,
    ) {
        self.ensure_function(ctx, addresses, discovery.target);
    }

    fn lift_discovered(
        &mut self,
        clean_ctx: &mut Context,
        addresses: &mut AddressIndex,
        discovery: Discovery,
    ) -> Result<LiftOutcome, String>;
}
