//! The lifting boundary between analysis and the disassembler.
//!
//! `qcode_analysis` owns the driver, but `harbinger` owns binary formats,
//! decoders, import handling, and p-code emission. The driver therefore invokes a
//! caller-provided mutable lifter service instead of depending on `harbinger`.

use qcode::{
    context::Context,
    discovery::{Discovery, DiscoveryKey},
};

pub struct PipelineServices<'a> {
    pub lifter: Option<&'a mut dyn Lifter>,
}

impl<'a> PipelineServices<'a> {
    pub fn none() -> Self {
        Self { lifter: None }
    }

    pub fn with_lifter(lifter: &'a mut dyn Lifter) -> Self {
        Self {
            lifter: Some(lifter),
        }
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
    fn seed_binary(&mut self, ctx: &mut Context) -> Result<Vec<Discovery>, String>;

    /// Pre-register a function stub at `addr` (idempotent). The driver calls this
    /// for every pending `DiscoveryKind::Function` before lifting any block in a
    /// drain, so the tail-call boundary (a direct branch into a known entry stays
    /// out of the caller) is independent of the order discoveries drain in.
    fn ensure_function(&mut self, ctx: &mut Context, addr: u64);

    fn ensure_discovered_function(&mut self, ctx: &mut Context, discovery: &Discovery) {
        self.ensure_function(ctx, discovery.target);
    }

    fn lift_discovered(
        &mut self,
        clean_ctx: &mut Context,
        discovery: Discovery,
    ) -> Result<LiftOutcome, String>;

    fn finish_lifting(&mut self, _clean_ctx: &mut Context) -> Result<(), String> {
        Ok(())
    }
}
