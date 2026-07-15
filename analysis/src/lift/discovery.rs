//! Driver helpers for recursive lifting through an injected lifter service.
//!
//! These helpers mutate the raw clean context only. The optimized context is
//! derived from that clean context and may enqueue new discoveries, but raw blocks
//! are never spliced into optimized IR.

use qcode::{context::Context, discovery::DiscoveryKind};

use crate::{
    Pass, PipelineEnv,
    pipeline::{LiftOutcome, LiftSummary, PipelineServices},
};

pub fn discover_addresses_in_binary(
    clean_ctx: &mut Context,
    services: &mut PipelineServices<'_>,
) -> Result<LiftSummary, String> {
    let Some(lifter) = services.lifter.as_deref_mut() else {
        return Ok(LiftSummary::default());
    };

    let entries = lifter.seed_binary(clean_ctx)?;
    let mut summary = LiftSummary::default();
    for entry in entries {
        if clean_ctx.discover(entry) {
            summary.enqueued += 1;
        }
    }
    Ok(summary)
}

pub fn lift_new_addresses(
    clean_ctx: &mut Context,
    services: &mut PipelineServices<'_>,
) -> Result<LiftSummary, String> {
    let Some(lifter) = services.lifter.as_deref_mut() else {
        return Ok(LiftSummary::default());
    };

    let pending = clean_ctx.drain_discoveries();
    if pending.is_empty() {
        return Ok(LiftSummary::default());
    }

    // Pre-register every pending function entry before lifting any block, so a
    // direct branch into one of these entries is recognized as a tail call (left
    // out of the branching function) regardless of the order discoveries drain in.
    for discovery in &pending {
        if matches!(discovery.kind, DiscoveryKind::Function { .. }) {
            lifter.ensure_discovered_function(clean_ctx, discovery);
        }
    }

    let mut summary = LiftSummary::default();
    for discovery in pending {
        let outcome = lifter.lift_discovered(clean_ctx, discovery)?;
        match outcome {
            LiftOutcome::Lifted { key, successors } => {
                clean_ctx.mark_discovery_lifted(key);
                summary.lifted += 1;
                for successor in successors {
                    if clean_ctx.discover(successor) {
                        summary.enqueued += 1;
                    }
                }
            }
            LiftOutcome::AlreadyLifted { key, successors } => {
                clean_ctx.mark_discovery_lifted(key);
                summary.already_lifted += 1;
                for successor in successors {
                    if clean_ctx.discover(successor) {
                        summary.enqueued += 1;
                    }
                }
            }
            LiftOutcome::Failed { key, reason } => {
                clean_ctx.mark_discovery_failed(key, reason);
                summary.failed += 1;
            }
            LiftOutcome::Skipped { key, reason } => {
                clean_ctx.mark_discovery_skipped(key, reason);
                summary.skipped += 1;
            }
        }
    }
    Ok(summary)
}

#[derive(Default)]
pub struct DiscoverAddressesInBinary;

impl Pass for DiscoverAddressesInBinary {
    const NAME: &'static str = "discover_addresses_in_binary";
    fn description(&self) -> &'static str {
        "Seed the binary's entry points as discovered functions for the lifter"
    }
    fn run(
        &self,
        _ctx: &mut Context,
        _env: &PipelineEnv,
        _targets: &[qcode::value::FunctionId],
    ) -> Result<crate::ModulePassOutcome, String> {
        Ok(crate::ModulePassOutcome::default())
    }
}

crate::register_module_pass!(DiscoverAddressesInBinary);

#[derive(Default)]
pub struct LiftNewAddresses;

impl Pass for LiftNewAddresses {
    const NAME: &'static str = "lift_new_addresses";
    fn description(&self) -> &'static str {
        "Lift pending discovered addresses into the raw clean IR"
    }
    fn run(
        &self,
        _ctx: &mut Context,
        _env: &PipelineEnv,
        _targets: &[qcode::value::FunctionId],
    ) -> Result<crate::ModulePassOutcome, String> {
        Ok(crate::ModulePassOutcome::default())
    }
}

crate::register_module_pass!(LiftNewAddresses);
