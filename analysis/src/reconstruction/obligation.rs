//! The reconstruction obligation database: what reconstruction still owes an
//! answer for, what it tried, and why the attempt failed.
//!
//! # The two-part split
//!
//! An obligation has a *set* and a *history*, and they are stored very
//! differently on purpose.
//!
//! The **set** is derived, never stored. [`enumerate_obligations`] recovers it
//! by scanning for live `BranchInd` / `CallInd` terminators. That makes the
//! milestone's exit criterion — every live indirect transfer has an obligation —
//! true *by construction*: there is no add/remove bookkeeping that could drift,
//! and an obligation that gets resolved into a real branch simply stops being
//! enumerated. The alternative (an authoritative stored set kept in sync with
//! the IR) has a desync failure mode this design cannot have.
//!
//! The **history** is stored, because it is the part the pipeline destroys.
//! Address discovery runs on a *disposable clone* of clean IR
//! (`pipeline/mod.rs`: `let mut ctx = clean.clone()`), and `Shared` is cloned by
//! value — so everything the jump-table pass learns dies with that clone unless
//! it rides an explicit channel back, exactly as discoveries do via
//! `drain_discoveries`. [`ObligationDb::merge_round`] is that channel's landing
//! point.
//!
//! # What this is not
//!
//! This is not an assumption engine. `Proposition` and the checkpoint/replay
//! driver remain the source of truth for boolean claims a transformation
//! consumed; obligations record evidence *around* them and reference them by
//! description. Building a competing engine is explicitly out of scope.
//!
//! # Scheduling gate
//!
//! Nothing here drives requeue yet. Dependency-directed replay — "a changed
//! fact requeues exactly the affected obligations" — needs the fact/generation
//! dependency graph, which is a later milestone. Until then the pipeline's
//! existing whole-function fingerprint remains the only retry trigger, and
//! [`ObligationStatus::is_retryable`] describes *eligibility*, not a trigger.

use std::collections::BTreeMap;

use qcode::{
    context::Context,
    obligation::{ObligationKey, TransferKind},
    value::{FunctionBody, insn::Mnemonic},
};

/// Lifecycle of an obligation.
///
/// The central distinction is [`is_retryable`](ObligationStatus::is_retryable):
/// a retryable failure may be attempted again when something it depended on
/// changes, whereas a permanent rejection must not be, or the discovery loop
/// re-does provably futile work every round until it hits its budget cap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObligationStatus {
    /// Not yet attempted, or attempted inconclusively. The initial state.
    Pending,

    /// Fully resolved to a known, exhaustive target set.
    Resolved { targets: Vec<u64> },

    /// Resolved, but only under a hypothesis that could later be contradicted
    /// (an immutable-memory assumption over the table, say). Distinct from
    /// `Resolved` because replay may have to undo it.
    ConditionallyResolved {
        targets: Vec<u64>,
        assumption: String,
    },

    /// A resolver examined this site and could not resolve it, but a change in
    /// its inputs could. Retryable — value ranges narrow as optimization
    /// proceeds, so this is the common case, not an error.
    Unresolved { reason: String },

    /// Provably not resolvable by any current resolver — the target expression
    /// is not of a shape any recognizer models. Not retryable on input changes
    /// alone; a new recognizer is what would change this.
    Rejected { reason: String },

    /// The resolver hit a limit (table-entry cap, round budget) rather than a
    /// semantic wall. Retryable, and distinguished from `Unresolved` because the
    /// fix is a bigger budget, not better information.
    BudgetExhausted { limit: String },
}

impl ObligationStatus {
    /// Whether a later attempt could plausibly succeed.
    ///
    /// `Resolved` is *not* retryable: it is a terminal success. Mirrors the
    /// terminal-state gating `DiscoveryQueue` uses to stop requeueing settled
    /// work.
    pub fn is_retryable(&self) -> bool {
        match self {
            ObligationStatus::Pending
            | ObligationStatus::Unresolved { .. }
            | ObligationStatus::BudgetExhausted { .. } => true,
            ObligationStatus::Resolved { .. }
            | ObligationStatus::ConditionallyResolved { .. }
            | ObligationStatus::Rejected { .. } => false,
        }
    }

    /// Whether reconstruction has an answer, however hedged.
    pub fn is_resolved(&self) -> bool {
        matches!(
            self,
            ObligationStatus::Resolved { .. } | ObligationStatus::ConditionallyResolved { .. }
        )
    }

    /// Rank used to pick a winner when a round reports a different outcome than
    /// the one on record. Higher wins. See [`ObligationDb::merge_round`].
    fn precedence(&self) -> u8 {
        match self {
            ObligationStatus::Pending => 0,
            ObligationStatus::BudgetExhausted { .. } => 1,
            ObligationStatus::Unresolved { .. } => 2,
            ObligationStatus::Rejected { .. } => 3,
            ObligationStatus::ConditionallyResolved { .. } => 4,
            ObligationStatus::Resolved { .. } => 5,
        }
    }
}

/// What reconstruction knows about one unresolved transfer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Obligation {
    pub key: ObligationKey,
    pub status: ObligationStatus,
    /// Entry address of the owning function, when it has one. Synthetic
    /// functions lack an address and cannot be re-lifted from the image.
    pub function: Option<u64>,
    /// How many rounds have attempted this site. Attempts that never reach a
    /// resolver (an indirect call today) do not count.
    pub attempts: u32,
    /// Which pass last reported on this obligation, for diagnostics.
    pub last_producer: Option<&'static str>,
}

impl Obligation {
    pub fn pending(key: ObligationKey, function: Option<u64>) -> Self {
        Self {
            key,
            status: ObligationStatus::Pending,
            function,
            attempts: 0,
            last_producer: None,
        }
    }

    /// A one-line explanation suitable for a report or log.
    pub fn explain(&self) -> String {
        let what = match &self.status {
            ObligationStatus::Pending if !self.key.kind.has_resolver() => {
                "pending (no resolver exists for indirect calls yet)".to_string()
            }
            ObligationStatus::Pending => "pending (not yet attempted)".to_string(),
            ObligationStatus::Resolved { targets } => {
                format!("resolved to {} target(s)", targets.len())
            }
            ObligationStatus::ConditionallyResolved {
                targets,
                assumption,
            } => format!(
                "resolved to {} target(s) under assumption: {assumption}",
                targets.len()
            ),
            ObligationStatus::Unresolved { reason } => format!("unresolved: {reason}"),
            ObligationStatus::Rejected { reason } => format!("rejected: {reason}"),
            ObligationStatus::BudgetExhausted { limit } => {
                format!("budget exhausted: {limit}")
            }
        };
        format!("{} {what} [{} attempt(s)]", self.key, self.attempts)
    }
}

/// Every obligation reconstruction currently knows about.
///
/// `BTreeMap` rather than a hash map so iteration is address-ordered and
/// reports are byte-for-byte reproducible across runs — the same reason
/// `DiscoveryQueue` uses one.
#[derive(Clone, Debug, Default)]
pub struct ObligationDb {
    records: BTreeMap<ObligationKey, Obligation>,
}

impl ObligationDb {
    pub fn get(&self, key: &ObligationKey) -> Option<&Obligation> {
        self.records.get(key)
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Address-ordered iteration.
    pub fn iter(&self) -> impl Iterator<Item = &Obligation> + '_ {
        self.records.values()
    }

    /// Obligations still owing an answer, excluding inert ones that no resolver
    /// can attempt. This is the honest "unresolved transfer" count for a report:
    /// counting indirect calls here would present a missing *feature* as a
    /// reconstruction failure.
    pub fn outstanding(&self) -> impl Iterator<Item = &Obligation> + '_ {
        self.records
            .values()
            .filter(|o| !o.status.is_resolved() && o.key.kind.has_resolver())
    }

    /// Inert obligations: live indirect transfers no resolver can attempt yet.
    pub fn inert(&self) -> impl Iterator<Item = &Obligation> + '_ {
        self.records.values().filter(|o| !o.key.kind.has_resolver())
    }

    /// Fold one round's observations into the database.
    ///
    /// `observed` is the full set enumerated from this round's IR, plus whatever
    /// outcomes the round's resolvers reported. Semantics:
    ///
    /// * A key absent from `observed` is **dropped** — the transfer no longer
    ///   exists in the IR, so the obligation was discharged by rewriting.
    /// * A key present in both keeps the higher-precedence status, so a
    ///   round that merely re-enumerates a site as `Pending` cannot erase a
    ///   real outcome learned earlier. This is what makes the database
    ///   accumulate across rounds despite the disposable context.
    /// * `attempts` accumulates whenever the round actually reported an outcome.
    pub fn merge_round(&mut self, observed: impl IntoIterator<Item = Obligation>) {
        let mut next: BTreeMap<ObligationKey, Obligation> = BTreeMap::new();

        for fresh in observed {
            // A round "attempted" a site when it reported something other than
            // the bare re-enumeration every live transfer produces.
            let attempted = !matches!(fresh.status, ObligationStatus::Pending);

            // `next` first: a single round reports the same key more than once
            // (enumeration yields `Pending` for every live site, then a resolver
            // reports its real outcome), and those duplicates must go through
            // the same precedence rule as cross-round merges — otherwise the
            // last writer wins and a resolver's finding can be erased by the
            // bare enumeration that accompanied it.
            let merged = match next
                .remove(&fresh.key)
                .or_else(|| self.records.remove(&fresh.key))
            {
                None => Obligation {
                    attempts: fresh.attempts + u32::from(attempted),
                    ..fresh
                },
                Some(prior) => {
                    let attempts = prior.attempts + u32::from(attempted);
                    let keep_fresh =
                        fresh.status.precedence() >= prior.status.precedence() && attempted;
                    Obligation {
                        key: fresh.key,
                        status: if keep_fresh {
                            fresh.status
                        } else {
                            prior.status
                        },
                        // The fresh scan knows the current owning function; a
                        // function split can move a site between functions.
                        function: fresh.function.or(prior.function),
                        attempts,
                        last_producer: if attempted {
                            fresh.last_producer
                        } else {
                            prior.last_producer
                        },
                    }
                }
            };
            next.insert(merged.key, merged);
        }

        self.records = next;
    }
}

/// Recover the obligation set from the IR: every live indirect transfer.
///
/// Terminator-only by construction — `BranchInd` and `CallInd` are both
/// terminators, so scanning each block's last instruction is complete rather
/// than an optimization.
///
/// The site address falls back to the block's when the terminator carries none,
/// mirroring the jump-table pass's `dispatch_source_addr`. The instruction
/// address is preferred because a block's start address is not stable under the
/// straight-line merging the optimized clone performs. A transfer with neither
/// is skipped: without a stable address there is no key, and such a site cannot
/// be re-lifted from the image anyway — the same reason the jump-table pass's
/// `discover` is a no-op for addressless functions.
pub fn enumerate_obligations(ctx: &Context<'_>) -> Vec<Obligation> {
    let mut found = Vec::new();

    for function in ctx.functions().filter(|f| !f.is_external()) {
        let body = FunctionBody::from_id(ctx, function.id);
        let entry = body.address();

        for block in body.blocks() {
            let Some(insn) = block.instructions().last() else {
                continue;
            };
            let kind = match insn.mnemonic() {
                Mnemonic::BranchInd(_) => TransferKind::IndirectBranch,
                Mnemonic::CallInd(_) => TransferKind::IndirectCall,
                _ => continue,
            };
            let Some(site) = insn.address().or_else(|| block.address()) else {
                continue;
            };
            found.push(Obligation::pending(ObligationKey { site, kind }, entry));
        }
    }

    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolved(key: ObligationKey, targets: &[u64]) -> Obligation {
        Obligation {
            status: ObligationStatus::Resolved {
                targets: targets.to_vec(),
            },
            last_producer: Some("handle_jump_tables"),
            ..Obligation::pending(key, Some(0x1000))
        }
    }

    fn unresolved(key: ObligationKey, reason: &str) -> Obligation {
        Obligation {
            status: ObligationStatus::Unresolved {
                reason: reason.to_string(),
            },
            last_producer: Some("handle_jump_tables"),
            ..Obligation::pending(key, Some(0x1000))
        }
    }

    #[test]
    fn resolved_is_terminal_but_unresolved_is_retryable() {
        assert!(!ObligationStatus::Resolved { targets: vec![] }.is_retryable());
        assert!(
            ObligationStatus::Unresolved {
                reason: String::new()
            }
            .is_retryable()
        );
        assert!(ObligationStatus::Pending.is_retryable());
    }

    #[test]
    fn rejection_is_permanent_but_budget_exhaustion_is_not() {
        // The whole point of separating these: a shape no recognizer models will
        // not become resolvable by re-running, but a table that blew the entry
        // cap might under a bigger budget.
        assert!(
            !ObligationStatus::Rejected {
                reason: String::new()
            }
            .is_retryable()
        );
        assert!(
            ObligationStatus::BudgetExhausted {
                limit: String::new()
            }
            .is_retryable()
        );
    }

    #[test]
    fn conditional_resolution_counts_as_resolved() {
        let status = ObligationStatus::ConditionallyResolved {
            targets: vec![0x10],
            assumption: "immutable memory".to_string(),
        };
        assert!(status.is_resolved());
        assert!(!status.is_retryable());
    }

    #[test]
    fn a_round_that_only_re_enumerates_does_not_erase_an_outcome() {
        // The load-bearing property. Every round re-enumerates every live site
        // as `Pending` from a fresh clone; without precedence, round N+1 would
        // wipe what round N learned and the database would never accumulate.
        let key = ObligationKey::branch(0x400);
        let mut db = ObligationDb::default();
        db.merge_round([resolved(key, &[0x10, 0x20])]);
        db.merge_round([Obligation::pending(key, Some(0x1000))]);

        assert!(db.get(&key).expect("kept").status.is_resolved());
    }

    #[test]
    fn duplicates_within_one_round_obey_precedence() {
        // The pipeline feeds enumeration and resolver outcomes into the same
        // round, so both orderings must land on the resolved status.
        let key = ObligationKey::branch(0x400);

        let mut enumeration_first = ObligationDb::default();
        enumeration_first.merge_round([
            Obligation::pending(key, Some(0x1000)),
            resolved(key, &[0x10]),
        ]);

        let mut outcome_first = ObligationDb::default();
        outcome_first.merge_round([
            resolved(key, &[0x10]),
            Obligation::pending(key, Some(0x1000)),
        ]);

        assert!(
            enumeration_first
                .get(&key)
                .expect("present")
                .status
                .is_resolved()
        );
        assert!(
            outcome_first
                .get(&key)
                .expect("present")
                .status
                .is_resolved()
        );
        assert_eq!(enumeration_first.get(&key).expect("present").attempts, 1);
        assert_eq!(outcome_first.get(&key).expect("present").attempts, 1);
    }

    #[test]
    fn a_disappearing_transfer_discharges_its_obligation() {
        let key = ObligationKey::branch(0x400);
        let mut db = ObligationDb::default();
        db.merge_round([unresolved(key, "unbounded index")]);
        assert_eq!(db.len(), 1);

        // Next round the branch was rewritten to a real CBranch, so it is no
        // longer enumerated.
        db.merge_round([]);
        assert!(db.is_empty());
    }

    #[test]
    fn attempts_accumulate_only_on_real_attempts() {
        let key = ObligationKey::branch(0x400);
        let mut db = ObligationDb::default();

        db.merge_round([Obligation::pending(key, Some(0x1000))]);
        assert_eq!(db.get(&key).expect("present").attempts, 0);

        db.merge_round([unresolved(key, "unbounded index")]);
        db.merge_round([unresolved(key, "unbounded index")]);
        assert_eq!(db.get(&key).expect("present").attempts, 2);
    }

    #[test]
    fn a_later_success_replaces_an_earlier_failure() {
        let key = ObligationKey::branch(0x400);
        let mut db = ObligationDb::default();
        db.merge_round([unresolved(key, "unbounded index")]);
        db.merge_round([resolved(key, &[0x10])]);

        let record = db.get(&key).expect("present");
        assert!(record.status.is_resolved());
        assert_eq!(record.attempts, 2);
    }

    #[test]
    fn a_later_failure_does_not_unresolve_a_success() {
        // Precedence, not recency. A round that re-attempts an already-resolved
        // site under a restricted analysis set must not downgrade it.
        let key = ObligationKey::branch(0x400);
        let mut db = ObligationDb::default();
        db.merge_round([resolved(key, &[0x10])]);
        db.merge_round([unresolved(key, "value range widened")]);

        assert!(db.get(&key).expect("present").status.is_resolved());
    }

    #[test]
    fn indirect_calls_are_inert_not_outstanding() {
        // An indirect call has no resolver, so counting it as an outstanding
        // reconstruction failure would report a missing feature as a defect.
        let branch = ObligationKey::branch(0x400);
        let call = ObligationKey::call(0x500);
        let mut db = ObligationDb::default();
        db.merge_round([
            Obligation::pending(branch, Some(0x1000)),
            Obligation::pending(call, Some(0x1000)),
        ]);

        assert_eq!(db.outstanding().count(), 1);
        assert_eq!(db.inert().count(), 1);
        assert_eq!(db.len(), 2);
    }

    #[test]
    fn a_resolved_branch_is_not_outstanding() {
        let key = ObligationKey::branch(0x400);
        let mut db = ObligationDb::default();
        db.merge_round([resolved(key, &[0x10])]);
        assert_eq!(db.outstanding().count(), 0);
    }

    #[test]
    fn iteration_is_address_ordered() {
        let mut db = ObligationDb::default();
        db.merge_round([
            Obligation::pending(ObligationKey::branch(0x900), None),
            Obligation::pending(ObligationKey::branch(0x100), None),
            Obligation::pending(ObligationKey::call(0x500), None),
        ]);

        let sites: Vec<u64> = db.iter().map(|o| o.key.site).collect();
        assert_eq!(sites, vec![0x100, 0x500, 0x900]);
    }

    #[test]
    fn explain_names_the_missing_resolver_for_indirect_calls() {
        let record = Obligation::pending(ObligationKey::call(0x400), None);
        assert!(record.explain().contains("no resolver"));
    }

    #[test]
    fn explain_carries_the_failure_reason() {
        let record = unresolved(
            ObligationKey::branch(0x400),
            "index range unbounded for width",
        );
        assert!(record.explain().contains("index range unbounded for width"));
    }

    mod enumeration {
        use super::*;
        use wazabin_qcode_macro::qcode;

        /// Two indirect transfers, in addressed blocks reached from `<entry>`.
        fn both_kinds() -> Context<'static> {
            let mut ctx = Context::new();
            qcode!(
                ctx,
                "
                varnode i64 A;
                fn fun:
                <entry>
                    %p = load(A:8, &A);
                    if %p goto <0x1010> else goto <0x1020>;
                <0x1010>
                    goto [%p];
                <0x1020>
                    call [%p];
                "
            );
            ctx
        }

        #[test]
        fn finds_both_indirect_transfer_kinds() {
            let ctx = both_kinds();
            let mut keys: Vec<_> = enumerate_obligations(&ctx).iter().map(|o| o.key).collect();
            keys.sort();

            assert_eq!(
                keys,
                vec![ObligationKey::branch(0x1010), ObligationKey::call(0x1020)]
            );
        }

        #[test]
        fn the_site_falls_back_to_the_block_address() {
            // Textual IR carries no per-instruction addresses, so this exercises
            // the `dispatch_source_addr`-style fallback rather than the
            // preferred instruction-address path.
            let ctx = both_kinds();
            let found = enumerate_obligations(&ctx);
            assert!(found.iter().all(|o| o.key.site != 0));
        }

        #[test]
        fn ignores_resolved_control_flow() {
            // A function with only direct transfers owes nothing. This is what
            // makes the set self-discharging: once the jump-table pass rewrites
            // a `BranchInd` into a real branch, it stops being enumerated.
            let mut ctx = Context::new();
            qcode!(
                ctx,
                "
                fn fun:
                <entry>
                    goto <0x1010>;
                <0x1010>
                    return at 0x1000;
                "
            );

            assert!(enumerate_obligations(&ctx).is_empty());
        }

        #[test]
        fn every_enumerated_obligation_starts_pending() {
            let ctx = both_kinds();
            let found = enumerate_obligations(&ctx);
            assert_eq!(found.len(), 2);
            assert!(found.iter().all(|o| o.status == ObligationStatus::Pending));
            assert!(found.iter().all(|o| o.attempts == 0));
        }

        #[test]
        fn enumeration_feeds_the_database_directly() {
            // The two halves compose: the derived set is exactly what
            // `merge_round` consumes, with no translation step between them.
            // Only the branch is outstanding; the indirect call is inert.
            let ctx = both_kinds();
            let mut db = ObligationDb::default();
            db.merge_round(enumerate_obligations(&ctx));

            assert_eq!(db.len(), 2);
            assert_eq!(db.outstanding().count(), 1);
            assert_eq!(db.inert().count(), 1);
        }
    }
}
