//! The channel that carries obligation outcomes out of the disposable analysis
//! context.
//!
//! Address discovery runs on a clone of clean IR that is dropped at the end of
//! every round, and `Shared` is cloned by value — so a resolver's findings
//! vanish with that clone unless something explicitly carries them back. The
//! discovery queue solves this by being drained into `clean`; obligations have
//! no home in `Context` at all (they are derived analysis state, deliberately
//! kept out of qcode core), so they ride on [`PipelineEnv`] instead, which the
//! pipeline owns and outlives the clone.
//!
//! [`PipelineEnv`]: crate::pipeline::PipelineEnv

use std::sync::Mutex;

use super::Obligation;

/// Where resolvers report what they learned about an obligation.
///
/// `Mutex`, not `RefCell`: `&PipelineEnv` is shared across worker threads by the
/// parallel function-pass driver and must stay `Sync`. Contention is a
/// non-issue — a lock is taken once per *attempted indirect transfer*, which is
/// a vanishingly small fraction of pass work.
///
/// Records arrive in whatever order the round produced them.
/// [`ObligationDb::merge_round`](super::ObligationDb::merge_round) keys them, so
/// order carries no meaning and nondeterministic interleaving cannot affect the
/// resulting database.
#[derive(Default, Debug)]
pub struct ObligationSink {
    records: Mutex<Vec<Obligation>>,
}

impl ObligationSink {
    /// Report an outcome for one obligation.
    ///
    /// A poisoned lock is ignored rather than propagated: obligations are
    /// diagnostic state, and losing a record must never turn into a pass
    /// failure that aborts reconstruction.
    pub fn record(&self, obligation: Obligation) {
        if let Ok(mut records) = self.records.lock() {
            records.push(obligation);
        }
    }

    /// Remove and return everything reported so far.
    pub fn take(&self) -> Vec<Obligation> {
        self.records
            .lock()
            .map(|mut records| std::mem::take(&mut *records))
            .unwrap_or_default()
    }

    pub fn is_empty(&self) -> bool {
        self.records.lock().map(|r| r.is_empty()).unwrap_or(true)
    }
}

#[cfg(test)]
mod tests {
    use qcode::obligation::ObligationKey;

    use super::*;
    use crate::reconstruction::ObligationStatus;

    fn unresolved(site: u64) -> Obligation {
        Obligation {
            status: ObligationStatus::Unresolved {
                reason: "test".to_string(),
            },
            ..Obligation::pending(ObligationKey::branch(site), None)
        }
    }

    #[test]
    fn take_drains() {
        let sink = ObligationSink::default();
        sink.record(unresolved(0x100));
        sink.record(unresolved(0x200));

        assert_eq!(sink.take().len(), 2);
        assert!(sink.is_empty());
        assert!(sink.take().is_empty());
    }

    #[test]
    fn is_shareable_across_threads() {
        // Guards the `Sync` requirement: `&PipelineEnv` crosses worker threads,
        // so a `RefCell` here would fail to compile the parallel driver.
        fn assert_sync<T: Sync>() {}
        assert_sync::<ObligationSink>();

        let sink = ObligationSink::default();
        std::thread::scope(|scope| {
            for i in 0..4 {
                let sink = &sink;
                scope.spawn(move || sink.record(unresolved(0x100 * (i + 1))));
            }
        });

        assert_eq!(sink.take().len(), 4);
    }
}
