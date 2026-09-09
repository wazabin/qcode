//! Where a run's time and work actually go.
//!
//! A machine like this has three phases, and they have wildly different costs
//! per occurrence and wildly different frequencies:
//!
//! * **Fetch** — reading instruction bytes out of guest memory, through the
//!   MMU's execute permission.
//! * **Decode and lift** — turning those bytes into QCode, plus the cleanup
//!   round over the result.
//! * **Execute** — interpreting the QCode.
//!
//! Fetch and lift happen once per *distinct* instruction; execute happens once
//! per instruction *executed*. So the split depends entirely on the workload: a
//! tight loop is almost all execute, while a long straight-line run is dominated
//! by lifting. Reporting one number for "throughput" without saying which
//! workload produced it is how a benchmark misleads.
//!
//! Counting is always on and costs a few increments. Timing is only taken
//! around lifting, which is rare; putting a clock read around each interpreter
//! step would cost more than the step. Execute time is therefore derived —
//! wall clock minus the measured phases — rather than measured directly.

use std::time::Duration;

/// Counters and timings for one machine's run.
#[derive(Debug, Default, Clone)]
pub struct Stats {
    /// P-code operations retired.
    pub steps: u64,
    /// Times an address had to be lifted: a translation-cache **miss**.
    pub lifts: u64,
    /// Times execution left a block and the target was already lifted: a
    /// translation-cache **hit** at the VM level.
    ///
    /// Both counters only see transitions that reach the VM. Most control flow
    /// never does: a branch inside the lifted graph is a direct block
    /// reference the interpreter follows without consulting any address index,
    /// so a steady-state loop performs *no* translation-cache lookups at all.
    /// Read a low `resolves` next to a high `steps` as "control flow is
    /// already resolved", not as "the cache is missing".
    pub resolves: u64,
    /// Block bodies executed by an installed [`BlockExecutor`](crate::BlockExecutor)
    /// rather than interpreted.
    pub native_bodies: u64,
    /// Blocks folded into a predecessor as a guest basic block was discovered,
    /// each one a unit the machine no longer enters and leaves separately.
    pub absorbed: u64,
    /// Lifted blocks emptied because the guest wrote over their bytes.
    pub evicted: u64,
    /// Instruction bytes read from guest memory.
    pub fetch_bytes: u64,
    /// Time spent reading instruction bytes.
    pub fetch: Duration,
    /// Time spent decoding and lowering to QCode.
    pub decode_lift: Duration,
    /// Time spent in the block-local cleanup round.
    pub optimize: Duration,
    /// Loads forwarded to their stored value by the cleanup round.
    pub forwarded_loads: u64,
    /// Stores removed by the cleanup round.
    pub removed_stores: u64,
}

impl Stats {
    /// Total time in the fetch and translation phases.
    pub fn translation(&self) -> Duration {
        self.fetch + self.decode_lift + self.optimize
    }

    /// The share of `elapsed` spent executing rather than translating.
    ///
    /// Derived, so it also absorbs whatever the VM spends on its own
    /// bookkeeping; it is an upper bound on interpretation cost, not an exact
    /// measure of it.
    pub fn execute(&self, elapsed: Duration) -> Duration {
        elapsed.saturating_sub(self.translation())
    }

    /// Translation-cache hit rate over the transitions the VM observed.
    /// `None` when there were no transitions to rate.
    pub fn hit_rate(&self) -> Option<f64> {
        let total = self.lifts + self.resolves;
        (total > 0).then(|| self.resolves as f64 / total as f64)
    }

    /// A one-line breakdown suitable for a benchmark to print.
    pub fn report(&self, elapsed: Duration) -> String {
        let percent = |part: Duration| {
            if elapsed.is_zero() {
                0.0
            } else {
                part.as_secs_f64() / elapsed.as_secs_f64() * 100.0
            }
        };
        format!(
            "steps={} lifts={} (translated) resolves={} (re-entered) lookup_hit_rate={} | \
             fetch={:?} ({:.1}%) decode+lift={:?} ({:.1}%) optimize={:?} ({:.1}%) \
             execute={:?} ({:.1}%)",
            self.steps,
            self.lifts,
            self.resolves,
            self.hit_rate()
                .map_or_else(|| "n/a".to_owned(), |rate| format!("{:.1}%", rate * 100.0)),
            self.fetch,
            percent(self.fetch),
            self.decode_lift,
            percent(self.decode_lift),
            self.optimize,
            percent(self.optimize),
            self.execute(elapsed),
            percent(self.execute(elapsed)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execute_time_is_what_translation_did_not_take() {
        let stats = Stats {
            fetch: Duration::from_millis(10),
            decode_lift: Duration::from_millis(30),
            optimize: Duration::from_millis(10),
            ..Stats::default()
        };
        assert_eq!(stats.translation(), Duration::from_millis(50));
        assert_eq!(
            stats.execute(Duration::from_millis(200)),
            Duration::from_millis(150)
        );
    }

    #[test]
    fn execute_time_never_goes_negative() {
        // Timer skew must not produce a nonsense figure.
        let stats = Stats {
            decode_lift: Duration::from_millis(100),
            ..Stats::default()
        };
        assert_eq!(stats.execute(Duration::from_millis(10)), Duration::ZERO);
    }

    #[test]
    fn hit_rate_needs_transitions_to_rate() {
        assert_eq!(Stats::default().hit_rate(), None);
        let stats = Stats {
            lifts: 1,
            resolves: 3,
            ..Stats::default()
        };
        assert_eq!(stats.hit_rate(), Some(0.75));
    }
}
