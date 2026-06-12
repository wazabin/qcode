//! Thread-local identity of the currently-running analysis pass.
//!
//! The pipeline driver brackets every pass invocation in a [`PassScope`] guard.
//! While the guard is alive, [`current_pass`] names the pass, so the
//! assumption API ([`Context::assume_true`](crate::context::Context::assume_true)
//! and friends), the [`pass_log!`](crate::pass_log) macro, and the
//! [`stat!`](crate::stat) counters attribute their records to the right pass
//! without any plumbing through call signatures — free helper functions get
//! the attribution too.
//!
//! Outside any scope, [`current_pass`] reports `"?"` so stray records remain
//! visible (and greppable) rather than silently unattributed.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

thread_local! {
    static CURRENT_PASS: Cell<&'static str> = const { Cell::new(UNATTRIBUTED) };
    static STATS: RefCell<BTreeMap<(&'static str, &'static str), u64>> =
        const { RefCell::new(BTreeMap::new()) };
}

/// The name reported outside any [`PassScope`].
pub const UNATTRIBUTED: &str = "?";

/// The name of the pass currently running on this thread (or
/// [`UNATTRIBUTED`]).
pub fn current_pass() -> &'static str {
    CURRENT_PASS.get()
}

/// RAII guard naming the current pass; restores the previous name on drop, so
/// scopes nest (a driver phase can wrap individual passes).
pub struct PassScope {
    prev: &'static str,
}

/// Enter a pass scope. Hold the returned guard for the duration of the pass.
pub fn enter(name: &'static str) -> PassScope {
    PassScope {
        prev: CURRENT_PASS.replace(name),
    }
}

impl Drop for PassScope {
    fn drop(&mut self) {
        CURRENT_PASS.set(self.prev);
    }
}

/// Add `n` to the counter `key` of the current pass. Prefer the
/// [`stat!`](crate::stat) macro.
pub fn record_stat(key: &'static str, n: u64) {
    STATS.with_borrow_mut(|stats| {
        *stats.entry((current_pass(), key)).or_insert(0) += n;
    });
}

/// Drain all counters accumulated on this thread since the last drain, as
/// `((pass, key), total)` in sorted order. The pipeline driver calls this once
/// per round to log a statistics table.
pub fn drain_stats() -> Vec<((&'static str, &'static str), u64)> {
    STATS
        .with_borrow_mut(|stats| std::mem::take(stats))
        .into_iter()
        .collect()
}

/// Log a message attributed to the current pass: the `log` target is the pass
/// name (so `RUST_LOG=mem2reg=debug` filters per pass) and the message is
/// prefixed with it.
///
/// Level conventions: `debug` = what the pass did, `trace` = per-instruction
/// detail, `warn` = suspicious but continuing; `info` is reserved for the
/// pipeline driver.
///
/// ```ignore
/// pass_log!(debug, "promoted {} stack slots", n);
/// ```
#[macro_export]
macro_rules! pass_log {
    ($lvl:ident, $($arg:tt)+) => {{
        let __pass = $crate::pass_scope::current_pass();
        $crate::__log::$lvl!(target: __pass, "[{}] {}", __pass, format_args!($($arg)+));
    }};
}

/// Add to a named counter of the current pass; the driver aggregates and logs
/// them per round. `stat!("slots_promoted")` increments by 1.
///
/// ```ignore
/// stat!("slots_promoted", promoted.len() as u64);
/// ```
#[macro_export]
macro_rules! stat {
    ($key:expr) => {
        $crate::pass_scope::record_stat($key, 1)
    };
    ($key:expr, $n:expr) => {
        $crate::pass_scope::record_stat($key, $n)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scopes_nest_and_restore() {
        assert_eq!(current_pass(), UNATTRIBUTED);
        {
            let _outer = enter("outer");
            assert_eq!(current_pass(), "outer");
            {
                let _inner = enter("inner");
                assert_eq!(current_pass(), "inner");
            }
            assert_eq!(current_pass(), "outer");
        }
        assert_eq!(current_pass(), UNATTRIBUTED);
    }

    #[test]
    fn stats_accumulate_per_pass_and_drain() {
        let _ = drain_stats();
        {
            let _s = enter("p1");
            record_stat("k", 2);
            record_stat("k", 3);
        }
        {
            let _s = enter("p2");
            record_stat("k", 1);
        }
        let drained = drain_stats();
        assert_eq!(drained, vec![(("p1", "k"), 5), (("p2", "k"), 1)]);
        assert!(drain_stats().is_empty());
    }
}
