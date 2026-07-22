//! First-class reconstruction obligations: the unresolved control transfers
//! reconstruction still owes an answer for, with their status and evidence.
//!
//! Derived analysis state, owned by the pipeline. Nothing here is serialized
//! into qcode bodies; only the stable [`ObligationKey`](qcode::obligation::ObligationKey)
//! identity lives in `qcode` core.

pub mod obligation;
pub mod sink;

pub use obligation::{Obligation, ObligationDb, ObligationStatus, enumerate_obligations};
pub use sink::ObligationSink;
