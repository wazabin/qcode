//! The provenance-aware unpacker, as a library.
//!
//! `main.rs` is a thin CLI over this module tree, and `userland/tests/unpack.rs`
//! pulls the same tree in with
//! `#[path = "../examples/unpack/lib.rs"] mod unpack;`, so the example and
//! its tests share one body of code without a published crate.
//!
//! The pipeline is four steps, one module each:
//!
//! 1. [`layout`] fixes where the hooks keep their state and which guest bytes
//!    provenance covers;
//! 2. [`hooks`] rewrites the lifted code so that every guest store stamps its
//!    own identity into the shadow and every block logs its first entry,
//!    without ever stopping the machine;
//! 3. [`driver`] runs a [`qcode_userland::Process`] with those hooks
//!    installed;
//! 4. [`graph`] joins the final IR, the final shadow and the log into a
//!    provenance graph, which [`artifact`] writes out.

// The two consumers reach different parts of the tree: `main.rs` runs a
// program, the tests inspect the layout and the graph. Neither is expected to
// touch all of it, so unused items are not a defect here the way they would
// be in a crate with one caller.
#![allow(dead_code)]

pub mod artifact;
pub mod driver;
pub mod graph;
pub mod hooks;
pub mod layout;
