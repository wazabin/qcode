//! Built-in pure intrinsics.
//!
//! Each submodule defines one family of intrinsics (their evaluators,
//! recognizers, simplifiers) and self-registers them with the global registry
//! via [`register_intrinsic!`](crate::register_intrinsic). The registry
//! machinery, the [`IntrinsicId`] handle and the [`Intrinsic`] mnemonic itself
//! live in [`crate::value::insn`]; this module is purely the catalogue of
//! concrete intrinsics.
//!
//! # Adding an intrinsic
//!
//! Add a new `mod` here (e.g. `rotate`), implement its `eval` (and optionally
//! `recognize` / `simplify`) functions, then `register_intrinsic! { … }` for
//! each name. The `inventory`-based registry picks it up automatically — no
//! central table to edit.

mod rotate;
