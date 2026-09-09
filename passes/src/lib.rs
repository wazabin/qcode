//! Block-local cleanup passes over the qcode IR.
//!
//! These are the transforms that a *lifter* wants while it is still building a
//! function: they are cheap, local, and safe to run on a partially discovered
//! CFG. They deliberately need no alias analysis, no calling convention, and no
//! knowledge of the target architecture — which is what lets them sit below
//! `qcode_analysis` and be used on their own by [`qcode_vm`] and other
//! consumers that only ever want the cleanup, never the decompiler.
//!
//! ## The context view
//!
//! A pass reads the module through a [`PassCtx`] and mutates a single
//! [`FunctionBody`] borrowed `&mut` from the bodies registry. Splitting a
//! [`Context`] into those two halves is what [`with_body_mut`] does. Because
//! `PassCtx` holds only shared references it is `Copy`, so a caller hands the
//! same view to every helper.
//!
//! `PassCtx` is the environment-free half of the richer view used by the full
//! analysis pipeline: it carries the module's shared IR state and published
//! interfaces, but no architecture configuration. Arch-aware passes live in
//! `qcode_analysis` and take the richer view instead.
//!
//! [`qcode_vm`]: https://docs.rs/qcode_vm
//!
//! # Example
//!
//! A pure instruction nothing reads is removed; one whose result is used is
//! kept.
//!
//! ```
//! use qcode::{context::Context, qcode};
//! use qcode_passes::remove_dead_insns;
//!
//! let mut ctx = Context::new();
//! qcode!(
//!     ctx,
//!     "
//!     fn f:
//!     <entry>
//!         %unused = i64 0x2 + 0x3;
//!         %kept   = i64 0x4 + 0x5;
//!         goto <exit @r=%kept>;
//!     <exit @r:i64>
//!         goto <0x1001>;
//!     "
//! );
//!
//! assert!(remove_dead_insns(&mut ctx, entry));
//! // Running it again is a no-op: the pass is idempotent.
//! assert!(!remove_dead_insns(&mut ctx, entry));
//! ```

use jstd::registry::Registry;
use qcode::{
    context::{Context, Shared},
    value::{
        BodyView, FunctionBody, FunctionId, function::FunctionInterface, util::body_mut::BodyMut,
    },
};

pub mod cfg;
pub mod dce;
pub mod symbolize;
mod terminator;

pub use cfg::absorb_straight_line;
pub use dce::{dead_insns, remove_dead_insns, remove_dead_insns_body};
pub use symbolize::{resolve_addresses, resolve_strings};
pub use terminator::replace_terminator_with_branch;

/// The bodies-free module view a block-local pass reads through:
/// `{shared, interfaces}`.
///
/// A pass structurally cannot reach another function's body through it, which
/// is what makes holding one the proof that the shared state is frozen while a
/// worker holds a disjoint `&mut` body. `Copy`, since it holds only shared
/// references.
#[derive(Clone, Copy)]
pub struct PassCtx<'ctx, 'str> {
    shared: &'ctx Shared<'str>,
    interfaces: &'ctx Registry<FunctionId, FunctionInterface<'str>>,
}

impl<'ctx, 'str> PassCtx<'ctx, 'str> {
    /// Build a view over `ctx`'s shared state and interfaces. The bodies are
    /// **not** captured; prefer [`split`], which proves that with a
    /// simultaneous `&mut` bodies borrow.
    pub fn new(ctx: &'ctx Context<'str>) -> Self {
        Self {
            shared: &ctx.shared,
            interfaces: &ctx.interfaces,
        }
    }

    /// Assemble a view from already-borrowed parts. This is the seam richer
    /// views (such as the analysis pipeline's) use to hand their own
    /// `{shared, interfaces}` to the passes here.
    pub fn from_parts(
        shared: &'ctx Shared<'str>,
        interfaces: &'ctx Registry<FunctionId, FunctionInterface<'str>>,
    ) -> Self {
        Self { shared, interfaces }
    }

    /// The module's shared IR state (interners, spaces, registers, name and
    /// address maps, memory image, truths).
    pub fn shr(&self) -> &'ctx Shared<'str> {
        self.shared
    }

    /// The published interface of function `f`.
    pub fn interface(&self, f: FunctionId) -> &'ctx FunctionInterface<'str> {
        &self.interfaces[f]
    }

    /// The whole interface registry.
    pub fn interfaces(&self) -> &'ctx Registry<FunctionId, FunctionInterface<'str>> {
        self.interfaces
    }

    /// Build the static read view for a pass's borrowed body.
    pub fn body_view<'body>(self, body: &'body FunctionBody<'str>) -> BodyView<'body, 'str>
    where
        'ctx: 'body,
    {
        BodyView::new(body, self.shared, self.interfaces)
    }

    /// Build the mutation host for a pass's exclusively borrowed body.
    pub fn host<'body>(self, body: &'body mut FunctionBody<'str>) -> BodyMut<'body, 'str>
    where
        'ctx: 'body,
    {
        BodyMut::new(body, self.shared, self.interfaces)
    }
}

/// Split a context into its mutable bodies registry and the read-only module
/// view, so a worker can hold one body `&mut` while reading shared state.
pub fn split<'a, 'str>(
    ctx: &'a mut Context<'str>,
) -> (
    &'a mut Registry<FunctionId, FunctionBody<'str>>,
    PassCtx<'a, 'str>,
) {
    (
        &mut ctx.bodies,
        PassCtx {
            shared: &ctx.shared,
            interfaces: &ctx.interfaces,
        },
    )
}

/// Run `f` against function `fid`'s body borrowed `&mut` out of `ctx`, with the
/// matching read-only view.
///
/// This is the `&mut Context` entry point the block-local passes expose to
/// callers that hold a whole context and neither a [`FunctionBody`] nor a view.
pub fn with_body_mut<'str, R>(
    ctx: &mut Context<'str>,
    fid: FunctionId,
    f: impl FnOnce(&mut FunctionBody<'str>, PassCtx<'_, 'str>) -> R,
) -> R {
    let (bodies, view) = split(ctx);
    f(&mut bodies[fid], view)
}
