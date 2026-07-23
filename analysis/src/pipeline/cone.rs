//! The cone-gated write handle for module-scoped passes (Milestone 3, step 4).
//!
//! A module [`Pass`](super::Pass) may *read* the whole program but may only
//! *write* to the functions in its **cone**. [`ConeMut`] is the capability that
//! enforces this: it owns the `&mut Context` and a [`Cone`], and it is the only
//! thing a module pass receives. Its mutation accessors assert cone membership on
//! the owning function and **panic** on a violation — a pass bug, not a
//! recoverable `Err`, mirroring [`BodyMut`]'s identity assert. Reads are
//! unrestricted: [`ConeMut::ctx`] hands out `&Context` freely.
//!
//! In this commit the cone is always [`Cone::Full`], so every assert passes and
//! the handle is behaviorally inert. The restriction becomes real when the
//! incremental driver (step 5) computes a non-full cone.
//!
//! [`BodyMut`]: qcode::value::util::body_mut::BodyMut

use qcode::{
    assumption::{AssumedCallEffect, Proposition},
    context::Context,
    discovery::Discovery,
    types::{TypeId, TypeManager},
    value::{FunctionBody, FunctionId, FunctionMutRef, VarnodeId},
};
use rustc_hash::FxHashSet;

/// The set of functions a module pass may write this run.
///
/// [`Cone::Full`] is the whole program (every function id in the context) and
/// contains everything in O(1). [`Cone::Set`] restricts to an explicit set, the
/// shape the incremental driver produces once it slices the module.
#[derive(Debug, Clone)]
pub enum Cone {
    /// Every function in the context — the inert, whole-program cone.
    Full,
    /// An explicit set of writable functions.
    Set(FxHashSet<FunctionId>),
}

impl Cone {
    /// Whether `f` is writable under this cone. O(1) either way.
    pub fn contains(&self, f: FunctionId) -> bool {
        match self {
            Cone::Full => true,
            Cone::Set(set) => set.contains(&f),
        }
    }
}

/// The only handle a module [`Pass`](super::Pass) receives: a `&mut Context`
/// wrapped so that writes are gated on the [`Cone`] while reads stay free.
///
/// It never hands an unrestricted `&mut Context` to a pass. Interface stamping
/// and body edits go through the cone-checked accessors
/// ([`function_mut`](Self::function_mut), [`ctx_for`](Self::ctx_for)); reads go
/// through [`ctx`](Self::ctx). Free operations that the cone does not gate
/// (minting, discovery queueing, shared truth/proposition writes — ruling 4's
/// module-scoped propositions already force whole-program replay) are reached
/// through the dedicated cone-free accessors below.
pub struct ConeMut<'ctx, 'str> {
    ctx: &'ctx mut Context<'str>,
    cone: Cone,
}

impl<'ctx, 'str> ConeMut<'ctx, 'str> {
    /// A handle over `ctx` with the whole-program [`Cone::Full`] — the inert
    /// cone every module stage runs with until step 5 slices it.
    pub fn full(ctx: &'ctx mut Context<'str>) -> Self {
        Self {
            ctx,
            cone: Cone::Full,
        }
    }

    /// A handle over `ctx` restricted to `cone`.
    pub fn new(ctx: &'ctx mut Context<'str>, cone: Cone) -> Self {
        Self { ctx, cone }
    }

    /// The cone this handle enforces.
    pub fn cone(&self) -> &Cone {
        &self.cone
    }

    /// Whether `f` is writable under this handle's cone.
    pub fn contains(&self, f: FunctionId) -> bool {
        self.cone.contains(f)
    }

    /// Read the whole module. Reads are never cone-restricted.
    pub fn ctx(&self) -> &Context<'str> {
        self.ctx
    }

    /// The functions this handle may write — the iteration surface that
    /// subsumes the old `targets` slice. `Cone::Full` yields every function id in
    /// the context; `Cone::Set` yields the set (in context order for determinism).
    pub fn cone_functions(&self) -> Vec<FunctionId> {
        match &self.cone {
            Cone::Full => self.ctx.function_ids(),
            Cone::Set(set) => self
                .ctx
                .function_ids()
                .into_iter()
                .filter(|id| set.contains(id))
                .collect(),
        }
    }

    // --- Cone-free shared-state accessors ------------------------------------
    //
    // These reach state the cone does *not* gate: module-scoped shared truths,
    // discovery queueing, minting, global memory protections, global varnode
    // types, the calling-convention cache. Ruling 4 makes the four module-scoped
    // propositions force whole-program replay, so no per-function gate applies to
    // them; interning a type or queueing a discovery is likewise not a
    // per-function write. Each delegates to `Context` and, crucially, never hands
    // a pass a `&mut Context` through which it could reach an out-of-cone body.

    /// Assume `prop` true (see [`Context::assume_true`]). Returns whether it was
    /// newly recorded or already held with the same polarity.
    pub fn assume_true(&mut self, prop: Proposition) -> bool {
        self.ctx.assume_true(prop)
    }

    /// Assume `prop` true only if no truth for it exists yet, returning whether
    /// this call actually mutated the truth map. A pass outcome must report only
    /// a real mutation as changed, so this is the reporting-correct variant of
    /// [`assume_true`](Self::assume_true).
    pub fn assume_true_if_new(&mut self, prop: Proposition) -> bool {
        self.ctx.truth(prop).is_none() && self.ctx.assume_true(prop)
    }

    /// Cache the assumed calling-convention effect for indirect/unresolved calls
    /// (see [`Context::set_assumed_call_convention`]).
    pub fn set_assumed_call_convention(&mut self, effect: Option<AssumedCallEffect>) {
        self.ctx.set_assumed_call_convention(effect);
    }

    /// Mark the binary's per-segment memory protections authoritative (see
    /// [`Context::mark_protections_known`]). Program-global, cone-free.
    pub fn mark_protections_known(&mut self) {
        self.ctx.mark_protections_known();
    }

    /// Set a *global* varnode's type (see [`Context::set_varnode_type`]). Varnode
    /// types live in shared state, not in any function body, so this is cone-free.
    pub fn set_varnode_type(&mut self, varnode: VarnodeId, type_id: TypeId) {
        self.ctx.set_varnode_type(varnode, type_id);
    }

    /// Queue a discovered code address for the lifter (see [`Context::discover`]).
    /// Returns whether the discovery was newly added.
    pub fn discover(&mut self, discovery: Discovery) -> bool {
        self.ctx.discover(discovery)
    }

    /// Record a synthetic call-graph edge `caller → callee_addr`. Returns whether
    /// it was newly added. Edge bookkeeping lives in shared state, not a body.
    pub fn add_synthetic_callee(&mut self, caller: FunctionId, callee_addr: u64) -> bool {
        self.ctx
            .shared
            .values
            .add_synthetic_callee(caller, callee_addr)
    }

    /// The program-global type registry, mutable — the cone-free minting surface.
    /// Interning/looking-up a type is shared-state work, not a per-function write,
    /// so it is not gated.
    pub fn types_mut(&mut self) -> &mut TypeManager {
        &mut self.ctx.shared.types
    }

    /// Cone-checked mutable access to `f`'s published interface (the effect /
    /// purity / signature setters). Panics if `f` is out of cone.
    pub fn function_mut(&mut self, f: FunctionId) -> FunctionMutRef<'str, '_> {
        self.assert_in_cone(f);
        FunctionBody::from_id_mut(self.ctx, f)
    }

    /// Cone-checked whole-`Context` mutation surface for a body edit on `f`.
    /// Panics if `f` is out of cone. This is *the* migrated-pass surface for
    /// transforms whose verbs are `Context`-inherent (instruction replacement,
    /// pure-body inlining, call-site rewrites), which a `FunctionMutRef` cannot
    /// express. The assert is the tripwire a step-5 slice trips when a pass
    /// reaches out of its cone.
    ///
    /// # Contract
    ///
    /// This is a **trusted single-function write surface**. The gate *checks* one
    /// thing — that `f` is in the cone — and then *trusts* the caller on
    /// everything else:
    ///
    /// * **Writes must stay within `f`'s body.** The returned handle is the whole
    ///   module, so a write to some *other* function's body is neither caught nor
    ///   stopped; it is a caller bug. A coordinated multi-function edit (e.g. a
    ///   callee interface plus its callers' call sites) is expressed as a sequence
    ///   of `ctx_for` calls, one per owning function, each individually asserted
    ///   before its write — never one `ctx_for` standing in for several functions.
    /// * **Reads of any function are always fine.** Cloning a callee's expression
    ///   into its caller, scanning the whole module — reads are never cone-gated.
    /// * **Shared and global writes do not belong here.** Truths/propositions,
    ///   discovery queueing, type interning, calling-convention caches and other
    ///   program-global state have dedicated cone-free accessors above; route them
    ///   there, not through this handle.
    ///
    /// Narrowing the return type so out-of-`f` writes become structurally
    /// unrepresentable is a future refactor; until then the single-function
    /// contract is a discipline the caller upholds, not one the type enforces.
    pub fn ctx_for(&mut self, f: FunctionId) -> &mut Context<'str> {
        self.assert_in_cone(f);
        self.ctx
    }

    fn assert_in_cone(&self, f: FunctionId) {
        assert!(
            self.cone.contains(f),
            "a module pass may not write function {f:?}: it is outside the pass's cone"
        );
    }

    /// Raw `&mut Context` for pipeline **infrastructure** only (the driver and the
    /// `module(<fn>)` adapter's split/barrier dance). Not reachable from pass
    /// implementations, which live outside `crate::pipeline`.
    pub(in crate::pipeline) fn ctx_mut(&mut self) -> &mut Context<'str> {
        self.ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode_macro::qcode;

    fn two_function_ctx() -> (Context<'static>, FunctionId, FunctionId) {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn in_cone:
            <a> return at i64 0;
            fn out_of_cone:
            <b> return at i64 0;
            "
        );
        (ctx, in_cone, out_of_cone)
    }

    #[test]
    fn full_cone_contains_everything() {
        let (mut ctx, a, b) = two_function_ctx();
        let cone = ConeMut::full(&mut ctx);
        assert!(cone.contains(a));
        assert!(cone.contains(b));
        assert_eq!(cone.cone_functions(), vec![a, b]);
    }

    #[test]
    fn partial_cone_restricts_membership_and_iteration() {
        let (mut ctx, a, b) = two_function_ctx();
        let set: FxHashSet<FunctionId> = [a].into_iter().collect();
        let cone = ConeMut::new(&mut ctx, Cone::Set(set));
        assert!(cone.contains(a));
        assert!(!cone.contains(b));
        assert_eq!(cone.cone_functions(), vec![a]);
    }

    /// The write handle is not vacuous: an in-cone interface write succeeds and an
    /// out-of-cone one panics on the membership assert.
    #[test]
    #[should_panic(expected = "outside the pass's cone")]
    fn out_of_cone_interface_write_panics() {
        let (mut ctx, a, b) = two_function_ctx();
        let set: FxHashSet<FunctionId> = [a].into_iter().collect();
        let mut cone = ConeMut::new(&mut ctx, Cone::Set(set));
        // In-cone write is allowed.
        cone.function_mut(a).set_is_pure(true);
        // Out-of-cone write must panic.
        cone.function_mut(b).set_is_pure(true);
    }

    #[test]
    #[should_panic(expected = "outside the pass's cone")]
    fn out_of_cone_body_surface_panics() {
        let (mut ctx, a, b) = two_function_ctx();
        let set: FxHashSet<FunctionId> = [a].into_iter().collect();
        let mut cone = ConeMut::new(&mut ctx, Cone::Set(set));
        let _ = cone.ctx_for(b);
    }
}
