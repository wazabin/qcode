//! Exclusive access to one installed function body ([`BodyMut`]), and to
//! several at once ([`BodiesMut`]).
//!
//! [`BodyView`] gives the read layer a `Copy` static provider that
//! routes arena reads to the pass's own function. [`BodyMut`] is its mutable
//! sibling: a single function's arenas borrowed `&mut` in place, for
//! exclusive mutation by one worker (the driver's [`Context::split_bodies`](crate::context::Context::split_bodies)
//! hands out disjoint bodies).
//!
//! A [`Context`](crate::context::Context) never hands out a bare `&mut FunctionBody`, and never its
//! bodies registry mutably: what a holder does with one is not observable —
//! it could move a body out with `mem::replace`, swap it with a body of
//! another module that happens to use the same ids, or put back a clone
//! taken before a tracked deletion — and every one of those changes which
//! addresses the module covers without any tracked mutator running. So the
//! body behind a `BodyMut` is reachable only through its verbs, each of which
//! ticks the module's clock exactly when it changes an address, and for
//! reading (by [`Deref`]). Replacing the body is not an operation the API
//! has.
//!
//! All *shared* data (types, varnodes, spaces, registers, name map) stays behind
//! the `&Shared` view, reachable read-only. The inherent verb + read methods below
//! route each write to the borrowed function's arena; a function pass must not mutate another
//! function (asserted). The module-scope twin of every verb is an inherent method
//! on [`Context`](crate::context::Context); the shorter-lived reborrow needed to
//! hand the host to a value that owns it by value (a [`Builder`](crate::builder::Builder)
//! or a mutation `BaseRef`) is [`BodyMut::reborrow`].

use std::ops::{Deref, Index};

use jstd::registry::{self, Identified, Registry};

use crate::{
    context::Shared,
    error::{Error, ErrorTy, Result},
    value::{
        BlockParamRef, BlockRef, BodyView, FunctionBody, FunctionId, FunctionRef, InstructionRef,
        QCodeView, TempSpace, TempSpaceId, ValueId,
        block::{BasicBlock, BlockId, EdgeData, EdgeId},
        block_param::{BlockParam, BlockParamId},
        function::FunctionInterface,
        insn::{Instruction, InstructionId, Mnemonic},
    },
};

/// A single function borrowed `&mut` in place — through
/// [`Context::body_mut`](crate::context::Context::body_mut) or [`BodiesMut`] — for exclusive mutation (its
/// interface stays in the module's interface registry, reachable read-only
/// through `interfaces`).
///
/// Dereferences to the body for reading. The body itself is reachable
/// mutably only through the verbs, so it cannot be replaced or swapped
/// behind the module's revision:
///
/// ```compile_fail,E0596
/// # use qcode::context::Context;
/// let mut left = Context::new();
/// let mut right = Context::new();
/// let f = left.anon_function();
/// let _ = right.anon_function();
/// std::mem::swap(&mut *left.body_mut(f), &mut *right.body_mut(f));
/// ```
///
/// ```compile_fail,E0594
/// # use qcode::{context::Context, value::FunctionBody};
/// let mut ctx = Context::new();
/// let f = ctx.anon_function();
/// *ctx.body_mut(f) = FunctionBody::detached();
/// ```
///
/// ```compile_fail,E0616
/// # use qcode::context::Context;
/// let mut ctx = Context::new();
/// let f = ctx.anon_function();
/// let (mut bodies, _, _) = ctx.split_bodies();
/// let _ = std::mem::take(bodies.get_mut(f).fun);
/// ```
///
/// Out of scope (and asserted against on construction): a function with
/// *reattributed* blocks — a roster block stored in, or parented to, a different
/// function. Those functions go through the sequential (module) path.
pub struct BodyMut<'a, 'str> {
    pub(crate) fun: &'a mut FunctionBody<'str>,
    /// The module's shared IR state, **read-only**. A checked-out function pass
    /// reaches shared data (types, literals, spaces, registers) immutably; it
    /// mints types/literals through the interners' `&self` paths, and mints no
    /// varnodes/temp-spaces (only the V1 argpromote pass does, and it runs on
    /// the module path). Holds **no** `&Context` — bodies are out of reach by
    /// construction (context-split stage 5b-ii Pin B).
    pub shared: &'a crate::context::Shared<'str>,
    /// Every function's published interface (never checked out): the
    /// caller-reasoning surface a pass may consult about its callees.
    pub interfaces:
        &'a jstd::registry::Registry<FunctionId, crate::value::function::FunctionInterface<'str>>,
}

impl<'a, 'str> BodyMut<'a, 'str> {
    /// Wrap `fun` over the module's shared state and interface registry. Block
    /// ownership is now derived from the storing arena (a rostered block lives in
    /// `fun`'s own arena by construction), so there is no reattribution state left
    /// to scan for here.
    pub(crate) fn new(
        fun: &'a mut FunctionBody<'str>,
        shared: &'a crate::context::Shared<'str>,
        interfaces: &'a jstd::registry::Registry<
            FunctionId,
            crate::value::function::FunctionInterface<'str>,
        >,
    ) -> Self {
        Self {
            fun,
            shared,
            interfaces,
        }
    }

    /// A shorter-lived `BodyMut` reborrowing this one's exclusive references, so
    /// the host can be handed to a mutation ref (which owns its host by value)
    /// without consuming the original.
    pub fn reborrow(&mut self) -> BodyMut<'_, 'str> {
        BodyMut {
            fun: &mut *self.fun,
            shared: self.shared,
            interfaces: self.interfaces,
        }
    }

    /// Narrows this pass backing to the concrete body-local builder.
    pub fn builder(&mut self, block: BlockId) -> crate::builder::Builder<'str, '_> {
        crate::builder::Builder::new(&mut *self.fun, self.shared, self.interfaces, block)
    }
}

impl<'str> Deref for BodyMut<'_, 'str> {
    type Target = FunctionBody<'str>;

    fn deref(&self) -> &Self::Target {
        self.fun
    }
}

/// The verb + read surface of a checked-out function pass, delegating to the
/// owned `FunctionBody`'s inherent verbs and `self.shared`. The module-scope twin of
/// each verb is an inherent method on [`Context`]; the
/// primitives below (`function{,_mut}`/`shared`/`view`, and the no-op
/// call-site cache) are the checked-out specializations.
impl<'a, 'str> BodyMut<'a, 'str> {
    // ---- primitives ---------------------------------------------------------

    /// The owned function's storage (write), for this crate's verbs. Panics
    /// if `f` is not this function.
    pub(crate) fn function_mut(&mut self, f: FunctionId) -> &mut FunctionBody<'str> {
        assert_eq!(
            f,
            self.fun.id(),
            "a checked-out function pass may not mutate another function"
        );
        self.fun
    }
    /// The owned function's storage (read).
    pub fn function(&self, f: FunctionId) -> &FunctionBody<'str> {
        assert_eq!(
            f,
            self.fun.id(),
            "a checked-out function pass may not read another function's arenas mutably"
        );
        self.fun
    }
    /// The module's shared IR state ([`Shared`]) (read).
    ///
    /// [`Shared`]: crate::context::Shared
    pub fn shr(&self) -> &crate::context::Shared<'str> {
        self.shared
    }
    /// The static immutable provider for shared reads over this pass body.
    pub fn view(&self) -> BodyView<'_, 'str> {
        BodyView::new(&*self.fun, self.shared, self.interfaces)
    }
    // ---- function-scoped read wrappers --------------------------------------

    /// A read [`BlockRef`] over `id`, body-routed.
    pub fn block_ref(&self, id: BlockId) -> BlockRef<'str, '_, BodyView<'_, 'str>> {
        self.view().block_ref(id)
    }
    /// A read [`InstructionRef`] over `id`.
    pub fn insn_ref(&self, id: InstructionId) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        self.view().insn_ref(id)
    }
    /// A read [`BlockParamRef`] over `id`.
    pub fn param_ref(&self, id: BlockParamId) -> BlockParamRef<'str, '_, BodyView<'_, 'str>> {
        self.view().param_ref(id)
    }
    /// A read [`FunctionRef`] over `id`.
    pub fn function_ref(&self, id: FunctionId) -> FunctionRef<'str, '_, BodyView<'_, 'str>> {
        self.view().function_ref(id)
    }

    // ---- births -------------------------------------------------------------
    //
    // (The derived mut accessors — `instruction_mut`/`block_mut`/
    // `block_param_mut` — are [`QCodeMut`](crate::value::QCodeMut) defaults.)

    pub fn push_edge(&mut self, edge: EdgeData) -> EdgeId {
        self.fun.edges.push(edge)
    }

    pub fn push_insn(&mut self, insn: Instruction<'str>) -> InstructionId {
        self.fun.push_insn(insn)
    }

    pub fn push_block(&mut self, block: BasicBlock<'str>) -> BlockId {
        self.fun.push_block(block)
    }

    pub fn make_block(&mut self) -> BlockId {
        self.fun.make_block()
    }

    pub fn push_block_param(&mut self, param: BlockParam<'str>) -> BlockParamId {
        self.fun.push_block_param(param)
    }

    pub fn push_mnemonic(&mut self, mnemonic: Mnemonic, size: usize) -> InstructionId {
        let shared = self.shared;
        self.fun.push_mnemonic(shared, mnemonic, size)
    }

    pub fn push_mnemonic_with_type(
        &mut self,
        mnemonic: Mnemonic,
        type_id: crate::types::TypeId,
    ) -> InstructionId {
        self.fun.push_mnemonic_with_type(mnemonic, type_id)
    }

    /// Adds a function-local temporary space; see
    /// [`FunctionBody::push_temp_space`].
    pub fn push_temp_space(&mut self, space: TempSpace) -> TempSpaceId {
        self.fun.push_temp_space(space)
    }

    /// Resolves the pass-local minted-callee placeholders in this body once
    /// the minted functions are installed; see
    /// [`FunctionBody::resolve_minted_callees`].
    pub fn resolve_minted_callees(
        &mut self,
        installed: &[FunctionId],
    ) -> std::result::Result<usize, u32> {
        self.fun.resolve_minted_callees(installed)
    }

    // ---- CFG / use-map verbs ------------------------------------------------
    //
    // The body-local mutation verbs live on the [`QCodeMut`] trait
    // (`value::view_mut`), shared with the module host. Only the verbs whose
    // spelling diverges between hosts stay inherent here.

    /// Remove CFG edge `edge_id` (unqualified; `EdgeId` is body-local). The
    /// module-path twin is the function-qualified
    /// [`Context::remove_cfg_edge`](crate::context::Context::remove_cfg_edge).
    pub fn remove_cfg_edge(&mut self, edge_id: EdgeId) {
        self.fun.remove_cfg_edge(edge_id)
    }

    // ---- names --------------------------------------------------------------

    pub fn register_local_name(
        &mut self,
        id: ValueId,
        name: std::borrow::Cow<'str, str>,
        old_name: Option<&str>,
    ) -> Result<()> {
        let existing = match id.name_scope_function() {
            Some(func) => self
                .function(func)
                .names
                .get(&name)
                .map(|id| id.qualify(func)),
            None => self.shr().get_named(&name),
        };
        if let Some(existing) = existing {
            return if existing == id {
                Ok(())
            } else {
                Err(Error::spanless(ErrorTy::DuplicateName(name.to_string())))
            };
        }
        match id.name_scope_function() {
            Some(func) => self
                .function_mut(func)
                .names
                .register(name, id.localize(func), old_name),
            None => {
                unimplemented!("a checked-out host has read-only shared access (mints via &self)")
            }
        }
    }
}

/// The bodies of a module, borrowed for a driver that holds several
/// exclusively at once, alongside the frozen module state they read.
///
/// Reads index the registry as usual; mutable access is a [`BodyMut`] per
/// body, one ([`get_mut`](Self::get_mut)) or a disjoint set
/// ([`select_mut`](Self::select_mut)) at a time. There is no way to add,
/// remove or replace a slot: the registry stays in lockstep with the module's
/// interfaces, and every body in it stays the module's.
pub struct BodiesMut<'a, 'str> {
    bodies: &'a mut Registry<FunctionId, FunctionBody<'str>>,
    shared: &'a Shared<'str>,
    interfaces: &'a Registry<FunctionId, FunctionInterface<'str>>,
}

impl<'a, 'str> BodiesMut<'a, 'str> {
    pub(crate) fn new(
        bodies: &'a mut Registry<FunctionId, FunctionBody<'str>>,
        shared: &'a Shared<'str>,
        interfaces: &'a Registry<FunctionId, FunctionInterface<'str>>,
    ) -> Self {
        Self {
            bodies,
            shared,
            interfaces,
        }
    }

    /// The number of functions in the module.
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// The body of `f`, for reading.
    pub fn get(&self, f: FunctionId) -> &FunctionBody<'str> {
        &self.bodies[f]
    }

    /// Every body with its id, in registry order.
    pub fn iter(&self) -> registry::Iter<'_, FunctionId, FunctionBody<'str>> {
        self.bodies.iter()
    }

    /// The body of `f`, exclusively.
    pub fn get_mut(&mut self, f: FunctionId) -> BodyMut<'_, 'str> {
        BodyMut::new(&mut self.bodies[f], self.shared, self.interfaces)
    }

    /// The bodies of `ids` at once, in the order given. Panics on a repeated
    /// or unknown id, as [`Registry::select_mut`] does.
    pub fn select_mut(&mut self, ids: &[FunctionId]) -> Vec<BodyMut<'_, 'str>> {
        let shared = self.shared;
        let interfaces = self.interfaces;
        self.bodies
            .select_mut(ids)
            .into_iter()
            .map(|body| BodyMut::new(body, shared, interfaces))
            .collect()
    }
}

impl<'str> Index<FunctionId> for BodiesMut<'_, 'str> {
    type Output = FunctionBody<'str>;

    fn index(&self, f: FunctionId) -> &Self::Output {
        &self.bodies[f]
    }
}

impl<'a, 'str> IntoIterator for &'a BodiesMut<'_, 'str> {
    type Item = Identified<FunctionId, &'a FunctionBody<'str>>;
    type IntoIter = registry::Iter<'a, FunctionId, FunctionBody<'str>>;

    fn into_iter(self) -> Self::IntoIter {
        self.bodies.iter()
    }
}
