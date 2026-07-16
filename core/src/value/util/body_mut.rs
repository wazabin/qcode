//! The exclusive mutation backing for a function pass ([`BodyMut`]).
//!
//! [`BodyView`] gives the read layer a `Copy` static provider that
//! routes arena reads to the pass's own function. [`BodyMut`] is its mutable
//! sibling: a single function's arenas borrowed `&mut` in place from
//! `Context.bodies[id]` for exclusive mutation by one worker (the driver's
//! `Context::split` hands out disjoint body borrows).
//!
//! All *shared* data (types, varnodes, spaces, registers, name map) stays behind
//! the `&Shared` view, reachable read-only. The inherent verb + read methods below
//! route each write to the borrowed function's arena; a function pass must not mutate another
//! function (asserted). The module-scope twin of every verb is an inherent method
//! on [`Context`](crate::context::Context); the shorter-lived reborrow needed to
//! hand the host to a value that owns it by value (a [`Builder`](crate::builder::Builder)
//! or a mutation `BaseRef`) is [`BodyMut::reborrow`].

use crate::{
    context::Context,
    error::{Error, ErrorTy, Result},
    value::{
        BlockParamRef, BlockRef, BodyView, FunctionBody, FunctionId, FunctionRef, InstructionRef,
        QCodeView, ValueId,
        block::{BasicBlock, BlockId, EdgeData, EdgeId},
        block_param::{BlockParam, BlockParamId},
        insn::{Instruction, InstructionId, Mnemonic},
    },
};

/// A single function borrowed `&mut` in place from `Context.bodies[id]` for
/// exclusive mutation (its interface stays in `Context.interfaces[id]`,
/// reachable read-only through `interfaces`).
///
/// Out of scope (and asserted against on construction): a function with
/// *reattributed* blocks — a roster block stored in, or parented to, a different
/// function. Those functions go through the sequential (module) path.
pub struct BodyMut<'a, 'str> {
    pub fun: &'a mut FunctionBody<'str>,
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
    pub fn new(
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

    /// Wrap `fun` (checked out under `id`) over a whole module `&Context` — the
    /// module/test-scope convenience constructor (narrows to the shared state +
    /// interface registry).
    pub fn from_ctx(fun: &'a mut FunctionBody<'str>, ctx: &'a Context<'str>) -> Self {
        Self::new(fun, &ctx.shared, &ctx.interfaces)
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

/// The verb + read surface of a checked-out function pass, delegating to the
/// owned `FunctionBody`'s inherent verbs and `self.shared`. The module-scope twin of
/// each verb is an inherent method on [`Context`](crate::context::Context); the
/// primitives below (`function{,_mut}`/`shared`/`view`, and the no-op
/// call-site cache) are the checked-out specializations.
impl<'a, 'str> BodyMut<'a, 'str> {
    // ---- primitives ---------------------------------------------------------

    /// The owned function's storage (write). Panics if `f` is not this function.
    pub fn function_mut(&mut self, f: FunctionId) -> &mut FunctionBody<'str> {
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

    /// A read [`BlockRef`](crate::value::BlockRef) over `id`, body-routed.
    pub fn block_ref(&self, id: BlockId) -> BlockRef<'str, '_, BodyView<'_, 'str>> {
        self.view().block_ref(id)
    }
    /// A read [`InstructionRef`](crate::value::InstructionRef) over `id`.
    pub fn insn_ref(&self, id: InstructionId) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        self.view().insn_ref(id)
    }
    /// A read [`BlockParamRef`](crate::value::BlockParamRef) over `id`.
    pub fn param_ref(&self, id: BlockParamId) -> BlockParamRef<'str, '_, BodyView<'_, 'str>> {
        self.view().param_ref(id)
    }
    /// A read [`FunctionRef`](crate::value::FunctionRef) over `id`.
    pub fn function_ref(&self, id: FunctionId) -> FunctionRef<'str, '_, BodyView<'_, 'str>> {
        self.view().function_ref(id)
    }

    // ---- derived arena accessors --------------------------------------------

    pub fn instruction_mut(&mut self, id: InstructionId) -> &mut Instruction<'str> {
        &mut self.function_mut(id.func).insns[id.local]
    }
    pub fn block_mut(&mut self, id: BlockId) -> &mut BasicBlock<'str> {
        &mut self.function_mut(id.func).blocks[id.local]
    }
    pub fn block_param_mut(&mut self, id: BlockParamId) -> &mut BlockParam<'str> {
        &mut self.function_mut(id.func).params[id.local]
    }

    // ---- births -------------------------------------------------------------

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
