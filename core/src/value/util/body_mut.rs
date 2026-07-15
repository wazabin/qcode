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
    /// Wrap `fun` over the module's shared state and
    /// interface registry. Asserts (all builds — this is the once-per-checkout
    /// enforcement point; `BodyView::new`'s copy is debug-only) the function
    /// owns only self-stored,
    /// self-parented blocks (no reattribution).
    pub fn new(
        fun: &'a mut FunctionBody<'str>,
        shared: &'a crate::context::Shared<'str>,
        interfaces: &'a jstd::registry::Registry<
            FunctionId,
            crate::value::function::FunctionInterface<'str>,
        >,
    ) -> Self {
        let id = fun.id();
        assert!(
            fun.roster.iter().all(|&local| {
                // Stored in this function's own arena and parented to it.
                let blk = &fun.blocks[local];
                blk.parent == Some(id)
            }),
            "BodyMut requires a function with no reattributed blocks"
        );
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

    /// Physically removes a block parameter and its local bookkeeping.
    /// Positional block and edge-argument rewrites belong to the caller and may
    /// complete later in the same transformation.
    pub fn remove_block_param(&mut self, id: BlockParamId) {
        self.function_mut(id.func).remove_block_param(id);
    }

    // ---- births -------------------------------------------------------------

    pub fn push_edge(&mut self, func: FunctionId, edge: EdgeData) -> EdgeId {
        self.function_mut(func).edges.push(edge)
    }

    pub fn push_insn(&mut self, func: FunctionId, insn: Instruction<'str>) -> InstructionId {
        self.function_mut(func).push_insn(insn)
    }

    pub fn push_block(&mut self, func: FunctionId, block: BasicBlock<'str>) -> BlockId {
        self.function_mut(func).push_block(block)
    }

    pub fn make_block(&mut self, func: FunctionId) -> BlockId {
        self.function_mut(func).make_block()
    }

    pub fn push_block_param(&mut self, func: FunctionId, param: BlockParam<'str>) -> BlockParamId {
        self.function_mut(func).push_block_param(param)
    }

    pub fn push_mnemonic(
        &mut self,
        func: FunctionId,
        mnemonic: Mnemonic,
        size: usize,
    ) -> InstructionId {
        let shared = self.shared;
        self.function_mut(func)
            .push_mnemonic(shared, mnemonic, size)
    }

    pub fn push_mnemonic_with_type(
        &mut self,
        func: FunctionId,
        mnemonic: Mnemonic,
        type_id: crate::types::TypeId,
    ) -> InstructionId {
        self.function_mut(func)
            .push_mnemonic_with_type(mnemonic, type_id)
    }

    pub fn insert_insn_before(
        &mut self,
        block: BlockId,
        before: InstructionId,
        insn: InstructionId,
    ) {
        self.function_mut(block.func)
            .insert_insn_before(block, before, insn)
    }

    // ---- CFG / use-map verbs ------------------------------------------------

    pub fn add_cfg_edge(&mut self, from: BlockId, to: BlockId) -> EdgeId {
        self.function_mut(from.func).add_cfg_edge(from, to)
    }

    pub fn remove_cfg_edge(&mut self, func: FunctionId, edge_id: EdgeId) {
        self.function_mut(func).remove_cfg_edge(edge_id)
    }

    pub fn replace_all_uses_with(&mut self, old: ValueId, new: ValueId) {
        if old == new {
            return;
        }
        let Some(func) = old.owning_function() else {
            return;
        };
        self.function_mut(func).replace_all_uses_with(old, new)
    }

    pub fn remove_instruction(&mut self, id: InstructionId) {
        self.function_mut(id.func).remove_instruction(id)
    }

    pub fn rehome_outgoing_edges(&mut self, keep: BlockId, remove: BlockId) {
        self.function_mut(keep.func)
            .rehome_outgoing_edges(keep, remove)
    }

    pub fn replace_instruction_mnemonic(&mut self, id: InstructionId, mnemonic: Mnemonic) {
        self.function_mut(id.func)
            .replace_instruction_mnemonic(id, mnemonic)
    }

    pub fn unroster_block(&mut self, block: BlockId) {
        self.function_mut(block.func).unroster_block(block)
    }

    pub fn delete_block(&mut self, block: BlockId) {
        self.function_mut(block.func).delete_block(block)
    }

    pub fn absorb_block(&mut self, keep: BlockId, other: BlockId, edge_ab: EdgeId) {
        self.function_mut(keep.func)
            .absorb_block(keep, other, edge_ab)
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
