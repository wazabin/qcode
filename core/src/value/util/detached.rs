//! The detached-body mutation host ([`DetachedMut`]).
//!
//! A pass that mints a fresh function builds it *detached*: a [`FunctionBody`]
//! with **no** registry identity yet ([`FunctionBody::detached`]). `DetachedMut`
//! is that body's exclusive mutation surface while it is being built — a thin
//! wrapper exposing **only body-local verbs** (`LocalValueId`/`LocalBlockId`/
//! `LocalInsnId` in and out). Owner values stay composite ([`ValueId`]); minted
//! values are local, so mixing the two is a type error — there is no qualifier in
//! scope (the detached body's [`id`](FunctionBody::id) panics before install) to
//! launder an owner id through.
//!
//! Every method is a one-line delegation to a `FunctionBody` inherent *local*
//! verb; none consults [`FunctionBody::id`].

use std::borrow::Cow;

use crate::{
    builder::Builder,
    context::Shared,
    error::Result,
    types::TypeId,
    value::{
        FunctionBody, FunctionId, LocalBlockId, LocalInsnId, LocalParamId, LocalValueId,
        block::{BasicBlock, EdgeId},
        block_param::BlockParam,
        function::FunctionInterface,
        insn::Mnemonic,
    },
};

/// The exclusive mutation host for a **detached** (minted, not-yet-installed)
/// function body. See the module docs: all verbs are body-local.
pub struct DetachedMut<'a, 'str> {
    /// The detached body being built (no registry identity until install).
    pub body: &'a mut FunctionBody<'str>,
    /// The module's shared IR state, read-only (types/literals minted through the
    /// interners' `&self` paths).
    pub shared: &'a Shared<'str>,
    /// Every function's published interface (never checked out).
    pub interfaces: &'a jstd::registry::Registry<FunctionId, FunctionInterface<'str>>,
}

impl<'a, 'str> DetachedMut<'a, 'str> {
    /// Wrap `body` over the module's shared state and interface registry.
    pub fn new(
        body: &'a mut FunctionBody<'str>,
        shared: &'a Shared<'str>,
        interfaces: &'a jstd::registry::Registry<FunctionId, FunctionInterface<'str>>,
    ) -> Self {
        Self {
            body,
            shared,
            interfaces,
        }
    }

    /// The module's shared IR state ([`Shared`]) (read).
    pub fn shr(&self) -> &Shared<'str> {
        self.shared
    }

    /// Mint a fresh empty block into the detached body.
    pub fn make_block(&mut self) -> LocalBlockId {
        self.body.make_block_local()
    }

    /// Set the detached body's entry block.
    pub fn set_root(&mut self, root: LocalBlockId) {
        self.body.set_root_id(Some(root));
    }

    /// A read of block `block`'s payload (params/instructions/edges).
    pub fn block(&self, block: LocalBlockId) -> &BasicBlock<'str> {
        self.body.block_local(block)
    }

    /// Add a directed CFG edge `from -> to`.
    pub fn add_cfg_edge(&mut self, from: LocalBlockId, to: LocalBlockId) -> EdgeId {
        self.body.add_cfg_edge_local(from, to)
    }

    /// Mint an instruction with `mnemonic` and an explicit result `type_id`.
    pub fn push_mnemonic_with_type(&mut self, mnemonic: Mnemonic, type_id: TypeId) -> LocalInsnId {
        self.body.push_mnemonic_with_type_local(mnemonic, type_id)
    }

    /// Append an already-minted instruction to the end of `block`.
    pub fn append_insn(&mut self, block: LocalBlockId, insn: LocalInsnId) {
        self.body.append_insn_local(block, insn);
    }

    /// Declare a block parameter, wiring it into `block`'s parameter list.
    pub fn push_block_param(
        &mut self,
        block: LocalBlockId,
        param: BlockParam<'str>,
    ) -> LocalParamId {
        self.body.push_block_param_local(block, param)
    }

    /// Replace instruction `insn`'s mnemonic in place, keeping reverse-uses synced.
    pub fn replace_instruction_mnemonic(&mut self, insn: LocalInsnId, mnemonic: Mnemonic) {
        self.body.replace_instruction_mnemonic_local(insn, mnemonic);
    }

    /// Instruction `insn`'s mnemonic (read).
    pub fn mnemonic(&self, insn: LocalInsnId) -> &Mnemonic {
        self.body.mnemonic_local(insn)
    }

    /// The result type of a body-local operand.
    pub fn type_of(&self, id: LocalValueId) -> TypeId {
        self.body.local_type_of(self.shared, id)
    }

    /// Set and register `block`'s name in the body's local name table.
    pub fn rename_block(&mut self, block: LocalBlockId, name: Cow<'str, str>) -> Result<()> {
        self.body.rename_block_local(block, name)
    }

    /// A builder positioned at `block`, driving the id-less local push surface.
    pub fn builder(&mut self, block: LocalBlockId) -> Builder<'str, '_> {
        Builder::new_local(self.body, self.shared, self.interfaces, block)
    }
}
