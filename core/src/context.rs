//! The central arena for all IR state: [`Context`].

use std::{borrow::Cow, collections::HashMap, fmt::Display};

use crate::{
    error::{Error, ErrorTy, Result},
    space::{Space, SpaceId},
    value::{
        BasicBlock, Function, FunctionId, Instruction, ValueId, ValueRef,
        block::{BlockId, BlockMutRef, BlockRef, EdgeData, EdgeId, EdgeMutRef, EdgeRef},
        insn::{InstructionId, InstructionRef, PCodeOpId},
        literal::{LiteralId, LiteralRef},
        registry::ValueRegistry,
        varnode::{Varnode, VarnodeId, VarnodeRef, register::RegisterId},
    },
};
use jstd::{graph::Graph, registry::Registry};

/// The central arena that owns all IR state.
///
/// `Context` is the single source of truth for every value (instructions,
/// varnodes, literals, blocks, functions), every memory space, and the
/// bidirectional maps that let you look up values by name or by machine
/// address.
///
/// # Usage
///
/// Create a context with [`Context::new`] and pass `&mut` references to a
/// [`Builder`](crate::builder::Builder) when constructing IR, or to analysis
/// passes when transforming it.
///
/// ```rust,ignore
/// use qcode_core::context::Context;
///
/// let mut ctx = Context::new();
/// // ctx.default_space is the RAM space created by new()
/// ```
///
/// # Lifetime parameter `'str`
///
/// The `'str` lifetime is the lifetime of interned string data used for names
/// and space identifiers. When names are owned (e.g. generated names), they
/// are stored as `Cow::Owned`; when they are borrowed from source data they are
/// `Cow::Borrowed` and must outlive the context.
#[derive(Default)]
pub struct Context<'str> {
    pub default_space: SpaceId,

    /// A mapping of space ids to their corresponding [`Space`]s.
    pub spaces: Registry<SpaceId, Space<'str>>,

    /// A mapping of pcode ops to their names
    pub pcode_ops: Registry<PCodeOpId, &'str str>,

    /// A mapping of names to spaces
    pub named_spaces: HashMap<&'str str, SpaceId>,

    /// Reverse mapping of name hints to value IDs, used to ensure that name hints are unique
    name_map: HashMap<Cow<'str, str>, ValueId>,

    /// Reverse mapping from addresses to value IDs
    address_map: HashMap<u64, ValueId>,

    /// A mapping of register IDs to their corresponding value IDs
    pub registers: HashMap<RegisterId, VarnodeId>,

    /// The values available in the context, indexed by their ID
    pub values: ValueRegistry<'str>,
}

impl<'str> Context<'str> {
    /// Creates a new, empty context with a single default RAM space.
    ///
    /// The default space has a word size of 1 byte and an address size of 8
    /// bytes (suitable for 64-bit architectures). Its [`SpaceId`] is stored in
    /// [`Context::default_space`].
    pub fn new() -> Self {
        let mut ctx = Self::default();
        // SPACE_CONST = SpaceId(0): virtual space for constant/immediate values
        ctx.spaces.push(Space::new(Some("const"), 1, 8));
        // default RAM space (SpaceId(1) = SPACE_UNIQUE; temp spaces start at SpaceId(2))
        let default_space = Space::new(None, 1, 8);
        ctx.default_space = ctx.spaces.push(default_space);
        ctx
    }

    /// Returns the [`SpaceId`] for the named space, or `None` if it has not
    /// been registered.
    pub fn try_get_space(&self, name: &str) -> Option<SpaceId> {
        self.named_spaces.get(name).copied()
    }

    /// Creates a new temporary address space and returns its ID.
    pub fn make_temp_space(&mut self) -> SpaceId {
        let default_space = &self.spaces[self.default_space];

        self.spaces.push(Space::new(
            None,
            default_space.word_size,
            default_space.addr_size,
        ))
    }

    /// Returns the [`BlockId`] for a block at `addr`, creating one if needed.
    ///
    /// The newly created block is named after the address in hex and registered
    /// in the address map.
    pub fn get_or_make_block(&mut self, addr: u64) -> BlockId {
        match BasicBlock::from_addr(self, addr) {
            Some(block) => block.id,
            None => BasicBlock::make(self).with_address(addr).id,
        }
    }

    /// Returns a list of all blocks in the context
    pub fn block_ids(&self) -> Vec<BlockId> {
        self.values.basic_blocks.iter().map(|b| b.id).collect()
    }

    /// Returns a list of all instructions in the context
    pub fn instruction_ids(&self) -> Vec<InstructionId> {
        self.values.instructions.iter().map(|i| i.id).collect()
    }

    /// Returns a list of all functions in the context
    pub fn function_ids(&self) -> Vec<FunctionId> {
        self.values.functions.iter().map(|f| f.id).collect()
    }

    /// Adds a directed edge in the CFG from `from` to `to`.
    pub fn add_cfg_edge(&mut self, from: BlockId, to: BlockId) {
        let edge_id = self.values.edges.push(EdgeData { from, to });
        BasicBlock::from_id_mut(self, from).add_edge(edge_id);
        BasicBlock::from_id_mut(self, to).add_edge(edge_id);
    }

    /// Returns the raw `u64` backing value of the literal `id`.
    pub fn get_literal_value(&self, id: LiteralId) -> u64 {
        self.values.literals[id].value
    }

    /// Returns an immutable reference to the instruction identified by `id`.
    pub fn get_insn(&self, id: InstructionId) -> InstructionRef<'str, '_> {
        InstructionRef::new(self, id)
    }

    /// Returns an immutable reference to the varnode mapped to the named
    /// register `id`.
    pub fn get_register(&self, id: RegisterId) -> VarnodeRef<'str, '_> {
        Varnode::from_id(self, self.registers[&id])
    }

    /// Returns a type-erased view of the value identified by `id`.
    ///
    /// Panics if `id` does not correspond to a value stored in this context.
    pub fn get_value(&self, id: ValueId) -> ValueRef<'str, '_> {
        ValueRef::new(id, self)
    }

    /// Returns the [`Space`] identified by `id`.
    pub fn get_space(&self, id: SpaceId) -> &Space<'str> {
        &self.spaces[id]
    }

    /// Creates a [`Value`] representing a constant value.
    pub fn get_const(&mut self, value: u64, size: usize) -> LiteralRef<'str, '_> {
        let id = self.values.get_or_make_literal(value, size);
        LiteralRef::new(self, id)
    }

    /// Return all instructions that use `value` as an operand.
    pub fn users(&self, value: impl Into<ValueId>) -> &[InstructionId] {
        self.values.users_of(value.into())
    }

    /// Replace every use of `old` with `new` across all instructions that
    /// reference `old`, and update the users reverse map accordingly.
    pub fn replace_all_uses_with(&mut self, old: impl Into<ValueId>, new: impl Into<ValueId>) {
        let old = old.into();
        let new = new.into();
        if old == new {
            return;
        }
        let users: Vec<InstructionId> = self.values.users_of(old).to_vec();
        for user_id in users {
            Instruction::from_id_mut(self, user_id)
                .mnemonic_mut()
                .replace_value(old, new);
            self.values.users.entry(new).or_default().push(user_id);
        }
        self.values.users.remove(&old);
    }

    /// Associates `addr` with `id` in the address map.
    ///
    /// Returns `Err(Error::DuplicateAddress(addr))` if the address is already
    /// mapped. Callers that have already checked (e.g. via
    /// [`get_at_addr`](Self::get_at_addr)) may safely `.expect(...)` the result.
    pub(crate) fn set_address(&mut self, addr: u64, id: ValueId) -> crate::error::Result<'str, ()> {
        if let Some(existing) = self.address_map.insert(addr, id) {
            // The only case where duplicates are allowed are for a function and its root block sharing an address
            // In that case, the function address should be kept.

            match (existing, id) {
                // A function and its entry block share an address by design.
                // The function always takes priority in the address map,
                // regardless of whether the root link has been established yet
                // (make_at_addr registers the address before calling make_root).
                (ValueId::Function(func_id), ValueId::BasicBlock(block_id))
                | (ValueId::BasicBlock(block_id), ValueId::Function(func_id)) => {
                    let mut function = Function::from_id_mut(self, func_id);
                    function.ensure_root(block_id)?;
                    self.address_map.insert(addr, ValueId::Function(func_id));
                    return Ok(());
                }

                _ => {}
            }

            Err(Error::spanless(ErrorTy::DuplicateAddress(addr, existing)))
        } else {
            Ok(())
        }
    }

    /// Changes the name of a value
    pub fn update_name(
        &mut self,
        name: Cow<'str, str>,
        id: ValueId,
        old_name: Option<&str>,
    ) -> Result<'str, ()> {
        if let Some(old_name) = old_name {
            self.name_map.remove(old_name);
        }
        match self.name_map.insert(name.clone(), id) {
            Some(_) => Err(Error::spanless(ErrorTy::DuplicateName(name.to_string()))),
            None => Ok(()),
        }
    }

    /// Attempts to get a value ID by its name.
    /// Returns `None` if no value with the given name exists.
    pub fn get_named(&self, name: &str) -> Option<ValueId> {
        self.name_map.get(name).copied()
    }

    /// Gets a value ID at a given address
    pub fn get_at_addr(&self, addr: &u64) -> Option<ValueId> {
        self.address_map.get(addr).copied()
    }

    /// Gets a unique name for a value, generating one
    /// if necessary by appending a numeric suffix to the provided name
    /// until an unused name is found.
    pub fn get_unique_name(&self, name: Cow<'str, str>) -> Cow<'str, str> {
        // If the name is already taken, we don't want to overwrite it, since that would make debugging harder
        // instead we add a numeric suffix to the name until we find an unused name
        let mut suffix = 0;

        let mut unique_name = Cow::Owned(name.to_string());

        while self.name_map.contains_key(&unique_name) {
            suffix += 1;
            unique_name = Cow::Owned(format!("{}_{suffix}", name));
        }
        unique_name
    }
}

impl Display for Context<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.values
            .basic_blocks
            .iter()
            .try_for_each(|block| BasicBlock::from_id(self, block.id).fmt(f))
    }
}

impl<'str> Graph for Context<'str> {
    type NodeId = BlockId;

    type EdgeId = EdgeId;

    type Node<'a>
        = BlockRef<'str, 'a>
    where
        Self: 'a;

    type Edge<'a>
        = EdgeRef<'str, 'a>
    where
        Self: 'a;

    type NodeMut<'a>
        = BlockMutRef<'str, 'a>
    where
        Self: 'a;

    type EdgeMut<'a>
        = EdgeMutRef<'str, 'a>
    where
        Self: 'a;

    fn get_node(&self, id: Self::NodeId) -> Option<Self::Node<'_>> {
        Some(BasicBlock::from_id(self, id))
    }

    fn get_node_mut(&mut self, id: Self::NodeId) -> Option<Self::NodeMut<'_>> {
        Some(BasicBlock::from_id_mut(self, id))
    }

    fn get_edge(&self, id: Self::EdgeId) -> Option<Self::Edge<'_>> {
        Some(EdgeRef::new(self, id))
    }

    fn get_edge_mut(&mut self, id: Self::EdgeId) -> Option<Self::EdgeMut<'_>> {
        Some(EdgeMutRef::new(self, id))
    }

    fn nodes(&self) -> impl Iterator<Item = Self::Node<'_>> + '_ {
        self.values
            .basic_blocks
            .iter()
            .map(|block| BasicBlock::from_id(self, block.id))
    }

    fn edges(&self) -> impl Iterator<Item = Self::Edge<'_>> + '_ {
        self.values
            .edges
            .iter()
            .map(|edge| EdgeRef::new(self, edge.id))
    }
}
