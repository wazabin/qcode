//! The central arena for all IR state: [`Context`].

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    fmt::Display,
};

use crate::{
    assumption::{Assumption, AssumptionId, AssumptionKind, AssumptionStatus},
    error::{Error, ErrorTy, Result},
    space::{Space, SpaceId},
    types::TypeManager,
    value::{
        BasicBlock, Function, FunctionId, FunctionRef, Instruction, ValueId,
        block::{BlockId, BlockMutRef, BlockRef, EdgeData, EdgeId, EdgeMutRef, EdgeRef},
        insn::{InstructionId, InstructionRef, Mnemonic, PCodeOpId},
        literal::{LiteralId, LiteralRef},
        registry::ValueRegistry,
        varnode::{Varnode, VarnodeId, VarnodeRef, register::RegisterId},
    },
};
use jstd::{
    graph::Graph,
    registry::{self, Registry},
};

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
#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct Context<'str> {
    pub default_space: SpaceId,

    /// A mapping of space ids to their corresponding [`Space`]s.
    pub(crate) spaces: Registry<SpaceId, Space>,

    /// A mapping of pcode ops to their names
    pub pcode_ops: Registry<PCodeOpId, Box<str>>,

    /// A mapping of names to spaces
    pub named_spaces: HashMap<Box<str>, SpaceId>,

    /// Reverse mapping of name hints to value IDs, used to ensure that name hints are unique
    name_map: HashMap<Cow<'str, str>, ValueId>,

    /// Reverse mapping from addresses to value IDs
    address_map: HashMap<u64, ValueId>,

    /// A mapping of register IDs to their corresponding value IDs
    pub registers: HashMap<RegisterId, VarnodeId>,

    /// The values available in the context, indexed by their ID
    pub values: ValueRegistry<'str>,

    /// Type registry: owns all [`Type`] objects and hands out [`TypeId`]s.
    pub types: TypeManager,
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
        // default RAM space (SpaceId(1)); temp spaces start at SpaceId(2)
        let default_space = Space::new(Some("ram"), 1, 8);
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
        let mut space = Space::new(None, default_space.word_size, default_space.addr_size);
        space.ty = crate::space::SpaceType::Temporary;
        self.spaces.push(space)
    }

    /// Adds a space to the context, registering its name, and returns its ID.
    pub fn add_space(&mut self, space: Space) -> SpaceId {
        let name_key: Option<Box<str>> = space.name.clone();
        let id = self.spaces.push(space);
        if let Some(name) = name_key {
            self.named_spaces.insert(name, id);
        }
        id
    }

    /// Returns the number of spaces registered in this context.
    pub fn space_count(&self) -> usize {
        self.spaces.len()
    }

    /// Replaces the spaces registry wholesale. Intended for initialization from a pre-built spec.
    pub fn load_spaces(&mut self, spaces: registry::Registry<SpaceId, Space>) {
        self.spaces = spaces;
    }

    /// Creates a new named temporary address space and returns its ID.
    pub fn make_named_temp_space(&mut self, name: impl Into<Box<str>>) -> SpaceId {
        let default_space = &self.spaces[self.default_space];
        let (word_size, addr_size) = (default_space.word_size, default_space.addr_size);
        self.spaces.push(Space {
            name: Some(name.into()),
            word_size,
            addr_size,
            ty: crate::space::SpaceType::Temporary,
        })
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

    /// Iterates over all the instructions in the context
    pub fn instructions(&self) -> impl Iterator<Item = InstructionRef<'str, '_>> + '_ {
        self.values
            .instructions
            .iter()
            .map(|i| Instruction::from_id(self, i.id))
    }

    /// Iterates over all the blocks in the context
    pub fn blocks(&self) -> impl Iterator<Item = BlockRef<'str, '_>> + '_ {
        self.values
            .basic_blocks
            .iter()
            .map(|b| BlockRef::from_id(self, b.id))
    }

    /// Iterates over all the functions in the context
    pub fn functions(&self) -> FunctionIter<'str, '_> {
        FunctionIter {
            ctx: self,
            inner: self.values.functions.iter(),
        }
    }

    /// Iterates over all the functions in the context
    /// alias for `functions()`
    pub fn iter(&self) -> FunctionIter<'str, '_> {
        self.functions()
    }

    pub fn varnodes(&self) -> impl Iterator<Item = VarnodeRef<'str, '_>> + '_ {
        self.values
            .varnodes
            .iter()
            .map(|v| Varnode::from_id(self, v.id))
    }

    /// Adds a directed edge in the CFG from `from` to `to`, returning its id.
    pub fn add_cfg_edge(&mut self, from: BlockId, to: BlockId) -> EdgeId {
        let edge_id = self.values.edges.push(EdgeData { from, to });
        BasicBlock::from_id_mut(self, from).add_edge(edge_id);
        BasicBlock::from_id_mut(self, to).add_edge(edge_id);
        edge_id
    }

    /// Removes a CFG edge, unlinking it from both incident blocks' edge sets.
    ///
    /// The backing [`EdgeData`] slot in the append-only registry is left in
    /// place (dangling), consistent with how removed instructions are handled;
    /// per-block traversal reads the block edge sets, which this updates.
    pub fn remove_cfg_edge(&mut self, edge_id: EdgeId) {
        let EdgeData { from, to } = self.values.edges[edge_id];
        BasicBlock::from_id_mut(self, from).remove_edge(edge_id);
        BasicBlock::from_id_mut(self, to).remove_edge(edge_id);
    }

    /// Records a heuristic [`Assumption`] and indexes it by the callee it
    /// predicts. Returns the new [`AssumptionId`].
    pub fn add_assumption(&mut self, kind: AssumptionKind) -> AssumptionId {
        let assumption = Assumption::new(kind);
        let callee = assumption.callee();
        let id = self.values.assumptions.push(assumption);
        if let Some(callee) = callee {
            self.values
                .assumptions_by_callee
                .entry(callee)
                .or_default()
                .push(id);
        }
        id
    }

    /// Returns the [`Assumption`] with the given id.
    pub fn assumption(&self, id: AssumptionId) -> &Assumption {
        &self.values.assumptions[id]
    }

    /// Returns the assumptions predicting `callee`, in insertion order.
    pub fn assumptions_for_callee(&self, callee: FunctionId) -> &[AssumptionId] {
        self.values
            .assumptions_by_callee
            .get(&callee)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Sets the verification status of an assumption.
    pub fn set_assumption_status(&mut self, id: AssumptionId, status: AssumptionStatus) {
        self.values.assumptions[id].status = status;
    }

    /// Iterates over all recorded assumptions.
    pub fn assumptions(&self) -> impl Iterator<Item = (AssumptionId, &Assumption)> {
        self.values
            .assumptions
            .iter()
            .map(|item| (item.id, item.inner))
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

    /// Creates a [`Value`] representing an integer constant of the given byte width.
    pub fn get_const(&mut self, value: u64, size: usize) -> LiteralRef<'str, '_> {
        let type_id = self.types.get_or_make_int(size);
        let id = self.values.get_or_make_typed_literal(value, type_id, size);
        LiteralRef::new(self, id)
    }

    /// Creates a typed constant literal.
    ///
    /// Unlike [`get_const`](Self::get_const) this accepts an arbitrary [`TypeId`],
    /// allowing StackAddress constants (e.g. the stack base) to preserve their
    /// type through constant folding.
    pub fn get_typed_const(
        &mut self,
        value: u64,
        type_id: crate::types::TypeId,
    ) -> LiteralRef<'str, '_> {
        let size = self.types.size_of(type_id);
        let id = self.values.get_or_make_typed_literal(value, type_id, size);
        LiteralRef::new(self, id)
    }

    /// Returns the [`TypeId`] of any [`ValueId`] in this context.
    ///
    /// Varnodes are typed as `Int(varnode.size())`. Blocks, functions, and other
    /// non-data values return `Int(0)`.
    pub fn type_of(&mut self, id: ValueId) -> crate::types::TypeId {
        match id {
            ValueId::Literal(lid) => self.values.literals[lid].type_id,
            ValueId::Instruction(iid) => self.values.instructions[iid].type_id,
            ValueId::BlockParam(pid) => self.values.block_params[pid].type_id,
            ValueId::Varnode(vid) => {
                let size = self.values.varnodes[vid].size_bytes();
                self.types.get_or_make_int(size)
            }
            // Exhaustive on purpose: a new ValueId variant must decide its type
            // here rather than silently inheriting the zero-width sentinel.
            ValueId::BasicBlock(_) | ValueId::Function(_) => self.types.get_or_make_int(0),
        }
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

    /// Replaces one instruction's mnemonic and keeps the reverse use map in sync.
    ///
    /// This is for transforms that change an instruction in place without
    /// changing its identity, parent block, address, or result type.
    pub fn replace_instruction_mnemonic(&mut self, id: InstructionId, mnemonic: Mnemonic) {
        let old_args = self.values.instructions[id].mnemonic().args();
        for arg in old_args {
            let mut remove_arg = false;
            if let Some(users) = self.values.users.get_mut(&arg) {
                users.retain(|&user| user != id);
                remove_arg = users.is_empty();
            }
            if remove_arg {
                self.values.users.remove(&arg);
            }
        }

        *Instruction::from_id_mut(self, id).mnemonic_mut() = mnemonic;

        for arg in self.values.instructions[id].mnemonic().args() {
            self.values.users.entry(arg).or_default().push(id);
        }
    }

    /// Removes an instruction from its parent block and unlinks all associated state:
    /// removes the instruction from the block's instruction list, clears `parent`,
    /// removes its name from the name map, clears the name field, and prunes it from
    /// the `users` reverse map for all its operands.
    ///
    /// If the instruction has no parent block the block-list and parent steps are
    /// skipped, but name and users cleanup still runs.
    pub fn remove_instruction(&mut self, id: InstructionId) {
        let parent = self.values.instructions[id].parent;
        let name = self.values.instructions[id].name.clone();

        if let Some(block_id) = parent {
            self.values.basic_blocks[block_id]
                .instructions
                .retain(|&i| i != id);

            // If this instruction was a terminator instruction in a basic block,
            // remove cfg edges
            if Instruction::from_id(self, id).mnemonic().is_terminator() {
                let mut edges_to_remove = HashSet::new();
                for edge in BasicBlock::from_id(self, block_id).successors() {
                    edges_to_remove.insert(edge.0);
                }

                for edge_id in edges_to_remove {
                    self.remove_cfg_edge(edge_id);
                }
            }
        }

        self.values.instructions[id].parent = None;

        if let Some(ref n) = name {
            self.name_map.remove(n.as_ref());
        }
        self.values.instructions[id].name = None;

        self.values.remove_instructions(&HashSet::from([id]));
    }

    /// Associates `addr` with `id` in the address map.
    ///
    /// Returns `Err(Error::DuplicateAddress(addr))` if the address is already
    /// mapped. Callers that have already checked (e.g. via
    /// [`get_at_addr`](Self::get_at_addr)) may safely `.expect(...)` the result.
    pub(crate) fn set_address(&mut self, addr: u64, id: ValueId) -> crate::error::Result<()> {
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
    ) -> Result<()> {
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
        self.functions().try_for_each(|fun| fun.fmt(f))?;

        self.blocks()
            .filter(|block| block.parent().is_none())
            .try_for_each(|block| block.fmt(f))
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

pub struct FunctionIter<'str, 'ctx> {
    ctx: &'ctx Context<'str>,
    inner: registry::Iter<'ctx, FunctionId, Function<'str>>,
}

impl<'str, 'ctx> Iterator for FunctionIter<'str, 'ctx> {
    type Item = FunctionRef<'str, 'ctx>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|f| FunctionRef::from_id(self.ctx, f.id))
    }
}

impl<'str, 'ctx> IntoIterator for &'ctx Context<'str> {
    type Item = FunctionRef<'str, 'ctx>;
    type IntoIter = FunctionIter<'str, 'ctx>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{
        BasicBlock, Function, ValueId,
        insn::{Binary, Binop, Call, IntBinop, Load, Mnemonic},
    };
    use qcode_macro::qcode;

    fn make_fn_with_blocks(ctx: &mut Context<'static>, name: &'static str, n: usize) -> FunctionId {
        let block_ids: Vec<BlockId> = (0..n).map(|_| BasicBlock::make(ctx).id).collect();
        let mut f = Function::make(ctx, name.into()).unwrap();
        for id in block_ids {
            f.add_block(id);
        }
        f.id
    }

    #[test]
    fn functions_iter_yields_all_functions() {
        let mut ctx = Context::new();
        make_fn_with_blocks(&mut ctx, "alpha", 1);
        make_fn_with_blocks(&mut ctx, "beta", 1);

        let names: Vec<_> = ctx.functions().map(|f| f.name().to_string()).collect();
        assert!(names.contains(&"alpha".to_string()));
        assert!(names.contains(&"beta".to_string()));
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn into_iterator_for_context_matches_functions() {
        let mut ctx = Context::new();
        make_fn_with_blocks(&mut ctx, "f1", 1);
        make_fn_with_blocks(&mut ctx, "f2", 1);

        let via_method: Vec<_> = ctx.functions().map(|f| f.id()).collect();
        let via_into: Vec<_> = (&ctx).into_iter().map(|f| f.id()).collect();
        assert_eq!(via_method, via_into);
    }

    #[test]
    fn blocks_iter_yields_all_blocks() {
        let mut ctx = Context::new();
        make_fn_with_blocks(&mut ctx, "g", 3);

        let count = ctx.blocks().count();
        assert_eq!(count, 3);
    }

    #[test]
    fn instructions_iter_yields_all_instructions() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 ptr;

            <block>
                store(&ptr, i64 0x1234);
                return [ptr];
            "
        );

        let count = ctx.instructions().count();
        assert!(count >= 1, "expected at least one instruction, got {count}");
    }

    #[test]
    fn remove_instruction_removes_from_block() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 x;
            <block>
                %a = load(i64, &x);
                %b = load(i64, &x);
                return [%a];
            "
        );
        let block_ref = BasicBlock::from_id(&ctx, block);
        let ids: Vec<_> = block_ref.instruction_ids().to_vec();
        let load_a = ids[0];
        let original_len = ids.len();

        ctx.remove_instruction(load_a);

        let remaining: Vec<_> = BasicBlock::from_id(&ctx, block).instruction_ids().to_vec();
        assert_eq!(remaining.len(), original_len - 1);
        assert!(!remaining.contains(&load_a));
    }

    #[test]
    fn remove_instruction_clears_parent() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 x;
            <block>
                %a = load(i64, &x);
                return [%a];
            "
        );
        let load_id = BasicBlock::from_id(&ctx, block).instruction_ids()[0];

        ctx.remove_instruction(load_id);

        assert!(
            ctx.get_insn(load_id).parent().is_none(),
            "parent should be None after removal"
        );
    }

    #[test]
    fn remove_instruction_frees_name() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 x;
            <block>
                %a = load(i64, &x);
                return [%a];
            "
        );
        let load_id = BasicBlock::from_id(&ctx, block).instruction_ids()[0];
        assert!(
            ctx.get_named("a").is_some(),
            "name should be in map before removal"
        );

        ctx.remove_instruction(load_id);

        assert!(
            ctx.get_named("a").is_none(),
            "name should be gone after removal"
        );
        assert!(
            ctx.get_insn(load_id).name().is_none(),
            "instruction name field should be cleared"
        );
    }

    #[test]
    fn remove_instruction_frees_name_for_reuse() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 x;
            <block>
                %a = load(i64, &x);
                return [%a];
            "
        );
        let load_id = BasicBlock::from_id(&ctx, block).instruction_ids()[0];

        ctx.remove_instruction(load_id);

        // Building another instruction named %a should succeed now.
        qcode!(
            ctx,
            "
            varnode i64 y;
            <block2>
                %a = load(i64, &y);
                return [%a];
            "
        );
        assert!(
            ctx.get_named("a").is_some(),
            "name should be reusable after removal"
        );
    }

    #[test]
    fn remove_instruction_updates_users_map() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 x;
            <block>
                %a = load(i64, &x);
                %b = %a + i64 1;
                return [%b];
            "
        );
        let ids: Vec<_> = BasicBlock::from_id(&ctx, block).instruction_ids().to_vec();
        let load_id = ids[0];
        let add_id = ids[1];

        assert!(
            ctx.users(load_id).contains(&add_id),
            "add should be a user of load before removal"
        );

        ctx.remove_instruction(add_id);

        assert!(
            ctx.users(load_id).is_empty(),
            "load should have no users after add is removed"
        );
    }

    #[test]
    fn replace_instruction_mnemonic_rewrites_callind_users() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 ptr;
            <block>
                call [ptr];
            "
        );
        let call_id = BasicBlock::from_id(&ctx, block).instruction_ids()[0];
        let ptr = match ctx.get_insn(call_id).mnemonic() {
            Mnemonic::CallInd(call) => call.ptr,
            other => panic!("expected CallInd, got {other:?}"),
        };
        assert_eq!(ctx.users(ptr), &[call_id]);

        let target = Function::make(&mut ctx, "target".into()).unwrap().id;
        ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target,
                args: vec![],
                clobbers: vec![],
            }),
        );

        assert!(
            ctx.users(ptr).is_empty(),
            "old indirect pointer should no longer list the rewritten call"
        );
        assert!(matches!(
            ctx.get_insn(call_id).mnemonic(),
            Mnemonic::Call(Call {
                target: actual,
                args,
                ..
            }) if *actual == target && args.is_empty()
        ));
    }

    #[test]
    fn replace_instruction_mnemonic_moves_operand_users() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 x;
            varnode i64 y;
            <block>
                %a = load(i64, x);
                return [%a];
            "
        );
        let load_id = BasicBlock::from_id(&ctx, block).instruction_ids()[0];
        let old_ptr = ValueId::Varnode(x);
        let new_ptr = ValueId::Varnode(y);
        assert_eq!(ctx.users(old_ptr), &[load_id]);
        assert!(ctx.users(new_ptr).is_empty());

        ctx.replace_instruction_mnemonic(
            load_id,
            Mnemonic::Load(Load {
                space: ctx.default_space,
                ptr: new_ptr,
                size: 8,
            }),
        );

        assert!(ctx.users(old_ptr).is_empty());
        assert_eq!(ctx.users(new_ptr), &[load_id]);
    }

    #[test]
    fn replace_instruction_mnemonic_tracks_repeated_operands() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 x;
            varnode i64 y;
            <block>
                %a = load(i64, x);
                return [%a];
            "
        );
        let load_id = BasicBlock::from_id(&ctx, block).instruction_ids()[0];
        let old_ptr = ValueId::Varnode(x);
        let new_arg = ValueId::Varnode(y);

        ctx.replace_instruction_mnemonic(
            load_id,
            Mnemonic::Binop(Binary {
                op: Binop::Int(IntBinop::Add),
                lhs: new_arg,
                rhs: new_arg,
            }),
        );

        assert!(ctx.users(old_ptr).is_empty());
        assert_eq!(
            ctx.users(new_arg),
            &[load_id, load_id],
            "a mnemonic using the same operand twice should record both uses"
        );
    }

    #[test]
    fn remove_instruction_unparented_noop() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 x;
            <block>
                %a = load(i64, &x);
                return [%a];
            "
        );
        let load_id = BasicBlock::from_id(&ctx, block).instruction_ids()[0];

        // Manually detach from block without using remove_instruction,
        // simulating an instruction with no parent.
        ctx.values.instructions[load_id].parent = None;

        // Should not panic even though parent is None.
        ctx.remove_instruction(load_id);

        assert!(ctx.get_named("a").is_none());
    }

    #[test]
    fn add_cfg_edge_returns_id_and_remove_unlinks_both_blocks() {
        let mut ctx = Context::new();
        let a = BasicBlock::make(&mut ctx).id;
        let b = BasicBlock::make(&mut ctx).id;

        let edge = ctx.add_cfg_edge(a, b);
        assert_eq!(
            BasicBlock::from_id(&ctx, a)
                .successors()
                .collect::<Vec<_>>(),
            vec![(edge, b)]
        );
        assert_eq!(
            BasicBlock::from_id(&ctx, b)
                .predecessors()
                .collect::<Vec<_>>(),
            vec![(edge, a)]
        );

        ctx.remove_cfg_edge(edge);
        assert!(BasicBlock::from_id(&ctx, a).successors().next().is_none());
        assert!(BasicBlock::from_id(&ctx, b).predecessors().next().is_none());
    }

    #[test]
    fn assumption_ledger_indexes_by_callee_and_tracks_status() {
        use crate::assumption::{AssumptionKind, AssumptionStatus, CallReturnsAssumption};
        use crate::value::insn::InstructionId;

        let mut ctx = Context::new();
        let callee = Function::make(&mut ctx, "callee".into()).unwrap().id;
        let call_block = BasicBlock::make(&mut ctx).id;
        let continuation = BasicBlock::make(&mut ctx).id;
        let edge = ctx.add_cfg_edge(call_block, continuation);

        let id = ctx.add_assumption(AssumptionKind::CallReturns(CallReturnsAssumption {
            callee,
            call_site: InstructionId::from(0usize),
            call_block,
            continuation,
            continuation_edge: edge,
        }));

        assert_eq!(ctx.assumptions_for_callee(callee), &[id]);
        assert_eq!(ctx.assumption(id).status, AssumptionStatus::Unverified);

        // The ledger is part of the arena, so it snapshots with a clone.
        let snapshot = ctx.clone();
        assert_eq!(snapshot.assumptions_for_callee(callee), &[id]);

        ctx.set_assumption_status(id, AssumptionStatus::Violated);
        assert_eq!(ctx.assumption(id).status, AssumptionStatus::Violated);
        // The independent snapshot is unaffected.
        assert_eq!(snapshot.assumption(id).status, AssumptionStatus::Unverified);
    }

    #[test]
    fn context_survives_bincode_round_trip() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 ptr;
            <block>
                %a = load(i64, &ptr);
                %b = %a + i64 0x10;
                store(&ptr, i64 0x1234);
                return [%b];
            "
        );

        // A StackAddress type exercises the custom TypeManager serialization.
        let stack_space = ctx.make_named_temp_space("stack");
        let sa = ctx.types.get_or_make_stack_address(8, Some(stack_space));
        let sa_size = ctx.types.size_of(sa);

        let blocks_before = ctx.block_ids().len();
        let insns_before = ctx.instruction_ids().len();
        let funcs_before = ctx.function_ids().len();

        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&ctx, config).expect("encode");
        let (restored, _): (Context<'static>, usize) =
            bincode::serde::decode_from_slice(&bytes, config).expect("decode");

        assert_eq!(restored.block_ids().len(), blocks_before);
        assert_eq!(restored.instruction_ids().len(), insns_before);
        assert_eq!(restored.function_ids().len(), funcs_before);
        // The StackAddress type round-trips: same id, same size, still a stack address.
        assert_eq!(restored.types.size_of(sa), sa_size);
        assert!(restored.types.is_stack_address(sa));
    }
}
