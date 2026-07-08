use crate::{
    context::Context,
    error::Result,
    value::{
        Function, Instruction, Value, ValueId,
        block_param::{BlockParam, BlockParamId, BlockParamMutRef, BlockParamRef},
        function::{FunctionId, FunctionMutRef, FunctionRef},
        insn::{InstructionId, InstructionRef},
        util::{
            base_ref::{BaseRef, WithCtx, WithCtxMut},
            named::{Named, Renameable, update_context_name},
        },
    },
};
use core::slice;
use jstd::graph::{FxBuildHasher, Graph};
use std::{
    borrow::Cow,
    collections::HashSet,
    fmt::{Display, Formatter},
};

use rustc_hash::FxHashMap as HashMap;

pub(crate) use self::cfg::EdgeData;
pub use self::cfg::{BlockId, EdgeId, EdgeMutRef, EdgeRef};
pub mod cfg;

/// A block of instructions.
/// This is the basic unit of code in our IR.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct BasicBlock<'str> {
    /// An optionnal name for this basic block
    name: Option<Cow<'str, str>>,

    /// Optional human-readable analysis note rendered under the block label.
    comment: Option<String>,

    /// Typed parameters declared at block entry (block-argument style).
    /// These are NOT part of `instructions`; use `params()` to iterate them.
    pub params: Vec<BlockParamId>,

    /// The ids of the instructions in this block
    pub instructions: Vec<InstructionId>,

    /// The set of edges that this block is incident to.
    ///
    /// Uses a fixed-seed hasher (matching `Context`'s `Graph::Hasher`) so that
    /// `predecessors()`/`successors()` iterate deterministically across runs.
    pub edges: HashSet<EdgeId, FxBuildHasher>,

    /// The address of this block, if it corresponds to a machine address.
    pub address: Option<u64>,

    /// Additional addresses that map to this block (accumulated from merged blocks).
    pub extra_addresses: Vec<u64>,

    /// The function this block belongs to, if any. Invariant for a live block:
    /// `parent == Some(id.func)` — the arena that stores the block is its
    /// owning function.
    pub parent: Option<FunctionId>,

    /// Tombstone flag. Per-function block arenas are append-only and never
    /// compacted, so a removed block stays in its arena; this marks it as
    /// logically deleted. Deleted blocks are skipped by
    /// [`FunctionRef::blocks`](crate::value::FunctionRef::blocks) and the
    /// whole-context block iterators.
    #[serde(default)]
    pub deleted: bool,
}

impl<'str> BasicBlock<'str> {
    /// Gets a reference to a block from its ID
    pub fn from_id<'ctx>(ctx: &'ctx Context<'str>, id: BlockId) -> BlockRef<'str, 'ctx> {
        BlockRef::new(ctx, id)
    }

    /// Gets a mutable reference to a block from its ID
    pub fn from_id_mut<'ctx>(ctx: &'ctx mut Context<'str>, id: BlockId) -> BlockMutRef<'str, 'ctx> {
        BlockMutRef::new(ctx, id)
    }

    /// Gets a reference to a block by name
    pub fn from_name<'ctx>(ctx: &'ctx Context<'str>, name: &str) -> Option<BlockRef<'str, 'ctx>> {
        ctx.get_named(name)
            .and_then(ValueId::as_block)
            .map(|id| BlockRef::new(ctx, id))
    }

    /// Gets a reference to a block by address
    /// A block and a function can share an address
    pub fn from_addr<'ctx>(ctx: &'ctx Context<'str>, addr: u64) -> Option<BlockRef<'str, 'ctx>> {
        ctx.get_at_addr(&addr).and_then(|id| match id {
            ValueId::BasicBlock(block_id) => Some(BlockRef::new(ctx, block_id)),
            ValueId::Function(function_id) => Function::from_id(ctx, function_id).root(),
            _ => None,
        })
    }

    /// Gets a mutable reference to a block by address
    /// A block and a function can share an address
    pub fn from_addr_mut<'ctx>(
        ctx: &'ctx mut Context<'str>,
        addr: u64,
    ) -> Option<BlockMutRef<'str, 'ctx>> {
        ctx.get_at_addr(&addr)
            .and_then(|id| match id {
                ValueId::BasicBlock(block_id) => Some(block_id),
                ValueId::Function(function_id) => Function::from_id(ctx, function_id)
                    .root()
                    .map(|root| root.id),
                _ => None,
            })
            .map(|id| BasicBlock::from_id_mut(ctx, id))
    }

    /// Create a new block, born into `func`'s block arena. Its `parent` is set
    /// to `func` (ownership == arena membership).
    pub fn make<'ctx>(ctx: &'ctx mut Context<'str>, func: FunctionId) -> BlockMutRef<'str, 'ctx> {
        let block = BasicBlock {
            parent: Some(func),
            ..BasicBlock::default()
        };
        let id = ctx.values.push_block(func, block);
        BlockMutRef::new(ctx, id)
    }

    /// Deep-clone the block at `orig` into a new block in the same context.
    /// Updates `value_map` with parameters and instructions remapping.
    pub fn clone_into_ctx(
        ctx: &mut Context<'str>,
        orig: BlockId,
        value_map: &mut HashMap<ValueId, ValueId>,
    ) -> BlockId {
        // Create a fresh block in the same function as `orig`.
        let new_block_id = BasicBlock::make(ctx, orig.func).id;

        let name = Cow::Owned(format!(
            "clone_{:x}",
            ctx.values.block(orig).address.unwrap_or(0)
        ));
        let unique_name = ctx.get_unique_name(name);
        BasicBlock::from_id_mut(ctx, new_block_id)
            .rename(unique_name)
            .expect("name was deduplicated");

        // Clone parameters
        for old_param_id in &ctx.values.block(orig).params.clone() {
            let old_param = ctx.values.block_param(*old_param_id).clone();
            let new_param_id = ctx.values.push_block_param(
                new_block_id.func,
                BlockParam {
                    parent: Some(new_block_id),
                    ..old_param
                },
            );

            BasicBlock::from_id_mut(ctx, new_block_id).push_existing_param(new_param_id);
            value_map.insert(
                ValueId::BlockParam(*old_param_id),
                ValueId::BlockParam(new_param_id),
            );
        }

        // Clone instructions
        let orig_insns = ctx.values.block(orig).instructions.clone();
        for &old_insn_id in orig_insns.iter() {
            // Extract information from the old instruciton
            let insn_ref = Instruction::from_id(ctx, old_insn_id);
            let size = insn_ref.size();
            let space = insn_ref.space().map(|s| s.id);

            // Create the new instruction and derive the cloned mnemonic
            let mut new_mnemonic = insn_ref.mnemonic().clone();

            // Only remap the values this instruction actually references. This
            // avoids scanning the whole (trace-wide) `value_map` per instruction
            // and sidesteps chained `old -> new -> newer` replacements that a
            // full iteration could trigger.
            for old in new_mnemonic.args() {
                if let Some(&new) = value_map.get(&old) {
                    new_mnemonic.replace_value(old, new);
                }
            }

            let new_insn_id = InstructionRef::from_mnemonic_with_space(
                ctx,
                new_block_id.func,
                new_mnemonic,
                size,
                space,
            )
            .id;

            BasicBlock::from_id_mut(ctx, new_block_id).push_insn(new_insn_id);

            value_map.insert(
                ValueId::Instruction(old_insn_id),
                ValueId::Instruction(new_insn_id),
            );
        }

        new_block_id
    }
}

// Shared read-only methods available on both BlockRef and BlockMutRef
impl<'s, 'ctx: 's, 'str: 'ctx, Ctx> BaseRef<Ctx, BlockId>
where
    Self: WithCtx<'s, 'ctx, 'str>,
{
    fn inner(&'s self) -> &'ctx BasicBlock<'str> {
        self.ctx().values.block(self.id)
    }

    /// Iterates over outgoing `(edge_id, successor_block_id)` pairs.
    pub fn successors(&'s self) -> impl Iterator<Item = (EdgeId, BlockId)> + 's {
        use jstd::graph::Node;
        BlockRef::new(self.ctx(), self.id)
            .children()
            .map(|item| (item.edge_id(), item.node_id()))
    }

    /// Iterates over incoming `(edge_id, predecessor_block_id)` pairs.
    pub fn predecessors(&'s self) -> impl Iterator<Item = (EdgeId, BlockId)> + 's {
        use jstd::graph::Node;
        BlockRef::new(self.ctx(), self.id)
            .parents()
            .map(|item| (item.edge_id(), item.node_id()))
    }

    pub fn name(&'s self) -> Option<&'ctx str> {
        self.inner().name.as_deref()
    }

    /// Returns the machine address of this block, if it has one.
    pub fn address(&'s self) -> Option<u64> {
        self.inner().address
    }

    pub fn comment(&'s self) -> Option<&'ctx str> {
        self.inner().comment.as_deref()
    }

    /// Iterates over this block's parameters in declaration order.
    pub fn params(&'s self) -> impl Iterator<Item = BlockParamRef<'str, 'ctx>> + 's {
        self.inner()
            .params
            .iter()
            .map(|&id| BlockParam::from_id(self.ctx(), id))
    }

    /// Returns the number of parameters declared on this block.
    pub fn num_params(&'s self) -> usize {
        self.inner().params.len()
    }

    /// Iterates over the instructions in this block
    pub fn instructions(&'s self) -> InstructionIter<'str, 'ctx> {
        let inner = self.inner();
        InstructionIter {
            ctx: self.ctx(),
            inner: inner.instructions.iter(),
        }
    }

    /// Iterates over the instructions in this block
    /// alias for `instructions()`
    pub fn iter(&'s self) -> InstructionIter<'str, 'ctx> {
        self.instructions()
    }

    pub fn instruction_ids(&'s self) -> &'ctx [InstructionId] {
        &self.inner().instructions
    }

    /// Does this block have any instructions?
    pub fn is_empty(&'s self) -> bool {
        self.inner().instructions.is_empty()
    }

    /// Does this block finish with a terminator instruction?
    pub fn is_terminated(&'s self) -> bool {
        self.iter().last().is_some_and(|insn| insn.is_terminator())
    }

    pub fn parent(&'s self) -> Option<FunctionRef<'str, 'ctx>> {
        self.inner()
            .parent
            .map(|fid| Function::from_id(self.ctx(), fid))
    }

    pub fn function(&'s self) -> Option<FunctionRef<'str, 'ctx>> {
        self.parent()
    }

    fn fmt(&'s self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = self.name().unwrap_or("unnamed");
        write!(f, "<{name}")?;
        for param in self.params() {
            write!(f, " ")?;
            param.fmt_decl(f)?;
        }
        writeln!(f, ">")?;

        if let Some(comment) = self.comment() {
            for line in comment.lines() {
                writeln!(f, "\t// {line}")?;
            }
        }

        self.iter().try_for_each(|instr| {
            write!(f, "\t")?;
            instr.as_statement().fmt(f)?;
            writeln!(f)
        })?;

        // A `call` / `call [..]` / `goto [..]` terminator encodes no successors in
        // its own syntax, so emit its out-edges as a `// -> <a>, <b>` hint that the
        // parser reads back into CFG edges (a direct/indirect call's return block,
        // an indirect jump's resolved targets). Without this such edges would be
        // lost on round-trip.
        use crate::value::insn::Mnemonic;
        if let Some(term) = self.iter().last()
            && matches!(
                term.mnemonic(),
                Mnemonic::Call(_) | Mnemonic::CallInd(_) | Mnemonic::BranchInd(_)
            )
        {
            let mut succ: Vec<&str> = self
                .successors()
                .map(|(_, b)| {
                    BasicBlock::from_id(self.ctx(), b)
                        .name()
                        .unwrap_or("unnamed")
                })
                .collect();
            if !succ.is_empty() {
                succ.sort_unstable();
                write!(f, "\t// ->")?;
                for (i, name) in succ.iter().enumerate() {
                    write!(f, "{} <{name}>", if i == 0 { "" } else { "," })?;
                }
                writeln!(f)?;
            }
        }

        Ok(())
    }
}

pub type BlockRef<'str, 'ctx> = BaseRef<&'ctx Context<'str>, BlockId>;

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 'ctx, 'str> for BlockRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'ctx Context<'str> {
        self.ctx
    }
}

impl Named for BlockRef<'_, '_> {
    fn name(&self) -> Option<&str> {
        self.ctx.values.block(self.id).name.as_deref()
    }
}

impl Display for BlockRef<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.fmt(f)
    }
}

impl<'str, 'ctx> Value<'str, 'ctx> for BlockRef<'str, 'ctx> {
    fn id(&self) -> ValueId {
        self.id()
    }

    fn size(&self) -> usize {
        0
    }
}

pub struct InstructionIter<'str, 'ctx> {
    ctx: &'ctx Context<'str>,
    inner: slice::Iter<'ctx, InstructionId>,
}

impl<'str, 'ctx> Iterator for InstructionIter<'str, 'ctx> {
    type Item = InstructionRef<'str, 'ctx>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|id| InstructionRef::new(self.ctx, *id))
    }
}

impl<'str, 'ctx> IntoIterator for &BlockRef<'str, 'ctx> {
    type Item = InstructionRef<'str, 'ctx>;
    type IntoIter = InstructionIter<'str, 'ctx>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub type BlockMutRef<'str, 'ctx> = BaseRef<&'ctx mut Context<'str>, BlockId>;

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 's, 'str> for BlockMutRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'s Context<'str> {
        self.ctx
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtxMut<'s, 'str> for BlockMutRef<'str, 'ctx> {
    fn ctx_mut(&'s mut self) -> &'s mut Context<'str> {
        self.ctx
    }
}

impl Named for BlockMutRef<'_, '_> {
    fn name(&self) -> Option<&str> {
        self.ctx.values.block(self.id).name.as_deref()
    }
}

impl<'str, 'ctx> Renameable<'str, 'ctx> for BlockMutRef<'str, 'ctx> {
    fn rename(&mut self, name: Cow<'str, str>) -> Result<()> {
        let id = self.id.into();
        let old_name = self
            .ctx
            .values
            .block(self.id)
            .name
            .as_deref()
            .map(str::to_owned);
        update_context_name(id, self.ctx, name.clone(), old_name.as_deref())?;
        self.ctx.values.block_mut(self.id).name = Some(name);
        Ok(())
    }
}

impl Display for BlockMutRef<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.fmt(f)
    }
}

impl<'str, 'ctx> Value<'str, 'ctx> for BlockMutRef<'str, 'ctx> {
    fn id(&self) -> ValueId {
        self.id()
    }

    fn size(&self) -> usize {
        0
    }
}

impl<'str, 'ctx> BlockMutRef<'str, 'ctx> {
    /// Builder-style address assignment. Panics if `addr` is already mapped.
    /// Use [`set_address`](Self::set_address) for fallible assignment.
    pub fn with_address(mut self, addr: u64) -> Self {
        self.set_address(addr)
            .expect("address is already mapped to a value");
        self
    }

    #[allow(unused_mut)]
    pub fn in_function(mut self, fun_id: FunctionId) -> Self {
        Function::from_id_mut(self.ctx, fun_id).add_block(self.id);
        self
    }

    pub fn with_id(&mut self, id: BlockId) -> &mut Self {
        self.id = id;
        self
    }

    /// Reborrows this `BlockMutRef`, shortening the lifetime.
    pub fn reborrow(&mut self) -> BlockMutRef<'str, '_> {
        BlockMutRef::from_id(self.ctx, self.id)
    }

    pub(in crate::value) fn inner_mut(&mut self) -> &mut BasicBlock<'str> {
        self.ctx.values.block_mut(self.id)
    }

    pub fn parent_mut(&mut self) -> Option<FunctionMutRef<'str, '_>> {
        self.ctx
            .values
            .block(self.id)
            .parent
            .map(|fid| Function::from_id_mut(self.ctx, fid))
    }

    pub fn as_ref(&self) -> BlockRef<'str, '_> {
        BlockRef::new(self.ctx, self.id)
    }

    pub fn set_comment(&mut self, comment: Option<String>) {
        self.inner_mut().comment = comment;
    }

    /// Declares a new parameter on this block with the given size in bytes.
    ///
    /// The parameter is appended to the block's `params` list and its `parent`
    /// is set to this block. It does NOT appear in `instructions`.
    /// Returns a mutable reference whose `ValueId` can be used as an operand.
    pub fn push_param(&mut self, size: usize) -> BlockParamMutRef<'str, '_> {
        let block_id = self.id;
        let index = self.inner().params.len();
        let type_id = self.ctx.types.get_or_make_int(size);
        let id = self.ctx.values.push_block_param(
            block_id.func,
            BlockParam {
                index,
                type_id,
                parent: Some(block_id),
                name: None,
                origin: None,
                protected: false,
            },
        );
        self.inner_mut().params.push(id);
        BlockParamMutRef::from_id(self.ctx, id)
    }

    /// Appends an already-created block parameter to the parameters list
    pub fn push_existing_param(&mut self, id: BlockParamId) {
        self.inner_mut().params.push(id);
    }

    fn insert_insn(&mut self, index: usize, insn_id: InstructionId) {
        Instruction::from_id_mut(self.ctx, insn_id)
            .inner_mut()
            .parent = Some(self.id);

        self.inner_mut().instructions.insert(index, insn_id);
    }

    /// Inserts an instruction at the start of this block, before all existing instructions.
    pub fn insert_insn_at_start(&mut self, insn_id: InstructionId) {
        self.insert_insn(0, insn_id);
    }

    /// Inserts an instruction at the given index, shifting later instructions right.
    /// Panics if `index > len`.
    pub fn insert_insn_at_index(&mut self, index: usize, insn_id: InstructionId) {
        self.insert_insn(index, insn_id);
    }

    /// Inserts an instruction before the instruction identified by `before_id` in this block.
    /// Panics if `before_id` is not an instruction in this block.
    pub fn insert_insn_before(&mut self, before_id: InstructionId, insn_id: InstructionId) {
        let index = self
            .inner()
            .instructions
            .iter()
            .position(|&id| id == before_id)
            .expect("before_id not found in block");
        self.insert_insn(index, insn_id);
    }

    /// Inserts an instruction after the instruction identified by `after_id` in this block.
    /// Panics if `after_id` is not an instruction in this block.
    pub fn insert_insn_after(&mut self, after_id: InstructionId, insn_id: InstructionId) {
        let index = self
            .inner()
            .instructions
            .iter()
            .position(|&id| id == after_id)
            .expect("after_id not found in block");
        self.insert_insn(index + 1, insn_id);
    }

    /// Pushes an instruction to the end of this block.
    pub fn push_insn(&mut self, id: InstructionId) {
        self.insert_insn(self.inner().instructions.len(), id);
    }

    /// Retains only the instructions for which `f` returns true, deleting the
    /// removed instructions.
    pub fn retain_insns(&mut self, mut f: impl FnMut(&InstructionId) -> bool) {
        let mut removed = Vec::new();
        self.inner_mut().instructions.retain(|id| {
            if f(id) {
                true
            } else {
                removed.push(*id);
                false
            }
        });
        for id in removed {
            self.ctx.remove_instruction(id);
        }
    }

    /// Adds an edge to this block's edge set.
    /// DO NOT USE THIS
    pub(crate) fn add_edge(&mut self, edge_id: EdgeId) {
        self.inner_mut().edges.insert(edge_id);
    }

    /// Removes an edge from this block's edge set.
    /// DO NOT USE THIS
    pub(crate) fn remove_edge(&mut self, edge_id: EdgeId) {
        self.inner_mut().edges.remove(&edge_id);
    }

    /// Removes the last instruction from this block.
    pub fn pop_insn(&mut self) {
        if let Some(last_id) = self.inner().instructions.last() {
            self.ctx.remove_instruction(*last_id);
        }
    }

    /// Appends a slice of instruction ids to this block.
    pub fn extend_insns(&mut self, insns: &[InstructionId]) {
        self.inner_mut().instructions.extend_from_slice(insns);
    }

    /// Removes this block from `function_id`'s block list, unlinks every CFG
    /// edge incident to it, and clears its parent.
    ///
    /// Detaching the edges is what keeps a deleted block from leaving phantom
    /// predecessors/successors on its neighbours (e.g. an unrolled-away loop
    /// body whose stale exit edge would otherwise inflate the exit block's
    /// predecessor count and block `simplify_cfg` from merging it).
    pub fn delete(&mut self, function_id: FunctionId) {
        // Snapshot first: `remove_cfg_edge` mutates this block's edge set. A
        // self-loop appears once in the set and unlinks cleanly (both endpoints
        // are this block, so the second remove is a no-op).
        let edges: Vec<EdgeId> = self.inner().edges.iter().copied().collect();
        for edge in edges {
            self.ctx.remove_cfg_edge(edge);
        }

        // Remove the block's instructions, not just the block. Otherwise they
        // linger in the value arena: still registered in their operands' use-lists,
        // so `ctx.users(v)` keeps returning a deleted block's `load`/`store`/etc.
        // even though the block is gone from the CFG. Arena-global analyses (e.g.
        // `argpromote`'s `analyze_param`, which scans `ctx.users`) would then see
        // phantom accesses from the deleted block. `remove_instruction` updates the
        // use-lists (via `remove_instructions`) and clears each instruction's
        // parent; outgoing edges were already dropped above.
        let insns: Vec<InstructionId> = self.inner().instructions.clone();
        for insn in insns {
            self.ctx.remove_instruction(insn);
        }

        // Detach the block's parameters too. They are value defs (e.g. a loop's
        // induction variable) just like instruction results; leaving them with a
        // stale `parent` pointing at the now-deleted block would dangle the same
        // way a tombstoned instruction would. Callers must already have unlinked
        // their uses (the params have no live readers once the block is gone).
        let params: Vec<BlockParamId> = self.inner().params.clone();
        for param in params {
            self.ctx.values.users.remove(&ValueId::BlockParam(param));
            self.ctx.values.block_param_mut(param).parent = None;
        }

        // Tombstone the block: drop it from its owner's roster (append-only
        // arena slot is never reclaimed).
        let _ = function_id;
        let id = self.id;
        self.ctx.values.unroster_block(id);
        let block = self.inner_mut();
        block.parent = None;
        block.deleted = true;
    }

    /// Absorbs `other` into this block: removes the terminal branch, appends
    /// `other`'s instructions, rehomes `other`'s outgoing edges to this block,
    /// removes `other` from `function_id`, and transfers `other`'s addresses.
    ///
    /// `edge_ab` must be the direct edge from this block to `other`.
    pub fn absorb_block(&mut self, other: BlockId, edge_ab: EdgeId, function_id: FunctionId) {
        let branch_args = self
            .inner()
            .instructions
            .last()
            .and_then(|&id| match self.ctx.values.instruction(id).mnemonic() {
                crate::value::insn::Mnemonic::Branch(branch) if branch.target == other => {
                    Some(branch.args.clone())
                }
                _ => None,
            })
            .unwrap_or_default();
        let other_params = self.ctx.values.block(other).params.clone();
        if !other_params.is_empty() {
            assert_eq!(
                other_params.len(),
                branch_args.len(),
                "cannot absorb block with {} params through branch with {} args",
                other_params.len(),
                branch_args.len()
            );
            for (param, arg) in other_params.into_iter().zip(branch_args) {
                self.ctx.replace_all_uses_with(param, arg);
            }
        }

        // Remove terminal branch.
        self.inner_mut().instructions.pop();

        // Move other's instructions into this block, updating their parent. The
        // source list must be drained, not just copied: leaving the ids in
        // `other.instructions` would put every absorbed instruction in two
        // blocks at once, so a later `remove_instruction` (which unlinks via the
        // instruction's `parent`) clears it from one block while it lingers in
        // the other — corrupting block membership.
        let b_insns = std::mem::take(&mut self.ctx.values.block_mut(other).instructions);
        let self_id = self.id;
        for &insn_id in &b_insns {
            self.ctx.values.instruction_mut(insn_id).parent = Some(self_id);
        }
        self.inner_mut().instructions.extend(b_insns);

        // Rehome other's edges to this block at the graph level.
        self.ctx.merge_nodes(self.id, other, edge_ab);

        // Tombstone `other`: with per-function block arenas, removal from the
        // function is a `deleted` flag (the arena slot is never reclaimed).
        let _ = function_id;
        let b_addr = self.ctx.values.block(other).address;
        let b_extra = self.ctx.values.block(other).extra_addresses.clone();
        self.ctx.values.unroster_block(other);
        {
            let other_block = self.ctx.values.block_mut(other);
            other_block.parent = None;
            other_block.deleted = true;
        }
        if let Some(addr) = b_addr {
            self.inner_mut().extra_addresses.push(addr);
        }
        self.inner_mut().extra_addresses.extend(b_extra);
    }

    /// Associates this block with `addr` in the context address map.
    /// Names the block after `addr` if it doesn't already have a name.
    /// Returns `Err` if another value is already mapped to `addr`.
    pub fn set_address(&mut self, addr: u64) -> Result<()> {
        self.inner_mut().address = Some(addr);
        self.ctx.set_address(addr, self.id.into())?;

        if self.name().is_none() {
            // Block labels share the global name map with register varnodes, whose
            // names are short mnemonics (`cf`, `sf`, `ax`, …). A block landing at a
            // low address whose hex spells such a name (0xcf → "cf") would otherwise
            // collide and panic through `with_address`. Disambiguate with a numeric
            // suffix, keeping the bare-hex label for the overwhelmingly common case.
            let label = self.ctx.get_unique_name(Cow::Owned(format!("{addr:x}")));
            self.rename(label)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use jstd::graph::Node;
    use qcode_macro::qcode;

    #[test]
    fn test_create_block_at_address() {
        let mut ctx = Context::new();
        let id = BasicBlock::make(&mut ctx).with_address(0x2000).id;
        let block_by_addr =
            BasicBlock::from_addr(&ctx, 0x2000).expect("block not found by address");
        assert_eq!(id, block_by_addr.id);
        assert_eq!(block_by_addr.address(), Some(0x2000));
    }

    #[test]
    #[should_panic(expected = "address is already mapped to a value")]
    fn test_create_block_at_duplicate_address() {
        let mut ctx = Context::new();
        BasicBlock::make(&mut ctx).with_address(0x2000);
        BasicBlock::make(&mut ctx).with_address(0x2000);
    }

    #[test]
    fn test_block_child() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <entry>
                goto <body>;
            <body>
            "
        );
        let entry = BasicBlock::from_name(&ctx, "entry").unwrap();
        let children: Vec<_> = entry
            .children()
            .map(|b| b.node().name().unwrap_or("").to_string())
            .collect();
        assert_eq!(children, ["body"]);
    }

    #[test]
    fn iter_yields_all_instructions() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 X;
            varnode i64 Y;

            <block>
                %x = load(X:8, &X);
                %y = load(Y:8, &Y);
                %sum = i64 %x + i64 %y;
                return at i64 0;
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let count = block.iter().count();
        assert_eq!(count, 4);

        let mut iter = block.iter();

        assert_eq!(
            iter.next().unwrap().as_statement().to_string(),
            "i64 %x = load(X:8, i64 X);"
        );
        assert_eq!(
            iter.next().unwrap().as_statement().to_string(),
            "i64 %y = load(Y:8, i64 Y);"
        );
        assert_eq!(
            iter.next().unwrap().as_statement().to_string(),
            "i64 %sum = i64 %x + i64 %y;"
        );
        assert_eq!(
            iter.next().unwrap().as_statement().to_string(),
            "return at i64 0x0;"
        );
    }

    #[test]
    fn into_iterator_for_block_ref_matches_iter() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 X;
            varnode i64 Y;

            <block>
                %x = load(X:8, &X);
                %y = load(Y:8, &Y);
                %sum = i64 %x + i64 %y;
                return at i64 0;
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let via_iter: Vec<_> = block.iter().map(|i| i.id).collect();
        let via_into: Vec<_> = (&block).into_iter().map(|i| i.id).collect();
        assert_eq!(via_iter, via_into);
    }

    #[test]
    fn push_param_adds_to_params_not_instructions() {
        let mut ctx = Context::new();
        let mut block = BasicBlock::make(&mut ctx);

        assert_eq!(block.num_params(), 0);
        assert_eq!(block.as_ref().instruction_ids().len(), 0);

        block.push_param(8);
        assert_eq!(block.num_params(), 1);
        assert_eq!(block.as_ref().instruction_ids().len(), 0);

        block.push_param(4);
        assert_eq!(block.num_params(), 2);
        assert_eq!(block.as_ref().instruction_ids().len(), 0);
    }

    #[test]
    fn params_iter_yields_in_order() {
        let mut ctx = Context::new();
        let mut block = BasicBlock::make(&mut ctx);

        let p0_id = block.push_param(8).id;
        let p1_id = block.push_param(4).id;
        let block_ref = block.as_ref();

        let param_ids: Vec<_> = block_ref.params().map(|p| p.id).collect();
        assert_eq!(param_ids, [p0_id, p1_id]);
        assert_eq!(block_ref.params().next().unwrap().index(), 0);
        assert_eq!(block_ref.params().nth(1).unwrap().index(), 1);
    }

    #[test]
    fn qcode_macro_block_with_params() {
        use crate::context::Context;
        use qcode_macro::qcode;

        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <entry @v1:i64 @v2:i32>
                goto <done @x=@v1 @y=@v2>;

            <done @x:i64 @y:i32>
                goto <0x1001>;
            "
        );

        let entry = BasicBlock::from_id(&ctx, entry);
        assert_eq!(entry.num_params(), 2, "entry should have 2 params");

        let params = entry.params().collect::<Vec<_>>();
        assert_eq!(params[0].name(), Some("v1"));
        assert_eq!(params[0].size(), 8);
        assert_eq!(params[1].name(), Some("v2"));
        assert_eq!(params[1].size(), 4);

        let done_block = BasicBlock::from_id(&ctx, done);
        assert_eq!(done_block.num_params(), 2, "done should have 2 params");
        let done_params = done_block.params().collect::<Vec<_>>();
        assert_eq!(done_params[0].size(), 8);
        assert_eq!(done_params[1].size(), 4);

        // The branch from entry should carry 2 args.
        let branch_insn = entry.iter().last().expect("entry has instructions");
        let crate::value::insn::Mnemonic::Branch(branch) = branch_insn.mnemonic() else {
            panic!("expected branch");
        };
        assert_eq!(branch.args.len(), 2);
    }

    #[test]
    fn block_display_uses_qcode_param_syntax() {
        use crate::context::Context;
        use qcode_macro::qcode;

        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <entry @a @b>
                goto <0x1001>;
            "
        );

        let entry = BasicBlock::from_id(&ctx, entry);
        assert!(entry.to_string().starts_with("<entry @a @b>\n"));
    }

    #[test]
    fn param_value_id_usable_in_instruction() {
        use crate::{builder::Builder, value::ValueId};
        let mut ctx = Context::new();
        let block_id = BasicBlock::make(&mut ctx).id;

        let param_id: ValueId = {
            let mut block = BasicBlock::from_id_mut(&mut ctx, block_id);
            block.push_param(8).id()
        };

        let mut builder = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, block_id));
        let sum = builder.push_add(param_id, param_id);
        assert_eq!(sum.size(), 8);
        unsafe { builder.dont_finalize() };
    }

    #[test]
    fn remove_terminator_branch() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <entry>
                goto <done>;
            <done>
            "
        );

        let mut entry_block = BasicBlock::from_id_mut(&mut ctx, entry);
        assert_eq!(entry_block.instruction_ids().len(), 1);
        assert_eq!(entry_block.successors().count(), 1);
        assert!(entry_block.is_terminated());

        entry_block.pop_insn();
        assert_eq!(entry_block.instruction_ids().len(), 0);
        assert_eq!(entry_block.successors().count(), 0);
        assert!(!entry_block.is_terminated());
    }

    #[test]
    fn remove_terminator_cbranch() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;

            <entry>
                %c = load(cond:1, cond);
                if %c goto <then_lbl> else goto <else_lbl>;

            <then_lbl>

            <else_lbl>
            "
        );

        let mut entry_block = BasicBlock::from_id_mut(&mut ctx, entry);
        assert_eq!(entry_block.successors().count(), 2);
        assert_eq!(entry_block.instruction_ids().len(), 2);
        assert!(entry_block.is_terminated());

        entry_block.pop_insn();

        assert_eq!(entry_block.successors().count(), 0);
        assert_eq!(entry_block.instruction_ids().len(), 1);
        assert!(!entry_block.is_terminated());

        assert_eq!(
            BasicBlock::from_id(&ctx, then_lbl).predecessors().count(),
            0
        );
        assert_eq!(
            BasicBlock::from_id(&ctx, else_lbl).predecessors().count(),
            0
        );
    }

    #[test]
    fn remove_terminator_return() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <entry>
                return at i64 0;
            "
        );

        let mut entry_block = BasicBlock::from_id_mut(&mut ctx, entry);
        assert_eq!(entry_block.instruction_ids().len(), 1);
        assert_eq!(entry_block.successors().count(), 0);

        entry_block.pop_insn();
        assert_eq!(entry_block.instruction_ids().len(), 0);
        assert_eq!(entry_block.successors().count(), 0);
    }

    #[test]
    fn clone_into_ctx_produces_distinct_ids() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 X;
            varnode i64 Y;

            <block>
                %x = load(X:8, &X);
                %y = load(Y:8, &Y);
                %sum = i64 %x + i64 %y;
                return at i64 0;
            "
        );

        let mut value_map = HashMap::default();
        let cloned_id = BasicBlock::clone_into_ctx(&mut ctx, block, &mut value_map);

        let orig = BasicBlock::from_id(&ctx, block);
        let cloned = BasicBlock::from_id(&ctx, cloned_id);

        assert_ne!(block, cloned_id, "cloned block must have a different id");
        assert_ne!(
            orig.name(),
            cloned.name(),
            "cloned block must have a different name"
        );

        assert_eq!(orig.instruction_ids().len(), cloned.instruction_ids().len());
        for (orig_id, clone_id) in orig.instruction_ids().iter().zip(cloned.instruction_ids()) {
            assert_ne!(
                orig_id, clone_id,
                "cloned instruction must have a different id"
            );
        }
    }

    #[test]
    fn clone_into_ctx_remaps_operands() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 X;
            varnode i64 Y;

            <block>
                %x = load(X:8, &X);
                %y = load(Y:8, &Y);
                %sum = i64 %x + i64 %y;
                return at i64 0;
            "
        );

        let mut value_map = HashMap::default();
        let cloned_id = BasicBlock::clone_into_ctx(&mut ctx, block, &mut value_map);
        let cloned = BasicBlock::from_id(&ctx, cloned_id);

        let orig_value_ids: HashSet<ValueId> = value_map.keys().copied().collect();

        // All operands in the clone must reference new (remapped) values, not the originals
        // so no value map key should be referenced
        for arg in cloned.iter().flat_map(|i| i.mnemonic().args()) {
            assert!(
                !orig_value_ids.contains(&arg),
                "cloned instruction still references original value {arg:?}"
            );
        }
    }

    /// Regression: deleting a block must unlink its CFG edges, so it leaves no
    /// phantom predecessor on a block it used to branch to. (An unrolled-away
    /// loop body's stale exit edge would otherwise inflate the exit block's
    /// predecessor count and block `simplify_cfg` from merging it.)
    #[test]
    fn delete_unlinks_incident_edges() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                goto <exit>;
            <b>
                goto <exit>;
            <exit>
                return at i64 0;
            "
        );

        assert_eq!(BasicBlock::from_id(&ctx, exit).predecessors().count(), 2);

        BasicBlock::from_id_mut(&mut ctx, b).delete(f);

        assert_eq!(
            BasicBlock::from_id(&ctx, exit).predecessors().count(),
            1,
            "deleted block's edge must not linger as a phantom predecessor"
        );
        assert_eq!(BasicBlock::from_id(&ctx, b).successors().count(), 0);
    }

    /// A self-loop edge appears once in the block's edge set and must unlink
    /// cleanly on delete without double-removal trouble.
    #[test]
    fn delete_unlinks_self_loop() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                goto <loop_hdr>;
            <loop_hdr>
                goto <loop_hdr>;
            "
        );

        assert!(
            BasicBlock::from_id(&ctx, loop_hdr)
                .successors()
                .any(|(_, s)| s == loop_hdr)
        );

        BasicBlock::from_id_mut(&mut ctx, loop_hdr).delete(f);

        assert_eq!(BasicBlock::from_id(&ctx, loop_hdr).successors().count(), 0);
        assert_eq!(
            BasicBlock::from_id(&ctx, loop_hdr).predecessors().count(),
            0
        );
    }

    /// Regression: deleting a block must also remove its instructions from the
    /// value arena's use-lists. Otherwise a deleted block's instruction lingers as
    /// an orphan — still registered as a user of its operands — so `ctx.users(v)`
    /// keeps returning it even though the block is gone from the CFG, misleading
    /// arena-global analyses (this caused `argpromote` to see a phantom access from
    /// a loop body the unroller had deleted).
    #[test]
    fn delete_removes_instructions_from_use_lists() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
            <a>
                %x = i64 1 + i64 2;
                goto <b>;
            <b>
                %y = %x + i64 3;
                goto <exit>;
            <exit>
                return at i64 0;
            "
        );

        // `%x` (in the surviving entry) is used by `%y` (in block `b`).
        assert!(
            ctx.users(crate::value::ValueId::Instruction(x))
                .to_vec()
                .contains(&y),
            "precondition: %x is used by %y"
        );

        BasicBlock::from_id_mut(&mut ctx, b).delete(f);

        assert!(
            !ctx.users(crate::value::ValueId::Instruction(x))
                .to_vec()
                .contains(&y),
            "deleting b must unregister %y from %x's use-list, not orphan it"
        );
        assert!(
            ctx.get_insn(y).parent().is_none(),
            "deleted instruction must have its parent cleared"
        );
    }
}
