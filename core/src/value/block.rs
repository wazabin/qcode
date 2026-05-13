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
use jstd::graph::Graph;
use std::{
    borrow::Cow,
    collections::HashSet,
    fmt::{Display, Formatter},
};

pub(crate) use self::cfg::EdgeData;
pub use self::cfg::{BlockId, EdgeId, EdgeMutRef, EdgeRef};
pub mod cfg;

/// A block of instructions.
/// This is the basic unit of code in our IR.
#[derive(Debug, Default, Clone)]
pub struct BasicBlock<'str> {
    /// An optionnal name for this basic block
    name: Option<Cow<'str, str>>,

    /// Typed parameters declared at block entry (block-argument style).
    /// These are NOT part of `instructions`; use `params()` to iterate them.
    pub params: Vec<BlockParamId>,

    /// The ids of the instructions in this block
    pub instructions: Vec<InstructionId>,

    /// The set of edges that this block is incident to.
    pub edges: HashSet<EdgeId>,

    /// The address of this block, if it corresponds to a machine address.
    pub address: Option<u64>,

    /// Additional addresses that map to this block (accumulated from merged blocks).
    pub extra_addresses: Vec<u64>,

    /// The function this block belongs to, if any.
    pub parent: Option<FunctionId>,
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

    /// Create a new block
    pub fn make<'ctx>(ctx: &'ctx mut Context<'str>) -> BlockMutRef<'str, 'ctx> {
        let id = ctx.values.push_block(BasicBlock::default());
        BlockMutRef::new(ctx, id)
    }
}

// Shared read-only methods available on both BlockRef and BlockMutRef
impl<'s, 'ctx: 's, 'str: 'ctx, Ctx> BaseRef<Ctx, BlockId>
where
    Self: WithCtx<'s, 'ctx, 'str>,
{
    fn inner(&'s self) -> &'ctx BasicBlock<'str> {
        &self.ctx().values.basic_blocks[self.id]
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
            write!(f, " {param}")?;
        }
        writeln!(f, ">")?;

        self.iter().try_for_each(|instr| {
            write!(f, "\t")?;
            instr.as_statement().fmt(f)?;
            writeln!(f)
        })?;

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
        self.ctx.values.basic_blocks[self.id].name.as_deref()
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
        self.ctx.values.basic_blocks[self.id].name.as_deref()
    }
}

impl<'str, 'ctx> Renameable<'str, 'ctx> for BlockMutRef<'str, 'ctx> {
    fn rename(&mut self, name: Cow<'str, str>) -> Result<'str, ()> {
        let id = self.id.into();
        let old_name = self.ctx.values.basic_blocks[self.id]
            .name
            .as_deref()
            .map(str::to_owned);
        update_context_name(id, self.ctx, name.clone(), old_name.as_deref())?;
        self.ctx.values.basic_blocks[self.id].name = Some(name);
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
        &mut self.ctx.values.basic_blocks[self.id]
    }

    pub fn parent_mut(&mut self) -> Option<FunctionMutRef<'str, '_>> {
        self.ctx.values.basic_blocks[self.id]
            .parent
            .map(|fid| Function::from_id_mut(self.ctx, fid))
    }

    pub fn as_ref(&self) -> BlockRef<'str, '_> {
        BlockRef::new(self.ctx, self.id)
    }

    /// Declares a new parameter on this block with the given size in bytes.
    ///
    /// The parameter is appended to the block's `params` list and its `parent`
    /// is set to this block. It does NOT appear in `instructions`.
    /// Returns a mutable reference whose `ValueId` can be used as an operand.
    pub fn push_param(&mut self, size: usize) -> BlockParamMutRef<'str, '_> {
        let block_id = self.id;
        let index = self.inner().params.len();
        let id = self.ctx.values.block_params.push(BlockParam {
            index,
            size,
            parent: Some(block_id),
            name: None,
        });
        self.inner_mut().params.push(id);
        BlockParamMutRef::from_id(self.ctx, id)
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
    pub fn add_edge(&mut self, edge_id: EdgeId) {
        self.inner_mut().edges.insert(edge_id);
    }

    /// Removes an edge from this block's edge set.
    pub fn remove_edge(&mut self, edge_id: EdgeId) {
        self.inner_mut().edges.remove(&edge_id);
    }

    /// Removes the last instruction from this block.
    pub fn pop_insn(&mut self) -> Option<InstructionId> {
        self.inner_mut().instructions.pop()
    }

    /// Appends a slice of instruction ids to this block.
    pub fn extend_insns(&mut self, insns: &[InstructionId]) {
        self.inner_mut().instructions.extend_from_slice(insns);
    }

    /// Removes this block from `function_id`'s block list and clears its parent.
    pub fn delete(&mut self, function_id: FunctionId) {
        let id = self.id;

        Function::from_id_mut(self.ctx, function_id)
            .inner_mut()
            .blocks
            .retain(|&b| b != id);

        self.inner_mut().parent = None;
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
            .and_then(|&id| match self.ctx.values.instructions[id].mnemonic() {
                crate::value::insn::Mnemonic::Branch(branch) if branch.target == other => {
                    Some(branch.args.clone())
                }
                _ => None,
            })
            .unwrap_or_default();
        let other_params = self.ctx.values.basic_blocks[other].params.clone();
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

        // Append other's instructions, updating their parent to point to this block.
        let b_insns = self.ctx.values.basic_blocks[other].instructions.clone();
        let self_id = self.id;
        for &insn_id in &b_insns {
            self.ctx.values.instructions[insn_id].parent = Some(self_id);
        }
        self.inner_mut().instructions.extend(b_insns);

        // Rehome other's edges to this block at the graph level.
        self.ctx.merge_nodes(self.id, other, edge_ab);

        // Remove other from function and clear its parent.
        Function::from_id_mut(self.ctx, function_id)
            .inner_mut()
            .blocks
            .retain(|&id| id != other);
        self.ctx.values.basic_blocks[other].parent = None;

        // Transfer other's addresses.
        let b_addr = self.ctx.values.basic_blocks[other].address;
        let b_extra = self.ctx.values.basic_blocks[other].extra_addresses.clone();
        if let Some(addr) = b_addr {
            self.inner_mut().extra_addresses.push(addr);
        }
        self.inner_mut().extra_addresses.extend(b_extra);
    }

    /// Associates this block with `addr` in the context address map.
    /// Names the block after `addr` if it doesn't already have a name.
    /// Returns `Err` if another value is already mapped to `addr`.
    pub fn set_address(&mut self, addr: u64) -> Result<'str, ()> {
        self.inner_mut().address = Some(addr);
        self.ctx.set_address(addr, self.id.into())?;

        if self.name().is_none() {
            self.rename(Cow::Owned(format!("{addr:x}")))?;
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
                %x = load(i64, &X);
                %y = load(i64, &Y);
                %sum = i64 %x + i64 %y;
                return [i64 0];
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let count = block.iter().count();
        assert_eq!(count, 4);

        let mut iter = block.iter();

        assert_eq!(
            iter.next().unwrap().as_statement().to_string(),
            "i64 %x = *[X]:8 X;"
        );
        assert_eq!(
            iter.next().unwrap().as_statement().to_string(),
            "i64 %y = *[Y]:8 Y;"
        );
        assert_eq!(
            iter.next().unwrap().as_statement().to_string(),
            "i64 %sum = i64 %x + i64 %y;"
        );
        assert_eq!(
            iter.next().unwrap().as_statement().to_string(),
            "return [0x0];"
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
                %x = load(i64, &X);
                %y = load(i64, &Y);
                %sum = i64 %x + i64 %y;
                return [i64 0];
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
}
