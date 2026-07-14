use crate::{
    context::Context,
    error::Result,
    value::{
        FunctionBody, Instruction, LocalValueId, ModuleView, QCodeView, Value, ValueId,
        block_param::{BlockParam, BlockParamId, BlockParamMutRef, BlockParamRef, LocalParamId},
        function::{FunctionId, FunctionMutRef, FunctionRef},
        insn::{InstructionId, InstructionRef, LocalInsnId, Mnemonic},
        util::{
            base_ref::{BaseRef, WithCtx, WithCtxMut},
            named::{Named, Renameable, update_context_name},
        host_mut::PassBacking,
        },
    },
};
use core::slice;
use jstd::graph::FxBuildHasher;
use std::{
    borrow::Cow,
    collections::HashSet,
    fmt::{Display, Formatter},
    marker::PhantomData,
};

use rustc_hash::FxHashMap as HashMap;

pub(crate) use self::cfg::EdgeData;
pub use self::cfg::{BlockId, EdgeId};
pub mod cfg;

/// Simultaneously replace operands without allowing a target-arena local id to
/// collide with a not-yet-replaced source-arena local id.
pub(crate) fn substitute_operands(mnemonic: &mut Mnemonic, pairs: &[(LocalValueId, LocalValueId)]) {
    let mut occupied: Vec<LocalValueId> = mnemonic
        .args()
        .into_iter()
        .chain(pairs.iter().map(|&(_, new)| new))
        .collect();
    let mut sentinels = Vec::with_capacity(pairs.len());
    let mut next = 0usize;
    for _ in pairs {
        let sentinel = loop {
            let candidate = LocalValueId::Varnode(crate::value::VarnodeId::from(next));
            next += 1;
            if !occupied.contains(&candidate) {
                occupied.push(candidate);
                break candidate;
            }
        };
        sentinels.push(sentinel);
    }
    for (&(old, _), &sentinel) in pairs.iter().zip(&sentinels) {
        mnemonic.replace_value(old, sentinel);
    }
    for (&(_, new), &sentinel) in pairs.iter().zip(&sentinels) {
        mnemonic.replace_value(sentinel, new);
    }
}

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
    pub params: Vec<LocalParamId>,

    /// The ids of the instructions in this block
    pub instructions: Vec<LocalInsnId>,

    /// The set of edges that this block is incident to, as bare body-local
    /// [`EdgeId`]s (see [`add_cfg_edge`](crate::context::Context::add_cfg_edge)).
    ///
    /// Strict IR locality (ruling 2) guarantees every edge incident to a block is
    /// stored in that block's own function arena, so the owning `FunctionId` is
    /// always the block's own `id.func` — it is recovered at the point of use
    /// rather than stored per edge (stage 6a, mirroring the stage-4 `EdgeId`
    /// strip).
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
}

impl<'str> BasicBlock<'str> {
    /// The ids of the instructions in this block, in order (raw `&BasicBlock`
    /// accessor). Routing target for the raw `.instructions` field reads (stage
    /// 6a §11); the field itself becomes private and localizes behind this
    /// accessor at the storage flip.
    pub fn instruction_ids(&self) -> &[LocalInsnId] {
        &self.instructions
    }

    /// The ids of this block's parameters, in declaration order (raw
    /// `&BasicBlock` accessor). Routing target for the raw `.params` field reads
    /// (stage 6a §11).
    pub fn param_ids(&self) -> &[LocalParamId] {
        &self.params
    }

    /// Gets a reference to a block from its ID
    pub fn from_id<'ctx>(ctx: &'ctx Context<'str>, id: BlockId) -> BlockRef<'str, 'ctx> {
        BlockRef::new(ModuleView::new(ctx), id)
    }

    /// Gets a mutable reference to a block from its ID
    pub fn from_id_mut<'ctx>(ctx: &'ctx mut Context<'str>, id: BlockId) -> BlockMutRef<'str, 'ctx> {
        BlockMutRef::new(ctx, id)
    }

    /// Gets a reference to a block by name. Block names are function-scoped, so
    /// this scans every function's local name table and returns the first match
    /// (names are unique within a function, not across the program). Prefer
    /// [`FunctionRef::local_named`](crate::value::FunctionRef::local_named) when
    /// the owning function is known.
    pub fn from_name<'ctx>(ctx: &'ctx Context<'str>, name: &str) -> Option<BlockRef<'str, 'ctx>> {
        ctx.functions()
            .find_map(|f| f.local_named(name))
            .and_then(ValueId::as_block)
            .map(|id| BasicBlock::from_id(ctx, id))
    }

    /// Create a new block, born into `func`'s block arena. Its `parent` is set
    /// to `func` (ownership == arena membership).
    pub fn make<'ctx>(ctx: &'ctx mut Context<'str>, func: FunctionId) -> BlockMutRef<'str, 'ctx> {
        let block = BasicBlock {
            parent: Some(func),
            ..BasicBlock::default()
        };
        let id = ctx.push_block(func, block);
        BlockMutRef::new(ctx, id)
    }

    /// Sets this block's name (crate-internal; the private `name` field is set
    /// through the generic builder, which lives in another module). The caller is
    /// responsible for registering the name in the owning function's name table.
    pub(crate) fn set_name(&mut self, name: Option<Cow<'str, str>>) {
        self.name = name;
    }

    /// The locally stored name, for owning-arena removal bookkeeping.
    pub(crate) fn local_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// A fresh, empty block value parented to `func` (crate-internal; the generic
    /// builder pushes it into `func`'s arena via the mutation host). Mirrors the
    /// literal in [`BasicBlock::make`], which can't be written outside this module
    /// because some fields are private.
    pub(crate) fn detached(func: FunctionId) -> Self {
        BasicBlock {
            parent: Some(func),
            ..BasicBlock::default()
        }
    }

    /// Structurally clone the block at `orig` into a fresh block owned by (and
    /// stored in) `target`, *without* remapping operands.
    ///
    /// This is the storage-move sibling of [`clone_into_ctx`](Self::clone_into_ctx):
    /// where `clone_into_ctx` builds a semantically independent copy inside the
    /// *same* function (used by the tracer), this reproduces `orig` verbatim in a
    /// *different* function's arenas — preserving each instruction's exact result
    /// [`TypeId`] and machine address, and the block's own name — so a caller
    /// relocating a reattributed block can then fix up the references in a single
    /// whole-function pass. It records `orig`'s params and instruction results in
    /// `value_map` (old id -> new id) but leaves the new instructions' operands and
    /// block targets pointing at the *originals*; the caller remaps them once the
    /// full map is known (so forward references between relocated blocks resolve).
    pub fn clone_block_into(
        ctx: &mut Context<'str>,
        orig: BlockId,
        target: FunctionId,
        value_map: &mut HashMap<ValueId, ValueId>,
    ) -> BlockId {
        let new_block_id = BasicBlock::make(ctx, target).id;

        // Preserve the original label, deduplicated within the target function's
        // own (function-scoped) name table.
        let name = ctx.block(orig).name.clone().unwrap_or_else(|| {
            Cow::Owned(format!("clone_{:x}", ctx.block(orig).address.unwrap_or(0)))
        });
        let unique_name = ctx.get_unique_name_in(target, name);
        BasicBlock::from_id_mut(ctx, new_block_id)
            .rename(unique_name)
            .expect("name was deduplicated");

        // Clone parameters verbatim (parent re-pointed at the new block).
        for old_param_local in ctx.block(orig).params.clone() {
            let old_param_id = BlockParamId::new(orig.func, old_param_local);
            let old_param = ctx.block_param(old_param_id).clone();
            let new_param_id = ctx.push_block_param(
                target,
                BlockParam {
                    parent: Some(new_block_id.local),
                    ..old_param
                },
            );
            BasicBlock::from_id_mut(ctx, new_block_id).push_existing_param(new_param_id);
            value_map.insert(
                ValueId::BlockParam(old_param_id),
                ValueId::BlockParam(new_param_id),
            );
        }

        // Clone instructions verbatim, preserving the exact result type and the
        // machine address. Operands are copied as-is; the caller remaps them.
        let orig_insns = ctx.block(orig).instructions.clone();
        for old_insn_local in orig_insns {
            let old_insn_id = InstructionId::new(orig.func, old_insn_local);
            let (mnemonic, type_id, address) = {
                let insn = Instruction::from_id(&*ctx, old_insn_id);
                (insn.mnemonic().clone(), insn.type_id(), insn.address())
            };
            let new_insn_id =
                InstructionRef::from_mnemonic_with_type(ctx, target, mnemonic, type_id).id;
            if let Some(addr) = address {
                ctx.instruction_mut(new_insn_id).set_address(addr);
            }
            BasicBlock::from_id_mut(ctx, new_block_id).push_insn(new_insn_id);
            value_map.insert(
                ValueId::Instruction(old_insn_id),
                ValueId::Instruction(new_insn_id),
            );
        }

        new_block_id
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

        let name = Cow::Owned(format!("clone_{:x}", ctx.block(orig).address.unwrap_or(0)));
        let unique_name = ctx.get_unique_name_in(new_block_id.func, name);
        BasicBlock::from_id_mut(ctx, new_block_id)
            .rename(unique_name)
            .expect("name was deduplicated");

        // Clone parameters
        for old_param_local in ctx.block(orig).params.clone() {
            let old_param_id = BlockParamId::new(orig.func, old_param_local);
            let old_param = ctx.block_param(old_param_id).clone();
            let new_param_id = ctx.push_block_param(
                new_block_id.func,
                BlockParam {
                    parent: Some(new_block_id.local),
                    ..old_param
                },
            );

            BasicBlock::from_id_mut(ctx, new_block_id).push_existing_param(new_param_id);
            value_map.insert(
                ValueId::BlockParam(old_param_id),
                ValueId::BlockParam(new_param_id),
            );
        }

        // Clone instructions
        let orig_insns = ctx.block(orig).instructions.clone();
        for old_insn_local in orig_insns {
            let old_insn_id = InstructionId::new(orig.func, old_insn_local);
            // Extract information from the old instruciton
            let insn_ref = Instruction::from_id(&*ctx, old_insn_id);
            let size = insn_ref.size();
            let space = insn_ref.space().map(|s| s.id);

            // Create the new instruction and derive the cloned mnemonic
            let mut new_mnemonic = insn_ref.mnemonic().clone();

            // Only remap the values this instruction actually references. This
            // avoids scanning the whole (trace-wide) `value_map` per instruction
            // and sidesteps chained `old -> new -> newer` replacements that a
            // full iteration could trigger.
            let pairs: Vec<_> = new_mnemonic
                .args()
                .into_iter()
                .filter_map(|old| {
                    // The clone still holds the source arena's bare-local operands, so
                    // qualify with `orig.func` for the map lookup and re-localize the
                    // mapped replacement against the clone's own arena.
                    let qualified = old.qualify(orig.func);
                    value_map
                        .get(&qualified)
                        .map(|&new| (old, new.localize(new_block_id.func)))
                })
                .collect();
            substitute_operands(&mut new_mnemonic, &pairs);

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
impl<'s, 'ctx: 's, 'str: 'ctx, R> BlockRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn inner(&'s self) -> &'ctx BasicBlock<'str> {
        self.view.block(self.id)
    }

    /// Iterates over outgoing `(edge_id, successor_block_id)` pairs.
    ///
    /// Routed through its [`QCodeView`] (this block's incident edge set and each edge's
    /// endpoints) rather than the `jstd` graph traits, so it reads correctly when
    /// the owning function is checked out. On the module path it yields exactly
    /// what `Node::children` did: the same incident-edge set, filtered to edges
    /// leaving this block.
    pub fn successors(&'s self) -> impl Iterator<Item = (EdgeId, BlockId)> + 's {
        let view = self.view;
        let id = self.id;
        self.inner().edges.iter().copied().filter_map(move |edge| {
            let e = view.edge(id.func, edge);
            (e.from == id).then_some((edge, e.to))
        })
    }

    /// Iterates over incoming `(edge_id, predecessor_block_id)` pairs. See
    /// [`successors`](Self::successors) for the routing rationale.
    pub fn predecessors(&'s self) -> impl Iterator<Item = (EdgeId, BlockId)> + 's {
        let view = self.view;
        let id = self.id;
        self.inner().edges.iter().copied().filter_map(move |edge| {
            let e = view.edge(id.func, edge);
            (e.to == id).then_some((edge, e.from))
        })
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
    pub fn params(&'s self) -> impl Iterator<Item = BlockParamRef<'str, 'ctx, R>> + 's {
        let func = self.id.func;
        self.inner()
            .params
            .iter()
            .map(move |&local| BlockParamRef::new(self.view, BlockParamId::new(func, local)))
    }

    /// Returns the number of parameters declared on this block.
    pub fn num_params(&'s self) -> usize {
        self.inner().params.len()
    }

    /// Iterates over the instructions in this block
    pub fn instructions(&'s self) -> InstructionIter<'str, 'ctx, R> {
        let inner = self.inner();
        InstructionIter {
            view: self.view,
            func: self.id.func,
            inner: inner.instructions.iter(),
            marker: PhantomData,
        }
    }

    /// Iterates over the instructions in this block
    /// alias for `instructions()`
    pub fn iter(&'s self) -> InstructionIter<'str, 'ctx, R> {
        self.instructions()
    }

    pub fn instruction_ids(&'s self) -> Vec<InstructionId> {
        let func = self.id.func;
        self.inner()
            .instructions
            .iter()
            .map(|&local| InstructionId::new(func, local))
            .collect()
    }

    /// Does this block have any instructions?
    pub fn is_empty(&'s self) -> bool {
        self.inner().instructions.is_empty()
    }

    /// Does this block finish with a terminator instruction?
    pub fn is_terminated(&'s self) -> bool {
        self.iter().last().is_some_and(|insn| insn.is_terminator())
    }

    pub fn parent(&'s self) -> Option<FunctionRef<'str, 'ctx, R>> {
        self.inner()
            .parent
            .map(|fid| FunctionRef::new(self.view, fid))
    }

    pub fn function(&'s self) -> Option<FunctionRef<'str, 'ctx, R>> {
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
                .map(|(_, b)| BlockRef::new(self.view, b).name().unwrap_or("unnamed"))
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

#[derive(Clone, Copy)]
pub struct BlockRef<'str, 'ctx, R = ModuleView<'ctx, 'str>> {
    pub id: BlockId,
    pub(in crate::value) view: R,
    marker: PhantomData<&'ctx &'str ()>,
}

impl<'str, 'ctx, R> BlockRef<'str, 'ctx, R> {
    pub fn new(view: R, id: BlockId) -> Self {
        Self {
            id,
            view,
            marker: PhantomData,
        }
    }

    pub fn id(&self) -> ValueId {
        self.id.into()
    }
}

impl<'str, 'ctx> BlockRef<'str, 'ctx> {
    pub fn from_id(ctx: &'ctx Context<'str>, id: BlockId) -> Self {
        Self::new(ModuleView::new(ctx), id)
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 'ctx, 'str> for BlockRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'ctx Context<'str> {
        // Module-scope-only escape hatch: shared-only reads go through
        // `host().shr()`; only whole-module walks (callees/callers) reach here,
        // and those panic on a checked-out host by design (context-split Pin B).
        self.view.context()
    }
}

impl<'str: 'ctx, 'ctx, R> Named for BlockRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn name(&self) -> Option<&str> {
        self.view.block(self.id).name.as_deref()
    }
}

impl<'str: 'ctx, 'ctx, R> Display for BlockRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        BlockRef::fmt(self, f)
    }
}

impl<'str: 'ctx, 'ctx, R> Value<'str, 'ctx> for BlockRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    fn id(&self) -> ValueId {
        self.id()
    }

    fn size(&self) -> usize {
        0
    }
}

pub struct InstructionIter<'str, 'ctx, R = ModuleView<'ctx, 'str>> {
    view: R,
    func: FunctionId,
    inner: slice::Iter<'ctx, LocalInsnId>,
    marker: PhantomData<&'str ()>,
}

impl<'str: 'ctx, 'ctx, R> Iterator for InstructionIter<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    type Item = InstructionRef<'str, 'ctx, R>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|&local| InstructionRef::new(self.view, InstructionId::new(self.func, local)))
    }
}

impl<'str: 'ctx, 'ctx, R> IntoIterator for &BlockRef<'str, 'ctx, R>
where
    R: QCodeView<'ctx, 'str>,
{
    type Item = InstructionRef<'str, 'ctx, R>;
    type IntoIter = InstructionIter<'str, 'ctx, R>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub type BlockMutRef<'str, 'ctx> = BaseRef<&'ctx mut Context<'str>, BlockId>;

impl<'str> BaseRef<&mut Context<'str>, BlockId> {
    pub fn is_terminated(&self) -> bool {
        self.ctx
            .block_ref(self.id)
            .iter()
            .last()
            .is_some_and(|insn| insn.is_terminator())
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtxMut<'s, 'str> for BlockMutRef<'str, 'ctx> {
    fn ctx_mut(&'s mut self) -> &'s mut Context<'str> {
        self.ctx
    }
}

// Read access over a mutation host: shared reads via the host's shared context,
// the read view via its static provider. Two concrete backings — `&mut Context`
// (module) and `PassBacking` (checked-out function pass) — each routing through
// the backing's inherent `shared`/`view`.
impl<'s, 'str> WithCtx<'s, 's, 'str> for BaseRef<&mut Context<'str>, BlockId>
where
    'str: 's,
{
    fn ctx(&'s self) -> &'s Context<'str> {
        self.ctx.shared()
    }
}
impl<'s, 'a, 'str> WithCtx<'s, 's, 'str> for BaseRef<PassBacking<'a, 'str>, BlockId>
where
    'str: 's,
{
    fn ctx(&'s self) -> &'s Context<'str> {
        // A checked-out pass backing carries no `&Context`; shared-only reads go
        // through the host's `shr()`. Nothing on the pass path reaches this.
        panic!("whole-context read on a checked-out mutation ref: module-scope only")
    }
}

impl Named for BlockMutRef<'_, '_> {
    fn name(&self) -> Option<&str> {
        self.ctx.block(self.id).name.as_deref()
    }
}

impl<'str, 'ctx> Renameable<'str, 'ctx> for BlockMutRef<'str, 'ctx> {
    fn rename(&mut self, name: Cow<'str, str>) -> Result<()> {
        let id = self.id.into();
        let old_name = self.ctx.block(self.id).name.as_deref().map(str::to_owned);
        update_context_name(id, self.ctx, name.clone(), old_name.as_deref())?;
        self.ctx.block_mut(self.id).name = Some(name);
        Ok(())
    }
}

// Naming/renaming a block through a checked-out host (concrete: a generic
// a fully generic backing can't prove `'str` outlives the returned `&str`). Block names are
// function-local, so this reads/writes the owned function's arena directly.
impl<'a, 'str> Named for BaseRef<PassBacking<'a, 'str>, BlockId> {
    fn name(&self) -> Option<&str> {
        self.ctx.fun.blocks[self.id.local].name.as_deref()
    }
}

impl<'a, 'str> Renameable<'str, 'a> for BaseRef<PassBacking<'a, 'str>, BlockId> {
    fn rename(&mut self, name: Cow<'str, str>) -> Result<()> {
        let id = self.id.into();
        let old_name = self.ctx.fun.blocks[self.id.local]
            .name
            .as_deref()
            .map(str::to_owned);
        self.ctx
            .register_local_name(id, name.clone(), old_name.as_deref())?;
        self.ctx.block_mut(self.id).name = Some(name);
        Ok(())
    }
}

// The own-block mutation verbs, emitted for each concrete mutation backing —
// `&mut Context` (module) and `PassBacking` (checked-out function pass). Both
// bodies are identical (they call the backing's inherent verbs); the macro keeps
// the pair in lockstep without a shared trait bound.
macro_rules! impl_block_mut_verbs {
    (<$($l:lifetime),*> $ctx:ty) => {
        impl<$($l),*> BaseRef<$ctx, BlockId> {
    /// Sets (or clears) this block's comment. Own-block edit, host-routed.
    pub fn set_comment(&mut self, comment: Option<String>) {
        self.ctx.block_mut(self.id).comment = comment;
    }

    /// Sets this block's name and registers it in the owning function's local name
    /// table (own-block edit, backing-routed). Mirrors the `Renameable` impls for the
    /// concrete module / pass block refs, but works over either backing, so
    /// a `FunctionPass` can name the blocks it mints. Returns an error only on a
    /// duplicate name.
    pub fn rename_local(&mut self, name: Cow<'str, str>) -> crate::error::Result<()> {
        let old_name = self
            .ctx
            .view()
            .block(self.id)
            .name
            .as_deref()
            .map(str::to_owned);
        self.ctx
            .register_local_name(self.id.into(), name.clone(), old_name.as_deref())?;
        self.ctx.block_mut(self.id).name = Some(name);
        Ok(())
    }

    fn insert_insn(&mut self, index: usize, insn_id: InstructionId) {
        self.ctx.instruction_mut(insn_id).parent = Some(self.id.local);
        self.ctx
            .block_mut(self.id)
            .instructions
            .insert(index, insn_id.localize(self.id.func));
    }

    /// Inserts an instruction at the given index, shifting later instructions
    /// right. Panics if `index > len`.
    pub fn insert_insn_at_index(&mut self, index: usize, insn_id: InstructionId) {
        self.insert_insn(index, insn_id);
    }

    /// Pushes an instruction to the end of this block.
    pub fn push_insn(&mut self, id: InstructionId) {
        let len = self.ctx.function(self.id.func).blocks[self.id.local]
            .instructions
            .len();
        self.insert_insn(len, id);
    }

    /// Inserts `insn_id` immediately before `before_id`. Panics if `before_id` is
    /// not in this block. Delegates to the backing's `insert_insn_before` verb.
    pub fn insert_insn_before(&mut self, before_id: InstructionId, insn_id: InstructionId) {
        let id = self.id;
        self.ctx.insert_insn_before(id, before_id, insn_id);
    }

    /// Removes this block from its function, including its payload. Delegates
    /// to the backing's `delete_block` verb.
    pub fn delete(&mut self, function_id: FunctionId) {
        let id = self.id;
        self.ctx.delete_block(id, function_id);
    }

    /// Absorbs `other` into this block. Delegates to the backing's `absorb_block` verb;
    /// `edge_ab` must be the direct edge from this block to `other`.
    pub fn absorb_block(&mut self, other: BlockId, edge_ab: EdgeId, function_id: FunctionId) {
        let id = self.id;
        self.ctx.absorb_block(id, other, edge_ab, function_id);
    }
        }
    };
}

impl_block_mut_verbs!(<'c, 'str> &'c mut Context<'str>);
impl_block_mut_verbs!(<'a, 'str> PassBacking<'a, 'str>);

impl Display for BlockMutRef<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.as_ref().fmt(f)
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
    fn inner(&self) -> &BasicBlock<'str> {
        self.ctx.block(self.id)
    }

    pub fn num_params(&self) -> usize {
        self.as_ref().num_params()
    }

    pub fn instruction_ids(&self) -> Vec<InstructionId> {
        self.as_ref().instruction_ids()
    }

    pub fn instructions(&self) -> impl Iterator<Item = InstructionRef<'str, '_>> {
        self.as_ref().instructions().collect::<Vec<_>>().into_iter()
    }

    pub fn address(&self) -> Option<u64> {
        self.as_ref().address()
    }

    pub fn successors(&self) -> impl Iterator<Item = (EdgeId, BlockId)> {
        self.as_ref().successors().collect::<Vec<_>>().into_iter()
    }

    /// Builder-style address assignment. Panics if `addr` is already mapped.
    /// Use [`set_address`](Self::set_address) for fallible assignment.
    pub fn with_address(mut self, addr: u64) -> Self {
        self.set_address(addr)
            .expect("address is already mapped to a value");
        self
    }

    /// Indexed construction variant of [`with_address`](Self::with_address).
    pub fn with_address_indexed(
        mut self,
        addresses: &mut crate::address_index::AddressIndex,
        addr: u64,
    ) -> Self {
        self.set_address_indexed(addresses, addr)
            .expect("address is already mapped to a value");
        self
    }

    #[allow(unused_mut)]
    pub fn in_function(mut self, fun_id: FunctionId) -> Self {
        FunctionBody::from_id_mut(self.ctx, fun_id).add_block(self.id);
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
        self.ctx.block_mut(self.id)
    }

    pub fn parent_mut(&mut self) -> Option<FunctionMutRef<'str, '_>> {
        self.ctx
            .block(self.id)
            .parent
            .map(|fid| FunctionBody::from_id_mut(self.ctx, fid))
    }

    pub fn as_ref(&self) -> BlockRef<'str, '_> {
        BlockRef::new(ModuleView::new(self.ctx), self.id)
    }

    /// Declares a new parameter on this block with the given size in bytes.
    ///
    /// The parameter is appended to the block's `params` list and its `parent`
    /// is set to this block. It does NOT appear in `instructions`.
    /// Returns a mutable reference whose `ValueId` can be used as an operand.
    pub fn push_param(&mut self, size: usize) -> BlockParamMutRef<'str, '_> {
        let block_id = self.id;
        let index = self.inner().params.len();
        let type_id = self.ctx.shared.types.get_or_make_int(size);
        let id = self.ctx.push_block_param(
            block_id.func,
            BlockParam {
                index,
                type_id,
                parent: Some(block_id.local),
                name: None,
                origin: None,
                protected: false,
            },
        );
        self.inner_mut().params.push(id.localize(block_id.func));
        BlockParamMutRef::from_id(self.ctx, id)
    }

    /// Appends an already-created block parameter to the parameters list
    pub fn push_existing_param(&mut self, id: BlockParamId) {
        let func = self.id.func;
        self.inner_mut().params.push(id.localize(func));
    }

    /// Inserts an instruction after the instruction identified by `after_id` in this block.
    /// Panics if `after_id` is not an instruction in this block.
    pub fn insert_insn_after(&mut self, after_id: InstructionId, insn_id: InstructionId) {
        let index = self
            .inner()
            .instructions
            .iter()
            .position(|&local| InstructionId::new(self.id.func, local) == after_id)
            .expect("after_id not found in block");
        self.insert_insn(index + 1, insn_id);
    }

    /// Retains only the instructions for which `f` returns true, deleting the
    /// removed instructions.
    pub fn retain_insns(&mut self, mut f: impl FnMut(&InstructionId) -> bool) {
        let func = self.id.func;
        let mut removed = Vec::new();
        self.inner_mut().instructions.retain(|&local| {
            let id = InstructionId::new(func, local);
            if f(&id) {
                true
            } else {
                removed.push(id);
                false
            }
        });
        for id in removed {
            self.ctx.remove_instruction(id);
        }
    }

    /// Adds an edge (body-local id) to this block's edge set.
    /// DO NOT USE THIS
    pub(crate) fn add_edge(&mut self, edge_id: EdgeId) {
        self.inner_mut().edges.insert(edge_id);
    }

    /// Removes an edge (body-local id) from this block's edge set.
    /// DO NOT USE THIS
    pub(crate) fn remove_edge(&mut self, edge_id: EdgeId) {
        self.inner_mut().edges.remove(&edge_id);
    }

    /// Removes the last instruction from this block.
    pub fn pop_insn(&mut self) {
        if let Some(&local) = self.inner().instructions.last() {
            self.ctx
                .remove_instruction(InstructionId::new(self.id.func, local));
        }
    }

    /// Appends a slice of instruction ids to this block.
    pub fn extend_insns(&mut self, insns: &[InstructionId]) {
        let func = self.id.func;
        self.inner_mut()
            .instructions
            .extend(insns.iter().map(|&id| id.localize(func)));
    }

    /// Associates this block with `addr` in the context address map.
    /// Names the block after `addr` if it doesn't already have a name.
    /// Returns `Err` if another value is already mapped to `addr`.
    pub fn set_address(&mut self, addr: u64) -> Result<()> {
        let mut addresses = crate::address_index::AddressIndex::analyze(&*self.ctx);
        self.set_address_indexed(&mut addresses, addr)
    }

    /// Assigns an address through a caller-owned construction index.
    pub fn set_address_indexed(
        &mut self,
        addresses: &mut crate::address_index::AddressIndex,
        addr: u64,
    ) -> Result<()> {
        let old_address = self.inner().address;
        self.inner_mut().address = Some(addr);
        if let Err(error) = self
            .ctx
            .set_address_indexed(addresses, addr, self.id.into())
        {
            self.inner_mut().address = old_address;
            return Err(error);
        }

        if self.name().is_none() {
            let label = self
                .ctx
                .get_unique_name_in(self.id.func, Cow::Owned(format!("{addr:x}")));
            self.rename(label)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::value::insn::{Binary, Binop, IntBinop, LocalInsnId};
    use qcode_macro::qcode;

    #[test]
    fn simultaneous_operand_substitution_does_not_chain_local_ids() {
        let a = LocalValueId::Instruction(LocalInsnId::from(1));
        let b = LocalValueId::Instruction(LocalInsnId::from(2));
        let c = LocalValueId::Instruction(LocalInsnId::from(3));
        let mut mnemonic = Mnemonic::Binop(Binary {
            op: Binop::Int(IntBinop::Add),
            lhs: a,
            rhs: b,
        });

        substitute_operands(&mut mnemonic, &[(a, b), (b, c)]);

        let Mnemonic::Binop(binary) = mnemonic else {
            unreachable!();
        };
        assert_eq!((binary.lhs, binary.rhs), (b, c));
    }

    #[test]
    fn test_create_block_at_address() {
        let mut ctx = Context::new();
        let id = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .with_address(0x2000)
        .id;
        let addresses = crate::address_index::AddressIndex::analyze(&ctx);
        let block_by_addr = BasicBlock::from_id(
            &ctx,
            addresses
                .block_at(0x2000)
                .expect("block not found by address"),
        );
        assert_eq!(id, block_by_addr.id);
        assert_eq!(block_by_addr.address(), Some(0x2000));
    }

    #[test]
    #[should_panic(expected = "address is already mapped to a value")]
    fn test_create_block_at_duplicate_address() {
        let mut ctx = Context::new();
        {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .with_address(0x2000);
        {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .with_address(0x2000);
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
            .successors()
            .map(|(_, b)| {
                BasicBlock::from_id(&ctx, b)
                    .name()
                    .unwrap_or("")
                    .to_string()
            })
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
        let mut block = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        };

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
        let mut block = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        };

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
        let block_id = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .id;

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
        for (orig_id, clone_id) in orig
            .instruction_ids()
            .into_iter()
            .zip(cloned.instruction_ids())
        {
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
        for arg in cloned.iter().flat_map(|i| i.operands()) {
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
        assert!(
            !ctx.contains_block(b),
            "deleted block payload must be absent"
        );
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

        assert!(
            !ctx.contains_block(loop_hdr),
            "deleted self-loop block payload must be absent"
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
            !ctx.contains_instruction(y),
            "deleted instruction payload must be physically absent"
        );
    }
}
