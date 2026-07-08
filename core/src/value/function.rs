use jstd::Identifier;
use rustc_hash::FxHashMap;
use std::{
    borrow::Cow,
    collections::BTreeSet,
    fmt::{Display, Formatter},
};

mod signature;
pub use signature::{FunctionSignature, ParamAttrs};

use jstd::registry::Registry;

use crate::{
    context::Context,
    error::{Error, ErrorTy, Result},
    value::{
        BasicBlock, BlockId, BlockRef, Instruction, InstructionId, Value, ValueId, Varnode,
        VarnodeId,
        block::EdgeData,
        block::cfg::{LocalBlockId, LocalEdgeId},
        block_param::{BlockParam, LocalParamId},
        insn::{Branch, LocalInsnId, Mnemonic},
        util::{
            base_ref::{BaseRef, WithCtx, WithCtxMut},
            named::{Named, Renameable, update_context_name},
        },
    },
};

#[derive(Identifier)]
pub struct FunctionId(u32);

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Function<'str> {
    /// The function's name.
    pub name: Cow<'str, str>,

    /// Optional entry address (from binary).
    pub address: Option<u64>,

    /// The entry block (dominates all other blocks in this function).
    pub root: Option<BlockId>,

    /// Instruction storage for this function. Function-scoped: the composite
    /// [`InstructionId`](crate::value::InstructionId) `{ func, local }` indexes
    /// here via `local`. Append-only with tombstones; never compacted.
    pub(crate) insns: Registry<LocalInsnId, Instruction<'str>>,

    /// Basic-block *storage* for this function. A block is born here and keeps
    /// its `id.func` for the life of the program (its `LocalBlockId` indexes
    /// this arena). Storage is decoupled from *ownership*: lifting may attribute
    /// a block to a different function than the one it was born in (see
    /// `reattribute_blocks`), so a block stored here can be owned elsewhere. Use
    /// [`FunctionRef::blocks`] (which reads [`roster`](Self::roster)) to iterate
    /// the blocks this function owns, not this arena directly.
    pub(crate) blocks: Registry<LocalBlockId, BasicBlock<'str>>,

    /// Ownership roster: the composite ids of the blocks this function owns, in
    /// order. Usually all live in this function's own `blocks` arena, but a
    /// reattributed block may be stored in another function's arena. Kept in
    /// sync with each block's `parent` by `add_block`/`remove_block`. Tombstoned
    /// blocks are filtered out on read.
    #[serde(default)]
    pub(crate) roster: Vec<BlockId>,

    /// Block-parameter storage for this function.
    pub(crate) params: Registry<LocalParamId, BlockParam<'str>>,

    /// CFG-edge storage for this function.
    pub(crate) edges: Registry<LocalEdgeId, EdgeData>,

    /// Addresses of every machine instruction lifted into this function, in
    /// ascending order. Recorded during recursive disassembly and preserved
    /// across optimization (which merges blocks and rewrites the IR), so the
    /// raw disassembly view can be reconstructed regardless of CFG changes.
    pub instruction_addrs: BTreeSet<u64>,

    /// Whether this is an external (imported) function.
    ///
    /// External functions have no lifted body — they are stubs for calls that
    /// go outside the binary (e.g. PLT thunks for shared-library functions).
    /// The recursive disassembler will not attempt to lift their body.
    pub is_external: bool,

    /// Optional ABI description used by alias analysis.
    pub signature: Option<FunctionSignature>,

    /// What semantic class this function belongs to.
    #[serde(default)]
    pub kind: FunctionKind,

    /// Function-local name table for this function's block, instruction, and
    /// block-param names (ruling 1 of the parallel-passes plan). Keeping these
    /// out of the global [`name_map`](crate::context::Context) lets two functions
    /// name values independently — a prerequisite for parallel function passes.
    /// A value's own `name` field is the source of truth for rendering; this only
    /// enforces uniqueness and resolves names within the function.
    #[serde(default)]
    pub(crate) names: crate::context::NameTable<'str>,

    /// Reverse use-def map, scoped to this function: for each [`ValueId`] the
    /// list of *this function's* instructions that use it as an operand. By the
    /// SSA ownership invariant every user of an instruction/param value is
    /// intra-function, so an SSA def's users all live here. Shared values
    /// (literals, varnodes) may be used by many functions; each records only its
    /// own uses, which is all any pass needs (no pass queries a shared value's
    /// users program-wide). Kept in sync by
    /// [`push_insn`](crate::value::registry::ValueRegistry::push_insn),
    /// [`remove_instructions`](crate::value::registry::ValueRegistry::remove_instructions),
    /// [`Context::replace_all_uses_with`](crate::context::Context::replace_all_uses_with),
    /// and [`Context::replace_instruction_mnemonic`](crate::context::Context::replace_instruction_mnemonic).
    #[serde(default)]
    pub(crate) users: FxHashMap<ValueId, Vec<InstructionId>>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FunctionKind {
    #[default]
    Machine,
    Lambda,
}

impl<'str> Function<'str> {
    fn new(name: Cow<'str, str>) -> Self {
        Self {
            name,
            address: None,
            root: None,
            insns: Registry::default(),
            blocks: Registry::default(),
            roster: Vec::new(),
            params: Registry::default(),
            edges: Registry::default(),
            instruction_addrs: BTreeSet::new(),
            is_external: false,
            signature: None,
            kind: FunctionKind::Machine,
            names: crate::context::NameTable::default(),
            users: FxHashMap::default(),
        }
    }

    /// This function's instructions that use `value` as an operand (see
    /// [`users`](Self::users)). Empty for a value this function never uses.
    pub fn users_of(&self, value: ValueId) -> &[InstructionId] {
        self.users.get(&value).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Iterate this function's recorded `(value, users)` reverse-use entries.
    /// Read-only; used by the users-map consistency verifier.
    pub fn user_map_entries(&self) -> impl Iterator<Item = (ValueId, &[InstructionId])> {
        self.users.iter().map(|(v, u)| (*v, u.as_slice()))
    }

    /// Gets a reference to a function from its ID
    pub fn from_id<'ctx>(ctx: &'ctx Context<'str>, id: FunctionId) -> FunctionRef<'str, 'ctx> {
        FunctionRef { ctx, id }
    }

    /// Gets a mutable reference to a function from its ID
    pub fn from_id_mut<'ctx>(
        ctx: &'ctx mut Context<'str>,
        id: FunctionId,
    ) -> FunctionMutRef<'str, 'ctx> {
        FunctionMutRef::new(ctx, id)
    }

    /// Gets a reference to a function by name
    pub fn from_name<'ctx>(
        ctx: &'ctx Context<'str>,
        name: &str,
    ) -> Option<FunctionRef<'str, 'ctx>> {
        ctx.get_named(name)
            .and_then(ValueId::as_function)
            .map(|id| FunctionRef { ctx, id })
    }

    /// Gets a reference to a function by address
    pub fn from_addr<'ctx>(ctx: &'ctx Context<'str>, addr: u64) -> Option<FunctionRef<'str, 'ctx>> {
        ctx.get_at_addr(&addr)
            .and_then(ValueId::as_function)
            .map(|id| FunctionRef { ctx, id })
    }

    /// Gets a mutable reference to a function by address
    pub fn from_addr_mut<'ctx>(
        ctx: &'ctx mut Context<'str>,
        addr: u64,
    ) -> Option<FunctionMutRef<'str, 'ctx>> {
        ctx.get_at_addr(&addr)
            .and_then(ValueId::as_function)
            .map(|id| FunctionMutRef::new(ctx, id))
    }

    /// Create a new function
    pub fn make<'ctx>(
        ctx: &'ctx mut Context<'str>,
        name: Cow<'str, str>,
    ) -> Result<FunctionMutRef<'str, 'ctx>> {
        let id = ctx.values.push_function(Function::new(name.clone()));
        ctx.update_name(name, id.into(), None)?;
        Ok(Self::from_id_mut(ctx, id))
    }

    /// Create a new pure value-level lambda function.
    pub fn make_lambda<'ctx>(
        ctx: &'ctx mut Context<'str>,
        name: Cow<'str, str>,
    ) -> Result<FunctionMutRef<'str, 'ctx>> {
        let mut function = Self::make(ctx, name)?;
        function.inner_mut().kind = FunctionKind::Lambda;
        function.set_is_pure(true);
        function.set_pure_reg(true);
        Ok(function)
    }

    /// Create a new function at a given address, generating a name if necessary.
    pub fn make_at_addr<'ctx>(
        ctx: &'ctx mut Context<'str>,
        address: u64,
        name: Option<Cow<'str, str>>,
    ) -> FunctionMutRef<'str, 'ctx> {
        let name = match name {
            Some(name) => name,
            None => Cow::Owned(format!("fn_{address:x}")),
        };

        let id = ctx.values.push_function(Function::new(name.clone()));

        Self::from_id_mut(ctx, id)
            .with_name(name)
            .expect("Function name is not unique")
            .with_address(address)
            .expect("Function address is not unique")
    }

    /// Like [`Function::make_at_addr`] but marks the result as external.
    ///
    /// External functions have no lifted body; the recursive disassembler will
    /// not try to explore them.
    pub fn make_external<'ctx>(
        ctx: &'ctx mut Context<'str>,
        address: u64,
        name: Option<Cow<'str, str>>,
    ) -> FunctionMutRef<'str, 'ctx> {
        let mut f = Self::make_at_addr(ctx, address, name);
        f.inner_mut().is_external = true;
        f
    }

    /// Returns the [`FunctionId`] for `addr`, creating a named stub if absent.
    pub fn from_addr_or_create<'ctx>(
        ctx: &'ctx mut Context<'str>,
        address: u64,
    ) -> FunctionMutRef<'str, 'ctx> {
        match ctx.get_at_addr(&address).and_then(ValueId::as_function) {
            Some(id) => Self::from_id_mut(ctx, id),
            None => Self::make_at_addr(ctx, address, None),
        }
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx, Ctx> BaseRef<Ctx, FunctionId>
where
    Self: WithCtx<'s, 'ctx, 'str>,
{
    fn inner(&'s self) -> &'ctx Function<'str> {
        &self.ctx().values.functions[self.id]
    }

    fn size(&self) -> usize {
        0
    }

    /// The `address` of the inner `Function`.
    pub fn address(&'s self) -> Option<u64> {
        self.inner().address
    }

    /// Whether the inner `Function` is external.
    pub fn is_external(&'s self) -> bool {
        self.inner().is_external
    }

    /// A reference to the signature of the inner `Function`, if any.
    pub fn signature(&'s self) -> Option<&'ctx FunctionSignature> {
        self.inner().signature.as_ref()
    }

    /// This function's instructions that use `value` as an operand. See
    /// [`Function::users_of`]; this is the function-scoped read every pass wants
    /// for an SSA value (all its users are intra-function).
    pub fn users_of(&'s self, value: ValueId) -> &'ctx [InstructionId] {
        self.inner().users_of(value)
    }

    /// Iterate this function's recorded `(value, users)` reverse-use entries
    /// (see [`Function::user_map_entries`]).
    pub fn user_map_entries(&'s self) -> impl Iterator<Item = (ValueId, &'ctx [InstructionId])> {
        self.inner().user_map_entries()
    }

    /// Resolve a block/instruction/param `name` within this function's local name
    /// table (see [`Function::names`]). `None` if this function has no such name.
    pub fn local_named(&'s self, name: &str) -> Option<ValueId> {
        self.inner().names.get(name)
    }

    /// Whether this function's full register effect is captured by its call
    /// interface: it reads no registers (only its explicit args) and writes
    /// exactly its [`clobbered_regs`](Self::clobbered_regs). See
    /// [`FunctionSignature::externally_resolved`].
    pub fn is_externally_resolved(&'s self) -> bool {
        self.inner()
            .signature
            .as_ref()
            .is_some_and(|s| s.externally_resolved)
    }

    /// The inferred pointer attributes for positional argument `index`, or `None`
    /// when this function has no analyzed attributes (treat conservatively: the
    /// argument escapes and may be written through). See
    /// [`FunctionSignature::param_attrs`].
    pub fn param_attr(&'s self, index: usize) -> Option<ParamAttrs> {
        self.inner()
            .signature
            .as_ref()
            .and_then(|s| s.param_attrs.as_ref())
            .and_then(|attrs| attrs.get(index))
            .copied()
    }

    /// The full per-parameter attribute vector, if analyzed.
    pub fn param_attrs(&'s self) -> Option<&'ctx [ParamAttrs]> {
        self.inner()
            .signature
            .as_ref()
            .and_then(|s| s.param_attrs.as_deref())
    }

    /// Registers concretely written by this function, as set by analysis.
    pub fn clobbered_regs(&'s self) -> Option<&'ctx [VarnodeId]> {
        self.inner()
            .signature
            .as_ref()
            .and_then(|s| s.clobbered.as_deref())
    }

    /// The non-register memory spaces this function may (transitively) write, as
    /// set by analysis. `Some(spaces)` is exact (a space not listed is never
    /// written); `None` means unknown/unbounded. See
    /// [`FunctionSignature::written_spaces`].
    pub fn written_spaces(&'s self) -> Option<&'ctx [crate::space::SpaceId]> {
        self.inner()
            .signature
            .as_ref()
            .and_then(|s| s.written_spaces.as_deref())
    }

    /// Whether `argpromote_registers` has functionalized this function's register
    /// side effects into a pure value function. See
    /// [`FunctionSignature::pure_reg`].
    pub fn is_pure_reg(&'s self) -> bool {
        self.inner().signature.as_ref().is_some_and(|s| s.pure_reg)
    }

    /// Whether argpromote has functionalized *every* side-effect channel of this
    /// function — it is a deterministic pure function of its by-value params,
    /// touching no caller-visible memory or registers. Strictly stronger than
    /// [`is_pure_reg`](Self::is_pure_reg). See [`FunctionSignature::is_pure`].
    pub fn is_pure(&'s self) -> bool {
        self.inner().signature.as_ref().is_some_and(|s| s.is_pure)
    }

    /// Whether this is a pure value-level lambda rather than a machine function.
    pub fn is_lambda(&'s self) -> bool {
        self.inner().kind == FunctionKind::Lambda
    }

    pub fn kind(&'s self) -> FunctionKind {
        self.inner().kind
    }

    /// Registers read before written (function inputs), as inferred by analysis.
    ///
    /// Legacy ABI input-register list. It is **`None` for `pure_reg` functions**
    /// (argpromote never populates it); their by-value root block params are the
    /// source of truth for the call interface, so prefer the params (e.g.
    /// [`input_arg_name`](Self::input_arg_name)). Retained only for the
    /// conventional/external calling-convention path (`summaries`).
    #[deprecated(
        note = "legacy ABI register list; None for pure_reg functions. Use the root block params \
                as the call interface; this remains only for the conventional/external path."
    )]
    pub fn input_regs(&'s self) -> Option<&'ctx [VarnodeId]> {
        self.inner()
            .signature
            .as_ref()
            .and_then(|s| s.inputs.as_deref())
    }

    /// The display name for the call-site argument bound to input `index`: the
    /// register name for a register input, or a synthesized `stack_<addr>` slot
    /// name for a stack-passed input (whose varnode is a nameless stack-space
    /// offset carrier). Mirrors mem2reg's `block_param_name_for_var` so a call
    /// argument reads with the same name as the callee's promoted stack
    /// parameter. `None` when there is no input at `index`.
    pub fn input_arg_name(&'s self, index: usize) -> Option<String> {
        // The root block param at `index` is the interface element a call
        // argument actually binds to, named after its register by
        // `argpromote_registers` or `stack_<addr>` by mem2reg's
        // `block_param_name_for_var`. Prefer it: it is the source of truth and is
        // populated even for `pure_reg` functions, whose ABI register list
        // (`input_regs`) is never filled in.
        if let Some(root) = self.root()
            && let Some(name) = root
                .params()
                .nth(index)
                .and_then(|p| p.name().map(str::to_owned))
        {
            return Some(name);
        }

        // Explicit per-argument names recorded by name-derived passes (e.g. an
        // external callee's C-prototype parameters, plus a synthesized
        // `return_address` slot). The source of truth for bodyless externals,
        // which have neither a root block nor an inferred input-register list.
        if let Some(name) = self
            .inner()
            .signature
            .as_ref()
            .and_then(|s| s.input_names.as_ref())
            .and_then(|names| names.get(index))
            .and_then(|n| n.as_deref())
        {
            return Some(name.to_owned());
        }

        // Fall back to the inferred input-register list: a register name, or a
        // synthesized `stack_<addr>` slot name for a stack-passed input.
        // Intentional use of the legacy list — only reached when the param has no
        // name (conventional functions, never `pure_reg`).
        #[allow(deprecated)]
        let input = self.input_regs()?.get(index).copied()?;
        let vn = Varnode::from_id(self.ctx(), input);
        if let Some(name) = vn.name() {
            return Some(name.to_owned());
        }
        let space = vn.space();
        if space.name.as_deref() == Some("stack") {
            return Some(format!("stack_{:x}", vn.address() as u64));
        }
        None
    }

    /// Registers saved and restored unchanged (preserved across calls), as
    /// inferred by analysis.
    pub fn saved_regs(&'s self) -> Option<&'ctx [VarnodeId]> {
        self.inner()
            .signature
            .as_ref()
            .and_then(|s| s.saved.as_deref())
    }

    /// The net change this function applies to the stack pointer between entry
    /// and return, as inferred by analysis. See [`FunctionSignature::stack_delta`].
    pub fn stack_delta(&'s self) -> Option<i64> {
        self.inner().signature.as_ref().and_then(|s| s.stack_delta)
    }

    /// Whether this function performs an unresolved/dynamic stack read (or
    /// forwards a stack pointer into one). See
    /// [`FunctionSignature::reads_unbounded_stack`].
    pub fn reads_unbounded_stack(&'s self) -> bool {
        self.inner()
            .signature
            .as_ref()
            .is_some_and(|s| s.reads_unbounded_stack)
    }

    /// Whether this function hands a pointer into its own frame to a callee that
    /// may read it unboundedly. See
    /// [`FunctionSignature::frame_escapes_to_unbounded`].
    pub fn frame_escapes_to_unbounded(&'s self) -> bool {
        self.inner()
            .signature
            .as_ref()
            .is_some_and(|s| s.frame_escapes_to_unbounded)
    }

    /// The name of the inner `Function`.
    pub fn name(&'s self) -> &'ctx str {
        self.inner().name.as_ref()
    }

    /// The addresses of every machine instruction lifted into this function, in
    /// ascending order. Unlike [`blocks`](Self::blocks), this is stable across
    /// optimization, so it drives the raw disassembly view.
    pub fn instruction_addrs(&'s self) -> impl Iterator<Item = u64> + 'ctx {
        self.inner().instruction_addrs.iter().copied()
    }

    /// The functions this function directly calls, deduplicated and ordered by
    /// id. Derived from the IR on demand — like [`BlockRef::successors`] reading
    /// the CFG — so it always reflects the current instructions. Indirect calls
    /// have no static target and are not included.
    ///
    /// A tail jump into another function's entry — the `Branch` a thunk or
    /// tail-call emits instead of a [`Call`](Mnemonic::Call) — is also a call
    /// edge and is included; see [`tail_call_target`].
    pub fn callees(&'s self) -> Vec<FunctionId> {
        let ctx = self.ctx();
        let mut callees = self
            .blocks()
            .flat_map(|block| {
                block
                    .instructions()
                    .filter_map(|insn| insn.mnemonic().call_target())
                    .chain(tail_call_target(ctx, block.id))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        // Synthetic edges (e.g. `entry → main`) recovered by a pass but not
        // backed by a direct call. Keyed by address; included once a function
        // exists at that address.
        callees.extend(
            ctx.values
                .synthetic_callees_of(self.id)
                .filter_map(|addr| Function::from_addr(ctx, addr).map(|function| function.id)),
        );
        callees.sort_by_key(|&id| Into::<usize>::into(id));
        callees.dedup();
        callees
    }

    /// Whether this function contains at least one indirect call — a
    /// [`CallInd`](Mnemonic::CallInd) through a computed function pointer that
    /// analysis could not resolve to a static [`Call`](Mnemonic::Call) target.
    pub fn has_indirect_call(&'s self) -> bool {
        self.blocks().any(|block| {
            block
                .instructions()
                .any(|insn| matches!(insn.mnemonic(), Mnemonic::CallInd(_)))
        })
    }

    /// Whether this function contains at least one [`Map`](Mnemonic::Map)
    /// instruction — a lane-wise array map operation. Surfaced as an advanced
    /// filter in the function list.
    pub fn has_map(&'s self) -> bool {
        self.blocks().any(|block| {
            block
                .instructions()
                .any(|insn| matches!(insn.mnemonic(), Mnemonic::Map(_)))
        })
    }

    /// Whether this function contains at least one [`Scan`](Mnemonic::Scan)
    /// instruction — a lane-wise prefix-fold array operation. Surfaced as an
    /// advanced filter in the function list, alongside [`has_map`](Self::has_map).
    pub fn has_scan(&'s self) -> bool {
        self.blocks().any(|block| {
            block
                .instructions()
                .any(|insn| matches!(insn.mnemonic(), Mnemonic::Scan(_)))
        })
    }

    /// The functions that directly call this one, deduplicated and ordered by
    /// id. Reads the reverse call graph maintained alongside the use-def map and
    /// resolves each call site to its enclosing function. Counterpart of
    /// [`callees`](Self::callees).
    pub fn callers(&'s self) -> Vec<FunctionId> {
        let ctx = self.ctx();
        let mut callers = ctx
            .values
            .call_sites_of(self.id)
            .iter()
            .filter_map(|&site| {
                Instruction::from_id(ctx, site)
                    .block()
                    .and_then(|block| block.function())
                    .map(|function| function.id)
            })
            .collect::<Vec<_>>();

        // Tail-call/thunk callers reach us through a `Branch` into our entry
        // rather than a recorded call site, so they are absent from the reverse
        // call-graph map. Recover them from the entry block's CFG predecessors:
        // any predecessor in another function whose terminator tail-jumps here.
        if let Some(root) = self.root() {
            for (_edge, pred_id) in root.predecessors() {
                if tail_call_target(ctx, pred_id) == Some(self.id)
                    && let Some(caller) = BasicBlock::from_id(ctx, pred_id).function()
                {
                    callers.push(caller.id);
                }
            }
        }

        callers.sort_by_key(|&id| Into::<usize>::into(id));
        callers.dedup();
        callers
    }

    /// The root block of this function, if it exists.
    pub fn root(&'s self) -> Option<BlockRef<'str, 'ctx>> {
        self.inner().root.map(|id| BlockRef::new(self.ctx(), id))
    }

    /// An iterator over the (live) blocks belonging to this function.
    pub fn blocks(&'s self) -> impl Iterator<Item = BlockRef<'str, 'ctx>> + 's {
        let ctx = self.ctx();
        let func = self.id;
        let mut ids = self.block_ids();
        // Total order: primarily by machine address, but break ties by the
        // function-local index. Address-less blocks (e.g. fallthrough splits,
        // whose `address()` is `None`) must still order deterministically.
        ids.sort_by_key(|&id| (BlockRef::new(ctx, id).address(), id.local));
        let _ = func;
        ids.into_iter().map(move |id| BlockRef::new(ctx, id))
    }

    /// The composite ids of this function's live (owned, non-tombstoned) blocks,
    /// in roster order.
    pub fn block_ids(&'s self) -> Vec<BlockId> {
        let ctx = self.ctx();
        self.inner()
            .roster
            .iter()
            .copied()
            .filter(|&id| !ctx.values.block(id).deleted)
            .collect()
    }

    /// The composite ids of this function's live (non-tombstoned) instructions,
    /// in arena order — including any currently detached (`parent == None`).
    pub fn instruction_ids(&'s self) -> Vec<InstructionId> {
        let func = self.id;
        self.inner()
            .insns
            .iter()
            .filter(|i| !i.deleted)
            .map(|i| InstructionId::new(func, i.id))
            .collect()
    }

    /// The composite ids of every CFG edge in this function's edge arena.
    /// Includes dangling edges (removal leaves the `EdgeData` slot in place),
    /// matching the previous whole-context `Graph::edges` behavior.
    pub fn edge_ids(&'s self) -> Vec<crate::value::block::EdgeId> {
        let func = self.id;
        self.inner()
            .edges
            .iter()
            .map(|e| crate::value::block::EdgeId::new(func, e.id))
            .collect()
    }

    /// Iterates over the (live) blocks in this function in arena order (i.e. not
    /// sorted by address, unlike [`blocks`](Self::blocks)).
    pub fn iter(&'s self) -> BlockIter<'str, 'ctx> {
        BlockIter {
            ctx: self.ctx(),
            inner: self.block_ids().into_iter(),
        }
    }

    fn fmt(&'s self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.is_external() {
            return writeln!(f, "extern fn {};", self.name());
        }
        let keyword = match self.kind() {
            FunctionKind::Machine => "fn",
            FunctionKind::Lambda => "lambda",
        };
        writeln!(f, "{keyword} {}:", self.name())?;
        for block in self.blocks() {
            block.fmt(f)?;
        }
        Ok(())
    }
}

/// If `block`'s terminator is an unconditional `Branch` into the *entry* of a
/// *different* function, return that function — the call-graph edge a thunk or
/// tail call produces (`jmp realfunc`) instead of a [`Call`](Mnemonic::Call).
///
/// Returns `None` for a fall-through branch within the same function, a branch
/// into the middle of another function (not a call), or any non-`Branch`
/// terminator. Shared by [`Function::callees`] and [`Function::callers`] so both
/// directions of the graph agree on what counts as a tail-call edge.
fn tail_call_target(ctx: &Context, block: BlockId) -> Option<FunctionId> {
    let block = BasicBlock::from_id(ctx, block);
    let Mnemonic::Branch(Branch { target, .. }) = block.instructions().last()?.mnemonic() else {
        return None;
    };
    let caller = block.function()?.id;
    let callee = BasicBlock::from_id(ctx, *target).function()?;
    let enters_at_entry = callee.root().map(|root| root.id) == Some(*target);
    (enters_at_entry && callee.id != caller).then_some(callee.id)
}

pub type FunctionRef<'str, 'ctx> = BaseRef<&'ctx Context<'str>, FunctionId>;

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 'ctx, 'str> for FunctionRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'ctx Context<'str> {
        self.ctx
    }
}

impl Named for FunctionRef<'_, '_> {
    fn name(&self) -> Option<&str> {
        Some(self.ctx.values.functions[self.id].name.as_ref())
    }
}

impl Display for FunctionRef<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.fmt(f)
    }
}

impl<'str, 'ctx> Value<'str, 'ctx> for FunctionRef<'str, 'ctx> {
    fn id(&self) -> ValueId {
        self.id()
    }

    fn size(&self) -> usize {
        self.size()
    }
}

pub struct BlockIter<'str, 'ctx> {
    ctx: &'ctx Context<'str>,
    inner: std::vec::IntoIter<BlockId>,
}

impl<'str, 'ctx> Iterator for BlockIter<'str, 'ctx> {
    type Item = BlockRef<'str, 'ctx>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|id| BlockRef::new(self.ctx, id))
    }
}

impl<'str, 'ctx> IntoIterator for &FunctionRef<'str, 'ctx> {
    type Item = BlockRef<'str, 'ctx>;
    type IntoIter = BlockIter<'str, 'ctx>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub type FunctionMutRef<'str, 'ctx> = BaseRef<&'ctx mut Context<'str>, FunctionId>;

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 's, 'str> for FunctionMutRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'s Context<'str> {
        self.ctx
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtxMut<'s, 'str> for FunctionMutRef<'str, 'ctx> {
    fn ctx_mut(&'s mut self) -> &'s mut Context<'str> {
        self.ctx
    }
}

impl Display for FunctionMutRef<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.fmt(f)
    }
}

impl<'ctx, 'str> Value<'str, 'ctx> for FunctionMutRef<'str, 'ctx> {
    fn id(&self) -> ValueId {
        self.id()
    }

    fn size(&self) -> usize {
        self.size()
    }
}

impl Named for FunctionMutRef<'_, '_> {
    fn name(&self) -> Option<&str> {
        Some(self.ctx.values.functions[self.id].name.as_ref())
    }
}

impl<'str, 'ctx> Renameable<'str, 'ctx> for FunctionMutRef<'str, 'ctx> {
    fn rename(&mut self, name: Cow<'str, str>) -> Result<()> {
        let id = self.id();
        let old_name = self.ctx.values.functions[self.id].name.as_ref().to_owned();
        update_context_name(id, self.ctx, name.clone(), Some(old_name.as_ref()))?;
        self.ctx.values.functions[self.id].name = name;
        Ok(())
    }
}

impl<'str, 'ctx> FunctionMutRef<'str, 'ctx> {
    pub(crate) fn inner_mut(&mut self) -> &mut Function<'str> {
        &mut self.ctx.values.functions[self.id]
    }

    fn set_address(&mut self, address: u64) -> Result<()> {
        self.inner_mut().address = Some(address);
        self.ctx.set_address(address, self.id.into())
    }

    fn with_address(mut self, address: u64) -> Result<Self> {
        self.set_address(address)?;
        Ok(self)
    }

    /// Sets a block as the root of this function.
    /// This will also add the block to the function's block list if it's not already present.
    /// This will also set the address of the function/block to the address of the root block/function if both addresses are unset.
    /// Panics if the function already has an address that doesn't match the root block's address.
    pub fn set_root(&mut self, id: BlockId) -> Result<()> {
        self.add_block(id);
        self.inner_mut().root = Some(id);

        let block_addr = BasicBlock::from_id(self.ctx, id).address();
        let self_addr = self.address();

        match (self_addr, block_addr) {
            (Some(fn_addr), Some(block_addr)) if fn_addr != block_addr => {
                return Err(Error::spanless(ErrorTy::FunctionRootAddressMismatch {
                    fn_addr,
                    block_addr,
                }));
            }
            (None, Some(addr)) => {
                self.set_address(addr)
                    .expect("This address should be valid");
            }
            (Some(addr), None) => {
                BasicBlock::from_id_mut(self.ctx, id)
                    .set_address(addr)
                    .expect("This address should be valid");
            }
            _ => {}
        }
        Ok(())
    }

    pub fn make_root(&mut self) -> BlockRef<'str, '_> {
        let func = self.id;
        let root = BasicBlock::make(self.ctx, func).id;
        self.set_root(root).expect("We just created the block");
        BlockRef::new(self.ctx, root)
    }

    pub fn ensure_root(&mut self, id: BlockId) -> Result<()> {
        if let Some(root) = self.inner().root {
            if root != id {
                return Err(Error::spanless(ErrorTy::FunctionRootMismatch {
                    expected: root,
                    actual: id,
                }));
            }
            Ok(())
        } else {
            self.set_root(id)
        }
    }

    pub fn set_external(&mut self, is_external: bool) {
        self.inner_mut().is_external = is_external;
        assert!(
            self.inner().blocks.is_empty(),
            "External functions should not have blocks"
        );
    }

    pub fn set_kind(&mut self, kind: FunctionKind) {
        self.inner_mut().kind = kind;
        if kind == FunctionKind::Lambda {
            self.set_is_pure(true);
            self.set_pure_reg(true);
        }
    }

    pub fn set_signature(&mut self, sig: FunctionSignature) {
        self.ctx.values.functions[self.id].signature = Some(sig);
    }

    /// Records the inferred per-parameter pointer attributes on this function.
    /// See [`FunctionSignature::param_attrs`].
    pub fn set_param_attrs(&mut self, attrs: Vec<ParamAttrs>) {
        self.inner_mut()
            .signature
            .get_or_insert_default()
            .param_attrs = Some(attrs);
    }

    /// Drops any inferred per-parameter attributes (e.g. after a signature
    /// rewrite changed the parameter list, invalidating the index alignment).
    pub fn clear_param_attrs(&mut self) {
        if let Some(sig) = self.inner_mut().signature.as_mut() {
            sig.param_attrs = None;
        }
    }

    /// Records the analysis-computed clobbered register set on this function.
    pub fn set_clobbered_regs(&mut self, regs: Vec<VarnodeId>) {
        self.inner_mut().signature.get_or_insert_default().clobbered = Some(regs);
    }

    /// Records the analysis-computed set of non-register spaces this function may
    /// write (`None` = unknown/unbounded). See
    /// [`FunctionSignature::written_spaces`].
    pub fn set_written_spaces(&mut self, spaces: Option<Vec<crate::space::SpaceId>>) {
        self.inner_mut()
            .signature
            .get_or_insert_default()
            .written_spaces = spaces;
    }

    /// Marks this function's register effect as fully captured by its call
    /// interface — reads no registers, writes exactly its clobbered set. See
    /// [`FunctionSignature::externally_resolved`].
    pub fn set_externally_resolved(&mut self, value: bool) {
        self.inner_mut()
            .signature
            .get_or_insert_default()
            .externally_resolved = value;
    }

    /// Records the analysis-inferred input (live-in) register set on this function.
    ///
    /// Legacy ABI input-register list — see [`input_regs`](Self::input_regs).
    /// Not set for `pure_reg` functions, whose param interface supersedes it.
    #[deprecated(
        note = "legacy ABI register list; None for pure_reg functions. Use the root block params \
                as the call interface; this remains only for the conventional/external path."
    )]
    pub fn set_input_regs(&mut self, regs: Vec<VarnodeId>) {
        self.inner_mut().signature.get_or_insert_default().inputs = Some(regs);
    }

    /// Records display names for this function's positional call arguments, one
    /// per `Call.args` slot. Consulted by [`input_arg_name`](Self::input_arg_name)
    /// for callees (chiefly externals) whose argument names come from a C
    /// prototype rather than a register or promoted stack param.
    pub fn set_input_arg_names(&mut self, names: Vec<Option<Box<str>>>) {
        self.inner_mut()
            .signature
            .get_or_insert_default()
            .input_names = Some(names);
    }

    /// Marks this function as fully functionalized over its register channel.
    /// See [`FunctionSignature::pure_reg`].
    pub fn set_pure_reg(&mut self, value: bool) {
        self.inner_mut().signature.get_or_insert_default().pure_reg = value;
    }

    /// Marks this function as fully functionalized over *every* side-effect
    /// channel — a deterministic pure function of its params. See
    /// [`FunctionSignature::is_pure`].
    pub fn set_is_pure(&mut self, value: bool) {
        self.inner_mut().signature.get_or_insert_default().is_pure = value;
    }

    /// Records the analysis-inferred saved (preserved) register set on this function.
    pub fn set_saved_regs(&mut self, regs: Vec<VarnodeId>) {
        self.inner_mut().signature.get_or_insert_default().saved = Some(regs);
    }

    /// Records the output (return-value) register set on this function.
    pub fn set_output_regs(&mut self, regs: Vec<VarnodeId>) {
        self.inner_mut().signature.get_or_insert_default().outputs = Some(regs);
    }

    /// Records the analysis-inferred net stack-pointer delta on this function.
    pub fn set_stack_delta(&mut self, delta: i64) {
        self.inner_mut()
            .signature
            .get_or_insert_default()
            .stack_delta = Some(delta);
    }

    /// Records whether this function performs an unresolved/dynamic stack read.
    /// See [`FunctionSignature::reads_unbounded_stack`].
    pub fn set_reads_unbounded_stack(&mut self, value: bool) {
        self.inner_mut()
            .signature
            .get_or_insert_default()
            .reads_unbounded_stack = value;
    }

    /// Records whether this function hands a pointer into its own frame to a
    /// callee that may read it unboundedly. See
    /// [`FunctionSignature::frame_escapes_to_unbounded`].
    pub fn set_frame_escapes_to_unbounded(&mut self, value: bool) {
        self.inner_mut()
            .signature
            .get_or_insert_default()
            .frame_escapes_to_unbounded = value;
    }

    /// Records the address of a machine instruction lifted into this function.
    pub fn add_instruction_addr(&mut self, addr: u64) {
        self.inner_mut().instruction_addrs.insert(addr);
    }

    /// Associates `block` with `function` by setting the block's `parent` field.
    ///
    /// With per-function block arenas, membership *is* arena ownership: a block
    /// lives in the arena of the function it was born into (`id.func`), and that
    /// must equal `self.id`. This is now effectively an assertion plus a
    /// `parent` (re)assignment; it no longer moves storage between functions.
    pub fn add_block(&mut self, id: BlockId) {
        let prev = self.ctx.values.block(id).parent;
        if prev == Some(self.id) {
            // Already owned; ensure the roster lists it exactly once (a freshly
            // `make`d block is auto-rostered, so this is usually a no-op).
            if !self.inner().roster.contains(&id) {
                self.inner_mut().roster.push(id);
            }
            return;
        }
        // Re-home: drop from the previous owner's roster, claim it here.
        if let Some(prev) = prev {
            self.ctx.values.functions[prev].roster.retain(|&b| b != id);
        }
        self.ctx.values.block_mut(id).parent = Some(self.id);
        self.inner_mut().roster.push(id);
    }

    /// Removes `block` from this function: drops it from the ownership roster and
    /// tombstones it. The arena slot is never reclaimed; the block is skipped by
    /// [`blocks`](Self::blocks).
    ///
    /// To *delete* a block with its CFG edges/instructions/params unwound, use
    /// [`BasicBlock::delete`].
    pub fn remove_block(&mut self, id: BlockId) {
        self.ctx.values.unroster_block(id);
        let block = self.ctx.values.block_mut(id);
        block.parent = None;
        block.deleted = true;
    }
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;

    /// A tail jump (`Branch`) into another function's entry is a call edge in
    /// both directions of the graph, even though the IR has no `Call`.
    #[test]
    fn tail_jump_into_entry_is_a_call_edge() {
        use crate::builder::Builder;

        let mut ctx = Context::new();

        // Callee at 0x2000: a single block that returns.
        let callee = Function::make_at_addr(&mut ctx, 0x2000, None).id;
        let callee_entry = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .with_address(0x2000)
        .id;
        let zero = ctx.get_const(0, 8).id();
        Builder::from_block(BasicBlock::from_id_mut(&mut ctx, callee_entry)).push_return(zero);
        Function::from_id_mut(&mut ctx, callee)
            .set_root(callee_entry)
            .unwrap();

        // Thunk at 0x1000: a lone `jmp` into the callee's entry.
        let thunk = Function::make_at_addr(&mut ctx, 0x1000, None).id;
        let thunk_entry = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .with_address(0x1000)
        .id;
        Builder::from_block(BasicBlock::from_id_mut(&mut ctx, thunk_entry))
            .push_branch(callee_entry);
        ctx.add_cfg_edge(thunk_entry, callee_entry);
        Function::from_id_mut(&mut ctx, thunk)
            .set_root(thunk_entry)
            .unwrap();

        assert_eq!(Function::from_id(&ctx, thunk).callees(), vec![callee]);
        assert_eq!(Function::from_id(&ctx, callee).callers(), vec![thunk]);
    }

    /// A `Branch` into the *middle* of another function is not a call edge — a
    /// call enters at the entry, not at an interior block.
    #[test]
    fn tail_jump_into_interior_block_is_not_a_call_edge() {
        use crate::builder::Builder;

        let mut ctx = Context::new();

        let callee = Function::make_at_addr(&mut ctx, 0x2000, None).id;
        let callee_entry = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .with_address(0x2000)
        .id;
        let interior = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .with_address(0x2008)
        .id;
        let zero = ctx.get_const(0, 8).id();
        Builder::from_block(BasicBlock::from_id_mut(&mut ctx, interior)).push_return(zero);
        Function::from_id_mut(&mut ctx, callee).add_block(interior);
        Function::from_id_mut(&mut ctx, callee)
            .set_root(callee_entry)
            .unwrap();

        let thunk = Function::make_at_addr(&mut ctx, 0x1000, None).id;
        let thunk_entry = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .with_address(0x1000)
        .id;
        Builder::from_block(BasicBlock::from_id_mut(&mut ctx, thunk_entry)).push_branch(interior);
        ctx.add_cfg_edge(thunk_entry, interior);
        Function::from_id_mut(&mut ctx, thunk)
            .set_root(thunk_entry)
            .unwrap();

        assert!(Function::from_id(&ctx, thunk).callees().is_empty());
        assert!(Function::from_id(&ctx, callee).callers().is_empty());
    }

    #[test]
    fn make_function_creates_function_with_correct_name_root_address() {
        let mut ctx = Context::new();
        let f = Function::make(&mut ctx, "main".into()).unwrap();
        assert_eq!(f.name(), "main");
    }

    #[test]
    fn get_function_by_name_returns_correct_function() {
        let mut ctx = Context::new();
        let id = Function::make(&mut ctx, "foo".into()).unwrap().id();
        let f = Function::from_name(&ctx, "foo").unwrap();
        assert_eq!(f.id(), id);
        assert_eq!(f.name(), "foo");
    }

    #[test]
    fn get_function_by_name_returns_none_if_not_found() {
        let ctx = Context::new();
        assert!(Function::from_name(&ctx, "nonexistent").is_none());
    }

    #[test]
    fn get_function_by_addr_returns_correct_function() {
        let mut ctx = Context::new();
        let id = Function::make_at_addr(&mut ctx, 0x2000, None).id();
        let f = Function::from_addr(&ctx, 0x2000).unwrap();
        assert_eq!(f.id(), id);
        assert_eq!(f.address(), Some(0x2000));
        assert_eq!(f.name(), "fn_2000");
    }

    #[test]
    fn get_function_by_addr_returns_none_if_missing() {
        let ctx = Context::new();
        assert!(Function::from_addr(&ctx, 0xdeadbeef).is_none());
    }

    #[test]
    fn add_block_via_function_mut_ref_updates_blocks_list() {
        let mut ctx = Context::new();
        let root = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .id;
        let extra = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .id;

        let mut baz = Function::make(&mut ctx, "baz".into()).unwrap();
        baz.add_block(root);
        baz.add_block(extra);

        let block_ids: Vec<_> = baz.blocks().map(|b| b.id).collect();
        assert!(block_ids.contains(&root));
        assert!(block_ids.contains(&extra));
    }

    #[test]
    fn display_shows_function_name_and_block_contents() {
        let mut ctx = Context::new();
        Function::make(&mut ctx, "display_test".into()).unwrap();

        let f = Function::from_name(&ctx, "display_test").unwrap();

        let s = f.to_string();
        assert!(s.contains("fn display_test:"));
    }

    /// A BasicBlock may be created at an address before the Function stub for
    /// that address is registered (e.g. when Sleigh emits a branch target block
    /// ahead of the function being lifted).  `set_address` must allow this and
    /// must attribute the block as the function's root.
    #[test]
    fn iter_yields_all_blocks() {
        let mut ctx = Context::new();
        let root = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .id;
        let extra = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .id;
        let mut f = Function::make(&mut ctx, "iter_fn".into()).unwrap();
        f.add_block(root);
        f.add_block(extra);

        let f = Function::from_name(&ctx, "iter_fn").unwrap();
        let ids: Vec<_> = f.iter().map(|b| b.id).collect();
        assert!(ids.contains(&root));
        assert!(ids.contains(&extra));
    }

    #[test]
    fn into_iterator_for_function_ref_matches_iter() {
        let mut ctx = Context::new();
        let b1 = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .id;
        let b2 = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .id;
        let mut f = Function::make(&mut ctx, "into_iter_fn".into()).unwrap();
        f.add_block(b1);
        f.add_block(b2);

        let f = Function::from_name(&ctx, "into_iter_fn").unwrap();
        let mut via_iter: Vec<usize> = f.iter().map(|b| usize::from(b.id.local)).collect();
        let mut via_into: Vec<usize> = (&f).into_iter().map(|b| usize::from(b.id.local)).collect();
        via_iter.sort();
        via_into.sort();
        assert_eq!(via_iter, via_into);
    }

    #[test]
    fn qcode_fn_single_block_populates_function() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn simple:
                <entry>
                    return at 0;
            "
        );

        let f = Function::from_name(&ctx, "simple").unwrap();
        assert_eq!(f.name(), "simple");
        assert!(f.root().is_some());
        assert_eq!(f.root().unwrap().name().unwrap(), "entry");
        assert_eq!(f.blocks().count(), 1);
    }

    #[test]
    fn qcode_fn_multi_block_populates_all_blocks() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn multiblock:
                <bb1>
                    if i8 1 goto <bb2> else goto <bb3>;

                <bb2>
                    goto <bb3>;

                <bb3>
                    return at 0;
            "
        );

        let f = Function::from_name(&ctx, "multiblock").unwrap();
        assert_eq!(f.root().unwrap().name().unwrap(), "bb1");
        let block_names: Vec<_> = f.blocks().filter_map(|b| b.name()).collect();
        assert!(block_names.contains(&"bb1"), "missing bb1");
        assert!(block_names.contains(&"bb2"), "missing bb2");
        assert!(block_names.contains(&"bb3"), "missing bb3");
        assert_eq!(f.blocks().count(), 3);
    }

    #[test]
    fn qcode_fn_id_variable_is_set() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn myfn:
                <start>
                    return at 0;
            "
        );

        let by_name = Function::from_name(&ctx, "myfn").unwrap();
        assert_eq!(by_name.name(), "myfn");
    }

    #[test]
    fn set_address_allows_function_at_existing_block_address() {
        let mut ctx = Context::new();

        // Simulate a branch-target block created at 0x1000 before the function
        // stub exists (as happens with tail-jumps to sibling functions).
        let block_id = {
            let __f = ctx.anon_function();
            BasicBlock::make(&mut ctx, __f)
        }
        .id;
        ctx.set_address(0x1000, ValueId::BasicBlock(block_id))
            .unwrap();

        // Registering a function at the same address must succeed.
        let fn_id = Function::make_at_addr(&mut ctx, 0x1000, None).id;

        // The function wins in the address map.
        assert!(Function::from_addr(&ctx, 0x1000).is_some());
        // The pre-existing block becomes the function's root.
        assert_eq!(
            Function::from_id(&ctx, fn_id).root().map(|b| b.id),
            Some(block_id)
        );
    }
}
