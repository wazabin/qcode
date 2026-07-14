//! Fluent IR builder: emit instructions into a [`BasicBlock`].
//!
//! The [`Builder`] is the primary way to construct IR. It holds a mutable
//! reference to a block inside a [`Context`] and exposes typed `push_*` methods
//! for every instruction kind.
//! When the builder is dropped (or [`Builder::finalize`] is called),
//! it verifies that the block ends with a terminator instruction.
//!
//! # Typical usage
//!
//! ```rust,ignore
//! use qcode_core::{context::Context, builder::Builder};
//!
//! let mut ctx = Context::new();
//!
//! // Create a builder positioned at machine address 0x1000.
//! let mut b = Builder::from_context(&mut ctx, 0x1000);
//!
//! // Emit instructions …
//!
//! // Terminate the block with an unconditional branch to 0x1010.
//! // This consumes the builder, so there is no need to call drop explicitly.
//! b.finalize(0x1010);
//! ```
//!
//! # Namespaces
//!
//! The builder maintains a *local namespace*: a map from string names to
//! [`ValueId`]s. This is used by the [`qcode!`](qcode_macro::qcode) macro and
//! the parser to resolve identifiers within a single block. Names in the
//! namespace do not need to match the IR-level name hints stored on values.

use std::{borrow::Cow, cmp};

use std::marker::PhantomData;

use rustc_hash::FxHashMap as HashMap;

use crate::{
    context::Context,
    space::{SPACE_CONST, Space, SpaceId, SpaceType},
    types::{AggregateField, TypeId},
    value::{
        BodyView, FunctionBody, Instruction, ModuleView, QCodeView, Renameable, Value, ValueId,
        ValueRef,
        block::{BasicBlock, BlockId, EdgeId},
        block_param::BlockParamMutRef,
        function::FunctionId,
        insn::{
            Apply, Assert, Binary, Binop, Branch, BranchInd, CBranch, Call, CallInd, Callee, Carry,
            Extract, FloatBinop, FloatToFloat, FloatToInt, Gep, InstructionId, InstructionRef,
            IntBinop, IntToFloat, IntrinsicApp, IntrinsicId, IsFloatNaN, Load, LzCount, Map,
            Mnemonic, PCodeOp, PCodeOpId, PopCount, Range, Return, ReturnValue, SBorrow, SCarry,
            Scan, Sext, Store, TailCall, Tuple, Unary, Unop, Zext,
        },
        util::{base_ref::BaseRef, pass_backing::PassBacking},
        varnode::{Varnode, VarnodeId},
    },
};

/// The IR mutation surface the [`Builder`] needs, named with a `bb_` prefix so a
/// backing type exposes it without method-name ambiguity.
///
/// This is the seam that decouples the fluent builder from its backing
/// (context-split Option A): the [`Builder`] is generic over `B: BuilderBacking`
/// and performs *every* mutation through these methods. Two backings exist — the
/// **module** builder over `&mut Context` (the lifter / lowering / emulator
/// construction path, which additionally mints temp spaces via
/// [`Builder::make_temp`]) and the **function-pass** builder over a `PassBacking`
/// (a pass's own body borrowed in place). Each is a direct impl below.
#[doc(hidden)]
pub trait BuilderBacking<'str> {
    type ReadView<'a>: QCodeView<'a, 'str>
    where
        Self: 'a,
        'str: 'a;

    /// The module's shared IR state (read) — types, literals, spaces, registers,
    /// maps.
    fn bb_shr(&self) -> &crate::context::Shared<'str>;
    /// The module's shared data (write), for temp/varnode minting. Only the module
    /// backing (`&mut Context`) provides it; a pass backing panics — it holds only
    /// `&Shared`.
    fn bb_shared_mut(&mut self) -> &mut Context<'str> {
        unimplemented!("temp/varnode minting requires the module Builder (&mut Context)")
    }
    /// A `Copy` read view for the builder's arena reads (`ValueRef::from_view`,
    /// instruction/block-param/block reads).
    fn bb_view(&self) -> Self::ReadView<'_>;
    /// Whether the read view may resolve `f`'s body without crossing a
    /// function-pass ownership boundary.
    fn bb_can_read_body(&self, f: FunctionId) -> bool;
    /// The owning function's storage (write).
    fn bb_function_mut(&mut self, f: FunctionId) -> &mut FunctionBody<'str>;
    /// The instruction `id`, routed to its owning function's arena (write).
    fn bb_instruction_mut(&mut self, id: InstructionId) -> &mut Instruction<'str>;
    /// The block `id`, routed to its owning function's arena (write).
    fn bb_block_mut(&mut self, id: BlockId) -> &mut BasicBlock<'str>;
    /// Register a function-local (block/instruction/param/Temp) or global name.
    fn bb_register_local_name(
        &mut self,
        id: ValueId,
        name: Cow<'str, str>,
        old: Option<&str>,
    ) -> crate::error::Result<()>;
    /// Push a fresh instruction into `func`'s arena (use-map + call-site upkeep).
    fn bb_push_insn(&mut self, func: FunctionId, insn: Instruction<'str>) -> InstructionId;
    /// Push a fresh block into `func`'s arena and onto its roster.
    fn bb_push_block(&mut self, func: FunctionId, block: BasicBlock<'str>) -> BlockId;
    /// Add a directed CFG edge `from -> to`, stored in `from`'s edge arena.
    fn bb_add_cfg_edge(&mut self, from: BlockId, to: BlockId) -> EdgeId;

    /// Append `id` to the end of `block`, setting its parent.
    fn bb_block_append_insn(&mut self, block: BlockId, id: InstructionId) {
        self.bb_instruction_mut(id).parent = Some(block.local);
        self.bb_block_mut(block)
            .instructions
            .push(id.localize(block.func));
    }
    /// Insert `id` at `index` in `block`, shifting later instructions right, and
    /// set its parent.
    fn bb_block_insert_insn_at(&mut self, block: BlockId, index: usize, id: InstructionId) {
        self.bb_instruction_mut(id).parent = Some(block.local);
        self.bb_block_mut(block)
            .instructions
            .insert(index, id.localize(block.func));
    }
}

/// The **module** builder backing (`&mut Context`): the lifter / lowering /
/// emulator construction path, which additionally mints temp spaces (concrete
/// `Builder<&mut Context>`, see [`Builder::make_temp`]). Every method routes
/// through `Context`'s inherent verbs — the builder no longer needs a host trait.
impl<'str> BuilderBacking<'str> for &mut Context<'str> {
    type ReadView<'a>
        = ModuleView<'a, 'str>
    where
        Self: 'a,
        'str: 'a;

    fn bb_shr(&self) -> &crate::context::Shared<'str> {
        &self.shared
    }
    fn bb_shared_mut(&mut self) -> &mut Context<'str> {
        self
    }
    fn bb_view(&self) -> Self::ReadView<'_> {
        ModuleView::new(self)
    }
    fn bb_can_read_body(&self, _f: FunctionId) -> bool {
        true
    }
    fn bb_function_mut(&mut self, f: FunctionId) -> &mut FunctionBody<'str> {
        &mut self.bodies[f]
    }
    fn bb_instruction_mut(&mut self, id: InstructionId) -> &mut Instruction<'str> {
        Context::instruction_mut(self, id)
    }
    fn bb_block_mut(&mut self, id: BlockId) -> &mut BasicBlock<'str> {
        Context::block_mut(self, id)
    }
    fn bb_register_local_name(
        &mut self,
        id: ValueId,
        name: Cow<'str, str>,
        old: Option<&str>,
    ) -> crate::error::Result<()> {
        use crate::error::{Error, ErrorTy};
        let existing = match id.name_scope_function() {
            Some(func) => self.bodies[func]
                .names
                .get(&name)
                .map(|id| id.qualify(func)),
            None => self.get_named(&name),
        };
        if let Some(existing) = existing {
            return if existing == id {
                Ok(())
            } else {
                Err(Error::spanless(ErrorTy::DuplicateName(name.to_string())))
            };
        }
        match id.name_scope_function() {
            Some(func) => self.bodies[func]
                .names
                .register(name, id.localize(func), old),
            None => self.update_name(name, id, old),
        }
    }
    fn bb_push_insn(&mut self, func: FunctionId, insn: Instruction<'str>) -> InstructionId {
        Context::push_insn(self, func, insn)
    }
    fn bb_push_block(&mut self, func: FunctionId, block: BasicBlock<'str>) -> BlockId {
        Context::push_block(self, func, block)
    }
    fn bb_add_cfg_edge(&mut self, from: BlockId, to: BlockId) -> EdgeId {
        Context::add_cfg_edge(self, from, to)
    }
}

/// The **function-pass** builder backing (a checked-out body): every method routes
/// through the owned `FunctionBody`'s inherent verbs plus the read-only shared context.
/// `bb_shared_mut` is intentionally left as the defaulted panic — a pass-time
/// builder holds a frozen shared view and cannot mint temp spaces.
impl<'str> BuilderBacking<'str> for PassBacking<'_, 'str> {
    type ReadView<'a>
        = BodyView<'a, 'str>
    where
        Self: 'a,
        'str: 'a;

    fn bb_shr(&self) -> &crate::context::Shared<'str> {
        self.shared
    }
    fn bb_view(&self) -> Self::ReadView<'_> {
        self.view()
    }
    fn bb_can_read_body(&self, f: FunctionId) -> bool {
        f == self.fun.id()
    }
    fn bb_function_mut(&mut self, f: FunctionId) -> &mut FunctionBody<'str> {
        assert_eq!(
            f,
            self.fun.id(),
            "a checked-out function pass may not mutate another function"
        );
        self.fun
    }
    fn bb_instruction_mut(&mut self, id: InstructionId) -> &mut Instruction<'str> {
        self.fun.insn_mut(id)
    }
    fn bb_block_mut(&mut self, id: BlockId) -> &mut BasicBlock<'str> {
        self.fun.block_mut(id)
    }
    fn bb_register_local_name(
        &mut self,
        id: ValueId,
        name: Cow<'str, str>,
        old: Option<&str>,
    ) -> crate::error::Result<()> {
        self.fun.register_local_name(self.shared, id, name, old)
    }
    fn bb_push_insn(&mut self, func: FunctionId, insn: Instruction<'str>) -> InstructionId {
        assert_eq!(func, self.fun.id());
        self.fun.push_insn(insn)
    }
    fn bb_push_block(&mut self, func: FunctionId, block: BasicBlock<'str>) -> BlockId {
        assert_eq!(func, self.fun.id());
        self.fun.push_block(block)
    }
    fn bb_add_cfg_edge(&mut self, from: BlockId, to: BlockId) -> EdgeId {
        self.fun.add_cfg_edge(from, to)
    }
}

/// A builder for constructing instructions in a block.
/// This provides a convenient API for creating instructions, and automatically
/// manages temporary values and labels.
pub struct Builder<'str, 'ctx, Ctx: BuilderBacking<'str> = &'ctx mut Context<'str>> {
    pub block: BaseRef<Ctx, BlockId>,

    /// Converts from names to value IDs in the current scope.
    namespace: HashMap<Cow<'str, str>, ValueId>,

    /// Names of local labels to their corresponding block IDs.
    local_labels: HashMap<Cow<'str, str>, BlockId>,

    /// The address at which instructions are added
    address: Option<u64>,

    /// Is the block terminated, i.e. does it end with a terminator
    /// If it is not the case, the block might be invalid
    pub(crate) is_terminated: bool,

    verify_terminated: bool,

    /// Explicit insert position for new instructions.
    ///
    /// `None` (default) appends to the end of the block.
    /// `Some(n)` inserts at index `n` and auto-advances after each push,
    /// so consecutive pushes form a contiguous sequence starting at `n`.
    insert_point: Option<usize>,

    /// Ties the (default) `'ctx` to the module-borrow lifetime when
    /// `Ctx = &'ctx mut Context`; phantom for other hosts.
    _ctx: PhantomData<&'ctx ()>,
}

/// Generates a canonical comparison method and its "greater-than" mirror (operands swapped).
macro_rules! cmp_pair {
    ($fwd:ident, $rev:ident, $op:expr) => {
        pub fn $fwd(
            &mut self,
            lhs: ValueId,
            rhs: ValueId,
        ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
            self.push_binop($op, lhs, rhs, Some(1))
        }
        pub fn $rev(
            &mut self,
            lhs: ValueId,
            rhs: ValueId,
        ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
            self.push_binop($op, rhs, lhs, Some(1))
        }
    };
}

/// Temp-space minting is a **module-Builder-only** capability (context-split
/// Option A): a fresh temporary address space is a `&mut Shared` push, so only
/// the Builder instantiated over `&mut Context` (the lifter / lowering / emulator
/// construction path) can mint one. A parallel-safe function-pass Builder holds a
/// frozen shared view and therefore cannot — and, per the audit, never needs to
/// (every temp-minting call site is on the `&mut Context` path). Restricting
/// these to the concrete instantiation makes that boundary a compile-time fact
/// instead of a runtime `unimplemented!()` on the checked-out host.
impl<'str, 'ctx> Builder<'str, 'ctx, &'ctx mut Context<'str>> {
    pub fn make_temp(&mut self, size: usize) -> VarnodeId {
        let space = self.context_mut().make_temp_space();
        Varnode::make(self.context_mut(), 0, size, space).id
    }

    /// Creates a new temporary value with the given name and size.
    /// This value is a memory value so does not need to follow any SSA rules.
    /// The name is deduplicated with a numeric suffix if already taken in the context.
    pub fn make_named_temp(&mut self, name: Cow<'str, str>, size: usize) -> VarnodeId {
        let space = self.context_mut().make_temp_space();
        let id = Varnode::make(self.context_mut(), 0, size, space).id;
        let unique_name = self.context_mut().get_unique_name(name);
        Varnode::from_id_mut(self.context_mut(), id)
            .rename(unique_name.clone())
            .expect("This name was deduplicated");
        self.context_mut().shared.spaces[space].name = Some(unique_name.as_ref().into());
        id
    }

    /// Creates a new temporary value identified by an integer `label`, used to
    /// derive its display name (`v{label}`) lazily.
    ///
    /// Unlike [`make_named_temp`](Self::make_named_temp), this does not allocate a
    /// name `String`, probe for a unique name, or insert into the context's name
    /// map — so it stays off the per-instruction hot path. The temporary's
    /// identity is its [`VarnodeId`]; callers that need distinct temporaries are
    /// responsible for using distinct varnodes (the emitter keys them by
    /// `(size, local)`), so no name-uniqueness check is needed.
    pub fn make_temp_labeled(&mut self, label: u32, size: usize) -> VarnodeId {
        let space = self.context_mut().make_temp_space();
        let id = Varnode::make(self.context_mut(), 0, size, space).id;
        Varnode::from_id_mut(self.context_mut(), id).set_label(label);
        id
    }
}

impl<'str, 'ctx, Ctx: BuilderBacking<'str>> Builder<'str, 'ctx, Ctx> {
    /// Creates a builder positioned at `block`.
    ///
    /// The block is borrowed mutably for the lifetime `'ctx`. New instructions
    /// will be appended to the end of `block`.
    pub fn from_block(block: BaseRef<Ctx, BlockId>) -> Self {
        let is_terminated = block
            .host_ref()
            .bb_view()
            .block_ref(block.id)
            .is_terminated();
        Self {
            is_terminated,
            verify_terminated: true,
            block,
            namespace: HashMap::default(),
            local_labels: HashMap::default(),
            address: None,
            insert_point: None,
            _ctx: PhantomData,
        }
    }

    /// A `Copy` read view over the builder's backing, for arena reads. The builder
    /// reads through the backing's static [`QCodeView`].
    pub fn view(&self) -> Ctx::ReadView<'_> {
        self.block.host_ref().bb_view()
    }

    /// Returns `true` if the current block ends with a terminator instruction.
    pub fn is_terminated(&self) -> bool {
        self.view().block_ref(self.block.id).is_terminated()
    }

    /// Sets the current address for instructions added by this builder.
    pub fn set_address(&mut self, addr: u64) {
        self.address = Some(addr);
    }

    /// Remove the current address
    pub fn clear_address(&mut self) {
        self.address = None;
    }

    /// Positions the builder at the beginning of the block.
    ///
    /// Subsequent `push_*` calls insert instructions starting at index 0,
    /// advancing by 1 after each push, so they appear in push order as a
    /// contiguous prefix before any pre-existing instructions.
    ///
    /// This allows inserting synthetic preamble instructions (e.g. a
    /// symbolic stack-pointer initialization) into a block that already
    /// contains lifted code, without disturbing the relative order of
    /// either the new or the existing instructions.
    pub fn set_insert_point_to_start(&mut self) {
        self.insert_point = Some(0);
    }

    /// Positions the builder immediately before an existing instruction in the
    /// current block.
    ///
    /// Subsequent `push_*` calls insert instructions starting at that position,
    /// advancing by 1 after each push, so they appear in push order immediately
    /// before `before_id` and after any earlier inserted instructions.
    ///
    /// Panics if `before_id` is not an instruction in the current block.
    pub fn set_insert_point_before(&mut self, before_id: InstructionId) {
        let index = self
            .view()
            .block_ref(self.block.id)
            .instruction_ids()
            .iter()
            .position(|&id| id == before_id)
            .expect("before_id not found in block");
        self.insert_point = Some(index);
    }

    /// Resets the insert point to append mode (the default).
    pub fn set_insert_point_to_end(&mut self) {
        self.insert_point = None;
    }

    /// Disables the termination check that runs when the builder is dropped.
    ///
    /// Normally dropping an un-terminated builder panics. Call this when the
    /// caller guarantees that either:
    /// - the block will be terminated by a *parent* builder before the overall
    ///   IR is used (e.g. when emitting code from a macro where the macro body
    ///   does not own the final branch), or
    /// - the block is intentionally left open as an intermediate state.
    ///
    /// # Safety
    ///
    /// The caller must ensure that by the time this context's IR is inspected
    /// or executed, the block owned by this builder is properly terminated
    /// (i.e. its last instruction is a terminator). Failing to do so will
    /// produce malformed IR that may panic or produce incorrect results in
    /// downstream passes.
    pub unsafe fn dont_finalize(&mut self) {
        self.verify_terminated = false;
    }

    /// Gets a sub-value from a given value, specified by a byte range.
    pub fn get_range(
        &mut self,
        src: ValueId,
        range: std::ops::Range<usize>,
    ) -> Option<ValueRef<'str, '_, Ctx::ReadView<'_>>> {
        let value = self.get_value(src);

        if range.is_empty() {
            return None;
        }

        let dst = match value {
            ValueRef::Literal(literal) => {
                let value = literal.value();
                let id = self.shr().get_const(value, range.len());
                self.get_value(id)
            }

            ValueRef::Varnode(varnode_ref) => {
                if range.end > varnode_ref.size() {
                    return None;
                }

                let base = varnode_ref.address() + range.start as i64;
                let space = varnode_ref.space().id;

                let id = Varnode::make(self.context_mut(), base, range.len(), space).id;

                Varnode::from_id(self.shr(), id).into()
            }

            ValueRef::Instruction(insn) => {
                if range.end > insn.size() {
                    return None;
                }

                self.push_instruction_in_space(
                    Mnemonic::Range(Range {
                        src: self.loc(src),
                        start: range.start,
                        size: range.len(),
                    }),
                    range.len(),
                    self.get_value(src).space().map(|s| s.id),
                )
                .into()
            }

            // Functions, blocks, and other non-data values have no byte range
            _ => return None,
        };

        Some(dst)
    }

    /// Pushes a `Range` instruction extracting `size` bytes starting at byte
    /// `start` of `src`. Unlike [`get_range`](Self::get_range), this always emits
    /// a `Range` instruction (no constant/varnode folding), so the result is a
    /// fresh SSA value — used by the `qcode!` macro's `src[start:end]` form.
    pub fn push_range(
        &mut self,
        src: ValueId,
        start: usize,
        size: usize,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let space = self.get_value(src).space().map(|s| s.id);
        self.push_instruction_in_space(
            Mnemonic::Range(Range {
                src: self.loc(src),
                start,
                size,
            }),
            size,
            space,
        )
    }

    /// Removes a name from the local namespace, freeing it for reuse.
    pub fn remove_alias(&mut self, name: &str) {
        self.namespace.remove(name);
    }

    /// Sets a name in the local alias map without changing the qcode name hint.
    /// The alias map maps sleigh names to values for macro lookups; re-aliasing is allowed.
    pub fn set_alias(&mut self, name: Cow<'str, str>, id: ValueId) {
        self.namespace.insert(name, id);
    }

    pub fn switch_to_block(&mut self, block: BlockId) {
        self.block.id = block;
        self.is_terminated = self.view().block_ref(block).is_terminated();
    }

    /// The block the builder is currently appending to.
    pub fn current_block(&self) -> BlockId {
        self.block.id
    }

    /// Gets the ID of a value in the current namespace
    pub fn try_get_value(&self, name: &str) -> Option<ValueRef<'str, '_, Ctx::ReadView<'_>>> {
        self.namespace.get(name).map(|&id| self.get_value(id))
    }

    /// The module's shared context (write) — types/varnodes/spaces mint. Routed
    /// through the host, so a checked-out builder mints into shared storage while
    /// arena writes stay in the owned function.
    pub fn context_mut(&mut self) -> &mut Context<'str> {
        self.block.host_mut().bb_shared_mut()
    }

    /// The module's shared IR state (read) — types/literals/spaces/registers.
    pub fn shr(&self) -> &crate::context::Shared<'str> {
        self.block.host_ref().bb_shr()
    }

    /// Retype an instruction's result as a pointer into `space` (host-routed
    /// mirror of [`InstructionMutRef::set_space`]): the type mint is shared, the
    /// `type_id` write goes to the owning function's arena.
    fn set_insn_space(&mut self, id: InstructionId, space: SpaceId) {
        if matches!(Space::from_id(self.shr(), space).ty, SpaceType::Register) {
            return;
        }
        let cur_type = self.block.host_mut().bb_instruction_mut(id).type_id;
        let size = self.shr().types.size_of(cur_type);
        let type_id = self.shr().types.get_or_make_space_address(size, space);
        self.block.host_mut().bb_instruction_mut(id).type_id = type_id;
    }

    /// Rename an instruction's result (host-routed mirror of the instruction
    /// `Renameable`): registers the (function-local) name in the owning function's
    /// table and sets the arena field.
    fn rename_insn(&mut self, id: InstructionId, name: Cow<'str, str>) -> crate::error::Result<()> {
        let old = self.block.host_mut().bb_instruction_mut(id).name.clone();
        self.block.host_mut().bb_register_local_name(
            ValueId::Instruction(id),
            name.clone(),
            old.as_deref(),
        )?;
        self.block.host_mut().bb_instruction_mut(id).name = Some(name);
        Ok(())
    }

    /// Adds an instruction at the end of the working block.
    ///
    /// # Panics
    ///
    /// Panics if the block is already terminated (ends with a branch/call/return).
    #[track_caller]
    fn push_instruction(
        &mut self,
        mnemonic: Mnemonic,
        size: usize,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let type_id = self.shr().types.get_or_make_int(size);
        self.push_instruction_with_type(mnemonic, type_id)
    }

    /// Adds an instruction with an explicit result type, for callers that compute
    /// the type themselves. Needed by passes that reference a *minted*
    /// (not-yet-installed) function from a `Map`/`Scan`/`Apply`: the typed
    /// `push_map`/`push_scan`/`push_apply` read the body function's return type
    /// through the shared context, where a minted placeholder has no installed
    /// body — so the pass supplies the type it already knows instead.
    #[track_caller]
    pub fn push_mnemonic_with_type(
        &mut self,
        mnemonic: Mnemonic,
        type_id: TypeId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_instruction_with_type(mnemonic, type_id)
    }

    #[track_caller]
    fn push_instruction_in_space(
        &mut self,
        mnemonic: Mnemonic,
        size: usize,
        _space: Option<SpaceId>,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let type_id = self.shr().types.get_or_make_int(size);
        self.push_instruction_with_type(mnemonic, type_id)
    }

    #[track_caller]
    fn push_instruction_with_type(
        &mut self,
        mnemonic: Mnemonic,
        type_id: TypeId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        if self.is_terminated && self.insert_point.is_none() {
            let block_address = self.view().block_ref(self.block.id).address();
            if let Some(address) = self.address.or(block_address) {
                panic!("cannot append instruction to a terminated block at {address:#x}");
            }
            panic!("cannot append instruction to a terminated block");
        }

        let func = self.block.id.func;
        let block_id = self.block.id;
        let insn = Instruction::new(type_id, mnemonic);
        let id = self.block.host_mut().bb_push_insn(func, insn);

        if let Some(address) = self.address {
            self.block
                .host_mut()
                .bb_instruction_mut(id)
                .set_address(address);
        }

        match self.insert_point {
            None => self.block.host_mut().bb_block_append_insn(block_id, id),
            Some(ref mut pos) => {
                let index = *pos;
                self.block
                    .host_mut()
                    .bb_block_insert_insn_at(block_id, index, id);
                *pos += 1;
            }
        }

        InstructionRef::new(self.block.host_ref().bb_view(), id)
    }

    fn get_value(&self, id: ValueId) -> ValueRef<'str, '_, Ctx::ReadView<'_>> {
        // Route through the host's read view so a checked-out builder resolves its
        // own function's SSA values (which live in the owned function, not the
        // shared context) correctly.
        ValueRef::from_view(self.view(), id)
    }

    /// Localize a qualified operand id for storage in a mnemonic built for this
    /// builder's working block. Strict IR locality (context-split ruling 2)
    /// guarantees the operand lives in this function's arena, so its owning
    /// `FunctionId` is the block's own `id.func`.
    fn loc(&self, id: ValueId) -> crate::value::LocalValueId {
        id.localize(self.block.id.func)
    }

    /// Localize a whole operand list (call/branch/tuple/intrinsic args).
    fn loc_vec(&self, ids: Vec<ValueId>) -> Vec<crate::value::LocalValueId> {
        let func = self.block.id.func;
        ids.into_iter().map(|v| v.localize(func)).collect()
    }

    /// The result type of `id`, host-routed (mirror of [`Context::type_of`]): a
    /// checked-out function's instruction/param types live in the owned arena.
    fn type_of(&mut self, id: ValueId) -> TypeId {
        match id {
            ValueId::Instruction(iid) => self.view().instruction(iid).type_id,
            ValueId::BlockParam(pid) => self.view().block_param(pid).type_id,
            other => self.view().type_of(other),
        }
    }

    /// The stored type of `id`, host-routed (mirror of [`Context::stored_type_of`]).
    fn stored_type_of(&self, id: ValueId) -> Option<TypeId> {
        match id {
            ValueId::Instruction(iid) => Some(self.view().instruction(iid).type_id),
            ValueId::BlockParam(pid) => Some(self.view().block_param(pid).type_id),
            other => self.view().stored_type_of(other),
        }
    }

    /// The common address-space provenance of two pointer-arithmetic operands.
    ///
    /// Returns the space carried by whichever operand has one (varnodes carry
    /// their space; pointer-typed instructions carry theirs), or `None` when the
    /// two disagree or neither has a space.
    fn merge_space_ids(&self, lhs: ValueId, rhs: ValueId) -> Option<SpaceId> {
        match (
            self.get_value(lhs).space().map(|s| s.id),
            self.get_value(rhs).space().map(|s| s.id),
        ) {
            (Some(lhs), Some(rhs)) if lhs == rhs => Some(lhs),
            (Some(space), None) | (None, Some(space)) => Some(space),
            _ => None,
        }
    }

    fn is_literal(&self, id: ValueId) -> bool {
        matches!(id, ValueId::Literal(_))
    }

    fn coerce_literal_size(&mut self, id: ValueId, size: usize) -> ValueId {
        let ValueId::Literal(lit_id) = id else {
            return id;
        };
        let literal = self.shr().values.literals[lit_id].clone();
        let current_size = self.shr().types.size_of(literal.type_id);
        if current_size == size || literal.symbolic.is_some() {
            return id;
        }
        self.shr().get_const(literal.value, size)
    }

    pub fn get_or_make_local_label(&mut self, name: Cow<'str, str>) -> BlockId {
        if let Some(&id) = self.local_labels.get(name.as_ref()) {
            return id;
        }
        // SLEIGH pcode label names (e.g. `start`, `end`) are only unique within a
        // single instruction's lowering, but block names are function-scoped.
        // Deduplicate with a numeric suffix; the `local_labels` map stays keyed by
        // the original name so within-instruction references still resolve here.
        // Routed through the host so a checked-out builder mints the block into its
        // owned function's arena (and registers the name in that function's table).
        let func = self.block.id.func;
        let unique_name = self
            .block
            .host_mut()
            .bb_function_mut(func)
            .names
            .unique(name.clone());
        let id = self
            .block
            .host_mut()
            .bb_push_block(func, BasicBlock::detached(func));
        self.block
            .host_mut()
            .bb_register_local_name(ValueId::BasicBlock(id), unique_name.clone(), None)
            .expect("name was deduplicated");
        self.block
            .host_mut()
            .bb_block_mut(id)
            .set_name(Some(unique_name));
        self.local_labels.insert(name, id);
        id
    }

    /// Ensures an operand is not a varnode.
    /// If the operand is a varnode, emits a load from the varnode into a new temporary local, and returns the temp.
    /// If the operand is already a local, returns it as-is.
    pub fn ensure_local(&mut self, src: ValueId) -> ValueId {
        let value = self.get_value(src);

        match value {
            ValueRef::Varnode(node) => {
                let node_id = node.id;
                let id = self
                    .push_load::<false>(src, node.size(), node.space().id)
                    .id();

                // If the varnode has a name, give the temp a related name for easier debugging
                if let Some(name) = Varnode::from_id(self.shr(), node_id).name() {
                    let lowered = name.to_lowercase();
                    let func = self.block.id.func;
                    let name = self.context_mut().get_unique_name_in(func, lowered.into());

                    Instruction::from_id_mut(
                        self.context_mut(),
                        id.as_instruction()
                            .expect("In this context, push_load creates an instruction"),
                    )
                    .rename(name)
                    .expect("This name was deduplicated");
                };

                id
            }

            _ => src,
        }
    }

    /// Loads a value from memory, given a pointer value. Optionally specify the address space and size of the load.
    /// If the load space is the special `CONST` space, the pointer is treated as an immediate value rather than an address.
    ///
    /// # Panics
    ///
    /// Panics if `space` is [`SPACE_CONST`] and `src` is not a `Literal` value.
    #[track_caller]
    pub fn push_load<const CHECK_LOCAL: bool>(
        &mut self,
        mut src: ValueId,
        size: usize,
        space: SpaceId,
    ) -> ValueRef<'str, '_, Ctx::ReadView<'_>> {
        if CHECK_LOCAL {
            src = self.ensure_local(src);
        }

        if space == SPACE_CONST {
            let src = self.get_value(src);

            match src {
                ValueRef::Literal(lit) => {
                    let value = lit.value();
                    let id = self.shr().get_const(value, size);
                    self.get_value(id)
                }

                _ => panic!("Expected literal value for CONST space load"),
            }
        } else {
            // Invariant: if the ptr is a varnode, it must live in the same space as the load.
            // A cross-space access (e.g. *[ram]:8 RSP) requires ensure_local first so that
            // the varnode's *value* is used as the address, not the varnode itself.

            match src {
                ValueId::Varnode(id) => {
                    let varnode = Varnode::from_id(self.shr(), id);
                    if varnode.space().id != space {
                        panic!(
                            "push_load: ptr is a varnode but its space {:?} does not match the load space {:?}; \
                             call ensure_local on the ptr first",
                            varnode.space().id,
                            space
                        );
                    }
                }

                ValueId::Instruction(id) => {
                    self.set_insn_space(id, space);
                }

                _ => {}
            }

            self.push_instruction(
                Mnemonic::Load(Load {
                    ptr: self.loc(src),
                    space: space.into(),
                    size,
                }),
                size,
            )
            .into()
        }
    }

    // --- Unary Ops ---

    fn push_unop(&mut self, op: Unop, src: ValueId) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        assert!(
            !src.is_varnode(),
            "push_unop: varnode operand is not allowed; use ensure_local or &name addressof syntax"
        );
        let size = self.get_value(src).size();
        self.push_instruction(
            Mnemonic::Unop(Unary {
                op,
                src: self.loc(src),
            }),
            size,
        )
    }

    /// Logical NOT of a `bool` value, canonically `src == false`.
    pub fn push_bool_not(&mut self, src: ValueId) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        debug_assert!(
            self.stored_type_of(src)
                .is_some_and(|t| self.shr().types.is_bool(t)),
            "push_bool_not: operand must be bool-typed"
        );
        let f = self.shr().get_bool_const(false);
        self.push_binop(Binop::Int(IntBinop::Equal), src, f, Some(1))
    }

    /// Creates a bitwise NOT operation on the given value.
    pub fn push_bit_negate(&mut self, src: ValueId) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_unop(Unop::IntNot, src)
    }

    /// Creates a negation operation on the given value.
    pub fn push_neg(&mut self, src: ValueId) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_unop(Unop::IntNegate, src)
    }

    /// Creates a float negation operation on the given value.
    pub fn push_fneg(&mut self, src: ValueId) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_unop(Unop::FloatNegate, src)
    }

    fn push_binop(
        &mut self,
        op: Binop,
        lhs: ValueId,
        rhs: ValueId,
        size: Option<usize>,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let lhs_size = self.get_value(lhs).size();
        let rhs_size = self.get_value(rhs).size();
        let operand_size = match (
            lhs_size == rhs_size,
            self.is_literal(lhs),
            self.is_literal(rhs),
        ) {
            (true, _, _) => lhs_size,
            (false, true, false) => rhs_size,
            (false, false, true) => lhs_size,
            (false, true, true) => lhs_size.max(rhs_size),
            // Two non-literal operands of differing size cannot be repaired here
            // without choosing a semantic cast. Lifters should emit explicit
            // zext/sext/range operations before constructing the binop.
            (false, false, false) => lhs_size,
        };
        let lhs = self.coerce_literal_size(lhs, operand_size);
        let rhs = self.coerce_literal_size(rhs, operand_size);
        assert_eq!(
            self.get_value(lhs).size(),
            self.get_value(rhs).size(),
            "push_binop: operands must have equal size; emit an explicit cast first"
        );

        // Determine result type using the TypeManager's arithmetic rules.
        let result_type = {
            let lhs_type = self.type_of(lhs);
            let rhs_type = self.type_of(rhs);
            self.shr().types.binop_result(lhs_type, op, rhs_type)
        };

        // Comparisons always override the result size to 1.
        let result_type = if let Some(forced_size) = size {
            let current_size = self.shr().types.size_of(result_type);
            if forced_size != current_size {
                self.shr().types.get_or_make_int(forced_size)
            } else {
                result_type
            }
        } else {
            result_type
        };

        // Restore address-space provenance for pointer arithmetic. When the
        // result is not already a space pointer (e.g. a `StackAddress` produced
        // from a stack-base operand), `Add`/`Sub` inherit the space of whichever
        // operand carries one — so `&A + k` points into `A`'s space. Register
        // spaces are excluded (pointer arithmetic is not allowed there).
        let result_type = if self.shr().types.space_of(result_type).is_none()
            && matches!(op, Binop::Int(IntBinop::Add | IntBinop::Sub))
        {
            match self.merge_space_ids(lhs, rhs) {
                Some(space)
                    if !matches!(Space::from_id(self.shr(), space).ty, SpaceType::Register) =>
                {
                    let size = self.shr().types.size_of(result_type);
                    self.shr().types.get_or_make_space_address(size, space)
                }
                _ => result_type,
            }
        } else {
            result_type
        };

        self.push_instruction_with_type(
            Mnemonic::Binop(Binary {
                op,
                lhs: self.loc(lhs),
                rhs: self.loc(rhs),
            }),
            result_type,
        )
    }

    // --- Arithmetic ---

    pub fn push_mul(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Int(IntBinop::Mul), lhs, rhs, None)
    }

    pub fn push_div(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Int(IntBinop::Div), lhs, rhs, None)
    }

    pub fn push_sdiv(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Int(IntBinop::Sdiv), lhs, rhs, None)
    }

    pub fn push_mod(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Int(IntBinop::Rem), lhs, rhs, None)
    }

    pub fn push_smod(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Int(IntBinop::Srem), lhs, rhs, None)
    }

    pub fn push_add(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Int(IntBinop::Add), lhs, rhs, None)
    }

    pub fn push_sub(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Int(IntBinop::Sub), lhs, rhs, None)
    }

    // --- Float Arithmetic ---

    pub fn push_fdiv(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Float(FloatBinop::Div), lhs, rhs, None)
    }

    pub fn push_fmul(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Float(FloatBinop::Mul), lhs, rhs, None)
    }

    pub fn push_fadd(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Float(FloatBinop::Add), lhs, rhs, None)
    }

    pub fn push_fsub(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Float(FloatBinop::Sub), lhs, rhs, None)
    }

    // --- Shifts ---

    pub fn push_shl(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Int(IntBinop::ShiftLeft), lhs, rhs, None)
    }

    pub fn push_shr(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Int(IntBinop::ShiftRight), lhs, rhs, None)
    }

    pub fn push_sshr(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Int(IntBinop::SShiftRight), lhs, rhs, None)
    }

    // --- Integer Comparisons ---
    // Greater-than variants swap operands of the less-than op.

    cmp_pair!(push_slt, push_sgt, Binop::Int(IntBinop::SLess));
    cmp_pair!(push_sle, push_sge, Binop::Int(IntBinop::SLessEqual));
    cmp_pair!(push_lt, push_gt, Binop::Int(IntBinop::Less));
    cmp_pair!(push_le, push_ge, Binop::Int(IntBinop::LessEqual));

    // --- Float Comparisons ---

    cmp_pair!(push_flt, push_fgt, Binop::Float(FloatBinop::Less));
    cmp_pair!(push_fle, push_fge, Binop::Float(FloatBinop::LessEqual));

    // --- Integer Equality ---

    pub fn push_eq(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Int(IntBinop::Equal), lhs, rhs, Some(1))
    }

    pub fn push_ne(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Int(IntBinop::NotEqual), lhs, rhs, Some(1))
    }

    pub fn push_feq(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Float(FloatBinop::Equal), lhs, rhs, Some(1))
    }

    pub fn push_fne(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Float(FloatBinop::NotEqual), lhs, rhs, Some(1))
    }

    // --- Bitwise ---

    /// Logical XOR of two `bool` operands — a bitwise `Xor` over `bool`, which
    /// yields `bool` (exact on the `{0,1}` domain).
    pub fn push_bool_xor(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        debug_assert!(
            self.both_bool(lhs, rhs),
            "push_bool_xor: operands must be bool"
        );
        self.push_binop(Binop::Int(IntBinop::Xor), lhs, rhs, None)
    }

    /// Logical AND of two `bool` operands (bitwise `And` over `bool`).
    pub fn push_bool_and(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        debug_assert!(
            self.both_bool(lhs, rhs),
            "push_bool_and: operands must be bool"
        );
        self.push_binop(Binop::Int(IntBinop::And), lhs, rhs, None)
    }

    /// Logical OR of two `bool` operands (bitwise `Or` over `bool`).
    pub fn push_bool_or(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        debug_assert!(
            self.both_bool(lhs, rhs),
            "push_bool_or: operands must be bool"
        );
        self.push_binop(Binop::Int(IntBinop::Or), lhs, rhs, None)
    }

    /// Whether both operands carry the `bool` type (a `debug_assert` guard).
    fn both_bool(&self, lhs: ValueId, rhs: ValueId) -> bool {
        let is_bool = |v: ValueId| {
            self.stored_type_of(v)
                .is_some_and(|t| self.shr().types.is_bool(t))
        };
        is_bool(lhs) && is_bool(rhs)
    }

    pub fn push_bit_xor(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Int(IntBinop::Xor), lhs, rhs, None)
    }

    pub fn push_bit_or(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Int(IntBinop::Or), lhs, rhs, None)
    }

    pub fn push_bit_and(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_binop(Binop::Int(IntBinop::And), lhs, rhs, None)
    }

    // --- Extensions & Conversions ---

    pub fn push_is_nan(&mut self, src: ValueId) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        assert!(
            !src.is_varnode(),
            "push_is_nan: varnode operand not allowed"
        );
        self.push_instruction(Mnemonic::IsFloatNaN(IsFloatNaN { src: self.loc(src) }), 1)
    }

    pub fn push_abs(&mut self, src: ValueId) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_unop(Unop::FloatAbs, src)
    }

    pub fn push_sqrt(&mut self, src: ValueId) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_unop(Unop::FloatSqrt, src)
    }

    pub fn push_floor(&mut self, src: ValueId) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_unop(Unop::FloatFloor, src)
    }

    pub fn push_ceil(&mut self, src: ValueId) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_unop(Unop::FloatCeil, src)
    }

    pub fn push_round(&mut self, src: ValueId) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_unop(Unop::FloatRound, src)
    }

    pub fn push_int_to_float(
        &mut self,
        src: ValueId,
        size: usize,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        assert!(
            !src.is_varnode(),
            "push_int_to_float: varnode operand not allowed"
        );
        self.push_instruction(
            Mnemonic::IntToFloat(IntToFloat {
                src: self.loc(src),
                size,
            }),
            size,
        )
    }

    pub fn push_float_to_float(
        &mut self,
        src: ValueId,
        size: usize,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        assert!(
            !src.is_varnode(),
            "push_float_to_float: varnode operand not allowed"
        );
        self.push_instruction(
            Mnemonic::FloatToFloat(FloatToFloat {
                src: self.loc(src),
                size,
            }),
            size,
        )
    }

    pub fn push_trunc(
        &mut self,
        src: ValueId,
        size: usize,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        assert!(!src.is_varnode(), "push_trunc: varnode operand not allowed");
        self.push_instruction(
            Mnemonic::FloatToInt(FloatToInt {
                src: self.loc(src),
                size,
            }),
            size,
        )
    }

    pub fn push_zext(
        &mut self,
        src: ValueId,
        size: usize,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        assert!(!src.is_varnode(), "push_zext: varnode operand not allowed");
        self.push_instruction(
            Mnemonic::Zext(Zext {
                src: self.loc(src),
                size,
            }),
            size,
        )
    }

    pub fn push_sext(
        &mut self,
        src: ValueId,
        size: usize,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        assert!(!src.is_varnode(), "push_sext: varnode operand not allowed");
        self.push_instruction(
            Mnemonic::Sext(Sext {
                src: self.loc(src),
                size,
            }),
            size,
        )
    }

    /// Builds an aggregate value from `fields` using default field names
    /// (`field1`, `field2`, ...). The result type is the
    /// [`Aggregate`](crate::types::TypeRepr::Aggregate) of the named fields'
    /// types.
    pub fn push_tuple(
        &mut self,
        fields: Vec<ValueId>,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let named_fields = fields
            .into_iter()
            .enumerate()
            .map(|(i, value)| (format!("field{}", i + 1), value))
            .collect();
        self.push_named_tuple(named_fields)
    }

    /// Builds an aggregate value from ordered named fields.
    pub fn push_named_tuple(
        &mut self,
        fields: Vec<(String, ValueId)>,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let field_types: Vec<TypeId> = fields.iter().map(|(_, f)| self.type_of(*f)).collect();
        let aggregate_fields = fields
            .iter()
            .zip(field_types)
            .map(|((name, _), type_id)| AggregateField::new(name.clone(), type_id))
            .collect();
        let ty = self
            .shr()
            .types
            .get_or_make_named_aggregate(aggregate_fields);
        let values = fields.into_iter().map(|(_, value)| value).collect();
        self.push_instruction_with_type(
            Mnemonic::Tuple(Tuple {
                fields: self.loc_vec(values),
            }),
            ty,
        )
    }

    /// Projects field `index` out of the aggregate value `agg`. The result type
    /// is that field's type. Panics if `agg` is not an aggregate with that field.
    pub fn push_extract(
        &mut self,
        agg: ValueId,
        index: usize,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let agg_ty = self.type_of(agg);
        let ty = self
            .shr()
            .types
            .field_type(agg_ty, index)
            .expect("push_extract: agg is not an aggregate with that field index");
        self.push_instruction_with_type(
            Mnemonic::Extract(Extract {
                agg: self.loc(agg),
                index,
            }),
            ty,
        )
    }

    /// Builds a total element-wise map `out[i] = body(src[i], captures…)` over the
    /// array value `src`. The body is **unary** in the element (index-aware bodies
    /// take an [`enumerate`](crate::value::insn::Intrinsic) tuple as that element);
    /// `body` is a function symbol, not an operand. Soundness of the body (pure,
    /// element-local) is the recognizer's obligation; the builder only wires the
    /// value graph.
    ///
    /// The result is `[U; N]` where `N` is `src`'s element count and `U` is the
    /// body's return type — which need not equal the input element type (e.g. a
    /// map over `enumerate(arr)` consumes tuples but returns bare elements). When
    /// the body is a bare symbol with no return (or `src` is not an array), the
    /// result falls back to `src`'s type.
    pub fn push_map(
        &mut self,
        body: impl Into<Callee>,
        src: ValueId,
        captures: Vec<ValueId>,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let body = body.into();
        let src_ty = self.type_of(src);
        // `map` preserves the source's sequence kind: an array maps to an array,
        // a list (e.g. `take_while`'s result) maps to a list of the same bound.
        let seq = self.shr().types.seq_of(src_ty);
        let ret_ty = body.real().and_then(|body| self.map_body_return_type(body));
        let ty = match (seq, ret_ty) {
            (Some((_, len, is_list)), Some(rt)) => {
                self.shr().types.get_or_make_seq(rt, len, is_list)
            }
            _ => src_ty,
        };
        self.push_instruction_with_type(
            Mnemonic::Map(Map {
                body,
                src: self.loc(src),
                captures: self.loc_vec(captures),
            }),
            ty,
        )
    }

    /// Builds a total left-scan `out[i] = body(acc_i, src[i], captures…)` with
    /// `acc_0 = init` over the array value `src` (see [`Scan`]). The body is
    /// **binary** in `(accumulator, element)` — index-aware bodies take an
    /// [`enumerate`](crate::value::insn::Intrinsic) tuple as the element; `body`
    /// is a function symbol, not an operand. Soundness of the body (pure, with the
    /// accumulator threaded only through the scan) is the recognizer's obligation.
    ///
    /// The result is `[U; N]` where `N` is `src`'s element count and `U` is the
    /// body's return type (also the accumulator type). When the body is a bare
    /// symbol with no return (or `src` is not an array), the result falls back to
    /// `src`'s type.
    pub fn push_scan(
        &mut self,
        body: impl Into<Callee>,
        init: ValueId,
        src: ValueId,
        captures: Vec<ValueId>,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let body = body.into();
        let src_ty = self.type_of(src);
        // Like `map`, a scan preserves the source's sequence kind and takes its
        // element type from the body's return type (the accumulator type).
        let seq = self.shr().types.seq_of(src_ty);
        let ret_ty = body.real().and_then(|body| self.map_body_return_type(body));
        let ty = match (seq, ret_ty) {
            (Some((_, len, is_list)), Some(rt)) => {
                self.shr().types.get_or_make_seq(rt, len, is_list)
            }
            _ => src_ty,
        };
        self.push_instruction_with_type(
            Mnemonic::Scan(Scan {
                body,
                init: self.loc(init),
                src: self.loc(src),
                captures: self.loc_vec(captures),
            }),
            ty,
        )
    }

    /// Builds a value-level application of a pure lambda function. Unlike
    /// [`push_call`](Self::push_call), this is an ordinary SSA instruction and
    /// does not terminate the current block.
    pub fn push_apply(
        &mut self,
        target: impl Into<Callee>,
        args: Vec<ValueId>,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let target = target.into();
        let ty = target
            .real()
            .and_then(|target| self.lambda_return_type(target))
            .unwrap_or_else(|| {
                args.first()
                    .map(|&arg| self.type_of(arg))
                    .unwrap_or_else(|| self.shr().types.get_or_make_int(0))
            });
        self.push_instruction_with_type(
            Mnemonic::Apply(Apply {
                target,
                args: self.loc_vec(args),
            }),
            ty,
        )
    }

    /// The type of the value returned by `body`'s first `Return`, or `None` if
    /// `body` has no root or returns nothing — used to size a [`push_map`] result.
    fn map_body_return_type(&self, body: FunctionId) -> Option<TypeId> {
        // A checked-out builder has no access to other function bodies, so no
        // return type is recoverable through this path.
        if !self.block.host_ref().bb_can_read_body(body) {
            return None;
        }
        let view = self.view();
        let root = view.function_ref(body).root()?.id;
        view.block_ref(root)
            .iter()
            .find_map(|i| match i.mnemonic() {
                Mnemonic::Return(r) => r
                    .value
                    .and_then(|v| self.stored_type_of(v.qualify(i.id.func))),
                _ => None,
            })
    }

    /// The type of the first value returned by a lambda body.
    fn lambda_return_type(&self, body: FunctionId) -> Option<TypeId> {
        // See `map_body_return_type` on the checked-out fallback.
        if !self.block.host_ref().bb_can_read_body(body) {
            return None;
        }
        self.view()
            .function_ref(body)
            .iter()
            .flat_map(|block| block.iter())
            .find_map(|i| match i.mnemonic() {
                Mnemonic::ReturnValue(r) => self.stored_type_of(r.value.qualify(i.id.func)),
                _ => None,
            })
    }

    /// Computes the address of the field at byte `offset` of the struct that
    /// `base` points at: `gep(base, offset)`. `base` must have a
    /// [`StructPointer`](crate::types::TypeRepr::StructPointer) type whose
    /// pointee has a field at exactly `offset`. The result type is a pointer
    /// (same width as `base`) to that field's type. Panics otherwise.
    pub fn push_gep(
        &mut self,
        base: ValueId,
        offset: usize,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let base_ty = self.type_of(base);
        let types = &self.shr().types;
        let ptr_width = types.size_of(base_ty);
        let pointee = types
            .pointee_of(base_ty)
            .expect("push_gep: base is not a struct pointer");
        let field_ty = types
            .field_by_offset(pointee, offset)
            .map(|(_, field)| field.type_id)
            .expect("push_gep: no field at that offset in the pointee struct");
        let ty = self
            .shr()
            .types
            .get_or_make_struct_pointer(ptr_width, field_ty);
        self.push_instruction_with_type(
            Mnemonic::Gep(Gep {
                base: self.loc(base),
                offset,
            }),
            ty,
        )
    }

    /// Like [`push_gep`](Builder::push_gep) but selects the field by name,
    /// resolving it to a byte offset via the pointee struct of `base`. Panics if
    /// `base` is not a struct pointer or has no field of that name.
    pub fn push_gep_field(
        &mut self,
        base: ValueId,
        name: &str,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let base_ty = self.type_of(base);
        let types = &self.shr().types;
        let pointee = types
            .pointee_of(base_ty)
            .expect("push_gep_field: base is not a struct pointer");
        let offset = types
            .aggregate_fields(pointee)
            .and_then(|fields| fields.iter().find(|f| f.name == name))
            .map(|f| f.offset)
            .expect("push_gep_field: pointee struct has no field of that name");
        self.push_gep(base, offset)
    }

    pub fn push_popcount(
        &mut self,
        src: ValueId,
        size: usize,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        assert!(
            !src.is_varnode(),
            "push_popcount: varnode operand not allowed"
        );
        self.push_instruction(Mnemonic::PopCount(PopCount { src: self.loc(src) }), size)
    }

    pub fn push_lzcount(
        &mut self,
        src: ValueId,
        size: usize,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        assert!(
            !src.is_varnode(),
            "push_lzcount: varnode operand not allowed"
        );
        self.push_instruction(Mnemonic::LzCount(LzCount { src: self.loc(src) }), size)
    }

    pub fn push_carry(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        assert!(
            !lhs.is_varnode() && !rhs.is_varnode(),
            "push_carry: varnode operand not allowed"
        );
        self.push_instruction(
            Mnemonic::Carry(Carry {
                lhs: self.loc(lhs),
                rhs: self.loc(rhs),
            }),
            1,
        )
    }

    pub fn push_scarry(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        assert!(
            !lhs.is_varnode() && !rhs.is_varnode(),
            "push_scarry: varnode operand not allowed"
        );
        self.push_instruction(
            Mnemonic::SCarry(SCarry {
                lhs: self.loc(lhs),
                rhs: self.loc(rhs),
            }),
            1,
        )
    }

    pub fn push_sborrow(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        assert!(
            !lhs.is_varnode() && !rhs.is_varnode(),
            "push_sborrow: varnode operand not allowed"
        );
        self.push_instruction(
            Mnemonic::SBorrow(SBorrow {
                lhs: self.loc(lhs),
                rhs: self.loc(rhs),
            }),
            1,
        )
    }

    pub fn push_pcode_op(
        &mut self,
        id: PCodeOpId,
        args: Vec<ValueId>,
        dst: Option<ValueId>,
        size: usize,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let args = args
            .into_iter()
            .map(|arg| self.ensure_local(arg))
            .collect::<Vec<_>>();

        self.push_instruction(
            Mnemonic::PCodeOp(PCodeOp {
                id,
                args: self.loc_vec(args),
                dst: dst.map(|d| self.loc(d)),
            }),
            size,
        )
    }

    /// Creates a pure intrinsic instruction (e.g. `rol`, `ror`, `enumerate`).
    ///
    /// Validates the operand count against the intrinsic's declared arity and
    /// types the node via the intrinsic's
    /// [`result_type`](crate::value::insn::Intrinsic::result_type)
    /// rule, so the result carries its full type (not just a width) — an array
    /// or aggregate result is projectable. Panics on an arity mismatch.
    #[track_caller]
    pub fn push_intrinsic(
        &mut self,
        id: IntrinsicId,
        args: Vec<ValueId>,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let desc = id.desc();
        assert_eq!(
            args.len(),
            desc.arity(),
            "intrinsic `{}` expects {} args, got {}",
            desc.name(),
            desc.arity(),
            args.len()
        );

        let args = args
            .into_iter()
            .map(|arg| self.ensure_local(arg))
            .collect::<Vec<_>>();

        let arg_types = args
            .iter()
            .map(|&arg| self.type_of(arg))
            .collect::<Vec<_>>();
        let type_id = desc.result_type(&self.shr().types, &arg_types);

        self.push_instruction_with_type(
            Mnemonic::Intrinsic(IntrinsicApp {
                id,
                args: self.loc_vec(args),
            }),
            type_id,
        )
    }

    // --- Loads & Stores ---

    /// Creates a copy instruction from `src` to `dst`.
    /// Note that `dst` must already exist as a [`Value`] in the current context, and this will not create a new temporary value.
    /// If `dst` is a varnode, we aren't allowed to write to it, this is a store operation
    /// If `src` is a varnode, we need to read from it first, then write to dst
    /// For values wider than 64 bits (e.g. XMM/YMM/ZMM registers), emits one store per 64-bit lane.
    pub fn push_copy(
        &mut self,
        src: ValueId,
        dst: VarnodeId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let node = Varnode::from_id(self.shr(), dst);
        let size = node.size();
        let space = node.space().id;

        const LANE_SIZE: usize = 8;

        if size > LANE_SIZE {
            let num_lanes = size.div_ceil(LANE_SIZE);
            let mut first_id = None;

            for lane in 0..num_lanes {
                let offset = lane * LANE_SIZE;
                let lane_size = cmp::min(LANE_SIZE, size - offset);

                let src_lane = self
                    .get_range(src, offset..offset + lane_size)
                    .expect("lane range in bounds")
                    .id();
                let src_lane = self.ensure_local(src_lane);

                let dst_lane = self
                    .get_range(dst.into(), offset..offset + lane_size)
                    .expect("lane range in bounds")
                    .id();

                let id = self
                    .push_instruction_in_space(
                        Mnemonic::Store(Store {
                            src: self.loc(src_lane),
                            ptr: self.loc(dst_lane),
                            space: space.into(),
                            size: lane_size,
                        }),
                        0,
                        Some(space),
                    )
                    .id;

                if let Some(name) = Varnode::from_id(self.shr(), dst).name() {
                    let name = Cow::Owned(format!("{}_lane{lane}", name.to_lowercase()));
                    let _ = self.rename_insn(id, name);
                }

                first_id.get_or_insert(id);
            }

            InstructionRef::new(self.view(), first_id.unwrap())
        } else {
            let src = self.ensure_local(src);
            // If dst is a varnode, we need to emit a store from src to dst
            let id = self
                .push_instruction_in_space(
                    Mnemonic::Store(Store {
                        src: self.loc(src),
                        ptr: self.loc(dst.into()),
                        space: space.into(),
                        size,
                    }),
                    0,
                    Some(space),
                )
                .id;

            // Add a name hint for the store instruction for easier debugging
            if let Some(name) = Varnode::from_id(self.shr(), dst).name() {
                let lowered = name.to_lowercase();
                let func = self.block.id.func;
                let name = self
                    .block
                    .host_mut()
                    .bb_function_mut(func)
                    .names
                    .unique(Cow::Owned(lowered));
                self.rename_insn(id, name)
                    .expect("This name was deduplicated");
            }

            InstructionRef::new(self.view(), id)
        }
    }

    #[track_caller]
    pub fn push_store(
        &mut self,
        src: ValueId,
        ptr: ValueId,
        space: SpaceId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let src = self.ensure_local(src);
        let size = self.get_value(src).size();

        match ptr {
            ValueId::Varnode(id) => {
                let varnode = Varnode::from_id(self.shr(), id);
                if varnode.space().id != space {
                    panic!(
                        "push_store: ptr is a varnode but its space {:?} does not match the store space {:?}; \
                             call ensure_local on the ptr first",
                        varnode.space().id,
                        space
                    );
                }
            }

            ValueId::Instruction(id) => {
                self.set_insn_space(id, space);
            }

            _ => {}
        }

        self.push_instruction(
            Mnemonic::Store(Store {
                src: self.loc(src),
                ptr: self.loc(ptr),
                space: space.into(),
                size,
            }),
            0,
        )
    }

    // --- Branches & Calls ---

    /// Declares a new parameter on the current block.
    ///
    /// Terminates this block with an unconditional jump to the given target block.
    /// The builder is now safe to drop without panicking, and the block is properly terminated.
    pub fn push_branch(&mut self, target: BlockId) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_branch_with_args(target, vec![])
    }

    /// Unconditional branch passing `args` to the target block's parameters.
    pub fn push_branch_with_args(
        &mut self,
        target: BlockId,
        args: Vec<ValueId>,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let current = self.block.id;
        self.block.host_mut().bb_add_cfg_edge(current, target);
        let target = target.localize(current.func);
        let id = self
            .push_instruction(
                Mnemonic::Branch(Branch {
                    target,
                    args: self.loc_vec(args),
                }),
                0,
            )
            .id;
        self.is_terminated = true;
        InstructionRef::new(self.view(), id)
    }

    pub fn push_cbranch(
        &mut self,
        condition: ValueId,
        target: BlockId,
        fallthrough: BlockId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_cbranch_with_args(condition, target, vec![], fallthrough, vec![])
    }

    /// Conditional branch with per-target arguments.
    pub fn push_cbranch_with_args(
        &mut self,
        condition: ValueId,
        target: BlockId,
        target_args: Vec<ValueId>,
        fallthrough: BlockId,
        fallthrough_args: Vec<ValueId>,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        assert!(
            !condition.is_varnode(),
            "push_cbranch: varnode condition not allowed; load the value first"
        );
        let current = self.block.id;
        self.block.host_mut().bb_add_cfg_edge(current, target);
        self.block.host_mut().bb_add_cfg_edge(current, fallthrough);
        let success_block = target.localize(current.func);
        let failure_block = fallthrough.localize(current.func);
        let id = self
            .push_instruction(
                Mnemonic::CBranch(CBranch {
                    success_block,
                    success_args: self.loc_vec(target_args),
                    condition: self.loc(condition),
                    failure_block,
                    failure_args: self.loc_vec(fallthrough_args),
                }),
                0,
            )
            .id;
        self.is_terminated = true;
        InstructionRef::new(self.view(), id)
    }

    pub fn push_branchind(&mut self, ptr: ValueId) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let id = self
            .push_instruction(Mnemonic::BranchInd(BranchInd { ptr: self.loc(ptr) }), 0)
            .id;
        self.is_terminated = true;
        InstructionRef::new(self.view(), id)
    }

    pub fn push_call(
        &mut self,
        target: impl Into<Callee>,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_call_with_args(target, vec![])
    }

    pub fn push_call_with_args(
        &mut self,
        target: impl Into<Callee>,
        args: Vec<ValueId>,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let target = target.into();
        let id = self
            .push_instruction(
                Mnemonic::Call(Call {
                    target,
                    args: self.loc_vec(args),
                    clobbers: vec![],
                }),
                0,
            )
            .id;
        self.is_terminated = true;
        InstructionRef::new(self.view(), id)
    }

    /// Tail call to another function's entry — a function-level terminator with
    /// no intra-function CFG successor (see [`TailCall`](crate::value::insn::TailCall)).
    /// Unlike [`push_branch`](Self::push_branch), this wires no CFG edge: control
    /// leaves the function.
    pub fn push_tail_call(
        &mut self,
        target: impl Into<Callee>,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_tail_call_with_args(target, vec![])
    }

    pub fn push_tail_call_with_args(
        &mut self,
        target: impl Into<Callee>,
        args: Vec<ValueId>,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let target = target.into();
        let id = self
            .push_instruction(
                Mnemonic::TailCall(TailCall {
                    target,
                    args: self.loc_vec(args),
                }),
                0,
            )
            .id;
        self.is_terminated = true;
        InstructionRef::new(self.view(), id)
    }

    pub fn push_call_ind(&mut self, ptr: ValueId) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_call_ind_with_args(ptr, vec![])
    }

    pub fn push_call_ind_with_args(
        &mut self,
        ptr: ValueId,
        args: Vec<ValueId>,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let id = self
            .push_instruction(
                Mnemonic::CallInd(CallInd {
                    ptr: self.loc(ptr),
                    args: self.loc_vec(args),
                }),
                0,
            )
            .id;
        self.is_terminated = true;
        InstructionRef::new(self.view(), id)
    }

    pub fn push_return(&mut self, ptr: ValueId) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_return_at(None, ptr)
    }

    pub fn push_return_with_value(
        &mut self,
        value: ValueId,
        ptr: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_return_at(Some(value), ptr)
    }

    fn push_return_at(
        &mut self,
        value: Option<ValueId>,
        ptr: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let id = self
            .push_instruction(
                Mnemonic::Return(Return {
                    ptr: self.loc(ptr),
                    value: value.map(|v| self.loc(v)),
                }),
                0,
            )
            .id;
        self.is_terminated = true;
        InstructionRef::new(self.view(), id)
    }

    pub fn push_return_value(
        &mut self,
        value: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        let id = self
            .push_instruction(
                Mnemonic::ReturnValue(ReturnValue {
                    value: self.loc(value),
                }),
                0,
            )
            .id;
        self.is_terminated = true;
        InstructionRef::new(self.view(), id)
    }

    // --- Assert ---

    /// Asserts that a condition holds at this point in execution
    pub fn push_assert(
        &mut self,
        condition: ValueId,
    ) -> InstructionRef<'str, '_, Ctx::ReadView<'_>> {
        self.push_instruction(
            Mnemonic::Assert(Assert {
                condition: self.loc(condition),
            }),
            0,
        )
    }
}

/// Module-path-only constructors and helpers: these mint whole functions or look
/// up/create blocks by machine address in the module registry, which only makes
/// sense over a `&mut Context`. A checked-out builder appends into an existing
/// owned block and mints owned blocks via [`Builder::get_or_make_local_label`].
impl<'str, 'ctx> Builder<'str, 'ctx, &'ctx mut Context<'str>> {
    /// The whole module `&Context` — module-builder-only (the lifter / lowering
    /// path); the pass builder reads shared state via [`Builder::shr`] and its
    /// own arenas via the read host.
    pub fn context(&self) -> &Context<'str> {
        self.block.host_ref()
    }

    /// Creates a builder positioned at the block for machine `address`, creating
    /// the block (and an anonymous host function if nothing is mapped) if needed.
    pub fn from_context<'m>(ctx: &'m mut Context<'str>, address: u64) -> Builder<'str, 'm> {
        let mut addresses = crate::address_index::AddressIndex::analyze(ctx);
        Self::from_context_indexed(ctx, &mut addresses, address)
    }

    /// Indexed construction variant of [`from_context`](Self::from_context).
    pub fn from_context_indexed<'m>(
        ctx: &'m mut Context<'str>,
        addresses: &mut crate::address_index::AddressIndex,
        address: u64,
    ) -> Builder<'str, 'm> {
        use crate::address_index::AddressTarget;

        let block_id = match addresses.get(address) {
            Some(AddressTarget::Function(function)) => {
                match FunctionBody::from_id(ctx, function).root() {
                    Some(root) => root.id,
                    None => ctx.get_or_make_block_indexed(addresses, address, function),
                }
            }
            Some(AddressTarget::Block(block)) => block,
            None => {
                let function = FunctionBody::make(ctx, Cow::Owned(format!("blk_{address:x}")))
                    .expect("anon host function")
                    .id;
                ctx.get_or_make_block_indexed(addresses, address, function)
            }
        };
        let mut builder = Builder::from_block(BasicBlock::from_id_mut(ctx, block_id));
        builder.set_address(address);
        builder
    }

    fn ensure_created_block_in_function(&mut self, block: BlockId) {
        if let Some(mut function) = self.block.parent_mut() {
            function.add_block(block);
        }
    }

    /// Gets or creates a block for a given machine address.
    pub fn get_or_make_block(&mut self, addr: u64) -> BlockId {
        let func = self.block.id.func;
        let id = self.context_mut().get_or_make_block(addr, func);

        if BasicBlock::from_id(self.context(), id).parent().is_none() {
            self.ensure_created_block_in_function(id);
        }

        id
    }

    /// Indexed construction variant of
    /// [`get_or_make_block`](Self::get_or_make_block).
    pub fn get_or_make_block_indexed(
        &mut self,
        addresses: &mut crate::address_index::AddressIndex,
        addr: u64,
    ) -> BlockId {
        let function = self.block.id.func;
        let id = self
            .context_mut()
            .get_or_make_block_indexed(addresses, addr, function);
        if BasicBlock::from_id(self.context(), id).parent().is_none() {
            self.ensure_created_block_in_function(id);
        }
        id
    }

    /// Gets or creates a function whose root is the local label block for `name`.
    pub fn get_or_make_local_function(&mut self, name: Cow<'str, str>) -> FunctionId {
        let existing = FunctionBody::from_name(self.context(), &name).map(|f| f.id);
        if let Some(fid) = existing {
            fid
        } else {
            FunctionBody::make(self.context_mut(), name)
                .expect("Name was checked above")
                .id
        }
    }

    /// Declares a new parameter on the current block. Returns a mutable reference
    /// whose `ValueId` can be used as an operand.
    pub fn push_param(&mut self, size: usize) -> BlockParamMutRef<'str, '_> {
        self.block.push_param(size)
    }

    /// If this block is not terminated, add a jump to the given address as a
    /// terminator instruction; the builder is then safe to drop.
    pub fn finalize(mut self, addr: u64) {
        if !self.block.is_terminated() {
            let target = self.get_or_make_block(addr);
            let branch = self.push_branch(target).id;
            if let Some(addr) = self.address {
                Instruction::from_id_mut(self.context_mut(), branch).set_address(addr);
            }
        }
    }
}

impl<'str, 'ctx, Ctx: BuilderBacking<'str>> Drop for Builder<'str, 'ctx, Ctx> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }

        if !self.is_terminated &&
        // I don't see how this can happen, but just in case, we also check if the block is actually terminated, to avoid panicking when dropping a builder that has already been finalized
        !self.is_terminated()
        {
            // This is terrible and should be done at compile time, but this is seamingly impossible in rust ?
            // panic!(
            //     "Builder must be finalized before drop, if you need to drop without finalizing, call `dont_finalize()` on the builder first"
            // );
        }
    }
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;
    use crate::context::Context;

    #[test]
    fn checked_builder_matches_module_builder() {
        use crate::value::{
            FunctionId, FunctionRef, block::BasicBlock, function::FunctionBody,
            util::pass_backing::PassBacking,
        };

        // The same body over any host: consts and a couple of binops (exercising
        // type + literal minting through the interners' `&self` paths), then a
        // branch to a freshly minted local-label block. No varnode/temp-space
        // minting — a checked-out host has read-only shared access.
        fn body<'str, 'ctx, Ctx: BuilderBacking<'str>>(b: &mut Builder<'str, 'ctx, Ctx>) {
            let c1 = b.shr().get_const(7, 8);
            let c2 = b.shr().get_const(9, 8);
            let sum = b.push_add(c1, c2).id();
            let _doubled = b.push_add(sum, sum).id();
            let lbl = b.get_or_make_local_label("next".into());
            b.push_branch(lbl);
        }

        // Structural snapshot: per block (name, per-instruction mnemonic Debug —
        // which includes the operand ids — and sorted successor block names).
        type BSnap = Vec<(String, Vec<String>, Vec<String>)>;
        fn snap(ctx: &Context, fid: FunctionId) -> BSnap {
            FunctionRef::from_id(ctx, fid)
                .blocks()
                .map(|blk| {
                    let name = blk.name().unwrap_or("?").to_string();
                    let insns: Vec<String> = blk
                        .instructions()
                        .map(|i| format!("{:?}", i.mnemonic()))
                        .collect();
                    let mut succ: Vec<String> = blk
                        .successors()
                        .map(|(_, s)| {
                            BasicBlock::from_id(ctx, s)
                                .name()
                                .unwrap_or("?")
                                .to_string()
                        })
                        .collect();
                    succ.sort();
                    (name, insns, succ)
                })
                .collect()
        }

        // ---- (a) module builder ---------------------------------------------
        let mut ctx_a = Context::new();
        let fid_a = FunctionBody::make(&mut ctx_a, "foo".into()).unwrap().id;
        let entry_a = FunctionBody::from_id_mut(&mut ctx_a, fid_a).make_root().id;
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut ctx_a, entry_a));
            body(&mut b);
        }
        let snap_a = snap(&ctx_a, fid_a);
        assert!(
            snap_a.iter().any(|(_, i, _)| !i.is_empty()),
            "sanity: built IR"
        );

        // ---- (b) checked-out builder ----------------------------------------
        let mut ctx_b = Context::new();
        let fid_b = FunctionBody::make(&mut ctx_b, "foo".into()).unwrap().id;
        let entry_b = FunctionBody::from_id_mut(&mut ctx_b, fid_b).make_root().id;
        {
            let mut host =
                PassBacking::new(&mut ctx_b.bodies[fid_b], &ctx_b.shared, &ctx_b.interfaces);
            let mut b = Builder::from_block(BaseRef::new(host.reborrow(), entry_b));
            body(&mut b);
        }
        let snap_b = snap(&ctx_b, fid_b);

        assert_eq!(
            snap_a, snap_b,
            "a body built through a checked-out builder must match the module-built body"
        );
    }

    /// `map` preserves its source's sequence kind: mapping over a `List<T>`
    /// (e.g. a `take_while` result) yields a `List<U>`, not a fixed array.
    #[test]
    fn map_over_a_list_yields_a_list() {
        use crate::value::{FunctionBody, insn::Return};

        let mut ctx = Context::new();
        let i8 = ctx.shared.types.get_or_make_int(1);

        // body: fn(i8) -> i8 returning its param (so the map result elem is i8).
        let body = FunctionBody::make(&mut ctx, "body".into()).unwrap().id;
        let broot = FunctionBody::from_id_mut(&mut ctx, body).make_root().id;
        let bp = BasicBlock::from_id_mut(&mut ctx, broot).push_param(1).id;
        let dummy = ctx.get_const(0, 8).id();
        let ret = InstructionRef::from_mnemonic_with_type(
            &mut ctx,
            body,
            Mnemonic::Return(Return {
                ptr: dummy.localize(body),
                value: Some(ValueId::BlockParam(bp).localize(body)),
            }),
            i8,
        )
        .id;
        BasicBlock::from_id_mut(&mut ctx, broot).push_insn(ret);

        // host: a value typed `List<i8>` (bound 4) to map over.
        let host = FunctionBody::make(&mut ctx, "host".into()).unwrap().id;
        let hentry = FunctionBody::from_id_mut(&mut ctx, host).make_root().id;
        let list_ty = ctx.shared.types.get_or_make_list(i8, 4);
        let src_pid = BasicBlock::from_id_mut(&mut ctx, hentry).push_param(4).id;
        ctx.block_param_mut(src_pid).type_id = list_ty;
        let src = ValueId::BlockParam(src_pid);

        let map_ty = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, hentry));
            b.push_map(body, src, Vec::new()).type_id()
        };

        assert_eq!(
            ctx.shared.types.array_of(map_ty),
            None,
            "map of a list is not an array"
        );
        assert_eq!(
            ctx.shared.types.list_of(map_ty),
            Some((i8, Some(4))),
            "map of List<i8> (bound 4) is List<i8> (bound 4)"
        );
    }

    #[test]
    fn cfg_branch_adds_one_node_and_one_edge() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <entry>
                goto <done>;
            <done>
                goto <0x1001>;
        "
        );
        // entry + done + 1001 = 3 nodes; entry -> done, done -> 1001 = 2 edges
        assert_eq!(ctx.block_ids().len(), 3);
        assert_eq!(ctx.functions().flat_map(|f| f.edge_ids()).count(), 2);
    }

    #[test]
    fn cfg_cbranch_adds_two_edges() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;

            <entry>
                %c = load(cond:1, cond);
                if %c goto <then_lbl> else goto <else_lbl>;

            <then_lbl>
                goto <0x1001>;

            <else_lbl>
                goto <0x1001>;
        "
        );
        // entry + then_lbl + else_lbl + 1001 = 4 nodes
        // entry->then_lbl, entry->else_lbl, then_lbl->1001, else_lbl->1001 = 4 edges
        assert_eq!(ctx.block_ids().len(), 4);
        assert_eq!(ctx.functions().flat_map(|f| f.edge_ids()).count(), 4);
    }

    #[test]
    fn cfg_branchind_adds_node_but_no_outgoing_edge() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <entry>
                local i64 ptr;
                goto [ptr];
        "
        );

        assert_eq!(ctx.block_ids().len(), 1);
        assert_eq!(ctx.functions().flat_map(|f| f.edge_ids()).count(), 0);

        assert_eq!(BasicBlock::from_id(&ctx, entry).successors().count(), 0);
    }

    #[test]
    fn cfg_return_adds_node_but_no_outgoing_edge() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <entry>
                local i64 ptr;
                return at ptr;
        "
        );
        assert_eq!(ctx.block_ids().len(), 1);
        assert_eq!(ctx.functions().flat_map(|f| f.edge_ids()).count(), 0);
        assert_eq!(BasicBlock::from_id(&ctx, entry).successors().count(), 0);
    }

    #[test]
    fn cfg_multi_block_qcode_program() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 v;

            <entry>
                goto <body>;

            <body>
                i64 %v0 = i64 &v + i64 1;
                goto <0x1001>;
        "
        );

        // entry + body + 1001 = 3 nodes
        // entry -> body, body -> 1001 = 2 edges
        assert_eq!(ctx.block_ids().len(), 3);
        assert_eq!(ctx.functions().flat_map(|f| f.edge_ids()).count(), 2);
    }

    #[test]
    #[ignore = "This feature is WIP"]
    #[should_panic(
        expected = "Builder must be finalized before drop, if you need to drop without finalizing, call `dont_finalize()` on the builder first"
    )]
    fn test_builder_drop() {
        // Test code for Builder drop behavior
        // This test will fail to compile if the drop implementation panics as expected
        let mut ctx = Context::new();
        let block_id = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0, __f)
        };
        let block_ref = BasicBlock::from_id_mut(&mut ctx, block_id);

        {
            let _builder = Builder::from_block(block_ref);
            // Not finalizing the builder, should panic when dropped
        }
    }

    #[test]
    fn test_builder_finalize() {
        // Test code for Builder drop behavior
        // This test will compile and run without panicking because we finalize the builder properly
        let mut ctx = Context::new();
        let block_id = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0, __f)
        };
        let block_ref = BasicBlock::from_id_mut(&mut ctx, block_id);

        {
            let _builder = Builder::from_block(block_ref);
            _builder.finalize(0x1000);
        }
    }

    #[test]
    fn test_named_temp_duplicate() {
        let mut ctx = Context::new();
        let mut builder = Builder::from_context(&mut ctx, 0x1000);

        let value = builder.make_named_temp("dup".into(), 4);
        let other_value = builder.make_named_temp("dup".into(), 4);
        builder.finalize(0x1000);

        assert_eq!(Varnode::from_id(&ctx, value).name(), Some("dup"));
        assert_eq!(Varnode::from_id(&ctx, other_value).name(), Some("dup_1"));
    }

    #[test]
    fn same_label_temps_in_different_functions_are_isolated() {
        let mut ctx = Context::new();

        let first = {
            let mut builder = Builder::from_context(&mut ctx, 0x1000);
            let temp = builder.make_temp_labeled(7, 4);
            builder.finalize(0x1000);
            temp
        };
        let second = {
            let mut builder = Builder::from_context(&mut ctx, 0x2000);
            let temp = builder.make_temp_labeled(7, 4);
            builder.finalize(0x2000);
            temp
        };

        assert_ne!(first, second);
        let first = Varnode::from_id(&ctx, first);
        let second = Varnode::from_id(&ctx, second);
        assert_eq!((first.label(), second.label()), (Some(7), Some(7)));
        assert_ne!(first.space().id, second.space().id);
        assert!(matches!(first.space().ty, SpaceType::Temporary));
        assert!(matches!(second.space().ty, SpaceType::Temporary));
    }

    #[test]
    fn qcode_local_decl_creates_named_temp() {
        let mut ctx = Context::new();

        qcode!(ctx, "varnode i64 ptr; <block> goto <0x1001>;");

        let ptr = Varnode::from_id(&ctx, ptr);

        assert_eq!(ptr.size(), 8);
        assert_eq!(ptr.name(), Some("ptr"));
    }

    #[test]
    fn qcode_standalone_local_decl_creates_named_temp() {
        let mut ctx = Context::new();
        qcode!(ctx, "varnode i64 ptr; <block> goto <0x1001>;");

        let ptr = Varnode::from_id(&ctx, ptr);

        assert_eq!(ptr.size(), 8);
        assert_eq!(ptr.name(), Some("ptr"));
    }

    #[test]
    fn qcode_varnode_decl_before_entry_block() {
        let mut ctx = Context::new();
        qcode!(ctx, "varnode i64 ptr; <block> goto <0x1001>;");

        let ptr = Varnode::from_id(&ctx, ptr);

        assert_eq!(ptr.size(), 8);
        assert_eq!(ptr.name(), Some("ptr"));
    }

    #[test]
    fn push_param_via_builder_visible_on_block() {
        let mut ctx = Context::new();
        let block_id = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0x1000, __f)
        };
        let mut builder = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, block_id));

        let p0 = builder.push_param(8);
        let p0_id = p0.id;
        let p1 = builder.push_param(4);
        let p1_id = p1.id;

        unsafe { builder.dont_finalize() };
        drop(builder);

        let block = BasicBlock::from_id(&ctx, block_id);
        assert_eq!(block.num_params(), 2);
        let param_ids: Vec<_> = block.params().map(|p| p.id).collect();
        assert_eq!(param_ids, [p0_id, p1_id]);
        assert_eq!(block.instruction_ids().len(), 0);
    }

    #[test]
    fn push_branch_with_args_via_builder() {
        let mut ctx = Context::new();
        // A branch edge is intra-function: source and target live in one function.
        let f = ctx.anon_function();
        let src_id = ctx.get_or_make_block(0x1000, f);
        let dst_id = ctx.get_or_make_block(0x2000, f);

        let param_val = BasicBlock::from_id_mut(&mut ctx, dst_id).push_param(8).id();

        {
            let mut builder = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, src_id));
            builder.push_branch_with_args(dst_id, vec![param_val]);
        }

        let block = BasicBlock::from_id(&ctx, src_id);
        let last = block.iter().last().expect("branch was added");
        let crate::value::insn::Mnemonic::Branch(branch) = last.mnemonic() else {
            panic!("expected branch");
        };
        assert_eq!(branch.target, dst_id.local);
        assert_eq!(branch.args.len(), 1);
        assert_eq!(branch.args[0], param_val.strip_func());
    }

    #[test]
    fn test_builder_adds_address_to_qcode() {
        let mut ctx = Context::new();
        let id_42 = ctx.get_const(42, 8).id();

        let not_insn_id = {
            let mut builder = Builder::from_context(&mut ctx, 0x1000);
            let not_insn_id = builder.push_bit_negate(id_42).id;
            builder.finalize(0x1001);

            not_insn_id
        };

        let insn = Instruction::from_id(&ctx, not_insn_id);

        assert_eq!(insn.address().unwrap(), 0x1000);
    }

    #[test]
    fn from_context_materializes_root_in_registered_function_arena() {
        let mut ctx = Context::new();
        let func = FunctionBody::make_at_addr(&mut ctx, 0x2000, None).id;
        assert!(FunctionBody::from_id(&ctx, func).root().is_none());

        let block = {
            let builder = Builder::from_context(&mut ctx, 0x2000);
            builder.current_block()
        };

        assert_eq!(block.func, func);
        assert_eq!(
            FunctionBody::from_id(&ctx, func).root().map(|root| root.id),
            Some(block)
        );
        assert!(FunctionBody::from_name(&ctx, "blk_2000").is_none());
    }

    #[test]
    fn append_after_terminated_block_panic_includes_current_address() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut ctx = Context::new();
            let value = ctx.get_const(0, 1).id();
            let mut builder = Builder::from_context(&mut ctx, 0x4010);
            let target = builder.get_or_make_block(0x4020);

            builder.push_branch(target);
            builder.set_address(0x4015);
            builder.push_bit_negate(value);
        }));

        let panic = result.expect_err("append should panic after a terminator");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&'static str>().copied())
            .expect("panic should carry a string message");

        assert!(
            message.contains("cannot append instruction to a terminated block at 0x4015"),
            "unexpected panic message: {message}"
        );
    }

    #[test]
    fn push_copy_supports_partial_final_lane() {
        let mut ctx = Context::new();
        let block_id = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0x1000, __f)
        };

        {
            let mut builder = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, block_id));
            let src = builder.make_named_temp("src".into(), 9);
            let dst = builder.make_named_temp("dst".into(), 9);
            builder.push_copy(src.into(), dst);
            builder.finalize(0x1001);
        }

        let store_sizes = BasicBlock::from_id(&ctx, block_id)
            .iter()
            .filter_map(|insn| match insn.mnemonic() {
                Mnemonic::Store(store) => Some(store.size),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(store_sizes, [8, 1]);
    }

    // --- insert-point tests ---

    #[test]
    fn insert_point_to_start_prepends_before_existing_instruction() {
        let mut ctx = Context::new();
        let block_id = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0x1000, __f)
        };
        let val = ctx.get_const(0, 8).id();

        let existing_id = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, block_id));
            let id = b.push_bit_negate(val).id;
            unsafe { b.dont_finalize() };
            id
        };

        let prepended_id = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, block_id));
            b.set_insert_point_to_start();
            unsafe { b.dont_finalize() };
            b.push_bit_negate(val).id
        };

        let ids = BasicBlock::from_id(&ctx, block_id).instruction_ids();
        assert_eq!(ids, [prepended_id, existing_id]);
    }

    #[test]
    fn multiple_pushes_with_insert_point_to_start_preserve_push_order() {
        let mut ctx = Context::new();
        let block_id = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0x1000, __f)
        };
        let val = ctx.get_const(0, 8).id();

        let existing_id = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, block_id));
            let id = b.push_bit_negate(val).id;
            unsafe { b.dont_finalize() };
            id
        };

        let (id0, id1, id2) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, block_id));
            b.set_insert_point_to_start();
            unsafe { b.dont_finalize() };
            (
                b.push_bit_negate(val).id,
                b.push_bit_negate(val).id,
                b.push_bit_negate(val).id,
            )
        };

        let ids = BasicBlock::from_id(&ctx, block_id).instruction_ids();
        assert_eq!(ids, [id0, id1, id2, existing_id]);
    }

    #[test]
    fn insert_point_before_existing_instruction_inserts_before_target() {
        let mut ctx = Context::new();
        let block_id = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0x1000, __f)
        };
        let val = ctx.get_const(0, 8).id();

        let (first_id, target_id) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, block_id));
            unsafe { b.dont_finalize() };
            (b.push_bit_negate(val).id, b.push_bit_negate(val).id)
        };

        let (inserted0, inserted1) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, block_id));
            b.set_insert_point_before(target_id);
            unsafe { b.dont_finalize() };
            (b.push_bit_negate(val).id, b.push_bit_negate(val).id)
        };

        let ids = BasicBlock::from_id(&ctx, block_id).instruction_ids();
        assert_eq!(ids, [first_id, inserted0, inserted1, target_id]);
    }

    #[test]
    fn insert_point_to_start_allows_push_into_terminated_block() {
        let mut ctx = Context::new();
        qcode!(ctx, "<entry> goto <0x1001>;");

        let val = ctx.get_const(1, 1).id();
        let new_id = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, entry));
            b.set_insert_point_to_start();
            unsafe { b.dont_finalize() };
            b.push_bit_negate(val).id
        };

        let block = BasicBlock::from_id(&ctx, entry);
        assert_eq!(block.instruction_ids()[0], new_id);
        // The original branch terminator is still present
        assert!(block.is_terminated());
    }

    #[test]
    fn set_insert_point_to_end_restores_append_mode() {
        let mut ctx = Context::new();
        let block_id = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0x1000, __f)
        };
        let val = ctx.get_const(0, 8).id();

        let (first_id, middle_id, last_id) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, block_id));
            unsafe { b.dont_finalize() };
            let first = b.push_bit_negate(val).id; // appended → index 0
            b.set_insert_point_to_start();
            let middle = b.push_bit_negate(val).id; // inserted at 0, first shifts to 1
            b.set_insert_point_to_end();
            let last = b.push_bit_negate(val).id; // appended → index 2
            (first, middle, last)
        };

        let ids = BasicBlock::from_id(&ctx, block_id).instruction_ids();
        assert_eq!(ids, [middle_id, first_id, last_id]);
    }
}
