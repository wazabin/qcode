//! Fluent IR builder: emit instructions into a [`BasicBlock`].
//!
//! The [`Builder`] is the primary way to construct IR. It holds a mutable
//! reference to a block inside a [`Context`] and exposes typed `push_*` methods
//! for every instruction kind.
//!
//! Terminating the block is the caller's responsibility ([`Builder::finalize`]
//! pushes the final branch for the common case); the invariant that every
//! rostered block ends in a terminator is enforced by the IR verifier, not at
//! builder drop. Appending *past* a terminator, however, panics immediately.
//! A builder may freely be dropped mid-block — e.g. after splicing
//! instructions before an existing anchor via
//! [`Builder::set_insert_point_before`].
//!
//! # Typical usage
//!
//! ```rust,ignore
//! use qcode_core::{context::Context, builder::Builder};
//!
//! let mut ctx = Context::new();
//!
//! // Create a builder positioned at machine address 0x1000.
//! let mut b = (&mut ctx).builder_at(0x1000);
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

use rustc_hash::FxHashMap as HashMap;

use crate::{
    space::{LocalMemorySpaceId, SPACE_CONST, Space, SpaceId, SpaceType},
    types::{AggregateField, TypeId},
    value::{
        BodyView, FunctionBody, Instruction, LocalBlockId, LocalValueId, Temp, TempId, TempSpace,
        ValueId, ValueRef,
        block::{BasicBlock, BlockId},
        block_param::{BlockParam, BlockParamId},
        function::FunctionId,
        insn::{
            Apply, Assert, Binary, Binop, Branch, BranchInd, CBranch, Call, CallInd, Callee, Carry,
            Extract, FloatBinop, FloatToFloat, FloatToInt, Gep, InstructionId, InstructionRef,
            IntBinop, IntToFloat, IntrinsicApp, IntrinsicId, IsFloatNaN, Load, LocalInsnId,
            LzCount, Map, Mnemonic, PCodeOp, PCodeOpId, PopCount, Range, Return, ReturnValue,
            SBorrow, SCarry, Scan, Sext, Store, TailCall, Tuple, Unary, Unop, Zext,
        },
        varnode::Varnode,
    },
};

#[cfg(test)]
use crate::value::TempRef;

/// A builder for constructing instructions in a block.
/// This provides a convenient API for creating instructions, and automatically
/// manages temporary values and labels.
pub struct Builder<'str, 'ctx> {
    body: &'ctx mut FunctionBody<'str>,
    shared: &'ctx crate::context::Shared<'str>,
    interfaces:
        &'ctx jstd::registry::Registry<FunctionId, crate::value::function::FunctionInterface<'str>>,

    /// The block currently receiving emitted instructions, as a body-local id.
    /// The engine never routes through its owning `FunctionId`, so a detached
    /// (id-less) body can be built. Composite callers read it via
    /// [`Builder::current_block`].
    pub(crate) block: LocalBlockId,

    /// Converts from names to value IDs in the current scope.
    namespace: HashMap<Cow<'str, str>, ValueId>,

    /// Names of local labels to their corresponding body-local block IDs.
    local_labels: HashMap<Cow<'str, str>, LocalBlockId>,

    /// The address at which instructions are added
    address: Option<u64>,

    /// Is the block terminated, i.e. does it end with a terminator
    /// If it is not the case, the block might be invalid
    pub(crate) is_terminated: bool,

    /// Explicit insert position for new instructions.
    ///
    /// `None` (default) appends to the end of the block.
    /// `Some(n)` inserts at index `n` and auto-advances after each push,
    /// so consecutive pushes form a contiguous sequence starting at `n`.
    insert_point: Option<usize>,
}

/// Generates a canonical comparison method and its "greater-than" mirror
/// (operands swapped), each with a composite skin and a body-local sibling.
macro_rules! cmp_pair {
    ($fwd:ident, $fwd_local:ident, $rev:ident, $rev_local:ident, $op:expr) => {
        pub fn $fwd(
            &mut self,
            lhs: ValueId,
            rhs: ValueId,
        ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
            let (lhs, rhs) = (self.loc(lhs), self.loc(rhs));
            let local = self.$fwd_local(lhs, rhs);
            self.insn_ref(local)
        }
        pub fn $fwd_local(&mut self, lhs: LocalValueId, rhs: LocalValueId) -> LocalInsnId {
            self.push_binop_local($op, lhs, rhs, Some(1))
        }
        pub fn $rev(
            &mut self,
            lhs: ValueId,
            rhs: ValueId,
        ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
            let (lhs, rhs) = (self.loc(lhs), self.loc(rhs));
            let local = self.$rev_local(lhs, rhs);
            self.insn_ref(local)
        }
        pub fn $rev_local(&mut self, lhs: LocalValueId, rhs: LocalValueId) -> LocalInsnId {
            self.push_binop_local($op, rhs, lhs, Some(1))
        }
    };
}

/// Generates a simple unary-op push method (composite skin + body-local sibling)
/// that delegates to [`push_unop_local`](Builder::push_unop_local).
macro_rules! unop_leaf {
    ($(#[$m:meta])* $name:ident, $lname:ident, $op:expr) => {
        $(#[$m])*
        pub fn $name(&mut self, src: ValueId) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
            let src = self.loc(src);
            let local = self.$lname(src);
            self.insn_ref(local)
        }
        /// Body-local sibling.
        pub fn $lname(&mut self, src: LocalValueId) -> LocalInsnId {
            self.push_unop_local($op, src)
        }
    };
}

/// Generates a simple binary-op push method (composite skin + body-local sibling)
/// that delegates to [`push_binop_local`](Builder::push_binop_local) with no
/// forced result size.
macro_rules! binop_leaf {
    ($(#[$m:meta])* $name:ident, $lname:ident, $op:expr, $size:expr) => {
        $(#[$m])*
        pub fn $name(
            &mut self,
            lhs: ValueId,
            rhs: ValueId,
        ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
            let (lhs, rhs) = (self.loc(lhs), self.loc(rhs));
            let local = self.$lname(lhs, rhs);
            self.insn_ref(local)
        }
        /// Body-local sibling.
        pub fn $lname(&mut self, lhs: LocalValueId, rhs: LocalValueId) -> LocalInsnId {
            self.push_binop_local($op, lhs, rhs, $size)
        }
    };
}

/// Generates a size-taking conversion push method (composite skin + body-local
/// sibling) whose mnemonic variant and payload type share the ident `$variant`
/// and carry a `{ src, size }` shape.
macro_rules! conv_leaf {
    ($name:ident, $lname:ident, $err:literal, $variant:ident) => {
        pub fn $name(
            &mut self,
            src: ValueId,
            size: usize,
        ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
            let src = self.loc(src);
            let local = self.$lname(src, size);
            self.insn_ref(local)
        }
        /// Body-local sibling.
        pub fn $lname(&mut self, src: LocalValueId, size: usize) -> LocalInsnId {
            assert!(!matches!(src, LocalValueId::Varnode(_)), $err);
            self.store_insn(Mnemonic::$variant($variant { src, size }), size)
        }
    };
}

impl<'str, 'ctx> Builder<'str, 'ctx> {
    fn fresh_temp_space(&mut self, name: Option<&str>) -> crate::value::TempSpaceId {
        let (word_size, addr_size) = {
            let default = self.shr().space(self.shr().default_space);
            (default.word_size, default.addr_size)
        };
        self.body
            .push_temp_space(TempSpace::new(name, word_size, addr_size))
    }

    /// Creates an anonymous body-local temporary memory value.
    pub fn make_temp(&mut self, size: usize) -> TempId {
        let space = self.fresh_temp_space(None);
        self.body.push_temp(Temp::new(0, size, space.local))
    }

    /// Creates a named body-local temporary memory value.
    pub fn make_named_temp(&mut self, name: Cow<'str, str>, size: usize) -> TempId {
        let unique = self.body.names.unique(name);
        let space = self.fresh_temp_space(Some(unique.as_ref()));
        self.body
            .push_temp(Temp::new(0, size, space.local).with_name(unique))
    }

    /// Creates a body-local temporary identified by a SLEIGH local label.
    pub fn make_temp_labeled(&mut self, label: u32, size: usize) -> TempId {
        let space = self.fresh_temp_space(None);
        let mut temp = Temp::new(0, size, space.local);
        temp.label = Some(label);
        self.body.push_temp(temp)
    }

    /// Creates a builder positioned at `block`.
    ///
    /// The block is borrowed mutably for the lifetime `'ctx`. New instructions
    /// will be appended to the end of `block`.
    pub fn new(
        body: &'ctx mut FunctionBody<'str>,
        shared: &'ctx crate::context::Shared<'str>,
        interfaces: &'ctx jstd::registry::Registry<
            FunctionId,
            crate::value::function::FunctionInterface<'str>,
        >,
        block: BlockId,
    ) -> Self {
        assert_eq!(
            body.id(),
            block.func,
            "Builder block must belong to its body"
        );
        Self::new_local(body, shared, interfaces, block.local)
    }

    /// Creates a builder positioned at a **body-local** block, without ever
    /// consulting the body's registry identity. This is the id-less constructor:
    /// it works on a detached (uninstalled) body just as well as an installed
    /// one. `is_terminated` is read straight from the block's own arena (its last
    /// instruction's mnemonic), never through the composite `BodyView` path.
    pub fn new_local(
        body: &'ctx mut FunctionBody<'str>,
        shared: &'ctx crate::context::Shared<'str>,
        interfaces: &'ctx jstd::registry::Registry<
            FunctionId,
            crate::value::function::FunctionInterface<'str>,
        >,
        block: LocalBlockId,
    ) -> Self {
        let is_terminated = body.blocks[block]
            .instructions
            .last()
            .is_some_and(|&i| body.insns[i].mnemonic().is_terminator());
        Self {
            body,
            shared,
            interfaces,
            is_terminated,
            block,
            namespace: HashMap::default(),
            local_labels: HashMap::default(),
            address: None,
            insert_point: None,
        }
    }

    /// This builder's owning function id. Skin-only: composite entry points call
    /// this to qualify local ids back to the boundary [`ValueId`] surface. Never
    /// invoked on the id-less (`new_local` + `push_*_local`) path.
    #[inline]
    fn func(&self) -> FunctionId {
        self.body.id()
    }

    /// Qualify a body-local instruction id into an [`InstructionRef`]. Skin-only.
    fn insn_ref(&self, local: LocalInsnId) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let id = InstructionId::new(self.func(), local);
        InstructionRef::new(self.view(), id)
    }

    /// A `Copy` read view over the builder's backing, for arena reads. The builder
    /// reads through the backing's static [`QCodeView`].
    pub fn view(&self) -> BodyView<'_, 'str> {
        BodyView::new(&*self.body, self.shared, self.interfaces)
    }

    /// Returns `true` if the current block ends with a terminator instruction.
    pub fn is_terminated(&self) -> bool {
        self.block_is_terminated(self.block)
    }

    /// Whether a body-local block ends with a terminator, read straight from the
    /// arenas (id-less).
    fn block_is_terminated(&self, block: LocalBlockId) -> bool {
        self.body.blocks[block]
            .instructions
            .last()
            .is_some_and(|&i| self.body.insns[i].mnemonic().is_terminator())
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
        let index = self.body.blocks[self.block]
            .instructions
            .iter()
            .position(|&id| id == before_id.local)
            .expect("before_id not found in block");
        self.insert_point = Some(index);
    }

    /// Resets the insert point to append mode (the default).
    pub fn set_insert_point_to_end(&mut self) {
        self.insert_point = None;
    }

    /// Gets a sub-value from a given value, specified by a byte range.
    pub fn get_range(
        &mut self,
        src: ValueId,
        range: std::ops::Range<usize>,
    ) -> Option<ValueRef<'str, '_, BodyView<'_, 'str>>> {
        let src = self.loc(src);
        let dst = self.get_range_local(src, range)?;
        Some(self.get_value(dst.qualify(self.func())))
    }

    /// Body-local core of [`get_range`](Self::get_range): folds a literal/temp
    /// sub-range in place and emits a `Range` instruction for varnode/instruction
    /// sources. Operands and result are body-local; no registry identity is used.
    pub fn get_range_local(
        &mut self,
        src: LocalValueId,
        range: std::ops::Range<usize>,
    ) -> Option<LocalValueId> {
        if range.is_empty() {
            return None;
        }

        let dst = match src {
            LocalValueId::Literal(lit) => {
                let value = self.shr().values.literals[lit].value;
                let id = self.shr().get_const(value, range.len());
                id.strip_func()
            }

            LocalValueId::Varnode(vid) => {
                let size = Varnode::from_id(self.shr(), vid).size();
                if range.end > size {
                    return None;
                }
                let local = self.store_insn(
                    Mnemonic::Range(Range {
                        src,
                        start: range.start,
                        size: range.len(),
                    }),
                    range.len(),
                );
                LocalValueId::Instruction(local)
            }

            LocalValueId::Temp(tlocal) => {
                let (address, size, space) = {
                    let temp = &self.body.temps[tlocal];
                    (temp.address, temp.size, temp.space)
                };
                if range.end > size {
                    return None;
                }
                let temp = Temp::new(address + range.start as i64, range.len(), space);
                let local = self.body.temps.push(temp);
                LocalValueId::Temp(local)
            }

            LocalValueId::Instruction(_) => {
                let size = self.lsize_of(src);
                if range.end > size {
                    return None;
                }
                let local = self.store_insn(
                    Mnemonic::Range(Range {
                        src,
                        start: range.start,
                        size: range.len(),
                    }),
                    range.len(),
                );
                LocalValueId::Instruction(local)
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
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let local = self.push_range_local(self.loc(src), start, size);
        self.insn_ref(local)
    }

    /// Body-local core of [`push_range`](Self::push_range).
    pub fn push_range_local(
        &mut self,
        src: LocalValueId,
        start: usize,
        size: usize,
    ) -> LocalInsnId {
        self.store_insn(Mnemonic::Range(Range { src, start, size }), size)
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
        self.switch_to_block_local(block.local);
    }

    /// Reposition the builder onto a body-local block (id-less).
    pub fn switch_to_block_local(&mut self, block: LocalBlockId) {
        self.block = block;
        self.is_terminated = self.block_is_terminated(block);
    }

    /// The block the builder is currently appending to.
    pub fn current_block(&self) -> BlockId {
        BlockId::new(self.func(), self.block)
    }

    /// Gets the ID of a value in the current namespace
    pub fn try_get_value(&self, name: &str) -> Option<ValueRef<'str, '_, BodyView<'_, 'str>>> {
        self.namespace.get(name).map(|&id| self.get_value(id))
    }

    /// The module's shared IR state (read) — types/literals/spaces/registers.
    pub fn shr(&self) -> &crate::context::Shared<'str> {
        self.shared
    }

    /// Retype instruction `local`'s result as a pointer into `space`. Register
    /// spaces are left untyped (pointer arithmetic is not allowed there). A
    /// body-local **temporary** space needs the registry identity to name its
    /// owner; on a detached body that retype is skipped (an install-time nicety,
    /// like debug naming). Body-local and id-free for the shared-space path.
    fn set_insn_space_local(&mut self, local: LocalInsnId, space: LocalMemorySpaceId) {
        if space.shared().is_some_and(|space| {
            matches!(Space::from_id(self.shr(), space).ty, SpaceType::Register)
        }) {
            return;
        }
        let qualified = match space {
            LocalMemorySpaceId::Shared(id) => crate::space::MemorySpaceId::Shared(id),
            LocalMemorySpaceId::Temp(t) => match self.body.try_id() {
                Some(func) => {
                    crate::space::MemorySpaceId::Temp(crate::value::TempSpaceId::new(func, t))
                }
                None => return,
            },
        };
        let cur_type = self.body.insns[local].type_id;
        let size = self.shr().types.size_of(cur_type);
        let type_id = self.shr().types.get_or_make_space_address(size, qualified);
        self.body.insns[local].type_id = type_id;
    }

    /// Rename instruction `local`'s result. On an installed body this registers
    /// the (function-local) name in the owning function's table (uniqueness
    /// enforced); on a detached body it sets only the arena field — the source of
    /// truth for rendering — since the local name table keys on the registry id.
    fn rename_insn_local(
        &mut self,
        local: LocalInsnId,
        name: Cow<'str, str>,
    ) -> crate::error::Result<()> {
        if self.body.try_id().is_some() {
            let id = InstructionId::new(self.func(), local);
            let old = self.body.insns[local].name.clone();
            self.body.register_local_name(
                self.shared,
                ValueId::Instruction(id),
                name.clone(),
                old.as_deref(),
            )?;
        }
        self.body.insns[local].name = Some(name);
        Ok(())
    }

    /// Rename an instruction's result — composite skin over
    /// [`rename_insn_local`](Self::rename_insn_local).
    pub(crate) fn rename_insn(
        &mut self,
        id: InstructionId,
        name: Cow<'str, str>,
    ) -> crate::error::Result<()> {
        self.rename_insn_local(id.local, name)
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
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let local = self.store_insn_with_type(mnemonic, type_id);
        self.insn_ref(local)
    }

    /// Body-local instruction-storage core: mint an `Int(size)`-typed
    /// instruction and append it to the working block.
    #[track_caller]
    fn store_insn(&mut self, mnemonic: Mnemonic, size: usize) -> LocalInsnId {
        let type_id = self.shr().types.get_or_make_int(size);
        self.store_insn_with_type(mnemonic, type_id)
    }

    /// Body-local instruction-storage core: appends `mnemonic` (typed `type_id`)
    /// into the working block's arena, records reverse-uses, honours the address
    /// and insert-point cursors, and returns the fresh body-local id. Consults no
    /// registry identity, so it drives an id-less (detached) body.
    #[track_caller]
    fn store_insn_with_type(&mut self, mnemonic: Mnemonic, type_id: TypeId) -> LocalInsnId {
        if self.is_terminated && self.insert_point.is_none() {
            let block_address = self.body.blocks[self.block].address;
            if let Some(address) = self.address.or(block_address) {
                panic!("cannot append instruction to a terminated block at {address:#x}");
            }
            panic!("cannot append instruction to a terminated block");
        }

        let block = self.block;
        let insn = Instruction::new(type_id, mnemonic);
        // Inlined `FunctionBody::push_insn`, id-less: append to the arena and
        // record each operand's reverse-use, keyed by its body-local form.
        let args: Vec<LocalValueId> = insn.mnemonic().args().into_iter().collect();
        let local = self.body.insns.push(insn);
        for arg in args {
            self.body.users.entry(arg).or_default().push(local);
        }

        if let Some(address) = self.address {
            self.body.insns[local].set_address(address);
        }

        match self.insert_point {
            None => {
                self.body.insns[local].parent = Some(block);
                self.body.blocks[block].instructions.push(local);
            }
            Some(ref mut pos) => {
                let index = *pos;
                self.body.insns[local].parent = Some(block);
                self.body.blocks[block].instructions.insert(index, local);
                *pos += 1;
            }
        }

        local
    }

    fn get_value(&self, id: ValueId) -> ValueRef<'str, '_, BodyView<'_, 'str>> {
        // Route through the host's read view so a checked-out builder resolves its
        // own function's SSA values (which live in the owned function, not the
        // shared context) correctly.
        ValueRef::from_view(self.view(), id)
    }

    /// Localize a qualified operand id for storage in a mnemonic. Skin-only: the
    /// composite entry points call this to drop the (installed) owning
    /// `FunctionId` before handing operands to a body-local core.
    fn loc(&self, id: ValueId) -> LocalValueId {
        id.localize(self.func())
    }

    /// Localize a whole operand list (call/branch/tuple/intrinsic args). Skin-only.
    fn loc_vec(&self, ids: Vec<ValueId>) -> Vec<LocalValueId> {
        let func = self.func();
        ids.into_iter().map(|v| v.localize(func)).collect()
    }

    /// The stored type of `id`, host-routed — composite skin over
    /// [`lstored_type_of`](Self::lstored_type_of).
    pub(crate) fn stored_type_of(&self, id: ValueId) -> Option<TypeId> {
        self.lstored_type_of(self.loc(id))
    }

    /// The result type of a body-local operand (id-less; see
    /// [`FunctionBody::local_type_of`]).
    fn ltype_of(&self, id: LocalValueId) -> TypeId {
        self.body.local_type_of(self.shared, id)
    }

    /// The stored type of a body-local operand, or `None` (id-less; see
    /// [`FunctionBody::local_stored_type_of`]).
    fn lstored_type_of(&self, id: LocalValueId) -> Option<TypeId> {
        self.body.local_stored_type_of(self.shared, id)
    }

    /// The size in bytes of a body-local operand.
    fn lsize_of(&self, id: LocalValueId) -> usize {
        self.shr().types.size_of(self.ltype_of(id))
    }

    /// The address-space provenance of a body-local operand, if any. Only
    /// varnodes and space-pointer instructions carry one (mirrors
    /// [`ValueRef::space`]).
    fn lspace_of(&self, id: LocalValueId) -> Option<SpaceId> {
        match id {
            LocalValueId::Varnode(vid) => Some(Varnode::from_id(self.shr(), vid).space().id),
            LocalValueId::Instruction(local) => {
                let ty = self.body.insns[local].type_id;
                self.shr().types.space_of(ty).and_then(|m| m.shared())
            }
            _ => None,
        }
    }

    pub(crate) fn set_insn_type(&mut self, id: InstructionId, type_id: TypeId) {
        self.body.insn_mut(id).type_id = type_id;
    }

    pub(crate) fn constrain_param_size(&mut self, id: BlockParamId, size: usize) {
        self.body.block_param_mut(id).type_id = self.shared.types.get_or_make_int(size);
    }

    pub fn set_param_type(&mut self, id: BlockParamId, type_id: TypeId) {
        self.body.block_param_mut(id).type_id = type_id;
    }

    pub(crate) fn add_cfg_edge(&mut self, from: BlockId, to: BlockId) {
        self.body.add_cfg_edge(from, to);
    }

    /// The common address-space provenance of two pointer-arithmetic operands.
    ///
    /// Returns the space carried by whichever operand has one (varnodes carry
    /// their space; pointer-typed instructions carry theirs), or `None` when the
    /// two disagree or neither has a space.
    fn merge_space_ids(&self, lhs: LocalValueId, rhs: LocalValueId) -> Option<SpaceId> {
        match (self.lspace_of(lhs), self.lspace_of(rhs)) {
            (Some(lhs), Some(rhs)) if lhs == rhs => Some(lhs),
            (Some(space), None) | (None, Some(space)) => Some(space),
            _ => None,
        }
    }

    fn is_literal(&self, id: LocalValueId) -> bool {
        matches!(id, LocalValueId::Literal(_))
    }

    fn coerce_literal_size(&mut self, id: LocalValueId, size: usize) -> LocalValueId {
        let LocalValueId::Literal(lit_id) = id else {
            return id;
        };
        let literal = self.shr().values.literals[lit_id].clone();
        let current_size = self.shr().types.size_of(literal.type_id);
        if current_size == size || literal.symbolic.is_some() {
            return id;
        }
        self.shr().get_const(literal.value, size).strip_func()
    }

    pub fn get_or_make_local_label(&mut self, name: Cow<'str, str>) -> BlockId {
        if let Some(&local) = self.local_labels.get(name.as_ref()) {
            return BlockId::new(self.func(), local);
        }
        // SLEIGH pcode label names (e.g. `start`, `end`) are only unique within a
        // single instruction's lowering, but block names are function-scoped.
        // Deduplicate with a numeric suffix; the `local_labels` map stays keyed by
        // the original name so within-instruction references still resolve here.
        // Routed through the host so a checked-out builder mints the block into its
        // owned function's arena (and registers the name in that function's table).
        let unique_name = self.body.names.unique(name.clone());
        let id = self.body.push_block(BasicBlock::detached());
        self.body
            .register_local_name(
                self.shared,
                ValueId::BasicBlock(id),
                unique_name.clone(),
                None,
            )
            .expect("name was deduplicated");
        self.body.block_mut(id).set_name(Some(unique_name));
        self.local_labels.insert(name, id.local);
        id
    }

    /// Resolve a canonical textual `$tempN` token to one body-local temporary
    /// space, creating it on first use. This is a lowering compatibility seam;
    /// analysis and lifter producers append their spaces directly to the body.
    pub fn get_or_make_local_temp_space(&mut self, name: &str) -> LocalMemorySpaceId {
        let (word_size, addr_size) = {
            let default = self.shr().space(self.shr().default_space);
            (default.word_size, default.addr_size)
        };
        let body = &mut *self.body;
        for index in 0..body.temp_spaces.len() {
            let local = crate::value::LocalTempSpaceId::from(index);
            if body.temp_spaces[local].name.as_deref() == Some(name) {
                return LocalMemorySpaceId::Temp(local);
            }
        }
        let id = body.push_temp_space(TempSpace::new(Some(name), word_size, addr_size));
        LocalMemorySpaceId::Temp(id.local)
    }

    /// Ensures an operand is not a memory value.
    /// If the operand is a shared varnode or body-local temporary, emits a load
    /// and returns its SSA result. Other values are already directly usable.
    pub fn ensure_local(&mut self, src: ValueId) -> ValueId {
        let src = self.loc(src);
        self.ensure_local_local(src).qualify(self.func())
    }

    /// Body-local core of [`ensure_local`](Self::ensure_local): loads a shared
    /// varnode or body-local temporary into an SSA value, giving the load a
    /// related debug name; other operands pass through. Id-free.
    pub fn ensure_local_local(&mut self, src: LocalValueId) -> LocalValueId {
        match src {
            LocalValueId::Varnode(vid) => {
                let node = Varnode::from_id(self.shr(), vid);
                let size = node.size();
                let space = node.space().id;
                let name = node.name().map(str::to_owned);
                let id = self.push_load_local::<false>(src, size, space);

                // If the varnode has a name, give the load a related name.
                if let (Some(name), LocalValueId::Instruction(local)) = (name, id) {
                    let unique = self.body.names.unique(name.to_lowercase().into());
                    self.rename_insn_local(local, unique)
                        .expect("This name was deduplicated");
                }

                id
            }

            LocalValueId::Temp(tlocal) => {
                let (size, space, name) = {
                    let temp = &self.body.temps[tlocal];
                    (
                        temp.size,
                        LocalMemorySpaceId::Temp(temp.space),
                        temp.name.clone(),
                    )
                };
                let id = self.push_load_local::<false>(src, size, space);
                let LocalValueId::Instruction(local) = id else {
                    unreachable!("non-constant temporary load creates an instruction");
                };

                if let Some(name) = name {
                    let unique = self.body.names.unique(Cow::Owned(name.to_lowercase()));
                    self.rename_insn_local(local, unique)
                        .expect("temporary load name was deduplicated");
                }

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
        src: ValueId,
        size: usize,
        space: impl Into<LocalMemorySpaceId>,
    ) -> ValueRef<'str, '_, BodyView<'_, 'str>> {
        let src = self.loc(src);
        let id = self.push_load_local::<CHECK_LOCAL>(src, size, space);
        self.get_value(id.qualify(self.func()))
    }

    /// Body-local core of [`push_load`](Self::push_load). Operand and result are
    /// body-local; consults no registry identity.
    #[track_caller]
    pub fn push_load_local<const CHECK_LOCAL: bool>(
        &mut self,
        mut src: LocalValueId,
        size: usize,
        space: impl Into<LocalMemorySpaceId>,
    ) -> LocalValueId {
        let space = space.into();
        if CHECK_LOCAL {
            src = self.ensure_local_local(src);
        }

        if space == SPACE_CONST {
            match src {
                LocalValueId::Literal(lit) => {
                    let value = self.shr().values.literals[lit].value;
                    let id = self.shr().get_const(value, size);
                    id.strip_func()
                }

                _ => panic!("Expected literal value for CONST space load"),
            }
        } else {
            // Invariant: if the ptr is a varnode, it must live in the same space as the load.
            // A cross-space access (e.g. *[ram]:8 RSP) requires ensure_local first so that
            // the varnode's *value* is used as the address, not the varnode itself.

            match src {
                LocalValueId::Varnode(id) => {
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

                LocalValueId::Instruction(local) => {
                    self.set_insn_space_local(local, space);
                }

                _ => {}
            }

            let local = self.store_insn(
                Mnemonic::Load(Load {
                    ptr: src,
                    space,
                    size,
                }),
                size,
            );
            LocalValueId::Instruction(local)
        }
    }

    // --- Unary Ops ---

    fn push_unop_local(&mut self, op: Unop, src: LocalValueId) -> LocalInsnId {
        assert!(
            !matches!(src, LocalValueId::Varnode(_)),
            "push_unop: varnode operand is not allowed; use ensure_local or &name addressof syntax"
        );
        let size = self.lsize_of(src);
        self.store_insn(Mnemonic::Unop(Unary { op, src }), size)
    }

    /// Logical NOT of a `bool` value, canonically `src == false`.
    pub fn push_bool_not(&mut self, src: ValueId) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let src = self.loc(src);
        let local = self.push_bool_not_local(src);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_bool_not`](Self::push_bool_not).
    pub fn push_bool_not_local(&mut self, src: LocalValueId) -> LocalInsnId {
        debug_assert!(
            self.lstored_type_of(src)
                .is_some_and(|t| self.shr().types.is_bool(t)),
            "push_bool_not: operand must be bool-typed"
        );
        let f = self.shr().get_bool_const(false).strip_func();
        self.push_binop_local(Binop::Int(IntBinop::Equal), src, f, Some(1))
    }

    unop_leaf!(
        /// Creates a bitwise NOT operation on the given value.
        push_bit_negate,
        push_bit_negate_local,
        Unop::IntNot
    );

    unop_leaf!(
        /// Creates a negation operation on the given value.
        push_neg,
        push_neg_local,
        Unop::IntNegate
    );

    unop_leaf!(
        /// Creates a float negation operation on the given value.
        push_fneg,
        push_fneg_local,
        Unop::FloatNegate
    );

    fn push_binop_local(
        &mut self,
        op: Binop,
        lhs: LocalValueId,
        rhs: LocalValueId,
        size: Option<usize>,
    ) -> LocalInsnId {
        let lhs_size = self.lsize_of(lhs);
        let rhs_size = self.lsize_of(rhs);
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
            self.lsize_of(lhs),
            self.lsize_of(rhs),
            "push_binop: operands must have equal size; emit an explicit cast first"
        );

        // Determine result type using the TypeManager's arithmetic rules.
        let result_type = {
            let lhs_type = self.ltype_of(lhs);
            let rhs_type = self.ltype_of(rhs);
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

        self.store_insn_with_type(Mnemonic::Binop(Binary { op, lhs, rhs }), result_type)
    }

    // --- Arithmetic ---

    binop_leaf!(push_mul, push_mul_local, Binop::Int(IntBinop::Mul), None);
    binop_leaf!(push_div, push_div_local, Binop::Int(IntBinop::Div), None);
    binop_leaf!(push_sdiv, push_sdiv_local, Binop::Int(IntBinop::Sdiv), None);
    binop_leaf!(push_mod, push_mod_local, Binop::Int(IntBinop::Rem), None);
    binop_leaf!(push_smod, push_smod_local, Binop::Int(IntBinop::Srem), None);
    binop_leaf!(push_add, push_add_local, Binop::Int(IntBinop::Add), None);
    binop_leaf!(push_sub, push_sub_local, Binop::Int(IntBinop::Sub), None);

    // --- Float Arithmetic ---

    binop_leaf!(
        push_fdiv,
        push_fdiv_local,
        Binop::Float(FloatBinop::Div),
        None
    );
    binop_leaf!(
        push_fmul,
        push_fmul_local,
        Binop::Float(FloatBinop::Mul),
        None
    );
    binop_leaf!(
        push_fadd,
        push_fadd_local,
        Binop::Float(FloatBinop::Add),
        None
    );
    binop_leaf!(
        push_fsub,
        push_fsub_local,
        Binop::Float(FloatBinop::Sub),
        None
    );

    // --- Shifts ---

    binop_leaf!(
        push_shl,
        push_shl_local,
        Binop::Int(IntBinop::ShiftLeft),
        None
    );
    binop_leaf!(
        push_shr,
        push_shr_local,
        Binop::Int(IntBinop::ShiftRight),
        None
    );
    binop_leaf!(
        push_sshr,
        push_sshr_local,
        Binop::Int(IntBinop::SShiftRight),
        None
    );

    // --- Integer Comparisons ---
    // Greater-than variants swap operands of the less-than op.

    cmp_pair!(
        push_slt,
        push_slt_local,
        push_sgt,
        push_sgt_local,
        Binop::Int(IntBinop::SLess)
    );
    cmp_pair!(
        push_sle,
        push_sle_local,
        push_sge,
        push_sge_local,
        Binop::Int(IntBinop::SLessEqual)
    );
    cmp_pair!(
        push_lt,
        push_lt_local,
        push_gt,
        push_gt_local,
        Binop::Int(IntBinop::Less)
    );
    cmp_pair!(
        push_le,
        push_le_local,
        push_ge,
        push_ge_local,
        Binop::Int(IntBinop::LessEqual)
    );

    // --- Float Comparisons ---

    cmp_pair!(
        push_flt,
        push_flt_local,
        push_fgt,
        push_fgt_local,
        Binop::Float(FloatBinop::Less)
    );
    cmp_pair!(
        push_fle,
        push_fle_local,
        push_fge,
        push_fge_local,
        Binop::Float(FloatBinop::LessEqual)
    );

    // --- Integer Equality ---

    binop_leaf!(push_eq, push_eq_local, Binop::Int(IntBinop::Equal), Some(1));
    binop_leaf!(
        push_ne,
        push_ne_local,
        Binop::Int(IntBinop::NotEqual),
        Some(1)
    );
    binop_leaf!(
        push_feq,
        push_feq_local,
        Binop::Float(FloatBinop::Equal),
        Some(1)
    );
    binop_leaf!(
        push_fne,
        push_fne_local,
        Binop::Float(FloatBinop::NotEqual),
        Some(1)
    );

    // --- Bitwise ---

    /// Logical XOR of two `bool` operands — a bitwise `Xor` over `bool`, which
    /// yields `bool` (exact on the `{0,1}` domain).
    pub fn push_bool_xor(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let (lhs, rhs) = (self.loc(lhs), self.loc(rhs));
        let local = self.push_bool_xor_local(lhs, rhs);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_bool_xor`](Self::push_bool_xor).
    pub fn push_bool_xor_local(&mut self, lhs: LocalValueId, rhs: LocalValueId) -> LocalInsnId {
        debug_assert!(
            self.both_bool(lhs, rhs),
            "push_bool_xor: operands must be bool"
        );
        self.push_binop_local(Binop::Int(IntBinop::Xor), lhs, rhs, None)
    }

    /// Logical AND of two `bool` operands (bitwise `And` over `bool`).
    pub fn push_bool_and(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let (lhs, rhs) = (self.loc(lhs), self.loc(rhs));
        let local = self.push_bool_and_local(lhs, rhs);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_bool_and`](Self::push_bool_and).
    pub fn push_bool_and_local(&mut self, lhs: LocalValueId, rhs: LocalValueId) -> LocalInsnId {
        debug_assert!(
            self.both_bool(lhs, rhs),
            "push_bool_and: operands must be bool"
        );
        self.push_binop_local(Binop::Int(IntBinop::And), lhs, rhs, None)
    }

    /// Logical OR of two `bool` operands (bitwise `Or` over `bool`).
    pub fn push_bool_or(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let (lhs, rhs) = (self.loc(lhs), self.loc(rhs));
        let local = self.push_bool_or_local(lhs, rhs);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_bool_or`](Self::push_bool_or).
    pub fn push_bool_or_local(&mut self, lhs: LocalValueId, rhs: LocalValueId) -> LocalInsnId {
        debug_assert!(
            self.both_bool(lhs, rhs),
            "push_bool_or: operands must be bool"
        );
        self.push_binop_local(Binop::Int(IntBinop::Or), lhs, rhs, None)
    }

    /// Whether both operands carry the `bool` type (a `debug_assert` guard).
    fn both_bool(&self, lhs: LocalValueId, rhs: LocalValueId) -> bool {
        let is_bool = |v: LocalValueId| {
            self.lstored_type_of(v)
                .is_some_and(|t| self.shr().types.is_bool(t))
        };
        is_bool(lhs) && is_bool(rhs)
    }

    binop_leaf!(
        push_bit_xor,
        push_bit_xor_local,
        Binop::Int(IntBinop::Xor),
        None
    );
    binop_leaf!(
        push_bit_or,
        push_bit_or_local,
        Binop::Int(IntBinop::Or),
        None
    );
    binop_leaf!(
        push_bit_and,
        push_bit_and_local,
        Binop::Int(IntBinop::And),
        None
    );

    // --- Extensions & Conversions ---

    pub fn push_is_nan(&mut self, src: ValueId) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let src = self.loc(src);
        let local = self.push_is_nan_local(src);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_is_nan`](Self::push_is_nan).
    pub fn push_is_nan_local(&mut self, src: LocalValueId) -> LocalInsnId {
        assert!(
            !matches!(src, LocalValueId::Varnode(_)),
            "push_is_nan: varnode operand not allowed"
        );
        self.store_insn(Mnemonic::IsFloatNaN(IsFloatNaN { src }), 1)
    }

    unop_leaf!(push_abs, push_abs_local, Unop::FloatAbs);
    unop_leaf!(push_sqrt, push_sqrt_local, Unop::FloatSqrt);
    unop_leaf!(push_floor, push_floor_local, Unop::FloatFloor);
    unop_leaf!(push_ceil, push_ceil_local, Unop::FloatCeil);
    unop_leaf!(push_round, push_round_local, Unop::FloatRound);

    conv_leaf!(
        push_int_to_float,
        push_int_to_float_local,
        "push_int_to_float: varnode operand not allowed",
        IntToFloat
    );
    conv_leaf!(
        push_float_to_float,
        push_float_to_float_local,
        "push_float_to_float: varnode operand not allowed",
        FloatToFloat
    );
    conv_leaf!(
        push_trunc,
        push_trunc_local,
        "push_trunc: varnode operand not allowed",
        FloatToInt
    );
    conv_leaf!(
        push_zext,
        push_zext_local,
        "push_zext: varnode operand not allowed",
        Zext
    );
    conv_leaf!(
        push_sext,
        push_sext_local,
        "push_sext: varnode operand not allowed",
        Sext
    );

    /// Builds an aggregate value from `fields` using default field names
    /// (`field1`, `field2`, ...). The result type is the
    /// [`Aggregate`](crate::types::TypeRepr::Aggregate) of the named fields'
    /// types.
    pub fn push_tuple(
        &mut self,
        fields: Vec<ValueId>,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let fields = self.loc_vec(fields);
        let local = self.push_tuple_local(fields);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_tuple`](Self::push_tuple).
    pub fn push_tuple_local(&mut self, fields: Vec<LocalValueId>) -> LocalInsnId {
        let named_fields = fields
            .into_iter()
            .enumerate()
            .map(|(i, value)| (format!("field{}", i + 1), value))
            .collect();
        self.push_named_tuple_local(named_fields)
    }

    /// Builds an aggregate value from ordered named fields.
    pub fn push_named_tuple(
        &mut self,
        fields: Vec<(String, ValueId)>,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let fields = fields
            .into_iter()
            .map(|(name, v)| (name, self.loc(v)))
            .collect();
        let local = self.push_named_tuple_local(fields);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_named_tuple`](Self::push_named_tuple).
    pub fn push_named_tuple_local(&mut self, fields: Vec<(String, LocalValueId)>) -> LocalInsnId {
        let field_types: Vec<TypeId> = fields.iter().map(|(_, f)| self.ltype_of(*f)).collect();
        let aggregate_fields = fields
            .iter()
            .zip(field_types)
            .map(|((name, _), type_id)| AggregateField::new(name.clone(), type_id))
            .collect();
        let ty = self
            .shr()
            .types
            .get_or_make_named_aggregate(aggregate_fields);
        self.push_named_tuple_local_with_type(fields, ty)
    }

    /// Build a named tuple using an explicitly selected aggregate-like type.
    /// Used for nominal function-return records whose identity must not be
    /// structurally interned by [`push_named_tuple_local`](Self::push_named_tuple_local).
    pub fn push_named_tuple_with_type(
        &mut self,
        fields: Vec<(String, ValueId)>,
        ty: TypeId,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let fields = fields
            .into_iter()
            .map(|(name, value)| (name, self.loc(value)))
            .collect();
        let local = self.push_named_tuple_local_with_type(fields, ty);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_named_tuple_with_type`](Self::push_named_tuple_with_type).
    pub fn push_named_tuple_local_with_type(
        &mut self,
        fields: Vec<(String, LocalValueId)>,
        ty: TypeId,
    ) -> LocalInsnId {
        debug_assert_eq!(
            self.shr().types.aggregate_fields(ty).map(<[_]>::len),
            Some(fields.len()),
            "explicit tuple type must declare every tuple field"
        );
        debug_assert!(fields.iter().enumerate().all(|(index, (name, value))| {
            self.shr()
                .types
                .aggregate_fields(ty)
                .and_then(|declared| declared.get(index))
                .is_some_and(|declared| {
                    declared.name == *name && declared.type_id == self.ltype_of(*value)
                })
        }));
        let values = fields.into_iter().map(|(_, value)| value).collect();
        self.store_insn_with_type(Mnemonic::Tuple(Tuple { fields: values }), ty)
    }

    /// Projects field `index` out of the aggregate value `agg`. The result type
    /// is that field's type. Panics if `agg` is not an aggregate with that field.
    pub fn push_extract(
        &mut self,
        agg: ValueId,
        index: usize,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let agg = self.loc(agg);
        let local = self.push_extract_local(agg, index);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_extract`](Self::push_extract).
    pub fn push_extract_local(&mut self, agg: LocalValueId, index: usize) -> LocalInsnId {
        let agg_ty = self.ltype_of(agg);
        let ty = self
            .shr()
            .types
            .field_type(agg_ty, index)
            .expect("push_extract: agg is not an aggregate with that field index");
        self.store_insn_with_type(Mnemonic::Extract(Extract { agg, index }), ty)
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
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let (src, captures) = (self.loc(src), self.loc_vec(captures));
        let local = self.push_map_local(body, src, captures);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_map`](Self::push_map).
    pub fn push_map_local(
        &mut self,
        body: impl Into<Callee>,
        src: LocalValueId,
        captures: Vec<LocalValueId>,
    ) -> LocalInsnId {
        let body = body.into();
        let src_ty = self.ltype_of(src);
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
        self.push_map_typed_local(body, src, captures, ty)
    }

    /// Builds a map with an explicitly prepared result type. Use this when the
    /// body is foreign to this Builder and its body-derived return type is not
    /// part of the published function interface.
    pub fn push_map_typed(
        &mut self,
        body: impl Into<Callee>,
        src: ValueId,
        captures: Vec<ValueId>,
        result_type: TypeId,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let (src, captures) = (self.loc(src), self.loc_vec(captures));
        let local = self.push_map_typed_local(body, src, captures, result_type);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_map_typed`](Self::push_map_typed).
    pub fn push_map_typed_local(
        &mut self,
        body: impl Into<Callee>,
        src: LocalValueId,
        captures: Vec<LocalValueId>,
        result_type: TypeId,
    ) -> LocalInsnId {
        self.store_insn_with_type(
            Mnemonic::Map(Map {
                body: body.into(),
                src,
                captures,
            }),
            result_type,
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
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let (init, src, captures) = (self.loc(init), self.loc(src), self.loc_vec(captures));
        let local = self.push_scan_local(body, init, src, captures);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_scan`](Self::push_scan).
    pub fn push_scan_local(
        &mut self,
        body: impl Into<Callee>,
        init: LocalValueId,
        src: LocalValueId,
        captures: Vec<LocalValueId>,
    ) -> LocalInsnId {
        let body = body.into();
        let src_ty = self.ltype_of(src);
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
        self.push_scan_typed_local(body, init, src, captures, ty)
    }

    /// Builds a scan with an explicitly prepared result type. This is the
    /// foreign-body counterpart to [`push_map_typed`](Self::push_map_typed).
    pub fn push_scan_typed(
        &mut self,
        body: impl Into<Callee>,
        init: ValueId,
        src: ValueId,
        captures: Vec<ValueId>,
        result_type: TypeId,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let (init, src, captures) = (self.loc(init), self.loc(src), self.loc_vec(captures));
        let local = self.push_scan_typed_local(body, init, src, captures, result_type);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_scan_typed`](Self::push_scan_typed).
    pub fn push_scan_typed_local(
        &mut self,
        body: impl Into<Callee>,
        init: LocalValueId,
        src: LocalValueId,
        captures: Vec<LocalValueId>,
        result_type: TypeId,
    ) -> LocalInsnId {
        self.store_insn_with_type(
            Mnemonic::Scan(Scan {
                body: body.into(),
                init,
                src,
                captures,
            }),
            result_type,
        )
    }

    /// Builds a value-level application of a pure lambda function. Unlike
    /// [`push_call`](Self::push_call), this is an ordinary SSA instruction and
    /// does not terminate the current block.
    pub fn push_apply(
        &mut self,
        target: impl Into<Callee>,
        args: Vec<ValueId>,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let args = self.loc_vec(args);
        let local = self.push_apply_local(target, args);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_apply`](Self::push_apply).
    pub fn push_apply_local(
        &mut self,
        target: impl Into<Callee>,
        args: Vec<LocalValueId>,
    ) -> LocalInsnId {
        let target = target.into();
        let ty = target
            .real()
            .and_then(|target| self.lambda_return_type(target))
            .unwrap_or_else(|| {
                args.first()
                    .map(|&arg| self.ltype_of(arg))
                    .unwrap_or_else(|| self.shr().types.get_or_make_int(0))
            });
        self.store_insn_with_type(Mnemonic::Apply(Apply { target, args }), ty)
    }

    /// The type of the value returned by `body`'s first `Return`, or `None` if
    /// `body` is not this (self) body, has no root, or returns nothing — used to
    /// size a [`push_map`] result. Id-less: reads this body's own arenas.
    fn map_body_return_type(&self, body: FunctionId) -> Option<TypeId> {
        if self.body.try_id() != Some(body) {
            return None;
        }
        let root = self.body.root_id()?;
        self.body.blocks[root].instructions.iter().find_map(|&i| {
            match self.body.insns[i].mnemonic() {
                Mnemonic::Return(r) => r.value.and_then(|v| self.lstored_type_of(v)),
                _ => None,
            }
        })
    }

    /// The type of the first value returned by a lambda body (this body). Id-less.
    fn lambda_return_type(&self, body: FunctionId) -> Option<TypeId> {
        if self.body.try_id() != Some(body) {
            return None;
        }
        self.body
            .roster
            .iter()
            .flat_map(|&b| self.body.blocks[b].instructions.iter().copied())
            .find_map(|i| match self.body.insns[i].mnemonic() {
                Mnemonic::ReturnValue(r) => self.lstored_type_of(r.value),
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
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let base = self.loc(base);
        let local = self.push_gep_local(base, offset);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_gep`](Self::push_gep).
    pub fn push_gep_local(&mut self, base: LocalValueId, offset: usize) -> LocalInsnId {
        let base_ty = self.ltype_of(base);
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
        self.store_insn_with_type(Mnemonic::Gep(Gep { base, offset }), ty)
    }

    /// Like [`push_gep`](Builder::push_gep) but selects the field by name,
    /// resolving it to a byte offset via the pointee struct of `base`. Panics if
    /// `base` is not a struct pointer or has no field of that name.
    pub fn push_gep_field(
        &mut self,
        base: ValueId,
        name: &str,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let base = self.loc(base);
        let local = self.push_gep_field_local(base, name);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_gep_field`](Self::push_gep_field).
    pub fn push_gep_field_local(&mut self, base: LocalValueId, name: &str) -> LocalInsnId {
        let base_ty = self.ltype_of(base);
        let types = &self.shr().types;
        let pointee = types
            .pointee_of(base_ty)
            .expect("push_gep_field: base is not a struct pointer");
        let offset = types
            .aggregate_fields(pointee)
            .and_then(|fields| fields.iter().find(|f| f.name == name))
            .map(|f| f.offset)
            .expect("push_gep_field: pointee struct has no field of that name");
        self.push_gep_local(base, offset)
    }

    pub fn push_popcount(
        &mut self,
        src: ValueId,
        size: usize,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let src = self.loc(src);
        let local = self.push_popcount_local(src, size);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_popcount`](Self::push_popcount).
    pub fn push_popcount_local(&mut self, src: LocalValueId, size: usize) -> LocalInsnId {
        assert!(
            !matches!(src, LocalValueId::Varnode(_)),
            "push_popcount: varnode operand not allowed"
        );
        self.store_insn(Mnemonic::PopCount(PopCount { src }), size)
    }

    pub fn push_lzcount(
        &mut self,
        src: ValueId,
        size: usize,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let src = self.loc(src);
        let local = self.push_lzcount_local(src, size);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_lzcount`](Self::push_lzcount).
    pub fn push_lzcount_local(&mut self, src: LocalValueId, size: usize) -> LocalInsnId {
        assert!(
            !matches!(src, LocalValueId::Varnode(_)),
            "push_lzcount: varnode operand not allowed"
        );
        self.store_insn(Mnemonic::LzCount(LzCount { src }), size)
    }

    pub fn push_carry(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let (lhs, rhs) = (self.loc(lhs), self.loc(rhs));
        let local = self.push_carry_local(lhs, rhs);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_carry`](Self::push_carry).
    pub fn push_carry_local(&mut self, lhs: LocalValueId, rhs: LocalValueId) -> LocalInsnId {
        assert!(
            !matches!(lhs, LocalValueId::Varnode(_)) && !matches!(rhs, LocalValueId::Varnode(_)),
            "push_carry: varnode operand not allowed"
        );
        self.store_insn(Mnemonic::Carry(Carry { lhs, rhs }), 1)
    }

    pub fn push_scarry(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let (lhs, rhs) = (self.loc(lhs), self.loc(rhs));
        let local = self.push_scarry_local(lhs, rhs);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_scarry`](Self::push_scarry).
    pub fn push_scarry_local(&mut self, lhs: LocalValueId, rhs: LocalValueId) -> LocalInsnId {
        assert!(
            !matches!(lhs, LocalValueId::Varnode(_)) && !matches!(rhs, LocalValueId::Varnode(_)),
            "push_scarry: varnode operand not allowed"
        );
        self.store_insn(Mnemonic::SCarry(SCarry { lhs, rhs }), 1)
    }

    pub fn push_sborrow(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let (lhs, rhs) = (self.loc(lhs), self.loc(rhs));
        let local = self.push_sborrow_local(lhs, rhs);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_sborrow`](Self::push_sborrow).
    pub fn push_sborrow_local(&mut self, lhs: LocalValueId, rhs: LocalValueId) -> LocalInsnId {
        assert!(
            !matches!(lhs, LocalValueId::Varnode(_)) && !matches!(rhs, LocalValueId::Varnode(_)),
            "push_sborrow: varnode operand not allowed"
        );
        self.store_insn(Mnemonic::SBorrow(SBorrow { lhs, rhs }), 1)
    }

    pub fn push_pcode_op(
        &mut self,
        id: PCodeOpId,
        args: Vec<ValueId>,
        dst: Option<ValueId>,
        size: usize,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let args = self.loc_vec(args);
        let dst = dst.map(|d| self.loc(d));
        let local = self.push_pcode_op_local(id, args, dst, size);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_pcode_op`](Self::push_pcode_op).
    pub fn push_pcode_op_local(
        &mut self,
        id: PCodeOpId,
        args: Vec<LocalValueId>,
        dst: Option<LocalValueId>,
        size: usize,
    ) -> LocalInsnId {
        let args = args
            .into_iter()
            .map(|arg| self.ensure_local_local(arg))
            .collect::<Vec<_>>();

        self.store_insn(Mnemonic::PCodeOp(PCodeOp { id, args, dst }), size)
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
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let args = self.loc_vec(args);
        let local = self.push_intrinsic_local(id, args);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_intrinsic`](Self::push_intrinsic).
    #[track_caller]
    pub fn push_intrinsic_local(
        &mut self,
        id: IntrinsicId,
        args: Vec<LocalValueId>,
    ) -> LocalInsnId {
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
            .map(|arg| self.ensure_local_local(arg))
            .collect::<Vec<_>>();

        let arg_types = args
            .iter()
            .map(|&arg| self.ltype_of(arg))
            .collect::<Vec<_>>();
        let type_id = desc.result_type(&self.shr().types, &arg_types);

        self.store_insn_with_type(Mnemonic::Intrinsic(IntrinsicApp { id, args }), type_id)
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
        dst: impl Into<ValueId>,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let src = self.loc(src);
        let dst = self.loc(dst.into());
        let local = self.push_copy_local(src, dst);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_copy`](Self::push_copy).
    pub fn push_copy_local(&mut self, src: LocalValueId, dst: LocalValueId) -> LocalInsnId {
        let (size, space, name) = match dst {
            LocalValueId::Varnode(vid) => {
                let node = Varnode::from_id(self.shr(), vid);
                (
                    node.size(),
                    LocalMemorySpaceId::Shared(node.space().id),
                    node.name().map(str::to_owned),
                )
            }
            LocalValueId::Temp(tlocal) => {
                let temp = &self.body.temps[tlocal];
                (
                    temp.size,
                    LocalMemorySpaceId::Temp(temp.space),
                    temp.name.as_deref().map(str::to_owned),
                )
            }
            _ => panic!("copy destination must be a varnode or body-local temporary"),
        };

        const LANE_SIZE: usize = 8;

        if size > LANE_SIZE {
            let num_lanes = size.div_ceil(LANE_SIZE);
            let mut first_id = None;

            for lane in 0..num_lanes {
                let offset = lane * LANE_SIZE;
                let lane_size = cmp::min(LANE_SIZE, size - offset);

                let src_lane = self
                    .get_range_local(src, offset..offset + lane_size)
                    .expect("lane range in bounds");
                let src_lane = self.ensure_local_local(src_lane);

                let dst_lane = self
                    .get_range_local(dst, offset..offset + lane_size)
                    .expect("lane range in bounds");

                let id = self.store_insn(
                    Mnemonic::Store(Store {
                        src: src_lane,
                        ptr: dst_lane,
                        space,
                        size: lane_size,
                    }),
                    0,
                );

                if let Some(name) = &name {
                    let name = Cow::Owned(format!("{}_lane{lane}", name.to_lowercase()));
                    let _ = self.rename_insn_local(id, name);
                }

                first_id.get_or_insert(id);
            }

            first_id.unwrap()
        } else {
            let src = self.ensure_local_local(src);
            // If dst is a varnode, we need to emit a store from src to dst
            let id = self.store_insn(
                Mnemonic::Store(Store {
                    src,
                    ptr: dst,
                    space,
                    size,
                }),
                0,
            );

            // Add a name hint for the store instruction for easier debugging
            if let Some(name) = &name {
                let lowered = name.to_lowercase();
                let name = self.body.names.unique(Cow::Owned(lowered));
                self.rename_insn_local(id, name)
                    .expect("This name was deduplicated");
            }

            id
        }
    }

    #[track_caller]
    pub fn push_store(
        &mut self,
        src: ValueId,
        ptr: ValueId,
        space: impl Into<LocalMemorySpaceId>,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let (src, ptr) = (self.loc(src), self.loc(ptr));
        let local = self.push_store_local(src, ptr, space);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_store`](Self::push_store).
    #[track_caller]
    pub fn push_store_local(
        &mut self,
        src: LocalValueId,
        ptr: LocalValueId,
        space: impl Into<LocalMemorySpaceId>,
    ) -> LocalInsnId {
        let space = space.into();
        let src = self.ensure_local_local(src);
        let size = self.lsize_of(src);

        match ptr {
            LocalValueId::Varnode(id) => {
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

            LocalValueId::Instruction(local) => {
                self.set_insn_space_local(local, space);
            }

            _ => {}
        }

        self.store_insn(
            Mnemonic::Store(Store {
                src,
                ptr,
                space,
                size,
            }),
            0,
        )
    }

    // --- Branches & Calls ---

    /// Declares a new parameter on the current block.
    pub fn push_param(&mut self, size: usize) -> BlockParamId {
        let local = self.push_param_local(size);
        crate::value::block_param::BlockParamId::new(self.func(), local)
    }

    /// Body-local sibling of [`push_param`](Self::push_param).
    pub fn push_param_local(&mut self, size: usize) -> crate::value::LocalParamId {
        let block = self.block;
        let index = self.body.blocks[block].params.len();
        let type_id = self.shared.types.get_or_make_int(size);
        let local = self.body.params.push(BlockParam {
            index,
            type_id,
            parent: Some(block),
            name: None,
            origin: None,
            protected: false,
        });
        self.body.blocks[block].params.push(local);
        local
    }

    /// Terminates the current block with a branch to an already-resolved local
    /// target. Module/address discovery must happen before the Builder borrow.
    pub fn finalize(mut self, target: BlockId) {
        self.finalize_local(target.local)
    }

    /// Body-local sibling of [`finalize`](Self::finalize).
    pub fn finalize_local(&mut self, target: LocalBlockId) {
        if !self.is_terminated() {
            let branch = self.push_branch_local(target);
            if let Some(address) = self.address {
                self.body.insns[branch].set_address(address);
            }
        }
    }

    /// Add a CFG edge from the working block to `target`, both body-local
    /// (id-less twin of [`FunctionBody::add_cfg_edge`]).
    fn add_cfg_edge_local(&mut self, from: LocalBlockId, to: LocalBlockId) {
        let edge_id = self
            .body
            .edges
            .push(crate::value::block::cfg::EdgeData { from, to });
        self.body.blocks[from].edges.insert(edge_id);
        self.body.blocks[to].edges.insert(edge_id);
    }

    /// Terminates this block with an unconditional jump to the given target block.
    /// The builder is now safe to drop without panicking, and the block is properly terminated.
    pub fn push_branch(&mut self, target: BlockId) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let local = self.push_branch_local(target.local);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_branch`](Self::push_branch).
    pub fn push_branch_local(&mut self, target: LocalBlockId) -> LocalInsnId {
        self.push_branch_with_args_local(target, vec![])
    }

    /// Unconditional branch passing `args` to the target block's parameters.
    pub fn push_branch_with_args(
        &mut self,
        target: BlockId,
        args: Vec<ValueId>,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let args = self.loc_vec(args);
        let local = self.push_branch_with_args_local(target.local, args);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_branch_with_args`](Self::push_branch_with_args).
    pub fn push_branch_with_args_local(
        &mut self,
        target: LocalBlockId,
        args: Vec<LocalValueId>,
    ) -> LocalInsnId {
        let current = self.block;
        self.add_cfg_edge_local(current, target);
        let id = self.store_insn(Mnemonic::Branch(Branch { target, args }), 0);
        self.is_terminated = true;
        id
    }

    pub fn push_cbranch(
        &mut self,
        condition: ValueId,
        target: BlockId,
        fallthrough: BlockId,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        self.push_cbranch_with_args(condition, target, vec![], fallthrough, vec![])
    }

    /// Body-local sibling of [`push_cbranch`](Self::push_cbranch).
    pub fn push_cbranch_local(
        &mut self,
        condition: LocalValueId,
        target: LocalBlockId,
        fallthrough: LocalBlockId,
    ) -> LocalInsnId {
        self.push_cbranch_with_args_local(condition, target, vec![], fallthrough, vec![])
    }

    /// Conditional branch with per-target arguments.
    pub fn push_cbranch_with_args(
        &mut self,
        condition: ValueId,
        target: BlockId,
        target_args: Vec<ValueId>,
        fallthrough: BlockId,
        fallthrough_args: Vec<ValueId>,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let condition = self.loc(condition);
        let target_args = self.loc_vec(target_args);
        let fallthrough_args = self.loc_vec(fallthrough_args);
        let local = self.push_cbranch_with_args_local(
            condition,
            target.local,
            target_args,
            fallthrough.local,
            fallthrough_args,
        );
        self.insn_ref(local)
    }

    /// Body-local sibling of
    /// [`push_cbranch_with_args`](Self::push_cbranch_with_args).
    pub fn push_cbranch_with_args_local(
        &mut self,
        condition: LocalValueId,
        target: LocalBlockId,
        target_args: Vec<LocalValueId>,
        fallthrough: LocalBlockId,
        fallthrough_args: Vec<LocalValueId>,
    ) -> LocalInsnId {
        assert!(
            !matches!(condition, LocalValueId::Varnode(_)),
            "push_cbranch: varnode condition not allowed; load the value first"
        );
        let current = self.block;
        self.add_cfg_edge_local(current, target);
        self.add_cfg_edge_local(current, fallthrough);
        let id = self.store_insn(
            Mnemonic::CBranch(CBranch {
                success_block: target,
                success_args: target_args,
                condition,
                failure_block: fallthrough,
                failure_args: fallthrough_args,
            }),
            0,
        );
        self.is_terminated = true;
        id
    }

    pub fn push_branchind(&mut self, ptr: ValueId) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let ptr = self.loc(ptr);
        let local = self.push_branchind_local(ptr);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_branchind`](Self::push_branchind).
    pub fn push_branchind_local(&mut self, ptr: LocalValueId) -> LocalInsnId {
        let id = self.store_insn(Mnemonic::BranchInd(BranchInd { ptr }), 0);
        self.is_terminated = true;
        id
    }

    pub fn push_call(
        &mut self,
        target: impl Into<Callee>,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        self.push_call_with_args(target, vec![])
    }

    pub fn push_call_with_args(
        &mut self,
        target: impl Into<Callee>,
        args: Vec<ValueId>,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let args = self.loc_vec(args);
        let local = self.push_call_with_args_local(target, args);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_call`](Self::push_call).
    pub fn push_call_local(&mut self, target: impl Into<Callee>) -> LocalInsnId {
        self.push_call_with_args_local(target, vec![])
    }

    /// Body-local sibling of [`push_call_with_args`](Self::push_call_with_args).
    pub fn push_call_with_args_local(
        &mut self,
        target: impl Into<Callee>,
        args: Vec<LocalValueId>,
    ) -> LocalInsnId {
        let target = target.into();
        let id = self.store_insn(
            Mnemonic::Call(Call {
                target,
                args,
                clobbers: vec![],
                tag: Default::default(),
            }),
            0,
        );
        self.is_terminated = true;
        id
    }

    /// Tail call to another function's entry — a function-level terminator with
    /// no intra-function CFG successor (see [`TailCall`](crate::value::insn::TailCall)).
    /// Unlike [`push_branch`](Self::push_branch), this wires no CFG edge: control
    /// leaves the function.
    pub fn push_tail_call(
        &mut self,
        target: impl Into<Callee>,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        self.push_tail_call_with_args(target, vec![])
    }

    pub fn push_tail_call_with_args(
        &mut self,
        target: impl Into<Callee>,
        args: Vec<ValueId>,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let args = self.loc_vec(args);
        let local = self.push_tail_call_with_args_local(target, args);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_tail_call`](Self::push_tail_call).
    pub fn push_tail_call_local(&mut self, target: impl Into<Callee>) -> LocalInsnId {
        self.push_tail_call_with_args_local(target, vec![])
    }

    /// Body-local sibling of
    /// [`push_tail_call_with_args`](Self::push_tail_call_with_args).
    pub fn push_tail_call_with_args_local(
        &mut self,
        target: impl Into<Callee>,
        args: Vec<LocalValueId>,
    ) -> LocalInsnId {
        let target = target.into();
        let id = self.store_insn(Mnemonic::TailCall(TailCall { target, args }), 0);
        self.is_terminated = true;
        id
    }

    pub fn push_call_ind(&mut self, ptr: ValueId) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        self.push_call_ind_with_args(ptr, vec![])
    }

    pub fn push_call_ind_with_args(
        &mut self,
        ptr: ValueId,
        args: Vec<ValueId>,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let (ptr, args) = (self.loc(ptr), self.loc_vec(args));
        let local = self.push_call_ind_with_args_local(ptr, args);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_call_ind`](Self::push_call_ind).
    pub fn push_call_ind_local(&mut self, ptr: LocalValueId) -> LocalInsnId {
        self.push_call_ind_with_args_local(ptr, vec![])
    }

    /// Body-local sibling of
    /// [`push_call_ind_with_args`](Self::push_call_ind_with_args).
    pub fn push_call_ind_with_args_local(
        &mut self,
        ptr: LocalValueId,
        args: Vec<LocalValueId>,
    ) -> LocalInsnId {
        let id = self.store_insn(Mnemonic::CallInd(CallInd { ptr, args }), 0);
        self.is_terminated = true;
        id
    }

    pub fn push_return(&mut self, ptr: ValueId) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let ptr = self.loc(ptr);
        let local = self.push_return_local(ptr);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_return`](Self::push_return).
    pub fn push_return_local(&mut self, ptr: LocalValueId) -> LocalInsnId {
        self.push_return_at_local(None, ptr)
    }

    pub fn push_return_with_value(
        &mut self,
        value: ValueId,
        ptr: ValueId,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let (value, ptr) = (self.loc(value), self.loc(ptr));
        let local = self.push_return_at_local(Some(value), ptr);
        self.insn_ref(local)
    }

    /// Body-local sibling of
    /// [`push_return_with_value`](Self::push_return_with_value).
    pub fn push_return_with_value_local(
        &mut self,
        value: LocalValueId,
        ptr: LocalValueId,
    ) -> LocalInsnId {
        self.push_return_at_local(Some(value), ptr)
    }

    fn push_return_at_local(
        &mut self,
        value: Option<LocalValueId>,
        ptr: LocalValueId,
    ) -> LocalInsnId {
        let id = self.store_insn(Mnemonic::Return(Return { ptr, value }), 0);
        self.is_terminated = true;
        id
    }

    pub fn push_return_value(
        &mut self,
        value: ValueId,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let value = self.loc(value);
        let local = self.push_return_value_local(value);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_return_value`](Self::push_return_value).
    pub fn push_return_value_local(&mut self, value: LocalValueId) -> LocalInsnId {
        let id = self.store_insn(Mnemonic::ReturnValue(ReturnValue { value }), 0);
        self.is_terminated = true;
        id
    }

    // --- Assert ---

    /// Asserts that a condition holds at this point in execution
    pub fn push_assert(
        &mut self,
        condition: ValueId,
    ) -> InstructionRef<'str, '_, BodyView<'_, 'str>> {
        let condition = self.loc(condition);
        let local = self.push_assert_local(condition);
        self.insn_ref(local)
    }

    /// Body-local sibling of [`push_assert`](Self::push_assert).
    pub fn push_assert_local(&mut self, condition: LocalValueId) -> LocalInsnId {
        self.store_insn(Mnemonic::Assert(Assert { condition }), 0)
    }
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;
    use crate::{context::Context, value::ModuleView};

    #[test]
    fn checked_builder_matches_module_builder() {
        use crate::value::{
            FunctionId, FunctionRef, block::BasicBlock, function::FunctionBody,
            util::body_mut::BodyMut,
        };

        // The same body over any host: consts and a couple of binops (exercising
        // type + literal minting through the interners' `&self` paths), then a
        // branch to a freshly minted local-label block. No varnode/temp-space
        // minting — a checked-out host has read-only shared access.
        fn body<'str>(b: &mut Builder<'str, '_>) {
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
            let mut b = ctx_a.builder(entry_a);
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
            let mut host = BodyMut::new(&mut ctx_b.bodies[fid_b], &ctx_b.shared, &ctx_b.interfaces);
            let mut b = host.builder(entry_b);
            body(&mut b);
        }
        let snap_b = snap(&ctx_b, fid_b);

        assert_eq!(
            snap_a, snap_b,
            "a body built through a checked-out builder must match the module-built body"
        );
    }

    /// An id-less (detached) body can be driven by `Builder::new_local` and the
    /// `push_*_local` verbs without ever acquiring a registry identity: the
    /// builder's engine is fully body-local.
    #[test]
    fn detached_body_builds_through_local_verbs() {
        let ctx = Context::new();
        let mut body = FunctionBody::detached();
        assert_eq!(body.try_id(), None, "sanity: body starts detached");

        // Mint entry and target blocks through the body's local verbs.
        let entry = body.push_block_local(BasicBlock::detached());
        let target = body.push_block_local(BasicBlock::detached());
        body.set_root_id(Some(entry));

        let c1 = ctx.shared.get_const(7, 8).strip_func();
        let c2 = ctx.shared.get_const(9, 8).strip_func();

        {
            let mut b = Builder::new_local(&mut body, &ctx.shared, &ctx.interfaces, entry);
            let sum = b.push_add_local(c1, c2);
            let sum = LocalValueId::Instruction(sum);
            let doubled = b.push_add_local(sum, sum);
            let _cmp = b.push_eq_local(LocalValueId::Instruction(doubled), c2);
            b.push_branch_local(target);
            assert!(b.is_terminated());
            b.switch_to_block_local(target);
            let ret = b.push_return_value_local(sum);
            let _ = ret;
        }

        // Instructions landed in the arenas, wired to their blocks.
        assert_eq!(body.blocks[entry].instructions.len(), 4);
        assert_eq!(body.blocks[target].instructions.len(), 1);
        let last = *body.blocks[entry].instructions.last().unwrap();
        assert!(body.insns[last].mnemonic().is_terminator());

        // The body never acquired an identity: it is still detached.
        assert_eq!(body.try_id(), None);
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
            let mut b = ctx.builder(hentry);
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
    fn test_builder_finalize() {
        // Test code for Builder drop behavior
        // This test will compile and run without panicking because we finalize the builder properly
        let mut ctx = Context::new();
        let block_id = {
            let __f = ctx.anon_function();
            ctx.get_or_make_block(0, __f)
        };
        let target = ctx.get_or_make_block(0x1000, block_id.func);

        {
            let builder = ctx.builder(block_id);
            builder.finalize(target);
        }
    }

    #[test]
    fn test_named_temp_duplicate() {
        let mut ctx = Context::new();
        let mut builder = ctx.builder_at(0x1000);

        let value = builder.make_named_temp("dup".into(), 4);
        let other_value = builder.make_named_temp("dup".into(), 4);
        let target = builder.current_block();
        builder.finalize(target);

        assert_eq!(
            TempRef::new(ModuleView::new(&ctx), value).name(),
            Some("dup")
        );
        assert_eq!(
            TempRef::new(ModuleView::new(&ctx), other_value).name(),
            Some("dup_1")
        );
    }

    #[test]
    fn same_label_temps_in_different_functions_are_isolated() {
        let mut ctx = Context::new();

        let first = {
            let mut builder = ctx.builder_at(0x1000);
            let temp = builder.make_temp_labeled(7, 4);
            let target = builder.current_block();
            builder.finalize(target);
            temp
        };
        let second = {
            let mut builder = ctx.builder_at(0x2000);
            let temp = builder.make_temp_labeled(7, 4);
            let target = builder.current_block();
            builder.finalize(target);
            temp
        };

        assert_ne!(first, second);
        let first = TempRef::new(ModuleView::new(&ctx), first);
        let second = TempRef::new(ModuleView::new(&ctx), second);
        assert_eq!((first.label(), second.label()), (Some(7), Some(7)));
        assert_ne!(first.space().id, second.space().id);
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
        let mut builder = ctx.builder(block_id);

        let p0 = builder.push_param(8);
        let p0_id = p0;
        let p1 = builder.push_param(4);
        let p1_id = p1;

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
            let mut builder = ctx.builder(src_id);
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
            let source = ctx.builder_at(0x1000).current_block();
            let target = ctx.get_or_make_block(0x1001, source.func);
            let mut builder = ctx.builder(source);
            builder.set_address(0x1000);
            let not_insn_id = builder.push_bit_negate(id_42).id;
            builder.finalize(target);

            not_insn_id
        };

        let insn = Instruction::from_id(&ctx, not_insn_id);

        assert_eq!(insn.address().unwrap(), 0x1000);
    }

    #[test]
    fn builder_at_materializes_root_in_registered_function_arena() {
        let mut ctx = Context::new();
        let func = FunctionBody::make_at_addr(&mut ctx, 0x2000, None).id;
        assert!(FunctionBody::from_id(&ctx, func).root().is_none());

        let block = {
            let builder = ctx.builder_at(0x2000);
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
            let source = ctx.builder_at(0x4010).current_block();
            let target = ctx.get_or_make_block(0x4020, source.func);
            let mut builder = ctx.builder(source);

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
            let target = ctx.get_or_make_block(0x1001, block_id.func);
            let mut builder = ctx.builder(block_id);
            let src = builder.make_named_temp("src".into(), 9);
            let dst = builder.make_named_temp("dst".into(), 9);
            builder.push_copy(src.into(), dst);
            builder.finalize(target);
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
            let mut b = ctx.builder(block_id);

            b.push_bit_negate(val).id
        };

        let prepended_id = {
            let mut b = ctx.builder(block_id);
            b.set_insert_point_to_start();
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
            let mut b = ctx.builder(block_id);

            b.push_bit_negate(val).id
        };

        let (id0, id1, id2) = {
            let mut b = ctx.builder(block_id);
            b.set_insert_point_to_start();
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
            let mut b = ctx.builder(block_id);
            (b.push_bit_negate(val).id, b.push_bit_negate(val).id)
        };

        let (inserted0, inserted1) = {
            let mut b = ctx.builder(block_id);
            b.set_insert_point_before(target_id);
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
            let mut b = ctx.builder(entry);
            b.set_insert_point_to_start();
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
            let mut b = ctx.builder(block_id);
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
