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

use std::{borrow::Cow, collections::HashMap};

use crate::{
    context::Context,
    error::Result,
    space::{SPACE_CONST, SpaceId},
    value::{
        Function, Instruction, Renameable, Value, ValueId, ValueRef,
        block::{BasicBlock, BlockId, BlockMutRef},
        function::FunctionId,
        insn::{
            Binary, Binop, BoolBinop, Branch, BranchInd, CBranch, Call, CallInd, Carry, FloatBinop,
            FloatToFloat, FloatToInt, InstructionRef, IntBinop, IntToFloat, IsFloatNaN, Load,
            LzCount, Mnemonic, PCodeOp, PCodeOpId, PopCount, Range, Return, SBorrow, SCarry, Sext,
            Store, Unary, Unop, Zext,
        },
        util::base_ref::{WithCtx, WithCtxMut},
        varnode::{Varnode, VarnodeId},
    },
};

/// A builder for constructing instructions in a block.
/// This provides a convenient API for creating instructions, and automatically
/// manages temporary values and labels.
pub struct Builder<'str, 'ctx> {
    pub block: BlockMutRef<'str, 'ctx>,

    /// Converts from names to value IDs in the current scope.
    namespace: HashMap<Cow<'str, str>, ValueId>,

    /// Names of local labels to their corresponding block IDs.
    local_labels: HashMap<Cow<'str, str>, BlockId>,

    /// Is the block terminated, i.e. does it end with a terminator
    /// If it is not the case, the block might be invalid
    pub(crate) is_terminated: bool,

    pub built_instructions: Vec<InstructionId>,

    verify_terminated: bool,
}

/// Generates a canonical comparison method and its "greater-than" mirror (operands swapped).
macro_rules! cmp_pair {
    ($fwd:ident, $rev:ident, $op:expr) => {
        pub fn $fwd(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
            self.push_binop($op, lhs, rhs, Some(1))
        }
        pub fn $rev(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
            self.push_binop($op, rhs, lhs, Some(1))
        }
    };
}

impl<'str, 'ctx> Builder<'str, 'ctx> {
    /// Creates a builder positioned at `block`.
    ///
    /// The block is borrowed mutably for the lifetime `'ctx`. New instructions
    /// will be appended to the end of `block`.
    pub fn from_block(block: BlockMutRef<'str, 'ctx>) -> Self {
        Self {
            is_terminated: block.is_terminated(),
            verify_terminated: true,
            block,
            built_instructions: Vec::new(),
            namespace: HashMap::new(),
            local_labels: HashMap::new(),
        }
    }

    /// Creates a builder positioned at the block for machine `address`,
    /// creating the block if it does not already exist.
    pub fn from_context<'m>(ctx: &'m mut Context<'str>, address: u64) -> Builder<'str, 'm> {
        let block_id = ctx.get_or_make_block(address);
        Builder::from_block(BasicBlock::from_id_mut(ctx, block_id))
    }

    /// Returns `true` if the current block ends with a terminator instruction.
    pub fn is_terminated(&self) -> bool {
        self.block.is_terminated()
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
    ) -> Option<ValueRef<'str, '_>> {
        let value = self.get_value(src);

        if range.is_empty() {
            return None;
        }

        let dst = match value {
            ValueRef::Literal(literal) => {
                let value = literal.value();
                self.context_mut().get_const(value, range.len()).into()
            }

            ValueRef::Varnode(varnode_ref) => {
                if range.end > varnode_ref.size() {
                    return None;
                }

                let base = varnode_ref.address() + range.start as i64;
                let space = varnode_ref.space().id;

                let id = Varnode::make(self.context_mut(), base, range.len(), space).id;

                Varnode::from_id(self.context(), id).into()
            }

            ValueRef::Instruction(insn) => {
                if range.end > insn.size() {
                    return None;
                }

                self.push_instruction(
                    Mnemonic::Range(Range {
                        src,
                        start: range.start,
                        size: range.len(),
                    }),
                    range.len(),
                )
                .into()
            }

            // Functions, blocks, and other non-data values have no byte range
            _ => return None,
        };

        Some(dst)
    }

    /// Creates a new builder with the same insert point but an empty namespace.
    /// This builder won't need to be finalized
    pub fn with_empty_namespaces(&mut self) -> Builder<'str, '_> {
        let mut builder = Builder::from_block(self.block.reborrow());

        unsafe {
            // This is safe because the parent builder will not be dropped while the child builder is still in use
            // It is responsible for finalizing the block
            builder.dont_finalize();
        }

        builder
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
        // This feels wrong, we are assigning a mut ref...
        self.block.with_id(block);
        self.is_terminated = self.block.is_terminated();
    }

    /// Gets the ID of a value in the current namespace
    pub fn try_get_value(&self, name: &str) -> Option<ValueRef<'str, '_>> {
        self.namespace
            .get(name)
            .map(|&id| self.context().get_value(id))
    }

    /// Returns the current context
    pub fn context_mut(&mut self) -> &mut Context<'str> {
        self.block.ctx_mut()
    }

    pub fn context(&self) -> &Context<'str> {
        self.block.as_ref().ctx()
    }

    /// Adds an instruction at the end of the working block.
    ///
    /// # Panics
    ///
    /// Panics if the block is already terminated (ends with a branch/call/return).
    #[track_caller]
    fn push_instruction(&mut self, mnemonic: Mnemonic, size: usize) -> InstructionRef<'str, '_> {
        if self.is_terminated {
            panic!("cannot push instruction to a terminated block");
        }

        let id = InstructionRef::from_mnemonic(self.context_mut(), mnemonic, size).id;
        self.block.push_insn(id);
        self.built_instructions.push(id);
        self.context().get_insn(id)
    }

    fn get_value(&self, id: ValueId) -> ValueRef<'str, '_> {
        self.context().get_value(id)
    }

    fn ensure_created_block_in_function(&mut self, block: BlockId) {
        if let Some(mut function) = self.block.parent_mut() {
            function.add_block(block);
        }
    }

    /// Gets or creates a block for a given address
    pub fn get_or_make_block(&mut self, addr: u64) -> BlockId {
        let id = self.context_mut().get_or_make_block(addr);

        if BasicBlock::from_id(self.context(), id).parent().is_none() {
            self.ensure_created_block_in_function(id);
        }

        id
    }

    pub fn get_or_make_local_label(&mut self, name: Cow<'str, str>) -> BlockId {
        if let Some(&id) = self.local_labels.get(name.as_ref()) {
            id
        } else {
            let id = BasicBlock::make(self.context_mut())
                .with_name(name.clone())
                .expect("Block name already exists")
                .id;
            self.local_labels.insert(name, id);
            self.ensure_created_block_in_function(id);
            id
        }
    }

    /// Gets or creates a function whose root is the local label block for `name`.
    pub fn get_or_make_local_function(&mut self, name: Cow<'str, str>) -> FunctionId {
        let existing = Function::from_name(self.context(), &name).map(|f| f.id);
        if let Some(fid) = existing {
            fid
        } else {
            Function::make(self.context_mut(), name)
                .expect("Name was checked above")
                .id
        }
    }

    pub fn make_temp(&mut self, size: usize) -> VarnodeId {
        let space = self.context_mut().make_temp_space();
        Varnode::make(self.context_mut(), 0, size, space).id
    }

    /// Creates a new temporary value with the given name and size.
    /// This value is a memory value so does not need to follow any SSA rules.
    /// The name is deduplicated with a numeric suffix if already taken in the context.
    pub fn make_named_temp(&mut self, name: Cow<'str, str>, size: usize) -> VarnodeId {
        let id = self.make_temp(size);
        let unique_name = self.context().get_unique_name(name);
        Varnode::from_id_mut(self.context_mut(), id)
            .rename(unique_name)
            .expect("This name was deduplicated");
        id
    }

    /// If this block is not terminated, add a jump to the given address as a terminator instruction.
    /// The builder is now safe to drop without panicking, and the block is properly terminated.
    /// Returns the instructions built by the builder.
    pub fn finalize(mut self, addr: u64) -> Vec<InstructionId> {
        if !self.block.is_terminated() {
            let target = self.get_or_make_block(addr);
            self.push_branch(target);
        }

        self.built_instructions.clone()
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
                if let Some(name) = Varnode::from_id(self.context(), node_id).name() {
                    let name = self.context().get_unique_name(name.to_lowercase().into());

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
    ) -> ValueRef<'str, '_> {
        if CHECK_LOCAL {
            src = self.ensure_local(src);
        }

        if space == SPACE_CONST {
            let src = self.get_value(src);

            match src {
                ValueRef::Literal(lit) => {
                    let value = lit.value();
                    self.context_mut().get_const(value, size).into()
                }

                _ => panic!("Expected literal value for CONST space load"),
            }
        } else {
            // Invariant: if the ptr is a varnode, it must live in the same space as the load.
            // A cross-space access (e.g. *[ram]:8 RSP) requires ensure_local first so that
            // the varnode's *value* is used as the address, not the varnode itself.
            debug_assert!(
                match self.context().get_value(src) {
                    ValueRef::Varnode(v) => v.space().id == space,
                    _ => true,
                },
                "push_load: ptr is a varnode but its space does not match the load space {:?}; \
                 call ensure_local on the ptr first",
                space,
            );
            let size = self.context().get_value(src).size();
            self.push_instruction(
                Mnemonic::Load(Load {
                    ptr: src,
                    space,
                    size,
                }),
                size,
            )
            .into()
        }
    }

    // --- Unary Ops ---

    fn push_unop(&mut self, op: Unop, src: ValueId) -> InstructionRef<'str, '_> {
        let src = self.ensure_local(src);
        let size = self.context().get_value(src).size();
        self.push_instruction(Mnemonic::Unop(Unary { op, src }), size)
    }

    /// Creates a logical NOT operation on the given value.
    pub fn push_bool_not(&mut self, src: ValueId) -> InstructionRef<'str, '_> {
        self.push_unop(Unop::BoolNot, src)
    }

    /// Creates a bitwise NOT operation on the given value.
    pub fn push_bit_negate(&mut self, src: ValueId) -> InstructionRef<'str, '_> {
        self.push_unop(Unop::IntNot, src)
    }

    /// Creates a negation operation on the given value.
    pub fn push_neg(&mut self, src: ValueId) -> InstructionRef<'str, '_> {
        self.push_unop(Unop::IntNegate, src)
    }

    /// Creates a float negation operation on the given value.
    pub fn push_fneg(&mut self, src: ValueId) -> InstructionRef<'str, '_> {
        self.push_unop(Unop::FloatNegate, src)
    }

    fn push_binop(
        &mut self,
        op: Binop,
        lhs: ValueId,
        rhs: ValueId,
        size: Option<usize>,
    ) -> InstructionRef<'str, '_> {
        let size = size.unwrap_or_else(|| self.context().get_value(lhs).size());
        let lhs = self.ensure_local(lhs);
        let rhs = self.ensure_local(rhs);
        self.push_instruction(Mnemonic::Binop(Binary { op, lhs, rhs }), size)
    }

    // --- Arithmetic ---

    pub fn push_mul(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Int(IntBinop::Mul), lhs, rhs, None)
    }

    pub fn push_div(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Int(IntBinop::Div), lhs, rhs, None)
    }

    pub fn push_sdiv(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Int(IntBinop::Sdiv), lhs, rhs, None)
    }

    pub fn push_mod(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Int(IntBinop::Rem), lhs, rhs, None)
    }

    pub fn push_smod(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Int(IntBinop::Srem), lhs, rhs, None)
    }

    pub fn push_add(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Int(IntBinop::Add), lhs, rhs, None)
    }

    pub fn push_sub(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Int(IntBinop::Sub), lhs, rhs, None)
    }

    // --- Float Arithmetic ---

    pub fn push_fdiv(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Float(FloatBinop::Div), lhs, rhs, None)
    }

    pub fn push_fmul(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Float(FloatBinop::Mul), lhs, rhs, None)
    }

    pub fn push_fadd(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Float(FloatBinop::Add), lhs, rhs, None)
    }

    pub fn push_fsub(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Float(FloatBinop::Sub), lhs, rhs, None)
    }

    // --- Shifts ---

    pub fn push_shl(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Int(IntBinop::ShiftLeft), lhs, rhs, None)
    }

    pub fn push_shr(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Int(IntBinop::ShiftRight), lhs, rhs, None)
    }

    pub fn push_sshr(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
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

    pub fn push_eq(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Int(IntBinop::Equal), lhs, rhs, Some(1))
    }

    pub fn push_ne(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Int(IntBinop::NotEqual), lhs, rhs, Some(1))
    }

    pub fn push_feq(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Float(FloatBinop::Equal), lhs, rhs, Some(1))
    }

    pub fn push_fne(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Float(FloatBinop::NotEqual), lhs, rhs, Some(1))
    }

    // --- Bitwise ---

    pub fn push_bool_xor(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        debug_assert_eq!(self.context().get_value(lhs).size(), 1);
        self.push_binop(Binop::Bool(BoolBinop::Xor), lhs, rhs, Some(1))
    }

    pub fn push_bool_and(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        debug_assert_eq!(self.context().get_value(lhs).size(), 1);
        self.push_binop(Binop::Bool(BoolBinop::And), lhs, rhs, Some(1))
    }

    pub fn push_bool_or(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        debug_assert_eq!(self.context().get_value(lhs).size(), 1);
        self.push_binop(Binop::Bool(BoolBinop::Or), lhs, rhs, Some(1))
    }

    pub fn push_bit_xor(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Int(IntBinop::Xor), lhs, rhs, None)
    }

    pub fn push_bit_or(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Int(IntBinop::Or), lhs, rhs, None)
    }

    pub fn push_bit_and(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        self.push_binop(Binop::Int(IntBinop::And), lhs, rhs, None)
    }

    // --- Extensions & Conversions ---

    pub fn push_is_nan(&mut self, src: ValueId) -> InstructionRef<'str, '_> {
        let src = self.ensure_local(src);
        self.push_instruction(Mnemonic::IsFloatNaN(IsFloatNaN { src }), 1)
    }

    pub fn push_abs(&mut self, src: ValueId) -> InstructionRef<'str, '_> {
        self.push_unop(Unop::FloatAbs, src)
    }

    pub fn push_sqrt(&mut self, src: ValueId) -> InstructionRef<'str, '_> {
        self.push_unop(Unop::FloatSqrt, src)
    }

    pub fn push_floor(&mut self, src: ValueId) -> InstructionRef<'str, '_> {
        self.push_unop(Unop::FloatFloor, src)
    }

    pub fn push_ceil(&mut self, src: ValueId) -> InstructionRef<'str, '_> {
        self.push_unop(Unop::FloatCeil, src)
    }

    pub fn push_round(&mut self, src: ValueId) -> InstructionRef<'str, '_> {
        self.push_unop(Unop::FloatRound, src)
    }

    pub fn push_int_to_float(&mut self, src: ValueId, size: usize) -> InstructionRef<'str, '_> {
        let src = self.ensure_local(src);
        self.push_instruction(Mnemonic::IntToFloat(IntToFloat { src, size }), size)
    }

    pub fn push_float_to_float(&mut self, src: ValueId, size: usize) -> InstructionRef<'str, '_> {
        let src = self.ensure_local(src);
        self.push_instruction(Mnemonic::FloatToFloat(FloatToFloat { src, size }), size)
    }

    pub fn push_trunc(&mut self, src: ValueId, size: usize) -> InstructionRef<'str, '_> {
        let src = self.ensure_local(src);
        self.push_instruction(Mnemonic::FloatToInt(FloatToInt { src, size }), size)
    }

    pub fn push_zext(&mut self, src: ValueId, size: usize) -> InstructionRef<'str, '_> {
        let src = self.ensure_local(src);
        self.push_instruction(Mnemonic::Zext(Zext { src, size }), size)
    }

    pub fn push_sext(&mut self, src: ValueId, size: usize) -> InstructionRef<'str, '_> {
        let src = self.ensure_local(src);
        self.push_instruction(Mnemonic::Sext(Sext { src, size }), size)
    }

    pub fn push_popcount(&mut self, src: ValueId, size: usize) -> InstructionRef<'str, '_> {
        let src = self.ensure_local(src);
        self.push_instruction(Mnemonic::PopCount(PopCount { src }), size)
    }

    pub fn push_lzcount(&mut self, src: ValueId, size: usize) -> InstructionRef<'str, '_> {
        let src = self.ensure_local(src);
        self.push_instruction(Mnemonic::LzCount(LzCount { src }), size)
    }

    pub fn push_carry(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        let lhs = self.ensure_local(lhs);
        let rhs = self.ensure_local(rhs);
        self.push_instruction(Mnemonic::Carry(Carry { lhs, rhs }), 1)
    }

    pub fn push_scarry(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        let lhs = self.ensure_local(lhs);
        let rhs = self.ensure_local(rhs);
        self.push_instruction(Mnemonic::SCarry(SCarry { lhs, rhs }), 1)
    }

    pub fn push_sborrow(&mut self, lhs: ValueId, rhs: ValueId) -> InstructionRef<'str, '_> {
        let lhs = self.ensure_local(lhs);
        let rhs = self.ensure_local(rhs);
        self.push_instruction(Mnemonic::SBorrow(SBorrow { lhs, rhs }), 1)
    }

    pub fn push_pcode_op(
        &mut self,
        id: PCodeOpId,
        args: Vec<ValueId>,
        dst: Option<ValueId>,
    ) -> InstructionRef<'str, '_> {
        let args = args
            .into_iter()
            .map(|arg| self.ensure_local(arg))
            .collect::<Vec<_>>();

        self.push_instruction(Mnemonic::PCodeOp(PCodeOp { id, args, dst }), 0)
    }

    // --- Loads & Stores ---

    /// Creates a copy instruction from `src` to `dst`.
    /// Note that `dst` must already exist as a [`Value`] in the current context, and this will not create a new temporary value.
    /// If `dst` is a varnode, we aren't allowed to write to it, this is a store operation
    /// If `src` is a varnode, we need to read from it first, then write to dst
    /// For values wider than 64 bits (e.g. XMM/YMM/ZMM registers), emits one store per 64-bit lane.
    pub fn push_copy(&mut self, src: ValueId, dst: VarnodeId) -> InstructionRef<'str, '_> {
        let node = Varnode::from_id(self.context(), dst);
        let size = node.size();
        let space = node.space().id;

        const LANE_SIZE: usize = 8;

        if size > LANE_SIZE {
            assert!(
                size.is_multiple_of(LANE_SIZE),
                "push_copy: value size {size} is not a multiple of {LANE_SIZE} bytes"
            );

            let num_lanes = size / LANE_SIZE;
            let mut first_id = None;

            for lane in 0..num_lanes {
                let offset = lane * LANE_SIZE;

                let src_lane = self
                    .get_range(src, offset..offset + LANE_SIZE)
                    .expect("lane range in bounds")
                    .id();
                let src_lane = self.ensure_local(src_lane);

                let dst_lane = self
                    .get_range(dst.into(), offset..offset + LANE_SIZE)
                    .expect("lane range in bounds")
                    .id();

                let id = self
                    .push_instruction(
                        Mnemonic::Store(Store {
                            src: src_lane,
                            ptr: dst_lane,
                            space,
                            size: LANE_SIZE,
                        }),
                        0,
                    )
                    .id;

                if let Some(name) = Varnode::from_id(self.context(), dst).name() {
                    let name = Cow::Owned(format!("{}_lane{lane}", name.to_lowercase()));
                    let _ = Instruction::from_id_mut(self.context_mut(), id).rename(name);
                }

                first_id.get_or_insert(id);
            }

            self.context().get_insn(first_id.unwrap())
        } else {
            let src = self.ensure_local(src);
            // If dst is a varnode, we need to emit a store from src to dst
            let id = self
                .push_instruction(
                    Mnemonic::Store(Store {
                        src,
                        ptr: dst.into(),
                        space,
                        size,
                    }),
                    0,
                )
                .id;

            // Add a name hint for the store instruction for easier debugging
            if let Some(name) = Varnode::from_id(self.context(), dst).name() {
                let name = self
                    .context()
                    .get_unique_name(Cow::Owned(name.to_lowercase()));
                Instruction::from_id_mut(self.context_mut(), id)
                    .rename(name)
                    .expect("This name was deduplicated");
            }

            self.context().get_insn(id)
        }
    }

    pub fn push_store(
        &mut self,
        src: ValueId,
        ptr: ValueId,
        space: SpaceId,
    ) -> InstructionRef<'str, '_> {
        // TODO(fix store local):
        let ptr = self.ensure_local(ptr);
        let src = self.ensure_local(src);
        let size = self.context().get_value(src).size();
        self.push_instruction(
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

    /// Terminates this block with an unconditional jump to the given target block.
    /// The builder is now safe to drop without panicking, and the block is properly terminated.
    pub fn push_branch(&mut self, target: BlockId) -> InstructionRef<'str, '_> {
        let current = self.block.id;
        self.context_mut().add_cfg_edge(current, target);
        let id = self
            .push_instruction(Mnemonic::Branch(Branch { target }), 0)
            .id;
        self.is_terminated = true;
        self.context().get_insn(id)
    }

    pub fn push_cbranch(
        &mut self,
        condition: ValueId,
        target: BlockId,
        fallthrough: BlockId,
    ) -> InstructionRef<'str, '_> {
        let condition = self.ensure_local(condition);
        let current = self.block.id;
        self.context_mut().add_cfg_edge(current, target);
        self.context_mut().add_cfg_edge(current, fallthrough);
        let id = self
            .push_instruction(
                Mnemonic::CBranch(CBranch {
                    target,
                    condition,
                    fallthrough,
                }),
                0,
            )
            .id;
        self.is_terminated = true;
        self.context().get_insn(id)
    }

    pub fn push_branchind(&mut self, ptr: ValueId) -> InstructionRef<'str, '_> {
        let ptr = self.ensure_local(ptr);
        let id = self
            .push_instruction(Mnemonic::BranchInd(BranchInd { ptr }), 0)
            .id;
        self.is_terminated = true;
        self.context().get_insn(id)
    }

    pub fn push_call(&mut self, target: FunctionId) -> InstructionRef<'str, '_> {
        let id = self
            .push_instruction(
                Mnemonic::Call(Call {
                    target,
                    args: vec![],
                }),
                0,
            )
            .id;
        self.is_terminated = true;
        self.context().get_insn(id)
    }

    pub fn push_call_ind(&mut self, ptr: ValueId) -> InstructionRef<'str, '_> {
        let ptr = self.ensure_local(ptr);
        let id = self
            .push_instruction(Mnemonic::CallInd(CallInd { ptr, args: vec![] }), 0)
            .id;
        self.is_terminated = true;
        self.context().get_insn(id)
    }

    pub fn push_return(&mut self, ptr: ValueId) -> InstructionRef<'str, '_> {
        let ptr = self.ensure_local(ptr);
        let id = self
            .push_instruction(Mnemonic::Return(Return { ptr, value: None }), 0)
            .id;
        self.is_terminated = true;
        self.context().get_insn(id)
    }
}

impl Drop for Builder<'_, '_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }

        if !self.is_terminated &&
        // I don't see how this can happen, but just in case, we also check if the block is actually terminated, to avoid panicking when dropping a builder that has already been finalized
        !self.block.is_terminated()
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
    use jstd::graph::{Graph, Node};
    use qcode_macro::qcode;

    use super::*;
    use crate::context::Context;

    #[test]
    fn cfg_branch_adds_one_node_and_one_edge() {
        let mut ctx = Context::new();
        {
            let mut builder = Builder::from_context(&mut ctx, 0x1000);
            qcode!(builder, "goto <done>; <done>");
            builder.finalize(0x1001);
        }
        // entry (1000) + done + 1001 = 3 nodes; entry→done, done→1001 = 2 edges
        assert_eq!(ctx.nodes().count(), 3);
        assert_eq!(ctx.edges().count(), 2);
    }

    #[test]
    fn cfg_cbranch_adds_two_edges() {
        let mut ctx = Context::new();
        {
            let mut builder = Builder::from_context(&mut ctx, 0x1000);
            qcode!(builder, "local i8 cond");
            qcode!(builder, "if {cond} goto <then_lbl> else goto <else_lbl>");
            builder.finalize(0x1001);
        }
        // entry + then_lbl + else_lbl + 1001 = 4 nodes
        // entry→then_lbl, entry→else_lbl, else_lbl→1001 = 3 edges
        assert_eq!(ctx.nodes().count(), 4);
        assert_eq!(ctx.edges().count(), 3);
    }

    // TODO:
    // #[test]
    // fn cfg_call_adds_one_edge() {
    //     let mut ctx = Context::new();
    //     {
    //         let mut builder = Builder::from_context(&mut ctx, 0x1000);
    //         qcode!(builder, "call <func>");
    //     }
    //     // entry + func + 1001 = 3 nodes; entry→func, func→1001 = 2 edges
    //     assert_eq!(ctx.nodes().count(), 3);
    //     assert_eq!(ctx.edges().count(), 2);
    // }

    #[test]
    fn cfg_branchind_adds_node_but_no_outgoing_edge() {
        let mut ctx = Context::new();
        let entry;
        {
            let mut builder = Builder::from_context(&mut ctx, 0x1000);
            entry = builder.block.id;
            qcode!(builder, "local i64 ptr");
            qcode!(builder, "goto [{ptr}]");
        }
        assert_eq!(ctx.nodes().count(), 1);
        assert_eq!(ctx.edges().count(), 0);

        assert_eq!(BasicBlock::from_id(&ctx, entry).children().count(), 0);
    }

    #[test]
    fn cfg_return_adds_node_but_no_outgoing_edge() {
        let mut ctx = Context::new();
        let entry;
        {
            let mut builder = Builder::from_context(&mut ctx, 0x1000);
            entry = builder.block.id;
            qcode!(builder, "local i64 ptr");
            qcode!(builder, "return [{ptr}]");
        }
        assert_eq!(ctx.nodes().count(), 1);
        assert_eq!(ctx.edges().count(), 0);
        assert_eq!(BasicBlock::from_id(&ctx, entry).children().count(), 0);
    }

    #[test]
    fn cfg_multi_block_qcode_program() {
        let mut ctx = Context::new();
        let mut builder = Builder::from_context(&mut ctx, 0x1000);
        qcode!(builder, "local i32 v");
        qcode!(builder, "goto <body>; <body> {v} + 1");
        builder.finalize(0x1001);

        // entry (1000) + body + 1001 = 3 nodes
        // entry→body, body→1001 = 2 edges
        assert_eq!(ctx.nodes().count(), 3);
        assert_eq!(ctx.edges().count(), 2);
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
        let block_id = ctx.get_or_make_block(0);
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
        let block_id = ctx.get_or_make_block(0);
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
    fn qcode_local_decl_creates_named_temp() {
        let mut ctx = Context::new();
        let ptr_id: VarnodeId;
        {
            let mut builder = Builder::from_context(&mut ctx, 0x1000);
            qcode!(builder, "local i64 ptr");
            ptr_id = ptr;
            builder.finalize(0x1001);
        }

        let ptr = Varnode::from_id(&ctx, ptr_id);

        assert_eq!(ptr.size(), 8);
        assert_eq!(ptr.name(), Some("ptr"));
    }

    #[test]
    fn qcode_standalone_local_decl_creates_named_temp() {
        let mut ctx = Context::new();
        let ptr_id: VarnodeId;
        {
            let mut builder = Builder::from_context(&mut ctx, 0x1000);
            qcode!(builder, "local i64 ptr as PTR");
            ptr_id = ptr;
            builder.finalize(0x1001);
        }

        let ptr = Varnode::from_id(&ctx, ptr_id);

        assert_eq!(ptr.size(), 8);
        assert_eq!(ptr.name(), Some("PTR"));
    }

    #[test]
    fn test_builder_adds_block_address_to_qcode() {
        let mut ctx = Context::new();
        let id_42 = ctx.get_const(42, 8).id();

        let not_insn_id = {
            let mut builder = Builder::from_context(&mut ctx, 0x1000);
            let not_insn_id = builder.push_bool_not(id_42).id;
            builder.finalize(0x1001);

            not_insn_id
        };

        let insn = Instruction::from_id(&ctx, not_insn_id);

        assert_eq!(insn.address().unwrap(), 0x1000);
    }
}
