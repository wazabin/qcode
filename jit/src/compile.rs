//! Translating one QCode block into a native function.
//!
//! # What this compiles, and what it declines
//!
//! A QCode block is already SSA, so the translation to Cranelift's SSA is
//! direct: an instruction's result becomes a Cranelift value, and a value used
//! only inside the block never reaches memory at all. That is the point of
//! compiling. The interpreter has to materialise every intermediate into its
//! value table because it cannot see past one operation at a time; compiled
//! code keeps them in registers.
//!
//! The compiler is deliberately partial. It handles integer arithmetic and
//! accesses to *flat* spaces — registers, uniques, per-function temporaries —
//! whose addresses are constants known at compile time, so a register access
//! becomes a load at a fixed offset from a base pointer. Anything else
//! ([`Unsupported`]) is declined, and the caller runs that block on the
//! interpreter instead. Declining is a normal outcome, not a failure: it is what
//! lets this be an alternative strategy rather than a replacement.
//!
//! Guest RAM is *not* handled here. Every RAM access must go through the MMU for
//! its permission check and fault reporting, and inlining that is a later step.
//!
//! # Terminators
//!
//! Control flow itself stays with the interpreter: it runs the terminator after
//! compiled code has run the body, so branch resolution, block parameters and
//! call semantics live in exactly one implementation. What the compiler has to
//! supply is the terminator's *operands* — a `cbranch` condition, a branch's
//! block arguments — because those are values the body computed and compiled
//! code keeps in registers, where the interpreter cannot see them.
//!
//! So a block's compiled function takes a second argument: an *export buffer*.
//! Every terminator operand defined in this block is written there as a `u64`,
//! and the runtime copies it into the interpreter's value table before handing
//! the terminator back. A terminator whose operands are all literals, addresses
//! or values from earlier blocks exports nothing and costs nothing — which is
//! the argument-less unconditional branch that straight-line guest code lifts
//! to.
//!
//! # Escaping values
//!
//! The same reasoning applies to any value that outlives the block, not just
//! the ones the terminator reads. Values crossing a block boundary as bare SSA
//! references do not arise in SLEIGH-lifted code — guest state travels through
//! registers and uniques, which are memory — but nothing in QCode forbids them,
//! and dropping one would be a silent miscompile rather than a decline. So a
//! block with a result used from outside it is declined outright.

use cranelift::prelude::*;
use qcode::{
    context::Context,
    space::MemorySpaceId,
    value::{
        BasicBlock, BlockId, ValueId, ValueRef,
        insn::{
            Binary, Binop, Carry, InstructionId, IntBinop, Load, Mnemonic, PopCount, Range,
            SBorrow, SCarry, Sext, Store, Unary, Unop, Zext,
        },
    },
};
use rustc_hash::{FxHashMap, FxHashSet};

/// Why a block could not be compiled.
///
/// Carried as a value because declining is expected: the caller falls back to
/// the interpreter, and the reason is useful for reporting coverage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unsupported {
    /// A mnemonic the compiler does not translate.
    Mnemonic(&'static str),
    /// An operand or result whose width is not a machine integer width.
    Width(usize),
    /// A memory access this backend will not perform directly.
    Access(&'static str),
    /// The block's terminator is not one the compiler leaves to the caller.
    Terminator(&'static str),
    /// An operand whose value the compiler cannot produce.
    Operand(&'static str),
    /// A value defined in this block and read from outside it.
    Escapes(&'static str),
}

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mnemonic(what) => write!(f, "unsupported mnemonic `{what}`"),
            Self::Width(size) => write!(f, "unsupported operand width {size}"),
            Self::Access(what) => write!(f, "unsupported memory access: {what}"),
            Self::Terminator(what) => write!(f, "unsupported terminator `{what}`"),
            Self::Operand(what) => write!(f, "unsupported operand: {what}"),
            Self::Escapes(what) => write!(f, "value escapes the block: {what}"),
        }
    }
}

/// The machine integer type for a `size`-byte value.
pub(crate) fn int_type(size: usize) -> Result<Type, Unsupported> {
    match size {
        1 => Ok(types::I8),
        2 => Ok(types::I16),
        4 => Ok(types::I32),
        8 => Ok(types::I64),
        other => Err(Unsupported::Width(other)),
    }
}

/// The flat spaces a compiled block touches, and how many bytes of each it must
/// be able to address.
///
/// Collected while compiling so the runtime can grow each space *before* taking
/// a base pointer: growing reallocates, and compiled code holds the pointer for
/// the length of the block.
#[derive(Debug, Default, Clone)]
pub struct SpaceTable {
    entries: Vec<(MemorySpaceId, usize)>,
}

impl SpaceTable {
    /// The index compiled code uses to find this space's base pointer, adding
    /// it to the table if it is new and widening its required size.
    fn slot(&mut self, space: MemorySpaceId, required: usize) -> usize {
        if let Some(index) = self.entries.iter().position(|(id, _)| *id == space) {
            self.entries[index].1 = self.entries[index].1.max(required);
            return index;
        }
        self.entries.push((space, required));
        self.entries.len() - 1
    }

    pub fn entries(&self) -> &[(MemorySpaceId, usize)] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Translates the body of one block into an already-open Cranelift function.
pub(crate) struct BlockTranslator<'a, 'ctx> {
    ctx: &'ctx Context<'ctx>,
    builder: FunctionBuilder<'a>,
    /// Base of the array of space base pointers, the compiled function's first
    /// argument.
    spaces_arg: Value,
    /// Base of the export buffer, the compiled function's second argument: one
    /// `u64` slot per entry of [`Self::exports`].
    exports_arg: Value,
    /// Cached base pointer per space slot, loaded once per block rather than per
    /// access.
    bases: FxHashMap<usize, Value>,
    /// Values produced by instructions in this block.
    values: FxHashMap<InstructionId, Value>,
    pub(crate) table: SpaceTable,
    /// Values the terminator reads, in export-buffer slot order.
    pub(crate) exports: Vec<Export>,
}

/// One value compiled code hands back for the interpreter to read.
#[derive(Debug, Clone, Copy)]
pub struct Export {
    /// The instruction whose result this is; the key it is filed under in the
    /// interpreter's value table.
    pub insn: InstructionId,
    /// Its declared width in bytes, which the slot's low bytes hold.
    pub size: usize,
}

impl<'a, 'ctx> BlockTranslator<'a, 'ctx> {
    pub(crate) fn new(
        ctx: &'ctx Context<'ctx>,
        builder: FunctionBuilder<'a>,
        entry: cranelift::prelude::Block,
    ) -> Self {
        let spaces_arg = builder.block_params(entry)[0];
        let exports_arg = builder.block_params(entry)[1];
        Self {
            ctx,
            builder,
            spaces_arg,
            exports_arg,
            bases: FxHashMap::default(),
            values: FxHashMap::default(),
            table: SpaceTable::default(),
            exports: Vec::new(),
        }
    }

    /// Loads (once) the base pointer for a space slot.
    fn base(&mut self, slot: usize) -> Value {
        if let Some(base) = self.bases.get(&slot) {
            return *base;
        }
        let offset = (slot * std::mem::size_of::<*mut u8>()) as i32;
        let base =
            self.builder
                .ins()
                .load(types::I64, MemFlags::trusted(), self.spaces_arg, offset);
        self.bases.insert(slot, base);
        base
    }

    /// The address a flat access resolves to, or `None` if it is not a constant.
    fn constant_address(&self, ptr: ValueId) -> Option<u64> {
        match ValueRef::new(ptr, self.ctx) {
            ValueRef::Literal(literal) => Some(literal.value()),
            ValueRef::Temp(temp) => Some(temp.address() as u64),
            ValueRef::Varnode(varnode) => Some(varnode.address() as u64),
            _ => None,
        }
    }

    /// Produces the Cranelift value for a QCode operand.
    fn operand(&mut self, id: ValueId, size: usize) -> Result<Value, Unsupported> {
        let ty = int_type(size)?;
        match id {
            ValueId::Literal(_) => {
                let ValueRef::Literal(literal) = ValueRef::new(id, self.ctx) else {
                    return Err(Unsupported::Operand("literal did not resolve"));
                };
                Ok(self.builder.ins().iconst(ty, literal.value() as i64))
            }
            ValueId::Instruction(insn) => self
                .values
                .get(&insn)
                .copied()
                .ok_or(Unsupported::Operand("value produced outside this block")),
            // A varnode or temp used as a *value* is its address, which only
            // appears as a pointer operand and is handled there.
            _ => Err(Unsupported::Operand("not a literal or in-block value")),
        }
    }

    /// The declared width of an operand.
    fn width_of(&self, id: ValueId) -> Result<usize, Unsupported> {
        let ty = self
            .ctx
            .stored_type_of(id)
            .ok_or(Unsupported::Operand("operand has no type"))?;
        Ok(self.ctx.shared.types.size_of(ty))
    }

    /// Translates every non-terminator instruction of `block`, then emits the
    /// exports its terminator will need.
    pub(crate) fn translate_body(&mut self, block: BlockId) -> Result<(), Unsupported> {
        let insns: Vec<InstructionId> = BasicBlock::from_id(self.ctx, block).instruction_ids();
        let Some((&terminator, body)) = insns.split_last() else {
            return Err(Unsupported::Terminator("block is empty"));
        };
        let own: FxHashSet<InstructionId> = insns.iter().copied().collect();

        for &insn_id in body {
            self.translate_one(insn_id)?;
            self.check_confined(insn_id, &own)?;
        }

        self.export_terminator_operands(terminator, &own)
    }

    /// Declines the block if `insn_id`'s result is read from outside it.
    ///
    /// Compiled code keeps a block-local value in a machine register, so a use
    /// from another block would read whatever the interpreter's value table
    /// happened to hold. Lifted guest code does not produce such a use — state
    /// crosses blocks through registers and uniques, which are memory — but a
    /// pass that introduced one must make the block decline, not miscompile.
    fn check_confined(
        &self,
        insn_id: InstructionId,
        own: &FxHashSet<InstructionId>,
    ) -> Result<(), Unsupported> {
        let value = ValueId::Instruction(insn_id);
        if self
            .ctx
            .users_of(value)
            .iter()
            .any(|user| !own.contains(user))
        {
            return Err(Unsupported::Escapes("result used from another block"));
        }
        Ok(())
    }

    /// Writes every terminator operand this block defines into the export
    /// buffer, so the interpreter can read it back before running the
    /// terminator.
    ///
    /// Operands the interpreter can already resolve on its own — literals,
    /// varnode and temp addresses, results of earlier blocks it walked — need
    /// nothing, so a terminator that reads only those exports nothing.
    fn export_terminator_operands(
        &mut self,
        terminator: InstructionId,
        own: &FxHashSet<InstructionId>,
    ) -> Result<(), Unsupported> {
        let insn = qcode::value::Instruction::from_id(self.ctx, terminator);
        let operands: Vec<ValueId> = insn
            .mnemonic()
            .args()
            .into_iter()
            .map(|arg| arg.qualify(terminator.func))
            .collect();

        for operand in operands {
            let ValueId::Instruction(def) = operand else {
                continue;
            };
            if !own.contains(&def) {
                continue;
            }
            if self.exports.iter().any(|export| export.insn == def) {
                continue;
            }
            let value = *self
                .values
                .get(&def)
                .ok_or(Unsupported::Terminator("operand was not compiled"))?;
            let size = self.width_of(operand)?;
            let slot = self.exports.len();
            let widened = self.widen_to_u64(value);
            self.builder.ins().store(
                MemFlags::trusted(),
                widened,
                self.exports_arg,
                (slot * std::mem::size_of::<u64>()) as i32,
            );
            self.exports.push(Export { insn: def, size });
        }
        Ok(())
    }

    /// `value` zero-extended to the export buffer's slot width.
    fn widen_to_u64(&mut self, value: Value) -> Value {
        if self.builder.func.dfg.value_type(value) == types::I64 {
            value
        } else {
            self.builder.ins().uextend(types::I64, value)
        }
    }

    fn translate_one(&mut self, insn_id: InstructionId) -> Result<(), Unsupported> {
        let insn = qcode::value::Instruction::from_id(self.ctx, insn_id);
        let func = insn_id.func;
        let result = match insn.mnemonic() {
            &Mnemonic::Load(Load { space, ptr, size }) => {
                let space = space.qualify(func);
                let addr = self
                    .constant_address(ptr.qualify(func))
                    .ok_or(Unsupported::Access("non-constant address"))?;
                let ty = int_type(size)?;
                let slot = self.table.slot(space, addr as usize + size);
                let base = self.base(slot);
                Some(self.builder.ins().load(
                    ty,
                    MemFlags::trusted(),
                    base,
                    i32::try_from(addr).map_err(|_| Unsupported::Access("address too large"))?,
                ))
            }

            &Mnemonic::Store(Store {
                space,
                ptr,
                size,
                src,
            }) => {
                let space = space.qualify(func);
                let addr = self
                    .constant_address(ptr.qualify(func))
                    .ok_or(Unsupported::Access("non-constant address"))?;
                int_type(size)?;
                let value = self.operand(src.qualify(func), size)?;
                let slot = self.table.slot(space, addr as usize + size);
                let base = self.base(slot);
                self.builder.ins().store(
                    MemFlags::trusted(),
                    value,
                    base,
                    i32::try_from(addr).map_err(|_| Unsupported::Access("address too large"))?,
                );
                None
            }

            Mnemonic::Binop(Binary { op, lhs, rhs }) => {
                let lhs_id = lhs.qualify(func);
                let rhs_id = rhs.qualify(func);
                let width = self.width_of(lhs_id)?;
                if self.width_of(rhs_id)? != width {
                    return Err(Unsupported::Operand("mismatched operand widths"));
                }
                let a = self.operand(lhs_id, width)?;
                let b = self.operand(rhs_id, width)?;
                Some(self.binop(*op, a, b)?)
            }

            Mnemonic::Unop(Unary { op, src }) => {
                let src_id = src.qualify(func);
                let width = self.width_of(src_id)?;
                let value = self.operand(src_id, width)?;
                match op {
                    Unop::IntNot => Some(self.builder.ins().bnot(value)),
                    Unop::IntNegate => Some(self.builder.ins().ineg(value)),
                    _ => return Err(Unsupported::Mnemonic("float unop")),
                }
            }

            &Mnemonic::Zext(Zext { src, size }) => {
                let src_id = src.qualify(func);
                let from = self.width_of(src_id)?;
                let value = self.operand(src_id, from)?;
                let ty = int_type(size)?;
                Some(match from.cmp(&size) {
                    std::cmp::Ordering::Less => self.builder.ins().uextend(ty, value),
                    std::cmp::Ordering::Equal => value,
                    std::cmp::Ordering::Greater => self.builder.ins().ireduce(ty, value),
                })
            }

            &Mnemonic::Sext(Sext { src, size }) => {
                let src_id = src.qualify(func);
                let from = self.width_of(src_id)?;
                let value = self.operand(src_id, from)?;
                let ty = int_type(size)?;
                Some(match from.cmp(&size) {
                    std::cmp::Ordering::Less => self.builder.ins().sextend(ty, value),
                    std::cmp::Ordering::Equal => value,
                    std::cmp::Ordering::Greater => self.builder.ins().ireduce(ty, value),
                })
            }

            &Mnemonic::Range(Range { src, start, size }) => {
                let src_id = src.qualify(func);
                let from = self.width_of(src_id)?;
                let value = self.operand(src_id, from)?;
                let from_ty = int_type(from)?;
                let ty = int_type(size)?;
                let shifted = if start == 0 {
                    value
                } else {
                    let amount = self.builder.ins().iconst(from_ty, (start * 8) as i64);
                    self.builder.ins().ushr(value, amount)
                };
                Some(if from == size {
                    shifted
                } else {
                    self.builder.ins().ireduce(ty, shifted)
                })
            }

            // The status-flag primitives. x86 lifting emits these for almost
            // every arithmetic instruction, so without them a block of ordinary
            // integer code would be declined outright.
            &Mnemonic::PopCount(PopCount { src }) => {
                let src_id = src.qualify(func);
                let from = self.width_of(src_id)?;
                let value = self.operand(src_id, from)?;
                let counted = self.builder.ins().popcnt(value);
                let out = self.width_of(ValueId::Instruction(insn_id))?;
                Some(self.resize(counted, from, out)?)
            }

            &Mnemonic::Carry(Carry { lhs, rhs }) => {
                let (a, b) = self.pair(lhs.qualify(func), rhs.qualify(func))?;
                // Unsigned overflow: the sum wrapped below either operand.
                let sum = self.builder.ins().iadd(a, b);
                Some(self.builder.ins().icmp(IntCC::UnsignedLessThan, sum, a))
            }

            &Mnemonic::SCarry(SCarry { lhs, rhs }) => {
                let (a, b) = self.pair(lhs.qualify(func), rhs.qualify(func))?;
                // Signed overflow of `a + b`: both operands differ in sign from
                // the result.
                let sum = self.builder.ins().iadd(a, b);
                let a_differs = self.builder.ins().bxor(a, sum);
                let b_differs = self.builder.ins().bxor(b, sum);
                let both = self.builder.ins().band(a_differs, b_differs);
                let ty = self.builder.func.dfg.value_type(both);
                let zero = self.builder.ins().iconst(ty, 0);
                Some(self.builder.ins().icmp(IntCC::SignedLessThan, both, zero))
            }

            &Mnemonic::SBorrow(SBorrow { lhs, rhs }) => {
                let (a, b) = self.pair(lhs.qualify(func), rhs.qualify(func))?;
                // Signed overflow of `a - b`: the operands differ in sign and
                // the result takes the subtrahend's.
                let diff = self.builder.ins().isub(a, b);
                let operands_differ = self.builder.ins().bxor(a, b);
                let result_differs = self.builder.ins().bxor(a, diff);
                let both = self.builder.ins().band(operands_differ, result_differs);
                let ty = self.builder.func.dfg.value_type(both);
                let zero = self.builder.ins().iconst(ty, 0);
                Some(self.builder.ins().icmp(IntCC::SignedLessThan, both, zero))
            }

            // The user-ops the interpreter models without architectural
            // effect. Mirrored here exactly rather than approximated: `undef`
            // is SLEIGH's explicit write of an undefined value, which concrete
            // emulation resolves to zero, and the LOCK markers constrain
            // nothing observable in a single-threaded replay.
            Mnemonic::PCodeOp(op) => {
                let name = &self.ctx.shared.pcode_ops[op.id];
                match (name.as_ref(), op.args.as_slice()) {
                    ("undef", []) => {
                        let out = self.width_of(ValueId::Instruction(insn_id))?;
                        let ty = int_type(out)?;
                        Some(self.builder.ins().iconst(ty, 0))
                    }
                    ("LOCK" | "UNLOCK", []) => None,
                    _ => return Err(Unsupported::Mnemonic("user p-code op")),
                }
            }

            other => return Err(Unsupported::Mnemonic(other.opcode())),
        };

        if let Some(value) = result {
            self.values.insert(insn_id, value);
        }
        Ok(())
    }

    /// Both operands of a two-operand primitive, checked for equal width.
    fn pair(&mut self, lhs: ValueId, rhs: ValueId) -> Result<(Value, Value), Unsupported> {
        let width = self.width_of(lhs)?;
        if self.width_of(rhs)? != width {
            return Err(Unsupported::Operand("mismatched operand widths"));
        }
        Ok((self.operand(lhs, width)?, self.operand(rhs, width)?))
    }

    /// Widens or narrows `value` from `from` bytes to `to` bytes, unsigned.
    fn resize(&mut self, value: Value, from: usize, to: usize) -> Result<Value, Unsupported> {
        let ty = int_type(to)?;
        int_type(from)?;
        Ok(match from.cmp(&to) {
            std::cmp::Ordering::Less => self.builder.ins().uextend(ty, value),
            std::cmp::Ordering::Equal => value,
            std::cmp::Ordering::Greater => self.builder.ins().ireduce(ty, value),
        })
    }

    fn binop(&mut self, op: Binop, a: Value, b: Value) -> Result<Value, Unsupported> {
        let ins = self.builder.ins();
        Ok(match op {
            Binop::Int(int) => match int {
                IntBinop::Add => ins.iadd(a, b),
                IntBinop::Sub => ins.isub(a, b),
                IntBinop::And => ins.band(a, b),
                IntBinop::Or => ins.bor(a, b),
                IntBinop::Xor => ins.bxor(a, b),
                IntBinop::Mul => ins.imul(a, b),
                // Shift amounts are taken modulo the width by both QCode's
                // interpreter and Cranelift, so no masking is needed here.
                IntBinop::ShiftLeft => ins.ishl(a, b),
                IntBinop::ShiftRight => ins.ushr(a, b),
                IntBinop::SShiftRight => ins.sshr(a, b),
                IntBinop::Equal => ins.icmp(IntCC::Equal, a, b),
                IntBinop::NotEqual => ins.icmp(IntCC::NotEqual, a, b),
                IntBinop::Less => ins.icmp(IntCC::UnsignedLessThan, a, b),
                IntBinop::LessEqual => ins.icmp(IntCC::UnsignedLessThanOrEqual, a, b),
                IntBinop::SLess => ins.icmp(IntCC::SignedLessThan, a, b),
                IntBinop::SLessEqual => ins.icmp(IntCC::SignedLessThanOrEqual, a, b),
                // Division traps on zero in Cranelift but is defined by the
                // guest architecture's own rules, so it is left to the
                // interpreter rather than guessed at.
                IntBinop::Div | IntBinop::Rem | IntBinop::Sdiv | IntBinop::Srem => {
                    return Err(Unsupported::Mnemonic("integer division"));
                }
                _ => return Err(Unsupported::Mnemonic("integer binop")),
            },
            Binop::Float(_) => return Err(Unsupported::Mnemonic("float binop")),
            _ => return Err(Unsupported::Mnemonic("binop")),
        })
    }

    pub(crate) fn finish(mut self) {
        self.builder.ins().return_(&[]);
        self.builder.finalize();
    }
}
