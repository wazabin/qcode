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
//! Guest RAM *is* handled here, but not by addressing it: an access to it owes
//! a translation, a permission check and a fault report, and those belong to
//! the MMU. What is inlined is the MMU's *answer* — a software TLB entry giving
//! the host address of a resident guest page, and the permission bytes sitting
//! beside that page's data. An access that the inlined form cannot settle (a
//! page not yet cached, one straddling a boundary, a permission it refuses)
//! calls back into the VM, which is the only implementation of what an access
//! means. See [`qcode_vm::jit_abi`].
//!
//! A faulting access stops the block where it happened and hands the fault back
//! through the function's status result. The stores that ran before it stay
//! applied, which is what the interpreter would have left behind too — so guest
//! RAM and the flat spaces are deliberately compiled *without* alias regions,
//! keeping Cranelift from reordering one store past another and making that
//! prefix something other than a prefix.
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

use cranelift::codegen::ir::BlockArg;
use cranelift::prelude::*;
use cranelift_module::FuncId;
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
use qcode_vm::{PAGE_PERM_OFFSET, PAGE_SIZE, TLB_ENTRIES, TlbEntry, perm};
use rustc_hash::{FxHashMap, FxHashSet};

/// Status a compiled block returns: it ran to the end of its body.
pub const BLOCK_OK: i64 = 0;
/// Status a compiled block returns: an access faulted and the block stopped
/// there. The fault itself is on the VM's memory.
pub const BLOCK_FAULT: i64 = 1;

/// Which side of memory an access is on, and what it therefore owes.
#[derive(Clone, Copy)]
enum Access {
    Load,
    Store,
}

impl Access {
    /// The permission bits every byte of the access must already have.
    ///
    /// [`perm::INIT`] is absent from the read set on purpose. Requiring it
    /// would be wrong whenever `check_uninit` is off — an uninitialized read is
    /// then perfectly legal — and the MMU refuses to cache a translation at all
    /// while it is on, so the inline path never runs under that rule.
    fn required(self) -> u8 {
        match self {
            Self::Load => perm::MAP | perm::READ,
            Self::Store => perm::MAP | perm::WRITE,
        }
    }
}

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

/// Which of the four integer divisions is being compiled.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Division {
    Unsigned,
    UnsignedRem,
    Signed,
    SignedRem,
}

impl Division {
    fn is_signed(self) -> bool {
        matches!(self, Self::Signed | Self::SignedRem)
    }

    /// Index of this operation's runtime helper, for the widths the machine
    /// cannot divide itself.
    fn helper(self) -> usize {
        match self {
            Self::Unsigned => 0,
            Self::UnsignedRem => 1,
            Self::Signed => 2,
            Self::SignedRem => 3,
        }
    }
}

/// Whether an over-wide shift leaves zero or the sign.
#[derive(Clone, Copy)]
enum ShiftKind {
    Logical,
    Arithmetic,
}

/// The machine integer type for a `size`-byte value.
///
/// 16 bytes is here because x86 integer code produces it constantly without any
/// 128-bit types being in sight: SLEIGH lifts a 64-bit `imul` as a widening
/// multiply into a 128-bit temporary, then slices the halves back out. Declining
/// that width left the multiply *and every block containing one* to the
/// interpreter, which was 99.9% of the interpreted block entries on the
/// benchmarks that lagged.
pub(crate) fn int_type(size: usize) -> Result<Type, Unsupported> {
    match size {
        1 => Ok(types::I8),
        2 => Ok(types::I16),
        4 => Ok(types::I32),
        8 => Ok(types::I64),
        16 => Ok(types::I128),
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
    /// Base of the software TLB, the third argument.
    tlb_arg: Value,
    /// The `VmMemory` the fallback helpers act on, the fourth argument.
    memory_arg: Value,
    /// The runtime's slow-path load and store, already referenced in this
    /// function.
    helpers: HelperRefs,
    /// The block that abandons the run and reports a fault, created on the
    /// first access that could take one.
    fault_block: Option<cranelift::prelude::Block>,
    /// Where the slow-path load leaves its result. One slot serves every
    /// access in the block: only one call is live at a time.
    load_slot: Option<codegen::ir::StackSlot>,
    /// Cached base pointer per space slot, loaded once per block rather than per
    /// access.
    bases: FxHashMap<usize, Value>,
    /// Values produced by instructions in this block, and imports already
    /// loaded.
    values: FxHashMap<InstructionId, Value>,
    /// This block's own instructions. A result read from outside them is an
    /// import: computed earlier — by the interpreter, or by compiled code
    /// that exported it — and waiting in the value buffer.
    own: FxHashSet<InstructionId>,
    /// Results of this code that instructions outside it read, and so must be
    /// written to the value buffer when it returns.
    escaping: Vec<InstructionId>,
    pub(crate) table: SpaceTable,
    /// Values read from the buffer on entry, in slot order. They take the
    /// first slots; exports follow.
    pub(crate) imports: Vec<Export>,
    /// Values the rest of the block reads, in slot order after the imports.
    pub(crate) exports: Vec<Export>,
}

/// The runtime entry points compiled code calls when an access cannot be
/// settled inline, as declared in the module.
#[derive(Debug, Clone, Copy)]
pub struct Helpers {
    pub load: FuncId,
    pub store: FuncId,
    /// The 128-bit divisions, in `Division` order.
    pub divisions: [FuncId; 4],
}

/// The same pair, resolved against one function being built.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HelperRefs {
    pub(crate) load: codegen::ir::FuncRef,
    pub(crate) store: codegen::ir::FuncRef,
    pub(crate) divisions: [codegen::ir::FuncRef; 4],
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
        helpers: HelperRefs,
    ) -> Self {
        let spaces_arg = builder.block_params(entry)[0];
        let exports_arg = builder.block_params(entry)[1];
        let tlb_arg = builder.block_params(entry)[2];
        let memory_arg = builder.block_params(entry)[3];
        Self {
            ctx,
            builder,
            spaces_arg,
            exports_arg,
            tlb_arg,
            memory_arg,
            helpers,
            fault_block: None,
            load_slot: None,
            bases: FxHashMap::default(),
            values: FxHashMap::default(),
            own: FxHashSet::default(),
            escaping: Vec::new(),
            table: SpaceTable::default(),
            imports: Vec::new(),
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

    /// Whether `space` is one this backend may address directly.
    ///
    /// Guest RAM is not, however constant its address: every access to it owes
    /// a permission check and a fault report, which is the MMU's to give. A
    /// RIP-relative operand resolves to a perfectly constant address and is
    /// still RAM — treating one as flat storage silently reads a fabricated,
    /// zero-filled space instead of the guest's memory. RAM is reached through
    /// [`Self::inline_access`] instead.
    fn is_flat(&self, space: MemorySpaceId) -> bool {
        space != MemorySpaceId::Shared(self.ctx.shared.default_space)
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
                // A QCode literal is at most 64 bits wide, so widening one to
                // a 16-byte operand loses nothing.
                Ok(self.constant(ty, u128::from(literal.value())))
            }
            ValueId::Instruction(insn) => {
                if let Some(&value) = self.values.get(&insn) {
                    return Ok(value);
                }
                self.import(insn, size)
            }
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

    /// Translates the body of `block` up to its first interrupting user
    /// operation, or all of it, then emits the exports the rest of the block
    /// will need.
    ///
    /// Returns the body index the interpreter continues from once the
    /// compiled code has run: the terminator's when nothing interrupts, and
    /// the interrupting op's otherwise. Compiled code runs the part before it,
    /// and the interpreter — positioned at the op by that index — raises the
    /// interrupt, exactly as it would have with no compiled code at all. The
    /// values the op and everything after it read from the compiled part are
    /// exported, the same way a terminator's operands are.
    ///
    /// `start` is the body index to begin at: 0 for a whole block, or the
    /// instruction after an interrupting op when the interpreter has run the
    /// block up to there and hands the rest back. Results of instructions
    /// before `start` that the compiled part reads are imported from the value
    /// buffer, where the caller places them from the interpreter's table.
    pub(crate) fn translate_body(
        &mut self,
        block: BlockId,
        start: usize,
    ) -> Result<usize, Unsupported> {
        let insns: Vec<InstructionId> = BasicBlock::from_id(self.ctx, block).instruction_ids();
        if insns.is_empty() {
            return Err(Unsupported::Terminator("block is empty"));
        }
        let body = &insns[..insns.len() - 1];
        if start > body.len() {
            return Err(Unsupported::Terminator("entry point past the body"));
        }
        let cut = body[start..]
            .iter()
            .position(|&insn| self.interrupts(insn))
            .map_or(body.len(), |offset| start + offset);
        self.own = insns.iter().copied().collect();

        for &insn_id in &body[start..cut] {
            self.translate_one(insn_id)?;
            self.note_escapes(insn_id);
        }

        let own = std::mem::take(&mut self.own);
        for &reader in &insns[cut..] {
            self.export_operands(reader, &own)?;
        }
        for &def in &std::mem::take(&mut self.escaping) {
            self.export(def)?;
        }
        Ok(cut)
    }

    /// Whether the interpreter stops at this instruction for the host: a user
    /// p-code op with no semantics of its own, which this backend has no way
    /// to run either. The ops it does model are the ones `translate_one`
    /// translates in place.
    fn interrupts(&self, insn_id: InstructionId) -> bool {
        let insn = qcode::value::Instruction::from_id(self.ctx, insn_id);
        let Mnemonic::PCodeOp(op) = insn.mnemonic() else {
            return false;
        };
        let name = &self.ctx.shared.pcode_ops[op.id];
        !matches!(
            (name.as_ref(), op.args.as_slice()),
            ("undef", []) | ("LOCK" | "UNLOCK", [])
        )
    }

    /// Notes that `insn_id`'s result is read from outside this block, so it
    /// has to be exported when the code returns.
    ///
    /// Compiled code keeps a block-local value in a machine register, which
    /// another block cannot see; exporting it to the value buffer is what
    /// lets that block import it. Lifted guest code rarely produces such a
    /// use — state crosses blocks through registers and uniques, which are
    /// memory — but an injected hook that splits a block does, and a value it
    /// computed before the split is read by the code after it.
    fn note_escapes(&mut self, insn_id: InstructionId) {
        let value = ValueId::Instruction(insn_id);
        if self
            .ctx
            .users_of(value)
            .iter()
            .any(|user| !self.own.contains(user))
        {
            self.escaping.push(insn_id);
        }
    }

    /// Writes every operand of `reader` that compiled code computed into the
    /// export buffer, so the interpreter can read it back before running
    /// `reader` itself — the terminator, or the tail of a block cut at an
    /// interrupt.
    ///
    /// Operands the interpreter can already resolve on its own — literals,
    /// varnode and temp addresses, results of earlier blocks it walked — need
    /// nothing, so a terminator that reads only those exports nothing.
    fn export_operands(
        &mut self,
        reader: InstructionId,
        own: &FxHashSet<InstructionId>,
    ) -> Result<(), Unsupported> {
        let insn = qcode::value::Instruction::from_id(self.ctx, reader);
        let operands: Vec<ValueId> = insn
            .mnemonic()
            .args()
            .into_iter()
            .map(|arg| arg.qualify(reader.func))
            .collect();

        for operand in operands {
            let ValueId::Instruction(def) = operand else {
                continue;
            };
            if !own.contains(&def) {
                continue;
            }
            self.export(def)?;
        }
        Ok(())
    }

    /// Writes the result of `def` into the next value-buffer slot, once,
    /// if compiled code produced it.
    ///
    /// A definition in this block that compiled code did not produce — an
    /// instruction past the cut — is the interpreter's to compute, and needs
    /// nothing.
    fn export(&mut self, def: InstructionId) -> Result<(), Unsupported> {
        if self.exports.iter().any(|export| export.insn == def) {
            return Ok(());
        }
        let Some(&value) = self.values.get(&def) else {
            return Ok(());
        };
        let size = self.width_of(ValueId::Instruction(def))?;
        // A slot is a `u64`. A terminator operand or an escaping value is a
        // condition or a small integer in practice, so this is a decline that
        // has never been observed rather than a width worth widening for.
        if size > std::mem::size_of::<u64>() {
            return Err(Unsupported::Terminator("operand wider than an export slot"));
        }
        let slot = self.imports.len() + self.exports.len();
        let widened = self.widen_to_u64(value);
        self.builder.ins().store(
            MemFlags::trusted(),
            widened,
            self.exports_arg,
            (slot * std::mem::size_of::<u64>()) as i32,
        );
        self.exports.push(Export { insn: def, size });
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

    /// Loads the result of `insn`, computed before this code was entered,
    /// from the next value-buffer slot.
    ///
    /// The result is there whenever the instruction ran: the interpreter
    /// files every result it computes, and compiled code exports the ones
    /// read from outside it. SSA guarantees the definition ran before any
    /// use, so a missing import is a runtime decline, not a wrong value.
    ///
    /// Imports are numbered from zero as they are met, and the exports emitted
    /// at the end of translation take the slots after them.
    fn import(&mut self, insn: InstructionId, size: usize) -> Result<Value, Unsupported> {
        if size > std::mem::size_of::<u64>() {
            return Err(Unsupported::Operand("import wider than a value slot"));
        }
        let slot = self.imports.len();
        let wide = self.builder.ins().load(
            types::I64,
            MemFlags::trusted(),
            self.exports_arg,
            (slot * std::mem::size_of::<u64>()) as i32,
        );
        let ty = int_type(size)?;
        let value = if ty == types::I64 {
            wide
        } else {
            self.builder.ins().ireduce(ty, wide)
        };
        self.imports.push(Export { insn, size });
        self.values.insert(insn, value);
        Ok(value)
    }

    fn translate_one(&mut self, insn_id: InstructionId) -> Result<(), Unsupported> {
        let insn = qcode::value::Instruction::from_id(self.ctx, insn_id);
        let func = insn_id.func;
        let result = match insn.mnemonic() {
            &Mnemonic::Load(Load { space, ptr, size }) => {
                let space = space.qualify(func);
                if !self.is_flat(space) {
                    let addr = self.guest_address(ptr.qualify(func))?;
                    let value = self.ram_load(addr, size)?;
                    return self.record(insn_id, Some(value));
                }
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
                if !self.is_flat(space) {
                    let addr = self.guest_address(ptr.qualify(func))?;
                    let value = self.operand(src.qualify(func), size)?;
                    self.ram_store(addr, value, size)?;
                    return Ok(());
                }
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
                    let amount = self.constant(from_ty, (start * 8) as u128);
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
                // Cranelift has no 128-bit `popcnt` lowering, and reaching it
                // would be a panic inside the backend rather than a decline.
                if from > std::mem::size_of::<u64>() {
                    return Err(Unsupported::Width(from));
                }
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
                let zero = self.constant(ty, 0);
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
                let zero = self.constant(ty, 0);
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
                        Some(self.constant(ty, 0))
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

    /// Flags for an access to guest memory or its permission bytes.
    ///
    /// Not [`MemFlags::trusted`]: that asserts alignment, and a guest access is
    /// unaligned whenever the guest says so. No alias region either, so
    /// Cranelift keeps these ordered against the flat-space stores around them
    /// — a faulting access must leave exactly the prefix that ran before it.
    ///
    /// The endianness is the host's, which is the guest's: this backend is
    /// built for an x86-64 guest on an x86-64 host, and the flat spaces have
    /// always been read the same way.
    fn guest_flags() -> MemFlags {
        // `notrap` is earned: the address has been translated and its
        // permissions checked before any of these run.
        MemFlags::new().with_notrap()
    }

    /// A constant of `ty`.
    ///
    /// `iconst` cannot make an `I128` — Cranelift builds one from its halves —
    /// so every constant the translator needs goes through here rather than
    /// each caller having to remember which widths are reachable.
    fn constant(&mut self, ty: Type, value: u128) -> Value {
        if ty == types::I128 {
            let low = self.builder.ins().iconst(types::I64, value as u64 as i64);
            let high = self
                .builder
                .ins()
                .iconst(types::I64, (value >> 64) as u64 as i64);
            self.builder.ins().iconcat(low, high)
        } else {
            self.builder.ins().iconst(ty, value as u64 as i64)
        }
    }

    /// `byte` repeated across every byte of `ty`.
    fn splat(byte: u8, ty: Type) -> i64 {
        let mut bits = [0u8; 8];
        for slot in bits.iter_mut().take(ty.bytes() as usize) {
            *slot = byte;
        }
        i64::from_le_bytes(bits)
    }

    /// The block that stops the run and reports a fault to the caller.
    ///
    /// Every faulting access jumps to this one block rather than carrying its
    /// own epilogue. It is only *created* here — its body is emitted by
    /// [`Self::finish`], because a builder may not leave a block that has not
    /// been terminated yet, and the block asking for this one is mid-access.
    fn fault_block(&mut self) -> cranelift::prelude::Block {
        if let Some(block) = self.fault_block {
            return block;
        }
        let block = self.builder.create_block();
        self.builder.set_cold_block(block);
        self.fault_block = Some(block);
        block
    }

    /// Continues in a fresh block, taking `target` instead when `cond` is
    /// non-zero.
    fn bail_if(&mut self, cond: Value, target: cranelift::prelude::Block) {
        let carry_on = self.builder.create_block();
        self.builder.ins().brif(cond, target, &[], carry_on, &[]);
        self.builder.switch_to_block(carry_on);
    }

    /// As [`Self::bail_if`], taking `target` when `cond` is zero.
    fn bail_unless(&mut self, cond: Value, target: cranelift::prelude::Block) {
        let carry_on = self.builder.create_block();
        self.builder.ins().brif(cond, carry_on, &[], target, &[]);
        self.builder.switch_to_block(carry_on);
    }

    /// A guest address as a 64-bit value.
    ///
    /// Unlike a flat access, a RAM pointer is a computed guest value: a
    /// literal, or something this block worked out. A varnode or temp *id*
    /// reaching here would mean the block is addressing RAM by the storage
    /// location's own address, which is not what a guest pointer is.
    fn guest_address(&mut self, ptr: ValueId) -> Result<Value, Unsupported> {
        let width = self.width_of(ptr)?;
        if width > 8 {
            return Err(Unsupported::Access("address wider than 64 bits"));
        }
        let value = self.operand(ptr, width)?;
        // Widened from the value's own type rather than the declared width. The
        // two agree on everything lifted so far, and if they ever stop, a
        // mismatched `iadd` below is a panic inside Cranelift rather than a
        // declined block.
        Ok(match self.builder.func.dfg.value_type(value) {
            types::I64 => value,
            ty if ty.is_int() => self.builder.ins().uextend(types::I64, value),
            _ => return Err(Unsupported::Access("address is not an integer")),
        })
    }

    /// Checks that a value really is the `size`-byte integer it is declared to
    /// be, so a machine store writes the width the guest asked for.
    fn checked_width(&self, value: Value, size: usize) -> Result<(), Unsupported> {
        if self.builder.func.dfg.value_type(value) == int_type(size)? {
            Ok(())
        } else {
            Err(Unsupported::Operand("value is not its declared width"))
        }
    }

    /// Translates `addr` and checks its permissions inline, branching to
    /// `fallback` at the first thing it cannot settle.
    ///
    /// Returns the host address of the access and the permission bytes it
    /// found there — the store path reuses the latter rather than loading it
    /// twice.
    fn inline_access(
        &mut self,
        addr: Value,
        size: usize,
        kind: Access,
        fallback: cranelift::prelude::Block,
    ) -> Result<(Value, Value), Unsupported> {
        let ty = int_type(size)?;
        let page_mask = (PAGE_SIZE - 1) as i64;

        // A translation covers one page, so an access spilling into the next
        // one is not this path's to make. A single byte never can.
        if size > 1 {
            let offset = self.builder.ins().band_imm(addr, page_mask);
            let last = self.builder.ins().iadd_imm(offset, size as i64 - 1);
            let spills = self.builder.ins().band_imm(last, !page_mask);
            self.bail_if(spills, fallback);
        }

        // The entry's *byte* offset comes straight out of the address: shifting
        // by the page bits would give the page number, so shifting by that
        // much less the entry size gives the offset of its entry directly.
        let entry_size = std::mem::size_of::<TlbEntry>() as i64;
        debug_assert!(entry_size.count_ones() == 1);
        let entry_bits = entry_size.trailing_zeros() as i64;
        let index_shift = PAGE_SIZE.trailing_zeros() as i64 - entry_bits;
        let shifted = self.builder.ins().ushr_imm(addr, index_shift);
        let offset = self
            .builder
            .ins()
            .band_imm(shifted, (TLB_ENTRIES as i64 - 1) << entry_bits);
        let entry = self.builder.ins().iadd(self.tlb_arg, offset);

        // The tag and the offset beside it are loaded as plain memory, not as
        // a no-alias region: the fallback rewrites this table, and a tag
        // hoisted above that call while its offset stayed below would pair one
        // page's tag with another page's address.
        let flags = MemFlags::trusted();
        let cached = self.builder.ins().load(types::I64, flags, entry, 0);
        let tag = self.builder.ins().band_imm(addr, !page_mask);
        let hit = self.builder.ins().icmp(IntCC::Equal, tag, cached);
        self.bail_unless(hit, fallback);

        let delta = self.builder.ins().load(types::I64, flags, entry, 8);
        let host = self.builder.ins().iadd(addr, delta);

        // Permissions are per byte and sit one fixed offset past the data, so
        // `size` of them load as one integer and check as one mask: every
        // required bit present in every byte is `required & !held == 0`.
        let held = self
            .builder
            .ins()
            .load(ty, Self::guest_flags(), host, PAGE_PERM_OFFSET as i32);
        let required = self
            .builder
            .ins()
            .iconst(ty, Self::splat(kind.required(), ty));
        let missing = self.builder.ins().band_not(required, held);
        self.bail_if(missing, fallback);

        Ok((host, held))
    }

    /// Whether a RAM access of `size` is one the inlined path is built for.
    ///
    /// The permission check splats its mask into an `Imm64` and the fallback
    /// passes the value as a `u64`, so both stop at eight bytes. Nothing wider
    /// reaches guest RAM in code built without SSE — the 128-bit values x86
    /// integer code produces live in lifter temporaries, which are flat.
    fn narrow_enough_for_ram(&self, size: usize) -> Result<(), Unsupported> {
        if size > std::mem::size_of::<u64>() {
            return Err(Unsupported::Access("guest RAM access wider than 8 bytes"));
        }
        Ok(())
    }

    /// The stack slot the slow-path load writes through.
    fn load_slot(&mut self) -> codegen::ir::StackSlot {
        if let Some(slot) = self.load_slot {
            return slot;
        }
        let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            8,
            3,
        ));
        self.load_slot = Some(slot);
        slot
    }

    /// Reads `size` bytes of guest RAM at `addr`.
    fn ram_load(&mut self, addr: Value, size: usize) -> Result<Value, Unsupported> {
        let ty = int_type(size)?;
        self.narrow_enough_for_ram(size)?;
        let done = self.builder.create_block();
        self.builder.append_block_param(done, ty);
        let fallback = self.builder.create_block();
        self.builder.set_cold_block(fallback);

        let (host, _) = self.inline_access(addr, size, Access::Load, fallback)?;
        let value = self.builder.ins().load(ty, Self::guest_flags(), host, 0);
        self.builder.ins().jump(done, &[BlockArg::from(value)]);

        self.builder.switch_to_block(fallback);
        let slot = self.load_slot();
        let out = self.builder.ins().stack_addr(types::I64, slot, 0);
        let width = self.builder.ins().iconst(types::I32, size as i64);
        let call = self
            .builder
            .ins()
            .call(self.helpers.load, &[self.memory_arg, addr, width, out]);
        let status = self.builder.inst_results(call)[0];
        let faulted = self.fault_block();
        self.bail_if(status, faulted);
        let wide = self.builder.ins().stack_load(types::I64, slot, 0);
        let narrowed = if ty == types::I64 {
            wide
        } else {
            self.builder.ins().ireduce(ty, wide)
        };
        self.builder.ins().jump(done, &[BlockArg::from(narrowed)]);

        self.builder.switch_to_block(done);
        Ok(self.builder.block_params(done)[0])
    }

    /// Writes `value` to `size` bytes of guest RAM at `addr`.
    fn ram_store(&mut self, addr: Value, value: Value, size: usize) -> Result<(), Unsupported> {
        let ty = int_type(size)?;
        self.narrow_enough_for_ram(size)?;
        self.checked_width(value, size)?;
        let done = self.builder.create_block();
        let fallback = self.builder.create_block();
        self.builder.set_cold_block(fallback);

        let (host, held) = self.inline_access(addr, size, Access::Store, fallback)?;
        self.builder
            .ins()
            .store(Self::guest_flags(), value, host, 0);
        // A written byte is a defined byte. The MMU's own `write` records this,
        // and the inline path has to as well: the bit is guest-visible state
        // the moment `check_uninit` is turned on, and a run whose JIT-written
        // bytes read as undefined would diverge from the same run interpreted.
        let init = self
            .builder
            .ins()
            .bor_imm(held, Self::splat(perm::INIT, ty));
        self.builder
            .ins()
            .store(Self::guest_flags(), init, host, PAGE_PERM_OFFSET as i32);
        self.builder.ins().jump(done, &[]);

        self.builder.switch_to_block(fallback);
        let width = self.builder.ins().iconst(types::I32, size as i64);
        let wide = self.widen_to_u64(value);
        let call = self
            .builder
            .ins()
            .call(self.helpers.store, &[self.memory_arg, addr, width, wide]);
        let status = self.builder.inst_results(call)[0];
        let faulted = self.fault_block();
        self.bail_if(status, faulted);
        self.builder.ins().jump(done, &[]);

        self.builder.switch_to_block(done);
        Ok(())
    }

    /// Files an instruction's result, if it has one, and reports success.
    fn record(&mut self, insn_id: InstructionId, result: Option<Value>) -> Result<(), Unsupported> {
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

    /// What an over-wide shift leaves behind.
    fn guard_shift(&mut self, shifted: Value, a: Value, amount: Value, kind: ShiftKind) -> Value {
        let ty = self.builder.func.dfg.value_type(a);
        let bits = u128::from(ty.bits());
        // Built as a value rather than with the `_imm` form: those take an
        // `Imm64`, which cannot describe a 128-bit operand.
        let width = self.constant(ty, bits);
        let in_range = self
            .builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, amount, width);
        let saturated = match kind {
            ShiftKind::Logical => self.constant(ty, 0),
            // Every bit becomes the sign bit.
            ShiftKind::Arithmetic => {
                let all = self.constant(ty, bits - 1);
                self.builder.ins().sshr(a, all)
            }
        };
        self.builder.ins().select(in_range, shifted, saturated)
    }

    /// Integer division, with the cases Cranelift traps on steered around.
    ///
    /// QCode leaves nothing undefined here, so neither does this:
    ///
    /// * a zero divisor yields zero, for all four operations;
    /// * signed division of the most negative value by `-1` wraps — the
    ///   quotient is that same value and the remainder is zero — where the
    ///   machine instruction would fault.
    ///
    /// Both are handled by dividing by `1` instead and then correcting, which
    /// needs no branch. The correction is only needed for the zero divisor:
    /// dividing the most negative value by `1` *already* gives exactly the
    /// wrapped quotient, and its remainder of zero, so the overflow case needs
    /// nothing but to be kept away from the divide.
    fn divide(&mut self, a: Value, b: Value, kind: Division) -> Result<Value, Unsupported> {
        let ty = self.builder.func.dfg.value_type(a);
        if ty == types::I128 {
            return Ok(self.divide_wide(a, b, kind));
        }
        let zero = self.constant(ty, 0);
        let one = self.constant(ty, 1);
        let by_zero = self.builder.ins().icmp(IntCC::Equal, b, zero);

        let mut avoid = by_zero;
        if kind.is_signed() {
            let most_negative = self.constant(ty, 1u128 << (ty.bits() - 1));
            let minus_one = self.constant(ty, u128::MAX);
            let a_is_min = self.builder.ins().icmp(IntCC::Equal, a, most_negative);
            let b_is_minus_one = self.builder.ins().icmp(IntCC::Equal, b, minus_one);
            let overflows = self.builder.ins().band(a_is_min, b_is_minus_one);
            avoid = self.builder.ins().bor(by_zero, overflows);
        }

        let divisor = self.builder.ins().select(avoid, one, b);
        let result = match kind {
            Division::Unsigned => self.builder.ins().udiv(a, divisor),
            Division::UnsignedRem => self.builder.ins().urem(a, divisor),
            Division::Signed => self.builder.ins().sdiv(a, divisor),
            Division::SignedRem => self.builder.ins().srem(a, divisor),
        };
        Ok(self.builder.ins().select(by_zero, zero, result))
    }

    /// A 128-bit division, through the runtime.
    ///
    /// x86-64 has no instruction for it and Cranelift no lowering to synthesise
    /// one, so this is a call — but only for the division itself. Declining
    /// instead would send the whole block to the interpreter, and on the
    /// benchmarks that divide, that block is the loop body.
    fn divide_wide(&mut self, a: Value, b: Value, kind: Division) -> Value {
        let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            16,
            4,
        ));
        let out = self.builder.ins().stack_addr(types::I64, slot, 0);
        // The halves are passed separately rather than as one 128-bit
        // argument: how a `__int128` travels is an ABI detail the two sides
        // would have to agree on silently, and getting it wrong corrupts
        // results rather than failing to link.
        let (a_low, a_high) = self.builder.ins().isplit(a);
        let (b_low, b_high) = self.builder.ins().isplit(b);
        let helper = self.helpers.divisions[kind.helper()];
        self.builder
            .ins()
            .call(helper, &[a_low, a_high, b_low, b_high, out]);
        let low = self.builder.ins().stack_load(types::I64, slot, 0);
        let high = self.builder.ins().stack_load(types::I64, slot, 8);
        self.builder.ins().iconcat(low, high)
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
                // An out-of-range shift is where QCode and Cranelift disagree,
                // so it cannot be left to the machine. QCode follows p-code: a
                // shift by the operand's width or more yields zero, and an
                // arithmetic one yields the sign. Cranelift instead *masks* the
                // amount, so a shift by 64 of a 64-bit value would be a shift by
                // none. Both are handled with a select rather than a branch.
                IntBinop::ShiftLeft => {
                    let shifted = ins.ishl(a, b);
                    return Ok(self.guard_shift(shifted, a, b, ShiftKind::Logical));
                }
                IntBinop::ShiftRight => {
                    let shifted = ins.ushr(a, b);
                    return Ok(self.guard_shift(shifted, a, b, ShiftKind::Logical));
                }
                IntBinop::SShiftRight => {
                    let shifted = ins.sshr(a, b);
                    return Ok(self.guard_shift(shifted, a, b, ShiftKind::Arithmetic));
                }
                IntBinop::Equal => ins.icmp(IntCC::Equal, a, b),
                IntBinop::NotEqual => ins.icmp(IntCC::NotEqual, a, b),
                IntBinop::Less => ins.icmp(IntCC::UnsignedLessThan, a, b),
                IntBinop::LessEqual => ins.icmp(IntCC::UnsignedLessThanOrEqual, a, b),
                IntBinop::SLess => ins.icmp(IntCC::SignedLessThan, a, b),
                IntBinop::SLessEqual => ins.icmp(IntCC::SignedLessThanOrEqual, a, b),
                // Division is not the guest architecture's business at this
                // level: QCode defines it completely — a zero divisor yields
                // zero, and signed division wraps rather than trapping. What
                // Cranelift does instead is *trap*, so the trapping cases are
                // steered away from rather than left to the interpreter.
                IntBinop::Div => return self.divide(a, b, Division::Unsigned),
                IntBinop::Rem => return self.divide(a, b, Division::UnsignedRem),
                IntBinop::Sdiv => return self.divide(a, b, Division::Signed),
                IntBinop::Srem => return self.divide(a, b, Division::SignedRem),
                _ => return Err(Unsupported::Mnemonic("integer binop")),
            },
            Binop::Float(_) => return Err(Unsupported::Mnemonic("float binop")),
            _ => return Err(Unsupported::Mnemonic("binop")),
        })
    }

    pub(crate) fn finish(mut self) {
        let ok = self.builder.ins().iconst(types::I32, BLOCK_OK);
        self.builder.ins().return_(&[ok]);
        // Now that the body's last block is terminated, the shared fault
        // epilogue can be filled.
        if let Some(block) = self.fault_block {
            self.builder.switch_to_block(block);
            let status = self.builder.ins().iconst(types::I32, BLOCK_FAULT);
            self.builder.ins().return_(&[status]);
        }
        // The fault block and the continuations every inline check splits off
        // are reached only by branches already emitted, so they can all be
        // sealed at once now that no more will be added.
        self.builder.seal_all_blocks();
        self.builder.finalize();
    }
}
