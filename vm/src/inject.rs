//! Rewriting lifted code before it runs: the mechanism every hook is built on.
//!
//! An injector is handed each block once it has been lifted and cleaned, and
//! again whenever the block grows, and may edit it through the ordinary
//! builder: insert a [`VM_INTERRUPT`] where the host wants control, add loads
//! and stores that count or log, rewrite an operation. The interpreter and
//! the JIT both run whatever the block then contains, so instrumentation is
//! compiled along with the code it instruments and costs nothing where none
//! was injected.
//!
//! # Idempotence
//!
//! A block is offered to an injector more than once: at discovery, after
//! absorption has folded more guest instructions into it, and after a new
//! injector is registered. The injector must therefore recognise its own
//! earlier edits and not repeat them. [`is_interrupt`] is what the built-in
//! injectors use for that.
//!
//! Most hooks are better written against [`crate::hook`], which chooses
//! sites, hides the builder and handles idempotence; this is the layer
//! underneath it.

use qcode::{
    context::Context,
    value::{
        BasicBlock, BlockId, InstructionId, ValueId, ValueRef,
        insn::{Mnemonic, VM_INTERRUPT},
    },
};

/// Edits a block before it runs. See the [module documentation](self).
pub trait CodeInjector {
    /// Rewrites `block`, which is lifted, cleaned and about to be entered.
    /// Must leave the block terminated, and must be idempotent.
    fn inject(&mut self, ctx: &mut Context<'static>, block: BlockId);
}

/// The first instruction of `block` lifted from guest address `addr`, if the
/// block covers it.
pub fn instruction_at_address(
    ctx: &Context<'static>,
    block: BlockId,
    addr: u64,
) -> Option<InstructionId> {
    BasicBlock::from_id(ctx, block)
        .instructions()
        .find(|insn| insn.address() == Some(addr))
        .map(|insn| insn.id)
}

/// Whether `insn` is a [`VM_INTERRUPT`] whose first operand is the literal
/// `code` and, when `arg` is given, whose second is the literal `arg`.
pub fn is_interrupt(
    ctx: &Context<'static>,
    insn: InstructionId,
    code: u64,
    arg: Option<u64>,
) -> bool {
    let insn = qcode::value::Instruction::from_id(ctx, insn);
    let Mnemonic::PCodeOp(op) = insn.mnemonic() else {
        return false;
    };
    if ctx.shared.pcode_ops[op.id].as_ref() != VM_INTERRUPT {
        return false;
    }
    let literal = |index: usize| -> Option<u64> {
        let value = op.args.get(index)?.qualify(insn.id.func);
        match ValueRef::new(value, ctx) {
            ValueRef::Literal(literal) => Some(literal.value()),
            _ => None,
        }
    };
    literal(0) == Some(code) && arg.is_none_or(|arg| literal(1) == Some(arg))
}

/// Inserts `vm.interrupt(code, args...)` into `block`, before `before` or at
/// the block's start when `before` is `None`. The op declares no result and
/// is stamped with guest address `address`, which is what the interrupt then
/// reports as its `pc`. `args` are any values visible at that point.
pub fn insert_interrupt(
    ctx: &mut Context<'static>,
    block: BlockId,
    before: Option<InstructionId>,
    address: Option<u64>,
    code: u64,
    args: &[ValueId],
) -> InstructionId {
    let op = ctx.shared.vm_interrupt_op();
    let operands: Vec<ValueId> = std::iter::once(ctx.shared.get_const(code, 8))
        .chain(args.iter().copied())
        .collect();
    let mut builder = ctx.builder(block);
    match before {
        Some(before) => builder.set_insert_point_before(before),
        None => builder.set_insert_point_to_start(),
    }
    if let Some(address) = address {
        builder.set_address(address);
    }
    builder.push_pcode_op(op, operands, None, 0).id
}
