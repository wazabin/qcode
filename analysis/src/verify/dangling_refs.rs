//! Verify no live instruction references a physically removed instruction.

use qcode::{context::Context, value::ValueId};

/// Every operand of a live instruction must reference a value that is still live. A
/// reference to a removed instruction is a dangling use-def edge — a pass deleted
/// a value without unlinking its users, leaving a use-after-free in the IR.
///
/// (`Context::instructions` yields live instructions only; we flag any of their
/// *operands* that point at an absent stable-arena ID.)
pub fn verify_no_dangling_refs(ctx: &Context) -> Vec<String> {
    let mut out = Vec::new();
    for insn in ctx.instructions() {
        for arg in insn.operands() {
            if let ValueId::Instruction(id) = arg
                && !ctx.contains_instruction(id)
            {
                out.push(format!(
                    "instruction {:?} references deleted instruction {id:?} (dangling use-def edge)",
                    insn.id
                ));
            }
        }
    }
    out
}
