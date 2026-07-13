//! Verify no live instruction references a physically removed SSA value.

use qcode::{context::Context, value::ValueId};

/// Every operand of a live instruction must reference a value that is still live. A
/// reference to a removed instruction or block parameter is a dangling use-def edge — a pass deleted
/// a value without unlinking its users, leaving a use-after-free in the IR.
///
/// (`Context::instructions` yields live instructions only; we flag any of their
/// *operands* that point at an absent stable-arena ID.)
pub fn verify_no_dangling_refs(ctx: &Context) -> Vec<String> {
    let mut out = Vec::new();
    for insn in ctx.instructions() {
        for arg in insn.operands() {
            let removed = match arg {
                ValueId::Instruction(id) => !ctx.contains_instruction(id),
                ValueId::BlockParam(id) => !ctx.contains_block_param(id),
                _ => false,
            };
            if removed {
                out.push(format!(
                    "instruction {:?} references removed value {arg:?} (dangling use-def edge)",
                    insn.id,
                ));
            }
        }
    }
    out
}
