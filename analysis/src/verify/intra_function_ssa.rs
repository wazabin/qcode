//! Verify SSA data operands stay within a single function.

use qcode::{context::Context, value::ValueId};

/// SSA is intra-function: an instruction's *data* operands that are themselves
/// instruction results or block parameters must belong to the same function as
/// the instruction using them. With per-function IR ownership this turns the
/// isolation goal into a checked invariant — a stray cross-function `ValueId`
/// operand means a pass leaked a handle across an outlining/reattribution
/// boundary without remapping it.
///
/// Control-flow block *targets* are deliberately **not** checked: a thunk or
/// tail-call `Branch` into another function's entry is a legitimate
/// cross-function edge (composite `EdgeId` routing handles it, and `split.rs`
/// later strips such edges).
pub fn verify_intra_function_ssa(ctx: &Context) -> Vec<String> {
    let mut out = Vec::new();
    for insn in ctx.instructions() {
        let func = insn.id.func;
        for arg in insn.mnemonic().args() {
            let operand_func = match arg {
                ValueId::Instruction(id) => Some(id.func),
                ValueId::BlockParam(id) => Some(id.func),
                _ => None,
            };
            if let Some(operand_func) = operand_func
                && operand_func != func
            {
                out.push(format!(
                    "instruction {:?} uses cross-function SSA operand {arg} \
                     (defined in {operand_func:?}, used in {func:?})",
                    insn.id
                ));
            }
        }
    }
    out
}
