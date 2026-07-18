//! Verify that every basic block ends in a terminator.

use qcode::{context::Context, value::FunctionBody};

/// Every basic block must end in a terminator (branch / cbranch / return / …).
/// A block that is empty, or whose last instruction is an ordinary value op, has
/// fall-through control flow with no defined successor — a malformed CFG.
pub fn verify_block_terminators(ctx: &Context) -> Vec<String> {
    verify_block_terminators_scoped(ctx, super::Scope::All)
}

pub(crate) fn verify_block_terminators_scoped(
    ctx: &Context,
    scope: super::Scope<'_>,
) -> Vec<String> {
    let mut out = Vec::new();
    for fid in scope.function_ids(ctx) {
        for block in FunctionBody::from_id(ctx, fid).iter() {
            match block.iter().last() {
                None => out.push(format!(
                    "fn {fid:?} block {:?} is empty (no terminator)",
                    block.id
                )),
                Some(last) if !last.mnemonic().is_terminator() => out.push(format!(
                    "fn {fid:?} block {:?} does not end in a terminator (last op: `{}`)",
                    block.id,
                    last.mnemonic().opcode()
                )),
                Some(_) => {}
            }
        }
    }
    out
}
