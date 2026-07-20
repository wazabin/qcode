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
        let function = FunctionBody::from_id(ctx, fid);
        let fname = function.name().to_owned();
        for block in function.iter() {
            // Name and address the block, not just its arena id: a bare
            // `BlockId(153:2610)` says nothing about *which* block, and locating it
            // otherwise means re-running the lift under a debugger.
            let label = block.name().map(str::to_owned).unwrap_or_default();
            let at = block
                .address()
                .map(|a| format!(" at {a:#x}"))
                .unwrap_or_default();
            let where_ = format!("fn `{fname}` block `{label}`{at} ({:?})", block.id);
            match block.iter().last() {
                None => out.push(format!("{where_} is empty (no terminator)")),
                Some(last) if !last.mnemonic().is_terminator() => out.push(format!(
                    "{where_} does not end in a terminator (last op: `{}`)",
                    last.mnemonic().opcode()
                )),
                Some(_) => {}
            }
        }
    }
    out
}
