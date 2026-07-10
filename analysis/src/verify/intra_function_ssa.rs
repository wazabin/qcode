//! Verify SSA data operands *and* CFG block targets stay within a single
//! function — strict IR locality (context-split ruling 2).

use qcode::{context::Context, value::ValueId};

/// IR references are intra-function on two axes:
///
/// 1. **Data operands.** An instruction's operands that are themselves
///    instruction results or block parameters must belong to the same function
///    as the instruction using them. A stray cross-function `ValueId` operand
///    means a pass leaked a handle across an outlining/reattribution boundary
///    without remapping it.
/// 2. **CFG block targets.** A terminator's static `target_blocks()` (a `Branch`
///    target, both `CBranch` arms) must live in the instruction's own function.
///    Cross-function control flow is modelled as a *function-level* `TailCall`
///    terminator (which carries a `FunctionId`, not a block, so it is not a
///    `target_blocks()` entry), never as a foreign `BlockId`. The entry
///    normalization (`split_overlapping_functions`, run at the optimization
///    choke point and every lifter discovery round) rewrites cross-function tail
///    jumps to `TailCall` and force-splits mid-function landings, so by the time
///    any pass runs no foreign block target may survive — a violation here means
///    a pass minted or repointed a terminator at another function's block.
///
/// With per-function IR ownership this turns the isolation goal into a checked
/// invariant on both axes.
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
        for target in insn.mnemonic().target_blocks() {
            if target.func != func {
                out.push(format!(
                    "instruction {:?} branches to cross-function block {target:?} \
                     (owned by {:?}, used in {func:?}); cross-function control flow \
                     must be a TailCall, not a foreign block target",
                    insn.id, target.func,
                ));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::verify_intra_function_ssa;
    use qcode::{
        builder::Builder,
        context::Context,
        value::{BasicBlock, Function},
    };
    use std::borrow::Cow;

    /// A `Branch` whose target block is owned by a *different* function is a strict
    /// IR locality violation (cross-function control flow must be a `TailCall`).
    #[test]
    fn flags_cross_function_block_target() {
        let mut ctx = Context::new();

        let f = Function::make_at_addr(&mut ctx, 0x1000, Some(Cow::Borrowed("f"))).id;
        let g = Function::make_at_addr(&mut ctx, 0x2000, Some(Cow::Borrowed("g"))).id;

        let f_entry = BasicBlock::make(&mut ctx, f).with_address(0x1000).id;
        Function::from_id_mut(&mut ctx, f)
            .set_root(f_entry)
            .unwrap();
        let g_entry = BasicBlock::make(&mut ctx, g).with_address(0x2000).id;
        Function::from_id_mut(&mut ctx, g)
            .set_root(g_entry)
            .unwrap();
        {
            let zero = ctx.get_const(0, 8).id();
            Builder::from_block(BasicBlock::from_id_mut(&mut ctx, g_entry)).push_return(zero);
        }

        // f's entry branches into g's block — a foreign BlockId target.
        Builder::from_block(BasicBlock::from_id_mut(&mut ctx, f_entry)).push_branch(g_entry);

        let diags = verify_intra_function_ssa(&ctx);
        assert_eq!(
            diags.len(),
            1,
            "expected one cross-function block-target diagnostic, got {diags:?}"
        );
        assert!(diags[0].contains("cross-function block"));
    }

    /// An intra-function `Branch` (target owned by the same function) is clean.
    #[test]
    fn accepts_intra_function_block_target() {
        let mut ctx = Context::new();

        let f = Function::make_at_addr(&mut ctx, 0x1000, Some(Cow::Borrowed("f"))).id;
        let entry = BasicBlock::make(&mut ctx, f).with_address(0x1000).id;
        let tail = BasicBlock::make(&mut ctx, f).with_address(0x1008).id;
        Function::from_id_mut(&mut ctx, f).set_root(entry).unwrap();
        {
            let zero = ctx.get_const(0, 8).id();
            Builder::from_block(BasicBlock::from_id_mut(&mut ctx, tail)).push_return(zero);
        }
        Builder::from_block(BasicBlock::from_id_mut(&mut ctx, entry)).push_branch(tail);

        assert!(verify_intra_function_ssa(&ctx).is_empty());
    }
}
