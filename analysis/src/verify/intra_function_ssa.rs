//! Verify SSA data operands, CFG block targets, CFG edges, and block storage all
//! stay within a single function — strict IR locality (context-split ruling 2).

use qcode::{
    context::Context,
    value::{BasicBlock, ValueId},
};

/// IR references are intra-function on four axes:
///
/// 1. **Data operands.** An instruction's operands that are themselves
///    instruction results or block parameters must belong to the same function
///    as the instruction using them. A stray cross-function `ValueId` operand
///    means a pass leaked a handle across an outlining/split boundary without
///    remapping it.
/// 2. **CFG block targets.** A terminator's static `target_blocks()` (a `Branch`
///    target, both `CBranch` arms) must live in the instruction's own function.
///    Cross-function control flow is modelled as a *function-level* `TailCall`
///    terminator (which carries a `FunctionId`, not a block, so it is not a
///    `target_blocks()` entry), never as a foreign `BlockId`. The lifter emits
///    these tail calls at construction (context-split ruling 2), so no foreign
///    block target may survive — a violation here means a pass minted or
///    repointed a terminator at another function's block.
/// 3. **CFG edges.** Every edge incident to a block is stored in that block's own
///    function (`from.func == to.func`). A cross-function edge would make every
///    per-function CFG walk (dominators, liveness, rename) wander into a foreign
///    function. This is the release-build companion of the `add_cfg_edge`
///    `debug_assert`.
/// 4. **Block storage.** Every block is self-stored: `block.id.func` (its arena)
///    equals `block.parent` (its owner). A reattributed block — owned by one
///    function, stored in another's arena — is inaccessible to a function pass
///    borrowing only its own body.
///
/// With per-function IR ownership this turns the isolation goal into a checked
/// invariant on all four axes.
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
    for block in ctx.blocks() {
        // Axis 4: self-storage (arena == owner).
        if let Some(parent) = block.parent().map(|f| f.id)
            && parent != block.id.func
        {
            out.push(format!(
                "block {:?} is stored in {:?} but owned by {parent:?}; \
                 every block must be self-stored",
                block.id, block.id.func,
            ));
        }
        // Axis 3: every incident CFG edge is intra-function.
        for (_, succ) in block.successors() {
            let succ_func = BasicBlock::from_id(ctx, succ).id.func;
            if succ_func != block.id.func {
                out.push(format!(
                    "block {:?} has a cross-function CFG edge to {succ:?} \
                     (stored in {succ_func:?}); CFG edges must be intra-function",
                    block.id,
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
        value::{
            BasicBlock, Function,
            insn::{Branch, Mnemonic},
        },
    };
    use std::borrow::Cow;

    /// A `Branch` whose target block is owned by a *different* function is a strict
    /// IR locality violation (cross-function control flow must be a `TailCall`).
    /// The forbidden shape is built by replacing a terminator's mnemonic directly
    /// (never via `push_branch`, which would wire — and assert against — a
    /// cross-function CFG edge).
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

        // f's entry terminator is repointed at g's block — a foreign BlockId target,
        // installed without wiring a cross-function edge.
        let zero = ctx.get_const(0, 8).id();
        let term = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, f_entry))
            .push_return(zero)
            .id;
        ctx.replace_instruction_mnemonic(
            term,
            Mnemonic::Branch(Branch {
                target: g_entry,
                args: vec![],
            }),
        );

        let diags = verify_intra_function_ssa(&ctx);
        assert_eq!(
            diags.len(),
            1,
            "expected one cross-function block-target diagnostic, got {diags:?}"
        );
        assert!(diags[0].contains("cross-function block"));
    }

    /// A block owned by one function but stored in another's arena (a reattributed
    /// block) is a self-storage violation.
    #[test]
    fn flags_reattributed_block_storage() {
        let mut ctx = Context::new();
        let f = Function::make_at_addr(&mut ctx, 0x1000, Some(Cow::Borrowed("f"))).id;
        let g = Function::make_at_addr(&mut ctx, 0x2000, Some(Cow::Borrowed("g"))).id;
        // Born in g's arena, then reattributed to f: owner=f, storage=g.
        let block = BasicBlock::make(&mut ctx, g).id;
        Function::from_id_mut(&mut ctx, f).add_block(block);

        let diags = verify_intra_function_ssa(&ctx);
        assert!(
            diags.iter().any(|d| d.contains("must be self-stored")),
            "expected a self-storage diagnostic, got {diags:?}"
        );
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
