//! Verify SSA data operands, CFG block targets, CFG edges, and block storage all
//! stay within a single function — strict IR locality (context-split ruling 2).

use qcode::{context::Context, value::BasicBlock};

/// IR references are intra-function on four axes:
///
/// 1. **Data operands** — *now type-proven, no runtime check.* An instruction's
///    operands are stored as bare body-local `LocalValueId`s (context-split stage
///    6a): every operand read qualifies with the instruction's own `id.func`, so
///    an operand cannot name another function's SSA value or block parameter at
///    all. The old dynamic check (`operand.func == insn.func`) became vacuous
///    under localization and is dropped — the type system now discharges this
///    axis, exactly as it does the CFG-block-target axis below.
/// 2. **CFG block targets** — *now type-proven, no runtime check.* A terminator's
///    static targets (`Branch::target`, both `CBranch` arms) are stored as bare
///    body-local `LocalBlockId`s (context-split stage 6a), so they cannot name a
///    foreign function's block at all: every read qualifies with the terminator's
///    own `id.func`. Cross-function control flow is modelled as a *function-level*
///    `TailCall` terminator (which carries a `FunctionId`, not a block). The old
///    dynamic check (`target.func == insn.func`) became vacuous under localization
///    and is dropped — the type system now discharges this axis.
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
/// invariant on the edge and storage axes (the operand and block-target axes are
/// discharged by the `LocalValueId`/`LocalBlockId` storage types).
pub fn verify_intra_function_ssa(ctx: &Context) -> Vec<String> {
    let mut out = Vec::new();
    // Axes 1 (data operands) and 2 (CFG block targets) are discharged by the
    // `LocalValueId`/`LocalBlockId` storage types: an instruction's operands and a
    // terminator's targets are body-local indices in `insn.func`'s own arena and
    // cannot reference another function's value or block.
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
        context::Context,
        value::{BasicBlock, FunctionBody},
    };
    use std::borrow::Cow;

    // The former `flags_cross_function_block_target` test is gone: a foreign block
    // target is now unrepresentable. `Branch::target`/`CBranch` arms store a bare
    // `LocalBlockId`, so there is no `BlockId` field to point at another function's
    // block — the axis-2 invariant is discharged by the type, not a runtime check.

    /// Cross-arena block storage is rejected at construction time.
    #[test]
    #[should_panic(expected = "cannot add a block stored in another function arena")]
    fn rejects_reattributed_block_storage() {
        let mut ctx = Context::new();
        let f = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some(Cow::Borrowed("f"))).id;
        let g = FunctionBody::make_at_addr(&mut ctx, 0x2000, Some(Cow::Borrowed("g"))).id;
        // Born in g's arena; attaching it to f is unsupported.
        let block = BasicBlock::make(&mut ctx, g).id;
        FunctionBody::from_id_mut(&mut ctx, f).add_block(block);
    }

    /// An intra-function `Branch` (target owned by the same function) is clean.
    #[test]
    fn accepts_intra_function_block_target() {
        let mut ctx = Context::new();

        let f = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some(Cow::Borrowed("f"))).id;
        let entry = BasicBlock::make(&mut ctx, f).with_address(0x1000).id;
        let tail = BasicBlock::make(&mut ctx, f).with_address(0x1008).id;
        FunctionBody::from_id_mut(&mut ctx, f)
            .set_root(entry)
            .unwrap();
        {
            let zero = ctx.get_const(0, 8).id();
            (&mut ctx).builder(tail).push_return(zero);
        }
        (&mut ctx).builder(entry).push_branch(tail);

        assert!(verify_intra_function_ssa(&ctx).is_empty());
    }
}
