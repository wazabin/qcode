//! Terminator rewriting shared by the CFG and DCE transforms.

use qcode::value::{
    BlockId, FunctionBody, QCodeView, ValueId,
    insn::{Branch, Mnemonic},
    util::base_ref::BaseRef,
};

use crate::PassCtx;

pub fn replace_terminator_with_branch<'a, 'str>(
    body: &'a mut FunctionBody<'str>,
    cx: PassCtx<'a, 'str>,
    block: BlockId,
    target: BlockId,
    args: Vec<ValueId>,
) {
    let mut old_successors = cx
        .body_view(body)
        .block_ref(block)
        .successors()
        .map(|(edge, _)| edge)
        .collect::<Vec<_>>();
    old_successors.sort_unstable();
    old_successors.dedup();
    for edge in old_successors {
        body.remove_cfg_edge(edge);
    }

    // Reuse the existing terminator only if the block actually ends in one. The
    // freshly-created unrolled blocks hold only copied *body* instructions (no
    // terminator yet); their last instruction is a real value (e.g. the induction
    // increment), which must not be clobbered into the branch — doing so destroys
    // that value and, when it is the exit argument, yields a branch that passes
    // itself. In that case append the branch instead.
    let term_id = cx
        .body_view(body)
        .block_ref(block)
        .instruction_ids()
        .last()
        .copied()
        .filter(|&id| cx.body_view(body).insn_ref(id).mnemonic().is_terminator());
    let local_target = target.localize(block.func);
    let args: Vec<_> = args
        .into_iter()
        .map(|arg| arg.localize(block.func))
        .collect();
    if let Some(term_id) = term_id {
        body.replace_instruction_mnemonic(
            term_id,
            Mnemonic::Branch(Branch {
                target: local_target,
                args,
            }),
        );
    } else {
        let branch = body.push_mnemonic(
            cx.shr(),
            Mnemonic::Branch(Branch {
                target: local_target,
                args,
            }),
            0,
        );
        let end = cx.body_view(body).block_ref(block).instruction_ids().len();
        {
            // TODO(5b-ii): `BaseRef::insert_insn_at_index` is not mirrored on
            // `FunctionBody`; go through a temporary host.
            let mut host = cx.host(body);
            BaseRef::new(host.reborrow(), block).insert_insn_at_index(end, branch);
        }
    }
    body.add_cfg_edge(block, target);
}
