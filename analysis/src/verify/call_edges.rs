//! Verify that calls have at most one caller-side continuation edge.

use qcode::{context::Context, value::insn::Mnemonic};

/// A returning direct or indirect call has one caller-side CFG successor: the
/// block where execution resumes. A known-noreturn call may have no successor,
/// but multiple outgoing edges (including duplicate parallel edges) are always
/// malformed.
pub fn verify_call_edges(ctx: &Context) -> Vec<String> {
    let mut out = Vec::new();
    for block in ctx.blocks() {
        let Some(last) = block.iter().last() else {
            continue;
        };
        if !matches!(last.mnemonic(), Mnemonic::Call(_) | Mnemonic::CallInd(_)) {
            continue;
        }

        let successors: Vec<_> = block
            .successors()
            .map(|(edge, target)| (edge, target))
            .collect();
        if successors.len() > 1 {
            out.push(format!(
                "fn {:?} call block {:?} has {} continuation edges {successors:?}; calls may have at most one continuation edge",
                block.id.func,
                block.id,
                successors.len(),
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::verify_call_edges;
    use qcode::{
        context::Context,
        value::{BasicBlock, FunctionBody, QCodeMut},
    };

    fn call_case() -> (
        Context<'static>,
        qcode::value::BlockId,
        qcode::value::BlockId,
    ) {
        let mut ctx = Context::new();
        let callee = FunctionBody::make(&mut ctx, "callee".into()).unwrap().id;
        let caller = FunctionBody::make(&mut ctx, "caller".into()).unwrap().id;
        let call_block = BasicBlock::make(&mut ctx, caller).id;
        let continuation = BasicBlock::make(&mut ctx, caller).id;
        FunctionBody::from_id_mut(&mut ctx, caller)
            .set_root(call_block)
            .unwrap();
        ctx.builder(call_block).push_call(callee);
        (ctx, call_block, continuation)
    }

    #[test]
    fn accepts_zero_or_one_call_continuation() {
        let (mut ctx, call_block, continuation) = call_case();
        assert!(verify_call_edges(&ctx).is_empty());

        ctx.add_cfg_edge(call_block, continuation);
        assert!(verify_call_edges(&ctx).is_empty());
    }

    #[test]
    fn rejects_multiple_call_continuations() {
        let (mut ctx, call_block, first) = call_case();
        let second = BasicBlock::make(&mut ctx, call_block.func).id;
        ctx.add_cfg_edge(call_block, first);
        ctx.add_cfg_edge(call_block, second);

        let diagnostics = verify_call_edges(&ctx);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("has 2 continuation edges"));
    }

    #[test]
    fn rejects_duplicate_parallel_call_edges() {
        let (mut ctx, call_block, continuation) = call_case();
        ctx.add_cfg_edge(call_block, continuation);
        ctx.add_cfg_edge(call_block, continuation);

        let diagnostics = verify_call_edges(&ctx);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("has 2 continuation edges"));
    }
}
