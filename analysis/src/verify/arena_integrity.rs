//! Thin analysis-layer entry point for core-owned body-arena validation.

use qcode::context::Context;

pub fn verify_body_arena_integrity(ctx: &Context<'_>) -> Vec<String> {
    qcode::verify_body_arena_integrity(ctx)
}

#[cfg(test)]
mod tests {
    use qcode::{
        context::Context,
        value::{BasicBlock, FunctionBody},
    };
    use qcode_macro::qcode;

    #[test]
    fn aggregate_verifier_stops_before_traversing_removed_target() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
            <entry>
                goto <target>;
            <target>
                return at i64 0;
            "
        );
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let edge = *ctx.block(entry).edges.iter().next().expect("edge");
        let target = ctx.edge(f, edge).to;
        BasicBlock::from_id_mut(&mut ctx, target).delete(f);

        let diagnostics = crate::verify::verify(&ctx);
        assert!(
            diagnostics
                .iter()
                .any(|d| d.contains("targets removed block")),
            "{diagnostics:#?}"
        );
    }
}
