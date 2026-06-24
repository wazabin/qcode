//! Intrinsic-recognition sub-pass: rewrite raw-IR idioms into pure
//! [`Mnemonic::Intrinsic`] nodes.
//!
//! Recognition is driven by the descriptors registered in `qcode` core: each
//! intrinsic optionally provides a `recognize` matcher gated by a `root_op`.
//! This pass looks up only the matchers whose root matches the current
//! instruction (e.g. `rol`'s idiom roots at `or`), so cost stays proportional
//! to the instructions actually shaped like an idiom — not the number of
//! registered intrinsics.
//!
//! A matched root is rewritten to the intrinsic; the now-unused shift/or math
//! it subsumed becomes pure-dead and is reclaimed by DCE.

use qcode::{
    context::Context,
    value::insn::{Binop, Intrinsic, Mnemonic, RootOp, recognizers_for},
};

use super::walk::{Claim, Editor, InsnCtx, SubPass};

/// Rewrite recognized idioms into intrinsics.
pub(super) struct Recognize;

impl SubPass for Recognize {
    type State = ();

    fn on_insn(&self, ctx: &mut Context, _state: &mut (), ic: &InsnCtx, ed: &mut Editor) -> Claim {
        if ic.size == 0 {
            return Claim::Pass;
        }
        let Some(root) = root_op_of(ic.mnemonic) else {
            return Claim::Pass;
        };

        for &id in recognizers_for(root) {
            let Some(recognize) = id.desc().recognize else {
                continue;
            };
            if let Some(args) = recognize(ctx, ic.insn_id) {
                ed.replace_with_new_insn(
                    ctx,
                    ic.block_id,
                    ic.insn_id,
                    Mnemonic::Intrinsic(Intrinsic { id, args }),
                    ic.size,
                );
                return Claim::Done;
            }
        }
        Claim::Pass
    }
}

/// The recognition root of `m`, if it can root an idiom.
fn root_op_of(m: &Mnemonic) -> Option<RootOp> {
    match m {
        Mnemonic::Binop(bin) => match bin.op {
            Binop::Int(op) => Some(RootOp::IntBinop(op)),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::gvn::{constant_fold_function, gvn_function};
    use qcode::{context::Context, value::insn::Mnemonic};
    use qcode_macro::qcode;

    /// Live (still-parented) instructions only — the arena retains removed ones.
    fn live_intrinsic_named(ctx: &Context, name: &str) -> bool {
        ctx.instructions().any(|insn| {
            insn.parent().is_some()
                && matches!(insn.mnemonic(), Mnemonic::Intrinsic(i) if i.id.name() == name)
        })
    }

    fn has_live_intrinsic(ctx: &Context) -> bool {
        ctx.instructions().any(|insn| {
            insn.parent().is_some() && matches!(insn.mnemonic(), Mnemonic::Intrinsic(_))
        })
    }

    /// `(x << 8) | (x >> 24)` over 32 bits is recognized as `rol(x, 8)`.
    #[test]
    fn recognizes_rotate_left_idiom() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 X;
            fn f:
                <entry>
                    %x  = load(i32, &X);
                    %s1 = %x << i32 8;
                    %s2 = %x >> i32 24;
                    %r  = %s1 | %s2;
                    return [%r];
            "
        );

        gvn_function(&mut ctx, f, None);

        // The `or` root is rewritten to a `rol` intrinsic; %r forwards to it.
        assert!(
            live_intrinsic_named(&ctx, "rol"),
            "shift-or idiom should be recognized as rol"
        );
    }

    /// A `rol` on two constants folds to a literal.
    #[test]
    fn folds_constant_rol() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
                <entry>
                    %r = $rol(i32 0x12345678, i32 8);
                    return [%r];
            "
        );

        constant_fold_function(&mut ctx, f);

        // No live intrinsic remains; it folded to the rotated constant.
        assert!(!has_live_intrinsic(&ctx), "constant rol should fold away");
    }

    /// `rol(x, 0)` simplifies to `x`.
    #[test]
    fn simplifies_rotate_by_zero() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 X;
            fn f:
                <entry>
                    %x = load(i32, &X);
                    %r = $rol(%x, i32 0);
                    return [%r];
            "
        );

        gvn_function(&mut ctx, f, None);

        assert!(!has_live_intrinsic(&ctx), "rol(x, 0) should simplify to x");
    }

    /// The recognized intrinsic prints with the `$` sigil.
    #[test]
    fn intrinsic_prints_with_sigil() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 X;
            fn f:
                <entry>
                    %x  = load(i32, &X);
                    %s1 = %x << i32 8;
                    %s2 = %x >> i32 24;
                    %r  = %s1 | %s2;
                    return [%r];
            "
        );
        gvn_function(&mut ctx, f, None);

        assert!(
            ctx.to_string().contains("$rol("),
            "intrinsic should print with the $ sigil, got:\n{ctx}"
        );
    }
}
