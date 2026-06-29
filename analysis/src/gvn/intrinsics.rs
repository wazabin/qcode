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
    value::insn::{Binop, IntrinsicApp, Mnemonic, RootOp, recognizers_for},
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
            if let Some(args) = id.desc().recognize(ctx, ic.insn_id) {
                // Recognized intrinsics (rol/ror) are width-preserving, so the
                // root's width is the result width.
                ed.replace_with_new_insn(
                    ctx,
                    ic.block_id,
                    ic.insn_id,
                    Mnemonic::Intrinsic(IntrinsicApp { id, args }),
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
    use qcode::{
        context::Context,
        value::{ValueId, insn::Mnemonic},
    };
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

    /// The constant rotate amount of the single live intrinsic named `name`.
    fn live_rotate_amount(ctx: &Context, name: &str) -> Option<u64> {
        ctx.instructions().find_map(|insn| {
            insn.parent()?;
            let Mnemonic::Intrinsic(i) = insn.mnemonic() else {
                return None;
            };
            if i.id.name() != name {
                return None;
            }
            match i.args[1] {
                ValueId::Literal(lid) => Some(ctx.get_literal_value(lid)),
                _ => None,
            }
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
                    %x  = load(X:4, &X);
                    %s1 = %x << i32 8;
                    %s2 = %x >> i32 24;
                    %r  = %s1 | %s2;
                    return at %r;
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
                    return at %r;
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
                    %x = load(X:4, &X);
                    %r = $rol(%x, i32 0);
                    return at %r;
            "
        );

        gvn_function(&mut ctx, f, None);

        assert!(!has_live_intrinsic(&ctx), "rol(x, 0) should simplify to x");
    }

    /// `ror(rol(x, c), c)` cancels to `x`, leaving no live intrinsic.
    #[test]
    fn inverse_rotate_cancels() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 X;
            fn f:
                <entry>
                    %x = load(X:4, &X);
                    %l = $rol(%x, i32 8);
                    %r = $ror(%l, i32 8);
                    return at %r;
            "
        );

        gvn_function(&mut ctx, f, None);

        // The outer `ror` cancels against the inner `rol`: it is gone and the
        // return forwards straight to `%x`. (The now-unused inner `rol` is left
        // for a DCE pass to reclaim.)
        assert!(
            !live_intrinsic_named(&ctx, "ror"),
            "ror(rol(x, c), c) should cancel away"
        );
        assert!(
            ctx.to_string().contains("return at i32 %x"),
            "return should forward to x after cancellation, got:\n{ctx}"
        );
    }

    /// A rotate by a constant ≥ width is normalised modulo the bit width:
    /// `rol(x, 40)` over 32 bits becomes `rol(x, 8)`.
    #[test]
    fn constant_amount_reduced_modulo_width() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 X;
            fn f:
                <entry>
                    %x = load(X:4, &X);
                    %r = $rol(%x, i32 40);
                    return at %r;
            "
        );

        gvn_function(&mut ctx, f, None);

        assert_eq!(
            live_rotate_amount(&ctx, "rol"),
            Some(8),
            "rol(x, 40) over 32 bits should normalise to rol(x, 8)"
        );
    }

    /// A rotate by a whole number of turns is the identity: `rol(x, 32)` over
    /// 32 bits collapses to `x`.
    #[test]
    fn whole_turn_rotate_is_identity() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 X;
            fn f:
                <entry>
                    %x = load(X:4, &X);
                    %r = $rol(%x, i32 32);
                    return at %r;
            "
        );

        gvn_function(&mut ctx, f, None);

        assert!(
            !has_live_intrinsic(&ctx),
            "rol(x, 32) over 32 bits should simplify to x"
        );
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
                    %x  = load(X:4, &X);
                    %s1 = %x << i32 8;
                    %s2 = %x >> i32 24;
                    %r  = %s1 | %s2;
                    return at %r;
            "
        );
        gvn_function(&mut ctx, f, None);

        assert!(
            ctx.to_string().contains("$rol("),
            "intrinsic should print with the $ sigil, got:\n{ctx}"
        );
    }
}
