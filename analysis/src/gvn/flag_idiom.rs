//! Flag-idiom sub-pass: collapse x86 signed-compare flag chains.

use qcode::value::{
    ValueId,
    insn::{Binary, Binop, IntBinop, Mnemonic},
    util::{base_ref::HostRef, host_mut::HostMut},
};

use super::fold::const_value;
use std::any::Any;

use super::walk::{Claim, Editor, InsnCtx, SubPass};

/// Collapse the signed-compare flag idiom into a single `s<`, materializing the
/// replacement before the matched instruction and forwarding its uses; the
/// dead flag math falls to DCE. Fully host-routed (own-function reads only), so
/// it runs over either the module or a checked-out function.
pub(super) struct FlagIdiom;

impl<'str, H: HostMut<'str>> SubPass<'str, H> for FlagIdiom {
    fn init_state(&self) -> Box<dyn Any> {
        Box::new(())
    }

    fn clone_state(&self, _state: &dyn Any) -> Box<dyn Any> {
        Box::new(())
    }

    fn on_insn(&self, host: &mut H, _state: &mut dyn Any, ic: &InsnCtx, ed: &mut Editor) -> Claim {
        if ic.mnemonic.is_terminator() || ic.size == 0 {
            return Claim::Pass;
        }
        match simplify_flag_idiom(host.read_host(), ic.mnemonic) {
            Some(new_mnemonic) => {
                ed.replace_with_new_insn(host, ic.block_id, ic.insn_id, new_mnemonic, ic.size);
                Claim::Done
            }
            None => Claim::Pass,
        }
    }
}

/// If `v` is defined by an `IntBinop::want` binop, return its `(lhs, rhs)`.
fn as_int_binop(host: HostRef, v: ValueId, want: IntBinop) -> Option<(ValueId, ValueId)> {
    let ValueId::Instruction(id) = v else {
        return None;
    };
    match host.instruction(id).mnemonic() {
        Mnemonic::Binop(Binary {
            lhs,
            rhs,
            op: Binop::Int(op),
        }) if *op == want => Some((*lhs, *rhs)),
        _ => None,
    }
}

/// If `v` is defined by an `sborrow`, return its `(lhs, rhs)`.
fn as_sborrow(host: HostRef, v: ValueId) -> Option<(ValueId, ValueId)> {
    let ValueId::Instruction(id) = v else {
        return None;
    };
    match host.instruction(id).mnemonic() {
        Mnemonic::SBorrow(sb) => Some((sb.lhs, sb.rhs)),
        _ => None,
    }
}

/// Recognize the x86 signed-less-than flag idiom and collapse it to `a s< b`.
///
/// A signed `cmp a, b` lowers to `OF != SF` of `a - b`:
/// ```text
///   %of  = sborrow(a, b)
///   %sub = a - b
///   %sf  = %sub s< 0
///   %lt  = %of != %sf      // == (a s< b)
/// ```
/// Returns the rewritten `SLess(a, b)` mnemonic. The original `sborrow`/`sub`/
/// `slt` instructions are left for DCE to remove once their last use is gone.
pub(super) fn simplify_flag_idiom(host: HostRef, m: &Mnemonic) -> Option<Mnemonic> {
    let &Mnemonic::Binop(Binary {
        lhs,
        rhs,
        op: Binop::Int(IntBinop::NotEqual),
    }) = m
    else {
        return None;
    };

    // The `!=` is commutative (and GVN may have normalized it), so try both
    // assignments of which side is the sborrow and which is the `s< 0`.
    let resolve = |sborrow_side: ValueId, slt_side: ValueId| -> Option<(ValueId, ValueId)> {
        let (a, b) = as_sborrow(host, sborrow_side)?;
        let (sub_v, zero) = as_int_binop(host, slt_side, IntBinop::SLess)?;
        if const_value(host.shared(), zero) != Some(0) {
            return None;
        }
        let (sa, sb) = as_int_binop(host, sub_v, IntBinop::Sub)?;
        (sa == a && sb == b).then_some((a, b))
    };

    let (a, b) = resolve(lhs, rhs).or_else(|| resolve(rhs, lhs))?;
    Some(Mnemonic::Binop(Binary {
        lhs: a,
        rhs: b,
        op: Binop::Int(IntBinop::SLess),
    }))
}

#[cfg(test)]
mod tests {
    use crate::AliasResult;
    use crate::gvn::gvn_function;
    use qcode::{context::Context, value::BasicBlock};
    use qcode_macro::qcode;

    /// `sborrow(a, b) != ((a - b) s< 0)` is the x86 signed-less-than idiom and
    /// must collapse to a single `a s< b`, leaving the flag math dead.
    #[test]
    fn test_flag_idiom_collapses_to_signed_less_than() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 A;
                varnode i32 B;

                fn cmp:
                    <entry>
                        %a = load(A:4, &A);
                        %b = load(B:4, &B);
                        %of = sborrow(%a, %b);
                        %sub = %a - %b;
                        %sf = %sub s< 0x0;
                        %lt = %of != %sf;
                        if %lt goto <t> else goto <e>;
                    <t>
                        return at 0x1000;
                    <e>
                        return at 0x2000;
                "
        );

        let aliases = AliasResult::simple(&ctx);
        gvn_function(&mut ctx, cmp, Some(&aliases));

        assert!(
            !BasicBlock::from_id(&ctx, entry)
                .instruction_ids()
                .contains(&lt),
            "the `!=` flag combination should be rewritten away"
        );

        // After DCE the dead sborrow/sub/slt chain disappears, leaving the
        // single signed comparison the idiom was recognized as.
        crate::remove_dead_insns(&mut ctx, entry);
        let text = BasicBlock::from_id(&ctx, entry).to_string();
        assert!(
            text.contains("s<"),
            "expected a single signed-less-than, got:\n{text}"
        );
        assert!(
            !text.contains("sborrow") && !text.contains("!="),
            "the flag math should be dead after the rewrite, got:\n{text}"
        );
    }
}
