//! Width-narrowing sub-pass: sink low-word truncations toward the leaves.
//!
//! A truncation to the low `W` bytes — the `Range` mnemonic `[0:W]` — commutes
//! with every operation whose low `W` bytes depend only on its operands' low `W`
//! bytes: `+ − × & | ^ ~ neg`. Pushing it down those ops, **cancelling** it
//! against widenings (`sext`/`zext`) and folding it into constants, collapses
//! the lift-time "widen, compute wide, truncate back" idiom to a single narrow
//! computation:
//!
//! ```text
//! %xa = sext(i64, %a)
//! %xb = sext(i64, %b)
//! %m  = %xa * %xb
//! %r  = %m[0:4]          ⇒  %r = %a *₃₂ %b      (low word of a product is width-agnostic)
//! ```
//!
//! This is the structural companion to [`mba_simplify`](crate::mba_simplify):
//! narrowing erases the `sext`/`mul`/`range` plumbing the MBA solver cannot
//! model, leaving a uniform-width MBA the solver *can* collapse.
//!
//! # Soundness
//!
//! Every rule rewrites the **low word only** — it materializes a *new* narrow
//! value and never touches the wide original, so any consumer of the wide
//! value's high bits is unaffected. The distributions are exact under
//! two's-complement wrapping. Truncation is *not* sunk through `>>`, `/`, `%`,
//! comparisons, loads, … (their low bytes depend on more than the operands' low
//! bytes); there it stops, leaving a `Range` of an opaque value. Dead wide
//! originals are left for DCE, per the GVN convention.

use std::any::Any;

use rustc_hash::FxHashMap as HashMap;

use qcode::value::{
    Value, ValueId, ValueRef,
    block::BlockId,
    insn::{Binary, Binop, InstructionId, IntBinop, Mnemonic, Range, Sext, Unary, Unop, Zext},
    util::base_ref::HostRef,
};

use super::walk::{Claim, Editor, InsnCtx, SubPassC};

use crate::{ContextView, FunctionBody};

/// Fully host-routed: reads resolve through a [`HostRef`], the narrow values it
/// materializes are pushed into the (possibly checked-out) function's own arena,
/// and constants are minted through the shared interners.
pub(super) struct NarrowTrunc;

/// The function-pass [`SubPassC`] impl (context-split stage 5b-ii):
/// eligibility reads through `body.read_host(cx)`, the recursive `narrow_to_c`
/// rewrite runs over `&mut PassBacking`, and the forward goes through
/// `Editor::replace_c`.
impl<'str> SubPassC<'str> for NarrowTrunc {
    fn init_state(&self) -> Box<dyn Any> {
        Box::new(())
    }

    fn clone_state(&self, _state: &dyn Any) -> Box<dyn Any> {
        Box::new(())
    }

    fn on_insn(
        &self,
        body: &mut FunctionBody<'_, 'str>,
        cx: ContextView<'_, 'str>,
        _state: &mut dyn Any,
        ic: &InsnCtx,
        ed: &mut Editor,
    ) -> Claim {
        let Mnemonic::Range(Range {
            src,
            start: 0,
            size,
        }) = ic.mnemonic
        else {
            return Claim::Pass;
        };
        let (src, w) = (*src, *size);
        if value_size(body.read_host(cx), src) != w && !src_transformable(body.read_host(cx), src) {
            return Claim::Pass;
        }
        let mut memo: HashMap<ValueId, ValueId> = HashMap::default();
        let narrowed = narrow_to_c(body, cx, src, w, ic.insn_id, ic.block_id, &mut memo);
        if narrowed == ic.id {
            return Claim::Pass;
        }
        ed.replace_c(body, cx, ic.insn_id, narrowed);
        Claim::Done
    }
}

/// Will [`narrow_to`] push through `v` rather than just wrap it in a `Range`?
fn src_transformable(host: HostRef, v: ValueId) -> bool {
    if numeric_const(host.shr(), v).is_some() {
        return true;
    }
    let ValueId::Instruction(iid) = v else {
        return false;
    };
    match host.insn_ref(iid).mnemonic() {
        Mnemonic::Binop(b) => matches!(
            b.op,
            Binop::Int(
                IntBinop::Add
                    | IntBinop::Sub
                    | IntBinop::Mul
                    | IntBinop::And
                    | IntBinop::Or
                    | IntBinop::Xor
            )
        ),
        Mnemonic::Unop(u) => matches!(u.op, Unop::IntNot | Unop::IntNegate),
        Mnemonic::Sext(_) | Mnemonic::Zext(_) => true,
        Mnemonic::Range(Range { start: 0, .. }) => true,
        _ => false,
    }
}

fn distributive(op: IntBinop) -> bool {
    matches!(
        op,
        IntBinop::Add
            | IntBinop::Sub
            | IntBinop::Mul
            | IntBinop::And
            | IntBinop::Or
            | IntBinop::Xor
    )
}

fn range_low(src: ValueId, size: usize) -> Mnemonic {
    Mnemonic::Range(Range {
        src,
        start: 0,
        size,
    })
}

// ---------------------------------------------------------------------------
// Concrete pass twins over (&mut FunctionBody, ContextView) — 5b-ii Pin A step 2.
// ---------------------------------------------------------------------------

/// Concrete pass twin of [`narrow_to`].
fn narrow_to_c<'str>(
    body: &mut FunctionBody<'_, 'str>,
    cx: ContextView<'_, 'str>,
    v: ValueId,
    w: usize,
    before: InstructionId,
    block: BlockId,
    memo: &mut HashMap<ValueId, ValueId>,
) -> ValueId {
    if value_size(body.read_host(cx), v) == w {
        return v;
    }
    if let Some(&cached) = memo.get(&v) {
        return cached;
    }

    let result = match v {
        ValueId::Instruction(iid) => match body.read_host(cx).insn_ref(iid).mnemonic().clone() {
            Mnemonic::Binop(Binary {
                op: Binop::Int(o),
                lhs,
                rhs,
            }) if distributive(o) => {
                let l = narrow_to_c(body, cx, lhs, w, before, block, memo);
                let rr = narrow_to_c(body, cx, rhs, w, before, block, memo);
                push_insn_c(
                    body,
                    cx,
                    Mnemonic::Binop(Binary {
                        op: Binop::Int(o),
                        lhs: l,
                        rhs: rr,
                    }),
                    w,
                    before,
                    block,
                )
            }
            Mnemonic::Unop(Unary { op, src }) if matches!(op, Unop::IntNot | Unop::IntNegate) => {
                let s = narrow_to_c(body, cx, src, w, before, block, memo);
                push_insn_c(
                    body,
                    cx,
                    Mnemonic::Unop(Unary { op, src: s }),
                    w,
                    before,
                    block,
                )
            }
            Mnemonic::Sext(Sext { src, .. }) => {
                narrow_extension_c(body, cx, src, w, true, before, block, memo)
            }
            Mnemonic::Zext(Zext { src, .. }) => {
                narrow_extension_c(body, cx, src, w, false, before, block, memo)
            }
            Mnemonic::Range(Range { src, start: 0, .. }) => {
                narrow_to_c(body, cx, src, w, before, block, memo)
            }
            _ => push_insn_c(body, cx, range_low(v, w), w, before, block),
        },
        _ if numeric_const(body.read_host(cx).shr(), v).is_some() => {
            let folded = numeric_const(body.read_host(cx).shr(), v).unwrap() & low_mask(w);
            body.read_host(cx).shr().get_const(folded, w)
        }
        _ => push_insn_c(body, cx, range_low(v, w), w, before, block),
    };

    memo.insert(v, result);
    result
}

/// Concrete pass twin of [`narrow_extension`].
#[allow(clippy::too_many_arguments)]
fn narrow_extension_c<'str>(
    body: &mut FunctionBody<'_, 'str>,
    cx: ContextView<'_, 'str>,
    src: ValueId,
    w: usize,
    sext: bool,
    before: InstructionId,
    block: BlockId,
    memo: &mut HashMap<ValueId, ValueId>,
) -> ValueId {
    if value_size(body.read_host(cx), src) >= w {
        return narrow_to_c(body, cx, src, w, before, block, memo);
    }
    let m = if sext {
        Mnemonic::Sext(Sext { src, size: w })
    } else {
        Mnemonic::Zext(Zext { src, size: w })
    };
    push_insn_c(body, cx, m, w, before, block)
}

/// Concrete pass twin of [`push_insn`].
fn push_insn_c<'str>(
    body: &mut FunctionBody<'_, 'str>,
    cx: ContextView<'_, 'str>,
    mnemonic: Mnemonic,
    size: usize,
    before: InstructionId,
    block: BlockId,
) -> ValueId {
    let id = body.push_mnemonic(cx, mnemonic, size);
    body.insert_insn_before(cx, block, before, id);
    ValueId::Instruction(id)
}

fn low_mask(w_bytes: usize) -> u64 {
    let bits = w_bytes * 8;
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

fn numeric_const(shared: &qcode::context::Shared, v: ValueId) -> Option<u64> {
    if let ValueId::Literal(id) = v {
        let lit = &shared.values.literals[id];
        if lit.symbolic.is_none() {
            return Some(lit.value);
        }
    }
    None
}

fn value_size(host: HostRef, v: ValueId) -> usize {
    ValueRef::from_host(host, v).size()
}

#[cfg(test)]
mod tests {
    use super::value_size;
    use crate::gvn::narrow_function;
    use crate::mba_simplify::mba_simplify;
    use qcode::value::util::base_ref::HostRef;
    use qcode::{
        context::Context,
        value::{BasicBlock, Function, FunctionId, Instruction, ValueId, insn::Mnemonic},
    };
    use qcode_emulator::{SizedValue, StandaloneEmulator};
    use qcode_macro::qcode;

    fn return_value(ctx: &Context, fun: FunctionId) -> ValueId {
        let root = Function::from_id(ctx, fun).root().expect("root").id;
        let &term = BasicBlock::from_id(ctx, root)
            .instruction_ids()
            .last()
            .expect("terminator");
        match Instruction::from_id(ctx, term).mnemonic() {
            Mnemonic::ReturnValue(r) => r.value,
            other => panic!("expected return, got {other:?}"),
        }
    }

    fn run(ctx: &Context, fun: FunctionId, a: u64, b: u64) -> Option<u64> {
        let root = Function::from_id(ctx, fun).root().expect("root").id;
        let ret = return_value(ctx, fun);
        let mut emu = StandaloneEmulator::new(root);
        emu.run_pure(
            ctx,
            fun,
            &[SizedValue::new(a, 4), SizedValue::new(b, 4)],
            100_000,
        )
        .expect("runs");
        emu.get_value(ctx, ret)
    }

    fn sample(ctx: &Context, fun: FunctionId) -> Vec<Option<u64>> {
        [
            (5, 3),
            (0, 0),
            (0xdead, 0xbeef),
            (1, 0xffff_ffff),
            (0x8000_0000, 0x8000_0001),
        ]
        .into_iter()
        .map(|(a, b)| run(ctx, fun, a, b))
        .collect()
    }

    /// The defining mnemonic of the (live) return value.
    fn return_def<'a>(ctx: &'a Context, fun: FunctionId) -> &'a Mnemonic {
        let ValueId::Instruction(iid) = return_value(ctx, fun) else {
            panic!("return is not an instruction");
        };
        Instruction::from_id(ctx, iid).mnemonic()
    }

    #[test]
    fn cancels_widening_around_multiply() {
        // (sext(a) * sext(b))[0:4]  ==  a *₃₂ b
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda widemul:
            <entry @a:i32 @b:i32>
                %xa = sext(i64, @a);
                %xb = sext(i64, @b);
                %m = %xa * %xb;
                %r = %m[0:4];
                return %r;
            "
        );
        let before = sample(&ctx, widemul);
        assert!(narrow_function(&mut ctx, widemul));
        // The live return is now a 4-byte multiply (widening sunk away).
        assert!(matches!(
            return_def(&ctx, widemul),
            Mnemonic::Binop(qcode::value::insn::Binary {
                op: qcode::value::insn::Binop::Int(qcode::value::insn::IntBinop::Mul),
                ..
            })
        ));
        assert_eq!(
            value_size(HostRef::from(&ctx), return_value(&ctx, widemul)),
            4
        );
        assert_eq!(sample(&ctx, widemul), before);
    }

    #[test]
    fn cancels_zext_through_bitwise() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda z:
            <entry @a:i32 @b:i32>
                %za = zext(i64, @a);
                %zb = zext(i64, @b);
                %o = %za | %zb;
                %r = %o[0:4];
                return %r;
            "
        );
        let before = sample(&ctx, z);
        assert!(narrow_function(&mut ctx, z));
        assert_eq!(value_size(HostRef::from(&ctx), return_value(&ctx, z)), 4);
        assert!(!matches!(return_def(&ctx, z), Mnemonic::Range(_)));
        assert_eq!(sample(&ctx, z), before);
    }

    #[test]
    fn folds_constant_truncation() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda c:
            <entry @a:i32 @b:i32>
                %xa = sext(i64, @a);
                %s = %xa + 0x100000007;
                %r = %s[0:4];
                return %r;
            "
        );
        let before = sample(&ctx, c);
        assert!(narrow_function(&mut ctx, c));
        assert_eq!(sample(&ctx, c), before);
        // low word of the constant is 7, so the result is a + 7 (mod 2^32).
        assert_eq!(run(&ctx, c, 10, 0), Some(17));
    }

    #[test]
    fn leaves_shift_right_truncation_alone() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda sr:
            <entry @a:i32 @b:i32>
                %xa = sext(i64, @a);
                %sh = %xa >> 0x8;
                %r = %sh[0:4];
                return %r;
            "
        );
        assert!(!narrow_function(&mut ctx, sr));
    }

    #[test]
    fn narrow_then_mba_collapses_widened_constant_multiply() {
        // The MT seeder's disguised K·A: an R2 MBA whose two products are formed
        // through 32→64 widening. `narrow` strips the widening so `mba_simplify`
        // sees a uniform-width MBA and collapses it to A·K.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda mtmul:
            <entry @a:i32 @b:i32>
                %ank = @a & 0x93f8769a;
                %na = ~ @a;
                %nak = %na & 0x6c078965;
                %s1 = sext(i64, %ank);
                %s2 = sext(i64, %nak);
                %pp1 = %s1 * %s2;
                %p1 = %pp1[0:4];
                %ak = @a & 0x6c078965;
                %aok = @a | 0x6c078965;
                %s3 = sext(i64, %aok);
                %s4 = sext(i64, %ak);
                %pp2 = %s3 * %s4;
                %p2 = %pp2[0:4];
                %r = %p1 + %p2;
                return %r;
            "
        );
        let before = sample(&ctx, mtmul);

        assert!(narrow_function(&mut ctx, mtmul));
        // DCE clears the now-dead sext/mul plumbing — as the real pipeline does
        // between GVN and mba_simplify — so their stale uses don't pin the
        // And/Or results and hide the boolean half of the MBA.
        let root = Function::from_id(&ctx, mtmul).root().expect("root").id;
        while crate::dce::remove_dead_insns(&mut ctx, root) {}
        // mba_simplify's surface is pass-scoped; run it over a `PassBacking`
        // borrowing the body in place alongside the read-only shared state.
        let mba_changed = {
            let mut host = qcode::value::util::host_mut::PassBacking::new(
                &mut ctx.bodies[mtmul],
                mtmul,
                &ctx.shared,
                &ctx.interfaces,
            );
            mba_simplify(&mut host, mtmul)
        };
        assert!(mba_changed);

        assert_eq!(sample(&ctx, mtmul), before);
        let k = 0x6c07_8965u64;
        assert_eq!(run(&ctx, mtmul, 7, 0), Some((7 * k) & 0xffff_ffff));
        // Live return collapsed to a single multiply.
        assert!(matches!(
            return_def(&ctx, mtmul),
            Mnemonic::Binop(qcode::value::insn::Binary {
                op: qcode::value::insn::Binop::Int(qcode::value::insn::IntBinop::Mul),
                ..
            })
        ));
    }
}
