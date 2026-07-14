//! Bitwise/arithmetic identity sub-pass: collapse multi-instruction DAGs that
//! compilers emit for `^`/`+` into the single operation they compute.
//!
//! Unlike [`super::fold::algebraic_identity`], which applies a law to a single
//! `Binop`'s own operands, these patterns span several instructions. They rely
//! on GVN having value-numbered the shared sub-expressions, so the same `(a, b)`
//! feeding two different operations are recognized as identical operands.
//!
//! All identities below are exact in `n`-bit modular arithmetic (every operation
//! is the same width), from the full-adder relation
//! `a + b == (a ^ b) + ((a & b) << 1)` and the inclusion/exclusion relation
//! `a + b == (a | b) + (a & b)`:
//!
//! ```text
//!   (a + b) - ((a & b) << 1)  →  a ^ b
//!   (a | b) - (a & b)         →  a ^ b
//!   (a ^ b) + ((a & b) << 1)  →  a + b
//!   (a | b) + (a & b)         →  a + b
//! ```

use qcode::value::{
    FunctionId, QCodeView, Value, ValueId, ValueRef,
    insn::{Binary, Binop, IntBinop, Mnemonic, Simplified},
};

use super::fold::{all_ones, const_value};
use std::any::Any;

use super::walk::{Claim, Editor, InsnCtx, SubPassC};

use crate::{ContextView, FunctionBody};

/// Recognize the add/and/shift (and or/and) idioms and rewrite the root
/// instruction to the single `^`/`+` it computes. The now-unused sub-expressions
/// are left for DCE. Every read goes through the selected function's static view
/// and every folded constant is minted through the shared interner.
pub(super) struct Identities;

/// The function-pass [`SubPassC`] impl (context-split stage 5b-ii):
/// reads route through `cx.body_view(body)`, the intrinsic/identity rewrites
/// through `Editor`'s `_c` methods, and the constant-interning
/// `simplify_bitwise_c`/`simplify_compare_c` helpers run over `&mut PassBacking`.
impl<'str> SubPassC<'str> for Identities {
    fn init_state(&self) -> Box<dyn Any> {
        Box::new(())
    }

    fn clone_state(&self, _state: &dyn Any) -> Box<dyn Any> {
        Box::new(())
    }

    fn on_insn(
        &self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        _state: &mut dyn Any,
        ic: &InsnCtx,
        ed: &mut Editor,
    ) -> Claim {
        if ic.mnemonic.is_terminator() || ic.size == 0 {
            return Claim::Pass;
        }
        if let Mnemonic::Intrinsic(intr) = ic.mnemonic {
            let id = intr.id;
            let args: Vec<ValueId> = intr
                .args
                .iter()
                .map(|a| a.qualify(ic.insn_id.func))
                .collect();
            match id.desc().simplify(cx.body_view(body), id, ic.size, &args) {
                Some(Simplified::Value(repl)) => {
                    ed.replace_c(body, cx, ic.insn_id, repl);
                    return Claim::Done;
                }
                Some(Simplified::Expression(mnemonic)) => {
                    ed.replace_with_new_insn_c(
                        body,
                        cx,
                        ic.block_id,
                        ic.insn_id,
                        mnemonic,
                        ic.size,
                    );
                    return Claim::Done;
                }
                None => {}
            }
        }
        if let Some(new_mnemonic) =
            simplify_identity(cx.body_view(body), ic.insn_id.func, ic.mnemonic)
        {
            ed.replace_with_new_insn_c(body, cx, ic.block_id, ic.insn_id, new_mnemonic, ic.size);
            return Claim::Done;
        }
        if simplify_bitwise_c(body, cx, ic, ed) {
            return Claim::Done;
        }
        if simplify_compare_c(body, cx, ic, ed) {
            return Claim::Done;
        }
        Claim::Pass
    }
}

/// If `v` is defined by an `IntBinop::want` binop, return its `(lhs, rhs)`.
fn as_int_binop<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    v: ValueId,
    want: IntBinop,
) -> Option<(ValueId, ValueId)> {
    let ValueId::Instruction(id) = v else {
        return None;
    };
    match host.insn_ref(id).mnemonic() {
        Mnemonic::Binop(Binary {
            lhs,
            rhs,
            op: Binop::Int(op),
        }) if *op == want => Some((lhs.qualify(id.func), rhs.qualify(id.func))),
        _ => None,
    }
}

/// Whether `(a, b)` and `(c, d)` are the same unordered operand pair. The feeding
/// operations (`+`/`^`/`|` and `&`) are all commutative, and GVN may have
/// normalized their operand order independently, so match either assignment.
fn same_operands(p: (ValueId, ValueId), q: (ValueId, ValueId)) -> bool {
    (p.0 == q.0 && p.1 == q.1) || (p.0 == q.1 && p.1 == q.0)
}

/// If `v` computes `x * 2` — as `x << 1`, `x * 2`, or `2 * x` — return `x`.
fn as_doubled<'ctx, 'str: 'ctx>(host: impl QCodeView<'ctx, 'str>, v: ValueId) -> Option<ValueId> {
    if let Some((x, amt)) = as_int_binop(host, v, IntBinop::ShiftLeft)
        && const_value(host.shared(), amt) == Some(1)
    {
        return Some(x);
    }
    if let Some((x, y)) = as_int_binop(host, v, IntBinop::Mul) {
        if const_value(host.shared(), y) == Some(2) {
            return Some(x);
        }
        if const_value(host.shared(), x) == Some(2) {
            return Some(y);
        }
    }
    None
}

/// If `v` is `(a & b) * 2`, return `(a, b)`.
fn as_doubled_and<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    v: ValueId,
) -> Option<(ValueId, ValueId)> {
    as_int_binop(host, as_doubled(host, v)?, IntBinop::And)
}

fn int_binop(lhs: ValueId, rhs: ValueId, op: IntBinop) -> Mnemonic {
    // Both operands live in the body the mnemonic is inserted into; store them
    // bare-local by stripping their own embedded func.
    Mnemonic::Binop(Binary {
        lhs: lhs.strip_func(),
        rhs: rhs.strip_func(),
        op: Binop::Int(op),
    })
}

/// Collapse a bitwise/arithmetic identity rooted at `m`, returning the
/// equivalent single-operation mnemonic, or `None` if no pattern matches.
pub(super) fn simplify_identity<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    func: FunctionId,
    m: &Mnemonic,
) -> Option<Mnemonic> {
    let &Mnemonic::Binop(Binary {
        lhs,
        rhs,
        op: Binop::Int(op),
    }) = m
    else {
        return None;
    };
    let (lhs, rhs) = (lhs.qualify(func), rhs.qualify(func));

    match op {
        // (a + b) - ((a & b) << 1)  →  a ^ b
        // (a | b) - (a & b)         →  a ^ b
        // `-` is not commutative: the reducible term is always the rhs.
        IntBinop::Sub => {
            if let Some((a, b)) = as_int_binop(host, lhs, IntBinop::Add)
                && let Some(cd) = as_doubled_and(host, rhs)
                && same_operands((a, b), cd)
            {
                return Some(int_binop(a, b, IntBinop::Xor));
            }
            if let Some((a, b)) = as_int_binop(host, lhs, IntBinop::Or)
                && let Some(cd) = as_int_binop(host, rhs, IntBinop::And)
                && same_operands((a, b), cd)
            {
                return Some(int_binop(a, b, IntBinop::Xor));
            }
            None
        }
        // (a ^ b) + ((a & b) << 1)  →  a + b
        // (a | b) + (a & b)         →  a + b
        // `+` is commutative, so the base and extra term may appear on either side.
        IntBinop::Add => {
            for (base, extra) in [(lhs, rhs), (rhs, lhs)] {
                if let Some((a, b)) = as_int_binop(host, base, IntBinop::Xor)
                    && let Some(cd) = as_doubled_and(host, extra)
                    && same_operands((a, b), cd)
                {
                    return Some(int_binop(a, b, IntBinop::Add));
                }
                if let Some((a, b)) = as_int_binop(host, base, IntBinop::Or)
                    && let Some(cd) = as_int_binop(host, extra, IntBinop::And)
                    && same_operands((a, b), cd)
                {
                    return Some(int_binop(a, b, IntBinop::Add));
                }
            }
            None
        }
        _ => None,
    }
}

/// How many low bits a round-down alignment mask clears, i.e. `k` for a mask of
/// the form `~(2^k − 1)` over `size` bytes (`mask & ~0 == all` with a contiguous
/// run of cleared low bits). `None` if `mask` is not such a mask.
fn align_mask_bits(mask: u64, size: usize) -> Option<u32> {
    let low = !mask & all_ones(size); // the bits the mask clears
    // `low == 2^k − 1` ⇔ `low + 1` is a power of two ⇔ the cleared bits are a
    // contiguous low run, so the mask only rounds *down* to a 2^k boundary.
    (low.wrapping_add(1) & low == 0).then(|| low.count_ones())
}

/// A sound lower bound on the number of low-order bits known to be zero in `v`
/// (its base-2 alignment exponent). Recurses through the arithmetic that inlined,
/// stack-realigning prologues build on top of one `sp & ~7`; `depth` bounds the
/// walk (straight-line SSA has no operand cycles, but loops over block params
/// could, so the bound is load-bearing). Every case is a sound *under*-estimate.
fn known_align<'ctx, 'str: 'ctx>(host: impl QCodeView<'ctx, 'str>, v: ValueId, depth: u32) -> u32 {
    if depth == 0 {
        return 0;
    }
    if let Some(c) = const_value(host.shared(), v) {
        // A constant's alignment is its trailing-zero count; `0` is aligned to
        // its full width.
        return if c == 0 {
            (size_bits(host, v)).min(64)
        } else {
            c.trailing_zeros()
        };
    }
    let ValueId::Instruction(id) = v else {
        return 0;
    };
    let &Mnemonic::Binop(Binary {
        lhs,
        rhs,
        op: Binop::Int(op),
    }) = host.insn_ref(id).mnemonic()
    else {
        return 0;
    };
    let (lhs, rhs) = (lhs.qualify(id.func), rhs.qualify(id.func));
    let (a, b) = (
        || known_align(host, lhs, depth - 1),
        || known_align(host, rhs, depth - 1),
    );
    match op {
        // `&` keeps a low bit zero if it is zero in *either* operand — so a mask
        // is what *establishes* alignment: `x & ~7` is 8-aligned regardless of x.
        IntBinop::And => a().max(b()),
        // `|`/`^`/`+`/`-` keep a low bit zero only if it is zero in *both*.
        IntBinop::Or | IntBinop::Xor | IntBinop::Add | IntBinop::Sub => a().min(b()),
        // A left shift by a constant appends that many trailing zeros.
        IntBinop::ShiftLeft => match const_value(host.shared(), rhs) {
            Some(s) => a().saturating_add(s as u32),
            None => 0,
        },
        // Alignments add under multiplication.
        IntBinop::Mul => a().saturating_add(b()),
        _ => 0,
    }
}

/// The bit width of `v`'s output.
fn size_bits<'ctx, 'str: 'ctx>(host: impl QCodeView<'ctx, 'str>, v: ValueId) -> u32 {
    (value_size(host, v) as u32).saturating_mul(8)
}

/// Bound on the recursion `known_align` does back through a realignment cascade.
/// Inlined prologues stack only a handful of `& ~7` on top of one another, so a
/// generous fixed depth covers them without risking pathological walks.
const ALIGN_DEPTH: u32 = 64;

/// The `(constant, other_operand)` pairs of a binop — one entry per side that is
/// a numeric constant (both-constant binops are handled by [`super::fold`] before
/// this sub-pass, so in practice at most one side is constant here).
fn const_operands<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    lhs: ValueId,
    rhs: ValueId,
) -> Vec<(u64, ValueId)> {
    let mut out = Vec::new();
    if let Some(c) = const_value(host.shared(), rhs) {
        out.push((c, lhs));
    }
    if let Some(c) = const_value(host.shared(), lhs) {
        out.push((c, rhs));
    }
    out
}

/// If `v` is `x OP c` (or `c OP x`) for integer op `want` with a numeric constant
/// `c`, return `(x, c)`.
fn binop_const<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    v: ValueId,
    want: IntBinop,
) -> Option<(ValueId, u64)> {
    let (lhs, rhs) = as_int_binop(host, v, want)?;
    const_operands(host, lhs, rhs)
        .into_iter()
        .next()
        .map(|(c, x)| (x, c))
}

/// `~v` over `size` bytes expressed *without* emitting a fresh xor: a folded
/// constant when `v` is constant, or `x` when `v` is `x ^ ~0` (double-negation).
/// `None` when representing `~v` would need a new instruction — the caller then
/// declines the rewrite, keeping it strictly size-reducing.
fn simplify_not<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    v: ValueId,
    size: usize,
) -> Option<ValueId> {
    let all = all_ones(size);
    if let Some(c) = const_value(host.shared(), v) {
        return Some(host.shared().get_const((!c) & all, size));
    }
    if let Some((x, c)) = binop_const(host, v, IntBinop::Xor)
        && (c & all) == all
    {
        return Some(x);
    }
    None
}

/// The byte width of `v`'s output.
fn value_size<'ctx, 'str: 'ctx>(host: impl QCodeView<'ctx, 'str>, v: ValueId) -> usize {
    ValueRef::from_view(host, v).size()
}

/// If `v` is `zext(src)`, returns `(src, src_width)`.
fn as_zext<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    v: ValueId,
) -> Option<(ValueId, usize)> {
    let ValueId::Instruction(id) = v else {
        return None;
    };
    match host.insn_ref(id).mnemonic() {
        Mnemonic::Zext(z) => {
            let src = z.src.qualify(id.func);
            Some((src, value_size(host, src)))
        }
        _ => None,
    }
}

/// Whether `v` always yields `0` or `1` — now simply whether it is `bool`-typed
/// (comparisons and logical `And`/`Or`/`Xor` over bool all carry the type).
fn is_boolean<'ctx, 'str: 'ctx>(host: impl QCodeView<'ctx, 'str>, v: ValueId) -> bool {
    host.stored_type_of(v)
        .is_some_and(|t| host.shared().types.is_bool(t))
}

/// The negation of an equality comparison: `==`↔`!=`. Ordering comparisons are
/// not flipped here (their negation swaps operand order and strictness).
fn negated_compare(op: IntBinop) -> Option<IntBinop> {
    match op {
        IntBinop::Equal => Some(IntBinop::NotEqual),
        IntBinop::NotEqual => Some(IntBinop::Equal),
        _ => None,
    }
}

/// Concrete pass twin of [`simplify_bitwise`] over a checked-out
/// `(&mut FunctionBody, ContextView)` (context-split stage 5b-ii Pin A): reads
/// route through `cx.body_view(body)`, rewrites through `Editor`'s `_c` methods.
fn simplify_bitwise_c<'str>(
    body: &mut FunctionBody<'str>,
    cx: ContextView<'_, 'str>,
    ic: &InsnCtx,
    ed: &mut Editor,
) -> bool {
    let &Mnemonic::Binop(Binary {
        lhs,
        rhs,
        op: Binop::Int(op),
    }) = ic.mnemonic
    else {
        return false;
    };
    let (lhs, rhs) = (lhs.qualify(ic.insn_id.func), rhs.qualify(ic.insn_id.func));
    let size = ic.size;
    let all = all_ones(size);

    match op {
        IntBinop::And => {
            for (outer, inner) in const_operands(cx.body_view(body), lhs, rhs) {
                if outer == 1
                    && (is_boolean(cx.body_view(body), inner)
                        || as_zext(cx.body_view(body), inner)
                            .is_some_and(|(src, _)| is_boolean(cx.body_view(body), src)))
                {
                    ed.replace_c(body, cx, ic.insn_id, inner);
                    return true;
                }
                if let Some(k) = align_mask_bits(outer, size)
                    && known_align(cx.body_view(body), inner, ALIGN_DEPTH) >= k
                {
                    ed.replace_c(body, cx, ic.insn_id, inner);
                    return true;
                }
                if let Some((x, c1)) = binop_const(cx.body_view(body), inner, IntBinop::And) {
                    let folded = cx
                        .body_view(body)
                        .shared()
                        .get_const((c1 & outer) & all, size);
                    ed.replace_with_new_insn_c(
                        body,
                        cx,
                        ic.block_id,
                        ic.insn_id,
                        int_binop(x, folded, IntBinop::And),
                        size,
                    );
                    return true;
                }
            }
            false
        }
        IntBinop::Xor => {
            for (outer, inner) in const_operands(cx.body_view(body), lhs, rhs) {
                if let Some((x, c1)) = binop_const(cx.body_view(body), inner, IntBinop::Xor) {
                    let folded = cx
                        .body_view(body)
                        .shared()
                        .get_const((c1 ^ outer) & all, size);
                    ed.replace_with_new_insn_c(
                        body,
                        cx,
                        ic.block_id,
                        ic.insn_id,
                        int_binop(x, folded, IntBinop::Xor),
                        size,
                    );
                    return true;
                }
                if outer != all {
                    continue;
                }
                for (dual_in, dual_out) in
                    [(IntBinop::Or, IntBinop::And), (IntBinop::And, IntBinop::Or)]
                {
                    let Some((a, b)) = as_int_binop(cx.body_view(body), inner, dual_in) else {
                        continue;
                    };
                    let (Some(na), Some(nb)) = (
                        simplify_not(cx.body_view(body), a, size),
                        simplify_not(cx.body_view(body), b, size),
                    ) else {
                        continue;
                    };
                    ed.replace_with_new_insn_c(
                        body,
                        cx,
                        ic.block_id,
                        ic.insn_id,
                        int_binop(na, nb, dual_out),
                        size,
                    );
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

/// Concrete pass twin of [`simplify_compare`] (see [`simplify_bitwise_c`]).
fn simplify_compare_c<'str>(
    body: &mut FunctionBody<'str>,
    cx: ContextView<'_, 'str>,
    ic: &InsnCtx,
    ed: &mut Editor,
) -> bool {
    match *ic.mnemonic {
        Mnemonic::Binop(Binary {
            lhs,
            rhs,
            op: Binop::Int(op),
        }) if matches!(op, IntBinop::Equal | IntBinop::NotEqual) => {
            let (lhs, rhs) = (lhs.qualify(ic.insn_id.func), rhs.qualify(ic.insn_id.func));
            for (c, other) in const_operands(cx.body_view(body), lhs, rhs) {
                if c != 0 {
                    continue;
                }
                if let Some((src, src_size)) = as_zext(cx.body_view(body), other) {
                    let zero = cx.body_view(body).shared().get_const(0, src_size);
                    let bool_ty = cx.body_view(body).shared().types.get_or_make_bool();
                    ed.replace_with_new_insn_typed_c(
                        body,
                        cx,
                        ic.block_id,
                        ic.insn_id,
                        int_binop(src, zero, op),
                        bool_ty,
                    );
                    return true;
                }
                if is_boolean(cx.body_view(body), other)
                    && value_size(cx.body_view(body), other) == ic.size
                {
                    match op {
                        IntBinop::NotEqual => {
                            ed.replace_c(body, cx, ic.insn_id, other);
                            return true;
                        }
                        IntBinop::Equal => {
                            if let ValueId::Instruction(id) = other
                                && let &Mnemonic::Binop(Binary {
                                    lhs: a,
                                    rhs: b,
                                    op: Binop::Int(inner),
                                }) = cx.body_view(body).insn_ref(id).mnemonic()
                                && let Some(flipped) = negated_compare(inner)
                            {
                                let (a, b) = (a.qualify(id.func), b.qualify(id.func));
                                let bool_ty = cx.body_view(body).shared().types.get_or_make_bool();
                                ed.replace_with_new_insn_typed_c(
                                    body,
                                    cx,
                                    ic.block_id,
                                    ic.insn_id,
                                    int_binop(a, b, flipped),
                                    bool_ty,
                                );
                                return true;
                            }
                        }
                        _ => {}
                    }
                }
            }
            false
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use crate::AliasResult;
    use crate::gvn::gvn_function;
    use qcode::{context::Context, value::BasicBlock};
    use qcode_macro::qcode;

    /// `(a + b) - ((a & b) << 1)` is the add/and/shift form of `a ^ b` and must
    /// collapse to a single xor, leaving the add/and/shift math dead.
    #[test]
    fn add_minus_doubled_and_collapses_to_xor() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 A;
                varnode i32 B;

                fn f:
                    <entry>
                        %a = load(A:4, &A);
                        %b = load(B:4, &B);
                        %sum = %a + %b;
                        %and = %a & %b;
                        %dbl = %and << 0x1;
                        %root = %sum - %dbl;
                        return at %root;
                "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        gvn_function(&mut ctx, f, Some(&aliases));

        assert!(
            !BasicBlock::from_id(&ctx, entry)
                .instruction_ids()
                .contains(&root),
            "the `sub` root should be rewritten to an xor"
        );

        crate::remove_dead_insns(&mut ctx, entry);
        let text = BasicBlock::from_id(&ctx, entry).to_string();
        assert!(text.contains('^'), "expected a single xor, got:\n{text}");
        assert!(
            !text.contains("<<") && !text.contains(" + ") && !text.contains(" - "),
            "the add/and/shift math should be dead after the rewrite, got:\n{text}"
        );
    }

    /// `(a | b) - (a & b)` is another form of `a ^ b`.
    #[test]
    fn or_minus_and_collapses_to_xor() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 A;
                varnode i32 B;

                fn f:
                    <entry>
                        %a = load(A:4, &A);
                        %b = load(B:4, &B);
                        %or = %a | %b;
                        %and = %a & %b;
                        %root = %or - %and;
                        return at %root;
                "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        gvn_function(&mut ctx, f, Some(&aliases));
        crate::remove_dead_insns(&mut ctx, entry);
        let text = BasicBlock::from_id(&ctx, entry).to_string();
        assert!(text.contains('^'), "expected a single xor, got:\n{text}");
        assert!(
            !text.contains(" | ") && !text.contains(" - "),
            "the or/and math should be dead, got:\n{text}"
        );
    }

    /// `(a | b) + (a & b)` is a form of `a + b`.
    #[test]
    fn or_plus_and_collapses_to_add() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 A;
                varnode i32 B;

                fn f:
                    <entry>
                        %a = load(A:4, &A);
                        %b = load(B:4, &B);
                        %or = %a | %b;
                        %and = %a & %b;
                        %root = %or + %and;
                        return at %root;
                "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        gvn_function(&mut ctx, f, Some(&aliases));
        crate::remove_dead_insns(&mut ctx, entry);
        let text = BasicBlock::from_id(&ctx, entry).to_string();
        assert!(text.contains(" + "), "expected a single add, got:\n{text}");
        assert!(
            !text.contains(" | ") && !text.contains(" & "),
            "the or/and math should be dead, got:\n{text}"
        );
    }

    /// The obfuscated `i + 1` a real loop emits — `((~i | ~1) ^ ~0) & ~2` is
    /// `i & 1` behind De Morgan plus a redundant mask, doubled and added to
    /// `i ^ 1` (the full-adder form of `i + 1`). The bitwise canonicalization
    /// must peel the obfuscation so the existing full-adder idiom collapses the
    /// whole thing to a single `i + 1`.
    #[test]
    fn obfuscated_increment_collapses_to_add_one() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 I;

                fn f:
                    <entry>
                        %i = load(I:4, &I);
                        %n1 = %i ^ 0xffffffff;
                        %n2 = %n1 | 0xfffffffe;
                        %n3 = %n2 ^ 0xffffffff;
                        %n4 = %n3 & 0xfffffffd;
                        %dbl = %n4 * 0x2;
                        %x = %i ^ 0x1;
                        %inc = %dbl + %x;
                        store(I:4, &I <- %inc);
                        return at 0x1000;
                "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        gvn_function(&mut ctx, f, Some(&aliases));
        crate::remove_dead_insns(&mut ctx, entry);
        let text = BasicBlock::from_id(&ctx, entry).to_string();
        assert!(
            text.contains(" + i32 0x1") || text.contains(" + i32 0x00000001"),
            "the obfuscated increment should collapse to `i + 1`, got:\n{text}"
        );
        assert!(
            !text.contains(" | ") && !text.contains(" * ") && !text.contains('~'),
            "the De Morgan / doubling obfuscation should be gone, got:\n{text}"
        );
    }

    /// `zext(!(x == 0)) != 0` is the flag-materialization shape a compiler emits
    /// for `if (x != 0)`. GVN must collapse the whole condition to a single
    /// `x != 0`, leaving the zext / not / redundant `!= 0` dead.
    #[test]
    fn zext_bool_compare_collapses_to_single_compare() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 X;

                fn f:
                    <entry>
                        %x = load(X:4, &X);
                        %eq = %x == 0x0;
                        %ne = %eq == false;
                        %z = zext(i32, %ne);
                        %cond = %z != 0x0;
                        if %cond goto <0x2000> else goto <0x1000>;
                    <0x1000>
                        return at 0x0;
                    <0x2000>
                        return at 0x1;
                "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        while gvn_function(&mut ctx, f, Some(&aliases)) {}
        crate::remove_dead_insns(&mut ctx, entry);
        let text = BasicBlock::from_id(&ctx, entry).to_string();

        assert!(
            !text.contains("zext"),
            "the zext should be dead, got:\n{text}"
        );
        assert!(
            !text.contains("== 0x0"),
            "the `== 0` flag test should be gone, got:\n{text}"
        );
        assert_eq!(
            text.matches("!=").count(),
            1,
            "the condition should collapse to a single `x != 0`, got:\n{text}"
        );
    }

    /// `zext(v) != 0` strips the zext even when `v` is not itself boolean.
    #[test]
    fn zext_compare_with_zero_strips_zext() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i8 V;
                varnode i8 B;

                fn f:
                    <entry>
                        %v = load(V:1, &V);
                        %z = zext(i32, %v);
                        %cond = %z != 0x0;
                        %c8 = zext(i8, %cond);
                        store(B:1, &B <- %c8);
                        return at 0x0;
                "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        while gvn_function(&mut ctx, f, Some(&aliases)) {}
        crate::remove_dead_insns(&mut ctx, entry);
        let text = BasicBlock::from_id(&ctx, entry).to_string();
        assert!(
            !text.contains("zext(i32"),
            "the i32 zext feeding the compare should be gone, got:\n{text}"
        );
        assert!(
            text.contains("!="),
            "the `!= 0` comparison on the i8 source must remain, got:\n{text}"
        );
    }

    /// `zext(i < 4) & 1 != 0` — the lifted shape of a `(i < 4) & 1` header guard
    /// after the comparison flag is spilled to a byte and reloaded — must collapse
    /// to the bare comparison `i < 4`. This is the `bool`-migration's `loop_to_map`
    /// unblock: `& 1` drops on the `{0,1}`-valued zext, then `zext(b) != 0 → b`.
    #[test]
    fn bool_and_one_ne_zero_collapses_to_comparison() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 I;
                varnode i8 B;

                fn f:
                    <entry>
                        %i = load(I:4, &I);
                        %lt = %i < 0x4;
                        %z = zext(i8, %lt);
                        %m = %z & 0x1;
                        %nz = %m != 0x0;
                        %c8 = zext(i8, %nz);
                        store(B:1, &B <- %c8);
                        return at 0x0;
                "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        while gvn_function(&mut ctx, f, Some(&aliases)) {}
        crate::remove_dead_insns(&mut ctx, entry);
        let text = BasicBlock::from_id(&ctx, entry).to_string();
        assert!(
            !text.contains(" & "),
            "the `& 1` mask should be gone, got:\n{text}"
        );
        assert!(
            text.contains(" < i32 0x4"),
            "the bare comparison must survive, got:\n{text}"
        );
    }

    /// The realignment cascade an inlined, stack-aligning prologue emits: one
    /// real `& ~7` to align `sp`, then a chain of `(aligned − 8·m) & ~7` whose
    /// masks are all no-ops. Every mask after the first must drop, leaving the
    /// frame as flat offsets off the single aligned base.
    #[test]
    fn realignment_cascade_drops_redundant_masks() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 SP;

                fn f:
                    <entry>
                        %sp = load(SP:4, &SP);
                        %s0 = %sp - 0x10;
                        %base = %s0 & 0xfffffff8;
                        %a0 = %base - 0x270;
                        %a1 = %a0 & 0xfffffff8;
                        %b0 = %a1 - 0x90;
                        %b1 = %b0 & 0xfffffff8;
                        %c0 = %b1 - 0x88;
                        %c1 = %c0 & 0xfffffff8;
                        return at %c1;
                "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        while gvn_function(&mut ctx, f, Some(&aliases)) {}
        crate::remove_dead_insns(&mut ctx, entry);
        let text = BasicBlock::from_id(&ctx, entry).to_string();
        assert_eq!(
            text.matches("0xfffffff8").count(),
            1,
            "only the base-establishing mask should survive, got:\n{text}"
        );
    }

    /// The mask must stay when alignment is *not* provable: `0xa4` is not an
    /// 8-multiple, so `(aligned − 0xa4) & ~7` genuinely rounds down and is not a
    /// no-op.
    #[test]
    fn non_aligned_offset_keeps_mask() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 SP;

                fn f:
                    <entry>
                        %sp = load(SP:4, &SP);
                        %base = %sp & 0xfffffff8;
                        %a0 = %base - 0xa4;
                        %a1 = %a0 & 0xfffffff8;
                        return at %a1;
                "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        while gvn_function(&mut ctx, f, Some(&aliases)) {}
        let text = BasicBlock::from_id(&ctx, entry).to_string();
        assert_eq!(
            text.matches("0xfffffff8").count(),
            2,
            "the non-8-multiple realignment mask must be preserved, got:\n{text}"
        );
    }

    /// An unrelated `sub` whose operands don't form the idiom must be left alone.
    #[test]
    fn unrelated_sub_is_untouched() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                varnode i32 A;
                varnode i32 B;

                fn f:
                    <entry>
                        %a = load(A:4, &A);
                        %b = load(B:4, &B);
                        %sum = %a + %b;
                        %and = %a & %b;
                        %root = %sum - %and;
                        return at %root;
                "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        gvn_function(&mut ctx, f, Some(&aliases));
        let text = BasicBlock::from_id(&ctx, entry).to_string();
        assert!(
            text.contains(" - "),
            "a non-idiom subtraction must be preserved, got:\n{text}"
        );
    }
}
