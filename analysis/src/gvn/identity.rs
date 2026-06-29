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

use qcode::{
    context::Context,
    value::{
        Value, ValueId, ValueRef,
        insn::{Binary, Binop, IntBinop, Mnemonic, Simplified, Unary, Unop},
    },
};

use super::fold::{all_ones, const_value};
use super::walk::{Claim, Editor, InsnCtx, SubPass};

/// Recognize the add/and/shift (and or/and) idioms and rewrite the root
/// instruction to the single `^`/`+` it computes. The now-unused sub-expressions
/// are left for DCE.
pub(super) struct Identities;

impl SubPass for Identities {
    type State = ();

    fn on_insn(&self, ctx: &mut Context, _state: &mut (), ic: &InsnCtx, ed: &mut Editor) -> Claim {
        if ic.mnemonic.is_terminator() || ic.size == 0 {
            return Claim::Pass;
        }
        // Pure intrinsics carry their own algebraic simplifier (e.g.
        // `rol(x, 0) → x`), which forwards uses to an existing value.
        if let Mnemonic::Intrinsic(intr) = ic.mnemonic {
            let id = intr.id;
            let args = intr.args.clone();
            match id.desc().simplify(ctx, id, ic.size, &args) {
                Some(Simplified::Value(repl)) => {
                    ed.replace(ctx, ic.insn_id, repl);
                    return Claim::Done;
                }
                Some(Simplified::Expression(mnemonic)) => {
                    ed.replace_with_new_insn(ctx, ic.block_id, ic.insn_id, mnemonic, ic.size);
                    return Claim::Done;
                }
                None => {}
            }
        }
        if let Some(new_mnemonic) = simplify_identity(ctx, ic.mnemonic) {
            ed.replace_with_new_insn(ctx, ic.block_id, ic.insn_id, new_mnemonic, ic.size);
            return Claim::Done;
        }
        // Constant-absorbing / De Morgan rewrites need `&mut Context` (they intern
        // a folded constant), so they live outside the borrow-only
        // `simplify_identity`. They canonicalize obfuscated bit math — e.g. an
        // `i & 1` emitted as `~(~i | ~1)` with a redundant outer mask — back into
        // the plain `&`/`^` the full-adder idioms above then recognize.
        if simplify_bitwise(ctx, ic, ed) {
            return Claim::Done;
        }
        if simplify_compare(ctx, ic, ed) {
            return Claim::Done;
        }
        Claim::Pass
    }
}

/// If `v` is defined by an `IntBinop::want` binop, return its `(lhs, rhs)`.
fn as_int_binop(ctx: &Context, v: ValueId, want: IntBinop) -> Option<(ValueId, ValueId)> {
    let ValueId::Instruction(id) = v else {
        return None;
    };
    match ctx.get_insn(id).mnemonic() {
        Mnemonic::Binop(Binary {
            lhs,
            rhs,
            op: Binop::Int(op),
        }) if *op == want => Some((*lhs, *rhs)),
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
fn as_doubled(ctx: &Context, v: ValueId) -> Option<ValueId> {
    if let Some((x, amt)) = as_int_binop(ctx, v, IntBinop::ShiftLeft)
        && const_value(ctx, amt) == Some(1)
    {
        return Some(x);
    }
    if let Some((x, y)) = as_int_binop(ctx, v, IntBinop::Mul) {
        if const_value(ctx, y) == Some(2) {
            return Some(x);
        }
        if const_value(ctx, x) == Some(2) {
            return Some(y);
        }
    }
    None
}

/// If `v` is `(a & b) * 2`, return `(a, b)`.
fn as_doubled_and(ctx: &Context, v: ValueId) -> Option<(ValueId, ValueId)> {
    as_int_binop(ctx, as_doubled(ctx, v)?, IntBinop::And)
}

fn int_binop(lhs: ValueId, rhs: ValueId, op: IntBinop) -> Mnemonic {
    Mnemonic::Binop(Binary {
        lhs,
        rhs,
        op: Binop::Int(op),
    })
}

/// Collapse a bitwise/arithmetic identity rooted at `m`, returning the
/// equivalent single-operation mnemonic, or `None` if no pattern matches.
pub(super) fn simplify_identity(ctx: &Context, m: &Mnemonic) -> Option<Mnemonic> {
    let &Mnemonic::Binop(Binary {
        lhs,
        rhs,
        op: Binop::Int(op),
    }) = m
    else {
        return None;
    };

    match op {
        // (a + b) - ((a & b) << 1)  →  a ^ b
        // (a | b) - (a & b)         →  a ^ b
        // `-` is not commutative: the reducible term is always the rhs.
        IntBinop::Sub => {
            if let Some((a, b)) = as_int_binop(ctx, lhs, IntBinop::Add)
                && let Some(cd) = as_doubled_and(ctx, rhs)
                && same_operands((a, b), cd)
            {
                return Some(int_binop(a, b, IntBinop::Xor));
            }
            if let Some((a, b)) = as_int_binop(ctx, lhs, IntBinop::Or)
                && let Some(cd) = as_int_binop(ctx, rhs, IntBinop::And)
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
                if let Some((a, b)) = as_int_binop(ctx, base, IntBinop::Xor)
                    && let Some(cd) = as_doubled_and(ctx, extra)
                    && same_operands((a, b), cd)
                {
                    return Some(int_binop(a, b, IntBinop::Add));
                }
                if let Some((a, b)) = as_int_binop(ctx, base, IntBinop::Or)
                    && let Some(cd) = as_int_binop(ctx, extra, IntBinop::And)
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
fn known_align(ctx: &Context, v: ValueId, depth: u32) -> u32 {
    if depth == 0 {
        return 0;
    }
    if let Some(c) = const_value(ctx, v) {
        // A constant's alignment is its trailing-zero count; `0` is aligned to
        // its full width.
        return if c == 0 {
            (size_bits(ctx, v)).min(64)
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
    }) = ctx.get_insn(id).mnemonic()
    else {
        return 0;
    };
    let (a, b) = (
        || known_align(ctx, lhs, depth - 1),
        || known_align(ctx, rhs, depth - 1),
    );
    match op {
        // `&` keeps a low bit zero if it is zero in *either* operand — so a mask
        // is what *establishes* alignment: `x & ~7` is 8-aligned regardless of x.
        IntBinop::And => a().max(b()),
        // `|`/`^`/`+`/`-` keep a low bit zero only if it is zero in *both*.
        IntBinop::Or | IntBinop::Xor | IntBinop::Add | IntBinop::Sub => a().min(b()),
        // A left shift by a constant appends that many trailing zeros.
        IntBinop::ShiftLeft => match const_value(ctx, rhs) {
            Some(s) => a().saturating_add(s as u32),
            None => 0,
        },
        // Alignments add under multiplication.
        IntBinop::Mul => a().saturating_add(b()),
        _ => 0,
    }
}

/// The bit width of `v`'s output.
fn size_bits(ctx: &Context, v: ValueId) -> u32 {
    (value_size(ctx, v) as u32).saturating_mul(8)
}

/// Bound on the recursion `known_align` does back through a realignment cascade.
/// Inlined prologues stack only a handful of `& ~7` on top of one another, so a
/// generous fixed depth covers them without risking pathological walks.
const ALIGN_DEPTH: u32 = 64;

/// The `(constant, other_operand)` pairs of a binop — one entry per side that is
/// a numeric constant (both-constant binops are handled by [`super::fold`] before
/// this sub-pass, so in practice at most one side is constant here).
fn const_operands(ctx: &Context, lhs: ValueId, rhs: ValueId) -> Vec<(u64, ValueId)> {
    let mut out = Vec::new();
    if let Some(c) = const_value(ctx, rhs) {
        out.push((c, lhs));
    }
    if let Some(c) = const_value(ctx, lhs) {
        out.push((c, rhs));
    }
    out
}

/// If `v` is `x OP c` (or `c OP x`) for integer op `want` with a numeric constant
/// `c`, return `(x, c)`.
fn binop_const(ctx: &Context, v: ValueId, want: IntBinop) -> Option<(ValueId, u64)> {
    let (lhs, rhs) = as_int_binop(ctx, v, want)?;
    const_operands(ctx, lhs, rhs)
        .into_iter()
        .next()
        .map(|(c, x)| (x, c))
}

/// `~v` over `size` bytes expressed *without* emitting a fresh xor: a folded
/// constant when `v` is constant, or `x` when `v` is `x ^ ~0` (double-negation).
/// `None` when representing `~v` would need a new instruction — the caller then
/// declines the rewrite, keeping it strictly size-reducing.
fn simplify_not(ctx: &mut Context, v: ValueId, size: usize) -> Option<ValueId> {
    let all = all_ones(size);
    if let Some(c) = const_value(ctx, v) {
        return Some(ctx.get_const((!c) & all, size).id());
    }
    if let Some((x, c)) = binop_const(ctx, v, IntBinop::Xor)
        && (c & all) == all
    {
        return Some(x);
    }
    None
}

/// Constant-absorbing and De Morgan rewrites that, unlike [`simplify_identity`],
/// must intern a folded constant and so take `&mut Context`. All are exact in
/// `size`-byte modular arithmetic and strictly simplify (never grow the DAG):
///
/// ```text
///   (x & c1) & c2   →  x & (c1 & c2)     stacked masks collapse
///   (x ^ c1) ^ c2   →  x ^ (c1 ^ c2)     stacked xor-constants collapse
///   (a | b) ^ ~0    →  (~a) & (~b)        De Morgan, only when both ~ collapse
///   (a & b) ^ ~0    →  (~a) | (~b)
/// ```
///
/// De Morgan fires only when each `~operand` is itself a constant or a
/// double-negation (so no new xor is created), which is exactly the shape
/// compilers emit for `a & const` as `~(~a | ~const)`. Returns whether it
/// rewrote the root.
fn simplify_bitwise(ctx: &mut Context, ic: &InsnCtx, ed: &mut Editor) -> bool {
    let &Mnemonic::Binop(Binary {
        lhs,
        rhs,
        op: Binop::Int(op),
    }) = ic.mnemonic
    else {
        return false;
    };
    let size = ic.size;
    let all = all_ones(size);

    match op {
        // (x & c1) & c2 → x & (c1 & c2), and dropping a redundant alignment mask.
        IntBinop::And => {
            for (outer, inner) in const_operands(ctx, lhs, rhs) {
                // `inner & mask → inner` when `inner` is already aligned to the
                // mask's granularity. This collapses the cascade of `& ~7` that
                // inlined, stack-realigning prologues emit: each mask after the
                // first sits on a base already proven 8-aligned minus an
                // 8-multiple, so it is a no-op.
                if let Some(k) = align_mask_bits(outer, size)
                    && known_align(ctx, inner, ALIGN_DEPTH) >= k
                {
                    ed.replace(ctx, ic.insn_id, inner);
                    return true;
                }
                if let Some((x, c1)) = binop_const(ctx, inner, IntBinop::And) {
                    let folded = ctx.get_const((c1 & outer) & all, size).id();
                    ed.replace_with_new_insn(
                        ctx,
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
            for (outer, inner) in const_operands(ctx, lhs, rhs) {
                // (x ^ c1) ^ c2 → x ^ (c1 ^ c2)
                if let Some((x, c1)) = binop_const(ctx, inner, IntBinop::Xor) {
                    let folded = ctx.get_const((c1 ^ outer) & all, size).id();
                    ed.replace_with_new_insn(
                        ctx,
                        ic.block_id,
                        ic.insn_id,
                        int_binop(x, folded, IntBinop::Xor),
                        size,
                    );
                    return true;
                }
                // De Morgan: `inner ^ ~0`, pushing the complement inward.
                if outer != all {
                    continue;
                }
                for (dual_in, dual_out) in
                    [(IntBinop::Or, IntBinop::And), (IntBinop::And, IntBinop::Or)]
                {
                    let Some((a, b)) = as_int_binop(ctx, inner, dual_in) else {
                        continue;
                    };
                    let (Some(na), Some(nb)) =
                        (simplify_not(ctx, a, size), simplify_not(ctx, b, size))
                    else {
                        continue;
                    };
                    ed.replace_with_new_insn(
                        ctx,
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

/// The byte width of `v`'s output.
fn value_size(ctx: &Context, v: ValueId) -> usize {
    ValueRef::new(v, ctx).size()
}

/// If `v` is `zext(src)`, returns `(src, src_width)`.
fn as_zext(ctx: &Context, v: ValueId) -> Option<(ValueId, usize)> {
    let ValueId::Instruction(id) = v else {
        return None;
    };
    match ctx.get_insn(id).mnemonic() {
        Mnemonic::Zext(z) => Some((z.src, value_size(ctx, z.src))),
        _ => None,
    }
}

/// Whether `v` is produced by an operation that always yields `0` or `1`: an
/// integer comparison, a boolean binop, or a logical `!`.
fn is_boolean(ctx: &Context, v: ValueId) -> bool {
    let ValueId::Instruction(id) = v else {
        return false;
    };
    match ctx.get_insn(id).mnemonic() {
        Mnemonic::Binop(Binary {
            op: Binop::Int(op), ..
        }) => op.is_comparison(),
        Mnemonic::Binop(Binary {
            op: Binop::Bool(_), ..
        }) => true,
        Mnemonic::Unop(Unary {
            op: Unop::BoolNot, ..
        }) => true,
        _ => false,
    }
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

/// Comparison-idiom simplifications around `zext` and boolean values — the shape
/// a compiler emits for `if (x != 0)` after flag materialization:
///
/// ```text
///   zext(v) == 0   →   v == 0          (zext is zero-preserving)
///   zext(v) != 0   →   v != 0
///   b != 0         →   b               (b already 0/1)
///   b == 0         →   !b
///   !(a == b)      →   a != b          (negating an equality test)
///   !(a != b)      →   a == b
/// ```
///
/// Together these collapse `zext(!(x == 0)) != 0` down to `x != 0`. Returns
/// whether the root was rewritten.
fn simplify_compare(ctx: &mut Context, ic: &InsnCtx, ed: &mut Editor) -> bool {
    match ic.mnemonic {
        &Mnemonic::Binop(Binary {
            lhs,
            rhs,
            op: Binop::Int(op),
        }) if matches!(op, IntBinop::Equal | IntBinop::NotEqual) => {
            for (c, other) in const_operands(ctx, lhs, rhs) {
                if c != 0 {
                    continue;
                }
                // `zext(v) ==/!= 0` → `v ==/!= 0`: comparing a zero-extended
                // value against 0 is comparing the source against 0.
                if let Some((src, src_size)) = as_zext(ctx, other) {
                    let zero = ctx.get_const(0, src_size).id();
                    ed.replace_with_new_insn(
                        ctx,
                        ic.block_id,
                        ic.insn_id,
                        int_binop(src, zero, op),
                        ic.size,
                    );
                    return true;
                }
                // A boolean is already `0`/`1`, so `b != 0` is `b` and `b == 0`
                // is `!b`. The width must match so the forwarded/negated value
                // is a drop-in for the comparison result.
                if is_boolean(ctx, other) && value_size(ctx, other) == ic.size {
                    match op {
                        IntBinop::NotEqual => {
                            ed.replace(ctx, ic.insn_id, other);
                            return true;
                        }
                        IntBinop::Equal => {
                            ed.replace_with_new_insn(
                                ctx,
                                ic.block_id,
                                ic.insn_id,
                                Mnemonic::Unop(Unary {
                                    op: Unop::BoolNot,
                                    src: other,
                                }),
                                ic.size,
                            );
                            return true;
                        }
                        _ => {}
                    }
                }
            }
            false
        }
        &Mnemonic::Unop(Unary {
            op: Unop::BoolNot,
            src,
        }) => {
            // `!(a == b)` → `a != b` and `!(a != b)` → `a == b`.
            let ValueId::Instruction(id) = src else {
                return false;
            };
            if let &Mnemonic::Binop(Binary {
                lhs,
                rhs,
                op: Binop::Int(inner),
            }) = ctx.get_insn(id).mnemonic()
                && let Some(flipped) = negated_compare(inner)
            {
                ed.replace_with_new_insn(
                    ctx,
                    ic.block_id,
                    ic.insn_id,
                    int_binop(lhs, rhs, flipped),
                    ic.size,
                );
                return true;
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
                        %a = load(i32, &A);
                        %b = load(i32, &B);
                        %sum = %a + %b;
                        %and = %a & %b;
                        %dbl = %and << 0x1;
                        %root = %sum - %dbl;
                        return [%root];
                "
        );

        let aliases = AliasResult::simple(&ctx);
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
                        %a = load(i32, &A);
                        %b = load(i32, &B);
                        %or = %a | %b;
                        %and = %a & %b;
                        %root = %or - %and;
                        return [%root];
                "
        );

        let aliases = AliasResult::simple(&ctx);
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
                        %a = load(i32, &A);
                        %b = load(i32, &B);
                        %or = %a | %b;
                        %and = %a & %b;
                        %root = %or + %and;
                        return [%root];
                "
        );

        let aliases = AliasResult::simple(&ctx);
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
                        %i = load(i32, &I);
                        %n1 = %i ^ 0xffffffff;
                        %n2 = %n1 | 0xfffffffe;
                        %n3 = %n2 ^ 0xffffffff;
                        %n4 = %n3 & 0xfffffffd;
                        %dbl = %n4 * 0x2;
                        %x = %i ^ 0x1;
                        %inc = %dbl + %x;
                        store(&I, %inc);
                        return [0x1000];
                "
        );

        let aliases = AliasResult::simple(&ctx);
        gvn_function(&mut ctx, f, Some(&aliases));
        crate::remove_dead_insns(&mut ctx, entry);
        let text = BasicBlock::from_id(&ctx, entry).to_string();
        assert!(
            text.contains(" + 0x1") || text.contains(" + 0x00000001"),
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
                        %x = load(i32, &X);
                        %eq = %x == 0x0;
                        %ne = ! %eq;
                        %z = zext(i32, %ne);
                        %cond = %z != 0x0;
                        if %cond goto <0x2000> else goto <0x1000>;
                    <0x1000>
                        return [0x0];
                    <0x2000>
                        return [0x1];
                "
        );

        let aliases = AliasResult::simple(&ctx);
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
                        %v = load(i8, &V);
                        %z = zext(i32, %v);
                        %cond = %z != 0x0;
                        %c8 = zext(i8, %cond);
                        store(&B, %c8);
                        return [0x0];
                "
        );

        let aliases = AliasResult::simple(&ctx);
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
                        %sp = load(i32, &SP);
                        %s0 = %sp - 0x10;
                        %base = %s0 & 0xfffffff8;
                        %a0 = %base - 0x270;
                        %a1 = %a0 & 0xfffffff8;
                        %b0 = %a1 - 0x90;
                        %b1 = %b0 & 0xfffffff8;
                        %c0 = %b1 - 0x88;
                        %c1 = %c0 & 0xfffffff8;
                        return [%c1];
                "
        );

        let aliases = AliasResult::simple(&ctx);
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
                        %sp = load(i32, &SP);
                        %base = %sp & 0xfffffff8;
                        %a0 = %base - 0xa4;
                        %a1 = %a0 & 0xfffffff8;
                        return [%a1];
                "
        );

        let aliases = AliasResult::simple(&ctx);
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
                        %a = load(i32, &A);
                        %b = load(i32, &B);
                        %sum = %a + %b;
                        %and = %a & %b;
                        %root = %sum - %and;
                        return [%root];
                "
        );

        let aliases = AliasResult::simple(&ctx);
        gvn_function(&mut ctx, f, Some(&aliases));
        let text = BasicBlock::from_id(&ctx, entry).to_string();
        assert!(
            text.contains(" - "),
            "a non-idiom subtraction must be preserved, got:\n{text}"
        );
    }
}
