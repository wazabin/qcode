//! A very minimal value-range analysis.
//!
//! Answers one question: what unsigned values can `v` take when control is at
//! a given block? The result is a single inclusive interval `[min, max]`,
//! masked to the value's byte width — no strides, no signed intervals, no
//! fixpoint over loops. Anything the analysis cannot reason about goes to the
//! full-width range (Top).
//!
//! Two sources of information are combined:
//! - a backward walk through SSA definitions (literals, zext/sext, add/sub/
//!   mul/shift/and with intervals, comparisons),
//! - guard refinement from dominating bound checks: walking up the
//!   single-predecessor chain from the query block, a `CBranch` whose
//!   condition compares the queried value against a constant narrows the
//!   interval according to which edge was taken.
//!
//! The motivating consumer is the jump-table pass
//! ([`crate::cfg::HandleJumpTables`]): bounding the index of a
//! `table_base + index * scale` indirect branch gives the table's size.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use qcode::{
    context::Context,
    value::{
        BasicBlock, Value, ValueId, ValueRef,
        block::BlockId,
        insn::{Binary, Binop, BoolBinop, InstructionId, IntBinop, Mnemonic, Unary, Unop},
    },
};

/// How many SSA definitions a single query may walk through.
const RECURSE_CAP: usize = 16;
/// How many single-predecessor blocks the guard walk may climb.
const GUARD_DEPTH_CAP: usize = 8;

/// An unsigned interval `[min, max]` (both endpoints inclusive), masked to the
/// bit-width of the value it describes. `min <= max` is an invariant: empty
/// intervals are never produced (a contradiction is clamped, not represented).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueRange {
    pub min: u64,
    pub max: u64,
}

impl ValueRange {
    /// The full range for a `size`-byte value: `[0, all_ones(size)]`. This is
    /// Top, the "don't know" element.
    pub fn top(size: usize) -> Self {
        ValueRange {
            min: 0,
            max: all_ones(size),
        }
    }

    /// The single point `[c, c]`.
    pub fn exact(c: u64) -> Self {
        ValueRange { min: c, max: c }
    }

    /// Number of distinct values in the interval, saturating at `u64::MAX`.
    /// For a bounded jump-table index this is the table-size bound.
    pub fn count(&self) -> u64 {
        (self.max - self.min).saturating_add(1)
    }

    /// True if this is the full range for a `size`-byte value (i.e. Top).
    pub fn is_full(&self, size: usize) -> bool {
        self.min == 0 && self.max == all_ones(size)
    }

    /// True if the interval is narrower than Top for the given width, i.e.
    /// usable as a bound.
    pub fn is_bounded(&self, size: usize) -> bool {
        !self.is_full(size)
    }

    /// Set intersection. If the intervals are disjoint (a contradiction, e.g.
    /// a guard on a dead edge), returns `self` unchanged: any non-empty
    /// over-approximation is sound, and we never represent emptiness.
    fn intersect(self, other: Self) -> Self {
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);
        if min > max {
            self
        } else {
            ValueRange { min, max }
        }
    }
}

/// The range of possible runtime values of `value` along any path reaching
/// `at_block`, evaluated at the entry of `at_block`.
///
/// `at_block` is the program point of the question — for the jump-table pass,
/// the block containing the `BranchInd`. Guards are discovered by walking the
/// single-predecessor chain upward from `at_block`, so a bound check
/// dominating it (`if idx < N goto dispatch`) tightens the result. Querying a
/// value defined *inside* `at_block` is fine: SSA dominance puts its
/// definition before the terminator, and any guard on it lives in a strictly
/// earlier block, so the walk loses nothing.
///
/// Falls back to [`ValueRange::top`] for anything it cannot reason about.
pub fn value_range(ctx: &Context, value: ValueId, at_block: BlockId) -> ValueRange {
    Solver {
        ctx,
        at_block,
        memo: HashMap::default(),
        in_progress: HashSet::default(),
    }
    .range(value, 0)
}

/// The all-ones bit pattern for a `size`-byte value.
fn all_ones(size: usize) -> u64 {
    if size >= 8 {
        u64::MAX
    } else {
        (1u64 << (size * 8)) - 1
    }
}

/// Concrete value of `v` when it is a non-symbolic literal, else `None`.
/// Symbolic literals (block/function/string references) are not numeric
/// constants and must not participate in interval arithmetic.
fn numeric_const(ctx: &Context, v: ValueId) -> Option<u64> {
    if let ValueId::Literal(id) = v
        && ctx.values.literals[id].symbolic.is_some()
    {
        return None;
    }
    match ValueRef::new(v, ctx) {
        ValueRef::Literal(c) => Some(c.value()),
        _ => None,
    }
}

fn value_size(ctx: &Context, v: ValueId) -> usize {
    ValueRef::new(v, ctx).size()
}

// ---------------------------------------------------------------------------
// Interval arithmetic (unsigned; overflow past the width mask => None => Top)
// ---------------------------------------------------------------------------

fn add(a: ValueRange, b: ValueRange, mask: u64) -> Option<ValueRange> {
    let min = a.min.checked_add(b.min)?;
    let max = a.max.checked_add(b.max)?;
    (max <= mask).then_some(ValueRange { min, max })
}

fn sub(a: ValueRange, b: ValueRange) -> Option<ValueRange> {
    // Underflow anywhere in the interval would wrap; bail out entirely.
    // Lazy `then` so the subtractions are evaluated only once the guard holds —
    // `then_some` would compute `a.min - b.max` eagerly and overflow when it fails.
    (a.min >= b.max).then(|| ValueRange {
        min: a.min - b.max,
        max: a.max - b.min,
    })
}

fn mul(a: ValueRange, b: ValueRange, mask: u64) -> Option<ValueRange> {
    let min = a.min.checked_mul(b.min)?;
    let max = a.max.checked_mul(b.max)?;
    (max <= mask).then_some(ValueRange { min, max })
}

fn shl(a: ValueRange, shift: u64, mask: u64) -> Option<ValueRange> {
    if shift >= 64 {
        return None;
    }
    mul(a, ValueRange::exact(1u64 << shift), mask)
}

/// The smallest interval containing both `a` and `b` (their join / convex hull).
/// A sound over-approximation of the set union `a ∪ b`.
fn hull(a: ValueRange, b: ValueRange) -> ValueRange {
    ValueRange {
        min: a.min.min(b.min),
        max: a.max.max(b.max),
    }
}

fn shr(a: ValueRange, shift: u64) -> ValueRange {
    let s = shift.min(63);
    ValueRange {
        min: a.min >> s,
        max: a.max >> s,
    }
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

struct Solver<'a> {
    ctx: &'a Context<'a>,
    at_block: BlockId,
    memo: HashMap<ValueId, ValueRange>,
    in_progress: HashSet<ValueId>,
}

impl Solver<'_> {
    fn range(&mut self, v: ValueId, depth: usize) -> ValueRange {
        let size = value_size(self.ctx, v);
        if depth >= RECURSE_CAP {
            return ValueRange::top(size);
        }
        if let Some(&r) = self.memo.get(&v) {
            return r;
        }
        if !self.in_progress.insert(v) {
            return ValueRange::top(size);
        }

        let arith = match v {
            ValueId::Literal(_) => match numeric_const(self.ctx, v) {
                Some(c) => ValueRange::exact(c),
                None => ValueRange::top(size),
            },
            ValueId::Instruction(id) => self.transfer(id, depth),
            // BlockParams (phis), varnodes and the rest are opaque: bounding
            // them is the guard refinement's job.
            _ => ValueRange::top(size),
        };
        // Guards are checked for every visited value, not just the query
        // root, so a guard on `idx` still bounds `zext(idx) * 4`.
        let result = arith.intersect(self.guard_refine(v));

        self.in_progress.remove(&v);
        self.memo.insert(v, result);
        result
    }

    /// Range of the value produced by instruction `id`, from its operands.
    fn transfer(&mut self, id: InstructionId, depth: usize) -> ValueRange {
        let ctx = self.ctx;
        let insn = ctx.get_insn(id);
        let size = insn.size();
        let mask = all_ones(size);
        let top = ValueRange::top(size);

        match insn.mnemonic() {
            Mnemonic::Binop(Binary {
                op: Binop::Int(op),
                lhs,
                rhs,
            }) => {
                let (lhs, rhs) = (*lhs, *rhs);
                match *op {
                    IntBinop::Equal
                    | IntBinop::NotEqual
                    | IntBinop::Less
                    | IntBinop::SLess
                    | IntBinop::LessEqual
                    | IntBinop::SLessEqual => ValueRange { min: 0, max: 1 },

                    IntBinop::Add => {
                        let a = self.range(lhs, depth + 1);
                        let b = self.range(rhs, depth + 1);
                        add(a, b, mask).unwrap_or(top)
                    }
                    IntBinop::Sub => {
                        let a = self.range(lhs, depth + 1);
                        let b = self.range(rhs, depth + 1);
                        sub(a, b).unwrap_or(top)
                    }
                    IntBinop::Mul => {
                        let a = self.range(lhs, depth + 1);
                        let b = self.range(rhs, depth + 1);
                        mul(a, b, mask).unwrap_or(top)
                    }
                    // Shifts only with a constant count: a shift interval's
                    // bounds are not the shifts of the value's bounds.
                    IntBinop::ShiftLeft => match numeric_const(ctx, rhs) {
                        Some(c) => shl(self.range(lhs, depth + 1), c, mask).unwrap_or(top),
                        None => top,
                    },
                    IntBinop::ShiftRight => match numeric_const(ctx, rhs) {
                        Some(c) => shr(self.range(lhs, depth + 1), c),
                        None => top,
                    },
                    // `x & c` can clear bits anywhere, so the min is 0; the
                    // result can exceed neither operand's max.
                    IntBinop::And => {
                        let bound = match (numeric_const(ctx, lhs), numeric_const(ctx, rhs)) {
                            (_, Some(c)) => self.range(lhs, depth + 1).max.min(c),
                            (Some(c), _) => self.range(rhs, depth + 1).max.min(c),
                            (None, None) => return top,
                        };
                        ValueRange { min: 0, max: bound }
                    }
                    _ => top,
                }
            }

            Mnemonic::Unop(Unary {
                op: Unop::BoolNot, ..
            }) => ValueRange { min: 0, max: 1 },

            // Zero-extension preserves the unsigned value, and the source
            // interval always fits in the wider output mask.
            Mnemonic::Zext(z) => self.range(z.src, depth + 1),
            // Sign-extension preserves the value only when the whole source
            // interval has the sign bit clear; a straddling interval becomes
            // two disjoint unsigned intervals, which we cannot represent.
            Mnemonic::Sext(s) => {
                let src_size = value_size(ctx, s.src);
                let src = self.range(s.src, depth + 1);
                let sign_bit = (all_ones(src_size) >> 1) + 1;
                if src.max < sign_bit { src } else { top }
            }
            // A low-bytes extract is the identity when the value fits.
            Mnemonic::Range(r) if r.start == 0 => {
                let src = self.range(r.src, depth + 1);
                if src.max <= mask { src } else { top }
            }

            // Loads, calls, unops, ... : opaque.
            _ => top,
        }
    }

    /// Intersection of all bound checks on `v` found by climbing the
    /// single-predecessor chain from `at_block`.
    ///
    /// Soundness: when a block has exactly one predecessor, every path to it
    /// came through that edge, so the predecessor's branch polarity is a fact
    /// here — no dominator tree needed. The walk stops at the first merge
    /// point (or entry), trading precision for simplicity.
    fn guard_refine(&self, v: ValueId) -> ValueRange {
        let ctx = self.ctx;
        let mask = all_ones(value_size(ctx, v));
        let mut acc = ValueRange { min: 0, max: mask };
        let mut cur = self.at_block;

        for _ in 0..GUARD_DEPTH_CAP {
            let pred = {
                let block = BasicBlock::from_id(ctx, cur);
                let mut preds = block.predecessors();
                let Some((_, pred)) = preds.next() else { break };
                if preds.next().is_some() {
                    break;
                }
                pred
            };

            let pred_block = BasicBlock::from_id(ctx, pred);
            if let Some(term) = pred_block.iter().last()
                && let Mnemonic::CBranch(cb) = term.mnemonic()
            {
                let taken = if cb.success_block == cur {
                    Some(true)
                } else if cb.failure_block == cur {
                    Some(false)
                } else {
                    None
                };
                if let Some(taken) = taken
                    && let Some(r) = self.refine_condition(cb.condition, v, taken, mask, 0)
                {
                    acc = acc.intersect(r);
                }
            }
            cur = pred;
        }
        acc
    }

    /// Peel value-preserving zero-extensions off `v`, returning the underlying
    /// value. `zext` preserves the unsigned value exactly, so a comparison on the
    /// narrow value bounds the widened one (and vice versa); the constant is
    /// masked to the query width by the caller. Other widenings are left intact
    /// to stay sound for unsigned interval reasoning.
    fn zext_core(&self, mut v: ValueId) -> ValueId {
        for _ in 0..RECURSE_CAP {
            let ValueId::Instruction(id) = v else { break };
            match self.ctx.get_insn(id).mnemonic() {
                Mnemonic::Zext(z) => v = z.src,
                _ => break,
            }
        }
        v
    }

    /// Recognize `base - c` (a constant subtrahend), the shape the `cmp`
    /// instruction lowers to before its result feeds a flag test. Returns
    /// `(base, c)`.
    fn as_sub_const(&self, v: ValueId) -> Option<(ValueId, u64)> {
        let ValueId::Instruction(id) = v else {
            return None;
        };
        if let Mnemonic::Binop(Binary {
            op: Binop::Int(IntBinop::Sub),
            lhs,
            rhs,
        }) = self.ctx.get_insn(id).mnemonic()
            && let Some(c) = numeric_const(self.ctx, *rhs)
        {
            return Some((*lhs, c));
        }
        None
    }

    /// The interval implied for `v` by a (possibly compound) boolean
    /// `condition` being `taken`.
    ///
    /// Compilers lower `switch` bound checks like `x <= 7` (x86 `cmp; ja`/`jbe`)
    /// into a negated disjunction of primitive comparisons, e.g.
    /// `!((x < 7) || (x == 7))`, so the dominating `CBranch` condition is rarely
    /// a bare comparison. This walks the boolean structure:
    ///
    /// - `!c` flips the taken polarity and recurses.
    /// - `a && b` / `a || b` decompose to their operands at the *same* polarity;
    ///   whether the two refinements combine by intersection or by union follows
    ///   from the De Morgan duality of `taken`. A union is over-approximated by
    ///   the interval hull (sound: `union ⊆ hull`), so `(x < 7) || (x == 7)`
    ///   taken yields `[0, 6] ⊔ [7, 7] = [0, 7]`.
    /// - a leaf comparison is handled by [`Self::refine_from_cmp`].
    fn refine_condition(
        &self,
        condition: ValueId,
        v: ValueId,
        taken: bool,
        mask: u64,
        depth: usize,
    ) -> Option<ValueRange> {
        if depth >= GUARD_DEPTH_CAP {
            return None;
        }
        let ValueId::Instruction(id) = condition else {
            return None;
        };
        match self.ctx.get_insn(id).mnemonic() {
            Mnemonic::Unop(Unary {
                op: Unop::BoolNot,
                src,
            }) => self.refine_condition(*src, v, !taken, mask, depth + 1),

            Mnemonic::Binop(Binary {
                op: Binop::Bool(op @ (BoolBinop::And | BoolBinop::Or)),
                lhs,
                rhs,
            }) => {
                let (lhs, rhs) = (*lhs, *rhs);
                let l = self.refine_condition(lhs, v, taken, mask, depth + 1);
                let r = self.refine_condition(rhs, v, taken, mask, depth + 1);
                // `OR` taken and `AND` not-taken are disjunctions (value in the
                // union of the operand facts); the other two are conjunctions.
                let union = (*op == BoolBinop::Or) == taken;
                if union {
                    // A union with an unknown operand is unknown — no bound.
                    match (l, r) {
                        (Some(a), Some(b)) => Some(hull(a, b)),
                        _ => None,
                    }
                } else {
                    // A conjunction: an unknown operand simply adds no constraint.
                    match (l, r) {
                        (Some(a), Some(b)) => Some(a.intersect(b)),
                        (Some(a), None) | (None, Some(a)) => Some(a),
                        (None, None) => None,
                    }
                }
            }

            _ => self.refine_from_cmp(condition, v, taken, mask),
        }
    }

    /// The interval implied for `v` by `condition` being `taken`, when the
    /// condition is a comparison between `v` (matched up to a value-preserving
    /// zero-extension, so a guard on `EDI` bounds `zext(EDI)`) and a
    /// non-symbolic constant.
    ///
    /// Signed comparisons are skipped: `v s< k` with `k >= 0` constrains `v`
    /// to `[0, k-1] ∪ [2^(bits-1), MAX]`, two disjoint unsigned intervals;
    /// refining to either half would be unsound.
    ///
    /// Endpoints that would step past the width (`k = 0` in a true `v < k`,
    /// `k = MASK` in a false `v <= k`) belong to statically dead edges; they
    /// are clamped to a one-point interval, which is vacuously sound there.
    fn refine_from_cmp(
        &self,
        condition: ValueId,
        v: ValueId,
        taken: bool,
        mask: u64,
    ) -> Option<ValueRange> {
        let ValueId::Instruction(cond_insn) = condition else {
            return None;
        };
        let Mnemonic::Binop(Binary {
            op: Binop::Int(op),
            lhs,
            rhs,
        }) = self.ctx.get_insn(cond_insn).mnemonic()
        else {
            return None;
        };

        // Split the comparison into its constant side and its value side.
        let (mut cmp_base, mut k, v_is_lhs) =
            match (numeric_const(self.ctx, *lhs), numeric_const(self.ctx, *rhs)) {
                (None, Some(k)) => (*lhs, k, true),
                (Some(k), None) => (*rhs, k, false),
                _ => return None,
            };

        // The value side may be the `cmp`-derived `base - c0` (a constant
        // subtrahend), as in the ZF idiom `(x - c0) == 0`. For an equality test
        // that is exactly `base == k + c0`, so fold `c0` into the constant.
        if matches!(*op, IntBinop::Equal | IntBinop::NotEqual)
            && let Some((base, c0)) = self.as_sub_const(cmp_base)
        {
            cmp_base = base;
            k = k.wrapping_add(c0);
        }

        let core = self.zext_core(v);
        if self.zext_core(cmp_base) != core {
            return None;
        }
        let k = k & mask;

        let range = match (*op, v_is_lhs, taken) {
            // v < k
            (IntBinop::Less, true, true) => ValueRange {
                min: 0,
                max: k.saturating_sub(1),
            },
            (IntBinop::Less, true, false) => ValueRange { min: k, max: mask },
            // k < v
            (IntBinop::Less, false, true) => ValueRange {
                min: k.saturating_add(1).min(mask),
                max: mask,
            },
            (IntBinop::Less, false, false) => ValueRange { min: 0, max: k },
            // v <= k
            (IntBinop::LessEqual, true, true) => ValueRange { min: 0, max: k },
            (IntBinop::LessEqual, true, false) => ValueRange {
                min: k.saturating_add(1).min(mask),
                max: mask,
            },
            // k <= v
            (IntBinop::LessEqual, false, true) => ValueRange { min: k, max: mask },
            (IntBinop::LessEqual, false, false) => ValueRange {
                min: 0,
                max: k.saturating_sub(1),
            },
            // v == k taken, v != k not taken
            (IntBinop::Equal, _, true) | (IntBinop::NotEqual, _, false) => ValueRange::exact(k),
            // The complements (`!=` taken, `==` not taken) exclude one point
            // from the middle of the interval: not representable, no gain.
            _ => return None,
        };
        Some(range)
    }
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;

    #[test]
    fn literal_is_exact() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <block>
                goto <0x1001>;
            "
        );

        let c = ctx.get_const(0x2a, 8).id();
        let r = value_range(&ctx, c, block);
        assert_eq!((r.min, r.max), (0x2a, 0x2a));
        assert_eq!(r.count(), 1);
    }

    #[test]
    fn load_is_top() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            <block>
                %a = load(i64, &A);
                goto <0x1001>;
            "
        );

        let r = value_range(&ctx, a.into(), block);
        assert!(r.is_full(8));
        assert!(!r.is_bounded(8));
    }

    #[test]
    fn guard_true_edge_less() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            <entry>
                %idx = load(i64, &A);
                %c = %idx < 0x5;
                if %c goto <disp> else goto <oob>;
            <disp>
                goto <0x1001>;
            <oob>
                goto <0x1002>;
            "
        );

        let r = value_range(&ctx, idx.into(), disp);
        assert_eq!((r.min, r.max), (0, 4));
        assert_eq!(r.count(), 5);
    }

    #[test]
    fn guard_false_edge_less() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            <entry>
                %idx = load(i64, &A);
                %c = %idx < 0x5;
                if %c goto <disp> else goto <oob>;
            <disp>
                goto <0x1001>;
            <oob>
                goto <0x1002>;
            "
        );

        let r = value_range(&ctx, idx.into(), oob);
        assert_eq!((r.min, r.max), (5, u64::MAX));
    }

    #[test]
    fn guard_reversed_operands() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            <entry>
                %idx = load(i64, &A);
                %c = i64 0x5 < %idx;
                if %c goto <hi> else goto <lo>;
            <hi>
                goto <0x1001>;
            <lo>
                goto <0x1002>;
            "
        );

        let r = value_range(&ctx, idx.into(), hi);
        assert_eq!((r.min, r.max), (6, u64::MAX));

        let r = value_range(&ctx, idx.into(), lo);
        assert_eq!((r.min, r.max), (0, 5));
    }

    #[test]
    fn guard_less_equal() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            <entry>
                %idx = load(i64, &A);
                %c = %idx <= 0x7;
                if %c goto <disp> else goto <oob>;
            <disp>
                goto <0x1001>;
            <oob>
                goto <0x1002>;
            "
        );

        let r = value_range(&ctx, idx.into(), disp);
        assert_eq!((r.min, r.max), (0, 7));
        assert_eq!(r.count(), 8);
    }

    /// The `cmp; ja`/`jbe` lowering: a bound check `idx <= 7` becomes
    /// `!((idx < 7) || ((idx - 7) == 0))` on the *above* edge, so the
    /// fall-through (dispatch) edge must bound `idx` to `[0, 7]`. Exercises
    /// boolean-negation/`||` see-through plus the `(x - k) == 0` ZF idiom, and a
    /// guard reached through a `zext` of the compared value.
    #[test]
    fn jbe_flag_lowering_bounds_index() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 A;
            <entry>
                %edi = load(i32, &A);
                %idx = zext(i64, %edi);
                %lt = %idx < 0x7;
                %sub = %edi - 0x7;
                %zf = %sub == 0x0;
                %le = %lt || %zf;
                %above = ! %le;
                if %above goto <oob> else goto <disp>;
            <disp>
                goto <0x1001>;
            <oob>
                goto <0x1002>;
            "
        );

        // The dispatch (fall-through) edge: idx in [0, 7].
        let r = value_range(&ctx, idx.into(), disp);
        assert_eq!((r.min, r.max), (0, 7));
        assert_eq!(r.count(), 8);

        // The above edge: `idx >= 7` — the `< 7` disjunct refines the lower
        // bound, while the `== 7` complement excludes only a midpoint (not
        // representable), so the result is the sound over-approximation [7, MAX].
        let r = value_range(&ctx, idx.into(), oob);
        assert_eq!(r.min, 7);
    }

    #[test]
    fn equality_guard() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            <entry>
                %idx = load(i64, &A);
                %c = %idx == 0x3;
                if %c goto <eq> else goto <ne>;
            <eq>
                goto <0x1001>;
            <ne>
                goto <0x1002>;
            "
        );

        let r = value_range(&ctx, idx.into(), eq);
        assert_eq!((r.min, r.max), (3, 3));

        // `==` not taken excludes a single midpoint: not representable.
        let r = value_range(&ctx, idx.into(), ne);
        assert!(r.is_full(8));
    }

    #[test]
    fn not_equal_false_edge_is_exact() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            <entry>
                %idx = load(i64, &A);
                %c = %idx != 0x3;
                if %c goto <ne> else goto <eq>;
            <eq>
                goto <0x1001>;
            <ne>
                goto <0x1002>;
            "
        );

        let r = value_range(&ctx, idx.into(), eq);
        assert_eq!((r.min, r.max), (3, 3));
    }

    /// A guard on the narrow `idx` must flow through `zext(idx)`.
    #[test]
    fn zext_carries_guard() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 A;
            <entry>
                %idx = load(i32, &A);
                %c = %idx < i32 0x10;
                if %c goto <disp> else goto <oob>;
            <disp>
                %w = zext(i64, %idx);
                goto <0x1001>;
            <oob>
                goto <0x1002>;
            "
        );

        let r = value_range(&ctx, w.into(), disp);
        assert_eq!((r.min, r.max), (0, 15));
    }

    #[test]
    fn sext_straddling_sign_is_top() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 A;
            <block>
                %x = load(i32, &A);
                %w = sext(i64, %x);
                goto <0x1001>;
            "
        );

        let r = value_range(&ctx, w.into(), block);
        assert!(r.is_full(8));
    }

    /// Sext of a range with the sign bit provably clear is the identity.
    #[test]
    fn sext_of_guarded_nonnegative_stays_bounded() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 A;
            <entry>
                %x = load(i32, &A);
                %c = %x < i32 0x10;
                if %c goto <disp> else goto <oob>;
            <disp>
                %w = sext(i64, %x);
                goto <0x1001>;
            <oob>
                goto <0x1002>;
            "
        );

        let r = value_range(&ctx, w.into(), disp);
        assert_eq!((r.min, r.max), (0, 15));
    }

    #[test]
    fn add_constant_shifts_interval() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            <entry>
                %idx = load(i64, &A);
                %c = %idx < 0x5;
                if %c goto <disp> else goto <oob>;
            <disp>
                %j = %idx + 0x10;
                goto <0x1001>;
            <oob>
                goto <0x1002>;
            "
        );

        let r = value_range(&ctx, j.into(), disp);
        assert_eq!((r.min, r.max), (0x10, 0x14));
    }

    /// An unbounded i8 plus a constant can wrap past the byte mask: Top.
    #[test]
    fn add_wraps_to_top() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 A;
            <block>
                %a = load(i8, &A);
                %j = %a + i8 0x10;
                goto <0x1001>;
            "
        );

        let r = value_range(&ctx, j.into(), block);
        assert!(r.is_full(1));
    }

    #[test]
    fn mul_scales_interval() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            <entry>
                %idx = load(i64, &A);
                %c = %idx < 0x4;
                if %c goto <disp> else goto <oob>;
            <disp>
                %j = %idx * 0x4;
                goto <0x1001>;
            <oob>
                goto <0x1002>;
            "
        );

        let r = value_range(&ctx, j.into(), disp);
        assert_eq!((r.min, r.max), (0, 12));
    }

    #[test]
    fn shl_scales_interval() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            <entry>
                %idx = load(i64, &A);
                %c = %idx <= 0x3;
                if %c goto <disp> else goto <oob>;
            <disp>
                %j = %idx << 0x2;
                goto <0x1001>;
            <oob>
                goto <0x1002>;
            "
        );

        let r = value_range(&ctx, j.into(), disp);
        assert_eq!((r.min, r.max), (0, 12));
    }

    /// `x & c` is bounded by `c` with no guard at all.
    #[test]
    fn and_constant_upper_bound() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            <block>
                %a = load(i64, &A);
                %j = %a & 0x7;
                goto <0x1001>;
            "
        );

        let r = value_range(&ctx, j.into(), block);
        assert_eq!((r.min, r.max), (0, 7));
        assert_eq!(r.count(), 8);
    }

    /// The guard walk stops at a merge point: neither incoming edge's branch
    /// polarity is a fact there.
    #[test]
    fn multi_pred_block_is_top() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            <entry>
                %idx = load(i64, &A);
                %c = %idx < 0x5;
                if %c goto <a> else goto <b>;
            <a>
                goto <merge>;
            <b>
                goto <merge>;
            <merge>
                goto <0x1001>;
            "
        );

        let r = value_range(&ctx, idx.into(), merge);
        assert!(r.is_full(8));
    }

    /// A guard on a *different* SSA value must not refine the query.
    #[test]
    fn no_guard_when_operands_differ() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;
            <entry>
                %idx = load(i64, &A);
                %other = load(i64, &B);
                %c = %other < 0x5;
                if %c goto <disp> else goto <oob>;
            <disp>
                goto <0x1001>;
            <oob>
                goto <0x1002>;
            "
        );

        let r = value_range(&ctx, idx.into(), disp);
        assert!(r.is_full(8));
    }

    /// Signed bound checks are not refined (the unsigned image of `s<` is two
    /// disjoint intervals).
    #[test]
    fn signed_guard_skipped() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            <entry>
                %idx = load(i64, &A);
                %c = %idx s< 0x5;
                if %c goto <disp> else goto <oob>;
            <disp>
                goto <0x1001>;
            <oob>
                goto <0x1002>;
            "
        );

        let r = value_range(&ctx, idx.into(), disp);
        assert!(r.is_full(8));
    }

    #[test]
    fn comparison_result_is_bool_range() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;
            <block>
                %a = load(i64, &A);
                %b = load(i64, &B);
                %c = %a < %b;
                goto <0x1001>;
            "
        );

        let r = value_range(&ctx, c.into(), block);
        assert_eq!((r.min, r.max), (0, 1));
        assert_eq!(r.count(), 2);
    }

    /// A loop carried through block params must terminate (params are leaves,
    /// and both the recursion and the guard walk are depth-capped).
    #[test]
    fn depth_cap_terminates_on_block_param_cycle() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <entry @seed:i64>
                goto <lp @i=@seed>;
            <lp @i:i64>
                %j = @i + 0x1;
                goto <lp @i=%j>;
            "
        );

        let r = value_range(&ctx, j.into(), lp);
        assert!(r.is_full(8));
    }
}
