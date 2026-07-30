//! Width-narrowing sub-pass: sink slice extracts toward the leaves.
//!
//! The demand is the `Range` mnemonic's byte interval `[start, start + size)`.
//! Pushing it down the operations it commutes with, **cancelling** it against
//! widenings (`sext`/`zext`) and folding it into constants, collapses the
//! lift-time "widen, compute wide, truncate back" idiom to a single narrow
//! computation:
//!
//! ```text
//! %xa = sext(i64, %a)
//! %xb = sext(i64, %b)
//! %m  = %xa * %xb
//! %r  = %m[0:4]          ⇒  %r = %a *₃₂ %b      (low word of a product is width-agnostic)
//! ```
//!
//! An interval anchored above byte 0 is what erases a constant an extract never
//! observes — the byte-patch idiom, where a mask touches only bytes the slice
//! excludes. Nothing here folds that away directly: the extract distributes over
//! the operands, the constant narrows to zero, and [`Fold`](super::fold::Fold)'s
//! `x ^ 0 = x` finishes the job on the next turn of the chain's fixpoint,
//! leaving a value [`Cse`](super::cse::Cse) can merge with the unmasked extract.
//!
//! ```text
//! %x  = @a ^ 0x55
//! %r  = %x[1:4]          ⇒  %r = @a[1:4]        (0x55 lives in byte 0, outside [1,4))
//! ```
//!
//! This is the structural companion to [`mba_simplify`](crate::mba_simplify):
//! narrowing erases the `sext`/`mul`/`range` plumbing the MBA solver cannot
//! model, leaving a uniform-width MBA the solver *can* collapse.
//!
//! # Soundness
//!
//! Every rule materializes a *new* value for the demanded interval and never
//! touches the wide original, so any consumer of the bytes outside it is
//! unaffected. The distributions are exact under two's-complement wrapping.
//!
//! Which operations commute with the extract **depends on where it starts**:
//!
//! - `& | ^ ~` are per-bit — bit `i` of the result reads only bit `i` of the
//!   operands — so they distribute over *any* interval.
//! - `+ − × neg` propagate carries upward, so byte `start` of the result depends
//!   on the bytes *below* `start`. They distribute only when `start == 0`.
//!   (`neg` counts as arithmetic: it is `~x + 1`.)
//!
//! A shift by a whole number of bytes is the exception among the shifts: it
//! *relocates* the window rather than blocking it, so it composes with the
//! interval exactly as a nested `Range` does — bounded by the operand, since past
//! that the window reaches into the fill the shift introduced (zeros, or sign
//! bits for an arithmetic shift).
//!
//! Otherwise the extract is *not* sunk through `>>`, `/`, `%`, comparisons,
//! loads, … (their bytes depend on more than the corresponding operand bytes);
//! there it stops, leaving a `Range` of an opaque value. Dead wide originals are
//! left for DCE, per the GVN convention.

use std::any::Any;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use qcode::value::{
    FunctionId, QCodeView, Value, ValueId, ValueRef,
    block::BlockId,
    insn::{Binary, Binop, InstructionId, IntBinop, Mnemonic, Range, Sext, Unary, Unop, Zext},
};

use super::walk::{Claim, Editor, InsnCtx, SubPass};

use crate::{ContextView, FunctionBody};

/// Reads resolve through the selected function's static view, narrow values are
/// pushed into that body's arena, and constants are minted through shared
/// interners.
pub(super) struct NarrowTrunc;

/// The function-pass [`SubPass`] impl (body-local):
/// eligibility reads through `cx.body_view(body)`, the recursive `narrow_to`
/// rewrite mutates the checked-out body, and the forward goes through
/// [`Editor::replace`].
impl<'str> SubPass<'str> for NarrowTrunc {
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
        let Mnemonic::Range(Range { src, start, size }) = ic.mnemonic else {
            return Claim::Pass;
        };
        let src = src.qualify(ic.insn_id.func);
        let slice = Slice {
            start: *start,
            size: *size,
        };
        // Worth recursing only if the extract is an identity on its source (so
        // it forwards) or the source is something we can push through.
        if !slice.is_whole(value_size(cx.body_view(body), src))
            && !src_transformable(cx.body_view(body), src, slice)
        {
            return Claim::Pass;
        }
        let mut memo: HashMap<(ValueId, Slice), ValueId> = HashMap::default();
        let mut active: HashSet<(ValueId, Slice)> = HashSet::default();
        let narrowed = narrow_to(
            body,
            cx,
            src,
            slice,
            ic.insn_id,
            ic.block_id,
            &mut memo,
            &mut active,
        );
        if narrowed == ic.id {
            return Claim::Pass;
        }
        ed.replace(body, cx, ic.insn_id, narrowed);
        Claim::Done
    }
}

/// The demanded byte interval `[start, start + size)` of a value.
///
/// Carried through the recursion rather than fixed, because a `Range` of a
/// `Range` composes: extracting `[s₂, …)` of `[s₁, …)` demands `[s₁ + s₂, …)` of
/// the inner source. The memo is therefore keyed by value *and* interval — the
/// same value can be demanded at two different offsets in one rewrite.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Slice {
    start: usize,
    size: usize,
}

impl Slice {
    /// Does this interval cover exactly a whole `m`-byte value, making the
    /// extract an identity?
    fn is_whole(self, m: usize) -> bool {
        self.start == 0 && self.size == m
    }
}

/// Will [`narrow_to`] push through `v` rather than just wrap it in a `Range`?
///
/// This must not over-report. The sub-pass runs to a fixpoint, so claiming a
/// value is transformable when [`narrow_to`] would in fact stop makes it
/// materialize an extract identical to the one being visited, replace the
/// original with it, report a change, and do the same again next round.
fn src_transformable<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    v: ValueId,
    slice: Slice,
) -> bool {
    if numeric_const(host.shared(), v).is_some() {
        return true;
    }
    let ValueId::Instruction(iid) = v else {
        return false;
    };
    match host.insn_ref(iid).mnemonic() {
        Mnemonic::Binop(b) => match b.op {
            Binop::Int(op) => {
                distributive(op, slice.start)
                    || byte_move(host, op, b.rhs.qualify(iid.func))
                        .and_then(|m| {
                            moved_slice(m, slice, value_size(host, b.lhs.qualify(iid.func)))
                        })
                        .is_some()
            }
            _ => false,
        },
        Mnemonic::Unop(u) => unop_distributive(&u.op, slice.start),
        Mnemonic::Sext(Sext { src, .. }) => {
            ext_transformable(host, src.qualify(iid.func), slice, true)
        }
        Mnemonic::Zext(Zext { src, .. }) => {
            ext_transformable(host, src.qualify(iid.func), slice, false)
        }
        // Composing two extracts always removes a level of indirection.
        Mnemonic::Range(_) => true,
        _ => false,
    }
}

/// The extension arm of [`src_transformable`]: mirrors [`narrow_extension`]'s
/// case split, reporting only the positions where it does something.
fn ext_transformable<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    src: ValueId,
    slice: Slice,
    sext: bool,
) -> bool {
    let m = value_size(host, src);
    // Cancels against the extension, folds to a zero fill, or re-extends.
    slice.start + slice.size <= m || (slice.start >= m && !sext) || slice.start == 0
}

/// Does `op` commute with an extract anchored at byte `start`?
///
/// See the module header: the bitwise operations are per-bit and distribute over
/// any interval; the arithmetic ones carry upward and need `start == 0`.
fn distributive(op: IntBinop, start: usize) -> bool {
    match op {
        IntBinop::And | IntBinop::Or | IntBinop::Xor => true,
        IntBinop::Add | IntBinop::Sub | IntBinop::Mul => start == 0,
        _ => false,
    }
}

/// The unary counterpart of [`distributive`]: `~` is per-bit, `neg` is `~x + 1`
/// and so carries.
fn unop_distributive(op: &Unop, start: usize) -> bool {
    match op {
        Unop::IntNot => true,
        Unop::IntNegate => start == 0,
        _ => false,
    }
}

/// A shift-like operation that moves its operand by a whole number of bytes.
///
/// Such a shift *relocates* the demanded window rather than blocking it, so it
/// composes with the interval exactly as a nested `Range` does.
#[derive(Clone, Copy)]
enum ByteMove {
    /// `x >> 8k`, logical or arithmetic: the window moves *up* by `k` bytes.
    Down(usize),
    /// `x << 8k`, or the `x * 2^8k` a folder rewrote it to: *down* by `k`.
    Up(usize),
}

/// Recognize `op` with a constant right operand as a whole-byte move.
fn byte_move<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    op: IntBinop,
    rhs: ValueId,
) -> Option<ByteMove> {
    let k = numeric_const(host.shared(), rhs)?;
    match op {
        IntBinop::ShiftRight | IntBinop::SShiftRight => {
            (k % 8 == 0).then_some(ByteMove::Down((k / 8) as usize))
        }
        IntBinop::ShiftLeft => (k % 8 == 0).then_some(ByteMove::Up((k / 8) as usize)),
        // A left shift the constant folder turned into a multiply.
        IntBinop::Mul if k.is_power_of_two() => {
            let bits = k.trailing_zeros();
            (bits % 8 == 0).then_some(ByteMove::Up((bits / 8) as usize))
        }
        _ => None,
    }
}

/// The window to demand of the shift's operand, or `None` when the move would
/// take it outside the operand.
///
/// A right shift is only transparent while the window stays inside the operand:
/// past that it reaches into the fill the shift introduced — zeros for a logical
/// shift, sign bits for an arithmetic one — which the operand cannot answer for.
/// A left shift is transparent while the window starts at or above the shift
/// distance; below it the bytes are the shift's own zero fill.
fn moved_slice(m: ByteMove, slice: Slice, src_size: usize) -> Option<Slice> {
    match m {
        ByteMove::Down(k) => {
            let start = slice.start.checked_add(k)?;
            (start.checked_add(slice.size)? <= src_size).then_some(Slice {
                start,
                size: slice.size,
            })
        }
        ByteMove::Up(k) => Some(Slice {
            start: slice.start.checked_sub(k)?,
            size: slice.size,
        }),
    }
}

fn range_of(src: ValueId, slice: Slice, func: FunctionId) -> Mnemonic {
    Mnemonic::Range(Range {
        src: src.localize(func),
        start: slice.start,
        size: slice.size,
    })
}

// ---------------------------------------------------------------------------
// Body-local narrowing helpers.
// ---------------------------------------------------------------------------

/// Recursively extract the byte interval `slice` of a value.
#[allow(clippy::too_many_arguments)]
fn narrow_to<'str>(
    body: &mut FunctionBody<'str>,
    cx: ContextView<'_, 'str>,
    v: ValueId,
    slice: Slice,
    before: InstructionId,
    block: BlockId,
    memo: &mut HashMap<(ValueId, Slice), ValueId>,
    active: &mut HashSet<(ValueId, Slice)>,
) -> ValueId {
    if slice.is_whole(value_size(cx.body_view(body), v)) {
        return v;
    }
    if let Some(&cached) = memo.get(&(v, slice)) {
        return cached;
    }
    // Cycle backstop: a self-referential value (e.g. a block parameter whose
    // narrowing recurses back through itself) would otherwise recurse forever,
    // since the memo is only populated after the recursive call returns. If `v`
    // is already on the active recursion path at this interval, wrap it opaquely
    // instead — a bounded, correct (if unoptimized) slice — rather than pushing
    // through the cycle. This must never be load-bearing: a sound IR has no cyclic
    // pure-value dependency (verification rejects one); it only prevents a stack
    // overflow if one slips through.
    if !active.insert((v, slice)) {
        return push_slice(body, cx, v, slice, before, block);
    }

    let result = match v {
        ValueId::Instruction(iid) => match cx.body_view(body).insn_ref(iid).mnemonic().clone() {
            // A whole-byte shift relocates the window instead of stopping it:
            // `(x >> 8k)[s..]` is `x[(s+k)..]`, `(x << 8k)[s..]` is `x[(s-k)..]`.
            // This is what collapses the widen-and-reassemble idiom — a value
            // rebuilt from its halves only to have one taken straight back out —
            // once the `Or` distributes and each side lands on a window it can
            // answer directly.
            Mnemonic::Binop(Binary {
                op: Binop::Int(o),
                lhs,
                rhs,
            }) if byte_move(cx.body_view(body), o, rhs.qualify(iid.func))
                .and_then(|m| {
                    moved_slice(
                        m,
                        slice,
                        value_size(cx.body_view(body), lhs.qualify(iid.func)),
                    )
                })
                .is_some() =>
            {
                let lhs = lhs.qualify(iid.func);
                let moved = byte_move(cx.body_view(body), o, rhs.qualify(iid.func))
                    .and_then(|m| moved_slice(m, slice, value_size(cx.body_view(body), lhs)))
                    .expect("guarded above");
                narrow_to(body, cx, lhs, moved, before, block, memo, active)
            }

            Mnemonic::Binop(Binary {
                op: Binop::Int(o),
                lhs,
                rhs,
            }) if distributive(o, slice.start) => {
                let l = narrow_to(
                    body,
                    cx,
                    lhs.qualify(iid.func),
                    slice,
                    before,
                    block,
                    memo,
                    active,
                );
                let rr = narrow_to(
                    body,
                    cx,
                    rhs.qualify(iid.func),
                    slice,
                    before,
                    block,
                    memo,
                    active,
                );
                push_insn(
                    body,
                    cx,
                    Mnemonic::Binop(Binary {
                        op: Binop::Int(o),
                        lhs: l.localize(block.func),
                        rhs: rr.localize(block.func),
                    }),
                    slice.size,
                    before,
                    block,
                )
            }
            Mnemonic::Unop(Unary { op, src }) if unop_distributive(&op, slice.start) => {
                let s = narrow_to(
                    body,
                    cx,
                    src.qualify(iid.func),
                    slice,
                    before,
                    block,
                    memo,
                    active,
                );
                push_insn(
                    body,
                    cx,
                    Mnemonic::Unop(Unary {
                        op,
                        src: s.localize(block.func),
                    }),
                    slice.size,
                    before,
                    block,
                )
            }
            Mnemonic::Sext(Sext { src, .. }) => narrow_extension(
                body,
                cx,
                v,
                src.qualify(iid.func),
                slice,
                true,
                before,
                block,
                memo,
                active,
            ),
            Mnemonic::Zext(Zext { src, .. }) => narrow_extension(
                body,
                cx,
                v,
                src.qualify(iid.func),
                slice,
                false,
                before,
                block,
                memo,
                active,
            ),
            // A slice of a slice composes: bytes `[s, s + n)` of bytes
            // `[start, …)` of `src` are bytes `[start + s, start + s + n)` of
            // `src` itself.
            Mnemonic::Range(Range { src, start, .. }) => narrow_to(
                body,
                cx,
                src.qualify(iid.func),
                Slice {
                    start: start + slice.start,
                    size: slice.size,
                },
                before,
                block,
                memo,
                active,
            ),
            _ => push_slice(body, cx, v, slice, before, block),
        },
        _ if numeric_const(cx.body_view(body).shared(), v).is_some() => {
            let whole = numeric_const(cx.body_view(body).shared(), v).unwrap();
            let folded = shift_out_low(whole, slice.start) & low_mask(slice.size);
            cx.body_view(body).shared().get_const(folded, slice.size)
        }
        _ => push_slice(body, cx, v, slice, before, block),
    };

    active.remove(&(v, slice));
    memo.insert((v, slice), result);
    result
}

/// Narrow through a sign or zero extension, where `ext` is the extension value
/// itself and `src` the value being extended.
///
/// Three positions matter, given `m = size(src)`:
///
/// - the interval sits **inside the original value** (`start + size <= m`) — the
///   extension contributes nothing to it, so it cancels;
/// - the interval sits **entirely in the extension** (`start >= m`) — every byte
///   is a fill byte, which for `zext` is a constant zero. For `sext` it is a
///   replication of the sign bit, which needs a splat this pass has no way to
///   materialize, so it stops;
/// - the interval **straddles** the boundary. Anchored at byte 0 this is the
///   pre-existing widening case and re-extends to the demanded width; anchored
///   above it would need a concat of a slice and fill bytes, so it stops.
#[allow(clippy::too_many_arguments)]
fn narrow_extension<'str>(
    body: &mut FunctionBody<'str>,
    cx: ContextView<'_, 'str>,
    ext: ValueId,
    src: ValueId,
    slice: Slice,
    sext: bool,
    before: InstructionId,
    block: BlockId,
    memo: &mut HashMap<(ValueId, Slice), ValueId>,
    active: &mut HashSet<(ValueId, Slice)>,
) -> ValueId {
    let m = value_size(cx.body_view(body), src);
    if slice.start + slice.size <= m {
        return narrow_to(body, cx, src, slice, before, block, memo, active);
    }
    if slice.start >= m && !sext {
        return cx.body_view(body).shared().get_const(0, slice.size);
    }
    if slice.start > 0 {
        return push_slice(body, cx, ext, slice, before, block);
    }
    let m = if sext {
        Mnemonic::Sext(Sext {
            src: src.localize(block.func),
            size: slice.size,
        })
    } else {
        Mnemonic::Zext(Zext {
            src: src.localize(block.func),
            size: slice.size,
        })
    };
    push_insn(body, cx, m, slice.size, before, block)
}

/// Stop pushing: materialize the demanded interval as a plain extract of `v`.
fn push_slice<'str>(
    body: &mut FunctionBody<'str>,
    cx: ContextView<'_, 'str>,
    v: ValueId,
    slice: Slice,
    before: InstructionId,
    block: BlockId,
) -> ValueId {
    push_insn(
        body,
        cx,
        range_of(v, slice, block.func),
        slice.size,
        before,
        block,
    )
}

/// Insert a narrowed instruction before `before`.
fn push_insn<'str>(
    body: &mut FunctionBody<'str>,
    cx: ContextView<'_, 'str>,
    mnemonic: Mnemonic,
    size: usize,
    before: InstructionId,
    block: BlockId,
) -> ValueId {
    let id = body.push_mnemonic(cx.shr(), mnemonic, size);
    body.insert_insn_before(block, before, id);
    ValueId::Instruction(id)
}

/// Drop the low `start` bytes of a constant, saturating to zero rather than
/// overflowing the shift when the interval starts past the top of a `u64`.
fn shift_out_low(value: u64, start: usize) -> u64 {
    u32::try_from(start * 8)
        .ok()
        .and_then(|bits| value.checked_shr(bits))
        .unwrap_or(0)
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

fn value_size<'ctx, 'str: 'ctx>(host: impl QCodeView<'ctx, 'str>, v: ValueId) -> usize {
    ValueRef::from_view(host, v).size()
}

#[cfg(test)]
mod tests {
    use super::value_size;
    use crate::gvn::narrow_function;
    use crate::mba_simplify::mba_simplify;
    use qcode::{
        context::Context,
        value::{
            BasicBlock, FunctionBody, FunctionId, Instruction, ModuleView, ValueId, insn::Mnemonic,
        },
    };
    use qcode_emulator::{SizedValue, StandaloneEmulator};
    use qcode_macro::qcode;

    fn return_value(ctx: &Context, fun: FunctionId) -> ValueId {
        let root = FunctionBody::from_id(ctx, fun).root().expect("root").id;
        let &term = BasicBlock::from_id(ctx, root)
            .instruction_ids()
            .last()
            .expect("terminator");
        match Instruction::from_id(ctx, term).mnemonic() {
            Mnemonic::ReturnValue(r) => r.value.qualify(term.func),
            other => panic!("expected return, got {other:?}"),
        }
    }

    fn run(ctx: &Context, fun: FunctionId, a: u64, b: u64) -> Option<u64> {
        let root = FunctionBody::from_id(ctx, fun).root().expect("root").id;
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
            value_size(ModuleView::new(&ctx), return_value(&ctx, widemul)),
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
        assert_eq!(value_size(ModuleView::new(&ctx), return_value(&ctx, z)), 4);
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

    /// A shift by a whole number of bytes *moves* the demanded window, so a
    /// truncation sinks through it: `(x >> 8)[0:4]` is `x[1:5]`. A shift by
    /// anything else does not — the window would straddle byte boundaries the
    /// operand cannot answer for.
    #[test]
    fn truncation_sinks_through_a_byte_aligned_shift_only() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda byte_sh:
            <entry @a:i32 @b:i32>
                %xa = sext(i64, @a);
                %sh = %xa >> 0x8;
                %r = %sh[0:4];
                return %r;
            "
        );
        let before = sample(&ctx, byte_sh);
        assert!(narrow_function(&mut ctx, byte_sh));
        assert_eq!(
            sample(&ctx, byte_sh),
            before,
            "moving the window must not change the value"
        );

        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda bit_sh:
            <entry @a:i32 @b:i32>
                %xa = sext(i64, @a);
                %sh = %xa >> 0x3;
                %r = %sh[0:4];
                return %r;
            "
        );
        assert!(
            !narrow_function(&mut ctx, bit_sh),
            "a sub-byte shift must still stop the truncation"
        );
    }

    /// The widen-and-reassemble idiom: a value rebuilt from its halves only to
    /// have one taken straight back out. The `Or` distributes, the low half
    /// narrows to zero in the high window, the `* 2^32` moves the window down,
    /// and what is left is the half itself.
    #[test]
    fn collapses_a_value_reassembled_from_halves() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda halves:
            <entry @a:i32 @b:i32>
                %lo = zext(i64, @a);
                %hi = zext(i64, @b);
                %up = %hi * 0x100000000;
                %re = %lo | %up;
                %r = %re[4:8];
                return %r;
            "
        );
        let before = sample(&ctx, halves);
        assert!(narrow_function(&mut ctx, halves));
        // Narrowing leaves `0 | @b`; folding the identity away is `Fold`'s half
        // of the chain, exactly as for a masked-out constant.
        assert!(crate::gvn::constant_fold_function(&mut ctx, halves));
        assert_eq!(sample(&ctx, halves), before);

        // The reassembly is gone entirely: the return is the input half itself,
        // not an expression over the rebuilt value.
        assert!(
            matches!(return_value(&ctx, halves), ValueId::BlockParam(_)),
            "the high window should be the half itself, not a computation:\n{}",
            qcode::value::FunctionBody::from_id(&ctx, halves)
        );
        // And it is the *high* half: for (a, b) the result is b.
        assert_eq!(
            run(&ctx, halves, 0x1111_1111, 0x2222_2222),
            Some(0x2222_2222)
        );
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
        let root = FunctionBody::from_id(&ctx, mtmul).root().expect("root").id;
        while crate::dce::remove_dead_insns(&mut ctx, root) {}
        // mba_simplify's surface is pass-scoped; run it over a `BodyMut`
        // borrowing the body in place alongside the read-only shared state.
        let mba_changed = {
            let mut host = qcode::value::util::body_mut::BodyMut::new(
                &mut ctx.bodies[mtmul],
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
