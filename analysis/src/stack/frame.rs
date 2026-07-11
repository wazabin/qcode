//! Frame-relative classification of stack pointers, independent of the
//! `@stack_base` brighten/lower representation.
//!
//! A stack address appears in up to three shapes across the pipeline:
//!   * `@stack_base ± k`   — the interned brighten literal (legacy).
//!   * `@SP ± k`           — affine arithmetic on the incoming stack-pointer
//!     parameter (`@SP` = the root-block param whose `origin` is the SP register,
//!     created by `rewrite_callee_registers`).
//!   * `(@SP & -mask) ± k` — a frame realigned by `and rsp, -N`.
//!
//! [`frame_class`] recognises all three and reports whether a pointer lands in
//! this function's own locals or in the caller's frame. It is the representation-
//! agnostic replacement for matching `StackAddress` literals directly, and the
//! foundation for the frame-freshness alias rule.

use qcode::{
    context::Context,
    value::{FunctionId, FunctionRef, ValueId, VarnodeId, util::base_ref::HostRef},
};

use crate::gvn::affine::Numbering;

/// Maximum depth of a realignment cascade we will follow back to `@SP`. Real
/// frames nest only a handful of `and rsp,-N` steps; this is a loop backstop.
const ALIGN_CASCADE_LIMIT: u32 = 32;

/// Where a stack pointer lands relative to the entry stack pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameClass {
    /// Below entry SP: an own-frame local, younger than anything the caller can
    /// already name (the basis of frame freshness).
    Local,
    /// At or above entry SP: the return-address slot (offset 0) or an incoming
    /// stack argument (offset > 0) — both live in the caller's frame.
    CallerFrame,
}

/// The incoming stack-pointer root parameter of `fid`: the root-block param whose
/// `origin` is the stack-pointer register `sp_reg`. `None` if the function has no
/// such param (e.g. it never touched the stack, or registers were not promoted).
pub(crate) fn incoming_sp_param<'a, 'str: 'a>(
    host: impl Into<HostRef<'a, 'str>>,
    fid: FunctionId,
    sp_reg: VarnodeId,
) -> Option<ValueId> {
    FunctionRef::new(host.into(), fid)
        .root()?
        .params()
        .find(|p| p.origin() == Some(ValueId::Varnode(sp_reg)))
        .map(|p| p.id())
}

/// Whether `base` is a power-of-two stack realignment of the incoming stack
/// pointer — `@SP & -mask` — or a *cascade* of such realignments anchored to `@SP`
/// through downward steps: `(((@SP - a) & -m1) - b) & -m2 …`, as emitted by nested
/// `sub rsp,k; and rsp,-N` sequences. A realigned base sits within the current
/// frame, so every slot built on it is a local.
///
/// Soundness rests on monotonicity: a round-down mask only *lowers* an address
/// (`x & -2^k ≤ x`), so a value already at-or-below entry `@SP` stays there. The
/// value feeding each mask must therefore reach `@SP` (or an already-recognised
/// aligned base) through a **non-positive** offset — a *positive* offset before a
/// mask could round to an address at/above entry `@SP` (the caller's frame), which
/// must not be classified as a local.
fn is_aligned_sp(numbering: &Numbering, sp_param: ValueId, base: ValueId, depth: u32) -> bool {
    if depth == 0 {
        return false;
    }
    let Some(term) = numbering.alignment_base(base) else {
        return false;
    };
    // Peel the affine offset of the value being aligned; it must go downward.
    let (inner, off) = numbering.base_offset(term).unwrap_or((term, 0));
    if off > 0 {
        return false;
    }
    inner == sp_param || is_aligned_sp(numbering, sp_param, inner, depth - 1)
}

/// The signed byte offset of `v` from the entry stack pointer `@SP`, when `v` is
/// an `@SP`-rooted stack address. Every representation of the same slot yields the
/// same offset, so this is the **canonical slot key** and stable slot identity.
///
/// Returns `None` for a realigned (`@SP & -mask`) base — which has no stable
/// `@SP`-relative offset — and for any non-stack pointer.
pub(crate) fn frame_offset(
    _ctx: &Context,
    numbering: &Numbering,
    sp_param: ValueId,
    v: ValueId,
) -> Option<i64> {
    // Affine `@SP ± k` (the bare param decomposes to itself at offset 0).
    let (base, off) = numbering.base_offset(v).unwrap_or((v, 0));
    (base == sp_param).then_some(off)
}

/// Classify pointer `v` against the frame whose incoming stack pointer is
/// `sp_param`, decomposing `@SP ± k` through `numbering`. Returns `None` when `v`
/// is not stack-pointer-rooted.
pub(crate) fn frame_class(
    ctx: &Context,
    numbering: &Numbering,
    sp_param: ValueId,
    v: ValueId,
) -> Option<FrameClass> {
    // An `@SP`/`@stack_base`-relative slot: classify by the sign of its offset.
    if let Some(off) = frame_offset(ctx, numbering, sp_param, v) {
        return Some(by_sign(off));
    }
    // A realigned frame base (`@SP & -mask`, possibly cascaded): everything offset
    // from it is a local — incoming args never flow through the alignment mask.
    let (base, _) = numbering.base_offset(v).unwrap_or((v, 0));
    is_aligned_sp(numbering, sp_param, base, ALIGN_CASCADE_LIMIT).then_some(FrameClass::Local)
}

/// Below the entry stack pointer (`off < 0`) is an own-frame local; the
/// return-address slot (`0`) and incoming stack arguments (`> 0`) are the
/// caller's frame.
fn by_sign(off: i64) -> FrameClass {
    if off < 0 {
        FrameClass::Local
    } else {
        FrameClass::CallerFrame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{
        builder::Builder,
        testing::TestContext,
        value::{BasicBlock, Function},
    };

    use crate::gvn::affine::precompute_forms;

    /// Build a single-block function whose root has an `@SP` param (origin = a
    /// register varnode). Returns `(fid, sp_param, sp_reg)`.
    fn sp_function(tc: &mut TestContext) -> (FunctionId, ValueId, VarnodeId) {
        let sp_reg = tc.r0; // stand-in stack-pointer register varnode
        let fid = Function::make(&mut tc.ctx, "f".into()).unwrap().id;
        let root = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x1000, __f)
        };
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        let pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx.block_param_mut(pid).origin = Some(ValueId::Varnode(sp_reg));
        (fid, ValueId::BlockParam(pid), sp_reg)
    }

    #[test]
    fn incoming_sp_param_found_by_origin() {
        let mut tc = TestContext::new();
        let (fid, sp, sp_reg) = sp_function(&mut tc);
        assert_eq!(incoming_sp_param(&tc.ctx, fid, sp_reg), Some(sp));
        // A different register is not the SP param.
        assert_eq!(incoming_sp_param(&tc.ctx, fid, tc.r1), None);
    }

    #[test]
    fn classifies_local_caller_and_aligned() {
        let mut tc = TestContext::new();
        let (fid, sp, _) = sp_function(&mut tc);
        let root = Function::from_id(&tc.ctx, fid).root().unwrap().id;

        let (local, caller_arg, ret_slot, aligned_slot, unrelated) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, root));
            let c8 = b.context_mut().get_const(8, 8).id();
            let neg16 = b.context_mut().get_const((-16i64) as u64, 8).id();
            let local = b.push_sub(sp, c8).id(); // @SP - 8  → below entry SP
            let caller_arg = b.push_add(sp, c8).id(); // @SP + 8  → incoming arg
            let ret_slot = sp; // @SP + 0  → return slot
            let aligned = b.push_bit_and(sp, neg16).id(); // @SP & -16
            let aligned_slot = b.push_add(aligned, c8).id(); // (@SP & -16) + 8
            // A pointer with no relation to @SP.
            let other = b.context_mut().get_const(0x4000, 8).id();
            let unrelated = b.push_add(other, c8).id();
            unsafe { b.dont_finalize() };
            (local, caller_arg, ret_slot, aligned_slot, unrelated)
        };

        let nb = precompute_forms(&tc.ctx, fid);
        let class = |v| frame_class(&tc.ctx, &nb, sp, v);

        // The canonical slot key agrees across representations and is `None`
        // exactly where there is no stable `@SP`-relative offset.
        assert_eq!(frame_offset(&tc.ctx, &nb, sp, local), Some(-8));
        assert_eq!(frame_offset(&tc.ctx, &nb, sp, caller_arg), Some(8));
        assert_eq!(frame_offset(&tc.ctx, &nb, sp, ret_slot), Some(0));
        assert_eq!(
            frame_offset(&tc.ctx, &nb, sp, aligned_slot),
            None,
            "a realigned base has no stable @SP-relative offset"
        );
        assert_eq!(frame_offset(&tc.ctx, &nb, sp, unrelated), None);

        assert_eq!(class(local), Some(FrameClass::Local), "@SP - 8 is a local");
        assert_eq!(
            class(caller_arg),
            Some(FrameClass::CallerFrame),
            "@SP + 8 is an incoming arg"
        );
        assert_eq!(
            class(ret_slot),
            Some(FrameClass::CallerFrame),
            "@SP + 0 is the return slot"
        );
        assert_eq!(
            class(aligned_slot),
            Some(FrameClass::Local),
            "a realigned slot is always a local"
        );
        assert_eq!(class(unrelated), None, "a non-@SP pointer is unclassified");
    }

    /// A cascade of `sub rsp,k; and rsp,-N` realignments — the shape a frame with
    /// several over-aligned local buffers produces — stays anchored to `@SP`, so
    /// slots built on the innermost aligned base are still locals.
    #[test]
    fn classifies_cascaded_realignment_as_local() {
        let mut tc = TestContext::new();
        let (fid, sp, _) = sp_function(&mut tc);
        let root = Function::from_id(&tc.ctx, fid).root().unwrap().id;

        let slot = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, root));
            let neg8 = b.context_mut().get_const((-8i64) as u64, 8).id();
            let k10 = b.context_mut().get_const(0x10, 8).id();
            let k270 = b.context_mut().get_const(0x270, 8).id();
            let k658 = b.context_mut().get_const(0x658, 8).id();
            // ((@SP - 0x10) & -8)
            let s1 = b.push_sub(sp, k10).id();
            let a1 = b.push_bit_and(s1, neg8).id();
            // (((… ) - 0x270) & -8)
            let s2 = b.push_sub(a1, k270).id();
            let a2 = b.push_bit_and(s2, neg8).id();
            // a slot in the innermost realigned frame: a2 - 0x658
            let slot = b.push_sub(a2, k658).id();
            unsafe { b.dont_finalize() };
            slot
        };

        let nb = precompute_forms(&tc.ctx, fid);
        // A cascaded base has no stable `@SP`-relative offset…
        assert_eq!(frame_offset(&tc.ctx, &nb, sp, slot), None);
        // …but is still classified as an own-frame local.
        assert_eq!(
            frame_class(&tc.ctx, &nb, sp, slot),
            Some(FrameClass::Local),
            "a slot on a cascaded realignment of @SP is a local"
        );
    }

    /// A mask applied to an address *above* entry `@SP` must NOT be a local: a
    /// round-down mask can leave it at/above `@SP`, in the caller's frame. Guards
    /// the downward-anchor soundness condition of [`is_aligned_sp`].
    #[test]
    fn rejects_realignment_above_entry_sp() {
        let mut tc = TestContext::new();
        let (fid, sp, _) = sp_function(&mut tc);
        let root = Function::from_id(&tc.ctx, fid).root().unwrap().id;

        let (aligned, slot) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, root));
            let neg8 = b.context_mut().get_const((-8i64) as u64, 8).id();
            let k10 = b.context_mut().get_const(0x10, 8).id();
            let k20 = b.context_mut().get_const(0x20, 8).id();
            // (@SP + 0x10) & -8  — anchored through a *positive* offset.
            let up = b.push_add(sp, k10).id();
            let aligned = b.push_bit_and(up, neg8).id();
            let slot = b.push_sub(aligned, k20).id();
            unsafe { b.dont_finalize() };
            (aligned, slot)
        };

        let nb = precompute_forms(&tc.ctx, fid);
        assert_eq!(
            frame_class(&tc.ctx, &nb, sp, aligned),
            None,
            "(@SP + k) & -mask may land in the caller frame — not a local"
        );
        assert_eq!(
            frame_class(&tc.ctx, &nb, sp, slot),
            None,
            "a slot on an upward-anchored realignment is not a local"
        );
    }
}
