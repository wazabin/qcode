//! Frame-relative classification of stack pointers.
//!
//! The stack pointer is a *normal register*: nothing distinguishes it in the IR,
//! and its entry value is whatever value the two general register-lowering
//! mechanisms give it (see [`entry_sp_value`]). A stack address is then plain
//! affine arithmetic on that value:
//!   * `@SP ± k`           — a fixed frame slot.
//!   * `(@SP & -mask) ± k` — a frame realigned by `and rsp, -N`.
//!
//! [`frame_class`] recognises both and reports whether a pointer lands in this
//! function's own locals or in the caller's frame. It is the representation-
//! agnostic replacement for matching `StackAddress` literals directly, and the
//! foundation for the frame-freshness alias rule.

use qcode::value::{
    FunctionId, FunctionRef, QCodeView, ValueId, Varnode, VarnodeId,
    insn::{Load, Mnemonic, Store},
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

/// The value that holds `fid`'s **entry stack pointer** — `@SP`, the root of every
/// frame-relative judgement in the crate. This is the single seam every consumer
/// resolves it through; there is no distinguished "SP concept" beyond it.
///
/// The stack pointer is lowered like any other register, and the two general
/// register-lowering mechanisms give it two possible shapes:
///
/// 1. **A root block param** whose `origin` is `sp_reg` — the by-value interface
///    input `argpromote`'s register channel materializes for a `pure_reg`
///    function (`param[i] ↔ Call.args[i]`), seeded at entry with
///    `store(SP, param)`. Preferred when present: every read of the register has
///    been rewritten onto it.
/// 2. **The root entry `load(SP)`** — `mem2reg`'s root lowering for a register
///    live-in to the root block (the mechanism used for every non-interface
///    register). The load reads the register cell before anything in the body
///    writes it, so it *is* the incoming value.
///
/// Shape 2 is accepted only when the root has no predecessors. A root with a back
/// edge re-executes its entry load, and mem2reg stores the loop-carried value into
/// the register cell on that edge, so on re-entry the load no longer reads the
/// caller's stack pointer.
///
/// `None` when neither shape is present (e.g. an external, or a function whose
/// first register-space write precedes any read of the stack pointer), which
/// leaves every frame-relative rule inert for that function.
pub(crate) fn entry_sp_value<'a, 'str: 'a>(
    host: impl qcode::value::QCodeView<'a, 'str>,
    fid: FunctionId,
    sp_reg: VarnodeId,
) -> Option<ValueId> {
    let root = FunctionRef::new(host, fid).root()?;
    if let Some(param) = root
        .params()
        .find(|p| p.origin() == Some(ValueId::Varnode(sp_reg)))
    {
        return Some(param.id());
    }
    if root.predecessors().next().is_some() {
        return None;
    }
    let sp = Varnode::from_id(host.shared(), sp_reg);
    let (sp_space, sp_size) = (sp.space().id, sp.size());
    for insn in root.iter() {
        let func = insn.id.func;
        match insn.mnemonic() {
            // The entry read of the whole register: this is the incoming value.
            Mnemonic::Load(Load { space, ptr, size })
                if *space == sp_space
                    && ptr.qualify(func) == ValueId::Varnode(sp_reg)
                    && *size == sp_size =>
            {
                return Some(ValueId::Instruction(insn.id));
            }
            // Reads of other cells, and writes to any *other* space (a temp spill,
            // a ram store), leave the register file untouched.
            Mnemonic::Load(_) => {}
            Mnemonic::Store(Store { space, .. }) if *space != sp_space => {}
            // Pure value computation: no memory effect at all.
            Mnemonic::Binop(_)
            | Mnemonic::Unop(_)
            | Mnemonic::Range(_)
            | Mnemonic::Zext(_)
            | Mnemonic::Sext(_)
            | Mnemonic::IntToFloat(_)
            | Mnemonic::FloatToFloat(_)
            | Mnemonic::FloatToInt(_)
            | Mnemonic::IsFloatNaN(_)
            | Mnemonic::PopCount(_)
            | Mnemonic::LzCount(_)
            | Mnemonic::Carry(_)
            | Mnemonic::SCarry(_)
            | Mnemonic::SBorrow(_)
            | Mnemonic::Intrinsic(_)
            | Mnemonic::Tuple(_)
            | Mnemonic::Extract(_)
            | Mnemonic::Gep(_) => {}
            // Anything else (a register-space store, a call, an opaque p-code op,
            // a terminator) may have already written the stack-pointer cell.
            _ => return None,
        }
    }
    None
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
///
/// DEBT(sp-normalization): this hand-rolled cascade walk is a special case of
/// generic masked-value-range reasoning — "what interval can `x & -2^k` occupy
/// given the interval of `x`" — which would decide the same question for any
/// base without an `@SP` anchor, a depth limit, or a bespoke non-positive-offset
/// side condition. The soundness argument above is the range argument, written
/// out by hand for one register.
fn is_aligned_sp<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    numbering: &Numbering,
    entry_sp: ValueId,
    base: ValueId,
    depth: u32,
) -> bool {
    if depth == 0 {
        return false;
    }
    let Some(term) = numbering.alignment_base(base) else {
        return false;
    };
    // Peel the affine offset of the value being aligned; it must go downward.
    let (inner, off) = numbering.base_offset(host, term).unwrap_or((term, 0));
    if off > 0 {
        return false;
    }
    inner == entry_sp || is_aligned_sp(host, numbering, entry_sp, inner, depth - 1)
}

/// The signed byte offset of `v` from the entry stack pointer `@SP`, when `v` is
/// an `@SP`-rooted stack address. Every representation of the same slot yields the
/// same offset, so this is the **canonical slot key** and stable slot identity.
///
/// Returns `None` for a realigned (`@SP & -mask`) base — which has no stable
/// `@SP`-relative offset — and for any non-stack pointer.
pub(crate) fn frame_offset<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    numbering: &Numbering,
    entry_sp: ValueId,
    v: ValueId,
) -> Option<i64> {
    // Affine `@SP ± k` (the bare param decomposes to itself at offset 0).
    let (base, off) = numbering.base_offset(host, v).unwrap_or((v, 0));
    (base == entry_sp).then_some(off)
}

/// Classify pointer `v` against the frame whose incoming stack pointer is
/// `entry_sp`, decomposing `@SP ± k` through `numbering`. Returns `None` when `v`
/// is not stack-pointer-rooted.
pub(crate) fn frame_class<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    numbering: &Numbering,
    entry_sp: ValueId,
    v: ValueId,
) -> Option<FrameClass> {
    // An `@SP`-relative slot: classify by the sign of its offset.
    if let Some(off) = frame_offset(host, numbering, entry_sp, v) {
        return Some(by_sign(off));
    }
    // A realigned frame base (`@SP & -mask`, possibly cascaded): everything offset
    // from it is a local — incoming args never flow through the alignment mask.
    let (base, _) = numbering.base_offset(host, v).unwrap_or((v, 0));
    is_aligned_sp(host, numbering, entry_sp, base, ALIGN_CASCADE_LIMIT).then_some(FrameClass::Local)
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
        testing::TestContext,
        value::{BasicBlock, FunctionBody, Value},
    };

    use crate::gvn::affine::precompute_forms;

    /// Build a single-block function whose root has an `@SP` param (origin = a
    /// register varnode). Returns `(fid, entry_sp, sp_reg)`.
    fn sp_function(tc: &mut TestContext) -> (FunctionId, ValueId, VarnodeId) {
        let sp_reg = tc.r0; // stand-in stack-pointer register varnode
        let fid = FunctionBody::make(&mut tc.ctx, "f".into()).unwrap().id;
        let root = { tc.ctx.get_or_make_block(0x1000, fid) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        let pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx
            .block_param_mut(pid)
            .set_origin_id(ValueId::Varnode(sp_reg).localize(pid.func));
        (fid, ValueId::BlockParam(pid), sp_reg)
    }

    /// Build a param-less single-block function — the shape of a function whose
    /// registers were never lifted into a call interface. Returns `(fid, root)`.
    fn bare_function(tc: &mut TestContext) -> (FunctionId, qcode::value::BlockId) {
        let fid = FunctionBody::make(&mut tc.ctx, "f".into()).unwrap().id;
        let root = tc.ctx.get_or_make_block(0x1000, fid);
        let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
        f.set_root(root).unwrap();
        f.add_block(root);
        (fid, root)
    }

    #[test]
    fn entry_sp_value_found_by_origin() {
        let mut tc = TestContext::new();
        let (fid, sp, sp_reg) = sp_function(&mut tc);
        assert_eq!(
            entry_sp_value(qcode::value::ModuleView::new(&tc.ctx), fid, sp_reg),
            Some(sp)
        );
        // A different register is not the SP param.
        assert_eq!(
            entry_sp_value(qcode::value::ModuleView::new(&tc.ctx), fid, tc.r1),
            None
        );
    }

    /// Shape 2: no interface param, so the entry stack pointer is the root entry
    /// `load(SP)` mem2reg's root lowering leaves at the top of the block. Reads of
    /// other registers and writes to other spaces before it are transparent.
    #[test]
    fn entry_sp_value_found_as_root_entry_load() {
        let mut tc = TestContext::new();
        let (fid, root) = bare_function(&mut tc);
        let (sp_reg, reg_space, ram) = (tc.r0, tc.reg_space, tc.ctx.shared.default_space);

        let sp_load = {
            let mut b = tc.ctx.builder(root);
            // A read of another register, and a spill of it into ram: neither can
            // have written the stack-pointer cell.
            let other = b
                .push_load::<false>(ValueId::Varnode(tc.r1), 8, reg_space)
                .id();
            let addr = b.shr().get_const(0x4000, 8);
            b.push_store(other, addr, ram);
            b.push_load::<false>(ValueId::Varnode(sp_reg), 8, reg_space)
                .id()
        };

        assert_eq!(
            entry_sp_value(qcode::value::ModuleView::new(&tc.ctx), fid, sp_reg),
            Some(sp_load),
        );
    }

    /// A read of the stack pointer *after* something wrote register space is not
    /// the incoming value — the write may have covered the stack-pointer cell.
    #[test]
    fn entry_sp_value_rejects_load_after_a_register_write() {
        let mut tc = TestContext::new();
        let (fid, root) = bare_function(&mut tc);
        let (sp_reg, reg_space) = (tc.r0, tc.reg_space);

        {
            let mut b = tc.ctx.builder(root);
            let zero = b.shr().get_const(0, 8);
            b.push_store(zero, ValueId::Varnode(sp_reg), reg_space);
            b.push_load::<false>(ValueId::Varnode(sp_reg), 8, reg_space);
        }

        assert_eq!(
            entry_sp_value(qcode::value::ModuleView::new(&tc.ctx), fid, sp_reg),
            None,
        );
    }

    /// A root with a predecessor re-executes its entry load, reading whatever the
    /// back edge left in the register cell — not the caller's stack pointer.
    #[test]
    fn entry_sp_value_rejects_root_entry_load_under_a_back_edge() {
        let mut tc = TestContext::new();
        let (fid, root) = bare_function(&mut tc);
        let (sp_reg, reg_space) = (tc.r0, tc.reg_space);

        let body = tc.ctx.get_or_make_block(0x2000, fid);
        FunctionBody::from_id_mut(&mut tc.ctx, fid).add_block(body);
        {
            let mut b = tc.ctx.builder(root);
            b.push_load::<false>(ValueId::Varnode(sp_reg), 8, reg_space);
            b.push_branch(body);
        }
        {
            let mut b = tc.ctx.builder(body);
            b.push_branch(root);
        }

        assert_eq!(
            entry_sp_value(qcode::value::ModuleView::new(&tc.ctx), fid, sp_reg),
            None,
        );
    }

    #[test]
    fn classifies_local_caller_and_aligned() {
        let mut tc = TestContext::new();
        let (fid, sp, _) = sp_function(&mut tc);
        let root = FunctionBody::from_id(&tc.ctx, fid).root().unwrap().id;

        let (local, caller_arg, ret_slot, aligned_slot, unrelated) = {
            let mut b = tc.ctx.builder(root);
            let c8 = b.shr().get_const(8, 8);
            let neg16 = b.shr().get_const((-16i64) as u64, 8);
            let local = b.push_sub(sp, c8).id(); // @SP - 8  → below entry SP
            let caller_arg = b.push_add(sp, c8).id(); // @SP + 8  → incoming arg
            let ret_slot = sp; // @SP + 0  → return slot
            let aligned = b.push_bit_and(sp, neg16).id(); // @SP & -16
            let aligned_slot = b.push_add(aligned, c8).id(); // (@SP & -16) + 8
            // A pointer with no relation to @SP.
            let other = b.shr().get_const(0x4000, 8);
            let unrelated = b.push_add(other, c8).id();
            (local, caller_arg, ret_slot, aligned_slot, unrelated)
        };

        let nb = precompute_forms(qcode::value::ModuleView::new(&tc.ctx), fid);
        let view = qcode::value::ModuleView::new(&tc.ctx);
        let class = |v| frame_class(view, &nb, sp, v);

        // The canonical slot key agrees across representations and is `None`
        // exactly where there is no stable `@SP`-relative offset.
        assert_eq!(frame_offset(view, &nb, sp, local), Some(-8));
        assert_eq!(frame_offset(view, &nb, sp, caller_arg), Some(8));
        assert_eq!(frame_offset(view, &nb, sp, ret_slot), Some(0));
        assert_eq!(
            frame_offset(view, &nb, sp, aligned_slot),
            None,
            "a realigned base has no stable @SP-relative offset"
        );
        assert_eq!(frame_offset(view, &nb, sp, unrelated), None);

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
        let root = FunctionBody::from_id(&tc.ctx, fid).root().unwrap().id;

        let slot = {
            let mut b = tc.ctx.builder(root);
            let neg8 = b.shr().get_const((-8i64) as u64, 8);
            let k10 = b.shr().get_const(0x10, 8);
            let k270 = b.shr().get_const(0x270, 8);
            let k658 = b.shr().get_const(0x658, 8);
            // ((@SP - 0x10) & -8)
            let s1 = b.push_sub(sp, k10).id();
            let a1 = b.push_bit_and(s1, neg8).id();
            // (((… ) - 0x270) & -8)
            let s2 = b.push_sub(a1, k270).id();
            let a2 = b.push_bit_and(s2, neg8).id();
            // a slot in the innermost realigned frame: a2 - 0x658

            b.push_sub(a2, k658).id()
        };

        let nb = precompute_forms(qcode::value::ModuleView::new(&tc.ctx), fid);
        // A cascaded base has no stable `@SP`-relative offset…
        let view = qcode::value::ModuleView::new(&tc.ctx);
        assert_eq!(frame_offset(view, &nb, sp, slot), None);
        // …but is still classified as an own-frame local.
        assert_eq!(
            frame_class(view, &nb, sp, slot),
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
        let root = FunctionBody::from_id(&tc.ctx, fid).root().unwrap().id;

        let (aligned, slot) = {
            let mut b = tc.ctx.builder(root);
            let neg8 = b.shr().get_const((-8i64) as u64, 8);
            let k10 = b.shr().get_const(0x10, 8);
            let k20 = b.shr().get_const(0x20, 8);
            // (@SP + 0x10) & -8  — anchored through a *positive* offset.
            let up = b.push_add(sp, k10).id();
            let aligned = b.push_bit_and(up, neg8).id();
            let slot = b.push_sub(aligned, k20).id();
            (aligned, slot)
        };

        let nb = precompute_forms(qcode::value::ModuleView::new(&tc.ctx), fid);
        assert_eq!(
            frame_class(qcode::value::ModuleView::new(&tc.ctx), &nb, sp, aligned),
            None,
            "(@SP + k) & -mask may land in the caller frame — not a local"
        );
        assert_eq!(
            frame_class(qcode::value::ModuleView::new(&tc.ctx), &nb, sp, slot),
            None,
            "a slot on an upward-anchored realignment is not a local"
        );
    }
}
