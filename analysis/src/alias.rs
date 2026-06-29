use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use qcode::{
    assumption::Proposition,
    context::Context,
    space::SpaceId,
    value::{
        BlockId, BlockParam, Function, FunctionId, Instruction, ValueId, ValueRef, VarnodeId,
        insn::{Binop, IntBinop, Mnemonic},
    },
};

use crate::gvn::affine::{Numbering, precompute_forms};
use crate::stack::frame::{FrameClass, frame_class, incoming_sp_param};

mod anderson;
mod simple;

pub use anderson::alias_analysis;
pub use simple::RegisterBase;

/// Per-function frame-freshness context, precomputed when an [`AliasResult`] is
/// built with [`AliasResult::with_frame_freshness`]. `None` leaves the rule inert
/// (hand-built results and callers without a stack-pointer register).
pub(crate) struct FrameInfo {
    /// The function this frame belongs to — the key for the
    /// [`Proposition::ArgsDisjointFromCallerFrame`] truth consulted by the
    /// assumed caller-frame rule in [`AliasResult::provably_disjoint`].
    fid: FunctionId,
    /// This function's root (entry) block — a pointer that peels to one of its
    /// params is an *incoming* pointer from the caller.
    root_block: Option<BlockId>,
    /// The incoming stack-pointer param `@SP`. Excluded from "input-derived": a
    /// pointer rooted at `@SP` is a frame pointer, not a caller-supplied data
    /// pointer, so the rule must not treat `@SP ± k` as an incoming pointer.
    sp_param: ValueId,
    /// Every value classified as an own-frame local (`@SP`-rooted slot below the
    /// entry stack pointer, or a realigned-frame slot) for this function.
    own_frame_locals: HashSet<ValueId>,
    /// Every value classified as a *caller-frame* slot (`@SP + k`, `k ≥ 0`: the
    /// return-address slot and incoming stack arguments). Used only by the
    /// assumed `ArgsDisjointFromCallerFrame` rule.
    caller_frame_slots: HashSet<ValueId>,
    /// The function's affine numbering, kept so [`AliasResult::provably_disjoint`]
    /// can classify an *arbitrary* pointer (`@SP ± k`, `@glob + k`, …) on demand —
    /// not just the precomputed own/caller-frame instruction sets — which the
    /// stack-vs-global rule needs for literal and globalized-param pointers.
    numbering: Numbering,
}

/// Abstract node in the alias graph.
///
/// `Unknown` is the top element: any value joined to it may alias everything.
/// `Id(n)` is a concrete node allocated during analysis.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeId {
    Unknown,
    Id(usize),
}

pub struct AliasResult {
    pub(crate) value_to_root: HashMap<ValueId, NodeId>,
    /// Exact byte intervals for pointer values whose location could be
    /// statically resolved: `(space_id, byte_start, byte_end)`.
    /// Populated by location-aware analyses (e.g. `simple`); empty otherwise.
    pub(crate) value_to_interval: HashMap<ValueId, (SpaceId, u64, u64)>,
    /// Frame-freshness context, when populated (see [`AliasResult::provably_disjoint`]).
    pub(crate) frame: Option<FrameInfo>,
}

impl AliasResult {
    /// The alias equivalence-class of `a`, or `None` if `a` was not
    /// involved in any constraint during analysis.
    pub fn alias_class(&self, a: ValueId) -> Option<NodeId> {
        self.value_to_root.get(&a).copied()
    }

    /// The exact byte interval `(space, start, end)` of `a`, when a
    /// location-aware analysis (e.g. [`AliasResult::simple`]) could statically
    /// resolve it. Returns `None` for values with no precise location.
    pub fn interval(&self, a: ValueId) -> Option<(SpaceId, u64, u64)> {
        self.value_to_interval.get(&a).copied()
    }

    /// Conservative must-alias query.
    ///
    /// Returns `true` only when `a` and `b` are guaranteed to refer to the
    /// exact same byte range (same address space, start, and end). Defaults
    /// to `false` when precise location information is unavailable.
    pub fn must_alias(&self, a: ValueId, b: ValueId) -> bool {
        self.interval(a)
            .zip(self.interval(b))
            .is_some_and(|(ia, ib)| ia == ib)
    }

    /// Returns true if `outer`'s byte interval fully contains `inner`'s (same
    /// space, `start_outer <= start_inner`, `end_outer >= end_inner`). Falls
    /// back to `false` when precise interval information is unavailable for
    /// either value.
    pub fn covers(&self, inner: ValueId, outer: ValueId) -> bool {
        self.interval(inner).zip(self.interval(outer)).is_some_and(
            |((si, start_i, end_i), (so, start_o, end_o))| {
                si == so && start_o <= start_i && end_o >= end_i
            },
        )
    }

    /// Returns all other tracked pointer values whose interval is contained
    /// within `ptr`'s interval in the same space (including values covering
    /// exactly the same range under a different `ValueId`), as
    /// `(sub_ptr, byte_offset_within_ptr, sub_size)`, sorted by offset then
    /// size for deterministic output.
    pub fn sub_intervals_of(&self, ptr: ValueId) -> Vec<(ValueId, usize, usize)> {
        let Some(&(ptr_space, ptr_start, ptr_end)) = self.value_to_interval.get(&ptr) else {
            return Vec::new();
        };
        let mut subs: Vec<_> = self
            .value_to_interval
            .iter()
            .filter_map(|(&other, &(space, start, end))| {
                if other == ptr || space != ptr_space {
                    return None;
                }
                if start >= ptr_start && end <= ptr_end {
                    let byte_offset = usize::try_from(start - ptr_start).ok()?;
                    let sub_size = usize::try_from(end - start).ok()?;
                    Some((other, byte_offset, sub_size))
                } else {
                    None
                }
            })
            .collect();
        subs.sort_unstable_by_key(|&(_, off, size)| (off, size));
        subs
    }

    /// Conservative may-alias query.
    ///
    /// Returns `true` if `a` and `b` share a class, or if either is
    /// `NodeId::Unknown`. Returns `false` if either value was never
    /// involved in any constraint (isolated - no alias relationship).
    pub fn may_alias(&self, ctx: &Context, a: ValueId, b: ValueId) -> bool {
        // If a and b don't share the same address space they can't alias.
        // Note this deliberately overrides `NodeId::Unknown` below: aliasing
        // means "same storage location", and values in different spaces can
        // never occupy the same location, however unknown their class is.
        let a_space = ValueRef::new(a, ctx).space().map(|s| s.id);
        let b_space = ValueRef::new(b, ctx).space().map(|s| s.id);
        if let (Some(sa), Some(sb)) = (a_space, b_space)
            && sa != sb
        {
            return false;
        }

        // Frame freshness: an own-frame local never aliases an incoming pointer
        // (sound), and a caller-frame slot is disjoint from one under the recorded
        // `ArgsDisjointFromCallerFrame` assumption (see `provably_disjoint`). Answer
        // "no" before the alias graph collapses both partitions to `Unknown`.
        if self.provably_disjoint(ctx, a, b) {
            return false;
        }

        match (self.alias_class(a), self.alias_class(b)) {
            (None, _) | (_, None) => false,
            (Some(NodeId::Unknown), _) | (_, Some(NodeId::Unknown)) => true,
            (Some(ra), Some(rb)) => ra == rb,
        }
    }

    /// Frame-freshness disjointness between a stack slot and an incoming pointer.
    ///
    /// Two rules, both querying one own/caller-frame slot against an
    /// *input-derived* value (one that peels to a non-`@SP` entry-block param):
    ///
    /// 1. **Own-frame (sound).** An own-frame local ([`FrameClass::Local`], an
    ///    `@SP`-rooted slot below entry SP) never aliases an incoming pointer: the
    ///    callee's frame is younger than anything the caller could already name. The
    ///    alias graph cannot express this — both partitions escape and collapse to
    ///    [`NodeId::Unknown`] — so this rule answers before that.
    ///
    /// 2. **Caller-frame (assumed).** A caller-frame slot ([`FrameClass::CallerFrame`],
    ///    `@SP + k`, `k ≥ 0`: the return-address slot and incoming stack-argument
    ///    slots) is taken disjoint from an incoming pointer *iff*
    ///    [`Proposition::ArgsDisjointFromCallerFrame`] is recorded true for this
    ///    function. This is **not** statically sound (a caller could pass the address
    ///    of one of its outgoing-argument slots), so it fires only under the recorded
    ///    assumption, which a verifier discharges and the checkpoint+replay driver
    ///    rolls back if contradicted. It unblocks forwarding a caller-frame slot load
    ///    across a store through an incoming pointer (the spilled-pointer reload).
    ///
    /// Inert (`false`) unless the result was built with
    /// [`AliasResult::with_frame_freshness`].
    pub fn provably_disjoint(&self, ctx: &Context, a: ValueId, b: ValueId) -> bool {
        let Some(frame) = &self.frame else {
            return false;
        };
        // Rule 1: own-frame local ⊥ incoming pointer (sound). Strict input-derivation
        // (no load peeling) keeps this rule a theorem, not an assumption.
        if (frame.own_frame_locals.contains(&a) && self.is_input_derived(ctx, b, false))
            || (frame.own_frame_locals.contains(&b) && self.is_input_derived(ctx, a, false))
        {
            return true;
        }
        // Rule 1b: stack ⊥ static-global. A runtime `@SP`-rooted stack address never
        // coincides with a fixed absolute address — a global literal `0x…`, or a
        // *globalized-global* parameter (`@glob_<addr>`, whose `origin` records the
        // literal it replaced). The live stack and the static image occupy disjoint
        // regions of the address space, so the two can never name the same byte.
        // Unlike rule 2 this needs no recorded assumption: it holds in the
        // decompilation memory model the same way frame freshness (rule 1) does.
        if (self.is_stack_rooted(ctx, a) && self.is_global_static(ctx, b))
            || (self.is_stack_rooted(ctx, b) && self.is_global_static(ctx, a))
        {
            return true;
        }
        // Rule 1c: a globalized-global slot ⊥ a pointer *loaded from it*. `@glob_<addr>`
        // holds a pointer; a store through `load(@glob) + …` addresses the buffer
        // behind the pointer, not the pointer slot itself. This is the global-slot
        // analogue of the caller-frame spilled-pointer reload (rule 2) and rides the
        // same [`Proposition::LoadedPointerDisjointFromSlot`] assumption — not
        // statically sound (a self-referential global `*pp == &pp` would break it),
        // recorded `Assumed`, caught by checkpoint+replay. It lets the in-loop reload
        // of a globalized pointer forward across the store that writes through it,
        // which `argpromote` needs to region-promote the buffer behind the global.
        //
        // We require an actual `load` on the path (`peels_to_load`), not merely
        // `is_input_derived`: a `@glob` slot is itself a root param, so the looser
        // test would wrongly call it disjoint from itself.
        let loaded_ptr_disjoint = ctx
            .truth(Proposition::LoadedPointerDisjointFromSlot(frame.fid))
            .is_some_and(|t| t.value);
        if loaded_ptr_disjoint
            && ((self.is_global_static(ctx, a) && self.peels_to_load(ctx, b))
                || (self.is_global_static(ctx, b) && self.peels_to_load(ctx, a)))
        {
            return true;
        }
        // Rule 2: caller-frame slot ⊥ incoming pointer, under the recorded
        // assumption (validated by replay).
        //
        // Memory forwarding keys cells by their *affine base*, and every
        // `@SP ± k` slot collapses to the bare `@SP` param (its base term) — so the
        // value that actually reaches this query for a stack cell is `sp_param`,
        // not the `@SP + k` instruction. We therefore also treat `sp_param` itself
        // as a caller-frame base: `disjoint(incoming_ptr, @SP)` claims the pointer
        // misses the *whole* own frame, which is the own-frame locals (sound by
        // rule 1) plus the caller-frame slots (this assumption) — both hold, so the
        // claim is justified. `caller_frame_slots` still covers the non-affine path
        // (e.g. a realigned frame) where the instruction value is passed directly.
        let caller_frame_assumed = ctx
            .truth(Proposition::ArgsDisjointFromCallerFrame(frame.fid))
            .is_some_and(|t| t.value);
        // Under [`Proposition::LoadedPointerDisjointFromSlot`], a value *loaded* from
        // a slot is treated as carrying incoming-pointer provenance (`peel_loads`):
        // the spilled-pointer reload idiom, where a buffer pointer is spilled to a
        // caller-frame slot and reloaded inside a loop. Without it the reload reads
        // as opaque and blocks forwarding (the deref's base is a `load`, so the loop
        // store cannot be shown disjoint from the slot). The frame footprint a
        // loaded pointer is taken disjoint from is the same one rule 2 already
        // assumes for a direct incoming pointer.
        let peel_loads = ctx
            .truth(Proposition::LoadedPointerDisjointFromSlot(frame.fid))
            .is_some_and(|t| t.value);
        let is_caller_frame =
            |v: ValueId| v == frame.sp_param || frame.caller_frame_slots.contains(&v);
        caller_frame_assumed
            && ((is_caller_frame(a) && self.is_input_derived(ctx, b, peel_loads))
                || (is_caller_frame(b) && self.is_input_derived(ctx, a, peel_loads)))
    }

    /// Whether `v` is an own-frame local under the populated frame-freshness
    /// context (see [`AliasResult::provably_disjoint`]).
    pub fn is_own_frame_local(&self, v: ValueId) -> bool {
        self.frame
            .as_ref()
            .is_some_and(|f| f.own_frame_locals.contains(&v))
    }

    /// Whether `v` is a stack-pointer-rooted address (`@SP ± k`, or a realigned
    /// `(@SP & -mask) ± k`) — i.e. it names a slot in this function's live stack
    /// frame. Inert without frame-freshness context. Used by the stack-vs-global
    /// disjointness rule.
    fn is_stack_rooted(&self, ctx: &Context, v: ValueId) -> bool {
        self.frame
            .as_ref()
            .is_some_and(|f| frame_class(ctx, &f.numbering, f.sp_param, v).is_some())
    }

    /// Whether `v` is a fixed static/global address: a literal absolute address,
    /// a *globalized-global* parameter (`@glob_<addr>`, whose `origin` is the
    /// address literal it replaced — see `argpromote::globals`), or affine
    /// arithmetic (`base + k`) over either. Such an address lives in the static
    /// image, never in the live stack.
    fn is_global_static(&self, ctx: &Context, v: ValueId) -> bool {
        match v {
            ValueId::Literal(_) => true,
            ValueId::BlockParam(pid) => {
                matches!(
                    BlockParam::from_id(ctx, pid).origin(),
                    Some(ValueId::Literal(_))
                )
            }
            _ => self
                .frame
                .as_ref()
                .and_then(|f| f.numbering.base_offset(v))
                .is_some_and(|(base, _)| base != v && self.is_global_static(ctx, base)),
        }
    }

    /// Whether `v` is *input-derived*: it peels — through address arithmetic
    /// (`add`/`sub` of an offset) and width casts (`zext`/`sext`/`range`) — to a
    /// parameter of this function's root block, i.e. a value that entered from the
    /// caller. The `add`/`sub` peel bails if either operand is itself an own-frame
    /// local, so a nonsensical `local + arg` mix is never read as input.
    /// With `peel_loads`, a `load(…)` also counts as input-derived (a reloaded
    /// spilled pointer carries incoming provenance) — see rule 2 / the
    /// [`Proposition::LoadedPointerDisjointFromSlot`] assumption.
    fn is_input_derived(&self, ctx: &Context, v: ValueId, peel_loads: bool) -> bool {
        let Some(frame) = &self.frame else {
            return false;
        };
        self.is_input_derived_rec(ctx, frame, v, peel_loads, &mut HashSet::default())
    }

    /// Whether `v` is a pointer obtained from the *contents of a slot* — through
    /// pointer arithmetic (`add`/`sub`) and width casts — i.e. a buffer pointer read
    /// out of a slot and then indexed. Two provenances qualify:
    ///   * an actual `load(…)` on the path (the reloaded pointer), or
    ///   * a by-value snapshot param whose `origin` is a global-static slot param —
    ///     `argpromote`'s partial-promotion materialization of `*slot` (see
    ///     `apply_partial`), which equals the loaded pointer by construction.
    ///
    /// Unlike [`is_input_derived`] (which stops at any root param), this requires one
    /// of those load provenances, so a bare slot address is never mistaken for a
    /// pointer loaded *from* it. Used by the globalized-global slot rule in
    /// [`AliasResult::provably_disjoint`].
    fn peels_to_load(&self, ctx: &Context, v: ValueId) -> bool {
        self.peels_to_load_rec(ctx, v, &mut HashSet::default())
    }

    fn peels_to_load_rec(&self, ctx: &Context, v: ValueId, seen: &mut HashSet<ValueId>) -> bool {
        if !seen.insert(v) {
            return false;
        }
        match v {
            // A partial-promotion snapshot of `*slot` (origin = the global slot param)
            // — materialized loaded pointer. The `BlockParam` origin distinguishes it
            // from the slot itself (whose origin is the address *literal*).
            ValueId::BlockParam(pid) => matches!(
                BlockParam::from_id(ctx, pid).origin(),
                Some(o @ ValueId::BlockParam(_)) if self.is_global_static(ctx, o)
            ),
            ValueId::Instruction(id) => match Instruction::from_id(ctx, id).mnemonic() {
                Mnemonic::Load(_) => true,
                Mnemonic::Binop(b) if matches!(b.op, Binop::Int(IntBinop::Add | IntBinop::Sub)) => {
                    self.peels_to_load_rec(ctx, b.lhs, seen)
                        || self.peels_to_load_rec(ctx, b.rhs, seen)
                }
                Mnemonic::Zext(z) => self.peels_to_load_rec(ctx, z.src, seen),
                Mnemonic::Sext(s) => self.peels_to_load_rec(ctx, s.src, seen),
                Mnemonic::Range(r) => self.peels_to_load_rec(ctx, r.src, seen),
                _ => false,
            },
            _ => false,
        }
    }

    fn is_input_derived_rec(
        &self,
        ctx: &Context,
        frame: &FrameInfo,
        v: ValueId,
        peel_loads: bool,
        seen: &mut HashSet<ValueId>,
    ) -> bool {
        if !seen.insert(v) {
            return false;
        }
        match v {
            // A root-block param other than `@SP` is a caller-supplied pointer.
            ValueId::BlockParam(id) => {
                v != frame.sp_param
                    && BlockParam::from_id(ctx, id)
                        .parent()
                        .is_some_and(|b| frame.root_block == Some(b.id))
            }
            ValueId::Instruction(id) => match Instruction::from_id(ctx, id).mnemonic() {
                // A reloaded spilled pointer: assumed to carry incoming provenance
                // (gated by the caller via `peel_loads`).
                Mnemonic::Load(_) if peel_loads => true,
                Mnemonic::Binop(b) if matches!(b.op, Binop::Int(IntBinop::Add | IntBinop::Sub)) => {
                    if frame.own_frame_locals.contains(&b.lhs)
                        || frame.own_frame_locals.contains(&b.rhs)
                    {
                        return false;
                    }
                    self.is_input_derived_rec(ctx, frame, b.lhs, peel_loads, seen)
                        || self.is_input_derived_rec(ctx, frame, b.rhs, peel_loads, seen)
                }
                Mnemonic::Zext(z) => self.is_input_derived_rec(ctx, frame, z.src, peel_loads, seen),
                Mnemonic::Sext(s) => self.is_input_derived_rec(ctx, frame, s.src, peel_loads, seen),
                Mnemonic::Range(r) => {
                    self.is_input_derived_rec(ctx, frame, r.src, peel_loads, seen)
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Populate the frame-freshness oracle for function `fid`, given the
    /// stack-pointer register varnode `sp_reg` (resolved from the arch config). A
    /// no-op when `sp_reg` is `None` or the function has no incoming `@SP` param,
    /// leaving [`AliasResult::provably_disjoint`] inert.
    pub fn with_frame_freshness(
        mut self,
        ctx: &Context,
        fid: FunctionId,
        sp_reg: Option<VarnodeId>,
    ) -> Self {
        let Some(sp_reg) = sp_reg else {
            return self;
        };
        let Some(sp) = incoming_sp_param(ctx, fid, sp_reg) else {
            return self;
        };
        let numbering = precompute_forms(ctx, fid);
        let mut own_frame_locals = HashSet::default();
        let mut caller_frame_slots = HashSet::default();
        for block in Function::from_id(ctx, fid).blocks() {
            for insn in block.iter() {
                let v = ValueId::Instruction(insn.id);
                match frame_class(ctx, &numbering, sp, v) {
                    Some(FrameClass::Local) => {
                        own_frame_locals.insert(v);
                    }
                    Some(FrameClass::CallerFrame) => {
                        caller_frame_slots.insert(v);
                    }
                    None => {}
                }
            }
        }
        let root_block = Function::from_id(ctx, fid).root().map(|r| r.id);
        self.frame = Some(FrameInfo {
            fid,
            root_block,
            sp_param: sp,
            own_frame_locals,
            caller_frame_slots,
            numbering,
        });
        self
    }
}

#[cfg(test)]
mod tests {
    use qcode::{
        context::Context,
        space::{Space, SpaceType},
        value::InstructionId,
    };

    use super::*;

    fn value(n: usize) -> ValueId {
        ValueId::Instruction(InstructionId::from(n))
    }

    /// An `AliasResult` with hand-written intervals: value(0) spans [0, 8),
    /// value(1) spans [0, 4), value(2) spans [4, 8), value(3) also spans
    /// [0, 8) (equal to value(0)), and value(4) spans [0, 8) in another space.
    fn interval_fixture() -> AliasResult {
        let mut ctx = Context::new();
        let mut space_a = Space::new(Some("a"), 1, 8);
        space_a.ty = SpaceType::Register;
        let sa = ctx.add_space(space_a);
        let mut space_b = Space::new(Some("b"), 1, 8);
        space_b.ty = SpaceType::Register;
        let sb = ctx.add_space(space_b);

        let value_to_interval = HashMap::from_iter([
            (value(0), (sa, 0, 8)),
            (value(1), (sa, 0, 4)),
            (value(2), (sa, 4, 8)),
            (value(3), (sa, 0, 8)),
            (value(4), (sb, 0, 8)),
        ]);
        AliasResult {
            value_to_root: HashMap::default(),
            value_to_interval,
            frame: None,
        }
    }

    #[test]
    fn must_alias_requires_identical_interval_and_space() {
        let r = interval_fixture();
        assert!(r.must_alias(value(0), value(3)));
        assert!(!r.must_alias(value(0), value(1)), "containment is not must");
        assert!(!r.must_alias(value(0), value(4)), "different space");
        assert!(!r.must_alias(value(0), value(9)), "untracked value");
    }

    #[test]
    fn covers_is_outer_contains_inner() {
        let r = interval_fixture();
        assert!(
            r.covers(value(1), value(0)),
            "outer [0,8) covers inner [0,4)"
        );
        assert!(!r.covers(value(0), value(1)), "inner does not cover outer");
        assert!(r.covers(value(0), value(3)), "equal intervals cover");
        assert!(!r.covers(value(1), value(4)), "different space");
        assert!(!r.covers(value(1), value(9)), "untracked value");
    }

    #[test]
    fn sub_intervals_include_equal_ranges_and_are_sorted() {
        let r = interval_fixture();
        assert_eq!(
            r.sub_intervals_of(value(0)),
            vec![(value(1), 0, 4), (value(3), 0, 8), (value(2), 4, 4)],
            "subs sorted by (offset, size); equal-interval value(3) included; \
             other-space value(4) and value(0) itself excluded"
        );
        assert!(r.sub_intervals_of(value(9)).is_empty(), "untracked value");
    }

    /// Frame freshness: a function's own stack local (`@SP - 8`) is provably
    /// disjoint from an incoming data pointer param and anything offset from it,
    /// but not from `@SP` itself, a caller-frame stack arg (`@SP + 8`), or itself.
    #[test]
    fn frame_freshness_local_disjoint_from_incoming_pointer() {
        use qcode::{
            builder::Builder,
            testing::TestContext,
            value::{BasicBlock, Function},
        };

        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let fid = Function::make(&mut tc.ctx, "f".into()).unwrap().id;
        let root = tc.ctx.get_or_make_block(0x1000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        // `@SP` param (origin = the stack-pointer reg) and a caller data-pointer param.
        let sp_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx.values.block_params[sp_pid].origin = Some(ValueId::Varnode(sp_reg));
        let sp = ValueId::BlockParam(sp_pid);
        let arg_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        let arg = ValueId::BlockParam(arg_pid);

        let (local, caller_arg, arg_plus) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, root));
            let c8 = b.context_mut().get_const(8, 8).id();
            let local = b.push_sub(sp, c8).id(); // @SP - 8  (own-frame local)
            let caller_arg = b.push_add(sp, c8).id(); // @SP + 8  (caller frame)
            let arg_plus = b.push_add(arg, c8).id(); // arg + 8  (input-derived)
            unsafe { b.dont_finalize() };
            (local, caller_arg, arg_plus)
        };

        let r = AliasResult::simple(&tc.ctx).with_frame_freshness(&tc.ctx, fid, Some(sp_reg));

        assert!(
            r.provably_disjoint(&tc.ctx, local, arg),
            "local ⊥ incoming arg"
        );
        assert!(
            r.provably_disjoint(&tc.ctx, arg, local),
            "rule is symmetric"
        );
        assert!(
            r.provably_disjoint(&tc.ctx, local, arg_plus),
            "local ⊥ a pointer offset from the incoming arg"
        );
        assert!(
            !r.provably_disjoint(&tc.ctx, local, local),
            "a local is not disjoint from itself"
        );
        assert!(
            !r.provably_disjoint(&tc.ctx, local, sp),
            "@SP is a frame pointer, not an incoming data pointer"
        );
        assert!(
            !r.provably_disjoint(&tc.ctx, local, caller_arg),
            "a caller-frame stack arg shares the @SP base (handled by offset disjointness)"
        );

        // Inert without the frame-freshness context.
        let plain = AliasResult::simple(&tc.ctx);
        assert!(!plain.provably_disjoint(&tc.ctx, local, arg));
    }

    /// Stack-vs-global rule: an `@SP`-rooted stack slot is disjoint from a static
    /// global address — both a bare literal address and a *globalized-global*
    /// parameter (`@glob_<addr>`, whose `origin` is the address literal).
    #[test]
    fn stack_is_disjoint_from_global_statics() {
        use qcode::{
            builder::Builder,
            testing::TestContext,
            value::{BasicBlock, Function},
        };

        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let fid = Function::make(&mut tc.ctx, "f".into()).unwrap().id;
        let root = tc.ctx.get_or_make_block(0x1000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        let sp_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx.values.block_params[sp_pid].origin = Some(ValueId::Varnode(sp_reg));
        let sp = ValueId::BlockParam(sp_pid);
        // A globalized-global param (origin set to the address literal below).
        let glob_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        let glob = ValueId::BlockParam(glob_pid);

        let (local, caller_arg, glob_plus, lit_addr, glob_addr) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, root));
            let c8 = b.context_mut().get_const(8, 8).id();
            let local = b.push_sub(sp, c8).id(); // @SP - 8  (own-frame local)
            let caller_arg = b.push_add(sp, c8).id(); // @SP + 8  (caller frame)
            let glob_plus = b.push_add(glob, c8).id(); // @glob + 8
            let lit_addr = b.context_mut().get_const(0x401000, 8).id(); // bare global address
            let glob_addr = b.context_mut().get_const(0x454df8, 8).id();
            unsafe { b.dont_finalize() };
            (local, caller_arg, glob_plus, lit_addr, glob_addr)
        };
        tc.ctx.values.block_params[glob_pid].origin = Some(glob_addr);

        let r = AliasResult::simple(&tc.ctx).with_frame_freshness(&tc.ctx, fid, Some(sp_reg));

        assert!(
            r.provably_disjoint(&tc.ctx, local, glob),
            "stack local ⊥ globalized-global param"
        );
        assert!(
            r.provably_disjoint(&tc.ctx, glob, local),
            "rule is symmetric"
        );
        assert!(
            r.provably_disjoint(&tc.ctx, local, lit_addr),
            "stack local ⊥ a bare literal global address"
        );
        assert!(
            r.provably_disjoint(&tc.ctx, local, glob_plus),
            "stack local ⊥ a pointer offset from a globalized global"
        );
        assert!(
            r.provably_disjoint(&tc.ctx, caller_arg, lit_addr),
            "a caller-frame slot is also stack-rooted, so ⊥ a global"
        );
        assert!(
            !r.provably_disjoint(&tc.ctx, lit_addr, glob),
            "two globals are not shown disjoint by this rule"
        );
        assert!(
            !r.provably_disjoint(&tc.ctx, local, caller_arg),
            "two stack pointers are not shown disjoint by this rule"
        );
        assert!(
            !r.may_alias(&tc.ctx, local, glob),
            "may_alias reflects the disjointness"
        );

        // Inert without the frame-freshness context.
        let plain = AliasResult::simple(&tc.ctx);
        assert!(!plain.provably_disjoint(&tc.ctx, local, lit_addr));
    }

    /// Rule 1c: a globalized-global slot `@glob` is disjoint from a pointer *loaded
    /// from it* (`load(@glob) + k`) — but only under `LoadedPointerDisjointFromSlot`.
    /// The slot must not be called disjoint from itself.
    #[test]
    fn globalized_slot_disjoint_from_loaded_pointer_under_assumption() {
        use qcode::{
            assumption::Proposition,
            builder::Builder,
            testing::TestContext,
            value::{BasicBlock, Function, Value},
        };

        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let fid = Function::make(&mut tc.ctx, "f".into()).unwrap().id;
        let root = tc.ctx.get_or_make_block(0x1000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        let sp_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx.values.block_params[sp_pid].origin = Some(ValueId::Varnode(sp_reg));
        let glob_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(4).id;
        let glob = ValueId::BlockParam(glob_pid);
        // A partial-promotion snapshot of `*@glob` (origin = the slot param), as
        // `apply_partial` materializes it.
        let snap_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(4).id;
        tc.ctx.values.block_params[snap_pid].origin = Some(glob);
        let snap = ValueId::BlockParam(snap_pid);

        let ram = tc.ctx.default_space;
        let (store_addr, snap_addr, glob_addr) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, root));
            let p = b.push_load::<false>(glob, 4, ram).id(); // P = *@glob (the buffer pointer)
            let c4 = b.context_mut().get_const(4, 4).id();
            let store_addr = b.push_add(p, c4).id(); // P + 4 (a write through the loaded pointer)
            let snap_addr = b.push_add(snap, c4).id(); // snapshot + 4 (write through the by-value pointer)
            let glob_addr = b.context_mut().get_const(0x454df8, 4).id();
            unsafe { b.dont_finalize() };
            (store_addr, snap_addr, glob_addr)
        };
        tc.ctx.values.block_params[glob_pid].origin = Some(glob_addr);

        let r = AliasResult::simple(&tc.ctx).with_frame_freshness(&tc.ctx, fid, Some(sp_reg));

        // Without the assumption, the reload-through store is opaque.
        assert!(
            !r.provably_disjoint(&tc.ctx, glob, store_addr),
            "no disjointness without LoadedPointerDisjointFromSlot"
        );

        tc.ctx
            .assume_true(Proposition::LoadedPointerDisjointFromSlot(fid));
        assert!(
            r.provably_disjoint(&tc.ctx, glob, store_addr),
            "@glob ⊥ a store through load(@glob) under the assumption"
        );
        assert!(
            r.provably_disjoint(&tc.ctx, store_addr, glob),
            "rule is symmetric"
        );
        assert!(
            r.provably_disjoint(&tc.ctx, snap_addr, glob),
            "the materialized snapshot of *@glob is also a pointer from the slot"
        );
        assert!(
            !r.provably_disjoint(&tc.ctx, glob, glob),
            "a slot is never disjoint from itself (the loaded-pointer path needs a real load)"
        );
    }

    /// Loaded-pointer rule: a deref through a pointer reloaded from a caller-frame
    /// slot (`load(@SP+4) + i`) is taken disjoint from the frame (`@SP`) — but only
    /// when *both* `ArgsDisjointFromCallerFrame` and `LoadedPointerDisjointFromSlot`
    /// are recorded (it rides the caller-frame rule via `peel_loads`).
    #[test]
    fn loaded_pointer_disjoint_from_slot_only_under_assumption() {
        use qcode::{
            assumption::Proposition,
            builder::Builder,
            testing::TestContext,
            value::{BasicBlock, Function, Value},
        };

        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let fid = Function::make(&mut tc.ctx, "f".into()).unwrap().id;
        let root = tc.ctx.get_or_make_block(0x1000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        let sp_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx.values.block_params[sp_pid].origin = Some(ValueId::Varnode(sp_reg));
        let sp = ValueId::BlockParam(sp_pid);

        let ram = tc.ctx.default_space;
        let addr = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, root));
            let c4 = b.context_mut().get_const(4, 8).id();
            let slot = b.push_add(sp, c4).id(); // @SP + 4 (caller-frame slot)
            let cc = b.push_load::<false>(slot, 8, ram).id(); // buf = load(@SP+4)
            let addr = b.push_add(cc, c4).id(); // buf + 4
            unsafe { b.dont_finalize() };
            addr
        };

        // Caller-frame disjointness is assumed, but the reload is opaque without the
        // loaded-pointer assumption, so the deref is not yet input-derived.
        tc.ctx
            .assume_true(Proposition::ArgsDisjointFromCallerFrame(fid));
        let r = AliasResult::simple(&tc.ctx).with_frame_freshness(&tc.ctx, fid, Some(sp_reg));
        assert!(
            !r.provably_disjoint(&tc.ctx, addr, sp),
            "the spilled reload is opaque without LoadedPointerDisjointFromSlot"
        );

        // With both assumptions, the deref counts as incoming and misses the frame.
        tc.ctx
            .assume_true(Proposition::LoadedPointerDisjointFromSlot(fid));
        let r = AliasResult::simple(&tc.ctx).with_frame_freshness(&tc.ctx, fid, Some(sp_reg));
        assert!(
            r.provably_disjoint(&tc.ctx, addr, sp),
            "load(@SP+4) + 4 ⊥ @SP under both assumptions"
        );
        assert!(r.provably_disjoint(&tc.ctx, sp, addr), "rule is symmetric");
    }

    /// Caller-frame rule: a caller-frame slot (`@SP + 8`) is disjoint from an
    /// incoming pointer *only* when `ArgsDisjointFromCallerFrame` is recorded true
    /// for the function — it is an assumption, not statically sound.
    #[test]
    fn caller_frame_disjoint_only_under_assumption() {
        use qcode::{
            assumption::Proposition,
            builder::Builder,
            testing::TestContext,
            value::{BasicBlock, Function},
        };

        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let fid = Function::make(&mut tc.ctx, "f".into()).unwrap().id;
        let root = tc.ctx.get_or_make_block(0x1000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        let sp_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx.values.block_params[sp_pid].origin = Some(ValueId::Varnode(sp_reg));
        let sp = ValueId::BlockParam(sp_pid);
        let arg_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        let arg = ValueId::BlockParam(arg_pid);

        let (caller_arg, arg_plus) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, root));
            let c8 = b.context_mut().get_const(8, 8).id();
            let caller_arg = b.push_add(sp, c8).id(); // @SP + 8  (caller-frame slot)
            let arg_plus = b.push_add(arg, c8).id(); // arg + 8  (input-derived)
            unsafe { b.dont_finalize() };
            (caller_arg, arg_plus)
        };

        let r = AliasResult::simple(&tc.ctx).with_frame_freshness(&tc.ctx, fid, Some(sp_reg));

        // Without the assumption, a caller-frame slot may alias an incoming pointer.
        assert!(
            !r.provably_disjoint(&tc.ctx, caller_arg, arg),
            "caller-frame slot is not statically disjoint from an incoming pointer"
        );

        tc.ctx
            .assume_true(Proposition::ArgsDisjointFromCallerFrame(fid));

        assert!(
            r.provably_disjoint(&tc.ctx, caller_arg, arg),
            "under the assumption, @SP+8 ⊥ incoming pointer"
        );
        assert!(
            r.provably_disjoint(&tc.ctx, arg_plus, caller_arg),
            "symmetric, and applies to an offset from the incoming pointer"
        );
        assert!(
            !r.provably_disjoint(&tc.ctx, caller_arg, sp),
            "@SP is a frame pointer, not an incoming data pointer"
        );
    }
}
