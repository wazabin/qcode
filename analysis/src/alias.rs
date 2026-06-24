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

use crate::gvn::affine::precompute_forms;
use crate::stack::frame::{FrameClass, frame_class, incoming_sp_param};

mod anderson;
mod simple;

pub use anderson::alias_analysis;

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
        // Rule 1: own-frame local ⊥ incoming pointer (sound).
        if (frame.own_frame_locals.contains(&a) && self.is_input_derived(ctx, b))
            || (frame.own_frame_locals.contains(&b) && self.is_input_derived(ctx, a))
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
        let is_caller_frame =
            |v: ValueId| v == frame.sp_param || frame.caller_frame_slots.contains(&v);
        caller_frame_assumed
            && ((is_caller_frame(a) && self.is_input_derived(ctx, b))
                || (is_caller_frame(b) && self.is_input_derived(ctx, a)))
    }

    /// Whether `v` is an own-frame local under the populated frame-freshness
    /// context (see [`AliasResult::provably_disjoint`]).
    pub fn is_own_frame_local(&self, v: ValueId) -> bool {
        self.frame
            .as_ref()
            .is_some_and(|f| f.own_frame_locals.contains(&v))
    }

    /// Whether `v` is *input-derived*: it peels — through address arithmetic
    /// (`add`/`sub` of an offset) and width casts (`zext`/`sext`/`range`) — to a
    /// parameter of this function's root block, i.e. a value that entered from the
    /// caller. The `add`/`sub` peel bails if either operand is itself an own-frame
    /// local, so a nonsensical `local + arg` mix is never read as input.
    fn is_input_derived(&self, ctx: &Context, v: ValueId) -> bool {
        let Some(frame) = &self.frame else {
            return false;
        };
        self.is_input_derived_rec(ctx, frame, v, &mut HashSet::default())
    }

    fn is_input_derived_rec(
        &self,
        ctx: &Context,
        frame: &FrameInfo,
        v: ValueId,
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
                Mnemonic::Binop(b) if matches!(b.op, Binop::Int(IntBinop::Add | IntBinop::Sub)) => {
                    if frame.own_frame_locals.contains(&b.lhs)
                        || frame.own_frame_locals.contains(&b.rhs)
                    {
                        return false;
                    }
                    self.is_input_derived_rec(ctx, frame, b.lhs, seen)
                        || self.is_input_derived_rec(ctx, frame, b.rhs, seen)
                }
                Mnemonic::Zext(z) => self.is_input_derived_rec(ctx, frame, z.src, seen),
                Mnemonic::Sext(s) => self.is_input_derived_rec(ctx, frame, s.src, seen),
                Mnemonic::Range(r) => self.is_input_derived_rec(ctx, frame, r.src, seen),
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

        assert!(r.provably_disjoint(&tc.ctx, local, arg), "local ⊥ incoming arg");
        assert!(r.provably_disjoint(&tc.ctx, arg, local), "rule is symmetric");
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
