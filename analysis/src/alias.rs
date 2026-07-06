use std::cell::RefCell;

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

mod provenance;
mod simple;

use provenance::Provenance;
pub use simple::RegisterBase;

/// Per-function frame-freshness context, precomputed when an [`AliasResult`] is
/// built with [`AliasResult::with_frame_freshness`]. `None` leaves the rule inert
/// (hand-built results and callers without a stack-pointer register).
pub(crate) struct FrameInfo {
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
    /// The function's affine numbering, kept so [`AliasResult::provably_disjoint`]
    /// can classify an *arbitrary* pointer (`@SP ± k`, `@glob + k`, …) on demand —
    /// not just the precomputed own/caller-frame instruction sets — which the
    /// stack-vs-global rule needs for literal and globalized-param pointers.
    numbering: Numbering,
    /// Hoisted [`Proposition::ArgsDisjointFromCallerFrame`] truth for this
    /// function — fixed for the lifetime of the result (assumptions are set
    /// before the result is built).
    caller_frame_assumed: bool,
    /// Hoisted [`Proposition::LoadedPointerDisjointFromSlot`] truth.
    loaded_ptr_assumed: bool,
    /// Memoized per-value provenance classification (see [`provenance`]). Behind
    /// a `RefCell` so [`AliasResult::provably_disjoint`] keeps its `&self`
    /// signature; the result is used single-threaded within a pass.
    provenance: RefCell<HashMap<ValueId, Provenance>>,
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
    /// `NodeId::Unknown`, or if either value was never seen during analysis
    /// (an untracked value carries no no-alias guarantee, so it must answer
    /// may-alias). Only the earlier layers — the different-space
    /// short-circuit and [`Self::provably_disjoint`] — can return `false`.
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
            // A value the analysis never saw gives us no guarantee. This happens
            // when a pass queries a value it created after the result was built;
            // "don't know" must answer may-alias, not no-alias.
            (None, _) | (_, None) => true,
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
        let pa = frame.provenance(ctx, a);
        let pb = frame.provenance(ctx, b);
        frame.disjoint_by_provenance(pa, pb) || frame.disjoint_by_provenance(pb, pa)
    }

    /// Whether `v` is an own-frame local under the populated frame-freshness
    /// context (see [`AliasResult::provably_disjoint`]).
    pub fn is_own_frame_local(&self, v: ValueId) -> bool {
        self.frame
            .as_ref()
            .is_some_and(|f| f.own_frame_locals.contains(&v))
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
        for block in Function::from_id(ctx, fid).blocks() {
            for insn in block.iter() {
                let v = ValueId::Instruction(insn.id);
                if let Some(FrameClass::Local) = frame_class(ctx, &numbering, sp, v) {
                    own_frame_locals.insert(v);
                }
            }
        }
        let root_block = Function::from_id(ctx, fid).root().map(|r| r.id);
        // Hoist the assumption lookups: they are fixed once the result is built
        // (a pass sets assumptions before building the oracle). A test that flips
        // an assumption after building must rebuild the result.
        let caller_frame_assumed = ctx
            .truth(Proposition::ArgsDisjointFromCallerFrame(fid))
            .is_some_and(|t| t.value);
        let loaded_ptr_assumed = ctx
            .truth(Proposition::LoadedPointerDisjointFromSlot(fid))
            .is_some_and(|t| t.value);
        self.frame = Some(FrameInfo {
            root_block,
            sp_param: sp,
            own_frame_locals,
            numbering,
            caller_frame_assumed,
            loaded_ptr_assumed,
            provenance: RefCell::new(HashMap::default()),
        });
        self
    }
}

impl FrameInfo {
    /// Provenance of `v`, memoized. A peel cycle contributes nothing: the
    /// in-progress value is seeded with the empty set before recursing, then
    /// overwritten with the final classification.
    fn provenance(&self, ctx: &Context, v: ValueId) -> Provenance {
        if let Some(&p) = self.provenance.borrow().get(&v) {
            return p;
        }
        self.provenance
            .borrow_mut()
            .insert(v, Provenance::default());
        let p = self.classify(ctx, v);
        self.provenance.borrow_mut().insert(v, p);
        p
    }

    /// The uncached classification of `v` — the union over its peel tree (see
    /// [`Provenance`]).
    fn classify(&self, ctx: &Context, v: ValueId) -> Provenance {
        use Provenance as P;
        // Stack provenance first: `@SP ± k`, realigned frames, and the bare `@SP`
        // param (offset 0 → caller frame).
        if let Some(fc) = frame_class(ctx, &self.numbering, self.sp_param, v) {
            return match fc {
                FrameClass::Local => P::OWN_FRAME,
                FrameClass::CallerFrame => P::CALLER_FRAME,
            };
        }
        match v {
            ValueId::Literal(_) => P::GLOBAL_STATIC,
            ValueId::BlockParam(pid) => {
                let bp = BlockParam::from_id(ctx, pid);
                match bp.origin() {
                    // A globalized-global slot (`@glob_<addr>`): origin is the literal.
                    Some(ValueId::Literal(_)) => P::GLOBAL_STATIC,
                    // A partial-promotion snapshot of `*slot` (origin = the global
                    // slot param) — the materialized loaded pointer.
                    Some(o @ ValueId::BlockParam(_))
                        if self.provenance(ctx, o).is_pure(P::GLOBAL_STATIC) =>
                    {
                        P::LOADED
                    }
                    // A non-`@SP` root-block param is a caller-supplied pointer.
                    _ if v != self.sp_param
                        && bp.parent().is_some_and(|b| self.root_block == Some(b.id)) =>
                    {
                        P::INPUT
                    }
                    _ => P::OPAQUE,
                }
            }
            ValueId::Instruction(id) => match Instruction::from_id(ctx, id).mnemonic() {
                Mnemonic::Load(_) => P::LOADED,
                Mnemonic::Zext(z) => self.provenance(ctx, z.src),
                Mnemonic::Sext(s) => self.provenance(ctx, s.src),
                Mnemonic::Range(r) => self.provenance(ctx, r.src),
                Mnemonic::Binop(b) if matches!(b.op, Binop::Int(IntBinop::Add | IntBinop::Sub)) => {
                    self.peel_addsub(ctx, b.lhs, b.rhs)
                }
                // Affine `base + const` over a global base — the old
                // `is_global_static` fallback for a global reached via a non-add op.
                _ => {
                    if let Some((base, _)) = self.numbering.base_offset(v)
                        && base != v
                        && self.provenance(ctx, base).is_pure(P::GLOBAL_STATIC)
                    {
                        P::GLOBAL_STATIC
                    } else {
                        P::OPAQUE
                    }
                }
            },
            _ => P::OPAQUE,
        }
    }

    /// Union the provenance of an add/sub's operands, dropping constant offsets
    /// (literals) and pure-opaque scalar terms (a strided index, an unclassifiable
    /// non-pointer). Both are *offsets*, not sources of the pointer, so they must
    /// not poison the result — matching the old peel, which contributed nothing
    /// for such operands.
    fn peel_addsub(&self, ctx: &Context, lhs: ValueId, rhs: ValueId) -> Provenance {
        use Provenance as P;
        let mut acc = P::default();
        for operand in [lhs, rhs] {
            if matches!(operand, ValueId::Literal(_)) {
                continue;
            }
            let p = self.provenance(ctx, operand);
            if p.is_pure(P::OPAQUE) {
                continue;
            }
            acc = acc.union(p);
        }
        if acc.is_empty() { P::OPAQUE } else { acc }
    }

    /// Whether an `x`-provenance pointer is disjoint from a `y`-provenance one
    /// (asymmetric; callers try both orders). See [`AliasResult::provably_disjoint`].
    fn disjoint_by_provenance(&self, px: Provenance, py: Provenance) -> bool {
        use Provenance as P;
        // Rule 1 (sound): own-frame local ⊥ a pure incoming pointer. The callee's
        // frame is younger than anything the caller could already name.
        if px.is_pure(P::OWN_FRAME) && py.is_pure(P::INPUT) {
            return true;
        }
        // Rule 1b (sound): a stack address ⊥ a static-global address — the live
        // stack and the static image occupy disjoint regions.
        if px.is_nonempty_subset_of(P::OWN_FRAME.union(P::CALLER_FRAME))
            && py.is_pure(P::GLOBAL_STATIC)
        {
            return true;
        }
        // Rule 1c (assumed): a globalized-global slot ⊥ a pointer *loaded from it*.
        // Excludes a slot from itself (a bare global has no LOADED bit) and mixed
        // `load(x) + @glob2` shapes (the global bit may re-enter the static image).
        if self.loaded_ptr_assumed
            && px.is_pure(P::GLOBAL_STATIC)
            && py.contains(P::LOADED)
            && !py.contains(P::OPAQUE)
            && !py.contains(P::GLOBAL_STATIC)
        {
            return true;
        }
        // Rule 2 (assumed): a caller-frame slot ⊥ an incoming pointer, and — under
        // the loaded-pointer assumption — a pointer reloaded from a slot.
        if self.caller_frame_assumed && px.is_pure(P::CALLER_FRAME) {
            if py.is_pure(P::INPUT) {
                return true;
            }
            if self.loaded_ptr_assumed && py.is_nonempty_subset_of(P::INPUT.union(P::LOADED)) {
                return true;
            }
        }
        false
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

        // The assumption is hoisted at build time, so rebuild the result after
        // recording it.
        tc.ctx
            .assume_true(Proposition::LoadedPointerDisjointFromSlot(fid));
        let r = AliasResult::simple(&tc.ctx).with_frame_freshness(&tc.ctx, fid, Some(sp_reg));
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

        // The assumption is hoisted at build time, so rebuild after recording it.
        tc.ctx
            .assume_true(Proposition::ArgsDisjointFromCallerFrame(fid));
        let r = AliasResult::simple(&tc.ctx).with_frame_freshness(&tc.ctx, fid, Some(sp_reg));

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

    /// Soundness tightening (review item 3): a pointer with *mixed* input and
    /// stack provenance (`arg + (sp - arg)`) is no longer treated as a pure
    /// incoming pointer, so it is not called disjoint from an own-frame local.
    /// A bare `arg + const` still is (guards against over-tightening).
    #[test]
    fn mixed_input_and_stack_is_not_input_derived() {
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
        let arg_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        let arg = ValueId::BlockParam(arg_pid);

        let (local, mix, arg_plus) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, root));
            let c8 = b.context_mut().get_const(8, 8).id();
            let local = b.push_sub(sp, c8).id(); // @SP - 8  (own-frame local)
            let sp_minus_arg = b.push_sub(sp, arg).id(); // sp - arg  (mixed)
            let mix = b.push_add(arg, sp_minus_arg).id(); // arg + (sp - arg)
            let arg_plus = b.push_add(arg, c8).id(); // arg + 8  (pure input)
            unsafe { b.dont_finalize() };
            (local, mix, arg_plus)
        };

        let r = AliasResult::simple(&tc.ctx).with_frame_freshness(&tc.ctx, fid, Some(sp_reg));

        assert!(
            !r.provably_disjoint(&tc.ctx, local, mix),
            "a mixed input/stack pointer is not a pure incoming pointer"
        );
        assert!(
            r.provably_disjoint(&tc.ctx, local, arg_plus),
            "arg + const is still a pure incoming pointer (not over-tightened)"
        );
    }

    /// Load poisoning: `arg + load(p)` carries both `INPUT` and `LOADED`, so it
    /// is only disjoint from a caller-frame slot when *both* the caller-frame and
    /// loaded-pointer assumptions are recorded, and never from an own-frame local
    /// with no assumptions.
    #[test]
    fn input_plus_load_needs_both_assumptions() {
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
        let arg_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        let arg = ValueId::BlockParam(arg_pid);
        let p_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        let p = ValueId::BlockParam(p_pid);

        let ram = tc.ctx.default_space;
        let (local, caller_arg, mix) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, root));
            let c8 = b.context_mut().get_const(8, 8).id();
            let local = b.push_sub(sp, c8).id(); // @SP - 8 (own-frame local)
            let caller_arg = b.push_add(sp, c8).id(); // @SP + 8 (caller frame)
            let loaded = b.push_load::<false>(p, 8, ram).id(); // load(p)
            let mix = b.push_add(arg, loaded).id(); // arg + load(p)
            unsafe { b.dont_finalize() };
            (local, caller_arg, mix)
        };

        // No assumptions: not disjoint from an own-frame local.
        let r = AliasResult::simple(&tc.ctx).with_frame_freshness(&tc.ctx, fid, Some(sp_reg));
        assert!(!r.provably_disjoint(&tc.ctx, local, mix));
        assert!(!r.provably_disjoint(&tc.ctx, caller_arg, mix));

        // Only the caller-frame assumption: still opaque (the load is not admitted).
        tc.ctx
            .assume_true(Proposition::ArgsDisjointFromCallerFrame(fid));
        let r = AliasResult::simple(&tc.ctx).with_frame_freshness(&tc.ctx, fid, Some(sp_reg));
        assert!(
            !r.provably_disjoint(&tc.ctx, caller_arg, mix),
            "the loaded operand is not admitted without LoadedPointerDisjointFromSlot"
        );

        // Both assumptions: the caller-frame slot is disjoint from arg + load(p).
        tc.ctx
            .assume_true(Proposition::LoadedPointerDisjointFromSlot(fid));
        let r = AliasResult::simple(&tc.ctx).with_frame_freshness(&tc.ctx, fid, Some(sp_reg));
        assert!(
            r.provably_disjoint(&tc.ctx, caller_arg, mix),
            "@SP+8 ⊥ arg + load(p) under both assumptions"
        );
    }

    /// Memoization smoke: a repeated query on a deep peel chain must not grow the
    /// memo table on the second call (every value is classified once).
    #[test]
    fn provenance_is_memoized() {
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
        let arg_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        let arg = ValueId::BlockParam(arg_pid);

        let (local, deep) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, root));
            let c8 = b.context_mut().get_const(8, 8).id();
            let local = b.push_sub(sp, c8).id();
            // A deep peel chain over the incoming arg.
            let mut deep = arg;
            for _ in 0..8 {
                deep = b.push_add(deep, c8).id();
            }
            unsafe { b.dont_finalize() };
            (local, deep)
        };

        let r = AliasResult::simple(&tc.ctx).with_frame_freshness(&tc.ctx, fid, Some(sp_reg));
        assert!(r.provably_disjoint(&tc.ctx, local, deep));
        let after_first = r.frame.as_ref().unwrap().provenance.borrow().len();
        assert!(r.provably_disjoint(&tc.ctx, local, deep));
        let after_second = r.frame.as_ref().unwrap().provenance.borrow().len();
        assert_eq!(
            after_first, after_second,
            "the second query classifies nothing new"
        );
    }
}
