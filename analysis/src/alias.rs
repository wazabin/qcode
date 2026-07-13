use std::cell::RefCell;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use qcode::{
    assumption::Proposition,
    space::SpaceId,
    value::{
        BlockId, FunctionId, ValueId, ValueRef, VarnodeId,
        insn::{Binop, IntBinop, Mnemonic},
        util::base_ref::HostRef,
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
    /// `true` when this activation never lets any own-frame address escape: no
    /// frame address is stored to memory, passed to a non-`nocapture` callee
    /// param, handed to an indirect/opaque sink, or listed in a `call.clobbers`.
    /// A *sound*, assumption-free bit (a body scan, no `Proposition`) that lets
    /// an own-frame local be proven disjoint from a pointer loaded from memory or
    /// derived from a call result — nothing in memory can name a slot this
    /// activation never let out. See [`FrameInfo::disjoint_by_provenance`] rule A.
    frame_uncaptured: bool,
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
    pub fn may_alias<'a, 'str: 'a>(
        &self,
        host: impl Into<HostRef<'a, 'str>>,
        a: ValueId,
        b: ValueId,
    ) -> bool {
        let host = host.into();
        // If a and b don't share the same address space they can't alias.
        // Note this deliberately overrides `NodeId::Unknown` below: aliasing
        // means "same storage location", and values in different spaces can
        // never occupy the same location, however unknown their class is.
        let a_space = ValueRef::from_host(host, a).space().map(|s| s.id);
        let b_space = ValueRef::from_host(host, b).space().map(|s| s.id);
        if let (Some(sa), Some(sb)) = (a_space, b_space)
            && sa != sb
        {
            return false;
        }

        // Frame freshness: an own-frame local never aliases an incoming pointer
        // (sound), and a caller-frame slot is disjoint from one under the recorded
        // `ArgsDisjointFromCallerFrame` assumption (see `provably_disjoint`). Answer
        // "no" before the alias graph collapses both partitions to `Unknown`.
        if self.provably_disjoint(host, a, b) {
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
    pub fn provably_disjoint<'a, 'str: 'a>(
        &self,
        host: impl Into<HostRef<'a, 'str>>,
        a: ValueId,
        b: ValueId,
    ) -> bool {
        let Some(frame) = &self.frame else {
            return false;
        };
        let host = host.into();
        let pa = frame.provenance(host, a);
        let pb = frame.provenance(host, b);
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
    pub fn with_frame_freshness<'a, 'str: 'a>(
        mut self,
        host: impl Into<HostRef<'a, 'str>>,
        fid: FunctionId,
        sp_reg: Option<VarnodeId>,
    ) -> Self {
        let host = host.into();
        let Some(sp_reg) = sp_reg else {
            return self;
        };
        let Some(sp) = incoming_sp_param(host, fid, sp_reg) else {
            return self;
        };
        let numbering = precompute_forms(host, fid);
        let mut own_frame_locals = HashSet::default();
        for block in host.function_ref(fid).blocks() {
            for insn in block.iter() {
                let v = ValueId::Instruction(insn.id);
                if let Some(FrameClass::Local) = frame_class(host.shr(), &numbering, sp, v) {
                    own_frame_locals.insert(v);
                }
            }
        }
        let root_block = host.function_ref(fid).root().map(|r| r.id);
        // Hoist the assumption lookups: they are fixed once the result is built
        // (a pass sets assumptions before building the oracle). A test that flips
        // an assumption after building must rebuild the result.
        let caller_frame_assumed = host
            .shr()
            .truth(Proposition::ArgsDisjointFromCallerFrame(fid))
            .is_some_and(|t| t.value);
        let loaded_ptr_assumed = host
            .shr()
            .truth(Proposition::LoadedPointerDisjointFromSlot(fid))
            .is_some_and(|t| t.value);
        let frame_uncaptured = !frame_is_captured(host, fid, &numbering, sp);
        self.frame = Some(FrameInfo {
            root_block,
            sp_param: sp,
            own_frame_locals,
            numbering,
            caller_frame_assumed,
            loaded_ptr_assumed,
            frame_uncaptured,
            provenance: RefCell::new(HashMap::default()),
        });
        self
    }
}

/// Whether any own-frame address escapes this activation — the negation of
/// [`FrameInfo::frame_uncaptured`]. A frame address is *captured* when a value
/// classifying to [`FrameClass::Local`] is:
///
/// - the `src` of a `Store` (the address itself written to memory), or
/// - argument `j` of a direct `Call` whose callee param `j` is not `nocapture`
///   (a callee with no signature has no `nocapture` bit, so it captures), or
/// - present in a `Call`'s `clobbers` (a location the callee writes), or
/// - an argument of a `CallInd`, or an operand of an opaque op (`PCodeOp` /
///   `Map` / `Scan`) that may have arbitrary memory effects.
///
/// Passing a frame address to a `nocapture` param does **not** capture it — that
/// is the whole point of the bit. `Intrinsic` is categorically pure (no memory
/// effects), so it cannot capture and needs no case. A frame address merely
/// loaded through, offset, or *returned* is not captured: a return hands the
/// address back as a call *result* in the caller (classified via
/// [`FrameInfo::classify`]'s call arm), never into this function's memory.
fn frame_is_captured(host: HostRef, fid: FunctionId, numbering: &Numbering, sp: ValueId) -> bool {
    let is_own_frame = |v: ValueId| {
        matches!(
            frame_class(host.shr(), numbering, sp, v),
            Some(FrameClass::Local)
        )
    };
    for block in host.function_ref(fid).blocks() {
        for insn in block.iter() {
            let func = insn.id.func;
            match insn.mnemonic() {
                Mnemonic::Store(s) if is_own_frame(s.src.qualify(func)) => {
                    return true;
                }
                Mnemonic::Call(c) => {
                    let target = c.target.real();
                    for (j, &arg) in c.args.iter().enumerate() {
                        if is_own_frame(arg.qualify(func))
                            && !target
                                .and_then(|target| host.function_ref(target).param_attr(j))
                                .is_some_and(|a| a.nocapture)
                        {
                            return true;
                        }
                    }
                    if c.clobbers.iter().any(|&cl| is_own_frame(cl.qualify(func))) {
                        return true;
                    }
                }
                Mnemonic::CallInd(c)
                    if c.args.iter().any(|&arg| is_own_frame(arg.qualify(func))) =>
                {
                    return true;
                }
                Mnemonic::PCodeOp(_) | Mnemonic::Map(_) | Mnemonic::Scan(_)
                    if insn.operands().iter().any(|&arg| is_own_frame(arg)) =>
                {
                    return true;
                }
                _ => {}
            }
        }
    }
    false
}

impl FrameInfo {
    /// Provenance of `v`, memoized. A peel cycle contributes nothing: the
    /// in-progress value is seeded with the empty set before recursing, then
    /// overwritten with the final classification.
    fn provenance(&self, host: HostRef, v: ValueId) -> Provenance {
        if let Some(&p) = self.provenance.borrow().get(&v) {
            return p;
        }
        self.provenance
            .borrow_mut()
            .insert(v, Provenance::default());
        let p = self.classify(host, v);
        self.provenance.borrow_mut().insert(v, p);
        p
    }

    /// The uncached classification of `v` — the union over its peel tree (see
    /// [`Provenance`]).
    fn classify(&self, host: HostRef, v: ValueId) -> Provenance {
        use Provenance as P;
        // Stack provenance first: `@SP ± k`, realigned frames, and the bare `@SP`
        // param (offset 0 → caller frame).
        if let Some(fc) = frame_class(host.shr(), &self.numbering, self.sp_param, v) {
            return match fc {
                FrameClass::Local => P::OWN_FRAME,
                FrameClass::CallerFrame => P::CALLER_FRAME,
            };
        }
        match v {
            ValueId::Literal(_) => P::GLOBAL_STATIC,
            ValueId::BlockParam(pid) => {
                let bp = host.param_ref(pid);
                match bp.origin() {
                    // A globalized-global slot (`@glob_<addr>`): origin is the literal.
                    Some(ValueId::Literal(_)) => P::GLOBAL_STATIC,
                    // A partial-promotion snapshot of `*slot` (origin = the global
                    // slot param) — the materialized loaded pointer.
                    Some(o @ ValueId::BlockParam(_))
                        if self.provenance(host, o).is_pure(P::GLOBAL_STATIC) =>
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
            ValueId::Instruction(id) if !host.contains_instruction(id) => P::OPAQUE,
            ValueId::Instruction(id) => match host.insn_ref(id).mnemonic() {
                Mnemonic::Load(_) => P::LOADED,
                Mnemonic::Call(c) => self.classify_call_result(host, id.func, c),
                Mnemonic::Zext(z) => self.provenance(host, z.src.qualify(id.func)),
                Mnemonic::Sext(s) => self.provenance(host, s.src.qualify(id.func)),
                Mnemonic::Range(r) => self.provenance(host, r.src.qualify(id.func)),
                Mnemonic::Binop(b) if matches!(b.op, Binop::Int(IntBinop::Add | IntBinop::Sub)) => {
                    self.peel_addsub(host, b.lhs.qualify(id.func), b.rhs.qualify(id.func))
                }
                // Affine `base + const` over a global base — the old
                // `is_global_static` fallback for a global reached via a non-add op.
                _ => {
                    if let Some((base, _)) = self.numbering.base_offset(v)
                        && base != v
                        && self.provenance(host, base).is_pure(P::GLOBAL_STATIC)
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

    /// Classify a direct-call *result* (Refinement B). A callee that cannot
    /// squirrel away any argument — every argument flows into a `nocapture` param
    /// and it writes no pointer clobbers — returns only what it derived from its
    /// args, from memory it read (`LOADED`), or from the static image
    /// (`GLOBAL_STATIC`). So the result's provenance is the union of the args'
    /// provenances widened by `LOADED | GLOBAL_STATIC`. Any callee with a
    /// non-`nocapture` (or unknown/signatureless) param, or a non-empty clobber
    /// set, stays `OPAQUE` — the classifier's default for a call.
    ///
    /// Capture-by-return: if an argument is itself an own-frame address (passed to
    /// a `nocapture` param, then handed back), the union carries `OWN_FRAME`, so
    /// the result is *not* a subset of rule A's `INPUT|LOADED|GLOBAL_STATIC` mask
    /// and stays correctly non-disjoint from the frame.
    fn classify_call_result(
        &self,
        host: HostRef,
        func: FunctionId,
        c: &qcode::value::insn::Call,
    ) -> Provenance {
        use Provenance as P;
        let Some(target) = c.target.real() else {
            return P::OPAQUE;
        };
        let callee = host.function_ref(target);
        let all_nocapture = c.clobbers.is_empty()
            && (0..c.args.len()).all(|j| callee.param_attr(j).is_some_and(|a| a.nocapture));
        if !all_nocapture {
            return P::OPAQUE;
        }
        let mut acc = P::LOADED.union(P::GLOBAL_STATIC);
        for &arg in &c.args {
            acc = acc.union(self.provenance(host, arg.qualify(func)));
        }
        acc
    }

    /// Union the provenance of an add/sub's operands, dropping constant offsets
    /// (literals) and pure-opaque scalar terms (a strided index, an unclassifiable
    /// non-pointer). Both are *offsets*, not sources of the pointer, so they must
    /// not poison the result — matching the old peel, which contributed nothing
    /// for such operands.
    fn peel_addsub(&self, host: HostRef, lhs: ValueId, rhs: ValueId) -> Provenance {
        use Provenance as P;
        let mut acc = P::default();
        for operand in [lhs, rhs] {
            if matches!(operand, ValueId::Literal(_)) {
                continue;
            }
            let p = self.provenance(host, operand);
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
        // Rule A (sound, `frame_uncaptured`): an own-frame local ⊥ any pointer
        // that came from the caller, from memory, or from the static image. If
        // this activation never let a frame address escape (see
        // [`frame_is_captured`]), then nothing in the caller's hands (`INPUT`),
        // nothing readable from memory (`LOADED`, incl. an all-nocapture call
        // result — see [`classify`]'s call arm), and nothing in the static image
        // (`GLOBAL_STATIC`) can name a slot of it. Assumption-free — no
        // `Proposition`, no replay rollback — unlike rule 1c / rule 2. Rule 1
        // above stays unconditional because own-frame ⊥ `INPUT` holds even when
        // the frame *is* captured (the caller's pointer predates our frame).
        if self.frame_uncaptured
            && px.is_pure(P::OWN_FRAME)
            && py.is_nonempty_subset_of(P::INPUT.union(P::LOADED).union(P::GLOBAL_STATIC))
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
        // the loaded-pointer assumption — a pointer reloaded from a slot, from the
        // static image, or an all-nocapture call result derived from those. The
        // mask includes `GLOBAL_STATIC`: a caller-frame slot is stack-rooted, so ⊥
        // the static image by the same argument as rule 1b (which already covers a
        // *pure* global; the widening admits mixed `INPUT|LOADED|GLOBAL_STATIC`
        // call results).
        if self.caller_frame_assumed && px.is_pure(P::CALLER_FRAME) {
            if py.is_pure(P::INPUT) {
                return true;
            }
            if self.loaded_ptr_assumed
                && py.is_nonempty_subset_of(P::INPUT.union(P::LOADED).union(P::GLOBAL_STATIC))
            {
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
        ValueId::Instruction(InstructionId::new(
            qcode::value::FunctionId::new(0),
            qcode::value::LocalInsnId::new(n),
        ))
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
            value::{BasicBlock, FunctionBody},
        };

        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let fid = FunctionBody::make(&mut tc.ctx, "f".into()).unwrap().id;
        let root = { tc.ctx.get_or_make_block(0x1000, fid) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        // `@SP` param (origin = the stack-pointer reg) and a caller data-pointer param.
        let sp_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx
            .block_param_mut(sp_pid)
            .set_origin_id(ValueId::Varnode(sp_reg).localize(sp_pid.func));
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

        let r = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            &tc.ctx,
            fid,
            Some(sp_reg),
        );

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
        let plain = AliasResult::simple_for_function(&tc.ctx, fid);
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
            value::{BasicBlock, FunctionBody},
        };

        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let fid = FunctionBody::make(&mut tc.ctx, "f".into()).unwrap().id;
        let root = { tc.ctx.get_or_make_block(0x1000, fid) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        let sp_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx
            .block_param_mut(sp_pid)
            .set_origin_id(ValueId::Varnode(sp_reg).localize(sp_pid.func));
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
        tc.ctx
            .block_param_mut(glob_pid)
            .set_origin_id(glob_addr.localize(glob_pid.func));

        let r = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            &tc.ctx,
            fid,
            Some(sp_reg),
        );

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
        let plain = AliasResult::simple_for_function(&tc.ctx, fid);
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
            value::{BasicBlock, FunctionBody, Value},
        };

        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let fid = FunctionBody::make(&mut tc.ctx, "f".into()).unwrap().id;
        let root = { tc.ctx.get_or_make_block(0x1000, fid) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        let sp_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx
            .block_param_mut(sp_pid)
            .set_origin_id(ValueId::Varnode(sp_reg).localize(sp_pid.func));
        let glob_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(4).id;
        let glob = ValueId::BlockParam(glob_pid);
        // A partial-promotion snapshot of `*@glob` (origin = the slot param), as
        // `apply_partial` materializes it.
        let snap_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(4).id;
        tc.ctx
            .block_param_mut(snap_pid)
            .set_origin_id(glob.localize(snap_pid.func));
        let snap = ValueId::BlockParam(snap_pid);

        let ram = tc.ctx.shared.default_space;
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
        tc.ctx
            .block_param_mut(glob_pid)
            .set_origin_id(glob_addr.localize(glob_pid.func));

        let r = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            &tc.ctx,
            fid,
            Some(sp_reg),
        );

        // Without the assumption, the reload-through store is opaque.
        assert!(
            !r.provably_disjoint(&tc.ctx, glob, store_addr),
            "no disjointness without LoadedPointerDisjointFromSlot"
        );

        // The assumption is hoisted at build time, so rebuild the result after
        // recording it.
        tc.ctx
            .assume_true(Proposition::LoadedPointerDisjointFromSlot(fid));
        let r = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            &tc.ctx,
            fid,
            Some(sp_reg),
        );
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
            value::{BasicBlock, FunctionBody, Value},
        };

        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let fid = FunctionBody::make(&mut tc.ctx, "f".into()).unwrap().id;
        let root = { tc.ctx.get_or_make_block(0x1000, fid) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        let sp_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx
            .block_param_mut(sp_pid)
            .set_origin_id(ValueId::Varnode(sp_reg).localize(sp_pid.func));
        let sp = ValueId::BlockParam(sp_pid);

        let ram = tc.ctx.shared.default_space;
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
        let r = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            &tc.ctx,
            fid,
            Some(sp_reg),
        );
        assert!(
            !r.provably_disjoint(&tc.ctx, addr, sp),
            "the spilled reload is opaque without LoadedPointerDisjointFromSlot"
        );

        // With both assumptions, the deref counts as incoming and misses the frame.
        tc.ctx
            .assume_true(Proposition::LoadedPointerDisjointFromSlot(fid));
        let r = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            &tc.ctx,
            fid,
            Some(sp_reg),
        );
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
            value::{BasicBlock, FunctionBody},
        };

        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let fid = FunctionBody::make(&mut tc.ctx, "f".into()).unwrap().id;
        let root = { tc.ctx.get_or_make_block(0x1000, fid) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        let sp_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx
            .block_param_mut(sp_pid)
            .set_origin_id(ValueId::Varnode(sp_reg).localize(sp_pid.func));
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

        let r = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            &tc.ctx,
            fid,
            Some(sp_reg),
        );

        // Without the assumption, a caller-frame slot may alias an incoming pointer.
        assert!(
            !r.provably_disjoint(&tc.ctx, caller_arg, arg),
            "caller-frame slot is not statically disjoint from an incoming pointer"
        );

        // The assumption is hoisted at build time, so rebuild after recording it.
        tc.ctx
            .assume_true(Proposition::ArgsDisjointFromCallerFrame(fid));
        let r = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            &tc.ctx,
            fid,
            Some(sp_reg),
        );

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
            value::{BasicBlock, FunctionBody},
        };

        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let fid = FunctionBody::make(&mut tc.ctx, "f".into()).unwrap().id;
        let root = { tc.ctx.get_or_make_block(0x1000, fid) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        let sp_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx
            .block_param_mut(sp_pid)
            .set_origin_id(ValueId::Varnode(sp_reg).localize(sp_pid.func));
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

        let r = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            &tc.ctx,
            fid,
            Some(sp_reg),
        );

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
            value::{BasicBlock, FunctionBody, Value},
        };

        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let fid = FunctionBody::make(&mut tc.ctx, "f".into()).unwrap().id;
        let root = { tc.ctx.get_or_make_block(0x1000, fid) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        let sp_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx
            .block_param_mut(sp_pid)
            .set_origin_id(ValueId::Varnode(sp_reg).localize(sp_pid.func));
        let sp = ValueId::BlockParam(sp_pid);
        let arg_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        let arg = ValueId::BlockParam(arg_pid);
        let p_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        let p = ValueId::BlockParam(p_pid);

        let ram = tc.ctx.shared.default_space;
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

        // No assumptions: the own-frame local *is* disjoint from `arg + load(p)`
        // via the sound Refinement A — this activation captures nothing, so no
        // caller pointer or loaded value can name a slot of its frame. The
        // caller-frame slot, by contrast, still needs the assumed rule 2 below.
        let r = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            &tc.ctx,
            fid,
            Some(sp_reg),
        );
        assert!(
            r.provably_disjoint(&tc.ctx, local, mix),
            "own-frame ⊥ input|loaded when the frame is uncaptured (Refinement A)"
        );
        assert!(!r.provably_disjoint(&tc.ctx, caller_arg, mix));

        // Only the caller-frame assumption: still opaque (the load is not admitted).
        tc.ctx
            .assume_true(Proposition::ArgsDisjointFromCallerFrame(fid));
        let r = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            &tc.ctx,
            fid,
            Some(sp_reg),
        );
        assert!(
            !r.provably_disjoint(&tc.ctx, caller_arg, mix),
            "the loaded operand is not admitted without LoadedPointerDisjointFromSlot"
        );

        // Both assumptions: the caller-frame slot is disjoint from arg + load(p).
        tc.ctx
            .assume_true(Proposition::LoadedPointerDisjointFromSlot(fid));
        let r = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            &tc.ctx,
            fid,
            Some(sp_reg),
        );
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
            value::{BasicBlock, FunctionBody},
        };

        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let fid = FunctionBody::make(&mut tc.ctx, "f".into()).unwrap().id;
        let root = { tc.ctx.get_or_make_block(0x1000, fid) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        let sp_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx
            .block_param_mut(sp_pid)
            .set_origin_id(ValueId::Varnode(sp_reg).localize(sp_pid.func));
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

        let r = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            &tc.ctx,
            fid,
            Some(sp_reg),
        );
        assert!(r.provably_disjoint(&tc.ctx, local, deep));
        let after_first = r.frame.as_ref().unwrap().provenance.borrow().len();
        assert!(r.provably_disjoint(&tc.ctx, local, deep));
        let after_second = r.frame.as_ref().unwrap().provenance.borrow().len();
        assert_eq!(
            after_first, after_second,
            "the second query classifies nothing new"
        );
    }

    // === Step 8: nocapture-driven frame-freshness refinements ===============

    use qcode::value::insn::{Call, CallInd};
    use qcode::value::{BasicBlock, FunctionBody, ParamAttrs, Value};

    /// Make function `name` with an `@SP` root param (origin = `sp_reg`); returns
    /// `(fid, root_block, @SP value)`.
    fn fn_with_sp(
        tc: &mut qcode::testing::TestContext,
        name: &'static str,
        addr: u64,
        sp_reg: VarnodeId,
    ) -> (FunctionId, BlockId, ValueId) {
        let fid = FunctionBody::make(&mut tc.ctx, name.into()).unwrap().id;
        let root = { tc.ctx.get_or_make_block(addr, fid) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        let sp_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx
            .block_param_mut(sp_pid)
            .set_origin_id(ValueId::Varnode(sp_reg).localize(sp_pid.func));
        (fid, root, ValueId::BlockParam(sp_pid))
    }

    /// A bodyless callee carrying the given per-param attributes.
    fn callee_with_attrs(
        tc: &mut qcode::testing::TestContext,
        name: &'static str,
        attrs: Vec<ParamAttrs>,
    ) -> FunctionId {
        let fid = FunctionBody::make(&mut tc.ctx, name.into()).unwrap().id;
        FunctionBody::from_id_mut(&mut tc.ctx, fid).set_param_attrs(attrs);
        fid
    }

    const NOCAPTURE: ParamAttrs = ParamAttrs {
        readonly: false,
        nocapture: true,
    };
    const CAPTURES: ParamAttrs = ParamAttrs {
        readonly: false,
        nocapture: false,
    };

    /// Refinement A (sound): when the frame is never captured, an own-frame local
    /// is disjoint from a pointer loaded from memory — which rule 1/1b/1c cannot
    /// prove (a `LOADED` pointer could be an escaped-and-reloaded frame address,
    /// but here nothing escaped).
    #[test]
    fn uncaptured_own_frame_disjoint_from_loaded_pointer() {
        use qcode::builder::Builder;
        let mut tc = qcode::testing::TestContext::new();
        let sp_reg = tc.r0;
        let (fid, root, sp) = fn_with_sp(&mut tc, "f", 0x1000, sp_reg);
        let ram = tc.ctx.shared.default_space;

        let (local, loaded) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, root));
            let c8 = b.context_mut().get_const(8, 8).id();
            let local = b.push_sub(sp, c8).id(); // @SP - 8 (own-frame local)
            let g = b.context_mut().get_const(0x404040, 8).id();
            let loaded = b.push_load::<false>(g, 8, ram).id(); // load(global) -> LOADED
            unsafe { b.dont_finalize() };
            (local, loaded)
        };

        let r = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            &tc.ctx,
            fid,
            Some(sp_reg),
        );
        assert!(
            r.frame.as_ref().unwrap().frame_uncaptured,
            "nothing captures the frame"
        );
        assert!(
            r.provably_disjoint(&tc.ctx, local, loaded),
            "own-frame local ⊥ a loaded pointer when the frame is uncaptured"
        );
        assert!(r.provably_disjoint(&tc.ctx, loaded, local), "symmetric");
    }

    /// A frame address written to memory captures it, so Refinement A goes inert:
    /// the own-frame local is no longer proven disjoint from a loaded pointer
    /// (which could now be the escaped address reloaded).
    #[test]
    fn captured_by_store_makes_refinement_a_inert() {
        use qcode::builder::Builder;
        let mut tc = qcode::testing::TestContext::new();
        let sp_reg = tc.r0;
        let (fid, root, sp) = fn_with_sp(&mut tc, "f", 0x1000, sp_reg);
        let ram = tc.ctx.shared.default_space;

        let (local, loaded) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, root));
            let c8 = b.context_mut().get_const(8, 8).id();
            let local = b.push_sub(sp, c8).id(); // @SP - 8
            let g = b.context_mut().get_const(0x404040, 8).id();
            b.push_store(local, g, ram); // *global = local  (frame address escapes)
            let loaded = b.push_load::<false>(g, 8, ram).id();
            unsafe { b.dont_finalize() };
            (local, loaded)
        };

        let r = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            &tc.ctx,
            fid,
            Some(sp_reg),
        );
        assert!(
            !r.frame.as_ref().unwrap().frame_uncaptured,
            "storing the frame address captures the frame"
        );
        assert!(
            !r.provably_disjoint(&tc.ctx, local, loaded),
            "Refinement A is inert once the frame is captured"
        );
    }

    /// Passing a frame address to a `nocapture` callee param does not capture it;
    /// passing it to a capturing (or signatureless) callee, or to `CallInd`, does.
    #[test]
    fn call_capture_respects_nocapture_bit() {
        use qcode::builder::Builder;

        // Helper: build `f` that calls `configure` to emit a call taking `@SP-8`,
        // then report `frame_uncaptured`.
        fn uncaptured_after(
            configure: impl FnOnce(&mut qcode::testing::TestContext, FunctionId) -> FunctionId,
            indirect: bool,
        ) -> bool {
            let mut tc = qcode::testing::TestContext::new();
            let sp_reg = tc.r0;
            let (fid, root, sp) = fn_with_sp(&mut tc, "f", 0x1000, sp_reg);
            let callee = configure(&mut tc, fid);
            let cid = {
                let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, root));
                let c8 = b.context_mut().get_const(8, 8).id();
                let local = b.push_sub(sp, c8).id(); // @SP - 8
                let cid = if indirect {
                    let t = b.context_mut().get_const(0x9000, 8).id();
                    b.push_call_ind(t).id
                } else {
                    b.push_call(callee).id
                };
                unsafe { b.dont_finalize() };
                (cid, local)
            };
            let (cid, local) = cid;
            let mn = if indirect {
                Mnemonic::CallInd(CallInd {
                    ptr: match tc.ctx.get_insn(cid).mnemonic() {
                        Mnemonic::CallInd(c) => c.ptr,
                        _ => unreachable!(),
                    },
                    args: vec![local.localize(cid.func)],
                })
            } else {
                Mnemonic::Call(Call {
                    target: qcode::value::insn::Callee::Real(callee),
                    args: vec![local.localize(cid.func)],
                    clobbers: vec![],
                })
            };
            tc.ctx.replace_instruction_mnemonic(cid, mn);
            let r = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
                &tc.ctx,
                fid,
                Some(sp_reg),
            );
            r.frame.as_ref().unwrap().frame_uncaptured
        }

        assert!(
            uncaptured_after(|tc, _| callee_with_attrs(tc, "g", vec![NOCAPTURE]), false),
            "a frame address into a nocapture param does not capture the frame"
        );
        assert!(
            !uncaptured_after(|tc, _| callee_with_attrs(tc, "g", vec![CAPTURES]), false),
            "a frame address into a capturing param captures the frame"
        );
        assert!(
            !uncaptured_after(
                |tc, _| FunctionBody::make(&mut tc.ctx, "g".into()).unwrap().id,
                false
            ),
            "a frame address into a signatureless callee captures the frame"
        );
        assert!(
            !uncaptured_after(|_, fid| fid, true),
            "a frame address into an indirect call captures the frame"
        );
    }

    /// Refinement B: the result of an all-`nocapture` direct call is classified
    /// as the union of its args' provenances widened by `LOADED | GLOBAL_STATIC`,
    /// so an own-frame local (uncaptured) is disjoint from `g(input)`. A callee
    /// with a capturing (or missing) param leaves the result `OPAQUE` — not
    /// disjoint.
    #[test]
    fn all_nocapture_call_result_classified_from_args() {
        use qcode::builder::Builder;

        fn local_disjoint_from_call_result(attrs: Vec<ParamAttrs>) -> bool {
            let mut tc = qcode::testing::TestContext::new();
            let sp_reg = tc.r0;
            let (fid, root, sp) = fn_with_sp(&mut tc, "f", 0x1000, sp_reg);
            // A caller-supplied (INPUT) data pointer param.
            let arg_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
            let input = ValueId::BlockParam(arg_pid);
            let callee = callee_with_attrs(&mut tc, "g", attrs);

            let (local, call_result) = {
                let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, root));
                let c8 = b.context_mut().get_const(8, 8).id();
                let local = b.push_sub(sp, c8).id(); // @SP - 8
                let cr = b.push_call(callee).id;
                unsafe { b.dont_finalize() };
                (local, cr)
            };
            tc.ctx.replace_instruction_mnemonic(
                call_result,
                Mnemonic::Call(Call {
                    target: qcode::value::insn::Callee::Real(callee),
                    args: vec![input.localize(call_result.func)], // g(input): INPUT arg, does not capture the frame
                    clobbers: vec![],
                }),
            );
            let r = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
                &tc.ctx,
                fid,
                Some(sp_reg),
            );
            assert!(
                r.frame.as_ref().unwrap().frame_uncaptured,
                "an INPUT arg does not capture the frame"
            );
            r.provably_disjoint(&tc.ctx, local, ValueId::Instruction(call_result))
        }

        assert!(
            local_disjoint_from_call_result(vec![NOCAPTURE]),
            "own-frame ⊥ an all-nocapture call result (INPUT|LOADED|GLOBAL_STATIC)"
        );
        assert!(
            !local_disjoint_from_call_result(vec![CAPTURES]),
            "a capturing callee leaves its result OPAQUE — not disjoint"
        );
        assert!(
            !local_disjoint_from_call_result(vec![]),
            "a signatureless callee leaves its result OPAQUE — not disjoint"
        );
    }

    /// Widened rule 2: under `ArgsDisjointFromCallerFrame` +
    /// `LoadedPointerDisjointFromSlot`, a caller-frame slot is disjoint from a
    /// mixed `INPUT | GLOBAL_STATIC` pointer (an all-nocapture call result shape).
    /// Rule 1b does not cover it (the `INPUT` bit makes it impure), and it is not
    /// disjoint without the assumptions.
    #[test]
    fn widened_rule2_caller_frame_disjoint_from_input_global_mix() {
        use qcode::assumption::Proposition;
        use qcode::builder::Builder;
        let mut tc = qcode::testing::TestContext::new();
        let sp_reg = tc.r0;
        let (fid, root, sp) = fn_with_sp(&mut tc, "f", 0x1000, sp_reg);
        // An INPUT param and a globalized-global param.
        let in_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        let input = ValueId::BlockParam(in_pid);
        let glob_pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        let glob = ValueId::BlockParam(glob_pid);

        let (caller_slot, mixed, glob_addr) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, root));
            let c8 = b.context_mut().get_const(8, 8).id();
            let caller_slot = b.push_add(sp, c8).id(); // @SP + 8 (caller frame)
            let mixed = b.push_add(input, glob).id(); // INPUT ∪ GLOBAL_STATIC
            let glob_addr = b.context_mut().get_const(0x454df8, 8).id();
            unsafe { b.dont_finalize() };
            (caller_slot, mixed, glob_addr)
        };
        tc.ctx
            .block_param_mut(glob_pid)
            .set_origin_id(glob_addr.localize(glob_pid.func));

        // Without the assumptions, rule 2 is inert and rule 1b cannot fire (mixed
        // is not a pure global).
        let plain = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            &tc.ctx,
            fid,
            Some(sp_reg),
        );
        assert!(
            !plain.provably_disjoint(&tc.ctx, caller_slot, mixed),
            "no caller-frame disjointness without the assumptions"
        );

        tc.ctx
            .assume_true(Proposition::ArgsDisjointFromCallerFrame(fid));
        tc.ctx
            .assume_true(Proposition::LoadedPointerDisjointFromSlot(fid));
        let r = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            &tc.ctx,
            fid,
            Some(sp_reg),
        );
        assert!(
            r.provably_disjoint(&tc.ctx, caller_slot, mixed),
            "caller-frame slot ⊥ INPUT|GLOBAL_STATIC under the assumptions (widened mask)"
        );
    }
}
