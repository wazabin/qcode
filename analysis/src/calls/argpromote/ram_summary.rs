//! RAM instantiation of the [`EffectChannel`](super::summary::EffectChannel)
//! fixpoint — stage 2b of moving the RAM argpromote channel onto the effect
//! engine (see `argpromote-ram-effects-migration`).
//!
//! A function's summary is its **outward memory footprint**: the set of scalar
//! fields ([`RamField`]) and bounded dynamic regions ([`RamRegion`]) it may
//! read or write, each hung off a [`RamBase`] a caller can rebase —
//! a positional pointer argument (`Param`), an absolute address (`Global`),
//! or (minted only by [`transfer`](EffectChannel::transfer)) a slot in the
//! summary owner's own frame (`Frame`). ⊤ is any footprint the lattice cannot
//! express: an unclassifiable access, a non-positional interface, a budget
//! overflow, or a call edge whose argument cannot be rebased.
//!
//! Outward-**invisible** accesses never enter a summary:
//!
//! - **Function-private spaces** (`space.shared()` is `None`): a promoted
//!   callee's shadow is a body-local `TempSpace` seeded from its own by-value
//!   snapshot params — it can neither observe nor alias anything the caller
//!   names.
//! - **Own-frame locals**, *written*, or read back after this function's own
//!   same-block same-slot write. An **unlicensed** read of an own-frame slot
//!   observes whatever dead frame last occupied those bytes (the classic
//!   uninitialized-local idiom) — that refutes the freshness hypothesis and
//!   is ⊤, see `uninit_frame_read_refutes_freshness`.
//!
//! The `transfer` rebases a callee's `Param(i)` entries through the actual
//! argument at each call site: through a caller pointer param (composing),
//! a literal address (→ `Global`), or a caller own-frame local (→ `Frame`,
//! **writes only** — the callee's write lands in the caller's frame and dies
//! with it, so it is dropped again one level further up; a callee *read*
//! through a frame argument would need store-licensing knowledge the transfer
//! does not have, so it stays ⊤). Loaded-pointer arguments under existing
//! disjointness assumptions are TODO — ⊤ for now.
//!
//! The blocking-call gate still admits only outward-invisible callees
//! (`Frame`-only or empty summaries); letting a caller compose a callee's
//! real `Param`/`Global` footprint into its own interface is the
//! materialization step that follows.

use qcode::{
    context::Context,
    value::{
        FunctionBody, FunctionId, ModuleView, ValueId, ValueRef, VarnodeId,
        insn::{InstructionId, Mnemonic},
    },
};
use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;

use crate::CallGraph;
use crate::calls::CallEdge;
use crate::sequence::{AddressRelation, MemoryAccess, collect_regions_for_base, relate_address};

use super::globals::global_slot;
use super::ram::OwnFrame;
use super::summary::{EffectChannel, EffectSummaries, solve_summaries};

/// Total entries (fields + regions) a summary may hold before it saturates.
/// Joins at call edges grow sets; saturation is the finite-height guarantee —
/// a saturated summary is treated as ⊤ by every consumer.
pub(crate) const MAX_EFFECT_ENTRIES: usize = 64;

/// What an effect entry's offsets are relative to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum RamBase {
    /// The pointer passed at this positional argument index of the summary
    /// owner's `pure_reg` interface (`param[i] ↔ Call.args[i]` lockstep).
    Param(u32),
    /// A slot in the summary owner's **own frame**: a constant offset (`< 0`)
    /// from its incoming `@SP`. Minted only by `transfer` (a callee effect
    /// rebased through an own-frame-local argument), always a *write*, and
    /// dropped again when transferred one level further up — the frame dies at
    /// return, so the effect is contained.
    Frame(i64),
    /// An absolute (literal) address in real ram.
    Global(u64),
}

/// One scalar effect: `size` bytes at `base + offset`, read or written.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RamField {
    pub base: RamBase,
    pub offset: i64,
    pub size: usize,
    pub write: bool,
}

/// One bounded dynamic-index effect: the half-open byte span
/// `[base + lo, base + hi)`, read (and written iff `write`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RamRegion {
    pub base: RamBase,
    pub lo: i64,
    pub hi: i64,
    pub write: bool,
}

/// The precise half of a summary: the exhaustively classified footprint.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Footprint {
    pub fields: FxHashSet<RamField>,
    pub regions: FxHashSet<RamRegion>,
}

impl Footprint {
    fn len(&self) -> usize {
        self.fields.len() + self.regions.len()
    }

    fn invisible(&self) -> bool {
        self.fields
            .iter()
            .all(|f| matches!(f.base, RamBase::Frame(_)))
            && self
                .regions
                .iter()
                .all(|r| matches!(r.base, RamBase::Frame(_)))
    }
}

/// A function's solved memory summary, two independently-⊤ components:
///
/// - `precise`: the exhaustive outward footprint (`None` = inexpressible —
///   an unclassifiable access, a non-lockstep interface, budget saturation, or
///   an unrebasable call edge). The blocking-call gate's authority.
/// - `written`: the coarse set of shared non-register spaces the function may
///   *store* to, own-frame and all (`None` = unbounded — an external, an
///   unresolved `BranchInd` escape, or an unbounded callee). What
///   `written_spaces` / `mem_forward`'s call-prune consume. Deliberately laxer
///   than `precise`: a function whose footprint defies classification still
///   usually has a bounded space set.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RamEffects {
    pub precise: Option<Footprint>,
    pub written: Option<std::collections::BTreeSet<qcode::space::SpaceId>>,
}

impl Default for RamEffects {
    fn default() -> Self {
        Self {
            precise: Some(Footprint::default()),
            written: Some(Default::default()),
        }
    }
}

impl RamEffects {
    /// Whether nothing in this footprint is observable by any caller: empty,
    /// or `Frame`-contained (writes into the owner's own frame, dead at
    /// return). This is the blocking-call gate's admission predicate.
    pub(crate) fn outward_invisible(&self) -> bool {
        self.precise.as_ref().is_some_and(Footprint::invisible)
    }
}

/// Per-caller classification state the transfer reuses across that caller's
/// call edges: the positional param index map (lockstep interfaces only) and
/// the frame/affine view.
struct CallerInfo {
    materialized: bool,
    params: FxHashMap<ValueId, u32>,
    frame: OwnFrame,
}

/// The RAM [`EffectChannel`]. `sp` is the stack-pointer varnode used to
/// recognise own-frame locals; without it the own-frame carve-outs are inert.
pub(crate) struct RamChannel {
    pub(crate) sp: Option<VarnodeId>,
    /// Lazily built per-caller state; the solve is single-threaded.
    callers: RefCell<FxHashMap<FunctionId, CallerInfo>>,
}

impl RamChannel {
    pub(crate) fn new(sp: Option<VarnodeId>) -> Self {
        Self {
            sp,
            callers: RefCell::new(FxHashMap::default()),
        }
    }

    /// Classify one call argument as a rebase target: the caller-side base the
    /// callee's `Param` entries land on, plus a constant offset shift.
    fn classify_arg(
        &self,
        ctx: &Context,
        caller: FunctionId,
        arg: ValueId,
    ) -> Option<(RamBase, i64)> {
        if let ValueRef::Literal(lit) = ValueRef::new(arg, ctx) {
            return Some((RamBase::Global(lit.value()), 0));
        }
        let mut callers = self.callers.borrow_mut();
        let info = callers.entry(caller).or_insert_with(|| {
            let f = FunctionBody::from_id(ctx, caller);
            let params = f
                .root()
                .map(|b| {
                    b.params()
                        .enumerate()
                        .map(|(i, p)| (p.id(), i as u32))
                        .collect()
                })
                .unwrap_or_default();
            CallerInfo {
                materialized: f.is_reg_materialized(),
                params,
                frame: OwnFrame::new(ctx, caller, self.sp),
            }
        });
        // A caller own-frame local: the callee's effect lands in the caller's
        // frame. Checked *before* the positional map — the `@SP` param is
        // itself a root param, so an `@SP - k` local would otherwise rebase to
        // `Param(sp_index)` and lose its containment. (Write-only admission is
        // enforced at the entry rebase.)
        if let Some(t) = info.frame.local_offset(ctx, arg) {
            return Some((RamBase::Frame(t), 0));
        }
        let (base, off) = info
            .frame
            .numbering()
            .base_offset(ModuleView::new(ctx), arg)
            .unwrap_or((arg, 0));
        // Any other `@SP`-rooted argument (the return-address slot, an
        // incoming stack arg, a realigned base): not containable, not
        // composable — ⊤.
        if Some(base) == info.frame.sp_param() {
            return None;
        }
        // A caller pointer param + const: composes positionally — but only if
        // the caller's own interface is lockstep, else the index is meaningless.
        if info.materialized
            && let Some(&j) = info.params.get(&base)
        {
            return Some((RamBase::Param(j), off));
        }
        // TODO(stage 2b): loaded-pointer arguments admitted by existing
        // disjointness assumptions (LoadedPointerDisjointFromSlot /
        // assume_arg_frame). ⊤ until the assumption plumbing lands.
        None
    }

    /// The precise footprint of `fid`'s classified outward accesses, or `None`
    /// (footprint-⊤) when any access defies the lattice.
    fn extract_footprint(
        &self,
        ctx: &Context,
        fid: FunctionId,
        own_frame: &mut Option<OwnFrame>,
        outward: &[MemoryAccess],
        raw: &[(qcode::value::LocalValueId, qcode::space::LocalMemorySpaceId)],
    ) -> Option<Footprint> {
        if outward.is_empty() {
            return Some(Footprint::default());
        }
        // Effects hang off positional argument indices, which are meaningful
        // only for a lockstep (`pure_reg`) interface.
        let f = FunctionBody::from_id(ctx, fid);
        if !f.is_reg_materialized() {
            return None;
        }
        let numbering = own_frame
            .get_or_insert_with(|| OwnFrame::new(ctx, fid, self.sp))
            .numbering();
        let params: Vec<(ValueId, u32)> = f
            .root()?
            .params()
            .enumerate()
            .map(|(i, p)| (p.id(), i as u32))
            .collect();

        let mut eff = Footprint::default();
        let mut claimed: FxHashSet<InstructionId> = FxHashSet::default();
        for &(param, idx) in &params {
            for access in outward {
                if let AddressRelation::Const(off) =
                    relate_address(ctx, numbering, param, access.ptr, access.block)
                {
                    eff.fields.insert(RamField {
                        base: RamBase::Param(idx),
                        offset: off,
                        size: access.size,
                        write: access.is_store,
                    });
                    claimed.insert(access.id);
                }
            }
            let region_set = collect_regions_for_base(ctx, numbering, param, outward);
            for ru in &region_set.regions {
                let lo = i64::try_from(ru.region.base_off).ok()?;
                let hi = lo.checked_add(i64::try_from(ru.region.byte_len()).ok()?)?;
                eff.regions.insert(RamRegion {
                    base: RamBase::Param(idx),
                    lo,
                    hi,
                    write: ru.region.has_write,
                });
                claimed.extend(ru.accesses.iter().copied());
            }
            // `rejected_accesses` stay unclaimed; the completeness check below
            // sends the function to ⊤ if nothing else claims them.
        }
        for (access, &(lptr, space)) in outward.iter().zip(raw) {
            if claimed.contains(&access.id) {
                continue;
            }
            if let Some((addr, _)) = global_slot(ctx, fid, lptr, space) {
                eff.fields.insert(RamField {
                    base: RamBase::Global(addr),
                    offset: 0,
                    size: access.size,
                    write: access.is_store,
                });
                claimed.insert(access.id);
            }
        }
        // Completeness: a single unclaimed outward access means the footprint
        // is not fully expressed — the summary must be ⊤, not a subset.
        if outward.iter().any(|a| !claimed.contains(&a.id)) {
            return None;
        }
        if eff.len() > MAX_EFFECT_ENTRIES {
            return None;
        }
        Some(eff)
    }
}

impl EffectChannel for RamChannel {
    type Effects = RamEffects;

    fn scan(&self, ctx: &Context, fid: FunctionId) -> Option<RamEffects> {
        // The two components fail independently: `written` collects every
        // shared non-register store space regardless of classifiability, and
        // only an unresolved `BranchInd` (a PLT-stub `goto [GOT]` escape into
        // code the summary cannot see) unbounds it. `precise` dies on the
        // first access the footprint lattice cannot express.
        let mut written: Option<std::collections::BTreeSet<qcode::space::SpaceId>> =
            Some(Default::default());
        let mut precise_top = false;
        // ---- outward-access filter (2a + freshness licensing) --------------
        // Built on the first shared-space access only: most memory-free
        // candidates have none, and the affine numbering behind the frame
        // classifier is the expensive part.
        let mut own_frame: Option<OwnFrame> = None;
        // Shared-space accesses that are caller-observable, to be claimed by
        // the footprint extraction below. `(access, raw ptr, space)` keeps the
        // unqualified pointer for `global_slot`.
        let mut outward: Vec<MemoryAccess> = Vec::new();
        let mut raw: Vec<(qcode::value::LocalValueId, qcode::space::LocalMemorySpaceId)> =
            Vec::new();
        let mut register_touch = false;
        for block in FunctionBody::from_id(ctx, fid).blocks() {
            // Own-frame slots this block has stored to, in program order (see
            // the freshness licensing rationale in the module docs).
            let mut stored: FxHashSet<(i64, usize)> = FxHashSet::default();
            for insn in block.iter() {
                let (space, lptr, size, is_store) = match insn.mnemonic() {
                    Mnemonic::Load(l) => (l.space, l.ptr, l.size, false),
                    Mnemonic::Store(s) => (s.space, s.ptr, s.size, true),
                    Mnemonic::BranchInd(_) => {
                        written = None;
                        continue;
                    }
                    _ => continue,
                };
                // Function-private (shadow/temp) space: outward-invisible.
                let Some(shared) = space.shared() else {
                    continue;
                };
                // Register traffic is the register channel's business; a
                // function still moving registers through loads/stores is not
                // functionalized and its footprint is not expressible here.
                if matches!(
                    qcode::space::Space::from_id(ctx, shared).ty,
                    qcode::space::SpaceType::Register
                ) {
                    register_touch = true;
                    continue;
                }
                // Coarse channel: every shared non-register *store* space,
                // own-frame included — parity with the retired SpaceChannel.
                if is_store && let Some(w) = written.as_mut() {
                    w.insert(shared);
                }
                let ptr = lptr.qualify(insn.id.func);
                let frame = own_frame.get_or_insert_with(|| OwnFrame::new(ctx, fid, self.sp));
                if frame.is_local(ctx, ptr) {
                    if is_store {
                        if let Some(off) = frame.local_offset(ctx, ptr) {
                            stored.insert((off, size));
                        }
                        continue;
                    }
                    if frame
                        .local_offset(ctx, ptr)
                        .is_some_and(|off| stored.contains(&(off, size)))
                    {
                        continue; // licensed reload of an own write
                    }
                    // Unlicensed own-frame read: observes the previous dead
                    // frame — the freshness hypothesis is refuted.
                    precise_top = true;
                    continue;
                }
                if frame.is_frame_slot(ctx, ptr) {
                    // A caller-frame slot (`@SP + k`, `k ≥ 0`): the return
                    // address / incoming stack arguments. promote_stack_args
                    // territory; not expressible here yet.
                    precise_top = true;
                    continue;
                }
                outward.push(MemoryAccess {
                    id: insn.id,
                    block: block.id,
                    is_store,
                    ptr,
                    size,
                });
                raw.push((lptr, space));
            }
        }
        Some(RamEffects {
            precise: (!precise_top && !register_touch)
                .then(|| self.extract_footprint(ctx, fid, &mut own_frame, &outward, &raw))
                .flatten(),
            written,
        })
    }

    fn external_leaf(&self, _ctx: &Context, _fid: FunctionId) -> Option<RamEffects> {
        // Componentwise external parity: the *gate* keeps treating a resolved
        // bodyless external as footprint-inert (legacy behavior — in practice
        // every external call site carries ABI clobbers, which the caller-side
        // clobber check rejects on its own), while the *coarse* channel keeps
        // it unbounded (an external may write anywhere), exactly as the
        // retired SpaceChannel did.
        Some(RamEffects {
            precise: Some(Footprint::default()),
            written: None,
        })
    }

    fn transfer(&self, ctx: &Context, edge: &CallEdge, callee: &RamEffects) -> Option<RamEffects> {
        // The coarse component moves by identity: space ids are global.
        let written = callee.written.clone();
        let Some(fp) = &callee.precise else {
            return Some(RamEffects {
                precise: None,
                written,
            });
        };
        // Only a real positional `Call` site can rebase `Param` entries. Any
        // other direct-like edge (tail call, `Apply`/`Map`/`Scan`, synthetic)
        // composes only an invisible footprint.
        let call_args = edge
            .site
            .and_then(|site| match ctx.get_insn(site).mnemonic() {
                Mnemonic::Call(c) => Some(c.args.clone()),
                _ => None,
            });
        let Some(args) = call_args else {
            return Some(RamEffects {
                precise: fp.invisible().then(Footprint::default),
                written,
            });
        };
        let caller = edge.caller;
        // The precise rebase degrades to footprint-⊤ on its own (an
        // unrebasable argument never unbounds the coarse component).
        let precise = (|| -> Option<Footprint> {
            let mut out = Footprint::default();
            // Rebase one entry's base; `write`-ness gates the own-frame landing.
            let rebase = |base: RamBase, write: bool| -> Option<Option<(RamBase, i64)>> {
                Some(match base {
                    // Contained in the callee's own frame — invisible here.
                    RamBase::Frame(_) => None,
                    RamBase::Global(a) => Some((RamBase::Global(a), 0)),
                    RamBase::Param(i) => {
                        let arg = args.get(i as usize)?.qualify(caller);
                        let (new_base, shift) = self.classify_arg(ctx, caller, arg)?;
                        if matches!(new_base, RamBase::Frame(_)) && !write {
                            // A callee *read* through a frame argument needs
                            // store-licensing knowledge (freshness) — ⊤.
                            return None;
                        }
                        Some((new_base, shift))
                    }
                })
            };
            for f in &fp.fields {
                match rebase(f.base, f.write)? {
                    None => {}
                    Some((base, shift)) => {
                        let offset = f.offset.checked_add(shift)?;
                        // A frame landing must stay strictly below the caller's
                        // entry SP — crossing into the return-address slot or the
                        // caller's caller frame is not containable.
                        if let RamBase::Frame(t) = base
                            && t.checked_add(offset)?
                                .checked_add(i64::try_from(f.size).ok()?)?
                                > 0
                        {
                            return None;
                        }
                        out.fields.insert(RamField { base, offset, ..*f });
                    }
                }
            }
            for r in &fp.regions {
                match rebase(r.base, r.write)? {
                    None => {}
                    Some((base, shift)) => {
                        let lo = r.lo.checked_add(shift)?;
                        let hi = r.hi.checked_add(shift)?;
                        if let RamBase::Frame(t) = base
                            && t.checked_add(hi)? > 0
                        {
                            return None;
                        }
                        out.regions.insert(RamRegion { base, lo, hi, ..*r });
                    }
                }
            }
            if out.len() > MAX_EFFECT_ENTRIES {
                return None;
            }
            Some(out)
        })();
        Some(RamEffects { precise, written })
    }

    fn join(&self, into: &mut RamEffects, from: &RamEffects) -> bool {
        let mut grew = false;
        // Precise: union under the entry budget; ⊤ absorbs.
        into.precise = match (into.precise.take(), &from.precise) {
            (Some(mut a), Some(b)) => {
                let before = a.len();
                a.fields.extend(b.fields.iter().copied());
                a.regions.extend(b.regions.iter().copied());
                grew |= a.len() != before;
                if a.len() > MAX_EFFECT_ENTRIES {
                    grew = true;
                    None
                } else {
                    Some(a)
                }
            }
            (None, _) => None,
            (Some(_), None) => {
                grew = true;
                None
            }
        };
        // Coarse: space-set union; ⊤ absorbs.
        into.written = match (into.written.take(), &from.written) {
            (Some(mut a), Some(b)) => {
                let before = a.len();
                a.extend(b.iter().copied());
                grew |= a.len() != before;
                Some(a)
            }
            (None, _) => None,
            (Some(_), None) => {
                grew = true;
                None
            }
        };
        grew
    }
}

/// Solve the outward-footprint summary for every function in `ctx`.
pub(crate) fn solve(
    ctx: &Context,
    graph: &CallGraph,
    sp: Option<VarnodeId>,
) -> EffectSummaries<RamChannel> {
    solve_summaries(ctx, graph, &RamChannel::new(sp))
}

/// Whether `fid`'s solved summary says it has no caller-observable memory
/// effect. `false` for ⊤, saturation, any real footprint, and call targets
/// outside the solved snapshot.
pub(crate) fn is_memory_free(summaries: &EffectSummaries<RamChannel>, fid: FunctionId) -> bool {
    summaries
        .try_get(fid)
        .is_some_and(|s| s.as_ref().is_ok_and(|e| e.outward_invisible()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::value::QCodeMut;
    use qcode::value::insn::{Call, CallTag, Callee};
    use qcode_macro::qcode;

    /// Find the (single) call in `block`, give it positional `args` and the
    /// regpure tag, and resolve it to `target`.
    fn regpure_call(
        tc: &mut qcode::testing::TestContext,
        block: qcode::value::BlockId,
        target: FunctionId,
        args: Vec<ValueId>,
    ) -> InstructionId {
        let call_id = qcode::value::BasicBlock::from_id(&tc.ctx, block)
            .iter()
            .find(|i| matches!(i.mnemonic(), Mnemonic::Call(_)))
            .unwrap()
            .id;
        tc.ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target: Callee::Real(target),
                args: args
                    .into_iter()
                    .map(|arg| arg.localize(call_id.func))
                    .collect(),
                clobbers: vec![],
                tag: CallTag::RegPure,
            }),
        );
        call_id
    }

    fn materialize(tc: &mut qcode::testing::TestContext, fid: FunctionId) {
        FunctionBody::from_id_mut(&mut tc.ctx, fid).set_effects(
            qcode::value::FunctionEffects::Materialized(
                qcode::value::RegisterInterfaceMap::default(),
            ),
        );
    }

    fn set_sp_origin(tc: &mut qcode::testing::TestContext, fid: FunctionId, sp: VarnodeId) {
        let pid = {
            let f = FunctionBody::from_id(&tc.ctx, fid);
            let p = f
                .root()
                .unwrap()
                .params()
                .find(|p| p.name() == Some("RSP"))
                .unwrap();
            match p.id() {
                ValueId::BlockParam(pid) => pid,
                _ => unreachable!(),
            }
        };
        tc.ctx
            .block_param_mut(pid)
            .set_origin_id(ValueId::Varnode(sp).localize(pid.func));
    }

    /// The transitive hole the legacy body-rescan gate had: `mid` is memory-free
    /// in its own body but calls a loading `leaf`, so its summary must be ⊤ —
    /// a caller handing `mid` a promoted pointer is not safe. (A register
    /// access is a caller-observable effect: still ⊤ under the region lattice.)
    #[test]
    fn memory_free_is_transitive() {
        let mut tc = qcode::testing::TestContext::new();
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn leaf:
                <leaf_entry>
                    %v = load(register:8, {r0});
                    return at i64 0;
            fn mid:
                <mid_entry>
                    call fn leaf();
                <mid_cont>
                    return at i64 0;
            fn pure:
                <pure_entry>
                    return at i64 0;
            "
        );
        let _ = (leaf_entry, mid_entry, mid_cont, pure_entry);
        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, None);
        assert!(!is_memory_free(&s, leaf));
        assert!(!is_memory_free(&s, mid));
        assert!(is_memory_free(&s, pure));
    }

    /// Accesses in a function-private (shadow/temp) space are outward-invisible.
    #[test]
    fn private_space_access_is_outward_invisible() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn shadowed:
                <entry @p:i64>
                    return at i64 0;
            "
        );
        let _ = entry;
        let shadow = tc.ctx.bodies[shadowed].push_temp_space(qcode::value::TempSpace::new(
            Some("test_shadow"),
            1,
            8,
        ));
        let shadow = qcode::space::LocalMemorySpaceId::Temp(shadow.local);
        let root = FunctionBody::from_id(&tc.ctx, shadowed).root().unwrap().id;
        let p = qcode::value::BasicBlock::from_id(&tc.ctx, root)
            .params()
            .next()
            .unwrap()
            .id();
        let mut b = (&mut tc.ctx).builder(root);
        b.set_insert_point_to_start();
        let a = b.shr().get_const(0x10, 8);
        b.push_store(p, a, shadow);

        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, None);
        assert!(
            is_memory_free(&s, shadowed),
            "a private-space store must not poison the summary"
        );
    }

    /// The uninitialized-local idiom that falsifies own-frame freshness:
    ///
    /// ```c
    /// void init(void)  { char *str = "Hello World!\n"; return; }
    /// void hello(void) { char *str; printf(str); return; }
    /// int main(void)   { init(); hello(); return 0; }
    /// ```
    ///
    /// Frame accounting: both are called from the same `main` call sites, so
    /// their entry `@RSP` values are equal (each `call` pushes the return
    /// address at the caller's `SP - 8`; the return address itself sits at
    /// offset `0`). Each prologue writes the saved-`RBP` slot at `-0x8`; the
    /// `str` local lives below the prologue at `-0x10`. `hello`'s prologue
    /// write at `-0x8` is its *own* store (licensed, private), but its read
    /// of the untouched `-0x10` observes the slot `init`'s dead frame left
    /// behind — on the machine, `hello` prints "Hello World!". Such a read
    /// refutes the freshness hypothesis: `hello` must be ⊤, and the
    /// composing caller with it. `init` (write-only frame traffic) and a
    /// spill/reload (read *after* own write) stay outward-invisible.
    #[test]
    fn uninit_frame_read_refutes_freshness() {
        let mut tc = qcode::testing::TestContext::new();
        let sp = tc.r3;
        qcode!(
            tc.ctx,
            "
            fn init:
                <init_entry @RSP:i64>
                    %rbp_slot = @RSP - i64 0x8;
                    store(ram:8, %rbp_slot <- i64 0x0);
                    %slot = @RSP - i64 0x10;
                    store(ram:8, %slot <- i64 0x4010);
                    return at i64 0;

            fn hello:
                <hello_entry @RSP:i64>
                    %rbp_slot = @RSP - i64 0x8;
                    store(ram:8, %rbp_slot <- i64 0x0);
                    %slot = @RSP - i64 0x10;
                    %str = load(ram:8, %slot);
                    return at %str;

            fn spill:
                <spill_entry @RSP:i64 @v:i64>
                    %slot = @RSP - i64 0x8;
                    store(ram:8, %slot <- @v);
                    %r = load(ram:8, %slot);
                    return at %r;

            fn entry:
                <entry_1>
                    call fn init();
                <entry_2>
                    call fn hello();
                <entry_3>
                    return at i64 0;
            "
        );
        let _ = (
            init_entry,
            hello_entry,
            spill_entry,
            entry_1,
            entry_2,
            entry_3,
        );
        for fid in [init, hello, spill] {
            set_sp_origin(&mut tc, fid, sp);
        }

        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, Some(sp));
        assert!(
            is_memory_free(&s, init),
            "a write-only own frame is dead on exit — outward-invisible"
        );
        assert!(
            is_memory_free(&s, spill),
            "a spill/reload (read after own write) is private frame traffic"
        );
        assert!(
            !is_memory_free(&s, hello),
            "reading an own-frame slot before writing it observes the previous \
             dead frame — the freshness hypothesis is refuted, hello is ⊤"
        );
        assert!(
            !is_memory_free(&s, entry),
            "hello's refuted freshness must poison the composing caller too"
        );
    }

    /// Stage 2b scan: a lockstep callee's param-relative derefs become
    /// `Param`-based fields — a real (non-⊤) footprint, still blocking.
    #[test]
    fn scan_extracts_param_fields() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn callee:
                <c_entry @p:i64 @q:i64>
                    store(ram:4, @p <- i32 9);
                    %a = @q + i64 0x8;
                    %v = load(ram:4, %a);
                    return at %v;
            "
        );
        let _ = c_entry;
        materialize(&mut tc, callee);
        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, None);
        let eff = s.get(callee).as_ref().expect("summary is expressible");
        let fp = eff.precise.as_ref().expect("footprint is expressible");
        let mut fields: Vec<RamField> = fp.fields.iter().copied().collect();
        fields.sort_by_key(|f| (f.base, f.offset));
        assert_eq!(
            fields,
            vec![
                RamField {
                    base: RamBase::Param(0),
                    offset: 0,
                    size: 4,
                    write: true
                },
                RamField {
                    base: RamBase::Param(1),
                    offset: 8,
                    size: 4,
                    write: false
                },
            ]
        );
        assert!(!is_memory_free(&s, callee), "a real footprint still blocks");
    }

    /// Stage 2b transfer: `Param` entries rebase through the actual call
    /// arguments — a caller pointer param + const composes positionally, a
    /// literal address lands on `Global`.
    #[test]
    fn transfer_rebases_param_and_literal_args() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn callee:
                <c_entry @p:i64 @q:i64>
                    store(ram:4, @p <- i32 9);
                    %a = @q + i64 0x8;
                    %v = load(ram:4, %a);
                    return at %v;

            fn caller:
                <k_entry @r:i64>
                    %arg = @r + i64 0x10;
                    goto <k_call>;
                <k_call>
                    call <callee>;
                <k_cont>
                    return at i64 0;
            "
        );
        let _ = (c_entry, k_entry);
        materialize(&mut tc, callee);
        materialize(&mut tc, caller);
        let lit = tc.ctx.get_const(0x4000, 8).id();
        let arg = {
            let block = qcode::value::BasicBlock::from_id(&tc.ctx, k_entry);
            block
                .iter()
                .find(|i| matches!(i.mnemonic(), Mnemonic::Binop(_)))
                .unwrap()
                .id
        };
        regpure_call(
            &mut tc,
            k_call,
            callee,
            vec![ValueId::Instruction(arg), lit],
        );
        tc.ctx.add_cfg_edge(k_call, k_cont);

        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, None);
        let eff = s.get(caller).as_ref().expect("summary is expressible");
        let fp = eff.precise.as_ref().expect("rebase must succeed");
        let mut fields: Vec<RamField> = fp.fields.iter().copied().collect();
        fields.sort_by_key(|f| (f.base, f.offset));
        assert_eq!(
            fields,
            vec![
                RamField {
                    base: RamBase::Param(0),
                    offset: 0x10,
                    size: 4,
                    write: true
                },
                RamField {
                    base: RamBase::Global(0x4000),
                    offset: 8,
                    size: 4,
                    write: false
                },
            ]
        );
        assert!(!is_memory_free(&s, caller));
    }

    /// Stage 2b frame containment: a callee *write* through a pointer to the
    /// caller's own-frame local lands on `Frame` — contained, outward-
    /// invisible, and dropped one level further up. A callee *read* through
    /// the same shape would need freshness licensing: ⊤.
    #[test]
    fn frame_landing_contains_writes_rejects_reads() {
        let mut tc = qcode::testing::TestContext::new();
        let sp = tc.r3;
        qcode!(
            tc.ctx,
            "
            fn sink:
                <s_entry @p:i64>
                    store(ram:8, @p <- i64 1);
                    return at i64 0;

            fn source:
                <o_entry @p:i64>
                    %v = load(ram:8, @p);
                    return at %v;

            fn wcaller:
                <w_entry @RSP:i64>
                    %loc = @RSP - i64 0x20;
                    goto <w_call>;
                <w_call>
                    call <sink>;
                <w_cont>
                    return at i64 0;

            fn rcaller:
                <r_entry @RSP:i64>
                    %loc = @RSP - i64 0x20;
                    goto <r_call>;
                <r_call>
                    call <source>;
                <r_cont>
                    return at i64 0;

            fn top:
                <t_entry>
                    call fn wcaller();
                <t_cont>
                    return at i64 0;
            "
        );
        let _ = (s_entry, o_entry, w_entry, r_entry, t_entry, t_cont);
        for fid in [sink, source, wcaller, rcaller] {
            materialize(&mut tc, fid);
        }
        for fid in [wcaller, rcaller] {
            set_sp_origin(&mut tc, fid, sp);
        }
        for (entry, call, cont, callee) in [
            (w_entry, w_call, w_cont, sink),
            (r_entry, r_call, r_cont, source),
        ] {
            let loc = qcode::value::BasicBlock::from_id(&tc.ctx, entry)
                .iter()
                .find(|i| matches!(i.mnemonic(), Mnemonic::Binop(_)))
                .unwrap()
                .id;
            regpure_call(&mut tc, call, callee, vec![ValueId::Instruction(loc)]);
            tc.ctx.add_cfg_edge(call, cont);
        }

        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, Some(sp));
        assert!(!is_memory_free(&s, sink), "sink's own footprint is real");
        assert!(
            is_memory_free(&s, wcaller),
            "the write into wcaller's own frame is contained — outward-invisible"
        );
        assert!(
            is_memory_free(&s, top),
            "the Frame entry must be dropped when transferred further up"
        );
        assert!(
            !is_memory_free(&s, rcaller),
            "a callee read through a frame pointer needs freshness licensing — ⊤"
        );
    }
}
