//! RAM instantiation of the [`EffectChannel`](super::summary::EffectChannel)
//! fixpoint — stage 2b of moving the RAM argpromote channel onto the effect
//! engine (see `argpromote-ram-effects-migration`).
//!
//! A function's summary is its **outward memory footprint**: the set of scalar
//! fields ([`RamField`]), bounded dynamic regions ([`RamRegion`]), and
//! extent-unknown whole objects ([`RamObject`], minted only for prototyped
//! externals) it may read or write, each hung off a [`RamBase`] a caller can
//! rebase —
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
//! does not have, so it stays ⊤). A loaded-pointer argument that is a spilled
//! incoming pointer param reloaded from a frame slot is forwarded back to its
//! bare param *before* the solve runs (the `argpromote-forward` pipeline step and
//! the `module(gvn)` after `assume_arg_frame` in `mark-pure`, under the
//! `LoadedPointerDisjointFromSlot` / `ArgsDisjointFromCallerFrame` assumptions),
//! so it reaches `classify_arg` as the param itself and composes on `Param(i)`; a
//! reload the assumption does not license stays ⊤.
//!
//! # External argmem and the confinement assumption
//!
//! A bodyless external with a known C prototype gets a real footprint instead of
//! ⊤: an external can only touch memory *we* model through the pointers *we* pass
//! it (its own libc-internal state lives outside the lifted image). So
//! [`external_leaf`](EffectChannel::external_leaf) mints whole-object
//! [`RamObject`] entries per pointer parameter — hung off the `Param(i)` it
//! arrived on — keyed off the parameter's [`ArgMemKind`]:
//!
//! - `ConstPtr` (`const char *`) → one read object.
//! - `OutPtr` (a write-only destination the `extern_argmem` table vouches for,
//!   `memset`/`memcpy` dest) → one `write` object, no read half.
//! - `MutPtr` (an ordinary non-`const` pointer, `strcat` dest / `realloc`) →
//!   **both** a `write` object and a read object. The read half is the point:
//!   it means a `Frame`-local landing goes ⊤ in `transfer` (freshness), so a
//!   caller cannot memory-free-fold a mutable-pointer external through an
//!   uninitialized local — the same uninit-local-read hole the bodied-function
//!   scan refutes (`uninit_frame_read_refutes_freshness`).
//!
//! A callback/function-pointer param (`qsort` re-enters our code), a `char **`
//! (transitive write escapes the addressed object), varargs (`printf`'s `%n`),
//! or no usable prototype fall back to ⊤.
//!
//! Because a whole-object entry has **no extent**, containing a rebased `Frame`
//! landing (the `memset(&local)` fold) rests on a *confinement assumption*, but
//! one now narrowed to true confinement: **a write-only (`OutPtr`) external
//! writes only within the object addressed by the pointer we pass it** (e.g.
//! `memset`'s length stays in bounds) — it never reads the frame local's stale
//! pre-call contents, because `OutPtr` carries no read object at all. A `MutPtr`
//! external, which *might* read those contents, now carries its read half and so
//! goes ⊤ on a frame landing instead of resting on the assumption. This
//! remaining confinement is **not** statically sound on its own — kin to
//! `CopyBuffersDisjoint` /
//! `LoadedPointerDisjointFromSlot`, which are recorded `Assumed` in the
//! [assumptions registry](qcode::assumption::Proposition) with the
//! checkpoint+replay net as backstop.
//!
//! This confinement is now *recorded*, not merely documented, via a flag-set on
//! the summary lattice. `transfer` runs inside the read-only (`&Context`) solve
//! and cannot call `assume_true`, so it does not register anything itself;
//! instead each [`RamEffects`] carries an `externals` flag-set naming every
//! prototyped external whose whole-object entry was folded into it — minted by
//! the caller doing the rebase (any landing base) and union-propagated
//! transitively up the call graph (so `g → f → memset` reaches `g`). The flag-set
//! is provenance, not footprint: it is exempt from [`MAX_EFFECT_ENTRIES`]
//! saturation. Then at the `&mut Context` promotion commit point (`ram::try_promote`,
//! any outcome ≠ `No`) the retail pass reads the promoted function `f`'s solved
//! flag-set and records one
//! [`Proposition::ExternalArgmemConfinement(f, e)`](qcode::assumption::Proposition)
//! per external `e`, so a future refutation invalidates exactly `f` under the
//! checkpoint+replay net (v1 has no verifier).
//!
//! The blocking-call gate still admits only outward-invisible callees
//! (`Frame`-only or empty summaries); letting a caller compose a callee's
//! real `Param`/`Global` footprint into its own interface is the
//! materialization step that follows.

use jstd::graph::analysis::{DominatorTree, compute_dominators};
use qcode::{
    context::Context,
    value::{
        ArgMemKind, BlockId, FunctionBody, FunctionId, ModuleView, ValueId, ValueRef, VarnodeId,
        insn::{InstructionId, Mnemonic},
    },
};
use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;

use crate::CallGraph;
use crate::calls::{CallEdge, CallTarget};
use crate::gvn::affine::Numbering;
use crate::sequence::{AddressRelation, MemoryAccess, collect_regions_for_base, relate_address};

use super::globals::global_slot;
use super::ram::OwnFrame;
use super::summary::{EffectChannel, EffectSummaries, solve_summaries};

/// Total entries (fields + regions) a summary may hold before it saturates.
/// Joins at call edges grow sets; saturation is the finite-height guarantee —
/// a saturated summary is treated as ⊤ by every consumer.
pub(crate) const MAX_EFFECT_ENTRIES: usize = 64;

// The footprint lattice value itself lives in `qcode` core — it is persisted on
// [`MemoryChannelState`](qcode::value::MemoryChannelState), and core cannot
// depend upward on this crate. Re-exported here under its historical path so
// this module (its only producer) reads unchanged.
pub(crate) use qcode::value::{Footprint, RamBase, RamField, RamObject, RamRegion};

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
    /// Provenance flag-set: every prototyped **external** whose whole-object
    /// argmem entries were folded into this summary — directly (a
    /// [`transfer`](EffectChannel::transfer) rebased the external's own object,
    /// on any landing base) or transitively (a callee summary that already
    /// carried the flag). It is **not** part of the footprint: it is exempt from
    /// [`MAX_EFFECT_ENTRIES`] saturation and is union-joined monotonically. Empty
    /// in [`external_leaf`](EffectChannel::external_leaf) — an external's own
    /// summary describes only its footprint; the flag is minted by the *caller*
    /// doing the rebase. At the promotion commit point each entry becomes an
    /// [`ExternalArgmemConfinement`](qcode::assumption::Proposition) assumption.
    pub externals: FxHashSet<FunctionId>,
}

impl Default for RamEffects {
    fn default() -> Self {
        Self {
            precise: Some(Footprint::default()),
            written: Some(Default::default()),
            externals: FxHashSet::default(),
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
        // Rule 5: a call argument that is a pointer into the caller's own private
        // (shadow/temp) space — what re-analysis of a shadow-rewritten body sees
        // for an admitted external's rebased own-frame pointer arg. The callee's
        // whole-object entries land on `Private` and are dropped from the outward
        // footprint (outward-invisible), still flagging the external.
        if ValueRef::new(arg, ctx)
            .memory_space()
            .is_some_and(|ms| ms.shared().is_none())
        {
            return Some((RamBase::Private, 0));
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
        if Some(base) == info.frame.entry_sp() {
            return None;
        }
        // A caller pointer param + const: composes positionally — but only if
        // the caller's own interface is lockstep, else the index is meaningless.
        if info.materialized
            && let Some(&j) = info.params.get(&base)
        {
            return Some((RamBase::Param(j), off));
        }
        // A loaded-pointer argument (a spilled incoming pointer param reloaded
        // from a frame slot) is NOT classified here: the forward-first pipeline
        // step (`argpromote-forward` / the `module(gvn)` after `assume_arg_frame`
        // in `mark-pure`) forwards such a reload back to its bare param under the
        // `LoadedPointerDisjointFromSlot` / `ArgsDisjointFromCallerFrame`
        // assumptions *before* the RAM solve runs, so the argument reaching this
        // classifier is already the param and lands on the `Param(j)` arm above.
        // Anything still loaded here (an own-frame spill of a local, a global, or
        // a reload the assumption did not license) is genuinely unrebasable — ⊤.
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

/// Own-frame read licenses minted by whole-object external *writes* (rule 1 of
/// externals-into-shadow). A prototyped external called with an `OutPtr`/`MutPtr`
/// argument that is one of the caller's own-frame locals writes that whole object;
/// under the [`ExternalArgmemConfinement`](qcode::assumption::Proposition)
/// assumption a subsequent same-base read of that local observes the external's
/// write (not the dead pre-call frame), so it does **not** refute freshness.
///
/// This licensing is deliberately unsound on its own (a read past the written
/// extent would still observe dead frame bytes) — it rests on the confinement
/// assumption, which is why a consumed license unions the external into the
/// summary's `externals` flag-set.
struct FrameLicenses {
    /// `(licensed own-frame pointer value, successor blocks of the licensing
    /// call, the prototyped external minting the license)`.
    entries: Vec<(ValueId, Vec<BlockId>, FunctionId)>,
    /// Dominator tree of the function; a read is licensed only in a block
    /// dominated by the licensing call's successor. `None` for a rootless body.
    doms: Option<DominatorTree<BlockId>>,
}

impl FrameLicenses {
    /// The external minting a license for an own-frame read at `read_ptr` in
    /// `read_block`, if any: a `licensed_ptr + const` address whose licensing
    /// call's successor dominates the read block. `None` = unlicensed.
    fn license_for(
        &self,
        ctx: &Context,
        numbering: &Numbering,
        read_ptr: ValueId,
        read_block: BlockId,
    ) -> Option<FunctionId> {
        let doms = self.doms.as_ref()?;
        for (lptr, succs, ext) in &self.entries {
            if !succs.iter().any(|s| doms.dominates(*s, read_block)) {
                continue;
            }
            if matches!(
                relate_address(ctx, numbering, *lptr, read_ptr, read_block),
                AddressRelation::Const(_)
            ) {
                return Some(*ext);
            }
        }
        None
    }
}

/// Collect the [`FrameLicenses`] for `fid`: one per own-frame-local `OutPtr`/
/// `MutPtr` argument of a prototyped-external call terminating a block.
fn build_frame_licenses(ctx: &Context, fid: FunctionId, frame: &OwnFrame) -> FrameLicenses {
    let function = FunctionBody::from_id(ctx, fid);
    let root = function.root().map(|b| b.id);
    let mut entries: Vec<(ValueId, Vec<BlockId>, FunctionId)> = Vec::new();
    for block in function.blocks() {
        let Some(term) = block.iter().last() else {
            continue;
        };
        let Mnemonic::Call(c) = term.mnemonic() else {
            continue;
        };
        let Some(ext) = c.target.real() else {
            continue;
        };
        let ef = FunctionBody::from_id(ctx, ext);
        if !ef.is_external() {
            continue;
        }
        let Some(argmem) = ef.argmem() else {
            continue;
        };
        if argmem.variadic {
            continue;
        }
        let succs: Vec<BlockId> = block.successors().map(|(_, s)| s).collect();
        for (i, kind) in argmem.params.iter().enumerate() {
            if !matches!(kind, ArgMemKind::OutPtr | ArgMemKind::MutPtr) {
                continue;
            }
            let Some(arg) = c.args.get(i) else {
                continue;
            };
            let arg = arg.qualify(term.id.func);
            if frame.is_local(ctx, arg) {
                entries.push((arg, succs.clone(), ext));
            }
        }
    }
    let doms = root.map(|r| compute_dominators(&function, r));
    FrameLicenses { entries, doms }
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
        // Read-after-external licensing (rule 1): built lazily on the first
        // unlicensed own-frame read, and the externals whose whole-object writes
        // license a read — folded into the summary's provenance flag-set, since
        // the non-⊤ verdict now rests on the confinement assumption.
        let mut licenses: Option<FrameLicenses> = None;
        let mut licensed_externals: FxHashSet<FunctionId> = FxHashSet::default();
        for block in FunctionBody::from_id(ctx, fid).blocks() {
            // Own-frame slots this block has stored to, in program order (see
            // the freshness licensing rationale in the module docs).
            let mut stored: FxHashSet<(i64, usize)> = FxHashSet::default();
            for insn in block.iter() {
                let (space, lptr, size, is_store) = match insn.mnemonic() {
                    Mnemonic::Load(l) => (l.space, l.ptr, l.size, false),
                    Mnemonic::Store(s) => (s.space, s.ptr, s.size, true),
                    Mnemonic::BranchInd(_) => {
                        // A surviving BranchInd is a genuine escape — a computed
                        // tail-jump into code the summary cannot see (jump-table
                        // resolution ran long before this). It unbounds the
                        // coarse channel *and* refutes the precise footprint: a
                        // function like `f(p) { goto [p] }` has no load/store of
                        // its own, but the code it jumps into may deref `p`
                        // arbitrarily, so it must not read as memory-free.
                        written = None;
                        precise_top = true;
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
                let frame: &OwnFrame =
                    own_frame.get_or_insert_with(|| OwnFrame::new(ctx, fid, self.sp));
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
                    // A whole-object external write may license this read (rule 1):
                    // a same-base read after a dominating `memset(&local)` observes
                    // the external's write, not the dead pre-call frame. Recorded
                    // in the externals flag-set because it rests on the confinement
                    // assumption.
                    let lic = licenses.get_or_insert_with(|| build_frame_licenses(ctx, fid, frame));
                    if let Some(ext) = lic.license_for(ctx, frame.numbering(), ptr, block.id) {
                        licensed_externals.insert(ext);
                        continue;
                    }
                    // Unlicensed own-frame read: observes the previous dead
                    // frame — the freshness hypothesis is refuted.
                    precise_top = true;
                    continue;
                }
                if frame.is_frame_slot(ctx, ptr) {
                    // A caller-frame slot (`@SP + k`, `k ≥ 0`): the return
                    // address / incoming stack arguments. Read/write asymmetry:
                    // a *read* of this interface region is invisible — the caller
                    // established that data and it is never redirected into the
                    // shadow (the frame stays real for the ABI), so its value is
                    // identical before and after promotion. It carries no new
                    // info to callers, so we neither ⊤ the summary nor push it
                    // outward. A *write* mutates an incoming arg slot — still
                    // promote_stack_args territory, not expressible here yet.
                    if is_store {
                        precise_top = true;
                    }
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
            // A bodied scan folds no callee externals through transfer, but a
            // read-after-external license *does* rest on the confinement
            // assumption, so its external is recorded here (rule 1).
            externals: licensed_externals,
        })
    }

    fn external_leaf(&self, ctx: &Context, fid: FunctionId) -> Option<RamEffects> {
        // An external can only touch memory *we* model through the pointers *we*
        // pass it — its own libc-internal state lives outside the lifted image.
        // So a prototyped external gets a real **argmem** footprint: one
        // whole-object effect per pointer parameter (write+read for a mutable
        // pointer, read-only for a `const` one), hung off the `Param(i)` that
        // pointer arrived on. `transfer` then rebases those objects through each
        // caller's actual argument (composing a caller pointer param, landing a
        // literal on `Global`, containing an own-frame-local write on `Frame`).
        //
        // Confinement is an ASSUMPTION: a whole-object write is taken to stay
        // within the addressed object (e.g. `memset`'s length in bounds), which
        // is what lets a frame-landing write be contained without extent
        // arithmetic. See `ExternalArgmemConfinement` in the module docs.
        let Some(argmem) = FunctionBody::from_id(ctx, fid).argmem() else {
            // No usable prototype: unbounded on both components (the pre-argmem
            // behavior). A caller shadow-promoted over such an external must not
            // pass the blocking gate — the external may write real ram through a
            // promoted pointer.
            return Some(RamEffects {
                precise: None,
                written: None,
                externals: FxHashSet::default(),
            });
        };
        // ⊤ carve-outs: a variadic external (`printf`'s `%n` / unknowable pointer
        // args) or any function-pointer/callback param (`qsort` re-enters our
        // code) — or a pointer the shallow model cannot bound (`char **`) —
        // defeats the argmem model entirely.
        if argmem.variadic
            || argmem
                .params
                .iter()
                .any(|k| matches!(k, ArgMemKind::Opaque))
        {
            return Some(RamEffects {
                precise: None,
                written: None,
                externals: FxHashSet::default(),
            });
        }
        let mut objects: std::collections::BTreeSet<RamObject> = Default::default();
        let mut any_write_ptr = false;
        for (i, kind) in argmem.params.iter().enumerate() {
            let base = RamBase::Param(i as u32);
            match kind {
                ArgMemKind::NonPtr => {}
                ArgMemKind::MutPtr => {
                    // Read+write mutable pointer (`strcat` dest, `realloc`): BOTH
                    // a write object AND a read object. The read half is what
                    // sends a frame-local landing to ⊤ in `transfer` (freshness),
                    // so a caller cannot memory-free-fold this through an
                    // uninitialized local.
                    objects.insert(RamObject { base, write: true });
                    objects.insert(RamObject { base, write: false });
                    any_write_ptr = true;
                }
                ArgMemKind::OutPtr => {
                    // Write-only destination (`memset`/`memcpy` dest, per the
                    // `extern_argmem` table): a pure whole-object write, no read
                    // half — a frame landing is contained (the `memset(&local)`
                    // fold).
                    objects.insert(RamObject { base, write: true });
                    any_write_ptr = true;
                }
                ArgMemKind::ConstPtr => {
                    objects.insert(RamObject { base, write: false });
                }
                // Filtered out above.
                ArgMemKind::Opaque => unreachable!(),
            }
        }
        // Pointers passed to externals target the default ram space (TLS/FS is
        // out of scope, per the existing conventions). A read-only footprint
        // writes nothing.
        let written = if any_write_ptr {
            Some([ctx.shared.default_space].into_iter().collect())
        } else {
            Some(std::collections::BTreeSet::new())
        };
        Some(RamEffects {
            precise: Some(Footprint {
                objects,
                ..Footprint::default()
            }),
            written,
            // An external's own summary describes only its footprint; the
            // confinement flag is minted by the caller that rebases it.
            externals: FxHashSet::default(),
        })
    }

    fn transfer(&self, ctx: &Context, edge: &CallEdge, callee: &RamEffects) -> Option<RamEffects> {
        // The coarse component moves by identity: space ids are global.
        let written = callee.written.clone();
        // Provenance propagation (rule 3): a callee's confinement flags always
        // rise into the caller — even when the callee is outward-invisible
        // (`Frame`-only) and contributes no footprint entry — so that
        // `g → f → memset` still stamps `g`'s summary with `{memset}`.
        let mut externals = callee.externals.clone();
        // The prototyped external this edge targets, if any — the only source of
        // whole-object entries. Rule 2 flags it below whenever one of its objects
        // is rebased into the caller's footprint, on any landing base.
        let callee_ext = match edge.target {
            CallTarget::Function(fid) if FunctionBody::from_id(ctx, fid).is_external() => Some(fid),
            _ => None,
        };
        let Some(fp) = &callee.precise else {
            return Some(RamEffects {
                precise: None,
                written,
                externals,
            });
        };
        // A positional site whose `args` are in `param[i] ↔ args[i]` lockstep
        // with the callee's root params can rebase `Param` entries. `Call`,
        // `TailCall`, and `Apply` all bind positionally against the callee's
        // root interface, so the same rebase logic applies unchanged.
        //
        // For a `TailCall` the caller's epilogue has already run before the tail
        // jump, so `@SP` at the site equals the caller's *entry* `@SP`. Any
        // `Frame(t)` offset a callee effect rebases through an own-frame-local
        // tail argument is therefore relative to that same entry `@SP`, and the
        // frame-containment check below (`t + offset + size > 0` rejects) stays
        // sound — a tail callee's frame overlaps the caller's already-dead frame
        // and its `Frame` writes die one level up exactly as a `Call`'s do.
        // (`classify_arg`'s own-frame view is value-based via `local_offset`, so
        // it does not depend on the runtime `@SP` value at the site regardless.)
        //
        // `Map`/`Scan` stay on the conservative branch: their element-wise
        // binding (`body(src[i])`) is NOT a positional lockstep against the
        // body's root params, so a `Param(i)` entry cannot be rebased through a
        // site argument.
        let call_args = edge
            .site
            .and_then(|site| match ctx.get_insn(site).mnemonic() {
                Mnemonic::Call(c) => Some(c.args.clone()),
                Mnemonic::TailCall(c) => Some(c.args.clone()),
                Mnemonic::Apply(a) => Some(a.args.clone()),
                _ => None,
            });
        let Some(args) = call_args else {
            return Some(RamEffects {
                precise: fp.invisible().then(Footprint::default),
                written,
                externals,
            });
        };
        let caller = edge.caller;
        // Whether an external's whole-object entry actually landed in the
        // caller's footprint below (rule 2 flags `callee_ext` only then).
        let mut ext_object_landed = false;
        // The precise rebase degrades to footprint-⊤ on its own (an
        // unrebasable argument never unbounds the coarse component).
        let precise = (|| -> Option<Footprint> {
            let mut out = Footprint::default();
            // Rebase one entry's base; `write`-ness gates the own-frame landing.
            let rebase = |base: RamBase, write: bool| -> Option<Option<(RamBase, i64)>> {
                Some(match base {
                    // Contained in the callee's own frame — invisible here.
                    RamBase::Frame(_) => None,
                    // Contained in the callee's own private space — invisible.
                    RamBase::Private => None,
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
            for o in &fp.objects {
                // A whole-object entry has no extent, so — unlike fields and
                // regions — there is NO containment/boundary arithmetic on a
                // frame landing: a callee's whole-object write into a caller
                // own-frame local is admitted outright under the confinement
                // assumption (the external writes only within the addressed
                // object, and the caller's frame local dies at return). See
                // stage 4 (`ExternalArgmemConfinement`) in the module docs. A
                // frame-landing object *read* is still rejected by `rebase`
                // (freshness), exactly like a field read — which is why a
                // read-write `MutPtr` external (both objects) goes ⊤ on a frame
                // landing while a write-only `OutPtr` one (write object only)
                // is contained.
                match rebase(o.base, o.write)? {
                    None => {}
                    Some((base, _shift)) => {
                        // A prototyped external's object was folded into the
                        // caller (any landing base) — the confinement assumption
                        // is now load-bearing here (rule 2).
                        ext_object_landed = true;
                        out.objects.insert(RamObject { base, ..*o });
                    }
                }
            }
            if out.len() > MAX_EFFECT_ENTRIES {
                return None;
            }
            Some(out)
        })();
        // Rule 2: stamp the external whose whole-object entry actually landed.
        if let Some(fid) = callee_ext
            && ext_object_landed
        {
            externals.insert(fid);
        }
        Some(RamEffects {
            precise,
            written,
            externals,
        })
    }

    fn join(&self, into: &mut RamEffects, from: &RamEffects) -> bool {
        let mut grew = false;
        // Precise: union under the entry budget; ⊤ absorbs.
        into.precise = match (into.precise.take(), &from.precise) {
            (Some(mut a), Some(b)) => {
                let before = a.len();
                a.fields.extend(b.fields.iter().copied());
                a.regions.extend(b.regions.iter().copied());
                a.objects.extend(b.objects.iter().copied());
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
        // Provenance flag-set: monotone union, exempt from the entry budget (it
        // is provenance, not footprint, so it never drives saturation to ⊤).
        let before = into.externals.len();
        into.externals.extend(from.externals.iter().copied());
        grew |= into.externals.len() != before;
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
        FunctionBody::from_id_mut(&mut tc.ctx, fid).set_register_effects(
            qcode::value::RegisterChannelState::Materialized(
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

    /// Make a bodyless external named `name` at `addr` and stamp it with the
    /// given prototype-derived argmem summary (as `external_sigs` would).
    fn external_argmem(
        tc: &mut qcode::testing::TestContext,
        addr: u64,
        name: &str,
        params: Vec<ArgMemKind>,
        variadic: bool,
    ) -> FunctionId {
        let f = qcode::value::FunctionBody::make_external(
            &mut tc.ctx,
            addr,
            Some(name.to_string().into()),
        )
        .id;
        FunctionBody::from_id_mut(&mut tc.ctx, f)
            .set_argmem(qcode::value::ExternArgmem { params, variadic });
        f
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
        let mut b = tc.ctx.builder(root);
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

    /// ITEM 1: a `TailCall` site rebases `Param` entries exactly like a `Call` —
    /// its args are positional-lockstep with the callee's root params. A callee
    /// that writes `Param(0)`, tail-called with a caller pointer param, lands a
    /// `Param(0)` write in the caller's summary.
    #[test]
    fn transfer_rebases_tailcall_args() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn callee:
                <c_entry @p:i64>
                    store(ram:8, @p <- i64 1);
                    return at i64 0;

            fn caller:
                <k_entry @r:i64>
                    tailcall fn callee(i64 @r);
            "
        );
        let _ = (c_entry, k_entry);
        materialize(&mut tc, callee);
        materialize(&mut tc, caller);
        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, None);
        let eff = s.get(caller).as_ref().expect("summary is expressible");
        let fp = eff.precise.as_ref().expect("tail rebase must succeed");
        let fields: Vec<RamField> = fp.fields.iter().copied().collect();
        assert_eq!(
            fields,
            vec![RamField {
                base: RamBase::Param(0),
                offset: 0,
                size: 8,
                write: true
            }],
            "the tail callee's Param(0) write rebases onto the caller's Param(0)"
        );
        assert!(!is_memory_free(&s, caller));
    }

    /// ITEM 1: an `Apply` site rebases `Param` entries exactly like a `Call` —
    /// its args are positional-lockstep with the callee's root params too.
    #[test]
    fn transfer_rebases_apply_args() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn callee:
                <c_entry @p:i64>
                    store(ram:8, @p <- i64 1);
                    return at i64 0;

            fn caller:
                <k_entry @r:i64>
                    %pack = apply callee(i64 @r);
                    return at i64 0;
            "
        );
        let _ = (c_entry, k_entry);
        materialize(&mut tc, callee);
        materialize(&mut tc, caller);
        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, None);
        let eff = s.get(caller).as_ref().expect("summary is expressible");
        let fp = eff.precise.as_ref().expect("apply rebase must succeed");
        let fields: Vec<RamField> = fp.fields.iter().copied().collect();
        assert_eq!(
            fields,
            vec![RamField {
                base: RamBase::Param(0),
                offset: 0,
                size: 8,
                write: true
            }],
            "the applied callee's Param(0) write rebases onto the caller's Param(0)"
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

    /// FIX 1: a surviving `BranchInd` (computed tail-jump) has no load/store of
    /// its own, but the code it jumps into may deref its pointer param
    /// arbitrarily. It must be footprint-⊤ (not memory-free) and coarse-⊤.
    #[test]
    fn branchind_on_param_is_not_memory_free() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn jump:
                <entry @p:i64>
                    goto [i64 @p];
            "
        );
        let _ = entry;
        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, None);
        assert!(
            !is_memory_free(&s, jump),
            "a surviving BranchInd is an escape into unseen code — not memory-free"
        );
        let eff = s.get(jump).as_ref().expect("summary");
        assert!(
            eff.precise.is_none(),
            "BranchInd refutes the precise footprint"
        );
        assert!(
            eff.written.is_none(),
            "BranchInd unbounds the coarse channel"
        );
    }

    /// FIX 2: an external's memory is fully-⊤ (a prototype bounds registers, not
    /// memory). A caller making a *clobber-free* RegPure call to an external —
    /// exactly what `rewrite_external_call_regpure` produces — must therefore not
    /// read as memory-free: the external (e.g. memset) may write real ram through
    /// a promoted pointer.
    #[test]
    fn clobber_free_external_call_blocks_caller() {
        let mut tc = qcode::testing::TestContext::new();
        let ext =
            qcode::value::FunctionBody::make_external(&mut tc.ctx, 0x9000, Some("memset".into()))
                .id;
        qcode!(
            tc.ctx,
            "
            fn caller:
                <k_entry @p:i64>
                    goto <k_call>;
                <k_call>
                    call fn caller();
                <k_cont>
                    return at i64 0;
            "
        );
        let _ = (k_entry, k_cont);
        materialize(&mut tc, caller);
        let p = FunctionBody::from_id(&tc.ctx, caller)
            .root()
            .unwrap()
            .params()
            .next()
            .unwrap()
            .id();
        // Redirect the placeholder call to the external as a clobber-free RegPure
        // site (regpure_call clears clobbers and tags RegPure).
        regpure_call(&mut tc, k_call, ext, vec![p]);
        tc.ctx.add_cfg_edge(k_call, k_cont);

        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, None);
        assert!(
            !is_memory_free(&s, caller),
            "a clobber-free RegPure call to an external must block the caller — \
             external memory is ⊤"
        );
    }

    /// STAGE 1: a whole-object entry rebases through the call arguments exactly
    /// like a field's base — a caller pointer param composes positionally, a
    /// literal lands on `Global`, and an own-frame-local write lands (contained)
    /// on `Frame`, all with **no** extent arithmetic.
    #[test]
    fn object_rebases_through_param_global_and_frame() {
        let mut tc = qcode::testing::TestContext::new();
        let sp = tc.r3;
        qcode!(
            tc.ctx,
            "
            fn callee:
                <c_entry @p:i64 @q:i64 @r:i64>
                    return at i64 0;

            fn caller:
                <k_entry @a:i64 @RSP:i64>
                    %loc = @RSP - i64 0x20;
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
        set_sp_origin(&mut tc, caller, sp);
        let lit = tc.ctx.get_const(0x4000, 8).id();
        let (a_param, loc) = {
            let f = FunctionBody::from_id(&tc.ctx, caller);
            let a_param = f.root().unwrap().params().next().unwrap().id();
            let loc = qcode::value::BasicBlock::from_id(&tc.ctx, k_entry)
                .iter()
                .find(|i| matches!(i.mnemonic(), Mnemonic::Binop(_)))
                .unwrap()
                .id;
            (a_param, ValueId::Instruction(loc))
        };
        let site = regpure_call(&mut tc, k_call, callee, vec![a_param, lit, loc]);
        tc.ctx.add_cfg_edge(k_call, k_cont);

        let ch = RamChannel::new(Some(sp));
        let callee_eff = RamEffects {
            precise: Some(Footprint {
                objects: [
                    RamObject {
                        base: RamBase::Param(0),
                        write: true,
                    },
                    RamObject {
                        base: RamBase::Param(1),
                        write: true,
                    },
                    RamObject {
                        base: RamBase::Param(2),
                        write: true,
                    },
                ]
                .into_iter()
                .collect(),
                ..Footprint::default()
            }),
            written: Some(Default::default()),
            externals: FxHashSet::default(),
        };
        let edge = CallEdge {
            caller,
            site: Some(site),
            target: crate::calls::CallTarget::Function(callee),
            kind: crate::calls::CallKind::Direct,
        };
        let out = ch
            .transfer(&tc.ctx, &edge, &callee_eff)
            .expect("transfer succeeds")
            .precise
            .expect("rebase succeeds");
        let mut objs: Vec<RamObject> = out.objects.iter().copied().collect();
        objs.sort_by_key(|o| o.base);
        assert_eq!(
            objs,
            vec![
                RamObject {
                    base: RamBase::Param(0),
                    write: true
                },
                RamObject {
                    base: RamBase::Frame(-0x20),
                    write: true
                },
                RamObject {
                    base: RamBase::Global(0x4000),
                    write: true
                },
            ]
        );
    }

    /// STAGE 1: a whole-object *read* rebased through an own-frame-local argument
    /// needs freshness licensing the transfer does not have — the same rule as a
    /// field read — so the caller's footprint is ⊤.
    #[test]
    fn frame_landing_object_read_is_top() {
        let mut tc = qcode::testing::TestContext::new();
        let sp = tc.r3;
        qcode!(
            tc.ctx,
            "
            fn callee:
                <c_entry @p:i64>
                    return at i64 0;

            fn caller:
                <k_entry @RSP:i64>
                    %loc = @RSP - i64 0x20;
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
        set_sp_origin(&mut tc, caller, sp);
        let loc = ValueId::Instruction(
            qcode::value::BasicBlock::from_id(&tc.ctx, k_entry)
                .iter()
                .find(|i| matches!(i.mnemonic(), Mnemonic::Binop(_)))
                .unwrap()
                .id,
        );
        let site = regpure_call(&mut tc, k_call, callee, vec![loc]);
        tc.ctx.add_cfg_edge(k_call, k_cont);

        let ch = RamChannel::new(Some(sp));
        let callee_eff = RamEffects {
            precise: Some(Footprint {
                objects: [RamObject {
                    base: RamBase::Param(0),
                    write: false,
                }]
                .into_iter()
                .collect(),
                ..Footprint::default()
            }),
            written: Some(Default::default()),
            externals: FxHashSet::default(),
        };
        let edge = CallEdge {
            caller,
            site: Some(site),
            target: crate::calls::CallTarget::Function(callee),
            kind: crate::calls::CallKind::Direct,
        };
        let out = ch
            .transfer(&tc.ctx, &edge, &callee_eff)
            .expect("transfer returns a summary");
        assert!(
            out.precise.is_none(),
            "a frame-landing object read must send the footprint to ⊤"
        );
    }

    /// STAGE 1: a footprint holding only a `Frame`-based whole-object *write* is
    /// outward-invisible (contained, dead at return).
    #[test]
    fn invisible_with_frame_write_object() {
        let mut fp = Footprint::default();
        fp.objects.insert(RamObject {
            base: RamBase::Frame(-0x20),
            write: true,
        });
        assert!(
            fp.invisible(),
            "a Frame-contained object write is outward-invisible"
        );
        let mut real = Footprint::default();
        real.objects.insert(RamObject {
            base: RamBase::Param(0),
            write: true,
        });
        assert!(
            !real.invisible(),
            "a Param-based object write is caller-observable"
        );
    }

    /// A memory READ never blocks promotion; only writes and address escapes do.
    /// A function that reads its own return address off a caller-frame slot
    /// (`load(@RSP)`, `k = 0`) — as every real lifted function does — plus a
    /// `memset(&local)` whose licensed read-back is contained, must stay
    /// outward-invisible. Before the read/write asymmetry the caller-frame read
    /// ⊤'d the summary and killed the fold. The negative half: a caller-frame
    /// *write* (mutating an incoming stack-arg slot, `k > 0`) still forces ⊤.
    #[test]
    fn caller_frame_read_does_not_block_memory_free() {
        let mut tc = qcode::testing::TestContext::new();
        let sp = tc.r3;
        let memset = external_argmem(
            &mut tc,
            0x9700,
            "memset",
            vec![ArgMemKind::OutPtr, ArgMemKind::NonPtr, ArgMemKind::NonPtr],
            false,
        );
        qcode!(
            tc.ctx,
            "
            fn reads_ok:
                <r_entry @RSP:i64>
                    %ret = load(ram:8, @RSP);
                    %loc = @RSP - i64 0x20;
                    goto <r_call>;
                <r_call>
                    call fn reads_ok();
                <r_cont>
                    %v = load(ram:1, %loc);
                    return at %ret;

            fn writes_arg:
                <w_entry @RSP:i64>
                    %slot = @RSP + i64 0x8;
                    store(ram:8, %slot <- i64 0x0);
                    return at i64 0;
            "
        );
        let _ = (r_entry, r_call, r_cont, w_entry);
        for fid in [reads_ok, writes_arg] {
            materialize(&mut tc, fid);
            set_sp_origin(&mut tc, fid, sp);
        }
        let loc = ValueId::Instruction(
            qcode::value::BasicBlock::from_id(&tc.ctx, r_entry)
                .iter()
                .filter(|i| matches!(i.mnemonic(), Mnemonic::Binop(_)))
                .last()
                .unwrap()
                .id,
        );
        regpure_call(&mut tc, r_call, memset, vec![loc]);
        tc.ctx.add_cfg_edge(r_call, r_cont);

        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, Some(sp));

        assert!(
            is_memory_free(&s, reads_ok),
            "a caller-frame-slot read (the return address) is invisible; with the \
             memset read-back licensed the function stays outward-invisible"
        );
        assert!(
            !is_memory_free(&s, writes_arg),
            "a caller-frame-slot *write* mutates an incoming arg slot — still ⊤"
        );
    }

    /// STAGE 2 / ruling 1b: a `memset`-like external whose dest is a **write-only**
    /// (`OutPtr`) param gets a single `Param(0)` whole-object write, and a caller
    /// whose only effect is calling it through an own-frame local composes to
    /// memory-free — the `memset(&local)` fold, the whole payoff. (`OutPtr` is
    /// what the `extern_argmem` write-only table upgrades memset's dest to; a bare
    /// `MutPtr` would carry a read half and block — see
    /// `strcat_mutptr_frame_local_is_top`.)
    #[test]
    fn memset_argmem_frame_local_call_is_memory_free() {
        let mut tc = qcode::testing::TestContext::new();
        let sp = tc.r3;
        let memset = external_argmem(
            &mut tc,
            0x9000,
            "memset",
            vec![ArgMemKind::OutPtr, ArgMemKind::NonPtr, ArgMemKind::NonPtr],
            false,
        );
        qcode!(
            tc.ctx,
            "
            fn caller:
                <k_entry @RSP:i64>
                    %loc = @RSP - i64 0x20;
                    goto <k_call>;
                <k_call>
                    call fn caller();
                <k_cont>
                    return at i64 0;
            "
        );
        let _ = (k_entry, k_cont);
        materialize(&mut tc, caller);
        set_sp_origin(&mut tc, caller, sp);
        let loc = ValueId::Instruction(
            qcode::value::BasicBlock::from_id(&tc.ctx, k_entry)
                .iter()
                .find(|i| matches!(i.mnemonic(), Mnemonic::Binop(_)))
                .unwrap()
                .id,
        );
        regpure_call(&mut tc, k_call, memset, vec![loc]);
        tc.ctx.add_cfg_edge(k_call, k_cont);

        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, Some(sp));

        // The external's own footprint is a single Param(0) write object.
        let ext = s.get(memset).as_ref().expect("external summary");
        let fp = ext
            .precise
            .as_ref()
            .expect("argmem footprint is expressible");
        assert_eq!(
            fp.objects.iter().copied().collect::<Vec<_>>(),
            vec![RamObject {
                base: RamBase::Param(0),
                write: true
            }]
        );
        assert!(
            ext.written
                .as_ref()
                .is_some_and(|w| w.contains(&tc.ctx.shared.default_space)),
            "a mutable-pointer external writes the default ram space"
        );
        assert!(
            !is_memory_free(&s, memset),
            "the external's own write is real"
        );
        assert!(
            is_memory_free(&s, caller),
            "memset(&local) into the caller's own frame is contained — memory-free"
        );
    }

    /// Ruling 1b: a `strcat`-like external whose dest is a read-write `MutPtr`
    /// carries BOTH a write object and a read object. The read object rebased
    /// through an own-frame local needs freshness licensing the transfer lacks,
    /// so a caller passing an own-frame local composes to ⊤ — NOT memory-free.
    /// This is the uninit-local-read hole the write-only/read-write split closes:
    /// `strcat(&local, s)` reads `&local`'s stale contents.
    #[test]
    fn strcat_mutptr_frame_local_is_top() {
        let mut tc = qcode::testing::TestContext::new();
        let sp = tc.r3;
        let strcat = external_argmem(
            &mut tc,
            0x9500,
            "strcat",
            vec![ArgMemKind::MutPtr, ArgMemKind::ConstPtr],
            false,
        );
        qcode!(
            tc.ctx,
            "
            fn caller:
                <k_entry @RSP:i64>
                    %loc = @RSP - i64 0x20;
                    goto <k_call>;
                <k_call>
                    call fn caller();
                <k_cont>
                    return at i64 0;
            "
        );
        let _ = (k_entry, k_cont);
        materialize(&mut tc, caller);
        set_sp_origin(&mut tc, caller, sp);
        let loc = ValueId::Instruction(
            qcode::value::BasicBlock::from_id(&tc.ctx, k_entry)
                .iter()
                .find(|i| matches!(i.mnemonic(), Mnemonic::Binop(_)))
                .unwrap()
                .id,
        );
        // Pass the frame local as the mutable dest; a literal as the const src.
        let lit = tc.ctx.get_const(0x4000, 8).id();
        regpure_call(&mut tc, k_call, strcat, vec![loc, lit]);
        tc.ctx.add_cfg_edge(k_call, k_cont);

        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, Some(sp));

        // The external's own footprint is a Param(0) write AND a Param(0) read.
        let ext = s.get(strcat).as_ref().expect("external summary");
        let fp = ext
            .precise
            .as_ref()
            .expect("argmem footprint is expressible");
        assert!(
            fp.objects.contains(&RamObject {
                base: RamBase::Param(0),
                write: true,
            }) && fp.objects.contains(&RamObject {
                base: RamBase::Param(0),
                write: false,
            }),
            "a MutPtr dest carries both a write and a read object"
        );
        let caller_eff = s.get(caller).as_ref().expect("caller summary");
        assert!(
            caller_eff.precise.is_none(),
            "the MutPtr read through a frame local needs freshness licensing — ⊤"
        );
        assert!(
            !is_memory_free(&s, caller),
            "strcat(&local, s) reads the stale frame local — not memory-free"
        );
    }

    /// STAGE 2: a `strlen`-like external (one `const` pointer) gets a read-only
    /// object; a caller passing an own-frame local composes to ⊤ — a callee read
    /// through a frame pointer needs freshness licensing the transfer lacks.
    #[test]
    fn strlen_argmem_const_ptr_frame_local_is_top() {
        let mut tc = qcode::testing::TestContext::new();
        let sp = tc.r3;
        let strlen = external_argmem(&mut tc, 0x9100, "strlen", vec![ArgMemKind::ConstPtr], false);
        qcode!(
            tc.ctx,
            "
            fn caller:
                <k_entry @RSP:i64>
                    %loc = @RSP - i64 0x20;
                    goto <k_call>;
                <k_call>
                    call fn caller();
                <k_cont>
                    return at i64 0;
            "
        );
        let _ = (k_entry, k_cont);
        materialize(&mut tc, caller);
        set_sp_origin(&mut tc, caller, sp);
        let loc = ValueId::Instruction(
            qcode::value::BasicBlock::from_id(&tc.ctx, k_entry)
                .iter()
                .find(|i| matches!(i.mnemonic(), Mnemonic::Binop(_)))
                .unwrap()
                .id,
        );
        regpure_call(&mut tc, k_call, strlen, vec![loc]);
        tc.ctx.add_cfg_edge(k_call, k_cont);

        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, Some(sp));

        let ext = s.get(strlen).as_ref().expect("external summary");
        let fp = ext
            .precise
            .as_ref()
            .expect("argmem footprint is expressible");
        assert_eq!(
            fp.objects.iter().copied().collect::<Vec<_>>(),
            vec![RamObject {
                base: RamBase::Param(0),
                write: false
            }]
        );
        assert!(
            ext.written.as_ref().is_some_and(|w| w.is_empty()),
            "a read-only external writes nothing"
        );
        let caller_eff = s.get(caller).as_ref().expect("caller summary");
        assert!(
            caller_eff.precise.is_none(),
            "a const-ptr read through a frame local needs freshness licensing — ⊤"
        );
        assert!(!is_memory_free(&s, caller));
    }

    /// STAGE 2: a `qsort`-like external with a function-pointer (callback)
    /// parameter is ⊤ — it re-enters our code and can touch anything.
    #[test]
    fn qsort_argmem_callback_is_top() {
        let mut tc = qcode::testing::TestContext::new();
        let qsort = external_argmem(
            &mut tc,
            0x9200,
            "qsort",
            vec![
                ArgMemKind::MutPtr,
                ArgMemKind::NonPtr,
                ArgMemKind::NonPtr,
                ArgMemKind::Opaque,
            ],
            false,
        );
        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, None);
        let ext = s.get(qsort).as_ref().expect("external summary");
        assert!(
            ext.precise.is_none(),
            "a callback param sends the footprint to ⊤"
        );
        assert!(ext.written.is_none(), "and the coarse channel to ⊤");
        assert!(!is_memory_free(&s, qsort));
    }

    /// STAGE 2: an `abs`-like external with no pointer parameters has an empty
    /// footprint — memory-free — and writes nothing.
    #[test]
    fn abs_argmem_no_pointer_is_memory_free() {
        let mut tc = qcode::testing::TestContext::new();
        let abs = external_argmem(&mut tc, 0x9300, "abs", vec![ArgMemKind::NonPtr], false);
        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, None);
        let ext = s.get(abs).as_ref().expect("external summary");
        let fp = ext
            .precise
            .as_ref()
            .expect("empty footprint is expressible");
        assert!(
            fp.objects.is_empty(),
            "no pointer params ⇒ no argmem objects"
        );
        assert!(
            ext.written.as_ref().is_some_and(|w| w.is_empty()),
            "a pointer-free external writes nothing"
        );
        assert!(
            is_memory_free(&s, abs),
            "a pointer-free external is memory-free"
        );
    }

    /// STAGE 2: an external with no argmem stamp (un-prototyped) stays fully-⊤ —
    /// the pre-argmem behavior a shadow-promoted caller must not pass the gate
    /// through.
    #[test]
    fn unprototyped_external_stays_top() {
        let mut tc = qcode::testing::TestContext::new();
        let ext =
            qcode::value::FunctionBody::make_external(&mut tc.ctx, 0x9400, Some("mystery".into()))
                .id;
        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, None);
        let e = s.get(ext).as_ref().expect("external summary");
        assert!(e.precise.is_none(), "no prototype ⇒ precise ⊤");
        assert!(e.written.is_none(), "no prototype ⇒ written ⊤");
    }

    /// FIX 6a: joining two footprints past `MAX_EFFECT_ENTRIES` saturates to ⊤ —
    /// a saturated summary is never memory-free.
    #[test]
    fn saturation_becomes_top() {
        let ch = RamChannel::new(None);
        let mk = |range: std::ops::Range<i64>| {
            let mut fp = Footprint::default();
            for offset in range {
                fp.fields.insert(RamField {
                    base: RamBase::Param(0),
                    offset,
                    size: 1,
                    write: false,
                });
            }
            RamEffects {
                precise: Some(fp),
                written: Some(Default::default()),
                externals: FxHashSet::default(),
            }
        };
        let mut into = mk(0..40);
        let from = mk(40..80);
        ch.join(&mut into, &from);
        assert!(
            into.precise.is_none(),
            "80 distinct entries overflow MAX_EFFECT_ENTRIES ({MAX_EFFECT_ENTRIES}) → ⊤"
        );
        assert!(
            !into.outward_invisible(),
            "a saturated summary is not memory-free"
        );
    }

    /// FIX 6b: a self-recursive function that derefs a pointer param and recurses
    /// on `param + const` must reach a sound fixpoint (the solve terminates) and
    /// carry a real footprint — not memory-free.
    #[test]
    fn recursive_rebasing_terminates_and_blocks() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn rec:
                <rec_entry @p:i64>
                    %v = load(ram:8, @p);
                    %next = @p + i64 0x8;
                    goto <rec_call>;
                <rec_call>
                    call fn rec();
                <rec_cont>
                    return at %v;
            "
        );
        let _ = (rec_entry, rec_cont);
        materialize(&mut tc, rec);
        let next = {
            let block = qcode::value::BasicBlock::from_id(&tc.ctx, rec_entry);
            block
                .iter()
                .find(|i| matches!(i.mnemonic(), Mnemonic::Binop(_)))
                .unwrap()
                .id
        };
        regpure_call(&mut tc, rec_call, rec, vec![ValueId::Instruction(next)]);
        tc.ctx.add_cfg_edge(rec_call, rec_cont);

        let graph = CallGraph::analyze(&tc.ctx);
        // Terminates (returning at all is the fixpoint assertion) and blocks.
        let s = solve(&tc.ctx, &graph, None);
        assert!(
            !is_memory_free(&s, rec),
            "the recursive pointer deref is a real footprint — not memory-free"
        );
    }

    /// RULE 1: a whole-object `memset(&local)` write licenses a subsequent
    /// same-base own-frame read in a dominated block — the summary is non-⊤,
    /// outward-invisible, and records `memset` in its `externals` flag-set. The
    /// same read WITHOUT the preceding memset stays ⊤ (uninit refutation).
    #[test]
    fn read_after_external_write_is_licensed() {
        let mut tc = qcode::testing::TestContext::new();
        let sp = tc.r3;
        let memset = external_argmem(
            &mut tc,
            0x9000,
            "memset",
            vec![ArgMemKind::OutPtr, ArgMemKind::NonPtr, ArgMemKind::NonPtr],
            false,
        );
        qcode!(
            tc.ctx,
            "
            fn caller:
                <k_entry @RSP:i64>
                    %loc = @RSP - i64 0x20;
                    goto <k_call>;
                <k_call>
                    call fn caller();
                <k_cont>
                    %v = load(ram:8, %loc);
                    return at %v;
            "
        );
        let _ = (k_entry, k_call, k_cont);
        materialize(&mut tc, caller);
        set_sp_origin(&mut tc, caller, sp);
        let loc = ValueId::Instruction(
            qcode::value::BasicBlock::from_id(&tc.ctx, k_entry)
                .iter()
                .find(|i| matches!(i.mnemonic(), Mnemonic::Binop(_)))
                .unwrap()
                .id,
        );
        regpure_call(&mut tc, k_call, memset, vec![loc]);
        tc.ctx.add_cfg_edge(k_call, k_cont);

        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, Some(sp));
        assert!(
            is_memory_free(&s, caller),
            "the licensed read + contained memset write is outward-invisible"
        );
        let eff = s.get(caller).as_ref().expect("caller summary");
        assert!(
            eff.externals.contains(&memset),
            "a consumed read license records the external in the flag-set"
        );
    }

    /// RULE 1 negative: the identical own-frame read with NO preceding memset
    /// observes the dead frame — unlicensed, ⊤ (freshness refuted). Guards that
    /// licensing did not weaken the uninit refutation.
    #[test]
    fn read_without_external_write_is_top() {
        let mut tc = qcode::testing::TestContext::new();
        let sp = tc.r3;
        qcode!(
            tc.ctx,
            "
            fn caller:
                <k_entry @RSP:i64>
                    %loc = @RSP - i64 0x20;
                    %v = load(ram:8, %loc);
                    return at %v;
            "
        );
        let _ = k_entry;
        materialize(&mut tc, caller);
        set_sp_origin(&mut tc, caller, sp);
        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, Some(sp));
        assert!(
            !is_memory_free(&s, caller),
            "an unlicensed own-frame read is ⊤"
        );
    }

    /// RULE 1: a read on a path that does NOT flow through the `memset` call is
    /// not dominated by the call's successor — unlicensed, ⊤.
    #[test]
    fn non_dominated_read_is_top() {
        let mut tc = qcode::testing::TestContext::new();
        let sp = tc.r3;
        let memset = external_argmem(
            &mut tc,
            0x9000,
            "memset",
            vec![ArgMemKind::OutPtr, ArgMemKind::NonPtr, ArgMemKind::NonPtr],
            false,
        );
        qcode!(
            tc.ctx,
            "
            fn caller:
                <k_entry @RSP:i64 @c:bool>
                    %loc = @RSP - i64 0x20;
                    if @c goto <k_call> else goto <k_read>;
                <k_call>
                    call fn caller();
                <k_cont>
                    return at i64 0;
                <k_read>
                    %v = load(ram:8, %loc);
                    return at %v;
            "
        );
        let _ = (k_entry, k_call, k_cont, k_read);
        materialize(&mut tc, caller);
        set_sp_origin(&mut tc, caller, sp);
        let loc = ValueId::Instruction(
            qcode::value::BasicBlock::from_id(&tc.ctx, k_entry)
                .iter()
                .find(|i| matches!(i.mnemonic(), Mnemonic::Binop(_)))
                .unwrap()
                .id,
        );
        regpure_call(&mut tc, k_call, memset, vec![loc]);
        tc.ctx.add_cfg_edge(k_call, k_cont);

        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, Some(sp));
        assert!(
            !is_memory_free(&s, caller),
            "a read the memset call does not dominate is unlicensed — ⊤"
        );
    }

    /// FIX 6c: the `written` component tracks ram stores and unbounds on escapes,
    /// and an external leaf is coarse-⊤.
    /// Forward-first (stage 2b): a caller that spills an incoming pointer param
    /// to an own-frame slot, reloads it, and passes the reload to a memory-writing
    /// callee is ⊤ on the raw IR (the loaded-pointer argument is unclassifiable),
    /// but once the pipeline's `argpromote-forward` `gvn` (built with
    /// frame-freshness under the recorded assumptions) forwards the reload back to
    /// the bare param, the argument reaching `classify_arg` is the param itself and
    /// the callee's `Param(0)` write composes onto the caller's `Param(1)`.
    #[test]
    fn spilled_param_reload_composes_after_forward() {
        use crate::alias::AliasResult;
        use crate::gvn::gvn_function;
        use qcode::assumption::Proposition;
        use qcode::value::ModuleView;

        let mut tc = qcode::testing::TestContext::new();
        let sp = tc.r3;
        qcode!(
            tc.ctx,
            "
            fn callee:
                <c_entry @p:i64>
                    store(ram:8, @p <- i64 1);
                    return at i64 0;

            fn caller:
                <k_entry @RSP:i64 @buf_in:i64 @other:i64>
                    %slot = @RSP - i64 0x8;
                    store(ram:8, %slot <- @buf_in);
                    store(ram:8, @other <- i64 0x99);
                    %buf = load(ram:8, %slot);
                    call fn caller();
                <k_cont>
                    return at i64 0;
            "
        );
        let _ = (c_entry, k_entry, k_cont);
        materialize(&mut tc, callee);
        materialize(&mut tc, caller);
        set_sp_origin(&mut tc, caller, sp);
        // The spilled reload passed as the (only) call argument.
        let buf = ValueId::Instruction(
            qcode::value::BasicBlock::from_id(&tc.ctx, k_entry)
                .iter()
                .find(|i| matches!(i.mnemonic(), Mnemonic::Load(_)))
                .unwrap()
                .id,
        );
        regpure_call(&mut tc, k_entry, callee, vec![buf]);
        tc.ctx.add_cfg_edge(k_entry, k_cont);

        // Baseline: on the raw IR the argument is a loaded pointer — unclassifiable,
        // so the transfer degrades the whole caller summary to ⊤.
        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, Some(sp));
        assert!(
            s.get(caller)
                .as_ref()
                .expect("caller summary")
                .precise
                .is_none(),
            "raw spilled-pointer reload is ⊤ (loaded-pointer argument unrebasable)"
        );

        // Forward-first: record the assumptions `assume_arg_frame` records, build
        // the frame-freshness alias oracle, and run `gvn` — exactly the
        // `argpromote-forward` pipeline step. This forwards `%buf` back to
        // `@buf_in`, the caller's `Param(1)`.
        tc.ctx
            .assume_true(Proposition::ArgsDisjointFromCallerFrame(caller));
        tc.ctx
            .assume_true(Proposition::LoadedPointerDisjointFromSlot(caller));
        let aliases = AliasResult::simple_for_function(&tc.ctx, caller).with_frame_freshness(
            ModuleView::new(&tc.ctx),
            caller,
            Some(sp),
        );
        gvn_function(&mut tc.ctx, caller, Some(&aliases));

        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, Some(sp));
        let eff = s.get(caller).as_ref().expect("caller summary");
        let fp = eff
            .precise
            .as_ref()
            .expect("the reload forwarded to the param — no longer ⊤");
        assert!(
            fp.fields.contains(&RamField {
                base: RamBase::Param(1),
                offset: 0,
                size: 8,
                write: true,
            }),
            "the callee's Param(0) write composes onto the caller's Param(1) \
             (the spilled `@buf_in`): {fp:?}"
        );
        assert!(
            !is_memory_free(&s, caller),
            "the composed footprint is a real (non-⊤) caller effect"
        );
    }

    #[test]
    fn written_component_tracks_ram_and_escapes() {
        let mut tc = qcode::testing::TestContext::new();
        let ext =
            qcode::value::FunctionBody::make_external(&mut tc.ctx, 0x9000, Some("ext".into())).id;
        qcode!(
            tc.ctx,
            "
            fn writer:
                <w_entry @p:i64>
                    store(ram:8, @p <- i64 1);
                    return at i64 0;
            fn jumper:
                <j_entry @p:i64>
                    goto [i64 @p];
            "
        );
        let _ = (w_entry, j_entry);
        let ram = tc.ctx.shared.default_space;
        let graph = CallGraph::analyze(&tc.ctx);
        let s = solve(&tc.ctx, &graph, None);

        let w = s.get(writer).as_ref().expect("writer summary");
        assert!(
            w.written.as_ref().is_some_and(|set| set.contains(&ram)),
            "a ram store lands in the written component"
        );
        let j = s.get(jumper).as_ref().expect("jumper summary");
        assert!(j.written.is_none(), "BranchInd ⇒ written = None");
        let e = s.get(ext).as_ref().expect("external summary");
        assert!(e.written.is_none(), "external_leaf ⇒ written = None");
        assert!(e.precise.is_none(), "external_leaf ⇒ precise = ⊤");
    }
}
