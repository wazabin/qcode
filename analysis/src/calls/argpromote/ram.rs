use std::borrow::Cow;
use std::collections::HashSet;

use rustc_hash::FxHashSet;

use qcode::{
    assumption::Proposition,
    builder::Builder,
    context::Context,
    space::{LocalMemorySpaceId, MemorySpaceId},
    types::TypeId,
    value::{
        ArgMemKind, BasicBlock, FunctionBody, FunctionId, TempSpace, TempSpaceId, Value, ValueId,
        VarnodeId,
        insn::{Call, InstructionId, Mnemonic},
    },
};

use crate::gvn::affine::{Numbering, precompute_forms};
use crate::sequence::{
    AddressRelation, MemoryAccess, SequenceRegion, collect_regions_for_base, relate_address,
};
use crate::stack::frame::{FrameClass, frame_class, incoming_sp_param};
use crate::{Pass, PipelineEnv};

use super::arg_index_of;

/// Recognises this function's own stack-frame locals (`@SP`-rooted slots below the
/// entry stack pointer). Inert when there is no stack-pointer register or no
/// incoming `@SP` param, in which case [`OwnFrame::is_local`] is always `false`.
pub(super) struct OwnFrame {
    sp_param: Option<ValueId>,
    numbering: Numbering,
}

impl OwnFrame {
    pub(super) fn new(ctx: &Context, fid: FunctionId, sp_reg: Option<VarnodeId>) -> Self {
        let sp_param =
            sp_reg.and_then(|r| incoming_sp_param(qcode::value::ModuleView::new(ctx), fid, r));
        Self {
            sp_param,
            numbering: precompute_forms(qcode::value::ModuleView::new(ctx), fid),
        }
    }

    /// Whether `addr` points into this function's own frame (classified
    /// [`FrameClass::Local`]).
    pub(super) fn is_local(&self, ctx: &Context, addr: ValueId) -> bool {
        self.sp_param.is_some_and(|sp| {
            frame_class(
                qcode::value::ModuleView::new(ctx),
                &self.numbering,
                sp,
                addr,
            ) == Some(FrameClass::Local)
        })
    }

    /// The affine numbering backing this frame view, shared with callers that
    /// need `base_offset` decomposition over the same function.
    pub(super) fn numbering(&self) -> &Numbering {
        &self.numbering
    }

    /// The incoming `@SP` param this frame is rooted at, if recognised.
    pub(super) fn sp_param(&self) -> Option<ValueId> {
        self.sp_param
    }

    /// The stable slot key of an own-frame local: `addr`'s constant offset from
    /// the incoming `@SP`, if `addr` is a plain `@SP - k` local. `None` for
    /// non-locals *and* for realigned-base locals (`@SP & -mask` has no stable
    /// `@SP`-relative offset, so such a slot can never license a read).
    pub(super) fn local_offset(&self, ctx: &Context, addr: ValueId) -> Option<i64> {
        let sp = self.sp_param?;
        let off = crate::stack::frame::frame_offset(
            qcode::value::ModuleView::new(ctx),
            &self.numbering,
            sp,
            addr,
        )?;
        (off < 0).then_some(off)
    }

    /// Whether `addr` is any `@SP`-rooted frame slot — an own-frame local or a
    /// caller-frame slot (`@SP + k`, `k ≥ 0`).
    pub(super) fn is_frame_slot(&self, ctx: &Context, addr: ValueId) -> bool {
        self.sp_param.is_some_and(|sp| {
            matches!(
                frame_class(
                    qcode::value::ModuleView::new(ctx),
                    &self.numbering,
                    sp,
                    addr,
                ),
                Some(FrameClass::Local | FrameClass::CallerFrame)
            )
        })
    }

    /// Whether `addr` is specifically a *caller-frame* slot (`@SP + k`, `k ≥ 0`):
    /// the return address / incoming stack args, established by the caller and
    /// never redirected into the shadow.
    pub(super) fn is_caller_frame_slot(&self, ctx: &Context, addr: ValueId) -> bool {
        self.sp_param.is_some_and(|sp| {
            frame_class(
                qcode::value::ModuleView::new(ctx),
                &self.numbering,
                sp,
                addr,
            ) == Some(FrameClass::CallerFrame)
        })
    }
}

/// Promotes every eligible by-reference in/out parameter in the module. Returns
/// `true` if anything changed.
///
/// Without a stack-pointer register the frame-deadness rule is inert; production
/// callers use [`argpromote_with_sp`].
pub fn argpromote(ctx: &mut Context) -> bool {
    argpromote_with_sp(ctx, None)
}

/// [`argpromote`] with the stack-pointer register `sp_reg`, so a function's own
/// stack frame is recognised and **excluded** from the returned write-set: own-
/// frame locals are destroyed at return, so the caller can never observe writes to
/// them (they are dead on exit). The redirected shadow store is left to DCE.
pub fn argpromote_with_sp(ctx: &mut Context, sp_reg: Option<VarnodeId>) -> bool {
    let targets = ctx.function_ids();
    let graph = crate::CallGraph::analyze(ctx);
    !argpromote_changed_functions_with_sp(ctx, sp_reg, &targets, &graph).is_empty()
}

fn argpromote_changed_functions_with_sp(
    ctx: &mut Context,
    sp_reg: Option<VarnodeId>,
    targets: &[FunctionId],
    graph: &crate::CallGraph,
) -> FxHashSet<FunctionId> {
    let target_set: FxHashSet<_> = targets.iter().copied().collect();
    let mut changed = FxHashSet::default();
    // Both channels gate every function on being address-taken; build that set once
    // (O(instructions)) instead of rescanning the whole program per function. It
    // stays valid across the loop: promotion threads only data values, never adding
    // a `ValueId::Function` operand. See [`super::address_taken_set`].
    let address_taken = super::address_taken_set(ctx);
    // A caller may keep a call only to a transitively memory-free callee (see
    // [`function_makes_blocking_call`]). That fact is solved on the effect
    // engine *before* any mutation, so the gate is independent of the visit
    // order below. (The sweep's own mutations — global slot growth, access
    // redirection into shadow — never make a memory-touching function
    // memory-free within this run, so the pre-solved answer stays valid; a
    // freshly promoted callee unblocks its callers on the pipeline's next
    // round, after cleanup drops its shadow accesses, exactly as before.)
    let ram_summaries = super::ram_summary::solve(ctx, graph, sp_reg);
    // Functions whose complete footprint the shadow path absorbed *this sweep*.
    // Their bodies are already access-free in shared spaces, and their call-site
    // rewrites have landed the footprint as ordinary accesses in each caller's
    // body — which the caller's own scan captures when its (later) visit comes.
    // Next run the summary itself sees the shadow-only body as invisible; this
    // set only bridges the in-sweep window.
    let mut absorbed: FxHashSet<FunctionId> = FxHashSet::default();
    // Callee-before-caller order kept for the apply side: a promoted callee's
    // call-site rewrites land before its caller is visited. One visit per
    // function (no fixpoint), so an already-promoted body is never re-promoted.
    let order = callee_first_order(ctx, graph);
    for fid in order {
        if !target_set.contains(&fid) {
            continue;
        }
        let callers = graph.callers(fid);
        if callers.iter().any(|id| !target_set.contains(id)) {
            continue;
        }
        // Globals (constant-address real-ram accesses) are materialized by the
        // RAM channel itself inside `try_promote`/`apply` (grow_globals retired).
        let outcome = try_promote(
            ctx,
            fid,
            sp_reg,
            &address_taken,
            graph,
            &ram_summaries,
            &absorbed,
        );
        // Any committed promotion (Shadow or Partial) that folded through a
        // prototyped external's argmem footprint rests on the confinement
        // assumption: record one `ExternalArgmemConfinement(f, e)` per external
        // `e` in `f`'s solved summary flag-set, so a refutation invalidates `f`
        // (checkpoint+replay). `No` changes nothing, so it registers nothing.
        if outcome != Promotion::No
            && let Ok(eff) = ram_summaries.get(fid)
        {
            let externals: Vec<FunctionId> = eff.externals.iter().copied().collect();
            for e in externals {
                ctx.assume_true(Proposition::ExternalArgmemConfinement(fid, e));
            }
        }
        match outcome {
            Promotion::No => {}
            Promotion::Partial => {
                changed.insert(fid);
                changed.extend(callers);
            }
            Promotion::Shadow => {
                // The full shadow path absorbed this function's entire real-ram
                // footprint: its remaining accesses live in its private shadow,
                // so calls to it are inert for every caller visited later in
                // this same sweep (see [`function_makes_blocking_call`]).
                absorbed.insert(fid);
                changed.insert(fid);
                changed.extend(callers);
            }
        }
    }
    changed
}

/// Functions in callee-before-caller (reverse-topological / DFS post) order, each
/// once. Recursion and cycles are handled by the visited set: a function in a
/// cycle is emitted once, which is fine — recursion is out of scope for purity, so
/// such a caller simply fails the blocking-call gate.
fn callee_first_order(ctx: &Context, graph: &crate::CallGraph) -> Vec<FunctionId> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    for root in ctx.function_ids() {
        if visited.contains(&root) {
            continue;
        }
        // (fid, children_expanded) — the marker entry emits in post order.
        let mut stack = vec![(root, false)];
        while let Some((fid, expanded)) = stack.pop() {
            if expanded {
                order.push(fid);
                continue;
            }
            if !visited.insert(fid) {
                continue;
            }
            stack.push((fid, true));
            for callee in graph.callees(fid) {
                if !visited.contains(&callee) {
                    stack.push((callee, false));
                }
            }
        }
    }
    order
}

/// One scalar read through a promoted pointer: a load of `size` bytes at a fixed
/// constant byte `offset` from the base. Each becomes its own by-value snapshot
/// parameter (seeded into the shadow at `base + offset`) and its own caller-side
/// `load(arg + offset, size)`. Keying the snapshot by `(offset, size)` — rather
/// than snapshotting one contiguous `[base, base+offset+size)` region — is what
/// lets a deref at a large fixed offset (e.g. a segment-relative `FS:0x30` read)
/// promote: the width gate applies to the access `size`, not `offset + size`.
#[derive(Clone, Copy)]
struct ReadField {
    /// Constant byte offset from the base pointer.
    offset: u64,
    /// Access width in bytes; 1/2/4/8.
    size: usize,
}

/// A dereferenced pointer parameter slated for promotion, and the IR sites the
/// rewrite must touch.
struct Promoted {
    /// The pointer parameter (a root block param) — its *address* value, kept in
    /// the rewritten function as the shadow-space index and write-set address.
    param: ValueId,
    /// Display name, used to name the added by-value snapshot param and to find
    /// the parameter's call-argument index.
    name: String,
    /// Call-argument index of the pointer parameter.
    arg_idx: usize,
    /// Byte width of the base pointer itself (the address width), used to build
    /// the `base + offset` constants on both the callee and caller side.
    base_size: usize,
    /// One snapshot scalar per distinct read `(offset, size)`, sorted by offset.
    /// Empty for a write-only pointer.
    reads: Vec<ReadField>,
    /// Load/store instructions whose space must switch to the shadow space.
    accesses: Vec<InstructionId>,
    /// Distinct `(address, size)` of each store target written through this
    /// parameter — one write-set entry per address.
    write_targets: Vec<(ValueId, usize)>,
    /// Dynamic-index region snapshotted through this parameter, if any. At most
    /// one per param (all dynamic accesses through the param merge into a single
    /// span). Snapshotted as one `Array` input + (if written) one write entry.
    region: Option<SequenceRegion>,
}

/// How a named parameter is used, deciding whether and how it is promoted.
struct ParamInfo {
    /// The param is a load/store base at least once (a real dereference). When
    /// false the param is left untouched (e.g. a plain integer, or a pointer used
    /// only in arithmetic and returned).
    is_deref: bool,
    /// One snapshot scalar per distinct read `(offset, size)`, sorted by offset.
    reads: Vec<ReadField>,
    /// Load/store instructions to redirect into the shadow (shadow path only).
    accesses: Vec<InstructionId>,
    /// Distinct `(address, size)` of each store target written through this param.
    write_targets: Vec<(ValueId, usize)>,
    /// Dynamic-index region snapshotted through this param, if any.
    region: Option<SequenceRegion>,
}

/// A materialized global slot: a constant real-ram address this function accesses.
/// The RAM channel owns global materialization — a read becomes a by-value input
/// carrying `mem[addr]` (seeded into the shadow at the literal address), a write
/// additionally rides out in the returned write-set. Grouped by `(addr, size)`.
struct GlobalAccess {
    /// Constant address value.
    addr: u64,
    /// Width of the address literal — the shadow index / pointer width.
    addr_size: usize,
    /// Access (value) width in bytes.
    size: usize,
    /// Whether any access at this slot is a store (rides out in the write-set).
    has_write: bool,
    /// The real-ram load/store instructions to redirect into the shadow.
    accesses: Vec<InstructionId>,
}

/// Collect `fid`'s global (constant real-ram address) accesses, grouped by
/// `(addr, size)` in first-seen (block/instruction) order — determinism the
/// parallel/sequential differential test relies on. A slot whose `glob_<addr>`
/// value param already exists on the root (its origin is the address literal) is
/// skipped: it was materialized on an earlier `argpromote-cleanup` round, so
/// re-minting would desync the interface (idempotence).
fn collect_globals(ctx: &Context, fid: FunctionId) -> Vec<GlobalAccess> {
    let Some(root) = FunctionBody::from_id(ctx, fid).root().map(|b| b.id) else {
        return Vec::new();
    };
    let existing_origins: HashSet<ValueId> = BasicBlock::from_id(ctx, root)
        .params()
        .filter_map(|p| p.origin())
        .collect();
    let mut globals: Vec<GlobalAccess> = Vec::new();
    for block in FunctionBody::from_id(ctx, fid).iter() {
        for insn in block.iter() {
            let (space, ptr, size, is_store) = match insn.mnemonic() {
                Mnemonic::Load(l) => (l.space, l.ptr, l.size, false),
                Mnemonic::Store(s) => (s.space, s.ptr, s.size, true),
                _ => continue,
            };
            let Some((addr, addr_size)) = super::globals::global_slot(ctx, fid, ptr, space) else {
                continue;
            };
            if existing_origins.contains(&ctx.get_const(addr, addr_size).id()) {
                continue;
            }
            if let Some(g) = globals
                .iter_mut()
                .find(|g| g.addr == addr && g.size == size)
            {
                g.has_write |= is_store;
                g.accesses.push(insn.id);
            } else {
                globals.push(GlobalAccess {
                    addr,
                    addr_size,
                    size,
                    has_write: is_store,
                    accesses: vec![insn.id],
                });
            }
        }
    }
    globals
}

fn try_promote(
    ctx: &mut Context,
    fid: FunctionId,
    sp_reg: Option<VarnodeId>,
    address_taken: &FxHashSet<FunctionId>,
    graph: &crate::CallGraph,
    ram_summaries: &super::summary::EffectSummaries<super::ram_summary::RamChannel>,
    absorbed: &FxHashSet<FunctionId>,
) -> Promotion {
    let f = FunctionBody::from_id(ctx, fid);
    if f.is_external() {
        return Promotion::No;
    }

    // Only functionalize functions whose call interface this pass owns. The whole
    // RAM channel keys snapshot args off the `param[i] ↔ Call.args[i]` lockstep
    // (see [`arg_index_of`]), and that lockstep is an invariant *only* for
    // `pure_reg` functions — the ones `argpromote_registers` established it for. A
    // conventional (non-`pure_reg`) function uses the register ABI, not positional
    // params, and accrues root params (live-in registers promoted by
    // mem2reg, an incoming `@SP` param, …) that no caller passes positionally. Trying
    // to snapshot a deref through such a param reads a `Call.args` slot that does not
    // exist. So skip them: a conventional function is left for the legacy path, which
    // is correct (the body is unchanged). `argpromote_registers` runs earlier and
    // makes every functionalizable function `pure_reg`, so this loses no real work.
    if !f.is_reg_materialized() {
        return Promotion::No;
    }

    let Some(root) = f.root().map(|b| b.id) else {
        return Promotion::No;
    };

    // Closed-world / direct-only gate: if the function's address is taken it may
    // be reached by an indirect call this pass cannot find and rewrite, leaving a
    // caller on the old by-reference ABI. (Callers in undiscovered code are an
    // accepted, unguardable gap — see the module docs.)
    if address_taken.contains(&fid) {
        return Promotion::No;
    }

    // Snapshot arguments are real data loaded at the caller — unlike a register
    // input or a global's address literal they have no implicit binding, so a
    // non-regpure (`Opaque`) direct site cannot be given one. Promoting past
    // such a site would either desync the `param[i] ↔ arg[i]` lockstep or trip
    // the `[lockstep]` asserts in `apply`; bail instead.
    if crate::calls::direct_call_sites(ctx, graph, fid)
        .into_iter()
        .any(|site| {
            !matches!(
                ctx.get_insn(site).mnemonic(),
                Mnemonic::Call(c) if c.tag.is_regpure()
            )
        })
    {
        qcode::pass_log!(
            debug,
            "argpromote {}: bail — a direct call site is not regpure (implicit binding \
             cannot carry snapshot args)",
            FunctionBody::from_id(ctx, fid).name(),
        );
        return Promotion::No;
    }

    // A call composes with our shadow promotion only if it is fully inert toward
    // memory: a *direct* call, with no clobbers (writes nothing the caller sees),
    // to a callee that is *transitively* memory-free (so nothing it reaches can
    // dereference a promoted pointer we hand it, or alias our shadow). Any
    // other call — indirect, clobbering, or memory-reaching — would have to bubble
    // its effects through ours, which stage 2's rebasing transfer will do; bail.
    if function_makes_blocking_call(ctx, fid, ram_summaries, absorbed, sp_reg) {
        return Promotion::No;
    }

    // Every return block carries its own copy of the write-set (the register
    // channel already emits a per-return tuple we append to). The shadow path needs
    // no write-dominance reasoning: each write slot is seeded as a by-value input, so
    // a path that does not execute the write reloads that seed and replays a no-op —
    // sound on any control flow (see [`apply`]).
    let returns: Vec<InstructionId> = FunctionBody::from_id(ctx, fid)
        .iter()
        .filter_map(|b| {
            let last = b.iter().last()?;
            matches!(last.mnemonic(), Mnemonic::Return(_)).then_some(last.id)
        })
        .collect();
    if returns.is_empty() {
        return Promotion::No;
    }

    // Classify every named root parameter. Dereferenced pointers are promoted and
    // share one shadow space (so aliasing among them stays correct); everything
    // else is left as-is.
    let candidates: Vec<(ValueId, String)> = BasicBlock::from_id(ctx, root)
        .params()
        .filter_map(|p| Some((p.id(), p.name()?.to_string())))
        .collect();

    // The affine view of every value, and the function's full real-ram load/store
    // list — both shared across all params so the per-param decomposition can peel
    // a strided `param + idx*scale + const` address (see [`relate_address`]).
    let numbering = precompute_forms(qcode::value::ModuleView::new(&*ctx), fid);
    let ram = ctx.shared.default_space;
    let accesses_in: Vec<MemoryAccess> = FunctionBody::from_id(ctx, fid)
        .iter()
        .flat_map(|b| {
            let bid = b.id;
            b.iter().filter_map(move |insn| match insn.mnemonic() {
                Mnemonic::Load(l) if l.space == ram => Some(MemoryAccess {
                    id: insn.id,
                    block: bid,
                    is_store: false,
                    ptr: l.ptr.qualify(insn.id.func),
                    size: l.size,
                }),
                Mnemonic::Store(s) if s.space == ram => Some(MemoryAccess {
                    id: insn.id,
                    block: bid,
                    is_store: true,
                    ptr: s.ptr.qualify(insn.id.func),
                    size: s.size,
                }),
                _ => None,
            })
        })
        .collect();

    // Rule 2: own-frame reads may be modelled unseeded (redirected into shadow
    // without a snapshot) exactly when the function's own solved summary already
    // proved every read observes an own — or licensed-external — write, i.e. the
    // function is outward-invisible. The scan did that reasoning; do not re-derive
    // freshness here.
    let own_frame_reads = OwnFrame::new(ctx, fid, sp_reg);
    let model_frame_reads = super::ram_summary::is_memory_free(ram_summaries, fid);

    let mut promoted: Vec<Promoted> = Vec::new();
    for (param, name) in candidates {
        let info = analyze_param(
            ctx,
            &numbering,
            &accesses_in,
            param,
            &own_frame_reads,
            model_frame_reads,
        );
        if !info.is_deref {
            continue;
        }
        // A deref param with no findable argument index cannot have its snapshot
        // loaded at the caller; skip it. Its accesses then stay unmodelled, which
        // steers the function to partial mode.
        let Some(arg_idx) = arg_index_of(ctx, fid, &name) else {
            continue;
        };
        let base_ty = ctx.type_of(param);
        let base_size = ctx.shared.types.size_of(base_ty);
        promoted.push(Promoted {
            param,
            name,
            arg_idx,
            base_size,
            reads: info.reads,
            accesses: info.accesses,
            write_targets: info.write_targets,
            region: info.region,
        });
    }

    // Mode selection. The shadow path is sound exactly when the shared shadow
    // captures the function's *entire* real-ram footprint: every load/store is a
    // captured deref of a promoted pointer, redirected into the shadow. Then the
    // store→load forwarder (which only sees shadow stores) cannot collapse a read
    // to its seed across an invisible aliasing write, and — because the shadow is
    // keyed by address value — a pointer stored as data and later re-dereferenced
    // resolves through the shadow too (no separate "escape" gate needed: the only
    // unmodelled re-dereference is one whose address is loaded at runtime, which is
    // itself an uncaptured access this check already rejects).
    //
    // When the footprint isn't fully captured we may fall back to *inputs-only*
    // partial promotion (see [`apply_partial`]), but only for read-only memory.
    // Seeding a snapshot into real RAM in a function that also stores is not
    // sufficient: a downstream forwarder can replace the later load with the
    // entry snapshot across an unmodelled aliasing write. Without a proof that
    // every store is disjoint, leave the function unchanged.
    //
    // We additionally require every write to be **caller-resolvable** (an own-frame
    // local — dead on exit — or a constant `promoted-param + offset`). Then each
    // write slot's initial value can be snapshot-seeded by the caller, so a write
    // that does not execute on some path replays a no-op, and no dominance reasoning
    // is needed. A function with any other (dynamic-address) write stays on the
    // partial path. (`other passes drop the redundant seed args.`)
    if !all_accesses_modelled(ctx, fid, &promoted, sp_reg)
        || !all_writes_resolvable(ctx, fid, &promoted, sp_reg)
        || !regions_disjoint(ctx, fid, &promoted, sp_reg)
    {
        if accesses_in.iter().any(|access| access.is_store) {
            qcode::pass_log!(
                debug,
                "argpromote {}: bail — partial promotion with stores may forward across aliases",
                FunctionBody::from_id(ctx, fid).name(),
            );
            return Promotion::No;
        }
        qcode::pass_log!(
            debug,
            "argpromote {}: partial (inputs-only) — footprint not fully modelled",
            FunctionBody::from_id(ctx, fid).name(),
        );
        let call_sites = crate::calls::direct_call_sites(ctx, graph, fid);
        return if apply_partial(ctx, fid, &promoted, &call_sites) {
            Promotion::Partial
        } else {
            Promotion::No
        };
    }
    qcode::pass_log!(
        debug,
        "argpromote {}: shadow path — {} promoted deref param(s)",
        FunctionBody::from_id(ctx, fid).name(),
        promoted.len(),
    );

    // ---- shadow path (footprint fully modelled, writes caller-resolvable) ---

    // Nothing to do unless some promoted pointer is actually dereferenced — read
    // (its loaded scalar moves to a by-value arg) or written (its stored value
    // moves to the returned write-set). A pointer touched only as data is a no-op.
    // A pointer touched only as data surfaces nothing; but a captured own-frame
    // read (rule 2) redirected into the shadow is real work even with no surfaced
    // read/write, so `accesses` must be considered too.
    // Global (constant-address) accesses the RAM channel materializes alongside
    // the promoted params. Collected here so a globals-only function (no promoted
    // deref params) still reaches `apply`.
    let globals = collect_globals(ctx, fid);

    if globals.is_empty()
        && promoted.iter().all(|p| {
            p.reads.is_empty()
                && p.write_targets.is_empty()
                && p.region.is_none()
                && p.accesses.is_empty()
        })
    {
        return Promotion::No;
    }

    let call_sites = crate::calls::direct_call_sites(ctx, graph, fid);
    if apply(ctx, fid, promoted, globals, sp_reg, &call_sites) {
        Promotion::Shadow
    } else {
        Promotion::No
    }
}

/// Whether every real-memory (default-space) load/store in `fid` is captured by
/// `promoted` — i.e. will be redirected into the shadow. Accesses already in a
/// shadow space (a prior promotion round) are inherently modelled and skipped, so
/// the check is idempotent. A single uncaptured ram access fails it: see the
/// all-or-nothing rationale at the call site.
///
/// Read/write asymmetry: an uncaptured real-ram **Load** whose address is a
/// caller-frame slot (`@SP + k`, `k ≥ 0` — the return address / incoming stack
/// args) is tolerated. Such a read is disjoint from everything the promotion
/// redirects: the caller established that data, it is never routed into the
/// shadow, and its value is identical before and after. A read never blocks
/// promotion; only writes and address escapes do. Any uncaptured **Store**, and
/// any uncaptured Load that is *not* a caller-frame slot (a genuine unmodeled
/// deref), still fails the check.
fn all_accesses_modelled(
    ctx: &Context,
    fid: FunctionId,
    promoted: &[Promoted],
    sp_reg: Option<VarnodeId>,
) -> bool {
    let ram = ctx.shared.default_space;
    let captured: HashSet<InstructionId> = promoted
        .iter()
        .flat_map(|p| p.accesses.iter().copied())
        .collect();
    let own_frame = OwnFrame::new(ctx, fid, sp_reg);
    FunctionBody::from_id(ctx, fid).iter().all(|block| {
        block.iter().all(|insn| {
            let (space, load_ptr) = match insn.mnemonic() {
                Mnemonic::Load(l) => (l.space, Some(l.ptr.qualify(insn.id.func))),
                Mnemonic::Store(s) => (s.space, None),
                // Not a memory access — nothing to model.
                _ => return true,
            };
            if space != ram || captured.contains(&insn.id) {
                return true;
            }
            // A global (constant real-ram address) is materialized by the RAM
            // channel's global path (a by-value input + optional write-set
            // entry), so it is modelled and never blocks the shadow path.
            let is_global = match insn.mnemonic() {
                Mnemonic::Load(l) => {
                    super::globals::global_slot(ctx, fid, l.ptr, l.space).is_some()
                }
                Mnemonic::Store(s) => {
                    super::globals::global_slot(ctx, fid, s.ptr, s.space).is_some()
                }
                _ => false,
            };
            if is_global {
                return true;
            }
            // Tolerate only an uncaptured caller-frame-slot *read* (the interface
            // region the promotion never touches). Everything else — any store,
            // any other unmodeled deref — remains fatal.
            load_ptr.is_some_and(|ptr| own_frame.is_caller_frame_slot(ctx, ptr))
        })
    })
}

/// Whether every store through a promoted pointer writes a **caller-resolvable**
/// address: either an own-frame local (excluded from the write-set as dead on exit)
/// or a constant non-negative `promoted-param + offset` (which the caller can
/// snapshot-seed and we can recompute at the return). A function with any other
/// (dynamic-address) write is kept on the partial path — the shadow path seeds every
/// surfaced write as an input arg, which needs the address known at the call site.
fn all_writes_resolvable(
    ctx: &Context,
    fid: FunctionId,
    promoted: &[Promoted],
    sp_reg: Option<VarnodeId>,
) -> bool {
    let own_frame = OwnFrame::new(ctx, fid, sp_reg);
    let numbering = precompute_forms(qcode::value::ModuleView::new(ctx), fid);
    let is_param = |v: ValueId| promoted.iter().any(|p| p.param == v);
    promoted.iter().all(|p| {
        p.write_targets.iter().all(|&(addr, _)| {
            if own_frame.is_local(ctx, addr) {
                return true;
            }
            // A constant offset of *either* sign is caller-resolvable (the caller
            // snapshots `arg + offset`); a dynamic address decomposes to a non-param
            // base and is rejected.
            let (base, _off) = numbering
                .base_offset(qcode::value::ModuleView::new(ctx), addr)
                .unwrap_or((addr, 0));
            is_param(base)
        })
    })
}

/// Conservative v1 disjointness gate for dynamic-index regions. A written region
/// is snapshotted and *replayed as one wide store* at the caller; for that replay
/// to be sound it must not overlap another replayed write the caller orders
/// independently. Two *incoming* pointer regions can essentially never be proven
/// disjoint (no `restrict`), so v1 promotes a region only when it is unambiguously
/// the sole non-frame writer:
///
/// * at most one region in the whole function, and
/// * if it is written, every *other* surfaced store either
///   - lands in an `@SP`-rooted frame slot — an own-frame local (sound by frame
///     freshness) or a caller-frame slot (under
///     [`Proposition::ArgsDisjointFromCallerFrame`]); the region base is a promoted
///     *incoming* pointer param, disjoint from the whole frame, so those frame
///     writes (e.g. argpromote's own spilled-arg seed stores, replayed as no-ops)
///     cannot overlap the region; or
///   - is a scalar at a constant offset of the **region's own base param** whose
///     byte range is disjoint from the region's span — e.g. a `count` field at
///     offset 0 next to an array body at offset ≥ 4. Same base + non-overlapping
///     offsets is offset-precise disjointness, so the independent replays cannot
///     collide.
///
/// Any *other* non-frame coexisting write — a second incoming pointer we cannot
/// separate, or a same-base write that overlaps the region — keeps the function on
/// the partial path.
///
/// (A read-only region has no replay, so it is always safe.)
fn regions_disjoint(
    ctx: &Context,
    fid: FunctionId,
    promoted: &[Promoted],
    sp_reg: Option<VarnodeId>,
) -> bool {
    if promoted.iter().filter(|p| p.region.is_some()).count() > 1 {
        return false;
    }
    // The single written region, if any: its base param (with its name) and its
    // `[lo, hi)` span.
    let Some((region_base, region_base_name, region_lo, region_hi)) =
        promoted.iter().find_map(|p| {
            p.region.filter(|r| r.has_write).map(|r| {
                (
                    p.param,
                    p.name.clone(),
                    r.base_off as i64,
                    r.base_off as i64 + r.byte_len() as i64,
                )
            })
        })
    else {
        // No *written* region: nothing replayed independently, so always safe.
        return true;
    };
    let caller_frame_assumed = ctx
        .truth(Proposition::ArgsDisjointFromCallerFrame(fid))
        .is_some_and(|t| t.value);
    // A pointer's storage slot is disjoint from the buffer it addresses — the
    // region rooted at its *loaded value* (`Proposition::LoadedPointerDisjointFromSlot`).
    let loaded_ptr_disjoint = ctx
        .truth(Proposition::LoadedPointerDisjointFromSlot(fid))
        .is_some_and(|t| t.value);
    let own_frame = OwnFrame::new(ctx, fid, sp_reg);
    let numbering = precompute_forms(qcode::value::ModuleView::new(ctx), fid);
    promoted.iter().all(|p| {
        p.write_targets.iter().all(|&(addr, wsize)| {
            if own_frame.is_local(ctx, addr)
                || (caller_frame_assumed && own_frame.is_frame_slot(ctx, addr))
            {
                return true;
            }
            // A write through a pointer param whose *loaded value* roots the region —
            // i.e. the region is this pointer's pointee — cannot overlap the region:
            // the buffer does not overlap the storage of the pointer that addresses
            // it (`LoadedPointerDisjointFromSlot`). The snapshot naming links the two:
            // the region base param is `p`'s deref, named `{p.name}_val_<off>`. This
            // is what lets a state-init function that *also* publishes its buffer
            // pointer into a global slot (`*pp = buf`) still region-promote `*buf`.
            if loaded_ptr_disjoint && region_base_name.starts_with(&format!("{}_val_", p.name)) {
                return true;
            }
            // Offset-precise disjointness against the written region: the scalar
            // must share the region's base param and miss its byte span.
            let (base, off) = numbering
                .base_offset(qcode::value::ModuleView::new(ctx), addr)
                .unwrap_or((addr, 0));
            base == region_base && (off + wsize as i64 <= region_lo || off >= region_hi)
        })
    })
}

/// What one [`try_promote`] visit did to a function.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Promotion {
    /// Nothing changed.
    No,
    /// Inputs-only partial promotion; real-ram loads remain in the body.
    Partial,
    /// Full shadow promotion: the entire real-ram footprint was absorbed.
    Shadow,
}

/// Whether `call_id` is an admissible prototyped-external argmem call whose
/// pointer arguments the shadow rewrite can rebase into the shadow space, and if
/// so the `(arg index, own-frame local pointer value)` pairs to redirect. Shared
/// by the blocking-call gate (rule 3) and the apply-side rebase (rule 4), so the
/// two never disagree: the gate admits a call exactly when the rewrite can rebase
/// every pointer argument.
///
/// Admission (all required):
/// * a clobber-free `Call` to a bodyless external with an `ExternArgmem`,
///   non-variadic, no `Opaque` param kinds;
/// * every `OutPtr`/`MutPtr`/`ConstPtr` argument is one of this function's own
///   frame locals (`OwnFrame::is_local`); `NonPtr` args are irrelevant;
/// * the call's return value is unused (a used return may carry a real pointer we
///   cannot rebase — an escape; `memset`'s return is dead). v1 approximates the
///   ruled "return type is a pointer AND used" by "used", which is stricter and
///   sound.
fn admissible_external_argmem_call(
    ctx: &Context,
    call_id: InstructionId,
    own_frame: &OwnFrame,
) -> Option<Vec<(usize, ValueId)>> {
    let Mnemonic::Call(c) = ctx.get_insn(call_id).mnemonic() else {
        return None;
    };
    if !c.clobbers.is_empty() {
        return None;
    }
    let ext = c.target.real()?;
    let ef = FunctionBody::from_id(ctx, ext);
    if !ef.is_external() {
        return None;
    }
    let argmem = ef.argmem()?;
    if argmem.variadic
        || argmem
            .params
            .iter()
            .any(|k| matches!(k, ArgMemKind::Opaque))
    {
        return None;
    }
    // Return-escape gate: a used return value may hand a real pointer to the
    // caller that we cannot rebase.
    if !FunctionBody::from_id(ctx, call_id.func)
        .users_of(ValueId::Instruction(call_id))
        .is_empty()
    {
        return None;
    }
    let mut rebase: Vec<(usize, ValueId)> = Vec::new();
    for (i, kind) in argmem.params.iter().enumerate() {
        match kind {
            ArgMemKind::NonPtr => {}
            ArgMemKind::OutPtr | ArgMemKind::MutPtr | ArgMemKind::ConstPtr => {
                let arg = c.args.get(i)?.qualify(call_id.func);
                if !own_frame.is_local(ctx, arg) {
                    return None;
                }
                // The rewrite rebuilds the arg as a fresh shadow-typed `@SP + off`
                // pointer (see rule 4), which needs a stable @SP-relative offset.
                // A realigned-base local (`@SP & -mask`) has none, so it cannot be
                // soundly rebased — leave the whole call unpromoted.
                own_frame.local_offset(ctx, arg)?;
                rebase.push((i, arg));
            }
            // Filtered out above.
            ArgMemKind::Opaque => return None,
        }
    }
    Some(rebase)
}

/// `true` if `function_id` makes any call this promotion cannot compose with.
/// A body walk (not an edge walk) so an *unresolved* direct call — which never
/// makes a [`CallGraph`] edge — still blocks. The callee verdict comes from the
/// pre-solved [`super::ram_summary`] fixpoint: a memory-free callee cannot
/// dereference a pointer passed to it — for reading *or writing* — anywhere in
/// its call tree, so handing it a promoted pointer is safe (nothing it reaches
/// can observe our shadow, alias it, or mutate a promoted address behind it).
pub(super) fn function_makes_blocking_call(
    ctx: &Context,
    function_id: FunctionId,
    ram_summaries: &super::summary::EffectSummaries<super::ram_summary::RamChannel>,
    absorbed: &FxHashSet<FunctionId>,
    sp_reg: Option<VarnodeId>,
) -> bool {
    // Own-frame view, built once, used to admit prototyped-external argmem calls
    // whose pointer args are this function's own locals (rules 3/4). Inert when
    // there is no stack pointer.
    let own_frame = OwnFrame::new(ctx, function_id, sp_reg);
    FunctionBody::from_id(ctx, function_id).blocks().any(|b| {
        b.iter().any(|i| match i.mnemonic() {
            // Indirect transfers: target unknown, cannot vet.
            Mnemonic::CallInd(_) => true,
            // A direct call is inert iff it clobbers nothing and the callee is
            // transitively memory-free; otherwise its effects would have to
            // bubble through ours.
            Mnemonic::Call(c) => {
                let inert = c.clobbers.is_empty()
                    && c.target.real().is_some_and(|target| {
                        // Inert: no caller-observable footprint per the solved
                        // summary, or the shadow path absorbed the footprint
                        // earlier in this sweep — the call-site rewrite has
                        // already landed it as this function's own accesses,
                        // which the scan below captures.
                        super::ram_summary::is_memory_free(ram_summaries, target)
                            || absorbed.contains(&target)
                    });
                // A prototyped-external argmem call whose pointer args are all
                // own-frame locals is also non-blocking: the shadow rewrite
                // rebases those args into the shadow (rule 4), so the external's
                // whole-object writes stay body-private and its footprint never
                // bubbles into the caller.
                !(inert || admissible_external_argmem_call(ctx, i.id, &own_frame).is_some())
            }
            // An `Apply` can now be impure (a tail call rewritten to
            // `apply g; return` by `retail_apply`), so it is vetted exactly like a
            // `Call`: inert iff its target is transitively memory-free or was
            // absorbed this sweep. `Apply` has no `clobbers` field, so there is no
            // clobber check — the register interface is threaded through its pack.
            Mnemonic::Apply(a) => !a.target.real().is_some_and(|target| {
                super::ram_summary::is_memory_free(ram_summaries, target)
                    || absorbed.contains(&target)
            }),
            // Other call-like transfers escape vetting entirely: a tail callee's
            // memory effects never bubble through the summary, and the
            // write-replays `apply` emits at Return exits would be skipped on the
            // tail path. A surviving BranchInd is a computed tail-jump into code
            // we cannot see. Both block. (`Map`/`Scan` are unvetted-by-assumption:
            // their bodies are pure by construction, so they never appear here as a
            // memory hazard.)
            Mnemonic::TailCall(_) | Mnemonic::BranchInd(_) => true,
            // Non-transfer mnemonics are inert; any new call-like transfer must
            // be matched explicitly above rather than falling through here.
            _ => false,
        })
    })
}

/// Classify how `param` is used (see [`ParamInfo`]). Walks every real-ram
/// load/store in the function and, via the affine [`relate_address`] decomposition,
/// collects the accesses to redirect into shadow, the distinct scalar write
/// targets, and the dynamic-index region — the by-value snapshot width is bounded
/// by the *reads* (writes may land at any offset).
///
/// There is no "escape" concept: a pointer value flowing into memory is not a
/// hazard on its own. The shadow path's soundness is guarded entirely by
/// [`all_accesses_modelled`] — every real-ram access must be a captured (and so
/// redirected) deref of a promoted pointer. A read whose address is not an affine
/// `param + offset` is simply *not captured*, which fails that gate and steers the
/// function to partial mode; a stored-as-data pointer is harmless precisely because
/// the place it lands is itself captured (regime B) or else uncaptured and rejected
/// by the gate.
///
/// `accesses_in` is the function's full real-ram load/store list (id, block,
/// is_store, ptr, size); precomputed once and shared across all params.
fn analyze_param(
    ctx: &Context,
    numbering: &Numbering,
    accesses_in: &[MemoryAccess],
    param: ValueId,
    own_frame: &OwnFrame,
    model_frame_reads: bool,
) -> ParamInfo {
    let mut accesses: Vec<InstructionId> = Vec::new();
    // (offset from base, access width) for each load — one snapshot scalar each.
    let mut read_fields: Vec<ReadField> = Vec::new();
    let mut write_targets: Vec<(ValueId, usize)> = Vec::new();
    let mut is_deref = false;

    // Record a load at constant byte `offset` of `size` bytes. The width is gated
    // on the *access size* alone (not `offset + size`), so a deref at a large
    // fixed offset still promotes. A scalar a register cannot hold (size ∉
    // {1,2,4,8}) is not snapshotable, so the read is left unmodelled (`false`),
    // which fails `all_accesses_modelled`.
    let mut push_read = |offset: u64, size: usize| -> bool {
        if !matches!(size, 1 | 2 | 4 | 8) {
            return false;
        }
        // Dedup by offset, widening to the largest access there (a narrower load
        // forwards from the wider seed via base+const matching).
        if let Some(f) = read_fields.iter_mut().find(|f| f.offset == offset) {
            f.size = f.size.max(size);
        } else {
            read_fields.push(ReadField { offset, size });
        }
        true
    };

    for access in accesses_in {
        match relate_address(ctx, numbering, param, access.ptr, access.block) {
            AddressRelation::Unrelated => {}
            AddressRelation::Const(off) => {
                if access.is_store {
                    // The write address rides out in the return set as data; any
                    // `param ± const` target qualifies (a negative offset wraps
                    // two's-complement at the caller).
                    is_deref = true;
                    accesses.push(access.id);
                    write_targets.push((access.ptr, access.size));
                } else if off >= 0 {
                    // Reads become caller-seeded snapshots at `arg + offset`, so only
                    // a non-negative offset is recomputable; a negative-offset read
                    // is left unmodelled for the all-or-nothing gate to reject.
                    if push_read(off as u64, access.size) {
                        is_deref = true;
                        accesses.push(access.id);
                    }
                } else if model_frame_reads && own_frame.is_local(ctx, access.ptr) {
                    // Rule 2: an own-frame read whose content is self-contained in the
                    // shadow (own stores + admitted external writes). Modelled by
                    // redirecting it into the shadow, with NO snapshot/seed — the
                    // summary already licensed it (outward-invisible), so no
                    // caller-side reconstruction is needed.
                    is_deref = true;
                    accesses.push(access.id);
                }
            }
            AddressRelation::Dynamic { .. } => {}
        }
    }

    let region_set = collect_regions_for_base(ctx, numbering, param, accesses_in);
    if region_set.regions.len() == 1 {
        is_deref = true;
        // The region is viable (an `Array` snapshot will be seeded for it), so its
        // folded accesses are now safe to redirect into shadow.
        accesses.extend(region_set.regions[0].accesses.iter().copied());
    } else if !region_set.regions.is_empty() {
        qcode::pass_log!(
            debug,
            "dropping {} disjoint region(s) through one param: argpromote still materializes \
             one Array region per param; function will fall to partial/no promotion",
            region_set.regions.len(),
        );
    }
    if !region_set.rejected_accesses.is_empty() {
        qcode::pass_log!(
            debug,
            "dropping {} unmodelled region access(es) (region failed to build) - \
             function will fall to partial/no promotion rather than orphan them in shadow",
            region_set.rejected_accesses.len(),
        );
    }
    // Rejected or unsupported dynamic accesses get no seed, so they are deliberately
    // NOT added to `accesses` — left unmodelled so `all_accesses_modelled` rejects the
    // function to partial mode, rather than orphan-redirecting an un-seeded shadow
    // read. (This is the TEB/PEB bug: a wide bounded fs-offset failed to build a
    // region, leaving the `load(teb + idx)` redirected to shadow with no snapshot arg.)

    // Deterministic offset order, shared by callee param creation and caller
    // argument loads.
    read_fields.sort_by_key(|f| f.offset);

    // Dedup write targets by address value.
    let mut deduped: Vec<(ValueId, usize)> = Vec::new();
    for wt in write_targets {
        if !deduped.iter().any(|(a, _)| *a == wt.0) {
            deduped.push(wt);
        }
    }

    let region = if region_set.regions.len() == 1 {
        Some(region_set.regions[0].region)
    } else {
        None
    };

    ParamInfo {
        is_deref,
        reads: read_fields,
        accesses,
        write_targets: deduped,
        region,
    }
}

/// Rewrite `fid` and every direct caller into the shadow-memory / write-set form.
/// Returns `false` if a precondition fails late (e.g. no callers).
fn apply(
    ctx: &mut Context,
    fid: FunctionId,
    mut promoted: Vec<Promoted>,
    globals: Vec<GlobalAccess>,
    sp_reg: Option<VarnodeId>,
    call_sites: &[InstructionId],
) -> bool {
    if call_sites.is_empty() {
        return false;
    }

    let ram = ctx.shared.default_space;
    let default = ctx.shared.space(ram);
    let (word_size, addr_size) = (default.word_size, default.addr_size);
    let shadow_ts =
        ctx.bodies[fid].push_temp_space(TempSpace::new(Some("argpromote"), word_size, addr_size));
    let shadow_local = shadow_ts.local;
    let shadow = LocalMemorySpaceId::Temp(shadow_local);
    // Every promoted pointer in this function shares one body-local shadow.
    // Equal addresses therefore collide for aliasing, while another function's
    // same-address shadow remains isolated by construction.

    // Deterministic order shared by the callee (param creation) and the callers
    // (snapshot argument order).
    promoted.sort_by_key(|p| p.arg_idx);

    // ---- callee rewrite -----------------------------------------------------

    // The distinct surfaced write targets. An own-frame local (an `@SP`-rooted slot
    // below entry SP) is dead on exit — the frame is destroyed at return — so it is
    // redirected to shadow (keeping `all_accesses_modelled` satisfied) but excluded
    // from the write-set; its readerless shadow store is left for DCE. Every other
    // write target is, by the resolvability gate in [`try_promote`], a constant
    // `promoted-param + offset` the caller can resolve. We seed each like a read
    // (snapshot its initial value), so a path that does *not* write it reloads that
    // value and replays a no-op — sound on any control flow, no dominance needed.
    // `(base_param, base_size, offset, size)`.
    let own_frame = OwnFrame::new(ctx, fid, sp_reg);
    let numbering = precompute_forms(qcode::value::ModuleView::new(&*ctx), fid);
    // `(base_param, base_size, signed offset, size, base arg index)`. A negative
    // offset (`base - k`, e.g. an own-frame slot when there is no stack-pointer to
    // recognise it) is encoded two's-complement into the address const, so
    // `base + offset` wraps to `base - k`. The arg index (when the base is a
    // promoted param) lets the caller-side replay *recompute* the write address
    // as `arg + offset` — affine in the caller, so the caller's own footprint
    // scan captures the replayed store; the extracted pack address is opaque.
    let mut write_slots: Vec<(ValueId, usize, i64, usize, Option<usize>)> = Vec::new();
    for p in &promoted {
        for &(addr, size) in &p.write_targets {
            if own_frame.is_local(ctx, addr) {
                continue;
            }
            let (base, offset) = numbering
                .base_offset(qcode::value::ModuleView::new(&*ctx), addr)
                .unwrap_or((addr, 0));
            let based = promoted.iter().find(|q| q.param == base);
            let base_size = based.map_or(p.base_size, |q| q.base_size);
            if !write_slots
                .iter()
                .any(|&(b, _, o, _, _)| b == base && o == offset)
            {
                write_slots.push((base, base_size, offset, size, based.map(|q| q.arg_idx)));
            }
        }
    }
    // A written region contributes one *wide* write slot: the whole `[base +
    // base_off, + byte_len)` span, reloaded from shadow at the return and replayed
    // as a single store at every caller. Its initial value is seeded by the region
    // input below, so a path that does not write replays a no-op — same discipline
    // as the scalar slots.
    for p in &promoted {
        if let Some(r) = p.region.filter(|r| r.has_write) {
            write_slots.push((
                p.param,
                p.base_size,
                r.base_off as i64,
                r.byte_len(),
                Some(p.arg_idx),
            ));
        }
    }
    // Each written global rides out as one write-set entry: shadow_addr = ram_addr
    // = the constant address (base = the address literal, offset 0), value read out
    // of shadow, replayed at the caller as `store(ram, addr, value)`. `arg_idx =
    // None` → the replay falls back to the extracted literal address (no base+offset
    // arithmetic). The initial value is seeded into shadow by the global input minted
    // below, so a path that does not store replays a no-op.
    for g in &globals {
        if g.has_write {
            let base = ctx.get_const(g.addr, g.addr_size).id();
            write_slots.push((base, g.addr_size, 0, g.size, None));
        }
    }

    // The slots to snapshot+seed as by-value input args: one per read scalar, plus
    // one per write target whose `(base, offset)` no read already covers. Each write
    // slot thus *also* becomes an input arg (its initial value); a later
    // `dead_signature` drops the arg if the body never reads it. Deterministic order
    // drives both callee param creation and the caller's snapshot args.
    //
    // NB (accepted): seeding a *write-only* slot makes the caller emit a speculative
    // `load(arg + offset)` of a location the original function only ever *wrote*. For
    // analysis this is harmless (the value is overwritten or replayed as a no-op);
    // were the rewritten code actually executed it could fault on a write-only
    // location the original never read. We accept this — it is the cost of dropping
    // the write-dominance requirement in favour of unconditional seeding, and the
    // address is always one the caller demonstrably holds a pointer to.
    // `(arg_idx, base, base_size, offset, size, name)`. `type_id` is set for a
    // region snapshot (an `Array`-typed input); `tag` distinguishes a region's
    // param-name suffix (`_arr_`) from a scalar's (`_val_`) so the two never
    // collide and idempotence holds.
    struct Snap {
        arg_idx: usize,
        base: ValueId,
        base_size: usize,
        offset: i64,
        size: usize,
        name: String,
        type_id: Option<TypeId>,
        tag: &'static str,
    }
    let mut snaps: Vec<Snap> = Vec::new();
    for p in &promoted {
        for f in &p.reads {
            snaps.push(Snap {
                arg_idx: p.arg_idx,
                base: p.param,
                base_size: p.base_size,
                offset: f.offset as i64,
                size: f.size,
                name: p.name.clone(),
                type_id: None,
                tag: "val",
            });
        }
    }
    // One `Array`-typed snapshot input per region — read or written. Built before
    // the write-slot auto-seed loop so a written region's `(base, base_off)` is
    // already covered and not double-seeded as a scalar.
    for p in &promoted {
        if let Some(r) = p.region {
            let elem_ty = ctx.shared.types.get_or_make_int(r.elem_size);
            let array_ty = ctx.shared.types.get_or_make_array(elem_ty, r.count);
            snaps.push(Snap {
                arg_idx: p.arg_idx,
                base: p.param,
                base_size: p.base_size,
                offset: r.base_off as i64,
                size: r.byte_len(),
                name: p.name.clone(),
                type_id: Some(array_ty),
                tag: "arr",
            });
        }
    }
    for &(base, base_size, offset, size, _) in &write_slots {
        if snaps.iter().any(|s| s.base == base && s.offset == offset) {
            continue;
        }
        let Some(q) = promoted.iter().find(|q| q.param == base) else {
            continue;
        };
        snaps.push(Snap {
            arg_idx: q.arg_idx,
            base,
            base_size,
            offset,
            size,
            name: q.name.clone(),
            type_id: None,
            tag: "val",
        });
    }

    // Add one by-value snapshot input per slot (reads and seeded writes alike), in
    // the deterministic `snaps` order so the callee params and caller args stay in
    // lockstep. Each is seeded into the shadow at its `base + offset` so the body's
    // redirected access forwards from it, and loaded at every caller from the same
    // address relocated onto the call argument (the base is an identical by-value
    // param on both sides).
    for s in &snaps {
        let (base, base_size, offset, size, arg_idx) =
            (s.base, s.base_size, s.offset, s.size, s.arg_idx);
        let name = format!("{}_{}_{:x}", s.name, s.tag, s.offset);
        let pname = s.name.clone();
        super::add_input(
            ctx,
            fid,
            size,
            Some(name),
            None,
            s.type_id,
            Some(call_sites),
            shadow,
            move |b| seed_addr(b, base, base_size, offset),
            move |ctx, call_id, block| {
                let Mnemonic::Call(call) = ctx.get_insn(call_id).mnemonic().clone() else {
                    unreachable!("append_caller_arg only visits direct calls");
                };
                // Invariant: a call site's args are in lockstep with the callee's
                // root params (see `arg_index_of`), so the pointer arg index is
                // valid. A mismatch means an upstream pass minted a root param
                // without threading the matching argument.
                assert!(
                    arg_idx < call.args.len(),
                    "[lockstep] call arg count {} < snapshot param index {} for a direct \
                     call: callee {} (param '{}')",
                    call.args.len(),
                    arg_idx,
                    FunctionBody::from_id(ctx, fid).name(),
                    pname,
                );
                let caller_base = call.args[arg_idx].qualify(call_id.func);
                let mut b = (ctx).builder(block);
                b.set_insert_point_before(call_id);
                let addr = seed_addr(&mut b, caller_base, base_size, offset);
                b.push_load::<false>(addr, size, ram).id()
            },
        );
    }

    // Materialize each global slot (read or written) as one by-value input
    // carrying `mem[addr]`, seeded into the shadow at the literal address so the
    // body's (about-to-be-redirected) access forwards from it. Minted *after* the
    // per-param snapshot loop, in deterministic first-seen order, so callee params
    // and caller args stay in lockstep. The origin is the address literal: implicit
    // sites bind the value from it (the emulator reads `mem[addr]`), while the
    // regpure/direct sites in `call_sites` load the value verbatim.
    for g in &globals {
        let (addr, addr_size, size) = (g.addr, g.addr_size, g.size);
        let origin = ctx.get_const(addr, addr_size).id();
        super::add_input(
            ctx,
            fid,
            size,
            Some(format!("glob_{:x}", addr)),
            Some(origin),
            None,
            Some(call_sites),
            shadow,
            move |b| b.shr().get_const(addr, addr_size),
            move |ctx, call_id, block| {
                let mut b = (ctx).builder(block);
                b.set_insert_point_before(call_id);
                let a = b.shr().get_const(addr, addr_size);
                b.push_load::<false>(a, size, ram).id()
            },
        );
    }

    // Redirect every promoted load/store into the shadow space, keeping the real
    // address as the index.
    for p in &promoted {
        for &acc in &p.accesses {
            let mut m = ctx.get_insn(acc).mnemonic().clone();
            match &mut m {
                Mnemonic::Load(l) => l.space = shadow,
                Mnemonic::Store(s) => s.space = shadow,
                _ => {}
            }
            ctx.replace_instruction_mnemonic(acc, m);
        }
    }
    // Redirect every global access into the shadow at its (unchanged) literal
    // address, so the seeded input value and the write-set read/replay all agree.
    for g in &globals {
        for &acc in &g.accesses {
            let mut m = ctx.get_insn(acc).mnemonic().clone();
            match &mut m {
                Mnemonic::Load(l) => l.space = shadow,
                Mnemonic::Store(s) => s.space = shadow,
                _ => {}
            }
            ctx.replace_instruction_mnemonic(acc, m);
        }
    }

    // Rule 4: rebase each admitted prototyped-external call's own-frame pointer
    // args into the shadow. The external's whole-object writes then land in the
    // same shadow the body's own loads/stores were redirected to, keeping the
    // (dead-on-exit) local's view coherent — its post-call shadow reads observe
    // the external's writes, not the pre-call frame. The call itself stays (a real
    // external call, clobbers and non-pointer args untouched). Each rebased arg is
    // a SEPARATE address instruction (its pointer IS shadow-provenance here, but it
    // must not share an instruction with any exported write-set/interface value).
    {
        let shadow_mem = MemorySpaceId::Temp(TempSpaceId::new(fid, shadow_local));
        let ext_calls: Vec<(InstructionId, Vec<(usize, ValueId)>)> =
            FunctionBody::from_id(ctx, fid)
                .iter()
                .flat_map(|b| b.iter().map(|i| i.id).collect::<Vec<_>>())
                .filter_map(|id| {
                    admissible_external_argmem_call(ctx, id, &own_frame).map(|pairs| (id, pairs))
                })
                .collect();
        for (call_id, pairs) in ext_calls {
            let Mnemonic::Call(mut call) = ctx.get_insn(call_id).mnemonic().clone() else {
                continue;
            };
            let Some(block) = ctx.get_insn(call_id).parent().map(|b| b.id) else {
                continue;
            };
            for (arg_idx, arg) in pairs {
                let base_size = ctx.shared.types.size_of(ctx.type_of(arg));
                // Reconstruct the local as `@SP + off` from the incoming stack
                // pointer (both gated present by `admissible_external_argmem_call`).
                let off = own_frame
                    .local_offset(ctx, arg)
                    .expect("admissible call gates on a reconstructible @SP offset");
                let sp = own_frame
                    .sp_param()
                    .expect("an own-frame local implies a recognised @SP param");
                let new_id = {
                    let mut b = ctx.builder(block);
                    b.set_insert_point_before(call_id);
                    // A FRESH `@SP + off` address instruction, then stamped into
                    // the shadow space. `off != 0`, so it neither folds (`x+0→x`)
                    // nor value-numbers with the real own-frame pointer — and being
                    // shadow-typed it stays opaque to affine reassociation/CSE.
                    // This is genuine shadow-provenance arithmetic, not an
                    // `arg + 0` whose add-zero the folder would strip back to the
                    // real pointer (the externals-into-shadow miscompile).
                    seed_addr(&mut b, sp, base_size, off)
                };
                let sty = ctx
                    .shared
                    .types
                    .get_or_make_space_address(base_size, shadow_mem);
                let ValueId::Instruction(new_insn) = new_id else {
                    unreachable!("a non-zero @SP offset yields an add instruction");
                };
                qcode::value::Instruction::from_id_mut(ctx, new_insn).set_type(sty);
                call.args[arg_idx] = new_id.localize(call_id.func);
            }
            ctx.replace_instruction_mnemonic(call_id, Mnemonic::Call(call));
        }
    }

    // Append the memory write-set: a flat, interleaved `(addr, value)` pair per
    // surfaced write, recomputing `base + offset` at each return (always in scope;
    // the seed makes the reload defined on every path) and replaying it into real
    // memory at every caller. The register channel (run earlier) may have already
    // populated each return with its positional outputs, so our pairs are appended
    // *after* them (`append = true`); separate top-level `(addr, value)` fields
    // also let `partial_inline` recompute either half at the caller independently.
    // A read-only promotion has no write slots, so this is a no-op — the returns
    // and result type stay untouched.
    super::append_outputs(
        ctx,
        fid,
        &write_slots,
        Some(call_sites),
        true,
        |i, _| {
            vec![
                format!("write{}_addr", i + 1),
                format!("write{}_value", i + 1),
            ]
        },
        |b, &(base, base_size, offset, size, _)| {
            // Read the written value out of the shadow. `push_load` stamps its
            // pointer instruction with the load space's provenance
            // (`set_insn_space_local`), so this address becomes shadow-qualified —
            // correct, and kept internal to the body.
            let shadow_addr = seed_addr(b, base, base_size, offset);
            let v = b.push_load::<false>(shadow_addr, size, shadow).id();
            // The *exported* write address is a real-RAM pointer (replayed as
            // `store(ram)` at the caller). Build it as its own instruction so it
            // keeps `base`'s real provenance instead of inheriting the shadow
            // space from the load above — otherwise the body-local shadow space
            // leaks through the return interface and dangles when a caller
            // clones the field (e.g. `partial_inline`). At `offset == 0` this is
            // the bare `base` param, already real-typed.
            let ram_addr = seed_addr(b, base, base_size, offset);
            vec![ram_addr, v]
        },
        |b, &(_, base_size, offset, _, arg_idx), ext, args| {
            // Recompute the address from the caller's own argument when the
            // base is a promoted param present at this (lockstep) site — an
            // affine `arg + offset` the caller's footprint scan can capture,
            // which is what lets a caller promote past this call in the same
            // sweep (see the absorbed-set gate). Fall back to the extracted
            // pack address otherwise.
            let addr = arg_idx
                .and_then(|i| args.get(i).copied())
                .map_or(ext[0], |base| seed_addr(b, base, base_size, offset));
            b.push_store(ext[1], addr, ram);
        },
    );

    true
}

/// Recompute a snapshot/write-slot address `base + offset` for a builder. The
/// offset is two's-complement encoded into `base_size` bytes, so a negative
/// offset (`base - k`) wraps correctly; `offset == 0` is the bare base.
fn seed_addr(b: &mut Builder<'_, '_>, base: ValueId, base_size: usize, offset: i64) -> ValueId {
    if offset == 0 {
        base
    } else {
        let k = b.shr().get_const(offset as u64, base_size);
        b.push_add(base, k).id()
    }
}

/// Partial (no-shadow) promotion — the fallback when [`try_promote`]'s footprint
/// is not fully modelled (a leaked address, or some real-ram access this pass
/// cannot redirect). It is **inputs-only**: for every promoted deref param it adds
/// a by-value snapshot param per *read* field and seeds it into **real ram** at
/// entry (`store(snap → base+offset, ram)`), leaving every load in place.
/// It redirects no access into shadow, surfaces **no write-set**, and marks nothing
/// pure. This path is restricted to functions with no real-RAM stores: otherwise
/// a downstream forwarder could collapse a load to its entry snapshot across an
/// unmodelled aliasing write.
///
/// Its purpose is **deep-deref discovery**. Exposing a read such as `load(param + k)`
/// as a by-value parameter lets a downstream forwarder fold the matching load to it,
/// turning a pointer *loaded* from `param + k` (i.e. `*(param + k)`) into a clean
/// by-value pointer param that a later fully-modelled round can shadow-promote.
/// A function with any store must either take the fully-modelled shadow path or
/// remain unchanged.
///
/// Idempotent: a read whose snapshot param already exists (matched by name) is
/// skipped, so re-visiting an already-partially-promoted function adds nothing and
/// the `mark-pure` `repeat_until = no_change` loop settles. Returns `true` only if a
/// new snapshot was added.
fn apply_partial(
    ctx: &mut Context,
    fid: FunctionId,
    promoted: &[Promoted],
    call_sites: &[InstructionId],
) -> bool {
    if call_sites.is_empty() {
        return false;
    }
    let root = FunctionBody::from_id(ctx, fid)
        .root()
        .map(|b| b.id)
        .unwrap();
    let ram = ctx.shared.default_space;

    // Existing root-param names, used to skip read fields already seeded by a
    // prior round (idempotence).
    let existing: HashSet<String> = BasicBlock::from_id(ctx, root)
        .params()
        .filter_map(|p| p.name().map(str::to_owned))
        .collect();

    // The new read snapshots to add, in a deterministic order shared by the callee
    // (param creation) and every caller (argument order): by arg index, then by
    // read offset (`reads` is already offset-sorted).
    struct NewSnap {
        arg_idx: usize,
        base: ValueId,
        base_size: usize,
        offset: u64,
        size: usize,
        name: String,
    }
    let mut order: Vec<&Promoted> = promoted.iter().collect();
    order.sort_by_key(|p| p.arg_idx);
    let mut new_snaps: Vec<NewSnap> = Vec::new();
    for p in &order {
        for f in &p.reads {
            let name = format!("{}_val_{:x}", p.name, f.offset);
            if existing.contains(&name) {
                continue;
            }
            new_snaps.push(NewSnap {
                arg_idx: p.arg_idx,
                base: p.param,
                base_size: p.base_size,
                offset: f.offset,
                size: f.size,
                name,
            });
        }
    }
    if new_snaps.is_empty() {
        return false;
    }

    // ---- callee: a by-value snapshot param per new read, seeded into real ram ---
    let mut seeds: Vec<(ValueId, usize, u64, ValueId)> = Vec::new();
    for ns in &new_snaps {
        let val_pid = BasicBlock::from_id_mut(ctx, root).push_param(ns.size).id;
        ctx.block_param_mut(val_pid).name = Some(Cow::Owned(ns.name.clone()));
        // An offset-0 snapshot *is* `*base` — record the base slot as its `origin`,
        // so when `base` is a global slot and this snapshot is used as a buffer
        // pointer, alias analysis can treat it as a pointer loaded from that slot
        // (disjoint from the slot under `LoadedPointerDisjointFromSlot`). This lets
        // the in-loop reload forward to this param so a later round region-promotes
        // the buffer. Only the offset-0 snapshot equals `*base` exactly.
        if ns.offset == 0 {
            ctx.block_param_mut(val_pid)
                .set_origin_id(ns.base.localize(val_pid.func));
        }
        seeds.push((
            ns.base,
            ns.base_size,
            ns.offset,
            ValueId::BlockParam(val_pid),
        ));
    }
    {
        let mut b = (ctx).builder(root);
        b.set_insert_point_to_start();
        for (base, base_size, offset, snap) in &seeds {
            let addr = if *offset == 0 {
                *base
            } else {
                let k = b.shr().get_const(*offset, *base_size);
                b.push_add(*base, k).id()
            };
            b.push_store(*snap, addr, ram);
        }
    }

    // ---- callers: append snapshot loads as args -------------------------------
    // Each snapshot is loaded from `arg + offset` in real ram — the same address the
    // callee re-seeds, relocated to the caller (faithful: the base is a by-value
    // param identical on both sides).
    for &call_id in call_sites {
        let Some(call_block) = ctx.get_insn(call_id).parent().map(|b| b.id) else {
            continue;
        };
        let (target, args, clobbers, tag) = match ctx.get_insn(call_id).mnemonic().clone() {
            Mnemonic::Call(c) => (c.target, c.args, c.clobbers, c.tag),
            _ => continue,
        };
        let callee_name = FunctionBody::from_id(ctx, fid).name().to_string();
        let caller_name = BasicBlock::from_id(ctx, call_block)
            .function()
            .map(|f| f.name().to_string())
            .unwrap_or_else(|| "<unknown>".to_string());
        let mut new_args = args.clone();
        {
            let mut b = (ctx).builder(call_block);
            b.set_insert_point_before(call_id);
            for ns in &new_snaps {
                // Invariant: call args are in lockstep with callee root params (see
                // `arg_index_of`). A mismatch means an upstream pass minted a root
                // param without threading the matching argument.
                assert!(
                    ns.arg_idx < args.len(),
                    "[lockstep] call arg count {} < snapshot param index {} for a direct call: \
                     callee {callee_name} (param '{}') ← caller {caller_name}",
                    args.len(),
                    ns.arg_idx,
                    ns.name,
                );
                let base = args[ns.arg_idx].qualify(call_id.func);
                let addr = if ns.offset == 0 {
                    base
                } else {
                    let k = b.shr().get_const(ns.offset, ns.base_size);
                    b.push_add(base, k).id()
                };
                let snap = b.push_load::<false>(addr, ns.size, ram).id();
                new_args.push(snap.localize(call_id.func));
            }
        }
        ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target,
                args: new_args,
                clobbers,
                tag,
            }),
        );
    }
    true
}

#[derive(Default)]
pub struct ArgPromote;

impl Pass for ArgPromote {
    const NAME: &'static str = "argpromote";
    fn description(&self) -> &'static str {
        "Promote by-reference in/out pointer parameters to by-value"
    }
    fn run(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        targets: &[FunctionId],
    ) -> Result<crate::ModulePassOutcome, String> {
        let sp_reg = ctx.shared.registers.get(&env.cfg.stack_pointer).copied();
        let graph = crate::CallGraph::analyze(ctx);
        Ok(
            crate::ModulePassOutcome::functions(argpromote_changed_functions_with_sp(
                ctx, sp_reg, targets, &graph,
            ))
            .preserving_global::<crate::CallGraphAnalysis>()
            .preserving_global::<crate::AddressAnalysis>(),
        )
    }

    fn run_with_analyses(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        targets: &[FunctionId],
        analyses: &mut crate::AnalysisManager,
    ) -> Result<crate::ModulePassOutcome, String> {
        let sp_reg = ctx.shared.registers.get(&env.cfg.stack_pointer).copied();
        let graph = analyses.global::<crate::CallGraphAnalysis>(ctx);
        Ok(
            crate::ModulePassOutcome::functions(argpromote_changed_functions_with_sp(
                ctx, sp_reg, targets, graph,
            ))
            .preserving_global::<crate::CallGraphAnalysis>()
            .preserving_global::<crate::AddressAnalysis>(),
        )
    }
}

crate::register_module_pass!(ArgPromote);
