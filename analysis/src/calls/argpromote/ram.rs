use std::borrow::Cow;
use std::collections::HashSet;

use rustc_hash::FxHashSet;

use qcode::{
    assumption::Proposition,
    builder::Builder,
    context::Context,
    space::LocalMemorySpaceId,
    types::TypeId,
    value::{
        BasicBlock, FunctionBody, FunctionId, TempSpace, Value, ValueId, VarnodeId,
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
struct OwnFrame {
    sp_param: Option<ValueId>,
    numbering: Numbering,
}

impl OwnFrame {
    fn new(ctx: &Context, fid: FunctionId, sp_reg: Option<VarnodeId>) -> Self {
        let sp_param =
            sp_reg.and_then(|r| incoming_sp_param(qcode::value::ModuleView::new(ctx), fid, r));
        Self {
            sp_param,
            numbering: precompute_forms(qcode::value::ModuleView::new(ctx), fid),
        }
    }

    /// Whether `addr` points into this function's own frame (classified
    /// [`FrameClass::Local`]).
    fn is_local(&self, ctx: &Context, addr: ValueId) -> bool {
        self.sp_param.is_some_and(|sp| {
            frame_class(
                qcode::value::ModuleView::new(ctx),
                &self.numbering,
                sp,
                addr,
            ) == Some(FrameClass::Local)
        })
    }

    /// Whether `addr` is any `@SP`-rooted frame slot — an own-frame local or a
    /// caller-frame slot (`@SP + k`, `k ≥ 0`).
    fn is_frame_slot(&self, ctx: &Context, addr: ValueId) -> bool {
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
    // Callee-before-caller order: a function may keep a call to a memory-free
    // callee (see [`function_makes_blocking_call`]), and that callee must already
    // be promoted — its own loads gone — for the caller to qualify. One visit per
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
        // Lift constant-address (global) accesses into params first, so the freshly
        // param-relative derefs are visible to `try_promote`'s footprint scan in the
        // same visit.
        if super::globals::globalize_constants(ctx, &address_taken, fid) {
            changed.insert(fid);
            changed.extend(callers.iter().copied());
        }
        if try_promote(ctx, fid, sp_reg, &address_taken, graph) {
            changed.insert(fid);
            changed.extend(callers);
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

fn try_promote(
    ctx: &mut Context,
    fid: FunctionId,
    sp_reg: Option<VarnodeId>,
    address_taken: &FxHashSet<FunctionId>,
    graph: &crate::CallGraph,
) -> bool {
    let f = FunctionBody::from_id(ctx, fid);
    if f.is_external() {
        return false;
    }

    // Only functionalize functions whose call interface this pass owns. The whole
    // RAM channel keys snapshot args off the `param[i] ↔ Call.args[i]` lockstep
    // (see [`arg_index_of`]), and that lockstep is an invariant *only* for
    // `pure_reg` functions — the ones `argpromote_registers` established it for. A
    // conventional (non-`pure_reg`) function uses the register ABI (`input_regs`),
    // not positional params, and accrues root params (live-in registers promoted by
    // mem2reg, an incoming `@SP` param, …) that no caller passes positionally. Trying
    // to snapshot a deref through such a param reads a `Call.args` slot that does not
    // exist. So skip them: a conventional function is left for the legacy path, which
    // is correct (the body is unchanged). `argpromote_registers` runs earlier and
    // makes every functionalizable function `pure_reg`, so this loses no real work.
    if !f.is_reg_materialized() {
        return false;
    }

    let Some(root) = f.root().map(|b| b.id) else {
        return false;
    };

    // Closed-world / direct-only gate: if the function's address is taken it may
    // be reached by an indirect call this pass cannot find and rewrite, leaving a
    // caller on the old by-reference ABI. (Callers in undiscovered code are an
    // accepted, unguardable gap — see the module docs.)
    if address_taken.contains(&fid) {
        return false;
    }

    // A call composes with our shadow promotion only if it is fully inert toward
    // memory: a *direct* call, with no clobbers (writes nothing the caller sees),
    // to a callee that reads no memory (so it cannot dereference any promoted
    // pointer we hand it, and there is no read/write alias with our shadow). Any
    // other call — indirect, clobbering, or memory-reading — would have to bubble
    // its effects through ours, which is out of scope; bail.
    if function_makes_blocking_call(ctx, fid) {
        return false;
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
        return false;
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

    let mut promoted: Vec<Promoted> = Vec::new();
    for (param, name) in candidates {
        let info = analyze_param(ctx, &numbering, &accesses_in, param);
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
    if !all_accesses_modelled(ctx, fid, &promoted)
        || !all_writes_resolvable(ctx, fid, &promoted, sp_reg)
        || !regions_disjoint(ctx, fid, &promoted, sp_reg)
    {
        if accesses_in.iter().any(|access| access.is_store) {
            qcode::pass_log!(
                debug,
                "argpromote {}: bail — partial promotion with stores may forward across aliases",
                FunctionBody::from_id(ctx, fid).name(),
            );
            return false;
        }
        qcode::pass_log!(
            debug,
            "argpromote {}: partial (inputs-only) — footprint not fully modelled",
            FunctionBody::from_id(ctx, fid).name(),
        );
        let call_sites = crate::calls::direct_call_sites(ctx, graph, fid);
        return apply_partial(ctx, fid, &promoted, &call_sites);
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
    if promoted
        .iter()
        .all(|p| p.reads.is_empty() && p.write_targets.is_empty() && p.region.is_none())
    {
        return false;
    }

    let call_sites = crate::calls::direct_call_sites(ctx, graph, fid);
    apply(ctx, fid, promoted, sp_reg, &call_sites)
}

/// Whether every real-memory (default-space) load/store in `fid` is captured by
/// `promoted` — i.e. will be redirected into the shadow. Accesses already in a
/// shadow space (a prior promotion round) are inherently modelled and skipped, so
/// the check is idempotent. A single uncaptured ram access fails it: see the
/// all-or-nothing rationale at the call site.
fn all_accesses_modelled(ctx: &Context, fid: FunctionId, promoted: &[Promoted]) -> bool {
    let ram = ctx.shared.default_space;
    let captured: HashSet<InstructionId> = promoted
        .iter()
        .flat_map(|p| p.accesses.iter().copied())
        .collect();
    FunctionBody::from_id(ctx, fid).iter().all(|block| {
        block.iter().all(|insn| {
            let space = match insn.mnemonic() {
                Mnemonic::Load(l) => l.space,
                Mnemonic::Store(s) => s.space,
                // Not a memory access — nothing to model.
                _ => return true,
            };
            space != ram || captured.contains(&insn.id)
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

/// `true` if `function_id` makes any call (direct or indirect).
fn function_makes_blocking_call(ctx: &Context, function_id: FunctionId) -> bool {
    FunctionBody::from_id(ctx, function_id).blocks().any(|b| {
        b.iter().any(|i| match i.mnemonic() {
            // Indirect transfers: target unknown, cannot vet.
            Mnemonic::CallInd(_) => true,
            // A direct call is inert iff it clobbers nothing and the callee
            // touches no memory; otherwise its effects would have to bubble
            // through ours.
            Mnemonic::Call(c) => {
                !c.clobbers.is_empty()
                    || c.target
                        .real()
                        .is_none_or(|target| function_accesses_memory(ctx, target))
            }
            _ => false,
        })
    })
}

/// `true` if `function_id`'s body contains any memory access (**load or store**).
/// A memory-free callee cannot dereference a pointer passed to it — for reading
/// *or writing* — so handing it a promoted pointer is safe (it can neither observe
/// our shadow, alias it, nor mutate a promoted address behind it). A store *is* a
/// dereference, so checking loads alone would let a store-only callee silently
/// write a promoted address our shadow no longer maintains.
fn function_accesses_memory(ctx: &Context, function_id: FunctionId) -> bool {
    FunctionBody::from_id(ctx, function_id).blocks().any(|b| {
        b.iter()
            .any(|i| matches!(i.mnemonic(), Mnemonic::Load(_) | Mnemonic::Store(_)))
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
    sp_reg: Option<VarnodeId>,
    call_sites: &[InstructionId],
) -> bool {
    if call_sites.is_empty() {
        return false;
    }

    let ram = ctx.shared.default_space;
    let default = ctx.shared.space(ram);
    let (word_size, addr_size) = (default.word_size, default.addr_size);
    let shadow =
        ctx.bodies[fid].push_temp_space(TempSpace::new(Some("argpromote"), word_size, addr_size));
    let shadow = LocalMemorySpaceId::Temp(shadow.local);
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
    // `(base_param, base_size, signed offset, size)`. A negative offset (`base - k`,
    // e.g. an own-frame slot when there is no stack-pointer to recognise it) is
    // encoded two's-complement into the address const, so `base + offset` wraps to
    // `base - k`.
    let mut write_slots: Vec<(ValueId, usize, i64, usize)> = Vec::new();
    for p in &promoted {
        for &(addr, size) in &p.write_targets {
            if own_frame.is_local(ctx, addr) {
                continue;
            }
            let (base, offset) = numbering
                .base_offset(qcode::value::ModuleView::new(&*ctx), addr)
                .unwrap_or((addr, 0));
            let base_size = promoted
                .iter()
                .find(|q| q.param == base)
                .map_or(p.base_size, |q| q.base_size);
            if !write_slots
                .iter()
                .any(|&(b, _, o, _)| b == base && o == offset)
            {
                write_slots.push((base, base_size, offset, size));
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
            write_slots.push((p.param, p.base_size, r.base_off as i64, r.byte_len()));
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
    for &(base, base_size, offset, size) in &write_slots {
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
        |b, &(base, base_size, offset, size)| {
            let addr = seed_addr(b, base, base_size, offset);
            let v = b.push_load::<false>(addr, size, shadow).id();
            vec![addr, v]
        },
        |b, _, ext| {
            b.push_store(ext[1], ext[0], ram);
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
        let (target, args, clobbers) = match ctx.get_insn(call_id).mnemonic().clone() {
            Mnemonic::Call(c) => (c.target, c.args, c.clobbers),
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
                tag: Default::default(),
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
