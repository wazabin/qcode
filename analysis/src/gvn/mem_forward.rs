//! Byte-level store-to-load forwarding for GVN.
//!
//! [`MemForward`] is the single block-state object that subsumes every memory
//! forwarding case the GVN block walk used to handle inline:
//!
//! - **Exact forward** — a same-width store/load of the same location.
//! - **Wide store → narrow load** — extract a sub-register with a [`Range`].
//! - **Narrow stores → wide load** (*coalesce*) — rebuild a value covered
//!   byte-for-byte by several partial writes, e.g. the x86 `xor eax,eax;
//!   setnz al; push eax` idiom.
//!
//! All three are the same operation: forwarding over an alias-resolved byte
//! run. Every cell is keyed by a [`Base`] plus a signed byte offset:
//!
//! - **`Base::Pinned`** — a pointer the [`AliasResult`] resolves to a concrete
//!   interval (registers, constant-folded stack slots); the offset is the
//!   absolute byte address, exactly as the flat `(space, addr)` map used to be.
//! - **`Base::Symbolic`** — a RAM pointer with no pinned interval, identified by
//!   its affine base value (from the precomputed [`Numbering`]); the offset is
//!   the signed affine constant. `(p + 4) - 4` and `p` share a base, so a
//!   `store(p, …)` forwards to a `load(p + c, …)` for `c` inside the store.
//!   With no oracle every pointer becomes its own degenerate symbolic base
//!   (offset 0), so only exact same-pointer reads forward.
//!
//! The whole structure is cloned and threaded down the dominator tree by
//! the dominator walk in [`super::walk`], so forwarding works across blocks; the pruning
//! methods drop entries a call clobbers or a loop back-edge invalidates.

use rustc_hash::FxHashMap as HashMap;

use jstd::graph::analysis::DominatorTree;

use crate::{AliasResult, ContextView, FunctionBody};
#[cfg(test)]
use qcode::context::Context;
use qcode::{
    assumption::Proposition,
    space::{LocalMemorySpaceId, Space, SpaceId, SpaceType},
    value::{
        FunctionId, LocalValueId, QCodeView, Value, ValueId, ValueRef, Varnode, VarnodeId,
        block::BlockId,
        insn::{Binary, Binop, IntBinop, Load, Mnemonic, Range, Store, Zext},
    },
};

use super::affine::Numbering;

/// One byte of forwarded memory: it equals byte `src_off` of value `src`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Cell {
    src: ValueId,
    src_off: usize,
}

/// A maximal run of consecutive load bytes backed by one source value.
///
/// `load_off` is the byte offset within the load; `size` bytes of it equal
/// bytes `src_off..src_off + size` of `src`.
#[derive(Clone, Copy, Debug)]
struct Segment {
    load_off: usize,
    size: usize,
    src: ValueId,
    src_off: usize,
}

/// The registers a call may clobber.
enum CallClobbers {
    /// Unknown target or a target with no recorded clobber set: every register.
    AllRegisters,
    /// A direct call's recorded per-callee clobber set.
    Regs(Vec<VarnodeId>),
}

/// The identity of a byte-addressable region the byte map keys on.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Base {
    /// A pointer the oracle pins to a concrete interval. All pinned pointers in
    /// a space share this base; the map offset is the absolute byte address.
    Pinned(LocalMemorySpaceId),
    /// A pointer with no pinned interval, identified by its affine base value.
    /// The map offset is the signed affine constant relative to that base.
    Symbolic(LocalMemorySpaceId, ValueId),
}

impl Base {
    fn space(self) -> LocalMemorySpaceId {
        match self {
            Base::Pinned(s) | Base::Symbolic(s, _) => s,
        }
    }
}

/// Decompose `ptr` (read/written in memory `space`) into its base identity and
/// the signed byte offset of its first byte within that base.
fn locate<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    ptr: ValueId,
    space: LocalMemorySpaceId,
    aliases: Option<&AliasResult>,
    numbering: &Numbering,
) -> (Base, i64) {
    if let Some((sp, start, _end)) = aliases.and_then(|a| a.interval(ptr)) {
        return (Base::Pinned(sp), start as i64);
    }
    match numbering.base_offset(host, ptr) {
        Some((base, off)) => (Base::Symbolic(space, base), off),
        None => (Base::Symbolic(space, ptr), 0),
    }
}

/// Whether the oracle gives `a` and `b` distinct, concrete alias classes —
/// *positive* proof they never refer to the same location. Absence of class
/// info (isolated values) or an `Unknown` class is **not** proof: such pairs
/// are treated as possibly-aliasing, so a forwarded cell is dropped on any
/// doubt.
fn proven_disjoint<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    aliases: Option<&AliasResult>,
    a: ValueId,
    b: ValueId,
) -> bool {
    let Some(aliases) = aliases else { return false };
    // Frame freshness: an own-frame local and an incoming pointer never alias.
    if aliases.provably_disjoint(host, a, b) {
        return true;
    }
    matches!(
        (aliases.alias_class(a), aliases.alias_class(b)),
        (Some(crate::alias::NodeId::Id(x)), Some(crate::alias::NodeId::Id(y))) if x != y
    )
}

/// Whether a cell at base `cb` is provably disjoint from an access at `base`
/// (whose pointer value is `ptr`), i.e. the cell can be kept across the access.
/// Only meaningful when `cb != base`. Conservative: returns `false` (not
/// disjoint) whenever disjointness cannot be proven.
fn cross_base_disjoint<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    aliases: Option<&AliasResult>,
    ptr: ValueId,
    base: Base,
    cb: Base,
) -> bool {
    // Different address spaces never overlap.
    if base.space() != cb.space() {
        return true;
    }
    match (base, cb) {
        // Symbolic vs symbolic: disjoint only if the oracle proves the base
        // values occupy different locations.
        (Base::Symbolic(_, sv), Base::Symbolic(_, ov)) => proven_disjoint(host, aliases, sv, ov),
        // Pinned access vs a symbolic cell: compare the access pointer against
        // the cell's base value.
        (Base::Pinned(_), Base::Symbolic(_, ov)) => proven_disjoint(host, aliases, ptr, ov),
        // Symbolic access vs a pinned cell: the cell's absolute address has no
        // value to query, so we cannot prove disjointness.
        (Base::Symbolic(_, _), Base::Pinned(_)) => false,
        // Two pinned bases in the same space are equal (caller's `cb == base`
        // check already kept them); unreachable here.
        (Base::Pinned(_), Base::Pinned(_)) => true,
    }
}

#[derive(Clone, Default)]
pub(super) struct MemForward {
    /// Per-byte forwarding, keyed by base identity and a signed byte offset.
    byte_map: HashMap<(Base, i64), Cell>,
}

impl MemForward {
    /// Record that, after this load executes, `value` is held at `load`'s
    /// location. Called for every load (forwarded or not) so a later identical
    /// load can reuse the result.
    pub(super) fn define_load<'ctx, 'str: 'ctx>(
        &mut self,
        host: impl QCodeView<'ctx, 'str>,
        func: FunctionId,
        load: &Load,
        value: ValueId,
        aliases: Option<&AliasResult>,
        numbering: &Numbering,
    ) {
        let (base, start) = locate(host, load.ptr.qualify(func), load.space, aliases, numbering);
        for (i, off) in (start..start + load.size as i64).enumerate() {
            self.byte_map.insert(
                (base, off),
                Cell {
                    src: value,
                    src_off: i,
                },
            );
        }
    }

    /// Group bytes `start..end` of `base` into maximal single-source segments,
    /// or `None` if any byte is unmapped (coverage gap — not forwardable).
    fn segments(&self, base: Base, start: i64, end: i64) -> Option<Vec<Segment>> {
        let mut segments: Vec<Segment> = Vec::new();
        for off in start..end {
            let cell = self.byte_map.get(&(base, off)).copied()?;
            let load_off = (off - start) as usize;
            match segments.last_mut() {
                // Extend the run when it continues the same source contiguously.
                Some(seg) if seg.src == cell.src && seg.src_off + seg.size == cell.src_off => {
                    seg.size += 1;
                }
                _ => segments.push(Segment {
                    load_off,
                    size: 1,
                    src: cell.src,
                    src_off: cell.src_off,
                }),
            }
        }
        Some(segments)
    }

    // -----------------------------------------------------------------------
    // Body-local forwarding helpers.
    // -----------------------------------------------------------------------

    /// Record the bytes written by a store.
    pub(super) fn record_store<'str>(
        &mut self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        func: FunctionId,
        store: &Store,
        aliases: Option<&AliasResult>,
        numbering: &Numbering,
    ) {
        let store_ptr = store.ptr.qualify(func);
        let store_src = store.src.qualify(func);
        let (base, start) = locate(
            cx.body_view(body),
            store_ptr,
            store.space,
            aliases,
            numbering,
        );
        let end = start + store.size as i64;

        self.byte_map.retain(|&(cb, _), _| {
            cb == base || cross_base_disjoint(cx.body_view(body), aliases, store_ptr, base, cb)
        });

        for off in start..end {
            self.byte_map.remove(&(base, off));
        }
        let covered = ValueRef::from_view(cx.body_view(body), store_src)
            .size()
            .min(store.size);
        for (i, off) in (start..start + covered as i64).enumerate() {
            self.byte_map.insert(
                (base, off),
                Cell {
                    src: store_src,
                    src_off: i,
                },
            );
        }
        if covered < store.size {
            let zero = cx
                .body_view(body)
                .shared()
                .get_const(0, store.size - covered);
            for (i, off) in (start + covered as i64..end).enumerate() {
                self.byte_map.insert(
                    (base, off),
                    Cell {
                        src: zero,
                        src_off: i,
                    },
                );
            }
        }
    }

    /// Rebuild a load when every requested byte is known.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_load<'str>(
        &mut self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        block_id: BlockId,
        insn_id: qcode::value::InstructionId,
        load: &Load,
        aliases: Option<&AliasResult>,
        numbering: &Numbering,
    ) -> Option<ValueId> {
        let (base, start) = locate(
            cx.body_view(body),
            load.ptr.qualify(insn_id.func),
            load.space,
            aliases,
            numbering,
        );
        let end = start + load.size as i64;
        let segments = self.segments(base, start, end)?;
        let load_size = load.size;

        let value = if segments.len() == 1
            && segments[0].load_off == 0
            && segments[0].size == load_size
            && segments[0].src_off == 0
            && ValueRef::from_view(cx.body_view(body), segments[0].src).size() == load_size
        {
            segments[0].src
        } else if load_size > 8 {
            self.rebuild_bytes(body, cx, &segments, load_size)?
        } else {
            self.rebuild(body, cx, block_id, insn_id, &segments, load_size)
        };
        Some(value)
    }

    /// Rebuild a scalar load from known segments.
    fn rebuild<'str>(
        &self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        block_id: BlockId,
        insn_id: qcode::value::InstructionId,
        segments: &[Segment],
        load_size: usize,
    ) -> ValueId {
        let mut acc: Option<ValueId> = None;
        for seg in segments {
            let piece = self.build_piece(body, cx, block_id, insn_id, seg, load_size);
            acc = Some(match acc {
                None => piece,
                Some(lhs) => {
                    let or = body.push_mnemonic(
                        cx.shr(),
                        Mnemonic::Binop(Binary {
                            op: Binop::Int(IntBinop::Or),
                            lhs: lhs.localize(block_id.func),
                            rhs: piece.localize(block_id.func),
                        }),
                        load_size,
                    );
                    body.insert_insn_before(block_id, insn_id, or);
                    or.into()
                }
            });
        }
        acc.expect("a fully-covered load has at least one segment")
    }

    /// Rebuild a wide constant load as a byte value.
    fn rebuild_bytes<'str>(
        &self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        segments: &[Segment],
        load_size: usize,
    ) -> Option<ValueId> {
        let ctx = cx.body_view(body).shared();
        let mut buf = vec![0u8; load_size];
        for seg in segments {
            let bytes: Vec<u8> = match seg.src {
                ValueId::Literal(lid) => {
                    let lit = &ctx.values.literals[lid];
                    if lit.symbolic.is_some() {
                        return None;
                    }
                    let masked = qcode::value::LiteralRef::from_id(ctx, lid).value();
                    let le = masked.to_le_bytes();
                    le.get(seg.src_off..seg.src_off + seg.size)?.to_vec()
                }
                ValueId::Bytes(bid) => {
                    let data = &ctx.values.bytes[bid].data;
                    data.get(seg.src_off..seg.src_off + seg.size)?.to_vec()
                }
                _ => return None,
            };
            buf.get_mut(seg.load_off..seg.load_off + seg.size)?
                .copy_from_slice(&bytes);
        }
        Some(ctx.get_bytes(buf))
    }

    /// Materialize one segment of a reconstructed scalar load.
    fn build_piece<'str>(
        &self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        block_id: BlockId,
        insn_id: qcode::value::InstructionId,
        seg: &Segment,
        load_size: usize,
    ) -> ValueId {
        let src_size = ValueRef::from_view(cx.body_view(body), seg.src).size();
        let extracted = if seg.src_off == 0 && src_size == seg.size {
            seg.src
        } else {
            let r = body.push_mnemonic(
                cx.shr(),
                Mnemonic::Range(Range {
                    src: seg.src.localize(block_id.func),
                    start: seg.src_off,
                    size: seg.size,
                }),
                seg.size,
            );
            body.insert_insn_before(block_id, insn_id, r);
            r.into()
        };

        let widened = if seg.size == load_size {
            extracted
        } else {
            let z = body.push_mnemonic(
                cx.shr(),
                Mnemonic::Zext(Zext {
                    src: extracted.localize(block_id.func),
                    size: load_size,
                }),
                load_size,
            );
            body.insert_insn_before(block_id, insn_id, z);
            z.into()
        };

        if seg.load_off == 0 {
            return widened;
        }
        let shamt = cx
            .body_view(body)
            .shared()
            .get_const((seg.load_off * 8) as u64, load_size);
        let s = body.push_mnemonic(
            cx.shr(),
            Mnemonic::Binop(Binary {
                op: Binop::Int(IntBinop::ShiftLeft),
                lhs: widened.localize(block_id.func),
                rhs: shamt.localize(block_id.func),
            }),
            load_size,
        );
        body.insert_insn_before(block_id, insn_id, s);
        s.into()
    }

    /// Drop everything the call terminating `block_id` (if any) may clobber, so
    /// no forwarded register value survives across it. Non-call blocks are no-ops.
    pub(super) fn prune_clobbered_by_call<'ctx, 'str: 'ctx>(
        &mut self,
        host: impl QCodeView<'ctx, 'str>,
        block_id: BlockId,
        aliases: Option<&AliasResult>,
    ) {
        let term = host
            .block_ref(block_id)
            .iter()
            .last()
            .map(|i| i.mnemonic().clone());
        // Pointers that escape into the callee (its arguments plus every
        // recorded clobbered/aliased location). The callee may store through any
        // of them, so a forwarded RAM cell one of them may reach is stale after
        // the call. `interval(p)` pins those resolvable to a concrete byte range
        // (e.g. `&local`); a pointer that does not pin is symbolic and, like an
        // unresolved store, must be treated as possibly reaching every pinned
        // RAM cell (see [`cross_base_disjoint`]).
        // `unknown_callee` is set for an indirect call: its target is unknown, so
        // it may write to any pinned RAM cell (a global or stack slot) even
        // without receiving a pointer to it. A direct call's memory writes are all
        // recorded in its `clobbers`, so only its escaping pointers matter.
        // The non-register spaces the callee may (transitively) store to. `Some`
        // is an exact witnessed set: a RAM cell whose space is absent cannot be
        // clobbered by this call, regardless of what pointers escape — so it
        // survives. `None` (an indirect callee, or one with no recorded summary)
        // is unbounded, falling through to the conservative escape-based prune.
        // This is what lets a functionalized callee that writes only its own
        // private scratch space leave the caller's spilled-pointer cell intact.
        let callee_written_spaces: Option<Vec<SpaceId>> = match &term {
            Some(Mnemonic::Call(call)) => call.target.real().and_then(|target| {
                host.interface(target)
                    .signature
                    .as_ref()
                    .and_then(|s| s.written_spaces.as_deref())
                    .map(<[_]>::to_vec)
            }),
            _ => None,
        };

        let qual = |v: LocalValueId| v.qualify(block_id.func);
        let (clobbers, escaping, unknown_callee): (CallClobbers, Vec<ValueId>, bool) = match &term {
            Some(Mnemonic::CallInd(call)) => (
                CallClobbers::AllRegisters,
                call.args.iter().map(|&a| qual(a)).collect(),
                true,
            ),
            Some(Mnemonic::Call(call)) => match call.target.real() {
                None => (
                    CallClobbers::AllRegisters,
                    call.args
                        .iter()
                        .chain(&call.clobbers)
                        .map(|&v| qual(v))
                        .collect(),
                    true,
                ),
                Some(target) => {
                    let callee = host.interface(target);
                    let regs = match callee
                        .signature
                        .as_ref()
                        .and_then(|s| s.clobbered.as_deref())
                    {
                        Some(regs) => CallClobbers::Regs(regs.to_vec()),
                        None => CallClobbers::AllRegisters,
                    };
                    // An argument flowing into a `readonly` callee param is never
                    // written through, so it does not clobber the RAM cells it may
                    // reach — exclude it from the escaping set. (Only `readonly` is
                    // consulted here; a `readonly` pointer the callee merely reads
                    // cannot invalidate a pinned cell even if it is captured.) The
                    // recorded `clobbers` are writes by definition and always escape.
                    let escaping = call
                        .args
                        .iter()
                        .enumerate()
                        .filter(|&(j, _)| {
                            !callee
                                .signature
                                .as_ref()
                                .and_then(|s| s.param_attrs.as_ref())
                                .and_then(|attrs| attrs.get(j))
                                .is_some_and(|a| a.readonly)
                        })
                        .map(|(_, &v)| qual(v))
                        .chain(call.clobbers.iter().map(|&c| qual(c)))
                        .collect();
                    (regs, escaping, false)
                }
            },
            _ => return,
        };

        // A pinned RAM cell at `(space, off)` is clobbered by the call if some
        // escaping pointer may write to it: one whose pinned interval overlaps it,
        // or any pointer that does not pin (symbolic — reach is unknown, so it may
        // alias any pinned cell). An unknown callee clobbers every pinned RAM cell.
        let symbolic_escape = unknown_callee
            || match aliases {
                Some(a) => escaping.iter().any(|&p| a.interval(p).is_none()),
                // No oracle: every escaping pointer is unresolved, so treat any
                // escape as possibly reaching every pinned RAM cell.
                None => !escaping.is_empty(),
            };
        let pinned_ram_clobbered = |space: LocalMemorySpaceId, off: i64| {
            if symbolic_escape {
                return true;
            }
            let Some(a) = aliases else { return false };
            escaping.iter().any(|&p| match a.interval(p) {
                Some((sp, start, end)) => sp == space && (start as i64) <= off && off < end as i64,
                None => false,
            })
        };

        let is_reg = |space: LocalMemorySpaceId| {
            space.shared().is_some_and(|space| {
                matches!(Space::from_id(host.shared(), space).ty, SpaceType::Register)
            })
        };

        // PROTOTYPE (realigned-frame): frame freshness extended from stores to
        // calls. A symbolic RAM cell whose base is an own-frame local the callee
        // never received a pointer to cannot be written by it — the callee's frame
        // is younger and it holds no pointer into ours. Keep such a cell across the
        // call; this is what lets a realigned-frame spill (whose base is the
        // `@SP & -mask` anchor, classified `Local`) survive intervening calls.
        // Conservative for an unknown (indirect) callee, which has no clobber
        // summary, and requires every escaping pointer to be provably disjoint from
        // the base.
        let own_frame_survives = |bv: ValueId| -> bool {
            if unknown_callee {
                return false;
            }
            let Some(a) = aliases else { return false };
            a.is_own_frame_local(bv) && escaping.iter().all(|&p| a.provably_disjoint(host, p, bv))
        };

        self.byte_map.retain(|&(base, off), _| {
            let space = base.space();
            if !is_reg(space) {
                // A body-local scratch space is owned by the caller. No callee,
                // including an unresolved indirect one, can name or mutate it.
                if space.shared().is_none() {
                    return true;
                }
                // A direct callee with a witnessed write-set that excludes this
                // space cannot touch the cell no matter what escapes into it, so
                // keep it. This is a per-callee memory summary, sound and needing
                // no frame reasoning: a `pure_reg` callee writing only its private
                // scratch space leaves the caller's real-`ram` cells intact.
                if let Some(ws) = &callee_written_spaces {
                    match space.shared() {
                        Some(space) if !ws.contains(&space) => return true,
                        Some(_) => {}
                        None => unreachable!("local spaces returned above"),
                    }
                }
                // RAM: a call may write through any symbolic pointer (no memory
                // summary exists), so drop symbolic RAM cells — except an own-frame
                // slot the callee cannot reach (see `own_frame_survives`). A pinned
                // RAM cell (a resolved stack slot) survives unless a pointer to it
                // escaped into the callee, which may then store through it.
                return match base {
                    Base::Symbolic(_, bv) => own_frame_survives(bv),
                    Base::Pinned(_) => !pinned_ram_clobbered(space, off),
                };
            }
            // Register cells: drop those the call clobbers.
            match base {
                Base::Pinned(_) => match &clobbers {
                    CallClobbers::AllRegisters => false,
                    CallClobbers::Regs(regs) => !regs.iter().any(|&r| {
                        let vn = Varnode::from_id(host.shared(), r);
                        vn.space().id == space
                            && vn.address() <= off
                            && off < vn.address() + vn.size() as i64
                    }),
                },
                Base::Symbolic(_, bv) => match &clobbers {
                    CallClobbers::AllRegisters => false,
                    CallClobbers::Regs(regs) => !regs.iter().any(|&r| match aliases {
                        Some(a) => a.may_alias(host, bv, ValueId::Varnode(r)),
                        None => true,
                    }),
                },
            }
        });
    }

    /// Whether spilled-pointer-reload peeling is sound for `block_id`'s function:
    /// it rides [`Proposition::LoadedPointerDisjointFromSlot`] (a value loaded from
    /// a slot is disjoint from that slot — no self-referential `*pp == &pp`), the
    /// same assumption the alias oracle's spilled-reload rules use. Off → the
    /// conservative opaque-pointer prune.
    pub(super) fn loaded_ptr_peeling<'ctx, 'str: 'ctx>(
        host: impl QCodeView<'ctx, 'str>,
        block_id: BlockId,
    ) -> bool {
        host.block_ref(block_id)
            .function()
            .map(|f| f.id)
            .is_some_and(|fid| {
                host.shared()
                    .truth(Proposition::LoadedPointerDisjointFromSlot(fid))
                    .is_some_and(|t| t.value)
            })
    }

    /// If `v` is a load whose location is, in the current byte map, fully covered
    /// by a single source value of the same width (an *exact* reload), return that
    /// value. This is the same single-segment cover [`try_load`] forwards on, used
    /// here to peel a spilled-pointer reload before disjointness testing.
    fn reload_forwards_to<'ctx, 'str: 'ctx>(
        &self,
        host: impl QCodeView<'ctx, 'str>,
        v: ValueId,
        aliases: Option<&AliasResult>,
        numbering: &Numbering,
    ) -> Option<ValueId> {
        let ValueId::Instruction(id) = v else {
            return None;
        };
        let Mnemonic::Load(load) = host.insn_ref(id).mnemonic().clone() else {
            return None;
        };
        let (base, start) = locate(
            host,
            load.ptr.qualify(id.func),
            load.space,
            aliases,
            numbering,
        );
        let segs = self.segments(base, start, start + load.size as i64)?;
        let [seg] = segs.as_slice() else { return None };
        (seg.load_off == 0
            && seg.size == load.size
            && seg.src_off == 0
            && ValueRef::from_view(host, seg.src).size() == load.size)
            .then_some(seg.src)
    }

    /// Locate `ptr`, peeling its affine base through spilled-pointer reloads: when
    /// the base is a reload of a cell the (inherited) byte map already resolves to
    /// a concrete value, substitute that value and re-locate. A store through
    /// `%buf = load(slot)` thus resolves to the address `slot` was spilled with —
    /// e.g. `(@SP - 0x1a0) + k` — so the loop-carried prune sees it as a precise
    /// frame interval, provably disjoint from an unrelated slot, instead of an
    /// opaque pointer that pessimistically clobbers every inherited cell. This is
    /// what breaks the chicken-and-egg where the buffer-fill stores (through the
    /// not-yet-forwarded reload) would otherwise drop the very spill cell the
    /// reload needs to forward.
    fn resolve_loaded_ptr<'ctx, 'str: 'ctx>(
        &self,
        host: impl QCodeView<'ctx, 'str>,
        ptr: ValueId,
        space: LocalMemorySpaceId,
        aliases: Option<&AliasResult>,
        numbering: &Numbering,
        peel: bool,
    ) -> (Base, i64) {
        let (mut base, mut off) = locate(host, ptr, space, aliases, numbering);
        if !peel {
            return (base, off);
        }
        // Bounded peel of reload chains (a spill of a spill, …).
        for _ in 0..8 {
            let Base::Symbolic(_, basev) = base else {
                break;
            };
            let Some(value) = self.reload_forwards_to(host, basev, aliases, numbering) else {
                break;
            };
            let (b2, o2) = locate(host, value, space, aliases, numbering);
            base = b2;
            off += o2;
        }
        (base, off)
    }

    /// At a loop header, drop forwarded values a store inside the loop may
    /// overwrite on a later iteration. Non-headers are left untouched.
    pub(super) fn prune_loop_carried<'ctx, 'str: 'ctx>(
        &mut self,
        host: impl QCodeView<'ctx, 'str>,
        block_id: BlockId,
        tree: &DominatorTree<BlockId>,
        aliases: Option<&AliasResult>,
        numbering: &Numbering,
    ) {
        // `dominates` is reflexive, so a self-loop also counts.
        let is_loop_header = host
            .block_ref(block_id)
            .predecessors()
            .any(|(_, pred)| tree.dominates(block_id, pred));
        if !is_loop_header || self.byte_map.is_empty() {
            return;
        }

        // Every block the header dominates is a candidate loop body: a store
        // there can clobber the inherited value on a later iteration (the header
        // itself is included — a store after the load but before the back edge
        // counts). Walk the dominator-tree subtree rooted at the header rather
        // than scanning — and dominance-testing — every block in the whole
        // program, which is O(program) per loop header and made gvn scale with
        // total lifted code instead of the current function.
        //
        // A tail-call edge is a real CFG edge, so the dominator subtree can reach
        // blocks owned by the callee. Those foreign blocks are not described by
        // this function's alias oracle — `may_alias` returns false for any pointer
        // it never scanned — so a foreign store there would be silently treated as
        // non-clobbering and leave a stale value forwarded across the loop. Skip
        // foreign blocks' stores entirely (they are the callee's concern, walked by
        // its own owner), but keep traversing through them to reach any owned
        // descendant, whose stores this function *does* reason about.
        let owner = ctx.values.basic_blocks[block_id].parent;
        let mut body = vec![block_id];
        let mut frontier = vec![block_id];
        while let Some(b) = frontier.pop() {
            for &child in tree.children_of(b) {
                if ctx.values.basic_blocks[child].parent == owner {
                    body.push(child);
                }
                frontier.push(child);
            }
        }
        let stores: Vec<Store> = body
            .iter()
            .flat_map(|&b| {
                host.block_ref(b)
                    .iter()
                    .filter_map(|insn| match insn.mnemonic() {
                        Mnemonic::Store(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let peel = Self::loaded_ptr_peeling(host, block_id);

        // Resolve each store's location once, peeling spilled-pointer reloads
        // against the inherited byte map (see `resolve_loaded_ptr`). `rep` is the
        // representative pointer value for the resolved base, used by
        // `cross_base_disjoint`'s pinned-vs-symbolic arm.
        let store_locs: Vec<(Base, i64, i64, ValueId)> = stores
            .iter()
            .map(|store| {
                let store_ptr = store.ptr.qualify(block_id.func);
                let (sb, s) =
                    self.resolve_loaded_ptr(host, store_ptr, store.space, aliases, numbering, peel);
                let rep = match sb {
                    Base::Symbolic(_, bv) => bv,
                    Base::Pinned(_) => store_ptr,
                };
                (sb, s, store.size as i64, rep)
            })
            .collect();

        self.byte_map.retain(|&(base, off), _| {
            !store_locs.iter().any(|&(sb, s, size, rep)| {
                if sb == base {
                    // Same base: overwritten only on the bytes it covers.
                    s <= off && off < s + size
                } else {
                    // Other base: may overwrite unless provably disjoint.
                    !cross_base_disjoint(host, aliases, rep, sb, base)
                }
            })
        });
    }

    /// Clear all forwarded state (used for blocks reachable from more than one
    /// dominator-tree entry, whose dominance claims are invalid).
    pub(super) fn clear(&mut self) {
        self.byte_map.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::testing::TestContext;
    use qcode::value::QCodeMut;
    use qcode::value::{BasicBlock, FunctionBody, ModuleView, TempSpace};
    use qcode_macro::qcode;

    /// GVN's affine numbering is a read-only snapshot built before the
    /// dominator walk. If an earlier block forwards and deletes one of the
    /// snapshot's affine base instructions, loaded-pointer peeling in a later
    /// block must ignore that stale decomposition rather than indexing the dead
    /// instruction.
    #[test]
    fn loaded_ptr_peeling_ignores_deleted_affine_base() {
        let mut tc = TestContext::new();
        qcode!(
            tc.ctx,
            "
                varnode i64 SLOT;

                fn f:
                    <entry>
                        %old = load(SLOT:8, &SLOT);
                        %ptr = %old + i64 0x4;
                        return %ptr;
            "
        );

        let numbering = crate::gvn::affine::precompute_forms(ModuleView::new(&tc.ctx), f);

        // Model an earlier GVN block forwarding `%old` to another value and
        // deleting the redundant instruction. `%ptr` itself remains live, but
        // the precomputed form still decomposes it as `%old + 4`.
        let replacement = tc.ctx.get_const(0x1000, 8).id();
        tc.ctx.replace_instruction(old, replacement);
        assert!(!tc.ctx.contains_instruction(old));

        let space: LocalMemorySpaceId = tc.ctx.shared.default_space.into();
        assert_eq!(
            numbering.base_offset(ModuleView::new(&tc.ctx), ValueId::Instruction(ptr)),
            None,
            "numbering must reject a decomposition whose base is no longer live"
        );
        let resolved = MemForward::default().resolve_loaded_ptr(
            ModuleView::new(&tc.ctx),
            ValueId::Instruction(ptr),
            space,
            None,
            &numbering,
            true,
        );

        assert_eq!(
            resolved,
            (Base::Symbolic(space, ValueId::Instruction(ptr)), 0),
            "a stale affine base must fall back to the live pointer itself"
        );
    }

    /// A store of `src` (width = location width) to varnode `vn`.
    fn store_to(tc: &TestContext, func: FunctionId, vn: VarnodeId, src: ValueId) -> Store {
        let v = Varnode::from_id(&tc.ctx, vn);
        Store {
            space: v.space().id.into(),
            ptr: ValueId::Varnode(vn).localize(func),
            src: src.localize(func),
            size: v.size(),
        }
    }

    fn load_of(tc: &TestContext, func: FunctionId, vn: VarnodeId) -> Load {
        let v = Varnode::from_id(&tc.ctx, vn);
        Load {
            space: v.space().id.into(),
            ptr: ValueId::Varnode(vn).localize(func),
            size: v.size(),
        }
    }

    /// Borrow `fid`'s body in place and run `f` against its
    /// `(&mut FunctionBody, ContextView)` — the only surface
    /// [`MemForward::record_store`]/[`MemForward::try_load`] speak now that the
    /// `&mut Context` module twins are gone. The `MemForward` state under test is
    /// owned by the caller (captured by `f`), so it outlives the borrow and can be
    /// inspected afterwards.
    fn with_body<R>(
        tc: &mut TestContext,
        fid: qcode::value::FunctionId,
        f: impl FnOnce(&mut FunctionBody<'static>, ContextView<'_, 'static>) -> R,
    ) -> R {
        use crate::pipeline::ContextSplit;
        let env = crate::test_util::dummy_env();
        let out = {
            let (bodies, view) = tc.ctx.split(&env);
            f(&mut bodies[fid], view)
        };
        out
    }

    /// An [`AliasResult`] with a concrete interval for each listed varnode.
    ///
    /// `AliasResult::simple` only records an interval for a pointer that is
    /// actually used in a load/store, so these unit tests — which poke the
    /// byte map directly — seed the intervals by hand. Same equivalence class
    /// per address so overlapping sub-registers may-alias as in production.
    fn manual_aliases(tc: &TestContext, vns: &[VarnodeId]) -> AliasResult {
        let mut value_to_interval = HashMap::default();
        let mut value_to_root = HashMap::default();
        for &vn in vns {
            let v = Varnode::from_id(&tc.ctx, vn);
            let start = v.address() as u64;
            let end = start + v.size() as u64;
            value_to_interval.insert(ValueId::Varnode(vn), (v.space().id.into(), start, end));
            // One class per starting address: overlapping sub-registers alias.
            value_to_root.insert(
                ValueId::Varnode(vn),
                crate::alias::NodeId::Id(start as usize),
            );
        }
        AliasResult {
            value_to_root,
            value_to_interval,
            frame: None,
        }
    }

    /// Segment grouping: contiguous bytes of one source merge; a discontinuity
    /// splits them.
    #[test]
    fn segments_group_contiguous_and_split_on_gap() {
        let tc = TestContext::new();
        let aliases = manual_aliases(&tc, &[tc.r0_lo32, tc.r1]);
        let (space, start, _) = aliases
            .interval(ValueId::Varnode(tc.r0_lo32))
            .expect("r0_lo32 has an interval");

        let base = Base::Pinned(space);
        let start = start as i64;
        let src = ValueId::Varnode(tc.r1);
        let mut mf = MemForward::default();
        // bytes 0,1 = src[0],src[1]  (contiguous) ; byte 2 = src[3] (jump)
        mf.byte_map.insert((base, start), Cell { src, src_off: 0 });
        mf.byte_map
            .insert((base, start + 1), Cell { src, src_off: 1 });
        mf.byte_map
            .insert((base, start + 2), Cell { src, src_off: 3 });

        let segs = mf.segments(base, start, start + 3).expect("fully covered");
        assert_eq!(segs.len(), 2, "discontiguous src_off splits the run");
        assert_eq!((segs[0].load_off, segs[0].size, segs[0].src_off), (0, 2, 0));
        assert_eq!((segs[1].load_off, segs[1].size, segs[1].src_off), (2, 1, 3));

        // A hole makes the load unforwardable.
        mf.byte_map.remove(&(base, start + 1));
        assert!(mf.segments(base, start, start + 3).is_none());
    }

    /// A later store overwrites only the bytes it covers.
    #[test]
    fn record_store_overwrites_only_covered_bytes() {
        let mut tc = TestContext::new();
        let wide = tc.ctx.get_const(0x1122_3344, 4).id();
        let byte = tc.ctx.get_const(0xAA, 1).id();
        let aliases = manual_aliases(&tc, &[tc.r0_lo32, tc.r0_byte0]);
        let (space, start, _) = aliases.interval(ValueId::Varnode(tc.r0_lo32)).unwrap();
        let fid = tc.ctx.anon_function();

        let base = Base::Pinned(space);
        let start = start as i64;
        let wide_store = store_to(&tc, fid, tc.r0_lo32, wide);
        let byte_store = store_to(&tc, fid, tc.r0_byte0, byte);
        let nb = Numbering::default();
        let mut mf = MemForward::default();
        with_body(&mut tc, fid, |body, cx| {
            mf.record_store(body, cx, fid, &wide_store, Some(&aliases), &nb);
            mf.record_store(body, cx, fid, &byte_store, Some(&aliases), &nb);
        });

        assert_eq!(mf.byte_map[&(base, start)].src, byte, "byte 0 overwritten");
        assert_eq!(
            mf.byte_map[&(base, start + 1)].src,
            wide,
            "byte 1 still from the wide store"
        );
    }

    /// Clobber pruning drops exactly the cells inside a clobbered register.
    #[test]
    fn clobber_pruning_is_interval_precise() {
        let tc = TestContext::new();
        let aliases = manual_aliases(&tc, &[tc.r0_lo32, tc.r1]);
        let (space, r0_start, _) = aliases.interval(ValueId::Varnode(tc.r0_lo32)).unwrap();
        let (_, r1_start, _) = aliases.interval(ValueId::Varnode(tc.r1)).unwrap();

        let base = Base::Pinned(space);
        let r0_start = r0_start as i64;
        let r1_start = r1_start as i64;
        let src = ValueId::Varnode(tc.r2);
        let mut mf = MemForward::default();
        mf.byte_map
            .insert((base, r0_start), Cell { src, src_off: 0 });
        mf.byte_map
            .insert((base, r1_start), Cell { src, src_off: 0 });

        // Simulate a call clobbering only r0 by retaining via the same predicate.
        let regs = [tc.r0_lo32];
        let is_reg = |sp: LocalMemorySpaceId| {
            sp.shared()
                .is_some_and(|sp| matches!(Space::from_id(&tc.ctx, sp).ty, SpaceType::Register))
        };
        mf.byte_map.retain(|&(b, off), _| {
            let sp = b.space();
            if !is_reg(sp) {
                return true;
            }
            !regs.iter().any(|&r| {
                let vn = Varnode::from_id(&tc.ctx, r);
                vn.space().id == sp && vn.address() <= off && off < vn.address() + vn.size() as i64
            })
        });

        assert!(!mf.byte_map.contains_key(&(base, r0_start)), "r0 dropped");
        assert!(mf.byte_map.contains_key(&(base, r1_start)), "r1 kept");
    }

    /// PROTOTYPE (realigned-frame): a spill into a realigned own-frame slot
    /// survives an intervening call whose callee receives no pointer into the
    /// frame, while a caller-frame (`@SP`-rooted) slot does not. This is the wall
    /// that keeps `%tmp5633 = load(reload-of-spilled-&buffer)` opaque in the
    /// `410a60` function — the reload sits behind four calls.
    #[test]
    fn realigned_own_frame_slot_survives_call() {
        use crate::gvn::affine::precompute_forms;
        use qcode::value::{BasicBlock, FunctionBody};

        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let fid = FunctionBody::make(&mut tc.ctx, "f".into()).unwrap().id;
        let callee = FunctionBody::make(&mut tc.ctx, "callee".into()).unwrap().id;
        let root = { tc.ctx.get_or_make_block(0x1000, fid) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
        }
        let pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx.block_param_mut(pid).origin = Some(ValueId::Varnode(sp_reg).localize(pid.func));
        let sp = ValueId::BlockParam(pid);
        let ram = tc.ctx.shared.default_space;

        // `slot = ((@SP - 0x10) & -8) - 0x78`; spill an 8-byte value into it, then
        // end the block with a direct call carrying no arguments (nothing escapes).
        let (aligned, slot_store, val) = {
            let mut b = (&mut tc.ctx).builder(root);
            let c10 = b.shr().get_const(0x10, 8);
            let neg8 = b.shr().get_const((-8i64) as u64, 8);
            let c78 = b.shr().get_const(0x78, 8);
            let val = b.shr().get_const(0xdead_beef, 8);
            let s = b.push_sub(sp, c10).id();
            let aligned = b.push_bit_and(s, neg8).id();
            let slot = b.push_sub(aligned, c78).id();
            let store = Store {
                space: ram.into(),
                ptr: slot.localize(root.func),
                src: val.localize(root.func),
                size: 8,
            };
            b.push_call(callee);
            (aligned, store, val)
        };

        let aliases = AliasResult::simple_for_function(&tc.ctx, fid).with_frame_freshness(
            qcode::value::ModuleView::new(&tc.ctx),
            fid,
            Some(sp_reg),
        );
        let nb = precompute_forms(qcode::value::ModuleView::new(&tc.ctx), fid);

        let mut mf = MemForward::default();
        with_body(&mut tc, fid, |body, cx| {
            mf.record_store(body, cx, root.func, &slot_store, Some(&aliases), &nb);
        });
        // A plain caller-frame `@SP - 4` cell, for contrast: its base is the `@SP`
        // param (classified CallerFrame), so it is not own-frame-private.
        let caller_slot = Base::Symbolic(ram.into(), sp);
        mf.byte_map.insert(
            (caller_slot, -4),
            Cell {
                src: val,
                src_off: 0,
            },
        );

        let realigned = Base::Symbolic(ram.into(), aligned);
        assert!(
            mf.byte_map.contains_key(&(realigned, -0x78)),
            "spill recorded at the realigned own-frame base"
        );

        mf.prune_clobbered_by_call(qcode::value::ModuleView::new(&tc.ctx), root, Some(&aliases));

        assert!(
            mf.byte_map.contains_key(&(realigned, -0x78)),
            "realigned own-frame spill survives the call (callee got no frame pointer)"
        );
        assert!(
            !mf.byte_map.contains_key(&(caller_slot, -4)),
            "a caller-frame @SP slot is still dropped across the call"
        );
    }

    /// A store whose src is narrower than the written location zero-extends: the
    /// low `src.size` bytes come from the value and the upper bytes are zero, so
    /// a full-width read is still fully covered.
    #[test]
    fn narrow_src_in_wide_location_zero_extends() {
        let mut tc = TestContext::new();
        let narrow = tc.ctx.get_const(0x1234, 2).id(); // 2-byte src
        let aliases = manual_aliases(&tc, &[tc.r0_lo32]);
        let (space, start, _) = aliases.interval(ValueId::Varnode(tc.r0_lo32)).unwrap();
        let fid = tc.ctx.anon_function();
        // Hand-built 4-byte store of a 2-byte value (src width < store size).
        let store = Store {
            space: space.into(),
            ptr: ValueId::Varnode(tc.r0_lo32).localize(fid),
            src: narrow.localize(fid),
            size: 4,
        };
        let base = Base::Pinned(space);
        let start = start as i64;
        let mut mf = MemForward::default();
        with_body(&mut tc, fid, |body, cx| {
            mf.record_store(body, cx, fid, &store, Some(&aliases), &Numbering::default());
        });

        assert_eq!(mf.byte_map.len(), 4, "all four written bytes are defined");
        assert_eq!(mf.byte_map[&(base, start)].src, narrow);
        assert_eq!(mf.byte_map[&(base, start + 1)].src, narrow);
        // Upper bytes resolve to a zero constant.
        let upper = mf.byte_map[&(base, start + 2)].src;
        assert_eq!(
            literal_of(&tc.ctx, upper),
            Some(0),
            "upper bytes zero-extend"
        );
        assert_eq!(mf.byte_map[&(base, start + 3)].src, upper);
    }

    fn literal_of(ctx: &Context, v: ValueId) -> Option<u64> {
        match v {
            ValueId::Literal(lid) => Some(ctx.shared.values.literals[lid].value),
            _ => None,
        }
    }

    /// Exact same-width forward returns the stored value with no new instructions.
    #[test]
    fn exact_forward_returns_src() {
        let mut tc = TestContext::new();
        let block_id = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x1000, __f)
        };
        let src = tc.ctx.get_const(0x42, 4).id();
        let aliases = manual_aliases(&tc, &[tc.r0_lo32]);

        let store = store_to(&tc, block_id.func, tc.r0_lo32, src);
        let load = load_of(&tc, block_id.func, tc.r0_lo32);
        let nb = Numbering::default();
        // A dummy instruction id to insert before; none is created here because
        // an exact forward materializes nothing.
        let dummy = qcode::value::InstructionId::default();
        let mut mf = MemForward::default();
        let forwarded = with_body(&mut tc, block_id.func, |body, cx| {
            mf.record_store(body, cx, block_id.func, &store, Some(&aliases), &nb);
            mf.try_load(body, cx, block_id, dummy, &load, Some(&aliases), &nb)
        })
        .expect("exact forward");
        assert_eq!(forwarded, src);
    }

    /// A call may write through any pointer (no memory-write summary), so it
    /// drops every symbolic RAM cell. A pinned RAM cell (a resolved stack slot)
    /// is unaffected by register clobbers and survives.
    #[test]
    fn call_drops_symbolic_ram_cells_keeps_pinned() {
        let mut tc = TestContext::new();
        let fun_id = FunctionBody::make(&mut tc.ctx, "f".into()).unwrap().id;
        let block = { tc.ctx.get_or_make_block(0x1000, fun_id) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fun_id);
            f.set_root(block).unwrap();
            f.add_block(block);
        }
        // The block ends in a (direct) call.
        {
            let mut b = (&mut tc.ctx).builder(block);
            b.push_call(fun_id);
        }

        let ram = tc.ctx.shared.default_space;
        let symbolic = Base::Symbolic(ram.into(), ValueId::Varnode(tc.r1));
        let pinned = Base::Pinned(ram.into());
        let src = ValueId::Varnode(tc.r2);

        let mut mf = MemForward::default();
        mf.byte_map.insert((symbolic, 0), Cell { src, src_off: 0 });
        mf.byte_map.insert((pinned, 0x40), Cell { src, src_off: 0 });

        mf.prune_clobbered_by_call(qcode::value::ModuleView::new(&tc.ctx), block, None);

        assert!(
            !mf.byte_map.contains_key(&(symbolic, 0)),
            "a call drops symbolic RAM cells"
        );
        assert!(
            mf.byte_map.contains_key(&(pinned, 0x40)),
            "a pinned RAM cell survives a call"
        );
    }

    /// Build `f` ending in `callee(arg)`, seed a pinned RAM cell at `0x40`, run
    /// the call prune, and report whether the cell survived. `readonly` sets the
    /// callee's param-0 `readonly` bit; `pin_arg` gives the argument a concrete
    /// RAM interval covering the cell (else the argument is symbolic — no
    /// resolvable reach).
    fn pinned_cell_survives_call(readonly: bool, pin_arg: bool) -> bool {
        use qcode::value::ParamAttrs;
        use qcode::value::insn::Call;

        let mut tc = TestContext::new();
        let ram = tc.ctx.shared.default_space;
        let callee = FunctionBody::make(&mut tc.ctx, "callee".into()).unwrap().id;
        // A resolved callee with an empty clobber set, so registers are irrelevant.
        FunctionBody::from_id_mut(&mut tc.ctx, callee).set_clobbered_regs(vec![]);
        if readonly {
            FunctionBody::from_id_mut(&mut tc.ctx, callee).set_param_attrs(vec![ParamAttrs {
                readonly: true,
                nocapture: false,
            }]);
        }

        let fun_id = FunctionBody::make(&mut tc.ctx, "f".into()).unwrap().id;
        let block = { tc.ctx.get_or_make_block(0x1000, fun_id) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fun_id);
            f.set_root(block).unwrap();
            f.add_block(block);
        }
        let arg = ValueId::Varnode(tc.r1);
        {
            let mut b = (&mut tc.ctx).builder(block);
            b.push_call(callee);
        }
        // The builder makes a call with no args; set them to `[arg]`.
        let cid = BasicBlock::from_id(&tc.ctx, block)
            .iter()
            .find(|i| matches!(i.mnemonic(), Mnemonic::Call(_)))
            .unwrap()
            .id;
        tc.ctx.replace_instruction_mnemonic(
            cid,
            Mnemonic::Call(Call {
                target: qcode::value::insn::Callee::Real(callee),
                args: vec![arg.localize(cid.func)],
                clobbers: vec![],
            }),
        );

        let mut value_to_interval = HashMap::default();
        if pin_arg {
            value_to_interval.insert(arg, (ram.into(), 0x40u64, 0x48u64));
        }
        let aliases = AliasResult {
            value_to_root: HashMap::default(),
            value_to_interval,
            frame: None,
        };

        let pinned = Base::Pinned(ram.into());
        let src = ValueId::Varnode(tc.r2);
        let mut mf = MemForward::default();
        mf.byte_map.insert((pinned, 0x40), Cell { src, src_off: 0 });
        mf.prune_clobbered_by_call(
            qcode::value::ModuleView::new(&tc.ctx),
            block,
            Some(&aliases),
        );
        mf.byte_map.contains_key(&(pinned, 0x40))
    }

    /// A call argument flowing into a `readonly` callee param cannot be written
    /// through, so the pinned RAM cell its interval covers survives the call —
    /// the store→load forwarding win. The same argument into a non-readonly param
    /// clobbers the cell.
    #[test]
    fn readonly_call_arg_keeps_pinned_ram_cell() {
        assert!(
            !pinned_cell_survives_call(false, true),
            "a writable pointer arg clobbers the pinned cell its interval covers"
        );
        assert!(
            pinned_cell_survives_call(true, true),
            "a readonly pointer arg leaves the pinned cell it covers intact"
        );
    }

    /// A `readonly` argument with no resolvable interval no longer triggers the
    /// symbolic-escape path that drops *every* pinned RAM cell — the highest-value
    /// exclusion. A symbolic writable argument still nukes them.
    #[test]
    fn symbolic_readonly_call_arg_does_not_nuke_pinned_cells() {
        assert!(
            !pinned_cell_survives_call(false, false),
            "a symbolic writable arg drops every pinned cell"
        );
        assert!(
            pinned_cell_survives_call(true, false),
            "a symbolic readonly arg no longer nukes pinned cells"
        );
    }

    /// A direct call to a callee with a witnessed write-set keeps every cell
    /// whose space the callee never writes — even a symbolic RAM cell that the
    /// no-summary path would drop — while still dropping cells in a space the
    /// callee does write. This is the per-callee memory summary (Path C): a
    /// `pure_reg` helper that writes only its private scratch space cannot clobber
    /// the caller's real-`ram` spill.
    #[test]
    fn call_keeps_cells_in_spaces_callee_never_writes() {
        let mut tc = TestContext::new();
        let caller = FunctionBody::make(&mut tc.ctx, "caller".into()).unwrap().id;
        let callee = FunctionBody::make(&mut tc.ctx, "callee".into()).unwrap().id;
        let block = { tc.ctx.get_or_make_block(0x1000, caller) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, caller);
            f.set_root(block).unwrap();
            f.add_block(block);
        }
        // The callee's witnessed write-set is exactly its own scratch space — it
        // never writes real `ram`.
        let ram = tc.ctx.shared.default_space;
        tc.ctx.bodies[callee].push_temp_space(TempSpace::new(Some("callee-scratch"), 1, 8));
        let caller_scratch =
            tc.ctx.bodies[caller].push_temp_space(TempSpace::new(Some("caller-scratch"), 1, 8));
        let caller_scratch = LocalMemorySpaceId::Temp(caller_scratch.local);
        // Body-local scratch is private to the callee and therefore absent from
        // the module-visible write summary.
        FunctionBody::from_id_mut(&mut tc.ctx, callee).set_written_spaces(Some(vec![]));

        // The caller block ends in a direct call to `callee`, passing a frame
        // pointer (so a pointer escapes — the no-summary path would drop the cell).
        {
            let mut b = (&mut tc.ctx).builder(block);
            b.push_call_with_args(callee, vec![ValueId::Varnode(tc.r1)]);
        }

        let ram_cell = Base::Symbolic(ram.into(), ValueId::Varnode(tc.r1));
        let caller_scratch_cell = Base::Symbolic(caller_scratch, ValueId::Varnode(tc.r2));
        let src = ValueId::Varnode(tc.r2);

        let mut mf = MemForward::default();
        mf.byte_map.insert((ram_cell, 0), Cell { src, src_off: 0 });
        mf.byte_map
            .insert((caller_scratch_cell, 0), Cell { src, src_off: 0 });

        mf.prune_clobbered_by_call(qcode::value::ModuleView::new(&tc.ctx), block, None);

        assert!(
            mf.byte_map.contains_key(&(ram_cell, 0)),
            "a RAM cell survives a call to a callee that writes only scratch"
        );
        assert!(
            mf.byte_map.contains_key(&(caller_scratch_cell, 0)),
            "a callee cannot clobber its caller's body-local scratch"
        );
    }
}
