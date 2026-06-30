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
//! interval. Pointers the [`AliasResult`] can pin to a concrete interval
//! (registers, constant-folded stack slots) are tracked byte-precisely in
//! [`byte_map`](MemForward::byte_map); everything else (heap pointers with no
//! precise location) falls back to the exact `Load`-mnemonic table in
//! [`opaque`](MemForward::opaque), preserving the old may-alias behavior.
//!
//! The whole structure is cloned and threaded down the dominator tree by
//! the dominator walk in [`super::walk`], so forwarding works across blocks; the pruning
//! methods drop entries a call clobbers or a loop back-edge invalidates.

use std::collections::HashMap;

use jstd::graph::analysis::DominatorTree;

use crate::AliasResult;
use qcode::{
    context::Context,
    space::{Space, SpaceId, SpaceType},
    value::{
        BasicBlock, Function, Value, ValueId, ValueRef, Varnode, VarnodeId,
        block::BlockId,
        insn::{Binary, Binop, InstructionRef, IntBinop, Load, Mnemonic, Range, Store, Zext},
    },
};

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

#[derive(Clone, Default)]
pub(super) struct MemForward {
    /// Per-byte forwarding for pointers with a concrete alias interval.
    byte_map: HashMap<(SpaceId, u64), Cell>,
    /// Exact `Load`-mnemonic → leader, for pointers with no precise interval.
    opaque: HashMap<Mnemonic, ValueId>,
}

impl MemForward {
    /// Update state for `store`, invalidating any forwarded value it overwrites.
    pub(super) fn record_store(
        &mut self,
        ctx: &mut Context,
        store: &Store,
        aliases: Option<&AliasResult>,
    ) {
        // Drop opaque load leaders this store may overwrite. With no oracle every
        // load is conservatively invalidated (mirrors the old single-block pass).
        self.opaque.retain(|k, _| {
            let Mnemonic::Load(load) = k else { return true };
            aliases.is_some_and(|a| !a.may_alias(ctx, store.ptr, load.ptr))
        });

        match aliases.and_then(|a| a.interval(store.ptr)) {
            // Pinned location: model the `store.size` bytes it actually writes,
            // starting at the pinned base. (`store.size` can be narrower than the
            // resolved varnode interval — a sub-register write.)
            Some((space, start, _end)) => {
                let store_end = start + store.size as u64;
                // The store always overwrites those bytes, so clear them first.
                for addr in start..store_end {
                    self.byte_map.remove(&(space, addr));
                }
                // The low `src.size` bytes come from the value. When the src is
                // narrower than the location (e.g. `MOV ESI, imm32`, whose
                // store width is the full 8-byte RSI), the store zero-extends:
                // the upper bytes are zero, so model them with a zero constant
                // rather than leaving them unmapped.
                let covered = ValueRef::new(store.src, ctx).size().min(store.size);
                for (i, addr) in (start..start + covered as u64).enumerate() {
                    self.byte_map.insert(
                        (space, addr),
                        Cell {
                            src: store.src,
                            src_off: i,
                        },
                    );
                }
                if covered < store.size {
                    let zero = ctx.get_const(0, store.size - covered).id();
                    for (i, addr) in (start + covered as u64..store_end).enumerate() {
                        self.byte_map.insert(
                            (space, addr),
                            Cell {
                                src: zero,
                                src_off: i,
                            },
                        );
                    }
                }
            }
            // No precise location. Conservatively forget every byte cell the
            // store could touch (its whole space), then record the exact load.
            None => {
                self.byte_map.retain(|&(space, _), _| space != store.space);
                if ValueRef::new(store.src, ctx).size() == store.size {
                    self.opaque
                        .insert(Mnemonic::Load(store.get_matching_load()), store.src);
                }
            }
        }
    }

    /// The value `load` forwards to, materializing any rebuild instructions
    /// before `insn_id` in `block_id`, or `None` if it is not fully covered.
    pub(super) fn try_load(
        &mut self,
        ctx: &mut Context,
        block_id: BlockId,
        insn_id: qcode::value::InstructionId,
        load: &Load,
        aliases: Option<&AliasResult>,
    ) -> Option<ValueId> {
        let Some((space, start, _end)) = aliases.and_then(|a| a.interval(load.ptr)) else {
            // Unpinned location: only exact opaque matches forward.
            return self.opaque.get(&Mnemonic::Load(load.clone())).copied();
        };

        // Cover the `load.size` bytes the load actually reads from the pinned base.
        let end = start + load.size as u64;
        let segments = self.segments(space, start, end)?;
        let load_size = load.size;

        let value = if segments.len() == 1
            && segments[0].load_off == 0
            && segments[0].size == load_size
            && segments[0].src_off == 0
            && ValueRef::new(segments[0].src, ctx).size() == load_size
        {
            // Exact whole-location forward: reuse the stored value directly.
            segments[0].src
        } else {
            self.rebuild(ctx, block_id, insn_id, &segments, load_size)
        };
        Some(value)
    }

    /// Record that, after this load executes, `value` is held at `load`'s
    /// location. Called for every load (forwarded or not) so a later identical
    /// load can reuse the result.
    pub(super) fn define_load(
        &mut self,
        load: &Load,
        value: ValueId,
        aliases: Option<&AliasResult>,
    ) {
        match aliases.and_then(|a| a.interval(load.ptr)) {
            Some((space, start, _end)) => {
                for (i, addr) in (start..start + load.size as u64).enumerate() {
                    self.byte_map.insert(
                        (space, addr),
                        Cell {
                            src: value,
                            src_off: i,
                        },
                    );
                }
            }
            None => {
                self.opaque.insert(Mnemonic::Load(load.clone()), value);
            }
        }
    }

    /// Group bytes `start..end` of `space` into maximal single-source segments,
    /// or `None` if any byte is unmapped (coverage gap — not forwardable).
    fn segments(&self, space: SpaceId, start: u64, end: u64) -> Option<Vec<Segment>> {
        let mut segments: Vec<Segment> = Vec::new();
        for addr in start..end {
            let cell = self.byte_map.get(&(space, addr)).copied()?;
            let load_off = (addr - start) as usize;
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

    /// Materialize `segments` into a `load_size`-byte value: each segment is
    /// extracted (`Range`), widened (`Zext`), shifted into place (`<<`), and the
    /// pieces are OR-ed together. Reuses `Range`/`Zext`/`<<`/`|`; constant pieces
    /// fold away in [`super::fold`].
    fn rebuild(
        &self,
        ctx: &mut Context,
        block_id: BlockId,
        insn_id: qcode::value::InstructionId,
        segments: &[Segment],
        load_size: usize,
    ) -> ValueId {
        let mut acc: Option<ValueId> = None;
        for seg in segments {
            let piece = self.build_piece(ctx, block_id, insn_id, seg, load_size);
            acc = Some(match acc {
                None => piece,
                Some(lhs) => {
                    let or = InstructionRef::from_mnemonic(
                        ctx,
                        Mnemonic::Binop(Binary {
                            op: Binop::Int(IntBinop::Or),
                            lhs,
                            rhs: piece,
                        }),
                        load_size,
                    )
                    .id;
                    BasicBlock::from_id_mut(ctx, block_id).insert_insn_before(insn_id, or);
                    or.into()
                }
            });
        }
        acc.expect("a fully-covered load has at least one segment")
    }

    /// Build one segment's contribution: `Range` of the source (skipped when the
    /// segment already equals the whole source), `Zext` to the load width
    /// (skipped when already that wide), then `<< load_off*8` (skipped at offset
    /// 0).
    fn build_piece(
        &self,
        ctx: &mut Context,
        block_id: BlockId,
        insn_id: qcode::value::InstructionId,
        seg: &Segment,
        load_size: usize,
    ) -> ValueId {
        let src_size = ValueRef::new(seg.src, ctx).size();
        let extracted = if seg.src_off == 0 && src_size == seg.size {
            seg.src
        } else {
            let r = InstructionRef::from_mnemonic(
                ctx,
                Mnemonic::Range(Range {
                    src: seg.src,
                    start: seg.src_off,
                    size: seg.size,
                }),
                seg.size,
            )
            .id;
            BasicBlock::from_id_mut(ctx, block_id).insert_insn_before(insn_id, r);
            r.into()
        };

        let widened = if seg.size == load_size {
            extracted
        } else {
            let z = InstructionRef::from_mnemonic(
                ctx,
                Mnemonic::Zext(Zext {
                    src: extracted,
                    size: load_size,
                }),
                load_size,
            )
            .id;
            BasicBlock::from_id_mut(ctx, block_id).insert_insn_before(insn_id, z);
            z.into()
        };

        if seg.load_off == 0 {
            return widened;
        }
        let shamt = ctx.get_const((seg.load_off * 8) as u64, load_size).id();
        let s = InstructionRef::from_mnemonic(
            ctx,
            Mnemonic::Binop(Binary {
                op: Binop::Int(IntBinop::ShiftLeft),
                lhs: widened,
                rhs: shamt,
            }),
            load_size,
        )
        .id;
        BasicBlock::from_id_mut(ctx, block_id).insert_insn_before(insn_id, s);
        s.into()
    }

    /// Drop everything the call terminating `block_id` (if any) may clobber, so
    /// no forwarded register value survives across it. Non-call blocks are no-ops.
    pub(super) fn prune_clobbered_by_call(
        &mut self,
        ctx: &Context,
        block_id: BlockId,
        aliases: Option<&AliasResult>,
    ) {
        let term = BasicBlock::from_id(ctx, block_id)
            .iter()
            .last()
            .map(|i| i.mnemonic().clone());
        let clobbers = match term {
            Some(Mnemonic::CallInd(_)) => CallClobbers::AllRegisters,
            Some(Mnemonic::Call(call)) => {
                match Function::from_id(ctx, call.target).clobbered_regs() {
                    Some(regs) => CallClobbers::Regs(regs.to_vec()),
                    None => CallClobbers::AllRegisters,
                }
            }
            _ => return,
        };

        let is_reg = |space| matches!(Space::from_id(ctx, space).ty, SpaceType::Register);

        // Byte cells: drop register bytes the call clobbers.
        self.byte_map.retain(|&(space, addr), _| {
            if !is_reg(space) {
                return true;
            }
            match &clobbers {
                CallClobbers::AllRegisters => false,
                CallClobbers::Regs(regs) => !regs.iter().any(|&r| {
                    let vn = Varnode::from_id(ctx, r);
                    vn.space().id == space
                        && (vn.address() as u64) <= addr
                        && addr < vn.address() as u64 + vn.size() as u64
                }),
            }
        });

        // Opaque register load leaders: same rule, via may-alias.
        self.opaque.retain(|m, _| {
            let Mnemonic::Load(load) = m else { return true };
            let load_space = ValueRef::new(load.ptr, ctx).space().map(|s| s.id);
            if !load_space.is_some_and(is_reg) {
                return true;
            }
            match &clobbers {
                CallClobbers::AllRegisters => false,
                CallClobbers::Regs(regs) => !regs.iter().any(|&r| match aliases {
                    Some(a) => a.may_alias(ctx, load.ptr, ValueId::Varnode(r)),
                    None => true,
                }),
            }
        });
    }

    /// At a loop header, drop forwarded values a store inside the loop may
    /// overwrite on a later iteration. Non-headers are left untouched.
    pub(super) fn prune_loop_carried(
        &mut self,
        ctx: &Context,
        block_id: BlockId,
        tree: &DominatorTree<BlockId>,
        aliases: Option<&AliasResult>,
    ) {
        // `dominates` is reflexive, so a self-loop also counts.
        let is_loop_header = BasicBlock::from_id(ctx, block_id)
            .predecessors()
            .any(|(_, pred)| tree.dominates(block_id, pred));
        if !is_loop_header || (self.byte_map.is_empty() && self.opaque.is_empty()) {
            return;
        }

        // Every block the header dominates is a candidate loop body: a store
        // there can clobber the inherited value on a later iteration (the header
        // itself is included — a store after the load but before the back edge
        // counts). Walk the dominator-tree subtree rooted at the header rather
        // than scanning — and dominance-testing — every block in the whole
        // program, which is O(program) per loop header and made gvn scale with
        // total lifted code instead of the current function.
        let mut body = vec![block_id];
        let mut frontier = vec![block_id];
        while let Some(b) = frontier.pop() {
            for &child in tree.children_of(b) {
                body.push(child);
                frontier.push(child);
            }
        }
        let stores: Vec<Store> = body
            .iter()
            .flat_map(|&b| {
                BasicBlock::from_id(ctx, b)
                    .iter()
                    .filter_map(|insn| match insn.mnemonic() {
                        Mnemonic::Store(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        self.byte_map.retain(|&(space, addr), _| {
            !stores
                .iter()
                .any(|store| match aliases.and_then(|a| a.interval(store.ptr)) {
                    Some((sp, s, e)) => sp == space && s <= addr && addr < e,
                    // A store we cannot pin may overwrite any byte of its space.
                    None => store.space == space,
                })
        });
        self.opaque.retain(|m, _| {
            let Mnemonic::Load(load) = m else { return true };
            let Some(aliases) = aliases else { return false };
            !stores
                .iter()
                .any(|store| aliases.may_alias(ctx, store.ptr, load.ptr))
        });
    }

    /// Clear all forwarded state (used for blocks reachable from more than one
    /// dominator-tree entry, whose dominance claims are invalid).
    pub(super) fn clear(&mut self) {
        self.byte_map.clear();
        self.opaque.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::testing::TestContext;

    /// A store of `src` (width = location width) to varnode `vn`.
    fn store_to(tc: &TestContext, vn: VarnodeId, src: ValueId) -> Store {
        let v = Varnode::from_id(&tc.ctx, vn);
        Store {
            space: v.space().id,
            ptr: ValueId::Varnode(vn),
            src,
            size: v.size(),
        }
    }

    fn load_of(tc: &TestContext, vn: VarnodeId) -> Load {
        let v = Varnode::from_id(&tc.ctx, vn);
        Load {
            space: v.space().id,
            ptr: ValueId::Varnode(vn),
            size: v.size(),
        }
    }

    /// An [`AliasResult`] with a concrete interval for each listed varnode.
    ///
    /// `AliasResult::simple` only records an interval for a pointer that is
    /// actually used in a load/store, so these unit tests — which poke the
    /// byte map directly — seed the intervals by hand. Same equivalence class
    /// per address so overlapping sub-registers may-alias as in production.
    fn manual_aliases(tc: &TestContext, vns: &[VarnodeId]) -> AliasResult {
        let mut value_to_interval = HashMap::new();
        let mut value_to_root = HashMap::new();
        for &vn in vns {
            let v = Varnode::from_id(&tc.ctx, vn);
            let start = v.address() as u64;
            let end = start + v.size() as u64;
            value_to_interval.insert(ValueId::Varnode(vn), (v.space().id, start, end));
            // One class per starting address: overlapping sub-registers alias.
            value_to_root.insert(
                ValueId::Varnode(vn),
                crate::alias::NodeId::Id(start as usize),
            );
        }
        AliasResult {
            value_to_root,
            value_to_interval,
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

        let src = ValueId::Varnode(tc.r1);
        let mut mf = MemForward::default();
        // bytes 0,1 = src[0],src[1]  (contiguous) ; byte 2 = src[3] (jump)
        mf.byte_map.insert((space, start), Cell { src, src_off: 0 });
        mf.byte_map
            .insert((space, start + 1), Cell { src, src_off: 1 });
        mf.byte_map
            .insert((space, start + 2), Cell { src, src_off: 3 });

        let segs = mf.segments(space, start, start + 3).expect("fully covered");
        assert_eq!(segs.len(), 2, "discontiguous src_off splits the run");
        assert_eq!((segs[0].load_off, segs[0].size, segs[0].src_off), (0, 2, 0));
        assert_eq!((segs[1].load_off, segs[1].size, segs[1].src_off), (2, 1, 3));

        // A hole makes the load unforwardable.
        mf.byte_map.remove(&(space, start + 1));
        assert!(mf.segments(space, start, start + 3).is_none());
    }

    /// A later store overwrites only the bytes it covers.
    #[test]
    fn record_store_overwrites_only_covered_bytes() {
        let mut tc = TestContext::new();
        let wide = tc.ctx.get_const(0x1122_3344, 4).id();
        let byte = tc.ctx.get_const(0xAA, 1).id();
        let aliases = manual_aliases(&tc, &[tc.r0_lo32, tc.r0_byte0]);
        let (space, start, _) = aliases.interval(ValueId::Varnode(tc.r0_lo32)).unwrap();

        let wide_store = store_to(&tc, tc.r0_lo32, wide);
        let byte_store = store_to(&tc, tc.r0_byte0, byte);
        let mut mf = MemForward::default();
        mf.record_store(&mut tc.ctx, &wide_store, Some(&aliases));
        mf.record_store(&mut tc.ctx, &byte_store, Some(&aliases));

        assert_eq!(mf.byte_map[&(space, start)].src, byte, "byte 0 overwritten");
        assert_eq!(
            mf.byte_map[&(space, start + 1)].src,
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

        let src = ValueId::Varnode(tc.r2);
        let mut mf = MemForward::default();
        mf.byte_map
            .insert((space, r0_start), Cell { src, src_off: 0 });
        mf.byte_map
            .insert((space, r1_start), Cell { src, src_off: 0 });

        // Simulate a call clobbering only r0 by retaining via the same predicate.
        let regs = [tc.r0_lo32];
        let is_reg = |sp| matches!(Space::from_id(&tc.ctx, sp).ty, SpaceType::Register);
        mf.byte_map.retain(|&(sp, addr), _| {
            if !is_reg(sp) {
                return true;
            }
            !regs.iter().any(|&r| {
                let vn = Varnode::from_id(&tc.ctx, r);
                vn.space().id == sp
                    && (vn.address() as u64) <= addr
                    && addr < vn.address() as u64 + vn.size() as u64
            })
        });

        assert!(!mf.byte_map.contains_key(&(space, r0_start)), "r0 dropped");
        assert!(mf.byte_map.contains_key(&(space, r1_start)), "r1 kept");
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
        // Hand-built 4-byte store of a 2-byte value (src width < store size).
        let store = Store {
            space,
            ptr: ValueId::Varnode(tc.r0_lo32),
            src: narrow,
            size: 4,
        };
        let mut mf = MemForward::default();
        mf.record_store(&mut tc.ctx, &store, Some(&aliases));

        assert_eq!(mf.byte_map.len(), 4, "all four written bytes are defined");
        assert_eq!(mf.byte_map[&(space, start)].src, narrow);
        assert_eq!(mf.byte_map[&(space, start + 1)].src, narrow);
        // Upper bytes resolve to a zero constant.
        let upper = mf.byte_map[&(space, start + 2)].src;
        assert_eq!(
            literal_of(&tc.ctx, upper),
            Some(0),
            "upper bytes zero-extend"
        );
        assert_eq!(mf.byte_map[&(space, start + 3)].src, upper);
    }

    fn literal_of(ctx: &Context, v: ValueId) -> Option<u64> {
        match v {
            ValueId::Literal(lid) => Some(ctx.values.literals[lid].value),
            _ => None,
        }
    }

    /// Exact same-width forward returns the stored value with no new instructions.
    #[test]
    fn exact_forward_returns_src() {
        let mut tc = TestContext::new();
        let block_id = tc.ctx.get_or_make_block(0x1000);
        let src = tc.ctx.get_const(0x42, 4).id();
        let aliases = manual_aliases(&tc, &[tc.r0_lo32]);

        let store = store_to(&tc, tc.r0_lo32, src);
        let mut mf = MemForward::default();
        mf.record_store(&mut tc.ctx, &store, Some(&aliases));

        let load = load_of(&tc, tc.r0_lo32);
        // A dummy instruction id to insert before; none is created here because
        // an exact forward materializes nothing.
        let dummy = qcode::value::InstructionId::from(0usize);
        let forwarded = mf
            .try_load(&mut tc.ctx, block_id, dummy, &load, Some(&aliases))
            .expect("exact forward");
        assert_eq!(forwarded, src);
    }
}
