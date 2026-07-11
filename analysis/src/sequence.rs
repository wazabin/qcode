//! Sequence/array region analysis.
//!
//! This module owns the reusable recognition of bounded, homogeneous indexed
//! memory regions. It deliberately stops at facts: callers decide whether a
//! region becomes an `Array` snapshot, a `map`, a `scan`, or some later higher
//! level construct.

use qcode::{
    context::Context,
    value::{
        Value, ValueId,
        block::BlockId,
        insn::{InstructionId, Mnemonic},
    },
};

use crate::gvn::affine::Numbering;
use crate::value_range;

/// Largest byte span a single dynamic-index region may snapshot. A bound the
/// `value_range` analysis over-approximates past this is treated as unbounded.
pub(crate) const MAX_REGION_BYTES: u64 = 4096;

/// One memory access, normalized enough for sequence-region analysis.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MemoryAccess {
    pub id: InstructionId,
    pub block: BlockId,
    pub is_store: bool,
    pub ptr: ValueId,
    pub size: usize,
}

/// A bounded, homogeneous indexed region rooted at one base value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SequenceRegion {
    /// Byte offset of the region start from the base pointer.
    pub base_off: u64,
    /// Element/access width in bytes. All accesses folded into this region agree.
    pub elem_size: usize,
    /// Element count: `ceil(byte_span / elem_size)`.
    pub count: usize,
    /// Whether any folded access writes into the region.
    pub has_write: bool,
}

impl SequenceRegion {
    /// Total region footprint in bytes.
    pub(crate) fn byte_len(&self) -> usize {
        self.count * self.elem_size
    }

    /// Half-open byte interval relative to the base.
    #[cfg(test)]
    pub(crate) fn byte_range(&self) -> (u64, u64) {
        (self.base_off, self.base_off + self.byte_len() as u64)
    }
}

/// A region plus the concrete instruction ids that belong to it.
#[derive(Debug, Clone)]
pub(crate) struct RegionUse {
    pub region: SequenceRegion,
    pub accesses: Vec<InstructionId>,
}

/// Dynamic-access classification for one base value.
#[derive(Debug, Clone, Default)]
pub(crate) struct RegionSet {
    pub regions: Vec<RegionUse>,
    /// Accesses that looked dynamic but could not become a bounded homogeneous
    /// region. Callers normally leave these unmodelled so their safety gate rejects
    /// the rewrite.
    pub rejected_accesses: Vec<InstructionId>,
}

/// How an address relates to a candidate sequence base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AddressRelation {
    /// A constant byte offset `base + off`.
    Const(i64),
    /// A bounded dynamic-index access whose element-start byte offset ranges over
    /// `[lo, hi]` inclusive.
    Dynamic { lo: i64, hi: i64 },
    /// Not an affine function of this base that sequence analysis can model.
    Unrelated,
}

#[derive(Debug, Clone)]
struct RegionAcc {
    min: u64,
    max_end: u64,
    elem: usize,
    has_write: bool,
    accesses: Vec<InstructionId>,
    ok: bool,
}

impl RegionAcc {
    fn new(lo: u64, hi: u64, size: usize, is_store: bool, id: InstructionId) -> Option<Self> {
        let end = hi.checked_add(size as u64)?;
        Some(Self {
            min: lo,
            max_end: end,
            elem: size,
            has_write: is_store,
            accesses: vec![id],
            ok: size != 0,
        })
    }

    fn overlaps(&self, lo: u64, end: u64) -> bool {
        lo < self.max_end && self.min < end
    }

    fn merge(&mut self, lo: u64, hi: u64, size: usize, is_store: bool, id: InstructionId) {
        if self.elem != size || size == 0 {
            self.ok = false;
        }
        self.min = self.min.min(lo);
        self.max_end = self.max_end.max(hi.saturating_add(size as u64));
        self.has_write |= is_store;
        self.accesses.push(id);
    }

    fn absorb(&mut self, other: RegionAcc) {
        if self.elem != other.elem || !other.ok {
            self.ok = false;
        }
        self.min = self.min.min(other.min);
        self.max_end = self.max_end.max(other.max_end);
        self.has_write |= other.has_write;
        self.accesses.extend(other.accesses);
    }

    fn build(self) -> Result<RegionUse, Vec<InstructionId>> {
        if !self.ok || self.elem == 0 {
            return Err(self.accesses);
        }
        let Some(span) = self.max_end.checked_sub(self.min) else {
            return Err(self.accesses);
        };
        if span == 0 || span > MAX_REGION_BYTES {
            return Err(self.accesses);
        }
        let count = span.div_ceil(self.elem as u64) as usize;
        Ok(RegionUse {
            region: SequenceRegion {
                base_off: self.min,
                elem_size: self.elem,
                count,
                has_write: self.has_write,
            },
            accesses: self.accesses,
        })
    }
}

/// `value` interpreted as signed at `width` bytes.
pub(crate) fn signed_at(value: u64, width: usize) -> i64 {
    let bits = width * 8;
    if bits == 0 || bits >= 64 {
        value as i64
    } else {
        ((value << (64 - bits)) as i64) >> (64 - bits)
    }
}

/// Decompose `addr` as `root + c` for a root base selected by `is_root`.
pub(crate) fn affine_base_const(
    numbering: &Numbering,
    addr: ValueId,
    is_root: &impl Fn(ValueId) -> bool,
) -> Option<(ValueId, i64)> {
    if is_root(addr) {
        return Some((addr, 0));
    }
    let (width, constant, terms) = numbering.affine_terms(addr)?;
    let [(t, k)] = terms[..] else {
        return None;
    };
    (k == 1 && is_root(t)).then(|| (t, signed_at(constant, width)))
}

/// Decompose `addr` as the strided lane address `base + idx*elem_size + c`, where
/// `is_base` identifies the region-base term (typically "rooted at a function
/// entry pointer").
///
/// The base carries coefficient 1; the index carries the element stride and is a
/// loop-varying block param. When `elem_size == 1` the two roles collide on
/// coefficient 1, so classification is by *role* (`is_base`) rather than by which
/// term the affine form happens to list first — otherwise a byte-stride fill whose
/// induction sorts ahead of its base pointer would bind them backwards.
pub(crate) fn affine_strided_lane(
    numbering: &Numbering,
    addr: ValueId,
    elem_size: usize,
    is_base: impl Fn(ValueId) -> bool,
) -> Option<(ValueId, ValueId, i64)> {
    let (width, constant, terms) = numbering.affine_terms(addr)?;
    let mut base = None;
    let mut idx = None;
    for (t, k) in terms {
        // The index is the element-strided block param that is *not* a base; the
        // base is the unit-coefficient term the caller recognizes as a region root.
        if idx.is_none()
            && k == elem_size as u64
            && matches!(t, ValueId::BlockParam(_))
            && !is_base(t)
        {
            idx = Some(t);
        } else if base.is_none() && k == 1 && is_base(t) {
            base = Some(t);
        } else {
            return None;
        }
    }
    Some((base?, idx?, signed_at(constant, width)))
}

/// Classify how `addr` relates to `base`, using affine arithmetic and
/// `value_range` to bound dynamic terms.
pub(crate) fn relate_address(
    ctx: &Context,
    numbering: &Numbering,
    base: ValueId,
    addr: ValueId,
    block: BlockId,
) -> AddressRelation {
    if addr == base {
        return AddressRelation::Const(0);
    }
    if let ValueId::Instruction(id) = addr
        && let Mnemonic::Gep(g) = ctx.get_insn(id).mnemonic()
        && g.base == base
    {
        return AddressRelation::Const(g.offset as i64);
    }
    let Some((width, constant, terms)) = numbering.affine_terms(addr) else {
        return AddressRelation::Unrelated;
    };
    let mut base_seen = false;
    let mut remaining: Vec<(ValueId, u64)> = Vec::new();
    for (t, k) in terms {
        if !base_seen && t == base && k == 1 {
            base_seen = true;
        } else {
            remaining.push((t, k));
        }
    }
    if !base_seen {
        return AddressRelation::Unrelated;
    }

    let mut lo = signed_at(constant, width) as i128;
    let mut hi = lo;
    for (term, coeff) in remaining {
        let stride = signed_at(coeff, width);
        if stride <= 0 {
            return AddressRelation::Unrelated;
        }
        let range = value_range(ctx, term, block);
        let term_size = qcode::value::ValueRef::new(term, ctx).size();
        if !range.is_bounded(term_size) {
            return AddressRelation::Unrelated;
        }
        lo += stride as i128 * range.min as i128;
        hi += stride as i128 * range.max as i128;
    }

    let (Ok(lo), Ok(hi)) = (i64::try_from(lo), i64::try_from(hi)) else {
        return AddressRelation::Unrelated;
    };
    if lo == hi {
        AddressRelation::Const(lo)
    } else {
        AddressRelation::Dynamic { lo, hi }
    }
}

/// Collect every bounded dynamic access through `base` into disjoint homogeneous
/// regions. Overlapping accesses of the same width are merged into one region;
/// non-overlapping accesses stay as multiple regions; overlapping accesses with
/// different widths reject that whole merged group.
pub(crate) fn collect_regions_for_base(
    ctx: &Context,
    numbering: &Numbering,
    base: ValueId,
    accesses: &[MemoryAccess],
) -> RegionSet {
    let mut groups: Vec<RegionAcc> = Vec::new();
    let mut rejected = Vec::new();

    for access in accesses {
        let AddressRelation::Dynamic { lo, hi } =
            relate_address(ctx, numbering, base, access.ptr, access.block)
        else {
            continue;
        };
        if lo < 0 || hi < lo {
            rejected.push(access.id);
            continue;
        }
        let lo = lo as u64;
        let hi = hi as u64;
        let Some(end) = hi.checked_add(access.size as u64) else {
            rejected.push(access.id);
            continue;
        };

        let mut overlapping = Vec::new();
        for (i, group) in groups.iter().enumerate() {
            if group.overlaps(lo, end) {
                overlapping.push(i);
            }
        }

        if overlapping.is_empty() {
            if let Some(group) = RegionAcc::new(lo, hi, access.size, access.is_store, access.id) {
                groups.push(group);
            } else {
                rejected.push(access.id);
            }
            continue;
        }

        let first = overlapping[0];
        groups[first].merge(lo, hi, access.size, access.is_store, access.id);
        for &idx in overlapping[1..].iter().rev() {
            let other = groups.swap_remove(idx);
            groups[first].absorb(other);
        }
    }

    let mut regions = Vec::new();
    for group in groups {
        match group.build() {
            Ok(region) => regions.push(region),
            Err(mut ids) => rejected.append(&mut ids),
        }
    }
    regions.sort_by_key(|r| r.region.base_off);
    RegionSet {
        regions,
        rejected_accesses: rejected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{
        builder::Builder,
        testing::TestContext,
        value::{BasicBlock, Function},
    };

    fn setup_loop(
        tc: &mut TestContext,
    ) -> (
        qcode::value::function::FunctionId,
        BlockId,
        ValueId,
        ValueId,
    ) {
        let fid = Function::make(&mut tc.ctx, "f".into()).unwrap().id;
        let entry = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x1000, __f)
        };
        let body = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x1010, __f)
        };
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
            f.add_block(body);
        }
        let base =
            ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(8).id);
        let i = ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, body).push_param(8).id);
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let zero = b.context_mut().get_const(0, 8).id();
            b.push_branch_with_args(body, vec![zero]);
        }
        (fid, body, base, i)
    }

    #[test]
    fn affine_strided_access_becomes_region() {
        let mut tc = TestContext::new();
        let (fid, body, base, i) = setup_loop(&mut tc);
        let ram = tc.ctx.shared.default_space;
        let access = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, body));
            let four = b.context_mut().get_const(4, 8).id();
            let scaled = b.push_mul(i, four).id();
            let addr0 = b.push_add(base, scaled).id();
            let addr = b.push_add(addr0, four).id();
            MemoryAccess {
                id: b.push_store(four, addr, ram).id,
                block: body,
                is_store: true,
                ptr: addr,
                size: 4,
            }
        };
        // Add a back-edge guard so value_range can bound i to [0, 19].
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, body));
            let one = b.context_mut().get_const(1, 8).id();
            let ni = b.push_add(i, one).id();
            let bound = b.context_mut().get_const(20, 8).id();
            let cond = b.push_lt(ni, bound).id();
            b.push_cbranch_with_args(cond, body, vec![ni], body, vec![ni]);
        }

        let numbering = crate::gvn::affine::precompute_forms(&tc.ctx, fid);
        let regions = collect_regions_for_base(&tc.ctx, &numbering, base, &[access]);
        assert_eq!(regions.rejected_accesses, Vec::<InstructionId>::new());
        assert_eq!(regions.regions.len(), 1);
        assert_eq!(
            regions.regions[0].region,
            SequenceRegion {
                base_off: 4,
                elem_size: 4,
                count: 20,
                has_write: true,
            }
        );
    }

    #[test]
    fn disjoint_dynamic_regions_stay_separate() {
        let mut tc = TestContext::new();
        let (fid, body, base, i) = setup_loop(&mut tc);
        let ram = tc.ctx.shared.default_space;
        let accesses = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, body));
            let one = b.context_mut().get_const(1, 8).id();
            let addr_a = b.push_add(base, i).id();
            let off = b.context_mut().get_const(32, 8).id();
            let addr_b0 = b.push_add(base, off).id();
            let addr_b = b.push_add(addr_b0, i).id();
            let a = b.push_store(one, addr_a, ram).id;
            let b_id = b.push_store(one, addr_b, ram).id;
            let ni = b.push_add(i, one).id();
            let bound = b.context_mut().get_const(8, 8).id();
            let cond = b.push_lt(ni, bound).id();
            b.push_cbranch_with_args(cond, body, vec![ni], body, vec![ni]);
            vec![
                MemoryAccess {
                    id: a,
                    block: body,
                    is_store: true,
                    ptr: addr_a,
                    size: 1,
                },
                MemoryAccess {
                    id: b_id,
                    block: body,
                    is_store: true,
                    ptr: addr_b,
                    size: 1,
                },
            ]
        };

        let numbering = crate::gvn::affine::precompute_forms(&tc.ctx, fid);
        let regions = collect_regions_for_base(&tc.ctx, &numbering, base, &accesses);
        assert_eq!(regions.regions.len(), 2);
        assert_eq!(regions.regions[0].region.byte_range(), (0, 8));
        assert_eq!(regions.regions[1].region.byte_range(), (32, 40));
    }

    #[test]
    fn overlapping_mismatched_width_rejects_region() {
        let mut tc = TestContext::new();
        let (fid, body, base, i) = setup_loop(&mut tc);
        let ram = tc.ctx.shared.default_space;
        let accesses = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, body));
            let one = b.context_mut().get_const(1, 8).id();
            let addr = b.push_add(base, i).id();
            let a = b.push_store(one, addr, ram).id;
            let ValueId::Instruction(b_id) = b.push_load::<false>(addr, 4, ram).id() else {
                panic!("load should produce an instruction value");
            };
            let ni = b.push_add(i, one).id();
            let bound = b.context_mut().get_const(8, 8).id();
            let cond = b.push_lt(ni, bound).id();
            b.push_cbranch_with_args(cond, body, vec![ni], body, vec![ni]);
            vec![
                MemoryAccess {
                    id: a,
                    block: body,
                    is_store: true,
                    ptr: addr,
                    size: 1,
                },
                MemoryAccess {
                    id: b_id,
                    block: body,
                    is_store: false,
                    ptr: addr,
                    size: 4,
                },
            ]
        };

        let numbering = crate::gvn::affine::precompute_forms(&tc.ctx, fid);
        let regions = collect_regions_for_base(&tc.ctx, &numbering, base, &accesses);
        assert!(regions.regions.is_empty());
        assert_eq!(regions.rejected_accesses.len(), 2);
    }
}
