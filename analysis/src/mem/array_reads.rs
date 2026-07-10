//! Read-only array promotion: "mem2reg for read-only arrays".
//!
//! `argpromote` snapshots a by-value array argument that a pure function only
//! *reads* into a private shadow region: a seed store `store(temp:N, @base <- @arr)`
//! of the whole array value, followed by any number of lane loads
//! `load(temp:esz, @base + i*esz + c)`. Because the region is never written past
//! the seed, [`array_promote`](crate::mem::array_promote) — which requires exactly
//! one strided lane *store* — never touches it, and the loads stay as memory
//! traffic that the fold recognizers (strlen Layer 1) cannot see.
//!
//! This pass rewrites every such lane load to a direct `at(@arr, word)` read of
//! the seeded array value and deletes the seed store. Their domains are disjoint
//! by construction: `array_promote` needs one strided lane store, `array_reads`
//! needs *zero* non-seed stores.
//!
//! Soundness (all gated in [`try_match`]): the region lives in a `Temporary`
//! (argpromote shadow) space — never real RAM, whose reads must stay loads;
//! `@arr` is a root array param, so it dominates every load and `at` is a pure
//! value op; the seed fully initializes the region; and every region access is
//! either the seed or an in-bounds lane load affine in `@base`. Replacing a load
//! of an unaliased, fully-seeded private region with `at` of the seeded value is
//! then unconditionally sound.

use qcode::{
    builder::Builder,
    space::{Space, SpaceId, SpaceType},
    types::TypeId,
    value::{
        FunctionId, FunctionRef, InstructionRef, ValueId,
        insn::{InstructionId, IntrinsicApp, IntrinsicId, Mnemonic},
        util::{
            base_ref::{BaseRef, HostRef},
            host_mut::HostMut,
        },
    },
};

use crate::gvn::affine::precompute_forms;
use crate::sequence::{affine_base_const, affine_strided_lane};

#[derive(Default)]
pub struct ArrayReads;

/// Host-routed mirror of [`qcode::context::Context::stored_type_of`]: reads the
/// checked-out function's owned arena for instruction/param results.
fn stored_type_of(host: HostRef, id: ValueId) -> Option<TypeId> {
    match id {
        // Instruction/param results live in the (possibly checked-out) function
        // arena, so route them through the host.
        ValueId::Instruction(iid) => Some(qcode::value::InstructionRef::new(host, iid).type_id()),
        ValueId::BlockParam(pid) => Some(host.block_param(pid).type_id),
        // Everything else is shared data; the Context method reads it directly.
        other => host.shared().stored_type_of(other),
    }
}

/// The word index of a recognized lane load into the snapshot array.
enum LaneIdx {
    /// A constant word `arr[w]`.
    Const(i64),
    /// A dynamic `arr[idx + od]` (`idx` a block param, `od` the constant word offset).
    Strided(ValueId, i64),
}

/// A recognized read-only shadow region and the lane loads to rewrite.
struct ReadsMatch {
    /// The `Array` snapshot value (a root param) the loads read from.
    arr: ValueId,
    /// The seed store `store(region, base <- arr)` (dead after the rewrite).
    seed_id: InstructionId,
    /// Lane loads to replace with `at(arr, word)`.
    loads: Vec<(InstructionId, LaneIdx)>,
}

/// A region access collected in a temporary (shadow) space.
struct Acc {
    id: InstructionId,
    ptr: ValueId,
    size: usize,
    space: SpaceId,
    stored: Option<ValueId>,
}

fn try_match(host: HostRef, fid: FunctionId) -> Option<ReadsMatch> {
    // v1 conservatism (mirrors `array_promote`): no calls/indirect control flow.
    for block in FunctionRef::new(host, fid).iter() {
        for insn in block.iter() {
            if matches!(
                insn.mnemonic(),
                Mnemonic::Call(_) | Mnemonic::CallInd(_) | Mnemonic::BranchInd(_)
            ) {
                return None;
            }
        }
    }

    // Collect every access in an argpromote shadow (`Temporary`) space.
    let is_temp =
        |sp: SpaceId| matches!(Space::from_id(host.shared(), sp).ty, SpaceType::Temporary);
    let mut accesses: Vec<Acc> = Vec::new();
    for block in FunctionRef::new(host, fid).iter() {
        for insn in block.iter() {
            match insn.mnemonic() {
                Mnemonic::Load(l) if is_temp(l.space) => accesses.push(Acc {
                    id: insn.id,
                    ptr: l.ptr,
                    size: l.size,
                    space: l.space,
                    stored: None,
                }),
                Mnemonic::Store(s) if is_temp(s.space) => accesses.push(Acc {
                    id: insn.id,
                    ptr: s.ptr,
                    size: s.size,
                    space: s.space,
                    stored: Some(s.src),
                }),
                _ => {}
            }
        }
    }

    // Seed store: `store(region, base <- arr)`, `arr` a root `[elem; N]` param,
    // `base` a root param, size `N * size_of(elem)`.
    let root_params: Vec<ValueId> = FunctionRef::new(host, fid)
        .root()?
        .params()
        .map(|p| p.id())
        .collect();
    let is_root = |v: ValueId| root_params.contains(&v);
    let seed = accesses.iter().find(|a| {
        a.stored.is_some_and(|src| {
            is_root(src)
                && stored_type_of(host, src)
                    .and_then(|t| host.shared().types.array_of(t))
                    .is_some()
        }) && is_root(a.ptr)
    })?;
    let arr = seed.stored.unwrap();
    let base = seed.ptr;
    let region_space = seed.space;
    let seed_id = seed.id;
    let (elem_ty, count) = host.shared().types.array_of(stored_type_of(host, arr)?)?;
    let esz = host.shared().types.size_of(elem_ty);
    if esz == 0 || count == 0 || seed.size != count * esz {
        return None;
    }
    let count = count as i64;

    // Address roots: `@base` (and any pass-through of it) name the region.
    let val_root = crate::loop_info::value_roots(host, fid);
    let root_of = |v: ValueId| -> Option<ValueId> { val_root.get(&v).copied() };
    let base_root = root_of(base)?;
    let is_base = |v: ValueId| root_of(v) == Some(base_root);
    let is_any_root = |v: ValueId| root_of(v).is_some();
    let numbering = precompute_forms(host, fid);

    // Classify every non-seed region access. Any store other than the seed, any
    // partial/misaligned load, or any region address not understood as affine in a
    // root base declines the promotion. Accesses affine in a *different* root base
    // are a different region and are ignored.
    let mut loads: Vec<(InstructionId, LaneIdx)> = Vec::new();
    for a in &accesses {
        if a.id == seed_id || a.space != region_space {
            continue;
        }
        if a.stored.is_some() {
            return None; // a store other than the seed — not read-only
        }
        if a.size != esz {
            return None; // partial / misaligned lane
        }
        // `is_base` pins the region base to `base_root`, so a general dynamic index
        // (even one that is itself a root parameter) is classified as the index,
        // not the base — order-independent for a unit element stride.
        if let Some((_b, idx, c)) = affine_strided_lane(&numbering, a.ptr, esz, is_base) {
            if c % esz as i64 != 0 {
                return None;
            }
            let od = c / esz as i64;
            if od < 0 || od >= count {
                return None;
            }
            loads.push((a.id, LaneIdx::Strided(idx, od)));
            continue;
        }
        // A strided lane of a *different* root region is a separate snapshot; skip
        // it rather than declining the whole promotion.
        if affine_strided_lane(&numbering, a.ptr, esz, is_any_root).is_some() {
            continue;
        }
        if let Some((b, c)) = affine_base_const(&numbering, a.ptr, &is_any_root) {
            if root_of(b) == Some(base_root) {
                if !is_base(b) || c % esz as i64 != 0 {
                    return None;
                }
                let w = c / esz as i64;
                if w < 0 || w >= count {
                    return None;
                }
                loads.push((a.id, LaneIdx::Const(w)));
            }
            continue; // affine in some root — ours (handled) or another region
        }
        return None; // an unmodelled access to the region
    }

    if loads.is_empty() {
        return None; // nothing to rewrite
    }
    Some(ReadsMatch {
        arr,
        seed_id,
        loads,
    })
}

/// Rewrite each matched lane load to `at(arr, word)` and drop the seed store.
fn apply<'str, H: HostMut<'str>>(host: &mut H, m: &ReadsMatch) -> bool {
    let at_id = IntrinsicId::from_name("at").expect("at registered");
    // The `at(arr, i)` result type is the array's element type. Compute it through
    // the shared type interner's `&self` path (no `shared_mut`, so it holds on a
    // checked-out host); this mirrors `at`'s `result_type`.
    let arr_ty = stored_type_of(host.read_host(), m.arr);
    let at_ty = arr_ty
        .and_then(|t| host.shared().types.seq_elem_of(t))
        .or(arr_ty)
        .expect("seeded array value has a type");
    for (load_id, lane) in &m.loads {
        let block = InstructionRef::new(host.read_host(), *load_id)
            .parent()
            .map(|b| b.id);
        let Some(block) = block else { continue };
        // Materialize the word index (checkout-safe builder: const/add only).
        let idx = {
            let mut b = Builder::from_block(BaseRef::new(host.reborrow_host(), block));
            b.set_insert_point_before(*load_id);
            let idx = build_index(&mut b, lane);
            unsafe { b.dont_finalize() };
            idx
        };
        // Build `at(arr, idx)` with the explicit element type and splice it before
        // the load (avoids the Builder's `context_mut` type-mint path).
        let at_val = host.push_mnemonic_with_type(
            block.func,
            Mnemonic::Intrinsic(IntrinsicApp {
                id: at_id,
                args: vec![m.arr, idx],
            }),
            at_ty,
        );
        host.insert_insn_before(block, *load_id, at_val);
        host.replace_all_uses_with(ValueId::Instruction(*load_id), ValueId::Instruction(at_val));
        host.remove_instruction(*load_id);
    }
    host.remove_instruction(m.seed_id);
    true
}

/// Materialize the word index of a lane load: a literal for a constant word, or
/// `idx (+ od)` at the index's own width for a dynamic lane.
fn build_index<'str, 'ctx, Ctx: HostMut<'str>>(
    b: &mut Builder<'str, 'ctx, Ctx>,
    lane: &LaneIdx,
) -> ValueId {
    match *lane {
        LaneIdx::Const(w) => b.context().get_const(w as u64, 8).id(),
        LaneIdx::Strided(idx, 0) => idx,
        LaneIdx::Strided(idx, od) => {
            let ty = b.context().type_of(idx);
            let width = b.context().types.size_of(ty);
            let mask = if width >= 8 {
                u64::MAX
            } else {
                (1u64 << (width * 8)) - 1
            };
            let c = b.context().get_const(od as u64 & mask, width).id();
            b.push_add(idx, c).id()
        }
    }
}

// ----- pass ------------------------------------------------------------------

use crate::{FunctionBody, FunctionPass, ModuleView};

impl FunctionPass for ArrayReads {
    const NAME: &'static str = "array_reads";

    fn description(&self) -> &'static str {
        "Rewrite read-only argpromote array snapshots to direct at(arr, i) reads"
    }

    fn run<'str>(
        &self,
        m: &ModuleView<'_, 'str>,
        f: &mut FunctionBody<'str>,
    ) -> Result<bool, String> {
        let fid = f.id();
        let mut host = f.host(m);
        if !FunctionRef::new(host.read_host(), fid).is_pure() {
            return Ok(false);
        }
        Ok(match try_match(host.read_host(), fid) {
            Some(matched) => apply(&mut host, &matched),
            None => false,
        })
    }
}

crate::register_function_pass!(ArrayReads);

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{
        builder::Builder,
        context::Context,
        testing::TestContext,
        value::{BasicBlock, Function, Value},
    };

    use crate::test_util::run_function_pass;

    /// Extra region traffic injected into the built function, to exercise the gates.
    #[derive(Clone, Copy, PartialEq)]
    enum Extra {
        /// Just the seed + two lane loads (const index 2, dynamic index `@i`).
        None,
        /// Add a lane store into the region (array_promote's territory).
        Store,
        /// Add a region load whose address is not affine in `@base`.
        Unknown,
    }

    /// Build `fn f(@arr:[i8;8], @base, @i)` seeding a region in `space` with `@arr`,
    /// reading lanes `base+2` and `base+@i`, plus any `extra` traffic. Returns `fid`.
    fn build(tc: &mut TestContext, ram_region: bool, extra: Extra) -> FunctionId {
        const N: usize = 8;
        let i8 = tc.ctx.types.get_or_make_int(1);
        let arr_ty = tc.ctx.types.get_or_make_array(i8, N);
        let space = if ram_region {
            tc.ctx.default_space
        } else {
            tc.ctx.make_temp_space()
        };
        let ram = tc.ctx.default_space;

        let fid = Function::make(&mut tc.ctx, "f".into()).unwrap().id;
        // Build the entry block *owned by* `fid` (block.func == fid), so the pass
        // can check the function out cleanly (no reattributed blocks).
        let entry = BasicBlock::make(&mut tc.ctx, fid).id;
        Function::from_id_mut(&mut tc.ctx, fid)
            .set_root(entry)
            .unwrap();
        let arr_pid = BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(N).id;
        tc.ctx.values.block_param_mut(arr_pid).type_id = arr_ty;
        let arr = ValueId::BlockParam(arr_pid);
        let base =
            ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(8).id);
        let i = ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(8).id);

        let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
        b.push_store(arr, base, space); // seed
        // const-index lane `base + 2`.
        let two = b.context_mut().get_const(2, 8).id();
        let a_const = b.push_add(base, two).id();
        b.push_load::<false>(a_const, 1, space);
        // dynamic-index lane `base + @i`.
        let a_dyn = b.push_add(base, i).id();
        b.push_load::<false>(a_dyn, 1, space);
        match extra {
            Extra::None => {}
            Extra::Store => {
                let z = b.context_mut().get_const(0, 1).id();
                b.push_store(z, a_dyn, space);
            }
            Extra::Unknown => {
                // An opaque pointer (a RAM load) is not affine in `@base`.
                let p = b.push_load::<false>(base, 8, ram).id();
                b.push_load::<false>(p, 1, space);
            }
        }
        let ret = b.context_mut().get_const(0, 8).id();
        b.push_return(ret);
        drop(b);

        Function::from_id_mut(&mut tc.ctx, fid).set_is_pure(true);
        fid
    }

    fn temp_load_count(ctx: &Context, fid: FunctionId) -> usize {
        Function::from_id(ctx, fid)
            .iter()
            .flat_map(|blk| blk.iter())
            .filter(|i| {
                matches!(i.mnemonic(), Mnemonic::Load(l)
                    if matches!(Space::from_id(ctx, l.space).ty, SpaceType::Temporary))
            })
            .count()
    }

    fn at_count(ctx: &Context, fid: FunctionId) -> usize {
        let at_id = IntrinsicId::from_name("at").unwrap();
        Function::from_id(ctx, fid)
            .iter()
            .flat_map(|blk| blk.iter())
            .filter(|i| matches!(i.mnemonic(), Mnemonic::Intrinsic(a) if a.id == at_id))
            .count()
    }

    #[test]
    fn read_only_region_promoted() {
        let mut tc = TestContext::new();
        let fid = build(&mut tc, /*ram*/ false, Extra::None);
        assert!(run_function_pass::<ArrayReads>(&mut tc.ctx, fid).unwrap());
        assert_eq!(temp_load_count(&tc.ctx, fid), 0, "lane loads become at()");
        assert_eq!(at_count(&tc.ctx, fid), 2, "both lanes rewritten to at()");
    }

    #[test]
    fn store_in_region_bails() {
        let mut tc = TestContext::new();
        let fid = build(&mut tc, /*ram*/ false, Extra::Store);
        assert!(
            !run_function_pass::<ArrayReads>(&mut tc.ctx, fid).unwrap(),
            "a non-seed store is array_promote's territory"
        );
        assert_eq!(at_count(&tc.ctx, fid), 0, "IR unchanged");
    }

    #[test]
    fn ram_region_bails() {
        let mut tc = TestContext::new();
        let fid = build(&mut tc, /*ram*/ true, Extra::None);
        assert!(
            !run_function_pass::<ArrayReads>(&mut tc.ctx, fid).unwrap(),
            "real RAM reads must stay loads"
        );
        assert_eq!(at_count(&tc.ctx, fid), 0, "IR unchanged");
    }

    #[test]
    fn unknown_address_bails() {
        let mut tc = TestContext::new();
        let fid = build(&mut tc, /*ram*/ false, Extra::Unknown);
        assert!(
            !run_function_pass::<ArrayReads>(&mut tc.ctx, fid).unwrap(),
            "an unmodelled region address declines the promotion"
        );
        assert_eq!(at_count(&tc.ctx, fid), 0, "IR unchanged");
    }
}
