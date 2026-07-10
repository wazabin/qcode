//! Canonicalize fixed `@SP ± N` stack addresses to one root-dominating
//! representative per offset.
//!
//! This is the `@SP`-rooted successor to the brighten/lower round-trip. brighten
//! used to relabel `@SP` uses to an interned `@stack_base` *literal* purely so
//! that `@stack_base − N` const-folded to a single literal — giving every
//! reference to a slot one stable `ValueId`. We keep `@SP` instead, but a slot's
//! address is now an instruction (`Sub(@SP, N)`), and two occurrences in
//! different blocks are distinct `ValueId`s. Downstream passes (notably mem2reg)
//! key stack slots by that `ValueId`, so they need exactly one per slot.
//!
//! [`canonicalize_sp_slots`] restores that invariant the way `lower_stack`
//! already materialized `incoming ± N` at the root: it builds one `@SP ± N`
//! instruction per distinct offset at the entry block (which dominates the whole
//! function) and points every load/store at that representative. It only touches
//! the affine `@SP ± N` form; `@stack_base` literals are left to the legacy path,
//! so this is inert until brighten is removed.

use std::collections::{BTreeSet, HashMap};

use qcode::{
    builder::Builder,
    value::{
        FunctionId, ValueId, Varnode, VarnodeId,
        insn::Mnemonic,
        util::{base_ref::BaseRef, host_mut::HostMut},
    },
};

use super::frame::incoming_sp_param;
use crate::gvn::affine::precompute_forms;

/// Rewrite every fixed `@SP ± N` load/store address in `fid` to a single
/// root-block representative per offset `N`. Returns whether anything changed.
pub fn canonicalize_sp_slots<'str, H: HostMut<'str>>(
    host: &mut H,
    fid: FunctionId,
    sp_reg: VarnodeId,
) -> bool {
    let numbering = precompute_forms(host.read_host(), fid);
    let Some(sp_param) = incoming_sp_param(host.read_host(), fid, sp_reg) else {
        return false;
    };
    let Some(root) = host.function_ref(fid).root().map(|b| b.id) else {
        return false;
    };

    // Each distinct load/store pointer that is `@SP ± N` with a fixed offset.
    let mut ptr_offset: HashMap<ValueId, i64> = HashMap::new();
    for block in host.function_ref(fid).blocks() {
        for insn in block.iter() {
            let ptr = match insn.mnemonic() {
                Mnemonic::Load(load) => load.ptr,
                Mnemonic::Store(store) => store.ptr,
                _ => continue,
            };
            if let Some((base, off)) = numbering.base_offset(ptr)
                && base == sp_param
            {
                ptr_offset.insert(ptr, off);
            }
        }
    }
    if ptr_offset.is_empty() {
        return false;
    }

    let ptr_width = Varnode::from_id(host.shared(), sp_reg).size();
    let offsets: BTreeSet<i64> = ptr_offset.values().copied().collect();

    // Reuse a prior run's representatives so this pass is idempotent: the first
    // root-block `@SP ± N` instruction per offset (in program order) dominates
    // every use of that slot. Offset 0 is `@SP` itself.
    let mut repr: HashMap<i64, ValueId> = HashMap::new();
    if offsets.contains(&0) {
        repr.insert(0, sp_param);
    }
    for insn in host.function_ref(fid)
        .root()
        .unwrap()
        .iter()
    {
        let v = ValueId::Instruction(insn.id);
        if let Some((base, off)) = numbering.base_offset(v)
            && base == sp_param
            && offsets.contains(&off)
        {
            repr.entry(off).or_insert(v);
        }
    }

    // Materialize the rest at the entry-block start (which dominates everything).
    let missing: Vec<i64> = offsets
        .iter()
        .copied()
        .filter(|o| !repr.contains_key(o))
        .collect();
    if !missing.is_empty() {
        let mut b = Builder::from_block(BaseRef::new(host.reborrow_host(), root));
        b.set_insert_point_to_start();
        for off in missing {
            let mag = b.context().get_const(off.unsigned_abs(), ptr_width).id();
            let rep = if off > 0 {
                b.push_add(sp_param, mag).id()
            } else {
                b.push_sub(sp_param, mag).id()
            };
            repr.insert(off, rep);
        }
        unsafe { b.dont_finalize() };
    }

    // Point every occurrence at its representative; the now-dead per-site
    // address instructions are reclaimed by DCE.
    let mut changed = false;
    for (ptr, off) in ptr_offset {
        let rep = repr[&off];
        if ptr != rep {
            host.replace_all_uses_with(ptr, rep);
            changed = true;
        }
    }
    changed
}

// ----- pass ------------------------------------------------------------------

use crate::{FunctionBody, FunctionPass, ModuleView};

/// Runs [`canonicalize_sp_slots`] over a function, resolving `@SP` from the
/// configured stack-pointer register. Replaces the legacy `brighten`/`lower_stack`
/// round-trip: it unifies the per-site `@SP ± N` address instructions mem2reg's
/// register forwarding produces into one representative per offset, so the
/// ValueId-keyed slot promoter sees a single pointer per slot.
#[derive(Default)]
pub struct CanonicalizeSpSlots;

impl FunctionPass for CanonicalizeSpSlots {
    const NAME: &'static str = "canonicalize_sp_slots";
    fn description(&self) -> &'static str {
        "Canonicalize @SP±N stack slots to one representative per offset"
    }
    fn run<'str>(
        &self,
        m: &ModuleView<'_, 'str>,
        f: &mut FunctionBody<'str>,
    ) -> std::result::Result<bool, String> {
        let sp_reg = m.ctx().registers[&m.env().cfg.stack_pointer];
        let fid = f.id();
        let mut host = f.host(m);
        Ok(canonicalize_sp_slots(&mut host, fid, sp_reg))
    }
}

crate::register_function_pass!(CanonicalizeSpSlots);

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{
        context::Context,
        testing::TestContext,
        value::{BasicBlock, Function, Value},
    };

    /// Two `load(@SP - 8)` in different blocks become one shared pointer after
    /// canonicalization — the single-`ValueId`-per-slot invariant mem2reg needs.
    #[test]
    fn unifies_same_slot_across_blocks() {
        let mut tc = TestContext::new();
        let sp_reg = tc.r0;
        let ram = tc.ctx.default_space;
        let fid = Function::make(&mut tc.ctx, "f".into()).unwrap().id;
        let root = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x1000, __f)
        };
        let other = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x2000, __f)
        };
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(root).unwrap();
            f.add_block(root);
            f.add_block(other);
        }
        let pid = BasicBlock::from_id_mut(&mut tc.ctx, root).push_param(8).id;
        tc.ctx.values.block_param_mut(pid).origin = Some(ValueId::Varnode(sp_reg));
        let sp = ValueId::BlockParam(pid);

        // `load(@SP - 8)` in each block — distinct Sub ValueIds.
        let load_in = |block, tc: &mut TestContext| {
            let mut b =
                qcode::builder::Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, block));
            let c8 = b.context_mut().get_const(8, 8).id();
            let addr = b.push_sub(sp, c8).id();
            let load = b.push_load::<false>(addr, 8, ram);
            let id = load.id();
            unsafe { b.dont_finalize() };
            id
        };
        let l0 = load_in(root, &mut tc);
        let l1 = load_in(other, &mut tc);

        let ptr_of = |tc: &TestContext, load: ValueId| {
            let ValueId::Instruction(id) = load else {
                unreachable!()
            };
            match tc.ctx.get_insn(id).mnemonic() {
                Mnemonic::Load(l) => l.ptr,
                _ => unreachable!(),
            }
        };
        assert_ne!(
            ptr_of(&tc, l0),
            ptr_of(&tc, l1),
            "distinct before canonicalization"
        );

        assert!(canonicalize_sp_slots(&mut &mut tc.ctx, fid, sp_reg));

        assert_eq!(
            ptr_of(&tc, l0),
            ptr_of(&tc, l1),
            "both loads share one representative pointer for slot -8"
        );

        // Idempotent: a second run reuses the representative and reports no change.
        let shared = ptr_of(&tc, l0);
        assert!(
            !canonicalize_sp_slots(&mut &mut tc.ctx, fid, sp_reg),
            "second run must be a no-op"
        );
        assert_eq!(
            ptr_of(&tc, l0),
            shared,
            "representative is stable across runs"
        );
    }
}
