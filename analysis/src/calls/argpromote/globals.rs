//! Constant-address (global) memory accesses as interface slots.
//!
//! A function that loads or stores at a *constant* real-ram address —
//! `load(ram:4, 0x454df8)` — reaches into absolute memory that no parameter
//! describes, keeping it opaque to the param-relative RAM channel (which only
//! models derefs of a promoted parameter, `param ± const`).
//!
//! Globals are part of the *effect system* (Jack's ruling, 2026-07-17): the
//! register channel's scan records each constant address as a
//! [`GlobalSlot`](qcode::value::GlobalSlot) effect, `materialize_interface`
//! mints one `glob_<addr>` by-value param per slot (recorded in
//! [`RegisterInterfaceMap::globals`](qcode::value::RegisterInterfaceMap)) and
//! this module rewrites the accesses to dereference the param. Call sites are
//! never touched directly:
//!
//! * a **regpure** site passes the address literal verbatim as a positional
//!   argument (threaded by `rewrite_call_regpure`, like a register input);
//! * an **implicit** (`Opaque`) site passes nothing — the binder seeds the
//!   param from the address literal recorded as its `origin`.
//!
//! Because the param equals the literal under either convention, the rewrite is
//! value-preserving, and the freshly param-relative deref is visible to the RAM
//! channel's footprint scan.
//!
//! [`grow_globals`] handles *late* discovery: a later round (constprop folding
//! an address to a constant) can surface a new global in an
//! already-materialized function. The interface then grows by appending — the
//! new param goes at the end of the root params, the literal is appended at
//! every **regpure** site (lockstep), and implicit sites stay zero-arg.

use qcode::value::QCodeMut;

use qcode::{
    context::Context,
    space::{LocalMemorySpaceId, Space, SpaceType},
    value::{
        BasicBlock, FunctionBody, FunctionEffects, FunctionId, GlobalSlot, LocalValueId, Value,
        ValueId, ValueRef,
        insn::{Load, Mnemonic, Store},
    },
};

use crate::calls::interface::append_entry_param_at_sites;

/// A writable global lives in real RAM. ROM/register/temporary addresses are left
/// alone: temporary is argpromote's own shadow, registers are varnodes (never a
/// constant deref), and ROM is read-only data outside this pass's remit.
fn is_real_ram(ctx: &Context, space: LocalMemorySpaceId) -> bool {
    space
        .shared()
        .is_some_and(|space| matches!(Space::from_id(ctx, space).ty, SpaceType::Ram))
}

/// Classify one load/store pointer as a global slot: a literal address into
/// real RAM. Returns the `(address, address width)` pair the effect channel
/// records.
pub(super) fn global_slot(
    ctx: &Context,
    fid: FunctionId,
    ptr: LocalValueId,
    space: LocalMemorySpaceId,
) -> Option<(u64, usize)> {
    let ptr = ptr.qualify(fid);
    if !matches!(ptr, ValueId::Literal(_)) || !is_real_ram(ctx, space) {
        return None;
    }
    let ValueRef::Literal(lit) = ValueRef::new(ptr, ctx) else {
        return None;
    };
    Some((lit.value(), lit.size()))
}

/// Mint the by-value param for one global slot on `fid` and redirect the
/// slot's accesses through it. `call_sites` receive the address literal as an
/// appended positional argument — the caller passes the **regpure** sites only
/// (materialization passes none; implicit sites bind the param from its
/// literal origin). Returns the new param, or `None` if `fid` has no root.
fn mint_global_param(
    ctx: &mut Context,
    fid: FunctionId,
    slot: GlobalSlot,
    call_sites: &[qcode::value::insn::InstructionId],
) -> Option<ValueId> {
    let addr = ctx.get_const(slot.addr, slot.size).id();
    let param = append_entry_param_at_sites(
        ctx,
        fid,
        slot.size,
        Some(format!("glob_{:x}", slot.addr)),
        // The literal origin is the single source for implicit binding, and
        // lets alias analysis treat the param as a static/global pointer,
        // disjoint from the live stack frame.
        Some(addr),
        call_sites,
        move |_, _, _| addr,
    )?;
    rewrite_accesses(ctx, fid, addr, param);
    Some(param)
}

// NOTE: the former `materialize_globals` (address-parameter path) is superseded
// by value-threading through the register effect channel (GLOBALS_AS_VARNODES.md).
// `grow_globals` below is retained for the late-discovery path pending the same
// migration.

/// Grow an already-materialized function's interface with globals surfaced
/// after materialization (a later constprop round folding an address to a
/// constant). Appends one slot per *new* constant address: the param at the end
/// of the root params, the literal argument at every regpure direct site, and
/// the slot in the interface map. Implicit sites stay zero-arg. Idempotent: a
/// rewritten access is param-relative, so re-scanning finds nothing, and an
/// address already in the map only has its (new) accesses redirected to the
/// existing param. Returns whether anything changed.
pub(super) fn grow_globals(ctx: &mut Context, graph: &crate::CallGraph, fid: FunctionId) -> bool {
    let f = FunctionBody::from_id(ctx, fid);
    if f.is_external() || !f.is_reg_materialized() {
        return false;
    }
    let Some(root) = f.root().map(|b| b.id) else {
        return false;
    };

    // Distinct constant real-ram addresses in the current body, in first-seen
    // order (deterministic: block/instruction order).
    let mut found: Vec<(u64, usize)> = Vec::new();
    for block in FunctionBody::from_id(ctx, fid).iter() {
        for insn in block.iter() {
            let (space, ptr) = match insn.mnemonic() {
                Mnemonic::Load(l) => (l.space, l.ptr),
                Mnemonic::Store(s) => (s.space, s.ptr),
                _ => continue,
            };
            if let Some(slot) = global_slot(ctx, fid, ptr, space)
                && !found.contains(&slot)
            {
                found.push(slot);
            }
        }
    }
    if found.is_empty() {
        return false;
    }

    let mapped: Vec<GlobalSlot> = FunctionBody::from_id(ctx, fid)
        .effects()
        .materialized()
        .map(|m| m.globals.clone())
        .unwrap_or_default();

    // Only regpure sites carry positional arguments; implicit (`Opaque`) and
    // indirect sites bind the new param from its literal origin, so they are
    // correct untouched — the very desync the old direct-site surgery had.
    let regpure_sites: Vec<_> = crate::calls::direct_call_sites(ctx, graph, fid)
        .into_iter()
        .filter(|&site| {
            matches!(
                ctx.get_insn(site).mnemonic(),
                Mnemonic::Call(c) if c.tag.is_regpure()
            )
        })
        .collect();

    let mut changed = false;
    for (addr, size) in found {
        let slot = GlobalSlot { addr, size };
        if mapped.contains(&slot) {
            // Already an interface slot: redirect the new accesses to the
            // existing param instead of minting a duplicate.
            let lit = ctx.get_const(addr, size).id();
            if let Some(param) = param_with_origin(ctx, root, lit) {
                rewrite_accesses(ctx, fid, lit, param);
                changed = true;
            }
            continue;
        }
        if mint_global_param(ctx, fid, slot, &regpure_sites).is_none() {
            continue;
        }
        // Record the appended slot in the interface map, keeping map order in
        // lockstep with param append order.
        if let FunctionEffects::Materialized(mut map) =
            FunctionBody::from_id(ctx, fid).effects().clone()
        {
            map.globals.push(slot);
            FunctionBody::from_id_mut(ctx, fid).set_effects(FunctionEffects::Materialized(map));
        }
        changed = true;
    }
    changed
}

/// The root param whose origin is the literal `lit`, if any.
fn param_with_origin(ctx: &Context, root: qcode::value::BlockId, lit: ValueId) -> Option<ValueId> {
    BasicBlock::from_id(ctx, root)
        .params()
        .find(|p| p.origin() == Some(lit))
        .map(|p| p.id())
}

/// Redirect every real-ram load/store at `addr` in `fid` to dereference `param`.
fn rewrite_accesses(ctx: &mut Context, fid: FunctionId, addr: ValueId, param: ValueId) {
    let ids: Vec<_> = FunctionBody::from_id(ctx, fid)
        .iter()
        .flat_map(|b| b.iter())
        .filter_map(|insn| match insn.mnemonic() {
            Mnemonic::Load(l)
                if l.ptr.qualify(insn.id.func) == addr && is_real_ram(ctx, l.space) =>
            {
                Some(insn.id)
            }
            Mnemonic::Store(s)
                if s.ptr.qualify(insn.id.func) == addr && is_real_ram(ctx, s.space) =>
            {
                Some(insn.id)
            }
            _ => None,
        })
        .collect();
    for id in ids {
        let new = match ctx.get_insn(id).mnemonic().clone() {
            Mnemonic::Load(l) => Mnemonic::Load(Load {
                ptr: param.localize(id.func),
                ..l
            }),
            Mnemonic::Store(s) => Mnemonic::Store(Store {
                ptr: param.localize(id.func),
                ..s
            }),
            _ => continue,
        };
        ctx.replace_instruction_mnemonic(id, new);
    }
}
