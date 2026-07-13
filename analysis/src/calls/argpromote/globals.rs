//! Lift fixed-address (global) memory accesses to parameters.
//!
//! A `pure_reg` function that loads or stores at a *constant* real-ram address —
//! `load(ram:4, 0x454df8)` — reaches into absolute memory that no parameter
//! describes. That keeps it impure and opaque to the param-relative RAM channel,
//! which only models derefs of a promoted parameter (`param ± const`).
//!
//! This step lifts each such constant address into a new by-value parameter
//! (`glob_<addr>`), rewrites the access to dereference the parameter, and threads
//! the *same address literal* as an argument at every direct caller. Because the
//! caller passes the literal unchanged, the parameter equals the constant at entry
//! and the rewrite is value-preserving. The access is now param-relative —
//! `load(ram:4, @glob_454df8)` — so the rest of argpromote (snapshotting, region
//! detection, shadow promotion) handles the memory behind it; this step only
//! functionalizes the *address*.

use rustc_hash::FxHashSet as HashSet;

use qcode::{
    context::Context,
    space::{Space, SpaceId, SpaceType},
    value::{
        FunctionBody, FunctionId, Value, ValueId, ValueRef,
        insn::{Load, Mnemonic, Store},
    },
};

use rustc_hash::FxHashSet;

use super::append_entry_param;

/// A writable global lives in real RAM. ROM/register/temporary addresses are left
/// alone: temporary is argpromote's own shadow, registers are varnodes (never a
/// constant deref), and ROM is read-only data outside this pass's remit.
fn is_real_ram(ctx: &Context, space: SpaceId) -> bool {
    matches!(Space::from_id(ctx, space).ty, SpaceType::Ram)
}

/// Lift every constant real-ram load/store address in `fid` into a parameter,
/// threading the address literal to each direct caller. Returns whether the
/// function changed.
pub(super) fn globalize_constants(
    ctx: &mut Context,
    address_taken: &FxHashSet<FunctionId>,
    fid: FunctionId,
) -> bool {
    let f = FunctionBody::from_id(ctx, fid);
    if f.is_external() || !f.is_pure_reg() {
        return false;
    }
    // Adding a param appends an argument at every direct caller's `Call.args`. An
    // address-taken function may also be reached by an indirect call this pass
    // cannot find and rewrite, which would desync the `param[i] ↔ arg[i]` lockstep.
    // Same closed-world gate as `try_promote`.
    if address_taken.contains(&fid) {
        return false;
    }

    // Distinct constant addresses used as a real-ram load/store base. Keyed by the
    // literal `ValueId` — literals are interned, so one id per (address, width).
    let mut globals: Vec<ValueId> = Vec::new();
    let mut seen: HashSet<ValueId> = HashSet::default();
    for block in FunctionBody::from_id(ctx, fid).iter() {
        for insn in block.iter() {
            let (space, ptr) = match insn.mnemonic() {
                Mnemonic::Load(l) => (l.space, l.ptr.qualify(insn.id.func)),
                Mnemonic::Store(s) => (s.space, s.ptr.qualify(insn.id.func)),
                _ => continue,
            };
            if matches!(ptr, ValueId::Literal(_)) && is_real_ram(ctx, space) && seen.insert(ptr) {
                globals.push(ptr);
            }
        }
    }
    if globals.is_empty() {
        return false;
    }

    for addr in globals {
        // The literal's width is the address width — the size of the new pointer
        // param. Read it (and the value, for the name) before the `&mut` borrow.
        let ValueRef::Literal(lit) = ValueRef::new(addr, ctx) else {
            continue;
        };
        let size = lit.size();
        let name = format!("glob_{:x}", lit.value());

        // The caller passes the address literal verbatim: literals are context-
        // global, so `addr` is a valid `ValueId` in any function's body. Record the
        // address literal as the param's `origin` so alias analysis recognizes this
        // param as a static/global pointer, disjoint from the live stack frame.
        let Some(param) =
            append_entry_param(ctx, fid, size, Some(name), Some(addr), move |_, _, _| addr)
        else {
            continue;
        };

        rewrite_accesses(ctx, fid, addr, param);
    }
    true
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
