//! Classifying constant-address (global) memory accesses.
//!
//! A function that loads or stores at a *constant* real-ram address —
//! `load(ram:4, 0x454df8)` — reaches into absolute memory that no parameter
//! describes. The RAM effect channel (`argpromote::ram`) is the single owner of
//! global materialization: it recognizes these accesses via [`global_slot`] and
//! threads each global's value/write through the interface. This module only
//! provides that classifier (and its `is_real_ram` helper); the former bespoke
//! register-channel / `grow_globals` interface-growth path was retired once the
//! RAM channel took ownership (Phase 3, see `GLOBALS_AS_VARNODES.md`).

use qcode::{
    context::Context,
    space::{LocalMemorySpaceId, Space, SpaceType},
    value::{FunctionId, LocalValueId, Value, ValueId, ValueRef},
};

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
