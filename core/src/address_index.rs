//! Disposable address lookup over an immutable qcode module snapshot.
//!
//! [`AddressIndex`] is derived state: consumers build it for a [`Context`], keep
//! it only while that context remains structurally unchanged, and rebuild it
//! after mutation. It is intentionally not stored in or serialized with the IR.
//!
//! # Provenance
//!
//! An index remembers the [`Revision`] of the context it was computed for —
//! which module instance, and at what point in that module's history of
//! address-bearing changes. It is [current](AddressIndex::is_current) while the
//! context is still at that revision. The mutators of this crate that take an
//! index alongside the context (`*_indexed`) move the context's revision and,
//! when the index was current going in, bring it along, so a caller that
//! threads one index through every mutation keeps it current for free; a
//! mutation made without the index leaves it behind, and it stays behind until
//! it is [refreshed](AddressIndex::refresh). A checked binding
//! ([`LiftTarget`](crate::lift::LiftTarget)) uses this to tell a complete index
//! from one that omits something, without rescanning the module.
//!
//! Editing the index's contents by hand — [`forget`](AddressIndex::forget),
//! [`set_block`](AddressIndex::set_block),
//! [`rehome_block`](AddressIndex::rehome_block),
//! [`register`](AddressIndex::register) — drops its provenance outright: an
//! index that has been edited follows from no revision of the context until
//! it is refreshed, or the caller [vouches](AddressIndex::mark_current) for
//! it. `mark_current` is the one way to claim currency without a rebuild; it
//! is for a caller that applied a mutation's effects to the index by hand,
//! and a wrong claim is that caller's bug. The address-bearing fields of
//! blocks and function interfaces are crate-private, and a body is reachable
//! mutably only through its verbs ([`BodyMut`](crate::value::BodyMut)), so
//! no change to what a module covers happens outside the mutators that move
//! the revision — see [`Context::revision`].

use rustc_hash::FxHashMap;

use crate::{
    context::{Context, Revision},
    error::{Error, ErrorTy, Result},
    value::{BlockId, FunctionBody, FunctionId, ValueId},
};

/// A live module entity selected by a machine address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressTarget {
    Function(FunctionId),
    Block(BlockId),
}

/// An immutable, disposable address-to-entity snapshot.
///
/// Equality compares contents — the addresses and what they name — and not
/// provenance, which is read through [`provenance`](Self::provenance).
#[derive(Debug, Clone, Default)]
pub struct AddressIndex {
    targets: FxHashMap<u64, AddressTarget>,
    /// The last address registered, with its target: what a lift asks for
    /// next, as an instruction's fall-through placeholder becomes the next
    /// instruction's entry. Answered without a probe of the map.
    last: Option<(u64, AddressTarget)>,
    /// Addresses known to start a block because something branches to them.
    ///
    /// Learned, not derived: an address becomes a boundary the first time a
    /// block has to be split at it. Remembering that is what stops a run from
    /// being folded across the same address again on the next pass, which for a
    /// loop header — the branch target that is discovered *after* the run
    /// through it — would otherwise repeat forever.
    boundaries: rustc_hash::FxHashSet<u64>,
    /// The context revision this index reflects, if it is known to reflect
    /// one; see the [module documentation](self).
    provenance: Option<Revision>,
}

impl PartialEq for AddressIndex {
    fn eq(&self, other: &Self) -> bool {
        self.targets == other.targets && self.boundaries == other.boundaries
    }
}

impl Eq for AddressIndex {}

impl AddressIndex {
    /// Computes an index for the current live shape of `ctx`.
    ///
    /// Primary and extra block addresses are indexed. Functions are installed
    /// last so a function wins the intentional collision with its entry block.
    pub fn analyze(ctx: &Context<'_>) -> Self {
        let mut targets = FxHashMap::default();

        for block_id in ctx.block_ids() {
            let block = ctx.block(block_id);
            if let Some(address) = block.address {
                targets
                    .entry(address)
                    .or_insert(AddressTarget::Block(block_id));
            }
            for &address in &block.extra_addresses {
                targets
                    .entry(address)
                    .or_insert(AddressTarget::Block(block_id));
            }
        }

        for function in ctx.functions() {
            if let Some(address) = function.address() {
                targets.insert(address, AddressTarget::Function(function.id));
            }
        }

        Self {
            targets,
            last: None,
            boundaries: rustc_hash::FxHashSet::default(),
            provenance: Some(ctx.revision()),
        }
    }

    /// Forgets every address and boundary, keeping the index's capacity. The
    /// index then describes no context until it is refreshed or marked.
    pub fn clear(&mut self) {
        self.targets.clear();
        self.last = None;
        self.boundaries.clear();
        self.provenance = None;
    }

    /// Recomputes this index after a structural mutation that changes several
    /// addresses at once (for example function splitting or block rehoming).
    pub fn refresh(&mut self, ctx: &Context<'_>) {
        *self = Self::analyze(ctx);
    }

    /// The context revision this index reflects: the one it was computed at,
    /// carried forward by every tracked mutation since. `None` for an index
    /// that was cleared or built by [`default`](Self::default).
    pub fn provenance(&self) -> Option<Revision> {
        self.provenance
    }

    /// Whether this index reflects `ctx` as it is now: it was computed for
    /// this very context instance (not a clone, not another module with the
    /// same ids), and every address-bearing change since was made with it.
    /// Then no address the context covers is missing from it.
    pub fn is_current(&self, ctx: &Context<'_>) -> bool {
        self.provenance == Some(ctx.revision())
    }

    /// Whether this index was computed for `ctx` at all, whatever has changed
    /// since.
    pub fn describes(&self, ctx: &Context<'_>) -> bool {
        self.provenance
            .is_some_and(|revision| revision.identity == ctx.identity())
    }

    /// Declares that this index reflects `ctx` as it is now.
    ///
    /// This is the low-level, caller-maintained path: the caller mutated the
    /// context without threading this index through, applied the effects by
    /// hand ([`set_block`](Self::set_block), [`forget`](Self::forget),
    /// [`register`](Self::register)), and vouches that nothing is missing. A
    /// checked binding trusts the claim; a wrong one lets a construction make a
    /// second block or function at an address the context already covers.
    /// Prefer the tracked `*_indexed` mutators, or a [`refresh`](Self::refresh),
    /// wherever the claim is not obviously true.
    pub fn mark_current(&mut self, ctx: &Context<'_>) {
        self.provenance = Some(ctx.revision());
    }

    /// Runs a mutation of `ctx` that updates this index as it goes, and keeps
    /// the index current across it if it was current before. An index that was
    /// already behind stays behind: the mutation cannot know what it missed.
    pub(crate) fn tracked<'str, R>(
        &mut self,
        ctx: &mut Context<'str>,
        mutation: impl FnOnce(&mut Context<'str>, &mut Self) -> R,
    ) -> R {
        let current = self.is_current(ctx);
        let result = mutation(ctx, self);
        if current {
            self.mark_current(ctx);
        }
        result
    }

    /// Re-point `addr` from a relocated block `old` to its clone `new`, in place.
    ///
    /// The incremental analogue of a [`refresh`](Self::refresh) after a block
    /// rehome: the caller already knows exactly which address moved and where, so
    /// there is no need to re-scan the whole module. A no-op unless `addr` is
    /// currently indexed to `old` — this preserves [`analyze`](Self::analyze)'s
    /// function-over-block precedence (a function entry that deliberately shadows
    /// its root block, or another block that already owns the address, is left
    /// untouched).
    ///
    /// Like every edit of the index's contents this drops its provenance: what
    /// the index says no longer follows from any revision of the context, until
    /// the caller [vouches](Self::mark_current) for it or refreshes.
    pub fn rehome_block(&mut self, addr: u64, old: BlockId, new: BlockId) {
        self.provenance = None;
        self.last = None;
        if self.targets.get(&addr) == Some(&AddressTarget::Block(old)) {
            self.targets.insert(addr, AddressTarget::Block(new));
        }
    }

    /// Drops `address` from the index, so it resolves to nothing until it is
    /// registered again. Used when a block stops covering an address. Drops
    /// the index's provenance, as [`rehome_block`](Self::rehome_block) does.
    pub fn forget(&mut self, address: u64) {
        self.provenance = None;
        self.last = None;
        self.targets.remove(&address);
    }

    /// Points `address` at `block`, whatever it pointed at before.
    ///
    /// For a caller that has just made `block` cover an address another block
    /// used to — absorbing that block, typically, which leaves the index
    /// naming something deleted. Drops the index's provenance, as
    /// [`rehome_block`](Self::rehome_block) does.
    pub fn set_block(&mut self, address: u64, block: BlockId) {
        self.provenance = None;
        self.last = Some((address, AddressTarget::Block(block)));
        self.targets.insert(address, AddressTarget::Block(block));
    }

    /// Records that `address` starts a block, and must keep starting one.
    pub fn mark_boundary(&mut self, address: u64) {
        self.boundaries.insert(address);
    }

    /// Whether `address` is known to start a block.
    pub fn is_boundary(&self, address: u64) -> bool {
        self.boundaries.contains(&address)
    }

    /// Registers one address-bearing entity during module construction.
    ///
    /// A function and one of its own blocks may intentionally share an entry
    /// address; the function remains the indexed target and the block becomes
    /// its root. Every other collision is rejected. Drops the index's
    /// provenance, as [`rehome_block`](Self::rehome_block) does; the tracked
    /// mutators that call this restore it once context and index agree again.
    pub fn register(
        &mut self,
        ctx: &mut Context<'_>,
        address: u64,
        target: AddressTarget,
    ) -> Result<()> {
        self.provenance = None;
        self.last = None;
        let existing = match self.targets.entry(address) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(target);
                self.last = Some((address, target));
                return Ok(());
            }
            std::collections::hash_map::Entry::Occupied(slot) => *slot.get(),
        };
        if existing == target {
            self.last = Some((address, target));
            return Ok(());
        }

        match (existing, target) {
            (AddressTarget::Function(function), AddressTarget::Block(block))
            | (AddressTarget::Block(block), AddressTarget::Function(function)) => {
                // A split first registers a rootless function over a block that
                // is still stored in the old function. Root only once storage
                // ownership matches; rehoming refreshes the index afterwards.
                if block.func == function {
                    FunctionBody::from_id_mut(ctx, function).ensure_root(block)?;
                }
                self.targets
                    .insert(address, AddressTarget::Function(function));
                Ok(())
            }
            _ => Err(Error::spanless(ErrorTy::DuplicateAddress(
                address,
                existing.into(),
            ))),
        }
    }

    /// Returns the live target registered at `address` in this snapshot.
    pub fn get(&self, address: u64) -> Option<AddressTarget> {
        match self.last {
            Some((last, target)) if last == address => Some(target),
            _ => self.targets.get(&address).copied(),
        }
    }

    /// Returns the function registered at `address`, if that is the target kind.
    pub fn function_at(&self, address: u64) -> Option<FunctionId> {
        match self.get(address) {
            Some(AddressTarget::Function(id)) => Some(id),
            _ => None,
        }
    }

    /// Returns the block registered at `address`, if that is the target kind.
    pub fn block_at(&self, address: u64) -> Option<BlockId> {
        match self.get(address) {
            Some(AddressTarget::Block(id)) => Some(id),
            _ => None,
        }
    }

    /// Returns the number of distinct indexed addresses.
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// Returns whether the snapshot contains no addresses.
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}

impl From<AddressTarget> for ValueId {
    fn from(target: AddressTarget) -> Self {
        match target {
            AddressTarget::Function(id) => id.into(),
            AddressTarget::Block(id) => id.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::QCodeMut;
    use crate::value::{BasicBlock, FunctionBody};

    #[test]
    fn function_wins_its_entry_block_collision() {
        let mut ctx = Context::new();
        let mut index = AddressIndex::analyze(&ctx);
        let function = FunctionBody::make_at_addr_indexed(&mut ctx, &mut index, 0x1000, None).id;
        let root = BasicBlock::make(&mut ctx, function)
            .with_address_indexed(&mut index, 0x1000)
            .id;

        assert_eq!(index.get(0x1000), Some(AddressTarget::Function(function)));
        assert_eq!(index.function_at(0x1000), Some(function));
        assert_eq!(index.block_at(0x1000), None);
        assert_eq!(ctx.function(function).root_id(), Some(root.local));
    }

    #[test]
    fn indexes_primary_and_extra_block_addresses() {
        let mut ctx = Context::new();
        let mut index = AddressIndex::analyze(&ctx);
        let function = ctx.anon_function();
        let block = BasicBlock::make(&mut ctx, function)
            .with_address_indexed(&mut index, 0x2000)
            .id;
        ctx.block_mut(block)
            .extra_addresses
            .extend([0x2001, 0x2002]);

        index.refresh(&ctx);
        assert_eq!(index.block_at(0x2000), Some(block));
        assert_eq!(index.block_at(0x2001), Some(block));
        assert_eq!(index.block_at(0x2002), Some(block));
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn excludes_deleted_body_shape() {
        let mut ctx = Context::new();
        let mut index = AddressIndex::analyze(&ctx);
        let function = ctx.anon_function();
        let block = BasicBlock::make(&mut ctx, function)
            .with_address_indexed(&mut index, 0x3000)
            .id;
        ctx.delete_block(block);

        index.refresh(&ctx);
        assert_eq!(index.get(0x3000), None);
        assert!(index.is_empty());
    }
}
