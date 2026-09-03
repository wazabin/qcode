//! Disposable address lookup over an immutable qcode module snapshot.
//!
//! [`AddressIndex`] is derived state: consumers build it for a [`Context`], keep
//! it only while that context remains structurally unchanged, and rebuild it
//! after mutation. It is intentionally not stored in or serialized with the IR.

use rustc_hash::FxHashMap;

use crate::{
    context::Context,
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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AddressIndex {
    targets: FxHashMap<u64, AddressTarget>,
    /// Addresses known to start a block because something branches to them.
    ///
    /// Learned, not derived: an address becomes a boundary the first time a
    /// block has to be split at it. Remembering that is what stops a run from
    /// being folded across the same address again on the next pass, which for a
    /// loop header — the branch target that is discovered *after* the run
    /// through it — would otherwise repeat forever.
    boundaries: rustc_hash::FxHashSet<u64>,
}

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
            boundaries: rustc_hash::FxHashSet::default(),
        }
    }

    /// Recomputes this index after a structural mutation that changes several
    /// addresses at once (for example function splitting or block rehoming).
    pub fn refresh(&mut self, ctx: &Context<'_>) {
        *self = Self::analyze(ctx);
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
    pub fn rehome_block(&mut self, addr: u64, old: BlockId, new: BlockId) {
        if self.targets.get(&addr) == Some(&AddressTarget::Block(old)) {
            self.targets.insert(addr, AddressTarget::Block(new));
        }
    }

    /// Drops `address` from the index, so it resolves to nothing until it is
    /// registered again. Used when a block stops covering an address.
    pub fn forget(&mut self, address: u64) {
        self.targets.remove(&address);
    }

    /// Points `address` at `block`, whatever it pointed at before.
    ///
    /// For a caller that has just made `block` cover an address another block
    /// used to — absorbing that block, typically, which leaves the index
    /// naming something deleted.
    pub fn set_block(&mut self, address: u64, block: BlockId) {
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
    /// its root. Every other collision is rejected.
    pub fn register(
        &mut self,
        ctx: &mut Context<'_>,
        address: u64,
        target: AddressTarget,
    ) -> Result<()> {
        let Some(existing) = self.targets.get(&address).copied() else {
            self.targets.insert(address, target);
            return Ok(());
        };
        if existing == target {
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
        self.targets.get(&address).copied()
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
