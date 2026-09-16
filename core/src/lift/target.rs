//! The checked destination of instruction lowering.
//!
//! An emitter never touches a [`Context`] and an [`AddressIndex`] as two
//! loose references. It binds them, with the function it lowers into, as a
//! [`LiftTarget`], and lowers each instruction inside a [`Construction`] the
//! target opens for it. The target answers the module-level questions an
//! instruction raises — which block stands at an address, which function a
//! call reaches — with typed errors where the raw indexed API asserts, and
//! the construction journals what those answers created so that a failed
//! instruction leaves nothing behind.
//!
//! # What a failed construction undoes
//!
//! Dropping a construction without [committing](Construction::commit) it, or
//! aborting it, takes back every block, instruction, edge and temporary the
//! instruction added to the function, every placeholder block it registered
//! for a branch target or fall-through, and the entry block's contents if the
//! entry existed before. Callees are only created at commit, so a failed
//! instruction never leaves a function behind either. One thing stays:
//! constants interned in the module's shared arenas, which are immutable and
//! unowned, so a spare one is invisible.
//!
//! # Splitting is deferred to commit
//!
//! Resolving a branch target that falls *interior* to an already-lifted block
//! means splitting that block, and a split discards its optimized instructions
//! (`split_block_at_address` cannot recover a faithful per-address boundary).
//! That would mutate state older than the construction, which no rollback can
//! put back — so the construction does not split while it may still fail. It
//! hands the emitter a fresh, address-less tail block to branch to and records
//! the split as pending; the split lands at commit, after every fallible check
//! has passed. An abandoned construction deletes the tail with its other blocks
//! and the pre-existing block is untouched.
//!
//! # Poison
//!
//! If a rollback finds state it cannot account for — an instruction, edge or
//! parameter above its marks that it did not delete, which means the emitter
//! wrote somewhere the journal does not cover — it does not pretend to have
//! rolled back. It **poisons the context** ([`Context::poison`]), and from then
//! on no target binds to that context and no construction begins in it, until
//! the context is disposed of. The flag lives on the context rather than on
//! the target because a target is a per-instruction guard: consumers rebind
//! for every instruction, so a flag on the guard would vanish exactly when the
//! failed lift returned.

use std::fmt;

use crate::{
    address_index::{AddressIndex, AddressTarget},
    builder::Builder,
    context::Context,
    lift::{CallTarget, ExitKind, Lifted},
    value::{BasicBlock, BlockId, FunctionBody, FunctionId, LocalBlockId, insn::Callee},
};

/// Why a target could not be bound, or a construction not carried out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetError {
    /// The function is not in the context.
    UnknownFunction(FunctionId),
    /// The function is external: it has no body to lower into.
    ExternalFunction(FunctionId),
    /// The index names something at `address` that the context does not have
    /// there. The index was built for another context, or the context was
    /// mutated behind it.
    StaleIndex { address: u64 },
    /// `address` is a block of another function, and a block belongs to one
    /// function's arena.
    ForeignBlock { address: u64, owner: FunctionId },
    /// `address` is the entry of another function.
    OwnedByFunction { address: u64, owner: FunctionId },
    /// An instruction was already lowered at `address`. A block is lifted
    /// once; it is not appended to.
    AlreadyLifted { address: u64, block: BlockId },
    /// A call recorded in the result names no callee the construction minted,
    /// so the emitted instruction still holds a placeholder callee.
    UnresolvedCallee { address: u64 },
    /// A previous rollback could not restore the context, which is poisoned
    /// (see [`Context::is_poisoned`]); nothing further is lifted into it.
    Poisoned,
}

impl fmt::Display for TargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownFunction(function) => write!(f, "{function:?} is not in the context"),
            Self::ExternalFunction(function) => {
                write!(f, "{function:?} is external and has no body")
            }
            Self::StaleIndex { address } => {
                write!(f, "the address index is stale at {address:#x}")
            }
            Self::ForeignBlock { address, owner } => {
                write!(f, "{address:#x} is a block of another function, {owner:?}")
            }
            Self::OwnedByFunction { address, owner } => {
                write!(
                    f,
                    "{address:#x} is the entry of another function, {owner:?}"
                )
            }
            Self::AlreadyLifted { address, block } => {
                write!(f, "{address:#x} was already lifted into {block:?}")
            }
            Self::UnresolvedCallee { address } => {
                write!(f, "the call to {address:#x} kept its placeholder callee")
            }
            Self::Poisoned => {
                f.write_str("the context was poisoned by a failed rollback and refuses lifting")
            }
        }
    }
}

impl std::error::Error for TargetError {}

/// A context, its address index and a host function, bound together for a
/// batch of instruction constructions.
///
/// See the [module documentation](self) for what a target guarantees.
pub struct LiftTarget<'a, 'str> {
    ctx: &'a mut Context<'str>,
    addresses: &'a mut AddressIndex,
    function: FunctionId,
}

impl<'a, 'str> LiftTarget<'a, 'str> {
    /// Binds `addresses` after rebuilding it from `ctx`, so it is complete and
    /// current, and selects `function` as the host.
    ///
    /// This is the safe binding: because the index is rebuilt here, an address
    /// the context already covers is always found, so no construction creates a
    /// second block or function at an address that already has one. Use it
    /// wherever the caller cannot cheaply prove its index is current — a fresh
    /// analysis, a Python wrapper, a one-off lift.
    ///
    /// The rebuild is O(module). A caller that lifts in a hot loop and keeps
    /// its index current itself uses [`bind_indexed`](Self::bind_indexed).
    pub fn bind(
        ctx: &'a mut Context<'str>,
        addresses: &'a mut AddressIndex,
        function: FunctionId,
    ) -> Result<Self, TargetError> {
        addresses.refresh(ctx);
        Self::bind_indexed(ctx, addresses, function)
    }

    /// Binds a caller-maintained `addresses` without rebuilding it, and selects
    /// `function` as the host.
    ///
    /// This is the advanced, hot-path binding for a caller that owns its index
    /// across every mutation and keeps it current — an emulator that lifts on
    /// demand, or a session that threads one index through a run. **The index
    /// must reflect every address-bearing entity in `ctx`.** The target checks
    /// each entry it acts on against the context and refuses a stale one
    /// ([`TargetError::StaleIndex`]), but it cannot detect an entity the index
    /// *omits*: an address the index does not list is taken to be free, and a
    /// construction there would make a second block or function beside the one
    /// already in the context. Where that guarantee is not cheap to uphold, use
    /// [`bind`](Self::bind), which rebuilds the index first. Establishing
    /// provenance without a rebuild needs context/index revision tracking,
    /// which does not exist yet.
    ///
    /// A [poisoned](Context::is_poisoned) context is refused outright.
    pub fn bind_indexed(
        ctx: &'a mut Context<'str>,
        addresses: &'a mut AddressIndex,
        function: FunctionId,
    ) -> Result<Self, TargetError> {
        if ctx.is_poisoned() {
            return Err(TargetError::Poisoned);
        }
        if usize::from(function) >= ctx.bodies.len() {
            return Err(TargetError::UnknownFunction(function));
        }
        if ctx.interfaces[function].is_external {
            return Err(TargetError::ExternalFunction(function));
        }
        Ok(Self {
            ctx,
            addresses,
            function,
        })
    }

    /// The function instructions are lowered into.
    pub fn function(&self) -> FunctionId {
        self.function
    }

    pub fn context(&self) -> &Context<'str> {
        self.ctx
    }

    pub fn addresses(&self) -> &AddressIndex {
        self.addresses
    }

    /// Whether a failed rollback has poisoned the context this target is
    /// bound to; see [`Context::is_poisoned`].
    pub fn is_poisoned(&self) -> bool {
        self.ctx.is_poisoned()
    }

    /// Opens the construction of the instruction at `address`.
    ///
    /// Its entry block is the block registered at `address`, or the host's
    /// root when `address` is the host's entry, or a fresh block; in every
    /// case it must still be empty.
    pub fn begin(
        &mut self,
        address: u64,
        length: usize,
    ) -> Result<Construction<'_, 'a, 'str>, TargetError> {
        if self.ctx.is_poisoned() {
            return Err(TargetError::Poisoned);
        }
        let body = &self.ctx.bodies[self.function];
        let mut journal = Journal {
            address,
            length,
            entry: BlockId::new(self.function, LocalBlockId::default()),
            entry_created: false,
            placeholders: Vec::new(),
            splits: Vec::new(),
            minted: Vec::new(),
            insns: body.insns.issued_len(),
            blocks: body.blocks.issued_len(),
            params: body.params.issued_len(),
            edges: body.edges.issued_len(),
            temps: body.temps.len(),
            temp_spaces: body.temp_spaces.len(),
        };
        let entry = match self.resolve_block(address)? {
            Resolved::Existing(entry) => {
                if self.ctx.bodies[self.function].block(entry).has_insns() {
                    return Err(TargetError::AlreadyLifted {
                        address,
                        block: entry,
                    });
                }
                entry
            }
            Resolved::Free => {
                journal.entry_created = true;
                self.make_block(address)
            }
            // The entry is interior to a lifted block: it becomes the tail of
            // a split that lands at commit. Until then the tail is an ordinary
            // fresh block of the host, so a failure just deletes it.
            Resolved::Interior(block) => {
                let tail = BasicBlock::make(self.ctx, self.function).id;
                journal.splits.push(PendingSplit {
                    block,
                    address,
                    tail,
                });
                tail
            }
        };
        journal.entry = entry;
        FunctionBody::from_id_mut(self.ctx, self.function).add_block(entry);
        Ok(Construction {
            target: self,
            journal,
            settled: false,
        })
    }

    /// What the host has at `address`, without changing anything: an existing
    /// block that starts there, a lifted block it is interior to, or nothing.
    fn resolve_block(&self, address: u64) -> Result<Resolved, TargetError> {
        let function = self.function;
        match self.addresses.get(address) {
            Some(AddressTarget::Function(owner)) => {
                let live = usize::from(owner) < self.ctx.bodies.len()
                    && self.ctx.interfaces[owner].address == Some(address);
                if !live {
                    return Err(TargetError::StaleIndex { address });
                }
                if owner != function {
                    return Err(TargetError::OwnedByFunction { address, owner });
                }
                Ok(match self.ctx.bodies[function].root_id() {
                    Some(root) => Resolved::Existing(BlockId::new(function, root)),
                    None => Resolved::Free,
                })
            }
            Some(AddressTarget::Block(block)) => {
                let live = usize::from(block.func) < self.ctx.bodies.len()
                    && self.ctx.bodies[block.func].blocks.contains(block.local)
                    && {
                        let stored = &self.ctx.bodies[block.func].blocks[block.local];
                        stored.address == Some(address) || stored.extra_addresses.contains(&address)
                    };
                if !live {
                    return Err(TargetError::StaleIndex { address });
                }
                if block.func != function {
                    return Err(TargetError::ForeignBlock {
                        address,
                        owner: block.func,
                    });
                }
                if self.ctx.bodies[function].blocks[block.local].address != Some(address) {
                    // The address is interior to a run that was folded into
                    // one block; something branches there after all.
                    return Ok(Resolved::Interior(block));
                }
                Ok(Resolved::Existing(block))
            }
            None => Ok(Resolved::Free),
        }
    }

    /// Makes a block of the host at `address` and registers it.
    fn make_block(&mut self, address: u64) -> BlockId {
        BasicBlock::make(self.ctx, self.function)
            .with_address_indexed(self.addresses, address)
            .id
    }
}

/// What [`resolve_block`](LiftTarget::resolve_block) found at an address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Resolved {
    /// A block that was already there and starts at the address.
    Existing(BlockId),
    /// Nothing; a construction may make a block there.
    Free,
    /// The address is interior to this lifted block, which absorbed it. A
    /// construction that needs a block there gets a fresh tail and splits the
    /// block at commit (see the [module documentation](self)).
    Interior(BlockId),
}

/// A placeholder block the construction made for an address.
#[derive(Debug, Clone, Copy)]
struct Placeholder {
    address: u64,
    block: BlockId,
}

/// A split of `block` at its interior `address`, promised for commit, whose
/// new half is `tail`: a fresh block of the host with no address yet.
#[derive(Debug, Clone, Copy)]
struct PendingSplit {
    block: BlockId,
    address: u64,
    tail: BlockId,
}

/// What one construction added, for taking it back.
#[derive(Debug)]
struct Journal {
    address: u64,
    length: usize,
    entry: BlockId,
    /// The entry was made by this construction. Otherwise it was an empty
    /// block that existed before, whose contents alone are the construction's.
    entry_created: bool,
    placeholders: Vec<Placeholder>,
    /// Splits of lifted blocks this construction branches into, performed at
    /// commit. Their tails were issued after the block mark, so a rollback
    /// deletes them like any other block and the split never happens.
    splits: Vec<PendingSplit>,
    /// Callees the construction promised, by minted slot, to create at commit.
    minted: Vec<u64>,
    // The body's issued lengths when the construction began.
    insns: usize,
    blocks: usize,
    params: usize,
    edges: usize,
    temps: usize,
    temp_spaces: usize,
}

/// The lowering of one instruction into a [`LiftTarget`].
///
/// Resolve the instruction's module-level targets first, then take a
/// [`builder`](Self::builder) for its entry and emit. [`commit`](Self::commit)
/// keeps the result; dropping the construction before that undoes it.
pub struct Construction<'t, 'a, 'str> {
    target: &'t mut LiftTarget<'a, 'str>,
    journal: Journal,
    settled: bool,
}

impl<'t, 'a, 'str> Construction<'t, 'a, 'str> {
    /// The address being lifted.
    pub fn address(&self) -> u64 {
        self.journal.address
    }

    /// The instruction's encoded length.
    pub fn length(&self) -> usize {
        self.journal.length
    }

    /// The block the instruction's p-code starts in.
    pub fn entry(&self) -> BlockId {
        self.journal.entry
    }

    pub fn function(&self) -> FunctionId {
        self.target.function
    }

    pub fn context(&self) -> &Context<'str> {
        self.target.ctx
    }

    /// The block of the host at `address`, for a branch target or the
    /// fall-through, made as a placeholder if none is there.
    ///
    /// An address interior to a lifted block gets the tail of a split that
    /// happens at commit; asking again for the same address gets the same
    /// tail.
    pub fn block_at(&mut self, address: u64) -> Result<BlockId, TargetError> {
        if let Some(split) = self.journal.splits.iter().find(|s| s.address == address) {
            return Ok(split.tail);
        }
        match self.target.resolve_block(address)? {
            Resolved::Existing(block) => Ok(block),
            Resolved::Free => {
                let block = self.target.make_block(address);
                self.journal
                    .placeholders
                    .push(Placeholder { address, block });
                Ok(block)
            }
            Resolved::Interior(block) => {
                let tail = BasicBlock::make(self.target.ctx, self.target.function).id;
                self.journal.splits.push(PendingSplit {
                    block,
                    address,
                    tail,
                });
                Ok(tail)
            }
        }
    }

    /// The function a direct call to `address` reaches.
    ///
    /// An existing function is named outright. One that does not exist yet
    /// is promised as a minted placeholder and created when the construction
    /// commits, so a failed instruction creates no function.
    pub fn callee_at(&mut self, address: u64) -> Result<Callee, TargetError> {
        if let Some(function) = self.target.addresses.function_at(address) {
            let live = usize::from(function) < self.target.ctx.bodies.len()
                && self.target.ctx.interfaces[function].address == Some(address);
            if !live {
                return Err(TargetError::StaleIndex { address });
            }
            return Ok(Callee::Real(function));
        }
        let slot = match self.journal.minted.iter().position(|&a| a == address) {
            Some(slot) => slot,
            None => {
                self.journal.minted.push(address);
                self.journal.minted.len() - 1
            }
        };
        Ok(Callee::Minted(slot as u32))
    }

    /// A builder positioned at `block`, which must be one of the host's.
    pub fn builder(&mut self, block: BlockId) -> Builder<'str, '_> {
        debug_assert_eq!(block.func, self.target.function);
        self.target.ctx.builder(block)
    }

    /// Keeps the instruction: creates the callees it promised and settles
    /// their call sites.
    pub fn commit(mut self, lifted: Lifted) -> Result<Lifted, TargetError> {
        let function = self.target.function;
        // Every promised callee must have a call site the result reports, and
        // every placeholder the instruction holds must be one of the promised
        // slots. Checked before anything is created: a function, once made,
        // cannot be taken back.
        let mut sites: Vec<Vec<crate::value::LocalInsnId>> =
            vec![Vec::new(); self.journal.minted.len()];
        for exit in lifted.exits() {
            let ExitKind::Call {
                callee: CallTarget::Address(address),
                ..
            } = exit.kind()
            else {
                continue;
            };
            if exit.site().func != function {
                continue;
            }
            let body = &self.target.ctx.bodies[function];
            let slot = body.insns[exit.site().local]
                .mnemonic()
                .minted_callee_slot();
            if let Some(slot) = slot
                && self.journal.minted.get(slot as usize) == Some(address)
            {
                sites[slot as usize].push(exit.site().local);
            }
        }
        if let Some(slot) = sites.iter().position(Vec::is_empty) {
            let address = self.journal.minted[slot];
            self.rollback();
            return Err(TargetError::UnresolvedCallee { address });
        }
        let body = &self.target.ctx.bodies[function];
        for raw in self.journal.insns..body.insns.issued_len() {
            let local = raw.into();
            if !body.insns.contains(local) {
                continue;
            }
            let Some(slot) = body.insns[local].mnemonic().minted_callee_slot() else {
                continue;
            };
            if !sites
                .get(slot as usize)
                .is_some_and(|sites| sites.contains(&local))
            {
                let address = self.journal.minted.get(slot as usize).copied();
                self.rollback();
                return Err(TargetError::UnresolvedCallee {
                    address: address.unwrap_or(self.journal.address),
                });
            }
        }

        // Every check that can fail has passed: land the splits, then the
        // callees. Neither can fail, so from here the instruction is committed.
        for split in std::mem::take(&mut self.journal.splits) {
            self.target.ctx.split_block_into(
                self.target.addresses,
                split.block,
                split.address,
                split.tail,
            );
        }
        for (slot, sites) in sites.into_iter().enumerate() {
            let address = self.journal.minted[slot];
            let real = FunctionBody::from_addr_or_create_indexed(
                self.target.ctx,
                self.target.addresses,
                address,
            )
            .id;
            let body = &mut self.target.ctx.bodies[function];
            for site in sites {
                body.insns[site]
                    .mnemonic_mut()
                    .resolve_minted_callee(slot as u32, real);
            }
        }
        self.settled = true;
        Ok(lifted)
    }

    /// Discards the instruction, undoing everything it added.
    pub fn abort(mut self) {
        self.rollback();
    }

    fn rollback(&mut self) {
        if self.settled {
            return;
        }
        self.settled = true;
        let function = self.target.function;
        let ctx = &mut *self.target.ctx;
        let addresses = &mut *self.target.addresses;

        // The entry keeps its identity when it existed before: branches from
        // other instructions already name it. Clearing it takes back this
        // construction's emissions and leaves the empty placeholder behind. (An
        // entry that is a pending split's tail is cleared here and deleted
        // below, like any block issued after the mark.)
        if !self.journal.entry_created {
            ctx.bodies[function].clear_block_instructions(self.journal.entry);
        }
        // Every other block the construction made — its own, the branch/
        // fall-through placeholders and the tails of pending splits — was
        // issued after the mark. Deleting them drops their instructions, edges
        // and names; a pending split simply never happens.
        let body = &mut ctx.bodies[function];
        for raw in (self.journal.blocks..body.blocks.issued_len()).rev() {
            let local: LocalBlockId = raw.into();
            if body.blocks.contains(local) {
                body.delete_block(BlockId::new(function, local));
            }
        }
        for placeholder in &self.journal.placeholders {
            if addresses.block_at(placeholder.address) == Some(placeholder.block) {
                addresses.forget(placeholder.address);
            }
        }
        if self.journal.entry_created
            && addresses.block_at(self.journal.address) == Some(self.journal.entry)
        {
            addresses.forget(self.journal.address);
        }
        body.take_back_temps(self.journal.temps, self.journal.temp_spaces);

        // What the mark said is what must be left; anything else was added
        // somewhere the journal does not know about — the emitter wrote into a
        // block it did not own — and cannot be taken back. Reclaim the ids
        // only when the function is exactly as it was; otherwise poison the
        // context (see the module documentation).
        let intact = (self.journal.insns..body.insns.issued_len())
            .all(|raw| !body.insns.contains(raw.into()))
            && (self.journal.params..body.params.issued_len())
                .all(|raw| !body.params.contains(raw.into()))
            && (self.journal.edges..body.edges.issued_len())
                .all(|raw| !body.edges.contains(raw.into()))
            && (self.journal.blocks..body.blocks.issued_len())
                .all(|raw| !body.blocks.contains(raw.into()));
        if intact {
            body.insns.truncate_issued(self.journal.insns);
            body.params.truncate_issued(self.journal.params);
            body.edges.truncate_issued(self.journal.edges);
            body.blocks.truncate_issued(self.journal.blocks);
        } else {
            ctx.poison();
        }
    }
}

impl Drop for Construction<'_, '_, '_> {
    fn drop(&mut self) {
        self.rollback();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        lift::{Continuation, Exit, ExitArm, Recorder},
        value::{FunctionBody, QCodeMut, ValueId},
    };

    fn host(ctx: &mut Context<'static>, addresses: &mut AddressIndex, address: u64) -> FunctionId {
        FunctionBody::make_at_addr_indexed(ctx, addresses, address, None).id
    }

    /// Builds one straight-line instruction that falls through, as an emitter
    /// would, and returns its record.
    fn emit_fallthrough(construction: &mut Construction<'_, '_, '_>) -> Lifted {
        let entry = construction.entry();
        let next = construction.block_at(construction.address() + 1).unwrap();
        let mut recorder = Recorder::new(construction.address(), 1, entry);
        let mut builder = construction.builder(entry);
        let temp = builder.make_temp(8);
        let one = builder.shr().get_const(1, 8);
        builder.push_copy(one, ValueId::Temp(temp));
        let site = builder.push_branch(next).id;
        recorder.exit(site, ExitArm::Unconditional, ExitKind::Fallthrough);
        recorder.finish()
    }

    #[test]
    fn binding_checks_the_function() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let stranger = FunctionId::from(7usize);
        assert_eq!(
            LiftTarget::bind_indexed(&mut ctx, &mut addresses, stranger).err(),
            Some(TargetError::UnknownFunction(stranger))
        );
        let external = FunctionBody::make_external_indexed(&mut ctx, &mut addresses, 0x50, None).id;
        assert_eq!(
            LiftTarget::bind_indexed(&mut ctx, &mut addresses, external).err(),
            Some(TargetError::ExternalFunction(external))
        );
    }

    #[test]
    fn a_committed_instruction_stays_and_the_next_is_refused() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = host(&mut ctx, &mut addresses, 0x1000);
        let entry = {
            let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();

            let mut construction = target.begin(0x1000, 1).unwrap();
            let lifted = emit_fallthrough(&mut construction);
            let entry = construction.entry();
            construction.commit(lifted).unwrap();

            assert_eq!(
                target.begin(0x1000, 1).err(),
                Some(TargetError::AlreadyLifted {
                    address: 0x1000,
                    block: entry
                })
            );
            entry
        };
        // The entry is the host's root, and the fall-through placeholder is
        // waiting at the next address.
        assert_eq!(ctx.function(function).root_id(), Some(entry.local));
        assert_eq!(ctx.block(entry).insn_count(), 2);
        let next = addresses.block_at(0x1001).unwrap();
        assert!(!ctx.block(next).has_insns());
    }

    #[test]
    fn a_dropped_construction_leaves_nothing_behind() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = host(&mut ctx, &mut addresses, 0x1000);
        // An existing, empty entry: a placeholder another instruction made.
        let entry = BasicBlock::make(&mut ctx, function)
            .with_address_indexed(&mut addresses, 0x1000)
            .id;
        let before = (ctx.clone(), addresses.clone());

        {
            let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
            {
                let mut construction = target.begin(0x1000, 1).unwrap();
                assert_eq!(construction.entry(), entry);
                let _ = emit_fallthrough(&mut construction);
                // A callee that does not exist is only promised.
                assert_eq!(construction.callee_at(0x2000).unwrap(), Callee::Minted(0));
                assert_eq!(construction.context().bodies.len(), 1);
            }
            assert!(!target.is_poisoned());
        }

        assert_eq!(addresses, before.1);
        assert_eq!(ctx.to_string(), before.0.to_string());
        let body = ctx.body(function);
        assert_eq!(body.insns.issued_len(), 0);
        assert_eq!(body.blocks.issued_len(), 1);
        assert_eq!(body.temps.len(), 0);
        assert!(body.users.is_empty());
        assert_eq!(ctx.bodies.len(), 1, "no callee was created");

        // And the address lifts afterwards.
        {
            let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
            let mut construction = target.begin(0x1000, 1).unwrap();
            let lifted = emit_fallthrough(&mut construction);
            construction.commit(lifted).unwrap();
        }
        assert_eq!(ctx.block(entry).insn_count(), 2);
    }

    #[test]
    fn a_promised_callee_is_created_at_commit() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = host(&mut ctx, &mut addresses, 0x1000);
        let site = {
            let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();

            let mut construction = target.begin(0x1000, 5).unwrap();
            let entry = construction.entry();
            let callee = construction.callee_at(0x2000).unwrap();
            assert_eq!(construction.callee_at(0x2000).unwrap(), callee);
            let mut recorder = Recorder::new(0x1000, 5, entry);
            let site = construction.builder(entry).push_call(callee).id;
            recorder.exit(
                site,
                ExitArm::Unconditional,
                ExitKind::Call {
                    callee: CallTarget::Address(0x2000),
                    continuation: Continuation::Next,
                },
            );
            construction.commit(recorder.finish()).unwrap();
            site
        };

        let callee = addresses.function_at(0x2000).expect("created at commit");
        let call = crate::value::Instruction::from_id(&ctx, site);
        assert_eq!(
            call.mnemonic().minted_callee_slot(),
            None,
            "the site was settled"
        );
        assert!(matches!(
            call.mnemonic(),
            crate::value::insn::Mnemonic::Call(call) if call.target == Callee::Real(callee)
        ));
    }

    #[test]
    fn a_call_the_result_does_not_report_is_refused_and_undone() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = host(&mut ctx, &mut addresses, 0x1000);
        {
            let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();

            let mut construction = target.begin(0x1000, 5).unwrap();
            let entry = construction.entry();
            let callee = construction.callee_at(0x2000).unwrap();
            construction.builder(entry).push_call(callee);
            let lifted = Recorder::new(0x1000, 5, entry).finish();
            assert_eq!(
                construction.commit(lifted).err(),
                Some(TargetError::UnresolvedCallee { address: 0x2000 })
            );
            assert!(!target.is_poisoned());
        }
        assert_eq!(addresses.function_at(0x2000), None);
        let root = ctx.function(function).root_id();
        assert!(root.is_none_or(|root| !ctx.block(BlockId::new(function, root)).has_insns()));
    }

    /// A function whose root block absorbed a straight-line run: it starts
    /// at 0x1000 and also covers the interior address 0x1004, and holds code.
    fn absorbed_run(
        ctx: &mut Context<'static>,
        addresses: &mut AddressIndex,
    ) -> (FunctionId, BlockId) {
        let function = FunctionBody::make_at_addr_indexed(ctx, addresses, 0x1000, None).id;
        let root = BasicBlock::make(ctx, function)
            .with_address_indexed(addresses, 0x1000)
            .id;
        ctx.block_mut(root).extra_addresses.push(0x1004);
        let zero = ctx.shared.get_const(0, 8);
        ctx.builder(root).push_branchind(zero);
        addresses.refresh(ctx);
        assert_eq!(addresses.block_at(0x1004), Some(root));
        (function, root)
    }

    #[test]
    fn a_split_lands_at_commit_and_an_abandoned_one_never_happens() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let (function, root) = absorbed_run(&mut ctx, &mut addresses);
        let before = (ctx.to_string(), addresses.clone());

        // An instruction at 0x2000 whose branch lands interior to root, then
        // abandoned: the pre-existing block is exactly as it was.
        {
            let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
            let mut construction = target.begin(0x2000, 2).unwrap();
            let entry = construction.entry();
            let tail = construction.block_at(0x1004).unwrap();
            assert_ne!(tail, root);
            assert_eq!(
                construction.block_at(0x1004).unwrap(),
                tail,
                "one tail per address"
            );
            assert_eq!(
                construction.context().block(tail).address,
                None,
                "not split yet"
            );
            construction.builder(entry).push_branch(tail);
            construction.abort();
            assert!(!target.is_poisoned());
        }
        assert_eq!(ctx.to_string(), before.0);
        assert_eq!(addresses, before.1);
        assert!(ctx.block(root).has_insns(), "the run was not split");
        assert!(!ctx.is_poisoned());

        // The same instruction committed: now the split lands.
        let tail = {
            let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
            let mut construction = target.begin(0x2000, 2).unwrap();
            let entry = construction.entry();
            let tail = construction.block_at(0x1004).unwrap();
            let mut recorder = Recorder::new(0x2000, 2, entry);
            let site = construction.builder(entry).push_branch(tail).id;
            recorder.exit(
                site,
                ExitArm::Unconditional,
                ExitKind::Branch { target: 0x1004 },
            );
            construction.commit(recorder.finish()).unwrap();
            tail
        };
        assert_eq!(addresses.block_at(0x1004), Some(tail));
        assert_eq!(ctx.block(tail).address, Some(0x1004));
        assert!(!ctx.block(tail).has_insns());
        assert!(
            !ctx.block(root).has_insns(),
            "the split emptied the original"
        );
        assert!(ctx.block(root).extra_addresses.is_empty());

        // Both halves are re-liftable, the tail as the entry of its address.
        let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
        let mut construction = target.begin(0x1004, 1).unwrap();
        assert_eq!(construction.entry(), tail);
        let entry = construction.entry();
        let mut recorder = Recorder::new(0x1004, 1, entry);
        let zero = construction.context().shared.get_const(0, 8);
        let site = construction.builder(entry).push_branchind(zero).id;
        recorder.exit(site, ExitArm::Unconditional, ExitKind::BranchInd);
        construction.commit(recorder.finish()).unwrap();
        assert!(ctx.block(tail).has_insns());
    }

    #[test]
    fn an_entry_interior_to_a_run_is_split_at_commit() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let (function, root) = absorbed_run(&mut ctx, &mut addresses);

        // Lifting 0x1004 itself while root still covers it: the entry is the
        // tail of a split. Abandoned, nothing changes.
        {
            let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
            let construction = target.begin(0x1004, 1).unwrap();
            assert_ne!(construction.entry(), root);
            drop(construction);
        }
        assert!(ctx.block(root).has_insns());
        assert_eq!(addresses.block_at(0x1004), Some(root));

        let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
        let mut construction = target.begin(0x1004, 1).unwrap();
        let entry = construction.entry();
        // A self-jump resolves to the very block being built.
        assert_eq!(construction.block_at(0x1004).unwrap(), entry);
        let mut recorder = Recorder::new(0x1004, 1, entry);
        let site = construction.builder(entry).push_branch(entry).id;
        recorder.exit(
            site,
            ExitArm::Unconditional,
            ExitKind::Branch { target: 0x1004 },
        );
        construction.commit(recorder.finish()).unwrap();
        assert_eq!(addresses.block_at(0x1004), Some(entry));
        assert!(ctx.block(entry).has_insns());
        assert!(!ctx.block(root).has_insns());
    }

    #[test]
    fn an_unaccountable_rollback_poisons_the_context_for_every_later_binding() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = host(&mut ctx, &mut addresses, 0x1000);
        // A placeholder an earlier instruction left for its fall-through.
        let other = BasicBlock::make(&mut ctx, function)
            .with_address_indexed(&mut addresses, 0x1010)
            .id;

        {
            let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
            let mut construction = target.begin(0x1000, 1).unwrap();
            let _ = emit_fallthrough(&mut construction);
            // The emitter misbehaves: it writes into a block the construction
            // does not own, which the journal cannot take back.
            let zero = construction.context().shared.get_const(0, 8);
            construction.builder(other).push_branchind(zero);
            construction.abort();
            assert!(target.is_poisoned());
            assert_eq!(target.begin(0x1020, 1).err(), Some(TargetError::Poisoned));
        }

        // The poison outlives the guard: it is the context's.
        assert!(ctx.is_poisoned());
        assert!(
            LiftTarget::bind_indexed(&mut ctx, &mut addresses, function)
                .is_err_and(|e| e == TargetError::Poisoned)
        );
        assert!(
            LiftTarget::bind(&mut ctx, &mut addresses, function)
                .is_err_and(|e| e == TargetError::Poisoned)
        );
        assert!(ctx.clone().is_poisoned());
        // The module is still readable — the stray instruction is there to see.
        assert!(ctx.block(other).has_insns());
    }

    #[test]
    fn foreign_and_stale_addresses_are_typed_errors() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = host(&mut ctx, &mut addresses, 0x1000);
        let other = host(&mut ctx, &mut addresses, 0x2000);
        let foreign = BasicBlock::make(&mut ctx, other)
            .with_address_indexed(&mut addresses, 0x2010)
            .id;
        // A block the index remembers but the context no longer has.
        let gone = BasicBlock::make(&mut ctx, function)
            .with_address_indexed(&mut addresses, 0x1050)
            .id;
        ctx.delete_block(gone);

        let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
        let mut construction = target.begin(0x1000, 1).unwrap();
        assert_eq!(
            construction.block_at(0x2010).err(),
            Some(TargetError::ForeignBlock {
                address: 0x2010,
                owner: foreign.func
            })
        );
        assert_eq!(
            construction.block_at(0x2000).err(),
            Some(TargetError::OwnedByFunction {
                address: 0x2000,
                owner: other
            })
        );
        assert_eq!(
            construction.block_at(0x1050).err(),
            Some(TargetError::StaleIndex { address: 0x1050 })
        );
        assert_eq!(construction.callee_at(0x2000).unwrap(), Callee::Real(other));
    }

    #[test]
    fn exits_survive_a_commit() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = host(&mut ctx, &mut addresses, 0x1000);
        let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
        let mut construction = target.begin(0x1000, 1).unwrap();
        let lifted = emit_fallthrough(&mut construction);
        let lifted = construction.commit(lifted).unwrap();
        assert!(lifted.falls_through());
        assert_eq!(
            lifted.exits(),
            &[Exit::new(
                lifted.exits()[0].site(),
                ExitArm::Unconditional,
                ExitKind::Fallthrough
            )]
        );
    }
}
