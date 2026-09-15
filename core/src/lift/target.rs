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
//! instruction never leaves a function behind either. Two things stay:
//!
//! - Constants interned in the module's shared arenas. They are immutable
//!   and unowned, so a spare one is invisible.
//! - A block that was split because a branch target fell inside it. Both
//!   halves are already empty placeholders asking to be lifted again, which
//!   is a consistent module whether or not this instruction succeeds.
//!
//! If a rollback finds state it cannot account for, the target is poisoned:
//! every further construction is refused until the target is discarded, so
//! the inconsistency is reported rather than built on.

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
    /// A previous rollback could not restore the function, so the target no
    /// longer vouches for it.
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
            Self::Poisoned => f.write_str("the target was poisoned by a failed rollback"),
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
    poisoned: bool,
}

impl<'a, 'str> LiftTarget<'a, 'str> {
    /// Binds `addresses`, which the caller built for `ctx` and has kept
    /// current, and selects `function` as the host.
    ///
    /// This is the binding for callers that own their index across mutations
    /// the target does not see, such as an emulator that lifts on demand. The
    /// index cannot be proven to match `ctx` here; what the target does is
    /// check every entry it acts on against the context and refuse a stale
    /// one, so a mismatch surfaces as a [`TargetError::StaleIndex`] at the
    /// address it concerns rather than as a block in the wrong place.
    pub fn bind_indexed(
        ctx: &'a mut Context<'str>,
        addresses: &'a mut AddressIndex,
        function: FunctionId,
    ) -> Result<Self, TargetError> {
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
            poisoned: false,
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

    /// Whether a failed rollback has left this target unusable.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
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
        if self.poisoned {
            return Err(TargetError::Poisoned);
        }
        let body = &self.ctx.bodies[self.function];
        let mut journal = Journal {
            address,
            length,
            entry: BlockId::new(self.function, LocalBlockId::default()),
            entry_created: false,
            placeholders: Vec::new(),
            minted: Vec::new(),
            insns: body.insns.issued_len(),
            blocks: body.blocks.issued_len(),
            params: body.params.issued_len(),
            edges: body.edges.issued_len(),
            temps: body.temps.len(),
            temp_spaces: body.temp_spaces.len(),
        };
        let (entry, created) = self.resolve_block(address)?;
        if !created && self.ctx.bodies[self.function].block(entry).has_insns() {
            return Err(TargetError::AlreadyLifted {
                address,
                block: entry,
            });
        }
        journal.entry = entry;
        journal.entry_created = created;
        if let Some(placeholder) = journal.placeholders.pop() {
            debug_assert_eq!(placeholder.block, entry);
        }
        FunctionBody::from_id_mut(self.ctx, self.function).add_block(entry);
        Ok(Construction {
            target: self,
            journal,
            settled: false,
        })
    }

    /// The block of the host at `address`, made if none is there. Returns
    /// whether it was made.
    fn resolve_block(&mut self, address: u64) -> Result<(BlockId, bool), TargetError> {
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
                match self.ctx.bodies[function].root_id() {
                    Some(root) => Ok((BlockId::new(function, root), false)),
                    None => Ok((self.make_block(address), true)),
                }
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
                    let tail = self
                        .ctx
                        .split_block_at_address(self.addresses, block, address);
                    return Ok((tail, false));
                }
                Ok((block, false))
            }
            None => Ok((self.make_block(address), true)),
        }
    }

    /// Makes a block of the host at `address` and registers it.
    fn make_block(&mut self, address: u64) -> BlockId {
        BasicBlock::make(self.ctx, self.function)
            .with_address_indexed(self.addresses, address)
            .id
    }
}

/// A placeholder block the construction made for an address.
#[derive(Debug, Clone, Copy)]
struct Placeholder {
    address: u64,
    block: BlockId,
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
    pub fn block_at(&mut self, address: u64) -> Result<BlockId, TargetError> {
        let (block, created) = self.target.resolve_block(address)?;
        if created {
            self.journal
                .placeholders
                .push(Placeholder { address, block });
        }
        Ok(block)
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
        // other instructions already name it.
        if !self.journal.entry_created {
            ctx.bodies[function].clear_block_instructions(self.journal.entry);
        }
        // Every block the construction made, its own and the placeholders,
        // was issued after the mark. Deleting them drops their instructions,
        // edges and names.
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
        // somewhere the journal does not know about.
        let intact = (self.journal.insns..body.insns.issued_len())
            .all(|raw| !body.insns.contains(raw.into()))
            && (self.journal.params..body.params.issued_len())
                .all(|raw| !body.params.contains(raw.into()))
            && (self.journal.edges..body.edges.issued_len())
                .all(|raw| !body.edges.contains(raw.into()));
        if intact {
            body.insns.truncate_issued(self.journal.insns);
            body.blocks.truncate_issued(self.journal.blocks);
            body.params.truncate_issued(self.journal.params);
            body.edges.truncate_issued(self.journal.edges);
        } else {
            self.target.poisoned = true;
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
