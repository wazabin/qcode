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
//! entry existed before. Callees — by address or by name — are only created at
//! commit, and a foreign block a transfer [promotes](Promotion::SplitFunction)
//! into a function of its own is only split then, so a failed instruction
//! never leaves a function behind either. One thing stays: constants and
//! varnodes interned in the module's shared arenas, which are immutable and
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
//! rolled back. It **poisons the context** ([`Context::is_poisoned`]), and from then
//! on no target binds to that context and no construction begins in it, until
//! the context is disposed of. The flag lives on the context rather than on
//! the target because a target is a per-instruction guard: consumers rebind
//! for every instruction, so a flag on the guard would vanish exactly when the
//! failed lift returned.

use std::borrow::Cow;
use std::fmt;
use std::ops::{Deref, DerefMut};

use rustc_hash::FxHashMap;

use crate::{
    address_index::{AddressIndex, AddressTarget},
    builder::Builder,
    context::Context,
    lift::{CallTarget, ExitArm, ExitKind, Lifted, Recorder},
    value::{
        BasicBlock, BlockId, FunctionBody, FunctionId, InstructionId, LocalBlockId, LocalInsnId,
        insn::{Callee, Mnemonic},
    },
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
    /// A call was promised to `address`, but `block` stands there — a block
    /// lifted earlier, of any function, or a placeholder this very
    /// instruction made for a branch or its fall-through — and a function
    /// cannot be made over a block that is not its own.
    CallIntoBlock { address: u64, block: BlockId },
    /// A previous rollback could not restore the context, which is poisoned
    /// (see [`Context::is_poisoned`]); nothing further is lifted into it.
    Poisoned,
    /// The index was not computed for this context: it describes another
    /// module instance (a clone of this one, or an unrelated module whose ids
    /// happen to match), or it was cleared and never rebuilt.
    ForeignIndex,
    /// The index was computed for this context, but the context's
    /// address-bearing shape has changed since without it, so it may omit an
    /// entity the context has. Refresh it, or bind through
    /// [`LiftTarget::bind_or_refresh`], which does.
    OutdatedIndex,
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
            Self::CallIntoBlock { address, block } => {
                write!(f, "{address:#x} is called, but {block:?} stands there")
            }
            Self::Poisoned => {
                f.write_str("the context was poisoned by a failed rollback and refuses lifting")
            }
            Self::ForeignIndex => f.write_str("the address index describes another context"),
            Self::OutdatedIndex => {
                f.write_str("the context changed since its address index was computed")
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
    /// Whether constructions name what they emit; see
    /// [`without_debug_names`](Self::without_debug_names).
    naming: bool,
}

impl<'a, 'str> LiftTarget<'a, 'str> {
    /// Binds `addresses`, rebuilding it from `ctx` first unless it is
    /// [current](AddressIndex::is_current), and selects `function` as the host.
    ///
    /// This is the safe binding, and the recommended one: an index that is
    /// current is complete, so no construction creates a second block or
    /// function at an address the context already covers; one that is not —
    /// computed for another context, or left behind by a mutation made
    /// without it — is rebuilt here, which costs O(module) once per such
    /// event and nothing otherwise. A caller that threads its index through
    /// every mutation pays the rebuild never; one that hands out
    /// [`Context::block_mut`] or a [`BodyMut`](crate::value::BodyMut) between
    /// lifts pays it at the next binding, every time.
    ///
    /// [`bind_indexed`](Self::bind_indexed) refuses instead of rebuilding, for
    /// a caller that would rather learn its index fell behind.
    pub fn bind_or_refresh(
        ctx: &'a mut Context<'str>,
        addresses: &'a mut AddressIndex,
        function: FunctionId,
    ) -> Result<Self, TargetError> {
        if !addresses.is_current(ctx) {
            addresses.refresh(ctx);
        }
        Self::bind_indexed(ctx, addresses, function)
    }

    /// Binds a caller-maintained `addresses` without rebuilding it, and selects
    /// `function` as the host.
    ///
    /// The index must be [current](AddressIndex::is_current) for `ctx`: computed
    /// for this context instance and carried through every address-bearing
    /// change since, by the tracked `*_indexed` mutators or by a caller that
    /// applied a change by hand and [vouched for it](AddressIndex::mark_current).
    /// An index computed for another context is refused as
    /// [`ForeignIndex`](TargetError::ForeignIndex); one the context has moved
    /// past as [`OutdatedIndex`](TargetError::OutdatedIndex). Only a current
    /// index is complete, and only a complete index lets a construction know
    /// that an address it does not list is free. The entries it does list are
    /// still checked against the context as they are used
    /// ([`StaleIndex`](TargetError::StaleIndex)).
    ///
    /// This is the hot-path binding for a caller that keeps its index current
    /// on purpose — an emulator that lifts on demand, a session that threads
    /// one index through a run — and wants to know when it has not. Everyone
    /// else uses [`bind_or_refresh`](Self::bind_or_refresh).
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
        if !addresses.describes(ctx) {
            return Err(TargetError::ForeignIndex);
        }
        if !addresses.is_current(ctx) {
            return Err(TargetError::OutdatedIndex);
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
            naming: true,
        })
    }

    /// Constructions into this target give their values no debug names: no
    /// register load named after its register, no block after its label.
    /// For IR that is read once and discarded — a scratch lift — where the
    /// names are minted, deduplicated and dropped without being printed.
    /// The IR means the same with or without them.
    pub fn without_debug_names(mut self) -> Self {
        self.naming = false;
        self
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
        // Every block in the arena is on the roster — `push_block` puts it
        // there, and only leaving the arena takes it off — so the entry needs
        // no rostering, and checking would scan the host's every block.
        debug_assert!(
            self.ctx.bodies[self.function].is_rostered(entry.local),
            "the entry block is not on its function's roster"
        );
        Ok(Construction {
            target: self,
            journal,
            record: Recorder::new(address, length, entry),
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

    /// Declares the index current again at the end of a construction.
    ///
    /// The index was current when the target bound, and the target holds
    /// both it and the context exclusively, so every change since is the
    /// construction's own: blocks it made at addresses through the index, or,
    /// on rollback, deleted and forgot again. Nothing was made at an address
    /// without going through the index, so the index is complete.
    fn settle_index(&mut self) {
        self.addresses.mark_current(self.ctx);
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

/// A callee the construction promised, by minted slot, and makes real at
/// commit.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Minted {
    /// The function at a machine address, made there if none is.
    Address(u64),
    /// The function of a name the semantics give, made if none has it.
    Named(Box<str>),
    /// A block of another function, at `address`, that becomes the entry of a
    /// function of its own when the split lands.
    Promoted { block: BlockId, address: u64 },
}

impl Minted {
    /// The address to report when the slot goes unresolved.
    fn address(&self) -> Option<u64> {
        match self {
            Self::Address(address) | Self::Promoted { address, .. } => Some(*address),
            Self::Named(_) => None,
        }
    }
}

/// Where a direct transfer to a machine address lands: see
/// [`Construction::transfer_at`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transfer {
    /// A block of the host: the transfer is an intra-function branch.
    Block(BlockId),
    /// Another function's entry: the transfer is inter-procedural, a tail
    /// call. A [minted](Callee::Minted) callee is one the commit creates by
    /// splitting the block it lands on out of its function.
    Function(Callee),
}

/// What [`Construction::transfer_at`] does with a target that is a block of
/// another function, not its entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Promotion {
    /// Refuse it: [`TargetError::ForeignBlock`].
    Refuse,
    /// Make it an entry: at commit the block is split out of its function
    /// into a function of its own (see [`Context::split_function_at_indexed`]),
    /// and the transfer is a tail call to that function, reported now as a
    /// minted callee. Only a block that *starts* at the address is promoted;
    /// an address interior to a foreign block is refused.
    SplitFunction,
}

/// What one construction added, for taking it back.
#[derive(Debug)]
struct Journal {
    /// The entry was made by this construction. Otherwise it was an empty
    /// block that existed before, whose contents alone are the construction's.
    entry_created: bool,
    placeholders: Vec<Placeholder>,
    /// Splits of lifted blocks this construction branches into, performed at
    /// commit. Their tails were issued after the block mark, so a rollback
    /// deletes them like any other block and the split never happens.
    splits: Vec<PendingSplit>,
    /// Callees the construction promised, by minted slot, to create at commit.
    minted: Vec<Minted>,
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
/// Resolve the instruction's module-level targets first, then take an
/// [`emitter`](Self::emitter) for its entry and emit, reporting the blocks
/// and exits as they come. [`commit`](Self::commit) keeps the result;
/// dropping the construction before that undoes it.
pub struct Construction<'t, 'a, 'str> {
    target: &'t mut LiftTarget<'a, 'str>,
    journal: Journal,
    /// The instruction's blocks and exits, as its emitter reports them.
    record: Recorder,
    settled: bool,
}

impl<'t, 'a, 'str> Construction<'t, 'a, 'str> {
    /// The address being lifted.
    pub fn address(&self) -> u64 {
        self.record.lifted().address()
    }

    /// The instruction's encoded length.
    pub fn length(&self) -> usize {
        self.record.lifted().length()
    }

    /// The block the instruction's p-code starts in.
    pub fn entry(&self) -> BlockId {
        self.record.lifted().entry()
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
        Ok(Callee::Minted(self.mint(Minted::Address(address))))
    }

    /// The function a call to `name` reaches: the one of that name, or one
    /// promised under it and made at commit, like [`callee_at`](Self::callee_at).
    pub fn callee_named(&mut self, name: &str) -> Callee {
        if let Some(function) = FunctionBody::from_name(self.target.ctx, name) {
            return Callee::Real(function.id);
        }
        Callee::Minted(self.mint(Minted::Named(Box::from(name))))
    }

    /// Where a direct transfer — a jump, or the fall-through — to `address`
    /// lands, with the policy for a target owned by another function.
    ///
    /// An address of the host, or one nothing owns, is a block of the host as
    /// [`block_at`](Self::block_at) gives it. Another function's entry is that
    /// function, to tail-call. A block of another function is refused, or
    /// [promoted](Promotion::SplitFunction) into a function to tail-call.
    pub fn transfer_at(
        &mut self,
        address: u64,
        promotion: Promotion,
    ) -> Result<Transfer, TargetError> {
        match self.block_at(address) {
            Ok(block) => Ok(Transfer::Block(block)),
            Err(TargetError::OwnedByFunction { owner, .. }) => {
                Ok(Transfer::Function(Callee::Real(owner)))
            }
            Err(TargetError::ForeignBlock { owner, .. })
                if promotion == Promotion::SplitFunction =>
            {
                let block = self
                    .target
                    .addresses
                    .block_at(address)
                    .expect("resolve_block found a foreign block here");
                if self.target.ctx.block(block).address != Some(address) {
                    return Err(TargetError::ForeignBlock { address, owner });
                }
                Ok(Transfer::Function(Callee::Minted(
                    self.mint(Minted::Promoted { block, address }),
                )))
            }
            Err(error) => Err(error),
        }
    }

    /// The slot of a promise, made once per distinct promise.
    fn mint(&mut self, minted: Minted) -> u32 {
        let slot = match self.journal.minted.iter().position(|m| *m == minted) {
            Some(slot) => slot,
            None => {
                self.journal.minted.push(minted);
                self.journal.minted.len() - 1
            }
        };
        slot as u32
    }

    /// An emitter positioned at the [entry](Self::entry): a builder bound to
    /// the host's body, so wherever it is repositioned it stays in the
    /// journal, that also takes the report of what the instruction owns and
    /// where it leaves.
    pub fn emitter(&mut self) -> Emitter<'_, 'str> {
        let mut builder = self.target.ctx.builder(self.record.lifted().entry());
        builder.set_naming(self.target.naming);
        Emitter {
            builder,
            record: &mut self.record,
        }
    }

    /// Keeps the instruction: creates the callees it promised and settles
    /// their call sites.
    pub fn commit(mut self) -> Result<Lifted, TargetError> {
        let function = self.target.function;
        let own_address = self.address();
        let unresolved = |minted: Option<&Minted>| TargetError::UnresolvedCallee {
            address: minted.and_then(Minted::address).unwrap_or(own_address),
        };
        // Every placeholder the instruction holds must be one of the promised
        // slots, every promised slot must have a site, and a structured call
        // holding one must be reported by the result as a call to what the
        // slot promises — a call the metadata omits is an instruction only
        // partly published. (A tail call is reported at the branch it stands
        // for, which may be a conditional branch elsewhere, so it is only
        // required to exist.) Checked before anything is created: a function,
        // once made, cannot be taken back. Nothing here allocates when the
        // instruction promised nothing, which is every instruction of a
        // flat lowering.
        let mut sites: Vec<Vec<LocalInsnId>> = vec![Vec::new(); self.journal.minted.len()];
        let body = &self.target.ctx.bodies[function];
        let mut reported: FxHashMap<LocalInsnId, &ExitKind> = FxHashMap::default();
        if !self.journal.minted.is_empty() {
            for exit in self.record.lifted().exits() {
                reported.entry(exit.site().local).or_insert(exit.kind());
            }
        }
        let reported = |local: LocalInsnId| reported.get(&local).copied();
        // An instruction that promised nothing and reports no call cannot
        // hold a minted slot it would be wrong about, so its operations are
        // not scanned: that scan was most of committing a lift's commonest
        // instruction.
        let may_hold_minted = !self.journal.minted.is_empty()
            || self
                .record
                .lifted()
                .exits()
                .iter()
                .any(|exit| exit.kind().is_call());
        let scanned = if may_hold_minted {
            self.journal.insns..body.insns.issued_len()
        } else {
            0..0
        };
        for raw in scanned {
            let local = raw.into();
            if !body.insns.contains(local) {
                continue;
            }
            let mnemonic = body.insns[local].mnemonic();
            let Some(slot) = mnemonic.minted_callee_slot() else {
                continue;
            };
            let Some(minted) = self.journal.minted.get(slot as usize) else {
                self.rollback();
                return Err(unresolved(None));
            };
            let consistent = match (mnemonic, reported(local)) {
                (Mnemonic::Call(_), Some(ExitKind::Call { callee, .. })) => {
                    match (minted, callee) {
                        (Minted::Address(minted), CallTarget::Address(address)) => {
                            minted == address
                        }
                        (Minted::Named(minted), CallTarget::Named(name)) => minted == name,
                        _ => false,
                    }
                }
                (Mnemonic::Call(_), _) => false,
                _ => true,
            };
            if !consistent {
                let error = unresolved(Some(minted));
                self.rollback();
                return Err(error);
            }
            sites[slot as usize].push(local);
        }
        if let Some(slot) = sites.iter().position(Vec::is_empty) {
            let error = unresolved(self.journal.minted.get(slot));
            self.rollback();
            return Err(error);
        }
        // A callee promised where a block stands would be made over that
        // block; the block may be this instruction's own placeholder, made
        // after the promise, so this is checked here and not when promising.
        for minted in &self.journal.minted {
            let Minted::Address(address) = *minted else {
                continue;
            };
            if let Some(block) = self.target.addresses.block_at(address) {
                self.rollback();
                return Err(TargetError::CallIntoBlock { address, block });
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
            let real = match &self.journal.minted[slot] {
                Minted::Address(address) => {
                    FunctionBody::from_addr_or_create_indexed(
                        self.target.ctx,
                        self.target.addresses,
                        *address,
                    )
                    .id
                }
                Minted::Named(name) => match FunctionBody::from_name(self.target.ctx, name) {
                    Some(function) => function.id,
                    None => {
                        FunctionBody::make(self.target.ctx, Cow::Owned(name.to_string()))
                            .expect("the name was free when it was promised")
                            .id
                    }
                },
                Minted::Promoted { block, .. } => self
                    .target
                    .ctx
                    .split_function_at_indexed(self.target.addresses, *block),
            };
            let body = &mut self.target.ctx.bodies[function];
            for site in sites {
                body.insns[site]
                    .mnemonic_mut()
                    .resolve_minted_callee(slot as u32, real);
            }
        }
        self.settled = true;
        self.target.settle_index();
        Ok(self.record.finish())
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
        let entry = self.record.lifted().entry();
        if !self.journal.entry_created {
            ctx.bodies[function].clear_block_instructions(entry);
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
        let address = self.record.lifted().address();
        if self.journal.entry_created && addresses.block_at(address) == Some(entry) {
            addresses.forget(address);
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
        self.target.settle_index();
    }
}

impl Drop for Construction<'_, '_, '_> {
    fn drop(&mut self) {
        self.rollback();
    }
}

/// A [`Builder`] into a construction's host that also takes the emitter's
/// report of the instruction: the blocks it owns and the places control
/// leaves it, at the operations that make them and before any lowering
/// choice. It dereferences to the builder.
///
/// The report is forward-only, as a p-code sink sees the instruction: a
/// call's continuation is settled by whatever [continues](Self::continue_in)
/// after it, or is the next machine instruction when nothing does.
pub struct Emitter<'c, 'str> {
    builder: Builder<'str, 'c>,
    record: &'c mut Recorder,
}

impl<'c, 'str> Emitter<'c, 'str> {
    /// Reports a block the instruction owns, in the order they are opened.
    pub fn block(&mut self, block: BlockId) {
        self.record.block(block);
    }

    /// Reports a transfer out of the instruction at `site`.
    ///
    /// A call's continuation is settled later: pass
    /// [`Continuation::Next`](crate::lift::Continuation::Next) and let
    /// [`continue_in`](Self::continue_in) or the commit decide.
    pub fn exit(&mut self, site: InstructionId, arm: ExitArm, kind: ExitKind) {
        self.record.exit(site, arm, kind);
    }

    /// Reports that the p-code after the last transfer was lowered into
    /// `block`. If that transfer was a call, `block` is where it returns to.
    pub fn continue_in(&mut self, block: BlockId) {
        self.record.continue_in(block);
    }
}

impl<'c, 'str> Deref for Emitter<'c, 'str> {
    type Target = Builder<'str, 'c>;

    fn deref(&self) -> &Self::Target {
        &self.builder
    }
}

impl DerefMut for Emitter<'_, '_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.builder
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        lift::{Continuation, ExitArm},
        value::{FunctionBody, QCodeMut, ValueId},
    };

    fn host(ctx: &mut Context<'static>, addresses: &mut AddressIndex, address: u64) -> FunctionId {
        FunctionBody::make_at_addr_indexed(ctx, addresses, address, None).id
    }

    /// Builds one straight-line instruction that falls through, as an emitter
    /// would.
    fn emit_fallthrough(construction: &mut Construction<'_, '_, '_>) {
        let next = construction.block_at(construction.address() + 1).unwrap();
        let mut emitter = construction.emitter();
        let temp = emitter.make_temp(8);
        let one = emitter.shr().get_const(1, 8);
        emitter.push_copy(one, ValueId::Temp(temp));
        let site = emitter.push_branch(next).id;
        emitter.exit(site, ExitArm::Unconditional, ExitKind::Fallthrough);
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
            emit_fallthrough(&mut construction);
            let entry = construction.entry();
            construction.commit().unwrap();

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
                emit_fallthrough(&mut construction);
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
        assert!(body.uses.is_empty());
        assert!(body.shared_first_use.is_empty());
        assert_eq!(ctx.bodies.len(), 1, "no callee was created");

        // And the address lifts afterwards.
        {
            let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
            let mut construction = target.begin(0x1000, 1).unwrap();
            emit_fallthrough(&mut construction);
            construction.commit().unwrap();
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
            let callee = construction.callee_at(0x2000).unwrap();
            assert_eq!(construction.callee_at(0x2000).unwrap(), callee);
            let mut emitter = construction.emitter();
            let site = emitter.push_call(callee).id;
            emitter.exit(
                site,
                ExitArm::Unconditional,
                ExitKind::Call {
                    callee: CallTarget::Address(0x2000),
                    continuation: Continuation::Next,
                },
            );
            construction.commit().unwrap();
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
    fn a_call_to_an_address_a_block_covers_is_refused_and_undone() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = host(&mut ctx, &mut addresses, 0x1000);
        let before = (ctx.to_string(), addresses.clone());
        // A call whose p-code ends with it, returning to the fall-through,
        // whose placeholder every emitter resolves first.
        let call_into = |construction: &mut Construction<'_, '_, '_>, address: u64| {
            let callee = construction.callee_at(address).unwrap();
            let next = construction.block_at(construction.address() + 1).unwrap();
            let mut emitter = construction.emitter();
            let site = emitter.push_call(callee).id;
            emitter.exit(
                site,
                ExitArm::Unconditional,
                ExitKind::Call {
                    callee: CallTarget::Address(address),
                    continuation: Continuation::Next,
                },
            );
            next
        };

        // `call $+1`: the callee's address is the instruction's own
        // fall-through, whose placeholder this construction holds.
        {
            let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
            let mut construction = target.begin(0x1000, 1).unwrap();
            let next = call_into(&mut construction, 0x1001);
            assert_eq!(
                construction.commit().err(),
                Some(TargetError::CallIntoBlock {
                    address: 0x1001,
                    block: next
                })
            );
            assert!(!target.is_poisoned());
        }
        assert_eq!(ctx.to_string(), before.0);
        assert_eq!(addresses, before.1);
        assert!(addresses.is_current(&ctx));
        assert_eq!(ctx.bodies.len(), 1, "no function was made");

        // A block another instruction left is refused the same way.
        let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
        let mut construction = target.begin(0x1000, 1).unwrap();
        emit_fallthrough(&mut construction);
        construction.commit().unwrap();
        let placeholder = target.addresses().block_at(0x1001).unwrap();
        let mut construction = target.begin(0x1010, 1).unwrap();
        call_into(&mut construction, 0x1001);
        assert_eq!(
            construction.commit().err(),
            Some(TargetError::CallIntoBlock {
                address: 0x1001,
                block: placeholder
            })
        );
        assert_eq!(target.context().bodies.len(), 1);
        assert_eq!(target.addresses().block_at(0x1001), Some(placeholder));
    }

    #[test]
    fn a_call_the_result_does_not_report_is_refused_and_undone() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = host(&mut ctx, &mut addresses, 0x1000);
        {
            let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();

            let mut construction = target.begin(0x1000, 5).unwrap();
            let callee = construction.callee_at(0x2000).unwrap();
            construction.emitter().push_call(callee);
            assert_eq!(
                construction.commit().err(),
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
        BasicBlock::from_id_mut(ctx, root).cover_address(0x1004);
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
            construction.emitter().push_branch(tail);
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
            let tail = construction.block_at(0x1004).unwrap();
            let mut emitter = construction.emitter();
            let site = emitter.push_branch(tail).id;
            emitter.exit(
                site,
                ExitArm::Unconditional,
                ExitKind::Branch { target: 0x1004 },
            );
            construction.commit().unwrap();
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
        let zero = construction.context().shared.get_const(0, 8);
        let mut emitter = construction.emitter();
        let site = emitter.push_branchind(zero).id;
        emitter.exit(site, ExitArm::Unconditional, ExitKind::BranchInd);
        construction.commit().unwrap();
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
        let mut emitter = construction.emitter();
        let site = emitter.push_branch(entry).id;
        emitter.exit(
            site,
            ExitArm::Unconditional,
            ExitKind::Branch { target: 0x1004 },
        );
        construction.commit().unwrap();
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
            emit_fallthrough(&mut construction);
            // The emitter misbehaves: it writes into a block the construction
            // does not own, which the journal cannot take back.
            let zero = construction.context().shared.get_const(0, 8);
            let mut builder = construction.emitter();
            builder.switch_to_block(other);
            builder.push_branchind(zero);
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
            LiftTarget::bind_or_refresh(&mut ctx, &mut addresses, function)
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
        // A block the index remembers but the context no longer has. Deleting
        // it behind the index leaves the index behind, which binding notices;
        // a caller that then vouches for the index anyway gets the per-entry
        // check instead.
        let gone = BasicBlock::make(&mut ctx, function)
            .with_address_indexed(&mut addresses, 0x1050)
            .id;
        ctx.delete_block(gone);
        assert_eq!(
            LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).err(),
            Some(TargetError::OutdatedIndex)
        );
        addresses.mark_current(&ctx);

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
    fn an_index_of_another_context_is_foreign_even_when_its_ids_match() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = host(&mut ctx, &mut addresses, 0x1000);
        // A clone has the same ids and, at this moment, the same shape — and
        // is still another module: an index built for it does not describe
        // this one.
        let twin = ctx.clone();
        assert_ne!(twin.identity(), ctx.identity());
        let mut foreign = AddressIndex::analyze(&twin);
        assert_eq!(foreign, addresses, "same contents");
        assert_eq!(
            LiftTarget::bind_indexed(&mut ctx, &mut foreign, function).err(),
            Some(TargetError::ForeignIndex)
        );
        // The safe binding rebuilds it for this context instead.
        LiftTarget::bind_or_refresh(&mut ctx, &mut foreign, function).unwrap();
        assert!(foreign.is_current(&ctx));
        // A revision travels with its context through moves, not clones.
        let moved = ctx;
        assert!(foreign.is_current(&moved));
    }

    #[test]
    fn an_index_that_omits_an_addressed_block_is_outdated_and_never_duplicates_it() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = host(&mut ctx, &mut addresses, 0x1000);
        assert!(addresses.is_current(&ctx));

        // Instruction-level work moves nothing: the index stays current.
        let zero = ctx.shared.get_const(0, 8);
        let scratch = BasicBlock::make(&mut ctx, function).id;
        ctx.builder(scratch).push_branchind(zero);
        assert!(
            addresses.is_current(&ctx),
            "an address-less block is not indexed"
        );

        // A block given an address behind the index: the index now omits it.
        let unlisted = BasicBlock::make(&mut ctx, function).with_address(0x1010).id;
        assert_eq!(addresses.block_at(0x1010), None);
        assert!(!addresses.is_current(&ctx));
        assert!(addresses.describes(&ctx));
        assert_eq!(
            LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).err(),
            Some(TargetError::OutdatedIndex)
        );

        // The safe binding refreshes, and a construction branching to that
        // address finds the block rather than making a second one.
        let mut target = LiftTarget::bind_or_refresh(&mut ctx, &mut addresses, function).unwrap();
        let mut construction = target.begin(0x1000, 1).unwrap();
        assert_eq!(construction.block_at(0x1010).unwrap(), unlisted);
        let mut emitter = construction.emitter();
        let site = emitter.push_branch(unlisted).id;
        emitter.exit(
            site,
            ExitArm::Unconditional,
            ExitKind::Branch { target: 0x1010 },
        );
        construction.commit().unwrap();
        let _ = target;
        assert!(
            addresses.is_current(&ctx),
            "a construction leaves it current"
        );
        assert_eq!(
            ctx.block_ids()
                .iter()
                .filter(|&&b| ctx.block(b).address == Some(0x1010))
                .count(),
            1
        );
    }

    #[test]
    fn editing_the_index_by_hand_drops_its_currency_so_bind_rebuilds_it() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = host(&mut ctx, &mut addresses, 0x1000);
        let existing = BasicBlock::make(&mut ctx, function)
            .with_address_indexed(&mut addresses, 0x1010)
            .id;
        let stray = BasicBlock::make(&mut ctx, function).id;
        assert!(addresses.is_current(&ctx));

        // An entry omitted by hand: the index no longer follows from the
        // context, so the safe binding rebuilds it and finds the block.
        addresses.forget(0x1010);
        assert!(!addresses.is_current(&ctx));
        assert!(!addresses.describes(&ctx));
        assert_eq!(
            LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).err(),
            Some(TargetError::ForeignIndex)
        );
        {
            let mut target =
                LiftTarget::bind_or_refresh(&mut ctx, &mut addresses, function).unwrap();
            let mut construction = target.begin(0x1000, 1).unwrap();
            assert_eq!(construction.block_at(0x1010).unwrap(), existing);
            construction.abort();
        }

        // An entry replaced by hand with the wrong block: rebuilt just the
        // same, and the right block comes back.
        addresses.set_block(0x1010, stray);
        assert!(!addresses.is_current(&ctx));
        {
            let mut target =
                LiftTarget::bind_or_refresh(&mut ctx, &mut addresses, function).unwrap();
            let mut construction = target.begin(0x1000, 1).unwrap();
            assert_eq!(construction.block_at(0x1010).unwrap(), existing);
            construction.abort();
        }
        assert!(addresses.is_current(&ctx));

        // The other hand edits drop currency too.
        addresses.rehome_block(0x1010, existing, stray);
        assert!(!addresses.is_current(&ctx));
        addresses.refresh(&ctx);
        addresses
            .register(&mut ctx, 0x1020, AddressTarget::Block(stray))
            .unwrap();
        assert!(!addresses.is_current(&ctx));

        // A hand edit that is right can be vouched for, which is the
        // caller-maintained path — and it is the caller's claim.
        addresses.refresh(&ctx);
        addresses.forget(0x1010);
        addresses.set_block(0x1010, existing);
        addresses.mark_current(&ctx);
        LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
        assert_eq!(
            ctx.block_ids()
                .iter()
                .filter(|&&b| ctx.block(b).address == Some(0x1010))
                .count(),
            1,
            "no duplicate was ever made"
        );
    }

    #[test]
    fn a_wrong_currency_claim_is_the_callers_bug_and_duplicates() {
        // The documented hazard of `mark_current`: the claim is trusted, so an
        // omission the caller did not know of becomes a second block at the
        // same address. This is why the claim is the caller's alone to make,
        // and why the tracked mutators are preferred.
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = host(&mut ctx, &mut addresses, 0x1000);
        let unlisted = BasicBlock::make(&mut ctx, function).with_address(0x1010).id;
        addresses.mark_current(&ctx);
        let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
        let mut construction = target.begin(0x1000, 1).unwrap();
        let duplicate = construction.block_at(0x1010).unwrap();
        assert_ne!(duplicate, unlisted);
        let mut emitter = construction.emitter();
        let site = emitter.push_branch(duplicate).id;
        emitter.exit(
            site,
            ExitArm::Unconditional,
            ExitKind::Branch { target: 0x1010 },
        );
        construction.commit().unwrap();
        let _ = target;
        assert_eq!(
            ctx.block_ids()
                .iter()
                .filter(|&&b| ctx.block(b).address == Some(0x1010))
                .count(),
            2,
            "what a wrong claim costs"
        );
    }

    #[test]
    fn rebinding_after_arbitrary_mutation_refreshes_or_refuses() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = host(&mut ctx, &mut addresses, 0x1000);
        {
            let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
            let mut construction = target.begin(0x1000, 1).unwrap();
            emit_fallthrough(&mut construction);
            construction.commit().unwrap();
        }
        assert!(addresses.is_current(&ctx));
        let placeholder = addresses.block_at(0x1001).unwrap();

        // Arbitrary mutation between lifts: a pass deletes the fall-through
        // placeholder without the index.
        ctx.delete_block(placeholder);
        assert_eq!(
            LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).err(),
            Some(TargetError::OutdatedIndex)
        );
        let mut target = LiftTarget::bind_or_refresh(&mut ctx, &mut addresses, function).unwrap();
        assert_eq!(target.addresses().block_at(0x1001), None, "refreshed");
        let mut construction = target.begin(0x1001, 1).unwrap();
        assert_ne!(
            construction.entry(),
            placeholder,
            "a fresh block, not the deleted id"
        );
        emit_fallthrough(&mut construction);
        construction.commit().unwrap();
        let _ = target;
        assert!(addresses.is_current(&ctx));
    }

    #[test]
    fn a_rolled_back_construction_leaves_the_index_current() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = host(&mut ctx, &mut addresses, 0x1000);
        let revision = ctx.revision();
        {
            let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
            let mut construction = target.begin(0x1000, 1).unwrap();
            emit_fallthrough(&mut construction);
            construction.abort();
            // Rebinding right away works: the rollback settled the index.
            target.begin(0x1000, 1).unwrap().abort();
        }
        assert!(addresses.is_current(&ctx));
        assert_ne!(ctx.revision(), revision, "the placeholders came and went");
    }

    #[test]
    fn a_reloaded_or_cloned_module_counts_its_bodies_changes_on_its_own_clock() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = host(&mut ctx, &mut addresses, 0x1000);
        let block = BasicBlock::make(&mut ctx, function)
            .with_address_indexed(&mut addresses, 0x1010)
            .id;

        let bytes = bincode::serde::encode_to_vec(&ctx, bincode::config::standard()).unwrap();
        let (mut reloaded, _): (Context<'static>, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        let mut index = AddressIndex::analyze(&reloaded);
        assert!(index.is_current(&reloaded));
        // A body-level mutation of the reloaded module reaches its clock...
        reloaded.bodies[function].delete_block(block);
        assert!(!index.is_current(&reloaded));
        assert_eq!(
            LiftTarget::bind_indexed(&mut reloaded, &mut index, function).err(),
            Some(TargetError::OutdatedIndex)
        );
        // ...and not the original's, which the clone below also leaves alone.
        assert!(addresses.is_current(&ctx));
        let mut twin = ctx.clone();
        twin.bodies[function].delete_block(block);
        assert!(addresses.is_current(&ctx));
        assert!(!AddressIndex::analyze(&ctx).is_current(&twin));
    }

    #[test]
    fn a_named_callee_is_made_at_commit_and_never_by_a_failed_construction() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = host(&mut ctx, &mut addresses, 0x1000);
        let before = ctx.to_string();
        {
            let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
            let mut construction = target.begin(0x1000, 2).unwrap();
            let callee = construction.callee_named("syscall");
            assert_eq!(callee, Callee::Minted(0));
            assert_eq!(
                construction.callee_named("syscall"),
                callee,
                "one slot per name"
            );
            construction.emitter().push_call(callee);
            construction.abort();
        }
        assert_eq!(ctx.to_string(), before);
        assert!(FunctionBody::from_name(&ctx, "syscall").is_none());

        let site = {
            let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
            let mut construction = target.begin(0x1000, 2).unwrap();
            let callee = construction.callee_named("syscall");
            let mut emitter = construction.emitter();
            let site = emitter.push_call(callee).id;
            emitter.exit(
                site,
                ExitArm::Unconditional,
                ExitKind::Call {
                    callee: CallTarget::Named("syscall".into()),
                    continuation: Continuation::Next,
                },
            );
            construction.commit().unwrap();
            site
        };
        let syscall = FunctionBody::from_name(&ctx, "syscall")
            .expect("made at commit")
            .id;
        let call = crate::value::Instruction::from_id(&ctx, site);
        assert!(matches!(
            call.mnemonic(),
            crate::value::insn::Mnemonic::Call(call) if call.target == Callee::Real(syscall)
        ));
        // A later construction names the same function rather than a second.
        let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
        let mut construction = target.begin(0x1002, 2).unwrap();
        assert_eq!(construction.callee_named("syscall"), Callee::Real(syscall));
    }

    #[test]
    fn a_call_reported_under_another_name_than_its_site_promises_is_refused() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = host(&mut ctx, &mut addresses, 0x1000);
        let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
        let mut construction = target.begin(0x1000, 2).unwrap();
        let callee = construction.callee_named("syscall");
        let mut emitter = construction.emitter();
        let site = emitter.push_call(callee).id;
        emitter.exit(
            site,
            ExitArm::Unconditional,
            ExitKind::Call {
                callee: CallTarget::Named("other".into()),
                continuation: Continuation::Next,
            },
        );
        assert_eq!(
            construction.commit().err(),
            Some(TargetError::UnresolvedCallee { address: 0x1000 })
        );
        assert!(FunctionBody::from_name(&ctx, "syscall").is_none());
    }

    /// Two functions: the host at 0x1000, and `other` at 0x2000 whose second
    /// block, at 0x2010, holds code and is reached from its root.
    fn host_and_other(
        ctx: &mut Context<'static>,
        addresses: &mut AddressIndex,
    ) -> (FunctionId, FunctionId, BlockId) {
        let function = host(ctx, addresses, 0x1000);
        let other = host(ctx, addresses, 0x2000);
        let root = BasicBlock::make(ctx, other)
            .with_address_indexed(addresses, 0x2000)
            .id;
        let foreign = BasicBlock::make(ctx, other)
            .with_address_indexed(addresses, 0x2010)
            .id;
        ctx.builder(root).push_branch(foreign);
        let zero = ctx.shared.get_const(0, 8);
        ctx.builder(foreign).push_branchind(zero);
        (function, other, foreign)
    }

    #[test]
    fn a_transfer_resolves_to_a_block_a_function_or_a_promoted_foreign_block() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let (function, other, foreign) = host_and_other(&mut ctx, &mut addresses);
        {
            let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
            let mut construction = target.begin(0x1000, 2).unwrap();

            // The host's own address: a block, made as a placeholder.
            let Transfer::Block(next) =
                construction.transfer_at(0x1002, Promotion::Refuse).unwrap()
            else {
                panic!()
            };
            assert_eq!(construction.context().block(next).address, Some(0x1002));
            // Another function's entry: that function, whatever the policy.
            assert_eq!(
                construction.transfer_at(0x2000, Promotion::Refuse).unwrap(),
                Transfer::Function(Callee::Real(other))
            );
            // A block of another function: refused, or promoted as a promise.
            assert_eq!(
                construction.transfer_at(0x2010, Promotion::Refuse).err(),
                Some(TargetError::ForeignBlock {
                    address: 0x2010,
                    owner: other
                })
            );
            assert_eq!(
                construction
                    .transfer_at(0x2010, Promotion::SplitFunction)
                    .unwrap(),
                Transfer::Function(Callee::Minted(0))
            );
            assert_eq!(
                construction
                    .transfer_at(0x2010, Promotion::SplitFunction)
                    .unwrap(),
                Transfer::Function(Callee::Minted(0)),
                "one promise per address"
            );
            // Nothing is split yet.
            assert_eq!(construction.context().block(foreign).address, Some(0x2010));
            assert_eq!(construction.context().functions().count(), 2);
        }
        // An address interior to a foreign block is not promoted.
        BasicBlock::from_id_mut(&mut ctx, foreign).cover_address(0x2014);
        assert!(!addresses.is_current(&ctx));
        addresses.refresh(&ctx);
        let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
        let mut construction = target.begin(0x1000, 2).unwrap();
        assert_eq!(
            construction
                .transfer_at(0x2014, Promotion::SplitFunction)
                .err(),
            Some(TargetError::ForeignBlock {
                address: 0x2014,
                owner: other
            })
        );
    }

    #[test]
    fn a_promotion_lands_at_commit_and_an_abandoned_one_leaves_the_other_function_alone() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let (function, other, foreign) = host_and_other(&mut ctx, &mut addresses);
        let before = (ctx.to_string(), addresses.clone());

        // A jump from the host into the middle of `other`, abandoned.
        {
            let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
            let mut construction = target.begin(0x1000, 2).unwrap();
            let Transfer::Function(callee) = construction
                .transfer_at(0x2010, Promotion::SplitFunction)
                .unwrap()
            else {
                panic!()
            };
            construction.emitter().push_tail_call(callee);
            construction.abort();
            assert!(!target.is_poisoned());
        }
        assert_eq!(ctx.to_string(), before.0);
        assert_eq!(addresses, before.1);
        assert!(addresses.is_current(&ctx));

        // The same jump committed: `other` is split at 0x2010, the new
        // function owns the block, and the jump is a tail call to it.
        let site = {
            let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
            let mut construction = target.begin(0x1000, 2).unwrap();
            let Transfer::Function(callee) = construction
                .transfer_at(0x2010, Promotion::SplitFunction)
                .unwrap()
            else {
                panic!()
            };
            let mut emitter = construction.emitter();
            let site = emitter.push_tail_call(callee).id;
            // The original transfer was a branch; the tail call is policy.
            emitter.exit(
                site,
                ExitArm::Unconditional,
                ExitKind::Branch { target: 0x2010 },
            );
            construction.commit().unwrap();
            site
        };
        let promoted = addresses.function_at(0x2010).expect("split out at commit");
        assert_ne!(promoted, other);
        assert_eq!(ctx.functions().count(), 3);
        let root = ctx
            .function(promoted)
            .root_id()
            .expect("the block became its root");
        assert!(ctx.block(BlockId::new(promoted, root)).has_insns());
        assert!(!ctx.contains_block(foreign), "rehomed out of `other`");
        let jump = crate::value::Instruction::from_id(&ctx, site);
        assert!(matches!(
            jump.mnemonic(),
            crate::value::insn::Mnemonic::TailCall(call) if call.target == Callee::Real(promoted)
        ));
        // `other`'s root now tail-calls the split-out function too.
        let other_root = BlockId::new(other, ctx.function(other).root_id().unwrap());
        assert!(matches!(
            BasicBlock::from_id(&ctx, other_root)
                .instructions()
                .last()
                .map(|i| i.mnemonic().clone()),
            Some(crate::value::insn::Mnemonic::TailCall(call)) if call.target == Callee::Real(promoted)
        ));
        assert!(addresses.is_current(&ctx));
    }

    #[test]
    fn exits_survive_a_commit() {
        let mut ctx = Context::new();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = host(&mut ctx, &mut addresses, 0x1000);
        let mut target = LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
        let mut construction = target.begin(0x1000, 1).unwrap();
        emit_fallthrough(&mut construction);
        let lifted = construction.commit().unwrap();
        assert!(lifted.falls_through());
        assert_eq!(lifted.exits().len(), 1);
        assert_eq!(lifted.exits()[0].arm(), ExitArm::Unconditional);
        assert_eq!(lifted.exits()[0].kind(), &ExitKind::Fallthrough);
    }
}
