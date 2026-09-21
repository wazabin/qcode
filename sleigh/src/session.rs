//! Sessions that own the IR they lift into.
//!
//! [`SleighLifter`] lowers into whatever context a caller binds, which suits
//! a VM or a disassembler that owns its module for other reasons. A consumer
//! that only wants IR out of bytes is better served by a session that owns
//! the context, keeps its address index current and decodes with an explicit
//! policy, so the caller never handles an index or a `FunctionId` at all:
//!
//! - A [`LiftSession`] accumulates instructions in one function and hands
//!   the context over at the end. Its handles are ordinary [`Lifted`] results
//!   whose blocks stay valid for the life of that context.
//! - A [`ScratchSession`] lifts one instruction at a time and discards it
//!   before the next. Its handles are views tied to the borrow of one lifted
//!   instruction, so the compiler refuses a view that outlives its
//!   instruction — see [`ScratchLifted`].
//!
//! Both decode with a [`FixedDecoder`] unless given a decoded instruction; a
//! sweep that carries context forward uses a [`LinearDecoder`] alongside a
//! session and feeds it decoded instructions.
//!
//! A failed lift rolls back; one that cannot poisons the context (see
//! [`Context::is_poisoned`]), and a session over a poisoned context refuses
//! to lift or to hand the context over. Disposing of the session is the
//! recovery; a scratch session recovers by itself, since discarding the
//! instruction discards the damage.

use std::{cell::OnceCell, sync::Arc};

use qcode::{
    address_index::AddressIndex,
    context::Context,
    lift::{Exit, ExitArm, ExitKind, LiftTarget, Lifted, ScratchStore, TargetError},
    space::SpaceId,
    value::{
        BlockId, FunctionBody, FunctionId, InstructionId, LocalBlockId, LocalInsnId, LocalTempId,
        LocalValueId, Varnode, VarnodeId,
        function::InsnIds,
        insn::{Callee, Mnemonic},
    },
};
use sleigh::{CompiledSpec, ContextBytes, ContextError, Instruction, RegisterId, RegisterSlice};

use crate::{
    FlatPcode, LiftError, SleighLifter,
    cache::{Instance, LiftCache, Lookup, Template},
    decode::FixedDecoder,
};

#[cfg(doc)]
use crate::decode::LinearDecoder;

/// Which function a session lifts into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Host {
    /// A function with no address: the instructions are not a function's
    /// entry, or the session does not care.
    Anonymous,
    /// The function whose entry is at this address. In an existing context,
    /// the function already there, or a new one if none is.
    At(u64),
}

/// A session that accumulates instructions in a function it owns. See the
/// [module documentation](self).
pub struct LiftSession<'l, 'spec> {
    lifter: &'l SleighLifter<'spec>,
    decoder: FixedDecoder<'spec>,
    ctx: Context<'static>,
    addresses: AddressIndex,
    function: FunctionId,
    cache: Option<Arc<LiftCache>>,
}

impl<'l, 'spec> LiftSession<'l, 'spec> {
    /// A session over a fresh context of the lifter's specification.
    pub fn new(lifter: &'l SleighLifter<'spec>, host: Host) -> Self {
        let mut ctx = lifter.new_context();
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = Self::select(&mut ctx, &mut addresses, host);
        Self {
            lifter,
            decoder: FixedDecoder::new(lifter.spec()),
            ctx,
            addresses,
            function,
            cache: None,
        }
    }

    /// A session continuing in `ctx`, which must be one of this lifter's
    /// specification and not [poisoned](Context::is_poisoned). `host` is
    /// resolved in that context.
    pub fn in_context(
        lifter: &'l SleighLifter<'spec>,
        mut ctx: Context<'static>,
        host: Host,
    ) -> Result<Self, LiftError> {
        lifter.check_compatible(&ctx)?;
        if ctx.is_poisoned() {
            return Err(TargetError::Poisoned.into());
        }
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = Self::select(&mut ctx, &mut addresses, host);
        Ok(Self {
            lifter,
            decoder: FixedDecoder::new(lifter.spec()),
            ctx,
            addresses,
            function,
            cache: None,
        })
    }

    fn select(ctx: &mut Context<'static>, addresses: &mut AddressIndex, host: Host) -> FunctionId {
        match host {
            Host::Anonymous => ctx.anon_function(),
            Host::At(address) => {
                FunctionBody::from_addr_or_create_indexed(ctx, addresses, address).id
            }
        }
    }

    /// Decodes every address with `context` instead of the specification's
    /// default.
    pub fn with_decode_context(mut self, context: ContextBytes) -> Result<Self, ContextError> {
        self.decoder = FixedDecoder::with_context(self.lifter.spec(), context)?;
        Ok(self)
    }

    /// Lifts through `cache`: an encoding seen before is replayed from its
    /// template instead of lowered again. Only [`lift`](Self::lift) uses it —
    /// the key needs the decode context, which the session's decoder has and
    /// a caller-decoded instruction does not carry. See [`LiftCache`].
    pub fn with_cache(mut self, cache: Arc<LiftCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    pub fn lifter(&self) -> &'l SleighLifter<'spec> {
        self.lifter
    }

    /// The function instructions are lifted into.
    pub fn function(&self) -> FunctionId {
        self.function
    }

    /// The IR so far. Every [`Lifted`] this session returned describes
    /// blocks of this context, valid until the IR is mutated.
    pub fn context(&self) -> &Context<'static> {
        &self.ctx
    }

    /// Whether a failed lift left the context holding IR it could not take
    /// back. A poisoned session refuses every further lift and
    /// [`into_context`](Self::into_context); its context is still readable
    /// through [`context`](Self::context). See [`Context::is_poisoned`].
    pub fn is_poisoned(&self) -> bool {
        self.ctx.is_poisoned()
    }

    #[cfg(test)]
    fn context_mut_for_test(&mut self) -> &mut Context<'static> {
        &mut self.ctx
    }

    /// Decodes the instruction at `address` from `bytes` with the session's
    /// fixed context and lifts it.
    pub fn lift(&mut self, address: u64, bytes: &[u8]) -> Result<Lifted, LiftError> {
        let instruction = self.decoder.decode(address, bytes)?;
        let Some(cache) = &self.cache else {
            return self.lift_decoded(&instruction);
        };
        let mut target =
            LiftTarget::bind_indexed(&mut self.ctx, &mut self.addresses, self.function)?;
        cache.lower(
            self.lifter,
            &mut target,
            &instruction,
            &self.decoder,
            self.lifter.flat_control_flow(),
        )
    }

    /// Lifts an instruction the caller decoded, with whatever context it
    /// chose.
    pub fn lift_decoded(&mut self, instruction: &Instruction<'_, '_>) -> Result<Lifted, LiftError> {
        let mut target =
            LiftTarget::bind_indexed(&mut self.ctx, &mut self.addresses, self.function)?;
        self.lifter.lift_into(&mut target, instruction)
    }

    /// Lifts already-flattened p-code of the session's specification, for a
    /// caller inspecting or caching that intermediate.
    pub fn lift_pcode(&mut self, flat: &FlatPcode) -> Result<Lifted, LiftError> {
        let mut target =
            LiftTarget::bind_indexed(&mut self.ctx, &mut self.addresses, self.function)?;
        self.lifter.lift_pcode_into(&mut target, flat)
    }

    /// Ends the session and hands over its context — unless a failed lift
    /// [poisoned](Self::is_poisoned) it, in which case the context is not
    /// published: it would carry IR no instruction accounts for.
    pub fn into_context(self) -> Result<Context<'static>, LiftError> {
        if self.ctx.is_poisoned() {
            return Err(TargetError::Poisoned.into());
        }
        Ok(self.ctx)
    }

    /// A session that discards each instruction before the next. See
    /// [`ScratchSession::new`].
    pub fn scratch(lifter: &'l SleighLifter<'spec>) -> ScratchSession<'l, 'spec> {
        ScratchSession::new(lifter)
    }
}

/// A session lifting one instruction at a time into storage it reuses.
///
/// Each [`lift`](Self::lift) discards the previous instruction first, so a
/// caller cannot forget to. The result borrows the session for as long as it
/// is examined, which is what keeps its views honest: the next lift needs the
/// session back, so no view of the previous instruction can be alive then.
///
/// ```compile_fail
/// # use wazabin_qcode_sleigh::{SleighLifter, session::ScratchSession};
/// # let spec = sleigh_precompile::x64::spec();
/// # let lifter = SleighLifter::new(spec);
/// let mut session = ScratchSession::new(&lifter);
/// let first = session.lift(0x1000, b"\x48\x89\xd8").unwrap();
/// let entry = first.entry();
/// let second = session.lift(0x1003, b"\x48\x89\xd8").unwrap(); // `first` is still borrowed
/// let _ = entry.address();
/// ```
///
/// The same goes for a resolved operand: it is a view, not a bare id.
///
/// ```compile_fail
/// # use wazabin_qcode_sleigh::{SleighLifter, session::ScratchSession};
/// # let spec = sleigh_precompile::x64::spec();
/// # let lifter = SleighLifter::new(spec);
/// let mut session = ScratchSession::new(&lifter);
/// let first = session.lift(0x1000, b"\x48\xc7\xc0\x44\x33\x22\x11").unwrap();
/// let operand = first.entry().instructions().next().unwrap().operands().next().unwrap();
/// let second = session.lift(0x1007, b"\x48\x89\xd8").unwrap(); // `operand` still borrows
/// let _ = operand.as_const();
/// ```
///
/// While a view is alive the session itself is out of reach, so nothing can
/// change the instruction the view shows.
///
/// ```compile_fail
/// # use wazabin_qcode_sleigh::{SleighLifter, session::ScratchSession};
/// # let spec = sleigh_precompile::x64::spec();
/// # let lifter = SleighLifter::new(spec);
/// let mut session = ScratchSession::new(&lifter);
/// let lifted = session.lift(0x1000, b"\x48\x89\xd8").unwrap();
/// let epoch = session.epoch(); // `lifted` still borrows the session
/// let _ = lifted.entry();
/// ```
pub struct ScratchSession<'l, 'spec> {
    lifter: &'l SleighLifter<'spec>,
    decoder: FixedDecoder<'spec>,
    store: ScratchStore,
    cache: Option<Arc<LiftCache>>,
}

impl<'l, 'spec> ScratchSession<'l, 'spec> {
    /// A scratch session for `lifter`. It lowers calls and returns as jumps
    /// (as [`SleighLifter::with_flat_control_flow`] does) whatever the lifter
    /// would: a structured call creates its callee function, and a function
    /// cannot be discarded with the instruction that created it. The exits
    /// still report calls and returns as what they were.
    pub fn new(lifter: &'l SleighLifter<'spec>) -> Self {
        Self {
            lifter,
            decoder: FixedDecoder::new(lifter.spec()),
            store: ScratchStore::new(lifter.new_context()),
            cache: None,
        }
    }

    /// Lifts through `cache`: an encoding seen before is replayed from its
    /// template instead of lowered again. Only [`lift`](Self::lift) uses it —
    /// the key needs the decode context, which the session's decoder has and
    /// a caller-decoded instruction does not carry. See [`LiftCache`].
    pub fn with_cache(mut self, cache: Arc<LiftCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Decodes every address with `context` instead of the specification's
    /// default.
    pub fn with_decode_context(mut self, context: ContextBytes) -> Result<Self, ContextError> {
        self.decoder = FixedDecoder::with_context(self.lifter.spec(), context)?;
        Ok(self)
    }

    /// Rebuilds the storage once instructions have interned `budget` more
    /// constants than the architecture defines; see
    /// [`ScratchStore::with_literal_budget`]. The default is
    /// [`qcode::lift::scratch::DEFAULT_LITERAL_BUDGET`].
    pub fn with_literal_budget(mut self, budget: usize) -> Self {
        self.store.set_literal_budget(budget);
        self
    }

    /// How many instructions have been lifted and discarded. Every block or
    /// instruction handle obtained before the latest lift belongs to an
    /// earlier epoch.
    pub fn epoch(&self) -> u64 {
        self.store.epoch()
    }

    /// Constants interned since the storage was built or last rebuilt; see
    /// [`ScratchStore::interned_literals`].
    pub fn interned_literals(&self) -> usize {
        self.store.interned_literals()
    }

    /// How many times the storage was rebuilt to stay within its literal
    /// budget; see [`ScratchStore::rebuilds`].
    pub fn rebuilds(&self) -> u64 {
        self.store.rebuilds()
    }

    /// The host function's arena footprint, for measuring what one
    /// instruction's worth of storage retains.
    pub fn arena_stats(&self) -> qcode::value::BodyArenaStats {
        self.store.arena_stats()
    }

    /// Discards the previous instruction, then decodes the one at `address`
    /// from `bytes` with the fixed context and lifts it.
    pub fn lift<'s, 'b>(
        &'s mut self,
        address: u64,
        bytes: &'b [u8],
    ) -> Result<ScratchLifted<'s, 'l, 'spec, 'b>, LiftError> {
        // A known encoding needs no decode: the views read the template, and
        // the instruction is decoded only if asked for.
        if let Some(cache) = &self.cache
            && let Some((template, instance)) =
                cache.find_undecoded(self.lifter, &self.decoder, true, address, bytes)?
        {
            let lifted = template.lifted_at(&instance, self.store.function());
            return Ok(ScratchLifted {
                session: self,
                instruction: OnceCell::new(),
                address,
                bytes: Some(bytes),
                lifted,
                template: Some((template, instance)),
            });
        }
        let instruction = self.decoder.decode(address, bytes)?;
        self.lower(instruction, true)
    }

    /// Discards the previous instruction, then lifts one the caller decoded.
    pub fn lift_decoded<'s, 'b>(
        &'s mut self,
        instruction: Instruction<'spec, 'b>,
    ) -> Result<ScratchLifted<'s, 'l, 'spec, 'b>, LiftError> {
        self.lower(instruction, false)
    }

    fn lower<'s, 'b>(
        &'s mut self,
        instruction: Instruction<'spec, 'b>,
        cached: bool,
    ) -> Result<ScratchLifted<'s, 'l, 'spec, 'b>, LiftError> {
        let cache = self.cache.as_deref().filter(|_| cached);
        let lookup = match cache {
            Some(cache) => cache.find(self.lifter, &instruction, &self.decoder, true)?,
            None => Lookup::Uncacheable,
        };
        if let Lookup::Hit(template, instance) = lookup {
            // Nothing is lowered: the views read the template.
            let address = instruction.address();
            let lifted = template.lifted_at(&instance, self.store.function());
            return Ok(ScratchLifted {
                session: self,
                instruction: OnceCell::from(instruction),
                address,
                bytes: None,
                lifted,
                template: Some((template, instance)),
            });
        }
        self.store.reset();
        // The instruction is read through the session's views and discarded,
        // never printed: its debug names would be minted for nothing.
        let mut target = self.store.target()?.without_debug_names();
        let lifted = match (cache, lookup) {
            (Some(cache), Lookup::Unknown) => {
                cache.miss(self.lifter, &mut target, &instruction, &self.decoder, true)?
            }
            _ => self.lifter.lower(&mut target, &instruction, true)?,
        };
        Ok(ScratchLifted {
            session: self,
            address: instruction.address(),
            instruction: OnceCell::from(instruction),
            bytes: None,
            lifted,
            template: None,
        })
    }
}

/// One instruction lifted by a [`ScratchSession`], readable until the next.
///
/// Every block, instruction and operand it exposes is a view carrying this
/// borrow, so none can be kept past the instruction. Operands are read by
/// position from the live instruction ([`ScratchInsn::operands`]); nothing
/// here takes a bare id and resolves it.
/// The storage recycles constant ids between instructions, so an id kept
/// from an earlier one may now number a different constant, and a facade
/// that resolved ids would hand back that other value. Raw
/// [`LocalValueId`]s still appear in [`ScratchInsn::mnemonic`] and as
/// [`ScratchInsn::result`]; they are keys for the caller's own tables while
/// the instruction is live, and nothing more.
///
/// An instruction the session's [cache](ScratchSession::with_cache) knows
/// is not lowered into the store at all: its views read the cached template
/// directly, with the same keys-not-handles contract. The store then still
/// holds the last instruction that was lowered.
pub struct ScratchLifted<'s, 'l, 'spec, 'b> {
    session: &'s mut ScratchSession<'l, 'spec>,
    /// Decoded when the instruction was lowered or asked for; a cached
    /// instruction found by its bytes is not decoded until then.
    instruction: OnceCell<Instruction<'spec, 'b>>,
    address: u64,
    /// The stream the instruction starts, kept until it is decoded.
    bytes: Option<&'b [u8]>,
    lifted: Lifted,
    /// The template the views read when the instruction came from the
    /// cache, and the instance of it the instruction is.
    template: Option<(Arc<Template>, Instance)>,
}

/// Where a view reads its instruction from: the store's context, or a
/// template and the instance it is read at. Both keep the context for the
/// architecture's varnodes.
#[derive(Clone, Copy)]
enum Source<'v> {
    Store(&'v Context<'static>),
    Template {
        ctx: &'v Context<'static>,
        template: &'v Template,
        instance: &'v Instance,
    },
}

impl<'v> Source<'v> {
    fn ctx(self) -> &'v Context<'static> {
        match self {
            Self::Store(ctx) | Self::Template { ctx, .. } => ctx,
        }
    }
}

impl<'s, 'l, 'spec, 'b> ScratchLifted<'s, 'l, 'spec, 'b> {
    fn source(&self) -> Source<'_> {
        let ctx = self.session.store.context();
        match &self.template {
            Some((template, instance)) => Source::Template {
                ctx,
                template,
                instance,
            },
            None => Source::Store(ctx),
        }
    }

    fn spec(&self) -> &'spec CompiledSpec {
        self.session.lifter.spec()
    }

    /// The decoded instruction: its text, operands and effects. Decoded at
    /// most once; a cached instruction is decoded here, on first request.
    pub fn decoded(&self) -> &Instruction<'spec, 'b> {
        self.instruction.get_or_init(|| {
            let bytes = self
                .bytes
                .expect("an undecoded instruction keeps its bytes");
            self.session
                .decoder
                .decode(self.address, bytes)
                .expect("the bytes decoded when the cache learned them")
        })
    }

    /// The instruction's own bytes.
    pub fn bytes(&self) -> &[u8] {
        match (self.instruction.get(), self.bytes) {
            (Some(instruction), _) => instruction.bytes(),
            (None, Some(bytes)) => &bytes[..self.lifted.length()],
            (None, None) => unreachable!("an instruction is decoded or keeps its bytes"),
        }
    }

    /// Whether the instruction was read from the session's cache rather
    /// than lowered.
    pub fn is_cached(&self) -> bool {
        self.template.is_some()
    }

    /// The plain result: address, length, exits and how control leaves. The
    /// block and instruction ids in it are this epoch's keys, not handles;
    /// the views below are the way to what they name.
    pub fn lifted(&self) -> &Lifted {
        &self.lifted
    }

    /// Every place control leaves the instruction, in emission order.
    pub fn exits(&self) -> impl Iterator<Item = ScratchExit<'_>> + '_ {
        let source = self.source();
        self.lifted.exits().iter().map(move |exit| ScratchExit {
            exit,
            site: ScratchInsn {
                source,
                spec: self.spec(),
                id: exit.site(),
            },
        })
    }

    /// The block control enters the instruction through.
    pub fn entry(&self) -> ScratchBlock<'_> {
        ScratchBlock {
            source: self.source(),
            spec: self.spec(),
            id: self.lifted.entry(),
        }
    }

    /// The instruction's blocks, the entry first.
    pub fn blocks(&self) -> impl Iterator<Item = ScratchBlock<'_>> + '_ {
        let source = self.source();
        let spec = self.spec();
        self.lifted
            .blocks()
            .iter()
            .map(move |&id| ScratchBlock { source, spec, id })
    }
}

/// A resolved operand of a [`ScratchInsn`], valid as long as the instruction.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub enum ScratchOperand<'v> {
    /// An interned constant: its raw value, and its width in bytes. The
    /// interner masks a constant to its width when it mints it.
    Const { value: u64, size: usize },
    /// A register or other named location of the architecture.
    Varnode(ScratchVarnode<'v>),
    /// The result of another instruction of the same scratch instruction.
    Result(ScratchInsn<'v>),
    /// A body-local temporary: the storage of an instruction-local unique.
    Temp(LocalTempId),
    /// A block of the same scratch instruction.
    Block(ScratchBlock<'v>),
    /// Any other kind of operand, kept as the key it is.
    Other(LocalValueId),
}

impl std::fmt::Debug for ScratchOperand<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Const { value, size } => write!(f, "Const({value:#x}, {size})"),
            Self::Varnode(varnode) => write!(f, "{varnode:?}"),
            Self::Result(insn) => write!(f, "Result({:?})", insn.id),
            Self::Temp(temp) => write!(f, "Temp({temp:?})"),
            Self::Block(block) => write!(f, "Block({:?})", block.id),
            Self::Other(other) => write!(f, "Other({other:?})"),
        }
    }
}

impl ScratchOperand<'_> {
    /// The value of a constant operand.
    pub fn as_const(&self) -> Option<u64> {
        match self {
            Self::Const { value, .. } => Some(*value),
            _ => None,
        }
    }
}

/// A register or other architectural location named by a scratch operand.
#[derive(Clone, Copy)]
pub struct ScratchVarnode<'v> {
    ctx: &'v Context<'static>,
    spec: &'v CompiledSpec,
    id: VarnodeId,
}

impl std::fmt::Debug for ScratchVarnode<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.name() {
            Some(name) => write!(f, "Varnode({name})"),
            None => write!(
                f,
                "Varnode({:?}:{}:{})",
                self.space(),
                self.address(),
                self.size()
            ),
        }
    }
}

impl<'v> ScratchVarnode<'v> {
    /// The id the varnode has in the module — architecture-defined, so it is
    /// the same in every scratch epoch, and usable as a key.
    pub fn id(&self) -> VarnodeId {
        self.id
    }

    pub fn space(&self) -> SpaceId {
        Varnode::from_id(self.ctx, self.id).space().id
    }

    pub fn address(&self) -> i64 {
        Varnode::from_id(self.ctx, self.id).address()
    }

    pub fn size(&self) -> usize {
        Varnode::from_id(self.ctx, self.id).size()
    }

    pub fn name(&self) -> Option<&str> {
        Varnode::from_id(self.ctx, self.id).name()
    }

    /// The location as SLEIGH names it: space, offset and size.
    pub fn varnode(&self) -> sleigh::Varnode {
        let v = Varnode::from_id(self.ctx, self.id);
        sleigh::Varnode::new(v.space().id, v.address() as u64, v.size())
    }

    /// This location as a slice of the widest architectural register that
    /// wholly contains it, keyed by [`RegisterId`] — what the
    /// specification's generated constants are
    /// (`sleigh_precompile::x64::regs::RAX`). `None` for a location no
    /// register contains, which includes every varnode outside a register
    /// space. See [`CompiledSpec::enclosing_register`].
    pub fn enclosing_register(&self) -> Option<RegisterSlice> {
        self.spec.enclosing_register(self.varnode())
    }

    /// Every architectural register sharing a byte with this location, in
    /// offset order. See [`CompiledSpec::overlapping_registers`].
    pub fn overlapping_registers(&self) -> impl Iterator<Item = RegisterId> + 'v {
        self.spec.overlapping_registers(self.varnode())
    }
}

/// One exit of a scratch instruction.
pub struct ScratchExit<'v> {
    exit: &'v Exit,
    site: ScratchInsn<'v>,
}

impl<'v> ScratchExit<'v> {
    /// The operation control leaves from.
    pub fn site(&self) -> ScratchInsn<'v> {
        self.site
    }

    pub fn arm(&self) -> ExitArm {
        self.exit.arm()
    }

    pub fn kind(&self) -> &'v ExitKind {
        self.exit.kind()
    }

    pub fn is_conditional(&self) -> bool {
        self.exit.is_conditional()
    }
}

/// A block of a scratch instruction. Two views are equal when they show the
/// same block of the same session: ids are recycled, so an id alone does not
/// name a block across sessions.
#[derive(Clone, Copy)]
pub struct ScratchBlock<'v> {
    source: Source<'v>,
    spec: &'v CompiledSpec,
    id: BlockId,
}

impl PartialEq for ScratchBlock<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.source.ctx(), other.source.ctx()) && self.id == other.id
    }
}

impl Eq for ScratchBlock<'_> {}

impl std::hash::Hash for ScratchBlock<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(self.source.ctx(), state);
        self.id.hash(state);
    }
}

impl std::fmt::Debug for ScratchBlock<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ScratchBlock({:?})", self.id)
    }
}

/// The instructions of a block, from either source.
enum BlockInsns<'v> {
    Store(InsnIds<'v, 'static>),
    Template(std::ops::Range<usize>),
}

impl<'v> ScratchBlock<'v> {
    /// The machine address this block stands at: the instruction's own for
    /// its entry, another instruction's for a block an exit leads to, none
    /// for a block internal to the instruction.
    pub fn address(&self) -> Option<u64> {
        match self.source {
            Source::Store(ctx) => ctx.block(self.id).address(),
            Source::Template {
                template, instance, ..
            } => template.block_address(usize::from(self.id.local), instance),
        }
    }

    /// Whether the block holds any instruction. A block an exit leads to is
    /// a placeholder, and empty.
    pub fn has_insns(&self) -> bool {
        match self.source {
            Source::Store(ctx) => ctx.block(self.id).has_insns(),
            Source::Template { template, .. } => template.block_has_ops(usize::from(self.id.local)),
        }
    }

    /// The block's instructions in order.
    pub fn instructions(&self) -> impl Iterator<Item = ScratchInsn<'v>> + 'v {
        let source = self.source;
        let spec = self.spec;
        let func = self.id.func;
        let block = usize::from(self.id.local);
        let ids = match source {
            Source::Store(ctx) => BlockInsns::Store(ctx.body(func).insn_ids(self.id.local)),
            Source::Template { template, .. } => BlockInsns::Template(template.ops_of(block)),
        };
        let template = match source {
            Source::Template { template, .. } => Some(template),
            Source::Store(_) => None,
        };
        let mut ids = ids;
        std::iter::from_fn(move || {
            let local = match &mut ids {
                BlockInsns::Store(ids) => ids.next()?,
                BlockInsns::Template(range) => {
                    let template = template.expect("a template source");
                    loop {
                        let k = range.next()?;
                        if template.op_block(k) == block {
                            break LocalInsnId::from(k);
                        }
                    }
                }
            };
            Some(ScratchInsn {
                source,
                spec,
                id: InstructionId::new(func, local),
            })
        })
    }
}

/// An instruction of a scratch block. Two views are equal when they show the
/// same instruction of the same session, as for [`ScratchBlock`].
#[derive(Clone, Copy)]
pub struct ScratchInsn<'v> {
    source: Source<'v>,
    spec: &'v CompiledSpec,
    id: InstructionId,
}

impl PartialEq for ScratchInsn<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.source.ctx(), other.source.ctx()) && self.id == other.id
    }
}

impl Eq for ScratchInsn<'_> {}

impl std::fmt::Debug for ScratchInsn<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ScratchInsn({:?})", self.id)
    }
}

impl<'v> ScratchInsn<'v> {
    /// The raw mnemonic. The operand ids inside are keys, not handles: the
    /// facade resolves none of them, and they must not be carried to another
    /// instruction. [`operands`](Self::operands) resolves them.
    pub fn mnemonic(&self) -> &'v Mnemonic {
        match self.source {
            Source::Store(ctx) => ctx.instruction(self.id).mnemonic(),
            Source::Template { template, .. } => template.op_mnemonic(usize::from(self.id.local)),
        }
    }

    /// The operand other instructions name this one's result by.
    pub fn result(&self) -> LocalValueId {
        LocalValueId::Instruction(self.id.local)
    }

    /// The operands of this instruction, in mnemonic order, resolved.
    pub fn operands(&self) -> impl Iterator<Item = ScratchOperand<'v>> + 'v {
        let this = *self;
        self.mnemonic()
            .args()
            .into_iter()
            .map(move |v| this.resolve(v))
    }

    /// The mnemonic's opcode name.
    pub fn opcode(&self) -> &'static str {
        self.mnemonic().opcode()
    }

    /// Resolves an operand *of this instruction*. Private on purpose: every
    /// caller is one of the accessors above, which take the id from the live
    /// mnemonic, so no id from another instruction or epoch reaches the
    /// interner through here.
    fn resolve(&self, v: LocalValueId) -> ScratchOperand<'v> {
        match v {
            LocalValueId::Literal(id) => match self.source {
                Source::Store(ctx) => {
                    let literal = &ctx.shared.values.literals[id];
                    ScratchOperand::Const {
                        value: ctx.get_literal_value(id),
                        size: ctx.shared.types.size_of(literal.type_id),
                    }
                }
                Source::Template {
                    template, instance, ..
                } => {
                    let (value, size) = template.literal_at(usize::from(id), instance);
                    ScratchOperand::Const { value, size }
                }
            },
            LocalValueId::Varnode(id) => ScratchOperand::Varnode(ScratchVarnode {
                ctx: self.source.ctx(),
                spec: self.spec,
                id,
            }),
            LocalValueId::Instruction(local) => ScratchOperand::Result(ScratchInsn {
                source: self.source,
                spec: self.spec,
                id: InstructionId::new(self.id.func, local),
            }),
            LocalValueId::Temp(temp) => ScratchOperand::Temp(temp),
            LocalValueId::BasicBlock(local) => ScratchOperand::Block(self.block_view(local)),
            other => ScratchOperand::Other(other),
        }
    }

    /// The instruction's block.
    pub fn block(&self) -> ScratchBlock<'v> {
        let local = match self.source {
            Source::Store(ctx) => {
                qcode::value::Instruction::from_id(ctx, self.id)
                    .parent()
                    .expect("a scratch instruction is in a block")
                    .id
                    .local
            }
            Source::Template { template, .. } => {
                LocalBlockId::from(template.op_block(usize::from(self.id.local)))
            }
        };
        self.block_view(local)
    }

    fn block_view(&self, local: qcode::value::LocalBlockId) -> ScratchBlock<'v> {
        ScratchBlock {
            source: self.source,
            spec: self.spec,
            id: BlockId::new(self.id.func, local),
        }
    }

    /// The target of an unconditional branch.
    pub fn branch_target(&self) -> Option<ScratchBlock<'v>> {
        match self.mnemonic() {
            Mnemonic::Branch(branch) => Some(self.block_view(branch.target)),
            _ => None,
        }
    }

    /// The taken and not-taken targets of a conditional branch.
    pub fn cbranch_targets(&self) -> Option<(ScratchBlock<'v>, ScratchBlock<'v>)> {
        match self.mnemonic() {
            Mnemonic::CBranch(cbranch) => Some((
                self.block_view(cbranch.success_block),
                self.block_view(cbranch.failure_block),
            )),
            _ => None,
        }
    }

    /// The entry address of a direct call's callee, when it has one.
    pub fn callee_address(&self) -> Option<u64> {
        let callee = match self.mnemonic() {
            Mnemonic::Call(call) => call.target,
            Mnemonic::TailCall(call) => call.target,
            _ => return None,
        };
        match (callee, self.source) {
            (Callee::Real(function), Source::Store(ctx)) => {
                FunctionBody::from_id(ctx, function).address()
            }
            (
                Callee::Minted(slot),
                Source::Template {
                    template, instance, ..
                },
            ) => template.callee_address(slot as usize, instance),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::lift::ExitKind;

    fn lifter() -> SleighLifter<'static> {
        SleighLifter::new(sleigh_precompile::x64::spec())
    }

    fn session_of<'a, 'l, 'spec>(
        lifted: &'a ScratchLifted<'_, 'l, 'spec, '_>,
    ) -> &'a ScratchSession<'l, 'spec> {
        lifted.session
    }

    #[test]
    fn a_session_accumulates_a_function_and_hands_over_its_context() {
        let lifter = lifter();
        let mut session = LiftSession::new(&lifter, Host::At(0x1000));
        let first = session.lift(0x1000, b"\x48\x89\xd8").unwrap();
        let second = session.lift(0x1003, b"\x74\x05").unwrap();
        assert_eq!(first.next_address(), second.address());
        assert_eq!(first.entry().func, session.function());
        assert_eq!(second.entry().func, session.function());
        let ctx = session.into_context().unwrap();
        assert_eq!(
            FunctionBody::from_id(&ctx, first.entry().func).address(),
            Some(0x1000)
        );
        assert_eq!(
            ctx.function(first.entry().func).root_id(),
            Some(first.entry().local)
        );
        assert!(ctx.block(second.entry()).has_insns());
    }

    #[test]
    fn a_session_continues_in_an_existing_context() {
        let lifter = lifter();
        let ctx = LiftSession::new(&lifter, Host::At(0x1000))
            .into_context()
            .unwrap();
        let mut session = LiftSession::in_context(&lifter, ctx, Host::At(0x1000)).unwrap();
        let lifted = session.lift(0x1000, b"\xc3").unwrap();
        assert_eq!(lifted.exits()[0].kind(), &ExitKind::Return);
        assert_eq!(session.context().functions().count(), 1);

        let other = SleighLifter::new(sleigh_precompile::x86::spec()).new_context();
        assert_eq!(
            LiftSession::in_context(&lifter, other, Host::Anonymous).err(),
            Some(LiftError::IncompatibleContext)
        );
    }

    #[test]
    fn a_poisoned_session_refuses_to_lift_or_publish() {
        let lifter = lifter();
        let mut session = LiftSession::new(&lifter, Host::At(0x1000));
        session.lift(0x1000, b"\x48\x89\xd8").unwrap();
        let function = session.function();

        // Poison the session's context the way a failed lift does: a
        // construction whose rollback cannot account for the function, here
        // because the emitter wrote into the fall-through placeholder the
        // first instruction left, which the construction does not own.
        {
            let ctx = session.context_mut_for_test();
            let mut addresses = AddressIndex::analyze(ctx);
            let placeholder = addresses.block_at(0x1003).unwrap();
            let mut target = LiftTarget::bind_or_refresh(ctx, &mut addresses, function).unwrap();
            let mut construction = target.begin(0x1010, 1).unwrap();
            let zero = construction.context().shared.get_const(0, 8);
            let mut emitter = construction.emitter();
            emitter.switch_to_block(placeholder);
            emitter.push_branchind(zero);
            construction.abort();
        }
        assert!(session.is_poisoned());

        // The poison is the context's, so it is there after the guard that
        // set it is gone: the session lifts nothing more...
        assert_eq!(
            session.lift(0x1010, b"\xc3").err(),
            Some(LiftError::Target(TargetError::Poisoned))
        );
        // ...its context can still be read, since the module is walkable...
        assert!(
            session
                .context()
                .block(
                    session
                        .context()
                        .function(function)
                        .root_id()
                        .map(|r| BlockId::new(function, r))
                        .unwrap()
                )
                .has_insns()
        );
        // ...but it is not published.
        assert!(matches!(
            session.into_context(),
            Err(LiftError::Target(TargetError::Poisoned))
        ));
    }

    #[test]
    fn a_scratch_lift_is_independent_of_the_previous_one() {
        let lifter = lifter();
        let mut session = ScratchSession::new(&lifter);
        let first_blocks = {
            let lifted = session.lift(0x1000, b"\xe8\x10\x00\x00\x00").unwrap();
            assert_eq!(lifted.decoded().to_string(), "CALL 4117");
            let exit = lifted.exits().next().unwrap();
            assert!(exit.kind().is_call());
            // Lowered as a jump to the callee's placeholder block, which the
            // view still shows as the call it was.
            let target = exit.site().branch_target().unwrap();
            assert_eq!(target.address(), Some(0x1015));
            assert!(!target.has_insns());
            lifted.blocks().count()
        };
        // Overlapping the previous instruction is fine: it is gone.
        let lifted = session.lift(0x1001, b"\x74\x05").unwrap();
        assert_eq!(lifted.blocks().count(), 2);
        assert_eq!(first_blocks, 1);
        let (taken, not_taken) = lifted
            .entry()
            .instructions()
            .last()
            .unwrap()
            .cbranch_targets()
            .unwrap();
        assert_eq!(taken.address(), Some(0x1008));
        assert_eq!(not_taken.address(), None);
        assert_eq!(lifted.blocks().nth(1), Some(not_taken));
        assert_eq!(session.epoch(), 2);
    }

    #[test]
    fn views_of_different_sessions_are_never_equal() {
        let lifter = lifter();
        let mut first = ScratchSession::new(&lifter);
        let mut second = ScratchSession::new(&lifter);
        let a = first.lift(0x1000, b"\x48\x89\xd8").unwrap();
        let b = second.lift(0x1000, b"\x48\x89\xd8").unwrap();
        assert_eq!(a.entry(), a.entry());
        assert_ne!(a.entry(), b.entry(), "same id, another session");
        assert_ne!(
            a.entry().instructions().next(),
            b.entry().instructions().next()
        );
        fn hash_of(block: ScratchBlock<'_>) -> u64 {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            block.hash(&mut hasher);
            hasher.finish()
        }
        assert_ne!(hash_of(a.entry()), hash_of(b.entry()));
    }

    #[test]
    fn a_failed_scratch_lift_does_not_spoil_the_next() {
        let lifter = lifter();
        let mut session = ScratchSession::new(&lifter);
        assert!(matches!(
            session.lift(0x1000, b"\x0f\xff").err(),
            Some(LiftError::Decode(_))
        ));
        let lifted = session.lift(0x1000, b"\x48\x89\xd8").unwrap();
        assert!(lifted.lifted().falls_through());
    }

    #[test]
    fn ten_thousand_scratch_lifts_retain_one_instruction() {
        let lifter = lifter();
        let mut session = ScratchSession::new(&lifter).with_literal_budget(1_024);
        // `mov rax, imm32` with a different immediate, address and, every
        // other time, a different instruction and size.
        let mut peak = 0;
        for i in 0..10_000u32 {
            let imm = i.to_le_bytes();
            let bytes: Vec<u8> = if i % 2 == 0 {
                vec![0x48, 0xc7, 0xc0, imm[0], imm[1], imm[2], imm[3]]
            } else {
                vec![0x74, imm[0]]
            };
            let lifted = session.lift(0x1000 + u64::from(i) * 7, &bytes).unwrap();
            assert!(lifted.lifted().falls_through());
            let stats = session.arena_stats();
            peak = peak.max(stats.instructions.issued);
            assert!(stats.blocks.issued <= 4, "{stats:?}");
        }
        // The per-instruction IR stays flat, and so does shared state: the
        // 5,000 distinct immediates were interned within the budget, the
        // storage rebuilt whenever it was exceeded, and no varnode was added.
        assert!(peak < 32, "{peak} instructions for one guest instruction");
        assert!(
            session.interned_literals() <= 1_024 + 8,
            "{}",
            session.interned_literals()
        );
        assert!(session.rebuilds() >= 4, "{}", session.rebuilds());
        assert!(session.arena_stats().blocks.issued <= 4);
    }

    #[test]
    fn a_recycled_constant_id_is_never_a_way_to_a_value() {
        let lifter = lifter();
        // Budget 0: the storage is rebuilt after every instruction that
        // interned a constant, so the next one reuses the first free id.
        let mut session = ScratchSession::new(&lifter).with_literal_budget(0);
        fn store_of<'v>(lifted: &'v ScratchLifted<'_, '_, '_, '_>) -> ScratchInsn<'v> {
            lifted
                .entry()
                .instructions()
                .find(|insn| matches!(insn.mnemonic(), Mnemonic::Store(_)))
                .unwrap()
        }
        fn raw_id(insn: &ScratchInsn<'_>) -> LocalValueId {
            match insn.mnemonic() {
                Mnemonic::Store(st) => st.src,
                _ => unreachable!(),
            }
        }
        fn value(insn: &ScratchInsn<'_>) -> u64 {
            let operands: Vec<_> = insn.operands().collect();
            assert_eq!(operands.len(), 2);
            assert!(matches!(operands[0], ScratchOperand::Varnode(v) if v.name() == Some("RAX")));
            operands[1].as_const().unwrap()
        }

        let first_id = {
            let lifted = session
                .lift(0x1000, b"\x48\xc7\xc0\x44\x33\x22\x11")
                .unwrap();
            let store = store_of(&lifted);
            assert_eq!(value(&store), 0x1122_3344);
            raw_id(&store)
        };
        assert_eq!(session.rebuilds(), 0);

        // The same numeric id now stands for a different constant: exactly
        // the collision a resolve-by-id accessor would silently alias. The
        // facade has no such accessor — a value is only reachable by role or
        // position from the live instruction, which reports the new constant.
        let lifted = session
            .lift(0x1007, b"\x48\xc7\xc0\x88\x77\x66\x55")
            .unwrap();
        let store = store_of(&lifted);
        assert_eq!(session_of(&lifted).rebuilds(), 1);
        assert_eq!(raw_id(&store), first_id, "the id was recycled");
        assert_eq!(value(&store), 0x5566_7788);
    }
}
