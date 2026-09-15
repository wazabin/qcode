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

use qcode::{
    address_index::AddressIndex,
    context::Context,
    lift::{Exit, ExitArm, ExitKind, LiftTarget, Lifted, ScratchStore},
    value::{
        BlockId, FunctionBody, FunctionId, InstructionId, LiteralId, LocalValueId, Varnode,
        VarnodeId, VarnodeRef,
        insn::{Callee, Mnemonic},
    },
};
use sleigh::{ContextBytes, ContextError, Instruction, InstructionPcode};

use crate::{LiftError, SleighLifter, decode::FixedDecoder};

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
        }
    }

    /// A session continuing in `ctx`, which must be one of this lifter's
    /// specification. `host` is resolved in that context.
    pub fn in_context(
        lifter: &'l SleighLifter<'spec>,
        mut ctx: Context<'static>,
        host: Host,
    ) -> Result<Self, LiftError> {
        lifter.check_compatible(&ctx)?;
        let mut addresses = AddressIndex::analyze(&ctx);
        let function = Self::select(&mut ctx, &mut addresses, host);
        Ok(Self {
            lifter,
            decoder: FixedDecoder::new(lifter.spec()),
            ctx,
            addresses,
            function,
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

    /// Decodes the instruction at `address` from `bytes` with the session's
    /// fixed context and lifts it.
    pub fn lift(&mut self, address: u64, bytes: &[u8]) -> Result<Lifted, LiftError> {
        let instruction = self.decoder.decode(address, bytes)?;
        self.lift_decoded(&instruction)
    }

    /// Lifts an instruction the caller decoded, with whatever context it
    /// chose.
    pub fn lift_decoded(&mut self, instruction: &Instruction<'_, '_>) -> Result<Lifted, LiftError> {
        let mut target =
            LiftTarget::bind_indexed(&mut self.ctx, &mut self.addresses, self.function)?;
        self.lifter.lift_into(&mut target, instruction)
    }

    /// Lifts already-flattened p-code, for a caller inspecting or caching
    /// that intermediate.
    pub fn lift_pcode(
        &mut self,
        address: u64,
        length: usize,
        pcode: &InstructionPcode,
    ) -> Result<Lifted, LiftError> {
        let mut target =
            LiftTarget::bind_indexed(&mut self.ctx, &mut self.addresses, self.function)?;
        self.lifter
            .lift_pcode_into(&mut target, address, length, pcode)
    }

    /// Ends the session and hands over its context.
    pub fn into_context(self) -> Context<'static> {
        self.ctx
    }

    /// A session that discards each instruction before the next. See
    /// [`ScratchSession::new`].
    pub fn scratch(
        lifter: &'l SleighLifter<'spec>,
    ) -> Result<ScratchSession<'l, 'spec>, LiftError> {
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
/// # let lifter = SleighLifter::new(spec).with_flat_control_flow();
/// let mut session = ScratchSession::new(&lifter).unwrap();
/// let first = session.lift(0x1000, b"\x48\x89\xd8").unwrap();
/// let entry = first.entry();
/// let second = session.lift(0x1003, b"\x48\x89\xd8").unwrap(); // `first` is still borrowed
/// let _ = entry.address();
/// ```
///
/// While a view is alive the session itself is out of reach, so nothing can
/// change the instruction the view shows.
///
/// ```compile_fail
/// # use wazabin_qcode_sleigh::{SleighLifter, session::ScratchSession};
/// # let spec = sleigh_precompile::x64::spec();
/// # let lifter = SleighLifter::new(spec).with_flat_control_flow();
/// let mut session = ScratchSession::new(&lifter).unwrap();
/// let lifted = session.lift(0x1000, b"\x48\x89\xd8").unwrap();
/// let epoch = session.epoch(); // `lifted` still borrows the session
/// let _ = lifted.entry();
/// ```
pub struct ScratchSession<'l, 'spec> {
    lifter: &'l SleighLifter<'spec>,
    decoder: FixedDecoder<'spec>,
    store: ScratchStore,
}

impl<'l, 'spec> ScratchSession<'l, 'spec> {
    /// A scratch session for `lifter`, which must lower calls and returns as
    /// jumps: a structured call creates its callee function, and a function
    /// cannot be discarded with the instruction that created it.
    pub fn new(lifter: &'l SleighLifter<'spec>) -> Result<Self, LiftError> {
        if !lifter.flat_control_flow {
            return Err(LiftError::ScratchNeedsFlatControlFlow);
        }
        Ok(Self {
            lifter,
            decoder: FixedDecoder::new(lifter.spec()),
            store: ScratchStore::new(lifter.new_context()),
        })
    }

    /// Decodes every address with `context` instead of the specification's
    /// default.
    pub fn with_decode_context(mut self, context: ContextBytes) -> Result<Self, ContextError> {
        self.decoder = FixedDecoder::with_context(self.lifter.spec(), context)?;
        Ok(self)
    }

    /// Rebuilds the context once instructions have interned this many
    /// literals; see [`ScratchStore::with_literal_budget`].
    pub fn with_literal_budget(mut self, budget: usize) -> Self {
        self.store = std::mem::replace(
            &mut self.store,
            ScratchStore::new(self.lifter.new_context()),
        )
        .with_literal_budget(budget);
        self
    }

    /// How many instructions have been lifted and discarded. Every handle
    /// obtained before the latest lift belongs to an earlier epoch.
    pub fn epoch(&self) -> u64 {
        self.store.epoch()
    }

    /// How many times the backing context was rebuilt to bound its memory.
    pub fn rebuilds(&self) -> u64 {
        self.store.rebuilds()
    }

    /// Constants interned since the context was built or last rebuilt.
    pub fn interned_literals(&self) -> usize {
        self.store.interned_literals()
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
        let instruction = self.decoder.decode(address, bytes)?;
        self.lift_decoded(instruction)
    }

    /// Discards the previous instruction, then lifts one the caller decoded.
    pub fn lift_decoded<'s, 'b>(
        &'s mut self,
        instruction: Instruction<'spec, 'b>,
    ) -> Result<ScratchLifted<'s, 'l, 'spec, 'b>, LiftError> {
        self.store.reset();
        let mut target = self.store.target()?;
        let lifted = self.lifter.lift_into(&mut target, &instruction)?;
        Ok(ScratchLifted {
            session: self,
            instruction,
            lifted,
        })
    }
}

/// One instruction lifted by a [`ScratchSession`], readable until the next.
///
/// Every block and instruction it exposes is a view carrying this borrow, so
/// none can be kept past the instruction. Raw IR ids do appear inside the
/// mnemonics a view shows — an operand is a [`LocalValueId`] — and those are
/// fine to use as keys while the instruction is live; nothing here resolves
/// one after it.
pub struct ScratchLifted<'s, 'l, 'spec, 'b> {
    session: &'s mut ScratchSession<'l, 'spec>,
    instruction: Instruction<'spec, 'b>,
    lifted: Lifted,
}

impl<'s, 'l, 'spec, 'b> ScratchLifted<'s, 'l, 'spec, 'b> {
    fn ctx(&self) -> &Context<'static> {
        self.session.store.context()
    }

    /// The decoded instruction: its text, operands and effects, without a
    /// second decode.
    pub fn decoded(&self) -> &Instruction<'spec, 'b> {
        &self.instruction
    }

    /// The instruction's own bytes.
    pub fn bytes(&self) -> &[u8] {
        self.instruction.bytes()
    }

    pub fn address(&self) -> u64 {
        self.lifted.address()
    }

    pub fn length(&self) -> usize {
        self.lifted.length()
    }

    pub fn next_address(&self) -> u64 {
        self.lifted.next_address()
    }

    /// See [`Lifted::falls_through`].
    pub fn falls_through(&self) -> bool {
        self.lifted.falls_through()
    }

    /// See [`Lifted::calls`].
    pub fn calls(&self) -> bool {
        self.lifted.calls()
    }

    /// Every place control leaves the instruction, in emission order.
    pub fn exits(&self) -> impl Iterator<Item = ScratchExit<'_>> + '_ {
        self.lifted.exits().iter().map(|exit| ScratchExit {
            exit,
            site: ScratchInsn {
                ctx: self.ctx(),
                id: exit.site(),
            },
        })
    }

    /// The block control enters the instruction through.
    pub fn entry(&self) -> ScratchBlock<'_> {
        ScratchBlock {
            ctx: self.ctx(),
            id: self.lifted.entry(),
        }
    }

    /// The instruction's blocks, the entry first.
    pub fn blocks(&self) -> impl Iterator<Item = ScratchBlock<'_>> + '_ {
        self.lifted.blocks().iter().map(|&id| ScratchBlock {
            ctx: self.ctx(),
            id,
        })
    }

    /// The value of an interned constant an operand names.
    pub fn literal(&self, id: LiteralId) -> u64 {
        self.ctx().get_literal_value(id)
    }

    /// The register or memory location an operand names.
    pub fn varnode(&self, id: VarnodeId) -> VarnodeRef<'static, '_> {
        Varnode::from_id(self.ctx(), id)
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
/// same block.
#[derive(Clone, Copy)]
pub struct ScratchBlock<'v> {
    ctx: &'v Context<'static>,
    id: BlockId,
}

impl PartialEq for ScratchBlock<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ScratchBlock<'_> {}

impl std::hash::Hash for ScratchBlock<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl std::fmt::Debug for ScratchBlock<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ScratchBlock({:?})", self.id)
    }
}

impl<'v> ScratchBlock<'v> {
    /// The machine address this block stands at: the instruction's own for
    /// its entry, another instruction's for a block an exit leads to, none
    /// for a block internal to the instruction.
    pub fn address(&self) -> Option<u64> {
        self.ctx.block(self.id).address
    }

    /// Whether the block holds any instruction. A block an exit leads to is
    /// a placeholder, and empty.
    pub fn has_insns(&self) -> bool {
        self.ctx.block(self.id).has_insns()
    }

    /// The block's instructions in order.
    pub fn instructions(&self) -> impl Iterator<Item = ScratchInsn<'v>> + 'v {
        let ctx = self.ctx;
        let func = self.id.func;
        ctx.body(func)
            .insn_ids(self.id.local)
            .map(move |local| ScratchInsn {
                ctx,
                id: InstructionId::new(func, local),
            })
    }
}

/// An instruction of a scratch block.
#[derive(Clone, Copy)]
pub struct ScratchInsn<'v> {
    ctx: &'v Context<'static>,
    id: InstructionId,
}

impl PartialEq for ScratchInsn<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ScratchInsn<'_> {}

impl std::fmt::Debug for ScratchInsn<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ScratchInsn({:?})", self.id)
    }
}

impl<'v> ScratchInsn<'v> {
    pub fn mnemonic(&self) -> &'v Mnemonic {
        self.ctx.instruction(self.id).mnemonic()
    }

    /// The operand other instructions name this one's result by.
    pub fn result(&self) -> LocalValueId {
        LocalValueId::Instruction(self.id.local)
    }

    /// The instruction's block.
    pub fn block(&self) -> ScratchBlock<'v> {
        let block = qcode::value::Instruction::from_id(self.ctx, self.id)
            .parent()
            .expect("a scratch instruction is in a block");
        ScratchBlock {
            ctx: self.ctx,
            id: block.id,
        }
    }

    fn block_view(&self, local: qcode::value::LocalBlockId) -> ScratchBlock<'v> {
        ScratchBlock {
            ctx: self.ctx,
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
        match callee {
            Callee::Real(function) => FunctionBody::from_id(self.ctx, function).address(),
            Callee::Minted(_) => None,
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

    #[test]
    fn a_session_accumulates_a_function_and_hands_over_its_context() {
        let lifter = lifter();
        let mut session = LiftSession::new(&lifter, Host::At(0x1000));
        let first = session.lift(0x1000, b"\x48\x89\xd8").unwrap();
        let second = session.lift(0x1003, b"\x74\x05").unwrap();
        assert_eq!(first.next_address(), second.address());
        assert_eq!(first.entry().func, session.function());
        assert_eq!(second.entry().func, session.function());
        let ctx = session.into_context();
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
        let ctx = LiftSession::new(&lifter, Host::At(0x1000)).into_context();
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
    fn a_scratch_session_needs_flat_control_flow() {
        let lifter = lifter();
        assert_eq!(
            ScratchSession::new(&lifter).err(),
            Some(LiftError::ScratchNeedsFlatControlFlow)
        );
    }

    #[test]
    fn a_scratch_lift_is_independent_of_the_previous_one() {
        let lifter = lifter().with_flat_control_flow();
        let mut session = ScratchSession::new(&lifter).unwrap();
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
    fn a_failed_scratch_lift_does_not_spoil_the_next() {
        let lifter = lifter().with_flat_control_flow();
        let mut session = ScratchSession::new(&lifter).unwrap();
        assert!(matches!(
            session.lift(0x1000, b"\x0f\xff").err(),
            Some(LiftError::Decode(_))
        ));
        let lifted = session.lift(0x1000, b"\x48\x89\xd8").unwrap();
        assert!(lifted.falls_through());
    }

    #[test]
    fn ten_thousand_scratch_lifts_retain_one_instruction() {
        let lifter = lifter().with_flat_control_flow();
        let mut session = ScratchSession::new(&lifter)
            .unwrap()
            .with_literal_budget(1000);
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
            assert!(lifted.falls_through());
            let stats = session.arena_stats();
            peak = peak.max(stats.instructions.issued);
            assert!(stats.blocks.issued <= 4, "{stats:?}");
            assert!(session.interned_literals() <= 1001);
        }
        assert!(peak < 32, "{peak} instructions for one guest instruction");
        assert!(session.rebuilds() > 0);
    }
}
