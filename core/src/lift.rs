//! What lifting one machine instruction produced.
//!
//! Every instruction emitter — whichever p-code shape it consumes — returns a
//! [`Lifted`]: the blocks the instruction owns and every place control leaves
//! it. Consumers such as function discovery and per-offset analysis read that
//! instead of re-deriving it from the IR, which would mean guessing from block
//! addresses whether a branch left the instruction, and would lose facts that
//! lowering erased on purpose (a `call` lowered to a jump is still a call).
//!
//! An [`Exit`] is recorded at the p-code operation that transfers control,
//! before any lowering choice. That makes the vocabulary independent of the
//! lowering mode: structured and flat control flow report the same exits for
//! the same instruction, and only the IR they point at differs.
//!
//! The metadata describes the *original* instruction. Resolving an import,
//! reclassifying a jump as a tail call, or deciding that a callee never
//! returns is policy layered on top by the consumer; nothing here is rewritten
//! for it.
//!
//! A result is valid until the IR it describes is mutated. Consume it before
//! running passes over the function.

pub mod target;

pub use target::{Construction, LiftTarget, TargetError};

use crate::value::{BlockId, InstructionId};

/// Where control lands when a call returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Continuation {
    /// The next machine instruction, at [`Lifted::next_address`]. The
    /// instruction's p-code ended with the call, so nothing of it runs after
    /// the callee returns.
    Next,
    /// A block of this instruction: p-code follows the call, and it was
    /// lowered there.
    Block(BlockId),
}

/// The statically named destination of a direct call.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CallTarget {
    /// A machine address.
    Address(u64),
    /// A function the semantics name rather than address, such as an
    /// intrinsic or an import stub named by the specification.
    Named(Box<str>),
}

/// How control leaves an instruction at one site.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExitKind {
    /// Execution continues at the next machine instruction, at
    /// [`Lifted::next_address`]. Reaching the end of the p-code and a branch
    /// to a label placed after the last operation both come here.
    Fallthrough,
    /// A jump to a machine address. A jump to the instruction's own address
    /// is still one of these: only branches to instruction-local labels are
    /// not exits.
    Branch { target: u64 },
    /// A jump to a computed address.
    BranchInd,
    /// A call to a statically named function.
    Call {
        callee: CallTarget,
        continuation: Continuation,
    },
    /// A call to a computed address.
    CallInd { continuation: Continuation },
    /// A return to a computed address.
    Return,
}

impl ExitKind {
    /// Whether this exit is a call, of either target kind.
    pub fn is_call(&self) -> bool {
        matches!(self, Self::Call { .. } | Self::CallInd { .. })
    }

    /// The continuation of a call exit.
    pub fn continuation(&self) -> Option<Continuation> {
        match self {
            Self::Call { continuation, .. } | Self::CallInd { continuation } => Some(*continuation),
            _ => None,
        }
    }
}

/// Which successor of the site's terminator takes the exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExitArm {
    /// The site has one successor.
    Unconditional,
    /// The site is a conditional branch and this is its taken arm.
    Taken,
    /// The site is a conditional branch and this is its not-taken arm.
    NotTaken,
}

/// One place control leaves an instruction.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Exit {
    site: InstructionId,
    arm: ExitArm,
    kind: ExitKind,
}

impl Exit {
    pub fn new(site: InstructionId, arm: ExitArm, kind: ExitKind) -> Self {
        Self { site, arm, kind }
    }

    /// The emitted operation the exit leaves from: a terminator, or a
    /// structured call that the rest of the instruction continues after.
    pub fn site(&self) -> InstructionId {
        self.site
    }

    /// Which arm of the site takes this exit.
    pub fn arm(&self) -> ExitArm {
        self.arm
    }

    pub fn kind(&self) -> &ExitKind {
        &self.kind
    }

    /// Whether the exit is taken on a condition: it is one arm of a
    /// conditional branch, not the site's only successor.
    pub fn is_conditional(&self) -> bool {
        self.arm != ExitArm::Unconditional
    }
}

/// The result of lifting one machine instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lifted {
    address: u64,
    length: usize,
    entry: BlockId,
    blocks: Vec<BlockId>,
    exits: Vec<Exit>,
}

impl Lifted {
    /// The machine address the instruction was lifted at.
    pub fn address(&self) -> u64 {
        self.address
    }

    /// The instruction's encoded length in bytes.
    pub fn length(&self) -> usize {
        self.length
    }

    /// The address of the instruction after this one, where a
    /// [`Fallthrough`](ExitKind::Fallthrough) exit and a
    /// [`Next`](Continuation::Next) continuation land.
    pub fn next_address(&self) -> u64 {
        self.address + self.length as u64
    }

    /// The block control enters the instruction through.
    pub fn entry(&self) -> BlockId {
        self.entry
    }

    /// Every block the instruction's p-code was lowered into, the entry
    /// first. This is the instruction's IR: the blocks its exits lead to are
    /// other instructions' and are not listed.
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    /// Every place control leaves the instruction, in emission order.
    ///
    /// An instruction with no exits never leaves its own blocks: its p-code
    /// loops, as `hlt` does. That is what the IR says, and it is reported as
    /// such rather than invented into a fall-through.
    pub fn exits(&self) -> &[Exit] {
        &self.exits
    }

    /// Whether some path through the instruction continues at
    /// [`next_address`](Self::next_address): a
    /// [`Fallthrough`](ExitKind::Fallthrough) exit on any arm, or a direct
    /// branch there, which is how `rep` semantics leave when their count is
    /// exhausted.
    pub fn falls_through(&self) -> bool {
        self.exits.iter().any(|exit| self.exit_reaches_next(exit))
    }

    /// Whether `exit` continues at [`next_address`](Self::next_address).
    pub fn exit_reaches_next(&self, exit: &Exit) -> bool {
        match exit.kind {
            ExitKind::Fallthrough => true,
            ExitKind::Branch { target } => target == self.next_address(),
            _ => false,
        }
    }

    /// Whether the instruction calls, by address or indirectly.
    pub fn calls(&self) -> bool {
        self.exits.iter().any(|exit| exit.kind.is_call())
    }
}

/// Builds a [`Lifted`] as an emitter lowers an instruction.
///
/// The emitter reports each block it opens and each transfer at the operation
/// that makes it, as a p-code sink sees them: forward-only, with no look-ahead.
/// What follows a call is therefore only known once the next operation
/// arrives, so the recorder resolves call continuations after the fact — to
/// the block the emitter [continues in](Self::continue_in), or to the next
/// machine instruction when nothing followed.
#[derive(Debug)]
pub struct Recorder {
    lifted: Lifted,
    /// The call exit whose continuation is not yet known.
    pending_call: Option<usize>,
}

impl Recorder {
    /// Starts recording an instruction at `address` whose p-code begins in
    /// `entry`.
    pub fn new(address: u64, length: usize, entry: BlockId) -> Self {
        Self {
            lifted: Lifted {
                address,
                length,
                entry,
                blocks: vec![entry],
                exits: Vec::new(),
            },
            pending_call: None,
        }
    }

    /// Records a block the instruction owns, in the order they are opened.
    pub fn block(&mut self, block: BlockId) {
        if !self.lifted.blocks.contains(&block) {
            self.lifted.blocks.push(block);
        }
    }

    /// Records a transfer out of the instruction.
    ///
    /// A call's continuation is settled later: pass
    /// [`Continuation::Next`] and let [`continue_in`](Self::continue_in) or
    /// [`finish`](Self::finish) decide.
    pub fn exit(&mut self, site: InstructionId, arm: ExitArm, kind: ExitKind) {
        let is_call = kind.is_call();
        self.lifted.exits.push(Exit::new(site, arm, kind));
        self.pending_call = is_call.then_some(self.lifted.exits.len() - 1);
    }

    /// Records that the p-code after the last transfer was lowered into
    /// `block`. If that transfer was a call, `block` is where it returns to.
    pub fn continue_in(&mut self, block: BlockId) {
        self.block(block);
        if let Some(index) = self.pending_call.take() {
            match &mut self.lifted.exits[index].kind {
                ExitKind::Call { continuation, .. } | ExitKind::CallInd { continuation } => {
                    *continuation = Continuation::Block(block);
                }
                _ => unreachable!("only calls are pending"),
            }
        }
    }

    /// Closes the record. A call nothing followed returns to the next machine
    /// instruction.
    pub fn finish(self) -> Lifted {
        self.lifted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{FunctionId, LocalBlockId, LocalInsnId};

    fn block(index: usize) -> BlockId {
        BlockId::new(FunctionId::default(), LocalBlockId::from(index))
    }

    fn site(index: usize) -> InstructionId {
        InstructionId::new(FunctionId::default(), LocalInsnId::from(index))
    }

    #[test]
    fn a_call_nothing_follows_returns_to_the_next_instruction() {
        let mut recorder = Recorder::new(0x1000, 5, block(0));
        recorder.exit(
            site(0),
            ExitArm::Unconditional,
            ExitKind::Call {
                callee: CallTarget::Address(0x2000),
                continuation: Continuation::Next,
            },
        );
        let lifted = recorder.finish();

        assert_eq!(lifted.next_address(), 0x1005);
        assert_eq!(lifted.blocks(), &[block(0)]);
        assert_eq!(
            lifted.exits()[0].kind().continuation(),
            Some(Continuation::Next)
        );
        assert!(lifted.calls());
        assert!(!lifted.falls_through());
    }

    #[test]
    fn a_call_something_follows_continues_in_that_block() {
        let mut recorder = Recorder::new(0x1000, 5, block(0));
        recorder.exit(
            site(0),
            ExitArm::Unconditional,
            ExitKind::CallInd {
                continuation: Continuation::Next,
            },
        );
        recorder.continue_in(block(1));
        recorder.exit(site(1), ExitArm::Unconditional, ExitKind::Fallthrough);
        let lifted = recorder.finish();

        assert_eq!(lifted.blocks(), &[block(0), block(1)]);
        assert_eq!(
            lifted.exits()[0].kind().continuation(),
            Some(Continuation::Block(block(1)))
        );
        assert!(lifted.falls_through());
    }

    #[test]
    fn continuing_after_a_branch_records_only_the_block() {
        let mut recorder = Recorder::new(0x1000, 2, block(0));
        recorder.exit(site(0), ExitArm::Taken, ExitKind::Branch { target: 0x1000 });
        recorder.continue_in(block(1));
        recorder.continue_in(block(1));
        let lifted = recorder.finish();

        assert_eq!(lifted.blocks(), &[block(0), block(1)]);
        assert!(lifted.exits()[0].is_conditional());
        assert_eq!(lifted.exits().len(), 1);
    }
}
