//! The run loop: a machine that owns its code, discovers more of it as the
//! guest reaches it, and stops with a reason instead of an error.
//!
//! Two things separate this from driving [`StandaloneEmulator`] directly.
//!
//! **The context is owned, not borrowed.** Lifting new code mutates the module,
//! which a `&Context` cannot allow. This is why the VM is built on
//! [`StandaloneEmulator`] — the lifetime-free emulator that takes `&Context` per
//! call — rather than on `Emulator<'ctx>`, which would pin the module for as
//! long as the machine exists.
//!
//! **Stopping is a value.** A guest that reads unmapped memory has not broken
//! the emulator; it has taken a fault, which a harness may want to report,
//! resume from, or count as a crash. So the loop returns [`VmExit`] and leaves
//! the machine intact and inspectable.

use qcode::value::LocalInsnId;
use qcode::{
    address_index::{AddressIndex, AddressTarget},
    context::Context,
    value::{
        BasicBlock, BlockId, InstructionId, ValueId,
        insn::{Mnemonic, PCodeOpId, VM_INTERRUPT},
    },
};
use qcode_emulator::{EmulatorErrorKind, EmulatorMemory, SizedValue, StandaloneEmulator};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    hook::{AddressHook, BlockEntryHook, WriteWatch},
    inject::CodeInjector,
    memory::{MemorySnapshot, VmMemory},
    mmu::MemFault,
    stats::Stats,
    table::{
        Callback, CodeRangeHook, HookAction, HookId, HookTable, InsnAction, MemAccess, ReadWatch,
    },
};

/// Why a lifting attempt failed.
#[derive(Debug, Clone)]
pub enum CodeError {
    /// The instruction bytes could not be fetched.
    Fault(MemFault),
    /// The bytes were fetched but did not decode, or did not lift.
    Decode(Box<str>),
}

/// Supplies code the machine has not seen yet.
///
/// Kept as a trait so the VM does not depend on SLEIGH: a decoder is a policy
/// choice (which specification, which variant), and a test wants to hand over
/// blocks without compiling one. An implementation reads instruction bytes from
/// the [`Mmu`](crate::Mmu) — via [`read_code`](crate::Mmu::read_code), so that
/// executing a non-executable page faults at the fetch — and lowers them into
/// `ctx`.
pub trait CodeSource {
    /// Lifts the code at `addr` into `ctx`.
    ///
    /// Returning `Ok(())` asserts that a block starting at `addr` now exists;
    /// the VM re-resolves the address itself rather than trusting a returned id,
    /// so a source is free to lift a whole run of instructions at once.
    ///
    /// `index` is the machine's live address lookup, and the implementation must
    /// keep it current as it adds blocks — every lifting entry point takes one
    /// for exactly this reason. Rebuilding it per instruction instead is
    /// quadratic in the size of the module discovered so far.
    ///
    /// `stats` is the machine's own counters: an implementation records the
    /// time it spends fetching and decoding there, so a benchmark can separate
    /// translation cost from interpretation cost.
    fn lift(
        &mut self,
        ctx: &mut Context<'static>,
        memory: &VmMemory,
        index: &mut AddressIndex,
        addr: u64,
        stats: &mut Stats,
    ) -> Result<(), CodeError>;
}

/// An alternative way to execute a block's body.
///
/// The interpreter is always present and always correct; an executor is an
/// *optimisation* that may decline any block for any reason, in which case the
/// interpreter runs it unchanged. That is what lets a backend be partial: a JIT
/// need only handle the shapes it handles well.
///
/// An executor runs the block's body, not its terminator. Control flow, block
/// parameters and call semantics stay in one implementation.
///
/// It is handed the whole emulator rather than just its memory because the
/// interpreter still has to run the terminator, and a terminator reads
/// operands — a `cbranch` condition, a branch's block arguments. Those are
/// values the body produced, so an executor that keeps them somewhere other
/// than the interpreter's value table must put them back before returning.
pub trait BlockExecutor {
    /// Runs everything in `block` except its terminator.
    ///
    /// `Ok(None)` means "not mine" and is not an error — the caller falls back
    /// to the interpreter.
    ///
    /// On `Ok(Some(_))` every value the terminator of [`Executed::block`] reads
    /// must be readable from `emu.insn_values`, exactly as if the interpreter
    /// had run that body.
    ///
    /// `chain` lets the executor run on past `block` into successors it also
    /// handles, instead of handing control back after one. Deciding a branch
    /// itself is how an executor keeps control inside its own code rather than
    /// paying a round trip per block. The caller withholds it when something
    /// needs to observe every block — a breakpoint is set, say — because blocks
    /// crossed this way are never offered to the interpreter.
    ///
    /// `start` is the body index to begin at. It is 0 when a block is
    /// entered, and the instruction after an interrupting op when the
    /// interpreter has run the block up to there and hands the rest over.
    /// Results the interpreter already computed for the block are in
    /// `emu.insn_values`; an executor may read them or decline.
    fn run_block(
        &mut self,
        ctx: &Context<'_>,
        emu: &mut StandaloneEmulator<VmMemory>,
        block: BlockId,
        start: usize,
        chain: bool,
    ) -> Result<Option<Executed>, EmulatorErrorKind>;

    /// Forgets everything derived from blocks that no longer exist in the
    /// form they were translated from.
    ///
    /// Called when the guest has written over lifted code: the blocks covering
    /// it are emptied and lifted again from the new bytes, and an executor
    /// keyed by block id would otherwise serve the translation of the old
    /// bytes — the id, and often even the instruction count, survive.
    fn invalidate(&mut self) {}
}

/// Where an executor left the machine.
#[derive(Debug, Clone, Copy)]
pub struct Executed {
    /// The block the interpreter continues in. With chaining this is the
    /// last of several, not the one that was asked for.
    pub block: BlockId,
    /// The body index of that block the interpreter continues from: its
    /// terminator, or an interrupting op the executor stopped short of.
    pub body: usize,
    /// Operations retired across every block run, for accounting.
    pub retired: u64,
}

/// Why the machine stopped.
#[derive(Debug, Clone)]
pub enum VmExit {
    /// The step budget ran out. The machine is resumable.
    InstructionLimit,
    /// Execution reached an address with a breakpoint on it. The breakpoint
    /// instruction has *not* been executed.
    Breakpoint(u64),
    /// A memory access failed.
    Fault(MemFault),
    /// Code at this address could not be lifted.
    Unlifted { addr: u64, error: CodeError },
    /// The machine stopped *at* a user p-code operation for the host to act:
    /// an explicit `vm.interrupt`, or an op the interpreter has no semantics
    /// for (`syscall`, `cpuid`, `rdtsc`, ...). Everything before the op in
    /// its block has retired and nothing after it has run. The machine stays
    /// there until [`Vm::resume`] supplies the op's effect; running again
    /// without resuming reports the same interrupt.
    Interrupt(Interrupt),
    /// A hook registered through the [table](crate::table) asked the run to
    /// stop. For a code or memory hook the machine is before the hooked
    /// guest instruction, and running again executes it without notifying
    /// the hook again. For an instruction hook the user op is still pending,
    /// as [`Interrupt`] describes, until the caller resumes it.
    HookStop(HookId),
    /// The interpreter reported something the VM does not model as a guest
    /// event — a malformed block, an internal failure.
    Error(Box<str>),
}

/// What kind of operation stopped the machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterruptKind {
    /// An explicit [`VM_INTERRUPT`] placed in the IR by a hook or an injector.
    /// `code` is its first operand.
    Explicit { code: u64 },
    /// An architecture user op with no interpreter semantics, named as in the
    /// SLEIGH specification.
    Intrinsic { op: PCodeOpId, name: Box<str> },
}

/// A stop at a user p-code operation, with what the host needs to act on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interrupt {
    pub kind: InterruptKind,
    /// The operation's instruction: where [`Vm::resume`] files its result.
    pub insn: InstructionId,
    /// The width in bytes of the result the operation declares, or 0 when it
    /// produces nothing.
    pub size: usize,
    /// The operation's operands as read at the stop — after `code` for an
    /// explicit interrupt. `None` where an operand is wider than 64 bits or
    /// could not be read.
    pub args: Vec<Option<u64>>,
    /// The guest address of the instruction that lifted to the operation.
    pub pc: Option<u64>,
}

/// A machine captured by [`Vm::snapshot`].
///
/// Opaque; hand it back to [`Vm::restore`]. Cheap to hold: guest RAM is
/// shared with the live machine until either side writes.
#[derive(Clone)]
pub struct VmSnapshot {
    memory: MemorySnapshot,
    block: BlockId,
    /// The op at that position — its id and guest address — so it can be
    /// found again after its block changes shape.
    op: Option<(LocalInsnId, Option<u64>)>,
    /// The guest instruction the position starts, if it starts one: where
    /// the machine can be put by lifting alone.
    boundary: Option<u64>,
    /// Values produced before the position that the rest of the block reads.
    live: Vec<(InstructionId, SizedValue)>,
}

/// Why [`Vm::snapshot`] refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotError {
    /// The machine is stopped at an interrupt that has not been resumed.
    Interrupted,
}

/// Why [`Vm::restore`] could not put the machine back where it was. Memory
/// and registers are restored either way.
#[derive(Debug, Clone)]
pub enum RestoreError {
    /// The snapshot was taken part-way through a guest instruction, and the
    /// block holding it has since been replaced.
    PositionLost,
    /// The instruction the snapshot starts could not be lifted again.
    Unlifted { addr: u64, error: CodeError },
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Interrupted => f.write_str("the machine is stopped at an unresumed interrupt"),
        }
    }
}

impl std::fmt::Display for RestoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PositionLost => f.write_str("the snapshot's position no longer exists"),
            Self::Unlifted { addr, error } => write!(f, "cannot lift {addr:#x}: {error:?}"),
        }
    }
}

impl std::error::Error for SnapshotError {}
impl std::error::Error for RestoreError {}

/// Why [`Vm::resume`] refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResumeError {
    /// The machine is not stopped at an interrupt.
    NotInterrupted,
    /// The operation declares a result of this many bytes and none was given.
    ResultRequired { size: usize },
}

impl std::fmt::Display for ResumeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotInterrupted => write!(f, "the machine is not stopped at an interrupt"),
            Self::ResultRequired { size } => {
                write!(f, "the interrupted operation needs a {size}-byte result")
            }
        }
    }
}

impl std::error::Error for ResumeError {}

/// A machine: an owned module, a memory, and a position in the code.
pub struct Vm<S> {
    ctx: Context<'static>,
    emu: StandaloneEmulator<VmMemory>,
    source: S,
    /// Counters and phase timings for this run.
    pub stats: Stats,
    /// An optional faster path for block bodies. `None` means the interpreter
    /// executes everything, which is always a valid way to run.
    executor: Option<Box<dyn BlockExecutor>>,
    /// Set by a lift that folded the block it filled into a predecessor, which
    /// leaves the machine already positioned. Taken by the step that asked for
    /// the lift.
    absorbed_into: Option<BlockId>,
    /// Whether freshly lifted blocks get a cleanup round.
    ///
    /// Lifting one machine instruction emits every side effect the
    /// specification describes, including flag computations the surrounding
    /// code never reads. Removing the ones with no users at all is sound
    /// block-locally — an instruction with no users cannot be observed — and is
    /// paid once per block instead of on every execution of it.
    pub optimize: bool,
    /// The block that has grown by absorption and not been cleaned since.
    ///
    /// Absorption folds a straight-line run one guest instruction at a time,
    /// and cleaning the whole enlarged block after each one is quadratic in the
    /// length of the run — which on unrolled code is the dominant cost of
    /// translation. The cleanup is deferred to the point the block is next
    /// entered at its first instruction, by which time the run has stopped
    /// growing and one pass does the work of all of them.
    ///
    /// At most one: absorption extends one run at a time, so a *different*
    /// block being absorbed into means the previous run has stopped growing
    /// and can be cleaned right there. Waiting for it to be entered again
    /// instead would let compiled code be built from the uncleaned form — and
    /// worse, would clean it in an interpreted run but not in a chained
    /// compiled one, leaving the two strategies running different QCode.
    dirty: Option<BlockId>,
    breakpoints: FxHashSet<u64>,
    /// The interrupt the machine is stopped at, until it is resumed.
    pending: Option<Interrupt>,
    /// Set by a resume: the machine is part-way into a block, and the
    /// executor gets one chance to take the rest of it.
    offer_rest: bool,
    /// Code rewriters, run over each block before it is first entered and
    /// again when it grows. See [`crate::inject`].
    injectors: Vec<Box<dyn CodeInjector>>,
    /// The injector set each block was last rewritten by, so registering an
    /// injector reaches blocks already lifted the next time they are entered.
    injected: FxHashMap<BlockId, u64>,
    /// Bumped by every registration.
    generation: u64,
    /// Callbacks the run calls itself. See [`crate::table`].
    table: HookTable<S>,
    /// The guest instruction a write into lifted code was seen under, until
    /// the machine leaves it. See [`Self::evict_written_code`].
    code_written_at: Option<Option<u64>>,
}

impl<S: CodeSource> Vm<S> {
    /// Builds a machine positioned at `entry`.
    pub fn new(ctx: Context<'static>, entry: BlockId, source: S) -> Self {
        let mut emu = StandaloneEmulator::<VmMemory>::new_in(entry);
        emu.memory.configure_spaces(&ctx);
        Self {
            ctx,
            emu,
            source,
            optimize: true,
            stats: Stats::default(),
            executor: None,
            absorbed_into: None,
            dirty: None,
            breakpoints: FxHashSet::default(),
            pending: None,
            offer_rest: false,
            injectors: Vec::new(),
            injected: FxHashMap::default(),
            generation: 0,
            table: HookTable::default(),
            code_written_at: None,
        }
    }

    /// Builds a machine positioned at a guest address, lifting the entry block
    /// if the module does not already contain it.
    pub fn at_address(
        mut ctx: Context<'static>,
        addr: u64,
        mut source: S,
        memory: VmMemory,
    ) -> Result<Self, CodeError> {
        // Built once here and handed to the machine, which keeps it current
        // from then on.
        let mut index = AddressIndex::analyze(&ctx);
        let mut stats = Stats::default();
        if resolve(&ctx, &index, addr).is_none() {
            stats.lifts += 1;
            source.lift(&mut ctx, &memory, &mut index, addr, &mut stats)?;
        }
        let entry = resolve(&ctx, &index, addr).ok_or_else(|| {
            CodeError::Decode(format!("no block at {addr:#x} after lifting").into())
        })?;
        let mut vm = Self::new(ctx, entry, source);
        vm.emu.memory = memory;
        vm.emu.memory.mmu.mark_code(addr, MAX_INSN_LEN);
        vm.emu.memory.configure_spaces(&vm.ctx);
        vm.emu.set_address_index(index);
        vm.stats = stats;
        Ok(vm)
    }

    /// Installs an alternative executor for block bodies, replacing any
    /// previous one. Purely an optimisation: removing it changes speed, not
    /// behaviour.
    pub fn set_block_executor(&mut self, executor: Box<dyn BlockExecutor>) {
        self.executor = Some(executor);
    }

    pub fn clear_block_executor(&mut self) {
        self.executor = None;
    }

    pub fn context(&self) -> &Context<'static> {
        &self.ctx
    }

    pub fn memory(&self) -> &VmMemory {
        &self.emu.memory
    }

    pub fn memory_mut(&mut self) -> &mut VmMemory {
        &mut self.emu.memory
    }

    /// The emulator underneath, for register access and harness seeding.
    pub fn emulator(&mut self) -> &mut StandaloneEmulator<VmMemory> {
        &mut self.emu
    }

    /// The guest address of the block about to execute, if it has one.
    pub fn pc(&self) -> Option<u64> {
        BasicBlock::from_id(&self.ctx, self.emu.block).address()
    }

    pub fn add_breakpoint(&mut self, addr: u64) -> bool {
        self.breakpoints.insert(addr)
    }

    pub fn remove_breakpoint(&mut self, addr: u64) -> bool {
        self.breakpoints.remove(&addr)
    }

    /// Registers a code rewriter. It runs over every block the machine enters
    /// from now on, including blocks lifted before this call, before the
    /// block executes or is compiled.
    pub fn add_injector(&mut self, injector: Box<dyn CodeInjector>) {
        self.injectors.push(injector);
        self.generation += 1;
    }

    /// Registers a [`Hook`](crate::hook::Hook): a code rewriter that picks
    /// its sites and emits through an [`Emitter`](crate::hook::Emitter).
    pub fn add_hook(&mut self, hook: impl crate::hook::Hook + 'static) {
        self.add_injector(Box::new(crate::hook::HookInjector::new(hook)));
    }

    // ---- Unicorn-shaped hooks. See [`crate::table`].

    /// Calls `callback` with the address at the entry of every block in
    /// `begin..=end` (anywhere when `begin > end`).
    pub fn hook_block(
        &mut self,
        begin: u64,
        end: u64,
        callback: impl FnMut(&mut Self, u64) -> HookAction + 'static,
    ) -> HookId {
        let id = self.table.register(Callback::Code(Box::new(callback)));
        self.add_hook(BlockEntryHook {
            begin,
            end,
            code: HookTable::<S>::code(id),
        });
        id
    }

    /// Calls `callback` before every guest instruction in `begin..=end`.
    pub fn hook_code(
        &mut self,
        begin: u64,
        end: u64,
        callback: impl FnMut(&mut Self, u64) -> HookAction + 'static,
    ) -> HookId {
        let id = self.table.register(Callback::Code(Box::new(callback)));
        self.add_hook(CodeRangeHook {
            begin,
            end,
            code: HookTable::<S>::code(id),
        });
        id
    }

    /// Calls `callback` before the guest instruction at `addr`.
    pub fn hook_address(
        &mut self,
        addr: u64,
        callback: impl FnMut(&mut Self, u64) -> HookAction + 'static,
    ) -> HookId {
        let id = self.table.register(Callback::Code(Box::new(callback)));
        self.add_hook(AddressHook::new([addr], HookTable::<S>::code(id)));
        id
    }

    /// Calls `callback` before every store to guest memory in `begin..=end`,
    /// with the address, width and value.
    pub fn hook_mem_write(
        &mut self,
        begin: u64,
        end: u64,
        callback: impl FnMut(&mut Self, &MemAccess) -> HookAction + 'static,
    ) -> HookId {
        let id = self.table.register(Callback::Mem(Box::new(callback)));
        self.add_hook(WriteWatch {
            begin,
            end,
            code: HookTable::<S>::code(id),
        });
        id
    }

    /// Calls `callback` before every load from guest memory in `begin..=end`,
    /// with the address and width.
    pub fn hook_mem_read(
        &mut self,
        begin: u64,
        end: u64,
        callback: impl FnMut(&mut Self, &MemAccess) -> HookAction + 'static,
    ) -> HookId {
        let id = self.table.register(Callback::Mem(Box::new(callback)));
        self.add_hook(ReadWatch {
            begin,
            end,
            code: HookTable::<S>::code(id),
        });
        id
    }

    /// Calls `callback` when the guest reaches the user op called `name` —
    /// `syscall`, `rdtsc`, `cpuid_basic`, ... — to supply its effect.
    pub fn hook_insn(
        &mut self,
        name: &str,
        callback: impl FnMut(&mut Self, &Interrupt) -> InsnAction + 'static,
    ) -> HookId {
        self.table.register(Callback::Insn {
            name: Some(Box::from(name)),
            callback: Box::new(callback),
        })
    }

    /// Calls `callback` for every user op the interpreter cannot run, whatever
    /// its name.
    pub fn hook_intr(
        &mut self,
        callback: impl FnMut(&mut Self, &Interrupt) -> InsnAction + 'static,
    ) -> HookId {
        self.table.register(Callback::Insn {
            name: None,
            callback: Box::new(callback),
        })
    }

    /// Removes a hook. Interrupts it injected stay in the code and resume
    /// silently; they cost a stop and nothing more.
    pub fn hook_del(&mut self, id: HookId) -> bool {
        self.table.remove(id)
    }

    /// Handles an interrupt the table owns, or an intrinsic a hook answers.
    ///
    /// Returns `None` when the run should carry on, and the exit to return
    /// otherwise. The callbacks are moved out for the duration so they can be
    /// handed the machine; ones registered meanwhile are kept.
    fn dispatch(&mut self, interrupt: &Interrupt) -> Option<VmExit> {
        let owner = HookTable::<S>::owner(interrupt);
        let intrinsic = match &interrupt.kind {
            InterruptKind::Intrinsic { name, .. } => Some(name.clone()),
            InterruptKind::Explicit { .. } => None,
        };
        if owner.is_none() && intrinsic.is_none() {
            return Some(VmExit::Interrupt(interrupt.clone()));
        }
        let mut callbacks = std::mem::take(&mut self.table.callbacks);
        let mut outcome: Option<VmExit> = None;
        let mut handled = false;
        for (id, callback) in &mut callbacks {
            match (callback, owner, &intrinsic) {
                (Callback::Code(callback), Some(owner), _) if *id == owner => {
                    let pc = interrupt
                        .args
                        .first()
                        .copied()
                        .flatten()
                        .unwrap_or_default();
                    if callback(self, pc) == HookAction::Stop {
                        outcome = Some(VmExit::HookStop(*id));
                    }
                    handled = true;
                }
                (Callback::Mem(callback), Some(owner), _) if *id == owner => {
                    let arg = |index: usize| interrupt.args.get(index).copied().flatten();
                    let access = MemAccess {
                        pc: interrupt.pc,
                        addr: arg(0).unwrap_or_default(),
                        size: arg(1).unwrap_or_default(),
                        value: arg(2),
                    };
                    if callback(self, &access) == HookAction::Stop {
                        outcome = Some(VmExit::HookStop(*id));
                    }
                    handled = true;
                }
                (Callback::Insn { name, callback }, None, Some(op))
                    if name.as_ref().is_none_or(|name| name == op) =>
                {
                    match callback(self, interrupt) {
                        InsnAction::Handled(value) => {
                            if let Err(error) = self.resume(value) {
                                outcome = Some(VmExit::Error(error.to_string().into()));
                            }
                            handled = true;
                        }
                        InsnAction::Stop => {
                            outcome = Some(VmExit::HookStop(*id));
                            handled = true;
                        }
                        InsnAction::Unhandled => continue,
                    }
                }
                _ => continue,
            }
            if outcome.is_some() {
                break;
            }
            if intrinsic.is_some() && handled {
                // One answer per op.
                break;
            }
        }
        // Hooks registered by a callback were pushed onto the empty table.
        callbacks.append(&mut self.table.callbacks);
        self.table.callbacks = callbacks;

        if owner.is_some() {
            // The table's interrupt is a notification, delivered now (or
            // owed to a hook since deleted). Whether the run goes on or
            // stops, the machine is left before the guest instruction, not
            // at the notification: a later run must not deliver it again.
            if let Err(error) = self.resume(None) {
                return Some(VmExit::Error(error.to_string().into()));
            }
            return outcome;
        }
        if let Some(exit) = outcome {
            return Some(exit);
        }
        if !handled {
            // An intrinsic no hook answered is the caller's.
            return Some(VmExit::Interrupt(interrupt.clone()));
        }
        None
    }

    /// Runs the injectors over `block` unless the current set already has.
    ///
    /// Only over lifted code: an empty block carrying an address is a request
    /// to lift it, and its instructions arrive — and get rewritten — once it
    /// is discovered.
    fn inject(&mut self, block: BlockId) {
        if self.injectors.is_empty()
            || self.injected.get(&block) == Some(&self.generation)
            || !BasicBlock::from_id(&self.ctx, block).is_terminated()
        {
            return;
        }
        // Moved out for the duration, so the injectors can be handed the
        // module without borrowing the machine twice.
        let mut injectors = std::mem::take(&mut self.injectors);
        for injector in &mut injectors {
            injector.inject(&mut self.ctx, block);
        }
        self.injectors = injectors;
        self.injected.insert(block, self.generation);
        // The interpreter may hold this block's instruction list, and it has
        // changed.
        self.emu.invalidate_block_cache();
    }

    /// Executes one instruction, lifting code on demand if control leaves the
    /// part of the module already known.
    ///
    /// Returns `None` when the step was ordinary, and `Some(exit)` when the
    /// machine stopped for a reason worth reporting.
    pub fn step(&mut self) -> Option<VmExit> {
        // Stopped at an operation nobody has resumed: the machine has not
        // moved, and stepping it would run the op again without its effect.
        if let Some(interrupt) = &self.pending {
            return Some(VmExit::Interrupt(interrupt.clone()));
        }
        // A branch to unlifted code fails *before* the emulator moves, so the
        // address can be lifted and the same step retried. One step may need
        // more than one discovery: a direct branch lands in an empty
        // placeholder, the retry lets an executor run the whole of the block
        // it filled, and the terminator left to the interpreter can be an
        // indirect branch to code nobody has lifted either. What ends the
        // retries is the same address coming back, which means the source
        // did not produce the block it claimed to: a source bug rather than
        // a discovery step.
        let mut discovered: Option<u64> = None;
        loop {
            // Lifted code the guest has written over goes before anything is
            // built from it or run.
            if self.code_written_at.is_some()
                && let Some(exit) = self.evict_written_code()
            {
                return Some(exit);
            }
            // At its first instruction a block is between runs, which is the
            // one moment a deferred cleanup can be taken without disturbing a
            // position inside it — and it has to happen before the executor
            // looks, or compiled code gets built from uncleaned QCode.
            if self.emu.idx == 0 {
                let block = self.emu.block;
                self.clean_before_entering(block);
                self.inject(block);
            }

            // At a block's first instruction — or just past an interrupt it
            // resumed from — an installed executor may run the rest of the
            // body at once, leaving the interpreter only the terminator.
            if (self.emu.idx == 0 || self.offer_rest)
                && let Some(executor) = self.executor.as_mut()
            {
                self.offer_rest = false;
                let block = self.emu.block;
                let start = self.emu.idx;
                // Blocks the executor runs are never offered to the interpreter, so
                // it may only run past the first when nothing needs to see them.
                let chain = self.breakpoints.is_empty();
                match executor.run_block(&self.ctx, &mut self.emu, block, start, chain) {
                    Ok(Some(run)) => {
                        // The operations were retired by the executor; they are
                        // counted so throughput stays comparable between strategies.
                        self.stats.steps += run.retired;
                        self.stats.native_bodies += 1;
                        // Positioning inside a block the interpreter has not walked
                        // into invalidates its cached instruction list.
                        self.emu.invalidate_block_cache();
                        self.emu.block = run.block;
                        self.emu.idx = run.body;
                        self.note_code_write();
                    }
                    Ok(None) => {}
                    Err(kind) => {
                        let fault = self.emu.memory.take_fault();
                        return Some(match fault {
                            Some(fault) => VmExit::Fault(fault),
                            None => self.exit_for(kind),
                        });
                    }
                }
            }

            let stepped_at = self.current_guest_address();
            match self.emu.step(&self.ctx) {
                Ok(()) => {
                    self.stats.steps += 1;
                    if self.code_written_at.is_none() && self.emu.memory.mmu.code_written() {
                        self.code_written_at = Some(stepped_at);
                    }
                    return None;
                }
                Err(error) => match error.kind {
                    EmulatorErrorKind::InvalidBlockAddress(addr)
                    | EmulatorErrorKind::UnknownAddress(addr)
                        if discovered != Some(addr) =>
                    {
                        discovered = Some(addr);
                        if let Some(exit) = self.discover(addr) {
                            return Some(exit);
                        }
                    }
                    // A direct branch to code that has not been lifted does not
                    // fail to resolve: the lifter materializes the target as an
                    // *empty* block at that address, and execution walks into
                    // it. So an empty block carrying an address is a request to
                    // discover it, not a malformed-IR error.
                    EmulatorErrorKind::EmptyBlock(block) => {
                        let Some(addr) = BasicBlock::from_id(&self.ctx, block).address() else {
                            return Some(VmExit::Error(
                                EmulatorErrorKind::EmptyBlock(block).to_string().into(),
                            ));
                        };
                        if discovered == Some(addr) {
                            return Some(self.exit_for(EmulatorErrorKind::EmptyBlock(block)));
                        }
                        // The address may already be lifted: a branch back into
                        // known code still gets a fresh placeholder block in the
                        // branching instruction's own function, and lifting it
                        // again would collide with the function that owns it.
                        // Resolving first is what makes loops work.
                        let before = self.emu.block;
                        self.reposition(addr);
                        if self.emu.block != before {
                            // Already lifted: a translation-cache hit.
                            self.stats.resolves += 1;
                            continue;
                        }
                        discovered = Some(addr);
                        if let Some(exit) = self.discover(addr) {
                            return Some(exit);
                        }
                        // Lifting may place the instruction in a *new* block
                        // rather than filling the placeholder the emulator is
                        // sitting in, so the machine has to be moved onto
                        // whatever now covers the address — unless the lift
                        // folded that block into its predecessor, which
                        // positions the machine itself. `addr` is interior to
                        // the absorbing block then, and resolving it by address
                        // would land at that block's *start*.
                        if self.absorbed_into.take().is_none() {
                            self.reposition(addr);
                        }
                    }
                    EmulatorErrorKind::MemoryReadError(addr)
                    | EmulatorErrorKind::MemoryWriteError(addr) => {
                        // The backend records the precise cause; the error alone
                        // could only say that an access at this address failed.
                        let fault = self.emu.memory.take_fault().unwrap_or(MemFault {
                            kind: crate::mmu::FaultKind::ReadUnmapped,
                            addr,
                        });
                        return Some(VmExit::Fault(fault));
                    }
                    kind => return Some(self.exit_for(kind)),
                },
            }
        }
    }

    /// The exit for an interpreter error that is not a memory fault or a
    /// discovery request.
    ///
    /// A stop at a user operation is a guest event with a typed exit; anything
    /// else is reported as the error it is.
    fn exit_for(&mut self, kind: EmulatorErrorKind) -> VmExit {
        match kind {
            EmulatorErrorKind::Interrupt | EmulatorErrorKind::UnsupportedPCodeOp(_) => {
                match self.interrupt_at_position() {
                    Some(interrupt) => {
                        self.pending = Some(interrupt.clone());
                        VmExit::Interrupt(interrupt)
                    }
                    // The error named an op, but the machine is not at one:
                    // an executor stopped somewhere it should not have.
                    None => VmExit::Error(kind.to_string().into()),
                }
            }
            other => VmExit::Error(other.to_string().into()),
        }
    }

    /// Describes the user operation at the machine's position, if that is
    /// what it is stopped at.
    fn interrupt_at_position(&mut self) -> Option<Interrupt> {
        let block = self.emu.block;
        let idx = self.emu.idx;
        if !self.ctx.contains_block(block) {
            return None;
        }
        let (insn, size, pc, op, operands) = {
            let insn = BasicBlock::from_id(&self.ctx, block)
                .instructions()
                .nth(idx)?;
            let Mnemonic::PCodeOp(op) = insn.mnemonic() else {
                return None;
            };
            let operands: Vec<ValueId> =
                op.args.iter().map(|arg| arg.qualify(block.func)).collect();
            // An instruction carries the address of the guest instruction it
            // was lifted from; one built by hand may not, in which case the
            // block's own address is the best that can be said.
            let pc = insn
                .address()
                .or_else(|| BasicBlock::from_id(&self.ctx, block).address());
            (insn.id, insn.size(), pc, op.id, operands)
        };
        let name = self.ctx.shared.pcode_ops[op].clone();
        let mut args: Vec<Option<u64>> = operands
            .into_iter()
            .map(|value| self.emu.get_value(&self.ctx, value))
            .collect();
        let kind = if name.as_ref() == VM_INTERRUPT {
            let code = if args.is_empty() {
                0
            } else {
                args.remove(0).unwrap_or(0)
            };
            InterruptKind::Explicit { code }
        } else {
            InterruptKind::Intrinsic { op, name }
        };
        Some(Interrupt {
            kind,
            insn,
            size,
            args,
            pc,
        })
    }

    /// The interrupt the machine is stopped at, if any.
    pub fn pending_interrupt(&self) -> Option<&Interrupt> {
        self.pending.as_ref()
    }

    /// Supplies the effect of the operation the machine is stopped at and
    /// steps past it.
    ///
    /// `value` is the operation's result, required when it declares one
    /// ([`Interrupt::size`] is non-zero) and ignored otherwise. Any other
    /// effect — a register written by a system call, memory filled by a host
    /// service — the caller applies through [`Vm::emulator`] and
    /// [`Vm::memory_mut`] before resuming. The operation's own instruction is
    /// not run; the one after it is next.
    pub fn resume(&mut self, value: Option<u128>) -> Result<(), ResumeError> {
        let Some(interrupt) = self.pending.as_ref() else {
            return Err(ResumeError::NotInterrupted);
        };
        if interrupt.size > 0 {
            let Some(value) = value else {
                return Err(ResumeError::ResultRequired {
                    size: interrupt.size,
                });
            };
            self.emu
                .insn_values
                .insert(interrupt.insn, SizedValue::from_bits(value, interrupt.size));
        }
        self.pending = None;
        self.stats.steps += 1;
        self.emu.idx += 1;
        self.offer_rest = true;
        Ok(())
    }

    /// The guest address of the operation the machine is positioned at, if
    /// the op records one.
    fn current_guest_address(&self) -> Option<u64> {
        let block = self.ctx.block(self.emu.block);
        let &local = block.instruction_ids().get(self.emu.idx)?;
        self.ctx
            .get_insn(InstructionId::new(self.emu.block.func, local))
            .address()
    }

    /// Records, after an executor ran, that lifted code was written: the
    /// machine sits at the end of the block that wrote it, and the eviction
    /// waits for the terminator's instruction to finish.
    fn note_code_write(&mut self) {
        if self.code_written_at.is_none() && self.emu.memory.mmu.code_written() {
            self.code_written_at = Some(self.current_guest_address());
        }
    }

    /// Throws away every lifted block the guest has written over, once the
    /// instruction that wrote is complete.
    ///
    /// A guest instruction is many operations, and the store may not be its
    /// last; emptying its block part-way through would lose the rest. So the
    /// write is noted with the instruction it happened under, and acted on at
    /// the first op that belongs to a different one — which on the
    /// interpreter is the very next guest instruction, and after compiled
    /// code is the one past the block it ran. The blocks covering a written
    /// page are emptied rather than deleted, so every branch into them stays
    /// valid and each is lifted again from the new bytes when control
    /// reaches it — the same shape a split leaves behind.
    fn evict_written_code(&mut self) -> Option<VmExit> {
        let written_under = self.code_written_at?;
        let now = self.current_guest_address();
        // Still inside the instruction that wrote (or inside an op with no
        // address, which is never the first of a new instruction).
        if now.is_none() || now == written_under {
            return None;
        }
        self.code_written_at = None;
        let pages = self.emu.memory.mmu.take_code_writes()?;
        let resume = now.expect("checked above");
        if self.evict_pages(&pages) == 0 {
            return None;
        }

        // The machine's own block may be among them. Either way it continues
        // at the instruction it was about to run, from whatever now covers
        // that address.
        if self.emu.block_at_address(&self.ctx, resume).is_none()
            && let Some(exit) = self.discover(resume)
        {
            return Some(exit);
        }
        self.absorbed_into = None;
        if let Some(block) = self.emu.block_at_address(&self.ctx, resume) {
            self.emu.block = block;
            self.emu.idx = 0;
            self.emu.invalidate_block_cache();
        }
        None
    }

    /// Empties every lifted block over `pages`, and forgets what the
    /// executor and the interpreter derived from them. Returns how many.
    fn evict_pages(&mut self, pages: &FxHashSet<u64>) -> u64 {
        let in_written = |addr: u64| pages.contains(&(addr >> 12));
        let mut index = self
            .emu
            .take_address_index()
            .unwrap_or_else(|| AddressIndex::analyze(&self.ctx));
        let mut evicted = 0u64;
        for block in self.ctx.block_ids() {
            let covered = {
                let block = self.ctx.block(block);
                block.address.is_some_and(in_written)
                    || block.extra_addresses.iter().copied().any(in_written)
            };
            if !covered || self.ctx.block(block).instruction_ids().is_empty() {
                continue;
            }
            // Its own address stays indexed: an empty block at an address is
            // this module's request to lift it. Whatever it absorbed is no
            // longer anyone's, until lifting settles which block covers it.
            let absorbed = std::mem::take(&mut self.ctx.block_mut(block).extra_addresses);
            for addr in absorbed {
                index.forget(addr);
            }
            self.ctx
                .body_mut(block.func)
                .clear_block_instructions(block);
            self.injected.remove(&block);
            if self.dirty == Some(block) {
                self.dirty = None;
            }
            evicted += 1;
        }
        self.emu.set_address_index(index);
        self.stats.evicted += evicted;
        if evicted == 0 {
            return 0;
        }
        if let Some(executor) = self.executor.as_mut() {
            executor.invalidate();
        }
        self.emu.invalidate_block_cache();
        evicted
    }

    // ---- Snapshots.

    /// The guest instruction the machine is about to start, if it is
    /// between instructions.
    ///
    /// A guest instruction is several operations, and a step budget can
    /// stop the machine between two of them. A snapshot taken there can
    /// only be restored while its block survives unchanged (see
    /// [`restore`](Self::restore)); one taken here can always be restored.
    /// [`step`](Self::step) the machine until this answers to get there.
    pub fn instruction_boundary(&self) -> Option<u64> {
        let block = self.emu.block;
        let idx = self.emu.idx;
        let ids = self.ctx.block(block).instruction_ids();
        let address_of = |local: LocalInsnId| {
            self.ctx
                .get_insn(InstructionId::new(block.func, local))
                .address()
        };
        // The op here carries a different address from the op before it, or
        // is the first of a block that starts at an address of its own.
        match (idx, ids.get(idx)) {
            (0, _) => self.ctx.block(block).address,
            (_, Some(&local)) => {
                address_of(local).filter(|&addr| address_of(ids[idx - 1]) != Some(addr))
            }
            _ => None,
        }
    }

    /// Captures the machine: memory, registers, and where it is.
    ///
    /// Guest RAM is shared with the snapshot copy-on-write, so this costs
    /// one pointer per resident page however large the image; see
    /// [`Mmu::snapshot`](crate::Mmu::snapshot). Lifted code is *not* copied:
    /// it is a function of the bytes it was lifted from, and a restore that
    /// changes those bytes throws the affected blocks away.
    ///
    /// Refused while the machine is stopped at an interrupt nobody has
    /// resumed: that stop is the caller's, and a snapshot would carry it.
    pub fn snapshot(&mut self) -> Result<VmSnapshot, SnapshotError> {
        if self.pending.is_some() {
            return Err(SnapshotError::Interrupted);
        }
        let block = self.emu.block;
        let idx = self.emu.idx;
        let ids = self.ctx.block(block).instruction_ids();
        let address_of = |local: LocalInsnId| {
            self.ctx
                .get_insn(InstructionId::new(block.func, local))
                .address()
        };
        let op = ids.get(idx).map(|&local| (local, address_of(local)));
        let boundary = self.instruction_boundary();
        // Values from earlier in the block that the rest of it reads. Only
        // those: the value table is sized by the module, not by the block,
        // and the snapshot must not be.
        let before: FxHashSet<LocalInsnId> = ids[..idx.min(ids.len())].iter().copied().collect();
        let mut live = Vec::new();
        let mut seen = FxHashSet::default();
        for &local in &ids[idx.min(ids.len())..] {
            let insn = self.ctx.get_insn(InstructionId::new(block.func, local));
            for operand in insn.operands() {
                let ValueId::Instruction(id) = operand else {
                    continue;
                };
                if !before.contains(&id.local) || !seen.insert(id) {
                    continue;
                }
                if let Some(value) = self.emu.insn_values.get(&id) {
                    live.push((id, *value));
                }
            }
        }
        Ok(VmSnapshot {
            memory: self.emu.memory.snapshot(),
            block,
            op,
            boundary,
            live,
        })
    }

    /// Puts the machine back where [`snapshot`](Self::snapshot) found it.
    ///
    /// Memory and registers are always restored. The position is restored
    /// exactly when the op it named is still where it was — blocks grow, get
    /// cleaned and get split between a snapshot and a restore, and the
    /// snapshot recognises its op by id and address rather than by index.
    /// When it is gone, the machine is put at the instruction the snapshot
    /// was about to start, lifted again if need be; a snapshot taken
    /// part-way through a guest instruction whose block has since gone
    /// cannot be positioned, and says so — its memory is restored, its
    /// position is not.
    pub fn restore(&mut self, snapshot: &VmSnapshot) -> Result<(), RestoreError> {
        self.emu.memory.restore(&snapshot.memory);
        self.pending = None;
        self.offer_rest = false;
        self.absorbed_into = None;
        self.code_written_at = None;
        self.emu.invalidate_block_cache();
        if let Some(pages) = self.emu.memory.mmu.take_code_writes() {
            self.evict_pages(&pages);
        }

        let block = snapshot.block;
        let position = if !self.ctx.contains_block(block) {
            None
        } else {
            let ids = self.ctx.block(block).instruction_ids();
            match snapshot.op {
                // The same op, wherever a cleanup ahead of it has moved it.
                Some((local, addr)) => ids.iter().position(|&l| l == local).filter(|_| {
                    self.ctx
                        .get_insn(InstructionId::new(block.func, local))
                        .address()
                        == addr
                }),
                // An empty placeholder: still the same one if it has no code.
                None => ids.is_empty().then_some(0),
            }
        };
        if let Some(idx) = position {
            self.emu.block = block;
            self.emu.idx = idx;
            for &(id, value) in &snapshot.live {
                self.emu.insn_values.insert(id, value);
            }
            return Ok(());
        }
        let Some(pc) = snapshot.boundary else {
            return Err(RestoreError::PositionLost);
        };
        let covering = |vm: &mut Self| {
            vm.emu
                .block_at_address(&vm.ctx, pc)
                .filter(|&b| vm.ctx.block(b).address == Some(pc))
        };
        if covering(self).is_none()
            && let Some(VmExit::Unlifted { addr, error }) = self.discover(pc)
        {
            return Err(RestoreError::Unlifted { addr, error });
        }
        self.absorbed_into = None;
        let block = covering(self).ok_or(RestoreError::PositionLost)?;
        self.emu.block = block;
        self.emu.idx = 0;
        Ok(())
    }

    /// Points the emulator at whatever block now covers `addr`, reusing the
    /// emulator's own cached index rather than building one.
    fn reposition(&mut self, addr: u64) {
        if let Some(block) = self.emu.block_at_address(&self.ctx, addr)
            && block != self.emu.block
        {
            self.emu.block = block;
            self.emu.idx = 0;
        }
    }

    /// Lifts `addr` and makes it visible to the emulator. Returns an exit only
    /// if the address could not be supplied.
    fn discover(&mut self, addr: u64) -> Option<VmExit> {
        // Moved out, updated in place by the lift, and moved back: the index
        // stays current without ever being rebuilt, and the borrow checker is
        // satisfied because nothing borrows the emulator across the lift.
        let mut index = self
            .emu
            .take_address_index()
            .unwrap_or_else(|| AddressIndex::analyze(&self.ctx));
        self.stats.lifts += 1;
        let result = self.source.lift(
            &mut self.ctx,
            &self.emu.memory,
            &mut index,
            addr,
            &mut self.stats,
        );
        self.emu.set_address_index(index);
        if let Err(error) = result {
            return Some(VmExit::Unlifted { addr, error });
        }
        self.emu.memory.mmu.mark_code(addr, MAX_INSN_LEN);
        if self.optimize
            && let Some(block) = self.emu.block_at_address(&self.ctx, addr)
        {
            // Forwarding first: it turns the temp round trips into direct value
            // uses, which is what leaves the surrounding computation dead.
            let started = std::time::Instant::now();
            let cleanup = crate::optimize::forward_temp_stores(&mut self.ctx, block);
            qcode_passes::remove_dead_insns(&mut self.ctx, block);
            self.stats.optimize += started.elapsed();
            self.stats.forwarded_loads += cleanup.forwarded_loads as u64;
            self.stats.removed_stores += cleanup.removed_stores as u64;
        }
        self.absorbed_into = self.absorb_into_basic_block(addr);
        None
    }

    /// Points every address `block` has absorbed back at `block`.
    ///
    /// Absorption deletes the blocks it takes in, and each of them was what the
    /// index named for its address. Left alone the index hands out ids of
    /// deleted blocks — and a run may absorb a whole chain, not just the block
    /// that was being discovered, so every address the absorber now covers has
    /// to be repointed, not only the one that prompted this.
    fn reindex_absorbed(&mut self, block: BlockId) {
        let covered = self.ctx.block(block).extra_addresses.clone();
        if covered.is_empty() {
            return;
        }
        let mut index = self
            .emu
            .take_address_index()
            .unwrap_or_else(|| AddressIndex::analyze(&self.ctx));
        for addr in covered {
            index.set_block(addr, block);
        }
        self.emu.set_address_index(index);
    }

    /// Re-runs the block cleanup over a block that has just grown.
    ///
    /// The cleanup at discovery saw a single guest instruction, where every
    /// register it writes is still live at the block's edge. Absorption puts a
    /// whole run in one block, and that is the first point at which a write
    /// nothing goes on to read is visible as dead: the flags an arithmetic
    /// instruction sets, when the next instruction overwrites all of them
    /// before the branch reads any.
    ///
    /// Deliberately block-local (no alias result): at discovery the rest of the
    /// CFG is still unknown, so only a store this block itself overwrites can
    /// be proven dead. Anything live at the exit stays.
    /// Records that `block` has grown and owes a cleanup, cleaning whatever
    /// run was growing before it.
    fn mark_dirty(&mut self, block: BlockId) {
        // Grown, so it holds guest instructions the injectors have not seen.
        self.injected.remove(&block);
        if !self.optimize {
            return;
        }
        let previous = self.dirty.replace(block);
        if let Some(previous) = previous
            && previous != block
            // Discovery retires blocks — splitting an absorbed run empties
            // both halves — so the one that was growing may be gone.
            && self.ctx.contains_block(previous)
        {
            self.reoptimize(previous);
        }
    }

    /// Cleans the block the machine is about to run, if it has grown since it
    /// was last cleaned.
    ///
    /// Only ever the block being entered, which is why a stale id cannot be
    /// reached here: absorption and splitting retire blocks that may still be
    /// listed, but the machine can only be about to run a live one. A leftover
    /// entry for a retired id is harmless — at worst it cleans a block whose id
    /// was reused, which is always safe.
    fn clean_before_entering(&mut self, block: BlockId) {
        if self.dirty == Some(block) {
            self.dirty = None;
            self.reoptimize(block);
        }
    }

    fn reoptimize(&mut self, block: BlockId) {
        if !self.optimize {
            return;
        }
        let started = std::time::Instant::now();
        let cleanup = crate::optimize::forward_temp_stores(&mut self.ctx, block);
        qcode_passes::remove_dead_insns(&mut self.ctx, block);
        self.stats.optimize += started.elapsed();
        self.stats.forwarded_loads += cleanup.forwarded_loads as u64;
        self.stats.removed_stores += cleanup.removed_stores as u64;
        // The interpreter may hold this block's instruction list, and some of
        // those instructions are gone.
        self.emu.invalidate_block_cache();
    }

    /// Folds the block just lifted at `addr` into its predecessor, when the two
    /// are a straight-line pair.
    ///
    /// Lifting is per guest instruction, so a run of straight-line guest code
    /// arrives as a chain of one-instruction blocks joined by unconditional
    /// branches. Left that way, every guest instruction is its own unit of
    /// execution: a separate compilation, a separate entry into compiled code
    /// and a separate return to the interpreter for its terminator. Folding the
    /// chain as it is discovered rebuilds the guest's *basic block*, which is
    /// the unit worth compiling.
    ///
    /// Absorbing an address does not settle that it belongs here — code
    /// discovered later may branch into the middle of the run, and
    /// [`Context::split_block_at_address`] breaks it apart again when it does.
    fn absorb_into_basic_block(&mut self, addr: u64) -> Option<BlockId> {
        let filled = self.emu.block_at_address(&self.ctx, addr)?;

        // Forward: the rest of this run may already be known. That is the shape
        // a split leaves behind — it re-establishes a block's *start* while
        // everything after it is still lifted — and the shape a back-edge into
        // the middle of a run creates generally.
        let forward = qcode_passes::absorb_straight_line(&mut self.ctx, filled);
        self.stats.absorbed += forward as u64;
        if forward > 0 {
            self.reindex_absorbed(filled);
            self.mark_dirty(filled);
        }

        // Backward: the straight-line predecessor that branched here, for the
        // ordinary case of a run discovered one guest instruction at a time.
        //
        // Not if something branches to this address: it has to keep *starting*
        // a block, or the split that established that would be undone here.
        // Extending it forward, above, stays fine — that moves its end, not its
        // start.
        if self
            .emu
            .address_index()
            .is_some_and(|index| index.is_boundary(addr))
        {
            return (forward > 0).then_some(filled);
        }
        // Exactly one predecessor, or absorbing would strand the others.
        // Collected eagerly so the module is free to be mutated below.
        let preds: Vec<BlockId> = BasicBlock::from_id(&self.ctx, filled)
            .predecessors()
            .map(|(_, block)| block)
            .take(2)
            .collect();
        let [head] = preds[..] else {
            return None;
        };
        if head == filled {
            return None;
        }
        // Only into a block that starts at a machine address. Absorbing makes
        // the head responsible for the absorbed addresses, and a later branch
        // to one of them splits the head apart again — which works by emptying
        // it and lifting it afresh. A block with no address of its own (the
        // fallthrough arm of a branch *inside* one instruction's p-code) has
        // nowhere to be lifted from, so emptying it leaves a hole nothing can
        // fill.
        if self.ctx.block(head).address.is_none() {
            return (forward > 0).then_some(filled);
        }
        // Where `filled`'s instructions land: the head's own, less the
        // terminator that absorption drops.
        let offset = self
            .ctx
            .block(head)
            .instruction_ids()
            .len()
            .saturating_sub(1);
        if qcode_passes::absorb_straight_line(&mut self.ctx, head) == 0 {
            return (forward > 0).then_some(filled);
        }
        self.stats.absorbed += 1;

        // What the head has already run: everything it held before the
        // absorbed instructions were appended. The machine resumes right
        // after the last of these still standing, whatever cleanup deletes
        // ahead of that point or injection inserts after it — an interrupt an
        // injector places before the first absorbed instruction, say, which
        // has to run before it.
        let executed: FxHashSet<LocalInsnId> = self.ctx.block(head).instruction_ids()[..offset]
            .iter()
            .copied()
            .collect();
        self.mark_dirty(head);

        self.reindex_absorbed(head);

        // The machine stopped at the empty placeholder this lift filled, which
        // absorption has just deleted; its instructions are in the head now.
        if self.emu.block == filled {
            // The absorbed instructions have not run, and the injectors have
            // not seen them: rewrite now, so a hook on the instruction about
            // to execute is not missed the first time.
            self.inject(head);
            let now = self.ctx.block(head).instruction_ids();
            let resumed = now
                .iter()
                .rposition(|local| executed.contains(local))
                .map_or(0, |last| last + 1);
            self.emu.block = head;
            self.emu.idx = resumed;
            self.emu.invalidate_block_cache();
        }
        Some(head)
    }

    /// Runs until the machine stops, or until `budget` p-code operations have
    /// been retired.
    ///
    /// Hooks registered through the [table](crate::table) are called from
    /// here and the machine resumes past them, so the run only returns for
    /// an exit the caller has to see.
    pub fn run(&mut self, budget: u64) -> VmExit {
        let deadline = self.stats.steps + budget;
        while self.stats.steps < deadline {
            // Checked before the step so a breakpoint reports the instruction
            // about to run, not the one after it, and so resuming from a
            // breakpoint is possible without immediately re-triggering it.
            //
            // Guarded on there being any breakpoint at all: `pc()` resolves the
            // block through the module arena, and paying that on every step to
            // consult an empty set cost about 6% of run time.
            if !self.breakpoints.is_empty()
                && let Some(pc) = self.pc()
                && self.breakpoints.contains(&pc)
                && self.stats.steps > 0
            {
                return VmExit::Breakpoint(pc);
            }
            match self.step() {
                None => {}
                Some(VmExit::Interrupt(interrupt)) => {
                    if let Some(exit) = self.dispatch(&interrupt) {
                        return exit;
                    }
                }
                Some(exit) => return exit,
            }
        }
        VmExit::InstructionLimit
    }
}

/// The longest instruction any supported architecture encodes, in bytes: the
/// span marked as code for a lift when the exact length is not known here.
const MAX_INSN_LEN: u64 = 16;

/// Resolves a guest address to a block against an existing index.
fn resolve(ctx: &Context<'_>, index: &AddressIndex, addr: u64) -> Option<BlockId> {
    match index.get(addr) {
        Some(AddressTarget::Block(block)) => Some(block),
        Some(AddressTarget::Function(function)) => {
            qcode::value::FunctionBody::from_id(ctx, function)
                .root()
                .map(|root| root.id)
        }
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mmu::{PAGE_SIZE, perm};
    use qcode::value::FunctionBody;

    /// A source that hands over one pre-planned block per address, so the
    /// discovery path can be exercised without a decoder.
    #[derive(Default)]
    struct Planned {
        /// Addresses this source is willing to supply, and how many times it was
        /// actually asked.
        available: Vec<u64>,
        pub calls: Vec<u64>,
    }

    impl CodeSource for Planned {
        fn lift(
            &mut self,
            ctx: &mut Context<'static>,
            memory: &VmMemory,
            index: &mut AddressIndex,
            addr: u64,
            _stats: &mut Stats,
        ) -> Result<(), CodeError> {
            self.calls.push(addr);
            // A real source fetches through the MMU, so executing unmapped or
            // non-executable memory faults at the fetch. Mirrored here.
            let mut byte = [0u8; 1];
            memory
                .mmu
                .read_code(addr, &mut byte)
                .map_err(CodeError::Fault)?;
            if !self.available.contains(&addr) {
                return Err(CodeError::Decode("no plan for this address".into()));
            }
            let function = FunctionBody::make_at_addr(ctx, addr, None).id;
            let block = BasicBlock::make(ctx, function).with_address(addr).id;
            index
                .register(ctx, addr, AddressTarget::Block(block))
                .map_err(|error| CodeError::Decode(format!("{error:?}").into()))?;
            Ok(())
        }
    }

    /// A module with a single empty block at `addr`.
    fn module(addr: u64) -> (Context<'static>, BlockId) {
        let mut ctx = Context::new();
        let function = FunctionBody::make_at_addr(&mut ctx, addr, None).id;
        let block = BasicBlock::make(&mut ctx, function).with_address(addr).id;
        (ctx, block)
    }

    fn executable_memory() -> VmMemory {
        let mut memory = VmMemory::new();
        memory.mmu.map(0x1000, PAGE_SIZE, perm::RX_INIT).unwrap();
        memory
    }

    #[test]
    fn an_empty_addressed_block_asks_the_source_for_code() {
        let (ctx, block) = module(0x1000);
        let mut vm = Vm::new(ctx, block, Planned::default());
        // An empty block carrying an address is how the lifter represents a
        // branch target it has not reached yet, so it is a discovery request.
        // This source cannot supply it, which is what makes the exit reportable.
        let exit = vm.run(16);
        assert!(
            matches!(exit, VmExit::Unlifted { addr: 0x1000, .. }),
            "expected a discovery attempt, got {exit:?}"
        );
        assert_eq!(vm.source.calls, vec![0x1000]);
    }

    #[test]
    fn an_empty_block_with_no_address_is_an_error() {
        // Nothing to discover: without an address there is no code to fetch.
        let mut ctx = Context::new();
        let function = FunctionBody::make_at_addr(&mut ctx, 0x1000, None).id;
        let block = BasicBlock::make(&mut ctx, function).id;
        let mut vm = Vm::new(ctx, block, Planned::default());
        assert!(matches!(vm.run(16), VmExit::Error(_)));
    }

    #[test]
    fn pc_reports_the_block_about_to_run() {
        let (ctx, block) = module(0x1000);
        let vm = Vm::new(ctx, block, Planned::default());
        assert_eq!(vm.pc(), Some(0x1000));
    }

    #[test]
    fn at_address_lifts_an_entry_the_module_lacks() {
        let ctx = Context::new();
        let source = Planned {
            available: vec![0x1000],
            calls: Vec::new(),
        };
        let vm = Vm::at_address(ctx, 0x1000, source, executable_memory())
            .expect("the source can supply this address");
        assert_eq!(vm.pc(), Some(0x1000));
    }

    #[test]
    fn at_address_reuses_a_block_the_module_already_has() {
        let (ctx, _) = module(0x1000);
        let vm = Vm::at_address(ctx, 0x1000, Planned::default(), executable_memory())
            .expect("no lifting is needed");
        assert_eq!(vm.pc(), Some(0x1000));
        // The source was never consulted.
        assert!(vm.source.calls.is_empty());
    }

    #[test]
    fn fetching_from_non_executable_memory_reports_the_fault() {
        let ctx = Context::new();
        let mut memory = VmMemory::new();
        memory.mmu.map(0x1000, PAGE_SIZE, perm::RW_INIT).unwrap();
        let source = Planned {
            available: vec![0x1000],
            calls: Vec::new(),
        };
        let error = Vm::at_address(ctx, 0x1000, source, memory)
            .err()
            .expect("the page is not executable");
        assert!(matches!(
            error,
            CodeError::Fault(MemFault {
                kind: crate::mmu::FaultKind::ExecViolation,
                addr: 0x1000
            })
        ));
    }

    #[test]
    fn an_unsuppliable_address_reports_where_it_stopped() {
        let ctx = Context::new();
        let error = Vm::at_address(ctx, 0x2000, Planned::default(), executable_memory())
            .err()
            .expect("nothing is mapped or planned at 0x2000");
        assert!(matches!(error, CodeError::Fault(_)));
    }

    #[test]
    fn breakpoints_are_recorded_and_removable() {
        let (ctx, block) = module(0x1000);
        let mut vm = Vm::new(ctx, block, Planned::default());
        assert!(vm.add_breakpoint(0x2000));
        assert!(!vm.add_breakpoint(0x2000));
        assert!(vm.remove_breakpoint(0x2000));
        assert!(!vm.remove_breakpoint(0x2000));
    }

    #[test]
    fn memory_is_reachable_and_backed_by_the_mmu() {
        let (ctx, block) = module(0x1000);
        let mut vm = Vm::new(ctx, block, Planned::default());
        vm.memory_mut()
            .mmu
            .map(0x4000, PAGE_SIZE, perm::RW_INIT)
            .unwrap();
        vm.memory_mut().mmu.write(0x4000, &[1, 2, 3]).unwrap();
        let mut out = [0; 3];
        vm.memory().mmu.read(0x4000, &mut out).unwrap();
        assert_eq!(out, [1, 2, 3]);
    }

    /// A block whose body is one user op, followed by a branch to an empty
    /// block at the next address, so continuing past the op is observable as
    /// a discovery request for that address.
    fn module_with_op(
        op_name: &str,
        args: Vec<u64>,
        size: usize,
    ) -> (Context<'static>, BlockId, InstructionId) {
        let (mut ctx, block) = module(0x1000);
        let op = ctx.shared.pcode_op(op_name);
        let target = BasicBlock::make(&mut ctx, block.func)
            .with_address(0x1001)
            .id;
        let args = args
            .into_iter()
            .map(|value| ctx.shared.get_const(value, 8))
            .collect();
        let insn = {
            let mut builder = ctx.builder(block);
            let insn = builder.push_pcode_op(op, args, None, size).id;
            builder.finalize(target);
            insn
        };
        (ctx, block, insn)
    }

    #[test]
    fn an_explicit_interrupt_stops_at_the_op_and_reports_its_operands() {
        let (ctx, block, insn) = module_with_op(VM_INTERRUPT, vec![7, 99], 8);
        let mut vm = Vm::new(ctx, block, Planned::default());
        let exit = vm.run(16);
        let VmExit::Interrupt(interrupt) = exit else {
            panic!("expected an interrupt, got {exit:?}");
        };
        assert_eq!(interrupt.kind, InterruptKind::Explicit { code: 7 });
        assert_eq!(interrupt.args, vec![Some(99)]);
        assert_eq!(interrupt.insn, insn);
        assert_eq!(interrupt.size, 8);
        assert_eq!(interrupt.pc, Some(0x1000));
        // The machine is at the op, not past it.
        assert_eq!(vm.emulator().idx, 0);
        assert_eq!(vm.pending_interrupt(), Some(&interrupt));
    }

    #[test]
    fn running_again_without_resuming_reports_the_same_interrupt() {
        let (ctx, block, _) = module_with_op(VM_INTERRUPT, vec![1], 0);
        let mut vm = Vm::new(ctx, block, Planned::default());
        let first = vm.run(16);
        let again = vm.run(16);
        assert!(
            matches!(&first, VmExit::Interrupt(i) if i.kind == InterruptKind::Explicit { code: 1 })
        );
        assert!(
            matches!(&again, VmExit::Interrupt(i) if i.kind == InterruptKind::Explicit { code: 1 })
        );
        assert_eq!(vm.emulator().idx, 0);
    }

    #[test]
    fn resume_files_the_result_and_continues_after_the_op() {
        let (ctx, block, insn) = module_with_op(VM_INTERRUPT, vec![7], 8);
        let mut vm = Vm::new(ctx, block, Planned::default());
        assert!(matches!(vm.run(16), VmExit::Interrupt(_)));
        // The op declares an 8-byte result, so resuming needs one.
        assert_eq!(
            vm.resume(None),
            Err(ResumeError::ResultRequired { size: 8 })
        );
        vm.resume(Some(42)).unwrap();
        assert_eq!(vm.pending_interrupt(), None);
        assert_eq!(
            vm.emulator().insn_values.get(&insn).map(|v| v.as_bits()),
            Some(42)
        );
        // Past the op, the branch runs and reaches the next address.
        assert!(matches!(vm.run(16), VmExit::Unlifted { addr: 0x1001, .. }));
    }

    #[test]
    fn resume_needs_an_interrupt() {
        let (ctx, block) = module(0x1000);
        let mut vm = Vm::new(ctx, block, Planned::default());
        assert_eq!(vm.resume(None), Err(ResumeError::NotInterrupted));
    }

    #[test]
    fn an_unmodelled_user_op_is_an_intrinsic_interrupt() {
        let (ctx, block, _) = module_with_op("rdpmc", vec![], 0);
        let mut vm = Vm::new(ctx, block, Planned::default());
        let exit = vm.run(16);
        let VmExit::Interrupt(interrupt) = exit else {
            panic!("expected an interrupt, got {exit:?}");
        };
        assert!(
            matches!(&interrupt.kind, InterruptKind::Intrinsic { name, .. } if name.as_ref() == "rdpmc")
        );
        assert_eq!(interrupt.size, 0);
        // Nothing to supply for an op with no result.
        vm.resume(None).unwrap();
        assert!(matches!(vm.run(16), VmExit::Unlifted { addr: 0x1001, .. }));
    }
}
