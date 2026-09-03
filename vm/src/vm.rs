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

use qcode::{
    address_index::{AddressIndex, AddressTarget},
    context::Context,
    value::{BasicBlock, BlockId},
};
use qcode_emulator::{EmulatorErrorKind, EmulatorMemory, StandaloneEmulator};
use rustc_hash::FxHashSet;

use crate::{memory::VmMemory, mmu::MemFault, stats::Stats};

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
    /// `Ok(false)` means "not mine" and is not an error — the caller falls back
    /// to the interpreter.
    ///
    /// On `Ok(true)` every value the terminator reads must be readable from
    /// `emu.insn_values`, exactly as if the interpreter had run the body.
    fn run_block(
        &mut self,
        ctx: &Context<'_>,
        emu: &mut StandaloneEmulator<VmMemory>,
        block: BlockId,
    ) -> Result<bool, EmulatorErrorKind>;
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
    /// The interpreter reported something the VM does not model as a guest
    /// event — an unsupported p-code op, a malformed block.
    Error(Box<str>),
}

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
    /// Whether freshly lifted blocks get a cleanup round.
    ///
    /// Lifting one machine instruction emits every side effect the
    /// specification describes, including flag computations the surrounding
    /// code never reads. Removing the ones with no users at all is sound
    /// block-locally — an instruction with no users cannot be observed — and is
    /// paid once per block instead of on every execution of it.
    pub optimize: bool,
    breakpoints: FxHashSet<u64>,
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
            breakpoints: FxHashSet::default(),
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

    /// Executes one instruction, lifting code on demand if control leaves the
    /// part of the module already known.
    ///
    /// Returns `None` when the step was ordinary, and `Some(exit)` when the
    /// machine stopped for a reason worth reporting.
    pub fn step(&mut self) -> Option<VmExit> {
        // A branch to unlifted code fails *before* the emulator moves, so the
        // address can be lifted and the same step retried. One retry is enough:
        // the second failure means the source did not produce the block it
        // claimed to, which is a source bug rather than a discovery step.
        for attempt in 0..2 {
        // At a block's first instruction, an installed executor may run the
        // whole body at once, leaving the interpreter only the terminator.
        if self.emu.idx == 0
            && let Some(executor) = self.executor.as_mut()
        {
            let block = self.emu.block;
            match executor.run_block(&self.ctx, &mut self.emu, block) {
                Ok(true) => {
                    let body = BasicBlock::from_id(&self.ctx, block)
                        .instruction_count()
                        .saturating_sub(1);
                    // The body's operations were retired by the executor; they
                    // are counted so throughput stays comparable between
                    // strategies.
                    self.stats.steps += body as u64;
                    self.stats.native_bodies += 1;
                    // Positioning inside a block the interpreter has not walked
                    // into invalidates its cached instruction list.
                    self.emu.invalidate_block_cache();
                    self.emu.idx = body;
                }
                Ok(false) => {}
                Err(kind) => {
                    let fault = self.emu.memory.take_fault();
                    return Some(match fault {
                        Some(fault) => VmExit::Fault(fault),
                        None => VmExit::Error(kind.to_string().into()),
                    });
                }
            }
        }

            match self.emu.step(&self.ctx) {
                Ok(()) => {
                    self.stats.steps += 1;
                    return None;
                }
                Err(error) => match error.kind {
                    EmulatorErrorKind::InvalidBlockAddress(addr)
                    | EmulatorErrorKind::UnknownAddress(addr)
                        if attempt == 0 =>
                    {
                        if let Some(exit) = self.discover(addr) {
                            return Some(exit);
                        }
                    }
                    // A direct branch to code that has not been lifted does not
                    // fail to resolve: the lifter materializes the target as an
                    // *empty* block at that address, and execution walks into
                    // it. So an empty block carrying an address is a request to
                    // discover it, not a malformed-IR error.
                    EmulatorErrorKind::EmptyBlock(block) if attempt == 0 => {
                        let Some(addr) = BasicBlock::from_id(&self.ctx, block).address() else {
                            return Some(VmExit::Error(
                                EmulatorErrorKind::EmptyBlock(block).to_string().into(),
                            ));
                        };
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
                        if let Some(exit) = self.discover(addr) {
                            return Some(exit);
                        }
                        // Lifting may place the instruction in a *new* block
                        // rather than filling the placeholder the emulator is
                        // sitting in, so the machine has to be moved onto
                        // whatever now covers the address.
                        self.reposition(addr);
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
                    kind => return Some(VmExit::Error(kind.to_string().into())),
                },
            }
        }
        None
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
        if self.optimize
            && let Some(block) = self.emu.block_at_address(&self.ctx, addr)
        {
            // Forwarding first: it turns the temp round trips into direct value
            // uses, which is what leaves the surrounding computation dead.
            let started = std::time::Instant::now();
            let cleanup = crate::optimize::forward_temp_stores(&mut self.ctx, block);
            qcode_analysis::dce::remove_dead_insns(&mut self.ctx, block);
            self.stats.optimize += started.elapsed();
            self.stats.forwarded_loads += cleanup.forwarded_loads as u64;
            self.stats.removed_stores += cleanup.removed_stores as u64;
        }
        None
    }

    /// Runs until the machine stops, or until `budget` p-code operations have
    /// been retired.
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
            if let Some(exit) = self.step() {
                return exit;
            }
        }
        VmExit::InstructionLimit
    }
}

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
        let error = Vm::at_address(ctx, 0x1000, source, memory).err()
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
}
