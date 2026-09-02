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
    address_index::AddressTarget,
    context::Context,
    value::{BasicBlock, BlockId},
};
use qcode_emulator::{EmulatorErrorKind, EmulatorMemory, StandaloneEmulator};
use rustc_hash::FxHashSet;

use crate::{memory::VmMemory, mmu::MemFault};

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
    fn lift(
        &mut self,
        ctx: &mut Context<'static>,
        memory: &VmMemory,
        addr: u64,
    ) -> Result<(), CodeError>;
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
    /// P-code operations retired since the machine was created.
    ///
    /// Deliberately *not* a guest-instruction count: one machine instruction
    /// lifts to many QCode operations, and the interpreter steps one operation
    /// at a time. Budgets are therefore in units of work done, which is the
    /// honest thing to bound a run by; a guest-instruction count is a separate
    /// measure and is not yet tracked.
    pub steps: u64,
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
            steps: 0,
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
        if block_at(&ctx, addr).is_none() {
            source.lift(&mut ctx, &memory, addr)?;
        }
        let entry = block_at(&ctx, addr).ok_or_else(|| {
            CodeError::Decode(format!("no block at {addr:#x} after lifting").into())
        })?;
        let mut vm = Self::new(ctx, entry, source);
        vm.emu.memory = memory;
        vm.emu.memory.configure_spaces(&vm.ctx);
        Ok(vm)
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
            match self.emu.step(&self.ctx) {
                Ok(()) => {
                    self.steps += 1;
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

    /// Points the emulator at whatever block now covers `addr`.
    fn reposition(&mut self, addr: u64) {
        if let Some(block) = block_at(&self.ctx, addr)
            && block != self.emu.block
        {
            self.emu.block = block;
            self.emu.idx = 0;
        }
    }

    /// Lifts `addr` and makes it visible to the emulator. Returns an exit only
    /// if the address could not be supplied.
    fn discover(&mut self, addr: u64) -> Option<VmExit> {
        if let Err(error) = self.source.lift(&mut self.ctx, &self.emu.memory, addr) {
            return Some(VmExit::Unlifted { addr, error });
        }
        // The emulator caches its address lookup over what was, until now, an
        // immutable module.
        self.emu.invalidate_address_index();
        None
    }

    /// Runs until the machine stops, or until `budget` p-code operations have
    /// been retired.
    pub fn run(&mut self, budget: u64) -> VmExit {
        let deadline = self.steps + budget;
        while self.steps < deadline {
            // Checked before the step so a breakpoint reports the instruction
            // about to run, not the one after it, and so resuming from a
            // breakpoint is possible without immediately re-triggering it.
            if let Some(pc) = self.pc()
                && self.breakpoints.contains(&pc)
                && self.steps > 0
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

/// Resolves a guest address to a block, without the emulator's cached index.
fn block_at(ctx: &Context<'_>, addr: u64) -> Option<BlockId> {
    let index = qcode::address_index::AddressIndex::analyze(ctx);
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
            addr: u64,
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
            BasicBlock::make(ctx, function).with_address(addr);
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
