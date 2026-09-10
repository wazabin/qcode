//! A freestanding image on a bare machine: no process, no stack contents,
//! no system calls.
//!
//! This is how the Embench harnesses and the benchmarks run C compiled with
//! `-nostdlib`: the image is entered at its ELF entry point (`main`, for
//! Embench) with a stack whose top holds a sentinel return address that is
//! not mapped. Returning from the entry therefore faults on the sentinel,
//! and [`returned`] recognises that exit as the program having finished.
//! What the program computed is read from the registers afterwards.

use qcode_vm::{Vm, VmExit, VmMemory, perm};
use sleigh_precompile::x64::spec;
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

use crate::loader::{self, LoadError};
use crate::regs::Regs;

/// The address the entry returns to: unmapped, so the run stops there.
pub const SENTINEL: u64 = 0xdead_0000;
/// The stack mapping.
pub const STACK: u64 = 0x7fff_0000;
pub const STACK_SIZE: u64 = 0x40000;
/// Initial `RSP`; the sentinel is the quadword at this address.
pub const STACK_TOP: u64 = 0x7fff_8000;

/// The machine type every runner in this crate produces.
pub type Machine = Vm<SleighCodeSource<'static>>;

/// Loads `image`, maps the stack with the sentinel on top, and positions the
/// machine at the entry point with the registers in their process-start
/// state and `RSP` set. The interpreter is installed; the caller sets a
/// block executor if it wants one.
pub fn machine(image: &[u8]) -> Result<Machine, LoadError> {
    let source = SleighCodeSource::new(spec());
    let ctx = source.new_context();
    let regs = Regs::resolve(&ctx).map_err(LoadError::Parse)?;
    let mut memory = VmMemory::new();
    let loaded = loader::load(image, &mut memory.mmu)?;
    memory
        .mmu
        .map(STACK, STACK_SIZE, perm::RW_INIT)
        .map_err(|e| LoadError::Map(e.to_string()))?;
    memory
        .mmu
        .write_unchecked(STACK_TOP, &SENTINEL.to_le_bytes(), perm::RW_INIT);

    let mut vm = Vm::at_address(ctx, loaded.entry, source, memory)
        .map_err(|e| LoadError::Parse(format!("cannot lift the entry point: {e:?}")))?;
    regs.reset(vm.memory_mut());
    regs.rsp.write(vm.memory_mut(), STACK_TOP);
    Ok(vm)
}

/// Whether `exit` is the entry point having returned onto the sentinel: the
/// program ran to its end. Any other exit, a fault, an unlifted instruction
/// or an exhausted budget, means it did not, and its registers are noise.
pub fn returned(exit: &VmExit) -> bool {
    matches!(exit, VmExit::Unlifted { addr, .. } if *addr == SENTINEL)
}

/// Reads a register by its SLEIGH name, `"EAX"` say, after a run.
pub fn register(vm: &mut Machine, name: &str) -> Option<u64> {
    let ctx = vm.context().clone();
    vm.emulator().read_varnode_by_name(&ctx, name)
}
