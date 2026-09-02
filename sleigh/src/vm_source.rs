//! A [`CodeSource`] that decodes guest memory with SLEIGH.
//!
//! This is what turns [`qcode_vm::Vm`] from a machine that must be handed its
//! code into one that discovers code the way a processor does: fetch from
//! memory at the program counter, decode, execute, repeat.
//!
//! Two properties matter here and are easy to get wrong.
//!
//! **The fetch goes through the MMU's execute permission.** Jumping into a
//! non-executable page must fault at the fetch, exactly as on hardware, rather
//! than quietly decoding data as code.
//!
//! **A fetch near the end of a mapping is not a fault.** An instruction is at
//! most [`MAX_INSTRUCTION_LEN`] bytes, but the one being decoded may be much
//! shorter and sit right against the end of a mapped region. Demanding the full
//! window would invent a fault the guest never takes, so the fetch shrinks to
//! whatever is readable and lets the decoder decide whether it has enough.

use qcode::{address_index::AddressIndex, context::Context};
use qcode_vm::{CodeError, CodeSource, VmMemory};
use sleigh::{CompiledSpec, Decoder};

use crate::SleighLifter;

/// The longest instruction any supported architecture encodes. x86-64's 15-byte
/// limit is the largest; a shorter-instruction architecture simply never uses
/// the tail of the window.
pub const MAX_INSTRUCTION_LEN: usize = 16;

/// Decodes and lifts guest memory on demand.
pub struct SleighCodeSource<'spec> {
    lifter: SleighLifter<'spec>,
    /// Retained across lifts so a run does not rebuild the address index for
    /// every instruction it discovers.
    index: Option<AddressIndex>,
}

impl<'spec> SleighCodeSource<'spec> {
    pub fn new(spec: &'spec CompiledSpec) -> Self {
        Self {
            lifter: SleighLifter::new(spec),
            index: None,
        }
    }

    /// A context initialized for this specification: spaces, registers and
    /// user p-code operations already installed.
    pub fn new_context(&self) -> Context<'static> {
        self.lifter.new_context()
    }

    pub fn lifter(&self) -> &SleighLifter<'spec> {
        &self.lifter
    }

    /// Fetches up to [`MAX_INSTRUCTION_LEN`] executable bytes at `addr`.
    ///
    /// Shrinks the window rather than faulting when fewer bytes are readable, so
    /// that a short instruction at the end of a mapping decodes. A fetch that
    /// cannot produce even one byte is a genuine fault and is reported as one.
    fn fetch(memory: &VmMemory, addr: u64) -> Result<Vec<u8>, CodeError> {
        let mut first = None;
        for len in (1..=MAX_INSTRUCTION_LEN).rev() {
            let mut bytes = vec![0u8; len];
            match memory.mmu.read_code(addr, &mut bytes) {
                Ok(()) => return Ok(bytes),
                // The widest window's fault is the informative one: it names the
                // first byte that is actually unreadable.
                Err(fault) => first.get_or_insert(fault),
            };
        }
        Err(CodeError::Fault(
            first.expect("the loop attempts at least one read"),
        ))
    }
}

impl CodeSource for SleighCodeSource<'_> {
    fn lift(
        &mut self,
        ctx: &mut Context<'static>,
        memory: &VmMemory,
        addr: u64,
    ) -> Result<(), CodeError> {
        let bytes = Self::fetch(memory, addr)?;

        // Built from the context the first time it is needed, then carried
        // across the run; `decode_and_lift_indexed` keeps it current as blocks
        // are added.
        let index = self
            .index
            .get_or_insert_with(|| AddressIndex::analyze(ctx));

        let decode_context = self.lifter.spec().new_context();
        let instruction = Decoder::new(self.lifter.spec())
            .decode_one(addr, &bytes, &decode_context)
            .map_err(|error| CodeError::Decode(error.to_string().into()))?;

        self.lifter
            .lift_instruction_indexed(ctx, index, &instruction, None)
            .map_err(|error| CodeError::Decode(format!("{error:?}").into()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode_vm::{Vm, VmExit, perm};

    fn spec() -> &'static CompiledSpec {
        sleigh_precompile::x64::spec()
    }

    /// A VM with `code` mapped executable at 0x1000 and a writable stack.
    fn machine(code: &[u8]) -> Vm<SleighCodeSource<'static>> {
        let source = SleighCodeSource::new(spec());
        let ctx = source.new_context();
        let mut memory = VmMemory::new();
        memory.mmu.write_unchecked(0x1000, code, perm::READ | perm::EXEC);
        memory.mmu.map(0x20000, 0x2000, perm::RW_INIT).unwrap();
        Vm::at_address(ctx, 0x1000, source, memory).expect("the entry decodes")
    }

    #[test]
    fn decodes_and_runs_a_real_instruction() {
        // mov eax, 1
        let mut vm = machine(&[0xb8, 0x01, 0x00, 0x00, 0x00]);
        vm.run(64);
        let ctx = vm.context().clone();
        let eax = vm
            .emulator()
            .read_varnode_by_name(&ctx, "EAX")
            .expect("EAX is a register in this specification");
        assert_eq!(eax, 1);
    }

    #[test]
    fn runs_a_straight_line_sequence_discovering_each_instruction() {
        // mov eax, 5 ; mov ebx, 7 ; add eax, ebx
        let mut vm = machine(&[
            0xb8, 0x05, 0x00, 0x00, 0x00, // mov eax, 5
            0xbb, 0x07, 0x00, 0x00, 0x00, // mov ebx, 7
            0x01, 0xd8, // add eax, ebx
        ]);
        // The budget is in p-code operations, of which each instruction takes
        // several.
        vm.run(64);
        let ctx = vm.context().clone();
        assert_eq!(
            vm.emulator().read_varnode_by_name(&ctx, "EAX"),
            Some(12),
            "each instruction was fetched and lifted on demand"
        );
    }

    #[test]
    fn a_taken_branch_discovers_its_target() {
        // jmp +2 ; (skipped) mov eax, 0xff ; mov eax, 1
        let mut vm = machine(&[
            0xeb, 0x05, // jmp to the third instruction
            0xb8, 0xff, 0x00, 0x00, 0x00, // mov eax, 0xff  (skipped)
            0xb8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
        ]);
        vm.run(64);
        let ctx = vm.context().clone();
        assert_eq!(
            vm.emulator().read_varnode_by_name(&ctx, "EAX"),
            Some(1),
            "the branch target was discovered and the skipped store never ran"
        );
    }

    #[test]
    fn a_store_to_unmapped_memory_stops_with_a_write_fault() {
        // mov rax, 0x9999000 ; mov [rax], ebx  -- nothing is mapped there.
        let mut vm = machine(&[
            0x48, 0xc7, 0xc0, 0x00, 0x90, 0x99, 0x09, // mov rax, 0x9990000
            0x89, 0x18, // mov [rax], ebx
        ]);
        let exit = vm.run(64);
        match exit {
            VmExit::Fault(fault) => assert_eq!(fault.kind, qcode_vm::FaultKind::WriteUnmapped),
            other => panic!("expected a write fault, got {other:?}"),
        }
    }

    #[test]
    fn fetching_from_a_non_executable_page_faults() {
        let source = SleighCodeSource::new(spec());
        let ctx = source.new_context();
        let mut memory = VmMemory::new();
        // Readable and writable, but not executable.
        memory.mmu.map(0x1000, 0x1000, perm::RW_INIT).unwrap();
        let error = Vm::at_address(ctx, 0x1000, source, memory)
            .err()
            .expect("the page is not executable");
        assert!(matches!(error, CodeError::Fault(_)));
    }

    #[test]
    fn a_short_instruction_at_the_end_of_a_mapping_still_decodes() {
        // `nop` is one byte. Placed so that the 16-byte fetch window runs off
        // the end of the mapping, which must not be reported as a fault.
        let source = SleighCodeSource::new(spec());
        let ctx = source.new_context();
        let mut memory = VmMemory::new();
        let last = 0x1000 + 0xfff;
        memory
            .mmu
            .write_unchecked(last, &[0x90], perm::READ | perm::EXEC);
        let vm = Vm::at_address(ctx, last, source, memory);
        assert!(vm.is_ok(), "a one-byte instruction at a mapping's end decodes");
    }

    #[test]
    fn running_into_unmapped_memory_stops_with_a_fault() {
        // `jmp` to an address that is not mapped at all.
        let mut vm = machine(&[0xeb, 0x7f]);
        let exit = vm.run(64);
        assert!(
            matches!(exit, VmExit::Unlifted { .. } | VmExit::Fault(_)),
            "expected a fault-shaped exit, got {exit:?}"
        );
    }
}
