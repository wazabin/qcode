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

use qcode::{
    address_index::AddressIndex,
    context::Context,
    value::{FunctionBody, FunctionId},
};
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
    /// The function every lifted instruction is placed in.
    ///
    /// Guest code is flat: it has branch targets, not call graphs the lifter can
    /// know about up front. Letting the lifter create a function per instruction
    /// instead makes every branch a *cross-function* edge, and a block can only
    /// belong to one function — so a backward branch into already-lifted code
    /// would collide with the function that owns that address. One function for
    /// the whole guest keeps every branch target local, which is what makes
    /// loops work.
    function: Option<FunctionId>,
}

impl<'spec> SleighCodeSource<'spec> {
    pub fn new(spec: &'spec CompiledSpec) -> Self {
        Self {
            lifter: SleighLifter::new(spec),
            function: None,
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
        index: &mut AddressIndex,
        addr: u64,
    ) -> Result<(), CodeError> {
        let bytes = Self::fetch(memory, addr)?;

        let decode_context = self.lifter.spec().new_context();
        let instruction = Decoder::new(self.lifter.spec())
            .decode_one(addr, &bytes, &decode_context)
            .map_err(|error| CodeError::Decode(error.to_string().into()))?;

        let function = match self.function {
            Some(function) => function,
            None => {
                let function = FunctionBody::from_addr_or_create_indexed(ctx, index, addr).id;
                self.function = Some(function);
                function
            }
        };

        self.lifter
            .lift_instruction_indexed(ctx, index, &instruction, Some(function))
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

    /// Scaling probe: the per-instruction cost must not grow with the size of
    /// the module already lifted. Reported rather than asserted, so it is a
    /// measurement rather than a flaky threshold.
    #[test]
    fn discovery_cost_does_not_grow_with_module_size() {
        for count in [200usize, 400, 800] {
            let code = vec![0x90; count]; // `nop` * count
            let mut vm = machine(&code);
            let start = std::time::Instant::now();
            vm.run(count as u64 * 8);
            let elapsed = start.elapsed();
            eprintln!(
                "{count} instructions: {:?} ({:?}/insn)",
                elapsed,
                elapsed / count as u32
            );
        }
    }

    /// Steady-state throughput: a loop is lifted once and then re-executed, so
    /// this isolates interpretation cost from discovery cost.
    ///
    /// Reports the *best* of several runs. A mean is meaningless on a machine
    /// doing anything else — an unrelated background build moved this figure by
    /// more than 2x — whereas the minimum approximates the uncontended cost.
    #[test]
    fn hot_loop_throughput() {
        let mut best: Option<(std::time::Duration, u64)> = None;
        for _ in 0..5 {
            // mov ecx, 2000 ; loop: dec ecx ; jnz loop
            let mut vm = machine(&[
                0xb9, 0xd0, 0x07, 0x00, 0x00, // mov ecx, 2000
                0xff, 0xc9, // dec ecx
                0x75, 0xfc, // jnz -4
            ]);
            let start = std::time::Instant::now();
            vm.run(4_000_000);
            let elapsed = start.elapsed();
            let ctx = vm.context().clone();
            assert_eq!(
                vm.emulator().read_varnode_by_name(&ctx, "ECX"),
                Some(0),
                "the loop must actually run to completion"
            );
            if best.is_none_or(|(previous, _)| elapsed < previous) {
                best = Some((elapsed, vm.steps));
            }
        }
        let (elapsed, steps) = best.expect("at least one run");
        // 2000 iterations of two instructions, plus the setup instruction.
        let instructions = 2000 * 2 + 1;
        eprintln!(
            "loop best-of-5: {steps} steps in {elapsed:?} \
             ({:.2}M pcode-ops/s, {:.2}M guest-insn/s)",
            steps as f64 / elapsed.as_secs_f64() / 1e6,
            f64::from(instructions) / elapsed.as_secs_f64() / 1e6,
        );
    }

    #[test]
    fn optimisation_round_effect() {
        // Flag-heavy arithmetic: each of these writes six status flags that the
        // next instruction overwrites without reading.
        let code = [
            0x01, 0xd8, // add eax, ebx
            0x29, 0xd8, // sub eax, ebx
            0x01, 0xd8, // add eax, ebx
            0x31, 0xd8, // xor eax, ebx
        ];
        for optimize in [false, true] {
            let source = SleighCodeSource::new(spec());
            let ctx = source.new_context();
            let mut memory = VmMemory::new();
            memory
                .mmu
                .write_unchecked(0x1000, &code, perm::READ | perm::EXEC);
            let mut vm = Vm::at_address(ctx, 0x1000, source, memory).expect("entry decodes");
            vm.optimize = optimize;
            vm.run(4096);
            let insns: usize = vm
                .context()
                .block_ids()
                .into_iter()
                .map(|b| vm.context().block(b).instruction_ids().len())
                .sum();
            eprintln!(
                "optimize={optimize}: steps={} module_insns={insns} cleanup={:?}",
                vm.steps, vm.cleanup
            );
        }
    }

    /// The cleanup round must be invisible: optimised and unoptimised runs of
    /// the same code have to agree on every architectural register and flag.
    /// This is the guard that makes an IR-rewriting pass safe to run by default.
    #[test]
    fn optimisation_preserves_architectural_state() {
        let programs: [&[u8]; 4] = [
            // Arithmetic and the flags it writes.
            &[0xb8, 0x39, 0x05, 0x00, 0x00, 0x01, 0xd8, 0x29, 0xd8, 0x31, 0xd8],
            // Shifts and rotates, which lean hard on temporaries.
            &[0xb8, 0xff, 0x00, 0x00, 0x00, 0xc1, 0xe0, 0x03, 0xd1, 0xe8, 0xc1, 0xc0, 0x05],
            // Multiply, and a byte-granular compare.
            &[0xb8, 0x07, 0x00, 0x00, 0x00, 0xbb, 0x09, 0x00, 0x00, 0x00, 0x0f, 0xaf, 0xc3, 0x38, 0xd8],
            // A loop, so the optimised block is re-entered many times.
            &[0xb9, 0x64, 0x00, 0x00, 0x00, 0xff, 0xc9, 0x75, 0xfc],
        ];
        let watched = [
            "RAX", "RBX", "RCX", "EAX", "EBX", "ECX", "CF", "ZF", "SF", "OF", "AF", "PF",
        ];

        for (index, code) in programs.iter().enumerate() {
            let mut states = Vec::new();
            for optimize in [false, true] {
                let source = SleighCodeSource::new(spec());
                let ctx = source.new_context();
                let mut memory = VmMemory::new();
                memory
                    .mmu
                    .write_unchecked(0x1000, code, perm::READ | perm::EXEC);
                let mut vm = Vm::at_address(ctx, 0x1000, source, memory).expect("entry decodes");
                vm.optimize = optimize;
                vm.run(100_000);
                let ctx = vm.context().clone();
                let state: Vec<_> = watched
                    .iter()
                    .map(|name| (*name, vm.emulator().read_varnode_by_name(&ctx, name)))
                    .collect();
                states.push(state);
            }
            assert_eq!(
                states[0], states[1],
                "program {index} diverged between unoptimised and optimised runs"
            );
            // A test that observed nothing would pass vacuously.
            assert!(
                states[0].iter().any(|(_, value)| value.is_some()),
                "program {index} observed no registers at all"
            );
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
