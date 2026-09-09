//! The run loop: a loaded image, a machine, and the host services behind it.
//!
//! # How a system call reaches the host
//!
//! When the interpreter meets a user p-code operation it has no semantics
//! for — `syscall`, `rdtsc`, the `cpuid_*` family,
//! `invalidInstructionException` for `ud2` — [`Vm::run`] returns
//! [`VmExit::Interrupt`] with the machine positioned **at** that operation:
//! every instruction before it has retired, nothing after it has, and the
//! block's terminator has not run. The exit names the operation, its
//! instruction and the width of the result it declares.
//!
//! To resume, the environment applies the operation's effect and calls
//! [`Vm::resume`]. For `syscall`, which produces no value, the effect is a
//! write to `RAX`. For a value-producing op such as `rdtsc` (an `i64`) or
//! `cpuid_*` (an `i128` packed as `EAX | EBX << 32 | EDX << 64 | ECX << 96`)
//! the value is handed to `resume`, which files it under the op's
//! instruction so the register stores that follow in the same block read it.
//!
//! The rule holds with and without the JIT installed: compiled code runs the
//! part of a block before such an op natively and hands the op itself to the
//! interpreter, which raises the same exit.

use std::path::PathBuf;
use std::time::Instant;

use qcode::value::BasicBlock;
use qcode_vm::{Interrupt, InterruptKind, Mmu, PAGE_SIZE, Vm, VmExit, VmMemory, perm};
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

use crate::fs::{Files, Stdio};
use crate::loader::{self, LoadedImage, page_up};
use crate::regs::Regs;
use crate::stack::{self, Identity};

/// How a process is set up.
#[derive(Debug, Clone)]
pub struct Config {
    pub argv: Vec<String>,
    pub envp: Vec<String>,
    /// Install the Cranelift JIT as the block executor.
    pub jit: bool,
    /// Print one line per system call to stderr, `strace`-style.
    pub trace: bool,
    /// Sandbox root for guest paths; `None` exposes the host filesystem.
    pub root: Option<PathBuf>,
    pub stdio: Stdio,
    /// The host path of the executable, for `/proc/self/exe`.
    pub exe_path: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            argv: vec!["prog".to_owned()],
            envp: Vec::new(),
            jit: false,
            trace: false,
            root: None,
            stdio: Stdio::Host,
            exe_path: "/prog".to_owned(),
        }
    }
}

/// Why a guest stopped for a reason other than exiting.
#[derive(Debug, Clone)]
pub struct Crash {
    pub pc: Option<u64>,
    pub reason: String,
    /// The signal Linux would have delivered, when there is an obvious one.
    pub signal: Option<i32>,
    pub registers: Vec<(&'static str, u64)>,
}

impl std::fmt::Display for Crash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.pc {
            Some(pc) => write!(f, "guest crashed at {pc:#x}: {}", self.reason)?,
            None => write!(f, "guest crashed: {}", self.reason)?,
        }
        if let Some(signal) = self.signal {
            write!(f, " (signal {signal})")?;
        }
        for (i, (name, value)) in self.registers.iter().enumerate() {
            if i % 4 == 0 {
                writeln!(f)?;
            }
            write!(f, "  {name:>7}={value:#018x}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Crash {}

#[derive(Debug, Clone)]
pub enum ProcessExit {
    /// The guest called `exit`/`exit_group` with this status.
    Exited(i32),
    Crashed(Crash),
    /// The operation budget ran out; the process may be run again.
    Budget,
}

/// The parts of the address space the environment manages.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AddressSpace {
    pub brk_start: u64,
    pub brk: u64,
    /// Where the next hint-less `mmap` is tried.
    pub mmap_next: u64,
}

/// Hint-less mappings start here and grow upward, well below the stack.
const MMAP_BASE: u64 = 0x7f00_0000_0000;

impl AddressSpace {
    /// The lowest page-aligned free range of `len` bytes at or above the
    /// cursor.
    pub fn find_free(&mut self, mmu: &Mmu, len: u64) -> Option<u64> {
        let mut start = self.mmap_next;
        loop {
            if start.checked_add(len)? >= stack::STACK_TOP - stack::STACK_SIZE {
                return None;
            }
            match (start..start + len)
                .step_by(PAGE_SIZE as usize)
                .find(|&page| mmu.permissions(page) & perm::MAP != 0)
            {
                None => {
                    self.mmap_next = start + len;
                    return Some(start);
                }
                Some(used) => start = used + PAGE_SIZE,
            }
        }
    }

    pub fn is_free(mmu: &Mmu, addr: u64, len: u64) -> bool {
        (addr..addr + len)
            .step_by(PAGE_SIZE as usize)
            .all(|page| mmu.permissions(page) & perm::MAP == 0)
    }
}

pub struct Process {
    pub(crate) vm: Vm<SleighCodeSource<'static>>,
    pub(crate) regs: Regs,
    pub files: Files,
    pub(crate) space: AddressSpace,
    pub(crate) image: LoadedImage,
    pub(crate) trace: bool,
    pub(crate) exit_code: Option<i32>,
    pub(crate) started: Instant,
    pub(crate) rng: u64,
    pub(crate) signal_mask: u64,
    pub(crate) sigactions: std::collections::HashMap<u64, [u8; 32]>,
    pub(crate) identity: Identity,
}

impl Process {
    /// Loads `image` and prepares the machine at its entry point.
    pub fn new(image: &[u8], config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
        let ctx = source.new_context();
        let regs = Regs::resolve(&ctx)?;
        let mut memory = VmMemory::new();
        let loaded = loader::load(image, &mut memory.mmu)?;
        let identity = Identity {
            uid: 1000,
            gid: 1000,
        };
        let random = 0x9e37_79b9_7f4a_7c15_u128.to_le_bytes();
        let rsp = stack::build(
            &mut memory.mmu,
            &loaded,
            &config.argv,
            &config.envp,
            identity,
            random,
        );

        let mut vm = Vm::at_address(ctx, loaded.entry, source, memory)
            .map_err(|e| format!("cannot lift the entry point {:#x}: {e:?}", loaded.entry))?;
        for (_, reg) in regs.named() {
            reg.write(vm.memory_mut(), 0);
        }
        regs.rsp.write(vm.memory_mut(), rsp);
        if config.jit {
            vm.set_block_executor(Box::new(qcode_jit::Jit::new()));
        }

        let files = Files::new(config.stdio, config.root, config.exe_path);
        Ok(Self {
            vm,
            regs,
            files,
            space: AddressSpace {
                brk_start: loaded.brk,
                brk: loaded.brk,
                mmap_next: MMAP_BASE,
            },
            image: loaded,
            trace: config.trace,
            exit_code: None,
            started: Instant::now(),
            rng: 0x2545_f491_4f6c_dd1d,
            signal_mask: 0,
            sigactions: Default::default(),
            identity,
        })
    }

    pub fn image(&self) -> &LoadedImage {
        &self.image
    }

    pub fn vm(&mut self) -> &mut Vm<SleighCodeSource<'static>> {
        &mut self.vm
    }

    /// Operations retired so far, across both strategies.
    pub fn steps(&self) -> u64 {
        self.vm.stats.steps
    }

    /// Runs the guest for up to `budget` p-code operations.
    pub fn run(&mut self, budget: u64) -> ProcessExit {
        if let Some(code) = self.exit_code {
            return ProcessExit::Exited(code);
        }
        let deadline = self.vm.stats.steps.saturating_add(budget);
        loop {
            let remaining = deadline.saturating_sub(self.vm.stats.steps);
            if remaining == 0 {
                return ProcessExit::Budget;
            }
            match self.vm.run(remaining) {
                VmExit::InstructionLimit => return ProcessExit::Budget,
                VmExit::Breakpoint(_) | VmExit::HookStop(_) => {}
                VmExit::Interrupt(interrupt) => match self.handle_interrupt(&interrupt) {
                    Ok(()) => {
                        if let Some(code) = self.exit_code {
                            return ProcessExit::Exited(code);
                        }
                    }
                    Err(crash) => return ProcessExit::Crashed(crash),
                },
                VmExit::Error(message) => {
                    return ProcessExit::Crashed(self.crash(message.to_string(), None));
                }
                VmExit::Fault(fault) => {
                    return ProcessExit::Crashed(self.crash(fault.to_string(), Some(11)));
                }
                VmExit::Unlifted { addr, error } => {
                    let reason = match &error {
                        qcode_vm::CodeError::Fault(fault) => {
                            format!("cannot fetch code at {addr:#x}: {fault}")
                        }
                        qcode_vm::CodeError::Decode(e) => {
                            format!("cannot lift code at {addr:#x}: {e}")
                        }
                    };
                    let signal = match error {
                        qcode_vm::CodeError::Fault(_) => Some(11),
                        qcode_vm::CodeError::Decode(_) => Some(4),
                    };
                    return ProcessExit::Crashed(self.crash(reason, signal));
                }
            }
        }
    }

    /// Supplies the effect of the operation the machine stopped at and steps
    /// past it.
    fn handle_interrupt(&mut self, interrupt: &Interrupt) -> Result<(), Crash> {
        let name = match &interrupt.kind {
            InterruptKind::Intrinsic { name, .. } => name.as_ref(),
            InterruptKind::Explicit { code } => {
                return Err(self.crash(format!("unexpected vm.interrupt({code})"), None));
            }
        };
        let value = match name {
            "syscall" => {
                let result = self.syscall();
                self.regs.rax.write(self.vm.memory_mut(), result);
                None
            }
            // Monotonic, and advancing with work done rather than wall time
            // so a run is reproducible.
            "rdtsc" => Some(u128::from(self.vm.stats.steps.wrapping_mul(8))),
            name if name.starts_with("cpuid") => {
                let leaf = self.regs.rax.read(self.vm.memory_mut()) as u32;
                let sub = self.regs.rcx.read(self.vm.memory_mut()) as u32;
                let [eax, ebx, ecx, edx] = cpuid(leaf, sub);
                Some(
                    u128::from(eax)
                        | u128::from(ebx) << 32
                        | u128::from(edx) << 64
                        | u128::from(ecx) << 96,
                )
            }
            "invalidInstructionException" => {
                return Err(self.crash("illegal instruction".to_owned(), Some(4)));
            }
            other => {
                return Err(self.crash(format!("unsupported p-code operation `{other}`"), None));
            }
        };
        self.vm
            .resume(value)
            .map_err(|error| self.crash(error.to_string(), None))
    }

    /// The guest address of the instruction the machine is positioned at.
    pub fn pc(&mut self) -> Option<u64> {
        let block = self.vm.emulator().block;
        let idx = self.vm.emulator().idx;
        let ctx = self.vm.context();
        if !ctx.contains_block(block) {
            return None;
        }
        let block = BasicBlock::from_id(ctx, block);
        block
            .instructions()
            .nth(idx)
            .and_then(|insn| insn.address())
            .or_else(|| block.address())
            .or_else(|| block.instructions().find_map(|insn| insn.address()))
    }

    pub fn registers(&mut self) -> Vec<(&'static str, u64)> {
        self.regs
            .named()
            .into_iter()
            .map(|(name, reg)| (name, reg.read(self.vm.memory_mut())))
            .collect()
    }

    fn crash(&mut self, reason: String, signal: Option<i32>) -> Crash {
        Crash {
            pc: self.pc(),
            reason,
            signal,
            registers: self.registers(),
        }
    }

    pub(crate) fn mmu(&mut self) -> &mut Mmu {
        &mut self.vm.memory_mut().mmu
    }

    /// Grows or shrinks the heap to end at `new_brk`, returning the resulting
    /// break — unchanged if the request could not be honoured.
    pub(crate) fn set_brk(&mut self, new_brk: u64) -> u64 {
        let AddressSpace { brk_start, brk, .. } = self.space;
        if new_brk < brk_start {
            return brk;
        }
        let (old_end, new_end) = (page_up(brk), page_up(new_brk));
        let mmu = self.mmu();
        if new_end > old_end {
            if !AddressSpace::is_free(mmu, old_end, new_end - old_end)
                || mmu.map(old_end, new_end - old_end, perm::RW_INIT).is_err()
            {
                return brk;
            }
        } else if new_end < old_end && mmu.unmap(new_end, old_end - new_end).is_err() {
            return brk;
        }
        self.space.brk = new_brk;
        new_brk
    }
}

/// A deliberately plain CPU: the baseline x86-64 feature set and nothing that
/// would steer a runtime's dispatch towards code the emulator does not lift.
fn cpuid(leaf: u32, _sub: u32) -> [u32; 4] {
    match leaf {
        // Highest basic leaf, and "GenuineIntel" in EBX, EDX, ECX order.
        0 => [7, 0x756e_6547, 0x6c65_746e, 0x4965_6e69],
        // Family 6 model 0x3a stepping 9; FPU, TSC, CX8, CMOV, MMX, FXSR, SSE,
        // SSE2 in EDX. Nothing in ECX.
        1 => [
            0x0003_06a9,
            0x0000_0800,
            0,
            (1 << 0)
                | (1 << 4)
                | (1 << 8)
                | (1 << 15)
                | (1 << 23)
                | (1 << 24)
                | (1 << 25)
                | (1 << 26),
        ],
        0x8000_0000 => [0x8000_0008, 0, 0, 0],
        // Long mode, NX.
        0x8000_0001 => [0, 0, 0, (1 << 29) | (1 << 20)],
        0x8000_0008 => [0x3030, 0, 0, 0],
        _ => [0, 0, 0, 0],
    }
}
