//! The run loop: a loaded image, a machine, and the host services behind it.
//!
//! A [`Process`] is one or more tasks — the program that was loaded and
//! whatever it forked — scheduled cooperatively on one host thread: a task
//! runs until it exits, forks, or has to wait (for a child, or on a pipe),
//! and the next runnable task takes over. See `Task` for what a fork
//! copies.
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

use crate::errno::{ENOEXEC, Errno};
use crate::fs::{Files, Stdio};
use crate::loader::{self, LoadedImage, page_up};
use crate::regs::Regs;
use crate::stack::{self, Identity};

/// The pid of the first task; children count up from it.
pub const ROOT_PID: u64 = 4242;

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

/// The machine of one task: what `execve` replaces and what a fork copies.
pub(crate) struct Machine {
    pub(crate) vm: Vm<SleighCodeSource<'static>>,
    pub(crate) regs: Regs,
    pub(crate) space: AddressSpace,
    pub(crate) image: LoadedImage,
}

impl Machine {
    /// Loads `image` and prepares a machine at its entry point.
    fn boot(
        image: &[u8],
        argv: &[String],
        envp: &[String],
        identity: Identity,
        jit: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
        let ctx = source.new_context();
        let regs = Regs::resolve(&ctx)?;
        let mut memory = VmMemory::new();
        let loaded = loader::load(image, &mut memory.mmu)?;
        let random = 0x9e37_79b9_7f4a_7c15_u128.to_le_bytes();
        let rsp = stack::build(&mut memory.mmu, &loaded, argv, envp, identity, random);

        let mut vm = Vm::at_address(ctx, loaded.entry, source, memory)
            .map_err(|e| format!("cannot lift the entry point {:#x}: {e:?}", loaded.entry))?;
        regs.reset(vm.memory_mut());
        regs.rsp.write(vm.memory_mut(), rsp);
        if jit {
            vm.set_block_executor(Box::new(qcode_jit::Jit::new()));
        }
        Ok(Self {
            vm,
            regs,
            space: AddressSpace {
                brk_start: loaded.brk,
                brk: loaded.brk,
                mmap_next: MMAP_BASE,
            },
            image: loaded,
        })
    }

    /// A copy of this machine positioned at `pc`, for a fork: memory is
    /// copied, lifted code is not carried over.
    fn duplicate_at(&mut self, pc: u64, jit: bool) -> Result<Self, String> {
        let memory = self.vm.memory_mut().clone();
        let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
        let ctx = source.new_context();
        let regs = Regs::resolve(&ctx)?;
        let mut vm = Vm::at_address(ctx, pc, source, memory)
            .map_err(|e| format!("cannot lift the fork return point {pc:#x}: {e:?}"))?;
        if jit {
            vm.set_block_executor(Box::new(qcode_jit::Jit::new()));
        }
        Ok(Self {
            vm,
            regs,
            space: self.space,
            image: self.image.clone(),
        })
    }
}

/// What a system call asked of the scheduler, beyond a value in `RAX`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Request {
    /// Create a child; the caller has not been resumed. `vfork` runs the
    /// child before the parent gets another turn.
    Fork { vfork: bool },
    /// The call cannot complete yet; the caller has not been resumed and
    /// re-issues it when it next runs.
    Block,
    /// End another task as if by a fatal signal.
    Kill { pid: u64, signal: i32 },
}

/// Why a task stopped running for now.
enum TaskExit {
    Exited(i32),
    Crashed(Crash),
    Budget,
    Requested(Request),
}

/// One thread of execution with its own machine and descriptor table.
pub(crate) struct Task {
    pub(crate) machine: Machine,
    pub(crate) files: Files,
    pub(crate) pid: u64,
    pub(crate) ppid: u64,
    /// Children not yet reaped, live or exited.
    pub(crate) children: Vec<u64>,
    /// Exited children awaiting `wait4`, with their wait status.
    pub(crate) zombies: Vec<(u64, i32)>,
    pub(crate) jit: bool,
    pub(crate) trace: bool,
    pub(crate) exit_code: Option<i32>,
    pub(crate) request: Option<Request>,
    pub(crate) started: Instant,
    pub(crate) rng: u64,
    pub(crate) signal_mask: u64,
    pub(crate) sigactions: std::collections::HashMap<u64, [u8; 32]>,
    pub(crate) identity: Identity,
}

impl Task {
    fn new(image: &[u8], config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        let identity = Identity {
            uid: 1000,
            gid: 1000,
        };
        let machine = Machine::boot(image, &config.argv, &config.envp, identity, config.jit)?;
        let files = Files::new(config.stdio, config.root, config.exe_path);
        Ok(Self {
            machine,
            files,
            pid: ROOT_PID,
            ppid: 1,
            children: Vec::new(),
            zombies: Vec::new(),
            jit: config.jit,
            trace: config.trace,
            exit_code: None,
            request: None,
            started: Instant::now(),
            rng: 0x2545_f491_4f6c_dd1d,
            signal_mask: 0,
            sigactions: Default::default(),
            identity,
        })
    }

    pub(crate) fn vm(&mut self) -> &mut Vm<SleighCodeSource<'static>> {
        &mut self.machine.vm
    }

    pub(crate) fn image(&self) -> &LoadedImage {
        &self.machine.image
    }

    pub(crate) fn mmu(&mut self) -> &mut Mmu {
        &mut self.machine.vm.memory_mut().mmu
    }

    fn steps(&self) -> u64 {
        self.machine.vm.stats.steps
    }

    /// The child of this task, which is stopped at a `fork`: the same memory
    /// and descriptors, `RAX` zero, positioned after the `syscall`
    /// instruction. The parent is left stopped for the caller to resume.
    fn fork(&mut self, pid: u64) -> Result<Task, Crash> {
        let Some(pc) = self.pc() else {
            return Err(self.crash("fork from an unknown position".to_owned(), None));
        };
        // `syscall` is two bytes and lifts to the one op the machine sits at.
        let next = pc + 2;
        let vm = &mut self.machine.vm;
        let regs = &self.machine.regs;
        regs.rax.write(vm.memory_mut(), 0);
        let machine = match self.machine.duplicate_at(next, self.jit) {
            Ok(machine) => machine,
            Err(reason) => return Err(self.crash(reason, None)),
        };
        let files = match self.files.fork() {
            Ok(files) => files,
            Err(e) => return Err(self.crash(format!("cannot duplicate descriptors: {e}"), None)),
        };
        Ok(Task {
            machine,
            files,
            pid,
            ppid: self.pid,
            children: Vec::new(),
            zombies: Vec::new(),
            jit: self.jit,
            trace: self.trace,
            exit_code: None,
            request: None,
            started: self.started,
            rng: self.rng,
            signal_mask: self.signal_mask,
            sigactions: self.sigactions.clone(),
            identity: self.identity,
        })
    }

    /// Replaces the machine with a fresh one running `image`, as `execve`
    /// does: descriptors survive except close-on-exec ones, handlers reset.
    pub(crate) fn exec(
        &mut self,
        image: &[u8],
        host_path: String,
        argv: &[String],
        envp: &[String],
    ) -> Result<(), Errno> {
        if image.len() < 4 || &image[..4] != b"\x7fELF" {
            return Err(ENOEXEC);
        }
        let machine =
            Machine::boot(image, argv, envp, self.identity, self.jit).map_err(|_| ENOEXEC)?;
        self.machine = machine;
        self.files.exe_path = host_path;
        self.files.close_on_exec();
        self.sigactions.clear();
        Ok(())
    }

    /// Runs the task for up to `budget` p-code operations.
    fn run(&mut self, budget: u64) -> TaskExit {
        if let Some(code) = self.exit_code {
            return TaskExit::Exited(code);
        }
        let deadline = self.steps().saturating_add(budget);
        loop {
            let remaining = deadline.saturating_sub(self.steps());
            if remaining == 0 {
                return TaskExit::Budget;
            }
            match self.machine.vm.run(remaining) {
                VmExit::InstructionLimit => return TaskExit::Budget,
                VmExit::Breakpoint(_) | VmExit::HookStop(_) => {}
                VmExit::Interrupt(interrupt) => match self.handle_interrupt(&interrupt) {
                    Ok(()) => {
                        if let Some(code) = self.exit_code {
                            return TaskExit::Exited(code);
                        }
                        if let Some(request) = self.request.take() {
                            return TaskExit::Requested(request);
                        }
                    }
                    Err(crash) => return TaskExit::Crashed(crash),
                },
                VmExit::Error(message) => {
                    return TaskExit::Crashed(self.crash(message.to_string(), None));
                }
                VmExit::Fault(fault) => {
                    return TaskExit::Crashed(self.crash(fault.to_string(), Some(11)));
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
                    return TaskExit::Crashed(self.crash(reason, signal));
                }
            }
        }
    }

    /// Supplies the effect of the operation the machine stopped at and steps
    /// past it — unless the call is a request to the scheduler or has to
    /// wait, in which case the machine stays stopped at it.
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
                if let Some(Request::Fork { .. } | Request::Block) = self.request {
                    return Ok(());
                }
                if self.machine.vm.pending_interrupt().is_none() {
                    // `execve` replaced the machine; there is nothing to resume.
                    return Ok(());
                }
                let vm = &mut self.machine.vm;
                self.machine.regs.rax.write(vm.memory_mut(), result);
                None
            }
            // Monotonic, and advancing with work done rather than wall time
            // so a run is reproducible.
            "rdtsc" => Some(u128::from(self.steps().wrapping_mul(8))),
            name if name.starts_with("cpuid") => {
                let vm = &mut self.machine.vm;
                let leaf = self.machine.regs.rax.read(vm.memory_mut()) as u32;
                let sub = self.machine.regs.rcx.read(vm.memory_mut()) as u32;
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
        self.machine
            .vm
            .resume(value)
            .map_err(|error| self.crash(error.to_string(), None))
    }

    /// The parent's side of a fork, once the child exists: `RAX` is the
    /// child's pid and the machine moves on.
    fn resume_after_fork(&mut self, child: u64) -> Result<(), Crash> {
        let vm = &mut self.machine.vm;
        self.machine.regs.rax.write(vm.memory_mut(), child);
        vm.resume(None)
            .map_err(|error| self.crash(error.to_string(), None))
    }

    /// The guest address of the instruction the machine is positioned at.
    pub(crate) fn pc(&mut self) -> Option<u64> {
        let vm = &mut self.machine.vm;
        let block = vm.emulator().block;
        let idx = vm.emulator().idx;
        let ctx = vm.context();
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

    pub(crate) fn registers(&mut self) -> Vec<(&'static str, u64)> {
        let vm = &mut self.machine.vm;
        self.machine
            .regs
            .named()
            .into_iter()
            .map(|(name, reg)| (name, reg.read(vm.memory_mut())))
            .collect()
    }

    pub(crate) fn crash(&mut self, reason: String, signal: Option<i32>) -> Crash {
        Crash {
            pc: self.pc(),
            reason,
            signal,
            registers: self.registers(),
        }
    }

    /// Grows or shrinks the heap to end at `new_brk`, returning the resulting
    /// break — unchanged if the request could not be honoured.
    pub(crate) fn set_brk(&mut self, new_brk: u64) -> u64 {
        let AddressSpace { brk_start, brk, .. } = self.machine.space;
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
        self.machine.space.brk = new_brk;
        new_brk
    }
}

/// A loaded program and everything it forks, run on one host thread.
pub struct Process {
    /// Slot 0 is the initial task and is never dropped; a child's slot is
    /// emptied once it has exited and been reaped or orphaned.
    tasks: Vec<Option<Task>>,
    current: usize,
    next_pid: u64,
    /// Steps retired by tasks that have since been dropped.
    retired: u64,
}

impl Process {
    /// Loads `image` and prepares the machine at its entry point.
    pub fn new(image: &[u8], config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            tasks: vec![Some(Task::new(image, config)?)],
            current: 0,
            next_pid: ROOT_PID + 1,
            retired: 0,
        })
    }

    fn root(&self) -> &Task {
        self.tasks[0]
            .as_ref()
            .expect("the initial task is never dropped")
    }

    fn root_mut(&mut self) -> &mut Task {
        self.tasks[0]
            .as_mut()
            .expect("the initial task is never dropped")
    }

    /// The initial task's descriptor table, which also holds the captured
    /// stdio every task shares.
    pub fn files(&self) -> &Files {
        &self.root().files
    }

    pub fn files_mut(&mut self) -> &mut Files {
        &mut self.root_mut().files
    }

    pub fn image(&self) -> &LoadedImage {
        self.root().image()
    }

    /// The initial task's machine.
    pub fn vm(&mut self) -> &mut Vm<SleighCodeSource<'static>> {
        self.root_mut().vm()
    }

    /// The guest address the initial task is positioned at.
    pub fn pc(&mut self) -> Option<u64> {
        self.root_mut().pc()
    }

    pub fn registers(&mut self) -> Vec<(&'static str, u64)> {
        self.root_mut().registers()
    }

    /// Operations retired so far, across every task and both strategies.
    pub fn steps(&self) -> u64 {
        self.retired + self.tasks.iter().flatten().map(Task::steps).sum::<u64>()
    }

    /// The first live task at or after `from`, wrapping around.
    fn next_live(&self, from: usize) -> usize {
        let n = self.tasks.len();
        (0..n)
            .map(|k| (from + k) % n)
            .find(|&i| {
                self.tasks[i]
                    .as_ref()
                    .is_some_and(|t| t.exit_code.is_none())
            })
            .expect("the initial task is live")
    }

    fn live_count(&self) -> usize {
        self.tasks
            .iter()
            .flatten()
            .filter(|t| t.exit_code.is_none())
            .count()
    }

    /// Runs the process for up to `budget` p-code operations in total, until
    /// the initial task exits.
    pub fn run(&mut self, budget: u64) -> ProcessExit {
        if let Some(code) = self.root().exit_code {
            return ProcessExit::Exited(code);
        }
        let deadline = self.steps().saturating_add(budget);
        // Consecutive turns in which a task retired nothing and could not
        // proceed; as many in a row as there are live tasks means nobody can.
        let mut blocked_streak = 0;
        loop {
            let remaining = deadline.saturating_sub(self.steps());
            if remaining == 0 {
                return ProcessExit::Budget;
            }
            let index = self.next_live(self.current);
            self.current = index;
            let task = self.tasks[index].as_mut().expect("live");
            let before = task.steps();
            let exit = task.run(remaining);
            if !matches!(exit, TaskExit::Requested(Request::Block)) || task.steps() != before {
                blocked_streak = 0;
            }
            match exit {
                TaskExit::Budget => return ProcessExit::Budget,
                TaskExit::Exited(code) => {
                    if index == 0 {
                        return ProcessExit::Exited(code);
                    }
                    self.reap(index, (code & 0xff) << 8);
                    self.current = index + 1;
                }
                TaskExit::Crashed(crash) => {
                    if index == 0 {
                        return ProcessExit::Crashed(crash);
                    }
                    log::warn!("child {}: {crash}", task.pid);
                    let status = crash.signal.unwrap_or(9);
                    self.tasks[index].as_mut().expect("live").exit_code = Some(128 + status);
                    self.reap(index, status);
                    self.current = index + 1;
                }
                TaskExit::Requested(Request::Block) => {
                    blocked_streak += 1;
                    if blocked_streak >= self.live_count() {
                        let root = self.root_mut();
                        return ProcessExit::Crashed(root.crash(
                            "every task is waiting on another: deadlock".to_owned(),
                            None,
                        ));
                    }
                    self.current = index + 1;
                }
                TaskExit::Requested(Request::Fork { vfork }) => {
                    let pid = self.next_pid;
                    self.next_pid += 1;
                    let parent = self.tasks[index].as_mut().expect("live");
                    let child = match parent.fork(pid) {
                        Ok(child) => child,
                        Err(crash) => {
                            if index == 0 {
                                return ProcessExit::Crashed(crash);
                            }
                            log::warn!("child {}: {crash}", parent.pid);
                            parent.exit_code = Some(128 + 9);
                            self.reap(index, 9);
                            continue;
                        }
                    };
                    if let Err(crash) = parent.resume_after_fork(pid) {
                        return ProcessExit::Crashed(crash);
                    }
                    parent.children.push(pid);
                    let slot = (1..self.tasks.len())
                        .find(|&i| self.tasks[i].is_none())
                        .unwrap_or_else(|| {
                            self.tasks.push(None);
                            self.tasks.len() - 1
                        });
                    self.tasks[slot] = Some(child);
                    // The child runs first, vfork or not: a vfork parent must
                    // not proceed before it, and a pipeline fills from its
                    // producer.
                    let _ = vfork;
                    self.current = slot;
                }
                TaskExit::Requested(Request::Kill { pid, signal }) => {
                    if let Some(victim) = self.tasks.iter().position(|t| {
                        t.as_ref()
                            .is_some_and(|t| t.pid == pid && t.exit_code.is_none())
                    }) {
                        if victim == 0 {
                            return ProcessExit::Exited(128 + signal);
                        }
                        self.tasks[victim].as_mut().expect("live").exit_code = Some(128 + signal);
                        self.reap(victim, signal);
                    }
                    self.current = index;
                }
            }
        }
    }

    /// Records the exit of the task in `slot` for its parent, hands its
    /// children to the initial task, and frees its machine.
    fn reap(&mut self, slot: usize, status: i32) {
        let task = self.tasks[slot].take().expect("live");
        self.retired += task.steps();
        let (pid, ppid) = (task.pid, task.ppid);
        let orphans = task.children.clone();
        drop(task);
        // A task's parent is live: an exiting parent hands its children to
        // the initial task, which is never dropped.
        if let Some(parent) = self
            .tasks
            .iter_mut()
            .flatten()
            .find(|t| t.pid == ppid && t.exit_code.is_none())
        {
            parent.zombies.push((pid, status));
        }
        if !orphans.is_empty() {
            for t in self.tasks.iter_mut().flatten() {
                if orphans.contains(&t.pid) {
                    t.ppid = ROOT_PID;
                }
            }
            self.root_mut().children.extend(orphans);
        }
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
