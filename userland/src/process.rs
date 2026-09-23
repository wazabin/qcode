//! The run loop: a loaded image, a machine, and the host services behind it.
//!
//! A [`Process`] is one or more tasks — the program that was loaded and
//! whatever it forked — scheduled cooperatively on one host thread: a task
//! runs until it exits, forks, has to wait (for a child, or on a pipe), or
//! has used its slice, and the next runnable task takes over. Every task
//! runs on the one [`Machine`] of the process: one `Vm`, one module of
//! lifted code, one set of hooks and state spaces. A switch parks the task
//! on the machine — its position, registers and address space — and
//! installs the next one's. See `Task` for what a fork copies.
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

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use qcode::space::{MemorySpaceId, SpaceId};
use qcode::value::BasicBlock;
use qcode_vm::flat::FlatSpace;
use qcode_vm::{
    FaultKind, Interrupt, InterruptKind, Mmu, MmuSnapshot, PAGE_SIZE, Parked, Vm, VmExit, VmMemory,
    perm,
};
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

use crate::errno::{self, ENOEXEC, Errno};
use crate::fs::{Files, Stdio};
use crate::guest;
use crate::loader::{self, LoadedImage, page_up};
use crate::regs::Regs;
use crate::signal::{
    self, SEGV_ACCERR, SEGV_MAPERR, SI_KERNEL, SIGFPE, SIGILL, SIGSEGV, SIGTRAP, SigInfo,
};
use crate::stack::{self, Identity};
use crate::syscall::SharedMap;
use crate::traps::{self, DivideHook, StepHook};

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

/// The one machine of a process, which every task takes turns on.
pub(crate) struct Machine {
    pub(crate) vm: Vm<SleighCodeSource<'static>>,
    pub(crate) regs: Regs,
    /// The flat spaces that belong to the task on the machine: registers,
    /// and the architecture's private and temporary spaces. Every space the
    /// module had at boot; a space a hook adds later is shared by every task.
    arch_spaces: Vec<MemorySpaceId>,
    /// The single-step flag's space, once a tracer first stepped a task.
    /// Shared by every task; the one put on the machine writes its own.
    step_space: Option<SpaceId>,
}

/// What a task leaves the machine with when another takes it.
struct Saved {
    parked: Parked,
    memory: MmuSnapshot,
    spaces: Vec<(MemorySpaceId, FlatSpace)>,
}

/// The stack's random bytes, fixed so a run is reproducible.
const AT_RANDOM: [u8; 16] = 0x9e37_79b9_7f4a_7c15_u128.to_le_bytes();

/// Maps `image` and its stack into `mmu`, returning the image and the
/// initial stack pointer.
fn load_into(
    mmu: &mut Mmu,
    image: &[u8],
    argv: &[String],
    envp: &[String],
    identity: Identity,
) -> Result<(LoadedImage, u64), loader::LoadError> {
    let loaded = loader::load(image, mmu)?;
    let rsp = stack::build(mmu, &loaded, argv, envp, identity, AT_RANDOM);
    Ok((loaded, rsp))
}

impl Machine {
    /// Loads `image` and prepares a machine at its entry point.
    fn boot(
        image: &[u8],
        argv: &[String],
        envp: &[String],
        identity: Identity,
        jit: bool,
    ) -> Result<(Self, LoadedImage), Box<dyn std::error::Error>> {
        let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
        let ctx = source.new_context();
        let regs = Regs::resolve(&ctx)?;
        let mut memory = VmMemory::new();
        let (loaded, rsp) = load_into(&mut memory.mmu, image, argv, envp, identity)?;

        let mut vm = Vm::at_address(ctx, loaded.entry, source, memory)
            .map_err(|e| format!("cannot lift the entry point {:#x}: {e:?}", loaded.entry))?;
        regs.reset(vm.memory_mut());
        regs.rsp.write(vm.memory_mut(), rsp);
        if jit {
            vm.set_block_executor(Box::new(qcode_jit::Jit::new()));
        }
        vm.add_hook(DivideHook);
        let arch_spaces = (0..vm.context().space_count())
            .map(|index| MemorySpaceId::from(SpaceId::from(index)))
            .filter(|&space| vm.memory().is_flat(space))
            .collect();
        Ok((
            Self {
                vm,
                regs,
                arch_spaces,
                step_space: None,
            },
            loaded,
        ))
    }

    /// Installs the single-step hook if it is not already, returning the
    /// flag's space.
    fn step_space(&mut self) -> SpaceId {
        if let Some(space) = self.step_space {
            return space;
        }
        let space = self
            .vm
            .state_space(traps::STEP_SPACE, 8)
            .expect("the step flag's space is one word");
        self.vm.add_hook(StepHook);
        self.step_space = Some(space);
        space
    }

    /// Sets the single-step flag, if the hook that reads it is installed.
    fn set_stepping(&mut self, on: bool) {
        if let Some(space) = self.step_space {
            self.vm
                .memory_mut()
                .flat_mut()
                .entry(MemorySpaceId::Shared(space))
                .write_u128(0, 8, u128::from(on))
                .expect("the step flag is one word");
        }
    }

    /// The task's own flat spaces, as they are on the machine.
    fn save_spaces(&self) -> Vec<(MemorySpaceId, FlatSpace)> {
        self.arch_spaces
            .iter()
            .filter_map(|&space| Some((space, self.vm.memory().flat().get(space)?.clone())))
            .collect()
    }
}

/// How a `fork`, `vfork` or `clone` makes its child.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct CloneArgs {
    /// The parent sleeps until the child execs or exits (`CLONE_VFORK`).
    pub vfork: bool,
    /// The child runs in the parent's memory (`CLONE_VM`, with
    /// `CLONE_VFORK` only): whatever it does to memory before it execs or
    /// exits, the parent finds when it wakes.
    pub share_vm: bool,
    /// The child's stack pointer, when not the parent's.
    pub stack: u64,
    /// Where the parent gets the child's tid (`CLONE_PARENT_SETTID`).
    pub parent_tid: u64,
    /// Where the child gets its own tid (`CLONE_CHILD_SETTID`).
    pub child_tid: u64,
    /// Where the child's tid is cleared when it exits or execs
    /// (`CLONE_CHILD_CLEARTID`).
    pub clear_tid: u64,
}

/// A step the scheduler asked of a traced task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Step {
    /// About to run the instruction it is stepped over.
    Armed,
    /// Ran it; stops before the next.
    Running,
}

/// What a `CLONE_VFORK` child hands back to the parent it held asleep.
struct Release {
    parent: u64,
    /// With `CLONE_VM`, the memory the two shared as the child left it.
    memory: Option<Shared>,
}

struct Shared {
    memory: MmuSnapshot,
    space: AddressSpace,
    shared_anon: Vec<(u64, u64)>,
    shared: Vec<SharedMap>,
}

/// What a system call asked of the scheduler, beyond a value in `RAX`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Request {
    /// Create a child; the caller has not been resumed.
    Fork(CloneArgs),
    /// The call cannot complete yet; the caller has not been resumed and
    /// re-issues it when it next runs.
    Block,
    /// End another task as if by a fatal signal.
    Kill { pid: u64, signal: i32 },
    /// A `ptrace` request on another task; the caller has not been resumed
    /// and gets the request's result in `RAX` once the scheduler has done it.
    Ptrace {
        request: u64,
        pid: u64,
        addr: u64,
        data: u64,
    },
}

/// Why a task stopped running for now.
enum TaskExit {
    Exited(i32),
    Crashed(Crash),
    Budget,
    Requested(Request),
    /// A traced task took a trap or a fault: it stops with `signal` for its
    /// tracer to inspect, positioned at `rip` as the tracer will see it.
    Stopped {
        signal: i32,
        rip: u64,
    },
}

/// `ptrace` requests (`asm/ptrace-abi.h`).
pub(crate) const PTRACE_TRACEME: u64 = 0;
const PTRACE_PEEKTEXT: u64 = 1;
const PTRACE_PEEKDATA: u64 = 2;
const PTRACE_POKETEXT: u64 = 4;
const PTRACE_POKEDATA: u64 = 5;
const PTRACE_CONT: u64 = 7;
const PTRACE_SINGLESTEP: u64 = 9;
const PTRACE_GETREGS: u64 = 12;
const PTRACE_SETREGS: u64 = 13;
const PTRACE_SEIZE: u64 = 0x4206;

/// `user_regs_struct`: 27 words, in the kernel's order.
const USER_REGS: usize = 27;
const REG_RIP: usize = 16;
const REG_CS: usize = 17;
const REG_EFLAGS: usize = 18;
const REG_SS: usize = 20;

/// One thread of execution with its own address space and descriptor table.
pub(crate) struct Task {
    /// The process's machine, while this task is the one running on it.
    pub(crate) machine: Option<Machine>,
    /// What the task left the machine with when it was parked; `None`
    /// while it holds the machine.
    saved: Option<Saved>,
    /// Where the registers live, for reading a parked task's.
    regs: Regs,
    pub(crate) space: AddressSpace,
    pub(crate) image: LoadedImage,
    pub(crate) files: Files,
    pub(crate) pid: u64,
    pub(crate) ppid: u64,
    /// Children not yet reaped, live or exited.
    pub(crate) children: Vec<u64>,
    /// Exited children awaiting `wait4`, with their wait status.
    pub(crate) zombies: Vec<(u64, i32)>,
    pub(crate) trace: bool,
    pub(crate) exit_code: Option<i32>,
    pub(crate) request: Option<Request>,
    pub(crate) started: Instant,
    pub(crate) rng: u64,
    pub(crate) signal_mask: u64,
    pub(crate) sigactions: std::collections::HashMap<u64, [u8; 32]>,
    /// `MAP_SHARED` mappings of files, carried back to them when another
    /// mapping could observe their bytes.
    pub(crate) shared: Vec<SharedMap>,
    pub(crate) identity: Identity,
    /// The task tracing this one, once it asked to be traced or was seized.
    pub(crate) traced_by: Option<u64>,
    /// The signal this traced task stopped with, until its tracer continues
    /// it. A stopped task is not scheduled and is not reaped.
    pub(crate) stopped: Option<i32>,
    /// Stops of the tasks this one traces, not yet collected by a wait.
    pub(crate) stops: Vec<(u64, i32)>,
    /// `MAP_SHARED | MAP_ANONYMOUS` ranges, which every task of the process
    /// sees the same: their bytes travel through the scheduler's arena at
    /// each switch. Inherited by a fork, dropped by `munmap`.
    pub(crate) shared_anon: Vec<(u64, u64)>,
    /// Tasks this one seized that are not its children, which its waits
    /// report on too.
    pub(crate) tracees: Vec<u64>,
    /// Why the task last stopped for its tracer, for a signal the tracer
    /// continues it with.
    stop_info: Option<SigInfo>,
    /// A signal to enter the handler for before the task next runs.
    pending_signal: Option<SigInfo>,
    /// A single step its tracer asked for.
    stepping: Option<Step>,
    /// The alternate signal stack: base and size.
    pub(crate) altstack: Option<(u64, u64)>,
    /// Cleared, as a 32-bit tid, when the task exits or execs
    /// (`CLONE_CHILD_CLEARTID`, `set_tid_address`).
    pub(crate) clear_child_tid: u64,
    /// The parent a `CLONE_VFORK` holds asleep until this task execs or
    /// exits.
    vfork_parent: Option<u64>,
    /// Whether this task runs in its vfork parent's memory.
    share_vm: bool,
    /// The `CLONE_VFORK` child this task sleeps on.
    vfork_child: Option<u64>,
    /// Set by an exec that woke the vfork parent; the scheduler hands it
    /// over.
    released: Option<Release>,
    /// Operations retired before the task was last put on the machine.
    retired: u64,
    /// The machine's count when the task was last put on it.
    steps_in: u64,
}

impl Task {
    fn new(image: &[u8], config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        let identity = Identity {
            uid: 1000,
            gid: 1000,
        };
        let (machine, image) =
            Machine::boot(image, &config.argv, &config.envp, identity, config.jit)?;
        let files = Files::new(config.stdio, config.root, config.exe_path);
        Ok(Self {
            regs: machine.regs,
            machine: Some(machine),
            saved: None,
            space: AddressSpace {
                brk_start: image.brk,
                brk: image.brk,
                mmap_next: MMAP_BASE,
            },
            image,
            files,
            pid: ROOT_PID,
            ppid: 1,
            children: Vec::new(),
            zombies: Vec::new(),
            trace: config.trace,
            exit_code: None,
            request: None,
            started: Instant::now(),
            rng: 0x2545_f491_4f6c_dd1d,
            signal_mask: 0,
            sigactions: Default::default(),
            shared: Vec::new(),
            identity,
            traced_by: None,
            stopped: None,
            stops: Vec::new(),
            shared_anon: Vec::new(),
            tracees: Vec::new(),
            stop_info: None,
            pending_signal: None,
            stepping: None,
            altstack: None,
            clear_child_tid: 0,
            vfork_parent: None,
            share_vm: false,
            vfork_child: None,
            released: None,
            retired: 0,
            steps_in: 0,
        })
    }

    /// The machine, which this task holds: every system call and every run
    /// happens on the task that is on it.
    pub(crate) fn machine(&mut self) -> &mut Machine {
        self.machine.as_mut().expect("the task holds the machine")
    }

    pub(crate) fn vm(&mut self) -> &mut Vm<SleighCodeSource<'static>> {
        &mut self.machine().vm
    }

    pub(crate) fn mmu(&mut self) -> &mut Mmu {
        &mut self.machine().vm.memory_mut().mmu
    }

    /// Whether the scheduler may run this task: live, not stopped by a
    /// tracer and not asleep on a vfork child.
    fn runnable(&self) -> bool {
        self.exit_code.is_none() && self.stopped.is_none() && self.vfork_child.is_none()
    }

    /// Operations this task has retired, on the machine and before.
    fn steps(&self) -> u64 {
        self.retired
            + self
                .machine
                .as_ref()
                .map_or(0, |m| m.vm.stats.steps - self.steps_in)
    }

    /// Takes the task off the machine, keeping its position, registers and
    /// address space, and returns the machine for the next task.
    fn park(&mut self) -> Machine {
        let mut machine = self.machine.take().expect("the task holds the machine");
        self.retired += machine.vm.stats.steps - self.steps_in;
        let parked = machine.vm.park();
        let spaces = machine.save_spaces();
        let memory = machine.vm.memory_mut().mmu.snapshot();
        self.saved = Some(Saved {
            parked,
            memory,
            spaces,
        });
        machine
    }

    /// Puts the task on the machine: its address space and registers first,
    /// then its position. The machine is the task's from here on, even if
    /// the position could not be found again — that is a crash of the task,
    /// reported to the caller.
    fn install(&mut self, mut machine: Machine) -> Result<(), Crash> {
        let saved = self
            .saved
            .take()
            .expect("a task off the machine was parked");
        machine.vm.memory_mut().mmu.restore(&saved.memory);
        for (space, contents) in saved.spaces {
            *machine.vm.memory_mut().flat_mut().entry(space) = contents;
        }
        self.steps_in = machine.vm.stats.steps;
        machine.set_stepping(self.stepping.is_some());
        let result = machine.vm.unpark(saved.parked);
        self.machine = Some(machine);
        result.map_err(|error| self.crash(format!("cannot put the task back: {error}"), None))
    }

    /// The child of this task, which is stopped at a `fork`: the same memory
    /// and descriptors, `RAX` zero, positioned after the `syscall`
    /// instruction. The parent is left stopped for the caller to resume.
    ///
    /// The child is off the machine: its address space is a copy-on-write
    /// snapshot of the parent's, its registers a copy. Lifted code is the
    /// process's and needs no copying.
    ///
    /// `args` may give the child its own stack, and places to store its tid
    /// in; with `CLONE_VM` the snapshot is the memory the parent gets back
    /// when the child releases it.
    fn fork(&mut self, pid: u64, args: CloneArgs) -> Result<Task, Crash> {
        let Some(pc) = self.pc() else {
            return Err(self.crash("fork from an unknown position".to_owned(), None));
        };
        // `syscall` is two bytes and lifts to the one op the machine sits at.
        let next = pc + 2;
        let machine = self.machine();
        let m = machine.vm.memory_mut();
        let rax = machine.regs.rax.read(m);
        let rsp = machine.regs.rsp.read(m);
        machine.regs.rax.write(m, 0);
        if args.stack != 0 {
            machine.regs.rsp.write(m, args.stack);
        }
        let spaces = machine.save_spaces();
        let m = machine.vm.memory_mut();
        machine.regs.rax.write(m, rax);
        machine.regs.rsp.write(m, rsp);
        let mut memory = m.mmu.snapshot();
        if args.child_tid != 0 {
            let mut scratch = Mmu::new();
            scratch.restore(&memory);
            // A tid the child cannot be told is its own loss, as on Linux.
            let _ = scratch.write(args.child_tid, &(pid as u32).to_le_bytes());
            memory = scratch.snapshot();
        }
        if args.parent_tid != 0 {
            let _ = guest::write_u32(self.mmu(), args.parent_tid, pid as u32);
        }
        let files = match self.files.fork() {
            Ok(files) => files,
            Err(e) => return Err(self.crash(format!("cannot duplicate descriptors: {e}"), None)),
        };
        Ok(Task {
            regs: self.regs,
            machine: None,
            saved: Some(Saved {
                parked: Parked::at_address(next),
                memory,
                spaces,
            }),
            space: self.space,
            image: self.image.clone(),
            files,
            pid,
            ppid: self.pid,
            children: Vec::new(),
            zombies: Vec::new(),
            trace: self.trace,
            exit_code: None,
            request: None,
            started: self.started,
            rng: self.rng,
            signal_mask: self.signal_mask,
            sigactions: self.sigactions.clone(),
            shared: self
                .shared
                .iter()
                .filter_map(SharedMap::duplicate)
                .collect(),
            identity: self.identity,
            traced_by: None,
            stopped: None,
            stops: Vec::new(),
            shared_anon: self.shared_anon.clone(),
            tracees: Vec::new(),
            stop_info: None,
            pending_signal: None,
            stepping: None,
            altstack: self.altstack,
            clear_child_tid: args.clear_tid,
            vfork_parent: args.vfork.then_some(self.pid),
            share_vm: args.vfork && args.share_vm,
            vfork_child: None,
            released: None,
            retired: 0,
            steps_in: 0,
        })
    }

    /// Writes `bytes` into the task's memory, on the machine or parked.
    fn write_memory(&mut self, addr: u64, bytes: &[u8]) -> bool {
        if self.machine.is_some() {
            return self.mmu().write(addr, bytes).is_ok();
        }
        let Some(saved) = self.saved.as_mut() else {
            return false;
        };
        let mut scratch = Mmu::new();
        scratch.restore(&saved.memory);
        if scratch.write(addr, bytes).is_err() {
            return false;
        }
        saved.memory = scratch.snapshot();
        true
    }

    /// Lets go of the vfork parent, if the task holds one asleep: an exec
    /// or an exit. The tid `CLONE_CHILD_CLEARTID` named is cleared first,
    /// in the memory the parent may get back.
    fn release(&mut self) -> Option<Release> {
        if self.clear_child_tid != 0 {
            let at = std::mem::take(&mut self.clear_child_tid);
            self.write_memory(at, &0u32.to_le_bytes());
        }
        let parent = self.vfork_parent.take()?;
        let memory = self.share_vm.then(|| {
            let memory = match self.machine.as_mut() {
                Some(machine) => machine.vm.memory_mut().mmu.snapshot(),
                None => self
                    .saved
                    .as_ref()
                    .expect("a task off the machine was parked")
                    .memory
                    .clone(),
            };
            Shared {
                memory,
                space: self.space,
                shared_anon: self.shared_anon.clone(),
                shared: std::mem::take(&mut self.shared),
            }
        });
        self.share_vm = false;
        Some(Release { parent, memory })
    }

    /// Takes back the memory a vfork child ran in, as it left it. The task
    /// is parked, asleep on the child.
    fn adopt(&mut self, shared: Shared) {
        if let Some(saved) = self.saved.as_mut() {
            saved.memory = shared.memory;
        }
        self.space = shared.space;
        self.shared_anon = shared.shared_anon;
        self.shared = shared.shared;
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
        let mut fresh = Mmu::new();
        let (loaded, rsp) =
            load_into(&mut fresh, image, argv, envp, self.identity).map_err(|_| ENOEXEC)?;
        self.released = self.release();
        let machine = self.machine();
        machine.vm.memory_mut().mmu.adopt(fresh);
        machine.regs.reset(machine.vm.memory_mut());
        machine.regs.rsp.write(machine.vm.memory_mut(), rsp);
        // Past the point of no return: the old image is gone whatever the
        // new one's entry does.
        machine.vm.position_at(loaded.entry).map_err(|_| ENOEXEC)?;
        self.space = AddressSpace {
            brk_start: loaded.brk,
            brk: loaded.brk,
            mmap_next: MMAP_BASE,
        };
        self.image = loaded;
        self.files.exe_path = host_path;
        self.files.close_on_exec();
        self.sigactions.clear();
        self.shared.clear();
        self.shared_anon.clear();
        self.altstack = None;
        Ok(())
    }

    /// Runs the task for up to `budget` p-code operations. A task cut off
    /// part-way through a guest instruction is parked and put back by op
    /// identity, the way a snapshot is, so the budget need not land on a
    /// boundary.
    fn run(&mut self, budget: u64) -> TaskExit {
        if let Some(code) = self.exit_code {
            return TaskExit::Exited(code);
        }
        if let Some(info) = self.pending_signal.take() {
            let rip = self.pc().unwrap_or(0);
            if let Err(crash) = self.deliver(info, rip) {
                return TaskExit::Crashed(crash);
            }
        }
        let deadline = self.vm().stats.steps.saturating_add(budget);
        loop {
            let remaining = deadline.saturating_sub(self.vm().stats.steps);
            if remaining == 0 {
                return TaskExit::Budget;
            }
            let exit = self.vm().run(remaining);
            if let Some(stop) = self.settle(exit) {
                return stop;
            }
        }
    }

    /// Acts on a machine exit; `None` when the task simply carries on.
    fn settle(&mut self, exit: VmExit) -> Option<TaskExit> {
        match exit {
            VmExit::InstructionLimit | VmExit::Breakpoint(_) | VmExit::HookStop(_) => None,
            VmExit::Interrupt(interrupt) if interrupt.kind == DIVIDE_ERROR_KIND => {
                let rip = interrupt.pc.or_else(|| self.pc()).unwrap_or(0);
                let info = SigInfo::raised(SIGFPE, signal::FPE_INTDIV, rip, 0);
                self.trap(info, rip, "integer divide by zero".to_owned())
            }
            VmExit::Interrupt(interrupt) if interrupt.kind == SINGLE_STEP_KIND => {
                self.step_check(&interrupt)
            }
            VmExit::Interrupt(interrupt)
                if matches!(&interrupt.kind, InterruptKind::Intrinsic { name, .. }
                    if name.as_ref() == "invalidInstructionException") =>
            {
                let rip = interrupt.pc.or_else(|| self.pc()).unwrap_or(0);
                let info = SigInfo::raised(SIGILL, signal::ILL_ILLOPN, rip, 6);
                self.trap(info, rip, "illegal instruction".to_owned())
            }
            VmExit::Interrupt(interrupt) => match self.handle_interrupt(&interrupt) {
                Ok(()) => {
                    if let Some(code) = self.exit_code {
                        return Some(TaskExit::Exited(code));
                    }
                    self.request.take().map(TaskExit::Requested)
                }
                Err(crash) => Some(TaskExit::Crashed(crash)),
            },
            VmExit::Error(message) => {
                Some(TaskExit::Crashed(self.crash(message.to_string(), None)))
            }
            VmExit::Fault(fault) => {
                let Some(pc) = self.pc() else {
                    return Some(TaskExit::Crashed(self.crash(fault.to_string(), Some(11))));
                };
                // `int3` lifts to a read of the interrupt descriptor table,
                // which is unmapped, so it arrives here as a fault too: the
                // byte at the position tells the two apart. Linux reports a
                // breakpoint one byte past it, and a faulting access at the
                // instruction itself.
                if guest::read(self.mmu(), pc, 1).is_ok_and(|b| b == [0xcc]) {
                    let info = SigInfo::raised(SIGTRAP, SI_KERNEL, 0, 3);
                    return self.trap(info, pc + 1, "breakpoint".to_owned());
                }
                let unmapped = matches!(
                    fault.kind,
                    FaultKind::ReadUnmapped | FaultKind::WriteUnmapped | FaultKind::ExecUnmapped
                );
                let mut info = SigInfo::raised(
                    SIGSEGV,
                    if unmapped { SEGV_MAPERR } else { SEGV_ACCERR },
                    fault.addr,
                    14,
                );
                // The page-fault error code: user mode, a write, a page
                // that was present.
                info.err = 4 | u64::from(fault.is_write()) << 1 | u64::from(!unmapped);
                self.trap(info, pc, fault.to_string())
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
                Some(TaskExit::Crashed(self.crash(reason, signal)))
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
                if let Some(Request::Fork(_) | Request::Block | Request::Ptrace { .. }) =
                    self.request
                {
                    return Ok(());
                }
                if self.vm().pending_interrupt().is_none() {
                    // `execve` replaced the image; there is nothing to resume.
                    return Ok(());
                }
                let machine = self.machine();
                machine.regs.rax.write(machine.vm.memory_mut(), result);
                None
            }
            // Monotonic, and advancing with work done rather than wall time
            // so a run is reproducible.
            "rdtsc" => Some(u128::from(self.steps().wrapping_mul(8))),
            name if name.starts_with("cpuid") => {
                let machine = self.machine();
                let leaf = machine.regs.rax.read(machine.vm.memory_mut()) as u32;
                let sub = machine.regs.rcx.read(machine.vm.memory_mut()) as u32;
                let [eax, ebx, ecx, edx] = cpuid(leaf, sub);
                Some(
                    u128::from(eax)
                        | u128::from(ebx) << 32
                        | u128::from(edx) << 64
                        | u128::from(ecx) << 96,
                )
            }
            other => {
                return Err(self.crash(format!("unsupported p-code operation `{other}`"), None));
            }
        };
        self.vm()
            .resume(value)
            .map_err(|error| self.crash(error.to_string(), None))
    }

    /// A signal the task's own instruction raised, positioned to resume at
    /// `rip`: a stop for its tracer if it has one, its handler if it
    /// installed one, and otherwise the end of it.
    fn trap(&mut self, info: SigInfo, rip: u64, reason: String) -> Option<TaskExit> {
        if self.traced_by.is_some() {
            self.stop_info = Some(info);
            return Some(TaskExit::Stopped {
                signal: info.signo,
                rip,
            });
        }
        if self.handler(info.signo).is_none() {
            return Some(TaskExit::Crashed(self.crash(reason, Some(info.signo))));
        }
        self.deliver(info, rip).err().map(TaskExit::Crashed)
    }

    /// The step flag's check before an instruction: the first after a
    /// `PTRACE_SINGLESTEP` is the instruction stepped over, the second the
    /// stop after it.
    fn step_check(&mut self, interrupt: &Interrupt) -> Option<TaskExit> {
        match self.stepping {
            Some(Step::Running) => {
                self.stepping = None;
                self.machine().set_stepping(false);
                let rip = interrupt.pc.or_else(|| self.pc()).unwrap_or(0);
                let info = SigInfo::raised(SIGTRAP, signal::TRAP_TRACE, rip, 1);
                self.trap(info, rip, "single step".to_owned())
            }
            step => {
                if step == Some(Step::Armed) {
                    self.stepping = Some(Step::Running);
                } else {
                    self.machine().set_stepping(false);
                }
                self.vm()
                    .resume(None)
                    .err()
                    .map(|error| TaskExit::Crashed(self.crash(error.to_string(), None)))
            }
        }
    }

    /// The parent's side of a fork, once the child exists: `RAX` is the
    /// child's pid and the machine moves on.
    fn resume_after_fork(&mut self, child: u64) -> Result<(), Crash> {
        self.resume_with(child)
    }

    /// Completes a system call the scheduler answered: `RAX` gets `value`
    /// and the machine moves on.
    fn resume_with(&mut self, value: u64) -> Result<(), Crash> {
        let machine = self.machine();
        machine.regs.rax.write(machine.vm.memory_mut(), value);
        machine
            .vm
            .resume(None)
            .map_err(|error| self.crash(error.to_string(), None))
    }

    /// Leaves a parked task positioned at `rip`, as a tracer sees it.
    fn place_at(&mut self, rip: u64) {
        if let Some(saved) = self.saved.as_mut() {
            saved.parked = Parked::at_address(rip);
        }
    }

    /// The saved register file of a parked task, in `user_regs_struct`
    /// order. Fields with no register behind them read as zero.
    fn user_regs(&self) -> Option<[u64; USER_REGS]> {
        let saved = self.saved.as_ref()?;
        let read = |reg: crate::regs::Reg| {
            saved
                .spaces
                .iter()
                .find(|(space, _)| *space == reg.space)
                .map_or(0, |(_, contents)| reg.read_saved(contents))
        };
        let r = &self.regs;
        let mut out = [0u64; USER_REGS];
        let named = [
            r.r15, r.r14, r.r13, r.r12, r.rbp, r.rbx, r.r11, r.r10, r.r9, r.r8, r.rax, r.rcx,
            r.rdx, r.rsi, r.rdi,
        ];
        for (i, reg) in named.into_iter().enumerate() {
            out[i] = read(reg);
        }
        out[REG_RIP] = saved.parked.address().unwrap_or(0);
        out[REG_CS] = 0x33;
        out[REG_EFLAGS] = r.eflags(read);
        out[19] = read(r.rsp);
        out[REG_SS] = 0x2b;
        out[21] = read(r.fs_base);
        out[22] = read(r.gs_base);
        Some(out)
    }

    /// Writes a `user_regs_struct` into a parked task's saved register file;
    /// `rip` repositions it.
    fn set_user_regs(&mut self, regs: &[u64; USER_REGS]) -> bool {
        let r = self.regs;
        let Some(saved) = self.saved.as_mut() else {
            return false;
        };
        let mut write = |reg: crate::regs::Reg, value: u64| {
            if let Some((_, contents)) = saved
                .spaces
                .iter_mut()
                .find(|(space, _)| *space == reg.space)
            {
                reg.write_saved(contents, value);
            }
        };
        let named = [
            r.r15, r.r14, r.r13, r.r12, r.rbp, r.rbx, r.r11, r.r10, r.r9, r.r8, r.rax, r.rcx,
            r.rdx, r.rsi, r.rdi,
        ];
        for (i, reg) in named.into_iter().enumerate() {
            write(reg, regs[i]);
        }
        write(r.rsp, regs[19]);
        write(r.fs_base, regs[21]);
        write(r.gs_base, regs[22]);
        r.set_eflags(regs[REG_EFLAGS], &mut write);
        saved.parked = Parked::at_address(regs[REG_RIP]);
        true
    }

    /// A word of a parked task's memory.
    fn peek(&self, addr: u64) -> Option<u64> {
        let saved = self.saved.as_ref()?;
        let mut scratch = Mmu::new();
        scratch.restore(&saved.memory);
        let mut word = [0u8; 8];
        scratch.read(addr, &mut word).ok()?;
        Some(u64::from_le_bytes(word))
    }

    /// Writes a word into a parked task's memory.
    fn poke(&mut self, addr: u64, value: u64) -> bool {
        let Some(saved) = self.saved.as_mut() else {
            return false;
        };
        let mut scratch = Mmu::new();
        scratch.restore(&saved.memory);
        if scratch.write(addr, &value.to_le_bytes()).is_err() {
            return false;
        }
        saved.memory = scratch.snapshot();
        true
    }

    /// The guest address of the instruction the task is positioned at, on
    /// the machine or parked.
    pub(crate) fn pc(&mut self) -> Option<u64> {
        let Some(machine) = self.machine.as_mut() else {
            return self.saved.as_ref().and_then(|saved| saved.parked.address());
        };
        let vm = &mut machine.vm;
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
        match (self.machine.as_mut(), self.saved.as_ref()) {
            (Some(machine), _) => machine
                .regs
                .named()
                .into_iter()
                .map(|(name, reg)| (name, reg.read(machine.vm.memory_mut())))
                .collect(),
            (None, Some(saved)) => {
                let read = |reg: crate::regs::Reg| {
                    saved
                        .spaces
                        .iter()
                        .find(|(space, _)| *space == reg.space)
                        .map_or(0, |(_, contents)| reg.read_saved(contents))
                };
                self.regs
                    .named()
                    .into_iter()
                    .map(|(name, reg)| (name, read(reg)))
                    .collect()
            }
            (None, None) => Vec::new(),
        }
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

/// A loaded program and everything it forks, run on one host thread.
pub struct Process {
    /// Slot 0 is the initial task and is never dropped; a child's slot is
    /// emptied once it has exited and been reaped or orphaned.
    tasks: Vec<Option<Task>>,
    current: usize,
    next_pid: u64,
    /// The machine, between the task that had it leaving and the next
    /// taking it; otherwise held by exactly one task.
    machine: Option<Machine>,
    /// The latest bytes of every shared anonymous mapping, by start address:
    /// written by a task leaving the machine, read by the task taking it.
    arena: HashMap<u64, Vec<u8>>,
    /// A one-word flat cell an instrumenter can point the scheduler at, kept
    /// holding the pid of the task currently on the machine: written on every
    /// switch and whenever a fork or exec changes who runs. Compiled hooks
    /// read it to stamp their records with the running task, so a record made
    /// either side of a cooperative switch is not mistaken for one task's.
    task_id_space: Option<SpaceId>,
    /// A tracer redirecting a tracee with `PTRACE_SETREGS`, recorded for a
    /// consumer that wants the one cross-task control-flow transfer the
    /// scheduler mediates: `(tracer pid, tracee pid, the rip it was set to,
    /// the tracer's own pc when it asked)`. A cooperative switch is *not* a
    /// control-flow edge and is deliberately not recorded here.
    ptrace_redirects: Vec<PtraceRedirect>,
}

/// A tracer pointing a tracee somewhere new through `ptrace`: the one
/// cross-task control-flow transfer a [`Process`] mediates, as opposed to a
/// bare cooperative switch, which transfers nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtraceRedirect {
    pub tracer: u64,
    pub tracee: u64,
    /// The instruction pointer the tracer set the tracee to.
    pub rip: u64,
    /// The tracer's own position when it issued the request, if known.
    pub tracer_pc: Option<u64>,
}

/// The exits of the two trap hooks.
const DIVIDE_ERROR_KIND: InterruptKind = InterruptKind::Explicit {
    code: traps::DIVIDE_ERROR,
};
const SINGLE_STEP_KIND: InterruptKind = InterruptKind::Explicit {
    code: traps::SINGLE_STEP,
};

/// Operations a task runs before the next runnable one gets a turn.
const SLICE: u64 = 1 << 20;

impl Process {
    /// Loads `image` and prepares the machine at its entry point.
    pub fn new(image: &[u8], config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            tasks: vec![Some(Task::new(image, config)?)],
            current: 0,
            next_pid: ROOT_PID + 1,
            machine: None,
            arena: HashMap::new(),
            task_id_space: None,
            ptrace_redirects: Vec::new(),
        })
    }

    /// Points the scheduler at a one-word flat cell to keep the running
    /// task's pid in, for compiled hooks that stamp their records with the
    /// task that made them. The cell is written now with the current task's
    /// pid, then on every switch and on fork/exec. The space must be a
    /// bounded flat space of at least four bytes (see
    /// [`Vm::state_space`](qcode_vm::Vm::state_space)).
    pub fn set_task_id_space(&mut self, space: SpaceId) {
        self.task_id_space = Some(space);
        self.write_task_id(self.current);
    }

    /// The cross-task control-flow transfers `ptrace` mediated, in order.
    pub fn ptrace_redirects(&self) -> &[PtraceRedirect] {
        &self.ptrace_redirects
    }

    /// Writes the pid of the task in `slot` into the task-id cell, if one was
    /// registered and the task holds the machine.
    fn write_task_id(&mut self, slot: usize) {
        let Some(space) = self.task_id_space else {
            return;
        };
        let Some(pid) = self.tasks.get(slot).and_then(|task| {
            let task = task.as_ref()?;
            task.machine.is_some().then_some(task.pid)
        }) else {
            return;
        };
        let _ = self
            .machine_mut()
            .vm
            .memory_mut()
            .flat_mut()
            .entry(MemorySpaceId::Shared(space))
            .write_u128(0, 4, u128::from(pid as u32));
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

    /// The slot of the task holding the machine, if one does.
    fn holder(&self) -> Option<usize> {
        self.tasks
            .iter()
            .position(|task| task.as_ref().is_some_and(|task| task.machine.is_some()))
    }

    fn machine_ref(&self) -> &Machine {
        match self.holder() {
            Some(slot) => self.tasks[slot].as_ref().expect("live").machine.as_ref(),
            None => self.machine.as_ref(),
        }
        .expect("the process has one machine")
    }

    fn machine_mut(&mut self) -> &mut Machine {
        match self.holder() {
            Some(slot) => self.tasks[slot].as_mut().expect("live").machine.as_mut(),
            None => self.machine.as_mut(),
        }
        .expect("the process has one machine")
    }

    /// Puts the task in `slot` on the machine, parking whichever task had
    /// it. An error is a crash of the incoming task, which holds the
    /// machine either way.
    fn switch_to(&mut self, slot: usize) -> Result<(), Crash> {
        if self.tasks[slot]
            .as_ref()
            .is_some_and(|task| task.machine.is_some())
        {
            return Ok(());
        }
        let machine = match self.holder() {
            Some(from) => {
                self.stash_shared(from);
                self.tasks[from].as_mut().expect("live").park()
            }
            None => self.machine.take().expect("the process has one machine"),
        };
        let result = self.tasks[slot].as_mut().expect("live").install(machine);
        self.load_shared(slot);
        result
    }

    /// Copies the shared anonymous mappings of the task in `slot`, which
    /// holds the machine, into the arena.
    fn stash_shared(&mut self, slot: usize) {
        let task = self.tasks[slot].as_mut().expect("live");
        if task.machine.is_none() {
            return;
        }
        for (start, len) in task.shared_anon.clone() {
            let mut bytes = vec![0u8; len as usize];
            if task.mmu().read(start, &mut bytes).is_ok() {
                self.arena.insert(start, bytes);
            }
        }
    }

    /// Writes the arena's bytes into the shared anonymous mappings of the
    /// task in `slot`, which has just taken the machine.
    fn load_shared(&mut self, slot: usize) {
        let task = self.tasks[slot].as_mut().expect("live");
        if task.machine.is_none() {
            return;
        }
        for (start, _) in task.shared_anon.clone() {
            if let Some(bytes) = self.arena.get(&start) {
                let _ = task.mmu().write(start, bytes);
            }
        }
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
        &self.root().image
    }

    /// The process's machine, which every task runs on: hooks and state
    /// spaces installed here apply to all of them.
    pub fn vm(&mut self) -> &mut Vm<SleighCodeSource<'static>> {
        &mut self.machine_mut().vm
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
        self.machine_ref().vm.stats.steps
    }

    /// The first runnable task at or after `from`, wrapping around; none
    /// when every task is stopped or asleep.
    fn next_live(&self, from: usize) -> Option<usize> {
        let n = self.tasks.len();
        (0..n)
            .map(|k| (from + k) % n)
            .find(|&i| self.tasks[i].as_ref().is_some_and(Task::runnable))
    }

    /// Wakes the vfork parent the task in `slot` let go of, handing it the
    /// memory they shared.
    fn wake_vfork_parent(&mut self, release: Release) {
        if let Some(parent) = self
            .tasks
            .iter_mut()
            .flatten()
            .find(|t| t.pid == release.parent && t.exit_code.is_none())
        {
            parent.vfork_child = None;
            if let Some(shared) = release.memory {
                parent.adopt(shared);
            }
        }
    }

    fn live_count(&self) -> usize {
        self.tasks.iter().flatten().filter(|t| t.runnable()).count()
    }

    /// Answers a `ptrace` request from the task in `tracer`, returning what
    /// its `RAX` gets.
    fn ptrace(
        &mut self,
        tracer: usize,
        request: u64,
        pid: u64,
        addr: u64,
        data: u64,
    ) -> (u64, Option<usize>) {
        let tracer_pid = self.tasks[tracer].as_ref().expect("live").pid;
        let target = self.tasks.iter().position(|t| {
            t.as_ref()
                .is_some_and(|t| t.pid == pid && t.exit_code.is_none())
        });
        // A stopped tracee of the caller, which is what every request but a
        // seize needs.
        let stopped = target.filter(|&i| {
            let t = self.tasks[i].as_ref().expect("live");
            t.traced_by == Some(tracer_pid) && t.stopped.is_some()
        });
        // A seized tracee gets the machine next: it runs until it stops,
        // blocks or exits before its tracer goes on, which is the order a
        // tracer that seizes a child it just forked relies on.
        let mut yield_to = None;
        let result: Result<u64, Errno> = match request {
            // Any task of the process, an ancestor included: they all run
            // as one user. Not the caller, and not one traced already.
            PTRACE_SEIZE => match target {
                Some(i) if i == tracer => Err(errno::EPERM),
                Some(i) => {
                    let task = self.tasks[i].as_mut().expect("live");
                    if task.traced_by.is_some() {
                        Err(errno::EPERM)
                    } else {
                        task.traced_by = Some(tracer_pid);
                        let is_child = task.ppid == tracer_pid;
                        let tracer = self.tasks[tracer].as_mut().expect("live");
                        if !is_child {
                            tracer.tracees.push(pid);
                        }
                        yield_to = Some(i);
                        Ok(0)
                    }
                }
                None => Err(errno::ESRCH),
            },
            PTRACE_GETREGS => stopped.ok_or(errno::ESRCH).and_then(|i| {
                let regs = self.tasks[i]
                    .as_ref()
                    .expect("live")
                    .user_regs()
                    .ok_or(errno::ESRCH)?;
                let mut bytes = Vec::with_capacity(USER_REGS * 8);
                for word in regs {
                    bytes.extend_from_slice(&word.to_le_bytes());
                }
                let tracer = self.tasks[tracer].as_mut().expect("live");
                guest::write(tracer.mmu(), data, &bytes)?;
                Ok(0)
            }),
            PTRACE_SETREGS => stopped.ok_or(errno::ESRCH).and_then(|i| {
                let tracer_slot = tracer;
                let tracer = self.tasks[tracer].as_mut().expect("live");
                let bytes = guest::read(tracer.mmu(), data, USER_REGS * 8)?;
                let mut regs = [0u64; USER_REGS];
                for (word, chunk) in regs.iter_mut().zip(bytes.chunks_exact(8)) {
                    *word = u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
                }
                if self.tasks[i].as_mut().expect("live").set_user_regs(&regs) {
                    // The one cross-task control-flow transfer the scheduler
                    // mediates: the tracer just pointed the tracee's rip
                    // somewhere. A consumer that reconstructs control flow
                    // wants this edge and not the bare switch that carries it.
                    let tracer_pc = self.tasks[tracer_slot].as_mut().expect("live").pc();
                    self.ptrace_redirects.push(PtraceRedirect {
                        tracer: tracer_pid,
                        tracee: pid,
                        rip: regs[REG_RIP],
                        tracer_pc,
                    });
                    Ok(0)
                } else {
                    Err(errno::ESRCH)
                }
            }),
            PTRACE_CONT | PTRACE_SINGLESTEP => stopped.ok_or(errno::ESRCH).and_then(|i| {
                let signal = data as i32;
                if !(0..=64).contains(&signal) {
                    return Err(errno::EIO);
                }
                if request == PTRACE_SINGLESTEP {
                    self.machine_mut().step_space();
                }
                let task = self.tasks[i].as_mut().expect("live");
                task.stopped = None;
                let stop = task.stop_info.take();
                if request == PTRACE_SINGLESTEP {
                    task.stepping = Some(Step::Armed);
                }
                if signal != 0 {
                    if task.handler(signal).is_some() {
                        // The signal the task stopped with keeps what its
                        // trap recorded; another is the tracer's.
                        task.pending_signal = Some(
                            stop.filter(|info| info.signo == signal)
                                .unwrap_or_else(|| SigInfo::sent(signal, tracer_pid)),
                        );
                    } else {
                        // Fatal, as a kill would be. The initial task is
                        // never reaped; the run loop sees it has ended.
                        task.exit_code = Some(128 + signal);
                        if i != 0 {
                            self.reap(i, signal);
                        }
                    }
                }
                Ok(0)
            }),
            PTRACE_PEEKTEXT | PTRACE_PEEKDATA => stopped.ok_or(errno::ESRCH).and_then(|i| {
                let word = self.tasks[i]
                    .as_ref()
                    .expect("live")
                    .peek(addr)
                    .ok_or(errno::EIO)?;
                let tracer = self.tasks[tracer].as_mut().expect("live");
                guest::write_u64(tracer.mmu(), data, word)?;
                Ok(0)
            }),
            PTRACE_POKETEXT | PTRACE_POKEDATA => stopped.ok_or(errno::ESRCH).and_then(|i| {
                if self.tasks[i].as_mut().expect("live").poke(addr, data) {
                    Ok(0)
                } else {
                    Err(errno::EIO)
                }
            }),
            _ => Err(errno::EIO),
        };
        let value = match result {
            Ok(v) => v,
            Err(e) => e.as_ret(),
        };
        (value, yield_to)
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
            if let Some(code) = self.root().exit_code {
                return ProcessExit::Exited(code);
            }
            let remaining = deadline.saturating_sub(self.steps());
            if remaining == 0 {
                return ProcessExit::Budget;
            }
            let Some(index) = self.next_live(self.current) else {
                let root = self.root_mut();
                return ProcessExit::Crashed(
                    root.crash("every task is stopped or asleep".to_owned(), None),
                );
            };
            self.current = index;
            let switched = self.switch_to(index);
            if switched.is_ok() {
                self.write_task_id(index);
            }
            let task = self.tasks[index].as_mut().expect("live");
            let before = task.steps();
            let exit = match switched {
                Ok(()) => task.run(remaining.min(SLICE)),
                Err(crash) => TaskExit::Crashed(crash),
            };
            if !matches!(exit, TaskExit::Requested(Request::Block)) || task.steps() != before {
                blocked_streak = 0;
            }
            if let Some(release) = task.released.take() {
                self.wake_vfork_parent(release);
            }
            match exit {
                TaskExit::Budget => {
                    if self.steps() >= deadline {
                        return ProcessExit::Budget;
                    }
                    // Its slice: the next runnable task gets a turn.
                    self.current = index + 1;
                }
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
                    let pid = self.tasks[index].as_ref().expect("live").pid;
                    log::warn!("child {pid}: {crash}");
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
                TaskExit::Requested(Request::Fork(args)) => {
                    let pid = self.next_pid;
                    self.next_pid += 1;
                    let parent = self.tasks[index].as_mut().expect("live");
                    let child = match parent.fork(pid, args) {
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
                    if args.vfork {
                        parent.vfork_child = Some(pid);
                    }
                    let slot = (1..self.tasks.len())
                        .find(|&i| self.tasks[i].is_none())
                        .unwrap_or_else(|| {
                            self.tasks.push(None);
                            self.tasks.len() - 1
                        });
                    self.tasks[slot] = Some(child);
                    // A vfork parent sleeps until its child execs or exits.
                    // After a plain fork the parent keeps its turn, as it
                    // does on Linux: a tracer seizes the child it just
                    // forked before the child reaches its first trap.
                    self.current = if args.vfork { slot } else { index };
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
                TaskExit::Requested(Request::Ptrace {
                    request,
                    pid,
                    addr,
                    data,
                }) => {
                    let (value, yield_to) = self.ptrace(index, request, pid, addr, data);
                    let task = self.tasks[index].as_mut().expect("live");
                    if let Err(crash) = task.resume_with(value) {
                        if index == 0 {
                            return ProcessExit::Crashed(crash);
                        }
                        log::warn!("child {}: {crash}", task.pid);
                        task.exit_code = Some(128 + 9);
                        self.reap(index, 9);
                    }
                    self.current = yield_to
                        .filter(|&slot| self.tasks[slot].as_ref().is_some_and(Task::runnable))
                        .unwrap_or(index);
                }
                TaskExit::Stopped { signal, rip } => {
                    // Off the machine, positioned where its tracer will see
                    // it, and reported to that tracer's next wait.
                    self.stash_shared(index);
                    let task = self.tasks[index].as_mut().expect("live");
                    let machine = task.park();
                    self.machine = Some(machine);
                    task.place_at(rip);
                    task.stopped = Some(signal);
                    let (pid, tracer) = (task.pid, task.traced_by);
                    if let Some(tracer) = self
                        .tasks
                        .iter_mut()
                        .flatten()
                        .find(|t| Some(t.pid) == tracer && t.exit_code.is_none())
                    {
                        tracer.stops.push((pid, signal));
                    }
                    self.current = index + 1;
                }
            }
        }
    }

    /// Records the exit of the task in `slot` for its parent, hands its
    /// children to the initial task, and frees its address space; the
    /// machine, if it held it, waits for the next task.
    fn reap(&mut self, slot: usize, status: i32) {
        self.stash_shared(slot);
        let mut task = self.tasks[slot].take().expect("live");
        let release = task.release();
        if let Some(machine) = task.machine.take() {
            self.machine = Some(machine);
        }
        let (pid, ppid) = (task.pid, task.ppid);
        let orphans = task.children.clone();
        drop(task);
        if let Some(release) = release {
            self.wake_vfork_parent(release);
        }
        // Its tracer stops waiting on it; its tracees go free.
        for t in self.tasks.iter_mut().flatten() {
            t.tracees.retain(|&p| p != pid);
            if t.traced_by == Some(pid) {
                t.traced_by = None;
                t.stopped = None;
                t.stepping = None;
            }
            t.stops.retain(|&(p, _)| p != pid);
        }
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
