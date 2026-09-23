//! Signals the guest's own instructions raise — a fault, `ud2`, `int3`, a
//! divide by zero, a single step — and their delivery to a handler the task
//! installed with `rt_sigaction`.
//!
//! Delivery follows the x86-64 kernel: an `rt_sigframe` is pushed on the
//! task's stack, or on its alternate stack when the action asks for it and
//! one is set; the handler is entered with the signal number, the frame's
//! `siginfo` and `ucontext` in `RDI`, `RSI` and `RDX`, and returns into the
//! action's restorer, which calls `rt_sigreturn` to reload the registers
//! from the frame. A handler that edits `uc_mcontext` — `rip` most of all —
//! changes where the task resumes. Floating-point state is not saved: the
//! frame's `fpstate` pointer is null, which the kernel's own `rt_sigreturn`
//! also accepts.

use crate::errno::{EINVAL, ENOMEM, EPERM, SysResult};
use crate::guest;
use crate::process::{Crash, Machine, Task};
use crate::regs::{Reg, Regs};

pub(crate) const SIGILL: i32 = 4;
pub(crate) const SIGTRAP: i32 = 5;
pub(crate) const SIGFPE: i32 = 8;
pub(crate) const SIGSEGV: i32 = 11;

/// `si_code` values.
pub(crate) const SI_USER: i32 = 0;
pub(crate) const SI_KERNEL: i32 = 0x80;
pub(crate) const SEGV_MAPERR: i32 = 1;
pub(crate) const SEGV_ACCERR: i32 = 2;
pub(crate) const FPE_INTDIV: i32 = 1;
pub(crate) const ILL_ILLOPN: i32 = 2;
pub(crate) const TRAP_TRACE: i32 = 2;

const SA_ONSTACK: u64 = 0x0800_0000;
const SA_RESTORER: u64 = 0x0400_0000;
const SA_NODEFER: u64 = 0x4000_0000;
const SA_RESETHAND: u64 = 0x8000_0000;

const SS_ONSTACK: u32 = 1;
const SS_DISABLE: u32 = 2;
const SS_AUTODISARM: u32 = 1 << 31;
const MINSIGSTKSZ: u64 = 2048;

/// `rt_sigframe`: the restorer's address, a `ucontext` and a `siginfo`.
const UC: u64 = 8;
const UC_STACK: u64 = UC + 16;
const MCONTEXT: u64 = UC + 40;
const SIGMASK: u64 = UC + 296;
const INFO: u64 = UC + 304;
const FRAME: u64 = INFO + 128;
/// Below the interrupted stack pointer, which the ABI lets a leaf use.
const RED_ZONE: u64 = 128;

/// `uc_mcontext.gregs` slots after the sixteen general registers.
const GREG_RIP: usize = 16;
const GREG_EFL: usize = 17;
const GREG_CSGSFS: usize = 18;
const GREG_ERR: usize = 19;
const GREG_TRAPNO: usize = 20;
const GREG_OLDMASK: usize = 21;
const GREG_CR2: usize = 22;
/// `cs` 0x33 and `ss` 0x2b, the 64-bit user selectors.
const CSGSFS: u64 = 0x33 | 0x2b << 48;

/// What a handler learns about the signal, and what the trap recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SigInfo {
    pub signo: i32,
    pub code: i32,
    /// `si_addr` for a fault, and `cr2` for a page fault.
    pub addr: u64,
    /// The exception vector, for `uc_mcontext.trapno`.
    pub trapno: u64,
    /// The exception's error code, for `uc_mcontext.err`.
    pub err: u64,
    /// The sender, for a signal sent rather than raised.
    pub pid: u64,
}

impl SigInfo {
    pub(crate) fn raised(signo: i32, code: i32, addr: u64, trapno: u64) -> Self {
        Self {
            signo,
            code,
            addr,
            trapno,
            err: 0,
            pid: 0,
        }
    }

    /// A signal another task sent.
    pub(crate) fn sent(signo: i32, pid: u64) -> Self {
        Self {
            signo,
            code: SI_USER,
            addr: 0,
            trapno: 0,
            err: 0,
            pid,
        }
    }

    fn bytes(&self) -> [u8; 128] {
        let mut out = [0u8; 128];
        out[0..4].copy_from_slice(&self.signo.to_le_bytes());
        out[8..12].copy_from_slice(&self.code.to_le_bytes());
        if self.code == SI_USER {
            out[16..20].copy_from_slice(&(self.pid as u32).to_le_bytes());
        } else {
            out[16..24].copy_from_slice(&self.addr.to_le_bytes());
        }
        out
    }
}

/// The registers `uc_mcontext.gregs` holds first, in its order.
fn gregs(r: &Regs) -> [Reg; 16] {
    [
        r.r8, r.r9, r.r10, r.r11, r.r12, r.r13, r.r14, r.r15, r.rdi, r.rsi, r.rbp, r.rbx, r.rdx,
        r.rax, r.rcx, r.rsp,
    ]
}

/// Whether `rsp` is on the alternate signal stack `altstack`.
fn on_altstack(altstack: Option<(u64, u64)>, rsp: u64) -> bool {
    altstack.is_some_and(|(sp, size)| rsp > sp && rsp - sp <= size)
}

fn word(bytes: &[u8], index: usize) -> u64 {
    u64::from_le_bytes(bytes[index * 8..index * 8 + 8].try_into().expect("8 bytes"))
}

impl Task {
    /// The action for `signo` if a handler will run for it: installed,
    /// neither default nor ignored, and not blocked. A blocked or ignored
    /// signal the task raises itself is fatal, as on Linux.
    pub(crate) fn handler(&self, signo: i32) -> Option<[u8; 32]> {
        let act = self.sigactions.get(&(signo as u64))?;
        let blocked = self.signal_mask & 1 << (signo - 1) != 0;
        (word(act, 0) > 1 && !blocked).then_some(*act)
    }

    /// Enters the handler for `info.signo` with a frame recording the task
    /// as it is on the machine, positioned at `rip`. An error is the task
    /// killed by the signal: no handler, or a stack the frame cannot be
    /// written to.
    pub(crate) fn deliver(&mut self, info: SigInfo, rip: u64) -> Result<(), Crash> {
        let signo = info.signo;
        let Some(act) = self.handler(signo) else {
            return Err(self.crash(format!("signal {signo} with no handler"), Some(signo)));
        };
        let (handler, flags, restorer, mask) =
            (word(&act, 0), word(&act, 1), word(&act, 2), word(&act, 3));
        let old_mask = self.signal_mask;
        let altstack = self.altstack;
        let Machine { vm, regs, .. } = self.machine();
        let regs = *regs;
        let m = vm.memory_mut();
        let mut mc = [0u64; 32];
        for (slot, reg) in mc.iter_mut().zip(gregs(&regs)) {
            *slot = reg.read(m);
        }
        let rsp = mc[15];
        mc[GREG_RIP] = rip;
        mc[GREG_EFL] = regs.eflags(|reg| reg.read(m));
        mc[GREG_CSGSFS] = CSGSFS;
        mc[GREG_ERR] = info.err;
        mc[GREG_TRAPNO] = info.trapno;
        mc[GREG_OLDMASK] = old_mask;
        mc[GREG_CR2] = if signo == SIGSEGV { info.addr } else { 0 };

        let on_alt = on_altstack(altstack, rsp);
        let top = match altstack {
            Some((sp, size)) if flags & SA_ONSTACK != 0 && !on_alt => sp + size,
            _ => rsp.wrapping_sub(RED_ZONE),
        };
        // Aligned as a call leaves it: the frame's first word, the return
        // address, sits 8 below a 16-byte boundary.
        let sp = (top.wrapping_sub(FRAME) & !15).wrapping_sub(8);
        let mut frame = vec![0u8; FRAME as usize];
        let restorer = if flags & SA_RESTORER != 0 {
            restorer
        } else {
            0
        };
        frame[..8].copy_from_slice(&restorer.to_le_bytes());
        let (ss_sp, ss_flags, ss_size) = match altstack {
            Some((sp, size)) => (sp, if on_alt { SS_ONSTACK } else { 0 }, size),
            None => (0, SS_DISABLE, 0),
        };
        let at = UC_STACK as usize;
        frame[at..at + 8].copy_from_slice(&ss_sp.to_le_bytes());
        frame[at + 8..at + 12].copy_from_slice(&ss_flags.to_le_bytes());
        frame[at + 16..at + 24].copy_from_slice(&ss_size.to_le_bytes());
        for (i, value) in mc.iter().enumerate() {
            let at = MCONTEXT as usize + i * 8;
            frame[at..at + 8].copy_from_slice(&value.to_le_bytes());
        }
        let at = SIGMASK as usize;
        frame[at..at + 8].copy_from_slice(&old_mask.to_le_bytes());
        frame[INFO as usize..].copy_from_slice(&info.bytes());
        if guest::write(&mut m.mmu, sp, &frame).is_err() {
            let reason = format!("cannot push a frame for signal {signo} at {sp:#x}");
            return Err(self.crash(reason, Some(SIGSEGV)));
        }

        regs.rsp.write(m, sp);
        regs.rdi.write(m, signo as u64);
        regs.rsi.write(m, sp + INFO);
        regs.rdx.write(m, sp + UC);
        regs.rax.write(m, 0);
        // The ABI enters a function with the direction flag clear.
        regs.set_eflags(mc[GREG_EFL] & !(1 << 10), |reg, value| reg.write(m, value));
        if let Err(error) = vm.position_at(handler) {
            let reason = format!("cannot enter the handler for signal {signo}: {error}");
            return Err(self.crash(reason, Some(SIGSEGV)));
        }

        self.signal_mask |= mask;
        if flags & SA_NODEFER == 0 {
            self.signal_mask |= 1 << (signo - 1);
        }
        if flags & SA_RESETHAND != 0 {
            self.sigactions.remove(&(signo as u64));
        }
        Ok(())
    }

    /// `rt_sigreturn`: reloads the registers and mask from the frame the
    /// handler's `ret` popped the restorer from, and resumes at its `rip`.
    /// There is no value for `RAX`: the frame has it.
    pub(crate) fn sys_rt_sigreturn(&mut self) -> SysResult {
        let Machine { vm, regs, .. } = self.machine();
        let regs = *regs;
        let m = vm.memory_mut();
        let frame = regs.rsp.read(m).wrapping_sub(8);
        let restored = guest::read(&m.mmu, frame + MCONTEXT, 256)
            .and_then(|mc| Ok((mc, guest::read_u64(&m.mmu, frame + SIGMASK)?)));
        let (mc, mask) = match restored {
            Ok(frame) => frame,
            Err(_) => {
                self.exit_code = Some(128 + SIGSEGV);
                return Ok(0);
            }
        };
        for (i, reg) in gregs(&regs).into_iter().enumerate() {
            reg.write(m, word(&mc, i));
        }
        regs.set_eflags(word(&mc, GREG_EFL), |reg, value| reg.write(m, value));
        let rip = word(&mc, GREG_RIP);
        let rax = word(&mc, 13);
        if vm.position_at(rip).is_err() {
            self.exit_code = Some(128 + SIGSEGV);
            return Ok(0);
        }
        // SIGKILL and SIGSTOP cannot be blocked.
        self.signal_mask = mask & !(1 << 8 | 1 << 18);
        Ok(rax)
    }

    /// `sigaltstack`: reports the alternate stack and replaces it.
    pub(crate) fn sys_sigaltstack(&mut self, ss: u64, old: u64) -> SysResult {
        let rsp = {
            let Machine { vm, regs, .. } = self.machine();
            regs.rsp.read(vm.memory_mut())
        };
        let on_alt = on_altstack(self.altstack, rsp);
        let new = if ss != 0 {
            Some(guest::read(self.mmu(), ss, 24)?)
        } else {
            None
        };
        if old != 0 {
            let (sp, flags, size) = match self.altstack {
                Some((sp, size)) => (sp, if on_alt { SS_ONSTACK } else { 0 }, size),
                None => (0, SS_DISABLE, 0),
            };
            let mut bytes = [0u8; 24];
            bytes[..8].copy_from_slice(&sp.to_le_bytes());
            bytes[8..12].copy_from_slice(&flags.to_le_bytes());
            bytes[16..].copy_from_slice(&size.to_le_bytes());
            guest::write(self.mmu(), old, &bytes)?;
        }
        if let Some(new) = new {
            if on_alt {
                return Err(EPERM);
            }
            let flags = u32::from_le_bytes(new[8..12].try_into().expect("4 bytes"));
            if flags & SS_DISABLE != 0 {
                self.altstack = None;
            } else if flags & !SS_AUTODISARM != 0 {
                return Err(EINVAL);
            } else {
                let size = word(&new, 2);
                if size < MINSIGSTKSZ {
                    return Err(ENOMEM);
                }
                self.altstack = Some((word(&new, 0), size));
            }
        }
        Ok(0)
    }
}
