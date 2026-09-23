//! The Linux x86-64 system call dispatcher.
//!
//! Number in `RAX`, arguments in `RDI, RSI, RDX, R10, R8, R9`, result in
//! `RAX` as a value or `-errno`. The kernel clobbers `RCX` and `R11`, which
//! the SLEIGH constructor does not model; no userland code relies on them
//! surviving a call, so nothing here touches them.

use std::fs::File;

use crate::fs::FdKind;
use std::os::unix::fs::FileExt;
use std::time::{SystemTime, UNIX_EPOCH};

use log::warn;
use qcode_vm::{PAGE_SIZE, perm};

use crate::errno::{
    self, EAGAIN, ECHILD, EINVAL, ENOMEM, ENOSYS, ENOTTY, ERANGE, Errno, SysResult,
};
use crate::fs::{AT_FDCWD, O_CLOEXEC};
use crate::guest;
use crate::loader::page_up;
use crate::process::{AddressSpace, CloneArgs, Machine, PTRACE_TRACEME, Request, Task};

const PROT_READ: u64 = 1;
const PROT_WRITE: u64 = 2;
const PROT_EXEC: u64 = 4;
const MAP_SHARED: u64 = 0x1;
const MAP_FIXED: u64 = 0x10;
const MAP_ANONYMOUS: u64 = 0x20;
const MAP_FIXED_NOREPLACE: u64 = 0x10_0000;

const ARCH_SET_GS: u64 = 0x1001;
const ARCH_SET_FS: u64 = 0x1002;
const ARCH_GET_FS: u64 = 0x1003;
const ARCH_GET_GS: u64 = 0x1004;

const AT_SYMLINK_NOFOLLOW: u64 = 0x100;
const AT_REMOVEDIR: u64 = 0x200;
const AT_EMPTY_PATH: u64 = 0x1000;

const SIGCHLD: u64 = 17;
const CLONE_VM: u64 = 0x100;
const CLONE_VFORK: u64 = 0x4000;
const CLONE_PARENT_SETTID: u64 = 0x10_0000;
const CLONE_CHILD_CLEARTID: u64 = 0x20_0000;
const CLONE_CHILD_SETTID: u64 = 0x100_0000;
/// The `clone` flags a fork or a vfork can honour: glibc's `fork` passes
/// `CLONE_CHILD_SETTID | CLONE_CHILD_CLEARTID | SIGCHLD`, `posix_spawn`
/// `CLONE_VM | CLONE_VFORK | SIGCHLD`. Any other asks for sharing — a
/// thread, a namespace — that tasks with their own memory cannot provide.
const CLONE_KNOWN: u64 =
    CLONE_VM | CLONE_VFORK | CLONE_PARENT_SETTID | CLONE_CHILD_CLEARTID | CLONE_CHILD_SETTID;
const WNOHANG: u64 = 1;
const WSTOPPED: u64 = 2;
const WEXITED: u64 = 4;
const P_ALL: u64 = 0;
const P_PID: u64 = 1;
const CLD_EXITED: i32 = 1;
const CLD_KILLED: i32 = 2;
const CLD_TRAPPED: i32 = 4;
const SIGPIPE: u64 = 13;
const POLLIN: u16 = 1;
const POLLOUT: u16 = 4;
const POLLNVAL: u16 = 0x20;

/// Names and argument counts, for the trace.
fn describe(nr: u64) -> (&'static str, usize) {
    match nr {
        0 => ("read", 3),
        1 => ("write", 3),
        101 => ("ptrace", 4),
        247 => ("waitid", 5),
        2 => ("open", 3),
        3 => ("close", 1),
        4 => ("stat", 2),
        5 => ("fstat", 2),
        6 => ("lstat", 2),
        7 => ("poll", 3),
        8 => ("lseek", 3),
        9 => ("mmap", 6),
        10 => ("mprotect", 3),
        11 => ("munmap", 2),
        12 => ("brk", 1),
        13 => ("rt_sigaction", 4),
        15 => ("rt_sigreturn", 0),
        14 => ("rt_sigprocmask", 4),
        16 => ("ioctl", 3),
        17 => ("pread64", 4),
        18 => ("pwrite64", 4),
        19 => ("readv", 3),
        20 => ("writev", 3),
        21 => ("access", 2),
        22 => ("pipe", 1),
        56 => ("clone", 5),
        57 => ("fork", 0),
        58 => ("vfork", 0),
        59 => ("execve", 3),
        61 => ("wait4", 4),
        24 => ("sched_yield", 0),
        28 => ("madvise", 3),
        32 => ("dup", 1),
        33 => ("dup2", 2),
        35 => ("nanosleep", 2),
        39 => ("getpid", 0),
        60 => ("exit", 1),
        62 => ("kill", 2),
        63 => ("uname", 1),
        72 => ("fcntl", 3),
        79 => ("getcwd", 2),
        80 => ("chdir", 1),
        82 => ("rename", 2),
        83 => ("mkdir", 2),
        84 => ("rmdir", 1),
        87 => ("unlink", 1),
        89 => ("readlink", 3),
        96 => ("gettimeofday", 2),
        102 => ("getuid", 0),
        104 => ("getgid", 0),
        107 => ("geteuid", 0),
        108 => ("getegid", 0),
        110 => ("getppid", 0),
        131 => ("sigaltstack", 2),
        158 => ("arch_prctl", 2),
        186 => ("gettid", 0),
        201 => ("time", 1),
        202 => ("futex", 6),
        217 => ("getdents64", 3),
        218 => ("set_tid_address", 1),
        228 => ("clock_gettime", 2),
        229 => ("clock_getres", 2),
        230 => ("clock_nanosleep", 4),
        231 => ("exit_group", 1),
        234 => ("tgkill", 3),
        257 => ("openat", 4),
        258 => ("mkdirat", 3),
        263 => ("unlinkat", 3),
        264 => ("renameat", 4),
        271 => ("ppoll", 5),
        262 => ("newfstatat", 4),
        267 => ("readlinkat", 4),
        269 => ("faccessat", 3),
        273 => ("set_robust_list", 2),
        292 => ("dup3", 3),
        293 => ("pipe2", 2),
        302 => ("prlimit64", 4),
        318 => ("getrandom", 3),
        334 => ("rseq", 4),
        439 => ("faccessat2", 4),
        _ => ("unknown", 6),
    }
}

fn prot_to_perm(prot: u64) -> u8 {
    let mut bits = perm::MAP | perm::INIT;
    if prot & PROT_READ != 0 {
        bits |= perm::READ;
    }
    if prot & PROT_WRITE != 0 {
        bits |= perm::WRITE;
    }
    if prot & PROT_EXEC != 0 {
        bits |= perm::EXEC;
    }
    bits
}

fn timespec(secs: i64, nanos: i64) -> [u8; 16] {
    let mut out = [0; 16];
    out[..8].copy_from_slice(&secs.to_le_bytes());
    out[8..].copy_from_slice(&nanos.to_le_bytes());
    out
}

/// A `MAP_SHARED` mapping of a file: the guest's bytes are carried back to
/// the file at `msync`, at `munmap`, and before the file is mapped again,
/// which is when another mapping could observe them.
#[derive(Debug)]
pub(crate) struct SharedMap {
    pub file: File,
    pub at: u64,
    pub len: u64,
    pub offset: u64,
}

impl SharedMap {
    pub(crate) fn duplicate(&self) -> Option<Self> {
        self.file.try_clone().ok().map(|file| Self {
            file,
            at: self.at,
            len: self.len,
            offset: self.offset,
        })
    }
}

impl Task {
    /// Services the `syscall` the machine is stopped at and returns the value
    /// for `RAX`.
    pub(crate) fn syscall(&mut self) -> u64 {
        let Machine { vm, regs, .. } = self.machine();
        let m = vm.memory_mut();
        let nr = regs.rax.read(m);
        let args = [
            regs.rdi.read(m),
            regs.rsi.read(m),
            regs.rdx.read(m),
            regs.r10.read(m),
            regs.r8.read(m),
            regs.r9.read(m),
        ];
        let result = self.dispatch(nr, args);
        if self.trace && self.request != Some(Request::Block) {
            let (name, arity) = describe(nr);
            let shown: Vec<String> = args[..arity].iter().map(|a| format!("{a:#x}")).collect();
            let outcome = match result {
                Ok(v) => format!("{v:#x}"),
                Err(e) => format!("-1 {e}"),
            };
            eprintln!("[{nr}] {name}({}) = {outcome}", shown.join(", "));
        }
        match result {
            Ok(v) => v,
            Err(e) => e.as_ret(),
        }
    }

    fn dispatch(&mut self, nr: u64, a: [u64; 6]) -> SysResult {
        match nr {
            0 => self.sys_read(a[0] as i32, a[1], a[2]),
            1 => self.sys_write(a[0] as i32, a[1], a[2]),
            2 => self.sys_openat(AT_FDCWD, a[0], a[1], a[2]),
            3 => self.files.close(a[0] as i32).map(|()| 0),
            4 => self.sys_stat(AT_FDCWD, a[0], a[1], true),
            5 => self.sys_fstat(a[0] as i32, a[1]),
            6 => self.sys_stat(AT_FDCWD, a[0], a[1], false),
            7 => self.sys_poll(a[0], a[1], a[2] as i64),
            271 => {
                let timeout = if a[2] == 0 { -1 } else { 1 };
                self.sys_poll(a[0], a[1], timeout)
            }
            8 => self.files.lseek(a[0] as i32, a[1] as i64, a[2] as u32),
            9 => self.sys_mmap(a[0], a[1], a[2], a[3], a[4] as i32, a[5]),
            10 => self.sys_mprotect(a[0], a[1], a[2]),
            26 => {
                self.sync_shared(a[0], a[1]);
                Ok(0)
            }
            11 => self.sys_munmap(a[0], a[1]),
            12 => Ok(self.set_brk(a[0])),
            13 => self.sys_rt_sigaction(a[0], a[1], a[2]),
            15 => self.sys_rt_sigreturn(),
            14 => self.sys_rt_sigprocmask(a[0], a[1], a[2], a[3]),
            16 => self.sys_ioctl(a[0] as i32, a[1], a[2]),
            17 => self.sys_pread(a[0] as i32, a[1], a[2], a[3]),
            18 => self.sys_pwrite(a[0] as i32, a[1], a[2], a[3]),
            19 => self.sys_readv(a[0] as i32, a[1], a[2]),
            20 => self.sys_writev(a[0] as i32, a[1], a[2]),
            21 => self.sys_access(AT_FDCWD, a[0]),
            22 => self.sys_pipe(a[0], 0),
            293 => self.sys_pipe(a[0], a[1]),
            24 | 28 | 35 | 230 => Ok(0),
            32 => self.files.dup(a[0] as i32, None, false).map(|fd| fd as u64),
            33 => self.sys_dup2(a[0] as i32, a[1] as i32),
            39 | 186 => Ok(self.pid),
            56 => self.sys_clone(a[0], a[1], a[2], a[3]),
            57 => {
                self.request = Some(Request::Fork(CloneArgs::default()));
                Ok(0)
            }
            58 => {
                self.request = Some(Request::Fork(CloneArgs {
                    vfork: true,
                    share_vm: true,
                    ..CloneArgs::default()
                }));
                Ok(0)
            }
            59 => self.sys_execve(a[0], a[1], a[2]),
            61 => self.sys_wait4(a[0] as i64, a[1], a[2]),
            101 => self.sys_ptrace(a[0], a[1], a[2], a[3]),
            247 => self.sys_waitid(a[0], a[1], a[2], a[3]),
            60 | 231 => {
                self.exit_code = Some((a[0] & 0xff) as i32);
                Ok(0)
            }
            62 | 234 => self.sys_kill(a),
            63 => self.sys_uname(a[0]),
            72 => self.sys_fcntl(a[0] as i32, a[1], a[2]),
            77 => self.files.truncate(a[0] as i32, a[1]).map(|()| 0),
            79 => self.sys_getcwd(a[0], a[1]),
            80 => {
                let path = guest::read_path(self.mmu(), a[0])?;
                self.files.chdir(&path).map(|()| 0)
            }
            82 => self.sys_renameat(AT_FDCWD, a[0], AT_FDCWD, a[1]),
            264 => self.sys_renameat(a[0] as i32, a[1], a[2] as i32, a[3]),
            83 => self.sys_mkdirat(AT_FDCWD, a[0], a[1]),
            258 => self.sys_mkdirat(a[0] as i32, a[1], a[2]),
            84 => self.sys_unlinkat(AT_FDCWD, a[0], AT_REMOVEDIR),
            87 => self.sys_unlinkat(AT_FDCWD, a[0], 0),
            263 => self.sys_unlinkat(a[0] as i32, a[1], a[2]),
            89 => self.sys_readlinkat(AT_FDCWD, a[0], a[1], a[2]),
            96 => self.sys_gettimeofday(a[0]),
            102 | 107 => Ok(self.identity.uid),
            104 | 108 => Ok(self.identity.gid),
            110 => Ok(self.ppid),
            131 => self.sys_sigaltstack(a[0], a[1]),
            158 => self.sys_arch_prctl(a[0], a[1]),
            201 => self.sys_time(a[0]),
            202 => self.sys_futex(a[0], a[1], a[2]),
            217 => self.sys_getdents64(a[0] as i32, a[1], a[2]),
            218 => {
                self.clear_child_tid = a[0];
                Ok(self.pid)
            }
            228 => self.sys_clock_gettime(a[0], a[1]),
            229 => {
                guest::write(self.mmu(), a[1], &timespec(0, 1))?;
                Ok(0)
            }
            257 => self.sys_openat(a[0] as i32, a[1], a[2], a[3]),
            262 => {
                let path = guest::read_path(self.mmu(), a[1])?;
                if a[3] & AT_EMPTY_PATH != 0 && path.is_empty() {
                    return self.sys_fstat(a[0] as i32, a[2]);
                }
                self.sys_stat(a[0] as i32, a[1], a[2], a[3] & AT_SYMLINK_NOFOLLOW == 0)
            }
            267 => self.sys_readlinkat(a[0] as i32, a[1], a[2], a[3]),
            269 | 439 => self.sys_access(a[0] as i32, a[1]),
            273 => Ok(0),
            292 => {
                if a[0] == a[1] {
                    return Err(EINVAL);
                }
                self.files
                    .dup(
                        a[0] as i32,
                        Some(a[1] as i32),
                        a[2] & u64::from(O_CLOEXEC) != 0,
                    )
                    .map(|fd| fd as u64)
            }
            302 => self.sys_prlimit64(a[1], a[3]),
            318 => self.sys_getrandom(a[0], a[1]),
            319 => self.sys_memfd_create(a[0], a[1]),
            334 => Err(ENOSYS),
            _ => {
                warn!("unimplemented syscall {nr} ({})", describe(nr).0);
                Err(ENOSYS)
            }
        }
    }

    /// `clone(flags, stack, parent_tid, child_tid, tls)` as a fork or a
    /// vfork: the flags glibc's `fork` and `posix_spawn` pass, and no
    /// thread.
    fn sys_clone(&mut self, flags: u64, stack: u64, parent_tid: u64, child_tid: u64) -> SysResult {
        let exit_signal = flags & 0xff;
        if exit_signal != SIGCHLD && exit_signal != 0 {
            return Err(EINVAL);
        }
        if flags & !0xff & !CLONE_KNOWN != 0 {
            warn!("clone with unsupported flags {flags:#x}");
            return Err(ENOSYS);
        }
        let vfork = flags & CLONE_VFORK != 0;
        if flags & CLONE_VM != 0 && !vfork {
            warn!("clone of a thread ({flags:#x})");
            return Err(ENOSYS);
        }
        let when = |flag: u64, addr: u64| if flags & flag != 0 { addr } else { 0 };
        self.request = Some(Request::Fork(CloneArgs {
            vfork,
            share_vm: flags & CLONE_VM != 0,
            stack,
            parent_tid: when(CLONE_PARENT_SETTID, parent_tid),
            child_tid: when(CLONE_CHILD_SETTID, child_tid),
            clear_tid: when(CLONE_CHILD_CLEARTID, child_tid),
        }));
        Ok(0)
    }

    /// `memfd_create`: an anonymous file. Sealing and huge pages are
    /// accepted and ignored; the name only shows in the descriptor's path.
    fn sys_memfd_create(&mut self, name: u64, flags: u64) -> SysResult {
        const MFD_CLOEXEC: u64 = 1;
        const MFD_KNOWN: u64 = 0x1f;
        if flags & !MFD_KNOWN != 0 {
            return Err(EINVAL);
        }
        let name = guest::read_path(self.mmu(), name)?;
        self.files
            .memfd(&name, flags & MFD_CLOEXEC != 0)
            .map(|fd| fd as u64)
    }

    fn sys_read(&mut self, fd: i32, buf: u64, count: u64) -> SysResult {
        let count = usize::try_from(count).map_err(|_| EINVAL)?;
        if count > 1 << 30 {
            return Err(EINVAL);
        }
        // A read into an unwritable buffer must fail before consuming input.
        if count > 0 {
            let probe = guest::read(self.mmu(), buf, 1)?;
            guest::write(self.mmu(), buf, &probe)?;
        }
        if self.files.read_would_block(fd) {
            self.request = Some(Request::Block);
            return Ok(0);
        }
        let mut host = vec![0; count];
        let n = self.files.read(fd, &mut host)?;
        guest::write(self.mmu(), buf, &host[..n])?;
        Ok(n as u64)
    }

    fn sys_pread(&mut self, fd: i32, buf: u64, count: u64, offset: u64) -> SysResult {
        let count = usize::try_from(count).map_err(|_| EINVAL)?;
        let mut host = vec![0; count.min(1 << 30)];
        let n = self.files.pread(fd, &mut host, offset)?;
        guest::write(self.mmu(), buf, &host[..n])?;
        Ok(n as u64)
    }

    fn sys_write(&mut self, fd: i32, buf: u64, count: u64) -> SysResult {
        let bytes = guest::read(self.mmu(), buf, usize::try_from(count).map_err(|_| EINVAL)?)?;
        if self.files.write_would_block(fd) {
            self.request = Some(Request::Block);
            return Ok(0);
        }
        let result = self.files.write(fd, &bytes);
        // A write with nobody left to read it kills the writer unless it
        // asked otherwise, which is how `yes | head` ends quietly.
        if result == Err(errno::EPIPE) && !self.handles(SIGPIPE) {
            self.exit_code = Some(128 + SIGPIPE as i32);
        }
        result.map(|n| n as u64)
    }

    /// Whether the task installed a handler for `signal` or ignores it.
    fn handles(&self, signal: u64) -> bool {
        self.sigactions
            .get(&signal)
            .is_some_and(|act| act[..8] != [0; 8])
    }

    /// `poll`: reports which descriptors are ready, or waits until one is.
    /// A timeout is not timed; a positive one waits like a negative one.
    fn sys_poll(&mut self, fds: u64, nfds: u64, timeout: i64) -> SysResult {
        if nfds > 1024 {
            return Err(EINVAL);
        }
        let mut ready = 0u64;
        let mut revents = Vec::with_capacity(nfds as usize);
        for i in 0..nfds {
            let entry = guest::read(self.mmu(), fds + i * 8, 8)?;
            let fd = i32::from_le_bytes(entry[..4].try_into().unwrap());
            let events = u16::from_le_bytes(entry[4..6].try_into().unwrap());
            let mut out = 0u16;
            if fd >= 0 {
                match self.files.get(fd) {
                    Err(_) => out |= POLLNVAL,
                    Ok(_) => {
                        if events & POLLIN != 0 && !self.files.read_would_block(fd) {
                            out |= POLLIN;
                        }
                        if events & POLLOUT != 0 && !self.files.write_would_block(fd) {
                            out |= POLLOUT;
                        }
                        out |= self.files.poll_hangup(fd);
                    }
                }
            }
            if out != 0 {
                ready += 1;
            }
            revents.push(out);
        }
        if ready == 0 && timeout != 0 {
            self.request = Some(Request::Block);
            return Ok(0);
        }
        for (i, out) in revents.iter().enumerate() {
            guest::write(self.mmu(), fds + i as u64 * 8 + 6, &out.to_le_bytes())?;
        }
        Ok(ready)
    }

    fn sys_unlinkat(&mut self, dirfd: i32, path: u64, flags: u64) -> SysResult {
        let path = guest::read_path(self.mmu(), path)?;
        let host = self.files.resolve(dirfd, &path)?;
        let result = if flags & AT_REMOVEDIR != 0 {
            std::fs::remove_dir(&host)
        } else {
            std::fs::remove_file(&host)
        };
        result.map(|()| 0).map_err(|e| errno::from_io(&e))
    }

    fn sys_mkdirat(&mut self, dirfd: i32, path: u64, mode: u64) -> SysResult {
        use std::os::unix::fs::DirBuilderExt;
        let path = guest::read_path(self.mmu(), path)?;
        let host = self.files.resolve(dirfd, &path)?;
        std::fs::DirBuilder::new()
            .mode(mode as u32 & 0o7777)
            .create(&host)
            .map(|()| 0)
            .map_err(|e| errno::from_io(&e))
    }

    fn sys_renameat(&mut self, olddir: i32, old: u64, newdir: i32, new: u64) -> SysResult {
        let old = guest::read_path(self.mmu(), old)?;
        let new = guest::read_path(self.mmu(), new)?;
        let from = self.files.resolve(olddir, &old)?;
        let to = self.files.resolve(newdir, &new)?;
        std::fs::rename(from, to)
            .map(|()| 0)
            .map_err(|e| errno::from_io(&e))
    }

    fn sys_pipe(&mut self, fds: u64, flags: u64) -> SysResult {
        // A write into an unmapped array must fail before the pipe exists.
        guest::read(self.mmu(), fds, 8)?;
        let (r, w) = self.files.pipe(flags & u64::from(O_CLOEXEC) != 0)?;
        let mut out = [0u8; 8];
        out[..4].copy_from_slice(&r.to_le_bytes());
        out[4..].copy_from_slice(&w.to_le_bytes());
        guest::write(self.mmu(), fds, &out)?;
        Ok(0)
    }

    /// Reads a NUL-terminated array of C strings.
    fn read_string_vector(&mut self, base: u64) -> Result<Vec<String>, Errno> {
        let mut out = Vec::new();
        if base == 0 {
            return Ok(out);
        }
        for i in 0..4096u64 {
            let ptr = guest::read_u64(self.mmu(), base + i * 8)?;
            if ptr == 0 {
                return Ok(out);
            }
            out.push(String::from_utf8_lossy(&guest::read_cstr(self.mmu(), ptr)?).into_owned());
        }
        Err(errno::E2BIG)
    }

    fn sys_execve(&mut self, path: u64, argv: u64, envp: u64) -> SysResult {
        let path = guest::read_path(self.mmu(), path)?;
        let argv = self.read_string_vector(argv)?;
        let envp = self.read_string_vector(envp)?;
        let guest_path = self.files.guest_absolute(AT_FDCWD, &path)?;
        let host = if guest_path == "/proc/self/exe" {
            std::path::PathBuf::from(&self.files.exe_path)
        } else {
            self.files.resolve(AT_FDCWD, &path)?
        };
        let image = std::fs::read(&host).map_err(|e| errno::from_io(&e))?;
        if std::fs::metadata(&host).is_ok_and(|m| m.is_dir()) {
            return Err(errno::EACCES);
        }
        self.exec(&image, host.display().to_string(), &argv, &envp)?;
        Ok(0)
    }

    /// `ptrace`: the caller asks to be traced by its parent here; anything
    /// on another task is the scheduler's to do.
    fn sys_ptrace(&mut self, request: u64, pid: u64, addr: u64, data: u64) -> SysResult {
        if request == PTRACE_TRACEME {
            self.traced_by = Some(self.ppid);
            return Ok(0);
        }
        self.request = Some(Request::Ptrace {
            request,
            pid,
            addr,
            data,
        });
        Ok(0)
    }

    /// `waitid`: reports a stopped tracee or an exited child in a
    /// `siginfo_t`, or waits for one.
    fn sys_waitid(&mut self, idtype: u64, id: u64, infop: u64, options: u64) -> SysResult {
        let matches = |child: u64| match idtype {
            P_ALL => true,
            P_PID => child == id,
            _ => false,
        };
        if idtype != P_ALL && idtype != P_PID {
            return Err(EINVAL);
        }
        let report = |task: &mut Self, code: i32, pid: u64, status: i32| -> SysResult {
            if infop != 0 {
                let mut si = [0u8; 128];
                si[0..4].copy_from_slice(&(SIGCHLD as i32).to_le_bytes());
                si[8..12].copy_from_slice(&code.to_le_bytes());
                si[16..20].copy_from_slice(&(pid as i32).to_le_bytes());
                si[24..28].copy_from_slice(&status.to_le_bytes());
                guest::write(task.mmu(), infop, &si)?;
            }
            Ok(0)
        };
        if options & WSTOPPED != 0
            && let Some(i) = self.stops.iter().position(|&(c, _)| matches(c))
        {
            let (child, signal) = self.stops.remove(i);
            return report(self, CLD_TRAPPED, child, signal);
        }
        if options & WEXITED != 0
            && let Some(i) = self.zombies.iter().position(|&(c, _)| matches(c))
        {
            let (child, status) = self.zombies.remove(i);
            self.children.retain(|&c| c != child);
            let (code, value) = if status & 0x7f == 0 {
                (CLD_EXITED, status >> 8)
            } else {
                (CLD_KILLED, status & 0x7f)
            };
            return report(self, code, child, value);
        }
        if !self
            .children
            .iter()
            .chain(&self.tracees)
            .any(|&c| matches(c))
        {
            return Err(ECHILD);
        }
        if options & WNOHANG != 0 {
            if infop != 0 {
                guest::write(self.mmu(), infop, &[0u8; 128])?;
            }
            return Ok(0);
        }
        self.request = Some(Request::Block);
        Ok(0)
    }

    fn sys_wait4(&mut self, pid: i64, status: u64, options: u64) -> SysResult {
        let matches = |child: u64| pid == -1 || pid == child as i64;
        // A tracee's stop goes to its tracer whatever the options say.
        if let Some(i) = self.stops.iter().position(|&(c, _)| matches(c)) {
            let (child, signal) = self.stops.remove(i);
            if status != 0 {
                guest::write_u32(self.mmu(), status, ((signal as u32) << 8) | 0x7f)?;
            }
            return Ok(child);
        }
        if let Some(i) = self.zombies.iter().position(|&(c, _)| matches(c)) {
            let (child, code) = self.zombies.remove(i);
            self.children.retain(|&c| c != child);
            if status != 0 {
                guest::write_u32(self.mmu(), status, code as u32)?;
            }
            return Ok(child);
        }
        if !self
            .children
            .iter()
            .chain(&self.tracees)
            .any(|&c| matches(c))
        {
            return Err(ECHILD);
        }
        if options & WNOHANG != 0 {
            return Ok(0);
        }
        self.request = Some(Request::Block);
        Ok(0)
    }

    fn sys_pwrite(&mut self, fd: i32, buf: u64, count: u64, offset: u64) -> SysResult {
        let bytes = guest::read(self.mmu(), buf, usize::try_from(count).map_err(|_| EINVAL)?)?;
        self.files.pwrite(fd, &bytes, offset).map(|n| n as u64)
    }

    fn iovecs(&mut self, iov: u64, count: u64) -> Result<Vec<(u64, u64)>, Errno> {
        if count > 1024 {
            return Err(EINVAL);
        }
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count {
            let base = guest::read_u64(self.mmu(), iov + i * 16)?;
            let len = guest::read_u64(self.mmu(), iov + i * 16 + 8)?;
            out.push((base, len));
        }
        Ok(out)
    }

    fn sys_writev(&mut self, fd: i32, iov: u64, count: u64) -> SysResult {
        let mut total = 0;
        for (base, len) in self.iovecs(iov, count)? {
            if len == 0 {
                continue;
            }
            let n = self.sys_write(fd, base, len)?;
            total += n;
            if n < len {
                break;
            }
        }
        Ok(total)
    }

    fn sys_readv(&mut self, fd: i32, iov: u64, count: u64) -> SysResult {
        let mut total = 0;
        for (base, len) in self.iovecs(iov, count)? {
            if len == 0 {
                continue;
            }
            let n = self.sys_read(fd, base, len)?;
            total += n;
            if n < len {
                break;
            }
        }
        Ok(total)
    }

    fn sys_openat(&mut self, dirfd: i32, path: u64, flags: u64, mode: u64) -> SysResult {
        let path = guest::read_path(self.mmu(), path)?;
        self.files
            .open(dirfd, &path, flags as u32, mode as u32)
            .map(|fd| fd as u64)
    }

    fn sys_stat(&mut self, dirfd: i32, path: u64, buf: u64, follow: bool) -> SysResult {
        let path = guest::read_path(self.mmu(), path)?;
        let stat = self.files.stat(dirfd, &path, follow)?;
        guest::write(self.mmu(), buf, &stat.to_bytes())?;
        Ok(0)
    }

    fn sys_fstat(&mut self, fd: i32, buf: u64) -> SysResult {
        let stat = self.files.fstat(fd)?;
        guest::write(self.mmu(), buf, &stat.to_bytes())?;
        Ok(0)
    }

    fn sys_access(&mut self, dirfd: i32, path: u64) -> SysResult {
        let path = guest::read_path(self.mmu(), path)?;
        self.files.access(dirfd, &path).map(|()| 0)
    }

    fn sys_readlinkat(&mut self, dirfd: i32, path: u64, buf: u64, size: u64) -> SysResult {
        let path = guest::read_path(self.mmu(), path)?;
        let target = self.files.readlink(dirfd, &path)?;
        let n = target.len().min(usize::try_from(size).map_err(|_| EINVAL)?);
        guest::write(self.mmu(), buf, &target[..n])?;
        Ok(n as u64)
    }

    fn sys_getcwd(&mut self, buf: u64, size: u64) -> SysResult {
        let mut cwd = self.files.cwd().as_bytes().to_vec();
        cwd.push(0);
        if (cwd.len() as u64) > size {
            return Err(ERANGE);
        }
        guest::write(self.mmu(), buf, &cwd)?;
        Ok(cwd.len() as u64)
    }

    fn sys_dup2(&mut self, old: i32, new: i32) -> SysResult {
        self.files.get(old)?;
        if old == new {
            return Ok(new as u64);
        }
        self.files.dup(old, Some(new), false).map(|fd| fd as u64)
    }

    fn sys_fcntl(&mut self, fd: i32, cmd: u64, arg: u64) -> SysResult {
        match cmd {
            // F_DUPFD, F_DUPFD_CLOEXEC
            0 | 1030 => {
                self.files.get(fd)?;
                let min = arg as i32;
                let new = self.files.dup(fd, None, cmd == 1030)?;
                if new < min {
                    self.files.close(new)?;
                    return self.files.dup(fd, Some(min), cmd == 1030).map(|f| f as u64);
                }
                Ok(new as u64)
            }
            // F_GETFD
            1 => Ok(u64::from(self.files.get(fd)?.cloexec)),
            // F_SETFD
            2 => {
                self.files.get_mut(fd)?.cloexec = arg & 1 != 0;
                Ok(0)
            }
            // F_GETFL
            3 => {
                let mode = u64::from(self.files.access_mode(fd)?);
                Ok(if self.files.get(fd)?.append {
                    mode | 0o2000
                } else {
                    mode
                })
            }
            // F_SETFL
            4 => {
                self.files.get(fd)?;
                Ok(0)
            }
            _ => Err(EINVAL),
        }
    }

    fn sys_mmap(
        &mut self,
        addr: u64,
        len: u64,
        prot: u64,
        flags: u64,
        fd: i32,
        offset: u64,
    ) -> SysResult {
        if len == 0 || !addr.is_multiple_of(PAGE_SIZE) || !offset.is_multiple_of(PAGE_SIZE) {
            return Err(EINVAL);
        }
        let len = page_up(len);
        let bits = prot_to_perm(prot);
        let Self { machine, space, .. } = self;
        let mmu = &machine
            .as_mut()
            .expect("the task holds the machine")
            .vm
            .memory_mut()
            .mmu;
        let at = if flags & (MAP_FIXED | MAP_FIXED_NOREPLACE) != 0 {
            if addr == 0 {
                return Err(EINVAL);
            }
            if flags & MAP_FIXED_NOREPLACE != 0 && !AddressSpace::is_free(mmu, addr, len) {
                return Err(errno::EEXIST);
            }
            addr
        } else if addr != 0 && AddressSpace::is_free(mmu, addr, len) {
            addr
        } else {
            space.find_free(mmu, len).ok_or(ENOMEM)?
        };
        if flags & MAP_FIXED != 0 {
            self.sync_shared(at, len);
            self.shared
                .retain(|m| m.at >= at + len || m.at + m.len <= at);
            self.shared_anon
                .retain(|&(s, l)| s >= at + len || s + l <= at);
        }
        if flags & MAP_SHARED != 0 && flags & MAP_ANONYMOUS != 0 {
            self.shared_anon.push((at, len));
        }
        if flags & MAP_ANONYMOUS == 0 {
            // Another mapping of the file may hold bytes this one must see.
            self.sync_shared(0, u64::MAX);
        }
        let mmu = self.mmu();
        if flags & MAP_FIXED != 0 {
            let _ = mmu.unmap(at, len);
        }
        mmu.map(at, len, bits).map_err(|_| ENOMEM)?;
        if flags & MAP_ANONYMOUS == 0 {
            // File-backed: a snapshot of the file's bytes. A MAP_SHARED
            // mapping is carried back to the file by `sync_shared`; a
            // private one never is.
            let mut data = vec![0; len as usize];
            let n = match self.files.pread(fd, &mut data, offset) {
                Ok(n) => n,
                Err(e) => {
                    let _ = self.mmu().unmap(at, len);
                    return Err(e);
                }
            };
            self.mmu().write_unchecked(at, &data[..n], bits);
            if flags & MAP_SHARED != 0
                && let Ok(fd) = self.files.get(fd)
                && let FdKind::File(file) = &fd.kind
                && let Ok(file) = file.try_clone()
            {
                self.shared.push(SharedMap {
                    file,
                    at,
                    len,
                    offset,
                });
            }
        }
        Ok(at)
    }

    /// Carries the guest's bytes of every shared mapping overlapping
    /// `[addr, addr + len)` back to its file.
    pub(crate) fn sync_shared(&mut self, addr: u64, len: u64) {
        let maps = std::mem::take(&mut self.shared);
        for m in &maps {
            let lo = m.at.max(addr);
            let hi = (m.at + m.len).min(addr.saturating_add(len));
            if lo < hi
                && let Ok(bytes) = guest::read(self.mmu(), lo, (hi - lo) as usize)
            {
                let _ = m.file.write_all_at(&bytes, m.offset + (lo - m.at));
            }
        }
        self.shared = maps;
    }

    fn sys_munmap(&mut self, addr: u64, len: u64) -> SysResult {
        if !addr.is_multiple_of(PAGE_SIZE) || len == 0 {
            return Err(EINVAL);
        }
        let len = page_up(len);
        self.sync_shared(addr, len);
        self.shared
            .retain(|m| m.at >= addr + len || m.at + m.len <= addr);
        self.shared_anon
            .retain(|&(s, l)| s >= addr + len || s + l <= addr);
        // Unmapping what is not mapped is not an error on Linux either.
        let _ = self.mmu().unmap(addr, len);
        Ok(0)
    }

    fn sys_mprotect(&mut self, addr: u64, len: u64, prot: u64) -> SysResult {
        if !addr.is_multiple_of(PAGE_SIZE) {
            return Err(EINVAL);
        }
        if len == 0 {
            return Ok(0);
        }
        self.mmu()
            .protect(addr, page_up(len), prot_to_perm(prot))
            .map_err(|_| ENOMEM)?;
        Ok(0)
    }

    fn sys_rt_sigaction(&mut self, signal: u64, act: u64, oldact: u64) -> SysResult {
        if !(1..=64).contains(&signal) {
            return Err(EINVAL);
        }
        if oldact != 0 {
            let old = self.sigactions.get(&signal).copied().unwrap_or([0; 32]);
            guest::write(self.mmu(), oldact, &old)?;
        }
        if act != 0 {
            let new: [u8; 32] = guest::read(self.mmu(), act, 32)?.try_into().unwrap();
            self.sigactions.insert(signal, new);
        }
        Ok(0)
    }

    fn sys_rt_sigprocmask(&mut self, how: u64, set: u64, oldset: u64, size: u64) -> SysResult {
        if size != 8 {
            return Err(EINVAL);
        }
        if oldset != 0 {
            let mask = self.signal_mask;
            guest::write_u64(self.mmu(), oldset, mask)?;
        }
        if set != 0 {
            let mask = guest::read_u64(self.mmu(), set)?;
            self.signal_mask = match how {
                0 => self.signal_mask | mask,
                1 => self.signal_mask & !mask,
                2 => mask,
                _ => return Err(EINVAL),
            };
        }
        Ok(0)
    }

    fn sys_ioctl(&mut self, fd: i32, request: u64, arg: u64) -> SysResult {
        if !self.files.is_tty(fd)? {
            return Err(self.files.ioctl_unsupported(fd));
        }
        match request {
            // TCGETS: a termios with the canonical-terminal defaults.
            0x5401 => {
                let mut termios = [0u8; 60];
                termios[0..4].copy_from_slice(&0x4500u32.to_le_bytes()); // c_iflag
                termios[4..8].copy_from_slice(&0x5u32.to_le_bytes()); // c_oflag
                termios[8..12].copy_from_slice(&0xbfu32.to_le_bytes()); // c_cflag
                termios[12..16].copy_from_slice(&0x8a3bu32.to_le_bytes()); // c_lflag
                guest::write(self.mmu(), arg, &termios)?;
                Ok(0)
            }
            // TIOCGWINSZ: rows, cols, xpixel, ypixel.
            0x5413 => {
                let mut ws = [0u8; 8];
                ws[0..2].copy_from_slice(&24u16.to_le_bytes());
                ws[2..4].copy_from_slice(&80u16.to_le_bytes());
                guest::write(self.mmu(), arg, &ws)?;
                Ok(0)
            }
            _ => Err(ENOTTY),
        }
    }

    fn sys_uname(&mut self, buf: u64) -> SysResult {
        let mut out = [0u8; 65 * 6];
        for (i, field) in [
            "Linux",
            "wazabin",
            "6.1.0-wazabin",
            "#1 SMP",
            "x86_64",
            "(none)",
        ]
        .iter()
        .enumerate()
        {
            out[i * 65..i * 65 + field.len()].copy_from_slice(field.as_bytes());
        }
        guest::write(self.mmu(), buf, &out)?;
        Ok(0)
    }

    fn sys_arch_prctl(&mut self, code: u64, addr: u64) -> SysResult {
        match code {
            ARCH_SET_FS => {
                let machine = self.machine();
                machine.regs.fs_base.write(machine.vm.memory_mut(), addr);
                Ok(0)
            }
            ARCH_SET_GS => {
                let machine = self.machine();
                machine.regs.gs_base.write(machine.vm.memory_mut(), addr);
                Ok(0)
            }
            ARCH_GET_FS => {
                let machine = self.machine();
                let base = machine.regs.fs_base.read(machine.vm.memory_mut());
                guest::write_u64(self.mmu(), addr, base)?;
                Ok(0)
            }
            ARCH_GET_GS => {
                let machine = self.machine();
                let base = machine.regs.gs_base.read(machine.vm.memory_mut());
                guest::write_u64(self.mmu(), addr, base)?;
                Ok(0)
            }
            _ => Err(EINVAL),
        }
    }

    fn sys_futex(&mut self, uaddr: u64, op: u64, val: u64) -> SysResult {
        // Strip FUTEX_PRIVATE_FLAG and FUTEX_CLOCK_REALTIME.
        match op & 0x7f {
            // FUTEX_WAIT, FUTEX_WAIT_BITSET: with one thread, nobody could ever
            // wake us, so a matching value is reported as a spurious wakeup.
            0 | 9 => {
                let current = guest::read_u32(self.mmu(), uaddr)?;
                if u64::from(current) != val & 0xffff_ffff {
                    Err(EAGAIN)
                } else {
                    Ok(0)
                }
            }
            // FUTEX_WAKE, FUTEX_WAKE_BITSET, FUTEX_REQUEUE, FUTEX_CMP_REQUEUE.
            1 | 10 | 3 | 4 => Ok(0),
            _ => Err(ENOSYS),
        }
    }

    fn sys_getdents64(&mut self, fd: i32, buf: u64, len: u64) -> SysResult {
        let records = self
            .files
            .getdents64(fd, usize::try_from(len).map_err(|_| EINVAL)?)?;
        guest::write(self.mmu(), buf, &records)?;
        Ok(records.len() as u64)
    }

    fn now(&self, clock: u64) -> Result<(i64, i64), Errno> {
        match clock {
            // CLOCK_REALTIME and its coarse variant.
            0 | 5 => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default();
                Ok((now.as_secs() as i64, i64::from(now.subsec_nanos())))
            }
            // CLOCK_MONOTONIC, PROCESS_CPUTIME, THREAD_CPUTIME, MONOTONIC_RAW,
            // MONOTONIC_COARSE, BOOTTIME: time since the process started, offset
            // so it never reads as zero.
            1..=4 | 6 | 7 => {
                let elapsed = self.started.elapsed();
                Ok((
                    elapsed.as_secs() as i64 + 1000,
                    i64::from(elapsed.subsec_nanos()),
                ))
            }
            _ => Err(EINVAL),
        }
    }

    fn sys_clock_gettime(&mut self, clock: u64, buf: u64) -> SysResult {
        let (secs, nanos) = self.now(clock)?;
        guest::write(self.mmu(), buf, &timespec(secs, nanos))?;
        Ok(0)
    }

    fn sys_gettimeofday(&mut self, buf: u64) -> SysResult {
        if buf != 0 {
            let (secs, nanos) = self.now(0)?;
            guest::write(self.mmu(), buf, &timespec(secs, nanos / 1000))?;
        }
        Ok(0)
    }

    fn sys_time(&mut self, buf: u64) -> SysResult {
        let (secs, _) = self.now(0)?;
        if buf != 0 {
            guest::write_u64(self.mmu(), buf, secs as u64)?;
        }
        Ok(secs as u64)
    }

    fn sys_prlimit64(&mut self, resource: u64, old: u64) -> SysResult {
        if old != 0 {
            let (cur, max) = match resource {
                // RLIMIT_STACK
                3 => (crate::stack::STACK_SIZE, u64::MAX),
                // RLIMIT_NOFILE
                7 => (1024, 1024),
                _ => (u64::MAX, u64::MAX),
            };
            let mut bytes = [0; 16];
            bytes[..8].copy_from_slice(&cur.to_le_bytes());
            bytes[8..].copy_from_slice(&max.to_le_bytes());
            guest::write(self.mmu(), old, &bytes)?;
        }
        Ok(0)
    }

    fn sys_getrandom(&mut self, buf: u64, len: u64) -> SysResult {
        let len = usize::try_from(len).map_err(|_| EINVAL)?.min(1 << 20);
        let mut bytes = Vec::with_capacity(len);
        while bytes.len() < len {
            // xorshift64*: deterministic, so runs are reproducible.
            self.rng ^= self.rng >> 12;
            self.rng ^= self.rng << 25;
            self.rng ^= self.rng >> 27;
            bytes.extend_from_slice(&self.rng.wrapping_mul(0x2545_f491_4f6c_dd1d).to_le_bytes());
        }
        bytes.truncate(len);
        guest::write(self.mmu(), buf, &bytes)?;
        Ok(len as u64)
    }

    fn sys_kill(&mut self, a: [u64; 6]) -> SysResult {
        // kill(pid, sig) or tgkill(tgid, tid, sig): a signal is never
        // delivered, so a fatal one ends its target as Linux would — reported
        // as 128 + signal, the shell's convention.
        let (target, signal) = if a[2] == 0 && a[1] < 65 {
            (a[0], a[1])
        } else {
            (a[1], a[2])
        };
        if signal == 0 {
            return Ok(0);
        }
        if target == self.pid || target == 0 {
            self.exit_code = Some(128 + signal as i32);
            return Ok(0);
        }
        // Another task of this process; the scheduler ends it. A pid that is
        // not one of ours is nobody we may signal.
        self.request = Some(Request::Kill {
            pid: target,
            signal: signal as i32,
        });
        Ok(0)
    }
}
