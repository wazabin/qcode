//! The Linux x86-64 system call dispatcher.
//!
//! Number in `RAX`, arguments in `RDI, RSI, RDX, R10, R8, R9`, result in
//! `RAX` as a value or `-errno`. The kernel clobbers `RCX` and `R11`, which
//! the SLEIGH constructor does not model; no userland code relies on them
//! surviving a call, so nothing here touches them.

use std::time::{SystemTime, UNIX_EPOCH};

use log::warn;
use qcode_vm::{PAGE_SIZE, perm};

use crate::errno::{
    self, EAGAIN, ECHILD, EINVAL, ENOMEM, ENOSYS, ENOTTY, ERANGE, Errno, SysResult,
};
use crate::fs::{AT_FDCWD, O_CLOEXEC};
use crate::guest;
use crate::loader::page_up;
use crate::process::{AddressSpace, Request, Task};

const PROT_READ: u64 = 1;
const PROT_WRITE: u64 = 2;
const PROT_EXEC: u64 = 4;
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
/// `clone` flags other than the exit signal; any of them asks for sharing a
/// fork cannot provide.
const CLONE_FLAGS: u64 = !0xff;
const WNOHANG: u64 = 1;
const SIGPIPE: u64 = 13;
const POLLIN: u16 = 1;
const POLLOUT: u16 = 4;
const POLLNVAL: u16 = 0x20;

/// Names and argument counts, for the trace.
fn describe(nr: u64) -> (&'static str, usize) {
    match nr {
        0 => ("read", 3),
        1 => ("write", 3),
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

impl Task {
    /// Services the `syscall` the machine is stopped at and returns the value
    /// for `RAX`.
    pub(crate) fn syscall(&mut self) -> u64 {
        let m = self.machine.vm.memory_mut();
        let nr = self.machine.regs.rax.read(m);
        let args = [
            self.machine.regs.rdi.read(m),
            self.machine.regs.rsi.read(m),
            self.machine.regs.rdx.read(m),
            self.machine.regs.r10.read(m),
            self.machine.regs.r8.read(m),
            self.machine.regs.r9.read(m),
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
            11 => self.sys_munmap(a[0], a[1]),
            12 => Ok(self.set_brk(a[0])),
            13 => self.sys_rt_sigaction(a[0], a[1], a[2]),
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
            56 => {
                if a[0] & CLONE_FLAGS != 0 || a[0] & 0xff != SIGCHLD {
                    return Err(ENOSYS);
                }
                self.request = Some(Request::Fork { vfork: false });
                Ok(0)
            }
            57 | 58 => {
                self.request = Some(Request::Fork { vfork: nr == 58 });
                Ok(0)
            }
            59 => self.sys_execve(a[0], a[1], a[2]),
            61 => self.sys_wait4(a[0] as i64, a[1], a[2]),
            60 | 231 => {
                self.exit_code = Some((a[0] & 0xff) as i32);
                Ok(0)
            }
            62 | 234 => self.sys_kill(a),
            63 => self.sys_uname(a[0]),
            72 => self.sys_fcntl(a[0] as i32, a[1], a[2]),
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
            131 => self.sys_sigaltstack(a[1]),
            158 => self.sys_arch_prctl(a[0], a[1]),
            201 => self.sys_time(a[0]),
            202 => self.sys_futex(a[0], a[1], a[2]),
            217 => self.sys_getdents64(a[0] as i32, a[1], a[2]),
            218 => Ok(self.pid),
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
            334 => Err(ENOSYS),
            _ => {
                warn!("unimplemented syscall {nr} ({})", describe(nr).0);
                Err(ENOSYS)
            }
        }
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

    fn sys_wait4(&mut self, pid: i64, status: u64, options: u64) -> SysResult {
        let matches = |child: u64| pid == -1 || pid == child as i64;
        if let Some(i) = self.zombies.iter().position(|&(c, _)| matches(c)) {
            let (child, code) = self.zombies.remove(i);
            self.children.retain(|&c| c != child);
            if status != 0 {
                guest::write_u32(self.mmu(), status, code as u32)?;
            }
            return Ok(child);
        }
        if !self.children.iter().any(|&c| matches(c)) {
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
        let mmu = &self.machine.vm.memory_mut().mmu;
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
            self.machine.space.find_free(mmu, len).ok_or(ENOMEM)?
        };
        let mmu = self.mmu();
        if flags & MAP_FIXED != 0 {
            let _ = mmu.unmap(at, len);
        }
        mmu.map(at, len, bits).map_err(|_| ENOMEM)?;
        if flags & MAP_ANONYMOUS == 0 {
            // File-backed: a private snapshot of the file's bytes. Writes are
            // never carried back to the host file, even for MAP_SHARED.
            let mut data = vec![0; len as usize];
            let n = match self.files.pread(fd, &mut data, offset) {
                Ok(n) => n,
                Err(e) => {
                    let _ = self.mmu().unmap(at, len);
                    return Err(e);
                }
            };
            self.mmu().write_unchecked(at, &data[..n], bits);
        }
        Ok(at)
    }

    fn sys_munmap(&mut self, addr: u64, len: u64) -> SysResult {
        if !addr.is_multiple_of(PAGE_SIZE) || len == 0 {
            return Err(EINVAL);
        }
        // Unmapping what is not mapped is not an error on Linux either.
        let _ = self.mmu().unmap(addr, page_up(len));
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

    fn sys_sigaltstack(&mut self, old: u64) -> SysResult {
        if old != 0 {
            // ss_sp, ss_flags = SS_DISABLE, ss_size.
            let mut bytes = [0; 24];
            bytes[8..12].copy_from_slice(&2u32.to_le_bytes());
            guest::write(self.mmu(), old, &bytes)?;
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
                self.machine
                    .regs
                    .fs_base
                    .write(self.machine.vm.memory_mut(), addr);
                Ok(0)
            }
            ARCH_SET_GS => {
                self.machine
                    .regs
                    .gs_base
                    .write(self.machine.vm.memory_mut(), addr);
                Ok(0)
            }
            ARCH_GET_FS => {
                let base = self.machine.regs.fs_base.read(self.machine.vm.memory_mut());
                guest::write_u64(self.mmu(), addr, base)?;
                Ok(0)
            }
            ARCH_GET_GS => {
                let base = self.machine.regs.gs_base.read(self.machine.vm.memory_mut());
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
