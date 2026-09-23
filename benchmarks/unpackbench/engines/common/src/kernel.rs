//! The Linux system calls the corpus needs, modelled on `userland/`: the same
//! stack top, `mmap` base, `brk` start and `MAP_SHARED` write-back (a shared
//! file mapping's bytes are carried back to the file at `msync`, at
//! `munmap`, and before the file is mapped again), so that UPX's memfd stub
//! runs and provenance windows land where they do under QCode.
//!
//! Files are read-only snapshots of host files, `memfd`s live in memory,
//! stdout and stderr are captured. The fork, exec and ptrace families stop
//! the run as unsupported: the baselines have no process model.

use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

use crate::{
    Guest, Reg,
    elf::{self, PAGE, page_up},
};

/// Top of the main thread's stack (exclusive), as in `userland/src/stack.rs`.
pub const STACK_TOP: u64 = 0x7fff_f000_0000;
pub const STACK_SIZE: u64 = 8 << 20;
/// Where hint-less mappings start, as in `userland/src/process.rs`.
pub const MMAP_BASE: u64 = 0x7f00_0000_0000;

const ENOENT: u64 = 2;
const EBADF: u64 = 9;
const ENOMEM: u64 = 12;
const EACCES: u64 = 13;
const EFAULT: u64 = 14;
const EEXIST: u64 = 17;
const EINVAL: u64 = 22;
const ENOTTY: u64 = 25;
const ESPIPE: u64 = 29;
const ERANGE: u64 = 34;
const ENOSYS: u64 = 38;

const MAP_SHARED: u64 = 0x01;
const MAP_FIXED: u64 = 0x10;
const MAP_ANONYMOUS: u64 = 0x20;
const MAP_FIXED_NOREPLACE: u64 = 0x10_0000;

/// Disjoint half-open intervals: what is mapped.
#[derive(Debug, Default, Clone)]
pub struct Ranges(BTreeMap<u64, u64>);

impl Ranges {
    /// The pieces of `start..end` that overlap a range.
    pub fn overlaps(&self, start: u64, end: u64) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        if let Some((&s, &e)) = self.0.range(..start).next_back()
            && e > start
        {
            out.push((start, e.min(end)));
            let _ = s;
        }
        for (&s, &e) in self.0.range(start..end) {
            out.push((s, e.min(end)));
        }
        out
    }
    /// The pieces of `start..end` no range covers.
    pub fn gaps(&self, start: u64, end: u64) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        let mut at = start;
        for (s, e) in self.overlaps(start, end) {
            if s > at {
                out.push((at, s));
            }
            at = at.max(e);
        }
        if at < end {
            out.push((at, end));
        }
        out
    }
    pub fn is_free(&self, start: u64, end: u64) -> bool {
        self.overlaps(start, end).is_empty()
    }
    pub fn insert(&mut self, start: u64, end: u64) {
        debug_assert!(self.is_free(start, end));
        self.0.insert(start, end);
    }
    /// Removes `start..end`, splitting the ranges it cuts.
    pub fn remove(&mut self, start: u64, end: u64) {
        let hit: Vec<(u64, u64)> = {
            let mut v = Vec::new();
            if let Some((&s, &e)) = self.0.range(..start).next_back()
                && e > start
            {
                v.push((s, e));
            }
            v.extend(self.0.range(start..end).map(|(&s, &e)| (s, e)));
            v
        };
        for (s, e) in hit {
            self.0.remove(&s);
            if s < start {
                self.0.insert(s, start);
            }
            if e > end {
                self.0.insert(end, e);
            }
        }
    }
}

type Data = Rc<RefCell<Vec<u8>>>;

#[derive(Clone)]
enum Fd {
    Stdin { pos: usize },
    Stdout,
    Stderr,
    File { data: Data, pos: usize, memfd: bool, dev: u64, ino: u64 },
}

struct SharedMap {
    data: Data,
    at: u64,
    len: u64,
    offset: u64,
}

/// What a system call asks of the driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Continue,
    Exit(i32),
    Unsupported(String),
}

pub struct Kernel {
    pub mapped: Ranges,
    pub image_lo: u64,
    pub image_hi: u64,
    pub entry: u64,
    brk_start: u64,
    brk: u64,
    mmap_next: u64,
    fds: Vec<Option<Fd>>,
    shared: Vec<SharedMap>,
    stdin: Rc<Vec<u8>>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    exe_path: String,
    rng: u64,
    next_ino: u64,
    /// System calls answered ENOSYS, by number, with how often.
    pub enosys: BTreeMap<u64, u64>,
    pub syscalls: u64,
    pub trace: bool,
}

fn cstr(s: &str) -> Vec<u8> {
    let mut v = s.as_bytes().to_vec();
    v.push(0);
    v
}

impl Kernel {
    /// Loads `bytes` into `guest`, builds the SysV stack and points `RIP` at
    /// the entry.
    pub fn boot(guest: &mut dyn Guest, path: &str, bytes: &[u8], argv: &[String], stdin: Vec<u8>) -> Result<Self, String> {
        let image = elf::parse(bytes)?;
        if image.interp {
            return Err("dynamically linked (PT_INTERP): the baselines load static images only".into());
        }
        let mut mapped = Ranges::default();
        let (lo, hi) = elf::load(&image, bytes, guest, &mut mapped);
        let stack_lo = STACK_TOP - STACK_SIZE;
        assert!(guest.map(stack_lo, STACK_SIZE), "cannot map the stack");
        mapped.insert(stack_lo, STACK_TOP);

        let k = Kernel {
            mapped,
            image_lo: lo,
            image_hi: hi,
            entry: image.entry,
            brk_start: hi,
            brk: hi,
            mmap_next: MMAP_BASE,
            fds: vec![Some(Fd::Stdin { pos: 0 }), Some(Fd::Stdout), Some(Fd::Stderr)],
            shared: Vec::new(),
            stdin: Rc::new(stdin),
            stdout: Vec::new(),
            stderr: Vec::new(),
            exe_path: std::fs::canonicalize(path).map(|p| p.display().to_string()).unwrap_or(path.to_string()),
            rng: 0x9e37_79b9_7f4a_7c15,
            next_ino: 1 << 40,
            enosys: BTreeMap::new(),
            syscalls: 0,
            trace: std::env::var_os("UNPACKBENCH_STRACE").is_some(),
        };

        // The stack, as userland/src/stack.rs builds it.
        let mut sp = STACK_TOP;
        let mut push = |g: &mut dyn Guest, b: &[u8]| {
            sp -= b.len() as u64;
            g.write(sp, b);
            sp
        };
        let execfn = push(guest, &cstr(argv.first().map(String::as_str).unwrap_or("")));
        let platform = push(guest, b"x86_64\0");
        let mut args: Vec<u64> = argv.iter().rev().map(|s| push(guest, &cstr(s))).collect();
        args.reverse();
        let random = push(guest, &0x9e37_79b9_7f4a_7c15_u128.to_le_bytes());
        let auxv: [(u64, u64); 19] = [
            (3, image.phdr),
            (4, image.phentsize as u64),
            (5, image.phnum as u64),
            (6, PAGE),
            (7, 0),
            (8, 0),
            (9, image.entry),
            (11, 1000),
            (12, 1000),
            (13, 1000),
            (14, 1000),
            (15, platform),
            (16, 0),
            (26, 0),
            (17, 100),
            (23, 0),
            (25, random),
            (31, execfn),
            (0, 0),
        ];
        let words = 1 + args.len() + 1 + 1 + auxv.len() * 2;
        let mut sp = sp & !0xf;
        if (words * 8) % 16 != 0 {
            sp -= 8;
        }
        sp -= (words * 8) as u64;
        let mut vec: Vec<u64> = vec![args.len() as u64];
        vec.extend(&args);
        vec.push(0);
        vec.push(0); // empty environment, as the unpack example runs with
        for (a, b) in auxv {
            vec.push(a);
            vec.push(b);
        }
        let raw: Vec<u8> = vec.iter().flat_map(|w| w.to_le_bytes()).collect();
        guest.write(sp, &raw);
        guest.set_reg(Reg::Rsp, sp);
        guest.set_reg(Reg::Rip, image.entry);
        Ok(k)
    }

    fn read_mem(g: &mut dyn Guest, addr: u64, len: usize) -> Result<Vec<u8>, u64> {
        let mut v = vec![0; len];
        if len > 0 && !g.read(addr, &mut v) {
            return Err(EFAULT);
        }
        Ok(v)
    }

    fn write_mem(g: &mut dyn Guest, addr: u64, data: &[u8]) -> Result<(), u64> {
        if !data.is_empty() && !g.write(addr, data) {
            return Err(EFAULT);
        }
        Ok(())
    }

    fn read_path(g: &mut dyn Guest, mut addr: u64) -> Result<String, u64> {
        let mut out = Vec::new();
        loop {
            let mut b = [0u8; 1];
            if !g.read(addr, &mut b) {
                return Err(EFAULT);
            }
            if b[0] == 0 {
                break;
            }
            out.push(b[0]);
            addr += 1;
            if out.len() > 4096 {
                return Err(EFAULT);
            }
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }

    /// Services the `syscall` the guest is stopped at, and writes `RAX`.
    pub fn syscall(&mut self, g: &mut dyn Guest) -> Action {
        self.syscalls += 1;
        let nr = g.reg(Reg::Rax);
        let a = [
            g.reg(Reg::Rdi),
            g.reg(Reg::Rsi),
            g.reg(Reg::Rdx),
            g.reg(Reg::R10),
            g.reg(Reg::R8),
            g.reg(Reg::R9),
        ];
        let result = match nr {
            // The process model the baselines lack.
            56 | 57 | 58 | 59 | 61 | 101 | 247 | 322 | 435 => {
                return Action::Unsupported(format!("syscall {nr} (fork/exec/ptrace family): no process model"));
            }
            60 | 231 => {
                if self.trace {
                    eprintln!("[{nr}] exit({})", a[0] as i32);
                }
                return Action::Exit((a[0] & 0xff) as i32);
            }
            _ => self.dispatch(g, nr, a),
        };
        if self.trace {
            eprintln!("[{nr}] ({:#x}, {:#x}, {:#x}, {:#x}) = {result:x?}", a[0], a[1], a[2], a[3]);
        }
        let ret = match result {
            Ok(v) => v,
            Err(e) => (e as i64).wrapping_neg() as u64,
        };
        g.set_reg(Reg::Rax, ret);
        Action::Continue
    }

    fn dispatch(&mut self, g: &mut dyn Guest, nr: u64, a: [u64; 6]) -> Result<u64, u64> {
        match nr {
            0 => self.sys_read(g, a[0], a[1], a[2], None),
            1 => self.sys_write(g, a[0], a[1], a[2]),
            2 => self.sys_open(g, a[0], a[1]),
            3 => {
                *self.fd_mut(a[0])? = None;
                Ok(0)
            }
            4 | 6 => {
                let p = Self::read_path(g, a[0])?;
                self.stat_path(g, &p, a[1])
            }
            5 => self.sys_fstat(g, a[0], a[1]),
            8 => self.sys_lseek(a[0], a[1] as i64, a[2]),
            9 => self.sys_mmap(g, a[0], a[1], a[3], a[4] as i32, a[5]),
            10 | 28 | 24 => Ok(0),
            11 => self.sys_munmap(g, a[0], a[1]),
            12 => Ok(self.sys_brk(g, a[0])),
            13 => {
                if a[2] != 0 {
                    Self::write_mem(g, a[2], &[0; 32])?;
                }
                Ok(0)
            }
            14 => {
                if a[2] != 0 {
                    Self::write_mem(g, a[2], &[0; 8])?;
                }
                Ok(0)
            }
            16 => Err(ENOTTY),
            17 => self.sys_read(g, a[0], a[1], a[2], Some(a[3])),
            19 => {
                let mut total = 0;
                for i in 0..a[2] {
                    let iov = Self::read_mem(g, a[1] + 16 * i, 16)?;
                    let base = u64::from_le_bytes(iov[..8].try_into().unwrap());
                    let len = u64::from_le_bytes(iov[8..].try_into().unwrap());
                    let n = self.sys_read(g, a[0], base, len, None)?;
                    total += n;
                    if n < len {
                        break;
                    }
                }
                Ok(total)
            }
            20 => {
                let mut total = 0;
                for i in 0..a[2] {
                    let iov = Self::read_mem(g, a[1] + 16 * i, 16)?;
                    let base = u64::from_le_bytes(iov[..8].try_into().unwrap());
                    let len = u64::from_le_bytes(iov[8..].try_into().unwrap());
                    total += self.sys_write(g, a[0], base, len)?;
                }
                Ok(total)
            }
            21 => {
                let p = Self::read_path(g, a[0])?;
                if std::path::Path::new(&p).exists() { Ok(0) } else { Err(ENOENT) }
            }
            25 => Err(ENOMEM),
            26 => {
                self.sync_shared(g, a[0], a[1]);
                Ok(0)
            }
            32 => {
                let fd = self.fd_mut(a[0])?.clone().expect("open");
                Ok(self.alloc_fd(fd))
            }
            33 | 292 => {
                let fd = self.fd_mut(a[0])?.clone().expect("open");
                let new = a[1] as usize;
                if new >= 1024 {
                    return Err(EBADF);
                }
                if self.fds.len() <= new {
                    self.fds.resize(new + 1, None);
                }
                self.fds[new] = Some(fd);
                Ok(new as u64)
            }
            39 | 186 => Ok(4242),
            110 => Ok(4241),
            63 => {
                let mut out = [0u8; 65 * 6];
                for (i, f) in ["Linux", "unpackbench", "6.1.0", "#1 SMP", "x86_64", "(none)"].iter().enumerate() {
                    out[i * 65..i * 65 + f.len()].copy_from_slice(f.as_bytes());
                }
                Self::write_mem(g, a[0], &out)?;
                Ok(0)
            }
            72 => Ok(0),
            77 => {
                if let Some(Fd::File { data, .. }) = self.fd_mut(a[0])? {
                    data.borrow_mut().resize(a[1] as usize, 0);
                    Ok(0)
                } else {
                    Err(EINVAL)
                }
            }
            79 => {
                let cwd = std::env::current_dir().map(|p| p.display().to_string()).unwrap_or("/".into());
                let c = cstr(&cwd);
                if c.len() as u64 > a[1] {
                    return Err(ERANGE);
                }
                Self::write_mem(g, a[0], &c)?;
                Ok(c.len() as u64)
            }
            89 => self.sys_readlink(g, a[0], a[1], a[2]),
            267 => self.sys_readlink(g, a[1], a[2], a[3]),
            96 => {
                if a[0] != 0 {
                    Self::write_mem(g, a[0], &[0; 16])?;
                }
                Ok(0)
            }
            102 | 104 | 107 | 108 => Ok(1000),
            131 => Ok(0),
            158 => match a[0] {
                0x1002 => {
                    g.set_reg(Reg::FsBase, a[1]);
                    Ok(0)
                }
                0x1001 => {
                    g.set_reg(Reg::GsBase, a[1]);
                    Ok(0)
                }
                0x1003 => {
                    let v = g.reg(Reg::FsBase);
                    Self::write_mem(g, a[1], &v.to_le_bytes())?;
                    Ok(0)
                }
                _ => Err(EINVAL),
            },
            201 => Ok(0),
            202 => Ok(0),
            218 => Ok(4242),
            228 => {
                Self::write_mem(g, a[1], &[0; 16])?;
                Ok(0)
            }
            257 => self.sys_open(g, a[1], a[2]),
            262 => {
                let p = Self::read_path(g, a[1])?;
                if p.is_empty() && a[3] & 0x1000 != 0 {
                    return self.sys_fstat(g, a[0], a[2]);
                }
                self.stat_path(g, &p, a[2])
            }
            273 => Ok(0),
            302 => {
                if a[3] != 0 {
                    let (cur, max) = match a[1] {
                        3 => (STACK_SIZE, u64::MAX),
                        7 => (1024, 1024),
                        _ => (u64::MAX, u64::MAX),
                    };
                    let mut b = [0u8; 16];
                    b[..8].copy_from_slice(&cur.to_le_bytes());
                    b[8..].copy_from_slice(&max.to_le_bytes());
                    Self::write_mem(g, a[3], &b)?;
                }
                Ok(0)
            }
            318 => {
                let len = (a[1] as usize).min(1 << 20);
                let mut out = Vec::with_capacity(len + 8);
                while out.len() < len {
                    self.rng ^= self.rng >> 12;
                    self.rng ^= self.rng << 25;
                    self.rng ^= self.rng >> 27;
                    out.extend_from_slice(&self.rng.wrapping_mul(0x2545_f491_4f6c_dd1d).to_le_bytes());
                }
                Self::write_mem(g, a[0], &out[..len])?;
                Ok(len as u64)
            }
            319 => {
                let name = Self::read_path(g, a[0])?;
                let _ = name;
                self.next_ino += 1;
                Ok(self.alloc_fd(Fd::File {
                    data: Rc::new(RefCell::new(Vec::new())),
                    pos: 0,
                    memfd: true,
                    dev: 0x1_0000,
                    ino: self.next_ino,
                }))
            }
            _ => {
                *self.enosys.entry(nr).or_default() += 1;
                Err(ENOSYS)
            }
        }
    }

    fn alloc_fd(&mut self, fd: Fd) -> u64 {
        match self.fds.iter().position(Option::is_none) {
            Some(i) => {
                self.fds[i] = Some(fd);
                i as u64
            }
            None => {
                self.fds.push(Some(fd));
                (self.fds.len() - 1) as u64
            }
        }
    }

    fn fd_mut(&mut self, fd: u64) -> Result<&mut Option<Fd>, u64> {
        match self.fds.get_mut(fd as usize) {
            Some(slot @ Some(_)) => Ok(slot),
            _ => Err(EBADF),
        }
    }

    fn sys_read(&mut self, g: &mut dyn Guest, fd: u64, buf: u64, len: u64, at: Option<u64>) -> Result<u64, u64> {
        let Kernel { fds, stdin, .. } = self;
        let slice = |src: &[u8], pos: &mut usize| {
            let start = at.map(|o| o as usize).unwrap_or(*pos).min(src.len());
            let end = start.saturating_add(len as usize).min(src.len());
            if at.is_none() {
                *pos = end;
            }
            src[start..end].to_vec()
        };
        let chunk = match fds.get_mut(fd as usize) {
            Some(Some(Fd::Stdin { pos })) => slice(stdin, pos),
            Some(Some(Fd::File { data, pos, .. })) => slice(&data.borrow(), pos),
            _ => return Err(EBADF),
        };
        Self::write_mem(g, buf, &chunk)?;
        Ok(chunk.len() as u64)
    }

    fn sys_write(&mut self, g: &mut dyn Guest, fd: u64, buf: u64, len: u64) -> Result<u64, u64> {
        let bytes = Self::read_mem(g, buf, len as usize)?;
        match self.fd_mut(fd)? {
            Some(Fd::Stdout) => self.stdout.extend_from_slice(&bytes),
            Some(Fd::Stderr) => self.stderr.extend_from_slice(&bytes),
            Some(Fd::File { data, pos, memfd: true, .. }) => {
                let mut d = data.borrow_mut();
                if d.len() < *pos + bytes.len() {
                    d.resize(*pos + bytes.len(), 0);
                }
                d[*pos..*pos + bytes.len()].copy_from_slice(&bytes);
                *pos += bytes.len();
            }
            _ => return Err(EBADF),
        }
        Ok(len)
    }

    fn host_path(&self, p: &str) -> String {
        if p == "/proc/self/exe" { self.exe_path.clone() } else { p.to_string() }
    }

    fn sys_open(&mut self, g: &mut dyn Guest, path: u64, flags: u64) -> Result<u64, u64> {
        let p = Self::read_path(g, path)?;
        if flags & 3 != 0 {
            return Err(EACCES);
        }
        let host = self.host_path(&p);
        let meta = std::fs::metadata(&host).map_err(|_| ENOENT)?;
        if meta.is_dir() {
            return Err(EACCES);
        }
        let data = std::fs::read(&host).map_err(|_| EACCES)?;
        use std::os::unix::fs::MetadataExt;
        Ok(self.alloc_fd(Fd::File { data: Rc::new(RefCell::new(data)), pos: 0, memfd: false, dev: meta.dev(), ino: meta.ino() }))
    }

    fn sys_lseek(&mut self, fd: u64, off: i64, whence: u64) -> Result<u64, u64> {
        match self.fd_mut(fd)? {
            Some(Fd::File { data, pos, .. }) => {
                let base = match whence {
                    0 => 0,
                    1 => *pos as i64,
                    2 => data.borrow().len() as i64,
                    _ => return Err(EINVAL),
                };
                let n = base + off;
                if n < 0 {
                    return Err(EINVAL);
                }
                *pos = n as usize;
                Ok(n as u64)
            }
            _ => Err(ESPIPE),
        }
    }

    fn write_stat(g: &mut dyn Guest, buf: u64, mode: u32, size: u64, dev: u64, ino: u64) -> Result<u64, u64> {
        let mut st = [0u8; 144];
        // ld.so tells files apart by (st_dev, st_ino).
        st[0..8].copy_from_slice(&dev.to_le_bytes());
        st[8..16].copy_from_slice(&ino.to_le_bytes());
        st[16..24].copy_from_slice(&1u64.to_le_bytes()); // st_nlink
        st[24..28].copy_from_slice(&mode.to_le_bytes());
        st[48..56].copy_from_slice(&size.to_le_bytes());
        st[56..64].copy_from_slice(&4096u64.to_le_bytes());
        st[64..72].copy_from_slice(&size.div_ceil(512).to_le_bytes());
        Self::write_mem(g, buf, &st)?;
        Ok(0)
    }

    fn sys_fstat(&mut self, g: &mut dyn Guest, fd: u64, buf: u64) -> Result<u64, u64> {
        let (mode, size, dev, ino) = match self.fd_mut(fd)? {
            Some(Fd::File { data, dev, ino, .. }) => (0o100755, data.borrow().len() as u64, *dev, *ino),
            _ => (0o20620, 0, 0x2_0000, fd + 1),
        };
        Self::write_stat(g, buf, mode, size, dev, ino)
    }

    fn stat_path(&mut self, g: &mut dyn Guest, p: &str, buf: u64) -> Result<u64, u64> {
        let meta = std::fs::metadata(self.host_path(p)).map_err(|_| ENOENT)?;
        let mode = if meta.is_dir() { 0o40755 } else { 0o100755 };
        use std::os::unix::fs::MetadataExt;
        Self::write_stat(g, buf, mode, meta.len(), meta.dev(), meta.ino())
    }

    fn sys_readlink(&mut self, g: &mut dyn Guest, path: u64, buf: u64, size: u64) -> Result<u64, u64> {
        let p = Self::read_path(g, path)?;
        let target = if p == "/proc/self/exe" {
            self.exe_path.clone()
        } else {
            std::fs::read_link(&p).map_err(|_| EINVAL)?.display().to_string()
        };
        let n = target.len().min(size as usize);
        Self::write_mem(g, buf, &target.as_bytes()[..n])?;
        Ok(n as u64)
    }

    fn sys_brk(&mut self, g: &mut dyn Guest, want: u64) -> u64 {
        if want < self.brk_start {
            return self.brk;
        }
        let (old, new) = (page_up(self.brk), page_up(want));
        if new > old {
            if !self.mapped.is_free(old, new) || !g.map(old, new - old) {
                return self.brk;
            }
            self.mapped.insert(old, new);
        } else if new < old {
            g.unmap(new, old - new);
            self.mapped.remove(new, old);
        }
        self.brk = want;
        want
    }

    fn find_free(&mut self, len: u64) -> Option<u64> {
        let mut start = self.mmap_next;
        loop {
            if start + len >= STACK_TOP - STACK_SIZE {
                return None;
            }
            match self.mapped.overlaps(start, start + len).last() {
                None => {
                    self.mmap_next = start + len;
                    return Some(start);
                }
                Some(&(_, e)) => start = page_up(e),
            }
        }
    }

    fn unmap_range(&mut self, g: &mut dyn Guest, at: u64, len: u64) {
        for (s, e) in self.mapped.overlaps(at, at + len) {
            g.unmap(s, e - s);
        }
        self.mapped.remove(at, at + len);
    }

    fn sys_mmap(&mut self, g: &mut dyn Guest, addr: u64, len: u64, flags: u64, fd: i32, offset: u64) -> Result<u64, u64> {
        if len == 0 || addr % PAGE != 0 || offset % PAGE != 0 {
            return Err(EINVAL);
        }
        let len = page_up(len);
        let at = if flags & (MAP_FIXED | MAP_FIXED_NOREPLACE) != 0 {
            if flags & MAP_FIXED_NOREPLACE != 0 && !self.mapped.is_free(addr, addr + len) {
                return Err(EEXIST);
            }
            addr
        } else if addr != 0 && self.mapped.is_free(addr, addr + len) {
            addr
        } else {
            self.find_free(len).ok_or(ENOMEM)?
        };
        let file = if flags & MAP_ANONYMOUS == 0 {
            match self.fds.get(fd as usize) {
                Some(Some(Fd::File { data, .. })) => Some(data.clone()),
                _ => return Err(EBADF),
            }
        } else {
            None
        };
        if flags & MAP_FIXED != 0 {
            self.sync_shared(g, at, len);
            self.shared.retain(|m| m.at >= at + len || m.at + m.len <= at);
            self.unmap_range(g, at, len);
        }
        if file.is_some() {
            // Another mapping of the file may hold bytes this one must see.
            self.sync_shared(g, 0, u64::MAX);
        }
        if !g.map(at, len) {
            return Err(ENOMEM);
        }
        self.mapped.insert(at, at + len);
        if let Some(data) = file {
            {
                let d = data.borrow();
                let from = (offset as usize).min(d.len());
                let to = (offset as usize + len as usize).min(d.len());
                g.write(at, &d[from..to]);
            }
            if flags & MAP_SHARED != 0 {
                self.shared.push(SharedMap { data, at, len, offset });
            }
        }
        Ok(at)
    }

    fn sync_shared(&mut self, g: &mut dyn Guest, addr: u64, len: u64) {
        for m in &self.shared {
            let lo = m.at.max(addr);
            let hi = (m.at + m.len).min(addr.saturating_add(len));
            if lo >= hi {
                continue;
            }
            let mut bytes = vec![0; (hi - lo) as usize];
            if g.read(lo, &mut bytes) {
                let mut d = m.data.borrow_mut();
                let off = (m.offset + (lo - m.at)) as usize;
                // As userland's `write_all_at`: the bytes past the end of
                // file in the last page reach the file too, which is how
                // UPX's exit trampoline survives the remap.
                if d.len() < off + bytes.len() {
                    d.resize(off + bytes.len(), 0);
                }
                d[off..off + bytes.len()].copy_from_slice(&bytes);
            }
        }
    }

    fn sys_munmap(&mut self, g: &mut dyn Guest, addr: u64, len: u64) -> Result<u64, u64> {
        if addr % PAGE != 0 || len == 0 {
            return Err(EINVAL);
        }
        let len = page_up(len);
        self.sync_shared(g, addr, len);
        self.shared.retain(|m| m.at >= addr + len || m.at + m.len <= addr);
        self.unmap_range(g, addr, len);
        Ok(0)
    }
}
