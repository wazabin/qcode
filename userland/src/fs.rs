//! The file-descriptor table, and the host filesystem behind it.
//!
//! Descriptors 0–2 are wired to the host's stdio, or — for a harness — to
//! in-memory buffers. Every other descriptor is a host file or directory,
//! opened under an optional sandbox root: with a root, guest path `/etc/x`
//! means host `<root>/etc/x`, and `..` cannot climb above the root.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{DirEntryExt, FileTypeExt, MetadataExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};

use crate::errno::{self, EBADF, EINVAL, EISDIR, EMFILE, ENOENT, ENOTDIR, ENOTTY, ESPIPE, Errno};

pub const O_ACCMODE: u32 = 3;
pub const O_RDONLY: u32 = 0;
pub const O_WRONLY: u32 = 1;
pub const O_RDWR: u32 = 2;
pub const O_CREAT: u32 = 0o100;
pub const O_EXCL: u32 = 0o200;
pub const O_TRUNC: u32 = 0o1000;
pub const O_APPEND: u32 = 0o2000;
pub const O_DIRECTORY: u32 = 0o200000;
pub const O_CLOEXEC: u32 = 0o2000000;
pub const AT_FDCWD: i32 = -100;

pub const S_IFMT: u32 = 0o170000;
pub const S_IFDIR: u32 = 0o040000;
pub const S_IFCHR: u32 = 0o020000;
pub const S_IFREG: u32 = 0o100000;
pub const S_IFIFO: u32 = 0o010000;
pub const S_IFLNK: u32 = 0o120000;

/// Highest descriptor number the table hands out.
const MAX_FDS: usize = 1024;

/// Where descriptors 0–2 go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stdio {
    /// The host's own stdin/stdout/stderr.
    Host,
    /// In-memory buffers, readable afterwards through [`Files::stdout`] and
    /// [`Files::stderr`]; stdin reads from [`Files::set_stdin`].
    Captured,
}

/// A directory entry, read ahead at `open` so `getdents64` is a slice walk.
#[derive(Debug, Clone)]
pub struct Dirent {
    pub ino: u64,
    pub kind: u8,
    pub name: Vec<u8>,
}

#[derive(Debug)]
pub enum FdKind {
    Stdin,
    Stdout,
    Stderr,
    File(File),
    Dir { entries: Vec<Dirent>, pos: usize },
}

#[derive(Debug)]
pub struct Fd {
    pub kind: FdKind,
    /// The guest-visible path this descriptor was opened with.
    pub path: String,
    pub cloexec: bool,
    pub append: bool,
}

/// Linux `struct stat` for x86-64 (144 bytes).
#[derive(Debug, Clone, Copy, Default)]
pub struct Stat {
    pub dev: u64,
    pub ino: u64,
    pub nlink: u64,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub rdev: u64,
    pub size: i64,
    pub blksize: i64,
    pub blocks: i64,
    pub atime: (i64, i64),
    pub mtime: (i64, i64),
    pub ctime: (i64, i64),
}

impl Stat {
    pub fn to_bytes(&self) -> [u8; 144] {
        let mut b = [0u8; 144];
        let mut put = |off: usize, v: u64| b[off..off + 8].copy_from_slice(&v.to_le_bytes());
        put(0, self.dev);
        put(8, self.ino);
        put(16, self.nlink);
        put(24, u64::from(self.mode) | (u64::from(self.uid) << 32));
        put(32, u64::from(self.gid));
        put(40, self.rdev);
        put(48, self.size as u64);
        put(56, self.blksize as u64);
        put(64, self.blocks as u64);
        put(72, self.atime.0 as u64);
        put(80, self.atime.1 as u64);
        put(88, self.mtime.0 as u64);
        put(96, self.mtime.1 as u64);
        put(104, self.ctime.0 as u64);
        put(112, self.ctime.1 as u64);
        b
    }

    fn from_metadata(m: &std::fs::Metadata) -> Self {
        Self {
            dev: m.dev(),
            ino: m.ino(),
            nlink: m.nlink(),
            mode: m.mode(),
            uid: m.uid(),
            gid: m.gid(),
            rdev: m.rdev(),
            size: m.size() as i64,
            blksize: m.blksize() as i64,
            blocks: m.blocks() as i64,
            atime: (m.atime(), m.atime_nsec()),
            mtime: (m.mtime(), m.mtime_nsec()),
            ctime: (m.ctime(), m.ctime_nsec()),
        }
    }

    /// What a terminal-ish character device looks like.
    fn tty() -> Self {
        Self {
            dev: 0x16,
            ino: 3,
            nlink: 1,
            mode: S_IFCHR | 0o620,
            uid: 1000,
            gid: 5,
            rdev: 0x8801,
            blksize: 1024,
            ..Default::default()
        }
    }
}

pub struct Files {
    fds: Vec<Option<Fd>>,
    root: Option<PathBuf>,
    cwd: String,
    stdio: Stdio,
    stdin: Vec<u8>,
    stdin_pos: usize,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    /// What `/proc/self/exe` resolves to.
    pub exe_path: String,
}

impl Files {
    pub fn new(stdio: Stdio, root: Option<PathBuf>, exe_path: String) -> Self {
        let mut fds: Vec<Option<Fd>> = Vec::with_capacity(8);
        for (kind, path) in [
            (FdKind::Stdin, "/dev/stdin"),
            (FdKind::Stdout, "/dev/stdout"),
            (FdKind::Stderr, "/dev/stderr"),
        ] {
            fds.push(Some(Fd {
                kind,
                path: path.to_owned(),
                cloexec: false,
                append: false,
            }));
        }
        Self {
            fds,
            root,
            cwd: "/".to_owned(),
            stdio,
            stdin: Vec::new(),
            stdin_pos: 0,
            stdout: Vec::new(),
            stderr: Vec::new(),
            exe_path,
        }
    }

    pub fn is_captured(&self) -> bool {
        self.stdio == Stdio::Captured
    }

    /// Bytes the guest wrote to descriptor 1, when captured.
    pub fn stdout(&self) -> &[u8] {
        &self.stdout
    }

    /// Bytes the guest wrote to descriptor 2, when captured.
    pub fn stderr(&self) -> &[u8] {
        &self.stderr
    }

    /// What captured stdin will hand the guest.
    pub fn set_stdin(&mut self, bytes: Vec<u8>) {
        self.stdin = bytes;
        self.stdin_pos = 0;
    }

    pub fn cwd(&self) -> &str {
        &self.cwd
    }

    pub fn get(&self, fd: i32) -> Result<&Fd, Errno> {
        usize::try_from(fd)
            .ok()
            .and_then(|i| self.fds.get(i))
            .and_then(Option::as_ref)
            .ok_or(EBADF)
    }

    pub fn get_mut(&mut self, fd: i32) -> Result<&mut Fd, Errno> {
        usize::try_from(fd)
            .ok()
            .and_then(|i| self.fds.get_mut(i))
            .and_then(Option::as_mut)
            .ok_or(EBADF)
    }

    /// Guest path (absolute, or relative to `dirfd`) to the host path it names.
    pub fn resolve(&self, dirfd: i32, path: &str) -> Result<PathBuf, Errno> {
        let guest = self.guest_absolute(dirfd, path)?;
        match &self.root {
            None => Ok(PathBuf::from(guest)),
            Some(root) => {
                // Devices the sandbox still allows through, by name.
                if guest == "/dev/null" {
                    return Ok(PathBuf::from("/dev/null"));
                }
                let mut host = root.clone();
                for component in Path::new(&guest).components() {
                    match component {
                        Component::Normal(c) => host.push(c),
                        Component::ParentDir if host != *root => {
                            host.pop();
                        }
                        _ => {}
                    }
                }
                Ok(host)
            }
        }
    }

    /// The guest's own idea of the absolute path.
    pub fn guest_absolute(&self, dirfd: i32, path: &str) -> Result<String, Errno> {
        if path.starts_with('/') {
            return Ok(normalize(path));
        }
        let dir = if dirfd == AT_FDCWD {
            self.cwd.clone()
        } else {
            let fd = self.get(dirfd)?;
            if !matches!(fd.kind, FdKind::Dir { .. }) {
                return Err(ENOTDIR);
            }
            fd.path.clone()
        };
        Ok(normalize(&format!("{dir}/{path}")))
    }

    fn install(&mut self, fd: Fd, min: usize) -> Result<i32, Errno> {
        let slot = (min..self.fds.len()).find(|&i| self.fds[i].is_none());
        let slot = match slot {
            Some(i) => i,
            None => {
                let i = self.fds.len().max(min);
                if i >= MAX_FDS {
                    return Err(EMFILE);
                }
                self.fds.resize_with(i + 1, || None);
                i
            }
        };
        self.fds[slot] = Some(fd);
        Ok(slot as i32)
    }

    pub fn open(&mut self, dirfd: i32, path: &str, flags: u32, mode: u32) -> Result<i32, Errno> {
        let guest = self.guest_absolute(dirfd, path)?;
        let host = if guest == "/proc/self/exe" {
            PathBuf::from(&self.exe_path)
        } else {
            self.resolve(dirfd, path)?
        };
        let access = flags & O_ACCMODE;
        let meta = std::fs::metadata(&host).ok();
        if let Some(m) = &meta
            && m.is_dir()
        {
            if access != O_RDONLY {
                return Err(EISDIR);
            }
            let mut entries: Vec<Dirent> = vec![
                Dirent {
                    ino: m.ino(),
                    kind: 4,
                    name: b".".to_vec(),
                },
                Dirent {
                    ino: m.ino(),
                    kind: 4,
                    name: b"..".to_vec(),
                },
            ];
            for entry in std::fs::read_dir(&host)
                .map_err(|e| errno::from_io(&e))?
                .flatten()
            {
                let kind = entry.file_type().map(dirent_type).unwrap_or(0);
                entries.push(Dirent {
                    ino: entry.ino(),
                    kind,
                    name: entry.file_name().as_encoded_bytes().to_vec(),
                });
            }
            return self.install(
                Fd {
                    kind: FdKind::Dir { entries, pos: 0 },
                    path: guest,
                    cloexec: flags & O_CLOEXEC != 0,
                    append: false,
                },
                0,
            );
        }
        if flags & O_DIRECTORY != 0 {
            return Err(if meta.is_some() { ENOTDIR } else { ENOENT });
        }
        let mut options = OpenOptions::new();
        options
            .read(access == O_RDONLY || access == O_RDWR)
            .write(access == O_WRONLY || access == O_RDWR)
            .append(flags & O_APPEND != 0)
            .create(flags & O_CREAT != 0)
            .create_new(flags & O_CREAT != 0 && flags & O_EXCL != 0)
            .truncate(flags & O_TRUNC != 0)
            .mode(mode);
        let file = options.open(&host).map_err(|e| errno::from_io(&e))?;
        self.install(
            Fd {
                kind: FdKind::File(file),
                path: guest,
                cloexec: flags & O_CLOEXEC != 0,
                append: flags & O_APPEND != 0,
            },
            0,
        )
    }

    pub fn close(&mut self, fd: i32) -> Result<(), Errno> {
        let slot = usize::try_from(fd).map_err(|_| EBADF)?;
        match self.fds.get_mut(slot) {
            Some(entry @ Some(_)) => {
                *entry = None;
                Ok(())
            }
            _ => Err(EBADF),
        }
    }

    pub fn read(&mut self, fd: i32, buf: &mut [u8]) -> Result<usize, Errno> {
        let stdio = self.stdio;
        match &mut self.get_mut(fd)?.kind {
            FdKind::Stdin => match stdio {
                Stdio::Host => std::io::stdin().read(buf).map_err(|e| errno::from_io(&e)),
                Stdio::Captured => {
                    let rest = &self.stdin[self.stdin_pos..];
                    let n = rest.len().min(buf.len());
                    buf[..n].copy_from_slice(&rest[..n]);
                    self.stdin_pos += n;
                    Ok(n)
                }
            },
            FdKind::Stdout | FdKind::Stderr => Err(EBADF),
            FdKind::File(file) => file.read(buf).map_err(|e| errno::from_io(&e)),
            FdKind::Dir { .. } => Err(EISDIR),
        }
    }

    pub fn pread(&mut self, fd: i32, buf: &mut [u8], offset: u64) -> Result<usize, Errno> {
        match &mut self.get_mut(fd)?.kind {
            FdKind::File(file) => {
                use std::os::unix::fs::FileExt;
                file.read_at(buf, offset).map_err(|e| errno::from_io(&e))
            }
            FdKind::Dir { .. } => Err(EISDIR),
            _ => Err(ESPIPE),
        }
    }

    pub fn write(&mut self, fd: i32, bytes: &[u8]) -> Result<usize, Errno> {
        let stdio = self.stdio;
        match &mut self.get_mut(fd)?.kind {
            FdKind::Stdin => Err(EBADF),
            FdKind::Stdout => match stdio {
                Stdio::Host => {
                    let mut out = std::io::stdout().lock();
                    out.write_all(bytes)
                        .and_then(|()| out.flush())
                        .map_err(|e| errno::from_io(&e))?;
                    Ok(bytes.len())
                }
                Stdio::Captured => {
                    self.stdout.extend_from_slice(bytes);
                    Ok(bytes.len())
                }
            },
            FdKind::Stderr => match stdio {
                Stdio::Host => {
                    std::io::stderr()
                        .write_all(bytes)
                        .map_err(|e| errno::from_io(&e))?;
                    Ok(bytes.len())
                }
                Stdio::Captured => {
                    self.stderr.extend_from_slice(bytes);
                    Ok(bytes.len())
                }
            },
            FdKind::File(file) => file.write(bytes).map_err(|e| errno::from_io(&e)),
            FdKind::Dir { .. } => Err(EBADF),
        }
    }

    pub fn pwrite(&mut self, fd: i32, bytes: &[u8], offset: u64) -> Result<usize, Errno> {
        match &mut self.get_mut(fd)?.kind {
            FdKind::File(file) => {
                use std::os::unix::fs::FileExt;
                file.write_at(bytes, offset).map_err(|e| errno::from_io(&e))
            }
            _ => Err(ESPIPE),
        }
    }

    pub fn lseek(&mut self, fd: i32, offset: i64, whence: u32) -> Result<u64, Errno> {
        match &mut self.get_mut(fd)?.kind {
            FdKind::File(file) => {
                let pos = match whence {
                    0 => SeekFrom::Start(u64::try_from(offset).map_err(|_| EINVAL)?),
                    1 => SeekFrom::Current(offset),
                    2 => SeekFrom::End(offset),
                    _ => return Err(EINVAL),
                };
                file.seek(pos).map_err(|e| errno::from_io(&e))
            }
            FdKind::Dir { entries, pos } => {
                if whence != 0 || offset < 0 {
                    return Err(EINVAL);
                }
                *pos = (offset as usize).min(entries.len());
                Ok(*pos as u64)
            }
            _ => Err(ESPIPE),
        }
    }

    pub fn fstat(&self, fd: i32) -> Result<Stat, Errno> {
        let entry = self.get(fd)?;
        match &entry.kind {
            FdKind::Stdin | FdKind::Stdout | FdKind::Stderr => Ok(Stat::tty()),
            FdKind::File(file) => file
                .metadata()
                .map(|m| Stat::from_metadata(&m))
                .map_err(|e| errno::from_io(&e)),
            FdKind::Dir { .. } => {
                let host = self.resolve(AT_FDCWD, &entry.path)?;
                std::fs::metadata(host)
                    .map(|m| Stat::from_metadata(&m))
                    .map_err(|e| errno::from_io(&e))
            }
        }
    }

    pub fn stat(&self, dirfd: i32, path: &str, follow: bool) -> Result<Stat, Errno> {
        let guest = self.guest_absolute(dirfd, path)?;
        if guest == "/proc/self/exe" {
            return std::fs::metadata(&self.exe_path)
                .map(|m| Stat::from_metadata(&m))
                .map_err(|e| errno::from_io(&e));
        }
        let host = self.resolve(dirfd, path)?;
        let meta = if follow {
            std::fs::metadata(&host)
        } else {
            std::fs::symlink_metadata(&host)
        };
        meta.map(|m| Stat::from_metadata(&m))
            .map_err(|e| errno::from_io(&e))
    }

    pub fn access(&self, dirfd: i32, path: &str) -> Result<(), Errno> {
        self.stat(dirfd, path, true).map(|_| ())
    }

    pub fn readlink(&self, dirfd: i32, path: &str) -> Result<Vec<u8>, Errno> {
        let guest = self.guest_absolute(dirfd, path)?;
        if guest == "/proc/self/exe" {
            return Ok(self.exe_path.as_bytes().to_vec());
        }
        let host = self.resolve(dirfd, path)?;
        let target = std::fs::read_link(host).map_err(|e| errno::from_io(&e))?;
        Ok(target.as_os_str().as_encoded_bytes().to_vec())
    }

    /// `dup`/`dup2`/`dup3`: a new descriptor at or above `min` (or exactly
    /// `at`) sharing the underlying host object.
    pub fn dup(&mut self, fd: i32, at: Option<i32>, cloexec: bool) -> Result<i32, Errno> {
        let source = self.get(fd)?;
        let kind = match &source.kind {
            FdKind::Stdin => FdKind::Stdin,
            FdKind::Stdout => FdKind::Stdout,
            FdKind::Stderr => FdKind::Stderr,
            FdKind::File(file) => FdKind::File(file.try_clone().map_err(|e| errno::from_io(&e))?),
            FdKind::Dir { entries, pos } => FdKind::Dir {
                entries: entries.clone(),
                pos: *pos,
            },
        };
        let copy = Fd {
            kind,
            path: source.path.clone(),
            cloexec,
            append: source.append,
        };
        match at {
            None => self.install(copy, 0),
            Some(target) => {
                let slot = usize::try_from(target).map_err(|_| EBADF)?;
                if slot >= MAX_FDS {
                    return Err(EBADF);
                }
                if slot >= self.fds.len() {
                    self.fds.resize_with(slot + 1, || None);
                }
                self.fds[slot] = Some(copy);
                Ok(target)
            }
        }
    }

    /// Whether descriptor `fd` is a terminal, for `ioctl`.
    pub fn is_tty(&self, fd: i32) -> Result<bool, Errno> {
        let entry = self.get(fd)?;
        Ok(match entry.kind {
            FdKind::Stdin => {
                self.stdio == Stdio::Host && std::io::IsTerminal::is_terminal(&std::io::stdin())
            }
            FdKind::Stdout => {
                self.stdio == Stdio::Host && std::io::IsTerminal::is_terminal(&std::io::stdout())
            }
            FdKind::Stderr => {
                self.stdio == Stdio::Host && std::io::IsTerminal::is_terminal(&std::io::stderr())
            }
            _ => false,
        })
    }

    /// Fills at most `len` bytes of `linux_dirent64` records.
    pub fn getdents64(&mut self, fd: i32, len: usize) -> Result<Vec<u8>, Errno> {
        let entry = self.get_mut(fd)?;
        let FdKind::Dir { entries, pos } = &mut entry.kind else {
            return Err(ENOTDIR);
        };
        let mut out = Vec::new();
        while *pos < entries.len() {
            let d = &entries[*pos];
            let reclen = (8 + 8 + 2 + 1 + d.name.len() + 1).div_ceil(8) * 8;
            if out.len() + reclen > len {
                if out.is_empty() {
                    return Err(EINVAL);
                }
                break;
            }
            out.extend_from_slice(&d.ino.to_le_bytes());
            out.extend_from_slice(&((*pos + 1) as u64).to_le_bytes());
            out.extend_from_slice(&(reclen as u16).to_le_bytes());
            out.push(d.kind);
            out.extend_from_slice(&d.name);
            out.resize(out.len() + reclen - (8 + 8 + 2 + 1 + d.name.len()), 0);
            *pos += 1;
        }
        Ok(out)
    }

    pub fn chdir(&mut self, path: &str) -> Result<(), Errno> {
        let guest = self.guest_absolute(AT_FDCWD, path)?;
        let host = self.resolve(AT_FDCWD, path)?;
        if !std::fs::metadata(host)
            .map_err(|e| errno::from_io(&e))?
            .is_dir()
        {
            return Err(ENOTDIR);
        }
        self.cwd = guest;
        Ok(())
    }

    pub fn ioctl_unsupported(&self, fd: i32) -> Errno {
        match self.get(fd) {
            Ok(_) => ENOTTY,
            Err(e) => e,
        }
    }
}

fn dirent_type(kind: std::fs::FileType) -> u8 {
    if kind.is_dir() {
        4
    } else if kind.is_symlink() {
        10
    } else if kind.is_fifo() {
        1
    } else if kind.is_char_device() {
        2
    } else if kind.is_block_device() {
        6
    } else if kind.is_socket() {
        12
    } else {
        8
    }
}

/// Lexically normalizes an absolute guest path: collapses `.`/`..` and
/// repeated slashes. `..` at the root stays at the root.
pub fn normalize(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            p => parts.push(p),
        }
    }
    if parts.is_empty() {
        "/".to_owned()
    } else {
        format!("/{}", parts.join("/"))
    }
}
