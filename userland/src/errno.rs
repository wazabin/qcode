//! Linux error numbers, and the `-errno` return convention.

/// A Linux error number, always positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Errno(pub i32);

impl Errno {
    /// The value a system call returns in `RAX` to report this error.
    pub fn as_ret(self) -> u64 {
        (-(self.0 as i64)) as u64
    }

    pub fn name(self) -> &'static str {
        match self.0 {
            1 => "EPERM",
            2 => "ENOENT",
            4 => "EINTR",
            5 => "EIO",
            9 => "EBADF",
            11 => "EAGAIN",
            12 => "ENOMEM",
            13 => "EACCES",
            14 => "EFAULT",
            17 => "EEXIST",
            20 => "ENOTDIR",
            21 => "EISDIR",
            22 => "EINVAL",
            24 => "EMFILE",
            25 => "ENOTTY",
            28 => "ENOSPC",
            29 => "ESPIPE",
            32 => "EPIPE",
            34 => "ERANGE",
            36 => "ENAMETOOLONG",
            38 => "ENOSYS",
            39 => "ENOTEMPTY",
            _ => "E???",
        }
    }
}

impl std::fmt::Display for Errno {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.name(), self.0)
    }
}

pub const EPERM: Errno = Errno(1);
pub const ENOENT: Errno = Errno(2);
pub const EINTR: Errno = Errno(4);
pub const EIO: Errno = Errno(5);
pub const EBADF: Errno = Errno(9);
pub const EAGAIN: Errno = Errno(11);
pub const ENOMEM: Errno = Errno(12);
pub const EACCES: Errno = Errno(13);
pub const EFAULT: Errno = Errno(14);
pub const EEXIST: Errno = Errno(17);
pub const ENOTDIR: Errno = Errno(20);
pub const EISDIR: Errno = Errno(21);
pub const EINVAL: Errno = Errno(22);
pub const EMFILE: Errno = Errno(24);
pub const ENOTTY: Errno = Errno(25);
pub const ENOSPC: Errno = Errno(28);
pub const ESPIPE: Errno = Errno(29);
pub const EPIPE: Errno = Errno(32);
pub const ERANGE: Errno = Errno(34);
pub const ENAMETOOLONG: Errno = Errno(36);
pub const ENOSYS: Errno = Errno(38);
pub const ENOTEMPTY: Errno = Errno(39);

/// Maps a host I/O error onto the closest Linux errno.
pub fn from_io(error: &std::io::Error) -> Errno {
    use std::io::ErrorKind as K;
    if let Some(code) = error.raw_os_error() {
        return Errno(code);
    }
    match error.kind() {
        K::NotFound => ENOENT,
        K::PermissionDenied => EACCES,
        K::AlreadyExists => EEXIST,
        K::InvalidInput => EINVAL,
        K::IsADirectory => EISDIR,
        K::NotADirectory => ENOTDIR,
        K::DirectoryNotEmpty => ENOTEMPTY,
        K::BrokenPipe => EPIPE,
        _ => EIO,
    }
}

/// The result of a system call: a value for `RAX`, or an error to negate.
pub type SysResult = Result<u64, Errno>;
