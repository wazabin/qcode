//! Guest-memory accessors: every copy between the host and the guest goes
//! through the MMU, and a failed access surfaces as `EFAULT` rather than a
//! crash of the environment.

use qcode_vm::Mmu;

use crate::errno::{EFAULT, ENAMETOOLONG, Errno};

/// Longest path the kernel accepts, and the cap on C strings read here.
pub const PATH_MAX: usize = 4096;

pub fn read(mmu: &Mmu, addr: u64, len: usize) -> Result<Vec<u8>, Errno> {
    let mut out = vec![0; len];
    mmu.read(addr, &mut out).map_err(|_| EFAULT)?;
    Ok(out)
}

pub fn write(mmu: &mut Mmu, addr: u64, bytes: &[u8]) -> Result<(), Errno> {
    mmu.write(addr, bytes).map_err(|_| EFAULT)
}

pub fn read_u64(mmu: &Mmu, addr: u64) -> Result<u64, Errno> {
    let mut buf = [0; 8];
    mmu.read(addr, &mut buf).map_err(|_| EFAULT)?;
    Ok(u64::from_le_bytes(buf))
}

pub fn read_u32(mmu: &Mmu, addr: u64) -> Result<u32, Errno> {
    let mut buf = [0; 4];
    mmu.read(addr, &mut buf).map_err(|_| EFAULT)?;
    Ok(u32::from_le_bytes(buf))
}

pub fn write_u64(mmu: &mut Mmu, addr: u64, value: u64) -> Result<(), Errno> {
    write(mmu, addr, &value.to_le_bytes())
}

pub fn write_u32(mmu: &mut Mmu, addr: u64, value: u32) -> Result<(), Errno> {
    write(mmu, addr, &value.to_le_bytes())
}

/// Reads a NUL-terminated string of at most [`PATH_MAX`] bytes, without the
/// terminator.
pub fn read_cstr(mmu: &Mmu, addr: u64) -> Result<Vec<u8>, Errno> {
    let mut out = Vec::new();
    let mut byte = [0u8];
    loop {
        if out.len() >= PATH_MAX {
            return Err(ENAMETOOLONG);
        }
        mmu.read(addr + out.len() as u64, &mut byte)
            .map_err(|_| EFAULT)?;
        if byte[0] == 0 {
            return Ok(out);
        }
        out.push(byte[0]);
    }
}

/// Reads a path argument as a Rust string; non-UTF-8 paths are rejected as
/// `ENOENT`, which is what the host would report for a name it cannot spell.
pub fn read_path(mmu: &Mmu, addr: u64) -> Result<String, Errno> {
    let bytes = read_cstr(mmu, addr)?;
    String::from_utf8(bytes).map_err(|_| crate::errno::ENOENT)
}
