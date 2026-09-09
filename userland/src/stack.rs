//! The SysV x86-64 initial process stack.
//!
//! From the top down: the argument and environment strings, 16 random bytes for
//! `AT_RANDOM`, padding to 16 bytes, then the vector the entry point reads
//! upward from `RSP`: `argc`, `argv[]`, `NULL`, `envp[]`, `NULL`, the auxiliary
//! vector as `(type, value)` pairs ending in `AT_NULL`. `RSP` is 16-byte
//! aligned at entry, as the ABI requires.

use qcode_vm::{Mmu, perm};

use crate::loader::LoadedImage;

pub const AT_NULL: u64 = 0;
pub const AT_PHDR: u64 = 3;
pub const AT_PHENT: u64 = 4;
pub const AT_PHNUM: u64 = 5;
pub const AT_PAGESZ: u64 = 6;
pub const AT_BASE: u64 = 7;
pub const AT_FLAGS: u64 = 8;
pub const AT_ENTRY: u64 = 9;
pub const AT_UID: u64 = 11;
pub const AT_EUID: u64 = 12;
pub const AT_GID: u64 = 13;
pub const AT_EGID: u64 = 14;
pub const AT_PLATFORM: u64 = 15;
pub const AT_HWCAP: u64 = 16;
pub const AT_CLKTCK: u64 = 17;
pub const AT_SECURE: u64 = 23;
pub const AT_RANDOM: u64 = 25;
pub const AT_HWCAP2: u64 = 26;
pub const AT_EXECFN: u64 = 31;

/// Top of the main thread's stack (exclusive).
pub const STACK_TOP: u64 = 0x7fff_f000_0000;
/// 8 MiB, the usual `RLIMIT_STACK`.
pub const STACK_SIZE: u64 = 8 << 20;

/// Identity the guest sees.
#[derive(Debug, Clone, Copy)]
pub struct Identity {
    pub uid: u64,
    pub gid: u64,
}

/// Maps the stack and writes the initial vector. Returns the entry `RSP`.
pub fn build(
    mmu: &mut Mmu,
    image: &LoadedImage,
    argv: &[String],
    envp: &[String],
    identity: Identity,
    random: [u8; 16],
) -> u64 {
    mmu.map(STACK_TOP - STACK_SIZE, STACK_SIZE, perm::RW_INIT)
        .expect("the stack range is a valid mapping");

    let mut sp = STACK_TOP;
    let mut push_bytes = |mmu: &mut Mmu, bytes: &[u8]| -> u64 {
        sp -= bytes.len() as u64;
        mmu.write(sp, bytes).expect("inside the stack mapping");
        sp
    };

    // Strings, highest first so that argv[0] ends up lowest — the order Linux
    // uses, which nothing depends on but which makes stack dumps familiar.
    let execfn = push_bytes(mmu, &cstr(argv.first().map(String::as_str).unwrap_or("")));
    let platform = push_bytes(mmu, b"x86_64\0");
    let mut env_ptrs: Vec<u64> = envp
        .iter()
        .rev()
        .map(|s| push_bytes(mmu, &cstr(s)))
        .collect();
    env_ptrs.reverse();
    let mut arg_ptrs: Vec<u64> = argv
        .iter()
        .rev()
        .map(|s| push_bytes(mmu, &cstr(s)))
        .collect();
    arg_ptrs.reverse();
    let random_ptr = push_bytes(mmu, &random);

    let mut auxv: Vec<(u64, u64)> = Vec::new();
    if let Some(phdr) = image.phdr {
        auxv.push((AT_PHDR, phdr));
        auxv.push((AT_PHENT, u64::from(image.phentsize)));
        auxv.push((AT_PHNUM, u64::from(image.phnum)));
    }
    auxv.push((AT_PAGESZ, qcode_vm::PAGE_SIZE));
    auxv.push((AT_BASE, 0));
    auxv.push((AT_FLAGS, 0));
    auxv.push((AT_ENTRY, image.entry));
    auxv.push((AT_UID, identity.uid));
    auxv.push((AT_EUID, identity.uid));
    auxv.push((AT_GID, identity.gid));
    auxv.push((AT_EGID, identity.gid));
    auxv.push((AT_PLATFORM, platform));
    // A baseline CPU: no feature bits that would steer a runtime towards code
    // this emulator does not lift.
    auxv.push((AT_HWCAP, 0));
    auxv.push((AT_HWCAP2, 0));
    auxv.push((AT_CLKTCK, 100));
    auxv.push((AT_SECURE, 0));
    auxv.push((AT_RANDOM, random_ptr));
    auxv.push((AT_EXECFN, execfn));
    auxv.push((AT_NULL, 0));

    // Words: argc, argv, NULL, envp, NULL, auxv pairs.
    let words = 1 + arg_ptrs.len() + 1 + env_ptrs.len() + 1 + auxv.len() * 2;
    let mut sp = sp & !0xf;
    if !(words * 8).is_multiple_of(16) {
        sp -= 8;
    }
    sp -= (words * 8) as u64;
    debug_assert_eq!(sp % 16, 0);

    let mut at = sp;
    let mut put = |mmu: &mut Mmu, value: u64| {
        mmu.write(at, &value.to_le_bytes())
            .expect("inside the stack mapping");
        at += 8;
    };
    put(mmu, arg_ptrs.len() as u64);
    for p in &arg_ptrs {
        put(mmu, *p);
    }
    put(mmu, 0);
    for p in &env_ptrs {
        put(mmu, *p);
    }
    put(mmu, 0);
    for (kind, value) in &auxv {
        put(mmu, *kind);
        put(mmu, *value);
    }
    sp
}

fn cstr(s: &str) -> Vec<u8> {
    let mut v = s.as_bytes().to_vec();
    v.push(0);
    v
}
