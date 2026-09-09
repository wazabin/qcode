//! A trivial taint tracker in the memory hooks: bytes copied out of a secret
//! buffer are tainted wherever they land.
//!
//! The rule is the simplest one that works for memory-to-memory moves: a
//! store is tainted when the load just before it read a tainted byte.
//! Registers are not modelled, so taint that rests in a register across
//! several instructions is lost — that is the "trivial" part, and where a
//! real tracker would add a register map keyed off `hook_code`.
//!
//! Run with `cargo run --example taint`.

use std::{cell::RefCell, collections::BTreeSet, rc::Rc};

use qcode_jit::Jit;
use qcode_vm::{HookAction, Vm, VmExit, VmMemory, perm};
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

const SECRET: u64 = 0x20000;
const DEST: u64 = 0x21000;

/// `rep movsb` copies 8 bytes of the secret to `DEST`; a constant lands at
/// `DEST+16`; a load-and-store moves 4 bytes from `DEST+2` to `DEST+32`.
const PROGRAM: &[u8] = &[
    0xbe, 0x00, 0x00, 0x02, 0x00, // mov esi, SECRET
    0xbf, 0x00, 0x10, 0x02, 0x00, // mov edi, DEST
    0xb9, 0x08, 0x00, 0x00, 0x00, // mov ecx, 8
    0xf3, 0xa4, // rep movsb
    0xc7, 0x04, 0x25, 0x10, 0x10, 0x02, 0x00, 0x00, 0x00, 0x00,
    0x00, // mov dword [DEST+16], 0
    0x8b, 0x04, 0x25, 0x02, 0x10, 0x02, 0x00, // mov eax, [DEST+2]
    0x89, 0x04, 0x25, 0x20, 0x10, 0x02, 0x00, // mov [DEST+32], eax
    0xeb, 0xfe, // jmp $
];
const END: u64 = 0x1000 + 42;

fn main() {
    for jit in [false, true] {
        let tainted = track(jit);
        for &addr in &tainted {
            println!("tainted: {addr:#x}");
        }
        let expect: BTreeSet<u64> = (SECRET + 2..SECRET + 6)
            .chain(DEST + 2..DEST + 6)
            .chain(DEST + 32..DEST + 36)
            .collect();
        assert_eq!(tainted, expect, "jit={jit}: taint was not tracked");
        println!("ok (jit={jit}): {} tainted bytes", tainted.len());
    }
}

/// Runs the program under the tracker and reports the tainted bytes.
fn track(jit: bool) -> BTreeSet<u64> {
    let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
    let ctx = source.new_context();
    let mut memory = VmMemory::new();
    memory
        .mmu
        .write_unchecked(0x1000, PROGRAM, perm::READ | perm::EXEC);
    memory.mmu.map(SECRET, 0x2000, perm::RW_INIT).unwrap();
    let mut vm = Vm::at_address(ctx, 0x1000, source, memory).unwrap();
    if jit {
        vm.set_block_executor(Box::new(Jit::new()));
    }

    // Bytes 2..6 of the secret are the sensitive ones.
    let tainted: Rc<RefCell<BTreeSet<u64>>> =
        Rc::new(RefCell::new((SECRET + 2..SECRET + 6).collect()));
    let last_read_tainted = Rc::new(RefCell::new(false));

    let (set, flag) = (tainted.clone(), last_read_tainted.clone());
    vm.hook_mem_read(SECRET, SECRET + 0x2000, move |_, access| {
        let bytes = access.addr..access.addr + access.size;
        *flag.borrow_mut() = bytes.into_iter().any(|b| set.borrow().contains(&b));
        HookAction::Continue
    });
    let (set, flag) = (tainted.clone(), last_read_tainted.clone());
    vm.hook_mem_write(SECRET, SECRET + 0x2000, move |_, access| {
        let bytes = access.addr..access.addr + access.size;
        let mut set = set.borrow_mut();
        if std::mem::take(&mut *flag.borrow_mut()) {
            set.extend(bytes);
        } else {
            for b in bytes {
                set.remove(&b);
            }
        }
        HookAction::Continue
    });

    vm.add_breakpoint(END);
    let exit = vm.run(100_000);
    assert!(matches!(exit, VmExit::Breakpoint(END)), "{exit:?}");

    // Cloned out from under the hooks, which still hold the shared set.
    tainted.borrow().clone()
}
