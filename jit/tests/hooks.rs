//! Hooks are rewrites of the lifted code, so they must behave identically on
//! the interpreter and under the JIT, and the JIT must keep running natively
//! around them.

use qcode_jit::Jit;
use qcode_vm::{AddressHook, BlockEntryHook, InterruptKind, Vm, VmExit, VmMemory, perm};
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

const ENTRY: u64 = 0x1000;

fn machine(code: &[u8], jit: bool) -> Vm<SleighCodeSource<'static>> {
    let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
    let ctx = source.new_context();
    let mut memory = VmMemory::new();
    memory
        .mmu
        .write_unchecked(ENTRY, code, perm::READ | perm::EXEC);
    let mut vm = Vm::at_address(ctx, ENTRY, source, memory).expect("the entry decodes");
    if jit {
        vm.set_block_executor(Box::new(Jit::new()));
    }
    vm
}

/// `mov ecx, 5; l: dec ecx; jnz l; mov eax, 42`, then unmapped bytes.
const LOOP: &[u8] = &[
    0xb9, 0x05, 0x00, 0x00, 0x00, // 1000: mov ecx, 5
    0xff, 0xc9, // 1005: dec ecx
    0x75, 0xfc, // 1007: jnz 1005
    0xb8, 0x2a, 0x00, 0x00, 0x00, // 1009: mov eax, 42
];

/// Straight-line arithmetic, one block once absorbed.
const ARITH: &[u8] = &[
    0xb8, 0x39, 0x05, 0x00, 0x00, // 1000: mov eax, 1337
    0xbb, 0x07, 0x00, 0x00, 0x00, // 1005: mov ebx, 7
    0x01, 0xd8, // 100a: add eax, ebx
    0x29, 0xd8, // 100c: sub eax, ebx
    0x31, 0xd8, // 100e: xor eax, ebx
];

/// Runs until something other than an explicit interrupt stops the machine,
/// calling `on_hook` at each interrupt with its code and address, and
/// returns the final values of the named registers.
fn drive(
    vm: &mut Vm<SleighCodeSource<'static>>,
    regs: &[&str],
    mut on_hook: impl FnMut(&mut Vm<SleighCodeSource<'static>>, u64, u64),
) -> Vec<Option<u64>> {
    for _ in 0..10_000 {
        match vm.run(1_000_000) {
            VmExit::Interrupt(interrupt) => {
                let InterruptKind::Explicit { code } = interrupt.kind else {
                    panic!("unexpected intrinsic {:?}", interrupt.kind);
                };
                let addr = interrupt.args[0].expect("the hook passes the address");
                assert_eq!(
                    interrupt.pc,
                    Some(addr),
                    "the stop is at the hooked address"
                );
                on_hook(vm, code, addr);
                vm.resume(None).unwrap();
            }
            _ => break,
        }
    }
    let ctx = vm.context().clone();
    regs.iter()
        .map(|name| vm.emulator().read_varnode_by_name(&ctx, name))
        .collect()
}

#[test]
fn a_block_entry_hook_fires_the_same_way_on_both_strategies() {
    let mut traces = Vec::new();
    for jit in [false, true] {
        let mut vm = machine(LOOP, jit);
        vm.add_injector(Box::new(BlockEntryHook {
            begin: 1,
            end: 0,
            code: 7,
        }));
        let mut trace = Vec::new();
        let regs = drive(&mut vm, &["ECX", "EAX"], |_, code, addr| {
            assert_eq!(code, 7);
            trace.push(addr);
        });
        assert_eq!(regs, vec![Some(0), Some(42)], "jit={jit}");
        assert_eq!(trace.first(), Some(&0x1000), "jit={jit}");
        assert_eq!(trace.last(), Some(&0x1009), "jit={jit}");
        assert!(
            trace.iter().filter(|&&addr| addr == 0x1005).count() >= 4,
            "jit={jit}: the loop body is entered once per iteration: {trace:x?}"
        );
        if jit {
            assert!(
                vm.stats.native_bodies > 0,
                "hooked blocks still run natively"
            );
        }
        traces.push(trace);
    }
    assert_eq!(traces[0], traces[1]);
}

#[test]
fn an_address_hook_in_the_middle_of_a_block_keeps_both_halves_native() {
    let mut results = Vec::new();
    for jit in [false, true] {
        let mut vm = machine(ARITH, jit);
        vm.add_injector(Box::new(AddressHook::new([0x100a], 3)));
        let mut fired = 0;
        let mut native_at_hook = 0;
        let mut idx_at_hook = 0;
        let regs = drive(&mut vm, &["EAX", "EBX"], |vm, code, addr| {
            assert_eq!((code, addr), (3, 0x100a));
            fired += 1;
            native_at_hook = vm.stats.native_bodies;
            idx_at_hook = vm.emulator().idx;
        });
        assert_eq!(fired, 1, "jit={jit}");
        assert_eq!(regs, vec![Some(1337 ^ 7), Some(7)], "jit={jit}");
        if jit {
            assert!(
                idx_at_hook > 0,
                "the two `mov`s before the hook are in the same block"
            );
            assert!(native_at_hook > 0, "the prefix ran natively");
            assert!(
                vm.stats.native_bodies > native_at_hook,
                "the rest of the block ran natively after the hook"
            );
        }
        results.push(regs);
    }
    assert_eq!(results[0], results[1]);
}

#[test]
fn a_hook_registered_later_reaches_code_already_lifted() {
    for jit in [false, true] {
        let mut vm = machine(LOOP, jit);
        // Run into the loop — a breakpoint on its body stops the machine there
        // on either strategy — then hook the instruction it keeps returning to.
        vm.add_breakpoint(0x1005);
        let exit = vm.run(1_000_000);
        assert!(
            matches!(exit, VmExit::Breakpoint(0x1005)),
            "jit={jit}: {exit:?}"
        );
        vm.remove_breakpoint(0x1005);
        vm.add_injector(Box::new(AddressHook::new([0x1005], 9)));
        let mut fired = 0;
        let regs = drive(&mut vm, &["ECX"], |_, code, addr| {
            assert_eq!((code, addr), (9, 0x1005));
            fired += 1;
        });
        assert_eq!(regs, vec![Some(0)], "jit={jit}");
        assert!(fired >= 1, "jit={jit}: the hook was reached");
    }
}

#[test]
fn state_written_at_a_hook_is_what_the_code_after_it_sees() {
    for jit in [false, true] {
        let mut vm = machine(LOOP, jit);
        vm.add_injector(Box::new(AddressHook::new([0x1005], 1)));
        let mut fired = 0;
        let regs = drive(&mut vm, &["ECX", "EAX"], |vm, _, _| {
            fired += 1;
            if fired == 1 {
                // Shorten the loop from five iterations to two.
                let ctx = vm.context().clone();
                vm.emulator().set_varnode_by_name(&ctx, "ECX", 2).unwrap();
            }
        });
        assert_eq!(fired, 2, "jit={jit}: dec runs twice after the write");
        assert_eq!(regs, vec![Some(0), Some(42)], "jit={jit}");
    }
}
