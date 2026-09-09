//! Hooks are rewrites of the lifted code, so they must behave identically on
//! the interpreter and under the JIT, and the JIT must keep running natively
//! around them.

use std::cell::RefCell;
use std::rc::Rc;

use qcode::{context::Context, value::BlockId};
use qcode_emulator::{EmulatorErrorKind, StandaloneEmulator};
use qcode_jit::Jit;
use qcode_vm::{
    AddressHook, BlockEntryHook, CompareHook, InterruptKind, Vm, VmExit, VmMemory, WriteWatch, perm,
};
use qcode_vm::{BlockExecutor, Executed};
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

const ENTRY: u64 = 0x1000;

/// A JIT the test keeps a handle on, to read its counters after the run.
struct SharedJit(Rc<RefCell<Jit>>);

impl BlockExecutor for SharedJit {
    fn run_block(
        &mut self,
        ctx: &Context<'_>,
        emu: &mut StandaloneEmulator<VmMemory>,
        block: BlockId,
        start: usize,
        chain: bool,
    ) -> Result<Option<Executed>, EmulatorErrorKind> {
        self.0.borrow_mut().run_block(ctx, emu, block, start, chain)
    }
}

fn machine(code: &[u8], jit: bool) -> Vm<SleighCodeSource<'static>> {
    let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
    let ctx = source.new_context();
    let mut memory = VmMemory::new();
    memory
        .mmu
        .write_unchecked(ENTRY, code, perm::READ | perm::EXEC);
    memory.mmu.map(0x20000, 0x2000, perm::RW_INIT).unwrap();
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
    drive_with(vm, regs, |vm, code, args| {
        let addr = args[0].expect("the hook passes the address");
        assert_eq!(
            vm.pending_interrupt().and_then(|i| i.pc),
            Some(addr),
            "the stop is at the hooked address"
        );
        on_hook(vm, code, addr);
    })
}

/// As [`drive`], handing each interrupt's code and arguments to `on_hook`.
fn drive_with(
    vm: &mut Vm<SleighCodeSource<'static>>,
    regs: &[&str],
    mut on_hook: impl FnMut(&mut Vm<SleighCodeSource<'static>>, u64, &[Option<u64>]),
) -> Vec<Option<u64>> {
    for _ in 0..10_000 {
        match vm.run(1_000_000) {
            VmExit::Interrupt(interrupt) => {
                let InterruptKind::Explicit { code } = interrupt.kind else {
                    panic!("unexpected intrinsic {:?}", interrupt.kind);
                };
                on_hook(vm, code, &interrupt.args);
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
        vm.add_hook(BlockEntryHook {
            begin: 1,
            end: 0,
            code: 7,
        });
        let mut trace = Vec::new();
        let regs = drive(&mut vm, &["ECX", "EAX"], |vm, code, addr| {
            assert_eq!(code, 7);
            assert_eq!(vm.pending_interrupt().and_then(|i| i.pc), Some(addr));
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
        vm.add_hook(AddressHook::new([0x100a], 3));
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
        vm.add_hook(AddressHook::new([0x1005], 9));
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
        vm.add_hook(AddressHook::new([0x1005], 1));
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

/// Writes in and out of a watched range; the loop at the end writes outside
/// it fifty times and must never leave compiled code.
const WRITES: &[u8] = &[
    0xb8, 0x44, 0x33, 0x22, 0x11, // 1000: mov eax, 0x11223344
    0x89, 0x04, 0x25, 0x00, 0x00, 0x02, 0x00, // 1005: mov [0x20000], eax   (watched)
    0x89, 0x04, 0x25, 0x00, 0x10, 0x02, 0x00, // 100c: mov [0x21000], eax
    0xc6, 0x04, 0x25, 0x10, 0x00, 0x02, 0x00,
    0x7f, // 1013: mov byte [0x20010], 0x7f (watched)
    0xb9, 0x32, 0x00, 0x00, 0x00, // 101b: mov ecx, 50
    0x89, 0x0c, 0x25, 0x08, 0x10, 0x02, 0x00, // 1020: mov [0x21008], ecx
    0xff, 0xc9, // 1027: dec ecx
    0x75, 0xf5, // 1029: jnz 1020
];

#[test]
fn a_write_watch_only_leaves_the_vm_for_writes_in_its_range() {
    for jit in [false, true] {
        let mut vm = machine(WRITES, false);
        let shared = Rc::new(RefCell::new(Jit::new()));
        if jit {
            vm.set_block_executor(Box::new(SharedJit(shared.clone())));
        }
        vm.add_hook(WriteWatch {
            begin: 0x20000,
            end: 0x20fff,
            code: 5,
        });
        let mut writes = Vec::new();
        let regs = drive_with(&mut vm, &["ECX"], |vm, code, args| {
            assert_eq!(code, 5);
            let pc = vm.pending_interrupt().and_then(|i| i.pc);
            writes.push((pc, args.to_vec()));
        });
        assert_eq!(regs, vec![Some(0)], "jit={jit}: the loop ran to completion");
        assert_eq!(
            writes,
            vec![
                (
                    Some(0x1005),
                    vec![Some(0x20000), Some(4), Some(0x1122_3344)]
                ),
                (Some(0x1013), vec![Some(0x20010), Some(1), Some(0x7f)]),
            ],
            "jit={jit}"
        );
        // The watched writes landed too.
        let mut word = [0u8; 4];
        vm.memory().mmu.read(0x20000, &mut word).unwrap();
        assert_eq!(u32::from_le_bytes(word), 0x1122_3344);
        let mut byte = [0u8; 1];
        vm.memory().mmu.read(0x20010, &mut byte).unwrap();
        assert_eq!(byte[0], 0x7f);
        if jit {
            // Fifty loop iterations of an unwatched store, each a range check
            // and a store in native code, and the loop split into two blocks
            // by the hook: at least a hundred native block runs, all chained.
            let stats = shared.borrow().stats.clone();
            assert!(
                stats.native_runs >= 100,
                "jit: expected the hooked loop to run natively, got {stats:?}"
            );
            // Every decline is an empty placeholder the lifter left for an
            // address nothing reached; no block holding hook code is declined.
            let ctx = vm.context().clone();
            let mut jit = shared.borrow_mut();
            for func in ctx.function_ids() {
                for block in qcode::value::FunctionBody::from_id(&ctx, func).block_ids() {
                    if let Err(reason) = jit.try_compile(&ctx, block) {
                        assert!(
                            ctx.block(block).instruction_ids().is_empty(),
                            "jit: block {block:?} declined: {reason}"
                        );
                    }
                }
            }
        }
    }
}

/// `mov eax, 1337; mov ebx, 7; cmp eax, ebx`.
const COMPARE: &[u8] = &[
    0xb8, 0x39, 0x05, 0x00, 0x00, // mov eax, 1337
    0xbb, 0x07, 0x00, 0x00, 0x00, // mov ebx, 7
    0x39, 0xd8, // cmp eax, ebx
];

#[test]
fn a_compare_hook_receives_the_operands_the_guest_compared() {
    let mut results = Vec::new();
    for jit in [false, true] {
        let mut vm = machine(COMPARE, jit);
        vm.add_hook(CompareHook { code: 11 });
        let mut seen = Vec::new();
        drive_with(&mut vm, &[], |_, code, args| {
            assert_eq!(code, 11);
            seen.push(args.to_vec());
        });
        assert!(
            seen.contains(&vec![Some(1337), Some(7)]),
            "jit={jit}: no comparison of eax against ebx was reported: {seen:?}"
        );
        results.push(seen);
    }
    assert_eq!(results[0], results[1]);
}
