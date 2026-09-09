//! The Unicorn-shaped hooks: callbacks called from inside `Vm::run`, on
//! either strategy, with the same observations.

use qcode_jit::Jit;
use qcode_vm::{HookAction, InsnAction, InterruptKind, MemAccess, Vm, VmExit, VmMemory, perm};
use std::cell::RefCell;
use std::rc::Rc;
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

const ENTRY: u64 = 0x1000;

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

fn register(vm: &mut Vm<SleighCodeSource<'static>>, name: &str) -> Option<u64> {
    let ctx = vm.context().clone();
    vm.emulator().read_varnode_by_name(&ctx, name)
}

/// `mov ecx, 5; l: dec ecx; jnz l; mov eax, 42`, then unmapped bytes.
const LOOP: &[u8] = &[
    0xb9, 0x05, 0x00, 0x00, 0x00, // 1000: mov ecx, 5
    0xff, 0xc9, // 1005: dec ecx
    0x75, 0xfc, // 1007: jnz 1005
    0xb8, 0x2a, 0x00, 0x00, 0x00, // 1009: mov eax, 42
];

/// Two watched writes among unwatched ones, then a loop of unwatched writes.
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
    0x8b, 0x04, 0x25, 0x00, 0x00, 0x02, 0x00, // 102b: mov eax, [0x20000]  (read, watched)
];

#[test]
fn hook_code_sees_every_instruction_in_its_range_once() {
    let mut traces = Vec::new();
    for jit in [false, true] {
        let mut vm = machine(LOOP, jit);
        let trace = Rc::new(RefCell::new(Vec::new()));
        let seen = trace.clone();
        vm.hook_code(1, 0, move |_, pc| {
            seen.borrow_mut().push(pc);
            HookAction::Continue
        });
        let exit = vm.run(1_000_000);
        assert!(
            !matches!(exit, VmExit::Interrupt(_) | VmExit::HookStop(_)),
            "jit={jit}: {exit:?}"
        );
        assert_eq!(register(&mut vm, "EAX"), Some(42), "jit={jit}");
        let trace = trace.borrow().clone();
        // mov ecx, five times (dec, jnz), mov eax.
        assert_eq!(trace.len(), 12, "jit={jit}: {trace:x?}");
        assert_eq!(trace[0], 0x1000);
        assert_eq!(trace[11], 0x1009);
        traces.push(trace);
    }
    assert_eq!(traces[0], traces[1]);
}

#[test]
fn hook_block_fires_in_registration_order_and_a_stop_ends_the_run() {
    for jit in [false, true] {
        let mut vm = machine(LOOP, jit);
        let order = Rc::new(RefCell::new(Vec::new()));
        let first = order.clone();
        let second = order.clone();
        vm.hook_block(1, 0, move |_, pc| {
            first.borrow_mut().push((1, pc));
            HookAction::Continue
        });
        let stopper = vm.hook_block(1, 0, move |_, pc| {
            second.borrow_mut().push((2, pc));
            if pc == 0x1009 {
                HookAction::Stop
            } else {
                HookAction::Continue
            }
        });
        let exit = vm.run(1_000_000);
        assert!(
            matches!(exit, VmExit::HookStop(id) if id == stopper),
            "jit={jit}: {exit:?}"
        );
        // Stopped *before* `mov eax, 42`.
        assert_eq!(register(&mut vm, "EAX"), Some(0), "jit={jit}");
        assert_eq!(register(&mut vm, "ECX"), Some(0), "jit={jit}");
        let order = order.borrow().clone();
        assert_eq!(&order[..2], &[(1, 0x1000), (2, 0x1000)], "jit={jit}");
        assert_eq!(order.last(), Some(&(2, 0x1009)), "jit={jit}");
        // Running again continues past the site.
        let exit = vm.run(1_000_000);
        assert!(!matches!(exit, VmExit::HookStop(_)), "jit={jit}: {exit:?}");
        assert_eq!(register(&mut vm, "EAX"), Some(42), "jit={jit}");
    }
}

#[test]
fn memory_hooks_report_the_access_and_can_stop_before_it_happens() {
    for jit in [false, true] {
        let mut vm = machine(WRITES, jit);
        let writes = Rc::new(RefCell::new(Vec::new()));
        let seen = writes.clone();
        let watch = vm.hook_mem_write(0x20000, 0x20fff, move |_, access| {
            seen.borrow_mut().push(*access);
            if access.size == 1 {
                HookAction::Stop
            } else {
                HookAction::Continue
            }
        });
        let reads = Rc::new(RefCell::new(Vec::new()));
        let seen = reads.clone();
        vm.hook_mem_read(0x20000, 0x20fff, move |_, access| {
            seen.borrow_mut().push(*access);
            HookAction::Continue
        });

        let exit = vm.run(1_000_000);
        assert!(
            matches!(exit, VmExit::HookStop(id) if id == watch),
            "jit={jit}: {exit:?}"
        );
        assert_eq!(
            writes.borrow().as_slice(),
            &[
                MemAccess {
                    pc: Some(0x1005),
                    addr: 0x20000,
                    size: 4,
                    value: Some(0x1122_3344)
                },
                MemAccess {
                    pc: Some(0x1013),
                    addr: 0x20010,
                    size: 1,
                    value: Some(0x7f)
                },
            ],
            "jit={jit}"
        );
        // Stopped before the byte store: it has not happened yet.
        let mut byte = [0u8; 1];
        vm.memory().mmu.read(0x20010, &mut byte).unwrap();
        assert_eq!(byte[0], 0, "jit={jit}");

        let exit = vm.run(1_000_000);
        assert!(
            !matches!(exit, VmExit::HookStop(_) | VmExit::Interrupt(_)),
            "jit={jit}: {exit:?}"
        );
        vm.memory().mmu.read(0x20010, &mut byte).unwrap();
        assert_eq!(byte[0], 0x7f, "jit={jit}");
        assert_eq!(
            reads.borrow().as_slice(),
            &[MemAccess {
                pc: Some(0x102b),
                addr: 0x20000,
                size: 4,
                value: None
            }],
            "jit={jit}"
        );
        assert_eq!(register(&mut vm, "EAX"), Some(0x1122_3344), "jit={jit}");
    }
}

/// `mov eax, 5; rdtsc; mov ebx, eax; mov ecx, edx`.
const RDTSC: &[u8] = &[
    0xb8, 0x05, 0x00, 0x00, 0x00, 0x0f, 0x31, 0x89, 0xc3, 0x89, 0xd1,
];

#[test]
fn an_instruction_hook_supplies_the_result_of_a_user_op() {
    for jit in [false, true] {
        let mut vm = machine(RDTSC, jit);
        vm.hook_insn("cpuid_basic", |_, _| InsnAction::Handled(Some(0)));
        vm.hook_insn("rdtsc", |_, interrupt| {
            assert!(matches!(&interrupt.kind, InterruptKind::Intrinsic { name, .. } if name.as_ref() == "rdtsc"));
            InsnAction::Handled(Some(0x1122_3344_5566_7788))
        });
        let exit = vm.run(1_000_000);
        assert!(!matches!(exit, VmExit::Interrupt(_)), "jit={jit}: {exit:?}");
        assert_eq!(register(&mut vm, "EBX"), Some(0x5566_7788), "jit={jit}");
        assert_eq!(register(&mut vm, "ECX"), Some(0x1122_3344), "jit={jit}");
    }
}

#[test]
fn an_unanswered_user_op_still_reaches_the_caller() {
    let mut vm = machine(RDTSC, false);
    vm.hook_intr(|_, _| InsnAction::Unhandled);
    let exit = vm.run(1_000_000);
    assert!(
        matches!(&exit, VmExit::Interrupt(i) if matches!(&i.kind, InterruptKind::Intrinsic { name, .. } if name.as_ref() == "rdtsc")),
        "{exit:?}"
    );
}

#[test]
fn a_deleted_hook_stops_being_called() {
    let mut vm = machine(LOOP, true);
    let count = Rc::new(RefCell::new(0));
    let seen = count.clone();
    let id = vm.hook_address(0x1005, move |_, _| {
        *seen.borrow_mut() += 1;
        HookAction::Stop
    });
    let exit = vm.run(1_000_000);
    assert!(
        matches!(exit, VmExit::HookStop(stopped) if stopped == id),
        "{exit:?}"
    );
    assert_eq!(*count.borrow(), 1);
    assert!(vm.hook_del(id));
    let exit = vm.run(1_000_000);
    assert!(!matches!(exit, VmExit::HookStop(_)), "{exit:?}");
    assert_eq!(*count.borrow(), 1, "the hook was not called after deletion");
    assert_eq!(register(&mut vm, "EAX"), Some(42));
}
