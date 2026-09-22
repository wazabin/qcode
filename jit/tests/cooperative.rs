//! Several guest tasks share one `Vm` — one module of lifted code, one MMU,
//! one set of registers on the machine at a time. A task is taken off with
//! [`Vm::park`] and put back with [`Vm::unpark`] around another task's
//! address space and registers, the way a cooperative scheduler runs them.

use qcode::space::MemorySpaceId;
use qcode::value::{ValueId, Varnode};
use qcode_jit::Jit;
use qcode_vm::flat::FlatSpace;
use qcode_vm::{MmuSnapshot, Parked, Vm, VmExit, VmMemory, perm};
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

const CODE: u64 = 0x1000;

fn machine(code: &[u8], jit: bool) -> Vm<SleighCodeSource<'static>> {
    let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
    let ctx = source.new_context();
    let mut memory = VmMemory::new();
    memory
        .mmu
        .write_unchecked(CODE, code, perm::RW_INIT | perm::EXEC);
    let mut vm = Vm::at_address(ctx, CODE, source, memory).expect("the entry decodes");
    if jit {
        vm.set_block_executor(Box::new(Jit::new()));
    }
    vm
}

/// Where a named register lives, resolved once from the context.
fn reg(vm: &Vm<SleighCodeSource<'static>>, name: &str) -> (MemorySpaceId, u64, usize) {
    let ctx = vm.context();
    let Some(ValueId::Varnode(id)) = ctx.get_named(name) else {
        panic!("no register {name}");
    };
    let v = Varnode::from_id(ctx, id);
    (v.space().id.into(), v.address() as u64, v.size())
}

fn set(vm: &mut Vm<SleighCodeSource<'static>>, name: &str, value: u64) {
    let (space, addr, size) = reg(vm, name);
    vm.memory_mut()
        .flat_mut()
        .entry(space)
        .write_u128(addr, size, u128::from(value))
        .expect("register storage");
}

fn get(vm: &mut Vm<SleighCodeSource<'static>>, name: &str) -> u64 {
    let (space, addr, size) = reg(vm, name);
    vm.memory()
        .flat()
        .get(space)
        .and_then(|s| s.read_bytes(addr, size).ok())
        .map(|b| {
            b.iter()
                .take(8)
                .rev()
                .fold(0u64, |a, &x| (a << 8) | u64::from(x))
        })
        .unwrap_or(0)
}

/// A task off the machine: its address space and its register file, kept so
/// another task can use the machine and this one be put back exactly.
struct Off {
    parked: Parked,
    memory: MmuSnapshot,
    registers: FlatSpace,
}

fn reg_space(vm: &Vm<SleighCodeSource<'static>>) -> MemorySpaceId {
    reg(vm, "RAX").0
}

/// Takes the running task off the machine and returns it saved.
fn park(vm: &mut Vm<SleighCodeSource<'static>>) -> Off {
    let space = reg_space(vm);
    let parked = vm.park();
    let registers = vm.memory().flat().get(space).expect("registers").clone();
    let memory = vm.memory_mut().mmu.snapshot();
    Off {
        parked,
        memory,
        registers,
    }
}

/// Puts a saved task back on the machine: its address space and registers,
/// then its position.
fn unpark(vm: &mut Vm<SleighCodeSource<'static>>, task: Off) {
    let space = reg_space(vm);
    vm.memory_mut().mmu.restore(&task.memory);
    *vm.memory_mut().flat_mut().entry(space) = task.registers;
    vm.unpark(task.parked).expect("the task goes back on");
}

/// Two tasks with the same code but different `EAX`, interleaved a slice at
/// a time on one machine, each end with their own answer. The parked task's
/// registers and position survive the other's turns.
#[test]
fn two_tasks_take_turns_on_one_machine() {
    for jit in [false, true] {
        // loop: dec ecx; jnz loop; then add eax, 7; jmp $
        let code = [
            0xff, 0xc9, // 1000: dec ecx
            0x75, 0xfc, // 1002: jnz 1000
            0x83, 0xc0, 0x07, // 1004: add eax, 7
            0xeb, 0xfe, // 1007: jmp $
        ];
        let done = 0x1007;
        let mut vm = machine(&code, jit);
        vm.add_breakpoint(done);

        // Task A: eax=10, ecx=5000. Start it and take it off mid-loop.
        set(&mut vm, "EAX", 10);
        set(&mut vm, "ECX", 5000);
        assert!(matches!(vm.run(37), VmExit::InstructionLimit));
        let a = park(&mut vm);

        // Task B: eax=100, ecx=3, on the same code, run to the end.
        set(&mut vm, "EAX", 100);
        set(&mut vm, "ECX", 3);
        // B starts at the entry, a fresh position.
        vm.position_at(CODE).expect("B starts at the entry");
        assert!(
            matches!(vm.run(10_000), VmExit::Breakpoint(a) if a == done),
            "jit={jit}: B"
        );
        assert_eq!(get(&mut vm, "EAX"), 107, "jit={jit}: B's answer");
        let b = park(&mut vm);

        // A resumes exactly where it left off and finishes its own loop.
        unpark(&mut vm, a);
        assert!(
            matches!(vm.run(1_000_000), VmExit::Breakpoint(a) if a == done),
            "jit={jit}: A"
        );
        assert_eq!(get(&mut vm, "EAX"), 17, "jit={jit}: A's answer");
        assert_eq!(get(&mut vm, "ECX"), 0, "jit={jit}: A's loop count");

        // B put back at its final jmp stays there.
        unpark(&mut vm, b);
        assert_eq!(
            get(&mut vm, "EAX"),
            107,
            "jit={jit}: B still holds its answer"
        );
    }
}

/// Two tasks hold different bytes at the same address: one `mov eax, 1`, the
/// other `mov eax, 2`, both at CODE. Each runs its own bytes though the
/// module lifts that address only once — unpark throws the stale block away.
#[test]
fn each_task_runs_its_own_bytes_at_a_shared_address() {
    for jit in [false, true] {
        let mut vm = machine(&[0xb8, 0x01, 0x00, 0x00, 0x00, 0xeb, 0xfe], jit); // mov eax,1; jmp $
        vm.add_breakpoint(0x1005);
        assert!(matches!(vm.run(1000), VmExit::Breakpoint(0x1005)));
        assert_eq!(get(&mut vm, "EAX"), 1, "jit={jit}: first task");
        let first = park(&mut vm);

        // The second task's memory holds mov eax,2 at the same address. A
        // checked write to a page that now holds lifted code records the
        // change, the way a guest store or a restore of divergent memory
        // does; unpark then throws the stale block away.
        vm.memory_mut()
            .mmu
            .write(CODE, &[0xb8, 0x02, 0x00, 0x00, 0x00, 0xeb, 0xfe])
            .expect("the code page is writable");
        vm.position_at(CODE).expect("second task at the entry");
        assert!(matches!(vm.run(1000), VmExit::Breakpoint(0x1005)));
        assert_eq!(
            get(&mut vm, "EAX"),
            2,
            "jit={jit}: second task ran the first's bytes"
        );

        // The first task, put back, restores its own address space — mov
        // eax,1 at CODE — and runs that, not the second task's bytes.
        unpark(&mut vm, first);
        vm.position_at(CODE).expect("first task back at the entry");
        assert!(matches!(vm.run(1000), VmExit::Breakpoint(0x1005)));
        assert_eq!(
            get(&mut vm, "EAX"),
            1,
            "jit={jit}: first task lost its bytes"
        );
    }
}
