//! Hooks are rewrites of the lifted code, so they must behave identically on
//! the interpreter and under the JIT, and the JIT must keep running natively
//! around them.

use std::cell::RefCell;
use std::rc::Rc;

use qcode::{
    context::Context,
    value::{BlockId, Instruction, insn::IntBinop},
};
use qcode_emulator::{EmulatorErrorKind, StandaloneEmulator};
use qcode_jit::Jit;
use qcode_vm::{
    AddressHook, BlockEntryHook, BlockView, CompareHook, Emitter, Hook, InterruptKind, Site, Vm,
    VmExit, VmMemory, WriteWatch, perm,
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
        chain: u64,
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
                            !ctx.block(block).has_insns(),
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

/// `mov ecx, 10; l: mov [0x21000], ecx; dec ecx; jnz l; mov eax, 42`: ten
/// stores to guest RAM, and nothing else that stores.
const STORING_LOOP: &[u8] = &[
    0xb9, 0x0a, 0x00, 0x00, 0x00, // 1000: mov ecx, 10
    0x89, 0x0c, 0x25, 0x00, 0x10, 0x02, 0x00, // 1005: mov [0x21000], ecx
    0xff, 0xc9, // 100c: dec ecx
    0x75, 0xf5, // 100e: jnz 1005
    0xb8, 0x2a, 0x00, 0x00, 0x00, // 1010: mov eax, 42
];

/// Where the counting hook keeps its tally, inside the RW page `machine` maps.
const COUNTER: u64 = 0x21800;

/// Counts guest stores in a guest word, with no interrupt at all: the whole
/// hook is a load, an add and a store, compiled with the code it instruments.
struct StoreCounter;

impl Hook for StoreCounter {
    fn sites(&mut self, block: &BlockView<'_>) -> Vec<Site> {
        block
            .stores()
            .into_iter()
            .filter(|site| {
                // The counter stores this hook emits are stores to RAM too,
                // and a block offered again after absorption would present
                // them as sites. They carry no guest address; guest stores do.
                Instruction::from_id(block.ctx, site.anchor())
                    .address()
                    .is_some()
            })
            .collect()
    }

    fn instrument(&mut self, _site: &Site, emit: &mut Emitter<'_>) {
        let counter = emit.constant(COUNTER, 8);
        let count = emit.load(counter, 8);
        let one = emit.constant(1, 8);
        let next = emit.binop(IntBinop::Add, count, one);
        emit.store(counter, next);
    }
}

#[test]
fn a_store_hook_counts_stores_inline_without_leaving_the_vm() {
    let mut counts = Vec::new();
    for jit in [false, true] {
        let mut vm = machine(STORING_LOOP, jit);
        vm.memory_mut().mmu.write(COUNTER, &[0; 8]).unwrap();
        vm.add_hook(StoreCounter);
        let regs = drive_with(&mut vm, &["ECX", "EAX"], |_, code, args| {
            panic!("jit={jit}: the hook never stops: code {code}, args {args:?}");
        });
        assert_eq!(regs, vec![Some(0), Some(42)], "jit={jit}");
        let mut word = [0u8; 8];
        vm.memory().mmu.read(COUNTER, &mut word).unwrap();
        let count = u64::from_le_bytes(word);
        assert_eq!(count, 10, "jit={jit}: one tally per guest store");
        // The guest's own store landed as well.
        let mut last = [0u8; 4];
        vm.memory().mmu.read(0x21000, &mut last).unwrap();
        assert_eq!(u32::from_le_bytes(last), 1, "jit={jit}");
        counts.push(count);
    }
    assert_eq!(counts[0], counts[1]);
}

/// Where the gated entry hook reads its one-byte flag.
const FLAG: u64 = 0x21900;

/// Stops at a block's entry only while a guest flag is zero, so the host can
/// arrange to be told once and never again.
struct GatedEntry {
    address: u64,
    code: u64,
}

impl Hook for GatedEntry {
    fn sites(&mut self, block: &BlockView<'_>) -> Vec<Site> {
        block
            .entry()
            .filter(
                |site| matches!(site, Site::BlockEntry { address, .. } if *address == self.address),
            )
            .into_iter()
            .collect()
    }

    fn instrument(&mut self, _site: &Site, emit: &mut Emitter<'_>) {
        let flag = emit.load(emit.constant(FLAG, 8), 1);
        let zero = emit.constant(0, 1);
        let cond = emit.binop(IntBinop::Equal, flag, zero);
        let address = emit.constant(emit.address().unwrap_or_default(), 8);
        emit.interrupt_if(cond, self.code, &[address]);
    }
}

#[test]
fn an_entry_hook_gated_by_a_guest_flag_stops_only_once() {
    for jit in [false, true] {
        let mut vm = machine(LOOP, jit);
        vm.memory_mut().mmu.write(FLAG, &[0]).unwrap();
        vm.add_hook(GatedEntry {
            address: 0x1005,
            code: 13,
        });
        let mut stops = Vec::new();
        let regs = drive(&mut vm, &["ECX", "EAX"], |vm, code, addr| {
            assert_eq!((code, addr), (13, 0x1005), "jit={jit}");
            stops.push(addr);
            // Raise the flag in guest memory; the hook's own compare then
            // keeps the machine inside compiled code for every later entry.
            vm.memory_mut().mmu.write(FLAG, &[1]).unwrap();
        });
        assert_eq!(stops, vec![0x1005], "jit={jit}: the gate opens once");
        assert_eq!(
            regs,
            vec![Some(0), Some(42)],
            "jit={jit}: the loop ran to completion"
        );
        let mut flag = [0u8; 1];
        vm.memory().mmu.read(FLAG, &mut flag).unwrap();
        assert_eq!(flag[0], 1, "jit={jit}");
    }
}

/// Ten straight-line guest instructions and no branch: one basic block once
/// discovery has folded the chain together.
const STRAIGHT: &[u8] = &[
    0xb8, 0x39, 0x05, 0x00, 0x00, // 1000: mov eax, 1337
    0xbb, 0x07, 0x00, 0x00, 0x00, // 1005: mov ebx, 7
    0x01, 0xd8, // 100a: add eax, ebx
    0x01, 0xd8, // 100c: add eax, ebx
    0x29, 0xd8, // 100e: sub eax, ebx
    0x31, 0xd8, // 1010: xor eax, ebx
    0x01, 0xd8, // 1012: add eax, ebx
    0xff, 0xc0, // 1014: inc eax
    0xff, 0xc1, // 1016: inc ecx
    0xff, 0xc1, // 1018: inc ecx
];

/// [`GatedEntry`] on the entry of *every* block, which is how a tracing
/// front-end gates a whole program: whatever discovery decides a block is,
/// its first instruction is checked.
struct GatedEntries {
    code: u64,
}

impl Hook for GatedEntries {
    fn sites(&mut self, block: &BlockView<'_>) -> Vec<Site> {
        block.entry().into_iter().collect()
    }

    fn instrument(&mut self, _site: &Site, emit: &mut Emitter<'_>) {
        let flag = emit.load(emit.constant(FLAG, 8), 1);
        let zero = emit.constant(0, 1);
        let cond = emit.binop(IntBinop::Equal, flag, zero);
        let address = emit.constant(emit.address().unwrap_or_default(), 8);
        emit.interrupt_if(cond, self.code, &[address]);
    }
}

/// Runs [`STRAIGHT`] to the end of the mapped code, optionally behind a gate
/// on every block entry, and reports what the machine did.
fn straight_line_run(jit: bool, gated: bool) -> (Vec<Option<u64>>, qcode_vm::Stats) {
    let mut vm = machine(STRAIGHT, jit);
    vm.memory_mut().mmu.write(FLAG, &[0]).unwrap();
    if gated {
        vm.add_hook(GatedEntries { code: 13 });
    }
    let mut stops = 0;
    let regs = drive(&mut vm, &["EAX", "EBX", "ECX"], |vm, code, _| {
        assert_eq!(code, 13);
        stops += 1;
        // Raise the flag: every later gate is decided inside compiled code.
        vm.memory_mut().mmu.write(FLAG, &[1]).unwrap();
    });
    assert_eq!(
        stops,
        usize::from(gated),
        "jit={jit}: the gate stops once and only when installed"
    );
    (regs, vm.stats.clone())
}

/// A conditional hook splits the block it instruments, and the guest code
/// behind the split must go on being folded into one basic block. Left
/// unabsorbed, every guest instruction after a gate is its own block — its
/// own compilation, its own entry into compiled code and its own return —
/// which is the whole cost of discovering a straight-line run all over again.
#[test]
fn an_entry_gate_does_not_stop_the_run_from_absorbing_straight_line_code() {
    for jit in [false, true] {
        let (plain_regs, plain) = straight_line_run(jit, false);
        let (gated_regs, gated) = straight_line_run(jit, true);
        assert_eq!(
            gated_regs, plain_regs,
            "jit={jit}: the gate changed the run"
        );
        assert!(
            gated.absorbed > 0,
            "jit={jit}: nothing was absorbed behind the gate's split \
             (plain absorbed {}, gated {})",
            plain.absorbed,
            gated.absorbed
        );
        assert!(
            gated.absorbed + 1 >= plain.absorbed,
            "jit={jit}: the gate cost absorptions: plain {}, gated {}",
            plain.absorbed,
            gated.absorbed
        );
        // The gate adds its own blocks — the check, the interrupt's detour —
        // but not one per guest instruction.
        assert!(
            gated.native_bodies <= plain.native_bodies + 4,
            "jit={jit}: the gate multiplied the bodies run: plain {}, gated {}",
            plain.native_bodies,
            gated.native_bodies
        );
    }
}

/// Where the call test keeps its counters, inside the RW page `machine` maps.
const CALL_DATA: u64 = 0x20000;
/// A far address nothing maps: reaching it ends the run.
const SENTINEL: u64 = 0xdead_0000;

/// A callee reached both directly and indirectly, so the second call lands on
/// an address the first one's discovery folded into the caller's block:
///
/// ```text
/// 1000: mov r8, CALL_DATA
/// 1007: inc dword [r8 + 4]      ; the prologue, which runs once
/// 100b: lea rax, [f]
/// 1012: call f                  ; direct: f is absorbed into this block
/// 1017: call rax                ; indirect, to the absorbed address
/// 1019: jmp [rip]               ; to the sentinel
/// 1030: f: mov dword [r8], 0x22
/// 1037:    inc dword [r8 + 8]   ; runs twice
/// 103b:    ret
/// ```
fn calling_program() -> Vec<u8> {
    let mut code = vec![
        0x49, 0xc7, 0xc0, 0x00, 0x00, 0x02, 0x00, // mov r8, 0x20000
        0x41, 0xff, 0x40, 0x04, // inc dword [r8 + 4]
        0x48, 0x8d, 0x05, 0x1e, 0x00, 0x00, 0x00, // lea rax, [rip + 0x1e] = 0x1030
        0xe8, 0x19, 0x00, 0x00, 0x00, // call 0x1030
        0xff, 0xd0, // call rax
        0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // jmp [rip + 0]
    ];
    code.extend_from_slice(&SENTINEL.to_le_bytes());
    code.resize(0x30, 0x90);
    code.extend_from_slice(&[
        0x41, 0xc7, 0x00, 0x22, 0x00, 0x00, 0x00, // mov dword [r8], 0x22
        0x41, 0xff, 0x40, 0x08, // inc dword [r8 + 8]
        0xc3, // ret
    ]);
    code
}

/// `indirect_into_absorbed.rs`'s shape, with a gate on every block entry:
/// the run a conditional hook's split left behind absorbs the callee too, and
/// the indirect call must still enter at the callee rather than at the start
/// of the run, which is the caller's prologue.
#[test]
fn an_indirect_call_into_a_gated_absorbed_run_enters_at_the_callee() {
    for jit in [false, true] {
        for gated in [false, true] {
            let mut vm = machine(&calling_program(), jit);
            vm.memory_mut().mmu.write(FLAG, &[0]).unwrap();
            let ctx = vm.context().clone();
            vm.emulator()
                .set_varnode_by_name(&ctx, "RSP", 0x21800)
                .unwrap();
            if gated {
                vm.add_hook(GatedEntries { code: 13 });
            }
            let mut stops = 0;
            drive_with(&mut vm, &[], |vm, code, _| {
                assert_eq!(code, 13, "jit={jit} gated={gated}");
                stops += 1;
                vm.memory_mut().mmu.write(FLAG, &[1]).unwrap();
            });
            assert_eq!(stops, usize::from(gated), "jit={jit} gated={gated}");
            let mut bytes = [0u8; 12];
            vm.memory_mut().mmu.read(CALL_DATA, &mut bytes).unwrap();
            let word = |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
            assert_eq!(word(0), 0x22, "jit={jit} gated={gated}");
            assert_eq!(
                word(2),
                2,
                "jit={jit} gated={gated}: the callee runs once per call"
            );
            assert_eq!(
                word(1),
                1,
                "jit={jit} gated={gated}: the indirect call reran the caller"
            );
        }
    }
}

/// Where the back-branch program keeps its tally.
const TALLY: u64 = 0x20000;

/// A straight-line run with an *indirect* branch back into its own middle,
/// so the branch is only discovered once the run has been folded into one
/// block — and, behind a gate, into the tail the gate's split left:
///
/// ```text
/// 1000: mov ecx, 3
/// 1005: mov rbx, 0x100f
/// 100c: nop; nop; nop
/// 100f: inc dword [TALLY]
/// 1016: dec ecx
/// 1018: jz 101c
/// 101a: jmp rbx            ; indirect, into the middle of the run
/// 101c: mov eax, 42
/// ```
const BACK_BRANCH: &[u8] = &[
    0xb9, 0x03, 0x00, 0x00, 0x00, // 1000: mov ecx, 3
    0x48, 0xc7, 0xc3, 0x0f, 0x10, 0x00, 0x00, // 1005: mov rbx, 0x100f
    0x90, 0x90, 0x90, // 100c: nop nop nop
    0xff, 0x04, 0x25, 0x00, 0x00, 0x02, 0x00, // 100f: inc dword [0x20000]
    0xff, 0xc9, // 1016: dec ecx
    0x74, 0x02, // 1018: jz 0x101c
    0xff, 0xe3, // 101a: jmp rbx
    0xb8, 0x2a, 0x00, 0x00, 0x00, // 101c: mov eax, 42
];

#[test]
fn a_branch_into_the_middle_of_a_gated_run_enters_where_it_points() {
    for jit in [false, true] {
        let mut results = Vec::new();
        for gated in [false, true] {
            let mut vm = machine(BACK_BRANCH, jit);
            vm.memory_mut().mmu.write(FLAG, &[0]).unwrap();
            vm.memory_mut().mmu.write(TALLY, &[0; 4]).unwrap();
            if gated {
                vm.add_hook(GatedEntries { code: 13 });
            }
            let mut stops = 0;
            let regs = drive_with(&mut vm, &["EAX", "ECX"], |vm, code, _| {
                assert_eq!(code, 13, "jit={jit} gated={gated}");
                stops += 1;
                vm.memory_mut().mmu.write(FLAG, &[1]).unwrap();
            });
            assert_eq!(stops, usize::from(gated), "jit={jit} gated={gated}");
            let mut tally = [0u8; 4];
            vm.memory().mmu.read(TALLY, &mut tally).unwrap();
            assert_eq!(
                u32::from_le_bytes(tally),
                3,
                "jit={jit} gated={gated}: the branch re-entered the run at its target"
            );
            assert_eq!(regs, vec![Some(42), Some(0)], "jit={jit} gated={gated}");
            results.push((regs, vm.stats.absorbed));
        }
        assert_eq!(results[0].0, results[1].0, "jit={jit}");
        assert!(
            results[1].1 > 0,
            "jit={jit}: the gated run absorbed nothing"
        );
    }
}
