//! The Embench images under QCode, with each instrumentation written two
//! ways where both make sense: as IR the machine compiles, and as a host
//! callback the machine stops for.

use std::{cell::Cell, rc::Rc};

use qcode::value::insn::IntBinop;
use qcode_jit::Jit;
use qcode_userland::bare;
use qcode_vm::{
    HookAction, InterruptKind, Vm, VmExit, perm,
    hook::{BlockView, Emitter, Hook, Site},
};

use crate::{Instr, Outcome, WATCH_LEN, elf};

/// Guest memory the compiled instrumentation keeps its state in.
const SCRATCH: u64 = 0x7ff0_0000;
const SCRATCH_SIZE: u64 = 0x30000;
/// A 64-bit counter.
const COUNTER: u64 = SCRATCH;
/// The previous block id, for the edge map.
const PREV: u64 = SCRATCH + 8;
/// The comparison log's write index.
const CMP_IDX: u64 = SCRATCH + 16;
/// 64 KiB of 8-bit edge counters.
const EDGE_MAP: u64 = SCRATCH + 0x1000;
const EDGE_MASK: u64 = 0xffff;
/// 4096 entries of (lhs, rhs).
const CMP_LOG: u64 = SCRATCH + 0x11000;
const CMP_MASK: u64 = 0xfff;

/// Interrupt code the raw injector hooks stop with.
const CODE: u64 = 7;

/// Wraps a hook to count the sites it instruments.
struct Counted<H> {
    hook: H,
    sites: Rc<Cell<u64>>,
}

impl<H: Hook> Hook for Counted<H> {
    fn sites(&mut self, block: &BlockView<'_>) -> Vec<Site> {
        self.hook.sites(block)
    }
    fn instrument(&mut self, site: &Site, emit: &mut Emitter<'_>) {
        self.sites.set(self.sites.get() + 1);
        self.hook.instrument(site, emit);
    }
}

/// `counter += 1` in guest memory at every block entry or every instruction.
struct CounterIr {
    per_instruction: bool,
    /// Keep the counter in a flat hook space rather than guest RAM.
    flat: bool,
}

fn bump(emit: &mut Emitter<'_>, at: u64) {
    let p = emit.constant(at, 8);
    let v = emit.load(p, 8);
    let one = emit.constant(1, 8);
    let v1 = emit.binop(IntBinop::Add, v, one);
    emit.store(v1, p);
}

impl Hook for CounterIr {
    fn sites(&mut self, block: &BlockView<'_>) -> Vec<Site> {
        if self.per_instruction {
            block.addresses()
        } else {
            block.entry().into_iter().collect()
        }
    }
    fn instrument(&mut self, _site: &Site, emit: &mut Emitter<'_>) {
        if self.flat {
            // The counter in a flat space of the hook's own, at offset 0.
            let space = emit.state_space("hook");
            let p = emit.constant(0, 8);
            let v = emit.load_from(space, p, 8);
            let one = emit.constant(1, 8);
            let v1 = emit.binop(IntBinop::Add, v, one);
            emit.store_to(space, v1, p);
        } else {
            bump(emit, COUNTER);
        }
    }
}

/// AFL's edge map: `map[(cur ^ prev) & MASK] += 1; prev = cur >> 1`.
struct EdgeIr;

impl Hook for EdgeIr {
    fn sites(&mut self, block: &BlockView<'_>) -> Vec<Site> {
        block.entry().into_iter().collect()
    }
    fn instrument(&mut self, site: &Site, emit: &mut Emitter<'_>) {
        let Site::BlockEntry { address, .. } = site else { return };
        // A block id from its address, spread over the map.
        let cur = ((address >> 4) ^ (address >> 12) ^ (address * 0x9e37_79b9)) & EDGE_MASK;
        let prev_p = emit.constant(PREV, 8);
        let prev = emit.load(prev_p, 8);
        let cur_v = emit.constant(cur, 8);
        let idx = emit.binop(IntBinop::Xor, prev, cur_v);
        let mask = emit.constant(EDGE_MASK, 8);
        let idx = emit.binop(IntBinop::And, idx, mask);
        let base = emit.constant(EDGE_MAP, 8);
        let p = emit.binop(IntBinop::Add, base, idx);
        let b = emit.load(p, 1);
        let one = emit.constant(1, 1);
        let b1 = emit.binop(IntBinop::Add, b, one);
        emit.store(b1, p);
        let next = emit.constant(cur >> 1, 8);
        emit.store(next, prev_p);
    }
}

/// The host at every store, with the address: Unicorn's memory hook shape.
struct StoreCb;

impl Hook for StoreCb {
    fn sites(&mut self, block: &BlockView<'_>) -> Vec<Site> {
        block.stores()
    }
    fn instrument(&mut self, _site: &Site, emit: &mut Emitter<'_>) {
        let Some((ptr, size, _)) = emit.store_operands() else { return };
        let ptr = emit.zext(ptr, 8);
        let size = emit.constant(size as u64, 8);
        emit.interrupt(CODE, &[ptr, size]);
    }
}

/// The host at every comparison, with the operands widened to 64 bits.
struct CmpCb;

impl Hook for CmpCb {
    fn sites(&mut self, block: &BlockView<'_>) -> Vec<Site> {
        block.compares()
    }
    fn instrument(&mut self, _site: &Site, emit: &mut Emitter<'_>) {
        let Some((_, lhs, rhs)) = emit.binop_operands() else { return };
        let lhs = emit.zext(lhs, 8);
        let rhs = emit.zext(rhs, 8);
        emit.interrupt(CODE, &[lhs, rhs]);
    }
}

/// Every comparison's operands appended to a ring buffer in guest memory.
struct CmpLogIr;

impl Hook for CmpLogIr {
    fn sites(&mut self, block: &BlockView<'_>) -> Vec<Site> {
        block.compares()
    }
    fn instrument(&mut self, _site: &Site, emit: &mut Emitter<'_>) {
        let Some((_, lhs, rhs)) = emit.binop_operands() else { return };
        let lhs = emit.zext(lhs, 8);
        let rhs = emit.zext(rhs, 8);
        let idx_p = emit.constant(CMP_IDX, 8);
        let idx = emit.load(idx_p, 8);
        let four = emit.constant(4, 8);
        let off = emit.binop(IntBinop::ShiftLeft, idx, four);
        let base = emit.constant(CMP_LOG, 8);
        let p = emit.binop(IntBinop::Add, base, off);
        emit.store(lhs, p);
        let eight = emit.constant(8, 8);
        let p8 = emit.binop(IntBinop::Add, p, eight);
        emit.store(rhs, p8);
        let one = emit.constant(1, 8);
        let idx1 = emit.binop(IntBinop::Add, idx, one);
        let mask = emit.constant(CMP_MASK, 8);
        let idx1 = emit.binop(IntBinop::And, idx1, mask);
        emit.store(idx1, idx_p);
    }
}

fn read_u64(vm: &Vm<wazabin_qcode_sleigh::vm_source::SleighCodeSource<'static>>, at: u64) -> u64 {
    let mut b = [0u8; 8];
    vm.memory().mmu.read(at, &mut b).expect("scratch is mapped");
    u64::from_le_bytes(b)
}

pub fn run(image: &[u8], jit: bool, instr: Instr) -> Outcome {
    let parsed = elf::parse(image);
    let watch = elf::first_rw(&parsed).expect("a writable segment");
    let mut vm = bare::machine(image).expect("the image loads");
    if jit {
        vm.set_block_executor(Box::new(Jit::new()));
    }
    vm.memory_mut()
        .mmu
        .map(SCRATCH, SCRATCH_SIZE, perm::RW_INIT)
        .unwrap();

    let sites = Rc::new(Cell::new(0u64));
    let host_calls = Rc::new(Cell::new(0u64));
    let events = Rc::new(Cell::new(0u64));
    let (hc, ev) = (host_calls.clone(), events.clone());
    let counted = |hook: Box<dyn Hook>| Counted { hook: Dyn(hook), sites: sites.clone() };
    match instr {
        Instr::None => {}
        Instr::BlockIr => vm.add_hook(counted(Box::new(CounterIr { per_instruction: false, flat: true }))),
        Instr::BlockRam => vm.add_hook(counted(Box::new(CounterIr { per_instruction: false, flat: false }))),
        Instr::InsnIr => vm.add_hook(counted(Box::new(CounterIr { per_instruction: true, flat: true }))),
        Instr::InsnRam => vm.add_hook(counted(Box::new(CounterIr { per_instruction: true, flat: false }))),
        Instr::EdgeIr => vm.add_hook(counted(Box::new(EdgeIr))),
        Instr::CmpIr => vm.add_hook(counted(Box::new(CmpLogIr))),
        Instr::WatchCb => vm.add_hook(counted(Box::new(StoreCb))),
        Instr::CmpCb => vm.add_hook(counted(Box::new(CmpCb))),
        Instr::BlockCb => {
            vm.hook_block(1, 0, move |_, _| {
                hc.set(hc.get() + 1);
                ev.set(ev.get() + 1);
                HookAction::Continue
            });
        }
        Instr::InsnCb => {
            vm.hook_code(1, 0, move |_, _| {
                hc.set(hc.get() + 1);
                ev.set(ev.get() + 1);
                HookAction::Continue
            });
        }
        Instr::WatchIr => {
            vm.hook_mem_write(watch, watch + WATCH_LEN - 1, move |_, _| {
                hc.set(hc.get() + 1);
                ev.set(ev.get() + 1);
                HookAction::Continue
            });
        }
    }

    let started = std::time::Instant::now();
    let exit = loop {
        let exit = vm.run(1 << 50);
        if std::env::var_os("HOOKBENCH_STATS").is_some() {
            eprintln!("exit {exit:?} steps={} pc={:?}", vm.stats.steps, vm.pc());
        }
        match &exit {
            VmExit::Interrupt(i) if matches!(i.kind, InterruptKind::Explicit { code } if code == CODE) => {
                host_calls.set(host_calls.get() + 1);
                if std::env::var_os("HOOKBENCH_TRACE").is_some() {
                    eprintln!("cb pc={:x?} args={:x?}", i.pc, i.args);
                }
                match instr {
                    Instr::WatchCb => {
                        let addr = i.args[0].unwrap_or(0);
                        let size = i.args[1].unwrap_or(0);
                        if addr < watch + WATCH_LEN && addr + size > watch {
                            events.set(events.get() + 1);
                        }
                    }
                    _ => events.set(events.get() + 1),
                }
                vm.resume(None).expect("resumable");
            }
            _ => break exit,
        }
    };
    let elapsed = started.elapsed();
    let finished = bare::returned(&exit);
    let verified = finished && bare::register(&mut vm, "EAX") == Some(0);
    match instr {
        Instr::BlockIr | Instr::InsnIr => {
            let space = vm.context().try_get_space("hook").expect("the hook space exists");
            let v = vm
                .memory_mut()
                .flat_mut()
                .read_u128(qcode::space::MemorySpaceId::Shared(space), 0, 8)
                .unwrap_or(0);
            events.set(v as u64);
        }
        Instr::BlockRam | Instr::InsnRam => events.set(read_u64(&vm, COUNTER)),
        Instr::EdgeIr => {
            let mut map = vec![0u8; (EDGE_MASK + 1) as usize];
            vm.memory().mmu.read(EDGE_MAP, &mut map).unwrap();
            events.set(map.iter().filter(|&&b| b != 0).count() as u64);
        }
        Instr::CmpIr => events.set(read_u64(&vm, CMP_IDX)),
        _ => {}
    }
    if let Some(path) = std::env::var_os("HOOKBENCH_DUMP") {
        std::fs::write(path, format!("{}", vm.context())).unwrap();
    }
    if std::env::var_os("HOOKBENCH_STATS").is_some() {
        eprintln!(
            "stats {:?}: steps={} lifts={} absorbed={} native_bodies={} evicted={}",
            instr, vm.stats.steps, vm.stats.lifts, vm.stats.absorbed, vm.stats.native_bodies, vm.stats.evicted
        );
    }
    Outcome {
        verified,
        exit: format!("{exit:?}"),
        elapsed,
        host_calls: host_calls.get(),
        events: events.get(),
        sites: sites.get(),
    }
}

/// One wrapper type over any hook.
struct Dyn(Box<dyn Hook>);

impl Hook for Dyn {
    fn sites(&mut self, block: &BlockView<'_>) -> Vec<Site> {
        self.0.sites(block)
    }
    fn instrument(&mut self, site: &Site, emit: &mut Emitter<'_>) {
        self.0.instrument(site, emit)
    }
}
