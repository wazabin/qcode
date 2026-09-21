//! The same images under icicle-emu. Block and instruction counters are
//! p-code an injector adds, as icicle's README shows; the memory watch is
//! an MMU hook; the block callback is an `Op::Hook` the JIT calls out to.

use std::{cell::Cell, rc::Rc};

use icicle_vm::{
    BlockTable, CodeInjector, Vm, VmExit,
    cpu::{BlockGroup, Config, Cpu, ExceptionCode, Mmu, mem::{Mapping, perm}},
};

use crate::{Instr, Outcome, SENTINEL, STACK, STACK_SIZE, STACK_TOP, WATCH_LEN, elf};

struct CounterInjector {
    store: u16,
    per_instruction: bool,
    hook: Option<pcode::HookId>,
    sites: Rc<Cell<u64>>,
}

impl CodeInjector for CounterInjector {
    fn inject(&mut self, _cpu: &mut Cpu, group: &BlockGroup, code: &mut BlockTable) {
        for idx in group.range() {
            let block = &mut code.blocks[idx];
            let mut out: Vec<pcode::Instruction> = Vec::with_capacity(block.pcode.instructions.len() + 8);
            let emit = |pc: &mut pcode::Block, out: &mut Vec<pcode::Instruction>| {
                match self.hook {
                    Some(id) => out.push(pcode::Op::Hook(id).into()),
                    None => {
                        let tmp = pc.alloc_tmp(8);
                        out.push((tmp, pcode::Op::Load(self.store), 0_u64).into());
                        out.push((tmp, pcode::Op::IntAdd, (tmp, 1_u64)).into());
                        out.push((pcode::Op::Store(self.store), (0_u64, tmp)).into());
                    }
                }
            };
            let insns = std::mem::take(&mut block.pcode.instructions);
            if !self.per_instruction {
                self.sites.set(self.sites.get() + 1);
                emit(&mut block.pcode, &mut out);
                out.extend(insns);
            } else {
                for insn in insns {
                    let marker = matches!(insn.op, pcode::Op::InstructionMarker);
                    out.push(insn);
                    if marker {
                        self.sites.set(self.sites.get() + 1);
                        emit(&mut block.pcode, &mut out);
                    }
                }
            }
            block.pcode.instructions = out;
            block.pcode.recompute_next_tmp();
            code.modified.insert(idx);
        }
    }
}

pub fn run(image: &[u8], instr: Instr) -> Option<Outcome> {
    let kind = match instr {
        Instr::None | Instr::BlockIr | Instr::BlockCb | Instr::InsnIr | Instr::InsnCb => instr,
        Instr::BlockRam => Instr::BlockIr,
        Instr::InsnRam => Instr::InsnIr,
        Instr::WatchIr | Instr::WatchCb => Instr::WatchCb,
        _ => return None,
    };
    let parsed = elf::parse(image);
    let watch = elf::first_rw(&parsed).expect("a writable segment");
    let mut vm: Vm = icicle_vm::build(&Config {
        triple: "x86_64-none".parse().unwrap(),
        enable_shadow_stack: false,
        ..Config::default()
    })
    .expect("icicle x86-64");
    for seg in &parsed.segments {
        let start = elf::page_down(seg.vaddr);
        let end = elf::page_up(seg.vaddr + seg.memsz.max(1));
        assert!(vm.cpu.mem.map_memory_len(
            start,
            end - start,
            Mapping { perm: perm::READ | perm::WRITE | perm::EXEC | perm::INIT, value: 0 }
        ));
        vm.cpu.mem.write_bytes(seg.vaddr, &seg.data, perm::NONE).unwrap();
    }
    assert!(vm.cpu.mem.map_memory_len(
        STACK,
        STACK_SIZE,
        Mapping { perm: perm::READ | perm::WRITE | perm::INIT, value: 0 }
    ));
    vm.cpu.mem.write_bytes(STACK_TOP, &SENTINEL.to_le_bytes(), perm::NONE).unwrap();
    let rsp = vm.cpu.arch.sleigh.get_varnode("RSP").unwrap();
    let rax = vm.cpu.arch.sleigh.get_varnode("RAX").unwrap();
    vm.cpu.write_reg(rsp, STACK_TOP);
    vm.cpu.write_pc(parsed.entry);

    let host_calls = Rc::new(Cell::new(0u64));
    let events = Rc::new(Cell::new(0u64));
    let sites = Rc::new(Cell::new(0u64));
    let counter = vm.cpu.trace.register_store(vec![0_u64]);
    match kind {
        Instr::None => {}
        Instr::BlockIr | Instr::InsnIr => {
            vm.add_injector(CounterInjector {
                store: counter.get_store_id(),
                per_instruction: kind == Instr::InsnIr,
                hook: None,
                sites: sites.clone(),
            });
        }
        Instr::BlockCb | Instr::InsnCb => {
            let (hc, ev) = (host_calls.clone(), events.clone());
            let id = vm.cpu.add_hook(move |_: &mut Cpu, _: u64| {
                hc.set(hc.get() + 1);
                ev.set(ev.get() + 1);
            });
            vm.add_injector(CounterInjector {
                store: counter.get_store_id(),
                per_instruction: kind == Instr::InsnCb,
                hook: Some(id),
                sites: sites.clone(),
            });
        }
        Instr::WatchCb => {
            let (hc, ev) = (host_calls.clone(), events.clone());
            vm.cpu.mem.add_write_hook(
                watch,
                watch + WATCH_LEN,
                Box::new(move |_: &mut Mmu, _: u64, _: &[u8]| {
                    hc.set(hc.get() + 1);
                    ev.set(ev.get() + 1);
                }),
            );
        }
        _ => unreachable!(),
    }

    let started = crate::Stopwatch::start();
    let exit = vm.run();
    let elapsed = started.elapsed();
    let pc = vm.cpu.read_pc();
    let finished = matches!(exit, VmExit::UnhandledException((ExceptionCode::ExecViolation, addr)) if addr == SENTINEL)
        || (pc == SENTINEL && !matches!(exit, VmExit::InstructionLimit));
    let result = vm.cpu.read_reg(rax);
    if matches!(kind, Instr::BlockIr | Instr::InsnIr) {
        let data = vm.cpu.trace[counter].as_any().downcast_ref::<Vec<u64>>().unwrap();
        events.set(data[0]);
    }
    Some(Outcome {
        verified: finished && result as u32 == 0,
        exit: format!("{exit:?} pc={pc:#x}"),
        elapsed,
        host_calls: host_calls.get(),
        events: events.get(),
        sites: sites.get(),
    })
}
