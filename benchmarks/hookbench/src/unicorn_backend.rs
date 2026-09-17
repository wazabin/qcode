//! The same images under Unicorn, through its Rust binding. Every hook is a
//! host callback: that is the only kind Unicorn has.

use std::{cell::Cell, rc::Rc};

use unicorn_engine::{Arch, HookType, MemType, Mode, Prot, RegisterX86, TcgOpCode, TcgOpFlag, Unicorn};

use crate::{Instr, Outcome, SENTINEL, STACK, STACK_SIZE, STACK_TOP, WATCH_LEN, elf};

pub fn run(image: &[u8], instr: Instr) -> Option<Outcome> {
    // Unicorn has no way to keep hook state in guest memory from the hook
    // itself, and no compare or instruction site that is not a callback.
    let kind = match instr {
        Instr::None => Instr::None,
        Instr::BlockCb => Instr::BlockCb,
        Instr::InsnCb => Instr::InsnCb,
        Instr::WatchIr | Instr::WatchCb => Instr::WatchCb,
        Instr::CmpCb => Instr::CmpCb,
        _ => return None,
    };
    let parsed = elf::parse(image);
    let watch = elf::first_rw(&parsed).expect("a writable segment");
    let mut uc = Unicorn::new(Arch::X86, Mode::MODE_64).expect("unicorn");
    for seg in &parsed.segments {
        let start = elf::page_down(seg.vaddr);
        let end = elf::page_up(seg.vaddr + seg.memsz.max(1));
        uc.mem_map(start, end - start, Prot::ALL).expect("map segment");
        uc.mem_write(seg.vaddr, &seg.data).expect("write segment");
    }
    uc.mem_map(STACK, STACK_SIZE, Prot::READ | Prot::WRITE).unwrap();
    uc.mem_write(STACK_TOP, &SENTINEL.to_le_bytes()).unwrap();
    // `until` is checked when a block at that address is about to run, which
    // needs the page to exist.
    uc.mem_map(elf::page_down(SENTINEL), 0x1000, Prot::ALL).unwrap();
    uc.reg_write(RegisterX86::RSP, STACK_TOP).unwrap();

    let host_calls = Rc::new(Cell::new(0u64));
    let events = Rc::new(Cell::new(0u64));
    let (hc, ev) = (host_calls.clone(), events.clone());
    match kind {
        Instr::None => {}
        Instr::BlockCb => {
            uc.add_block_hook(1, 0, move |_, _, _| {
                hc.set(hc.get() + 1);
                ev.set(ev.get() + 1);
            })
            .unwrap();
        }
        Instr::InsnCb => {
            uc.add_code_hook(1, 0, move |_, _, _| {
                hc.set(hc.get() + 1);
                ev.set(ev.get() + 1);
            })
            .unwrap();
        }
        Instr::WatchCb => {
            uc.add_mem_hook(HookType::MEM_WRITE, watch, watch + WATCH_LEN - 1, move |_, _: MemType, _, _, _| {
                hc.set(hc.get() + 1);
                ev.set(ev.get() + 1);
                true
            })
            .unwrap();
        }
        Instr::CmpCb => {
            uc.add_tcg_hook(TcgOpCode::SUB, TcgOpFlag::CMP, 1, 0, move |_, _, _, _, _| {
                hc.set(hc.get() + 1);
                ev.set(ev.get() + 1);
            })
            .unwrap();
        }
        _ => unreachable!(),
    }

    let started = std::time::Instant::now();
    let result = uc.emu_start(parsed.entry, SENTINEL, 0, 0);
    let elapsed = started.elapsed();
    let rip = uc.reg_read(RegisterX86::RIP).unwrap();
    let rax = uc.reg_read(RegisterX86::RAX).unwrap();
    let finished = result.is_ok() && rip == SENTINEL;
    Some(Outcome {
        verified: finished && rax as u32 == 0,
        exit: format!("{result:?} rip={rip:#x}"),
        elapsed,
        host_calls: host_calls.get(),
        events: events.get(),
        sites: 0,
    })
}
