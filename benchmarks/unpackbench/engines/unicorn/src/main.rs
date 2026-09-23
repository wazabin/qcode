//! The unpack task under Unicorn (Rust binding, `~/dev/unicorn`).
//!
//! Every hook is a host callback, the only kind Unicorn has: one per guest
//! store into either provenance window (`UC_HOOK_MEM_WRITE`, range-limited),
//! one per block entry (`UC_HOOK_BLOCK`). The loader and the system calls are
//! `unpackbase`'s, entered from a `UC_HOOK_INSN` on `syscall`.
//!
//! Unicorn has one execution strategy, QEMU's TCG: `--interp` is ignored and
//! the run reports `jit`.

use std::time::Instant;

use rustc_hash::FxHashMap;
use serde_json::json;
use unicorn_engine::{Arch, HookType, MemType, Mode, Prot, RegisterX86, Unicorn, X86Insn};
use unpackbase::{
    Guest, Outcome, Reg,
    cli::Cli,
    harvest::Run,
    kernel::{Action, Kernel},
    record::{Block, Layout, MAX_BLOCKS, Record, SHADOW_LEN, Site},
};

struct State {
    kernel: Option<Kernel>,
    action: Option<Action>,
    layout: Layout,
    edges: bool,
    rec: Record,
    site_of: FxHashMap<u64, u16>,
    block_of: FxHashMap<u64, u32>,
    last: Option<u32>,
    fault: Option<(u64, u64)>,
}

struct UcGuest<'a, 'b>(&'a mut Unicorn<'b, State>);

fn reg_id(r: Reg) -> RegisterX86 {
    match r {
        Reg::Rax => RegisterX86::RAX,
        Reg::Rdi => RegisterX86::RDI,
        Reg::Rsi => RegisterX86::RSI,
        Reg::Rdx => RegisterX86::RDX,
        Reg::R10 => RegisterX86::R10,
        Reg::R8 => RegisterX86::R8,
        Reg::R9 => RegisterX86::R9,
        Reg::Rsp => RegisterX86::RSP,
        Reg::Rip => RegisterX86::RIP,
        Reg::FsBase => RegisterX86::FS_BASE,
        Reg::GsBase => RegisterX86::GS_BASE,
    }
}

impl Guest for UcGuest<'_, '_> {
    fn read(&mut self, addr: u64, buf: &mut [u8]) -> bool {
        self.0.mem_read(addr, buf).is_ok()
    }
    fn write(&mut self, addr: u64, data: &[u8]) -> bool {
        self.0.mem_write(addr, data).is_ok()
    }
    fn map(&mut self, addr: u64, len: u64) -> bool {
        self.0.mem_map(addr, len, Prot::ALL).is_ok()
    }
    fn unmap(&mut self, addr: u64, len: u64) {
        let _ = self.0.mem_unmap(addr, len);
    }
    fn reg(&mut self, r: Reg) -> u64 {
        self.0.reg_read(reg_id(r)).unwrap_or(0)
    }
    fn set_reg(&mut self, r: Reg, v: u64) {
        let _ = self.0.reg_write(reg_id(r), v);
    }
}

// HOOK-BEGIN
/// A guest store into a window: stamp its site id over the bytes written.
fn on_store(uc: &mut Unicorn<'_, State>, addr: u64, size: usize) {
    let pc = uc.reg_read(RegisterX86::RIP).unwrap_or(0);
    let s = uc.get_data_mut();
    let id = match s.site_of.get(&pc) {
        Some(&id) => id,
        None => {
            let n = s.rec.sites.len() + 1;
            s.rec.sites_saturated |= n > u16::MAX as usize;
            let id = n.min(u16::MAX as usize) as u16;
            s.rec.sites.push(Site { pc, size });
            s.site_of.insert(pc, id);
            id
        }
    };
    let at = (s.layout.shadow_of(addr) / 2) as usize;
    s.rec.shadow[at..at + size].fill(id);
}

/// A block entry: log the first one, and its predecessor with `--edges`.
fn on_block(uc: &mut Unicorn<'_, State>, addr: u64, size: u32) {
    let s = uc.get_data_mut();
    let k = match s.block_of.get(&addr) {
        Some(&k) => k,
        None => {
            if s.rec.blocks.len() as u64 >= MAX_BLOCKS {
                s.rec.blocks_saturated = true;
                return;
            }
            let k = s.rec.blocks.len() as u32;
            s.rec.blocks.push(Block { addr, end: addr + size as u64 });
            s.block_of.insert(addr, k);
            s.rec.log.push((k, s.last));
            k
        }
    };
    if s.edges {
        s.last = Some(k);
    }
}
// HOOK-END

fn run(cli: &Cli, bytes: &[u8]) -> Result<Outcome, String> {
    let started = Instant::now();
    let state = State {
        kernel: None,
        action: None,
        layout: Layout::at(0, 0),
        edges: cli.edges,
        rec: Record::default(),
        site_of: FxHashMap::default(),
        block_of: FxHashMap::default(),
        last: None,
        fault: None,
    };
    let mut uc = Unicorn::new_with_data(Arch::X86, Mode::MODE_64, state).map_err(|e| format!("unicorn: {e:?}"))?;
    let argv: Vec<String> = std::iter::once(cli.elf.clone()).chain(cli.args.iter().cloned()).collect();
    let kernel = Kernel::boot(&mut UcGuest(&mut uc), &cli.elf, bytes, &argv, cli.stdin.clone())?;
    let layout = Layout::new(kernel.image_lo, kernel.image_hi)?;
    let entry = kernel.entry;
    {
        let s = uc.get_data_mut();
        s.layout = layout;
        s.kernel = Some(kernel);
        if cli.hooks {
            s.rec.shadow = vec![0u16; SHADOW_LEN / 2];
        }
    }

    uc.add_insn_sys_hook(X86Insn::SYSCALL, 1, 0, |uc| {
        let mut kernel = uc.get_data_mut().kernel.take().expect("the kernel");
        let action = kernel.syscall(&mut UcGuest(uc));
        uc.get_data_mut().kernel = Some(kernel);
        if action != Action::Continue {
            uc.get_data_mut().action = Some(action);
            let _ = uc.emu_stop();
        }
    })
    .map_err(|e| format!("syscall hook: {e:?}"))?;
    // Only ever fires on a fault: names the address in the stop reason.
    uc.add_mem_hook(HookType::MEM_INVALID, 1, 0, |uc, _: MemType, addr, _, _| {
        let pc = uc.reg_read(RegisterX86::RIP).unwrap_or(0);
        uc.get_data_mut().fault = Some((addr, pc));
        false
    })
    .map_err(|e| format!("fault hook: {e:?}"))?;
    if cli.hooks {
        for w in layout.windows() {
            uc.add_mem_hook(HookType::MEM_WRITE, w.start, w.end() - 1, |uc, _: MemType, addr, size, _| {
                on_store(uc, addr, size);
                true
            })
            .map_err(|e| format!("store hook: {e:?}"))?;
        }
        uc.add_block_hook(1, 0, on_block).map_err(|e| format!("block hook: {e:?}"))?;
    }

    let setup_ms = started.elapsed().as_secs_f64() * 1e3;
    let started = Instant::now();
    let result = uc.emu_start(entry, 0, 0, cli.budget.unwrap_or(0) as usize);
    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
    let rip = uc.reg_read(RegisterX86::RIP).unwrap_or(0);
    let s = uc.get_data_mut();
    let action = s.action.take();
    let fault = s.fault.map(|(a, pc)| format!(" (access {a:#x} from {pc:#x})")).unwrap_or_default();
    let kernel = s.kernel.take().expect("the kernel");
    let rec = std::mem::take(&mut s.rec);
    let (exit, stop_reason, crashed, unsupported) = match (&action, &result) {
        (Some(Action::Exit(c)), _) => (Some(*c), "exit".to_string(), false, None),
        (Some(Action::Unsupported(r)), _) => (None, format!("unsupported: {r}"), true, Some(r.clone())),
        (_, Err(e)) => (None, format!("crashed: {e:?} at {rip:#x}{fault}"), true, None),
        (_, Ok(())) if cli.budget.is_some() => (None, format!("budget of {} instructions exhausted", cli.budget.unwrap()), true, None),
        (_, Ok(())) => (None, format!("stopped at {rip:#x} without exiting"), true, None),
    };
    let mut warnings = Vec::new();
    if !kernel.enosys.is_empty() {
        warnings.push(format!("system calls answered ENOSYS: {:?}", kernel.enosys));
    }
    let extra = json!({"syscalls": kernel.syscalls});
    let run = Run {
        engine: "unicorn".into(),
        strategy: "jit".into(),
        path: cli.elf.clone(),
        entry,
        exit,
        stop_reason,
        crashed,
        stdout: kernel.stdout,
        stderr: kernel.stderr,
        steps: None,
        layout,
        hooks: cli.hooks,
        edges: cli.edges,
        record: cli.hooks.then_some(rec),
        warnings,
    };
    let read = Box::new(move |addr: u64, buf: &mut [u8]| uc.mem_read(addr, buf).is_ok());
    Ok(Outcome { run, wall_ms, setup_ms, read, unsupported, extra })
}

fn main() {
    let lines = unpackbase::hook_lines(&[include_str!("main.rs")]);
    unpackbase::drive("unicorn", lines, run);
}
