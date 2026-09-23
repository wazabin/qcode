//! The unpack task under icicle-emu (`~/dev/icicle-emu`).
//!
//! icicle's fastest hook is injected p-code, so that is what this driver
//! uses: a [`CodeInjector`] rewrites every lifted block group, putting
//! before each guest store the p-code that stamps the site id into a shadow
//! trace store, and at the head of each group the p-code that appends its
//! first entry to a log — the same branchless shapes as the unpack example's
//! compiled hooks, and no host callback at run time. The JIT (Cranelift)
//! compiles them with the guest code; `--interp` runs icicle's interpreter.
//!
//! The loader and the system calls are `unpackbase`'s, behind an icicle
//! [`Environment`] that services `ExceptionCode::Syscall`. icicle needs
//! `GHIDRA_SRC` pointing at a directory holding `Ghidra/Processors`.

use std::time::Instant;

use icicle_vm::{
    BlockTable, CodeInjector, Vm, VmExit,
    cpu::{
        BlockGroup, Config, Cpu, Environment, ExceptionCode,
        cpu::Exception,
        debug_info::DebugInfo,
        mem::{Mapping, perm},
    },
};
use pcode::{Op, Value, VarNode};
use rustc_hash::FxHashMap;
use serde_json::json;
use unpackbase::{
    Guest, Outcome, Reg,
    cli::Cli,
    harvest::Run,
    kernel::{Action, Kernel},
    record::{Block, Layout, MAX_BLOCKS, Record, SHADOW_LEN, Site},
};

const ENTRIES_LEN: usize = (16 + 8 * (MAX_BLOCKS + 1)) as usize;
const NO_PRED: u64 = u32::MAX as u64;

struct Regs {
    rax: VarNode,
    rdi: VarNode,
    rsi: VarNode,
    rdx: VarNode,
    r10: VarNode,
    r8: VarNode,
    r9: VarNode,
    rsp: VarNode,
    fs: VarNode,
    gs: VarNode,
}

struct CpuGuest<'a>(&'a mut Cpu, &'a Regs);

impl Guest for CpuGuest<'_> {
    fn read(&mut self, addr: u64, buf: &mut [u8]) -> bool {
        self.0.mem.read_bytes_large(addr, buf, perm::NONE).is_ok()
    }
    fn write(&mut self, addr: u64, data: &[u8]) -> bool {
        self.0.mem.write_bytes_large(addr, data, perm::NONE).is_ok()
    }
    fn map(&mut self, addr: u64, len: u64) -> bool {
        let p = perm::READ | perm::WRITE | perm::EXEC | perm::INIT;
        self.0.mem.map_memory_len(addr, len, Mapping { perm: p, value: 0 })
    }
    fn unmap(&mut self, addr: u64, len: u64) {
        self.0.mem.unmap_memory_len(addr, len);
    }
    fn reg(&mut self, r: Reg) -> u64 {
        match r {
            Reg::Rip => self.0.read_pc(),
            r => self.0.read_reg(var(self.1, r)),
        }
    }
    fn set_reg(&mut self, r: Reg, v: u64) {
        match r {
            Reg::Rip => self.0.write_pc(v),
            r => self.0.write_reg(var(self.1, r), v),
        }
    }
}

fn var(regs: &Regs, r: Reg) -> VarNode {
    match r {
        Reg::Rax => regs.rax,
        Reg::Rdi => regs.rdi,
        Reg::Rsi => regs.rsi,
        Reg::Rdx => regs.rdx,
        Reg::R10 => regs.r10,
        Reg::R8 => regs.r8,
        Reg::R9 => regs.r9,
        Reg::Rsp => regs.rsp,
        Reg::FsBase => regs.fs,
        Reg::GsBase => regs.gs,
        Reg::Rip => unreachable!(),
    }
}

/// The Linux the guest sees: `unpackbase`'s kernel on `syscall`.
struct Linux {
    kernel: Kernel,
    regs: Regs,
    action: Option<Action>,
    debug_info: DebugInfo,
}

impl Environment for Linux {
    fn load(&mut self, _: &mut Cpu, _: &[u8]) -> Result<(), String> {
        Ok(())
    }

    fn handle_exception(&mut self, cpu: &mut Cpu) -> Option<VmExit> {
        if ExceptionCode::from_u32(cpu.exception.code) != ExceptionCode::Syscall {
            return None;
        }
        match self.kernel.syscall(&mut CpuGuest(cpu, &self.regs)) {
            Action::Continue => {
                // As icicle-linux resumes: at the instruction after `syscall`.
                let next = cpu.read_reg(cpu.arch.reg_next_pc);
                cpu.exception = Exception::new(ExceptionCode::ExternalAddr, next);
                None
            }
            other => {
                self.action = Some(other);
                Some(VmExit::Halt)
            }
        }
    }

    fn debug_info(&self) -> Option<&DebugInfo> {
        Some(&self.debug_info)
    }

    fn snapshot(&mut self) -> Box<dyn std::any::Any> {
        Box::new(())
    }

    fn restore(&mut self, _: &Box<dyn std::any::Any>) {}
}

/// What the injector knows that the trace stores do not: what each site id
/// and each block index mean.
#[derive(Default)]
struct Meaning {
    sites: Vec<Site>,
    site_of: FxHashMap<u64, u16>,
    blocks: Vec<Block>,
    sites_saturated: bool,
    blocks_saturated: bool,
}

struct Unpack {
    layout: Layout,
    edges: bool,
    shadow: u16,
    entries: u16,
    visited: u16,
    meaning: std::rc::Rc<std::cell::RefCell<Meaning>>,
}

// HOOK-BEGIN
impl Unpack {
    /// Before a guest store of `size` bytes at `addr`: stamp `id` over the
    /// shadow of the bytes written, into the window that holds `addr` or the
    /// sink, without a branch.
    fn stamp(&self, b: &mut pcode::Block, out: &mut Vec<pcode::Instruction>, addr: Value, size: u64, id: u16) {
        let a = match addr {
            Value::Var(v) if v.size == 8 => Value::Var(v),
            other => {
                let t = b.alloc_tmp(8);
                out.push(t.zext_from(other));
                Value::Var(t)
            }
        };
        let dst = b.alloc_tmp(8);
        out.push((dst, Op::Copy, 0_u64).into());
        for w in self.layout.windows() {
            let off = b.alloc_tmp(8);
            let inside = b.alloc_tmp(1);
            let mask = b.alloc_tmp(8);
            out.push((off, Op::IntSub, (a, w.start)).into());
            out.push((inside, Op::IntLess, (off, w.len)).into());
            out.push((mask, Op::ZeroExtend, inside).into());
            out.push((mask, Op::IntSub, (0_u64, mask)).into());
            out.push((off, Op::IntLeft, (off, 1_u64)).into());
            out.push((off, Op::IntAdd, (off, w.shadow)).into());
            out.push((off, Op::IntAnd, (off, mask)).into());
            out.push((dst, Op::IntOr, (dst, off)).into());
        }
        let ids = u64::from(id) * 0x0001_0001_0001_0001;
        let mut left = 2 * size;
        let mut at = 0;
        while left > 0 {
            let chunk: u64 = if left >= 8 { 8 } else if left >= 4 { 4 } else { 2 };
            let value = match chunk {
                8 => Value::Const(ids, 8),
                4 => Value::Const(ids & 0xffff_ffff, 4),
                _ => Value::Const(ids & 0xffff, 2),
            };
            let ptr = if at == 0 {
                dst
            } else {
                let p = b.alloc_tmp(8);
                out.push((p, Op::IntAdd, (dst, at)).into());
                p
            };
            out.push((Op::Store(self.shadow), (Value::Var(ptr), value)).into());
            at += chunk;
            left -= chunk;
        }
    }

    /// At the head of block `k`: append `(k, last)` at the log's cursor and
    /// advance the cursor by `1 - visited[k]`, so only first entries stay.
    fn enter(&self, b: &mut pcode::Block, out: &mut Vec<pcode::Instruction>, k: u64) {
        let seen = b.alloc_tmp(1);
        let seen8 = b.alloc_tmp(8);
        let cursor = b.alloc_tmp(8);
        let slot = b.alloc_tmp(8);
        let at = b.alloc_tmp(8);
        out.push((seen, Op::Load(self.visited), k).into());
        out.push((cursor, Op::Load(self.entries), 0_u64).into());
        out.push((at, Op::IntLeft, (cursor, 3_u64)).into());
        out.push((at, Op::IntAdd, (at, 16_u64)).into());
        if self.edges {
            let last = b.alloc_tmp(4);
            out.push((last, Op::Load(self.entries), 8_u64).into());
            out.push((slot, Op::ZeroExtend, last).into());
            out.push((slot, Op::IntLeft, (slot, 32_u64)).into());
            out.push((slot, Op::IntOr, (slot, k)).into());
            out.push((Op::Store(self.entries), (8_u64, Value::Const(k, 4))).into());
        } else {
            out.push((slot, Op::Copy, (NO_PRED << 32) | k).into());
        }
        out.push((Op::Store(self.entries), (at, slot)).into());
        out.push((Op::Store(self.visited), (k, Value::Const(1, 1))).into());
        out.push((seen8, Op::ZeroExtend, seen).into());
        out.push((cursor, Op::IntAdd, (cursor, 1_u64)).into());
        out.push((cursor, Op::IntSub, (cursor, seen8)).into());
        out.push((Op::Store(self.entries), (0_u64, cursor)).into());
    }
}

impl CodeInjector for Unpack {
    fn inject(&mut self, _: &mut Cpu, group: &BlockGroup, code: &mut BlockTable) {
        let mut meaning = self.meaning.borrow_mut();
        let k = meaning.blocks.len() as u64;
        let entry = if k < MAX_BLOCKS {
            meaning.blocks.push(Block { addr: group.start, end: group.end });
            Some(k)
        } else {
            meaning.blocks_saturated = true;
            None
        };
        let mut pc = group.start;
        for idx in group.range() {
            let block = &mut code.blocks[idx];
            // Fresh temporaries must not collide with the lifter's.
            block.pcode.recompute_next_tmp();
            let insns = std::mem::take(&mut block.pcode.instructions);
            let mut out = Vec::with_capacity(insns.len() * 2);
            if idx == group.blocks.0
                && let Some(k) = entry
            {
                self.enter(&mut block.pcode, &mut out, k);
            }
            for insn in insns {
                match insn.op {
                    Op::InstructionMarker => pc = insn.inputs.first().as_u64(),
                    Op::Store(pcode::RAM_SPACE) => {
                        let [addr, value] = insn.inputs.get();
                        let size = value.size() as u64;
                        let next = meaning.sites.len() + 1;
                        let id = match meaning.site_of.get(&pc) {
                            Some(&id) => id,
                            None => {
                                meaning.sites_saturated |= next > u16::MAX as usize;
                                let id = next.min(u16::MAX as usize) as u16;
                                meaning.sites.push(Site { pc, size: size as usize });
                                meaning.site_of.insert(pc, id);
                                id
                            }
                        };
                        self.stamp(&mut block.pcode, &mut out, addr, size, id);
                    }
                    _ => {}
                }
                out.push(insn);
            }
            block.pcode.instructions = out;
            block.pcode.recompute_next_tmp();
            code.modified.insert(idx);
        }
    }
}
// HOOK-END

fn run(cli: &Cli, bytes: &[u8]) -> Result<Outcome, String> {
    let started = Instant::now();
    let mut vm: Vm = icicle_vm::build(&Config {
        triple: "x86_64-none".parse().unwrap(),
        enable_shadow_stack: false,
        enable_jit: !cli.interp,
        ..Config::default()
    })
    .map_err(|e| format!("icicle x86-64 (is GHIDRA_SRC set?): {e:?}"))?;
    let r = |name: &str| vm.cpu.arch.sleigh.get_varnode(name).unwrap();
    let regs = Regs {
        rax: r("RAX"),
        rdi: r("RDI"),
        rsi: r("RSI"),
        rdx: r("RDX"),
        r10: r("R10"),
        r8: r("R8"),
        r9: r("R9"),
        rsp: r("RSP"),
        fs: r("FS_OFFSET"),
        gs: r("GS_OFFSET"),
    };
    let argv: Vec<String> = std::iter::once(cli.elf.clone()).chain(cli.args.iter().cloned()).collect();
    let kernel = Kernel::boot(&mut CpuGuest(&mut vm.cpu, &regs), &cli.elf, bytes, &argv, cli.stdin.clone())?;
    let layout = Layout::new(kernel.image_lo, kernel.image_hi)?;
    let entry = kernel.entry;
    vm.set_env(Linux { kernel, regs, action: None, debug_info: DebugInfo::default() });
    vm.icount_limit = cli.budget.unwrap_or(5_000_000_000);

    // The three trace stores: host buffers the injected p-code indexes.
    let mut shadow = vec![0u16; if cli.hooks { SHADOW_LEN / 2 } else { 0 }];
    let mut entries = vec![0u8; if cli.hooks { ENTRIES_LEN } else { 0 }];
    let mut visited = vec![0u8; if cli.hooks { MAX_BLOCKS as usize } else { 0 }];
    let meaning = std::rc::Rc::new(std::cell::RefCell::new(Meaning::default()));
    if cli.hooks {
        entries[8..12].copy_from_slice(&(NO_PRED as u32).to_le_bytes());
        let trace = &mut vm.cpu.trace;
        let s = trace.register_store((shadow.as_mut_ptr() as *mut u8, shadow.len() * 2));
        let e = trace.register_store((entries.as_mut_ptr(), entries.len()));
        let v = trace.register_store((visited.as_mut_ptr(), visited.len()));
        vm.add_injector(Unpack {
            layout,
            edges: cli.edges,
            shadow: s.get_store_id(),
            entries: e.get_store_id(),
            visited: v.get_store_id(),
            meaning: meaning.clone(),
        });
    }
    let setup_ms = started.elapsed().as_secs_f64() * 1e3;

    let started = Instant::now();
    let exit = vm.run();
    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
    let pc = vm.cpu.read_pc();
    let steps = vm.cpu.icount();
    let limit = vm.icount_limit;
    let env = vm.env_mut::<Linux>().expect("our environment");
    let action = env.action.take();
    let stdout = std::mem::take(&mut env.kernel.stdout);
    let stderr = std::mem::take(&mut env.kernel.stderr);
    let mut warnings = Vec::new();
    if !env.kernel.enosys.is_empty() {
        warnings.push(format!("system calls answered ENOSYS: {:?}", env.kernel.enosys));
    }
    let syscalls = env.kernel.syscalls;
    let (exit_status, stop_reason, crashed, unsupported) = match (&action, &exit) {
        (Some(Action::Exit(c)), _) => (Some(*c), "exit".to_string(), false, None),
        (Some(Action::Unsupported(r)), _) => (None, format!("unsupported: {r}"), true, Some(r.clone())),
        (_, VmExit::InstructionLimit) => (None, format!("budget of {limit} instructions exhausted"), true, None),
        (_, other) => (None, format!("crashed: {other:?} at {pc:#x}"), true, None),
    };

    let record = cli.hooks.then(|| {
        let m = meaning.borrow();
        let cursor = u64::from_le_bytes(entries[..8].try_into().unwrap()) as usize;
        let log = (0..cursor)
            .map(|i| {
                let at = 16 + 8 * i;
                let slot = u64::from_le_bytes(entries[at..at + 8].try_into().unwrap());
                let pred = (slot >> 32) as u32;
                (slot as u32, (pred != NO_PRED as u32).then_some(pred))
            })
            .collect();
        Record {
            sites: m.sites.clone(),
            blocks: m.blocks.clone(),
            log,
            // The same heap buffer the trace store points at: moving the
            // Vec does not move it, and the VM is kept alive below.
            shadow: std::mem::take(&mut shadow),
            sites_saturated: m.sites_saturated,
            blocks_saturated: m.blocks_saturated,
        }
    });
    let extra = json!({"syscalls": syscalls, "blocks_lifted": meaning.borrow().blocks.len()});
    let run = Run {
        engine: "icicle".into(),
        strategy: if cli.interp { "interpreter" } else { "jit" }.into(),
        path: cli.elf.clone(),
        entry,
        exit: exit_status,
        stop_reason,
        crashed,
        stdout,
        stderr,
        steps: Some(steps),
        layout,
        hooks: cli.hooks,
        edges: cli.edges,
        record,
        warnings,
    };
    // The other two trace stores point into these buffers.
    let keep = (entries, visited);
    let read = Box::new(move |addr: u64, buf: &mut [u8]| {
        let _ = &keep;
        vm.cpu.mem.read_bytes_large(addr, buf, perm::NONE).is_ok()
    });
    Ok(Outcome { run, wall_ms, setup_ms, read, unsupported, extra })
}

fn main() {
    let lines = unpackbase::hook_lines(&[include_str!("main.rs")]);
    unpackbase::drive("icicle", lines, run);
}
