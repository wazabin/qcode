//! Register access that does not go through the emulator's name lookup.
//!
//! The obvious way to read `RAX` is `emulator().read_varnode_by_name(&ctx,
//! "RAX")`, but that borrows the machine's `Context` — which the VM owns and
//! grows as it lifts code — at the same time as its emulator. Cloning the
//! context per system call is what that costs. Instead, every register the
//! environment touches is resolved to its `(space, offset, size)` once, and
//! read or written directly in the register space's flat storage.

use qcode::{context::Context, space::MemorySpaceId, value::ValueId, value::Varnode};
use qcode_vm::VmMemory;

/// One register's storage.
#[derive(Debug, Clone, Copy)]
pub struct Reg {
    pub space: MemorySpaceId,
    pub addr: u64,
    pub size: usize,
}

impl Reg {
    pub fn resolve(ctx: &Context<'_>, name: &str) -> Option<Self> {
        let Some(ValueId::Varnode(id)) = ctx.get_named(name) else {
            return None;
        };
        let varnode = Varnode::from_id(ctx, id);
        Some(Self {
            space: varnode.space().id.into(),
            addr: varnode.address() as u64,
            size: varnode.size(),
        })
    }

    pub fn read(&self, memory: &mut VmMemory) -> u64 {
        memory
            .flat_mut()
            .read_u128(self.space, self.addr, self.size)
            .unwrap_or(0) as u64
    }

    pub fn write(&self, memory: &mut VmMemory, value: u64) {
        memory
            .flat_mut()
            .entry(self.space)
            .write_u128(self.addr, self.size, u128::from(value))
            .expect("register storage is always addressable");
    }
}

/// The x86-64 registers the environment reads and writes.
#[derive(Debug, Clone, Copy)]
pub struct Regs {
    pub rax: Reg,
    pub rbx: Reg,
    pub rcx: Reg,
    pub rdx: Reg,
    pub rsi: Reg,
    pub rdi: Reg,
    pub rbp: Reg,
    pub rsp: Reg,
    pub r8: Reg,
    pub r9: Reg,
    pub r10: Reg,
    pub r11: Reg,
    pub r12: Reg,
    pub r13: Reg,
    pub r14: Reg,
    pub r15: Reg,
    /// The SLEIGH x86 specification models the `FS` segment base as the
    /// `FS_OFFSET` register, which every `%fs:`-relative access adds in.
    pub fs_base: Reg,
    pub gs_base: Reg,
    pub fpu_control: Reg,
    pub fpu_status: Reg,
    pub fpu_tag: Reg,
    pub mxcsr: Reg,
}

impl Regs {
    pub fn resolve(ctx: &Context<'_>) -> Result<Self, String> {
        let get = |name: &str| {
            Reg::resolve(ctx, name).ok_or_else(|| format!("register `{name}` is not in the spec"))
        };
        Ok(Self {
            rax: get("RAX")?,
            rbx: get("RBX")?,
            rcx: get("RCX")?,
            rdx: get("RDX")?,
            rsi: get("RSI")?,
            rdi: get("RDI")?,
            rbp: get("RBP")?,
            rsp: get("RSP")?,
            r8: get("R8")?,
            r9: get("R9")?,
            r10: get("R10")?,
            r11: get("R11")?,
            r12: get("R12")?,
            r13: get("R13")?,
            r14: get("R14")?,
            r15: get("R15")?,
            fs_base: get("FS_OFFSET")?,
            gs_base: get("GS_OFFSET")?,
            fpu_control: get("FPUControlWord")?,
            fpu_status: get("FPUStatusWord")?,
            fpu_tag: get("FPUTagWord")?,
            mxcsr: get("MXCSR")?,
        })
    }

    /// Puts the machine in the state a process starts in: every
    /// general-purpose register zero and the floating-point units as
    /// `finit` and a reset leave them.
    ///
    /// The floating-point state matters more than it looks. The SLEIGH
    /// semantics of every x87 instruction begin by checking for an unmasked
    /// exception left pending, and do nothing while there is one. With the
    /// control word zero every exception is unmasked, and with the tag word
    /// zero every stack slot is occupied, so the first `fld` is a stack
    /// overflow that stays pending and every x87 instruction after it is a
    /// no-op. musl's printf formats floats through long doubles, which is
    /// how busybox printed `-nan` for zero and `seq` printed nothing.
    pub fn reset(&self, memory: &mut VmMemory) {
        for (_, reg) in self.named() {
            reg.write(memory, 0);
        }
        // Precision control 64 bits, round to nearest, all exceptions masked.
        self.fpu_control.write(memory, 0x037f);
        self.fpu_status.write(memory, 0);
        // Two bits per slot, `11` meaning empty.
        self.fpu_tag.write(memory, 0xffff);
        // Round to nearest, all exceptions masked.
        self.mxcsr.write(memory, 0x1f80);
    }

    /// All general-purpose registers, in a stable order for dumps.
    pub fn named(&self) -> [(&'static str, Reg); 18] {
        [
            ("rax", self.rax),
            ("rbx", self.rbx),
            ("rcx", self.rcx),
            ("rdx", self.rdx),
            ("rsi", self.rsi),
            ("rdi", self.rdi),
            ("rbp", self.rbp),
            ("rsp", self.rsp),
            ("r8", self.r8),
            ("r9", self.r9),
            ("r10", self.r10),
            ("r11", self.r11),
            ("r12", self.r12),
            ("r13", self.r13),
            ("r14", self.r14),
            ("r15", self.r15),
            ("fs_base", self.fs_base),
            ("gs_base", self.gs_base),
        ]
    }
}
