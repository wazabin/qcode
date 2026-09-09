//! Unicorn-shaped hooks: callbacks the machine calls from inside [`Vm::run`].
//!
//! Each registration installs a [`Hook`](crate::hook::Hook) whose interrupt
//! carries a code from a reserved range, and files the callback under it.
//! When a run reaches such an interrupt the callback runs with the machine
//! stopped at the site, the machine resumes, and the run goes on — unless the
//! callback asks to [`stop`](HookAction::Stop), in which case the run returns
//! [`VmExit::HookStop`]. Interrupts with any other code, and every other
//! exit, come back to the caller as before.
//!
//! Several hooks may watch the same site. They fire in registration order,
//! each from its own interrupt, and a stop ends the run before the later ones
//! run: Unicorn's chain with Qiling's veto.
//!
//! Instruction hooks are different in kind: they answer an architecture user
//! op the interpreter has no semantics for — `syscall`, `rdtsc`, `cpuid` —
//! and so must supply its result. [`InsnAction::Handled`] carries that.

use qcode::value::ValueId;

use crate::{
    hook::{BlockView, Emitter, Hook, Site},
    vm::{Interrupt, InterruptKind},
};

/// A registered hook, for [`hook_del`](crate::Vm::hook_del).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HookId(pub u64);

/// What a code or memory hook wants the run to do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookAction {
    /// Resume the guest and carry on.
    Continue,
    /// Leave the machine stopped at the site and return from the run.
    Stop,
}

/// What an instruction hook did about the user op it was called for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsnAction {
    /// The op's effect has been applied; this is its result, if it declares
    /// one. The machine resumes past it.
    Handled(Option<u128>),
    /// Not this hook's op: the next hook is asked, and if none handles it the
    /// interrupt is returned to the caller.
    Unhandled,
    /// Leave the machine stopped at the op and return from the run.
    Stop,
}

/// A guest memory access a hook observes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemAccess {
    /// The guest instruction making the access.
    pub pc: Option<u64>,
    pub addr: u64,
    pub size: u64,
    /// The value being written; `None` for a read, or a value wider than
    /// 64 bits.
    pub value: Option<u64>,
}

/// Interrupt codes the table owns. Anything at or above this is a table
/// hook's; user injectors keep their codes below it.
pub const TABLE_CODES: u64 = 1 << 62;

pub(crate) type CodeCallback<S> = Box<dyn FnMut(&mut crate::Vm<S>, u64) -> HookAction>;
pub(crate) type MemCallback<S> = Box<dyn FnMut(&mut crate::Vm<S>, &MemAccess) -> HookAction>;
pub(crate) type InsnCallback<S> = Box<dyn FnMut(&mut crate::Vm<S>, &Interrupt) -> InsnAction>;

pub(crate) enum Callback<S> {
    Code(CodeCallback<S>),
    Mem(MemCallback<S>),
    /// For the user op called `name`, or any user op when `None`.
    Insn {
        name: Option<Box<str>>,
        callback: InsnCallback<S>,
    },
}

/// The registered callbacks, by hook.
pub(crate) struct HookTable<S> {
    next: u64,
    pub(crate) callbacks: Vec<(HookId, Callback<S>)>,
}

impl<S> Default for HookTable<S> {
    fn default() -> Self {
        Self {
            next: 0,
            callbacks: Vec::new(),
        }
    }
}

impl<S> HookTable<S> {
    pub(crate) fn register(&mut self, callback: Callback<S>) -> HookId {
        let id = HookId(self.next);
        self.next += 1;
        self.callbacks.push((id, callback));
        id
    }

    pub(crate) fn remove(&mut self, id: HookId) -> bool {
        let before = self.callbacks.len();
        self.callbacks.retain(|(have, _)| *have != id);
        self.callbacks.len() != before
    }

    /// The interrupt code a hook's injected interrupts carry.
    pub(crate) fn code(id: HookId) -> u64 {
        TABLE_CODES + id.0
    }

    /// The hook an explicit interrupt belongs to, if it is the table's.
    pub(crate) fn owner(interrupt: &Interrupt) -> Option<HookId> {
        match interrupt.kind {
            InterruptKind::Explicit { code } if code >= TABLE_CODES => {
                Some(HookId(code - TABLE_CODES))
            }
            _ => None,
        }
    }
}

/// Whether `addr` lies in `begin..=end`, or anywhere when `begin > end` —
/// Unicorn's convention for "no range".
pub(crate) fn in_range(begin: u64, end: u64, addr: u64) -> bool {
    begin > end || (begin..=end).contains(&addr)
}

/// Stops before every guest instruction in a range: `hook_code`.
pub(crate) struct CodeRangeHook {
    pub begin: u64,
    pub end: u64,
    pub code: u64,
}

impl Hook for CodeRangeHook {
    fn sites(&mut self, block: &BlockView<'_>) -> Vec<Site> {
        block
            .addresses()
            .into_iter()
            .filter(|site| {
                matches!(site, Site::Address { address, .. } if in_range(self.begin, self.end, *address))
            })
            .collect()
    }

    fn instrument(&mut self, _site: &Site, emit: &mut Emitter<'_>) {
        let address = emit.constant(emit.address().unwrap_or_default(), 8);
        emit.interrupt(self.code, &[address]);
    }
}

/// Stops before every load from guest memory in a range, with the address
/// and width: `hook_mem_read`. The range check is IR, as in
/// [`WriteWatch`](crate::hook::WriteWatch).
pub(crate) struct ReadWatch {
    pub begin: u64,
    pub end: u64,
    pub code: u64,
}

impl Hook for ReadWatch {
    fn sites(&mut self, block: &BlockView<'_>) -> Vec<Site> {
        block.loads()
    }

    fn instrument(&mut self, _site: &Site, emit: &mut Emitter<'_>) {
        let Some((ptr, size)) = emit.load_operands() else {
            return;
        };
        let cond = emit.in_range(ptr, self.begin, self.end);
        let args: Vec<ValueId> = vec![emit.zext(ptr, 8), emit.constant(size as u64, 8)];
        emit.interrupt_if(cond, self.code, &args);
    }
}
