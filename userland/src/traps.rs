//! Hooks for the processor exceptions the lifted code does not raise itself.
//!
//! The p-code of an integer division divides by zero quietly, yielding
//! zero; [`DivideHook`] checks the divisor first and stops with
//! [`DIVIDE_ERROR`], which the task turns into `SIGFPE`. [`StepHook`] stops
//! before every guest instruction while the machine's step flag is set,
//! which is how a tracee is single-stepped; it is installed the first time
//! a tracer asks, so a process that never steps never pays for it.

use qcode::value::BasicBlock;
use qcode::value::insn::{Binop, IntBinop, Mnemonic};
use qcode_vm::hook::{BlockView, Emitter, Hook, Site};

/// The interrupt code of a division by zero. Below the hook table's codes.
pub(crate) const DIVIDE_ERROR: u64 = 0x7573_0001;
/// The interrupt code of a single-step check.
pub(crate) const SINGLE_STEP: u64 = 0x7573_0002;
/// The state space holding the step flag, one word: non-zero while the task
/// on the machine is being stepped.
pub(crate) const STEP_SPACE: &str = "userland.step";

/// Stops before an integer division whose divisor is zero, positioned
/// before the guest instruction has written anything: the x86 division
/// semantics compute the quotient before they store a register.
pub(crate) struct DivideHook;

impl Hook for DivideHook {
    fn sites(&mut self, view: &BlockView<'_>) -> Vec<Site> {
        let ctx = view.ctx;
        let first = view
            .since
            .or_else(|| BasicBlock::from_id(ctx, view.block).first_instruction());
        let mut sites = Vec::new();
        // One check per guest instruction: its remainder divides by the
        // same value as its quotient, which is checked first.
        let mut last = None;
        for insn in std::iter::successors(first.map(|id| ctx.get_insn(id)), |insn| insn.next()) {
            let Mnemonic::Binop(binary) = insn.mnemonic() else {
                continue;
            };
            if !matches!(
                binary.op,
                Binop::Int(IntBinop::Div | IntBinop::Sdiv | IntBinop::Rem | IntBinop::Srem)
            ) {
                continue;
            }
            let address = insn.address();
            if address.is_some() && address == last {
                continue;
            }
            last = address;
            sites.push(Site::Compare { insn: insn.id });
        }
        sites
    }

    fn instrument(&mut self, _site: &Site, emit: &mut Emitter<'_>) {
        let Some((_, _, divisor)) = emit.binop_operands() else {
            return;
        };
        let zero = emit.constant(0, emit.size_of(divisor));
        let cond = emit.binop(IntBinop::Equal, divisor, zero);
        emit.interrupt_if(cond, DIVIDE_ERROR, &[]);
    }
}

/// Stops before each guest instruction while the step flag is set.
pub(crate) struct StepHook;

impl Hook for StepHook {
    fn sites(&mut self, view: &BlockView<'_>) -> Vec<Site> {
        view.addresses()
    }

    fn instrument(&mut self, _site: &Site, emit: &mut Emitter<'_>) {
        let space = emit.state_space(STEP_SPACE);
        let zero = emit.constant(0, 8);
        let flag = emit.load_from(space, zero, 8);
        let cond = emit.binop(IntBinop::NotEqual, flag, zero);
        emit.interrupt_if(cond, SINGLE_STEP, &[]);
    }
}
