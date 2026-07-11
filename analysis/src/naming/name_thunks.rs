//! Name thunks after the function they forward to.
//!
//! A *thunk* is a function whose whole body is a single unconditional jump to
//! another function's entry — `jmp realfunc`, the shape a compiler emits for an
//! ICF-merged alias, a `/INCREMENTAL` jump stub, or a tail-call trampoline.
//! Function-boundary splitting ([`crate::split_overlapping_functions`]) leaves
//! such a function as one block ending in a [`Branch`] to the callee's entry.
//!
//! Left alone these carry a generated `fn_<addr>` name, which tells the reader
//! nothing. This pass renames them `thunk_<callee>` so the listing shows where
//! the jump actually goes. It only touches generated names — a function that
//! already has a real symbol (an export, a demangled C++ name) keeps it.
//!
//! Ordered after [`cpp_demangle`](super::cpp_demangle) so the callee name is
//! already demangled when it is borrowed into the thunk's name.

use std::borrow::Cow;

use qcode::value::{
    BlockRef, FunctionId, FunctionRef,
    insn::{Branch, Mnemonic},
    util::base_ref::HostRef,
};

use crate::{ContextView, FunctionBody, FunctionPass, Outcome};

#[derive(Default)]
pub struct NameThunks;

impl FunctionPass for NameThunks {
    const NAME: &'static str = "name_thunks";

    fn description(&self) -> &'static str {
        "Name single-jump thunks after the function they forward to"
    }

    fn run<'str>(
        &self,
        f: &mut FunctionBody<'_, 'str>,
        m: ContextView<'_, 'str>,
    ) -> Result<Outcome<'str>, String> {
        let fid = f.id();
        // Read-only analysis of the body and the callee's *interface* (its
        // published name, read from the shared context), then buffer the
        // self-rename.
        let new_name: Option<String> = {
            let hr = f.read_host(m);
            let function = FunctionRef::new(hr, fid);

            // Only rename functions still carrying their generated `fn_<addr>`
            // name; a real symbol (export / demangled name) is authoritative.
            match function.address() {
                Some(addr) if function.name() == format!("fn_{addr:x}") => {
                    // A thunk is a lone block jumping to another function.
                    thunk_target(hr, fid).map(|callee_id| {
                        let callee = FunctionRef::new(hr, callee_id).name();
                        format!("thunk_{callee}")
                    })
                }
                _ => None,
            }
        };

        match new_name {
            Some(name) => {
                // Buffered; the driver uniquifies and applies it at the barrier.
                f.effects_mut().rename_self(Cow::Owned(name));
                Ok(Outcome::changed(true))
            }
            None => Ok(Outcome::unchanged()),
        }
    }
}

/// If `fun_id` is a single-block function ending in an unconditional `Branch` to
/// a *different* function's entry, return that callee. Otherwise `None`.
fn thunk_target(host: HostRef, fun_id: FunctionId) -> Option<FunctionId> {
    let function = FunctionRef::new(host, fun_id);

    let mut blocks = function.blocks();
    let block = blocks.next()?;
    if blocks.next().is_some() {
        return None;
    }

    let callee = match block.instructions().last()?.mnemonic() {
        // Post-split, a tail jump into another function's entry is a function-level
        // `TailCall` carrying the callee's id directly.
        Mnemonic::TailCall(tc) => tc.target,
        // A raw `jmp realfunc` not yet rewritten by `split_overlapping_functions`.
        Mnemonic::Branch(Branch { target, .. }) => match host {
            // Module scope sees every arena: read the target block's parent.
            HostRef::Module(_) => BlockRef::new(host, *target).function()?.id,
            // A checked-out host cannot read a foreign block's arena (context-split
            // Pin B). The driver normalizes cross-function references away before
            // every function stage (`split_overlapping_functions`), and strict
            // locality parents a block to its storing function — so the id's
            // owning-function qualifier *is* the callee when this arm is reached
            // (unit-test adapter path).
            HostRef::Checked { .. } if target.func != fun_id => target.func,
            HostRef::Checked { .. } => BlockRef::new(host, *target).function()?.id,
        },
        _ => return None,
    };
    (callee != fun_id).then_some(callee)
}

crate::register_function_pass!(NameThunks);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::run_function_pass;
    use qcode::builder::Builder;
    use qcode::context::Context;
    use qcode::value::{BasicBlock, BlockId, Function};

    /// A bodyless callee `name` at 0x2000 to forward to; returns its entry block.
    fn make_callee(ctx: &mut Context, name: &str) -> BlockId {
        let callee = Function::make_at_addr(ctx, 0x2000, Some(name.to_owned().into())).id;
        let entry = BasicBlock::make(ctx, callee).with_address(0x2000).id;
        let zero = ctx.get_const(0, 8).id();
        Builder::from_block(BasicBlock::from_id_mut(ctx, entry)).push_return(zero);
        Function::from_id_mut(ctx, callee).set_root(entry).unwrap();
        entry
    }

    /// A single-block function at 0x1000 (`name`) whose only instruction jumps to
    /// `target`, wired into the CFG. Returns its id.
    fn make_thunk(ctx: &mut Context, name: &str, target: BlockId) -> FunctionId {
        let f = Function::make_at_addr(ctx, 0x1000, Some(name.to_owned().into())).id;
        let block = BasicBlock::make(ctx, f).with_address(0x1000).id;
        Builder::from_block(BasicBlock::from_id_mut(ctx, block)).push_branch(target);
        ctx.add_cfg_edge(block, target);
        Function::from_id_mut(ctx, f).set_root(block).unwrap();
        f
    }

    /// A lone `jmp other` under a generated name is renamed `thunk_<callee>`.
    #[test]
    fn single_jump_is_named_after_callee() {
        let mut ctx = Context::new();
        let callee = make_callee(&mut ctx, "realfunc");
        let f = make_thunk(&mut ctx, "fn_1000", callee);

        let changed = run_function_pass::<NameThunks>(&mut ctx, f).unwrap();
        assert!(changed);
        assert_eq!(Function::from_id(&ctx, f).name(), "thunk_realfunc");
    }

    /// A function with a real symbol name keeps it.
    #[test]
    fn named_function_is_left_alone() {
        let mut ctx = Context::new();
        let callee = make_callee(&mut ctx, "realfunc");
        let f = make_thunk(&mut ctx, "helper", callee);

        let changed = run_function_pass::<NameThunks>(&mut ctx, f).unwrap();
        assert!(!changed);
        assert_eq!(Function::from_id(&ctx, f).name(), "helper");
    }

    /// A multi-block function is not a thunk.
    #[test]
    fn multi_block_function_is_not_a_thunk() {
        let mut ctx = Context::new();
        let callee = make_callee(&mut ctx, "realfunc");
        let f = make_thunk(&mut ctx, "fn_1000", callee);
        // A second block (owned by `f`) means it is no longer a lone-jump thunk.
        let _extra = BasicBlock::make(&mut ctx, f).with_address(0x1008).id;

        let changed = run_function_pass::<NameThunks>(&mut ctx, f).unwrap();
        assert!(!changed);
    }
}
