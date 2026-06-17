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

use qcode::{
    context::Context,
    value::{
        BasicBlock, Function, FunctionId, Renameable,
        insn::{Branch, Mnemonic},
    },
};

use crate::{FunctionPass, PipelineEnv};

#[derive(Default)]
pub struct NameThunks;

impl FunctionPass for NameThunks {
    const NAME: &'static str = "name_thunks";

    fn description(&self) -> &'static str {
        "Name single-jump thunks after the function they forward to"
    }

    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        let function = Function::from_id(ctx, fun_id);

        // Only rename functions still carrying their generated `fn_<addr>` name;
        // a real symbol (export / demangled name) is authoritative and kept.
        let Some(addr) = function.address() else {
            return Ok(false);
        };
        if function.name() != format!("fn_{addr:x}") {
            return Ok(false);
        }

        // A thunk is a lone block whose terminator jumps to another function.
        let Some(callee_id) = thunk_target(ctx, fun_id) else {
            return Ok(false);
        };

        let callee = Function::from_id(ctx, callee_id).name().to_string();
        let name = ctx.get_unique_name(Cow::Owned(format!("thunk_{callee}")));
        Function::from_id_mut(ctx, fun_id)
            .rename(name)
            .map_err(|e| e.to_string())?;
        Ok(true)
    }
}

/// If `fun_id` is a single-block function ending in an unconditional `Branch` to
/// a *different* function's entry, return that callee. Otherwise `None`.
fn thunk_target(ctx: &Context, fun_id: FunctionId) -> Option<FunctionId> {
    let function = Function::from_id(ctx, fun_id);

    let mut blocks = function.blocks();
    let block = blocks.next()?;
    if blocks.next().is_some() {
        return None;
    }

    let Mnemonic::Branch(Branch { target, .. }) = block.instructions().last()?.mnemonic() else {
        return None;
    };

    let callee = BasicBlock::from_id(ctx, *target).function()?;
    (callee.id != fun_id).then_some(callee.id)
}

crate::register_function_pass!(NameThunks);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::run_function_pass;
    use qcode::builder::Builder;
    use qcode::value::BlockId;

    /// A bodyless callee `name` at 0x2000 to forward to; returns its entry block.
    fn make_callee(ctx: &mut Context, name: &str) -> BlockId {
        let callee = Function::make_at_addr(ctx, 0x2000, Some(name.to_owned().into())).id;
        let entry = BasicBlock::make(ctx).with_address(0x2000).id;
        let zero = ctx.get_const(0, 8).id();
        Builder::from_block(BasicBlock::from_id_mut(ctx, entry)).push_return(zero);
        Function::from_id_mut(ctx, callee).set_root(entry).unwrap();
        entry
    }

    /// A single-block function at 0x1000 (`name`) whose only instruction jumps to
    /// `target`, wired into the CFG. Returns its id.
    fn make_thunk(ctx: &mut Context, name: &str, target: BlockId) -> FunctionId {
        let block = BasicBlock::make(ctx).with_address(0x1000).id;
        Builder::from_block(BasicBlock::from_id_mut(ctx, block)).push_branch(target);
        ctx.add_cfg_edge(block, target);
        let f = Function::make_at_addr(ctx, 0x1000, Some(name.to_owned().into())).id;
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
        // A second block means it is no longer a lone-jump thunk.
        let extra = BasicBlock::make(&mut ctx).with_address(0x1008).id;
        Function::from_id_mut(&mut ctx, f).add_block(extra);

        let changed = run_function_pass::<NameThunks>(&mut ctx, f).unwrap();
        assert!(!changed);
    }
}
