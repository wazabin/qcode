//! Name thunks after the function they forward to.
//!
//! A *thunk* is a function whose whole body is a single unconditional jump to
//! another function's entry — `jmp realfunc`, the shape a compiler emits for an
//! ICF-merged alias, a `/INCREMENTAL` jump stub, or a tail-call trampoline. Under
//! strict IR locality (context-split ruling 2) the lifter emits this inter-
//! procedural jump as a function-level [`TailCall`] at construction, so such a
//! function is one block ending in a `TailCall` carrying the callee's id.
//!
//! Left alone these carry a generated `fn_<addr>` name, which tells the reader
//! nothing. This pass renames them `thunk_<callee>` so the listing shows where
//! the jump actually goes. It only touches generated names — a function that
//! already has a real symbol (an export, a demangled C++ name) keeps it.
//!
//! Ordered after [`cpp_demangle`](super::cpp_demangle) so the callee name is
//! already demangled when it is borrowed into the thunk's name.

use std::borrow::Cow;

use qcode::value::{FunctionId, FunctionRef, QCodeView, insn::Mnemonic};

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
        f: &mut FunctionBody<'str>,
        m: ContextView<'_, 'str>,
        _next_minted: &mut u32,
    ) -> Result<Outcome<'str>, String> {
        let fid = f.id();
        // Read-only analysis of the body and the callee's *interface* (its
        // published name, read from the shared context), then buffer the
        // self-rename.
        let new_name: Option<String> = {
            let hr = m.body_view(f);
            let function = FunctionRef::new(hr, fid);

            // Only rename functions still carrying their generated `fn_<addr>`
            // name; a real symbol (export / demangled name) is authoritative.
            match function.address() {
                Some(addr) if function.name() == format!("fn_{addr:x}") => {
                    // A thunk is a lone block jumping to another function.
                    thunk_target(hr, fid).map(|callee_id| {
                        let callee = &hr.interface(callee_id).name;
                        format!("thunk_{callee}")
                    })
                }
                _ => None,
            }
        };

        match new_name {
            Some(name) => {
                // Returned; the driver uniquifies and applies it at the barrier.
                Ok(Outcome::renamed(Cow::Owned(name))
                    .preserving_global::<crate::CallGraphAnalysis>()
                    .preserving_global::<crate::AddressAnalysis>()
                    .preserving_local::<crate::AliasAnalysis>())
            }
            None => Ok(Outcome::unchanged()),
        }
    }
}

/// If `fun_id` is a single-block function ending in a `TailCall` to a *different*
/// function, return that callee. Otherwise `None`. (Strict IR locality: a tail
/// jump into another function is a function-level `TailCall`, never a foreign
/// `Branch`.)
fn thunk_target<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    fun_id: FunctionId,
) -> Option<FunctionId> {
    let function = FunctionRef::new(host, fun_id);

    let mut blocks = function.blocks();
    let block = blocks.next()?;
    if blocks.next().is_some() {
        return None;
    }

    let Mnemonic::TailCall(tc) = block.instructions().last()?.mnemonic() else {
        return None;
    };
    tc.target.real().filter(|&target| target != fun_id)
}

crate::register_function_pass!(NameThunks);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::run_function_pass;

    use qcode::context::Context;
    use qcode::value::{BasicBlock, FunctionBody};

    /// A bodyless callee `name` at 0x2000 to forward to; returns its function id.
    fn make_callee(ctx: &mut Context, name: &str) -> FunctionId {
        let callee = FunctionBody::make_at_addr(ctx, 0x2000, Some(name.to_owned().into())).id;
        let entry = BasicBlock::make(ctx, callee).with_address(0x2000).id;
        let zero = ctx.get_const(0, 8).id();
        (ctx).builder(entry).push_return(zero);
        FunctionBody::from_id_mut(ctx, callee)
            .set_root(entry)
            .unwrap();
        callee
    }

    /// A single-block function at 0x1000 (`name`) whose only instruction tail-calls
    /// `callee` (the strict-local thunk shape). Returns its id.
    fn make_thunk(ctx: &mut Context, name: &str, callee: FunctionId) -> FunctionId {
        let f = FunctionBody::make_at_addr(ctx, 0x1000, Some(name.to_owned().into())).id;
        let block = BasicBlock::make(ctx, f).with_address(0x1000).id;
        (ctx).builder(block).push_tail_call(callee);
        FunctionBody::from_id_mut(ctx, f).set_root(block).unwrap();
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
        assert_eq!(FunctionBody::from_id(&ctx, f).name(), "thunk_realfunc");
    }

    /// A function with a real symbol name keeps it.
    #[test]
    fn named_function_is_left_alone() {
        let mut ctx = Context::new();
        let callee = make_callee(&mut ctx, "realfunc");
        let f = make_thunk(&mut ctx, "helper", callee);

        let changed = run_function_pass::<NameThunks>(&mut ctx, f).unwrap();
        assert!(!changed);
        assert_eq!(FunctionBody::from_id(&ctx, f).name(), "helper");
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
