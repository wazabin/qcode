//! The inverse of [`brighten_stack`](crate::brighten_stack): lower the synthetic
//! `@stack_base` literals back onto the real stack pointer once they have served
//! their purpose.
//!
//! `brighten_stack` seeds the stack pointer with a `StackAddress`-typed
//! `@stack_base` literal so mem2reg/gvn can promote stack-relative loads/stores
//! into per-slot literals. After all the stack-aware analysis has run, those
//! `@stack_base ± N` literals would otherwise leak into the output (e.g.
//! `RSP = @stack_base+0x8`). This pass rewrites every surviving stack-address
//! literal into arithmetic on the function's *incoming* stack pointer — a
//! root-block parameter named after the stack-pointer register — so the same
//! write reads as `RSP = RSP + 0x8` and the entry-slot pointer `@stack_base+0x0`
//! reads as plain `RSP`.
//!
//! It must run last, after the summary pass has read each function's
//! `RSP = @stack_base + N` epilogue to recover its
//! [`stack_delta`](qcode::value::function::FunctionSignature::stack_delta): once
//! lowered, that signal is gone.

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};

use qcode::{
    builder::Builder,
    context::Context,
    types::stack_base,
    value::{
        BasicBlock, Function, FunctionId, ValueId, Varnode, VarnodeId, block::BlockId,
        insn::InstructionId,
    },
};

/// The root-block parameter representing the incoming stack pointer, named after
/// the stack-pointer register (e.g. `RSP`). Reuses an existing same-named param
/// if mem2reg already surfaced one; otherwise appends a fresh one. The name lets
/// the emulator seed it from the register file on entry (see
/// `seed_entry_params`), and `compute_input_regs` recover it as an input.
fn incoming_stack_pointer(ctx: &mut Context, root: BlockId, stack_ptr: VarnodeId) -> ValueId {
    let name = Varnode::from_id(ctx, stack_ptr).name().map(str::to_owned);

    if let Some(name) = &name {
        for param in BasicBlock::from_id(ctx, root).params() {
            if param.name() == Some(name.as_str()) {
                // `Value::id` of a block param is its `ValueId::BlockParam`.
                return param.id();
            }
        }
    }

    let size = Varnode::from_id(ctx, stack_ptr).size();
    let param_id = BasicBlock::from_id_mut(ctx, root).push_param(size).id;
    if let Some(name) = name {
        ctx.values.block_params[param_id].name = Some(Cow::Owned(name));
    }
    ValueId::BlockParam(param_id)
}

/// Rewrite every surviving `@stack_base ± N` literal in `function_id` into
/// `incoming_stack_pointer ± N`. A no-op for functions that never touched the
/// stack (no `StackAddress` literal was ever interned).
pub fn lower_stack(ctx: &mut Context, function_id: FunctionId, stack_ptr: VarnodeId) -> bool {
    let Some(sa_id) = ctx.types.stack_address_id() else {
        return false;
    };
    // The pointer width sizes both the offset-from-base recovery and the
    // materialized `incoming ± magnitude` arithmetic (4 for ESP, 8 for RSP).
    let ptr_width = ctx.types.size_of(sa_id);
    let base = stack_base(ptr_width);

    let insns: Vec<InstructionId> = Function::from_id(ctx, function_id)
        .iter()
        .flat_map(|b| b.instruction_ids().to_vec())
        .collect();

    // The actual stack-address literals this function references, mapped to their
    // signed offset from `STACK_BASE`. Key off the real `ValueId` rather than a
    // reconstructed one: brighten's `@stack_base` literal is *uninterned* (created
    // via `push_literal`), so equal-valued literals can have distinct ids.
    let mut lit_offsets: HashMap<ValueId, i64> = HashMap::new();
    for &iid in &insns {
        for arg in ctx.get_insn(iid).mnemonic().args() {
            if let ValueId::Literal(lid) = arg
                && ctx.values.literals[lid].type_id == sa_id
            {
                let off = ctx.values.literals[lid].value.wrapping_sub(base) as i64;
                lit_offsets.insert(arg, off);
            }
        }
    }
    if lit_offsets.is_empty() {
        return false;
    }

    let root = Function::from_id(ctx, function_id)
        .root()
        .expect("a function with stack activity has a root block")
        .id;
    let incoming = incoming_stack_pointer(ctx, root, stack_ptr);

    // One replacement per distinct offset: the incoming pointer itself for offset
    // 0, otherwise `incoming ± |offset|` materialised once at the root-block start
    // so it dominates every use.
    let offsets: BTreeSet<i64> = lit_offsets.values().copied().collect();
    let mut offset_repl: HashMap<i64, ValueId> = HashMap::new();
    {
        let mut builder = Builder::from_block(BasicBlock::from_id_mut(ctx, root));
        builder.set_insert_point_to_start();
        unsafe { builder.dont_finalize() };

        for &off in &offsets {
            let repl = if off == 0 {
                incoming
            } else {
                let magnitude = builder
                    .context_mut()
                    .get_const(off.unsigned_abs(), ptr_width)
                    .id();
                if off > 0 {
                    builder.push_add(incoming, magnitude).id()
                } else {
                    builder.push_sub(incoming, magnitude).id()
                }
            };
            offset_repl.insert(off, repl);
        }
    }

    // Rewrite operands per instruction (never globally: the `@stack_base`
    // literals are shared with functions not yet lowered).
    // `replace_instruction_mnemonic` keeps the use-def map consistent.
    for &iid in &insns {
        let mut mnemonic = ctx.get_insn(iid).mnemonic().clone();
        let args = mnemonic.args();
        let mut changed = false;
        for (&lit, &off) in &lit_offsets {
            if args.contains(&lit) {
                mnemonic.replace_value(lit, offset_repl[&off]);
                changed = true;
            }
        }
        if changed {
            ctx.replace_instruction_mnemonic(iid, mnemonic);
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{
        testing::TestContext,
        value::{Function, FunctionId, insn::Mnemonic},
    };

    /// A `StackAddress`-typed literal for `STACK_BASE + offset`.
    fn stack_addr_lit(ctx: &mut Context, offset: i64) -> ValueId {
        let sa = ctx.types.get_or_make_stack_address(8, None);
        ctx.get_typed_const(stack_base(8).wrapping_add(offset as u64), sa)
            .id()
    }

    /// Build a single-block function rooted at `0x1000`, populated by `f`.
    fn build_fn(
        tc: &mut TestContext,
        f: impl FnOnce(&mut qcode::builder::Builder<'static, '_>),
    ) -> FunctionId {
        let fun = Function::make(&mut tc.ctx, "f".into()).unwrap().id;
        let block = tc.ctx.get_or_make_block(0x1000);
        Function::from_id_mut(&mut tc.ctx, fun)
            .set_root(block)
            .unwrap();
        let mut b = qcode::builder::Builder::from_context(&mut tc.ctx, 0x1000);
        f(&mut b);
        unsafe { b.dont_finalize() };
        drop(b);
        fun
    }

    /// The incoming stack-pointer root param of `fun` (named after the SP
    /// varnode, `r3` in `TestContext`), if present.
    fn rsp_param(ctx: &Context, fun: FunctionId) -> Option<ValueId> {
        let root = Function::from_id(ctx, fun).root().unwrap().id;
        BasicBlock::from_id(ctx, root)
            .params()
            .find(|p| p.name() == Some("r3"))
            .map(|p| p.id())
    }

    /// True if any instruction in `fun` still references a `StackAddress` literal.
    fn has_stack_address(ctx: &Context, fun: FunctionId) -> bool {
        let sa = ctx.types.stack_address_id();
        Function::from_id(ctx, fun).iter().any(|b| {
            b.iter().any(|i| {
                i.mnemonic().args().iter().any(|a| {
                    matches!(a, ValueId::Literal(l) if Some(ctx.values.literals[*l].type_id) == sa)
                })
            })
        })
    }

    #[test]
    fn final_rsp_store_becomes_incoming_plus_delta() {
        let mut tc = TestContext::new();
        let (sp, reg) = (tc.r3, tc.reg_space); // r3 is named "r3", used as the SP
        let fun = build_fn(&mut tc, |b| {
            let v = stack_addr_lit(b.context_mut(), 8);
            b.push_store(v, ValueId::Varnode(sp), reg);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
        });

        lower_stack(&mut tc.ctx, fun, sp);

        assert!(
            !has_stack_address(&tc.ctx, fun),
            "no @stack_base may remain"
        );
        // The store source is now `incoming + 8`, an Int binop over the RSP param.
        let root = Function::from_id(&tc.ctx, fun).root().unwrap().id;
        let store = BasicBlock::from_id(&tc.ctx, root)
            .iter()
            .find_map(|i| match i.mnemonic() {
                Mnemonic::Store(s) if s.ptr == ValueId::Varnode(sp) => Some(s.src),
                _ => None,
            })
            .expect("RSP store present");
        assert!(
            matches!(store, ValueId::Instruction(_)),
            "store src should be the incoming+8 add, got {store:?}"
        );
    }

    #[test]
    fn entry_slot_load_becomes_incoming() {
        let mut tc = TestContext::new();
        let (sp, ram) = (tc.r3, tc.reg_space);
        let fun = build_fn(&mut tc, |b| {
            let slot = stack_addr_lit(b.context_mut(), 0);
            b.push_load::<false>(slot, 8, ram);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
        });

        lower_stack(&mut tc.ctx, fun, sp);

        let incoming = rsp_param(&tc.ctx, fun).expect("RSP param created");
        let root = Function::from_id(&tc.ctx, fun).root().unwrap().id;
        let load_ptr = BasicBlock::from_id(&tc.ctx, root)
            .iter()
            .find_map(|i| match i.mnemonic() {
                Mnemonic::Load(l) => Some(l.ptr),
                _ => None,
            })
            .expect("load present");
        assert_eq!(
            load_ptr, incoming,
            "the entry-slot (offset 0) pointer lowers to the incoming RSP directly"
        );
    }

    #[test]
    fn no_stack_address_is_noop() {
        let mut tc = TestContext::new();
        let (sp, reg) = (tc.r3, tc.reg_space);
        let fun = build_fn(&mut tc, |b| {
            let v = b.context_mut().get_const(7u64, 8).id();
            b.push_store(v, ValueId::Varnode(sp), reg);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
        });

        lower_stack(&mut tc.ctx, fun, sp);

        assert!(
            rsp_param(&tc.ctx, fun).is_none(),
            "no RSP param should be added when there is no stack activity"
        );
    }

    #[test]
    fn incoming_rsp_param_reused() {
        // Two distinct uses of the same offset share one incoming param, and two
        // sites at the same nonzero offset share a single add.
        let mut tc = TestContext::new();
        let (sp, reg) = (tc.r3, tc.reg_space);
        let fun = build_fn(&mut tc, |b| {
            let a = stack_addr_lit(b.context_mut(), 0);
            b.push_load::<false>(a, 8, reg);
            let c = stack_addr_lit(b.context_mut(), 8);
            b.push_store(c, ValueId::Varnode(sp), reg);
            let ret = b.context_mut().get_const(0u64, 8).id();
            b.push_return(ret);
        });

        lower_stack(&mut tc.ctx, fun, sp);

        assert!(rsp_param(&tc.ctx, fun).is_some());
        assert!(!has_stack_address(&tc.ctx, fun));
    }
}

// ----- pass ------------------------------------------------------------------

use crate::{FunctionPass, PipelineEnv};

#[derive(Default)]
pub struct LowerStack;

impl FunctionPass for LowerStack {
    const NAME: &'static str = "lower_stack";
    fn description(&self) -> &'static str {
        "Rewrite @stack_base literals back onto the real stack pointer"
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        env: &PipelineEnv,
    ) -> Result<bool, String> {
        Ok(lower_stack(ctx, fun_id, env.sp_varnode))
    }
}

crate::register_function_pass!(LowerStack);
