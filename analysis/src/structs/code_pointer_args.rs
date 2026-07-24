//! `propagate_code_pointer_args`: interprocedural code-pointer seeding
//! (prototype, piece 1).
//!
//! An indirect call through a root parameter — `call [@param_k]` — proves that
//! parameter is a **code pointer**: every caller passes a function address as
//! argument `k` (root params are in `param[i] ↔ Call.args[i]` lockstep with the
//! call site, see [`argpromote`](crate::calls::argpromote)). This pass walks that
//! backward interprocedural edge: for each callee param that is called through,
//! it resolves the corresponding argument at every direct caller; when the
//! argument is a constant address, it is a function entry, so we queue it as a
//! [`Discovery`] for the recursive lifter.
//!
//! This is the honest generalization of [`discover_libc_main`](crate::cfg::crt):
//! instead of one hardcoded callee (`__libc_start_main`) and one hardcoded
//! argument (`main`), any internal higher-order function seeds its callbacks.
//!
//! Prototype scope: the leaf resolver only accepts a **literal** argument (the
//! common shape after constprop). Copy/phi chains and function-pointer tables in
//! data are out of scope for this measurement.

use qcode::{
    discovery::{Discovery, FunctionDiscoveryReason},
    value::{FunctionBody, FunctionId, ValueId, insn::Mnemonic},
};
use rustc_hash::FxHashMap;

use crate::{Pass, PipelineEnv};

#[derive(Default)]
pub struct PropagateCodePointerArgs;

impl Pass for PropagateCodePointerArgs {
    const NAME: &'static str = "propagate_code_pointer_args";

    fn description(&self) -> &'static str {
        "Seed function discoveries from constant code-pointer arguments"
    }

    fn run(
        &self,
        cone: &mut crate::ConeMut,
        _env: &PipelineEnv,
    ) -> Result<crate::ModulePassOutcome, String> {
        let functions = cone.cone_functions();
        let ctx = cone.ctx();

        // Phase 1: for each internal function, the root-param indices that are
        // called through an indirect call (`call [@param_k]`).
        let mut code_ptr_params: FxHashMap<FunctionId, Vec<usize>> = FxHashMap::default();
        for &fid in &functions {
            let f = FunctionBody::from_id(ctx, fid);
            if f.is_external() {
                continue;
            }
            let Some(root) = f.root() else { continue };
            // Root params, in interface order.
            let params: Vec<ValueId> = root.params().map(|p| p.id()).collect();
            if params.is_empty() {
                continue;
            }
            let mut indices = Vec::new();
            for block in f.blocks() {
                for insn in block.iter() {
                    if let Mnemonic::CallInd(ci) = insn.mnemonic() {
                        let ptr = ci.ptr.qualify(fid);
                        if let Some(idx) = params.iter().position(|&p| p == ptr) {
                            if !indices.contains(&idx) {
                                indices.push(idx);
                            }
                        }
                    }
                }
            }
            if !indices.is_empty() {
                code_ptr_params.insert(fid, indices);
            }
        }

        if code_ptr_params.is_empty() {
            return Ok(crate::ModulePassOutcome::module_if(false));
        }

        // Phase 2: at every direct caller of such a callee, resolve the argument
        // feeding a code-pointer parameter to a constant address.
        // (addr, caller, source_addr)
        let mut seeds: Vec<(u64, FunctionId, u64)> = Vec::new();
        for &caller in &functions {
            let f = FunctionBody::from_id(ctx, caller);
            if f.is_external() {
                continue;
            }
            for block in f.blocks() {
                for insn in block.iter() {
                    let (target, args) = match insn.mnemonic() {
                        Mnemonic::Call(call) => (call.target.real(), &call.args),
                        Mnemonic::TailCall(call) => (call.target.real(), &call.args),
                        _ => continue,
                    };
                    let Some(target) = target else { continue };
                    let Some(indices) = code_ptr_params.get(&target) else {
                        continue;
                    };
                    let source = insn.address().unwrap_or(0);
                    for &idx in indices {
                        let Some(&arg) = args.get(idx) else { continue };
                        if let Some(addr) = literal_addr(ctx, arg.qualify(caller)) {
                            seeds.push((addr, caller, source));
                        }
                    }
                }
            }
        }

        // Phase 3: queue discoveries. `discover` / `add_synthetic_callee` are
        // cone-free shared writes; a non-executable address is rejected at lift.
        let mut changed = false;
        for (addr, caller, source) in seeds {
            changed |= cone.add_synthetic_callee(caller, addr);
            changed |= cone.discover(
                Discovery::function(addr)
                    .with_function_reason(FunctionDiscoveryReason::CodePointer)
                    .from_addr(source),
            );
        }

        Ok(crate::ModulePassOutcome::module_if(changed))
    }
}

/// The concrete address of a literal-valued argument, if any.
fn literal_addr(ctx: &qcode::context::Context<'_>, arg: ValueId) -> Option<u64> {
    match arg {
        ValueId::Literal(lid) => Some(ctx.shared.values.literals[lid].value),
        _ => None,
    }
}

crate::register_module_pass!(PropagateCodePointerArgs);
