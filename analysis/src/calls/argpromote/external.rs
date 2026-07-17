//! External-call argument channel: resolve the arguments at call sites to
//! bodyless **external** (imported) functions from their materialized C
//! interface.
//!
//! [`argpromote_registers`](super::argpromote_registers) and the RAM channel
//! functionalize *bodied* callees and, in doing so, thread each caller's
//! `Call.args`. An external stub has no body, so neither runs and its calls
//! render argument-less (`call fn SHGetSpecialFolderPathW()`). This channel
//! fills that gap for any external callee whose
//! [`ExternInterface`](qcode::value::ExternInterface) was materialized earlier by
//! [`external_sigs`](crate::assumptions::external_sig): for each one it appends
//! the positional `Call.args` at every direct caller — a register reload (System
//! V integer/SSE args) or an SP-relative stack load (32-bit `stdcall`/`cdecl`,
//! and System V register overflow) — and leaves the following gvn round to
//! forward each load to the value the caller set up (a register write, or the
//! `push` that stored the stack slot).
//!
//! This pass no longer consults `cabi` or the binary: the whole call interface
//! (argument slots, names, and the return register) was planned once by
//! `external_sigs` and lives on the callee's `FunctionSignature`.
//!
//! It only ever *adds* arguments to a call; the bodyless callee is never
//! touched. The `args.len() == idx` guard keeps it idempotent across pipeline
//! rounds and aligned `arg[i] ↔ param i`.

use qcode::{
    context::Context,
    space::SpaceId,
    value::{
        BasicBlock, ExternArg, ExternSlot, FunctionBody, FunctionId, Instruction, Value, ValueId,
        Varnode, VarnodeId, insn::Mnemonic,
    },
};

use super::super::append_caller_arg;
use crate::{Pass, PipelineEnv};

/// Append the resolved `Call.args` at every direct caller of each external
/// function with a materialized interface. Returns the set of changed callers.
fn argpromote_external_changed_functions(
    ctx: &mut Context,
    env: &PipelineEnv,
    targets: &[FunctionId],
) -> rustc_hash::FxHashSet<FunctionId> {
    let target_set: rustc_hash::FxHashSet<_> = targets.iter().copied().collect();
    let ptr_width = (env.cfg.bitness / 8).max(1) as usize;
    let Some(sp) = env.sp_varnode else {
        return rustc_hash::FxHashSet::default();
    };
    let sp_space = Varnode::from_id(&*ctx, sp).space().id;
    let default_space = ctx.shared.default_space;

    // Only externals that are actually *called* can gain arguments: both
    // `bind_external_args` and `bind_external_return` key on a direct `Call` whose
    // target is the external. Filtering on the direct call-site index (an O(1) map
    // lookup) skips the whole-program instruction scan for every
    // imported-but-unreferenced symbol — the bulk of an import table.
    let graph = crate::CallGraph::analyze(ctx);
    let externals: Vec<FunctionId> = ctx
        .functions()
        .filter(|f| f.is_external())
        .map(|f| f.id)
        .filter(|&id| !crate::calls::direct_call_sites(ctx, &graph, id).is_empty())
        .collect();
    let callers: rustc_hash::FxHashMap<FunctionId, Vec<FunctionId>> = externals
        .iter()
        .map(|&fid| (fid, graph.callers(fid)))
        .collect();
    drop(graph);

    if externals.is_empty() {
        return rustc_hash::FxHashSet::default();
    }

    let mut changed = rustc_hash::FxHashSet::default();
    for fid in externals {
        if callers[&fid]
            .iter()
            .any(|caller| !target_set.contains(caller))
        {
            continue;
        }
        // The call interface materialized by `external_sigs`. An external with no
        // interface (unknown prototype / not selected / unplaceable) is skipped.
        let func = FunctionBody::from_id(ctx, fid);
        let Some(iface) = func.extern_interface() else {
            continue;
        };
        let plan = iface.args.clone();
        // The single ABI return register, recorded on the signature by
        // `external_sigs` (`None` for a void/aggregate return).
        let ret = func
            .signature()
            .and_then(|s| s.outputs.as_ref())
            .and_then(|o| o.first().copied());

        if bind_external_args(ctx, fid, &plan, sp, sp_space, default_space, ptr_width) {
            changed.extend(callers[&fid].iter().copied());
        }
        if let Some(ret) = ret
            && bind_external_return(ctx, fid, ret)
        {
            changed.extend(callers[&fid].iter().copied());
        }
    }
    changed
}

/// Give every direct caller of resolved external `fid` a return value: type the
/// `call` result as the return register's scalar and store it back into that
/// register in the call's continuation block. A later gvn round forwards a
/// post-call read of the register to the call result — the scalar analogue of
/// argpromote's returned write-set (`res = foo(); store(reg <- res)`), for a
/// single ABI return register. Idempotent: a call whose continuation already
/// stores its result to `ret` is left untouched.
fn bind_external_return(ctx: &mut Context, fid: FunctionId, ret: VarnodeId) -> bool {
    let ret_space = Varnode::from_id(&*ctx, ret).space().id;
    let size = Varnode::from_id(&*ctx, ret).size();
    let int_ty = ctx.shared.types.get_or_make_int(size);

    let call_sites = crate::calls::fresh_direct_call_sites(ctx, fid);

    let mut changed = false;
    for call_id in call_sites {
        // The call's fall-through continuation, where the return register becomes
        // live. A call with no successor (e.g. a noreturn tail) is skipped.
        let Some(cont) = ctx
            .get_insn(call_id)
            .parent()
            .and_then(|b| b.successors().next().map(|(_, s)| s))
        else {
            continue;
        };
        let result = ValueId::Instruction(call_id);

        // Idempotency: skip a continuation that already stores this call's result
        // into the return register.
        let already = BasicBlock::from_id(ctx, cont).iter().any(|insn| {
            matches!(
                insn.mnemonic(),
                Mnemonic::Store(s)
                    if s.src.qualify(insn.id.func) == result
                        && s.ptr.qualify(insn.id.func) == ValueId::Varnode(ret)
            )
        });
        if already {
            continue;
        }

        Instruction::from_id_mut(ctx, call_id).set_type(int_ty);
        let mut b = (ctx).builder(cont);
        b.set_insert_point_to_start();
        b.push_store(result, ValueId::Varnode(ret), ret_space);
        changed = true;
    }
    changed
}

/// Thread `plan` through every direct caller of `fid`, one positional argument at
/// a time and in order, so the `args.len() == idx` guard keeps each call's
/// arguments aligned and the pass idempotent. Argument display names were
/// recorded on `fid` by `external_sigs`, so this only rewrites call sites.
fn bind_external_args(
    ctx: &mut Context,
    fid: FunctionId,
    plan: &[ExternArg],
    sp: VarnodeId,
    sp_space: SpaceId,
    default_space: SpaceId,
    ptr_width: usize,
) -> bool {
    let mut changed = false;
    for (idx, arg) in plan.iter().enumerate() {
        let slot = arg.slot;
        changed |= append_caller_arg(ctx, fid, |ctx, call_id, block| {
            let Mnemonic::Call(c) = ctx.get_insn(call_id).mnemonic() else {
                return None;
            };
            // Only the call whose next positional slot is exactly this one, so
            // arguments land in order and an already-bound call is skipped.
            if c.args.len() != idx {
                return None;
            }
            let mut b = (ctx).builder(block);
            b.set_insert_point_before(call_id);
            let value = match slot {
                ExternSlot::Reg(vn, size) => {
                    let space = Varnode::from_id(b.shr(), vn).space().id;
                    b.push_load::<false>(ValueId::Varnode(vn), size, space).id()
                }
                ExternSlot::Stack { offset, size } => {
                    let sp_val = b
                        .push_load::<false>(ValueId::Varnode(sp), ptr_width, sp_space)
                        .id();
                    let addr = if offset == 0 {
                        sp_val
                    } else {
                        let off = b.shr().get_const(offset as u64, ptr_width);
                        b.push_add(sp_val, off).id()
                    };
                    b.push_load::<false>(addr, size, default_space).id()
                }
            };
            Some(value)
        });
    }
    changed
}

#[derive(Default)]
pub struct ArgPromoteExternal;

impl Pass for ArgPromoteExternal {
    const NAME: &'static str = "argpromote_external";
    fn description(&self) -> &'static str {
        "Resolve call arguments to external functions from their materialized C interface"
    }
    fn run(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        targets: &[FunctionId],
    ) -> Result<crate::ModulePassOutcome, String> {
        Ok(
            crate::ModulePassOutcome::functions(argpromote_external_changed_functions(
                ctx, env, targets,
            ))
            .preserving_global::<crate::CallGraphAnalysis>()
            .preserving_global::<crate::AddressAnalysis>(),
        )
    }
}

crate::register_module_pass!(ArgPromoteExternal);
