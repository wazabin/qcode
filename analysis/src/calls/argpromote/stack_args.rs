//! `promote_stack_args`: lift an incoming caller-frame stack read into a
//! by-value input parameter — "level 1" of promoting a stack-passed argument.
//!
//! # Why the RAM channel cannot do this
//!
//! The RAM channel promotes a *deref through a pointer param* (`param + offset`)
//! into a snapshot. `@SP` is an ordinary param and is not excluded from its
//! candidate list, so one might expect `load(@SP + k)` to promote by itself. It
//! does not, and the reason is ordering rather than exclusion: the channel is
//! all-or-nothing per function, bailing when
//! `all_writes_resolvable` fails, and a write is resolvable only if it targets an
//! own-frame local or a constant `promoted-param + offset`. A buffer whose
//! pointer arrives on the stack is written through `load(@SP + k) + i` — the
//! *loaded value*, not a param — so the write is dynamic-address, the function
//! bails, and nothing is promoted. The dependency is circular within one run:
//! the write only becomes resolvable once the pointer is already a param.
//!
//! This pass breaks that cycle. It is a purely local rewrite that needs no model
//! of the function's memory footprint, so it can run where the RAM channel bails.
//! Once it has turned `load(@SP + k)` into a param, the pointer *is* a param, its
//! region is a strided `param + idx*scale + const` the RAM channel already
//! models, and level 2 (`argpromote` proper, then `array_promote` and
//! `loop_to_map`) proceeds normally.
//!
//! On a 32-bit cdecl target this is the difference between every argument being
//! opaque memory and the array pipeline working at all.
//!
//! # Binding / soundness
//!
//! Each direct caller passes the callee's `@SP` value as a positional argument
//! (the `pure_reg` interface). The callee's slot at `@SP + k` is therefore
//! `load(ram, <that @SP argument> + k)` evaluated at the caller immediately
//! before the call — the exact byte the callee reads, whatever the ABI. So the
//! promoted param is bound by threading that load at every direct site via
//! [`append_entry_param_at_sites`], keeping the `param[i] ↔ Call.args[i]`
//! lockstep.
//!
//! Offset 0 — the return-address slot — is promoted like any other: in this model
//! the return address *is* an argument, not a special case. The binding stays
//! correct because the lifter models the return-address push as an explicit store
//! ahead of the `Call`, so the caller-side `load(sp_arg + 0)` inserted before the
//! call observes it.
//!
//! Scope: only functions reachable through **direct** calls are bound; a caller
//! in undiscovered code, or an indirect (`CallInd`) caller, keeps the old stack
//! ABI — the same accepted, unguarded gap as the rest of `argpromote`. Only plain
//! `load(@SP + k)` reads are promoted (the incoming scalar/pointer argument),
//! never writes to the caller frame.

use rustc_hash::FxHashMap as HashMap;
use std::borrow::Cow;

use qcode::space::Space;
use qcode::value::QCodeMut;
use qcode::{
    context::Context,
    value::{
        FunctionBody, FunctionId, ModuleView, Value, ValueId, VarnodeId,
        insn::{InstructionId, Mnemonic},
    },
};

use crate::calls::interface::append_entry_param_at_sites;
use crate::gvn::affine::precompute_forms;
use crate::stack::frame::{entry_sp_value, frame_offset};
use crate::{Pass, PipelineEnv};

#[derive(Default)]
pub struct PromoteStackArgs;

impl Pass for PromoteStackArgs {
    const NAME: &'static str = "promote_stack_args";

    fn description(&self) -> &'static str {
        "Promotes incoming caller-frame stack reads to by-value params"
    }

    fn run(
        &self,
        cone: &mut crate::ConeMut,
        env: &PipelineEnv,
    ) -> Result<crate::ModulePassOutcome, String> {
        let targets = cone.cone_functions();
        let Some(sp_reg) = env.sp_varnode else {
            return Ok(crate::ModulePassOutcome::default());
        };
        let graph = crate::CallGraph::analyze(cone.ctx());

        // Only functions whose call interface is positional (`pure_reg`) — the
        // ones whose direct callers pass `@SP` as an argument we can key on.
        let mut changed = rustc_hash::FxHashSet::default();
        for &fid in &targets {
            let f = FunctionBody::from_id(cone.ctx(), fid);
            if f.is_external() || !f.is_reg_materialized() {
                continue;
            }
            // One direct-site query per function, from the single pass-level
            // graph: the sites gate below and every per-slot interface append
            // reuse it. The snapshot stays valid across this pass's mutations —
            // argument appends rewrite existing `Call` instructions in place and
            // the deleted slot loads are not call sites, so no call instruction
            // is created, destroyed, or re-keyed. (Rebuilding the graph inside
            // `append_entry_param` per promoted slot was quadratic in module
            // size and dominated the pass on real binaries.)
            let sites = crate::calls::direct_call_sites(cone.ctx(), &graph, fid);
            if has_implicit_direct_site(cone.ctx(), &sites) {
                continue;
            }
            if promote_one(cone, fid, sp_reg, &sites) {
                changed.insert(fid);
                // `append_entry_param` also rewrites every direct caller (new
                // binding load + extra `Call` arg), so those callers are dirtied
                // too — downstream `only_dirty` stages must revisit them, and the
                // between-pass verifier must include them in scope.
                changed.extend(graph.callers(fid));
            }
        }

        Ok(crate::ModulePassOutcome {
            module_changed: !changed.is_empty(),
            changed_functions: changed,
            type_requests: Vec::new(),
            preserved_analyses: crate::PreservedAnalyses::none(),
        })
    }
}

/// Whether any direct call site of `fid` binds implicitly (a non-regpure,
/// `Opaque` tag).
///
/// The same gate the RAM channel applies (see `ram.rs`, "snapshot arguments"),
/// and for the same reason: the value this pass threads is a `load(ram, @SP + k)`
/// evaluated *at the caller*. Unlike a register input or a global's address
/// literal it is real loaded data with no implicit counterpart, so an `Opaque`
/// site — which by the dual binding convention must carry **zero** args and seed
/// its params at entry — cannot be given one.
///
/// Appending an argument there anyway is precisely what
/// [`verify::pure_reg_call_args`] rule 2 forbids, and it is how this pass came to
/// be blamed for interface breakage in otherwise healthy callees: the callee kept
/// a perfectly consistent interface (root params ⊇ `map.inputs`) while its
/// implicit call site silently acquired an argument.
///
/// [`verify::pure_reg_call_args`]: crate::verify::verify_pure_reg_call_args
fn has_implicit_direct_site(ctx: &Context, sites: &[InstructionId]) -> bool {
    let implicit = sites.iter().any(|&site| {
        !matches!(
            ctx.get_insn(site).mnemonic(),
            Mnemonic::Call(c) if c.tag.is_regpure()
        )
    });
    if implicit {
        qcode::pass_log!(
            debug,
            "promote_stack_args: bail — a direct call site binds implicitly \
             (it cannot carry a caller-evaluated stack load)",
        );
    }
    implicit
}

fn promote_one(
    cone: &mut crate::ConeMut,
    fid: FunctionId,
    sp_reg: VarnodeId,
    sites: &[InstructionId],
) -> bool {
    let Some(entry_sp) = entry_sp_value(ModuleView::new(cone.ctx()), fid, sp_reg) else {
        return false;
    };
    if FunctionBody::from_id(cone.ctx(), fid).root().is_none() {
        return false;
    }
    let numbering = precompute_forms(ModuleView::new(cone.ctx()), fid);
    let ram = cone.ctx().shared.default_space;

    // Distinct incoming caller-frame read slots `(offset >= 0, size)` → the loads
    // that read them. Offset 0 is the return-address slot, which this model treats
    // as argument slot 0 like any other; a negative offset is an own-frame local
    // and belongs to mem2reg, not here.
    let mut slots: HashMap<(i64, usize), Vec<InstructionId>> = HashMap::default();
    for block in FunctionBody::from_id(cone.ctx(), fid).blocks() {
        for insn in block.iter() {
            let Mnemonic::Load(l) = insn.mnemonic() else {
                continue;
            };
            if l.space != ram {
                continue;
            }
            let ptr = l.ptr.qualify(insn.id.func);
            if let Some(off) = frame_offset(ModuleView::new(cone.ctx()), &numbering, entry_sp, ptr)
                && off >= 0
            {
                slots.entry((off, l.size)).or_default().push(insn.id);
            }
        }
    }
    if slots.is_empty() {
        return false;
    }

    // The positional index of the callee's `@SP` param: every direct caller's
    // `Call.args[sp_index]` is the `@SP` value we key the caller-side load on.
    // Only the *param* shape of the entry stack pointer carries an argument slot;
    // when it is a root entry `load(SP)` the function has no call interface to
    // thread a stack argument through, and this bails.
    let Some(sp_index) = FunctionBody::from_id(cone.ctx(), fid)
        .root()
        .and_then(|b| b.params().position(|p| p.id() == entry_sp))
    else {
        return false;
    };
    let ptr_width = Space::from_id(cone.ctx(), ram).addr_size;

    // Deterministic order.
    let mut ordered: Vec<((i64, usize), Vec<InstructionId>)> = slots.into_iter().collect();
    ordered.sort_by_key(|((off, size), _)| (*off, *size));

    // Write phase. `append_entry_param_at_sites` threads a binding load + extra
    // `Call` arg into every direct caller as well as adding the callee's param,
    // so every function it touches must be in the cone: assert each caller then
    // the callee (each individually checked), then run the coordinated write
    // through the callee's whole-context handle. Asserting only now — past every
    // early bail — keeps a no-op promotion from tripping the gate.
    // `has_implicit_direct_site` already guaranteed every `site` is a regpure
    // `Call`, so each is genuinely rewritten.
    for &site in sites {
        let _ = cone.ctx_for(site.func);
    }
    let ctx = cone.ctx_for(fid);

    let mut changed = false;
    for ((off, size), loads) in ordered {
        // Add the by-value param and, at every direct caller, bind it to
        // `load(ram, @SP-argument + off)` — the exact slot the callee reads.
        let Some(param) = append_entry_param_at_sites(
            ctx,
            fid,
            size,
            Some(format!("stack_{off:x}")),
            None,
            sites,
            move |ctx, call_id, block| {
                let mnemonic = ctx.get_insn(call_id).mnemonic().clone();
                // Fall back to a poison (keeping `param[i] ↔ arg[i]` lockstep) at any
                // direct site we cannot key on the callee's `@SP` argument: a non-`Call`
                // site (tail-call/apply, no positional args), or a `Call` whose arg list
                // is shorter than `sp_index` — a caller not (yet) presenting the callee's
                // full positional `pure_reg` interface. Such sites are excluded elsewhere
                // in practice; the poison just avoids an out-of-bounds panic here.
                let sp_arg = match &mnemonic {
                    Mnemonic::Call(c) => c.args.get(sp_index).copied(),
                    _ => None,
                };
                let Some(sp_arg) = sp_arg else {
                    let ty = ctx.shared.types.get_or_make_int(size);
                    return ValueId::Poison(ctx.shared.values.push_poison(ty));
                };
                let sp_val = sp_arg.qualify(call_id.func);
                let mut b = (ctx).builder(block);
                b.set_insert_point_before(call_id);
                let addr = if off == 0 {
                    sp_val
                } else {
                    let k = b.shr().get_const(off as u64, ptr_width);
                    b.push_add(sp_val, k).id()
                };
                b.push_load::<false>(addr, size, ram).id()
            },
        ) else {
            continue;
        };
        if let ValueId::BlockParam(pid) = param {
            ctx.block_param_mut(pid).name = Some(Cow::Owned(format!("stack_{off:x}")));
        }
        for load_id in loads {
            ctx.replace_all_uses_with(ValueId::Instruction(load_id), param);
            ctx.remove_instruction(load_id);
        }
        changed = true;
    }
    changed
}

crate::register_module_pass!(PromoteStackArgs);
