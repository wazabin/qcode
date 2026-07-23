//! `promote_stack_args` (prototype): lift an incoming caller-frame stack read
//! into a by-value input parameter, so the alias oracle sees a real incoming
//! pointer instead of an opaque `load(@SP + k)` (GLOBALS_AS_VARNODES.md follow-up
//! "A": RAM stack reads weren't in the effect channel).
//!
//! The register/RAM channels never promote `@SP`-relative reads: `@SP` is the
//! stack-pointer param, deliberately excluded from functionalization and lacking
//! an argument index, so its derefs (`load(@SP + k)`, `k > 0` — an incoming
//! stack argument) stay as memory. That keeps a stack-passed pointer opaque, so
//! frame-freshness can't prove it disjoint from own-frame locals, which blocks
//! mem2reg from promoting a memory-carried loop counter — and so a counted
//! `map` loop over the buffer is never recognized.
//!
//! This pass promotes each distinct incoming caller-frame read slot to a
//! by-value root param and rewrites the loads to it.
//!
//! # Binding / soundness
//!
//! Each direct caller passes the callee's `@SP` value as a positional argument
//! (the `pure_reg` interface). The callee's slot at `@SP + k` is therefore
//! `load(ram, <that @SP argument> + k)` evaluated at the caller immediately
//! before the call — the exact byte the callee reads, whatever the ABI. So the
//! promoted param is bound by threading that load at every direct site via
//! [`append_entry_param`], keeping the `param[i] ↔ Call.args[i]` lockstep.
//!
//! Prototype scope: only functions this pass can reach through **direct** calls
//! are bound; a caller in undiscovered code, or an indirect (`CallInd`) caller,
//! keeps the old stack ABI — the same accepted, unguarded gap as the rest of
//! `argpromote`. v0 also only promotes plain `load(@SP + k)` reads (the incoming
//! scalar/pointer argument), not writes to the caller frame.

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
use crate::stack::frame::{frame_offset, incoming_sp_param};
use crate::{Pass, PipelineEnv};

pub struct PromoteStackArgs;

impl Default for PromoteStackArgs {
    fn default() -> Self {
        Self
    }
}

impl Pass for PromoteStackArgs {
    const NAME: &'static str = "promote_stack_args";

    fn description(&self) -> &'static str {
        "Promotes incoming caller-frame stack reads to by-value params (prototype)"
    }

    fn run(
        &self,
        cone: &mut crate::ConeMut,
        env: &PipelineEnv,
    ) -> Result<crate::ModulePassOutcome, String> {
        // CONE-HATCH: remove when promote_stack_args migrates.
        let targets = cone.cone_functions();
        let ctx = cone.bypass_cone_unmigrated_hatch();
        let Some(sp_reg) = env.sp_varnode else {
            return Ok(crate::ModulePassOutcome::default());
        };
        let graph = crate::CallGraph::analyze(ctx);

        // Only functions whose call interface is positional (`pure_reg`) — the
        // ones whose direct callers pass `@SP` as an argument we can key on.
        let mut changed = rustc_hash::FxHashSet::default();
        for &fid in &targets {
            let f = FunctionBody::from_id(ctx, fid);
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
            let sites = crate::calls::direct_call_sites(ctx, &graph, fid);
            if has_implicit_direct_site(ctx, &sites) {
                continue;
            }
            if promote_one(ctx, fid, sp_reg, &sites) {
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
    ctx: &mut Context,
    fid: FunctionId,
    sp_reg: VarnodeId,
    sites: &[InstructionId],
) -> bool {
    let Some(sp_param) = incoming_sp_param(ModuleView::new(ctx), fid, sp_reg) else {
        return false;
    };
    let Some(root) = FunctionBody::from_id(ctx, fid).root().map(|b| b.id) else {
        return false;
    };
    let numbering = precompute_forms(ModuleView::new(ctx), fid);
    let ram = ctx.shared.default_space;

    // Distinct incoming caller-frame read slots `(offset > 0, size)` → the loads
    // that read them. Offset 0 is the return-address slot, not an argument.
    let mut slots: HashMap<(i64, usize), Vec<qcode::value::insn::InstructionId>> =
        HashMap::default();
    for block in FunctionBody::from_id(ctx, fid).blocks() {
        for insn in block.iter() {
            let Mnemonic::Load(l) = insn.mnemonic() else {
                continue;
            };
            if l.space != ram {
                continue;
            }
            let ptr = l.ptr.qualify(insn.id.func);
            if let Some(off) = frame_offset(ModuleView::new(ctx), &numbering, sp_param, ptr)
                && off > 0
            {
                slots.entry((off, l.size)).or_default().push(insn.id);
            }
        }
    }
    if slots.is_empty() {
        return false;
    }
    let _ = root;

    // The positional index of the callee's `@SP` param: every direct caller's
    // `Call.args[sp_index]` is the `@SP` value we key the caller-side load on.
    let Some(sp_index) = FunctionBody::from_id(ctx, fid)
        .root()
        .and_then(|b| b.params().position(|p| p.id() == sp_param))
    else {
        return false;
    };
    let ram = ctx.shared.default_space;
    let ptr_width = Space::from_id(&*ctx, ram).addr_size;

    // Deterministic order.
    let mut ordered: Vec<((i64, usize), Vec<InstructionId>)> = slots.into_iter().collect();
    ordered.sort_by_key(|((off, size), _)| (*off, *size));

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
