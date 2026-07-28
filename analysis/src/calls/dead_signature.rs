//! `dead_signature`: drop dead arguments and dead returned values from
//! functionalized (`pure_reg`) functions, rewriting every direct call site to
//! match.
//!
//! Two complementary trims, run to a fixpoint over a worklist:
//!
//! * **dead argument** — an input the body never reads. After
//!   `argpromote_registers` (and the mem2reg/DCE that follows it) a truly-unused
//!   input either has no root block param left (DCE already pruned it) or a param
//!   with zero users. Such an input has its root param dropped and its positional
//!   argument deleted at every direct call site.
//! * **dead returned field** — a `pure_reg` function returns its register writes
//!   as an aggregate write-set; the caller projects each field with an `extract`.
//!   A field whose `extract` is dead at *every* call site has already been DCE'd
//!   away (aggregate values are extract-only by invariant, so a missing extract
//!   *is* the deadness signal — we delete none). Such fields are dropped from the
//!   callee's return `Tuple`, the call-result aggregate type is rebuilt, and the
//!   surviving extracts are renumbered through one old→new index map.
//!
//! ## Scope and soundness
//!
//! Only `pure_reg` functions are touched. That flag (set by
//! `argpromote_registers` on success) already implies the function is
//! non-external, is not address-taken, and has only direct callers — so the
//! closed-world rewrite reaches every caller. The functionalized return is the
//! source of truth, so the field trim rewrites only the return `Tuple` and the
//! call-site extracts.
//!
//! As with `argpromote`, a caller in code we never disassembled would still bind
//! to the old shape; that gap is accepted and unguarded.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use qcode::{
    context::Context,
    types::AggregateField,
    value::{
        BlockParamId, FunctionBody, FunctionId, Instruction, QCodeMut, ValueId,
        insn::{Extract, InstructionId, Mnemonic, Return, Tuple},
    },
};

use crate::{CallGraph, Pass, PipelineEnv, calls::interface::remove_entry_params_at_sites};

/// Bound on worklist iterations: each *changing* iteration strictly removes at
/// least one param or returned field (a quantity bounded by the module), so this
/// only guards against an unforeseen non-terminating rewrite.
const MAX_ITERS: usize = 100_000;

/// Trim unused arguments and architecture-approved dead returned fields from
/// every `pure_reg` function, rewriting all direct call sites.
pub fn dead_signature(
    ctx: &mut Context,
    killable_registers: &HashSet<qcode::value::VarnodeId>,
) -> bool {
    let targets = ctx.function_ids();
    let graph = CallGraph::analyze(ctx);
    let mut cone = crate::ConeMut::full(ctx);
    !dead_signature_changed_functions(&mut cone, &targets, &graph, killable_registers).is_empty()
}

fn dead_signature_changed_functions(
    cone: &mut crate::ConeMut,
    targets: &[FunctionId],
    graph: &CallGraph,
    killable: &HashSet<qcode::value::VarnodeId>,
) -> HashSet<FunctionId> {
    let mut changed = HashSet::default();
    let target_set: HashSet<_> = targets.iter().copied().collect();
    let mut worklist: Vec<FunctionId> = targets
        .iter()
        .copied()
        .filter(|&f| FunctionBody::from_id(cone.ctx(), f).is_reg_materialized())
        .collect();
    let mut queued: HashSet<_> = worklist.iter().copied().collect();

    // Reverse call-site index, callee → its `Call` instructions, built in one
    // pass. Call targets never change within this pass and instructions are only
    // ever deleted (never retargeted), so this superset stays valid for the whole
    // fixpoint — consumers just skip ids that have since been deleted.
    let call_index = build_call_index(cone.ctx(), graph);

    let mut iters = 0;
    while let Some(fid) = worklist.pop() {
        queued.remove(&fid);
        iters += 1;
        if iters > MAX_ITERS {
            break;
        }
        if !FunctionBody::from_id(cone.ctx(), fid).is_reg_materialized() {
            continue;
        }
        let sites = direct_call_sites(cone.ctx(), fid, &call_index);
        if sites.iter().any(|site| !target_set.contains(&site.func)) {
            continue;
        }

        // Every function this iteration may write — the callee `fid` and each
        // direct caller — must be in the cone. The `target_set` guard above
        // already confines them to `targets` (= the cone on a narrowed run);
        // these `ctx_for` asserts turn that into a hard tripwire (each caller
        // then the callee, all individually checked).
        for site in &sites {
            let _ = cone.ctx_for(site.func);
        }
        let ctx = cone.ctx_for(fid);

        let mut touched: HashSet<FunctionId> = HashSet::default();
        let arg_changed = trim_dead_args(ctx, fid, &call_index, &mut touched);
        let ret_changed = trim_dead_return_fields(ctx, fid, &call_index, killable, &mut touched);

        if arg_changed || ret_changed {
            touched.insert(fid);
            // DCE the dirtied functions (drops the now-unused arg-setup loads /
            // the returned-field computations the trim exposed), then re-queue
            // them: a freed value may expose the next dead arg or field.
            for t in touched {
                changed.insert(t);
                dce_function(ctx, t);
                if queued.insert(t) {
                    worklist.push(t);
                }
            }
        }
    }
    changed
}

/// Build the reverse call-site index `callee → its Call instructions` in one
/// scan over every instruction, replacing the per-callee rescan the trims used
/// to do (which made the worklist `O(functions × total_instructions)`).
fn build_call_index(ctx: &Context, graph: &CallGraph) -> HashMap<FunctionId, Vec<InstructionId>> {
    let mut index: HashMap<FunctionId, Vec<InstructionId>> = HashMap::default();
    for callee in ctx.function_ids() {
        let sites = super::direct_call_sites(ctx, graph, callee);
        if !sites.is_empty() {
            index.insert(callee, sites);
        }
    }
    index
}

/// Live direct call sites of `fid` from the prebuilt index: the recorded ids,
/// minus any deleted since the index was built.
fn direct_call_sites(
    ctx: &Context,
    fid: FunctionId,
    call_index: &HashMap<FunctionId, Vec<InstructionId>>,
) -> Vec<InstructionId> {
    call_index
        .get(&fid)
        .into_iter()
        .flatten()
        .copied()
        .filter(|&id| ctx.contains_instruction(id))
        .collect()
}

/// Every `Return` terminator in `fid`.
fn returns_of(ctx: &Context, fid: FunctionId) -> Vec<InstructionId> {
    FunctionBody::from_id(ctx, fid)
        .iter()
        .filter_map(|b| {
            let last = b.iter().last()?;
            matches!(last.mnemonic(), Mnemonic::Return(_)).then_some(last.id)
        })
        .collect()
}

/// Run per-block DCE across `fid`, removing the instructions a trim made dead.
fn dce_function(ctx: &mut Context, fid: FunctionId) {
    let blocks: Vec<_> = FunctionBody::from_id(ctx, fid)
        .blocks()
        .map(|b| b.id)
        .collect();
    for b in blocks {
        crate::remove_dead_insns(ctx, b);
    }
}

// ---------------------------------------------------------------------------
// dead arguments
// ---------------------------------------------------------------------------

/// Remove every input `fid` never reads. A `pure_reg` function's entry params
/// are aligned index-for-index with every caller's `Call.args`, so a param with
/// no users is a dead argument; drop them together through the shared interface
/// helper, which keeps both in lockstep. Records each caller in `touched`.
/// Returns `true` if anything changed.
///
/// This is the same operation DCE's no-pred param sweep performs, so the two stay
/// consistent; running it here as well lets the dead-signature worklist expose
/// and reclaim dead args between return-field trims without a separate DCE round.
fn trim_dead_args(
    ctx: &mut Context,
    fid: FunctionId,
    call_index: &HashMap<FunctionId, Vec<InstructionId>>,
    touched: &mut HashSet<FunctionId>,
) -> bool {
    let Some(root) = FunctionBody::from_id(ctx, fid).root().map(|b| b.id) else {
        return false;
    };

    let inputs = FunctionBody::from_id(ctx, fid)
        .effects()
        .materialized()
        .map(|map| map.inputs.clone())
        .unwrap_or_default();
    let params = ctx.block(root).params.clone();
    let dead: Vec<usize> = params
        .iter()
        .enumerate()
        .filter(|(i, p)| {
            if inputs.get(*i).is_none() {
                return false;
            }
            let p = BlockParamId::new(root.func, **p);
            ctx.users(p).is_empty()
        })
        .map(|(i, _)| i)
        .collect();
    if dead.is_empty() {
        return false;
    }

    let call_sites = direct_call_sites(ctx, fid, call_index);
    for &call_id in &call_sites {
        if let Some(caller) = ctx.get_insn(call_id).function().map(|f| f.id) {
            touched.insert(caller);
        }
    }
    remove_entry_params_at_sites(ctx, fid, &dead, &call_sites);

    true
}

// ---------------------------------------------------------------------------
// dead returned fields
// ---------------------------------------------------------------------------

/// Drop every returned write-set field that no caller projects: trim the callee
/// return `Tuple`s, rebuild the call-result aggregate type, and renumber the
/// surviving extracts. Records each caller in `touched`. Returns `true` if
/// anything changed.
fn trim_dead_return_fields(
    ctx: &mut Context,
    fid: FunctionId,
    call_index: &HashMap<FunctionId, Vec<InstructionId>>,
    killable: &HashSet<qcode::value::VarnodeId>,
    touched: &mut HashSet<FunctionId>,
) -> bool {
    let call_sites = direct_call_sites(ctx, fid, call_index);
    if call_sites.is_empty() {
        return false;
    }

    // The returned aggregate type (uniform across call sites). A `pure_reg`
    // function always returns the write-set aggregate.
    let Some(agg_ty) = ctx.stored_type_of(ValueId::Instruction(call_sites[0])) else {
        return false;
    };
    let Some(fields) = ctx.shared.types.aggregate_fields(agg_ty).map(<[_]>::to_vec) else {
        return false;
    };
    let n = fields.len();
    if n == 0 {
        return false;
    }
    let Some(map) = FunctionBody::from_id(ctx, fid)
        .effects()
        .materialized()
        .cloned()
    else {
        return false;
    };
    if map.returns != n || map.outputs.len() < n {
        return false;
    }

    // A field is live iff some surviving `extract` projects it at some call site.
    // Aggregate results are extract-only by invariant; a non-extract use would
    // mean we cannot reason per-field, so bail (keep everything) in release.
    let mut live = vec![false; n];
    for &call_id in &call_sites {
        let result = ValueId::Instruction(call_id);
        for u in ctx.users(result) {
            match ctx.get_insn(u).mnemonic() {
                Mnemonic::Extract(e) if e.agg.qualify(u.func) == result => {
                    if let Some(slot) = live.get_mut(e.index) {
                        *slot = true;
                    }
                }
                _ => {
                    debug_assert!(false, "aggregate call result used by a non-extract");
                    return false;
                }
            }
        }
    }
    for (i, &output) in map.outputs[..n].iter().enumerate() {
        if !killable.contains(&output) {
            live[i] = true;
        }
    }

    if live.iter().all(|&l| l) {
        return false;
    }

    // Kept field indices and the old→new remap for surviving extracts.
    let kept: Vec<usize> = (0..n).filter(|&i| live[i]).collect();
    let mut new_index = vec![None; n];
    for (new_i, &old_i) in kept.iter().enumerate() {
        new_index[old_i] = Some(new_i);
    }

    // Preserve a function-owned return record's identity while revising its
    // fields. Legacy structural aggregates still use structural interning.
    let new_fields: Vec<AggregateField> = kept.iter().map(|&i| fields[i].clone()).collect();
    let new_ty = if ctx.shared.types.function_return(fid) == Some(agg_ty) {
        ctx.shared
            .types
            .edit_function_return(fid, new_fields)
            .expect("owned function return type must remain editable")
    } else {
        ctx.shared.types.get_or_make_named_aggregate(new_fields)
    };

    // Rewrite each callee return: trim the tuple to the kept fields, or drop the
    // returned value entirely when nothing survives.
    for ret_id in returns_of(ctx, fid) {
        let Mnemonic::Return(Return {
            ptr,
            value: Some(value),
        }) = ctx.get_insn(ret_id).mnemonic().clone()
        else {
            continue;
        };
        let ValueId::Instruction(tuple_id) = value.qualify(ret_id.func) else {
            continue;
        };
        let Mnemonic::Tuple(tuple) = ctx.get_insn(tuple_id).mnemonic().clone() else {
            continue;
        };

        if kept.is_empty() {
            ctx.replace_instruction_mnemonic(ret_id, Mnemonic::Return(Return { ptr, value: None }));
            continue;
        }

        let new_tuple_fields: Vec<qcode::value::LocalValueId> = kept
            .iter()
            .filter_map(|&i| tuple.fields.get(i).copied())
            .collect();
        ctx.replace_instruction_mnemonic(
            tuple_id,
            Mnemonic::Tuple(Tuple {
                fields: new_tuple_fields,
            }),
        );
        Instruction::from_id_mut(ctx, tuple_id).set_type_resized(new_ty);
    }

    // Retype each call result and renumber the surviving extracts.
    for &call_id in &call_sites {
        Instruction::from_id_mut(ctx, call_id).set_type_resized(new_ty);
        if let Some(caller) = ctx.get_insn(call_id).function().map(|f| f.id) {
            touched.insert(caller);
        }
        let result = ValueId::Instruction(call_id);
        let extracts: Vec<InstructionId> = ctx.users(result).to_vec();
        for u in extracts {
            let Mnemonic::Extract(Extract { agg, index }) = ctx.get_insn(u).mnemonic().clone()
            else {
                continue;
            };
            if let Some(new_i) = new_index[index] {
                ctx.replace_instruction_mnemonic(
                    u,
                    Mnemonic::Extract(Extract { agg, index: new_i }),
                );
            }
        }
    }

    let mut outputs: Vec<_> = kept.iter().map(|&i| map.outputs[i]).collect();
    outputs.extend_from_slice(&map.outputs[n..]);
    FunctionBody::from_id_mut(ctx, fid).set_register_effects(
        qcode::value::RegisterChannelState::Materialized(qcode::value::RegisterInterfaceMap {
            inputs: map.inputs,
            outputs,
            returns: kept.len(),
            projections: map.projections,
        }),
    );

    true
}

#[derive(Default)]
pub struct DeadSignature;

impl Pass for DeadSignature {
    const NAME: &'static str = "dead_signature";
    fn description(&self) -> &'static str {
        "Remove unused arguments and architecture-approved dead return effects"
    }
    fn run(
        &self,
        cone: &mut crate::ConeMut,
        env: &PipelineEnv,
    ) -> Result<crate::ModulePassOutcome, String> {
        let targets = cone.cone_functions();
        let graph = CallGraph::analyze(cone.ctx());
        Ok(
            crate::ModulePassOutcome::functions(dead_signature_changed_functions(
                cone,
                &targets,
                &graph,
                &env.cfg.killable_registers,
            ))
            .preserving_global::<crate::CallGraphAnalysis>()
            .preserving_global::<crate::AddressAnalysis>(),
        )
    }

    fn run_with_analyses(
        &self,
        cone: &mut crate::ConeMut,
        env: &PipelineEnv,
        analyses: &mut crate::AnalysisManager,
    ) -> Result<crate::ModulePassOutcome, String> {
        let targets = cone.cone_functions();
        let graph = analyses.global::<crate::CallGraphAnalysis>(cone.ctx());
        Ok(
            crate::ModulePassOutcome::functions(dead_signature_changed_functions(
                cone,
                &targets,
                graph,
                &env.cfg.killable_registers,
            ))
            .preserving_global::<crate::CallGraphAnalysis>()
            .preserving_global::<crate::AddressAnalysis>(),
        )
    }
}

crate::register_module_pass!(DeadSignature);

#[cfg(test)]
mod tests {
    use qcode::{
        types::{AggregateField, TypeId},
        value::{BasicBlock, BlockId, VarnodeId, insn::Call},
    };
    use qcode_macro::qcode;

    use super::*;

    /// Give `block`'s call instruction the target `target` and `args`.
    fn set_call(
        tc: &mut qcode::testing::TestContext,
        block: BlockId,
        target: FunctionId,
        args: Vec<ValueId>,
    ) -> InstructionId {
        let call_id = BasicBlock::from_id(&tc.ctx, block)
            .iter()
            .find(|i| matches!(i.mnemonic(), Mnemonic::Call(_)))
            .unwrap()
            .id;
        tc.ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target: qcode::value::insn::Callee::Real(target),
                args: args
                    .into_iter()
                    .map(|arg| arg.localize(call_id.func))
                    .collect(),
                clobbers: vec![],
                tag: Default::default(),
            }),
        );
        call_id
    }

    /// Turn `fid` into the `pure_reg` shape: attach a write-set `Tuple` of
    /// `fields` as the single return's value, mark it pure-reg, and record its
    /// `inputs`. Returns the returned aggregate's type.
    fn make_pure_reg_return(
        tc: &mut qcode::testing::TestContext,
        fid: FunctionId,
        inputs: Vec<VarnodeId>,
        fields: Vec<(String, ValueId)>,
    ) -> TypeId {
        let ret_id = returns_of(&tc.ctx, fid)[0];
        let ret_block = tc.ctx.get_insn(ret_id).parent().map(|b| b.id).unwrap();
        let Mnemonic::Return(Return { ptr, .. }) = tc.ctx.get_insn(ret_id).mnemonic().clone()
        else {
            unreachable!()
        };
        let field_count = fields.len();
        let return_fields = fields
            .iter()
            .map(|(name, value)| AggregateField::new(name.clone(), tc.ctx.type_of(*value)))
            .collect();
        let return_type = tc
            .ctx
            .shared
            .types
            .create_function_return(fid, return_fields)
            .unwrap();
        let tuple = {
            let mut b = tc.ctx.builder(ret_block);
            b.set_insert_point_before(ret_id);
            b.push_named_tuple_with_type(fields, return_type).id
        };
        tc.ctx.replace_instruction_mnemonic(
            ret_id,
            Mnemonic::Return(Return {
                ptr,
                value: Some(ValueId::Instruction(tuple).localize(ret_id.func)),
            }),
        );
        FunctionBody::from_id_mut(&mut tc.ctx, fid).set_register_effects(
            qcode::value::RegisterChannelState::Materialized(qcode::value::RegisterInterfaceMap {
                inputs,
                outputs: [tc.r0, tc.r1, tc.r2, tc.r3][..field_count].to_vec(),
                returns: field_count,
                projections: Vec::new(),
            }),
        );
        return_type
    }

    fn killable(tc: &qcode::testing::TestContext) -> HashSet<VarnodeId> {
        [
            tc.r0,
            tc.r1,
            tc.r2,
            tc.r3,
            tc.r0_lo32,
            tc.r0_lo16,
            tc.r0_byte0,
            tc.r0_byte1,
            tc.r0_byte2,
            tc.r0_byte3,
        ]
        .into_iter()
        .collect()
    }

    /// Field count of `fid`'s single return write-set tuple, or `None` if the
    /// return carries no value.
    fn return_field_count(tc: &qcode::testing::TestContext, fid: FunctionId) -> Option<usize> {
        let ret = returns_of(&tc.ctx, fid)[0];
        let Mnemonic::Return(Return { value: Some(v), .. }) = tc.ctx.get_insn(ret).mnemonic()
        else {
            return None;
        };
        let qcode::value::LocalValueId::Instruction(t) = v else {
            return None;
        };
        match tc.ctx.get_insn(InstructionId::new(ret.func, *t)).mnemonic() {
            Mnemonic::Tuple(t) => Some(t.fields.len()),
            _ => None,
        }
    }

    fn call_args(tc: &qcode::testing::TestContext, call_id: InstructionId) -> Vec<ValueId> {
        match tc.ctx.get_insn(call_id).mnemonic() {
            Mnemonic::Call(c) => c.args.iter().map(|arg| arg.qualify(call_id.func)).collect(),
            _ => unreachable!(),
        }
    }

    /// An input whose root param is never read is dropped from the signature and
    /// from the call site, while the read input (and its argument) survives.
    #[test]
    fn drops_unread_argument() {
        let mut tc = qcode::testing::TestContext::new();
        let (vr0, vr1) = (tc.r0, tc.r1);
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry @r0:i64 @r1:i64>
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
            "
        );
        let _ = g;
        // Production names each promoted register param after its register (so
        // `input_arg_name(i)` matches the param); the qcode macro leaves them
        // unnamed, so mirror the naming here. Params are positional: [0]↔r0,
        // [1]↔r1.
        let param_ids: Vec<ValueId> = BasicBlock::from_id(&tc.ctx, entry)
            .params()
            .map(|p| p.id())
            .collect();
        for (pv, name) in param_ids.iter().zip(["r0", "r1"]) {
            if let ValueId::BlockParam(pid) = pv {
                tc.ctx.block_param_mut(*pid).name = Some(std::borrow::Cow::Owned(name.into()));
            }
        }
        // f returns a one-field write-set of its r1 param; the r0 param is unused
        // (the post-mem2reg pure-reg shape: the body reads params, not varnodes).
        let r1_param = param_ids[1];
        let agg = make_pure_reg_return(
            &mut tc,
            f,
            vec![vr0, vr1],
            vec![("o0".to_owned(), r1_param)],
        );

        let a = tc.ctx.get_const(0x10, 8).id();
        let b = tc.ctx.get_const(0x20, 8).id();
        let call_id = set_call(&mut tc, g_call, f, vec![a, b]);
        tc.ctx.add_cfg_edge(g_call, g_cont);
        // Keep the single return field live so only the arg trim fires.
        Instruction::from_id_mut(&mut tc.ctx, call_id).set_type(agg);
        {
            let reg_space = tc.reg_space;
            let mut bld = tc.ctx.builder(g_cont);
            bld.set_insert_point_to_start();
            let f0 = bld.push_extract(ValueId::Instruction(call_id), 0).id();
            bld.push_store(f0, ValueId::Varnode(vr0), reg_space);
        }

        let killable = HashSet::default();
        assert!(
            dead_signature(&mut tc.ctx, &killable),
            "unused arguments are removable regardless of return-effect policy"
        );

        assert_eq!(
            FunctionBody::from_id(&tc.ctx, f)
                .root()
                .unwrap()
                .params()
                .count(),
            1,
            "only the read input survives as a root param"
        );
        assert_eq!(
            call_args(&tc, call_id),
            vec![b],
            "the dropped input's argument is removed at the call site"
        );
        assert_eq!(
            BasicBlock::from_id(&tc.ctx, entry).params().count(),
            1,
            "the dead root param is removed"
        );
    }

    /// A promoted caller-frame snapshot is a RAM-channel parameter after the
    /// register-input prefix. Once cleanup forwards the snapshot directly, the
    /// stack-register parameter is dead and must be removable without removing
    /// or reindexing the snapshot as though it were another register input.
    #[test]
    fn drops_dead_stack_register_but_keeps_ram_snapshot_param() {
        let mut tc = qcode::testing::TestContext::new();
        let rsp_reg = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry @rsp:i64 @rsp_val_0:i64>
                    return at @rsp_val_0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
            "
        );
        let _ = g;
        FunctionBody::from_id_mut(&mut tc.ctx, f).set_register_effects(
            qcode::value::RegisterChannelState::Materialized(qcode::value::RegisterInterfaceMap {
                inputs: vec![rsp_reg],
                outputs: vec![],
                returns: 0,
                projections: Vec::new(),
            }),
        );
        let stack = tc.ctx.get_const(0x1000, 8).id();
        let return_address = tc.ctx.get_const(0x4000, 8).id();
        let call_id = set_call(&mut tc, g_call, f, vec![stack, return_address]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        let killable = [rsp_reg].into_iter().collect();
        assert!(dead_signature(&mut tc.ctx, &killable));

        let params: Vec<_> = FunctionBody::from_id(&tc.ctx, f)
            .root()
            .unwrap()
            .params()
            .collect();
        assert_eq!(params.len(), 1, "only the RAM snapshot should remain");
        assert_eq!(params[0].name(), Some("rsp_val_0"));
        assert_eq!(
            call_args(&tc, call_id),
            vec![return_address],
            "the snapshot's positional call argument must remain"
        );
        assert!(
            FunctionBody::from_id(&tc.ctx, f)
                .effects()
                .materialized()
                .unwrap()
                .inputs
                .is_empty(),
            "the stale stack-register effect must be pruned"
        );
    }

    /// With more than one caller, dropping a dead register input trims the root
    /// param, shrinks the materialized `inputs` map in lockstep, and removes the
    /// dead argument at *every* call site — the map is per-callee (updated once),
    /// the arguments per-site (updated for each). Guards the interface-desync the
    /// `materialized_interface` verifier catches.
    #[test]
    fn drops_unread_argument_across_multiple_callers() {
        let mut tc = qcode::testing::TestContext::new();
        let (vr0, vr1) = (tc.r0, tc.r1);
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry @r0:i64 @r1:i64>
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;

            fn h:
                <h_entry>
                    goto <h_call>;
                <h_call>
                    call <f>;
                <h_cont>
                    return at i64 0;
            "
        );
        let _ = (g, h);
        // Params are positional: [0]↔r0, [1]↔r1; r0 goes unread.
        let param_ids: Vec<ValueId> = BasicBlock::from_id(&tc.ctx, entry)
            .params()
            .map(|p| p.id())
            .collect();
        for (pv, name) in param_ids.iter().zip(["r0", "r1"]) {
            if let ValueId::BlockParam(pid) = pv {
                tc.ctx.block_param_mut(*pid).name = Some(std::borrow::Cow::Owned(name.into()));
            }
        }
        let r1_param = param_ids[1];
        let agg = make_pure_reg_return(
            &mut tc,
            f,
            vec![vr0, vr1],
            vec![("o0".to_owned(), r1_param)],
        );

        // Two callers, each passing [a, b]; only b (the r1 arg) should survive.
        let mut call_sites = Vec::new();
        for (call_block, cont_block) in [(g_call, g_cont), (h_call, h_cont)] {
            let a = tc.ctx.get_const(0x10, 8).id();
            let b = tc.ctx.get_const(0x20, 8).id();
            let call_id = set_call(&mut tc, call_block, f, vec![a, b]);
            tc.ctx.add_cfg_edge(call_block, cont_block);
            Instruction::from_id_mut(&mut tc.ctx, call_id).set_type(agg);
            {
                let reg_space = tc.reg_space;
                let mut bld = tc.ctx.builder(cont_block);
                bld.set_insert_point_to_start();
                let f0 = bld.push_extract(ValueId::Instruction(call_id), 0).id();
                bld.push_store(f0, ValueId::Varnode(vr0), reg_space);
            }
            call_sites.push((call_id, b));
        }

        let killable = killable(&tc);
        assert!(
            dead_signature(&mut tc.ctx, &killable),
            "the unread r0 arg should be dropped"
        );

        // Callee: one surviving root param and a map whose inputs shrank to [r1].
        assert_eq!(
            BasicBlock::from_id(&tc.ctx, entry).params().count(),
            1,
            "the dead root param is removed"
        );
        let qcode::value::RegisterChannelState::Materialized(map) =
            FunctionBody::from_id(&tc.ctx, f).effects().register.clone()
        else {
            panic!("f stays materialized");
        };
        assert_eq!(
            map.inputs,
            vec![vr1],
            "the dropped input is removed from the interface map in lockstep",
        );

        // Every caller drops the dead argument, keeping the surviving one.
        for (call_id, b) in call_sites {
            assert_eq!(
                call_args(&tc, call_id),
                vec![b],
                "each call site drops the dead arg",
            );
        }
    }

    /// A returned field no caller projects is dropped from the callee tuple, and
    /// a surviving higher-index extract is renumbered down.
    #[test]
    fn trims_unprojected_return_field_and_renumbers() {
        let mut tc = qcode::testing::TestContext::new();
        let r0 = tc.r0;
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry>
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
            "
        );
        let _ = (g, entry);
        let c0 = tc.ctx.get_const(7, 8).id();
        let c1 = tc.ctx.get_const(9, 8).id();
        let agg = make_pure_reg_return(
            &mut tc,
            f,
            vec![],
            vec![("o0".to_owned(), c0), ("o1".to_owned(), c1)],
        );

        let call_id = set_call(&mut tc, g_call, f, vec![]);
        tc.ctx.add_cfg_edge(g_call, g_cont);
        Instruction::from_id_mut(&mut tc.ctx, call_id).set_type(agg);
        // The caller projects only field 1 (the second output).
        let extract_id = {
            let mut bld = tc.ctx.builder(g_cont);
            bld.set_insert_point_to_start();
            let f1 = bld.push_extract(ValueId::Instruction(call_id), 1);
            let id = f1.id;
            let val = f1.id();
            bld.push_store(val, ValueId::Varnode(r0), tc.reg_space);
            id
        };

        let killable = killable(&tc);
        assert!(
            dead_signature(&mut tc.ctx, &killable),
            "field 0 is unprojected and should be trimmed"
        );

        assert_eq!(
            return_field_count(&tc, f),
            Some(1),
            "the callee return tuple keeps only the live field"
        );
        assert_eq!(
            tc.ctx.shared.types.function_return(f),
            Some(agg),
            "trimming must preserve the function-owned return identity"
        );
        assert_eq!(tc.ctx.shared.types.aggregate_fields(agg).unwrap().len(), 1);
        assert_eq!(
            tc.ctx.stored_type_of(ValueId::Instruction(call_id)),
            Some(agg)
        );
        let Mnemonic::Extract(Extract { index, .. }) = tc.ctx.get_insn(extract_id).mnemonic()
        else {
            panic!("surviving projection must still be an extract");
        };
        assert_eq!(*index, 0, "the surviving extract is renumbered from 1 to 0");
    }

    /// When no caller projects any field, the whole returned value is dropped.
    #[test]
    fn drops_return_value_when_all_fields_dead() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry>
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
            "
        );
        let _ = (g, entry);
        let c0 = tc.ctx.get_const(7, 8).id();
        let agg = make_pure_reg_return(&mut tc, f, vec![], vec![("o0".to_owned(), c0)]);
        let call_id = set_call(&mut tc, g_call, f, vec![]);
        tc.ctx.add_cfg_edge(g_call, g_cont);
        Instruction::from_id_mut(&mut tc.ctx, call_id).set_type(agg);
        // No extract: nothing projects the result.

        let killable = killable(&tc);
        assert!(dead_signature(&mut tc.ctx, &killable));
        assert_eq!(
            return_field_count(&tc, f),
            None,
            "an entirely-unused return drops its value"
        );
    }

    #[test]
    fn preserves_unused_non_killable_return() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry>
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
            "
        );
        let _ = (g, entry);
        let c0 = tc.ctx.get_const(7, 8).id();
        let agg = make_pure_reg_return(&mut tc, f, vec![], vec![("o0".to_owned(), c0)]);
        let call_id = set_call(&mut tc, g_call, f, vec![]);
        tc.ctx.add_cfg_edge(g_call, g_cont);
        Instruction::from_id_mut(&mut tc.ctx, call_id).set_type(agg);

        assert!(!dead_signature(&mut tc.ctx, &HashSet::default()));
        assert_eq!(
            return_field_count(&tc, f),
            Some(1),
            "an unused general-purpose return is outside architecture policy"
        );
    }

    /// A function that was never functionalized (`pure_reg == false`) is left
    /// untouched even with an unread param.
    #[test]
    fn skips_non_pure_reg_functions() {
        let mut tc = qcode::testing::TestContext::new();
        qcode!(
            tc.ctx,
            "
            fn f:
                <entry @r0:i64 @r1:i64>
                    %u = @r1 + i64 1;
                    return at i64 0;

            fn g:
                <g_entry>
                    goto <g_call>;
                <g_call>
                    call <f>;
                <g_cont>
                    return at i64 0;
            "
        );
        let _ = (g, entry);
        // Deliberately not marked pure_reg (no materialized effects).
        let a = tc.ctx.get_const(0x10, 8).id();
        let b = tc.ctx.get_const(0x20, 8).id();
        let call_id = set_call(&mut tc, g_call, f, vec![a, b]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        let killable = killable(&tc);
        assert!(
            !dead_signature(&mut tc.ctx, &killable),
            "non-pure-reg functions are skipped"
        );
        assert_eq!(call_args(&tc, call_id).len(), 2, "no argument is dropped");
    }
}
