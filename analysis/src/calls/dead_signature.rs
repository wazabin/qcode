//! `dead_signature`: drop dead arguments and dead returned values from
//! functionalized (`pure_reg`) functions, rewriting every direct call site to
//! match.
//!
//! Two complementary trims, run to a fixpoint over a worklist:
//!
//! * **dead argument** — an input the body never reads. After
//!   `argpromote_registers` (and the mem2reg/DCE that follows it) a truly-unused
//!   input either has no root block param left (DCE already pruned it) or a param
//!   with zero users. Such an input is removed from `signature.inputs`, its root
//!   param is dropped, and its positional argument is deleted at every direct
//!   call site.
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
//! source of truth and `signature.outputs` (the ABI register list) is analyzed
//! independently, so it is intentionally **not** updated by the field trim.
//!
//! As with `argpromote`, a caller in code we never disassembled would still bind
//! to the old shape; that gap is accepted and unguarded.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use qcode::{
    context::Context,
    types::AggregateField,
    value::{
        BlockParamId, FunctionBody, FunctionId, Instruction, ValueId,
        insn::{Extract, InstructionId, Mnemonic, Return, Tuple},
    },
};

use crate::{Pass, PipelineEnv, remove_entry_param};

/// Bound on worklist iterations: each *changing* iteration strictly removes at
/// least one param or returned field (a quantity bounded by the module), so this
/// only guards against an unforeseen non-terminating rewrite.
const MAX_ITERS: usize = 100_000;

/// Trim dead args and dead returned fields from every `pure_reg` function,
/// rewriting all direct call sites. Returns `true` if anything changed.
pub fn dead_signature(ctx: &mut Context) -> bool {
    let mut changed = false;
    let mut worklist: Vec<FunctionId> = ctx
        .function_ids()
        .into_iter()
        .filter(|&f| FunctionBody::from_id(ctx, f).is_pure_reg())
        .collect();

    // Reverse call-site index, callee → its `Call` instructions, built in one
    // pass. Call targets never change within this pass and instructions are only
    // ever deleted (never retargeted), so this superset stays valid for the whole
    // fixpoint — consumers just skip ids that have since been deleted.
    let call_index = build_call_index(ctx);

    let mut iters = 0;
    while let Some(fid) = worklist.pop() {
        iters += 1;
        if iters > MAX_ITERS {
            break;
        }
        if !FunctionBody::from_id(ctx, fid).is_pure_reg() {
            continue;
        }

        let mut touched: HashSet<FunctionId> = HashSet::default();
        let arg_changed = trim_dead_args(ctx, fid, &call_index, &mut touched);
        let ret_changed = trim_dead_return_fields(ctx, fid, &call_index, &mut touched);

        if arg_changed || ret_changed {
            changed = true;
            touched.insert(fid);
            // DCE the dirtied functions (drops the now-unused arg-setup loads /
            // the returned-field computations the trim exposed), then re-queue
            // them: a freed value may expose the next dead arg or field.
            for t in touched {
                dce_function(ctx, t);
                if !worklist.contains(&t) {
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
fn build_call_index(ctx: &Context) -> HashMap<FunctionId, Vec<InstructionId>> {
    let mut index: HashMap<FunctionId, Vec<InstructionId>> = HashMap::default();
    for insn in ctx.instructions() {
        if let Mnemonic::Call(c) = insn.mnemonic()
            && let Some(target) = c.target.real()
        {
            index.entry(target).or_default().push(insn.id);
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
/// are aligned index-for-index with `input_regs` and every caller's `Call.args`,
/// so a param with no users is a dead argument; drop each through the shared
/// [`remove_entry_param`], which keeps all three in lockstep. Records each caller
/// in `touched`. Returns `true` if anything changed.
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

    let params = ctx.block(root).params.clone();
    let dead: Vec<usize> = params
        .iter()
        .enumerate()
        .filter(|(_, p)| {
            let p = BlockParamId::new(root.func, **p);
            ctx.users(p).is_empty() && !ctx.block_param(p).protected
        })
        .map(|(i, _)| i)
        .collect();
    if dead.is_empty() {
        return false;
    }

    for call_id in direct_call_sites(ctx, fid, call_index) {
        if let Some(caller) = ctx.get_insn(call_id).function().map(|f| f.id) {
            touched.insert(caller);
        }
    }
    // Remove high index first so the lower indices stay valid.
    for &index in dead.iter().rev() {
        remove_entry_param(ctx, fid, index);
    }

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

    if live.iter().all(|&l| l) {
        return false;
    }

    // Kept field indices and the old→new remap for surviving extracts.
    let kept: Vec<usize> = (0..n).filter(|&i| live[i]).collect();
    let mut new_index = vec![None; n];
    for (new_i, &old_i) in kept.iter().enumerate() {
        new_index[old_i] = Some(new_i);
    }

    // The trimmed aggregate type, shared by the call results and the callee
    // return tuples (same field names/types ⇒ same structural type).
    let new_fields: Vec<AggregateField> = kept.iter().map(|&i| fields[i].clone()).collect();
    let new_ty = ctx.shared.types.get_or_make_named_aggregate(new_fields);

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

    true
}

#[derive(Default)]
pub struct DeadSignature;

impl Pass for DeadSignature {
    const NAME: &'static str = "dead_signature";
    fn description(&self) -> &'static str {
        "Remove dead arguments and dead returned fields from functionalized functions"
    }
    fn run(&self, ctx: &mut Context, _env: &PipelineEnv) -> Result<bool, String> {
        Ok(dead_signature(ctx))
    }
}

crate::register_module_pass!(DeadSignature);

#[cfg(test)]
mod tests {
    use qcode::{
        builder::Builder,
        types::TypeId,
        value::{BasicBlock, BlockId, Varnode, VarnodeId, insn::Call},
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
        let tuple = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, ret_block));
            b.set_insert_point_before(ret_id);
            b.push_named_tuple(fields).id
        };
        tc.ctx.replace_instruction_mnemonic(
            ret_id,
            Mnemonic::Return(Return {
                ptr,
                value: Some(ValueId::Instruction(tuple).localize(ret_id.func)),
            }),
        );
        FunctionBody::from_id_mut(&mut tc.ctx, fid).set_input_regs(inputs);
        FunctionBody::from_id_mut(&mut tc.ctx, fid).set_pure_reg(true);
        tc.ctx.stored_type_of(ValueId::Instruction(tuple)).unwrap()
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
            let mut bld = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, g_cont));
            bld.set_insert_point_to_start();
            let f0 = bld.push_extract(ValueId::Instruction(call_id), 0).id();
            bld.push_store(f0, ValueId::Varnode(vr0), reg_space);
        }

        assert!(
            dead_signature(&mut tc.ctx),
            "the unread r0 arg should be dropped"
        );

        assert_eq!(
            FunctionBody::from_id(&tc.ctx, f).input_regs().unwrap(),
            &[vr1],
            "only the read input survives"
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
            let mut bld = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, g_cont));
            bld.set_insert_point_to_start();
            let f1 = bld.push_extract(ValueId::Instruction(call_id), 1);
            let id = f1.id;
            let val = f1.id();
            bld.push_store(val, ValueId::Varnode(r0), tc.reg_space);
            id
        };

        assert!(
            dead_signature(&mut tc.ctx),
            "field 0 is unprojected and should be trimmed"
        );

        assert_eq!(
            return_field_count(&tc, f),
            Some(1),
            "the callee return tuple keeps only the live field"
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

        assert!(dead_signature(&mut tc.ctx));
        assert_eq!(
            return_field_count(&tc, f),
            None,
            "an entirely-unused return drops its value"
        );
    }

    /// A function that was never functionalized (`pure_reg == false`) is left
    /// untouched even with an unread param.
    #[test]
    fn skips_non_pure_reg_functions() {
        let mut tc = qcode::testing::TestContext::new();
        let (vr0, vr1) = (tc.r0, tc.r1);
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
        FunctionBody::from_id_mut(&mut tc.ctx, f).set_input_regs(vec![vr0, vr1]);
        // Deliberately not marked pure_reg.
        let a = tc.ctx.get_const(0x10, 8).id();
        let b = tc.ctx.get_const(0x20, 8).id();
        let call_id = set_call(&mut tc, g_call, f, vec![a, b]);
        tc.ctx.add_cfg_edge(g_call, g_cont);

        assert!(
            !dead_signature(&mut tc.ctx),
            "non-pure-reg functions are skipped"
        );
        assert_eq!(call_args(&tc, call_id).len(), 2, "no argument is dropped");
        let _ = Varnode::from_id(&tc.ctx, vr1);
    }
}
