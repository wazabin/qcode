//! Selectively rematerialize SSA expressions from their current stored values.
//!
//! This is the presentation-oriented inverse of store-to-load forwarding.  It
//! uses the same byte-precise [`MemoryState`](crate::memory_state::MemoryState),
//! alias invalidation, call pruning, and dominator walk as GVN, so a reload is
//! only introduced after the store establishing that exact memory version.

use std::any::Any;

use jstd::graph::analysis::DominatorTree;
use qcode::value::{
    QCodeView, ValueId,
    block::BlockId,
    insn::{Binary, Binop, IntBinop, Load, Mnemonic},
};

use crate::{
    AliasAnalysis, AliasResult, ContextView, FunctionBody, FunctionPass, LocalAnalysisManager,
    Outcome, memory_state::MemoryState,
};

use super::{
    affine::Numbering,
    walk::{Claim, Editor, InsnCtx, SubPass, run_dominator_walk},
};

struct RematerializeVariables;

impl<'str> SubPass<'str> for RematerializeVariables {
    fn init_state(&self) -> Box<dyn Any> {
        Box::new(MemoryState::default())
    }

    fn clone_state(&self, state: &dyn Any) -> Box<dyn Any> {
        Box::new(
            state
                .downcast_ref::<MemoryState>()
                .expect("memory state")
                .clone(),
        )
    }

    fn on_block_entry(
        &self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        state: &mut dyn Any,
        block: BlockId,
        tree: &DominatorTree<BlockId>,
        aliases: Option<&AliasResult>,
        numbering: &Numbering,
        is_shared: bool,
    ) {
        let state = state.downcast_mut::<MemoryState>().expect("memory state");
        if is_shared {
            state.clear();
        }
        state.prune_join_paths(cx.body_view(body), block, tree, aliases, numbering);
        state.prune_loop_carried(cx.body_view(body), block, tree, aliases, numbering);
    }

    fn on_insn(
        &self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        state: &mut dyn Any,
        ic: &InsnCtx,
        ed: &mut Editor,
    ) -> Claim {
        let state = state.downcast_mut::<MemoryState>().expect("memory state");
        match ic.mnemonic {
            Mnemonic::Store(store) => {
                state.record_store(body, cx, ic.insn_id.func, store, ic.aliases, ic.numbering);
                Claim::Done
            }
            Mnemonic::Load(load) => {
                state.define_load(
                    cx.body_view(body),
                    ic.insn_id.func,
                    load,
                    ic.id,
                    ic.aliases,
                    ic.numbering,
                );
                Claim::Done
            }
            _ if ic.size != 0 => {
                let locations = state.stored_locations();

                // Replace operands that are themselves the current value of a
                // stored variable. This handles aggregate return packs: the
                // value's definition precedes its store, but the tuple using it
                // follows the store and must read the variable's current version.
                let mut rebuilt = ic.mnemonic.clone();
                let mut replaced_operand = false;
                let rematerialize_operands = matches!(ic.mnemonic, Mnemonic::Tuple(_));
                for operand in ic
                    .mnemonic
                    .args()
                    .into_iter()
                    .map(|operand| operand.qualify(ic.insn_id.func))
                {
                    if !rematerialize_operands {
                        break;
                    }
                    // Look through a width-changing wrapper. Machine return
                    // registers commonly contain `zext(stored_i32)`; rebuilding
                    // that wrapper at the return pack lets its source be the
                    // current variable read rather than the pre-store SSA value.
                    let wrapped = match operand {
                        ValueId::Instruction(id) => {
                            let mnemonic = cx.body_view(body).insn_ref(id).mnemonic();
                            matches!(
                                mnemonic,
                                Mnemonic::Zext(_) | Mnemonic::Sext(_) | Mnemonic::Range(_)
                            )
                            .then(|| mnemonic.args().first().copied().map(|v| v.qualify(id.func)))
                            .flatten()
                        }
                        _ => None,
                    };
                    let stored_value = wrapped.unwrap_or(operand);
                    let Some(location) = locations.iter().find(|location| {
                        location.value == stored_value
                            && cx
                                .body_view(body)
                                .shared()
                                .types
                                .size_of(cx.body_view(body).type_of(stored_value))
                                == location.size
                    }) else {
                        continue;
                    };
                    let load_ty = cx.body_view(body).type_of(stored_value);
                    let load = body.push_mnemonic_with_type(
                        Mnemonic::Load(Load {
                            space: location.space,
                            ptr: location.ptr.localize(ic.insn_id.func),
                            size: location.size,
                        }),
                        load_ty,
                    );
                    body.insert_insn_before(ic.block_id, ic.insn_id, load);
                    let replacement = if let ValueId::Instruction(wrapper) = operand
                        && wrapped.is_some()
                    {
                        let mut mnemonic = cx.body_view(body).insn_ref(wrapper).mnemonic().clone();
                        mnemonic.replace_value(
                            stored_value.strip_func(),
                            ValueId::Instruction(load).strip_func(),
                        );
                        let wrapper = body
                            .push_mnemonic_with_type(mnemonic, cx.body_view(body).type_of(operand));
                        body.insert_insn_before(ic.block_id, ic.insn_id, wrapper);
                        ValueId::Instruction(wrapper)
                    } else {
                        ValueId::Instruction(load)
                    };
                    rebuilt.replace_value(operand.strip_func(), replacement.strip_func());
                    replaced_operand = true;
                }
                if replaced_operand {
                    ed.replace_with_new_insn_typed(
                        body,
                        cx,
                        ic.block_id,
                        ic.insn_id,
                        rebuilt,
                        ic.type_id,
                    );
                    return Claim::Done;
                }

                let pair = locations.iter().enumerate().find_map(|(i, a)| {
                    locations[i..].iter().find_map(|b| {
                        (a.size == ic.size
                            && b.size == ic.size
                            && ic.numbering.is_sum(ic.id, a.value, b.value))
                        .then_some((*a, *b))
                    })
                });
                let Some((a, b)) = pair else {
                    return Claim::Pass;
                };

                // This is a read-only profitability estimate: replacing this root
                // always removes at least the root itself, while ordinary DCE later
                // discovers the complete cascading dead cone.  No speculative IR
                // mutation and rollback is required.
                if cx
                    .body_view(body)
                    .function_ref(ic.insn_id.func)
                    .users_of(ic.id)
                    .is_empty()
                {
                    return Claim::Pass;
                }

                let make_load =
                    |body: &mut FunctionBody<'str>, loc: crate::memory_state::StoredLocation| {
                        body.push_mnemonic_with_type(
                            Mnemonic::Load(Load {
                                space: loc.space,
                                ptr: loc.ptr.localize(ic.insn_id.func),
                                size: loc.size,
                            }),
                            ic.type_id,
                        )
                    };
                let lhs = make_load(body, a);
                body.insert_insn_before(ic.block_id, ic.insn_id, lhs);
                let rhs = make_load(body, b);
                body.insert_insn_before(ic.block_id, ic.insn_id, rhs);
                let sum = body.push_mnemonic_with_type(
                    Mnemonic::Binop(Binary {
                        op: Binop::Int(IntBinop::Add),
                        lhs: ValueId::Instruction(lhs).localize(ic.insn_id.func),
                        rhs: ValueId::Instruction(rhs).localize(ic.insn_id.func),
                    }),
                    ic.type_id,
                );
                body.insert_insn_before(ic.block_id, ic.insn_id, sum);
                ed.replace(body, cx, ic.insn_id, ValueId::Instruction(sum));
                Claim::Done
            }
            _ => Claim::Pass,
        }
    }

    fn after_block(
        &self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        state: &mut dyn Any,
        block: BlockId,
        aliases: Option<&AliasResult>,
        _numbering: &Numbering,
    ) {
        state
            .downcast_mut::<MemoryState>()
            .expect("memory state")
            .prune_clobbered_by_call(cx.body_view(body), block, aliases);
    }
}

/// Run selective variable-read rematerialization on one function.
pub fn variable_reads_function(
    ctx: &mut qcode::context::Context,
    function: qcode::value::FunctionId,
) -> bool {
    let aliases = AliasResult::simple_for_function(ctx, function);
    crate::with_body_mut(ctx, function, |body, cx| {
        run_dominator_walk(
            body,
            cx,
            function,
            &[Box::new(RematerializeVariables)],
            Some(&aliases),
        )
    })
}

#[derive(Default)]
pub struct VariableReads;

impl FunctionPass for VariableReads {
    const NAME: &'static str = "variable_reads";

    fn description(&self) -> &'static str {
        "Rematerialize equivalent expressions from current stored variables"
    }

    fn run<'str>(
        &self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        _next_minted: &mut u32,
    ) -> Result<Outcome<'str>, String> {
        let aliases = cx
            .env()
            .alias_base(cx.shr())
            .for_function(cx.body_view(body), body.id())
            .with_frame_freshness(cx.body_view(body), body.id(), cx.env().sp_varnode);
        let changed = run_dominator_walk(
            body,
            cx,
            body.id(),
            &[Box::new(RematerializeVariables)],
            Some(&aliases),
        );
        Ok(Outcome::changed(changed))
    }

    fn run_with_analyses<'str>(
        &self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        _next_minted: &mut u32,
        analyses: &mut LocalAnalysisManager,
    ) -> Result<Outcome<'str>, String> {
        let aliases = analyses.get::<AliasAnalysis>(body, cx);
        let changed = run_dominator_walk(
            body,
            cx,
            body.id(),
            &[Box::new(RematerializeVariables)],
            Some(aliases),
        );
        Ok(Outcome::changed(changed))
    }
}

crate::register_function_pass!(VariableReads);

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::value::{FunctionBody, insn::Mnemonic};
    use wazabin_qcode_macro::qcode;

    #[test]
    fn rematerializes_sum_from_two_current_variables() {
        let mut ctx = qcode::context::Context::new();
        qcode!(
            ctx,
            "
                varnode i32 A;
                varnode i32 B;
                fn f:
                    <entry @x:i32 @y:i32>
                        store(A:4, &A <- @x);
                        store(B:4, &B <- @y);
                        %sum = @y + @x;
                        return %sum;
            "
        );

        assert!(variable_reads_function(&mut ctx, f));
        assert!(!ctx.contains_instruction(sum));
        let loads = FunctionBody::from_id(&ctx, f)
            .blocks()
            .flat_map(|b| b.iter())
            .filter(|i| matches!(i.mnemonic(), Mnemonic::Load(_)))
            .count();
        assert_eq!(loads, 2);
    }
}
