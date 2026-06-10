use std::collections::{HashMap, HashSet};

use jstd::graph::analysis::DominatorTree;

use crate::AliasResult;
use qcode::{
    context::Context,
    space::{Space, SpaceType},
    value::{
        BasicBlock, Function, Value, ValueId, ValueRef, VarnodeId,
        block::BlockId,
        insn::{InstructionId, InstructionRef, Load, Mnemonic, Range},
    },
};

use super::integer::{algebraic_identity, constant_folding, normalize, simplify_flag_idiom};

// ---------------------------------------------------------------------------
// GVN pass
// ---------------------------------------------------------------------------

/// Process all instructions in `block_id` with an inherited value table.
///
/// Returns the updated table (inherited entries plus new entries from this block)
/// for descendants in the dominator tree to inherit.
pub(super) fn gvn_block_inner(
    ctx: &mut Context,
    block_id: BlockId,
    inherited: &HashMap<Mnemonic, ValueId>,
    aliases: Option<&AliasResult>,
) -> HashMap<Mnemonic, ValueId> {
    let mut table = inherited.clone();
    // Maps a narrow-load mnemonic to (wide_src, byte_offset_within_wide, sub_size),
    // populated when a wide store covers sub-register locations.
    let mut range_table: HashMap<Mnemonic, (ValueId, usize, usize)> = HashMap::new();
    let mut redundant: HashSet<InstructionId> = HashSet::new();

    let insns = BasicBlock::from_id(ctx, block_id)
        .instruction_ids()
        .to_vec();

    for insn_id in insns {
        let insn = ctx.get_insn(insn_id);
        let id = insn.id();
        let insn_size = insn.size();
        let mut mnemonic = insn.mnemonic().clone();

        match mnemonic {
            Mnemonic::Store(store) => {
                let store_ptr = store.ptr;
                table.retain(|k, _| {
                    if let Mnemonic::Load(load) = k {
                        aliases.is_some_and(|aliases| !aliases.may_alias(ctx, store_ptr, load.ptr))
                    } else {
                        true
                    }
                });
                // Only forward when the stored value is exactly as wide as the
                // location. Some lifts (e.g. `MOV ESI, imm32`, which zero-extends
                // into the 8-byte RSI) emit a wide store of a narrower literal;
                // forwarding that value straight into a full-width load would feed
                // a mis-sized constant into later folding.
                if ValueRef::new(store.src, ctx).size() == store.size {
                    table.insert(Mnemonic::Load(store.get_matching_load()), store.src);
                }

                // Invalidate range_table entries for locations this store may overwrite.
                range_table.retain(|k, _| {
                    if let Mnemonic::Load(sub_load) = k {
                        aliases
                            .is_none_or(|aliases| !aliases.may_alias(ctx, store_ptr, sub_load.ptr))
                    } else {
                        true
                    }
                });

                // Forward sub-register loads from this wide store.
                if let Some(aliases) = aliases {
                    for (sub_ptr, byte_off, sub_size) in aliases.sub_intervals_of(store.ptr) {
                        let sub_load = Load {
                            space: store.space,
                            ptr: sub_ptr,
                            size: sub_size,
                        };
                        range_table
                            .insert(Mnemonic::Load(sub_load), (store.src, byte_off, sub_size));
                    }
                }
            }

            _ if mnemonic.is_terminator() || insn_size == 0 => {}

            _ => {
                if let Some(cst) = constant_folding(ctx, &mnemonic, insn_size) {
                    ctx.replace_all_uses_with(id, cst);
                    redundant.insert(insn_id);
                    continue;
                }

                if let Some(simplified) = algebraic_identity(ctx, &mnemonic, insn_size) {
                    ctx.replace_all_uses_with(id, simplified);
                    redundant.insert(insn_id);
                    continue;
                }

                // Collapse the signed-compare flag idiom into a single `s<`,
                // materializing the replacement before this instruction and
                // forwarding its uses; the dead flag math falls to DCE.
                if let Some(new_mnemonic) = simplify_flag_idiom(ctx, &mnemonic) {
                    let new_id = InstructionRef::from_mnemonic(ctx, new_mnemonic, insn_size).id;
                    BasicBlock::from_id_mut(ctx, block_id).insert_insn_before(insn_id, new_id);
                    ctx.replace_all_uses_with(id, new_id);
                    redundant.insert(insn_id);
                    continue;
                }

                normalize(&mut mnemonic);

                match table.get(&mnemonic) {
                    Some(&leader) => {
                        ctx.replace_all_uses_with(id, leader);
                        redundant.insert(insn_id);
                    }
                    None => {
                        // For loads not in the main table, check whether a wider store
                        // already covers this sub-register location.
                        if let Mnemonic::Load(_) = &mnemonic
                            && let Some(&(wide_src, byte_off, sub_size)) =
                                range_table.get(&mnemonic)
                        {
                            let range_id = InstructionRef::from_mnemonic(
                                ctx,
                                Mnemonic::Range(Range {
                                    src: wide_src,
                                    start: byte_off,
                                    size: sub_size,
                                }),
                                sub_size,
                            )
                            .id;
                            BasicBlock::from_id_mut(ctx, block_id)
                                .insert_insn_before(insn_id, range_id);
                            ctx.replace_all_uses_with(id, range_id);
                            table.insert(mnemonic, range_id.into());
                            redundant.insert(insn_id);
                            continue;
                        }
                        table.insert(mnemonic, id);
                    }
                }
            }
        }
    }

    BasicBlock::from_id_mut(ctx, block_id).retain_insns(|insn| !redundant.contains(insn));
    table
}

/// The registers a call may clobber, used to invalidate forwarded `Load` leaders.
enum CallClobbers {
    /// Unknown target (`CallInd`) or a target with no recorded clobber set:
    /// conservatively every register.
    AllRegisters,
    /// A direct call's recorded per-callee clobber set.
    Regs(Vec<VarnodeId>),
}

/// Drop register `Load` leaders from `table` that the call terminating `block_id`
/// (if any) may clobber, so they are not forwarded into the call's continuation.
/// Non-call blocks pass `table` through untouched.
pub(super) fn prune_clobbered_by_call<'a>(
    ctx: &Context,
    block_id: BlockId,
    table: &'a HashMap<Mnemonic, ValueId>,
    aliases: Option<&AliasResult>,
) -> std::borrow::Cow<'a, HashMap<Mnemonic, ValueId>> {
    let term = BasicBlock::from_id(ctx, block_id)
        .iter()
        .last()
        .map(|i| i.mnemonic().clone());
    let clobbers = match term {
        Some(Mnemonic::CallInd(_)) => CallClobbers::AllRegisters,
        Some(Mnemonic::Call(call)) => match Function::from_id(ctx, call.target).clobbered_regs() {
            Some(regs) => CallClobbers::Regs(regs.to_vec()),
            None => CallClobbers::AllRegisters,
        },
        _ => return std::borrow::Cow::Borrowed(table),
    };

    let is_reg = |space| matches!(Space::from_id(ctx, space).ty, SpaceType::Register);
    let has_clobbered_reg_load = table
        .keys()
        .any(|m| matches!(m, Mnemonic::Load(load) if is_reg(load.space)));
    if !has_clobbered_reg_load {
        return std::borrow::Cow::Borrowed(table);
    }

    let mut pruned = table.clone();
    pruned.retain(|m, _| {
        let Mnemonic::Load(load) = m else { return true };
        if !is_reg(load.space) {
            return true;
        }
        match &clobbers {
            CallClobbers::AllRegisters => false,
            CallClobbers::Regs(regs) => !regs.iter().any(|&r| match aliases {
                Some(a) => a.may_alias(ctx, load.ptr, ValueId::Varnode(r)),
                None => true,
            }),
        }
    });
    std::borrow::Cow::Owned(pruned)
}

pub(super) fn prune_loop_carried_loads<'a>(
    ctx: &Context,
    block_id: BlockId,
    inherited: &'a HashMap<Mnemonic, ValueId>,
    tree: &DominatorTree<BlockId>,
    aliases: Option<&AliasResult>,
) -> std::borrow::Cow<'a, HashMap<Mnemonic, ValueId>> {
    let is_loop_header = BasicBlock::from_id(ctx, block_id)
        .predecessors()
        .any(|(_, pred)| pred != block_id && tree.dominates(block_id, pred));
    if !is_loop_header || !inherited.keys().any(|m| matches!(m, Mnemonic::Load(_))) {
        return std::borrow::Cow::Borrowed(inherited);
    }

    let mut pruned = inherited.clone();
    pruned.retain(|mnemonic, _| {
        let Mnemonic::Load(load) = mnemonic else {
            return true;
        };
        let Some(aliases) = aliases else {
            return false;
        };

        !ctx.block_ids()
            .into_iter()
            .filter(|&candidate| candidate != block_id && tree.dominates(block_id, candidate))
            .flat_map(|candidate| {
                BasicBlock::from_id(ctx, candidate)
                    .iter()
                    .map(|insn| insn.mnemonic().clone())
                    .collect::<Vec<_>>()
            })
            .any(|mnemonic| {
                matches!(
                    mnemonic,
                    Mnemonic::Store(store) if aliases.may_alias(ctx, store.ptr, load.ptr)
                )
            })
    });
    std::borrow::Cow::Owned(pruned)
}
