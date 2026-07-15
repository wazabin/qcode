//! Structural validation for function-owned arenas.
//!
//! This scanner lives in core so it can inspect malformed private arena state
//! without exposing storage internals as public API. It never indexes an ID
//! before proving that the corresponding payload is live.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    context::Context,
    space::{LocalMemorySpaceId, MemorySpaceId},
    value::{BlockId, BlockParamId, InstructionId, LocalValueId, insn::Mnemonic},
};

fn missing_type_temp_space(
    ctx: &Context<'_>,
    function: crate::value::FunctionId,
    type_id: crate::types::TypeId,
) -> Option<crate::value::TempSpaceId> {
    let MemorySpaceId::Temp(space) = ctx.shared.types.space_of(type_id)? else {
        return None;
    };
    (space.func != function || usize::from(space.local) >= ctx.bodies[function].temp_spaces.len())
        .then_some(space)
}

/// Validate the ownership and cross-reference invariants of every function
/// body's block, instruction, parameter, CFG-edge, and temporary arenas.
///
/// Diagnostics from this function describe potentially unsafe structural
/// corruption. Callers should not run higher-level traversals until it returns
/// an empty vector.
pub fn verify_body_arena_integrity(ctx: &Context<'_>) -> Vec<String> {
    let mut out = Vec::new();

    for body in ctx.bodies.iter() {
        let fid = body.id;
        let live_blocks: FxHashSet<_> = body.blocks.iter().map(|block| block.id).collect();
        let live_insns: FxHashSet<_> = body.insns.iter().map(|insn| insn.id).collect();
        let live_params: FxHashSet<_> = body.params.iter().map(|param| param.id).collect();
        let live_edges: FxHashSet<_> = body.edges.iter().map(|edge| edge.id).collect();

        for temp in body.temps.iter() {
            if usize::from(temp.space) >= body.temp_spaces.len() {
                out.push(format!(
                    "function {fid:?}: temporary {:?} references missing temporary space {:?}",
                    crate::value::TempId::new(fid, temp.id),
                    crate::value::TempSpaceId::new(fid, temp.space)
                ));
            }
        }

        let mut roster_count = FxHashMap::default();
        for &local in &body.roster {
            *roster_count.entry(local).or_insert(0usize) += 1;
            if !live_blocks.contains(&local) {
                out.push(format!(
                    "function {fid:?}: roster references removed block {:?}",
                    BlockId::new(fid, local)
                ));
            }
        }
        for (&local, &count) in &roster_count {
            if count != 1 {
                out.push(format!(
                    "function {fid:?}: block {:?} appears {count} times in the roster",
                    BlockId::new(fid, local)
                ));
            }
        }
        for &local in &live_blocks {
            let count = roster_count.get(&local).copied().unwrap_or(0);
            if count != 1 {
                out.push(format!(
                    "function {fid:?}: live block {:?} has roster count {count}",
                    BlockId::new(fid, local)
                ));
            }
        }

        if let Some(root) = body.root_id() {
            let root_id = BlockId::new(fid, root);
            if !live_blocks.contains(&root) {
                out.push(format!(
                    "function {fid:?}: root {root_id:?} has no live block payload"
                ));
            } else if roster_count.get(&root).copied().unwrap_or(0) != 1 {
                out.push(format!(
                    "function {fid:?}: root {root_id:?} is not rostered exactly once"
                ));
            }
        }

        let mut insn_membership: FxHashMap<_, Vec<_>> = FxHashMap::default();
        let mut param_membership: FxHashMap<_, Vec<(crate::value::LocalBlockId, usize)>> =
            FxHashMap::default();

        for block_entry in body.blocks.iter() {
            let local = block_entry.id;
            let block_id = BlockId::new(fid, local);
            let block = &*block_entry;

            if block.parent != Some(fid) {
                out.push(format!(
                    "function {fid:?}: live block {block_id:?} has parent {:?}",
                    block.parent
                ));
            }

            for &insn_local in &block.instructions {
                insn_membership.entry(insn_local).or_default().push(local);
                let insn_id = InstructionId::new(fid, insn_local);
                if !live_insns.contains(&insn_local) {
                    out.push(format!(
                        "block {block_id:?} references removed instruction {insn_id:?}"
                    ));
                    continue;
                }
                if body.insns[insn_local].parent != Some(local) {
                    out.push(format!(
                        "block {block_id:?} contains {insn_id:?}, whose parent is {:?}",
                        body.insns[insn_local].parent
                    ));
                }
            }

            for (index, &param_local) in block.params.iter().enumerate() {
                param_membership
                    .entry(param_local)
                    .or_default()
                    .push((local, index));
                let param_id = BlockParamId::new(fid, param_local);
                if !live_params.contains(&param_local) {
                    out.push(format!(
                        "block {block_id:?} references removed parameter {param_id:?}"
                    ));
                    continue;
                }
                let param = &body.params[param_local];
                if param.parent != Some(local) {
                    out.push(format!(
                        "block {block_id:?} contains {param_id:?}, whose parent is {:?}",
                        param.parent
                    ));
                }
                if param.index != index {
                    out.push(format!(
                        "block {block_id:?} contains {param_id:?} at index {index}, but payload index is {}",
                        param.index
                    ));
                }
            }

            for &edge_id in &block.edges {
                if !live_edges.contains(&edge_id) {
                    out.push(format!(
                        "block {block_id:?} references removed CFG edge {edge_id:?}"
                    ));
                    continue;
                }
                let edge = &body.edges[edge_id];
                if edge.from != block_id && edge.to != block_id {
                    out.push(format!(
                        "block {block_id:?} lists non-incident CFG edge {edge_id:?} ({:?} -> {:?})",
                        edge.from, edge.to
                    ));
                }
            }
        }

        for insn_entry in body.insns.iter() {
            let local = insn_entry.id;
            let insn_id = InstructionId::new(fid, local);
            let memberships = insn_membership
                .get(&local)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if let Some(parent) = insn_entry.parent {
                if !live_blocks.contains(&parent) {
                    out.push(format!(
                        "instruction {insn_id:?} has removed parent {:?}",
                        BlockId::new(fid, parent)
                    ));
                }
                if memberships != [parent] {
                    out.push(format!(
                        "instruction {insn_id:?} has parent {:?} but block memberships {memberships:?}",
                        BlockId::new(fid, parent)
                    ));
                }
            } else if !memberships.is_empty() {
                out.push(format!(
                    "detached instruction {insn_id:?} still appears in blocks {memberships:?}"
                ));
            }

            if let Some(space) = missing_type_temp_space(ctx, fid, insn_entry.type_id) {
                out.push(format!(
                    "instruction {insn_id:?} result type references missing temporary space {space:?}"
                ));
            }

            for arg in insn_entry.mnemonic().args() {
                let missing = match arg {
                    LocalValueId::Instruction(id) => !live_insns.contains(&id),
                    LocalValueId::BlockParam(id) => !live_params.contains(&id),
                    LocalValueId::BasicBlock(id) => !live_blocks.contains(&id),
                    LocalValueId::Temp(id) => usize::from(id) >= body.temps.len(),
                    _ => false,
                };
                if missing {
                    out.push(format!(
                        "instruction {insn_id:?} references removed local value {:?}",
                        arg.qualify(fid)
                    ));
                }
            }

            let mnemonic_space = match insn_entry.mnemonic() {
                Mnemonic::Load(load) => Some(load.space),
                Mnemonic::Store(store) => Some(store.space),
                _ => None,
            };
            if let Some(LocalMemorySpaceId::Temp(space)) = mnemonic_space
                && usize::from(space) >= body.temp_spaces.len()
            {
                out.push(format!(
                    "instruction {insn_id:?} references missing temporary space {:?}",
                    crate::value::TempSpaceId::new(fid, space)
                ));
            }

            let mut check_target = |target| {
                if !live_blocks.contains(&target) {
                    out.push(format!(
                        "instruction {insn_id:?} targets removed block {:?}",
                        BlockId::new(fid, target)
                    ));
                }
            };
            match insn_entry.mnemonic() {
                Mnemonic::Branch(branch) => check_target(branch.target),
                Mnemonic::CBranch(branch) => {
                    check_target(branch.success_block);
                    check_target(branch.failure_block);
                }
                _ => {}
            }
        }

        for param_entry in body.params.iter() {
            let local = param_entry.id;
            let param_id = BlockParamId::new(fid, local);
            let memberships = param_membership
                .get(&local)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            match param_entry.parent {
                Some(parent) if !live_blocks.contains(&parent) => out.push(format!(
                    "parameter {param_id:?} has removed parent {:?}",
                    BlockId::new(fid, parent)
                )),
                None => out.push(format!("live parameter {param_id:?} is detached")),
                Some(_) => {}
            }
            if memberships.len() != 1 {
                out.push(format!(
                    "live parameter {param_id:?} has {} block memberships",
                    memberships.len()
                ));
            }
            if let Some(space) = missing_type_temp_space(ctx, fid, param_entry.type_id) {
                out.push(format!(
                    "parameter {param_id:?} type references missing temporary space {space:?}"
                ));
            }
            if let Some(LocalValueId::Temp(temp)) = param_entry.origin
                && usize::from(temp) >= body.temps.len()
            {
                out.push(format!(
                    "parameter {param_id:?} origin references missing temporary {:?}",
                    crate::value::TempId::new(fid, temp)
                ));
            }
        }

        for edge_entry in body.edges.iter() {
            let edge_id = edge_entry.id;
            let edge = &*edge_entry;
            if edge.from.func != fid || edge.to.func != fid {
                out.push(format!(
                    "function {fid:?}: edge {edge_id:?} crosses function arenas ({:?} -> {:?})",
                    edge.from, edge.to
                ));
                continue;
            }
            let from_live = live_blocks.contains(&edge.from.local);
            let to_live = live_blocks.contains(&edge.to.local);
            if !from_live || !to_live {
                out.push(format!(
                    "function {fid:?}: edge {edge_id:?} has removed endpoint ({:?} -> {:?})",
                    edge.from, edge.to
                ));
                continue;
            }
            if !body.blocks[edge.from.local].edges.contains(&edge_id) {
                out.push(format!(
                    "edge {edge_id:?} is missing from source block {:?}",
                    edge.from
                ));
            }
            if !body.blocks[edge.to.local].edges.contains(&edge_id) {
                out.push(format!(
                    "edge {edge_id:?} is missing from target block {:?}",
                    edge.to
                ));
            }
        }

        for (name, value) in body.names.entries() {
            let live = match value {
                LocalValueId::BasicBlock(id) => live_blocks.contains(&id),
                LocalValueId::Instruction(id) => live_insns.contains(&id),
                LocalValueId::BlockParam(id) => live_params.contains(&id),
                LocalValueId::Temp(id) => usize::from(id) < body.temps.len(),
                _ => false,
            };
            if !live {
                out.push(format!(
                    "function {fid:?}: local name {name:?} references absent or non-local value {value:?}"
                ));
            }
        }

        for (&value, users) in &body.users {
            let key_live = match value {
                LocalValueId::Instruction(id) => live_insns.contains(&id),
                LocalValueId::BlockParam(id) => live_params.contains(&id),
                LocalValueId::BasicBlock(id) => live_blocks.contains(&id),
                LocalValueId::Temp(id) => usize::from(id) < body.temps.len(),
                _ => true,
            };
            if !key_live {
                out.push(format!(
                    "function {fid:?}: users map contains removed key {:?}",
                    value.qualify(fid)
                ));
            }
            for &user in users {
                if !live_insns.contains(&user) {
                    out.push(format!(
                        "function {fid:?}: users map for {:?} references removed instruction {:?}",
                        value.qualify(fid),
                        InstructionId::new(fid, user)
                    ));
                }
            }
        }

        let stats = body.arena_stats();
        let max_issued = u32::MAX as usize + 1;
        for (kind, issued) in [
            ("instruction", stats.instructions.issued),
            ("block", stats.blocks.issued),
            ("parameter", stats.params.issued),
            ("edge", stats.edges.issued),
        ] {
            if issued > max_issued {
                out.push(format!(
                    "function {fid:?}: {kind} arena issued cursor {issued} exceeds u32 ID space"
                ));
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use qcode_macro::qcode;

    use super::*;
    use crate::value::{BasicBlock, FunctionBody, LocalTempSpaceId, Temp, ValueId};

    fn fixture() -> Context<'static> {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
            <entry @x:i64>
                %y = i64 @x + i64 1;
                goto <exit>;
            <exit>
                return at i64 %y;
            "
        );
        ctx
    }

    fn assert_has(ctx: &Context<'_>, needle: &str) {
        let diagnostics = verify_body_arena_integrity(ctx);
        assert!(
            diagnostics.iter().any(|d| d.contains(needle)),
            "expected diagnostic containing {needle:?}, got {diagnostics:#?}"
        );
    }

    #[test]
    fn valid_body_is_clean() {
        let ctx = fixture();
        assert_eq!(verify_body_arena_integrity(&ctx), Vec::<String>::new());
    }

    #[test]
    fn reports_removed_root_without_indexing_it() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let dead = BasicBlock::make(&mut ctx, f).id;
        ctx.delete_block(dead);
        ctx.function_mut(f).set_root_id(Some(dead.local));

        assert_has(&ctx, "root");
    }

    #[test]
    fn reports_duplicate_roster_membership() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        ctx.bodies[f].roster.push(entry.local);

        assert_has(&ctx, "appears 2 times in the roster");
    }

    #[test]
    fn reports_unrostered_block_and_wrong_owner() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let block = BasicBlock::make(&mut ctx, f).id;
        ctx.bodies[f].roster.retain(|&local| local != block.local);
        ctx.block_mut(block).parent = None;

        assert_has(&ctx, "roster count 0");
        assert_has(&ctx, "has parent None");
    }

    #[test]
    fn reports_instruction_membership_and_parent_disagreement() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let y = ctx
            .block(entry)
            .instructions
            .iter()
            .copied()
            .find(|&id| !ctx.bodies[f].insns[id].mnemonic().is_terminator())
            .expect("value instruction");
        ctx.block_mut(entry).instructions.push(y);

        assert_has(&ctx, "block memberships");
    }

    #[test]
    fn permits_live_detached_instruction() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let detached = ctx
            .block(entry)
            .instructions
            .iter()
            .copied()
            .find(|&id| !ctx.bodies[f].insns[id].mnemonic().is_terminator())
            .expect("value instruction");
        ctx.block_mut(entry)
            .instructions
            .retain(|&id| id != detached);
        ctx.bodies[f].insns[detached].parent = None;

        assert_eq!(verify_body_arena_integrity(&ctx), Vec::<String>::new());
    }

    #[test]
    fn reports_stale_instruction_and_parameter_membership() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let insn = ctx
            .block(entry)
            .instructions
            .iter()
            .copied()
            .find(|&id| !ctx.bodies[f].insns[id].mnemonic().is_terminator())
            .expect("value instruction");
        let param = ctx.block(entry).params[0];
        ctx.bodies[f].insns.remove(insn);
        ctx.bodies[f].params.remove(param);

        assert_has(&ctx, "references removed instruction");
        assert_has(&ctx, "references removed parameter");
        assert_has(&ctx, "references removed local value");
        assert_has(&ctx, "users map for");
    }

    #[test]
    fn reports_parameter_index_and_parent_disagreement() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let param = BlockParamId::new(f, ctx.block(entry).params[0]);
        ctx.block_param_mut(param).index = 7;
        ctx.block_param_mut(param).parent = None;

        assert_has(&ctx, "payload index is 7");
        assert_has(&ctx, "live parameter");
    }

    #[test]
    fn reports_missing_edge_adjacency() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let edge = *ctx.block(entry).edges.iter().next().expect("edge");
        let target = ctx.edge(f, edge).to;
        ctx.block_mut(target).edges.remove(&edge);

        assert_has(&ctx, "missing from target block");
    }

    #[test]
    fn reports_stale_and_non_incident_adjacency_without_indexing() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let edge = *ctx.block(entry).edges.iter().next().expect("edge");
        let unrelated = BasicBlock::make(&mut ctx, f).id;
        ctx.block_mut(unrelated).edges.insert(edge);

        assert_has(&ctx, "lists non-incident CFG edge");

        ctx.bodies[f].edges.remove(edge);
        assert_has(&ctx, "references removed CFG edge");
    }

    #[test]
    fn reports_cross_function_edge_endpoint() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let edge = *ctx.block(entry).edges.iter().next().expect("edge");
        let g = FunctionBody::make(&mut ctx, Cow::Borrowed("g"))
            .expect("function")
            .id;
        let foreign = BasicBlock::make(&mut ctx, g).id;
        ctx.bodies[f].edges[edge].to = foreign;

        assert_has(&ctx, "crosses function arenas");
    }

    #[test]
    fn reports_branch_targeting_removed_block() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let edge = *ctx.block(entry).edges.iter().next().expect("edge");
        let target = ctx.edge(f, edge).to;
        ctx.delete_block(target);

        assert_has(&ctx, "targets removed block");
    }

    #[test]
    fn reports_stale_local_name_and_users_entries() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let dead_block = BasicBlock::make(&mut ctx, f).id;
        ctx.delete_block(dead_block);
        ctx.bodies[f]
            .names
            .register(
                Cow::Borrowed("stale"),
                ValueId::BasicBlock(dead_block).localize(f),
                None,
            )
            .expect("register corruption fixture");

        let live_block = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let dead_param = BasicBlock::from_id_mut(&mut ctx, live_block)
            .push_param(8)
            .id;
        ctx.block_mut(live_block).params.pop();
        ctx.remove_block_param(dead_param);
        ctx.bodies[f]
            .users
            .insert(ValueId::BlockParam(dead_param).strip_func(), Vec::new());

        assert_has(&ctx, "local name \"stale\"");
        assert_has(&ctx, "users map contains removed key");
    }

    #[test]
    fn reports_temporary_with_missing_local_space() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        ctx.bodies[f]
            .temps
            .push(Temp::new(0, 8, LocalTempSpaceId::from(7)));

        assert_has(&ctx, "references missing temporary space");
    }

    #[test]
    fn reports_dangling_temporary_operands_spaces_origins_and_types() {
        use crate::value::insn::Load;

        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let insn = ctx
            .block(entry)
            .instructions
            .iter()
            .copied()
            .find(|&id| !ctx.bodies[f].insns[id].mnemonic().is_terminator())
            .expect("value instruction");
        let missing_temp = crate::value::LocalTempId::from(0);
        let missing_space = LocalTempSpaceId::from(0);
        *ctx.bodies[f].insns[insn].mnemonic_mut() = Mnemonic::Load(Load {
            space: LocalMemorySpaceId::Temp(missing_space),
            ptr: LocalValueId::Temp(missing_temp),
            size: 8,
        });
        ctx.bodies[f].insns[insn].type_id = ctx.shared.types.get_or_make_space_address(
            8,
            MemorySpaceId::Temp(crate::value::TempSpaceId::new(f, missing_space)),
        );
        let param = ctx.block(entry).params[0];
        ctx.bodies[f].params[param].origin = Some(LocalValueId::Temp(missing_temp));

        assert_has(&ctx, "references removed local value");
        assert_has(&ctx, "references missing temporary space");
        assert_has(&ctx, "result type references missing temporary space");
        assert_has(&ctx, "origin references missing temporary");
    }
}
