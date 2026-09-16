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
    verify_body_arena_integrity_scoped(ctx, None)
}

/// [`verify_body_arena_integrity`], restricted to the functions in `scope`
/// (`None` means every function). Arena invariants are strictly per-body, so a
/// caller that knows which functions changed can skip the rest.
pub fn verify_body_arena_integrity_scoped(
    ctx: &Context<'_>,
    scope: Option<&FxHashSet<crate::value::FunctionId>>,
) -> Vec<String> {
    let mut out = Vec::new();

    for body in ctx.bodies.iter() {
        if scope.is_some_and(|set| !set.contains(&body.id)) {
            continue;
        }
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

            // Block ownership is derived from the storing arena (`fid`); there is
            // no per-block parent field left to disagree with it.

            // The list is walked along its links, checking each against the
            // one before it, its length against the block's count, and its
            // end against the block's `last`.
            let list = block.instructions;
            let mut walked = 0;
            let mut prev = None;
            let mut at = list.first;
            while let Some(insn_local) = at {
                walked += 1;
                if walked > list.len {
                    out.push(format!(
                        "block {block_id:?} links more instructions than its count of {}",
                        list.len
                    ));
                    break;
                }
                insn_membership.entry(insn_local).or_default().push(local);
                let insn_id = InstructionId::new(fid, insn_local);
                if !live_insns.contains(&insn_local) {
                    out.push(format!(
                        "block {block_id:?} references removed instruction {insn_id:?}"
                    ));
                    break;
                }
                let insn = &body.insns[insn_local];
                if insn.parent != Some(local) {
                    out.push(format!(
                        "block {block_id:?} contains {insn_id:?}, whose parent is {:?}",
                        insn.parent
                    ));
                }
                if insn.prev != prev {
                    out.push(format!(
                        "block {block_id:?}: {insn_id:?} links back to {:?}, not to {prev:?}",
                        insn.prev
                    ));
                }
                prev = Some(insn_local);
                at = insn.next;
            }
            if walked != list.len {
                out.push(format!(
                    "block {block_id:?} links {walked} instructions but counts {}",
                    list.len
                ));
            }
            if list.last != prev {
                out.push(format!(
                    "block {block_id:?} ends at {prev:?} but its last is {:?}",
                    list.last
                ));
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
                if edge.from != block_id.local && edge.to != block_id.local {
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
                // A pure instruction that names its own result as an operand is a
                // self-referential (unsatisfiable) value: an SSA-invariant
                // violation that makes recursive value-walkers loop. Catch the
                // direct case here — it is the concrete shape a bad forward mints
                // (`%x = %y + %x`).
                if arg == LocalValueId::Instruction(local) {
                    out.push(format!(
                        "instruction {insn_id:?} references itself as an operand"
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
            // Edge endpoints are bare `LocalBlockId`s stored in this body's arena
            // (stage 1); there is no cross-arena state left to probe.
            let from_live = live_blocks.contains(&edge.from);
            let to_live = live_blocks.contains(&edge.to);
            if !from_live || !to_live {
                out.push(format!(
                    "function {fid:?}: edge {edge_id:?} has removed endpoint ({:?} -> {:?})",
                    edge.from, edge.to
                ));
                continue;
            }
            if !body.blocks[edge.from].edges.contains(&edge_id) {
                out.push(format!(
                    "edge {edge_id:?} is missing from source block {:?}",
                    edge.from
                ));
            }
            if !body.blocks[edge.to].edges.contains(&edge_id) {
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

        verify_use_edges(
            &body,
            fid,
            &live_insns,
            &live_params,
            &live_blocks,
            &mut out,
        );

        let stats = body.arena_stats();
        // Widened so the bound exists on 32-bit targets (wasm) too.
        let max_issued = u64::from(u32::MAX) + 1;
        for (kind, issued) in [
            ("instruction", stats.instructions.issued),
            ("block", stats.blocks.issued),
            ("parameter", stats.params.issued),
            ("edge", stats.edges.issued),
        ] {
            if issued as u64 > max_issued {
                out.push(format!(
                    "function {fid:?}: {kind} arena issued cursor {issued} exceeds u32 ID space"
                ));
            }
        }
    }

    out
}

/// Checks a body's use edges against its operands: every operand occurrence
/// whose value has storage here has exactly one edge, every edge names the
/// operand at its `(user, operand_index)`, every edge is reachable exactly
/// once from its value's list head, and no edge points at removed storage.
/// Together these say the edges are what [`FunctionBody::rebuild_uses`]
/// would derive.
fn verify_use_edges(
    body: &crate::value::FunctionBody<'_>,
    fid: crate::value::FunctionId,
    live_insns: &FxHashSet<crate::value::LocalInsnId>,
    live_params: &FxHashSet<crate::value::LocalParamId>,
    live_blocks: &FxHashSet<crate::value::LocalBlockId>,
    out: &mut Vec<String>,
) {
    let has_storage = |value: LocalValueId| match value {
        LocalValueId::Instruction(id) => live_insns.contains(&id),
        LocalValueId::BlockParam(id) => live_params.contains(&id),
        LocalValueId::BasicBlock(id) => live_blocks.contains(&id),
        LocalValueId::Temp(id) => usize::from(id) < body.temps.len(),
        _ => true,
    };

    // Every edge names a live user and its operand, and a value with storage.
    let mut edges_per_operand: FxHashMap<(crate::value::LocalInsnId, usize), usize> =
        FxHashMap::default();
    for edge in body.use_edges() {
        let edge_id = edge.id;
        let user = InstructionId::new(fid, edge.user);
        let index = usize::from(edge.operand_index);
        if !live_insns.contains(&edge.user) {
            out.push(format!(
                "function {fid:?}: use {edge_id:?} of {:?} has removed user {user:?}",
                edge.value.qualify(fid)
            ));
            continue;
        }
        if !has_storage(edge.value) {
            out.push(format!(
                "function {fid:?}: use {edge_id:?} by {user:?} names removed value {:?}",
                edge.value.qualify(fid)
            ));
        }
        match body.insns[edge.user].mnemonic().operand(index) {
            Some(operand) if operand == edge.value => {
                *edges_per_operand.entry((edge.user, index)).or_default() += 1;
            }
            Some(operand) => out.push(format!(
                "function {fid:?}: use {edge_id:?} says operand {index} of {user:?} is {:?} but it is {:?}",
                edge.value.qualify(fid),
                operand.qualify(fid)
            )),
            None => out.push(format!(
                "function {fid:?}: use {edge_id:?} names operand {index} of {user:?}, which has no such operand"
            )),
        }
    }

    // Every operand occurrence with storage has exactly one edge naming it.
    for insn in body.insns.iter() {
        let user = InstructionId::new(fid, insn.id);
        let mut index = 0;
        insn.mnemonic().for_each_operand(|value| {
            let edges = edges_per_operand
                .get(&(insn.id, index))
                .copied()
                .unwrap_or(0);
            if has_storage(value) && edges != 1 {
                out.push(format!(
                    "function {fid:?}: operand {index} of {user:?} ({:?}) has {edges} use edges",
                    value.qualify(fid)
                ));
            }
            index += 1;
        });
    }

    // Every edge is reachable exactly once, from its own value's head.
    let mut reached: FxHashSet<crate::value::uses::UseId> = FxHashSet::default();
    let mut walk = |value: LocalValueId, out: &mut Vec<String>| {
        let mut at = body.first_use_of(value);
        let mut steps = 0;
        while let Some(edge_id) = at {
            if !body.uses.contains(edge_id) {
                out.push(format!(
                    "function {fid:?}: use list of {:?} reaches freed use {edge_id:?}",
                    value.qualify(fid)
                ));
                break;
            }
            let edge = &body.uses[edge_id];
            if edge.value != value {
                out.push(format!(
                    "function {fid:?}: use list of {:?} holds use {edge_id:?} of {:?}",
                    value.qualify(fid),
                    edge.value.qualify(fid)
                ));
            }
            if !reached.insert(edge_id) {
                out.push(format!(
                    "function {fid:?}: use {edge_id:?} is reachable more than once (from {:?})",
                    value.qualify(fid)
                ));
                break;
            }
            steps += 1;
            if steps > body.uses.len() {
                out.push(format!(
                    "function {fid:?}: use list of {:?} cycles",
                    value.qualify(fid)
                ));
                break;
            }
            at = edge.next;
        }
    };
    for insn in body.insns.iter() {
        walk(LocalValueId::Instruction(insn.id), out);
    }
    for param in body.params.iter() {
        walk(LocalValueId::BlockParam(param.id), out);
    }
    for block in body.blocks.iter() {
        walk(LocalValueId::BasicBlock(block.id), out);
    }
    for temp in body.temps.iter() {
        walk(LocalValueId::Temp(temp.id), out);
    }
    for &value in body.shared_first_use.keys() {
        if matches!(
            value,
            LocalValueId::Instruction(_)
                | LocalValueId::BlockParam(_)
                | LocalValueId::BasicBlock(_)
                | LocalValueId::Temp(_)
        ) {
            out.push(format!(
                "function {fid:?}: local value {:?} has its use-list head in the shared map",
                value.qualify(fid)
            ));
        }
        walk(value, out);
    }
    if reached.len() != body.uses.len() {
        out.push(format!(
            "function {fid:?}: {} of {} use edges are reachable from a use-list head",
            reached.len(),
            body.uses.len()
        ));
    }
}

#[cfg(test)]
mod tests {
    use crate::value::QCodeMut;
    use std::borrow::Cow;

    use wazabin_qcode_macro::qcode;

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
    fn reports_self_referential_instruction() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        // `%y = @x + 1` — rewrite its `@x` operand to `%y` itself.
        let y = BasicBlock::from_id(&ctx, entry)
            .first_instruction()
            .unwrap();
        let x = ctx
            .instruction(y)
            .mnemonic()
            .args()
            .into_iter()
            .next()
            .unwrap();
        ctx.bodies[f]
            .insn_mut(y)
            .mnemonic_mut()
            .replace_value(x, LocalValueId::Instruction(y.local));

        assert_has(&ctx, "references itself as an operand");
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
    fn reports_unrostered_block() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let block = BasicBlock::make(&mut ctx, f).id;
        ctx.bodies[f].roster.retain(|&local| local != block.local);

        assert_has(&ctx, "roster count 0");
    }

    #[test]
    fn reports_instruction_membership_and_parent_disagreement() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let y = ctx.bodies[f]
            .insn_ids(entry.local)
            .find(|&id| !ctx.bodies[f].insns[id].mnemonic().is_terminator())
            .expect("value instruction");
        // A second block whose list also reaches `y`.
        let other = ctx.bodies[f].make_block();
        let list = &mut ctx.bodies[f].blocks[other.local].instructions;
        list.first = Some(y);
        list.last = Some(y);
        list.len = 1;

        assert_has(&ctx, "block memberships");
    }

    #[test]
    fn permits_live_detached_instruction() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let detached = ctx.bodies[f]
            .insn_ids(entry.local)
            .find(|&id| !ctx.bodies[f].insns[id].mnemonic().is_terminator())
            .expect("value instruction");
        ctx.bodies[f].unlink(detached);

        assert_eq!(verify_body_arena_integrity(&ctx), Vec::<String>::new());
    }

    #[test]
    fn reports_stale_instruction_and_parameter_membership() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let insn = ctx.bodies[f]
            .insn_ids(entry.local)
            .find(|&id| !ctx.bodies[f].insns[id].mnemonic().is_terminator())
            .expect("value instruction");
        let param = ctx.block(entry).params[0];
        ctx.bodies[f].insns.remove(insn);
        ctx.bodies[f].params.remove(param);

        assert_has(&ctx, "references removed instruction");
        assert_has(&ctx, "references removed parameter");
        assert_has(&ctx, "references removed local value");
        assert_has(&ctx, "has removed user");
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
        let target = BlockId::new(f, ctx.edge(f, edge).to);
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

    // A cross-function edge endpoint is unrepresentable since stage 1: edge
    // endpoints are bare `LocalBlockId`s carrying no function qualifier, so the
    // old `reports_cross_function_edge_endpoint` probe and fixture are gone.

    #[test]
    fn reports_branch_targeting_removed_block() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let edge = *ctx.block(entry).edges.iter().next().expect("edge");
        let target = BlockId::new(f, ctx.edge(f, edge).to);
        ctx.delete_block(target);

        assert_has(&ctx, "targets removed block");
    }

    #[test]
    fn reports_stale_local_name() {
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

        assert_has(&ctx, "local name \"stale\"");
    }

    /// The use edges of a value whose storage is pulled out from under them
    /// point at nothing; the operands they mirror dangle too.
    #[test]
    fn reports_use_edges_of_removed_value() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let x = ctx.block(entry).params[0];
        ctx.bodies[f].params.remove(x);

        assert_has(&ctx, "references removed local value");
        assert_has(&ctx, "names removed value");
    }

    /// An operand rewritten behind the body's back has an edge that names
    /// the old value, and the new operand has no edge.
    #[test]
    fn reports_operand_rewritten_without_its_edge() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let y = BasicBlock::from_id(&ctx, entry)
            .first_instruction()
            .unwrap();
        let one = ctx.get_const(1, 8).id().strip_func();
        ctx.bodies[f].insns[y.local]
            .mnemonic_mut()
            .set_operand(0, one);

        assert_has(&ctx, "but it is");
        assert_has(&ctx, "has 0 use edges");
    }

    /// A use list must reach each edge once, and only edges of its own value.
    #[test]
    fn reports_use_list_that_leads_astray() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let y = BasicBlock::from_id(&ctx, entry)
            .first_instruction()
            .unwrap();
        let x = ctx.block(entry).params[0];
        // Point `@x`'s head at `%y`'s only use (by the return).
        let y_use = ctx.bodies[f].insns[y.local].first_use;
        assert!(y_use.is_some());
        ctx.bodies[f].params[x].first_use = y_use;

        assert_has(&ctx, "is reachable more than once");
        assert_has(&ctx, "holds use");
        assert_has(&ctx, "use edges are reachable from a use-list head");
    }

    /// The edges are a function of the operands: rebuilding them from
    /// scratch reproduces the incrementally maintained graph.
    #[test]
    fn rebuilt_use_edges_match_maintained_ones() {
        let mut ctx = fixture();
        let f = ctx.function_ids()[0];
        let entry = FunctionBody::from_id(&ctx, f).root().expect("root").id;
        let y = BasicBlock::from_id(&ctx, entry)
            .first_instruction()
            .unwrap();
        let two = ctx.get_const(2, 8).id();
        let z = crate::value::InstructionRef::from_mnemonic(
            &mut ctx,
            f,
            Mnemonic::Binop(crate::value::insn::Binary {
                op: crate::value::insn::Binop::Int(crate::value::insn::IntBinop::Add),
                lhs: ValueId::Instruction(y).strip_func(),
                rhs: ValueId::Instruction(y).strip_func(),
            }),
            8,
        )
        .id;
        let terminator = BasicBlock::from_id(&ctx, entry).last_instruction().unwrap();
        ctx.insert_insn_before(entry, terminator, z);
        ctx.replace_all_uses_with(ValueId::Instruction(y), two);
        ctx.bodies[f].replace_operand(z, 1, ValueId::Instruction(y));
        assert_eq!(verify_body_arena_integrity(&ctx), Vec::<String>::new());

        let edges = |ctx: &Context<'_>| {
            let mut edges: Vec<_> = ctx.bodies[f]
                .use_edges()
                .map(|e| (format!("{:?}", e.value), e.user, e.operand_index))
                .collect();
            edges.sort();
            edges
        };
        let maintained = edges(&ctx);
        ctx.bodies[f].rebuild_uses();
        assert_eq!(verify_body_arena_integrity(&ctx), Vec::<String>::new());
        assert_eq!(edges(&ctx), maintained);
        assert!(
            maintained
                .iter()
                .any(|(v, u, i)| *v == format!("{:?}", two.strip_func())
                    && *u == z.local
                    && *i == 0)
        );
        assert!(maintained.iter().any(|(v, u, i)| *v
            == format!("{:?}", ValueId::Instruction(y).strip_func())
            && *u == z.local
            && *i == 1));
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
        let insn = ctx.bodies[f]
            .insn_ids(entry.local)
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
