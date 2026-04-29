use jstd::graph::analysis::compute_dominators;
use qcode::value::{
    BasicBlock, BlockId, BlockParamId, Function, FunctionId, Instruction, ValueId, Varnode,
    insn::{Branch, CBranch, InstructionId, Load, Mnemonic, Store},
};
use qcode::{
    context::Context,
    value::{FunctionRef, block::BlockRef},
};
use std::collections::{HashMap, HashSet};

pub fn mem2reg(ctx: &mut Context, function_id: FunctionId) {
    let vars = collect_promotable_vars(ctx, function_id);
    if vars.is_empty() {
        return;
    }

    let root_id = Function::from_id(ctx, function_id)
        .root()
        .expect("function has no root")
        .id;

    let frontier: HashMap<BlockId, HashSet<BlockId>> = {
        let dom = compute_dominators(&*ctx, root_id);
        dom.dominator_frontier().clone()
    };

    let var_params = insert_block_params(ctx, function_id, &vars, &frontier);

    let mut visited = HashSet::new();
    let mut frames = vec![Frame::new()];
    decide_values_start_from(ctx, root_id, &mut visited, &mut frames, &var_params);

    remove_promoted_stores(ctx, function_id, &vars);
}

fn collect_promotable_vars(ctx: &Context, function_id: FunctionId) -> HashSet<ValueId> {
    let mut vars = HashSet::new();
    for block in Function::from_id(ctx, function_id).blocks() {
        for insn in block.iter() {
            match insn.mnemonic() {
                Mnemonic::Load(Load { ptr, .. }) | Mnemonic::Store(Store { ptr, .. })
                    if ptr.is_varnode() =>
                {
                    vars.insert(*ptr);
                }
                _ => {}
            }
        }
    }
    vars
}

fn insert_block_params(
    ctx: &mut Context,
    function_id: FunctionId,
    vars: &HashSet<ValueId>,
    frontier: &HashMap<BlockId, HashSet<BlockId>>,
) -> HashMap<BlockId, HashMap<ValueId, BlockParamId>> {
    let mut var_params: HashMap<BlockId, HashMap<ValueId, BlockParamId>> = HashMap::new();

    for &var in vars {
        let ValueId::Varnode(varnode_id) = var else {
            continue;
        };
        let size = Varnode::from_id(ctx, varnode_id).size();

        let phi_positions = {
            let function = Function::from_id(ctx, function_id);
            find_phi_insert_positions(var, &function, frontier)
        };

        for block_id in phi_positions {
            let param_id = BasicBlock::from_id_mut(ctx, block_id).push_param(size).id;
            var_params
                .entry(block_id)
                .or_default()
                .insert(var, param_id);
        }
    }

    var_params
}

fn compute_branch_args(
    target: BlockId,
    var_params: &HashMap<BlockId, HashMap<ValueId, BlockParamId>>,
    frames: &[Frame],
    ctx: &Context,
) -> Vec<ValueId> {
    let Some(params) = var_params.get(&target) else {
        return vec![];
    };

    let mut indexed: Vec<(usize, ValueId)> = params
        .iter()
        .map(|(&var, &param_id)| {
            let index = ctx.values.block_params[param_id].index;
            let (_, val) = decide_variable_value(var, frames).unwrap_or_else(|| {
                panic!(
                    "variable {:?} undefined at branch to parameterized block",
                    var
                )
            });
            (index, val)
        })
        .collect();
    indexed.sort_by_key(|(index, _)| *index);
    indexed.into_iter().map(|(_, val)| val).collect()
}

fn remove_promoted_stores(ctx: &mut Context, function_id: FunctionId, vars: &HashSet<ValueId>) {
    let block_ids: Vec<BlockId> = Function::from_id(ctx, function_id)
        .blocks()
        .map(|b| b.id)
        .collect();

    for block_id in block_ids {
        let insn_ids: Vec<InstructionId> = BasicBlock::from_id(ctx, block_id)
            .instruction_ids()
            .to_vec();

        for insn_id in insn_ids {
            let is_promoted_store = matches!(
                Instruction::from_id(ctx, insn_id).mnemonic(),
                Mnemonic::Store(Store { ptr, .. }) if vars.contains(ptr)
            );
            if is_promoted_store {
                ctx.remove_instruction(insn_id);
            }
        }
    }
}

fn block_contains_store_to_var(block: &BlockRef, var: ValueId) -> bool {
    block.iter().any(|insn| {
        if let Mnemonic::Store(Store { ptr, .. }) = insn.mnemonic() {
            *ptr == var
        } else {
            false
        }
    })
}

fn find_phi_insert_positions(
    var: ValueId,
    function: &FunctionRef,
    frontier: &HashMap<BlockId, HashSet<BlockId>>,
) -> HashSet<BlockId> {
    let block_containing_store = function
        .blocks()
        .filter(|b| block_contains_store_to_var(b, var))
        .map(|b| b.id)
        .collect::<Vec<_>>();

    let mut worklist = block_containing_store.clone();
    let mut result = HashSet::new();

    while let Some(block) = worklist.pop() {
        for dominated in frontier[&block].iter().copied() {
            result.insert(dominated);

            if !block_containing_store.contains(&dominated) {
                worklist.push(dominated);
            }
        }
    }

    result
}

type LocalValue = (BlockId, ValueId);
type Frame = HashMap<ValueId, LocalValue>;

fn decide_variable_value(var: ValueId, frames: &[Frame]) -> Option<LocalValue> {
    for frame in frames.iter().rev() {
        if let Some(value) = frame.get(&var) {
            return Some(*value);
        }
    }

    None
}

fn decide_values_start_from(
    ctx: &mut Context,
    block: BlockId,
    visited: &mut HashSet<BlockId>,
    frames: &mut Vec<Frame>,
    var_params: &HashMap<BlockId, HashMap<ValueId, BlockParamId>>,
) {
    if visited.contains(&block) {
        return;
    }
    visited.insert(block);

    // Seed the current frame with block params so loads within this block
    // and its Branch-reachable descendants see them as the current SSA value.
    if let Some(params) = var_params.get(&block) {
        let frame = frames.last_mut().unwrap();
        for (&var, &param_id) in params {
            frame.insert(var, (block, ValueId::BlockParam(param_id)));
        }
    }

    let insn_ids: Vec<InstructionId> = BasicBlock::from_id(ctx, block).instruction_ids().to_vec();

    for insn_id in insn_ids {
        // Clone the mnemonic so we can release the immutable borrow on ctx
        // before taking mutable borrows in each arm.
        let mnemonic = Instruction::from_id(ctx, insn_id).mnemonic().clone();

        match mnemonic {
            Mnemonic::Store(Store { ptr, src, .. }) if ptr.is_varnode() => {
                frames.last_mut().unwrap().insert(ptr, (block, src));
            }

            Mnemonic::Load(Load { ptr, .. }) if ptr.is_varnode() => {
                let (_, load_value) = decide_variable_value(ptr, frames).unwrap_or_else(|| {
                    panic!(
                        "Unable to decide value for variable {:?} in block {:?}",
                        ptr, block
                    )
                });
                ctx.replace_all_uses_with(ValueId::Instruction(insn_id), load_value);
                ctx.remove_instruction(insn_id);
            }

            Mnemonic::CBranch(CBranch {
                success_block,
                failure_block,
                ..
            }) => {
                // Compute args before recursing; even if a target is already visited
                // (back-edge), we still need to wire the correct values.
                let success_args = compute_branch_args(success_block, var_params, frames, ctx);
                let failure_args = compute_branch_args(failure_block, var_params, frames, ctx);

                if let Mnemonic::CBranch(cb) = Instruction::from_id_mut(ctx, insn_id).mnemonic_mut()
                {
                    cb.success_args = success_args;
                    cb.failure_args = failure_args;
                }

                frames.push(Frame::new());
                decide_values_start_from(ctx, success_block, visited, frames, var_params);
                frames.pop();

                frames.push(Frame::new());
                decide_values_start_from(ctx, failure_block, visited, frames, var_params);
                frames.pop();
            }

            Mnemonic::Branch(Branch { target, .. }) => {
                let args = compute_branch_args(target, var_params, frames, ctx);
                if let Mnemonic::Branch(b) = Instruction::from_id_mut(ctx, insn_id).mnemonic_mut() {
                    b.args = args;
                }
                decide_values_start_from(ctx, target, visited, frames, var_params);
            }

            Mnemonic::Call(_) | Mnemonic::CallInd(_) | Mnemonic::BranchInd(_) => {
                panic!("mem2reg does not support calls or indirect branches yet");
            }

            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {

    use jstd::graph::analysis::compute_dominators;
    use qcode::value::{BasicBlock, Function, insn::Mnemonic};
    use qcode_macro::qcode;

    use super::*;

    #[test]
    fn phi_insert_pos() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 A;

            fn test:
                <bb1>
                    store(&A, i32 1);
                    if i8 1 goto <bb2> else goto <bb4>;

                <bb2>
                    %a0 = load(i32, &A);
                    store(&A, i32 0);
                    goto <bb3>;

                <bb3>
                    %a1 = load(i32, &A);
                    store(&A, i32 1);
                    if i8 1 goto <bb2> else goto <bb8>;

                <bb4>
                    if i8 1 goto <bb5> else goto <bb6>;

                <bb5>
                    %a2 = load(i32, &A);
                    store(&A, i32 2);
                    goto <bb7>;

                <bb6>
                    %a3 = load(i32, &A);
                    store(&A, i32 3);
                    goto <bb7>;

                <bb7>
                    %a4 = load(i32, &A);
                    goto <bb8>;

                <bb8>
                    %a5 = load(i32, &A);
                    return [0];
        "
        );

        let function = Function::from_id(&ctx, test);
        let dom = compute_dominators(&ctx, function.root().unwrap().id);
        let frontier = dom.dominator_frontier();

        let result = find_phi_insert_positions(A.into(), &function, frontier);

        let named_result = result
            .iter()
            .map(|b| BasicBlock::from_id(&ctx, *b).name().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(
            named_result.len(),
            3,
            "Expected 3 blocks to require arguments nodes"
        );
        assert!(named_result.contains(&"bb2"), "bb2 should be in the result");
        assert!(named_result.contains(&"bb7"), "bb7 should be in the result");
        assert!(named_result.contains(&"bb8"), "bb8 should be in the result");
    }

    #[test]
    fn mem2reg_diamond() {
        // Diamond CFG: two paths each storing different values, joined at exit.
        // exit needs a block param for A.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 A;

            fn test:
                <entry>
                    store(&A, i32 1);
                    if i8 1 goto <left> else goto <right>;

                <left>
                    store(&A, i32 2);
                    goto <exit>;

                <right>
                    store(&A, i32 3);
                    goto <exit>;

                <exit>
                    %val = load(i32, &A);
                    return [0];
        "
        );

        mem2reg(&mut ctx, test);

        // exit block should have exactly one block param for A
        let exit_block = BasicBlock::from_id(&ctx, exit);
        assert_eq!(exit_block.num_params(), 1, "exit should have 1 block param");

        // No loads or stores to A should remain
        let function = Function::from_id(&ctx, test);
        for block in function.blocks() {
            for insn in block.iter() {
                assert!(
                    !matches!(insn.mnemonic(), Mnemonic::Load(_) | Mnemonic::Store(_)),
                    "no loads or stores should remain after mem2reg"
                );
            }
        }

        // left and right branches to exit should carry args
        let left_block = BasicBlock::from_id(&ctx, left);
        let left_branch = left_block.iter().last().unwrap();
        let Mnemonic::Branch(b) = left_branch.mnemonic() else {
            panic!("expected branch");
        };
        assert_eq!(b.args.len(), 1, "left→exit branch should pass 1 arg");

        let right_block = BasicBlock::from_id(&ctx, right);
        let right_branch = right_block.iter().last().unwrap();
        let Mnemonic::Branch(b) = right_branch.mnemonic() else {
            panic!("expected branch");
        };
        assert_eq!(b.args.len(), 1, "right→exit branch should pass 1 arg");
    }

    #[test]
    fn mem2reg_linear() {
        // No join points: no block params needed.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 A;

            fn test:
                <entry>
                    store(&A, i32 42);
                    goto <exit>;

                <exit>
                    %val = load(i32, &A);
                    return [0];
        "
        );

        mem2reg(&mut ctx, test);

        // No block params anywhere
        let function = Function::from_id(&ctx, test);
        for block in function.blocks() {
            assert_eq!(
                block.num_params(),
                0,
                "no block params expected in linear CFG"
            );
        }

        // No loads or stores remain
        for block in Function::from_id(&ctx, test).blocks() {
            for insn in block.iter() {
                assert!(
                    !matches!(insn.mnemonic(), Mnemonic::Load(_) | Mnemonic::Store(_)),
                    "no loads or stores should remain"
                );
            }
        }
    }

    #[test]
    fn remove_mem() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 A;

            fn test:
                <bb1>
                    store(&A, i32 1);
                    if i8 1 goto <bb2> else goto <bb4>;

                <bb2>
                    %a0 = load(i32, &A);
                    %t0 = %a0 + i32 1;
                    store(&A, %t0);
                    goto <bb3>;

                <bb3>
                    %a1 = load(i32, &A);
                    %t1 = %a1 + %a1;
                    store(&A, %t1);
                    if i8 1 goto <bb2> else goto <bb8>;

                <bb4>
                    if i8 1 goto <bb5> else goto <bb6>;

                <bb5>
                    %a2 = load(i32, &A);
                    %t2 = %a2 + i32 2;
                    store(&A, %t2);
                    goto <bb7>;

                <bb6>
                    %a3 = load(i32, &A);
                    %t3 = %a3 + i32 3;
                    store(&A, %t3);
                    goto <bb7>;

                <bb7>
                    %a4 = load(i32, &A);
                    goto <bb8>;

                <bb8>
                    %a5 = load(i32, &A);
                    return [0];
        "
        );

        let function = Function::from_id(&ctx, test);
        let dom = compute_dominators(&ctx, function.root().unwrap().id);
        let frontier = dom.dominator_frontier();

        let result = find_phi_insert_positions(A.into(), &function, frontier);

        let named_result = result
            .iter()
            .map(|b| BasicBlock::from_id(&ctx, *b).name().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(
            named_result.len(),
            3,
            "Expected 3 blocks to require phi nodes"
        );
        assert!(named_result.contains(&"bb2"), "bb2 should be in the result");
        assert!(named_result.contains(&"bb7"), "bb7 should be in the result");
        assert!(named_result.contains(&"bb8"), "bb8 should be in the result");
    }
}
