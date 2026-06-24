use std::collections::HashMap;

use qcode::{
    context::Context,
    value::{
        BasicBlock, BlockId, Instruction, InstructionId, InstructionRef, ValueId,
        insn::{Assert, Branch, CBranch, Mnemonic, Unary, Unop},
    },
};

/// A sequence of instruction addresses captured during execution.
pub trait Trace {
    fn list_addresses(&self) -> &[u64];
}

/// The result of mapping a [`Trace`] onto a [`Context`]: a sequence of cloned
/// basic blocks with branch conditions constrained to match the observed path.
pub struct ResolvedPath {
    /// Cloned block ids in trace order, living in the same context as the original.
    pub blocks: Vec<BlockId>,
    /// Maps original ValueIds to their cloned counterparts.
    pub value_map: HashMap<ValueId, ValueId>,
}

impl ResolvedPath {
    /// Resolves a perfect trace against the IR in `ctx`.
    ///
    /// For each consecutive pair of addresses in the trace:
    /// 1. The block at the current address is deep-cloned into `ctx`.
    /// 2. If the block ends with a [`CBranch`], an assert (optionally negated)
    ///    is inserted to record which branch was actually taken.
    ///
    /// After all blocks are cloned, every cloned `CBranch` terminator is
    /// replaced with an unconditional branch to the next block in trace order.
    pub fn from_perfect_trace(ctx: &mut Context, trace: &impl Trace) -> Self {
        let addresses = trace.list_addresses();
        let mut blocks = Vec::new();
        let mut value_map = HashMap::new();

        // Iterating over current address + successor to resolve paths
        for window in addresses.windows(2) {
            let [current_addr, next_addr] = window else {
                continue;
            };

            // Find the original block at current address
            let Some(orig_block_id) = BasicBlock::from_addr(ctx, *current_addr).map(|b| b.id)
            else {
                eprintln!("[trace] warning: no block at {current_addr:#x}, skipping");
                continue;
            };

            // Deep clone the block
            let new_block_id = BasicBlock::clone_into_ctx(ctx, orig_block_id, &mut value_map);

            let terminator = ctx.values.basic_blocks[orig_block_id]
                .instructions
                .last()
                .map(|&id| ctx.values.instructions[id].mnemonic().clone());

            match terminator {
                Some(Mnemonic::CBranch(cbranch)) => {
                    // Check for negation
                    let Some(negate) = resolve_cbranch_negate(ctx, &cbranch, *next_addr) else {
                        continue;
                    };
                    // CBranch is at the end of the cloned block
                    let cbranch_id = *BasicBlock::from_id(ctx, new_block_id)
                        .instruction_ids()
                        .last()
                        .unwrap();

                    let raw_condition = value_map
                        .get(&cbranch.condition)
                        .copied()
                        .unwrap_or(cbranch.condition);

                    // Add (negated) assert
                    insert_trace_assert(ctx, new_block_id, cbranch_id, raw_condition, negate);

                    blocks.push(new_block_id);
                }

                Some(
                    Mnemonic::Branch(_)
                    | Mnemonic::Call(_)
                    | Mnemonic::CallInd(_)
                    | Mnemonic::Return(_)
                    | Mnemonic::BranchInd(_),
                ) => {
                    blocks.push(new_block_id);
                }

                None => {
                    eprintln!("[trace] warning: block at {current_addr:#x} is empty");
                    blocks.push(new_block_id);
                }

                Some(other) => {
                    eprintln!(
                        "[trace] warning: unexpected terminator {other:?} at {current_addr:#x}"
                    );
                    blocks.push(new_block_id);
                }
            }
        }

        // Wire each cloned CBranch block to its successor in trace order.
        // This replaces the two-target CBranch with an unconditional branch,
        // since the assert above already encodes which path was taken.
        for i in 0..blocks.len().saturating_sub(1) {
            let block_id = blocks[i];
            let next_block_id = blocks[i + 1];

            let is_cbranch = BasicBlock::from_id(ctx, block_id)
                .instruction_ids()
                .last()
                .is_some_and(|&id| {
                    matches!(
                        Instruction::from_id(ctx, id).mnemonic(),
                        Mnemonic::CBranch(_)
                    )
                });

            if is_cbranch {
                wire_cbranch_to_next(ctx, block_id, next_block_id);
            }
        }

        Self { blocks, value_map }
    }
}

/// Checks if a negate is needed for the CBranch conditional
fn resolve_cbranch_negate(ctx: &Context, cbranch: &CBranch, next_addr: u64) -> Option<bool> {
    if BasicBlock::from_id(ctx, cbranch.success_block).address() == Some(next_addr) {
        Some(false)
    } else if BasicBlock::from_id(ctx, cbranch.failure_block).address() == Some(next_addr) {
        Some(true)
    } else {
        eprintln!("[trace] error: next address {next_addr:#x} is neither success nor failure.");
        None
    }
}

/// Inserts a (negated) assert before `cbranch_id` in `block_id`.
fn insert_trace_assert(
    ctx: &mut Context,
    block_id: BlockId,
    cbranch_id: InstructionId,
    condition: ValueId,
    negate: bool,
) {
    // Generate the not instruction if needed
    let condition = if negate {
        let not_id = InstructionRef::from_mnemonic(
            ctx,
            Mnemonic::Unop(Unary {
                op: Unop::BoolNot,
                src: condition,
            }),
            1,
        )
        .id;
        BasicBlock::from_id_mut(ctx, block_id).insert_insn_before(cbranch_id, not_id);
        ValueId::Instruction(not_id)
    } else {
        condition
    };

    // Setup assert and insert it (along with not) before the cbranch
    let assert_id =
        InstructionRef::from_mnemonic(ctx, Mnemonic::Assert(Assert { condition }), 0).id;
    BasicBlock::from_id_mut(ctx, block_id).insert_insn_before(cbranch_id, assert_id);
}

/// Replaces the cbranch terminator of `block_id` with an unconditional branch to `next_block_id`.
fn wire_cbranch_to_next(ctx: &mut Context, block_id: BlockId, next_block_id: BlockId) {
    BasicBlock::from_id_mut(ctx, block_id).pop_insn();
    ctx.add_cfg_edge(block_id, next_block_id);
    let branch_id = InstructionRef::from_mnemonic(
        ctx,
        Mnemonic::Branch(Branch {
            target: next_block_id,
            args: vec![],
        }),
        0,
    )
    .id;
    BasicBlock::from_id_mut(ctx, block_id).push_insn(branch_id);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{context::Context, value::BasicBlock};
    use qcode_macro::qcode;

    struct SimpleTrace(Vec<u64>);
    impl Trace for SimpleTrace {
        fn list_addresses(&self) -> &[u64] {
            &self.0
        }
    }

    /// Builds a simple context with a conditional branch and returns a trace
    /// resolved through the success path.
    fn make_cbranch_ctx() -> (Context<'static>, BlockId, BlockId, BlockId, BlockId) {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;
            <entry>
                %a = load(i64, &A);
                %b = load(i64, &B);
                if i8 1 goto <success> else goto <failure>;

            <success>
                %v1 = %a + %b;
                goto <out>;

            <failure>
                goto <out>;

            <out>
                goto <0x1001>;"
        );
        BasicBlock::from_id_mut(&mut ctx, entry)
            .set_address(0x1000)
            .unwrap();
        BasicBlock::from_id_mut(&mut ctx, success)
            .set_address(0x1010)
            .unwrap();
        BasicBlock::from_id_mut(&mut ctx, failure)
            .set_address(0x1020)
            .unwrap();
        BasicBlock::from_id_mut(&mut ctx, out)
            .set_address(0x1030)
            .unwrap();
        (ctx, entry, success, failure, out)
    }

    /// Builds a simple context with a conditional branch and returns a trace
    /// resolved through the success path.
    fn make_loop_ctx() -> (Context<'static>, BlockId, BlockId, BlockId, BlockId) {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 COND;

            <entry>
                goto <cond>;

            <cond>
                %c = load(i8, &COND);
                if i8 %c goto <body> else goto <exit>;

            <body>
                goto <cond>;

            <exit>
                goto <0x1001>;
            "
        );
        BasicBlock::from_id_mut(&mut ctx, entry)
            .set_address(0x1000)
            .unwrap();
        BasicBlock::from_id_mut(&mut ctx, cond)
            .set_address(0x1010)
            .unwrap();
        BasicBlock::from_id_mut(&mut ctx, body)
            .set_address(0x1020)
            .unwrap();
        BasicBlock::from_id_mut(&mut ctx, exit)
            .set_address(0x1030)
            .unwrap();
        (ctx, entry, cond, body, exit)
    }

    // Success path should insert an assert but no nop
    #[test]
    fn test_success_path_inserts_assert_no_not() {
        let (mut ctx, _, _, _, _) = make_cbranch_ctx();

        let orig_block = BasicBlock::from_addr(&ctx, 0x1000).unwrap();
        assert!(
            matches!(
                orig_block.iter().last().unwrap().mnemonic(),
                Mnemonic::CBranch(_)
            ),
            "original block must end with cbranch"
        );

        // entry -> success -> out
        let trace = SimpleTrace(vec![0x1000, 0x1010, 0x1030]);
        let resolved = ResolvedPath::from_perfect_trace(&mut ctx, &trace);

        let block = BasicBlock::from_id(&ctx, resolved.blocks[0]);
        let insns: Vec<_> = block.iter().collect();
        let n = insns.len();

        assert!(
            matches!(insns[n - 1].mnemonic(), Mnemonic::Branch(_)),
            "last must be unconditional branch"
        );
        assert!(
            matches!(insns[n - 2].mnemonic(), Mnemonic::Assert(_)),
            "second to last must be assert"
        );
        assert!(
            !matches!(insns[n-3].mnemonic(), Mnemonic::Unop(u) if u.op == Unop::BoolNot),
            "no BoolNot should be present before assert on success path"
        );
    }

    #[test]
    fn test_failure_path_inserts_negated_assert() {
        let (mut ctx, _, _, _, _) = make_cbranch_ctx();

        let orig_block = BasicBlock::from_addr(&ctx, 0x1000).unwrap();
        assert!(
            matches!(
                orig_block.iter().last().unwrap().mnemonic(),
                Mnemonic::CBranch(_)
            ),
            "original block must end with cbranch"
        );

        // entry -> success -> out
        let trace = SimpleTrace(vec![0x1000, 0x1020, 0x1030]);
        let resolved = ResolvedPath::from_perfect_trace(&mut ctx, &trace);

        let block = BasicBlock::from_id(&ctx, resolved.blocks[0]);
        let insns: Vec<_> = block.iter().collect();
        let n = insns.len();

        assert!(
            matches!(insns[n - 1].mnemonic(), Mnemonic::Branch(_)),
            "last must be unconditional branch"
        );
        assert!(
            matches!(insns[n - 2].mnemonic(), Mnemonic::Assert(_)),
            "second to last must be assert"
        );
        assert!(
            matches!(insns[n-3].mnemonic(), Mnemonic::Unop(u) if u.op == Unop::BoolNot),
            "third to last must be BoolNot on failure path"
        );
    }

    #[test]
    fn test_loop_clones_distinct_names_and_insns() {
        let (mut ctx, _, _, _, _) = make_loop_ctx();
        // entry -> (cond -> body)*2 -> cond -> exit
        let trace = SimpleTrace(vec![0x1000, 0x1010, 0x1020, 0x1010, 0x1020, 0x1010, 0x1030]);
        let resolved = ResolvedPath::from_perfect_trace(&mut ctx, &trace);

        let names: Vec<_> = resolved
            .blocks
            .iter()
            .map(|&id| {
                BasicBlock::from_id(&ctx, id)
                    .name()
                    .unwrap_or("")
                    .to_string()
            })
            .collect();

        // All names must be unique
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(
            unique.len(),
            names.len(),
            "all cloned block names must be distinct: {names:?}"
        );

        // Collect all instruction ids across all cloned blocks
        let all_ids: Vec<_> = resolved
            .blocks
            .iter()
            .flat_map(|&id| BasicBlock::from_id(&ctx, id).instruction_ids().to_vec())
            .collect();

        let unique: std::collections::HashSet<_> = all_ids.iter().collect();
        assert_eq!(
            unique.len(),
            all_ids.len(),
            "cloned blocks must not share instruction ids"
        );
    }
}
