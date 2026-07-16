use qcode::value::QCodeMut;
use rustc_hash::FxHashMap as HashMap;

use qcode::{
    address_index::{AddressIndex, AddressTarget},
    context::Context,
    value::{
        BasicBlock, BlockId, Instruction, InstructionId, InstructionRef, ValueId,
        insn::{Assert, Binary, Binop, Branch, CBranch, IntBinop, Mnemonic},
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
        let address_index = AddressIndex::analyze(ctx);
        let mut blocks = Vec::new();
        let mut value_map = HashMap::default();

        // Iterating over current address + successor to resolve paths
        for window in addresses.windows(2) {
            let [current_addr, next_addr] = window else {
                continue;
            };

            // Find the original block at current address
            let Some(orig_block_id) = block_at_address(ctx, &address_index, *current_addr) else {
                eprintln!("[trace] warning: no block at {current_addr:#x}, skipping");
                continue;
            };

            // Deep clone the block
            let new_block_id = BasicBlock::clone_into_ctx(ctx, orig_block_id, &mut value_map);

            let terminator = BasicBlock::from_id(ctx, orig_block_id)
                .instruction_ids()
                .last()
                .map(|&id| Instruction::from_id(ctx, id).mnemonic().clone());

            match terminator {
                Some(Mnemonic::CBranch(cbranch)) => {
                    // Check for negation
                    let Some(negate) =
                        resolve_cbranch_negate(ctx, orig_block_id.func, &cbranch, *next_addr)
                    else {
                        continue;
                    };
                    // CBranch is at the end of the cloned block
                    let cbranch_id = *BasicBlock::from_id(ctx, new_block_id)
                        .instruction_ids()
                        .last()
                        .unwrap();

                    let orig_condition = cbranch.condition.qualify(orig_block_id.func);
                    let raw_condition = value_map
                        .get(&orig_condition)
                        .copied()
                        .unwrap_or(orig_condition);

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

        // Wire each cloned block to its successor in trace order so the resolved
        // path is self-contained rather than pointing back into the original CFG.
        //
        // - `CBranch`: replaced with an unconditional branch (the assert inserted
        //   above already encodes which side was taken).
        // - `Branch`: its target is retargeted to the cloned successor.
        //
        // `Call`/`CallInd`/`Return`/`BranchInd` are intentionally left untouched:
        // their targets are either indirect, absent, or semantically meaningful
        // (the original callee), so consumers should rely on `blocks` order across
        // those boundaries.
        for i in 0..blocks.len().saturating_sub(1) {
            let block_id = blocks[i];
            let next_block_id = blocks[i + 1];

            let terminator = BasicBlock::from_id(ctx, block_id)
                .instruction_ids()
                .last()
                .map(|&id| Instruction::from_id(ctx, id).mnemonic().clone());

            match terminator {
                Some(Mnemonic::CBranch(_)) => wire_cbranch_to_next(ctx, block_id, next_block_id),
                Some(Mnemonic::Branch(_)) => wire_branch_to_next(ctx, block_id, next_block_id),
                _ => {}
            }
        }

        Self { blocks, value_map }
    }
}

/// Resolves an address to a block while respecting the index's deliberate rule
/// that a function wins its collision with its entry block.
fn block_at_address(ctx: &Context<'_>, addresses: &AddressIndex, address: u64) -> Option<BlockId> {
    match addresses.get(address)? {
        AddressTarget::Block(block) => Some(block),
        AddressTarget::Function(function) => ctx
            .function(function)
            .root_id()
            .map(|local| BlockId::new(function, local)),
    }
}

/// Checks if a negate is needed for the CBranch conditional
fn resolve_cbranch_negate(
    ctx: &Context,
    func: qcode::value::FunctionId,
    cbranch: &CBranch,
    next_addr: u64,
) -> Option<bool> {
    // The CBranch's targets are body-local indices in `func`'s arena.
    if BasicBlock::from_id(ctx, BlockId::new(func, cbranch.success_block)).address()
        == Some(next_addr)
    {
        Some(false)
    } else if BasicBlock::from_id(ctx, BlockId::new(func, cbranch.failure_block)).address()
        == Some(next_addr)
    {
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
    // Generate the negation if needed — canonically `condition == false`.
    let condition = if negate {
        let f = ctx.get_bool_const(false).id();
        let bool_ty = ctx.shared.types.get_or_make_bool();
        let not_id = InstructionRef::from_mnemonic_with_type(
            ctx,
            block_id.func,
            Mnemonic::Binop(Binary {
                op: Binop::Int(IntBinop::Equal),
                lhs: condition.localize(block_id.func),
                rhs: f.localize(block_id.func),
            }),
            bool_ty,
        )
        .id;
        BasicBlock::from_id_mut(ctx, block_id).insert_insn_before(cbranch_id, not_id);
        ValueId::Instruction(not_id)
    } else {
        condition
    };

    // Setup assert and insert it (along with not) before the cbranch
    let assert_id = InstructionRef::from_mnemonic(
        ctx,
        block_id.func,
        Mnemonic::Assert(Assert {
            condition: condition.localize(block_id.func),
        }),
        0,
    )
    .id;
    BasicBlock::from_id_mut(ctx, block_id).insert_insn_before(cbranch_id, assert_id);
}

/// Replaces the cbranch terminator of `block_id` with an unconditional branch to `next_block_id`.
fn wire_cbranch_to_next(ctx: &mut Context, block_id: BlockId, next_block_id: BlockId) {
    BasicBlock::from_id_mut(ctx, block_id).pop_insn();
    ctx.add_cfg_edge(block_id, next_block_id);
    let branch_id = InstructionRef::from_mnemonic(
        ctx,
        block_id.func,
        Mnemonic::Branch(Branch {
            target: next_block_id.localize(block_id.func),
            args: vec![],
        }),
        0,
    )
    .id;
    BasicBlock::from_id_mut(ctx, block_id).push_insn(branch_id);
}

/// Retargets the unconditional branch terminator of `block_id` to `next_block_id`,
/// preserving the arguments passed to the block parameters.
fn wire_branch_to_next(ctx: &mut Context, block_id: BlockId, next_block_id: BlockId) {
    let last_id = *BasicBlock::from_id(ctx, block_id)
        .instruction_ids()
        .last()
        .expect("caller guarantees a branch terminator");
    let args = match Instruction::from_id(ctx, last_id).mnemonic() {
        Mnemonic::Branch(branch) => branch.args.clone(),
        _ => unreachable!("caller guarantees a branch terminator"),
    };

    BasicBlock::from_id_mut(ctx, block_id).pop_insn();
    ctx.add_cfg_edge(block_id, next_block_id);
    let branch_id = InstructionRef::from_mnemonic(
        ctx,
        block_id.func,
        Mnemonic::Branch(Branch {
            target: next_block_id.localize(block_id.func),
            args,
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
    use qcode::{
        context::Context,
        value::{BasicBlock, FunctionBody},
    };
    use qcode_macro::qcode;

    struct SimpleTrace(Vec<u64>);
    impl Trace for SimpleTrace {
        fn list_addresses(&self) -> &[u64] {
            &self.0
        }
    }

    #[test]
    fn address_index_function_collision_resolves_to_entry_block() {
        let mut ctx = Context::new();
        let function = FunctionBody::make_at_addr(&mut ctx, 0x1000, None).id;
        let entry = BasicBlock::make(&mut ctx, function).with_address(0x1000).id;
        FunctionBody::from_id_mut(&mut ctx, function)
            .set_root(entry)
            .unwrap();

        let addresses = AddressIndex::analyze(&ctx);
        assert_eq!(
            addresses.get(0x1000),
            Some(AddressTarget::Function(function))
        );
        assert_eq!(block_at_address(&ctx, &addresses, 0x1000), Some(entry));
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
                %a = load(A:8, &A);
                %b = load(B:8, &B);
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

    /// Builds a context with a loop (`cond` -> `body` -> `cond`) so a trace can
    /// exercise repeated cloning of the same blocks.
    fn make_loop_ctx() -> (Context<'static>, BlockId, BlockId, BlockId, BlockId) {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 COND;

            <entry>
                goto <cond>;

            <cond>
                %c = load(COND:1, &COND);
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

        let addresses = AddressIndex::analyze(&ctx);
        let orig_block =
            BasicBlock::from_id(&ctx, block_at_address(&ctx, &addresses, 0x1000).unwrap());
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
            !matches!(insns[n-3].mnemonic(), Mnemonic::Binop(b) if b.op == Binop::Int(IntBinop::Equal)),
            "no negation (`cond == false`) should be present before assert on success path"
        );
    }

    #[test]
    fn test_failure_path_inserts_negated_assert() {
        let (mut ctx, _, _, _, _) = make_cbranch_ctx();

        let addresses = AddressIndex::analyze(&ctx);
        let orig_block =
            BasicBlock::from_id(&ctx, block_at_address(&ctx, &addresses, 0x1000).unwrap());
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
            matches!(insns[n-3].mnemonic(), Mnemonic::Binop(b) if b.op == Binop::Int(IntBinop::Equal)),
            "third to last must be the negation (`cond == false`) on failure path"
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

        // Every block but the last must end with an unconditional branch wired to
        // the next cloned block, so the path stays inside the cloned CFG.
        let cloned: std::collections::HashSet<_> = resolved.blocks.iter().copied().collect();
        for window in resolved.blocks.windows(2) {
            let [block_id, next_id] = window else {
                continue;
            };
            let block = BasicBlock::from_id(&ctx, *block_id);
            match block.iter().last().map(|i| i.mnemonic().clone()) {
                Some(Mnemonic::Branch(branch)) => {
                    let target = BlockId::new(block_id.func, branch.target);
                    assert_eq!(
                        target, *next_id,
                        "cloned branch must target the cloned successor"
                    );
                    assert!(
                        cloned.contains(&target),
                        "branch target must be a cloned block, not the original CFG"
                    );
                }
                other => panic!("expected unconditional branch terminator, got {other:?}"),
            }
        }
    }
}
