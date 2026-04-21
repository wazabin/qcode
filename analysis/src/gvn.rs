use std::collections::{HashMap, HashSet};

use jstd::graph::analysis::{DominatorTree, compute_dominators};

use crate::AliasResult;

use qcode::{
    context::Context,
    value::{
        BasicBlock, Value, ValueId, ValueRef,
        block::{BlockId, BlockMutRef},
        function::FunctionId,
        insn::{Binary, Binop, BoolBinop, FloatBinop, InstructionId, IntBinop, Mnemonic, Unop},
        literal::LiteralRef,
        util::base_ref::WithCtxMut,
    },
};

// ---------------------------------------------------------------------------
// Normalization (used for commutative binops)
// ---------------------------------------------------------------------------

fn value_id_index(v: ValueId) -> usize {
    match v {
        ValueId::Literal(x) => x.into(),
        ValueId::Instruction(x) => x.into(),
        ValueId::Varnode(x) => x.into(),
        ValueId::BasicBlock(x) => x.into(),
        ValueId::Function(x) => x.into(),
        _ => todo!("unsupported value id type in value_id_index: {:?}", v),
    }
}

fn is_commutative(op: &Binop) -> bool {
    matches!(
        op,
        Binop::Int(
            IntBinop::Add
                | IntBinop::Mul
                | IntBinop::And
                | IntBinop::Or
                | IntBinop::Xor
                | IntBinop::Equal
                | IntBinop::NotEqual
        ) | Binop::Bool(BoolBinop::And | BoolBinop::Or | BoolBinop::Xor)
            | Binop::Float(FloatBinop::Equal | FloatBinop::NotEqual)
    )
}

/// We want to canonicalize commutative binops so that e.g. a + b and b + a are
/// represented by the same value number.
fn normalize(m: &mut Mnemonic) {
    if let Mnemonic::Binop(b) = m
        && is_commutative(&b.op)
    {
        let l = value_id_index(b.lhs);
        let r = value_id_index(b.rhs);
        if l > r {
            std::mem::swap(&mut b.lhs, &mut b.rhs);
        }
    }
}

// ---------------------------------------------------------------------------
// Constant folding
// ---------------------------------------------------------------------------

fn get_const<'a>(ctx: &'a Context<'a>, v: ValueId) -> Option<LiteralRef<'a, 'a>> {
    match ctx.get_value(v) {
        ValueRef::Literal(c) => Some(c),
        _ => None,
    }
}

// TODO: use the existing `evaluate_const` machinery instead of re-implementing it here.
fn constant_folding(ctx: &mut Context, m: &Mnemonic) -> Option<ValueId> {
    match m {
        &Mnemonic::Binop(Binary { lhs, rhs, op }) => {
            let lhs = get_const(ctx, lhs)?;
            let rhs = get_const(ctx, rhs)?;

            let l = lhs.value();
            let r = rhs.value();

            let size = lhs.size();
            let mask = lhs.mask();

            debug_assert_eq!(size, rhs.size(), "mismatched operand sizes in binop");

            let (value, size) = match op {
                Binop::Int(IntBinop::Equal) => ((l == r) as u64, 1),
                Binop::Int(IntBinop::NotEqual) => ((l != r) as u64, 1),
                Binop::Int(IntBinop::Less) => ((l < r) as u64, 1),
                Binop::Int(IntBinop::LessEqual) => ((l <= r) as u64, 1),

                Binop::Int(IntBinop::Add) => (l.wrapping_add(r) & mask, size),
                Binop::Int(IntBinop::Sub) => (l.wrapping_sub(r) & mask, size),

                Binop::Int(IntBinop::Xor) => ((l ^ r) & mask, size),
                Binop::Int(IntBinop::And) => ((l & r) & mask, size),
                Binop::Int(IntBinop::Or) => ((l | r) & mask, size),
                Binop::Int(IntBinop::ShiftLeft) => ((l << r) & mask, size),
                Binop::Int(IntBinop::ShiftRight) => ((l >> r) & mask, size),

                Binop::Int(IntBinop::Mul) => (l.wrapping_mul(r) & mask, size),
                Binop::Int(IntBinop::Div) => (l.wrapping_div(r) & mask, size),
                Binop::Int(IntBinop::Rem) => (l.wrapping_rem(r) & mask, size),

                Binop::Bool(BoolBinop::And) => ((l != 0 && r != 0) as u64, 1),
                Binop::Bool(BoolBinop::Or) => ((l != 0 || r != 0) as u64, 1),
                Binop::Bool(BoolBinop::Xor) => (((l != 0) as u64 ^ (r != 0) as u64), 1),

                _ => {
                    return None;
                }
            };

            Some(ctx.get_const(value, size).id())
        }

        Mnemonic::Unop(unop) => {
            let src = get_const(ctx, unop.src)?;

            let v = src.value();
            let size = src.size();
            let mask = src.mask();

            let value = match unop.op {
                Unop::IntNot => !v,
                Unop::IntNegate => v.wrapping_neg(),

                _ => {
                    return None;
                }
            };

            Some(ctx.get_const(value & mask, size).id())
        }

        Mnemonic::Zext(zext) => {
            let src = get_const(ctx, zext.src)?;
            Some(ctx.get_const(src.value(), zext.size).id())
        }

        Mnemonic::Sext(sext) => {
            let src = get_const(ctx, sext.src)?;
            let value = ((src.value() as i64) << (64 - src.size() * 8)) >> (64 - sext.size * 8);
            Some(ctx.get_const(value as u64, sext.size).id())
        }

        _ => None,
    }
}

// ---------------------------------------------------------------------------
// GVN pass
// ---------------------------------------------------------------------------

/// Process all instructions in `block_id` with an inherited value table.
///
/// Returns the updated table (inherited entries plus new entries from this block)
/// for descendants in the dominator tree to inherit.
fn gvn_block_inner(
    ctx: &mut Context,
    block_id: BlockId,
    inherited: &HashMap<Mnemonic, ValueId>,
    aliases: Option<&AliasResult>,
) -> HashMap<Mnemonic, ValueId> {
    let mut table = inherited.clone();
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
                table.insert(Mnemonic::Load(store.get_matching_load()), store.src);
            }

            _ if mnemonic.is_terminator() || insn_size == 0 => {}

            _ => {
                if let Some(cst) = constant_folding(ctx, &mnemonic) {
                    ctx.replace_all_uses_with(id, cst);
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
                        table.insert(mnemonic, id);
                    }
                }
            }
        }
    }

    BasicBlock::from_id_mut(ctx, block_id).retain_insns(|insn| !redundant.contains(insn));
    table
}

/// Single-block GVN pass (preserved for backward compatibility).
///
/// Processes instructions in registry (insertion) order. Loads are GVN-able
/// but are invalidated by intervening stores that may-alias the load pointer
/// according to `aliases`. Pure instructions are always GVN-able.
/// Terminators, calls, and `PCodeOp` are excluded.
pub fn gvn(block: &mut BlockMutRef, aliases: Option<&AliasResult>) {
    let block_id = block.id;
    gvn_block_inner(block.ctx_mut(), block_id, &HashMap::new(), aliases);
}

fn gvn_block_rec(
    ctx: &mut Context,
    block_id: BlockId,
    inherited: &HashMap<Mnemonic, ValueId>,
    tree: &DominatorTree<BlockId>,
    aliases: Option<&AliasResult>,
) {
    let updated = gvn_block_inner(ctx, block_id, inherited, aliases);
    for &child in tree.children_of(block_id) {
        gvn_block_rec(ctx, child, &updated, tree, aliases);
    }
}

/// Dominator-tree GVN over an entire function.
///
/// Walks the dominator tree in pre-order, propagating the value table from each
/// block to its dominated successors. A value computed in a dominator is always
/// available to every descendant, so redundant recomputations across blocks are
/// eliminated. Store/load invalidation follows the same alias-aware rules as the
/// single-block pass.
pub fn gvn_function(ctx: &mut Context, func_id: FunctionId, aliases: Option<&AliasResult>) {
    let root = match ctx.values.functions[func_id].root {
        Some(r) => r,
        None => return,
    };

    let tree = compute_dominators(ctx, root);
    gvn_block_rec(ctx, root, &HashMap::new(), &tree, aliases);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;
    use qcode::{context::Context, value::BasicBlock};

    // 1. Same binop -> second is redundant, leader is first
    #[test]
    fn test_same_binop_redundant() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;

            <block>
                %a = load(i64, &A);
                %b = load(i64, &B);

                %v1 = a + b;
                %v2 = a + b;

                goto <0x1001>;
        "
        );

        let mut block = BasicBlock::from_id_mut(&mut ctx, block);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));

        gvn(&mut block, None);

        assert!(block.instruction_ids().contains(&v1));
        assert!(!block.instruction_ids().contains(&v2));
    }

    // 2. Commutative normalization: a + b and b + a -> same value number
    #[test]
    fn test_commutative_normalization() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;

            <block>
                %a = load(i64, &A);
                %b = load(i64, &B);
                %v1 = %a + %b;
                %v2 = %b + %a;
                goto <0x1001>;
        "
        );

        let mut block = BasicBlock::from_id_mut(&mut ctx, block);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));

        gvn(&mut block, None);

        assert!(block.instruction_ids().contains(&v1));
        assert!(!block.instruction_ids().contains(&v2));
    }

    // 3. Non-commutative not swapped: a - b != b - a
    #[test]
    fn test_non_commutative_not_swapped() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;

            <block>
                %a = load(i64, &A);
                %b = load(i64, &B);
                %v1 = %a - %b;
                %v2 = %b - %a;
                goto <0x1001>;
        "
        );

        let mut block = BasicBlock::from_id_mut(&mut ctx, block);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));

        gvn(&mut block, None);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));
    }

    // 4. Different ops -> distinct
    #[test]
    fn test_different_ops_distinct() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;

            <block>
                %a = load(i64, &A);
                %b = load(i64, &B);
                %v1 = %a + %b;
                %v2 = %a * %b;
                goto <0x1001>;
        "
        );

        let mut block = BasicBlock::from_id_mut(&mut ctx, block);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));

        gvn(&mut block, None);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));
    }

    // 5. Cross-block redundancy: a+b in entry propagates to dominated successor
    #[test]
    fn test_gvn_function_cross_block_redundancy() {
        // Layout: entry → succ (linear chain, succ dominated by entry)
        // entry: v1 = a + b
        // succ:  v2 = a + b  <- redundant, dominated by entry
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;

            fn f:
                <entry>
                    %a = load(i64, &A);
                    %b = load(i64, &B);

                    %v1 = %a + %b;
                    goto <succ>;

                <succ>
                    %v2 = %a + %b;
                    return [0x1000];
            "
        );

        gvn_function(&mut ctx, f, None);

        assert!(
            BasicBlock::from_id(&ctx, entry)
                .instruction_ids()
                .contains(&v1)
        );
        assert!(
            !BasicBlock::from_id(&ctx, succ)
                .instruction_ids()
                .contains(&v2),
            "a+b in dominated successor should be eliminated"
        );
    }

    // 6. Diamond CFG: a+b computed in one branch is NOT available at merge
    #[test]
    fn test_gvn_function_no_propagation_across_merge() {
        // Layout: entry → left, entry → right, left → merge, right → merge
        // left:  v1 = a + b
        // right: (no a+b)
        // merge: v2 = a + b  <- NOT redundant; merge is dominated only by entry
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;

            fn g:
                <entry>
                    %a = load(i64, &A);
                    %b = load(i64, &B);
                    if i8 1 goto <left> else goto <right>;

                <left>
                    %v1 = a + b;
                    goto <merge>;

                <right>
                    goto <merge>;

                <merge>
                    %v2 = a + b;"
        );

        gvn_function(&mut ctx, g, None);

        assert!(
            BasicBlock::from_id(&ctx, left)
                .instruction_ids()
                .contains(&v1)
        );
        assert!(
            BasicBlock::from_id(&ctx, merge)
                .instruction_ids()
                .contains(&v2),
            "a+b at merge must NOT be eliminated: merge is not dominated by left"
        );
    }

    // 7. Constant propagation
    #[test]
    fn test_constant_propagation() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;
            <block>
                store(&A, i64 5);
                %a = load(i64, &A);
                %v1 = a + 2;
                %v2 = v1 + 3;
                store(&B, v2);
                goto <0x1001>;"
        );

        let aliases = AliasResult::simple(&ctx);

        let mut block = BasicBlock::from_id_mut(&mut ctx, block);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));

        println!("{}", block);
        gvn(&mut block, Some(&aliases));
        println!("{}", block);

        assert!(!block.instruction_ids().contains(&v1));
        assert!(!block.instruction_ids().contains(&v2));
        assert!(block.to_string().contains("B = 0xa"));
    }
}
