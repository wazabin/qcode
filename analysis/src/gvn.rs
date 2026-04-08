use std::collections::{HashMap, HashSet};

use crate::AliasResult;

use qcode::{
    context::Context,
    value::{
        Value, ValueId, ValueRef,
        block::BlockMutRef,
        insn::{Binary, Binop, BoolBinop, FloatBinop, InstructionId, IntBinop, Mnemonic, Unop},
        literal::LiteralRef,
        util::base_ref::{WithCtx, WithCtxMut},
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
                Binop::Bool(BoolBinop::Xor) => (((l != 0) as u64 ^ (r != 0) as u64) as u64, 1),

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

/// Simple GVN pass over all instructions in `block`.
///
/// Processes instructions in registry (insertion) order. Loads are GVN-able
/// but are invalidated by intervening stores that may-alias the load pointer
/// according to `aliases`. Pure instructions are always GVN-able.
/// Terminators, calls, and `PCodeOp` are excluded.
pub fn gvn(block: &mut BlockMutRef, aliases: Option<&AliasResult>) {
    let mut table: HashMap<Mnemonic, ValueId> = HashMap::new();
    let mut redundant: HashSet<InstructionId> = HashSet::new();

    let insns = block.instruction_ids().to_vec();

    for insn_id in insns {
        let insn = block.ctx().get_insn(insn_id);
        let id = insn.id();

        let mut mnemonic = insn.mnemonic().clone();

        match mnemonic {
            Mnemonic::Store(store) => {
                let store_ptr = store.ptr;

                table.retain(|k, _| {
                    if let Mnemonic::Load(load) = k {
                        aliases.is_some_and(|aliases| !aliases.may_alias(store_ptr, load.ptr))
                    } else {
                        true
                    }
                });

                table.insert(Mnemonic::Load(store.get_matching_load()), store.src);
            }

            _ if mnemonic.is_terminator() || insn.size() == 0 => {}

            _ => {
                if let Some(cst) = constant_folding(block.ctx_mut(), &mnemonic) {
                    block.ctx_mut().replace_all_uses_with(id, cst);
                    redundant.insert(insn_id);
                    continue;
                }

                normalize(&mut mnemonic);

                match table.get(&mnemonic) {
                    Some(&leader) => {
                        block.ctx_mut().replace_all_uses_with(id, leader);
                        redundant.insert(insn_id);
                    }

                    None => {
                        table.insert(mnemonic, id);
                    }
                }
            }
        }
    }

    block.retain_insns(|insn| !redundant.contains(insn));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;
    use qcode::{builder::Builder, context::Context, value::BasicBlock};

    // 1. Same binop -> second is redundant, leader is first
    #[test]
    fn test_same_binop_redundant() {
        let mut ctx = Context::new();
        let mut builder = Builder::from_context(&mut ctx, 0x1000);
        let block_id = builder.block.id;

        qcode!("local i64 a; local i64 b;");
        let v1 = qcode!("{a} + {b}");
        let v2 = qcode!("{a} + {b}");

        builder.finalize(0x1001);

        let mut block = BasicBlock::from_id_mut(&mut ctx, block_id);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));

        // Simplify the block
        gvn(&mut block, None);

        assert!(block.instruction_ids().contains(&v1));
        assert!(!block.instruction_ids().contains(&v2));
    }

    // 2. Commutative normalization: a + b and b + a -> same value number
    #[test]
    fn test_commutative_normalization() {
        let mut ctx = Context::new();
        let mut builder = Builder::from_context(&mut ctx, 0x1000);
        let block_id = builder.block.id;

        qcode!("local i64 a; local i64 b;");
        let v1 = qcode!("{a} + {b}");
        let v2 = qcode!("{b} + {a}");

        builder.finalize(0x1001);

        let mut block = BasicBlock::from_id_mut(&mut ctx, block_id);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));

        // Simplify the block
        gvn(&mut block, None);

        assert!(block.instruction_ids().contains(&v1));
        assert!(!block.instruction_ids().contains(&v2));
    }

    // 3. Non-commutative not swapped: a - b ≠ b - a
    #[test]
    fn test_non_commutative_not_swapped() {
        let mut ctx = Context::new();
        let mut builder = Builder::from_context(&mut ctx, 0x1000);
        let block_id = builder.block.id;

        qcode!("local i64 a; local i64 b;");
        let v1 = qcode!("{a} - {b}");
        let v2 = qcode!("{b} - {a}");

        builder.finalize(0x1001);

        let mut block = BasicBlock::from_id_mut(&mut ctx, block_id);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));

        // Simplify the block
        gvn(&mut block, None);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));
    }

    // 4. Different ops -> distinct
    #[test]
    fn test_different_ops_distinct() {
        let mut ctx = Context::new();
        let mut builder = Builder::from_context(&mut ctx, 0x1000);
        let block_id = builder.block.id;

        qcode!("local i64 a; local i64 b;");
        let v1 = qcode!("{a} + {b}");
        let v2 = qcode!("{a} * {b}");

        builder.finalize(0x1001);

        let mut block = BasicBlock::from_id_mut(&mut ctx, block_id);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));

        // Simplify the block
        gvn(&mut block, None);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));
    }

    // 5. Constant propagation
    #[test]
    #[ignore = "WIP: constant folding not fully implemented yet"]
    fn test_constant_propagation() {
        let mut ctx = Context::new();
        let mut builder = Builder::from_context(&mut ctx, 0x1000);
        let block_id = builder.block.id;

        qcode!("local i64 a; local i64 b;");
        let v1 = qcode!("store({a}, i64 5); {a} + 2");
        let v2 = qcode!("{v1} + 3");
        qcode!("store({b}, {v2})");

        builder.finalize(0x1001);

        let aliases = AliasResult::from_space_ids(&ctx);

        let mut block = BasicBlock::from_id_mut(&mut ctx, block_id);

        assert!(block.instruction_ids().contains(&v1));
        assert!(block.instruction_ids().contains(&v2));

        println!("{}", block);
        gvn(&mut block, Some(&aliases));
        println!("{}", block);

        assert!(!block.instruction_ids().contains(&v1));
        assert!(!block.instruction_ids().contains(&v2));
        assert!(block.to_string().contains("b = 0xa"));
    }
}
