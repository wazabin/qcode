use std::collections::{HashMap, HashSet};

use jstd::graph::analysis::{DominatorTree, compute_dominators};

use crate::AliasResult;

use qcode::{
    context::Context,
    space::{Space, SpaceType},
    value::{
        BasicBlock, Function, Value, ValueId, ValueRef, VarnodeId,
        block::{BlockId, BlockMutRef},
        function::FunctionId,
        insn::{
            Binary, Binop, BoolBinop, FloatBinop, InstructionId, InstructionRef, IntBinop, Load,
            Mnemonic, Range, Unop,
        },
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
        ValueId::BlockParam(x) => x.into(),
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
    match ValueRef::new(v, ctx) {
        ValueRef::Literal(c) => Some(c),
        _ => None,
    }
}

fn is_symbolic_literal(ctx: &Context, v: ValueId) -> bool {
    let ValueId::Literal(id) = v else {
        return false;
    };
    ctx.values.literals[id].symbolic.is_some()
}

// TODO: use the existing `evaluate_const` machinery instead of re-implementing it here.
fn constant_folding(ctx: &mut Context, m: &Mnemonic, _output_size: usize) -> Option<ValueId> {
    match m {
        &Mnemonic::Binop(Binary { lhs, rhs, op }) => {
            let lhs = get_const(ctx, lhs)?;
            let rhs = get_const(ctx, rhs)?;
            // Block/Function/String symbolic literals are not numeric constants and
            // must not be folded (folding would discard the symbolic annotation).
            // StackAddress-typed literals have symbolic=None and are foldable; their
            // type is preserved through the output type computed by binop_result below.
            if is_symbolic_literal(ctx, lhs.id()) || is_symbolic_literal(ctx, rhs.id()) {
                return None;
            }
            // Shifts take a shift *count* whose operand width may be narrower than
            // the value being shifted (e.g. `shr eax, cl`: 4-byte value, 1-byte
            // count). For every other binop the operands must be the same width —
            // the lifter guarantees it, so a mismatch is a type error.
            let is_shift = matches!(
                op,
                Binop::Int(IntBinop::ShiftLeft | IntBinop::ShiftRight | IntBinop::SShiftRight)
            );
            assert!(
                is_shift || lhs.size() == rhs.size(),
                "type error in binop constant folding: lhs is {} bytes, rhs is {} bytes",
                lhs.size(),
                rhs.size()
            );

            let l = lhs.value();
            let r = rhs.value();

            let size = lhs.size();
            let lhs_type = lhs.type_id();
            let rhs_type = rhs.type_id();

            let mask = if size >= 8 {
                u64::MAX
            } else {
                (1u64 << (size * 8)) - 1
            };

            let value = match op {
                Binop::Int(IntBinop::Equal) => (l == r) as u64,
                Binop::Int(IntBinop::NotEqual) => (l != r) as u64,
                Binop::Int(IntBinop::Less) => (l < r) as u64,
                Binop::Int(IntBinop::LessEqual) => (l <= r) as u64,

                Binop::Int(IntBinop::Add) => l.wrapping_add(r) & mask,
                Binop::Int(IntBinop::Sub) => l.wrapping_sub(r) & mask,

                Binop::Int(IntBinop::Xor) => (l ^ r) & mask,
                Binop::Int(IntBinop::And) => (l & r) & mask,
                Binop::Int(IntBinop::Or) => (l | r) & mask,
                Binop::Int(IntBinop::ShiftLeft) => (l << r) & mask,
                Binop::Int(IntBinop::ShiftRight) => (l >> r) & mask,
                // Arithmetic shift: sign-extend the lhs from its byte width to
                // i64, shift, then re-mask to the lhs width.
                Binop::Int(IntBinop::SShiftRight) => {
                    let bits = size * 8;
                    let signed = if bits >= 64 {
                        l as i64
                    } else {
                        ((l << (64 - bits)) as i64) >> (64 - bits)
                    };
                    (signed >> r.min(63)) as u64 & mask
                }

                Binop::Int(IntBinop::Mul) => l.wrapping_mul(r) & mask,
                Binop::Int(IntBinop::Div) => l.wrapping_div(r) & mask,
                Binop::Int(IntBinop::Rem) => l.wrapping_rem(r) & mask,

                Binop::Bool(BoolBinop::And) => (l != 0 && r != 0) as u64,
                Binop::Bool(BoolBinop::Or) => (l != 0 || r != 0) as u64,
                Binop::Bool(BoolBinop::Xor) => (l != 0) as u64 ^ (r != 0) as u64,

                _ => {
                    return None;
                }
            };

            // Preserve the semantic type (e.g. StackAddress) through folding.
            let out_type = ctx.types.binop_result(lhs_type, op, rhs_type);
            Some(ctx.get_typed_const(value, out_type).id())
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

        Mnemonic::Range(range) => {
            // Extract `range.size` bytes starting at byte `range.start` of a
            // constant (e.g. EDI = low 4 bytes of a wide RDI literal).
            let src = get_const(ctx, range.src)?;
            let shifted = src.value() >> (range.start * 8);
            let mask = if range.size >= 8 {
                u64::MAX
            } else {
                (1u64 << (range.size * 8)) - 1
            };
            Some(ctx.get_const(shifted & mask, range.size).id())
        }

        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Algebraic identities
// ---------------------------------------------------------------------------

/// The all-ones bit pattern for an `output_size`-byte value.
fn all_ones(output_size: usize) -> u64 {
    if output_size >= 8 {
        u64::MAX
    } else {
        (1u64 << (output_size * 8)) - 1
    }
}

/// Concrete value of `v` when it is a non-symbolic literal, else `None`.
fn const_value(ctx: &Context, v: ValueId) -> Option<u64> {
    if is_symbolic_literal(ctx, v) {
        return None;
    }
    get_const(ctx, v).map(|c| c.value())
}

/// Algebraic simplifications that, unlike [`constant_folding`], do *not* require
/// both operands to be constant: idempotent, self-inverse, identity-element and
/// annihilator laws. Returns the value the instruction collapses to (an existing
/// operand or an interned constant) when a law applies.
fn algebraic_identity(ctx: &mut Context, m: &Mnemonic, output_size: usize) -> Option<ValueId> {
    let &Mnemonic::Binop(Binary { lhs, rhs, op }) = m else {
        return None;
    };
    let Binop::Int(op) = op else {
        return None;
    };

    // x OP x laws (operand identity, independent of any constant value).
    if lhs == rhs {
        match op {
            // x & x = x ; x | x = x
            IntBinop::And | IntBinop::Or => return Some(lhs),
            // x ^ x = 0 ; x - x = 0
            IntBinop::Xor | IntBinop::Sub => return Some(ctx.get_const(0, output_size).id()),
            _ => {}
        }
    }

    let l = const_value(ctx, lhs);
    let r = const_value(ctx, rhs);

    match op {
        // x + 0 = x ; 0 + x = x ; x - 0 = x  (Sub is not commutative)
        IntBinop::Add => {
            if r == Some(0) {
                return Some(lhs);
            }
            if l == Some(0) {
                return Some(rhs);
            }
        }
        IntBinop::Sub
            if r == Some(0) => {
                return Some(lhs);
            }
        // x | 0 = x ; x ^ 0 = x  (both commutative)
        IntBinop::Or | IntBinop::Xor => {
            if r == Some(0) {
                return Some(lhs);
            }
            if l == Some(0) {
                return Some(rhs);
            }
        }
        // x * 0 = 0 ; x * 1 = x
        IntBinop::Mul => {
            if l == Some(0) || r == Some(0) {
                return Some(ctx.get_const(0, output_size).id());
            }
            if r == Some(1) {
                return Some(lhs);
            }
            if l == Some(1) {
                return Some(rhs);
            }
        }
        // x & 0 = 0 ; x & ~0 = x
        IntBinop::And => {
            if l == Some(0) || r == Some(0) {
                return Some(ctx.get_const(0, output_size).id());
            }
            if r == Some(all_ones(output_size)) {
                return Some(lhs);
            }
            if l == Some(all_ones(output_size)) {
                return Some(rhs);
            }
        }
        // x << 0 = x ; x >> 0 = x  (shift amount is the rhs)
        IntBinop::ShiftLeft | IntBinop::ShiftRight | IntBinop::SShiftRight
            if r == Some(0) => {
                return Some(lhs);
            }
        _ => {}
    }

    None
}

// ---------------------------------------------------------------------------
// Flag-idiom recognition
// ---------------------------------------------------------------------------

/// If `v` is defined by an `IntBinop::want` binop, return its `(lhs, rhs)`.
fn as_int_binop(ctx: &Context, v: ValueId, want: IntBinop) -> Option<(ValueId, ValueId)> {
    let ValueId::Instruction(id) = v else {
        return None;
    };
    match ctx.get_insn(id).mnemonic() {
        Mnemonic::Binop(Binary {
            lhs,
            rhs,
            op: Binop::Int(op),
        }) if *op == want => Some((*lhs, *rhs)),
        _ => None,
    }
}

/// If `v` is defined by an `sborrow`, return its `(lhs, rhs)`.
fn as_sborrow(ctx: &Context, v: ValueId) -> Option<(ValueId, ValueId)> {
    let ValueId::Instruction(id) = v else {
        return None;
    };
    match ctx.get_insn(id).mnemonic() {
        Mnemonic::SBorrow(sb) => Some((sb.lhs, sb.rhs)),
        _ => None,
    }
}

/// Recognize the x86 signed-less-than flag idiom and collapse it to `a s< b`.
///
/// A signed `cmp a, b` lowers to `OF != SF` of `a - b`:
/// ```text
///   %of  = sborrow(a, b)
///   %sub = a - b
///   %sf  = %sub s< 0
///   %lt  = %of != %sf      // == (a s< b)
/// ```
/// Returns the rewritten `SLess(a, b)` mnemonic. The original `sborrow`/`sub`/
/// `slt` instructions are left for DCE to remove once their last use is gone.
fn simplify_flag_idiom(ctx: &Context, m: &Mnemonic) -> Option<Mnemonic> {
    let &Mnemonic::Binop(Binary {
        lhs,
        rhs,
        op: Binop::Int(IntBinop::NotEqual),
    }) = m
    else {
        return None;
    };

    // The `!=` is commutative (and GVN may have normalized it), so try both
    // assignments of which side is the sborrow and which is the `s< 0`.
    let resolve = |sborrow_side: ValueId, slt_side: ValueId| -> Option<(ValueId, ValueId)> {
        let (a, b) = as_sborrow(ctx, sborrow_side)?;
        let (sub_v, zero) = as_int_binop(ctx, slt_side, IntBinop::SLess)?;
        if const_value(ctx, zero) != Some(0) {
            return None;
        }
        let (sa, sb) = as_int_binop(ctx, sub_v, IntBinop::Sub)?;
        (sa == a && sb == b).then_some((a, b))
    };

    let (a, b) = resolve(lhs, rhs).or_else(|| resolve(rhs, lhs))?;
    Some(Mnemonic::Binop(Binary {
        lhs: a,
        rhs: b,
        op: Binop::Int(IntBinop::SLess),
    }))
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

/// Constant-fold every foldable instruction in `func_id` to interned literals,
/// iterating to a fixpoint.
///
/// Canonicalizes pointer arithmetic (e.g. the `stack_base - 8` then `+ offset`
/// chains left by brighten+mem2reg) into single stack-address literals so an
/// [`AliasResult`](crate::AliasResult) built *afterwards* sees per-slot locations
/// instead of collapsing every slot onto the shared `stack_base` root. This keeps
/// the oracle consistent with the pointers [`gvn_function`] reasons about: without
/// it, GVN folds these adds into fresh literals the precomputed oracle has never
/// seen, so loop-carried stack stores become invisible to
/// [`prune_loop_carried_loads`] and are wrongly forwarded across loop back-edges.
///
/// Folding only — no CSE or load/store forwarding. Returns `true` if anything
/// changed.
pub fn constant_fold_function(ctx: &mut Context, func_id: FunctionId) -> bool {
    let block_ids: Vec<BlockId> = Function::from_id(ctx, func_id)
        .iter()
        .map(|block| block.id)
        .collect();

    let mut changed_any = false;
    loop {
        let mut changed = false;
        for &block_id in &block_ids {
            let insns = BasicBlock::from_id(ctx, block_id)
                .instruction_ids()
                .to_vec();
            let mut redundant: HashSet<InstructionId> = HashSet::new();

            for insn_id in insns {
                let insn = ctx.get_insn(insn_id);
                let id = insn.id();
                let size = insn.size();
                let mnemonic = insn.mnemonic().clone();

                if mnemonic.is_terminator() || size == 0 {
                    continue;
                }

                if let Some(cst) = constant_folding(ctx, &mnemonic, size) {
                    ctx.replace_all_uses_with(id, cst);
                    redundant.insert(insn_id);
                    changed = true;
                } else if let Some(simplified) = algebraic_identity(ctx, &mnemonic, size) {
                    ctx.replace_all_uses_with(id, simplified);
                    redundant.insert(insn_id);
                    changed = true;
                }
            }

            if !redundant.is_empty() {
                BasicBlock::from_id_mut(ctx, block_id)
                    .retain_insns(|insn| !redundant.contains(insn));
            }
        }
        changed_any |= changed;
        if !changed {
            break;
        }
    }
    changed_any
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
    let inherited = prune_loop_carried_loads(ctx, block_id, inherited, tree, aliases);
    let updated = gvn_block_inner(ctx, block_id, inherited.as_ref(), aliases);
    // A block that ends in a call clobbers registers: its dominated children run
    // after the call, so register `Load` leaders the call clobbers must not be
    // forwarded into them (e.g. a caller's post-call `RAX` read is the callee's
    // result, not a value computed before the call).
    let for_children = prune_clobbered_by_call(ctx, block_id, &updated, aliases);
    for &child in tree.children_of(block_id) {
        gvn_block_rec(ctx, child, for_children.as_ref(), tree, aliases);
    }
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
fn prune_clobbered_by_call<'a>(
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

fn prune_loop_carried_loads<'a>(
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

    // Blocks unreachable from the function root are not covered by the walk
    // above. The common case is the fall-through after a `call`: a `Call` is a
    // block terminator, but the lifter records no CFG edge from the call site
    // back to its return block, so the entire post-call region (register reload
    // chains, the epilogue, ...) is orphaned. Optimize each such region with its
    // own dominator tree, rooted at its entry — an unreachable block with no
    // predecessor. Once calls grow a proper return edge this loop simply finds
    // nothing to do.
    let is_reachable = |b: BlockId| b == root || tree.dominates(root, b);
    let block_ids: Vec<BlockId> = Function::from_id(ctx, func_id)
        .iter()
        .map(|block| block.id)
        .collect();

    for block_id in block_ids {
        if is_reachable(block_id) {
            continue;
        }
        // Only start at region entries; interior blocks are reached by the
        // sub-walk from their entry.
        if BasicBlock::from_id(ctx, block_id)
            .predecessors()
            .next()
            .is_some()
        {
            continue;
        }
        let subtree = compute_dominators(ctx, block_id);
        gvn_block_rec(ctx, block_id, &HashMap::new(), &subtree, aliases);
    }
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

                %v1 = %a + %b;
                %v2 = %a + %b;

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
                    %v1 = %a + %b;
                    goto <merge>;

                <right>
                    goto <merge>;

                <merge>
                    %v2 = %a + %b;"
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

    #[test]
    fn test_gvn_function_does_not_forward_loads_across_loop_header() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i32 A;

            fn loop_load:
                <entry>
                    store(&A, i32 0);
                    goto <header>;

                <header>
                    %v = load(i32, &A);
                    if i8 1 goto <body> else goto <exit>;

                <body>
                    store(&A, i32 1);
                    goto <header>;

                <exit>
                    return [0x1000];
            "
        );

        let aliases = AliasResult::simple(&ctx);
        gvn_function(&mut ctx, loop_load, Some(&aliases));

        assert!(
            BasicBlock::from_id(&ctx, header)
                .instruction_ids()
                .contains(&v),
            "header load must not be replaced by the entry store; the backedge may overwrite it"
        );
    }

    #[test]
    fn test_gvn_function_forwards_loads_across_loop_header_when_body_stores_do_not_alias() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i32 A;
            varnode i32 B;

            fn loop_load:
                <entry>
                    store(&A, i32 7);
                    goto <header>;

                <header>
                    %v = load(i32, &A);
                    if i8 1 goto <body> else goto <exit>;

                <body>
                    store(&B, i32 1);
                    goto <header>;

                <exit>
                    return [0x1000];
            "
        );

        let aliases = AliasResult::simple(&ctx);
        gvn_function(&mut ctx, loop_load, Some(&aliases));

        assert!(
            !BasicBlock::from_id(&ctx, header)
                .instruction_ids()
                .contains(&v),
            "header load should be replaced by the dominating store when loop stores do not alias"
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
                %v1 = %a + 2;
                %v2 = %v1 + 3;
                store(&B, %v2);
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

    #[test]
    #[should_panic(expected = "type error in binop constant folding")]
    fn constant_folding_reports_mixed_size_literals() {
        let mut ctx = Context::new();
        let lhs = ctx.get_const(0xf0, 4).id();
        let rhs = ctx.get_const(0xff, 1).id();

        let _ = constant_folding(
            &mut ctx,
            &Mnemonic::Binop(Binary {
                op: Binop::Int(IntBinop::And),
                lhs,
                rhs,
            }),
            4,
        );
    }

    #[test]
    fn constant_folding_preserves_stack_address_type_through_folding() {
        let mut ctx = Context::new();
        let stack = ctx.add_space(qcode::space::Space {
            name: Some(Box::from("stack")),
            word_size: 1,
            addr_size: 8,
            ty: qcode::space::SpaceType::Ram,
        });
        // StackAddress-typed literals carry provenance in their TypeId, not in a
        // symbolic annotation. Constant folding must propagate that type so that
        // alias analysis can still distinguish SA results from plain integers.
        let sa_type = ctx.types.get_or_make_stack_address(8, Some(stack));
        let base_lid = ctx
            .values
            .get_or_make_typed_literal(0x1000_0000_0000_0000, sa_type, 8);
        let base = qcode::value::ValueId::Literal(base_lid);
        let offset = ctx.get_const(8, 8).id();

        let folded = constant_folding(
            &mut ctx,
            &Mnemonic::Binop(Binary {
                op: Binop::Int(IntBinop::Sub),
                lhs: base,
                rhs: offset,
            }),
            8,
        );

        let folded_id = folded.expect("SA - Int should constant-fold to a SA-typed literal");
        let qcode::value::ValueId::Literal(lid) = folded_id else {
            panic!("folded result must be a literal");
        };
        assert_eq!(
            ctx.values.literals[lid].type_id, sa_type,
            "folded SA - Int must preserve the StackAddress TypeId for alias analysis"
        );
    }

    // -----------------------------------------------------------------------
    // Algebraic identities
    // -----------------------------------------------------------------------

    /// `x & x` is idempotent: the AND is replaced by `x` itself.
    #[test]
    fn test_algebraic_and_self_is_idempotent() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;
            <block>
                %a = load(i64, &A);
                %v = %a & %a;
                store(&B, %v);
                goto <0x1001>;
        "
        );

        let aliases = AliasResult::simple(&ctx);
        let mut block = BasicBlock::from_id_mut(&mut ctx, block);
        assert!(block.instruction_ids().contains(&v));

        gvn(&mut block, Some(&aliases));

        assert!(
            !block.instruction_ids().contains(&v),
            "x & x should be eliminated"
        );
        assert!(
            block.instruction_ids().contains(&a),
            "the AND should be replaced by x itself, which stays live"
        );
    }

    /// `x + 0` collapses to `x`.
    #[test]
    fn test_algebraic_add_zero_identity() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;
            <block>
                %a = load(i64, &A);
                %v = %a + 0x0;
                store(&B, %v);
                goto <0x1001>;
        "
        );

        let aliases = AliasResult::simple(&ctx);
        let mut block = BasicBlock::from_id_mut(&mut ctx, block);

        gvn(&mut block, Some(&aliases));

        assert!(
            !block.instruction_ids().contains(&v),
            "x + 0 should be eliminated"
        );
        assert!(block.instruction_ids().contains(&a));
    }

    /// `x ^ x` folds to the zero constant.
    #[test]
    fn test_algebraic_xor_self_is_zero() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;
            <block>
                %a = load(i64, &A);
                %v = %a ^ %a;
                store(&B, %v);
                goto <0x1001>;
        "
        );

        let aliases = AliasResult::simple(&ctx);
        let mut block = BasicBlock::from_id_mut(&mut ctx, block);

        gvn(&mut block, Some(&aliases));

        assert!(
            !block.instruction_ids().contains(&v),
            "x ^ x should be eliminated"
        );
        assert!(
            block.to_string().contains("B = 0x0"),
            "x ^ x should fold to the zero constant, got:\n{block}"
        );
    }

    // -----------------------------------------------------------------------
    // Signed-compare flag idiom
    // -----------------------------------------------------------------------

    /// `sborrow(a, b) != ((a - b) s< 0)` is the x86 signed-less-than idiom and
    /// must collapse to a single `a s< b`, leaving the flag math dead.
    #[test]
    fn test_flag_idiom_collapses_to_signed_less_than() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 A;
            varnode i32 B;

            fn cmp:
                <entry>
                    %a = load(i32, &A);
                    %b = load(i32, &B);
                    %of = sborrow(%a, %b);
                    %sub = %a - %b;
                    %sf = %sub s< 0x0;
                    %lt = %of != %sf;
                    if %lt goto <t> else goto <e>;
                <t>
                    return [0x1000];
                <e>
                    return [0x2000];
            "
        );

        let aliases = AliasResult::simple(&ctx);
        gvn_function(&mut ctx, cmp, Some(&aliases));

        assert!(
            !BasicBlock::from_id(&ctx, entry)
                .instruction_ids()
                .contains(&lt),
            "the `!=` flag combination should be rewritten away"
        );

        // After DCE the dead sborrow/sub/slt chain disappears, leaving the
        // single signed comparison the idiom was recognized as.
        crate::remove_dead_insns(&mut ctx, entry);
        let text = BasicBlock::from_id(&ctx, entry).to_string();
        assert!(
            text.contains("s<"),
            "expected a single signed-less-than, got:\n{text}"
        );
        assert!(
            !text.contains("sborrow") && !text.contains("!="),
            "the flag math should be dead after the rewrite, got:\n{text}"
        );
    }

    // -----------------------------------------------------------------------
    // Register store→load forwarding
    // -----------------------------------------------------------------------

    /// Storing to a register and reading it straight back must forward the
    /// stored value, even when overlapping sub-registers (r0/r0_lo32/...) put
    /// the location in a multi-member alias class. Mirrors the post-call
    /// `*[register]:4 EAX = v; %r = *[register]:4 EAX` reload chains the lifter
    /// emits across every fixture.
    #[test]
    fn test_register_store_load_forwarding() {
        use qcode::{builder::Builder, testing::TestContext, value::Function};

        let mut tc = TestContext::new();
        let fun_id = Function::make(&mut tc.ctx, "test".into()).unwrap().id;
        let block_id = tc.ctx.get_or_make_block(0x1000);
        Function::from_id_mut(&mut tc.ctx, fun_id)
            .set_root(block_id)
            .unwrap();

        let reg_space = tc.reg_space;
        let eax = ValueId::Varnode(tc.r0_lo32);
        let other = ValueId::Varnode(tc.r1);

        let load_id;
        {
            let mut b = Builder::from_context(&mut tc.ctx, 0x1000);
            let c = b.context_mut().get_const(0x12345678, 4).id();
            b.push_store(c, eax, reg_space); // EAX = c
            let loaded = b.push_load::<false>(eax, 4, reg_space).id(); // %r = EAX
            b.push_store(loaded, other, reg_space); // use %r (keeps it live)
            load_id = loaded;
            unsafe { b.dont_finalize() };
        }

        let aliases = AliasResult::simple(&tc.ctx);
        gvn_function(&mut tc.ctx, fun_id, Some(&aliases));

        let ValueId::Instruction(load_insn) = load_id else {
            panic!("push_load should produce an instruction value");
        };
        assert!(
            !BasicBlock::from_id(&tc.ctx, block_id)
                .instruction_ids()
                .contains(&load_insn),
            "register reload should be forwarded to the stored value, got:\n{}",
            BasicBlock::from_id(&tc.ctx, block_id)
        );
    }

    /// A `call` is a block terminator with no CFG edge to its fall-through, so
    /// the post-call block is unreachable from the function root. `gvn_function`
    /// must still optimize it — otherwise the register reload chains the lifter
    /// emits after every call survive untouched.
    #[test]
    fn test_register_forwarding_in_orphaned_post_call_block() {
        use qcode::{builder::Builder, testing::TestContext, value::Function};

        let mut tc = TestContext::new();
        let fun_id = Function::make(&mut tc.ctx, "test".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let post_call = tc.ctx.get_or_make_block(0x2000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fun_id);
            f.set_root(entry).unwrap();
            f.add_block(entry);
            f.add_block(post_call);
        }

        let reg_space = tc.reg_space;
        let eax = ValueId::Varnode(tc.r0_lo32);
        let other = ValueId::Varnode(tc.r1);

        // Entry ends in a `call`; no edge links it to `post_call`.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_call(fun_id);
            unsafe { b.dont_finalize() };
        }

        // Orphaned fall-through: store a register and read it straight back.
        let load_id;
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, post_call));
            let c = b.context_mut().get_const(0x42, 4).id();
            b.push_store(c, eax, reg_space);
            let loaded = b.push_load::<false>(eax, 4, reg_space).id();
            b.push_store(loaded, other, reg_space);
            load_id = loaded;
            unsafe { b.dont_finalize() };
        }

        let aliases = AliasResult::simple(&tc.ctx);
        gvn_function(&mut tc.ctx, fun_id, Some(&aliases));

        let ValueId::Instruction(load_insn) = load_id else {
            panic!("push_load should produce an instruction value");
        };
        assert!(
            !BasicBlock::from_id(&tc.ctx, post_call)
                .instruction_ids()
                .contains(&load_insn),
            "forwarding must reach the orphaned post-call block, got:\n{}",
            BasicBlock::from_id(&tc.ctx, post_call)
        );
    }
}
