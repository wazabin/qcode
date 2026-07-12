//! De-pipelining: remove a pipelined loop-carried "delay" register.
//!
//! A copy-until-terminator loop (`while (*src) *dst++ = *src++;`) is lowered with
//! a software-pipelined carry: the byte loaded at iteration `n` is stored at
//! iteration `n+1`, threaded through a loop-carried register. After argpromote
//! the header looks like:
//!
//! ```text
//! <H(@EAX, @ECX, @DL)>
//!     store(ram:1, @ECX <- @DL)          // uses the *previous* byte
//!     %t  = load(ram:1, @EAX)            // loads the *next* byte
//!     %ea = @EAX + 1
//!     %ec = @ECX + 1
//!     if %t != 0 goto <H @EAX=%ea @ECX=%ec @DL=%t> else goto <exit @ECX=%ec>
//! ```
//!
//! `@DL` is not redundant (its two incomings — the preheader's pre-loaded byte
//! and the back-edge's `%t` — are distinct), so the congruence pass correctly
//! keeps it. But it is a *delay register*: at iteration `n`, `@DL = load(@EAX − 1)`
//! (the previous iteration's load, since `@EAX` steps by 1). This pass re-derives
//! that value in the current iteration and drops the param.
//!
//! ## Recognition — the re-read must reproduce the carry on *every* entry
//!
//! Re-deriving `@DL` as `load(@EAX − step)` is only value-preserving if that
//! re-read equals the carry on every edge into the header, the prologue included
//! — not just the back-edge whose `load(@EAX)` shape names the carry. The
//! recognizer therefore requires, from *each* predecessor, that the carry's
//! incoming is `load(space, A, size)` and the induction variable's incoming `B`
//! satisfies `B = A + step` (so `load(B − step) = load(A)` is exactly that
//! incoming). A genuine pipelined prologue (`%dl = load(src)`, `@EAX = src + 1`)
//! passes; a constant seed or a mismatched pre-step is rejected, leaving the loop
//! untouched rather than miscompiling its first iteration.
//!
//! ## Soundness — the disjointness assumption
//!
//! Re-deriving `@DL` as `load(@EAX − step)` *re-reads* memory the body also writes
//! through `@ECX`. The re-read preserves the value only if the source and
//! destination buffers do not overlap — the C `strcpy`/`memcpy` contract (overlap
//! is `memmove`'s undefined-behaviour territory). The alias oracle cannot prove
//! two incoming pointer args disjoint, so this pass *records the assumption*
//! [`Proposition::CopyBuffersDisjoint`] for the function (the move the user
//! authorized: assume the C contract). It is `Assumed`, not `Known`; a future
//! verifier could refute it for a provably-overlapping caller.

use rustc_hash::FxHashSet as HashSet;

use jstd::graph::analysis::{DominatorTree, compute_dominators};

use qcode::{
    assumption::Proposition,
    context::Context,
    value::{
        BasicBlock, BlockId, BlockParamId, Function, FunctionId, InstructionRef, ValueId,
        insn::{Binary, Binop, IntBinop, Load, Mnemonic},
    },
};

use crate::dce::remove_params_from_block;
use crate::{Pass, PipelineEnv};

/// A detected pipelined carry: param `carry` (index `k`) on header `header` whose
/// back-edge value is `load(space, iv, size)` with `iv` an affine induction
/// variable stepping by `step`.
struct Pipelined {
    header: BlockId,
    k: usize,
    carry: BlockParamId,
    iv: ValueId,
    step: u64,
    space: qcode::space::SpaceId,
    size: usize,
}

/// `c` if `v` is the non-symbolic integer literal `c`.
fn literal(ctx: &Context, v: ValueId) -> Option<u64> {
    match qcode::value::ValueRef::new(v, ctx) {
        qcode::value::ValueRef::Literal(l) => Some(l.value()),
        _ => None,
    }
}

/// The positive constant step `s` if `v` is `iv + s` (either operand order).
fn affine_step(ctx: &Context, v: ValueId, iv: ValueId) -> Option<u64> {
    let ValueId::Instruction(id) = v else {
        return None;
    };
    let Mnemonic::Binop(b) = ctx.get_insn(id).mnemonic() else {
        return None;
    };
    if !matches!(b.op, Binop::Int(IntBinop::Add)) {
        return None;
    }
    let s = if b.lhs == iv {
        literal(ctx, b.rhs)?
    } else if b.rhs == iv {
        literal(ctx, b.lhs)?
    } else {
        return None;
    };
    (s != 0).then_some(s)
}

/// The argument bound to position `k` of `header` by `pred`'s terminator.
fn incoming_from(ctx: &Context, pred: BlockId, header: BlockId, k: usize) -> Option<ValueId> {
    let &term = ctx.block(pred).instruction_ids().last()?;
    match ctx.get_insn(term).mnemonic() {
        Mnemonic::Branch(b) if b.target == header => b.args.get(k).copied(),
        Mnemonic::CBranch(c) => {
            if c.success_block == header {
                c.success_args.get(k).copied()
            } else if c.failure_block == header {
                c.failure_args.get(k).copied()
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Find the first pipelined-carry param on a loop header of `fid`.
fn find_pipelined(
    ctx: &Context,
    fid: FunctionId,
    dom: &DominatorTree<BlockId>,
) -> Option<Pipelined> {
    for block in Function::from_id(ctx, fid).iter() {
        let header = block.id;

        // Back-edge predecessors are those `header` dominates; a header with none
        // is not a loop header.
        let preds: Vec<BlockId> = BasicBlock::from_id(ctx, header)
            .predecessors()
            .map(|(_, p)| p)
            .collect();
        let Some(backedge) = preds.iter().copied().find(|&p| dom.dominates(header, p)) else {
            continue;
        };

        // The disjointness assumption is only meaningful for a copy-shaped loop —
        // one that stores. (With no store the re-read is trivially safe, but then
        // there is nothing to de-pipeline against either.)
        let has_store = BasicBlock::from_id(ctx, header)
            .iter()
            .any(|i| matches!(i.mnemonic(), Mnemonic::Store(_)));
        if !has_store {
            continue;
        }

        let params: Vec<BlockParamId> = BasicBlock::from_id(ctx, header)
            .params()
            .map(|p| p.id)
            .collect();
        for (k, &carry) in params.iter().enumerate() {
            let carry_val = ValueId::BlockParam(carry);
            // The carry must actually be read; an unused param is plain DCE.
            if ctx.users(carry_val).is_empty() {
                continue;
            }
            // Back-edge value of the carry must be a load of an induction variable.
            let Some(bv) = incoming_from(ctx, backedge, header, k) else {
                continue;
            };
            let ValueId::Instruction(bv_id) = bv else {
                continue;
            };
            let Mnemonic::Load(Load { space, ptr, size }) = *ctx.get_insn(bv_id).mnemonic() else {
                continue;
            };
            // The load address must be a header param (the IV) stepping affinely.
            let ValueId::BlockParam(iv_pid) = ptr else {
                continue;
            };
            if ctx.block_param(iv_pid).parent_id() != Some(header) {
                continue;
            }
            let Some(j) = params.iter().position(|&p| p == iv_pid) else {
                continue;
            };
            let Some(iv_back) = incoming_from(ctx, backedge, header, j) else {
                continue;
            };
            let Some(step) = affine_step(ctx, iv_back, ptr) else {
                continue;
            };

            // Soundness: de-pipelining replaces the carry with `load(iv − step)`
            // recomputed in the header, so that re-read must reproduce the carry's
            // value on *every* entry to the header — not only the back-edge the
            // shape was identified from, but the prologue seed too. (Checking just
            // the back-edge would miscompile the first iteration of a loop whose
            // prologue seeds the carry with anything other than the matching byte.)
            //
            // Uniform condition, over all predecessors: the carry's incoming value
            // must be `load(space, A, size)` and the iv's incoming value `B` must
            // satisfy `B = A + step`, so that `load(B − step) = load(A)` is exactly
            // that incoming carry. The back-edge (carry `load(iv)`, iv `iv + step`)
            // satisfies this by construction; a genuine pipelined prologue pre-loads
            // the matching byte (`%dl = load(src)`, `@EAX = src + 1`); a constant
            // seed or a mismatched pre-step is rejected.
            let consistent = preds.iter().all(|&pred| {
                let (Some(ValueId::Instruction(carry_in)), Some(iv_in)) = (
                    incoming_from(ctx, pred, header, k),
                    incoming_from(ctx, pred, header, j),
                ) else {
                    return false;
                };
                let Mnemonic::Load(Load {
                    space: cs,
                    ptr: a,
                    size: csz,
                }) = *ctx.get_insn(carry_in).mnemonic()
                else {
                    return false;
                };
                cs == space && csz == size && affine_step(ctx, iv_in, a) == Some(step)
            });
            if !consistent {
                continue;
            }

            return Some(Pipelined {
                header,
                k,
                carry,
                iv: ptr,
                step,
                space,
                size,
            });
        }
    }
    None
}

/// Rewrite one pipelined carry: insert `load(iv − step)` at the header's start,
/// forward the carry's uses to it, drop the param, and record the disjointness
/// assumption that justifies the re-read.
fn apply(ctx: &mut Context, fid: FunctionId, p: &Pipelined) {
    let first = *ctx.block(p.header).instruction_ids().first().unwrap();

    // iv − step  (same width as the induction variable).
    let iv_ty = ctx.type_of(p.iv);
    let width = ctx.shared.types.size_of(iv_ty);
    let step_lit = ctx.get_const(p.step, width).id();
    let sub = InstructionRef::from_mnemonic_with_type(
        ctx,
        p.header.func,
        Mnemonic::Binop(Binary {
            op: Binop::Int(IntBinop::Sub),
            lhs: p.iv,
            rhs: step_lit,
        }),
        iv_ty,
    )
    .id;
    BasicBlock::from_id_mut(ctx, p.header).insert_insn_before(first, sub);

    // load(space, iv − step, size) — the previous iteration's byte, re-derived.
    let load_ty = ctx.shared.types.get_or_make_int(p.size);
    let prev = InstructionRef::from_mnemonic_with_type(
        ctx,
        p.header.func,
        Mnemonic::Load(Load {
            space: p.space,
            ptr: ValueId::Instruction(sub),
            size: p.size,
        }),
        load_ty,
    )
    .id;
    BasicBlock::from_id_mut(ctx, p.header).insert_insn_before(first, prev);

    ctx.replace_all_uses_with(ValueId::BlockParam(p.carry), ValueId::Instruction(prev));
    remove_params_from_block(ctx, p.header, &HashSet::from_iter([p.k]));

    // Record the move's soundness premise: the re-read is value-preserving only
    // because src ⊥ dst (the C copy contract).
    ctx.assume_true(Proposition::CopyBuffersDisjoint(fid));
}

/// De-pipeline every pipelined-carry loop across all functions. Returns whether
/// anything changed.
pub(crate) fn depipeline(ctx: &mut Context) -> bool {
    let fids: Vec<FunctionId> = ctx.function_ids();
    let mut changed = false;
    for fid in fids {
        let Some(root) = Function::from_id(ctx, fid).root().map(|b| b.id) else {
            continue;
        };
        // CFG edges are unchanged by param removal, so the dominator tree stays
        // valid across the per-function fixpoint.
        let dom = compute_dominators(&qcode::value::Function::from_id(ctx, fid), root);
        while let Some(p) = find_pipelined(ctx, fid, &dom) {
            apply(ctx, fid, &p);
            changed = true;
        }
    }
    changed
}

#[derive(Default)]
pub struct Depipeline;

impl Pass for Depipeline {
    const NAME: &'static str = "depipeline";
    fn description(&self) -> &'static str {
        "Remove a pipelined loop-carried delay register (assumes disjoint buffers)"
    }
    fn run(&self, ctx: &mut Context, _env: &PipelineEnv) -> Result<bool, String> {
        Ok(depipeline(ctx))
    }
}

crate::register_module_pass!(Depipeline);

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{builder::Builder, testing::TestContext, value::Value};

    /// Build the canonical pipelined strcpy loop and return `(fid, header)`.
    fn build_pipelined(tc: &mut TestContext) -> (FunctionId, BlockId) {
        let ram = tc.ctx.shared.default_space;
        let fid = Function::make(&mut tc.ctx, "strcpy".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000, fid);
        let header = tc.ctx.get_or_make_block(0x1010, fid);
        let exit = tc.ctx.get_or_make_block(0x1020, fid);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
            f.add_block(header);
            f.add_block(exit);
        }

        // entry: incoming src/dst pointers + an initial carried byte.
        let src = ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(4).id);
        let dst = ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(4).id);

        // header params: @EAX (src ptr), @ECX (dst ptr), @DL (carried byte).
        let eax = ValueId::BlockParam(
            BasicBlock::from_id_mut(&mut tc.ctx, header)
                .push_param(4)
                .id,
        );
        let ecx = ValueId::BlockParam(
            BasicBlock::from_id_mut(&mut tc.ctx, header)
                .push_param(4)
                .id,
        );
        let dl = ValueId::BlockParam(
            BasicBlock::from_id_mut(&mut tc.ctx, header)
                .push_param(1)
                .id,
        );
        // exit param (dst end), so the loop is not trivially private.
        let _exit_ecx =
            ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, exit).push_param(4).id);

        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let one = b.context_mut().get_const(1, 4).id();
            // Pipelined prologue: pre-load src[0] and pre-step the source pointer,
            // so the carry seed `load(src)` matches the in-loop re-read
            // `load((src + 1) − 1)` the de-pipeliner synthesizes. The recognizer
            // requires this consistency (see `leaves_inconsistent_prologue_seed_alone`).
            let p0 = b.push_load::<false>(src, 1, ram).id();
            let eax0 = b.push_add(src, one).id();
            b.push_branch_with_args(header, vec![eax0, dst, p0]);
        }
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, header));
            let one = b.context_mut().get_const(1, 4).id();
            b.push_store(dl, ecx, ram); // store(ram:1, @ECX <- @DL)
            let t = b.push_load::<false>(eax, 1, ram).id(); // %t = load(ram:1, @EAX)
            let ea = b.push_add(eax, one).id();
            let ec = b.push_add(ecx, one).id();
            // condition = %t (loop while nonzero); also feeds @DL on the back-edge.
            b.push_cbranch_with_args(t, header, vec![ea, ec, t], exit, vec![ec]);
        }
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, exit));
            let dummy = b.context_mut().get_const(0, 8).id();
            b.push_return(dummy);
        }
        (fid, header)
    }

    /// The pipelined `@DL` carry is removed and the disjointness assumption is
    /// recorded for the function.
    #[test]
    fn removes_pipelined_carry_and_records_assumption() {
        let mut tc = TestContext::new();
        let (fid, header) = build_pipelined(&mut tc);

        let before = BasicBlock::from_id(&tc.ctx, header).num_params();
        assert!(
            depipeline(&mut tc.ctx),
            "the pipelined carry must be removed"
        );
        let after = BasicBlock::from_id(&tc.ctx, header).num_params();
        assert_eq!(after, before - 1, "exactly the @DL param is dropped");

        // The re-read load (a second `load` in the header) was materialized.
        let loads = BasicBlock::from_id(&tc.ctx, header)
            .iter()
            .filter(|i| matches!(i.mnemonic(), Mnemonic::Load(_)))
            .count();
        assert_eq!(
            loads, 2,
            "the re-derived load(@EAX - 1) joins the original load"
        );

        // The disjointness assumption is recorded (Assumed).
        assert_eq!(
            tc.ctx
                .truth(Proposition::CopyBuffersDisjoint(fid))
                .map(|t| t.value),
            Some(true),
            "CopyBuffersDisjoint must be assumed for the de-pipelined function"
        );
    }

    /// A loop in the pipelined *shape* (store + carry `load(iv)` on the back-edge)
    /// but whose prologue seeds the carry with a constant — not the matching
    /// `load(src)` — is left untouched: de-pipelining would change the first stored
    /// byte. No assumption is recorded. This is the unsound case the prologue-seed
    /// consistency check rejects.
    #[test]
    fn leaves_inconsistent_prologue_seed_alone() {
        let mut tc = TestContext::new();
        let ram = tc.ctx.shared.default_space;
        let fid = Function::make(&mut tc.ctx, "notstrcpy".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x3000, fid);
        let header = tc.ctx.get_or_make_block(0x3010, fid);
        let exit = tc.ctx.get_or_make_block(0x3020, fid);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
            f.add_block(header);
            f.add_block(exit);
        }
        let src = ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(4).id);
        let dst = ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(4).id);
        let eax = ValueId::BlockParam(
            BasicBlock::from_id_mut(&mut tc.ctx, header)
                .push_param(4)
                .id,
        );
        let ecx = ValueId::BlockParam(
            BasicBlock::from_id_mut(&mut tc.ctx, header)
                .push_param(4)
                .id,
        );
        let dl = ValueId::BlockParam(
            BasicBlock::from_id_mut(&mut tc.ctx, header)
                .push_param(1)
                .id,
        );
        let _exit_ecx =
            ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, exit).push_param(4).id);
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            // Inconsistent prologue: a constant carried byte and an un-stepped src,
            // so the seed does not equal `load(src)` = `load((src) − 1 + 1)`.
            let init = b.context_mut().get_const(0x41, 1).id();
            b.push_branch_with_args(header, vec![src, dst, init]);
        }
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, header));
            let one = b.context_mut().get_const(1, 4).id();
            b.push_store(dl, ecx, ram);
            let t = b.push_load::<false>(eax, 1, ram).id();
            let ea = b.push_add(eax, one).id();
            let ec = b.push_add(ecx, one).id();
            b.push_cbranch_with_args(t, header, vec![ea, ec, t], exit, vec![ec]);
        }
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, exit));
            let dummy = b.context_mut().get_const(0, 8).id();
            b.push_return(dummy);
        }

        assert!(
            !depipeline(&mut tc.ctx),
            "a prologue seed that does not match load(iv − step) must not de-pipeline"
        );
        assert_eq!(
            tc.ctx.truth(Proposition::CopyBuffersDisjoint(fid)),
            None,
            "no disjointness assumption is recorded when the loop is left alone"
        );
    }

    /// A loop whose carried param is a genuine recurrence (not `load(iv)`) is not
    /// touched, and no assumption is recorded.
    #[test]
    fn leaves_non_pipelined_loop_alone() {
        let mut tc = TestContext::new();
        let ram = tc.ctx.shared.default_space;
        let fid = Function::make(&mut tc.ctx, "sum".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x2000, fid);
        let header = tc.ctx.get_or_make_block(0x2010, fid);
        let exit = tc.ctx.get_or_make_block(0x2020, fid);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
            f.add_block(header);
            f.add_block(exit);
        }
        let p = ValueId::BlockParam(
            BasicBlock::from_id_mut(&mut tc.ctx, header)
                .push_param(4)
                .id,
        );
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let z = b.context_mut().get_const(0, 4).id();
            b.push_branch_with_args(header, vec![z]);
        }
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, header));
            let one = b.context_mut().get_const(1, 4).id();
            let addr = b.context_mut().get_const(0x4000, 4).id();
            // A store (so the copy-shape gate passes) but the carry is `p + 1`,
            // not `load(iv)` — a real accumulator, not a delay register.
            b.push_store(p, addr, ram);
            let next = b.push_add(p, one).id();
            let cond = b.push_load::<false>(addr, 1, ram).id();
            b.push_cbranch_with_args(cond, header, vec![next], exit, vec![]);
        }
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, exit));
            let dummy = b.context_mut().get_const(0, 8).id();
            b.push_return(dummy);
        }

        assert!(
            !depipeline(&mut tc.ctx),
            "an accumulator loop is not de-pipelined"
        );
        assert_eq!(tc.ctx.truth(Proposition::CopyBuffersDisjoint(fid)), None);
    }
}
