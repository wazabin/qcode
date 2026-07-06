//! strlen recognizers: fold a bounded NUL-scan to `len(take_while(arr))`.
//!
//! Two layers over the same idea — a `while (s[i]) i++;` loop counts the leading
//! nonzero bytes, which is exactly `len(take_while(s))`:
//!
//! - **Layer 1** ([`try_match_strlen`]) matches the `at`-form snapshot scan that
//!   [`array_reads`](crate::mem::array_reads) produces from an argpromote
//!   read-only snapshot: the lane read is `at(@arr, i)` and the count escapes on
//!   the exit edge. It rewrites that count to `len(take_while(@arr))`.
//! - **Layer 2** ([`try_match_strlen_ptr`]) matches the general raw-`char*` scan
//!   (no snapshot): a +1 pointer induction whose length is the pointer difference
//!   `end - base`, rewritten to `len(take_while(@base))` over the unbounded string.
//!
//! Both leave the now-dead scan for later DCE. The total-map recognizer lives in
//! the sibling [`super::loop_to_map`] and accumulator scans in
//! [`super::loop_to_scan`].

use rustc_hash::FxHashSet as HashSet;

use qcode::{
    builder::Builder,
    context::Context,
    space::{Space, SpaceId, SpaceType},
    value::{
        BasicBlock, BlockId, Function, FunctionId, ValueId,
        insn::{
            Binop, Branch, CBranch, InstructionId, IntBinop, IntrinsicApp, IntrinsicId, Mnemonic,
            Unary, Unop,
        },
    },
};

use crate::loop_info::{delete_private_loop, incoming, is_increment, literal};
use crate::{Pass, PipelineEnv};

fn is_temp(ctx: &Context, s: SpaceId) -> bool {
    matches!(Space::from_id(ctx, s).ty, SpaceType::Temporary)
}

// ===========================================================================
// strlen recognizer — len(take_while(arr))
// ===========================================================================
//
// A bounded NUL-scan over a snapshot array — `while (arr[i]) i++;` capped by the
// snapshot bound `N` — counts the leading nonzero bytes. That count is exactly
// `len(take_while(arr))`: `take_while` truncates the array at its first zero, and
// `len` of the resulting list is its data-dependent length. The recognizer proves
// the loop is such a scan and rewrites the escaping count to `len(take_while(@arr))`
// — the projectable form — leaving the now-dead scan for DCE.
//
// Shape (carried-array form, produced by `array_reads` from an argpromote
// read-only snapshot): a read-only scan whose lane read is `at(@arr, i)` and whose
// exit is governed by the loaded byte rather than a static index bound:
//
//   <entry @arr:[i8;N] ...>  goto <head @i=0>;
//   <head @i>   %b = at(@arr, @i);                     // lane read
//               if %b != 0 goto <body> else goto <exit @i>;   // NUL test
//   <body>      goto <head @i=@i+1>;
//   <exit @count>  ... uses of @count = strlen ...
//
// The lane byte governs the loop exit (the take_while predicate), and the index
// carried out on the exit edge is the scan length. v1: byte lanes, snapshot-backed
// source (the array value `@arr`); the unbounded raw-`char*` case is Layer 2.

/// A recognized bounded NUL-scan whose escaping count is `len(take_while(arr))`.
struct StrlenMatch {
    /// The `Array` snapshot scanned (a root `[i8;N]` param).
    arr: ValueId,
    /// The exit-block parameter holding the scan length (its uses become `len`).
    count_param: ValueId,
    /// The loop header (carries the induction param, holds the `at` and NUL test).
    header_block: BlockId,
    /// The loop body (the per-iteration increment).
    body_block: BlockId,
    /// The exit block (where `len`/`take_while` are inserted; holds `count_param`).
    exit_block: BlockId,
    /// The header's sole out-of-loop predecessor (rerouted to the exit on delete).
    preheader: BlockId,
    /// Whether the loop is wholly private and can be deleted (every value the
    /// loop defines is used only inside it).
    deletable: bool,
}

/// Whether `cond` is true exactly when `elem` is **nonzero** (`Some(true)`, a
/// "continue while nonzero" guard), true exactly when `elem` is **zero**
/// (`Some(false)`), or not a zero-test of `elem` at all (`None`). Handles the bare
/// byte used as a predicate, `elem != 0` / `elem == 0` (either operand order), and
/// a `BoolNot` wrapper.
fn nonzero_polarity(ctx: &Context, cond: ValueId, elem: ValueId) -> Option<bool> {
    if cond == elem {
        return Some(true); // the raw byte as a bool: true ⟺ nonzero
    }
    let ValueId::Instruction(id) = cond else {
        return None;
    };
    match ctx.get_insn(id).mnemonic() {
        Mnemonic::Binop(b) => {
            let zero_test = (b.lhs == elem && literal(ctx, b.rhs) == Some(0))
                || (b.rhs == elem && literal(ctx, b.lhs) == Some(0));
            if !zero_test {
                return None;
            }
            match b.op {
                Binop::Int(IntBinop::NotEqual) => Some(true),
                Binop::Int(IntBinop::Equal) => Some(false),
                _ => None,
            }
        }
        Mnemonic::Unop(Unary {
            op: Unop::BoolNot,
            src,
        }) => nonzero_polarity(ctx, *src, elem).map(|p| !p),
        _ => None,
    }
}

/// Match a bounded NUL-scan in `fid`, or `None` for any other shape.
fn try_match_strlen(ctx: &Context, fid: FunctionId) -> Option<StrlenMatch> {
    let at_id = IntrinsicId::from_name("at")?;
    let insert_id = IntrinsicId::from_name("insert")?;
    let root_params: Vec<ValueId> = Function::from_id(ctx, fid)
        .root()?
        .params()
        .map(|p| p.id())
        .collect();
    let is_root = |v: ValueId| root_params.contains(&v);

    // The lane read is the sole `at(@arr, idx)` over a root `[i8;N]` param. More than
    // one `at` on that array (or any `insert` into it) means the loop is not a plain
    // read-only NUL scan, so bail.
    let mut lane: Option<(InstructionId, ValueId, ValueId)> = None; // (at id, arr, idx)
    for block in Function::from_id(ctx, fid).iter() {
        for insn in block.iter() {
            let Mnemonic::Intrinsic(IntrinsicApp { id, args }) = insn.mnemonic() else {
                continue;
            };
            if *id == insert_id && args.first().is_some_and(|&a| is_root(a)) {
                return None; // a write into a snapshot — not read-only
            }
            if *id != at_id || args.len() != 2 || !is_root(args[0]) {
                continue;
            }
            let byte_array = ctx
                .stored_type_of(args[0])
                .and_then(|t| ctx.types.array_of(t))
                .is_some_and(|(elem, _)| ctx.types.size_of(elem) == 1);
            if !byte_array {
                continue;
            }
            if lane.is_some() {
                return None; // more than one lane read — not the canonical scan
            }
            lane = Some((insn.id, args[0], args[1]));
        }
    }
    let (at_insn, arr, index) = lane?;
    let elem_val = ValueId::Instruction(at_insn);

    // The index is a header param initialised to 0 and stepped by +1 — so it counts
    // iterations from 0. (Unlike the map recognizer there is no static upper bound:
    // termination is the NUL test below, and the snapshot bound `N` caps it.)
    let ValueId::BlockParam(pid) = index else {
        return None;
    };
    let header = ctx.values.block_params[pid].parent?;
    // The lane read must live in the header: the NUL test that governs the loop
    // reads it there, and the count is the index at that test.
    if ctx.get_insn(at_insn).parent().map(|b| b.id) != Some(header) {
        return None;
    }
    let k = BasicBlock::from_id(ctx, header)
        .params()
        .position(|p| p.id() == index)?;
    // The index must start at 0 on *every* entry edge and step by +1 on the
    // back-edge: each non-increment (preheader) incoming has to be the literal 0,
    // or some entry could start the count off-zero and the rewrite would be wrong.
    let incoming = incoming(ctx, header, k);
    let inits: Vec<ValueId> = incoming
        .iter()
        .copied()
        .filter(|&v| !is_increment(ctx, v, index))
        .collect();
    if inits.is_empty()
        || !inits.iter().all(|&v| literal(ctx, v) == Some(0))
        || !incoming.iter().any(|&v| is_increment(ctx, v, index))
    {
        return None;
    }

    // Termination: the header ends in a CBranch governed by the loaded byte's
    // zero-test. Its continue edge re-enters the loop (the body); its other edge
    // leaves to the exit, carrying the index — that carried value is the scan
    // length. Keying on the *header*'s terminator (not any matching CBranch in the
    // function) ensures the NUL test is the loop's governing exit.
    {
        let Some(term) = BasicBlock::from_id(ctx, header).iter().last() else {
            return None;
        };
        let Mnemonic::CBranch(CBranch {
            condition,
            success_block,
            success_args,
            failure_block,
            failure_args,
        }) = term.mnemonic()
        else {
            return None;
        };
        let nonzero_continues = nonzero_polarity(ctx, *condition, elem_val)?;
        // The exit edge is the one taken when the byte is zero.
        let (exit_block, exit_args) = if nonzero_continues {
            (*failure_block, failure_args)
        } else {
            (*success_block, success_args)
        };
        // The exit edge must carry the index (the count) to an exit-block param.
        let kx = exit_args.iter().position(|&v| v == index)?;
        let count_param = BasicBlock::from_id(ctx, exit_block)
            .params()
            .nth(kx)
            .map(|p| p.id())?;

        // The body is the continue target; header/body/exit must be distinct so the
        // rewrite (and any deletion) can address them separately.
        let body_block = if nonzero_continues {
            *success_block
        } else {
            *failure_block
        };
        if header == body_block || header == exit_block || body_block == exit_block {
            return None;
        }

        // The body's terminator must be an unconditional branch back to the header:
        // the back-edge is the only way out of the body, so the header's NUL test is
        // the loop's *sole* data-dependent exit. Without this, a second `break`
        // (e.g. on another byte value) would make the count not the first-zero index.
        let back_ok = BasicBlock::from_id(ctx, body_block).iter().last().is_some_and(|t| {
            matches!(t.mnemonic(), Mnemonic::Branch(Branch { target, .. }) if *target == header)
        });
        if !back_ok {
            return None;
        }

        // The preheader is the header's sole out-of-loop predecessor: the reroute
        // source when the dead loop is deleted (there is no seed store to key on).
        let out_of_loop: Vec<BlockId> = BasicBlock::from_id(ctx, header)
            .predecessors()
            .map(|(_, p)| p)
            .filter(|&p| p != body_block)
            .collect();
        let [preheader] = out_of_loop[..] else {
            return None;
        };

        // Deletable iff wholly private (mirrors the map recognizer): the exit carries
        // only the count, and every value the loop defines is used only inside it.
        let loop_blocks = [header, body_block];
        let in_loop = |v: ValueId| {
            ctx.users(v).iter().all(|&u| {
                ctx.get_insn(u)
                    .parent()
                    .is_some_and(|b| loop_blocks.contains(&b.id))
            })
        };
        let exit_only_count = BasicBlock::from_id(ctx, exit_block)
            .params()
            .all(|p| p.id() == count_param);
        let deletable = exit_only_count
            && loop_blocks.iter().all(|&blk| {
                let b = BasicBlock::from_id(ctx, blk);
                b.params().all(|p| p.id() == index || in_loop(p.id()))
                    && b.iter().all(|i| in_loop(ValueId::Instruction(i.id)))
            });

        Some(StrlenMatch {
            arr,
            count_param,
            header_block: header,
            body_block,
            exit_block,
            preheader,
            deletable,
        })
    }
}

/// Rewrite a matched NUL-scan: replace the escaping count with
/// `len(take_while(@arr))`, then (when the scan is wholly private) strip the seed
/// and delete the dead loop.
fn apply_strlen(ctx: &mut Context, fid: FunctionId, m: &StrlenMatch) -> bool {
    // take_while(@arr) then len(...) of it, inserted at the top of the exit block.
    let tw_id = IntrinsicId::from_name("take_while").expect("take_while registered");
    let len_id = IntrinsicId::from_name("len").expect("len registered");
    let first = BasicBlock::from_id(ctx, m.exit_block)
        .iter()
        .next()
        .map(|i| i.id);
    let len_val = {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.exit_block));
        if let Some(at) = first {
            b.set_insert_point_before(at);
        }
        let tw = b.push_intrinsic(tw_id, vec![m.arr]).id();
        b.push_intrinsic(len_id, vec![tw]).id()
    };
    ctx.replace_all_uses_with(m.count_param, len_val);

    if m.deletable {
        // The count was the loop's only escape and is now forwarded to `len`, so the
        // whole scan is dead. Order matters:
        //   1. drop the exit's count param (rewrites the header cbranch's exit-edge
        //      args while the header still exists),
        //   2. reroute the preheader straight to the (now param-less) exit,
        //   3. delete the dead loop blocks. (There is no seed store to strip — the
        //      `at`-form scan reads the root array param directly.)
        let kx = BasicBlock::from_id(ctx, m.exit_block)
            .params()
            .position(|p| p.id() == m.count_param);
        if let Some(kx) = kx {
            crate::dce::remove_params_from_block(ctx, m.exit_block, &HashSet::from_iter([kx]));
        }
        // Reroute the preheader straight to the (now param-less) exit and delete the
        // dead loop blocks (body before header, as they were emitted).
        delete_private_loop(
            ctx,
            fid,
            m.preheader,
            &[m.body_block, m.header_block],
            m.exit_block,
            Vec::new(),
        );
    }
    // Otherwise the loop also feeds outside consumers, so it keeps running; only the
    // count was forwarded to `len` above.
    true
}

/// Recognize bounded NUL-scan loops across all pure functions, rewriting each
/// escaping count to `len(take_while(arr))`. Returns `true` if anything changed.
pub(crate) fn recognize_strlens(ctx: &mut Context) -> bool {
    let fids: Vec<FunctionId> = ctx.function_ids();
    let mut changed = false;
    for fid in fids {
        if !Function::from_id(ctx, fid).is_pure() {
            continue;
        }
        if let Some(m) = try_match_strlen(ctx, fid) {
            changed |= apply_strlen(ctx, fid, &m);
        }
    }
    changed
}

// ===========================================================================
// strlen recognizer (Layer 2) — raw pointer scan, no snapshot
// ===========================================================================
//
// The general `strlen` is a raw `char*` scan with no bounded snapshot (argpromote
// only snapshots *bounded* regions, and a NUL scan's index is unbounded). After
// `argpromote_registers` the pointer is a param; the loop steps it and stops at the
// first zero byte, and the length is the pointer *difference* `end - base`:
//
//   <entry @s0>   goto <head @s=@s0>;
//   <head @s>     %b = load(ram:1, @s);
//                 if %b != 0 goto <body> else goto <exit @s>;     // NUL test
//   <body>        goto <head @s=@s+1>;
//   <exit @end>   %len = @end - @s0;     ... uses of %len = strlen ...
//
// The source for `take_while` is the *pointer* `@s0` (an unbounded string view, see
// [`TypeManager::get_or_make_unbounded_list`]), and `end - base` is exactly
// `len(take_while(@s0))`. Unlike Layer 1 this needs no purity gate (the raw loads
// make the function impure) and no snapshot; the soundness is purely structural —
// a +1 pointer induction whose sole exit is the NUL test, with the length read off
// as the end-minus-base difference. The dead scan is left for later DCE.
//
// [`TypeManager::get_or_make_unbounded_list`]: qcode::types::TypeManager::get_or_make_unbounded_list

/// A recognized raw-pointer NUL-scan whose `end - base` difference is `strlen`.
struct StrlenPtrMatch {
    /// The string base pointer `@s0` (the `take_while` source).
    base: ValueId,
    /// The `end - base` pointer-difference instruction (becomes `len(take_while)`).
    diff_id: InstructionId,
    /// The block holding that difference (where `len`/`take_while` are inserted).
    diff_block: BlockId,
}

/// Match a raw-pointer NUL-scan in `fid`, or `None` for any other shape.
fn try_match_strlen_ptr(ctx: &Context, fid: FunctionId) -> Option<StrlenPtrMatch> {
    for block in Function::from_id(ctx, fid).iter() {
        let header = block.id;
        let params: Vec<ValueId> = BasicBlock::from_id(ctx, header)
            .params()
            .map(|p| p.id())
            .collect();
        for (k, &s) in params.iter().enumerate() {
            // Induction pointer: stepped by +1 on the back-edge, initialised to a
            // single base pointer `@s0`. Requiring *exactly one* non-increment
            // incoming pins the base unambiguously — with two entry pointers, the
            // `end - base` length would only be `strlen` on the matching entry.
            let incoming = incoming(ctx, header, k);
            if !incoming.iter().any(|&v| is_increment(ctx, v, s)) {
                continue;
            }
            let bases: Vec<ValueId> = incoming
                .iter()
                .copied()
                .filter(|&v| !is_increment(ctx, v, s))
                .collect();
            let [base] = bases[..] else {
                continue;
            };

            // A byte load at the pointer, in real memory (not a shadow snapshot —
            // that is Layer 1's `is_temp` region).
            let load = BasicBlock::from_id(ctx, header)
                .iter()
                .find_map(|i| match i.mnemonic() {
                    Mnemonic::Load(l) if l.ptr == s && l.size == 1 && !is_temp(ctx, l.space) => {
                        Some(i.id)
                    }
                    _ => None,
                });
            let Some(load_id) = load else {
                continue;
            };
            let elem_val = ValueId::Instruction(load_id);

            // The header's terminator is the NUL test; its continue edge re-enters
            // the loop and its other edge leaves, carrying the end pointer.
            let Some(term) = BasicBlock::from_id(ctx, header).iter().last() else {
                continue;
            };
            let Mnemonic::CBranch(CBranch {
                condition,
                success_block,
                success_args,
                failure_block,
                failure_args,
            }) = term.mnemonic()
            else {
                continue;
            };
            let Some(nonzero_continues) = nonzero_polarity(ctx, *condition, elem_val) else {
                continue;
            };
            let (exit_block, exit_args, body_block) = if nonzero_continues {
                (*failure_block, failure_args, *success_block)
            } else {
                (*success_block, success_args, *failure_block)
            };
            // The exit edge must carry the induction pointer (the end pointer).
            if !exit_args.iter().any(|&v| v == s) {
                continue;
            }
            if header == body_block || header == exit_block {
                continue;
            }
            // The body is an unconditional back-edge: the NUL test is the loop's
            // sole exit (else `end - base` is not the first-zero offset).
            let back_ok = BasicBlock::from_id(ctx, body_block).iter().last().is_some_and(|t| {
                matches!(t.mnemonic(), Mnemonic::Branch(Branch { target, .. }) if *target == header)
            });
            if !back_ok {
                continue;
            }

            // The escaping length is the pointer difference `end - base`, where the
            // end pointer is the exit-block param fed the induction pointer. Matched
            // by `lhs - base` with `lhs` an exit param carrying `s`.
            let kx = exit_args.iter().position(|&v| v == s)?;
            let end_param = BasicBlock::from_id(ctx, exit_block)
                .params()
                .nth(kx)
                .map(|p| p.id())?;
            for b2 in Function::from_id(ctx, fid).iter() {
                let bid = b2.id;
                for i in b2.iter() {
                    if let Mnemonic::Binop(bin) = i.mnemonic()
                        && matches!(bin.op, Binop::Int(IntBinop::Sub))
                        && bin.lhs == end_param
                        && bin.rhs == base
                    {
                        return Some(StrlenPtrMatch {
                            base,
                            diff_id: i.id,
                            diff_block: bid,
                        });
                    }
                }
            }
        }
    }
    None
}

/// Rewrite a matched raw-pointer scan: replace its `end - base` difference with
/// `len(take_while(@base))` over the unbounded string at `@base`.
fn apply_strlen_ptr(ctx: &mut Context, m: &StrlenPtrMatch) -> bool {
    let tw_id = IntrinsicId::from_name("take_while").expect("take_while registered");
    let len_id = IntrinsicId::from_name("len").expect("len registered");
    let len_val = {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.diff_block));
        b.set_insert_point_before(m.diff_id);
        let tw = b.push_intrinsic(tw_id, vec![m.base]).id();
        b.push_intrinsic(len_id, vec![tw]).id()
    };
    ctx.replace_all_uses_with(ValueId::Instruction(m.diff_id), len_val);
    ctx.remove_instruction(m.diff_id);
    // The scan now produces nothing used outside it; later DCE removes the dead loop.
    true
}

/// Recognize raw-pointer NUL-scan loops across all functions, rewriting each
/// `end - base` length to `len(take_while(base))`. Returns `true` if anything changed.
pub(crate) fn recognize_strlens_ptr(ctx: &mut Context) -> bool {
    let fids: Vec<FunctionId> = ctx.function_ids();
    let mut changed = false;
    for fid in fids {
        if let Some(m) = try_match_strlen_ptr(ctx, fid) {
            changed |= apply_strlen_ptr(ctx, &m);
        }
    }
    changed
}

#[derive(Default)]
pub struct Strlen;

impl Pass for Strlen {
    const NAME: &'static str = "strlen";
    fn description(&self) -> &'static str {
        "Rewrite a bounded NUL-scan as len(take_while(arr)) (snapshot and raw-pointer forms)"
    }
    fn run(&self, ctx: &mut Context, _env: &PipelineEnv) -> Result<bool, String> {
        // Layer 1 (at-form snapshot) then Layer 2 (raw char*); mutually exclusive
        // on any one function.
        let mut changed = recognize_strlens(ctx);
        changed |= recognize_strlens_ptr(ctx);
        Ok(changed)
    }
}

crate::register_module_pass!(Strlen);

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{
        builder::Builder,
        testing::TestContext,
        value::{Value, insn::Mnemonic},
    };

    use crate::test_util::run_function_pass;

    // ===== recognizer (two-buffer) =========================================

    /// Build a counted, byte-lane total-map loop `dst[i] = src[i]` over `N`
    /// elements in the canonical post-argpromote shape:
    ///   entry: seed `*[shadow]:N base_src = arr`; goto head(i=0)
    ///   head:  if i < N goto body else goto exit
    ///   body:  e = *[shadow]:1 (base_src+i); *[shadow]:1 (base_dst+i) = e; i+1
    ///   exit:  wv = *[shadow]:N base_dst; store(ram, &out = wv); return
    /// When `distinct_dst` is false, `base_dst == base_src` (the in-place case).
    /// When `stray` is true, an extra shadow load on the destination base is
    /// injected so the region bookkeeping should reject the loop.
    ///
    /// Returns `(fid, exit_block, arr_value)`.
    fn build_copy_loop(
        tc: &mut TestContext,
        distinct_dst: bool,
        stray: bool,
    ) -> (FunctionId, BlockId, ValueId) {
        const N: usize = 4;
        let i8 = tc.ctx.types.get_or_make_int(1);
        let arr_ty = tc.ctx.types.get_or_make_array(i8, N);
        let shadow = tc.ctx.make_temp_space();
        let ram = tc.ctx.default_space;

        let fid = Function::make(&mut tc.ctx, "copy".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let header = tc.ctx.get_or_make_block(0x1010);
        let body = tc.ctx.get_or_make_block(0x1020);
        let exit = tc.ctx.get_or_make_block(0x1030);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
            f.add_block(header);
            f.add_block(body);
            f.add_block(exit);
        }

        // Root params: arr:[i8;N], base_src, base_dst (the call interface).
        let arr_pid = BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(N).id;
        tc.ctx.values.block_params[arr_pid].type_id = arr_ty;
        let arr = ValueId::BlockParam(arr_pid);
        let base_src =
            ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(8).id);
        let base_dst = if distinct_dst {
            ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(8).id)
        } else {
            base_src
        };

        // Header induction param.
        let i = ValueId::BlockParam(
            BasicBlock::from_id_mut(&mut tc.ctx, header)
                .push_param(8)
                .id,
        );

        // entry: seed store + preheader branch.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let zero = b.context_mut().get_const(0, 8).id();
            b.push_store(arr, base_src, shadow); // *[shadow]:N base_src = arr
            b.push_branch_with_args(header, vec![zero]);
        }
        // header: counted guard.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, header));
            let n = b.context_mut().get_const(N as u64, 8).id();
            let cond = b.push_lt(i, n).id();
            b.push_cbranch_with_args(cond, body, vec![], exit, vec![]);
        }
        // body: read src lane, write dst lane, increment.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, body));
            let one = b.context_mut().get_const(1, 8).id();
            let addr_src = b.push_add(base_src, i).id();
            let elem = b.push_load::<false>(addr_src, 1, shadow).id();
            let addr_dst = b.push_add(base_dst, i).id();
            b.push_store(elem, addr_dst, shadow); // *[shadow]:1 (base_dst+i) = elem
            let inc = b.push_add(i, one).id();
            b.push_branch_with_args(header, vec![inc]);
        }
        // exit: wide reload (write-set) + an external use + return.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, exit));
            if stray {
                // An extra shadow access on the destination region: should defeat
                // the exactness check.
                b.push_load::<false>(base_dst, 1, shadow);
            }
            let wv = b.push_load::<false>(base_dst, N, shadow).id();
            let out = b.context_mut().get_const(0x9000, 8).id();
            b.push_store(wv, out, ram); // external consumer of the write-set
            let dummy = b.context_mut().get_const(0, 8).id();
            b.push_return(dummy);
        }

        Function::from_id_mut(&mut tc.ctx, fid).set_is_pure(true);
        (fid, exit, arr)
    }
    // ===== recognizer (strlen) =============================================

    /// Build a bounded NUL-scan `while (arr[i]) i++;` over an `N`-byte snapshot in
    /// the canonical post-argpromote shape, with the scan length escaping via the
    /// exit block param (consumed by an external store):
    ///   entry: seed `*[shadow]:N base = arr`; goto head(i=0)
    ///   head:  %b = *[shadow]:1 (base+i); if %b != 0 goto body else goto exit(i)
    ///   body:  goto head(i+1)
    ///   exit:  store(ram, &out = count); return
    /// When `extra_break` is true the body ends in a *second* data-dependent exit
    /// (a `break` on another condition) instead of a clean back-edge, so the NUL
    /// test is no longer the loop's sole exit and recognition must decline.
    ///
    /// Returns `(fid, exit_block, arr_value)`.
    fn build_strlen_loop(
        tc: &mut TestContext,
        extra_break: bool,
    ) -> (FunctionId, BlockId, ValueId) {
        const N: usize = 8;
        let i8 = tc.ctx.types.get_or_make_int(1);
        let arr_ty = tc.ctx.types.get_or_make_array(i8, N);
        let shadow = tc.ctx.make_temp_space();
        let ram = tc.ctx.default_space;

        let fid = Function::make(&mut tc.ctx, "slen".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let header = tc.ctx.get_or_make_block(0x1010);
        let body = tc.ctx.get_or_make_block(0x1020);
        let exit = tc.ctx.get_or_make_block(0x1030);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
            f.add_block(header);
            f.add_block(body);
            f.add_block(exit);
        }

        let arr_pid = BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(N).id;
        tc.ctx.values.block_params[arr_pid].type_id = arr_ty;
        let arr = ValueId::BlockParam(arr_pid);
        let base =
            ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(8).id);

        let i = ValueId::BlockParam(
            BasicBlock::from_id_mut(&mut tc.ctx, header)
                .push_param(8)
                .id,
        );
        let count =
            ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, exit).push_param(8).id);

        // entry: seed store + preheader branch (i = 0).
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let zero = b.context_mut().get_const(0, 8).id();
            b.push_store(arr, base, shadow);
            b.push_branch_with_args(header, vec![zero]);
        }
        // head: load lane, NUL test; continue while nonzero, else exit carrying i.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, header));
            let addr = b.push_add(base, i).id();
            let byte = b.push_load::<false>(addr, 1, shadow).id();
            let zero1 = b.context_mut().get_const(0, 1).id();
            let nz = b.push_ne(byte, zero1).id();
            b.push_cbranch_with_args(nz, body, vec![], exit, vec![i]);
        }
        // body: increment, then either a clean back-edge or (extra_break) a second
        // exit on `i < 100` — a break that defeats the sole-exit requirement.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, body));
            let one = b.context_mut().get_const(1, 8).id();
            let inc = b.push_add(i, one).id();
            if extra_break {
                let hundred = b.context_mut().get_const(100, 8).id();
                let lt = b.push_lt(inc, hundred).id();
                b.push_cbranch_with_args(lt, header, vec![inc], exit, vec![inc]);
            } else {
                b.push_branch_with_args(header, vec![inc]);
            }
        }
        // exit: external consumer of the length + return.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, exit));
            let out = b.context_mut().get_const(0x9000, 8).id();
            b.push_store(count, out, ram);
            let dummy = b.context_mut().get_const(0, 8).id();
            b.push_return(dummy);
        }

        Function::from_id_mut(&mut tc.ctx, fid).set_is_pure(true);
        (fid, exit, arr)
    }

    /// `len(take_while(@arr))` over `@arr`, if such a chain appears in `block`.
    fn len_take_while_src(ctx: &Context, block: BlockId) -> Option<ValueId> {
        BasicBlock::from_id(ctx, block).iter().find_map(|i| {
            let Mnemonic::Intrinsic(len) = i.mnemonic() else {
                return None;
            };
            if len.id.name() != "len" {
                return None;
            }
            let ValueId::Instruction(tw_id) = *len.args.first()? else {
                return None;
            };
            match ctx.get_insn(tw_id).mnemonic() {
                Mnemonic::Intrinsic(tw) if tw.id.name() == "take_while" => tw.args.first().copied(),
                _ => None,
            }
        })
    }

    /// A bounded NUL-scan over an argpromote snapshot: `array_reads` first rewrites
    /// the shadow lane loads to `at(@arr, i)`, then the strlen recognizer folds the
    /// escaping length to `len(take_while(@arr))` over the source array.
    #[test]
    fn array_reads_then_strlen_folds_nul_scan() {
        let mut tc = TestContext::new();
        let (fid, exit, arr) = build_strlen_loop(&mut tc, /*extra_break*/ false);

        assert!(
            run_function_pass::<crate::mem::array_reads::ArrayReads>(&mut tc.ctx, fid).unwrap(),
            "array_reads promotes the read-only shadow snapshot to at(@arr, i)"
        );
        assert!(
            recognize_strlens(&mut tc.ctx),
            "bounded NUL-scan must recognize on the at-form"
        );
        assert_eq!(
            len_take_while_src(&tc.ctx, exit),
            Some(arr),
            "the escaping count becomes len(take_while(@arr))"
        );
        assert!(Function::from_id(&tc.ctx, fid).is_pure());
        // The private scan is deleted: only entry + exit remain.
        assert_eq!(
            Function::from_id(&tc.ctx, fid).iter().count(),
            2,
            "the dead scan loop is removed"
        );
    }

    /// A loop with a second data-dependent exit (a `break` besides the NUL test) is
    /// not a clean `take_while`: its count is the first of *either* terminator, not
    /// the first zero, so recognition must decline even after `array_reads`.
    #[test]
    fn second_break_is_not_a_strlen() {
        let mut tc = TestContext::new();
        let (fid, exit, _arr) = build_strlen_loop(&mut tc, /*extra_break*/ true);

        run_function_pass::<crate::mem::array_reads::ArrayReads>(&mut tc.ctx, fid).unwrap();
        assert!(
            !recognize_strlens(&mut tc.ctx),
            "a loop with a second exit is not a NUL-scan strlen"
        );
        assert_eq!(
            len_take_while_src(&tc.ctx, exit),
            None,
            "no len/take_while inserted"
        );
    }

    /// A copy loop (which *stores*) is not a NUL-scan strlen: `array_reads` refuses
    /// the region (a non-seed store), and the strlen recognizer finds no `at` scan.
    #[test]
    fn copy_loop_is_not_a_strlen() {
        let mut tc = TestContext::new();
        let (fid, exit, _arr) =
            build_copy_loop(&mut tc, /*distinct_dst*/ true, /*stray*/ false);

        assert!(
            !run_function_pass::<crate::mem::array_reads::ArrayReads>(&mut tc.ctx, fid).unwrap(),
            "array_reads declines a region with a store"
        );
        assert!(
            !recognize_strlens(&mut tc.ctx),
            "a loop that stores is not a read-only NUL-scan"
        );
        assert_eq!(
            len_take_while_src(&tc.ctx, exit),
            None,
            "no len/take_while inserted"
        );
    }

    // ===== recognizer (strlen, raw pointer / Layer 2) ======================

    /// Build a raw `char*` NUL-scan with the length as a pointer difference, the
    /// general post-`argpromote_registers` shape (no snapshot, raw RAM loads):
    ///   entry: goto head(s = s0)
    ///   head:  %b = load(ram:1, s); if %b != 0 goto body else goto exit(s)
    ///   body:  goto head(s + 1)
    ///   exit:  %len = end - s0; store(ram, &out = %len); return
    /// When `with_diff` is false the exit consumes the end pointer directly instead
    /// of `end - s0`, so there is no length difference to rewrite and recognition
    /// must decline.
    ///
    /// Returns `(fid, exit_block, s0_base)`.
    fn build_strlen_ptr_loop(
        tc: &mut TestContext,
        with_diff: bool,
    ) -> (FunctionId, BlockId, ValueId) {
        let ram = tc.ctx.default_space;
        let fid = Function::make(&mut tc.ctx, "strlen".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x2000);
        let header = tc.ctx.get_or_make_block(0x2010);
        let body = tc.ctx.get_or_make_block(0x2020);
        let exit = tc.ctx.get_or_make_block(0x2030);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
            f.add_block(header);
            f.add_block(body);
            f.add_block(exit);
        }

        // The base string pointer `@s0` (a root param).
        let s0 = ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(8).id);
        // Header induction pointer and exit end-pointer params.
        let s = ValueId::BlockParam(
            BasicBlock::from_id_mut(&mut tc.ctx, header)
                .push_param(8)
                .id,
        );
        let end = ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, exit).push_param(8).id);

        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_branch_with_args(header, vec![s0]);
        }
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, header));
            let byte = b.push_load::<false>(s, 1, ram).id();
            let zero = b.context_mut().get_const(0, 1).id();
            let nz = b.push_ne(byte, zero).id();
            b.push_cbranch_with_args(nz, body, vec![], exit, vec![s]);
        }
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, body));
            let one = b.context_mut().get_const(1, 8).id();
            let s1 = b.push_add(s, one).id();
            b.push_branch_with_args(header, vec![s1]);
        }
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, exit));
            // strlen = end - base; or (no diff) the end pointer is consumed directly.
            let escaping = if with_diff {
                b.push_sub(end, s0).id()
            } else {
                end
            };
            let out = b.context_mut().get_const(0x9000, 8).id();
            b.push_store(escaping, out, ram);
            let dummy = b.context_mut().get_const(0, 8).id();
            b.push_return(dummy);
        }
        (fid, exit, s0)
    }

    /// A raw `char*` NUL-scan has its `end - base` length rewritten to
    /// `len(take_while(@base))` over the unbounded string at the base pointer.
    #[test]
    fn raw_pointer_nul_scan_recognized_as_len_take_while() {
        let mut tc = TestContext::new();
        let (_fid, exit, s0) = build_strlen_ptr_loop(&mut tc, /*with_diff*/ true);

        assert!(
            recognize_strlens_ptr(&mut tc.ctx),
            "raw-pointer NUL-scan must recognize"
        );
        assert_eq!(
            len_take_while_src(&tc.ctx, exit),
            Some(s0),
            "the pointer-difference length becomes len(take_while(@base))"
        );
        // The take_while source is the bare pointer, so its result is an *unbounded*
        // list (no static footprint).
        let tw_ty = BasicBlock::from_id(&tc.ctx, exit)
            .iter()
            .find_map(|i| match i.mnemonic() {
                Mnemonic::Intrinsic(intr) if intr.id.name() == "take_while" => Some(i.type_id()),
                _ => None,
            });
        let tw_ty = tw_ty.expect("a take_while was inserted");
        assert_eq!(
            tc.ctx.types.list_of(tw_ty).map(|(_, b)| b),
            Some(None),
            "unbounded list"
        );
    }

    /// Without the `end - base` difference there is no length expression to rewrite,
    /// so the scan is left untouched (no spurious `take_while`/`len` is invented).
    #[test]
    fn raw_pointer_scan_without_difference_declined() {
        let mut tc = TestContext::new();
        let (_fid, exit, _s0) = build_strlen_ptr_loop(&mut tc, /*with_diff*/ false);

        assert!(
            !recognize_strlens_ptr(&mut tc.ctx),
            "no end-base difference means nothing to rewrite"
        );
        assert_eq!(
            len_take_while_src(&tc.ctx, exit),
            None,
            "no len/take_while inserted"
        );
    }
}
