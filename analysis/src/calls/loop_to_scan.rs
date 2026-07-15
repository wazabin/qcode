//! Loop-to-scan: fold the value-carried `insert`/`at` fill loop that
//! [`array_promote`](crate::mem::array_promote) produces into a single `scanl`.
//!
//! `array_promote` rewrites a memory-carried array-fill loop into a carried
//! array threaded through the loop with `insert`/`at`, seeded at lane 0 and
//! stored wide at the exit:
//!
//! ```text
//!   <entry @seed @base>
//!       %e0 = trunc(i32, @seed);
//!       %a0 = insert(zero:[i32;N], 0, %e0);
//!       goto <head @i=1 @buf=@base @arr=%a0>;
//!   <head @i @buf @arr>  if @i == N goto <exit @arr> else goto <body @j @b @arr>;
//!   <body @j @b @arr>
//!       %prev = at(@arr, @j-1);           // previous element = previous accumulator
//!       %next = f(%prev, @j);             // pure body of (acc, index)
//!       %arr' = insert(@arr, @j, %next);
//!       goto <head @i=@j+1 @buf=@b @arr=%arr'>;
//!   <exit @arr>  store(ram, @base <- @arr); return …;
//! ```
//!
//! The loop computes `arr[0] = seed`, `arr[j] = f(arr[j-1], j)` for `j ∈ [1, N)`.
//! That is exactly a left-scan: with `acc_0 = seed` and `acc_{k+1} = f(acc_k, k+1)`,
//! the `N-1` produced elements are `arr[1..N]`, and `arr[0]` is the seed itself.
//! So the whole fill folds to
//!
//! ```text
//!   %scan = scanl @f seed iota(N-1);              // arr[1..N]
//!   %full = concat(singleton(seed), %scan);       // arr[0..N]
//!   store(ram, @base <- %full);
//! ```
//!
//! where `@f` is the outlined pure body of `(accumulator, index)`. The residual
//! `insert`/`at` loop is left in place (its array is now unused) for `dce` to
//! remove. The recognizer trusts `array_promote`'s coverage proof — the array's
//! full length `N` comes straight from the exit store's element type.

use qcode::value::{
    BlockId, FunctionId, QCodeView, ValueId,
    insn::{InstructionId, IntrinsicApp, IntrinsicId, Mnemonic},
};

use super::carried_array::{classify_body_reads, exit_view, find_carried_array};
use super::outline::{ScanElem, outline_scan_body};
use crate::loop_info::{
    cbranch_exit, delete_private_loop, incoming, is_increment, is_loop_private, literal,
    param_parent, param_pos, value_defined_in,
};
use crate::pipeline::{ContextView, FunctionBody, Minted, Outcome};
use crate::{FunctionPass, register_function_pass};

#[derive(Default)]
pub struct LoopToScan;

/// A recognized `insert`/`at` fill loop (see the module docs).
struct ScanMatch {
    /// The loop-exit block holding the wide store.
    exit: BlockId,
    /// The loop header (guard block) and body; `header == body` in the rotated
    /// (do-while) shape. Used to delete the residual loop once it is private.
    header: BlockId,
    body: BlockId,
    /// The loop-carried array as seen by the exit block (the store's source).
    /// Every other exit use of it (e.g. `at(arr, k)` from `array_promote`'s
    /// exit-load rewrite) is redirected to the folded array so the residual
    /// loop actually dies.
    arr_exit: ValueId,
    /// The lane-0 seed value (`acc_0`, also the first output element).
    seed_val: ValueId,
    /// The body's per-lane result (`%next`), the scan body's return value.
    stored_val: ValueId,
    /// The `at(arr, index-1)` result — the previous element, i.e. the accumulator.
    prev_val: ValueId,
    /// The body induction parameter used in the body expression.
    index: ValueId,
    /// The loop index at the first body iteration (`iota` element 0 maps here).
    index_start: i64,
    /// The full array length `N`.
    count: usize,
    /// The array element type.
    elem_ty: qcode::types::TypeId,
    /// Set when the body reads the region's *original* element at its own lane
    /// (`at(arr, index)` on the single carried array) — the loop is a scan over the
    /// original array rather than a pure generation over `iota`. Holds
    /// `(element-read value, original-array value to slice)`, where the array is the
    /// seed insert's base operand (the preheader wide `Load`); the scan ranges over
    /// `l0[1..]`.
    elem: Option<(ValueId, ValueId)>,
}

/// Recognize the `insert`/`at` fill loop in `fid`.
fn try_match<'a, 'str: 'a>(host: impl QCodeView<'a, 'str>, fid: FunctionId) -> Option<ScanMatch> {
    // Anchor on the shared carried-array matcher, then narrow to the scan shape:
    // a lane-0 accumulator seed insert must exist (`ca.seed`).
    let ca = find_carried_array(host, fid)?;
    // The shared matcher also accepts a *header-carried* index (`ca.index` a
    // param of `ca.header`, not of `ca.body`). Scan v1 is audited for the
    // body-copied shape only — gate the other out explicitly so the relaxed
    // matcher can never reach `apply` unaudited.
    // TODO(loop-fold-header-index): fold header-carried scans by moving this
    // recognizer's induction walk onto `NaturalLoop::unit_induction`, mirroring
    // `loop_to_map::try_match`.
    if param_parent(host, ca.index) != Some(ca.body) {
        return None;
    }
    let (seed_val, seed_arr0) = ca.seed?;
    let header = ca.header;
    let body = ca.body;
    let index = ca.index;
    let elem_ty = ca.elem_ty;
    let count = ca.count;
    let stored_val = ca.stored_val;

    // The loop body carries the array back to the header; the guard block's other
    // successor is the loop exit. `header`'s terminator is the `CBranch` guard
    // (rotated: it lives on the body/header itself; split: on the distinct guard).
    // The exit is the guard successor that is not itself a header predecessor.
    let (exit, _stay) = cbranch_exit(host, header)?;

    // The array value as the exit block sees it: an exit pass-through param that
    // copies `arr_h` on the header→exit edge, or `arr_h` itself when gvn coalesced
    // that trivial pass-through away (then the exit reads the header param directly).
    let arr_src = exit_view(host, &ca, exit);

    // The body's `at(arr_b, ·)` reads on the single carried array: the carry
    // `at(arr_b, index-1)` (always present) and, for an original-array scan, the
    // own-lane original read `at(arr_b, index)`.
    let reads = classify_body_reads(host, &ca)?;
    let prev_id = reads.prev?;
    let prev_val = ValueId::Instruction(prev_id);
    let elem_read = reads.own;

    // An own-lane original read turns the fold into a scan over the original array
    // `l0[1..]`. The original array is the seed insert's base operand `seed_arr0`,
    // defined in the preheader (so it dominates the exit rewrite site). It must be a
    // wide `Load` of the region — a `splat` base would mean literal zero, not
    // `l0[i]`. v1 array-input support is split-shape only.
    let elem = match elem_read {
        Some(e_read) => {
            if body == header {
                return None; // rotated array-input not supported in v1
            }
            let ValueId::Instruction(a0) = seed_arr0 else {
                return None;
            };
            if !matches!(host.insn_ref(a0).mnemonic(), Mnemonic::Load(_)) {
                return None;
            }
            Some((e_read, seed_arr0))
        }
        None => None,
    };

    // Induction: the values feeding `index` start at a single literal `s` and step
    // by one on the back-edge. In the rotated shape (`body == header`) the index
    // param's own two incomings are the preheader init and the back-edge
    // increment; in the split shape `index` copies a header param, whose incomings
    // carry the init and increment.
    let k_index = param_pos(host, body, index)?;
    let feeds = if body == header {
        incoming(host, body, k_index)
    } else {
        let [hp] = incoming(host, body, k_index)[..] else {
            return None;
        };
        if param_parent(host, hp) != Some(header) {
            return None;
        }
        let k_hp = param_pos(host, header, hp)?;
        incoming(host, header, k_hp)
    };
    if !feeds.iter().any(|&v| is_increment(host, v, index)) {
        return None;
    }
    let inits: Vec<i64> = feeds
        .iter()
        .filter(|&&v| !is_increment(host, v, index))
        .filter_map(|&v| literal(host, v).map(|x| x as i64))
        .collect();
    let [index_start] = inits[..] else {
        return None;
    };

    Some(ScanMatch {
        exit,
        header,
        body,
        arr_exit: arr_src,
        seed_val,
        stored_val,
        prev_val,
        index,
        index_start,
        count,
        elem_ty,
        elem,
    })
}

/// Rewrite a matched fill loop: outline the `(acc, index)` body and replace the
/// wide exit store with `store(ram, base <- concat(singleton(seed), scanl @body
/// seed iota(N-1)))`. The residual loop is left for `dce`.
fn apply<'str>(
    mv: ContextView<'_, 'str>,
    body: &mut FunctionBody<'str>,
    next_minted: &mut u32,
    minted: &mut Vec<Minted<'str>>,
    m: &ScanMatch,
) -> bool {
    let fid = body.id();
    let (index_ty, i64_ty, name) = {
        let host = mv.body_view(body);
        (
            host.type_of(m.index),
            host.shared().types.get_or_make_int(8),
            format!("{}_scan_body", host.function_ref(fid).name()),
        )
    };
    let n1 = m.count - 1;

    let iota_id = IntrinsicId::from_name("iota").expect("iota registered");
    let singleton_id = IntrinsicId::from_name("singleton").expect("singleton registered");
    let concat_id = IntrinsicId::from_name("concat").expect("concat registered");

    // Outline the `(acc, ·)` body before any rewrite (a non-closed body leaves
    // the loop untouched), keeping the src shape.
    enum Src {
        /// `l0[1..]` slice of the original array (data-input scan).
        Slice { l0_exit: ValueId, esz: usize },
        /// A fresh `iota(N-1)` (pure generation).
        Iota,
    }
    let (body_fn, src_kind, src_arr_ty) = match m.elem {
        Some((elem_read, l0_exit)) => {
            let (esz, src_arr_ty) = {
                let types = &mv.body_view(body).shared().types;
                (
                    types.size_of(m.elem_ty),
                    types.get_or_make_array(m.elem_ty, n1),
                )
            };
            let Some(body_fn) = outline_scan_body(
                mv,
                body,
                next_minted,
                minted,
                &name,
                m.stored_val,
                m.prev_val,
                m.index,
                Some(elem_read),
                m.index_start,
                m.elem_ty,
                index_ty,
                ScanElem::Data(m.elem_ty),
            ) else {
                return false;
            };
            (body_fn, Src::Slice { l0_exit, esz }, src_arr_ty)
        }
        None => {
            let src_arr_ty = mv
                .body_view(body)
                .shared()
                .types
                .get_or_make_array(i64_ty, n1);
            let Some(body_fn) = outline_scan_body(
                mv,
                body,
                next_minted,
                minted,
                &name,
                m.stored_val,
                m.prev_val,
                m.index,
                None,
                m.index_start,
                m.elem_ty,
                index_ty,
                ScanElem::Scalar(i64_ty),
            ) else {
                return false;
            };
            (body_fn, Src::Iota, src_arr_ty)
        }
    };

    // Anchor every new instruction before the exit block's *first* instruction,
    // not just before the wide store: `array_promote` may have rewritten other
    // exit loads to `at(arr_exit, k)` earlier in the block, and those get
    // redirected to the folded array below — which must therefore dominate them.
    let Some(anchor) = mv
        .body_view(body)
        .block_ref(m.exit)
        .iter()
        .next()
        .map(|i| i.id)
    else {
        return false;
    };
    // Pre-existing exit instructions whose `arr_exit` uses are redirected.
    let preexisting: Vec<InstructionId> = mv
        .body_view(body)
        .block_ref(m.exit)
        .iter()
        .map(|i| i.id)
        .collect();

    // Materialize the scan source. Data-input: the `l0[1..]` byte-slice from
    // element 1 (length `N-1`). Pure generation: `iota(N-1)` typed `[i64; N-1]`.
    let src = match src_kind {
        Src::Slice { l0_exit, esz } => {
            let slice = body.push_mnemonic_with_type(
                Mnemonic::Range(qcode::value::insn::Range {
                    src: l0_exit.localize(fid),
                    start: esz,
                    size: n1 * esz,
                }),
                src_arr_ty,
            );
            body.insert_insn_before(m.exit, anchor, slice);
            ValueId::Instruction(slice)
        }
        Src::Iota => {
            let n1_const = mv.shr().get_const(n1 as u64, 8);
            let iota = body.push_mnemonic_with_type(
                Mnemonic::Intrinsic(IntrinsicApp {
                    id: iota_id,
                    args: vec![n1_const.localize(fid)],
                }),
                src_arr_ty,
            );
            body.insert_insn_before(m.exit, anchor, iota);
            ValueId::Instruction(iota)
        }
    };

    // scan(iota) → singleton(seed) → concat, ahead of every exit use. The scan's
    // body is a *minted* function `push_scan` cannot read for its result type, so
    // build the node with an explicit type: a sequence of the accumulator
    // (stored-value) type with the source's length/kind.
    let scan = {
        let body_ret = mv.body_view(body).type_of(m.stored_val);
        let ty = crate::calls::outline::seq_result_type(mv.body_view(body), src, body_ret);
        let id = body.push_mnemonic_with_type(
            Mnemonic::Scan(qcode::value::insn::Scan {
                body: body_fn,
                init: m.seed_val.localize(fid),
                src: src.localize(fid),
                captures: Vec::new(),
            }),
            ty,
        );
        body.insert_insn_before(m.exit, anchor, id);
        ValueId::Instruction(id)
    };
    let full = {
        let mut host = mv.host(body);
        let mut b = host.builder(m.exit);
        b.set_insert_point_before(anchor);
        let sing = b.push_intrinsic(singleton_id, vec![m.seed_val]).id();
        b.push_intrinsic(concat_id, vec![sing, scan]).id()
    };

    // Redirect every exit use of the loop-carried array — the wide store and any
    // `at(arr, k)` reads `array_promote` left for exit loads — to the folded
    // array, leaving the loop's own array dead for `dce`.
    for id in preexisting {
        let mut mn = mv.body_view(body).insn_ref(id).mnemonic().clone();
        mn.replace_value(m.arr_exit.localize(fid), full.localize(fid));
        body.replace_instruction_mnemonic(id, mn);
    }

    // Delete the residual loop when it is now wholly private (mirrors
    // `loop_to_map`'s deletable path): the exit carries no params and every
    // value the loop defines is used only inside it. Then the loop computes
    // nothing observable — reroute the single preheader straight to the exit
    // and delete the loop blocks. Otherwise (a loop value threaded past the
    // exit, e.g. a clobbered register in the return envelope) leave it for
    // later dead-arg/dce rounds.
    let loop_blocks: Vec<BlockId> = if m.header == m.body {
        vec![m.body]
    } else {
        vec![m.header, m.body]
    };
    let private = is_loop_private(mv.body_view(body), &loop_blocks);
    // Each exit param (a now-dead array pass-through the later gvn would have
    // coalesced) must be re-fed from a preheader-available value: its
    // header-edge incoming directly if loop-invariant, or — when it copies a
    // loop param — that param's own loop-invariant (preheader) incoming.
    let exit_args: Option<Vec<ValueId>> = mv
        .body_view(body)
        .block_ref(m.exit)
        .params()
        .map(|p| p.id())
        .collect::<Vec<_>>()
        .into_iter()
        .map(|p| {
            let rh = mv.body_view(body);
            let k = param_pos(rh, m.exit, p)?;
            let [v] = incoming(rh, m.exit, k)[..] else {
                return None;
            };
            if !value_defined_in(rh, &loop_blocks, v) {
                return Some(v);
            }
            if !matches!(v, ValueId::BlockParam(_)) {
                return None;
            }
            let kv = param_pos(rh, m.header, v)?;
            let init: Vec<ValueId> = incoming(rh, m.header, kv)
                .into_iter()
                .filter(|&w| !value_defined_in(rh, &loop_blocks, w))
                .collect();
            match init[..] {
                [w] => Some(w),
                _ => None,
            }
        })
        .collect();
    if private && let Some(exit_args) = exit_args {
        let preheaders: Vec<BlockId> = mv
            .body_view(body)
            .block_ref(m.header)
            .predecessors()
            .map(|(_, p)| p)
            .filter(|p| !loop_blocks.contains(p))
            .collect();
        if let [preheader] = preheaders[..] {
            // `delete_private_loop` is still host-generic (a cross-module helper,
            // migrated in its own chunk), so drive it through a scoped host.
            let mut host = mv.host(body);
            delete_private_loop(&mut host, preheader, &loop_blocks, m.exit, exit_args);
        }
    }
    true
}

impl FunctionPass for LoopToScan {
    const NAME: &'static str = "loop_to_scan";

    fn description(&self) -> &'static str {
        "Fold the value-carried insert/at fill loop from array_promote into a scanl"
    }

    fn run<'str>(
        &self,
        f: &mut FunctionBody<'str>,
        m: ContextView<'_, 'str>,
        next_minted: &mut u32,
    ) -> Result<Outcome<'str>, String> {
        if let Some(sm) = try_match(m.body_view(f), f.id()) {
            let mut minted = Vec::new();
            let changed = apply(m, f, next_minted, &mut minted, &sm);
            return Ok(Outcome::with_minted(changed, minted));
        }
        Ok(Outcome::unchanged())
    }
}

register_function_pass!(LoopToScan);

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;
    use qcode::{context::Context, value::FunctionBody};

    use crate::mem::array_promote::ArrayPromote;
    use crate::test_util::run_function_pass;

    // The exact seam this rewrite touches: `array_promote` promotes the seeded
    // prefix sum `out[0]=seed; out[i]=out[i-1]+l[i]` into a single carried array
    // (original init + carry read `at(arr,i-1)` + own-lane read `at(arr,i)`), and
    // `loop_to_scan` must then fold it to a `scanl` over the original array. A
    // recognizer mismatch across the two passes is invisible to their per-pass
    // tests, so exercise the chain directly.
    #[test]
    fn array_promote_then_scan_folds_prefix_sum() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn prefix:
            <entry @seed:i64 @base:i64>
                %e0 = @seed[0:4];
                store(ram:4, @base <- %e0);
                goto <head @i=1 @buf=@base>;
            <head @i:i64 @buf:i64>
                %done = @i == 624;
                if %done goto <exit> else goto <body @j=@i @b=@buf>;
            <body @j:i64 @b:i64>
                %jm1 = @j - 1;
                %roff = %jm1 * 4;
                %raddr = @b + %roff;
                %prev = load(ram:4, %raddr);
                %coff = @j * 4;
                %caddr = @b + %coff;
                %cur = load(ram:4, %caddr);
                %next = %cur + %prev;
                store(ram:4, %caddr <- %next);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        assert!(
            run_function_pass::<ArrayPromote>(&mut ctx, prefix).unwrap(),
            "array_promote should promote the prefix sum"
        );
        assert!(
            run_function_pass::<LoopToScan>(&mut ctx, prefix).unwrap(),
            "loop_to_scan should fold the promoted single-array loop"
        );
        let ir = format!("{}", FunctionBody::from_id(&ctx, prefix));
        assert!(
            ir.contains("scanl"),
            "the promoted loop should fold to a scanl over the original array: {ir}"
        );
        let scan_body = FunctionBody::from_id(&ctx, prefix)
            .iter()
            .flat_map(|block| block.iter())
            .find_map(|insn| match insn.mnemonic() {
                Mnemonic::Scan(scan) => Some(scan.body),
                _ => None,
            })
            .expect("folded function contains a scan");
        assert!(
            scan_body.real().is_some(),
            "the install barrier must patch the minted scan body"
        );
    }

    // The *header-carried* prefix sum: the body reads the header induction param
    // `@i` directly (no body index param). The relaxed carried-array matcher now
    // surfaces this shape, but scan v1 is audited for the body-copied form only —
    // `loop_to_scan` must decline it (no rewrite, no scanl), not mis-fold it.
    // TODO(loop-fold-header-index): fold this once the scan recognizer moves onto
    // `unit_induction` (see the gate in `try_match`).
    #[test]
    fn header_carried_scan_shape_declined() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn prefix:
            <entry @seed:i64 @base:i64>
                %e0 = @seed[0:4];
                store(ram:4, @base <- %e0);
                goto <head @i=1>;
            <head @i:i64>
                %done = @i == 624;
                if %done goto <exit> else goto <body>;
            <body>
                %im1 = @i - 1;
                %roff = %im1 * 4;
                %raddr = @base + %roff;
                %prev = load(ram:4, %raddr);
                %coff = @i * 4;
                %caddr = @base + %coff;
                %cur = load(ram:4, %caddr);
                %next = %cur + %prev;
                store(ram:4, %caddr <- %next);
                %i1 = @i + 1;
                goto <head @i=%i1>;
            <exit>
                return at i64 0x0;
            "
        );
        assert!(
            run_function_pass::<ArrayPromote>(&mut ctx, prefix).unwrap(),
            "the header-carried prefix sum should promote"
        );
        assert!(
            !run_function_pass::<LoopToScan>(&mut ctx, prefix).unwrap(),
            "scan v1 must decline the header-carried index shape"
        );
        let ir = format!("{}", FunctionBody::from_id(&ctx, prefix));
        assert!(!ir.contains("scanl"), "declined, not rewritten: {ir}");
    }
}
