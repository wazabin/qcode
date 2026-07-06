//! Loop-to-map: recognize a total element-wise array loop on the shared
//! carried-array form (see [`super::carried_array`], produced by
//! [`array_promote`](crate::mem::array_promote)) and rewrite it as a single
//! [`Map`](qcode::value::insn::Map) over the array value, outlining the
//! per-element body into a fresh pure function.
//!
//! This file owns the total-map recognizer ([`try_match`]/[`apply`]). The bounded
//! NUL-scan "strlen" recognizers live in the sibling [`super::strlen`] and
//! accumulator scans in [`super::loop_to_scan`]. The per-element bodies are
//! outlined into fresh pure functions by the shared [`super::outline`] machinery.

use qcode::{
    builder::Builder,
    context::Context,
    space::{Space, SpaceType},
    value::{
        BasicBlock, BlockId, Function, FunctionId, ValueId,
        insn::{InstructionId, IntrinsicId, Mnemonic},
    },
};

use super::carried_array::{CarriedArray, classify_body_reads, exit_view, find_carried_array};
use super::outline::{outline_expression, outline_tupled, pure_slice};
use crate::loop_info::{
    cbranch_exit, delete_private_loop, incoming, is_increment, is_loop_private, literal,
    param_parent, param_pos,
};
use crate::{Pass, PipelineEnv};

// ===========================================================================
// Total-map recognizer
// ===========================================================================
//
// After the single-carried-array `array_promote`, an element-wise fill loop over
// a bounded region threads one array-typed header param through the loop with
// `insert`/`at`, initialized from a preheader wide `Load` (in-place map over the
// original data) or a `splat(0, N)` (pure generation), and stored wide at the
// exit:
//
//   <entry @base>
//       %arr0 = load(ram:N*esz, @base)   // reads-original init  — OR —
//       %arr0 = splat(0, N)              // pure-generation init
//       goto <head @i=0 @arr=%arr0>;
//   <head @i @arr:[e;N]>
//       if @i == N goto <exit @arr> else goto <body @j=@i @a=@arr>;
//   <body @j @a>
//       %x   = at(@a, @j)                // optional own-lane original read
//       %v   = body(@j, %x)              // pure
//       %ins = insert(@a, @j, %v)        // carry
//       goto <head @i=@j+1 @arr=%ins>;
//   <exit @arr>  store(ram:N*esz, @base <- @arr); ...
//
// A map is exactly a carried array *without an accumulator*: the body writes lane
// `@j` but never reads lane `@j-1` (that carry read is the scan shape,
// [`loop_to_scan`](super::loop_to_scan)'s pattern). This recognizer shares the
// carried-array matcher ([`super::carried_array`]) with `loop_to_scan` and is
// mutually exclusive with it by the single test `ca.seed.is_none()`.
//
// It rewrites the exit view of the carried array to a `map`: when the body reads
// only the element it is `map(body, arr0)` with a unary `body(elem)`; when it
// also reads the index it is `map(body, enumerate(arr0))` with `body(tuple)`
// unpacking the `(index, elem)` pair. The dead loop is deleted when wholly
// private, else left for later cleanup; the function stays pure and correct.
//
// v1 promotes a *single* region, so the old two-buffer `dst[i] = body(src[i])`
// copy no longer surfaces as a clean map (the second-region read makes the body
// impure for outlining and the match bails) — an accepted generality regression.

/// A recognized total-map loop built on the shared carried-array matcher.
struct MapMatch {
    /// The carried array and its loop structure (header/body/arr/index/…).
    ca: CarriedArray,
    /// The loop exit block (holds the wide store / return-envelope pack).
    exit: BlockId,
    /// The carried array as the exit block sees it (the rewrite target).
    arr_exit: ValueId,
    /// The `at(arr_b, index)` own-lane original read, if the body reads its lane.
    elem_read: Option<ValueId>,
    /// The initial array the map ranges over: the preheader wide `Load` (reads
    /// original) or the `splat(0, N)` (pure generation).
    init_arr: ValueId,
    /// The RAM store consuming `arr_exit`, when the region is real memory (else
    /// the exit uses are a private-shadow return envelope and `arr_exit`'s uses
    /// are simply replaced).
    store_id: Option<InstructionId>,
    /// Whether the loop is wholly private and can be deleted after the rewrite
    /// (exit carries only the array pass-through, every loop value used only in
    /// the loop). Otherwise the residual loop is left for later dce.
    deletable: bool,
}

/// Match the canonical total-map loop in `fid` on the shared carried-array form,
/// or `None` for any other shape (the function is then left untouched).
fn try_match(ctx: &mut Context, fid: FunctionId) -> Option<MapMatch> {
    // A map is a carried array with no lane-0 accumulator seed (a seed is the scan
    // shape, `loop_to_scan`'s pattern — the two are mutually exclusive here) and a
    // real initial array to range over.
    let ca = find_carried_array(ctx, fid)?;
    if ca.seed.is_some() {
        return None;
    }
    let init_arr = ca.init?;

    // No accumulator carry read (`at(arr_b, index-1)`); the own-lane original read
    // (`at(arr_b, index)`) is optional.
    let reads = classify_body_reads(ctx, &ca)?;
    if reads.prev.is_some() {
        return None;
    }
    let elem_read = reads.own;

    // Split shape only: the rewrite and deletion address header and body
    // separately. A rotated (`header == body`) map is deferred (mirrors
    // `loop_to_scan`'s v1 rejection).
    if ca.header == ca.body {
        return None;
    }

    // Totality is `array_promote`'s coverage proof: the carried array has exactly
    // `count` lanes and the promotion writes every one. We only confirm the index
    // the body carries inits at 0 and steps by +1 on the back-edge (the promoted
    // guard is an equality `@i == N`, which `value_range` cannot narrow, so we do
    // not lean on it here — mirroring `loop_to_scan`).
    let k_index = param_pos(ctx, ca.body, ca.index)?;
    let feeds = {
        let [hp] = incoming(ctx, ca.body, k_index)[..] else {
            return None;
        };
        if param_parent(ctx, hp) != Some(ca.header) {
            return None;
        }
        let k_hp = param_pos(ctx, ca.header, hp)?;
        incoming(ctx, ca.header, k_hp)
    };
    if !feeds.iter().any(|&v| is_increment(ctx, v, ca.index)) {
        return None;
    }
    let inits: Vec<i64> = feeds
        .iter()
        .filter(|&&v| !is_increment(ctx, v, ca.index))
        .filter_map(|&v| literal(ctx, v).map(|x| x as i64))
        .collect();
    if inits != [0] {
        return None; // v1 maps tile [0, N) from index 0
    }

    // Exit discovery: the header terminator is the loop guard; the exit is the
    // successor that is not itself a header predecessor.
    let (exit, _stay) = cbranch_exit(ctx, ca.header)?;
    let arr_exit = exit_view(ctx, &ca, exit);

    // When the body reads its own original lane, the map ranges over the *original*
    // array, so the init must be the preheader wide `Load` — reading lane `j` of a
    // `splat`-init region is legal but yields literal zero, not `l0[j]`, so matching
    // it as an element read would be wrong (mirrors `loop_to_scan`'s scan check).
    if elem_read.is_some() {
        let ValueId::Instruction(iid) = init_arr else {
            return None;
        };
        if !matches!(ctx.get_insn(iid).mnemonic(), Mnemonic::Load(_)) {
            return None;
        }
    }

    // Consumer: a RAM store of the carried array's exit view (real memory). When
    // absent (private argpromote shadow, wide temp store already dce'd) the exit
    // uses of `arr_exit` are the return envelope — the rewrite just replaces them.
    let store_id = BasicBlock::from_id(ctx, exit)
        .iter()
        .find_map(|i| match i.mnemonic() {
            Mnemonic::Store(s)
                if matches!(Space::from_id(ctx, s.space).ty, SpaceType::Ram)
                    && s.src == arr_exit =>
            {
                Some(i.id)
            }
            _ => None,
        });

    // Deletable iff wholly private: every value the loop defines is used only inside
    // it (the exit's array pass-through is re-fed from the preheader init in `apply`).
    let loop_blocks = [ca.header, ca.body];
    let deletable = is_loop_private(ctx, &loop_blocks);

    Some(MapMatch {
        ca,
        exit,
        arr_exit,
        elem_read,
        init_arr,
        store_id,
        deletable,
    })
}

/// Does the per-element body actually read the loop index? `enumerate` is only
/// worth inserting when it does; a value-only body maps directly over the array.
fn body_uses_index(
    ctx: &Context,
    stored_val: ValueId,
    index: ValueId,
    elem_read: Option<ValueId>,
) -> bool {
    if stored_val == index {
        return true;
    }
    let mut inputs = vec![index];
    inputs.extend(elem_read);
    match pure_slice(ctx, stored_val, &inputs) {
        Some(slice) => slice
            .iter()
            .any(|&iid| ctx.get_insn(iid).mnemonic().args().contains(&index)),
        None => false,
    }
}

/// Rewrite a matched loop: outline the per-element body, replace the exit view of
/// the carried array with a `map`, then (when wholly private) reroute the
/// preheader past the loop and delete the dead loop blocks.
///
/// The map source depends on whether the body reads the index. A value-only body
/// maps over the array directly — `map(body, arr0)`, `body(elem)`. An index-aware
/// body maps over `enumerate(arr0)`, whose element is the `(index, elem)` tuple —
/// `map(body, enumerate(arr0))`, `body(tuple)` unpacking it. Returns `false` if
/// the body is not a closed pure expression of `(index, element?)` (then nothing
/// is changed — outlining is all-or-nothing and runs before any rewrite).
fn apply(ctx: &mut Context, fid: FunctionId, m: &MapMatch) -> bool {
    let name = format!("{}_map_body", Function::from_id(ctx, fid).name());
    let enum_id = IntrinsicId::from_name("enumerate").expect("enumerate registered");
    let uses_index = body_uses_index(ctx, m.ca.stored_val, m.ca.index, m.elem_read);

    // Outline the body before any rewrite, so a non-closed body leaves the loop
    // untouched.
    let body_fn = if uses_index {
        let arr_ty = ctx.type_of(m.init_arr);
        let enum_ty = enum_id.desc().result_type(&mut ctx.types, &[arr_ty]);
        let Some((tuple_ty, _)) = ctx.types.array_of(enum_ty) else {
            return false;
        };
        match outline_tupled(
            ctx,
            &name,
            m.ca.stored_val,
            m.ca.index,
            m.elem_read,
            tuple_ty,
        ) {
            Some(f) => f,
            None => return false,
        }
    } else {
        // A value-only body needs an element to map over; without an own-lane read
        // there is nothing to bind (a pure constant fill is out of v1 scope).
        let Some(elem) = m.elem_read else {
            return false;
        };
        match outline_expression(ctx, &name, m.ca.stored_val, &[elem]) {
            Some(f) => f,
            None => return false,
        }
    };

    // Build `map(body, enumerate(arr0))` (index-aware) or `map(body, arr0)`
    // (value-only) ahead of the consumer, then forward the exit view to it.
    let anchor = m
        .store_id
        .or_else(|| BasicBlock::from_id(ctx, m.exit).iter().next().map(|i| i.id));
    let map_val = {
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, m.exit));
        if let Some(at) = anchor {
            b.set_insert_point_before(at);
        }
        let src = if uses_index {
            b.push_intrinsic(enum_id, vec![m.init_arr]).id()
        } else {
            m.init_arr
        };
        b.push_map(body_fn, src, Vec::new()).id()
    };
    ctx.replace_all_uses_with(m.arr_exit, map_val);

    // Delete the residual loop when wholly private (mirrors `loop_to_scan::apply`):
    // reroute the single preheader straight to the exit, re-feeding each exit param
    // from a preheader-available value, then delete the loop blocks.
    let loop_blocks = [m.ca.header, m.ca.body];
    let defined_in_loop = |ctx: &Context, v: ValueId| match v {
        ValueId::BlockParam(_) => param_parent(ctx, v).is_some_and(|b| loop_blocks.contains(&b)),
        ValueId::Instruction(id) => ctx
            .get_insn(id)
            .parent()
            .is_some_and(|b| loop_blocks.contains(&b.id)),
        _ => false,
    };
    let exit_args: Option<Vec<ValueId>> = BasicBlock::from_id(ctx, m.exit)
        .params()
        .map(|p| p.id())
        .collect::<Vec<_>>()
        .into_iter()
        .map(|p| {
            let k = param_pos(ctx, m.exit, p)?;
            let [v] = incoming(ctx, m.exit, k)[..] else {
                return None;
            };
            if !defined_in_loop(ctx, v) {
                return Some(v);
            }
            if !matches!(v, ValueId::BlockParam(_)) {
                return None;
            }
            let kv = param_pos(ctx, m.ca.header, v)?;
            let init: Vec<ValueId> = incoming(ctx, m.ca.header, kv)
                .into_iter()
                .filter(|&w| !defined_in_loop(ctx, w))
                .collect();
            match init[..] {
                [w] => Some(w),
                _ => None,
            }
        })
        .collect();
    if m.deletable
        && let Some(exit_args) = exit_args
    {
        let preheaders: Vec<BlockId> = BasicBlock::from_id(ctx, m.ca.header)
            .predecessors()
            .map(|(_, p)| p)
            .filter(|p| !loop_blocks.contains(p))
            .collect();
        if let [preheader] = preheaders[..] {
            delete_private_loop(ctx, fid, preheader, &loop_blocks, m.exit, exit_args);
        }
    }
    true
}

/// Recognize total-map loops across all pure functions, rewriting each to a
/// `map`. Returns `true` if anything changed.
pub(crate) fn recognize_total_maps(ctx: &mut Context) -> bool {
    let fids: Vec<FunctionId> = ctx.function_ids();
    let mut changed = false;
    for fid in fids {
        if !Function::from_id(ctx, fid).is_pure() {
            continue;
        }
        if let Some(m) = try_match(ctx, fid) {
            changed |= apply(ctx, fid, &m);
        }
    }
    changed
}
#[derive(Default)]
pub struct LoopToMap;

impl Pass for LoopToMap {
    const NAME: &'static str = "loop_to_map";
    fn description(&self) -> &'static str {
        "Rewrite a total element-wise array loop as a single map"
    }
    fn run(&self, ctx: &mut Context, _env: &PipelineEnv) -> Result<bool, String> {
        Ok(recognize_total_maps(ctx))
    }
}

crate::register_module_pass!(LoopToMap);

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::value::insn::{Binop, IntBinop, Mnemonic};
    use qcode_macro::qcode;

    use crate::AliasResult;
    use crate::gvn::gvn_function;
    use crate::mem::array_promote::ArrayPromote;
    use crate::test_util::run_function_pass;

    /// Promote `fid`, mark it pure (the real pipeline functionalizes it before
    /// `loop_to_map`, which gates on purity), then fold its total map. Returns
    /// `(promoted, folded)`.
    fn promote_then_map(ctx: &mut Context, fid: FunctionId) -> (bool, bool) {
        let promoted = run_function_pass::<ArrayPromote>(ctx, fid).unwrap();
        Function::from_id_mut(ctx, fid).set_is_pure(true);
        let folded = recognize_total_maps(ctx);
        (promoted, folded)
    }

    /// An in-place indexed map whose element body is *index-free*
    /// (`out[i] = out[i] ^ 0x5a`): `array_promote` threads one carried array with
    /// an own-lane `at` read, and `loop_to_map` folds it to `map(f, arr0)` — a
    /// value-only body, so **no** `enumerate` is inserted.
    #[test]
    fn array_promote_then_map_folds_xor_fill() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn xorbuf:
            <entry @base:i64>
                goto <head @i=0 @buf=@base>;
            <head @i:i64 @buf:i64>
                %done = @i == 16;
                if %done goto <exit> else goto <body @j=@i @b=@buf>;
            <body @j:i64 @b:i64>
                %off = @j * 4;
                %addr = @b + %off;
                %x = load(ram:4, %addr);
                %v = %x ^ 0x5a;
                store(ram:4, %addr <- %v);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        let (promoted, folded) = promote_then_map(&mut ctx, xorbuf);
        assert!(promoted, "the indexed fill should promote");
        assert!(folded, "the promoted loop should fold to a map");
        let ir = format!("{}", Function::from_id(&ctx, xorbuf));
        assert!(ir.contains("<$>"), "folds to a map: {ir}");
        assert!(
            !ir.contains("enumerate"),
            "an index-free body needs no enumerate: {ir}"
        );
        // Exactly one outlined pure body, and it recomputes the xor.
        let bodies: Vec<FunctionId> = ctx
            .function_ids()
            .into_iter()
            .filter(|&f| f != xorbuf && Function::from_id(&ctx, f).is_pure())
            .collect();
        assert_eq!(bodies.len(), 1, "one map body outlined");
        let broot = Function::from_id(&ctx, bodies[0]).root().unwrap().id;
        assert!(
            BasicBlock::from_id(&ctx, broot).iter().any(|i| matches!(
                i.mnemonic(),
                Mnemonic::Binop(b) if matches!(b.op, Binop::Int(IntBinop::Xor))
            )),
            "the body recomputes the xor"
        );
    }

    /// The motivating `bool`-migration case (410a60-class): the loop's continue
    /// guard is the lifted `(i < 16) & 1 != 0` — a comparison spilled to a byte,
    /// masked with `1`, and re-tested — not a bare comparison. GVN's `& 1` +
    /// `zext(b) != 0` collapse rewrites it back to `i < 16`, after which
    /// `array_promote`/`loop_to_map` fold the fill exactly as for a clean guard.
    #[test]
    fn masked_guard_loop_folds_to_map_after_gvn() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn xorbuf:
            <entry @base:i64>
                goto <head @i=0 @buf=@base>;
            <head @i:i64 @buf:i64>
                %lt = @i < 16;
                %z = zext(i8, %lt);
                %m = %z & 0x1;
                %cont = %m != 0x0;
                if %cont goto <body @j=@i @b=@buf> else goto <exit>;
            <body @j:i64 @b:i64>
                %off = @j * 4;
                %addr = @b + %off;
                %x = load(ram:4, %addr);
                %v = %x ^ 0x5a;
                store(ram:4, %addr <- %v);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );

        // Collapse the masked guard to the bare `i < 16` first.
        let aliases = AliasResult::simple(&ctx);
        while gvn_function(&mut ctx, xorbuf, Some(&aliases)) {}
        let header = format!("{}", Function::from_id(&ctx, xorbuf));
        assert!(
            !header.contains(" & i8 0x1") && !header.contains("!= i8 0x0"),
            "the `& 1 != 0` mask should be gone after GVN:\n{header}"
        );

        let (promoted, folded) = promote_then_map(&mut ctx, xorbuf);
        assert!(promoted, "the masked-guard fill should promote after GVN");
        assert!(folded, "the promoted loop should fold to a map");
        let ir = format!("{}", Function::from_id(&ctx, xorbuf));
        assert!(ir.contains("<$>"), "folds to a map: {ir}");
    }

    /// The same loop with an *index-aware* body (`out[i] = out[i] + i`) folds to
    /// `map(f, enumerate(arr0))` — the index forces an `enumerate`.
    #[test]
    fn index_aware_body_maps_enumerate() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn addidx:
            <entry @base:i64>
                goto <head @i=0 @buf=@base>;
            <head @i:i64 @buf:i64>
                %done = @i == 16;
                if %done goto <exit> else goto <body @j=@i @b=@buf>;
            <body @j:i64 @b:i64>
                %off = @j * 4;
                %addr = @b + %off;
                %x = load(ram:4, %addr);
                %jt = @j[0:4];
                %v = %x + %jt;
                store(ram:4, %addr <- %v);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        let (promoted, folded) = promote_then_map(&mut ctx, addidx);
        assert!(promoted && folded);
        let ir = format!("{}", Function::from_id(&ctx, addidx));
        assert!(
            ir.contains("<$>") && ir.contains("enumerate"),
            "an index-aware body maps over enumerate: {ir}"
        );
    }

    /// A write-only generation `out[i] = i * 3` promotes to a `splat(0, N)` init
    /// (no region load); folding maps over `enumerate(splat)` — the map source
    /// chains back to the splat, and no new region `Load` is introduced.
    #[test]
    fn generation_loop_maps_over_splat() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn generate:
            <entry @base:i64>
                goto <head @i=0 @buf=@base>;
            <head @i:i64 @buf:i64>
                %done = @i == 16;
                if %done goto <exit> else goto <body @j=@i @b=@buf>;
            <body @j:i64 @b:i64>
                %jt = trunc(i32, @j);
                %next = %jt * 3;
                %off = @j * 4;
                %addr = @b + %off;
                store(ram:4, %addr <- %next);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        let (promoted, folded) = promote_then_map(&mut ctx, generate);
        assert!(promoted && folded);
        let ir = format!("{}", Function::from_id(&ctx, generate));
        assert!(
            ir.contains("<$>") && ir.contains("$splat("),
            "maps over splat: {ir}"
        );
        assert!(
            !ir.contains("load(ram"),
            "generation introduces no region load: {ir}"
        );
    }

    /// The seeded prefix sum (a scan: lane-0 seed insert + `at(arr, i-1)` carry
    /// read) is `loop_to_scan`'s shape, not a map — `recognize_total_maps` must
    /// decline it (mutual exclusion via `ca.seed`).
    #[test]
    fn scan_shape_is_not_matched_by_map() {
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
        assert!(run_function_pass::<ArrayPromote>(&mut ctx, prefix).unwrap());
        Function::from_id_mut(&mut ctx, prefix).set_is_pure(true);
        assert!(
            !recognize_total_maps(&mut ctx),
            "a seeded scan is not a map"
        );
    }

    /// All-or-nothing outlining: a body reaching an unrelated RAM load is not a
    /// closed pure expression, so the map is not applied and nothing changes.
    #[test]
    fn impure_body_left_alone() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn impure:
            <entry @base:i64 @other:i64>
                goto <head @i=0 @buf=@base>;
            <head @i:i64 @buf:i64>
                %done = @i == 16;
                if %done goto <exit> else goto <body @j=@i @b=@buf>;
            <body @j:i64 @b:i64>
                %off = @j * 4;
                %addr = @b + %off;
                %x = load(ram:4, %addr);
                %y = load(ram:4, @other);
                %v = %x + %y;
                store(ram:4, %addr <- %v);
                %j1 = @j + 1;
                goto <head @i=%j1 @buf=@b>;
            <exit>
                return at i64 0x0;
            "
        );
        run_function_pass::<ArrayPromote>(&mut ctx, impure).unwrap();
        Function::from_id_mut(&mut ctx, impure).set_is_pure(true);
        assert!(
            !recognize_total_maps(&mut ctx),
            "an impure body must not fold to a map"
        );
    }
}
