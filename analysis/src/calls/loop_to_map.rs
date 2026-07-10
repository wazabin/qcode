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
    space::{Space, SpaceType},
    value::{
        BlockId, FunctionId, ValueId,
        insn::{InstructionId, IntrinsicId, Mnemonic},
        util::{base_ref::BaseRef, base_ref::HostRef, host_mut::HostMut},
    },
};

use super::carried_array::{CarriedArray, classify_body_reads, exit_view, find_carried_array};
use super::outline::{outline_expression, outline_tupled, pure_slice};
use crate::loop_info::{
    cbranch_exit, delete_private_loop, incoming, is_loop_private, param_parent, param_pos,
    recognize_loops, users_of,
};
use crate::pipeline::{FunctionBody, ModuleView};
use crate::{FunctionPass, register_function_pass};

// ===========================================================================
// Total-map recognizer
// ===========================================================================
//
// After the single-carried-array `array_promote`, an element-wise fill loop over
// a bounded region threads one array-typed header param through the loop with
// `insert`/`at`, initialized from a preheader wide `Load` (in-place map over the
// original data), a `splat(0, N)` (pure generation), or an incoming by-value array
// param (an argpromoted buffer), and stored wide at the exit:
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
    /// The initial array the map ranges over: a preheader wide `Load` (reads
    /// original), a `splat(0, N)` (pure generation), or an incoming by-value array
    /// param (an argpromoted buffer). The map reproduces the loop for any of them.
    init_arr: ValueId,
    /// The RAM store consuming `arr_exit`, when the region is real memory (else
    /// the exit uses are a private-shadow return envelope and `arr_exit`'s uses
    /// are simply replaced).
    store_id: Option<InstructionId>,
}

/// Match the canonical total-map loop in `fid` on the shared carried-array form,
/// or `None` for any other shape (the function is then left untouched).
fn try_match(host: HostRef, fid: FunctionId) -> Option<MapMatch> {
    // A map is a carried array with no lane-0 accumulator seed (a seed is the scan
    // shape, `loop_to_scan`'s pattern — the two are mutually exclusive here) and a
    // real initial array to range over.
    let ca = find_carried_array(host, fid)?;
    if ca.seed.is_some() {
        return None;
    }
    let init_arr = ca.init?;

    // No accumulator carry read (`at(arr_b, index-1)`); the own-lane original read
    // (`at(arr_b, index)`) is optional.
    let reads = classify_body_reads(host, &ca)?;
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
    // `count` lanes and the promotion writes every one. The index check — inits at
    // 0, steps by +1 on the back-edge — is the shared `unit_induction` resolver's,
    // which accepts both split-loop index shapes (body-copied and header-carried)
    // so this recognizer never learns the difference. Its trip bound comes from
    // the guard; it must agree with the coverage proof's lane count.
    let lp = recognize_loops(host, fid)
        .into_iter()
        .find(|l| l.header == ca.header && l.body == ca.body)?;
    let ind = lp.unit_induction(host, ca.index)?;
    if ind.start != 0 {
        return None; // v1 maps tile [0, N) from index 0
    }
    if ind.count != ca.count as i64 {
        return None; // guard bound and array length must agree
    }

    // Exit discovery: the header terminator is the loop guard; the exit is the
    // successor that is not itself a header predecessor.
    let (exit, _stay) = cbranch_exit(host, ca.header)?;
    let arr_exit = exit_view(host, &ca, exit);

    // The map ranges over `init_arr` and reproduces the loop for *any* init value:
    // in-order coverage (array_promote's proof) makes lane `j` read before it is
    // written, so `at(arr_b, j) == init_arr[j]` regardless of whether the init is a
    // preheader wide `Load`, a `splat`, or an incoming by-value array param. There is
    // no init-shape obligation to enforce here — `init_arr` is already an array value.

    // Consumer: a RAM store of the carried array's exit view (real memory). When
    // absent (private argpromote shadow, wide temp store already dce'd) the exit
    // uses of `arr_exit` are the return envelope — the rewrite just replaces them.
    let store_id = host
        .block_ref(exit)
        .iter()
        .find_map(|i| match i.mnemonic() {
            Mnemonic::Store(s)
                if matches!(Space::from_id(host.shared(), s.space).ty, SpaceType::Ram)
                    && s.src == arr_exit =>
            {
                Some(i.id)
            }
            _ => None,
        });

    // Deletability of the residual loop is decided in `apply`, after the carried
    // array's escaping use has been redirected to the map (a pre-rewrite check would
    // still see that use in the collapsed-exit form).

    Some(MapMatch {
        ca,
        exit,
        arr_exit,
        elem_read,
        init_arr,
        store_id,
    })
}

/// Does the per-element body actually read the loop index? `enumerate` is only
/// worth inserting when it does; a value-only body maps directly over the array.
fn body_uses_index(
    host: HostRef,
    stored_val: ValueId,
    index: ValueId,
    elem_read: Option<ValueId>,
) -> bool {
    if stored_val == index {
        return true;
    }
    let mut inputs = vec![index];
    inputs.extend(elem_read);
    match pure_slice(host, stored_val, &inputs) {
        Some(slice) => slice
            .iter()
            .any(|&iid| host.insn_ref(iid).mnemonic().args().contains(&index)),
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
fn apply<'str>(m: &ModuleView<'_, 'str>, body: &mut FunctionBody<'str>, mm: &MapMatch) -> bool {
    let fid = body.id();
    let enum_id = IntrinsicId::from_name("enumerate").expect("enumerate registered");
    let (name, uses_index, tuple_ty) = {
        let host = body.read_host(m);
        let name = format!("{}_map_body", host.function_ref(fid).name());
        let uses_index = body_uses_index(host, mm.ca.stored_val, mm.ca.index, mm.elem_read);
        // The enumerate element type is needed before outlining (an index-aware
        // body unpacks the `(index, elem)` tuple); resolve it while only reading.
        let tuple_ty = if uses_index {
            let arr_ty = host.type_of(mm.init_arr);
            let enum_ty = enum_id.desc().result_type(&host.shared().types, &[arr_ty]);
            match host.shared().types.array_of(enum_ty) {
                Some((tuple_ty, _)) => Some(tuple_ty),
                None => return false,
            }
        } else {
            None
        };
        (name, uses_index, tuple_ty)
    };

    // Outline the body before any rewrite, so a non-closed body (or an exhausted
    // mint pool) leaves the loop untouched.
    let body_fn = if uses_index {
        match outline_tupled(
            m,
            body,
            &name,
            mm.ca.stored_val,
            mm.ca.index,
            mm.elem_read,
            tuple_ty.expect("index-aware body has a tuple type"),
        ) {
            Some(f) => f,
            None => return false,
        }
    } else {
        // A value-only body needs an element to map over; without an own-lane read
        // there is nothing to bind (a pure constant fill is out of v1 scope).
        let Some(elem) = mm.elem_read else {
            return false;
        };
        match outline_expression(m, body, &name, mm.ca.stored_val, &[elem]) {
            Some(f) => f,
            None => return false,
        }
    };

    let mut host = body.host(m);

    // Build `map(body, enumerate(arr0))` (index-aware) or `map(body, arr0)`
    // (value-only) ahead of the consumer, then forward the exit view to it.
    let anchor = mm
        .store_id
        .or_else(|| host.block_ref(mm.exit).iter().next().map(|i| i.id));
    // The map source (`enumerate(arr0)` when index-aware, else `arr0`), built
    // ahead of the consumer.
    let src = if uses_index {
        let mut b = Builder::from_block(BaseRef::new(host.reborrow_host(), mm.exit));
        if let Some(at) = anchor {
            b.set_insert_point_before(at);
        }
        b.push_intrinsic(enum_id, vec![mm.init_arr]).id()
    } else {
        mm.init_arr
    };
    // Build the `map` node with an explicit result type: the body is a *minted*
    // function, so `push_map`'s "read the body's return type" cannot see it — the
    // element type is the loop's stored value.
    let map_val = {
        let body_ret = host.read_host().type_of(mm.ca.stored_val);
        let ty = crate::calls::outline::seq_result_type(host.read_host(), src, body_ret);
        let id = host.push_mnemonic_with_type(
            fid,
            Mnemonic::Map(qcode::value::insn::Map {
                body: body_fn,
                src,
                captures: Vec::new(),
            }),
            ty,
        );
        // Insert right where the source (or the anchor) sits, before the consumer.
        match anchor {
            Some(at) => host.insert_insn_before(mm.exit, at, id),
            None => BaseRef::new(host.reborrow_host(), mm.exit).push_insn(id),
        }
        ValueId::Instruction(id)
    };
    // Redirect the exit view of the carried array to the map. When redundant-φ
    // elimination collapsed the exit pass-through, the exit view *is* the header
    // array param, which is also the in-loop `at`/`insert` base — those in-loop
    // uses must stay on the param (the map is defined in the exit and would not
    // dominate them). So redirect only uses outside the loop; when the exit view
    // is a distinct exit param (the uncollapsed case) it has no in-loop uses and
    // this is exactly `replace_all_uses_with`.
    let loop_blocks = [mm.ca.header, mm.ca.body];
    let exit_users: Vec<InstructionId> = users_of(host.read_host(), mm.arr_exit).to_vec();
    for id in exit_users {
        if host
            .insn_ref(id)
            .parent()
            .is_some_and(|b| loop_blocks.contains(&b.id))
        {
            continue;
        }
        let mut mn = host.insn_ref(id).mnemonic().clone();
        mn.replace_value(mm.arr_exit, map_val);
        host.replace_instruction_mnemonic(id, mn);
    }

    // Deletability must reflect the *post-redirect* state. The redirect above moved
    // the carried array's only out-of-loop use (the wide store / return-envelope
    // `pack`) onto the map, so in the collapsed-exit form (`arr_exit == arr_h`) the
    // header array param becomes loop-private only now. Deciding this before the
    // rewrite would wrongly see that escaping use and keep the loop — and nothing
    // later deletes it: a self-carried loop's own guard and back-edge keep the index
    // and array live through the CFG, so no ordinary dce can collect the cycle.
    let deletable = is_loop_private(host.read_host(), &loop_blocks);

    // Delete the residual loop when wholly private (mirrors `loop_to_scan::apply`):
    // reroute the single preheader straight to the exit, re-feeding each exit param
    // from a preheader-available value, then delete the loop blocks.
    let defined_in_loop = |host: HostRef, v: ValueId| match v {
        ValueId::BlockParam(_) => param_parent(host, v).is_some_and(|b| loop_blocks.contains(&b)),
        ValueId::Instruction(id) => host
            .insn_ref(id)
            .parent()
            .is_some_and(|b| loop_blocks.contains(&b.id)),
        _ => false,
    };
    let exit_args: Option<Vec<ValueId>> = host
        .block_ref(mm.exit)
        .params()
        .map(|p| p.id())
        .collect::<Vec<_>>()
        .into_iter()
        .map(|p| {
            let rh = host.read_host();
            let k = param_pos(rh, mm.exit, p)?;
            let [v] = incoming(rh, mm.exit, k)[..] else {
                return None;
            };
            if !defined_in_loop(rh, v) {
                return Some(v);
            }
            if !matches!(v, ValueId::BlockParam(_)) {
                return None;
            }
            let kv = param_pos(rh, mm.ca.header, v)?;
            let init: Vec<ValueId> = incoming(rh, mm.ca.header, kv)
                .into_iter()
                .filter(|&w| !defined_in_loop(rh, w))
                .collect();
            match init[..] {
                [w] => Some(w),
                _ => None,
            }
        })
        .collect();
    if deletable && let Some(exit_args) = exit_args {
        let preheaders: Vec<BlockId> = host
            .block_ref(mm.ca.header)
            .predecessors()
            .map(|(_, p)| p)
            .filter(|p| !loop_blocks.contains(p))
            .collect();
        if let [preheader] = preheaders[..] {
            delete_private_loop(&mut host, fid, preheader, &loop_blocks, mm.exit, exit_args);
        }
    }
    true
}

/// Recognize a total-map loop in this function and fold it to a `map`, outlining
/// the per-element body into a fresh minted pure function. Only pure functions
/// are considered (the recognizer's coverage/totality reasoning relies on
/// `array_promote` having functionalized the region). Returns `true` if changed.
pub(crate) fn recognize_total_map<'str>(
    m: &ModuleView<'_, 'str>,
    body: &mut FunctionBody<'str>,
) -> bool {
    let fid = body.id();
    if !body.read_host(m).function_ref(fid).is_pure() {
        return false;
    }
    let Some(mm) = try_match(body.read_host(m), fid) else {
        return false;
    };
    apply(m, body, &mm)
}

#[derive(Default)]
pub struct LoopToMap;

impl FunctionPass for LoopToMap {
    const NAME: &'static str = "loop_to_map";
    const MINTS: bool = true;
    fn description(&self) -> &'static str {
        "Rewrite a total element-wise array loop as a single map"
    }
    fn run<'str>(
        &self,
        m: &ModuleView<'_, 'str>,
        f: &mut FunctionBody<'str>,
    ) -> Result<bool, String> {
        Ok(recognize_total_map(m, f))
    }
}

register_function_pass!(LoopToMap);

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::value::insn::{Binop, IntBinop, Mnemonic};
    use qcode::{
        context::Context,
        value::{BasicBlock, Function},
    };
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
        let folded = run_function_pass::<LoopToMap>(ctx, fid).unwrap();
        (promoted, folded)
    }

    /// Like [`promote_then_map`] but runs redundant-φ elimination between promote
    /// and fold, mirroring the real pipeline: `array_promote` mints a distinct
    /// body array param, which `dead_block_args` then collapses onto the header
    /// param (its single header→body incoming). The result is the *fully*
    /// header-carried form — both index and carried array read from the header —
    /// that the recognizer sees in practice.
    fn promote_collapse_map(ctx: &mut Context, fid: FunctionId) -> (bool, bool) {
        use crate::dce::remove_dead_block_args;
        let promoted = run_function_pass::<ArrayPromote>(ctx, fid).unwrap();
        let blocks: Vec<BlockId> = Function::from_id(ctx, fid).iter().map(|b| b.id).collect();
        let root = Function::from_id(ctx, fid).root().map(|b| b.id);
        while remove_dead_block_args(ctx, &blocks, root) {}
        Function::from_id_mut(ctx, fid).set_is_pure(true);
        let folded = run_function_pass::<LoopToMap>(ctx, fid).unwrap();
        (promoted, folded)
    }

    /// The real-pipeline shape: after promote **and** redundant-φ elimination the
    /// carried array's carry base is the *header* array param (the minted body
    /// param collapsed onto it), so the whole loop is fully header-carried. The
    /// fold must still recognize it. This is the case that reached the field as
    /// `promoted=true, folded=false` before the body block was derived from the
    /// carry insert rather than from `param_parent(arr_b)`.
    #[test]
    fn fully_header_carried_fill_folds_after_collapse() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn xorbuf:
            <entry @base:i64>
                goto <head @i=0>;
            <head @i:i64>
                %done = @i == 16;
                if %done goto <exit> else goto <body>;
            <body>
                %off = @i * 4;
                %addr = @base + %off;
                %x = load(ram:4, %addr);
                %v = %x ^ 0x5a;
                store(ram:4, %addr <- %v);
                %i1 = @i + 1;
                goto <head @i=%i1>;
            <exit>
                return at i64 0x0;
            "
        );
        let (promoted, folded) = promote_collapse_map(&mut ctx, xorbuf);
        assert!(promoted, "the fill should promote");
        assert!(folded, "the fully header-carried loop should fold to a map");
        let ir = format!("{}", Function::from_id(&ctx, xorbuf));
        assert!(ir.contains("<$>"), "folds to a map: {ir}");
        // The collapsed-exit form escapes the carried array to the wide store, so
        // deletability must be judged *after* the redirect — the dead loop (its carry
        // `$insert`) must be gone, not left behind for a dce that never collects it.
        assert!(
            !ir.contains("$insert"),
            "the residual dead loop must be deleted in the collapsed-exit case: {ir}"
        );
    }

    /// The field case (an obfuscated `out[i] = f(out[i], i)` byte loop): fully
    /// header-carried *and* index-aware. After promote + redundant-φ elimination
    /// both the carried array and the index are read from the header, and the
    /// body consumes the index — so it must fold to `map(f, enumerate(arr0))`.
    /// This drives `outline_tupled` with a header-param index input.
    #[test]
    fn fully_header_carried_index_aware_folds_enumerate() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn mix:
            <entry @base:i64>
                goto <head @i=0>;
            <head @i:i64>
                %done = @i < 24;
                if %done goto <body> else goto <exit>;
            <body>
                %addr = @base + @i;
                %cur = load(ram:1, %addr);
                %it = @i[0:1];
                %x = %cur ^ %it;
                %y = %x + %cur;
                store(ram:1, %addr <- %y);
                %i1 = @i + 1;
                goto <head @i=%i1>;
            <exit>
                return at i64 0x0;
            "
        );
        let (promoted, folded) = promote_collapse_map(&mut ctx, mix);
        assert!(promoted, "the fill should promote");
        assert!(
            folded,
            "the fully header-carried index-aware loop should fold"
        );
        let ir = format!("{}", Function::from_id(&ctx, mix));
        assert!(
            ir.contains("<$>") && ir.contains("enumerate"),
            "index-aware body maps over enumerate: {ir}"
        );
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
        let aliases = AliasResult::simple_for_function(&ctx, xorbuf);
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

    /// The header-carried twin of `array_promote_then_map_folds_xor_fill`: the
    /// body reads the header induction param `@i` directly instead of copying it
    /// into a body param — the shape redundant-φ elimination (`dead_block_args`)
    /// leaves behind. The fold must not depend on the index shape: a value-only
    /// body maps without an `enumerate`, exactly like its body-copied twin.
    #[test]
    fn header_carried_xor_fill_folds() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn xorbuf:
            <entry @base:i64>
                goto <head @i=0>;
            <head @i:i64>
                %done = @i == 16;
                if %done goto <exit> else goto <body>;
            <body>
                %off = @i * 4;
                %addr = @base + %off;
                %x = load(ram:4, %addr);
                %v = %x ^ 0x5a;
                store(ram:4, %addr <- %v);
                %i1 = @i + 1;
                goto <head @i=%i1>;
            <exit>
                return at i64 0x0;
            "
        );
        let (promoted, folded) = promote_then_map(&mut ctx, xorbuf);
        assert!(promoted, "the header-carried fill should promote");
        assert!(folded, "the header-carried promoted loop should fold");
        let ir = format!("{}", Function::from_id(&ctx, xorbuf));
        assert!(ir.contains("<$>"), "folds to a map: {ir}");
        assert!(
            !ir.contains("enumerate"),
            "an index-free body needs no enumerate: {ir}"
        );
    }

    /// Header-carried with an *index-aware* body (`out[i] = out[i] + i`, the body
    /// consuming the header param `@i` directly): folds to
    /// `map(f, enumerate(arr0))`. This also proves the outline path binds a free
    /// header-param index as a body input (design §4 — `pure_slice` stops at
    /// declared inputs regardless of their defining block).
    #[test]
    fn header_carried_index_aware_body_maps_enumerate() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn addidx:
            <entry @base:i64>
                goto <head @i=0>;
            <head @i:i64>
                %done = @i == 16;
                if %done goto <exit> else goto <body>;
            <body>
                %off = @i * 4;
                %addr = @base + %off;
                %x = load(ram:4, %addr);
                %it = @i[0:4];
                %v = %x + %it;
                store(ram:4, %addr <- %v);
                %i1 = @i + 1;
                goto <head @i=%i1>;
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

    /// The field shape (410a60-class, argpromoted): a promoted, fully header-carried
    /// in-place map whose initial array is an **incoming by-value array param** (an
    /// argpromoted buffer) rather than a preheader wide `Load`. `array_promote` always
    /// builds a `Load` init for an own-lane read, so the by-value form is reproduced
    /// here by swapping that `Load` for a fresh entry array param. The own-lane read
    /// still reads genuine original data (`at(arr_b, j) == init[j]` by coverage), so
    /// `map(body, param)` is correct — this is exactly the case the removed init-shape
    /// guard wrongly rejected for not being a `Load`.
    #[test]
    fn param_init_own_lane_read_folds_to_map() {
        use crate::dce::remove_dead_block_args;
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn xorbuf:
            <entry @base:i64>
                goto <head @i=0>;
            <head @i:i64>
                %done = @i == 16;
                if %done goto <exit> else goto <body>;
            <body>
                %off = @i * 4;
                %addr = @base + %off;
                %x = load(ram:4, %addr);
                %v = %x ^ 0x5a;
                store(ram:4, %addr <- %v);
                %i1 = @i + 1;
                goto <head @i=%i1>;
            <exit>
                return at i64 0x0;
            "
        );
        // Promote + redundant-φ elimination → the fully header-carried, `Load`-init
        // map loop (identical to `fully_header_carried_fill_folds_after_collapse`).
        assert!(run_function_pass::<ArrayPromote>(&mut ctx, xorbuf).unwrap());
        let blocks: Vec<BlockId> = Function::from_id(&ctx, xorbuf)
            .iter()
            .map(|b| b.id)
            .collect();
        let root = Function::from_id(&ctx, xorbuf).root().map(|b| b.id);
        while remove_dead_block_args(&mut ctx, &blocks, root) {}

        // Swap the array-typed preheader `Load` for a fresh incoming array param —
        // the argpromoted by-value buffer form the field IR actually carries.
        let cur_blocks: Vec<BlockId> = Function::from_id(&ctx, xorbuf)
            .iter()
            .map(|b| b.id)
            .collect();
        let mut load_ids = Vec::new();
        for bid in &cur_blocks {
            for i in BasicBlock::from_id(&ctx, *bid).iter() {
                if matches!(i.mnemonic(), Mnemonic::Load(_)) {
                    load_ids.push(i.id);
                }
            }
        }
        let load_id = load_ids
            .into_iter()
            .find(|&id| {
                let ty = ctx.type_of(ValueId::Instruction(id));
                ctx.types.array_of(ty).is_some()
            })
            .expect("array-typed preheader init load");
        let arr_ty = ctx.type_of(ValueId::Instruction(load_id));
        let arr_sz = ctx.types.size_of(arr_ty);
        let entry = Function::from_id(&ctx, xorbuf).root().unwrap().id;
        let pid = BasicBlock::from_id_mut(&mut ctx, entry)
            .push_param(arr_sz)
            .id;
        ctx.values.block_param_mut(pid).type_id = arr_ty;
        ctx.replace_all_uses_with(ValueId::Instruction(load_id), ValueId::BlockParam(pid));

        Function::from_id_mut(&mut ctx, xorbuf).set_is_pure(true);
        assert!(
            run_function_pass::<LoopToMap>(&mut ctx, xorbuf).unwrap(),
            "a param-init own-lane map must fold (the deleted Load-only guard blocked it)"
        );
        let ir = format!("{}", Function::from_id(&ctx, xorbuf));
        assert!(ir.contains("<$>"), "folds to a map: {ir}");
        assert!(
            !ir.contains("enumerate"),
            "an index-free body needs no enumerate: {ir}"
        );
        assert!(
            !ir.contains("$insert"),
            "the residual dead loop must be deleted for a param-init map: {ir}"
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
            !run_function_pass::<LoopToMap>(&mut ctx, prefix).unwrap(),
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
            !run_function_pass::<LoopToMap>(&mut ctx, impure).unwrap(),
            "an impure body must not fold to a map"
        );
    }
}
