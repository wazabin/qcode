//! Whole-array emulation of a `map` over a **fully-known** source: the shapes
//! `body <$> b"…"` and `body <$> enumerate(b"…")`.
//!
//! When a [`Map`]'s source is a constant [`Bytes`](qcode::value::Bytes) blob (or
//! an `enumerate` of one) and every capture is a constant, the map is a constant
//! array: emulate the pure body on each lane and concatenate the per-lane outputs
//! into a single `Bytes` literal typed as the map's result array. Unlike the
//! per-lane projection in [`array_project`](super::array_project) — which
//! recovers one element on demand — this materializes the *entire* result up
//! front; DCE reaps it if every real use turned out to be a single lane.
//!
//! Two lane shapes, both bodies unary in the element:
//! * **`body <$> b"…"`** — `body(src[k])`, the element a scalar byte slice.
//! * **`body <$> enumerate(b"…")`** — `body((index: k, elem: src[k]))`, the
//!   element the `enumerate` tuple, fed to the body via
//!   [`BodyArg::Aggregate`](qcode_emulator::BodyArg) so its `Extract`s resolve.

use qcode::{
    context::Context,
    types::TypeId,
    value::{
        ValueId,
        insn::{Map, Mnemonic, Scan},
    },
};

use qcode_emulator::{BodyArg, SizedValue, StandaloneEmulator};

use crate::calls::return_field;

use super::fold::const_value;
use std::any::Any;

use super::walk::{Claim, Editor, InsnCtx, ModuleSubPass};

/// Upper bound on emulated instructions per lane. Map bodies are loop-free pure
/// expressions, so this only guards against a degenerate body.
const STEP_BUDGET: usize = 100_000;

/// Replace `body <$> b"…"` / `body <$> enumerate(b"…")` with the emulated
/// constant `Bytes` array. Reads the pure *body callee*'s IR, so it runs only on
/// the module host (dispatched by the [`concretize`](super::concretize) pass).
pub(super) struct EmulateMap;

impl<'str> ModuleSubPass<'str> for EmulateMap {
    fn init_state(&self) -> Box<dyn Any> {
        Box::new(())
    }

    fn clone_state(&self, _state: &dyn Any) -> Box<dyn Any> {
        Box::new(())
    }

    fn on_insn(
        &self,
        host: &mut Context<'str>,
        _state: &mut dyn Any,
        ic: &InsnCtx,
        ed: &mut Editor,
    ) -> Claim {
        let ctx: &mut Context = host;
        let folded = match ic.mnemonic.clone() {
            Mnemonic::Map(map) => self.emulate(ctx, ic, &map),
            Mnemonic::Scan(scan) => self.emulate_scan(ctx, ic, &scan),
            _ => return Claim::Pass,
        };
        match folded {
            Some(bytes) => {
                ed.replace(ctx, ic.insn_id, bytes);
                Claim::Done
            }
            None => Claim::Pass,
        }
    }
}

/// How each lane's element is fed to the body.
enum Lane {
    /// `body(src[k])`: the element is a scalar `esz`-byte slice.
    Scalar { esz: usize },
    /// `body((index: k, elem: src[k]))`: the element is the `enumerate` tuple,
    /// `index` of `index_sz` bytes followed by the `esz`-byte element.
    Enumerate { esz: usize, index_sz: usize },
}

impl EmulateMap {
    fn emulate(&self, ctx: &mut Context, ic: &InsnCtx, map: &Map) -> Option<ValueId> {
        // The source must be a fully-known constant: `b"…"` or `enumerate(b"…")`.
        let (data, lane) = const_source(ctx, map.src.qualify(ic.insn_id.func))?;
        let esz = match lane {
            Lane::Scalar { esz } | Lane::Enumerate { esz, .. } => esz,
        };
        if esz == 0 || esz > 8 || data.len() % esz != 0 {
            return None;
        }
        let n = data.len() / esz;

        // Output element width, from the map's result array type.
        let map_ty = ctx.type_of(ic.id);
        let (out_elem, out_n) = ctx.shared.types.array_of(map_ty)?;
        let osz = ctx.shared.types.size_of(out_elem);
        if osz == 0 || osz > 8 || out_n != n {
            return None;
        }

        // Body param sizes: param 0 is the element, the rest are the captures.
        let body = map.body.real()?;
        let root = qcode::value::FunctionBody::from_id(ctx, body).root()?.id;
        let param_sizes: Vec<usize> = qcode::value::BasicBlock::from_id(ctx, root)
            .params()
            .map(|p| p.size())
            .collect();
        if param_sizes.len() != 1 + map.captures.len() {
            return None;
        }

        // Every capture must be a constant for the body to be fully evaluable.
        let mut capture_args: Vec<BodyArg> = Vec::with_capacity(map.captures.len());
        for (&cap, &size) in map.captures.iter().zip(&param_sizes[1..]) {
            let value = const_value(&*ctx, cap.qualify(ic.insn_id.func))?;
            capture_args.push(BodyArg::Scalar(SizedValue::new(value, size)));
        }

        // Emulate the body on each lane and concatenate the outputs.
        let mut out = Vec::with_capacity(n * osz);
        for k in 0..n {
            let elem = read_le(&data, k * esz, esz);
            let element = match lane {
                Lane::Scalar { esz } => BodyArg::Scalar(SizedValue::new(elem, esz)),
                Lane::Enumerate { esz, index_sz } => BodyArg::Aggregate(vec![
                    SizedValue::new(k as u64, index_sz),
                    SizedValue::new(elem, esz),
                ]),
            };
            let mut args = Vec::with_capacity(1 + capture_args.len());
            args.push(element);
            args.extend(capture_args.iter().cloned());

            let mut emu = StandaloneEmulator::new(root);
            emu.run_map_body(ctx, body, &args, STEP_BUDGET).ok()?;
            let ret = return_field(ctx, emu.current_block(), 0)?;
            let mut lane_bytes = emu.get_value_bytes(ctx, ret)?;
            lane_bytes.resize(osz, 0);
            out.extend_from_slice(&lane_bytes);
        }

        // Materialize the result carrying the map's array type. A short array
        // (≤ 8 bytes) is represented as a numeric `Literal`, exactly as such
        // constants *enter* the pipeline (see [`const_source`]); only wider
        // results become a `Bytes` blob. Emitting a literal keeps the value in
        // the numeric domain the rest of GVN handles — a `Bytes` operand would
        // crash the commutative-binop canonicalizer's `value_id_key`.
        if out.len() <= 8 {
            let value = read_le(&out, 0, out.len());
            return Some(ctx.get_typed_const(value, map_ty).id());
        }
        let bid = ctx.get_bytes(out).id();
        if let ValueId::Bytes(b) = bid {
            ctx.shared.values.bytes[b].type_id = map_ty;
        }
        Some(bid)
    }

    /// Emulate a [`Scan`] over a fully-known constant source with a constant
    /// initial accumulator: thread `acc` left-to-right through the lanes
    /// (`acc = body(acc, src[k], captures…)`) and concatenate each step's `acc`
    /// into the result array. The body is **binary** — `acc` is param 0, the lane
    /// element param 1 — so the per-lane args prepend `acc` to the element. Returns
    /// `None` unless the source, the captures, and `init` are all constant.
    fn emulate_scan(&self, ctx: &mut Context, ic: &InsnCtx, scan: &Scan) -> Option<ValueId> {
        let (data, lane) = const_source(ctx, scan.src.qualify(ic.insn_id.func))?;
        let esz = match lane {
            Lane::Scalar { esz } | Lane::Enumerate { esz, .. } => esz,
        };
        if esz == 0 || esz > 8 || data.len() % esz != 0 {
            return None;
        }
        let n = data.len() / esz;

        // Output element width (also the accumulator width), from the result type.
        let scan_ty = ctx.type_of(ic.id);
        let (out_elem, out_n) = ctx.shared.types.array_of(scan_ty)?;
        let osz = ctx.shared.types.size_of(out_elem);
        if osz == 0 || osz > 8 || out_n != n {
            return None;
        }

        // Body param sizes: param 0 is the accumulator, param 1 the element, the
        // rest the captures.
        let body = scan.body.real()?;
        let root = qcode::value::FunctionBody::from_id(ctx, body).root()?.id;
        let param_sizes: Vec<usize> = qcode::value::BasicBlock::from_id(ctx, root)
            .params()
            .map(|p| p.size())
            .collect();
        if param_sizes.len() != 2 + scan.captures.len() {
            return None;
        }
        let acc_sz = param_sizes[0];
        if acc_sz == 0 || acc_sz > 8 {
            return None;
        }

        // Every capture must be a constant for the body to be fully evaluable.
        let mut capture_args: Vec<BodyArg> = Vec::with_capacity(scan.captures.len());
        for (&cap, &size) in scan.captures.iter().zip(&param_sizes[2..]) {
            let value = const_value(&*ctx, cap.qualify(ic.insn_id.func))?;
            capture_args.push(BodyArg::Scalar(SizedValue::new(value, size)));
        }

        // The initial accumulator must be a constant.
        let mut init_bytes = const_bytes(ctx, scan.init.qualify(ic.insn_id.func))?;
        init_bytes.resize(acc_sz, 0);
        let mut acc = read_le(&init_bytes, 0, acc_sz);

        // Thread the accumulator across the lanes, emitting each step's value.
        let mut out = Vec::with_capacity(n * osz);
        for k in 0..n {
            let elem = read_le(&data, k * esz, esz);
            let element = match lane {
                Lane::Scalar { esz } => BodyArg::Scalar(SizedValue::new(elem, esz)),
                Lane::Enumerate { esz, index_sz } => BodyArg::Aggregate(vec![
                    SizedValue::new(k as u64, index_sz),
                    SizedValue::new(elem, esz),
                ]),
            };
            let mut args = Vec::with_capacity(2 + capture_args.len());
            args.push(BodyArg::Scalar(SizedValue::new(acc, acc_sz)));
            args.push(element);
            args.extend(capture_args.iter().cloned());

            let mut emu = StandaloneEmulator::new(root);
            emu.run_map_body(ctx, body, &args, STEP_BUDGET).ok()?;
            let ret = return_field(ctx, emu.current_block(), 0)?;
            let mut lane_bytes = emu.get_value_bytes(ctx, ret)?;
            lane_bytes.resize(osz, 0);
            out.extend_from_slice(&lane_bytes);
            // Carry: the next accumulator is this lane's emitted value.
            acc = read_le(&out, k * osz, osz);
        }

        if out.len() <= 8 {
            let value = read_le(&out, 0, out.len());
            return Some(ctx.get_typed_const(value, scan_ty).id());
        }
        let bid = ctx.get_bytes(out).id();
        if let ValueId::Bytes(b) = bid {
            ctx.shared.values.bytes[b].type_id = scan_ty;
        }
        Some(bid)
    }
}

/// Resolve a `map` source that is a fully-known constant: a constant array
/// (`Lane::Scalar`) or an `enumerate` of one (`Lane::Enumerate`). The constant
/// may be a [`Bytes`](qcode::value::Bytes) blob *or* a numeric `Literal` — a
/// short array (≤ 8 bytes) that fits a `u64` is stored as a plain literal (e.g.
/// `0x1f1e1d2c` for `[i8;4]`), so both must be accepted. Returns the raw source
/// bytes (little-endian, memory order) and the lane shape.
fn const_source(ctx: &Context, src: ValueId) -> Option<(Vec<u8>, Lane)> {
    // Direct constant array: `body <$> <const>`.
    if let Some(data) = const_bytes(ctx, src) {
        let esz = array_elem_size(ctx, ctx.stored_type_of(src)?)?;
        return Some((data, Lane::Scalar { esz }));
    }

    // `body <$> enumerate(<const>)`.
    let ValueId::Instruction(id) = src else {
        return None;
    };
    let Mnemonic::Intrinsic(intr) = ctx.get_insn(id).mnemonic().clone() else {
        return None;
    };
    if intr.id.name() != "enumerate" {
        return None;
    }
    let data = const_bytes(ctx, intr.args.first()?.qualify(id.func))?;
    // The element and index widths are fixed by `enumerate`'s result type
    // (`[(index, elem); n]`), independent of how the source constant is stored.
    let enum_ty = ctx.stored_type_of(src)?;
    let (tuple_ty, _) = ctx.shared.types.array_of(enum_ty)?;
    let index_sz = ctx
        .shared
        .types
        .size_of(ctx.shared.types.field_type(tuple_ty, 0)?);
    let esz = ctx
        .shared
        .types
        .size_of(ctx.shared.types.field_type(tuple_ty, 1)?);
    Some((data, Lane::Enumerate { esz, index_sz }))
}

/// The constant little-endian bytes of `v`, if it is a fully-known constant: a
/// `Bytes` blob or a non-symbolic numeric `Literal` (sized by its type).
fn const_bytes(ctx: &Context, v: ValueId) -> Option<Vec<u8>> {
    match v {
        ValueId::Bytes(bid) => Some(ctx.shared.values.bytes[bid].data.clone()),
        ValueId::Literal(lid) => {
            let lit = &ctx.shared.values.literals[lid];
            if lit.symbolic.is_some() {
                return None;
            }
            let size = ctx.shared.types.size_of(lit.type_id);
            if size == 0 || size > 8 {
                return None;
            }
            let value = qcode::value::LiteralRef::from_id(ctx, lid).value();
            Some(value.to_le_bytes()[..size].to_vec())
        }
        _ => None,
    }
}

/// The element byte-width of an `Array(elem, n)` type.
fn array_elem_size(ctx: &Context, ty: TypeId) -> Option<usize> {
    let (elem, _) = ctx.shared.types.array_of(ty)?;
    Some(ctx.shared.types.size_of(elem))
}

/// Read `size` (≤ 8) little-endian bytes at `start` as a `u64`.
fn read_le(data: &[u8], start: usize, size: usize) -> u64 {
    let mut buf = [0u8; 8];
    buf[..size].copy_from_slice(&data[start..start + size]);
    u64::from_le_bytes(buf)
}

#[cfg(test)]
mod tests {
    use qcode::{
        testing::TestContext,
        types::TypeId,
        value::{
            BasicBlock, FunctionBody, FunctionId, Value, ValueId,
            block::BlockId,
            insn::{IntrinsicId, Mnemonic, Return},
        },
    };

    /// Terminate `entry` with `return value` (the builder only offers a bare
    /// `push_return(ptr)`; map results flow through a value-carrying return).
    fn return_value(tc: &mut TestContext, entry: BlockId, value: ValueId) {
        let (ptr, ret) = {
            let mut b = (&mut tc.ctx).builder(entry);
            let ptr = b.shr().get_const(0, 8);
            let ret = b.push_return(ptr).id();
            (ptr, ret)
        };
        let ValueId::Instruction(rid) = ret else {
            unreachable!()
        };
        tc.ctx.replace_instruction_mnemonic(
            rid,
            Mnemonic::Return(Return {
                ptr: ptr.localize(rid.func),
                value: Some(value.localize(rid.func)),
            }),
        );
    }

    /// `body(elem: i8) -> elem + 1`, marked pure — a unary scalar map body.
    fn build_inc_body(tc: &mut TestContext) -> FunctionId {
        let fid = FunctionBody::make(&mut tc.ctx, "inc".into()).unwrap().id;
        let entry = { tc.ctx.get_or_make_block(0x1000, fid) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let (inc, ptr, ret);
        {
            let mut b = (&mut tc.ctx).builder(entry);
            let elem = b.push_param(1).id();
            let one = b.shr().get_const(1, 1);
            inc = b.push_add(elem, one).id();
            ptr = b.shr().get_const(0, 8);
            ret = b.push_return(ptr).id();
            unsafe { b.dont_finalize() };
        }
        let ValueId::Instruction(rid) = ret else {
            unreachable!()
        };
        tc.ctx.replace_instruction_mnemonic(
            rid,
            Mnemonic::Return(Return {
                ptr: ptr.localize(rid.func),
                value: Some(inc.localize(rid.func)),
            }),
        );
        FunctionBody::from_id_mut(&mut tc.ctx, fid).set_is_pure(true);
        fid
    }

    /// `body(t: (index: i64, elem: i8)) -> elem + (index as i8)`, marked pure —
    /// an index-aware body reading both tuple fields.
    fn build_index_body(tc: &mut TestContext, tuple_ty: TypeId) -> FunctionId {
        let fid = FunctionBody::make(&mut tc.ctx, "addidx".into()).unwrap().id;
        let entry = { tc.ctx.get_or_make_block(0x2000, fid) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let tsz = tc.ctx.shared.types.size_of(tuple_ty);
        let t = {
            let mut b = (&mut tc.ctx).builder(entry);
            b.push_param(tsz).id()
        };
        if let ValueId::BlockParam(pid) = t {
            tc.ctx.block_param_mut(pid).type_id = tuple_ty;
        }
        let (sum, ptr, ret) = {
            let mut b = (&mut tc.ctx).builder(entry);
            let index = b.push_extract(t, 0).id();
            let idx_lo = b.get_range(index, 0..1).unwrap().id();
            let elem = b.push_extract(t, 1).id();
            let sum = b.push_add(elem, idx_lo).id();
            let ptr = b.shr().get_const(0, 8);
            let ret = b.push_return(ptr).id();
            unsafe { b.dont_finalize() };
            (sum, ptr, ret)
        };
        let ValueId::Instruction(rid) = ret else {
            unreachable!()
        };
        tc.ctx.replace_instruction_mnemonic(
            rid,
            Mnemonic::Return(Return {
                ptr: ptr.localize(rid.func),
                value: Some(sum.localize(rid.func)),
            }),
        );
        FunctionBody::from_id_mut(&mut tc.ctx, fid).set_is_pure(true);
        fid
    }

    /// The constant memory-order bytes a host's `map` collapsed to, if any.
    /// A wide result is a `Bytes` blob; a short array (≤ 8 bytes) is a numeric
    /// `Literal` typed as the array, so both are decoded here.
    fn map_bytes(tc: &TestContext, host: FunctionId) -> Option<Vec<u8>> {
        let root = FunctionBody::from_id(&tc.ctx, host).root()?.id;
        BasicBlock::from_id(&tc.ctx, root).iter().find_map(|i| {
            let Mnemonic::Return(Return { value: Some(v), .. }) = i.mnemonic() else {
                return None;
            };
            match v.qualify(root.func) {
                ValueId::Bytes(bid) => Some(tc.ctx.shared.values.bytes[bid].data.clone()),
                ValueId::Literal(lid) => {
                    let lit = &tc.ctx.shared.values.literals[lid];
                    let size = tc.ctx.shared.types.size_of(lit.type_id);
                    Some(lit.value.to_le_bytes()[..size].to_vec())
                }
                _ => None,
            }
        })
    }

    /// `inc <$> b"\x01\x02\x03"` emulates to `b"\x02\x03\x04"`.
    #[test]
    fn emulates_map_over_bytes() {
        let mut tc = TestContext::new();
        let body = build_inc_body(&mut tc);

        let host = FunctionBody::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = { tc.ctx.get_or_make_block(0x5000, host) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, host);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let src = tc.ctx.get_bytes(vec![0x01, 0x02, 0x03]).id();
        let map_val = {
            let mut b = (&mut tc.ctx).builder(entry);
            b.push_map(body, src, Vec::new()).id()
        };
        return_value(&mut tc, entry, map_val);

        super::super::concretize::concretize_function(&mut tc.ctx, host);

        assert_eq!(
            map_bytes(&tc, host),
            Some(vec![0x02, 0x03, 0x04]),
            "inc over a byte blob folds to the per-lane constants"
        );
    }

    /// `inc <$> 0x1f1e1d2c` (a numeric `[i8;4]` literal, not a `Bytes` blob)
    /// emulates to `b"\x2d\x1e\x1f\x20"` — the literal source arm of `const_bytes`.
    #[test]
    fn emulates_map_over_int_literal() {
        let mut tc = TestContext::new();
        let body = build_inc_body(&mut tc);

        let host = FunctionBody::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = { tc.ctx.get_or_make_block(0x7000, host) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, host);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        // A short `[i8;4]` array stored as a plain numeric literal (`0x1f1e1d2c`),
        // exactly as constprop hands it to `map` in the reported sample.
        let i8 = tc.ctx.shared.types.get_or_make_int(1);
        let arr_ty = tc.ctx.shared.types.get_or_make_array(i8, 4);
        let src = tc.ctx.get_typed_const(0x1f1e1d2c, arr_ty).id();
        let map_val = {
            let mut b = (&mut tc.ctx).builder(entry);
            b.push_map(body, src, Vec::new()).id()
        };
        return_value(&mut tc, entry, map_val);

        super::super::concretize::concretize_function(&mut tc.ctx, host);

        assert_eq!(
            map_bytes(&tc, host),
            Some(vec![0x2d, 0x1e, 0x1f, 0x20]),
            "inc over a numeric-literal array folds per-lane (each LE byte + 1)"
        );
    }

    /// `addidx <$> enumerate(b"\x10\x20\x30")` emulates to `b"\x10\x21\x32"`
    /// (`elem + index`), exercising the aggregate `(index, elem)` lane seeding.
    #[test]
    fn emulates_map_over_enumerate_bytes() {
        let mut tc = TestContext::new();
        let enum_id = IntrinsicId::from_name("enumerate").unwrap();

        // The enumerate tuple type for an `[i8; N]` source.
        let i8 = tc.ctx.shared.types.get_or_make_int(1);
        let arr_ty = tc.ctx.shared.types.get_or_make_array(i8, 3);
        let enum_result_ty = enum_id.desc().result_type(&tc.ctx.shared.types, &[arr_ty]);
        let (tuple_ty, _) = tc.ctx.shared.types.array_of(enum_result_ty).unwrap();
        let body = build_index_body(&mut tc, tuple_ty);

        let host = FunctionBody::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = { tc.ctx.get_or_make_block(0x6000, host) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, host);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let src = tc.ctx.get_bytes(vec![0x10, 0x20, 0x30]).id();
        let map_val = {
            let mut b = (&mut tc.ctx).builder(entry);
            let en = b.push_intrinsic(enum_id, vec![src]).id();
            b.push_map_typed(body, en, Vec::new(), arr_ty).id()
        };
        return_value(&mut tc, entry, map_val);

        super::super::concretize::concretize_function(&mut tc.ctx, host);

        assert_eq!(
            map_bytes(&tc, host),
            Some(vec![0x10, 0x21, 0x32]),
            "addidx over enumerate of a byte blob folds per-lane (elem + index)"
        );
    }

    /// `body(acc, (index, elem)) -> acc + elem`, marked pure — a running-sum scan
    /// body, binary in `(accumulator, enumerate tuple)`.
    fn build_sum_body(tc: &mut TestContext, tuple_ty: TypeId) -> FunctionId {
        let fid = FunctionBody::make(&mut tc.ctx, "scansum".into())
            .unwrap()
            .id;
        let entry = { tc.ctx.get_or_make_block(0x7000, fid) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let tsz = tc.ctx.shared.types.size_of(tuple_ty);
        let (sum, ptr, ret) = {
            let mut b = (&mut tc.ctx).builder(entry);
            let acc = b.push_param(1).id(); // param 0: accumulator (i8)
            let t = b.push_param(tsz).id(); // param 1: enumerate tuple
            if let ValueId::BlockParam(pid) = t {
                b.set_param_type(pid, tuple_ty);
            }
            let elem = b.push_extract(t, 1).id();
            let sum = b.push_add(acc, elem).id();
            let ptr = b.shr().get_const(0, 8);
            let ret = b.push_return(ptr).id();
            unsafe { b.dont_finalize() };
            (sum, ptr, ret)
        };
        let ValueId::Instruction(rid) = ret else {
            unreachable!()
        };
        tc.ctx.replace_instruction_mnemonic(
            rid,
            Mnemonic::Return(Return {
                ptr: ptr.localize(rid.func),
                value: Some(sum.localize(rid.func)),
            }),
        );
        FunctionBody::from_id_mut(&mut tc.ctx, fid).set_is_pure(true);
        fid
    }

    /// `scanl (acc+elem) 0 enumerate(b"\x01\x02\x03")` emulates to the prefix sums
    /// `b"\x01\x03\x06"` — exercising the accumulator threading across lanes.
    #[test]
    fn emulates_scan_prefix_sum_over_enumerate() {
        let mut tc = TestContext::new();
        let enum_id = IntrinsicId::from_name("enumerate").unwrap();

        let i8 = tc.ctx.shared.types.get_or_make_int(1);
        let arr_ty = tc.ctx.shared.types.get_or_make_array(i8, 3);
        let enum_result_ty = enum_id.desc().result_type(&tc.ctx.shared.types, &[arr_ty]);
        let (tuple_ty, _) = tc.ctx.shared.types.array_of(enum_result_ty).unwrap();
        let body = build_sum_body(&mut tc, tuple_ty);

        let host = FunctionBody::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = { tc.ctx.get_or_make_block(0x8000, host) };
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, host);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let src = tc.ctx.get_bytes(vec![0x01, 0x02, 0x03]).id();
        let scan_val = {
            let mut b = (&mut tc.ctx).builder(entry);
            let en = b.push_intrinsic(enum_id, vec![src]).id();
            let init = b.shr().get_const(0, 1);
            b.push_scan_typed(body, init, en, Vec::new(), arr_ty).id()
        };
        return_value(&mut tc, entry, scan_val);

        super::super::concretize::concretize_function(&mut tc.ctx, host);

        assert_eq!(
            map_bytes(&tc, host),
            Some(vec![0x01, 0x03, 0x06]),
            "the running-sum scan folds to its prefix sums"
        );
    }
}
