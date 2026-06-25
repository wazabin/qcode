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
        insn::{Map, Mnemonic},
    },
};

use qcode_emulator::{BodyArg, SizedValue, StandaloneEmulator};

use crate::calls::return_field;

use super::fold::const_value;
use super::walk::{Claim, Editor, InsnCtx, SubPass};

/// Upper bound on emulated instructions per lane. Map bodies are loop-free pure
/// expressions, so this only guards against a degenerate body.
const STEP_BUDGET: usize = 100_000;

/// Replace `body <$> b"…"` / `body <$> enumerate(b"…")` with the emulated
/// constant `Bytes` array.
pub(super) struct EmulateMap;

impl SubPass for EmulateMap {
    type State = ();

    fn on_insn(&self, ctx: &mut Context, _state: &mut (), ic: &InsnCtx, ed: &mut Editor) -> Claim {
        let Mnemonic::Map(map) = ic.mnemonic.clone() else {
            return Claim::Pass;
        };
        match self.emulate(ctx, ic, &map) {
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
        let (data, lane) = const_source(ctx, map.src)?;
        let esz = match lane {
            Lane::Scalar { esz } | Lane::Enumerate { esz, .. } => esz,
        };
        if esz == 0 || esz > 8 || data.len() % esz != 0 {
            return None;
        }
        let n = data.len() / esz;

        // Output element width, from the map's result array type.
        let map_ty = ctx.type_of(ic.id);
        let (out_elem, out_n) = ctx.types.array_of(map_ty)?;
        let osz = ctx.types.size_of(out_elem);
        if osz == 0 || osz > 8 || out_n != n {
            return None;
        }

        // Body param sizes: param 0 is the element, the rest are the captures.
        let root = qcode::value::Function::from_id(ctx, map.body).root()?.id;
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
            let value = const_value(ctx, cap)?;
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
            emu.run_map_body(ctx, map.body, &args, STEP_BUDGET).ok()?;
            let ret = return_field(ctx, emu.current_block(), 0)?;
            let mut lane_bytes = emu.get_value_bytes(ctx, ret)?;
            lane_bytes.resize(osz, 0);
            out.extend_from_slice(&lane_bytes);
        }

        // Materialize the result as a constant blob carrying the map's array type.
        let bid = ctx.get_bytes(out).id();
        if let ValueId::Bytes(b) = bid {
            ctx.values.bytes[b].type_id = map_ty;
        }
        Some(bid)
    }
}

/// Resolve a `map` source that is a fully-known constant: a `Bytes` blob
/// (`Lane::Scalar`) or an `enumerate` of one (`Lane::Enumerate`). Returns the
/// raw source bytes and the lane shape.
fn const_source(ctx: &Context, src: ValueId) -> Option<(Vec<u8>, Lane)> {
    match src {
        ValueId::Bytes(bid) => {
            let esz = array_elem_size(ctx, ctx.values.bytes[bid].type_id)?;
            Some((ctx.values.bytes[bid].data.clone(), Lane::Scalar { esz }))
        }
        ValueId::Instruction(id) => {
            let Mnemonic::Intrinsic(intr) = ctx.get_insn(id).mnemonic().clone() else {
                return None;
            };
            if intr.id.name() != "enumerate" {
                return None;
            }
            let ValueId::Bytes(bid) = *intr.args.first()? else {
                return None;
            };
            let esz = array_elem_size(ctx, ctx.values.bytes[bid].type_id)?;
            // The index field width is fixed by `enumerate`'s result type.
            let enum_ty = ctx.stored_type_of(src)?;
            let (tuple_ty, _) = ctx.types.array_of(enum_ty)?;
            let index_sz = ctx.types.size_of(ctx.types.field_type(tuple_ty, 0)?);
            Some((
                ctx.values.bytes[bid].data.clone(),
                Lane::Enumerate { esz, index_sz },
            ))
        }
        _ => None,
    }
}

/// The element byte-width of an `Array(elem, n)` type.
fn array_elem_size(ctx: &Context, ty: TypeId) -> Option<usize> {
    let (elem, _) = ctx.types.array_of(ty)?;
    Some(ctx.types.size_of(elem))
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
        builder::Builder,
        testing::TestContext,
        types::TypeId,
        value::{
            BasicBlock, Function, FunctionId, Value, ValueId,
            block::BlockId,
            insn::{IntrinsicId, Mnemonic, Return},
        },
    };

    /// Terminate `entry` with `return value` (the builder only offers a bare
    /// `push_return(ptr)`; map results flow through a value-carrying return).
    fn return_value(tc: &mut TestContext, entry: BlockId, value: ValueId) {
        let (ptr, ret) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let ptr = b.context_mut().get_const(0, 8).id();
            let ret = b.push_return(ptr).id();
            (ptr, ret)
        };
        let ValueId::Instruction(rid) = ret else {
            unreachable!()
        };
        tc.ctx.replace_instruction_mnemonic(
            rid,
            Mnemonic::Return(Return {
                ptr,
                value: Some(value),
            }),
        );
    }

    /// `body(elem: i8) -> elem + 1`, marked pure — a unary scalar map body.
    fn build_inc_body(tc: &mut TestContext) -> FunctionId {
        let fid = Function::make(&mut tc.ctx, "inc".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let (inc, ptr, ret);
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let elem = b.push_param(1).id();
            let one = b.context_mut().get_const(1, 1).id();
            inc = b.push_add(elem, one).id();
            ptr = b.context_mut().get_const(0, 8).id();
            ret = b.push_return(ptr).id();
            unsafe { b.dont_finalize() };
        }
        let ValueId::Instruction(rid) = ret else {
            unreachable!()
        };
        tc.ctx.replace_instruction_mnemonic(
            rid,
            Mnemonic::Return(Return {
                ptr,
                value: Some(inc),
            }),
        );
        Function::from_id_mut(&mut tc.ctx, fid).set_is_pure(true);
        fid
    }

    /// `body(t: (index: i64, elem: i8)) -> elem + (index as i8)`, marked pure —
    /// an index-aware body reading both tuple fields.
    fn build_index_body(tc: &mut TestContext, tuple_ty: TypeId) -> FunctionId {
        let fid = Function::make(&mut tc.ctx, "addidx".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x2000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let tsz = tc.ctx.types.size_of(tuple_ty);
        let t = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_param(tsz).id()
        };
        if let ValueId::BlockParam(pid) = t {
            tc.ctx.values.block_params[pid].type_id = tuple_ty;
        }
        let (sum, ptr, ret) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let index = b.push_extract(t, 0).id();
            let idx_lo = b.get_range(index, 0..1).unwrap().id();
            let elem = b.push_extract(t, 1).id();
            let sum = b.push_add(elem, idx_lo).id();
            let ptr = b.context_mut().get_const(0, 8).id();
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
                ptr,
                value: Some(sum),
            }),
        );
        Function::from_id_mut(&mut tc.ctx, fid).set_is_pure(true);
        fid
    }

    /// The constant `Bytes` blob a host's `map` collapsed to, if any.
    fn map_bytes(tc: &TestContext, host: FunctionId) -> Option<Vec<u8>> {
        let root = Function::from_id(&tc.ctx, host).root()?.id;
        BasicBlock::from_id(&tc.ctx, root).iter().find_map(|i| {
            if let Mnemonic::Return(Return { value: Some(v), .. }) = i.mnemonic()
                && let ValueId::Bytes(bid) = v
            {
                Some(tc.ctx.values.bytes[*bid].data.clone())
            } else {
                None
            }
        })
    }

    /// `inc <$> b"\x01\x02\x03"` emulates to `b"\x02\x03\x04"`.
    #[test]
    fn emulates_map_over_bytes() {
        let mut tc = TestContext::new();
        let body = build_inc_body(&mut tc);

        let host = Function::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x5000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, host);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let src = tc.ctx.get_bytes(vec![0x01, 0x02, 0x03]).id();
        let map_val = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_map(body, src, Vec::new()).id()
        };
        return_value(&mut tc, entry, map_val);

        let aliases = crate::AliasResult::simple(&tc.ctx);
        while super::super::gvn_function(&mut tc.ctx, host, Some(&aliases)) {}

        assert_eq!(
            map_bytes(&tc, host),
            Some(vec![0x02, 0x03, 0x04]),
            "inc over a byte blob folds to the per-lane constants"
        );
    }

    /// `addidx <$> enumerate(b"\x10\x20\x30")` emulates to `b"\x10\x21\x32"`
    /// (`elem + index`), exercising the aggregate `(index, elem)` lane seeding.
    #[test]
    fn emulates_map_over_enumerate_bytes() {
        let mut tc = TestContext::new();
        let enum_id = IntrinsicId::from_name("enumerate").unwrap();

        // The enumerate tuple type for an `[i8; N]` source.
        let i8 = tc.ctx.types.get_or_make_int(1);
        let arr_ty = tc.ctx.types.get_or_make_array(i8, 3);
        let enum_result_ty = enum_id.desc().result_type(&mut tc.ctx.types, &[arr_ty]);
        let (tuple_ty, _) = tc.ctx.types.array_of(enum_result_ty).unwrap();
        let body = build_index_body(&mut tc, tuple_ty);

        let host = Function::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x6000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, host);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let src = tc.ctx.get_bytes(vec![0x10, 0x20, 0x30]).id();
        let map_val = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let en = b.push_intrinsic(enum_id, vec![src]).id();
            b.push_map(body, en, Vec::new()).id()
        };
        return_value(&mut tc, entry, map_val);

        let aliases = crate::AliasResult::simple(&tc.ctx);
        while super::super::gvn_function(&mut tc.ctx, host, Some(&aliases)) {}

        assert_eq!(
            map_bytes(&tc, host),
            Some(vec![0x10, 0x21, 0x32]),
            "addidx over enumerate of a byte blob folds per-lane (elem + index)"
        );
    }
}
