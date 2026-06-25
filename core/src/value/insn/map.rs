//! `map`: a total element-wise map over an array value.
//!
//! [`Map`] applies the pure function `body` to every lane of the array `src`,
//! producing an array of the same length: conceptually `out[i] = body(i, src[i],
//! captures…)`. It is the projectable representation of a loop that rewrites a
//! buffer element-wise (see `ARGPROMOTE_ARRAY_MAP.md`).
//!
//! `body` is a **function symbol** (like [`Call::target`](super::Call)), not a
//! value operand, so `Map` stays an ordinary first-order SSA instruction: its
//! value operands are `src` plus any loop-invariant `captures` the body closes
//! over. The projection rewrite
//! `Range(Map(body, src), k·sz, sz) → body(k, Range(src, k·sz, sz), captures…)`
//! recovers one element as an expression without materializing the whole array.

use crate::{
    context::Context,
    value::{Function, ValueId, ValueRef, function::FunctionId},
};
use std::fmt::Formatter;

use super::mnemonic::MnemonicKind;

/// A total element-wise map `out[i] = body(i, src[i], captures…)`. The result
/// type is the array type of `src` (v1: the body preserves the element width).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Map {
    /// The pure per-element function, applied at each lane. A symbol, not an
    /// operand — exactly like a direct call's target.
    pub body: FunctionId,
    /// The array value mapped over.
    pub src: ValueId,
    /// Loop-invariant values the body closes over (the index and element are
    /// supplied per-lane by the map itself). Empty for a closed body.
    pub captures: Vec<ValueId>,
}

impl MnemonicKind for Map {
    fn opcode(&self) -> &'static str {
        "map"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        write!(
            f,
            "map(@{}, {}",
            Function::from_id(ctx, self.body).name(),
            ValueRef::new(self.src, ctx),
        )?;
        for &c in &self.captures {
            write!(f, ", {}", ValueRef::new(c, ctx))?;
        }
        f.write_str(");")
    }

    fn args(&self) -> Vec<ValueId> {
        let mut args = Vec::with_capacity(1 + self.captures.len());
        args.push(self.src);
        args.extend(self.captures.iter().copied());
        args
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        builder::Builder,
        testing::TestContext,
        value::{
            BasicBlock, Function, ValueId,
            insn::{Mnemonic, mnemonic::MnemonicKind},
        },
    };

    #[test]
    fn map_builds_with_array_result_and_symbol_body() {
        let mut tc = TestContext::new();

        // A pure per-element body function (its content is irrelevant here).
        let body = Function::make(&mut tc.ctx, "body".into()).unwrap().id;

        // A host function holding an `[i8;20]`-typed value to map over.
        let host = Function::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, host);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let i8 = tc.ctx.types.get_or_make_int(1);
        let array_ty = tc.ctx.types.get_or_make_array(i8, 20);

        let (src, cap) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            (b.push_param(20).id(), b.push_param(4).id())
        };
        // Type the source as the array (params default to int of their width)
        // *before* building the map, so the map's result type picks it up.
        if let ValueId::BlockParam(pid) = src {
            tc.ctx.values.block_params[pid].type_id = array_ty;
        }
        let map_val = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_map(body, src, vec![cap]).id()
        };
        let ValueId::Instruction(map_id) = map_val else {
            panic!("push_map should yield an instruction value");
        };

        let m = match tc.ctx.get_insn(map_id).mnemonic().clone() {
            Mnemonic::Map(m) => m,
            other => panic!("expected Map, got {other:?}"),
        };

        // `body` is a symbol; `src` + captures are the value operands.
        assert_eq!(m.body, body);
        assert_eq!(m.args(), vec![src, cap], "src then captures are the operands");
        assert!(!m.args().contains(&ValueId::Function(body)), "body is not an operand");

        // Result type is the array type of `src`.
        assert_eq!(tc.ctx.type_of(map_val), array_ty);

        // replace_value rewrites operands but never the body symbol.
        let new_src = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_param(20).id()
        };
        let mut rewritten = Mnemonic::Map(m);
        rewritten.replace_value(src, new_src);
        let Mnemonic::Map(r) = rewritten else { unreachable!() };
        assert_eq!(r.src, new_src);
        assert_eq!(r.body, body, "body symbol is untouched by replace_value");
        assert_eq!(r.captures, vec![cap]);
    }
}
