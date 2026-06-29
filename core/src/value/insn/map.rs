//! `map`: a total element-wise map over an array value.
//!
//! [`Map`] applies the pure **unary** function `body` to every lane of the array
//! `src`, producing an array of the same length: conceptually `out[i] =
//! body(src[i], captures…)`. The body takes the element only — index-aware bodies
//! map over [`enumerate`](super::Intrinsic)`(arr)`, whose element is the `(index,
//! elem)` tuple. The result element type is the body's return type, which need
//! not equal the input element type. It is the projectable representation of a
//! loop that rewrites a buffer element-wise (see `ARGPROMOTE_ARRAY_MAP.md`).
//!
//! `body` is a **function symbol** (like [`Call::target`](super::Call)), not a
//! value operand, so `Map` stays an ordinary first-order SSA instruction: its
//! value operands are `src` plus any loop-invariant `captures` the body closes
//! over. The projection rewrite
//! `Range(Map(body, src), k·osz, osz) → body(Range(src, k·isz, isz), captures…)`
//! recovers one element as an expression without materializing the whole array.

use crate::{
    context::Context,
    value::{Function, ValueId, ValueRef, function::FunctionId},
};
use std::fmt::Formatter;

use super::mnemonic::{Args, MnemonicKind};
use smallvec::SmallVec;

/// A total element-wise map `out[i] = body(src[i], captures…)`. The result is
/// `[U; N]` where `N` is `src`'s length and `U` is the body's return type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Map {
    /// The pure per-element function, applied at each lane. A symbol, not an
    /// operand — exactly like a direct call's target. Unary in the element.
    pub body: FunctionId,
    /// The array value mapped over.
    pub src: ValueId,
    /// Loop-invariant values the body closes over (the element is supplied
    /// per-lane by the map itself). Empty for a closed body.
    pub captures: Vec<ValueId>,
}

impl MnemonicKind for Map {
    fn opcode(&self) -> &'static str {
        "map"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        // `foo <$> arr` (Haskell `fmap`): apply the per-element body over the
        // array. Captures render as a partial application of the body —
        // `(foo c0 c1) <$> arr` — since the element is supplied by the map
        // itself, not written here.
        let body = Function::from_id(ctx, self.body).name();
        if self.captures.is_empty() {
            write!(f, "{} <$> {};", body, ValueRef::new(self.src, ctx))
        } else {
            write!(f, "({}", body)?;
            for &c in &self.captures {
                write!(f, " {}", ValueRef::new(c, ctx))?;
            }
            write!(f, ") <$> {};", ValueRef::new(self.src, ctx))
        }
    }

    fn args(&self) -> Args {
        let mut args: Args = SmallVec::with_capacity(1 + self.captures.len());
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

    /// A `map` renders as Haskell `fmap`: `@body <$> src` (and a partial
    /// application `(@body c0) <$> src` when it captures loop invariants).
    #[test]
    fn map_renders_as_fmap() {
        let mut tc = TestContext::new();
        let body = Function::make(&mut tc.ctx, "foo".into()).unwrap().id;
        let host = Function::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x2000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, host);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let i8 = tc.ctx.types.get_or_make_int(1);
        let array_ty = tc.ctx.types.get_or_make_array(i8, 8);
        let (src, cap) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            (b.push_param(8).id(), b.push_param(4).id())
        };
        if let ValueId::BlockParam(pid) = src {
            tc.ctx.values.block_params[pid].type_id = array_ty;
        }

        let plain = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_map(body, src, Vec::new()).id()
        };
        let ValueId::Instruction(plain_id) = plain else {
            unreachable!()
        };
        let rendered = tc.ctx.get_insn(plain_id).as_statement().to_string();
        assert!(
            rendered.contains("foo <$>"),
            "map renders as fmap, got: {rendered}"
        );

        let with_cap = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_map(body, src, vec![cap]).id()
        };
        let ValueId::Instruction(cap_id) = with_cap else {
            unreachable!()
        };
        let rendered = tc.ctx.get_insn(cap_id).as_statement().to_string();
        assert!(
            rendered.contains("(foo ") && rendered.contains(") <$>"),
            "a capturing map renders as a partial application, got: {rendered}"
        );
    }

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
        assert_eq!(
            m.args().to_vec(),
            vec![src, cap],
            "src then captures are the operands"
        );
        assert!(
            !m.args().contains(&ValueId::Function(body)),
            "body is not an operand"
        );

        // Result type is the array type of `src`.
        assert_eq!(tc.ctx.type_of(map_val), array_ty);

        // replace_value rewrites operands but never the body symbol.
        let new_src = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_param(20).id()
        };
        let mut rewritten = Mnemonic::Map(m);
        rewritten.replace_value(src, new_src);
        let Mnemonic::Map(r) = rewritten else {
            unreachable!()
        };
        assert_eq!(r.src, new_src);
        assert_eq!(r.body, body, "body symbol is untouched by replace_value");
        assert_eq!(r.captures, vec![cap]);
    }
}
