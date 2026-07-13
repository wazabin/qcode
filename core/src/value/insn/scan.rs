//! `scan`: a total left-scan (prefix fold) over an array value.
//!
//! [`Scan`] threads an accumulator left-to-right across every lane of the array
//! `src`, emitting the accumulator after each step: conceptually
//!
//! ```text
//!   acc_0   = init
//!   acc_i+1 = body(acc_i, src[i], captures…)
//!   out[i]  = acc_i+1
//! ```
//!
//! producing an array of the same length as `src`. It is the projectable
//! representation of a loop whose per-element write depends on the previous
//! iteration's result — `out[i] = f(out[i-1], i)` — which [`Map`](super::Map)
//! cannot express because its body is element-local. The MT19937 seeding loop
//! `mt[i] = 1812433253 * (mt[i-1] ^ (mt[i-1] >> 30)) + i` is the canonical case;
//! its `src` is `enumerate(arr)`, so the body's element is the `(index, elem)`
//! tuple and the `elem` half is simply unused.
//!
//! Like [`Map`](super::Map), `body` is a **function symbol** (not a value
//! operand), so `Scan` stays an ordinary first-order SSA instruction. Its value
//! operands are `init` and `src` plus any loop-invariant `captures` the body
//! closes over. The body is **binary in (accumulator, element)**: its first
//! parameter is the carried accumulator (typed as the result element), its second
//! is the lane element of `src`.

use crate::value::LocalValueId;

use super::{
    Callee,
    mnemonic::{Args, MnemonicKind},
};
use smallvec::SmallVec;

/// A total left-scan `out[i] = acc_i+1` where `acc_i+1 = body(acc_i, src[i],
/// captures…)` and `acc_0 = init`. The result is `[U; N]` where `N` is `src`'s
/// length and `U` is the body's return type (also the accumulator's type).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Scan {
    /// The pure per-element fold function, applied at each lane. A symbol, not an
    /// operand — exactly like a direct call's target. Binary in `(accumulator,
    /// element)`.
    pub body: Callee,
    /// The initial accumulator value (`acc_0`).
    pub init: LocalValueId,
    /// The array value scanned over.
    pub src: LocalValueId,
    /// Loop-invariant values the body closes over (the accumulator and element
    /// are supplied per-lane by the scan itself). Empty for a closed body.
    pub captures: Vec<LocalValueId>,
}

impl MnemonicKind for Scan {
    fn opcode(&self) -> &'static str {
        "scan"
    }

    fn args(&self) -> Args {
        let mut args = SmallVec::with_capacity(2 + self.captures.len());
        args.push(self.init);
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
            BasicBlock, FunctionBody, ValueId,
            insn::{Callee, Mnemonic, mnemonic::MnemonicKind},
        },
    };

    /// A `scan` renders as `scanl @body init src` (and a partial application
    /// `scanl (@body c0) init src` when it captures loop invariants).
    #[test]
    fn scan_renders_as_scanl() {
        let mut tc = TestContext::new();
        let body = FunctionBody::make(&mut tc.ctx, "foo".into()).unwrap().id;
        let host = FunctionBody::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x2000, host);
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, host);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let i32_ty = tc.ctx.shared.types.get_or_make_int(4);
        let array_ty = tc.ctx.shared.types.get_or_make_array(i32_ty, 8);
        let (init, src, cap) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            (
                b.push_param(4).id(),
                b.push_param(32).id(),
                b.push_param(4).id(),
            )
        };
        if let ValueId::BlockParam(pid) = src {
            tc.ctx.block_param_mut(pid).type_id = array_ty;
        }

        let plain = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_scan(body, init, src, Vec::new()).id()
        };
        let ValueId::Instruction(plain_id) = plain else {
            unreachable!()
        };
        let rendered = tc.ctx.get_insn(plain_id).as_statement().to_string();
        assert!(
            rendered.contains("scanl @foo"),
            "scan renders as scanl, got: {rendered}"
        );

        let with_cap = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_scan(body, init, src, vec![cap]).id()
        };
        let ValueId::Instruction(cap_id) = with_cap else {
            unreachable!()
        };
        let rendered = tc.ctx.get_insn(cap_id).as_statement().to_string();
        assert!(
            rendered.contains("scanl (@foo "),
            "a capturing scan renders as a partial application, got: {rendered}"
        );
    }

    /// `push_scan` yields an array-typed value whose operands are `init`, `src`,
    /// then captures — with `body` kept as a symbol, never an operand — and
    /// `replace_value` rewrites the operands but never the body.
    #[test]
    fn scan_builds_with_array_result_and_symbol_body() {
        let mut tc = TestContext::new();
        let body = FunctionBody::make(&mut tc.ctx, "body".into()).unwrap().id;
        let host = FunctionBody::make(&mut tc.ctx, "host".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000, host);
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, host);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let i32_ty = tc.ctx.shared.types.get_or_make_int(4);
        let array_ty = tc.ctx.shared.types.get_or_make_array(i32_ty, 20);
        let (init, src, cap) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            (
                b.push_param(4).id(),
                b.push_param(80).id(),
                b.push_param(4).id(),
            )
        };
        if let ValueId::BlockParam(pid) = src {
            tc.ctx.block_param_mut(pid).type_id = array_ty;
        }
        let scan_val = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_scan(body, init, src, vec![cap]).id()
        };
        let ValueId::Instruction(scan_id) = scan_val else {
            panic!("push_scan should yield an instruction value");
        };
        let m = match tc.ctx.get_insn(scan_id).mnemonic().clone() {
            Mnemonic::Scan(m) => m,
            other => panic!("expected Scan, got {other:?}"),
        };
        assert_eq!(m.body, Callee::Real(body));
        assert_eq!(
            m.args().to_vec(),
            vec![init.strip_func(), src.strip_func(), cap.strip_func()],
            "init, src, then captures are the operands"
        );
        assert!(
            !m.args().contains(&ValueId::Function(body).strip_func()),
            "body is not an operand"
        );
        // Result type is the array type of `src` (same length, body return elem).
        assert_eq!(tc.ctx.type_of(scan_val), array_ty);

        let new_src = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_param(80).id()
        };
        let mut rewritten = Mnemonic::Scan(m);
        rewritten.replace_value(src.strip_func(), new_src.strip_func());
        let Mnemonic::Scan(r) = rewritten else {
            unreachable!()
        };
        assert_eq!(r.src, new_src.strip_func());
        assert_eq!(
            r.body,
            Callee::Real(body),
            "body symbol is untouched by replace_value"
        );
        assert_eq!(r.init, init.strip_func());
        assert_eq!(r.captures, vec![cap.strip_func()]);
    }
}
