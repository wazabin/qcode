//! MBA (mixed boolean-arithmetic) simplification.
//!
//! De-obfuscates strength-reduced / MBA-obfuscated integer arithmetic back to a
//! canonical form so a lifted loop body reads like the source computation
//! instead of a long `*2 / ~ / & / |` bit-trick blob. This is pure legibility +
//! enabling-of-GVN; it does no control-flow or memory work.
//!
//! Rather than carry a hand-written list of rewrite patterns, this pass bridges
//! qcode to the [`rumba_core`] MBA *solver*: it re-derives the simplest form of
//! an expression from its semantics (truth-table / polynomial reduction), so it
//! collapses arbitrary obfuscations of the same family — `a + ~(a*2) ⇒ ~a`,
//! `(a&b)(a|b) + (a&~b)(~a&b) ⇒ a*b`, and anything else algebraically equal —
//! not just two fixed shapes.
//!
//! # Pipeline
//!
//! For each *maximal* MBA-rooted subgraph in the function:
//!
//! 1. **Extract** the subgraph into a [`rumba_core::expr::Expr`]. Operators in
//!    `{+, -, *, &, |, ^, ~, neg, <<-by-constant}` become `Expr` nodes; constant
//!    literals become `Const`; everything else (loads, block params, shifts by a
//!    variable, width-changing `sext`/`range`, …) is abstracted as an opaque
//!    `Var`, interned so repeated uses of the same value share one variable.
//!    Only a region that genuinely *mixes* arithmetic (`+ − × neg <<`) and
//!    boolean (`& | ^ ~`) operators is handed to the solver — a pure-arithmetic
//!    tree is const_fold/gvn's job, and a pure-bitwise tree has no MBA to undo.
//! 2. **Solve** with [`simplify_mba`].
//! 3. **Rebuild** `~`, `^` and `|`. The solver only ever returns `{+, *, &}`
//!    (plus `Scale`/`Const`/`Var`), encoding the others arithmetically
//!    (`~x = -1 + -1·x`, `x|y = x + y - (x&y)`, `x^y = x + y - 2(x&y)`); a small
//!    bottom-up matcher folds those back for legibility. See [`canonicalize`].
//! 4. **Emit** the rebuilt graph immediately before the root and forward the
//!    root's uses to it — but only when the result is strictly cheaper than the
//!    region it replaces. The now-dead region is pruned in place.
//!
//! # Soundness & width
//!
//! [`rumba_core`] works over fixed-width two's-complement `iN` with `N ≤ 64`;
//! all rules are exact under wrapping (no signed/overflow caveats). A region is
//! processed at a *single* width — the root's — and any operand of a different
//! width, or a width-changing op, is taken as an opaque leaf. In particular the
//! 32→64 widened-multiply idiom (`(sext·sext)[0:4]`) is left to a separate
//! narrowing canonicalization to expose; here it simply reads as opaque leaves.

use rustc_hash::FxHashMap as HashMap;

use qcode::{
    context::Context,
    value::{
        BasicBlock, Function, FunctionId, Instruction, InstructionRef, Value, ValueId, ValueRef,
        block::BlockId,
        insn::{Binary, Binop, InstructionId, IntBinop, Mnemonic, Unary, Unop},
    },
};

use rumba_core::{
    expr::{Expr, VarId},
    simplify::simplify_mba,
    varint::{VarInt, make_mask},
};

use crate::{FunctionPass, PipelineEnv};

#[derive(Default)]
pub struct MbaSimplify;

impl FunctionPass for MbaSimplify {
    const NAME: &'static str = "mba_simplify";

    fn description(&self) -> &'static str {
        "De-obfuscate integer MBA expressions via the rumba solver"
    }

    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        Ok(mba_simplify(ctx, fun_id))
    }
}

crate::register_function_pass!(MbaSimplify);

/// Maximum width rumba can model. Wider roots are skipped.
const MAX_WIDTH_BYTES: usize = 8;

pub fn mba_simplify(ctx: &mut Context, fun: FunctionId) -> bool {
    let roots: Vec<InstructionId> = Function::from_id(ctx, fun)
        .blocks()
        .flat_map(|b| b.instruction_ids().to_vec())
        .filter(|&iid| is_root(ctx, iid))
        .collect();

    let mut changed = false;
    for root in roots {
        changed |= try_simplify_root(ctx, root);
    }
    changed
}

/// A *maximal* MBA instruction: one that is live and is not a single-use
/// operand of another same-width MBA instruction (which would subsume it).
fn is_root(ctx: &Context, iid: InstructionId) -> bool {
    if !is_mba_insn(ctx, iid) {
        return false;
    }
    let users = ctx.users(ValueId::Instruction(iid));
    if users.is_empty() {
        return false; // dead; leave for DCE / a prior root's prune
    }
    if users.len() == 1
        && is_mba_insn(ctx, users[0])
        && insn_size(ctx, users[0]) == insn_size(ctx, iid)
    {
        return false; // subsumed by its parent region
    }
    true
}

fn try_simplify_root(ctx: &mut Context, root: InstructionId) -> bool {
    let size = insn_size(ctx, root);
    if size == 0 || size > MAX_WIDTH_BYTES {
        return false;
    }
    let n = (size * 8) as u8;
    let mask = make_mask(n);

    // 1. Classify the region cheaply first — no rumba `Expr` is built unless
    //    this is a genuine MBA: a *mix* of arithmetic and boolean ops, and more
    //    than a lone instruction.
    let mut c = Classify {
        ctx,
        region: size,
        has_arith: false,
        has_bool: false,
        count: 0,
    };
    c.visit(root);
    let region_cost = c.count;
    if region_cost <= 1 || !(c.has_arith && c.has_bool) {
        return false;
    }

    // 2. Build the rumba `Expr` (immutable borrow, scoped so `ctx` is free after).
    let (raw, leaves) = {
        let mut ex = Extract {
            ctx: &*ctx,
            region: size,
            mask,
            leaves: Vec::new(),
            index: HashMap::default(),
        };
        let raw = ex.expand(root);
        (raw, ex.leaves)
    };

    // 3 + 4. Solve, then rebuild ~/^/|.
    let pretty = canonicalize(simplify_mba(raw, n), mask);

    // 4. Only commit when the result is strictly cheaper.
    if cost(&pretty, mask) >= region_cost {
        return false;
    }
    let block = Instruction::from_id(ctx, root)
        .parent()
        .expect("a root instruction lives in a block")
        .id;
    let new_val = emit(ctx, &pretty, &leaves, size, mask, root, block);
    ctx.replace_all_uses_with(root, new_val);
    prune_dead(ctx, root);
    true
}

// --- qcode subgraph -> rumba Expr ------------------------------------------

struct Extract<'a, 'str> {
    ctx: &'a Context<'str>,
    /// Region width in bytes; nodes of other widths are taken as opaque leaves.
    region: usize,
    mask: u64,
    /// Opaque leaves, indexed by `VarId`.
    leaves: Vec<ValueId>,
    index: HashMap<ValueId, usize>,
}

impl Extract<'_, '_> {
    fn leaf(&mut self, v: ValueId) -> Expr {
        let idx = *self.index.entry(v).or_insert_with(|| {
            let i = self.leaves.len();
            self.leaves.push(v);
            i
        });
        Expr::Var(VarId(idx))
    }

    /// Translate an operand: a constant, an inlinable MBA child, or a leaf.
    fn child(&mut self, v: ValueId) -> Expr {
        if let Some(c) = numeric_const(self.ctx, v) {
            return Expr::Const(VarInt::from(c & self.mask));
        }
        match inlinable_child(self.ctx, self.region, v) {
            Some(iid) => self.expand(iid),
            None => self.leaf(v),
        }
    }

    /// Translate an MBA instruction's whole subtree.
    fn expand(&mut self, iid: InstructionId) -> Expr {
        match Instruction::from_id(self.ctx, iid).mnemonic() {
            Mnemonic::Binop(b) => {
                let (lhs, rhs) = (b.lhs, b.rhs);
                match int_op(&b.op).expect("is_mba_insn gates the opset") {
                    IntBinop::Add => Expr::Add(vec![self.child(lhs), self.child(rhs)]),
                    IntBinop::Sub => {
                        let l = self.child(lhs);
                        let r = self.child(rhs);
                        Expr::Add(vec![l, Expr::Scale(VarInt::MAX, Box::new(r))])
                    }
                    IntBinop::Mul => Expr::Mul(vec![self.child(lhs), self.child(rhs)]),
                    IntBinop::And => Expr::And(vec![self.child(lhs), self.child(rhs)]),
                    IntBinop::Or => Expr::Or(vec![self.child(lhs), self.child(rhs)]),
                    IntBinop::Xor => Expr::Xor(vec![self.child(lhs), self.child(rhs)]),
                    IntBinop::ShiftLeft => {
                        // `x << c` with constant `c < width` is `x * 2^c`.
                        let c = numeric_const(self.ctx, rhs).expect("shl gate ensures const");
                        let factor = 1u64.wrapping_shl(c as u32) & self.mask;
                        Expr::Scale(VarInt::from(factor), Box::new(self.child(lhs)))
                    }
                    _ => unreachable!("is_mba_insn gates the opset"),
                }
            }
            Mnemonic::Unop(u) => {
                let src = u.src;
                match u.op {
                    Unop::IntNot => Expr::Not(Box::new(self.child(src))),
                    Unop::IntNegate => Expr::Scale(VarInt::MAX, Box::new(self.child(src))),
                    _ => unreachable!("is_mba_insn gates the opset"),
                }
            }
            _ => unreachable!("is_mba_insn gates the opset"),
        }
    }
}

// --- region classification (no `Expr` built) -------------------------------

/// Which half of the MBA an operator belongs to.
enum OpClass {
    Arith,
    Bool,
}

/// Walks the same region [`Extract`] would, but only tallies operator classes
/// and instruction count — so a non-MBA region is rejected before any rumba
/// `Expr` is allocated.
struct Classify<'a, 'str> {
    ctx: &'a Context<'str>,
    region: usize,
    has_arith: bool,
    has_bool: bool,
    count: usize,
}

impl Classify<'_, '_> {
    fn visit(&mut self, iid: InstructionId) {
        self.count += 1;
        match mba_class(self.ctx, iid) {
            Some(OpClass::Arith) => self.has_arith = true,
            Some(OpClass::Bool) => self.has_bool = true,
            None => {}
        }
        for op in Instruction::from_id(self.ctx, iid).mnemonic().args() {
            if numeric_const(self.ctx, op).is_some() {
                continue;
            }
            if let Some(child) = inlinable_child(self.ctx, self.region, op) {
                self.visit(child);
            }
        }
    }
}

// --- rebuild ~, ^, | from the solver's {+, *, &} output --------------------

/// Strip the singleton wrappers the solver emits (`And([x]) == x`, etc.).
fn unwrap_singletons(e: Expr) -> Expr {
    let e = e.map(unwrap_singletons);
    match e {
        Expr::And(ref v)
        | Expr::Or(ref v)
        | Expr::Xor(ref v)
        | Expr::Add(ref v)
        | Expr::Mul(ref v)
            if v.len() == 1 =>
        {
            v[0].clone()
        }
        other => other,
    }
}

fn scale_parts(e: &Expr, mask: u64) -> Option<(u64, &Expr)> {
    match e {
        Expr::Scale(c, inner) => Some((c.get(mask), inner)),
        _ => None,
    }
}

/// Bottom-up matcher folding the solver's arithmetic encodings back into `~`,
/// `|` and `^`. `mask == -1` and `mask - 1 == -2` at this width.
fn rebuild_bitops(e: Expr, mask: u64) -> Expr {
    let e = e.map(|c| rebuild_bitops(c, mask));
    let Expr::Add(terms) = &e else { return e };

    // ~X  ==  -1 + (-1)·X
    if terms.len() == 2 {
        for i in 0..2 {
            let Expr::Const(c) = &terms[i] else { continue };
            if c.get(mask) != mask {
                continue;
            }
            if let Some((coeff, x)) = scale_parts(&terms[1 - i], mask)
                && coeff == mask
            {
                return Expr::Not(Box::new(x.clone()));
            }
        }
    }

    // X|Y == X + Y - (X&Y)   ;   X^Y == X + Y - 2(X&Y)
    if terms.len() == 3 {
        for i in 0..3 {
            let Some((coeff, inner)) = scale_parts(&terms[i], mask) else {
                continue;
            };
            let Expr::And(ab) = inner else { continue };
            if ab.len() != 2 {
                continue;
            }
            let others: Vec<&Expr> = (0..3).filter(|&j| j != i).map(|j| &terms[j]).collect();
            let matches_and = (others[0] == &ab[0] && others[1] == &ab[1])
                || (others[0] == &ab[1] && others[1] == &ab[0]);
            if !matches_and {
                continue;
            }
            let (a, b) = (ab[0].clone(), ab[1].clone());
            if coeff == mask {
                return Expr::Or(vec![a, b]);
            }
            if coeff == mask.wrapping_sub(1) {
                return Expr::Xor(vec![a, b]);
            }
        }
    }
    e
}

/// Run [`unwrap_singletons`] + [`rebuild_bitops`] to a fixpoint.
fn canonicalize(e: Expr, mask: u64) -> Expr {
    let mut cur = unwrap_singletons(e);
    for _ in 0..16 {
        let next = unwrap_singletons(rebuild_bitops(cur.clone(), mask));
        if next == cur {
            break;
        }
        cur = next;
    }
    cur
}

/// Number of qcode instructions [`emit`] would materialize for `e`.
fn cost(e: &Expr, mask: u64) -> usize {
    match e {
        Expr::Var(_) | Expr::Const(_) => 0,
        Expr::Not(x) => 1 + cost(x, mask),
        Expr::Scale(c, x) => match c.get(mask) {
            0 => 0,
            1 => cost(x, mask),
            _ => 1 + cost(x, mask),
        },
        Expr::And(v) | Expr::Or(v) | Expr::Xor(v) | Expr::Add(v) | Expr::Mul(v) => {
            v.iter().map(|c| cost(c, mask)).sum::<usize>() + v.len().saturating_sub(1)
        }
    }
}

// --- rumba Expr -> qcode subgraph ------------------------------------------

#[allow(clippy::too_many_arguments)]
fn emit(
    ctx: &mut Context,
    e: &Expr,
    leaves: &[ValueId],
    size: usize,
    mask: u64,
    before: InstructionId,
    block: BlockId,
) -> ValueId {
    match e {
        Expr::Var(VarId(i)) => leaves[*i],
        Expr::Const(c) => ctx.get_const(c.get(mask), size).id(),
        Expr::Not(x) => {
            let xv = emit(ctx, x, leaves, size, mask, before, block);
            push_insn(
                ctx,
                Mnemonic::Unop(Unary {
                    op: Unop::IntNot,
                    src: xv,
                }),
                size,
                before,
                block,
            )
        }
        Expr::Scale(c, x) => {
            let cv = c.get(mask);
            if cv == 0 {
                return ctx.get_const(0, size).id();
            }
            let xv = emit(ctx, x, leaves, size, mask, before, block);
            if cv == 1 {
                return xv;
            }
            let cval = ctx.get_const(cv, size).id();
            push_binop(ctx, IntBinop::Mul, cval, xv, size, before, block)
        }
        Expr::And(v) => fold_emit(ctx, v, IntBinop::And, leaves, size, mask, before, block),
        Expr::Or(v) => fold_emit(ctx, v, IntBinop::Or, leaves, size, mask, before, block),
        Expr::Xor(v) => fold_emit(ctx, v, IntBinop::Xor, leaves, size, mask, before, block),
        Expr::Add(v) => fold_emit(ctx, v, IntBinop::Add, leaves, size, mask, before, block),
        Expr::Mul(v) => fold_emit(ctx, v, IntBinop::Mul, leaves, size, mask, before, block),
    }
}

#[allow(clippy::too_many_arguments)]
fn fold_emit(
    ctx: &mut Context,
    operands: &[Expr],
    op: IntBinop,
    leaves: &[ValueId],
    size: usize,
    mask: u64,
    before: InstructionId,
    block: BlockId,
) -> ValueId {
    if operands.is_empty() {
        // Empty n-ary node = the operator's identity element.
        let id = match op {
            IntBinop::And => mask,
            IntBinop::Mul => 1,
            _ => 0,
        };
        return ctx.get_const(id, size).id();
    }
    let mut acc = emit(ctx, &operands[0], leaves, size, mask, before, block);
    for e in &operands[1..] {
        let rhs = emit(ctx, e, leaves, size, mask, before, block);
        acc = push_binop(ctx, op, acc, rhs, size, before, block);
    }
    acc
}

fn push_binop(
    ctx: &mut Context,
    op: IntBinop,
    lhs: ValueId,
    rhs: ValueId,
    size: usize,
    before: InstructionId,
    block: BlockId,
) -> ValueId {
    push_insn(
        ctx,
        Mnemonic::Binop(Binary {
            op: Binop::Int(op),
            lhs,
            rhs,
        }),
        size,
        before,
        block,
    )
}

fn push_insn(
    ctx: &mut Context,
    mnemonic: Mnemonic,
    size: usize,
    before: InstructionId,
    block: BlockId,
) -> ValueId {
    let id = InstructionRef::from_mnemonic(ctx, mnemonic, size).id;
    BasicBlock::from_id_mut(ctx, block).insert_insn_before(before, id);
    ValueId::Instruction(id)
}

/// Remove `iid` and any operand subtree that becomes userless once it is gone.
fn prune_dead(ctx: &mut Context, iid: InstructionId) {
    if !ctx.users(ValueId::Instruction(iid)).is_empty() {
        return;
    }
    let operands = Instruction::from_id(ctx, iid).mnemonic().args();
    ctx.remove_instruction(iid);
    for op in operands {
        if let ValueId::Instruction(child) = op {
            prune_dead(ctx, child);
        }
    }
}

// --- predicates / accessors ------------------------------------------------

fn int_op(op: &Binop) -> Option<IntBinop> {
    match op {
        Binop::Int(o) => Some(*o),
        _ => None,
    }
}

/// An MBA child of `v` to inline (instruction, same width, modelable op, used
/// only here), or `None` if `v` should be an opaque leaf. Callers handle
/// constants before reaching this.
fn inlinable_child(ctx: &Context, region: usize, v: ValueId) -> Option<InstructionId> {
    if let ValueId::Instruction(iid) = v
        && value_size(ctx, v) == region
        && is_mba_insn(ctx, iid)
        && ctx.users(v).len() == 1
    {
        return Some(iid);
    }
    None
}

/// The MBA half an instruction's operator belongs to (`None` if not modelable).
fn mba_class(ctx: &Context, iid: InstructionId) -> Option<OpClass> {
    match Instruction::from_id(ctx, iid).mnemonic() {
        Mnemonic::Binop(b) => match int_op(&b.op)? {
            IntBinop::Add | IntBinop::Sub | IntBinop::Mul | IntBinop::ShiftLeft => {
                Some(OpClass::Arith)
            }
            IntBinop::And | IntBinop::Or | IntBinop::Xor => Some(OpClass::Bool),
            _ => None,
        },
        Mnemonic::Unop(u) => match u.op {
            Unop::IntNegate => Some(OpClass::Arith),
            Unop::IntNot => Some(OpClass::Bool),
            _ => None,
        },
        _ => None,
    }
}

/// Is `iid` an integer op this pass can model? `shl` qualifies only with a
/// constant shift below the operand width (so it is exactly `x * 2^c`).
fn is_mba_insn(ctx: &Context, iid: InstructionId) -> bool {
    match Instruction::from_id(ctx, iid).mnemonic() {
        Mnemonic::Binop(b) => match b.op {
            Binop::Int(o) => match o {
                IntBinop::Add
                | IntBinop::Sub
                | IntBinop::Mul
                | IntBinop::And
                | IntBinop::Or
                | IntBinop::Xor => true,
                IntBinop::ShiftLeft => {
                    let bits = (insn_size(ctx, iid) * 8) as u64;
                    matches!(numeric_const(ctx, b.rhs), Some(c) if c < bits)
                }
                _ => false,
            },
            _ => false,
        },
        Mnemonic::Unop(u) => matches!(u.op, Unop::IntNot | Unop::IntNegate),
        _ => false,
    }
}

/// The constant value of `v`, if it is a plain (non-symbolic) integer literal.
fn numeric_const(ctx: &Context, v: ValueId) -> Option<u64> {
    if let ValueId::Literal(id) = v {
        let lit = &ctx.values.literals[id];
        if lit.symbolic.is_none() {
            return Some(lit.value);
        }
    }
    None
}

fn insn_size(ctx: &Context, iid: InstructionId) -> usize {
    Instruction::from_id(ctx, iid).size()
}

fn value_size(ctx: &Context, v: ValueId) -> usize {
    ValueRef::new(v, ctx).size()
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::value::insn::Mnemonic;
    use qcode_emulator::{SizedValue, StandaloneEmulator};
    use qcode_macro::qcode;

    fn return_value(ctx: &Context, fun: FunctionId) -> ValueId {
        let root = Function::from_id(ctx, fun).root().expect("root").id;
        let &term = BasicBlock::from_id(ctx, root)
            .instruction_ids()
            .last()
            .expect("terminator");
        match Instruction::from_id(ctx, term).mnemonic() {
            Mnemonic::ReturnValue(r) => r.value,
            other => panic!("expected return, got {other:?}"),
        }
    }

    fn run(ctx: &Context, fun: FunctionId, a: u64, b: u64) -> Option<u64> {
        let root = Function::from_id(ctx, fun).root().expect("root").id;
        let ret = return_value(ctx, fun);
        let mut emu = StandaloneEmulator::new(root);
        emu.run_pure(
            ctx,
            fun,
            &[SizedValue::new(a, 4), SizedValue::new(b, 4)],
            100_000,
        )
        .expect("runs");
        emu.get_value(ctx, ret)
    }

    fn insn_total(ctx: &Context, fun: FunctionId) -> usize {
        Function::from_id(ctx, fun)
            .blocks()
            .map(|b| b.instruction_ids().len())
            .sum()
    }

    /// Emulate on a spread of input pairs.
    fn sample(ctx: &Context, fun: FunctionId) -> Vec<Option<u64>> {
        [
            (5, 3),
            (0, 0),
            (0xdead, 0xbeef),
            (1, 0xffff_ffff),
            (0x1234_5678, 9),
        ]
        .into_iter()
        .map(|(a, b)| run(ctx, fun, a, b))
        .collect()
    }

    #[test]
    fn collapses_r1_not() {
        // a + ~(a*2)  ==  ~a
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda r1:
            <entry @a:i32 @b:i32>
                %m = @a * 0x2;
                %n = ~ %m;
                %r = @a + %n;
                return %r;
            "
        );
        let before = sample(&ctx, r1);
        let n_before = insn_total(&ctx, r1);

        assert!(mba_simplify(&mut ctx, r1));

        // Return is now a single `~a`.
        let ret = return_value(&ctx, r1);
        let ValueId::Instruction(iid) = ret else {
            panic!("expected insn")
        };
        assert!(matches!(
            Instruction::from_id(&ctx, iid).mnemonic(),
            Mnemonic::Unop(Unary {
                op: Unop::IntNot,
                ..
            })
        ));
        assert!(insn_total(&ctx, r1) < n_before);
        assert_eq!(sample(&ctx, r1), before);
    }

    #[test]
    fn collapses_r2_multiply() {
        // (a&b)(a|b) + (a&~b)(~a&b)  ==  a*b
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda r2:
            <entry @a:i32 @b:i32>
                %ab = @a & @b;
                %aob = @a | @b;
                %p1 = %ab * %aob;
                %nb = ~ @b;
                %na = ~ @a;
                %x = @a & %nb;
                %y = %na & @b;
                %p2 = %x * %y;
                %r = %p1 + %p2;
                return %r;
            "
        );
        let before = sample(&ctx, r2);
        let n_before = insn_total(&ctx, r2);

        assert!(mba_simplify(&mut ctx, r2));

        let ret = return_value(&ctx, r2);
        let ValueId::Instruction(iid) = ret else {
            panic!("expected insn")
        };
        assert!(matches!(
            Instruction::from_id(&ctx, iid).mnemonic(),
            Mnemonic::Binop(Binary {
                op: Binop::Int(IntBinop::Mul),
                ..
            })
        ));
        assert!(insn_total(&ctx, r2) < n_before);
        assert_eq!(sample(&ctx, r2), before);
        // Spot-check the algebra: 5 * 3 == 15.
        assert_eq!(run(&ctx, r2, 5, 3), Some(15));
    }

    #[test]
    fn collapses_constant_multiply() {
        // The MT seeder's disguised constant multiply: same R2 shape with one
        // operand the constant K and ~K precomputed as a literal.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda r2k:
            <entry @a:i32 @b:i32>
                %ak = @a & 0x6c078965;
                %aok = @a | 0x6c078965;
                %p1 = %ak * %aok;
                %ank = @a & 0x93f8769a;
                %na = ~ @a;
                %nak = %na & 0x6c078965;
                %p2 = %ank * %nak;
                %r = %p1 + %p2;
                return %r;
            "
        );
        let before = sample(&ctx, r2k);
        assert!(mba_simplify(&mut ctx, r2k));
        assert_eq!(sample(&ctx, r2k), before);
        // a * K at 32 bits.
        let k = 0x6c07_8965u64;
        assert_eq!(run(&ctx, r2k, 7, 0), Some((7 * k) & 0xffff_ffff));
    }

    #[test]
    fn leaves_lone_op_untouched() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda single:
            <entry @a:i32 @b:i32>
                %r = @a & @b;
                return %r;
            "
        );
        assert!(!mba_simplify(&mut ctx, single));
    }

    #[test]
    fn leaves_pure_arithmetic_untouched() {
        // (a + b) * a — no boolean op, so not an MBA: const_fold/gvn's job.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda arith:
            <entry @a:i32 @b:i32>
                %s = @a + @b;
                %r = %s * @a;
                return %r;
            "
        );
        assert!(!mba_simplify(&mut ctx, arith));
    }

    #[test]
    fn leaves_pure_bitwise_untouched() {
        // (a & b) | ~a — no arithmetic op, so nothing for the MBA solver to undo.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda bits:
            <entry @a:i32 @b:i32>
                %ab = @a & @b;
                %na = ~ @a;
                %r = %ab | %na;
                return %r;
            "
        );
        assert!(!mba_simplify(&mut ctx, bits));
    }
}
