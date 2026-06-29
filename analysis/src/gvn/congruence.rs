//! Structural value-number congruence over expression trees.
//!
//! Builds a hash-consed symbolic form ([`SymId`]) for each value so that two
//! distinct `ValueId`s computing the same value unify. Arithmetic is flattened
//! through the affine [`NormalForm`](super::affine) (so `a + b`, `b + a` and a
//! recomputed `a + b` agree); other pure, re-executable ops (casts, variable
//! `mul`/`shl`, comparisons, intrinsics, geps, ...) become structural [`Sym::Op`]
//! nodes with commutative operands sorted; loads, calls, block params, varnodes
//! and literals are opaque identity leaves.
//!
//! Leaving memory ops (loads/calls) as per-`ValueId` leaves is the soundness
//! boundary: two syntactically identical loads are *not* congruent, because
//! intervening stores may give them different values. Only genuinely
//! re-derivable pure computation is unified. Leaves are injective — equal
//! [`SymId`] of two leaves implies the same `ValueId` — so a congruence between
//! two expressions means they are the same function of the *same* leaf values.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use qcode::{
    context::Context,
    value::{ValueId, insn::Mnemonic},
};

use super::affine::{NormalForm, Numbering};
use super::cse::is_commutative;

/// A sentinel operand used to blank out the `ValueId`s inside an [`Sym::Op`]
/// key, so the key carries only the op kind and immediates; the operands are
/// tracked separately as the node's child [`SymId`]s. Any fixed value works:
/// it never escapes the key.
fn operand_sentinel() -> ValueId {
    ValueId::Literal(0usize.into())
}

/// Interned structural-form id.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct SymId(u32);

#[derive(Clone, PartialEq, Eq, Hash)]
enum Sym {
    /// Opaque identity leaf: block params, varnodes, literals, loads, calls and
    /// any non-whitelisted op. Congruent only to the same `ValueId`.
    Leaf(ValueId),
    /// `constant + Σ coeffᵢ·termᵢ` at `width` bytes; terms sorted by `SymId`.
    Affine(usize, u64, Vec<(SymId, u64)>),
    /// A pure non-affine op: the mnemonic with operands blanked (carrying the op
    /// kind and immediates), the result width, and the operand sym ids in order
    /// (commutative binops sorted).
    Op(Mnemonic, usize, Vec<SymId>),
}

/// Hash-consing structural congruence over a fixed set of forms.
pub(crate) struct Congruence {
    forms: Numbering,
    interner: HashMap<Sym, SymId>,
    syms: Vec<Sym>,
    memo: HashMap<ValueId, SymId>,
    building: HashSet<ValueId>,
}

impl Congruence {
    /// Build a congruence engine over precomputed affine `forms` (see
    /// [`precompute_forms_for_blocks`](super::affine::precompute_forms_for_blocks)).
    pub(crate) fn new(forms: Numbering) -> Self {
        Congruence {
            forms,
            interner: HashMap::default(),
            syms: Vec::new(),
            memo: HashMap::default(),
            building: HashSet::default(),
        }
    }

    /// The structural id of `v`. Equal ids ⟺ provably the same value.
    pub(crate) fn id(&mut self, ctx: &Context, v: impl Into<ValueId>) -> SymId {
        let v = v.into();
        if let Some(&s) = self.memo.get(&v) {
            return s;
        }
        if !self.building.insert(v) {
            // A cycle (should not arise in acyclic SSA): treat as opaque.
            return self.intern(Sym::Leaf(v));
        }
        let s = self.compute(ctx, v);
        self.building.remove(&v);
        self.memo.insert(v, s);
        s
    }

    /// Whether `a` and `b` are structurally congruent.
    pub(crate) fn congruent(
        &mut self,
        ctx: &Context,
        a: impl Into<ValueId>,
        b: impl Into<ValueId>,
    ) -> bool {
        let (a, b) = (a.into(), b.into());
        a == b || self.id(ctx, a) == self.id(ctx, b)
    }

    fn intern(&mut self, s: Sym) -> SymId {
        if let Some(&id) = self.interner.get(&s) {
            return id;
        }
        let id = SymId(self.syms.len() as u32);
        self.syms.push(s.clone());
        self.interner.insert(s, id);
        id
    }

    fn compute(&mut self, ctx: &Context, v: ValueId) -> SymId {
        // Affine values flatten arithmetic structure (and unify reassociations).
        // A self-leaf (`1·v + 0`) means the op did not decompose — fall through
        // to op/leaf classification.
        if let Some(NormalForm::Affine {
            width,
            constant,
            terms,
        }) = self.forms.lookup_form(v).cloned()
            && !(constant == 0 && terms.len() == 1 && terms[0] == (v, 1))
        {
            let mut ts: Vec<(SymId, u64)> =
                terms.iter().map(|&(t, c)| (self.id(ctx, t), c)).collect();
            // Distinct term `ValueId`s may collapse to one `SymId` (congruent
            // terms): re-merge their coefficients so the form stays canonical.
            ts = merge_sym_terms(ts, width);
            return self.intern(Sym::Affine(width, constant, ts));
        }

        if let ValueId::Instruction(id) = v {
            let insn = ctx.get_insn(id);
            let mnemonic = insn.mnemonic().clone();
            let size = insn.size();
            if is_pure_value_op(&mnemonic) {
                let args = mnemonic.args();
                let mut arg_syms: Vec<SymId> = args.iter().map(|&a| self.id(ctx, a)).collect();
                if let Mnemonic::Binop(b) = &mnemonic
                    && is_commutative(&b.op)
                    && arg_syms.len() == 2
                {
                    arg_syms.sort_by_key(|s| s.0);
                }
                // Blank the operands so the key carries only kind + immediates.
                let mut key = mnemonic.clone();
                for a in args.iter().copied().collect::<HashSet<_>>() {
                    key.replace_value(a, operand_sentinel());
                }
                return self.intern(Sym::Op(key, size, arg_syms));
            }
        }

        self.intern(Sym::Leaf(v))
    }
}

/// Sum coefficients of terms that collapsed to the same `SymId`, drop zeros at
/// `width`, and sort by `SymId` for a canonical order.
fn merge_sym_terms(terms: Vec<(SymId, u64)>, width: usize) -> Vec<(SymId, u64)> {
    let m = mask_for(width);
    let mut acc: Vec<(SymId, u64)> = Vec::with_capacity(terms.len());
    for (s, c) in terms {
        match acc.iter_mut().find(|(ss, _)| *ss == s) {
            Some(e) => e.1 = e.1.wrapping_add(c) & m,
            None => acc.push((s, c & m)),
        }
    }
    acc.retain(|(_, c)| *c != 0);
    acc.sort_by_key(|(s, _)| s.0);
    acc
}

fn mask_for(width: usize) -> u64 {
    if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (width * 8)) - 1
    }
}

/// Whether `m` produces a value that is a pure, side-effect-free, deterministic
/// function of its operands — so two such instructions with congruent operands
/// compute the same value. Memory reads (`Load`), calls, stores, branches and
/// `PCodeOp`/`Map` are excluded (their value can depend on state we do not model
/// here, or they have effects).
fn is_pure_value_op(m: &Mnemonic) -> bool {
    matches!(
        m,
        Mnemonic::Unop(_)
            | Mnemonic::Binop(_)
            | Mnemonic::Range(_)
            | Mnemonic::IntToFloat(_)
            | Mnemonic::FloatToFloat(_)
            | Mnemonic::FloatToInt(_)
            | Mnemonic::Zext(_)
            | Mnemonic::Sext(_)
            | Mnemonic::IsFloatNaN(_)
            | Mnemonic::PopCount(_)
            | Mnemonic::LzCount(_)
            | Mnemonic::Carry(_)
            | Mnemonic::SCarry(_)
            | Mnemonic::SBorrow(_)
            | Mnemonic::Extract(_)
            | Mnemonic::Tuple(_)
            | Mnemonic::Intrinsic(_)
            | Mnemonic::Gep(_)
    )
}

#[cfg(test)]
mod tests {
    use super::super::affine::precompute_forms_for_blocks;
    use super::*;
    use qcode::value::{BlockId, FunctionId, FunctionRef};
    use qcode_macro::qcode;

    fn engine(ctx: &Context, f: FunctionId) -> Congruence {
        let blocks: Vec<BlockId> = FunctionRef::from_id(ctx, f)
            .blocks()
            .map(|b| b.id)
            .collect();
        Congruence::new(precompute_forms_for_blocks(ctx, &blocks))
    }

    /// `a + b` and `b + a` are congruent (commutative arithmetic, flattened
    /// through the affine form), while `a - b` and `b - a` are not.
    #[test]
    fn commutative_add_congruent_sub_not() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;
            fn f:
            <entry>
                %a = load(i64, &A);
                %b = load(i64, &B);
                %ab = %a + %b;
                %ba = %b + %a;
                %sub1 = %a - %b;
                %sub2 = %b - %a;
                return at 0x0;
            "
        );
        let mut e = engine(&ctx, f);
        assert!(e.congruent(&ctx, ab, ba), "a+b ≡ b+a");
        assert!(!e.congruent(&ctx, sub1, sub2), "a-b ≢ b-a");
    }

    /// A recomputed pure expression over the *same* leaves unifies even though
    /// it is a distinct instruction (`%ab1` and `%ab2` are different ValueIds).
    #[test]
    fn recomputed_affine_unifies() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;
            fn f:
            <entry>
                %a = load(i64, &A);
                %b = load(i64, &B);
                %ab1 = %a + %b;
                %ab2 = %a + %b;
                return at 0x0;
            "
        );
        let mut e = engine(&ctx, f);
        assert_ne!(ab1, ab2, "distinct instructions");
        assert!(e.congruent(&ctx, ab1, ab2), "recomputed a+b unifies");
    }

    /// Two distinct loads of the same address are NOT congruent: a load is an
    /// identity leaf, because an intervening store could change its value. This
    /// is the soundness boundary.
    #[test]
    fn distinct_loads_not_congruent() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            fn f:
            <entry>
                %c1 = load(i64, &A);
                %c2 = load(i64, &A);
                return at 0x0;
            "
        );
        let mut e = engine(&ctx, f);
        assert_ne!(c1, c2);
        assert!(
            !e.congruent(&ctx, c1, c2),
            "two loads of &A are not congruent"
        );
    }

    /// A pure non-affine op (`zext`) recomputed over the same operand unifies,
    /// but `zext` of two different operands does not.
    #[test]
    fn pure_cast_unifies_by_operand() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 A;
            varnode i32 B;
            fn f:
            <entry>
                %a = load(i32, &A);
                %b = load(i32, &B);
                %za1 = zext(i64, %a);
                %za2 = zext(i64, %a);
                %zb = zext(i64, %b);
                return at 0x0;
            "
        );
        let mut e = engine(&ctx, f);
        assert!(e.congruent(&ctx, za1, za2), "zext(a) recomputed unifies");
        assert!(!e.congruent(&ctx, za1, zb), "zext(a) ≢ zext(b)");
    }

    /// Memory edge case: identical *arithmetic over distinct loads* of the same
    /// address must NOT unify. `%c1 + 1` and `%c2 + 1` share the same affine
    /// shape, but their leaf loads are distinct instructions, so the affine forms
    /// carry different leaf ids and stay incongruent. (If loads were structural
    /// nodes keyed by address, this would wrongly collapse.)
    #[test]
    fn arithmetic_over_distinct_loads_not_congruent() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            fn f:
            <entry>
                %c1 = load(i64, &A);
                %c2 = load(i64, &A);
                %p1 = %c1 + 0x1;
                %p2 = %c2 + 0x1;
                return at 0x0;
            "
        );
        let mut e = engine(&ctx, f);
        assert!(e.congruent(&ctx, p1, p1), "reflexive");
        assert!(
            !e.congruent(&ctx, p1, p2),
            "c1+1 ≢ c2+1 (distinct load leaves)"
        );
    }

    /// The same load reused (same ValueId) *is* congruent through arithmetic:
    /// `%c + 1` built twice off one load unifies, because the leaf is one SSA
    /// value. This is the case the pass relies on to collapse loop-invariants.
    #[test]
    fn arithmetic_over_same_load_congruent() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            fn f:
            <entry>
                %c = load(i64, &A);
                %p1 = %c + 0x1;
                %p2 = %c + 0x1;
                return at 0x0;
            "
        );
        let mut e = engine(&ctx, f);
        assert_ne!(p1, p2);
        assert!(e.congruent(&ctx, p1, p2), "c+1 over one load unifies");
    }

    /// Variable `mul` is commutative and congruent under swap; the congruence
    /// also composes — `(a*b)+c` ≡ `c+(b*a)`.
    #[test]
    fn variable_mul_commutes_and_composes() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;
            varnode i64 C;
            fn f:
            <entry>
                %a = load(i64, &A);
                %b = load(i64, &B);
                %c = load(i64, &C);
                %m1 = %a * %b;
                %m2 = %b * %a;
                %e1 = %m1 + %c;
                %e2 = %c + %m2;
                return at 0x0;
            "
        );
        let mut e = engine(&ctx, f);
        assert!(e.congruent(&ctx, m1, m2), "a*b ≡ b*a");
        assert!(e.congruent(&ctx, e1, e2), "(a*b)+c ≡ c+(b*a)");
    }
}
