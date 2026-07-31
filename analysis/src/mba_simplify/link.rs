//! Constant linking — a preprocessing step for the rumba solver.
//!
//! Rewrites the magic constants appearing in bitwise contexts as bitwise
//! formulas over a minimal shared *symbolic* basis, so that the solver can
//! expose and cancel ties between constants that are not independent (e.g. `c`
//! and `c | 1`). rumba handles a complement pair (`c` and `~c`) internally; this
//! generalises that to any tie, and a complement pair is just the `r == 1` case
//! of the construction below.
//!
//! A constant is a subset of bit positions. The bitwise expressions over a set
//! of constants form a Boolean algebra whose atoms are the coarsest partition of
//! bit positions on which every constant is constant (the *background* atom,
//! bits set in no constant, is included when some bit is uncovered). With `G`
//! atoms, a base of `r = ceil(log2 G)` masks addresses all of them; each atom
//! gets an `r`-bit code and each base mask ORs the atoms whose code has that bit
//! set. Every original constant is then the OR of its atoms, i.e. a purely
//! bitwise formula over the `r` base masks.
//!
//! The base masks are represented as *fresh variables* so the reconstruction
//! does not constant-fold straight back to the original magic numbers. This is
//! lossless precisely because the base is an independent basis: treating
//! independent masks as free variables loses no valid simplification. The
//! concrete mask values are substituted back once the solver is done.
//!
//! This lives here rather than in `rumba-core` because it is not part of the
//! published solver: [`link_constants`] runs on the `Expr` on its way *into*
//! [`simplify_mba_cached`](rumba_core::simplify::simplify_mba_cached), and
//! [`substitute`] + [`fold_consts`] run on the way back out.

use rustc_hash::FxHashMap as HashMap;

use rumba_core::expr::{Expr, VarId};

use super::make_mask;

/// The result of a successful linking: the rewritten expression (over fresh
/// base variables) and the map from those variables to their concrete masks.
pub(super) struct Linked {
    pub expr: Expr,
    pub subs: HashMap<VarId, u64>,
}

/// Collects the values of every constant appearing as an operand of a bitwise
/// operator (`&`, `|`, `^`, `~`). Constants in arithmetic position (operands of
/// `+`, `*`, scaling) are left untouched.
fn collect_bitwise_consts(e: &Expr, parent_bitwise: bool, mask: u64, out: &mut Vec<u64>) {
    match e {
        Expr::Const(c) => {
            if parent_bitwise {
                out.push(c & mask);
            }
        }
        Expr::Var(_) => {}
        Expr::Not(x) => collect_bitwise_consts(x, true, mask, out),
        Expr::Scale(_, x) => collect_bitwise_consts(x, false, mask, out),
        Expr::And(xs) | Expr::Or(xs) | Expr::Xor(xs) => {
            for x in xs {
                collect_bitwise_consts(x, true, mask, out);
            }
        }
        Expr::Add(xs) | Expr::Mul(xs) => {
            for x in xs {
                collect_bitwise_consts(x, false, mask, out);
            }
        }
    }
}

/// Computes the atom partition: groups bit positions `0..n` by their signature
/// across the constants. The signature of position `p` packs bit `p` of each
/// constant into a `u64` (hence the `m <= 64` requirement). Returns, for each
/// distinct signature, the mask of bit positions carrying it.
fn compute_atoms(consts: &[u64], n: u8) -> HashMap<u64, u64> {
    let mut atoms: HashMap<u64, u64> = HashMap::default();
    for p in 0..n {
        let mut sig = 0u64;
        for (i, &c) in consts.iter().enumerate() {
            sig |= ((c >> p) & 1) << i;
        }
        *atoms.entry(sig).or_insert(0) |= 1u64 << p;
    }
    atoms
}

/// `ceil(log2 g)` for `g >= 1`.
fn base_size(g: usize) -> usize {
    if g <= 1 {
        0
    } else {
        (g as u64).next_power_of_two().trailing_zeros() as usize
    }
}

/// Builds the bitwise formula reconstructing constant index `i` from the base
/// variables: the OR, over the atoms whose signature includes `i`, of the
/// conjunction of the base variables selected by that atom's code.
fn build_formula(items: &[(u64, u64)], i: usize, base_vars: &[VarId]) -> Expr {
    let mut terms = vec![];

    for (idx, (sig, _atom)) in items.iter().enumerate() {
        if (sig >> i) & 1 != 1 {
            continue;
        }

        // The atom's code is its index; literal j is b_j or ~b_j.
        let lits: Vec<Expr> = base_vars
            .iter()
            .enumerate()
            .map(|(j, &v)| {
                let lit = Expr::Var(v);
                if (idx >> j) & 1 == 1 { lit } else { !lit }
            })
            .collect();

        terms.push(unwrap_one(lits, Expr::And));
    }

    match terms.len() {
        0 => Expr::zero(),
        _ => unwrap_one(terms, Expr::Or),
    }
}

/// `build(v)`, but a one-element `v` yields its element rather than a singleton
/// wrapper the solver would have to strip again.
fn unwrap_one(mut v: Vec<Expr>, build: fn(Vec<Expr>) -> Expr) -> Expr {
    match v.len() {
        1 => v.remove(0),
        _ => build(v),
    }
}

/// Replaces every bitwise-context constant with its formula over the base
/// variables. Mirrors the traversal of [`collect_bitwise_consts`].
fn rewrite(e: Expr, parent_bitwise: bool, mask: u64, formulas: &HashMap<u64, Expr>) -> Expr {
    let bit = |x: Expr| rewrite(x, true, mask, formulas);
    let arith = |x: Expr| rewrite(x, false, mask, formulas);
    match e {
        Expr::Const(c) => match formulas.get(&(c & mask)) {
            Some(f) if parent_bitwise => f.clone(),
            _ => Expr::Const(c),
        },
        Expr::Var(_) => e,
        Expr::Not(x) => !bit(*x),
        Expr::Scale(v, x) => v * arith(*x),
        Expr::And(xs) => Expr::And(xs.into_iter().map(bit).collect()),
        Expr::Or(xs) => Expr::Or(xs.into_iter().map(bit).collect()),
        Expr::Xor(xs) => Expr::Xor(xs.into_iter().map(bit).collect()),
        Expr::Add(xs) => Expr::Add(xs.into_iter().map(arith).collect()),
        Expr::Mul(xs) => Expr::Mul(xs.into_iter().map(arith).collect()),
    }
}

/// Substitutes the base variables by their concrete mask values.
pub(super) fn substitute(e: Expr, subs: &HashMap<VarId, u64>) -> Expr {
    let go = |x: Expr| substitute(x, subs);
    match e {
        Expr::Var(v) => match subs.get(&v) {
            Some(&val) => Expr::make_const(val),
            None => Expr::Var(v),
        },
        Expr::Const(_) => e,
        Expr::Not(x) => !go(*x),
        Expr::Scale(v, x) => v * go(*x),
        Expr::And(xs) => Expr::And(xs.into_iter().map(go).collect()),
        Expr::Or(xs) => Expr::Or(xs.into_iter().map(go).collect()),
        Expr::Xor(xs) => Expr::Xor(xs.into_iter().map(go).collect()),
        Expr::Add(xs) => Expr::Add(xs.into_iter().map(go).collect()),
        Expr::Mul(xs) => Expr::Mul(xs.into_iter().map(go).collect()),
    }
}

/// Folds constant-only subtrees, bottom-up, at width `mask`.
///
/// Needed because [`substitute`] turns base variables back into constants, which
/// leaves nodes like `And([Const, Not(Const)])` that the solver never saw and
/// that rumba's own `reduce` (private since 1.0.0) would have collapsed. Without
/// this, `cost` over-counts the linked result and [`emit`](super::emit) would
/// materialize constant arithmetic as real instructions.
///
/// Deliberately narrow: constant operands within one n-ary node are combined,
/// identities dropped and annihilators propagated. No algebraic reasoning over
/// non-constant operands — that is the solver's job, and it has already run.
pub(super) fn fold_consts(e: Expr, mask: u64) -> Expr {
    let go = |x: Expr| fold_consts(x, mask);
    match e {
        Expr::Var(_) | Expr::Const(_) => e,
        Expr::Not(x) => match go(*x) {
            Expr::Const(c) => Expr::Const(!c & mask),
            other => !other,
        },
        Expr::Scale(k, x) => match go(*x) {
            Expr::Const(c) => Expr::Const(k.wrapping_mul(c) & mask),
            other => k * other,
        },
        Expr::And(xs) => fold_nary(xs, mask, mask, Some(0), |a, b| a & b, Expr::And, go),
        Expr::Or(xs) => fold_nary(xs, mask, 0, Some(mask), |a, b| a | b, Expr::Or, go),
        Expr::Xor(xs) => fold_nary(xs, mask, 0, None, |a, b| a ^ b, Expr::Xor, go),
        Expr::Add(xs) => fold_nary(xs, mask, 0, None, |a, b| a.wrapping_add(b), Expr::Add, go),
        Expr::Mul(xs) => fold_nary(
            xs,
            mask,
            1,
            Some(0),
            |a, b| a.wrapping_mul(b),
            Expr::Mul,
            go,
        ),
    }
}

/// Folds one n-ary node: recurse into operands, combine the constant ones with
/// `op` starting from `identity`, and short-circuit on `annihilator`. The
/// accumulated constant is dropped when it is the identity, and the node
/// unwrapped when a single operand survives.
fn fold_nary(
    xs: Vec<Expr>,
    mask: u64,
    identity: u64,
    annihilator: Option<u64>,
    op: fn(u64, u64) -> u64,
    build: fn(Vec<Expr>) -> Expr,
    go: impl Fn(Expr) -> Expr,
) -> Expr {
    let mut acc = identity;
    let mut rest = Vec::with_capacity(xs.len());
    for x in xs {
        match go(x) {
            Expr::Const(c) => acc = op(acc, c & mask) & mask,
            other => rest.push(other),
        }
    }
    if Some(acc) == annihilator {
        return Expr::Const(acc);
    }
    if acc != identity || rest.is_empty() {
        rest.push(Expr::Const(acc));
    }
    unwrap_one(rest, build)
}

/// Attempts to link the constants of `e`. Returns `None` (cheaply) when there is
/// nothing to gain: fewer than two distinct constants, or a base that is no
/// smaller than the number of constants.
pub(super) fn link_constants(e: &Expr, n: u8) -> Option<Linked> {
    let mask = make_mask(n);

    let mut consts = vec![];
    collect_bitwise_consts(e, false, mask, &mut consts);
    consts.sort_unstable();
    consts.dedup();

    let m = consts.len();
    if !(2..=64).contains(&m) {
        return None;
    }

    // Only fire on a genuine tie. If the constants are pairwise disjoint *and* do
    // not cover the word (a background atom exists), they are independent:
    // packing them into a smaller base is mere cardinality compression with no
    // cancellation to gain, and it entangles otherwise-independent bits.
    let union = consts.iter().fold(0u64, |a, &c| a | c);
    let popcount_sum: u32 = consts.iter().map(|&c| c.count_ones()).sum();
    let pairwise_disjoint = popcount_sum == union.count_ones();
    if pairwise_disjoint && union != mask {
        return None;
    }

    let atoms = compute_atoms(&consts, n);
    let r = base_size(atoms.len());

    // No reduction in the number of opaque constants is possible.
    if m <= r {
        return None;
    }

    qcode::pass_log!(
        debug,
        "linking {m} constants: {} atoms, base of {r} masks",
        atoms.len()
    );

    // Deterministic atom order so codes (and hence the base) are stable.
    let mut items: Vec<(u64, u64)> = atoms.into_iter().collect();
    items.sort_unstable();

    // Base mask j = OR of atoms whose code has bit j set.
    let mut base = vec![0u64; r];
    for (idx, (_sig, atom)) in items.iter().enumerate() {
        for (j, b) in base.iter_mut().enumerate() {
            if (idx >> j) & 1 == 1 {
                *b |= atom;
            }
        }
    }

    // Fresh variable ids, above every existing variable.
    let start = e
        .get_vars()
        .iter()
        .map(|v| v.0)
        .max()
        .map(|x| x + 1)
        .unwrap_or(0);
    let base_vars: Vec<VarId> = (0..r).map(|j| VarId(start + j)).collect();

    let mut formulas: HashMap<u64, Expr> = HashMap::default();
    for (i, &c) in consts.iter().enumerate() {
        formulas.insert(c, build_formula(&items, i, &base_vars));
    }

    let subs: HashMap<VarId, u64> = base_vars
        .iter()
        .enumerate()
        .map(|(j, &v)| (v, base[j]))
        .collect();

    let expr = rewrite(e.clone(), false, mask, &formulas);

    Some(Linked { expr, subs })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rumba_core::simplify::{SimplifyCache, simplify_mba_cached};

    const N: u8 = 32;

    /// `0x2d6a963a` and its 32-bit complement `0xd29569c5`.
    const C: u64 = 0x2d6a963a;
    const NOT_C: u64 = 0xd29569c5;

    fn var(i: usize) -> Expr {
        Expr::Var(VarId(i))
    }

    fn and(a: Expr, b: Expr) -> Expr {
        Expr::And(vec![a, b])
    }

    /// A spread of assignments for `v0..v3`, evaluated at `N` bits.
    fn samples(e: &Expr) -> Vec<u64> {
        [
            [0, 0, 0, 0],
            [1, 0, 1, 0],
            [0xffff_ffff, 0x1234_5678, 7, 0],
            [0x8000_0000, 0xdead_beef, 0x5555_5555, 0xaaaa_aaaa],
            [0x2d6a_963a, 0xd295_69c5, 1, 0xffff_ffff],
        ]
        .iter()
        .map(|vars| e.eval(vars, N))
        .collect()
    }

    fn solve(e: Expr, cache: &SimplifyCache) -> Expr {
        simplify_mba_cached(e, N, cache).expect("solvable")
    }

    /// The rewrite is meaning-preserving on its own: linking then immediately
    /// substituting back and folding must reproduce the original expression's
    /// behaviour, with no solver involved. This is the soundness property the
    /// whole step rests on.
    #[test]
    fn link_then_substitute_round_trips() {
        let e = Expr::Add(vec![
            and(var(0), Expr::Const(C)),
            and(var(1), Expr::Const(NOT_C)),
            and(var(2), Expr::Const(C | 1)),
        ]);
        let linked = link_constants(&e, N).expect("tied constants must link");

        let back = fold_consts(substitute(linked.expr, &linked.subs), make_mask(N));
        assert_eq!(
            samples(&back),
            samples(&e),
            "substituting the base masks back must restore the original meaning"
        );
    }

    /// A complement pair is the `r == 1` case: `(v0 & c) + (v0 & ~c)` is `v0`.
    #[test]
    fn complement_pair_collapses_to_the_variable() {
        let e = Expr::Add(vec![
            and(var(0), Expr::Const(C)),
            and(var(0), Expr::Const(NOT_C)),
        ]);
        assert!(
            link_constants(&e, N).is_some(),
            "a complement pair is a tie"
        );

        let linked = link_constants(&e, N).expect("links");
        let cache = SimplifyCache::new();
        let solved = fold_consts(
            substitute(solve(linked.expr, &cache), &linked.subs),
            make_mask(N),
        );

        assert_eq!(samples(&solved), samples(&e), "must stay equivalent");
        assert_eq!(solved, var(0), "(v0 & c) + (v0 & ~c) collapses to v0");
    }

    /// Constants tied by a single bit — `c` and `c | 1` — are the case rumba's
    /// built-in complement check cannot see, and the reason this step exists.
    ///
    /// Note it takes a *third* constant to be worth doing: `{c, c|1}` alone
    /// partitions into three atoms (bits of `c`, bit 0, the rest), so the base is
    /// `r = 2` masks for `m = 2` constants and [`link_constants`] declines. Adding
    /// `~c` covers the word, leaving three atoms for three constants.
    #[test]
    fn constants_tied_by_one_bit_link_and_stay_equivalent() {
        let e = Expr::Or(vec![
            and(var(0), Expr::Const(C)),
            and(var(1), Expr::Const(C | 1)),
            and(var(2), Expr::Const(NOT_C)),
        ]);
        let linked = link_constants(&e, N).expect("c, c|1 and ~c are tied");

        let cache = SimplifyCache::new();
        let solved = fold_consts(
            substitute(solve(linked.expr, &cache), &linked.subs),
            make_mask(N),
        );
        assert_eq!(samples(&solved), samples(&e), "must stay equivalent");
    }

    /// Two constants tied by one bit do not pay for themselves — the base is no
    /// smaller than the set it replaces, so linking declines.
    #[test]
    fn a_tie_that_does_not_shrink_the_base_is_declined() {
        let e = Expr::Or(vec![
            and(var(0), Expr::Const(C)),
            and(var(1), Expr::Const(C | 1)),
        ]);
        assert!(link_constants(&e, N).is_none(), "m = 2 but r = 2: no gain");
    }

    /// Pairwise-disjoint constants that leave bits uncovered are independent:
    /// there is no tie to cancel, so linking must bail rather than entangle them.
    #[test]
    fn independent_constants_are_left_alone() {
        let e = Expr::Add(vec![
            and(var(0), Expr::Const(0xff)),
            and(var(1), Expr::Const(0xff00)),
        ]);
        assert!(
            link_constants(&e, N).is_none(),
            "disjoint masks not covering the word carry no tie"
        );
    }

    /// A single constant has nothing to be tied to.
    #[test]
    fn a_lone_constant_does_not_link() {
        let e = and(var(0), Expr::Const(C));
        assert!(link_constants(&e, N).is_none());
    }

    /// Constants in arithmetic position are not masks and must not be linked —
    /// only operands of `& | ^ ~` are collected.
    #[test]
    fn arithmetic_constants_are_not_collected() {
        let e = Expr::Add(vec![var(0), Expr::Const(C), Expr::Const(NOT_C)]);
        assert!(
            link_constants(&e, N).is_none(),
            "addends are values, not bit masks"
        );
    }

    #[test]
    fn fold_consts_combines_and_unwraps() {
        let mask = make_mask(N);

        // Constant operands combine; the node unwraps when one operand is left.
        assert_eq!(
            fold_consts(Expr::And(vec![Expr::Const(0xf0), Expr::Const(0x3c)]), mask),
            Expr::Const(0x30)
        );
        // Identity operands drop out.
        assert_eq!(
            fold_consts(Expr::And(vec![var(0), Expr::Const(mask)]), mask),
            var(0)
        );
        assert_eq!(
            fold_consts(Expr::Add(vec![var(0), Expr::Const(0)]), mask),
            var(0)
        );
        // Annihilators short-circuit.
        assert_eq!(
            fold_consts(Expr::And(vec![var(0), Expr::Const(0)]), mask),
            Expr::Const(0)
        );
        assert_eq!(
            fold_consts(Expr::Mul(vec![var(0), Expr::Const(0)]), mask),
            Expr::Const(0)
        );
        // Results are masked to the region width.
        assert_eq!(
            fold_consts(Expr::Not(Box::new(Expr::Const(0))), mask),
            Expr::Const(mask)
        );
        // Non-constant operands survive alongside the folded constant.
        assert_eq!(
            fold_consts(
                Expr::Add(vec![var(0), Expr::Const(3), Expr::Const(4)]),
                mask
            ),
            Expr::Add(vec![var(0), Expr::Const(7)])
        );
    }
}
