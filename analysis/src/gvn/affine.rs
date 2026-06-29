//! Normal-form value numbering for arithmetic and bitwise-mask expressions.
//!
//! Syntactic CSE ([`super::cse`]) numbers an instruction by its mnemonic, so two
//! expressions that compute the same value via different shapes never unify. This
//! module canonicalizes the *covered* operations into a width-tagged normal form
//! so that, e.g., `(@ESP - 0xc) + 4`, `@ESP - 8` and `@ESP + 0xfffffff8` all hash
//! to the same [`NormalForm`] and forward to whichever dominating value already
//! computes it.
//!
//! Two normal-form kinds (arithmetic and bitwise never mix):
//! * **Affine**: `c + Σ kᵢ·termᵢ`, all arithmetic wrapping mod `2^(width*8)`.
//!   `sub` contributes a `-1` coefficient, `mul`/`shl` by a constant a scale,
//!   `neg` a `-1` scale.
//! * **Mask**: `term OP const` for `OP ∈ {and, or, xor}`, coalescing same-op
//!   constant-mask chains over a single term.
//!
//! Anything else — loads, calls, casts (`zext`/`sext`), block params, non-linear
//! products, symbolic literals — is an opaque leaf term. The *key* of such an
//! instruction falls back to the normalized mnemonic ([`NormalForm::Opaque`]).
//!
//! ## Key vs. emitted IR
//! The key uses wrapping `u64` with everything as addition, maximizing forwarding
//! collisions. Materialization (when no dominating equal exists) interprets each
//! constant/coefficient as *signed at the operation width* and renders negatives
//! with `sub`, preserving stack-frame offsets without special-casing pointer
//! types. Reconstruction is deterministic (terms ordered by `value_id_key`) so it
//! is idempotent: a value already in canonical form is detected and left in place.

use rustc_hash::FxHashMap as HashMap;

use qcode::{
    context::Context,
    value::{
        BasicBlock, Function, ValueId,
        block::BlockId,
        function::FunctionId,
        insn::{Binary, Binop, InstructionId, InstructionRef, IntBinop, Mnemonic, Unary, Unop},
    },
};

use super::cse::{normalize, value_id_key};
use super::fold::const_value;

/// The value-numbering key. `Affine`/`Mask` are also stored per value as the
/// "arithmetic view" used to compose parents; `Opaque` is key-only.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum NormalForm {
    /// `constant + Σ coeff·term`, wrapping mod `2^(width*8)`. `terms` is sorted by
    /// [`value_id_key`] with zero coefficients dropped; `width` is in bytes.
    Affine {
        width: usize,
        constant: u64,
        terms: Vec<(ValueId, u64)>,
    },
    /// `term op mask` for `op ∈ {And, Or, Xor}`, `mask` reduced mod width.
    Mask {
        width: usize,
        term: ValueId,
        op: IntBinop,
        mask: u64,
    },
    /// Fallback: the commutativity-normalized mnemonic (loads, calls, casts, …).
    Opaque(Mnemonic),
}

/// Per-walk value-numbering state, cloned down the dominator tree.
#[derive(Clone, Default)]
pub(crate) struct Numbering {
    /// Canonical key → the dominating value that computes it (the leader).
    leaders: HashMap<NormalForm, ValueId>,
    /// Value → its arithmetic view (always `Affine` or `Mask`), used to compose
    /// the forms of instructions that consume it.
    forms: HashMap<ValueId, NormalForm>,
}

// ---------------------------------------------------------------------------
// Width helpers
// ---------------------------------------------------------------------------

/// Affine reasoning is limited to widths the wrapping `u64` arithmetic models
/// exactly. Wider values (e.g. packed 96-bit aggregates) stay opaque.
const MAX_AFFINE_WIDTH: usize = 8;

fn mask_for(width: usize) -> u64 {
    if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (width * 8)) - 1
    }
}

/// Whether `mask` is a round-down alignment mask at `width` bytes: `-2^k`
/// truncated to the width, i.e. the cleared low bits form a contiguous run
/// (`2^k - 1`) and every higher bit is set. `x & mask` then satisfies
/// `x & mask ≤ x`, the monotonicity the frame classifier relies on.
fn is_round_down_mask(mask: u64, width: usize) -> bool {
    let low = !mask & mask_for(width); // the bits this mask clears
    low.wrapping_add(1) & low == 0 // low == 2^k - 1  ⟹  a contiguous low run
}

/// Interpret `value` as signed at `width` bytes. A zero-width value (a
/// result-less instruction such as a store) has no meaningful magnitude — its
/// sign is zero — which keeps the shift below well-defined.
fn signed(value: u64, width: usize) -> i64 {
    let bits = width * 8;
    if bits == 0 {
        0
    } else if bits >= 64 {
        value as i64
    } else {
        ((value << (64 - bits)) as i64) >> (64 - bits)
    }
}

// ---------------------------------------------------------------------------
// Affine term arithmetic
// ---------------------------------------------------------------------------

/// Merge `b` into `a`, summing coefficients of shared terms, dropping zeros, and
/// re-sorting by `value_id_key` for a canonical order.
fn merge_terms(mut a: Vec<(ValueId, u64)>, b: Vec<(ValueId, u64)>, m: u64) -> Vec<(ValueId, u64)> {
    for (v, k) in b {
        match a.iter_mut().find(|(vv, _)| *vv == v) {
            Some(e) => e.1 = e.1.wrapping_add(k) & m,
            None => a.push((v, k & m)),
        }
    }
    a.retain(|(_, k)| *k & m != 0);
    a.sort_by_key(|(v, _)| value_id_key(*v));
    a
}

fn scale_terms(terms: &[(ValueId, u64)], k: u64, m: u64) -> Vec<(ValueId, u64)> {
    let mut out: Vec<(ValueId, u64)> = terms
        .iter()
        .map(|(v, c)| (*v, c.wrapping_mul(k) & m))
        .filter(|(_, c)| *c != 0)
        .collect();
    out.sort_by_key(|(v, _)| value_id_key(*v));
    out
}

/// The arithmetic view of an operand: its stored affine form, a constant, or an
/// opaque leaf `1·v`. Mask/opaque values are treated as opaque leaves.
fn affine_view(
    ctx: &Context,
    v: ValueId,
    width: usize,
    state: &Numbering,
) -> (u64, Vec<(ValueId, u64)>) {
    if let Some(NormalForm::Affine {
        width: w,
        constant,
        terms,
    }) = state.forms.get(&v)
        && *w == width
    {
        return (*constant, terms.clone());
    }
    if let Some(c) = const_value(ctx, v) {
        return (c & mask_for(width), vec![]);
    }
    (0, vec![(v, 1)])
}

fn leaf(id: ValueId, width: usize) -> NormalForm {
    NormalForm::Affine {
        width,
        constant: 0,
        terms: vec![(id, 1)],
    }
}

/// Whether `form` is the trivial leaf of `id` itself (`1·id + 0`), i.e. the op
/// did not decompose into anything but itself.
fn is_self_leaf(form: &NormalForm, id: ValueId) -> bool {
    matches!(
        form,
        NormalForm::Affine { constant: 0, terms, .. }
            if terms.len() == 1 && terms[0] == (id, 1)
    )
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// Compute the arithmetic view of the value `id` produced by `mnemonic`. Returns
/// an `Affine`/`Mask` form for covered ops, else the opaque leaf `1·id`.
pub(super) fn arith_form(
    ctx: &Context,
    id: ValueId,
    mnemonic: &Mnemonic,
    width: usize,
    state: &Numbering,
) -> NormalForm {
    if width == 0 || width > MAX_AFFINE_WIDTH {
        return leaf(id, width);
    }
    let m = mask_for(width);

    match mnemonic {
        Mnemonic::Binop(Binary {
            op: Binop::Int(op),
            lhs,
            rhs,
        }) => match op {
            IntBinop::Add => {
                let (cl, tl) = affine_view(ctx, *lhs, width, state);
                let (cr, tr) = affine_view(ctx, *rhs, width, state);
                NormalForm::Affine {
                    width,
                    constant: cl.wrapping_add(cr) & m,
                    terms: merge_terms(tl, tr, m),
                }
            }
            IntBinop::Sub => {
                let (cl, tl) = affine_view(ctx, *lhs, width, state);
                let (cr, tr) = affine_view(ctx, *rhs, width, state);
                NormalForm::Affine {
                    width,
                    constant: cl.wrapping_sub(cr) & m,
                    terms: merge_terms(tl, scale_terms(&tr, m /* -1 */, m), m),
                }
            }
            IntBinop::Mul => {
                // Affine only when exactly one side is a constant scale.
                if let Some(k) = const_value(ctx, *rhs) {
                    scale_affine(ctx, *lhs, k & m, width, state)
                } else if let Some(k) = const_value(ctx, *lhs) {
                    scale_affine(ctx, *rhs, k & m, width, state)
                } else {
                    leaf(id, width)
                }
            }
            IntBinop::ShiftLeft => {
                // x << s  ==  x * 2^s  (constant amount, in range).
                match const_value(ctx, *rhs) {
                    Some(s) if s < (width as u64 * 8) && s < 64 => {
                        scale_affine(ctx, *lhs, (1u64 << s) & m, width, state)
                    }
                    _ => leaf(id, width),
                }
            }
            IntBinop::And | IntBinop::Or | IntBinop::Xor => {
                if let Some(k) = const_value(ctx, *rhs) {
                    mask_form(*lhs, *op, k & m, width, state)
                } else if let Some(k) = const_value(ctx, *lhs) {
                    mask_form(*rhs, *op, k & m, width, state)
                } else {
                    leaf(id, width)
                }
            }
            _ => leaf(id, width),
        },
        Mnemonic::Unop(Unary {
            op: Unop::IntNegate,
            src,
        }) => {
            scale_affine(ctx, *src, m /* -1 */, width, state)
        }
        // `gep(base, off)` ≡ `base + off` (a constant byte offset). Decompose it
        // like an `Add` so a field address numbers the same as the equivalent
        // pointer arithmetic: this lets memory forwarding unify a `gep(p.field)`
        // load with a seed/store written to `p + off`. The gep itself is kept
        // syntactic (never rewritten into an add) by [`key_for`], which forces a
        // `Gep` mnemonic to an opaque key.
        Mnemonic::Gep(g) => {
            let (c, t) = affine_view(ctx, g.base, width, state);
            NormalForm::Affine {
                width,
                constant: c.wrapping_add(g.offset as u64) & m,
                terms: t,
            }
        }
        _ => leaf(id, width),
    }
}

fn scale_affine(ctx: &Context, v: ValueId, k: u64, width: usize, state: &Numbering) -> NormalForm {
    let m = mask_for(width);
    let (c, t) = affine_view(ctx, v, width, state);
    NormalForm::Affine {
        width,
        constant: c.wrapping_mul(k) & m,
        terms: scale_terms(&t, k, m),
    }
}

/// Build a `Mask` form for `operand op const`, coalescing with a same-op mask
/// chain on `operand` and collapsing identities (`& all-ones`, `| 0`, `^ 0` → the
/// term; `& 0` → `0`; `| all-ones` → all-ones).
fn mask_form(
    operand: ValueId,
    op: IntBinop,
    k: u64,
    width: usize,
    state: &Numbering,
) -> NormalForm {
    let all = mask_for(width);
    // Coalesce with an inner same-op mask if present.
    let (term, mask) = match state.forms.get(&operand) {
        Some(NormalForm::Mask {
            width: w,
            term,
            op: iop,
            mask,
        }) if *w == width && *iop == op => (
            *term,
            match op {
                IntBinop::And => mask & k,
                IntBinop::Or => mask | k,
                IntBinop::Xor => mask ^ k,
                _ => unreachable!(),
            },
        ),
        _ => (operand, k),
    };

    // Identity collapses.
    match op {
        IntBinop::And if mask == all => leaf(term, width),
        IntBinop::And if mask == 0 => NormalForm::Affine {
            width,
            constant: 0,
            terms: vec![],
        },
        IntBinop::Or if mask == 0 => leaf(term, width),
        IntBinop::Or if mask == all => NormalForm::Affine {
            width,
            constant: all,
            terms: vec![],
        },
        IntBinop::Xor if mask == 0 => leaf(term, width),
        _ => NormalForm::Mask {
            width,
            term,
            op,
            mask,
        },
    }
}

/// The value-numbering key for `id`: the arithmetic form for covered ops that
/// genuinely decomposed, else the normalized mnemonic.
pub(super) fn key_for(form: &NormalForm, id: ValueId, mnemonic: &Mnemonic) -> NormalForm {
    // A `Gep` decomposes affinely (see [`arith_form`]) so its *consumers* unify
    // through it, but the gep instruction itself must stay syntactic: it carries
    // struct field typing the canonicalizer would discard by rebuilding it as a
    // bare `add`. Key it opaquely so CSE claims it as-is rather than materializing
    // an affine replacement.
    if matches!(mnemonic, Mnemonic::Gep(_)) {
        let mut m = mnemonic.clone();
        normalize(&mut m);
        return NormalForm::Opaque(m);
    }
    if is_self_leaf(form, id) {
        let mut m = mnemonic.clone();
        normalize(&mut m);
        NormalForm::Opaque(m)
    } else {
        form.clone()
    }
}

// ---------------------------------------------------------------------------
// Materialization
// ---------------------------------------------------------------------------

/// Insert a fresh instruction `(mnemonic, type)` before `at` in `block`.
fn emit(
    ctx: &mut Context,
    block: BlockId,
    at: InstructionId,
    mnemonic: Mnemonic,
    ty: qcode::types::TypeId,
) -> ValueId {
    let new = InstructionRef::from_mnemonic_with_type(ctx, mnemonic, ty).id;
    BasicBlock::from_id_mut(ctx, block).insert_insn_before(at, new);
    ValueId::Instruction(new)
}

/// Reuse-or-create the value computing `form` (a sub-expression). Trivial forms
/// resolve to a literal or the bare term; otherwise a dominating leader is reused
/// if present, else the canonical instruction is emitted and registered.
fn build_value(
    ctx: &mut Context,
    block: BlockId,
    at: InstructionId,
    form: &NormalForm,
    state: &mut Numbering,
) -> ValueId {
    if let NormalForm::Affine {
        width,
        constant,
        terms,
    } = form
    {
        if terms.is_empty() {
            return ctx.get_const(*constant, *width).id();
        }
        if terms.len() == 1 && terms[0].1 == 1 && *constant == 0 {
            return terms[0].0;
        }
    }
    if let Some(&leader) = state.leaders.get(form) {
        return leader;
    }
    let int_ty = match form {
        NormalForm::Affine { width, .. } | NormalForm::Mask { width, .. } => {
            ctx.types.get_or_make_int(*width)
        }
        NormalForm::Opaque(_) => unreachable!("opaque forms are never materialized"),
    };
    let m = canonical_mnemonic(ctx, block, at, form, state);
    let vid = emit(ctx, block, at, m, int_ty);
    state.leaders.insert(form.clone(), vid);
    state.forms.insert(vid, form.clone());
    vid
}

/// The single-level canonical mnemonic for `form`, building its operands via
/// [`build_value`] (which reuses dominating sub-results). Deterministic: terms in
/// `value_id_key` order, negatives rendered as `sub`, the constant applied last.
fn canonical_mnemonic(
    ctx: &mut Context,
    block: BlockId,
    at: InstructionId,
    form: &NormalForm,
    state: &mut Numbering,
) -> Mnemonic {
    match form {
        NormalForm::Mask {
            width,
            term,
            op,
            mask,
        } => Mnemonic::Binop(Binary {
            op: Binop::Int(*op),
            lhs: *term,
            rhs: ctx.get_const(*mask, *width).id(),
        }),
        NormalForm::Affine {
            width,
            constant,
            terms,
        } => {
            let width = *width;
            let constant = *constant;

            // The constant, when present, is the trailing element (added last).
            // It is the last element only when there is at least one term to
            // anchor it; a pure constant is handled as a trivial form earlier.
            if constant != 0 {
                let prefix = NormalForm::Affine {
                    width,
                    constant: 0,
                    terms: terms.clone(),
                };
                let pv = build_value(ctx, block, at, &prefix, state);
                let (op, lit) = signed_lit(ctx, signed(constant, width), width);
                return Mnemonic::Binop(Binary {
                    op: Binop::Int(op),
                    lhs: pv,
                    rhs: lit,
                });
            }

            // constant == 0. Order operations as: positive terms added (in key
            // order), then negative terms subtracted (in key order). This keeps
            // `b - a` as `sub(b, a)` rather than introducing a negate.
            let last_neg = terms
                .iter()
                .rev()
                .find(|(_, k)| signed(*k, width) < 0)
                .copied();
            let (last_v, last_k) = match last_neg {
                Some(t) => t,
                // No negative terms: the trailing element is the highest-key
                // positive term.
                None => *terms.last().unwrap(),
            };

            let prefix_terms: Vec<(ValueId, u64)> = terms
                .iter()
                .copied()
                .filter(|(v, _)| *v != last_v)
                .collect();

            // Lone term: `-x` → negate, `k·x` → mul by the raw (wrapping) coeff.
            if prefix_terms.is_empty() {
                if signed(last_k, width) == -1 {
                    return Mnemonic::Unop(Unary {
                        op: Unop::IntNegate,
                        src: last_v,
                    });
                }
                return Mnemonic::Binop(Binary {
                    op: Binop::Int(IntBinop::Mul),
                    lhs: last_v,
                    rhs: ctx.get_const(last_k, width).id(),
                });
            }

            let prefix = NormalForm::Affine {
                width,
                constant: 0,
                terms: prefix_terms,
            };
            let pv = build_value(ctx, block, at, &prefix, state);
            let s = signed(last_k, width);
            let (op, mag) = if s < 0 {
                (IntBinop::Sub, s.unsigned_abs() & mask_for(width))
            } else {
                (IntBinop::Add, last_k)
            };
            let tv = scaled_value(ctx, block, at, last_v, mag, width, state);
            Mnemonic::Binop(Binary {
                op: Binop::Int(op),
                lhs: pv,
                rhs: tv,
            })
        }
        NormalForm::Opaque(_) => unreachable!("opaque forms are never materialized"),
    }
}

/// The value of `mag·term` (a positive magnitude): the bare term when `mag == 1`,
/// else a reused-or-created `mul`.
fn scaled_value(
    ctx: &mut Context,
    block: BlockId,
    at: InstructionId,
    term: ValueId,
    mag: u64,
    width: usize,
    state: &mut Numbering,
) -> ValueId {
    if mag == 1 {
        return term;
    }
    let form = NormalForm::Affine {
        width,
        constant: 0,
        terms: vec![(term, mag)],
    };
    build_value(ctx, block, at, &form, state)
}

/// `(op, literal)` for adding a signed constant: `sub |s|` when negative.
fn signed_lit(ctx: &mut Context, s: i64, width: usize) -> (IntBinop, ValueId) {
    if s < 0 {
        (
            IntBinop::Sub,
            ctx.get_const(s.unsigned_abs() & mask_for(width), width)
                .id(),
        )
    } else {
        (
            IntBinop::Add,
            ctx.get_const(s as u64 & mask_for(width), width).id(),
        )
    }
}

/// Materialize the canonical form for `key` at `at`. Returns the original
/// `at` value if it is already canonical (no rewrite); otherwise the value of a
/// reused dominating leader or a freshly emitted canonical expression.
///
/// `root_ty` is the result type to give a newly created root instruction so that
/// StackAddress/symbolic typing is preserved.
pub(super) fn materialize(
    ctx: &mut Context,
    block: BlockId,
    at: InstructionId,
    at_mnemonic: &Mnemonic,
    key: &NormalForm,
    root_ty: qcode::types::TypeId,
    state: &mut Numbering,
) -> ValueId {
    // Trivial roots: forward straight to the literal / bare term.
    if let NormalForm::Affine {
        width,
        constant,
        terms,
    } = key
    {
        if terms.is_empty() {
            return ctx.get_const(*constant, *width).id();
        }
        if terms.len() == 1 && terms[0].1 == 1 && *constant == 0 {
            return terms[0].0;
        }
    }
    // A dominating value already computes this form: reuse it.
    if let Some(&leader) = state.leaders.get(key) {
        return leader;
    }
    // Build the canonical root; if it matches the current instruction, the value
    // is already canonical — leave it in place.
    let m = canonical_mnemonic(ctx, block, at, key, state);
    if &m == at_mnemonic {
        let vid = ValueId::Instruction(at);
        state.leaders.insert(key.clone(), vid);
        return vid;
    }
    let vid = emit(ctx, block, at, m, root_ty);
    state.leaders.insert(key.clone(), vid);
    state.forms.insert(vid, key.clone());
    vid
}

impl Numbering {
    /// Record the arithmetic view of `id` so consumers can compose it.
    pub(super) fn record_form(&mut self, id: ValueId, form: NormalForm) {
        self.forms.insert(id, form);
    }

    /// The stored arithmetic view of `v` (`Affine`/`Mask`), if any. Used by the
    /// structural congruence engine to flatten affine subtrees.
    pub(super) fn lookup_form(&self, v: ValueId) -> Option<&NormalForm> {
        self.forms.get(&v)
    }

    /// Claim `id` as the leader for `key` if no dominating leader exists yet,
    /// returning the existing dominating leader otherwise.
    pub(super) fn lookup(&self, key: &NormalForm) -> Option<ValueId> {
        self.leaders.get(key).copied()
    }

    pub(super) fn claim(&mut self, key: NormalForm, id: ValueId) {
        self.leaders.entry(key).or_insert(id);
    }

    /// Drop all inherited leaders. Used on entry to a block reachable from more
    /// than one walk root, whose inherited dominance claims are invalid — a
    /// leader from a per-entry dominator-tree ancestor need not actually
    /// dominate it. (The per-value `forms` are SSA decompositions and only ever
    /// consulted for genuine operands, so they stay valid.)
    pub(super) fn clear_leaders(&mut self) {
        self.leaders.clear();
    }

    /// The affine decomposition of `ptr` as a single base term plus a signed
    /// byte offset, i.e. `ptr == base + constant` with a unit coefficient.
    /// Returns `None` for multi-term, scaled, masked, or non-affine pointers —
    /// the caller then treats the whole pointer value as its own base.
    ///
    /// This is the affine base-identity used by memory forwarding: `(p + 4) - 4`
    /// and `p` decompose to the same `base = p`, so they unify for free.
    pub(crate) fn base_offset(&self, ptr: ValueId) -> Option<(ValueId, i64)> {
        match self.forms.get(&ptr)? {
            NormalForm::Affine {
                width,
                constant,
                terms,
            } if terms.len() == 1 && terms[0].1 == 1 => {
                Some((terms[0].0, signed(*constant, *width)))
            }
            _ => None,
        }
    }

    /// The full affine decomposition of `v` as `(width, constant, terms)` where
    /// `v == constant + Σ coeff·term` (wrapping mod `2^(width*8)`), or `None` if
    /// `v` has no affine arithmetic view (a bare leaf, a mask, or an opaque op).
    ///
    /// Unlike [`base_offset`], which only accepts the single-term unit-coefficient
    /// `base + const` shape, this exposes the *whole* sum so a caller can pick out a
    /// base pointer term and treat the remaining (possibly scaled, possibly
    /// dynamic) terms as a strided index — e.g. `(base + idx*4) + 4` decomposes to
    /// `constant=4, terms=[(base,1),(idx,4)]`.
    pub(crate) fn affine_terms(&self, v: ValueId) -> Option<(usize, u64, Vec<(ValueId, u64)>)> {
        match self.forms.get(&v)? {
            NormalForm::Affine {
                width,
                constant,
                terms,
            } => Some((*width, *constant, terms.clone())),
            _ => None,
        }
    }

    /// If `v`'s arithmetic view is `term & mask` for a power-of-two **round-down**
    /// alignment mask (`-2^k`: a contiguous run of clear low bits with every higher
    /// bit set), return `term` — the value being aligned. Such a mask can only
    /// *lower* an address (`x & -2^k ≤ x`); that monotonicity is what lets the frame
    /// classifier keep a realigned own-frame base own-frame. `None` for any other
    /// form. (`& 0` and `& all-ones` never reach here — [`mask_form`] collapses both.)
    pub(crate) fn alignment_base(&self, v: ValueId) -> Option<ValueId> {
        match self.forms.get(&v)? {
            NormalForm::Mask {
                width,
                term,
                op: IntBinop::And,
                mask,
            } if is_round_down_mask(*mask, *width) => Some(*term),
            _ => None,
        }
    }

    /// Whether `v`'s normal form is built (transitively) on `target` as a term.
    ///
    /// Unlike [`base_offset`], which only recognises the clean `target + const`
    /// shape, this answers "does `v` *depend on* `target` arithmetically at all" —
    /// catching `target + reg` (a dynamically indexed stack access) and a realigned
    /// `(target & -mask) + k`, where the offset from `target` is not a fixed
    /// constant. A stack pointer that mentions `@SP` this way but is not a fixed
    /// slot may alias any slot, so it disables slot promotion.
    pub(crate) fn affine_mentions(&self, v: ValueId, target: ValueId) -> bool {
        self.affine_mentions_rec(v, target, &mut std::collections::HashSet::new())
    }

    fn affine_mentions_rec(
        &self,
        v: ValueId,
        target: ValueId,
        seen: &mut std::collections::HashSet<ValueId>,
    ) -> bool {
        if v == target {
            return true;
        }
        if !seen.insert(v) {
            return false;
        }
        match self.forms.get(&v) {
            Some(NormalForm::Affine { terms, .. }) => terms
                .iter()
                .any(|(t, _)| self.affine_mentions_rec(*t, target, seen)),
            Some(NormalForm::Mask { term, .. }) => self.affine_mentions_rec(*term, target, seen),
            _ => false,
        }
    }
}

/// Precompute the position-independent affine view (`forms`) of every
/// instruction value in `func_id`, memoized into a fresh [`Numbering`].
///
/// A value's arithmetic view depends only on the SSA graph (its operands'
/// views), not on dominance, so it can be computed once up front and shared
/// read-only with sub-passes that need a pointer's base identity before the
/// dominator walk reaches the pointer's definition (see memory forwarding). Only
/// `forms` is populated; `leaders` stay empty (they are dominance-sensitive).
pub(crate) fn precompute_forms(ctx: &Context, func_id: FunctionId) -> Numbering {
    let mut numbering = Numbering::default();
    let ids: Vec<ValueId> = Function::from_id(ctx, func_id)
        .iter()
        .flat_map(|block| {
            block
                .instruction_ids()
                .iter()
                .map(|&id| ValueId::Instruction(id))
                .collect::<Vec<_>>()
        })
        .collect();
    for id in ids {
        ensure_form(ctx, id, &mut numbering);
    }
    numbering
}

/// Like [`precompute_forms`] but seeded from an explicit block set rather than a
/// whole function. Operand recursion follows the SSA graph regardless of block,
/// so passing a function's full block list is equivalent to `precompute_forms`.
pub(crate) fn precompute_forms_for_blocks(ctx: &Context, blocks: &[BlockId]) -> Numbering {
    let mut numbering = Numbering::default();
    let ids: Vec<ValueId> = blocks
        .iter()
        .flat_map(|&b| {
            BasicBlock::from_id(ctx, b)
                .instruction_ids()
                .iter()
                .map(|&id| ValueId::Instruction(id))
                .collect::<Vec<_>>()
        })
        .collect();
    for id in ids {
        ensure_form(ctx, id, &mut numbering);
    }
    numbering
}

#[cfg(test)]
mod spike {
    //! Throwaway probe for the brighten/lower removal: does the affine numbering
    //! give `@SP - const` stable slot identity without the `@stack_base` literal?
    use super::*;
    use qcode::{
        builder::Builder,
        testing::TestContext,
        value::{BasicBlock, Function, Value},
    };

    #[test]
    fn sp_relative_identity_holds_but_alignment_reroots() {
        let mut tc = TestContext::new();
        let fun = Function::make(&mut tc.ctx, "f".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fun);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        // `@SP`: the incoming stack-pointer entry parameter.
        let sp = ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(8).id);

        let (s1, s2, threaded, aligned, al) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let c8 = b.context_mut().get_const(8, 8).id();
            let c20 = b.context_mut().get_const(0x20, 8).id();
            let neg16 = b.context_mut().get_const((-16i64) as u64, 8).id();
            // Two independent occurrences of `@SP - 8` (distinct ValueIds).
            let s1 = b.push_sub(sp, c8).id();
            let s2 = b.push_sub(sp, c8).id();
            // `[rsp+8]` after `sub rsp, 0x20`: (@SP - 0x20) + 8.
            let sub_rsp = b.push_sub(sp, c20).id();
            let threaded = b.push_add(sub_rsp, c8).id();
            // `[rsp+8]` after `and rsp, -16`: (@SP & -16) + 8.
            let aligned = b.push_bit_and(sp, neg16).id();
            let al = b.push_add(aligned, c8).id();
            unsafe { b.dont_finalize() };
            (s1, s2, threaded, aligned, al)
        };

        let nb = precompute_forms(&tc.ctx, fun);

        // VERDICT 1 — plain slots get stable `(@SP, offset)` identity, the same for
        // every independent occurrence, with no `@stack_base` literal and no
        // dominance information. This is the make-or-break for the migration.
        assert_eq!(nb.base_offset(s1), Some((sp, -8)));
        assert_eq!(nb.base_offset(s2), Some((sp, -8)));

        // VERDICT 2 — SP moves thread through affine composition for free: a slot
        // reached after `sub rsp, 0x20` still roots at `@SP`, at the summed offset.
        assert_eq!(nb.base_offset(threaded), Some((sp, -0x18)));

        // VERDICT 3 — the regression: `and rsp, -16` is a *mask*, not affine, so the
        // slot re-roots at the aligned base (`@SP & -16`) instead of `@SP`. Identity
        // among post-alignment slots survives; the link back to `@SP` is lost.
        assert_eq!(nb.base_offset(aligned), None);
        assert_eq!(nb.base_offset(al), Some((aligned, 8)));
    }

    /// `affine_mentions` recognises every pointer transitively built on `@SP`,
    /// including the dynamically indexed (`@SP + reg`) and realigned
    /// (`(@SP & -mask) + k`) shapes that `base_offset` rejects — these are exactly
    /// the stack pointers that must disable slot promotion.
    #[test]
    fn affine_mentions_catches_dynamic_and_aligned_sp() {
        let mut tc = TestContext::new();
        let ram = tc.ctx.default_space;
        let reg = tc.reg_space;
        let fun = Function::make(&mut tc.ctx, "f".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fun);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let sp = ValueId::BlockParam(BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(8).id);

        let (fixed, indexed, aligned_slot, unrelated) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let c8 = b.context_mut().get_const(8, 8).id();
            let neg16 = b.context_mut().get_const((-16i64) as u64, 8).id();
            // A non-constant index loaded from a register.
            let idx = b.push_load::<false>(ValueId::Varnode(tc.r1), 8, reg).id();
            let fixed = b.push_sub(sp, c8).id(); // @SP - 8   (fixed slot)
            let indexed = b.push_add(sp, idx).id(); // @SP + reg (dynamic)
            let aligned = b.push_bit_and(sp, neg16).id();
            let aligned_slot = b.push_add(aligned, c8).id(); // (@SP & -16) + 8
            let other = b.context_mut().get_const(0x4000, 8).id();
            let unrelated = b.push_add(other, idx).id(); // base + reg, no @SP
            let _ = b.push_load::<false>(indexed, 1, ram);
            unsafe { b.dont_finalize() };
            (fixed, indexed, aligned_slot, unrelated)
        };

        let nb = precompute_forms(&tc.ctx, fun);

        assert!(nb.affine_mentions(fixed, sp), "@SP - 8 is built on @SP");
        assert!(nb.affine_mentions(indexed, sp), "@SP + reg is built on @SP");
        assert!(
            nb.affine_mentions(aligned_slot, sp),
            "(@SP & -16) + 8 is built on @SP through the mask term"
        );
        assert!(
            !nb.affine_mentions(unrelated, sp),
            "a pointer with no @SP term is not mentioned"
        );
        // `@SP` itself trivially mentions `@SP`.
        assert!(nb.affine_mentions(sp, sp));
    }

    /// A `gep(base, off)` decomposes to the same affine base+offset as `base + off`,
    /// so a field address unifies with the equivalent pointer arithmetic for memory
    /// forwarding. This is the fix for argpromote reading an un-seeded shadow slot:
    /// the seed is stored at `p + off` (an add) while the body dereferences
    /// `gep(p.field)`; without gep decomposition the two never unify and the
    /// redirected load forwards from nothing.
    #[test]
    fn gep_decomposes_to_base_plus_offset_like_an_add() {
        use qcode::types::AggregateField;
        let mut tc = TestContext::new();
        let fun = Function::make(&mut tc.ctx, "f".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fun);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        // A struct with a field at byte 0x60, so `push_gep` accepts the base.
        let i64_ty = tc.ctx.types.get_or_make_int(8);
        let s_ty = tc.ctx.types.get_or_make_struct(
            "S",
            0x68,
            vec![AggregateField::new_at("f", i64_ty, 0x60)],
        );
        let ptr_ty = tc.ctx.types.get_or_make_struct_pointer(8, s_ty);
        let pid = BasicBlock::from_id_mut(&mut tc.ctx, entry).push_param(8).id;
        tc.ctx.values.block_params[pid].type_id = ptr_ty;
        let p = ValueId::BlockParam(pid);

        let (gep, add) = {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            let c60 = b.context_mut().get_const(0x60, 8).id();
            let gep = b.push_gep(p, 0x60).id(); // gep(p + 0x60)
            let add = b.push_add(p, c60).id(); // p + 0x60
            unsafe { b.dont_finalize() };
            (gep, add)
        };

        let nb = precompute_forms(&tc.ctx, fun);

        // Both decompose to the same affine base+offset, so memory forwarding
        // treats them as the same cell.
        assert_eq!(nb.base_offset(gep), Some((p, 0x60)));
        assert_eq!(nb.base_offset(add), nb.base_offset(gep));
        // The gep is still recognised as built on `p`.
        assert!(nb.affine_mentions(gep, p));
    }
}

/// Memoize the affine form of `v`, recursing into operands first so that nested
/// pointer arithmetic (e.g. `(p + 4) - 4`) fully decomposes. A placeholder leaf
/// is inserted before recursing to break any operand cycle.
fn ensure_form(ctx: &Context, v: ValueId, numbering: &mut Numbering) {
    if numbering.forms.contains_key(&v) {
        return;
    }
    let ValueId::Instruction(id) = v else {
        return;
    };
    let insn = ctx.get_insn(id);
    let width = insn.size();
    let mnemonic = insn.mnemonic().clone();
    numbering.forms.insert(v, leaf(v, width));
    for arg in mnemonic.args() {
        ensure_form(ctx, arg, numbering);
    }
    let form = arith_form(ctx, v, &mnemonic, width, numbering);
    numbering.forms.insert(v, form);
}
