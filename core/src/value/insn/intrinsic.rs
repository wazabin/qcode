//! Pure intrinsic functions: named, side-effect-free operations such as `rol`
//! and `ror`.
//!
//! An intrinsic is represented by a single [`Mnemonic::Intrinsic`] variant
//! carrying an [`IntrinsicId`] plus its operands — there is no dedicated
//! mnemonic per intrinsic and no control-flow call. Semantics live in an
//! [`IntrinsicDesc`] looked up from a process-global registry.
//!
//! # Purity
//!
//! `Mnemonic::Intrinsic` is *categorically pure*: it has no memory effects and
//! no observable side effects. Passes treat the variant itself as the purity
//! contract (DCE may drop an unused intrinsic; GVN/CSE may dedup one). Anything
//! impure (syscalls, `rdtsc`, …) stays a [`PCodeOp`](super::PCodeOp).
//!
//! # Registry
//!
//! Built-in intrinsics self-register with [`inventory`] via
//! [`register_intrinsic!`], mirroring the pass registry. The id-indexed table
//! and the name→id map are built once, lazily, from `inventory::iter`. An
//! [`IntrinsicId`] is a runtime handle (cheap to copy/match/hash) but
//! *serializes by name*, so the on-disk form is stable regardless of link
//! order and an unknown name is a clean deserialize error.

use std::collections::HashMap;
use std::sync::OnceLock;

use super::binop::IntBinop;
use super::mnemonic::MnemonicKind;
use crate::{
    context::Context,
    value::{InstructionId, ValueId, ValueRef},
};

/// A stable-by-name handle into the intrinsic registry.
///
/// Cheap to copy, match, and hash in hot pass code; serialized as the
/// intrinsic's name so the id is never written to disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntrinsicId(usize);

impl IntrinsicId {
    /// Resolve an intrinsic by name, or `None` if no intrinsic is registered
    /// under `name`.
    pub fn from_name(name: &str) -> Option<Self> {
        registry().by_name.get(name).copied()
    }

    /// The registered name of this intrinsic (e.g. `"rol"`).
    pub fn name(self) -> &'static str {
        self.desc().name
    }

    /// The full descriptor for this intrinsic.
    pub fn desc(self) -> &'static IntrinsicDesc {
        registry().descs[self.0]
    }
}

impl serde::Serialize for IntrinsicId {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

impl<'de> serde::Deserialize<'de> for IntrinsicId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let name = <std::borrow::Cow<'de, str>>::deserialize(d)?;
        IntrinsicId::from_name(&name)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown intrinsic `{name}`")))
    }
}

/// The IR shape an intrinsic's idiom is rooted at, used to gate recognition so
/// only relevant recognizers run per instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RootOp {
    /// Rooted at an integer binop (e.g. `rol`/`ror` root at `IntBinop::Or`).
    IntBinop(IntBinop),
}

/// Evaluator for an intrinsic: concrete operands `(bits, byte_width)` and an
/// `out_size`-byte result. `None` means "not foldable / trap".
pub type IntrinsicEval = fn(&[(u128, usize)], usize) -> Option<u128>;

/// Recognizer for an intrinsic's raw-IR idiom, returning the intrinsic's
/// operands when the instruction matches.
pub type IntrinsicRecognize = fn(&Context, InstructionId) -> Option<Vec<ValueId>>;

/// Algebraic simplifier for an intrinsic. Receives the intrinsic's
/// [`IntrinsicId`] (so a shared simplifier can branch on which intrinsic it is,
/// e.g. `rol` vs `ror`), the result byte width, and the operands; returns a
/// [`Simplified`] outcome when a rewrite applies.
pub type IntrinsicSimplify = fn(&mut Context, IntrinsicId, usize, &[ValueId]) -> Option<Simplified>;

/// Static description of one intrinsic: its name, arity, result-size rule, the
/// shared evaluator (used by both constant folding and the emulator), and
/// optional recognition / simplification hooks.
pub struct IntrinsicDesc {
    /// Textual name, e.g. `"rol"`. Unique across the registry.
    pub name: &'static str,
    /// Number of operands the intrinsic takes.
    pub arity: usize,
    /// Computes the result byte width from the operand byte widths.
    pub result_size: fn(&[usize]) -> usize,
    /// Evaluate on concrete operands. `None` means "not foldable / trap" — fold
    /// bails, the emulator raises.
    pub eval: IntrinsicEval,
    /// IR shape this intrinsic's idiom roots at, if it participates in
    /// recognition.
    pub root_op: Option<RootOp>,
    /// Recognize the raw-IR idiom rooted at the given instruction.
    pub recognize: Option<IntrinsicRecognize>,
    /// Algebraic simplification on the intrinsic's own operands — e.g.
    /// `rol(x, 0) → x` or `rol(a, c) → rol(a, c mod bits)`.
    pub simplify: Option<IntrinsicSimplify>,
}

/// The result of an intrinsic's [`simplify`](IntrinsicDesc::simplify) hook.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Simplified {
    /// Forward all uses of the intrinsic to this existing value, e.g.
    /// `rol(x, 0) → x` or `ror(rol(a, c), c) → a`.
    Value(ValueId),
    /// Replace the intrinsic instruction with a new expression of the same
    /// width. The mnemonic is free to be a different intrinsic (e.g.
    /// `rol(a, c) → ror(a, bits - c)`) or a plain operation (e.g. an
    /// [`IntBinop`]-rooted `+`).
    Expression(super::Mnemonic),
}

/// One intrinsic's registration, submitted via [`inventory::submit!`] (see
/// [`register_intrinsic!`]) and collected into the global registry.
pub struct IntrinsicRegistration(pub IntrinsicDesc);

inventory::collect!(IntrinsicRegistration);

struct Registry {
    /// Descriptors indexed by [`IntrinsicId`].
    descs: Vec<&'static IntrinsicDesc>,
    /// Name → id, for parsing and serde.
    by_name: HashMap<&'static str, IntrinsicId>,
    /// Ids grouped by recognition root, so the recognizer pass can fetch only
    /// the relevant ones per instruction.
    by_root: HashMap<RootOp, Vec<IntrinsicId>>,
}

fn registry() -> &'static Registry {
    static REG: OnceLock<Registry> = OnceLock::new();
    REG.get_or_init(|| {
        // Sort by name so ids are deterministic within a build regardless of
        // inventory iteration order.
        let mut descs: Vec<&'static IntrinsicDesc> = inventory::iter::<IntrinsicRegistration>()
            .map(|r| &r.0)
            .collect();
        descs.sort_by_key(|d| d.name);

        let mut by_name = HashMap::new();
        let mut by_root: HashMap<RootOp, Vec<IntrinsicId>> = HashMap::new();
        for (idx, desc) in descs.iter().enumerate() {
            let id = IntrinsicId(idx);
            let prev = by_name.insert(desc.name, id);
            assert!(
                prev.is_none(),
                "duplicate intrinsic registration: {}",
                desc.name
            );
            if let Some(root) = desc.root_op {
                by_root.entry(root).or_default().push(id);
            }
        }

        Registry {
            descs,
            by_name,
            by_root,
        }
    })
}

/// Every intrinsic whose recognition idiom roots at `root`. Empty if none.
pub fn recognizers_for(root: RootOp) -> &'static [IntrinsicId] {
    registry()
        .by_root
        .get(&root)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

/// A pure intrinsic instruction: an [`IntrinsicId`] applied to its operands.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Intrinsic {
    pub id: IntrinsicId,
    pub args: Vec<ValueId>,
}

impl MnemonicKind for Intrinsic {
    fn opcode(&self) -> &'static str {
        self.id.name()
    }

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        let args = self
            .args
            .iter()
            .map(|&arg| ValueRef::new(arg, ctx).to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "${}({});", self.id.name(), args)
    }

    fn args(&self) -> Vec<ValueId> {
        self.args.clone()
    }
}

// ---------------------------------------------------------------------------
// Registration macro
// ---------------------------------------------------------------------------

/// Register a built-in intrinsic with the global registry.
///
/// ```ignore
/// register_intrinsic! {
///     name: "rol",
///     arity: 2,
///     result_size: |sz| sz[0],
///     eval: eval_rol,
///     root_op: Some(RootOp::IntBinop(IntBinop::Or)),
///     recognize: Some(recognize_rol),
///     simplify: Some(simplify_rotate),
/// }
/// ```
#[macro_export]
macro_rules! register_intrinsic {
    (
        name: $name:expr,
        arity: $arity:expr,
        result_size: $result_size:expr,
        eval: $eval:expr,
        root_op: $root_op:expr,
        recognize: $recognize:expr,
        simplify: $simplify:expr $(,)?
    ) => {
        inventory::submit! {
            $crate::value::insn::IntrinsicRegistration($crate::value::insn::IntrinsicDesc {
                name: $name,
                arity: $arity,
                result_size: $result_size,
                eval: $eval,
                root_op: $root_op,
                recognize: $recognize,
                simplify: $simplify,
            })
        }
    };
}

// ---------------------------------------------------------------------------
// Helpers shared by the built-in intrinsics (see `crate::intrinsics`)
// ---------------------------------------------------------------------------

/// The unsigned mask for a `bytes`-wide value, saturating at 128 bits.
pub(crate) fn mask_for(bytes: usize) -> u128 {
    let bits = (bytes * 8).min(128);
    if bits == 0 {
        0
    } else if bits == 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

/// The non-symbolic constant value of `v`, or `None`.
pub(crate) fn const_u64(ctx: &Context, v: ValueId) -> Option<u64> {
    match ValueRef::new(v, ctx) {
        ValueRef::Literal(lit) => {
            let ValueId::Literal(id) = v else {
                return None;
            };
            if ctx.values.literals[id].symbolic.is_some() {
                return None;
            }
            Some(lit.value())
        }
        _ => None,
    }
}

/// If `v` is defined by an `IntBinop::want`, return its `(lhs, rhs)`.
pub(crate) fn as_int_binop(
    ctx: &Context,
    v: ValueId,
    want: IntBinop,
) -> Option<(ValueId, ValueId)> {
    use super::{Binary, Binop, Mnemonic};
    let ValueId::Instruction(id) = v else {
        return None;
    };
    match ctx.get_insn(id).mnemonic() {
        Mnemonic::Binop(Binary {
            lhs,
            rhs,
            op: Binop::Int(op),
        }) if *op == want => Some((*lhs, *rhs)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intrinsic_id_serializes_by_name() {
        let config = bincode::config::standard();
        let rol = IntrinsicId::from_name("rol").unwrap();

        // Serialized form is the name string, so it round-trips by name.
        let bytes = bincode::serde::encode_to_vec(rol, config).unwrap();
        let name: String = bincode::serde::decode_from_slice(&bytes, config).unwrap().0;
        assert_eq!(name, "rol");

        let (back, _): (IntrinsicId, usize) =
            bincode::serde::decode_from_slice(&bytes, config).unwrap();
        assert_eq!(back, rol);

        // An unknown name is a clean error, not a corrupt id.
        let bad = bincode::serde::encode_to_vec("nope", config).unwrap();
        assert!(bincode::serde::decode_from_slice::<IntrinsicId, _>(&bad, config).is_err());
    }
}
