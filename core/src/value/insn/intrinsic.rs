//! Pure intrinsic functions: named, side-effect-free operations such as `rol`
//! and `ror`.
//!
//! An intrinsic is represented by a single [`Mnemonic::Intrinsic`] variant
//! carrying an [`IntrinsicId`] plus its operands — there is no dedicated
//! mnemonic per intrinsic and no control-flow call. Semantics live in an
//! [`Intrinsic`] definition looked up from a process-global registry.
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

use rustc_hash::FxHashMap as HashMap;
use std::sync::OnceLock;

use super::binop::IntBinop;
use super::mnemonic::{Args, MnemonicKind};
use crate::{
    types::{TypeId, TypeManager},
    value::{InstructionId, ValueId, ValueRef, util::base_ref::HostRef},
};
use smallvec::SmallVec;

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
        self.desc().name()
    }

    /// The [`Intrinsic`] definition this id resolves to.
    pub fn desc(self) -> &'static dyn Intrinsic {
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

/// The definition of one *kind* of intrinsic — its name, arity, result typing,
/// shared evaluator, and optional recognition / simplification behaviour.
///
/// An [`IntrinsicId`] is a by-name handle resolving to a single `&'static dyn
/// Intrinsic`; the [`IntrinsicApp`] mnemonic is an *application* of that
/// definition to operands. Built-in definitions are unit structs registered via
/// [`register_intrinsic!`](crate::register_intrinsic).
pub trait Intrinsic: Sync {
    /// Textual name, e.g. `"rol"`. Unique across the registry.
    fn name(&self) -> &'static str;

    /// Number of operands the intrinsic takes.
    fn arity(&self) -> usize;

    /// The result type for an application to operands of types `args`. A sized
    /// integer is just a type, so a width-only intrinsic returns
    /// `types.get_or_make_int(width)`; an array-producing one returns the array
    /// type.
    fn result_type(&self, types: &TypeManager, args: &[TypeId]) -> TypeId;

    /// Evaluate on concrete operands `(bits, byte_width)`, producing an
    /// `out_size`-byte result. `None` means "not foldable / trap" — constant
    /// folding bails, the emulator raises.
    fn eval(&self, args: &[(u128, usize)], out_size: usize) -> Option<u128>;

    /// IR shape this intrinsic's idiom roots at, if it participates in
    /// recognition.
    fn root_op(&self) -> Option<RootOp> {
        None
    }

    /// Recognize the raw-IR idiom rooted at `at`, returning the intrinsic's
    /// operands when the instruction matches. Reads the IR through a [`HostRef`]
    /// (so a checked-out function's own SSA values resolve) and mints any derived
    /// operand literals through the shared interner (e.g. a rotate amount
    /// recovered as the `log2` of a strength-reduced multiplier).
    fn recognize(&self, _host: HostRef, _at: InstructionId) -> Option<Vec<ValueId>> {
        None
    }

    /// Algebraic simplification on the intrinsic's own operands — e.g.
    /// `rol(x, 0) → x` or `rol(a, c) → rol(a, c mod bits)`. Receives the
    /// applied [`IntrinsicId`] (so a shared simplifier can branch on `rol` vs
    /// `ror`), the result byte width, and the operands. Reads through a
    /// [`HostRef`] and mints replacement literals through the shared interner.
    fn simplify(
        &self,
        _host: HostRef,
        _id: IntrinsicId,
        _out_size: usize,
        _args: &[ValueId],
    ) -> Option<Simplified> {
        None
    }
}

/// The result of an intrinsic's [`simplify`](Intrinsic::simplify) hook.
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
pub struct IntrinsicRegistration(pub &'static dyn Intrinsic);

inventory::collect!(IntrinsicRegistration);

struct Registry {
    /// Definitions indexed by [`IntrinsicId`].
    descs: Vec<&'static dyn Intrinsic>,
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
        let mut descs: Vec<&'static dyn Intrinsic> = inventory::iter::<IntrinsicRegistration>()
            .map(|r| r.0)
            .collect();
        descs.sort_by_key(|d| d.name());

        let mut by_name = HashMap::default();
        let mut by_root: HashMap<RootOp, Vec<IntrinsicId>> = HashMap::default();
        for (idx, desc) in descs.iter().enumerate() {
            let id = IntrinsicId(idx);
            let prev = by_name.insert(desc.name(), id);
            assert!(
                prev.is_none(),
                "duplicate intrinsic registration: {}",
                desc.name()
            );
            if let Some(root) = desc.root_op() {
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
pub struct IntrinsicApp {
    pub id: IntrinsicId,
    pub args: Vec<ValueId>,
}

impl MnemonicKind for IntrinsicApp {
    fn opcode(&self) -> &'static str {
        self.id.name()
    }

    fn args(&self) -> Args {
        SmallVec::from_vec(self.args.clone())
    }
}

// ---------------------------------------------------------------------------
// Registration macro
// ---------------------------------------------------------------------------

/// Register a built-in intrinsic with the global registry.
///
/// Takes a unit-struct value implementing [`Intrinsic`]; the registry stores it
/// as a `&'static dyn Intrinsic`.
///
/// ```ignore
/// struct Rol;
/// impl Intrinsic for Rol { /* … */ }
/// register_intrinsic!(Rol);
/// ```
#[macro_export]
macro_rules! register_intrinsic {
    ($def:expr $(,)?) => {
        inventory::submit! {
            $crate::value::insn::IntrinsicRegistration(&$def)
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
pub(crate) fn const_u64(host: HostRef, v: ValueId) -> Option<u64> {
    match ValueRef::from_host(host, v) {
        ValueRef::Literal(lit) => {
            let ValueId::Literal(id) = v else {
                return None;
            };
            if host.shr().values.literals[id].symbolic.is_some() {
                return None;
            }
            Some(lit.value())
        }
        _ => None,
    }
}

/// If `v` is defined by an `IntBinop::want`, return its `(lhs, rhs)`.
pub(crate) fn as_int_binop(
    host: HostRef,
    v: ValueId,
    want: IntBinop,
) -> Option<(ValueId, ValueId)> {
    use super::{Binary, Binop, Mnemonic};
    let ValueId::Instruction(id) = v else {
        return None;
    };
    match host.instruction(id).mnemonic() {
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
