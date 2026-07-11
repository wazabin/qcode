use crate::{
    assumption::{KnownContradiction, Proposition, Truth, Violation},
    types::TypeId,
    value::{
        bytes::{Bytes, BytesId},
        function::FunctionId,
        insn::InstructionId,
        interner::{Interner, LiteralInterner},
        literal::{Literal, LiteralId},
        varnode::{Varnode, VarnodeId},
    },
};
// NOTE (IR-ownership refactor): instruction/block/param/edge *storage* lives in
// each `Function` (see `Function::insns/blocks/params/edges`), and the function
// bodies/interfaces now live directly on [`Context`](crate::context::Context)
// (`bodies`/`interfaces`). This registry keeps only the global value arenas
// (literals, bytes, varnodes) plus the cross-function maps (`call_sites`,
// `synthetic_callees`, truths). The composite-id routing accessors that consult
// the function bodies moved onto `Context` in the context-split reshape.
use jstd::registry::Registry;
use rustc_hash::FxHashMap as HashMap;
use std::collections::BTreeSet;

/// Central storage arena for all IR values in a [`Context`](crate::context::Context).
///
/// Each field is a typed arena ([`Registry`]) keyed by the corresponding ID
/// type. Values are append-only: once pushed, their ID is stable for the
/// lifetime of the registry and their data is never moved.
///
/// # Invariants
///
/// - **`push_insn` is final.** Instructions are immutable after insertion.
///   The owning function's `users` map is populated at push time from the
///   instruction's operands and is not updated if operands are later altered via
///   interior mutation. Use
///   [`Context::replace_all_uses_with`](crate::context::Context::replace_all_uses_with)
///   to rewrite operands while keeping `users` consistent.
///
/// - **`users` is managed internally.** The reverse use-def map now lives in
///   each [`Function`] (function-scoped; see [`Function::users`]). Do not mutate
///   it directly. Read it through
///   [`FunctionRef::users_of`](crate::value::FunctionRef::users_of) /
///   [`Context::users`](crate::context::Context::users), and remove dead
///   instructions via
///   [`Context::remove_instructions`](crate::context::Context::remove_instructions).
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ValueRegistry<'str> {
    /// Literal (constant) interner. Behind an `RwLock` (see
    /// [`LiteralInterner`]) so constants can be minted through a shared `&`; the
    /// dedup cache lives inside it.
    pub literals: LiteralInterner,

    /// Opaque byte-blob constant interner (constants wider than a `u64`).
    #[serde(default)]
    pub bytes: Interner<BytesId, Bytes>,

    /// User-forced rendering overrides for `Bytes` blobs (e.g. from the GUI
    /// Strings pane). Absent entries render under [`BytesDisplay::Auto`].
    #[serde(default)]
    pub(crate) bytes_display: HashMap<BytesId, crate::value::BytesDisplay>,

    /// Varnode storage.
    pub varnodes: Registry<VarnodeId, Varnode<'str>>,

    /// Per-varnode type overrides. A varnode is normally typed `Int(size)`; an
    /// entry here gives it a richer *global* type instead (e.g. the `FS_OFFSET`
    /// register typed `PtrTo<TEB>` by the TEB-seeding pass). Consulted by
    /// [`Context::type_of`](crate::context::Context::type_of) /
    /// [`stored_type_of`](crate::context::Context::stored_type_of). Set via
    /// [`Context::set_varnode_type`](crate::context::Context::set_varnode_type).
    #[serde(default)]
    pub(crate) varnode_types: HashMap<VarnodeId, TypeId>,

    /// Truth map of the assumption system: what each [`Proposition`] is
    /// currently assumed or known to be (see [`crate::assumption`]). Accessed
    /// through [`Context::assume_true`](crate::context::Context::assume_true)
    /// and friends.
    pub(crate) truths: HashMap<Proposition, Truth>,

    /// Proven facts that contradicted an assumption this round; non-empty means
    /// the checkpoint+replay driver must discard this working copy and replay.
    pub(crate) violations: Vec<Violation>,

    /// Proven facts that contradicted an existing *known* fact this round (e.g. a
    /// user override the analysis disproved). A hard error for the driver, not a
    /// replay signal. Transient per round, so not serialized.
    #[serde(default, skip)]
    pub(crate) known_contradictions: Vec<KnownContradiction>,

    /// Reverse call graph: for each callee [`FunctionId`], the direct-call sites
    /// (instructions) that target it. Kept in sync alongside `users` by
    /// [`Context::push_insn`](crate::context::Context::push_insn),
    /// [`Context::remove_instructions`](crate::context::Context::remove_instructions), and
    /// [`Context::replace_instruction_mnemonic`](crate::context::Context::replace_instruction_mnemonic).
    /// Indirect calls have no static target and are not recorded here.
    #[serde(default)]
    pub(crate) call_sites: HashMap<FunctionId, Vec<InstructionId>>,

    /// Synthetic forward call-graph edges that are not backed by a direct `Call`
    /// instruction: `caller FunctionId → set of callee entry addresses`. Used for
    /// relationships a pass recovers but the IR can't express as a direct call —
    /// e.g. `entry → main`, where `main` is passed to `__libc_start_main` as a
    /// pointer argument rather than called. Keyed by address so the edge resolves
    /// once a function exists at the callee, independent of when it materializes.
    /// Merged into [`FunctionRef::callees`](crate::value::FunctionRef::callees).
    #[serde(default)]
    pub(crate) synthetic_callees: HashMap<FunctionId, BTreeSet<u64>>,
}

impl<'str> ValueRegistry<'str> {
    /// Returns a canonical [`LiteralId`] for the given typed constant.
    ///
    /// The value is masked to `type_id`'s size before lookup. Symbolic literals
    /// (created via [`push_literal`](Self::push_literal)) are not included in
    /// the intern cache and will not alias with constants produced here.
    ///
    /// Call [`Context::get_const`] for the common `Int(size)` case; use this
    /// method directly when you need to preserve a non-`Int` type (e.g.
    /// [`StackAddress`](crate::types::StackAddress)) through folding.
    pub fn get_or_make_typed_literal(&self, value: u64, type_id: TypeId, size: usize) -> LiteralId {
        self.literals
            .get_or_make_typed_literal(value, type_id, size)
    }

    /// Pushes a [`Literal`] with arbitrary fields (e.g. with a symbolic ref)
    /// without interning. Use [`get_or_make_typed_literal`](Self::get_or_make_typed_literal)
    /// for plain integer constants.
    pub fn push_literal(&self, literal: Literal) -> LiteralId {
        self.literals.push_literal(literal)
    }

    /// Returns the direct-call sites (instructions) targeting `callee`.
    pub fn call_sites_of(&self, callee: FunctionId) -> &[InstructionId] {
        self.call_sites
            .get(&callee)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Records a synthetic forward call-graph edge `caller → callee_addr` (see
    /// [`synthetic_callees`](Self::synthetic_callees)). Returns `true` if the
    /// edge was newly added, so callers can drive a fixpoint without spinning.
    pub fn add_synthetic_callee(&mut self, caller: FunctionId, callee_addr: u64) -> bool {
        self.synthetic_callees
            .entry(caller)
            .or_default()
            .insert(callee_addr)
    }

    /// Returns the synthetic callee entry addresses recorded for `caller`.
    pub fn synthetic_callees_of(&self, caller: FunctionId) -> impl Iterator<Item = u64> + '_ {
        self.synthetic_callees
            .get(&caller)
            .into_iter()
            .flatten()
            .copied()
    }

    pub fn push_varnode(&mut self, varnode: Varnode<'str>) -> VarnodeId {
        self.varnodes.push(varnode)
    }
}
