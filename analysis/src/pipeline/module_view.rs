//! The types that give a function pass its restricted, parallel-safe view of the
//! world (Stage 5 of the parallel-function-passes plan; see `PARALLEL_PASSES.md`).
//!
//! A function pass reads the module's *published interface* through a shared
//! [`ModuleView`] and mutates *only its own function* through a `&mut`
//! [`FunctionBody`]. Effects on global state that a few passes legitimately need
//! (assumptions, discoveries, self-renames, address claims) are **buffered** in
//! [`Effects`] and drained by the driver at check-in, so the pass itself touches
//! no global mutable state — which is what lets workers run in parallel with the
//! `ModuleView` `&`-shared and the bodies disjoint `&mut`.
//!
//! In Stage 5 the driver drives this sequentially (checkout → run → check-in for
//! one function at a time); Stage 6 runs the checkouts on `std::thread::scope`
//! workers. The types are identical either way.

use std::borrow::Cow;

use qcode::{
    assumption::Proposition,
    context::Context,
    discovery::Discovery,
    value::{Function, FunctionId, ValueId},
};

use super::PipelineEnv;

/// Global effects a function pass requests, buffered for the driver to apply at
/// check-in (in worklist order). Everything here is either an idempotent keyed
/// insert or a first-writer-wins claim, so replay is deterministic and needs no
/// merge heuristics.
#[derive(Default)]
pub struct Effects<'str> {
    /// `assume_true(prop)` / `assume(prop, value)` requests (from
    /// `handle_jump_tables`' `ImmutableMemory` assumption).
    pub assumptions: Vec<(Proposition, bool)>,
    /// `discover_code` requests (keyed, idempotent) — new code addresses the
    /// jump-table pass resolved.
    pub discoveries: Vec<Discovery>,
    /// A buffered self-rename claim (from `cpp_demangle` / `name_thunks`),
    /// applied as a `get_unique_name` claim at check-in.
    pub self_rename: Option<Cow<'str, str>>,
    /// New machine-address → value claims (blocks minted at a machine address),
    /// applied into `address_map` with `set_address`'s priority rule.
    pub address_claims: Vec<(u64, ValueId)>,
}

impl<'str> Effects<'str> {
    /// Whether nothing has been buffered (the common case — most passes request
    /// no global effects at all).
    pub fn is_empty(&self) -> bool {
        self.assumptions.is_empty()
            && self.discoveries.is_empty()
            && self.self_rename.is_none()
            && self.address_claims.is_empty()
    }

    /// Buffer an `assume_true(prop)` request.
    pub fn assume_true(&mut self, prop: Proposition) {
        self.assumptions.push((prop, true));
    }

    /// Buffer a `discover` request.
    pub fn discover(&mut self, discovery: Discovery) {
        self.discoveries.push(discovery);
    }

    /// Buffer a self-rename claim (last writer wins within one run; the driver
    /// resolves it to a unique name at check-in).
    pub fn rename_self(&mut self, name: Cow<'str, str>) {
        self.self_rename = Some(name);
    }

    /// Buffer a machine-address claim for a value defined this run.
    pub fn claim_address(&mut self, addr: u64, value: ValueId) {
        self.address_claims.push((addr, value));
    }
}

/// The read-only module interface a function pass may consult: architecture /
/// ABI, the interners (mintable through `&self`), and every *other* function's
/// published interface (name, address, signature, purity, clobber/write
/// summaries) — but **not** other functions' in-flight bodies (except the
/// pure-bodies view, which is stage-invariant; added with the gvn port).
///
/// v1 wraps `&Context` and `&PipelineEnv`. Because the driver checks the pass's
/// own function *out* of the context before building the view, reading the
/// context here never aliases the `&mut FunctionBody` the pass also holds.
pub struct ModuleView<'ctx, 'str> {
    ctx: &'ctx Context<'str>,
    env: &'ctx PipelineEnv,
}

impl<'ctx, 'str> ModuleView<'ctx, 'str> {
    /// Build a view over `ctx` (with the pass's own function checked out) and the
    /// pipeline environment.
    pub fn new(ctx: &'ctx Context<'str>, env: &'ctx PipelineEnv) -> Self {
        Self { ctx, env }
    }

    /// The pipeline environment (register layout, ABI, OS, bitness, stack
    /// pointer, shared alias base).
    pub fn env(&self) -> &'ctx PipelineEnv {
        self.env
    }

    /// The underlying context, for the read-only module queries a pass makes
    /// (interners, registers, spaces, memory image, other functions' published
    /// interface, truths). The pass's own function is absent here.
    pub fn ctx(&self) -> &'ctx Context<'str> {
        self.ctx
    }
}

/// The pass's own function, checked out of the module so the pass owns it
/// exclusively, plus the [`Effects`] buffer and the function-minting pool.
///
/// The function is moved out of the registry at checkout and reinstalled at
/// check-in; this exclusive ownership is what lets parallel workers hold disjoint
/// `&mut FunctionBody`s.
pub struct FunctionBody<'str> {
    /// This function's id (the registry key; a checked-out [`Function`] does not
    /// store its own id).
    id: FunctionId,
    /// The checked-out function (its arenas, roster, root, users, names).
    fun: Function<'str>,
    /// Global effects buffered this run (drained by the driver at check-in).
    effects: Effects<'str>,
    /// Never-observed placeholder [`FunctionId`]s the pass may materialize new
    /// functions into (loop outliners mint exactly one). Unused ids return to the
    /// driver's pool at check-in.
    reserved_ids: Vec<FunctionId>,
    /// Functions built this run against `reserved_ids`, installed at check-in.
    minted: Vec<Function<'str>>,
}

impl<'str> FunctionBody<'str> {
    /// Wrap the checked-out function `fun` (id `id`), carrying `reserved_ids` for
    /// any function it mints.
    pub fn new(id: FunctionId, fun: Function<'str>, reserved_ids: Vec<FunctionId>) -> Self {
        Self {
            id,
            fun,
            effects: Effects::default(),
            reserved_ids,
            minted: Vec::new(),
        }
    }

    /// This function's id.
    pub fn id(&self) -> FunctionId {
        self.id
    }

    /// The owned function (read).
    pub fn function(&self) -> &Function<'str> {
        &self.fun
    }

    /// The owned function (mutate).
    pub fn function_mut(&mut self) -> &mut Function<'str> {
        &mut self.fun
    }

    /// The effect buffer (mutate) — passes push assumption/discovery/rename/
    /// address claims here instead of touching global state.
    pub fn effects_mut(&mut self) -> &mut Effects<'str> {
        &mut self.effects
    }

    /// How many reserved ids remain for minting.
    pub fn reserved_remaining(&self) -> usize {
        self.reserved_ids.len()
    }

    /// Consume the body at check-in, yielding the reinstallable function, its
    /// buffered effects, the functions it minted, and any unused reserved ids
    /// (returned to the driver's pool).
    pub fn into_parts(
        self,
    ) -> (
        Function<'str>,
        Effects<'str>,
        Vec<Function<'str>>,
        Vec<FunctionId>,
    ) {
        (self.fun, self.effects, self.minted, self.reserved_ids)
    }
}
