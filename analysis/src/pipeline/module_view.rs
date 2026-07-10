//! The types that give a function pass its restricted, parallel-safe view of the
//! world (Stage 5 of the parallel-function-passes plan; see `PARALLEL_PASSES.md`).
//!
//! A function pass reads the module's *published interface* through a shared
//! [`ModuleView`] and mutates *only its own function* through a `&mut`
//! [`FunctionBody`]. The one effect on global state a pass legitimately needs (a
//! self-rename) is **buffered** in [`Effects`] and drained by the driver at
//! check-in, so the pass itself touches no global mutable state — which is what
//! lets workers run in parallel with the `ModuleView` `&`-shared and the bodies
//! disjoint `&mut`.
//!
//! In Stage 5 the driver drives this sequentially (checkout → run → check-in for
//! one function at a time); Stage 6 runs the checkouts on `std::thread::scope`
//! workers. The types are identical either way.

use std::borrow::Cow;

use qcode::{
    context::Context,
    value::{
        Function, FunctionId, FunctionKind,
        function::FunctionInterface,
        util::{base_ref::HostRef, host_mut::CheckedOut},
    },
};

use super::PipelineEnv;

/// A function minted by a pass this run: its reserved id, its interface, and its
/// body. Installed into the reserved slot by the driver at check-in.
pub type Minted<'str> = (FunctionId, FunctionInterface<'str>, Function<'str>);

/// Global effects a function pass requests, buffered for the driver to apply at
/// check-in (in worklist order). The self-rename is a first-writer-wins claim, so
/// replay is deterministic and needs no merge heuristics.
#[derive(Default)]
pub struct Effects<'str> {
    /// A buffered self-rename claim (from `cpp_demangle` / `name_thunks`),
    /// applied as a `get_unique_name` claim at check-in.
    pub self_rename: Option<Cow<'str, str>>,
}

impl<'str> Effects<'str> {
    /// Buffer a self-rename claim (last writer wins within one run; the driver
    /// resolves it to a unique name at check-in).
    pub fn rename_self(&mut self, name: Cow<'str, str>) {
        self.self_rename = Some(name);
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
///
/// This is the target `ContextView` of the context-split migration (stage 5b):
/// morally a bodies-free view of the module (arch, interners, interfaces, env),
/// transitionally still wrapping a whole `&Context` until the `split()` reshape
/// (5b-ii) narrows it to `&Shared`. It is `Copy` (it holds only shared
/// references), so a worker hands the same view to every helper and sub-ref.
#[derive(Clone, Copy)]
pub struct ContextView<'ctx, 'str> {
    ctx: &'ctx Context<'str>,
    env: &'ctx PipelineEnv,
}

/// Transitional alias: the old name for [`ContextView`]. Removed once every call
/// site is migrated (stage 5b-i.3).
pub type ModuleView<'ctx, 'str> = ContextView<'ctx, 'str>;

impl<'ctx, 'str> ContextView<'ctx, 'str> {
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

    /// The published interface of function `f` (name, address, signature, purity,
    /// clobber/write summaries) — the caller-reasoning surface. Interfaces are
    /// never checked out, so this always reads the shared registry.
    pub fn interface(&self, f: FunctionId) -> &'ctx FunctionInterface<'str> {
        &self.ctx.values.interfaces[f]
    }

    /// The underlying whole `&Context` — the transitional escape hatch for the
    /// read-only module queries a pass makes (interners, registers, spaces,
    /// memory image, other functions' published interface, truths) that do not
    /// yet have a narrowed accessor. Narrowed to `&Shared` in stage 5b-ii; every
    /// caller that still needs the whole context is a migration TODO.
    pub fn shared_ctx(&self) -> &'ctx Context<'str> {
        self.ctx
    }

    /// The underlying context, for the read-only module queries a pass makes
    /// (interners, registers, spaces, memory image, other functions' published
    /// interface, truths). The pass's own function is absent here.
    ///
    /// Transitional twin of [`shared_ctx`](Self::shared_ctx); call sites migrate
    /// to `shared_ctx` in stage 5b-i.3, after which this method is removed.
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
    /// Functions built this run against drawn `reserved_ids` (paired with the id
    /// each was drawn for, and its interface), installed by the driver at
    /// check-in.
    minted: Vec<Minted<'str>>,
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

    /// A [`CheckedOut`] mutation host over this body's owned function and the
    /// module's read-only shared context. This is how a `FunctionPass` reads
    /// (via [`CheckedOut::read_host`]) and mutates (via the [`HostMut`] surface)
    /// its function — construct block/instruction refs and `Builder`s over it.
    ///
    /// [`HostMut`]: qcode::value::util::host_mut::HostMut
    pub fn host<'a>(&'a mut self, m: &'a ModuleView<'_, 'str>) -> CheckedOut<'a, 'str> {
        CheckedOut::new(&mut self.fun, self.id, m.ctx())
    }

    /// The effect buffer (mutate) — passes push a self-rename claim here instead of
    /// touching global state.
    pub fn effects_mut(&mut self) -> &mut Effects<'str> {
        &mut self.effects
    }

    /// A `Copy` read view over this body's owned function and the shared context
    /// — the recognizer-side twin of [`host`](Self::host) for passes that only
    /// need to *read* while holding other borrows.
    pub fn read_host<'a>(&'a self, m: &'a ModuleView<'_, 'str>) -> HostRef<'a, 'str> {
        HostRef::Checked {
            fun: &self.fun,
            shared: m.ctx(),
            id: self.id,
        }
    }

    /// Mint a new function (`PARALLEL_PASSES.md` ruling 3): draw one reserved id
    /// from the pool the driver assigned this checkout, create a detached
    /// [`Function`] shell under `name` (buffered **raw** — global uniquification
    /// happens when the driver installs it at check-in) with the given `kind`,
    /// and return its id. `pure` marks it a deterministic pure function
    /// (`is_pure` + the implied `pure_reg`), which every current outliner's body
    /// is. Build the body through [`host_with_minted`](Self::host_with_minted).
    ///
    /// Returns `None` when the reservation pool is exhausted — the calling pass
    /// then simply stops promoting (skips its remaining candidates).
    pub fn mint_function(
        &mut self,
        name: Cow<'str, str>,
        kind: FunctionKind,
        pure: bool,
    ) -> Option<FunctionId> {
        if self.reserved_ids.is_empty() {
            return None;
        }
        let id = self.reserved_ids.remove(0);
        let mut interface = FunctionInterface::new(name);
        interface.kind = kind;
        if pure {
            let sig = interface.signature.get_or_insert_default();
            sig.is_pure = true;
            // Full purity implies register purity — the GUI badge keys off the
            // latter (mirrors `outline_core` / `make_lambda`).
            sig.pure_reg = true;
        }
        self.minted.push((id, interface, Function::empty_body()));
        Some(id)
    }

    /// Split this body into a read view of the *own* function and an exclusive
    /// [`CheckedOut`] mutation host over the minted function `minted` (a
    /// [`mint_function`](Self::mint_function) result). This is how an outliner
    /// builds a minted body: it clones expression slices out of its own function
    /// (read) into the minted one (write), both against the same shared context.
    ///
    /// Panics if `minted` was not minted by this body.
    pub fn host_with_minted<'a>(
        &'a mut self,
        m: &'a ModuleView<'_, 'str>,
        minted: FunctionId,
    ) -> (HostRef<'a, 'str>, CheckedOut<'a, 'str>) {
        let own = HostRef::Checked {
            fun: &self.fun,
            shared: m.ctx(),
            id: self.id,
        };
        let fun = self
            .minted
            .iter_mut()
            .find(|(id, _, _)| *id == minted)
            .map(|(_, _, f)| f)
            .expect("host_with_minted: not a function minted by this body");
        (own, CheckedOut::new(fun, minted, m.ctx()))
    }

    /// Consume the body at check-in, yielding the reinstallable function, its
    /// buffered effects, the functions it minted, and any unused reserved ids
    /// (returned to the driver's pool).
    pub fn into_parts(
        self,
    ) -> (
        Function<'str>,
        Effects<'str>,
        Vec<Minted<'str>>,
        Vec<FunctionId>,
    ) {
        (self.fun, self.effects, self.minted, self.reserved_ids)
    }
}
