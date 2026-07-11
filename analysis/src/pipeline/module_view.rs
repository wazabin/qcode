//! The types that give a function pass its restricted, parallel-safe view of the
//! world (Stage 5 of the parallel-function-passes plan; see `PARALLEL_PASSES.md`).
//!
//! A function pass reads the module's *published interface* through a shared
//! [`ContextView`] and mutates *only its own function* through a `&mut`
//! [`FunctionBody`]. The one effect on global state a pass legitimately needs (a
//! self-rename) is **buffered** in [`Effects`] and drained by the driver at
//! check-in, so the pass itself touches no global mutable state — which is what
//! lets workers run in parallel with the `ContextView` `&`-shared and the bodies
//! disjoint `&mut`.
//!
//! In Stage 5 the driver drives this sequentially (checkout → run → check-in for
//! one function at a time); Stage 6 runs the checkouts on `std::thread::scope`
//! workers. The types are identical either way.

use std::borrow::Cow;

use qcode::{
    context::Context,
    error::Result,
    types::TypeId,
    value::{
        BlockParamRef, BlockRef, Function, FunctionId, FunctionKind, FunctionRef, InstructionRef,
        ValueId,
        block::{BasicBlock, BlockId, EdgeId},
        block_param::{BlockParam, BlockParamId},
        function::FunctionInterface,
        insn::{Instruction, InstructionId, Mnemonic},
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
        &self.ctx.interfaces[f]
    }

    /// The underlying whole `&Context` — the transitional escape hatch for the
    /// read-only module queries a pass makes (interners, registers, spaces,
    /// memory image, other functions' published interface, truths) that do not
    /// yet have a narrowed accessor. Narrowed to `&Shared` in stage 5b-ii; every
    /// caller that still needs the whole context is a migration TODO.
    pub fn shared_ctx(&self) -> &'ctx Context<'str> {
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
    pub fn host<'a>(&'a mut self, cx: ContextView<'a, 'str>) -> CheckedOut<'a, 'str> {
        CheckedOut::new(&mut self.fun, self.id, cx.shared_ctx())
    }

    /// The effect buffer (mutate) — passes push a self-rename claim here instead of
    /// touching global state.
    pub fn effects_mut(&mut self) -> &mut Effects<'str> {
        &mut self.effects
    }

    /// A `Copy` read view over this body's owned function and the shared context
    /// — the recognizer-side twin of [`host`](Self::host) for passes that only
    /// need to *read* while holding other borrows.
    pub fn read_host<'a>(&'a self, cx: ContextView<'a, 'str>) -> HostRef<'a, 'str> {
        HostRef::Checked {
            fun: &self.fun,
            shared: cx.shared_ctx(),
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
        cx: ContextView<'a, 'str>,
        minted: FunctionId,
    ) -> (HostRef<'a, 'str>, CheckedOut<'a, 'str>) {
        let own = HostRef::Checked {
            fun: &self.fun,
            shared: cx.shared_ctx(),
            id: self.id,
        };
        let fun = self
            .minted
            .iter_mut()
            .find(|(id, _, _)| *id == minted)
            .map(|(_, _, f)| f)
            .expect("host_with_minted: not a function minted by this body");
        (own, CheckedOut::new(fun, minted, cx.shared_ctx()))
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

/// Inherent verb + read-accessor surface (context-split stage 5b-ii(a)).
///
/// Every mutation verb of [`HostMut`] and every read accessor of [`HostRef`] a
/// function pass calls today through `f.host(cx)` / `f.read_host(cx)` is mirrored
/// here as an inherent method on the body itself: `body.verb(cx, …)` instead of
/// `host.verb(…)`. This commit is **purely additive** — each method is a
/// behaviour-identical delegation to a freshly built [`CheckedOut`] (for the
/// verbs) or [`HostRef`] (for the reads); no call site changes yet. Follow-on
/// commits migrate helpers off the generic `H: HostMut` onto this surface, and a
/// later commit reimplements the verb bodies directly on `self`'s arenas, at
/// which point the delegation disappears.
///
/// Where a `HostMut` verb takes an explicit `func: FunctionId` for the pass's own
/// function, the inherent method drops that parameter and supplies
/// [`self.id()`](Self::id) instead — a function pass only ever mints/mutates into
/// its own body.
impl<'str> FunctionBody<'str> {
    // ---- births -------------------------------------------------------------
    //
    // `push_edge` (`Context::push_edge`) is intentionally NOT mirrored: its
    // `EdgeData` parameter is `pub(crate)` in `qcode::value::block`, so it cannot
    // be named from this crate without making `EdgeData` public (a core design
    // change, out of this commit's additive scope). No pass calls `push_edge`
    // directly — edges are created through `add_cfg_edge` — so nothing needs it.

    /// Push a fresh instruction into this body's arena (recording operand uses and
    /// the call-site cache). Mirrors [`HostMut::push_insn`] with `func = self.id()`.
    pub fn push_insn(
        &mut self,
        cx: ContextView<'_, 'str>,
        insn: Instruction<'str>,
    ) -> InstructionId {
        let _ = cx;
        self.fun.push_insn(self.id, insn)
    }

    /// Push a fresh block into this body's arena and onto its roster. Mirrors
    /// [`HostMut::push_block`] with `func = self.id()`.
    pub fn push_block(&mut self, cx: ContextView<'_, 'str>, block: BasicBlock<'str>) -> BlockId {
        let _ = cx;
        self.fun.push_block(self.id, block)
    }

    /// Mint a fresh empty block, parented to this body and rostered. Mirrors
    /// [`HostMut::make_block`] with `func = self.id()`.
    pub fn make_block(&mut self, cx: ContextView<'_, 'str>) -> BlockId {
        let _ = cx;
        self.fun.make_block(self.id)
    }

    /// Push a fresh block parameter into this body's arena. Mirrors
    /// [`HostMut::push_block_param`] with `func = self.id()`.
    pub fn push_block_param(
        &mut self,
        cx: ContextView<'_, 'str>,
        param: BlockParam<'str>,
    ) -> BlockParamId {
        let _ = cx;
        self.fun.push_block_param(self.id, param)
    }

    /// Mint an `Int(size)`-typed instruction with `mnemonic`. Mirrors
    /// [`HostMut::push_mnemonic`] with `func = self.id()`.
    pub fn push_mnemonic(
        &mut self,
        cx: ContextView<'_, 'str>,
        mnemonic: Mnemonic,
        size: usize,
    ) -> InstructionId {
        self.fun
            .push_mnemonic(self.id, cx.shared_ctx(), mnemonic, size)
    }

    /// Mint an instruction with `mnemonic` and an explicit result `type_id`.
    /// Mirrors [`HostMut::push_mnemonic_with_type`] with `func = self.id()`.
    pub fn push_mnemonic_with_type(
        &mut self,
        cx: ContextView<'_, 'str>,
        mnemonic: Mnemonic,
        type_id: TypeId,
    ) -> InstructionId {
        let _ = cx;
        self.fun.push_mnemonic_with_type(self.id, mnemonic, type_id)
    }

    /// Insert `insn` immediately before `before` in `block`. Mirrors
    /// [`HostMut::insert_insn_before`].
    pub fn insert_insn_before(
        &mut self,
        cx: ContextView<'_, 'str>,
        block: BlockId,
        before: InstructionId,
        insn: InstructionId,
    ) {
        let _ = cx;
        self.fun.insert_insn_before(block, before, insn)
    }

    // ---- CFG / use-map verbs ------------------------------------------------

    /// Add a directed CFG edge `from -> to`. Mirrors [`HostMut::add_cfg_edge`].
    pub fn add_cfg_edge(
        &mut self,
        cx: ContextView<'_, 'str>,
        from: BlockId,
        to: BlockId,
    ) -> EdgeId {
        let _ = cx;
        self.fun.add_cfg_edge(from, to)
    }

    /// Remove CFG edge `edge_id` from this body. Mirrors
    /// [`HostMut::remove_cfg_edge`] with `func = self.id()`.
    pub fn remove_cfg_edge(&mut self, cx: ContextView<'_, 'str>, edge_id: EdgeId) {
        let _ = cx;
        self.fun.remove_cfg_edge(self.id, edge_id)
    }

    /// Replace every use of `old` with `new` across this body. Mirrors
    /// [`HostMut::replace_all_uses_with`].
    pub fn replace_all_uses_with(&mut self, cx: ContextView<'_, 'str>, old: ValueId, new: ValueId) {
        let _ = cx;
        self.fun.replace_all_uses_with(old, new)
    }

    /// Remove instruction `id` from this body (unlink edges, tombstone, prune
    /// uses). Mirrors [`HostMut::remove_instruction`].
    pub fn remove_instruction(&mut self, cx: ContextView<'_, 'str>, id: InstructionId) {
        let _ = cx;
        self.fun.remove_instruction(id)
    }

    /// Rehome `remove`'s outgoing edges onto `keep` and drop the direct edge.
    /// Mirrors [`HostMut::merge_nodes`].
    pub fn merge_nodes(
        &mut self,
        cx: ContextView<'_, 'str>,
        keep: BlockId,
        remove: BlockId,
        direct_edge: EdgeId,
    ) {
        let _ = cx;
        self.fun.merge_nodes(keep, remove, direct_edge)
    }

    /// Replace an instruction's mnemonic in place, keeping use/call-site maps in
    /// sync. Mirrors [`HostMut::replace_instruction_mnemonic`].
    pub fn replace_instruction_mnemonic(
        &mut self,
        cx: ContextView<'_, 'str>,
        id: InstructionId,
        mnemonic: Mnemonic,
    ) {
        let _ = cx;
        self.fun.replace_instruction_mnemonic(id, mnemonic)
    }

    /// Drop `block` from its owner's roster. Mirrors [`HostMut::unroster_block`].
    pub fn unroster_block(&mut self, cx: ContextView<'_, 'str>, block: BlockId) {
        let _ = cx;
        self.fun.unroster_block(block)
    }

    /// Remove `block` from this body (unlink edges, remove insns, detach params,
    /// tombstone). Mirrors [`HostMut::delete_block`] with `function_id = self.id()`.
    pub fn delete_block(&mut self, cx: ContextView<'_, 'str>, block: BlockId) {
        let _ = cx;
        self.fun.delete_block(block)
    }

    /// Absorb `other` into `keep` across the direct edge `edge_ab`. Mirrors
    /// [`HostMut::absorb_block`] with `function_id = self.id()`.
    pub fn absorb_block(
        &mut self,
        cx: ContextView<'_, 'str>,
        keep: BlockId,
        other: BlockId,
        edge_ab: EdgeId,
    ) {
        let _ = cx;
        self.fun.absorb_block(keep, other, edge_ab)
    }

    /// Register `name` for `id` in the owning table (function-local for
    /// block/insn/param, else global). Mirrors [`HostMut::register_local_name`].
    pub fn register_local_name(
        &mut self,
        cx: ContextView<'_, 'str>,
        id: ValueId,
        name: std::borrow::Cow<'str, str>,
        old_name: Option<&str>,
    ) -> Result<()> {
        self.fun
            .register_local_name(cx.shared_ctx(), id, name, old_name)
    }

    // ---- mutable arena accessors --------------------------------------------
    //
    // These return `&mut` borrows *into this body*, so they cannot be routed
    // through a freshly built `CheckedOut` (the temporary host would be dropped
    // before the borrow is returned). They delegate straight to the underlying
    // `Function` arena accessors — behaviour-identical to the `Context` versions,
    // which resolve to the same `self.fun.<arena>[id.local]` — and take no `cx`.

    /// The instruction `id`, mutably. Mirror of [`HostMut::instruction_mut`].
    pub fn instruction_mut(&mut self, id: InstructionId) -> &mut Instruction<'str> {
        self.fun.insn_mut(id)
    }

    /// The block `id`, mutably. Mirror of [`HostMut::block_mut`].
    pub fn block_mut(&mut self, id: BlockId) -> &mut BasicBlock<'str> {
        self.fun.block_mut(id)
    }

    /// The block parameter `id`, mutably. Mirror of [`HostMut::block_param_mut`].
    pub fn block_param_mut(&mut self, id: BlockParamId) -> &mut BlockParam<'str> {
        self.fun.block_param_mut(id)
    }

    // ---- read accessors -----------------------------------------------------

    /// The block `id`, routed to this body's arena. Mirror of [`HostRef::block`].
    pub fn block<'a>(&'a self, cx: ContextView<'a, 'str>, id: BlockId) -> &'a BasicBlock<'str> {
        self.read_host(cx).block(id)
    }

    /// The instruction `id`, routed to this body's arena. Mirror of
    /// [`HostRef::instruction`].
    pub fn insn<'a>(
        &'a self,
        cx: ContextView<'a, 'str>,
        id: InstructionId,
    ) -> &'a Instruction<'str> {
        self.read_host(cx).instruction(id)
    }

    /// The block parameter `id`, routed to this body's arena. Mirror of
    /// [`HostRef::block_param`].
    pub fn block_param<'a>(
        &'a self,
        cx: ContextView<'a, 'str>,
        id: BlockParamId,
    ) -> &'a BlockParam<'str> {
        self.read_host(cx).block_param(id)
    }

    // NB: the `edge` read accessor (`HostRef::edge`, returning `&EdgeData`) is
    // intentionally NOT mirrored — `EdgeData` is `pub(crate)` in core, so it
    // cannot be named from this crate (see the `push_edge` note above). No pass
    // reads a raw `&EdgeData`; edge endpoints are reached through the wrapper-ref
    // surface (`BlockRef::successors`, …).

    /// This body's instructions that use `value` as an operand. Mirror of
    /// [`Function::users_of`]; body-local, so it needs no `cx`.
    pub fn users_of(&self, value: ValueId) -> &[InstructionId] {
        self.fun.users_of(value)
    }

    // ---- wrapper-ref constructors -------------------------------------------

    /// A [`BlockRef`] over `id`, routed to this body. Mirror of
    /// [`HostRef::block_ref`].
    pub fn block_ref<'a>(&'a self, cx: ContextView<'a, 'str>, id: BlockId) -> BlockRef<'str, 'a> {
        self.read_host(cx).block_ref(id)
    }

    /// An [`InstructionRef`] over `id`, routed to this body. Mirror of
    /// [`HostRef::insn_ref`].
    pub fn insn_ref<'a>(
        &'a self,
        cx: ContextView<'a, 'str>,
        id: InstructionId,
    ) -> InstructionRef<'str, 'a> {
        self.read_host(cx).insn_ref(id)
    }

    /// A [`BlockParamRef`] over `id`, routed to this body. Mirror of
    /// [`HostRef::param_ref`].
    pub fn param_ref<'a>(
        &'a self,
        cx: ContextView<'a, 'str>,
        id: BlockParamId,
    ) -> BlockParamRef<'str, 'a> {
        self.read_host(cx).param_ref(id)
    }

    /// A [`FunctionRef`] over `f` (interface-routed for a foreign function).
    /// Mirror of [`HostRef::function_ref`].
    pub fn function_ref<'a>(
        &'a self,
        cx: ContextView<'a, 'str>,
        f: FunctionId,
    ) -> FunctionRef<'str, 'a> {
        self.read_host(cx).function_ref(f)
    }

    /// A [`FunctionRef`] over this body's *own* function. The self-directed twin
    /// of [`function_ref`](Self::function_ref).
    pub fn self_ref<'a>(&'a self, cx: ContextView<'a, 'str>) -> FunctionRef<'str, 'a> {
        self.read_host(cx).function_ref(self.id)
    }
}
