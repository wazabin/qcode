//! The types that give a function pass its restricted, parallel-safe view of the
//! world (Stage 5 of the parallel-function-passes plan; see `PARALLEL_PASSES.md`).
//!
//! A function pass reads the module's *published interface* through a shared
//! [`ContextView`] and mutates *only its own function* through a `&mut`
//! [`FunctionBody`] — the body borrowed `&mut` in place from the bodies registry
//! by the driver's [`split`](ContextSplit::split). The one effect on global state
//! a pass legitimately needs (a self-rename) is **returned** in [`Outcome::rename`]
//! and applied by the driver at the post-run barrier, so the pass itself touches
//! no global mutable state — which is what lets workers run in parallel with the
//! `ContextView` `&`-shared and the bodies disjoint `&mut`.
//!
//! The driver `split`s the context once, borrows every worklist body disjointly
//! (worklist order) via [`select_mut`](jstd::registry::Registry::select_mut), runs
//! each function's pass fixpoint — sequentially or on `std::thread::scope` workers
//! — then drops the split borrow and does the barrier work in worklist order. The
//! types are identical either way.

use std::borrow::Cow;

use jstd::registry::Registry;
use qcode::{
    context::{Context, Shared},
    error::Result,
    types::TypeId,
    value::{
        BlockParamRef, BlockRef, Function, FunctionId, FunctionKind, FunctionRef, InstructionRef,
        ValueId,
        block::{BasicBlock, BlockId, EdgeId},
        block_param::{BlockParam, BlockParamId},
        function::FunctionInterface,
        insn::{Instruction, InstructionId, Mnemonic},
        util::{base_ref::HostRef, host_mut::PassBacking},
    },
};

use super::PipelineEnv;

/// A function minted by a pass this run: its reserved id, its interface, and its
/// body. Installed into the reserved slot by the driver at the barrier.
pub type Minted<'str> = (FunctionId, FunctionInterface<'str>, Function<'str>);

/// The result of one function-pass run (context-split ruling 3): whether it
/// changed the IR, an optional self-rename claim, and any functions it minted.
///
/// Returned **by value** from [`FunctionPass::run`](super::FunctionPass::run), so
/// a pass touches no wrapper scratch: `rename` subsumes the old `Effects` buffer
/// and `minted` subsumes the old `FunctionBody.minted` field. The driver replays
/// `rename` and installs `minted` at the post-run barrier in worklist order,
/// exactly as it drained the wrapper before — the transport changes, the barrier
/// semantics do not.
#[derive(Default)]
pub struct Outcome<'str> {
    /// Whether the pass changed the function's IR (the old `Ok(bool)`).
    pub changed: bool,
    /// A buffered self-rename claim (from `cpp_demangle` / `name_thunks`), applied
    /// as a `get_unique_name` claim at the barrier. Last writer wins across a
    /// function's pass fixpoint, so replay is deterministic.
    pub rename: Option<Cow<'str, str>>,
    /// Functions this run minted (the loop outliners), installed by the driver at
    /// the barrier. Concatenated across a function's pass fixpoint.
    pub minted: Vec<Minted<'str>>,
}

impl<'str> Outcome<'str> {
    /// An unchanged outcome — no rename, no minted functions.
    pub fn unchanged() -> Self {
        Self::default()
    }

    /// An outcome reporting `changed`, with no rename or minted functions (the
    /// common case for the mechanical `Ok(bool)` → `Ok(Outcome::changed(bool))`
    /// sweep).
    pub fn changed(changed: bool) -> Self {
        Self {
            changed,
            ..Self::default()
        }
    }

    /// A changed outcome carrying a self-rename claim (the driver uniquifies and
    /// applies it at the barrier).
    pub fn renamed(name: Cow<'str, str>) -> Self {
        Self {
            changed: true,
            rename: Some(name),
            minted: Vec::new(),
        }
    }
}

impl<'str> From<bool> for Outcome<'str> {
    fn from(changed: bool) -> Self {
        Self::changed(changed)
    }
}

/// The read-only module interface a function pass may consult: architecture /
/// ABI, the interners (mintable through `&self`), and every *other* function's
/// published interface (name, address, signature, purity, clobber/write
/// summaries) — but **not** other functions' in-flight bodies (except the
/// pure-bodies view, which is stage-invariant; added with the gvn port).
///
/// The **bodies-free** module view of the context-split design (stage 5b-ii):
/// `{shared, interfaces, env}`. A function pass structurally cannot read another
/// function's body through it — holding a `ContextView` is the proof that
/// regimes 1–3 are frozen while workers hold disjoint `&mut` bodies. It is
/// `Copy` (it holds only shared references), so a worker hands the same view to
/// every helper and sub-ref.
#[derive(Clone, Copy)]
pub struct ContextView<'ctx, 'str> {
    shared: &'ctx Shared<'str>,
    interfaces: &'ctx Registry<FunctionId, FunctionInterface<'str>>,
    env: &'ctx PipelineEnv,
}

impl<'ctx, 'str> ContextView<'ctx, 'str> {
    /// Build a view over `ctx`'s shared state + interfaces and the pipeline
    /// environment. Takes the whole `&Context` for caller convenience (the
    /// driver) and narrows; [`Context`]'s bodies are NOT captured —
    /// prefer [`split`](ContextSplit::split), which proves that with a
    /// simultaneous `&mut` bodies borrow.
    pub fn new(ctx: &'ctx Context<'str>, env: &'ctx PipelineEnv) -> Self {
        Self {
            shared: &ctx.shared,
            interfaces: &ctx.interfaces,
            env,
        }
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
        &self.interfaces[f]
    }

    /// The module's shared IR state (interners, spaces, registers, name/address
    /// maps, memory image, truths).
    pub fn shr(&self) -> &'ctx Shared<'str> {
        self.shared
    }

    /// The whole interface registry (for building a [`PassBacking`]/[`HostRef`]).
    pub fn interfaces(&self) -> &'ctx Registry<FunctionId, FunctionInterface<'str>> {
        self.interfaces
    }
}

/// The driver-side disjoint borrow of the context-split design (00-overview):
/// `&mut bodies` alongside a frozen, bodies-free [`ContextView`]. Rust's
/// disjoint-field borrows prove the safety — no `unsafe`, no relocation.
pub trait ContextSplit<'str> {
    /// Split into the mutable bodies registry and the read-only module view.
    /// `env` is threaded as an argument so `Context` stays env-free.
    fn split<'a>(
        &'a mut self,
        env: &'a PipelineEnv,
    ) -> (
        &'a mut Registry<FunctionId, Function<'str>>,
        ContextView<'a, 'str>,
    );
}

impl<'str> ContextSplit<'str> for Context<'str> {
    fn split<'a>(
        &'a mut self,
        env: &'a PipelineEnv,
    ) -> (
        &'a mut Registry<FunctionId, Function<'str>>,
        ContextView<'a, 'str>,
    ) {
        (
            &mut self.bodies,
            ContextView {
                shared: &self.shared,
                interfaces: &self.interfaces,
                env,
            },
        )
    }
}

/// The pass's own function, borrowed `&mut` in place from the bodies registry so
/// the pass owns it exclusively, plus the function-minting pool.
///
/// The body is borrowed by the driver's [`split`](ContextSplit::split) and never
/// leaves the registry; the exclusive `&mut` is what lets parallel workers hold
/// disjoint `&mut FunctionBody`s over the same frozen [`ContextView`]. The one
/// global effect a pass legitimately requests — a self-rename — is returned in
/// [`Outcome::rename`] and applied by the driver at the barrier.
pub struct FunctionBody<'a, 'str> {
    /// This function's id (the registry key; a [`Function`] does not store its
    /// own id).
    id: FunctionId,
    /// The function being optimized, borrowed **in place** from the bodies
    /// registry (its arenas, roster, root, users, names). The body never leaves
    /// the registry — the `&mut` is what gives the pass exclusive access while the
    /// frozen [`ContextView`] shares the rest of the module.
    fun: &'a mut Function<'str>,
    /// Never-observed placeholder [`FunctionId`]s the pass may materialize new
    /// functions into (loop outliners mint exactly one). Unused ids return to the
    /// driver's pool at the barrier.
    reserved_ids: Vec<FunctionId>,
    /// Functions built this run against drawn `reserved_ids` (paired with the id
    /// each was drawn for, and its interface), installed by the driver at
    /// the barrier.
    minted: Vec<Minted<'str>>,
}

impl<'a, 'str> FunctionBody<'a, 'str> {
    /// Wrap the function `fun` (id `id`) borrowed in place from the bodies
    /// registry, carrying `reserved_ids` for any function it mints.
    pub fn new(id: FunctionId, fun: &'a mut Function<'str>, reserved_ids: Vec<FunctionId>) -> Self {
        Self {
            id,
            fun,
            reserved_ids,
            minted: Vec::new(),
        }
    }

    /// This function's id.
    pub fn id(&self) -> FunctionId {
        self.id
    }

    /// The borrowed function (read).
    pub fn function(&self) -> &Function<'str> {
        &*self.fun
    }

    /// A [`PassBacking`] mutation host over this body's borrowed function and the
    /// module's read-only shared context. This is how a `FunctionPass` reads
    /// (via [`PassBacking::read_host`]) and mutates its function — construct
    /// block/instruction refs and `Builder`s over it.
    pub fn host<'b>(&'b mut self, cx: ContextView<'b, 'str>) -> PassBacking<'b, 'str> {
        PassBacking::new(&mut *self.fun, self.id, cx.shr(), cx.interfaces())
    }

    /// A `Copy` read view over this body's borrowed function and the shared
    /// context — the recognizer-side twin of [`host`](Self::host) for passes that
    /// only need to *read* while holding other borrows.
    pub fn read_host<'b>(&'b self, cx: ContextView<'b, 'str>) -> HostRef<'b, 'str> {
        HostRef::Checked {
            fun: &*self.fun,
            shared: cx.shr(),
            interfaces: cx.interfaces(),
            id: self.id,
        }
    }

    /// Mint a new function (`PARALLEL_PASSES.md` ruling 3): draw one reserved id
    /// from the pool the driver assigned this run, create a detached
    /// [`Function`] shell under `name` (buffered **raw** — global uniquification
    /// happens when the driver installs it at the barrier) with the given `kind`,
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
    /// [`PassBacking`] mutation host over the minted function `minted` (a
    /// [`mint_function`](Self::mint_function) result). This is how an outliner
    /// builds a minted body: it clones expression slices out of its own function
    /// (read) into the minted one (write), both against the same shared context.
    ///
    /// Panics if `minted` was not minted by this body.
    pub fn host_with_minted<'b>(
        &'b mut self,
        cx: ContextView<'b, 'str>,
        minted: FunctionId,
    ) -> (HostRef<'b, 'str>, PassBacking<'b, 'str>) {
        let own = HostRef::Checked {
            fun: &*self.fun,
            shared: cx.shr(),
            interfaces: cx.interfaces(),
            id: self.id,
        };
        let fun = self
            .minted
            .iter_mut()
            .find(|(id, _, _)| *id == minted)
            .map(|(_, _, f)| f)
            .expect("host_with_minted: not a function minted by this body");
        (
            own,
            PassBacking::new(fun, minted, cx.shr(), cx.interfaces()),
        )
    }

    /// Consume the body at the barrier, yielding the functions it minted and any
    /// unused reserved ids (returned to the driver's pool). The self-rename claim
    /// travels in [`Outcome::rename`], not here. The function itself stays borrowed
    /// in place in the registry — there is no body to reinstall.
    pub fn into_parts(self) -> (Vec<Minted<'str>>, Vec<FunctionId>) {
        (self.minted, self.reserved_ids)
    }
}

/// Inherent verb + read-accessor surface (context-split stage 5b-ii).
///
/// Every mutation a function pass makes and every read accessor it needs is an
/// inherent method on the body itself: `body.verb(cx, …)`. The mutation verbs
/// delegate to the owning [`Function`]'s inherent verbs (supplying the ambient
/// [`id`](Self::id)); the read accessors route through a [`HostRef`] built from
/// `self.fun` + `cx`.
///
/// Where a [`Function`] verb takes an explicit `func: FunctionId` for the pass's own
/// function, the inherent method drops that parameter and supplies
/// [`self.id()`](Self::id) instead — a function pass only ever mints/mutates into
/// its own body.
impl<'body, 'str> FunctionBody<'body, 'str> {
    // ---- births -------------------------------------------------------------
    //
    // `push_edge` (`Context::push_edge`) is intentionally NOT mirrored: its
    // `EdgeData` parameter is `pub(crate)` in `qcode::value::block`, so it cannot
    // be named from this crate without making `EdgeData` public (a core design
    // change, out of this commit's additive scope). No pass calls `push_edge`
    // directly — edges are created through `add_cfg_edge` — so nothing needs it.

    /// Push a fresh instruction into this body's arena (recording operand uses and
    /// the call-site cache). Mirrors [`Function::push_insn`] with `func = self.id()`.
    pub fn push_insn(
        &mut self,
        cx: ContextView<'_, 'str>,
        insn: Instruction<'str>,
    ) -> InstructionId {
        let _ = cx;
        self.fun.push_insn(self.id, insn)
    }

    /// Push a fresh block into this body's arena and onto its roster. Mirrors
    /// [`Function::push_block`] with `func = self.id()`.
    pub fn push_block(&mut self, cx: ContextView<'_, 'str>, block: BasicBlock<'str>) -> BlockId {
        let _ = cx;
        self.fun.push_block(self.id, block)
    }

    /// Mint a fresh empty block, parented to this body and rostered. Mirrors
    /// [`Function::make_block`] with `func = self.id()`.
    pub fn make_block(&mut self, cx: ContextView<'_, 'str>) -> BlockId {
        let _ = cx;
        self.fun.make_block(self.id)
    }

    /// Push a fresh block parameter into this body's arena. Mirrors
    /// [`Function::push_block_param`] with `func = self.id()`.
    pub fn push_block_param(
        &mut self,
        cx: ContextView<'_, 'str>,
        param: BlockParam<'str>,
    ) -> BlockParamId {
        let _ = cx;
        self.fun.push_block_param(self.id, param)
    }

    /// Mint an `Int(size)`-typed instruction with `mnemonic`. Mirrors
    /// [`Function::push_mnemonic`] with `func = self.id()`.
    pub fn push_mnemonic(
        &mut self,
        cx: ContextView<'_, 'str>,
        mnemonic: Mnemonic,
        size: usize,
    ) -> InstructionId {
        self.fun.push_mnemonic(self.id, cx.shr(), mnemonic, size)
    }

    /// Mint an instruction with `mnemonic` and an explicit result `type_id`.
    /// Mirrors [`Function::push_mnemonic_with_type`] with `func = self.id()`.
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
    /// [`Function::insert_insn_before`].
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

    /// Add a directed CFG edge `from -> to`. Mirrors [`Function::add_cfg_edge`].
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
    /// [`Function::remove_cfg_edge`] with `func = self.id()`.
    pub fn remove_cfg_edge(&mut self, cx: ContextView<'_, 'str>, edge_id: EdgeId) {
        let _ = cx;
        self.fun.remove_cfg_edge(self.id, edge_id)
    }

    /// Replace every use of `old` with `new` across this body. Mirrors
    /// [`Function::replace_all_uses_with`].
    pub fn replace_all_uses_with(&mut self, cx: ContextView<'_, 'str>, old: ValueId, new: ValueId) {
        let _ = cx;
        self.fun.replace_all_uses_with(old, new)
    }

    /// Remove instruction `id` from this body (unlink edges, tombstone, prune
    /// uses). Mirrors [`Function::remove_instruction`].
    pub fn remove_instruction(&mut self, cx: ContextView<'_, 'str>, id: InstructionId) {
        let _ = cx;
        self.fun.remove_instruction(id)
    }

    /// Rehome `remove`'s outgoing edges onto `keep` and drop the direct edge.
    /// Mirrors [`Function::merge_nodes`].
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
    /// sync. Mirrors [`Function::replace_instruction_mnemonic`].
    pub fn replace_instruction_mnemonic(
        &mut self,
        cx: ContextView<'_, 'str>,
        id: InstructionId,
        mnemonic: Mnemonic,
    ) {
        let _ = cx;
        self.fun.replace_instruction_mnemonic(id, mnemonic)
    }

    /// Drop `block` from its owner's roster. Mirrors [`Function::unroster_block`].
    pub fn unroster_block(&mut self, cx: ContextView<'_, 'str>, block: BlockId) {
        let _ = cx;
        self.fun.unroster_block(block)
    }

    /// Remove `block` from this body (unlink edges, remove insns, detach params,
    /// tombstone). Mirrors [`Function::delete_block`] with `function_id = self.id()`.
    pub fn delete_block(&mut self, cx: ContextView<'_, 'str>, block: BlockId) {
        let _ = cx;
        self.fun.delete_block(block)
    }

    /// Absorb `other` into `keep` across the direct edge `edge_ab`. Mirrors
    /// [`Function::absorb_block`] with `function_id = self.id()`.
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
    /// block/insn/param, else global). Mirrors [`Function::register_local_name`].
    pub fn register_local_name(
        &mut self,
        cx: ContextView<'_, 'str>,
        id: ValueId,
        name: std::borrow::Cow<'str, str>,
        old_name: Option<&str>,
    ) -> Result<()> {
        self.fun.register_local_name(cx.shr(), id, name, old_name)
    }

    // ---- mutable arena accessors --------------------------------------------
    //
    // These return `&mut` borrows *into this body*, so they cannot be routed
    // through a freshly built `PassBacking` (the temporary host would be dropped
    // before the borrow is returned). They delegate straight to the underlying
    // `Function` arena accessors — behaviour-identical to the `Context` versions,
    // which resolve to the same `self.fun.<arena>[id.local]` — and take no `cx`.

    /// The instruction `id`, mutably. Mirror of [`Function::insn_mut`].
    pub fn instruction_mut(&mut self, id: InstructionId) -> &mut Instruction<'str> {
        self.fun.insn_mut(id)
    }

    /// The block `id`, mutably. Mirror of [`Function::block_mut`].
    pub fn block_mut(&mut self, id: BlockId) -> &mut BasicBlock<'str> {
        self.fun.block_mut(id)
    }

    /// The block parameter `id`, mutably. Mirror of [`Function::block_param_mut`].
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::dummy_env;
    use qcode::value::insn::Mnemonic;
    use qcode_macro::qcode;

    /// The split's borrow story: hold `&mut bodies[fid]` (and mutate through the
    /// inherent `Function` verbs) while simultaneously reading the module through
    /// the bodies-free `ContextView` — interners, interfaces, env. Rust's
    /// disjoint-field borrows prove no aliasing; no `unsafe` anywhere.
    #[test]
    fn split_borrows_bodies_and_view_disjointly() {
        let mut ctx = Context::new();
        qcode!(ctx, "fn callee: <c_entry> return at 0x2000;");
        qcode!(ctx, "fn f: <entry> return at 0x1000;");
        let env = dummy_env();

        let insn_count_before;
        {
            let (bodies, view) = ctx.split(&env);
            // Disjoint &mut borrows of two distinct bodies, worklist order.
            let mut slots = bodies.select_mut(&[f, callee]);
            let (own, rest) = slots.split_first_mut().unwrap();

            // View reads while the body borrows are live.
            assert_eq!(view.interface(callee).name.as_ref(), "callee");
            let k = view.shr().get_const(42, 8);

            // Mutate the own body through its inherent verbs, consuming the
            // view-minted literal — the exact pass-shaped usage.
            let root = own.root.expect("root");
            insn_count_before = own.block(root).instructions.len();
            let insn = own.push_mnemonic(
                f,
                view.shr(),
                Mnemonic::Zext(qcode::value::insn::Zext { src: k, size: 8 }),
                8,
            );
            let first = own.block(root).instructions[0];
            own.insert_insn_before(root, first, insn);
            let _ = rest;
        }

        // The split borrow has ended; the whole context is usable again.
        let root = ctx.bodies[f].root.expect("root");
        assert_eq!(
            ctx.bodies[f].block(root).instructions.len(),
            insn_count_before + 1
        );
        assert!(ctx.shared.values.literals.iter().any(|l| l.value == 42));
    }
}
