//! The pass traits, the per-function adapter boundary, and the name registry.
//!
//! Every pass that can appear in a pipeline TOML implements one of two traits:
//!
//! - [`FunctionPass`] — operates on a single function through the restricted,
//!   parallel-safe [`ContextView`]/[`FunctionBody`] surface. Most optimizations
//!   (mem2reg, gvn, dce, …) are these. Function-scoped TOML stages run them
//!   function-major and can loop them to a per-function fixpoint.
//! - [`Pass`] — operates on the whole program (`Context`). The interprocedural
//!   "milestone" steps (binding, summaries, external signatures, clobber seeding)
//!   are these.
//!
//! Both pull their architecture-specific inputs from [`PipelineEnv`] at run time,
//! so the registry can construct every pass with no arguments.
//!
//! ## Adding a pass
//!
//! A pass lives entirely in its own module file: define the struct, implement
//! [`FunctionPass`] (or [`Pass`]) for it, and register it with one
//! [`inventory::submit!`] of a [`PassRegistration`]. Nothing in this file needs to
//! change. See [`crate::naming::cpp_demangle::CppDemangle`] for the smallest
//! possible example pass.
//!
//! [`FunctionPass`] requires [`Default`], so a pass can precompute and store
//! expensive values once at construction (see the `CppDemangle` example, which
//! builds its `DemangleOptions` in `Default`). Because a `Default` supertrait makes
//! a trait non-object-safe, the registry stores passes behind the object-safe
//! [`DynFunctionPass`] shim via the [`FunctionPassAdapter`] wrapper.

use std::sync::OnceLock;

use qcode::{
    context::Context,
    space::Space,
    value::{FunctionBody, FunctionId, RegisterId, Renameable, VarnodeId},
};
use rustc_hash::FxHashSet;

use super::{
    AnalysisManager, ArchConfig, CallingConvention, ContextSplit, ContextView,
    LocalAnalysisManager, Outcome, PreservedAnalyses,
};
use crate::structure::Program;
use crate::RegisterBase;

#[cfg(test)]
fn call_graph_snapshot(ctx: &Context<'_>) -> Vec<crate::CallEdge> {
    crate::CallGraph::analyze(ctx)
        .edges()
        .map(|(_, edge)| *edge)
        .collect()
}

#[cfg(test)]
fn address_snapshot(ctx: &Context<'_>) -> qcode::address_index::AddressIndex {
    qcode::address_index::AddressIndex::analyze(ctx)
}

/// The architecture-specific inputs the register-aware passes need, resolved once
/// per pipeline run and shared by reference with every pass.
pub struct PipelineEnv {
    /// Register layout / calling convention, built by `harbinger::arch::arch_config`.
    pub cfg: ArchConfig,
    /// The stack-pointer *varnode* (`cfg.stack_pointer` resolved through
    /// `ctx.shared.registers`), cached so passes don't re-resolve it each call.
    pub sp_varnode: Option<VarnodeId>,
    /// The loaded binary, shared with the loader when this is a live lift.
    /// Passes read initialized memory (jump-table slots, `.rodata` constants)
    /// through it. `None` for headless/textual runs; reloaded snapshots wrap
    /// the deserialized `MemoryImage` instead. `Arc<dyn ...>` (the trait
    /// carries `Send + Sync` bounds) so `&PipelineEnv` stays `Sync` for the
    /// parallel driver.
    pub binary: Option<std::sync::Arc<dyn binfmt::BinaryFormat>>,
    /// Function-independent register/varnode alias base, built once on first use and
    /// shared by reference across the per-function GVN/LICM/DCE/mem2reg runs (see
    /// [`PipelineEnv::alias_base`]). A `OnceLock` (not `RefCell`) so `&PipelineEnv`
    /// is `Sync` and can be shared across worker threads (Stage 6); the base is
    /// immutable once built and never rebuilt.
    alias_base: OnceLock<RegisterBase>,
}

impl PipelineEnv {
    /// Resolve the stack-pointer varnode from `cfg` against `ctx` once. No lifter:
    /// the lifting passes will be inert.
    pub fn new(
        ctx: &Context,
        cfg: ArchConfig,
        binary: Option<std::sync::Arc<dyn binfmt::BinaryFormat>>,
    ) -> Self {
        let sp_varnode = ctx.shared.registers[&cfg.stack_pointer];
        let mut env = Self::from_parts(cfg, sp_varnode);
        env.binary = binary;
        env
    }

    /// Build an env for running arch-agnostic passes on hand-written IR (CLI and
    /// other tools), where there is no machine architecture to resolve. The
    /// stack pointer is absent and the ABI is empty, so architecture-dependent
    /// passes are inert while arch-agnostic transforms still run normally.
    pub fn headless(ctx: &Context) -> Self {
        let bitness = (Space::from_id(ctx, ctx.shared.default_space).addr_size * 8) as u8;
        let cfg = ArchConfig {
            // Unused by arch-agnostic passes; `sp_varnode` is explicitly absent.
            stack_pointer: RegisterId::from(0usize),
            dead_flag_regs: Vec::new(),
            abi: CallingConvention::default(),
            os: ctx.target_os(),
            bitness,
        };
        Self::from_parts(cfg, None)
    }

    /// Build an env from already-resolved parts, without consulting a `ctx`. Used by
    /// unit tests that construct a throwaway env for arch-agnostic passes; prefer
    /// [`PipelineEnv::new`] in production.
    pub(crate) fn from_parts(cfg: ArchConfig, sp_varnode: impl Into<Option<VarnodeId>>) -> Self {
        Self {
            cfg,
            sp_varnode: sp_varnode.into(),
            binary: None,
            alias_base: OnceLock::new(),
        }
    }

    /// The shared register/varnode alias base ("Part A" of the simple alias
    /// analysis) for `ctx`. Built once on first use and returned by reference
    /// thereafter; the per-function GVN/LICM/DCE/mem2reg passes finish it with
    /// [`RegisterBase::for_function`], so the O(varnodes·log) base build runs once
    /// per pipeline run instead of once per function.
    ///
    /// The base is never rebuilt. The register/global varnode set is established at
    /// lift time and does not grow during the optimization passes that consult the
    /// oracle, so the base built on first use stays valid for the whole run. (The
    /// old code rebuilt it whenever `ctx.varnode_count()` changed; instrumentation
    /// showed that rebuild never fired across the entire test suite.) Body-local
    /// temporaries do not change the shared varnode set; the `debug_assert` below
    /// catches an unexpected module-varnode mint loudly rather than silently.
    pub fn alias_base(&self, shared: &qcode::context::Shared) -> &RegisterBase {
        let base = self.alias_base.get_or_init(|| RegisterBase::build(shared));
        debug_assert_eq!(
            base.varnode_count(),
            shared.varnode_count(),
            "register alias base built from a stale varnode set; a pass minted varnodes mid-run"
        );
        base
    }
}

/// Object-safe dispatch shim for [`FunctionPass`].
///
/// [`FunctionPass`] can't be made into a trait object — its [`Default`] supertrait
/// and `NAME` associated const are both non-object-safe. This shim mirrors it as
/// instance methods and is implemented by [`FunctionPassAdapter`], so the registry can store
/// `Box<dyn DynFunctionPass>`.
pub trait DynFunctionPass: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;

    /// Run the pass over one function of `ctx`: split the context, run the pass on
    /// the body borrowed in place from the bodies registry, then run the barrier
    /// (replay buffered effects, install any minted functions, resync call sites).
    /// Used by the `module(<fn>)` adapter and unit tests, which have no driver-owned
    /// worklist to run on.
    fn run(&self, ctx: &mut Context, fun_id: FunctionId, env: &PipelineEnv)
    -> Result<bool, String>;

    fn run_with_analyses(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        env: &PipelineEnv,
        analyses: &mut AnalysisManager,
    ) -> Result<bool, String>;

    /// Run the pass on a [`FunctionBody`] the driver has *already* borrowed from the
    /// bodies registry, so the driver owns the barrier. This is the surface the
    /// parallel driver (and the sequential fixpoint) use to run a pass on a body they
    /// hold `&mut`; the returned [`Outcome`]'s rename replay and the barrier are the
    /// driver's job, not this method's. `next_minted` is the owner's stage-local
    /// placeholder cursor and is shared across every pass/fixpoint iteration.
    fn run_checked<'str>(
        &self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        next_minted: &mut u32,
        analyses: &mut LocalAnalysisManager,
    ) -> Result<Outcome<'str>, String>;
}

/// The parallel-safe function pass trait (Stage 5 of the
/// parallel-function-passes plan; see `PARALLEL_PASSES.md`).
///
/// The litmus test the signature enforces: **a function pass may read the
/// module's published interface and mutate its own function — nothing else.** It
/// reads the module through a `&`-shared [`ContextView`] and mutates only its own
/// [`FunctionBody`], returning global effects for the driver to replay at the
/// barrier. Missing types are requested before body mutation and cause a retry
/// after publication. With no path to global mutable state, workers can run these in
/// parallel (Stage 6) with the `ContextView` shared and the bodies disjoint.
///
/// The [`FunctionPassAdapter`] lets a `FunctionPass` be stored and driven through the
/// object-safe [`DynFunctionPass`] the registry speaks (it performs the split →
/// run → barrier dance internally), so a `module(<fn>)` stage and unit tests can
/// run one straight over a `&mut Context`.
pub trait FunctionPass: Default {
    const NAME: &'static str;
    fn description(&self) -> &'static str;
    fn run<'str>(
        &self,
        f: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        // Driver-owned placeholder cursor, reset once per owner at stage entry.
        next_minted: &mut u32,
    ) -> Result<Outcome<'str>, String>;

    /// Analysis-aware entry point. Existing passes use [`FunctionPass::run`];
    /// consumers of cached local analyses override this method.
    fn run_with_analyses<'str>(
        &self,
        f: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        next_minted: &mut u32,
        _analyses: &mut LocalAnalysisManager,
    ) -> Result<Outcome<'str>, String> {
        self.run(f, cx, next_minted)
    }
}

/// Adapts a [`FunctionPass`] to the object-safe [`DynFunctionPass`] the registry
/// and the sequential driver speak, encapsulating the whole run + barrier protocol
/// in one place:
///
/// 1. [`split`](ContextSplit::split) the context and borrow the target body `&mut`
///    in place alongside the bodies-free [`ContextView`].
/// 2. Run the pass on the borrowed [`FunctionBody`].
/// 3. Drop the split borrow, then install any minted callees, resync the owner's
///    call sites, and apply the returned [`Outcome`]'s self-rename.
///
/// The parallel driver instead borrows a whole worklist of bodies disjointly and
/// calls [`DynFunctionPass::run_checked`] directly on each; the adapter's `run` is
/// the whole-`Context` bridge for the `module(<fn>)` spelling and tests.
pub struct FunctionPassAdapter<T: FunctionPass> {
    inner: T,
}

impl<T: FunctionPass> Default for FunctionPassAdapter<T> {
    fn default() -> Self {
        Self {
            inner: T::default(),
        }
    }
}

impl<T: FunctionPass + Send + Sync> DynFunctionPass for FunctionPassAdapter<T> {
    fn name(&self) -> &'static str {
        T::NAME
    }
    fn description(&self) -> &'static str {
        FunctionPass::description(&self.inner)
    }
    fn run_checked<'str>(
        &self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        next_minted: &mut u32,
        analyses: &mut LocalAnalysisManager,
    ) -> Result<Outcome<'str>, String> {
        FunctionPass::run_with_analyses(&self.inner, body, cx, next_minted, analyses)
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        env: &PipelineEnv,
    ) -> Result<bool, String> {
        self.run_with_analyses(ctx, fun_id, env, &mut AnalysisManager::default())
    }
    fn run_with_analyses(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        env: &PipelineEnv,
        analyses: &mut AnalysisManager,
    ) -> Result<bool, String> {
        #[cfg(test)]
        let call_graph_before = call_graph_snapshot(ctx);
        #[cfg(test)]
        let addresses_before = address_snapshot(ctx);
        // Split the context: borrow the target body `&mut` in place and run the
        // pass over the frozen module view; the view is bodies-free, so it cannot
        // alias the borrowed body. A missing type is requested without mutation,
        // published at this bridge's barrier, then the pass is retried.
        let mut next_minted = 0;
        let mut type_request_retries = 0;
        let outcome = loop {
            let outcome = {
                let (bodies, view) = ctx.split(env);
                let mut local = analyses.take_local(fun_id);
                let outcome =
                    self.run_checked(&mut bodies[fun_id], view, &mut next_minted, &mut local)?;
                if outcome.changed || !outcome.minted.is_empty() {
                    local.invalidate(&outcome.preserved_analyses);
                }
                analyses.put_local(fun_id, local);
                outcome
            };
            if outcome.type_requests.is_empty() {
                break outcome;
            }
            if outcome.changed || outcome.rename.is_some() || !outcome.minted.is_empty() {
                return Err(format!(
                    "{}: a type-request outcome must not also mutate IR, rename, or mint functions",
                    T::NAME
                ));
            }
            ctx.shared
                .types
                .create_requested_types(&outcome.type_requests);
            type_request_retries += 1;
            if type_request_retries >= 64 {
                return Err(format!(
                    "{}: type requests did not settle after 64 retries",
                    T::NAME
                ));
            }
        };
        // Barrier, in the driver's order: install and resolve minted callees,
        // then apply the returned self-rename. The body was mutated in place.
        let preserved = outcome.preserved_analyses.clone();
        let installed = install_minted(ctx, T::NAME, outcome.minted)?;
        let patched = resolve_minted_callees(ctx, T::NAME, fun_id, &installed)?;
        replay_rename(ctx, T::NAME, fun_id, outcome.rename)?;
        let changed = outcome.changed || patched;
        if changed {
            analyses.invalidate_globals(&preserved);
        }
        #[cfg(test)]
        if preserved.preserves_global_analysis::<crate::CallGraphAnalysis>() {
            assert_eq!(
                call_graph_before,
                call_graph_snapshot(ctx),
                "{} reported preserving CallGraphAnalysis but changed its result",
                T::NAME,
            );
        }
        #[cfg(test)]
        if preserved.preserves_global_analysis::<crate::AddressAnalysis>() {
            assert_eq!(
                addresses_before,
                address_snapshot(ctx),
                "{} reported preserving AddressAnalysis but changed its result",
                T::NAME,
            );
        }
        Ok(changed)
    }
}

/// Run `f` over a `(&mut FunctionBody, ContextView)` for `fid`, borrowing the
/// body `&mut` in place via [`split`](ContextSplit::split).
///
/// This is the bridge the whole-`Context` optimization entry points
/// (`gvn_function`, `constant_fold_function`, `narrow_function`, `mem2reg`,
/// `mem2reg_framed`) use to reach the concrete function-pass core: their callers
/// hold a `&mut Context` but neither a [`FunctionBody`] nor a [`PipelineEnv`], and
/// they supply their own alias oracle, so the [`ContextView`]'s headless env is
/// not consulted by the cores. Because the concrete [`BodyMut`] path
/// debug-asserts the body is self-stored, every caller must feed a function with
/// no reattributed blocks — which, post the driver's `split_overlapping_functions`
/// normalization, every production function is.
///
/// [`BodyMut`]: qcode::value::util::body_mut::BodyMut
pub(crate) fn with_body_mut<'str, R>(
    ctx: &mut Context<'str>,
    fid: FunctionId,
    f: impl FnOnce(&mut FunctionBody<'str>, ContextView<'_, 'str>) -> R,
) -> R {
    let env = PipelineEnv::headless(ctx);
    {
        let (bodies, view) = ctx.split(&env);
        f(&mut bodies[fid], view)
    }
}

/// Append a pass's detached minted functions to the real function registries at
/// the barrier (master thread, worklist order): predict the next lockstep ID,
/// rebind the body's ambient ownership metadata to it, uniquify the buffered raw
/// name, append interface and body together, and register the name.
/// Returns the installed ids so the driver can mark them dirty for downstream
/// `only_dirty` stages.
pub(super) fn install_minted<'str>(
    ctx: &mut Context<'str>,
    pass: &str,
    minted: Vec<super::Minted<'str>>,
) -> Result<Vec<FunctionId>, String> {
    for (expected_slot, entry) in minted.iter().enumerate() {
        let slot = entry.slot();
        if slot as usize != expected_slot {
            return Err(format!(
                "{pass}: minted function slot #{slot} is out of order; expected #{expected_slot}"
            ));
        }
    }
    let mut installed = Vec::with_capacity(minted.len());
    for minted in minted {
        let id = FunctionId::from(ctx.bodies.len());
        let (_, mut interface, body) = minted.into_installed_parts(id);
        let name = std::mem::take(&mut interface.name);
        let unique = ctx.get_unique_name(name);
        interface.name = unique.clone();
        let appended = ctx.push_function(interface, body);
        debug_assert_eq!(appended, id, "function registry append returned wrong id");
        ctx.update_name(unique, id.into(), None)
            .map_err(|e| format!("{pass}: minted-function name registration failed: {e}"))?;
        installed.push(id);
    }
    Ok(installed)
}

/// Resolve pass-local callee placeholders in `owner` after its minted functions
/// have been installed at the barrier. `installed[k]` is the real function for
/// `Callee::Minted(k)`; both vectors are produced in deterministic mint order.
///
/// This must run after [`install_minted`]. Returns whether any owner or installed
/// body was patched.
pub(super) fn resolve_minted_callees(
    ctx: &mut Context<'_>,
    pass: &str,
    owner: FunctionId,
    installed: &[FunctionId],
) -> Result<bool, String> {
    let mut changed = false;
    for fun_id in std::iter::once(owner).chain(installed.iter().copied()) {
        // The walk covers the whole live instruction arena, not just
        // instructions linked into rostered blocks. A temporarily detached
        // instruction is still live pass state and must not carry an
        // unresolved placeholder past the barrier.
        match ctx.bodies[fun_id].resolve_minted_callees(installed) {
            Ok(patched) => changed |= patched != 0,
            Err(slot) => {
                return Err(format!(
                    "{pass}: function {fun_id:?} references minted callee #{slot}, but only {} were installed",
                    installed.len()
                ));
            }
        }
    }
    Ok(changed)
}

#[cfg(test)]
mod minted_barrier_tests {
    use super::*;
    use qcode::value::QCodeMut;
    use qcode::{
        testing::TestContext,
        value::{
            BasicBlock, InstructionRef,
            insn::{Call, Callee, InstructionId, Mnemonic},
        },
    };

    fn caller_with_minted_call(slot: u32) -> (TestContext, FunctionId, InstructionId, FunctionId) {
        let mut tc = TestContext::new();
        let caller = FunctionBody::make(&mut tc.ctx, "caller".into()).unwrap().id;
        let callee = FunctionBody::make(&mut tc.ctx, "callee".into()).unwrap().id;
        let block = tc.ctx.get_or_make_block(0x1000, caller);
        FunctionBody::from_id_mut(&mut tc.ctx, caller)
            .set_root(block)
            .unwrap();
        {
            let mut builder = tc.ctx.builder(block);
            builder.push_call(callee);
        }
        let call_id = BasicBlock::from_id(&tc.ctx, block)
            .iter()
            .find(|insn| matches!(insn.mnemonic(), Mnemonic::Call(_)))
            .unwrap()
            .id;
        tc.ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(Call {
                target: Callee::Minted(slot),
                args: Vec::new(),
                clobbers: Vec::new(),
            }),
        );
        (tc, caller, call_id, callee)
    }

    #[test]
    fn minted_callee_is_patched_before_resync() {
        let (mut tc, caller, call_id, callee) = caller_with_minted_call(0);

        assert!(resolve_minted_callees(&mut tc.ctx, "test", caller, &[callee]).unwrap());
        let Mnemonic::Call(call) = tc.ctx.get_insn(call_id).mnemonic() else {
            panic!("call disappeared");
        };
        assert_eq!(call.target, Callee::Real(callee));
    }

    #[test]
    fn minted_body_can_reference_a_sibling_slot() {
        let (mut tc, caller, _, first) = caller_with_minted_call(0);
        let sibling = FunctionBody::make(&mut tc.ctx, "sibling".into())
            .unwrap()
            .id;
        let block = tc.ctx.get_or_make_block(0x2000, first);
        FunctionBody::from_id_mut(&mut tc.ctx, first)
            .set_root(block)
            .unwrap();
        let sibling_call = {
            let mut builder = tc.ctx.builder(block);
            builder.push_call(sibling);
            drop(builder);
            BasicBlock::from_id(&tc.ctx, block)
                .iter()
                .find(|insn| matches!(insn.mnemonic(), Mnemonic::Call(_)))
                .unwrap()
                .id
        };
        tc.ctx.replace_instruction_mnemonic(
            sibling_call,
            Mnemonic::Call(Call {
                target: Callee::Minted(1),
                args: Vec::new(),
                clobbers: Vec::new(),
            }),
        );

        assert!(resolve_minted_callees(&mut tc.ctx, "test", caller, &[first, sibling]).unwrap());
        let Mnemonic::Call(call) = tc.ctx.get_insn(sibling_call).mnemonic() else {
            panic!("call disappeared");
        };
        assert_eq!(call.target, Callee::Real(sibling));
    }

    #[test]
    fn minted_callee_slot_must_have_an_installed_function() {
        let (mut tc, caller, _, callee) = caller_with_minted_call(1);

        let err = resolve_minted_callees(&mut tc.ctx, "test", caller, &[callee]).unwrap_err();
        assert!(err.contains("minted callee #1"), "{err}");
    }

    #[test]
    fn detached_instruction_cannot_hide_an_unresolved_minted_callee() {
        let (mut tc, caller, _, callee) = caller_with_minted_call(0);
        // Born directly into the owner's arena and deliberately never appended
        // to a block: roster/block iteration cannot see this live instruction.
        let detached = InstructionRef::from_mnemonic(
            &mut tc.ctx,
            caller,
            Mnemonic::Call(Call {
                target: Callee::Minted(1),
                args: Vec::new(),
                clobbers: Vec::new(),
            }),
            0,
        )
        .id;

        let err = resolve_minted_callees(&mut tc.ctx, "test", caller, &[callee]).unwrap_err();
        assert!(err.contains("minted callee #1"), "{err}");
        let Mnemonic::Call(call) = tc.ctx.get_insn(detached).mnemonic() else {
            panic!("detached call disappeared");
        };
        assert_eq!(call.target, Callee::Minted(1));
    }
}

/// Apply a function pass's returned self-rename claim ([`Outcome::rename`]) into
/// the context at the barrier. Runs on the master thread in worklist order; the
/// claim is a last-writer-wins request the pass already resolved for its run.
///
/// [`Outcome::rename`]: super::Outcome::rename
pub(super) fn replay_rename<'str>(
    ctx: &mut Context<'str>,
    pass: &str,
    fun_id: FunctionId,
    rename: Option<std::borrow::Cow<'str, str>>,
) -> Result<bool, String> {
    let mut changed = false;
    // A returned self-rename (cpp_demangle / name_thunks): the function is live in
    // the module, so resolve the requested name against the now-complete global
    // map (`get_unique_name` suffixes on collision) and apply it exactly as a
    // `FunctionMutRef::rename` would — the same global-name-map update.
    if let Some(name) = rename {
        let unique = ctx.get_unique_name(name);
        FunctionBody::from_id_mut(ctx, fun_id)
            .rename(unique)
            .map_err(|e| format!("{pass}: self-rename replay failed: {e}"))?;
        changed = true;
    }
    Ok(changed)
}

/// Mutations produced by a whole-program pass. `changed_functions` must include
/// every affected function and seeds the next module-fixpoint round's shared
/// function worklist. A pass that changes shared state, or cannot name a safe
/// function superset, sets `module_changed` to conservatively target the module.
#[derive(Debug)]
pub struct ModulePassOutcome {
    pub changed_functions: FxHashSet<FunctionId>,
    pub module_changed: bool,
    /// Types needed before this module pass can mutate the program. The driver
    /// publishes them and retries the pass immediately.
    pub type_requests: Vec<qcode::types::TypeRequest>,
    /// Analyses preserved by this particular invocation.
    pub(crate) preserved_analyses: PreservedAnalyses,
}

impl Default for ModulePassOutcome {
    fn default() -> Self {
        Self {
            changed_functions: FxHashSet::default(),
            module_changed: false,
            type_requests: Vec::new(),
            preserved_analyses: PreservedAnalyses::all(),
        }
    }
}

impl ModulePassOutcome {
    /// The analyses this particular invocation reported preserving.
    pub fn preserved_analyses(&self) -> &PreservedAnalyses {
        &self.preserved_analyses
    }

    pub fn changed(&self) -> bool {
        self.module_changed || !self.changed_functions.is_empty()
    }

    pub fn function(function: FunctionId) -> Self {
        Self {
            changed_functions: [function].into_iter().collect(),
            module_changed: false,
            type_requests: Vec::new(),
            preserved_analyses: PreservedAnalyses::none(),
        }
    }

    pub fn functions(functions: impl IntoIterator<Item = FunctionId>) -> Self {
        let changed_functions: FxHashSet<_> = functions.into_iter().collect();
        if changed_functions.is_empty() {
            Self::default()
        } else {
            Self {
                changed_functions,
                module_changed: false,
                type_requests: Vec::new(),
                preserved_analyses: PreservedAnalyses::none(),
            }
        }
    }

    pub fn module() -> Self {
        Self {
            changed_functions: FxHashSet::default(),
            module_changed: true,
            type_requests: Vec::new(),
            preserved_analyses: PreservedAnalyses::none(),
        }
    }

    pub fn module_if(changed: bool) -> Self {
        if changed {
            Self::module()
        } else {
            Self::default()
        }
    }

    /// Ask the driver to publish types and rerun this pass before any module
    /// mutation is performed.
    pub fn requesting_types(requests: impl IntoIterator<Item = qcode::types::TypeRequest>) -> Self {
        Self {
            type_requests: requests.into_iter().collect(),
            ..Self::default()
        }
    }

    /// Report that this particular invocation preserved global analysis `A`.
    pub fn preserving_global<A: super::GlobalAnalysis>(mut self) -> Self {
        self.preserved_analyses.preserve_global::<A>();
        self
    }

    /// Report that this particular invocation preserved local analysis `A` for
    /// every function named by `changed_functions` (or all functions when
    /// `module_changed` is set).
    pub fn preserving_local<A: super::LocalAnalysis>(mut self) -> Self {
        self.preserved_analyses.preserve_local::<A>();
        self
    }
}

/// A whole-program pass (an interprocedural milestone). Like [`FunctionPass`],
/// its [`NAME`] is the single source of truth for the pipeline name.
///
/// [`NAME`]: Pass::NAME
pub trait Pass: Default {
    const NAME: &'static str;
    fn description(&self) -> &'static str;
    fn run(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        targets: &[FunctionId],
    ) -> Result<ModulePassOutcome, String>;

    /// Analysis-aware entry point. Existing passes use [`Pass::run`]; consumers
    /// of cached global or local analyses override this method.
    fn run_with_analyses(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        targets: &[FunctionId],
        _analyses: &mut AnalysisManager,
    ) -> Result<ModulePassOutcome, String> {
        self.run(ctx, env, targets)
    }
}

/// Object-safe dispatch shim for [`Pass`], mirroring [`DynFunctionPass`].
pub trait DynPass {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn run(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        targets: &[FunctionId],
    ) -> Result<ModulePassOutcome, String>;
    fn run_with_analyses(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        targets: &[FunctionId],
        analyses: &mut AnalysisManager,
    ) -> Result<ModulePassOutcome, String>;
    /// If this module pass is a `module(<fn_pass>)` adapter, the wrapped
    /// per-function pass; `None` for a genuine whole-program pass. An incremental
    /// module-stage runner uses this to drive every adapter over the round's shared
    /// target-function set. Running the inner pass
    /// directly is equivalent to [`DynPass::run`] (which loops it over all
    /// functions), so callers may always fall back to `run`.
    fn as_module_fn(&self) -> Option<&dyn DynFunctionPass> {
        None
    }
}

impl<T: Pass> DynPass for T {
    fn name(&self) -> &'static str {
        T::NAME
    }
    fn description(&self) -> &'static str {
        Pass::description(self)
    }
    fn run(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        targets: &[FunctionId],
    ) -> Result<ModulePassOutcome, String> {
        Pass::run(self, ctx, env, targets)
    }
    fn run_with_analyses(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        targets: &[FunctionId],
        analyses: &mut AnalysisManager,
    ) -> Result<ModulePassOutcome, String> {
        #[cfg(test)]
        let call_graph_before = call_graph_snapshot(ctx);
        #[cfg(test)]
        let addresses_before = address_snapshot(ctx);
        let outcome = Pass::run_with_analyses(self, ctx, env, targets, analyses)?;
        #[cfg(test)]
        if outcome
            .preserved_analyses()
            .preserves_global_analysis::<crate::CallGraphAnalysis>()
        {
            assert_eq!(
                call_graph_before,
                call_graph_snapshot(ctx),
                "{} reported preserving CallGraphAnalysis but changed its result",
                T::NAME,
            );
        }
        #[cfg(test)]
        if outcome
            .preserved_analyses()
            .preserves_global_analysis::<crate::AddressAnalysis>()
        {
            assert_eq!(
                addresses_before,
                address_snapshot(ctx),
                "{} reported preserving AddressAnalysis but changed its result",
                T::NAME,
            );
        }
        Ok(outcome)
    }
}

/// A decompilation pass: reads the (immutable, already-optimized) qcode IR of one
/// function and reads/writes the high-level [`Program`] AST, returning `Ok(true)`
/// if it changed the AST so a stage can loop it to a fixpoint.
///
/// This is the third pass scope, distinct from [`FunctionPass`] and [`Pass`],
/// which both *mutate* the IR. A decompilation pass never touches the IR:
/// decompilation is a read-only view over qcode that progressively builds and
/// refines the AST. Region structuring, loop refinement, and the SAILR deopt
/// transforms (switch-case recovery, tail duplication, …) are all this kind of
/// pass. Its [`NAME`] is the single source of truth for the pipeline name.
///
/// [`NAME`]: DecompilePass::NAME
pub trait DecompilePass: Default {
    const NAME: &'static str;
    fn description(&self) -> &'static str;
    fn run(&self, ctx: &Context, fun_id: FunctionId, program: &mut Program)
    -> Result<bool, String>;
}

/// Object-safe dispatch shim for [`DecompilePass`], mirroring [`DynFunctionPass`].
pub trait DynDecompilePass {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn run(&self, ctx: &Context, fun_id: FunctionId, program: &mut Program)
    -> Result<bool, String>;
}

impl<T: DecompilePass> DynDecompilePass for T {
    fn name(&self) -> &'static str {
        T::NAME
    }
    fn description(&self) -> &'static str {
        DecompilePass::description(self)
    }
    fn run(
        &self,
        ctx: &Context,
        fun_id: FunctionId,
        program: &mut Program,
    ) -> Result<bool, String> {
        DecompilePass::run(self, ctx, fun_id, program)
    }
    fn run_with_analyses(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        analyses: &mut AnalysisManager,
    ) -> Result<ModulePassOutcome, String> {
        #[cfg(test)]
        let call_graph_before = call_graph_snapshot(ctx);
        let outcome = Pass::run_with_analyses(self, ctx, env, analyses)?;
        #[cfg(test)]
        if outcome
            .preserved_analyses()
            .preserves_global_analysis::<crate::CallGraphAnalysis>()
        {
            assert_eq!(
                call_graph_before,
                call_graph_snapshot(ctx),
                "{} reported preserving CallGraphAnalysis but changed its result",
                T::NAME,
            );
        }
        Ok(outcome)
    }
}

// ----- registry --------------------------------------------------------------

/// A pass resolved from its TOML name, tagged by which scope it runs in.
pub enum RegisteredPass {
    Function(Box<dyn DynFunctionPass>),
    Module(Box<dyn DynPass>),
    Decompile(Box<dyn DynDecompilePass>),
}

/// One pass's registration, submitted from the pass's own module via
/// [`inventory::submit!`] and collected here. `make` constructs a fresh boxed pass.
pub struct PassRegistration {
    pub name: &'static str,
    pub make: fn() -> RegisteredPass,
}

inventory::collect!(PassRegistration);

/// Construct the pass registered under `name`, or `None` if unknown.
///
/// The `module(<fn_pass>)` spelling resolves to a [`ModuleFnAdapter`] over the
/// named per-function pass, so a function pass can run in a module-scoped stage
/// (see the adapter docs).
pub fn make_pass(name: &str) -> Option<RegisteredPass> {
    if let Some(inner) = module_adapter_inner(name) {
        // `module(...)` only wraps a per-function pass; wrapping a module pass
        // (or an unknown name) is a config error and fails to resolve.
        let RegisteredPass::Function(inner) = make_pass(inner)? else {
            return None;
        };
        return Some(RegisteredPass::Module(Box::new(ModuleFnAdapter { inner })));
    }
    inventory::iter::<PassRegistration>()
        .find(|r| r.name == name)
        .map(|r| (r.make)())
}

/// The inner pass name of a `module(<inner>)` adapter spelling, if `name` is one.
fn module_adapter_inner(name: &str) -> Option<&str> {
    name.strip_prefix("module(")?
        .strip_suffix(')')
        .map(str::trim)
}

/// A [`FunctionPass`] adapted to run module-wide. A `module(<fn_pass>)` entry in a
/// module-scoped stage runs the wrapped per-function pass over every non-external
/// function exactly once, OR-ing their change flags.
///
/// Its purpose is to let a function pass join a module stage's `repeat_until`
/// fixpoint — e.g. looping `argpromote_registers` (module) together with
/// `module(mem2reg)` / `module(const_fold)` until the arguments are fully
/// threaded — without minting a bespoke module pass per function pass. The
/// `module(...)` spelling keeps it visible in the TOML that the underlying pass is
/// a function pass being run program-wide.
pub(super) struct ModuleFnAdapter {
    pub(super) inner: Box<dyn DynFunctionPass>,
}

impl DynPass for ModuleFnAdapter {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn description(&self) -> &'static str {
        self.inner.description()
    }
    fn run(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        targets: &[FunctionId],
    ) -> Result<ModulePassOutcome, String> {
        self.run_with_analyses(ctx, env, targets, &mut AnalysisManager::default())
    }
    fn run_with_analyses(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        targets: &[FunctionId],
        analyses: &mut AnalysisManager,
    ) -> Result<ModulePassOutcome, String> {
        super::config::run_standalone_module_fn(ctx, env, self.inner.as_ref(), targets, analyses)
    }
    fn as_module_fn(&self) -> Option<&dyn DynFunctionPass> {
        Some(&*self.inner)
    }
}

/// Register a [`FunctionPass`] under its `NAME`, wrapping it in a [`FunctionPassAdapter`]
/// so the registry stores it as an ordinary [`DynFunctionPass`]. Place one call in
/// the pass's own module; the registry builds the pass with [`Default`] each time
/// it resolves the name, which it reads from the pass's [`FunctionPass::NAME`].
///
/// ```ignore
/// register_function_pass!(ExamplePass);
/// ```
#[macro_export]
macro_rules! register_function_pass {
    ($ty:ty) => {
        inventory::submit! {
            $crate::PassRegistration {
                name: <$ty as $crate::FunctionPass>::NAME,
                make: || $crate::RegisteredPass::Function(::std::boxed::Box::new(
                    <$crate::FunctionPassAdapter<$ty> as ::core::default::Default>::default(),
                )),
            }
        }
    };
}

/// Register a whole-program ([`Pass`]) milestone. Like
/// [`register_function_pass!`], but for module-scoped passes; reads the name
/// from [`Pass::NAME`].
#[macro_export]
macro_rules! register_module_pass {
    ($ty:ty) => {
        inventory::submit! {
            $crate::PassRegistration {
                name: <$ty as $crate::Pass>::NAME,
                make: || $crate::RegisteredPass::Module(::std::boxed::Box::new(
                    <$ty as ::core::default::Default>::default(),
                )),
            }
        }
    };
}

/// Register a decompilation pass. Like [`register_function_pass!`], but for
/// AST-scoped passes; reads the name from [`DecompilePass::NAME`].
#[macro_export]
macro_rules! register_decompile_pass {
    ($ty:ty) => {
        inventory::submit! {
            $crate::PassRegistration {
                name: <$ty as $crate::DecompilePass>::NAME,
                make: || $crate::RegisteredPass::Decompile(::std::boxed::Box::new(
                    <$ty as ::core::default::Default>::default(),
                )),
            }
        }
    };
}

/// Every registered pass name, sorted, joined for error messages when a pipeline
/// names an unknown pass.
pub fn known_pass_names() -> String {
    let mut names: Vec<&'static str> = inventory::iter::<PassRegistration>()
        .map(|r| r.name)
        .collect();
    names.sort_unstable();
    names.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_env_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<PipelineEnv>();
    }

    #[test]
    fn headless_env_does_not_invent_architecture_state() {
        let ctx = Context::new();
        let spaces = ctx.space_count();
        let varnodes = ctx.varnode_count();

        let env = PipelineEnv::headless(&ctx);

        assert_eq!(env.sp_varnode, None);
        assert_eq!(ctx.space_count(), spaces);
        assert_eq!(ctx.varnode_count(), varnodes);
    }

    #[test]
    fn registered_names_are_unique() {
        let mut names: Vec<&'static str> = inventory::iter::<PassRegistration>()
            .map(|r| r.name)
            .collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            total,
            names.len(),
            "duplicate pass registration; make_pass resolves whichever it finds first"
        );
    }

    #[test]
    fn every_registration_resolves_with_its_own_name() {
        for reg in inventory::iter::<PassRegistration>() {
            let pass = make_pass(reg.name).expect("registered name must resolve");
            let (name, description) = match &pass {
                RegisteredPass::Function(p) => (p.name(), p.description()),
                RegisteredPass::Module(p) => (p.name(), p.description()),
                RegisteredPass::Decompile(p) => (p.name(), p.description()),
            };
            assert_eq!(name, reg.name, "registration name must match the pass NAME");
            assert!(!description.is_empty(), "{name} has an empty description");
        }
    }

    #[test]
    fn unknown_name_does_not_resolve() {
        assert!(make_pass("not_a_registered_pass").is_none());
    }

    #[test]
    fn unchanged_module_outcome_preserves_everything() {
        let outcome = ModulePassOutcome::functions([]);
        assert!(
            outcome
                .preserved_analyses()
                .preserves_global_analysis::<crate::CallGraphAnalysis>()
        );
        assert!(
            outcome
                .preserved_analyses()
                .preserves_local_analysis::<crate::AliasAnalysis>()
        );
    }

    #[test]
    fn changed_module_outcome_invalidates_unreported_analyses() {
        let function = FunctionId::from(0usize);
        let outcome =
            ModulePassOutcome::function(function).preserving_local::<crate::AliasAnalysis>();
        assert!(
            !outcome
                .preserved_analyses()
                .preserves_global_analysis::<crate::CallGraphAnalysis>()
        );
        assert!(
            outcome
                .preserved_analyses()
                .preserves_local_analysis::<crate::AliasAnalysis>()
        );
    }
}
