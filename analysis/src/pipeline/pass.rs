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
    value::{FunctionBody, FunctionId, RegisterId, Renameable, Varnode, VarnodeId},
};

use super::{ArchConfig, CallingConvention, ContextSplit, ContextView, Outcome};
use crate::structure::Program;
use crate::RegisterBase;

/// The architecture-specific inputs the register-aware passes need, resolved once
/// per pipeline run and shared by reference with every pass.
pub struct PipelineEnv {
    /// Register layout / calling convention, built by `harbinger::arch::arch_config`.
    pub cfg: ArchConfig,
    /// The stack-pointer *varnode* (`cfg.stack_pointer` resolved through
    /// `ctx.shared.registers`), cached so passes don't re-resolve it each call.
    pub sp_varnode: VarnodeId,
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
    pub fn new(ctx: &Context, cfg: ArchConfig) -> Self {
        let sp_varnode = ctx.shared.registers[&cfg.stack_pointer];
        Self::from_parts(cfg, sp_varnode)
    }

    /// Build an env for running arch-agnostic passes on hand-written IR (CLI and
    /// other tools), where there is no machine architecture to resolve. The
    /// stack pointer is a throwaway varnode and the ABI is empty, so passes that
    /// genuinely need register/ABI/stack information must not use this env —
    /// arch-agnostic transforms (e.g. `loop_to_recursion`, `gvn`, `dce`) are fine.
    pub fn headless(ctx: &mut Context) -> Self {
        let space = ctx.make_temp_space();
        let bitness = (Space::from_id(&*ctx, ctx.shared.default_space).addr_size * 8) as u8;
        let sp_varnode = Varnode::make(ctx, 0, (bitness / 8) as usize, space).id;
        let cfg = ArchConfig {
            // Unused by arch-agnostic passes; the real SP is `sp_varnode` above.
            stack_pointer: RegisterId::from(0usize),
            dead_flag_regs: Vec::new(),
            abi: CallingConvention::default(),
            os: ctx.target_os(),
            bitness,
        };
        Self::from_parts(cfg, sp_varnode)
    }

    /// Build an env from already-resolved parts, without consulting a `ctx`. Used by
    /// unit tests that construct a throwaway env for arch-agnostic passes; prefer
    /// [`PipelineEnv::new`] in production.
    pub(crate) fn from_parts(cfg: ArchConfig, sp_varnode: VarnodeId) -> Self {
        Self {
            cfg,
            sp_varnode,
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
    /// showed that rebuild never fired across the entire test suite.) Any varnodes a
    /// pass mints mid-run live in fresh temporary spaces that are disjoint from the
    /// register file and are resolved per-function in "Part B", not by this shared
    /// base — so a stale count would still be sound, but the `debug_assert` below
    /// catches an unexpected mid-run mint loudly rather than silently.
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
    ) -> Result<Outcome<'str>, String>;
}

/// The parallel-safe function pass trait (Stage 5 of the
/// parallel-function-passes plan; see `PARALLEL_PASSES.md`).
///
/// The litmus test the signature enforces: **a function pass may read the
/// module's published interface and mutate its own function — nothing else.** It
/// reads the module through a `&`-shared [`ContextView`] and mutates only its own
/// [`FunctionBody`], buffering the one legitimate global effect (a self-rename)
/// for the driver to replay at the
/// barrier. With no path to global mutable state, workers can run these in
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
    ) -> Result<Outcome<'str>, String> {
        FunctionPass::run(&self.inner, body, cx, next_minted)
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        env: &PipelineEnv,
    ) -> Result<bool, String> {
        // Split the context: borrow the target body `&mut` in place and run the
        // pass over the frozen module view; the view is bodies-free, so it cannot
        // alias the borrowed body.
        let outcome = {
            let (bodies, view) = ctx.split(env);
            let mut next_minted = 0;
            self.run_checked(&mut bodies[fun_id], view, &mut next_minted)?
        };
        // Barrier, in the driver's order: install and resolve minted callees,
        // then apply the returned self-rename. The body was mutated in place.
        let installed = install_minted(ctx, T::NAME, outcome.minted)?;
        let patched = resolve_minted_callees(ctx, T::NAME, fun_id, &installed)?;
        replay_rename(ctx, T::NAME, fun_id, outcome.rename)?;
        Ok(outcome.changed || patched)
    }
}

/// Run `f` over a `(&mut FunctionBody, ContextView)` for `fid` — the body borrowed
/// `&mut` in place via [`split`](ContextSplit::split) — then drop the split borrow
/// and finish the split-borrow barrier: the run protocol of
/// [`FunctionPassAdapter::run`], minus minting.
///
/// This is the bridge the whole-`Context` optimization entry points
/// (`gvn_function`, `constant_fold_function`, `narrow_function`, `mem2reg`,
/// `mem2reg_framed`) use to reach the concrete function-pass core: their callers
/// hold a `&mut Context` but neither a [`FunctionBody`] nor a [`PipelineEnv`], and
/// they supply their own alias oracle, so the [`ContextView`]'s env is a
/// throwaway the cores never read. Because the concrete [`PassBacking`] path
/// debug-asserts the body is self-stored, every caller must feed a function with
/// no reattributed blocks — which, post the driver's `split_overlapping_functions`
/// normalization, every production function is.
///
/// [`PassBacking`]: qcode::value::util::pass_backing::PassBacking
pub(crate) fn with_checked_out_body<'str, R>(
    ctx: &mut Context<'str>,
    fid: FunctionId,
    f: impl FnOnce(&mut FunctionBody<'str>, ContextView<'_, 'str>) -> R,
) -> R {
    let env = detached_env();
    {
        let (bodies, view) = ctx.split(&env);
        // These entry points buffer no effects and mint nothing, so the drained
        // scratch is discarded.
        f(&mut bodies[fid], view)
    }
}

/// A throwaway [`PipelineEnv`] for [`with_checked_out_body`]: the concrete
/// optimization cores never consult `env()`, so its stack pointer / ABI are
/// placeholders.
fn detached_env() -> PipelineEnv {
    PipelineEnv::from_parts(
        ArchConfig {
            stack_pointer: RegisterId::from(0usize),
            dead_flag_regs: Vec::new(),
            abi: CallingConvention::default(),
            os: qcode::context::TargetOs::Unknown,
            bitness: 64,
        },
        VarnodeId::from(0usize),
    )
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
    for (slot, &real) in installed.iter().enumerate() {
        let slot = u32::try_from(slot).expect("more than u32::MAX minted functions installed");
        changed |= ctx.bodies[owner].resolve_minted_callee(slot, real) != 0;
        for &minted_id in installed {
            changed |= ctx.bodies[minted_id].resolve_minted_callee(slot, real) != 0;
        }
    }
    for fun_id in std::iter::once(owner).chain(installed.iter().copied()) {
        // Inspect the whole live instruction arena, not just instructions linked
        // into rostered blocks. A temporarily detached instruction is still live
        // pass state and must not carry an unresolved placeholder past the
        // barrier.
        let unresolved = ctx.bodies[fun_id].minted_callee_slots().into_iter().next();
        if let Some(slot) = unresolved {
            return Err(format!(
                "{pass}: function {fun_id:?} references minted callee #{slot}, but only {} were installed",
                installed.len()
            ));
        }
    }
    Ok(changed)
}

#[cfg(test)]
mod minted_barrier_tests {
    use super::*;
    use qcode::{
        builder::Builder,
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
            let mut builder = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, block));
            builder.push_call(callee);
            unsafe { builder.dont_finalize() };
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
            let mut builder = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, block));
            builder.push_call(sibling);
            unsafe { builder.dont_finalize() };
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

/// A whole-program pass (an interprocedural milestone). `run` returns `Ok(true)`
/// if it changed anything; the only milestones today report `Ok(false)`. Like
/// [`FunctionPass`], its [`NAME`] is the single source of truth for the pipeline
/// name.
///
/// [`NAME`]: Pass::NAME
pub trait Pass: Default {
    const NAME: &'static str;
    fn description(&self) -> &'static str;
    fn run(&self, ctx: &mut Context, env: &PipelineEnv) -> Result<bool, String>;
}

/// Object-safe dispatch shim for [`Pass`], mirroring [`DynFunctionPass`].
pub trait DynPass {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn run(&self, ctx: &mut Context, env: &PipelineEnv) -> Result<bool, String>;
    /// If this module pass is a `module(<fn_pass>)` adapter, the wrapped
    /// per-function pass; `None` for a genuine whole-program pass. A dirty-tracking
    /// module-stage runner uses this to drive the adapter function-by-function and
    /// skip functions already at the pass's fixpoint. Running the inner pass
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
    fn run(&self, ctx: &mut Context, env: &PipelineEnv) -> Result<bool, String> {
        Pass::run(self, ctx, env)
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
struct ModuleFnAdapter {
    inner: Box<dyn DynFunctionPass>,
}

impl DynPass for ModuleFnAdapter {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn description(&self) -> &'static str {
        self.inner.description()
    }
    fn run(&self, ctx: &mut Context, env: &PipelineEnv) -> Result<bool, String> {
        let fun_ids: Vec<FunctionId> = ctx
            .functions()
            .filter(|f| !f.is_external())
            // Honor `--ignore`: a function-pass run module-wide must still skip
            // functions the user marked ignored.
            .filter(|f| !ctx.is_function_ignored(f.address()))
            .map(|f| f.id)
            .collect();
        let mut changed = false;
        for fun_id in fun_ids {
            changed |= self.inner.run(ctx, fun_id, env)?;
        }
        Ok(changed)
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
}
