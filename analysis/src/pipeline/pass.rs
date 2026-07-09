//! The pass traits, the per-function adapter boundary, and the name registry.
//!
//! Every pass that can appear in a pipeline TOML implements one of two traits:
//!
//! - [`FunctionPass`] — operates on a single function. Most optimizations
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
//! [`DynFunctionPass`] shim, which every `FunctionPass` gets for free via a blanket
//! impl.

use std::sync::OnceLock;

use qcode::{
    context::Context,
    space::Space,
    value::{Function, FunctionId, RegisterId, Renameable, Varnode, VarnodeId},
};

use super::{ArchConfig, CallingConvention, FunctionBody, ModuleView};
use crate::structure::Program;
use crate::RegisterBase;

/// The architecture-specific inputs the register-aware passes need, resolved once
/// per pipeline run and shared by reference with every pass.
pub struct PipelineEnv {
    /// Register layout / calling convention, built by `harbinger::arch::arch_config`.
    pub cfg: ArchConfig,
    /// The stack-pointer *varnode* (`cfg.stack_pointer` resolved through
    /// `ctx.registers`), cached so passes don't re-resolve it each call.
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
        let sp_varnode = ctx.registers[&cfg.stack_pointer];
        Self::from_parts(cfg, sp_varnode)
    }

    /// Build an env for running arch-agnostic passes on hand-written IR (CLI and
    /// other tools), where there is no machine architecture to resolve. The
    /// stack pointer is a throwaway varnode and the ABI is empty, so passes that
    /// genuinely need register/ABI/stack information must not use this env —
    /// arch-agnostic transforms (e.g. `loop_to_recursion`, `gvn`, `dce`) are fine.
    pub fn headless(ctx: &mut Context) -> Self {
        let space = ctx.make_temp_space();
        let bitness = (Space::from_id(ctx, ctx.default_space).addr_size * 8) as u8;
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
    pub fn alias_base(&self, ctx: &Context) -> &RegisterBase {
        let base = self.alias_base.get_or_init(|| RegisterBase::build(ctx));
        debug_assert_eq!(
            base.varnode_count(),
            ctx.varnode_count(),
            "register alias base built from a stale varnode set; a pass minted varnodes mid-run"
        );
        base
    }
}

/// A pass over a single function. `run` returns `Ok(true)` if it changed the IR,
/// so a function-scoped stage can loop it to a fixpoint. Passes that don't track
/// change return `Ok(false)` and must not be placed in a `repeat_until` stage.
///
/// The [`Default`] bound lets a pass precompute and store values once at
/// construction; the registry builds every pass with `Default::default()`. [`NAME`]
/// is the single source of truth for the pipeline name — `register_function_pass!`
/// reads it, so it never needs repeating.
///
/// [`NAME`]: FunctionPass::NAME
pub trait FunctionPass: Default {
    const NAME: &'static str;
    fn description(&self) -> &'static str;
    fn run(&self, ctx: &mut Context, fun_id: FunctionId, env: &PipelineEnv)
    -> Result<bool, String>;
}

/// Object-safe dispatch shim for [`FunctionPass`].
///
/// [`FunctionPass`] can't be made into a trait object — its [`Default`] supertrait
/// and `NAME` associated const are both non-object-safe. This shim mirrors it as
/// instance methods and is blanket-impl'd for every `FunctionPass`, so the registry
/// can store `Box<dyn DynFunctionPass>`.
pub trait DynFunctionPass: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn run(&self, ctx: &mut Context, fun_id: FunctionId, env: &PipelineEnv)
    -> Result<bool, String>;

    /// Whether this pass is a [`FunctionPassV2`] (wrapped in a [`V2Adapter`]).
    /// A stage all of whose passes are V2 can be driven over checked-out bodies —
    /// the precondition for the parallel driver (Stage 6). Legacy [`FunctionPass`]
    /// passes answer `false`.
    fn is_v2(&self) -> bool {
        false
    }

    /// If this is a V2 pass, a handle that runs it on a body the caller has
    /// *already* checked out (no internal checkout), so the driver owns the
    /// check-out/check-in protocol. `None` for a legacy [`FunctionPass`].
    fn as_v2(&self) -> Option<&dyn CheckedV2Pass> {
        None
    }
}

impl<T: FunctionPass + Send + Sync> DynFunctionPass for T {
    fn name(&self) -> &'static str {
        T::NAME
    }
    fn description(&self) -> &'static str {
        FunctionPass::description(self)
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        env: &PipelineEnv,
    ) -> Result<bool, String> {
        FunctionPass::run(self, ctx, fun_id, env)
    }
}

/// A [`FunctionPassV2`] driven over a [`FunctionBody`] the caller already checked
/// out — no internal checkout. This is the surface the parallel driver (and the
/// sequential all-V2 fixpoint) use to run a V2 pass on a body they own; buffered
/// [`Effects`] replay and the check-in protocol are the driver's job, not this
/// method's.
///
/// [`Effects`]: super::Effects
pub trait CheckedV2Pass {
    fn run_checked<'str>(
        &self,
        m: &ModuleView<'_, 'str>,
        body: &mut FunctionBody<'str>,
    ) -> Result<bool, String>;
}

/// The parallel-safe successor to [`FunctionPass`] (Stage 5 of the
/// parallel-function-passes plan; see `PARALLEL_PASSES.md`).
///
/// The litmus test the signature enforces: **a function pass may read the
/// module's published interface and mutate its own function — nothing else.** It
/// reads the module through a `&`-shared [`ModuleView`] and mutates only its own
/// [`FunctionBody`], buffering the few legitimate global effects (assumptions,
/// discoveries, self-renames, address claims) for the driver to replay at
/// check-in. With no path to global mutable state, workers can run these in
/// parallel (Stage 6) with the `ModuleView` shared and the bodies disjoint.
///
/// A pass migrates from [`FunctionPass`] to this trait one at a time; the
/// [`V2Adapter`] lets a `FunctionPassV2` be stored and driven exactly like a
/// legacy [`FunctionPass`] (it performs the checkout → run → check-in dance
/// internally), so the sequential pipeline needs no changes while the port is in
/// flight.
pub trait FunctionPassV2: Default {
    const NAME: &'static str;
    fn description(&self) -> &'static str;
    fn run<'str>(
        &self,
        m: &ModuleView<'_, 'str>,
        f: &mut FunctionBody<'str>,
    ) -> Result<bool, String>;
}

/// Adapts a [`FunctionPassV2`] to the object-safe [`DynFunctionPass`] the registry
/// and the sequential driver speak, encapsulating the whole check-out/check-in
/// protocol in one place:
///
/// 1. Check the target function out of the context ([`Context::checkout_function`]).
/// 2. Build a [`ModuleView`] over the now-disjoint `&Context` and run the pass on
///    the owned [`FunctionBody`].
/// 3. Check the function back in and replay any buffered [`Effects`] in place.
///
/// Stage 6's parallel driver will instead check out a whole worklist and call the
/// `FunctionPassV2` directly on each disjoint body; the adapter is the sequential
/// bridge that keeps the pipeline running identically during the incremental port.
pub struct V2Adapter<T: FunctionPassV2> {
    inner: T,
}

impl<T: FunctionPassV2> Default for V2Adapter<T> {
    fn default() -> Self {
        Self {
            inner: T::default(),
        }
    }
}

impl<T: FunctionPassV2> CheckedV2Pass for V2Adapter<T> {
    fn run_checked<'str>(
        &self,
        m: &ModuleView<'_, 'str>,
        body: &mut FunctionBody<'str>,
    ) -> Result<bool, String> {
        FunctionPassV2::run(&self.inner, m, body)
    }
}

impl<T: FunctionPassV2 + Send + Sync> DynFunctionPass for V2Adapter<T> {
    fn name(&self) -> &'static str {
        T::NAME
    }
    fn description(&self) -> &'static str {
        FunctionPassV2::description(&self.inner)
    }
    fn is_v2(&self) -> bool {
        true
    }
    fn as_v2(&self) -> Option<&dyn CheckedV2Pass> {
        Some(self)
    }
    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        env: &PipelineEnv,
    ) -> Result<bool, String> {
        // Check out: the pass owns its function exclusively; the context it reads
        // through `ModuleView` no longer holds (and so cannot alias) that function.
        let fun = ctx.checkout_function(fun_id);
        // Sequential bridge: no reserved ids yet (function minting is wired with
        // the first V2 outliner port). A V2 pass that mints is guarded below.
        let mut body = FunctionBody::new(fun_id, fun, Vec::new());
        let changed = {
            let view = ModuleView::new(ctx, env);
            self.run_checked(&view, &mut body)?
        };
        let (fun, effects, minted, _reserved) = body.into_parts();
        ctx.checkin_function(fun_id, fun);
        replay_effects(ctx, T::NAME, fun_id, effects, minted)?;
        Ok(changed)
    }
}

/// Replay a V2 pass's buffered [`Effects`] into the context at check-in. Runs on
/// the master thread in worklist order; every item is an idempotent keyed insert
/// or a first-writer-wins claim, so the order within one pass's buffer is
/// immaterial.
///
/// Effect kinds are wired as the passes that produce them are ported (each port
/// commit lands its replay arm). An unwired effect surfaces as a hard error rather
/// than a silent drop, so a mis-ordered port fails loudly instead of miscompiling.
pub(super) fn replay_effects<'str>(
    ctx: &mut Context<'str>,
    pass: &str,
    fun_id: FunctionId,
    effects: super::Effects<'str>,
    minted: Vec<qcode::value::Function<'str>>,
) -> Result<bool, String> {
    let mut changed = false;
    for (prop, value) in effects.assumptions {
        debug_assert!(value, "only assume_true is buffered");
        changed |= ctx.assume_true(prop);
    }
    for discovery in effects.discoveries {
        changed |= ctx.discover(discovery);
    }
    // A buffered self-rename (cpp_demangle / name_thunks): the function is already
    // checked in, so resolve the requested name against the now-complete global
    // map (`get_unique_name` suffixes on collision) and apply it exactly as a
    // `FunctionMutRef::rename` would — the same global-name-map update.
    if let Some(name) = effects.self_rename {
        let unique = ctx.get_unique_name(name);
        Function::from_id_mut(ctx, fun_id)
            .rename(unique)
            .map_err(|e| format!("{pass}: self-rename replay failed: {e}"))?;
        changed = true;
    }
    if !effects.address_claims.is_empty() {
        return Err(format!(
            "{pass}: V2 address-claim replay not wired yet (port handle_jump_tables first)"
        ));
    }
    if !minted.is_empty() {
        return Err(format!(
            "{pass}: V2 function minting not wired yet (port the loop outliners first)"
        ));
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

/// Register a per-function pass. Place one call in the pass's own module; the
/// registry builds the pass with [`Default`] each time it resolves the name, which
/// it reads from the pass's [`FunctionPass::NAME`].
///
/// ```ignore
/// register_function_pass!(CppDemangle);
/// ```
#[macro_export]
macro_rules! register_function_pass {
    ($ty:ty) => {
        inventory::submit! {
            $crate::PassRegistration {
                name: <$ty as $crate::FunctionPass>::NAME,
                make: || $crate::RegisteredPass::Function(::std::boxed::Box::new(
                    <$ty as ::core::default::Default>::default(),
                )),
            }
        }
    };
}

/// Register a [`FunctionPassV2`] under its `NAME`, wrapping it in a [`V2Adapter`]
/// so the registry stores it as an ordinary [`DynFunctionPass`]. Drop-in
/// replacement for [`register_function_pass!`] used as each pass is ported; the
/// pipeline TOML and the driver are none the wiser.
///
/// ```ignore
/// register_function_pass_v2!(ExamplePass);
/// ```
#[macro_export]
macro_rules! register_function_pass_v2 {
    ($ty:ty) => {
        inventory::submit! {
            $crate::PassRegistration {
                name: <$ty as $crate::FunctionPassV2>::NAME,
                make: || $crate::RegisteredPass::Function(::std::boxed::Box::new(
                    <$crate::V2Adapter<$ty> as ::core::default::Default>::default(),
                )),
            }
        }
    };
}

/// Register a whole-program ([`Pass`]) milestone. Like [`register_function_pass!`],
/// but for module-scoped passes; reads the name from [`Pass::NAME`].
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
