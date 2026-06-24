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

use std::cell::{Ref, RefCell};

use qcode::{
    context::Context,
    value::{FunctionId, VarnodeId},
};

use super::ArchConfig;
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
    /// Function-independent register/varnode alias base, built lazily and shared
    /// across the per-function GVN runs (see [`PipelineEnv::alias_base`]). Behind a
    /// `RefCell` because passes hold `&PipelineEnv`; sound because the function-pass
    /// runner is single-threaded.
    alias_base: RefCell<Option<RegisterBase>>,
}

impl PipelineEnv {
    /// Resolve the stack-pointer varnode from `cfg` against `ctx` once. No lifter:
    /// the lifting passes will be inert.
    pub fn new(ctx: &Context, cfg: ArchConfig) -> Self {
        let sp_varnode = ctx.registers[&cfg.stack_pointer];
        Self::from_parts(cfg, sp_varnode)
    }

    /// Build an env from already-resolved parts, without consulting a `ctx`. Used by
    /// unit tests that construct a throwaway env for arch-agnostic passes; prefer
    /// [`PipelineEnv::new`] in production.
    pub(crate) fn from_parts(cfg: ArchConfig, sp_varnode: VarnodeId) -> Self {
        Self {
            cfg,
            sp_varnode,
            alias_base: RefCell::new(None),
        }
    }

    /// The shared register/varnode alias base ("Part A" of the simple alias
    /// analysis) for `ctx`. Built on first use and reused while the varnode set is
    /// unchanged; rebuilt only when a varnode is added mid-run (the registry is
    /// append-only, so a changed `varnode_count` is the validity key). The
    /// per-function GVN pass finishes it with [`RegisterBase::for_function`], so the
    /// O(varnodes·log) base build no longer runs on every function.
    pub fn alias_base(&self, ctx: &Context) -> Ref<'_, RegisterBase> {
        let valid = matches!(
            &*self.alias_base.borrow(),
            Some(base) if base.varnode_count() == ctx.varnode_count()
        );
        if !valid {
            *self.alias_base.borrow_mut() = Some(RegisterBase::build(ctx));
        }
        Ref::map(self.alias_base.borrow(), |o| {
            o.as_ref().expect("alias_base just built")
        })
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
pub trait DynFunctionPass {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn run(&self, ctx: &mut Context, fun_id: FunctionId, env: &PipelineEnv)
    -> Result<bool, String>;
}

impl<T: FunctionPass> DynFunctionPass for T {
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
