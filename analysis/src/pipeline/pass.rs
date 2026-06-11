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

use qcode::{
    context::Context,
    value::{FunctionId, VarnodeId},
};

use super::ArchConfig;

/// The architecture-specific inputs the register-aware passes need, resolved once
/// per pipeline run and shared by reference with every pass.
pub struct PipelineEnv {
    /// Register layout / calling convention, built by `harbinger::arch::arch_config`.
    pub cfg: ArchConfig,
    /// The stack-pointer *varnode* (`cfg.stack_pointer` resolved through
    /// `ctx.registers`), cached so passes don't re-resolve it each call.
    pub sp_varnode: VarnodeId,
}

impl PipelineEnv {
    /// Resolve the stack-pointer varnode from `cfg` against `ctx` once.
    pub fn new(ctx: &Context, cfg: ArchConfig) -> Self {
        let sp_varnode = ctx.registers[&cfg.stack_pointer];
        Self { cfg, sp_varnode }
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

// ----- registry --------------------------------------------------------------

/// A pass resolved from its TOML name, tagged by which scope it runs in.
pub enum RegisteredPass {
    Function(Box<dyn DynFunctionPass>),
    Module(Box<dyn DynPass>),
}

/// One pass's registration, submitted from the pass's own module via
/// [`inventory::submit!`] and collected here. `make` constructs a fresh boxed pass.
pub struct PassRegistration {
    pub name: &'static str,
    pub make: fn() -> RegisteredPass,
}

inventory::collect!(PassRegistration);

/// Construct the pass registered under `name`, or `None` if unknown.
pub fn make_pass(name: &str) -> Option<RegisteredPass> {
    inventory::iter::<PassRegistration>()
        .find(|r| r.name == name)
        .map(|r| (r.make)())
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
