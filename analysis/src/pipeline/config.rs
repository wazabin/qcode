//! Parsing a pipeline from TOML and running it.
//!
//! A [`Pipeline`] is an ordered list of stages; see `default_pipeline.toml` for
//! the schema. Names are resolved against the registry eagerly at [`parse`] time,
//! so a pipeline that names an unknown pass (or puts a per-function pass in a
//! module-scoped stage) fails to load and never runs.
//!
//! [`parse`]: Pipeline::parse

use serde::Deserialize;

use qcode::{
    context::Context,
    value::{FunctionId, FunctionRef},
};

use super::PipelineProgress;
use super::pass::{FunctionPass, PASS_NAMES, Pass, PipelineEnv, RegisteredPass, make_pass};

/// The canonical default pipeline, compiled into the binary. Used by
/// `analyze_default` and as the GUI's starting pipeline.
pub const DEFAULT_PIPELINE_TOML: &str = include_str!("default_pipeline.toml");

/// Guard against a `repeat_until` stage that never converges.
const MAX_FIXPOINT_ITERS: usize = 100;

#[derive(Deserialize)]
struct PipelineConfig {
    #[serde(default)]
    stage: Vec<StageConfig>,
}

#[derive(Deserialize)]
struct StageConfig {
    name: String,
    scope: Scope,
    passes: Vec<String>,
    #[serde(default)]
    repeat_until: Option<RepeatCond>,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Scope {
    Function,
    Module,
}

/// The condition under which a stage repeats. Only `no_change` exists today.
#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RepeatCond {
    NoChange,
}

/// A stage's resolved passes, partitioned by scope.
enum StagePasses {
    Function(Vec<Box<dyn FunctionPass>>),
    Module(Vec<Box<dyn Pass>>),
}

struct Stage {
    name: String,
    passes: StagePasses,
    repeat_until: Option<RepeatCond>,
}

/// A parsed, name-resolved analysis pipeline ready to run.
pub struct Pipeline {
    stages: Vec<Stage>,
}

impl Default for Pipeline {
    /// The canonical [`DEFAULT_PIPELINE_TOML`] pipeline. Panics only if that
    /// compiled-in TOML is malformed, which a unit test guards against.
    fn default() -> Self {
        Pipeline::parse(DEFAULT_PIPELINE_TOML).expect("default pipeline TOML is valid")
    }
}

impl Pipeline {
    /// Parse and name-resolve a pipeline from TOML source. Every pass name is
    /// resolved against the registry up front; an unknown name (or a scope
    /// mismatch) is a hard error and the pipeline does not load.
    pub fn parse(toml_src: &str) -> Result<Pipeline, String> {
        let config: PipelineConfig =
            toml::from_str(toml_src).map_err(|e| format!("pipeline TOML parse error: {e}"))?;

        let mut stages = Vec::with_capacity(config.stage.len());
        for sc in config.stage {
            let passes = match sc.scope {
                Scope::Function => StagePasses::Function(resolve_function_passes(&sc)?),
                Scope::Module => StagePasses::Module(resolve_module_passes(&sc)?),
            };
            stages.push(Stage {
                name: sc.name,
                passes,
                repeat_until: sc.repeat_until,
            });
        }
        Ok(Pipeline { stages })
    }

    /// Build a single function-scoped stage from pass names (à-la-carte), for
    /// tools that run a chosen set of per-function passes directly over a program
    /// without the binding milestones or the checkpoint+replay driver. Names are
    /// resolved eagerly; an unknown or whole-program name is an error.
    pub fn function_passes(names: &[&str]) -> Result<Pipeline, String> {
        let sc = StageConfig {
            name: "passes".to_string(),
            scope: Scope::Function,
            passes: names.iter().map(|n| n.to_string()).collect(),
            repeat_until: None,
        };
        Ok(Pipeline {
            stages: vec![Stage {
                name: sc.name.clone(),
                passes: StagePasses::Function(resolve_function_passes(&sc)?),
                repeat_until: None,
            }],
        })
    }

    /// Run every stage in order over `ctx`. `round` and `progress` are threaded
    /// through to [`PipelineProgress`] events for the GUI/loader; pass `0` and an
    /// empty closure when neither matters.
    pub fn run(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        round: usize,
        progress: &mut impl FnMut(PipelineProgress),
    ) -> Result<(), String> {
        for stage in &self.stages {
            match &stage.passes {
                StagePasses::Module(passes) => {
                    run_module_stage(ctx, env, stage, passes, round, progress)?
                }
                StagePasses::Function(passes) => {
                    run_function_stage(ctx, env, stage, passes, round, progress)?
                }
            }
        }
        Ok(())
    }
}

fn resolve_function_passes(sc: &StageConfig) -> Result<Vec<Box<dyn FunctionPass>>, String> {
    let mut resolved = Vec::with_capacity(sc.passes.len());
    for name in &sc.passes {
        match make_pass(name) {
            Some(RegisteredPass::Function(p)) => resolved.push(p),
            Some(RegisteredPass::Module(_)) => {
                return Err(format!(
                    "pass \"{name}\" in stage \"{}\" is a whole-program pass, but the stage is scope=function",
                    sc.name
                ));
            }
            None => return Err(unknown_pass(name, &sc.name)),
        }
    }
    Ok(resolved)
}

fn resolve_module_passes(sc: &StageConfig) -> Result<Vec<Box<dyn Pass>>, String> {
    let mut resolved = Vec::with_capacity(sc.passes.len());
    for name in &sc.passes {
        match make_pass(name) {
            Some(RegisteredPass::Module(p)) => resolved.push(p),
            Some(RegisteredPass::Function(_)) => {
                return Err(format!(
                    "pass \"{name}\" in stage \"{}\" is a per-function pass, but the stage is scope=module",
                    sc.name
                ));
            }
            None => return Err(unknown_pass(name, &sc.name)),
        }
    }
    Ok(resolved)
}

fn unknown_pass(name: &str, stage: &str) -> String {
    format!(
        "unknown pass \"{name}\" in stage \"{stage}\". Known passes: {}",
        PASS_NAMES.join(", ")
    )
}

/// Run a whole-program stage; if `repeat_until` is set, loop the stage (OR-ing
/// the passes' change flags) until nothing changes or the iteration cap is hit.
fn run_module_stage(
    ctx: &mut Context,
    env: &PipelineEnv,
    stage: &Stage,
    passes: &[Box<dyn Pass>],
    round: usize,
    progress: &mut impl FnMut(PipelineProgress),
) -> Result<(), String> {
    let mut iters = 0;
    loop {
        let mut changed = false;
        for p in passes {
            progress(PipelineProgress::WholeProgramPhase {
                round,
                stage: stage.name.clone(),
                pass: p.name(),
            });
            changed |= p.run(ctx, env).map_err(|e| format!("{}: {e}", p.name()))?;
        }
        iters += 1;
        if stage.repeat_until.is_none() || !changed {
            return Ok(());
        }
        if iters >= MAX_FIXPOINT_ITERS {
            return Err(format!(
                "stage \"{}\" did not converge after {MAX_FIXPOINT_ITERS} iterations",
                stage.name
            ));
        }
    }
}

/// Run a function-scoped stage function-major: for each non-external function,
/// run the stage's passes; if `repeat_until` is set, loop that function's passes
/// to a fixpoint before moving to the next function.
fn run_function_stage(
    ctx: &mut Context,
    env: &PipelineEnv,
    stage: &Stage,
    passes: &[Box<dyn FunctionPass>],
    round: usize,
    progress: &mut impl FnMut(PipelineProgress),
) -> Result<(), String> {
    let fun_ids: Vec<FunctionId> = ctx
        .functions()
        .filter(|f| !f.is_external())
        .map(|f| f.id)
        .collect();
    let total = fun_ids.len();

    for (index, fun_id) in fun_ids.into_iter().enumerate() {
        let function = FunctionRef::from_id(ctx, fun_id).name().to_string();
        let mut iters = 0;
        loop {
            let mut changed = false;
            for p in passes {
                progress(PipelineProgress::FunctionPass {
                    round,
                    stage: stage.name.clone(),
                    function: function.clone(),
                    index: index + 1,
                    total,
                    pass: p.name(),
                });
                changed |= p
                    .run(ctx, fun_id, env)
                    .map_err(|e| format!("{}: {e}", p.name()))?;
            }
            iters += 1;
            if stage.repeat_until.is_none() || !changed {
                break;
            }
            if iters >= MAX_FIXPOINT_ITERS {
                return Err(format!(
                    "stage \"{}\" did not converge on function {function} after {MAX_FIXPOINT_ITERS} iterations",
                    stage.name
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_pipeline_parses() {
        Pipeline::default();
    }

    fn parse_err(toml_src: &str) -> String {
        match Pipeline::parse(toml_src) {
            Ok(_) => panic!("expected parse error"),
            Err(e) => e,
        }
    }

    #[test]
    fn unknown_pass_is_rejected() {
        let err = parse_err(
            r#"
            [[stage]]
            name = "x"
            scope = "function"
            passes = ["mem2rg"]
            "#,
        );
        assert!(err.contains("unknown pass \"mem2rg\""), "{err}");
        assert!(
            err.contains("mem2reg"),
            "error should list valid names: {err}"
        );
    }

    #[test]
    fn scope_mismatch_is_rejected() {
        let err = parse_err(
            r#"
            [[stage]]
            name = "x"
            scope = "function"
            passes = ["bind_args"]
            "#,
        );
        assert!(err.contains("whole-program pass"), "{err}");
    }
}
