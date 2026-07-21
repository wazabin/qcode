//! Parsing a pipeline from TOML and running it.
//!
//! A [`Pipeline`] is an ordered list of stages; see `default_pipeline.toml` for
//! the schema. Names are resolved against the registry eagerly at [`parse`] time,
//! so a pipeline that names an unknown pass (or puts a per-function pass in a
//! module-scoped stage) fails to load and never runs.
//!
//! [`parse`]: Pipeline::parse

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

use serde::Deserialize;

use qcode::{
    context::Context,
    value::{FunctionId, FunctionRef},
};

use super::lifter::PipelineServices;
use super::pass::{
    DynFunctionPass, DynPass, PipelineEnv, RegisteredPass, install_minted, known_pass_names,
    make_pass, replay_rename, resolve_minted_callees,
};
use super::{
    AnalysisManager, ContextSplit, ContextView, FunctionBody, LocalAnalysisManager, Outcome,
    PipelineProgress, PreservedAnalyses,
};

/// The canonical default pipeline, compiled into the binary. Used by
/// `analyze_default` and as the GUI's starting pipeline.
pub const DEFAULT_PIPELINE_TOML: &str = include_str!("default_pipeline.toml");
#[cfg(not(target_arch = "wasm32"))]
const DEFAULT_PIPELINE_FILE: &str = "default.toml";

/// Guard against a `repeat_until` stage that never converges.
const MAX_FIXPOINT_ITERS: usize = 100;

/// Minimum worklist size before a stage fans out across threads. Below
/// this the thread-spawn + split/barrier overhead outweighs the win, so the
/// stage runs sequentially (identical output either way).
const PARALLEL_THRESHOLD: usize = 4;

/// Iteration count at which a `repeat_until` stage starts *tracing* itself to
/// diagnose non-convergence. A healthy stage settles in a handful of iterations,
/// so tracing engages only once a stage is clearly struggling — keeping the
/// (string-rendering) fingerprint cost off the common path. Once engaged, every
/// pass that reports a change has the resulting IR fingerprinted and logged, and a
/// recurring fingerprint is flagged as a proven cycle (see [`FixpointTracer`]).
const FIXPOINT_WATCH_ITERS: usize = 16;

/// Build the error a `repeat_until` stage returns when it hits the iteration cap.
///
/// It flows through the ordinary `Result<_, String>` plumbing, but the analysis
/// driver recognizes it via [`is_nonconvergence`] and treats it as a *soft, best-
/// effort halt* — stopping the pipeline and surfacing the (last, non-converged) IR
/// for inspection — rather than a hard failure. Every non-convergence site routes
/// through here so the phrasing [`is_nonconvergence`] keys on stays in one place.
pub(crate) fn nonconvergence_error(stage: &str, function: Option<&str>) -> String {
    match function {
        Some(f) => format!(
            "stage \"{stage}\" did not converge on function {f} after {MAX_FIXPOINT_ITERS} iterations"
        ),
        None => format!("stage \"{stage}\" did not converge after {MAX_FIXPOINT_ITERS} iterations"),
    }
}

/// Whether a pipeline error string is a stage non-convergence (see
/// [`nonconvergence_error`]). Used by the driver to halt best-effort instead of
/// panicking, so the non-converged IR can be viewed rather than lost to a crash.
pub(crate) fn is_nonconvergence(error: &str) -> bool {
    error.contains("did not converge")
}

/// Name of the stage that marks the boundary between the lifting (clean-IR) phase
/// and the optimization phase. Stages before it run on the persistent clean IR
/// (recursive disassembly); stages from it onward run on the derived optimized
/// clone. The marker stage itself is a delimiter and normally carries no passes.
pub const LIFT_BARRIER: &str = "code-discovery-fixpoint";

/// Last function-local pass needed for address discovery in the default pipeline.
/// Interprocedural analysis is intentionally deferred until no more discovered
/// addresses need lifting.
const ADDRESS_DISCOVERY_PASS: &str = "handle_jump_tables";

#[derive(Deserialize)]
struct PipelineConfig {
    #[serde(default)]
    description: Option<String>,
    /// Debugging aid: when set, run the whole-program verifier after every
    /// pipeline stage and fail at the first invariant violation.
    #[serde(default)]
    debug: bool,
    #[serde(default)]
    stage: Vec<StageConfig>,
}

#[derive(Deserialize)]
struct PipelineMetadata {
    #[serde(default)]
    description: Option<String>,
}

#[derive(Deserialize)]
struct StageConfig {
    name: String,
    scope: Scope,
    passes: Vec<String>,
    #[serde(default)]
    repeat_until: Option<RepeatCond>,
    /// Function-scoped stages skip external (bodyless) functions by default,
    /// since the value-producing passes have nothing to chew on. Naming passes
    /// like `cpp_demangle` want them too, so a stage can opt in.
    #[serde(default)]
    include_external: bool,
    /// Restrict a function-scoped stage to functions changed by the previous
    /// stage. Module-scoped changes are treated conservatively as "unknown", so
    /// a following dirty-only stage runs on all functions if a module pass changed.
    #[serde(default)]
    only_dirty: bool,
    /// Debugging aid: function selectors (name, or decimal/hex address) whose
    /// QCode is dumped to stderr before this stage runs. Empty in normal use.
    #[serde(default)]
    dump: Vec<String>,
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
    Function(Vec<Box<dyn DynFunctionPass>>),
    Module(Vec<Box<dyn DynPass>>),
}

struct Stage {
    name: String,
    passes: StagePasses,
    repeat_until: Option<RepeatCond>,
    /// Run function-scoped passes on external functions too (see [`StageConfig`]).
    include_external: bool,
    /// Restrict this function-scoped stage to functions dirtied by the previous stage.
    only_dirty: bool,
    /// Function selectors whose QCode is dumped before the stage (see [`StageConfig`]).
    dump: Vec<String>,
}

/// A parsed, name-resolved analysis pipeline ready to run.
pub struct Pipeline {
    stages: Vec<Stage>,
    debug: bool,
    /// Function entry addresses the user asked to skip optimizing (`--ignore`).
    /// Carried from the CLI to the loader, which stamps them onto the lifted
    /// [`Context`](qcode::context::Context) before analysis runs.
    ignored_functions: HashSet<u64>,
    /// Code addresses exported from a previous run, replayed into the lifter's
    /// first pass so the disas↔analyze fixpoint converges in fewer rounds
    /// (`--code-map`). The loader seeds these onto the context before lifting.
    seeds: Vec<qcode::discovery::CodeSeed>,
}

/// A TOML pipeline available from the user's runtime pipeline directory.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineFile {
    pub name: String,
    pub description: String,
    pub path: PathBuf,
}

impl Default for Pipeline {
    /// The canonical [`DEFAULT_PIPELINE_TOML`] pipeline. Panics only if that
    /// compiled-in TOML is malformed, which a unit test guards against.
    fn default() -> Self {
        Pipeline::parse(DEFAULT_PIPELINE_TOML).expect("default pipeline TOML is valid")
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn list_user_pipelines_in(dir: &Path) -> Result<Vec<PipelineFile>, String> {
    ensure_user_pipeline_dir(dir)?;
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)
        .map_err(|e| format!("failed to read pipeline directory {}: {e}", dir.display()))?
    {
        let entry = entry.map_err(|e| {
            format!(
                "failed to read an entry in pipeline directory {}: {e}",
                dir.display()
            )
        })?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        let Some(name) = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(str::to_string)
        else {
            continue;
        };
        files.push(pipeline_file(path, name));
    }
    sort_pipeline_files(&mut files);
    Ok(files)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn create_user_pipeline_from_default_in(dir: &Path) -> Result<PipelineFile, String> {
    ensure_user_pipeline_dir(dir)?;
    let path = unique_pipeline_path(dir);
    write_default_pipeline_to_path(path)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn create_named_user_pipeline_from_default_in(
    dir: &Path,
    name: &str,
) -> Result<PipelineFile, String> {
    ensure_user_pipeline_dir(dir)?;
    let name = normalize_pipeline_name(name)?;
    let path = dir.join(format!("{name}.toml"));
    if path.exists() {
        return Err(format!("pipeline \"{name}\" already exists"));
    }
    write_default_pipeline_to_path(path)
}

#[cfg(not(target_arch = "wasm32"))]
fn write_default_pipeline_to_path(path: PathBuf) -> Result<PipelineFile, String> {
    std::fs::write(&path, DEFAULT_PIPELINE_TOML)
        .map_err(|e| format!("failed to write new pipeline {}: {e}", path.display()))?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("pipeline")
        .to_string();
    Ok(pipeline_file(path, name))
}

#[cfg(not(target_arch = "wasm32"))]
fn normalize_pipeline_name(name: &str) -> Result<String, String> {
    let name = name.trim().trim_end_matches(".toml").trim();
    if name.is_empty() {
        return Err("pipeline name cannot be empty".to_string());
    }
    if name == "default" {
        return Err("default is reserved for the built-in pipeline".to_string());
    }
    if name.contains('/') || name.contains('\\') {
        return Err("pipeline name cannot contain path separators".to_string());
    }
    if name == "." || name == ".." || name.starts_with('.') {
        return Err("pipeline name cannot start with a dot".to_string());
    }
    Ok(name.to_string())
}

#[cfg(not(target_arch = "wasm32"))]
fn unique_pipeline_path(dir: &Path) -> PathBuf {
    let first = dir.join("pipeline.toml");
    if !first.exists() {
        return first;
    }

    for i in 2.. {
        let path = dir.join(format!("pipeline-{i}.toml"));
        if !path.exists() {
            return path;
        }
    }
    unreachable!("unbounded counter should find a pipeline filename")
}

#[cfg(not(target_arch = "wasm32"))]
fn pipeline_file(path: PathBuf, name: String) -> PipelineFile {
    let description = std::fs::read_to_string(&path)
        .ok()
        .and_then(|src| toml::from_str::<PipelineMetadata>(&src).ok())
        .and_then(|metadata| metadata.description)
        .unwrap_or_default();
    PipelineFile {
        name,
        description,
        path,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn sort_pipeline_files(files: &mut [PipelineFile]) {
    files.sort_by(
        |a, b| match (a.name.as_str() == "default", b.name.as_str() == "default") {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.name.cmp(&b.name),
        },
    );
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_named_user_pipeline_in(dir: &Path, name: &str) -> Result<Pipeline, String> {
    let files = list_user_pipelines_in(dir)?;
    let file = files
        .into_iter()
        .find(|file| file.name == name)
        .ok_or_else(|| format!("unknown pipeline \"{name}\""))?;
    let src = std::fs::read_to_string(&file.path)
        .map_err(|e| format!("failed to read pipeline {}: {e}", file.path.display()))?;
    Pipeline::parse(&src).map_err(|e| format!("{}: {e}", file.path.display()))
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure_user_pipeline_dir(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir)
        .map_err(|e| format!("failed to create pipeline directory {}: {e}", dir.display()))?;

    // `default.toml` is a *managed* file: we keep it byte-for-byte in sync with
    // the embedded `DEFAULT_PIPELINE_TOML` so that pipeline changes shipped in
    // the binary take effect without the user having to delete a stale copy. (A
    // copy made by an older binary would otherwise shadow the embedded default
    // forever — e.g. miss a newly added pass.) Users who want a customized
    // pipeline create a differently-named file; `default` is not theirs to edit.
    let path = dir.join(DEFAULT_PIPELINE_FILE);
    let up_to_date = std::fs::read_to_string(&path)
        .map(|existing| existing == DEFAULT_PIPELINE_TOML)
        .unwrap_or(false);
    if !up_to_date {
        std::fs::write(&path, DEFAULT_PIPELINE_TOML)
            .map_err(|e| format!("failed to write default pipeline {}: {e}", path.display()))?;
    }

    Ok(())
}

impl Pipeline {
    /// Ordered names of the pipeline's stages, for progress display. Stages with
    /// no passes (e.g. the lift-boundary marker) are included so the reported
    /// stage name from [`PipelineProgress`] always resolves to an index here.
    pub fn stage_names(&self) -> Vec<String> {
        self.stages.iter().map(|s| s.name.clone()).collect()
    }

    /// Parse and name-resolve a pipeline from TOML source. Every pass name is
    /// resolved against the registry up front; an unknown name (or a scope
    /// mismatch) is a hard error and the pipeline does not load.
    pub fn parse(toml_src: &str) -> Result<Pipeline, String> {
        let config: PipelineConfig =
            toml::from_str(toml_src).map_err(|e| format!("pipeline TOML parse error: {e}"))?;
        let PipelineConfig {
            description: _description,
            debug,
            stage,
        } = config;

        let mut stages = Vec::with_capacity(stage.len());
        for sc in stage {
            if sc.only_dirty && sc.scope != Scope::Function {
                return Err(format!(
                    "stage \"{}\" sets only_dirty=true, but only function-scoped stages can use it",
                    sc.name
                ));
            }
            let passes = match sc.scope {
                Scope::Function => StagePasses::Function(resolve_function_passes(&sc)?),
                Scope::Module => StagePasses::Module(resolve_module_passes(&sc)?),
            };
            stages.push(Stage {
                name: sc.name,
                passes,
                repeat_until: sc.repeat_until,
                include_external: sc.include_external,
                only_dirty: sc.only_dirty,
                dump: sc.dump,
            });
        }
        Ok(Pipeline {
            stages,
            debug,
            ignored_functions: HashSet::default(),
            seeds: Vec::new(),
        })
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
            include_external: false,
            only_dirty: false,
            dump: Vec::new(),
        };
        Ok(Pipeline {
            stages: vec![Stage {
                name: sc.name.clone(),
                passes: StagePasses::Function(resolve_function_passes(&sc)?),
                repeat_until: None,
                include_external: sc.include_external,
                only_dirty: sc.only_dirty,
                dump: Vec::new(),
            }],
            debug: false,
            ignored_functions: HashSet::default(),
            seeds: Vec::new(),
        })
    }

    /// Mark the given function entry addresses to be skipped by every
    /// per-function pass (`--ignore`). The loader stamps these onto the lifted
    /// context before analysis. Returns `self` for builder-style chaining.
    pub fn with_ignored_functions(mut self, addrs: HashSet<u64>) -> Self {
        self.ignored_functions = addrs;
        self
    }

    /// The function entry addresses this pipeline run skips optimizing.
    pub fn ignored_functions(&self) -> &HashSet<u64> {
        &self.ignored_functions
    }

    /// Replay exported code addresses (`--code-map`) into the lifter's first
    /// pass. The loader seeds these onto the context before lifting. Returns
    /// `self` for builder-style chaining.
    pub fn with_seeds(mut self, seeds: Vec<qcode::discovery::CodeSeed>) -> Self {
        self.seeds = seeds;
        self
    }

    /// The exported code addresses this pipeline run pre-seeds onto the context.
    pub fn seeds(&self) -> &[qcode::discovery::CodeSeed] {
        &self.seeds
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
        self.run_stages(0..self.stages.len(), ctx, env, round, progress)
    }

    /// Run the clean-IR lifting stages before the `code-discovery-fixpoint`
    /// barrier. These are ordinary TOML stages; the caller supplies lifter
    /// services for TOML-visible lifting passes.
    pub fn run_lifting_phase(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        services: &mut PipelineServices<'_>,
        round: usize,
        progress: &mut impl FnMut(PipelineProgress),
    ) -> Result<(), String> {
        let end = self.barrier_index().unwrap_or(self.stages.len());
        let mut dirty_functions = Some(HashSet::default());
        let mut cache = FixpointCache::default();
        let mut analyses = AnalysisManager::default();
        for stage in &self.stages[..end] {
            match &stage.passes {
                StagePasses::Module(passes) => {
                    let outcome = run_lifting_module_stage(
                        ctx,
                        env,
                        services,
                        stage,
                        passes,
                        &mut cache,
                        &mut analyses,
                        round,
                        progress,
                    )?;
                    dirty_functions = if outcome.module_changed {
                        None
                    } else {
                        Some(outcome.changed_functions)
                    };
                }
                StagePasses::Function(passes) => {
                    dirty_functions = Some(run_function_stage(
                        ctx,
                        env,
                        stage,
                        passes,
                        dirty_functions.as_ref(),
                        None,
                        &mut cache,
                        &mut analyses,
                        round,
                        progress,
                    )?);
                }
            }
            self.verify_after_stage(ctx, stage)?;
        }
        Ok(())
    }

    /// Index of the [`LIFT_BARRIER`] marker stage, if present.
    fn barrier_index(&self) -> Option<usize> {
        self.stages.iter().position(|s| s.name == LIFT_BARRIER)
    }

    /// Run only the function-local analysis needed to discover additional code
    /// addresses. Module-scoped stages are skipped here; the full interprocedural
    /// pipeline runs only after lifting reaches a fixpoint.
    ///
    /// `restrict`, when `Some`, limits analysis to functions whose clean IR changed
    /// since the previous discovery round. Address discovery is function-local (only
    /// `Function` stages run here; `Module` stages are skipped), so a function with
    /// unchanged IR re-derives exactly the discoveries it produced last round — which
    /// were already drained — making it safe to skip. This is the dominant cost on
    /// large binaries: a late round re-optimizes thousands of functions to find a
    /// handful of new addresses.
    pub fn run_address_discovery_phase(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        restrict: Option<&HashSet<FunctionId>>,
        round: usize,
        progress: &mut impl FnMut(PipelineProgress),
    ) -> Result<(), String> {
        let start = self.barrier_index().map(|i| i + 1).unwrap_or(0);
        let mut dirty_functions = Some(HashSet::default());
        let mut cache = FixpointCache::default();
        let mut analyses = AnalysisManager::default();
        for stage in &self.stages[start..] {
            match &stage.passes {
                StagePasses::Function(passes) => {
                    let reaches_discovery_pass =
                        passes.iter().any(|p| p.name() == ADDRESS_DISCOVERY_PASS);
                    dirty_functions = Some(run_function_stage(
                        ctx,
                        env,
                        stage,
                        passes,
                        dirty_functions.as_ref(),
                        restrict,
                        &mut cache,
                        &mut analyses,
                        round,
                        progress,
                    )?);
                    self.verify_after_stage(ctx, stage)?;
                    if reaches_discovery_pass {
                        return Ok(());
                    }
                }
                StagePasses::Module(passes) => {
                    // Module stages are normally skipped during discovery (address
                    // discovery is function-local). The lone exception is the
                    // module-scoped jump-table stage, which is itself the discovery
                    // pass: it queues `discover_code` targets, so it must run here.
                    // Like the function branch above, stop once it has run — later
                    // stages are for the final optimization phase, not discovery. It
                    // ignores `restrict` and scans every function, but a function with
                    // no indirect branch bails cheaply and re-queued targets were
                    // already drained, so re-running an unchanged function is a no-op.
                    if passes.iter().any(|p| p.name() == ADDRESS_DISCOVERY_PASS) {
                        run_module_stage(
                            ctx,
                            env,
                            stage,
                            passes,
                            &mut cache,
                            &mut analyses,
                            round,
                            progress,
                        )?;
                        self.verify_after_stage(ctx, stage)?;
                        return Ok(());
                    }
                    self.verify_after_stage(ctx, stage)?;
                }
            }
        }
        Ok(())
    }

    fn run_stages(
        &self,
        range: std::ops::Range<usize>,
        ctx: &mut Context,
        env: &PipelineEnv,
        round: usize,
        progress: &mut impl FnMut(PipelineProgress),
    ) -> Result<(), String> {
        let mut dirty_functions = Some(HashSet::default());
        let mut cache = FixpointCache::default();
        let mut analyses = AnalysisManager::default();
        for stage in &self.stages[range] {
            match &stage.passes {
                StagePasses::Module(passes) => {
                    let outcome = run_module_stage(
                        ctx,
                        env,
                        stage,
                        passes,
                        &mut cache,
                        &mut analyses,
                        round,
                        progress,
                    )?;
                    dirty_functions = if outcome.module_changed {
                        None
                    } else {
                        Some(outcome.changed_functions)
                    };
                }
                StagePasses::Function(passes) => {
                    dirty_functions = Some(run_function_stage(
                        ctx,
                        env,
                        stage,
                        passes,
                        dirty_functions.as_ref(),
                        None,
                        &mut cache,
                        &mut analyses,
                        round,
                        progress,
                    )?);
                }
            }
            self.verify_after_stage(ctx, stage)?;
        }
        Ok(())
    }

    fn verify_after_stage(&self, ctx: &Context, stage: &Stage) -> Result<(), String> {
        if !self.debug {
            return Ok(());
        }
        let diagnostics = crate::verify(ctx);
        if diagnostics.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "verifier failed after stage \"{}\":\n{}",
                stage.name,
                diagnostics.join("\n")
            ))
        }
    }
}

fn resolve_function_passes(sc: &StageConfig) -> Result<Vec<Box<dyn DynFunctionPass>>, String> {
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

fn resolve_module_passes(sc: &StageConfig) -> Result<Vec<Box<dyn DynPass>>, String> {
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

/// Dump the QCode of a stage's `dump` functions to stderr before it runs. A
/// selector matches a function by exact name or by decimal/hex address; an
/// unmatched selector is reported rather than silently skipped.
fn dump_stage_inputs(ctx: &Context, dump: &[String], stage_name: &str) {
    for sel in dump {
        let addr = sel
            .strip_prefix("0x")
            .and_then(|h| u64::from_str_radix(h, 16).ok())
            .or_else(|| sel.parse::<u64>().ok());
        let target = ctx
            .functions()
            .find(|f| f.name() == sel.as_str() || (addr.is_some() && f.address() == addr))
            .map(|f| f.id);
        match target {
            Some(fid) => {
                eprintln!("==== dump before stage \"{stage_name}\": {sel} ====");
                eprint!("{}", FunctionRef::from_id(ctx, fid));
            }
            None => eprintln!("==== dump before stage \"{stage_name}\": {sel} (not found) ===="),
        }
    }
}

fn unknown_pass(name: &str, stage: &str) -> String {
    format!(
        "unknown pass \"{name}\" in stage \"{stage}\". Known passes: {}",
        known_pass_names()
    )
}

#[derive(Default)]
struct ModuleStageOutcome {
    changed_functions: HashSet<FunctionId>,
    module_changed: bool,
}

/// Run a whole-program stage; if `repeat_until` is set, loop the stage (OR-ing
/// the passes' change flags) until nothing changes or the iteration cap is hit.
#[allow(clippy::too_many_arguments)]
fn run_module_stage(
    ctx: &mut Context,
    env: &PipelineEnv,
    stage: &Stage,
    passes: &[Box<dyn DynPass>],
    cache: &mut FixpointCache,
    analyses: &mut AnalysisManager,
    round: usize,
    progress: &mut impl FnMut(PipelineProgress),
) -> Result<ModuleStageOutcome, String> {
    dump_stage_inputs(ctx, &stage.dump, &stage.name);
    let stage_name: std::sync::Arc<str> = stage.name.as_str().into();
    let mut round_targets = ctx.function_ids();
    let mut iters = 0;
    let mut stage_outcome = ModuleStageOutcome::default();
    let mut tracer = FixpointTracer::default();
    let tracer_label = format!("stage {} <module>", stage.name);
    loop {
        let watching = stage.repeat_until.is_some() && iters >= FIXPOINT_WATCH_ITERS;
        let mut next_set = HashSet::default();
        let mut round_changed = false;
        let mut round_module_changed = false;
        for p in passes {
            progress(PipelineProgress::WholeProgramPhase {
                round,
                stage: stage_name.clone(),
                pass: p.name(),
            });
            let _scope = qcode::pass_scope::enter(p.name());
            #[cfg(not(target_arch = "wasm32"))]
            let started = std::time::Instant::now();
            let mut type_request_retries = 0;
            let outcome = loop {
                let outcome = if let Some(inner) = p.as_module_fn() {
                    let eligible: Vec<_> = round_targets
                        .iter()
                        .copied()
                        .filter(|&id| {
                            let function = FunctionBody::from_id(ctx, id);
                            (stage.include_external || !function.is_external())
                                && !ctx.is_function_ignored(function.address())
                        })
                        .collect();
                    super::pass::ModulePassOutcome::functions(run_module_fn_adapter(
                        ctx,
                        env,
                        stage,
                        inner,
                        &eligible,
                        cache,
                        analyses,
                        round,
                        resolve_threads(),
                        progress,
                    )?)
                } else {
                    p.run_with_analyses(ctx, env, &round_targets, analyses)
                        .map_err(|e| format!("{}: {e}", p.name()))?
                };
                if outcome.type_requests.is_empty() {
                    break outcome;
                }
                if outcome.changed() {
                    return Err(format!(
                        "{}: a type-request outcome must not also mutate the module",
                        p.name()
                    ));
                }
                ctx.shared
                    .types
                    .create_requested_types(&outcome.type_requests);
                type_request_retries += 1;
                if type_request_retries >= MAX_FIXPOINT_ITERS {
                    return Err(format!(
                        "{}: type requests did not settle after {MAX_FIXPOINT_ITERS} retries",
                        p.name()
                    ));
                }
            };
            let pass_changed = outcome.changed();
            if pass_changed {
                if p.as_module_fn().is_none() {
                    invalidate_module_analyses(analyses, &outcome);
                    if outcome.module_changed {
                        for id in ctx.function_ids() {
                            cache.mark_dirty(id);
                        }
                    } else {
                        for id in outcome.changed_functions.iter().copied() {
                            cache.mark_dirty(id);
                        }
                    }
                }
                next_set.extend(outcome.changed_functions.iter().copied());
                stage_outcome
                    .changed_functions
                    .extend(outcome.changed_functions.iter().copied());
                round_module_changed |= outcome.module_changed;
                stage_outcome.module_changed |= outcome.module_changed;
            }
            #[cfg(not(target_arch = "wasm32"))]
            log::debug!(
                target: "pipeline",
                "{} ran in {:.2?} ({})",
                p.name(),
                started.elapsed(),
                if pass_changed { "changed" } else { "no change" },
            );
            // Between-pass invariant check (opt-in via `QCODE_VERIFY`): pin a broken
            // invariant to the pass that produced it, scoped to what it changed
            // (an unchanged function verified clean after the previous pass).
            crate::verify::verify_after(
                ctx,
                p.name(),
                if outcome.module_changed {
                    crate::verify::Scope::All
                } else {
                    crate::verify::Scope::Functions(&outcome.changed_functions)
                },
            );
            if watching && pass_changed {
                let fp = module_fingerprint(ctx);
                if tracer.observe(&tracer_label, iters + 1, p.name(), fp) {
                    // A proven cycle cannot converge; every further iteration
                    // re-treads the same states. Stop the stage best-effort at
                    // the recurring state instead of spinning to the cap.
                    log::warn!(
                        target: "pipeline::fixpoint",
                        "stage {} (module): stopping best-effort on the proven cycle at \
                         iteration {}",
                        stage.name,
                        iters + 1,
                    );
                    return Ok(stage_outcome);
                }
            }
            round_changed |= pass_changed;
        }
        iters += 1;
        if stage.repeat_until.is_none() || !round_changed {
            return Ok(stage_outcome);
        }
        if iters >= MAX_FIXPOINT_ITERS {
            log::warn!(
                target: "pipeline::fixpoint",
                "stage {} (module) hit the {MAX_FIXPOINT_ITERS}-iteration cap; see the \
                 `pipeline::fixpoint` trace above for the fighting passes",
                stage.name,
            );
            return Err(nonconvergence_error(&stage.name, None));
        }
        round_targets = if round_module_changed {
            ctx.function_ids()
        } else {
            ctx.function_ids()
                .into_iter()
                .filter(|id| next_set.contains(id))
                .collect()
        };
    }
}

/// Apply a module pass's preservation report at the scope named by its outcome.
/// Exact dirty functions lose only their own unpreserved local analyses; a
/// module-wide change conservatively applies to every local cache.
fn invalidate_module_analyses(
    analyses: &mut AnalysisManager,
    outcome: &super::pass::ModulePassOutcome,
) {
    analyses.invalidate_globals(&outcome.preserved_analyses);
    if outcome.module_changed {
        analyses.invalidate_all_locals(&outcome.preserved_analyses);
    } else {
        for function in outcome.changed_functions.iter().copied() {
            analyses.invalidate_local(function, &outcome.preserved_analyses);
        }
    }
}

/// Run one `module(<function-pass>)` adapter through the shared function-stage
/// execution engine. The pass runs once per function in this module iteration;
/// the containing module stage owns repetition and caller invalidation.
#[allow(clippy::too_many_arguments)]
fn run_module_fn_adapter(
    ctx: &mut Context,
    env: &PipelineEnv,
    stage: &Stage,
    pass: &dyn DynFunctionPass,
    fun_ids: &[FunctionId],
    cache: &mut FixpointCache,
    analyses: &mut AnalysisManager,
    round: usize,
    threads: usize,
    progress: &mut impl FnMut(PipelineProgress),
) -> Result<HashSet<FunctionId>, String> {
    let passes = [pass];
    run_function_worklist(
        ctx, env, stage, &passes, fun_ids, cache, analyses, round, threads, false, progress,
    )
}

/// Standalone [`DynPass::run`](super::pass::DynPass::run) bridge for a
/// `module(<function-pass>)` adapter. Normal pipeline execution reaches the same
/// worklist engine through [`run_module_stage`]; keeping this bridge here prevents
/// the adapter's object-safe fallback from growing a second serial execution
/// implementation.
pub(super) fn run_standalone_module_fn(
    ctx: &mut Context,
    env: &PipelineEnv,
    pass: &dyn DynFunctionPass,
    targets: &[FunctionId],
    analyses: &mut AnalysisManager,
) -> Result<super::pass::ModulePassOutcome, String> {
    let stage = Stage {
        name: format!("module({})", pass.name()),
        passes: StagePasses::Function(Vec::new()),
        repeat_until: None,
        include_external: false,
        only_dirty: false,
        dump: Vec::new(),
    };
    let target_set: HashSet<_> = targets.iter().copied().collect();
    let fun_ids: Vec<_> = ctx
        .functions()
        .filter(|f| !f.is_external() && !ctx.is_function_ignored(f.address()))
        .map(|f| f.id)
        .filter(|id| target_set.contains(id))
        .collect();
    let mut cache = FixpointCache::default();
    let changed = run_function_worklist(
        ctx,
        env,
        &stage,
        &[pass],
        &fun_ids,
        &mut cache,
        analyses,
        0,
        resolve_threads(),
        false,
        &mut |_| {},
    )?;
    Ok(super::pass::ModulePassOutcome::functions(changed))
}

/// Run a clean-IR whole-program stage during recursive lifting. Most passes are
/// ordinary analysis passes; the two lifting pass names are TOML-visible
/// adapters over the caller-provided lifter service.
#[allow(clippy::too_many_arguments)]
fn run_lifting_module_stage(
    ctx: &mut Context,
    env: &PipelineEnv,
    services: &mut PipelineServices<'_>,
    stage: &Stage,
    passes: &[Box<dyn DynPass>],
    cache: &mut FixpointCache,
    analyses: &mut AnalysisManager,
    round: usize,
    progress: &mut impl FnMut(PipelineProgress),
) -> Result<ModuleStageOutcome, String> {
    dump_stage_inputs(ctx, &stage.dump, &stage.name);
    let stage_name: std::sync::Arc<str> = stage.name.as_str().into();
    // `lift_new_addresses` drains a finite discovery queue and may enqueue the
    // directly reachable successors it just decoded.  Do not apply the generic
    // optimization fixpoint cap to that queue drain: stopping every 100 batches
    // forces the outer discovery driver to run expensive address analysis over
    // partially lifted functions before continuing the already-known work.
    // Other repeatable lifting-phase stages retain the non-convergence guard.
    let drains_discoveries = passes.iter().any(|p| p.name() == "lift_new_addresses");
    let mut iters = 0;
    let mut stage_outcome = ModuleStageOutcome::default();
    let mut round_targets = ctx.function_ids();
    loop {
        let mut round_changed = false;
        let mut next_set = HashSet::default();
        let mut round_module_changed = false;
        for p in passes {
            progress(PipelineProgress::WholeProgramPhase {
                round,
                stage: stage_name.clone(),
                pass: p.name(),
            });
            let _scope = qcode::pass_scope::enter(p.name());
            #[cfg(not(target_arch = "wasm32"))]
            let started = std::time::Instant::now();
            let outcome = match p.name() {
                "discover_addresses_in_binary" => {
                    let changed = crate::discover_addresses_in_binary(ctx, services, analyses)
                        .map_err(|e| format!("{}: {e}", p.name()))?
                        .changed();
                    super::pass::ModulePassOutcome::module_if(changed)
                        .preserving_global::<crate::AddressAnalysis>()
                }
                "lift_new_addresses" => {
                    let summary = crate::lift_new_addresses(ctx, services, analyses)
                        .map_err(|e| format!("{}: {e}", p.name()))?;
                    super::pass::ModulePassOutcome::module_if(summary.changed())
                        .preserving_global::<crate::AddressAnalysis>()
                }
                _ => p
                    .run_with_analyses(ctx, env, &round_targets, analyses)
                    .map_err(|e| format!("{}: {e}", p.name()))?,
            };
            let pass_changed = outcome.changed();
            if pass_changed {
                invalidate_module_analyses(analyses, &outcome);
                if outcome.module_changed {
                    for id in ctx.function_ids() {
                        cache.mark_dirty(id);
                    }
                } else {
                    for id in outcome.changed_functions.iter().copied() {
                        cache.mark_dirty(id);
                    }
                }
                next_set.extend(outcome.changed_functions.iter().copied());
                stage_outcome
                    .changed_functions
                    .extend(outcome.changed_functions.iter().copied());
                round_module_changed |= outcome.module_changed;
                stage_outcome.module_changed |= outcome.module_changed;
            }
            #[cfg(not(target_arch = "wasm32"))]
            log::debug!(
                target: "pipeline",
                "{} ran in {:.2?} ({})",
                p.name(),
                started.elapsed(),
                if pass_changed { "changed" } else { "no change" },
            );
            round_changed |= pass_changed;
        }
        iters += 1;
        if stage.repeat_until.is_none() || !round_changed {
            return Ok(stage_outcome);
        }
        if !drains_discoveries && iters >= MAX_FIXPOINT_ITERS {
            return Err(nonconvergence_error(&stage.name, None));
        }
        round_targets = if round_module_changed {
            ctx.function_ids()
        } else {
            ctx.function_ids()
                .into_iter()
                .filter(|id| next_set.contains(id))
                .collect()
        };
    }
}

/// Per-`(function, pass)` fixpoint memo for a single pipeline run.
///
/// Once a function-scoped pass returns "no change" on a function, it has reached a
/// fixpoint there and is skipped on that function until some *other* pass modifies
/// the function. The default pipeline runs `gvn`/`dead_store`/`dce` in several
/// separate stages; without this, each stage re-runs every pass on every function
/// even when nothing has touched the function since — the bulk of the per-function
/// pass time.
///
/// Scope is one pipeline run (`Pipeline::run` builds a fresh cache). The
/// checkpoint+replay driver re-clones clean IR each round, so the cache naturally
/// resets between rounds — preserving the invariant that no pass ever sees its own
/// output across rounds.
#[derive(Default)]
struct FixpointCache {
    /// Bumped whenever a pass reports it modified a function; a clean-mark recorded
    /// at an older generation no longer counts as clean.
    generation: HashMap<FunctionId, u64>,
    /// `(function, pass)` → the function generation at which the pass last reported
    /// no change.
    clean: HashMap<(FunctionId, &'static str), u64>,
}

impl FixpointCache {
    fn generation_of(&self, f: FunctionId) -> u64 {
        self.generation.get(&f).copied().unwrap_or(0)
    }

    /// True if `pass` already reached a fixpoint on `f` and `f` is unchanged since.
    fn is_clean(&self, f: FunctionId, pass: &'static str) -> bool {
        self.clean.get(&(f, pass)).copied() == Some(self.generation_of(f))
    }

    /// Record that `pass` reached a fixpoint on `f` at its current generation.
    fn mark_clean(&mut self, f: FunctionId, pass: &'static str) {
        let g = self.generation_of(f);
        self.clean.insert((f, pass), g);
    }

    /// Record that a pass modified `f`, invalidating every pass's clean-mark for it
    /// (their recorded generation no longer matches).
    fn mark_dirty(&mut self, f: FunctionId) {
        *self.generation.entry(f).or_insert(0) += 1;
    }

    /// A private cache holding only `funcs`' entries, for a parallel worker to
    /// mutate in isolation. Workers own disjoint function sets, so their local
    /// caches touch disjoint keys and merge back conflict-free via
    /// [`absorb`](Self::absorb).
    fn extract(&self, funcs: &HashSet<FunctionId>) -> FixpointCache {
        FixpointCache {
            generation: self
                .generation
                .iter()
                .filter(|(f, _)| funcs.contains(f))
                .map(|(f, g)| (*f, *g))
                .collect(),
            clean: self
                .clean
                .iter()
                .filter(|((f, _), _)| funcs.contains(f))
                .map(|(k, v)| (*k, *v))
                .collect(),
        }
    }

    /// Fold a worker's local cache back in. Clean marks are only ever added or
    /// bumped (never removed) during a stage, and each worker owns a disjoint
    /// function set, so overwriting keys with the worker's final values is exact.
    fn absorb(&mut self, other: FixpointCache) {
        self.generation.extend(other.generation);
        self.clean.extend(other.clean);
    }
}

/// A fingerprint of one function's rendered body, used to detect whether a
/// module stage modified it. Rendering the IR captures operand rewrites,
/// insertions/removals, retypes, and CFG edits; block order is address-sorted
/// (deterministic) so an unchanged function fingerprints identically across a
/// stage. Arena ids embedded in the render are normalized away first, so a body
/// that churns its temporaries without changing structure fingerprints
/// identically (see [`fingerprint_display`]).
pub(super) fn function_fingerprint(ctx: &Context, fun_id: FunctionId) -> u64 {
    fingerprint_display(FunctionRef::from_id(ctx, fun_id))
}

/// Cheap structural fingerprint of a function's clean IR, for the discovery loop's
/// per-round `restrict` set (which functions changed since last round).
///
/// Hashes the arena content directly — block ids, addresses, params, and each
/// instruction's id/opcode/operands/type/address, plus edges — instead of
/// rendering the whole body to a `String` and scanning it like
/// [`function_fingerprint`]. That render is the dominant per-round cost on large
/// binaries (it runs for every function every round).
///
/// It deliberately does **not** normalize arena ids. Normalization exists so a
/// body that churns its temporaries (mem2reg/gvn) fingerprints stably for cycle
/// detection — but between discovery rounds `clean` is only *grown* by the lifting
/// phase, never optimized, so an untouched function keeps byte-identical ids and
/// hashes the same, while any lifted or split function changes ids (or content)
/// and hashes differently. Raw ids are therefore both sufficient and cheaper here.
pub(super) fn cheap_function_fingerprint(ctx: &Context, fun_id: FunctionId) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut h = std::collections::hash_map::DefaultHasher::new();
    let body = FunctionBody::from_id(ctx, fun_id);
    for block_id in body.block_ids() {
        block_id.hash(&mut h);
        let block = qcode::value::BasicBlock::from_id(ctx, block_id);
        block.address().hash(&mut h);
        for param in block.params() {
            param.id().hash(&mut h);
            param.type_id().hash(&mut h);
        }
        for insn in block.instructions() {
            insn.id().hash(&mut h);
            insn.mnemonic().opcode().hash(&mut h);
            for arg in insn.mnemonic().args() {
                arg.hash(&mut h);
            }
            insn.type_id().hash(&mut h);
            insn.address().hash(&mut h);
        }
        for (_, succ) in block.successors() {
            succ.hash(&mut h);
        }
    }
    h.finish()
}

/// Hash a renderable value (a `FunctionRef` over *any* host), normalizing the
/// arena ids that the render embeds so the result is *alpha-equivalent*: two
/// structurally identical bodies fingerprint the same even if every temporary was
/// deleted and re-created in between.
///
/// This normalization is the whole point. An unnamed value renders as
/// `%tmp{local_id:x}` (and an unnamed block param as `@param{local_id:x}`), so the
/// rendered text carries raw arena indices. Those indices are monotonic and never
/// reused, so a pass that rewrites an instruction shifts them permanently: a
/// fixpoint oscillating between two structurally identical states produced a
/// *different* fingerprint every single iteration, and [`FixpointTracer`] — which
/// only trips on a repeated fingerprint — could never fire. Measured on `test21`:
/// `mem2reg` and `gvn` traded 252 moves on one function with 252 distinct
/// fingerprints and zero detected cycles, spinning to the iteration cap and
/// leaving the body unsimplified.
///
/// Each distinct id-bearing atom is therefore replaced by its first-appearance
/// ordinal before hashing, which preserves structure (two atoms are the same iff
/// they were the same before) while discarding absolute ids. Named values are left
/// alone: their names are already churn-stable and carry real meaning.
///
/// Used to fingerprint a function whether it is live in the module
/// ([`function_fingerprint`]) or checked out of it (the per-function fixpoint
/// tracer, which renders the body through its `BodyMut` host). Unlike the previous
/// streaming implementation this materializes the render, which is affordable
/// because fingerprinting only runs once a stage is already being traced for
/// non-convergence (`iters >= FIXPOINT_WATCH_ITERS`), never on the common path.
pub(super) fn fingerprint_display(d: impl std::fmt::Display) -> u64 {
    use std::hash::Hasher;

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hash_id_normalized(&mut hasher, &d.to_string());
    hasher.finish()
}

/// Atom prefixes whose rendered suffix is a raw body-local arena index
/// (`instruction_atom` / `block_param_atom` in `qcode::value::insn::segment`).
const ID_BEARING_ATOMS: [&str; 2] = ["%tmp", "@param"];

/// Feed `rendered` to `hasher`, replacing each distinct id-bearing atom with its
/// first-appearance ordinal. See [`fingerprint_display`].
fn hash_id_normalized(hasher: &mut impl std::hash::Hasher, rendered: &str) {
    let mut ordinals: HashMap<&str, u32> = HashMap::default();
    let mut rest = rendered;

    while let Some((at, prefix)) = ID_BEARING_ATOMS
        .iter()
        .filter_map(|p| rest.find(p).map(|i| (i, *p)))
        .min_by_key(|&(i, _)| i)
    {
        let after = &rest[at + prefix.len()..];
        let digits = after
            .find(|c: char| !c.is_ascii_hexdigit())
            .unwrap_or(after.len());

        // Only a hex run that ends the identifier is an arena index. A bare prefix,
        // or one running into further identifier characters, belongs to a *named*
        // value that merely starts with these letters (`%tmpfoo`) — pass it through
        // rather than risk conflating two distinct names into one ordinal, which
        // would fabricate a cycle that is not there.
        let ends_identifier = after[digits..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_');
        if digits == 0 || !ends_identifier {
            hasher.write(&rest.as_bytes()[..at + prefix.len()]);
            rest = after;
            continue;
        }

        hasher.write(&rest.as_bytes()[..at]);
        let next = ordinals.len() as u32;
        let ordinal = *ordinals
            .entry(&rest[at..at + prefix.len() + digits])
            .or_insert(next);
        hasher.write(prefix.as_bytes());
        hasher.write(&ordinal.to_le_bytes());
        rest = &after[digits..];
    }

    hasher.write(rest.as_bytes());
}

/// Whole-program structural fingerprint: the address-sorted list of per-function
/// [`function_fingerprint`]s. Stable across a stage that changes nothing, and — like
/// its per-function basis — id-churn tolerant, so a module stage that oscillates
/// back to a prior whole-program state fingerprints identically. That tolerance
/// comes from the id normalization in [`fingerprint_display`]: the rendered IR
/// *does* embed arena indices (`%tmp1f3`), so hashing it verbatim is not
/// churn-tolerant, which is what previously blinded the cycle detector. Used only
/// while a stage is being traced for non-convergence, so the O(program) render cost
/// is off the common path.
fn module_fingerprint(ctx: &Context) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut per_fn: Vec<(Option<u64>, u64)> = ctx
        .functions()
        .map(|f| (f.address(), function_fingerprint(ctx, f.id)))
        .collect();
    per_fn.sort_unstable();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    per_fn.hash(&mut hasher);
    hasher.finish()
}

/// Diagnoses why a `repeat_until` stage will not converge. Engaged only once a
/// stage passes [`FIXPOINT_WATCH_ITERS`], it fingerprints the IR after each pass
/// that reports a change and:
///   * logs the transition (`pass -> new fingerprint`) at `debug` on the
///     `pipeline::fixpoint` target — the play-by-play of which passes keep moving
///     the IR (i.e. who is "fighting"); enable with
///     `RUST_LOG=pipeline::fixpoint=debug`;
///   * when a fingerprint *recurs*, warns loudly with the exact loop body — proof
///     the stage is cycling between states rather than making forward progress,
///     which no amount of further iteration can fix. A recurring fingerprint means
///     two or more passes are undoing each other. [`observe`](Self::observe)
///     returns `true` on recurrence so the driver stops the fixpoint best-effort
///     right there instead of spinning to the iteration cap.
///
/// Detection is best-effort: it catches an exact return to a prior rendered state.
/// A stage that churns fresh names every iteration (never rendering identically)
/// shows as an endless transition log with no `CYCLE` line — still diagnostic, just
/// via the per-pass trail rather than a single verdict.
#[derive(Default)]
struct FixpointTracer {
    /// fingerprint -> (iteration, pass) that first produced it.
    seen: HashMap<u64, (usize, &'static str)>,
    /// Ordered (iteration, pass, fingerprint) trail since watching began, used to
    /// reconstruct the loop body when a fingerprint recurs.
    trail: Vec<(usize, &'static str, u64)>,
    /// Set once a recurring fingerprint has been reported, so the same cycle is not
    /// re-logged on every subsequent iteration.
    cycle_reported: bool,
}

impl FixpointTracer {
    /// Record that `pass` produced fingerprint `fp` at 1-based `iter`. `label` names
    /// the stage and function (or `<module>`). Logs the transition and, on the first
    /// recurrence, the detected cycle.
    ///
    /// Returns `true` when `fp` recurred — proof the fixpoint is cycling between
    /// states rather than converging, so the caller should stop iterating
    /// best-effort instead of spinning to the iteration cap.
    fn observe(&mut self, label: &str, iter: usize, pass: &'static str, fp: u64) -> bool {
        log::debug!(
            target: "pipeline::fixpoint",
            "{label} iter {iter}: {pass} moved IR -> fingerprint {fp:#018x}",
        );
        self.trail.push((iter, pass, fp));
        if let Some(&(first_iter, first_pass)) = self.seen.get(&fp) {
            if !self.cycle_reported {
                self.cycle_reported = true;
                let body: Vec<String> = self
                    .trail
                    .iter()
                    .skip_while(|&&(i, _, h)| !(i == first_iter && h == fp))
                    .map(|&(i, p, h)| format!("    iter {i}: {p} -> {h:#018x}"))
                    .collect();
                log::warn!(
                    target: "pipeline::fixpoint",
                    "{label}: CYCLE DETECTED — fingerprint {fp:#018x} first produced by \
                     `{first_pass}` at iter {first_iter}, reproduced by `{pass}` at iter \
                     {iter}. The stage is oscillating, not converging; the loop body is:\n{}",
                    body.join("\n"),
                );
            }
            true
        } else {
            self.seen.insert(fp, (iter, pass));
            false
        }
    }
}

/// Run a function-scoped stage function-major: for each non-external function,
/// run the stage's passes; if `repeat_until` is set, loop that function's passes
/// to a fixpoint before moving to the next function.
#[allow(clippy::too_many_arguments)]
fn run_function_stage(
    ctx: &mut Context,
    env: &PipelineEnv,
    stage: &Stage,
    passes: &[Box<dyn DynFunctionPass>],
    previous_dirty: Option<&HashSet<FunctionId>>,
    // Cross-round skip: when `Some`, only these functions are processed. The address
    // discovery loop passes the set of functions whose clean IR changed since the
    // previous round — every other function would re-derive the same (already-drained)
    // discoveries, so re-optimizing it is pure waste. `None` processes all functions.
    restrict: Option<&HashSet<FunctionId>>,
    cache: &mut FixpointCache,
    analyses: &mut AnalysisManager,
    round: usize,
    progress: &mut impl FnMut(PipelineProgress),
) -> Result<HashSet<FunctionId>, String> {
    run_function_stage_with_threads(
        ctx,
        env,
        stage,
        passes,
        previous_dirty,
        restrict,
        cache,
        analyses,
        round,
        progress,
        resolve_threads(),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_function_stage_with_threads(
    ctx: &mut Context,
    env: &PipelineEnv,
    stage: &Stage,
    passes: &[Box<dyn DynFunctionPass>],
    previous_dirty: Option<&HashSet<FunctionId>>,
    restrict: Option<&HashSet<FunctionId>>,
    cache: &mut FixpointCache,
    analyses: &mut AnalysisManager,
    round: usize,
    progress: &mut impl FnMut(PipelineProgress),
    threads: usize,
) -> Result<HashSet<FunctionId>, String> {
    // Every producer of reattributed blocks (the recursive lifter, and
    // `split_overlapping_functions` during discovery rounds) discharges strict
    // locality at its own tail, so every function reaching this stage is already
    // self-stored — the invariant `BodyMut::new` asserts at construction. No
    // storage normalization is needed here.
    dump_stage_inputs(ctx, &stage.dump, &stage.name);
    let fun_ids: Vec<FunctionId> = ctx
        .functions()
        .filter(|f| stage.include_external || !f.is_external())
        // Honor `--ignore`: never run a per-function pass on a function the user
        // marked ignored. It stays lifted, just unoptimized.
        .filter(|f| !ctx.is_function_ignored(f.address()))
        .map(|f| f.id)
        .filter(|id| !stage.only_dirty || previous_dirty.is_none_or(|dirty| dirty.contains(id)))
        .filter(|id| restrict.is_none_or(|r| r.contains(id)))
        .collect();
    let pass_refs: Vec<&dyn DynFunctionPass> = passes.iter().map(Box::as_ref).collect();
    run_function_worklist(
        ctx,
        env,
        stage,
        &pass_refs,
        &fun_ids,
        cache,
        analyses,
        round,
        threads,
        stage.repeat_until.is_some(),
        progress,
    )
}

/// Shared sequential/parallel executor for ordinary function stages and
/// `module(<function-pass>)` adapters.
#[allow(clippy::too_many_arguments)]
fn run_function_worklist(
    ctx: &mut Context,
    env: &PipelineEnv,
    stage: &Stage,
    passes: &[&dyn DynFunctionPass],
    fun_ids: &[FunctionId],
    cache: &mut FixpointCache,
    analyses: &mut AnalysisManager,
    round: usize,
    threads: usize,
    repeat_until: bool,
    progress: &mut impl FnMut(PipelineProgress),
) -> Result<HashSet<FunctionId>, String> {
    let total = fun_ids.len();
    let stage_name: std::sync::Arc<str> = stage.name.as_str().into();

    // Function-major stages run each pass thousands of times, so timing is
    // aggregated per pass over the whole stage rather than logged per call.
    let mut elapsed: HashMap<&'static str, (std::time::Duration, usize, usize)> =
        HashMap::default();
    let mut dirty = HashSet::default();
    let mut pending = fun_ids.to_vec();
    let mut type_request_rounds = 0;

    // Every function pass is driven over a body borrowed in place from the bodies
    // registry (`run_one_function`) — the shape Stage 6 runs on worker threads.

    while !pending.is_empty() {
        let fun_ids = pending.as_slice();
        let mut retry = Vec::new();

        // Parallelize the stage across threads once the worklist is worth the fan-out
        // cost. Strict IR locality (context-split ruling 2) is established at the
        // optimization entry and every discovery round, so every function body is
        // closed over its own blocks — there are no cross-function edges to entangle
        // two checked-out bodies, and every function is parallel-eligible. Tiny
        // worklists, `QCODE_THREADS=1`, and wasm fall through to the sequential loop
        // below, byte-for-byte identical.
        let parallel_set: HashSet<FunctionId> =
            if threads > 1 && fun_ids.len() >= PARALLEL_THRESHOLD {
                run_stage_parallel(
                    ctx,
                    env,
                    stage,
                    passes,
                    fun_ids,
                    &stage_name,
                    total,
                    round,
                    cache,
                    analyses,
                    &mut elapsed,
                    &mut dirty,
                    threads,
                    repeat_until,
                    progress,
                    &mut retry,
                )?;
                fun_ids.iter().copied().collect()
            } else {
                HashSet::default()
            };

        // The sequential lane: every function when the stage did not parallelize;
        // nothing when it did (all were handled in parallel above).
        {
            for (index, fun_id) in fun_ids.iter().copied().enumerate() {
                if parallel_set.contains(&fun_id) {
                    continue;
                }
                let function: std::sync::Arc<str> = FunctionRef::from_id(ctx, fun_id).name().into();
                let mut local_analyses = analyses.take_local(fun_id);
                // Split the context: borrow this function's body `&mut` in place from
                // the bodies registry and run its whole pass fixpoint on it over the
                // frozen module view; then drop the split borrow and do the barrier work
                // (install minted callees and replay the buffered rename).
                let outcome = {
                    let (bodies, view) = ctx.split(env);
                    run_one_function(
                        passes,
                        &mut bodies[fun_id],
                        view,
                        cache,
                        &mut local_analyses,
                        &mut elapsed,
                        &stage.name,
                        &function,
                        repeat_until,
                        |pass| {
                            progress(PipelineProgress::FunctionPass {
                                round,
                                stage: stage_name.clone(),
                                function: function.clone(),
                                index: index + 1,
                                total,
                                pass,
                            });
                        },
                    )?
                };
                analyses.put_local(fun_id, local_analyses);
                if !outcome.type_requests.is_empty() {
                    ctx.shared
                        .types
                        .create_requested_types(&outcome.type_requests);
                    retry.push(fun_id);
                }
                let installed = install_minted(ctx, &stage.name, outcome.minted)?;
                let patched = resolve_minted_callees(ctx, &stage.name, fun_id, &installed)?;
                replay_rename(ctx, &stage.name, fun_id, outcome.rename)?;
                // Minted functions are new work for downstream `only_dirty` stages.
                let verify_scope: HashSet<FunctionId> =
                    installed.iter().copied().chain([fun_id]).collect();
                dirty.extend(installed);
                // Opt-in `QCODE_VERIFY` check once the split borrow has ended — the body
                // is reachable through `ctx` again — pinning any invariant break to this
                // stage, scoped to the one function it ran on (plus its mints).
                // A no-op unless `QCODE_VERIFY` is set.
                crate::verify::verify_after(
                    ctx,
                    &stage.name,
                    crate::verify::Scope::Functions(&verify_scope),
                );
                if outcome.changed || patched {
                    analyses.invalidate_globals(&outcome.preserved_analyses);
                    dirty.insert(fun_id);
                }
            }
        }

        if retry.is_empty() {
            break;
        }
        type_request_rounds += 1;
        if type_request_rounds >= MAX_FIXPOINT_ITERS {
            return Err(format!(
                "stage {}: type requests did not settle after {MAX_FIXPOINT_ITERS} barrier rounds",
                stage.name
            ));
        }
        pending = retry;
    }

    if log::log_enabled!(target: "pipeline", log::Level::Debug) {
        let mut rows: Vec<_> = elapsed.into_iter().collect();
        rows.sort_by_key(|b| std::cmp::Reverse(b.1.0));
        for (pass, (time, runs, changes)) in rows {
            log::debug!(
                target: "pipeline",
                "stage {}: {pass} took {time:.2?} over {runs} runs ({changes} changed)",
                stage.name,
            );
        }
    }
    Ok(dirty)
}

/// Run one function's per-stage pass fixpoint over a body the driver has already
/// checked out. Reads the module through `m`, mutates only
/// `body`, updates the per-function fixpoint `cache` and the per-pass `elapsed`
/// table, and calls `on_pass(name)` before each pass (the caller emits / forwards
/// the progress event). Returns whether the function changed.
///
/// This is the reusable unit Stage 6's parallel driver runs on worker threads over
/// disjoint `&mut FunctionBody`s; the sequential driver calls it one function at a
/// time. It touches no global mutable state — the driver replays its buffered
/// effects at the barrier after the run. The `MAX_FIXPOINT_ITERS`
/// non-convergence guard and the per-function `FixpointTracer` cycle detection are
/// preserved exactly as on the in-place path.
#[allow(clippy::too_many_arguments)]
fn run_one_function<'str>(
    passes: &[&dyn DynFunctionPass],
    body: &mut FunctionBody<'str>,
    cx: ContextView<'_, 'str>,
    cache: &mut FixpointCache,
    analyses: &mut LocalAnalysisManager,
    elapsed: &mut HashMap<&'static str, (std::time::Duration, usize, usize)>,
    stage_name: &str,
    function_name: &str,
    repeat_until: bool,
    mut on_pass: impl FnMut(&'static str),
) -> Result<Outcome<'str>, String> {
    let fun_id = body.id();
    let mut next_minted = 0;
    let mut iters = 0;
    let mut function_changed = false;
    // Aggregate the per-pass outcomes across the fixpoint: `rename` last-writer-wins
    // (matching the old `Effects` overwrite), `minted` concatenated. `changed` is
    // tracked as `function_changed` and folded in at return.
    let mut agg_rename: Option<std::borrow::Cow<'str, str>> = None;
    let mut agg_minted: Vec<super::Minted<'str>> = Vec::new();
    let mut agg_type_requests: Vec<qcode::types::TypeRequest> = Vec::new();
    let mut preserved_analyses = PreservedAnalyses::all();
    let mut tracer = FixpointTracer::default();
    let tracer_label = format!("stage {stage_name} fn {function_name}");
    'fixpoint: loop {
        let watching = repeat_until && iters >= FIXPOINT_WATCH_ITERS;
        let mut changed = false;
        for p in passes {
            if cache.is_clean(fun_id, p.name()) {
                continue;
            }
            on_pass(p.name());
            let _scope = qcode::pass_scope::enter(p.name());
            #[cfg(not(target_arch = "wasm32"))]
            let started = std::time::Instant::now();
            let outcome = p
                .run_checked(body, cx, &mut next_minted, analyses)
                .map_err(|e| format!("{}: {e}", p.name()))?;
            if !outcome.type_requests.is_empty() {
                if outcome.changed || outcome.rename.is_some() || !outcome.minted.is_empty() {
                    return Err(format!(
                        "{}: a type-request outcome must not also mutate IR, rename, or mint functions",
                        p.name()
                    ));
                }
                agg_type_requests.extend(outcome.type_requests);
                break 'fixpoint;
            }
            // Minting is an observable stage mutation even when a pass forgot to
            // set its body-change bit. Keep the producer dirty and continue a
            // requested fixpoint rather than caching it as clean while publishing
            // new functions at the barrier.
            let pass_changed = outcome.changed || !outcome.minted.is_empty();
            let pass_preserved = outcome.preserved_analyses.clone();
            if outcome.rename.is_some() {
                agg_rename = outcome.rename;
            }
            agg_minted.extend(outcome.minted);
            if pass_changed {
                analyses.invalidate(&pass_preserved);
                preserved_analyses.intersect(&pass_preserved);
                cache.mark_dirty(fun_id);
            } else {
                cache.mark_clean(fun_id, p.name());
            }
            let entry = elapsed.entry(p.name()).or_default();
            #[cfg(not(target_arch = "wasm32"))]
            {
                entry.0 += started.elapsed();
            }
            entry.1 += 1;
            entry.2 += pass_changed as usize;
            if watching && pass_changed {
                // Fingerprint the checked-out body through its own host (it is
                // absent from `ctx`, so `function_fingerprint` cannot see it).
                let fp =
                    fingerprint_display(qcode::value::FunctionRef::new(cx.body_view(body), fun_id));
                if tracer.observe(&tracer_label, iters + 1, p.name(), fp) {
                    log::warn!(
                        target: "pipeline::fixpoint",
                        "stage {stage_name} fn {function_name}: stopping best-effort on the \
                         proven cycle at iteration {}",
                        iters + 1,
                    );
                    function_changed = true;
                    break 'fixpoint;
                }
            }
            changed |= pass_changed;
        }
        function_changed |= changed;
        iters += 1;
        if !repeat_until || !changed {
            break;
        }
        if iters >= MAX_FIXPOINT_ITERS {
            log::warn!(
                target: "pipeline::fixpoint",
                "stage {stage_name} fn {function_name} hit the {MAX_FIXPOINT_ITERS}-iteration \
                 cap; see the `pipeline::fixpoint` trace above for the fighting passes",
            );
            return Err(nonconvergence_error(stage_name, Some(function_name)));
        }
    }
    Ok(Outcome {
        changed: function_changed,
        rename: agg_rename,
        minted: agg_minted,
        type_requests: agg_type_requests,
        preserved_analyses,
    })
}

/// How many worker threads a stage may use. `QCODE_THREADS` overrides the
/// default (`available_parallelism`); `QCODE_THREADS=1` forces the sequential path.
/// wasm has no threads, so it is always `1`.
fn resolve_threads() -> usize {
    #[cfg(target_arch = "wasm32")]
    {
        1
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Ok(v) = std::env::var("QCODE_THREADS")
            && let Ok(n) = v.parse::<usize>()
        {
            return n.max(1);
        }
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
}

/// Per-function driver metadata snapshotted before the split borrow: the id and
/// display name.
type FnMeta = (FunctionId, std::sync::Arc<str>);

/// One worklist function on its way through a parallel stage: its identity, the
/// body a worker mutates (borrowed `&mut` in place from the bodies registry), and
/// the [`Outcome`] the worker produced (changed / rename / minted).
struct ParallelEntry<'a, 'str> {
    index: usize,
    fun_id: FunctionId,
    name: std::sync::Arc<str>,
    body: &'a mut FunctionBody<'str>,
    outcome: Outcome<'str>,
    analyses: LocalAnalysisManager,
}

/// What one worker thread accumulates locally and hands back for deterministic
/// merge on the master thread. The mutated bodies travel back through the shared
/// `&mut [ParallelEntry]` slice, not here.
struct WorkerOutput {
    cache: FixpointCache,
    elapsed: HashMap<&'static str, (std::time::Duration, usize, usize)>,
    stats: Vec<((&'static str, &'static str), u64)>,
}

/// Run a stage across `threads` worker threads (Stage 6 of the
/// parallel-passes plan). Checks out the whole worklist, runs each function's pass
/// fixpoint on a disjoint `&mut FunctionBody` on a `std::thread::scope` worker over
/// the `&`-shared [`ContextView`], then runs the barrier **in worklist
/// order**. Output is byte-identical to the sequential path: the frozen module
/// view, deterministic (contiguous, worklist-ordered) work assignment, and the
/// worklist-ordered barrier leave nothing to run-to-run chance.
#[allow(clippy::too_many_arguments)]
fn run_stage_parallel(
    ctx: &mut Context,
    env: &PipelineEnv,
    stage: &Stage,
    passes: &[&dyn DynFunctionPass],
    fun_ids: &[FunctionId],
    stage_name: &std::sync::Arc<str>,
    total: usize,
    round: usize,
    cache: &mut FixpointCache,
    analyses: &mut AnalysisManager,
    elapsed: &mut HashMap<&'static str, (std::time::Duration, usize, usize)>,
    dirty: &mut HashSet<FunctionId>,
    threads: usize,
    repeat_until: bool,
    progress: &mut impl FnMut(PipelineProgress),
    retry: &mut Vec<FunctionId>,
) -> Result<(), String> {
    // 1. Snapshot per-function display names before the split borrow freezes the
    //    context.
    let mut metas: Vec<FnMeta> = Vec::with_capacity(fun_ids.len());
    for &fun_id in fun_ids {
        let name: std::sync::Arc<str> = FunctionRef::from_id(ctx, fun_id).name().into();
        metas.push((fun_id, name));
    }
    let local_analyses: Vec<_> = fun_ids
        .iter()
        .copied()
        .map(|fun_id| analyses.take_local(fun_id))
        .collect();

    let chunk_size = fun_ids.len().div_ceil(threads).max(1);
    let stage_label = stage.name.clone();

    // 2-5. Under the split borrow: borrow every worklist body `&mut` in place
    //    (disjoint, worklist order), run the workers over the frozen view, merge
    //    their local state, then drain each body into an owned per-function
    //    outcome. The split borrow ends with this block, releasing `ctx` for the
    //    barrier.
    let results = {
        let (bodies, view) = ctx.split(env);
        // Disjoint `&mut` borrows of every worklist body, in worklist order.
        let slots = bodies.select_mut(fun_ids);
        let mut entries: Vec<ParallelEntry> = metas
            .into_iter()
            .zip(slots)
            .zip(local_analyses)
            .enumerate()
            .map(
                |(index, (((fun_id, name), slot), analyses))| ParallelEntry {
                    index,
                    fun_id,
                    name,
                    body: slot,
                    outcome: Outcome::default(),
                    analyses,
                },
            )
            .collect();

        // Contiguous, worklist-ordered chunks — deterministic assignment. Run
        // workers on one shared read-only view; forward progress over a channel the
        // master thread pumps live while the workers compute.
        let (tx, rx) = std::sync::mpsc::channel::<PipelineProgress>();
        let worker_outputs: Vec<Result<WorkerOutput, String>> = {
            let stage_label = &stage_label;
            std::thread::scope(|scope| {
                let mut handles = Vec::new();
                for chunk in entries.chunks_mut(chunk_size) {
                    let funcs: HashSet<FunctionId> = chunk.iter().map(|e| e.fun_id).collect();
                    let mut local_cache = cache.extract(&funcs);
                    let tx = tx.clone();
                    let stage_name = stage_name.clone();
                    // Some analysis passes (notably mem2reg's `decide_values_start_from`
                    // renamer) recurse along the CFG DFS, so their native-stack depth
                    // scales with the function's longest block chain. Obfuscated inputs
                    // produce functions deep enough to overflow the default ~2MB worker
                    // stack, so give each worker a generous stack.
                    let handle = std::thread::Builder::new()
                        .name("qcode-pass-worker".into())
                        .stack_size(256 * 1024 * 1024)
                        .spawn_scoped(scope, move || -> Result<WorkerOutput, String> {
                            let mut local_elapsed: HashMap<
                                &'static str,
                                (std::time::Duration, usize, usize),
                            > = HashMap::default();
                            for e in chunk.iter_mut() {
                                let index = e.index;
                                let name = e.name.clone();
                                let stage_name = stage_name.clone();
                                let tx = &tx;
                                let outcome = run_one_function(
                                    passes,
                                    e.body,
                                    view,
                                    &mut local_cache,
                                    &mut e.analyses,
                                    &mut local_elapsed,
                                    stage_label,
                                    &name,
                                    repeat_until,
                                    |pass| {
                                        // Send failure only means the master stopped
                                        // pumping (it never does before join); ignore it.
                                        let _ = tx.send(PipelineProgress::FunctionPass {
                                            round,
                                            stage: stage_name.clone(),
                                            function: name.clone(),
                                            index: index + 1,
                                            total,
                                            pass,
                                        });
                                    },
                                )?;
                                e.outcome = outcome;
                            }
                            // Drain this thread's `stat!` counters before it exits — the
                            // thread-local table is otherwise lost — for re-absorption.
                            Ok(WorkerOutput {
                                cache: local_cache,
                                elapsed: local_elapsed,
                                stats: qcode::pass_scope::drain_stats(),
                            })
                        })
                        .expect("failed to spawn qcode pass worker thread");
                    handles.push(handle);
                }
                // The master holds no sender: once every worker's sender drops the
                // channel closes and the pump loop ends.
                drop(tx);
                while let Ok(event) = rx.recv() {
                    progress(event);
                }
                handles
                    .into_iter()
                    .map(|h| {
                        h.join().unwrap_or_else(|payload| {
                            let msg = payload
                                .downcast_ref::<&str>()
                                .map(|s| (*s).to_owned())
                                .or_else(|| payload.downcast_ref::<String>().cloned())
                                .unwrap_or_else(|| "unknown panic payload".to_owned());
                            panic!("function-pass worker panicked: {msg}");
                        })
                    })
                    .collect()
            })
        };

        // Merge worker-local state deterministically (worklist / chunk order).
        for outcome in worker_outputs {
            let out = outcome?;
            cache.absorb(out.cache);
            for (pass, (dur, runs, changes)) in out.elapsed {
                let entry = elapsed.entry(pass).or_default();
                entry.0 += dur;
                entry.1 += runs;
                entry.2 += changes;
            }
            qcode::pass_scope::absorb_stats(out.stats);
        }

        // Move each outcome out (releasing the entries and their body borrows),
        // in worklist order, for the barrier below.
        entries
            .into_iter()
            .map(|e| (e.fun_id, e.outcome, e.analyses))
            .collect::<Vec<_>>()
    };

    // 6. Barrier (master, worklist order): install and resolve minted callees,
    //    apply the returned self-rename, and record dirtiness. The bodies were
    //    mutated in place, so there is nothing to reinstall.
    for (fun_id, outcome, local_analyses) in results {
        analyses.put_local(fun_id, local_analyses);
        if !outcome.type_requests.is_empty() {
            ctx.shared
                .types
                .create_requested_types(&outcome.type_requests);
            retry.push(fun_id);
        }
        let installed = install_minted(ctx, &stage.name, outcome.minted)?;
        let patched = resolve_minted_callees(ctx, &stage.name, fun_id, &installed)?;
        replay_rename(ctx, &stage.name, fun_id, outcome.rename)?;
        let verify_scope: HashSet<FunctionId> = installed.iter().copied().chain([fun_id]).collect();
        dirty.extend(installed);
        // Opt-in `QCODE_VERIFY` check once the split borrow has ended, scoped to
        // the one function this outcome belongs to (plus its mints). A no-op
        // unless enabled.
        crate::verify::verify_after(
            ctx,
            &stage.name,
            crate::verify::Scope::Functions(&verify_scope),
        );
        if outcome.changed || patched {
            analyses.invalidate_globals(&outcome.preserved_analyses);
            dirty.insert(fun_id);
        }
    }
    Ok(())
}

#[cfg(test)]
mod fingerprint_tests {
    use super::fingerprint_display;

    /// The property the cycle detector depends on: renumbering temporaries without
    /// changing structure must not change the fingerprint. Before normalization
    /// these two hashed differently, so an oscillating fixpoint produced a fresh
    /// fingerprint every iteration and no cycle was ever detected.
    #[test]
    fn renumbered_temporaries_fingerprint_identically() {
        let before = "%tmp1 = i32 %tmp2 + i32 0x4;\nreturn %tmp1;";
        let after = "%tmpa0 = i32 %tmpa1 + i32 0x4;\nreturn %tmpa0;";
        assert_eq!(
            fingerprint_display(before),
            fingerprint_display(after),
            "structurally identical bodies must fingerprint identically",
        );
    }

    /// Normalization must not flatten real differences into a false match — that
    /// would report a cycle that is not there and stop a converging fixpoint early.
    #[test]
    fn structural_differences_still_change_the_fingerprint() {
        let a = "%tmp1 = i32 %tmp2 + i32 0x4;";
        // Same ids, different operator.
        let b = "%tmp1 = i32 %tmp2 * i32 0x4;";
        // Same shape, but the two operands are now the *same* value — a real
        // structural difference that ordinals must preserve.
        let c = "%tmp1 = i32 %tmp1 + i32 0x4;";
        assert_ne!(fingerprint_display(a), fingerprint_display(b));
        assert_ne!(fingerprint_display(a), fingerprint_display(c));
    }

    /// Distinct atom kinds must not collide: `%tmp0` and `@param0` both normalize
    /// to ordinal 0, and only the retained prefix keeps them apart.
    #[test]
    fn instruction_and_param_atoms_do_not_collide() {
        assert_ne!(
            fingerprint_display("return %tmp0;"),
            fingerprint_display("return @param0;"),
        );
    }

    /// A *named* value that merely begins with an atom prefix is not an arena
    /// index and must pass through untouched, or two unrelated names would collapse
    /// into one ordinal and fabricate a cycle.
    #[test]
    fn named_values_resembling_atoms_are_not_normalized() {
        assert_ne!(
            fingerprint_display("return %tmpfoo;"),
            fingerprint_display("return %tmpbar;"),
        );
        // `%tmpab` is a name (letters continue past the hex run `ab`), not `%tmp` +
        // index `ab`; it must not be conflated with a genuine `%tmpab` index... but
        // it also must not crash or mis-slice on the boundary.
        assert_ne!(
            fingerprint_display("return %tmpabz;"),
            fingerprint_display("return %tmpaby;"),
        );
    }

    /// Ordinals are assigned by first appearance, which in real IR is pinned by
    /// the defining instructions. Swapping which *defined* value an operand reads
    /// is therefore a genuine difference, not a renaming.
    ///
    /// (Operands alone are not enough to show this: `%tmpA + %tmpB` and
    /// `%tmpB + %tmpA` with no definitions in sight really are alpha-equivalent,
    /// and hashing them equal is correct.)
    #[test]
    fn swapping_which_defined_value_is_read_is_significant() {
        let defs = "%tmp2 = load(ram:4, 0x10);\n%tmp3 = load(ram:4, 0x20);\n";
        assert_ne!(
            fingerprint_display(format!("{defs}%tmp1 = %tmp2 + %tmp3;")),
            fingerprint_display(format!("{defs}%tmp1 = %tmp3 + %tmp2;")),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::pass::ModuleFnAdapter;
    use crate::pipeline::{LiftOutcome, Lifter};
    use crate::{FunctionPass, FunctionPassAdapter};
    use qcode::{
        context::Context,
        discovery::Discovery,
        value::{BasicBlock, FunctionBody, FunctionKind},
    };
    use std::{
        collections::VecDeque,
        sync::{
            Arc, LazyLock, Mutex,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
    };

    static MODULE_ADAPTER_ACTIVE: AtomicUsize = AtomicUsize::new(0);
    static MODULE_ADAPTER_MAX_ACTIVE: AtomicUsize = AtomicUsize::new(0);
    static ROUND_PROBE_CHANGED: AtomicBool = AtomicBool::new(false);
    static ROUND_PROBE_A_RUNS: AtomicUsize = AtomicUsize::new(0);
    static ROUND_PROBE_B_RUNS: AtomicUsize = AtomicUsize::new(0);
    static CACHE_PROBE_RUNS: LazyLock<Mutex<HashMap<FunctionId, usize>>> =
        LazyLock::new(|| Mutex::new(HashMap::default()));
    static TARGET_LOCAL_BUILDS: AtomicUsize = AtomicUsize::new(0);
    static MODULE_TEST_LOCK: Mutex<()> = Mutex::new(());

    struct TargetLocal;

    impl crate::LocalAnalysis for TargetLocal {
        type Result = usize;

        fn analyze<'str>(_body: &FunctionBody<'str>, _cx: ContextView<'_, 'str>) -> Self::Result {
            TARGET_LOCAL_BUILDS.fetch_add(1, Ordering::SeqCst) + 1
        }
    }

    enum ScriptedReport {
        Functions(Vec<FunctionId>),
        PreservedFunctions(Vec<FunctionId>),
        TypeRequests(Vec<qcode::types::TypeRequest>),
        Module,
        None,
    }

    struct ScriptedModulePass {
        reports: Mutex<VecDeque<ScriptedReport>>,
        seen: Arc<Mutex<Vec<Vec<FunctionId>>>>,
    }

    impl ScriptedModulePass {
        fn new(
            reports: impl IntoIterator<Item = ScriptedReport>,
        ) -> (Self, Arc<Mutex<Vec<Vec<FunctionId>>>>) {
            let seen = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    reports: Mutex::new(reports.into_iter().collect()),
                    seen: seen.clone(),
                },
                seen,
            )
        }

        fn invoke(&self, targets: &[FunctionId]) -> crate::ModulePassOutcome {
            self.seen.lock().unwrap().push(targets.to_vec());
            match self
                .reports
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(ScriptedReport::None)
            {
                ScriptedReport::Functions(ids) => crate::ModulePassOutcome::functions(ids),
                ScriptedReport::PreservedFunctions(ids) => {
                    crate::ModulePassOutcome::functions(ids).preserving_local::<TargetLocal>()
                }
                ScriptedReport::TypeRequests(requests) => {
                    crate::ModulePassOutcome::requesting_types(requests)
                }
                ScriptedReport::Module => crate::ModulePassOutcome::module(),
                ScriptedReport::None => crate::ModulePassOutcome::default(),
            }
        }
    }

    impl DynPass for ScriptedModulePass {
        fn name(&self) -> &'static str {
            "scripted_module"
        }

        fn description(&self) -> &'static str {
            "Test module pass with scripted outcomes"
        }

        fn run(
            &self,
            _ctx: &mut Context,
            _env: &PipelineEnv,
            targets: &[FunctionId],
        ) -> Result<crate::ModulePassOutcome, String> {
            Ok(self.invoke(targets))
        }

        fn run_with_analyses(
            &self,
            _ctx: &mut Context,
            _env: &PipelineEnv,
            targets: &[FunctionId],
            _analyses: &mut AnalysisManager,
        ) -> Result<crate::ModulePassOutcome, String> {
            Ok(self.invoke(targets))
        }
    }

    #[derive(Default)]
    struct CacheProbe;

    impl FunctionPass for CacheProbe {
        const NAME: &'static str = "cache_probe";

        fn description(&self) -> &'static str {
            "Records adapter executions by function"
        }

        fn run<'str>(
            &self,
            f: &mut FunctionBody<'str>,
            _cx: ContextView<'_, 'str>,
            _next_minted: &mut u32,
        ) -> Result<Outcome<'str>, String> {
            *CACHE_PROBE_RUNS.lock().unwrap().entry(f.id()).or_default() += 1;
            Ok(Outcome::default())
        }
    }

    #[derive(Default)]
    struct AlwaysChangeProbe;

    impl FunctionPass for AlwaysChangeProbe {
        const NAME: &'static str = "always_change_probe";

        fn description(&self) -> &'static str {
            "Reports every adapter target changed"
        }

        fn run<'str>(
            &self,
            _f: &mut FunctionBody<'str>,
            _cx: ContextView<'_, 'str>,
            _next_minted: &mut u32,
        ) -> Result<Outcome<'str>, String> {
            Ok(Outcome::changed(true))
        }
    }

    fn repeated_module_stage() -> Stage {
        Stage {
            name: "target-fixpoint".into(),
            passes: StagePasses::Module(Vec::new()),
            repeat_until: Some(RepeatCond::NoChange),
            include_external: false,
            only_dirty: false,
            dump: Vec::new(),
        }
    }

    fn run_test_module_stage(
        ctx: &mut Context,
        stage: &Stage,
        passes: &[Box<dyn DynPass>],
    ) -> ModuleStageOutcome {
        let env = PipelineEnv::headless(ctx);
        run_module_stage(
            ctx,
            &env,
            stage,
            passes,
            &mut FixpointCache::default(),
            &mut AnalysisManager::default(),
            0,
            &mut |_| {},
        )
        .unwrap()
    }

    #[derive(Default)]
    struct ModuleAdapterParallelProbe;

    impl FunctionPass for ModuleAdapterParallelProbe {
        const NAME: &'static str = "module_adapter_parallel_probe";

        fn description(&self) -> &'static str {
            "Test pass that records concurrent module-adapter execution"
        }

        fn run<'str>(
            &self,
            _f: &mut FunctionBody<'str>,
            _cx: ContextView<'_, 'str>,
            _next_minted: &mut u32,
        ) -> Result<Outcome<'str>, String> {
            let active = MODULE_ADAPTER_ACTIVE.fetch_add(1, Ordering::SeqCst) + 1;
            MODULE_ADAPTER_MAX_ACTIVE.fetch_max(active, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(10));
            MODULE_ADAPTER_ACTIVE.fetch_sub(1, Ordering::SeqCst);
            Ok(Outcome::default())
        }
    }

    #[derive(Default)]
    struct RoundProbeA;

    impl FunctionPass for RoundProbeA {
        const NAME: &'static str = "round_probe_a";

        fn description(&self) -> &'static str {
            "Changes exactly one function on its first module round"
        }

        fn run<'str>(
            &self,
            _f: &mut FunctionBody<'str>,
            _cx: ContextView<'_, 'str>,
            _next_minted: &mut u32,
        ) -> Result<Outcome<'str>, String> {
            ROUND_PROBE_A_RUNS.fetch_add(1, Ordering::SeqCst);
            Ok(Outcome::changed(
                !ROUND_PROBE_CHANGED.swap(true, Ordering::SeqCst),
            ))
        }
    }

    #[derive(Default)]
    struct RoundProbeB;

    impl FunctionPass for RoundProbeB {
        const NAME: &'static str = "round_probe_b";

        fn description(&self) -> &'static str {
            "Records the module round target set"
        }

        fn run<'str>(
            &self,
            _f: &mut FunctionBody<'str>,
            _cx: ContextView<'_, 'str>,
            _next_minted: &mut u32,
        ) -> Result<Outcome<'str>, String> {
            ROUND_PROBE_B_RUNS.fetch_add(1, Ordering::SeqCst);
            Ok(Outcome::default())
        }
    }

    #[test]
    fn module_function_adapter_uses_shared_parallel_runner() {
        let _guard = MODULE_TEST_LOCK.lock().unwrap();
        MODULE_ADAPTER_ACTIVE.store(0, Ordering::SeqCst);
        MODULE_ADAPTER_MAX_ACTIVE.store(0, Ordering::SeqCst);

        let pipeline = Pipeline::parse(
            r#"
            [[stage]]
            name = "module-adapter"
            scope = "module"
            passes = ["module(gvn)"]
            "#,
        )
        .expect("test pipeline parses");
        let stage = &pipeline.stages[0];
        let mut ctx = Context::new();
        for i in 0..8 {
            FunctionBody::make_at_addr(&mut ctx, 0x1000 + i * 0x10, None);
        }
        let fun_ids = ctx.function_ids();
        let env = PipelineEnv::headless(&ctx);
        let pass = FunctionPassAdapter::<ModuleAdapterParallelProbe>::default();
        let mut analyses = AnalysisManager::default();
        let mut cache = FixpointCache::default();
        let changed = run_module_fn_adapter(
            &mut ctx,
            &env,
            stage,
            &pass,
            &fun_ids,
            &mut cache,
            &mut analyses,
            1,
            4,
            &mut |_| {},
        )
        .expect("module adapter runs");

        assert!(changed.is_empty());
        assert!(
            MODULE_ADAPTER_MAX_ACTIVE.load(Ordering::SeqCst) > 1,
            "module adapter did not execute concurrently"
        );
    }

    #[test]
    fn module_adapters_share_one_target_set_per_round() {
        let _guard = MODULE_TEST_LOCK.lock().unwrap();
        ROUND_PROBE_CHANGED.store(false, Ordering::SeqCst);
        ROUND_PROBE_A_RUNS.store(0, Ordering::SeqCst);
        ROUND_PROBE_B_RUNS.store(0, Ordering::SeqCst);

        let pipeline = Pipeline::parse(
            r#"
            [[stage]]
            name = "module-round-targets"
            scope = "module"
            passes = ["module(gvn)"]
            repeat_until = "no_change"
            "#,
        )
        .expect("test pipeline parses");
        let stage = &pipeline.stages[0];
        let mut ctx = Context::new();
        for i in 0..4 {
            FunctionBody::make_at_addr(&mut ctx, 0x1000 + i * 0x10, None);
        }
        let env = PipelineEnv::headless(&ctx);
        let passes: Vec<Box<dyn DynPass>> = vec![
            Box::new(ModuleFnAdapter {
                inner: Box::new(FunctionPassAdapter::<RoundProbeA>::default()),
            }),
            Box::new(ModuleFnAdapter {
                inner: Box::new(FunctionPassAdapter::<RoundProbeB>::default()),
            }),
        ];
        let mut analyses = AnalysisManager::default();
        let mut cache = FixpointCache::default();

        let outcome = run_module_stage(
            &mut ctx,
            &env,
            stage,
            &passes,
            &mut cache,
            &mut analyses,
            1,
            &mut |_| {},
        )
        .expect("incremental module stage runs");
        assert_eq!(outcome.changed_functions.len(), 1);

        // Both adapters receive all four targets in round one. Probe A dirties
        // one before probe B runs; B then settles on that newer generation, so
        // its ordinary fixpoint mark correctly skips redundant round-two work.
        assert_eq!(ROUND_PROBE_A_RUNS.load(Ordering::SeqCst), 5);
        assert_eq!(ROUND_PROBE_B_RUNS.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn genuine_module_pass_narrows_next_round_to_exact_changes() {
        let mut ctx = Context::new();
        let foo = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some("foo".into())).id;
        let bar = FunctionBody::make_at_addr(&mut ctx, 0x2000, Some("bar".into())).id;
        let (pass, seen) =
            ScriptedModulePass::new([ScriptedReport::Functions(vec![foo]), ScriptedReport::None]);

        let outcome = run_test_module_stage(&mut ctx, &repeated_module_stage(), &[Box::new(pass)]);

        assert_eq!(*seen.lock().unwrap(), vec![vec![foo, bar], vec![foo]]);
        assert_eq!(outcome.changed_functions, HashSet::from_iter([foo]));
    }

    #[test]
    fn module_type_requests_publish_and_retry_immediately() {
        let mut ctx = Context::new();
        let foo = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some("foo".into())).id;
        let byte = ctx.shared.types.get_or_make_int(1);
        let request = qcode::types::TypeRequest::array(byte, 17);
        let (pass, seen) = ScriptedModulePass::new([
            ScriptedReport::TypeRequests(vec![request]),
            ScriptedReport::None,
        ]);

        let outcome = run_test_module_stage(&mut ctx, &repeated_module_stage(), &[Box::new(pass)]);

        assert!(!outcome.module_changed);
        assert!(outcome.changed_functions.is_empty());
        assert!(ctx.shared.types.get_array(byte, 17).is_some());
        assert_eq!(*seen.lock().unwrap(), vec![vec![foo], vec![foo]]);
    }

    #[test]
    fn exact_multi_function_reporting_controls_the_next_round() {
        let mut ctx = Context::new();
        let foo = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some("foo".into())).id;
        let bar = FunctionBody::make_at_addr(&mut ctx, 0x2000, Some("bar".into())).id;
        let baz = FunctionBody::make_at_addr(&mut ctx, 0x3000, Some("baz".into())).id;
        let (pass, seen) = ScriptedModulePass::new([
            ScriptedReport::Functions(vec![foo, bar]),
            ScriptedReport::None,
        ]);

        run_test_module_stage(&mut ctx, &repeated_module_stage(), &[Box::new(pass)]);

        assert_eq!(
            *seen.lock().unwrap(),
            vec![vec![foo, bar, baz], vec![foo, bar]]
        );
    }

    #[test]
    fn changed_callee_does_not_implicitly_schedule_its_caller() {
        let mut ctx = Context::new();
        let foo = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some("foo".into())).id;
        let bar = FunctionBody::make_at_addr(&mut ctx, 0x2000, Some("bar".into())).id;
        let entry = ctx.get_or_make_block(0x2000, bar);
        FunctionBody::from_id_mut(&mut ctx, bar)
            .set_root(entry)
            .unwrap();
        (&mut ctx).builder(entry).push_call(foo);
        assert_eq!(crate::CallGraph::analyze(&ctx).callers(foo), vec![bar]);

        let (pass, seen) =
            ScriptedModulePass::new([ScriptedReport::Functions(vec![foo]), ScriptedReport::None]);
        run_test_module_stage(&mut ctx, &repeated_module_stage(), &[Box::new(pass)]);

        assert_eq!(seen.lock().unwrap()[1], vec![foo]);
    }

    #[test]
    fn explicitly_reported_callee_and_caller_are_both_scheduled() {
        let mut ctx = Context::new();
        let foo = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some("foo".into())).id;
        let bar = FunctionBody::make_at_addr(&mut ctx, 0x2000, Some("bar".into())).id;
        let entry = ctx.get_or_make_block(0x2000, bar);
        FunctionBody::from_id_mut(&mut ctx, bar)
            .set_root(entry)
            .unwrap();
        (&mut ctx).builder(entry).push_call(foo);

        let (pass, seen) = ScriptedModulePass::new([
            ScriptedReport::Functions(vec![foo, bar]),
            ScriptedReport::None,
        ]);
        run_test_module_stage(&mut ctx, &repeated_module_stage(), &[Box::new(pass)]);

        assert_eq!(seen.lock().unwrap()[1], vec![foo, bar]);
    }

    #[test]
    fn genuine_module_change_invalidates_only_its_adapter_target() {
        let _guard = MODULE_TEST_LOCK.lock().unwrap();
        CACHE_PROBE_RUNS.lock().unwrap().clear();
        let mut ctx = Context::new();
        let foo = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some("foo".into())).id;
        let bar = FunctionBody::make_at_addr(&mut ctx, 0x2000, Some("bar".into())).id;
        let (module, seen) =
            ScriptedModulePass::new([ScriptedReport::Functions(vec![foo]), ScriptedReport::None]);
        let passes: Vec<Box<dyn DynPass>> = vec![
            Box::new(ModuleFnAdapter {
                inner: Box::new(FunctionPassAdapter::<CacheProbe>::default()),
            }),
            Box::new(module),
        ];

        run_test_module_stage(&mut ctx, &repeated_module_stage(), &passes);

        assert_eq!(*seen.lock().unwrap(), vec![vec![foo, bar], vec![foo]]);
        let runs = CACHE_PROBE_RUNS.lock().unwrap();
        assert_eq!(runs[&foo], 2);
        assert_eq!(runs[&bar], 1);
    }

    #[test]
    fn mixed_module_stage_unions_changes_without_mutating_round_targets() {
        let _guard = MODULE_TEST_LOCK.lock().unwrap();
        ROUND_PROBE_CHANGED.store(false, Ordering::SeqCst);
        ROUND_PROBE_A_RUNS.store(0, Ordering::SeqCst);
        let mut ctx = Context::new();
        let foo = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some("foo".into())).id;
        let bar = FunctionBody::make_at_addr(&mut ctx, 0x2000, Some("bar".into())).id;
        let baz = FunctionBody::make_at_addr(&mut ctx, 0x3000, Some("baz".into())).id;
        let (module, seen) =
            ScriptedModulePass::new([ScriptedReport::Functions(vec![bar]), ScriptedReport::None]);
        let passes: Vec<Box<dyn DynPass>> = vec![
            Box::new(ModuleFnAdapter {
                inner: Box::new(FunctionPassAdapter::<RoundProbeA>::default()),
            }),
            Box::new(module),
        ];

        run_test_module_stage(&mut ctx, &repeated_module_stage(), &passes);

        // The genuine pass still receives the original all-function target list
        // after the adapter changes foo. The next round is the ordered union.
        assert_eq!(
            *seen.lock().unwrap(),
            vec![vec![foo, bar, baz], vec![foo, bar]]
        );
        assert_eq!(ROUND_PROBE_A_RUNS.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn module_changed_retargets_all_and_stales_all_adapter_marks() {
        let _guard = MODULE_TEST_LOCK.lock().unwrap();
        CACHE_PROBE_RUNS.lock().unwrap().clear();
        let mut ctx = Context::new();
        let foo = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some("foo".into())).id;
        let bar = FunctionBody::make_at_addr(&mut ctx, 0x2000, Some("bar".into())).id;
        let (module, seen) =
            ScriptedModulePass::new([ScriptedReport::Module, ScriptedReport::None]);
        let passes: Vec<Box<dyn DynPass>> = vec![
            Box::new(ModuleFnAdapter {
                inner: Box::new(FunctionPassAdapter::<CacheProbe>::default()),
            }),
            Box::new(module),
        ];

        let outcome = run_test_module_stage(&mut ctx, &repeated_module_stage(), &passes);

        assert!(outcome.module_changed);
        assert_eq!(*seen.lock().unwrap(), vec![vec![foo, bar], vec![foo, bar]]);
        let runs = CACHE_PROBE_RUNS.lock().unwrap();
        assert_eq!(runs[&foo], 2);
        assert_eq!(runs[&bar], 2);
    }

    #[test]
    fn module_stage_preservation_and_local_invalidation_are_target_scoped() {
        let _guard = MODULE_TEST_LOCK.lock().unwrap();
        TARGET_LOCAL_BUILDS.store(0, Ordering::SeqCst);
        let mut ctx = Context::new();
        let foo = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some("foo".into())).id;
        let bar = FunctionBody::make_at_addr(&mut ctx, 0x2000, Some("bar".into())).id;
        let env = PipelineEnv::headless(&ctx);
        let stage = Stage {
            repeat_until: None,
            ..repeated_module_stage()
        };
        let mut analyses = AnalysisManager::default();
        for id in [foo, bar] {
            let mut local = analyses.take_local(id);
            {
                let (bodies, view) = ctx.split(&env);
                let _ = local.get::<TargetLocal>(&bodies[id], view);
            }
            analyses.put_local(id, local);
        }
        assert_eq!(TARGET_LOCAL_BUILDS.load(Ordering::SeqCst), 2);

        let (preserving, _) =
            ScriptedModulePass::new([ScriptedReport::PreservedFunctions(vec![foo])]);
        let mut cache = FixpointCache::default();
        run_module_stage(
            &mut ctx,
            &env,
            &stage,
            &[Box::new(preserving)],
            &mut cache,
            &mut analyses,
            0,
            &mut |_| {},
        )
        .unwrap();
        for id in [foo, bar] {
            let mut local = analyses.take_local(id);
            {
                let (bodies, view) = ctx.split(&env);
                let _ = local.get::<TargetLocal>(&bodies[id], view);
            }
            analyses.put_local(id, local);
        }
        assert_eq!(TARGET_LOCAL_BUILDS.load(Ordering::SeqCst), 2);

        let (invalidating, _) = ScriptedModulePass::new([ScriptedReport::Functions(vec![foo])]);
        run_module_stage(
            &mut ctx,
            &env,
            &stage,
            &[Box::new(invalidating)],
            &mut cache,
            &mut analyses,
            0,
            &mut |_| {},
        )
        .unwrap();
        for id in [foo, bar] {
            let mut local = analyses.take_local(id);
            {
                let (bodies, view) = ctx.split(&env);
                let _ = local.get::<TargetLocal>(&bodies[id], view);
            }
            analyses.put_local(id, local);
        }
        assert_eq!(TARGET_LOCAL_BUILDS.load(Ordering::SeqCst), 3);

        let (module_wide, _) = ScriptedModulePass::new([ScriptedReport::Module]);
        run_module_stage(
            &mut ctx,
            &env,
            &stage,
            &[Box::new(module_wide)],
            &mut cache,
            &mut analyses,
            0,
            &mut |_| {},
        )
        .unwrap();
        for id in [foo, bar] {
            let mut local = analyses.take_local(id);
            {
                let (bodies, view) = ctx.split(&env);
                let _ = local.get::<TargetLocal>(&bodies[id], view);
            }
            analyses.put_local(id, local);
        }
        assert_eq!(TARGET_LOCAL_BUILDS.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn adapter_eligibility_filters_external_and_ignored_targets_only() {
        let _guard = MODULE_TEST_LOCK.lock().unwrap();
        CACHE_PROBE_RUNS.lock().unwrap().clear();
        let mut ctx = Context::new();
        let normal = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some("normal".into())).id;
        let ignored = FunctionBody::make_at_addr(&mut ctx, 0x2000, Some("ignored".into())).id;
        let external = FunctionBody::make_external(&mut ctx, 0x3000, Some("external".into())).id;
        ctx.set_ignored_functions(HashSet::from_iter([0x2000]));
        let (module, seen) = ScriptedModulePass::new([ScriptedReport::None]);
        let passes: Vec<Box<dyn DynPass>> = vec![
            Box::new(module),
            Box::new(ModuleFnAdapter {
                inner: Box::new(FunctionPassAdapter::<CacheProbe>::default()),
            }),
        ];

        run_test_module_stage(
            &mut ctx,
            &Stage {
                repeat_until: None,
                ..repeated_module_stage()
            },
            &passes,
        );

        assert_eq!(seen.lock().unwrap()[0], vec![normal, ignored, external]);
        let runs = CACHE_PROBE_RUNS.lock().unwrap();
        assert_eq!(runs.get(&normal), Some(&1));
        assert!(!runs.contains_key(&ignored));
        assert!(!runs.contains_key(&external));
        drop(runs);

        CACHE_PROBE_RUNS.lock().unwrap().clear();
        run_test_module_stage(
            &mut ctx,
            &Stage {
                repeat_until: None,
                include_external: true,
                ..repeated_module_stage()
            },
            &[Box::new(ModuleFnAdapter {
                inner: Box::new(FunctionPassAdapter::<CacheProbe>::default()),
            })],
        );
        let runs = CACHE_PROBE_RUNS.lock().unwrap();
        assert_eq!(runs.get(&normal), Some(&1));
        assert!(!runs.contains_key(&ignored));
        assert_eq!(runs.get(&external), Some(&1));
    }

    #[test]
    fn downstream_only_dirty_stage_receives_exact_module_changes() {
        let _guard = MODULE_TEST_LOCK.lock().unwrap();
        CACHE_PROBE_RUNS.lock().unwrap().clear();
        let mut ctx = Context::new();
        let foo = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some("foo".into())).id;
        let bar = FunctionBody::make_at_addr(&mut ctx, 0x2000, Some("bar".into())).id;
        let env = PipelineEnv::headless(&ctx);
        let module_stage = Stage {
            repeat_until: None,
            ..repeated_module_stage()
        };
        let (module, _) = ScriptedModulePass::new([ScriptedReport::Functions(vec![foo])]);
        let mut cache = FixpointCache::default();
        let mut analyses = AnalysisManager::default();
        let outcome = run_module_stage(
            &mut ctx,
            &env,
            &module_stage,
            &[Box::new(module)],
            &mut cache,
            &mut analyses,
            0,
            &mut |_| {},
        )
        .unwrap();
        assert!(!outcome.module_changed);
        assert_eq!(outcome.changed_functions, HashSet::from_iter([foo]));

        let function_stage = Stage {
            name: "only-dirty".into(),
            passes: StagePasses::Function(Vec::new()),
            repeat_until: None,
            include_external: false,
            only_dirty: true,
            dump: Vec::new(),
        };
        let passes: Vec<Box<dyn DynFunctionPass>> =
            vec![Box::new(FunctionPassAdapter::<CacheProbe>::default())];
        run_function_stage(
            &mut ctx,
            &env,
            &function_stage,
            &passes,
            Some(&outcome.changed_functions),
            None,
            &mut cache,
            &mut analyses,
            0,
            &mut |_| {},
        )
        .unwrap();

        let runs = CACHE_PROBE_RUNS.lock().unwrap();
        assert_eq!(runs.get(&foo), Some(&1));
        assert!(!runs.contains_key(&bar));
    }

    #[test]
    fn module_adapter_sequential_and_parallel_results_match() {
        let build = || {
            let mut ctx = Context::new();
            for i in 0..8 {
                FunctionBody::make_at_addr(&mut ctx, 0x1000 + i * 0x10, None);
            }
            ctx
        };
        let mut sequential = build();
        let mut parallel = build();
        let stage = Stage {
            repeat_until: None,
            ..repeated_module_stage()
        };
        let pass = FunctionPassAdapter::<AlwaysChangeProbe>::default();

        let run = |ctx: &mut Context, threads| {
            let ids = ctx.function_ids();
            let env = PipelineEnv::headless(ctx);
            run_module_fn_adapter(
                ctx,
                &env,
                &stage,
                &pass,
                &ids,
                &mut FixpointCache::default(),
                &mut AnalysisManager::default(),
                0,
                threads,
                &mut |_| {},
            )
            .unwrap()
        };

        assert_eq!(run(&mut sequential, 1), run(&mut parallel, 4));
    }

    struct ChainedLifter {
        last: u64,
        lifted: usize,
    }

    impl Lifter for ChainedLifter {
        fn seed_binary(
            &mut self,
            _ctx: &mut Context,
            _addresses: &mut qcode::address_index::AddressIndex,
        ) -> Result<Vec<Discovery>, String> {
            Ok(Vec::new())
        }

        fn ensure_function(
            &mut self,
            _ctx: &mut Context,
            _addresses: &mut qcode::address_index::AddressIndex,
            _addr: u64,
        ) {
        }

        fn lift_discovered(
            &mut self,
            _ctx: &mut Context,
            _addresses: &mut qcode::address_index::AddressIndex,
            discovery: Discovery,
        ) -> Result<LiftOutcome, String> {
            let key = discovery.key();
            self.lifted += 1;
            let successors = (discovery.target < self.last)
                .then(|| Discovery::function(discovery.target + 1))
                .into_iter()
                .collect();
            Ok(LiftOutcome::Lifted { key, successors })
        }
    }

    #[test]
    fn lifting_stage_drains_more_than_generic_fixpoint_cap() {
        let pipeline = Pipeline::parse(
            r#"
            [[stage]]
            name = "lift"
            scope = "module"
            passes = ["lift_new_addresses"]
            repeat_until = "no_change"
            "#,
        )
        .expect("test pipeline parses");
        let mut ctx = Context::new();
        assert!(ctx.discover(Discovery::function(0)));
        let expected = MAX_FIXPOINT_ITERS + 2;
        let mut lifter = ChainedLifter {
            last: expected as u64 - 1,
            lifted: 0,
        };
        let env = PipelineEnv::headless(&mut ctx);
        let mut services = PipelineServices::with_lifter(&mut lifter);

        pipeline
            .run_lifting_phase(&mut ctx, &env, &mut services, 1, &mut |_| {})
            .expect("lifting drains the complete discovery chain");

        assert_eq!(lifter.lifted, expected);
        assert!(ctx.has_no_discoveries());
    }

    fn run_function_only_pipeline_with_threads(
        pipeline: &Pipeline,
        ctx: &mut Context,
        threads: usize,
    ) {
        let env = PipelineEnv::headless(ctx);
        let mut dirty = Some(HashSet::default());
        let mut cache = FixpointCache::default();
        let mut analyses = AnalysisManager::default();
        let mut progress = |_| {};
        for stage in &pipeline.stages {
            let StagePasses::Function(passes) = &stage.passes else {
                panic!("test pipeline must contain only function stages");
            };
            dirty = Some(
                run_function_stage_with_threads(
                    ctx,
                    &env,
                    stage,
                    passes,
                    dirty.as_ref(),
                    None,
                    &mut cache,
                    &mut analyses,
                    0,
                    &mut progress,
                    threads,
                )
                .expect("function stage runs"),
            );
        }
    }

    fn demangle_then_thunk_context() -> (Context<'static>, FunctionId) {
        let mut ctx = Context::new();
        // Registry order matters for the old sequential behavior: publish the
        // callee's demangle before visiting its later thunk.
        let callee =
            FunctionBody::make_at_addr(&mut ctx, 0x1000, Some("_ZN5space3fooEv".to_owned().into()))
                .id;
        let _dummy_a = FunctionBody::make_at_addr(&mut ctx, 0x2000, None).id;
        let _dummy_b = FunctionBody::make_at_addr(&mut ctx, 0x3000, None).id;
        let thunk = FunctionBody::make_at_addr(&mut ctx, 0x4000, None).id;
        let block = BasicBlock::make(&mut ctx, thunk).with_address(0x4000).id;
        (&mut ctx).builder(block).push_tail_call(callee);
        FunctionBody::from_id_mut(&mut ctx, thunk)
            .set_root(block)
            .unwrap();
        (ctx, thunk)
    }

    #[test]
    fn demangle_barrier_makes_thunk_naming_thread_count_independent() {
        let pipeline = Pipeline::parse(
            r#"
            [[stage]]
            name = "demangle"
            scope = "function"
            passes = ["cpp_demangle"]
            include_external = true

            [[stage]]
            name = "name-thunks"
            scope = "function"
            passes = ["name_thunks"]
            include_external = true
            "#,
        )
        .expect("test pipeline parses");
        let (base, thunk) = demangle_then_thunk_context();
        assert!(base.functions().count() >= PARALLEL_THRESHOLD);
        let mut sequential = base.clone();
        let mut parallel = base;

        run_function_only_pipeline_with_threads(&pipeline, &mut sequential, 1);
        run_function_only_pipeline_with_threads(&pipeline, &mut parallel, 4);

        let sequential_names: Vec<_> = sequential
            .functions()
            .map(|f| f.name().to_owned())
            .collect();
        let parallel_names: Vec<_> = parallel.functions().map(|f| f.name().to_owned()).collect();
        assert_eq!(sequential_names, parallel_names);
        assert_eq!(
            FunctionBody::from_id(&sequential, thunk).name(),
            "thunk_space::foo"
        );
    }

    #[derive(Default)]
    struct RequestArrayType;

    impl FunctionPass for RequestArrayType {
        const NAME: &'static str = "test_request_array_type";

        fn description(&self) -> &'static str {
            "Request one owner-specific array type, then rename after publication"
        }

        fn run<'str>(
            &self,
            f: &mut FunctionBody<'str>,
            cx: ContextView<'_, 'str>,
            _next_minted: &mut u32,
        ) -> Result<Outcome<'str>, String> {
            let elem = cx.shr().types.get_int(1);
            let count = usize::from(f.id()) + 1;
            if cx.shr().types.get_array(elem, count).is_none() {
                return Ok(Outcome::requesting_type(qcode::types::TypeRequest::array(
                    elem, count,
                )));
            }
            Ok(Outcome::renamed(format!("typed_{count}").into()))
        }
    }

    crate::register_function_pass!(RequestArrayType);

    #[test]
    fn returned_type_requests_are_published_deterministically() {
        let pipeline = Pipeline::parse(
            r#"
            [[stage]]
            name = "request-types"
            scope = "function"
            passes = ["test_request_array_type"]
            include_external = true
            "#,
        )
        .expect("test pipeline parses");
        let mut base = Context::new();
        let byte = base.shared.types.get_or_make_int(1);
        for index in 0..PARALLEL_THRESHOLD.max(4) {
            FunctionBody::make(&mut base, format!("f{index}").into()).unwrap();
        }
        let mut sequential = base.clone();
        let mut parallel = base;

        run_function_only_pipeline_with_threads(&pipeline, &mut sequential, 1);
        run_function_only_pipeline_with_threads(&pipeline, &mut parallel, 4);

        let sequential_types: Vec<_> = sequential
            .function_ids()
            .into_iter()
            .map(|fid| {
                sequential
                    .shared
                    .types
                    .get_array(byte, usize::from(fid) + 1)
            })
            .collect();
        let parallel_types: Vec<_> = parallel
            .function_ids()
            .into_iter()
            .map(|fid| parallel.shared.types.get_array(byte, usize::from(fid) + 1))
            .collect();
        assert!(sequential_types.iter().all(Option::is_some));
        assert_eq!(sequential_types, parallel_types);
        assert_eq!(
            sequential
                .functions()
                .map(|f| f.name().to_owned())
                .collect::<Vec<_>>(),
            parallel
                .functions()
                .map(|f| f.name().to_owned())
                .collect::<Vec<_>>()
        );
    }

    #[derive(Default)]
    struct MintWithoutChanged;

    impl FunctionPass for MintWithoutChanged {
        const NAME: &'static str = "test_mint_without_changed";

        fn description(&self) -> &'static str {
            "Test pass that mints while omitting the body-change bit"
        }

        fn run<'str>(
            &self,
            f: &mut FunctionBody<'str>,
            _cx: ContextView<'_, 'str>,
            next_minted: &mut u32,
        ) -> Result<Outcome<'str>, String> {
            let mut minted = Vec::new();
            super::super::mint_function(
                f,
                next_minted,
                &mut minted,
                "minted".into(),
                FunctionKind::Machine,
                false,
            );
            Ok(Outcome::with_minted(false, minted))
        }
    }

    #[test]
    fn minting_counts_as_a_pass_change() {
        let mut ctx = Context::new();
        let owner = FunctionBody::make(&mut ctx, "owner".into()).unwrap().id;
        let env = PipelineEnv::headless(&mut ctx);
        let passes: Vec<Box<dyn DynFunctionPass>> = vec![Box::new(FunctionPassAdapter::<
            MintWithoutChanged,
        >::default())];
        let mut cache = FixpointCache::default();
        let mut elapsed = HashMap::default();
        let mut local_analyses = LocalAnalysisManager::default();
        let pass_refs: Vec<&dyn DynFunctionPass> = passes.iter().map(Box::as_ref).collect();
        let outcome = {
            let (bodies, view) = ctx.split(&env);
            run_one_function(
                &pass_refs,
                &mut bodies[owner],
                view,
                &mut cache,
                &mut local_analyses,
                &mut elapsed,
                "mint",
                "owner",
                false,
                |_| {},
            )
            .unwrap()
        };

        assert!(outcome.changed);
        assert_eq!(outcome.minted.len(), 1);
        assert!(!cache.is_clean(owner, MintWithoutChanged::NAME));
        assert_eq!(elapsed[MintWithoutChanged::NAME].2, 1);
    }

    #[test]
    fn default_pipeline_parses() {
        Pipeline::default();
    }

    #[test]
    fn debug_flag_is_parsed() {
        let pipeline = Pipeline::parse(
            r#"
            debug = true

            [[stage]]
            name = "x"
            scope = "module"
            passes = []
            "#,
        )
        .expect("debug pipeline parses");
        assert!(pipeline.debug);
    }

    #[test]
    fn nonconvergence_error_round_trips() {
        // The driver keys on the phrasing to halt best-effort instead of panicking,
        // so both forms this constructor emits must be recognized, and unrelated
        // pipeline errors must not be mistaken for a non-convergence.
        let module = nonconvergence_error("promote", None);
        let func = nonconvergence_error("promote", Some("fn_430c40"));
        assert!(is_nonconvergence(&module), "module form: {module}");
        assert!(is_nonconvergence(&func), "function form: {func}");
        assert!(func.contains("fn_430c40"), "names the culprit: {func}");
        assert!(!is_nonconvergence("gvn: some other failure"));
        assert!(!is_nonconvergence(
            "verifier failed after stage \"promote\":\nbad invariant"
        ));
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
            passes = ["summaries"]
            "#,
        );
        assert!(err.contains("whole-program pass"), "{err}");
    }

    #[test]
    fn only_dirty_is_function_scope_only() {
        let err = parse_err(
            r#"
            [[stage]]
            name = "x"
            scope = "module"
            passes = []
            only_dirty = true
            "#,
        );
        assert!(err.contains("only_dirty=true"), "{err}");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn runtime_pipeline_dir_is_seeded_when_empty() {
        let dir = std::env::temp_dir().join(format!(
            "harbinger-pipelines-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock after Unix epoch")
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let files = list_user_pipelines_in(&dir).expect("runtime pipelines list");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].name, "default");
        assert!(files[0].description.contains("Default whole-program"));
        assert!(files[0].path.ends_with(DEFAULT_PIPELINE_FILE));
        load_named_user_pipeline_in(&dir, "default").expect("seeded default pipeline parses");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn stale_default_is_resynced_to_embedded() {
        let dir = std::env::temp_dir().join(format!(
            "harbinger-pipelines-resync-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock after Unix epoch")
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp pipeline dir");
        // A `default.toml` written by an older binary (here, arbitrary stale
        // content) must be overwritten with the embedded default, not preserved.
        let default_path = dir.join(DEFAULT_PIPELINE_FILE);
        std::fs::write(&default_path, "description = \"stale\"\n").expect("write stale default");

        list_user_pipelines_in(&dir).expect("runtime pipelines list");

        assert_eq!(
            std::fs::read_to_string(&default_path).expect("read resynced default"),
            DEFAULT_PIPELINE_TOML,
            "default.toml is kept in sync with the embedded pipeline"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn runtime_pipeline_list_pins_default_then_sorts_by_name() {
        let dir = std::env::temp_dir().join(format!(
            "harbinger-pipelines-sort-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock after Unix epoch")
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp pipeline dir");
        std::fs::write(dir.join("zeta.toml"), "description = \"Zed\"\n").expect("write zeta");
        std::fs::write(dir.join("default.toml"), "description = \"Default\"\n")
            .expect("write default");
        std::fs::write(dir.join("alpha.toml"), "description = \"Alpha\"\n").expect("write alpha");

        let files = list_user_pipelines_in(&dir).expect("runtime pipelines list");
        let names: Vec<_> = files.iter().map(|file| file.name.as_str()).collect();
        assert_eq!(names, ["default", "alpha", "zeta"]);
        assert_eq!(files[1].description, "Alpha");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn new_runtime_pipeline_copies_default_with_unique_name() {
        let dir = std::env::temp_dir().join(format!(
            "harbinger-pipelines-create-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock after Unix epoch")
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let first = create_user_pipeline_from_default_in(&dir).expect("create first pipeline");
        let second = create_user_pipeline_from_default_in(&dir).expect("create second pipeline");
        assert_eq!(first.name, "pipeline");
        assert_eq!(second.name, "pipeline-2");
        assert_eq!(
            std::fs::read_to_string(first.path).expect("read created pipeline"),
            DEFAULT_PIPELINE_TOML
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn named_runtime_pipeline_uses_requested_name() {
        let dir = std::env::temp_dir().join(format!(
            "harbinger-pipelines-named-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock after Unix epoch")
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let file = create_named_user_pipeline_from_default_in(&dir, "strings-only")
            .expect("create named pipeline");
        assert_eq!(file.name, "strings-only");
        assert!(file.path.ends_with("strings-only.toml"));
        assert!(
            create_named_user_pipeline_from_default_in(&dir, "strings-only")
                .expect_err("duplicate name should fail")
                .contains("already exists")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
