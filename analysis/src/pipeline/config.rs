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
    ContextSplit, ContextView, FunctionBody, Outcome, PipelineProgress, ProgressSink, YieldSignal,
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
        ctx: &mut Context<'_>,
        env: &PipelineEnv,
        round: usize,
        progress: &mut impl ProgressSink,
    ) -> Result<(), String> {
        pollster::block_on(self.run_async(ctx, env, round, progress))
    }

    pub async fn run_async(
        &self,
        ctx: &mut Context<'_>,
        env: &PipelineEnv,
        round: usize,
        progress: &mut impl ProgressSink,
    ) -> Result<(), String> {
        self.run_stages(0..self.stages.len(), ctx, env, round, progress)
            .await
    }

    /// Run the clean-IR lifting stages before the `code-discovery-fixpoint`
    /// barrier. These are ordinary TOML stages; the caller supplies lifter
    /// services for TOML-visible lifting passes.
    pub fn run_lifting_phase(
        &self,
        ctx: &mut Context<'_>,
        env: &PipelineEnv,
        services: &mut PipelineServices<'_>,
        round: usize,
        progress: &mut impl ProgressSink,
    ) -> Result<(), String> {
        pollster::block_on(self.run_lifting_phase_async(ctx, env, services, round, progress))
    }

    pub async fn run_lifting_phase_async(
        &self,
        ctx: &mut Context<'_>,
        env: &PipelineEnv,
        services: &mut PipelineServices<'_>,
        round: usize,
        progress: &mut impl ProgressSink,
    ) -> Result<(), String> {
        let end = self.barrier_index().unwrap_or(self.stages.len());
        let mut dirty_functions = Some(HashSet::default());
        let mut cache = FixpointCache::default();
        for stage in &self.stages[..end] {
            match &stage.passes {
                StagePasses::Module(passes) => {
                    let before = cache.snapshot_clean(ctx);
                    if run_lifting_module_stage(ctx, env, services, stage, passes, round, progress)
                        .await?
                    {
                        dirty_functions = None;
                    }
                    cache.invalidate_changed(ctx, before);
                }
                StagePasses::Function(passes) => {
                    dirty_functions = Some(
                        run_function_stage(
                            ctx,
                            env,
                            stage,
                            passes,
                            dirty_functions.as_ref(),
                            None,
                            &mut cache,
                            round,
                            progress,
                        )
                        .await?,
                    );
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
        ctx: &mut Context<'_>,
        env: &PipelineEnv,
        restrict: Option<&HashSet<FunctionId>>,
        round: usize,
        progress: &mut impl ProgressSink,
    ) -> Result<(), String> {
        pollster::block_on(
            self.run_address_discovery_phase_async(ctx, env, restrict, round, progress),
        )
    }

    pub async fn run_address_discovery_phase_async(
        &self,
        ctx: &mut Context<'_>,
        env: &PipelineEnv,
        restrict: Option<&HashSet<FunctionId>>,
        round: usize,
        progress: &mut impl ProgressSink,
    ) -> Result<(), String> {
        let start = self.barrier_index().map(|i| i + 1).unwrap_or(0);
        let mut dirty_functions = Some(HashSet::default());
        let mut cache = FixpointCache::default();
        for stage in &self.stages[start..] {
            match &stage.passes {
                StagePasses::Function(passes) => {
                    let reaches_discovery_pass =
                        passes.iter().any(|p| p.name() == ADDRESS_DISCOVERY_PASS);
                    dirty_functions = Some(
                        run_function_stage(
                            ctx,
                            env,
                            stage,
                            passes,
                            dirty_functions.as_ref(),
                            restrict,
                            &mut cache,
                            round,
                            progress,
                        )
                        .await?,
                    );
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
                        run_module_stage(ctx, env, stage, passes, round, progress)?;
                        self.verify_after_stage(ctx, stage)?;
                        return Ok(());
                    }
                    self.verify_after_stage(ctx, stage)?;
                }
            }
        }
        Ok(())
    }

    async fn run_stages(
        &self,
        range: std::ops::Range<usize>,
        ctx: &mut Context<'_>,
        env: &PipelineEnv,
        round: usize,
        progress: &mut impl ProgressSink,
    ) -> Result<(), String> {
        let mut dirty_functions = Some(HashSet::default());
        let mut cache = FixpointCache::default();
        for stage in &self.stages[range] {
            match &stage.passes {
                StagePasses::Module(passes) => {
                    // Fingerprint the clean functions, run the stage, then invalidate
                    // only those the stage actually modified (see `snapshot_clean`).
                    let before = cache.snapshot_clean(ctx);
                    if run_module_stage(ctx, env, stage, passes, round, progress).await? {
                        dirty_functions = None;
                    }
                    cache.invalidate_changed(ctx, before);
                }
                StagePasses::Function(passes) => {
                    dirty_functions = Some(
                        run_function_stage(
                            ctx,
                            env,
                            stage,
                            passes,
                            dirty_functions.as_ref(),
                            None,
                            &mut cache,
                            round,
                            progress,
                        )
                        .await?,
                    );
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
            Some(RegisteredPass::Decompile(_)) => {
                return Err(format!(
                    "pass \"{name}\" in stage \"{}\" is a decompilation pass, not usable in an IR stage",
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
            Some(RegisteredPass::Decompile(_)) => {
                return Err(format!(
                    "pass \"{name}\" in stage \"{}\" is a decompilation pass, not usable in an IR stage",
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

/// Run a whole-program stage; if `repeat_until` is set, loop the stage (OR-ing
/// the passes' change flags) until nothing changes or the iteration cap is hit.
async fn run_module_stage(
    ctx: &mut Context<'_>,
    env: &PipelineEnv,
    stage: &Stage,
    passes: &[Box<dyn DynPass>],
    round: usize,
    progress: &mut impl ProgressSink,
) -> Result<bool, String> {
    dump_stage_inputs(ctx, &stage.dump, &stage.name);
    let stage_name: std::sync::Arc<str> = stage.name.as_str().into();

    // A `repeat_until` module stage that contains `module(<fn_pass>)` adapters
    // (currently only `mark-pure`) re-runs each wrapped per-function pass over the
    // *whole* program every fixpoint iteration, though each iteration typically
    // changes only a handful of functions. Drive it incrementally instead: skip a
    // function an adapter has already settled, invalidating it (and its callers)
    // only when its body or a callee's purity changes.
    if stage.repeat_until.is_some() && passes.iter().any(|p| p.as_module_fn().is_some()) {
        return run_module_stage_incremental(ctx, env, stage, passes, &stage_name, round, progress);
    }

    let mut iters = 0;
    let mut stage_changed = false;
    let mut tracer = FixpointTracer::default();
    let tracer_label = format!("stage {} <module>", stage.name);
    loop {
        let watching = stage.repeat_until.is_some() && iters >= FIXPOINT_WATCH_ITERS;
        let mut changed = false;
        for p in passes {
            progress.report(PipelineProgress::WholeProgramPhase {
                round,
                stage: stage_name.clone(),
                pass: p.name(),
            });
            let _scope = qcode::pass_scope::enter(p.name());
            #[cfg(not(target_arch = "wasm32"))]
            let started = std::time::Instant::now();
            let pass_changed = p.run(ctx, env).map_err(|e| format!("{}: {e}", p.name()))?;
            #[cfg(not(target_arch = "wasm32"))]
            log::debug!(
                target: "pipeline",
                "{} ran in {:.2?} ({})",
                p.name(),
                started.elapsed(),
                if pass_changed { "changed" } else { "no change" },
            );
            // Between-pass invariant check (opt-in via `QCODE_VERIFY`): pin a broken
            // invariant to the pass that produced it.
            crate::verify::verify_after(ctx, p.name());
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
                    return Ok(true);
                }
            }
            changed |= pass_changed;
        }
        stage_changed |= changed;
        iters += 1;
        if stage.repeat_until.is_none() || !changed {
            return Ok(stage_changed);
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
    }
}

/// The set of functions currently flagged `is_pure`.
fn pure_function_set(ctx: &Context) -> HashSet<FunctionId> {
    ctx.functions()
        .filter(|f| f.is_pure())
        .map(|f| f.id)
        .collect()
}

/// A cheap structural hash of one function's body — block ids and params, then each
/// instruction's id and mnemonic (which carries its operand value-ids, so an operand
/// rewrite changes the hash). Unlike [`function_fingerprint`] it does no string
/// rendering, so it is affordable to call over a large clean set every fixpoint
/// iteration; it need only detect same-run before/after changes, not be stable
/// across runs. Used by [`run_module_stage_incremental`] to invalidate settled
/// functions a whole-program pass rewrote.
fn function_body_hash(ctx: &Context, fun_id: FunctionId) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for block in FunctionRef::from_id(ctx, fun_id).blocks() {
        block.id.hash(&mut h);
        for param in block.params() {
            param.id().hash(&mut h);
        }
        for insn in block.iter() {
            insn.id.hash(&mut h);
            insn.mnemonic().hash(&mut h);
        }
    }
    h.finish()
}

/// Mark every caller of `fun_id` dirty in `cache`. A change to a function's body,
/// signature, or purity can change what its callers' `gvn`/`dce` do (pure-call
/// emulation and dead-pure-call removal both read the callee), so a settled caller
/// must be reprocessed.
fn invalidate_callers(graph: &crate::CallGraph, cache: &mut FixpointCache, fun_id: FunctionId) {
    for caller in graph.callers(fun_id) {
        cache.mark_dirty(caller);
    }
}

/// Incremental driver for a `repeat_until` module stage carrying `module(<fn_pass>)`
/// adapters (see [`run_module_stage`]).
///
/// Correctness rests on tracking, per `(function, pass)`, whether the adapter has
/// reached a fixpoint on that function and the function is unchanged since
/// ([`FixpointCache`]). Two interprocedural dependencies force extra invalidation,
/// because `gvn`/`dce` read a *callee* when optimizing a caller (pure-call
/// emulation / dead-pure-call removal):
///   * when an adapter changes a function, its callers are invalidated;
///   * around a genuine whole-program pass (e.g. `argpromote`, `mark_pure`), any
///     function whose body fingerprint changed — and any function newly flagged
///     `is_pure` — invalidates itself and its callers.
///
/// The result is identical IR to re-running every adapter over every function each
/// iteration, at a fraction of the work.
fn run_module_stage_incremental(
    ctx: &mut Context,
    env: &PipelineEnv,
    stage: &Stage,
    passes: &[Box<dyn DynPass>],
    stage_name: &std::sync::Arc<str>,
    round: usize,
    progress: &mut impl FnMut(PipelineProgress),
) -> Result<bool, String> {
    let mut cache = FixpointCache::default();
    let mut known_pure = pure_function_set(ctx);
    let mut iters = 0;
    let mut stage_changed = false;
    let mut tracer = FixpointTracer::default();
    let tracer_label = format!("stage {} <module,incremental>", stage.name);
    loop {
        let watching = iters >= FIXPOINT_WATCH_ITERS;
        let mut changed = false;
        for p in passes {
            progress(PipelineProgress::WholeProgramPhase {
                round,
                stage: stage_name.clone(),
                pass: p.name(),
            });
            let _scope = qcode::pass_scope::enter(p.name());
            #[cfg(not(target_arch = "wasm32"))]
            let started = std::time::Instant::now();

            let pass_changed = if let Some(inner) = p.as_module_fn() {
                // Per-function adapter: run only functions not already settled for
                // this pass; a change dirties the function (all passes' marks) and
                // its callers, a no-change settles it.
                let fun_ids: Vec<FunctionId> = ctx
                    .functions()
                    .filter(|f| !f.is_external())
                    .filter(|f| !ctx.is_function_ignored(f.address()))
                    .map(|f| f.id)
                    .collect();
                let mut any = false;
                for fun_id in fun_ids {
                    if cache.is_clean(fun_id, p.name()) {
                        continue;
                    }
                    if inner.run(ctx, fun_id, env)? {
                        cache.mark_dirty(fun_id);
                        // `inner` may have introduced a new incoming edge to a
                        // function processed later in this pass. Rebuild after
                        // every changing adapter rather than updating a hidden
                        // graph incrementally.
                        let graph = crate::CallGraph::analyze(ctx);
                        invalidate_callers(&graph, &mut cache, fun_id);
                        any = true;
                    } else {
                        cache.mark_clean(fun_id, p.name());
                    }
                }
                any
            } else {
                // Genuine whole-program pass. Only *settled* (clean) functions risk
                // being wrongly skipped later, and any already-dirty function it
                // rewrites had its callers invalidated when it first went dirty — so
                // fingerprinting the clean set alone soundly catches every new
                // invalidation. Diff those fingerprints across the pass and dirty the
                // ones that moved (plus their callers); also dirty callers of any
                // function newly flagged `is_pure`.
                let before: Vec<(FunctionId, u64)> = cache
                    .clean_function_ids()
                    .into_iter()
                    .map(|f| (f, function_body_hash(ctx, f)))
                    .collect();
                let pc = p.run(ctx, env).map_err(|e| format!("{}: {e}", p.name()))?;
                if pc {
                    let graph = crate::CallGraph::analyze(ctx);
                    for (f, old) in before {
                        if function_body_hash(ctx, f) != old {
                            cache.mark_dirty(f);
                            invalidate_callers(&graph, &mut cache, f);
                        }
                    }
                    let now_pure = pure_function_set(ctx);
                    for &f in now_pure.difference(&known_pure) {
                        invalidate_callers(&graph, &mut cache, f);
                    }
                    known_pure = now_pure;
                }
                pc
            };

            #[cfg(not(target_arch = "wasm32"))]
            log::debug!(
                target: "pipeline",
                "{} ran in {:.2?} ({})",
                p.name(),
                started.elapsed(),
                if pass_changed { "changed" } else { "no change" },
            );
            crate::verify::verify_after(ctx, p.name());
            if watching && pass_changed {
                let fp = module_fingerprint(ctx);
                if tracer.observe(&tracer_label, iters + 1, p.name(), fp) {
                    // A proven cycle cannot converge; every further iteration
                    // re-treads the same states. Stop the stage best-effort at
                    // the recurring state instead of spinning to the cap.
                    log::warn!(
                        target: "pipeline::fixpoint",
                        "stage {} (module,incremental): stopping best-effort on the proven \
                         cycle at iteration {}",
                        stage.name,
                        iters + 1,
                    );
                    return Ok(true);
                }
            }
            changed |= pass_changed;
        }
        stage_changed |= changed;
        iters += 1;
        if !changed {
            return Ok(stage_changed);
        }
        if iters >= MAX_FIXPOINT_ITERS {
            log::warn!(
                target: "pipeline::fixpoint",
                "stage {} (module,incremental) hit the {MAX_FIXPOINT_ITERS}-iteration cap; \
                 see the `pipeline::fixpoint` trace above for the fighting passes",
                stage.name,
            );
            return Err(nonconvergence_error(&stage.name, None));
        }
    }
}

/// Run a clean-IR whole-program stage during recursive lifting. Most passes are
/// ordinary analysis passes; the two lifting pass names are TOML-visible
/// adapters over the caller-provided lifter service.
async fn run_lifting_module_stage(
    ctx: &mut Context<'_>,
    env: &PipelineEnv,
    services: &mut PipelineServices<'_>,
    stage: &Stage,
    passes: &[Box<dyn DynPass>],
    round: usize,
    progress: &mut impl ProgressSink,
) -> Result<bool, String> {
    dump_stage_inputs(ctx, &stage.dump, &stage.name);
    let stage_name: std::sync::Arc<str> = stage.name.as_str().into();
    let mut iters = 0;
    let mut stage_changed = false;
    loop {
        let mut changed = false;
        for p in passes {
            progress.report(PipelineProgress::WholeProgramPhase {
                round,
                stage: stage_name.clone(),
                pass: p.name(),
            });
            let _scope = qcode::pass_scope::enter(p.name());
            #[cfg(not(target_arch = "wasm32"))]
            let started = std::time::Instant::now();
            let pass_changed = match p.name() {
                "discover_addresses_in_binary" => {
                    crate::discover_addresses_in_binary(ctx, services)
                        .map_err(|e| format!("{}: {e}", p.name()))?
                        .changed()
                }
                "lift_new_addresses" => {
                    let summary = crate::lift_new_addresses(ctx, services)
                        .map_err(|e| format!("{}: {e}", p.name()))?;
                    if summary.changed()
                        && let Some(lifter) = services.lifter.as_deref_mut()
                    {
                        lifter
                            .finish_lifting(ctx)
                            .map_err(|e| format!("{}: {e}", p.name()))?;
                    }
                    summary.changed()
                }
                _ => p.run(ctx, env).map_err(|e| format!("{}: {e}", p.name()))?,
            };
            #[cfg(not(target_arch = "wasm32"))]
            log::debug!(
                target: "pipeline",
                "{} ran in {:.2?} ({})",
                p.name(),
                started.elapsed(),
                if pass_changed { "changed" } else { "no change" },
            );
            changed |= pass_changed;
        }
        stage_changed |= changed;
        iters += 1;
        if stage.repeat_until.is_none() || !changed {
            return Ok(stage_changed);
        }
        if iters >= MAX_FIXPOINT_ITERS {
            return Err(nonconvergence_error(&stage.name, None));
        }
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

    /// The functions that currently hold at least one fixpoint mark — the only ones
    /// a module stage could wrongly let us skip, so the only ones worth fingerprinting.
    fn clean_function_ids(&self) -> HashSet<FunctionId> {
        self.clean.keys().map(|(f, _)| *f).collect()
    }

    /// Fingerprint every currently-clean function before a module stage runs.
    ///
    /// A module pass may touch *any* function, and the interprocedural milestones
    /// (`bind_args`, `seed_clobbers`, `summaries`) rewrite IR while reporting
    /// `Ok(false)` — so we cannot trust the stage's change flag. Instead we diff a
    /// cheap per-function fingerprint across the stage and invalidate only the
    /// functions that actually moved (a module stage *can* dirty everything, but
    /// usually rewrites a handful of call sites). Cost scales with the marks held,
    /// not the program size.
    fn snapshot_clean(&self, ctx: &Context) -> Vec<(FunctionId, u64)> {
        self.clean_function_ids()
            .into_iter()
            .map(|f| (f, function_fingerprint(ctx, f)))
            .collect()
    }

    /// Invalidate exactly the functions whose fingerprint changed since
    /// [`snapshot_clean`](Self::snapshot_clean).
    fn invalidate_changed(&mut self, ctx: &Context, before: Vec<(FunctionId, u64)>) {
        for (f, old) in before {
            if function_fingerprint(ctx, f) != old {
                self.mark_dirty(f);
            }
        }
    }
}

/// A fingerprint of one function's rendered body, used to detect whether a
/// module stage modified it. Rendering the IR captures operand rewrites,
/// insertions/removals, retypes, and CFG edits; block order is address-sorted
/// (deterministic) so an unchanged function fingerprints identically across a
/// stage. The render is streamed straight into the hasher — no intermediate
/// `String` of the whole body is ever built, so the cost is the formatting
/// walk alone.
pub(super) fn function_fingerprint(ctx: &Context, fun_id: FunctionId) -> u64 {
    fingerprint_display(FunctionRef::from_id(ctx, fun_id))
}

/// Hash a renderable value (a `FunctionRef` over *any* host) by streaming its
/// `Display` output straight into the hasher — no intermediate `String`. Used to
/// fingerprint a function whether it is live in the module ([`function_fingerprint`])
/// or checked out of it (the per-function fixpoint tracer, which renders the body
/// through its `BodyMut` host).
pub(super) fn fingerprint_display(d: impl std::fmt::Display) -> u64 {
    use std::fmt::Write as _;
    use std::hash::Hasher;

    struct HashWriter(std::collections::hash_map::DefaultHasher);
    impl std::fmt::Write for HashWriter {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            self.0.write(s.as_bytes());
            Ok(())
        }
    }

    let mut writer = HashWriter(std::collections::hash_map::DefaultHasher::new());
    write!(writer, "{d}").expect("writing into a hasher cannot fail");
    writer.0.finish()
}

/// Whole-program structural fingerprint: the address-sorted list of per-function
/// [`function_fingerprint`]s. Stable across a stage that changes nothing, and — like
/// its per-function basis — id-churn tolerant (it hashes rendered IR, not instruction
/// ids), so a module stage that oscillates back to a prior whole-program state
/// fingerprints identically. Used only while a stage is being traced for
/// non-convergence, so the O(program) render cost is off the common path.
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
async fn run_function_stage(
    ctx: &mut Context<'_>,
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
    round: usize,
    progress: &mut impl ProgressSink,
) -> Result<HashSet<FunctionId>, String> {
    run_function_stage_with_threads(
        ctx,
        env,
        stage,
        passes,
        previous_dirty,
        restrict,
        cache,
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
    let total = fun_ids.len();
    let stage_name: std::sync::Arc<str> = stage.name.as_str().into();

    // Function-major stages run each pass thousands of times, so timing is
    // aggregated per pass over the whole stage rather than logged per call.
    let mut elapsed: HashMap<&'static str, (std::time::Duration, usize, usize)> =
        HashMap::default();

    let mut dirty = HashSet::default();

    // Every function pass is driven over a body borrowed in place from the bodies
    // registry (`run_one_function`) — the shape Stage 6 runs on worker threads.

    /*
    /*
    /*
    for (index, fun_id) in fun_ids.into_iter().enumerate() {
        let function: std::sync::Arc<str> = FunctionRef::from_id(ctx, fun_id).name().into();
        /*
        let mut iters = 0;
        let mut function_changed = false;
        // Engaged only once this function's fixpoint is clearly struggling
        // (`FIXPOINT_WATCH_ITERS`); traces which passes keep moving the IR and
        // flags a proven cycle. Cheap on the common path (never allocates until a
        // pass changes the IR past the threshold).
        let mut tracer = FixpointTracer::default();
        let tracer_label = format!("stage {} fn {function}", stage.name);
        'fixpoint: loop {
            let watching = stage.repeat_until.is_some() && iters >= FIXPOINT_WATCH_ITERS;
            let mut changed = false;
            for p in passes {
                // Skip a pass that already reached a fixpoint on this function and
                // has not been dirtied since (by an earlier pass this iteration, a
                // prior stage, or — via `invalidate_all` — a module pass).
                if cache.is_clean(fun_id, p.name()) {
                    continue;
                }
                progress.report(PipelineProgress::FunctionPass {
                    round,
                    stage: stage_name.clone(),
                    function: function.clone(),
                    index: index + 1,
                    total,
                    pass: p.name(),
                });
                let _scope = qcode::pass_scope::enter(p.name());
                #[cfg(not(target_arch = "wasm32"))]
                let started = std::time::Instant::now();
                let pass_changed = p
                    .run(ctx, fun_id, env)
                    .map_err(|e| format!("{}: {e}", p.name()))?;
                if pass_changed {
                    // The function moved: invalidate every pass's fixpoint mark.
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
                // Between-pass invariant check (opt-in via `QCODE_VERIFY`).
                crate::verify::verify_after(ctx, p.name());
                // Trace the fingerprint after every pass that moved the IR, so a
                // struggling fixpoint reveals which passes keep fighting and whether
                // the IR is truly cycling (a recurring fingerprint) vs. slowly churning.
                if watching && pass_changed {
                    let fp = function_fingerprint(ctx, fun_id);
                    if tracer.observe(&tracer_label, iters + 1, p.name(), fp) {
                        // A proven cycle cannot converge; leave this function at
                        // the recurring state and move on to the next one instead
                        // of spinning to the iteration cap.
                        log::warn!(
                            target: "pipeline::fixpoint",
                            "stage {} fn {function}: stopping best-effort on the proven \
                             cycle at iteration {}",
                            stage.name,
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
            if stage.repeat_until.is_none() || !changed {
                break;
            }
            if iters >= MAX_FIXPOINT_ITERS {
                log::warn!(
                    target: "pipeline::fixpoint",
                    "stage {} fn {function} hit the {MAX_FIXPOINT_ITERS}-iteration cap; \
                     see the `pipeline::fixpoint` trace above for the fighting passes",
                    stage.name,
                );
                return Err(nonconvergence_error(&stage.name, Some(&function)));
            }
        }
        */
        let function_changed = if all_v2 {
            // Check the function out, run its whole pass fixpoint on the owned body,
            // then reinstall it and replay any buffered effects — the sequential
            // form of the parallel check-in protocol.
            let fun = ctx.checkout_function(fun_id);
            let mut body = FunctionBody::new(fun_id, fun, Vec::new());
            let function_changed = {
                let view = ModuleView::new(ctx, env);
                run_one_function(
    */
    */
    // Function-minting reservations (`PARALLEL_PASSES.md` ruling 3): when a stage
    // contains a minting pass, reserve `MINT_RESERVE` ids per worklist function up
    // front, in worklist order — the identical discipline on both lanes, so minted
    // ids are byte-identical between the sequential and parallel runs. Unused ids
    // are collected per function at the barrier and recycled (in worklist order) at
    // the end of the stage, for the next stage to consume.
    let mut reservations: HashMap<FunctionId, Vec<FunctionId>> = HashMap::default();
    let mut leftover: HashMap<FunctionId, Vec<FunctionId>> = HashMap::default();
    if passes.iter().any(|p| p.mints()) {
        for &fun_id in &fun_ids {
            let ids = pool.reserve(ctx, MINT_RESERVE);
            reservations.insert(fun_id, ids);
        }
    }

    */
    // Parallelize the stage across threads once the worklist is worth the fan-out
    // cost. Strict IR locality (context-split ruling 2) is established at the
    // optimization entry and every discovery round, so every function body is
    // closed over its own blocks — there are no cross-function edges to entangle
    // two checked-out bodies, and every function is parallel-eligible. Tiny
    // worklists, `QCODE_THREADS=1`, and wasm fall through to the sequential loop
    // below, byte-for-byte identical.
    let parallel_set: HashSet<FunctionId> = if threads > 1 && fun_ids.len() >= PARALLEL_THRESHOLD {
        run_stage_parallel(
            ctx,
            env,
            stage,
            passes,
            &fun_ids,
            &stage_name,
            total,
            round,
            cache,
            &mut elapsed,
            &mut dirty,
            threads,
            progress,
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
                    &mut elapsed,
                    &stage.name,
                    &function,
                    stage.repeat_until.is_some(),
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
            let installed = install_minted(ctx, &stage.name, outcome.minted)?;
            let patched = resolve_minted_callees(ctx, &stage.name, fun_id, &installed)?;
            replay_rename(ctx, &stage.name, fun_id, outcome.rename)?;
            // Minted functions are new work for downstream `only_dirty` stages.
            dirty.extend(installed);
            // Opt-in `QCODE_VERIFY` check once the split borrow has ended — the body
            // is reachable through `ctx` again — pinning any invariant break to this
            // stage. A no-op unless `QCODE_VERIFY` is set.
            crate::verify::verify_after(ctx, &stage.name);
            if outcome.changed || patched {
                dirty.insert(fun_id);
            }
        }

        // Cooperative yield point: one per function (never per pass). The default
        // sink returns immediately; a wasm sink may suspend until the next frame
        // and can cancel, in which case we stop early and return the work done so far.
        if let YieldSignal::Cancelled = progress.yield_now().await {
            return Ok(dirty);
        }
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
    passes: &[Box<dyn DynFunctionPass>],
    body: &mut FunctionBody<'str>,
    cx: ContextView<'_, 'str>,
    cache: &mut FixpointCache,
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
                .run_checked(body, cx, &mut next_minted)
                .map_err(|e| format!("{}: {e}", p.name()))?;
            // Minting is an observable stage mutation even when a pass forgot to
            // set its body-change bit. Keep the producer dirty and continue a
            // requested fixpoint rather than caching it as clean while publishing
            // new functions at the barrier.
            let pass_changed = outcome.changed || !outcome.minted.is_empty();
            if outcome.rename.is_some() {
                agg_rename = outcome.rename;
            }
            agg_minted.extend(outcome.minted);
            if pass_changed {
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
    passes: &[Box<dyn DynFunctionPass>],
    fun_ids: &[FunctionId],
    stage_name: &std::sync::Arc<str>,
    total: usize,
    round: usize,
    cache: &mut FixpointCache,
    elapsed: &mut HashMap<&'static str, (std::time::Duration, usize, usize)>,
    dirty: &mut HashSet<FunctionId>,
    threads: usize,
    progress: &mut impl FnMut(PipelineProgress),
) -> Result<(), String> {
    // 1. Snapshot per-function display names before the split borrow freezes the
    //    context.
    let mut metas: Vec<FnMeta> = Vec::with_capacity(fun_ids.len());
    for &fun_id in fun_ids {
        let name: std::sync::Arc<str> = FunctionRef::from_id(ctx, fun_id).name().into();
        metas.push((fun_id, name));
    }

    let chunk_size = fun_ids.len().div_ceil(threads).max(1);
    let repeat_until = stage.repeat_until.is_some();
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
            .enumerate()
            .map(|(index, ((fun_id, name), slot))| ParallelEntry {
                index,
                fun_id,
                name,
                body: slot,
                outcome: Outcome::default(),
            })
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
                    let handle = scope.spawn(move || -> Result<WorkerOutput, String> {
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
                    });
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
                    .map(|h| h.join().expect("function-pass worker panicked"))
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
            .map(|e| (e.fun_id, e.outcome))
            .collect::<Vec<_>>()
    };

    // 6. Barrier (master, worklist order): install and resolve minted callees,
    //    apply the returned self-rename, and record dirtiness. The bodies were
    //    mutated in place, so there is nothing to reinstall.
    for (fun_id, outcome) in results {
        let installed = install_minted(ctx, &stage.name, outcome.minted)?;
        let patched = resolve_minted_callees(ctx, &stage.name, fun_id, &installed)?;
        replay_rename(ctx, &stage.name, fun_id, outcome.rename)?;
        dirty.extend(installed);
        // Opt-in `QCODE_VERIFY` check once the split borrow has ended. A no-op
        // unless enabled.
        crate::verify::verify_after(ctx, &stage.name);
        if outcome.changed || patched {
            dirty.insert(fun_id);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FunctionPass, FunctionPassAdapter};
    use qcode::{
        context::Context,
        value::{BasicBlock, FunctionBody, FunctionKind},
    };

    fn run_function_only_pipeline_with_threads(
        pipeline: &Pipeline,
        ctx: &mut Context,
        threads: usize,
    ) {
        let env = PipelineEnv::headless(ctx);
        let mut dirty = Some(HashSet::default());
        let mut cache = FixpointCache::default();
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
            Ok(Outcome {
                changed: false,
                rename: None,
                minted,
            })
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
        let outcome = {
            let (bodies, view) = ctx.split(&env);
            run_one_function(
                &passes,
                &mut bodies[owner],
                view,
                &mut cache,
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
