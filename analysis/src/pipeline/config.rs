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
    DynFunctionPass, DynPass, PipelineEnv, RegisteredPass, known_pass_names, make_pass,
};
use super::{PipelineProgress, ProgressSink, YieldSignal};

/// The canonical default pipeline, compiled into the binary. Used by
/// `analyze_default` and as the GUI's starting pipeline.
pub const DEFAULT_PIPELINE_TOML: &str = include_str!("default_pipeline.toml");
#[cfg(not(target_arch = "wasm32"))]
const DEFAULT_PIPELINE_FILE: &str = "default.toml";

/// Guard against a `repeat_until` stage that never converges.
const MAX_FIXPOINT_ITERS: usize = 100;

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
        })
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
                    if reaches_discovery_pass {
                        return Ok(());
                    }
                }
                StagePasses::Module(_) => {}
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
        }
        Ok(())
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
            let pass_changed = p.run(ctx, env).map_err(|e| format!("{}: {e}", p.name()))?;
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
            return Err(format!(
                "stage \"{}\" did not converge after {MAX_FIXPOINT_ITERS} iterations",
                stage.name
            ));
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
            return Err(format!(
                "stage \"{}\" did not converge after {MAX_FIXPOINT_ITERS} iterations",
                stage.name
            ));
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

/// A cheap structural fingerprint of one function's body, used to detect whether a
/// module stage modified it. Rendering the IR captures operand rewrites,
/// insertions/removals, and CFG edits; block order is address-sorted (deterministic)
/// so an unchanged function fingerprints identically across a stage.
pub(super) fn function_fingerprint(ctx: &Context, fun_id: FunctionId) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    FunctionRef::from_id(ctx, fun_id)
        .to_string()
        .hash(&mut hasher);
    hasher.finish()
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
    dump_stage_inputs(ctx, &stage.dump, &stage.name);
    let fun_ids: Vec<FunctionId> = ctx
        .functions()
        .filter(|f| stage.include_external || !f.is_external())
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

    for (index, fun_id) in fun_ids.into_iter().enumerate() {
        let function: std::sync::Arc<str> = FunctionRef::from_id(ctx, fun_id).name().into();
        let mut iters = 0;
        let mut function_changed = false;
        loop {
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
                changed |= pass_changed;
            }
            function_changed |= changed;
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
        if function_changed {
            dirty.insert(fun_id);
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
