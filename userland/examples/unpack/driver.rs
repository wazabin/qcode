//! Loading, running and reporting: the glue the CLI and the tests share.
//!
//! The process itself is [`qcode_userland`]'s: [`Process::new`] maps the ELF,
//! builds the SysV stack, and services system calls on the host, and
//! [`Process::run`] drives the machine. All this module adds is the three
//! bounded state spaces of [`super::layout`] and the two hooks of
//! [`super::hooks`], installed on `process.vm()` before the first
//! instruction runs.
//!
//! Because neither hook ever stops the machine, the run is one call: no
//! resume loop, no host round trip per block, and nothing for
//! `Task::handle_interrupt` to mistake for a system call.

use std::{cell::RefCell, rc::Rc};

use qcode::space::SpaceId;
use qcode_userland::fs::Stdio;
use qcode_userland::{Config, Process, ProcessExit};

use super::{
    hooks::{self, EntryHook, LogEntry, ProvenanceHook, Recorder},
    layout::{self, Layout},
};

/// How a run is configured.
#[derive(Debug, Clone)]
pub struct Options {
    /// Install the Cranelift JIT block executor.
    pub jit: bool,
    /// Maximum number of p-code operations.
    pub budget: u64,
    /// Install the provenance and first-entry hooks.
    pub hooks: bool,
    /// Record the observed control-flow edge into each block as well.
    pub edges: bool,
    /// Arguments after the program name; the program name itself is always
    /// `argv[0]`.
    pub args: Vec<String>,
    /// What the guest reads on its standard input.
    pub stdin: Option<Vec<u8>>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            jit: false,
            budget: 5_000_000_000,
            hooks: true,
            edges: false,
            args: Vec::new(),
            stdin: None,
        }
    }
}

/// The three state spaces the hooks share, once the machine has made them.
#[derive(Debug, Clone, Copy)]
pub struct Spaces {
    pub shadow: SpaceId,
    pub entries: SpaceId,
    pub visited: SpaceId,
}

/// What a run produced.
#[derive(Debug, Clone)]
pub struct Outcome {
    /// The image that was run, as it was named on the command line.
    pub path: String,
    /// How the run was configured.
    pub options: Options,
    /// The status the guest passed to `exit`/`exit_group`, if it got there.
    pub exit_status: Option<i32>,
    /// Bytes the guest wrote to fd 1.
    pub stdout: Vec<u8>,
    /// Bytes the guest wrote to fd 2.
    pub stderr: Vec<u8>,
    /// P-code operations retired.
    pub steps: u64,
    /// Why the machine stopped, in one line.
    pub stop_reason: String,
    /// Whether the run ended anywhere other than the guest's own `exit`.
    pub crashed: bool,
    /// Where the image landed, and what provenance therefore covers.
    pub layout: Layout,
    /// The entry point the run started at.
    pub entry: u64,
    /// The spaces to read the shadow and the log out of, when hooks ran.
    pub spaces: Option<Spaces>,
    /// What the hooks knew that the spaces did not; empty with `--no-hooks`.
    pub recorder: Recorder,
    /// The blocks that ran, once each, in the order they first ran.
    pub log: Vec<LogEntry>,
    /// Lifted blocks the machine threw away because the guest wrote over
    /// their bytes: this branch's self-modifying-code counter.
    pub evicted: u64,
    /// Blocks folded into a predecessor as guest basic blocks were
    /// discovered.
    pub absorbed: u64,
    /// Block bodies run by the JIT rather than interpreted.
    pub native_bodies: u64,
}

/// Loads `path`, runs it, and returns what the run produced.
pub fn run(path: &str, options: &Options) -> Result<Outcome, String> {
    run_keeping(path, options).map(|(_, outcome)| outcome)
}

/// [`run`], handing the process back as well.
///
/// The final context and the final state spaces are what [`super::graph`]
/// harvests, so a caller that wants an artifact needs the machine and not
/// just the outcome.
pub fn run_keeping(path: &str, options: &Options) -> Result<(Process, Outcome), String> {
    let image = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let config = Config {
        argv: std::iter::once(path.to_owned())
            .chain(options.args.iter().cloned())
            .collect(),
        envp: Vec::new(),
        jit: options.jit,
        trace: false,
        root: None,
        stdio: Stdio::Captured,
        exe_path: path.to_owned(),
    };
    let mut process =
        Process::new(&image, config).map_err(|e| format!("cannot load {path}: {e}"))?;
    if let Some(stdin) = &options.stdin {
        process.files_mut().set_stdin(stdin.clone());
    }

    let loaded = process.image();
    let image_lo = loaded
        .segments
        .iter()
        .map(|segment| segment.start)
        .min()
        .unwrap_or(loaded.base);
    let image_hi = loaded
        .segments
        .iter()
        .map(|segment| segment.end)
        .max()
        .unwrap_or(image_lo);
    let entry = loaded.entry;
    let layout = Layout::new(image_lo, image_hi)
        .map_err(|e| format!("cannot track the provenance of {path}: {e}"))?;

    let recorder = Rc::new(RefCell::new(Recorder::default()));
    let spaces = if options.hooks {
        let spaces = make_spaces(&mut process)?;
        let vm = process.vm();
        // The provenance hook first, so that it sees a block before any
        // other rewrite reshapes it; neither hook splits a block, so the
        // order is a convention rather than a constraint.
        vm.add_hook(ProvenanceHook::new(Rc::clone(&recorder), layout));
        vm.add_hook(EntryHook::new(Rc::clone(&recorder), options.edges));
        Some(spaces)
    } else {
        None
    };

    let exit = process.run(options.budget);
    let steps = process.steps();
    let log = match spaces {
        Some(spaces) => {
            hooks::read_log(process.vm().memory().flat(), spaces.entries, options.edges)
        }
        None => Vec::new(),
    };
    let stats = process.vm().stats.clone();

    let (exit_status, stop_reason, crashed) = match &exit {
        ProcessExit::Exited(status) => (Some(*status), "exit".to_owned(), false),
        ProcessExit::Crashed(crash) => (None, format!("crashed: {crash}"), true),
        ProcessExit::Budget => (
            None,
            format!("budget of {} operations exhausted", options.budget),
            true,
        ),
    };

    let outcome = Outcome {
        path: path.to_owned(),
        options: options.clone(),
        exit_status,
        stdout: process.files().stdout().to_vec(),
        stderr: process.files().stderr().to_vec(),
        steps,
        stop_reason,
        crashed,
        layout,
        entry,
        spaces,
        recorder: recorder.borrow().clone(),
        log,
        evicted: stats.evicted,
        absorbed: stats.absorbed,
        native_bodies: stats.native_bodies,
    };
    Ok((process, outcome))
}

/// Creates the three bounded state spaces the hooks address.
///
/// Before the hooks, on purpose: `Emitter::state_space` finds a space by
/// name and makes an *unbounded* one if there is none, and compiled code
/// reaches an unbounded space only at constant addresses.
fn make_spaces(process: &mut Process) -> Result<Spaces, String> {
    let made = |name: &str, made: Result<SpaceId, qcode_vm::StateSpaceError>| {
        made.map_err(|e| format!("cannot make the hook space `{name}`: {e}"))
    };
    let shadow = made(
        layout::SHADOW_SPACE,
        process
            .vm()
            .state_space(layout::SHADOW_SPACE, layout::SHADOW_LEN),
    )?;
    let entries = made(
        layout::ENTRIES_SPACE,
        process
            .vm()
            .state_space(layout::ENTRIES_SPACE, layout::ENTRIES_LEN),
    )?;
    let visited = made(
        layout::VISITED_SPACE,
        process
            .vm()
            .state_space(layout::VISITED_SPACE, layout::VISITED_LEN),
    )?;
    Ok(Spaces {
        shadow,
        entries,
        visited,
    })
}
