//! Sliced-vs-full differential harness (Milestone 3, step 1).
//!
//! Milestone 3 replaces the whole-program "clone `clean`, re-optimize
//! everything, every round" driver with an *incremental* one that restores and
//! re-optimizes only the invalidated cone (see
//! `docs/plans/milestone-3-effect-deltas/02-incremental-invalidation.md`). That
//! driver does not exist yet — it is step 5 of the plan's build order, and it is
//! gated on effect persistence, the memory-channel delta, and the type-level
//! write handle.
//!
//! Step 1 lands the *comparison infrastructure* first, so that every later step
//! has something to be checked against. This module provides:
//!
//! - [`ContextDigest`] — a run-independent digest of a post-analysis
//!   [`Context`]: rendered IR per function, plus the [`FunctionEffects`] of
//!   every function, both the `register` and the `memory` channel.
//! - [`assert_analysis_equivalent`] — a diagnosable equality assertion over two
//!   digests, reporting *which* function and *which* channel diverged.
//!
//! ## Why a digest rather than comparing `Context`s directly
//!
//! Two independent analysis runs over the same baseline allocate their arenas
//! independently: `FunctionId`s minted by passes, block/instruction ids, and
//! interned-name ordering may legitimately differ. The existing hard gate on the
//! parallel driver (`parallel_pipeline_matches_sequential_ir`) resolves this the
//! same way — it compares the *rendered* IR, which names values structurally,
//! not by arena index. This module reuses exactly that approach and extends it
//! to effects:
//!
//! - Functions are keyed by `name@address`, not by `FunctionId`. Names are
//!   uniqued at mint time (`Context::get_unique_name`) and the address pins
//!   lifted functions to their entry point, so the pair is a stable identity
//!   across runs.
//! - `VarnodeId`/`SpaceId` inside [`FunctionEffects`] are rendered to their
//!   architectural *names* before comparison. These live in the frozen `Shared`
//!   architecture interners rather than in per-function arenas, so their ids are
//!   in fact stable between runs over one baseline — but rendering them is what
//!   makes a failure readable (`rax` rather than `VarnodeId(41)`), and it keeps
//!   the digest correct if a future run is built from a separately-lifted
//!   context.
//!
//! Nothing here is loose: every field of [`FunctionEffects`] reaches the digest,
//! and the digest compares by exact string equality.

use std::collections::BTreeMap;

use qcode::context::Context;
use qcode::value::function::{
    Footprint, FunctionEffects, MemoryChannelState, RegisterChannelState, WrittenSpacesState,
};

/// A run-independent digest of one function: its rendered body IR (absent for
/// externals, which have an interface but no body) and its effect summary.
#[derive(Clone, PartialEq, Eq)]
pub struct FunctionDigest {
    /// Rendered body IR, or `None` for an external (bodyless) function.
    pub ir: Option<String>,
    /// Rendered `FunctionEffects::register`.
    pub register_effects: String,
    /// Rendered `FunctionEffects::memory`.
    pub memory_effects: String,
}

/// A run-independent digest of a whole post-analysis [`Context`], keyed by a
/// stable per-function identity (`name@address`).
#[derive(Clone, PartialEq, Eq)]
pub struct ContextDigest {
    functions: BTreeMap<String, FunctionDigest>,
}

impl ContextDigest {
    /// Digest every function in `ctx` — bodied and external alike. Effects are
    /// read from the interface registry (which carries externals too); IR is
    /// read from the body registry.
    pub fn of(ctx: &Context<'_>) -> Self {
        let mut functions = BTreeMap::new();
        for func in ctx.functions() {
            let key = function_key(func.name(), func.address());
            // The body and interface registries are held in lockstep under the
            // same `FunctionId`, so every interface — externals included — is
            // reachable from the body iterator. Externals render as a bodyless
            // stub, which is itself part of the compared IR.
            let ir = if func.is_external() {
                None
            } else {
                Some(format!("{func}"))
            };
            let effects = func.effects();
            let digest = FunctionDigest {
                ir,
                register_effects: render_register_channel(ctx, effects),
                memory_effects: render_memory_channel(ctx, effects),
            };
            if let Some(previous) = functions.insert(key.clone(), digest) {
                // Function keys must be unique, otherwise the digest silently
                // compares a different pair of functions on each side. Names are
                // uniqued at mint time, so a collision is a real invariant
                // violation, not something to tolerate.
                assert!(
                    previous == functions[&key],
                    "duplicate function key `{key}` with differing content: the \
                     digest cannot identify functions across runs"
                );
            }
        }
        Self { functions }
    }

    /// The stable keys of every digested function, in sorted order.
    pub fn function_keys(&self) -> impl Iterator<Item = &str> {
        self.functions.keys().map(String::as_str)
    }
}

/// The stable cross-run identity of a function.
fn function_key(name: &str, address: Option<u64>) -> String {
    match address {
        Some(address) => format!("{name}@{address:#x}"),
        None => format!("{name}@none"),
    }
}

/// Render a varnode by its architectural name, falling back to the generated
/// temporary label and finally to the raw id.
fn render_varnode(ctx: &Context<'_>, id: qcode::value::VarnodeId) -> String {
    let varnode = qcode::value::Varnode::from_id(&ctx.shared, id);
    match (varnode.name(), varnode.label()) {
        (Some(name), _) => name.to_string(),
        (None, Some(label)) => format!("v{label}"),
        (None, None) => format!("varnode:{id}"),
    }
}

/// Render a space by its architectural name, falling back to the raw id.
fn render_space(ctx: &Context<'_>, id: qcode::space::SpaceId) -> String {
    match ctx.shared.space(id).name.as_deref() {
        Some(name) => name.to_string(),
        None => format!("space:{id}"),
    }
}

/// Render the register channel exhaustively — every variant, and every field of
/// every variant, so a divergence anywhere in the summary is observable.
fn render_register_channel(ctx: &Context<'_>, effects: &FunctionEffects) -> String {
    let render_all = |ids: &[qcode::value::VarnodeId]| {
        ids.iter()
            .map(|id| render_varnode(ctx, *id))
            .collect::<Vec<_>>()
            .join(", ")
    };
    match &effects.register {
        RegisterChannelState::Unsolved => "unsolved".to_string(),
        RegisterChannelState::Top => "top".to_string(),
        RegisterChannelState::Solved(sets) => format!(
            "solved(reads: [{}], writes: [{}])",
            render_all(&sets.reads),
            render_all(&sets.writes),
        ),
        RegisterChannelState::Materialized(map) => format!(
            "materialized(inputs: [{}], outputs: [{}], returns: {})",
            render_all(&map.inputs),
            render_all(&map.outputs),
            map.returns,
        ),
    }
}

/// Render the memory channel exhaustively: the coarse written-space tri-state
/// and the precise RAM footprint persisted alongside it.
///
/// This destructures [`MemoryChannelState`] deliberately, so that adding a field
/// to the channel is a compile error here rather than a silently unchecked
/// component of the digest.
fn render_memory_channel(ctx: &Context<'_>, effects: &FunctionEffects) -> String {
    let MemoryChannelState {
        coarse,
        precise,
        materialized,
    } = &effects.memory;
    format!(
        "{}; {}; {}",
        render_coarse_spaces(ctx, coarse),
        render_footprint(precise.as_ref()),
        render_memory_interface(materialized.as_ref()),
    )
}

/// Render the materialized memory interface, or `unmaterialized`.
fn render_memory_interface(map: Option<&qcode::value::MemoryInterfaceMap>) -> String {
    let Some(map) = map else {
        return "unmaterialized".to_string();
    };
    let slots = |slots: &[qcode::value::InterfaceSlot]| {
        slots
            .iter()
            .map(render_interface_slot)
            .collect::<Vec<_>>()
            .join(",")
    };
    format!("in[{}] out[{}]", slots(&map.inputs), slots(&map.outputs))
}

fn render_interface_slot(slot: &qcode::value::InterfaceSlot) -> String {
    let qcode::value::InterfaceSlot { base, offset, size } = *slot;
    match base {
        qcode::value::SlotBase::Arg(i) => format!("[arg{i}+{offset}:{size}]"),
        qcode::value::SlotBase::Global(addr) => format!("[{addr:#x}+{offset}:{size}]"),
        qcode::value::SlotBase::Unmappable => format!("[?+{offset}:{size}]"),
    }
}

/// Render the coarse written-space tri-state.
fn render_coarse_spaces(ctx: &Context<'_>, coarse: &WrittenSpacesState) -> String {
    match coarse {
        WrittenSpacesState::Unstamped => "unstamped".to_string(),
        WrittenSpacesState::Unbounded => "unbounded".to_string(),
        WrittenSpacesState::Bounded(spaces) => {
            let mut names: Vec<_> = spaces.iter().map(|id| render_space(ctx, *id)).collect();
            // The channel documents `Bounded` as a *set*; sorting by rendered
            // name keeps the digest independent of the solve's insertion order.
            names.sort();
            format!("bounded([{}])", names.join(", "))
        }
    }
}

/// Render the precise RAM footprint, or `top` for the inexpressible (⊤) case.
///
/// No id rendering is needed: every [`RamBase`] variant carries a plain scalar
/// (an argument index, a frame offset, an absolute address) rather than an
/// arena id, so the rendering is already run-independent. Nor is any sorting
/// needed — [`Footprint`]'s components are `BTreeSet`s precisely so their
/// iteration order is deterministic; if it were not, this digest would catch it,
/// which is the point.
fn render_footprint(precise: Option<&Footprint>) -> String {
    let Some(footprint) = precise else {
        return "footprint(top)".to_string();
    };
    let render = |direction: &str, locations: &qcode::value::RamLocations| {
        let fields = locations
            .fields
            .iter()
            .map(|f| format!("{direction}{:?}+{}:{}", f.base, f.offset, f.size));
        let regions = locations
            .regions
            .iter()
            .map(|r| format!("{direction}{:?}+[{}, {})", r.base, r.lo, r.hi));
        let objects = locations
            .objects
            .iter()
            .map(|o| format!("{direction}{:?}+*", o.base));
        fields.chain(regions).chain(objects).collect::<Vec<_>>()
    };
    let entries = render("r", &footprint.reads)
        .into_iter()
        .chain(render("w", &footprint.writes))
        .collect::<Vec<_>>();
    format!("footprint([{}])", entries.join(", "))
}

/// Assert that two post-analysis contexts are equivalent in final IR and in
/// `FunctionEffects` (both channels), reporting the first divergence precisely.
///
/// `label` names the subject under test (typically the fixture); `full_name` and
/// `sliced_name` name the two drivers, so the failure says which side is which.
pub fn assert_analysis_equivalent(
    label: &str,
    full_name: &str,
    full: &ContextDigest,
    sliced_name: &str,
    sliced: &ContextDigest,
) {
    // 1. The function sets themselves.
    let only_full: Vec<_> = full
        .function_keys()
        .filter(|k| !sliced.functions.contains_key(*k))
        .collect();
    let only_sliced: Vec<_> = sliced
        .function_keys()
        .filter(|k| !full.functions.contains_key(*k))
        .collect();
    assert!(
        only_full.is_empty() && only_sliced.is_empty(),
        "`{label}`: function sets diverged\n  only in {full_name}: {only_full:?}\n  \
         only in {sliced_name}: {only_sliced:?}",
    );

    // 2. Per function, per channel.
    for (key, full_fn) in &full.functions {
        let sliced_fn = &sliced.functions[key];

        if full_fn.register_effects != sliced_fn.register_effects {
            panic!(
                "`{label}`: function `{key}`: register effect channel diverged\n  \
                 {full_name}: {}\n  {sliced_name}: {}",
                full_fn.register_effects, sliced_fn.register_effects,
            );
        }
        if full_fn.memory_effects != sliced_fn.memory_effects {
            panic!(
                "`{label}`: function `{key}`: memory effect channel diverged\n  \
                 {full_name}: {}\n  {sliced_name}: {}",
                full_fn.memory_effects, sliced_fn.memory_effects,
            );
        }
        if full_fn.ir != sliced_fn.ir {
            panic!(
                "`{label}`: function `{key}`: IR diverged\n{}",
                describe_ir_divergence(
                    full_name,
                    full_fn.ir.as_deref(),
                    sliced_name,
                    sliced_fn.ir.as_deref(),
                ),
            );
        }
    }
}

/// A readable account of *where* two rendered function bodies differ: the first
/// differing line with its neighbours, rather than two full dumps.
fn describe_ir_divergence(
    full_name: &str,
    full: Option<&str>,
    sliced_name: &str,
    sliced: Option<&str>,
) -> String {
    let (full, sliced) = match (full, sliced) {
        (Some(full), Some(sliced)) => (full, sliced),
        (full, sliced) => {
            return format!(
                "  one side is bodyless: {full_name} has body: {}, {sliced_name} has body: {}",
                full.is_some(),
                sliced.is_some(),
            );
        }
    };
    let full_lines: Vec<_> = full.lines().collect();
    let sliced_lines: Vec<_> = sliced.lines().collect();
    for (i, (a, b)) in full_lines.iter().zip(sliced_lines.iter()).enumerate() {
        if a != b {
            let context_start = i.saturating_sub(3);
            let preceding = full_lines[context_start..i].join("\n");
            return format!(
                "  first divergence at line {}\n{preceding}\n  {full_name:>12}: {a}\n  \
                 {sliced_name:>12}: {b}",
                i + 1,
            );
        }
    }
    format!(
        "  bodies agree on the first {} lines but differ in length: {full_name} has {}, \
         {sliced_name} has {}",
        full_lines.len().min(sliced_lines.len()),
        full_lines.len(),
        sliced_lines.len(),
    )
}

#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
mod tests {
    use super::*;
    use harbinger::arch::x64;
    use harbinger::format::elf::ElfBinary;

    fn load_fixture_elf(name: &str) -> ElfBinary {
        let path = std::path::PathBuf::from(env!("FIXTURES_DIR")).join(name);
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        ElfBinary::parse(&bytes).unwrap_or_else(|e| panic!("failed to parse {name} ELF: {e}"))
    }

    /// Total precise-footprint entries persisted across a digested module.
    ///
    /// The memory channel's *coarse* half is only a set of spaces; the precise
    /// footprint is what carries address granularity, and it is the half a
    /// step-3 memory delta is meant to compare. A differential run over a module
    /// whose footprints are all empty therefore verifies nothing about that
    /// delta while still passing — the exact blindness this counter exists to
    /// rule out.
    fn footprint_entries(ctx: &Context<'_>) -> usize {
        ctx.functions()
            .filter_map(|f| f.effects().memory.precise.as_ref())
            .map(Footprint::len)
            .sum()
    }

    /// The verification harness for Milestone 3: the incremental ("sliced")
    /// driver must produce the same final IR *and* the same `FunctionEffects`
    /// per function as today's full whole-program driver.
    ///
    /// The sliced side is currently an explicit alias for the full driver — see
    /// [`run_sliced`] — so this passes trivially today. **Step 5 of
    /// `docs/plans/milestone-3-effect-deltas/02-incremental-invalidation.md`
    /// (the incremental driver: restore-from-clean, bidirectional cone,
    /// delta-driven propagation, budgeted narrowing) replaces that one line**,
    /// at which point this test becomes the real gate. Steps 2–4 are inert and
    /// must keep it green as they land.
    ///
    /// Modelled on `parallel_pipeline_matches_sequential_ir` (the equivalent
    /// hard gate on the parallel function-pass driver), and comparing IR the
    /// same way — rendered, not by arena id — plus effects, which that test does
    /// not cover.
    #[test]
    fn sliced_pipeline_matches_full_ir_and_effects() {
        let fixtures = [
            "hello",
            "arithmetic_mix",
            "branch_table",
            "jump_table",
            "loops_accumulate",
            "recursive_gcd",
            "struct_copy",
            "function_pointers",
            "pointer_array",
            "memory_aliasing",
        ];
        for fixture in fixtures {
            let elf = if fixture.starts_with('/') {
                let bytes = std::fs::read(fixture).unwrap();
                ElfBinary::parse(&bytes).unwrap()
            } else {
                load_fixture_elf(fixture)
            };
            let mut disasm = x64::Disassembler::from_binary(elf);
            let mut ctx = x64::make_context();
            disasm.lift_binary(&mut ctx);
            let cfg = harbinger::arch::arch_config(&ctx).unwrap();
            let binary = disasm.binary_handle();

            let full = run_full(&ctx, &cfg, binary.clone());
            let sliced = run_sliced(&ctx, &cfg, binary);

            assert_analysis_equivalent(
                fixture,
                "full",
                &ContextDigest::of(&full),
                "sliced",
                &ContextDigest::of(&sliced),
            );
        }
    }

    /// The full whole-program driver: clone the clean baseline and re-optimize
    /// every function, every round.
    fn run_full<'s>(
        baseline: &Context<'s>,
        cfg: &qcode_analysis::ArchConfig,
        binary: qcode_analysis::pipeline::BinaryHandle,
    ) -> Context<'s> {
        qcode_analysis::analyze_default(baseline, cfg, Some(binary))
    }

    /// The incremental ("sliced") driver.
    ///
    /// STEP 5 SEAM: today this is deliberately an alias for [`run_full`], so the
    /// harness lands and passes ahead of the driver it exists to verify.
    /// Swapping in the real incremental driver is a one-line change here.
    fn run_sliced<'s>(
        baseline: &Context<'s>,
        cfg: &qcode_analysis::ArchConfig,
        binary: qcode_analysis::pipeline::BinaryHandle,
    ) -> Context<'s> {
        run_full(baseline, cfg, binary)
    }

    /// A system binary used as a differential subject with *real* precise
    /// footprints.
    ///
    /// **Temporary, and deliberately non-hermetic.** Every fixture in
    /// [`sliced_pipeline_matches_full_ir_and_effects`] is a single-translation-unit
    /// C program that the pipeline promotes to nothing: measured across all ten,
    /// the persisted precise footprint is empty for *every* function. That makes
    /// them structurally incapable of verifying a memory-channel delta (step 3),
    /// even though they pass. `/usr/bin/ls` yields 41 functions with non-empty
    /// footprints and 81 entries, essentially all via `external_leaf`'s argmem
    /// objects.
    ///
    /// Non-hermeticity is bounded: both sides analyze the *same bytes in the same
    /// process*, so a different `ls` cannot produce a spurious divergence — it
    /// changes only which code is exercised. The test skips when the binary is
    /// absent rather than failing.
    ///
    /// Replace with a purpose-built fixture once one exists that produces
    /// non-empty footprints. Note that a hermetic fixture exercising the *bodied*
    /// path may not help: `extract_footprint` produced a non-empty footprint for
    /// zero production functions in measurement, because argpromote's purpose is
    /// to erase outward memory effects. See the design doc's step-2 findings.
    const SYSTEM_BINARY: &str = "/usr/bin/ls";

    #[test]
    fn sliced_pipeline_matches_full_on_system_binary() {
        let Ok(bytes) = std::fs::read(SYSTEM_BINARY) else {
            eprintln!("skipping: {SYSTEM_BINARY} not present");
            return;
        };
        let Ok(elf) = ElfBinary::parse(&bytes) else {
            eprintln!("skipping: {SYSTEM_BINARY} is not a parseable ELF");
            return;
        };

        let mut disasm = x64::Disassembler::from_binary(elf);
        let mut ctx = x64::make_context();
        disasm.lift_binary(&mut ctx);
        let cfg = harbinger::arch::arch_config(&ctx).unwrap();
        let binary = disasm.binary_handle();

        let full = run_full(&ctx, &cfg, binary.clone());
        let sliced = run_sliced(&ctx, &cfg, binary);

        // Guard the reason this subject exists. If a future change empties the
        // persisted footprints, this test would keep passing while silently
        // verifying nothing about the memory channel — fail loudly instead.
        let entries = footprint_entries(&full);
        assert!(
            entries > 0,
            "{SYSTEM_BINARY}: expected non-empty precise footprints, got {entries} entries; \
             this subject exists precisely to give the memory channel real signal"
        );

        assert_analysis_equivalent(
            SYSTEM_BINARY,
            "full",
            &ContextDigest::of(&full),
            "sliced",
            &ContextDigest::of(&sliced),
        );
    }
}
