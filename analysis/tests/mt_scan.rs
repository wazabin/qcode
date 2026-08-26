//! Differential test for the memory-loop → `scanl` functionalization pipeline.
//!
//! `examples/qcode/mt19937_init.qcode` is the honest lifting of the MT19937
//! seeding loop: a RAM-carried fill loop that stores `mt[i]` and reloads
//! `mt[i-1]`. The `array_promote` + `loop_to_scan` passes rewrite it into the
//! functional form `store(base <- concat(singleton(seed), scanl f seed …))`,
//! and `dce` + `simplify_cfg` delete the now-dead residual loop.
//!
//! This test emulates the function *before* and *after* that pipeline on random
//! seeds and asserts that (a) both agree byte-for-byte on the 624×4-byte state
//! array written at `@base`, and (b) both match an independent Rust MT19937
//! reference — so the rewrite is behavior-preserving, not merely well-typed.

use qcode::context::Context;
use qcode::value::{BlockParamId, FunctionId};
use qcode_analysis::{PipelineEnv, RegisteredPass, make_pass};
use qcode_emulator::{SizedValue, StandaloneEmulator};

const SOURCE: &str = include_str!("../../examples/qcode/mt19937_init.qcode");

/// MT19937 word count and the base address we place the state array at.
const N: usize = 624;
const BASE: u64 = 0x10_0000;

/// The independent reference: `mt[0] = seed`, `mt[i] = 1812433253 *
/// (mt[i-1] ^ (mt[i-1] >> 30)) + i`, all mod 2^32.
fn mt19937_reference(seed: u32) -> [u32; N] {
    let mut mt = [0u32; N];
    mt[0] = seed;
    for i in 1..N {
        mt[i] = 1812433253u32
            .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
            .wrapping_add(i as u32);
    }
    mt
}

/// Parse the example into a fresh context, returning it with `mt_init`'s id.
fn load() -> (Context<'static>, FunctionId) {
    let mut ctx = Context::new();
    qcode::lower::lower_str(&mut ctx, SOURCE).expect("mt19937_init.qcode parses and lowers");
    let fid = ctx
        .function_ids()
        .into_iter()
        .find(|&f| qcode::value::FunctionBody::from_id(&ctx, f).name() == "mt_init")
        .expect("mt_init present");
    (ctx, fid)
}

/// The `@seed` / `@base` entry-param ids of `fid`'s root block.
fn entry_params(ctx: &Context<'_>, fid: FunctionId) -> (BlockParamId, BlockParamId) {
    let root = qcode::value::FunctionBody::from_id(ctx, fid)
        .root()
        .expect("mt_init has a root")
        .id;
    let mut seed = None;
    let mut base = None;
    for p in qcode::value::BasicBlock::from_id(ctx, root).params() {
        match p.name() {
            Some("seed") => seed = Some(p.id),
            Some("base") => base = Some(p.id),
            _ => {}
        }
    }
    (seed.expect("@seed param"), base.expect("@base param"))
}

/// Emulate `fid` with the given seed and read back the `N`-word state array
/// written at `BASE`.
fn emulate_state(ctx: &Context<'_>, fid: FunctionId, seed: u32) -> Vec<u8> {
    let (seed_p, base_p) = entry_params(ctx, fid);
    let root = qcode::value::FunctionBody::from_id(ctx, fid)
        .root()
        .unwrap()
        .id;

    let mut emu = StandaloneEmulator::new(root);
    // Zero-fill (and configure) the RAM region so a stray read never trips on
    // uninitialized memory; the function overwrites all N words regardless.
    emu.write_memory(ctx, ctx.shared.default_space, BASE, &vec![0u8; N * 4])
        .expect("configure ram region");

    // `@seed` / `@base` are not register-named, so `run_function`'s register
    // seeding leaves them untouched — bind them directly.
    emu.block_param_values
        .insert(seed_p, SizedValue::new(seed as u64, 8));
    emu.block_param_values
        .insert(base_p, SizedValue::new(BASE, 8));

    emu.run_function(ctx, fid).expect("mt_init emulates");
    emu.read_memory(ctx, ctx.shared.default_space, BASE, N * 4)
        .expect("read state array")
}

/// Run the functionalization pipeline over every function in `ctx`.
fn functionalize(ctx: &mut Context<'_>) {
    let env = PipelineEnv::headless(ctx);
    for name in [
        "array_promote",
        "gvn",
        "loop_to_scan",
        "gvn",
        "dce",
        "simplify_cfg",
    ] {
        let RegisteredPass::Function(pass) = make_pass(name).expect("pass registered") else {
            panic!("{name} is a function pass");
        };
        for fid in ctx.function_ids() {
            pass.run(ctx, fid, &env)
                .unwrap_or_else(|e| panic!("pass `{name}` failed: {e}"));
        }
    }
}

/// The pipeline must actually reach the `scanl` form (guards against a silent
/// no-op that would make the differential comparison vacuous).
#[test]
fn pipeline_produces_scanl_form() {
    let (mut ctx, _) = load();
    functionalize(&mut ctx);
    let rendered = format!("{ctx}");
    assert!(
        rendered.contains("scanl") && rendered.contains("$concat"),
        "expected the functional scanl+concat form, got:\n{rendered}"
    );
    // The residual fill loop must be gone: once the scan carries the whole
    // computation, `loop_to_scan` reroutes the preheader past the now-private
    // loop and deletes its blocks. A surviving `$insert`/`$at` back-edge means
    // the loop stayed alive (e.g. the returned seed kept it pinned).
    assert!(
        !rendered.contains("$insert") && !rendered.contains("$at("),
        "expected the fill loop to be deleted, but insert/at survive:\n{rendered}"
    );
}

#[test]
fn scan_functionalization_preserves_state_array() {
    let (pre_ctx, pre_fid) = load();
    let (mut post_ctx, post_fid) = load();
    functionalize(&mut post_ctx);

    // A spread of seeds including the usual defaults and edge values.
    let mut seed: u32 = 0x1234_5678;
    for iter in 0..64 {
        // A few fixed seeds, then a cheap LCG-driven spread.
        let s = match iter {
            0 => 0,
            1 => 1,
            2 => 5489, // the canonical MT19937 default seed
            3 => u32::MAX,
            _ => {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                seed
            }
        };

        let reference: Vec<u8> = mt19937_reference(s)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let pre = emulate_state(&pre_ctx, pre_fid, s);
        let post = emulate_state(&post_ctx, post_fid, s);

        assert_eq!(
            pre, reference,
            "pre-pipeline mt_init diverges from reference at seed {s:#x}"
        );
        assert_eq!(
            post, reference,
            "post-pipeline (scanl) mt_init diverges from reference at seed {s:#x}"
        );
    }
}
