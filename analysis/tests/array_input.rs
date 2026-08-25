//! Differential test for the *array-input* functionalization path: a fill loop
//! that reads the region's original memory gets folded into a `scanl` over the
//! original array (`l0`), not over `iota`.
//!
//! The prefix sum `l[i] = l[i] + l[i-1]` → `scanl (+) l[0] l[1..]` is exercised.
//! The function is emulated *before* and *after* the pipeline over random
//! original memory and must agree byte-for-byte with an independent reference —
//! so the rewrite preserves behavior, and the emulator handles the loaded-array
//! scan source (`Range` slice, data-element scan body).
//!
//! (The seedless indexed map `l[i] = l[i]*3 + i` promotes in `array_promote` but
//! is left unfolded downstream — folding it to `map (enumerate l)` is out of scope
//! for the single-carried-array model; see `array_promote`'s unit tests.)

use qcode::context::Context;
use qcode::value::{BlockParamId, FunctionId};
use qcode_analysis::{PipelineEnv, RegisteredPass, make_pass};
use qcode_emulator::{SizedValue, StandaloneEmulator};

const N: usize = 624;
const BASE: u64 = 0x10_0000;

// A seeded left-scan over the original array: `out[0] = seed`, and
// `out[i] = out[i-1] + l[i]` for `i in 1..N`. Lane 0 is written by the seed
// store; lanes `1..N` read the *original* element `l[i]` (`%cur`) and the
// previous *result* `out[i-1]` (`%prev`, the carry). Folds to
// `scanl (\acc x -> acc + x) seed l[1..]`.
const PREFIX_SRC: &str = r#"
fn prefix_sum:
<entry @seed:i64 @base:i64>
    %e0 = @seed[0:4];
    store(ram:4, @base <- %e0);
    goto <head @i=1 @buf=@base>;
<head @i:i64 @buf:i64>
    %done = @i == 624;
    if %done goto <exit> else goto <body @j=@i @b=@buf>;
<body @j:i64 @b:i64>
    %jm1 = @j - 1;
    %roff = %jm1 * 4;
    %raddr = @b + %roff;
    %prev = load(ram:4, %raddr);
    %coff = @j * 4;
    %caddr = @b + %coff;
    %cur = load(ram:4, %caddr);
    %next = %cur + %prev;
    store(ram:4, %caddr <- %next);
    %j1 = @j + 1;
    goto <head @i=%j1 @buf=@b>;
<exit>
    return at i64 0x0;
"#;

/// `out[0] = seed`, `out[i] = out[i-1] + l[i]` for `i in 1..N`.
fn prefix_reference(orig: &[u32; N], seed: u32) -> [u32; N] {
    let mut out = *orig;
    out[0] = seed;
    for i in 1..N {
        out[i] = orig[i].wrapping_add(out[i - 1]);
    }
    out
}

const PREFIX_SEED: u32 = 0xABCD_1234;

fn load(src: &str, name: &str) -> (Context<'static>, FunctionId) {
    let mut ctx = Context::new();
    qcode::lower::lower_str(&mut ctx, src).expect("source parses and lowers");
    let fid = ctx
        .function_ids()
        .into_iter()
        .find(|&f| qcode::value::FunctionBody::from_id(&ctx, f).name() == name)
        .expect("function present");
    (ctx, fid)
}

fn base_param(ctx: &Context<'_>, fid: FunctionId) -> BlockParamId {
    let root = qcode::value::FunctionBody::from_id(ctx, fid)
        .root()
        .unwrap()
        .id;
    qcode::value::BasicBlock::from_id(ctx, root)
        .params()
        .find(|p| p.name() == Some("base"))
        .expect("@base param")
        .id
}

fn named_param(ctx: &Context<'_>, fid: FunctionId, name: &str) -> Option<BlockParamId> {
    let root = qcode::value::FunctionBody::from_id(ctx, fid)
        .root()
        .unwrap()
        .id;
    qcode::value::BasicBlock::from_id(ctx, root)
        .params()
        .find(|p| p.name() == Some(name))
        .map(|p| p.id)
}

fn emulate_state(
    ctx: &Context<'_>,
    fid: FunctionId,
    orig: &[u32; N],
    seed: Option<u32>,
) -> Vec<u8> {
    let base_p = base_param(ctx, fid);
    let root = qcode::value::FunctionBody::from_id(ctx, fid)
        .root()
        .unwrap()
        .id;
    let bytes: Vec<u8> = orig.iter().flat_map(|w| w.to_le_bytes()).collect();

    let mut emu = StandaloneEmulator::new(root);
    emu.write_memory(ctx, ctx.shared.default_space, BASE, &bytes)
        .expect("seed original ram region");
    emu.block_param_values
        .insert(base_p, SizedValue::new(BASE, 8));
    if let (Some(seed), Some(seed_p)) = (seed, named_param(ctx, fid, "seed")) {
        emu.block_param_values
            .insert(seed_p, SizedValue::new(seed as u64, 8));
    }
    emu.run_function(ctx, fid).expect("emulates");
    emu.read_memory(ctx, ctx.shared.default_space, BASE, N * 4)
        .expect("read state array")
}

fn functionalize(ctx: &mut Context<'_>) {
    let env = PipelineEnv::headless(ctx);
    for name in [
        "array_promote",
        "gvn",
        "loop_to_scan",
        "loop_to_map",
        "gvn",
        "dce",
        "simplify_cfg",
    ] {
        match make_pass(name).expect("pass registered") {
            RegisteredPass::Function(pass) => {
                for fid in ctx.function_ids() {
                    pass.run(ctx, fid, &env)
                        .unwrap_or_else(|e| panic!("pass `{name}` failed: {e}"));
                }
            }
            RegisteredPass::Module(pass) => {
                let mut cone = qcode_analysis::ConeMut::full(ctx);
                pass.run(&mut cone, &env)
                    .unwrap_or_else(|e| panic!("pass `{name}` failed: {e}"));
            }
            // None of the passes named above is a decompile pass, which would
            // need a structured `Program` this helper does not build.
            RegisteredPass::Decompile(_) => {
                panic!("pass `{name}` is a decompile pass; this helper runs IR passes only")
            }
        }
    }
}

/// A spread of random original arrays.
fn random_arrays() -> impl Iterator<Item = [u32; N]> {
    (0..16).map(|iter| {
        let mut s: u32 = 0x9e37_79b9u32.wrapping_add(iter);
        let mut a = [0u32; N];
        for w in a.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *w = s;
        }
        a
    })
}

#[test]
fn prefix_sum_folds_to_scanl_over_original() {
    let (mut ctx, _) = load(PREFIX_SRC, "prefix_sum");
    functionalize(&mut ctx);
    let rendered = format!("{ctx}");
    assert!(
        rendered.contains("scanl") && !rendered.contains("iota"),
        "expected a scanl over the original array (no iota), got:\n{rendered}"
    );
}

#[test]
fn prefix_sum_preserves_state_array() {
    let (pre_ctx, pre_fid) = load(PREFIX_SRC, "prefix_sum");
    let (mut post_ctx, post_fid) = load(PREFIX_SRC, "prefix_sum");
    functionalize(&mut post_ctx);
    for orig in random_arrays() {
        let reference: Vec<u8> = prefix_reference(&orig, PREFIX_SEED)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let pre = emulate_state(&pre_ctx, pre_fid, &orig, Some(PREFIX_SEED));
        let post = emulate_state(&post_ctx, post_fid, &orig, Some(PREFIX_SEED));
        assert_eq!(pre, reference, "pre-pipeline prefix_sum diverges");
        assert_eq!(post, reference, "post-pipeline prefix_sum diverges");
    }
}
