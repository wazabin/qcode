//! Tests for the `unpack` example.
//!
//! The example's module tree is pulled in here rather than duplicated, so the
//! tests drive exactly the code the command line drives.
//!
//! Five layers, from the cheapest to the most expensive: the layout is
//! consistent (no guest at all); `selfdecrypt` runs to its exit and the
//! harvest names both of its generations, byte for byte, on either strategy;
//! a static glibc `hello` built at test time runs under the hooks and
//! generates nothing; the same `hello` packed with UPX unpacks to one
//! region, the first real packer; and, where one is installed, a BusyBox
//! applet runs under them too — a real libc, a real filesystem, and a few
//! million operations of it.

#[path = "../examples/unpack/lib.rs"]
mod unpack;

use std::{fs, path::Path, path::PathBuf, process::Command};

use unpack::{
    artifact,
    driver::{self, Options},
    layout::{self, Layout},
};

/// A fixture by name, or `None` when it is absent — a missing fixture skips
/// its test rather than failing the suite.
fn fixture(name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/unpack")
        .join(name);
    path.exists().then_some(path)
}

/// A directory for one test's artifact, empty whatever a previous run left
/// there.
fn out_dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("unpack")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    dir
}

/// A written `graph.json`, parsed.
fn read_graph(dir: &Path) -> serde_json::Value {
    let text = fs::read_to_string(dir.join("graph.json")).expect("graph.json was written");
    serde_json::from_str(&text).expect("graph.json is JSON")
}

/// A `"0x..."` string as the number it names.
fn hex(value: &serde_json::Value) -> u64 {
    let text = value.as_str().expect("a hexadecimal string");
    u64::from_str_radix(text.trim_start_matches("0x"), 16).expect("a hexadecimal number")
}

/// A node's `range`.
fn range(node: &serde_json::Value) -> (u64, u64) {
    (hex(&node["range"][0]), hex(&node["range"][1]))
}

/// The nodes of a generation.
fn of_generation(graph: &serde_json::Value, generation: u64) -> Vec<&serde_json::Value> {
    graph["nodes"]
        .as_array()
        .expect("nodes is a list")
        .iter()
        .filter(|node| node["generation"].as_u64() == Some(generation))
        .collect()
}

// ---- The layout, which needs no guest at all.

/// The shadow describes two windows, two bytes per guest byte, and fits in a
/// bounded flat space with the sink at offset zero.
#[test]
fn the_shadow_covers_both_windows_two_bytes_at_a_time() {
    let layout = Layout::new(0x40_0000, 0x4d_1000).expect("a normal image has a layout");

    let image = layout.image_window();
    let mmap = layout.mmap_window();
    assert_eq!(image.start, 0x40_0000);
    assert_eq!(image.len, layout::IMAGE_WINDOW_LEN);
    assert_eq!(mmap.start, layout::MMAP_WINDOW_BASE);

    assert_eq!(layout.shadow_of(0x40_0000), Some(layout::IMAGE_SHADOW_OFF));
    assert_eq!(
        layout.shadow_of(0x40_0001),
        Some(layout::IMAGE_SHADOW_OFF + 2)
    );
    assert_eq!(
        layout.shadow_of(layout::MMAP_WINDOW_BASE),
        Some(layout::MMAP_SHADOW_OFF)
    );
    assert_eq!(layout.shadow_of(image.end()), None);

    // Facts about constants, so they hold at compile time; a runtime
    // `assert!` on them would only be a lint.
    const _: () = assert!(layout::SINK_OFF == 0);
    const _: () = assert!(layout::IMAGE_SHADOW_OFF >= layout::SINK_SIZE);
    const _: () = assert!(layout::MMAP_SHADOW_OFF > layout::IMAGE_SHADOW_OFF);
    const _: () = assert!(layout::SHADOW_LEN as u64 <= layout::STATE_SPACE_MAX);
    const _: () = assert!(layout::ENTRIES_LEN as u64 <= layout::STATE_SPACE_MAX);
    const _: () = assert!(layout::VISITED_LEN as u64 <= layout::STATE_SPACE_MAX);
    // A stamp at the top of either window stays inside the space: the widest
    // store the x86-64 specification lifts is sixteen bytes, and its stamp
    // is thirty-two.
    let top = layout.shadow_of(mmap.end() - 1).expect("the last byte");
    assert!(top + 32 <= layout::SHADOW_LEN as u64);
}

/// An image the shadow cannot describe is refused, rather than silently
/// tracked in part.
#[test]
fn an_image_wider_than_the_window_is_refused() {
    assert!(Layout::new(0x40_0000, 0x40_0000 + layout::IMAGE_WINDOW_LEN + 1).is_err());
}

// ---- selfdecrypt: two generations, an mmap, and an exact answer.

/// The stage-1 image: the 127 bytes of the RWX buffer stage 0 decrypts in
/// place.
const STAGE1: std::ops::Range<u64> = 0x401000..0x40107f;
/// The single store site of stage 0's decryption loop.
const STAGE0_STORE: u64 = 0x4000c0;
/// The single store site of stage 1's decryption loop.
const STAGE1_STORE: u64 = 0x40103a;

fn selfdecrypt() -> Option<String> {
    Some(fixture("selfdecrypt")?.to_str()?.to_owned())
}

/// The whole POC in one test: `selfdecrypt` runs to its exit under the JIT
/// with both hooks installed, and the harvest says — from the shadow bytes
/// alone, never from a disassembly — that stage 1 was written by stage 0 and
/// stage 2 by stage 1.
#[test]
fn selfdecrypt_attributes_each_stage_to_the_one_that_wrote_it() {
    let Some(path) = selfdecrypt() else { return };
    let dir = out_dir("selfdecrypt-jit");

    let options = Options {
        jit: true,
        edges: true,
        ..Options::default()
    };
    let (mut process, outcome) =
        driver::run_keeping(&path, &options).expect("selfdecrypt loads and runs");

    // Nothing stopped the run but the guest's own exit: no hook of this
    // example may reach `Task::handle_interrupt`, which would crash the task.
    assert!(!outcome.crashed, "{}", outcome.stop_reason);
    assert_eq!(outcome.stop_reason, "exit");
    assert_eq!(outcome.exit_status, Some(7));
    assert_eq!(outcome.stdout, b"unpacked: stage 2\n");
    assert!(outcome.stderr.is_empty());
    assert!(!outcome.recorder.sites_saturated);
    assert!(!outcome.recorder.blocks_saturated);

    let summary = artifact::write(&dir, &mut process, &outcome).expect("the artifact is written");
    let graph = read_graph(&dir);
    assert_eq!(
        graph["warnings"].as_array().map(Vec::len),
        Some(0),
        "{:?}",
        graph["warnings"]
    );
    assert_eq!(summary.warnings, 0);

    // The store site behind each id, so a node's provenance can be read as
    // the program counter that wrote it.
    let pc_of = |id: u64| -> u64 {
        let site = graph["sites"]
            .as_array()
            .expect("sites is a list")
            .iter()
            .find(|site| site["id"].as_u64() == Some(id))
            .expect("every stamped id names a site");
        hex(&site["pc"])
    };

    // Stage 1 is generation 1, lives in the RWX segment, and every byte of
    // it was written by stage 0's one store.
    let stage1 = of_generation(&graph, 1);
    assert!(!stage1.is_empty(), "no generation-1 node");
    for node in &stage1 {
        let (lo, hi) = range(node);
        assert!(
            STAGE1.start <= lo && hi <= STAGE1.end,
            "the generation-1 node at {lo:#x}..{hi:#x} is outside the stage-1 buffer"
        );
        let pcs: Vec<u64> = node["sites"]
            .as_array()
            .expect("a node's sites")
            .iter()
            .map(|id| pc_of(id.as_u64().expect("a site id")))
            .collect();
        assert!(!pcs.is_empty(), "{lo:#x} has no provenance");
        assert!(
            pcs.iter().all(|&pc| pc == STAGE0_STORE),
            "the generation-1 node at {lo:#x} was written from {pcs:x?}, not only \
             from stage 0's store at {STAGE0_STORE:#x}"
        );
    }

    // Stage 2 is generation 2, lives where `mmap` placed it, and stage 1
    // wrote it.
    let mmap = outcome.layout.mmap_window();
    let stage2 = of_generation(&graph, 2);
    assert!(!stage2.is_empty(), "no generation-2 node");
    for node in &stage2 {
        let (lo, hi) = range(node);
        assert!(
            lo >= mmap.start && hi <= mmap.end(),
            "the generation-2 node at {lo:#x}..{hi:#x} is outside the mmap window"
        );
        let pcs: Vec<u64> = node["sites"]
            .as_array()
            .expect("a node's sites")
            .iter()
            .map(|id| pc_of(id.as_u64().expect("a site id")))
            .collect();
        assert!(!pcs.is_empty(), "{lo:#x} has no provenance");
        assert!(
            pcs.iter().all(|&pc| pc == STAGE1_STORE),
            "the generation-2 node at {lo:#x} was written from {pcs:x?}, not only \
             from stage 1's store at {STAGE1_STORE:#x}"
        );
    }

    // Nothing is generated without a writer.
    let generated_by: Vec<u64> = graph["edges"]
        .as_array()
        .expect("edges is a list")
        .iter()
        .filter(|edge| edge["kind"] == "generated_by")
        .map(|edge| hex(&edge["to"]))
        .collect();
    for node in stage1.iter().chain(&stage2) {
        let addr = hex(&node["addr"]);
        assert!(
            generated_by.contains(&addr),
            "the generated node at {addr:#x} has no generated_by edge"
        );
    }

    // Two stages, two regions, and the sizes the fixture's README documents.
    let regions = graph["regions"].as_array().expect("regions is a list");
    let sized = |generation: u64| -> Vec<u64> {
        regions
            .iter()
            .filter(|region| region["generation"].as_u64() == Some(generation))
            .map(|region| region["bytes_len"].as_u64().expect("a region length"))
            .collect()
    };
    assert_eq!(
        sized(1),
        [127],
        "one generation-1 region, the stage-1 image"
    );
    assert_eq!(sized(2), [54], "one generation-2 region, the stage-2 image");
    assert_eq!(regions.len(), 2);
    for region in regions {
        let start = region["start"].as_str().expect("a region start");
        let generation = region["generation"].as_u64().expect("a region generation");
        let bytes = fs::read(
            dir.join("regions")
                .join(format!("{start}-g{generation}.bin")),
        )
        .expect("the region's bytes were written");
        assert_eq!(
            bytes.len() as u64,
            region["bytes_len"].as_u64().expect("a region length"),
            "regions/{start}-g{generation}.bin is as long as the region says"
        );
    }

    // And the IR itself round-trips: `context.bin` is the context the run
    // ended with, not a summary of it.
    let encoded = fs::read(dir.join("context.bin")).expect("context.bin was written");
    let (decoded, _) = bincode::serde::decode_from_slice::<qcode::context::Context<'static>, _>(
        &encoded,
        bincode::config::standard(),
    )
    .expect("context.bin decodes");
    assert_eq!(
        decoded.blocks().count(),
        process.vm().context().blocks().count()
    );

    let meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("context.meta.json")).unwrap())
            .expect("context.meta.json is JSON");
    assert_eq!(
        meta["blocks"].as_u64(),
        Some(process.vm().context().blocks().count() as u64)
    );
    assert_eq!(meta["bytes"].as_u64(), Some(encoded.len() as u64));

    let _ = fs::remove_dir_all(&dir);
}

/// The hooks are a rewrite of the lifted code, so the interpreter and the
/// JIT run the same program and must harvest the same graph — byte for byte,
/// with and without `--edges`.
///
/// The one range that legitimately differs is the run's own name for itself,
/// `program.strategy`, which is substituted rather than ignored: everything
/// else, down to the order of the edges and the hashes of the blocks, has to
/// match exactly.
#[test]
fn the_two_strategies_write_the_same_graph() {
    let Some(path) = selfdecrypt() else { return };

    for edges in [false, true] {
        let mut written = Vec::new();
        for jit in [false, true] {
            let dir = out_dir(&format!("selfdecrypt-same-{jit}-{edges}"));
            let options = Options {
                jit,
                edges,
                ..Options::default()
            };
            let (mut process, outcome) =
                driver::run_keeping(&path, &options).expect("selfdecrypt loads and runs");
            assert_eq!(outcome.exit_status, Some(7), "jit={jit}");
            assert!(!outcome.crashed, "jit={jit}: {}", outcome.stop_reason);
            artifact::write(&dir, &mut process, &outcome).expect("the artifact is written");
            let text = fs::read_to_string(dir.join("graph.json")).expect("graph.json was written");
            written.push(text.replace(r#""strategy": "interpreter""#, r#""strategy": "jit""#));
            let _ = fs::remove_dir_all(&dir);
        }
        assert_eq!(
            written[0], written[1],
            "the interpreter and the JIT disagree about the graph (edges={edges})"
        );
    }
}

/// `--edges` names the block each stage was entered from, which is the one
/// thing the static successors of the final IR cannot give: both stage
/// transitions are indirect jumps.
#[test]
fn edges_link_a_stage_to_the_block_that_jumped_into_it() {
    let Some(path) = selfdecrypt() else { return };
    let dir = out_dir("selfdecrypt-edges");

    let options = Options {
        jit: true,
        edges: true,
        ..Options::default()
    };
    let (mut process, outcome) =
        driver::run_keeping(&path, &options).expect("selfdecrypt loads and runs");
    artifact::write(&dir, &mut process, &outcome).expect("the artifact is written");
    let graph = read_graph(&dir);

    let observed: Vec<(u64, u64)> = graph["edges"]
        .as_array()
        .expect("edges is a list")
        .iter()
        .filter(|edge| edge["kind"] == "observed")
        .map(|edge| (hex(&edge["from"]), hex(&edge["to"])))
        .collect();
    assert!(!observed.is_empty(), "no observed edge was recorded");

    // Stage 0 jumped into the stage-1 buffer, and stage 1 into the mapping.
    let mmap = outcome.layout.mmap_window();
    assert!(
        observed
            .iter()
            .any(|&(from, to)| from < STAGE1.start && STAGE1.contains(&to)),
        "no observed edge enters stage 1: {observed:x?}"
    );
    assert!(
        observed
            .iter()
            .any(|&(from, to)| STAGE1.contains(&from) && mmap.contains(to)),
        "no observed edge enters stage 2: {observed:x?}"
    );

    // Without `--edges` the same run records none of them: absent, not wrong.
    let plain = out_dir("selfdecrypt-no-edges");
    let (mut process, outcome) = driver::run_keeping(
        &path,
        &Options {
            jit: true,
            ..Options::default()
        },
    )
    .expect("selfdecrypt loads and runs");
    artifact::write(&plain, &mut process, &outcome).expect("the artifact is written");
    let graph = read_graph(&plain);
    assert!(
        graph["edges"]
            .as_array()
            .expect("edges is a list")
            .iter()
            .all(|edge| edge["kind"] != "observed"),
        "observed edges were recorded without --edges"
    );

    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&plain);
}

/// The first-entry log holds each block that ran exactly once, in the order
/// it first ran — which is what makes it a log and not a trace.
#[test]
fn the_log_holds_each_block_that_ran_exactly_once() {
    let Some(path) = selfdecrypt() else { return };

    let options = Options {
        jit: true,
        ..Options::default()
    };
    let outcome = driver::run(&path, &options).expect("selfdecrypt loads and runs");

    let seen: Vec<u32> = outcome.log.iter().map(|entry| entry.k).collect();
    assert!(!seen.is_empty());
    let mut distinct = seen.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), seen.len(), "a block was logged twice");
    assert!(
        seen.iter()
            .all(|&k| (k as usize) < outcome.recorder.blocks.len()),
        "the log names a block the recorder never instrumented"
    );

    // Stage 0's loop runs its body 127 times and stage 1's 54 times, so the
    // run is dominated by entries that added nothing to the log.
    assert!(
        outcome.steps > 100 * seen.len() as u64,
        "{} steps for {} logged blocks",
        outcome.steps,
        seen.len()
    );
}

/// Without the hooks the run is the same run — same output, same status —
/// and the artifact simply has nothing to say about provenance.
#[test]
fn a_run_without_hooks_produces_the_same_program_and_no_graph() {
    let Some(path) = selfdecrypt() else { return };
    let dir = out_dir("selfdecrypt-no-hooks");

    let options = Options {
        jit: true,
        hooks: false,
        ..Options::default()
    };
    let (mut process, outcome) =
        driver::run_keeping(&path, &options).expect("selfdecrypt loads and runs");
    assert_eq!(outcome.exit_status, Some(7));
    assert_eq!(outcome.stdout, b"unpacked: stage 2\n");
    assert!(outcome.recorder.sites.is_empty());
    assert!(outcome.log.is_empty());

    let summary = artifact::write(&dir, &mut process, &outcome).expect("the artifact is written");
    assert!(summary.nodes >= 1, "an uninstrumented run lifted no block");
    assert_eq!(summary.executed, 0, "nothing logs an entry without hooks");
    assert!(summary.generated.is_empty());
    assert!(summary.regions.is_empty());

    let _ = fs::remove_dir_all(&dir);
}

// ---- Real libcs.

/// Builds `hello.c` with `gcc -static -O2`, or `None` when the toolchain
/// cannot.
fn build_static_hello() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("unpack/hello");
    fs::create_dir_all(&dir).ok()?;
    let source = dir.join("hello.c");
    fs::write(
        &source,
        "#include <stdio.h>\nint main(void) { puts(\"hello\"); return 0; }\n",
    )
    .ok()?;
    let elf = dir.join("hello");
    let built = Command::new("gcc")
        .args(["-static", "-O2", "-o"])
        .arg(&elf)
        .arg(&source)
        .status()
        .ok()?;
    (built.success() && elf.exists()).then_some(elf)
}

/// A static glibc `hello`, built at test time: the only case here that runs a
/// real libc rather than hand-written assembly, and so the only guard on the
/// hooks against a program that resolves ifuncs, sets up TLS and goes through
/// `__libc_start_main`.
///
/// Skipped when `gcc` cannot produce a static binary.
#[test]
fn a_static_glibc_hello_runs_under_the_hooks() {
    let Some(elf) = build_static_hello() else {
        eprintln!("skipped: gcc cannot build a static x86-64 binary here");
        return;
    };
    let dir = out_dir("glibc-hello");
    let path = elf
        .to_str()
        .expect("the temporary path is UTF-8")
        .to_owned();

    let options = Options {
        jit: true,
        ..Options::default()
    };
    let (mut process, outcome) =
        driver::run_keeping(&path, &options).expect("the static hello loads");
    assert!(!outcome.crashed, "{}", outcome.stop_reason);
    assert_eq!(outcome.exit_status, Some(0));
    assert_eq!(outcome.stdout, b"hello\n");
    assert!(!outcome.recorder.blocks_saturated, "the log filled up");
    assert!(!outcome.recorder.sites_saturated, "the site ids saturated");

    let summary = artifact::write(&dir, &mut process, &outcome).expect("the artifact is written");
    assert!(summary.nodes >= 1, "a libc run lifted no block");
    assert!(summary.executed >= 1, "a libc run entered no block");
    // Nothing a static hello does writes code it then runs, so no code is
    // generated and no region is of kind code; the program does write data,
    // which is now captured as data regions (that is expected, not a fault).
    assert!(summary.generated.is_empty(), "{:?}", summary.generated);
    assert!(
        summary.regions.iter().all(|r| r.3 == "data"),
        "a static hello produced a code region: {:?}",
        summary.regions
    );

    let _ = fs::remove_dir_all(&dir);
}

/// The first real packer: a static glibc `hello` packed with UPX (the
/// fixture's README says how). The stub decompresses the whole program into
/// a `MAP_SHARED` mapping of a memfd at the image base and maps the file back
/// over it executable, so the harvest sees one generation-1 region — the
/// unpacked program, byte for byte where the loader would have put it —
/// written by the stub's decompressor, and nothing deeper.
///
/// JIT only: the same run takes half a minute under the interpreter, and the
/// two strategies lift one loop head of the stub's fold differently, so
/// they do not write byte-identical graphs for this program (a node and a
/// site more under the interpreter; the regions and generations agree).
#[test]
fn a_upx_packed_hello_unpacks_to_one_region() {
    let Some(path) = fixture("hello.upx") else {
        return;
    };
    let path = path.to_str().expect("the fixture path is UTF-8").to_owned();
    let dir = out_dir("upx-hello");

    let options = Options {
        jit: true,
        edges: true,
        ..Options::default()
    };
    let (mut process, outcome) =
        driver::run_keeping(&path, &options).expect("hello.upx loads and runs");
    assert!(!outcome.crashed, "{}", outcome.stop_reason);
    assert_eq!(outcome.stop_reason, "exit");
    assert_eq!(outcome.exit_status, Some(0));
    assert_eq!(outcome.stdout, b"hello\n");
    assert!(outcome.stderr.is_empty());
    assert!(!outcome.recorder.blocks_saturated, "the log filled up");
    assert!(!outcome.recorder.sites_saturated, "the site ids saturated");

    let summary = artifact::write(&dir, &mut process, &outcome).expect("the artifact is written");
    let graph = read_graph(&dir);

    // The unpacked program comes back as one generation-1 CODE region inside
    // the image window at the base (half a megabyte the stub wrote), and its
    // writable segments come back as DATA regions — the data recovery the
    // reviewer asked for. hw-upx's two RW `PT_LOAD`s are at 0x47b000 and
    // 0x4a3108.
    let image = outcome.layout.image_window();
    let regions = graph["regions"].as_array().expect("regions is a list");
    let code: Vec<&serde_json::Value> = regions.iter().filter(|r| r["kind"] == "code").collect();
    assert_eq!(code.len(), 1, "expected one code region: {code:?}");
    let region = code[0];
    assert_eq!(region["generation"].as_u64(), Some(1));
    let start = hex(&region["start"]);
    let len = region["bytes_len"].as_u64().expect("a region length");
    assert!(len > 400_000, "the code region is {len} bytes");
    assert!(
        start >= image.start && start + len <= image.end(),
        "the region {start:#x}+{len:#x} leaves the image window"
    );
    assert_eq!(summary.regions.iter().filter(|r| r.3 == "code").count(), 1);

    // The data segments are recovered: their bytes are captured as data
    // regions summing to the bulk of the two writable segments.
    let data_bytes: u64 = regions
        .iter()
        .filter(|r| r["kind"] == "data")
        .map(|r| r["bytes_len"].as_u64().unwrap_or(0))
        .sum();
    assert!(
        data_bytes > 150_000,
        "the writable segments were not recovered: {data_bytes} data bytes"
    );
    assert!(
        regions.iter().any(|r| {
            r["kind"] == "data" && {
                let s = hex(&r["start"]);
                (0x47b000..0x4a3108 + 0x20000).contains(&s)
            }
        }),
        "the RW data segments at 0x47b000/0x4a3108 were not captured"
    );

    // Every generated node is generation 1 and lies in that region: the
    // program the stub unpacked runs no unpacker of its own.
    let generated: Vec<&serde_json::Value> = graph["nodes"]
        .as_array()
        .expect("nodes is a list")
        .iter()
        .filter(|node| node["generated"] == true)
        .collect();
    assert!(
        generated.len() > 1000,
        "{} generated nodes",
        generated.len()
    );
    for node in &generated {
        let (lo, hi) = range(node);
        assert_eq!(
            node["generation"].as_u64(),
            Some(1),
            "the generated node at {lo:#x} is not generation 1"
        );
        assert!(
            start <= lo && hi <= start + len,
            "the generated node at {lo:#x}..{hi:#x} is outside the region"
        );
    }
    assert!(of_generation(&graph, 2).is_empty(), "a second generation");

    // The region's bytes are the file the loader would have mapped: they
    // start with the unpacked program's code, not with the packed stub's.
    let bytes = fs::read(dir.join(format!("regions/{:#x}-g1.bin", start)))
        .expect("the region's bytes were written");
    assert_eq!(bytes.len() as u64, len);

    // Byte-exact: the region is the unpacked program's `R E` segment from
    // eight bytes past its base (the ELF magic the packed file already had)
    // to the segment's end, followed by the stub's exit trampoline
    // (`endbr64; syscall; pop rdx; pop rax; jmp rax`), which it writes just
    // past the text and leaves through. The digest is that of bytes
    // `0x8..0x7adfd` of the unpacked `hello` the fixture was packed from
    // (see the fixture README).
    const TEXT_END: u64 = 0x47adfd;
    const TEXT_SHA256: &str = "9400e9e66d468ef4af501831d7cca8a21e3db2440a715c7f99d9abb9767d87bc";
    const TRAMPOLINE: [u8; 11] = [
        0xf3, 0x0f, 0x1e, 0xfa, 0x0f, 0x05, 0x5a, 0x58, 0x3e, 0xff, 0xe0,
    ];
    assert_eq!(
        start, 0x400008,
        "the region does not start at the image base"
    );
    let text = &bytes[..(TEXT_END - start) as usize];
    assert_eq!(
        sha256_hex(text),
        TEXT_SHA256,
        "the unpacked text differs from the original binary"
    );
    assert_eq!(
        &bytes[text.len()..],
        &TRAMPOLINE,
        "past the text is not the stub's trampoline"
    );

    // And the IR round-trips, packed program and stub in the one context.
    let encoded = fs::read(dir.join("context.bin")).expect("context.bin was written");
    let (decoded, _) = bincode::serde::decode_from_slice::<qcode::context::Context<'static>, _>(
        &encoded,
        bincode::config::standard(),
    )
    .expect("context.bin decodes");
    assert_eq!(
        decoded.blocks().count(),
        process.vm().context().blocks().count()
    );

    let _ = fs::remove_dir_all(&dir);
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The BusyBox binary to test with, if one is installed.
fn busybox() -> Option<PathBuf> {
    std::env::var("BUSYBOX")
        .ok()
        .map(PathBuf::from)
        .into_iter()
        .chain(
            ["/usr/sbin/busybox", "/usr/bin/busybox", "/bin/busybox"]
                .iter()
                .map(PathBuf::from),
        )
        .find(|path| path.exists())
}

/// A BusyBox applet under the hooks: a real libc, the real filesystem, and
/// several million operations of both, with every guest store stamping a
/// shadow byte and every block logging its entry.
///
/// This is the one case that says the hooks survive a program nobody wrote
/// for them. Skipped when no BusyBox is installed.
#[test]
fn a_busybox_applet_runs_under_the_hooks() {
    let Some(busybox) = busybox() else {
        eprintln!("skipped: no busybox (set BUSYBOX to point at a static one)");
        return;
    };
    let path = busybox.to_str().expect("the path is UTF-8").to_owned();

    let options = Options {
        jit: true,
        args: vec!["echo".to_owned(), "hi".to_owned()],
        ..Options::default()
    };
    let (_, outcome) = match driver::run_keeping(&path, &options) {
        Ok(run) => run,
        Err(message) => {
            eprintln!("skipped: {message}");
            return;
        }
    };
    if outcome.crashed {
        eprintln!(
            "skipped: {} does not run under this environment: {}",
            busybox.display(),
            outcome.stop_reason
        );
        return;
    }
    assert_eq!(outcome.exit_status, Some(0), "{}", outcome.stop_reason);
    assert_eq!(
        String::from_utf8_lossy(&outcome.stdout),
        "hi\n",
        "busybox echo did not print through the hooks"
    );
    assert!(!outcome.log.is_empty(), "no block logged an entry");
    assert!(
        !outcome.recorder.sites.is_empty(),
        "no store site was instrumented"
    );
}
