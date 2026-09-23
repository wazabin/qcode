# unpackbench: the contract between corpus, runner, truth and report

All paths relative to `benchmarks/unpackbench/`. Read `../../EVALUATION_PLAN.md`
for the why; this file is the how, and every script must follow it so the
pieces can be written independently.

## Layout

```
build-corpus.sh          builds corpus/ and writes corpus/manifest.json
corpus/<family>/<name>/  one directory per binary (see below)
run-all.sh               runs engines over the corpus into runs/<label>/
truth.py                 compares a run against the manifest's ground truth
report.py                aggregates runs/ into results.json and report.html
engines/<engine>/        one driver per engine (qcode is the unpack example)
runs/<label>/<family>/<name>/<engine>/   one directory per (binary, engine)
```

## Corpus directory

```
corpus/<family>/<name>/
  bin                      the ELF to run (static, or the ld.so form: see manifest)
  source/                  sources and the exact build commands (build.sh)
  truth/
    segments.json          [{"start":"0x..","end":"0x..","sha256":"...","kind":"code|data"}]
                           bytes the run must recover byte-exactly (families 1.1, 1.2, 1.3)
    sites.json             [{"pc":"0x..","generation":N}] expected writer sites (1.2, 1.3, 1.4)
    edges.json             [{"from":"0x..","to":"0x.."}] expected observed edges (1.3, 1.4)
                           or the reference run's edges (1.1, 1.5), see manifest.reference
    stdout, exit           expected stdout bytes and exit code
    stdin                  optional input to feed the guest
```

## manifest.json

```json
{
  "built": "2026-09-22T..",
  "tools": {"upx": "5.2.1-devel..", "gcc": "..", "tigress": "4.0.10", "busybox": ".."},
  "binaries": [
    {
      "family": "1.1", "name": "hw-upx-best",
      "path": "corpus/1.1/hw-upx-best/bin",
      "argv": ["hw"], "stdin": null,
      "launch": "static" | "ldso",
      "seed": "hw", "generator": "upx --best",
      "expect": {"exit": 0, "stdout_sha256": ".."},
      "truth": {"segments": true, "sites": false, "edges": "reference"},
      "reference": "corpus/1.5/hw/bin"
    }
  ]
}
```

`launch: ldso` means the runner executes `/lib64/ld-linux-x86-64.so.2 <path>`
with `--root /`. `truth.edges` is `"reference"` (edges come from running the
unmodified seed in `reference` under the same hooks), `"file"` (edges.json
is hand-written), or `false`.

## Run directory

```
runs/<label>/<family>/<name>/<engine>/
  meta.json      {"engine":..,"config":{"jit":true,"hooks":true,"edges":true},
                  "reps":5,"wall_ms":[..],"steps":[..],"exit":..,"stdout_sha256":..,
                  "crashed":false,"stop_reason":"exit","vm":{"absorbed":..,"native_bodies":..,"evicted":..},
                  "hook_source_lines":N}
  graph.json     the engine's graph in the unpack example's schema (baselines write the same schema)
  regions/       <0xstart>-g<gen>.bin
  stderr.txt
```

Wall numbers are medians over `reps` after one discarded warm-up run.
A `(binary, engine)` the engine cannot run has `meta.json` with
`"unsupported": "<reason>"` and no graph.

## truth.py output

`runs/<label>/<family>/<name>/<engine>/truth.json`:

```json
{"segments":[{"start":..,"expected":N,"recovered":N,"mismatch":N,"extra":N,"exact":true}],
 "sites":{"expected":N,"found":N,"correct":N},
 "edges":{"reference":N,"found":N,"recall":0.0},
 "static_miss": N,
 "regions":{"expected":N,"found":N}}
```

## report.py output

`results.json` (every run label, every row; append-only history: a new
label adds rows, never rewrites old ones) and `report.html`, a single
self-contained file (inline CSS and SVG, no network), with one section per
experiment E1..E7 of the plan, a per-family table, and a progress section
listing every run label with its date and machine.
