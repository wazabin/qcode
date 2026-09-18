#!/usr/bin/env python3
"""Turns the sweep's JSON and native results into Markdown tables.

Usage: report.py <results dir> [<label>=<dir> ...]   (the OUT of run-all.sh,
then later runs of part of the sweep, oldest first; each replaces the rows
it re-measured and keeps a column in the progress table)
"""
import glob, json, os, sys
from collections import defaultdict

out = sys.argv[1] if len(sys.argv) > 1 else "target/results"
runs = [("first sweep", out)] + [a.split("=", 1) for a in sys.argv[2:]]

def load_rows(d):
    rows = []
    for path in glob.glob(os.path.join(d, "*.json")):
        rows += json.load(open(path))
    return rows

def index(rows):
    by = defaultdict(dict)
    for r in rows:
        by[(r["engine"], r["instr"])][r["image"]] = r
    return by

run_by = [(label, d, index(load_rows(d))) for label, d in runs]
seen = {}
for _, _, b in reversed(run_by):
    for key, imgs in b.items():
        for img, r in imgs.items():
            seen.setdefault((key, img), r)
rows = list(seen.values())
# (engine, instr) -> image -> row, the latest word on each
by = index(rows)
native = defaultdict(dict)  # variant -> image -> (ns, verified, events)
if os.path.exists(os.path.join(out, "native.txt")):
    for line in open(os.path.join(out, "native.txt")):
        parts = line.split()
        if len(parts) == 5 and parts[0][0].isalpha():
            name, variant, ns, ok, ev = parts
            native[variant][name] = (int(ns), ok == "1", int(ev))

images = sorted({r["image"] for r in rows})

def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return float("nan")
    import math
    return math.exp(sum(math.log(x) for x in xs) / len(xs))

def ms(ns):
    return f"{ns / 1e6:.1f}"

# 1. Baselines: every engine, no instrumentation, per image.
print("## Baseline: no instrumentation (ms, min of repeats)\n")
engines = ["native", "qcode-interp", "qcode-jit", "icicle", "unicorn"]
print("| image | " + " | ".join(engines) + " |")
print("|---|" + "---:|" * len(engines))
for img in images:
    cells = []
    for e in engines:
        if e == "native":
            v = native["plain"].get(img)
            cells.append(f"{v[0] / 1e6:.3f}" if v else "")
        else:
            r = by[(e, "none")].get(img)
            cells.append(ms(r["elapsed_ns"]) + ("" if r["verified"] else " ✗") if r else "")
    print(f"| {img} | " + " | ".join(cells) + " |")
print()

# 2. Slowdown of each instrumentation relative to the engine's own baseline.
print("## Slowdown relative to each engine's own baseline (geometric mean over images)\n")
kinds = ["block-ir", "block-ram", "block-cb", "insn-ir", "insn-ram", "insn-cb", "edge-ir", "watch-ir", "watch-cb", "cmp-ir", "cmp-cb"]
native_kind = {"block-ir": "block", "block-ram": "block", "block-cb": "block", "edge-ir": "edge", "cmp-ir": "cmp", "cmp-cb": "cmp", "watch-ir": "watch", "watch-cb": "watch"}
print("| instrumentation | native (compiler) | qcode-jit | icicle | unicorn | qcode-interp |")
print("|---|---:|---:|---:|---:|---:|")
for k in kinds:
    cells = []
    nk = native_kind.get(k)
    if nk:
        ratios = []
        for img in images:
            a, b = native["plain"].get(img), native[nk].get(img)
            if a and b and b[1]:
                ratios.append(b[0] / a[0])
        cells.append(f"{geomean(ratios):.2f}×" if ratios else "")
    else:
        cells.append("")
    for e in ["qcode-jit", "icicle", "unicorn", "qcode-interp"]:
        ratios = []
        for img in images:
            a, b = by[(e, "none")].get(img), by[(e, k)].get(img)
            if a and b and a["verified"] and b["verified"]:
                ratios.append(b["elapsed_ns"] / a["elapsed_ns"])
        n = len(ratios)
        cells.append(f"{geomean(ratios):.2f}× (n={n})" if ratios else "")
    print(f"| {k} | " + " | ".join(cells) + " |")
print()

# 3. Cost per host call: (t_instr - t_none) / host_calls, in ns, over images with many calls.
print("## Cost per host call (ns, median over images with ≥ 10k calls)\n")
print("| instrumentation | qcode-jit | icicle | unicorn |")
print("|---|---:|---:|---:|")
import statistics
for k in ["block-cb", "insn-cb", "watch-ir", "watch-cb", "cmp-cb"]:
    cells = []
    for e in ["qcode-jit", "icicle", "unicorn"]:
        costs = []
        for img in images:
            a, b = by[(e, "none")].get(img), by[(e, k)].get(img)
            if a and b and b["verified"] and b["host_calls"] >= 10_000:
                costs.append((b["elapsed_ns"] - a["elapsed_ns"]) / b["host_calls"])
        cells.append(f"{statistics.median(costs):.0f} (n={len(costs)})" if costs else "")
    print(f"| {k} | " + " | ".join(cells) + " |")
print()

# 4. Per-event cost of compiled instrumentation on qcode-jit.
print("## Compiled instrumentation on qcode-jit: overhead per event (ns, median over images)\n")
print("| instrumentation | ns/event | events per image (median) | sites per image (median) |")
print("|---|---:|---:|---:|")
for k in ["block-ir", "block-ram", "insn-ir", "insn-ram"]:
    costs, evs, sites = [], [], []
    for img in images:
        a, b = by[("qcode-jit", "none")].get(img), by[("qcode-jit", k)].get(img)
        if a and b and b["verified"] and b["events"] > 0:
            costs.append((b["elapsed_ns"] - a["elapsed_ns"]) / b["events"])
            evs.append(b["events"]); sites.append(b["sites"])
    if costs:
        print(f"| {k} | {statistics.median(costs):.1f} | {statistics.median(evs):.0f} | {statistics.median(sites):.0f} |")
print()

# 5. Progress: the qcode-jit column of every run.
if len(run_by) > 1:
    print("## How the qcode-jit numbers moved (oldest first)\n")
    print("| instrumentation | " + " | ".join(label for label, _, _ in run_by) + " |")
    print("|---|" + "---:|" * len(run_by))
    for k in kinds:
        cells = []
        for _, _, b in run_by:
            ratios = []
            fails_ = 0
            for img in images:
                a, c = b[("qcode-jit", "none")].get(img), b[("qcode-jit", k)].get(img)
                if a and c and a["verified"] and c["verified"]:
                    ratios.append(c["elapsed_ns"] / a["elapsed_ns"])
                elif c and not c["verified"]:
                    fails_ += 1
            cells.append((f"{geomean(ratios):.2f}× (n={len(ratios)})" if ratios else "") + (f" {fails_} ✗" if fails_ else ""))
        print(f"| {k} | " + " | ".join(cells) + " |")
    print()
    for label, d, b in run_by:
        notes = open(os.path.join(d, "notes.txt")).read().strip() if os.path.exists(os.path.join(d, "notes.txt")) else ""
        print(f"- {label}: {notes}" if notes else f"- {label}")
    print()

# 6. Failures.
fails = [r for r in rows if not r["verified"]]
if fails:
    print("## Runs that did not verify\n")
    for r in fails:
        print(f"- {r['engine']} {r['instr']} {r['image']}: {r['exit']}")
    print()
for v in native:
    bad = [i for i, t in native[v].items() if not t[1]]
    if bad:
        print(f"- native {v} did not verify: {', '.join(bad)}")

# 7. Full table.
print("\n## Every run\n")
print("| engine | instr | image | ms | host calls | events | sites | ok |")
print("|---|---|---|---:|---:|---:|---:|---|")
for (e, k), imgs in sorted(by.items()):
    for img, r in sorted(imgs.items()):
        print(f"| {e} | {k} | {img} | {ms(r['elapsed_ns'])} | {r['host_calls']} | {r['events']} | {r['sites']} | {'ok' if r['verified'] else r['exit'][:40]} |")
print("\n### Native (ns per benchmark iteration, min of 30)\n")
print("| image | plain | block | edge | cmp | watch (hits) |")
print("|---|---:|---:|---:|---:|---:|")
for img in sorted(native["plain"]):
    c = [f"{native[v][img][0]:,}" if img in native[v] else "" for v in ["plain", "block", "edge", "cmp"]]
    w = native["watch"].get(img)
    c.append(f"{w[0]:,} ({w[2]})" if w else "")
    print(f"| {img} | " + " | ".join(c) + " |")
