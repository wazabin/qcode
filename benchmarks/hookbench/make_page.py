#!/usr/bin/env python3
"""Builds the results page (HTML) from the sweep's JSON, native.txt and
unicorn-py.txt. Usage: make_page.py <results dir> <out.html>"""
import glob, json, math, os, statistics, sys, html
from collections import defaultdict

res = sys.argv[1]
out = sys.argv[2]
rows = []
for p in glob.glob(os.path.join(res, "*.json")):
    rows += json.load(open(p))
by = defaultdict(dict)
for r in rows:
    by[(r["engine"], r["instr"])][r["image"]] = r
native = defaultdict(dict)
np_ = os.path.join(res, "native.txt")
if os.path.exists(np_):
    for line in open(np_):
        f = line.split()
        if len(f) == 5 and f[0][0].isalpha():
            native[f[1]][f[0]] = (int(f[2]), f[3] == "1", int(f[4]))
upy = defaultdict(dict)
up = os.path.join(res, "unicorn-py.txt")
if os.path.exists(up):
    for line in open(up):
        f = line.split()
        if len(f) == 6 and f[0] == "unicorn-py":
            upy[f[1]][f[2]] = (float(f[3]) * 1e6, int(f[4]), f[5] == "ok")
images = sorted({r["image"] for r in rows})

def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None

def slow(engine, k):
    if engine == "native":
        nk = {"block-ir": "block", "block-cb": "block", "edge-ir": "edge", "cmp-ir": "cmp", "cmp-cb": "cmp", "watch-ir": "watch", "watch-cb": "watch"}.get(k)
        if not nk: return None, 0
        xs = [native[nk][i][0] / native["plain"][i][0] for i in images if i in native[nk] and i in native["plain"] and native[nk][i][1]]
        return gm(xs), len(xs)
    if engine == "unicorn-py":
        kk = {"block-cb": "block-cb", "insn-cb": "insn-cb", "watch-cb": "watch-cb", "watch-ir": "watch-cb"}.get(k)
        if not kk: return None, 0
        xs = [upy[kk][i][0] / upy["none"][i][0] for i in images if i in upy.get(kk, {}) and i in upy.get("none", {}) and upy[kk][i][2]]
        return gm(xs), len(xs)
    xs = []
    for i in images:
        a, b = by[(engine, "none")].get(i), by[(engine, k)].get(i)
        if a and b and a["verified"] and b["verified"]:
            xs.append(b["elapsed_ns"] / a["elapsed_ns"])
    return gm(xs), len(xs)

def per_call(engine, k):
    cs = []
    if engine == "unicorn-py":
        kk = {"block-cb": "block-cb", "insn-cb": "insn-cb", "watch-cb": "watch-cb", "watch-ir": "watch-cb"}.get(k)
        if not kk: return None, 0
        for i in images:
            a, b = upy.get("none", {}).get(i), upy.get(kk, {}).get(i)
            if a and b and b[2] and b[1] >= 10000:
                cs.append((b[0] - a[0]) / b[1])
        return (statistics.median(cs) if cs else None), len(cs)
    for i in images:
        a, b = by[(engine, "none")].get(i), by[(engine, k)].get(i)
        if a and b and b["verified"] and b["host_calls"] >= 10000:
            cs.append((b["elapsed_ns"] - a["elapsed_ns"]) / b["host_calls"])
    return (statistics.median(cs) if cs else None), len(cs)

KINDS = ["block-ir", "block-cb", "insn-ir", "insn-cb", "edge-ir", "watch-ir", "watch-cb", "cmp-ir", "cmp-cb"]
ENGINES = [("qcode-jit", "QCode JIT", "s1"), ("icicle", "icicle", "s2"), ("unicorn", "Unicorn (C API)", "s3"), ("unicorn-py", "Unicorn (Python)", "s4"), ("native", "Native, compiler-instrumented", "s5")]
LABEL = {"block-ir": "Block counter, compiled", "block-cb": "Block counter, callback", "insn-ir": "Instruction counter, compiled", "insn-cb": "Instruction counter, callback", "edge-ir": "AFL edge map, compiled", "watch-ir": "Write watch, range check compiled", "watch-cb": "Write watch, callback per store", "cmp-ir": "Compare log, compiled", "cmp-cb": "Compare log, callback"}

slowdown = {(e, k): slow(e, k) for e, _, _ in ENGINES for k in KINDS}
calls = {(e, k): per_call(e, k) for e, _, _ in ENGINES if e != "native" for k in ["block-cb", "insn-cb", "watch-cb", "cmp-cb"]}

def fmt_x(v):
    return f"{v:.2f}×" if v else "—"

# ---- charts: horizontal bars, log scale, one row per instrumentation, one bar per engine
def bar_chart(kinds, engines, get, maxv, unit, log=True, width=760):
    rowh = 18; gap = 10; left = 250; right = 70
    present = {k: [e for e in engines if get(e[0], k)[0]] for k in kinds}
    h = sum(len(present[k]) * rowh + gap for k in kinds) + 40
    W = width
    def x(v):
        if v is None or v <= 0: return left
        if log:
            return left + (math.log10(v) - 0) / (math.log10(maxv)) * (W - left - right)
        return left + v / maxv * (W - left - right)
    svg = [f'<svg class="chart" viewBox="0 0 {W} {h}" width="100%" role="img" aria-label="bar chart">']
    # grid
    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000] if log else [0, maxv / 4, maxv / 2, 3 * maxv / 4, maxv]
    for t in ticks:
        if log and t > maxv: break
        xx = x(t) if t > 0 else left
        svg.append(f'<line x1="{xx:.1f}" y1="10" x2="{xx:.1f}" y2="{h-30}" class="grid"/>')
        lab = f"{t:g}{unit}"
        svg.append(f'<text x="{xx:.1f}" y="{h-14}" class="tick" text-anchor="middle">{lab}</text>')
    y = 10
    for k in kinds:
        per = len(present[k])
        svg.append(f'<text x="{left-10}" y="{y + per*rowh/2 + 4}" class="lab" text-anchor="end">{html.escape(LABEL.get(k,k))}</text>')
        for (e, name, cls) in present[k]:
            v, n = get(e, k)
            w = max(x(v) - left, 2)
            svg.append(f'<rect x="{left}" y="{y+2}" width="{w:.1f}" height="{rowh-4}" rx="3" class="{cls}"><title>{html.escape(name)}: {v:.2f}{unit} over {n} images</title></rect>')
            svg.append(f'<text x="{left + w + 5:.1f}" y="{y + rowh - 5}" class="val">{v:.2f}{unit} <tspan class="who">{html.escape(name)}</tspan></text>')
            y += rowh
        y += gap
    svg.append("</svg>")
    return "\n".join(svg)

maxslow = max([v for (v, n) in slowdown.values() if v] + [10]) * 1.3
chart1 = bar_chart(KINDS, ENGINES, lambda e, k: slowdown[(e, k)], maxslow, "×")
maxcall = max([v for (v, n) in calls.values() if v] + [100]) * 1.3
chart2 = bar_chart(["block-cb", "insn-cb", "watch-cb", "cmp-cb"], [x for x in ENGINES if x[0] != "native"], lambda e, k: calls[(e, k)], maxcall, " ns")

# ---- qcode per-image compiled vs callback
def ms(ns): return f"{ns/1e6:.1f}"
img_rows = []
for i in images:
    base = by[("qcode-jit", "none")].get(i)
    if not base: continue
    cells = [i, ms(base["elapsed_ns"])]
    for k in ["block-ir", "block-cb", "insn-ir", "insn-cb", "edge-ir", "watch-ir", "cmp-ir"]:
        r = by[("qcode-jit", k)].get(i)
        cells.append((f"{r['elapsed_ns']/base['elapsed_ns']:.2f}×" if r["verified"] else "✗") if r else "")
    img_rows.append(cells)

# ---- baselines
base_rows = []
for i in images:
    c = [i]
    n = native["plain"].get(i); c.append(f"{n[0]/1e6:.3f}" if n else "")
    for e in ["qcode-interp", "qcode-jit", "icicle", "unicorn"]:
        r = by[(e, "none")].get(i); c.append(ms(r["elapsed_ns"]) + ("" if r["verified"] else " ✗") if r else "")
    u = upy.get("none", {}).get(i); c.append(f"{u[0]/1e6:.1f}" if u else "")
    base_rows.append(c)

fails = [r for r in rows if not r["verified"]]

def table(head, body, cls="num"):
    t = ['<div class="scroll"><table class="' + cls + '"><thead><tr>' + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr></thead><tbody>"]
    for r in body:
        t.append("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>")
    t.append("</tbody></table></div>")
    return "\n".join(t)

# headline numbers
h_block_ir = slowdown[("qcode-jit", "block-ir")][0]
h_block_cb = slowdown[("qcode-jit", "block-cb")][0]
h_insn_ir = slowdown[("qcode-jit", "insn-ir")][0]
h_insn_cb = slowdown[("qcode-jit", "insn-cb")][0]
n_img = len(images)
load = open(os.path.join(res, "load.txt")).read().strip().splitlines() if os.path.exists(os.path.join(res, "load.txt")) else []

page = f"""<title>QCode Hook Costs</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap">
<style>
:root {{
  --bg: #f7f7f4; --surface: #ffffff; --ink: #14161a; --ink-2: #4d5259; --ink-3: #7d838c; --rule: #dcdfe3; --accent: #1c5cab;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #8a8f98; --grid: #e6e8ec; --bad: #d03b3b;
  color-scheme: light;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #17181b; --surface: #1f2126; --ink: #f2f3f5; --ink-2: #c3c6cc; --ink-3: #8d929b; --rule: #33363d; --accent: #86b6ef;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #8a8f98; --grid: #2b2e34; --bad: #e66767; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #17181b; --surface: #1f2126; --ink: #f2f3f5; --ink-2: #c3c6cc; --ink-3: #8d929b; --rule: #33363d; --accent: #86b6ef;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #8a8f98; --grid: #2b2e34; --bad: #e66767; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--ink); font-family: "IBM Plex Sans", system-ui, sans-serif; font-size: 15px; line-height: 1.5; margin: 0; padding-block: 32px 64px; padding-inline: 16px; }}
main {{ max-width: 980px; margin: 0 auto; }}
h1 {{ font-family: "Fraunces", Georgia, serif; font-weight: 600; font-size: 2.2rem; line-height: 1.1; margin: 0 0 8px; text-wrap: balance; }}
h2 {{ font-family: "Fraunces", Georgia, serif; font-weight: 500; font-size: 1.45rem; margin: 48px 0 12px; text-wrap: balance; }}
p {{ max-width: 68ch; color: var(--ink-2); }}
.lede {{ font-size: 1.05rem; color: var(--ink-2); max-width: 70ch; }}
.eyebrow {{ font-family: "IBM Plex Mono", monospace; font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-3); }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 24px 0; }}
.tile {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 14px 16px; }}
.tile .n {{ font-family: "IBM Plex Mono", monospace; font-size: 1.9rem; font-weight: 500; font-variant-numeric: tabular-nums; }}
.tile .l {{ color: var(--ink-3); font-size: .85rem; }}
.chart {{ display: block; max-width: 100%; height: auto; margin: 8px 0 4px; }}
.chart .grid {{ stroke: var(--grid); stroke-width: 1; }}
.chart .tick, .chart .lab, .chart .val, .chart .na {{ font-family: "IBM Plex Mono", monospace; font-size: 11px; fill: var(--ink-2); }}
.chart .lab {{ font-family: "IBM Plex Sans", sans-serif; fill: var(--ink); font-size: 12px; }}
.chart .who {{ fill: var(--ink-3); }}
.chart .s1 {{ fill: var(--s1); }} .chart .s2 {{ fill: var(--s2); }} .chart .s3 {{ fill: var(--s3); }} .chart .s4 {{ fill: var(--s4); }} .chart .s5 {{ fill: var(--s5); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: .85rem; color: var(--ink-2); margin: 4px 0 12px; }}
.legend span::before {{ content: ""; display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }}
.legend .s1::before {{ background: var(--s1); }} .legend .s2::before {{ background: var(--s2); }} .legend .s3::before {{ background: var(--s3); }} .legend .s4::before {{ background: var(--s4); }} .legend .s5::before {{ background: var(--s5); }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: .88rem; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid var(--rule); text-align: left; white-space: nowrap; }}
th {{ color: var(--ink-3); font-weight: 500; font-size: .78rem; letter-spacing: .04em; text-transform: uppercase; }}
table.num td:not(:first-child) {{ font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; text-align: right; }}
table.num th:not(:first-child) {{ text-align: right; }}
details {{ margin-top: 12px; }} summary {{ cursor: pointer; color: var(--accent); }}
.note {{ border-left: 3px solid var(--s4); padding: 4px 12px; color: var(--ink-2); }}
.bad {{ color: var(--bad); }}
ul {{ color: var(--ink-2); max-width: 70ch; }} li {{ margin: 4px 0; }}
code {{ font-family: "IBM Plex Mono", monospace; font-size: .9em; }}
</style>
<main>
<div class="eyebrow">Embench-IoT · {n_img} images · x86-64 · min of 5 runs</div>
<h1>What a hook costs when it compiles</h1>
<p class="lede">The same instrumentation written two ways under QCode — as IR the machine compiles with the guest, and as a host callback the machine stops for — against icicle-emu, Unicorn through its C API and through Python (the shape Qiling builds on), and native binaries instrumented by the compiler.</p>

<div class="tiles">
  <div class="tile"><div class="n">{fmt_x(h_block_ir)}</div><div class="l">QCode JIT: block counter compiled to IR</div></div>
  <div class="tile"><div class="n">{fmt_x(h_block_cb)}</div><div class="l">QCode JIT: the same as a callback</div></div>
  <div class="tile"><div class="n">{fmt_x(h_insn_ir)}</div><div class="l">QCode JIT: per-instruction counter, compiled</div></div>
  <div class="tile"><div class="n">{fmt_x(h_insn_cb)}</div><div class="l">QCode JIT: the same as a callback</div></div>
</div>

<h2>Slowdown by instrumentation, relative to each engine's own uninstrumented run</h2>
<p>Geometric mean over the images that verified. Log scale. A bar at 1× means the instrumentation was free.</p>
<div class="legend">{"".join(f'<span class="{c}">{html.escape(n)}</span>' for _, n, c in ENGINES)}</div>
{chart1}

<h2>What one host callback costs</h2>
<p>Extra time over the uninstrumented run divided by the number of times the host was entered; median over images with at least 10,000 calls.</p>
<div class="legend">{"".join(f'<span class="{c}">{html.escape(n)}</span>' for e, n, c in ENGINES if e != "native")}</div>
{chart2}

<h2>Baselines: the uninstrumented run, per image</h2>
<p>Milliseconds; native is one iteration of the benchmark function, the emulators run the image once from <code>main</code> to its return. ✗ marks a run that did not verify.</p>
{table(["image", "native", "qcode-interp", "qcode-jit", "icicle", "unicorn", "unicorn-py"], base_rows)}

<h2>QCode JIT, per image: compiled instrumentation against the callback</h2>
<p>Slowdown relative to that image's uninstrumented run.</p>
{table(["image", "base ms", "block-ir", "block-cb", "insn-ir", "insn-cb", "edge-ir", "watch-ir", "cmp-ir"], img_rows)}

<h2>How to read this</h2>
<ul>
<li><b>What "compiled" means per engine.</b> QCode: the hook emits QCode before the site (loads, stores, arithmetic, a conditional detour to an interrupt) and the JIT compiles it with the block. icicle: an injector splices p-code into the lifted block. Unicorn has no compiled form — every hook is a C callback, and Qiling adds Python dispatch on top. Native: <code>gcc -fsanitize-coverage=trace-pc</code> and <code>trace-cmp</code>, plus four hardware watchpoints through <code>perf</code> for the write watch.</li>
<li><b>Sites are not the same across engines.</b> QCode's block is the lifted block after absorption, which is the guest's basic block; its comparison site is every integer comparison in the p-code, which on x86 includes every flag computation, so <code>cmp-*</code> instruments an order of magnitude more sites than Unicorn's <code>cmp</code>-instruction hook. Counts are in the full table.</li>
<li><b>The write watch</b> covers 32 bytes at the start of each image's writable segment; the number of hits varies from none to hundreds of thousands per image, and the per-hit cost is what the second chart isolates.</li>
<li><b>Why the callback is expensive under QCode.</b> A callback is a machine stop: compiled code exits, the interpreter describes the interrupt, the table dispatches, the machine resumes and compiled code is re-entered. Three accidental costs in that path were removed while building this benchmark (a diagnostic string rendered per stop, an O(n) walk to find the position, a rebuilt instruction list per block); what remains is the design. Under icicle the callback is a native call from JIT code; under Unicorn a C call from TCG code.</li>
<li><b>Two limitations found on the way.</b> Re-entering compiled code in the middle of an instruction's p-code, which a store hook needs, produced wrong state on two images; the VM now finishes such a block in the interpreter, which is why <code>watch-cb</code> is the slow QCode column. An interrupt before every comparison (<code>cmp-cb</code>) still fails on crc32 with an interpreter value error and is reported as a failure below.</li>
<li><b>The native comparison is a floor, not a peer.</b> The compiler instruments from source; the emulators instrument a binary. Patching a binary to get the same counters — finding sites, making room, preserving flags and registers — is the work the hook layer does away with, and the point of the compiled column is that it does so at a cost in the same order as the compiler's own.</li>
</ul>

<h2>Runs that did not verify</h2>
{("<ul>" + "".join(f"<li class='bad'>{html.escape(r['engine'])} · {html.escape(r['instr'])} · {html.escape(r['image'])}: <code>{html.escape(r['exit'][:120])}</code></li>" for r in fails) + "</ul>") if fails else "<p>Every run verified.</p>"}

<h2>Method</h2>
<p>Images: Embench-IoT built freestanding for x86-64 with gcc -O2 and SSE off, entered at <code>main</code> with a sentinel return address; each run checks the benchmark's own verification. Host: Intel i7-10850H, 12 threads. Load average at start and end of the sweep: <code>{html.escape(" · ".join(load))}</code>. Harness: <code>benchmarks/hookbench</code> on the <code>emulator-suite</code> branch; <code>run-all.sh</code> reproduces every number here.</p>

<details><summary>Every run</summary>
{table(["engine", "instr", "image", "ms", "host calls", "events", "sites", "ok"], [[e, k, i, ms(r["elapsed_ns"]), r["host_calls"], r["events"], r["sites"], "ok" if r["verified"] else r["exit"][:40]] for (e, k), imgs in sorted(by.items()) for i, r in sorted(imgs.items())])}
<h3>Native, ns per iteration (min of 30), with watchpoint hits</h3>
{table(["image", "plain", "block", "edge", "cmp", "watch", "watch hits"], [[i] + [f"{native[v][i][0]:,}" if i in native[v] else "" for v in ["plain","block","edge","cmp","watch"]] + [native["watch"][i][2] if i in native["watch"] else ""] for i in sorted(native["plain"])])}
</details>
</main>
"""
open(out, "w").write(page)
print("wrote", out, len(page))
