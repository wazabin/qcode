#!/usr/bin/env python3
"""Aggregate runs/ into results.json (append-only) and report.html.

    report.py [--runs DIR] [--results FILE] [--html FILE] [--label L]

results.json keeps every label ever reported: a label seen for the first time
adds its rows; a label already there only gains rows for (family, name,
engine) triples it did not have. Nothing already recorded is rewritten, so
deleting runs/ loses no history. A run whose graph has not been through
truth.py yet is left for a later call.

report.html is one self-contained file (inline CSS, SVG drawn here, no
script, no network): experiments E1..E7 of EVALUATION_PLAN.md §5 for one
label (--label, default the most recent), a per-family table, and a progress
section over every label.
"""

import argparse
import html
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import ub  # noqa: E402

SCHEMA = 1
ENGINE_ORDER = ["qcode", "icicle", "unicorn", "qiling"]  # colour slots, fixed
FAMILY_TITLES = {
    "1.1": "compressed executables",
    "1.2": "runtime code generation",
    "1.3": "self-modifying fixtures",
    "1.4": "split control flow",
    "1.5": "negative controls",
}


# --- results.json -------------------------------------------------------------

def row_from(run_dir):
    meta = ub.load_json(run_dir / "meta.json")
    if meta is None:
        return None
    truth = ub.load_json(run_dir / "truth.json")
    if truth is None and (run_dir / "graph.json").exists():
        return None  # truth.py has not seen it yet
    b = meta.get("binary", {})
    configs = {}
    for cfg, r in (meta.get("configs") or {}).items():
        configs[cfg] = {k: r.get(k) for k in (
            "unsupported", "timeout", "crashed", "stop_reason", "exit", "stdout_sha256",
            "wall_ms", "wall_ms_median", "proc_ms", "steps", "steps_median", "spread",
            "vm", "graph", "graph_sha256", "wall_scope") if k in r}
    if truth:
        truth = {k: v for k, v in truth.items() if k not in ("family", "name", "engine")}
    return {
        "family": b.get("family"), "name": b.get("name"), "engine": meta.get("engine"),
        "launch": b.get("launch"), "seed": b.get("seed"), "generator": b.get("generator"),
        "binary_sha256": b.get("sha256"),
        "unsupported": meta.get("unsupported"), "primary": meta.get("primary"),
        "exit": meta.get("exit"), "exit_ok": meta.get("exit_ok"), "stdout_ok": meta.get("stdout_ok"),
        "crashed": meta.get("crashed"), "stop_reason": meta.get("stop_reason"),
        "reps": meta.get("reps"), "hook_source_lines": meta.get("hook_source_lines"),
        "configs": configs, "truth": truth,
    }


def update_results(runs, results_path):
    results = ub.load_json(results_path) or {"schema": SCHEMA, "labels": {}}
    labels = results.setdefault("labels", {})
    added = 0
    for label_dir in sorted(p for p in Path(runs).glob("*") if (p / "label.json").exists()):
        lmeta = ub.load_json(label_dir / "label.json")
        name = lmeta.get("label", label_dir.name)
        entry = labels.get(name)
        if entry is None:
            entry = labels[name] = {"meta": lmeta, "rows": [],
                                    "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        have = {(r["family"], r["name"], r["engine"]) for r in entry["rows"]}
        for m in sorted(label_dir.glob("*/*/*/meta.json")):
            row = row_from(m.parent)
            if row is None:
                continue
            key = (row["family"], row["name"], row["engine"])
            if key in have:
                continue
            entry["rows"].append(row)
            have.add(key)
            added += 1
    ub.write_json(results_path, results)
    return results, added


# --- small helpers ------------------------------------------------------------

def esc(x):
    return html.escape("" if x is None else str(x))


def fmt_ms(v):
    if v is None:
        return "–"
    if v >= 10_000:
        return f"{v / 1000:.1f} s"
    if v >= 1000:
        return f"{v / 1000:.2f} s"
    return f"{v:.1f} ms" if v >= 10 else f"{v:.2f} ms"


def fmt_int(v):
    return "–" if v is None else f"{v:,}"


def fmt_ratio(v):
    return "–" if v is None else f"{v:.2f}×"


def fmt_frac(v, digits=3):
    return "–" if v is None else f"{v:.{digits}f}"


def engine_slot(engine):
    return (ENGINE_ORDER.index(engine) + 1) if engine in ENGINE_ORDER else 8


def badge(ok, yes="pass", no="fail", unknown="n/a"):
    if ok is None:
        return f'<span class="badge na">{unknown}</span>'
    return (f'<span class="badge good">&#10003; {yes}</span>' if ok
            else f'<span class="badge bad">&#10007; {no}</span>')


def cfg(row, name, key="wall_ms_median"):
    c = (row.get("configs") or {}).get(name) or {}
    return None if c.get("unsupported") else c.get(key)


def ratio(a, b):
    return a / b if a is not None and b else None


def table(headers, rows, numeric=None, cls=""):
    numeric = numeric or set()
    out = [f'<div class="tablewrap"><table class="{cls}"><thead><tr>']
    for i, h in enumerate(headers):
        out.append(f'<th class="{"num" if i in numeric else ""}">{h}</th>')
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>" + "".join(
            f'<td class="{"num" if i in numeric else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>")
    if not rows:
        out.append(f'<tr><td colspan="{len(headers)}" class="empty">no rows in this label</td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def binname(r):
    return f'{esc(r["family"])}/{esc(r["name"])}'


def code_segments(r):
    t = r.get("truth") or {}
    return [s for s in t.get("segments", []) if s.get("kind", "code") == "code"]


# --- SVG ----------------------------------------------------------------------

def nice_ticks(lo, hi, n=5):
    span = hi - lo if hi > lo else 1
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    start = math.floor(lo / step + 1e-9) * step
    end = math.ceil(hi / step - 1e-9) * step
    count = int(round((end - start) / step))
    return [round(start + i * step, 10) for i in range(count + 1)]


def log_ticks(lo, hi):
    ticks = []
    for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for m in (1, 3):
            v = m * 10 ** e
            if lo <= v <= hi:
                ticks.append(v)
    return ticks


def hbar_chart(title, rows, x_label, *, log=False, x_max=None, tick_fmt=str,
               value_fmt=str, reference=None, legend=None):
    """rows: [(label, [(value, slot, series_name, tooltip)])]: grouped
    horizontal bars, one group per label, one bar per series."""
    bar_h, gap, group_gap = 14, 2, 12
    left, right, top, bottom = 190, 90, 34, 44
    width = 760
    plot_w = width - left - right
    values = [v for _, bars in rows for v, *_ in bars if v is not None and (not log or v > 0)]
    if not values:
        return f'<p class="empty">{esc(title)}: no data.</p>'
    if log:
        lo = 10 ** math.floor(math.log10(min(values)))
        hi = 10 ** math.ceil(math.log10(max(values) * 1.05))
        ticks = log_ticks(lo, hi)
        scale = lambda v: (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * plot_w
    else:
        lo, hi = 0.0, x_max if x_max is not None else max(values) * 1.08
        ticks = nice_ticks(lo, hi)
        hi = max(hi, ticks[-1])
        scale = lambda v: (v - lo) / (hi - lo) * plot_w
    heights = [len(bars) * (bar_h + gap) - gap for _, bars in rows]
    plot_h = sum(heights) + group_gap * (len(rows) - 1)
    height = top + plot_h + bottom + (22 if legend else 0)
    s = [f'<figure><svg viewBox="0 0 {width} {height}" width="100%" role="img" '
         f'aria-label="{esc(title)}"><title>{esc(title)}</title>']
    s.append(f'<g transform="translate({left},{top})">')
    for t in ticks:
        x = scale(t)
        s.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{plot_h}"/>')
        s.append(f'<text class="tick" x="{x:.1f}" y="{plot_h + 16}" text-anchor="middle">{esc(tick_fmt(t))}</text>')
    if reference is not None:
        x = scale(reference)
        s.append(f'<line class="ref" x1="{x:.1f}" x2="{x:.1f}" y1="-6" y2="{plot_h}"/>')
    y = 0
    for (label, bars), h in zip(rows, heights):
        s.append(f'<text class="rowlabel" x="-10" y="{y + h / 2 + 4:.1f}" text-anchor="end">{esc(label)}</text>')
        for v, slot, series, tip in bars:
            if v is None or (log and v <= 0):
                s.append(f'<text class="na" x="4" y="{y + bar_h - 3}">{esc(series)}: no data</text>')
            else:
                w = max(scale(v), 1.5)
                s.append(f'<g><title>{esc(tip)}</title>'
                         f'<rect class="s{slot}" x="0" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3"/>'
                         f'<text class="value" x="{w + 5:.1f}" y="{y + bar_h - 3}">{esc(value_fmt(v))}</text></g>')
            y += bar_h + gap
        y += group_gap - gap
    s.append(f'<line class="axis" x1="0" x2="0" y1="0" y2="{plot_h}"/>')
    s.append(f'<text class="axislabel" x="{plot_w / 2:.1f}" y="{plot_h + 36}" text-anchor="middle">{esc(x_label)}</text>')
    s.append("</g>")
    if legend:
        lx = left
        for slot, name in legend:
            s.append(f'<rect class="s{slot}" x="{lx}" y="{height - 16}" width="12" height="12" rx="2"/>'
                     f'<text class="legend" x="{lx + 17}" y="{height - 6}">{esc(name)}</text>')
            lx += 30 + 8 * len(name)
    s.append(f'<text class="charttitle" x="{left}" y="18">{esc(title)}</text>')
    s.append("</svg></figure>")
    return "".join(s)


def stacked_chart(title, rows, x_label, legend, value_fmt=str):
    """rows: [(label, [(value, slot, name)], total_label)]: one stacked bar each."""
    bar_h, gap = 16, 10
    left, right, top, bottom = 190, 70, 34, 66
    width = 760
    plot_w = width - left - right
    totals = [sum(v for v, *_ in segs if v) for _, segs, _ in rows]
    if not rows or not max(totals, default=0):
        return f'<p class="empty">{esc(title)}: no data.</p>'
    ticks = nice_ticks(0, max(totals) * 1.05)
    hi = ticks[-1]
    scale = lambda v: v / hi * plot_w
    plot_h = len(rows) * (bar_h + gap) - gap
    height = top + plot_h + bottom
    s = [f'<figure><svg viewBox="0 0 {width} {height}" width="100%" role="img" '
         f'aria-label="{esc(title)}"><title>{esc(title)}</title><g transform="translate({left},{top})">']
    for t in ticks:
        x = scale(t)
        s.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{plot_h}"/>'
                 f'<text class="tick" x="{x:.1f}" y="{plot_h + 16}" text-anchor="middle">{esc(value_fmt(t))}</text>')
    x1 = scale(1.0)
    s.append(f'<line class="ref" x1="{x1:.1f}" x2="{x1:.1f}" y1="-6" y2="{plot_h}"/>')
    for i, (label, segs, total_label) in enumerate(rows):
        y = i * (bar_h + gap)
        s.append(f'<text class="rowlabel" x="-10" y="{y + bar_h - 3}" text-anchor="end">{esc(label)}</text>')
        x = 0.0
        for v, slot, name in segs:
            if not v:
                continue
            w = scale(v)
            s.append(f'<g><title>{esc(label)}: {esc(name)} {esc(value_fmt(v))}</title>'
                     f'<rect class="s{slot}" x="{x:.1f}" y="{y}" width="{max(w - 2, 1):.1f}" height="{bar_h}" rx="3"/></g>')
            x += w
        s.append(f'<text class="value" x="{x + 5:.1f}" y="{y + bar_h - 3}">{esc(total_label)}</text>')
    s.append(f'<line class="axis" x1="0" x2="0" y1="0" y2="{plot_h}"/>'
             f'<text class="axislabel" x="{plot_w / 2:.1f}" y="{plot_h + 36}" text-anchor="middle">{esc(x_label)}</text></g>')
    lx = left
    for slot, name in legend:
        s.append(f'<rect class="s{slot}" x="{lx}" y="{height - 16}" width="12" height="12" rx="2"/>'
                 f'<text class="legend" x="{lx + 17}" y="{height - 6}">{esc(name)}</text>')
        lx += 30 + 7 * len(name)
    s.append(f'<text class="charttitle" x="{left}" y="18">{esc(title)}</text></svg></figure>')
    return "".join(s)


def line_chart(title, points, y_label, value_fmt=str):
    """points: [(x_label, value)] in order, one series."""
    pts = [(l, v) for l, v in points if v is not None]
    if len(pts) < 2:
        return ""
    left, right, top, bottom, width, height = 60, 30, 34, 60, 760, 240
    pw, ph = width - left - right, height - top - bottom
    vals = [v for _, v in pts]
    ticks = nice_ticks(min(0.0, min(vals)), max(vals) * 1.1)
    hi, lo = ticks[-1], ticks[0]
    xs = lambda i: i / (len(pts) - 1) * pw
    ys = lambda v: ph - (v - lo) / (hi - lo) * ph
    s = [f'<figure><svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{esc(title)}">'
         f'<title>{esc(title)}</title><g transform="translate({left},{top})">']
    for t in ticks:
        s.append(f'<line class="grid" x1="0" x2="{pw}" y1="{ys(t):.1f}" y2="{ys(t):.1f}"/>'
                 f'<text class="tick" x="-8" y="{ys(t) + 4:.1f}" text-anchor="end">{esc(value_fmt(t))}</text>')
    path = " ".join(f'{"M" if i == 0 else "L"}{xs(i):.1f},{ys(v):.1f}' for i, (_, v) in enumerate(pts))
    s.append(f'<path class="line s1" d="{path}"/>')
    for i, (l, v) in enumerate(pts):
        s.append(f'<g><title>{esc(l)}: {esc(value_fmt(v))}</title><circle class="s1 dot" cx="{xs(i):.1f}" cy="{ys(v):.1f}" r="4.5"/></g>'
                 f'<text class="tick" x="{xs(i):.1f}" y="{ph + 16}" text-anchor="middle">{esc(l)}</text>')
    s.append(f'<text class="axislabel" transform="translate(-44,{ph / 2}) rotate(-90)" text-anchor="middle">{esc(y_label)}</text>'
             f'<text class="axislabel" x="{pw / 2}" y="{ph + 40}" text-anchor="middle">run label</text></g>'
             f'<text class="charttitle" x="{left}" y="18">{esc(title)}</text></svg></figure>')
    return "".join(s)


# --- sections -------------------------------------------------------------------

def section(sid, title, claim, body, criterion=None, verdict=None):
    head = f'<section id="{sid}"><h2>{esc(title)}</h2><p class="claim">{claim}</p>'
    if criterion:
        head += f'<p class="criterion"><b>Pass criterion.</b> {criterion} {"" if verdict is None else verdict}</p>'
    return head + body + "</section>"


def e1(rows):
    rs = [r for r in rows if r["family"] in ("1.1", "1.2", "1.3", "1.5") and not r.get("unsupported")]
    body_rows, bars, verdicts = [], [], []
    for r in rs:
        t = r.get("truth") or {}
        segs = code_segments(r)
        reg = t.get("regions") or {}
        sites = t.get("sites") or {}
        exp = sum(s["expected"] for s in segs)
        rec = sum((s.get("recovered") or 0) for s in segs)
        mis = sum((s.get("mismatch") or 0) for s in segs)
        miss = sum((s.get("missing") or 0) for s in segs)
        extra = sum((s.get("extra") or 0) for s in segs) + (t.get("extra_unattributed") or 0)
        exact = all(s.get("exact") for s in segs) if segs else None
        if r["family"] == "1.5":
            ok = reg.get("found") == 0
        else:
            ok = exact
            if sites.get("expected"):
                ok = bool(ok is not False and sites.get("correct") == sites.get("expected"))
        verdicts.append(ok)
        data = [s for s in t.get("segments", []) if s.get("kind") == "data"]
        data_note = (f'<br><span class="muted">data: {fmt_int(sum(s.get("recovered") or 0 for s in data))}'
                     f' / {fmt_int(sum(s["expected"] for s in data))}</span>') if data else ""
        first = next((s.get("first_diff") for s in segs if s.get("first_diff")), None)
        body_rows.append([
            binname(r), esc(r["engine"]), badge(run_ok(r), "ok", "wrong"),
            f'{sum(1 for s in segs if s.get("exact"))} / {len(segs)}' if segs else "–",
            fmt_int(exp) if segs else "–", (fmt_int(rec) + data_note) if segs else "–",
            fmt_int(mis) if segs else "–", fmt_int(miss) if segs else "–",
            fmt_int(extra), f'{fmt_int(reg.get("found"))} / {fmt_int(reg.get("expected"))}',
            fmt_int(reg.get("generations")),
            "–" if sites.get("expected") is None else f'{sites.get("correct")} / {sites.get("expected")}',
            (f'<span class="muted">first gap {esc(first)}</span> ' if first else "") + badge(ok)])
        if segs:
            bars.append((f'{r["family"]}/{r["name"]}', [(rec / exp if exp else None, engine_slot(r["engine"]),
                        r["engine"], f'{r["family"]}/{r["name"]} {r["engine"]}: {rec:,} of {exp:,} bytes '
                        f'({mis:,} mismatching, {miss:,} missing, {extra:,} extra)')]))
    fig = hbar_chart("Figure 1. Code bytes recovered over code bytes expected", bars,
                     "fraction of expected code bytes recovered byte-exactly", x_max=1.0,
                     tick_fmt=lambda v: f"{v:g}",
                     value_fmt=lambda v: "1 (exact)" if v == 1 else f"{v:.6f}", reference=1.0)
    tab = table(["binary", "engine", "run", "code segs exact", "bytes expected", "recovered",
                 "mismatch", "missing", "extra", "regions found / exp.", "gen.", "sites correct", "verdict"],
                body_rows, numeric={4, 5, 6, 7, 8, 10})
    known = [v for v in verdicts if v is not None]
    verdict = None if not known else badge(all(known), f"{sum(known)}/{len(known)} pass", f"{sum(known)}/{len(known)} pass")
    return section("E1", "E1 Code recovery (RQ1)",
                   "Families 1.1, 1.2, 1.3 under <code>--jit --edges</code>; controls (1.5) must yield no region. "
                   "<i>missing</i> counts expected bytes no region covers (UPX leaves the first 8 bytes, the ELF magic, "
                   "as the packed file had them); <i>extra</i> counts region bytes outside every expected segment "
                   "(the UPX exit trampoline).",
                   tab + fig,
                   "every code segment byte-exact; every site attributed with the right generation; controls yield zero regions.",
                   verdict)


def run_ok(r):
    if r.get("unsupported"):
        return None
    return (not r.get("crashed")) and r.get("exit_ok") is not False and r.get("stdout_ok") is not False


def e2(rows):
    rs = [r for r in rows if r["family"] in ("1.1", "1.2", "1.3", "1.4", "1.5") and not r.get("unsupported")]
    body, bars, verdicts = [], [], []
    for r in rs:
        t = r.get("truth") or {}
        e = t.get("edges") or {}
        rec = e.get("recall")
        if r["family"] in ("1.1", "1.3") and rec is not None:
            verdicts.append(rec >= 1.0)
        state = esc(e.get("pending")) if e.get("pending") else esc(e.get("source") or "–")
        body.append([binname(r), esc(r["engine"]), fmt_int(e.get("reference")), fmt_int(e.get("found")),
                     fmt_frac(rec), fmt_int(e.get("observed")), fmt_int(t.get("static_miss")),
                     fmt_int((t.get("graph") or {}).get("static_edges")), f'<span class="muted">{state}</span>'])
        if rec is not None:
            bars.append((f'{r["family"]}/{r["name"]}', [(rec, engine_slot(r["engine"]), r["engine"],
                         f'{r["family"]}/{r["name"]} {r["engine"]}: {e.get("found")} of {e.get("reference")} reference edges')]))
    fig = hbar_chart("Figure 2a. Observed-edge recall against the reference", bars,
                     "recall (reference edges found in the graph)", x_max=1.0,
                     tick_fmt=lambda v: f"{v:g}", value_fmt=lambda v: f"{v:.3f}", reference=1.0)
    tab = table(["binary", "engine", "reference edges", "found", "recall", "observed", "static miss",
                 "static edges", "reference source"], body, numeric={2, 3, 4, 5, 6, 7})
    verdict = None if not verdicts else badge(all(verdicts), f"{sum(verdicts)}/{len(verdicts)} at 1.0",
                                               f"{sum(verdicts)}/{len(verdicts)} at 1.0")
    return section("E2", "E2 Control-flow recovery (RQ2)",
                   "Observed edges are first-entry edges (the block that ran just before each block's first entry). "
                   "<i>Static miss</i> counts observed edges absent from the static <code>control_flow</code> edges of "
                   "the same graph. Cross-process edges (1.4) are not in the graph schema yet.",
                   tab + fig, "recall 1.0 on families 1.1 and 1.3; on 1.4, every cross-process oracle edge present.",
                   verdict)


def e3(rows):
    rs = [r for r in rows if r["family"] in ("1.1", "1.5")]
    body, bars, engines = [], [], []
    by_bin = {}
    for r in rs:
        by_bin.setdefault((r["family"], r["name"]), []).append(r)
        if r["engine"] not in engines:
            engines.append(r["engine"])
    engines.sort(key=lambda e: (engine_slot(e), e))
    qratio = {}
    for key in sorted(by_bin):
        group = sorted(by_bin[key], key=lambda r: engine_slot(r["engine"]))
        rowbars = []
        for r in group:
            if r.get("unsupported"):
                body.append([f'{esc(key[0])}/{esc(key[1])}', esc(r["engine"]),
                             f'<span class="badge na">unsupported</span> <span class="muted">{esc(r["unsupported"])}</span>',
                             "", "", "", "", "", "", ""])
                rowbars.append((None, engine_slot(r["engine"]), r["engine"], "unsupported"))
                continue
            base, hooks, edges = cfg(r, "nohooks"), cfg(r, "hooks"), cfg(r, "edges")
            sb, sh = cfg(r, "nohooks", "steps_median"), cfg(r, "hooks", "steps_median")
            vb = (r["configs"].get("nohooks") or {}).get("vm") or {}
            vh = (r["configs"].get("hooks") or {}).get("vm") or {}
            same = None
            if vb and vh:
                same = vb.get("native_bodies") == vh.get("native_bodies") and vb.get("absorbed") == vh.get("absorbed")
            rh = ratio(hooks, base)
            if r["engine"] == "qcode":
                qratio[key] = rh
            body.append([f'{esc(key[0])}/{esc(key[1])}', esc(r["engine"]), fmt_ms(base), fmt_ms(hooks), fmt_ms(edges),
                         fmt_ratio(rh), fmt_ratio(ratio(edges, base)), fmt_ratio(ratio(sh, sb)),
                         fmt_int(r.get("hook_source_lines")),
                         badge(same, "equal", "differ") if r["engine"] == "qcode" else "–"])
            rowbars.append((hooks, engine_slot(r["engine"]), r["engine"],
                            f'{key[0]}/{key[1]} {r["engine"]}: {fmt_ms(hooks)} with hooks, {fmt_ms(base)} without'))
        bars.append((f"{key[0]}/{key[1]}", rowbars))
    # Pass: QCode's ratio at or below every baseline's on every binary.
    verdicts = []
    for key, group in by_bin.items():
        q = qratio.get(key)
        for r in group:
            if r["engine"] != "qcode" and not r.get("unsupported"):
                other = ratio(cfg(r, "hooks"), cfg(r, "nohooks"))
                if q is not None and other is not None:
                    verdicts.append(q <= other)
    verdict = (badge(all(verdicts), f"{sum(verdicts)}/{len(verdicts)} pairs", f"{sum(verdicts)}/{len(verdicts)} pairs")
               if verdicts else '<span class="badge na">no baseline in this label</span>')
    fig = hbar_chart("Figure 2. Wall time with the task's hooks, per engine (log scale)", bars,
                     "wall time, median, ms (log scale)", log=True,
                     tick_fmt=lambda v: f"{v:g}", value_fmt=fmt_ms,
                     legend=[(engine_slot(e), e) for e in engines] if len(engines) > 1 else None)
    tab = table(["binary", "engine", "no hooks", "hooks", "hooks + edges", "hooks ratio", "edges ratio",
                 "steps ratio", "hook lines", "JIT bodies w/ and w/o hooks"], body, numeric={2, 3, 4, 5, 6, 7, 8})
    scopes = {}
    for r in rs:
        for c in (r.get("configs") or {}).values():
            if c.get("wall_scope"):
                scopes.setdefault(r["engine"], c["wall_scope"])
    if scopes:
        tab += '<p class="muted">Wall scope per engine (compare ratios, not absolute walls, across engines): ' + \
            "; ".join(f"<b>{esc(e)}</b>: {esc(v)}" for e, v in sorted(scopes.items(), key=lambda x: engine_slot(x[0]))) + "</p>"
    return section("E3", "E3 Cost and instrumentation, four engines (RQ3, C1, C2)",
                   "Medians over the label's reps after one discarded warm-up; wall is the engine's own measure. <i>Hook lines</i> counts non-blank, "
                   "non-comment lines of each engine's hook source.",
                   tab + fig, "QCode's ratio at or below every baseline's on every binary; an engine that cannot run a "
                   "binary is marked, not omitted.", verdict)


def e4(rows):
    rs = [r for r in rows if r["family"] == "1.4"]
    body = []
    for r in rs:
        body.append([binname(r), esc(r["engine"]), badge(run_ok(r), "ok", "wrong"),
                     fmt_ms(cfg(r, "nohooks")), fmt_ms(cfg(r, "hooks")), fmt_ratio(ratio(cfg(r, "hooks"), cfg(r, "nohooks"))),
                     esc(r.get("stop_reason"))])
    note = ("Children, traps serviced, switches and bytes per switch are not reported by the unpack example yet; "
            "the table shows what the harness has.")
    return section("E4", "E4 Process model (RQ4)", note,
                   table(["binary", "engine", "run", "no hooks", "hooks", "ratio", "stop"], body, numeric={3, 4, 5}),
                   "oracle output and exit reproduced for every corpus guest; voracious reaches exit with 15 fan-out edges.")


def e5(rows):
    rs = [r for r in rows if r["family"] in ("1.1", "1.2") and not r.get("unsupported")]
    body = [[binname(r), esc(r["engine"]), fmt_int((r.get("truth") or {}).get("regions", {}).get("found")),
             '<span class="badge na">not measured</span>'] for r in rs]
    return section("E5", "E5 Static consumption (C4)",
                   "Each QCode run directory keeps <code>context.bin</code>; the static pass over it and over the "
                   "original file is not part of this harness yet.",
                   table(["binary", "engine", "regions", "resolved targets (context.bin vs file)"], body, numeric={2}),
                   "the pass resolves targets inside generated regions from context.bin that it cannot see in the file.")


def e6(rows):
    body, verdicts = [], []
    for r in rows:
        if r.get("unsupported"):
            continue
        c = r.get("configs") or {}
        reps_same = []
        for name in ("hooks", "edges", "interp"):
            shas = [s for s in (c.get(name) or {}).get("graph_sha256") or [] if s]
            if shas:
                reps_same.append(f'{name}: {"identical" if len(set(shas)) == 1 else f"{len(set(shas))} distinct"}')
        je, ie = c.get("edges") or {}, c.get("interp") or {}
        cross = delta_nodes = delta_edges = delta_steps = None
        if je.get("graph_sha256") and ie.get("graph_sha256"):
            cross = je["graph_sha256"][-1] == ie["graph_sha256"][-1]
            gj, gi = je.get("graph") or {}, ie.get("graph") or {}
            delta_nodes = (gi.get("nodes") or 0) - (gj.get("nodes") or 0)
            delta_edges = ((gi.get("static_edges") or 0) + (gi.get("observed_edges") or 0)
                           - (gj.get("static_edges") or 0) - (gj.get("observed_edges") or 0))
            if ie.get("steps_median") is not None and je.get("steps_median") is not None:
                delta_steps = ie["steps_median"] - je["steps_median"]
            verdicts.append(cross)
        if not reps_same and cross is None:
            continue
        body.append([binname(r), esc(r["engine"]), esc("; ".join(reps_same)) or "–",
                     badge(cross, "identical", "differ", "not run"),
                     "–" if delta_nodes is None else f"{delta_nodes:+d}",
                     "–" if delta_edges is None else f"{delta_edges:+d}",
                     "–" if delta_steps is None else f"{delta_steps:+,.0f}"])
    verdict = None if not verdicts else badge(all(verdicts), f"{sum(verdicts)}/{len(verdicts)} identical",
                                              f"{sum(verdicts)}/{len(verdicts)} identical")
    return section("E6", "E6 Determinism and stability (RQ5)",
                   "Graph bytes across the timed reps of each configuration, and between the JIT "
                   "(<code>edges</code>) and the interpreter (<code>interp</code>, run with <code>CONFIGS=...,interp</code>). "
                   "Graphs compare with <code>program.strategy</code> substituted, as the unpack tests do. Deltas are interpreter minus JIT.",
                   table(["binary", "engine", "across reps", "interpreter vs JIT", "Δ nodes", "Δ edges", "Δ steps"],
                         body, numeric={4, 5, 6}),
                   "identical, or the delta explained by lifted-copy duplication only.", verdict)


def e7(rows):
    rs = [r for r in rows if r["engine"] == "qcode" and not r.get("unsupported")]
    bars = []
    body = []
    for r in sorted(rs, key=lambda r: (r["family"], r["name"])):
        base, hooks, edges = cfg(r, "nohooks"), cfg(r, "hooks"), cfg(r, "edges")
        if not base:
            continue
        segs = [(1.0, 7, "no hooks")]
        rh = ratio(hooks, base)
        re_ = ratio(edges, base)
        if rh is not None:
            segs.append((max(rh - 1.0, 0.0), 1, "+ provenance and entry hooks"))
        if re_ is not None:
            segs.append((max(re_ - (rh if rh is not None else 1.0), 0.0), 2, "+ observed edges"))
        total = re_ if re_ is not None else rh
        bars.append((f'{r["family"]}/{r["name"]}', segs, fmt_ratio(total)))
        sb, sh, se = (cfg(r, n, "steps_median") for n in ("nohooks", "hooks", "edges"))
        body.append([binname(r), fmt_ms(base), fmt_ratio(rh), fmt_ratio(re_), fmt_ratio(ratio(sh, sb)),
                     fmt_ratio(ratio(se, sb))])
    fig = stacked_chart("Figure 3. Cost per feature, relative to the run without hooks", bars,
                        "wall time relative to no hooks (1.0 = the program alone)",
                        [(7, "no hooks"), (1, "+ provenance and entry hooks"), (2, "+ observed edges")],
                        value_fmt=lambda v: f"{v:g}")
    return section("E7", "E7 Ablation",
                   "QCode only. Increments below zero (noise) are drawn as zero; the table keeps the raw ratios. "
                   "Provenance-only, the JIT threshold and the switch-copy variants need example flags that do not exist yet.",
                   table(["binary", "no hooks", "hooks ×", "hooks + edges ×", "steps × hooks", "steps × edges"],
                         body, numeric={1, 2, 3, 4, 5}) + fig)


def per_family(rows):
    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    body = []
    for f in sorted(fams):
        rs = fams[f]
        ok = [run_ok(r) for r in rs if not r.get("unsupported")]
        segs = [s for r in rs for s in code_segments(r)]
        recalls = [((r.get("truth") or {}).get("edges") or {}).get("recall") for r in rs]
        ratios = [ratio(cfg(r, "hooks"), cfg(r, "nohooks")) for r in rs if r["engine"] == "qcode"]
        regs = [(r.get("truth") or {}).get("regions") or {} for r in rs]
        exp = [g.get("expected") for g in regs if g.get("expected") is not None]
        body.append([esc(f), esc(FAMILY_TITLES.get(f, "")), fmt_int(len({r["name"] for r in rs})),
                     esc(", ".join(sorted({r["engine"] for r in rs}, key=engine_slot))),
                     f'{sum(1 for o in ok if o)} / {len(ok)}',
                     f'{sum(1 for s in segs if s.get("exact"))} / {len(segs)}' if segs else "–",
                     f'{sum(g.get("found") or 0 for g in regs)} / {sum(exp) if exp else "–"}',
                     fmt_frac(ub.median([x for x in recalls if x is not None])),
                     fmt_ratio(ub.median([x for x in ratios if x is not None])),
                     fmt_int(sum(1 for r in rs if r.get("unsupported")))])
    return ('<section id="families"><h2>Per family</h2>' +
            table(["family", "", "binaries", "engines", "runs correct", "code segs exact", "regions found / exp.",
                   "median recall", "median hooks × (qcode)", "unsupported"], body, numeric={2, 7, 8, 9}) + "</section>")


def progress(results):
    labels = results["labels"]
    order = sorted(labels, key=lambda l: labels[l]["meta"].get("date") or "")
    body, points = [], []
    for l in order:
        e = labels[l]
        m = e["meta"]
        mach = m.get("machine") or {}
        rows = e["rows"]
        ok = [run_ok(r) for r in rows if not r.get("unsupported")]
        segs = [s for r in rows for s in code_segments(r)]
        ratios = [ratio(cfg(r, "hooks"), cfg(r, "nohooks")) for r in rows if r["engine"] == "qcode"]
        med = ub.median([x for x in ratios if x is not None])
        points.append((l, med))
        commit = (m.get("commit") or "")[:10] + ("+dirty" if m.get("dirty") else "")
        body.append([esc(l), esc((m.get("date") or "")[:16].replace("T", " ")),
                     esc(f'{mach.get("host", "?")} · {mach.get("cpu", "?")}'), f"<code>{esc(commit)}</code>",
                     esc(", ".join(m.get("engines") or [])),
                     esc(m.get("families") if isinstance(m.get("families"), str) else ", ".join(m.get("families") or [])),
                     fmt_int(len(rows)), f'{sum(1 for o in ok if o)} / {len(ok)}',
                     f'{sum(1 for s in segs if s.get("exact"))} / {len(segs)}' if segs else "–", fmt_ratio(med)])
    fig = line_chart("Median QCode hooks ratio per label", points, "hooks / no hooks", value_fmt=lambda v: f"{v:.2f}×")
    return ('<section id="progress"><h2>Progress</h2><p class="claim">Every label ever reported, from results.json '
            '(append-only).</p>' +
            table(["label", "date (UTC)", "machine", "commit", "engines", "families", "rows", "runs correct",
                   "code segs exact", "median hooks ×"], body, numeric={6, 9}) + fig + "</section>")


CSS = """
.viz-root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;
--s7:#b9b7ae;--s8:#e34948;--good:#006300;--goodbg:#e3f3e3;--bad:#b02a2a;--badbg:#fbe6e6;--nabg:#efeee9}
@media (prefers-color-scheme:dark){.viz-root{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;
--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;
--s2:#d95926;--s3:#199e70;--s4:#c98500;--s7:#5a5955;--s8:#e66767;--good:#0ca30c;--goodbg:#132a13;--bad:#e66767;
--badbg:#2e1515;--nabg:#262624}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1240px;margin:0 auto;padding:24px 20px 60px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:0 0 6px}
section{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px 20px;margin:18px 0}
.claim,.meta{color:var(--ink2);margin:0 0 10px}.criterion{margin:0 0 12px;color:var(--ink2)}
.muted{color:var(--muted);font-size:12px}
nav a{color:var(--ink2);margin-right:12px;text-decoration:none;border-bottom:1px solid var(--axis)}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0 10px}
th{text-align:left;font-weight:600;color:var(--ink2);border-bottom:1px solid var(--axis);padding:5px 8px;white-space:nowrap}
td{border-bottom:1px solid var(--grid);padding:5px 8px;vertical-align:top}
.num{text-align:right;font-variant-numeric:tabular-nums}
td.empty{color:var(--muted);text-align:center}
code{font-size:12px}
.badge{display:inline-block;padding:0 7px;border-radius:9px;font-size:12px;white-space:nowrap}
.badge.good{color:var(--good);background:var(--goodbg)}.badge.bad{color:var(--bad);background:var(--badbg)}
.badge.na{color:var(--ink2);background:var(--nabg)}
figure{margin:8px 0 0}svg{display:block;width:100%;max-width:880px;height:auto}
svg text{fill:var(--ink2);font:12px system-ui,-apple-system,"Segoe UI",sans-serif}
svg .charttitle{fill:var(--ink);font-weight:600;font-size:13px}
svg .tick,svg .legend{fill:var(--muted);font-variant-numeric:tabular-nums}
svg .axislabel{fill:var(--ink2)}svg .value{fill:var(--ink2);font-size:11px}svg .na{fill:var(--muted);font-style:italic}
svg .grid{stroke:var(--grid);stroke-width:1}svg .axis{stroke:var(--axis);stroke-width:1}
svg .ref{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3}
svg .s1{fill:var(--s1)}svg .s2{fill:var(--s2)}svg .s3{fill:var(--s3)}svg .s4{fill:var(--s4)}
svg .s7{fill:var(--s7)}svg .s8{fill:var(--s8)}
svg .line{fill:none;stroke:var(--s1);stroke-width:2}svg .dot{stroke:var(--surface);stroke-width:2}
svg g:hover rect{opacity:.85}
"""


def render(results, label):
    entry = results["labels"][label]
    rows = sorted(entry["rows"], key=lambda r: (r["family"], r["name"], engine_slot(r["engine"])))
    m = entry["meta"]
    mach = m.get("machine") or {}
    head = (f'<h1>unpackbench</h1><p class="meta">Label <b>{esc(label)}</b> · {esc((m.get("date") or "")[:16].replace("T", " "))} UTC · '
            f'{esc(mach.get("host"))}, {esc(mach.get("cpu"))}, governor {esc(mach.get("governor"))} · '
            f'commit <code>{esc((m.get("commit") or "")[:10])}{"+dirty" if m.get("dirty") else ""}</code> · '
            f'{esc(m.get("reps"))} reps after one warm-up · configs {esc(", ".join(m.get("configs") or []))} · '
            f'corpus built {esc(m.get("manifest_built"))}</p>'
            '<nav>' + "".join(f'<a href="#{s}">{s}</a>' for s in
                              ("families", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "progress")) + '</nav>')
    parts = [head, per_family(rows), e1(rows), e2(rows), e3(rows), e4(rows), e5(rows), e6(rows), e7(rows),
             progress(results)]
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>unpackbench · {esc(label)}</title><style>{CSS}</style></head>'
            f'<body class="viz-root"><main>{"".join(parts)}</main></body></html>\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, default=ub.BENCH / "runs")
    ap.add_argument("--results", type=Path, default=ub.BENCH / "results.json")
    ap.add_argument("--html", type=Path, default=ub.BENCH / "report.html")
    ap.add_argument("--label", help="the label the experiment sections show (default: the most recent)")
    a = ap.parse_args()
    results, added = update_results(a.runs, a.results)
    if not results["labels"]:
        sys.exit("no labels to report")
    labels = results["labels"]
    label = a.label or max(labels, key=lambda l: labels[l]["meta"].get("date") or "")
    if label not in labels:
        sys.exit(f"unknown label {label}")
    a.html.write_text(render(results, label))
    print(f"[report] {added} new row(s) in {a.results}; {a.html} shows {label} "
          f"({len(labels[label]['rows'])} rows, {len(labels)} label(s))")


if __name__ == "__main__":
    main()
