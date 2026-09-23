#!/usr/bin/env python3
"""The sweep behind run-all.sh: every (binary, engine, config) of the corpus,
REPS timed runs after one discarded warm-up, into runs/<label>/.

Configured by the environment run-all.sh passes through (see there). A run
that times out, crashes, or is unsupported is recorded in meta.json and the
sweep moves on.
"""

import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ub  # noqa: E402

SPREAD_LIMIT = 0.10  # plan §4: rerun any point with more than 10 % spread


def env_list(name, default):
    value = os.environ.get(name, "").strip()
    return [v.strip() for v in value.split(",") if v.strip()] if value else default


def git(*args):
    try:
        return subprocess.run(["git", "-C", str(ub.REPO), *args], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        return None


def machine():
    cpu = None
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    governor = None
    try:
        governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        pass
    return {"host": socket.gethostname(), "cpu": cpu, "cores": os.cpu_count(),
            "kernel": platform.release(), "governor": governor,
            "python": platform.python_version()}


def run_once(driver, cfg, out, entry, bin_path, stdin, budget, timeout, log_stem):
    """One driver invocation. Returns the result dict (with `timeout` set on
    a timeout) and keeps the driver's stdout/stderr beside `log_stem`."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    argv = entry.get("argv") or []
    ldso = (entry.get("launch") or "static") == "ldso"
    if driver.name == "driver":
        cmd = [str(driver), "--config", cfg, "--out", str(out), "--budget", str(budget),
               "--launch", "ldso" if ldso else "static"]
        if stdin:
            cmd += ["--stdin", str(stdin)]
        cmd += ["--", str(bin_path), *argv[1:]]
    else:
        # The unpack example's own command line (engines/<e>/run, the
        # baselines): one run per call, contract meta.json into --out.
        prog = [LDSO, "--root", "/"] if ldso else [str(bin_path)]
        args = ([str(bin_path)] if ldso else []) + list(argv[1:])
        cmd = [str(driver), *prog, "--out", str(out), "--reps", "1", *RUN_FLAGS[cfg]]
        if os.environ.get("BASELINE_BUDGET"):
            cmd += ["--budget", os.environ["BASELINE_BUDGET"]]
        if stdin:
            cmd += ["--stdin", str(stdin)]
        cmd += ["--", *args]
    started = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)
    try:
        so, se = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        so, se = proc.communicate()
        timed_out = True
    elapsed = (time.perf_counter() - started) * 1e3
    Path(str(log_stem) + ".stdout").write_bytes(so)
    Path(str(log_stem) + ".stderr").write_bytes(se)
    if timed_out:
        return {"timeout": True, "crashed": True, "stop_reason": f"timeout after {timeout} s",
                "wall_ms": None, "proc_ms": elapsed}
    result = ub.load_json(out / "result.json")
    if result is None and driver.name != "driver":
        result = from_run_meta(ub.load_json(out / "meta.json"))
    if result is None:
        return {"crashed": True, "wall_ms": None, "proc_ms": elapsed,
                "stop_reason": f"driver failed (status {proc.returncode}, no result.json)"}
    return result


LDSO = os.environ.get("LDSO", "/lib64/ld-linux-x86-64.so.2")
RUN_FLAGS = {"nohooks": ["--jit", "--no-hooks"], "hooks": ["--jit"],
             "edges": ["--jit", "--edges"], "interp": ["--interp", "--edges"]}


def from_run_meta(m):
    """A `run`-style engine's meta.json, as the result a driver would write."""
    if m is None:
        return None
    if m.get("unsupported"):
        return {"unsupported": m["unsupported"]}
    walls = m.get("wall_ms") or [None]
    steps = m.get("steps") or [None]
    return {"wall_ms": walls[-1], "steps": steps[-1], "exit": m.get("exit"),
            "stop_reason": m.get("stop_reason"), "crashed": bool(m.get("crashed")),
            "stdout_sha256": m.get("stdout_sha256"), "vm": m.get("vm"),
            "setup_ms": (m.get("setup_ms") or [None])[-1], "wall_scope": m.get("wall_scope"),
            "hook_source_lines": m.get("hook_source_lines")}


def graph_digest(out):
    """sha256 of graph.json with `program.strategy` substituted to "jit", so
    the interpreter's graph and the JIT's compare byte for byte (as the
    unpack tests do)."""
    g = out / "graph.json"
    if not g.exists():
        return None
    text = g.read_bytes().replace(b'"strategy": "interpreter"', b'"strategy": "jit"')
    return ub.sha256_bytes(text)


def graph_counts(out):
    """Node, edge and region counts of a run's graph, for E6's deltas."""
    g = ub.load_json(out / "graph.json")
    if g is None:
        return None
    nodes, edges = g.get("nodes", []), g.get("edges", [])
    return {"nodes": len(nodes), "executed": sum(1 for n in nodes if n.get("executed")),
            "generated": sum(1 for n in nodes if n.get("generated")),
            "static_edges": sum(1 for e in edges if e.get("origin") == "static"),
            "observed_edges": sum(1 for e in edges if e.get("origin") == "observed"),
            "sites": len(g.get("sites", [])), "regions": len(g.get("regions", []))}


def run_config(driver, cfg, entry, bin_path, stdin, reps, budget, timeout, logs, scratch):
    """REPS + 1 runs of one configuration; the first is the warm-up."""
    record = {"config": ub.CONFIGS[cfg], "wall_ms": [], "proc_ms": [], "steps": [],
              "graph_sha256": []}
    last_out = None
    for i in range(reps + 1):
        out = scratch / f"{cfg}-{i}"
        r = run_once(driver, cfg, out, entry, bin_path, stdin, budget, timeout,
                     logs / f"{cfg}-{i}")
        if "unsupported" in r:
            return {"unsupported": r["unsupported"]}, None
        if r.get("timeout"):
            record.update(timeout=True, crashed=True, stop_reason=r["stop_reason"])
            break
        for key in ("exit", "stop_reason", "crashed", "stdout_sha256", "vm", "wall_scope",
                    "hook_source_lines"):
            record[key] = r.get(key)
        if i == 0:
            record["warmup_wall_ms"] = r.get("wall_ms")
        else:
            record["wall_ms"].append(r.get("wall_ms"))
            record["proc_ms"].append(r.get("proc_ms"))
            record["steps"].append(r.get("steps"))
            record["graph_sha256"].append(graph_digest(out))
        if last_out is not None and last_out != out:
            shutil.rmtree(last_out, ignore_errors=True)
        last_out = out
        if r.get("crashed") and i == 0:
            # A crash is deterministic enough: one run records it.
            record["wall_ms"].append(r.get("wall_ms"))
            record["steps"].append(r.get("steps"))
            record["graph_sha256"].append(graph_digest(out))
            break
    if last_out is not None:
        record["graph"] = graph_counts(last_out)
    walls = [w for w in record["wall_ms"] if w is not None]
    record["wall_ms_median"] = ub.median(walls)
    record["steps_median"] = ub.median(record["steps"])
    if walls and record["wall_ms_median"]:
        record["spread"] = (max(walls) - min(walls)) / record["wall_ms_median"]
    return record, last_out


def main():
    corpus = Path(os.environ.get("CORPUS", ub.BENCH / "corpus")).resolve()
    manifest = ub.load_json(corpus / "manifest.json")
    if manifest is None:
        sys.exit(f"no manifest at {corpus / 'manifest.json'}")
    root = corpus.parent  # manifest paths are relative to the bench root
    runs = Path(os.environ.get("RUNS", ub.BENCH / "runs")).resolve()
    engines = env_list("ENGINES", ["qcode"])
    families = env_list("FAMILIES", None)
    names = env_list("NAMES", None)
    configs = env_list("CONFIGS", ["nohooks", "hooks", "edges"])
    reps = int(os.environ.get("REPS", "5"))
    budget = int(float(os.environ.get("BUDGET", "5e9")))
    timeout = int(os.environ.get("TIMEOUT", "600"))
    label = os.environ.get("LABEL") or datetime.now().strftime("%Y%m%d-%H%M")
    for c in configs:
        if c not in ub.CONFIGS:
            sys.exit(f"unknown config {c}; known: {', '.join(ub.CONFIGS)}")

    label_dir = runs / label
    label_dir.mkdir(parents=True, exist_ok=True)
    label_meta = ub.load_json(label_dir / "label.json") or {}
    label_meta.update({
        "label": label,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": machine(),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "engines": sorted(set(label_meta.get("engines", [])) | set(engines)),
        "families": families or "all",
        "configs": configs, "reps": reps, "budget": budget, "timeout_s": timeout,
        "corpus": str(corpus), "manifest_built": manifest.get("built"),
        "tools": manifest.get("tools"),
    })
    ub.write_json(label_dir / "label.json", label_meta)

    binaries = [b for b in manifest.get("binaries", [])
                if (families is None or b["family"] in families)
                and (names is None or b["name"] in names)]
    # References (family 1.5) first, so truth can use them for the others.
    binaries.sort(key=lambda b: (b["family"] != "1.5", b["family"], b["name"]))
    print(f"[sweep] label {label}: {len(binaries)} binaries x {engines} x {configs}, "
          f"{reps} reps + warm-up", flush=True)

    for engine in engines:
        edir = ub.BENCH / "engines" / engine
        driver = edir / "driver" if (edir / "driver").exists() else edir / "run"
        spec = ub.load_json(edir / "engine.json") or {}
        hook_lines = sum(filter(None, (ub.count_source_lines(ub.REPO / p)
                                       for p in spec.get("hook_source", [])))) or None
        supported = set(spec.get("configs", ub.CONFIGS))
        for entry in binaries:
            run_dir = ub.entry_dir(label_dir, entry["family"], entry["name"], engine)
            if run_dir.exists():
                shutil.rmtree(run_dir)
            logs = run_dir / "logs"
            logs.mkdir(parents=True)
            bin_path = ub.resolve(entry["path"], root)
            truth_dir = bin_path.parent / "truth"
            stdin = ub.resolve(entry["stdin"], root) if entry.get("stdin") else None
            if stdin is None and (truth_dir / "stdin").exists():
                stdin = truth_dir / "stdin"
            meta = {"engine": engine, "label": label,
                    "binary": dict(entry, abspath=str(bin_path), truth_dir=str(truth_dir),
                                   sha256=ub.sha256_file(bin_path) if bin_path.exists() else None),
                    "reps": reps, "hook_source_lines": hook_lines, "configs": {}}
            print(f"[sweep] {engine} {entry['family']}/{entry['name']}", flush=True)
            if not driver.exists():
                meta["unsupported"] = f"no driver for engine {engine}"
            elif not bin_path.exists():
                meta["unsupported"] = f"missing binary {bin_path}"
            kept = {}
            with tempfile.TemporaryDirectory(prefix="unpackbench-") as scratch:
                scratch = Path(scratch)
                for cfg in configs if "unsupported" not in meta else []:
                    if cfg not in supported:
                        meta["configs"][cfg] = {"unsupported": f"{engine} has no {cfg} configuration"}
                        continue
                    record, out = run_config(driver, cfg, entry, bin_path, stdin, reps,
                                             budget, timeout, logs, scratch)
                    spread = record.get("spread")
                    if spread is not None and spread > SPREAD_LIMIT and reps > 1:
                        again, out2 = run_config(driver, cfg, entry, bin_path, stdin, reps,
                                                 budget, timeout, logs, scratch / "rerun")
                        if again.get("spread") is not None and again["spread"] < spread:
                            again["rerun_of_spread"] = spread
                            record, out = again, out2
                        else:
                            record["rerun_spread"] = again.get("spread")
                    if meta.get("hook_source_lines") is None and record.get("hook_source_lines"):
                        meta["hook_source_lines"] = record["hook_source_lines"]
                    meta["configs"][cfg] = record
                    if out is not None:
                        kept[cfg] = out
                    mark = record.get("unsupported") or record.get("stop_reason")
                    print(f"[sweep]   {cfg:8} median {record.get('wall_ms_median')} ms, "
                          f"{mark}", flush=True)
                # The graph the directory keeps: the richest configuration run.
                primary = next((c for c in ub.PRIMARY_ORDER if c in kept
                                and "unsupported" not in meta["configs"][c]), None)
                if primary is not None:
                    for item in kept[primary].iterdir():
                        if item.name in ("result.json", "meta.json", "stderr.txt"):
                            continue
                        shutil.move(str(item), run_dir / item.name)
            if primary is None and "unsupported" not in meta:
                reasons = {c: r.get("unsupported") for c, r in meta["configs"].items()}
                if reasons and all(reasons.values()):
                    meta["unsupported"] = "; ".join(sorted(set(reasons.values())))
            if primary is not None:
                rec = meta["configs"][primary]
                expect = entry.get("expect") or {}
                meta.update({
                    "primary": primary,
                    "config": rec["config"],
                    "wall_ms": rec["wall_ms"], "steps": rec["steps"],
                    "exit": rec.get("exit"), "stdout_sha256": rec.get("stdout_sha256"),
                    "crashed": bool(rec.get("crashed")), "stop_reason": rec.get("stop_reason"),
                    "vm": rec.get("vm"),
                    "exit_ok": None if "exit" not in expect else rec.get("exit") == expect["exit"],
                    "stdout_ok": None if "stdout_sha256" not in expect
                    else rec.get("stdout_sha256") == expect["stdout_sha256"],
                })
                last = logs / f"{primary}-{reps}"
                if not Path(str(last) + ".stderr").exists():
                    last = logs / f"{primary}-0"
                for ext in ("stderr", "stdout"):
                    src = Path(f"{last}.{ext}")
                    if src.exists():
                        shutil.copy(src, run_dir / f"{ext}.txt")
            ub.write_json(run_dir / "meta.json", meta)

    label_meta["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ub.write_json(label_dir / "label.json", label_meta)


if __name__ == "__main__":
    main()
