#!/usr/bin/env python3
"""Compare runs against the corpus ground truth (CONTRACT.md, truth.py output).

    truth.py [--corpus DIR] runs/<label> [runs/<label>/<family>/<name>/<engine> ...]

Writes truth.json beside every meta.json that has a graph.json.

- segments: each expected segment (truth/segments.json) against the run's
  regions/. The expected bytes come from truth/segments/<0xstart>.bin, else
  from the reference seed ELF (manifest `reference`) at those addresses; either
  way they must hash to the listed sha256. Without bytes only `exact` is known
  (sha256 of the recovered bytes) and mismatch is null.
  recovered = covered bytes equal to the expected ones; mismatch = covered
  bytes that differ; extra = bytes of the regions touching the segment that
  lie outside every expected segment (the UPX exit trampoline, say).
- edges: the observed control-flow edges of the reference run (the seed's run
  in the same label, same engine) or truth/edges.json, against the observed
  edges of this run. Pending when the reference has not run.
- static_miss: observed edges absent from the static control_flow edges of
  the same graph.
- sites: expected (pc, generation) pairs from truth/sites.json against the
  graph's sites, a site's generation being that of the code it wrote.
"""

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import ub  # noqa: E402

H = ub.hexint


# --- ELF: the bytes a loader would put at [start, end) -----------------------

def elf_bytes(path, start, end):
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    if data[:4] != b"\x7fELF" or data[4] != 2:
        return None
    phoff, = struct.unpack_from("<Q", data, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
    out = bytearray(end - start)
    covered = 0
    for i in range(phnum):
        p_type, _flags, off, vaddr, _pa, filesz, memsz, _al = struct.unpack_from(
            "<IIQQQQQQ", data, phoff + i * phentsize)
        if p_type != 1:  # PT_LOAD
            continue
        lo, hi = max(start, vaddr), min(end, vaddr + memsz)
        if lo >= hi:
            continue
        file_hi = min(hi, vaddr + filesz)
        if lo < file_hi:
            out[lo - start:file_hi - start] = data[off + lo - vaddr:off + file_hi - vaddr]
        covered += hi - lo
    return bytes(out) if covered == end - start else None


# --- the run ----------------------------------------------------------------

def load_regions(run_dir, graph):
    """[(start, end, generation, bytes)], one per region, highest generation
    last so it wins where two overlap."""
    regions = []
    for r in graph.get("regions", []):
        start, end, gen = H(r["start"]), H(r["end"]), r.get("generation", 0)
        path = run_dir / "regions" / f"{start:#x}-g{gen}.bin"
        data = path.read_bytes() if path.exists() else None
        regions.append((start, end, gen, data))
    regions.sort(key=lambda r: r[2])
    return regions


def expected_bytes(seg, truth_dir, reference_bin):
    start, end = H(seg["start"]), H(seg["end"])
    candidates = [truth_dir / "segments" / f"{start:#x}.bin", truth_dir / f"{start:#x}.bin"]
    for c in candidates:
        if c.exists():
            data = c.read_bytes()
            if len(data) == end - start and (not seg.get("sha256") or ub.sha256_bytes(data) == seg["sha256"]):
                return data, str(c)
    if reference_bin is not None:
        data = elf_bytes(reference_bin, start, end)
        if data is not None and (not seg.get("sha256") or ub.sha256_bytes(data) == seg["sha256"]):
            return data, str(reference_bin)
    return None, None


def compare_segments(segments, regions, truth_dir, reference_bin):
    spans = [(H(s["start"]), H(s["end"])) for s in segments]

    def outside_all(lo, hi):
        """Bytes of [lo, hi) outside every expected segment."""
        n = hi - lo
        for s, e in spans:
            n -= max(0, min(hi, e) - max(lo, s))
        return max(n, 0)

    out = []
    touched = set()
    for seg in segments:
        start, end = H(seg["start"]), H(seg["end"])
        exp, source = expected_bytes(seg, truth_dir, reference_bin)
        got = bytearray(end - start)
        have = bytearray(end - start)  # 1 where a region covers the byte
        extra = 0
        for i, (rs, re_, _gen, data) in enumerate(regions):
            lo, hi = max(start, rs), min(end, re_)
            if lo >= hi or data is None:
                continue
            touched.add(i)
            got[lo - start:hi - start] = data[lo - rs:hi - rs]
            have[lo - start:hi - start] = b"\x01" * (hi - lo)
            extra += outside_all(rs, re_)
        covered = sum(have)
        row = {"start": seg["start"], "end": seg["end"], "kind": seg.get("kind", "code"),
               "expected": end - start, "covered": covered, "extra": extra}
        if exp is not None:
            mismatch = 0
            if covered:
                for i in range(end - start):
                    if have[i] and got[i] != exp[i]:
                        mismatch += 1
            row.update(recovered=covered - mismatch, mismatch=mismatch,
                       missing=(end - start) - covered, source=source)
            row["exact"] = mismatch == 0 and covered == end - start
            # Faithful recovery where a byte was captured, apart from full
            # coverage: the UPX stub never rewrites the 8-byte ELF e_ident,
            # so a byte-perfect code segment still reads 8 bytes short.
            row["exact_where_covered"] = mismatch == 0 and covered > 0
            row["coverage"] = covered / (end - start) if end > start else 1.0
            if not row["exact"] and covered:
                # Where the first difference or gap is: a pointer for the reader.
                for i in range(end - start):
                    if not have[i] or got[i] != exp[i]:
                        row["first_diff"] = f"{start + i:#x}"
                        break
        else:
            exact = covered == end - start and ub.sha256_bytes(bytes(got)) == seg.get("sha256")
            row.update(recovered=(end - start) if exact else None, mismatch=0 if exact else None,
                       missing=(end - start) - covered, exact=exact,
                       source="sha256 only (no expected bytes)")
        out.append(row)
    unattributed = sum((r[1] - r[0]) for i, r in enumerate(regions) if i not in touched)
    return out, unattributed


def observed(graph):
    # Intra-task control-flow edges the entry log witnessed. The schema uses
    # kind "observed"; a cross-task "ptrace" edge is a causal redirection, not
    # a control-flow edge, and is excluded from recall.
    return {(H(e["from"]), H(e["to"])) for e in graph.get("edges", [])
            if e.get("kind") == "observed"}


def ptrace_edges(graph):
    return {(H(e["from"]), H(e["to"])) for e in graph.get("edges", [])
            if e.get("kind") == "ptrace"}


def static(graph):
    return {(H(e["from"]), H(e["to"])) for e in graph.get("edges", [])
            if e.get("kind") == "static"}


def site_pairs(graph):
    """{(pc, generation written)}: the graph's sites by pc, each with the
    generation of the nodes it wrote."""
    pcs = {s["id"]: H(s["pc"]) for s in graph.get("sites", [])}
    pairs = set()
    for n in graph.get("nodes", []):
        for sid in n.get("sites", []):
            if sid in pcs:
                pairs.add((pcs[sid], n.get("generation", 0)))
    return pairs



# The high mmap arena differs per engine (qemu and our emulator pick different
# bases), so only image-range addresses are comparable across engines.
MMAP_FLOOR = 0x7F00_0000_0000


def exec_oracle(graph, truth_dir):
    """Independent recall: of the block addresses qemu executed in the image
    range, how many fall inside a block our graph actually executed. Not
    circular — qemu is a different engine with no view of our hooks."""
    data = ub.load_json(truth_dir / "exec_addrs.json")
    if data is None:
        return None
    blocks = [b for b in data.get("blocks", [])]
    image = [b for b in blocks if b < MMAP_FLOOR]
    spans = sorted((lo, hi) for lo, hi in
                   ((H(n["range"][0]), H(n["range"][1])) for n in graph.get("nodes", [])
                    if n.get("executed") and n.get("range"))
                   if lo < MMAP_FLOOR)

    def covered(a):
        for lo, hi in spans:
            if lo <= a < hi:
                return True
            if lo > a:
                break
        return False

    hit = sum(1 for a in image if covered(a))
    return {"engine": data.get("engine", "qemu-x86_64"),
            "blocks": len(blocks), "image_blocks": len(image),
            "covered": hit, "miss": len(image) - hit,
            "coverage": (hit / len(image)) if image else None}


def judge(run_dir, corpus_root, label_dir):
    meta = ub.load_json(run_dir / "meta.json")
    graph = ub.load_json(run_dir / "graph.json")
    if meta is None or graph is None or meta.get("unsupported"):
        return None
    b = meta["binary"]
    family, engine = b["family"], meta["engine"]
    truth_dir = Path(b.get("truth_dir") or ub.resolve(b["path"], corpus_root).parent / "truth")
    reference_bin = ub.resolve(b.get("reference"), corpus_root) if b.get("reference") else None
    t = {"family": family, "name": b["name"], "engine": engine, "config": meta.get("config")}

    # Segments.
    segments = ub.load_json(truth_dir / "segments.json")
    regions = load_regions(run_dir, graph)
    if segments is not None:
        t["segments"], t["extra_unattributed"] = compare_segments(
            segments, regions, truth_dir, reference_bin)
    else:
        t["segments"] = []
    code = [s for s in (segments or []) if s.get("kind", "code") == "code"]
    if segments is not None:
        expected_regions = len(code)
    elif family == "1.5":
        expected_regions = 0
    else:
        expected_regions = None
    t["regions"] = {"expected": expected_regions, "found": len(regions),
                    "bytes": sum(r[1] - r[0] for r in regions),
                    "generations": max((r[2] for r in regions), default=0)}

    # Sites.
    found = site_pairs(graph)
    pcs = {x["id"]: H(x["pc"]) for x in graph.get("sites", [])}
    writer_ids = {i for n in graph.get("nodes", []) for i in n.get("sites", []) if i in pcs}
    sites = ub.load_json(truth_dir / "sites.json")
    t["sites"] = {"expected": None if sites is None else len(sites),
                  "found": len(found),
                  "correct": None if sites is None else
                  len({(H(s["pc"]), s["generation"]) for s in sites} & found),
                  # writer sites per lifted copy vs per pc (plan §3, the known duplication)
                  "copies": len(writer_ids),
                  "pcs": len({pcs[i] for i in writer_ids}),
                  "all_sites": len(graph.get("sites", []))}

    # Edges.
    obs = observed(graph)
    mode = (b.get("truth") or {}).get("edges")
    if mode == "ptrace":
        # Family 1.4: the cross-task control-flow decisions a tracer makes,
        # scored against the hand-listed ptrace edges. Not circular: these are
        # the tracer's SETREGS/CONT redirections, read from the disassembly.
        listed = ub.load_json(truth_dir / "edges.json") or []
        want = {(H(e["from"]), H(e["to"])) for e in listed}
        got = ptrace_edges(graph)
        t["edges"] = {"mode": "ptrace", "reference": len(want), "found": len(want & got),
                      "recall": (len(want & got) / len(want)) if want else None,
                      "observed": len(got), "kind": "ptrace"}
        t["exec_oracle"] = None
        t["tasks"] = len({n["task"] for n in graph.get("nodes", [])})
        nodes = graph.get("nodes", [])
        t["graph"] = {"nodes": len(nodes),
                      "executed": sum(1 for n in nodes if n.get("executed")),
                      "generated": sum(1 for n in nodes if n.get("generated")),
                      "static_edges": len(static(graph)), "observed_edges": len(obs),
                      "ptrace_edges": len(got),
                      "warnings": len(graph.get("warnings", []))}
        t["static_miss"] = None
        t["regions"] = {"expected": None, "found": len(regions),
                        "bytes": sum(r[1]-r[0] for r in regions),
                        "generations": max((r[2] for r in regions), default=0)}
        t["sites"] = {"expected": None, "found": len(site_pairs(graph)),
                      "correct": None, "copies": None, "pcs": None,
                      "all_sites": len(graph.get("sites", []))}
        t["run_ok"] = (not meta.get("crashed")) and meta.get("exit_ok") is not False
        ub.write_json(run_dir / "truth.json", t)
        return t
    ref = None
    edges = {"reference": None, "found": None, "recall": None, "observed": len(obs)}
    if not graph.get("flags", {}).get("edges_recorded", bool(obs)):
        edges["pending"] = "the kept run recorded no observed edges (no --edges)"
    elif mode == "file" or (mode is None and (truth_dir / "edges.json").exists()):
        listed = ub.load_json(truth_dir / "edges.json") or []
        ref = {(H(e["from"]), H(e["to"])) for e in listed}
        edges["source"] = "truth/edges.json"
    elif mode == "reference" or (mode is None and family == "1.5"):
        fn = ub.ref_family_name(b.get("reference")) or (family, b["name"])
        ref_dir = ub.entry_dir(label_dir, fn[0], fn[1], engine)
        ref_graph = ub.load_json(ref_dir / "graph.json")
        if ref_graph is None or not observed(ref_graph):
            edges["pending"] = f"no reference run with edges at {fn[0]}/{fn[1]}/{engine} in this label"
        else:
            ref = observed(ref_graph)
            edges["source"] = f"run {fn[0]}/{fn[1]}/{engine}"
    if ref is not None:
        hit = ref & obs
        edges.update(reference=len(ref), found=len(hit),
                     recall=(len(hit) / len(ref)) if ref else None)
    t["edges"] = edges
    t["exec_oracle"] = exec_oracle(graph, truth_dir)
    # An engine without a static CFG (flags.static_edges false) has no static miss.
    has_static = graph.get("flags", {}).get("static_edges", True) is not False
    t["static_miss"] = len(obs - static(graph)) if obs and has_static else None

    # The run itself.
    nodes = graph.get("nodes", [])
    t["graph"] = {"nodes": len(nodes), "executed": sum(1 for n in nodes if n.get("executed")),
                  "generated": sum(1 for n in nodes if n.get("generated")),
                  "static_edges": len(static(graph)), "observed_edges": len(obs),
                  "warnings": len(graph.get("warnings", []))}
    t["run_ok"] = (not meta.get("crashed")) and meta.get("exit_ok") is not False \
        and meta.get("stdout_ok") is not False
    ub.write_json(run_dir / "truth.json", t)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=ub.BENCH / "corpus")
    ap.add_argument("paths", nargs="+", type=Path)
    a = ap.parse_args()
    corpus_root = a.corpus.resolve().parent
    for path in a.paths:
        path = path.resolve()
        metas = [path / "meta.json"] if (path / "meta.json").exists() else sorted(path.glob("*/*/*/meta.json"))
        for m in metas:
            run_dir = m.parent
            label_dir = run_dir.parents[2]
            t = judge(run_dir, corpus_root, label_dir)
            if t is None:
                continue
            segs = t["segments"]
            exact = sum(1 for s in segs if s.get("exact"))
            e = t["edges"]
            recall = "pending" if e.get("pending") else (
                "-" if e["recall"] is None else f"{e['recall']:.3f}")
            print(f"[truth] {t['family']}/{t['name']}/{t['engine']}: segments {exact}/{len(segs)} exact, "
                  f"regions {t['regions']['found']}/{t['regions']['expected']}, recall {recall}, "
                  f"static_miss {t['static_miss']}")


if __name__ == "__main__":
    main()
