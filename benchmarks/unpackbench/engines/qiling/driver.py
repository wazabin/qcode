#!/usr/bin/env python3
"""The unpack task under Qiling (pinned 1.4.6, in .venv).

Qiling runs the ELF with its own Linux loader and system calls. The hooks are
Qiling's: `hook_mem_write` over each provenance window stamps the storing
pc's site id into a shadow array, `hook_block` logs first entries (and, with
--edges, the block that ran before). One Python callback per event.

What Qiling 1.4.6 lacks for UPX's memfd stub is added as custom system calls
(`set_syscall`): memfd_create, ftruncate and msync on an in-memory file, and
MAP_SHARED write-back (a shared file mapping's bytes reach the file at msync,
at munmap, and before the file is mapped again, as userland/ does). Unknown
system calls answer ENOSYS rather than Qiling's default of leaving RAX alone.
The fork, exec and ptrace families stop the run as unsupported.

Usage: run <ELF> --out DIR [--edges] [--no-hooks] [--stdin FILE]
           [--budget N] [--reps N] -- args...
"""
import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))

# The unpack example's windows (engines/common/src/record.rs): the image
# plus 6 MiB, and the first ~2 MiB of the mmap arena.
PAGE = 4096
IMAGE_WINDOW_LEN = 6 << 20
MMAP_WINDOW_LEN = (((1 << 24) - 16 - 64) // 2 - IMAGE_WINDOW_LEN) & ~(PAGE - 1)
MAX_REGION = 4 << 20
MAX_BLOCKS = 1 << 20

ENOSYS = 38
UNSUPPORTED = {56: "clone", 57: "fork", 58: "vfork", 59: "execve", 61: "wait4",
               101: "ptrace", 247: "waitid", 322: "execveat", 435: "clone3"}


def hook_source_lines():
    """Non-blank, non-comment lines between the HOOK markers of this file."""
    n, inside = 0, False
    for line in open(__file__):
        t = line.strip()
        if t.startswith("# HOOK-BEGIN"):
            inside = True
        elif t.startswith("# HOOK-END"):
            inside = False
        elif inside and t and not t.startswith("#"):
            n += 1
    return n


class MemFd:
    """An anonymous in-memory file, enough of Qiling's file interface for the
    stub's write, lseek, fstat, mmap, ftruncate and close."""

    def __init__(self, name):
        self.name = f"/memfd:{name} (deleted)"
        self.data = bytearray()
        self.pos = 0
        self.closed = False

    def read(self, n=-1):
        end = len(self.data) if n is None or n < 0 else min(len(self.data), self.pos + n)
        out = bytes(self.data[self.pos:end])
        self.pos = max(self.pos, end)
        return out

    def write(self, b):
        b = bytes(b)
        end = self.pos + len(b)
        if len(self.data) < end:
            self.data.extend(b"\0" * (end - len(self.data)))
        self.data[self.pos:end] = b
        self.pos = end
        return len(b)

    def seek(self, off, whence=0):
        self.pos = (off if whence == 0 else self.pos + off if whence == 1 else len(self.data) + off)
        return self.pos

    lseek = seek

    def tell(self):
        return self.pos

    def close(self):
        self.closed = True

    def fileno(self):
        return -1

    def fstat(self):
        from qiling.os.posix.stat import Fstat
        st = Fstat(os.open("/dev/null", os.O_RDONLY))
        st.st_size = len(self.data)
        st.st_mode = 0o100600
        return st


class Kernel:
    """Qiling's syscalls, plus what the memfd stub needs."""

    def __init__(self, ql):
        self.ql = ql
        self.shared = []  # [MemFd, at, len, offset]
        self.exit = None
        self.unsupported = None
        self.enosys = {}

    def sync(self, addr, length):
        for f, at, ln, off in self.shared:
            lo, hi = max(at, addr), min(at + ln, addr + length)
            if lo < hi:
                b = self.ql.mem.read(lo, hi - lo)
                o = off + (lo - at)
                if len(f.data) < o + len(b):
                    f.data.extend(b"\0" * (o + len(b) - len(f.data)))
                f.data[o:o + len(b)] = b

    def install(self):
        from qiling.const import QL_INTERCEPT
        from qiling.os.posix.syscall import mman
        ql, k = self.ql, self

        def memfd_create(ql, name, flags):
            fd = next(i for i in range(3, 1024) if ql.os.fd[i] is None)
            ql.os.fd[fd] = MemFd(ql.os.utils.read_cstring(name))
            return fd

        def ftruncate(ql, fd, length):
            f = ql.os.fd[fd]
            if isinstance(f, MemFd):
                del f.data[length:]
                f.data.extend(b"\0" * (length - len(f.data)))
                return 0
            return -22

        def msync(ql, addr, length, flags):
            k.sync(addr, length)
            return 0

        def mmap(ql, addr, length, prot, flags, fd, pgoffset):
            fd = fd if fd < (1 << 63) else fd - (1 << 64)
            f = ql.os.fd[fd] if 0 <= fd < 1024 else None
            span = (length + PAGE - 1) & ~(PAGE - 1)
            if flags & 0x10:  # MAP_FIXED: carry the old shared bytes back first
                k.sync(addr, span)
                k.shared = [m for m in k.shared if m[1] >= addr + span or m[1] + m[2] <= addr]
            if isinstance(f, MemFd):
                k.sync(0, 1 << 64)
                f.pos = pgoffset
            at = mman.ql_syscall_mmap(ql, addr, length, prot, flags, fd, pgoffset)
            if isinstance(f, MemFd) and at < (1 << 63):
                # Qiling copies `length` bytes; Linux maps the whole last page,
                # which is where UPX leaves its exit trampoline.
                tail = bytes(f.data[pgoffset + length:pgoffset + span])
                if tail:
                    ql.mem.write(at + length, tail)
                if flags & 1:
                    k.shared.append([f, at, span, pgoffset])
            return at

        def munmap(ql, addr, length):
            span = (length + PAGE - 1) & ~(PAGE - 1)
            k.sync(addr, span)
            k.shared = [m for m in k.shared if m[1] >= addr + span or m[1] + m[2] <= addr]
            return mman.ql_syscall_munmap(ql, addr, length)

        def on_exit(ql, code, *rest):
            k.exit = code & 0xff

        ql.os.set_syscall(319, memfd_create)
        ql.os.set_syscall("ftruncate", ftruncate)
        ql.os.set_syscall(26, msync)
        ql.os.set_syscall("mmap", mmap)
        ql.os.set_syscall("munmap", munmap)
        ql.os.set_syscall("exit_group", on_exit, QL_INTERCEPT.ENTER)
        ql.os.set_syscall("exit", on_exit, QL_INTERCEPT.ENTER)
        for nr, name in UNSUPPORTED.items():
            def stop(ql, *a, nr=nr, name=name):
                k.unsupported = f"syscall {nr} ({name}, fork/exec/ptrace family): no process model"
                ql.stop()
                return -ENOSYS
            ql.os.set_syscall(nr, stop)
        for nr in (334,):  # rseq: glibc must see ENOSYS, not a stale RAX
            ql.os.set_syscall(nr, lambda ql, a, b, c, d: -ENOSYS)


def run_once(args):
    from qiling import Qiling
    from qiling.const import QL_VERBOSE
    from qiling.extensions import pipe

    t0 = time.perf_counter()
    argv = [os.path.abspath(args.elf)] + args.args
    verbose = QL_VERBOSE.DEFAULT if os.environ.get("UNPACKBENCH_STRACE") else QL_VERBOSE.DISABLED
    ql = Qiling(argv, "/", env={}, verbose=verbose)
    ql.os.stdin = pipe.SimpleInStream(0)
    ql.os.stdin.write(args.stdin)
    ql.os.stdout = pipe.SimpleOutStream(1)
    ql.os.stderr = pipe.SimpleOutStream(2)
    kernel = Kernel(ql)
    kernel.install()

    image = ql.loader.images[0]
    image_lo = image.base & ~(PAGE - 1)
    windows = [(image_lo, IMAGE_WINDOW_LEN), (ql.loader.mmap_address, MMAP_WINDOW_LEN)]
    shadows = [array("H", bytes(2 * n)) for _, n in windows]
    sites, site_of, blocks, block_of, log = [], {}, [], {}, []
    last = [None]
    edges = args.edges
    ids = [array("H", [0]) * 64]

    # HOOK-BEGIN
    def store_hook(start, shadow):
        def on_store(ql, access, addr, size, value):
            pc = ql.arch.regs.arch_pc
            sid = site_of.get(pc)
            if sid is None:
                sites.append((pc, size))
                sid = site_of[pc] = min(len(sites), 0xFFFF)
                ids.append(array("H", [sid]) * 64)
            at = addr - start
            shadow[at:at + size] = ids[sid][:size]
        return on_store

    def on_block(ql, addr, size):
        k = block_of.get(addr)
        if k is None:
            if len(blocks) >= MAX_BLOCKS:
                return
            k = block_of[addr] = len(blocks)
            blocks.append((addr, addr + size))
            log.append((k, last[0]))
        if edges:
            last[0] = k
    # HOOK-END

    if args.hooks:
        for (start, n), shadow in zip(windows, shadows):
            ql.hook_mem_write(store_hook(start, shadow), begin=start, end=start + n - 1)
        ql.hook_block(on_block)

    setup_ms = (time.perf_counter() - t0) * 1e3
    t1 = time.perf_counter()
    error = None
    try:
        if args.budget:
            ql.run(count=args.budget)
        else:
            ql.run()
    except Exception as e:  # a guest fault surfaces as a Unicorn error
        error = f"{type(e).__name__}: {e} at {ql.arch.regs.arch_pc:#x}"
    wall_ms = (time.perf_counter() - t1) * 1e3

    def drain(s):
        s.seek = lambda *a: 0
        return bytes(s.getvalue())

    out = dict(
        ql=ql, setup_ms=setup_ms, wall_ms=wall_ms, windows=windows, shadows=shadows,
        sites=sites, blocks=blocks, log=log,
        stdout=bytes(ql.os.stdout.getvalue()), stderr=bytes(ql.os.stderr.getvalue()),
        exit=kernel.exit, unsupported=kernel.unsupported,
    )
    if kernel.unsupported:
        out["stop"] = f"unsupported: {kernel.unsupported}"
    elif kernel.exit is not None:
        out["stop"] = "exit"
    elif error:
        out["stop"] = f"crashed: {error}"
    elif args.budget:
        out["stop"] = f"budget of {args.budget} instructions exhausted"
    else:
        out["stop"] = f"stopped at {ql.arch.regs.arch_pc:#x} without exiting"
    return out


# ---- harvest: engines/common/src/harvest.rs, in Python

def harvest(r, edges_recorded):
    windows, shadows = r["windows"], r["shadows"]

    def ids(s, e):
        if e <= s:
            return None
        for (start, n), sh in zip(windows, shadows):
            if start <= s < start + n:
                if e > start + n:
                    return None
                return sh[s - start:e - start]
        return None

    def in_window(a):
        return any(start <= a < start + n for start, n in windows)

    ranges = {}
    for a, e in r["blocks"]:
        ranges[a] = max(ranges.get(a, e), e)
    nodes = [dict(addr=a, end=max(e, a + 1), generated=False, generation=0, sites=[],
                  first_entry=None, executed=False) for a, e in sorted(ranges.items())]
    index = {n["addr"]: i for i, n in enumerate(nodes)}
    blocks = r["blocks"]

    def node_of_block(k):
        return index.get(blocks[k][0]) if k is not None and k < len(blocks) else None

    for n in nodes:
        v = ids(n["addr"], n["end"])
        if v is not None:
            seen = sorted(set(x for x in v if x))
            n["generated"], n["sites"] = bool(seen), seen
    for pos, (k, _) in enumerate(r["log"]):
        i = node_of_block(k)
        if i is not None and not nodes[i]["executed"]:
            nodes[i]["executed"], nodes[i]["first_entry"] = True, pos

    starts = [n["addr"] for n in nodes]
    import bisect

    def covering(a):
        at = bisect.bisect_right(starts, a)
        for back in range(1, min(at, 32) + 1):
            if a < nodes[at - back]["end"]:
                return at - back
        return None

    owner = [covering(pc) for pc, _ in r["sites"]]

    def site_gen(sid, of):
        p = owner[sid - 1] if 0 < sid <= len(owner) else None
        return nodes[p]["generation"] + 1 if p is not None and p != of else 1

    warnings = []
    gen = [i for i, n in enumerate(nodes) if n["generated"]]
    for i in gen:
        nodes[i]["generation"] = 1
    settled = not gen
    for _ in range(len(gen) + 1):
        changed = False
        for i in gen:
            want = max((site_gen(s, i) for s in nodes[i]["sites"]), default=1)
            if want != nodes[i]["generation"]:
                nodes[i]["generation"], changed = want, True
        if not changed:
            settled = True
            break
    if not settled:
        warnings.append("the generation fixed point did not settle: the write graph has a cycle, and the generations reported are a lower bound")
    outside = [n["addr"] for n in nodes if n["executed"] and not in_window(n["addr"])]
    if outside:
        warnings.append(f"{len(outside)} executed node(s) lie outside the provenance windows, the first at {outside[0]:#x}; their bytes carry no shadow and read as ungenerated")
    for n in nodes:
        for s in n["sites"]:
            pc = r["sites"][s - 1][0]
            if n["addr"] <= pc < n["end"]:
                warnings.append(f"the node at {n['addr']:#x} holds the store at {pc:#x} that wrote it: it modified its own code")
                break
    stamped = sorted(set(s for n in nodes for s in n["sites"]))
    unattributed = sum(1 for s in stamped if owner[s - 1] is None)
    if unattributed:
        warnings.append(f"{unattributed} store site(s) that wrote executed code are inside no node; what they wrote is one generation deep at most")

    edges = set()
    for n in nodes:
        for s in n["sites"]:
            p = owner[s - 1]
            if p is not None:
                edges.add((nodes[p]["addr"], n["addr"], "generated_by", None, s))
    if edges_recorded:
        for k, pred in r["log"]:
            to, frm = node_of_block(k), node_of_block(pred)
            if to is not None and frm is not None:
                edges.add((nodes[frm]["addr"], nodes[to]["addr"], "control_flow", "observed", None))

    def window_of(a):
        for start, n in windows:
            if start <= a < start + n:
                return start, start + n
        return None

    def grow_down(frm):
        w = window_of(frm - 1)
        if not w:
            return frm
        at = frm
        while at > w[0] and frm - at < MAX_REGION:
            v = ids(at - 1, at)
            if not v or v[0] == 0:
                break
            at -= 1
        return at

    def grow_up(frm, lo):
        w = window_of(frm)
        if not w:
            return frm
        at = frm
        while at < w[1] and at - lo < MAX_REGION:
            v = ids(at, at + 1)
            if not v or v[0] == 0:
                break
            at += 1
        return at

    seeds = [n for n in nodes if n["executed"] and n["generated"]]
    runs = []
    for n in seeds:
        v = ids(n["addr"], n["end"])
        if v is None:
            continue
        at = 0
        while at < len(v):
            if v[at] == 0:
                at += 1
                continue
            to = at
            while to < len(v) and v[to]:
                to += 1
            lo, hi = n["addr"] + at, n["addr"] + to
            at = to
            if any(lo >= a and hi <= b for a, b in runs):
                continue
            lo = grow_down(lo)
            hi = grow_up(hi, lo)
            if hi - lo >= MAX_REGION:
                warnings.append(f"the generated run at {lo:#x} reached the {MAX_REGION:#x}-byte harvest limit and is reported truncated")
            runs.append((lo, hi))
    merged = []
    for lo, hi in sorted(set(runs)):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    regions = []
    for lo, hi in merged:
        v = ids(lo, hi)
        at = 0
        while at < len(v):
            g = site_gen(v[at], None)
            to = at
            while to < len(v) and site_gen(v[to], None) == g:
                to += 1
            reg = (lo + at, lo + to, g)
            if any(n["addr"] < reg[1] and reg[0] < n["end"] for n in seeds):
                regions.append(reg)
            at = to
    return nodes, sorted(edges, key=lambda e: (e[0], e[1], e[2], e[3] or "", e[4] or 0)), sorted(regions), warnings


def write(args, r, walls, setups):
    out = args.out
    os.makedirs(out, exist_ok=True)
    shutil.rmtree(os.path.join(out, "regions"), ignore_errors=True)
    os.makedirs(os.path.join(out, "regions"))
    edges_recorded = args.hooks and args.edges
    nodes, edges, regions, warnings = harvest(r, edges_recorded) if args.hooks else ([], [], [], [])
    for lo, hi, g in regions:
        name = f"{lo:#x}-g{g}.bin"
        try:
            b = bytes(r["ql"].mem.read(lo, hi - lo))
        except Exception:
            b = bytes(hi - lo)
            warnings.append(f"the generated region at {lo:#x} is no longer readable; regions/{name} holds zeroes")
        open(os.path.join(out, "regions", name), "wb").write(b)
    h = lambda v: f"{v:#x}"
    exit_status = r["exit"]
    crashed = r["stop"] != "exit"

    def node_json(n):
        d = dict(addr=h(n["addr"]), range=[h(n["addr"]), h(n["end"])], generated=n["generated"],
                 generation=n["generation"], sites=n["sites"], executed=n["executed"])
        if n["first_entry"] is not None:
            d["first_entry"] = n["first_entry"]
        return d

    def edge_json(e):
        d = {"from": h(e[0]), "to": h(e[1]), "kind": e[2]}
        if e[3]:
            d["origin"] = e[3]
        if e[4]:
            d["site"] = e[4]
        return d

    graph = {
        "program": {"path": args.elf, "entry": h(r["ql"].loader.entry_point), "exit_status": exit_status,
                    "stopped": r["stop"], "steps": None, "strategy": "jit", "engine": "qiling"},
        "io": {"stdout": r["stdout"].decode("utf-8", "replace"), "stderr": r["stderr"].decode("utf-8", "replace")},
        "windows": [{"start": h(s), "end": h(s + n), "bytes_len": n} for s, n in r["windows"]],
        "nodes": [node_json(n) for n in nodes],
        "edges": [edge_json(e) for e in edges],
        "sites": [{"id": min(i + 1, 0xFFFF), "pc": h(pc), "size": size} for i, (pc, size) in enumerate(r["sites"])],
        "regions": [{"start": h(lo), "end": h(hi), "generation": g, "bytes_len": hi - lo} for lo, hi, g in regions],
        "flags": {"hooks": args.hooks, "edges_recorded": edges_recorded, "crashed": crashed,
                  "sites_saturated": len(r["sites"]) > 0xFFFF, "blocks_saturated": len(r["blocks"]) >= MAX_BLOCKS,
                  "evicted": 0, "static_edges": False},
        "warnings": warnings,
    }
    json.dump(graph, open(os.path.join(out, "graph.json"), "w"), indent=2)
    open(os.path.join(out, "stderr.txt"), "wb").write(r["stderr"])
    meta = base_meta(args)
    meta.update({
        "reps": len(walls), "wall_ms": walls, "setup_ms": setups, "steps": [None] * len(walls),
        "wall_scope": "run only: machine setup and image load are in setup_ms, harvest excluded",
        "exit": exit_status, "stdout_sha256": hashlib.sha256(r["stdout"]).hexdigest(),
        "crashed": crashed, "stop_reason": r["stop"], "vm": None,
        "regions": [{"start": h(lo), "end": h(hi), "generation": g} for lo, hi, g in regions],
    })
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2, sort_keys=True)
    return graph


def base_meta(args):
    import qiling
    import unicorn
    return {"engine": "qiling", "hook_source_lines": hook_source_lines(),
            "config": {"jit": True, "hooks": args.hooks, "edges": args.edges, "budget": args.budget},
            "versions": {"qiling": qiling.__version__, "unicorn": unicorn.__version__,
                         "python": sys.version.split()[0]}}


def unsupported(args, reason, stderr=b""):
    os.makedirs(args.out, exist_ok=True)
    for f in ("graph.json",):
        try:
            os.remove(os.path.join(args.out, f))
        except FileNotFoundError:
            pass
    shutil.rmtree(os.path.join(args.out, "regions"), ignore_errors=True)
    meta = base_meta(args)
    meta["unsupported"] = reason
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2, sort_keys=True)
    open(os.path.join(args.out, "stderr.txt"), "wb").write(stderr + f"\nunsupported: {reason}\n".encode())
    print(f"unsupported: {reason}", file=sys.stderr)


def main():
    argv = sys.argv[1:]
    guest = []
    if "--" in argv:
        i = argv.index("--")
        argv, guest = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(prog="run")
    p.add_argument("elf")
    p.add_argument("--out", required=True)
    p.add_argument("--edges", action="store_true")
    p.add_argument("--no-hooks", dest="hooks", action="store_false")
    p.add_argument("--stdin")
    p.add_argument("--budget", type=int)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--jit", action="store_true")
    p.add_argument("--interp", action="store_true")
    p.add_argument("--root")
    args, extra = p.parse_known_args(argv)
    args.args = extra + guest
    args.stdin = open(args.stdin, "rb").read() if args.stdin else b""

    total = args.reps + 1 if args.reps > 1 else 1
    walls, setups, r = [], [], None
    for i in range(total):
        try:
            r = run_once(args)
        except Exception as e:
            unsupported(args, f"Qiling cannot load it: {type(e).__name__}: {e}")
            return
        if r["unsupported"]:
            unsupported(args, r["unsupported"], r["stderr"])
            return
        if total > 1 and i == 0:
            continue
        walls.append(r["wall_ms"])
        setups.append(r["setup_ms"])
    g = write(args, r, walls, setups)
    print(f"engine:   qiling (unicorn {__import__('unicorn').__version__})")
    print(f"stopped:  {r['stop']}")
    print(f"exit:     {r['exit']}")
    print(f"wall:     {statistics.median(walls):.1f} ms (median of {len(walls)})")
    print(f"nodes:    {len(g['nodes'])}")
    print(f"sites:    {len(g['sites'])}")
    for reg in g["regions"]:
        print(f"region:   {reg['start']}..{reg['end']} gen {reg['generation']} ({reg['bytes_len']} bytes)")
    for w in g["warnings"]:
        print(f"warning:  {w}")
    sys.stdout.write(r["stdout"].decode("utf-8", "replace"))


if __name__ == "__main__":
    main()
