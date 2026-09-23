#!/usr/bin/env python3
"""An independent execution oracle: run a binary under qemu-x86_64 and record
the guest address of every basic block it executes.

This is deliberately a *different* engine from the one under test, so recall
measured against it is not circular (the reviewer's first point): qemu lifts
and runs the packed program with no knowledge of our provenance hooks, and its
`-d exec` trace names the guest PC of each translation block. We keep the set
of those PCs as `truth/exec_addrs.json`; truth.py then asks how many of them
fall inside a block our graph actually executed.

Usage:
  qemu_oracle.py --bin PATH --out truth/exec_addrs.json [--stdin FILE]
                 [--launch static|ldso] [--timeout S] [-- ARGS...]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

# "Trace 0: 0x.. [xxxxxxxx/<guest_pc>/xxxxxxxx/xxxxxxxx]"
TRACE = re.compile(rb"\[[0-9a-f]+/([0-9a-f]+)/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stdin")
    ap.add_argument("--launch", default="static")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()

    qemu = "qemu-x86_64"
    cmd = [qemu, "-d", "exec"]
    if a.launch == "ldso":
        # Run the system loader as the program, as the harness does.
        cmd += ["/lib64/ld-linux-x86-64.so.2", os.path.abspath(a.bin)]
    else:
        cmd += [os.path.abspath(a.bin)]
    cmd += a.args

    with tempfile.NamedTemporaryFile(suffix=".qlog", delete=False) as tf:
        log = tf.name
    cmd = [qemu, "-d", "exec", "-D", log] + cmd[3:]
    stdin = open(a.stdin, "rb") if a.stdin else subprocess.DEVNULL
    try:
        subprocess.run(cmd, stdin=stdin, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=a.timeout, check=False)
    except subprocess.TimeoutExpired:
        pass
    finally:
        if stdin not in (subprocess.DEVNULL, None):
            stdin.close()

    addrs = set()
    try:
        with open(log, "rb") as f:
            for line in f:
                m = TRACE.search(line)
                if m:
                    addrs.add(int(m.group(1), 16))
    finally:
        try:
            os.unlink(log)
        except OSError:
            pass

    if not addrs:
        print(f"qemu_oracle: no executed blocks captured for {a.bin}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"engine": "qemu-x86_64", "blocks": sorted(addrs)}, f)
    print(f"qemu_oracle: {len(addrs)} executed block addresses -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
