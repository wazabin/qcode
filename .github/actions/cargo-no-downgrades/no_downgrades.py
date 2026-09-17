#!/usr/bin/env python3
"""Refuses a Cargo.lock that pins any package below the base branch's version.

Usage: no_downgrades.py <base Cargo.lock> <head Cargo.lock>

A dependency bumped on main and then pinned back by a branch that was cut
before the bump is the usual way a downgrade slips into a merge. Only
packages the base already locks are compared, by (name, source); a package
locked at several versions is compared by its highest.
"""

import sys
import tomllib


def versions(path):
    with open(path, "rb") as file:
        lock = tomllib.load(file)
    locked = {}
    for package in lock.get("package", []):
        key = (package["name"], package.get("source"))
        version = parse(package["version"])
        locked[key] = max(locked.get(key, version), version)
    return locked


def parse(version):
    # Build metadata (`1.2.3+llvm`) does not order versions.
    version, _, _ = version.partition("+")
    release, _, pre = version.partition("-")
    # A pre-release sorts below the release it precedes.
    return tuple(int(part) for part in release.split(".")), pre == "", pre


def main(base_path, head_path):
    base = versions(base_path)
    head = versions(head_path)
    downgrades = [
        (name, base[key], head[key])
        for key in base
        if key in head and head[key] < base[key]
        for name in [key[0]]
    ]
    for name, was, now in downgrades:
        print(f"{name}: {show(was)} on the base branch, {show(now)} here")
    if downgrades:
        print("Cargo.lock downgrades a package the base branch already had; rebase and re-pin.")
        return 1
    return 0


def show(version):
    release, is_release, pre = version
    text = ".".join(map(str, release))
    return text if is_release else f"{text}-{pre}"


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
