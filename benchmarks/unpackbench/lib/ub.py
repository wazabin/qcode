"""Helpers shared by the unpackbench scripts (sweep, truth, report).

Everything here follows CONTRACT.md: addresses are "0x.." strings, run
directories are runs/<label>/<family>/<name>/<engine>/.
"""

import hashlib
import json
import os
import statistics
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent  # benchmarks/unpackbench
REPO = BENCH.parent.parent

# The configurations the harness knows, in the order the report reads them.
# `interp` is `edges` under the interpreter (E6); the rest run the JIT.
CONFIGS = {
    "nohooks": {"jit": True, "hooks": False, "edges": False},
    "hooks": {"jit": True, "hooks": True, "edges": False},
    "edges": {"jit": True, "hooks": True, "edges": True},
    "interp": {"jit": False, "hooks": True, "edges": True},
}
# Which configuration's graph a run directory keeps, best first.
PRIMARY_ORDER = ["edges", "hooks", "interp", "nohooks"]


def hexint(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return int(str(value), 0)


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=1, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def resolve(path, root):
    """A manifest path: absolute as is, else relative to the bench root that
    holds the corpus directory."""
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else (Path(root) / p)


def entry_dir(label_dir, family, name, engine):
    return Path(label_dir) / family / name / engine


def ref_family_name(reference):
    """`corpus/1.5/hw/bin` -> ("1.5", "hw")."""
    if not reference:
        return None
    parts = Path(reference).parts
    if len(parts) < 3:
        return None
    return parts[-3], parts[-2]


def count_source_lines(path):
    """Non-blank lines that are not only a comment (// # -- ;)."""
    n = 0
    try:
        with open(path, errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith(("//", "--", ";", '"""', "*", "/*")):
                    continue
                if s.startswith("#") and not s.startswith(("#[", "#!")):
                    continue
                n += 1
    except FileNotFoundError:
        return None
    return n
