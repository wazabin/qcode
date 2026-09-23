#!/usr/bin/env python3
"""Merge this builder's families into corpus/manifest.json without disturbing
another builder's (family 1.4 is produced separately).

    merge_manifest.py <manifest.json> <mine.json> <fam1,fam2,..>

<mine.json> = {"tools": {..}, "binaries": [..]} for the families this builder
owns. Existing entries in those families are replaced; every other family is
kept. `tools` is merged (mine wins). `built` is set to mine's timestamp.
"""
import json, sys
from pathlib import Path

man_path, mine_path, fams = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
mine = json.loads(Path(mine_path).read_text())
existing = {}
if Path(man_path).exists():
    try:
        existing = json.loads(Path(man_path).read_text())
    except json.JSONDecodeError:
        existing = {}

kept = [b for b in existing.get("binaries", []) if b.get("family") not in fams]
merged = dict(existing)
merged["built"] = mine.get("built", existing.get("built"))
tools = dict(existing.get("tools", {}))
tools.update(mine.get("tools", {}))
merged["tools"] = tools
binaries = kept + mine.get("binaries", [])
binaries.sort(key=lambda b: (b["family"], b["name"]))
merged["binaries"] = binaries
Path(man_path).write_text(json.dumps(merged, indent=1) + "\n")
print(f"[merge] {len(mine.get('binaries', []))} mine + {len(kept)} kept "
      f"= {len(binaries)} binaries")
