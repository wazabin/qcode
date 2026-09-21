#!/usr/bin/env python3
"""Tabulate iai-callgrind's JSON output as a Markdown table.

    cargo bench -p wazabin-qcode-sleigh --bench lift -- --output-format=json > bench.json
    python3 .github/scripts/bench-table.py < bench.json

One record per benchmark, one per line. The benchmark functions become
columns and the case ids rows, both in the order they ran. A run against a
baseline gets a change column per function. The measure is instructions
retired, which callgrind counts exactly, so every non-zero change is a real
change in the work done; a change of a percent or more is marked.
"""

import json
import sys

MARK = 1.0


def parse(lines):
    """Yield `(function, id, instructions, change)` per benchmark."""
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        summary = record["profiles"][0]["summaries"]["total"]["summary"]
        ir = summary["Callgrind"]["Ir"]
        metrics = ir["metrics"]
        if "Both" in metrics:
            new, _ = metrics["Both"]
            change = float(ir["diffs"]["diff_pct"])
        else:
            new = metrics["Single"]
            change = None
        yield record["function_name"], record["id"], new["Int"], change


def cell(change):
    if change is None:
        return ""
    text = f"{change:+.2f}%"
    if change <= -MARK:
        return f"**{text}** 🟢"
    if change >= MARK:
        return f"**{text}** 🔴"
    return text


def main():
    results = {}
    groups, values = [], []
    for group, value, instructions, change in parse(sys.stdin):
        if group not in groups:
            groups.append(group)
        if value not in values:
            values.append(value)
        results[group, value] = (instructions, change)
    if not results:
        print("No results.")
        return

    compared = any(change is not None for _, change in results.values())
    header = ["instruction"]
    align = [":--"]
    for group in groups:
        header.append(group)
        align.append("--:")
        if compared:
            header.append("Δ")
            align.append("--:")
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(align) + " |")
    for value in values:
        row = [f"`{value}`"]
        for group in groups:
            instructions, change = results.get((group, value), (None, None))
            row.append(f"{instructions:,}" if instructions is not None else "")
            if compared:
                row.append(cell(change))
        print("| " + " | ".join(row) + " |")
    print()
    if compared:
        print(
            "Instructions retired per run under callgrind, and the change "
            "against the base. The count is exact, so every non-zero change is "
            f"a change in the work done; 🟢/🔴 mark those of {MARK:g}% or more."
        )
    else:
        print("Instructions retired per run under callgrind.")


if __name__ == "__main__":
    main()
