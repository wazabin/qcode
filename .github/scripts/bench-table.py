#!/usr/bin/env python3
"""Tabulate Criterion's text output as a Markdown table.

    cargo bench -p wazabin-qcode-sleigh --bench lift -- --noplot | tee bench.txt
    python3 .github/scripts/bench-table.py < bench.txt

Every benchmark is `group/value`; the values become rows and the groups
columns, both in the order Criterion printed them. A run against a baseline
gets a change column per group with Criterion's own verdict: a change is
marked only when Criterion reports it as an improvement or a regression, so
the table applies the same significance test and noise threshold as the run.
"""

import re
import sys

# `decode/nop              time:   [203.50 ns 205.25 ns 207.01 ns]`; the id
# takes the line to itself when it is too long for the column.
ID_LINE = re.compile(r"^(?P<id>[^\s/]+/\S+)\s*(?:time:\s*\[(?P<time>[^\]]*)\])?$")
TIME_LINE = re.compile(r"^\s+time:\s*\[(?P<time>[^\]]*)\]")
# `change: [−45.623% −44.868% −44.130%] (p = 0.00 < 0.05)`; Criterion writes a
# Unicode minus.
CHANGE_LINE = re.compile(r"^\s+change:\s*\[[^\s\]]+\s+(?P<change>[^\s\]]+)\s+[^\s\]]+\]")
VERDICTS = {
    "Performance has improved.": "improved",
    "Performance has regressed.": "regressed",
    "Change within noise threshold.": "noise",
    "No change in performance detected.": "noise",
}


def parse(lines):
    """Yield `(id, time, change, verdict)` per benchmark, in output order."""
    current = None
    for line in lines:
        line = line.rstrip("\n")
        if match := ID_LINE.match(line):
            if current:
                yield current
            current = [match["id"], match["time"], None, None]
            continue
        if not current:
            continue
        if match := TIME_LINE.match(line):
            current[1] = match["time"]
        elif match := CHANGE_LINE.match(line):
            current[2] = match["change"].replace("−", "-")
        elif (verdict := VERDICTS.get(line.strip())) is not None:
            current[3] = verdict
    if current:
        yield current


def estimate(time):
    """The middle of `[lower point upper]`, with its unit."""
    parts = time.split()
    return " ".join(parts[2:4]) if len(parts) == 6 else time


def cell(change, verdict):
    if verdict == "improved":
        return f"**{change}** 🟢"
    if verdict == "regressed":
        return f"**{change}** 🔴"
    return change or ""


def main():
    results = {}
    groups, values = [], []
    for id, time, change, verdict in parse(sys.stdin):
        group, value = id.split("/", 1)
        if group not in groups:
            groups.append(group)
        if value not in values:
            values.append(value)
        results[id] = (time, change, verdict)
    if not results:
        print("No results.")
        return

    compared = any(change for _, change, _ in results.values())
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
            time, change, verdict = results.get(f"{group}/{value}", (None, None, None))
            row.append(estimate(time) if time else "")
            if compared:
                row.append(cell(change, verdict))
        print("| " + " | ".join(row) + " |")
    if compared:
        print()
        print(
            "Criterion's point estimate per iteration and its change against the "
            "base; 🟢/🔴 mark the changes Criterion reports as an improvement or "
            "a regression, the others are within its noise threshold."
        )
    else:
        print()
        print("Criterion's point estimate per iteration.")


if __name__ == "__main__":
    main()
