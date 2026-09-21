#!/usr/bin/env bash
# The QCode JIT rows on a machine that will not go quiet: each run timed by
# its thread's CPU clock (HOOKBENCH_CPUTIME=1, rows carry "clock": "cpu"),
# with retired user instructions counted by perf beside it in
# instructions.txt. The report and the page show such a run in the progress
# table only. See README.md.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGES=${IMAGES:-$HERE/../../target/embench}
OUT=${OUT:?set OUT to the run directory}
REPEAT=${REPEAT:-5}
ENGINE=${ENGINE:-qcode-jit}
BIN=$HERE/target/release/hookbench
INSTRS=${INSTRS:-"none block-ir block-ram block-cb insn-ir insn-ram insn-cb edge-ir edge-ram watch-ir watch-cb cmp-ir cmp-ram cmp-cb"}
mkdir -p "$OUT"
uptime > "$OUT/load.txt"
: > "$OUT/$ENGINE.log"
: > "$OUT/instructions.txt"
for instr in $INSTRS; do
  for img in $(ls "$IMAGES"/*.elf | xargs -n1 basename | sed 's/\.elf$//'); do
    HOOKBENCH_CPUTIME=1 timeout "${TIMEOUT:-600}" "$BIN" --images "$IMAGES" --only "$img" --engine "$ENGINE" \
      --instr "$instr" --repeat "$REPEAT" --json "$OUT/$ENGINE-$instr-$img.json" 2>&1 \
      | tail -n +2 >> "$OUT/$ENGINE.log" \
      || echo "$ENGINE $instr $img TIMEOUT/ERROR rc=$?" >> "$OUT/$ENGINE.log"
    count=$(perf stat -e instructions:u -x, timeout "${TIMEOUT:-600}" "$BIN" --images "$IMAGES" --only "$img" \
      --engine "$ENGINE" --instr "$instr" --repeat 1 2>&1 >/dev/null | awk -F, '/instructions:u/ {print $1}')
    echo "$ENGINE $instr $img ${count:-?}" >> "$OUT/instructions.txt"
  done
done
uptime >> "$OUT/load.txt"
echo done >> "$OUT/load.txt"
