#!/usr/bin/env bash
# The whole sweep: every engine over every image, results as JSON under
# target/results. GHIDRA_SRC must point at a directory holding
# Ghidra/Processors for icicle; the Embench images come from
# benchmarks/embench/build.sh and the native binaries from native/build.sh.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGES=${IMAGES:-$HERE/../../target/embench}
OUT=${OUT:-$HERE/target/results}
REPEAT=${REPEAT:-5}
BIN=$HERE/target/release/hookbench
mkdir -p "$OUT"
uptime > "$OUT/load.txt"
# One process per (engine, instrumentation, image), each under a timeout, so
# a run that hangs costs one row rather than the sweep.
INSTRS="none block-ir block-cb insn-ir insn-cb edge-ir watch-ir watch-cb cmp-ir cmp-cb"
for engine in qcode-jit unicorn icicle; do
  : > "$OUT/$engine.log"
  for instr in $INSTRS; do
    for img in $(ls "$IMAGES"/*.elf | xargs -n1 basename | sed 's/\.elf$//'); do
      timeout "${TIMEOUT:-600}" "$BIN" --images "$IMAGES" --only "$img" --engine "$engine" --instr "$instr" \
        --repeat "$REPEAT" --json "$OUT/$engine-$instr-$img.json" 2>&1 | tail -n +2 >> "$OUT/$engine.log" \
        || echo "$engine $instr $img TIMEOUT/ERROR rc=$?" >> "$OUT/$engine.log"
    done
  done
done
(cd "$HERE/target/native" && for b in $(ls *.plain | sed 's/\.plain$//'); do
  for v in plain block edge cmp watch; do "./$b.$v" 30 "$b" "$v"; done
done) > "$OUT/native.txt" 2>&1
# The interpreter is two orders of magnitude slower than the JIT: a subset of
# images, and only the instrumentation that is IR.
for img in crc32 tarfind depthconv matmult-int; do
  for instr in none block-ir insn-ir edge-ir watch-ir cmp-ir; do
    "$BIN" --images "$IMAGES" --only "$img" --engine qcode-interp --instr "$instr" --repeat 1 \
      --json "$OUT/qcode-interp-$img-$instr.json" >> "$OUT/qcode-interp.log" 2>&1
  done
done
uptime >> "$OUT/load.txt"
echo done >> "$OUT/load.txt"
