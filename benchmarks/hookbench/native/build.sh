#!/usr/bin/env bash
# Builds every Embench benchmark natively in five variants, with the flags of
# the emulated images so the code shape matches:
#   plain  - no instrumentation
#   block  - gcc -fsanitize-coverage=trace-pc, a call per basic block
#   edge   - the same, with an AFL-style edge map in the callback
#   cmp    - gcc -fsanitize-coverage=trace-cmp, a call per comparison
#   watch  - plain code, four hardware watchpoints on .data through perf
set -euo pipefail
EMBENCH=${EMBENCH:?set EMBENCH to an embench-iot checkout}
OUT=${OUT:-target/native}
HERE=$(cd "$(dirname "$0")" && pwd)
QCODE_EMBENCH="$HERE/../../embench"
CFLAGS=(-O2 -std=gnu99 -DHAVE_CONFIG_H -DWARMUP_HEAT=1 -DGLOBAL_SCALE_FACTOR=1
  -mno-sse -mno-sse2 -mno-mmx -fno-tree-vectorize -fno-stack-protector -fno-builtin
  -mno-red-zone -fno-asynchronous-unwind-tables -fno-pic -no-pie
  -Wno-implicit-function-declaration -I"$QCODE_EMBENCH" -I"$EMBENCH/support")
SKIP=" wikisort slre "
mkdir -p "$OUT"
for dir in "$EMBENCH"/src/*/; do
  name=$(basename "$dir")
  [[ $SKIP == *" $name "* ]] && continue
  for variant in plain block edge cmp watch; do
    case $variant in
      plain) extra=(); rt=rt_none.c ;;
      block) extra=(-fsanitize-coverage=trace-pc); rt=rt_block.c ;;
      edge)  extra=(-fsanitize-coverage=trace-pc); rt=rt_edge.c ;;
      cmp)   extra=(-fsanitize-coverage=trace-cmp); rt=rt_cmp.c ;;
      watch) extra=(); rt=rt_watch.c ;;
    esac
    # Only the benchmark's own sources are instrumented; the harness and
    # runtime are not, which is what the emulated hooks see too (the images
    # hold nothing else).
    objs=()
    for src in "$dir"/*.c "$EMBENCH/support/beebsc.c"; do
      obj="$OUT/$name.$variant.$(basename "$src" .c).o"
      gcc "${CFLAGS[@]}" "${extra[@]}" -I"$dir" -c "$src" -o "$obj"
      objs+=("$obj")
    done
    gcc "${CFLAGS[@]}" -I"$dir" -c "$HERE/main.c" -o "$OUT/$name.$variant.main.o"
    sym=$(awk -v n="$name" '$1==n {print $2}' "$HERE/watch-symbols.txt")
    gcc "${CFLAGS[@]}" -DWATCH_SYMBOL="${sym:-__bss_start}" -c "$HERE/$rt" -o "$OUT/$name.$variant.rt.o"
    gcc -no-pie -o "$OUT/$name.$variant" "${objs[@]}" "$OUT/$name.$variant.main.o" "$OUT/$name.$variant.rt.o" \
      || echo "FAILED $name $variant"
  done
done
rm -f "$OUT"/*.o
echo "built -> $OUT"
