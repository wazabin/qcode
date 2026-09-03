#!/usr/bin/env bash
# Builds Embench-IoT benchmarks as freestanding static x86-64 images for the VM.
#
# Each benchmark links to a fixed text address with no libc and no startup code,
# entered directly at `main`. The harness supplies the stack and a sentinel
# return address, so `main`'s `ret` is what ends the run and its return value —
# 0 when the benchmark verified — is read straight out of RAX.
#
# SSE is off. The lifter does not yet cover 128-bit storage, and GCC will
# autovectorise these loops given the chance.
set -euo pipefail

EMBENCH=${EMBENCH:?set EMBENCH to an embench-iot checkout}
OUT=${OUT:-target/embench}
HERE=$(cd "$(dirname "$0")" && pwd)

# Embench's own defaults (sconstruct.py): warmup_heat=1, gsf=1. Warming is
# pointless on an emulator with no cache, but it is left at the upstream value
# so the work done matches a stock Embench run.
CFLAGS=(
  -O2 -std=gnu99
  -DHAVE_CONFIG_H -DWARMUP_HEAT=1 -DGLOBAL_SCALE_FACTOR=1
  -mno-sse -mno-sse2 -mno-mmx -fno-tree-vectorize
  -ffreestanding -fno-stack-protector -fno-builtin
  -mno-red-zone -fno-asynchronous-unwind-tables -fno-pic -no-pie
  -Wno-implicit-function-declaration
  -I"$HERE" -I"$EMBENCH/support"
)
LDFLAGS=(-nostdlib -nostartfiles -static -no-pie -Wl,-Ttext=0x400000 -Wl,-e,main -Wl,--build-id=none)

# Not built:
#   wikisort - returns a float in an SSE register, which -mno-sse forbids
#   slre     - wants glibc's locale-internal ctype tables (__ctype_b_loc)
SKIP=" wikisort slre "

mkdir -p "$OUT"
built=0 failed=0
for dir in "$EMBENCH"/src/*/; do
  name=$(basename "$dir")
  [[ $SKIP == *" $name "* ]] && continue
  if gcc "${CFLAGS[@]}" -I"$dir" "${LDFLAGS[@]}" \
        "$dir"/*.c "$EMBENCH/support/main.c" "$EMBENCH/support/beebsc.c" \
        "$HERE/boardsupport.c" -o "$OUT/$name.elf" 2>"$OUT/$name.log"; then
    built=$((built + 1))
  else
    failed=$((failed + 1))
    echo "FAILED $name: $(head -1 "$OUT/$name.log")"
  fi
done
echo "built $built, failed $failed -> $OUT"
