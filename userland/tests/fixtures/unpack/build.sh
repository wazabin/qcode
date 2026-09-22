#!/usr/bin/env bash
# Reproducible build of the `selfdecrypt` fixture: a static, libc-less,
# two-generation self-decrypting x86-64 Linux ELF.
#
#   stage 2  (payload)         -> stage2.enc  (XOR key KEY2)
#   stage 1  (mmap + decrypt)  -> stage1.enc  (XOR key KEY1), embeds stage2.enc
#   stage 0  (_start)          -> selfdecrypt, embeds stage1.enc in a RWX section
#
# Fixed XOR keys and no timestamps/build-id, so the committed binary is stable.
set -euo pipefail

cd "$(dirname "$0")"

# --- fixed parameters (change these and the committed binary changes) --------
KEY1=0x5A          # stage 0 -> stage 1
KEY2=0xA7          # stage 1 -> stage 2

CC=${CC:-gcc}
OBJCOPY=${OBJCOPY:-objcopy}
PY=${PY:-python3}

xor() {  # xor <in> <out> <key>
    "$PY" - "$1" "$2" "$3" <<'EOF'
import sys
src, dst, key = sys.argv[1], sys.argv[2], int(sys.argv[3], 0)
data = open(src, "rb").read()
open(dst, "wb").write(bytes(b ^ key for b in data))
EOF
}

# --- stage 2: assemble, extract raw bytes, encrypt ---------------------------
$CC -c -nostdlib -ffreestanding stage2.S -o stage2.o
$OBJCOPY -O binary -j .text stage2.o stage2.bin
xor stage2.bin stage2.enc "$KEY2"

# --- stage 1: assemble (embeds stage2.enc), extract, encrypt -----------------
$CC -c -nostdlib -ffreestanding -DKEY2="$KEY2" stage1.S -o stage1.o
$OBJCOPY -O binary -j .text stage1.o stage1.bin
xor stage1.bin stage1.enc "$KEY1"

# --- stage 0 / final ELF: assemble (embeds stage1.enc) and link --------------
# -fcf-protection=none + -mx86-used-note=no keep GCC from emitting a
# .note.gnu.property, so the final ELF is just the PT_LOAD segments.
$CC -c -nostdlib -ffreestanding -fcf-protection=none -Wa,-mx86-used-note=no \
    -DKEY1="$KEY1" stage0.S -o stage0.o
$CC -static -nostdlib -no-pie -fno-pie -fno-stack-protector \
    -Wl,--build-id=none \
    stage0.o -o selfdecrypt

# Strip everything but keep the ELF entry (_start); shrinks and de-noises it.
strip -s selfdecrypt

# --- report ------------------------------------------------------------------
echo "stage2.bin : $(wc -c < stage2.bin) bytes  (key $KEY2)"
echo "stage1.bin : $(wc -c < stage1.bin) bytes  (key $KEY1)"
echo "selfdecrypt: $(wc -c < selfdecrypt) bytes"

# Clean intermediates; keep only sources, the script, and the final binary.
rm -f stage0.o stage1.o stage2.o \
      stage1.bin stage2.bin stage1.enc stage2.enc
