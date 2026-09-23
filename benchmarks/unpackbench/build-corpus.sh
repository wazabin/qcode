#!/usr/bin/env bash
# Build the unpackbench corpus for families 1.1, 1.2, 1.3 and 1.5 and merge
# their entries into corpus/manifest.json (see CONTRACT.md and EVALUATION_PLAN
# §1, §2). Family 1.4 is built by a separate script; this one never touches
# corpus/1.4 nor 1.4's manifest entries.
#
# Idempotent: it rebuilds its four families from scratch on every run and each
# binary's ground truth is recomputed, so re-running yields the same corpus.
# One summary line per binary; a final tally per family.
#
# Tools (paths overridable by environment):
#   UPX      packer                         [~/dev/upx/build/upx, 5.2.1-devel]
#   TIGRESS_HOME  Tigress install           [~/Downloads/tigress/4.0.10]
#   EMBENCH  embench-iot checkout           [~/dev/embench-iot]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

UPX="${UPX:-$HOME/dev/upx/build/upx}"
export TIGRESS_HOME="${TIGRESS_HOME:-$HOME/Downloads/tigress/4.0.10}"
export PATH="$TIGRESS_HOME:$PATH"
TIGRESS="$TIGRESS_HOME/tigress"
EMBENCH="${EMBENCH:-$HOME/dev/embench-iot}"

CORPUS="$here/corpus"
SEEDS="$here/seeds"
FIX="$here/fixtures"
TOOLS="$here/tools"
BUILD="$here/build"          # scratch, safe to delete
MYFAMS="1.1,1.2,1.3,1.5"

CC="${CC:-gcc}"
CFLAGS_STATIC="-static -O2"
TENV="x86_64:Linux:Gcc:4.6"

# --- tool versions -----------------------------------------------------------
UPX_VER="$("$UPX" --version 2>/dev/null | head -1 || echo 'missing')"
GCC_VER="$($CC -dumpfullversion 2>/dev/null || echo '?')"
TIGRESS_VER="4.0.10"
BUSYBOX_NOTE="dropped: the runner passes argv[0]=<bin path>, but busybox's multi-call applet dispatch needs argv[0] basename 'busybox'; no applet resolves, so packed busybox seeds cannot be exercised through the harness"
UPX42_NOTE="absent: only upx 5.2.1-devel is installed here, so the 4.2 /proc/self/exe stub path was not built"

# --- reset only the families this script owns --------------------------------
for f in 1.1 1.2 1.3 1.5; do rm -rf "$CORPUS/$f"; done
rm -rf "$BUILD"; mkdir -p "$BUILD" "$CORPUS"

declare -A COUNT=( [1.1]=0 [1.2]=0 [1.3]=0 [1.5]=0 )

# write_entry writes corpus/<fam>/<name>/entry.json from the E* environment.
write_entry() {
  EOUT="$1" python3 - <<'PY'
import json, os
def env(k, d=None): return os.environ.get(k, d)
e = {
 "family": env("EFAM"), "name": env("ENAME"), "path": env("EPATH"),
 "argv": json.loads(env("EARGV", "null")), "stdin": None,
 "launch": env("ELAUNCH", "static"),
 "seed": env("ESEED") or None, "generator": env("EGEN") or None,
 "expect": {"exit": int(env("EEXIT")), "stdout_sha256": env("ESHA")},
 "truth": {"segments": env("ETSEG") == "1", "sites": env("ETSITES") == "1",
           "edges": json.loads(env("ETEDGES", "false"))},
}
if env("EREF"): e["reference"] = env("EREF")
if env("ENOTE"): e["tools_note"] = env("ENOTE")
json.dump(e, open(os.environ["EOUT"], "w"), indent=1)
open(os.environ["EOUT"], "a").write("\n")
PY
}

# run_native BIN OUTDIR [args...] -> sets EXIT and SHA, writes truth/stdout,exit
run_native() {
  local bin="$1" td="$2"; shift 2
  mkdir -p "$td"
  if "$bin" "$@" >"$td/stdout" 2>/dev/null; then EXIT=0; else EXIT=$?; fi
  SHA="$(sha256sum "$td/stdout" | cut -d' ' -f1)"
  printf '%s' "$EXIT" >"$td/exit"
}

summary() { # fam name exit sha
  printf '[corpus] %-4s %-22s exit=%-3s stdout=%s…\n' "$1" "$2" "$3" "${4:0:12}"
}

# =============================================================================
# Seeds (static ELFs), reused by families 1.5 (unmodified) and 1.1 (packed).
# =============================================================================
build_seed() { # name -> $BUILD/<name>.elf
  local name="$1"
  case "$name" in
    hw)    $CC $CFLAGS_STATIC -o "$BUILD/hw.elf"   "$SEEDS/hw.c" ;;
    tiny)  $CC $CFLAGS_STATIC -o "$BUILD/tiny.elf" "$SEEDS/tiny.c" ;;
    nbody) $CC $CFLAGS_STATIC -o "$BUILD/nbody.elf" "$SEEDS/nbody.c" ;;
    crc32) $CC $CFLAGS_STATIC -DGLOBAL_SCALE_FACTOR=1 \
             -I"$EMBENCH/support" -I"$EMBENCH/src/crc32" \
             -o "$BUILD/crc32.elf" "$SEEDS/embmain.c" \
             "$EMBENCH/src/crc32/crc_32.c" "$EMBENCH/support/beebsc.c" ;;
    sha256) $CC $CFLAGS_STATIC -DGLOBAL_SCALE_FACTOR=1 \
             -I"$EMBENCH/support" -I"$EMBENCH/src/nettle-sha256" \
             -o "$BUILD/sha256.elf" "$SEEDS/embmain_sha.c" \
             "$EMBENCH/src/nettle-sha256/nettle-sha256.c" "$EMBENCH/support/beebsc.c" ;;
  esac
}

SEED_NAMES=(hw tiny nbody crc32 sha256)
for s in "${SEED_NAMES[@]}"; do build_seed "$s"; done

# --- 1.5 negative controls: the unmodified seeds -----------------------------
for s in "${SEED_NAMES[@]}"; do
  d="$CORPUS/1.5/$s"; mkdir -p "$d/source"
  cp "$BUILD/$s.elf" "$d/bin"; chmod +x "$d/bin"
  run_native "$d/bin" "$d/truth"
  # source record
  cp "$SEEDS/$s.c" "$d/source/" 2>/dev/null || true
  cat >"$d/source/build.sh" <<SB
#!/usr/bin/env bash
# unmodified static seed \`$s\` (negative control, and the seed for 1.1/1.2)
$( case "$s" in
   crc32)  echo "gcc -static -O2 -DGLOBAL_SCALE_FACTOR=1 -I<embench>/support -I<embench>/src/crc32 embmain.c crc_32.c beebsc.c -o bin" ;;
   sha256) echo "gcc -static -O2 -DGLOBAL_SCALE_FACTOR=1 -I<embench>/support -I<embench>/src/nettle-sha256 embmain_sha.c nettle-sha256.c beebsc.c -o bin" ;;
   *)      echo "gcc -static -O2 -o bin $s.c" ;;
   esac )
SB
  EFAM=1.5 ENAME="$s" EPATH="corpus/1.5/$s/bin" EARGV=null ELAUNCH=static \
    ESEED="$s" EGEN="" EEXIT="$EXIT" ESHA="$SHA" \
    ETSEG=0 ETSITES=0 ETEDGES='"reference"' write_entry "$d/entry.json"
  COUNT[1.5]=$((COUNT[1.5]+1)); summary 1.5 "$s" "$EXIT" "$SHA"
done

# =============================================================================
# 1.1 compressed executables: UPX over the seeds. segments.json from the seed.
# =============================================================================
pack_variant() { # seed flag-name "upx flags..."
  local seed="$1" tag="$2"; shift 2
  local name="$seed-upx-$tag" d="$CORPUS/1.1/$seed-upx-$tag"
  mkdir -p "$d/source" "$d/truth"
  cp "$BUILD/$seed.elf" "$d/bin"; chmod +x "$d/bin"
  "$UPX" -q "$@" "$d/bin" >/dev/null 2>&1
  run_native "$d/bin" "$d/truth"
  python3 "$TOOLS/mkseg.py" "$BUILD/$seed.elf" >"$d/truth/segments.json"
  cat >"$d/source/build.sh" <<SB
#!/usr/bin/env bash
# 1.1: the static seed \`$seed\` (see corpus/1.5/$seed) packed with:
#   upx -q $* <seed> -> bin
# segments.json is the seed's PT_LOAD image, which the stub reconstructs.
SB
  EFAM=1.1 ENAME="$name" EPATH="corpus/1.1/$name/bin" EARGV=null ELAUNCH=static \
    ESEED="$seed" EGEN="upx $*" EEXIT="$EXIT" ESHA="$SHA" \
    EREF="corpus/1.5/$seed/bin" \
    ETSEG=1 ETSITES=0 ETEDGES='"reference"' write_entry "$d/entry.json"
  COUNT[1.1]=$((COUNT[1.1]+1)); summary 1.1 "$name" "$EXIT" "$SHA"
}

# hw exercises every compressor; the others get default and --best.
pack_variant hw default
pack_variant hw best        --best
pack_variant hw lzma        --lzma
pack_variant hw nrv2b       --nrv2b
pack_variant hw ultrabrute  --ultra-brute
pack_variant hw nofilter    --no-filter
for s in tiny nbody crc32 sha256; do
  pack_variant "$s" default
  pack_variant "$s" best --best
done

# =============================================================================
# 1.2 Tigress: runtime code generation with known provenance, on two seeds.
# Jit/JitDynamic dump generated code; here we record the emitting function's
# pc in sites.json and mark segments false (dumping the emitted buffer needs an
# instrumented rebuild; deferred per EVALUATION_PLAN §2's fallback).
# =============================================================================
tigress_variant() { # seedfile func local  transform-tag  tigress-args...
  local seedfile="$1" func="$2" local_="$3" tag="$4"; shift 4
  local seedbase; seedbase="$(basename "$seedfile" .c)"
  local name="$seedbase-$tag"
  local d="$CORPUS/1.2/$name"
  local gen="$BUILD/$name.c"
  mkdir -p "$d/source" "$d/truth"
  if ! ( cd "$BUILD" && "$TIGRESS" --Environment="$TENV" "$@" --out="$gen" "$seedfile" ) \
        >"$BUILD/$name.tigress.log" 2>&1; then
    echo "[corpus] 1.2  $name DROPPED (tigress failed; see build/$name.tigress.log)"
    rm -rf "$d"; return 0
  fi
  if ! $CC $CFLAGS_STATIC -o "$d/bin" "$gen" >>"$BUILD/$name.tigress.log" 2>&1; then
    echo "[corpus] 1.2  $name DROPPED (gcc -static failed; see build/$name.tigress.log)"
    rm -rf "$d"; return 0
  fi
  chmod +x "$d/bin"
  run_native "$d/bin" "$d/truth"
  local sites=0 tsites=0
  if [[ "$tag" == jit || "$tag" == jitdyn ]]; then
    # the emitting function jit_generate_code writes every generated byte
    local pc
    pc="$(nm "$d/bin" 2>/dev/null | awk '$3=="jit_generate_code"{print "0x"$1}')"
    if [[ -n "$pc" ]]; then
      printf '[\n {"pc": "%s", "generation": 1}\n]\n' "$pc" >"$d/truth/sites.json"
      tsites=1
    fi
  fi
  cp "$gen" "$d/source/transformed.c"
  cp "$seedfile" "$d/source/"
  cat >"$d/source/build.sh" <<SB
#!/usr/bin/env bash
# 1.2: Tigress $tag on \`$func\` in $(basename "$seedfile"), then a static build:
#   tigress --Environment=$TENV $* --out=transformed.c $(basename "$seedfile")
#   gcc -static -O2 transformed.c -o bin
SB
  local tseg=0
  EFAM=1.2 ENAME="$name" EPATH="corpus/1.2/$name/bin" EARGV=null ELAUNCH=static \
    ESEED="$seedbase" EGEN="tigress $tag" EEXIT="$EXIT" ESHA="$SHA" \
    ETSEG="$tseg" ETSITES="$tsites" ETEDGES=false write_entry "$d/entry.json"
  COUNT[1.2]=$((COUNT[1.2]+1)); summary 1.2 "$name" "$EXIT" "$SHA"
}

for pair in "compute.c:compute:s" "poly.c:evalpoly:acc"; do
  IFS=: read -r file func loc <<<"$pair"
  sf="$SEEDS/tigress/$file"
  tigress_variant "$sf" "$func" "$loc" jit \
    --Transform=Jit --Functions="$func"
  tigress_variant "$sf" "$func" "$loc" jitdyn \
    --Transform=JitDynamic --Functions="$func"
  tigress_variant "$sf" "$func" "$loc" encdata \
    --Transform=InitOpaque --Functions=main \
    --Transform=EncodeData --Functions="$func" --LocalVariables="$func:$loc" \
    --Transform=Virtualize --Functions="$func"
  tigress_variant "$sf" "$func" "$loc" split \
    --Transform=InitEntropy --Functions=main \
    --Transform=InitOpaque --Functions=main --InitOpaqueStructs=list,array \
    --Transform=Split --Functions="$func" \
    --Transform=Flatten --Functions="$func" \
    --Transform=AddOpaque --Functions="$func" --AddOpaqueKinds=call
done

# =============================================================================
# 1.3 self-modifying fixtures (selfdecrypt style, with ground truth).
#   overwrite  overwrites code it already ran (two generations, same address)
#   stores16   decryptor using 16-byte SSE (movdqu) stores
#   outside    stage mapped outside the emulator's mmap window (expect a warning)
# =============================================================================
xorpad() { # in out key size(0=no pad)
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import sys
src, dst, key, sz = sys.argv[1], sys.argv[2], int(sys.argv[3], 0), int(sys.argv[4])
d = open(src, "rb").read()
if sz: d = d.ljust(sz, b"\x90")
open(dst, "wb").write(bytes(b ^ key for b in d))
PY
}
seg_json() { # start kind sha  -> one segment object (helper for fixtures)
  printf '{"start": "%s", "end": "%s", "sha256": "%s", "kind": "%s"}' "$1" "$2" "$3" "$4"
}

fixture_overwrite() {
  local name=overwrite
  local d="$CORPUS/1.3/$name" w="$BUILD/$name"
  local BUFSZ=64 K1=0x5A K2=0xA7
  mkdir -p "$d/source" "$d/truth" "$w"
  $CC -c -nostdlib -ffreestanding "$FIX/overwrite_p1.S" -o "$w/p1.o"
  objcopy -O binary -j .text "$w/p1.o" "$w/p1.bin"
  $CC -c -nostdlib -ffreestanding "$FIX/overwrite_p2.S" -o "$w/p2.o"
  objcopy -O binary -j .text "$w/p2.o" "$w/p2.bin"
  # padded plaintext blobs are the code ground truth
  python3 -c "open('$w/p1.pad','wb').write(open('$w/p1.bin','rb').read().ljust($BUFSZ,b'\x90'))"
  python3 -c "open('$w/p2.pad','wb').write(open('$w/p2.bin','rb').read().ljust($BUFSZ,b'\x90'))"
  xorpad "$w/p1.bin" "$w/overwrite_p1.enc" "$K1" "$BUFSZ"
  xorpad "$w/p2.bin" "$w/overwrite_p2.enc" "$K2" "$BUFSZ"
  ( cd "$w" && $CC -c -nostdlib -ffreestanding -fcf-protection=none \
       -Wa,-mx86-used-note=no -DBUFSZ=$BUFSZ -DKEY1=$K1 -DKEY2=$K2 \
       "$FIX/overwrite_loader.S" -o loader.o )
  $CC -static -nostdlib -no-pie -fno-pie -Wl,--build-id=none "$w/loader.o" -o "$d/bin" 2>/dev/null
  chmod +x "$d/bin"
  run_native "$d/bin" "$d/truth"
  # genbuf is the .rwx PT_LOAD base; both generations live there
  local base
  base="$(readelf -lW "$d/bin" | awk '/RWE/{print $3}')"
  local end; end="$(printf '0x%x' $(( base + BUFSZ )))"
  local h1 h2; h1="$(sha256sum "$w/p1.pad" | cut -d' ' -f1)"; h2="$(sha256sum "$w/p2.pad" | cut -d' ' -f1)"
  { echo '['; seg_json "$base" "$end" "$h1" code; echo ','; seg_json "$base" "$end" "$h2" code; echo; echo ']'; } >"$d/truth/segments.json"
  # writer sites: the two `mov %al,(%rdi)` stores, gen 1 then gen 2
  local pcs; pcs=($(objdump -d --no-show-raw-insn "$d/bin" | awk '/mov +%al,\(%rdi\)/{print $1}' | tr -d ':'))
  { echo '['; printf ' {"pc": "0x%s", "generation": 1},\n' "${pcs[0]}"
    printf ' {"pc": "0x%s", "generation": 2}\n' "${pcs[1]}"; echo ']'; } >"$d/truth/sites.json"
  cp "$FIX"/overwrite_*.S "$d/source/"
  cat >"$d/source/build.sh" <<SB
#!/usr/bin/env bash
# 1.3 overwrite: stage 0 decrypts P1 (gen1) into the RWX .rwx buffer, calls it,
# then overwrites the SAME buffer with P2 (gen2) and runs that. Two generations
# at one address -> versioning. Keys K1=$K1 K2=$K2, buffer $BUFSZ bytes.
SB
  EFAM=1.3 ENAME="$name" EPATH="corpus/1.3/$name/bin" EARGV=null ELAUNCH=static \
    ESEED="" EGEN="hand-written self-modifying fixture" EEXIT="$EXIT" ESHA="$SHA" \
    ETSEG=1 ETSITES=1 ETEDGES=false write_entry "$d/entry.json"
  COUNT[1.3]=$((COUNT[1.3]+1)); summary 1.3 "$name" "$EXIT" "$SHA"
}

fixture_mmap() { # name loader payload key mmapbase-hex movdqu?  extra-D...
  local name="$1" loader="$2" payload="$3" key="$4" mbase="$5" store_re="$6"; shift 6
  local d="$CORPUS/1.3/$name" w="$BUILD/$name"
  mkdir -p "$d/source" "$d/truth" "$w"
  $CC -c -nostdlib -ffreestanding "$FIX/$payload" -o "$w/pl.o"
  objcopy -O binary -j .text "$w/pl.o" "$w/pl.bin"
  local sz pad chunks
  sz="$(wc -c <"$w/pl.bin")"
  pad=$(( (sz + 15) / 16 * 16 ))
  chunks=$(( pad / 16 ))
  local blobsz=$sz Dsz="-DLEN=$sz"
  local encsz=0
  if [[ "$name" == stores16 ]]; then blobsz=$pad; encsz=$pad; Dsz=""; fi
  # padded plaintext is the ground-truth stage
  python3 -c "open('$w/pl.pad','wb').write(open('$w/pl.bin','rb').read().ljust($blobsz,b'\x90'))"
  xorpad "$w/pl.bin" "$w/${payload%_payload.S}_payload.enc" "$key" "$encsz"
  # 8-byte splat of the key for the SSE path
  local k8; k8=$(printf '0x%02x%02x%02x%02x%02x%02x%02x%02x' $((key)) $((key)) $((key)) $((key)) $((key)) $((key)) $((key)) $((key)))
  ( cd "$w" && $CC -c -nostdlib -ffreestanding "$@" -DKEY=$key -DKEY8=$k8 \
       -DCHUNKS=$chunks $Dsz "$FIX/$loader" -o loader.o )
  $CC -static -nostdlib -no-pie -fno-pie -Wl,--build-id=none "$w/loader.o" -o "$d/bin" 2>/dev/null
  chmod +x "$d/bin"
  run_native "$d/bin" "$d/truth"
  local end; end="$(printf '0x%x' $(( mbase + blobsz )))"
  local h; h="$(sha256sum "$w/pl.pad" | cut -d' ' -f1)"
  { echo '['; seg_json "$(printf '0x%x' "$mbase")" "$end" "$h" code; echo; echo ']'; } >"$d/truth/segments.json"
  local pcs; pcs=($(objdump -d --no-show-raw-insn "$d/bin" | awk "/$store_re/{print \$1}" | tr -d ':'))
  { echo '['; printf ' {"pc": "0x%s", "generation": 1}\n' "${pcs[0]}"; echo ']'; } >"$d/truth/sites.json"
  cp "$FIX/$loader" "$FIX/$payload" "$d/source/"
  cat >"$d/source/build.sh" <<SB
#!/usr/bin/env bash
# 1.3 $name: stage 0 mmaps an RWX page, decrypts the embedded stage into it
# (key $key), then jumps in. See $loader / $payload.
SB
  EFAM=1.3 ENAME="$name" EPATH="corpus/1.3/$name/bin" EARGV=null ELAUNCH=static \
    ESEED="" EGEN="hand-written self-modifying fixture" EEXIT="$EXIT" ESHA="$SHA" \
    ETSEG=1 ETSITES=1 ETEDGES=false write_entry "$d/entry.json"
  COUNT[1.3]=$((COUNT[1.3]+1)); summary 1.3 "$name" "$EXIT" "$SHA"
}

fixture_overwrite
# stores16: anonymous mmap -> emulator serves it at MMAP_BASE 0x7f0000000000
fixture_mmap stores16 stores16_loader.S stores16_payload.S 0xA7 0x7f0000000000 'movdqu +%xmm0,\(%rdi\)'
# outside: MAP_FIXED at 0x550000000000, outside the emulator's mmap window
fixture_mmap outside  outside_loader.S  outside_payload.S  0x3C 0x550000000000 'mov +%al,\(%rdi\)' -DFIXED=0x550000000000

# =============================================================================
# Manifest: assemble this script's entries and merge, preserving family 1.4.
# =============================================================================
built="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
MINE="$BUILD/mine.json"
BUILT="$built" UPX_VER="$UPX_VER" GCC_VER="$GCC_VER" TIGRESS_VER="$TIGRESS_VER" \
BUSYBOX_NOTE="$BUSYBOX_NOTE" UPX42_NOTE="$UPX42_NOTE" CORPUS="$CORPUS" MINE="$MINE" \
python3 - <<'PY'
import glob, json, os
binaries = []
for fam in ("1.1", "1.2", "1.3", "1.5"):
    for ej in sorted(glob.glob(os.path.join(os.environ["CORPUS"], fam, "*", "entry.json"))):
        binaries.append(json.load(open(ej)))
mine = {
 "built": os.environ["BUILT"],
 "tools": {"upx": os.environ["UPX_VER"], "gcc": os.environ["GCC_VER"],
           "tigress": os.environ["TIGRESS_VER"], "busybox": os.environ["BUSYBOX_NOTE"],
           "upx_4.2": os.environ["UPX42_NOTE"]},
 "binaries": binaries,
}
json.dump(mine, open(os.environ["MINE"], "w"), indent=1)
PY

python3 "$TOOLS/merge_manifest.py" "$CORPUS/manifest.json" "$MINE" "$MYFAMS"

echo
echo "[corpus] families: 1.1=${COUNT[1.1]} 1.2=${COUNT[1.2]} 1.3=${COUNT[1.3]} 1.5=${COUNT[1.5]}"
echo "[corpus] tools: upx='$UPX_VER' gcc=$GCC_VER tigress=$TIGRESS_VER"
echo "[corpus] corpus bytes: $(du -sb "$CORPUS" | cut -f1)"
