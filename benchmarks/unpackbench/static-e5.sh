#!/usr/bin/env bash
# E5 (claim C4): run the `consume` example over every unpack artifact under a
# run directory, writing e5.json next to each context.bin.
#
#   static-e5.sh runs/<label>                  # every (binary, qcode) dir below
#   static-e5.sh /tmp/unpack/selfdecrypt       # one artifact dir
#   static-e5.sh DIR --elf corpus/1.2/x/bin    # extra args go to `consume`
#
# The original ELF defaults to graph.json's program.path. A relative one is
# resolved from benchmarks/unpackbench/, where run-all.sh runs (CONTRACT.md),
# so this script runs `consume` from there. A directory whose meta.json says
# "unsupported" has no context.bin and is skipped.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$HERE/../..
if [ $# -lt 1 ]; then
  echo "usage: $0 RUN_DIR [consume args...]" >&2
  exit 2
fi
RUN=$(cd "$1" && pwd) || exit 2
shift
cargo build -q --release --manifest-path "$ROOT/Cargo.toml" -p wazabin-qcode-userland --example consume || exit 1
BIN=$ROOT/target/release/examples/consume
cd "$HERE" || exit 1
status=0
found=0
while IFS= read -r ctx; do
  dir=$(dirname "$ctx")
  found=$((found + 1))
  echo "== ${dir#"$RUN"/}"
  if ! "$BIN" "$dir" "$@"; then
    echo "$dir: consume failed" >&2
    status=1
  fi
done < <(find "$RUN" -name context.bin -print | sort)
if [ "$found" = 0 ]; then
  echo "no context.bin under $RUN" >&2
  exit 1
fi
exit $status
