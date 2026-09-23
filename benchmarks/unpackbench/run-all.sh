#!/usr/bin/env bash
# Run engines over the unpackbench corpus into runs/<label>/, then compare
# against ground truth and refresh the report. See CONTRACT.md.
#
#   ENGINES=qcode FAMILIES=1.1,1.5 REPS=5 LABEL=name \
#   CONFIGS=nohooks,hooks,edges ./run-all.sh
#
# Environment (defaults in brackets):
#   ENGINES   comma list of engines/<engine>/ drivers            [qcode]
#   FAMILIES  comma list of corpus families                      [all]
#   NAMES     comma list of binary names, to run a subset        [all]
#   CONFIGS   nohooks,hooks,edges,interp (interp = edges, no JIT) [nohooks,hooks,edges]
#   REPS      timed runs per config, after one discarded warm-up [5]
#   LABEL     run label, the directory under runs/               [YYYYmmdd-HHMM]
#   BUDGET    p-code operation budget per run                    [5e9]
#   TIMEOUT   seconds per engine invocation before it is killed  [600]
#   CORPUS    directory holding manifest.json                    [./corpus]
#   RUNS      runs directory                                     [./runs]
#   RESULTS   results.json to append to                          [./results.json]
#   REPORT    report.html to write                               [./report.html]
#   NO_BUILD  set to skip building the engines
#   NO_REPORT set to skip truth.py and report.py
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

export ENGINES="${ENGINES:-qcode}"
export LABEL="${LABEL:-$(date +%Y%m%d-%H%M)}"
export CORPUS="${CORPUS:-$here/corpus}"
export RUNS="${RUNS:-$here/runs}"
RESULTS="${RESULTS:-$here/results.json}"
REPORT="${REPORT:-$here/report.html}"

if [[ -z "${NO_BUILD:-}" ]]; then
    IFS=',' read -ra engines <<<"$ENGINES"
    for engine in "${engines[@]}"; do
        spec="$here/engines/$engine/engine.json"
        if [[ -x "$here/engines/$engine/build.sh" ]]; then
            echo "[run-all] building $engine"
            "$here/engines/$engine/build.sh"
        elif [[ -f "$spec" ]] && python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("build") else 1)' "$spec"; then
            echo "[run-all] building $engine"
            (cd "$repo" && python3 -c 'import json,sys,subprocess; sys.exit(subprocess.call(json.load(open(sys.argv[1]))["build"]))' "$spec")
        fi
    done
fi

python3 "$here/lib/sweep.py"

if [[ -z "${NO_REPORT:-}" ]]; then
    python3 "$here/truth.py" --corpus "$CORPUS" "$RUNS/$LABEL"
    python3 "$here/report.py" --runs "$RUNS" --results "$RESULTS" --html "$REPORT"
fi
echo "[run-all] done: $RUNS/$LABEL"
