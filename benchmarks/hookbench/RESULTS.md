## Baseline: no instrumentation (ms, min of repeats)

| image | native | qcode-interp | qcode-jit | icicle | unicorn |
|---|---:|---:|---:|---:|---:|
| aha-mont64 | 0.261 |  | 50.5 | 23.7 | 1.2 |
| crc32 | 0.336 | 6034.6 | 24.8 | 10.8 | 16.2 |
| depthconv | 0.167 | 5284.1 | 32.0 | 12.0 | 4.0 |
| edn | 0.214 |  | 59.2 | 48.3 | 9.3 |
| huffbench | 0.168 |  | 85.9 | 51.1 | 14.2 |
| matmult-int | 0.147 | 5773.0 | 43.8 | 23.4 | 27.3 |
| md5sum | 0.137 |  | 52.7 | 29.5 | 10.0 |
| nettle-aes | 0.149 |  | 64.7 | 63.3 | 6.6 |
| nettle-sha256 | 0.221 |  | 170.6 | 141.8 | 12.9 |
| nsichneu | 0.179 |  | 322.1 | 214.9 | 23.7 |
| picojpeg | 0.174 |  | 160.5 | 112.2 | 22.8 |
| qrduino | 0.438 |  | 296.8 | 235.1 | 9.6 |
| sglib-combined | 0.223 |  | 130.9 | 95.1 | 16.5 |
| statemate | 0.174 |  | 45.2 | 35.2 | 50.9 |
| tarfind | 0.082 | 3048.8 | 29.6 | 20.2 | 23.6 |
| ud | 0.376 |  | 59.3 | 36.7 | 12.3 |
| xgboost | 0.706 |  | 77.1 | 23.9 | 9.8 |

## Slowdown relative to each engine's own baseline (geometric mean over images)

| instrumentation | native (compiler) | qcode-jit | icicle | unicorn | qcode-interp |
|---|---:|---:|---:|---:|---:|
| block-ir | 2.34× | 1.05× (n=17) | 1.04× (n=16) |  | 1.05× (n=4) |
| block-ram | 2.34× | 1.15× (n=17) |  |  |  |
| block-cb | 2.34× | 5.88× (n=17) | 1.08× (n=17) | 1.13× (n=17) |  |
| insn-ir |  | 1.10× (n=17) | 1.08× (n=17) |  | 1.20× (n=4) |
| insn-ram |  | 1.58× (n=17) |  |  |  |
| insn-cb |  | 17.11× (n=17) | 1.33× (n=17) | 3.71× (n=17) |  |
| edge-ir | 3.13× | 1.13× (n=17) |  |  |  |
| edge-ram | 3.13× | 1.33× (n=17) |  |  | 1.11× (n=4) |
| watch-ir | 1.99× | 1.65× (n=17) | 1.11× (n=17) | 1.87× (n=17) | 1.07× (n=4) |
| watch-cb | 1.99× | 3.86× (n=17) | 1.11× (n=17) | 1.87× (n=17) |  |
| cmp-ir | 1.96× | 1.94× (n=17) |  |  |  |
| cmp-ram | 1.96× | 4.93× (n=17) |  |  | 4.53× (n=4) |
| cmp-cb | 1.96× | 78.10× (n=17) |  | 1.13× (n=17) |  |

## Cost per host call (ns, median over images with ≥ 10k calls)

| instrumentation | qcode-jit | icicle | unicorn |
|---|---:|---:|---:|
| block-cb | 566 (n=17) | 6 (n=17) | 3 (n=17) |
| insn-cb | 366 (n=17) | 4 (n=17) | 8 (n=17) |
| watch-ir | 1197 (n=2) | 108 (n=2) | 126 (n=2) |
| watch-cb | 871 (n=15) | 103 (n=2) | 132 (n=2) |
| cmp-cb | 644 (n=17) |  | 3 (n=16) |

## Compiled instrumentation on qcode-jit: overhead per event (ns, median over images)

| instrumentation | ns/event | events per image (median) | sites per image (median) |
|---|---:|---:|---:|
| block-ir | 5.5 | 586384 | 71 |
| block-ram | 16.1 | 586384 | 71 |
| insn-ir | 2.2 | 2788252 | 466 |
| insn-ram | 12.8 | 2788252 | 466 |

## How the qcode-jit numbers moved (oldest first)

| instrumentation | first sweep | fix1 | fix2 (CPU time, insns) | fix3 (CPU time, insns) | main2 (CPU time, insns) | warm (CPU time, insns) | warm-wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| block-ir | 1.26× (n=17) | 1.14× (n=17) |  | 1.01× (n=17) (1.03× insns) | 1.05× (n=17) (1.03× insns) | 1.05× (n=17) (1.03× insns) | 1.05× (n=17) |
| block-ram | 1.41× (n=17) | 1.31× (n=17) |  | 1.22× (n=17) (1.14× insns) | 1.22× (n=17) (1.15× insns) | 1.12× (n=17) (1.09× insns) | 1.15× (n=17) |
| block-cb | 5.47× (n=17) | 5.32× (n=17) |  | 5.26× (n=17) (3.60× insns) | 4.85× (n=17) (3.53× insns) | 5.66× (n=17) (3.83× insns) | 5.88× (n=17) |
| insn-ir | 1.33× (n=17) | 1.20× (n=17) |  | 1.07× (n=17) (1.05× insns) | 1.06× (n=17) (1.05× insns) | 1.03× (n=17) (1.05× insns) | 1.10× (n=17) |
| insn-ram | 1.85× (n=17) | 1.69× (n=17) |  | 1.61× (n=17) (1.43× insns) | 1.56× (n=17) (1.44× insns) | 1.49× (n=17) (1.41× insns) | 1.58× (n=17) |
| insn-cb | 14.69× (n=17) | 13.51× (n=17) |  | 15.30× (n=17) (9.71× insns) | 13.29× (n=17) (9.79× insns) | 15.95× (n=17) (10.72× insns) | 17.11× (n=17) |
| edge-ir |  |  | 1.12× (n=17) (1.14× insns) | 1.73× (n=17) (1.08× insns) | 1.13× (n=17) (1.09× insns) | 1.07× (n=17) (1.07× insns) | 1.13× (n=17) |
| edge-ram | 1.55× (n=17) | 1.52× (n=17) | 1.48× (n=17) (1.35× insns) | 2.01× (n=17) (1.29× insns) | 1.46× (n=17) (1.29× insns) | 1.22× (n=17) (1.19× insns) | 1.33× (n=17) |
| watch-ir | 1.71× (n=17) | 1.58× (n=17) |  | 3.07× (n=17) (1.37× insns) | 1.61× (n=17) (1.39× insns) | 1.61× (n=17) (1.37× insns) | 1.65× (n=17) |
| watch-cb | 11.33× (n=17) | 3.20× (n=17) |  | 3.76× (n=17) (2.58× insns) | 3.54× (n=17) (2.61× insns) | 3.66× (n=17) (2.75× insns) | 3.86× (n=17) |
| cmp-ir |  |  | 2.52× (n=17) (2.14× insns) | 2.59× (n=17) (1.99× insns) | 1.85× (n=17) (1.62× insns) | 1.81× (n=17) (1.65× insns) | 1.94× (n=17) |
| cmp-ram | 6.91× (n=17) | 6.39× (n=17) | 4.08× (n=17) (5.01× insns) | 6.75× (n=17) (4.81× insns) | 4.42× (n=17) (3.59× insns) | 4.59× (n=17) (3.58× insns) | 4.93× (n=17) |
| cmp-cb | 14.55× (n=1) 16 ✗ | 88.12× (n=17) |  | 96.10× (n=17) (73.49× insns) | 58.92× (n=17) (50.49× insns) | 72.81× (n=17) (55.92× insns) | 78.10× (n=17) |

- first sweep
- fix1: After the JIT cache fix (commit c83141b): blocks carry a revision stamp, the cache is keyed on it, and compiled code is resumed from anywhere in a block. Only the QCode JIT was re-run; the other engines are unchanged.
- fix2: After bounded hook spaces (commit c3ad0a6): the edge map and the compare log leave guest RAM for a flat space the JIT indexes by a computed offset, one compare against the bound per access. edge-ir and cmp-ir are the hook-space versions; edge-ram and cmp-ram are what the earlier runs' edge-ir and cmp-ir measured. Only the QCode JIT was re-run, on these kinds and the baseline. The machine was loaded (see the load), so this run is timed by thread CPU time rather than the wall clock and is kept out of the charts; retired user instructions, which the load does not move, are given in parentheses.
- fix3: After the growing-block fix (commit 085e89a): a block offered to the injectors after each absorbed instruction no longer costs a rebuild of the interpreter's list, nor a walk of the whole block by the hook. Every QCode JIT row re-run, still on a loaded machine: CPU time, with retired user instructions beside it.
- main2: After the rebase onto main at 0e30226 (main's lifting consolidation, SSA uniques within a block and memoized lifts; the branch's own code unchanged). Every QCode JIT row re-run, on a loaded machine: CPU time, with retired user instructions beside it. The baseline itself moved: a lift is cheaper and a block holds fewer operations.
The cmp-ir huffbench instruction count was re-measured by hand: the sweep's perf run of it exited early (405374 instructions).
- warm: After the JIT warm-up (commit 7a925de): a block is compiled on its second entry at a revision, not its first, so straight-line code is not compiled after each absorbed instruction and code that runs once is not compiled at all. Every QCode JIT row re-run on a loaded machine (load 2 to 6): CPU time, with retired user instructions beside it. The watch-cb crc32 row was re-timed after the sweep (497 ms under a load spike; 205 ms quiet, and 2.83G instructions against 2.88G before the change).
- warm-wall: The wall-clock run after the JIT warm-up (commit 7a925de), on a quiet machine (load 1 to 2), --repeat 5, min: every QCode JIT row, filling the chart rows. Rows a burst of load from another job inflated (wall time more than 10% over the thread CPU time of the same row in results-warm: watch-ir, cmp-ir, and cmp-cb on nsichneu and nettle-sha256) were re-measured the same way once the machine was quiet again; every row is now within 11% of its CPU time.

## Runs that did not verify

- icicle block-ir sglib-combined: UnhandledException(code=ReadUnmapped, value=0x4ee7) pc=0x40209a


## Every run

| engine | instr | image | ms | host calls | events | sites | ok |
|---|---|---|---:|---:|---:|---:|---|
| icicle | block-cb | aha-mont64 | 26.0 | 506129 | 506129 | 62 | ok |
| icicle | block-cb | crc32 | 12.9 | 526015 | 526015 | 26 | ok |
| icicle | block-cb | depthconv | 13.4 | 371548 | 371548 | 38 | ok |
| icicle | block-cb | edn | 52.3 | 410566 | 410566 | 70 | ok |
| icicle | block-cb | huffbench | 57.5 | 692955 | 692955 | 162 | ok |
| icicle | block-cb | matmult-int | 26.2 | 603666 | 603666 | 49 | ok |
| icicle | block-cb | md5sum | 35.1 | 348352 | 348352 | 75 | ok |
| icicle | block-cb | nettle-aes | 68.6 | 71809 | 71809 | 65 | ok |
| icicle | block-cb | nettle-sha256 | 148.2 | 171180 | 171180 | 80 | ok |
| icicle | block-cb | nsichneu | 238.1 | 771897 | 771897 | 657 | ok |
| icicle | block-cb | picojpeg | 122.8 | 419005 | 419005 | 318 | ok |
| icicle | block-cb | qrduino | 238.4 | 518116 | 518116 | 504 | ok |
| icicle | block-cb | sglib-combined | 96.4 | 730627 | 730627 | 254 | ok |
| icicle | block-cb | statemate | 37.3 | 326595 | 326595 | 80 | ok |
| icicle | block-cb | tarfind | 21.0 | 365673 | 365673 | 56 | ok |
| icicle | block-cb | ud | 35.3 | 482567 | 482567 | 60 | ok |
| icicle | block-cb | xgboost | 23.7 | 1051627 | 1051627 | 44 | ok |
| icicle | block-ir | aha-mont64 | 25.7 | 0 | 506129 | 62 | ok |
| icicle | block-ir | crc32 | 11.7 | 0 | 526015 | 26 | ok |
| icicle | block-ir | depthconv | 12.7 | 0 | 371548 | 38 | ok |
| icicle | block-ir | edn | 52.4 | 0 | 410566 | 70 | ok |
| icicle | block-ir | huffbench | 55.1 | 0 | 692955 | 162 | ok |
| icicle | block-ir | matmult-int | 25.5 | 0 | 603666 | 49 | ok |
| icicle | block-ir | md5sum | 32.6 | 0 | 348352 | 75 | ok |
| icicle | block-ir | nettle-aes | 66.7 | 0 | 71809 | 65 | ok |
| icicle | block-ir | nettle-sha256 | 149.8 | 0 | 171180 | 80 | ok |
| icicle | block-ir | nsichneu | 221.7 | 0 | 771897 | 657 | ok |
| icicle | block-ir | picojpeg | 121.3 | 0 | 419005 | 318 | ok |
| icicle | block-ir | qrduino | 238.0 | 0 | 518116 | 504 | ok |
| icicle | block-ir | sglib-combined | 73.1 | 0 | 20191 | 215 | UnhandledException(code=ReadUnmapped, va |
| icicle | block-ir | statemate | 34.7 | 0 | 326595 | 80 | ok |
| icicle | block-ir | tarfind | 20.8 | 0 | 365673 | 56 | ok |
| icicle | block-ir | ud | 34.6 | 0 | 482567 | 60 | ok |
| icicle | block-ir | xgboost | 23.1 | 0 | 1051627 | 44 | ok |
| icicle | insn-cb | aha-mont64 | 36.1 | 2428965 | 2428965 | 383 | ok |
| icicle | insn-cb | crc32 | 16.0 | 2278993 | 2278993 | 87 | ok |
| icicle | insn-cb | depthconv | 20.1 | 2998977 | 2998977 | 106 | ok |
| icicle | insn-cb | edn | 64.3 | 4109871 | 4109871 | 668 | ok |
| icicle | insn-cb | huffbench | 67.6 | 2833398 | 2833398 | 641 | ok |
| icicle | insn-cb | matmult-int | 36.0 | 4072598 | 4072598 | 309 | ok |
| icicle | insn-cb | md5sum | 39.4 | 2398542 | 2398542 | 390 | ok |
| icicle | insn-cb | nettle-aes | 80.1 | 2622958 | 2622958 | 962 | ok |
| icicle | insn-cb | nettle-sha256 | 212.1 | 4476540 | 4476540 | 2714 | ok |
| icicle | insn-cb | nsichneu | 261.4 | 2167729 | 2167729 | 1853 | ok |
| icicle | insn-cb | picojpeg | 145.6 | 3300870 | 3300870 | 1573 | ok |
| icicle | insn-cb | qrduino | 289.0 | 3671519 | 3671519 | 3271 | ok |
| icicle | insn-cb | sglib-combined | 112.1 | 2818502 | 2818502 | 1047 | ok |
| icicle | insn-cb | statemate | 42.4 | 2235592 | 2235592 | 431 | ok |
| icicle | insn-cb | tarfind | 25.1 | 1942284 | 1942284 | 210 | ok |
| icicle | insn-cb | ud | 46.1 | 3023154 | 3023154 | 359 | ok |
| icicle | insn-cb | xgboost | 29.2 | 3936135 | 3936135 | 147 | ok |
| icicle | insn-ir | aha-mont64 | 27.0 | 0 | 2428965 | 383 | ok |
| icicle | insn-ir | crc32 | 11.9 | 0 | 2278993 | 87 | ok |
| icicle | insn-ir | depthconv | 13.5 | 0 | 2998977 | 106 | ok |
| icicle | insn-ir | edn | 52.5 | 0 | 4109871 | 668 | ok |
| icicle | insn-ir | huffbench | 55.8 | 0 | 2833398 | 641 | ok |
| icicle | insn-ir | matmult-int | 26.4 | 0 | 4072598 | 309 | ok |
| icicle | insn-ir | md5sum | 33.3 | 0 | 2398542 | 390 | ok |
| icicle | insn-ir | nettle-aes | 69.7 | 0 | 2622958 | 962 | ok |
| icicle | insn-ir | nettle-sha256 | 166.0 | 0 | 4476540 | 2714 | ok |
| icicle | insn-ir | nsichneu | 226.3 | 0 | 2167729 | 1853 | ok |
| icicle | insn-ir | picojpeg | 123.5 | 0 | 3300870 | 1573 | ok |
| icicle | insn-ir | qrduino | 247.6 | 0 | 3671519 | 3271 | ok |
| icicle | insn-ir | sglib-combined | 96.5 | 0 | 2818502 | 1047 | ok |
| icicle | insn-ir | statemate | 35.3 | 0 | 2235592 | 431 | ok |
| icicle | insn-ir | tarfind | 20.7 | 0 | 1942284 | 210 | ok |
| icicle | insn-ir | ud | 35.3 | 0 | 3023154 | 359 | ok |
| icicle | insn-ir | xgboost | 25.3 | 0 | 3936135 | 147 | ok |
| icicle | none | aha-mont64 | 23.7 | 0 | 0 | 0 | ok |
| icicle | none | crc32 | 10.8 | 0 | 0 | 0 | ok |
| icicle | none | depthconv | 12.0 | 0 | 0 | 0 | ok |
| icicle | none | edn | 48.3 | 0 | 0 | 0 | ok |
| icicle | none | huffbench | 51.1 | 0 | 0 | 0 | ok |
| icicle | none | matmult-int | 23.4 | 0 | 0 | 0 | ok |
| icicle | none | md5sum | 29.5 | 0 | 0 | 0 | ok |
| icicle | none | nettle-aes | 63.3 | 0 | 0 | 0 | ok |
| icicle | none | nettle-sha256 | 141.8 | 0 | 0 | 0 | ok |
| icicle | none | nsichneu | 214.9 | 0 | 0 | 0 | ok |
| icicle | none | picojpeg | 112.2 | 0 | 0 | 0 | ok |
| icicle | none | qrduino | 235.1 | 0 | 0 | 0 | ok |
| icicle | none | sglib-combined | 95.1 | 0 | 0 | 0 | ok |
| icicle | none | statemate | 35.2 | 0 | 0 | 0 | ok |
| icicle | none | tarfind | 20.2 | 0 | 0 | 0 | ok |
| icicle | none | ud | 36.7 | 0 | 0 | 0 | ok |
| icicle | none | xgboost | 23.9 | 0 | 0 | 0 | ok |
| icicle | watch-cb | aha-mont64 | 23.6 | 3 | 3 | 0 | ok |
| icicle | watch-cb | crc32 | 16.9 | 175275 | 175275 | 0 | ok |
| icicle | watch-cb | depthconv | 13.9 | 0 | 0 | 0 | ok |
| icicle | watch-cb | edn | 51.3 | 1066 | 1066 | 0 | ok |
| icicle | watch-cb | huffbench | 52.6 | 768 | 768 | 0 | ok |
| icicle | watch-cb | matmult-int | 31.6 | 3360 | 3360 | 0 | ok |
| icicle | watch-cb | md5sum | 34.0 | 536 | 536 | 0 | ok |
| icicle | watch-cb | nettle-aes | 65.1 | 0 | 0 | 0 | ok |
| icicle | watch-cb | nettle-sha256 | 147.0 | 0 | 0 | 0 | ok |
| icicle | watch-cb | nsichneu | 217.0 | 2466 | 2466 | 0 | ok |
| icicle | watch-cb | picojpeg | 123.5 | 402 | 402 | 0 | ok |
| icicle | watch-cb | qrduino | 236.3 | 78 | 78 | 0 | ok |
| icicle | watch-cb | sglib-combined | 92.8 | 768 | 768 | 0 | ok |
| icicle | watch-cb | statemate | 55.3 | 116585 | 116585 | 0 | ok |
| icicle | watch-cb | tarfind | 23.0 | 1739 | 1739 | 0 | ok |
| icicle | watch-cb | ud | 37.3 | 1786 | 1786 | 0 | ok |
| icicle | watch-cb | xgboost | 22.6 | 0 | 0 | 0 | ok |
| icicle | watch-ir | aha-mont64 | 24.7 | 3 | 3 | 0 | ok |
| icicle | watch-ir | crc32 | 17.2 | 175275 | 175275 | 0 | ok |
| icicle | watch-ir | depthconv | 13.8 | 0 | 0 | 0 | ok |
| icicle | watch-ir | edn | 51.4 | 1066 | 1066 | 0 | ok |
| icicle | watch-ir | huffbench | 51.7 | 768 | 768 | 0 | ok |
| icicle | watch-ir | matmult-int | 32.5 | 3360 | 3360 | 0 | ok |
| icicle | watch-ir | md5sum | 34.5 | 536 | 536 | 0 | ok |
| icicle | watch-ir | nettle-aes | 64.8 | 0 | 0 | 0 | ok |
| icicle | watch-ir | nettle-sha256 | 144.4 | 0 | 0 | 0 | ok |
| icicle | watch-ir | nsichneu | 216.9 | 2466 | 2466 | 0 | ok |
| icicle | watch-ir | picojpeg | 120.5 | 402 | 402 | 0 | ok |
| icicle | watch-ir | qrduino | 233.1 | 78 | 78 | 0 | ok |
| icicle | watch-ir | sglib-combined | 92.4 | 768 | 768 | 0 | ok |
| icicle | watch-ir | statemate | 56.1 | 116585 | 116585 | 0 | ok |
| icicle | watch-ir | tarfind | 22.9 | 1739 | 1739 | 0 | ok |
| icicle | watch-ir | ud | 37.0 | 1786 | 1786 | 0 | ok |
| icicle | watch-ir | xgboost | 22.0 | 0 | 0 | 0 | ok |
| qcode-interp | block-ir | crc32 | 6120.8 | 0 | 350902 | 19 | ok |
| qcode-interp | block-ir | depthconv | 5575.4 | 0 | 635575 | 31 | ok |
| qcode-interp | block-ir | matmult-int | 6243.2 | 0 | 1200608 | 48 | ok |
| qcode-interp | block-ir | tarfind | 3222.9 | 0 | 331253 | 47 | ok |
| qcode-interp | cmp-ram | crc32 | 38188.0 | 0 | 1092 | 324 | ok |
| qcode-interp | cmp-ram | depthconv | 22439.4 | 0 | 464 | 327 | ok |
| qcode-interp | cmp-ram | matmult-int | 16891.8 | 0 | 615 | 625 | ok |
| qcode-interp | cmp-ram | tarfind | 16320.1 | 0 | 302 | 578 | ok |
| qcode-interp | edge-ram | crc32 | 6146.4 | 0 | 21 | 19 | ok |
| qcode-interp | edge-ram | depthconv | 5736.4 | 0 | 34 | 31 | ok |
| qcode-interp | edge-ram | matmult-int | 7016.3 | 0 | 62 | 48 | ok |
| qcode-interp | edge-ram | tarfind | 3421.2 | 0 | 57 | 47 | ok |
| qcode-interp | insn-ir | crc32 | 6746.6 | 0 | 2278822 | 108 | ok |
| qcode-interp | insn-ir | depthconv | 6455.5 | 0 | 2998976 | 104 | ok |
| qcode-interp | insn-ir | matmult-int | 7274.4 | 0 | 4072595 | 283 | ok |
| qcode-interp | insn-ir | tarfind | 3728.6 | 0 | 1942280 | 234 | ok |
| qcode-interp | none | crc32 | 6034.6 | 0 | 0 | 0 | ok |
| qcode-interp | none | depthconv | 5284.1 | 0 | 0 | 0 | ok |
| qcode-interp | none | matmult-int | 5773.0 | 0 | 0 | 0 | ok |
| qcode-interp | none | tarfind | 3048.8 | 0 | 0 | 0 | ok |
| qcode-interp | watch-ir | crc32 | 6201.7 | 175275 | 175275 | 0 | ok |
| qcode-interp | watch-ir | depthconv | 5319.7 | 0 | 0 | 0 | ok |
| qcode-interp | watch-ir | matmult-int | 6573.2 | 3360 | 3360 | 0 | ok |
| qcode-interp | watch-ir | tarfind | 3346.2 | 1739 | 1739 | 0 | ok |
| qcode-jit | block-cb | aha-mont64 | 494.4 | 428535 | 428535 | 0 | ok |
| qcode-jit | block-cb | crc32 | 416.3 | 350902 | 350902 | 0 | ok |
| qcode-jit | block-cb | depthconv | 392.0 | 635574 | 635574 | 0 | ok |
| qcode-jit | block-cb | edn | 311.3 | 817048 | 817048 | 0 | ok |
| qcode-jit | block-cb | huffbench | 436.5 | 797610 | 797610 | 0 | ok |
| qcode-jit | block-cb | matmult-int | 350.9 | 1200608 | 1200608 | 0 | ok |
| qcode-jit | block-cb | md5sum | 363.0 | 586384 | 586384 | 0 | ok |
| qcode-jit | block-cb | nettle-aes | 211.4 | 115772 | 115772 | 0 | ok |
| qcode-jit | block-cb | nettle-sha256 | 525.8 | 246040 | 246040 | 0 | ok |
| qcode-jit | block-cb | nsichneu | 973.6 | 773121 | 773121 | 0 | ok |
| qcode-jit | block-cb | picojpeg | 706.2 | 515125 | 515125 | 0 | ok |
| qcode-jit | block-cb | qrduino | 888.6 | 566955 | 566955 | 0 | ok |
| qcode-jit | block-cb | sglib-combined | 553.7 | 818285 | 818285 | 0 | ok |
| qcode-jit | block-cb | statemate | 214.8 | 336637 | 336637 | 0 | ok |
| qcode-jit | block-cb | tarfind | 225.6 | 329608 | 329608 | 0 | ok |
| qcode-jit | block-cb | ud | 462.1 | 748654 | 748654 | 0 | ok |
| qcode-jit | block-cb | xgboost | 661.5 | 1317252 | 1317252 | 0 | ok |
| qcode-jit | block-ir | aha-mont64 | 49.1 | 0 | 428535 | 43 | ok |
| qcode-jit | block-ir | crc32 | 25.5 | 0 | 350902 | 19 | ok |
| qcode-jit | block-ir | depthconv | 33.5 | 0 | 635574 | 30 | ok |
| qcode-jit | block-ir | edn | 60.9 | 0 | 817048 | 79 | ok |
| qcode-jit | block-ir | huffbench | 91.9 | 0 | 797610 | 148 | ok |
| qcode-jit | block-ir | matmult-int | 46.7 | 0 | 1200608 | 48 | ok |
| qcode-jit | block-ir | md5sum | 54.7 | 0 | 586384 | 76 | ok |
| qcode-jit | block-ir | nettle-aes | 63.4 | 0 | 115772 | 71 | ok |
| qcode-jit | block-ir | nettle-sha256 | 172.0 | 0 | 246040 | 69 | ok |
| qcode-jit | block-ir | nsichneu | 325.5 | 0 | 773121 | 652 | ok |
| qcode-jit | block-ir | picojpeg | 169.6 | 0 | 515125 | 322 | ok |
| qcode-jit | block-ir | qrduino | 308.3 | 0 | 566955 | 504 | ok |
| qcode-jit | block-ir | sglib-combined | 142.4 | 0 | 818285 | 278 | ok |
| qcode-jit | block-ir | statemate | 49.4 | 0 | 336637 | 75 | ok |
| qcode-jit | block-ir | tarfind | 32.5 | 0 | 329608 | 46 | ok |
| qcode-jit | block-ir | ud | 63.6 | 0 | 748654 | 66 | ok |
| qcode-jit | block-ir | xgboost | 85.7 | 0 | 1317252 | 35 | ok |
| qcode-jit | block-ram | aha-mont64 | 54.9 | 0 | 428535 | 43 | ok |
| qcode-jit | block-ram | crc32 | 25.0 | 0 | 350902 | 19 | ok |
| qcode-jit | block-ram | depthconv | 35.3 | 0 | 635574 | 30 | ok |
| qcode-jit | block-ram | edn | 67.4 | 0 | 817048 | 79 | ok |
| qcode-jit | block-ram | huffbench | 104.0 | 0 | 797610 | 148 | ok |
| qcode-jit | block-ram | matmult-int | 51.8 | 0 | 1200608 | 48 | ok |
| qcode-jit | block-ram | md5sum | 61.3 | 0 | 586384 | 76 | ok |
| qcode-jit | block-ram | nettle-aes | 71.8 | 0 | 115772 | 71 | ok |
| qcode-jit | block-ram | nettle-sha256 | 177.0 | 0 | 246040 | 69 | ok |
| qcode-jit | block-ram | nsichneu | 389.8 | 0 | 773121 | 652 | ok |
| qcode-jit | block-ram | picojpeg | 204.4 | 0 | 515125 | 322 | ok |
| qcode-jit | block-ram | qrduino | 357.5 | 0 | 566955 | 504 | ok |
| qcode-jit | block-ram | sglib-combined | 162.0 | 0 | 818285 | 278 | ok |
| qcode-jit | block-ram | statemate | 53.3 | 0 | 336637 | 75 | ok |
| qcode-jit | block-ram | tarfind | 35.0 | 0 | 329608 | 46 | ok |
| qcode-jit | block-ram | ud | 67.8 | 0 | 748654 | 66 | ok |
| qcode-jit | block-ram | xgboost | 82.7 | 0 | 1317252 | 35 | ok |
| qcode-jit | cmp-cb | aha-mont64 | 9626.9 | 16094853 | 16094853 | 1423 | ok |
| qcode-jit | cmp-cb | crc32 | 8034.7 | 12437230 | 12437230 | 189 | ok |
| qcode-jit | cmp-cb | depthconv | 5727.8 | 9828253 | 9828253 | 230 | ok |
| qcode-jit | cmp-cb | edn | 4985.5 | 8998843 | 8998843 | 1321 | ok |
| qcode-jit | cmp-cb | huffbench | 3771.7 | 6462517 | 6462517 | 1244 | ok |
| qcode-jit | cmp-cb | matmult-int | 5967.2 | 9575726 | 9575726 | 438 | ok |
| qcode-jit | cmp-cb | md5sum | 3462.7 | 5818149 | 5818149 | 646 | ok |
| qcode-jit | cmp-cb | nettle-aes | 8253.1 | 11546929 | 11546929 | 3905 | ok |
| qcode-jit | cmp-cb | nettle-sha256 | 27087.4 | 18263365 | 18263365 | 11123 | ok |
| qcode-jit | cmp-cb | nsichneu | 6232.1 | 5261397 | 5261397 | 4412 | ok |
| qcode-jit | cmp-cb | picojpeg | 9090.3 | 11900029 | 11900029 | 4699 | ok |
| qcode-jit | cmp-cb | qrduino | 9275.3 | 10279164 | 10279164 | 9856 | ok |
| qcode-jit | cmp-cb | sglib-combined | 3742.2 | 5075876 | 5075876 | 1905 | ok |
| qcode-jit | cmp-cb | statemate | 1459.2 | 2169660 | 2169660 | 421 | ok |
| qcode-jit | cmp-cb | tarfind | 4286.0 | 6616627 | 6616627 | 386 | ok |
| qcode-jit | cmp-cb | ud | 4905.6 | 6905552 | 6905552 | 758 | ok |
| qcode-jit | cmp-cb | xgboost | 4124.1 | 6380841 | 6380841 | 310 | ok |
| qcode-jit | cmp-ir | aha-mont64 | 110.8 | 0 | 1669 | 1423 | ok |
| qcode-jit | cmp-ir | crc32 | 46.3 | 0 | 1774 | 189 | ok |
| qcode-jit | cmp-ir | depthconv | 54.7 | 0 | 1949 | 230 | ok |
| qcode-jit | cmp-ir | edn | 116.0 | 0 | 4027 | 1321 | ok |
| qcode-jit | cmp-ir | huffbench | 155.2 | 0 | 3125 | 1244 | ok |
| qcode-jit | cmp-ir | matmult-int | 71.7 | 0 | 3374 | 438 | ok |
| qcode-jit | cmp-ir | md5sum | 89.3 | 0 | 1829 | 646 | ok |
| qcode-jit | cmp-ir | nettle-aes | 223.1 | 0 | 305 | 3905 | ok |
| qcode-jit | cmp-ir | nettle-sha256 | 670.8 | 0 | 3397 | 11123 | ok |
| qcode-jit | cmp-ir | nsichneu | 549.1 | 0 | 2133 | 4412 | ok |
| qcode-jit | cmp-ir | picojpeg | 368.6 | 0 | 1149 | 4699 | ok |
| qcode-jit | cmp-ir | qrduino | 738.1 | 0 | 2300 | 9856 | ok |
| qcode-jit | cmp-ir | sglib-combined | 225.3 | 0 | 932 | 1905 | ok |
| qcode-jit | cmp-ir | statemate | 62.6 | 0 | 2876 | 421 | ok |
| qcode-jit | cmp-ir | tarfind | 54.5 | 0 | 1587 | 386 | ok |
| qcode-jit | cmp-ir | ud | 93.8 | 0 | 3792 | 758 | ok |
| qcode-jit | cmp-ir | xgboost | 99.9 | 0 | 3369 | 310 | ok |
| qcode-jit | cmp-ram | aha-mont64 | 376.1 | 0 | 1669 | 1423 | ok |
| qcode-jit | cmp-ram | crc32 | 164.3 | 0 | 1774 | 189 | ok |
| qcode-jit | cmp-ram | depthconv | 151.8 | 0 | 1949 | 230 | ok |
| qcode-jit | cmp-ram | edn | 317.7 | 0 | 4027 | 1321 | ok |
| qcode-jit | cmp-ram | huffbench | 319.0 | 0 | 3125 | 1244 | ok |
| qcode-jit | cmp-ram | matmult-int | 152.4 | 0 | 3374 | 438 | ok |
| qcode-jit | cmp-ram | md5sum | 200.8 | 0 | 1829 | 646 | ok |
| qcode-jit | cmp-ram | nettle-aes | 928.5 | 0 | 305 | 3905 | ok |
| qcode-jit | cmp-ram | nettle-sha256 | 2318.8 | 0 | 3397 | 11123 | ok |
| qcode-jit | cmp-ram | nsichneu | 1238.8 | 0 | 2133 | 4412 | ok |
| qcode-jit | cmp-ram | picojpeg | 1026.3 | 0 | 1149 | 4699 | ok |
| qcode-jit | cmp-ram | qrduino | 2021.6 | 0 | 2300 | 9856 | ok |
| qcode-jit | cmp-ram | sglib-combined | 473.7 | 0 | 932 | 1905 | ok |
| qcode-jit | cmp-ram | statemate | 113.4 | 0 | 2876 | 421 | ok |
| qcode-jit | cmp-ram | tarfind | 138.3 | 0 | 1587 | 386 | ok |
| qcode-jit | cmp-ram | ud | 202.4 | 0 | 3792 | 758 | ok |
| qcode-jit | cmp-ram | xgboost | 157.4 | 0 | 3369 | 310 | ok |
| qcode-jit | edge-ir | aha-mont64 | 53.4 | 0 | 58 | 43 | ok |
| qcode-jit | edge-ir | crc32 | 27.2 | 0 | 21 | 19 | ok |
| qcode-jit | edge-ir | depthconv | 36.1 | 0 | 33 | 30 | ok |
| qcode-jit | edge-ir | edn | 66.9 | 0 | 106 | 79 | ok |
| qcode-jit | edge-ir | huffbench | 104.1 | 0 | 207 | 148 | ok |
| qcode-jit | edge-ir | matmult-int | 49.0 | 0 | 62 | 48 | ok |
| qcode-jit | edge-ir | md5sum | 59.0 | 0 | 89 | 76 | ok |
| qcode-jit | edge-ir | nettle-aes | 68.6 | 0 | 94 | 71 | ok |
| qcode-jit | edge-ir | nettle-sha256 | 178.2 | 0 | 90 | 69 | ok |
| qcode-jit | edge-ir | nsichneu | 370.8 | 0 | 650 | 652 | ok |
| qcode-jit | edge-ir | picojpeg | 186.4 | 0 | 454 | 322 | ok |
| qcode-jit | edge-ir | qrduino | 342.6 | 0 | 741 | 504 | ok |
| qcode-jit | edge-ir | sglib-combined | 155.4 | 0 | 395 | 278 | ok |
| qcode-jit | edge-ir | statemate | 52.2 | 0 | 82 | 75 | ok |
| qcode-jit | edge-ir | tarfind | 34.7 | 0 | 56 | 46 | ok |
| qcode-jit | edge-ir | ud | 68.2 | 0 | 91 | 66 | ok |
| qcode-jit | edge-ir | xgboost | 89.9 | 0 | 40 | 35 | ok |
| qcode-jit | edge-ram | aha-mont64 | 55.6 | 0 | 58 | 43 | ok |
| qcode-jit | edge-ram | crc32 | 27.8 | 0 | 21 | 19 | ok |
| qcode-jit | edge-ram | depthconv | 38.2 | 0 | 33 | 30 | ok |
| qcode-jit | edge-ram | edn | 74.6 | 0 | 106 | 79 | ok |
| qcode-jit | edge-ram | huffbench | 120.0 | 0 | 207 | 148 | ok |
| qcode-jit | edge-ram | matmult-int | 57.2 | 0 | 62 | 48 | ok |
| qcode-jit | edge-ram | md5sum | 68.5 | 0 | 89 | 76 | ok |
| qcode-jit | edge-ram | nettle-aes | 77.9 | 0 | 94 | 71 | ok |
| qcode-jit | edge-ram | nettle-sha256 | 180.8 | 0 | 90 | 69 | ok |
| qcode-jit | edge-ram | nsichneu | 469.8 | 0 | 650 | 652 | ok |
| qcode-jit | edge-ram | picojpeg | 235.7 | 0 | 454 | 322 | ok |
| qcode-jit | edge-ram | qrduino | 425.6 | 0 | 741 | 504 | ok |
| qcode-jit | edge-ram | sglib-combined | 203.3 | 0 | 395 | 278 | ok |
| qcode-jit | edge-ram | statemate | 65.5 | 0 | 82 | 75 | ok |
| qcode-jit | edge-ram | tarfind | 43.0 | 0 | 56 | 46 | ok |
| qcode-jit | edge-ram | ud | 83.2 | 0 | 91 | 66 | ok |
| qcode-jit | edge-ram | xgboost | 124.4 | 0 | 40 | 35 | ok |
| qcode-jit | insn-cb | aha-mont64 | 1073.3 | 2421390 | 2421390 | 0 | ok |
| qcode-jit | insn-cb | crc32 | 923.6 | 2278821 | 2278821 | 0 | ok |
| qcode-jit | insn-cb | depthconv | 1014.6 | 2998975 | 2998975 | 0 | ok |
| qcode-jit | insn-cb | edn | 1255.1 | 4092147 | 4092147 | 0 | ok |
| qcode-jit | insn-cb | huffbench | 1046.1 | 2788252 | 2788252 | 0 | ok |
| qcode-jit | insn-cb | matmult-int | 1092.3 | 4023490 | 4023490 | 0 | ok |
| qcode-jit | insn-cb | md5sum | 929.5 | 2397929 | 2397929 | 0 | ok |
| qcode-jit | insn-cb | nettle-aes | 1132.4 | 2619634 | 2619634 | 0 | ok |
| qcode-jit | insn-cb | nettle-sha256 | 3656.5 | 4466392 | 4466392 | 0 | ok |
| qcode-jit | insn-cb | nsichneu | 2171.8 | 2165260 | 2165260 | 0 | ok |
| qcode-jit | insn-cb | picojpeg | 1849.3 | 3289287 | 3289287 | 0 | ok |
| qcode-jit | insn-cb | qrduino | 2324.7 | 3659330 | 3659330 | 0 | ok |
| qcode-jit | insn-cb | sglib-combined | 1255.5 | 2787222 | 2787222 | 0 | ok |
| qcode-jit | insn-cb | statemate | 843.3 | 2228925 | 2228925 | 0 | ok |
| qcode-jit | insn-cb | tarfind | 681.7 | 1937298 | 1937298 | 0 | ok |
| qcode-jit | insn-cb | ud | 1114.5 | 2933844 | 2933844 | 0 | ok |
| qcode-jit | insn-cb | xgboost | 1415.4 | 3933565 | 3933565 | 0 | ok |
| qcode-jit | insn-ir | aha-mont64 | 52.1 | 0 | 2421390 | 466 | ok |
| qcode-jit | insn-ir | crc32 | 27.2 | 0 | 2278821 | 107 | ok |
| qcode-jit | insn-ir | depthconv | 35.4 | 0 | 2998975 | 103 | ok |
| qcode-jit | insn-ir | edn | 64.3 | 0 | 4092147 | 662 | ok |
| qcode-jit | insn-ir | huffbench | 97.8 | 0 | 2788252 | 701 | ok |
| qcode-jit | insn-ir | matmult-int | 47.7 | 0 | 4023490 | 251 | ok |
| qcode-jit | insn-ir | md5sum | 58.0 | 0 | 2397929 | 433 | ok |
| qcode-jit | insn-ir | nettle-aes | 73.1 | 0 | 2619634 | 1045 | ok |
| qcode-jit | insn-ir | nettle-sha256 | 198.6 | 0 | 4466392 | 2816 | ok |
| qcode-jit | insn-ir | nsichneu | 339.6 | 0 | 2165260 | 1852 | ok |
| qcode-jit | insn-ir | picojpeg | 179.6 | 0 | 3289287 | 1880 | ok |
| qcode-jit | insn-ir | qrduino | 335.8 | 0 | 3659330 | 3539 | ok |
| qcode-jit | insn-ir | sglib-combined | 139.7 | 0 | 2787222 | 1135 | ok |
| qcode-jit | insn-ir | statemate | 49.9 | 0 | 2228925 | 462 | ok |
| qcode-jit | insn-ir | tarfind | 34.8 | 0 | 1937298 | 228 | ok |
| qcode-jit | insn-ir | ud | 64.0 | 0 | 2933844 | 420 | ok |
| qcode-jit | insn-ir | xgboost | 85.6 | 0 | 3933565 | 178 | ok |
| qcode-jit | insn-ram | aha-mont64 | 70.0 | 0 | 2421390 | 466 | ok |
| qcode-jit | insn-ram | crc32 | 30.9 | 0 | 2278821 | 107 | ok |
| qcode-jit | insn-ram | depthconv | 42.5 | 0 | 2998975 | 103 | ok |
| qcode-jit | insn-ram | edn | 95.9 | 0 | 4092147 | 662 | ok |
| qcode-jit | insn-ram | huffbench | 140.7 | 0 | 2788252 | 701 | ok |
| qcode-jit | insn-ram | matmult-int | 60.3 | 0 | 4023490 | 251 | ok |
| qcode-jit | insn-ram | md5sum | 83.5 | 0 | 2397929 | 433 | ok |
| qcode-jit | insn-ram | nettle-aes | 126.1 | 0 | 2619634 | 1045 | ok |
| qcode-jit | insn-ram | nettle-sha256 | 339.3 | 0 | 4466392 | 2816 | ok |
| qcode-jit | insn-ram | nsichneu | 499.1 | 0 | 2165260 | 1852 | ok |
| qcode-jit | insn-ram | picojpeg | 298.1 | 0 | 3289287 | 1880 | ok |
| qcode-jit | insn-ram | qrduino | 534.0 | 0 | 3659330 | 3539 | ok |
| qcode-jit | insn-ram | sglib-combined | 217.7 | 0 | 2787222 | 1135 | ok |
| qcode-jit | insn-ram | statemate | 80.7 | 0 | 2228925 | 462 | ok |
| qcode-jit | insn-ram | tarfind | 46.0 | 0 | 1937298 | 228 | ok |
| qcode-jit | insn-ram | ud | 85.2 | 0 | 2933844 | 420 | ok |
| qcode-jit | insn-ram | xgboost | 101.1 | 0 | 3933565 | 178 | ok |
| qcode-jit | none | aha-mont64 | 50.5 | 0 | 0 | 0 | ok |
| qcode-jit | none | crc32 | 24.8 | 0 | 0 | 0 | ok |
| qcode-jit | none | depthconv | 32.0 | 0 | 0 | 0 | ok |
| qcode-jit | none | edn | 59.2 | 0 | 0 | 0 | ok |
| qcode-jit | none | huffbench | 85.9 | 0 | 0 | 0 | ok |
| qcode-jit | none | matmult-int | 43.8 | 0 | 0 | 0 | ok |
| qcode-jit | none | md5sum | 52.7 | 0 | 0 | 0 | ok |
| qcode-jit | none | nettle-aes | 64.7 | 0 | 0 | 0 | ok |
| qcode-jit | none | nettle-sha256 | 170.6 | 0 | 0 | 0 | ok |
| qcode-jit | none | nsichneu | 322.1 | 0 | 0 | 0 | ok |
| qcode-jit | none | picojpeg | 160.5 | 0 | 0 | 0 | ok |
| qcode-jit | none | qrduino | 296.8 | 0 | 0 | 0 | ok |
| qcode-jit | none | sglib-combined | 130.9 | 0 | 0 | 0 | ok |
| qcode-jit | none | statemate | 45.2 | 0 | 0 | 0 | ok |
| qcode-jit | none | tarfind | 29.6 | 0 | 0 | 0 | ok |
| qcode-jit | none | ud | 59.3 | 0 | 0 | 0 | ok |
| qcode-jit | none | xgboost | 77.1 | 0 | 0 | 0 | ok |
| qcode-jit | watch-cb | aha-mont64 | 68.6 | 5223 | 3 | 45 | ok |
| qcode-jit | watch-cb | crc32 | 207.3 | 350569 | 175275 | 28 | ok |
| qcode-jit | watch-cb | depthconv | 69.3 | 54128 | 0 | 14 | ok |
| qcode-jit | watch-cb | edn | 202.3 | 173163 | 1065 | 120 | ok |
| qcode-jit | watch-cb | huffbench | 290.7 | 348279 | 768 | 99 | ok |
| qcode-jit | watch-cb | matmult-int | 361.0 | 596259 | 3360 | 41 | ok |
| qcode-jit | watch-cb | md5sum | 179.5 | 213211 | 536 | 86 | ok |
| qcode-jit | watch-cb | nettle-aes | 200.4 | 79169 | 0 | 128 | ok |
| qcode-jit | watch-cb | nettle-sha256 | 587.6 | 258433 | 0 | 171 | ok |
| qcode-jit | watch-cb | nsichneu | 406.5 | 3719 | 2466 | 26 | ok |
| qcode-jit | watch-cb | picojpeg | 772.8 | 439371 | 402 | 367 | ok |
| qcode-jit | watch-cb | qrduino | 570.8 | 93934 | 78 | 352 | ok |
| qcode-jit | watch-cb | sglib-combined | 415.9 | 308596 | 768 | 202 | ok |
| qcode-jit | watch-cb | statemate | 698.1 | 1022639 | 116584 | 176 | ok |
| qcode-jit | watch-cb | tarfind | 497.3 | 536809 | 1739 | 50 | ok |
| qcode-jit | watch-cb | ud | 299.9 | 207216 | 1786 | 69 | ok |
| qcode-jit | watch-cb | xgboost | 174.0 | 104207 | 0 | 30 | ok |
| qcode-jit | watch-ir | aha-mont64 | 52.0 | 3 | 3 | 0 | ok |
| qcode-jit | watch-ir | crc32 | 100.5 | 175275 | 175275 | 0 | ok |
| qcode-jit | watch-ir | depthconv | 37.3 | 0 | 0 | 0 | ok |
| qcode-jit | watch-ir | edn | 86.4 | 1066 | 1066 | 0 | ok |
| qcode-jit | watch-ir | huffbench | 128.0 | 768 | 768 | 0 | ok |
| qcode-jit | watch-ir | matmult-int | 107.2 | 3360 | 3360 | 0 | ok |
| qcode-jit | watch-ir | md5sum | 81.2 | 536 | 536 | 0 | ok |
| qcode-jit | watch-ir | nettle-aes | 89.1 | 0 | 0 | 0 | ok |
| qcode-jit | watch-ir | nettle-sha256 | 173.0 | 0 | 0 | 0 | ok |
| qcode-jit | watch-ir | nsichneu | 327.0 | 2466 | 2466 | 0 | ok |
| qcode-jit | watch-ir | picojpeg | 234.3 | 402 | 402 | 0 | ok |
| qcode-jit | watch-ir | qrduino | 339.7 | 78 | 78 | 0 | ok |
| qcode-jit | watch-ir | sglib-combined | 195.6 | 768 | 768 | 0 | ok |
| qcode-jit | watch-ir | statemate | 273.9 | 116585 | 116585 | 0 | ok |
| qcode-jit | watch-ir | tarfind | 94.5 | 1739 | 1739 | 0 | ok |
| qcode-jit | watch-ir | ud | 93.8 | 1786 | 1786 | 0 | ok |
| qcode-jit | watch-ir | xgboost | 91.0 | 0 | 0 | 0 | ok |
| unicorn | block-cb | aha-mont64 | 1.8 | 243614 | 243614 | 0 | ok |
| unicorn | block-cb | crc32 | 18.0 | 526015 | 526015 | 0 | ok |
| unicorn | block-cb | depthconv | 4.7 | 266588 | 266588 | 0 | ok |
| unicorn | block-cb | edn | 10.5 | 410566 | 410566 | 0 | ok |
| unicorn | block-cb | huffbench | 15.9 | 577959 | 577959 | 0 | ok |
| unicorn | block-cb | matmult-int | 27.6 | 603666 | 603666 | 0 | ok |
| unicorn | block-cb | md5sum | 10.9 | 348084 | 348084 | 0 | ok |
| unicorn | block-cb | nettle-aes | 6.6 | 71809 | 71809 | 0 | ok |
| unicorn | block-cb | nettle-sha256 | 14.0 | 148660 | 148660 | 0 | ok |
| unicorn | block-cb | nsichneu | 32.7 | 771897 | 771897 | 0 | ok |
| unicorn | block-cb | picojpeg | 24.2 | 298867 | 298867 | 0 | ok |
| unicorn | block-cb | qrduino | 11.4 | 479302 | 479302 | 0 | ok |
| unicorn | block-cb | sglib-combined | 18.8 | 696227 | 696227 | 0 | ok |
| unicorn | block-cb | statemate | 51.3 | 326595 | 326595 | 0 | ok |
| unicorn | block-cb | tarfind | 24.0 | 365485 | 365485 | 0 | ok |
| unicorn | block-cb | ud | 12.7 | 482567 | 482567 | 0 | ok |
| unicorn | block-cb | xgboost | 11.3 | 1045727 | 1045727 | 0 | ok |
| unicorn | cmp-cb | aha-mont64 | 1.9 | 188258 | 188258 | 0 | ok |
| unicorn | cmp-cb | crc32 | 16.0 | 343 | 343 | 0 | ok |
| unicorn | cmp-cb | depthconv | 5.6 | 368245 | 368245 | 0 | ok |
| unicorn | cmp-cb | edn | 10.6 | 407997 | 407997 | 0 | ok |
| unicorn | cmp-cb | huffbench | 15.9 | 476332 | 476332 | 0 | ok |
| unicorn | cmp-cb | matmult-int | 28.6 | 603320 | 603320 | 0 | ok |
| unicorn | cmp-cb | md5sum | 11.3 | 310881 | 310881 | 0 | ok |
| unicorn | cmp-cb | nettle-aes | 6.7 | 61213 | 61213 | 0 | ok |
| unicorn | cmp-cb | nettle-sha256 | 13.4 | 126128 | 126128 | 0 | ok |
| unicorn | cmp-cb | nsichneu | 31.9 | 771877 | 771877 | 0 | ok |
| unicorn | cmp-cb | picojpeg | 23.6 | 306702 | 306702 | 0 | ok |
| unicorn | cmp-cb | qrduino | 11.7 | 367749 | 367749 | 0 | ok |
| unicorn | cmp-cb | sglib-combined | 17.8 | 350326 | 350326 | 0 | ok |
| unicorn | cmp-cb | statemate | 50.0 | 206657 | 206657 | 0 | ok |
| unicorn | cmp-cb | tarfind | 24.5 | 265363 | 265363 | 0 | ok |
| unicorn | cmp-cb | ud | 13.2 | 384315 | 384315 | 0 | ok |
| unicorn | cmp-cb | xgboost | 10.8 | 476648 | 476648 | 0 | ok |
| unicorn | insn-cb | aha-mont64 | 21.5 | 2428965 | 2428965 | 0 | ok |
| unicorn | insn-cb | crc32 | 35.6 | 2278993 | 2278993 | 0 | ok |
| unicorn | insn-cb | depthconv | 28.0 | 2998977 | 2998977 | 0 | ok |
| unicorn | insn-cb | edn | 42.4 | 4109871 | 4109871 | 0 | ok |
| unicorn | insn-cb | huffbench | 36.1 | 2833398 | 2833398 | 0 | ok |
| unicorn | insn-cb | matmult-int | 54.2 | 4072598 | 4072598 | 0 | ok |
| unicorn | insn-cb | md5sum | 27.7 | 2398542 | 2398542 | 0 | ok |
| unicorn | insn-cb | nettle-aes | 35.2 | 2622958 | 2622958 | 0 | ok |
| unicorn | insn-cb | nettle-sha256 | 140.6 | 4476540 | 4476540 | 0 | ok |
| unicorn | insn-cb | nsichneu | 94.6 | 2167729 | 2167729 | 0 | ok |
| unicorn | insn-cb | picojpeg | 65.5 | 3300870 | 3300870 | 0 | ok |
| unicorn | insn-cb | qrduino | 41.1 | 3671519 | 3671519 | 0 | ok |
| unicorn | insn-cb | sglib-combined | 41.4 | 2818502 | 2818502 | 0 | ok |
| unicorn | insn-cb | statemate | 83.1 | 2235592 | 2235592 | 0 | ok |
| unicorn | insn-cb | tarfind | 38.6 | 1942284 | 1942284 | 0 | ok |
| unicorn | insn-cb | ud | 44.3 | 3023154 | 3023154 | 0 | ok |
| unicorn | insn-cb | xgboost | 36.4 | 3936135 | 3936135 | 0 | ok |
| unicorn | none | aha-mont64 | 1.2 | 0 | 0 | 0 | ok |
| unicorn | none | crc32 | 16.2 | 0 | 0 | 0 | ok |
| unicorn | none | depthconv | 4.0 | 0 | 0 | 0 | ok |
| unicorn | none | edn | 9.3 | 0 | 0 | 0 | ok |
| unicorn | none | huffbench | 14.2 | 0 | 0 | 0 | ok |
| unicorn | none | matmult-int | 27.3 | 0 | 0 | 0 | ok |
| unicorn | none | md5sum | 10.0 | 0 | 0 | 0 | ok |
| unicorn | none | nettle-aes | 6.6 | 0 | 0 | 0 | ok |
| unicorn | none | nettle-sha256 | 12.9 | 0 | 0 | 0 | ok |
| unicorn | none | nsichneu | 23.7 | 0 | 0 | 0 | ok |
| unicorn | none | picojpeg | 22.8 | 0 | 0 | 0 | ok |
| unicorn | none | qrduino | 9.6 | 0 | 0 | 0 | ok |
| unicorn | none | sglib-combined | 16.5 | 0 | 0 | 0 | ok |
| unicorn | none | statemate | 50.9 | 0 | 0 | 0 | ok |
| unicorn | none | tarfind | 23.6 | 0 | 0 | 0 | ok |
| unicorn | none | ud | 12.3 | 0 | 0 | 0 | ok |
| unicorn | none | xgboost | 9.8 | 0 | 0 | 0 | ok |
| unicorn | watch-cb | aha-mont64 | 1.3 | 3 | 3 | 0 | ok |
| unicorn | watch-cb | crc32 | 34.4 | 175275 | 175275 | 0 | ok |
| unicorn | watch-cb | depthconv | 12.0 | 0 | 0 | 0 | ok |
| unicorn | watch-cb | edn | 23.1 | 1066 | 1066 | 0 | ok |
| unicorn | watch-cb | huffbench | 22.0 | 768 | 768 | 0 | ok |
| unicorn | watch-cb | matmult-int | 40.5 | 3360 | 3360 | 0 | ok |
| unicorn | watch-cb | md5sum | 15.3 | 536 | 536 | 0 | ok |
| unicorn | watch-cb | nettle-aes | 20.6 | 0 | 0 | 0 | ok |
| unicorn | watch-cb | nettle-sha256 | 26.6 | 0 | 0 | 0 | ok |
| unicorn | watch-cb | nsichneu | 63.1 | 2466 | 2466 | 0 | ok |
| unicorn | watch-cb | picojpeg | 31.6 | 402 | 402 | 0 | ok |
| unicorn | watch-cb | qrduino | 18.3 | 78 | 78 | 0 | ok |
| unicorn | watch-cb | sglib-combined | 28.6 | 768 | 768 | 0 | ok |
| unicorn | watch-cb | statemate | 69.6 | 116585 | 116585 | 0 | ok |
| unicorn | watch-cb | tarfind | 26.5 | 1739 | 1739 | 0 | ok |
| unicorn | watch-cb | ud | 19.7 | 1786 | 1786 | 0 | ok |
| unicorn | watch-cb | xgboost | 34.4 | 0 | 0 | 0 | ok |
| unicorn | watch-ir | aha-mont64 | 1.3 | 3 | 3 | 0 | ok |
| unicorn | watch-ir | crc32 | 34.3 | 175275 | 175275 | 0 | ok |
| unicorn | watch-ir | depthconv | 12.5 | 0 | 0 | 0 | ok |
| unicorn | watch-ir | edn | 23.6 | 1066 | 1066 | 0 | ok |
| unicorn | watch-ir | huffbench | 22.4 | 768 | 768 | 0 | ok |
| unicorn | watch-ir | matmult-int | 41.2 | 3360 | 3360 | 0 | ok |
| unicorn | watch-ir | md5sum | 14.8 | 536 | 536 | 0 | ok |
| unicorn | watch-ir | nettle-aes | 20.2 | 0 | 0 | 0 | ok |
| unicorn | watch-ir | nettle-sha256 | 26.9 | 0 | 0 | 0 | ok |
| unicorn | watch-ir | nsichneu | 62.4 | 2466 | 2466 | 0 | ok |
| unicorn | watch-ir | picojpeg | 31.2 | 402 | 402 | 0 | ok |
| unicorn | watch-ir | qrduino | 18.8 | 78 | 78 | 0 | ok |
| unicorn | watch-ir | sglib-combined | 28.2 | 768 | 768 | 0 | ok |
| unicorn | watch-ir | statemate | 68.2 | 116585 | 116585 | 0 | ok |
| unicorn | watch-ir | tarfind | 26.2 | 1739 | 1739 | 0 | ok |
| unicorn | watch-ir | ud | 20.1 | 1786 | 1786 | 0 | ok |
| unicorn | watch-ir | xgboost | 35.0 | 0 | 0 | 0 | ok |

### Native (ns per benchmark iteration, min of 30)

| image | plain | block | edge | cmp | watch (hits) |
|---|---:|---:|---:|---:|---:|
| aha-mont64 | 261,182 | 563,577 | 756,491 | 439,971 | 279,889 (0) |
| crc32 | 336,481 | 479,189 | 682,689 | 337,279 | 315,447 (0) |
| depthconv | 167,347 | 320,238 | 471,740 | 330,201 | 167,541 (0) |
| edn | 213,654 | 379,840 | 610,069 | 424,704 | 215,178 (0) |
| huffbench | 167,633 | 888,489 | 1,118,218 | 549,631 | 172,931 (0) |
| matmult-int | 146,975 | 380,180 | 525,975 | 336,234 | 1,057,329 (98280) |
| md5sum | 136,948 | 411,760 | 562,512 | 332,467 | 137,592 (0) |
| nettle-aes | 148,544 | 198,798 | 254,188 | 206,846 | 148,334 (0) |
| nettle-sha256 | 221,044 | 282,018 | 308,377 | 269,741 | 219,735 (0) |
| nsichneu | 179,398 | 857,114 | 1,311,684 | 920,502 | 879,071 (73920) |
| picojpeg | 173,888 | 529,007 | 693,796 | 422,826 | 287,510 (10050) |
| qrduino | 437,708 | 916,693 | 1,164,817 | 772,116 | 465,850 (1950) |
| sglib-combined | 222,921 | 1,112,903 | 1,406,442 | 659,091 | 420,932 (22320) |
| statemate | 173,991 | 334,343 | 425,833 | 315,783 | 32,556,060 (3496500) |
| tarfind | 82,264 | 168,866 | 228,249 | 133,706 | 175,809 (12420) |
| ud | 376,305 | 818,638 | 1,119,180 | 684,783 | 917,631 (53550) |
| xgboost | 706,134 | 1,346,767 | 1,731,035 | 885,555 | 715,001 (0) |
