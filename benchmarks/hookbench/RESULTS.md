## Baseline: no instrumentation (ms, min of repeats)

| image | native | qcode-interp | qcode-jit | icicle | unicorn |
|---|---:|---:|---:|---:|---:|
| aha-mont64 | 0.261 |  | 60.0 | 23.7 | 1.2 |
| crc32 | 0.336 | 6034.6 | 33.0 | 10.8 | 16.2 |
| depthconv | 0.167 | 5284.1 | 37.3 | 12.0 | 4.0 |
| edn | 0.214 |  | 77.3 | 48.3 | 9.3 |
| huffbench | 0.168 |  | 108.3 | 51.1 | 14.2 |
| matmult-int | 0.147 | 5773.0 | 57.9 | 23.4 | 27.3 |
| md5sum | 0.137 |  | 66.0 | 29.5 | 10.0 |
| nettle-aes | 0.149 |  | 88.3 | 63.3 | 6.6 |
| nettle-sha256 | 0.221 |  | 209.6 | 141.8 | 12.9 |
| nsichneu | 0.179 |  | 448.4 | 214.9 | 23.7 |
| picojpeg | 0.174 |  | 214.9 | 112.2 | 22.8 |
| qrduino | 0.438 |  | 400.6 | 235.1 | 9.6 |
| sglib-combined | 0.223 |  | 178.0 | 95.1 | 16.5 |
| statemate | 0.174 |  | 59.0 | 35.2 | 50.9 |
| tarfind | 0.082 | 3048.8 | 41.4 | 20.2 | 23.6 |
| ud | 0.376 |  | 78.6 | 36.7 | 12.3 |
| xgboost | 0.706 |  | 93.4 | 23.9 | 9.8 |

## Slowdown relative to each engine's own baseline (geometric mean over images)

| instrumentation | native (compiler) | qcode-jit | icicle | unicorn | qcode-interp |
|---|---:|---:|---:|---:|---:|
| block-ir | 2.34× | 1.14× (n=17) | 1.04× (n=16) |  | 1.05× (n=4) |
| block-ram | 2.34× | 1.31× (n=17) |  |  |  |
| block-cb | 2.34× | 5.32× (n=17) | 1.08× (n=17) | 1.13× (n=17) |  |
| insn-ir |  | 1.20× (n=17) | 1.08× (n=17) |  | 1.20× (n=4) |
| insn-ram |  | 1.69× (n=17) |  |  |  |
| insn-cb |  | 13.51× (n=17) | 1.33× (n=17) | 3.71× (n=17) |  |
| edge-ir | 3.13× | 1.52× (n=17) |  |  | 1.11× (n=4) |
| watch-ir | 1.99× | 1.58× (n=17) | 1.11× (n=17) | 1.87× (n=17) | 1.07× (n=4) |
| watch-cb | 1.99× | 3.20× (n=17) | 1.11× (n=17) | 1.87× (n=17) |  |
| cmp-ir | 1.96× | 6.39× (n=17) |  |  | 4.53× (n=4) |
| cmp-cb | 1.96× | 88.12× (n=17) |  | 1.13× (n=17) |  |

## Cost per host call (ns, median over images with ≥ 10k calls)

| instrumentation | qcode-jit | icicle | unicorn |
|---|---:|---:|---:|
| block-cb | 626 (n=17) | 6 (n=17) | 3 (n=17) |
| insn-cb | 357 (n=17) | 4 (n=17) | 8 (n=17) |
| watch-ir | 1211 (n=2) | 108 (n=2) | 126 (n=2) |
| watch-cb | 763 (n=15) | 103 (n=2) | 132 (n=2) |
| cmp-cb | 601 (n=17) |  | 3 (n=16) |

## Compiled instrumentation on qcode-jit: overhead per event (ns, median over images)

| instrumentation | ns/event | events per image (median) | sites per image (median) |
|---|---:|---:|---:|
| block-ir | 8.7 | 586384 | 71 |
| block-ram | 29.6 | 586384 | 71 |
| insn-ir | 2.5 | 2797078 | 496 |
| insn-ram | 15.8 | 2797078 | 496 |

## How the qcode-jit numbers moved (oldest first)

| instrumentation | first sweep | fix1 |
|---|---:|---:|
| block-ir | 1.26× (n=17) | 1.14× (n=17) |
| block-ram | 1.41× (n=17) | 1.31× (n=17) |
| block-cb | 5.47× (n=17) | 5.32× (n=17) |
| insn-ir | 1.33× (n=17) | 1.20× (n=17) |
| insn-ram | 1.85× (n=17) | 1.69× (n=17) |
| insn-cb | 14.69× (n=17) | 13.51× (n=17) |
| edge-ir | 1.55× (n=17) | 1.52× (n=17) |
| watch-ir | 1.71× (n=17) | 1.58× (n=17) |
| watch-cb | 11.33× (n=17) | 3.20× (n=17) |
| cmp-ir | 6.91× (n=17) | 6.39× (n=17) |
| cmp-cb | 14.55× (n=1) 16 ✗ | 88.12× (n=17) |

- first sweep
- fix1: After the JIT cache fix (commit c83141b): blocks carry a revision stamp, the cache is keyed on it, and compiled code is resumed from anywhere in a block. Only the QCode JIT was re-run; the other engines are unchanged.

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
| qcode-interp | cmp-ir | crc32 | 38188.0 | 0 | 1092 | 324 | ok |
| qcode-interp | cmp-ir | depthconv | 22439.4 | 0 | 464 | 327 | ok |
| qcode-interp | cmp-ir | matmult-int | 16891.8 | 0 | 615 | 625 | ok |
| qcode-interp | cmp-ir | tarfind | 16320.1 | 0 | 302 | 578 | ok |
| qcode-interp | edge-ir | crc32 | 6146.4 | 0 | 21 | 19 | ok |
| qcode-interp | edge-ir | depthconv | 5736.4 | 0 | 34 | 31 | ok |
| qcode-interp | edge-ir | matmult-int | 7016.3 | 0 | 62 | 48 | ok |
| qcode-interp | edge-ir | tarfind | 3421.2 | 0 | 57 | 47 | ok |
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
| qcode-jit | block-cb | aha-mont64 | 614.4 | 428535 | 428535 | 0 | ok |
| qcode-jit | block-cb | crc32 | 528.3 | 350902 | 350902 | 0 | ok |
| qcode-jit | block-cb | depthconv | 435.0 | 635575 | 635575 | 0 | ok |
| qcode-jit | block-cb | edn | 335.8 | 817048 | 817048 | 0 | ok |
| qcode-jit | block-cb | huffbench | 463.3 | 797622 | 797622 | 0 | ok |
| qcode-jit | block-cb | matmult-int | 369.4 | 1200608 | 1200608 | 0 | ok |
| qcode-jit | block-cb | md5sum | 383.5 | 586384 | 586384 | 0 | ok |
| qcode-jit | block-cb | nettle-aes | 299.5 | 115772 | 115772 | 0 | ok |
| qcode-jit | block-cb | nettle-sha256 | 1082.9 | 246603 | 246603 | 0 | ok |
| qcode-jit | block-cb | nsichneu | 1011.4 | 773121 | 773121 | 0 | ok |
| qcode-jit | block-cb | picojpeg | 856.1 | 515129 | 515129 | 0 | ok |
| qcode-jit | block-cb | qrduino | 1004.2 | 567451 | 567451 | 0 | ok |
| qcode-jit | block-cb | sglib-combined | 580.9 | 819979 | 819979 | 0 | ok |
| qcode-jit | block-cb | statemate | 230.9 | 336637 | 336637 | 0 | ok |
| qcode-jit | block-cb | tarfind | 277.7 | 331253 | 331253 | 0 | ok |
| qcode-jit | block-cb | ud | 516.7 | 748654 | 748654 | 0 | ok |
| qcode-jit | block-cb | xgboost | 700.0 | 1317252 | 1317252 | 0 | ok |
| qcode-jit | block-ir | aha-mont64 | 64.3 | 0 | 428535 | 43 | ok |
| qcode-jit | block-ir | crc32 | 34.4 | 0 | 350902 | 19 | ok |
| qcode-jit | block-ir | depthconv | 39.9 | 0 | 635575 | 31 | ok |
| qcode-jit | block-ir | edn | 81.9 | 0 | 817048 | 79 | ok |
| qcode-jit | block-ir | huffbench | 114.5 | 0 | 797622 | 149 | ok |
| qcode-jit | block-ir | matmult-int | 62.6 | 0 | 1200608 | 48 | ok |
| qcode-jit | block-ir | md5sum | 70.1 | 0 | 586384 | 76 | ok |
| qcode-jit | block-ir | nettle-aes | 100.9 | 0 | 115772 | 71 | ok |
| qcode-jit | block-ir | nettle-sha256 | 644.1 | 0 | 246603 | 70 | ok |
| qcode-jit | block-ir | nsichneu | 458.6 | 0 | 773121 | 652 | ok |
| qcode-jit | block-ir | picojpeg | 227.7 | 0 | 515129 | 323 | ok |
| qcode-jit | block-ir | qrduino | 417.7 | 0 | 567451 | 510 | ok |
| qcode-jit | block-ir | sglib-combined | 186.5 | 0 | 819979 | 279 | ok |
| qcode-jit | block-ir | statemate | 62.5 | 0 | 336637 | 75 | ok |
| qcode-jit | block-ir | tarfind | 43.5 | 0 | 331253 | 47 | ok |
| qcode-jit | block-ir | ud | 85.1 | 0 | 748654 | 66 | ok |
| qcode-jit | block-ir | xgboost | 104.5 | 0 | 1317252 | 36 | ok |
| qcode-jit | block-ram | aha-mont64 | 70.8 | 0 | 428535 | 43 | ok |
| qcode-jit | block-ram | crc32 | 35.3 | 0 | 350902 | 19 | ok |
| qcode-jit | block-ram | depthconv | 43.9 | 0 | 635575 | 31 | ok |
| qcode-jit | block-ram | edn | 92.6 | 0 | 817048 | 79 | ok |
| qcode-jit | block-ram | huffbench | 143.9 | 0 | 797622 | 149 | ok |
| qcode-jit | block-ram | matmult-int | 70.6 | 0 | 1200608 | 48 | ok |
| qcode-jit | block-ram | md5sum | 82.1 | 0 | 586384 | 76 | ok |
| qcode-jit | block-ram | nettle-aes | 113.2 | 0 | 115772 | 71 | ok |
| qcode-jit | block-ram | nettle-sha256 | 660.8 | 0 | 246603 | 70 | ok |
| qcode-jit | block-ram | nsichneu | 586.2 | 0 | 773121 | 652 | ok |
| qcode-jit | block-ram | picojpeg | 295.4 | 0 | 515129 | 323 | ok |
| qcode-jit | block-ram | qrduino | 511.2 | 0 | 567451 | 510 | ok |
| qcode-jit | block-ram | sglib-combined | 237.4 | 0 | 819979 | 279 | ok |
| qcode-jit | block-ram | statemate | 74.6 | 0 | 336637 | 75 | ok |
| qcode-jit | block-ram | tarfind | 51.2 | 0 | 331253 | 47 | ok |
| qcode-jit | block-ram | ud | 94.8 | 0 | 748654 | 66 | ok |
| qcode-jit | block-ram | xgboost | 105.6 | 0 | 1317252 | 36 | ok |
| qcode-jit | cmp-cb | aha-mont64 | 18273.0 | 31790885 | 31790885 | 2450 | ok |
| qcode-jit | cmp-cb | crc32 | 16867.2 | 28021828 | 28021828 | 324 | ok |
| qcode-jit | cmp-cb | depthconv | 8470.3 | 15024592 | 15024592 | 327 | ok |
| qcode-jit | cmp-cb | edn | 6720.1 | 11964664 | 11964664 | 2241 | ok |
| qcode-jit | cmp-cb | huffbench | 4692.8 | 8078215 | 8078215 | 1807 | ok |
| qcode-jit | cmp-cb | matmult-int | 6721.3 | 10207847 | 10207847 | 625 | ok |
| qcode-jit | cmp-cb | md5sum | 4606.9 | 7677399 | 7677399 | 806 | ok |
| qcode-jit | cmp-cb | nettle-aes | 19265.4 | 27197666 | 27197666 | 9115 | ok |
| qcode-jit | cmp-cb | nettle-sha256 | 67467.6 | 38203707 | 38203707 | 23640 | ok |
| qcode-jit | cmp-cb | nsichneu | 7062.1 | 6343984 | 6343984 | 5301 | ok |
| qcode-jit | cmp-cb | picojpeg | 17601.0 | 23848500 | 23848500 | 8189 | ok |
| qcode-jit | cmp-cb | qrduino | 15962.3 | 18201433 | 18201433 | 18417 | ok |
| qcode-jit | cmp-cb | sglib-combined | 4389.4 | 6652028 | 6652028 | 2777 | ok |
| qcode-jit | cmp-cb | statemate | 1405.2 | 2336218 | 2336218 | 443 | ok |
| qcode-jit | cmp-cb | tarfind | 7009.7 | 11600174 | 11600174 | 578 | ok |
| qcode-jit | cmp-cb | ud | 5582.1 | 7773549 | 7773549 | 1050 | ok |
| qcode-jit | cmp-cb | xgboost | 3958.0 | 6559787 | 6559787 | 455 | ok |
| qcode-jit | cmp-ir | aha-mont64 | 666.7 | 0 | 1829 | 2450 | ok |
| qcode-jit | cmp-ir | crc32 | 306.5 | 0 | 1092 | 324 | ok |
| qcode-jit | cmp-ir | depthconv | 241.8 | 0 | 464 | 327 | ok |
| qcode-jit | cmp-ir | edn | 560.1 | 0 | 248 | 2241 | ok |
| qcode-jit | cmp-ir | huffbench | 469.6 | 0 | 903 | 1807 | ok |
| qcode-jit | cmp-ir | matmult-int | 200.6 | 0 | 615 | 625 | ok |
| qcode-jit | cmp-ir | md5sum | 269.2 | 0 | 1495 | 806 | ok |
| qcode-jit | cmp-ir | nettle-aes | 2236.0 | 0 | 226 | 9115 | ok |
| qcode-jit | cmp-ir | nettle-sha256 | 10688.2 | 0 | 315 | 23640 | ok |
| qcode-jit | cmp-ir | nsichneu | 1789.8 | 0 | 3376 | 5301 | ok |
| qcode-jit | cmp-ir | picojpeg | 1943.1 | 0 | 1588 | 8189 | ok |
| qcode-jit | cmp-ir | qrduino | 3645.9 | 0 | 2905 | 18417 | ok |
| qcode-jit | cmp-ir | sglib-combined | 688.0 | 0 | 124 | 2777 | ok |
| qcode-jit | cmp-ir | statemate | 154.2 | 0 | 1498 | 443 | ok |
| qcode-jit | cmp-ir | tarfind | 234.8 | 0 | 302 | 578 | ok |
| qcode-jit | cmp-ir | ud | 265.0 | 0 | 3437 | 1050 | ok |
| qcode-jit | cmp-ir | xgboost | 191.5 | 0 | 2091 | 455 | ok |
| qcode-jit | edge-ir | aha-mont64 | 77.6 | 0 | 58 | 43 | ok |
| qcode-jit | edge-ir | crc32 | 39.5 | 0 | 21 | 19 | ok |
| qcode-jit | edge-ir | depthconv | 50.5 | 0 | 34 | 31 | ok |
| qcode-jit | edge-ir | edn | 109.9 | 0 | 106 | 79 | ok |
| qcode-jit | edge-ir | huffbench | 168.3 | 0 | 208 | 149 | ok |
| qcode-jit | edge-ir | matmult-int | 80.8 | 0 | 62 | 48 | ok |
| qcode-jit | edge-ir | md5sum | 95.8 | 0 | 89 | 76 | ok |
| qcode-jit | edge-ir | nettle-aes | 125.9 | 0 | 94 | 71 | ok |
| qcode-jit | edge-ir | nettle-sha256 | 679.2 | 0 | 91 | 70 | ok |
| qcode-jit | edge-ir | nsichneu | 721.3 | 0 | 650 | 652 | ok |
| qcode-jit | edge-ir | picojpeg | 360.1 | 0 | 455 | 323 | ok |
| qcode-jit | edge-ir | qrduino | 605.6 | 0 | 746 | 510 | ok |
| qcode-jit | edge-ir | sglib-combined | 292.3 | 0 | 396 | 279 | ok |
| qcode-jit | edge-ir | statemate | 89.3 | 0 | 82 | 75 | ok |
| qcode-jit | edge-ir | tarfind | 60.5 | 0 | 57 | 47 | ok |
| qcode-jit | edge-ir | ud | 108.9 | 0 | 91 | 66 | ok |
| qcode-jit | edge-ir | xgboost | 132.6 | 0 | 40 | 36 | ok |
| qcode-jit | insn-cb | aha-mont64 | 1210.5 | 2428485 | 2428485 | 0 | ok |
| qcode-jit | insn-cb | crc32 | 1019.4 | 2278822 | 2278822 | 0 | ok |
| qcode-jit | insn-cb | depthconv | 1024.6 | 2998976 | 2998976 | 0 | ok |
| qcode-jit | insn-cb | edn | 1234.0 | 4109614 | 4109614 | 0 | ok |
| qcode-jit | insn-cb | huffbench | 1045.3 | 2795647 | 2795647 | 0 | ok |
| qcode-jit | insn-cb | matmult-int | 1043.8 | 4072595 | 4072595 | 0 | ok |
| qcode-jit | insn-cb | md5sum | 893.7 | 2398400 | 2398400 | 0 | ok |
| qcode-jit | insn-cb | nettle-aes | 1178.8 | 2621639 | 2621639 | 0 | ok |
| qcode-jit | insn-cb | nettle-sha256 | 4846.3 | 4475405 | 4475405 | 0 | ok |
| qcode-jit | insn-cb | nsichneu | 2050.5 | 2167728 | 2167728 | 0 | ok |
| qcode-jit | insn-cb | picojpeg | 1880.5 | 3298380 | 3298380 | 0 | ok |
| qcode-jit | insn-cb | qrduino | 2443.5 | 3670529 | 3670529 | 0 | ok |
| qcode-jit | insn-cb | sglib-combined | 1258.8 | 2797078 | 2797078 | 0 | ok |
| qcode-jit | insn-cb | statemate | 798.4 | 2235589 | 2235589 | 0 | ok |
| qcode-jit | insn-cb | tarfind | 694.4 | 1942280 | 1942280 | 0 | ok |
| qcode-jit | insn-cb | ud | 1147.5 | 2994568 | 2994568 | 0 | ok |
| qcode-jit | insn-cb | xgboost | 1366.2 | 3936130 | 3936130 | 0 | ok |
| qcode-jit | insn-ir | aha-mont64 | 68.3 | 0 | 2428485 | 496 | ok |
| qcode-jit | insn-ir | crc32 | 33.9 | 0 | 2278822 | 108 | ok |
| qcode-jit | insn-ir | depthconv | 39.1 | 0 | 2998976 | 104 | ok |
| qcode-jit | insn-ir | edn | 85.3 | 0 | 4109614 | 692 | ok |
| qcode-jit | insn-ir | huffbench | 117.4 | 0 | 2795647 | 734 | ok |
| qcode-jit | insn-ir | matmult-int | 63.8 | 0 | 4072595 | 283 | ok |
| qcode-jit | insn-ir | md5sum | 69.6 | 0 | 2398400 | 447 | ok |
| qcode-jit | insn-ir | nettle-aes | 116.6 | 0 | 2621639 | 1068 | ok |
| qcode-jit | insn-ir | nettle-sha256 | 1072.2 | 0 | 4475405 | 2845 | ok |
| qcode-jit | insn-ir | nsichneu | 462.6 | 0 | 2167728 | 1858 | ok |
| qcode-jit | insn-ir | picojpeg | 239.4 | 0 | 3298380 | 1928 | ok |
| qcode-jit | insn-ir | qrduino | 434.8 | 0 | 3670529 | 3633 | ok |
| qcode-jit | insn-ir | sglib-combined | 187.8 | 0 | 2797078 | 1165 | ok |
| qcode-jit | insn-ir | statemate | 64.6 | 0 | 2235589 | 468 | ok |
| qcode-jit | insn-ir | tarfind | 44.3 | 0 | 1942280 | 234 | ok |
| qcode-jit | insn-ir | ud | 85.0 | 0 | 2994568 | 436 | ok |
| qcode-jit | insn-ir | xgboost | 103.0 | 0 | 3936130 | 187 | ok |
| qcode-jit | insn-ram | aha-mont64 | 93.3 | 0 | 2428485 | 496 | ok |
| qcode-jit | insn-ram | crc32 | 42.4 | 0 | 2278822 | 108 | ok |
| qcode-jit | insn-ram | depthconv | 51.4 | 0 | 2998976 | 104 | ok |
| qcode-jit | insn-ram | edn | 126.5 | 0 | 4109614 | 692 | ok |
| qcode-jit | insn-ram | huffbench | 177.2 | 0 | 2795647 | 734 | ok |
| qcode-jit | insn-ram | matmult-int | 83.1 | 0 | 4072595 | 283 | ok |
| qcode-jit | insn-ram | md5sum | 103.9 | 0 | 2398400 | 447 | ok |
| qcode-jit | insn-ram | nettle-aes | 176.9 | 0 | 2621639 | 1068 | ok |
| qcode-jit | insn-ram | nettle-sha256 | 1230.0 | 0 | 4475405 | 2845 | ok |
| qcode-jit | insn-ram | nsichneu | 690.1 | 0 | 2167728 | 1858 | ok |
| qcode-jit | insn-ram | picojpeg | 394.7 | 0 | 3298380 | 1928 | ok |
| qcode-jit | insn-ram | qrduino | 696.9 | 0 | 3670529 | 3633 | ok |
| qcode-jit | insn-ram | sglib-combined | 291.2 | 0 | 2797078 | 1165 | ok |
| qcode-jit | insn-ram | statemate | 100.9 | 0 | 2235589 | 468 | ok |
| qcode-jit | insn-ram | tarfind | 63.7 | 0 | 1942280 | 234 | ok |
| qcode-jit | insn-ram | ud | 112.8 | 0 | 2994568 | 436 | ok |
| qcode-jit | insn-ram | xgboost | 114.4 | 0 | 3936130 | 187 | ok |
| qcode-jit | none | aha-mont64 | 60.0 | 0 | 0 | 0 | ok |
| qcode-jit | none | crc32 | 33.0 | 0 | 0 | 0 | ok |
| qcode-jit | none | depthconv | 37.3 | 0 | 0 | 0 | ok |
| qcode-jit | none | edn | 77.3 | 0 | 0 | 0 | ok |
| qcode-jit | none | huffbench | 108.3 | 0 | 0 | 0 | ok |
| qcode-jit | none | matmult-int | 57.9 | 0 | 0 | 0 | ok |
| qcode-jit | none | md5sum | 66.0 | 0 | 0 | 0 | ok |
| qcode-jit | none | nettle-aes | 88.3 | 0 | 0 | 0 | ok |
| qcode-jit | none | nettle-sha256 | 209.6 | 0 | 0 | 0 | ok |
| qcode-jit | none | nsichneu | 448.4 | 0 | 0 | 0 | ok |
| qcode-jit | none | picojpeg | 214.9 | 0 | 0 | 0 | ok |
| qcode-jit | none | qrduino | 400.6 | 0 | 0 | 0 | ok |
| qcode-jit | none | sglib-combined | 178.0 | 0 | 0 | 0 | ok |
| qcode-jit | none | statemate | 59.0 | 0 | 0 | 0 | ok |
| qcode-jit | none | tarfind | 41.4 | 0 | 0 | 0 | ok |
| qcode-jit | none | ud | 78.6 | 0 | 0 | 0 | ok |
| qcode-jit | none | xgboost | 93.4 | 0 | 0 | 0 | ok |
| qcode-jit | watch-cb | aha-mont64 | 78.4 | 5223 | 3 | 45 | ok |
| qcode-jit | watch-cb | crc32 | 200.1 | 350569 | 175275 | 28 | ok |
| qcode-jit | watch-cb | depthconv | 67.0 | 54128 | 0 | 14 | ok |
| qcode-jit | watch-cb | edn | 221.6 | 173163 | 1065 | 120 | ok |
| qcode-jit | watch-cb | huffbench | 299.9 | 348279 | 768 | 99 | ok |
| qcode-jit | watch-cb | matmult-int | 370.6 | 596259 | 3360 | 41 | ok |
| qcode-jit | watch-cb | md5sum | 180.7 | 213211 | 536 | 86 | ok |
| qcode-jit | watch-cb | nettle-aes | 267.5 | 79169 | 0 | 128 | ok |
| qcode-jit | watch-cb | nettle-sha256 | 1319.1 | 258433 | 0 | 171 | ok |
| qcode-jit | watch-cb | nsichneu | 453.3 | 3719 | 2466 | 26 | ok |
| qcode-jit | watch-cb | picojpeg | 715.5 | 439371 | 402 | 367 | ok |
| qcode-jit | watch-cb | qrduino | 568.5 | 93934 | 78 | 352 | ok |
| qcode-jit | watch-cb | sglib-combined | 413.4 | 308596 | 768 | 202 | ok |
| qcode-jit | watch-cb | statemate | 649.8 | 1022639 | 116584 | 176 | ok |
| qcode-jit | watch-cb | tarfind | 540.3 | 536809 | 1739 | 50 | ok |
| qcode-jit | watch-cb | ud | 291.2 | 207216 | 1786 | 69 | ok |
| qcode-jit | watch-cb | xgboost | 159.6 | 104207 | 0 | 30 | ok |
| qcode-jit | watch-ir | aha-mont64 | 67.1 | 3 | 3 | 0 | ok |
| qcode-jit | watch-ir | crc32 | 112.5 | 175275 | 175275 | 0 | ok |
| qcode-jit | watch-ir | depthconv | 44.8 | 0 | 0 | 0 | ok |
| qcode-jit | watch-ir | edn | 113.8 | 1066 | 1066 | 0 | ok |
| qcode-jit | watch-ir | huffbench | 156.0 | 768 | 768 | 0 | ok |
| qcode-jit | watch-ir | matmult-int | 120.5 | 3360 | 3360 | 0 | ok |
| qcode-jit | watch-ir | md5sum | 97.4 | 536 | 536 | 0 | ok |
| qcode-jit | watch-ir | nettle-aes | 124.4 | 0 | 0 | 0 | ok |
| qcode-jit | watch-ir | nettle-sha256 | 267.1 | 0 | 0 | 0 | ok |
| qcode-jit | watch-ir | nsichneu | 456.2 | 2466 | 2466 | 0 | ok |
| qcode-jit | watch-ir | picojpeg | 307.7 | 402 | 402 | 0 | ok |
| qcode-jit | watch-ir | qrduino | 454.1 | 78 | 78 | 0 | ok |
| qcode-jit | watch-ir | sglib-combined | 238.3 | 768 | 768 | 0 | ok |
| qcode-jit | watch-ir | statemate | 288.4 | 116585 | 116585 | 0 | ok |
| qcode-jit | watch-ir | tarfind | 102.3 | 1739 | 1739 | 0 | ok |
| qcode-jit | watch-ir | ud | 113.0 | 1786 | 1786 | 0 | ok |
| qcode-jit | watch-ir | xgboost | 107.0 | 0 | 0 | 0 | ok |
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
