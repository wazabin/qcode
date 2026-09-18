## Baseline: no instrumentation (ms, min of repeats)

| image | native | qcode-interp | qcode-jit | icicle | unicorn |
|---|---:|---:|---:|---:|---:|
| aha-mont64 | 0.261 |  | 59.2 | 23.7 | 1.2 |
| crc32 | 0.336 | 6034.6 | 32.2 | 10.8 | 16.2 |
| depthconv | 0.167 | 5284.1 | 37.1 | 12.0 | 4.0 |
| edn | 0.214 |  | 76.0 | 48.3 | 9.3 |
| huffbench | 0.168 |  | 109.7 | 51.1 | 14.2 |
| matmult-int | 0.147 | 5773.0 | 58.6 | 23.4 | 27.3 |
| md5sum | 0.137 |  | 65.4 | 29.5 | 10.0 |
| nettle-aes | 0.149 |  | 89.1 | 63.3 | 6.6 |
| nettle-sha256 | 0.221 |  | 216.7 | 141.8 | 12.9 |
| nsichneu | 0.179 |  | 453.1 | 214.9 | 23.7 |
| picojpeg | 0.174 |  | 219.1 | 112.2 | 22.8 |
| qrduino | 0.438 |  | 420.7 | 235.1 | 9.6 |
| sglib-combined | 0.223 |  | 183.0 | 95.1 | 16.5 |
| statemate | 0.174 |  | 59.2 | 35.2 | 50.9 |
| tarfind | 0.082 | 3048.8 | 42.4 | 20.2 | 23.6 |
| ud | 0.376 |  | 79.3 | 36.7 | 12.3 |
| xgboost | 0.706 |  | 95.7 | 23.9 | 9.8 |

## Slowdown relative to each engine's own baseline (geometric mean over images)

| instrumentation | native (compiler) | qcode-jit | icicle | unicorn | qcode-interp |
|---|---:|---:|---:|---:|---:|
| block-ir | 2.34× | 1.31× (n=17) | 1.04× (n=16) |  | 1.05× (n=4) |
| block-cb | 2.34× | 5.47× (n=17) | 1.08× (n=17) | 1.13× (n=17) |  |
| insn-ir |  | 1.77× (n=17) | 1.08× (n=17) |  | 1.20× (n=4) |
| insn-cb |  | 14.69× (n=17) | 1.33× (n=17) | 3.71× (n=17) |  |
| edge-ir | 3.13× | 1.55× (n=17) |  |  | 1.11× (n=4) |
| watch-ir | 1.99× | 1.71× (n=17) | 1.11× (n=17) | 1.87× (n=17) | 1.07× (n=4) |
| watch-cb | 1.99× | 11.33× (n=17) | 1.11× (n=17) | 1.87× (n=17) |  |
| cmp-ir | 1.96× | 6.91× (n=17) |  |  | 4.53× (n=4) |
| cmp-cb | 1.96× | 14.55× (n=1) |  | 1.13× (n=17) |  |

## Cost per host call (ns, median over images with ≥ 10k calls)

| instrumentation | qcode-jit | icicle | unicorn |
|---|---:|---:|---:|
| block-cb | 646 (n=17) | 6 (n=17) | 3 (n=17) |
| insn-cb | 384 (n=17) | 4 (n=17) | 8 (n=17) |
| watch-ir | 1358 (n=2) | 108 (n=2) | 126 (n=2) |
| watch-cb | 3948 (n=15) | 103 (n=2) | 132 (n=2) |
| cmp-cb | 968 (n=1) |  | 3 (n=16) |

## Compiled instrumentation on qcode-jit: overhead per event (ns, median over images)

| instrumentation | ns/event | events per image (median) | sites per image (median) |
|---|---:|---:|---:|
| block-ir | 30.7 | 586384 | 71 |
| insn-ir | 19.0 | 2797078 | 496 |

## Runs that did not verify

- qcode-jit cmp-cb aha-mont64: Error("value 0 is too large to represent")
- qcode-jit cmp-cb crc32: Error("value 0 is too large to represent")
- qcode-jit cmp-cb depthconv: Error("value 0 is too large to represent")
- qcode-jit cmp-cb edn: Error("value 0 is too large to represent")
- qcode-jit cmp-cb huffbench: Error("value 0 is too large to represent")
- qcode-jit cmp-cb matmult-int: Error("value 0 is too large to represent")
- qcode-jit cmp-cb md5sum: Error("value 0 is too large to represent")
- qcode-jit cmp-cb nettle-aes: Error("value 0 is too large to represent")
- qcode-jit cmp-cb nettle-sha256: Error("value 0 is too large to represent")
- qcode-jit cmp-cb picojpeg: Error("value 0 is too large to represent")
- qcode-jit cmp-cb qrduino: Error("value 0 is too large to represent")
- qcode-jit cmp-cb sglib-combined: Error("value 0 is too large to represent")
- qcode-jit cmp-cb statemate: Error("value 0 is too large to represent")
- qcode-jit cmp-cb tarfind: Error("value 0 is too large to represent")
- qcode-jit cmp-cb ud: Error("value 0 is too large to represent")
- qcode-jit cmp-cb xgboost: Error("value 0 is too large to represent")
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
| qcode-jit | block-cb | aha-mont64 | 639.7 | 428535 | 428535 | 0 | ok |
| qcode-jit | block-cb | crc32 | 553.9 | 350902 | 350902 | 0 | ok |
| qcode-jit | block-cb | depthconv | 435.8 | 635575 | 635575 | 0 | ok |
| qcode-jit | block-cb | edn | 349.7 | 817048 | 817048 | 0 | ok |
| qcode-jit | block-cb | huffbench | 468.3 | 797622 | 797622 | 0 | ok |
| qcode-jit | block-cb | matmult-int | 381.2 | 1200608 | 1200608 | 0 | ok |
| qcode-jit | block-cb | md5sum | 393.1 | 586384 | 586384 | 0 | ok |
| qcode-jit | block-cb | nettle-aes | 315.1 | 115772 | 115772 | 0 | ok |
| qcode-jit | block-cb | nettle-sha256 | 1137.7 | 246603 | 246603 | 0 | ok |
| qcode-jit | block-cb | nsichneu | 1058.4 | 773121 | 773121 | 0 | ok |
| qcode-jit | block-cb | picojpeg | 885.2 | 515129 | 515129 | 0 | ok |
| qcode-jit | block-cb | qrduino | 1065.2 | 567451 | 567451 | 0 | ok |
| qcode-jit | block-cb | sglib-combined | 589.2 | 819979 | 819979 | 0 | ok |
| qcode-jit | block-cb | statemate | 236.1 | 336637 | 336637 | 0 | ok |
| qcode-jit | block-cb | tarfind | 288.9 | 331253 | 331253 | 0 | ok |
| qcode-jit | block-cb | ud | 563.2 | 748654 | 748654 | 0 | ok |
| qcode-jit | block-cb | xgboost | 738.3 | 1317252 | 1317252 | 0 | ok |
| qcode-jit | block-ir | aha-mont64 | 72.1 | 0 | 428535 | 43 | ok |
| qcode-jit | block-ir | crc32 | 37.2 | 0 | 350902 | 19 | ok |
| qcode-jit | block-ir | depthconv | 44.3 | 0 | 635575 | 31 | ok |
| qcode-jit | block-ir | edn | 93.4 | 0 | 817048 | 79 | ok |
| qcode-jit | block-ir | huffbench | 141.9 | 0 | 797622 | 149 | ok |
| qcode-jit | block-ir | matmult-int | 70.3 | 0 | 1200608 | 48 | ok |
| qcode-jit | block-ir | md5sum | 82.3 | 0 | 586384 | 76 | ok |
| qcode-jit | block-ir | nettle-aes | 112.3 | 0 | 115772 | 71 | ok |
| qcode-jit | block-ir | nettle-sha256 | 660.0 | 0 | 246603 | 70 | ok |
| qcode-jit | block-ir | nsichneu | 597.1 | 0 | 773121 | 652 | ok |
| qcode-jit | block-ir | picojpeg | 297.6 | 0 | 515129 | 323 | ok |
| qcode-jit | block-ir | qrduino | 524.8 | 0 | 567451 | 510 | ok |
| qcode-jit | block-ir | sglib-combined | 239.7 | 0 | 819979 | 279 | ok |
| qcode-jit | block-ir | statemate | 75.5 | 0 | 336637 | 75 | ok |
| qcode-jit | block-ir | tarfind | 52.5 | 0 | 331253 | 47 | ok |
| qcode-jit | block-ir | ud | 94.5 | 0 | 748654 | 66 | ok |
| qcode-jit | block-ir | xgboost | 109.5 | 0 | 1317252 | 36 | ok |
| qcode-jit | cmp-cb | aha-mont64 | 307.6 | 67261 | 67261 | 2241 | Error("value 0 is too large to represent |
| qcode-jit | cmp-cb | crc32 | 136.9 | 163904 | 163904 | 303 | Error("value 0 is too large to represent |
| qcode-jit | cmp-cb | depthconv | 53.2 | 9189 | 9189 | 277 | Error("value 0 is too large to represent |
| qcode-jit | cmp-cb | edn | 534.4 | 145391 | 145391 | 2123 | Error("value 0 is too large to represent |
| qcode-jit | cmp-cb | huffbench | 178.3 | 108993 | 108993 | 807 | Error("value 0 is too large to represent |
| qcode-jit | cmp-cb | matmult-int | 319.7 | 325915 | 325915 | 556 | Error("value 0 is too large to represent |
| qcode-jit | cmp-cb | md5sum | 172.7 | 114627 | 114627 | 764 | Error("value 0 is too large to represent |
| qcode-jit | cmp-cb | nettle-aes | 167.2 | 2979 | 2979 | 1384 | Error("value 0 is too large to represent |
| qcode-jit | cmp-cb | nettle-sha256 | 46253.9 | 67885 | 67885 | 23480 | Error("value 0 is too large to represent |
| qcode-jit | cmp-cb | nsichneu | 6590.9 | 6343984 | 6343984 | 5301 | ok |
| qcode-jit | cmp-cb | picojpeg | 251.0 | 36671 | 36671 | 1536 | Error("value 0 is too large to represent |
| qcode-jit | cmp-cb | qrduino | 461.4 | 12975 | 12975 | 2947 | Error("value 0 is too large to represent |
| qcode-jit | cmp-cb | sglib-combined | 500.4 | 207057 | 207057 | 2308 | Error("value 0 is too large to represent |
| qcode-jit | cmp-cb | statemate | 46.9 | 732 | 732 | 278 | Error("value 0 is too large to represent |
| qcode-jit | cmp-cb | tarfind | 370.8 | 487219 | 487219 | 563 | Error("value 0 is too large to represent |
| qcode-jit | cmp-cb | ud | 29.2 | 1509 | 1509 | 420 | Error("value 0 is too large to represent |
| qcode-jit | cmp-cb | xgboost | 64.5 | 50966 | 50966 | 338 | Error("value 0 is too large to represent |
| qcode-jit | cmp-ir | aha-mont64 | 708.3 | 0 | 1829 | 2450 | ok |
| qcode-jit | cmp-ir | crc32 | 318.1 | 0 | 1092 | 324 | ok |
| qcode-jit | cmp-ir | depthconv | 254.9 | 0 | 464 | 327 | ok |
| qcode-jit | cmp-ir | edn | 626.9 | 0 | 248 | 2241 | ok |
| qcode-jit | cmp-ir | huffbench | 519.2 | 0 | 903 | 1807 | ok |
| qcode-jit | cmp-ir | matmult-int | 215.1 | 0 | 615 | 625 | ok |
| qcode-jit | cmp-ir | md5sum | 301.4 | 0 | 1495 | 806 | ok |
| qcode-jit | cmp-ir | nettle-aes | 2542.8 | 0 | 226 | 9115 | ok |
| qcode-jit | cmp-ir | nettle-sha256 | 13836.8 | 0 | 315 | 23640 | ok |
| qcode-jit | cmp-ir | nsichneu | 1879.8 | 0 | 3376 | 5301 | ok |
| qcode-jit | cmp-ir | picojpeg | 2150.3 | 0 | 1588 | 8189 | ok |
| qcode-jit | cmp-ir | qrduino | 4071.0 | 0 | 2905 | 18417 | ok |
| qcode-jit | cmp-ir | sglib-combined | 730.9 | 0 | 124 | 2777 | ok |
| qcode-jit | cmp-ir | statemate | 161.2 | 0 | 1498 | 443 | ok |
| qcode-jit | cmp-ir | tarfind | 248.5 | 0 | 302 | 578 | ok |
| qcode-jit | cmp-ir | ud | 282.2 | 0 | 3437 | 1050 | ok |
| qcode-jit | cmp-ir | xgboost | 202.6 | 0 | 2091 | 455 | ok |
| qcode-jit | edge-ir | aha-mont64 | 79.0 | 0 | 58 | 43 | ok |
| qcode-jit | edge-ir | crc32 | 38.7 | 0 | 21 | 19 | ok |
| qcode-jit | edge-ir | depthconv | 49.5 | 0 | 34 | 31 | ok |
| qcode-jit | edge-ir | edn | 110.4 | 0 | 106 | 79 | ok |
| qcode-jit | edge-ir | huffbench | 170.2 | 0 | 208 | 149 | ok |
| qcode-jit | edge-ir | matmult-int | 81.9 | 0 | 62 | 48 | ok |
| qcode-jit | edge-ir | md5sum | 96.7 | 0 | 89 | 76 | ok |
| qcode-jit | edge-ir | nettle-aes | 131.3 | 0 | 94 | 71 | ok |
| qcode-jit | edge-ir | nettle-sha256 | 737.8 | 0 | 91 | 70 | ok |
| qcode-jit | edge-ir | nsichneu | 727.4 | 0 | 650 | 652 | ok |
| qcode-jit | edge-ir | picojpeg | 371.0 | 0 | 455 | 323 | ok |
| qcode-jit | edge-ir | qrduino | 624.9 | 0 | 746 | 510 | ok |
| qcode-jit | edge-ir | sglib-combined | 301.0 | 0 | 396 | 279 | ok |
| qcode-jit | edge-ir | statemate | 94.0 | 0 | 82 | 75 | ok |
| qcode-jit | edge-ir | tarfind | 64.1 | 0 | 57 | 47 | ok |
| qcode-jit | edge-ir | ud | 118.8 | 0 | 91 | 66 | ok |
| qcode-jit | edge-ir | xgboost | 141.6 | 0 | 40 | 36 | ok |
| qcode-jit | insn-cb | aha-mont64 | 1272.2 | 2428485 | 2428485 | 0 | ok |
| qcode-jit | insn-cb | crc32 | 1147.7 | 2278822 | 2278822 | 0 | ok |
| qcode-jit | insn-cb | depthconv | 1174.3 | 2998976 | 2998976 | 0 | ok |
| qcode-jit | insn-cb | edn | 1405.6 | 4109614 | 4109614 | 0 | ok |
| qcode-jit | insn-cb | huffbench | 1140.6 | 2795647 | 2795647 | 0 | ok |
| qcode-jit | insn-cb | matmult-int | 1208.5 | 4072595 | 4072595 | 0 | ok |
| qcode-jit | insn-cb | md5sum | 986.0 | 2398400 | 2398400 | 0 | ok |
| qcode-jit | insn-cb | nettle-aes | 1329.4 | 2621639 | 2621639 | 0 | ok |
| qcode-jit | insn-cb | nettle-sha256 | 5429.4 | 4475405 | 4475405 | 0 | ok |
| qcode-jit | insn-cb | nsichneu | 2212.8 | 2167728 | 2167728 | 0 | ok |
| qcode-jit | insn-cb | picojpeg | 2029.5 | 3298380 | 3298380 | 0 | ok |
| qcode-jit | insn-cb | qrduino | 2554.6 | 3670529 | 3670529 | 0 | ok |
| qcode-jit | insn-cb | sglib-combined | 1367.6 | 2797078 | 2797078 | 0 | ok |
| qcode-jit | insn-cb | statemate | 869.8 | 2235589 | 2235589 | 0 | ok |
| qcode-jit | insn-cb | tarfind | 753.4 | 1942280 | 1942280 | 0 | ok |
| qcode-jit | insn-cb | ud | 1223.2 | 2994568 | 2994568 | 0 | ok |
| qcode-jit | insn-cb | xgboost | 1470.3 | 3936130 | 3936130 | 0 | ok |
| qcode-jit | insn-ir | aha-mont64 | 99.2 | 0 | 2428485 | 496 | ok |
| qcode-jit | insn-ir | crc32 | 44.7 | 0 | 2278822 | 108 | ok |
| qcode-jit | insn-ir | depthconv | 55.3 | 0 | 2998976 | 104 | ok |
| qcode-jit | insn-ir | edn | 136.4 | 0 | 4109614 | 692 | ok |
| qcode-jit | insn-ir | huffbench | 187.7 | 0 | 2795647 | 734 | ok |
| qcode-jit | insn-ir | matmult-int | 87.1 | 0 | 4072595 | 283 | ok |
| qcode-jit | insn-ir | md5sum | 111.0 | 0 | 2398400 | 447 | ok |
| qcode-jit | insn-ir | nettle-aes | 192.2 | 0 | 2621639 | 1068 | ok |
| qcode-jit | insn-ir | nettle-sha256 | 1280.4 | 0 | 4475405 | 2845 | ok |
| qcode-jit | insn-ir | nsichneu | 728.5 | 0 | 2167728 | 1858 | ok |
| qcode-jit | insn-ir | picojpeg | 416.4 | 0 | 3298380 | 1928 | ok |
| qcode-jit | insn-ir | qrduino | 735.3 | 0 | 3670529 | 3633 | ok |
| qcode-jit | insn-ir | sglib-combined | 306.8 | 0 | 2797078 | 1165 | ok |
| qcode-jit | insn-ir | statemate | 107.9 | 0 | 2235589 | 468 | ok |
| qcode-jit | insn-ir | tarfind | 64.7 | 0 | 1942280 | 234 | ok |
| qcode-jit | insn-ir | ud | 118.8 | 0 | 2994568 | 436 | ok |
| qcode-jit | insn-ir | xgboost | 123.6 | 0 | 3936130 | 187 | ok |
| qcode-jit | none | aha-mont64 | 59.2 | 0 | 0 | 0 | ok |
| qcode-jit | none | crc32 | 32.2 | 0 | 0 | 0 | ok |
| qcode-jit | none | depthconv | 37.1 | 0 | 0 | 0 | ok |
| qcode-jit | none | edn | 76.0 | 0 | 0 | 0 | ok |
| qcode-jit | none | huffbench | 109.7 | 0 | 0 | 0 | ok |
| qcode-jit | none | matmult-int | 58.6 | 0 | 0 | 0 | ok |
| qcode-jit | none | md5sum | 65.4 | 0 | 0 | 0 | ok |
| qcode-jit | none | nettle-aes | 89.1 | 0 | 0 | 0 | ok |
| qcode-jit | none | nettle-sha256 | 216.7 | 0 | 0 | 0 | ok |
| qcode-jit | none | nsichneu | 453.1 | 0 | 0 | 0 | ok |
| qcode-jit | none | picojpeg | 219.1 | 0 | 0 | 0 | ok |
| qcode-jit | none | qrduino | 420.7 | 0 | 0 | 0 | ok |
| qcode-jit | none | sglib-combined | 183.0 | 0 | 0 | 0 | ok |
| qcode-jit | none | statemate | 59.2 | 0 | 0 | 0 | ok |
| qcode-jit | none | tarfind | 42.4 | 0 | 0 | 0 | ok |
| qcode-jit | none | ud | 79.3 | 0 | 0 | 0 | ok |
| qcode-jit | none | xgboost | 95.7 | 0 | 0 | 0 | ok |
| qcode-jit | watch-cb | aha-mont64 | 128.1 | 5223 | 3 | 45 | ok |
| qcode-jit | watch-cb | crc32 | 3284.5 | 350569 | 175275 | 28 | ok |
| qcode-jit | watch-cb | depthconv | 180.7 | 54128 | 0 | 14 | ok |
| qcode-jit | watch-cb | edn | 982.9 | 173163 | 1065 | 120 | ok |
| qcode-jit | watch-cb | huffbench | 998.1 | 348279 | 768 | 99 | ok |
| qcode-jit | watch-cb | matmult-int | 2129.2 | 596259 | 3360 | 41 | ok |
| qcode-jit | watch-cb | md5sum | 761.3 | 213211 | 536 | 86 | ok |
| qcode-jit | watch-cb | nettle-aes | 1707.8 | 79169 | 0 | 128 | ok |
| qcode-jit | watch-cb | nettle-sha256 | 8109.0 | 258433 | 0 | 171 | ok |
| qcode-jit | watch-cb | nsichneu | 474.2 | 3719 | 2466 | 26 | ok |
| qcode-jit | watch-cb | picojpeg | 2189.5 | 439371 | 402 | 367 | ok |
| qcode-jit | watch-cb | qrduino | 709.6 | 93934 | 78 | 352 | ok |
| qcode-jit | watch-cb | sglib-combined | 972.5 | 308596 | 768 | 202 | ok |
| qcode-jit | watch-cb | statemate | 1367.5 | 1022639 | 116584 | 176 | ok |
| qcode-jit | watch-cb | tarfind | 2161.7 | 536809 | 1739 | 50 | ok |
| qcode-jit | watch-cb | ud | 1535.5 | 207216 | 1786 | 69 | ok |
| qcode-jit | watch-cb | xgboost | 967.9 | 104207 | 0 | 30 | ok |
| qcode-jit | watch-ir | aha-mont64 | 74.5 | 3 | 3 | 0 | ok |
| qcode-jit | watch-ir | crc32 | 127.3 | 175275 | 175275 | 0 | ok |
| qcode-jit | watch-ir | depthconv | 48.2 | 0 | 0 | 0 | ok |
| qcode-jit | watch-ir | edn | 120.9 | 1066 | 1066 | 0 | ok |
| qcode-jit | watch-ir | huffbench | 170.5 | 768 | 768 | 0 | ok |
| qcode-jit | watch-ir | matmult-int | 127.5 | 3360 | 3360 | 0 | ok |
| qcode-jit | watch-ir | md5sum | 105.0 | 536 | 536 | 0 | ok |
| qcode-jit | watch-ir | nettle-aes | 137.8 | 0 | 0 | 0 | ok |
| qcode-jit | watch-ir | nettle-sha256 | 282.7 | 0 | 0 | 0 | ok |
| qcode-jit | watch-ir | nsichneu | 490.3 | 2466 | 2466 | 0 | ok |
| qcode-jit | watch-ir | picojpeg | 347.2 | 402 | 402 | 0 | ok |
| qcode-jit | watch-ir | qrduino | 524.4 | 78 | 78 | 0 | ok |
| qcode-jit | watch-ir | sglib-combined | 256.6 | 768 | 768 | 0 | ok |
| qcode-jit | watch-ir | statemate | 312.5 | 116585 | 116585 | 0 | ok |
| qcode-jit | watch-ir | tarfind | 113.0 | 1739 | 1739 | 0 | ok |
| qcode-jit | watch-ir | ud | 128.4 | 1786 | 1786 | 0 | ok |
| qcode-jit | watch-ir | xgboost | 115.3 | 0 | 0 | 0 | ok |
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
