#!/usr/bin/env python3
"""Emit truth/segments.json for an ELF's PT_LOAD segments (CONTRACT.md).

    mkseg.py <elf> [--base-offset HEX] [--start-override HEX]

For each PT_LOAD: start=p_vaddr, end=p_vaddr+p_memsz, sha256 over the bytes a
loader maps there (filesz bytes from the file, then zero fill to memsz),
kind="code" if PF_X else "data". Addresses are "0x.." strings.
"""
import hashlib, json, struct, sys
from pathlib import Path

def segments(path):
    data = Path(path).read_bytes()
    assert data[:4] == b"\x7fELF" and data[4] == 2, "not a 64-bit ELF"
    phoff, = struct.unpack_from("<Q", data, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
    out = []
    for i in range(phnum):
        p_type, flags, off, vaddr, _pa, filesz, memsz, _al = struct.unpack_from(
            "<IIQQQQQQ", data, phoff + i * phentsize)
        if p_type != 1 or memsz == 0:
            continue
        mapped = bytearray(memsz)
        n = min(filesz, memsz)
        mapped[:n] = data[off:off + n]
        out.append({"start": hex(vaddr), "end": hex(vaddr + memsz),
                    "sha256": hashlib.sha256(bytes(mapped)).hexdigest(),
                    "kind": "code" if (flags & 1) else "data"})
    return out

if __name__ == "__main__":
    print(json.dumps(segments(sys.argv[1]), indent=1))
