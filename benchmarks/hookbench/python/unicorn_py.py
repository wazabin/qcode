#!/usr/bin/env python3
"""The Embench images under Unicorn's Python binding: the hook shape Qiling
builds on. Qiling adds its own dispatch on top of these callbacks, so this
is the floor of what a Qiling hook costs.

Prints "unicorn-py <instr> <image> <ms> <calls> <ok>" per run."""
import struct, sys, time, glob, os
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_HOOK_BLOCK, UC_HOOK_CODE, UC_HOOK_MEM_WRITE, UC_PROT_ALL, UC_PROT_READ, UC_PROT_WRITE
from unicorn.x86_const import UC_X86_REG_RSP, UC_X86_REG_RAX, UC_X86_REG_RIP

SENTINEL = 0xdead0000
STACK, STACK_SIZE, STACK_TOP = 0x7fff0000, 0x40000, 0x7fff8000
WATCH_LEN = 32
PAGE = 0x1000

def segments(b):
    assert b[:4] == b"\x7fELF" and b[4] == 2
    entry = struct.unpack_from("<Q", b, 0x18)[0]
    phoff = struct.unpack_from("<Q", b, 0x20)[0]
    phentsize, phnum = struct.unpack_from("<HH", b, 0x36)
    segs = []
    for i in range(phnum):
        p = phoff + i * phentsize
        typ, flags = struct.unpack_from("<II", b, p)
        if typ != 1:
            continue
        off, vaddr, _, filesz, memsz = struct.unpack_from("<QQQQQ", b, p + 8)
        segs.append((vaddr, b[off:off + filesz], memsz, flags))
    return entry, segs

def run(image, instr, repeat):
    entry, segs = segments(image)
    watch = next(v for v, _, _, f in segs if f & 2)
    best = None
    for _ in range(repeat):
        uc = Uc(UC_ARCH_X86, UC_MODE_64)
        for vaddr, data, memsz, _ in segs:
            start = vaddr & ~(PAGE - 1)
            end = (vaddr + max(memsz, 1) + PAGE - 1) & ~(PAGE - 1)
            uc.mem_map(start, end - start, UC_PROT_ALL)
            uc.mem_write(vaddr, data)
        uc.mem_map(STACK, STACK_SIZE, UC_PROT_READ | UC_PROT_WRITE)
        uc.mem_write(STACK_TOP, struct.pack("<Q", SENTINEL))
        uc.mem_map(SENTINEL & ~(PAGE - 1), PAGE, UC_PROT_ALL)
        uc.reg_write(UC_X86_REG_RSP, STACK_TOP)
        calls = [0]
        def cb(uc, address, size, user):
            calls[0] += 1
        def mem_cb(uc, access, address, size, value, user):
            calls[0] += 1
            return True
        if instr == "block-cb":
            uc.hook_add(UC_HOOK_BLOCK, cb)
        elif instr == "insn-cb":
            uc.hook_add(UC_HOOK_CODE, cb)
        elif instr == "watch-cb":
            uc.hook_add(UC_HOOK_MEM_WRITE, mem_cb, begin=watch, end=watch + WATCH_LEN - 1)
        t0 = time.perf_counter()
        uc.emu_start(entry, SENTINEL)
        t1 = time.perf_counter()
        ok = uc.reg_read(UC_X86_REG_RIP) == SENTINEL and (uc.reg_read(UC_X86_REG_RAX) & 0xffffffff) == 0
        if best is None or t1 - t0 < best[0]:
            best = (t1 - t0, calls[0], ok)
    return best

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "../../../target/embench"
    repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    for path in sorted(glob.glob(os.path.join(d, "*.elf"))):
        name = os.path.basename(path)[:-4]
        image = open(path, "rb").read()
        for instr in ["none", "block-cb", "insn-cb", "watch-cb"]:
            t, calls, ok = run(image, instr, repeat)
            print(f"unicorn-py {instr} {name} {t * 1e3:.1f} {calls} {'ok' if ok else 'FAIL'}", flush=True)
