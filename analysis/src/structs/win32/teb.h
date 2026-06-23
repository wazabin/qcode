// Windows TEB/PEB layout for the TEB-seeding pass (x86 / 32-bit).
//
// Parsed by `build.rs` with libclang (`-m32`) into nominal qcode structs. Only
// the fields the analysis names are declared; everything else is anonymous
// padding (`unsigned char _padN[width]`) so clang computes the right offsets
// without listing every member. Built-in C types are used (no <stdint.h>) to
// keep the translation unit self-contained.
//
// Declaration order matters: a struct must appear before any struct that points
// at it (the loader resolves pointee structs as it goes — no forward refs).

struct PEB {
    unsigned char  _pad0[0x2];
    unsigned char  BeingDebugged;      // 0x02
    unsigned char  _pad1[0x15];
    void*          ProcessHeap;        // 0x18
    unsigned char  _pad2[0x4c];
    unsigned int   NtGlobalFlag;       // 0x68
};

struct TEB {
    unsigned char  _pad0[0x30];
    struct PEB*    ProcessEnvironmentBlock;   // 0x30
};
