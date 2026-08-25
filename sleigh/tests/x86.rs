mod support;

#[cfg(test)]
mod tests {
    use super::support::x86;

    #[test]
    fn test_push_ebp() {
        let insn = x86::Disassembler::from_bytes(0x1000, b"\x55")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "PUSH EBP");
        x86::lift(&mut x86::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_mov_eax_ptr_ecx() {
        let insn = x86::Disassembler::from_bytes(0x1000, b"\x8b\x01")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOV EAX,dword ptr [ECX]");
        x86::lift(&mut x86::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_add_eax_ebx() {
        let insn = x86::Disassembler::from_bytes(0x1000, b"\x01\xd8")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "ADD EAX,EBX");
        x86::lift(&mut x86::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_mov_eax_imm32() {
        let insn = x86::Disassembler::from_bytes(0x1000, b"\xb8\x78\x56\x34\x12")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOV EAX,305419896");
        x86::lift(&mut x86::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_ret() {
        let insn = x86::Disassembler::from_bytes(0x1000, b"\xc3")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "RET");
        x86::lift(&mut x86::make_context(), &insn, None).unwrap();
    }

    /// `INT3` lifts without panicking.
    ///
    /// Its SLEIGH semantics are `...; call [intloc]; return [0:1];` — a statement
    /// *after* a call, which terminates its block. The trailing `return` used to be
    /// appended to the terminated block and trip the builder's assertion, panicking
    /// the entire lift. `0xCC` is MSVC's inter-function padding byte, so any
    /// discovery that stepped one byte into padding took the whole run down.
    #[test]
    fn test_int3_emits_return_in_a_continuation_block() {
        let insn = x86::Disassembler::from_bytes(0x1000, b"\xcc")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "INT3");
        let mut ctx = x86::make_context();
        x86::lift(&mut ctx, &insn, None).unwrap();
        let rendered = ctx.to_string();
        assert!(
            rendered.contains("call ["),
            "the trap should still lower to an indirect call: {rendered}",
        );
        assert!(
            rendered.contains("return"),
            "the post-call return must survive, not be dropped: {rendered}",
        );
    }

    /// The same shape via `INT1` (`0xF1`), the other x86 instruction whose p-code
    /// continues past a `call`. Found in obfuscated code as an anti-debug trap.
    #[test]
    fn test_int1_emits_return_in_a_continuation_block() {
        let insn = x86::Disassembler::from_bytes(0x1000, b"\xf1")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "INT1");
        let mut ctx = x86::make_context();
        x86::lift(&mut ctx, &insn, None).unwrap();
        assert!(ctx.to_string().contains("return"));
    }
}
