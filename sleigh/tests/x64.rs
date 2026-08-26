mod support;

#[cfg(test)]
mod tests {
    use super::support::x64;
    use qcode::context::Context;
    use qcode_emulator::Emulator;

    fn x87_space(ctx: &Context<'_>) -> qcode::space::SpaceId {
        *ctx.shared
            .named_spaces
            .get("x87")
            .expect("x86-64 SLEIGH spec has an x87 physical-file space")
    }

    fn write_x87_slot(emu: &mut Emulator<'_>, ctx: &Context<'_>, slot: usize, value: u128) {
        let mut bytes = [0; 10];
        bytes.copy_from_slice(&value.to_le_bytes()[..10]);
        emu.write_memory(x87_space(ctx), (slot * 10) as u64, &bytes)
            .unwrap();
    }

    fn read_x87_slot(emu: &mut Emulator<'_>, ctx: &Context<'_>, slot: usize) -> u128 {
        let bytes = emu
            .read_memory(x87_space(ctx), (slot * 10) as u64, 10)
            .unwrap();
        let mut wide = [0; 16];
        wide[..10].copy_from_slice(&bytes);
        u128::from_le_bytes(wide)
    }

    #[test]
    fn test_mov_rcx_ptr_rdx() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x48\x8b\x0a")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOV RCX,qword ptr [RDX]");

        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_pshufd_imm_order_operands() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x66\x0f\x70\xe4\x00")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "PSHUFD XMM4, XMM4, 0");

        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_mov_ptr_r8_disp_r9() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x4d\x89\x48\x10")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOV qword ptr [R8 + 16],R9");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_mov_eax_imm32() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xb8\x78\x56\x34\x12")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOV EAX,305419896");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_mov_r10_imm64() {
        let insn =
            x64::Disassembler::from_bytes(0x1000, b"\x49\xba\x88\x77\x66\x55\x44\x33\x22\x11")
                .next()
                .unwrap();
        assert_eq!(insn.to_string(), "MOV R10,1234605616436508552");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_lea_r11_rip_relative() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x4c\x8d\x1d\x20\x00\x00\x00")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "LEA R11,[4135]");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_add_rax_rcx() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x48\x01\xc8")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "ADD RAX,RCX");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_sub_rdx_imm8() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x48\x83\xea\x05")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "SUB RDX,5");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_imul_rsi_rdi() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x48\x0f\xaf\xf7")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "IMUL RSI,RDI");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_xor_r8_r8() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x4d\x31\xc0")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "XOR R8,R8");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_cmp_r9_r10() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x4d\x39\xd1")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "CMP R9,R10");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_cdqe() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x48\x98")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "CDQE");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_cqo() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x48\x99")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "CQO");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_shl_rax_imm() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x48\xc1\xe0\x03")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "SHL RAX,3");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    #[ignore = "flat SLEIGH p-code lowering does not yet support this instruction form"]
    fn test_sar_rcx_cl() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x48\xd3\xf9")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "SAR RCX,CL");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_rol_rdx_one() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x48\xd1\xc2")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "ROL RDX,1");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_push_rbx() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x53")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "PUSH RBX");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_pop_r12() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x41\x5c")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "POP R12");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_pushfq() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x9c")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "PUSHFQ");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_popfq() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x9d")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "POPFQ");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    #[ignore = "flat SLEIGH p-code lowering does not yet support this instruction form"]
    fn test_call_rax() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xff\xd0")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "CALL RAX");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_jmp_rel32_missing() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xe9\x00\x00\x00\x00")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "JMP 4101");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_ret() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xc3")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "RET");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_mov_rax_indexed() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x48\x8b\x44\x8b\x20")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOV RAX,qword ptr [RBX + RCX*4 + 32]");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_mov_byte_ptr_addr() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xc6\x05\x00\x01\x00\x00\x7f")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOV byte ptr [4359],127");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    #[ignore = "flat SLEIGH p-code lowering does not yet support this instruction form"]
    fn test_movaps_xmm0_xmm1() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x0f\x28\xc1")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOVAPS XMM0, XMM1");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    #[ignore = "flat SLEIGH p-code lowering does not yet support this instruction form"]
    fn test_movsd_xmm2_ptr_rax() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xf2\x0f\x10\x10")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOVSD XMM2, qword ptr [RAX]");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    #[ignore = "flat SLEIGH p-code lowering does not yet support this instruction form"]
    fn test_addss_xmm3_xmm4() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xf3\x0f\x58\xdc")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "ADDSS XMM3, XMM4");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_movsb_rep() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xf3\xa4")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOVSB.REP RDI,RSI");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_cmpsb_repe() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xf3\xa6")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "CMPSB.REPE RDI,RSI");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_scasb_repne() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xf2\xae")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "SCASB.REPNE RDI");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_stosq_rep() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xf3\x48\xab")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "STOSQ.REP RDI");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_lodsb_rep() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xf3\xac")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "LODSB.REP RSI");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_add_lock() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xf0\x48\x83\x00\x01")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "ADD.LOCK qword ptr [RAX],1");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_xchg_lock() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xf0\x48\x87\x0b")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "XCHG.LOCK qword ptr [RBX],RCX");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_cmpxchg_lock() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xf0\x48\x0f\xb1\x37")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "CMPXCHG.LOCK qword ptr [RDI],RSI");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_dec_ebx_zero_extends_into_rbx_in_long_mode() {
        let insn = x64::Disassembler::from_bytes(0x401000, b"\xff\xcb")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "DEC EBX");

        let mut ctx = x64::make_context();
        x64::lift(&mut ctx, &insn, None).unwrap();
        let text = ctx.to_string();

        assert!(
            text.contains("zext(i64, i32") && text.contains("RBX <- i64"),
            "missing RBX zero-extension in:\n{text}"
        );

        let mut emu = Emulator::from_address(&ctx, 0x401000);
        emu.set_register(x64::RBX, 0x1234_5678_0000_0000).unwrap();
        emu.run_block().unwrap();

        assert_eq!(emu.read_register(x64::EBX), Some(0xffff_ffff));
        assert_eq!(emu.read_register(x64::RBX), Some(0x0000_0000_ffff_ffff));
    }

    #[test]
    fn test_mov_ebx_imm32_zero_extends_into_rbx_in_long_mode() {
        let insn = x64::Disassembler::from_bytes(0x401000, b"\xbb\x78\x56\x34\x12")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOV EBX,305419896");

        let mut ctx = x64::make_context();
        x64::lift(&mut ctx, &insn, None).unwrap();

        let mut emu = Emulator::from_address(&ctx, 0x401000);
        emu.set_register(x64::RBX, 0xfeed_face_0000_0000).unwrap();
        emu.run_block().unwrap();

        assert_eq!(emu.read_register(x64::EBX), Some(0x1234_5678));
        assert_eq!(emu.read_register(x64::RBX), Some(0x0000_0000_1234_5678));
    }

    #[test]
    fn test_mov_ax_bx() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x66\x89\xd8")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOV AX,BX");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_add_word_ptr() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x66\x83\x00\x05")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "ADD word ptr [RAX],5");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_push_imm32() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x68\x34\x12\x00\x00")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "PUSH 4660");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_imul_ax_cx() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x66\x0f\xaf\xc1")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "IMUL AX,CX");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_mov_eax_ptr_ecx() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x67\x8b\x01")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOV EAX,dword ptr [ECX]");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_mov_dword_ptr_indexed() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x67\x89\x04\xb2")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOV dword ptr [EDX + ESI*4],EAX");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_lea_eax_offset() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x67\x8d\x43\x04")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "LEA EAX,[EBX + 4]");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_mov_fs() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x64\x48\x8b\x04\x25\x00\x00\x00\x00")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOV RAX,qword ptr FS:[0]");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_mov_gs() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x65\x48\x8b\x1c\x25\x30\x00\x00\x00")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOV RBX,qword ptr GS:[48]");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_mov_cs() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x2e\x8b\x05\x10\x00\x00\x00")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOV EAX,dword ptr CS:[4119]");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_mov_rdx_rsp() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x48\x8b\x14\x24")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOV RDX,qword ptr [RSP]");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_mov_r8_r9() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x4d\x89\xc8")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOV R8,R9");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_add_r10_r11() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x4d\x01\xda")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "ADD R10,R11");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_mov_r12_r13_ptr() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x4d\x8b\x65\x00")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOV R12,qword ptr [R13]");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    fn test_lea_r14_r15_offset() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x4d\x8d\x77\x08")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "LEA R14,[R15 + 8]");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    #[ignore = "flat SLEIGH p-code lowering does not yet support this instruction form"]
    fn test_movupd_xmm0_xmm1() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x66\x0f\x10\xc1")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOVUPD XMM0, XMM1");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    #[ignore = "flat SLEIGH p-code lowering does not yet support this instruction form"]
    fn test_movsd_xmm2_xmm3() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xf2\x0f\x10\xd3")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOVSD XMM2, XMM3");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    #[ignore = "flat SLEIGH p-code lowering does not yet support this instruction form"]
    fn test_movss_xmm4_xmm5() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xf3\x0f\x10\xe5")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOVSS XMM4, XMM5");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    #[ignore = "flat SLEIGH p-code lowering does not yet support this instruction form"]
    fn test_addpd_xmm6_xmm7() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x66\x0f\x58\xf7")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "ADDPD XMM6, XMM7");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    #[ignore = "flat SLEIGH p-code lowering does not yet support this instruction form"]
    fn test_addsd_xmm1_xmm2() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xf2\x0f\x58\xca")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "ADDSD XMM1, XMM2");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    #[test]
    #[ignore = "flat SLEIGH p-code lowering does not yet support this instruction form"]
    fn test_addss_xmm3_xmm4_second() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xf3\x0f\x58\xdc")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "ADDSS XMM3, XMM4");
        x64::lift(&mut x64::make_context(), &insn, None).unwrap();
    }

    /// `MOV [RBP-4], EDI` (89 7D FC) encodes a signed 8-bit displacement -4
    /// (0xFC).  The lifted IR must sign-extend it to a full 64-bit constant
    /// `0xfffffffffffffffc`, not zero-extend to `0xfc` (252).
    ///
    /// Regression test for the bug where `generate_pcode_args` used
    /// `field_size` (1 byte) for signed fields, causing `*[const]:8 simm8`
    /// to zero-extend the masked byte rather than preserving the sign.
    #[test]
    fn test_signed_displacement_sign_extended() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x89\x7d\xfc")
            .next()
            .unwrap();
        // A `signed` field displays signed; the point of this test is the IR
        // below, which must carry the sign-extended 64-bit value.
        assert_eq!(insn.to_string(), "MOV dword ptr [RBP + -4],EDI");

        let mut ctx = x64::make_context();
        x64::lift(&mut ctx, &insn, None).unwrap();

        let ir = ctx.to_string();
        assert!(
            ir.contains("0xfffffffffffffffc"),
            "expected sign-extended -4 (0xfffffffffffffffc) in IR, got:\n{ir}"
        );
        assert!(
            !ir.contains("+ 0xfc"),
            "unexpected zero-extended displacement 0xfc in IR; should be 0xfffffffffffffffc:\n{ir}"
        );
    }

    /// The concrete interpreter must see x87's control word while evaluating
    /// the f80 p-code emitted for an actual SLEIGH arithmetic instruction.
    #[test]
    fn test_fadd_uses_x87_rounding_and_precision_control() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xd8\xc1")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "FADD ST0, ST1");

        let mut ctx = x64::make_context();
        x64::lift(&mut ctx, &insn, None).unwrap();
        let mut emu = Emulator::from_address(&ctx, 0x1000);
        // RC=up and PC=single.  The operand is half a single-precision ULP,
        // so the PC rounding produces the next single-precision quantum.
        emu.set_register(x64::FPUCONTROLWORD, 0x087f).unwrap();
        let one = 0x3fff_8000_0000_0000_0000u128;
        let half_single_ulp = 0x3fe7_8000_0000_0000_0000u128;
        write_x87_slot(&mut emu, &ctx, 0, one);
        write_x87_slot(&mut emu, &ctx, 1, half_single_ulp);
        emu.run_block().unwrap();

        assert_eq!(read_x87_slot(&mut emu, &ctx, 0), one + (1u128 << 40));
        assert_ne!(emu.read_register(x64::FPUSTATUSWORD).unwrap() & 0x20, 0);
    }

    /// `FST ST1` must copy through TOP-derived physical slots, including the
    /// final high 16 bits of the f80 payload.
    #[test]
    fn test_fst_i80_uses_physical_slots_at_nonzero_top() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xdd\xd1")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "FST ST1");

        let mut ctx = x64::make_context();
        x64::lift(&mut ctx, &insn, None).unwrap();
        let mut emu = Emulator::from_address(&ctx, 0x1000);
        let source = 0x3fff_8000_0000_0000_0000u128; // x87 +1
        emu.set_register(x64::FPUSTATUSWORD, 3 << 11).unwrap();
        write_x87_slot(&mut emu, &ctx, 3, source); // logical ST0
        write_x87_slot(&mut emu, &ctx, 4, 0); // logical ST1
        emu.run_block().unwrap();

        assert_eq!(read_x87_slot(&mut emu, &ctx, 4), source);
    }

    /// `FFREE ST1` must mark the TOP-mapped physical tag pair empty with
    /// ordinary p-code rather than a value-only user-op.
    #[test]
    fn test_ffree_clears_physical_tag_at_nonzero_top() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xdd\xc1")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "FFREE ST1");

        let mut ctx = x64::make_context();
        x64::lift(&mut ctx, &insn, None).unwrap();
        let mut emu = Emulator::from_address(&ctx, 0x1000);
        emu.set_register(x64::FPUSTATUSWORD, 3 << 11).unwrap();
        emu.set_register(x64::FPUTAGWORD, 0).unwrap();
        emu.run_block().unwrap();

        // ST1 at TOP=3 is physical R4, whose full tag pair occupies bits 8:9.
        assert_eq!(emu.read_register(x64::FPUTAGWORD), Some(0x0300));
        let ir = ctx.to_string();
        assert!(
            ir.contains("FPUTagWord"),
            "expected physical tag update:\n{ir}"
        );
        assert!(
            !ir.contains("ffree("),
            "FFREE must not remain a user-op:\n{ir}"
        );
    }

    #[test]
    fn test_fxch_swaps_top_mapped_payload_and_full_tag_classes() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\xd9\xc9")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "FXCH ST1");

        let mut ctx = x64::make_context();
        x64::lift(&mut ctx, &insn, None).unwrap();
        let mut emu = Emulator::from_address(&ctx, 0x1000);
        emu.set_register(x64::FPUSTATUSWORD, 3 << 11).unwrap();
        write_x87_slot(&mut emu, &ctx, 3, 0x1111);
        write_x87_slot(&mut emu, &ctx, 4, 0x2222);
        // R3 is zero-class (01); R4 is special-class (10).
        emu.set_register(x64::FPUTAGWORD, 0x0240).unwrap();
        emu.run_block().unwrap();

        assert_eq!(read_x87_slot(&mut emu, &ctx, 3), 0x2222);
        assert_eq!(read_x87_slot(&mut emu, &ctx, 4), 0x1111);
        assert_eq!(emu.read_register(x64::FPUTAGWORD), Some(0x0180));
    }

    /// MMX operands are low-64 views of the same physical x87 slots. A write
    /// updates only its destination payload's high 16 bits and marks the full
    /// x87 tag word valid; its source slot remains otherwise untouched.
    #[test]
    fn test_mmx_writes_share_physical_x87_file() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x0f\x6f\xc1")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "MOVQ MM0, MM1");

        let mut ctx = x64::make_context();
        x64::lift(&mut ctx, &insn, None).unwrap();
        let mut emu = Emulator::from_address(&ctx, 0x1000);
        let source_low = 0x1122_3344_5566_7788u128;
        let source = source_low | (0x1234u128 << 64);
        write_x87_slot(&mut emu, &ctx, 0, 0);
        write_x87_slot(&mut emu, &ctx, 1, source);
        emu.set_register(x64::FPUSTATUSWORD, 3 << 11).unwrap();
        emu.set_register(x64::FPUTAGWORD, 0xffff).unwrap();
        emu.run_block().unwrap();

        assert_eq!(
            read_x87_slot(&mut emu, &ctx, 0),
            source_low | (0xffffu128 << 64)
        );
        assert_eq!(read_x87_slot(&mut emu, &ctx, 1), source);
        assert_eq!(emu.read_register(x64::FPUTAGWORD), Some(0));
        assert_eq!(emu.read_register(x64::FPUSTATUSWORD), Some(0));
    }

    #[test]
    fn test_fxsave_uses_an_abridged_physical_tag_byte() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x0f\xae\x07")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "FXSAVE [RDI]");

        let mut ctx = x64::make_context();
        x64::lift(&mut ctx, &insn, None).unwrap();
        let mut emu = Emulator::from_address(&ctx, 0x1000);
        emu.set_register(x64::RDI, 0x3000).unwrap();
        emu.set_register(x64::FPUTAGWORD, 0xffff).unwrap();
        emu.run_block().unwrap();

        assert_eq!(
            emu.inspect_memory(ctx.shared.default_space, 0x3004, 1),
            Some(vec![0]),
            "all eight physical full-tag pairs are empty"
        );
    }

    #[test]
    fn test_emms_empties_shared_x87_tags_without_moving_payloads() {
        let insn = x64::Disassembler::from_bytes(0x1000, b"\x0f\x77")
            .next()
            .unwrap();
        assert_eq!(insn.to_string(), "EMMS");

        let mut ctx = x64::make_context();
        x64::lift(&mut ctx, &insn, None).unwrap();
        let mut emu = Emulator::from_address(&ctx, 0x1000);
        let payload = 0x3fff_8000_0000_0000_0000u128;
        write_x87_slot(&mut emu, &ctx, 5, payload);
        emu.set_register(x64::FPUTAGWORD, 0).unwrap();
        emu.run_block().unwrap();

        assert_eq!(read_x87_slot(&mut emu, &ctx, 5), payload);
        assert_eq!(emu.read_register(x64::FPUTAGWORD), Some(0xffff));
    }

    /// x87 compares against a memory operand mix float widths: SLEIGH's
    /// `FICOM m32` builds `local tmp = int2float(m32)` (a 4-byte float) and
    /// feeds it to the `fcom` macro, which compares it against the 10-byte
    /// `ST0`. The emitter must insert an explicit `float2float` widening
    /// instead of handing mismatched sizes to `push_binop`, which panics.
    #[test]
    fn test_x87_memory_compare_widens_float_operands() {
        for (bytes, mnemonic) in [
            (&b"\xda\x17"[..], "FICOM dword ptr [RDI]"),
            (&b"\xda\x1f"[..], "FICOMP dword ptr [RDI]"),
            (&b"\xde\x17"[..], "FICOM word ptr [RDI]"),
            (&b"\xde\x1f"[..], "FICOMP word ptr [RDI]"),
            (&b"\xd8\x17"[..], "FCOM float ptr [RDI]"),
            (&b"\xd8\x1f"[..], "FCOMP float ptr [RDI]"),
        ] {
            let insn = x64::Disassembler::from_bytes(0x1000, bytes).next().unwrap();
            assert_eq!(insn.to_string(), mnemonic);

            let mut ctx = x64::make_context();
            x64::lift(&mut ctx, &insn, None).unwrap();

            let ir = ctx.to_string();
            assert!(
                ir.contains("= int2float(f80,") || ir.contains("= float2float(f80,"),
                "{mnemonic}: expected the flat lifter to widen the operand to f80, got:\n{ir}"
            );
        }
    }
}
