//! Diagnostic: which blocks the compiler declines on the VM's own lifting path,
//! and why.
use qcode::value::BasicBlock;
use qcode_jit::Jit;
use qcode_vm::{Vm, VmMemory, perm};
use wazabin_qcode_sleigh::vm_source::SleighCodeSource;

#[test]
#[ignore = "diagnostic"]
fn report_decline_reasons() {
    let code: &[u8] = &[
        0xb8, 0x39, 0x05, 0x00, 0x00, // mov eax, 1337
        0x01, 0xd8, // add eax, ebx
        0xff, 0xc9, // dec ecx
        0x75, 0xfc, // jnz
    ];
    let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
    let ctx = source.new_context();
    let mut memory = VmMemory::new();
    memory
        .mmu
        .write_unchecked(0x1000, code, perm::READ | perm::EXEC);
    let mut vm = Vm::at_address(ctx, 0x1000, source, memory).expect("entry decodes");
    vm.run(5000);

    let ctx = vm.context().clone();
    let mut jit = Jit::new();
    for block in ctx.block_ids() {
        let b = BasicBlock::from_id(&ctx, block);
        let n = b.instruction_ids().len();
        let outcome = jit.try_compile(&ctx, block);
        eprintln!(
            "block {:?} addr={:x?} insns={n} -> {}",
            block,
            b.address(),
            match outcome {
                Ok(()) => "compiled".to_owned(),
                Err(reason) => format!("declined: {reason}"),
            }
        );
    }
}
