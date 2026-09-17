//! Register slices on lifted code, against the real x86-64 specification:
//! zero-extension is explicit in the p-code rather than a rule the consumer
//! must know, and the scratch facade reports each store as a slice of its
//! enclosing register, keyed by the generated `regs` constants.

use qcode::value::insn::Mnemonic;
use sleigh::RegisterSlice;
use sleigh_precompile::x64::regs;
use wazabin_qcode_sleigh::{
    SleighLifter,
    session::{ScratchLifted, ScratchOperand, ScratchSession},
};

/// The destination slice of every register store in the entry block.
fn register_stores(lifted: &ScratchLifted<'_, '_, '_, '_>) -> Vec<RegisterSlice> {
    lifted
        .entry()
        .instructions()
        .filter(|insn| matches!(insn.mnemonic(), Mnemonic::Store { .. }))
        .map(|insn| {
            let Some(ScratchOperand::Varnode(dst)) = insn.operands().next() else {
                panic!("a register store's first operand is its destination");
            };
            dst.enclosing_register()
                .expect("the destination is a GPR slice")
        })
        .collect()
}

#[test]
fn zero_extension_is_explicit_in_the_lifted_code() {
    let lifter = SleighLifter::new(sleigh_precompile::x64::spec());
    let mut session = ScratchSession::new(&lifter);

    // MOV EAX, EDI: a 4-byte store to EAX, then RAX = zext(EAX).
    let lifted = session.lift(0x1000, b"\x89\xf8").unwrap();
    assert_eq!(
        register_stores(&lifted),
        vec![
            RegisterSlice {
                register: regs::RAX,
                offset: 0,
                size: 4
            },
            RegisterSlice {
                register: regs::RAX,
                offset: 0,
                size: 8
            },
        ]
    );
    drop(lifted);

    // MOV AL, BH: a 1-byte store and nothing else — the other bytes stay.
    let lifted = session.lift(0x1002, b"\x88\xf8").unwrap();
    assert_eq!(
        register_stores(&lifted),
        vec![RegisterSlice {
            register: regs::RAX,
            offset: 0,
            size: 1
        }]
    );
    let overlaps: Vec<_> = {
        let Some(ScratchOperand::Varnode(dst)) = lifted
            .entry()
            .instructions()
            .find(|insn| matches!(insn.mnemonic(), Mnemonic::Store { .. }))
            .and_then(|insn| insn.operands().next())
        else {
            panic!("the store's destination is a register");
        };
        dst.overlapping_registers().collect()
    };
    assert_eq!(overlaps, vec![regs::RAX, regs::EAX, regs::AX, regs::AL]);
}
