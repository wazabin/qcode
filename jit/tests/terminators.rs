//! What compiled code owes the interpreter about the values it computes.
//!
//! The interpreter runs the terminator after compiled code has run the body, so
//! any value the terminator reads must be exported back into its value table —
//! and so must any value read from *another* block, which that block imports
//! in turn.

use qcode::{
    context::Context,
    value::{FunctionBody, InstructionId, ValueId},
};
use qcode_emulator::{EmulatorMemory, StandaloneEmulator};
use qcode_jit::Jit;
use qcode_vm::VmMemory;
use wazabin_qcode_macro::qcode;

#[test]
fn a_value_read_from_another_block_is_exported_and_imported() {
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
        fn escapes:
        <entry>
            %a = 1 + 2;
            goto <next>;

        <next>
            %b = %a + 1;
            return %b;
        "
    );
    let blocks = FunctionBody::from_id(&ctx, escapes).block_ids();
    let (entry, next) = (blocks[0], blocks[1]);
    let a = InstructionId::new(escapes, ctx.block(entry).instruction_ids()[0]);
    let b = InstructionId::new(escapes, ctx.block(next).instruction_ids()[0]);

    let mut jit = Jit::new();
    assert_eq!(
        jit.try_compile(&ctx, entry),
        Ok(()),
        "a block whose result crosses a block boundary compiles and exports it"
    );

    // Run the first block natively: its escaping result lands in the value
    // table, where the interpreter — or the next block's compiled code, which
    // imports it — reads it.
    let mut emu = StandaloneEmulator::<VmMemory>::new_in(entry);
    emu.memory.configure_spaces(&ctx);
    jit.run_block(&ctx, &mut emu, entry, 0, false)
        .expect("the block runs")
        .expect("the block was compiled");
    assert_eq!(emu.insn_values.get(&a).map(|v| v.as_bits()), Some(3));

    // The second block imports it and computes on it.
    emu.block = next;
    emu.idx = 0;
    jit.run_block(&ctx, &mut emu, next, 0, false)
        .expect("the block runs")
        .expect("the block was compiled");
    assert_eq!(emu.insn_values.get(&b).map(|v| v.as_bits()), Some(4));
    let _ = ValueId::Instruction(b);
}

#[test]
fn a_condition_consumed_by_the_blocks_own_terminator_compiles() {
    // The counterpart to the test above: the same shape of computed value, but
    // read only by this block's `cbranch`. That is not an escape — compiled
    // code exports the condition so the interpreter can branch on it — so the
    // block must compile rather than trip the guard.
    let mut ctx = Context::new();
    qcode!(
        ctx,
        "
        fn confined:
        <entry>
            %a = 1 + 2;
            %c = %a == 3;
            if %c goto <yes> else goto <no>;

        <yes>
            return 1;

        <no>
            return 0;
        "
    );
    let entry = FunctionBody::from_id(&ctx, confined).block_ids()[0];

    let mut jit = Jit::new();
    assert_eq!(jit.try_compile(&ctx, entry), Ok(()));
}
