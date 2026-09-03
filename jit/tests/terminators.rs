//! What compiled code owes the interpreter about the values it computes.
//!
//! The interpreter runs the terminator after compiled code has run the body, so
//! any value the terminator reads must be exported back into its value table —
//! and any value read from *another* block cannot be, so the block is declined.

use qcode::{context::Context, value::FunctionBody};
use qcode_jit::{Jit, Unsupported};
use wazabin_qcode_macro::qcode;

#[test]
fn a_value_read_from_another_block_declines_the_block() {
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
    let entry = FunctionBody::from_id(&ctx, escapes).block_ids()[0];

    let mut jit = Jit::new();
    assert_eq!(
        jit.try_compile(&ctx, entry),
        Err(Unsupported::Escapes("result used from another block")),
        "a block whose result crosses a block boundary must be declined, not \
         compiled with the value left in a register"
    );
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
