//! What compiled code owes the interpreter about the values it computes.
//!
//! The interpreter runs the terminator after compiled code has run the body, so
//! any value the terminator reads must be exported back into its value table.

use qcode::{context::Context, value::FunctionBody};
use qcode_jit::Jit;
use wazabin_qcode_macro::qcode;

#[test]
fn a_condition_consumed_by_the_blocks_own_terminator_compiles() {
    // The condition is computed by this block and read by its own `cbranch`.
    // Compiled code exports it so the interpreter can branch on it, so a block
    // shaped like this must compile rather than be declined for its terminator.
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
