//! How maintaining the use edges scales with a value's fanout.
//!
//! A value's uses are a linked list threaded through the use edges, so
//! recording one and moving all of them cost the same however many there
//! are, and bulk deletion walks each operand's list once. This probe times
//! lifting a block whose every instruction reads one shared literal, then
//! rewriting all its users and deleting it in bulk, at growing sizes; each
//! line should take about twice the one before it. Run with:
//!
//! ```sh
//! cargo test --release -p wazabin-qcode --test use_edges_probe -- --ignored --nocapture
//! ```
//!
//! It uses only verbs that predate the edges, so the same file measures the
//! map of vectors they replaced.

use qcode::{
    context::Context,
    value::{BasicBlock, ValueId},
};
use rustc_hash::FxHashSet;
use std::time::Instant;

fn ctx_const(b: &mut qcode::builder::Builder<'_, '_>, n: u64) -> ValueId {
    b.shr().get_const(n, 8)
}

#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn a_value_with_many_users_costs_each_of_them_once() {
    let sizes = [25_000usize, 50_000, 100_000, 200_000];
    for &n in &sizes {
        let mut ctx = Context::new();
        let f = ctx.anon_function();
        let block = ctx.get_or_make_block(0x1000, f);
        let other = ctx.get_const(9, 8).id();

        // Every instruction reads one shared value and the one before it.
        let start = Instant::now();
        let ids: Vec<_> = {
            let mut b = ctx.builder(block);
            let seven = ctx_const(&mut b, 7);
            let shared = b.push_bit_negate(seven).id;
            let mut prev = shared;
            let mut ids = vec![shared];
            for _ in 1..n {
                prev = b
                    .push_binop(
                        qcode::value::insn::Binop::Int(qcode::value::insn::IntBinop::Add),
                        ValueId::Instruction(prev),
                        ValueId::Instruction(shared),
                    )
                    .id;
                ids.push(prev);
            }
            ids
        };
        let shared = ValueId::Instruction(ids[0]);
        let lift = start.elapsed();

        // Every user of the shared value is rewritten to a literal.
        let start = Instant::now();
        ctx.bodies[f].replace_all_uses_with(shared, other);
        let rewrite = start.elapsed();
        assert!(!ctx.bodies[f].has_users(shared));

        // Then all but the last go, sharing their operand.
        let start = Instant::now();
        let dead: FxHashSet<_> = ids[..n - 1].iter().map(|id| id.local).collect();
        ctx.bodies[f].remove_block_instructions(block, &dead);
        let delete = start.elapsed();
        assert_eq!(BasicBlock::from_id(&ctx, block).instructions().count(), 1);

        println!(
            "{n:>7} users: lift {lift:>9.3?}   rewrite all uses {rewrite:>9.3?}   bulk delete {delete:>9.3?}"
        );
    }
}
