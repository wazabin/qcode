//! How editing the front of a block scales with the block's length.
//!
//! A block's instruction order is a linked list threaded through the
//! instructions, so inserting before an instruction and removing one cost
//! the same however long the block is. This probe times both at the front
//! of blocks of growing size; each line should take about twice the one
//! before it, not four times. Run with:
//!
//! ```sh
//! cargo test --release -p wazabin-qcode --test insn_order_probe -- --ignored --nocapture
//! ```
//!
//! It uses only verbs that predate the list, so the same file measures the
//! vector representation it replaced.

use qcode::{context::Context, value::BasicBlock};
use std::time::Instant;

#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn editing_the_front_of_a_long_block_does_not_scale_with_it() {
    let sizes = [20_000usize, 40_000, 80_000, 160_000];
    for &n in &sizes {
        let mut ctx = Context::new();
        let f = ctx.anon_function();
        let block = ctx.get_or_make_block(0x1000, f);
        let scratch = ctx.get_or_make_block(0x2000, f);
        // Each instruction negates a constant of its own, so every use
        // list stays one long and only the block list is measured.
        let consts: Vec<_> = (0..n as u64).map(|i| ctx.get_const(i, 8).id()).collect();

        // Every instruction is minted in the scratch block and moved to the
        // front of the block under test, before the one moved there last.
        let start = Instant::now();
        let mut ids = Vec::with_capacity(n);
        let mut front = ctx.builder(block).push_bit_negate(consts[0]).id;
        ids.push(front);
        for &c in &consts[1..] {
            let id = ctx.builder(scratch).push_bit_negate(c).id;
            ctx.body_mut(f).move_insn_before(id, front);
            front = id;
            ids.push(id);
        }
        let inserted = start.elapsed();
        assert_eq!(BasicBlock::from_id(&ctx, block).len(), n);

        // Then the first instruction goes, n times.
        let start = Instant::now();
        for &id in ids.iter().rev() {
            ctx.body_mut(f).remove_instruction(id);
        }
        let removed = start.elapsed();
        assert_eq!(BasicBlock::from_id(&ctx, block).len(), 0);

        eprintln!(
            "n={n:>7}: {n} insertions at the front in {inserted:>12?}, {n} removals of the first in {removed:>12?}"
        );
    }
}
