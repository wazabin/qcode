//! What the public API guarantees about an address index's provenance when
//! function bodies are lent out, swapped, replaced, cloned or reloaded.
//!
//! Every test here goes through the crate's public surface only: a context
//! owner cannot reach a body except by loan, and cannot reach the registries
//! mutably at all, so these are the paths that exist.

use std::mem;

use qcode::{
    address_index::AddressIndex,
    context::Context,
    lift::{LiftTarget, TargetError},
    value::{BasicBlock, BlockId, FunctionBody, FunctionId},
};

fn host(ctx: &mut Context<'static>, addresses: &mut AddressIndex, address: u64) -> FunctionId {
    FunctionBody::make_at_addr_indexed(ctx, addresses, address, None).id
}

fn addressed_block(
    ctx: &mut Context<'static>,
    addresses: &mut AddressIndex,
    function: FunctionId,
    address: u64,
) -> BlockId {
    BasicBlock::make(ctx, function)
        .with_address_indexed(addresses, address)
        .id
}

/// A module with one host at `base` and one more block at `base + 0x10`,
/// and an index current for it.
fn module(base: u64) -> (Context<'static>, AddressIndex, FunctionId, BlockId) {
    let mut ctx = Context::new();
    let mut addresses = AddressIndex::analyze(&ctx);
    let function = host(&mut ctx, &mut addresses, base);
    let block = addressed_block(&mut ctx, &mut addresses, function, base + 0x10);
    assert!(addresses.is_current(&ctx));
    (ctx, addresses, function, block)
}

/// Binds safely and asks the construction for the block at `address`: the
/// existing one, not a duplicate.
fn block_found_by_a_safe_binding(
    ctx: &mut Context<'static>,
    addresses: &mut AddressIndex,
    function: FunctionId,
    address: u64,
) -> BlockId {
    let mut target = LiftTarget::bind(ctx, addresses, function).unwrap();
    let mut construction = target
        .begin(function_address(target.context(), function), 1)
        .unwrap();
    let found = construction.block_at(address).unwrap();
    construction.abort();
    found
}

fn function_address(ctx: &Context<'_>, function: FunctionId) -> u64 {
    ctx.interface(function).address().unwrap()
}

fn blocks_at(ctx: &Context<'_>, address: u64) -> usize {
    ctx.block_ids()
        .into_iter()
        .filter(|&b| ctx.block(b).address() == Some(address))
        .count()
}

#[test]
fn a_loan_that_changes_no_address_leaves_the_index_current() {
    let (mut ctx, mut addresses, function, _) = module(0x1000);
    let before = ctx.revision();

    // Instruction-level and unaddressed work through a loan: the hot path
    // an emulator takes between lifts.
    let scratch = ctx.body_mut(function).make_block();
    ctx.body_mut(function).block_params_mut(scratch).clear();
    {
        let (mut bodies, _shared, _interfaces) = ctx.split_bodies();
        let _ = bodies.get_mut(function).make_block();
        let mut loans = bodies.select_mut(&[function]);
        let _ = loans[0].make_block();
    }
    assert_eq!(ctx.revision(), before);
    assert!(addresses.is_current(&ctx));
    LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();

    // A tracked change through a loan is counted, as always.
    ctx.body_mut(function).delete_block(scratch);
    assert_eq!(
        ctx.revision(),
        before,
        "an unaddressed block is not indexed"
    );
    let block = addressed_block(&mut ctx, &mut addresses, function, 0x1020);
    ctx.body_mut(function).delete_block(block);
    assert_ne!(ctx.revision(), before);
    assert!(!addresses.is_current(&ctx));
}

#[test]
fn swapping_bodies_between_two_modules_with_equal_ids_leaves_both_indexes_behind() {
    let (mut left, mut left_index, function, left_block) = module(0x1000);
    let (mut right, mut right_index, right_function, right_block) = module(0x2000);
    assert_eq!(
        function, right_function,
        "the same numeric id in both modules"
    );

    mem::swap(
        &mut *left.body_mut(function),
        &mut *right.body_mut(function),
    );

    // Both modules now cover addresses their indexes never listed.
    assert!(!left_index.is_current(&left));
    assert!(!right_index.is_current(&right));
    assert_eq!(
        LiftTarget::bind_indexed(&mut left, &mut left_index, function).err(),
        Some(TargetError::OutdatedIndex)
    );
    assert_eq!(
        LiftTarget::bind_indexed(&mut right, &mut right_index, function).err(),
        Some(TargetError::OutdatedIndex)
    );

    // The safe binding rebuilds and finds the swapped-in blocks; nothing is
    // made twice.
    assert_eq!(
        block_found_by_a_safe_binding(&mut left, &mut left_index, function, 0x2010),
        right_block
    );
    assert_eq!(
        block_found_by_a_safe_binding(&mut right, &mut right_index, function, 0x1010),
        left_block
    );
    assert_eq!(blocks_at(&left, 0x2010), 1);
    assert_eq!(blocks_at(&right, 0x1010), 1);
    assert!(left_index.is_current(&left));
    assert!(right_index.is_current(&right));

    // From here on each body ticks the module it is in, not the one it came
    // from.
    left.body_mut(function).delete_block(right_block);
    assert!(!left_index.is_current(&left));
    assert!(right_index.is_current(&right));
    right.body_mut(function).delete_block(left_block);
    assert!(!right_index.is_current(&right));
}

#[test]
fn swapping_through_the_split_borrow_is_caught_the_same_way() {
    let (mut left, mut left_index, function, _) = module(0x1000);
    let (mut right, right_index, _, right_block) = module(0x2000);

    {
        let (mut left_bodies, _, _) = left.split_bodies();
        let (mut right_bodies, _, _) = right.split_bodies();
        let mut ours = left_bodies.select_mut(&[function]);
        let mut theirs = right_bodies.get_mut(function);
        mem::swap(&mut *ours[0], &mut *theirs);
    }
    assert!(!left_index.is_current(&left));
    assert!(!right_index.is_current(&right));
    assert_eq!(
        block_found_by_a_safe_binding(&mut left, &mut left_index, function, 0x2010),
        right_block
    );
    assert_eq!(blocks_at(&left, 0x2010), 1);
}

#[test]
fn a_clone_and_its_original_swap_like_any_two_modules() {
    let (mut ctx, mut addresses, function, block) = module(0x1000);
    let mut twin = ctx.clone();
    let mut twin_index = AddressIndex::analyze(&twin);
    assert_ne!(twin.identity(), ctx.identity());
    assert_eq!(
        LiftTarget::bind_indexed(&mut ctx, &mut twin_index, function).err(),
        Some(TargetError::ForeignIndex)
    );

    // The twin's changes are its own...
    let extra = addressed_block(&mut twin, &mut twin_index, function, 0x1020);
    assert!(addresses.is_current(&ctx));
    assert!(twin_index.is_current(&twin));

    // ...until its body is put into the original.
    mem::swap(&mut *ctx.body_mut(function), &mut *twin.body_mut(function));
    assert!(!addresses.is_current(&ctx));
    assert!(!twin_index.is_current(&twin));
    assert_eq!(
        block_found_by_a_safe_binding(&mut ctx, &mut addresses, function, 0x1020),
        extra
    );
    assert_eq!(blocks_at(&ctx, 0x1020), 1);
    assert_eq!(
        block_found_by_a_safe_binding(&mut twin, &mut twin_index, function, 0x1010),
        block
    );
    // Deleting through the original moves the original's revision only.
    ctx.body_mut(function).delete_block(extra);
    assert!(!addresses.is_current(&ctx));
    assert!(twin_index.is_current(&twin));
}

#[test]
fn a_reloaded_module_lends_its_bodies_on_its_own_clock() {
    let (ctx, _, function, _) = module(0x1000);
    let original = ctx.revision();
    let bytes = bincode::serde::encode_to_vec(&ctx, bincode::config::standard()).unwrap();
    let (mut reloaded, _): (Context<'static>, _) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
    let mut index = AddressIndex::analyze(&reloaded);
    assert_ne!(reloaded.identity(), ctx.identity());
    assert!(!AddressIndex::analyze(&ctx).describes(&reloaded));

    // A loan that changes nothing keeps the reloaded index current...
    let _ = reloaded.body_mut(function).make_block();
    assert!(index.is_current(&reloaded));
    LiftTarget::bind_indexed(&mut reloaded, &mut index, function).unwrap();

    // ...and a swap into the reloaded module is caught like any other.
    let (mut other, _, _, other_block) = module(0x3000);
    mem::swap(
        &mut *reloaded.body_mut(function),
        &mut *other.body_mut(function),
    );
    assert!(!index.is_current(&reloaded));
    assert_eq!(
        block_found_by_a_safe_binding(&mut reloaded, &mut index, function, 0x3010),
        other_block
    );
    reloaded.body_mut(function).delete_block(other_block);
    assert!(!index.is_current(&reloaded));
    assert_eq!(
        ctx.revision(),
        original,
        "the module it was read from is untouched"
    );
}

#[test]
fn a_body_replaced_and_restored_is_counted_both_times() {
    let (mut ctx, mut addresses, function, block) = module(0x1000);

    // Moved out: the slot holds an empty body.
    let original = mem::replace(
        &mut *ctx.body_mut(function),
        FunctionBody::empty_with_id(function),
    );
    assert!(!addresses.is_current(&ctx));
    addresses.refresh(&ctx);
    assert_eq!(addresses.block_at(0x1010), None);
    assert!(addresses.is_current(&ctx));

    // Moved back in: the very same value, whose addresses the refreshed
    // index does not list.
    *ctx.body_mut(function) = original;
    assert!(!addresses.is_current(&ctx));
    assert_eq!(
        block_found_by_a_safe_binding(&mut ctx, &mut addresses, function, 0x1010),
        block
    );
    assert_eq!(blocks_at(&ctx, 0x1010), 1);
    assert!(!ctx.is_poisoned());
}

#[test]
fn a_clone_taken_before_a_tracked_deletion_cannot_be_put_back_unnoticed() {
    let (mut ctx, mut addresses, function, block) = module(0x1000);
    let snapshot = ctx.body(function).clone();

    BasicBlock::from_id_mut(&mut ctx, block).delete();
    assert!(!addresses.is_current(&ctx));
    addresses.refresh(&ctx);
    assert_eq!(addresses.block_at(0x1010), None);

    *ctx.body_mut(function) = snapshot;
    assert!(!addresses.is_current(&ctx));
    assert_eq!(
        LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).err(),
        Some(TargetError::OutdatedIndex)
    );
    assert_eq!(
        block_found_by_a_safe_binding(&mut ctx, &mut addresses, function, 0x1010),
        block
    );
    assert_eq!(blocks_at(&ctx, 0x1010), 1);
}

#[test]
fn a_slot_given_a_body_of_another_function_poisons_the_module() {
    let (mut ctx, mut addresses, function, _) = module(0x1000);
    let other = host(&mut ctx, &mut addresses, 0x2000);
    {
        let (mut bodies, _, _) = ctx.split_bodies();
        let mut loans = bodies.select_mut(&[function, other]);
        let (ours, theirs) = loans.split_at_mut(1);
        mem::swap(&mut *ours[0], &mut *theirs[0]);
    }
    assert!(ctx.is_poisoned());
    assert!(!addresses.is_current(&ctx));
    assert_eq!(
        LiftTarget::bind(&mut ctx, &mut addresses, function).err(),
        Some(TargetError::Poisoned)
    );

    let (mut ctx, mut addresses, function, _) = module(0x1000);
    *ctx.body_mut(function) = FunctionBody::detached();
    assert!(ctx.is_poisoned());
    assert_eq!(
        LiftTarget::bind(&mut ctx, &mut addresses, function).err(),
        Some(TargetError::Poisoned)
    );
}

#[test]
fn installing_something_that_already_carries_addresses_moves_the_revision() {
    let (mut ctx, mut addresses, function, block) = module(0x1000);

    let copy = ctx.block(block).clone();
    let pushed = ctx.push_block(function, copy);
    assert!(!addresses.is_current(&ctx));
    assert_ne!(pushed, block);
    addresses.refresh(&ctx);

    let interface = ctx.interface(function).clone();
    let next = FunctionId::from(ctx.function_count());
    ctx.push_function(interface, FunctionBody::empty_with_id(next));
    assert!(!addresses.is_current(&ctx));
    addresses.refresh(&ctx);

    let mut body = FunctionBody::detached();
    body.install_id(FunctionId::from(ctx.function_count()));
    ctx.push_function(
        qcode::value::function::FunctionInterface::new("fresh".into()),
        body,
    );
    assert!(
        addresses.is_current(&ctx),
        "an empty, address-less function is not indexed"
    );
}

#[test]
fn a_block_handed_out_mutably_counts_as_a_change_but_its_parameters_do_not() {
    let (mut ctx, mut addresses, function, block) = module(0x1000);
    let _ = ctx.block_mut(block);
    assert!(!addresses.is_current(&ctx));
    addresses.refresh(&ctx);

    ctx.body_mut(function).block_params_mut(block).clear();
    assert!(addresses.is_current(&ctx));
}

#[test]
fn a_forgotten_loan_keeps_the_module_unsettled_until_the_next_binding_repairs_it() {
    let (mut left, mut left_index, function, left_block) = module(0x1000);
    let (mut right, mut right_index, _, right_block) = module(0x2000);

    // Swapped through two loans that are never returned: no settlement runs.
    let mut ours = left.body_mut(function);
    let mut theirs = right.body_mut(function);
    mem::swap(&mut *ours, &mut *theirs);
    mem::forget(ours);
    mem::forget(theirs);

    // The revision alone would still match; the unsettled loan is what
    // keeps the index from passing for current.
    assert!(left.has_unsettled_loans());
    assert!(right.has_unsettled_loans());
    assert!(!left_index.is_current(&left));
    assert!(!right_index.is_current(&right));
    assert_eq!(
        LiftTarget::bind_indexed(&mut left, &mut left_index, function).err(),
        Some(TargetError::OutdatedIndex)
    );
    assert!(!left.has_unsettled_loans(), "binding settled the leak");

    // The safe binding finds the swapped-in block, once.
    assert_eq!(
        block_found_by_a_safe_binding(&mut left, &mut left_index, function, 0x2010),
        right_block
    );
    assert_eq!(blocks_at(&left, 0x2010), 1);
    assert_eq!(
        block_found_by_a_safe_binding(&mut right, &mut right_index, function, 0x1010),
        left_block
    );
    assert!(left_index.is_current(&left));
    assert!(right_index.is_current(&right));

    // Settlement relinked the bodies: each now ticks the module it is in.
    left.body_mut(function).delete_block(right_block);
    assert!(!left_index.is_current(&left));
    assert!(right_index.is_current(&right));
    right.body_mut(function).delete_block(left_block);
    assert!(!right_index.is_current(&right));
}

#[test]
fn a_forgotten_loan_that_replaced_the_body_is_settled_by_the_next_loan() {
    let (mut ctx, mut addresses, function, block) = module(0x1000);
    let snapshot = ctx.body(function).clone();
    BasicBlock::from_id_mut(&mut ctx, block).delete();
    addresses.refresh(&ctx);
    assert!(addresses.is_current(&ctx));

    let mut loan = ctx.body_mut(function);
    *loan = snapshot;
    mem::forget(loan);
    assert!(!addresses.is_current(&ctx));
    // A refresh while unsettled sees the true contents but still cannot be
    // current: the restored body is on a detached clock until settled.
    addresses.refresh(&ctx);
    assert!(!addresses.is_current(&ctx));

    // The next loan settles first, then the refreshed index follows the
    // module again.
    let _ = ctx.body_mut(function).make_block();
    assert!(!ctx.has_unsettled_loans());
    assert!(!addresses.is_current(&ctx), "settlement moved the revision");
    assert_eq!(
        block_found_by_a_safe_binding(&mut ctx, &mut addresses, function, 0x1010),
        block
    );
    assert_eq!(blocks_at(&ctx, 0x1010), 1);
    // ...and the relinked body ticks this module.
    ctx.body_mut(function).delete_block(block);
    assert!(!addresses.is_current(&ctx));
}

#[test]
fn a_forgotten_loan_that_left_another_function_in_the_slot_poisons_at_settlement() {
    let (mut ctx, mut addresses, function, _) = module(0x1000);
    let other = host(&mut ctx, &mut addresses, 0x2000);
    {
        let (mut bodies, _, _) = ctx.split_bodies();
        let mut loans = bodies.select_mut(&[function, other]);
        let (ours, theirs) = loans.split_at_mut(1);
        mem::swap(&mut *ours[0], &mut *theirs[0]);
        mem::forget(loans);
    }
    assert!(ctx.has_unsettled_loans());
    assert!(!ctx.is_poisoned(), "nothing has settled yet");
    assert_eq!(
        LiftTarget::bind(&mut ctx, &mut addresses, function).err(),
        Some(TargetError::Poisoned)
    );
    assert!(ctx.is_poisoned());

    let (mut ctx, _, function, _) = module(0x1000);
    let mut loan = ctx.body_mut(function);
    *loan = FunctionBody::detached();
    mem::forget(loan);
    ctx.settle_loans();
    assert!(ctx.is_poisoned());
}

#[test]
fn a_forgotten_loan_that_changed_nothing_costs_one_settlement_and_no_rebuild_of_truth() {
    let (mut ctx, mut addresses, function, block) = module(0x1000);
    mem::forget(ctx.body_mut(function));
    assert!(!addresses.is_current(&ctx));
    // Settled by the safe binding: the index is rebuilt once and then stays
    // current across ordinary loans again.
    assert_eq!(
        block_found_by_a_safe_binding(&mut ctx, &mut addresses, function, 0x1010),
        block
    );
    assert!(addresses.is_current(&ctx));
    let _ = ctx.body_mut(function).make_block();
    assert!(addresses.is_current(&ctx));
    LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();
}
