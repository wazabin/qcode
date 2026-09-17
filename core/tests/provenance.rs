//! What the public API guarantees about an address index's provenance when
//! function bodies are mutated, cloned or reloaded.
//!
//! Every test here goes through the crate's public surface only: a context
//! owner reaches a body mutably only as a [`BodyMut`] — its verbs, never the
//! value — and cannot reach the registries mutably at all, so these are the
//! paths that exist. The paths that do not (swapping or replacing a body
//! through `body_mut`, taking the body out of a `BodiesMut`) are the
//! compile-fail doctests on [`BodyMut`] and [`Context::bodies`].
//!
//! [`BodyMut`]: qcode::value::BodyMut
//! [`Context::bodies`]: qcode::context::Context::bodies

use qcode::{
    address_index::AddressIndex,
    context::Context,
    lift::{LiftTarget, TargetError},
    value::{BasicBlock, BlockId, FunctionBody, FunctionId, QCodeMut},
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
    let mut target = LiftTarget::bind_or_refresh(ctx, addresses, function).unwrap();
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
fn a_body_borrowed_for_unaddressed_work_leaves_the_index_current() {
    let (mut ctx, mut addresses, function, _) = module(0x1000);
    let before = ctx.revision();

    // Instruction-level and unaddressed work through the body host: the hot
    // path an emulator takes between lifts.
    let scratch = ctx.body_mut(function).make_block();
    ctx.body_mut(function).block_params_mut(scratch).clear();
    {
        let (mut bodies, _shared, _interfaces) = ctx.split_bodies();
        let _ = bodies.get_mut(function).make_block();
        let mut hosts = bodies.select_mut(&[function]);
        let _ = hosts[0].make_block();
    }
    assert_eq!(ctx.revision(), before);
    assert!(addresses.is_current(&ctx));
    LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).unwrap();

    // A tracked change through the host is counted, as always.
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
fn deleting_through_the_split_borrow_moves_the_revision_once() {
    let (mut ctx, mut addresses, function, block) = module(0x1000);
    let other = host(&mut ctx, &mut addresses, 0x2000);
    let before = ctx.revision();
    {
        let (mut bodies, _, _) = ctx.split_bodies();
        let mut hosts = bodies.select_mut(&[function, other]);
        let (ours, theirs) = hosts.split_at_mut(1);
        let _ = theirs[0].make_block();
        ours[0].delete_block(block);
    }
    assert_ne!(ctx.revision(), before);
    assert!(!addresses.is_current(&ctx));
    assert_eq!(
        LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).err(),
        Some(TargetError::OutdatedIndex)
    );
    addresses.refresh(&ctx);
    assert_eq!(addresses.block_at(0x1010), None);
    assert_eq!(blocks_at(&ctx, 0x1010), 0);
}

#[test]
fn a_clone_is_another_module_with_its_own_clock() {
    let (mut ctx, mut addresses, function, block) = module(0x1000);
    let mut twin = ctx.clone();
    let mut twin_index = AddressIndex::analyze(&twin);
    assert_ne!(twin.identity(), ctx.identity());
    assert_eq!(
        LiftTarget::bind_indexed(&mut ctx, &mut twin_index, function).err(),
        Some(TargetError::ForeignIndex)
    );

    // The twin's changes are its own, through every mutation path.
    let extra = addressed_block(&mut twin, &mut twin_index, function, 0x1020);
    assert!(addresses.is_current(&ctx));
    assert!(twin_index.is_current(&twin));
    twin.body_mut(function).delete_block(extra);
    assert!(addresses.is_current(&ctx));
    assert!(!twin_index.is_current(&twin));
    twin_index.refresh(&twin);
    {
        let (mut bodies, _, _) = twin.split_bodies();
        bodies.get_mut(function).delete_block(block);
    }
    assert!(addresses.is_current(&ctx));
    assert!(!twin_index.is_current(&twin));

    // And the original's are the original's.
    ctx.body_mut(function).delete_block(block);
    assert!(!addresses.is_current(&ctx));
    assert_eq!(
        LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).err(),
        Some(TargetError::OutdatedIndex)
    );
    assert_eq!(blocks_at(&ctx, 0x1010), 0);
}

#[test]
fn a_reloaded_module_mutates_on_its_own_clock() {
    let (ctx, _, function, block) = module(0x1000);
    let original = ctx.revision();
    let bytes = bincode::serde::encode_to_vec(&ctx, bincode::config::standard()).unwrap();
    let (mut reloaded, _): (Context<'static>, _) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
    let mut index = AddressIndex::analyze(&reloaded);
    assert_ne!(reloaded.identity(), ctx.identity());
    assert!(!AddressIndex::analyze(&ctx).describes(&reloaded));

    // Unaddressed work keeps the reloaded index current...
    let _ = reloaded.body_mut(function).make_block();
    assert!(index.is_current(&reloaded));
    LiftTarget::bind_indexed(&mut reloaded, &mut index, function).unwrap();

    // ...and an addressed change moves the reloaded module's revision only.
    reloaded.body_mut(function).delete_block(block);
    assert!(!index.is_current(&reloaded));
    assert_eq!(
        LiftTarget::bind_indexed(&mut reloaded, &mut index, function).err(),
        Some(TargetError::OutdatedIndex)
    );
    assert_eq!(
        ctx.revision(),
        original,
        "the module it was read from is untouched"
    );
    assert_eq!(blocks_at(&ctx, 0x1010), 1);
    assert_eq!(blocks_at(&reloaded, 0x1010), 0);
}

#[test]
fn a_clone_of_a_body_is_detached_until_push_function_installs_and_counts_it() {
    let (mut ctx, addresses, function, block) = module(0x1000);
    // The one way a body value leaves a module: a read-only clone, on a
    // detached clock. Mutating the clone moves nothing in the module.
    let mut snapshot = ctx.body(function).clone();
    snapshot.delete_block(block);
    assert!(addresses.is_current(&ctx));
    assert_eq!(blocks_at(&ctx, 0x1010), 1);

    // The one way it enters a module: `push_function`, under its own id,
    // counted at installation since it carries addressed blocks, and on the
    // installing module's clock from then on.
    let mut fresh = Context::new();
    let mut fresh_index = AddressIndex::analyze(&fresh);
    let installed =
        fresh.push_function(ctx.interface(function).clone(), ctx.body(function).clone());
    assert_eq!(installed, function);
    assert!(!fresh_index.is_current(&fresh));
    assert_eq!(
        block_found_by_a_safe_binding(&mut fresh, &mut fresh_index, function, 0x1010),
        block
    );
    assert_eq!(blocks_at(&fresh, 0x1010), 1);
    fresh.body_mut(function).delete_block(block);
    assert!(!fresh_index.is_current(&fresh));
    assert!(addresses.is_current(&ctx), "the source module is untouched");
    ctx.body_mut(function).delete_block(block);
    assert!(!addresses.is_current(&ctx));
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
    let _ = ctx.body_mut(function).block_mut(block);
    assert!(!addresses.is_current(&ctx));
}

#[test]
fn a_poisoned_module_is_refused_by_every_binding() {
    let (mut ctx, mut addresses, function, _) = module(0x1000);
    ctx.poison();
    assert_eq!(
        LiftTarget::bind_or_refresh(&mut ctx, &mut addresses, function).err(),
        Some(TargetError::Poisoned)
    );
    assert_eq!(
        LiftTarget::bind_indexed(&mut ctx, &mut addresses, function).err(),
        Some(TargetError::Poisoned)
    );
    // The flag is part of the module: a clone and a reload carry it.
    assert!(ctx.clone().is_poisoned());
    let bytes = bincode::serde::encode_to_vec(&ctx, bincode::config::standard()).unwrap();
    let (reloaded, _): (Context<'static>, _) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
    assert!(reloaded.is_poisoned());
}
