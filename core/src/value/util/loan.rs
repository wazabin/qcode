//! Exclusive access to installed function bodies that the module's revision
//! accounts for.
//!
//! A [`Context`](crate::context::Context) never hands out a bare
//! `&mut FunctionBody`, and never its bodies registry mutably. What a holder
//! does with exclusive access is not observable — it can move a body out
//! with `mem::replace`, swap it with a body of another module that happens
//! to use the same ids, or put back a clone taken before a tracked deletion —
//! and every one of those changes which addresses the module covers without
//! any tracked mutator running. So the access comes as a [`BodyLoan`], a
//! guard that remembers which body *value* it lent (every body carries an
//! identity of its own, fresh on construction, cloning and deserialization)
//! and, when it is returned, settles the slot against the module:
//!
//! - the slot holds the value that was lent: nothing to do — every
//!   address-bearing change the holder made ran through a body mutator, and
//!   those tick the module's clock themselves;
//! - the slot holds another value with the right function id: it was
//!   replaced or swapped; the value is linked to this module's clock (a body
//!   that arrived from another module would otherwise keep ticking that
//!   module) and the revision moves, so every index of this module is behind;
//! - the slot holds a value of another function id, or a detached one: the
//!   registry no longer describes its own functions, which no rebuild can
//!   repair; the module is [poisoned](crate::context::Context::is_poisoned).
//!
//! The other module in a swap lent its body the same way, so it settles the
//! same way. A loan costs one identity comparison to settle, so a caller that
//! keeps its index current across instruction-level edits — an emulator
//! optimizing between lifts — still binds it without a rebuild.
//!
//! [`BodiesMut`] is the same guarantee over the whole registry, for a driver
//! that borrows several bodies at once: loans by id, read-only access to the
//! rest, and no way to add, remove or replace a slot.

use std::{
    ops::{Deref, DerefMut, Index},
    sync::atomic::{AtomicBool, Ordering},
};

use jstd::registry::{self, Identified, Registry};

use crate::{
    context::ShapeClock,
    value::{BodyIdentity, FunctionBody, FunctionId},
};

/// Exclusive access to one installed function body, settled against its
/// module's revision when dropped. See the [module documentation](self).
///
/// Dereferences to the body; the loan itself adds nothing to it.
///
/// A loan borrows its module for as long as it lives, so no index of that
/// module can be consulted while a body is out:
///
/// ```compile_fail,E0499
/// # use qcode::context::Context;
/// # use qcode::address_index::AddressIndex;
/// # use qcode::lift::LiftTarget;
/// let mut ctx = Context::new();
/// let f = ctx.anon_function();
/// let mut addresses = AddressIndex::analyze(&ctx);
/// let body = ctx.body_mut(f);
/// let _target = LiftTarget::bind(&mut ctx, &mut addresses, f); // `body` still borrows `ctx`
/// let _ = body.root_id();
/// ```
pub struct BodyLoan<'a, 'str> {
    body: &'a mut FunctionBody<'str>,
    slot: FunctionId,
    lent: BodyIdentity,
    clock: &'a ShapeClock,
    poisoned: &'a AtomicBool,
}

impl<'a, 'str> BodyLoan<'a, 'str> {
    pub(crate) fn new(
        body: &'a mut FunctionBody<'str>,
        slot: FunctionId,
        clock: &'a ShapeClock,
        poisoned: &'a AtomicBool,
    ) -> Self {
        let lent = body.identity();
        Self {
            body,
            slot,
            lent,
            clock,
            poisoned,
        }
    }

    /// The registry slot this loan is for.
    pub fn slot(&self) -> FunctionId {
        self.slot
    }
}

impl<'str> Deref for BodyLoan<'_, 'str> {
    type Target = FunctionBody<'str>;

    fn deref(&self) -> &Self::Target {
        self.body
    }
}

impl<'str> DerefMut for BodyLoan<'_, 'str> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.body
    }
}

impl Drop for BodyLoan<'_, '_> {
    fn drop(&mut self) {
        if self.body.try_id() != Some(self.slot) {
            // Whatever is in the slot now is not a body of this function: the
            // ids stored in it name another function's arenas, or none. The
            // shape moved too, but no rebuild recovers a registry whose slots
            // do not hold their own functions.
            self.poisoned.store(true, Ordering::Relaxed);
            self.body.link_clock(self.clock.clone());
            self.clock.tick();
            return;
        }
        if self.body.identity() != self.lent {
            // Another value of this function: swapped in from another module,
            // moved back in after a replacement, or a clone. Its addresses are
            // whatever they are; from here on they tick this module.
            self.body.link_clock(self.clock.clone());
            self.clock.tick();
        }
    }
}

/// The bodies of a module, borrowed for a driver that lends several out at
/// once. See the [module documentation](self).
///
/// Reads index the registry as usual; mutable access is by
/// [loan](Self::get_mut), one body or a [disjoint set](Self::select_mut) at a
/// time. There is no way to add, remove or replace a slot: the registry stays
/// in lockstep with the module's interfaces, and every body in it stays the
/// module's.
pub struct BodiesMut<'a, 'str> {
    bodies: &'a mut Registry<FunctionId, FunctionBody<'str>>,
    clock: &'a ShapeClock,
    poisoned: &'a AtomicBool,
}

impl<'a, 'str> BodiesMut<'a, 'str> {
    pub(crate) fn new(
        bodies: &'a mut Registry<FunctionId, FunctionBody<'str>>,
        clock: &'a ShapeClock,
        poisoned: &'a AtomicBool,
    ) -> Self {
        Self {
            bodies,
            clock,
            poisoned,
        }
    }

    /// The number of functions in the module.
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// The body of `f`, for reading.
    pub fn get(&self, f: FunctionId) -> &FunctionBody<'str> {
        &self.bodies[f]
    }

    /// Every body with its id, in registry order.
    pub fn iter(&self) -> registry::Iter<'_, FunctionId, FunctionBody<'str>> {
        self.bodies.iter()
    }

    /// Lends the body of `f` out exclusively, until the loan is dropped.
    pub fn get_mut(&mut self, f: FunctionId) -> BodyLoan<'_, 'str> {
        BodyLoan::new(&mut self.bodies[f], f, self.clock, self.poisoned)
    }

    /// Lends the bodies of `ids` out at once, in the order given. Panics on
    /// a repeated or unknown id, as [`Registry::select_mut`] does.
    pub fn select_mut(&mut self, ids: &[FunctionId]) -> Vec<BodyLoan<'_, 'str>> {
        let clock = self.clock;
        let poisoned = self.poisoned;
        self.bodies
            .select_mut(ids)
            .into_iter()
            .zip(ids)
            .map(|(body, &f)| BodyLoan::new(body, f, clock, poisoned))
            .collect()
    }
}

impl<'str> Index<FunctionId> for BodiesMut<'_, 'str> {
    type Output = FunctionBody<'str>;

    fn index(&self, f: FunctionId) -> &Self::Output {
        &self.bodies[f]
    }
}

impl<'a, 'str> IntoIterator for &'a BodiesMut<'_, 'str> {
    type Item = Identified<FunctionId, &'a FunctionBody<'str>>;
    type IntoIter = registry::Iter<'a, FunctionId, FunctionBody<'str>>;

    fn into_iter(self) -> Self::IntoIter {
        self.bodies.iter()
    }
}
