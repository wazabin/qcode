//! Append-only interners for constants (literals and byte blobs), wrapped in an
//! `RwLock` so they can be **minted through a shared `&` reference** — the
//! prerequisite for a function pass creating a constant while it holds only
//! `&ModuleView`. Interned ids are globally stable and never remapped.
//!
//! Reads (`Index`) return a `&T` that outlives the read guard. This is sound
//! because the backing [`Registry`] is *address-stable*: it never moves an
//! element once pushed (see `jstd::registry::Registry`), so the reference stays
//! valid for the life of the interner even as later mints grow it. The read lock
//! only guards the brief indexing.

use std::ops::{Index, IndexMut};
use std::sync::RwLock;

use jstd::registry::{Identified, Identifier, Registry};
use rustc_hash::FxHashMap as HashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    types::TypeId,
    value::literal::{Literal, LiteralId},
};

/// Mask `value` to `size` bytes (the low `size*8` bits), matching the width a
/// literal of that size can hold.
fn mask_to_size(value: u64, size: usize) -> u64 {
    if size >= 8 {
        value
    } else {
        value & ((1u64 << (size * 8)) - 1)
    }
}

/// A generic append-only interner: an address-stable [`Registry`] behind an
/// `RwLock`. Pushes take a write lock; reads (`Index`) take a read lock and hand
/// back a stable `&T`.
pub struct Interner<Id: Identifier, T> {
    inner: RwLock<Registry<Id, T>>,
}

impl<Id: Identifier, T> Default for Interner<Id, T> {
    fn default() -> Self {
        Self {
            inner: RwLock::new(Registry::default()),
        }
    }
}

impl<Id: Identifier, T: Clone> Clone for Interner<Id, T> {
    fn clone(&self) -> Self {
        Self {
            inner: RwLock::new(self.read().clone()),
        }
    }
}

impl<Id: Identifier, T> Interner<Id, T> {
    fn read(&self) -> std::sync::RwLockReadGuard<'_, Registry<Id, T>> {
        self.inner.read().expect("interner RwLock poisoned")
    }

    /// Appends `value`, returning its stable id. Takes a write lock.
    pub fn push(&self, value: T) -> Id {
        self.inner
            .write()
            .expect("interner RwLock poisoned")
            .push(value)
    }

    /// The number of interned values.
    pub fn len(&self) -> usize {
        self.read().len()
    }

    /// Whether the interner is empty.
    pub fn is_empty(&self) -> bool {
        self.read().is_empty()
    }
}

impl<Id: Identifier, T: Clone> Interner<Id, T> {
    /// An owned snapshot of all `(id, value)` pairs in id order. Owned (cloned)
    /// so it does not borrow through the lock; callers that only need ids or a
    /// stable `&T` should prefer indexing.
    pub fn iter(&self) -> impl Iterator<Item = Identified<Id, T>> {
        self.read()
            .iter()
            .map(|item| Identified::new(item.id, item.inner.clone()))
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<Id: Identifier, T> Index<Id> for Interner<Id, T> {
    type Output = T;

    fn index(&self, id: Id) -> &T {
        let ptr: *const T = {
            let reg = self.read();
            &reg[id] as *const T
        };
        // SAFETY: the backing registry is address-stable (elements never move
        // once pushed), so `ptr` is valid for the life of `self`; the borrow is
        // tied to `&self` here.
        unsafe { &*ptr }
    }
}

impl<Id: Identifier, T> IndexMut<Id> for Interner<Id, T> {
    fn index_mut(&mut self, id: Id) -> &mut T {
        // Exclusive access via `&mut self`: no lock needed, and no stability
        // trick — this is the plain in-place mutation path for interned values
        // that are edited right after creation (e.g. a byte blob's element type).
        &mut self.inner.get_mut().expect("interner RwLock poisoned")[id]
    }
}

impl<Id: Identifier, T: Serialize> Serialize for Interner<Id, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.read().serialize(serializer)
    }
}

impl<'de, Id: Identifier, T: Deserialize<'de>> Deserialize<'de> for Interner<Id, T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self {
            inner: RwLock::new(Registry::deserialize(deserializer)?),
        })
    }
}

/// The literal interner: an address-stable literal [`Registry`] plus its
/// dedup cache `(masked value, type) → LiteralId`, together behind one `RwLock`
/// so the check-or-mint is atomic. Reads (`Index`) hand back a stable `&Literal`.
#[derive(Default)]
pub struct LiteralInterner {
    inner: RwLock<LiteralPool>,
}

#[derive(Default, Clone)]
struct LiteralPool {
    literals: Registry<LiteralId, Literal>,
    /// Intern cache for non-symbolic literals. Derived from `literals`; not
    /// serialized (rebuilt on load).
    cache: HashMap<(u64, TypeId), LiteralId>,
}

impl Clone for LiteralInterner {
    fn clone(&self) -> Self {
        Self {
            inner: RwLock::new(self.read().clone()),
        }
    }
}

impl LiteralInterner {
    fn read(&self) -> std::sync::RwLockReadGuard<'_, LiteralPool> {
        self.inner.read().expect("literal interner RwLock poisoned")
    }

    /// Returns a canonical [`LiteralId`] for the given typed constant, minting
    /// (and caching) it if new. The value is masked to `type_id`'s size before
    /// lookup; symbolic literals (from [`push_literal`](Self::push_literal)) are
    /// not cached and never alias with constants produced here.
    pub fn get_or_make_typed_literal(&self, value: u64, type_id: TypeId, size: usize) -> LiteralId {
        let value = mask_to_size(value, size);
        // Hot path: cache hit under a read lock.
        if let Some(&id) = self.read().cache.get(&(value, type_id)) {
            return id;
        }
        // Miss: take the write lock and re-check before minting.
        let mut pool = self
            .inner
            .write()
            .expect("literal interner RwLock poisoned");
        if let Some(&id) = pool.cache.get(&(value, type_id)) {
            return id;
        }
        let id = pool.literals.push(Literal {
            value,
            type_id,
            symbolic: None,
        });
        pool.cache.insert((value, type_id), id);
        id
    }

    /// Pushes a [`Literal`] with arbitrary fields (e.g. a symbolic ref) without
    /// interning.
    pub fn push_literal(&self, literal: Literal) -> LiteralId {
        self.inner
            .write()
            .expect("literal interner RwLock poisoned")
            .literals
            .push(literal)
    }

    /// The number of interned literals.
    pub fn len(&self) -> usize {
        self.read().literals.len()
    }

    /// Whether the interner is empty.
    pub fn is_empty(&self) -> bool {
        self.read().literals.is_empty()
    }

    /// An owned snapshot of all `(id, literal)` pairs in id order. Owned (cloned)
    /// so it does not borrow through the lock — used by the post-lift address /
    /// string resolvers, which then mutate literals via [`IndexMut`].
    pub fn iter(&self) -> impl Iterator<Item = Identified<LiteralId, Literal>> {
        self.read()
            .literals
            .iter()
            .map(|item| Identified::new(item.id, item.inner.clone()))
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl Index<LiteralId> for LiteralInterner {
    type Output = Literal;

    fn index(&self, id: LiteralId) -> &Literal {
        let ptr: *const Literal = {
            let pool = self.read();
            &pool.literals[id] as *const Literal
        };
        // SAFETY: the literal registry is address-stable (see `Interner::index`).
        unsafe { &*ptr }
    }
}

impl IndexMut<LiteralId> for LiteralInterner {
    fn index_mut(&mut self, id: LiteralId) -> &mut Literal {
        // Exclusive `&mut self` access (used to attach a symbolic ref to an
        // existing literal after lifting); no lock or stability trick needed. The
        // dedup cache still points at this id — attaching a symbol does not change
        // the `(value, type)` key it was interned under.
        &mut self
            .inner
            .get_mut()
            .expect("literal interner RwLock poisoned")
            .literals[id]
    }
}

impl Serialize for LiteralInterner {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Serialize just the literals (a flat sequence, unchanged wire format);
        // the cache is derived data and is rebuilt on load.
        self.read().literals.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for LiteralInterner {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let literals = Registry::<LiteralId, Literal>::deserialize(deserializer)?;
        // Rebuild the dedup cache so post-load minting reuses existing literals
        // instead of creating duplicates. Non-symbolic literals only, first wins.
        let mut cache: HashMap<(u64, TypeId), LiteralId> = HashMap::default();
        for item in literals.iter() {
            if item.inner.symbolic.is_none() {
                cache
                    .entry((item.inner.value, item.inner.type_id))
                    .or_insert(item.id);
            }
        }
        Ok(Self {
            inner: RwLock::new(LiteralPool { literals, cache }),
        })
    }
}
