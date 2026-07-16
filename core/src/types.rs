//! First-class type system for qcode IR values.
//!
//! Every value (literal, instruction result, block param) carries a [`TypeId`]
//! that encodes both its size and its semantic kind. Types are interned once in
//! a [`TypeManager`] attached to the [`Context`]; all passes use [`TypeId`] as
//! a lightweight `Copy` handle.
//!
//! # Type taxonomy
//!
//! | Concrete type    | Meaning                                        |
//! |------------------|------------------------------------------------|
//! | [`IntType`]      | Plain integer of *n* bytes                     |
//! | [`BoolType`]     | A byte-stored boolean, domain `{0, 1}`         |
//! | [`StackAddress`] | Pointer-width address in the stack memory space |
//!
//! # Bool
//!
//! `bool` is its own type (byte-stored, `size() == 1`) minted only by
//! comparisons and the `true`/`false` literals. The verifier pins its domain to
//! `{0, 1}` and rejects mixing `bool` with `iN` in a binop, so bitwise
//! `And`/`Or`/`Xor` over `bool` operands *is* logical and/or/xor.
//!
//! # TODO
//!
//! - Pointer types for RAM/register spaces.

use std::sync::{
    Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard,
    atomic::{AtomicPtr, Ordering},
};

use rustc_hash::FxHashMap as HashMap;

use crate::{
    space::MemorySpaceId,
    value::insn::{Binop, IntBinop},
};

// ---------------------------------------------------------------------------
// TypeId
// ---------------------------------------------------------------------------

/// A lightweight handle to a type registered in [`TypeManager`].
///
/// `TypeId` is `Copy + Hash + Eq` and carries no context borrow. Convert to a
/// concrete [`Type`] via [`TypeManager::get`].
#[derive(
    Copy, Clone, Hash, Eq, PartialEq, Debug, Ord, PartialOrd, serde::Serialize, serde::Deserialize,
)]
pub struct TypeId(u32);

// ---------------------------------------------------------------------------
// Type trait
// ---------------------------------------------------------------------------

pub trait Type: Send + Sync {
    /// Width of values of this type in bytes.
    fn size(&self) -> usize;

    /// The memory space this type lives in, if it is a pointer type.
    fn space(&self) -> Option<MemorySpaceId> {
        None
    }

    /// The ordered field types, if this is an [`AggregateType`] or nominal
    /// [`StructType`].
    fn fields(&self) -> Option<&[AggregateField]> {
        None
    }

    /// The name of this type, if it is a nominal [`StructType`].
    fn struct_name(&self) -> Option<&str> {
        None
    }

    /// The pointee type, if this is a [`StructPointer`].
    fn pointee(&self) -> Option<TypeId> {
        None
    }

    /// The `(elem, count)` pair, if this is an [`ArrayType`]. Returns `None` for
    /// every other type — this is the *only* discriminator element-aware code
    /// uses to tell an array from the width-N scalar it otherwise looks like.
    fn array(&self) -> Option<(TypeId, usize)> {
        None
    }

    /// The `(elem, bound)` pair, if this is a [`ListType`] — a variable-length
    /// sequence whose `bound` is the static element upper bound (`Some(n)`) or
    /// `None` when unbounded (a pointer-sourced string). Returns `None` (the outer
    /// option) for every non-list type. This is the discriminator that tells a
    /// *list* (`take_while`'s result) from a fixed-length [`array`](Type::array):
    /// both look like width-N scalars structurally, but a list's length is not
    /// statically known.
    fn list(&self) -> Option<(TypeId, Option<usize>)> {
        None
    }

    /// Clones this type into a fresh boxed trait object.
    ///
    /// This enables `Clone for Box<dyn Type>` (and hence `Clone` for
    /// [`TypeManager`] and [`Context`](crate::context::Context)), which the GUI
    /// relies on to fork a context before running an analysis pipeline.
    fn clone_box(&self) -> Box<dyn Type>;

    /// Describes this type in a flat, serializable form.
    ///
    /// Used to persist the [`TypeManager`] across a saved session: trait objects
    /// cannot be serialized directly, so each type reports a [`TypeRepr`] from
    /// which it can be reconstructed.
    fn repr(&self) -> TypeRepr;
}

/// Serializable description of a concrete [`Type`].
///
/// There are only three concrete types, each fully described by a byte width and
/// (for pointers) the memory space it points into. [`TypeManager`] serializes its
/// type table as a `Vec<TypeRepr>` and replays the `get_or_make_*` constructors
/// on load, which reproduces both the interned [`TypeId`] indices and the lookup
/// maps exactly.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub enum TypeRepr {
    Int {
        size: usize,
    },
    /// A byte-stored boolean whose value domain is `{0, 1}`. Minted only by
    /// comparisons and the `true`/`false` literals; `size()` is always 1.
    Bool,
    SpaceAddress {
        size: usize,
        space: MemorySpaceId,
    },
    /// A fixed, ordered group of named field types — the functional-IR
    /// representation of a tuple. Used by `argpromote` to return
    /// `(real_return, write-set)`. Abstract: it has no physical return-register
    /// ABI.
    Aggregate {
        fields: Vec<AggregateField>,
    },
    /// A named, nominal struct with explicit per-field byte offsets — the
    /// pointee of a [`StructPointer`]. Identity is the `name`, not the field
    /// list, so two structs with coincident layouts stay distinct. Sparse: only
    /// the fields of interest are listed; `size` is the real struct size and
    /// need not equal the fields' extent.
    Struct {
        name: String,
        size: usize,
        fields: Vec<AggregateField>,
    },
    /// A pointer to a nominal [`Struct`](TypeRepr::Struct) (or any other type),
    /// of the given byte width. `pointee` is the [`TypeId`] it points at.
    StructPointer {
        size: usize,
        pointee: TypeId,
    },
    /// A fixed-length homogeneous array of `count` elements of type `elem`,
    /// laid out contiguously. Its byte width is `count * sizeof(elem)`.
    ///
    /// Deliberately **disguised as a width-N scalar**: it answers
    /// [`Type::size`] like any integer and does *not* expose [`Type::fields`],
    /// so structural passes (mem2reg, alias, DCE, GVN value-numbering) handle it
    /// unchanged. Only element-aware sites (`argpromote`, the `Extract`/`Range`
    /// over `Map` rewrite, emulation) consult [`Type::array`]. Lane projection is
    /// defined as a contiguous bit-slice: `Extract(arr, k) ≡ Range(arr,
    /// k*sizeof(elem), sizeof(elem))`.
    Array {
        elem: TypeId,
        count: usize,
    },
    /// A variable-length homogeneous sequence of `elem` — the result of
    /// [`take_while`](crate::intrinsics). `bound` is the static storage upper bound
    /// in elements (`Some(n)` for a `take_while` over a fixed `[T; n]` array), or
    /// `None` when the source is an unbounded pointer (a `char*` string of unknown
    /// length). Like [`Array`](TypeRepr::Array) a *bounded* list is disguised as a
    /// width-N scalar (its `size` is the `bound` footprint); an *unbounded* list has
    /// no materialized footprint (`size` 0) — it is a handle consumed only by
    /// `len`/`map`, never stored. Only [`Type::list`] tells either apart from a
    /// fixed array; the runtime length is the position of the first failing element.
    List {
        elem: TypeId,
        bound: Option<usize>,
    },
}

/// One field of an aggregate or struct type.
///
/// Field names are part of aggregate identity. For structural aggregates the
/// slots are addressed by numeric index and `offset` is informational (the
/// running byte sum); for nominal [`StructType`]s `offset` is the field's real
/// byte offset and is the key a [`Gep`](crate::value::insn::Gep) resolves on.
#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AggregateField {
    pub name: String,
    pub type_id: TypeId,
    /// Byte offset of this field within its containing aggregate/struct.
    pub offset: usize,
}

impl AggregateField {
    /// Field with offset `0`. Used by structural aggregates, where the slot is
    /// addressed by index and the offset is not consulted.
    pub fn new(name: impl Into<String>, type_id: TypeId) -> Self {
        Self::new_at(name, type_id, 0)
    }

    /// Field at an explicit byte `offset`. Used by nominal [`StructType`]s.
    pub fn new_at(name: impl Into<String>, type_id: TypeId, offset: usize) -> Self {
        Self {
            name: name.into(),
            type_id,
            offset,
        }
    }
}

impl Clone for Box<dyn Type> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// ---------------------------------------------------------------------------
// Concrete types
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct IntType {
    size: usize,
}

impl Type for IntType {
    fn size(&self) -> usize {
        self.size
    }

    fn clone_box(&self) -> Box<dyn Type> {
        Box::new(self.clone())
    }

    fn repr(&self) -> TypeRepr {
        TypeRepr::Int { size: self.size }
    }
}

/// A byte-stored boolean, value domain `{0, 1}`. `size()` is always 1 (the type
/// system is byte-granular). Distinct from `Int(1)` so the verifier can reject
/// `bool`/`iN` mixing and so bitwise ops over `bool` read as logical and/or/xor.
#[derive(Clone)]
struct BoolType;

impl Type for BoolType {
    fn size(&self) -> usize {
        1
    }

    fn clone_box(&self) -> Box<dyn Type> {
        Box::new(self.clone())
    }

    fn repr(&self) -> TypeRepr {
        TypeRepr::Bool
    }
}

/// A pointer-typed value carrying the memory space it points into.
///
/// This represents arbitrary space provenance — e.g. the result of `&A + offset`,
/// which points into the space
/// of varnode `A`. It is the type-system encoding of the address-space tag that
/// pointer-producing instructions used to carry as a separate field. Its byte
/// width is the producing instruction's width (not necessarily the space's
/// address size), matching the operand-derived size of pointer arithmetic.
#[derive(Clone)]
pub struct SpaceAddress {
    size: usize,
    space: MemorySpaceId,
}

impl Type for SpaceAddress {
    fn size(&self) -> usize {
        self.size
    }

    fn space(&self) -> Option<MemorySpaceId> {
        Some(self.space)
    }

    fn clone_box(&self) -> Box<dyn Type> {
        Box::new(self.clone())
    }

    fn repr(&self) -> TypeRepr {
        TypeRepr::SpaceAddress {
            size: self.size,
            space: self.space,
        }
    }
}

/// A fixed, ordered group of field types — the functional-IR tuple. Its `size`
/// is the sum of its fields' sizes (a nominal layout; aggregates are abstract and
/// never lowered to a physical ABI, so the value is informational only).
#[derive(Clone)]
struct AggregateType {
    fields: Vec<AggregateField>,
    size: usize,
}

impl Type for AggregateType {
    fn size(&self) -> usize {
        self.size
    }

    fn fields(&self) -> Option<&[AggregateField]> {
        Some(&self.fields)
    }

    fn clone_box(&self) -> Box<dyn Type> {
        Box::new(self.clone())
    }

    fn repr(&self) -> TypeRepr {
        TypeRepr::Aggregate {
            fields: self.fields.clone(),
        }
    }
}

/// A named, nominal struct with explicit per-field byte offsets.
///
/// Unlike [`AggregateType`], identity is the **name** (not the field list), so
/// two structs that happen to share a layout stay distinct types. Field lists
/// are **sparse** — only the fields of interest are recorded — and `size` is the
/// real struct size, which need not equal the fields' extent. This is the
/// pointee of a [`StructPointer`] and the type a
/// [`Gep`](crate::value::insn::Gep) resolves field offsets against.
#[derive(Clone)]
struct StructType {
    name: String,
    fields: Vec<AggregateField>,
    size: usize,
}

impl Type for StructType {
    fn size(&self) -> usize {
        self.size
    }

    fn fields(&self) -> Option<&[AggregateField]> {
        Some(&self.fields)
    }

    fn struct_name(&self) -> Option<&str> {
        Some(&self.name)
    }

    fn clone_box(&self) -> Box<dyn Type> {
        Box::new(self.clone())
    }

    fn repr(&self) -> TypeRepr {
        TypeRepr::Struct {
            name: self.name.clone(),
            size: self.size,
            fields: self.fields.clone(),
        }
    }
}

/// A pointer of a given byte width pointing at `pointee` (typically a nominal
/// [`StructType`]). Carries the pointee identity so a chain of
/// [`Gep`](crate::value::insn::Gep) + `load` can resolve successive fields.
#[derive(Clone)]
struct StructPointer {
    size: usize,
    pointee: TypeId,
}

impl Type for StructPointer {
    fn size(&self) -> usize {
        self.size
    }

    fn pointee(&self) -> Option<TypeId> {
        Some(self.pointee)
    }

    fn clone_box(&self) -> Box<dyn Type> {
        Box::new(self.clone())
    }

    fn repr(&self) -> TypeRepr {
        TypeRepr::StructPointer {
            size: self.size,
            pointee: self.pointee,
        }
    }
}

/// A fixed-length homogeneous array — see [`TypeRepr::Array`]. `size` is cached
/// as `count * sizeof(elem)`; the array is opaque (no `fields()`) so it presents
/// to structural passes exactly as a width-`size` integer would.
#[derive(Clone)]
struct ArrayType {
    elem: TypeId,
    count: usize,
    size: usize,
}

impl Type for ArrayType {
    fn size(&self) -> usize {
        self.size
    }

    fn array(&self) -> Option<(TypeId, usize)> {
        Some((self.elem, self.count))
    }

    fn clone_box(&self) -> Box<dyn Type> {
        Box::new(self.clone())
    }

    fn repr(&self) -> TypeRepr {
        TypeRepr::Array {
            elem: self.elem,
            count: self.count,
        }
    }
}

/// A variable-length homogeneous sequence — see [`TypeRepr::List`]. `bound` is the
/// static element upper bound (`Some`) or `None` when unbounded; `size` is the
/// `bound`-element footprint for a bounded list and 0 for an unbounded one (it has
/// no materialized storage). Structurally a bounded list is indistinguishable from
/// a width-`size` scalar; only [`Type::list`] recovers `(elem, bound)`.
#[derive(Clone)]
struct ListType {
    elem: TypeId,
    bound: Option<usize>,
    size: usize,
}

impl Type for ListType {
    fn size(&self) -> usize {
        self.size
    }

    fn list(&self) -> Option<(TypeId, Option<usize>)> {
        Some((self.elem, self.bound))
    }

    fn clone_box(&self) -> Box<dyn Type> {
        Box::new(self.clone())
    }

    fn repr(&self) -> TypeRepr {
        TypeRepr::List {
            elem: self.elem,
            bound: self.bound,
        }
    }
}

// ---------------------------------------------------------------------------
// TypeManager
// ---------------------------------------------------------------------------

/// The default `field1`, `field2`, ... naming applied when an aggregate is
/// built from bare types. Shared by the interner's hit-path probe and the
/// mint path so both key the cache identically.
fn default_named_fields(fields: Vec<TypeId>) -> Vec<AggregateField> {
    fields
        .into_iter()
        .enumerate()
        .map(|(i, type_id)| AggregateField::new(format!("field{}", i + 1), type_id))
        .collect()
}

/// Registry that owns all [`Type`] objects and hands out interned [`TypeId`]s.
///
/// Types are created once (at context construction or when the architecture is
/// configured) and never removed. All methods that *only read* types take
/// `&self`; methods that may create new `Int` types on demand take `&mut self`.
#[derive(Clone)]
struct TypeManagerInner {
    types: Vec<Box<dyn Type>>,
    /// Fast lookup: Int size → TypeId.
    int_by_size: HashMap<usize, TypeId>,
    /// The interned `bool` type, once created.
    bool_id: Option<TypeId>,
    /// Fast lookup: (size, space) → SpaceAddress TypeId.
    space_address: HashMap<(usize, MemorySpaceId), TypeId>,
    /// Fast lookup: named field-type list → Aggregate TypeId.
    aggregate_by_fields: HashMap<Vec<AggregateField>, TypeId>,
    /// Nominal lookup: struct name → StructType TypeId.
    struct_by_name: HashMap<String, TypeId>,
    /// Fast lookup: (size, pointee) → StructPointer TypeId.
    struct_pointer: HashMap<(usize, TypeId), TypeId>,
    /// Fast lookup: (elem, count) → Array TypeId.
    array_by_elem_count: HashMap<(TypeId, usize), TypeId>,
    /// Fast lookup: (elem, bound) → List TypeId (`bound` `None` = unbounded).
    list_by_elem_bound: HashMap<(TypeId, Option<usize>), TypeId>,
}

impl Default for TypeManagerInner {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeManagerInner {
    fn new() -> Self {
        Self {
            types: Vec::new(),
            int_by_size: HashMap::default(),
            bool_id: None,
            space_address: HashMap::default(),
            aggregate_by_fields: HashMap::default(),
            struct_by_name: HashMap::default(),
            struct_pointer: HashMap::default(),
            array_by_elem_count: HashMap::default(),
            list_by_elem_bound: HashMap::default(),
        }
    }

    fn register(&mut self, ty: Box<dyn Type>) -> TypeId {
        let id = TypeId(self.types.len() as u32);
        self.types.push(ty);
        id
    }

    /// Returns the [`TypeId`] for `Int(size)`, creating the type if it does not
    /// yet exist.
    pub fn get_or_make_int(&mut self, size: usize) -> TypeId {
        if let Some(&id) = self.int_by_size.get(&size) {
            return id;
        }
        let id = self.register(Box::new(IntType { size }));
        self.int_by_size.insert(size, id);
        id
    }

    /// Returns the [`TypeId`] for the byte-stored `bool` type, creating it if it
    /// does not yet exist.
    pub fn get_or_make_bool(&mut self) -> TypeId {
        if let Some(id) = self.bool_id {
            return id;
        }
        let id = self.register(Box::new(BoolType));
        self.bool_id = Some(id);
        id
    }

    /// The interned `bool` [`TypeId`], if it has been created.
    pub fn bool_id(&self) -> Option<TypeId> {
        self.bool_id
    }

    /// Whether `id` is the byte-stored `bool` type.
    pub fn is_bool(&self, id: TypeId) -> bool {
        matches!(self.get(id).repr(), TypeRepr::Bool)
    }

    /// Returns the [`TypeId`] for a [`SpaceAddress`] of the given byte width
    /// pointing into `space`, creating it if it does not yet exist.
    pub fn get_or_make_space_address(&mut self, size: usize, space: MemorySpaceId) -> TypeId {
        if let Some(&id) = self.space_address.get(&(size, space)) {
            return id;
        }
        let id = self.register(Box::new(SpaceAddress { size, space }));
        self.space_address.insert((size, space), id);
        id
    }

    /// Returns the [`TypeId`] for an [`AggregateType`] with the given ordered,
    /// named fields, creating it if it does not yet exist. Each field type must
    /// already be registered (it always is in practice: you build the field
    /// types before grouping them).
    pub fn get_or_make_named_aggregate(&mut self, fields: Vec<AggregateField>) -> TypeId {
        for (i, field) in fields.iter().enumerate() {
            assert!(
                !fields[..i].iter().any(|prev| prev.name == field.name),
                "aggregate field names must be unique"
            );
        }
        if let Some(&id) = self.aggregate_by_fields.get(&fields) {
            return id;
        }
        let size = fields.iter().map(|f| self.size_of(f.type_id)).sum();
        let id = self.register(Box::new(AggregateType {
            fields: fields.clone(),
            size,
        }));
        self.aggregate_by_fields.insert(fields, id);
        id
    }

    /// Returns the nominal [`StructType`] named `name`, creating it if it does
    /// not yet exist. Identity is the name: a second call with the same `name`
    /// returns the original `TypeId` and **ignores** `size`/`fields`.
    pub fn get_or_make_struct(
        &mut self,
        name: impl Into<String>,
        size: usize,
        fields: Vec<AggregateField>,
    ) -> TypeId {
        let name = name.into();
        if let Some(&id) = self.struct_by_name.get(&name) {
            return id;
        }
        let id = self.register(Box::new(StructType {
            name: name.clone(),
            fields,
            size,
        }));
        self.struct_by_name.insert(name, id);
        id
    }

    /// The [`TypeId`] of the nominal struct named `name`, if registered.
    pub fn struct_by_name(&self, name: &str) -> Option<TypeId> {
        self.struct_by_name.get(name).copied()
    }

    /// Returns the [`TypeId`] for a [`StructPointer`] of the given byte width
    /// pointing at `pointee`, creating it if it does not yet exist.
    pub fn get_or_make_struct_pointer(&mut self, size: usize, pointee: TypeId) -> TypeId {
        if let Some(&id) = self.struct_pointer.get(&(size, pointee)) {
            return id;
        }
        let id = self.register(Box::new(StructPointer { size, pointee }));
        self.struct_pointer.insert((size, pointee), id);
        id
    }

    /// Returns the [`TypeId`] for an [`ArrayType`] of `count` elements of type
    /// `elem`, creating it if it does not yet exist. `elem` must already be
    /// registered (it always is: you build the element type first).
    pub fn get_or_make_array(&mut self, elem: TypeId, count: usize) -> TypeId {
        if let Some(&id) = self.array_by_elem_count.get(&(elem, count)) {
            return id;
        }
        let size = self.size_of(elem) * count;
        let id = self.register(Box::new(ArrayType { elem, count, size }));
        self.array_by_elem_count.insert((elem, count), id);
        id
    }

    /// Returns the [`TypeId`] for a *bounded* [`ListType`] — a variable-length
    /// sequence of at most `bound` elements of type `elem` — creating it if it does
    /// not yet exist. `elem` must already be registered. For a pointer-sourced
    /// string of unknown length, see [`get_or_make_unbounded_list`].
    ///
    /// [`get_or_make_unbounded_list`]: Self::get_or_make_unbounded_list
    pub fn get_or_make_list(&mut self, elem: TypeId, bound: usize) -> TypeId {
        self.get_or_make_list_opt(elem, Some(bound))
    }

    /// Returns the [`TypeId`] for an *unbounded* [`ListType`] of `elem` — a string
    /// of unknown length (a `char*`), with no static footprint (`size` 0).
    pub fn get_or_make_unbounded_list(&mut self, elem: TypeId) -> TypeId {
        self.get_or_make_list_opt(elem, None)
    }

    /// Shared constructor for bounded (`Some`) and unbounded (`None`) lists.
    fn get_or_make_list_opt(&mut self, elem: TypeId, bound: Option<usize>) -> TypeId {
        if let Some(&id) = self.list_by_elem_bound.get(&(elem, bound)) {
            return id;
        }
        // A bounded list footprints its `bound` elements; an unbounded one is a
        // handle with no materialized storage (size 0).
        let size = bound.map_or(0, |b| self.size_of(elem) * b);
        let id = self.register(Box::new(ListType { elem, bound, size }));
        self.list_by_elem_bound.insert((elem, bound), id);
        id
    }

    /// Returns a reference to the concrete [`Type`] for `id`.
    pub fn get(&self, id: TypeId) -> &dyn Type {
        &*self.types[id.0 as usize]
    }

    /// Returns the byte width of values with type `id`.
    pub fn size_of(&self, id: TypeId) -> usize {
        self.get(id).size()
    }

    /// Computes the result [`TypeId`] for a binary operation on `lhs op rhs`.
    ///
    /// Comparisons yield `bool`; `And`/`Or`/`Xor` over `bool` operands stay `bool`
    /// (this *is* logical and/or/xor); every other integer/float op preserves the
    /// left operand's type (so a pointer-typed operand keeps its space provenance
    /// through `ptr + offset`).
    pub fn binop_result(&mut self, lhs: TypeId, op: Binop, rhs: TypeId) -> TypeId {
        // `None` only when the result is `bool` and `bool` isn't interned yet.
        if let Some(id) = self.binop_result_probe(lhs, op, rhs) {
            id
        } else {
            self.get_or_make_bool()
        }
    }

    /// The read-only arm of [`binop_result`](Self::binop_result): resolves the
    /// result type without minting, returning `None` exactly when the result is
    /// `bool` and `bool` has not been interned yet (the caller then mints it).
    fn binop_result_probe(&self, lhs: TypeId, op: Binop, _rhs: TypeId) -> Option<TypeId> {
        match op {
            Binop::Int(int_op) => match int_op {
                IntBinop::Equal
                | IntBinop::NotEqual
                | IntBinop::Less
                | IntBinop::LessEqual
                | IntBinop::SLess
                | IntBinop::SLessEqual => self.bool_id,
                // Bitwise and/or/xor over bool operands is logical and/or/xor and
                // preserves the bool type; over ints it preserves the int type.
                IntBinop::And | IntBinop::Or | IntBinop::Xor if self.is_bool(lhs) => Some(lhs),
                _ => Some(lhs),
            },
            Binop::Float(float_op) => {
                if float_op.is_comparison() {
                    self.bool_id
                } else {
                    Some(lhs)
                }
            }
        }
    }
}

/// The type interner: a global, append-only table of interned [`Type`]s behind a
/// [`RwLock`] so that types can be minted through a shared `&` reference (a
/// prerequisite for running function passes in parallel against a shared
/// `ContextView`). Reads — including the `get_or_make_*` hit path — take a read
/// lock; only a cache miss takes the write lock (and re-checks under it).
/// Interned [`TypeId`]s are globally stable and never remapped.
pub struct TypeManager {
    inner: RwLock<TypeManagerInner>,
    /// Lock-free read index over the append-only type objects in `inner`.
    ///
    /// Publishing replaces this pointer after a successful mint. Old indexes
    /// stay owned by `published_generations`, so a reader that raced with a
    /// publication can safely finish through the generation it loaded. The
    /// pointed-to `Type` objects themselves live in `inner.types`; those are
    /// boxed, append-only, and therefore never move or disappear.
    published: AtomicPtr<PublishedTypes>,
    // Each generation needs its own stable heap address after this Vec grows.
    #[allow(clippy::vec_box)]
    published_generations: Mutex<Vec<Box<PublishedTypes>>>,
}

/// One immutable generation of the lock-free TypeId -> Type pointer index.
///
/// The raw trait-object pointers target `Box<dyn Type>` pointees owned by the
/// corresponding [`TypeManagerInner`]. They are immutable, `Send + Sync`, and
/// remain allocated for the manager's entire lifetime. A generation is never
/// modified after publication.
struct PublishedTypes {
    entries: Box<[*const dyn Type]>,
}

// SAFETY: every entry points to an immutable `dyn Type + Send + Sync` allocation
// owned for the full lifetime of the enclosing TypeManager. PublishedTypes never
// mutates an entry or the pointee after construction.
unsafe impl Send for PublishedTypes {}
// SAFETY: see the `Send` implementation above; concurrent access is read-only.
unsafe impl Sync for PublishedTypes {}

impl PublishedTypes {
    fn from_inner(inner: &TypeManagerInner) -> Self {
        Self {
            entries: inner
                .types
                .iter()
                .map(|ty| &**ty as *const dyn Type)
                .collect(),
        }
    }
}

impl Default for TypeManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for TypeManager {
    fn clone(&self) -> Self {
        Self::from_inner(self.read().clone())
    }
}

impl TypeManager {
    pub fn new() -> Self {
        Self::from_inner(TypeManagerInner::new())
    }

    fn from_inner(inner: TypeManagerInner) -> Self {
        let generation = Box::new(PublishedTypes::from_inner(&inner));
        let published = AtomicPtr::new((&*generation as *const PublishedTypes).cast_mut());
        Self {
            inner: RwLock::new(inner),
            published,
            published_generations: Mutex::new(vec![generation]),
        }
    }

    fn read(&self) -> RwLockReadGuard<'_, TypeManagerInner> {
        self.inner.read().expect("type manager RwLock poisoned")
    }

    fn write(&self) -> RwLockWriteGuard<'_, TypeManagerInner> {
        self.inner.write().expect("type manager RwLock poisoned")
    }

    /// Publish the current append-only type table for lock-free readers.
    /// Caller holds the write lock, so only one generation can be constructed
    /// at a time and every registered type is fully initialized first.
    fn publish(&self, inner: &TypeManagerInner) {
        let generation = Box::new(PublishedTypes::from_inner(inner));
        let ptr = (&*generation as *const PublishedTypes).cast_mut();
        self.published_generations
            .lock()
            .expect("type publication generation lock poisoned")
            .push(generation);
        self.published.store(ptr, Ordering::Release);
    }

    fn published(&self) -> &PublishedTypes {
        let ptr = self.published.load(Ordering::Acquire);
        debug_assert!(!ptr.is_null(), "type publication pointer is null");
        // SAFETY: `from_inner` installs the initial generation before the manager
        // becomes observable. Every later generation is retained in
        // `published_generations` for the manager's lifetime and is immutable.
        unsafe { &*ptr }
    }

    /// Run one double-checked mint operation and publish only when it appended a
    /// new type. Cache hits therefore retain the current generation unchanged.
    fn mint(&self, f: impl FnOnce(&mut TypeManagerInner) -> TypeId) -> TypeId {
        let mut inner = self.write();
        let old_len = inner.types.len();
        let id = f(&mut inner);
        if inner.types.len() != old_len {
            self.publish(&inner);
        }
        id
    }

    // --- mint path (double-checked: read-lock hit, write-lock miss) -------
    //
    // Types are minted rarely and reused constantly, so each `get_or_make_*`
    // probes its cache under the read lock first and only takes the write lock
    // on a miss. The inner method re-checks its cache under the write lock, so
    // two racing minters agree on one id.

    pub fn get_or_make_int(&self, size: usize) -> TypeId {
        if let Some(&id) = self.read().int_by_size.get(&size) {
            return id;
        }
        self.mint(|inner| inner.get_or_make_int(size))
    }
    pub fn get_or_make_bool(&self) -> TypeId {
        if let Some(id) = self.read().bool_id {
            return id;
        }
        self.mint(TypeManagerInner::get_or_make_bool)
    }
    pub fn get_or_make_space_address(
        &self,
        size: usize,
        space: impl Into<MemorySpaceId>,
    ) -> TypeId {
        let space = space.into();
        if let Some(&id) = self.read().space_address.get(&(size, space)) {
            return id;
        }
        self.mint(|inner| inner.get_or_make_space_address(size, space))
    }
    /// Returns the [`TypeId`] for an [`AggregateType`] with default field names
    /// (`field1`, `field2`, ...), creating it if it does not yet exist.
    pub fn get_or_make_aggregate(&self, fields: Vec<TypeId>) -> TypeId {
        self.get_or_make_named_aggregate(default_named_fields(fields))
    }
    pub fn get_or_make_named_aggregate(&self, fields: Vec<AggregateField>) -> TypeId {
        if let Some(&id) = self.read().aggregate_by_fields.get(&fields) {
            return id;
        }
        self.mint(|inner| inner.get_or_make_named_aggregate(fields))
    }
    pub fn get_or_make_struct(
        &self,
        name: impl Into<String>,
        size: usize,
        fields: Vec<AggregateField>,
    ) -> TypeId {
        let name = name.into();
        if let Some(&id) = self.read().struct_by_name.get(&name) {
            return id;
        }
        self.mint(|inner| inner.get_or_make_struct(name, size, fields))
    }
    pub fn get_or_make_struct_pointer(&self, size: usize, pointee: TypeId) -> TypeId {
        if let Some(&id) = self.read().struct_pointer.get(&(size, pointee)) {
            return id;
        }
        self.mint(|inner| inner.get_or_make_struct_pointer(size, pointee))
    }
    pub fn get_or_make_array(&self, elem: TypeId, count: usize) -> TypeId {
        if let Some(&id) = self.read().array_by_elem_count.get(&(elem, count)) {
            return id;
        }
        self.mint(|inner| inner.get_or_make_array(elem, count))
    }
    pub fn get_or_make_list(&self, elem: TypeId, bound: usize) -> TypeId {
        if let Some(&id) = self.read().list_by_elem_bound.get(&(elem, Some(bound))) {
            return id;
        }
        self.mint(|inner| inner.get_or_make_list(elem, bound))
    }
    pub fn get_or_make_unbounded_list(&self, elem: TypeId) -> TypeId {
        if let Some(&id) = self.read().list_by_elem_bound.get(&(elem, None)) {
            return id;
        }
        self.mint(|inner| inner.get_or_make_unbounded_list(elem))
    }
    /// Build the sequence type of the given kind: a [`List`](Self::get_or_make_list)
    /// when `is_list`, else a fixed [`Array`](Self::get_or_make_array). The inverse
    /// of [`seq_of`](Self::seq_of).
    pub fn get_or_make_seq(&self, elem: TypeId, len: usize, is_list: bool) -> TypeId {
        if is_list {
            self.get_or_make_list(elem, len)
        } else {
            self.get_or_make_array(elem, len)
        }
    }
    pub fn binop_result(&self, lhs: TypeId, op: Binop, rhs: TypeId) -> TypeId {
        if let Some(id) = self.read().binop_result_probe(lhs, op, rhs) {
            return id;
        }
        self.mint(|inner| inner.binop_result(lhs, op, rhs))
    }

    // --- Registry-key reads (interner lock) ------------------------------

    pub fn bool_id(&self) -> Option<TypeId> {
        self.read().bool_id()
    }
    pub fn struct_by_name(&self, name: &str) -> Option<TypeId> {
        self.read().struct_by_name(name)
    }

    // --- Published TypeId reads (lock-free) ------------------------------

    pub fn is_bool(&self, id: TypeId) -> bool {
        matches!(self.get(id).repr(), TypeRepr::Bool)
    }
    pub fn size_of(&self, id: TypeId) -> usize {
        self.get(id).size()
    }
    pub fn space_of(&self, id: TypeId) -> Option<MemorySpaceId> {
        self.get(id).space()
    }
    pub fn pointee_of(&self, id: TypeId) -> Option<TypeId> {
        self.get(id).pointee()
    }
    pub fn array_of(&self, id: TypeId) -> Option<(TypeId, usize)> {
        self.get(id).array()
    }
    pub fn list_of(&self, id: TypeId) -> Option<(TypeId, Option<usize>)> {
        self.get(id).list()
    }
    pub fn seq_of(&self, id: TypeId) -> Option<(TypeId, usize, bool)> {
        if let Some((elem, count)) = self.array_of(id) {
            return Some((elem, count, false));
        }
        self.list_of(id)
            .and_then(|(elem, bound)| bound.map(|bound| (elem, bound, true)))
    }
    pub fn seq_elem_of(&self, id: TypeId) -> Option<TypeId> {
        self.array_of(id)
            .map(|(elem, _)| elem)
            .or_else(|| self.list_of(id).map(|(elem, _)| elem))
    }
    pub fn type_name(&self, id: TypeId) -> String {
        match self.get(id).repr() {
            TypeRepr::Bool => "bool".to_string(),
            TypeRepr::Struct { name, .. } => name,
            TypeRepr::StructPointer { pointee, .. } => {
                format!("{}*", self.type_name(pointee))
            }
            TypeRepr::Array { elem, count } => {
                format!("[{};{}]", self.type_name(elem), count)
            }
            TypeRepr::List { elem, bound } => match bound {
                Some(bound) => format!("[{};<={}]", self.type_name(elem), bound),
                None => format!("[{};*]", self.type_name(elem)),
            },
            _ => format!("i{}", self.size_of(id) * 8),
        }
    }
    pub fn field_type(&self, id: TypeId, index: usize) -> Option<TypeId> {
        self.aggregate_fields(id)?
            .get(index)
            .map(|field| field.type_id)
    }
    pub fn field_index(&self, id: TypeId, name: &str) -> Option<usize> {
        self.aggregate_fields(id)?
            .iter()
            .position(|field| field.name == name)
    }

    // --- reference reads (built on the append-only-stable `get`) ----------

    /// Returns a reference to the concrete [`Type`] for `id`.
    ///
    /// The lookup loads one immutable published index generation and takes no
    /// lock. The reference remains valid across later publications because the
    /// type table is append-only: types are created once and never removed, and
    /// each `Box<dyn Type>` pointee is heap-allocated and never moved.
    pub fn get(&self, id: TypeId) -> &dyn Type {
        let entries = &self.published().entries;
        let ptr = *entries.get(id.0 as usize).unwrap_or_else(|| {
            panic!(
                "missing published type {id:?}; published type count is {}",
                entries.len()
            )
        });
        // SAFETY: publication records pointers to immutable boxed type objects.
        // The table is append-only and the boxes remain owned by `inner.types`
        // until this manager is dropped.
        unsafe { &*ptr }
    }

    pub fn struct_name_of(&self, id: TypeId) -> Option<&str> {
        self.get(id).struct_name()
    }
    pub fn aggregate_fields(&self, id: TypeId) -> Option<&[AggregateField]> {
        self.get(id).fields()
    }
    pub fn field_by_offset(&self, id: TypeId, offset: usize) -> Option<(usize, &AggregateField)> {
        self.aggregate_fields(id)?
            .iter()
            .enumerate()
            .find(|(_, field)| field.offset == offset)
    }
    pub fn field_name(&self, id: TypeId, index: usize) -> Option<&str> {
        self.aggregate_fields(id)?
            .get(index)
            .map(|field| field.name.as_str())
    }
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------
//
// `TypeManager` owns `Box<dyn Type>` trait objects, which serde cannot derive
// over. Instead we serialize the type table as a `Vec<TypeRepr>` (the flat
// description each type reports via `Type::repr`) and replay the `get_or_make_*`
// constructors on load. Replaying in order reproduces the interned `TypeId`
// indices and rebuilds the lookup maps (`int_by_size`, `stack_address`,
// `space_address`) exactly, so no other field needs to be persisted.

impl serde::Serialize for TypeManager {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let reprs: Vec<TypeRepr> = self
            .published()
            .entries
            .iter()
            .map(|&ptr| {
                // SAFETY: the same publication invariant used by `get` applies
                // to every pointer in this immutable generation.
                unsafe { &*ptr }.repr()
            })
            .collect();
        reprs.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for TypeManager {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let reprs = Vec::<TypeRepr>::deserialize(deserializer)?;
        let manager = TypeManager::new();
        for repr in reprs {
            match repr {
                TypeRepr::Int { size } => {
                    manager.get_or_make_int(size);
                }
                TypeRepr::Bool => {
                    manager.get_or_make_bool();
                }
                TypeRepr::SpaceAddress { size, space } => {
                    manager.get_or_make_space_address(size, space);
                }
                // Field types have lower TypeIds (built before the aggregate),
                // so replaying in order guarantees they already exist here.
                TypeRepr::Aggregate { fields } => {
                    manager.get_or_make_named_aggregate(fields);
                }
                TypeRepr::Struct { name, size, fields } => {
                    manager.get_or_make_struct(name, size, fields);
                }
                // The pointee has a lower TypeId (built before the pointer),
                // so replaying in order guarantees it already exists here.
                TypeRepr::StructPointer { size, pointee } => {
                    manager.get_or_make_struct_pointer(size, pointee);
                }
                // The element type has a lower TypeId (built before the array),
                // so replaying in order guarantees it already exists here.
                TypeRepr::Array { elem, count } => {
                    manager.get_or_make_array(elem, count);
                }
                // The element type has a lower TypeId (built before the list),
                // so replaying in order guarantees it already exists here.
                TypeRepr::List { elem, bound } => match bound {
                    Some(b) => {
                        manager.get_or_make_list(elem, b);
                    }
                    None => {
                        manager.get_or_make_unbounded_list(elem);
                    }
                },
            }
        }
        Ok(manager)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binop_result_bool_rules() {
        use crate::value::insn::{Binop, FloatBinop, IntBinop};
        let tm = TypeManager::new();
        let i32 = tm.get_or_make_int(4);
        let boolt = tm.get_or_make_bool();

        // Comparisons over ints yield bool.
        assert_eq!(tm.binop_result(i32, Binop::Int(IntBinop::Less), i32), boolt);
        assert_eq!(
            tm.binop_result(i32, Binop::Int(IntBinop::Equal), i32),
            boolt
        );
        // Float comparisons yield bool.
        assert_eq!(
            tm.binop_result(i32, Binop::Float(FloatBinop::Less), i32),
            boolt
        );
        // Bitwise over bool operands stays bool (logical and/or/xor).
        assert_eq!(
            tm.binop_result(boolt, Binop::Int(IntBinop::And), boolt),
            boolt
        );
        assert_eq!(
            tm.binop_result(boolt, Binop::Int(IntBinop::Or), boolt),
            boolt
        );
        // Bitwise over ints preserves the int type.
        assert_eq!(tm.binop_result(i32, Binop::Int(IntBinop::And), i32), i32);
        // Arithmetic preserves the left type.
        assert_eq!(tm.binop_result(i32, Binop::Int(IntBinop::Add), i32), i32);
    }

    #[test]
    fn bool_is_byte_stored_and_interned() {
        let tm = TypeManager::new();
        let b = tm.get_or_make_bool();
        assert_eq!(tm.size_of(b), 1);
        assert!(tm.is_bool(b));
        assert_eq!(tm.get_or_make_bool(), b);
        assert_eq!(tm.type_name(b), "bool");
        let i8 = tm.get_or_make_int(1);
        assert!(!tm.is_bool(i8));
    }

    #[test]
    fn bool_round_trips_through_serde() {
        let tm = TypeManager::new();
        let _i8 = tm.get_or_make_int(1);
        let b = tm.get_or_make_bool();
        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&tm, config).unwrap();
        let (back, _): (TypeManager, _) =
            bincode::serde::decode_from_slice(&bytes, config).unwrap();
        assert!(back.is_bool(b));
        assert_eq!(back.size_of(b), 1);
    }

    #[test]
    fn array_is_a_disguised_width_n_scalar() {
        let tm = TypeManager::new();
        let i8 = tm.get_or_make_int(1);
        let arr = tm.get_or_make_array(i8, 20);

        // Width is count * sizeof(elem) — structural passes see a 20-byte scalar.
        assert_eq!(tm.size_of(arr), 20);
        // Disguise: no aggregate fields, so tuple/struct machinery skips it.
        assert!(tm.aggregate_fields(arr).is_none());
        // Element-aware sites recover (elem, count).
        assert_eq!(tm.array_of(arr), Some((i8, 20)));
        // Interned: same (elem, count) → same TypeId.
        assert_eq!(tm.get_or_make_array(i8, 20), arr);
        assert_ne!(tm.get_or_make_array(i8, 21), arr);
        // Pretty name for dumps.
        assert_eq!(tm.type_name(arr), "[i8;20]");
    }

    #[test]
    fn array_round_trips_through_serde() {
        let tm = TypeManager::new();
        let i8 = tm.get_or_make_int(1);
        let arr = tm.get_or_make_array(i8, 20);

        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&tm, config).unwrap();
        let (back, _): (TypeManager, _) =
            bincode::serde::decode_from_slice(&bytes, config).unwrap();
        // Replaying constructors in TypeId order reproduces the same handles.
        assert_eq!(back.array_of(arr), Some((i8, 20)));
        assert_eq!(back.size_of(arr), 20);
    }

    #[test]
    fn list_round_trips_through_serde() {
        let tm = TypeManager::new();
        let i8 = tm.get_or_make_int(1);
        let list = tm.get_or_make_list(i8, 20);

        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&tm, config).unwrap();
        let (back, _): (TypeManager, _) =
            bincode::serde::decode_from_slice(&bytes, config).unwrap();
        // The list survives as a list (not a fixed array) with its bound and
        // footprint intact.
        assert_eq!(back.list_of(list), Some((i8, Some(20))));
        assert_eq!(back.array_of(list), None);
        assert_eq!(back.size_of(list), 20);
    }

    #[test]
    fn unbounded_list_round_trips_and_has_no_footprint() {
        let tm = TypeManager::new();
        let i8 = tm.get_or_make_int(1);
        let list = tm.get_or_make_unbounded_list(i8);
        // Unbounded: a list with no static bound and no materialized footprint.
        assert_eq!(tm.list_of(list), Some((i8, None)));
        assert_eq!(tm.size_of(list), 0);
        // Distinct from any bounded list of the same element.
        assert_ne!(list, tm.get_or_make_list(i8, 20));

        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&tm, config).unwrap();
        let (back, _): (TypeManager, _) =
            bincode::serde::decode_from_slice(&bytes, config).unwrap();
        assert_eq!(back.list_of(list), Some((i8, None)));
        assert_eq!(back.array_of(list), None);
    }

    #[test]
    fn newly_minted_type_is_published_before_return() {
        let tm = TypeManager::new();
        let i16 = tm.get_or_make_int(2);
        let array = tm.get_or_make_array(i16, 7);

        assert_eq!(tm.size_of(array), 14);
        assert_eq!(tm.array_of(array), Some((i16, 7)));
    }

    #[test]
    fn concurrent_mint_and_published_reads_are_consistent() {
        let tm = TypeManager::new();
        let byte = tm.get_or_make_int(1);

        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for count in 1..=128 {
                        let array = tm.get_or_make_array(byte, count);
                        assert_eq!(tm.size_of(array), count);
                        assert_eq!(tm.array_of(array), Some((byte, count)));
                        assert_eq!(tm.size_of(byte), 1);
                    }
                });
            }
        });
    }
}
