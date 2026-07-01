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
//! | [`StackAddress`] | Pointer-width address in the stack memory space |
//!
//! # TODO
//!
//! - `Bool`: currently modelled as `Int(1)`. Will become its own type once the
//!   verify pass is implemented, so that `Bool + Bool` can be rejected.
//! - Pointer types for RAM/register spaces.

use rustc_hash::FxHashMap as HashMap;

use crate::{
    space::SpaceId,
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
    fn space(&self) -> Option<SpaceId> {
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
    SpaceAddress {
        size: usize,
        space: SpaceId,
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
    space: SpaceId,
}

impl Type for SpaceAddress {
    fn size(&self) -> usize {
        self.size
    }

    fn space(&self) -> Option<SpaceId> {
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

/// Registry that owns all [`Type`] objects and hands out interned [`TypeId`]s.
///
/// Types are created once (at context construction or when the architecture is
/// configured) and never removed. All methods that *only read* types take
/// `&self`; methods that may create new `Int` types on demand take `&mut self`.
#[derive(Clone)]
pub struct TypeManager {
    types: Vec<Box<dyn Type>>,
    /// Fast lookup: Int size → TypeId.
    int_by_size: HashMap<usize, TypeId>,
    /// Fast lookup: (size, space) → SpaceAddress TypeId.
    space_address: HashMap<(usize, SpaceId), TypeId>,
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

impl Default for TypeManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeManager {
    pub fn new() -> Self {
        Self {
            types: Vec::new(),
            int_by_size: HashMap::default(),
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

    /// Returns the [`TypeId`] for a [`SpaceAddress`] of the given byte width
    /// pointing into `space`, creating it if it does not yet exist.
    pub fn get_or_make_space_address(&mut self, size: usize, space: SpaceId) -> TypeId {
        if let Some(&id) = self.space_address.get(&(size, space)) {
            return id;
        }
        let id = self.register(Box::new(SpaceAddress { size, space }));
        self.space_address.insert((size, space), id);
        id
    }

    /// Returns the [`TypeId`] for an [`AggregateType`] with default field names
    /// (`field1`, `field2`, ...), creating it if it does not yet exist.
    pub fn get_or_make_aggregate(&mut self, fields: Vec<TypeId>) -> TypeId {
        let fields = fields
            .into_iter()
            .enumerate()
            .map(|(i, type_id)| AggregateField::new(format!("field{}", i + 1), type_id))
            .collect();
        self.get_or_make_named_aggregate(fields)
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

    /// The pointee type of `id`, if `id` is a [`StructPointer`].
    pub fn pointee_of(&self, id: TypeId) -> Option<TypeId> {
        self.get(id).pointee()
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

    /// The `(elem, count)` of `id`, if `id` is an [`ArrayType`].
    pub fn array_of(&self, id: TypeId) -> Option<(TypeId, usize)> {
        self.get(id).array()
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

    /// The `(elem, bound)` of `id`, if `id` is a [`ListType`]; `bound` is `None`
    /// for an unbounded (pointer-sourced) list.
    pub fn list_of(&self, id: TypeId) -> Option<(TypeId, Option<usize>)> {
        self.get(id).list()
    }

    /// Unified view of any *sequence* type — a fixed [`array`](Type::array) or a
    /// variable-length [`list`](Type::list) — as `(elem, len, is_list)`, where
    /// `len` is the count (array) or bound (list). `None` for non-sequences. This
    /// is what lets `map`/`enumerate`/`take_while` operate on either kind: read
    /// the operand with `seq_of`, rebuild the result with
    /// [`get_or_make_seq`](Self::get_or_make_seq) preserving the kind.
    pub fn seq_of(&self, id: TypeId) -> Option<(TypeId, usize, bool)> {
        if let Some((elem, count)) = self.array_of(id) {
            return Some((elem, count, false));
        }
        // Only a *bounded* list reports a static length for seq-preserving rebuilds
        // (`map`/`enumerate`); an unbounded pointer-sourced list declines here.
        if let Some((elem, Some(bound))) = self.list_of(id) {
            return Some((elem, bound, true));
        }
        None
    }

    /// The element type of any *sequence* — a fixed [`array`](Type::array) or a
    /// [`list`](Type::list) of any bound, *including an unbounded* `[T;*]` list
    /// that [`seq_of`](Self::seq_of) declines (because it has no static length).
    /// This is the accessor length-erased code (`scanl`/`iota`/`concat` results,
    /// `at`) uses when it needs the element type but not the count.
    pub fn seq_elem_of(&self, id: TypeId) -> Option<TypeId> {
        if let Some((elem, _)) = self.array_of(id) {
            return Some(elem);
        }
        if let Some((elem, _)) = self.list_of(id) {
            return Some(elem);
        }
        None
    }

    /// Build the sequence type of the given kind: a [`List`](Self::get_or_make_list)
    /// when `is_list`, else a fixed [`Array`](Self::get_or_make_array). The inverse
    /// of [`seq_of`](Self::seq_of).
    pub fn get_or_make_seq(&mut self, elem: TypeId, len: usize, is_list: bool) -> TypeId {
        if is_list {
            self.get_or_make_list(elem, len)
        } else {
            self.get_or_make_array(elem, len)
        }
    }

    /// A short display name for `id`, used by the IR formatters in place of the
    /// raw `i<bits>` width. Nominal structs print their name and struct pointers
    /// print `Pointee*`; everything else (integers, stack/space addresses, and
    /// the structural aggregates used by `argpromote`) keeps its width-based
    /// `i<bits>` form, so existing IR/signature assertions are unaffected.
    pub fn type_name(&self, id: TypeId) -> String {
        match self.get(id).repr() {
            TypeRepr::Struct { name, .. } => name,
            TypeRepr::StructPointer { pointee, .. } => format!("{}*", self.type_name(pointee)),
            TypeRepr::Array { elem, count } => format!("[{};{}]", self.type_name(elem), count),
            TypeRepr::List { elem, bound } => match bound {
                Some(b) => format!("[{};<={}]", self.type_name(elem), b),
                None => format!("[{};*]", self.type_name(elem)),
            },
            _ => format!("i{}", self.size_of(id) * 8),
        }
    }

    /// The name of `id`, if `id` is a nominal [`StructType`].
    pub fn struct_name_of(&self, id: TypeId) -> Option<&str> {
        self.get(id).struct_name()
    }

    /// The field of struct/aggregate `id` whose byte offset is exactly `offset`,
    /// returned as `(index, field)`. Used by [`Gep`](crate::value::insn::Gep)
    /// resolution, which matches on exact offset.
    pub fn field_by_offset(&self, id: TypeId, offset: usize) -> Option<(usize, &AggregateField)> {
        self.aggregate_fields(id)?
            .iter()
            .enumerate()
            .find(|(_, field)| field.offset == offset)
    }

    /// The ordered named fields of `id`, or `None` if `id` is not an aggregate.
    pub fn aggregate_fields(&self, id: TypeId) -> Option<&[AggregateField]> {
        self.get(id).fields()
    }

    /// The type of field `index` of aggregate `id`, if `id` is an aggregate with
    /// at least `index + 1` fields.
    pub fn field_type(&self, id: TypeId, index: usize) -> Option<TypeId> {
        self.aggregate_fields(id)?
            .get(index)
            .map(|field| field.type_id)
    }

    /// The name of field `index` of aggregate `id`, if it exists.
    pub fn field_name(&self, id: TypeId, index: usize) -> Option<&str> {
        self.aggregate_fields(id)?
            .get(index)
            .map(|field| field.name.as_str())
    }

    /// The index of field `name` of aggregate `id`, if it exists.
    pub fn field_index(&self, id: TypeId, name: &str) -> Option<usize> {
        self.aggregate_fields(id)?
            .iter()
            .position(|field| field.name == name)
    }

    /// Returns a reference to the concrete [`Type`] for `id`.
    pub fn get(&self, id: TypeId) -> &dyn Type {
        &*self.types[id.0 as usize]
    }

    /// Returns the byte width of values with type `id`.
    pub fn size_of(&self, id: TypeId) -> usize {
        self.get(id).size()
    }

    /// Returns the memory space associated with `id`, if any. Non-`None` only for
    /// [`SpaceAddress`]/[`StructPointer`] pointer types.
    pub fn space_of(&self, id: TypeId) -> Option<SpaceId> {
        self.get(id).space()
    }

    /// Computes the result [`TypeId`] for a binary operation on `lhs op rhs`.
    ///
    /// Comparisons yield a 1-byte boolean; every other integer/float op preserves
    /// the left operand's type (so a pointer-typed operand keeps its space
    /// provenance through `ptr + offset`).
    pub fn binop_result(&mut self, lhs: TypeId, op: Binop, _rhs: TypeId) -> TypeId {
        match op {
            Binop::Int(int_op) => match int_op {
                IntBinop::Equal
                | IntBinop::NotEqual
                | IntBinop::Less
                | IntBinop::LessEqual
                | IntBinop::SLess
                | IntBinop::SLessEqual => self.get_or_make_int(1),
                _ => lhs,
            },
            Binop::Bool(_) => self.get_or_make_int(1),
            Binop::Float(_) => lhs,
        }
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
        let reprs: Vec<TypeRepr> = self.types.iter().map(|t| t.repr()).collect();
        reprs.serialize(serializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_is_a_disguised_width_n_scalar() {
        let mut tm = TypeManager::new();
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
        let mut tm = TypeManager::new();
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
        let mut tm = TypeManager::new();
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
        let mut tm = TypeManager::new();
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
}

impl<'de> serde::Deserialize<'de> for TypeManager {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let reprs = Vec::<TypeRepr>::deserialize(deserializer)?;
        let mut manager = TypeManager::new();
        for repr in reprs {
            match repr {
                TypeRepr::Int { size } => {
                    manager.get_or_make_int(size);
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
