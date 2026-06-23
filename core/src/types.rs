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

    /// Returns `true` for [`StackAddress`] types.
    fn is_stack_address(&self) -> bool {
        false
    }

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
    StackAddress {
        size: usize,
        space: Option<SpaceId>,
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

/// The synthetic base address assigned to a 64-bit function's stack pointer by
/// the brighten pass. Stack-slot literals hold an *absolute* address of the form
/// `STACK_BASE + offset` (offsets are negative for the locals below the entry
/// stack pointer), so their display derives the signed offset from this base.
pub const STACK_BASE: u64 = 0x1000_0000_0000_0000;

/// The 32-bit counterpart of [`STACK_BASE`]. Chosen so that `base ± frame_size`
/// stays positive and well within 32 bits (frames are at most a few KiB), and so
/// it survives truncation to a 4-byte stack-pointer register unchanged.
pub const STACK_BASE_32: u64 = 0x1000_0000;

/// The synthetic stack base for a given pointer width in bytes. The brighten pass
/// seeds the stack pointer with `stack_base(ptr_width)`; the lower-stack pass,
/// call-summary delta recovery, and literal display all recover a slot's signed
/// offset as `literal.value - stack_base(ptr_width)`, where `ptr_width` is the
/// width of the slot's `StackAddress` type.
pub fn stack_base(ptr_width: usize) -> u64 {
    match ptr_width {
        4 => STACK_BASE_32,
        _ => STACK_BASE,
    }
}

/// A pointer-width address that is known to live in the stack memory space.
///
/// # Arithmetic rules
///
/// | Expression    | Result       |
/// |---------------|--------------|
/// | `SA + Int`    | `SA`         |
/// | `SA - Int`    | `SA`         |
/// | `SA - SA`     | `Int(ptr_w)` |
/// | `SA & Int`    | `SA`         |
/// | `SA \| Int`   | `SA`         |
/// | `SA == SA`    | `Int(1)`     |
/// | `SA < SA`     | `Int(1)`     |
/// | `SA + SA`     | `Int` + warn |
/// | `SA * _`      | `Int` + warn |
/// | `SA ^ _`      | `Int` + warn |
/// | `SA << _`     | `Int` + warn |
#[derive(Clone)]
pub struct StackAddress {
    size: usize,
    space: Option<SpaceId>,
}

impl Type for StackAddress {
    fn size(&self) -> usize {
        self.size
    }

    fn is_stack_address(&self) -> bool {
        true
    }

    fn space(&self) -> Option<SpaceId> {
        self.space
    }

    fn clone_box(&self) -> Box<dyn Type> {
        Box::new(self.clone())
    }

    fn repr(&self) -> TypeRepr {
        TypeRepr::StackAddress {
            size: self.size,
            space: self.space,
        }
    }
}

/// A pointer-typed value carrying the memory space it points into.
///
/// Unlike [`StackAddress`], this represents arbitrary (non-stack) space
/// provenance — e.g. the result of `&A + offset`, which points into the space
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
    /// Singleton: there is at most one StackAddress type per context.
    stack_address: Option<TypeId>,
    /// Fast lookup: (size, space) → SpaceAddress TypeId.
    space_address: HashMap<(usize, SpaceId), TypeId>,
    /// Fast lookup: named field-type list → Aggregate TypeId.
    aggregate_by_fields: HashMap<Vec<AggregateField>, TypeId>,
    /// Nominal lookup: struct name → StructType TypeId.
    struct_by_name: HashMap<String, TypeId>,
    /// Fast lookup: (size, pointee) → StructPointer TypeId.
    struct_pointer: HashMap<(usize, TypeId), TypeId>,
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
            stack_address: None,
            space_address: HashMap::default(),
            aggregate_by_fields: HashMap::default(),
            struct_by_name: HashMap::default(),
            struct_pointer: HashMap::default(),
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

    /// Returns the singleton [`TypeId`] for `StackAddress`, creating it if
    /// needed. `size` is the architecture pointer width; `space` is the stack
    /// [`SpaceId`] (used by [`TypeManager::space_of`] for display and alias queries).
    pub fn get_or_make_stack_address(&mut self, size: usize, space: Option<SpaceId>) -> TypeId {
        if let Some(id) = self.stack_address {
            return id;
        }
        let id = self.register(Box::new(StackAddress { size, space }));
        self.stack_address = Some(id);
        id
    }

    /// Returns the stored [`TypeId`] for the `StackAddress` singleton, if it
    /// has been registered.
    pub fn stack_address_id(&self) -> Option<TypeId> {
        self.stack_address
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

    /// A short display name for `id`, used by the IR formatters in place of the
    /// raw `i<bits>` width. Nominal structs print their name and struct pointers
    /// print `Pointee*`; everything else (integers, stack/space addresses, and
    /// the structural aggregates used by `argpromote`) keeps its width-based
    /// `i<bits>` form, so existing IR/signature assertions are unaffected.
    pub fn type_name(&self, id: TypeId) -> String {
        match self.get(id).repr() {
            TypeRepr::Struct { name, .. } => name,
            TypeRepr::StructPointer { pointee, .. } => format!("{}*", self.type_name(pointee)),
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

    /// Returns `true` when `id` is the `StackAddress` type.
    pub fn is_stack_address(&self, id: TypeId) -> bool {
        self.get(id).is_stack_address()
    }

    /// Returns the memory space associated with `id`, if any.
    ///
    /// Non-`None` only for [`StackAddress`] types that were registered with a
    /// known stack [`SpaceId`].
    pub fn space_of(&self, id: TypeId) -> Option<SpaceId> {
        self.get(id).space()
    }

    /// Returns `true` if values of type `a` and `b` may alias.
    ///
    /// `StackAddress` and `Int` are in disjoint address spaces and never alias.
    pub fn can_alias(&self, a: TypeId, b: TypeId) -> bool {
        self.is_stack_address(a) == self.is_stack_address(b)
    }

    /// Computes the result [`TypeId`] for a binary operation on `lhs op rhs`.
    ///
    /// Type errors (e.g. `SA * Int`) demote to `Int` and are flagged here via
    /// `eprintln!`. A dedicated verify pass should be added to surface these
    /// properly.
    ///
    /// # TODO
    ///
    /// Replace `eprintln!` with a structured diagnostic once the verify pass
    /// exists.
    pub fn binop_result(&mut self, lhs: TypeId, op: Binop, rhs: TypeId) -> TypeId {
        let lhs_stack = self.is_stack_address(lhs);
        let rhs_stack = self.is_stack_address(rhs);

        match op {
            Binop::Int(int_op) => match (lhs_stack, rhs_stack, int_op) {
                // SA + Int → SA
                (true, false, IntBinop::Add) => lhs,
                // Int + SA → SA
                (false, true, IntBinop::Add) => rhs,
                // SA - Int → SA
                (true, false, IntBinop::Sub) => lhs,
                // SA - SA → Int(ptr_width)
                (true, true, IntBinop::Sub) => {
                    let ptr_size = self.size_of(lhs);
                    self.get_or_make_int(ptr_size)
                }
                // SA & Int → SA  (alignment masking)
                (true, false, IntBinop::And) => lhs,
                // Int & SA → SA  (alignment masking)
                (false, true, IntBinop::And) => rhs,
                // SA | Int → SA
                (true, false, IntBinop::Or) => lhs,
                // Int | SA → SA
                (false, true, IntBinop::Or) => rhs,
                // Comparisons always yield a 1-byte boolean, regardless of
                // whether the operands are stack addresses or plain integers.
                (
                    _,
                    _,
                    IntBinop::Equal
                    | IntBinop::NotEqual
                    | IntBinop::Less
                    | IntBinop::LessEqual
                    | IntBinop::SLess
                    | IntBinop::SLessEqual,
                ) => self.get_or_make_int(1),
                // Type errors: SA + SA, SA * _, SA ^ _, SA << _, etc.
                (true, _, _) | (_, true, _) => {
                    // TODO: emit structured diagnostic (verify pass)
                    eprintln!(
                        "qcode type warning: invalid StackAddress operation {:?}; demoting to Int",
                        int_op
                    );
                    let size = self.size_of(lhs);
                    self.get_or_make_int(size)
                }
                // Both Int: result is Int (same type as lhs after coercion)
                (false, false, _) => lhs,
            },
            // Bool operations yield a 1-byte boolean and are not defined on
            // StackAddress operands.
            Binop::Bool(_) => {
                if lhs_stack || rhs_stack {
                    eprintln!(
                        "qcode type warning: bool operation on StackAddress; demoting to Int"
                    );
                }
                self.get_or_make_int(1)
            }
            // Float operations preserve the operand width and are not defined on
            // StackAddress operands.
            Binop::Float(_) => {
                if lhs_stack || rhs_stack {
                    eprintln!(
                        "qcode type warning: float operation on StackAddress; demoting to Int"
                    );
                    let size = self.size_of(lhs);
                    self.get_or_make_int(size)
                } else {
                    lhs
                }
            }
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

impl<'de> serde::Deserialize<'de> for TypeManager {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let reprs = Vec::<TypeRepr>::deserialize(deserializer)?;
        let mut manager = TypeManager::new();
        for repr in reprs {
            match repr {
                TypeRepr::Int { size } => {
                    manager.get_or_make_int(size);
                }
                TypeRepr::StackAddress { size, space } => {
                    manager.get_or_make_stack_address(size, space);
                }
                TypeRepr::SpaceAddress { size, space } => {
                    manager.get_or_make_space_address(size, space);
                }
                // Field types have lower TypeIds (built before the aggregate),
                // so replaying in order guarantees they already exist here.
                TypeRepr::Aggregate { fields } => {
                    manager.get_or_make_named_aggregate(fields);
                }
                TypeRepr::Struct {
                    name,
                    size,
                    fields,
                } => {
                    manager.get_or_make_struct(name, size, fields);
                }
                // The pointee has a lower TypeId (built before the pointer),
                // so replaying in order guarantees it already exists here.
                TypeRepr::StructPointer { size, pointee } => {
                    manager.get_or_make_struct_pointer(size, pointee);
                }
            }
        }
        Ok(manager)
    }
}
