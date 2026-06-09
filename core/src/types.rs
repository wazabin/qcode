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

use std::collections::HashMap;

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
#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug, Ord, PartialOrd)]
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

    /// Clones this type into a fresh boxed trait object.
    ///
    /// This enables `Clone for Box<dyn Type>` (and hence `Clone` for
    /// [`TypeManager`] and [`Context`](crate::context::Context)), which the GUI
    /// relies on to fork a context before running an analysis pipeline.
    fn clone_box(&self) -> Box<dyn Type>;
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
            int_by_size: HashMap::new(),
            stack_address: None,
            space_address: HashMap::new(),
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
                // SA - Int → SA
                (true, false, IntBinop::Sub) => lhs,
                // SA - SA → Int(ptr_width)
                (true, true, IntBinop::Sub) => {
                    let ptr_size = self.size_of(lhs);
                    self.get_or_make_int(ptr_size)
                }
                // SA & Int → SA  (alignment masking)
                (true, false, IntBinop::And) => lhs,
                // SA | Int → SA
                (true, false, IntBinop::Or) => lhs,
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
