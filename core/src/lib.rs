//! Core IR for the harbinger-lifter binary analysis framework.
//!
//! `qcode-core` provides a typed, SSA-style intermediate representation (IR)
//! used to model the semantics of lifted machine code. It is modelled after
//! GHIDRA's p-code IR. Values however have more variety. On top of GHIDRA's
//! [`Varnode`] (a named memory location), qcode also has [`Instruction`]
//! (an SSA value computed by an operation), [`BasicBlock`]
//! (a control-flow node), [`Function`] and [`Literal`] (a constant value).
//!
//! # Core Concepts
//!
//! ## Memory spaces
//!
//! A [`Space`] is a named, uniformly-addressed memory region — RAM, ROM, or a
//! register file. Every varnode belongs to exactly one space. The context is
//! initialised with a default RAM space; additional spaces (e.g. a register
//! space) can be added with [`Space::new`].
//!
//! ## Values
//!
//! All IR entities are addressed through [`ValueId`], a cheap `Copy`
//! discriminated union:
//!
//! |           Variant        |                  Description                  |
//! |--------------------------|-----------------------------------------------|
//! | [`ValueId::Literal`]     | An integer constant, possibly with a label    |
//! | [`ValueId::Instruction`] | An SSA value produced by an [`Instruction`]   |
//! | [`ValueId::Varnode`]     | A named memory location (register, global, …) |
//! | [`ValueId::BasicBlock`]  | A control-flow node / label                   |
//! | [`ValueId::Function`]    | A lifted or external function                 |
//!
//! All values live inside a [`Context`], which acts as an arena. You retrieve
//! a value by calling [`Context::get_value`], which returns a [`ValueRef`]
//! that borrows the context for its lifetime.
//!
//! ## Context
//!
//! [`Context`] is the single owner of all IR state — spaces, values,
//! instructions, blocks, and functions. Create one with [`Context::new`] and
//! pass `&mut` references to the [`Builder`] and analysis passes.
//!
//! ## Builder
//!
//! [`Builder`] is the API for constructing IR. It provides methods to create
//! instructions, blocks, and functions, and to manipulate the control flow
//! graph. A builder is always tied to a specific context and block. You can
//! obtain one from an existing block or directly from the context:
//!
//! ```rust,ignore
//! use qcode_core::{context::Context, builder::Builder};
//!
//! let mut ctx = Context::new();
//!
//! // Start a new block at machine address 0x1000
//! let mut builder = Builder::from_context(&mut ctx, 0x1000);
//!
//! // Emit a load from memory
//! let ptr = /* some ValueId */;
//! let value = builder.push_load(ptr, 8, ctx.default_space);
//!
//! // Terminate the block: emit an unconditional branch to 0x1010
//! builder.finalize(0x1010);
//! ```
//!
//! Alternatively, use the [`qcode!`](qcode_macro::qcode) proc-macro for a
//! convenient text-format DSL when writing tests or exploring the IR.
//!
//! ## Lifetime parameters
//!
//! Two lifetime parameters appear throughout this crate:
//!
//! - `'str` — the lifetime of interned string data (names, space names).
//!   Typically tied to a `&'str str` borrowed from the binary image or from a
//!   string literal.
//! - `'ctx` — the lifetime of a borrow of the [`Context`]. Reference types
//!   like [`InstructionRef`], [`BlockRef`], and [`FunctionRef`] carry `'ctx`
//!   to ensure they do not outlive the arena.
//!
//! [`Space`]:                   crate::space::Space
//! [`Space::new`]:              crate::space::Space::new
//! [`ValueId`]:                 crate::value::ValueId
//! [`ValueId::Literal`]:        crate::value::ValueId::Literal
//! [`ValueId::Instruction`]:    crate::value::ValueId::Instruction
//! [`ValueId::Varnode`]:        crate::value::ValueId::Varnode
//! [`ValueId::BasicBlock`]:     crate::value::ValueId::BasicBlock
//! [`ValueId::Function`]:       crate::value::ValueId::Function
//! [`Instruction`]:             crate::value::Instruction
//! [`InstructionRef`]:          crate::value::InstructionRef
//! [`BlockRef`]:                crate::value::BlockRef
//! [`FunctionRef`]:             crate::value::FunctionRef
//! [`ValueRef`]:                crate::value::ValueRef
//! [`Varnode`]:                 crate::value::Varnode
//! [`BasicBlock`]:              crate::value::BasicBlock

pub mod assumption;
pub mod builder;
pub mod context;
pub mod error;
pub mod space;
pub mod types;
pub mod value;

#[cfg(any(test, feature = "testing"))]
pub mod testing;
