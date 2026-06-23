//! Struct-typing analysis: recover named struct-field accesses from raw
//! pointer arithmetic, and the platform-specific seeds that drive it.
//!
//! - [`typing`] — the architecture-neutral [`StructTyping`] forward-fixpoint
//!   pass (`add → gep`, load inherits the field type).
//! - [`win32`] — Windows-specific seeding (type the `FS_OFFSET` register as
//!   `PtrTo<TEB>` from the `teb.h` layout) that gives the typing pass a root.

pub mod typing;
pub mod win32;

pub use typing::StructTyping;
