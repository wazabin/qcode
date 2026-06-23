//! Windows TEB seeding: type the `FS_OFFSET` segment-base register as a global
//! `PtrTo<TEB>` (from the `teb.h` layout) so [`StructTyping`](super::typing)
//! can unfold the TEB/PEB field chain. See [`seed`] for details.

mod hstruct;
mod seed;

pub use seed::{WindowsTebSeed, register_teb_structs, seed_teb_register};
