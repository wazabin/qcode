// Flat, serializable description of a C struct layout parsed from a header.
//
// `build.rs` parses `teb.h` with libclang and bakes a `Vec<HStruct>` into
// the crate; `teb_seed` deserializes it and registers each as a nominal qcode
// struct. This file is shared verbatim with the build script via `include!`, so
// it must depend only on `serde`.

/// One named field of an [`HStruct`], with its clang-computed byte offset.
/// Anonymous padding (header members named `_…`) is dropped during parsing, so
/// every `HField` is a field the analysis actually names.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HField {
    pub name: String,
    pub offset: usize,
    pub kind: HFieldKind,
}

/// A field's type: a scalar of `n` bytes, or a pointer (of `width` bytes) to a
/// named struct — the latter is what lets struct typing chain one hop further.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum HFieldKind {
    Int { size: usize },
    StructPtr { pointee: String, width: usize },
}

/// A parsed struct: its name, total byte size, and the named fields of interest.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HStruct {
    pub name: String,
    pub size: usize,
    pub fields: Vec<HField>,
}
