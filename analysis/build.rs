//! Parses `src/structs/win32/teb.h` (the Windows TEB/PEB layout) with libclang
//! at build time and bakes the struct layouts into the crate, so the TEB-seeding
//! pass never hand-translates field offsets into Rust. Mirrors `cabi`'s build
//! script.
//!
//! Parsed for **32-bit** (`-m32`), so pointers are 4 bytes and offsets match the
//! x86 TEB. Anonymous padding members (named `_…`) are dropped — they only exist
//! in the header to push real fields to their true offsets, which clang computes.

use std::{env, fs, path::PathBuf};

use clang::{Clang, EntityKind, Index, TypeKind};

include!("src/structs/win32/hstruct.rs");

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/structs/win32/hstruct.rs");
    println!("cargo:rerun-if-changed=src/structs/win32/teb.h");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let header = manifest.join("src/structs/win32/teb.h");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let structs = parse(&header);

    let bytes = bincode::serde::encode_to_vec(&structs, bincode::config::standard()).unwrap();
    let path = out_dir.join("teb_structs.bin");
    fs::write(&path, bytes).unwrap();
    println!("cargo:rustc-env=QCODE_TEB_STRUCTS={}", path.display());
}

fn parse(header: &std::path::Path) -> Vec<HStruct> {
    let clang = Clang::new().expect("libclang must be available at build time");
    let index = Index::new(&clang, false, false);
    let tu = index
        .parser(header)
        .arguments(&["-m32".to_string()])
        .parse()
        .expect("failed to parse teb.h");

    let mut structs = Vec::new();
    for ent in tu.get_entity().get_children() {
        if ent.get_kind() != EntityKind::StructDecl {
            continue;
        }
        let Some(name) = ent.get_name() else { continue };
        let Some(ty) = ent.get_type() else { continue };
        let size = ty.get_sizeof().unwrap_or(0);

        let mut fields = Vec::new();
        for field in ent.get_children() {
            if field.get_kind() != EntityKind::FieldDecl {
                continue;
            }
            let Some(fname) = field.get_name() else {
                continue;
            };
            // Padding members exist only to set offsets; clang already accounted
            // for them, so we don't emit them as named fields.
            if fname.starts_with('_') {
                continue;
            }
            let offset_bits = field.get_offset_of_field().unwrap_or(0);
            let offset = offset_bits / 8;
            let fty = field.get_type().expect("field has a type");
            let kind = match fty.get_kind() {
                TypeKind::Pointer => {
                    let width = fty.get_sizeof().unwrap_or(4);
                    // `struct X*` → chainable pointer to the named struct;
                    // `void*`/other → opaque scalar pointer.
                    let pointee = fty
                        .get_pointee_type()
                        .and_then(|p| p.get_declaration())
                        .filter(|d| d.get_kind() == EntityKind::StructDecl)
                        .and_then(|d| d.get_name());
                    match pointee {
                        Some(pointee) => HFieldKind::StructPtr { pointee, width },
                        None => HFieldKind::Int { size: width },
                    }
                }
                _ => HFieldKind::Int {
                    size: fty.get_sizeof().unwrap_or(0),
                },
            };
            fields.push(HField {
                name: fname,
                offset,
                kind,
            });
        }

        structs.push(HStruct { name, size, fields });
    }
    structs
}
