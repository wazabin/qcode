//! Reconstruction of `.rodata` string constants as named C objects.
//!
//! A call like `puts(0x401178)` is faithful but unreadable: the interesting fact
//! is that `0x401178` is where the image keeps `"harbinger"`. This module turns
//! the address into a file-scope declaration
//!
//! ```c
//! static const char str_401178[] = "harbinger";
//! ```
//!
//! and renders the argument as `str_401178`.
//!
//! **Why a named object and not an inline literal.** `puts("harbinger")` would
//! be a *different* program: a C string literal has its own address, so any
//! pointer comparison, offset arithmetic, or second use of the same rodata cell
//! would silently disagree with the binary. Address identity is what the
//! original code has, and a named object is the only C spelling that preserves
//! it — two uses of `str_401178` are the same pointer, exactly as two loads of
//! `0x401178` are.
//!
//! **Why only read-only memory.** Bytes in a writable section are not a
//! constant: the process may overwrite them before the call runs, so printing
//! their load-time contents would be a lie. Reconstruction therefore requires
//! [`BinaryFormat::is_known_read_only`] — a *positive* proof, not merely "not
//! known writable", so a container that records no permissions (PE sections
//! before this was parsed, a raw blob) declines rather than guesses.
//!
//! **Where the type comes from.** The IR keeps no C types: `external_sigs`
//! projects the `cabi` prototype down to register slots, parameter names, and a
//! coarse [`ArgMemKind`], then drops the `CType`. Rendering is the only consumer
//! that needs the pointee, so the prototype is re-consulted here, at emit time,
//! rather than widening what the analysis pipeline carries. The IR still gates
//! the lookup: only a callee whose
//! [`argmem`](qcode::value::FunctionRef::argmem) summary exists — i.e. one
//! `external_sigs` actually prototyped — is looked up, so a local function that
//! merely shares a name with a libc symbol is never given libc's types.

use std::cell::RefCell;
use std::collections::BTreeMap;

use binfmt::BinaryFormat;
use cabi::CType;
use qcode::{
    context::Context,
    space::Space,
    value::{ArgMemKind, FunctionRef, LiteralRef, ValueId, function::FunctionId},
};

use super::tokens::{LineBuf, TokenKind, TokenLine};

/// Longest string we will walk looking for the NUL terminator. A pointer into
/// the middle of a table, or into a region that happens to have no terminator,
/// must fail fast rather than scan the whole image.
const MAX_STRING_LEN: usize = 4096;

/// The strings reconstructed while emitting one function, keyed by address so
/// two uses of the same rodata cell share a single declaration.
///
/// Resolution is memoized (including negative results) because the emitter asks
/// twice: once while rendering the call arguments, and once when the collected
/// declarations are prepended to the output.
pub(crate) struct StringPool<'a> {
    binary: Option<&'a dyn BinaryFormat>,
    target: cabi::AbiTarget,
    /// `addr -> Some(escaped C string body)` for reconstructed addresses,
    /// `addr -> None` for addresses that failed a check.
    seen: RefCell<BTreeMap<u64, Option<String>>>,
}

impl<'a> StringPool<'a> {
    pub(crate) fn new(ctx: &Context, binary: Option<&'a dyn BinaryFormat>) -> Self {
        let bits = (Space::from_id(ctx, ctx.shared.default_space).addr_size * 8) as u8;
        StringPool {
            binary,
            target: crate::pipeline::cabi_abi_target(ctx.target_os(), bits),
            seen: RefCell::new(BTreeMap::new()),
        }
    }

    /// The name to render for argument `index` of a call to `callee`, when that
    /// argument is a literal address holding a reconstructible string.
    ///
    /// `None` — the overwhelmingly common case — leaves the argument to the
    /// normal numeric rendering.
    pub(crate) fn arg_name(
        &self,
        ctx: &Context,
        callee: Option<FunctionId>,
        index: usize,
        arg: ValueId,
    ) -> Option<String> {
        // Only a *constant* address names a fixed object; a computed pointer is
        // a different string on every path through the function.
        let ValueId::Literal(literal) = arg else {
            return None;
        };
        let addr = LiteralRef::from_id(ctx, literal).value();
        if !self.binds_to_char_pointer(ctx, callee?, index) {
            return None;
        }
        let mut seen = self.seen.borrow_mut();
        let entry = seen
            .entry(addr)
            .or_insert_with(|| self.reconstruct(addr).map(|bytes| escape_c(&bytes)));
        entry.is_some().then(|| object_name(addr))
    }

    /// Whether argument `index` of `callee` binds to a `char *` / `const char *`
    /// parameter of its C prototype.
    ///
    /// `Call.args` is lockstep with the callee's materialized register-interface
    /// inputs, which `external_sigs` builds in prototype order, so `args[i]`
    /// binds `params[i]` for every index that exists on both sides.
    fn binds_to_char_pointer(&self, ctx: &Context, callee: FunctionId, index: usize) -> bool {
        let function = FunctionRef::from_id(ctx, callee);
        // A prototyped external is the only callee whose name may be resolved
        // against the C library tables. Without this, a local `puts` of the
        // program's own would be handed libc's signature.
        let argmem = match function.argmem() {
            Some(argmem) => argmem,
            None => return false,
        };
        if !matches!(
            argmem.params.get(index),
            Some(ArgMemKind::ConstPtr | ArgMemKind::MutPtr)
        ) {
            return false;
        }
        // `puts@plt` / `puts@GLIBC_2.2.5` name the same prototype as `puts`.
        let symbol = function.name();
        let symbol = symbol.split('@').next().unwrap_or(symbol);
        let Some(proto) = cabi::lookup(self.target, symbol) else {
            return false;
        };
        matches!(
            proto.params.get(index).map(|param| &param.ty),
            Some(CType::Pointer { pointee, .. }) if matches!(**pointee, CType::Integer { bytes: 1, .. })
        )
    }

    /// Read the NUL-terminated bytes at `addr`, or `None` if any check fails:
    /// no image, the address is not provably read-only, no terminator within
    /// [`MAX_STRING_LEN`], or the bytes are not text.
    fn reconstruct(&self, addr: u64) -> Option<Vec<u8>> {
        let binary = self.binary?;
        if !binary.is_known_read_only(addr) {
            return None;
        }
        let bytes = binary.read_cstring(addr, Some(MAX_STRING_LEN))?;
        // Text, not an arbitrary byte run that happens to contain a zero:
        // valid UTF-8 keeps ASCII and real multi-byte text and rejects the
        // binary tables that share a section with the string literals.
        std::str::from_utf8(&bytes).ok()?;
        Some(bytes)
    }

    /// The `static const char str_<addr>[] = "...";` declarations for every
    /// address reconstructed so far, in address order.
    pub(crate) fn declarations(&self) -> Vec<TokenLine> {
        self.seen
            .borrow()
            .iter()
            .filter_map(|(&addr, text)| Some((addr, text.as_ref()?)))
            .map(|(addr, text)| {
                let mut buf = LineBuf::default();
                buf.keyword("static");
                buf.space();
                buf.keyword("const");
                buf.space();
                buf.push("char", TokenKind::Type);
                buf.space();
                buf.push(object_name(addr), TokenKind::Variable);
                buf.punct("[");
                buf.punct("]");
                buf.space();
                buf.push("=", TokenKind::Operator);
                buf.space();
                buf.push(text.clone(), TokenKind::String);
                buf.punct(";");
                buf.into_line(0, None)
            })
            .collect()
    }
}

/// The C identifier standing for the rodata object at `addr`.
fn object_name(addr: u64) -> String {
    format!("str_{addr:x}")
}

/// Render `bytes` as a complete, quoted C string literal.
///
/// Printable ASCII passes through; `"` and `\` are backslash-escaped; the
/// control characters with a standard spelling use it; anything else becomes
/// `\xNN`. A hex escape is greedy in C, so a following hex digit would be
/// swallowed into the escape — the literal is split there (`"\x1b" "f"`, which
/// C concatenates back into one array) rather than silently changing the byte.
fn escape_c(bytes: &[u8]) -> String {
    let mut out = String::from("\"");
    let mut after_hex_escape = false;
    for &byte in bytes {
        let escaped = match byte {
            b'"' => Some("\\\"".to_owned()),
            b'\\' => Some("\\\\".to_owned()),
            0x07 => Some("\\a".to_owned()),
            0x08 => Some("\\b".to_owned()),
            0x09 => Some("\\t".to_owned()),
            0x0a => Some("\\n".to_owned()),
            0x0b => Some("\\v".to_owned()),
            0x0c => Some("\\f".to_owned()),
            0x0d => Some("\\r".to_owned()),
            0x20..=0x7e => None,
            _ => Some(format!("\\x{byte:02x}")),
        };
        match escaped {
            Some(text) => {
                after_hex_escape = text.starts_with("\\x");
                out.push_str(&text);
            }
            None => {
                if after_hex_escape && byte.is_ascii_hexdigit() {
                    out.push_str("\" \"");
                }
                after_hex_escape = false;
                out.push(byte as char);
            }
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::{emit_c, lower_function};
    use qcode::{context::Context, memory_image::MemoryImage, value::FunctionBody};
    use qcode_macro::qcode;

    /// Where the read-only copy of the test string lives.
    const RO_ADDR: u64 = 0x401178;
    /// Where the writable copy lives — same bytes, mutable section.
    const RW_ADDR: u64 = 0x402178;

    /// An image holding `bytes` twice: once read-only, once writable, so a test
    /// can ask for the same string from either kind of memory.
    fn image(bytes: &[u8]) -> MemoryImage {
        let mut image = MemoryImage::default();
        image.add_segment(RO_ADDR, bytes.to_vec(), false, false);
        image.add_segment(RW_ADDR, bytes.to_vec(), false, true);
        image
    }

    /// Stamp `name` onto a bodyless external carrying the argmem summary
    /// `external_sigs` gives a prototyped callee — the fact that licenses the
    /// cabi lookup at emit time.
    fn prototyped_external(ctx: &mut Context<'static>, name: &str, params: Vec<ArgMemKind>) {
        let id = FunctionBody::make_external(ctx, 0x9000, Some(name.to_string().into())).id;
        FunctionBody::from_id_mut(ctx, id).set_argmem(qcode::value::ExternArgmem {
            params,
            variadic: false,
        });
    }

    /// The whole point: a literal address that binds to `puts`'s `const char *`
    /// and lands in read-only memory becomes a named object, declared above the
    /// function and referenced by name at the call.
    #[test]
    fn read_only_string_becomes_a_named_object() {
        let mut ctx = Context::new();
        prototyped_external(&mut ctx, "puts", vec![ArgMemKind::ConstPtr]);
        qcode!(
            ctx,
            "
            fn main:
                <entry>
                    call fn puts(@s=i64 0x401178);
                <cont>
                    return at i64 0;
            "
        );
        let _ = (entry, cont);
        let c = emit_c(
            &ctx,
            &lower_function(&ctx, main),
            Some(&image(b"harbinger\0")),
        );

        assert!(
            c.contains("static const char str_401178[] = \"harbinger\";"),
            "expected a rodata object declaration:\n{c}"
        );
        assert!(
            c.contains("puts(str_401178)"),
            "the argument should read as the object:\n{c}"
        );
        // The declaration is file-scope, above the function that uses it.
        let decl = c.find("static const char").expect("declaration present");
        let header = c.find("fn main").expect("header present");
        assert!(decl < header, "declaration must precede the function:\n{c}");
    }

    /// Two calls naming the same address share one declaration — the object has
    /// one identity, and repeating it would suggest two.
    #[test]
    fn two_uses_of_one_address_share_a_declaration() {
        let mut ctx = Context::new();
        prototyped_external(&mut ctx, "puts", vec![ArgMemKind::ConstPtr]);
        qcode!(
            ctx,
            "
            fn main:
                <entry>
                    call fn puts(@s=i64 0x401178);
                <mid>
                    call fn puts(@s=i64 0x401178);
                <cont>
                    return at i64 0;
            "
        );
        let _ = (entry, mid, cont);
        let c = emit_c(&ctx, &lower_function(&ctx, main), Some(&image(b"hi\0")));

        assert_eq!(
            c.matches("static const char str_401178[]").count(),
            1,
            "one address means one declaration:\n{c}"
        );
        assert_eq!(
            c.matches("puts(str_401178)").count(),
            2,
            "both call sites name the same object:\n{c}"
        );
    }

    /// Bytes in a writable section are not a constant — the process may
    /// overwrite them before the call runs — so the address stays numeric.
    #[test]
    fn writable_memory_is_refused() {
        let mut ctx = Context::new();
        prototyped_external(&mut ctx, "puts", vec![ArgMemKind::ConstPtr]);
        qcode!(
            ctx,
            "
            fn main:
                <entry>
                    call fn puts(@s=i64 0x402178);
                <cont>
                    return at i64 0;
            "
        );
        let _ = (entry, cont);
        let c = emit_c(
            &ctx,
            &lower_function(&ctx, main),
            Some(&image(b"harbinger\0")),
        );

        assert!(
            !c.contains("static const char"),
            "a writable section must not be reconstructed:\n{c}"
        );
        assert!(
            c.contains("puts(0x402178)"),
            "the argument keeps its numeric rendering:\n{c}"
        );
    }

    /// Without a terminator inside the cap there is no string, only a pointer
    /// into some larger object.
    #[test]
    fn missing_terminator_is_refused() {
        let mut ctx = Context::new();
        prototyped_external(&mut ctx, "puts", vec![ArgMemKind::ConstPtr]);
        qcode!(
            ctx,
            "
            fn main:
                <entry>
                    call fn puts(@s=i64 0x401178);
                <cont>
                    return at i64 0;
            "
        );
        let _ = (entry, cont);
        let c = emit_c(
            &ctx,
            &lower_function(&ctx, main),
            Some(&image(b"no terminator")),
        );

        assert!(
            !c.contains("static const char"),
            "an unterminated run is not a string:\n{c}"
        );
        assert!(
            c.contains("puts(0x401178)"),
            "the argument keeps its numeric rendering:\n{c}"
        );
    }

    /// A non-printable byte does not disqualify a string; it is escaped.
    #[test]
    fn non_printable_bytes_are_escaped() {
        let mut ctx = Context::new();
        prototyped_external(&mut ctx, "puts", vec![ArgMemKind::ConstPtr]);
        qcode!(
            ctx,
            "
            fn main:
                <entry>
                    call fn puts(@s=i64 0x401178);
                <cont>
                    return at i64 0;
            "
        );
        let _ = (entry, cont);
        let c = emit_c(
            &ctx,
            &lower_function(&ctx, main),
            Some(&image(b"line\n\x01\0")),
        );

        assert!(
            c.contains(r#"static const char str_401178[] = "line\n\x01";"#),
            "expected escaped control bytes:\n{c}"
        );
    }

    /// The scope guard: `memcpy`'s parameters are pointers too, but a `void *`
    /// is not a string, so a literal address there stays numeric.
    #[test]
    fn void_pointer_parameters_are_left_alone() {
        let mut ctx = Context::new();
        prototyped_external(
            &mut ctx,
            "memcpy",
            vec![ArgMemKind::MutPtr, ArgMemKind::ConstPtr, ArgMemKind::NonPtr],
        );
        qcode!(
            ctx,
            "
            fn main:
                <entry>
                    call fn memcpy(@d=i64 0x401178, @s=i64 0x401178, @n=i64 0x4);
                <cont>
                    return at i64 0;
            "
        );
        let _ = (entry, cont);
        let c = emit_c(
            &ctx,
            &lower_function(&ctx, main),
            Some(&image(b"harbinger\0")),
        );

        assert!(
            !c.contains("static const char"),
            "`void *` is not `char *`:\n{c}"
        );
    }

    /// A local function that merely shares a name with a libc symbol has no
    /// prototype, so libc's types are never imposed on it.
    #[test]
    fn unprototyped_callee_is_left_alone() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn puts:
                <puts_entry>
                    return at i64 0;
            fn main:
                <entry>
                    call fn puts(@s=i64 0x401178);
                <cont>
                    return at i64 0;
            "
        );
        let _ = (puts_entry, entry, cont, puts);
        let c = emit_c(
            &ctx,
            &lower_function(&ctx, main),
            Some(&image(b"harbinger\0")),
        );

        assert!(
            !c.contains("static const char"),
            "only a prototyped external resolves against the C tables:\n{c}"
        );
    }

    /// Without an image there is nothing to read, so every address renders as
    /// the number it is.
    #[test]
    fn no_image_means_no_reconstruction() {
        let mut ctx = Context::new();
        prototyped_external(&mut ctx, "puts", vec![ArgMemKind::ConstPtr]);
        qcode!(
            ctx,
            "
            fn main:
                <entry>
                    call fn puts(@s=i64 0x401178);
                <cont>
                    return at i64 0;
            "
        );
        let _ = (entry, cont);
        let c = emit_c(&ctx, &lower_function(&ctx, main), None);

        assert!(
            !c.contains("static const char"),
            "no image, no string:\n{c}"
        );
        assert!(c.contains("puts(0x401178)"), "numeric rendering:\n{c}");
    }

    #[test]
    fn plain_ascii_is_quoted_verbatim() {
        assert_eq!(escape_c(b"harbinger"), "\"harbinger\"");
    }

    #[test]
    fn quotes_and_backslashes_are_escaped() {
        assert_eq!(escape_c(br#"a"b\c"#), r#""a\"b\\c""#);
    }

    #[test]
    fn control_characters_use_their_standard_spelling() {
        assert_eq!(escape_c(b"a\n\tb"), "\"a\\n\\tb\"");
    }

    #[test]
    fn other_non_printables_become_hex_escapes() {
        assert_eq!(escape_c(b"a\x01z"), "\"a\\x01z\"");
    }

    /// `"\x01f"` would read as the single byte `0x1f` in C, so the literal has
    /// to be split before the hex digit.
    #[test]
    fn hex_escape_followed_by_a_hex_digit_splits_the_literal() {
        assert_eq!(escape_c(b"\x01f"), "\"\\x01\" \"f\"");
        // A non-hex-digit needs no split.
        assert_eq!(escape_c(b"\x01z"), "\"\\x01z\"");
    }
}
