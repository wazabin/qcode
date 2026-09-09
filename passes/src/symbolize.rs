//! Attaching meaning to bare integer literals.
//!
//! A lifted program is full of constants that are really references: a call
//! target, a branch destination, the address of a string. The lifter cannot
//! tell those apart from arithmetic, so it emits them as plain literals and
//! leaves the question open. These passes answer it, by consulting the address
//! index for code references and the containing binary for data ones, and
//! record the answer as a [`SymbolicRef`] on the literal itself.

use qcode::{
    address_index::{AddressIndex, AddressTarget},
    context::Context,
    value::literal::SymbolicRef,
};

/// Annotate every literal that names a known function or block.
pub fn resolve_addresses(ctx: &mut Context) {
    let addresses = AddressIndex::analyze(ctx);
    let pairs: Vec<(_, u64)> = ctx
        .shared
        .values
        .literals
        .iter()
        .map(|item| (item.id, item.value))
        .collect();

    for (id, value) in pairs {
        let addr = value;
        let symbolic = match addresses.get(addr) {
            Some(AddressTarget::Function(function)) => Some(SymbolicRef::Function(function)),
            Some(AddressTarget::Block(block)) => Some(SymbolicRef::Block(block)),
            None => None,
        };
        if let Some(sym) = symbolic {
            ctx.shared.values.literals[id].symbolic = Some(sym);
        }
    }
}

/// Annotate every still-unresolved literal that points at a printable string.
///
/// `read_cstring` reads a NUL-terminated printable ASCII string from an
/// address in the containing binary, returning `None` when the address holds
/// no such string. It is a closure rather than a binary-format trait so this
/// crate stays independent of any particular container format; callers holding
/// a `BinaryFormat` pass `|addr| fmt.read_printable_cstring(addr, Some(256))`.
///
/// Literals that already carry a [`SymbolicRef`] are left alone, so running
/// [`resolve_addresses`] first gives code references priority over data ones.
pub fn resolve_strings(ctx: &mut Context, read_cstring: impl Fn(u64) -> Option<Vec<u8>>) {
    let pairs: Vec<_> = ctx
        .shared
        .values
        .literals
        .iter()
        .filter(|item| item.symbolic.is_none())
        .map(|item| (item.id, item.value))
        .collect();

    for (id, value) in pairs {
        let Some(bytes) = read_cstring(value) else {
            continue;
        };
        if bytes.is_empty() {
            continue;
        }
        let s = String::from_utf8(bytes).expect("printable ASCII is valid UTF-8");
        ctx.shared.values.literals[id].symbolic = Some(SymbolicRef::String(s));
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_strings;
    use qcode::{context::Context, value::literal::SymbolicRef};

    /// A stand-in for a binary that holds `"hello"` at `0x1000`.
    fn hello_at_0x1000(addr: u64) -> Option<Vec<u8>> {
        (addr == 0x1000).then(|| b"hello".to_vec())
    }

    #[test]
    fn resolve_strings_annotates_printable_cstrings() {
        let mut ctx = Context::new();
        let lit_id = ctx.get_const(0x1000, 8).id;

        resolve_strings(&mut ctx, hello_at_0x1000);

        assert!(matches!(
            &ctx.shared.values.literals[lit_id].symbolic,
            Some(SymbolicRef::String(s)) if s == "hello"
        ));
    }

    #[test]
    fn resolve_strings_leaves_unreadable_addresses_alone() {
        let mut ctx = Context::new();
        let lit_id = ctx.get_const(0x2000, 8).id;

        resolve_strings(&mut ctx, hello_at_0x1000);

        assert!(ctx.shared.values.literals[lit_id].symbolic.is_none());
    }
}
