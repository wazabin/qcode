//! Verify each pointer value is used in at most one address space.

use std::collections::HashMap;

use qcode::{
    context::Context,
    space::MemorySpaceId,
    value::{ValueId, insn::Mnemonic},
};

/// A pointer *value* (the `ptr` operand of a load/store) must refer to a single
/// address space across all its uses. The simple alias analysis relies on this
/// (it keys equivalence classes by space), so a value spanning two spaces — e.g. a
/// shadow-redirected access merged with a real-ram access — is a corruption. Mirrors
/// the runtime assertion in `alias::simple`, surfaced here so `verify_after` pins it
/// to the pass that produced it rather than to whichever pass next runs alias.
pub fn verify_pointer_spaces(ctx: &Context) -> Vec<String> {
    verify_pointer_spaces_scoped(ctx, super::Scope::All)
}

pub(crate) fn verify_pointer_spaces_scoped(ctx: &Context, scope: super::Scope<'_>) -> Vec<String> {
    // A pointer value is function-local (operands are body-local), so scoping
    // the scan per function cannot miss a cross-function space conflict.
    let mut seen: HashMap<ValueId, MemorySpaceId> = HashMap::new();
    let mut out = Vec::new();
    for insn in scope.instructions(ctx) {
        let (ptr, space) = match insn.mnemonic() {
            Mnemonic::Load(l) => (l.ptr.qualify(insn.id.func), l.space),
            Mnemonic::Store(s) => (s.ptr.qualify(insn.id.func), s.space),
            _ => continue,
        };
        let space = space.qualify(insn.id.func);
        match seen.get(&ptr) {
            Some(&prev) if prev != space => out.push(format!(
                "pointer {ptr:?} used in multiple spaces ({prev:?} vs {space:?})"
            )),
            Some(_) => {}
            None => {
                seen.insert(ptr, space);
            }
        }
    }
    out
}
