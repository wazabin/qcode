//! Verify each pointer value is used in at most one address space.

use std::collections::HashMap;

use qcode::{
    context::Context,
    space::SpaceId,
    value::{ValueId, insn::Mnemonic},
};

/// A pointer *value* (the `ptr` operand of a load/store) must refer to a single
/// address space across all its uses. The simple alias analysis relies on this
/// (it keys equivalence classes by space), so a value spanning two spaces — e.g. a
/// shadow-redirected access merged with a real-ram access — is a corruption. Mirrors
/// the runtime assertion in `alias::simple`, surfaced here so `verify_after` pins it
/// to the pass that produced it rather than to whichever pass next runs alias.
pub fn verify_pointer_spaces(ctx: &Context) -> Vec<String> {
    let mut seen: HashMap<ValueId, SpaceId> = HashMap::new();
    let mut out = Vec::new();
    for insn in ctx.instructions() {
        let (ptr, space) = match insn.mnemonic() {
            Mnemonic::Load(l) => (l.ptr.qualify(insn.id.func), l.space),
            Mnemonic::Store(s) => (s.ptr.qualify(insn.id.func), s.space),
            _ => continue,
        };
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
