//! Verify each function's reverse use-def map (`users`) is consistent with its
//! live instructions.
//!
//! Since the IR-ownership refactor the `users` map is function-scoped (it lives
//! in each `Function`), maintained incrementally by `push_insn`,
//! `remove_instructions`, `replace_all_uses_with`, and
//! `replace_instruction_mnemonic`. A pass that mutates an instruction's operands
//! without routing through those helpers desynchronizes the map, which then
//! silently misreports users (e.g. DCE keeps a value it should drop, or drops
//! one it should keep). This rule recomputes the expected map from the live
//! operands and flags any divergence, in both directions:
//!
//! * a live instruction uses `v` but is missing from `users_of(v)` (stale gap);
//! * `users_of(v)` lists an instruction that no longer uses `v` (stale entry).

use rustc_hash::{FxHashMap, FxHashSet};

use qcode::{
    context::Context,
    value::{FunctionId, InstructionId, ValueId},
};

pub fn verify_users_map(ctx: &Context) -> Vec<String> {
    verify_users_map_scoped(ctx, super::Scope::All)
}

pub(crate) fn verify_users_map_scoped(ctx: &Context, scope: super::Scope<'_>) -> Vec<String> {
    // Expected users, from a single scan of every live instruction's operands,
    // keyed by owning function so each function's map is checked in isolation.
    let mut expected: FxHashMap<FunctionId, FxHashMap<ValueId, Vec<InstructionId>>> =
        FxHashMap::default();
    for insn in scope.instructions(ctx) {
        for arg in insn.operands() {
            expected
                .entry(insn.id.func)
                .or_default()
                .entry(arg)
                .or_default()
                .push(insn.id);
        }
    }

    let mut out = Vec::new();
    let empty = FxHashMap::default();
    for func in ctx.functions() {
        if !scope.contains(func.id) {
            continue;
        }
        let exp_map = expected.get(&func.id).unwrap_or(&empty);
        let mut recorded_keys: FxHashSet<ValueId> = FxHashSet::default();

        // Every recorded entry must match the live operands for that value.
        for (value, recorded) in func.user_map_entries() {
            recorded_keys.insert(value);
            let mut rec = recorded.to_vec();
            let mut exp = exp_map.get(&value).cloned().unwrap_or_default();
            rec.sort_unstable();
            exp.sort_unstable();
            if rec != exp {
                out.push(format!(
                    "function {:?}: users_of({value}) = {rec:?} but live operands give {exp:?}",
                    func.id
                ));
            }
        }

        // Every value used by a live instruction must have a recorded entry.
        for (value, exp) in exp_map {
            if !recorded_keys.contains(value) {
                out.push(format!(
                    "function {:?}: {value} is used by {exp:?} but has no users entry",
                    func.id
                ));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::context::Context;
    use qcode::value::QCodeMut;
    use wazabin_qcode_macro::qcode;

    /// A well-formed function built through the normal helpers verifies clean,
    /// including a value used twice (`@a` feeds both adds).
    #[test]
    fn consistent_map_passes() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
                <entry @a:i64>
                    %b = i64 @a + i64 0x1;
                    %c = i64 %b + i64 @a;
                    return at i64 %c;
            "
        );
        let diags = verify_users_map(&ctx);
        assert!(diags.is_empty(), "{diags:?}");
    }

    /// Removing an instruction through the normal path prunes it from its
    /// operands' user lists, so the map stays consistent afterward.
    #[test]
    fn removal_keeps_map_consistent() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
                <entry @a:i64>
                    %b = i64 @a + i64 0x1;
                    %dead = i64 @a + i64 0x2;
                    return at i64 %b;
            "
        );
        // Find and remove the unused `%dead` instruction (the second add).
        let dead = qcode::value::FunctionRef::from_id(&ctx, f)
            .blocks()
            .flat_map(|blk| blk.iter().map(|i| i.id).collect::<Vec<_>>())
            .find(|&iid| {
                ctx.users(ValueId::Instruction(iid)).is_empty()
                    && !ctx.get_insn(iid).mnemonic().is_terminator()
            })
            .expect("an unused non-terminator instruction");
        ctx.remove_instruction(dead);
        let diags = verify_users_map(&ctx);
        assert!(diags.is_empty(), "{diags:?}");
    }

    #[test]
    fn direct_operand_mutation_reports_users_disagreement() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
                <entry @a:i64 @replacement:i64>
                    %b = i64 @a + i64 0x1;
                    return at i64 %b;
            "
        );
        qcode::value::Instruction::from_id_mut(&mut ctx, b)
            .mnemonic_mut()
            .replace_value(
                ValueId::BlockParam(a).strip_func(),
                ValueId::BlockParam(replacement).strip_func(),
            );

        let diagnostics = verify_users_map(&ctx);
        assert!(
            diagnostics.iter().any(|d| d.contains("live operands give")),
            "{diagnostics:#?}"
        );
    }
}
