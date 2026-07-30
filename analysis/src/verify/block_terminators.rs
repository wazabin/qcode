//! Verify that every basic block ends in a terminator, and that a `switch`
//! terminator is well-formed.

use rustc_hash::FxHashSet as HashSet;

use qcode::{
    context::Context,
    value::{BasicBlock, BlockId, FunctionBody, insn::Mnemonic},
};

/// Every basic block must end in a terminator (branch / cbranch / return / …).
/// A block that is empty, or whose last instruction is an ordinary value op, has
/// fall-through control flow with no defined successor — a malformed CFG.
pub fn verify_block_terminators(ctx: &Context) -> Vec<String> {
    verify_block_terminators_scoped(ctx, super::Scope::All)
}

pub(crate) fn verify_block_terminators_scoped(
    ctx: &Context,
    scope: super::Scope<'_>,
) -> Vec<String> {
    let mut out = Vec::new();
    for fid in scope.function_ids(ctx) {
        let function = FunctionBody::from_id(ctx, fid);
        let fname = function.name().to_owned();
        for block in function.iter() {
            // Name and address the block, not just its arena id: a bare
            // `BlockId(153:2610)` says nothing about *which* block, and locating it
            // otherwise means re-running the lift under a debugger.
            let label = block.name().map(str::to_owned).unwrap_or_default();
            let at = block
                .address()
                .map(|a| format!(" at {a:#x}"))
                .unwrap_or_default();
            let where_ = format!("fn `{fname}` block `{label}`{at} ({:?})", block.id);
            match block.iter().last() {
                None => out.push(format!("{where_} is empty (no terminator)")),
                Some(last) if !last.mnemonic().is_terminator() => out.push(format!(
                    "{where_} does not end in a terminator (last op: `{}`)",
                    last.mnemonic().opcode()
                )),
                Some(last) => {
                    if let Mnemonic::Switch(switch) = last.mnemonic() {
                        verify_switch(ctx, switch, block.id, &where_, &mut out);
                    }
                }
            }
        }
    }
    out
}

/// Structural well-formedness of a `switch`: distinct non-empty cases, targets
/// in this function, and an argument list matching each target's parameters.
///
/// Deliberately *not* checked: that the cases cover the scrutinee's range when
/// there is no default. That is an analysis result, not a structural property —
/// deciding it needs `value_range`, and merely imprecise range inference would
/// then fail verification on correct IR.
fn verify_switch(
    ctx: &Context,
    switch: &qcode::value::insn::Switch,
    block: BlockId,
    where_: &str,
    out: &mut Vec<String>,
) {
    if switch.cases.is_empty() {
        out.push(format!("{where_} has a `switch` with no cases"));
    }

    let mut seen: HashSet<u64> = HashSet::default();
    for case in &switch.cases {
        if !seen.insert(case.value) {
            out.push(format!(
                "{where_} has a `switch` with duplicate case value {:#x}",
                case.value
            ));
        }
    }

    let arms = switch
        .cases
        .iter()
        .map(|case| (Some(case.value), case.target, &case.args))
        .chain(switch.default.map(|t| (None, t, &switch.default_args)));
    for (value, target, args) in arms {
        let which = match value {
            Some(value) => format!("case {value:#x}"),
            None => "default".to_owned(),
        };
        let target = BlockId::new(block.func, target);
        let params = BasicBlock::from_id(ctx, target).params().count();
        if params != args.len() {
            out.push(format!(
                "{where_} `switch` {which} passes {} argument(s) to a target with {params} \
                 parameter(s)",
                args.len(),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode_macro::qcode;

    /// A well-formed dispatch, including an arm that passes a block argument.
    #[test]
    fn accepts_a_well_formed_switch() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
                <entry @i:i64 @p:i64>
                    switch @i { 0x0 => <a>, 0x3 => <b @v=@p>, default => <d> };
                <a>
                    return @i;
                <b @v:i64>
                    return @v;
                <d>
                    return @i;
            "
        );
        assert!(verify_block_terminators(&ctx).is_empty());
    }

    /// Two arms selecting on the same value make the dispatch ambiguous.
    #[test]
    fn rejects_duplicate_case_values() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
                <entry @i:i64>
                    switch @i { 0x1 => <a>, 0x1 => <b> };
                <a>
                    return @i;
                <b>
                    return @i;
            "
        );
        let errors = verify_block_terminators(&ctx);
        assert!(
            errors.iter().any(|e| e.contains("duplicate case value")),
            "expected a duplicate-case diagnostic, got {errors:?}"
        );
    }

    /// An arm must bind exactly the parameters its target declares.
    #[test]
    fn rejects_an_arm_with_the_wrong_argument_count() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
                <entry @i:i64>
                    switch @i { 0x0 => <a>, 0x1 => <b> };
                <a>
                    return @i;
                <b @v:i64>
                    return @v;
            "
        );
        let errors = verify_block_terminators(&ctx);
        assert!(
            errors.iter().any(|e| e.contains("argument(s) to a target")),
            "expected an arity diagnostic, got {errors:?}"
        );
    }
}
