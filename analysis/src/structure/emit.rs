//! Pseudo-C emission for a lowered [`Program`].
//!
//! Emits a stream of classified [`TokenLine`]s: compound C statements (pure
//! single-use values are folded into their users; only side-effecting or
//! multiply-used values become their own assignments), syntax-classified for
//! highlighting, and tagged with indentation and block provenance.

use std::collections::HashSet;

use qcode::{
    context::Context,
    value::{
        BlockId, Function, Instruction, InstructionId, Value, ValueId, ValueRef, Varnode,
        VarnodeId, function::FunctionId, insn::Mnemonic,
    },
};

use super::{
    ast::{Program, Stmt},
    lower_expr::{
        deref_location, instruction_name, lower_condition, lower_defining_expr, lower_expr_rooted,
    },
    tokens::{LineBuf, TokenKind, TokenLine},
};

/// Renders `program` as classified, indented token lines.
///
/// When the program knows its source function, the body is wrapped in a
/// `fn name(args) -> rets {` … `}` header and indented one level beneath it.
pub fn emit_tokens(ctx: &Context, program: &Program) -> Vec<TokenLine> {
    let roots = compute_roots(ctx, program);
    let mut out = Vec::new();
    match program.function {
        Some(function_id) => {
            out.push(header_line(ctx, function_id));
            emit_stmts(ctx, program, &program.stmts, 1, &roots, &mut out);
            out.push(brace_line("}", 0));
        }
        None => emit_stmts(ctx, program, &program.stmts, 0, &roots, &mut out),
    }
    out
}

/// Builds the `fn name(inputs) -> outputs {` header from the function's
/// recovered signature. A missing signature (or an empty input/output list)
/// simply renders empty parentheses / no return arrow.
fn header_line(ctx: &Context, function_id: FunctionId) -> TokenLine {
    let function = Function::from_id(ctx, function_id);
    let sig = function.signature();

    let mut buf = LineBuf::default();
    buf.keyword("fn");
    buf.space();
    buf.push(function.name().to_string(), TokenKind::Label);

    buf.punct("(");
    if let Some(inputs) = sig.and_then(|s| s.inputs.as_deref()) {
        emit_regs(ctx, inputs, &mut buf);
    }
    buf.punct(")");

    if let Some(outputs) = sig
        .and_then(|s| s.outputs.as_deref())
        .filter(|o| !o.is_empty())
    {
        buf.space();
        buf.punct("->");
        buf.space();
        emit_regs(ctx, outputs, &mut buf);
    }

    buf.space();
    buf.punct("{");
    buf.into_line(0, None)
}

/// Pushes a comma-separated list of register names (as variables) onto `buf`.
fn emit_regs(ctx: &Context, regs: &[VarnodeId], buf: &mut LineBuf) {
    for (i, &id) in regs.iter().enumerate() {
        if i > 0 {
            buf.punct(",");
            buf.space();
        }
        buf.push(Varnode::from_id(ctx, id).to_string(), TokenKind::Variable);
    }
}

/// Renders `program` as pseudo-C source text (indentation via spaces).
pub fn emit_c(ctx: &Context, program: &Program) -> String {
    let mut out = String::new();
    for line in emit_tokens(ctx, program) {
        out.push_str(&"    ".repeat(line.indent));
        out.push_str(&line.text());
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------
// Root selection: which instructions get their own statement.
// ---------------------------------------------------------------------------

/// The instructions that are emitted as their own statements. Everything else
/// (pure, used exactly once) is inlined into its user.
fn compute_roots(ctx: &Context, program: &Program) -> HashSet<InstructionId> {
    let mut raws = Vec::new();
    collect_raws(&program.stmts, &mut raws);
    raws.into_iter().filter(|&id| is_root(ctx, id)).collect()
}

fn collect_raws(stmts: &[Stmt], out: &mut Vec<InstructionId>) {
    for stmt in stmts {
        match stmt {
            Stmt::Raw(id) => out.push(*id),
            Stmt::If { then, els, .. } => {
                collect_raws(then, out);
                collect_raws(els, out);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::Loop { body } => {
                collect_raws(body, out);
            }
            Stmt::Switch { cases, default, .. } => {
                for case in cases {
                    collect_raws(&case.body, out);
                }
                collect_raws(default, out);
            }
            _ => {}
        }
    }
}

/// Whether instruction `id` is emitted as its own statement, rather than inlined
/// into its single user. This is the one inlining contract the backend shares:
/// [`refine::is_condition_only`](super::refine::is_condition_only) is exactly its
/// negation for a [`Stmt::Raw`], so a statement one phase treats as invisibly
/// folded is one the emitter actually inlines.
pub(crate) fn is_root(ctx: &Context, id: InstructionId) -> bool {
    let insn = Instruction::from_id(ctx, id);
    if is_side_effecting(insn.mnemonic()) {
        return true;
    }
    // A value used by more than one instruction is named to avoid duplicating
    // its expression; a value used zero times is dead and gets elided (not a
    // root). Exactly-once uses are inlined into their user — except a memory
    // load, which reads state a later store or call can change, so it may fold
    // into its user only when nothing between them can write that memory.
    let users = ctx.users(ValueId::Instruction(id));
    if users.len() != 1 {
        return users.len() > 1;
    }
    is_memory_read(insn.mnemonic()) && !load_safe_to_fold(ctx, id, users[0])
}

/// Whether a single-use load may be inlined into its user rather than named:
/// only when the user is in the same block and no instruction between them can
/// write memory, so the loaded value cannot have changed at the point of use.
/// Otherwise folding it would reorder the read past an aliasing write (the
/// classic `t = *x; *x = …; use(t)` hazard).
fn load_safe_to_fold(ctx: &Context, load: InstructionId, user: InstructionId) -> bool {
    let load_insn = Instruction::from_id(ctx, load);
    let Some(block) = load_insn.block() else {
        return false;
    };
    if Instruction::from_id(ctx, user).block().map(|b| b.id) != Some(block.id) {
        return false;
    }
    // Walk the block from the load to its user; any memory writer in between
    // (store, call, or other side effect) forbids the fold.
    let mut after_load = false;
    for insn in block.instructions() {
        if insn.id == load {
            after_load = true;
        } else if insn.id == user {
            return after_load;
        } else if after_load && is_side_effecting(insn.mnemonic()) {
            return false;
        }
    }
    false
}

/// Whether an instruction must always be a statement because it has effects.
pub(crate) fn is_side_effecting(m: &Mnemonic) -> bool {
    matches!(
        m,
        Mnemonic::Store(_)
            | Mnemonic::Call(_)
            | Mnemonic::CallInd(_)
            | Mnemonic::Return(_)
            | Mnemonic::PCodeOp(_)
            | Mnemonic::Assert(_)
    )
}

/// Whether an instruction reads memory — a load, whose result is only valid
/// until the next aliasing write. Such values are not freely foldable.
pub(crate) fn is_memory_read(m: &Mnemonic) -> bool {
    matches!(m, Mnemonic::Load(_))
}

// ---------------------------------------------------------------------------
// Statement emission.
// ---------------------------------------------------------------------------

fn emit_stmts(
    ctx: &Context,
    program: &Program,
    stmts: &[Stmt],
    indent: usize,
    roots: &HashSet<InstructionId>,
    out: &mut Vec<TokenLine>,
) {
    let mut line_indent = indent;
    for stmt in stmts {
        if matches!(stmt, Stmt::Label(_)) {
            emit_stmt(ctx, program, stmt, indent, roots, out);
            line_indent = indent + 1;
        } else {
            emit_stmt(ctx, program, stmt, line_indent, roots, out);
        }
    }
}

fn emit_stmt(
    ctx: &Context,
    program: &Program,
    stmt: &Stmt,
    indent: usize,
    roots: &HashSet<InstructionId>,
    out: &mut Vec<TokenLine>,
) {
    match stmt {
        Stmt::Label(block) => {
            let mut buf = LineBuf::default();
            buf.push(label_of(program, *block), TokenKind::Label);
            buf.punct(":");
            out.push(buf.into_line(indent, Some(*block)));
        }
        Stmt::Raw(id) => {
            // Non-root values are inlined into their users, not emitted here.
            if !roots.contains(id) {
                return;
            }
            let block = Instruction::from_id(ctx, *id).block().map(|b| b.id);
            let mut buf = statement(ctx, *id, roots);
            // The statement's operand expressions were lowered with `Some(roots)`,
            // so `buf.insns` holds every genuinely inlined operand plus any root
            // referenced by name. Drop the named roots (each is its own line) and
            // add the statement's own instruction.
            buf.insns.retain(|i| !roots.contains(i));
            buf.note_insn(*id);
            out.push(buf.into_line(indent, block));
        }
        Stmt::Assign { param, value } => {
            // A phi copy `param = value;`. The value is lowered against the roots
            // so an inlined operand renders inline and a named root by name.
            let mut buf = LineBuf::default();
            lower_expr_rooted(ctx, *param, roots).write_tokens(&mut buf);
            assign(&mut buf);
            lower_expr_rooted(ctx, *value, roots).write_tokens(&mut buf);
            buf.punct(";");
            out.push(buf.into_line(indent, None));
        }
        Stmt::Goto(target) => {
            let mut buf = LineBuf::default();
            buf.keyword("goto");
            buf.space();
            buf.push(label_of(program, *target), TokenKind::Label);
            buf.punct(";");
            out.push(buf.into_line(indent, None));
        }
        Stmt::GotoIf { cond, target } => {
            let mut buf = LineBuf::default();
            buf.keyword("if");
            buf.space();
            buf.punct("(");
            lower_condition(ctx, cond, roots).write_tokens(&mut buf);
            buf.punct(")");
            buf.space();
            buf.keyword("goto");
            buf.space();
            buf.push(label_of(program, *target), TokenKind::Label);
            buf.punct(";");
            out.push(buf.into_line(indent, None));
        }
        Stmt::If { cond, then, els } => {
            let mut head = LineBuf::default();
            head.keyword("if");
            head.space();
            head.punct("(");
            lower_condition(ctx, cond, roots).write_tokens(&mut head);
            head.punct(")");
            head.space();
            head.punct("{");
            out.push(head.into_line(indent, None));

            emit_stmts(ctx, program, then, indent + 1, roots, out);

            if els.is_empty() {
                out.push(brace_line("}", indent));
            } else {
                out.push(brace_line("} else {", indent));
                emit_stmts(ctx, program, els, indent + 1, roots, out);
                out.push(brace_line("}", indent));
            }
        }
        Stmt::While { cond, body } => {
            let mut head = LineBuf::default();
            head.keyword("while");
            head.space();
            head.punct("(");
            lower_condition(ctx, cond, roots).write_tokens(&mut head);
            head.punct(")");
            head.space();
            head.punct("{");
            out.push(head.into_line(indent, None));
            emit_stmts(ctx, program, body, indent + 1, roots, out);
            out.push(brace_line("}", indent));
        }
        Stmt::DoWhile { cond, body } => {
            out.push(brace_line("do {", indent));
            emit_stmts(ctx, program, body, indent + 1, roots, out);
            let mut tail = LineBuf::default();
            tail.punct("}");
            tail.space();
            tail.keyword("while");
            tail.space();
            tail.punct("(");
            lower_condition(ctx, cond, roots).write_tokens(&mut tail);
            tail.punct(")");
            tail.punct(";");
            out.push(tail.into_line(indent, None));
        }
        Stmt::Loop { body } => {
            let mut head = LineBuf::default();
            head.keyword("while");
            head.space();
            head.punct("(");
            head.keyword("true");
            head.punct(")");
            head.space();
            head.punct("{");
            out.push(head.into_line(indent, None));
            emit_stmts(ctx, program, body, indent + 1, roots, out);
            out.push(brace_line("}", indent));
        }
        Stmt::Break => {
            let mut buf = LineBuf::default();
            buf.keyword("break");
            buf.punct(";");
            out.push(buf.into_line(indent, None));
        }
        Stmt::Continue => {
            let mut buf = LineBuf::default();
            buf.keyword("continue");
            buf.punct(";");
            out.push(buf.into_line(indent, None));
        }
        Stmt::Switch {
            scrutinee,
            cases,
            default,
        } => {
            // Rust-style `match`: no fallthrough (so no `break`), fat-arrow arms,
            // `|`-joined labels, and a `_` catch-all.
            let mut head = LineBuf::default();
            head.keyword("match");
            head.space();
            scrutinee.write_tokens(&mut head);
            head.space();
            head.punct("{");
            out.push(head.into_line(indent, None));

            // Case labels are interpreted at the scrutinee's width, so a value
            // whose sign bit is set there prints as a signed decimal (`-1`)
            // rather than a full-width unsigned constant (`0xffffffffffffffff`).
            let width = scrutinee.value.map(|v| ValueRef::new(v, ctx).size());
            for case in cases {
                let mut arm = LineBuf::default();
                for (i, &v) in case.values.iter().enumerate() {
                    if i > 0 {
                        arm.space();
                        arm.punct("|");
                        arm.space();
                    }
                    arm.push(format_case_value(v, width), TokenKind::Number);
                }
                arm.space();
                arm.punct("=>");
                arm.space();
                arm.punct("{");
                // Map the arm's labels back to the folded-away equality tests.
                for &insn in &case.insns {
                    arm.note_insn(insn);
                }
                out.push(arm.into_line(indent + 1, None));
                emit_stmts(ctx, program, &case.body, indent + 2, roots, out);
                out.push(brace_line("}", indent + 1));
            }

            // `match` must be exhaustive, so emit the catch-all even when empty.
            let mut arm = LineBuf::default();
            arm.push("_", TokenKind::Keyword);
            arm.space();
            arm.punct("=>");
            arm.space();
            arm.punct("{");
            out.push(arm.into_line(indent + 1, None));
            emit_stmts(ctx, program, default, indent + 2, roots, out);
            out.push(brace_line("}", indent + 1));

            out.push(brace_line("}", indent));
        }
    }
}

/// Builds the tokens for a single root instruction rendered as a C statement.
fn statement(ctx: &Context, id: InstructionId, roots: &HashSet<InstructionId>) -> LineBuf {
    let insn = Instruction::from_id(ctx, id);
    let mut buf = LineBuf::default();
    match insn.mnemonic() {
        Mnemonic::Store(s) => {
            // <location> = src;  (a named varnode reads as the variable, a
            // computed address as `*ptr`).
            deref_location(ctx, s.ptr, s.size, Some(roots)).write_tokens(&mut buf);
            assign(&mut buf);
            lower_expr_rooted(ctx, s.src, roots).write_tokens(&mut buf);
            buf.punct(";");
        }
        Mnemonic::Return(r) => {
            buf.keyword("return");
            if let Some(value) = r.value {
                buf.space();
                lower_expr_rooted(ctx, value, roots).write_tokens(&mut buf);
            }
            buf.punct(";");
        }
        Mnemonic::Call(c) => {
            let name = Function::from_id(ctx, c.target).name().to_string();
            buf.push_function(name, TokenKind::Label, Some(c.target));
            call_args(ctx, &c.args, roots, &mut buf);
            buf.punct(";");
        }
        Mnemonic::CallInd(c) => {
            buf.punct("(");
            buf.push("*", TokenKind::Operator);
            lower_expr_rooted(ctx, c.ptr, roots).write_tokens(&mut buf);
            buf.punct(")");
            call_args(ctx, &c.args, roots, &mut buf);
            buf.punct(";");
        }
        // A named value: `name = <defining expression>;`.
        _ => {
            buf.push_value(
                instruction_name(ctx, id),
                TokenKind::Variable,
                Some(ValueId::Instruction(id)),
            );
            assign(&mut buf);
            lower_defining_expr(ctx, id, roots).write_tokens(&mut buf);
            buf.punct(";");
        }
    }
    buf
}

fn call_args(ctx: &Context, args: &[ValueId], roots: &HashSet<InstructionId>, buf: &mut LineBuf) {
    buf.punct("(");
    for (i, &arg) in args.iter().enumerate() {
        if i > 0 {
            buf.punct(",");
            buf.space();
        }
        lower_expr_rooted(ctx, arg, roots).write_tokens(buf);
    }
    buf.punct(")");
}

fn assign(buf: &mut LineBuf) {
    buf.space();
    buf.push("=", TokenKind::Operator);
    buf.space();
}

fn brace_line(text: &str, indent: usize) -> TokenLine {
    let mut buf = LineBuf::default();
    buf.punct(text);
    buf.into_line(indent, None)
}

/// Formats a `switch`/`match` case label. A constant whose sign bit is set at the
/// scrutinee's `width` (in bytes) prints as a signed decimal — `-1` rather than
/// the sign-extended `0xffffffffffffffff` — so negative case values read
/// naturally. Non-negative values (and values of unknown width) stay hex.
fn format_case_value(v: u64, width: Option<usize>) -> String {
    if let Some(w) = width
        && (1..=8).contains(&w)
    {
        let bits = w * 8;
        let fits = bits == 64 || v < (1u64 << bits);
        let negative = (v >> (bits - 1)) & 1 == 1;
        if fits && negative {
            let signed = if bits == 64 {
                v as i64
            } else {
                v as i64 - (1i64 << bits)
            };
            return format!("{signed}");
        }
    }
    format!("0x{v:x}")
}

/// The label name for `block`, falling back to a synthetic name if the program
/// somehow lacks one (should not happen for well-formed programs).
fn label_of(program: &Program, block: BlockId) -> String {
    program
        .label(block)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("bb_{}", Into::<usize>::into(block)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::{decompile_function, lower_function, tokens::TokenKind};
    use qcode_macro::qcode;

    #[test]
    fn single_use_values_fold_into_a_compound_statement() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 x;
            varnode i32 y;

            fn f:
            <entry>
                %a = load(i32, &x);
                %b = i32 %a + i32 0x1;
                store(&y, %b);
                local i64 ptr;
                return [ptr];
            "
        );

        let program = lower_function(&ctx, f);
        let c = emit_c(&ctx, &program);

        // The load and the add are single-use, so they fold into the store; no
        // intermediate assignments are emitted.
        assert!(
            c.contains("y = x + 0x1;"),
            "expected folded compound store, got:\n{c}"
        );
        assert!(!c.contains("%a"), "the load should be inlined:\n{c}");
        assert!(!c.contains("%b"), "the add should be inlined:\n{c}");
    }

    #[test]
    fn condition_references_root_not_reread_memory() {
        // A load kept live across an aliasing store becomes a named root. The
        // branch that tests it must reference that name, not re-inline the
        // dereference and re-read memory after the store.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 pp;
            varnode i64 qq;
            varnode i8 flag;

            fn f:
            <entry>
                %p = load(i64, &pp);
                %v = load(i8, %p);
                %q = load(i64, &qq);
                store(%q, i8 0x0);
                if %v goto <then_lbl> else goto <merge>;
            <then_lbl>
                store(&flag, i8 0x1);
                goto <merge>;
            <merge>
                local i64 ptr;
                return [ptr];
            "
        );
        let c = emit_c(&ctx, &decompile_function(&ctx, f).unwrap());
        // The load is named (a root because of the intervening store) and the
        // condition uses that name rather than re-dereferencing memory. The
        // byte-wide load through `pp` carries its access width as a pointer cast.
        assert!(
            c.contains("v = *(uint8_t *)pp;"),
            "load should be a named, width-annotated root:\n{c}"
        );
        assert!(
            c.contains("if (v)") && !c.contains("(uint8_t *)pp)"),
            "condition should reference the root, not re-read memory:\n{c}"
        );
    }

    #[test]
    fn signed_comparison_casts_its_operand() {
        // A signed `s<` casts its operand so the output is not silently unsigned.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 x;
            varnode i32 y;

            fn f:
            <entry>
                %a = load(i32, &x);
                %c = i32 %a s< i32 0x5;
                if %c goto <then_lbl> else goto <merge>;
            <then_lbl>
                store(&y, i32 0x1);
                goto <merge>;
            <merge>
                local i64 ptr;
                return [ptr];
            "
        );
        let c = emit_c(&ctx, &decompile_function(&ctx, f).unwrap());
        assert!(
            c.contains("(int32_t)x") && c.contains("< 0x5"),
            "signed compare should cast its operand:\n{c}"
        );
    }

    #[test]
    fn unsigned_comparison_stays_bare() {
        // The unsigned counterpart of the signed test: no cast.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 x;
            varnode i32 y;

            fn f:
            <entry>
                %a = load(i32, &x);
                %c = i32 %a < i32 0x5;
                if %c goto <then_lbl> else goto <merge>;
            <then_lbl>
                store(&y, i32 0x1);
                goto <merge>;
            <merge>
                local i64 ptr;
                return [ptr];
            "
        );
        let c = emit_c(&ctx, &decompile_function(&ctx, f).unwrap());
        assert!(
            !c.contains("(int32_t)") && c.contains("x < 0x5"),
            "unsigned compare should stay bare:\n{c}"
        );
    }

    #[test]
    fn sign_extension_extends_the_sign_not_zero() {
        // `sext` reads its source as signed before widening, rendering
        // `(int64_t)(int32_t)x`; a plain `(int64_t)x` would zero-extend in C.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 x;
            varnode i64 z;

            fn f:
            <entry>
                %a = load(i32, &x);
                %w = sext(i64, %a);
                store(&z, %w);
                local i64 ptr;
                return [ptr];
            "
        );
        let c = emit_c(&ctx, &lower_function(&ctx, f));
        assert!(
            c.contains("(int64_t)(int32_t)x"),
            "sext should sign-extend, not zero-extend:\n{c}"
        );
    }

    #[test]
    fn loads_do_not_fold_across_an_aliasing_store() {
        // The classic swap: `t = *x; u = *y; *x = u; *y = t`. If the single-use
        // loads folded into the stores, the output would collapse to `x = y; y =
        // x`, losing the old value of x. Each load must instead be named, because
        // an aliasing store sits between it and its use.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 x;
            varnode i32 y;

            fn f:
            <entry>
                %t = load(i32, &x);
                %u = load(i32, &y);
                store(&x, %u);
                store(&y, %t);
                local i64 ptr;
                return [ptr];
            "
        );

        let program = lower_function(&ctx, f);
        let c = emit_c(&ctx, &program);

        // The old value of x must be saved to a temp before the store to x
        // clobbers it — that load may not fold past the intervening store.
        assert!(c.contains("t = x;"), "load of x should be named:\n{c}");
        assert!(
            c.contains("y = t;"),
            "y must be restored from the saved temp:\n{c}"
        );
        // The bug: folding the load of x into `*y = …` past the store to x, so
        // `y` reads the already-overwritten `x`.
        assert!(
            !c.contains("y = x;"),
            "load folded across the aliasing store to x:\n{c}"
        );
    }

    #[test]
    fn partial_width_access_carries_its_size() {
        // A byte load and a word store through computed pointers must render their
        // access width, so a partial access is not mistaken for a full-width one.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 pp;
            varnode i64 qq;

            fn f:
            <entry>
                %p = load(i64, &pp);
                %b = load(i8, %p);
                %q = load(i64, &qq);
                store(%q, i32 0x0);
                store(%p, i8 %b);
                local i64 ptr;
                return [ptr];
            "
        );

        let program = lower_function(&ctx, f);
        let c = emit_c(&ctx, &program);
        // Byte accesses through pointer `p`, word store through `qq`.
        assert!(
            c.contains("*(uint8_t *)p") && c.contains("*(uint32_t *)qq"),
            "loads/stores should carry their access width:\n{c}"
        );
    }

    #[test]
    fn folded_expression_preserves_required_parentheses() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 a;

            fn f:
            <entry>
                %sum = i32 0x1 + i32 0x2;
                %scaled = i32 %sum * i32 0x5;
                store(&a, %scaled);
                local i64 ptr;
                return [ptr];
            "
        );

        let program = lower_function(&ctx, f);
        let c = emit_c(&ctx, &program);

        assert!(
            c.contains("a = (0x1 + 0x2) * 0x5;"),
            "expected parenthesized folded expression, got:\n{c}"
        );
        assert!(
            !c.contains("%sum") && !c.contains("%scaled"),
            "single-use intermediates should be folded away:\n{c}"
        );
    }

    #[test]
    fn structured_tokens_carry_nested_indentation() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;
            varnode i32 x;

            fn f:
            <entry>
                %c = load(i8, &cond);
                if %c goto <then_lbl> else goto <merge>;
            <then_lbl>
                store(&x, i32 0x1);
                goto <merge>;
            <merge>
                local i64 ptr;
                return [ptr];
            "
        );

        let program = decompile_function(&ctx, f).unwrap();
        let lines = emit_tokens(&ctx, &program);

        // Depth 2: one level for the `fn` header, one for the enclosing `if`.
        assert!(
            lines
                .iter()
                .any(|line| line.indent == 2 && line.text() == "x = 0x1;"),
            "expected nested store to carry indent depth 2, got:\n{lines:#?}"
        );
    }

    #[test]
    fn flat_labels_indent_their_block_body() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;

            fn f:
            <entry>
                %c = load(i8, &cond);
                if %c goto <body> else goto <exit_lbl>;
            <body>
                goto <entry>;
            <exit_lbl>
                local i64 ptr;
                return [ptr];
            "
        );

        // The flat lowering always labels every block and indents its body; test
        // it directly rather than relying on a structuring fallback.
        let program = lower_function(&ctx, f);
        let lines = emit_tokens(&ctx, &program);

        // Everything sits one level in from the `fn` header: the label at depth
        // 1, its block body at depth 2.
        assert!(
            lines
                .iter()
                .any(|line| line.indent == 1 && line.text() == "entry:"),
            "expected label one level under the fn header, got:\n{lines:#?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.indent == 2 && line.text().starts_with("if (cond) goto ")),
            "expected flat block body to be indented under its label, got:\n{lines:#?}"
        );
    }

    #[test]
    fn folded_statement_maps_back_to_its_source_instructions() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 x;
            varnode i32 y;

            fn f:
            <entry>
                %a = load(i32, &x);
                %b = i32 %a + i32 0x1;
                store(&y, %b);
                local i64 ptr;
                return [ptr];
            "
        );

        let program = lower_function(&ctx, f);
        let lines = emit_tokens(&ctx, &program);

        // The single compound store line must carry all three folded-in
        // instructions (the load, the add, and the store itself).
        let store_line = lines
            .iter()
            .find(|l| l.text() == "y = x + 0x1;")
            .expect("expected the folded store line");
        let opcodes: Vec<String> = store_line
            .insns
            .iter()
            .map(|&id| Instruction::from_id(&ctx, id).opcode().to_string())
            .collect();
        assert_eq!(
            store_line.insns.len(),
            3,
            "load + add + store should all map to this line, got {opcodes:?}"
        );

        // A value token carries the IR value it renders, so a front-end can
        // highlight its uses; plain punctuation does not.
        let x_token = store_line
            .tokens
            .iter()
            .find(|t| t.text == "x")
            .expect("expected the `x` variable token");
        assert!(x_token.value.is_some(), "value tokens carry provenance");
        let eq_token = store_line.tokens.iter().find(|t| t.text == "=").unwrap();
        assert!(eq_token.value.is_none(), "punctuation carries no value");
    }

    #[test]
    fn condition_line_maps_back_to_its_comparison_and_load() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 x;
            varnode i32 y;

            fn f:
            <entry>
                %v = load(i32, &x);
                %c = i32 %v == i32 0x5;
                if %c goto <then_lbl> else goto <merge>;
            <then_lbl>
                store(&y, i32 0x1);
                goto <merge>;
            <merge>
                local i64 ptr;
                return [ptr];
            "
        );

        let program = decompile_function(&ctx, f).unwrap();
        let lines = emit_tokens(&ctx, &program);

        // The `if (x == 0x5)` header folds in the comparison and the load, both
        // of which must be recoverable from the line's instruction set.
        let cond_line = lines
            .iter()
            .find(|l| l.text().starts_with("if (") && l.text().contains("=="))
            .expect("expected the folded if header");
        assert!(
            cond_line.insns.len() >= 2,
            "the compare and the load should both map to the condition, got {:?}",
            cond_line
                .insns
                .iter()
                .map(|&id| Instruction::from_id(&ctx, id).opcode().to_string())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn tokens_classify_keywords_numbers_and_operators() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 y;

            fn f:
            <entry>
                store(&y, i32 0x2a);
                local i64 ptr;
                return [ptr];
            "
        );

        let program = lower_function(&ctx, f);
        let lines = emit_tokens(&ctx, &program);
        let kinds: Vec<TokenKind> = lines
            .iter()
            .flat_map(|l| l.tokens.iter().map(|t| t.kind))
            .collect();
        assert!(kinds.contains(&TokenKind::Keyword), "return is a keyword");
        assert!(kinds.contains(&TokenKind::Number), "0x2a is a number");
        assert!(
            kinds.contains(&TokenKind::Operator),
            "= and * are operators"
        );
    }
}
