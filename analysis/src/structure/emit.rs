//! Pseudo-C emission for a lowered [`Program`].
//!
//! Emits a stream of classified [`TokenLine`]s: compound C statements (pure
//! single-use values are folded into their users; only side-effecting or
//! multiply-used values become their own assignments), syntax-classified for
//! highlighting, and tagged with indentation and block provenance.

use std::collections::HashSet;

use qcode::{
    context::Context,
    value::{BlockId, Function, Instruction, InstructionId, ValueId, insn::Mnemonic},
};

use super::{
    ast::{Program, Stmt},
    lower_expr::{deref_location, instruction_name, lower_defining_expr, lower_expr_rooted},
    tokens::{LineBuf, TokenKind, TokenLine},
};

/// Renders `program` as classified, indented token lines.
pub fn emit_tokens(ctx: &Context, program: &Program) -> Vec<TokenLine> {
    let roots = compute_roots(ctx, program);
    let mut out = Vec::new();
    emit_stmts(ctx, program, &program.stmts, 0, &roots, &mut out);
    out
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
            _ => {}
        }
    }
}

fn is_root(ctx: &Context, id: InstructionId) -> bool {
    let insn = Instruction::from_id(ctx, id);
    if is_side_effecting(insn.mnemonic()) {
        return true;
    }
    // A value used by more than one instruction is named to avoid duplicating
    // its expression; a value used zero times is dead and gets elided (not a
    // root). Exactly-once uses are inlined into their user.
    ctx.users(ValueId::Instruction(id)).len() > 1
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
            out.push(statement(ctx, *id, roots).into_line(indent, block));
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
            cond.write_tokens(&mut buf);
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
            cond.write_tokens(&mut head);
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
            cond.write_tokens(&mut head);
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
            cond.write_tokens(&mut tail);
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
            deref_location(ctx, s.ptr, Some(roots)).write_tokens(&mut buf);
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
            buf.push(name, TokenKind::Label);
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
            buf.push(instruction_name(ctx, id), TokenKind::Variable);
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

        let program = decompile_function(&ctx, f);
        let lines = emit_tokens(&ctx, &program);

        assert!(
            lines
                .iter()
                .any(|line| line.indent == 1 && line.text() == "x = 0x1;"),
            "expected nested store to carry indent depth 1, got:\n{lines:#?}"
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

        assert!(
            lines
                .iter()
                .any(|line| line.indent == 0 && line.text() == "entry:"),
            "expected label at base indent, got:\n{lines:#?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.indent == 1 && line.text().starts_with("if (cond) goto ")),
            "expected flat block body to be indented under its label, got:\n{lines:#?}"
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
