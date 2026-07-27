//! Lightweight decompiler facade for the current body-local qcode IR.
//!
//! The historical structuring implementation under this directory targets the
//! pre-context-split ID model. This facade keeps decompilation available while
//! emitting stable, provenance-tagged pseudo-C directly from the current IR.

use std::fmt::{Display, Formatter};

use qcode::context::Context;
use qcode::value::{BlockId, FunctionId, FunctionRef, InstructionId, ValueId, insn::Mnemonic};

/// The syntactic role of a rendered token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Keyword,
    Type,
    Number,
    Variable,
    Label,
    Operator,
    Punctuation,
    Plain,
}

/// One syntax-classified piece of decompiled output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub text: String,
    pub kind: TokenKind,
    pub value: Option<ValueId>,
    pub function: Option<FunctionId>,
}

impl Token {
    fn plain(text: impl Into<String>, kind: TokenKind) -> Self {
        Self {
            text: text.into(),
            kind,
            value: None,
            function: None,
        }
    }
}

/// One output line with navigation and low-level provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenLine {
    pub indent: usize,
    pub tokens: Vec<Token>,
    pub block: Option<BlockId>,
    pub insns: Vec<InstructionId>,
}

impl TokenLine {
    pub fn text(&self) -> String {
        self.tokens
            .iter()
            .map(|token| token.text.as_str())
            .collect()
    }
}

/// Decompiled pseudo-C plus its token stream.
#[derive(Debug, Clone)]
pub struct Program {
    lines: Vec<TokenLine>,
    gotos: usize,
}

impl Program {
    pub fn goto_count(&self) -> usize {
        self.gotos
    }

    pub fn token_lines(&self) -> &[TokenLine] {
        &self.lines
    }
}

impl Display for Program {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&emit_c_without_context(self))
    }
}

/// Decompile one function from the current body-local IR.
pub fn decompile_function(ctx: &Context, function_id: FunctionId) -> Result<Program, String> {
    let function = FunctionRef::from_id(ctx, function_id);
    let mut lines = Vec::new();
    let mut gotos = 0;

    lines.push(TokenLine {
        indent: 0,
        tokens: vec![
            Token::plain("function ", TokenKind::Keyword),
            Token {
                text: function.name().to_string(),
                kind: TokenKind::Label,
                value: None,
                function: Some(function_id),
            },
            Token::plain("() {", TokenKind::Punctuation),
        ],
        block: None,
        insns: Vec::new(),
    });

    for block in &function {
        let label = block
            .name()
            .map(str::to_owned)
            .unwrap_or_else(|| format!("bb_{}", usize::from(block.id.local)));
        lines.push(TokenLine {
            indent: 1,
            tokens: vec![
                Token::plain(label, TokenKind::Label),
                Token::plain(":", TokenKind::Punctuation),
            ],
            block: Some(block.id),
            insns: Vec::new(),
        });

        for instruction in &block {
            let id = instruction.id;
            let (text, kind, callee) = render_instruction(ctx, id);
            gotos += usize::from(matches!(
                instruction.mnemonic(),
                Mnemonic::Branch(_) | Mnemonic::CBranch(_)
            ));
            lines.push(TokenLine {
                indent: 2,
                tokens: vec![Token {
                    text,
                    kind,
                    value: Some(ValueId::Instruction(id)),
                    function: callee,
                }],
                block: Some(block.id),
                insns: vec![id],
            });
        }
    }

    lines.push(TokenLine {
        indent: 0,
        tokens: vec![Token::plain("}", TokenKind::Punctuation)],
        block: None,
        insns: Vec::new(),
    });

    Ok(Program { lines, gotos })
}

fn render_instruction(ctx: &Context, id: InstructionId) -> (String, TokenKind, Option<FunctionId>) {
    let instruction = qcode::value::InstructionRef::from_id(ctx, id);
    let callee = match instruction.mnemonic() {
        Mnemonic::Call(call) => call.target.real(),
        _ => None,
    };
    let kind = match instruction.mnemonic() {
        Mnemonic::Branch(_) | Mnemonic::CBranch(_) | Mnemonic::Return(_) => TokenKind::Keyword,
        Mnemonic::Call(_) | Mnemonic::CallInd(_) => TokenKind::Label,
        _ => TokenKind::Plain,
    };
    (instruction.as_statement().to_string(), kind, callee)
}

/// Render pseudo-C text.
pub fn emit_c(_ctx: &Context, program: &Program) -> String {
    emit_c_without_context(program)
}

fn emit_c_without_context(program: &Program) -> String {
    let mut output = String::new();
    for line in &program.lines {
        output.push_str(&"    ".repeat(line.indent));
        output.push_str(&line.text());
        output.push('\n');
    }
    output
}

/// Return the frontend-neutral token stream.
pub fn emit_tokens(_ctx: &Context, program: &Program) -> Vec<TokenLine> {
    program.lines.clone()
}
