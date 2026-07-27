//! A GUI-agnostic token stream for rendering decompiled pseudo-C.
//!
//! The emitter classifies every piece of output text ([`TokenKind`]) so a
//! front-end can syntax-highlight without re-parsing, and reports structural
//! `indent` and block provenance per line. Colours are the front-end's concern.

use qcode::value::{BlockId, InstructionId, ValueId, function::FunctionId};

/// The syntactic role of a token, for syntax highlighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    /// A C keyword (`if`, `else`, `while`, `goto`, `return`).
    Keyword,
    /// A type name in a cast (`uint32_t`).
    Type,
    /// A numeric literal.
    Number,
    /// A variable / SSA temporary / register name.
    Variable,
    /// A called function or label name (jump target).
    Label,
    /// An operator (`+`, `==`, `*`, ...).
    Operator,
    /// Punctuation (parentheses, braces, commas, semicolons).
    Punctuation,
    /// Plain/uncategorised text.
    Plain,
}

/// A single classified piece of text (no newlines).
///
/// A token optionally carries provenance back to the IR so a front-end can make
/// it interactive: `value` is the [`ValueId`] this token renders (clicking it can
/// highlight that value's uses), and `function` is the callee a call token names
/// (clicking it can navigate there).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub text: String,
    pub kind: TokenKind,
    /// The IR value this token renders, when it renders one.
    pub value: Option<ValueId>,
    /// The function this token names, for call tokens.
    pub function: Option<FunctionId>,
}

impl Token {
    pub fn new(text: impl Into<String>, kind: TokenKind) -> Self {
        Token {
            text: text.into(),
            kind,
            value: None,
            function: None,
        }
    }

    /// A token that renders IR `value`.
    pub fn with_value(text: impl Into<String>, kind: TokenKind, value: Option<ValueId>) -> Self {
        Token {
            text: text.into(),
            kind,
            value,
            function: None,
        }
    }

    /// A token that names a callee `function`.
    pub fn with_function(
        text: impl Into<String>,
        kind: TokenKind,
        function: Option<FunctionId>,
    ) -> Self {
        Token {
            text: text.into(),
            kind,
            value: None,
            function,
        }
    }
}

/// One line of decompiled output: its indentation depth, tokens, the block it
/// belongs to (for cross-pane navigation), and the set of IR instructions that
/// folded into it (so a front-end can highlight the corresponding low-level
/// code when the line is selected).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenLine {
    pub indent: usize,
    pub tokens: Vec<Token>,
    pub block: Option<BlockId>,
    pub insns: Vec<InstructionId>,
}

impl TokenLine {
    /// The concatenated plain text of the line, without indentation.
    pub fn text(&self) -> String {
        self.tokens.iter().map(|t| t.text.as_str()).collect()
    }
}

/// A small builder for assembling a line's tokens, accumulating the IR
/// instructions the line folds from as tokens are pushed.
#[derive(Default)]
pub(crate) struct LineBuf {
    pub(crate) tokens: Vec<Token>,
    pub(crate) insns: Vec<InstructionId>,
}

impl LineBuf {
    pub(crate) fn push(&mut self, text: impl Into<String>, kind: TokenKind) {
        self.tokens.push(Token::new(text, kind));
    }

    /// Pushes a token rendering IR `value`, recording it as a contributing
    /// instruction when it is an instruction result.
    pub(crate) fn push_value(
        &mut self,
        text: impl Into<String>,
        kind: TokenKind,
        value: Option<ValueId>,
    ) {
        if let Some(ValueId::Instruction(id)) = value {
            self.note_insn(id);
        }
        self.tokens.push(Token::with_value(text, kind, value));
    }

    /// Pushes a token naming callee `function`.
    pub(crate) fn push_function(
        &mut self,
        text: impl Into<String>,
        kind: TokenKind,
        function: Option<FunctionId>,
    ) {
        self.tokens.push(Token::with_function(text, kind, function));
    }

    /// Records an instruction that folded into this line but has no token of its
    /// own (e.g. the statement's own root, or a folded-away comparison).
    pub(crate) fn note_insn(&mut self, id: InstructionId) {
        self.insns.push(id);
    }

    pub(crate) fn keyword(&mut self, text: &str) {
        self.push(text, TokenKind::Keyword);
    }

    pub(crate) fn punct(&mut self, text: &str) {
        self.push(text, TokenKind::Punctuation);
    }

    pub(crate) fn space(&mut self) {
        self.push(" ", TokenKind::Plain);
    }

    pub(crate) fn into_line(mut self, indent: usize, block: Option<BlockId>) -> TokenLine {
        // Order is irrelevant to consumers; dedup so a value used twice on one
        // line is not reported twice.
        self.insns
            .sort_unstable_by_key(|id| (usize::from(id.func), usize::from(id.local)));
        self.insns.dedup();
        TokenLine {
            indent,
            tokens: self.tokens,
            block,
            insns: self.insns,
        }
    }
}
