//! A GUI-agnostic token stream for rendering decompiled pseudo-C.
//!
//! The emitter classifies every piece of output text ([`TokenKind`]) so a
//! front-end can syntax-highlight without re-parsing, and reports structural
//! `indent` and block provenance per line. Colours are the front-end's concern.

use qcode::value::BlockId;

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub text: String,
    pub kind: TokenKind,
}

impl Token {
    pub fn new(text: impl Into<String>, kind: TokenKind) -> Self {
        Token {
            text: text.into(),
            kind,
        }
    }
}

/// One line of decompiled output: its indentation depth, tokens, and the block
/// it belongs to (for cross-pane navigation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenLine {
    pub indent: usize,
    pub tokens: Vec<Token>,
    pub block: Option<BlockId>,
}

impl TokenLine {
    /// The concatenated plain text of the line, without indentation.
    pub fn text(&self) -> String {
        self.tokens.iter().map(|t| t.text.as_str()).collect()
    }
}

/// A small builder for assembling a line's tokens.
#[derive(Default)]
pub(crate) struct LineBuf {
    pub(crate) tokens: Vec<Token>,
}

impl LineBuf {
    pub(crate) fn push(&mut self, text: impl Into<String>, kind: TokenKind) {
        self.tokens.push(Token::new(text, kind));
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

    pub(crate) fn into_line(self, indent: usize, block: Option<BlockId>) -> TokenLine {
        TokenLine {
            indent,
            tokens: self.tokens,
            block,
        }
    }
}
