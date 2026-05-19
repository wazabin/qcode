use std::{fmt::Display, ops::Range};

use crate::value::{BlockId, ValueId};

#[derive(Debug, PartialEq, Eq)]
pub enum ErrorTy {
    RangeOutOfBounds {
        range: Range<usize>,
        available: usize,
    },

    ArgumentCountMismatch {
        expected: usize,
        actual: usize,
    },

    /// Could not determine the size of an expression
    UnknownSize,

    /// Unknown name in an expression
    UnknownIdentifier(Box<str>, &'static str),

    SizeMismatch {
        expected: usize,
        actual: usize,
    },

    /// Attempted to set a name on a literal value
    TriedToNameLiteral,

    /// Attempted to set a name that already exists in the current scope
    NameAlreadyExists(Box<str>),

    UnknownMacro(Box<str>),

    UnknownAddress(u64),

    /// A macro definition contains multiple exports
    MultipleExports,

    /// The export statement is not the last statement in a macro definition
    ExportNotLast,

    /// Attempted to use a function as an expression, but it is a statement
    FunctionStatement,

    /// A macro argument is not const
    NonConstArgument,

    MissingArgument(Box<str>),

    /// A name was registered but a value with that name already exists
    DuplicateName(String),

    /// Two values were assigned to the same machine address
    DuplicateAddress(u64, ValueId),

    /// A function and it's root block have conflicting addresses
    FunctionRootAddressMismatch {
        fn_addr: u64,
        block_addr: u64,
    },

    FunctionRootMismatch {
        expected: BlockId,
        actual: BlockId,
    },
}

#[derive(Debug)]
pub struct Error {
    pub ty: ErrorTy,
    /// Byte range `(start, end)` into the prepared source, if available.
    pub span: Option<(usize, usize)>,
}

pub type Result<T> = std::result::Result<T, Error>;

impl std::error::Error for Error {}

impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        self.ty == other.ty
    }
}

impl Eq for Error {}

impl Error {
    pub fn new(ty: ErrorTy, span: (usize, usize)) -> Self {
        Self {
            ty,
            span: Some(span),
        }
    }

    pub fn with_span(mut self, span: (usize, usize)) -> Self {
        self.span = Some(span);
        self
    }

    pub fn spanless(ty: ErrorTy) -> Self {
        Self { ty, span: None }
    }

    pub fn range_out_of_bounds(range: Range<usize>, span: (usize, usize)) -> Self {
        Self::new(
            ErrorTy::RangeOutOfBounds {
                range,
                available: 0,
            },
            span,
        )
    }

    pub fn argument_count_mismatch(expected: usize, actual: usize, span: (usize, usize)) -> Self {
        Self::new(ErrorTy::ArgumentCountMismatch { expected, actual }, span)
    }

    pub fn unknown_size(span: (usize, usize)) -> Self {
        Self::new(ErrorTy::UnknownSize, span)
    }

    pub fn unknown_identifier(name: &str, ty: &'static str, span: (usize, usize)) -> Self {
        Self::new(ErrorTy::UnknownIdentifier(name.into(), ty), span)
    }

    pub fn size_mismatch(expected: usize, actual: usize, span: (usize, usize)) -> Self {
        Self::new(ErrorTy::SizeMismatch { expected, actual }, span)
    }

    pub fn tried_to_name_literal(span: (usize, usize)) -> Self {
        Self::new(ErrorTy::TriedToNameLiteral, span)
    }

    pub fn name_already_exists(name: &str, span: (usize, usize)) -> Self {
        Self::new(ErrorTy::NameAlreadyExists(name.into()), span)
    }

    pub fn unknown_macro(name: &str, span: (usize, usize)) -> Self {
        Self::new(ErrorTy::UnknownMacro(name.into()), span)
    }

    pub fn multiple_exports(span: (usize, usize)) -> Self {
        Self::new(ErrorTy::MultipleExports, span)
    }

    pub fn export_not_last(span: (usize, usize)) -> Self {
        Self::new(ErrorTy::ExportNotLast, span)
    }

    pub fn function_is_a_statement(span: (usize, usize)) -> Self {
        Self::new(ErrorTy::FunctionStatement, span)
    }

    pub fn missing_argument(name: &str, span: (usize, usize)) -> Self {
        Self::new(ErrorTy::MissingArgument(name.into()), span)
    }

    pub fn non_const_arg(span: (usize, usize)) -> Self {
        Self::new(ErrorTy::NonConstArgument, span)
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match &self.ty {
            ErrorTy::RangeOutOfBounds { range, available } => {
                format!("Range {range:?} is out of bounds for available size {available}")
            }

            ErrorTy::ArgumentCountMismatch { expected, actual } => {
                format!("Expected {expected} arguments but got {actual}")
            }

            ErrorTy::UnknownSize => "Could not determine the size of this expression".to_string(),

            ErrorTy::UnknownIdentifier(name, ty) => format!("Unknown {ty}: {name}"),

            ErrorTy::SizeMismatch { expected, actual } => {
                format!("Expected size {expected} but got size {actual}")
            }

            ErrorTy::TriedToNameLiteral => "Attempted to set a name on a literal value".to_string(),

            ErrorTy::NameAlreadyExists(name) => {
                format!("A name '{name}' already exists in the current scope")
            }

            ErrorTy::UnknownMacro(name) => format!("Unknown macro: {name}"),

            ErrorTy::MultipleExports => "A macro definition contains multiple exports".to_string(),

            ErrorTy::ExportNotLast => {
                "The export statement is not the last statement in a macro definition".to_string()
            }

            ErrorTy::FunctionStatement => {
                "Attempted to use a function as an expression, but it is a statement".to_string()
            }

            ErrorTy::MissingArgument(name) => format!("Missing argument: {name}"),

            ErrorTy::NonConstArgument => "Argument must be a compile-time constant".to_string(),

            ErrorTy::DuplicateName(name) => format!("duplicate name: {name}"),

            ErrorTy::DuplicateAddress(addr, value) => {
                format!("address {addr:#x} is already mapped to a value: {value:?}")
            }

            ErrorTy::UnknownAddress(addr) => {
                format!("address {addr:#x} is not mapped to any value")
            }

            ErrorTy::FunctionRootAddressMismatch {
                fn_addr,
                block_addr,
            } => {
                format!(
                    "Function root address mismatch: function address {fn_addr:#x}, block address {block_addr:#x}"
                )
            }

            ErrorTy::FunctionRootMismatch { expected, actual } => {
                format!("Function root mismatch: expected root block {expected:?}, got {actual:?}")
            }
        };

        if let Some((start, end)) = self.span {
            write!(f, "{message} (bytes {start}..{end})")
        } else {
            write!(f, "{message}")
        }
    }
}
