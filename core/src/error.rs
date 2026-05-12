use common::raw_parsing::Rule;
use pest::Span;
use std::{fmt::Display, ops::Range};

use crate::value::{BlockId, ValueId};

#[derive(Debug, PartialEq, Eq)]
pub enum ErrorTy<'str> {
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
    UnknownIdentifier(&'str str, &'static str),

    SizeMismatch {
        expected: usize,
        actual: usize,
    },

    /// Attempted to set a name on a literal value
    TriedToNameLiteral,

    /// Attempted to set a name that already exists in the current scope
    NameAlreadyExists(&'str str),

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

    MissingArgument(&'str str),

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
pub struct Error<'str> {
    pub ty: ErrorTy<'str>,
    pub span: Option<Span<'str>>,
}

pub type Result<'str, T> = std::result::Result<T, Error<'str>>;

impl<'str> std::error::Error for Error<'str> {}

impl<'str> PartialEq for Error<'str> {
    fn eq(&self, other: &Self) -> bool {
        self.ty == other.ty
    }
}

impl<'str> Eq for Error<'str> {}

impl<'str> Error<'str> {
    pub fn new(ty: ErrorTy<'str>, span: Span<'str>) -> Self {
        Self {
            ty,
            span: Some(span),
        }
    }

    pub fn with_span(mut self, span: Span<'str>) -> Self {
        self.span = Some(span);
        self
    }

    pub fn spanless(ty: ErrorTy<'str>) -> Self {
        Self { ty, span: None }
    }

    pub fn range_out_of_bounds(range: Range<usize>, span: Span<'str>) -> Self {
        Self::new(
            ErrorTy::RangeOutOfBounds {
                range,
                available: 0,
            },
            span,
        )
    }

    pub fn argument_count_mismatch(expected: usize, actual: usize, span: Span<'str>) -> Self {
        Self::new(ErrorTy::ArgumentCountMismatch { expected, actual }, span)
    }

    pub fn unknown_size(span: Span<'str>) -> Self {
        Self::new(ErrorTy::UnknownSize, span)
    }

    pub fn unknown_identifier(name: &'str str, ty: &'static str, span: Span<'str>) -> Self {
        Self::new(ErrorTy::UnknownIdentifier(name, ty), span)
    }

    pub fn size_mismatch(expected: usize, actual: usize, span: Span<'str>) -> Self {
        Self::new(ErrorTy::SizeMismatch { expected, actual }, span)
    }

    pub fn tried_to_name_literal(span: Span<'str>) -> Self {
        Self::new(ErrorTy::TriedToNameLiteral, span)
    }

    pub fn name_already_exists(name: &'str str, span: Span<'str>) -> Self {
        Self::new(ErrorTy::NameAlreadyExists(name), span)
    }

    pub fn unknown_macro(name: &'str str, span: Span<'str>) -> Self {
        Self::new(ErrorTy::UnknownMacro(name.into()), span)
    }

    pub fn multiple_exports(span: Span<'str>) -> Self {
        Self::new(ErrorTy::MultipleExports, span)
    }

    pub fn export_not_last(span: Span<'str>) -> Self {
        Self::new(ErrorTy::ExportNotLast, span)
    }

    pub fn function_is_a_statement(span: Span<'str>) -> Self {
        Self::new(ErrorTy::FunctionStatement, span)
    }

    pub fn missing_argument(name: &'str str, span: Span<'str>) -> Self {
        Self::new(ErrorTy::MissingArgument(name), span)
    }

    pub fn non_const_arg(span: Span<'str>) -> Self {
        Self::new(ErrorTy::NonConstArgument, span)
    }
}

impl Display for Error<'_> {
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

        match self.span {
            Some(span) => write!(
                f,
                "{}",
                pest::error::Error::<Rule>::new_from_span(
                    pest::error::ErrorVariant::CustomError { message },
                    span
                )
            ),
            None => write!(f, "{message}"),
        }
    }
}
