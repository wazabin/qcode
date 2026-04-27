#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourcePosition {
    pub offset: usize,
    pub line: usize,
    pub column: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceSpan {
    pub start: SourcePosition,
    pub end: SourcePosition,
}

impl SourceSpan {
    pub fn contains_offset(&self, offset: usize) -> bool {
        self.start.offset <= offset && offset < self.end.offset
    }
}

#[derive(Clone, Debug)]
pub enum Atom {
    /// `{name}` — captures a Rust variable from the surrounding scope.
    External(String),
    /// `%name` — references an SSA instruction result.
    Ssa(String),
    /// `@name` — references a block parameter.
    BlockParam(String),
    /// bare `name` — references a varnode (valid only in pointer positions).
    Varnode(String),
    /// `&name` — takes the address of a varnode.
    AddressOf(String),
    Int(u64),
}

#[derive(Clone, Debug)]
pub struct TypedAtom {
    pub size_bytes: Option<usize>,
    pub atom: Atom,
    pub span: SourceSpan,
}

#[derive(Clone, Debug)]
pub enum ExprNode {
    Atom(TypedAtom),
    Unop {
        op: String,
        src: TypedAtom,
    },
    Binary {
        lhs: TypedAtom,
        op: String,
        rhs: TypedAtom,
    },
    Cast {
        op: CastOp,
        size_bytes: usize,
        src: TypedAtom,
    },
    Load {
        size_bytes: usize,
        ptr: TypedAtom,
    },
    Store {
        ptr: TypedAtom,
        src: TypedAtom,
    },
    FuncCall {
        op: String,
        args: Vec<TypedAtom>,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum CastOp {
    Zext,
    Sext,
    IntToFloat,
    FloatToFloat,
    Trunc,
}

#[derive(Clone, Debug)]
pub struct BlockParamDecl {
    pub name: String,
    pub size_bytes: Option<usize>,
}

/// A branch target or label declaration — either a named label or a block address.
#[derive(Clone, Debug)]
pub enum Label {
    /// A named label such as `<entry>` or `<done @v1 @v2>`. Generates a `BlockId` binding.
    Named {
        name: String,
        /// Block parameters declared on this label (e.g. `@v1`, `@v2:i64`).
        /// Non-empty only when this `Label` appears inside a `LabelDecl`.
        params: Vec<BlockParamDecl>,
        span: SourceSpan,
    },
    /// A numeric address such as `<0x1001>`. Sets the block's address; no binding generated.
    Address { value: u64, span: SourceSpan },
}

impl Label {
    pub fn span(&self) -> &SourceSpan {
        match self {
            Self::Named { span, .. } | Self::Address { span, .. } => span,
        }
    }

    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Named { name, .. } => Some(name),
            Self::Address { .. } => None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Statement {
    LocalDecl {
        name: String,
        name_span: SourceSpan,
        display_name: String,
        size_bytes: usize,
        span: SourceSpan,
    },
    Assign {
        name: String,
        name_span: SourceSpan,
        expr: ExprNode,
        span: SourceSpan,
    },
    Expr(ExprNode),
    LabelDecl {
        label: Label,
        span: SourceSpan,
    },
    Branch {
        target: Label,
        /// Per-parameter arguments: `(param_name, value)` in declaration order.
        /// Non-empty when the branch was written as `goto <block @v1=e1 @v2=e2>`.
        args: Vec<(String, TypedAtom)>,
        span: SourceSpan,
    },
    BranchInd {
        ptr: TypedAtom,
        span: SourceSpan,
    },
    CBranch {
        condition: TypedAtom,
        target: Label,
        target_args: Vec<(String, TypedAtom)>,
        fallthrough: Label,
        fallthrough_args: Vec<(String, TypedAtom)>,
        span: SourceSpan,
    },
    Call {
        target: Label,
        span: SourceSpan,
    },
    CallInd {
        ptr: TypedAtom,
        span: SourceSpan,
    },
    Return {
        ptr: TypedAtom,
        span: SourceSpan,
    },
}

/// A function declaration (`fn name: <entry> stmts...`).
#[derive(Clone, Debug)]
pub struct FnDecl {
    pub name: String,
    pub name_span: SourceSpan,
    pub span: SourceSpan,
    pub statements: Vec<Statement>,
}

/// Top-level program representation.
#[derive(Clone, Debug)]
pub enum Program {
    Statements(Vec<Statement>),
    Functions {
        varnodes: Vec<Statement>,
        fns: Vec<FnDecl>,
    },
}
