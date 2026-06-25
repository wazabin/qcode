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
pub struct TupleField {
    pub name: Option<String>,
    pub value: TypedAtom,
}

#[derive(Clone, Debug)]
pub enum ExtractField {
    Name(String),
    Index(u64),
}

/// The field selector of a `gep(...)` — by field name or by raw byte offset.
#[derive(Clone, Debug)]
pub enum GepField {
    Name(String),
    Offset(u64),
}

/// The declared type of a struct field: either a scalar of `n` bytes, or a
/// pointer to a named (nominal) struct, written `Foo*`.
#[derive(Clone, Debug)]
pub enum StructFieldType {
    Int(usize),
    StructPtr(String),
}

/// One field of a `type Foo { ... }` declaration. A field named `_` is padding:
/// it advances the running offset by its byte size without naming a slot.
#[derive(Clone, Debug)]
pub struct StructFieldDecl {
    pub name: String,
    pub ty: StructFieldType,
}

impl StructFieldDecl {
    pub fn is_padding(&self) -> bool {
        self.name == "_"
    }
}

/// A nominal struct definition: `type Foo { a: 4, _: 5, b: 2 }`. Field offsets
/// are the running byte sum (padding included); the struct `size` is the total.
#[derive(Clone, Debug)]
pub struct StructDecl {
    pub name: String,
    pub fields: Vec<StructFieldDecl>,
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
    /// A pure intrinsic call, e.g. `$rol(%x, %k)`. `name` excludes the `$`.
    Intrinsic {
        name: String,
        args: Vec<TypedAtom>,
    },
    /// `body <$> src` / `(body c0 c1) <$> src` — an element-wise `map` over the
    /// array `src`. `body` names a function symbol declared in the same program;
    /// `captures` are the loop-invariant operands the body closes over.
    Map {
        body: String,
        src: TypedAtom,
        captures: Vec<TypedAtom>,
    },
    /// `pack(a=x, b=y)` — build an aggregate value from its named fields.
    Tuple {
        fields: Vec<TupleField>,
    },
    /// `extract(agg.field)` — project a field out of an aggregate value.
    Extract {
        agg: TypedAtom,
        field: ExtractField,
    },
    /// `gep(base.field)` — compute the address of a struct field (typed, named
    /// pointer arithmetic; no memory access).
    Gep {
        base: TypedAtom,
        field: GepField,
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
        /// A `Foo*` struct-pointer type declared on the assignment, if any. When
        /// present, the result value is retyped to that struct pointer (used to
        /// seed struct typing in tests). A plain `iN`/`fN` declared type is not
        /// recorded here — it only drives size coercion of the rhs.
        decl_struct_ptr: Option<String>,
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
    Assert {
        condition: TypedAtom,
        span: SourceSpan,
    },
    /// A comment attached to this statement, written as `# text` on the preceding line.
    Commented {
        comment: String,
        inner: Box<Statement>,
    },
}

impl Statement {
    /// Strips any wrapping `Commented` variant and returns the inner statement.
    pub fn inner(&self) -> &Statement {
        match self {
            Self::Commented { inner, .. } => inner.inner(),
            other => other,
        }
    }
}

/// A function declaration (`fn name: <entry> stmts...`).
#[derive(Clone, Debug)]
pub struct FnDecl {
    pub name: String,
    pub name_span: SourceSpan,
    pub span: SourceSpan,
    pub statements: Vec<Statement>,
}

/// Top-level program representation. Any leading `type` declarations are
/// collected into `structs`; `kind` is the statement or function body.
#[derive(Clone, Debug)]
pub struct Program {
    pub structs: Vec<StructDecl>,
    pub kind: ProgramKind,
}

#[derive(Clone, Debug)]
pub enum ProgramKind {
    Statements(Vec<Statement>),
    Functions {
        varnodes: Vec<Statement>,
        fns: Vec<FnDecl>,
    },
}
