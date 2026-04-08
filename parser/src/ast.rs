#[derive(Clone, Debug)]
pub enum Atom {
    External(String),
    Local(String),
    Int(u64),
}

#[derive(Clone, Debug)]
pub struct TypedAtom {
    pub size_bytes: Option<usize>,
    pub atom: Atom,
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
pub enum Statement {
    LocalDecl {
        name: String,
        display_name: String,
        size_bytes: usize,
    },
    Assign {
        name: String,
        expose: bool,
        expr: ExprNode,
    },
    Expr(ExprNode),
    LabelDecl {
        name: String,
    },
    Branch {
        target: String,
    },
    BranchInd {
        ptr: TypedAtom,
    },
    CBranch {
        condition: TypedAtom,
        target: String,
        fallthrough: String,
    },
    Call {
        target: String,
    },
    CallInd {
        ptr: TypedAtom,
    },
    Return {
        ptr: TypedAtom,
    },
}
