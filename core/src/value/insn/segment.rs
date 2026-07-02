//! Token-segment rendering of instructions for rich (colored, clickable) display.
//!
//! [`instruction_segments`] produces the *same bytes* as an instruction's
//! [`Display`](std::fmt::Display) (`InstructionStatement`) — concatenating every
//! returned [`Token`]'s text reproduces the canonical textual IR exactly — while
//! additionally tagging each run with a semantic [`TokenKind`] (for coloring) and
//! an optional [`Link`] (for click-through to the referenced value, function, or
//! block). The textual format stays the single source of truth: a test asserts
//! `concat(segments) == format!("{insn}")` for every mnemonic, so the two can
//! never drift.

use crate::{
    context::Context,
    space::Space,
    value::{
        BasicBlock, BlockParam, Function, ValueId,
        block::BlockId,
        bytes::BytesRef,
        function::FunctionId,
        insn::{InstructionRef, Mnemonic},
        literal::LiteralRef,
        varnode::Varnode,
    },
};

/// What a token *is*, semantically — drives syntax coloring in a viewer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    /// A type name (`i32`, `f64`), the access width of a load/store, or a cast
    /// target width.
    Type,
    /// An instruction-result reference (`%name` / `%tmp…`).
    Variable,
    /// A block parameter (`@name`), or a parameter/argument name on the left of
    /// `=` in a branch/call argument.
    BlockParam,
    /// A varnode / register reference.
    Varnode,
    /// A literal constant (`0x2`, `&<blk>`, `&"str"`) or a bare numeric offset.
    Literal,
    /// A byte-string literal (`b"…"`).
    Bytes,
    /// A reserved word or mnemonic (`load`, `goto`, `call fn`, `zext`, …).
    Keyword,
    /// An operator (`+`, ` = `, ` <- `, ` <$> `, `.`).
    Operator,
    /// Structural punctuation (`(`, `)`, `,`, `:`, `[`, `]`, `;`, spaces).
    Punctuation,
    /// A basic-block label (`<name>`).
    Label,
    /// An aggregate field name (`lhs`, `.val`).
    Field,
    /// A function reference (call/apply/map/scan target, or function used as a
    /// value operand).
    Function,
    /// A memory space name (`ram`, `register`, …).
    Space,
}

/// A click-through target carried by a token, when it names something navigable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Link {
    Value(ValueId),
    Function(FunctionId),
    Block(BlockId),
}

/// One contiguous run of rendered text with its semantic kind and optional link.
///
/// Concatenating the `text` of every token of an instruction yields exactly the
/// instruction's `Display` output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub text: String,
    pub kind: TokenKind,
    pub link: Option<Link>,
}

impl Token {
    fn new(text: impl Into<String>, kind: TokenKind, link: Option<Link>) -> Self {
        Token {
            text: text.into(),
            kind,
            link,
        }
    }
}

/// Accumulator with terse push helpers, kept private to this module.
struct Seg<'a, 'str> {
    ctx: &'a Context<'str>,
    out: Vec<Token>,
}

impl<'a, 'str> Seg<'a, 'str> {
    fn push(&mut self, text: impl Into<String>, kind: TokenKind, link: Option<Link>) {
        self.out.push(Token::new(text, kind, link));
    }

    fn kw(&mut self, text: &str) {
        self.push(text, TokenKind::Keyword, None);
    }

    fn op(&mut self, text: impl Into<String>) {
        self.push(text, TokenKind::Operator, None);
    }

    fn punct(&mut self, text: &str) {
        self.push(text, TokenKind::Punctuation, None);
    }

    /// The `<ty> ` prefix shared by every typed operand. Mirrors
    /// `write!(f, "{} ", type_name)` in `ValueRef`'s `Display`.
    fn ty(&mut self, type_id: crate::types::TypeId) {
        self.push(
            format!("{} ", self.ctx.types.type_name(type_id)),
            TokenKind::Type,
            None,
        );
    }

    /// A value operand, rendered exactly as `ValueRef`'s `Display`: `<ty> <atom>`
    /// for scalars (instruction, block param, literal, bytes, varnode), and bare
    /// for functions/blocks.
    fn value(&mut self, id: ValueId) {
        let link = Some(Link::Value(id));
        match id {
            ValueId::Instruction(iid) => {
                let r = InstructionRef::new(self.ctx, iid);
                self.ty(r.type_id());
                self.push(instruction_atom(self.ctx, iid), TokenKind::Variable, link);
            }
            ValueId::BlockParam(pid) => {
                let r = BlockParam::from_id(self.ctx, pid);
                self.ty(r.type_id());
                self.push(block_param_atom(self.ctx, pid), TokenKind::BlockParam, link);
            }
            ValueId::Literal(lid) => {
                let r = LiteralRef::new(self.ctx, lid);
                self.ty(r.type_id());
                self.push(r.to_string(), TokenKind::Literal, link);
            }
            ValueId::Bytes(bid) => {
                let r = BytesRef::new(self.ctx, bid);
                self.ty(r.type_id());
                self.push(r.to_string(), TokenKind::Bytes, link);
            }
            ValueId::Varnode(vid) => {
                let r = Varnode::from_id(self.ctx, vid);
                self.push(format!("i{} ", r.size() * 8), TokenKind::Type, None);
                self.push(r.to_string(), TokenKind::Varnode, link);
            }
            ValueId::Function(fid) => {
                let name = Function::from_id(self.ctx, fid).name().to_string();
                self.push(
                    format!("<{name}>"),
                    TokenKind::Function,
                    Some(Link::Function(fid)),
                );
            }
            ValueId::BasicBlock(bid) => {
                // A block used as a value renders via the block's own `Display`
                // (never `ValueRef`'s, which routes back here — that would recurse).
                self.push(
                    BasicBlock::from_id(self.ctx, bid).to_string(),
                    TokenKind::Label,
                    Some(Link::Block(bid)),
                );
            }
        }
    }

    /// A value operand printed *bare* (no type prefix) for instruction results,
    /// matching `fmt_bare_value` used by `extract`/`gep`. Non-instruction values
    /// fall back to the regular typed rendering.
    fn bare_value(&mut self, id: ValueId) {
        match id {
            ValueId::Instruction(iid) => {
                self.push(
                    instruction_atom(self.ctx, iid),
                    TokenKind::Variable,
                    Some(Link::Value(id)),
                );
            }
            other => self.value(other),
        }
    }

    /// A direct branch/cbranch target: `<name @p=arg …>`. Mirrors
    /// `fmt_branch_target`.
    fn branch_target(&mut self, target: BlockId, args: &[ValueId]) {
        let block = BasicBlock::from_id(self.ctx, target);
        let name = block.name().unwrap_or("unnamed");
        self.push(
            format!("<{name}"),
            TokenKind::Label,
            Some(Link::Block(target)),
        );

        let params = block.params().collect::<Vec<_>>();
        for (i, &arg) in args.iter().enumerate() {
            self.punct(" ");
            match params.get(i) {
                Some(param) => self.push(param.to_string(), TokenKind::BlockParam, None),
                None => self.push(format!("@arg{i}"), TokenKind::BlockParam, None),
            }
            self.op("=");
            self.value(arg);
        }

        self.push(">", TokenKind::Label, None);
    }
}

/// The bare atom for an instruction result: `%name` or `%tmp<id>`.
fn instruction_atom(ctx: &Context<'_>, id: crate::value::InstructionId) -> String {
    match ctx.values.instructions[id].name.as_deref() {
        Some(name) => format!("%{name}"),
        None => format!("%tmp{:x}", usize::from(id)),
    }
}

/// The bare atom for a block parameter: `@name` or `@param<id>`.
fn block_param_atom(ctx: &Context<'_>, id: crate::value::BlockParamId) -> String {
    let r = BlockParam::from_id(ctx, id);
    match r.name() {
        Some(name) => format!("@{name}"),
        None => format!("@param{:x}", usize::from(id)),
    }
}

/// Render an instruction as colored, linkable tokens. Concatenating the tokens'
/// text equals the instruction's `Display` (`as_statement()`) output.
pub fn instruction_segments(insn: &InstructionRef<'_, '_>) -> Vec<Token> {
    let ctx = insn.ctx;
    let mut seg = Seg {
        ctx,
        out: Vec::new(),
    };

    // LHS: `<ty> %name = ` (mirrors `InstructionStatement` + `InstructionRef`'s
    // inherent `fmt`). Terminators and other size-0 instructions have no LHS.
    if insn.size() != 0 {
        seg.ty(insn.type_id());
        seg.push(
            instruction_atom(ctx, insn.id),
            TokenKind::Variable,
            Some(Link::Value(ValueId::Instruction(insn.id))),
        );
        seg.op(" = ");
    }

    match insn.mnemonic() {
        Mnemonic::Tuple(t) => tuple_with_type(&mut seg, t, insn.type_id()),
        m => mnemonic_segments(&mut seg, m),
    }

    seg.out
}

fn tuple_with_type(seg: &mut Seg, t: &crate::value::insn::Tuple, type_id: crate::types::TypeId) {
    seg.kw("pack");
    seg.punct("(");
    for (i, &field) in t.fields.iter().enumerate() {
        if i > 0 {
            seg.punct(", ");
        }
        let name = seg
            .ctx
            .types
            .field_name(type_id, i)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("field{}", i + 1));
        seg.push(name, TokenKind::Field, None);
        seg.op("=");
        seg.value(field);
    }
    seg.punct(");");
}

fn mnemonic_segments(seg: &mut Seg, m: &Mnemonic) {
    use crate::value::insn::Unop;
    match m {
        Mnemonic::Load(l) => {
            seg.kw("load");
            seg.punct("(");
            seg.push(space_name(seg.ctx, l.space), TokenKind::Space, None);
            seg.punct(":");
            seg.push(l.size.to_string(), TokenKind::Type, None);
            seg.punct(", ");
            seg.value(l.ptr);
            seg.punct(");");
        }
        Mnemonic::Store(s) => {
            seg.kw("store");
            seg.punct("(");
            seg.push(space_name(seg.ctx, s.space), TokenKind::Space, None);
            seg.punct(":");
            seg.push(s.size.to_string(), TokenKind::Type, None);
            seg.punct(", ");
            seg.value(s.ptr);
            seg.op(" <- ");
            seg.value(s.src);
            seg.punct(");");
        }
        Mnemonic::Branch(b) => {
            seg.kw("goto ");
            seg.branch_target(b.target, &b.args);
            seg.punct(";");
        }
        Mnemonic::BranchInd(b) => {
            seg.kw("goto ");
            seg.punct("[");
            seg.value(b.ptr);
            seg.punct("];");
        }
        Mnemonic::CBranch(cb) => {
            seg.kw("if ");
            seg.value(cb.condition);
            seg.kw(" goto ");
            seg.branch_target(cb.success_block, &cb.success_args);
            seg.kw(" else goto ");
            seg.branch_target(cb.failure_block, &cb.failure_args);
            seg.punct(";");
        }
        Mnemonic::Apply(a) => {
            seg.kw("apply ");
            seg.push(
                Function::from_id(seg.ctx, a.target).name().to_string(),
                TokenKind::Function,
                Some(Link::Function(a.target)),
            );
            seg.punct("(");
            for (i, &arg) in a.args.iter().enumerate() {
                if i > 0 {
                    seg.punct(", ");
                }
                seg.value(arg);
            }
            seg.punct(");");
        }
        Mnemonic::Call(c) => {
            seg.kw("call fn ");
            seg.push(
                Function::from_id(seg.ctx, c.target).name().to_string(),
                TokenKind::Function,
                Some(Link::Function(c.target)),
            );
            seg.punct("(");
            for (i, &arg) in c.args.iter().enumerate() {
                if i > 0 {
                    seg.punct(", ");
                }
                seg.push(
                    call_arg_name(seg.ctx, c.target, i),
                    TokenKind::BlockParam,
                    None,
                );
                seg.value(arg);
            }
            seg.punct(");");
        }
        Mnemonic::CallInd(c) => {
            seg.kw("call ");
            seg.punct("[");
            seg.value(c.ptr);
            seg.punct("]");
            if !c.args.is_empty() {
                seg.punct("(");
                for (i, &arg) in c.args.iter().enumerate() {
                    if i > 0 {
                        seg.punct(", ");
                    }
                    seg.value(arg);
                }
                seg.punct(")");
            }
            seg.punct(";");
        }
        Mnemonic::Return(r) => match r.value {
            Some(value) => {
                seg.kw("return ");
                seg.value(value);
                seg.kw(" at ");
                seg.value(r.ptr);
                seg.punct(";");
            }
            None => {
                seg.kw("return at ");
                seg.value(r.ptr);
                seg.punct(";");
            }
        },
        Mnemonic::ReturnValue(r) => {
            seg.kw("return ");
            seg.value(r.value);
            seg.punct(";");
        }
        Mnemonic::Unop(u) => match u.op {
            Unop::IntNegate | Unop::IntNot | Unop::BoolNot | Unop::FloatNegate => {
                seg.op(format!("{} ", u.op));
                seg.value(u.src);
                seg.punct(";");
            }
            _ => {
                seg.kw(&u.op.to_string());
                seg.punct("(");
                seg.value(u.src);
                seg.punct(");");
            }
        },
        Mnemonic::Binop(b) => {
            seg.value(b.lhs);
            seg.op(format!(" {} ", b.op));
            seg.value(b.rhs);
            seg.punct(";");
        }
        Mnemonic::Zext(z) => cast(seg, "zext", 'i', z.size, z.src),
        Mnemonic::Sext(s) => cast(seg, "sext", 'i', s.size, s.src),
        Mnemonic::IntToFloat(c) => cast(seg, "int2float", 'f', c.size, c.src),
        Mnemonic::FloatToFloat(c) => cast(seg, "float2float", 'f', c.size, c.src),
        Mnemonic::FloatToInt(c) => cast(seg, "trunc", 'i', c.size, c.src),
        Mnemonic::Range(r) => {
            seg.value(r.src);
            seg.punct("[");
            seg.push(r.start.to_string(), TokenKind::Literal, None);
            seg.punct(":");
            seg.push((r.start + r.size).to_string(), TokenKind::Literal, None);
            seg.punct("];");
        }
        Mnemonic::IsFloatNaN(o) => unary_call(seg, "nan", o.src),
        Mnemonic::LzCount(o) => unary_call(seg, "lzcount", o.src),
        Mnemonic::PopCount(o) => unary_call(seg, "popcount", o.src),
        Mnemonic::Carry(o) => binary_call(seg, "carry", o.lhs, o.rhs),
        Mnemonic::SCarry(o) => binary_call(seg, "scarry", o.lhs, o.rhs),
        Mnemonic::SBorrow(o) => binary_call(seg, "sborrow", o.lhs, o.rhs),
        Mnemonic::Assert(a) => {
            seg.kw("assert ");
            seg.value(a.condition);
            seg.punct(";");
        }
        // `Tuple` is normally routed through `tuple_with_type` (it always has a
        // result type); this bare form mirrors `Tuple`'s own `MnemonicKind::fmt`.
        Mnemonic::Tuple(t) => {
            seg.kw("pack");
            seg.punct("(");
            for (i, &field) in t.fields.iter().enumerate() {
                if i > 0 {
                    seg.punct(", ");
                }
                seg.push(format!("field{}", i + 1), TokenKind::Field, None);
                seg.op("=");
                seg.value(field);
            }
            seg.punct(");");
        }
        Mnemonic::Extract(e) => {
            seg.kw("extract");
            seg.punct("(");
            seg.bare_value(e.agg);
            let name = e
                .field_name(seg.ctx)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("field{}", e.index + 1));
            seg.push(format!(".{name}"), TokenKind::Field, None);
            seg.punct(");");
        }
        Mnemonic::Gep(g) => {
            seg.kw("gep");
            seg.punct("(");
            seg.bare_value(g.base);
            match g.field_name(seg.ctx) {
                Some(name) => seg.push(format!(".{name}"), TokenKind::Field, None),
                None => {
                    seg.op(" + ");
                    seg.push(format!("{:#x}", g.offset), TokenKind::Literal, None);
                }
            }
            seg.punct(");");
        }
        Mnemonic::Map(map) => {
            let body = Function::from_id(seg.ctx, map.body).name().to_string();
            if map.captures.is_empty() {
                seg.push(body, TokenKind::Function, Some(Link::Function(map.body)));
                seg.op(" <$> ");
                seg.value(map.src);
                seg.punct(";");
            } else {
                seg.punct("(");
                seg.push(body, TokenKind::Function, Some(Link::Function(map.body)));
                for &c in &map.captures {
                    seg.punct(" ");
                    seg.value(c);
                }
                seg.op(") <$> ");
                seg.value(map.src);
                seg.punct(";");
            }
        }
        Mnemonic::Scan(scan) => {
            let body = Function::from_id(seg.ctx, scan.body).name().to_string();
            if scan.captures.is_empty() {
                seg.kw("scanl ");
                seg.push(
                    format!("@{body}"),
                    TokenKind::Function,
                    Some(Link::Function(scan.body)),
                );
                seg.punct(" ");
                seg.value(scan.init);
                seg.punct(" ");
                seg.value(scan.src);
                seg.punct(";");
            } else {
                seg.kw("scanl ");
                seg.punct("(");
                seg.push(
                    format!("@{body}"),
                    TokenKind::Function,
                    Some(Link::Function(scan.body)),
                );
                for &c in &scan.captures {
                    seg.punct(" ");
                    seg.value(c);
                }
                seg.punct(") ");
                seg.value(scan.init);
                seg.punct(" ");
                seg.value(scan.src);
                seg.punct(";");
            }
        }
        Mnemonic::PCodeOp(p) => {
            let op = seg.ctx.pcode_ops[p.id].to_string();
            if let Some(dst) = p.dst {
                seg.value(dst);
                seg.op(" = ");
            }
            seg.kw(&op);
            seg.punct("(");
            for (i, &arg) in p.args.iter().enumerate() {
                if i > 0 {
                    seg.punct(", ");
                }
                seg.value(arg);
            }
            seg.punct(");");
        }
        Mnemonic::Intrinsic(intr) => {
            seg.kw(&format!("${}", intr.id.name()));
            seg.punct("(");
            for (i, &arg) in intr.args.iter().enumerate() {
                if i > 0 {
                    seg.punct(", ");
                }
                seg.value(arg);
            }
            seg.punct(");");
        }
    }
}

fn cast(seg: &mut Seg, kw: &str, prefix: char, size: usize, src: ValueId) {
    seg.kw(kw);
    seg.punct("(");
    seg.push(format!("{prefix}{}", size * 8), TokenKind::Type, None);
    seg.punct(", ");
    seg.value(src);
    seg.punct(");");
}

fn unary_call(seg: &mut Seg, kw: &str, src: ValueId) {
    seg.kw(kw);
    seg.punct("(");
    seg.value(src);
    seg.punct(");");
}

fn binary_call(seg: &mut Seg, kw: &str, lhs: ValueId, rhs: ValueId) {
    seg.kw(kw);
    seg.punct("(");
    seg.value(lhs);
    seg.punct(", ");
    seg.value(rhs);
    seg.punct(");");
}

/// The space name as printed by `load`/`store`'s `fmt`: the named space, or a
/// `space: <id>` fallback for an unnamed space.
fn space_name(ctx: &Context<'_>, space: crate::space::SpaceId) -> String {
    let space_ref = Space::from_id(ctx, space);
    match space_ref.name.as_deref() {
        Some(name) => name.to_string(),
        None => format!("space: {space}"),
    }
}

/// The `@name=` / `@arg<i>=` prefix for a direct-call argument. Mirrors
/// `fmt_call_arg_name`.
fn call_arg_name(ctx: &Context<'_>, target: FunctionId, index: usize) -> String {
    match Function::from_id(ctx, target).input_arg_name(index) {
        Some(name) => format!("@{name}="),
        None => format!("@arg{index}="),
    }
}

/// The token stream for a mnemonic's rendering (the right-hand side, without the
/// `<ty> %name = ` result binding). Concatenating the token text equals
/// [`Mnemonic`]'s `Display` — `Mnemonic::fmt` is implemented by writing these.
pub fn mnemonic_tokens(ctx: &Context<'_>, m: &Mnemonic) -> Vec<Token> {
    let mut seg = Seg {
        ctx,
        out: Vec::new(),
    };
    mnemonic_segments(&mut seg, m);
    seg.out
}

/// The token stream for a single value operand. Concatenating the token text
/// equals [`ValueRef`](crate::value::ValueRef)'s `Display` — which is implemented
/// by writing these.
pub fn value_tokens(ctx: &Context<'_>, id: ValueId) -> Vec<Token> {
    let mut seg = Seg {
        ctx,
        out: Vec::new(),
    };
    seg.value(id);
    seg.out
}
