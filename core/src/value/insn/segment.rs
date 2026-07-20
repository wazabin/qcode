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
    context::{Context, Shared},
    space::Space,
    value::{
        BasicBlock, FunctionBody, LocalBlockId, LocalValueId, QCodeView, ValueId,
        block::BlockId,
        bytes::BytesRef,
        function::FunctionId,
        insn::{Callee, InstructionRef, Mnemonic},
        literal::{LiteralId, LiteralRef, SymbolicRef},
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
struct Seg<'ctx, 'str, R> {
    view: R,
    out: Vec<Token>,
    marker: std::marker::PhantomData<&'ctx &'str ()>,
}

impl<'ctx, 'str: 'ctx, R> Seg<'ctx, 'str, R>
where
    R: QCodeView<'ctx, 'str>,
{
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
            format!("{} ", self.view.shared().types.type_name(type_id)),
            TokenKind::Type,
            None,
        );
    }

    /// A value operand, rendered exactly as `ValueRef`'s `Display`: `<ty> <atom>`
    /// for scalars (instruction, block param, literal, bytes, varnode), and bare
    /// for functions/blocks.
    fn value(&mut self, id: ValueId) {
        let link = Some(Link::Value(id));
        // Note: instruction / block-param operands are never foreign. They are
        // stored as bare body-local ids and qualified with their reader's own
        // `func`, so a cross-function data operand is unrepresentable — no guard
        // is needed (or wanted: one would mask a mis-qualification). Only the
        // *absolute* ids below (blocks, and symbolic block literals) can name
        // another function.
        match id {
            ValueId::Instruction(iid) => {
                let r = self.view.insn_ref(iid);
                self.ty(r.type_id());
                self.push(instruction_atom(self.view, iid), TokenKind::Variable, link);
            }
            ValueId::BlockParam(pid) => {
                let r = self.view.param_ref(pid);
                self.ty(r.type_id());
                self.push(
                    block_param_atom(self.view, pid),
                    TokenKind::BlockParam,
                    link,
                );
            }
            ValueId::Literal(lid) => {
                let r = LiteralRef::from_id(self.view.shared(), lid);
                self.ty(r.type_id());
                // The literal *atom* is rendered from the whole `&Context`, so a
                // symbolic block/function literal resolves its target name (a
                // `&Shared`-backed `LiteralRef` cannot — context-split 5b-ii #1).
                self.push(literal_atom_view(self.view, lid), TokenKind::Literal, link);
            }
            ValueId::Bytes(bid) => {
                let r = BytesRef::from_id(self.view.shared(), bid);
                self.ty(r.type_id());
                self.push(r.to_string(), TokenKind::Bytes, link);
            }
            ValueId::Varnode(vid) => {
                let r = Varnode::from_id(self.view.shared(), vid);
                self.push(format!("i{} ", r.size() * 8), TokenKind::Type, None);
                self.push(r.to_string(), TokenKind::Varnode, link);
            }
            ValueId::Poison(pid) => {
                let ty = self.view.shared().values.poisons[pid].type_id;
                self.ty(ty);
                self.push("poison".to_string(), TokenKind::Literal, link);
            }
            ValueId::Temp(id) => {
                let r = self.view.temp_ref(id);
                self.push(format!("i{} ", r.size() * 8), TokenKind::Type, None);
                self.push(r.to_string(), TokenKind::Varnode, link);
            }
            ValueId::Function(fid) => {
                let name = self.view.interface(fid).name.to_string();
                self.push(
                    format!("<{name}>"),
                    TokenKind::Function,
                    Some(Link::Function(fid)),
                );
            }
            ValueId::BasicBlock(bid) => {
                // A block used as a value renders via the block's own `Display`
                // (never `ValueRef`'s, which routes back here — that would recurse).
                // A block in another function (a transient during discovery) is
                // unreadable through a function-scoped view, so render its id
                // instead of resolving the foreign body's block text.
                let text = if self.view.owner().is_some_and(|o| o != bid.func) {
                    format!("<{bid}>")
                } else {
                    self.view.block_ref(bid).to_string()
                };
                self.push(text, TokenKind::Label, Some(Link::Block(bid)));
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
                    instruction_atom(self.view, iid),
                    TokenKind::Variable,
                    Some(Link::Value(id)),
                );
            }
            other => self.value(other),
        }
    }

    /// A direct branch/cbranch target: `<name @p=arg …>`. Mirrors
    /// `fmt_branch_target`. The `target` is a bare body-local index; `func` is the
    /// terminator's owning function (strict IR locality ⇒ the target lives in that
    /// same arena), used to recover the full [`BlockId`].
    fn branch_target(&mut self, func: FunctionId, target: LocalBlockId, args: &[LocalValueId]) {
        let target = BlockId::new(func, target);
        let block = self.view.block_ref(target);
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
            self.value(arg.qualify(func));
        }

        self.push(">", TokenKind::Label, None);
    }
}

/// The bare atom for an instruction result: `%name` or `%tmp<id>`.
fn instruction_atom<'ctx, 'str: 'ctx>(
    view: impl QCodeView<'ctx, 'str>,
    id: crate::value::InstructionId,
) -> String {
    match view.instruction(id).name.as_deref() {
        Some(name) => format!("%{name}"),
        None => format!("%tmp{:x}", usize::from(id.local)),
    }
}

/// The bare atom for a block parameter: `@name` or `@param<id>`.
fn block_param_atom<'ctx, 'str: 'ctx>(
    view: impl QCodeView<'ctx, 'str>,
    id: crate::value::BlockParamId,
) -> String {
    let r = view.param_ref(id);
    match r.name() {
        Some(name) => format!("@{name}"),
        None => format!("@param{:x}", usize::from(id.local)),
    }
}

/// Render an instruction as colored, linkable tokens. Concatenating the tokens'
/// text equals the instruction's `Display` (`as_statement()`) output.
pub fn instruction_segments<'ctx, 'str: 'ctx, R>(insn: &InstructionRef<'str, 'ctx, R>) -> Vec<Token>
where
    R: QCodeView<'ctx, 'str>,
{
    let view = insn.view;
    let mut seg = Seg {
        view,
        out: Vec::new(),
        marker: std::marker::PhantomData,
    };

    // LHS: `<ty> %name = ` (mirrors `InstructionStatement` + `InstructionRef`'s
    // inherent `fmt`). Terminators and other size-0 instructions have no LHS.
    if insn.size() != 0 {
        seg.ty(insn.type_id());
        seg.push(
            instruction_atom(view, insn.id),
            TokenKind::Variable,
            Some(Link::Value(ValueId::Instruction(insn.id))),
        );
        seg.op(" = ");
    }

    match insn.mnemonic() {
        Mnemonic::Tuple(t) => tuple_with_type(&mut seg, insn.id.func, t, insn.type_id()),
        m => mnemonic_segments(&mut seg, insn.id.func, m),
    }

    seg.out
}

fn tuple_with_type<'ctx, 'str: 'ctx>(
    seg: &mut Seg<'ctx, 'str, impl QCodeView<'ctx, 'str>>,
    func: FunctionId,
    t: &crate::value::insn::Tuple,
    type_id: crate::types::TypeId,
) {
    seg.kw("pack");
    seg.punct("(");
    for (i, &field) in t.fields.iter().enumerate() {
        if i > 0 {
            seg.punct(", ");
        }
        let name = seg
            .view
            .shared()
            .types
            .field_name(type_id, i)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("field{}", i + 1));
        seg.push(name, TokenKind::Field, None);
        seg.op("=");
        seg.value(field.qualify(func));
    }
    seg.punct(");");
}

fn mnemonic_segments<'ctx, 'str: 'ctx>(
    seg: &mut Seg<'ctx, 'str, impl QCodeView<'ctx, 'str>>,
    func: FunctionId,
    m: &Mnemonic,
) {
    use crate::value::insn::Unop;
    match m {
        Mnemonic::Load(l) => {
            seg.kw("load");
            seg.punct("(");
            seg.push(space_name(seg.view, func, l.space), TokenKind::Space, None);
            seg.punct(":");
            seg.push(l.size.to_string(), TokenKind::Type, None);
            seg.punct(", ");
            seg.value(l.ptr.qualify(func));
            seg.punct(");");
        }
        Mnemonic::Store(s) => {
            seg.kw("store");
            seg.punct("(");
            seg.push(space_name(seg.view, func, s.space), TokenKind::Space, None);
            seg.punct(":");
            seg.push(s.size.to_string(), TokenKind::Type, None);
            seg.punct(", ");
            seg.value(s.ptr.qualify(func));
            seg.op(" <- ");
            seg.value(s.src.qualify(func));
            seg.punct(");");
        }
        Mnemonic::Branch(b) => {
            seg.kw("goto ");
            seg.branch_target(func, b.target, &b.args);
            seg.punct(";");
        }
        Mnemonic::BranchInd(b) => {
            seg.kw("goto ");
            seg.punct("[");
            seg.value(b.ptr.qualify(func));
            seg.punct("];");
        }
        Mnemonic::CBranch(cb) => {
            seg.kw("if ");
            seg.value(cb.condition.qualify(func));
            seg.kw(" goto ");
            seg.branch_target(func, cb.success_block, &cb.success_args);
            seg.kw(" else goto ");
            seg.branch_target(func, cb.failure_block, &cb.failure_args);
            seg.punct(";");
        }
        Mnemonic::Apply(a) => {
            seg.kw("apply ");
            let (target, link) = callee_name_link(seg.view, a.target);
            seg.push(target, TokenKind::Function, link);
            seg.punct("(");
            for (i, &arg) in a.args.iter().enumerate() {
                if i > 0 {
                    seg.punct(", ");
                }
                seg.value(arg.qualify(func));
            }
            seg.punct(");");
        }
        Mnemonic::Call(c) => {
            seg.kw("call fn ");
            let (target, link) = callee_name_link(seg.view, c.target);
            seg.push(target, TokenKind::Function, link);
            seg.punct("(");
            for (i, &arg) in c.args.iter().enumerate() {
                if i > 0 {
                    seg.punct(", ");
                }
                let arg_name = c
                    .target
                    .real()
                    .map(|target| call_arg_name(seg.view, target, i))
                    .unwrap_or_else(|| format!("@arg{i}="));
                seg.push(arg_name, TokenKind::BlockParam, None);
                seg.value(arg.qualify(func));
            }
            seg.punct(");");
        }
        Mnemonic::TailCall(tc) => {
            seg.kw("tailcall fn ");
            let (target, link) = callee_name_link(seg.view, tc.target);
            seg.push(target, TokenKind::Function, link);
            seg.punct("(");
            for (i, &arg) in tc.args.iter().enumerate() {
                if i > 0 {
                    seg.punct(", ");
                }
                seg.value(arg.qualify(func));
            }
            seg.punct(");");
        }
        Mnemonic::CallInd(c) => {
            seg.kw("call ");
            seg.punct("[");
            seg.value(c.ptr.qualify(func));
            seg.punct("]");
            if !c.args.is_empty() {
                seg.punct("(");
                for (i, &arg) in c.args.iter().enumerate() {
                    if i > 0 {
                        seg.punct(", ");
                    }
                    seg.value(arg.qualify(func));
                }
                seg.punct(")");
            }
            seg.punct(";");
        }
        // No operands: the marker alone. Kept short so a run of unlifted padding
        // stays readable.
        Mnemonic::BadInsn(_) => {
            seg.kw("badinsn");
            seg.punct(";");
        }
        Mnemonic::Return(r) => match r.value {
            Some(value) => {
                seg.kw("return ");
                seg.value(value.qualify(func));
                seg.kw(" at ");
                seg.value(r.ptr.qualify(func));
                seg.punct(";");
            }
            None => {
                seg.kw("return at ");
                seg.value(r.ptr.qualify(func));
                seg.punct(";");
            }
        },
        Mnemonic::ReturnValue(r) => {
            seg.kw("return ");
            seg.value(r.value.qualify(func));
            seg.punct(";");
        }
        Mnemonic::Unop(u) => match u.op {
            Unop::IntNegate | Unop::IntNot | Unop::FloatNegate => {
                seg.op(format!("{} ", u.op));
                seg.value(u.src.qualify(func));
                seg.punct(";");
            }
            _ => {
                seg.kw(&u.op.to_string());
                seg.punct("(");
                seg.value(u.src.qualify(func));
                seg.punct(");");
            }
        },
        Mnemonic::Binop(b) => {
            seg.value(b.lhs.qualify(func));
            seg.op(format!(" {} ", b.op));
            seg.value(b.rhs.qualify(func));
            seg.punct(";");
        }
        Mnemonic::Zext(z) => cast(seg, func, "zext", 'i', z.size, z.src),
        Mnemonic::Sext(s) => cast(seg, func, "sext", 'i', s.size, s.src),
        Mnemonic::IntToFloat(c) => cast(seg, func, "int2float", 'f', c.size, c.src),
        Mnemonic::FloatToFloat(c) => cast(seg, func, "float2float", 'f', c.size, c.src),
        Mnemonic::FloatToInt(c) => cast(seg, func, "trunc", 'i', c.size, c.src),
        Mnemonic::Range(r) => {
            seg.value(r.src.qualify(func));
            seg.punct("[");
            seg.push(r.start.to_string(), TokenKind::Literal, None);
            seg.punct(":");
            seg.push((r.start + r.size).to_string(), TokenKind::Literal, None);
            seg.punct("];");
        }
        Mnemonic::IsFloatNaN(o) => unary_call(seg, func, "nan", o.src),
        Mnemonic::LzCount(o) => unary_call(seg, func, "lzcount", o.src),
        Mnemonic::PopCount(o) => unary_call(seg, func, "popcount", o.src),
        Mnemonic::Carry(o) => binary_call(seg, func, "carry", o.lhs, o.rhs),
        Mnemonic::SCarry(o) => binary_call(seg, func, "scarry", o.lhs, o.rhs),
        Mnemonic::SBorrow(o) => binary_call(seg, func, "sborrow", o.lhs, o.rhs),
        Mnemonic::Assert(a) => {
            seg.kw("assert ");
            seg.value(a.condition.qualify(func));
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
                seg.value(field.qualify(func));
            }
            seg.punct(");");
        }
        Mnemonic::Extract(e) => {
            seg.kw("extract");
            seg.punct("(");
            seg.bare_value(e.agg.qualify(func));
            let name = e
                .field_name_view(seg.view, func)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("field{}", e.index + 1));
            seg.push(format!(".{name}"), TokenKind::Field, None);
            seg.punct(");");
        }
        Mnemonic::Gep(g) => {
            seg.kw("gep");
            seg.punct("(");
            seg.bare_value(g.base.qualify(func));
            match g.field_name_view(seg.view, func) {
                Some(name) => seg.push(format!(".{name}"), TokenKind::Field, None),
                None => {
                    seg.op(" + ");
                    seg.push(format!("{:#x}", g.offset), TokenKind::Literal, None);
                }
            }
            seg.punct(");");
        }
        Mnemonic::Map(map) => {
            let (body, link) = callee_name_link(seg.view, map.body);
            if map.captures.is_empty() {
                seg.push(body, TokenKind::Function, link);
                seg.op(" <$> ");
                seg.value(map.src.qualify(func));
                seg.punct(";");
            } else {
                seg.punct("(");
                seg.push(body, TokenKind::Function, link);
                for &c in &map.captures {
                    seg.punct(" ");
                    seg.value(c.qualify(func));
                }
                seg.op(") <$> ");
                seg.value(map.src.qualify(func));
                seg.punct(";");
            }
        }
        Mnemonic::Scan(scan) => {
            let (body, link) = callee_name_link(seg.view, scan.body);
            let body = match scan.body {
                Callee::Real(_) => format!("@{body}"),
                Callee::Minted(_) => body,
            };
            if scan.captures.is_empty() {
                seg.kw("scanl ");
                seg.push(body, TokenKind::Function, link);
                seg.punct(" ");
                seg.value(scan.init.qualify(func));
                seg.punct(" ");
                seg.value(scan.src.qualify(func));
                seg.punct(";");
            } else {
                seg.kw("scanl ");
                seg.punct("(");
                seg.push(body, TokenKind::Function, link);
                for &c in &scan.captures {
                    seg.punct(" ");
                    seg.value(c.qualify(func));
                }
                seg.punct(") ");
                seg.value(scan.init.qualify(func));
                seg.punct(" ");
                seg.value(scan.src.qualify(func));
                seg.punct(";");
            }
        }
        Mnemonic::PCodeOp(p) => {
            let op = seg.view.shared().pcode_ops[p.id].to_string();
            if let Some(dst) = p.dst {
                seg.value(dst.qualify(func));
                seg.op(" = ");
            }
            seg.kw(&op);
            seg.punct("(");
            for (i, &arg) in p.args.iter().enumerate() {
                if i > 0 {
                    seg.punct(", ");
                }
                seg.value(arg.qualify(func));
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
                seg.value(arg.qualify(func));
            }
            seg.punct(");");
        }
    }
}

fn cast<'ctx, 'str: 'ctx>(
    seg: &mut Seg<'ctx, 'str, impl QCodeView<'ctx, 'str>>,
    func: FunctionId,
    kw: &str,
    prefix: char,
    size: usize,
    src: LocalValueId,
) {
    seg.kw(kw);
    seg.punct("(");
    seg.push(format!("{prefix}{}", size * 8), TokenKind::Type, None);
    seg.punct(", ");
    seg.value(src.qualify(func));
    seg.punct(");");
}

fn unary_call<'ctx, 'str: 'ctx>(
    seg: &mut Seg<'ctx, 'str, impl QCodeView<'ctx, 'str>>,
    func: FunctionId,
    kw: &str,
    src: LocalValueId,
) {
    seg.kw(kw);
    seg.punct("(");
    seg.value(src.qualify(func));
    seg.punct(");");
}

fn binary_call<'ctx, 'str: 'ctx>(
    seg: &mut Seg<'ctx, 'str, impl QCodeView<'ctx, 'str>>,
    func: FunctionId,
    kw: &str,
    lhs: LocalValueId,
    rhs: LocalValueId,
) {
    seg.kw(kw);
    seg.punct("(");
    seg.value(lhs.qualify(func));
    seg.punct(", ");
    seg.value(rhs.qualify(func));
    seg.punct(");");
}

/// The space name as printed by `load`/`store`'s `fmt`: the named space, or a
/// `space: <id>` fallback for an unnamed space.
fn space_name<'ctx, 'str: 'ctx>(
    view: impl QCodeView<'ctx, 'str>,
    func: FunctionId,
    space: crate::space::LocalMemorySpaceId,
) -> String {
    match space.qualify(func) {
        crate::space::MemorySpaceId::Shared(space) => {
            let space_ref = Space::from_id(view.shared(), space);
            match space_ref.name.as_deref() {
                Some(name) => name.to_string(),
                None => format!("space: {space}"),
            }
        }
        // `$tempN` is an explicit body-local space token. The numeric local ID
        // makes the canonical print stable even when a display name is absent,
        // duplicated, or not a qcode identifier; lowering recreates one local
        // space per token in first-use order.
        crate::space::MemorySpaceId::Temp(space) => {
            format!("$temp{}", usize::from(space.local))
        }
    }
}

/// The `@name=` / `@arg<i>=` prefix for a direct-call argument. Mirrors
/// `fmt_call_arg_name`.
fn call_arg_name<'ctx, 'str: 'ctx>(
    view: impl QCodeView<'ctx, 'str>,
    target: FunctionId,
    index: usize,
) -> String {
    // The authoritative name is the callee's root block param, which lives in the
    // callee's *body*. A function-scoped view (a `BodyView`, as used by the
    // pass-fixpoint fingerprint) may not read another function's body at all, so
    // for a foreign callee fall back to the interface-only name — the C-prototype
    // argument name, if any, else the positional form. Purely cosmetic: only the
    // rendered argument label changes, never the operand itself.
    let foreign = view.owner().is_some_and(|owner| owner != target);
    let name = if foreign {
        view.interface(target)
            .signature
            .as_ref()
            .and_then(|s| s.extern_interface.as_ref())
            .and_then(|iface| iface.args.get(index))
            .and_then(|a| a.name.as_ref().map(|n| n.to_string()))
    } else {
        view.function_ref(target).input_arg_name(index)
    };
    match name {
        Some(name) => format!("@{name}="),
        None => format!("@arg{index}="),
    }
}

/// Render a real function symbol or an unresolved pass-local placeholder.
/// Placeholders deliberately carry no link: they are not installed functions.
fn callee_name_link<'ctx, 'str: 'ctx>(
    view: impl QCodeView<'ctx, 'str>,
    callee: Callee,
) -> (String, Option<Link>) {
    match callee {
        Callee::Real(id) => (
            view.interface(id).name.to_string(),
            Some(Link::Function(id)),
        ),
        Callee::Minted(slot) => (format!("<minted:{slot}>"), None),
    }
}

/// The token stream for a single value operand. Concatenating the token text
/// equals [`ValueRef`](crate::value::ValueRef)'s `Display` — which is implemented
/// by writing these.
pub fn value_tokens(ctx: &Context<'_>, id: ValueId) -> Vec<Token> {
    let mut seg = Seg {
        view: crate::value::ModuleView::new(ctx),
        out: Vec::new(),
        marker: std::marker::PhantomData,
    };
    seg.value(id);
    seg.out
}

/// Provider-generic value rendering used by immutable arena-cluster refs.
pub fn value_tokens_view<'ctx, 'str: 'ctx, R>(view: R, id: ValueId) -> Vec<Token>
where
    R: QCodeView<'ctx, 'str>,
{
    let link = Some(Link::Value(id));
    let shared = view.shared();
    let mut out = Vec::new();
    let mut typed = |type_id, text: String, kind| {
        out.push(Token::new(
            format!("{} ", shared.types.type_name(type_id)),
            TokenKind::Type,
            None,
        ));
        out.push(Token::new(text, kind, link));
    };

    match id {
        ValueId::Instruction(iid) => {
            let insn = view.instruction(iid);
            let atom = insn.name.as_deref().map_or_else(
                || {
                    let local: usize = iid.local.into();
                    format!("%tmp{local:x}")
                },
                |name| format!("%{name}"),
            );
            typed(insn.type_id, atom, TokenKind::Variable);
        }
        ValueId::BlockParam(pid) => {
            let param = view.block_param(pid);
            let atom = param.name.as_deref().map_or_else(
                || {
                    let local: usize = pid.local.into();
                    format!("@param{local:x}")
                },
                |name| format!("@{name}"),
            );
            typed(param.type_id, atom, TokenKind::BlockParam);
        }
        ValueId::Literal(lid) => {
            let literal = &shared.values.literals[lid];
            let atom = literal_atom_view(view, lid);
            typed(literal.type_id, atom, TokenKind::Literal);
        }
        ValueId::Bytes(id) => {
            let value = BytesRef::from_id(shared, id);
            typed(value.type_id(), value.to_string(), TokenKind::Bytes);
        }
        ValueId::Varnode(id) => {
            let value = Varnode::from_id(shared, id);
            out.push(Token::new(
                format!("i{} ", value.size() * 8),
                TokenKind::Type,
                None,
            ));
            out.push(Token::new(value.to_string(), TokenKind::Varnode, link));
        }
        ValueId::Temp(id) => {
            let value = view.temp_ref(id);
            out.push(Token::new(
                format!("i{} ", value.size() * 8),
                TokenKind::Type,
                None,
            ));
            out.push(Token::new(value.to_string(), TokenKind::Varnode, link));
        }
        ValueId::Function(id) => out.push(Token::new(
            format!("<{}>", view.interface(id).name),
            TokenKind::Function,
            Some(Link::Function(id)),
        )),
        ValueId::BasicBlock(id) => out.push(Token::new(
            view.block_ref(id).to_string(),
            TokenKind::Label,
            Some(Link::Block(id)),
        )),
        ValueId::Poison(id) => {
            typed(
                shared.values.poisons[id].type_id,
                "poison".to_string(),
                TokenKind::Literal,
            );
        }
    }
    out
}

/// The rendered *atom* (no `<ty>` prefix) of the literal `id`, resolved against
/// the whole `&Context` so symbolic block/function literals show their target
/// name. This is the full-context twin of `LiteralRef`'s `Display` (which, being
/// `&Shared`-backed, cannot reach body/interface names and falls back to the
/// numeric form). Used by the instruction renderer and the dataflow graph.
/// Concatenating with the type prefix reproduces the pre-narrowing rendering
/// byte-for-byte (context-split stage 5b-ii item #1).
pub fn literal_atom(ctx: &Context<'_>, id: LiteralId) -> String {
    let literal = &ctx.shared.values.literals[id];
    match &literal.symbolic {
        Some(SymbolicRef::Block(bid)) => match BasicBlock::from_id(ctx, *bid).name() {
            Some(name) => format!("&<{}>", name),
            None => format!("&<0x{:x}>", literal.value),
        },
        Some(SymbolicRef::Function(fid)) => {
            format!("&<{}>", FunctionBody::from_id(ctx, *fid).name())
        }
        Some(SymbolicRef::String(s)) => format!("&{:?}", s),
        None if ctx.shared.types.is_bool(literal.type_id) => {
            (if literal.value != 0 { "true" } else { "false" }).to_string()
        }
        None => format!("0x{:x}", literal.value),
    }
}

fn literal_atom_view<'ctx, 'str: 'ctx>(view: impl QCodeView<'ctx, 'str>, id: LiteralId) -> String {
    let shared = view.shared();
    let literal = &shared.values.literals[id];
    match &literal.symbolic {
        // A symbolic block-ref into *another* function (a transient during
        // discovery/jump-table recovery) cannot be name-resolved through a
        // function-scoped `BodyView` — reading the foreign body trips the locality
        // guard — so fall back to the numeric form. A whole-module view (`owner()
        // == None`) resolves the name normally.
        Some(SymbolicRef::Block(id)) if view.owner().is_some_and(|o| o != id.func) => {
            format!("&<0x{:x}>", literal.value)
        }
        Some(SymbolicRef::Block(id)) => view.block_ref(*id).name().map_or_else(
            || format!("&<0x{:x}>", literal.value),
            |name| format!("&<{name}>"),
        ),
        Some(SymbolicRef::Function(id)) => format!("&<{}>", view.interface(*id).name),
        Some(SymbolicRef::String(value)) => format!("&{value:?}"),
        None if shared.types.is_bool(literal.type_id) => {
            (if literal.value != 0 { "true" } else { "false" }).to_string()
        }
        None => format!("0x{:x}", literal.value),
    }
}

/// The token stream for a **shared-leaf** value operand (literal, bytes, varnode),
/// rendered from only the module's [`Shared`] IR state. The `&Shared` twin of
/// [`value_tokens`] for the operands a `&Shared`-backed [`ValueRef`] can hold;
/// symbolic block/function literals fall back to the numeric form (their names
/// live in bodies/interfaces, out of a `&Shared`'s reach). Panics on
/// arena-cluster ids, which a shared-leaf ref never carries.
pub fn value_tokens_shared(shared: &Shared<'_>, id: ValueId) -> Vec<Token> {
    let link = Some(Link::Value(id));
    let mut out = Vec::new();
    match id {
        ValueId::Literal(lid) => {
            let r = LiteralRef::from_id(shared, lid);
            out.push(Token::new(
                format!("{} ", shared.types.type_name(r.type_id())),
                TokenKind::Type,
                None,
            ));
            out.push(Token::new(r.to_string(), TokenKind::Literal, link));
        }
        ValueId::Bytes(bid) => {
            let r = BytesRef::from_id(shared, bid);
            out.push(Token::new(
                format!("{} ", shared.types.type_name(r.type_id())),
                TokenKind::Type,
                None,
            ));
            out.push(Token::new(r.to_string(), TokenKind::Bytes, link));
        }
        ValueId::Varnode(vid) => {
            let r = Varnode::from_id(shared, vid);
            out.push(Token::new(
                format!("i{} ", r.size() * 8),
                TokenKind::Type,
                None,
            ));
            out.push(Token::new(r.to_string(), TokenKind::Varnode, link));
        }
        ValueId::Poison(pid) => {
            let ty = shared.values.poisons[pid].type_id;
            out.push(Token::new(
                format!("{} ", shared.types.type_name(ty)),
                TokenKind::Type,
                None,
            ));
            out.push(Token::new("poison".to_string(), TokenKind::Literal, link));
        }
        _ => panic!("value_tokens_shared: not a shared-leaf value id"),
    }
    out
}
