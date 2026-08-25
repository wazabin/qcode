//! Pseudo-C emission for a lowered [`Program`].
//!
//! Emits a stream of classified [`TokenLine`]s: compound C statements (pure
//! single-use values are folded into their users; only side-effecting or
//! multiply-used values become their own assignments), syntax-classified for
//! highlighting, and tagged with indentation and block provenance.

use std::collections::{HashMap, HashSet};

use qcode::{
    context::Context,
    value::{
        BlockId, BlockParamRef, FunctionBody, FunctionRef, Instruction, InstructionId,
        LocalValueId, Value, ValueId, ValueRef, Varnode, VarnodeId, function::FunctionId,
        insn::Mnemonic,
    },
};

use super::{
    ast::{Program, Stmt},
    cast::{Expr, ExprKind},
    lower_expr::{
        deref_location, instruction_declared_signed, instruction_name, lower_condition,
        lower_defining_expr, lower_expr_rooted,
    },
    strings::StringPool,
    tokens::{LineBuf, TokenKind, TokenLine},
};

/// Renders `program` as classified, indented token lines.
///
/// When the program knows its source function, the body is wrapped in a
/// `fn name(args) -> rets {` … `}` header and indented one level beneath it.
///
/// `binary` is the loaded image the program was lifted from, when the caller
/// still holds it. It is only read to reconstruct string constants out of
/// read-only memory (see [`strings`](super::strings)); passing `None` renders
/// every such argument as the bare address it is in the IR.
pub fn emit_tokens(
    ctx: &Context,
    program: &Program,
    binary: Option<&dyn wazabin_binary::BinaryFormat>,
) -> Vec<TokenLine> {
    let roots = compute_roots(ctx, program);
    let hoisted = hoisted_roots(ctx, program, &roots);
    let strings = StringPool::new(ctx, binary);
    // The body is rendered first: which rodata objects need declaring is only
    // known once every call argument has been through the pool, and the
    // declarations belong above the function that uses them.
    let mut body = Vec::new();
    match program.function {
        Some(function_id) => {
            body.push(header_line(ctx, program, function_id));
            emit_hoisted_decls(ctx, &roots, &hoisted, 1, &mut body);
            emit_stmts(
                ctx,
                program,
                &program.stmts,
                1,
                &roots,
                &hoisted,
                &strings,
                &mut body,
            );
            body.push(brace_line("}", 0));
        }
        None => {
            emit_hoisted_decls(ctx, &roots, &hoisted, 0, &mut body);
            emit_stmts(
                ctx,
                program,
                &program.stmts,
                0,
                &roots,
                &hoisted,
                &strings,
                &mut body,
            );
        }
    }

    let mut out = strings.declarations();
    if !out.is_empty() {
        out.push(LineBuf::default().into_line(0, None));
    }
    out.append(&mut body);
    out
}

/// Bare declarations for the values whose definition sits in a scope some reader
/// of them has already left. Their defining statement then assigns without
/// re-declaring.
/// Infer the pointee type of an SSA address from its direct load/store users.
/// Mixed-width accesses stay untyped; a consistent cell is emitted as `T *`.
fn inferred_pointee(ctx: &Context, id: InstructionId) -> Option<qcode::types::TypeId> {
    let value = ValueId::Instruction(id);
    let mut pointee = None;
    for user in ctx.users(value) {
        let insn = Instruction::from_id(ctx, user);
        let candidate = match insn.mnemonic() {
            Mnemonic::Load(load) if load.ptr.qualify(user.func) == value => Some(insn.type_id()),
            Mnemonic::Store(store) if store.ptr.qualify(user.func) == value => {
                Some(ctx.type_of(store.src.qualify(user.func)))
            }
            _ => None,
        };
        if let Some(candidate) = candidate {
            if pointee.is_some_and(|old| old != candidate) {
                return None;
            }
            pointee = Some(candidate);
        }
    }
    pointee
}

fn emit_inferred_decl_type(ctx: &Context, id: InstructionId, buf: &mut LineBuf) {
    if let Some(pointee) = inferred_pointee(ctx, id) {
        emit_c_type(ctx, pointee, ctx.shared.types.size_of(pointee), buf);
        buf.space();
        buf.punct("*");
    } else if instruction_declared_signed(ctx, id) {
        buf.push(
            format!("int{}_t", Instruction::from_id(ctx, id).size() * 8),
            TokenKind::Type,
        );
    } else {
        let insn = Instruction::from_id(ctx, id);
        emit_c_type(ctx, insn.type_id(), insn.size(), buf);
    }
}

fn emit_hoisted_decls(
    ctx: &Context,
    roots: &HashSet<InstructionId>,
    hoisted: &HashSet<InstructionId>,
    indent: usize,
    out: &mut Vec<TokenLine>,
) {
    let mut ids: Vec<InstructionId> = hoisted.iter().copied().collect();
    ids.sort_unstable_by_key(|id| (id.func, usize::from(id.local)));
    for id in ids {
        let mut buf = LineBuf::default();
        emit_inferred_decl_type(ctx, id, &mut buf);
        buf.space();
        buf.push_value(
            instruction_name(ctx, id, roots),
            TokenKind::Variable,
            Some(ValueId::Instruction(id)),
        );
        buf.punct(";");
        out.push(buf.into_line(indent, None));
    }
}

/// Builds the `fn name(inputs) -> { output types } {` header from the function's
/// recovered signature. A missing signature (or an empty input/output list)
/// simply renders empty parentheses / no return arrow.
fn header_line(ctx: &Context, program: &Program, function_id: FunctionId) -> TokenLine {
    let function = FunctionRef::from_id(ctx, function_id);
    let registers = function.effects().materialized();

    let mut buf = LineBuf::default();
    buf.keyword("fn");
    buf.space();
    buf.push(function.name().to_string(), TokenKind::Label);

    buf.punct("(");
    if let Some(registers) = registers {
        let input_types: Vec<_> = function
            .root()
            .map(|root| root.params().map(|param| param.type_id()).collect())
            .unwrap_or_default();
        emit_typed_regs(ctx, &registers.inputs, &input_types, &mut buf);
        if let Some(root) = function.root() {
            for (extra_index, param) in root.params().skip(registers.inputs.len()).enumerate() {
                if !registers.inputs.is_empty() || extra_index > 0 {
                    buf.punct(",");
                    buf.space();
                }
                emit_typed_param(ctx, param, &mut buf);
            }
        }
    }
    buf.punct(")");

    if let Some(outputs) = registers
        .map(|registers| &registers.outputs[..registers.returns])
        .filter(|outputs| !outputs.is_empty())
    {
        // Poison is internal ABI-clobber bookkeeping, not a decompiled result.
        let visible = visible_return_indices(ctx, function_id, outputs.len());
        if !visible.is_empty() {
            buf.space();
            buf.punct("->");
            buf.space();
            buf.punct("{");
            let returned = returned_field_types(ctx, program, outputs.len());
            for (emitted_index, &index) in visible.iter().enumerate() {
                if emitted_index > 0 {
                    buf.punct(",");
                    buf.space();
                }
                let id = outputs[index];
                // Prefer the type of the value actually returned in this slot: the
                // interface records only a register varnode, which carries no
                // type, so a returned code pointer would otherwise print as a
                // plain integer of the same width.
                match returned
                    .as_ref()
                    .and_then(|types| types.get(index).copied())
                {
                    Some(type_id) => {
                        emit_c_type(ctx, type_id, Varnode::from_id(ctx, id).size(), &mut buf)
                    }
                    None => emit_uint_type(Varnode::from_id(ctx, id).size(), &mut buf),
                }
            }
            buf.punct("}");
        }
    }

    buf.space();
    buf.punct("{");
    buf.into_line(0, None)
}

/// Pushes a comma-separated list of typed register arguments onto `buf`.
///
/// Register effects recover each argument's width but not its signedness, so
/// use the corresponding unsigned fixed-width C type rather than inventing a
/// signed source-level type.
fn emit_typed_regs(
    ctx: &Context,
    regs: &[VarnodeId],
    param_types: &[qcode::types::TypeId],
    buf: &mut LineBuf,
) {
    for (i, &id) in regs.iter().enumerate() {
        if i > 0 {
            buf.punct(",");
            buf.space();
        }
        let register = Varnode::from_id(ctx, id);
        if let Some(&type_id) = param_types.get(i) {
            emit_c_type(ctx, type_id, ctx.shared.types.size_of(type_id), buf);
        } else {
            emit_uint_type(register.size(), buf);
        }
        buf.space();
        buf.push(register.to_string(), TokenKind::Variable);
    }
}

fn emit_uint_type(size: usize, buf: &mut LineBuf) {
    buf.push(format!("uint{}_t", size * 8), TokenKind::Type);
}

/// Pushes the C spelling of `type_id` with no declarator name, so a pointer
/// renders as `code_t*`. Falls back to the unsigned type for `size` when the
/// qcode type carries nothing beyond its width.
fn emit_c_type(ctx: &Context, type_id: qcode::types::TypeId, size: usize, buf: &mut LineBuf) {
    match ctx.shared.types.get(type_id).repr() {
        qcode::types::TypeRepr::CodePointer { .. } => {
            buf.push("code_t", TokenKind::Type);
            buf.punct("*");
        }
        qcode::types::TypeRepr::SpaceAddress { .. } => {
            buf.push("void", TokenKind::Type);
            buf.space();
            buf.punct("*");
        }
        qcode::types::TypeRepr::StructPointer { pointee, .. } => {
            match ctx.shared.types.get(pointee).repr() {
                qcode::types::TypeRepr::Struct { name, .. } => {
                    buf.keyword("struct");
                    buf.space();
                    buf.push(name, TokenKind::Type);
                }
                _ => emit_c_type(ctx, pointee, ctx.shared.types.size_of(pointee), buf),
            }
            buf.space();
            buf.punct("*");
        }
        _ => emit_uint_type(size, buf),
    }
}

/// The types of the values the function's return pack actually yields, aligned
/// with the interface's output slots.
///
/// The interface records each output as a register [`VarnodeId`], which carries
/// a width but no type. The returned values do carry types, so the declared
/// return type is read off the `Tuple` feeding the `Return`. `None` when the
/// program has no such return, or when it does not line up with `outputs`.
fn returned_field_types(
    ctx: &Context,
    program: &Program,
    outputs: usize,
) -> Option<Vec<qcode::types::TypeId>> {
    let mut raws = Vec::new();
    collect_raws(&program.stmts, &mut raws);
    raws.into_iter().find_map(|return_id| {
        let Mnemonic::Return(ret) = Instruction::from_id(ctx, return_id).mnemonic() else {
            return None;
        };
        let ValueId::Instruction(tuple_id) = ret.value?.qualify(return_id.func) else {
            return None;
        };
        let Mnemonic::Tuple(tuple) = Instruction::from_id(ctx, tuple_id).mnemonic() else {
            return None;
        };
        (tuple.fields.len() == outputs).then(|| {
            tuple
                .fields
                .iter()
                .map(|&field| ctx.type_of(field.qualify(return_id.func)))
                .collect()
        })
    })
}

/// Return-pack positions that are meaningful decompiler outputs.
///
/// A poison in any return arm means that position is an ABI clobber on at least
/// one path. Omit it from every rendered return so headers and initializers
/// remain aligned.
fn visible_return_indices(ctx: &Context, function_id: FunctionId, outputs: usize) -> Vec<usize> {
    let mut visible = vec![true; outputs];
    for block in FunctionBody::from_id(ctx, function_id).blocks() {
        let Some(return_id) = block
            .iter()
            .find(|insn| matches!(insn.mnemonic(), Mnemonic::Return(_)))
            .map(|insn| insn.id)
        else {
            continue;
        };
        let Mnemonic::Return(ret) = Instruction::from_id(ctx, return_id).mnemonic() else {
            unreachable!("return instruction changed while rendering")
        };
        let Some(ValueId::Instruction(tuple_id)) =
            ret.value.map(|value| value.qualify(function_id))
        else {
            continue;
        };
        let Mnemonic::Tuple(tuple) = Instruction::from_id(ctx, tuple_id).mnemonic() else {
            continue;
        };
        if tuple.fields.len() != outputs {
            continue;
        }
        for (index, field) in tuple.fields.iter().enumerate() {
            if field.qualify(function_id).is_poison() {
                visible[index] = false;
            }
        }
    }
    visible
        .into_iter()
        .enumerate()
        .filter_map(|(index, visible)| visible.then_some(index))
        .collect()
}

fn emit_typed_param(ctx: &Context, param: BlockParamRef<'_, '_>, buf: &mut LineBuf) {
    emit_c_type(ctx, param.type_id(), param.size(), buf);
    buf.space();
    buf.push(
        param
            .name()
            .map(str::to_owned)
            .unwrap_or_else(|| format!("param{}", usize::from(param.id.local))),
        TokenKind::Variable,
    );
}

/// Renders `program` as pseudo-C source text (indentation via spaces).
///
/// `binary` is the loaded image, used only for string-constant reconstruction;
/// see [`emit_tokens`].
pub fn emit_c(
    ctx: &Context,
    program: &Program,
    binary: Option<&dyn wazabin_binary::BinaryFormat>,
) -> String {
    let mut out = String::new();
    for line in emit_tokens(ctx, program, binary) {
        out.push_str(&"    ".repeat(line.indent));
        out.push_str(&line.text());
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------
// Root selection: which instructions get their own statement.
// ---------------------------------------------------------------------------

/// The instructions that are emitted as their own statements. Everything else
/// (pure, used exactly once) is inlined into its user.
fn compute_roots(ctx: &Context, program: &Program) -> HashSet<InstructionId> {
    let mut raws = Vec::new();
    collect_raws(&program.stmts, &mut raws);
    raws.into_iter().filter(|&id| is_root(ctx, id)).collect()
}

/// Roots whose declaration has to move to the top of the function.
///
/// A root is declared where it is defined (`uintN_t a = …;`), which is correct
/// only while every reference sits in that scope or one nested inside it. A loop
/// body and an `if` arm are scopes: a value defined in one and read after it
/// would be declared inside a block the reader has already left, so the name
/// dangles. gcc produces exactly that shape when it turns a recursive call into
/// a loop and the result is read after it.
///
/// Scopes are identified by the path of enclosing-body ids at each point, so a
/// reference is inside a definition's scope exactly when that definition's path
/// is a prefix of the reference's.
///
/// A value can be defined more than once: cross-jump reversal duplicates a block,
/// so the same instruction is emitted in several scopes, each declaring and using
/// it locally. Such a value escapes only if some reference is covered by *none*
/// of its definitions — checking against a single one would hoist every
/// duplicated block's locals for no reason.
fn hoisted_roots(
    ctx: &Context,
    program: &Program,
    roots: &HashSet<InstructionId>,
) -> HashSet<InstructionId> {
    let mut defs: HashMap<InstructionId, Vec<Vec<usize>>> = HashMap::new();
    let mut refs: Vec<(InstructionId, Vec<usize>)> = Vec::new();
    let mut next_scope = 0usize;
    scan_scopes(
        ctx,
        &program.stmts,
        roots,
        &mut Vec::new(),
        &mut next_scope,
        &mut defs,
        &mut refs,
    );

    refs.into_iter()
        .filter(|(id, at)| {
            defs.get(id)
                .is_some_and(|declared| !declared.iter().any(|scope| at.starts_with(scope)))
        })
        .map(|(id, _)| id)
        .collect()
}

/// Record where each root is defined and where each is referenced, tagging both
/// with the stack of enclosing scopes.
#[allow(clippy::too_many_arguments)]
fn scan_scopes(
    ctx: &Context,
    stmts: &[Stmt],
    roots: &HashSet<InstructionId>,
    path: &mut Vec<usize>,
    next_scope: &mut usize,
    defs: &mut HashMap<InstructionId, Vec<Vec<usize>>>,
    refs: &mut Vec<(InstructionId, Vec<usize>)>,
) {
    let nested = |body: &[Stmt],
                  path: &mut Vec<usize>,
                  next_scope: &mut usize,
                  defs: &mut HashMap<InstructionId, Vec<Vec<usize>>>,
                  refs: &mut Vec<(InstructionId, Vec<usize>)>| {
        *next_scope += 1;
        path.push(*next_scope);
        scan_scopes(ctx, body, roots, path, next_scope, defs, refs);
        path.pop();
    };

    for stmt in stmts {
        match stmt {
            Stmt::Raw(id) => {
                if roots.contains(id) {
                    defs.entry(*id).or_default().push(path.clone());
                }
                let mut seen = HashSet::default();
                let mut found = Vec::new();
                for arg in Instruction::from_id(ctx, *id).mnemonic().args() {
                    value_root_refs(ctx, arg.qualify(id.func), roots, &mut found, &mut seen);
                }
                refs.extend(found.into_iter().map(|r| (r, path.clone())));
            }
            Stmt::Assign { value, .. } | Stmt::SaveTemp { value, .. } => {
                let mut seen = HashSet::default();
                let mut found = Vec::new();
                value_root_refs(ctx, *value, roots, &mut found, &mut seen);
                refs.extend(found.into_iter().map(|r| (r, path.clone())));
            }
            Stmt::GotoIf { cond, .. } => expr_root_refs(cond, roots, path, refs),
            Stmt::If { cond, then, els } => {
                expr_root_refs(cond, roots, path, refs);
                nested(then, path, next_scope, defs, refs);
                nested(els, path, next_scope, defs, refs);
            }
            Stmt::While { cond, body } | Stmt::DoWhile { cond, body } => {
                expr_root_refs(cond, roots, path, refs);
                nested(body, path, next_scope, defs, refs);
            }
            Stmt::Loop { body } => nested(body, path, next_scope, defs, refs),
            Stmt::Switch {
                scrutinee,
                cases,
                default,
            } => {
                expr_root_refs(scrutinee, roots, path, refs);
                for case in cases {
                    nested(&case.body, path, next_scope, defs, refs);
                }
                nested(default, path, next_scope, defs, refs);
            }
            _ => {}
        }
    }
}

/// Roots an expression names directly (an inlined sub-expression is expanded in
/// place and names nothing).
fn expr_root_refs(
    expr: &Expr,
    roots: &HashSet<InstructionId>,
    path: &[usize],
    refs: &mut Vec<(InstructionId, Vec<usize>)>,
) {
    let mut found = Vec::new();
    collect_expr_roots(expr, roots, &mut found);
    refs.extend(found.into_iter().map(|r| (r, path.to_vec())));
}

fn collect_expr_roots(expr: &Expr, roots: &HashSet<InstructionId>, out: &mut Vec<InstructionId>) {
    if let Some(ValueId::Instruction(id)) = expr.value
        && roots.contains(&id)
    {
        out.push(id);
    }
    match &expr.kind {
        ExprKind::Const(_) | ExprKind::Var(_) => {}
        ExprKind::Unary(_, e)
        | ExprKind::Deref { ptr: e, .. }
        | ExprKind::Cast { expr: e, .. }
        | ExprKind::Field { base: e, .. } => collect_expr_roots(e, roots, out),
        ExprKind::Binary(_, a, b) => {
            collect_expr_roots(a, roots, out);
            collect_expr_roots(b, roots, out);
        }
        ExprKind::Aggregate(fields) => {
            for (_, e) in fields {
                collect_expr_roots(e, roots, out);
            }
        }
        ExprKind::Unknown { operands, .. } => {
            for e in operands {
                collect_expr_roots(e, roots, out);
            }
        }
    }
}

/// Walk down from `v`, stopping at each root: those are the values the rendered
/// expression refers to by name rather than expanding.
fn value_root_refs(
    ctx: &Context,
    v: ValueId,
    roots: &HashSet<InstructionId>,
    out: &mut Vec<InstructionId>,
    seen: &mut HashSet<InstructionId>,
) {
    let ValueId::Instruction(id) = v else { return };
    if roots.contains(&id) {
        out.push(id);
        return;
    }
    if !seen.insert(id) {
        return;
    }
    for arg in Instruction::from_id(ctx, id).mnemonic().args() {
        value_root_refs(ctx, arg.qualify(id.func), roots, out, seen);
    }
}

fn collect_raws(stmts: &[Stmt], out: &mut Vec<InstructionId>) {
    for stmt in stmts {
        match stmt {
            Stmt::Raw(id) => out.push(*id),
            Stmt::If { then, els, .. } => {
                collect_raws(then, out);
                collect_raws(els, out);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::Loop { body } => {
                collect_raws(body, out);
            }
            Stmt::Switch { cases, default, .. } => {
                for case in cases {
                    collect_raws(&case.body, out);
                }
                collect_raws(default, out);
            }
            _ => {}
        }
    }
}

/// Whether instruction `id` is emitted as its own statement, rather than inlined
/// into its single user. This is the one inlining contract the backend shares:
/// [`refine::is_condition_only`](super::refine::is_condition_only) is exactly its
/// negation for a [`Stmt::Raw`], so a statement one phase treats as invisibly
/// folded is one the emitter actually inlines.
pub(crate) fn is_root(ctx: &Context, id: InstructionId) -> bool {
    let insn = Instruction::from_id(ctx, id);
    if is_side_effecting(insn.mnemonic()) {
        return true;
    }
    // A value used by more than one instruction is named to avoid duplicating
    // its expression; a value used zero times is dead and gets elided (not a
    // root). Exactly-once uses are inlined into their user — except a memory
    // load, which reads state a later store or call can change, so it may fold
    // into its user only when nothing between them can write that memory.
    let users = ctx.users(ValueId::Instruction(id));
    if users.len() != 1 {
        return users.len() > 1;
    }
    is_memory_read(insn.mnemonic()) && !load_safe_to_fold(ctx, id, users[0])
}

/// Whether a single-use load may be inlined into its user rather than named:
/// only when the user is in the same block and no instruction between them can
/// write memory, so the loaded value cannot have changed at the point of use.
/// Otherwise folding it would reorder the read past an aliasing write (the
/// classic `t = *x; *x = …; use(t)` hazard).
fn load_safe_to_fold(ctx: &Context, load: InstructionId, user: InstructionId) -> bool {
    let load_insn = Instruction::from_id(ctx, load);
    let Some(block) = load_insn.block() else {
        return false;
    };
    if Instruction::from_id(ctx, user).block().map(|b| b.id) != Some(block.id) {
        return false;
    }
    // Walk the block from the load to its user; any memory writer in between
    // (store, call, or other side effect) forbids the fold.
    let mut after_load = false;
    for insn in block.instructions() {
        if insn.id == load {
            after_load = true;
        } else if insn.id == user {
            return after_load;
        } else if after_load && is_side_effecting(insn.mnemonic()) {
            return false;
        }
    }
    false
}

/// Whether an instruction must always be a statement because it has effects.
/// Shares its definition with DCE via [`Mnemonic::has_side_effects`] so the two
/// cannot drift (e.g. DCE deleting an `Assert` the emitter would print).
pub(crate) fn is_side_effecting(m: &Mnemonic) -> bool {
    m.has_side_effects()
}

/// Whether an instruction reads memory — a load, whose result is only valid
/// until the next aliasing write. Such values are not freely foldable.
pub(crate) fn is_memory_read(m: &Mnemonic) -> bool {
    matches!(m, Mnemonic::Load(_))
}

// ---------------------------------------------------------------------------
// Statement emission.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn emit_stmts(
    ctx: &Context,
    program: &Program,
    stmts: &[Stmt],
    indent: usize,
    roots: &HashSet<InstructionId>,
    hoisted: &HashSet<InstructionId>,
    strings: &StringPool,
    out: &mut Vec<TokenLine>,
) {
    let mut line_indent = indent;
    for stmt in stmts {
        if matches!(stmt, Stmt::Label(_)) {
            emit_stmt(ctx, program, stmt, indent, roots, hoisted, strings, out);
            line_indent = indent + 1;
        } else {
            emit_stmt(
                ctx,
                program,
                stmt,
                line_indent,
                roots,
                hoisted,
                strings,
                out,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_stmt(
    ctx: &Context,
    program: &Program,
    stmt: &Stmt,
    indent: usize,
    roots: &HashSet<InstructionId>,
    hoisted: &HashSet<InstructionId>,
    strings: &StringPool,
    out: &mut Vec<TokenLine>,
) {
    match stmt {
        Stmt::Label(block) => {
            let mut buf = LineBuf::default();
            buf.push(label_of(program, *block), TokenKind::Label);
            buf.punct(":");
            out.push(buf.into_line(indent, Some(*block)));
        }
        Stmt::Raw(id) => {
            // Non-root values are inlined into their users, not emitted here.
            if !roots.contains(id) {
                return;
            }
            // A store of poison models an ABI-clobbered register. It is vital to
            // analysis but has no source-level statement to decompile.
            if matches!(Instruction::from_id(ctx, *id).mnemonic(), Mnemonic::Store(store)
                if store.src.qualify(id.func).is_poison())
            {
                return;
            }
            let block = Instruction::from_id(ctx, *id).block().map(|b| b.id);
            if emit_named_return_tuple(ctx, program, *id, indent, roots, out) {
                return;
            }
            let mut buf = statement(ctx, *id, roots, hoisted, strings);
            // The statement's operand expressions were lowered with `Some(roots)`,
            // so `buf.insns` holds every genuinely inlined operand plus any root
            // referenced by name. Drop the named roots (each is its own line) and
            // add the statement's own instruction.
            buf.insns.retain(|i| !roots.contains(i));
            buf.note_insn(*id);
            out.push(buf.into_line(indent, block));
        }
        Stmt::Assign { param, value } => {
            // A phi copy `param = value;`. The value is lowered against the roots
            // so an inlined operand renders inline and a named root by name.
            let mut buf = LineBuf::default();
            lower_expr_rooted(ctx, *param, roots).write_tokens(&mut buf);
            assign(&mut buf);
            lower_expr_rooted(ctx, *value, roots).write_tokens(&mut buf);
            buf.punct(";");
            out.push(buf.into_line(indent, None));
        }
        Stmt::SaveTemp { temp, value } => {
            let mut buf = LineBuf::default();
            buf.push(temp_name(*temp), TokenKind::Variable);
            assign(&mut buf);
            lower_expr_rooted(ctx, *value, roots).write_tokens(&mut buf);
            buf.punct(";");
            out.push(buf.into_line(indent, None));
        }
        Stmt::AssignTemp { param, temp } => {
            let mut buf = LineBuf::default();
            lower_expr_rooted(ctx, *param, roots).write_tokens(&mut buf);
            assign(&mut buf);
            buf.push(temp_name(*temp), TokenKind::Variable);
            buf.punct(";");
            out.push(buf.into_line(indent, None));
        }
        Stmt::Goto(target) => {
            let mut buf = LineBuf::default();
            buf.keyword("goto");
            buf.space();
            buf.push(label_of(program, *target), TokenKind::Label);
            buf.punct(";");
            out.push(buf.into_line(indent, None));
        }
        Stmt::GotoIf { cond, target } => {
            let mut buf = LineBuf::default();
            buf.keyword("if");
            buf.space();
            buf.punct("(");
            lower_condition(ctx, cond, roots).write_tokens(&mut buf);
            buf.punct(")");
            buf.space();
            buf.keyword("goto");
            buf.space();
            buf.push(label_of(program, *target), TokenKind::Label);
            buf.punct(";");
            out.push(buf.into_line(indent, None));
        }
        Stmt::If { cond, then, els } => {
            let mut head = LineBuf::default();
            head.keyword("if");
            head.space();
            head.punct("(");
            lower_condition(ctx, cond, roots).write_tokens(&mut head);
            head.punct(")");
            head.space();
            head.punct("{");
            out.push(head.into_line(indent, None));

            emit_stmts(ctx, program, then, indent + 1, roots, hoisted, strings, out);

            if els.is_empty() {
                out.push(brace_line("}", indent));
            } else {
                // Emit the braces as their own tokens (rather than one combined
                // `} else {` string) so the front-end can match them as brackets.
                let mut mid = LineBuf::default();
                mid.punct("}");
                mid.space();
                mid.keyword("else");
                mid.space();
                mid.punct("{");
                out.push(mid.into_line(indent, None));
                emit_stmts(ctx, program, els, indent + 1, roots, hoisted, strings, out);
                out.push(brace_line("}", indent));
            }
        }
        Stmt::While { cond, body } => {
            let mut head = LineBuf::default();
            head.keyword("while");
            head.space();
            head.punct("(");
            lower_condition(ctx, cond, roots).write_tokens(&mut head);
            head.punct(")");
            head.space();
            head.punct("{");
            out.push(head.into_line(indent, None));
            emit_stmts(ctx, program, body, indent + 1, roots, hoisted, strings, out);
            out.push(brace_line("}", indent));
        }
        Stmt::DoWhile { cond, body } => {
            let mut head = LineBuf::default();
            head.keyword("do");
            head.space();
            head.punct("{");
            out.push(head.into_line(indent, None));
            emit_stmts(ctx, program, body, indent + 1, roots, hoisted, strings, out);
            let mut tail = LineBuf::default();
            tail.punct("}");
            tail.space();
            tail.keyword("while");
            tail.space();
            tail.punct("(");
            lower_condition(ctx, cond, roots).write_tokens(&mut tail);
            tail.punct(")");
            tail.punct(";");
            out.push(tail.into_line(indent, None));
        }
        Stmt::Loop { body } => {
            let mut head = LineBuf::default();
            head.keyword("while");
            head.space();
            head.punct("(");
            head.keyword("true");
            head.punct(")");
            head.space();
            head.punct("{");
            out.push(head.into_line(indent, None));
            emit_stmts(ctx, program, body, indent + 1, roots, hoisted, strings, out);
            out.push(brace_line("}", indent));
        }
        Stmt::Break => {
            let mut buf = LineBuf::default();
            buf.keyword("break");
            buf.punct(";");
            out.push(buf.into_line(indent, None));
        }
        Stmt::Continue => {
            let mut buf = LineBuf::default();
            buf.keyword("continue");
            buf.punct(";");
            out.push(buf.into_line(indent, None));
        }
        Stmt::Switch {
            scrutinee,
            cases,
            default,
        } => {
            // Rust-style `match`: no fallthrough (so no `break`), fat-arrow arms,
            // `|`-joined labels, and a `_` catch-all.
            let mut head = LineBuf::default();
            head.keyword("match");
            head.space();
            scrutinee.write_tokens(&mut head);
            head.space();
            head.punct("{");
            out.push(head.into_line(indent, None));

            // Case labels are interpreted at the scrutinee's width, so a value
            // whose sign bit is set there prints as a signed decimal (`-1`)
            // rather than a full-width unsigned constant (`0xffffffffffffffff`).
            let width = scrutinee.value.map(|v| ValueRef::new(v, ctx).size());
            for case in cases {
                let mut arm = LineBuf::default();
                for (i, &v) in case.values.iter().enumerate() {
                    if i > 0 {
                        arm.space();
                        arm.punct("|");
                        arm.space();
                    }
                    arm.push(format_case_value(v, width), TokenKind::Number);
                }
                arm.space();
                arm.punct("=>");
                arm.space();
                arm.punct("{");
                // Map the arm's labels back to the folded-away equality tests.
                for &insn in &case.insns {
                    arm.note_insn(insn);
                }
                out.push(arm.into_line(indent + 1, None));
                emit_stmts(
                    ctx,
                    program,
                    &case.body,
                    indent + 2,
                    roots,
                    hoisted,
                    strings,
                    out,
                );
                out.push(brace_line("}", indent + 1));
            }

            // `match` must be exhaustive, so emit the catch-all even when empty.
            let mut arm = LineBuf::default();
            arm.push("_", TokenKind::Keyword);
            arm.space();
            arm.punct("=>");
            arm.space();
            arm.punct("{");
            out.push(arm.into_line(indent + 1, None));
            emit_stmts(
                ctx,
                program,
                default,
                indent + 2,
                roots,
                hoisted,
                strings,
                out,
            );
            out.push(brace_line("}", indent + 1));

            out.push(brace_line("}", indent));
        }
    }
}

/// Emit a functionalized register return as a named brace initializer.
fn emit_named_return_tuple(
    ctx: &Context,
    program: &Program,
    return_id: InstructionId,
    indent: usize,
    roots: &HashSet<InstructionId>,
    out: &mut Vec<TokenLine>,
) -> bool {
    let Mnemonic::Return(ret) = Instruction::from_id(ctx, return_id).mnemonic() else {
        return false;
    };
    let Some(value) = ret.value else {
        return false;
    };
    let ValueId::Instruction(tuple_id) = value.qualify(return_id.func) else {
        return false;
    };
    let Mnemonic::Tuple(tuple) = Instruction::from_id(ctx, tuple_id).mnemonic() else {
        return false;
    };
    let Some(function_id) = program.function else {
        return false;
    };
    let function = FunctionRef::from_id(ctx, function_id);
    let Some(registers) = function.effects().materialized() else {
        return false;
    };
    let outputs = &registers.outputs[..registers.returns];
    if outputs.len() != tuple.fields.len() {
        return false;
    }
    let visible = visible_return_indices(ctx, function_id, outputs.len());

    let mut head = LineBuf::default();
    head.keyword("return");
    head.space();
    head.punct("{");
    head.note_insn(return_id);
    out.push(head.into_line(
        indent,
        Instruction::from_id(ctx, return_id).block().map(|b| b.id),
    ));

    for (emitted_index, &index) in visible.iter().enumerate() {
        let output = outputs[index];
        let field = tuple.fields[index];
        let mut line = LineBuf::default();
        line.push(
            Varnode::from_id(ctx, output).to_string(),
            TokenKind::Variable,
        );
        line.punct(":");
        line.space();
        let mut expr = lower_expr_rooted(ctx, field.qualify(return_id.func), roots);
        let output_bits = Varnode::from_id(ctx, output).size() * 8;
        if matches!(
            expr.kind,
            ExprKind::Cast {
                bits,
                ..
            } if bits == output_bits
        ) {
            let ExprKind::Cast { expr: inner, .. } = expr.kind else {
                unreachable!("matched cast above")
            };
            expr = *inner;
        }
        expr.write_tokens(&mut line);
        if emitted_index + 1 != visible.len() {
            line.punct(",");
        }
        out.push(line.into_line(indent + 1, None));
    }

    let mut tail = LineBuf::default();
    tail.punct("}");
    tail.punct(";");
    out.push(tail.into_line(indent, None));
    true
}

/// Builds the tokens for a single root instruction rendered as a C statement.
fn statement(
    ctx: &Context,
    id: InstructionId,
    roots: &HashSet<InstructionId>,
    hoisted: &HashSet<InstructionId>,
    strings: &StringPool,
) -> LineBuf {
    let insn = Instruction::from_id(ctx, id);
    let qualify = |value: LocalValueId| value.qualify(id.func);
    let mut buf = LineBuf::default();
    match insn.mnemonic() {
        Mnemonic::Store(s) => {
            // <location> = src;  (a named varnode reads as the variable, a
            // computed address as `*ptr`).
            deref_location(ctx, qualify(s.ptr), s.size, Some(roots)).write_tokens(&mut buf);
            assign(&mut buf);
            lower_expr_rooted(ctx, qualify(s.src), roots).write_tokens(&mut buf);
            buf.punct(";");
        }
        Mnemonic::Return(r) => {
            buf.keyword("return");
            if let Some(value) = r.value {
                buf.space();
                lower_expr_rooted(ctx, qualify(value), roots).write_tokens(&mut buf);
            }
            buf.punct(";");
        }
        Mnemonic::Call(c) => {
            bind_result(ctx, id, roots, &mut buf);
            let target = c.target.real();
            let name = target
                .map(|target| FunctionRef::from_id(ctx, target).name().to_string())
                .unwrap_or_else(|| format!("minted_{}", c.target.minted().unwrap_or_default()));
            buf.push_function(name, TokenKind::Label, target);
            call_args(ctx, id.func, target, &c.args, roots, strings, &mut buf);
            buf.punct(";");
        }
        // A tail call transfers control to the callee and returns whatever it
        // returns, so it reads as a return of the call. Without this it fell to
        // the generic `opcode(args)` renderer and printed as an SSA definition of
        // a zero-width value — `uint0_t b = tailcall();` — losing the callee
        // entirely. gcc produces these when it splits a cold arm into its own
        // `.cold` function and jumps to it.
        Mnemonic::TailCall(t) => {
            buf.keyword("return");
            buf.space();
            let target = t.target.real();
            let name = target
                .map(|target| FunctionRef::from_id(ctx, target).name().to_string())
                .unwrap_or_else(|| format!("minted_{}", t.target.minted().unwrap_or_default()));
            buf.push_function(name, TokenKind::Label, target);
            call_args(ctx, id.func, target, &t.args, roots, strings, &mut buf);
            buf.punct(";");
        }
        // An indirect branch whose successors were never resolved reaches the
        // backend only when there is no goto that stands for it (see
        // `is_replaced_by_goto`). It reads as a computed goto: control leaves
        // through a pointer this function computed. In practice it is almost
        // always an indirect tail call (`jmp *%rax`), which would read more
        // naturally as `return (*ptr)();` — but the same shape is also an
        // unresolved intra-function jump table, and calling *that* a tail call
        // would be a lie. `goto *ptr;` is true of both.
        Mnemonic::BranchInd(b) => {
            buf.keyword("goto");
            buf.space();
            buf.push("*", TokenKind::Operator);
            lower_expr_rooted(ctx, qualify(b.ptr), roots).write_tokens(&mut buf);
            buf.punct(";");
        }
        Mnemonic::CallInd(c) => {
            bind_result(ctx, id, roots, &mut buf);
            buf.punct("(");
            buf.push("*", TokenKind::Operator);
            lower_expr_rooted(ctx, qualify(c.ptr), roots).write_tokens(&mut buf);
            buf.punct(")");
            // An indirect callee has no prototype to bind arguments to, so no
            // string reconstruction is possible here.
            call_args(ctx, id.func, None, &c.args, roots, strings, &mut buf);
            buf.punct(";");
        }
        // An SSA result is defined exactly once, so its assignment is also its
        // declaration: `uintN_t name = <defining expression>;`.
        _ => {
            // A hoisted value was already declared at the top of the function;
            // re-declaring here would shadow it inside this scope, which is the
            // very thing the hoist exists to avoid.
            if !hoisted.contains(&id) {
                emit_inferred_decl_type(ctx, id, &mut buf);
                buf.space();
            }
            buf.push_value(
                instruction_name(ctx, id, roots),
                TokenKind::Variable,
                Some(ValueId::Instruction(id)),
            );
            assign(&mut buf);
            lower_defining_expr(ctx, id, roots).write_tokens(&mut buf);
            buf.punct(";");
        }
    }
    buf
}

/// Name a call's result when something reads it.
///
/// A call is emitted as a statement for its effect, so the result had no name —
/// yet its `extract`s refer to it, and rendered they would name a value the
/// reader has never seen bound (or, worse, one that happens to collide with an
/// unrelated local).
///
/// The binding is declared like any other SSA definition. A write-set
/// aggregate's width is not informative on its own, but leaving the one call
/// result as the only undeclared name in a function reads as an assignment to
/// something outside it.
fn bind_result(
    ctx: &Context,
    id: InstructionId,
    roots: &HashSet<InstructionId>,
    buf: &mut LineBuf,
) {
    if ctx.users(ValueId::Instruction(id)).is_empty() {
        return;
    }
    match visible_call_result_types(ctx, id).as_deref() {
        Some([]) => return,
        Some([type_id]) => emit_c_type(ctx, *type_id, ctx.shared.types.size_of(*type_id), buf),
        Some(types) => emit_uint_type(
            types
                .iter()
                .map(|&type_id| ctx.shared.types.size_of(type_id))
                .sum(),
            buf,
        ),
        None => emit_uint_type(Instruction::from_id(ctx, id).size(), buf),
    }
    buf.space();
    buf.push_value(
        instruction_name(ctx, id, roots),
        TokenKind::Variable,
        Some(ValueId::Instruction(id)),
    );
    assign(buf);
}

/// The types of the non-clobber fields in a direct call's result pack.
///
/// External calls append ABI clobbers to their physical return pack, and bodied
/// callees can carry the same poison slots in their returns. Neither belongs in
/// a source-facing call-result type.
fn visible_call_result_types(
    ctx: &Context,
    call_id: InstructionId,
) -> Option<Vec<qcode::types::TypeId>> {
    let Mnemonic::Call(call) = Instruction::from_id(ctx, call_id).mnemonic() else {
        return None;
    };
    let target = call.target.real()?;
    let registers = FunctionRef::from_id(ctx, target).effects().materialized()?;
    let visible = visible_return_indices(ctx, target, registers.returns);
    let call_type = Instruction::from_id(ctx, call_id).type_id();
    visible
        .into_iter()
        .map(|index| ctx.shared.types.field_type(call_type, index))
        .collect()
}

/// Renders a call's parenthesized argument list.
///
/// `callee` is the resolved target, when there is one: an argument that binds to
/// a `char *` parameter of its C prototype and is a literal address into
/// read-only memory renders as the name of a reconstructed rodata object rather
/// than as the address (see [`strings`](super::strings)).
fn call_args(
    ctx: &Context,
    function: FunctionId,
    callee: Option<FunctionId>,
    args: &[LocalValueId],
    roots: &HashSet<InstructionId>,
    strings: &StringPool,
    buf: &mut LineBuf,
) {
    buf.punct("(");
    for (i, &arg) in args.iter().enumerate() {
        if i > 0 {
            buf.punct(",");
            buf.space();
        }
        let value = arg.qualify(function);
        match strings.arg_name(ctx, callee, i, value) {
            Some(name) => buf.push_value(name, TokenKind::Variable, Some(value)),
            None => lower_expr_rooted(ctx, value, roots).write_tokens(buf),
        }
    }
    buf.punct(")");
}

/// The display name of a phi-copy cycle temp. The `phi_` prefix keeps it clear
/// of the `v*`/`p_*`/varnode namespaces.
fn temp_name(n: usize) -> String {
    format!("phi_tmp{n}")
}

fn assign(buf: &mut LineBuf) {
    buf.space();
    buf.push("=", TokenKind::Operator);
    buf.space();
}

fn brace_line(text: &str, indent: usize) -> TokenLine {
    let mut buf = LineBuf::default();
    buf.punct(text);
    buf.into_line(indent, None)
}

/// Formats a `switch`/`match` case label. A constant whose sign bit is set at the
/// scrutinee's `width` (in bytes) prints as a signed decimal — `-1` rather than
/// the sign-extended `0xffffffffffffffff` — so negative case values read
/// naturally. Non-negative values (and values of unknown width) stay hex.
fn format_case_value(v: u64, width: Option<usize>) -> String {
    if let Some(w) = width
        && (1..=8).contains(&w)
    {
        let bits = w * 8;
        let fits = bits == 64 || v < (1u64 << bits);
        let negative = (v >> (bits - 1)) & 1 == 1;
        if fits && negative {
            let signed = if bits == 64 {
                v as i64
            } else {
                v as i64 - (1i64 << bits)
            };
            return format!("{signed}");
        }
    }
    format!("0x{v:x}")
}

/// The label name for `block`, falling back to a synthetic name if the program
/// somehow lacks one (should not happen for well-formed programs).
fn label_of(program: &Program, block: BlockId) -> String {
    program
        .label(block)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("bb_{}", usize::from(block.local)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::{decompile_function, lower_function, tokens::TokenKind};
    use qcode::value::{
        FunctionBody, LocalValueId, QCodeMut, RegisterChannelState, RegisterInterfaceMap,
    };
    use wazabin_qcode_macro::qcode;

    #[test]
    fn function_arguments_include_their_recovered_width() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 x;
            varnode i64 y;

            fn f:
            <entry>
                return at i64 0;
            "
        );
        FunctionBody::from_id_mut(&mut ctx, f).set_register_effects(
            RegisterChannelState::Materialized(RegisterInterfaceMap {
                inputs: vec![x, y],
                outputs: vec![],
                returns: 0,
                projections: Vec::new(),
            }),
        );

        let c = emit_c(&ctx, &lower_function(&ctx, f), None);
        assert!(
            c.starts_with("fn f(uint32_t x, uint64_t y)"),
            "function arguments should carry fixed-width C types:\n{c}"
        );
    }

    #[test]
    fn function_arguments_include_ram_snapshot_params_after_registers() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 RSP;

            fn f:
            <entry @rsp:i64 @RSP_val_0:i64>
                return at @RSP_val_0;
            "
        );
        FunctionBody::from_id_mut(&mut ctx, f).set_register_effects(
            RegisterChannelState::Materialized(RegisterInterfaceMap {
                inputs: vec![RSP],
                outputs: vec![],
                returns: 0,
                projections: Vec::new(),
            }),
        );

        let c = emit_c(&ctx, &lower_function(&ctx, f), None);
        assert!(
            c.starts_with("fn f(uint64_t RSP, uint64_t RSP_val_0)"),
            "RAM snapshot params should follow the register-input prefix:\n{c}"
        );
    }

    #[test]
    fn named_ssa_results_are_declared_with_their_width() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 x;
            varnode i32 y;

            fn f:
            <entry>
                %sum = i32 0x1 + i32 0x2;
                store(x:4, &x <- %sum);
                store(y:4, &y <- %sum);
                return at i64 0;
            "
        );

        let c = emit_c(&ctx, &lower_function(&ctx, f), None);
        assert!(
            c.contains("uint32_t sum = 0x1 + 0x2;"),
            "a named SSA root should be declared at its definition:\n{c}"
        );
    }

    #[test]
    fn shared_signed_value_keeps_signed_declaration_and_explicit_zero_extension() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 x;
            varnode i64 signed_out;
            varnode i64 unsigned_out;

            fn f:
            <entry>
                %xv = load(x:4, &x);
                %d = i32 %xv + i32 0x2;
                %signed = sext(i64, %d);
                %unsigned = zext(i64, %d);
                store(signed_out:8, &signed_out <- %signed);
                store(unsigned_out:8, &unsigned_out <- %unsigned);
                return at i64 0;
            "
        );

        let c = emit_c(&ctx, &lower_function(&ctx, f), None);
        assert!(
            c.contains("int32_t d = x + 0x2;"),
            "a shared sign-extended value should be declared signed:\n{c}"
        );
        assert!(
            c.contains("signed_out = d;"),
            "C should widen the signed declaration implicitly:\n{c}"
        );
        assert!(
            c.contains("unsigned_out = (uint64_t)(uint32_t)d;"),
            "zero-extension from a signed declaration must preserve the 32-bit pattern:\n{c}"
        );
    }

    #[test]
    fn scalar_ranges_emit_as_fixed_width_casts() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 lhs;
            varnode i64 rhs;
            varnode i32 low_out;
            varnode i32 high_out;

            fn f:
            <entry>
                %lhs = load(lhs:8, &lhs);
                %rhs = load(rhs:8, &rhs);
                %quotient = i64 %lhs s/ i64 %rhs;
                %low = %quotient[0:4];
                %high = %quotient[4:8];
                store(low_out:4, &low_out <- %low);
                store(high_out:4, &high_out <- %high);
                return at i64 0;
            "
        );

        let c = emit_c(&ctx, &lower_function(&ctx, f), None);
        assert!(
            c.contains("low_out = (int32_t)")
                && c.contains("high_out = (uint32_t)")
                && c.contains(">> 0x20"),
            "scalar ranges should lower to signed/unsigned casts and shifts:\n{c}"
        );
        assert!(
            !c.contains("range("),
            "scalar ranges must not leak into emitted C:\n{c}"
        );
    }

    #[test]
    fn poison_return_slots_are_omitted_from_decompiled_output() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 RAX;
            varnode i64 RCX;

            fn f:
            <entry>
                %answer = i64 0x2a + i64 0x0;
                %clobber = i64 0x0 + i64 0x0;
                %result = pack(RAX=%answer, RCX=%clobber);
                return %result at i64 0;
            "
        );
        let poison = ctx.get_poison(ctx.shared.types.get_or_make_int(8));
        ctx.replace_instruction_mnemonic(
            result,
            Mnemonic::Tuple(qcode::value::insn::Tuple {
                fields: vec![
                    LocalValueId::Instruction(answer.localize(f)),
                    poison.localize(f),
                ],
            }),
        );
        FunctionBody::from_id_mut(&mut ctx, f).set_register_effects(
            RegisterChannelState::Materialized(RegisterInterfaceMap {
                inputs: vec![],
                outputs: vec![RAX, RCX],
                returns: 2,
                projections: Vec::new(),
            }),
        );

        let c = emit_c(&ctx, &lower_function(&ctx, f), None);
        assert!(
            c.starts_with("fn f() -> {uint64_t}"),
            "poison must not appear in the return type:\n{c}"
        );
        assert!(
            c.contains("return {") && c.contains("RAX:"),
            "poison must not appear in the return initializer:\n{c}"
        );
        assert!(
            !c.contains("RCX:") && !c.contains("poison"),
            "decompiled output must not expose poison:\n{c}"
        );
    }

    #[test]
    fn call_result_type_omits_callee_poison_slots() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 RAX;
            varnode i64 RCX;

            fn callee:
            <callee_entry>
                %answer = i64 0x2a + i64 0x0;
                %clobber = i64 0x0 + i64 0x0;
                %result = pack(RAX=%answer, RCX=%clobber);
                return %result at i64 0;

            fn caller:
            <caller_entry>
                call fn callee();
            <caller_cont>
                return at i64 0;
            "
        );
        let poison = ctx.get_poison(ctx.shared.types.get_or_make_int(8));
        ctx.replace_instruction_mnemonic(
            result,
            Mnemonic::Tuple(qcode::value::insn::Tuple {
                fields: vec![
                    LocalValueId::Instruction(answer.localize(callee)),
                    poison.localize(callee),
                ],
            }),
        );
        FunctionBody::from_id_mut(&mut ctx, callee).set_register_effects(
            RegisterChannelState::Materialized(RegisterInterfaceMap {
                inputs: vec![],
                outputs: vec![RAX, RCX],
                returns: 2,
                projections: Vec::new(),
            }),
        );

        let caller_entry = FunctionBody::from_id(&ctx, caller).root().unwrap().id;
        let call_id = qcode::value::BasicBlock::from_id(&ctx, caller_entry)
            .iter()
            .find(|insn| matches!(insn.mnemonic(), Mnemonic::Call(_)))
            .unwrap()
            .id;
        let return_id = FunctionBody::from_id(&ctx, caller)
            .blocks()
            .flat_map(|block| block.iter())
            .find(|insn| matches!(insn.mnemonic(), Mnemonic::Return(_)))
            .unwrap()
            .id;
        let return_block = Instruction::from_id(&ctx, return_id).block().unwrap().id;
        let pack_type =
            ctx.shared
                .types
                .get_or_make_aggregate(vec![ctx.shared.types.get_or_make_int(8); 2]);
        Instruction::from_id_mut(&mut ctx, call_id).set_type(pack_type);
        let Mnemonic::Return(return_) = Instruction::from_id(&ctx, return_id).mnemonic() else {
            unreachable!("selected return is not a return")
        };
        let return_ptr = return_.ptr;
        ctx.remove_instruction(return_id);
        let rax_space = Varnode::from_id(&ctx, RAX).space().id;
        let rcx_space = Varnode::from_id(&ctx, RCX).space().id;
        let mut builder = ctx.builder(return_block);
        // This is the normal post-call clobber replay. It must stay in the IR,
        // but it must not leak into the decompiled caller.
        builder.push_store(poison, ValueId::Varnode(RCX), rcx_space);
        let answer = builder.push_extract(ValueId::Instruction(call_id), 0).id();
        builder.push_store(answer, ValueId::Varnode(RAX), rax_space);
        builder.push_return_local(return_ptr);

        let c = emit_c(&ctx, &lower_function(&ctx, caller), None);
        assert!(
            c.contains("uint64_t") && c.contains("= callee();"),
            "the call result should contain only the real return value:\n{c}"
        );
        assert!(
            !c.contains("uint128_t") && !c.contains("poison") && !c.contains("RCX ="),
            "the call result and its clobber replay must not expose poison:\n{c}"
        );
    }

    #[test]
    fn named_return_field_omits_its_destination_width_cast() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 EAX;
            varnode i64 RAX;

            fn f:
            <entry>
                %value = load(EAX:4, &EAX);
                %wide = zext(i64, %value);
                %result = pack(RAX=%wide);
                return %result at i64 0;
            "
        );
        FunctionBody::from_id_mut(&mut ctx, f).set_register_effects(
            RegisterChannelState::Materialized(RegisterInterfaceMap {
                inputs: vec![EAX],
                outputs: vec![RAX],
                returns: 1,
                projections: Vec::new(),
            }),
        );

        let c = emit_c(&ctx, &lower_function(&ctx, f), None);
        assert!(
            c.contains("RAX: EAX") && !c.contains("RAX: (uint64_t)"),
            "the typed return field should provide the widening context:\n{c}"
        );
    }

    #[test]
    fn single_use_values_fold_into_a_compound_statement() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 x;
            varnode i32 y;

            fn f:
            <entry>
                %a = load(x:4, &x);
                %b = i32 %a + i32 0x1;
                store(y:4, &y <- %b);
                return at i64 0;
            "
        );

        let program = lower_function(&ctx, f);
        let c = emit_c(&ctx, &program, None);

        // The load and the add are single-use, so they fold into the store; no
        // intermediate assignments are emitted.
        assert!(
            c.contains("y = x + 0x1;"),
            "expected folded compound store, got:\n{c}"
        );
        assert!(!c.contains("%a"), "the load should be inlined:\n{c}");
        assert!(!c.contains("%b"), "the add should be inlined:\n{c}");
    }

    #[test]
    fn condition_references_root_not_reread_memory() {
        // A load kept live across an aliasing store becomes a named root. The
        // branch that tests it must reference that name, not re-inline the
        // dereference and re-read memory after the store.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 pp;
            varnode i64 qq;
            varnode i8 flag;

            fn f:
            <entry>
                %p = load(pp:8, &pp);
                %v = load(ram:1, %p);
                %q = load(qq:8, &qq);
                store(ram:1, %q <- i8 0x0);
                if %v goto <then_lbl> else goto <merge>;
            <then_lbl>
                store(flag:1, &flag <- i8 0x1);
                goto <merge>;
            <merge>
                return at i64 0;
            "
        );
        let c = emit_c(&ctx, &decompile_function(&ctx, f).unwrap(), None);
        // The load is named (a root because of the intervening store) and the
        // condition uses that name rather than re-dereferencing memory. The
        // byte-wide load through `pp` carries its access width as a pointer cast.
        assert!(
            c.contains("v = *(uint8_t *)pp;"),
            "load should be a named, width-annotated root:\n{c}"
        );
        assert!(
            c.contains("if (v)") && !c.contains("(uint8_t *)pp)"),
            "condition should reference the root, not re-read memory:\n{c}"
        );
    }

    #[test]
    fn signed_comparison_does_not_recast_same_width_operand() {
        // The operand is already i32, so a same-width `(int32_t)` cast is noise.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 x;
            varnode i32 y;

            fn f:
            <entry>
                %a = load(x:4, &x);
                %c = i32 %a s< i32 0x5;
                if %c goto <then_lbl> else goto <merge>;
            <then_lbl>
                store(y:4, &y <- i32 0x1);
                goto <merge>;
            <merge>
                return at i64 0;
            "
        );
        let c = emit_c(&ctx, &decompile_function(&ctx, f).unwrap(), None);
        assert!(
            !c.contains("(int32_t)x") && c.contains("x < 0x5"),
            "same-width signed operand should stay bare:\n{c}"
        );
    }

    #[test]
    fn unsigned_comparison_stays_bare() {
        // The unsigned counterpart of the signed test: no cast.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 x;
            varnode i32 y;

            fn f:
            <entry>
                %a = load(x:4, &x);
                %c = i32 %a < i32 0x5;
                if %c goto <then_lbl> else goto <merge>;
            <then_lbl>
                store(y:4, &y <- i32 0x1);
                goto <merge>;
            <merge>
                return at i64 0;
            "
        );
        let c = emit_c(&ctx, &decompile_function(&ctx, f).unwrap(), None);
        assert!(
            !c.contains("(int32_t)") && c.contains("x < 0x5"),
            "unsigned compare should stay bare:\n{c}"
        );
    }

    #[test]
    fn sign_extension_extends_the_sign_not_zero() {
        // The source is already signed i32 and the i64 destination provides the
        // widening context, so C performs the sign extension implicitly.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 x;
            varnode i64 z;

            fn f:
            <entry>
                %a = load(x:4, &x);
                %w = sext(i64, %a);
                store(z:4, &z <- %w);
                return at i64 0;
            "
        );
        let c = emit_c(&ctx, &lower_function(&ctx, f), None);
        assert!(
            c.contains("z = x;") && !c.contains("(int64_t)") && !c.contains("(int32_t)x"),
            "a direct signed widening should be implicit:\n{c}"
        );
    }

    #[test]
    fn loads_do_not_fold_across_an_aliasing_store() {
        // The classic swap: `t = *x; u = *y; *x = u; *y = t`. If the single-use
        // loads folded into the stores, the output would collapse to `x = y; y =
        // x`, losing the old value of x. Each load must instead be named, because
        // an aliasing store sits between it and its use.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 x;
            varnode i32 y;

            fn f:
            <entry>
                %t = load(x:4, &x);
                %u = load(y:4, &y);
                store(x:4, &x <- %u);
                store(y:4, &y <- %t);
                return at i64 0;
            "
        );

        let program = lower_function(&ctx, f);
        let c = emit_c(&ctx, &program, None);

        // The old value of x must be saved to a temp before the store to x
        // clobbers it — that load may not fold past the intervening store.
        assert!(c.contains("t = x;"), "load of x should be named:\n{c}");
        assert!(
            c.contains("y = t;"),
            "y must be restored from the saved temp:\n{c}"
        );
        // The bug: folding the load of x into `*y = …` past the store to x, so
        // `y` reads the already-overwritten `x`.
        assert!(
            !c.contains("y = x;"),
            "load folded across the aliasing store to x:\n{c}"
        );
    }

    #[test]
    fn partial_width_access_carries_its_size() {
        // A byte load and a word store through computed pointers must render their
        // access width, so a partial access is not mistaken for a full-width one.
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 pp;
            varnode i64 qq;

            fn f:
            <entry>
                %p = load(pp:8, &pp);
                %b = load(ram:1, %p);
                %q = load(qq:8, &qq);
                store(ram:4, %q <- i32 0x0);
                store(ram:1, %p <- %b);
                return at i64 0;
            "
        );

        let program = lower_function(&ctx, f);
        let c = emit_c(&ctx, &program, None);
        // The consistently byte-accessed named pointer carries its width in its
        // declaration, so its uses need no repeated cast. The anonymous `qq`
        // address still needs an explicit word-width cast.
        assert!(
            c.contains("uint8_t * p")
                && c.contains("*p")
                && !c.contains("*(uint8_t *)p")
                && c.contains("*(uint32_t *)qq"),
            "loads/stores should carry their access width without redundant casts:\n{c}"
        );
    }

    #[test]
    fn folded_expression_preserves_required_parentheses() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 a;

            fn f:
            <entry>
                %sum = i32 0x1 + i32 0x2;
                %scaled = i32 %sum * i32 0x5;
                store(a:4, &a <- %scaled);
                return at i64 0;
            "
        );

        let program = lower_function(&ctx, f);
        let c = emit_c(&ctx, &program, None);

        assert!(
            c.contains("a = (0x1 + 0x2) * 0x5;"),
            "expected parenthesized folded expression, got:\n{c}"
        );
        assert!(
            !c.contains("%sum") && !c.contains("%scaled"),
            "single-use intermediates should be folded away:\n{c}"
        );
    }

    #[test]
    fn structured_tokens_carry_nested_indentation() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;
            varnode i32 x;

            fn f:
            <entry>
                %c = load(cond:1, &cond);
                if %c goto <then_lbl> else goto <merge>;
            <then_lbl>
                store(x:4, &x <- i32 0x1);
                goto <merge>;
            <merge>
                return at i64 0;
            "
        );

        let program = decompile_function(&ctx, f).unwrap();
        let lines = emit_tokens(&ctx, &program, None);

        // Depth 2: one level for the `fn` header, one for the enclosing `if`.
        assert!(
            lines
                .iter()
                .any(|line| line.indent == 2 && line.text() == "x = 0x1;"),
            "expected nested store to carry indent depth 2, got:\n{lines:#?}"
        );
    }

    #[test]
    fn flat_labels_indent_their_block_body() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;

            fn f:
            <entry>
                %c = load(cond:1, &cond);
                if %c goto <body> else goto <exit_lbl>;
            <body>
                goto <entry>;
            <exit_lbl>
                return at i64 0;
            "
        );

        // The flat lowering always labels every block and indents its body; test
        // it directly rather than relying on a structuring fallback.
        let program = lower_function(&ctx, f);
        let lines = emit_tokens(&ctx, &program, None);

        // Everything sits one level in from the `fn` header: the label at depth
        // 1, its block body at depth 2.
        assert!(
            lines
                .iter()
                .any(|line| line.indent == 1 && line.text() == "entry:"),
            "expected label one level under the fn header, got:\n{lines:#?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.indent == 2 && line.text().starts_with("if (cond) goto ")),
            "expected flat block body to be indented under its label, got:\n{lines:#?}"
        );
    }

    #[test]
    fn folded_statement_maps_back_to_its_source_instructions() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 x;
            varnode i32 y;

            fn f:
            <entry>
                %a = load(x:4, &x);
                %b = i32 %a + i32 0x1;
                store(y:4, &y <- %b);
                return at i64 0;
            "
        );

        let program = lower_function(&ctx, f);
        let lines = emit_tokens(&ctx, &program, None);

        // The single compound store line must carry all three folded-in
        // instructions (the load, the add, and the store itself).
        let store_line = lines
            .iter()
            .find(|l| l.text() == "y = x + 0x1;")
            .expect("expected the folded store line");
        let opcodes: Vec<String> = store_line
            .insns
            .iter()
            .map(|&id| Instruction::from_id(&ctx, id).opcode().to_string())
            .collect();
        assert_eq!(
            store_line.insns.len(),
            3,
            "load + add + store should all map to this line, got {opcodes:?}"
        );

        // A value token carries the IR value it renders, so a front-end can
        // highlight its uses; plain punctuation does not.
        let x_token = store_line
            .tokens
            .iter()
            .find(|t| t.text == "x")
            .expect("expected the `x` variable token");
        assert!(x_token.value.is_some(), "value tokens carry provenance");
        let eq_token = store_line.tokens.iter().find(|t| t.text == "=").unwrap();
        assert!(eq_token.value.is_none(), "punctuation carries no value");
    }

    #[test]
    fn condition_line_maps_back_to_its_comparison_and_load() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 x;
            varnode i32 y;

            fn f:
            <entry>
                %v = load(x:4, &x);
                %c = i32 %v == i32 0x5;
                if %c goto <then_lbl> else goto <merge>;
            <then_lbl>
                store(y:4, &y <- i32 0x1);
                goto <merge>;
            <merge>
                return at i64 0;
            "
        );

        let program = decompile_function(&ctx, f).unwrap();
        let lines = emit_tokens(&ctx, &program, None);

        // The `if (x == 0x5)` header folds in the comparison and the load, both
        // of which must be recoverable from the line's instruction set.
        let cond_line = lines
            .iter()
            .find(|l| l.text().starts_with("if (") && l.text().contains("=="))
            .expect("expected the folded if header");
        assert!(
            cond_line.insns.len() >= 2,
            "the compare and the load should both map to the condition, got {:?}",
            cond_line
                .insns
                .iter()
                .map(|&id| Instruction::from_id(&ctx, id).opcode().to_string())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn tokens_classify_keywords_numbers_and_operators() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 y;

            fn f:
            <entry>
                store(y:4, &y <- i32 0x2a);
                return at i64 0;
            "
        );

        let program = lower_function(&ctx, f);
        let lines = emit_tokens(&ctx, &program, None);
        let kinds: Vec<TokenKind> = lines
            .iter()
            .flat_map(|l| l.tokens.iter().map(|t| t.kind))
            .collect();
        assert!(kinds.contains(&TokenKind::Keyword), "return is a keyword");
        assert!(kinds.contains(&TokenKind::Number), "0x2a is a number");
        assert!(
            kinds.contains(&TokenKind::Operator),
            "= and * are operators"
        );
    }
}
