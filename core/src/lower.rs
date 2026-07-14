//! Runtime lowering of parsed QCode text into qcode IR.
//!
//! The `qcode!` proc-macro parses QCode and *emits Rust code* that builds the IR
//! at the call site — it cannot run at runtime. This module is the runtime twin:
//! it walks the same [`qcode_parser`] AST and drives the [`Builder`] directly, so
//! tools (a CLI, the macro itself) share a single lowering implementation.
//!
//! It lives in `qcode` core (not a separate crate) so the `qcode!` macro can
//! route through it from core's own tests without creating a second copy of
//! `qcode` via a dev-dependency cycle.
//!
//! [`lower_str`] returns a [`Symbols`] table mapping each source name to the id
//! it produced, so callers can look declarations back up after lowering.

use std::borrow::Cow;
use std::collections::HashMap;

use crate::{
    address_index::{AddressIndex, AddressTarget},
    builder::Builder,
    context::Context,
    types::AggregateField,
    value::{
        BasicBlock, BlockParam, BlockParamId, FunctionBody, FunctionId, Instruction, InstructionId,
        Renameable, TempId, TempRef, Value, ValueId, ValueRef, Varnode, VarnodeId,
        block::BlockId,
        insn::{Callee, IntrinsicId},
    },
};
use qcode_parser::ast::{
    Atom, Callee as ParsedCallee, CastOp, ExprNode, ExtractField, FnDecl, FnKind, GepField, Label,
    Program, ProgramKind, Statement, StructDecl, StructFieldType, TypedAtom,
};

/// Names produced while lowering a program, so callers can recover the ids by the
/// identifier that appeared in the source. Block/param/ssa/varnode names are
/// last-write-wins across functions, matching the macro's shared-name behavior.
#[derive(Debug, Default, Clone)]
pub struct Symbols {
    pub functions: HashMap<String, FunctionId>,
    pub blocks: HashMap<String, BlockId>,
    pub ssa: HashMap<String, InstructionId>,
    pub varnodes: HashMap<String, VarnodeId>,
    pub temps: HashMap<String, TempId>,
    pub block_params: HashMap<String, BlockParamId>,
}

impl Symbols {
    pub fn function(&self, name: &str) -> FunctionId {
        self.functions[name]
    }
    pub fn block(&self, name: &str) -> BlockId {
        self.blocks[name]
    }
    pub fn ssa(&self, name: &str) -> InstructionId {
        self.ssa[name]
    }
    pub fn varnode(&self, name: &str) -> VarnodeId {
        self.varnodes[name]
    }
    pub fn temp(&self, name: &str) -> TempId {
        self.temps[name]
    }
    pub fn block_param(&self, name: &str) -> BlockParamId {
        self.block_params[name]
    }
}

/// Parse and lower QCode source text into `ctx`.
pub fn lower_str(ctx: &mut Context, source: &str) -> Result<Symbols, String> {
    lower_str_with_externals(ctx, source, HashMap::new())
}

/// Like [`lower_str`], but `externals` supplies values for `{name}` capture atoms
/// — used by the `qcode!` macro to inject in-scope Rust values into the program.
pub fn lower_str_with_externals(
    ctx: &mut Context,
    source: &str,
    externals: HashMap<String, ValueId>,
) -> Result<Symbols, String> {
    let program = qcode_parser::qcode_from_str(source).map_err(|e| e.to_string())?;
    lower_program_with_externals(ctx, &program, externals)
}

/// Lower an already-parsed program into `ctx`.
pub fn lower_program(ctx: &mut Context, program: &Program) -> Result<Symbols, String> {
    lower_program_with_externals(ctx, program, HashMap::new())
}

/// Lower a parsed program, resolving `{name}` capture atoms via `externals`.
pub fn lower_program_with_externals(
    ctx: &mut Context,
    program: &Program,
    externals: HashMap<String, ValueId>,
) -> Result<Symbols, String> {
    let mut symbols = Symbols::default();
    register_structs(ctx, &program.structs);
    let mut addresses = AddressIndex::analyze(ctx);

    match &program.kind {
        ProgramKind::Statements(statements) => {
            let mut locals = HashMap::new();
            lower_statement_block(
                ctx,
                &mut addresses,
                statements,
                &mut locals,
                &mut symbols,
                &externals,
            )?;
        }
        ProgramKind::Functions { varnodes, fns } => {
            let mut globals = HashMap::new();
            for stmt in varnodes {
                let Statement::LocalDecl {
                    name, size_bytes, ..
                } = stmt.inner()
                else {
                    return Err("top-level list may only contain varnode declarations".into());
                };
                let id = make_global_varnode(ctx, name, *size_bytes);
                globals.insert(name.clone(), Local::Varnode(id));
                symbols.varnodes.insert(name.clone(), id);
            }

            // Create every function shell first so `apply`/`map` can resolve
            // sibling functions regardless of declaration order.
            for fn_decl in fns {
                let fid = make_function(ctx, fn_decl)?;
                symbols.functions.insert(fn_decl.name.clone(), fid);
            }
            for fn_decl in fns {
                let fid = symbols.functions[&fn_decl.name];
                lower_fn_body(
                    ctx,
                    &mut addresses,
                    fid,
                    fn_decl,
                    &globals,
                    &mut symbols,
                    &externals,
                )?;
            }
        }
    }
    Ok(symbols)
}

#[derive(Clone, Copy)]
enum Local {
    Varnode(VarnodeId),
    Temp(TempId),
    Instruction(InstructionId),
    BlockParam(BlockParamId),
}

impl Local {
    fn value_id(self) -> ValueId {
        match self {
            Local::Varnode(id) => id.into(),
            Local::Temp(id) => id.into(),
            Local::Instruction(id) => id.into(),
            Local::BlockParam(id) => id.into(),
        }
    }
}

fn register_structs(ctx: &mut Context, structs: &[StructDecl]) {
    for s in structs {
        let mut offset = 0usize;
        let mut fields = Vec::new();
        for field in &s.fields {
            let (size, ty) = match &field.ty {
                StructFieldType::Int(n) => (*n, ctx.shared.types.get_or_make_int(*n)),
                StructFieldType::StructPtr(target) => {
                    let pointee = ctx.shared.types.get_or_make_struct(target, 0, Vec::new());
                    (
                        8usize,
                        ctx.shared.types.get_or_make_struct_pointer(8, pointee),
                    )
                }
            };
            if !field.is_padding() {
                fields.push(AggregateField::new_at(&field.name, ty, offset));
            }
            offset += size;
        }
        ctx.shared.types.get_or_make_struct(&s.name, offset, fields);
    }
}

fn make_global_varnode(ctx: &mut Context, name: &str, size: usize) -> VarnodeId {
    let unique = ctx.get_unique_name(Cow::Owned(name.to_owned()));
    let space = ctx.get_or_make_named_space(unique.as_ref());
    let id = Varnode::make(ctx, 0, size, space).id;
    let _ = Varnode::from_id_mut(ctx, id).rename(unique);
    id
}

fn make_function(ctx: &mut Context, fn_decl: &FnDecl) -> Result<FunctionId, String> {
    let name = Cow::Owned(fn_decl.name.clone());
    let f = match fn_decl.kind {
        FnKind::Machine => FunctionBody::make(ctx, name),
        FnKind::Lambda => FunctionBody::make_lambda(ctx, name),
    };
    f.map(|r| r.id).map_err(|e| e.to_string())
}

/// Collects, for every named label, its declared block-parameter names in order.
fn collect_block_param_names(statements: &[Statement]) -> HashMap<String, Vec<String>> {
    let mut out = HashMap::new();
    for s in statements {
        if let Statement::LabelDecl {
            label: Label::Named { name, params, .. },
            ..
        } = s.inner()
        {
            out.insert(
                name.clone(),
                params.iter().map(|p| p.name.clone()).collect(),
            );
        }
    }
    out
}

fn lower_fn_body(
    ctx: &mut Context,
    addresses: &mut AddressIndex,
    fid: FunctionId,
    fn_decl: &FnDecl,
    globals: &HashMap<String, Local>,
    symbols: &mut Symbols,
    externals: &HashMap<String, ValueId>,
) -> Result<(), String> {
    let statements = &fn_decl.statements;
    let first = statements
        .first()
        .ok_or_else(|| format!("fn `{}`: body cannot be empty", fn_decl.name))?;
    let Statement::LabelDecl {
        label: Label::Named { name: entry, .. },
        ..
    } = first.inner()
    else {
        return Err(format!(
            "fn `{}`: first statement must be a named label like `<entry>`",
            fn_decl.name
        ));
    };
    let entry = entry.clone();

    // Create entry + every other named block and its params before building.
    let entry_id = BasicBlock::make(ctx, fid)
        .with_name(Cow::Owned(entry.clone()))
        .map_err(|e| e.to_string())?
        .id;
    FunctionBody::from_id_mut(ctx, fid)
        .set_root(entry_id)
        .map_err(|e| e.to_string())?;
    symbols.blocks.insert(entry.clone(), entry_id);

    let mut block_ids: HashMap<String, BlockId> = HashMap::new();
    block_ids.insert(entry.clone(), entry_id);
    create_blocks_and_params(ctx, statements, Some(fid), &entry, &mut block_ids, symbols);

    lower_body(
        ctx, addresses, statements, &entry, &block_ids, globals, symbols, externals,
    )
}

/// Statement-mode program: no enclosing function; the first label is the entry.
fn lower_statement_block(
    ctx: &mut Context,
    addresses: &mut AddressIndex,
    statements: &[Statement],
    globals: &mut HashMap<String, Local>,
    symbols: &mut Symbols,
    externals: &HashMap<String, ValueId>,
) -> Result<(), String> {
    // Leading varnode declarations form a preamble before the first label.
    let split = statements
        .iter()
        .position(|s| !matches!(s.inner(), Statement::LocalDecl { .. }))
        .unwrap_or(statements.len());
    let (preamble, body) = statements.split_at(split);
    for stmt in preamble {
        let Statement::LocalDecl {
            name, size_bytes, ..
        } = stmt.inner()
        else {
            unreachable!()
        };
        let id = make_global_varnode(ctx, name, *size_bytes);
        globals.insert(name.clone(), Local::Varnode(id));
        symbols.varnodes.insert(name.clone(), id);
    }
    if body.is_empty() {
        return Err("qcode program cannot be empty".into());
    }
    let Statement::LabelDecl {
        label: Label::Named { name: entry, .. },
        ..
    } = body[0].inner()
    else {
        return Err("statement program must start with a named label, e.g. `<block>`".into());
    };
    let entry = entry.clone();

    let mut block_ids: HashMap<String, BlockId> = HashMap::new();
    // A bare-block program (no `fn`) still forms one CFG, so all its blocks must
    // live in a single function; mint one anonymous host up front.
    let host = ctx.anon_function();
    create_blocks_and_params(ctx, body, Some(host), "", &mut block_ids, symbols);

    lower_body(
        ctx, addresses, body, &entry, &block_ids, globals, symbols, externals,
    )
}

/// Pre-creates all named blocks (except `skip_entry`, already made) and their
/// params. `func` attaches the blocks to a function in function-mode.
fn create_blocks_and_params(
    ctx: &mut Context,
    statements: &[Statement],
    func: Option<FunctionId>,
    skip_entry: &str,
    block_ids: &mut HashMap<String, BlockId>,
    symbols: &mut Symbols,
) {
    for stmt in statements {
        let Statement::LabelDecl {
            label: Label::Named { name, params, .. },
            ..
        } = stmt.inner()
        else {
            continue;
        };
        let block_id = if name == skip_entry {
            block_ids[name]
        } else {
            // A block must be born into a function's arena. Function-mode always
            // supplies one; the bare-block DSL path has none, so mint an
            // anonymous host function for the block to live in.
            let fid = func.unwrap_or_else(|| ctx.anon_function());
            let id = BasicBlock::make(ctx, fid)
                .with_name(Cow::Owned(name.clone()))
                .expect("qcode: block name conflict")
                .id;
            block_ids.insert(name.clone(), id);
            symbols.blocks.insert(name.clone(), id);
            id
        };
        for param in params {
            let pid = BasicBlock::from_id_mut(ctx, block_id)
                .push_param(param.size_bytes.unwrap_or(0))
                .id;
            let _ = BlockParam::from_id_mut(ctx, pid).rename(Cow::Owned(param.name.clone()));
            symbols.block_params.insert(param.name.clone(), pid);
        }
    }
}

#[allow(clippy::too_many_arguments)] // Explicit construction index stays operation-scoped.
fn lower_body(
    ctx: &mut Context,
    addresses: &mut AddressIndex,
    statements: &[Statement],
    entry: &str,
    block_ids: &HashMap<String, BlockId>,
    globals: &HashMap<String, Local>,
    symbols: &mut Symbols,
    externals: &HashMap<String, ValueId>,
) -> Result<(), String> {
    let block_param_names = collect_block_param_names(statements);
    let mut locals = globals.clone();
    // Seed the entry block's params, then build on the entry block.
    if let Some(Statement::LabelDecl {
        label: Label::Named { params, .. },
        ..
    }) = statements.first().map(|s| s.inner())
    {
        for param in params {
            locals.insert(
                param.name.clone(),
                Local::BlockParam(symbols.block_params[&param.name]),
            );
        }
    }

    let entry_id = block_ids[entry];
    let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, entry_id));
    let mut lw = Lowerer {
        b: &mut b,
        addresses,
        locals: &mut locals,
        block_ids,
        block_param_names: &block_param_names,
        symbols,
        externals,
    };
    for stmt in statements.iter().skip(1) {
        lw.statement(stmt.inner())?;
    }
    Ok(())
}

struct Lowerer<'a, 'str, 'ctx> {
    b: &'a mut Builder<'str, 'ctx>,
    addresses: &'a mut AddressIndex,
    locals: &'a mut HashMap<String, Local>,
    block_ids: &'a HashMap<String, BlockId>,
    block_param_names: &'a HashMap<String, Vec<String>>,
    symbols: &'a mut Symbols,
    externals: &'a HashMap<String, ValueId>,
}

impl Lowerer<'_, '_, '_> {
    fn block(&mut self, label: &Label) -> Result<BlockId, String> {
        let resolved = match label {
            Label::Named { name, .. } => self
                .block_ids
                .get(name)
                .copied()
                .ok_or_else(|| format!("unknown block <{name}>"))?,
            Label::Address { value, .. } => {
                let current = self.b.current_block().func;
                let foreign = match self.addresses.get(*value) {
                    Some(AddressTarget::Function(owner)) if owner != current => Some(owner),
                    Some(AddressTarget::Block(block)) if block.func != current => Some(block.func),
                    _ => None,
                };
                if let Some(owner) = foreign {
                    return Err(format!(
                        "control-flow target {label:?} resolves to storage owned by {owner:?}, \
                         but the branch is in {current:?}; cross-function control flow must be a \
                         call/tail call, not a foreign block target",
                    ));
                }
                self.b.get_or_make_block_indexed(self.addresses, *value)
            }
        };
        // Strict IR locality (context-split ruling 2): a control-flow target must
        // be a block of the *current* function. Named labels resolve through the
        // function-local `block_ids` map, so they can never be foreign; the only
        // way textual qcode can name another function's block is an address label
        // (`goto <0xADDR>` or a `// -> <0xADDR>` edge hint) that the global address
        // map already owns for a different function. Reject it — cross-function
        // control flow is a `call` / tail call, never a foreign block target.
        let current = self.b.current_block().func;
        let owner = BasicBlock::from_id(self.b.context(), resolved)
            .parent()
            .map(|f| f.id);
        if let Some(owner) = owner
            && owner != current
        {
            return Err(format!(
                "control-flow target {label:?} resolves to a block owned by {owner:?}, \
                 but the branch is in {current:?}; cross-function control flow must be a \
                 call/tail call, not a foreign block target",
            ));
        }
        Ok(resolved)
    }

    /// Add CFG edges from the just-terminated current block to each label in a
    /// terminator's `// -> ...` edge hint. Used for terminators whose own syntax
    /// encodes no successors: a `call`'s return block, or an indirect `goto`'s
    /// resolved targets.
    fn add_edge_hints(&mut self, targets: &[Label]) -> Result<(), String> {
        if targets.is_empty() {
            return Ok(());
        }
        let from = self.b.current_block();
        for target in targets {
            let to = self.block(target)?;
            self.b.context_mut().add_cfg_edge(from, to);
        }
        Ok(())
    }

    fn statement(&mut self, stmt: &Statement) -> Result<(), String> {
        match stmt {
            Statement::LocalDecl {
                name, size_bytes, ..
            } => {
                let id = self
                    .b
                    .make_named_temp(Cow::Owned(name.clone()), *size_bytes);
                self.locals.insert(name.clone(), Local::Temp(id));
                self.symbols.temps.insert(name.clone(), id);
            }

            Statement::Assign {
                name,
                expr,
                decl_struct_ptr,
                ..
            } => {
                let value = self.expr(expr)?;
                let ValueId::Instruction(id) = value else {
                    return Err(format!("`%{name}` must be bound to an instruction result"));
                };
                let _ = Instruction::from_id_mut(self.b.context_mut(), id)
                    .rename(Cow::Owned(name.clone()));
                if let Some(struct_name) = decl_struct_ptr {
                    let pointee = self.b.context_mut().shared.types.get_or_make_struct(
                        struct_name,
                        0,
                        Vec::new(),
                    );
                    let sp = self
                        .b
                        .context_mut()
                        .shared
                        .types
                        .get_or_make_struct_pointer(8, pointee);
                    Instruction::from_id_mut(self.b.context_mut(), id).set_type(sp);
                }
                self.locals.insert(name.clone(), Local::Instruction(id));
                self.symbols.ssa.insert(name.clone(), id);
            }

            Statement::Expr(expr) => {
                self.expr(expr)?;
            }

            Statement::LabelDecl {
                label: Label::Named { name, params, .. },
                ..
            } => {
                let id = self.block_ids[name];
                self.b.switch_to_block(id);
                for param in params {
                    self.locals.insert(
                        param.name.clone(),
                        Local::BlockParam(self.symbols.block_params[&param.name]),
                    );
                }
            }
            Statement::LabelDecl {
                label: Label::Address { value, .. },
                ..
            } => {
                let blk = self.b.get_or_make_block_indexed(self.addresses, *value);
                self.b.switch_to_block(blk);
            }

            Statement::Branch { target, args, .. } => {
                if args.is_empty() {
                    let t = self.block(target)?;
                    self.b.push_branch(t);
                } else {
                    let t = self.block(target)?;
                    let argv = self.branch_args(target, args)?;
                    self.b.push_branch_with_args(t, argv);
                }
            }

            Statement::BranchInd { ptr, targets, .. } => {
                let p = self.ptr_atom(ptr)?;
                self.b.push_branchind(p);
                self.add_edge_hints(targets)?;
            }

            Statement::CBranch {
                condition,
                target,
                target_args,
                fallthrough,
                fallthrough_args,
                ..
            } => {
                let cond = self.atom(condition, None)?;
                let t = self.block(target)?;
                let f = self.block(fallthrough)?;
                let ta = self.branch_args(target, target_args)?;
                let fa = self.branch_args(fallthrough, fallthrough_args)?;
                self.b.push_cbranch_with_args(cond, t, ta, f, fa);
                self.b.switch_to_block(f);
            }

            Statement::Call {
                target,
                tail,
                args,
                targets,
                ..
            } => {
                let t = self.call_callee(target);
                // Arg names are decorative (the callee's parameter names as
                // printed); only the positional atoms are bound.
                let argv = args
                    .iter()
                    .map(|(_, atom)| self.atom(atom, None))
                    .collect::<Result<Vec<_>, _>>()?;
                if *tail {
                    self.b.push_tail_call_with_args(t, argv);
                } else {
                    self.b.push_call_with_args(t, argv);
                    self.add_edge_hints(targets)?;
                }
            }

            Statement::CallInd {
                ptr, args, targets, ..
            } => {
                let p = self.ptr_atom(ptr)?;
                let argv = self.atoms(args)?;
                self.b.push_call_ind_with_args(p, argv);
                self.add_edge_hints(targets)?;
            }

            Statement::Return { ptr, value, .. } => {
                let p = self.ptr_atom(ptr)?;
                if let Some(value) = value {
                    let v = self.atom(value, None)?;
                    self.b.push_return_with_value(v, p);
                } else {
                    self.b.push_return(p);
                }
            }

            Statement::ReturnValue { value, .. } => {
                let v = self.atom(value, None)?;
                self.b.push_return_value(v);
            }

            Statement::Assert { condition, .. } => {
                let c = self.atom(condition, None)?;
                self.b.push_assert(c);
            }

            Statement::Commented { inner, .. } => self.statement(inner)?,
        }
        Ok(())
    }

    fn branch_args(
        &mut self,
        target: &Label,
        args: &[(String, TypedAtom)],
    ) -> Result<Vec<ValueId>, String> {
        if args.is_empty() {
            return Ok(Vec::new());
        }
        let Label::Named { name, .. } = target else {
            return Err("block arguments can only be passed to named labels".into());
        };
        let params = self
            .block_param_names
            .get(name)
            .ok_or_else(|| format!("unknown branch target <{name}>"))?
            .clone();
        if args.len() != params.len() {
            return Err(format!(
                "branch to <{name}> passes {} args but target declares {} params",
                args.len(),
                params.len()
            ));
        }
        let mut out = Vec::with_capacity(params.len());
        for param_name in &params {
            let value = args
                .iter()
                .find(|(arg, _)| arg == param_name)
                .map(|(_, v)| v)
                .ok_or_else(|| format!("branch to <{name}> missing argument @{param_name}"))?;
            let v = self.atom(value, None)?;
            // Constrain the destination param's width to the argument's, matching
            // the macro's size propagation across block edges.
            let size = ValueRef::new(v, self.b.context()).size();
            let pid = self.symbols.block_params[param_name];
            BlockParam::from_id_mut(self.b.context_mut(), pid).constrain_size(size);
            out.push(v);
        }
        Ok(out)
    }

    fn expr(&mut self, expr: &ExprNode) -> Result<ValueId, String> {
        match expr {
            ExprNode::Atom(atom) => self.atom(atom, None),

            ExprNode::Unop { op, src } => {
                let s = self.atom(src, None)?;
                Ok(match op.as_str() {
                    "~" => self.b.push_bit_negate(s).id(),
                    "-" => self.b.push_neg(s).id(),
                    "f-" => self.b.push_fneg(s).id(),
                    "abs" => self.b.push_abs(s).id(),
                    "sqrt" => self.b.push_sqrt(s).id(),
                    "floor" => self.b.push_floor(s).id(),
                    "ceil" => self.b.push_ceil(s).id(),
                    "round" => self.b.push_round(s).id(),
                    _ => return Err(format!("unsupported unary operator: {op}")),
                })
            }

            ExprNode::Binary { lhs, op, rhs } => {
                let lhs_hint = self.size_hint(lhs);
                let rhs_hint = self.size_hint(rhs);
                let l = self.atom(lhs, rhs_hint)?;
                let r = self.atom(rhs, lhs_hint)?;
                Ok(self.binop(op, l, r)?)
            }

            ExprNode::Cast {
                op,
                size_bytes,
                src,
            } => {
                let s = self.atom(src, None)?;
                let n = *size_bytes;
                Ok(match op {
                    CastOp::Zext => self.b.push_zext(s, n).id(),
                    CastOp::Sext => self.b.push_sext(s, n).id(),
                    CastOp::IntToFloat => self.b.push_int_to_float(s, n).id(),
                    CastOp::FloatToFloat => self.b.push_float_to_float(s, n).id(),
                    CastOp::Trunc => self.b.push_trunc(s, n).id(),
                })
            }

            ExprNode::Load {
                space,
                size_bytes,
                ptr,
            } => {
                let p = self.ptr_atom(ptr)?;
                let space = if let Some(name) = space.strip_prefix('$') {
                    self.b.get_or_make_local_temp_space(name)
                } else {
                    self.b.context_mut().get_or_make_named_space(space).into()
                };
                Ok(self.b.push_load::<false>(p, *size_bytes, space).id())
            }

            ExprNode::Store {
                space,
                size_bytes,
                ptr,
                src,
            } => {
                let p = self.ptr_atom(ptr)?;
                let s = self.atom(src, Some(*size_bytes))?;
                let space = if let Some(name) = space.strip_prefix('$') {
                    self.b.get_or_make_local_temp_space(name)
                } else {
                    self.b.context_mut().get_or_make_named_space(space).into()
                };
                Ok(self.b.push_store(s, p, space).id())
            }

            ExprNode::FuncCall { op, args } => self.func_call(op, args),

            ExprNode::Intrinsic { name, args } => {
                let id = IntrinsicId::from_name(name)
                    .ok_or_else(|| format!("unknown intrinsic `{name}`"))?;
                let argv = self.atoms(args)?;
                Ok(self.b.push_intrinsic(id, argv).id())
            }

            ExprNode::Apply { target, args } => {
                let target = self.existing_callee(target, "apply")?;
                let argv = self.atoms(args)?;
                Ok(self.b.push_apply(target, argv).id())
            }

            ExprNode::Map {
                body,
                src,
                captures,
            } => {
                let body = self.existing_callee(body, "map")?;
                let s = self.atom(src, None)?;
                let caps = self.atoms(captures)?;
                Ok(self.b.push_map(body, s, caps).id())
            }

            ExprNode::Scan {
                body,
                init,
                src,
                captures,
            } => {
                let body = self.existing_callee(body, "scan")?;
                let i = self.atom(init, None)?;
                let s = self.atom(src, None)?;
                let caps = self.atoms(captures)?;
                Ok(self.b.push_scan(body, i, s, caps).id())
            }

            ExprNode::Tuple { fields } => {
                let mut named = Vec::with_capacity(fields.len());
                for (i, f) in fields.iter().enumerate() {
                    let name = f.name.clone().unwrap_or_else(|| format!("field{}", i + 1));
                    let v = self.atom(&f.value, None)?;
                    named.push((name, v));
                }
                Ok(self.b.push_named_tuple(named).id())
            }

            ExprNode::Extract { agg, field } => {
                let a = self.atom(agg, None)?;
                let index = match field {
                    ExtractField::Index(i) => *i as usize,
                    ExtractField::Name(name) => {
                        let ty = self
                            .b
                            .context()
                            .stored_type_of(a)
                            .ok_or("extract: aggregate has no stored type")?;
                        self.b
                            .context()
                            .shared
                            .types
                            .field_index(ty, name)
                            .ok_or_else(|| format!("extract: no field `{name}`"))?
                    }
                };
                Ok(self.b.push_extract(a, index).id())
            }

            ExprNode::Gep { base, field } => {
                let base_v = self.atom(base, None)?;
                Ok(match field {
                    GepField::Offset(off) => self.b.push_gep(base_v, *off as usize).id(),
                    GepField::Name(name) => self.b.push_gep_field(base_v, name).id(),
                })
            }

            ExprNode::Range { src, start, end } => {
                let s = self.atom(src, None)?;
                let src_size = ValueRef::new(s, self.b.context()).size();
                let start = start.map(|v| v as usize).unwrap_or(0);
                let end = end.map(|v| v as usize).unwrap_or(src_size);
                Ok(self.b.push_range(s, start, end - start).id())
            }
        }
    }

    fn binop(&mut self, op: &str, l: ValueId, r: ValueId) -> Result<ValueId, String> {
        Ok(match op {
            "+" => self.b.push_add(l, r).id(),
            "-" => self.b.push_sub(l, r).id(),
            "*" => self.b.push_mul(l, r).id(),
            "/" => self.b.push_div(l, r).id(),
            "&" => self.b.push_bit_and(l, r).id(),
            "|" => self.b.push_bit_or(l, r).id(),
            "^" => self.b.push_bit_xor(l, r).id(),
            "<<" => self.b.push_shl(l, r).id(),
            ">>" => self.b.push_shr(l, r).id(),
            "s>>" => self.b.push_sshr(l, r).id(),
            "==" => self.b.push_eq(l, r).id(),
            "!=" => self.b.push_ne(l, r).id(),
            "<" => self.b.push_lt(l, r).id(),
            "<=" => self.b.push_le(l, r).id(),
            ">" => self.b.push_gt(l, r).id(),
            ">=" => self.b.push_ge(l, r).id(),
            "s<" => self.b.push_slt(l, r).id(),
            "s<=" => self.b.push_sle(l, r).id(),
            "s>" => self.b.push_sgt(l, r).id(),
            "s>=" => self.b.push_sge(l, r).id(),
            "%" => self.b.push_mod(l, r).id(),
            "s/" => self.b.push_sdiv(l, r).id(),
            "s%" => self.b.push_smod(l, r).id(),
            "f+" => self.b.push_fadd(l, r).id(),
            "f-" => self.b.push_fsub(l, r).id(),
            "f*" => self.b.push_fmul(l, r).id(),
            "f/" => self.b.push_fdiv(l, r).id(),
            "f==" => self.b.push_feq(l, r).id(),
            "f!=" => self.b.push_fne(l, r).id(),
            "f<" => self.b.push_flt(l, r).id(),
            "f<=" => self.b.push_fle(l, r).id(),
            "f>" => self.b.push_fgt(l, r).id(),
            "f>=" => self.b.push_fge(l, r).id(),
            _ => return Err(format!("unsupported operator: {op}")),
        })
    }

    fn func_call(&mut self, op: &str, args: &[TypedAtom]) -> Result<ValueId, String> {
        let expect = |n: usize| -> Result<(), String> {
            if args.len() == n {
                Ok(())
            } else {
                Err(format!("{op} expects {n} argument(s)"))
            }
        };
        Ok(match op {
            "nan" => {
                expect(1)?;
                let s = self.atom(&args[0], None)?;
                self.b.push_is_nan(s).id()
            }
            "popcount" => {
                expect(1)?;
                let s = self.atom(&args[0], None)?;
                self.b.push_popcount(s, 1).id()
            }
            "lzcount" => {
                expect(1)?;
                let s = self.atom(&args[0], None)?;
                self.b.push_lzcount(s, 1).id()
            }
            "carry" => {
                expect(2)?;
                let l = self.atom(&args[0], None)?;
                let r = self.atom(&args[1], None)?;
                self.b.push_carry(l, r).id()
            }
            "scarry" => {
                expect(2)?;
                let l = self.atom(&args[0], None)?;
                let r = self.atom(&args[1], None)?;
                self.b.push_scarry(l, r).id()
            }
            "sborrow" => {
                expect(2)?;
                let l = self.atom(&args[0], None)?;
                let r = self.atom(&args[1], None)?;
                self.b.push_sborrow(l, r).id()
            }
            _ => return Err(format!("unsupported function call: {op}")),
        })
    }

    fn atoms(&mut self, atoms: &[TypedAtom]) -> Result<Vec<ValueId>, String> {
        atoms.iter().map(|a| self.atom(a, None)).collect()
    }

    /// Lower an atom in a value position. `size_hint` sizes bare integer literals.
    fn atom(&mut self, typed: &TypedAtom, size_hint: Option<usize>) -> Result<ValueId, String> {
        match &typed.atom {
            Atom::External(name) => {
                // A capture resolves to a program-local if one shadows it,
                // otherwise to a value injected by the caller (the macro).
                if let Some(local) = self.locals.get(name).copied() {
                    if matches!(local, Local::BlockParam(_)) {
                        return Ok(self.coerce_block_param(local, typed.size_bytes, size_hint));
                    }
                    let v = local.value_id();
                    self.check_size(v, typed.size_bytes, name)?;
                    Ok(v)
                } else if let Some(&value) = self.externals.get(name) {
                    self.check_size(value, typed.size_bytes, name)?;
                    Ok(value)
                } else {
                    Err(format!("unknown external capture `{{{name}}}`"))
                }
            }
            Atom::Ssa(name) => {
                let local = self
                    .locals
                    .get(name)
                    .copied()
                    .ok_or_else(|| format!("unknown SSA value `%{name}`"))?;
                let Local::Instruction(_) = local else {
                    return Err(format!("`%{name}` is not an SSA value"));
                };
                let v = local.value_id();
                self.check_size(v, typed.size_bytes, name)?;
                Ok(v)
            }
            Atom::BlockParam(name) => {
                let local = self
                    .locals
                    .get(name)
                    .copied()
                    .ok_or_else(|| format!("unknown block param `@{name}`"))?;
                let Local::BlockParam(_) = local else {
                    return Err(format!("`@{name}` is not a block param"));
                };
                Ok(self.coerce_block_param(local, typed.size_bytes, size_hint))
            }
            Atom::Varnode(name) => Err(format!(
                "`{name}` is a varnode; use `&{name}` or `load(sz, {name})` in a value position"
            )),
            Atom::AddressOf(name) => {
                let local = self
                    .locals
                    .get(name)
                    .copied()
                    .ok_or_else(|| format!("unknown varnode `{name}` in addressof"))?;
                let (addr_size, value) = match local {
                    Local::Varnode(vid) => (
                        Varnode::from_id(self.b.context(), vid).space().addr_size,
                        vid.into(),
                    ),
                    Local::Temp(id) => (
                        TempRef::new(self.b.view(), id).space().addr_size(),
                        id.into(),
                    ),
                    _ => {
                        return Err(format!(
                            "`&{name}`: addressof applies only to memory values"
                        ));
                    }
                };
                // `&v` yields an address: its width is the space's pointer width,
                // not `v`'s value width.
                if let Some(expected) = typed.size_bytes
                    && addr_size != expected
                {
                    return Err(format!(
                        "qcode size mismatch for `&{name}`: expected {expected} bytes, got {addr_size}"
                    ));
                }
                Ok(value)
            }
            Atom::Int(value) => {
                let size = typed.size_bytes.or(size_hint).unwrap_or(8);
                Ok(self.b.context_mut().get_const(*value, size).id())
            }
            Atom::Bool(value) => Ok(self.b.context_mut().get_bool_const(*value).id()),
        }
    }

    /// Validate that a concrete value matches an explicit `iN`/`fN` annotation.
    /// Mirrors the old macro's compile-time `assert_eq!`, but as a runtime error
    /// (which the `qcode!` macro surfaces by `expect`-ing the lowering result).
    fn check_size(
        &self,
        value: ValueId,
        explicit: Option<usize>,
        name: &str,
    ) -> Result<(), String> {
        if let Some(expected) = explicit {
            let actual = ValueRef::new(value, self.b.context()).size();
            if actual != expected {
                return Err(format!(
                    "qcode size mismatch for `{name}`: expected {expected} bytes, got {actual}"
                ));
            }
        }
        Ok(())
    }

    /// Constrain a block param to a requested width, returning its value id.
    fn coerce_block_param(
        &mut self,
        local: Local,
        explicit: Option<usize>,
        hint: Option<usize>,
    ) -> ValueId {
        if let (Local::BlockParam(pid), Some(size)) = (local, explicit.or(hint)) {
            BlockParam::from_id_mut(self.b.context_mut(), pid).constrain_size(size);
        }
        local.value_id()
    }

    /// Lower an atom in a pointer position, where a bare varnode is valid.
    fn ptr_atom(&mut self, typed: &TypedAtom) -> Result<ValueId, String> {
        if let Atom::Varnode(name) = &typed.atom {
            let local = self
                .locals
                .get(name)
                .copied()
                .ok_or_else(|| format!("unknown varnode `{name}`"))?;
            if !matches!(local, Local::Varnode(_) | Local::Temp(_)) {
                return Err(format!("`{name}` is not a varnode"));
            }
            Ok(local.value_id())
        } else {
            self.atom(typed, None)
        }
    }

    fn call_callee(&mut self, callee: &ParsedCallee) -> Callee {
        match callee {
            ParsedCallee::Named(name) => {
                Callee::Real(self.b.get_or_make_local_function(Cow::Owned(name.clone())))
            }
            ParsedCallee::Minted(slot) => Callee::Minted(*slot),
        }
    }

    fn existing_callee(&self, callee: &ParsedCallee, operation: &str) -> Result<Callee, String> {
        match callee {
            ParsedCallee::Named(name) => self
                .symbols
                .functions
                .get(name)
                .copied()
                .map(Callee::Real)
                .ok_or_else(|| format!("{operation}: unknown function `{name}`")),
            ParsedCallee::Minted(slot) => Ok(Callee::Minted(*slot)),
        }
    }

    /// The known byte width of an atom, used to size the other operand's literals.
    fn size_hint(&self, typed: &TypedAtom) -> Option<usize> {
        if let Some(explicit) = typed.size_bytes {
            return Some(explicit);
        }
        let local = match &typed.atom {
            Atom::External(name) | Atom::Ssa(name) | Atom::BlockParam(name) => {
                self.locals.get(name).copied()?
            }
            _ => return None,
        };
        Some(ValueRef::new(local.value_id(), self.b.context()).size())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minted_callees_render_parse_and_lower_in_all_direct_forms() {
        use crate::value::insn::Mnemonic;

        fn host_block(ctx: &mut Context<'_>, name: &str) -> BlockId {
            let function = FunctionBody::make(ctx, Cow::Owned(name.to_owned()))
                .unwrap()
                .id;
            let block = BasicBlock::make(ctx, function).id;
            let mut function = FunctionBody::from_id_mut(ctx, function);
            function.add_block(block);
            function.set_root(block).unwrap();
            block
        }

        let mut rendered_ctx = Context::new();
        let block = host_block(&mut rendered_ctx, "rendered");
        let arg = rendered_ctx.get_const(1, 8).id();
        let rendered_ids = {
            let mut builder =
                Builder::from_block(BasicBlock::from_id_mut(&mut rendered_ctx, block));
            let apply = builder.push_apply(Callee::Minted(1), vec![arg]).id();
            let map = builder.push_map(Callee::Minted(2), arg, Vec::new()).id();
            let scan = builder
                .push_scan(Callee::Minted(3), arg, arg, Vec::new())
                .id();
            let call = builder
                .push_call_with_args(Callee::Minted(4), vec![arg])
                .id();
            [apply, map, scan, call].map(|value| match value {
                ValueId::Instruction(id) => id,
                _ => unreachable!(),
            })
        };
        let rendered = rendered_ids.map(|id| rendered_ctx.get_insn(id).as_statement().to_string());

        let tail_block = host_block(&mut rendered_ctx, "rendered_tail");
        let tail_id = {
            let mut builder =
                Builder::from_block(BasicBlock::from_id_mut(&mut rendered_ctx, tail_block));
            let value = builder
                .push_tail_call_with_args(Callee::Minted(5), vec![arg])
                .id();
            let ValueId::Instruction(id) = value else {
                unreachable!()
            };
            id
        };
        let tail = rendered_ctx.get_insn(tail_id).as_statement().to_string();

        let mut forms = rendered.into_iter().collect::<Vec<_>>();
        forms.push(tail);
        for (index, statement) in forms.iter().enumerate() {
            let slot = index as u32 + 1;
            assert!(
                statement.contains(&format!("<minted:{slot}>")),
                "renderer must emit the canonical placeholder: {statement}"
            );
            let mut lowered = Context::new();
            let source = format!("fn host:\n<entry>\n{statement}");
            let symbols = lower_str(&mut lowered, &source).expect("rendered form must lower");
            let instruction = FunctionBody::from_id(&lowered, symbols.function("host"))
                .root()
                .unwrap()
                .iter()
                .next()
                .unwrap();
            let actual = match instruction.mnemonic() {
                Mnemonic::Apply(value) => value.target,
                Mnemonic::Map(value) => value.body,
                Mnemonic::Scan(value) => value.body,
                Mnemonic::Call(value) => value.target,
                Mnemonic::TailCall(value) => value.target,
                other => panic!("unexpected lowered mnemonic: {other:?}"),
            };
            assert_eq!(actual, Callee::Minted(slot));
        }
    }

    #[test]
    fn lowers_fib_lambda_roundtrips() {
        let mut ctx = Context::new();
        let syms = lower_str(
            &mut ctx,
            "
            lambda fib_loop:
            <entry @input:i64>
                goto <head @hn=@input @a=0 @b=1>;
            <head @hn:i64 @a:i64 @b:i64>
                %done = @hn == 0;
                if %done goto <exit @r=@a> else goto <body @m=@hn @x=@a @y=@b>;
            <body @m:i64 @x:i64 @y:i64>
                %next = @x + @y;
                %m1 = @m - 1;
                goto <head @hn=%m1 @a=@y @b=%next>;
            <exit @r:i64>
                return @r;
            ",
        )
        .expect("lowers");

        let fid = syms.function("fib_loop");
        let f = FunctionBody::from_id(&ctx, fid);
        assert!(f.is_lambda());
        let text = f.to_string();
        assert!(text.contains("lambda fib_loop"));
        assert!(text.contains("i64 @hn == i64 0x0"));
        assert!(text.contains("i64 @x + i64 @y"));
    }

    /// Strict IR locality (context-split ruling 2): named labels are
    /// function-scoped, so the only way textual qcode can name another function's
    /// block is an *address* goto (`goto <0xADDR>`) that resolves — through the
    /// global address map — to a block already owned by a different function. The
    /// lowerer must reject it: cross-function control flow is a call/tail call.
    #[test]
    fn rejects_cross_function_address_branch() {
        use std::borrow::Cow;

        let mut ctx = Context::new();
        // A pre-existing function `g` owning a block at 0x2000.
        let g = FunctionBody::make_at_addr(&mut ctx, 0x2000, Some(Cow::Borrowed("g"))).id;
        let g_blk = BasicBlock::make(&mut ctx, g).with_address(0x2000).id;
        FunctionBody::from_id_mut(&mut ctx, g)
            .set_root(g_blk)
            .unwrap();

        // Lowering a function `f` that branches to address 0x2000 must fail: the
        // address resolves to g's block, a foreign target.
        let err = lower_str(
            &mut ctx,
            "
            fn f:
            <entry>
                goto <0x2000>;
            ",
        )
        .expect_err("cross-function address goto must be rejected");
        assert!(
            err.contains("cross-function control flow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_cross_function_rootless_function_address_branch() {
        use std::borrow::Cow;

        let mut ctx = Context::new();
        let g = FunctionBody::make_at_addr(&mut ctx, 0x2000, Some(Cow::Borrowed("g"))).id;
        assert!(FunctionBody::from_id(&ctx, g).root().is_none());

        let err = lower_str(
            &mut ctx,
            "
            fn f:
            <entry>
                goto <0x2000>;
            ",
        )
        .expect_err("rootless foreign function address goto must be rejected");
        assert!(
            err.contains("cross-function control flow"),
            "unexpected error: {err}"
        );
    }
}
