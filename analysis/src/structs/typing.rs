//! Struct-field typing: turn raw `base + offset` pointer arithmetic into typed,
//! named [`Gep`] field accesses, propagating struct-pointer types forward.
//!
//! Seeded by a value already typed as a struct pointer (e.g. the Windows-x86
//! seeding pass retyping the TEB base, or a test annotation), this pass runs a
//! forward fixpoint over the function:
//!
//! * **add → gep.** An `int_add(p, c)` where `p : PtrTo<S>` and the constant `c`
//!   exactly matches a field offset of `S` is rewritten to `gep(p, c)`, typed
//!   `PtrTo<field.type>`. The field type may itself be a struct pointer, which is
//!   how the chain advances one hop.
//! * **load inherits the field type.** A `load(ptr)` where `ptr : PtrTo<F>` and
//!   the load width equals `size_of(F)` retypes its result to `F`. When `F` is a
//!   struct pointer, the loaded value becomes the next hop's typed base.
//!
//! Resolution is **exact-match only**: an offset that names no field, or a load
//! whose width disagrees with the field size, is left untyped (a plain
//! `base + const` / integer read). Non-constant offsets are never lowered.

use std::borrow::Cow;

use qcode::{
    context::Context,
    space::{Space, SpaceId, SpaceType},
    types::TypeId,
    value::{
        BlockParam, Function, FunctionId, Instruction, Renameable, ValueId,
        insn::{Binary, Binop, Gep, InstructionId, IntBinop, Mnemonic},
    },
};

use crate::{FunctionPass, PipelineEnv};

#[derive(Default)]
pub struct StructTyping;

impl FunctionPass for StructTyping {
    const NAME: &'static str = "struct_typing";

    fn description(&self) -> &'static str {
        "Recover named struct-field accesses from segment/struct-pointer arithmetic"
    }

    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        // Nothing this pass does can fire unless some value or operand reachable in
        // the function already carries a struct / struct-pointer type: add→gep,
        // register-read, and load typing all key on a struct-pointer-typed operand,
        // and renaming keys on a struct-typed value. On a function with no complex
        // types (the common case) the whole fixpoint + rename sweep is a guaranteed
        // no-op, so bail before allocating or scanning it twice. Per-function so it
        // stays correct once non-Windows struct recovery lands.
        if !function_has_struct_types(ctx, fun_id) {
            return Ok(false);
        }

        let insn_ids: Vec<InstructionId> = Function::from_id(ctx, fun_id)
            .blocks()
            .flat_map(|b| b.iter().map(|i| i.id).collect::<Vec<_>>())
            .collect();

        let mut changed_any = false;
        loop {
            let mut changed = false;
            for &id in &insn_ids {
                if type_instruction(ctx, id) {
                    changed = true;
                    changed_any = true;
                }
            }
            if !changed {
                break;
            }
        }

        // Once the types have settled, rename every struct-typed SSA value and
        // argument after the struct it (points to): a value of type `PEB*`
        // becomes `%peb`. This runs over the whole function so it also picks up
        // arguments typed by an upstream seed.
        changed_any |= rename_struct_values(ctx, fun_id);

        Ok(changed_any)
    }
}

/// Renames struct-typed SSA values and block parameters after the struct they
/// reference (e.g. a `PEB*` value becomes `%peb`), keeping names unique within
/// the function. Returns `true` if any value was renamed.
fn rename_struct_values(ctx: &mut Context, fun_id: FunctionId) -> bool {
    let fun = Function::from_id(ctx, fun_id);
    let values: Vec<ValueId> = fun
        .blocks()
        .flat_map(|b| {
            b.params()
                .map(|p| p.id())
                .chain(b.iter().map(|i| ValueId::Instruction(i.id)))
                .collect::<Vec<_>>()
        })
        .collect();

    let mut changed = false;
    for value in values {
        let Some(base) = ctx
            .stored_type_of(value)
            .and_then(|t| struct_base_name(ctx, t))
        else {
            continue;
        };
        if let Some(name) = unique_name(ctx, value, &base) {
            let renamed = match value {
                ValueId::Instruction(id) => Instruction::from_id_mut(ctx, id).rename(name).is_ok(),
                ValueId::BlockParam(id) => BlockParam::from_id_mut(ctx, id).rename(name).is_ok(),
                _ => false,
            };
            changed |= renamed;
        }
    }
    changed
}

/// The lowercased struct name a value of type `ty` should be named after: the
/// pointee's name when `ty` is a struct pointer, or the struct's own name when
/// `ty` is a struct value. `None` for non-struct types.
fn struct_base_name(ctx: &Context, ty: TypeId) -> Option<String> {
    let struct_ty = ctx.types.pointee_of(ty).unwrap_or(ty);
    ctx.types.struct_name_of(struct_ty).map(str::to_lowercase)
}

/// Picks a unique name for `value` from `base`, `base1`, `base2`, … Returns
/// `None` if `value` is already named with such a candidate (nothing to do).
fn unique_name<'str>(ctx: &Context, value: ValueId, base: &str) -> Option<Cow<'str, str>> {
    for n in 0.. {
        let candidate = if n == 0 {
            base.to_string()
        } else {
            format!("{base}{n}")
        };
        match ctx.get_named(&candidate) {
            Some(owner) if owner == value => return None,
            Some(_) => continue,
            None => return Some(Cow::Owned(candidate)),
        }
    }
    None
}

/// Whether any value or operand reachable in `fun_id` currently carries a struct
/// or struct-pointer type — the precondition for [`StructTyping`] to do anything.
/// A single linear scan; returns on the first struct-ish type found. Checks
/// instruction results, block params, *and* operands, because the seed can live on
/// an operand varnode (the Windows TEB seed retypes the `FS_OFFSET` register that a
/// `load(register, reg)` reads) rather than on a value the function defines.
fn function_has_struct_types(ctx: &Context, fun_id: FunctionId) -> bool {
    let is_struct_ish = |v: ValueId| {
        ctx.stored_type_of(v).is_some_and(|t| {
            ctx.types.pointee_of(t).is_some() || ctx.types.struct_name_of(t).is_some()
        })
    };
    Function::from_id(ctx, fun_id).blocks().any(|b| {
        b.params().any(|p| is_struct_ish(p.id()))
            || b.iter().any(|i| {
                is_struct_ish(ValueId::Instruction(i.id))
                    || i.mnemonic().args().iter().copied().any(is_struct_ish)
            })
    })
}

/// Attempts one typing step on instruction `id`. Returns `true` if it changed
/// the IR (rewrote an add to a gep, or retyped a load result).
fn type_instruction(ctx: &mut Context, id: InstructionId) -> bool {
    match ctx.values.instructions[id].mnemonic().clone() {
        Mnemonic::Binop(Binary {
            op: Binop::Int(IntBinop::Add),
            lhs,
            rhs,
        }) => try_add_to_gep(ctx, id, lhs, rhs),
        // A register read (`load` from the register space) yields the register's
        // own value type — which the TEB seed overrode to `PtrTo<TEB>`. A normal
        // RAM load dereferences a field pointer.
        Mnemonic::Load(load) if is_register_space(ctx, load.space) => {
            try_type_register_read(ctx, id, load.ptr, load.size)
        }
        Mnemonic::Load(load) => try_type_load(ctx, id, load.ptr, load.size),
        _ => false,
    }
}

/// Whether `space` is the processor register file.
fn is_register_space(ctx: &Context, space: SpaceId) -> bool {
    matches!(Space::from_id(ctx, space).ty, SpaceType::Register)
}

/// `load(register, reg)` is a register read: its result takes the register's own
/// value type. Only acts when that type is a (struct) pointer — i.e. the TEB
/// seed overrode `FS_OFFSET` to `PtrTo<TEB>`; plain integer registers are left
/// untouched. Exact-size match only.
fn try_type_register_read(ctx: &mut Context, id: InstructionId, reg: ValueId, size: usize) -> bool {
    let Some(reg_ty) = ctx.stored_type_of(reg) else {
        return false;
    };
    if ctx.types.pointee_of(reg_ty).is_none() || ctx.types.size_of(reg_ty) != size {
        return false;
    }
    if ctx.stored_type_of(ValueId::Instruction(id)) == Some(reg_ty) {
        return false;
    }
    Instruction::from_id_mut(ctx, id).set_type(reg_ty);
    true
}

/// `int_add(base, const)` with `base : PtrTo<S>` and `const` an exact field
/// offset of `S` → `gep(base, off)` typed `PtrTo<field.type>`.
fn try_add_to_gep(ctx: &mut Context, id: InstructionId, lhs: ValueId, rhs: ValueId) -> bool {
    for (base, off_op) in [(lhs, rhs), (rhs, lhs)] {
        let Some(pointee) = ctx
            .stored_type_of(base)
            .and_then(|t| ctx.types.pointee_of(t))
        else {
            continue;
        };
        let Some(offset) = const_offset(ctx, off_op) else {
            continue;
        };
        let Some((_, field)) = ctx.types.field_by_offset(pointee, offset) else {
            continue;
        };
        let field_ty = field.type_id;
        let width = ctx.types.size_of(ctx.stored_type_of(base).unwrap());
        let result_ty = ctx.types.get_or_make_struct_pointer(width, field_ty);
        ctx.replace_instruction_mnemonic(id, Mnemonic::Gep(Gep { base, offset }));
        Instruction::from_id_mut(ctx, id).set_type(result_ty);
        return true;
    }
    false
}

/// `load(ptr)` with `ptr : PtrTo<F>` and `load.size == size_of(F)` → result
/// retyped to `F`. Exact-size match only; otherwise left as an integer read.
fn try_type_load(ctx: &mut Context, id: InstructionId, ptr: ValueId, size: usize) -> bool {
    let Some(field_ty) = ctx
        .stored_type_of(ptr)
        .and_then(|t| ctx.types.pointee_of(t))
    else {
        return false;
    };
    if ctx.types.size_of(field_ty) != size {
        return false;
    }
    if ctx.stored_type_of(ValueId::Instruction(id)) == Some(field_ty) {
        return false;
    }
    Instruction::from_id_mut(ctx, id).set_type(field_ty);
    true
}

/// The concrete constant value of `op` as a byte offset, or `None` if `op` is
/// not a plain (non-symbolic) integer literal.
fn const_offset(ctx: &Context, op: ValueId) -> Option<usize> {
    let ValueId::Literal(lid) = op else {
        return None;
    };
    let lit = &ctx.values.literals[lid];
    if lit.symbolic.is_some() {
        return None;
    }
    Some(lit.value as usize)
}

crate::register_function_pass!(StructTyping);

#[cfg(test)]
mod tests {
    use qcode::value::{Function, ValueId};
    use qcode_macro::qcode;

    use super::*;
    use crate::test_util::run_function_pass;
    use qcode::context::Context;

    /// Collect the `gep` statements of a function in program order.
    fn gep_strings(ctx: &Context, fun: FunctionId) -> Vec<String> {
        Function::from_id(ctx, fun)
            .blocks()
            .flat_map(|b| {
                b.iter()
                    .filter(|i| matches!(i.mnemonic(), Mnemonic::Gep(_)))
                    .map(|i| i.as_statement().to_string())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn types_a_two_hop_field_chain() {
        let mut ctx = Context::new();
        // Root { inner: Inner* @0x10 }, Inner { val: i32 @0x08 }. Inner is
        // declared first so the `Inner*` field of Root resolves.
        qcode!(
            ctx,
            "
            type Inner { _: 8, val: 4 };
            type Root { _: 0x10, inner: Inner* };
            fn f:
                <entry>
                    varnode i64 base;
                    Root* %root = load(base:8, base);
                    %inner_slot = %root + 0x10;
                    %inner = load(ram:8, %inner_slot);
                    %val_slot = %inner + 8;
                    %vv = load(ram:4, %val_slot);
                    return at i32 0;
            "
        );

        let changed = run_function_pass::<StructTyping>(&mut ctx, f).unwrap();
        assert!(changed);

        // Both hops became named geps.
        let geps = gep_strings(&ctx, f);
        assert!(
            geps.iter().any(|g| g.contains("gep(%root.inner)")),
            "geps: {geps:?}"
        );
        assert!(
            geps.iter().any(|g| g.contains("gep(%inner.val)")),
            "geps: {geps:?}"
        );

        // The loaded `inner` pointer inherited the `Inner*` field type.
        let inner_id = Function::from_id(&ctx, f)
            .blocks()
            .flat_map(|b| {
                b.iter()
                    .filter(|i| i.name() == Some("inner"))
                    .map(|i| i.id)
                    .collect::<Vec<_>>()
            })
            .next()
            .expect("%inner exists");
        let inner_ty = ctx.stored_type_of(ValueId::Instruction(inner_id)).unwrap();
        assert!(
            ctx.types.pointee_of(inner_ty).is_some(),
            "%inner should be a struct pointer"
        );
    }

    #[test]
    fn renames_struct_typed_values_after_their_struct() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            type Inner { _: 8, val: 4 };
            type Root { _: 0x10, inner: Inner* };
            fn f:
                <entry>
                    varnode i64 base;
                    Root* %x = load(base:8, base);
                    %slot = %x + 0x10;
                    %y = load(ram:8, %slot);
                    return at i32 0;
            "
        );

        run_function_pass::<StructTyping>(&mut ctx, f).unwrap();

        // The `Root*` value `%x` and the `Inner*` value `%y` are renamed after
        // the structs they reference.
        assert!(ctx.get_named("root").is_some());
        assert!(ctx.get_named("inner").is_some());
        assert!(ctx.get_named("x").is_none());
        assert!(ctx.get_named("y").is_none());
    }

    #[test]
    fn leaves_unknown_offset_and_dynamic_offset_untyped() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            type Root { _: 0x10, inner: 8 };
            fn f:
                <entry>
                    varnode i64 base;
                    varnode i64 dynp;
                    Root* %root = load(base:8, base);
                    %no_field = %root + 0x99;
                    %d = load(dynp:8, dynp);
                    %dynamic = %root + %d;
                    return at i32 0;
            "
        );

        run_function_pass::<StructTyping>(&mut ctx, f).unwrap();

        // No field at 0x99 and a non-constant offset: neither is lowered to gep.
        assert!(gep_strings(&ctx, f).is_empty());
    }
}
