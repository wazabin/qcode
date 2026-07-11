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
    space::{Space, SpaceId, SpaceType},
    types::TypeId,
    value::{
        FunctionId, ValueId,
        insn::{Binary, Binop, Gep, InstructionId, IntBinop, Mnemonic},
        util::base_ref::{BaseRef, HostRef},
    },
};

use crate::{ContextView, FunctionBody, FunctionPass};

// TODO(5b-ii): Public functions below are thin wrappers marked for future migration

#[derive(Default)]
pub struct StructTyping;

impl FunctionPass for StructTyping {
    const NAME: &'static str = "struct_typing";

    fn description(&self) -> &'static str {
        "Recover named struct-field accesses from segment/struct-pointer arithmetic"
    }

    fn run<'str>(
        &self,
        f: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
    ) -> Result<bool, String> {
        let fid = f.id();
        Ok(struct_typing(f, cx, fid))
    }
}

/// Recover struct-field accesses in `fun_id`, mutating the function through
/// concrete `(body, cx)` (see the module docs). Returns `true` if the IR changed.
pub fn struct_typing<'a, 'str>(
    body: &'a mut FunctionBody<'str>,
    cx: ContextView<'a, 'str>,
    fun_id: FunctionId,
) -> bool {
    // Nothing this pass does can fire unless some value or operand reachable in
    // the function already carries a struct / struct-pointer type: add→gep,
    // register-read, and load typing all key on a struct-pointer-typed operand,
    // and renaming keys on a struct-typed value. On a function with no complex
    // types (the common case) the whole fixpoint + rename sweep is a guaranteed
    // no-op, so bail before allocating or scanning it twice. Per-function so it
    // stays correct once non-Windows struct recovery lands.
    if !function_has_struct_types(body.read_host(cx), fun_id) {
        return false;
    }

    let insn_ids: Vec<InstructionId> = body
        .function_ref(cx, fun_id)
        .blocks()
        .flat_map(|b| b.instruction_ids().to_vec())
        .collect();

    let mut changed_any = false;
    loop {
        let mut changed = false;
        for &id in &insn_ids {
            if type_instruction(body, cx, id) {
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
    changed_any |= rename_struct_values(body, cx, fun_id);

    changed_any
}

/// The stored [`TypeId`] of `id`, host-routed: a checked-out function's
/// instruction/param types live in its owned arena, other kinds in shared data.
fn stored_type_of<'str>(host: HostRef<'_, 'str>, id: ValueId) -> Option<TypeId> {
    match id {
        ValueId::Instruction(iid) => Some(host.insn_ref(iid).type_id()),
        ValueId::BlockParam(pid) => Some(host.param_ref(pid).type_id()),
        other => host.shared().stored_type_of(other),
    }
}

/// Renames struct-typed SSA values and block parameters after the struct they
/// reference (e.g. a `PEB*` value becomes `%peb`), keeping names unique within
/// the function. Returns `true` if any value was renamed.
fn rename_struct_values<'a, 'str>(
    body: &'a mut FunctionBody<'str>,
    cx: ContextView<'a, 'str>,
    fun_id: FunctionId,
) -> bool {
    let values: Vec<ValueId> = body
        .function_ref(cx, fun_id)
        .blocks()
        .flat_map(|b| {
            b.params()
                .map(|p| p.id())
                .chain(b.instruction_ids().iter().map(|&i| ValueId::Instruction(i)))
                .collect::<Vec<_>>()
        })
        .collect();

    let mut changed = false;
    for value in values {
        let Some(base) = stored_type_of(body.read_host(cx), value)
            .and_then(|t| struct_base_name(body.read_host(cx), t))
        else {
            continue;
        };
        if let Some(name) = unique_name(body.read_host(cx), fun_id, value, &base) {
            let renamed = match value {
                ValueId::Instruction(id) => {
                    BaseRef::new(body.host(cx), id).rename_local(name).is_ok()
                }
                ValueId::BlockParam(id) => {
                    BaseRef::new(body.host(cx), id).rename_local(name).is_ok()
                }
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
fn struct_base_name(host: HostRef, ty: TypeId) -> Option<String> {
    let types = &host.shared().shared.types;
    let struct_ty = types.pointee_of(ty).unwrap_or(ty);
    types.struct_name_of(struct_ty).map(str::to_lowercase)
}

/// Picks a unique name for `value` from `base`, `base1`, `base2`, … Returns
/// `None` if `value` is already named with such a candidate (nothing to do).
fn unique_name<'str>(
    host: HostRef,
    fun_id: FunctionId,
    value: ValueId,
    base: &str,
) -> Option<Cow<'str, str>> {
    let function = host.function_ref(fun_id);
    for n in 0.. {
        let candidate = if n == 0 {
            base.to_string()
        } else {
            format!("{base}{n}")
        };
        // `value` is an SSA def (its name is function-scoped), so check the
        // candidate for freedom in this function's table, not globally.
        match function.local_named(&candidate) {
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
fn function_has_struct_types(host: HostRef, fun_id: FunctionId) -> bool {
    let is_struct_ish = |v: ValueId| {
        stored_type_of(host, v).is_some_and(|t| {
            let types = &host.shared().shared.types;
            types.pointee_of(t).is_some() || types.struct_name_of(t).is_some()
        })
    };
    host.function_ref(fun_id).blocks().any(|b| {
        b.params().any(|p| is_struct_ish(p.id()))
            || b.iter().any(|i| {
                is_struct_ish(ValueId::Instruction(i.id))
                    || i.mnemonic().args().iter().copied().any(is_struct_ish)
            })
    })
}

/// Attempts one typing step on instruction `id`. Returns `true` if it changed
/// the IR (rewrote an add to a gep, or retyped a load result).
fn type_instruction<'a, 'str>(
    body: &'a mut FunctionBody<'str>,
    cx: ContextView<'a, 'str>,
    id: InstructionId,
) -> bool {
    match body.insn_ref(cx, id).mnemonic().clone() {
        Mnemonic::Binop(Binary {
            op: Binop::Int(IntBinop::Add),
            lhs,
            rhs,
        }) => try_add_to_gep(body, cx, id, lhs, rhs),
        // A register read (`load` from the register space) yields the register's
        // own value type — which the TEB seed overrode to `PtrTo<TEB>`. A normal
        // RAM load dereferences a field pointer.
        Mnemonic::Load(load) if is_register_space(body.read_host(cx), load.space) => {
            try_type_register_read(body, cx, id, load.ptr, load.size)
        }
        Mnemonic::Load(load) => try_type_load(body, cx, id, load.ptr, load.size),
        _ => false,
    }
}

/// Whether `space` is the processor register file.
fn is_register_space(host: HostRef, space: SpaceId) -> bool {
    matches!(Space::from_id(host.shared(), space).ty, SpaceType::Register)
}

/// `load(register, reg)` is a register read: its result takes the register's own
/// value type. Only acts when that type is a (struct) pointer — i.e. the TEB
/// seed overrode `FS_OFFSET` to `PtrTo<TEB>`; plain integer registers are left
/// untouched. Exact-size match only.
fn try_type_register_read<'a, 'str>(
    body: &'a mut FunctionBody<'str>,
    cx: ContextView<'a, 'str>,
    id: InstructionId,
    reg: ValueId,
    size: usize,
) -> bool {
    let Some(reg_ty) = stored_type_of(body.read_host(cx), reg) else {
        return false;
    };
    let (is_ptr, reg_size) = {
        let types = &cx.shared_ctx().shared.types;
        (types.pointee_of(reg_ty).is_some(), types.size_of(reg_ty))
    };
    if !is_ptr || reg_size != size {
        return false;
    }
    if stored_type_of(body.read_host(cx), ValueId::Instruction(id)) == Some(reg_ty) {
        return false;
    }
    BaseRef::new(body.host(cx), id).set_result_type(reg_ty);
    true
}

/// `int_add(base, const)` with `base : PtrTo<S>` and `const` an exact field
/// offset of `S` → `gep(base, off)` typed `PtrTo<field.type>`.
fn try_add_to_gep<'a, 'str>(
    body: &'a mut FunctionBody<'str>,
    cx: ContextView<'a, 'str>,
    id: InstructionId,
    lhs: ValueId,
    rhs: ValueId,
) -> bool {
    for (base, off_op) in [(lhs, rhs), (rhs, lhs)] {
        let Some(base_ty) = stored_type_of(body.read_host(cx), base) else {
            continue;
        };
        let Some(pointee) = cx.shared_ctx().shared.types.pointee_of(base_ty) else {
            continue;
        };
        let Some(offset) = const_offset(body.read_host(cx), off_op) else {
            continue;
        };
        let field_ty = match cx
            .shared_ctx()
            .shared
            .types
            .field_by_offset(pointee, offset)
        {
            Some((_, field)) => field.type_id,
            None => continue,
        };
        let width = cx.shared_ctx().shared.types.size_of(base_ty);
        let result_ty = cx
            .shared_ctx()
            .shared
            .types
            .get_or_make_struct_pointer(width, field_ty);
        body.replace_instruction_mnemonic(cx, id, Mnemonic::Gep(Gep { base, offset }));
        BaseRef::new(body.host(cx), id).set_result_type(result_ty);
        return true;
    }
    false
}

/// `load(ptr)` with `ptr : PtrTo<F>` and `load.size == size_of(F)` → result
/// retyped to `F`. Exact-size match only; otherwise left as an integer read.
fn try_type_load<'a, 'str>(
    body: &'a mut FunctionBody<'str>,
    cx: ContextView<'a, 'str>,
    id: InstructionId,
    ptr: ValueId,
    size: usize,
) -> bool {
    let Some(ptr_ty) = stored_type_of(body.read_host(cx), ptr) else {
        return false;
    };
    let Some(field_ty) = cx.shared_ctx().shared.types.pointee_of(ptr_ty) else {
        return false;
    };
    if cx.shared_ctx().shared.types.size_of(field_ty) != size {
        return false;
    }
    if stored_type_of(body.read_host(cx), ValueId::Instruction(id)) == Some(field_ty) {
        return false;
    }
    BaseRef::new(body.host(cx), id).set_result_type(field_ty);
    true
}

/// The concrete constant value of `op` as a byte offset, or `None` if `op` is
/// not a plain (non-symbolic) integer literal.
fn const_offset(host: HostRef, op: ValueId) -> Option<usize> {
    let ValueId::Literal(lid) = op else {
        return None;
    };
    let lit = &host.shared().shared.values.literals[lid];
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
            ctx.shared.types.pointee_of(inner_ty).is_some(),
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
        // the structs they reference. Value names are function-scoped, so resolve
        // them within `f`.
        let f = qcode::value::FunctionRef::from_id(&ctx, f);
        assert!(f.local_named("root").is_some());
        assert!(f.local_named("inner").is_some());
        assert!(f.local_named("x").is_none());
        assert!(f.local_named("y").is_none());
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
