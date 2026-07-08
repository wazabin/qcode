//! Windows TEB seeding: type the segment-base register (`FS_OFFSET` on x86) as a
//! global `PtrTo<TEB>` so that [`StructTyping`](crate::structs::typing) can unfold
//! the TEB/PEB field chain.
//!
//! The lifter emits `fs:[off]` as `Binop(Add, varnode(FS_OFFSET), off)` — the
//! register is used *directly* as a value operand (`emit.rs` `ident_value`). A
//! varnode is normally typed `Int(size)`; this pass installs a **global** type
//! override on that one register (see [`Context::set_varnode_type`]), so every
//! such access in every function reads a `TEB*`. It records an
//! [`Assumed`](qcode::assumption::Certainty) [`Proposition::WindowsTeb`] as an
//! analyst aid + override hook (no verifier in v1).
//!
//! The TEB/PEB layout is **not** hand-written here: it is parsed from `teb.h`
//! by `build.rs` (libclang) and baked in as [`HStruct`]s, which
//! [`register_teb_structs`] turns into nominal qcode structs.
//!
//! [`WindowsTebSeed`] is the registered pass: it gates on the Windows-x86
//! platform (read from [`ArchConfig`](crate::ArchConfig) on the [`PipelineEnv`]),
//! resolves the `FS_OFFSET` register, and calls [`seed_teb_register`] (the
//! reusable core that registers the structs, types the varnode, and records the
//! assumption).

use std::collections::HashMap;

use qcode::{
    assumption::Proposition,
    context::{Context, TargetOs},
    types::{AggregateField, TypeId},
    value::{ValueId, Varnode, VarnodeId},
};

use crate::{Pass, PipelineEnv};

use super::hstruct::{HFieldKind, HStruct};

/// The TEB/PEB layouts parsed from `teb.h` at build time.
fn baked_structs() -> Vec<HStruct> {
    let bytes = include_bytes!(env!("QCODE_TEB_STRUCTS"));
    bincode::serde::decode_from_slice(bytes, bincode::config::standard())
        .expect("decode baked teb structs")
        .0
}

/// Registers every struct from `teb.h` as a nominal qcode struct and returns the
/// name → [`TypeId`] map. Declaration order in the header guarantees a pointee
/// struct is registered before any struct that points at it.
pub fn register_teb_structs(ctx: &mut Context) -> HashMap<String, TypeId> {
    let mut by_name: HashMap<String, TypeId> = HashMap::new();
    for s in baked_structs() {
        let fields = s
            .fields
            .iter()
            .map(|f| {
                let type_id = match &f.kind {
                    HFieldKind::Int { size } => ctx.types.get_or_make_int(*size),
                    HFieldKind::StructPtr { pointee, width } => {
                        let pointee_ty = *by_name.get(pointee).unwrap_or_else(|| {
                            panic!("pointee struct {pointee} not yet registered")
                        });
                        ctx.types.get_or_make_struct_pointer(*width, pointee_ty)
                    }
                };
                AggregateField::new_at(f.name.clone(), type_id, f.offset)
            })
            .collect();
        let id = ctx.types.get_or_make_struct(s.name.clone(), s.size, fields);
        by_name.insert(s.name, id);
    }
    by_name
}

/// Pointer width in bytes for a Windows `bitness` (32 → 4, otherwise 8).
fn ptr_width(bitness: u8) -> usize {
    if bitness == 32 { 4 } else { 8 }
}

/// Types the `fs_offset` register varnode as a global `PtrTo<TEB>` (using the
/// `teb.h` layout) and records the [`Proposition::WindowsTeb`] assumption.
/// Returns `false` if the header defines no `TEB` struct.
pub fn seed_teb_register(ctx: &mut Context, fs_offset: VarnodeId, bitness: u8) -> bool {
    let structs = register_teb_structs(ctx);
    let Some(&teb) = structs.get("TEB") else {
        return false;
    };
    let teb_ptr = ctx
        .types
        .get_or_make_struct_pointer(ptr_width(bitness), teb);
    ctx.set_varnode_type(fs_offset, teb_ptr);
    ctx.assume_true(Proposition::WindowsTeb { bitness });
    true
}

/// The varnode backing the named register (e.g. `"FS_OFFSET"`), if the context
/// has it. Register varnodes carry their sleigh name.
fn register_varnode(ctx: &Context, name: &str) -> Option<VarnodeId> {
    ctx.registers
        .values()
        .copied()
        .find(|&vid| Varnode::from_id(ctx, vid).name() == Some(name))
}

/// Registered **module** pass: on a **Windows x86** target, type the `FS_OFFSET`
/// segment base as a global `PtrTo<TEB>` (the platform gate reads
/// [`ArchConfig`](crate::ArchConfig) on the [`PipelineEnv`], populated from the
/// PE header). A no-op on any other platform, or once the register is already
/// typed.
///
/// It is a whole-program [`Pass`], not a [`FunctionPass`]: it ignores any single
/// function and mutates *global* module state (a varnode-type override, global
/// struct types, and a module assumption). One application covers every
/// function, so it belongs in a module-scoped stage.
#[derive(Default)]
pub struct WindowsTebSeed;

impl Pass for WindowsTebSeed {
    const NAME: &'static str = "windows_teb_seed";

    fn description(&self) -> &'static str {
        "Type the FS_OFFSET register as PtrTo<TEB> on Windows x86"
    }

    fn run(&self, ctx: &mut Context, env: &PipelineEnv) -> Result<bool, String> {
        if env.cfg.os != TargetOs::Windows || env.cfg.bitness != 32 {
            return Ok(false);
        }
        let Some(fs) = register_varnode(ctx, "FS_OFFSET") else {
            return Ok(false);
        };
        // The override is global; if it's already a pointer, nothing to do.
        if ctx
            .stored_type_of(ValueId::Varnode(fs))
            .and_then(|t| ctx.types.pointee_of(t))
            .is_some()
        {
            return Ok(false);
        }
        Ok(seed_teb_register(ctx, fs, 32))
    }
}

crate::register_module_pass!(WindowsTebSeed);

#[cfg(test)]
mod tests {
    use qcode::assumption::{Certainty, Proposition};
    use qcode::context::Context;
    use qcode::value::{Function, ValueId, insn::Mnemonic};
    use qcode_macro::qcode;

    use super::*;
    use crate::structs::typing::StructTyping;
    use crate::test_util::run_function_pass;

    /// The `teb.h` layout parses to the expected offsets and pointer field.
    #[test]
    fn parses_teb_header_layout() {
        let mut ctx = Context::new();
        let structs = register_teb_structs(&mut ctx);

        let peb = structs["PEB"];
        let (_, being_debugged) = ctx.types.field_by_offset(peb, 0x02).expect("BeingDebugged");
        assert_eq!(being_debugged.name, "BeingDebugged");
        assert!(ctx.types.field_by_offset(peb, 0x18).is_some()); // ProcessHeap
        assert!(ctx.types.field_by_offset(peb, 0x68).is_some()); // NtGlobalFlag

        // TEB.ProcessEnvironmentBlock @0x30 is a pointer to PEB.
        let teb = structs["TEB"];
        let (_, peb_field) = ctx
            .types
            .field_by_offset(teb, 0x30)
            .expect("PEB pointer @0x30");
        let pointee = ctx
            .types
            .pointee_of(peb_field.type_id)
            .expect("is a pointer");
        assert_eq!(ctx.types.struct_name_of(pointee), Some("PEB"));
    }

    /// End-to-end: a typed `FS_OFFSET` register feeds struct typing. The lifted
    /// shape `fs:[0x30]` is `Binop(Add, varnode(fs), 0x30)`, authored in qcode as
    /// `&fs + 0x30` (which lowers to the varnode used directly as a value).
    #[test]
    fn seeded_register_unfolds_to_named_gep() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
                <entry>
                    varnode i32 fs;
                    %peb_slot = &fs + i32 0x30;
                    return at i32 0;
            "
        );

        assert!(seed_teb_register(&mut ctx, fs, 32));
        // The register now reads as a TEB pointer everywhere.
        let ty = ctx.type_of(ValueId::Varnode(fs));
        assert!(ctx.types.pointee_of(ty).is_some());

        run_function_pass::<StructTyping>(&mut ctx, f).unwrap();

        // `&fs + 0x30` became the named field access `gep(fs.ProcessEnvironmentBlock)`.
        let gep = Function::from_id(&ctx, f)
            .blocks()
            .flat_map(|b| {
                b.iter()
                    .filter(|i| matches!(i.mnemonic(), Mnemonic::Gep(_)))
                    .map(|i| i.as_statement().to_string())
                    .collect::<Vec<_>>()
            })
            .next()
            .expect("a gep was produced");
        assert!(gep.contains("ProcessEnvironmentBlock"), "got: {gep}");

        // The assumption is recorded as Assumed.
        let truth = ctx
            .truth(Proposition::WindowsTeb { bitness: 32 })
            .expect("assumption recorded");
        assert!(truth.value);
        assert_eq!(truth.certainty, Certainty::Assumed);
    }

    /// A [`PipelineEnv`] with the given OS/bitness for gating tests.
    fn env_for(os: TargetOs, bitness: u8) -> PipelineEnv {
        use crate::{ArchConfig, CallingConvention};
        use qcode::value::{RegisterId, VarnodeId};
        PipelineEnv::from_parts(
            ArchConfig {
                stack_pointer: RegisterId::from(0usize),
                dead_flag_regs: Vec::new(),
                abi: CallingConvention::default(),
                os,
                bitness,
            },
            VarnodeId::from(0usize),
        )
    }

    /// The registered pass types `FS_OFFSET` only on a Windows-x86 env.
    #[test]
    fn pass_gates_on_windows_x86() {
        use qcode::value::{RegisterId, Renameable, Varnode};
        use std::borrow::Cow;

        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
                <entry>
                    return at i32 0;
            "
        );
        // Register an `FS_OFFSET` register varnode (as the lifter would).
        let space = ctx.default_space;
        let fs = Varnode::make(&mut ctx, 0x110, 4, space).id;
        Varnode::from_id_mut(&mut ctx, fs)
            .rename(Cow::Borrowed("FS_OFFSET"))
            .unwrap();
        ctx.registers.insert(RegisterId::from(0usize), fs);

        // Wrong platform: no-op, register stays untyped.
        assert!(
            !WindowsTebSeed
                .run(&mut ctx, &env_for(TargetOs::Linux, 64))
                .unwrap()
        );
        let t = ctx.type_of(ValueId::Varnode(fs));
        assert!(ctx.types.pointee_of(t).is_none());

        // Windows x86: types the register and is idempotent on a second run.
        assert!(
            WindowsTebSeed
                .run(&mut ctx, &env_for(TargetOs::Windows, 32))
                .unwrap()
        );
        let t = ctx.type_of(ValueId::Varnode(fs));
        assert!(ctx.types.pointee_of(t).is_some());
        assert!(
            !WindowsTebSeed
                .run(&mut ctx, &env_for(TargetOs::Windows, 32))
                .unwrap()
        );
    }
}
