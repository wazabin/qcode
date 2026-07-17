//! Signatures and call interfaces for known external (imported) functions, from
//! their C prototypes.
//!
//! External functions are bodyless stubs, so [`call_summary`](crate::calls)
//! cannot infer their inputs. Instead this pass looks each function name up in a
//! per-binary [`cabi::Selection`] — the prototype tables for the libraries the
//! binary actually links against — and **materializes** everything downstream
//! passes need, so `cabi` is consulted in exactly this one pass:
//!
//! * the register signature (`input_regs`, `output_regs`, `param_attrs`) under
//!   the [`CallingConvention`] supplied by [`PipelineEnv`], plus the caller-saved
//!   clobber set and the `externally_resolved` flag;
//! * a persisted [`ExternInterface`] on the function's signature — the ordered
//!   call slots (register or stack, incl. the synthesized `stdcall`/`cdecl`
//!   `return_address` slot) and the display names — consumed later by
//!   [`argpromote_external`](crate::calls::argpromote), which then needs neither
//!   `cabi` nor the binary.
//!
//! Selection is strict: an empty linked-library list yields an empty selection
//! and the pass is a no-op. Hex lifts / DSL tests opt in via `--assume-libs`
//! (`TestContext::assume_libs`), which seeds `Context::linked_libraries`.
//!
//! Run before [`set_all_function_summaries`](crate::set_all_function_summaries):
//! once an external callee has a signature, the argument-producing passes can
//! produce real arguments at its call sites.

use cabi::{CFunctionProto, CType, Config, Selection};
use qcode::{
    context::Context,
    value::{
        ExternArg, ExternInterface, ExternSlot, FunctionBody, FunctionEffects, FunctionId,
        ParamAttrs, RegisterInterfaceMap, VarnodeId,
    },
};

use crate::pipeline::{CallingConvention, cabi_abi_target};

use crate::{Pass, PipelineEnv};

#[derive(Default)]
pub struct ExternalSigs;

impl Pass for ExternalSigs {
    const NAME: &'static str = "external_sigs";

    fn description(&self) -> &'static str {
        "Give known external (libc) functions signatures from their C prototypes"
    }

    fn run(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        targets: &[FunctionId],
    ) -> Result<crate::ModulePassOutcome, String> {
        if !abi_is_known(&env.cfg.abi) {
            return Ok(crate::ModulePassOutcome::default());
        }
        let affected: Vec<FunctionId> = targets
            .iter()
            .copied()
            .filter(|&id| FunctionBody::from_id(ctx, id).is_external())
            .collect();
        let sel = selection_for(ctx, env);
        let ptr_width = ptr_width(env);
        let stack_only = env.cfg.bitness == 32;
        apply_external_signatures(ctx, &affected, &env.cfg.abi, &sel, ptr_width, stack_only);
        Ok(crate::ModulePassOutcome::functions(affected)
            .preserving_global::<crate::CallGraphAnalysis>()
            .preserving_global::<crate::AddressAnalysis>())
    }
}

crate::register_module_pass!(ExternalSigs);

/// Pointer width in bytes for the analysis target (at least 1).
fn ptr_width(env: &PipelineEnv) -> usize {
    (env.cfg.bitness / 8).max(1) as usize
}

/// Build the per-binary prototype [`Selection`] for this run: the effective
/// [`Config`], the analysis [`cabi::AbiTarget`], and the binary's linked-library
/// names. The live binary handle is preferred; when it reports no libraries
/// (e.g. a raw `Blob`) the loader/`--assume-libs`-seeded `Context` list is used.
/// An empty list yields an empty selection, so the pass becomes a no-op.
fn selection_for(ctx: &Context, env: &PipelineEnv) -> Selection {
    let target = cabi_abi_target(env.cfg.os, env.cfg.bitness);
    let linked: Vec<String> = env
        .binary
        .as_ref()
        .map(|b| b.linked_libraries())
        .filter(|libs| !libs.is_empty())
        .unwrap_or_else(|| ctx.linked_libraries().to_vec());
    cabi::select(&Config::load(), target, &linked)
}

/// System V argument classes for a scalar value.
enum Class {
    /// INTEGER class: integers, enums, pointers — passed in a GP register.
    Integer,
    /// SSE class: float/double — passed in an XMM register.
    Sse,
}

/// Classify a scalar [`CType`], or `None` for `void`/aggregates we cannot map.
///
/// We deliberately ignore the integer byte width: SysV allocates a whole 64-bit
/// GP register per INTEGER argument (a narrow value just occupies the low bits),
/// and per-typedef widths from the build-time extractor are unreliable for some
/// builtin typedefs (e.g. `size_t`). Using the full register is ABI-correct.
fn classify(ty: &CType) -> Option<Class> {
    match ty {
        CType::Integer { .. } | CType::Pointer { .. } => Some(Class::Integer),
        CType::Float { .. } => Some(Class::Sse),
        // By-value aggregates use the memory/split classes, which we do not
        // model; treat them (like `void`) as unmappable.
        CType::Void | CType::Struct { .. } | CType::Other => None,
    }
}

/// The full 64-bit view of a GP register (the SysV argument/return register).
fn gp64(gp: &crate::pipeline::GpReg) -> Option<VarnodeId> {
    gp.for_bytes(8)
}

/// The argument registers and return register for `proto` under `abi`, or `None`
/// if any parameter (or the return) is an aggregate/unsupported type — such
/// functions are left unsigned rather than mapped incorrectly.
fn map_prototype(
    proto: &CFunctionProto,
    abi: &CallingConvention,
) -> Option<(Vec<VarnodeId>, Vec<VarnodeId>, Vec<ParamAttrs>)> {
    // An aggregate return uses a hidden pointer argument (memory class), which
    // would shift every argument. Rather than mis-map, skip the function.
    if matches!(proto.return_type, CType::Other | CType::Struct { .. }) {
        return None;
    }

    let mut inputs = Vec::with_capacity(proto.params.len());
    // Per-argument attributes, kept in lockstep with `inputs`. A `const`-qualified
    // pointer pointee (`const char *`) makes the argument `readonly`: the C
    // contract forbids the callee writing through it. Externs are never
    // `nocapture` (C `const` says nothing about capture — `strchr`/`tsearch`
    // retain their pointer). Non-pointer arguments carry no attribute.
    let mut attrs: Vec<ParamAttrs> = Vec::with_capacity(proto.params.len());
    let mut next_int = 0usize;
    let mut next_sse = 0usize;
    for param in &proto.params {
        match classify(&param.ty)? {
            Class::Integer => {
                // Out of GP registers: the rest are stack-passed, which we do
                // not model. Keep the register-passed prefix.
                let Some(gp) = abi.int_args.get(next_int) else {
                    break;
                };
                next_int += 1;
                let vn = gp64(gp)?;
                inputs.push(vn);
                attrs.push(ParamAttrs {
                    readonly: is_const_pointer(&param.ty),
                    nocapture: false,
                });
            }
            Class::Sse => {
                let Some(&vn) = abi.sse_args.get(next_sse) else {
                    break;
                };
                next_sse += 1;
                inputs.push(vn);
                attrs.push(ParamAttrs::default());
            }
        }
    }

    let outputs = match classify(&proto.return_type) {
        Some(Class::Integer) => abi.int_ret.as_ref().and_then(gp64).into_iter().collect(),
        Some(Class::Sse) => abi.sse_ret.into_iter().collect(),
        None => Vec::new(), // void: no return register
    };

    Some((inputs, outputs, attrs))
}

/// Plan the ordered call slots for `proto` under the calling convention selected
/// by `(abi, ptr_width, stack_only)`. `None` if any parameter is unplaceable (a
/// by-value aggregate), in which case the function is left with no interface.
///
/// * `stack_only` (32-bit `stdcall`/`cdecl`): every argument is a stack slot.
/// * otherwise (x64 System V): integers fill `abi.int_args`, floats fill
///   `abi.sse_args`, and the remainder overflow to the stack.
///
/// Stack slots are positional from the call-site stack pointer. On the
/// stack-only path the lifted `call` pushes the return address and decrements
/// the stack pointer to point *at* it, so `[SP+0]` is the return address: a
/// synthesized leading `return_address` argument occupies that slot and the
/// prototype's own parameters start one pointer-width above it. (Without this
/// the first parameter would alias the return address and the last would fall
/// off the end.) For System V register overflow the lifted `call` does not model
/// the push, so overflow arguments begin at offset 0.
///
/// Each planned argument also carries the prototype parameter name (for display),
/// its `const`-derived `readonly` attribute, and whether it is a pointer.
fn plan_args(
    proto: &CFunctionProto,
    abi: &CallingConvention,
    ptr_width: usize,
    stack_only: bool,
) -> Option<Vec<ExternArg>> {
    let mut args = Vec::with_capacity(proto.params.len() + usize::from(stack_only));
    let mut next_int = 0usize;
    let mut next_sse = 0usize;
    // On the stack-only path offset 0 is the return address (see above), so the
    // prototype's own stack arguments start one pointer-width above it.
    let mut next_stack = if stack_only { ptr_width as i64 } else { 0 };

    if stack_only {
        args.push(ExternArg {
            slot: ExternSlot::Stack {
                offset: 0,
                size: ptr_width,
            },
            name: Some("return_address".into()),
            attrs: ParamAttrs::default(),
        });
    }

    let take_stack = |next_stack: &mut i64| {
        let slot = ExternSlot::Stack {
            offset: *next_stack,
            size: ptr_width,
        };
        *next_stack += ptr_width as i64;
        slot
    };

    for param in &proto.params {
        let class = classify(&param.ty)?;
        let name = param.name.clone();
        let attrs = ParamAttrs {
            readonly: is_const_pointer(&param.ty),
            nocapture: false,
        };
        let slot = if stack_only {
            take_stack(&mut next_stack)
        } else {
            match class {
                Class::Integer => match abi.int_args.get(next_int).and_then(|g| g.for_bytes(8)) {
                    Some(vn) => {
                        next_int += 1;
                        ExternSlot::Reg(vn, ptr_width)
                    }
                    None => take_stack(&mut next_stack),
                },
                Class::Sse => match abi.sse_args.get(next_sse).copied() {
                    Some(vn) => {
                        next_sse += 1;
                        ExternSlot::Reg(vn, ptr_width)
                    }
                    None => take_stack(&mut next_stack),
                },
            }
        };
        args.push(ExternArg { slot, name, attrs });
    }
    Some(args)
}

/// Whether `ty` is a pointer to a `const`-qualified pointee (`const char *`),
/// the C signal that the callee will not write through the pointer.
fn is_const_pointer(ty: &CType) -> bool {
    matches!(
        ty,
        CType::Pointer {
            const_pointee: true,
            ..
        }
    )
}

/// Assign a signature and call interface to `fun_id` if it is a known external
/// function present in `sel`.
pub fn apply_external_signature(
    ctx: &mut Context,
    fun_id: FunctionId,
    abi: &CallingConvention,
    sel: &Selection,
    ptr_width: usize,
    stack_only: bool,
) {
    if !abi_is_known(abi) {
        return; // no convention for this architecture
    }
    if !FunctionBody::from_id(ctx, fun_id).is_external() {
        return;
    }
    // Import stubs are named like `printf@plt` or `printf@GLIBC_2.2.5`; the C
    // symbol is the part before the first `@`.
    let raw = FunctionBody::from_id(ctx, fun_id).name().to_string();
    let name = raw.split('@').next().unwrap_or(&raw);
    let Some(proto) = sel.lookup(name) else {
        return;
    };
    let Some((inputs, outputs, param_attrs)) = map_prototype(proto, abi) else {
        return;
    };
    let plan = plan_args(proto, abi, ptr_width, stack_only);

    // The register-channel interface mapping (argpromote v2, ruling 6a): this
    // materialized external's inputs are its argument registers and its outputs
    // are its return register(s) ∪ the ABI caller-saved clobber set. Growing the
    // outputs to include the clobbers is what lets a materialized caller's return
    // pack cover them (bug-2-external). No varargs check — a prototyped variadic
    // external materializes like any other. This is the single source of truth:
    // `reg_summary::external_leaf` reads it, and pass 2 never grows an external
    // branch.
    let mut pack_outputs: Vec<VarnodeId> = outputs.clone();
    for &clobber in &abi.caller_saved {
        if !pack_outputs.contains(&clobber) {
            pack_outputs.push(clobber);
        }
    }
    let reg_map = RegisterInterfaceMap {
        inputs: inputs.clone(),
        outputs: pack_outputs,
    };

    let mut f = FunctionBody::from_id_mut(ctx, fun_id);
    // Legacy ABI register list, kept for the external/conventional path this
    // function serves (a C prototype); pure_reg callees use block params instead.
    #[allow(deprecated)]
    f.set_input_regs(inputs);
    f.set_output_regs(outputs);
    f.set_param_attrs(param_attrs);
    f.set_effects(FunctionEffects::Materialized(reg_map));
    // The prototype fully describes this callee's register effect: its inputs are
    // the arguments the caller passes (a register reload, or — for stdcall/cdecl
    // — a stack load supplied by `argpromote_external`), and its writes are the
    // convention's caller-saved (volatile) registers. Marking it resolved lets
    // the value passes drop the conservative "reads/writes every register"
    // fallback and treat the call precisely. See
    // [`FunctionSignature::externally_resolved`].
    f.set_clobbered_regs(abi.caller_saved.clone());
    f.set_externally_resolved(true);

    // The materialized call interface consumed by `argpromote_external`: the
    // ordered call slots and the per-slot display names. Only set when the
    // arguments are all placeable (a by-value aggregate leaves it absent).
    if let Some(args) = plan {
        f.set_input_arg_names(args.iter().map(|a| a.name.clone()).collect());
        f.set_extern_interface(ExternInterface { args });
    }
}

/// Whether `abi` carries enough of a convention to resolve external callees: it
/// either passes integer arguments in registers (x64 System V) or, for a
/// stack-only convention (x86 stdcall/cdecl), at least names its caller-saved
/// registers so a resolved call has a clobber set.
fn abi_is_known(abi: &CallingConvention) -> bool {
    !abi.int_args.is_empty() || !abi.caller_saved.is_empty()
}

/// Apply [`apply_external_signature`] to every external function, building the
/// selection from `ctx`'s linked-library list (no live binary handle).
pub fn apply_all_external_signatures(ctx: &mut Context, abi: &CallingConvention, sel: &Selection) {
    if !abi_is_known(abi) {
        return;
    }
    // Pointer width / stack-only mode are unknown without a `PipelineEnv`; derive
    // them from the selection's target so callers without an env still get a
    // consistent interface.
    let ptr_width = (sel_bits(sel) / 8).max(1) as usize;
    let stack_only = sel_bits(sel) == 32;
    let ids: Vec<FunctionId> = ctx
        .functions()
        .filter(|f| f.is_external())
        .map(|f| f.id)
        .collect();
    apply_external_signatures(ctx, &ids, abi, sel, ptr_width, stack_only);
}

/// The pointer width (in bits) the selection was built for.
fn sel_bits(sel: &Selection) -> u8 {
    sel.target().bits
}

fn apply_external_signatures(
    ctx: &mut Context,
    ids: &[FunctionId],
    abi: &CallingConvention,
    sel: &Selection,
    ptr_width: usize,
    stack_only: bool,
) {
    for id in ids.iter().copied() {
        apply_external_signature(ctx, id, abi, sel, ptr_width, stack_only);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::GpReg;
    use cabi::{AbiTarget, CParam, Platform};
    use qcode::{
        testing::TestContext,
        value::{FunctionBody, VarnodeId},
    };

    /// A toy convention over the TestContext registers: two GP arg registers
    /// (r0, r1) at a single 8-byte width, one SSE arg (r2), int return r3.
    fn toy_abi(tc: &TestContext) -> CallingConvention {
        let gp = |vn: VarnodeId| GpReg {
            widths: vec![(8, vn)],
        };
        CallingConvention {
            int_args: vec![gp(tc.r0), gp(tc.r1)],
            sse_args: vec![tc.r2],
            int_ret: Some(gp(tc.r3)),
            sse_ret: Some(tc.r2),
            caller_saved: vec![tc.r3],
        }
    }

    fn external(tc: &mut TestContext, name: &str) -> FunctionId {
        FunctionBody::make_external(&mut tc.ctx, 0x9000, Some(name.to_string().into())).id
    }

    /// The host target the embedded libc table was extracted for.
    fn host_target() -> AbiTarget {
        let platform = if cfg!(windows) {
            Platform::Windows
        } else {
            Platform::Linux
        };
        AbiTarget::new(platform, 64)
    }

    /// A selection that resolves the builtin host libc table (opt-in: libc is in
    /// the linked list), so the tests exercise the real embedded prototypes.
    fn host_sel() -> Selection {
        cabi::select(&Config::baked(), host_target(), &["libc.so.6".to_string()])
    }

    /// An empty selection (nothing linked): the strict default.
    fn empty_sel() -> Selection {
        cabi::select(&Config::baked(), host_target(), &[])
    }

    fn apply(ctx: &mut Context, f: FunctionId, abi: &CallingConvention, sel: &Selection) {
        apply_external_signature(ctx, f, abi, sel, 8, false);
    }

    #[test]
    fn integer_and_pointer_params_take_gp_registers() {
        // memcpy(void*, const void*, size_t) -> three INTEGER-class args.
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        let f = external(&mut tc, "memcpy");
        apply(&mut tc.ctx, f, &abi, &host_sel());
        // Only two GP registers exist in the toy ABI; the third arg is stack.
        let inputs = FunctionBody::from_id(&tc.ctx, f).input_regs().unwrap();
        assert_eq!(inputs, [tc.r0, tc.r1]);
        // memcpy returns void* -> the integer return register.
        let outputs = FunctionBody::from_id(&tc.ctx, f)
            .signature()
            .unwrap()
            .outputs
            .clone()
            .unwrap();
        assert_eq!(outputs, vec![tc.r3]);
    }

    #[test]
    fn const_pointer_param_is_readonly() {
        // memcpy(void *dst, const void *src, size_t) — the second pointer is
        // `const`-qualified, so its argument is readonly; the first is not.
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        let f = external(&mut tc, "memcpy");
        apply(&mut tc.ctx, f, &abi, &host_sel());
        let func = FunctionBody::from_id(&tc.ctx, f);
        assert!(
            !func.param_attr(0).unwrap().readonly,
            "dst is written through → not readonly"
        );
        assert!(
            func.param_attr(1).unwrap().readonly,
            "const src is never written through → readonly"
        );
        assert!(
            !func.param_attr(1).unwrap().nocapture,
            "externs are never nocapture (C const says nothing about capture)"
        );
    }

    #[test]
    fn float_params_take_sse_registers() {
        // pow(double, double) -> first goes to the SSE register; the second has
        // no SSE register in the toy ABI, so it is dropped (stack).
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        let f = external(&mut tc, "pow");
        apply(&mut tc.ctx, f, &abi, &host_sel());
        let inputs = FunctionBody::from_id(&tc.ctx, f).input_regs().unwrap();
        assert_eq!(inputs, [tc.r2]);
    }

    #[test]
    fn unknown_function_is_left_unsigned() {
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        let f = external(&mut tc, "definitely_not_a_libc_function_xyz");
        apply(&mut tc.ctx, f, &abi, &host_sel());
        assert!(FunctionBody::from_id(&tc.ctx, f).signature().is_none());
    }

    #[test]
    fn empty_abi_is_a_noop() {
        let mut tc = TestContext::new();
        let f = external(&mut tc, "memcpy");
        apply(&mut tc.ctx, f, &CallingConvention::default(), &host_sel());
        assert!(FunctionBody::from_id(&tc.ctx, f).signature().is_none());
    }

    /// The strict default: a binary whose linked list lacks libc gets no libc
    /// signatures, even for a symbol that is in the embedded builtin table.
    #[test]
    fn empty_selection_signs_nothing() {
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        let f = external(&mut tc, "memcpy");
        apply(&mut tc.ctx, f, &abi, &empty_sel());
        assert!(
            FunctionBody::from_id(&tc.ctx, f).signature().is_none(),
            "no libc in the linked list → no libc signature"
        );
    }

    /// A prototyped external is stamped `Materialized` (ruling 6a): its output
    /// mapping is the return register(s) ∪ the ABI caller-saved clobbers, so a
    /// materialized caller's return pack can cover them (bug-2-external).
    #[test]
    fn prototyped_external_stamps_materialized_with_clobbers() {
        use qcode::value::FunctionEffects;
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc); // caller_saved = [r3], int_ret = r3
        let f = external(&mut tc, "memcpy");
        apply(&mut tc.ctx, f, &abi, &host_sel());
        let func = FunctionBody::from_id(&tc.ctx, f);
        let FunctionEffects::Materialized(map) = func.effects() else {
            panic!("prototyped external must be stamped Materialized");
        };
        // memcpy's two register args (r0, r1) are the inputs.
        assert_eq!(map.inputs, vec![tc.r0, tc.r1]);
        // Outputs = return (r3) ∪ caller_saved (r3), deduped to just r3.
        assert!(
            map.outputs.contains(&tc.r3),
            "the clobber/return register must appear in the output pack"
        );
    }

    /// Materialized [`ExternInterface`] round-trips through the signature and the
    /// planned stdcall slots include the synthesized return-address slot.
    #[test]
    fn materializes_extern_interface() {
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        let f = external(&mut tc, "memcpy");
        // 32-bit stack-only planning: return-address slot then three stack args.
        apply_external_signature(&mut tc.ctx, f, &abi, &host_sel(), 4, true);
        let func = FunctionBody::from_id(&tc.ctx, f);
        let iface = func.extern_interface().expect("interface materialized");
        assert_eq!(iface.args[0].slot, ExternSlot::Stack { offset: 0, size: 4 });
        assert_eq!(iface.args[0].name.as_deref(), Some("return_address"));
        // memcpy has three parameters, so four slots total.
        assert_eq!(iface.args.len(), 4);
    }

    /// The materialized interface survives a context serde round-trip (the
    /// `.harbinger` snapshot path), so a reloaded snapshot needs neither the
    /// binary nor cabi for `argpromote_external` to run.
    #[test]
    fn extern_interface_survives_serde_round_trip() {
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        let f = external(&mut tc, "memcpy");
        apply_external_signature(&mut tc.ctx, f, &abi, &host_sel(), 4, true);
        let before = FunctionBody::from_id(&tc.ctx, f)
            .extern_interface()
            .expect("interface materialized")
            .clone();

        let config = bincode::config::standard();
        let encoded = bincode::serde::encode_to_vec(&tc.ctx, config).expect("context serializes");
        let (restored, _): (Context<'static>, usize) =
            bincode::serde::decode_from_slice(&encoded, config).expect("context deserializes");
        let f2 = FunctionBody::from_name(&restored, "memcpy")
            .expect("function survives")
            .id;
        assert_eq!(
            FunctionBody::from_id(&restored, f2).extern_interface(),
            Some(&before),
            "ExternInterface must survive the snapshot round-trip"
        );
    }

    // --- plan_args (ported verbatim from argpromote/external.rs) --------------

    fn proto(params: Vec<CType>) -> CFunctionProto {
        CFunctionProto {
            name: "f".into(),
            return_type: CType::Integer {
                bytes: 4,
                signed: true,
            },
            params: params
                .into_iter()
                .map(|ty| CParam { name: None, ty })
                .collect(),
            variadic: false,
            header: None,
        }
    }

    fn ptr() -> CType {
        CType::Pointer {
            pointee: Box::new(CType::Void),
            const_pointee: false,
        }
    }

    fn int() -> CType {
        CType::Integer {
            bytes: 4,
            signed: true,
        }
    }

    /// stdcall: a synthesized `return_address` slot occupies `[SP+0]` and the
    /// prototype's own arguments are positional 4-byte stack slots starting one
    /// pointer-width above it (matching SHGetSpecialFolderPathW: 4 args, so 5
    /// planned slots — the return address would otherwise be mistaken for the
    /// first argument and the last argument dropped).
    #[test]
    fn stdcall_places_return_address_then_all_args_on_the_stack() {
        let p = proto(vec![ptr(), ptr(), int(), int()]);
        let abi = CallingConvention::default();
        let plan = plan_args(&p, &abi, 4, true).expect("placeable");
        let slots: Vec<_> = plan.iter().map(|a| a.slot).collect();
        assert_eq!(
            slots,
            vec![
                ExternSlot::Stack { offset: 0, size: 4 },
                ExternSlot::Stack { offset: 4, size: 4 },
                ExternSlot::Stack { offset: 8, size: 4 },
                ExternSlot::Stack {
                    offset: 12,
                    size: 4
                },
                ExternSlot::Stack {
                    offset: 16,
                    size: 4
                },
            ]
        );
    }

    /// Prototype parameter names ride through onto the planned arguments.
    #[test]
    fn stdcall_carries_prototype_parameter_names() {
        let mut p = proto(vec![ptr(), int()]);
        p.params[0].name = Some("pszPath".into());
        p.params[1].name = Some("csidl".into());
        let plan = plan_args(&p, &CallingConvention::default(), 4, true).expect("placeable");
        let names: Vec<_> = plan.iter().map(|a| a.name.as_deref()).collect();
        assert_eq!(
            names,
            vec![Some("return_address"), Some("pszPath"), Some("csidl")]
        );
    }

    /// A by-value aggregate parameter is unplaceable, so the whole function is
    /// skipped rather than mis-bound.
    #[test]
    fn aggregate_param_skips_the_function() {
        let p = proto(vec![int(), CType::Struct { name: None }]);
        assert!(plan_args(&p, &CallingConvention::default(), 4, true).is_none());
    }
}
