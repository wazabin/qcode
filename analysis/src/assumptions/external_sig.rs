//! Signatures for known external (libc) functions, from C prototypes.
//!
//! External functions are bodyless stubs, so [`call_summary`](crate::calls)
//! cannot infer their inputs. Instead we look the function name up in the
//! build-time [`cabi`] prototype database and assign argument/return registers
//! using the System V calling convention supplied by [`CallingConvention`].
//!
//! Run before [`set_all_function_summaries`](crate::set_all_function_summaries):
//! once an external callee has a signature, the argument-producing passes can
//! produce real arguments at its call sites.

use cabi::{AbiTarget, CFunctionProto, CType};
use qcode::{
    context::{Context, TargetOs},
    value::{FunctionBody, FunctionId, ParamAttrs, VarnodeId},
};

use crate::pipeline::CallingConvention;

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
        let target = abi_target(ctx, env);
        apply_external_signatures(ctx, &affected, &env.cfg.abi, target);
        Ok(crate::ModulePassOutcome::functions(affected)
            .preserving_global::<crate::CallGraphAnalysis>())
    }
}

/// The cabi table to consult for this binary: its platform (from the container
/// format) and pointer width.
fn abi_target(ctx: &Context, env: &PipelineEnv) -> AbiTarget {
    let platform = match ctx.target_os() {
        TargetOs::Windows => cabi::Platform::Windows,
        // ELF/unknown binaries use the SysV/libc (host) table.
        TargetOs::Linux | TargetOs::Unknown => cabi::Platform::Linux,
    };
    AbiTarget::new(platform, env.cfg.bitness)
}

crate::register_module_pass!(ExternalSigs);

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

/// Assign a signature to `fun_id` if it is a known external function.
pub fn apply_external_signature(
    ctx: &mut Context,
    fun_id: FunctionId,
    abi: &CallingConvention,
    target: AbiTarget,
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
    let Some(proto) = cabi::lookup(target, name) else {
        return;
    };
    let Some((inputs, outputs, param_attrs)) = map_prototype(proto, abi) else {
        return;
    };

    let mut f = FunctionBody::from_id_mut(ctx, fun_id);
    // Legacy ABI register list, kept for the external/conventional path this
    // function serves (a C prototype); pure_reg callees use block params instead.
    #[allow(deprecated)]
    f.set_input_regs(inputs);
    f.set_output_regs(outputs);
    f.set_param_attrs(param_attrs);
    // The prototype fully describes this callee's register effect: its inputs are
    // the arguments the caller passes (a register reload, or — for stdcall/cdecl
    // — a stack load supplied by `argpromote_external`), and its writes are the
    // convention's caller-saved (volatile) registers. Marking it resolved lets
    // the value passes drop the conservative "reads/writes every register"
    // fallback and treat the call precisely. See
    // [`FunctionSignature::externally_resolved`].
    f.set_clobbered_regs(abi.caller_saved.clone());
    f.set_externally_resolved(true);
}

/// Whether `abi` carries enough of a convention to resolve external callees: it
/// either passes integer arguments in registers (x64 System V) or, for a
/// stack-only convention (x86 stdcall/cdecl), at least names its caller-saved
/// registers so a resolved call has a clobber set.
fn abi_is_known(abi: &CallingConvention) -> bool {
    !abi.int_args.is_empty() || !abi.caller_saved.is_empty()
}

/// Apply [`apply_external_signature`] to every external function.
pub fn apply_all_external_signatures(
    ctx: &mut Context,
    abi: &CallingConvention,
    target: AbiTarget,
) {
    if !abi_is_known(abi) {
        return;
    }
    let ids: Vec<FunctionId> = ctx
        .functions()
        .filter(|f| f.is_external())
        .map(|f| f.id)
        .collect();
    apply_external_signatures(ctx, &ids, abi, target);
}

fn apply_external_signatures(
    ctx: &mut Context,
    ids: &[FunctionId],
    abi: &CallingConvention,
    target: AbiTarget,
) {
    for id in ids.iter().copied() {
        apply_external_signature(ctx, id, abi, target);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::GpReg;
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

    /// The cabi table extracted for this build host (where the libc symbols the
    /// tests look up actually live). The toy ABI is 64-bit.
    fn host() -> AbiTarget {
        let platform = if cfg!(windows) {
            cabi::Platform::Windows
        } else {
            cabi::Platform::Linux
        };
        AbiTarget::new(platform, 64)
    }

    #[test]
    fn integer_and_pointer_params_take_gp_registers() {
        // memcpy(void*, const void*, size_t) -> three INTEGER-class args.
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        let f = external(&mut tc, "memcpy");
        apply_external_signature(&mut tc.ctx, f, &abi, host());
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
        apply_external_signature(&mut tc.ctx, f, &abi, host());
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
        apply_external_signature(&mut tc.ctx, f, &abi, host());
        let inputs = FunctionBody::from_id(&tc.ctx, f).input_regs().unwrap();
        assert_eq!(inputs, [tc.r2]);
    }

    #[test]
    fn unknown_function_is_left_unsigned() {
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        let f = external(&mut tc, "definitely_not_a_libc_function_xyz");
        apply_external_signature(&mut tc.ctx, f, &abi, host());
        assert!(FunctionBody::from_id(&tc.ctx, f).signature().is_none());
    }

    #[test]
    fn empty_abi_is_a_noop() {
        let mut tc = TestContext::new();
        let f = external(&mut tc, "memcpy");
        apply_external_signature(&mut tc.ctx, f, &CallingConvention::default(), host());
        assert!(FunctionBody::from_id(&tc.ctx, f).signature().is_none());
    }
}
