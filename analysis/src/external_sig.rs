//! Signatures for known external (libc) functions, from C prototypes.
//!
//! External functions are bodyless stubs, so [`call_summary`](crate::call_summary)
//! cannot infer their inputs. Instead we look the function name up in the
//! build-time [`cabi`] prototype database and assign argument/return registers
//! using the System V calling convention supplied by [`CallingConvention`].
//!
//! Run before [`set_all_function_summaries`](crate::set_all_function_summaries)
//! and [`bind_all_call_args`](crate::bind_all_call_args): once an external
//! callee has a signature, the existing binding pass produces real arguments at
//! its call sites.

use cabi::{CFunctionProto, CType};
use qcode::{
    context::Context,
    value::{Function, FunctionId, VarnodeId},
};

use crate::pipeline::CallingConvention;

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
        CType::Integer { .. } | CType::Pointer => Some(Class::Integer),
        CType::Float { .. } => Some(Class::Sse),
        CType::Void | CType::Other => None,
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
) -> Option<(Vec<VarnodeId>, Vec<VarnodeId>)> {
    // An aggregate return uses a hidden pointer argument (memory class), which
    // would shift every argument. Rather than mis-map, skip the function.
    if matches!(proto.return_type, CType::Other) {
        return None;
    }

    let mut inputs = Vec::with_capacity(proto.params.len());
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
                let Some(vn) = gp64(gp) else {
                    return None;
                };
                inputs.push(vn);
            }
            Class::Sse => {
                let Some(&vn) = abi.sse_args.get(next_sse) else {
                    break;
                };
                next_sse += 1;
                inputs.push(vn);
            }
        }
    }

    let outputs = match classify(&proto.return_type) {
        Some(Class::Integer) => abi.int_ret.as_ref().and_then(gp64).into_iter().collect(),
        Some(Class::Sse) => abi.sse_ret.into_iter().collect(),
        None => Vec::new(), // void: no return register
    };

    Some((inputs, outputs))
}

/// Assign a signature to `fun_id` if it is a known external function.
pub fn apply_external_signature(ctx: &mut Context, fun_id: FunctionId, abi: &CallingConvention) {
    if abi.int_args.is_empty() {
        return; // no convention for this architecture
    }
    if !Function::from_id(ctx, fun_id).is_external() {
        return;
    }
    // Import stubs are named like `printf@plt` or `printf@GLIBC_2.2.5`; the C
    // symbol is the part before the first `@`.
    let raw = Function::from_id(ctx, fun_id).name().to_string();
    let name = raw.split('@').next().unwrap_or(&raw);
    let Some(proto) = cabi::lookup(name) else {
        return;
    };
    let Some((inputs, outputs)) = map_prototype(proto, abi) else {
        return;
    };

    let mut f = Function::from_id_mut(ctx, fun_id);
    f.set_input_regs(inputs);
    f.set_output_regs(outputs);
}

/// Apply [`apply_external_signature`] to every external function.
pub fn apply_all_external_signatures(ctx: &mut Context, abi: &CallingConvention) {
    if abi.int_args.is_empty() {
        return;
    }
    let ids: Vec<FunctionId> = ctx
        .functions()
        .filter(|f| f.is_external())
        .map(|f| f.id)
        .collect();
    for id in ids {
        apply_external_signature(ctx, id, abi);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::GpReg;
    use qcode::{
        testing::TestContext,
        value::{Function, VarnodeId},
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
        }
    }

    fn external(tc: &mut TestContext, name: &str) -> FunctionId {
        Function::make_external(&mut tc.ctx, 0x9000, Some(name.to_string().into())).id
    }

    #[test]
    fn integer_and_pointer_params_take_gp_registers() {
        // memcpy(void*, const void*, size_t) -> three INTEGER-class args.
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        let f = external(&mut tc, "memcpy");
        apply_external_signature(&mut tc.ctx, f, &abi);
        // Only two GP registers exist in the toy ABI; the third arg is stack.
        let inputs = Function::from_id(&tc.ctx, f).input_regs().unwrap();
        assert_eq!(inputs, [tc.r0, tc.r1]);
        // memcpy returns void* -> the integer return register.
        let outputs = Function::from_id(&tc.ctx, f)
            .signature()
            .unwrap()
            .outputs
            .clone()
            .unwrap();
        assert_eq!(outputs, vec![tc.r3]);
    }

    #[test]
    fn float_params_take_sse_registers() {
        // pow(double, double) -> first goes to the SSE register; the second has
        // no SSE register in the toy ABI, so it is dropped (stack).
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        let f = external(&mut tc, "pow");
        apply_external_signature(&mut tc.ctx, f, &abi);
        let inputs = Function::from_id(&tc.ctx, f).input_regs().unwrap();
        assert_eq!(inputs, [tc.r2]);
    }

    #[test]
    fn unknown_function_is_left_unsigned() {
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        let f = external(&mut tc, "definitely_not_a_libc_function_xyz");
        apply_external_signature(&mut tc.ctx, f, &abi);
        assert!(Function::from_id(&tc.ctx, f).signature().is_none());
    }

    #[test]
    fn empty_abi_is_a_noop() {
        let mut tc = TestContext::new();
        let f = external(&mut tc, "memcpy");
        apply_external_signature(&mut tc.ctx, f, &CallingConvention::default());
        assert!(Function::from_id(&tc.ctx, f).signature().is_none());
    }
}
