//! External-call argument channel: resolve the arguments at call sites to
//! bodyless **external** (imported) functions from their C prototype.
//!
//! [`argpromote_registers`](super::argpromote_registers) and the RAM channel
//! functionalize *bodied* callees and, in doing so, thread each caller's
//! `Call.args`. An external stub has no body, so neither runs and its calls
//! render argument-less (`call fn SHGetSpecialFolderPathW()`). This channel
//! fills that gap for any external callee whose prototype is known to [`cabi`]:
//! for each one it appends the positional `Call.args` at every direct caller —
//! a register reload (System V integer/SSE args) or an SP-relative stack load
//! (32-bit `stdcall`/`cdecl`, and System V register overflow) — and leaves the
//! following gvn round to forward each load to the value the caller set up
//! (a register write, or the `push` that stored the stack slot).
//!
//! It only ever *adds* arguments to a call; the bodyless callee is never
//! touched. The `args.len() == idx` guard keeps it idempotent across pipeline
//! rounds and aligned `arg[i] ↔ param i`.

use qcode::{
    builder::Builder,
    context::Context,
    space::SpaceId,
    value::{BasicBlock, Function, FunctionId, Value, ValueId, Varnode, VarnodeId, insn::Mnemonic},
};

use cabi::{AbiTarget, CType, Platform};

use super::super::append_caller_arg;
use crate::pipeline::CallingConvention;
use crate::{Pass, PipelineEnv};

/// System V scalar class of a parameter (the only two we can place).
#[derive(Clone, Copy)]
enum Class {
    /// Integer/pointer — a general-purpose register, else the stack.
    Integer,
    /// Float/double — an SSE register, else the stack.
    Sse,
}

/// Classify a scalar parameter type, or `None` for a by-value aggregate / type
/// we cannot place (which forces the whole function to be skipped).
fn classify(ty: &CType) -> Option<Class> {
    match ty {
        CType::Integer { .. } | CType::Pointer { .. } => Some(Class::Integer),
        CType::Float { .. } => Some(Class::Sse),
        CType::Void | CType::Struct { .. } | CType::Other => None,
    }
}

/// How one argument is loaded at the call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    /// A register argument: reload the register (`size` bytes) live at the call.
    Reg { vn: VarnodeId, size: usize },
    /// A stack argument at `offset` bytes above the call-site stack pointer.
    Stack { offset: i64, size: usize },
}

/// Plan the argument slots for `proto` under the calling convention selected by
/// `(abi, ptr_width, stack_only)`. `None` if any parameter is unplaceable (a
/// by-value aggregate), in which case the function is left unresolved.
///
/// * `stack_only` (32-bit `stdcall`/`cdecl`): every argument is a stack slot.
/// * otherwise (x64 System V): integers fill `abi.int_args`, floats fill
///   `abi.sse_args`, and the remainder overflow to the stack.
///
/// Stack slots are positional from the call-site stack pointer: the first stack
/// argument sits at offset 0 (the lifted `call` does not model the return-address
/// push as an SP-register decrement), the next a pointer-width above it, etc.
fn plan_args(
    proto: &cabi::CFunctionProto,
    abi: &CallingConvention,
    ptr_width: usize,
    stack_only: bool,
) -> Option<Vec<Slot>> {
    let mut slots = Vec::with_capacity(proto.params.len());
    let mut next_int = 0usize;
    let mut next_sse = 0usize;
    let mut next_stack = 0i64;

    let push_stack = |slots: &mut Vec<Slot>, next_stack: &mut i64| {
        slots.push(Slot::Stack {
            offset: *next_stack,
            size: ptr_width,
        });
        *next_stack += ptr_width as i64;
    };

    for param in &proto.params {
        let class = classify(&param.ty)?;
        if stack_only {
            push_stack(&mut slots, &mut next_stack);
            continue;
        }
        match class {
            Class::Integer => match abi.int_args.get(next_int).and_then(|g| g.for_bytes(8)) {
                Some(vn) => {
                    next_int += 1;
                    slots.push(Slot::Reg { vn, size: ptr_width });
                }
                None => push_stack(&mut slots, &mut next_stack),
            },
            Class::Sse => match abi.sse_args.get(next_sse).copied() {
                Some(vn) => {
                    next_sse += 1;
                    slots.push(Slot::Reg { vn, size: ptr_width });
                }
                None => push_stack(&mut slots, &mut next_stack),
            },
        }
    }
    Some(slots)
}

/// Append the resolved `Call.args` at every direct caller of each external
/// function with a known prototype. Returns `true` if any call was changed.
pub fn argpromote_external(ctx: &mut Context, env: &PipelineEnv) -> bool {
    let platform = match env.cfg.os {
        qcode::context::TargetOs::Windows => Platform::Windows,
        _ => Platform::Linux,
    };
    let target = AbiTarget::new(platform, env.cfg.bitness);
    let ptr_width = (env.cfg.bitness / 8).max(1) as usize;
    let stack_only = env.cfg.bitness == 32;
    let sp = env.sp_varnode;
    let sp_space = Varnode::from_id(ctx, sp).space().id;
    let default_space = ctx.default_space;

    let externals: Vec<FunctionId> = ctx
        .functions()
        .filter(|f| f.is_external())
        .map(|f| f.id)
        .collect();

    let mut changed = false;
    for fid in externals {
        let raw = Function::from_id(ctx, fid).name().to_string();
        let sym = raw.split('@').next().unwrap_or(&raw);
        let Some(proto) = cabi::lookup(target, sym) else {
            continue;
        };
        let Some(plan) = plan_args(proto, &env.cfg.abi, ptr_width, stack_only) else {
            continue;
        };
        if bind_external_args(ctx, fid, &plan, sp, sp_space, default_space, ptr_width) {
            changed = true;
        }
    }
    changed
}

/// Thread `plan` through every direct caller of `fid`, one positional argument at
/// a time and in order, so the `args.len() == idx` guard keeps each call's
/// arguments aligned and the pass idempotent.
fn bind_external_args(
    ctx: &mut Context,
    fid: FunctionId,
    plan: &[Slot],
    sp: VarnodeId,
    sp_space: SpaceId,
    default_space: SpaceId,
    ptr_width: usize,
) -> bool {
    let mut changed = false;
    for (idx, &slot) in plan.iter().enumerate() {
        changed |= append_caller_arg(ctx, fid, |ctx, call_id, block| {
            let Mnemonic::Call(c) = ctx.get_insn(call_id).mnemonic() else {
                return None;
            };
            // Only the call whose next positional slot is exactly this one, so
            // arguments land in order and an already-bound call is skipped.
            if c.args.len() != idx {
                return None;
            }
            let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, block));
            b.set_insert_point_before(call_id);
            let value = match slot {
                Slot::Reg { vn, size, .. } => {
                    let space = Varnode::from_id(b.context(), vn).space().id;
                    b.push_load::<false>(ValueId::Varnode(vn), size, space).id()
                }
                Slot::Stack { offset, size } => {
                    let sp_val = b
                        .push_load::<false>(ValueId::Varnode(sp), ptr_width, sp_space)
                        .id();
                    let addr = if offset == 0 {
                        sp_val
                    } else {
                        let off = b.context_mut().get_const(offset as u64, ptr_width).id();
                        b.push_add(sp_val, off).id()
                    };
                    b.push_load::<false>(addr, size, default_space).id()
                }
            };
            Some(value)
        });
    }
    changed
}

#[derive(Default)]
pub struct ArgPromoteExternal;

impl Pass for ArgPromoteExternal {
    const NAME: &'static str = "argpromote_external";
    fn description(&self) -> &'static str {
        "Resolve call arguments to external functions from their C prototype"
    }
    fn run(&self, ctx: &mut Context, env: &PipelineEnv) -> Result<bool, String> {
        Ok(argpromote_external(ctx, env))
    }
}

crate::register_module_pass!(ArgPromoteExternal);

#[cfg(test)]
mod tests {
    use super::*;
    use cabi::{CFunctionProto, CParam};

    fn proto(params: Vec<CType>) -> CFunctionProto {
        CFunctionProto {
            name: "f".into(),
            return_type: CType::Integer { bytes: 4, signed: true },
            params: params
                .into_iter()
                .map(|ty| CParam { name: None, ty })
                .collect(),
            variadic: false,
            header: None,
        }
    }

    fn ptr() -> CType {
        CType::Pointer { pointee: Box::new(CType::Void) }
    }

    fn int() -> CType {
        CType::Integer { bytes: 4, signed: true }
    }

    /// stdcall: every argument is a positional 4-byte stack slot, SP-relative,
    /// the first at offset 0 (matching SHGetSpecialFolderPathW: 4 args).
    #[test]
    fn stdcall_places_all_args_on_the_stack() {
        let p = proto(vec![ptr(), ptr(), int(), int()]);
        let abi = CallingConvention::default();
        let slots = plan_args(&p, &abi, 4, true).expect("placeable");
        assert_eq!(
            slots,
            vec![
                Slot::Stack { offset: 0, size: 4 },
                Slot::Stack { offset: 4, size: 4 },
                Slot::Stack { offset: 8, size: 4 },
                Slot::Stack { offset: 12, size: 4 },
            ]
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
