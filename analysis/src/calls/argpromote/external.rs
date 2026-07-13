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
    value::{
        BasicBlock, FunctionBody, FunctionId, Instruction, InstructionId, Value, ValueId, Varnode,
        VarnodeId, insn::Mnemonic,
    },
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

/// One planned positional argument: where to load it from, and the display name
/// it should carry at the call site (the prototype parameter name, or
/// `return_address` for the synthesized stdcall slot).
#[derive(Clone, Debug, PartialEq, Eq)]
struct PlannedArg {
    slot: Slot,
    name: Option<Box<str>>,
}

/// Plan the argument slots for `proto` under the calling convention selected by
/// `(abi, ptr_width, stack_only)`. `None` if any parameter is unplaceable (a
/// by-value aggregate), in which case the function is left unresolved.
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
/// Each planned argument also carries the prototype parameter name for display.
fn plan_args(
    proto: &cabi::CFunctionProto,
    abi: &CallingConvention,
    ptr_width: usize,
    stack_only: bool,
) -> Option<Vec<PlannedArg>> {
    let mut args = Vec::with_capacity(proto.params.len() + usize::from(stack_only));
    let mut next_int = 0usize;
    let mut next_sse = 0usize;
    // On the stack-only path offset 0 is the return address (see below), so the
    // prototype's own stack arguments start one pointer-width above it.
    let mut next_stack = if stack_only { ptr_width as i64 } else { 0 };

    if stack_only {
        args.push(PlannedArg {
            slot: Slot::Stack {
                offset: 0,
                size: ptr_width,
            },
            name: Some("return_address".into()),
        });
    }

    let push_stack = |args: &mut Vec<PlannedArg>, next_stack: &mut i64, name: Option<Box<str>>| {
        args.push(PlannedArg {
            slot: Slot::Stack {
                offset: *next_stack,
                size: ptr_width,
            },
            name,
        });
        *next_stack += ptr_width as i64;
    };

    for param in &proto.params {
        let class = classify(&param.ty)?;
        let name = param.name.clone();
        if stack_only {
            push_stack(&mut args, &mut next_stack, name);
            continue;
        }
        match class {
            Class::Integer => match abi.int_args.get(next_int).and_then(|g| g.for_bytes(8)) {
                Some(vn) => {
                    next_int += 1;
                    args.push(PlannedArg {
                        slot: Slot::Reg {
                            vn,
                            size: ptr_width,
                        },
                        name,
                    });
                }
                None => push_stack(&mut args, &mut next_stack, name),
            },
            Class::Sse => match abi.sse_args.get(next_sse).copied() {
                Some(vn) => {
                    next_sse += 1;
                    args.push(PlannedArg {
                        slot: Slot::Reg {
                            vn,
                            size: ptr_width,
                        },
                        name,
                    });
                }
                None => push_stack(&mut args, &mut next_stack, name),
            },
        }
    }
    Some(args)
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
    let sp_space = Varnode::from_id(&*ctx, sp).space().id;
    let default_space = ctx.shared.default_space;

    // Only externals that are actually *called* can gain arguments: both
    // `bind_external_args` and `bind_external_return` key on a direct `Call` whose
    // target is the external. Filtering on the direct call-site index (an O(1) map
    // lookup) skips the `cabi::lookup` and the whole-program instruction scan for
    // every imported-but-unreferenced symbol — the bulk of an import table.
    let externals: Vec<FunctionId> = ctx
        .functions()
        .filter(|f| f.is_external())
        .map(|f| f.id)
        .filter(|&id| !ctx.shared.values.call_sites_of(id).is_empty())
        .collect();

    if externals.is_empty() {
        return false;
    }

    let mut changed = false;
    for fid in externals {
        let raw = FunctionBody::from_id(ctx, fid).name().to_string();
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
        if let Some(ret) = return_slot(proto, &env.cfg.abi, ptr_width)
            && bind_external_return(ctx, fid, ret)
        {
            changed = true;
        }
    }
    changed
}

/// The register a call to this prototype returns its value in: the integer
/// return register for an integer/pointer result, the SSE return register for a
/// float result, `None` for `void` or an aggregate return (which we do not
/// model). Mirrors [`plan_args`]' argument classification.
fn return_slot(
    proto: &cabi::CFunctionProto,
    abi: &CallingConvention,
    ptr_width: usize,
) -> Option<VarnodeId> {
    match classify(&proto.return_type)? {
        Class::Integer => abi.int_ret.as_ref().and_then(|g| g.for_bytes(ptr_width)),
        Class::Sse => abi.sse_ret,
    }
}

/// Give every direct caller of resolved external `fid` a return value: type the
/// `call` result as the return register's scalar and store it back into that
/// register in the call's continuation block. A later gvn round forwards a
/// post-call read of the register to the call result — the scalar analogue of
/// argpromote's returned write-set (`res = foo(); store(reg <- res)`), for a
/// single ABI return register. Idempotent: a call whose continuation already
/// stores its result to `ret` is left untouched.
fn bind_external_return(ctx: &mut Context, fid: FunctionId, ret: VarnodeId) -> bool {
    let ret_space = Varnode::from_id(&*ctx, ret).space().id;
    let size = Varnode::from_id(&*ctx, ret).size();
    let int_ty = ctx.shared.types.get_or_make_int(size);

    let call_sites: Vec<InstructionId> = ctx
        .instructions()
        .filter_map(|insn| match insn.mnemonic() {
            Mnemonic::Call(c) if c.target.real() == Some(fid) => Some(insn.id),
            _ => None,
        })
        .collect();

    let mut changed = false;
    for call_id in call_sites {
        // The call's fall-through continuation, where the return register becomes
        // live. A call with no successor (e.g. a noreturn tail) is skipped.
        let Some(cont) = ctx
            .get_insn(call_id)
            .parent()
            .and_then(|b| b.successors().next().map(|(_, s)| s))
        else {
            continue;
        };
        let result = ValueId::Instruction(call_id);

        // Idempotency: skip a continuation that already stores this call's result
        // into the return register.
        let already = BasicBlock::from_id(ctx, cont).iter().any(|insn| {
            matches!(
                insn.mnemonic(),
                Mnemonic::Store(s)
                    if s.src.qualify(insn.id.func) == result
                        && s.ptr.qualify(insn.id.func) == ValueId::Varnode(ret)
            )
        });
        if already {
            continue;
        }

        Instruction::from_id_mut(ctx, call_id).set_type(int_ty);
        let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, cont));
        b.set_insert_point_to_start();
        b.push_store(result, ValueId::Varnode(ret), ret_space);
        changed = true;
    }
    changed
}

/// Thread `plan` through every direct caller of `fid`, one positional argument at
/// a time and in order, so the `args.len() == idx` guard keeps each call's
/// arguments aligned and the pass idempotent. Also records the planned argument
/// names on `fid` so each call site renders them (e.g. `@return_address=…`,
/// `@hwnd=…`).
fn bind_external_args(
    ctx: &mut Context,
    fid: FunctionId,
    plan: &[PlannedArg],
    sp: VarnodeId,
    sp_space: SpaceId,
    default_space: SpaceId,
    ptr_width: usize,
) -> bool {
    FunctionBody::from_id_mut(ctx, fid)
        .set_input_arg_names(plan.iter().map(|a| a.name.clone()).collect());

    let mut changed = false;
    for (idx, arg) in plan.iter().enumerate() {
        let slot = arg.slot;
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
        assert_eq!(
            plan,
            vec![
                PlannedArg {
                    slot: Slot::Stack { offset: 0, size: 4 },
                    name: Some("return_address".into()),
                },
                PlannedArg {
                    slot: Slot::Stack { offset: 4, size: 4 },
                    name: None
                },
                PlannedArg {
                    slot: Slot::Stack { offset: 8, size: 4 },
                    name: None
                },
                PlannedArg {
                    slot: Slot::Stack {
                        offset: 12,
                        size: 4
                    },
                    name: None
                },
                PlannedArg {
                    slot: Slot::Stack {
                        offset: 16,
                        size: 4
                    },
                    name: None
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

    /// A `void` (or aggregate) return has no return register, so no return value
    /// is bound — the call stays result-less.
    #[test]
    fn void_or_aggregate_return_has_no_return_slot() {
        let abi = CallingConvention::default();
        let mut p = proto(vec![]);
        p.return_type = CType::Void;
        assert!(return_slot(&p, &abi, 4).is_none());
        p.return_type = CType::Struct { name: None };
        assert!(return_slot(&p, &abi, 4).is_none());
    }

    /// A by-value aggregate parameter is unplaceable, so the whole function is
    /// skipped rather than mis-bound.
    #[test]
    fn aggregate_param_skips_the_function() {
        let p = proto(vec![int(), CType::Struct { name: None }]);
        assert!(plan_args(&p, &CallingConvention::default(), 4, true).is_none());
    }
}
