//! Signatures and call interfaces for known external (imported) functions, from
//! their C prototypes.
//!
//! External functions are bodyless stubs, so [`call_summary`](crate::calls)
//! cannot infer their inputs. Instead this pass looks each function name up in a
//! per-binary [`cabi::Selection`] — the prototype tables for the libraries the
//! binary actually links against — and **materializes** everything downstream
//! passes need, so `cabi` is consulted in exactly this one pass:
//!
//! * the register interface, materialized as
//!   [`RegisterChannelState::Materialized`](qcode::value::FunctionEffects) under the
//!   [`CallingConvention`] supplied by [`PipelineEnv`]: its inputs are the
//!   argument registers, its outputs the return register(s) ∪ the caller-saved
//!   clobber set — plus the per-parameter `param_attrs`;
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
//! Run before the `summaries` pass: once an external callee has a signature,
//! the argument-producing passes can produce real arguments at its call sites.

use cabi::{CFunctionProto, CParam, CType, Config, Selection};
use qcode::{
    context::Context,
    value::{
        ArgMemKind, ExternArg, ExternArgmem, ExternInterface, ExternSlot, FunctionBody, FunctionId,
        ParamAttrs, RegisterChannelState, RegisterInterfaceMap, VarnodeId,
    },
};

use crate::pipeline::{CallingConvention, cabi_abi_target};

use crate::{Pass, PipelineEnv};

use rustc_hash::FxHashMap;
use std::sync::LazyLock;

/// The checked-in write-only destination table (`extern_argmem.toml`), compiled
/// into the binary. Lists, per external symbol name, the C-prototype positional
/// parameter indices that libc semantics guarantee are *pure destinations*
/// (never read before write). See the file header for the inclusion bar.
const EXTERN_ARGMEM_TOML: &str = include_str!("extern_argmem.toml");

/// The TOML shape of [`EXTERN_ARGMEM_TOML`]: a single `[write_only]` table
/// mapping symbol name → the write-only parameter indices.
#[derive(serde::Deserialize)]
struct ExternArgmemTable {
    #[serde(default)]
    write_only: FxHashMap<String, Vec<usize>>,
}

/// Parse-once view of the write-only table: symbol name → write-only C-prototype
/// param indices. Parsed lazily on first use (matching the `cabi` `OnceLock`
/// idiom); a malformed embedded table is a build-time authoring bug, so a parse
/// failure panics rather than silently disabling the upgrade.
fn write_only_table() -> &'static FxHashMap<String, Vec<usize>> {
    static TABLE: LazyLock<FxHashMap<String, Vec<usize>>> = LazyLock::new(|| {
        toml::from_str::<ExternArgmemTable>(EXTERN_ARGMEM_TOML)
            .expect("embedded extern_argmem.toml must parse")
            .write_only
    });
    &TABLE
}

/// Upgrade the write-only destination parameters of `name` from `MutPtr` to
/// `OutPtr` in `argmem_kinds`, per the [`write_only_table`]. `argmem_kinds` is
/// lockstep with the register-interface inputs, which are the register-passed
/// prefix of the prototype parameters *in prototype order*, so a C-prototype
/// positional index maps directly onto it (bounds-checked for the overflow tail
/// that spilled to the stack). Only a `MutPtr` is upgraded — a `ConstPtr`,
/// `Opaque`, or non-pointer at that index is left alone (the table only claims
/// write-only-ness for what the prototype already made a mutable pointer).
fn upgrade_write_only(name: &str, argmem_kinds: &mut [ArgMemKind]) {
    let Some(indices) = write_only_table().get(name) else {
        return;
    };
    for &i in indices {
        if let Some(kind) = argmem_kinds.get_mut(i)
            && *kind == ArgMemKind::MutPtr
        {
            *kind = ArgMemKind::OutPtr;
        }
    }
}

#[derive(Default)]
pub struct ExternalSigs;

impl Pass for ExternalSigs {
    const NAME: &'static str = "external_sigs";

    fn description(&self) -> &'static str {
        "Give known external (libc) functions signatures from their C prototypes"
    }

    fn run(
        &self,
        cone: &mut crate::ConeMut,
        env: &PipelineEnv,
    ) -> Result<crate::ModulePassOutcome, String> {
        let targets = cone.cone_functions();
        if !abi_is_known(&env.cfg.abi) {
            return Ok(crate::ModulePassOutcome::default());
        }
        let affected: Vec<FunctionId> = targets
            .iter()
            .copied()
            .filter(|&id| FunctionBody::from_id(cone.ctx(), id).is_external())
            .collect();
        // Whole-program read for planning; each stamp is a single-function
        // interface write confined to `id`, so it goes through the cone.
        let sel = selection_for(cone.ctx(), env);
        let ptr_width = ptr_width(env);
        let stack_only = env.cfg.bitness == 32;
        for id in affected.iter().copied() {
            apply_external_signature(
                cone.ctx_for(id),
                id,
                &env.cfg.abi,
                &sel,
                ptr_width,
                stack_only,
            );
        }
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

/// The mapped register interface of a prototyped external: argument registers,
/// return register(s), per-input [`ParamAttrs`], and per-input [`ArgMemKind`],
/// all in lockstep with the argument registers.
type MappedProto = (
    Vec<VarnodeId>,
    Vec<VarnodeId>,
    Vec<ParamAttrs>,
    Vec<ArgMemKind>,
);

/// The argument registers and return register for `proto` under `abi`, or `None`
/// if any parameter (or the return) is an aggregate/unsupported type — such
/// functions are left unsigned rather than mapped incorrectly.
fn map_prototype(proto: &CFunctionProto, abi: &CallingConvention) -> Option<MappedProto> {
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
    // Per-input argmem kinds, kept in lockstep with `inputs` (and so with the
    // materialized register interface / `Param(i)` positions the RAM channel
    // rebases). Read by `ram_summary::external_leaf`.
    let mut argmem: Vec<ArgMemKind> = Vec::with_capacity(proto.params.len());
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
                argmem.push(argmem_kind(&param.ty));
            }
            Class::Sse => {
                let Some(&vn) = abi.sse_args.get(next_sse) else {
                    break;
                };
                next_sse += 1;
                inputs.push(vn);
                attrs.push(ParamAttrs::default());
                // A float in an SSE register is never a pointer.
                argmem.push(ArgMemKind::NonPtr);
            }
        }
    }

    let outputs = match classify(&proto.return_type) {
        Some(Class::Integer) => abi.int_ret.as_ref().and_then(gp64).into_iter().collect(),
        Some(Class::Sse) => abi.sse_ret.into_iter().collect(),
        None => Vec::new(), // void: no return register
    };

    Some((inputs, outputs, attrs, argmem))
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

/// The RAM argmem kind of one parameter type: how (if at all) an external can
/// reach memory *we* model through it. Feeds the RAM effect channel's
/// `external_leaf`.
///
/// A function/callback pointer (`int (*)(...)`) lowers to a pointer to
/// [`CType::Other`] (function types are not modelled), and a pointer to a
/// pointer (`char **`) admits a *transitive* write that escapes the shallow
/// whole-object model. Both are classified [`ArgMemKind::Opaque`] — a
/// conservative over-approximation of "unbounded through this pointer" that sends
/// the whole external footprint to ⊤. Pointers to a scalar/void/named-aggregate
/// pointee are flat data pointers (`libc`'s own aggregate state lives outside the
/// lifted image), so they carry a bounded whole-object effect.
fn argmem_kind(ty: &CType) -> ArgMemKind {
    match ty {
        CType::Pointer {
            pointee,
            const_pointee,
        } => {
            if matches!(**pointee, CType::Other | CType::Pointer { .. }) {
                ArgMemKind::Opaque
            } else if *const_pointee {
                ArgMemKind::ConstPtr
            } else {
                ArgMemKind::MutPtr
            }
        }
        _ => ArgMemKind::NonPtr,
    }
}

/// Mechanical glibc symbol aliases: a versioned re-spelling of a standard libc
/// function that has *the same C signature* as the unprefixed name but that no
/// header declares under the prefixed spelling. `<stdio.h>` declares `sscanf`;
/// the compiler emits a call to `__isoc23_sscanf` (glibc's C23-conformant
/// scanf family) or `__isoc99_sscanf` (the C99 one), and the prototype table —
/// built from headers — has no entry for either. Stripping the prefix at lookup
/// recovers the real, precise prototype.
///
/// Deliberately narrow: only families where the alias is the *same function
/// under a different symbol version*, so the signature is identical by
/// construction. The `__*_chk` fortified family is NOT here — `__memcpy_chk`
/// takes an extra `size_t destlen` parameter, so it is a different signature.
const GLIBC_ALIAS_PREFIXES: &[&str] = &["__isoc23_", "__isoc99_"];

/// The unprefixed libc name behind a mechanical glibc alias, if `name` is one.
fn glibc_alias_base(name: &str) -> Option<&str> {
    GLIBC_ALIAS_PREFIXES
        .iter()
        .find_map(|prefix| name.strip_prefix(prefix))
}

/// The register-channel interface map for a callee whose argument registers are
/// `inputs` and whose ABI return register(s) are `outputs`, under `abi`.
///
/// Returns-first ordering: the return register(s) lead the output pack and the
/// convention's caller-saved (volatile) set follows as a clobber tail (poison at
/// a rewritten site). Growing the outputs to include the clobbers is what lets a
/// materialized caller's return pack cover them (bug-2-external). Both the
/// prototyped path and the ABI fallback go through here, so the two agree on the
/// clobber set by construction.
fn register_interface_map(
    inputs: Vec<VarnodeId>,
    outputs: &[VarnodeId],
    abi: &CallingConvention,
) -> RegisterInterfaceMap {
    let mut pack_outputs: Vec<VarnodeId> = outputs.to_vec();
    for &clobber in &abi.caller_saved {
        if !pack_outputs.contains(&clobber) {
            pack_outputs.push(clobber);
        }
    }
    RegisterInterfaceMap {
        inputs,
        returns: outputs.len(),
        projections: Vec::new(),
        outputs: pack_outputs,
    }
}

/// The synthetic "unknown signature" prototype for an external with no C
/// declaration: every argument-passing register of `abi` is a parameter and the
/// return is an integer. Fed through [`map_prototype`] exactly like a real
/// prototype, so the fallback interface is derived from the calling-convention
/// model alone — no architecture is hardcoded.
fn abi_unknown_proto(abi: &CallingConvention) -> CFunctionProto {
    let param = |ty| CParam { name: None, ty };
    let params = abi
        .int_args
        .iter()
        .map(|_| {
            param(CType::Integer {
                bytes: 8,
                signed: false,
            })
        })
        .chain(
            abi.sse_args
                .iter()
                .map(|_| param(CType::Float { bytes: 8 })),
        )
        .collect();
    CFunctionProto {
        name: "".into(),
        return_type: CType::Integer {
            bytes: 8,
            signed: true,
        },
        params,
        variadic: false,
        header: None,
    }
}

/// Ruling: **"obeys the platform ABI" is a sound assumption for an arbitrary
/// unprototyped external symbol.** So an external the prototype table cannot
/// name is not ⊤ for the register channel — it is materialized from the calling
/// convention alone: its inputs are *every* argument-passing register and its
/// outputs are the return register(s) ∪ *every* caller-saved register. Both
/// halves over-approximate: a real callee reads a prefix of the argument
/// registers and clobbers a subset of the volatile ones.
///
/// This runs only when `sel` has no prototype for the symbol (a real prototype
/// always wins, and is strictly more precise).
///
/// Unlike the prototyped path this stamps *only* the register channel: with no
/// prototype there is nothing to say about parameter attributes, pointer argmem
/// kinds, or stack slots, so those stay absent and the RAM channel keeps
/// treating the callee conservatively.
fn apply_abi_fallback_signature(ctx: &mut Context, fun_id: FunctionId, abi: &CallingConvention) {
    let proto = abi_unknown_proto(abi);
    let Some((inputs, outputs, ..)) = map_prototype(&proto, abi) else {
        return;
    };
    let reg_map = register_interface_map(inputs, &outputs, abi);
    FunctionBody::from_id_mut(ctx, fun_id)
        .set_register_effects(RegisterChannelState::Materialized(reg_map));
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
    // Precise prototype first; then the mechanical glibc alias spelling
    // (`__isoc23_sscanf` → `sscanf`); then, with nothing declared for the
    // symbol at all, the ABI-only fallback.
    let proto = sel
        .lookup(name)
        .or_else(|| glibc_alias_base(name).and_then(|base| sel.lookup(base)));
    let Some(proto) = proto else {
        apply_abi_fallback_signature(ctx, fun_id, abi);
        return;
    };
    let Some((inputs, outputs, param_attrs, mut argmem_kinds)) = map_prototype(proto, abi) else {
        // An unmappable prototype (by-value aggregate) tells us nothing precise,
        // but the callee still obeys the ABI.
        apply_abi_fallback_signature(ctx, fun_id, abi);
        return;
    };
    // Ruling 1b: distinguish write-only destination pointers (memset/memcpy dest
    // — safe to admit through a frame local) from read-write mutable pointers
    // (must keep their read half so a frame landing goes ⊤). The prototype alone
    // cannot tell them apart (both are non-`const` pointers → `MutPtr`), so the
    // curated `extern_argmem.toml` table names the pure destinations by symbol,
    // upgrading them to `OutPtr`. `name` is already `@`-suffix-stripped above.
    upgrade_write_only(name, &mut argmem_kinds);
    let plan = plan_args(proto, abi, ptr_width, stack_only);

    // The register-channel interface mapping (argpromote v2, ruling 6a): this
    // materialized external's inputs are its argument registers and its outputs
    // are its return register(s) ∪ the ABI caller-saved clobber set. No varargs
    // check — a prototyped variadic external materializes like any other. This is
    // the single source of truth: `reg_summary::external_leaf` reads it, and
    // pass 2 never grows an external branch.
    let reg_map = register_interface_map(inputs.clone(), &outputs, abi);

    let mut f = FunctionBody::from_id_mut(ctx, fun_id);
    f.set_param_attrs(param_attrs);
    // The prototype-derived argmem summary consumed by the RAM effect channel:
    // per-input pointer kinds (lockstep with `inputs`) plus the variadic flag.
    f.set_argmem(ExternArgmem {
        params: argmem_kinds,
        variadic: proto.variadic,
    });
    // The prototype fully describes this callee's register effect, so it is
    // materialized directly: its inputs are the argument registers the caller
    // passes and its outputs are the return register(s) ∪ the convention's
    // caller-saved clobber set. This is the single source of truth for the
    // register channel — the value passes read it (via `FunctionEffects`) and
    // drop the conservative "reads/writes every register" fallback.
    f.set_register_effects(RegisterChannelState::Materialized(reg_map));

    // The materialized call interface consumed by `argpromote_external`: the
    // ordered call slots and the per-slot display names. Only set when the
    // arguments are all placeable (a by-value aggregate leaves it absent).
    if let Some(args) = plan {
        f.set_extern_interface(ExternInterface { args });
    }
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
        external_at(tc, name, 0x9000)
    }

    /// Externals must have distinct addresses, so tests that make several of them
    /// place each at its own slot.
    fn external_at(tc: &mut TestContext, name: &str, addr: u64) -> FunctionId {
        FunctionBody::make_external(&mut tc.ctx, addr, Some(name.to_string().into())).id
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
        let func = FunctionBody::from_id(&tc.ctx, f);
        let map = func
            .effects()
            .materialized()
            .expect("prototyped external must be materialized");
        // Only two GP registers exist in the toy ABI; the third arg is stack.
        assert_eq!(map.inputs, [tc.r0, tc.r1]);
        // memcpy returns void* -> the integer return register leads the pack.
        assert_eq!(map.returns, 1);
        assert_eq!(map.outputs[0], tc.r3);
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
        let func = FunctionBody::from_id(&tc.ctx, f);
        let map = func
            .effects()
            .materialized()
            .expect("prototyped external must be materialized");
        assert_eq!(map.inputs, [tc.r2]);
    }

    #[test]
    fn unknown_function_is_left_unsigned() {
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        let f = external(&mut tc, "definitely_not_a_libc_function_xyz");
        apply(&mut tc.ctx, f, &abi, &host_sel());
        assert!(FunctionBody::from_id(&tc.ctx, f).signature().is_none());
    }

    /// Ruling 1: an external with no prototype at all is *not* ⊤ — it is
    /// materialized from the calling convention alone, over-approximating on
    /// both sides: every argument register is an input, and the return register
    /// ∪ every caller-saved register is an output.
    #[test]
    fn prototypeless_external_falls_back_to_the_abi_interface() {
        use qcode::value::RegisterChannelState;
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc); // int_args = [r0, r1], sse_args = [r2], ret/clobber = r3
        let f = external(&mut tc, "definitely_not_a_libc_function_xyz");
        apply(&mut tc.ctx, f, &abi, &host_sel());
        let func = FunctionBody::from_id(&tc.ctx, f);
        let RegisterChannelState::Materialized(map) = &func.effects().register else {
            panic!("a prototype-less external must still be stamped Materialized");
        };
        assert_eq!(
            map.inputs,
            vec![tc.r0, tc.r1, tc.r2],
            "every argument-passing register of the convention is an input"
        );
        assert_eq!(map.returns, 1, "the ABI return register leads the pack");
        assert_eq!(map.outputs, vec![tc.r3], "return ∪ caller-saved, deduped");
        // No prototype means nothing precise to say about memory or slots.
        assert!(func.argmem().is_none());
        assert!(func.extern_interface().is_none());
    }

    /// The ABI fallback and the prototyped path agree on the clobber set: both
    /// build their output pack through `register_interface_map`, so a
    /// prototype-less external's outputs are a superset of a prototyped one's.
    #[test]
    fn fallback_and_prototyped_paths_agree_on_clobbers() {
        use qcode::value::RegisterChannelState;
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        let known = external_at(&mut tc, "memcpy", 0x9000);
        let unknown = external_at(&mut tc, "definitely_not_a_libc_function_xyz", 0x9100);
        apply(&mut tc.ctx, known, &abi, &host_sel());
        apply(&mut tc.ctx, unknown, &abi, &host_sel());
        let outputs = |f| match &FunctionBody::from_id(&tc.ctx, f).effects().register {
            RegisterChannelState::Materialized(map) => map.outputs.clone(),
            other => panic!("expected Materialized, got {other:?}"),
        };
        for clobber in &abi.caller_saved {
            assert!(outputs(known).contains(clobber));
            assert!(outputs(unknown).contains(clobber));
        }
    }

    /// A real prototype always wins over the ABI fallback: `memcpy` keeps its
    /// two-register interface rather than claiming every argument register.
    #[test]
    fn a_real_prototype_beats_the_abi_fallback() {
        use qcode::value::RegisterChannelState;
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        let f = external(&mut tc, "memcpy");
        apply(&mut tc.ctx, f, &abi, &host_sel());
        let RegisterChannelState::Materialized(map) =
            &FunctionBody::from_id(&tc.ctx, f).effects().register
        else {
            panic!("prototyped external must be Materialized");
        };
        assert_eq!(
            map.inputs,
            vec![tc.r0, tc.r1],
            "the prototype's two register args, not the whole convention"
        );
    }

    /// Ruling 2: the mechanical glibc aliases resolve to the unprefixed
    /// prototype, so they get the *precise* interface rather than the fallback.
    #[test]
    fn glibc_isoc_aliases_resolve_to_the_unprefixed_prototype() {
        use qcode::value::RegisterChannelState;
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        // strtol(const char *nptr, char **endptr, int base): three INTEGER args,
        // two of which fit the toy ABI's GP registers.
        let plain = external_at(&mut tc, "strtol", 0x9000);
        let aliased = external_at(&mut tc, "__isoc23_strtol", 0x9100);
        let c99 = external_at(&mut tc, "__isoc99_scanf", 0x9200);
        let scanf = external_at(&mut tc, "scanf", 0x9300);
        for f in [plain, aliased, c99, scanf] {
            apply(&mut tc.ctx, f, &abi, &host_sel());
        }
        let iface = |f| match &FunctionBody::from_id(&tc.ctx, f).effects().register {
            RegisterChannelState::Materialized(map) => map.clone(),
            other => panic!("expected Materialized, got {other:?}"),
        };
        assert_eq!(iface(aliased), iface(plain), "__isoc23_strtol == strtol");
        assert_eq!(iface(c99), iface(scanf), "__isoc99_scanf == scanf");
        // And the aliased form carries the prototype's argmem, not the fallback's
        // absence of one.
        assert!(
            FunctionBody::from_id(&tc.ctx, aliased).argmem().is_some(),
            "the alias must pick up the real prototype's argmem summary"
        );
    }

    /// The alias strip is not a general prefix strip: `__memcpy_chk` takes an
    /// extra `destlen` parameter, so it must NOT borrow `memcpy`'s prototype.
    #[test]
    fn only_the_listed_alias_families_are_normalized() {
        assert_eq!(glibc_alias_base("__isoc23_strtol"), Some("strtol"));
        assert_eq!(glibc_alias_base("__isoc99_sscanf"), Some("sscanf"));
        assert_eq!(glibc_alias_base("__memcpy_chk"), None);
        assert_eq!(glibc_alias_base("strtol"), None);
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
        use qcode::value::RegisterChannelState;
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc); // caller_saved = [r3], int_ret = r3
        let f = external(&mut tc, "memcpy");
        apply(&mut tc.ctx, f, &abi, &host_sel());
        let func = FunctionBody::from_id(&tc.ctx, f);
        let RegisterChannelState::Materialized(map) = &func.effects().register else {
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

    /// The prototype-derived argmem summary is stamped in lockstep with the
    /// register inputs: memcpy(void *dst, const void *src, size_t) → a mutable
    /// pointer, a const pointer, then a non-pointer scalar.
    #[test]
    fn argmem_classifies_memcpy_pointers() {
        use qcode::value::ArgMemKind;
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        let f = external(&mut tc, "memcpy");
        apply(&mut tc.ctx, f, &abi, &host_sel());
        let func = FunctionBody::from_id(&tc.ctx, f);
        let argmem = func.argmem().expect("prototyped external has argmem");
        assert!(!argmem.variadic, "memcpy is not variadic");
        // Only two GP registers in the toy ABI, so only the first two params are
        // register-passed (lockstep with `inputs`).
        // memcpy's dest is a curated write-only destination (never reads dest),
        // so ruling 1b upgrades index 0 from MutPtr to OutPtr; src stays const.
        assert_eq!(
            argmem.params,
            vec![ArgMemKind::OutPtr, ArgMemKind::ConstPtr],
            "dst is a write-only destination (upgraded), src is a const pointer"
        );
    }

    /// The embedded `extern_argmem.toml` write-only table parses and carries the
    /// curated entries (smoke test), including memset's dest at index 0.
    #[test]
    fn write_only_table_parses() {
        let table = write_only_table();
        assert_eq!(table.get("memset").map(Vec::as_slice), Some(&[0][..]));
        assert_eq!(table.get("memcpy").map(Vec::as_slice), Some(&[0][..]));
        assert!(
            table.get("strcat").is_none(),
            "strcat reads its dest — must NOT be in the write-only table"
        );
        assert!(
            table.get("realloc").is_none(),
            "realloc reads the old block — must NOT be write-only"
        );
    }

    /// `upgrade_write_only` upgrades only the listed `MutPtr` indices to `OutPtr`,
    /// bounds-checked, and leaves everything else (including a `ConstPtr` at a
    /// listed index) untouched.
    #[test]
    fn upgrade_write_only_upgrades_listed_mutptrs() {
        // memset dest at 0 upgrades; a NonPtr scalar at 1 is unchanged.
        let mut kinds = vec![ArgMemKind::MutPtr, ArgMemKind::NonPtr];
        upgrade_write_only("memset", &mut kinds);
        assert_eq!(kinds, vec![ArgMemKind::OutPtr, ArgMemKind::NonPtr]);
        // A symbol not in the table is left alone.
        let mut kinds = vec![ArgMemKind::MutPtr];
        upgrade_write_only("strcat", &mut kinds);
        assert_eq!(kinds, vec![ArgMemKind::MutPtr]);
    }

    /// Ruling 1b, index-mapping pin: applying the full signature to `memset`
    /// upgrades exactly the dest parameter (positional index 0) to `OutPtr`, in
    /// lockstep with the register inputs — the read half is dropped so the
    /// `memset(&local)` fold survives.
    #[test]
    fn memset_dest_is_upgraded_to_outptr() {
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        let f = external(&mut tc, "memset");
        apply(&mut tc.ctx, f, &abi, &host_sel());
        let func = FunctionBody::from_id(&tc.ctx, f);
        let argmem = func.argmem().expect("prototyped external has argmem");
        // memset(void *dest, int c, size_t n): dest (index 0) → OutPtr; the toy
        // ABI has two GP regs so index 1 (the int) is also present as NonPtr.
        assert_eq!(
            argmem.params[0],
            ArgMemKind::OutPtr,
            "memset's dest is a curated write-only destination"
        );
    }

    /// A mutable-pointer external NOT in the write-only table keeps `MutPtr`
    /// (its read half is retained → a frame landing goes ⊤).
    #[test]
    fn strcat_dest_stays_mutptr() {
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        let f = external(&mut tc, "strcat");
        apply(&mut tc.ctx, f, &abi, &host_sel());
        let func = FunctionBody::from_id(&tc.ctx, f);
        let argmem = func.argmem().expect("prototyped external has argmem");
        assert_eq!(
            argmem.params[0],
            ArgMemKind::MutPtr,
            "strcat reads its dest — stays a read-write MutPtr"
        );
    }

    /// A variadic prototype (printf) carries the `variadic` flag through.
    #[test]
    fn argmem_marks_variadic_prototype() {
        let mut tc = TestContext::new();
        let abi = toy_abi(&tc);
        let f = external(&mut tc, "printf");
        apply(&mut tc.ctx, f, &abi, &host_sel());
        let func = FunctionBody::from_id(&tc.ctx, f);
        let argmem = func.argmem().expect("prototyped external has argmem");
        assert!(argmem.variadic, "printf is variadic");
    }

    /// `argmem_kind` maps each shape: scalar → NonPtr, mutable/const data
    /// pointers → Mut/ConstPtr, and a function-pointer or pointer-to-pointer →
    /// Opaque (⊤ trigger).
    #[test]
    fn argmem_kind_classifies_pointer_shapes() {
        use qcode::value::ArgMemKind;
        let ptr_to = |pointee: CType, c: bool| CType::Pointer {
            pointee: Box::new(pointee),
            const_pointee: c,
        };
        assert_eq!(argmem_kind(&int()), ArgMemKind::NonPtr);
        assert_eq!(argmem_kind(&ptr_to(CType::Void, false)), ArgMemKind::MutPtr);
        assert_eq!(
            argmem_kind(&ptr_to(CType::Void, true)),
            ArgMemKind::ConstPtr
        );
        assert_eq!(
            argmem_kind(&ptr_to(CType::Struct { name: None }, false)),
            ArgMemKind::MutPtr,
            "a flat named-aggregate pointer is a bounded whole-object pointer"
        );
        assert_eq!(
            argmem_kind(&ptr_to(CType::Other, false)),
            ArgMemKind::Opaque,
            "a function/callback pointer (pointee Other) is opaque"
        );
        assert_eq!(
            argmem_kind(&ptr_to(ptr_to(CType::Void, false), false)),
            ArgMemKind::Opaque,
            "a pointer-to-pointer admits a transitive write — opaque"
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
