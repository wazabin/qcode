//! `argpromote`: functionalize a function's memory side effects via shadow
//! memory and a returned write-set.
//!
//! Some functions mutate memory through pointer parameters (and globals). This
//! pass rewrites such a function so it has no side effects: it operates on a
//! private *shadow* copy of memory and *returns the writes it made* as data —
//! a flat aggregate of interleaved `address, value` fields — which the caller
//! replays:
//!
//! ```text
//!   void f(int *p, int *q) { *q = 100; *p += 1; }
//!     callee:  f(p_ptr, q_ptr, p)  ->  return (q_ptr, 100, p_ptr, p + 1)
//!     caller:  r = f(&x, &x, x);  store(r.w0_addr, r.w0_value); store(r.w1_addr, r.w1_value)
//! ```
//!
//! ## Why aliasing is handled
//!
//! Every dereferenced pointer parameter shares **one shadow space**, keyed by the
//! real address values. When two pointers are equal they collide in the shadow,
//! so a `load p` after a `store q` sees the written value — the value
//! computation stays correct with no equality guards and no anti-alias gate. The
//! writes are replayed by the caller in any order (aliased addresses carry the
//! same final value), which preserves the caller-visible effect.
//!
//! ## Mechanism (see [`analyze_param`], [`try_promote`], [`apply`])
//!
//! * each dereferenced pointer param keeps its *address* and gains a by-value
//!   snapshot of the bounded region it *reads* (1/2/4/8 bytes; write-only params
//!   need no snapshot). The caller loads that snapshot and passes it in.
//! * the callee seeds the shadow from the snapshot at entry, its loads/stores are
//!   redirected into the shadow (the real address kept as the index), and at the
//!   single return it reloads each written address and hands back the write-set
//!   through [`Return::value`] as an [`Aggregate`](qcode::types::TypeRepr::Aggregate).
//! * a real register-based return value is left untouched on its own channel.
//!
//! Eligibility (anything else leaves the function untouched):
//!
//! * at least one promoted pointer is written (otherwise nothing to move);
//! * the function makes no call (composition of effectful callees is deferred);
//! * it has a single return block (so written addresses dominate the return);
//! * every dereferenced pointer is promotable — an address leaked to memory bails
//!   the whole function, since promoting some-but-not-all would be unsound;
//! * the function's address is never taken, so every caller is a direct
//!   [`Call`] this pass can rewrite. NB: this assumes a *closed world* over
//!   discovered code — a caller we never disassembled would still see the old
//!   by-reference ABI. That gap is intentional and unguarded.
//!
//! Out of scope for now: loops / dynamically-sized write sets, addresses loaded
//! from memory (`**pp`), lifting register/stack effects, and effectful-callee
//! composition. See `ARGPROMOTE_DESIGN.md`.

use qcode::value::QCodeMut;
use qcode::{
    builder::Builder,
    context::Context,
    space::LocalMemorySpaceId,
    types::{AggregateField, TypeId},
    value::{
        BasicBlock, BlockId, FunctionBody, FunctionId, Instruction, ValueId,
        insn::{InstructionId, Mnemonic},
    },
};
use rustc_hash::FxHashSet;

use super::{append_entry_param, interface::append_entry_param_at_sites};

mod globals;
mod mark_pure;
mod ram;
mod ram_summary;
mod reg_summary;
mod registers;
mod stack_args;
pub(crate) use crate::calls::effect_engine as summary;
#[cfg(test)]
mod tests;

pub use mark_pure::mark_pure_functions;
pub use ram::argpromote;
pub use registers::{RegPurityGates, RegPurityReason, argpromote_registers, reg_purity};

/// The call-argument index whose synthesized name matches `name`.
///
/// Indexes over the root block's parameters — the real call interface of a
/// `pure_reg` callee (the functions this pass augments after the register/stack
/// channels). The pointer is passed positionally, whether in a register or on
/// the stack, and the root params are in lockstep with `Call.args` (see
/// [`FunctionBody::input_arg_name`]), so a param's index *is* its argument index.
pub(crate) fn arg_index_of(ctx: &Context, fid: FunctionId, name: &str) -> Option<usize> {
    let len = FunctionBody::from_id(ctx, fid)
        .root()
        .map_or(0, |b| b.params().count());
    (0..len).find(|&i| FunctionBody::from_id(ctx, fid).input_arg_name(i).as_deref() == Some(name))
}

/// Every function whose address is taken as a value — used as a value anywhere
/// (stored, passed, or the target of an indirect call), computed for all
/// functions in one O(instructions) pass. Direct calls reference the target
/// through `Call::target`, which is *not* an operand, so they do not count.
/// Callers
/// that gate every function on this — the argpromote channels — build it once at
/// the top of their per-function loop and look up, turning an O(functions ×
/// instructions) scan into O(instructions + functions).
///
/// Safe to build once and reuse across a channel's mutating loop: promotion only
/// threads *data* values (new params/args, replayed write-sets) and never adds or
/// removes a `ValueId::Function` operand, so the set is invariant while the loop
/// runs. (Instruction *deletion* could only shrink it — the conservative
/// direction for a gate that skips address-taken functions.)
pub(crate) fn address_taken_set(ctx: &Context) -> FxHashSet<FunctionId> {
    let mut set = FxHashSet::default();
    for insn in ctx.instructions() {
        for arg in insn.operands() {
            if let ValueId::Function(fid) = arg {
                set.insert(fid);
            }
        }
    }
    set
}

/// Every function that is the target of at least one direct [`Mnemonic::Call`],
/// computed from a call-graph snapshot. Mirrors [`address_taken_set`]: callers
/// that gate every function on "has a direct caller" — the register channel and
/// the loader's per-function purity query — build it once and look up, turning
/// an O(functions × instructions) rescan into O(instructions + functions).
///
/// Safe to reuse across a channel's mutating loop: promotion rewrites a callee's
/// interface but never adds or removes a direct `Call.target` edge, so the set of
/// called functions is invariant while the loop runs.
pub(crate) fn called_function_set_from_graph(
    ctx: &Context,
    graph: &crate::CallGraph,
) -> FxHashSet<FunctionId> {
    graph
        .edges()
        .filter_map(|(_, edge)| {
            let site = edge.site?;
            let Mnemonic::Call(call) = ctx.get_insn(site).mnemonic() else {
                return None;
            };
            call.target.real()
        })
        .collect()
}

// ===========================================================================
// Channel-agnostic interface primitives (shared by the RAM and register paths)
// ===========================================================================
//
// All three argpromote channels grow a `pure_reg` function's call interface in
// exactly two ways: by adding a by-value *input* (a new param, seeded at entry,
// threaded as an argument at every caller) or by appending *outputs* to the
// returned write-set (new return-tuple fields the caller extracts and replays).
// The channels differ only in the *encoding* — which space a seed lands in, how
// an output field is computed, and how it is replayed. These two helpers own the
// param↔arg lockstep and the write-set-resize bookkeeping so neither is hand-
// rolled (and silently desynced) per channel.

/// Add one by-value input to `fid` and thread it through every direct caller in
/// lockstep (`param[i] ↔ arg[i]`): a new root block param (`size` bytes, optional
/// `name`, `origin`, and result `type_id`), seeded at function entry into
/// `seed_space` at the address built by `seed_addr`, and an appended positional
/// `Call.args` value built per site by `caller_value`. Returns the new param's
/// `ValueId`, or `None` if `fid` has no root block.
///
/// The seed store makes the body's first read of the input resolve to the
/// caller-supplied value; `seed_space` selects the channel (a register file, the
/// shared shadow, or real ram). The lockstep is maintained by
/// [`append_entry_param`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_input(
    ctx: &mut Context,
    fid: FunctionId,
    size: usize,
    name: Option<String>,
    origin: Option<ValueId>,
    type_id: Option<TypeId>,
    call_sites: Option<&[InstructionId]>,
    seed_space: impl Into<LocalMemorySpaceId>,
    seed_addr: impl FnOnce(&mut Builder) -> ValueId,
    caller_value: impl FnMut(&mut Context, InstructionId, BlockId) -> ValueId,
) -> Option<ValueId> {
    let seed_space = seed_space.into();
    let param = if let Some(call_sites) = call_sites {
        append_entry_param_at_sites(ctx, fid, size, name, origin, call_sites, caller_value)?
    } else {
        append_entry_param(ctx, fid, size, name, origin, caller_value)?
    };
    if let (ValueId::BlockParam(pid), Some(ty)) = (param, type_id) {
        ctx.block_param_mut(pid).type_id = ty;
    }
    let root = FunctionBody::from_id(ctx, fid).root().map(|b| b.id)?;
    let mut b = (ctx).builder(root);
    b.set_insert_point_to_start();
    let addr = seed_addr(&mut b);
    b.push_store(param, addr, seed_space);
    Some(param)
}

/// Append `slots` as outputs of `fid`'s returned write-set and replay them at
/// every direct caller. Each slot contributes one or more named tuple fields at
/// every `Return` (named by `field_names`, valued by `at_return`) plus a caller-
/// side `replay` that consumes those fields extracted back out of the call
/// result. A positional register output is one field per slot; a RAM write is an
/// `(addr, value)` pair.
///
/// `append` preserves the write-set an earlier channel already emitted (the RAM
/// channel's pairs sit *after* the register channel's positional outputs),
/// dropping only stale `write*` fields so a re-run replaces rather than
/// duplicates them; `!append` overwrites the return value outright (the register
/// channel is the first to touch the slot). The call result is retyped to the
/// assembled aggregate — resized when appending, since the register channel may
/// have already typed it. A no-op when `slots` is empty (a read-only promotion
/// leaves the returns and result type untouched).
#[allow(clippy::too_many_arguments)]
pub(crate) fn append_outputs<S>(
    ctx: &mut Context,
    fid: FunctionId,
    slots: &[S],
    call_sites: Option<&[InstructionId]>,
    append: bool,
    field_names: impl Fn(usize, &S) -> Vec<String>,
    at_return: impl Fn(&mut Builder, &S) -> Vec<ValueId>,
    replay: impl Fn(&mut Builder, &S, &[ValueId]),
) {
    if slots.is_empty() {
        return;
    }
    // Field count per slot — drives both tuple assembly and the caller's extract
    // stride. Uniform within a channel, but derived per slot to stay encoding-free.
    let arities: Vec<usize> = slots
        .iter()
        .enumerate()
        .map(|(i, s)| field_names(i, s).len())
        .collect();

    let returns: Vec<InstructionId> = FunctionBody::from_id(ctx, fid)
        .iter()
        .filter_map(|b| {
            let last = b.iter().last()?;
            matches!(last.mnemonic(), Mnemonic::Return(_)).then_some(last.id)
        })
        .collect();

    // `base_len` — the number of preserved leading fields the caller's extracts
    // skip past. Identical at every return (each carries the same earlier-channel
    // write-set), so the last assignment is the value the caller loop uses.
    let mut base_len = 0usize;
    let mut writeset_ty = None;
    for ret_id in returns {
        let Some(ret_block) = ctx.get_insn(ret_id).parent().map(|b| b.id) else {
            continue;
        };
        let base_fields = if append {
            surviving_writeset_fields(ctx, ret_id)
        } else {
            Vec::new()
        };
        base_len = base_fields.len();
        let fields = {
            let mut b = (ctx).builder(ret_block);
            b.set_insert_point_before(ret_id);
            let mut fields = base_fields;
            for (i, s) in slots.iter().enumerate() {
                let names = field_names(i, s);
                let vals = at_return(&mut b, s);
                for (name, v) in names.into_iter().zip(vals) {
                    fields.push((name, v));
                }
            }
            fields
        };
        let tuple_fields: Vec<AggregateField> = fields
            .iter()
            .map(|(name, value)| AggregateField::new(name.clone(), ctx.type_of(*value)))
            .collect();
        let return_ty = if let Some(return_ty) = writeset_ty {
            debug_assert_eq!(
                ctx.shared.types.aggregate_fields(return_ty),
                Some(tuple_fields.as_slice()),
                "all returns of a function must agree on the return-record layout"
            );
            return_ty
        } else if ctx.shared.types.function_return(fid).is_some() {
            ctx.shared
                .types
                .edit_function_return(fid, tuple_fields)
                .expect("existing function return type must remain editable")
        } else {
            ctx.shared
                .types
                .create_function_return(fid, tuple_fields)
                .expect("function return type must be created exactly once")
        };
        let tuple = {
            let mut b = (ctx).builder(ret_block);
            b.set_insert_point_before(ret_id);
            ValueId::Instruction(b.push_named_tuple_with_type(fields, return_ty).id)
        };
        let mut m = ctx.get_insn(ret_id).mnemonic().clone();
        if let Mnemonic::Return(ref mut r) = m {
            r.value = Some(tuple.localize(ret_id.func));
        }
        ctx.replace_instruction_mnemonic(ret_id, m);
        writeset_ty = Some(return_ty);
    }
    let Some(writeset_ty) = writeset_ty else {
        return;
    };

    let owned_call_sites;
    let call_sites = if let Some(call_sites) = call_sites {
        call_sites
    } else {
        owned_call_sites = super::fresh_direct_call_sites(ctx, fid);
        &owned_call_sites
    };
    for &call_id in call_sites {
        // The call now yields the write-set aggregate. Resize when appending: the
        // register channel may have already typed it to its (smaller) positional
        // write-set, and we are growing it with the appended fields.
        if append {
            Instruction::from_id_mut(ctx, call_id).set_type_resized(writeset_ty);
        } else {
            Instruction::from_id_mut(ctx, call_id).set_type(writeset_ty);
        }
        let Some(call_block) = ctx.get_insn(call_id).parent().map(|b| b.id) else {
            continue;
        };
        let Some(cont) = BasicBlock::from_id(ctx, call_block)
            .successors()
            .next()
            .map(|(_, b)| b)
        else {
            continue;
        };
        let result = ValueId::Instruction(call_id);
        let mut b = (ctx).builder(cont);
        b.set_insert_point_to_start();
        let mut idx = base_len;
        for (s, &arity) in slots.iter().zip(&arities) {
            let extracted: Vec<ValueId> = (0..arity)
                .map(|k| ValueId::Instruction(b.push_extract(result, idx + k).id))
                .collect();
            replay(&mut b, s, &extracted);
            idx += arity;
        }
    }
}

/// The fields of `ret_id`'s current write-set tuple that a fresh promotion round
/// must preserve — every field except the stale `write*` memory pairs a prior RAM
/// round may have appended (re-running replaces them, and duplicate field names
/// are illegal). Empty when the return carries no tuple.
fn surviving_writeset_fields(ctx: &mut Context, ret_id: InstructionId) -> Vec<(String, ValueId)> {
    let Mnemonic::Return(r) = ctx.get_insn(ret_id).mnemonic().clone() else {
        return Vec::new();
    };
    let Some(val @ ValueId::Instruction(tid)) = r.value.map(|v| v.qualify(ret_id.func)) else {
        return Vec::new();
    };
    let Mnemonic::Tuple(t) = ctx.get_insn(tid).mnemonic().clone() else {
        return Vec::new();
    };
    let ty = ctx.type_of(val);
    t.fields
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let name = ctx
                .shared
                .types
                .field_name(ty, i)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("field{}", i + 1));
            (name, v.qualify(tid.func))
        })
        .filter(|(name, _)| !name.starts_with("write"))
        .collect()
}
