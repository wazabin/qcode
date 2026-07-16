//! The mutable-host counterpart of [`QCodeView`]: one trait carrying the
//! body-local mutation verbs, implemented by both mutation hosts.
//!
//! [`Context`] resolves any function body in the module (the sequential/module
//! path); [`BodyMut`] resolves exactly one checked-out body (the function-pass
//! path, which panics on a foreign [`FunctionId`]). Every provided verb is a
//! one-liner into the canonical inherent implementation on [`FunctionBody`], so
//! the verb logic exists in exactly one place and the hosts cannot drift.
//!
//! The trait's contract is strictly **body-local** mutation. Anything that
//! writes shared module state — name registration in the global table, varnode
//! or function minting — is deliberately absent and stays inherent on
//! [`Context`], so "a checked-out pass cannot touch shared state" is documented
//! by the type system. The births (`push_block`, `push_insn`, `push_mnemonic*`,
//! …) and `remove_cfg_edge` also stay inherent per host: their spellings
//! diverge (the module path names the function, the checked-out path doesn't),
//! and they are thin arena pushes with no drift-prone logic.

use jstd::registry::Registry;
use rustc_hash::{FxHashMap, FxHashSet};

use std::borrow::Cow;

use crate::{
    context::{Context, Shared},
    error::Result,
    value::{
        BasicBlock, BodyView, FunctionBody, FunctionId, Instruction, ModuleView, QCodeView,
        ValueId,
        block::{BlockId, EdgeId},
        block_param::BlockParam,
        block_param::BlockParamId,
        function::FunctionInterface,
        insn::{InstructionId, LocalInsnId, Mnemonic},
        util::body_mut::BodyMut,
    },
};

/// Body-local mutation capability shared by the module host ([`Context`]) and
/// the checked-out pass host ([`BodyMut`]).
///
/// The primitives (`function_mut`, `shr`, `interfaces`, `view`) are the whole
/// per-host surface; every verb is a provided method delegating to the
/// [`FunctionBody`] canon through the function named by its arguments' ids.
pub trait QCodeMut<'str> {
    /// The host's `Copy` read provider ([`ModuleView`] or [`BodyView`]).
    type View<'v>: QCodeView<'v, 'str>
    where
        Self: 'v,
        'str: 'v;

    /// The storage of the function `id` (write). The checked-out host panics if
    /// `id` is not its own function.
    fn function_mut(&mut self, id: FunctionId) -> &mut FunctionBody<'str>;

    /// The module's shared IR state (read-only through this trait).
    fn shr(&self) -> &Shared<'str>;

    /// Every function's published interface (the caller-reasoning surface).
    fn interfaces(&self) -> &Registry<FunctionId, FunctionInterface<'str>>;

    /// The static immutable provider for reads over this host.
    fn view(&self) -> Self::View<'_>;

    // ---- derived arena accessors ---------------------------------------------

    /// Mutably borrows the instruction `id` from its owning function's arena.
    fn instruction_mut(&mut self, id: InstructionId) -> &mut Instruction<'str> {
        &mut self.function_mut(id.func).insns[id.local]
    }

    /// Mutably borrows the block `id` from its owning function's arena.
    fn block_mut(&mut self, id: BlockId) -> &mut BasicBlock<'str> {
        &mut self.function_mut(id.func).blocks[id.local]
    }

    /// Mutably borrows the block parameter `id` from its owning function's arena.
    fn block_param_mut(&mut self, id: BlockParamId) -> &mut BlockParam<'str> {
        &mut self.function_mut(id.func).params[id.local]
    }

    // ---- body-local verbs (canon: inherent methods on `FunctionBody`) -------

    /// Register `name` for the function-scoped `id` (block/instruction/param/
    /// Temp) in its owning function's local name table. Panics on a
    /// global-scoped `id` — shared-name registration is a module-only operation
    /// outside this trait's body-local contract. Errors only on a duplicate
    /// name.
    fn register_body_name(
        &mut self,
        id: ValueId,
        name: Cow<'str, str>,
        old_name: Option<&str>,
    ) -> Result<()> {
        let func = id
            .name_scope_function()
            .expect("register_body_name on a global-scoped value");
        self.function_mut(func)
            .register_body_name(id, name, old_name)
    }

    /// Physically removes a block parameter and its local bookkeeping.
    /// Positional block and edge-argument rewrites belong to the caller and may
    /// complete later in the same transformation.
    fn remove_block_param(&mut self, id: BlockParamId) {
        self.function_mut(id.func).remove_block_param(id);
    }

    /// Insert `insn` immediately before `before` in `block`, setting its parent.
    fn insert_insn_before(&mut self, block: BlockId, before: InstructionId, insn: InstructionId) {
        self.function_mut(block.func)
            .insert_insn_before(block, before, insn);
    }

    /// Move `insn` immediately before the arbitrary live instruction `before`,
    /// preserving the moved instruction's stable ID. Both instructions must
    /// belong to the same function; the destination block is inferred from the
    /// anchor.
    fn move_insn_before(&mut self, insn: InstructionId, before: InstructionId) {
        self.function_mut(insn.func).move_insn_before(insn, before);
    }

    /// Adds a directed edge in the CFG from `from` to `to`, returning its id.
    ///
    /// Both endpoints must belong to the same function: CFG edges are strictly
    /// intra-function (context-split ruling 2). An inter-procedural transfer is
    /// a function-level `TailCall`/`Call`, never an edge — the lifter emits
    /// those at construction, so no producer creates a cross-function edge. The
    /// permanent `debug_assert` below is the tripwire that keeps that invariant
    /// honest (it is the probe from 06a §10, now a keeper because the invariant
    /// finally holds).
    fn add_cfg_edge(&mut self, from: BlockId, to: BlockId) -> EdgeId {
        debug_assert_eq!(
            from.func, to.func,
            "cross-function CFG edge {from:?} -> {to:?} (strict IR locality, ruling 2)"
        );
        self.function_mut(from.func).add_cfg_edge(from, to)
    }

    /// Replace every use of `old` with `new` across `old`'s owning function and
    /// update the reverse use-map (SSA defs only; all uses are intra-function).
    fn replace_all_uses_with(&mut self, old: impl Into<ValueId>, new: impl Into<ValueId>) {
        let old = old.into();
        let new = new.into();
        if old == new {
            return;
        }
        // A shared value (literal/bytes/varnode) has no owning function and no
        // locatable user list; replacing its uses is a no-op here.
        let Some(func) = old.owning_function() else {
            return;
        };
        self.function_mut(func).replace_all_uses_with(old, new);
    }

    /// Remove instruction `id` from its block, unlink its outgoing CFG edges if
    /// a terminator, clear its name, prune its operand use-lists, and
    /// physically drop its payload.
    fn remove_instruction(&mut self, id: InstructionId) {
        self.function_mut(id.func).remove_instruction(id);
    }

    /// Physically removes a set of instructions after pruning their operands
    /// from the reverse-use maps, grouped per owning function. Call after
    /// removing them from their parent blocks and unlinking any CFG edges owned
    /// by terminators.
    fn remove_instructions(&mut self, dead: &FxHashSet<InstructionId>) {
        let mut by_func: FxHashMap<FunctionId, FxHashSet<LocalInsnId>> = FxHashMap::default();
        for &id in dead {
            by_func.entry(id.func).or_default().insert(id.local);
        }
        for (func, dead) in by_func {
            self.function_mut(func).remove_instructions(&dead);
        }
    }

    /// Rehome `remove`'s outgoing CFG edges onto `keep`. The direct edge and
    /// `keep`'s forwarding terminator have already been removed by the caller.
    fn rehome_outgoing_edges(&mut self, keep: BlockId, remove: BlockId) {
        self.function_mut(keep.func)
            .rehome_outgoing_edges(keep, remove);
    }

    /// Replace an instruction's mnemonic in place, keeping the reverse use-map
    /// in sync. For transforms that change an instruction without changing its
    /// identity, parent block, address, or result type.
    fn replace_instruction_mnemonic(&mut self, id: InstructionId, mnemonic: Mnemonic) {
        self.function_mut(id.func)
            .replace_instruction_mnemonic(id, mnemonic);
    }

    /// Drop `block` from its function's ownership roster. Ownership is derived
    /// from the storing arena (`block.func`); the arena slot is untouched.
    fn unroster_block(&mut self, block: BlockId) {
        self.function_mut(block.func).unroster_block(block);
    }

    /// Remove `block` from its function: unlink every incident CFG edge, remove
    /// its instructions and params, clear ownership metadata, then drop its
    /// payload.
    fn delete_block(&mut self, block: BlockId) {
        self.function_mut(block.func).delete_block(block);
    }

    /// Absorb `other` into `keep`: drop `keep`'s terminal branch, append
    /// `other`'s instructions, rehome its outgoing edges, and remove it.
    /// `edge_ab` is the direct edge `keep -> other`.
    fn absorb_block(&mut self, keep: BlockId, other: BlockId, edge_ab: EdgeId) {
        self.function_mut(keep.func)
            .absorb_block(keep, other, edge_ab);
    }
}

impl<'str> QCodeMut<'str> for Context<'str> {
    type View<'v>
        = ModuleView<'v, 'str>
    where
        Self: 'v,
        'str: 'v;

    fn function_mut(&mut self, id: FunctionId) -> &mut FunctionBody<'str> {
        &mut self.bodies[id]
    }

    fn shr(&self) -> &Shared<'str> {
        &self.shared
    }

    fn interfaces(&self) -> &Registry<FunctionId, FunctionInterface<'str>> {
        &self.interfaces
    }

    fn view(&self) -> ModuleView<'_, 'str> {
        ModuleView::new(self)
    }
}

impl<'a, 'str> QCodeMut<'str> for BodyMut<'a, 'str> {
    type View<'v>
        = BodyView<'v, 'str>
    where
        Self: 'v,
        'str: 'v;

    fn function_mut(&mut self, id: FunctionId) -> &mut FunctionBody<'str> {
        BodyMut::function_mut(self, id)
    }

    fn shr(&self) -> &Shared<'str> {
        self.shared
    }

    fn interfaces(&self) -> &Registry<FunctionId, FunctionInterface<'str>> {
        self.interfaces
    }

    fn view(&self) -> BodyView<'_, 'str> {
        BodyMut::view(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{BasicBlock, insn::InstructionId};

    /// The same generic transform runs unchanged over both mutation hosts.
    fn absorb_forwarding_pair<'str>(host: &mut impl QCodeMut<'str>, func: FunctionId) {
        let (keep, other, edge) = {
            let view = host.view();
            let ids = view.function_ref(func).block_ids();
            let [keep, other] = ids[..] else {
                panic!("expected exactly two rostered blocks");
            };
            let edge = *view.block(keep).edges.iter().next().expect("edge");
            (keep, other, edge)
        };
        host.absorb_block(keep, other, edge);
    }

    fn forwarding_pair(ctx: &mut Context<'_>) -> (FunctionId, InstructionId) {
        let func = FunctionBody::make(ctx, "f".into()).unwrap().id;
        let keep = BasicBlock::make(ctx, func).id;
        let other = BasicBlock::make(ctx, func).id;
        ctx.bodies[func].set_root_id(Some(keep.local));
        let value = ctx.get_const(7, 8).id();
        let ret = ctx.builder(other).push_return(value).id;
        ctx.builder(keep).push_branch(other);
        (func, ret)
    }

    #[test]
    fn module_host_runs_generic_transform() {
        let mut ctx = Context::new();
        let (func, ret) = forwarding_pair(&mut ctx);
        absorb_forwarding_pair(&mut ctx, func);
        assert_eq!(ctx.view().function_ref(func).block_ids().len(), 1);
        assert!(ctx.contains_instruction(ret));
    }

    #[test]
    fn checked_out_host_runs_generic_transform() {
        let mut ctx = Context::new();
        let (func, ret) = forwarding_pair(&mut ctx);
        {
            let mut host = BodyMut::new(&mut ctx.bodies[func], &ctx.shared, &ctx.interfaces);
            absorb_forwarding_pair(&mut host, func);
            assert_eq!(
                QCodeMut::view(&host).function_ref(func).block_ids().len(),
                1
            );
        }
        assert!(ctx.contains_instruction(ret));
    }

    #[test]
    fn shared_value_rauw_is_a_noop() {
        let mut ctx = Context::new();
        let (_, ret) = forwarding_pair(&mut ctx);
        let lit = ctx.get_const(7, 8).id();
        let other = ctx.get_const(9, 8).id();
        QCodeMut::replace_all_uses_with(&mut ctx, lit, other);
        assert!(ctx.contains_instruction(ret));
    }
}
