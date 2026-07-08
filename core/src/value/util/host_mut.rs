//! The exclusive *mutation* host (Stage 5.3a step 2 of the parallel-function-
//! passes plan; see `PARALLEL_PASSES.md`).
//!
//! [`HostRef`](super::base_ref::HostRef) gave the read layer a `Copy` view that
//! routes arena reads to a checked-out function. This module is its mutable
//! sibling: [`HostMut`] abstracts "where a function's arenas live" for *writes*,
//! implemented for both the whole module (`&mut Context`, behaviour-identical to
//! today) and a single checked-out function ([`CheckedOut`]).
//!
//! A checked-out function has been swapped out of the module registry (its slot
//! holds a [`Function::sentinel`]); its arenas live in an owned `&mut Function`,
//! and all *shared* data (types, varnodes, spaces, registers, name map) stays in
//! the module. `HostMut` routes each composite-id write to the arena that owns it:
//! the owned function when `id.func` is the checked-out one, and it **panics** on
//! any other function — a function pass must not mutate another function.
//!
//! The generic mutation-ref methods and the CFG/use-map verbs are written once
//! against `HostMut`, so they work over either host with no `&mut Context`
//! reborrow tricks (the checked-out host is passed by an explicit
//! [`CheckedOut::reborrow`]).

use crate::{
    context::Context,
    error::{Error, ErrorTy, Result},
    value::{
        Function, FunctionId, ValueId,
        block::{BasicBlock, BlockId, EdgeData, EdgeId},
        block_param::{BlockParam, BlockParamId},
        insn::{Instruction, InstructionId},
    },
};

use super::base_ref::HostRef;

/// A single function checked out of `shared` for exclusive mutation. `fun`'s slot
/// in `shared.values.functions[id]` currently holds a [`Function::sentinel`]; its
/// real arenas are owned here.
///
/// Out of scope (and asserted against on construction): a function with
/// *reattributed* blocks — a roster block stored in, or parented to, a different
/// function. Those functions go through the sequential (module) path.
pub struct CheckedOut<'a, 'str> {
    pub fun: &'a mut Function<'str>,
    pub id: FunctionId,
    pub shared: &'a mut Context<'str>,
}

impl<'a, 'str> CheckedOut<'a, 'str> {
    /// Wrap `fun` (checked out of `shared` under `id`). Debug-asserts the function
    /// owns only self-stored, self-parented blocks (no reattribution).
    pub fn new(fun: &'a mut Function<'str>, id: FunctionId, shared: &'a mut Context<'str>) -> Self {
        debug_assert!(
            fun.roster.iter().all(|b| {
                // Stored in this function's own arena, and (if live) parented to it.
                b.func == id && {
                    let blk = &fun.blocks[b.local];
                    blk.deleted || blk.parent == Some(id)
                }
            }),
            "CheckedOut requires a function with no reattributed blocks"
        );
        Self { fun, id, shared }
    }

    /// A shorter-lived `CheckedOut` reborrowing this one's exclusive references, so
    /// the host can be handed to a mutation ref (which owns its host by value)
    /// without consuming the original.
    pub fn reborrow(&mut self) -> CheckedOut<'_, 'str> {
        CheckedOut {
            fun: &mut *self.fun,
            id: self.id,
            shared: &mut *self.shared,
        }
    }
}

/// The write side of the arena-routing abstraction. Read access is provided by
/// [`read_host`](HostMut::read_host) (a [`HostRef`]); everything mutable — arena
/// writes, births, the CFG/use-map verbs, local-name registration — routes here.
pub trait HostMut<'str> {
    /// The owning function's storage (write). Panics if `f` is not routable by
    /// this host (a checked-out host only owns its one function).
    fn function_mut<'b>(&'b mut self, f: FunctionId) -> &'b mut Function<'str>;
    /// The owning function's storage (read).
    fn function<'b>(&'b self, f: FunctionId) -> &'b Function<'str>;
    /// The module's shared data (read) — types, varnodes, spaces, registers, maps.
    fn shared<'b>(&'b self) -> &'b Context<'str>;
    /// The module's shared data (write) — for minting types/varnodes/spaces.
    fn shared_mut<'b>(&'b mut self) -> &'b mut Context<'str>;
    /// A `Copy` read view over this host, for the mutation refs' read methods.
    fn read_host<'b>(&'b self) -> HostRef<'b, 'str>;

    /// Record `site` as a call site of `target` in the global cache. A no-op for a
    /// checked-out host: the driver rebuilds `call_sites` by diffing at check-in
    /// (`PARALLEL_PASSES.md`, ruling 6).
    fn record_call_site(&mut self, target: FunctionId, site: InstructionId);
    /// Drop `site` from `target`'s call-site list. No-op for a checked-out host.
    fn forget_call_site(&mut self, target: FunctionId, site: InstructionId);

    // ---- derived arena accessors -------------------------------------------

    fn instruction_mut(&mut self, id: InstructionId) -> &mut Instruction<'str> {
        &mut self.function_mut(id.func).insns[id.local]
    }
    fn block_mut(&mut self, id: BlockId) -> &mut BasicBlock<'str> {
        &mut self.function_mut(id.func).blocks[id.local]
    }
    fn block_param_mut(&mut self, id: BlockParamId) -> &mut BlockParam<'str> {
        &mut self.function_mut(id.func).params[id.local]
    }

    // ---- births -------------------------------------------------------------

    fn push_edge(&mut self, func: FunctionId, edge: EdgeData) -> EdgeId {
        let local = self.function_mut(func).edges.push(edge);
        EdgeId::new(func, local)
    }

    /// Push a fresh instruction into `func`'s arena, recording each operand's use
    /// in that function's reverse-use map and (for a direct call) the call-site
    /// cache. Mirrors [`crate::value::registry::ValueRegistry::push_insn`].
    fn push_insn(&mut self, func: FunctionId, insn: Instruction<'str>) -> InstructionId {
        let args: Vec<ValueId> = insn.mnemonic().args().into_iter().collect();
        let call_target = insn.mnemonic().call_target();
        let local = self.function_mut(func).insns.push(insn);
        let id = InstructionId::new(func, local);
        for arg in args {
            self.function_mut(func)
                .users
                .entry(arg)
                .or_default()
                .push(id);
        }
        if let Some(target) = call_target {
            self.record_call_site(target, id);
        }
        id
    }

    /// Push a fresh block into `func`'s arena and onto its ownership roster.
    /// Mirrors [`crate::value::registry::ValueRegistry::push_block`].
    fn push_block(&mut self, func: FunctionId, block: BasicBlock<'str>) -> BlockId {
        let local = self.function_mut(func).blocks.push(block);
        let id = BlockId::new(func, local);
        self.function_mut(func).roster.push(id);
        id
    }

    /// Push a fresh block parameter into `func`'s arena.
    fn push_block_param(&mut self, func: FunctionId, param: BlockParam<'str>) -> BlockParamId {
        let local = self.function_mut(func).params.push(param);
        BlockParamId::new(func, local)
    }

    // ---- CFG / use-map verbs (mirror the `Context` inherent methods) ---------

    /// Adds a directed CFG edge `from -> to`, stored in `from`'s edge arena and
    /// linked into both incident blocks' edge sets. Mirrors
    /// [`Context::add_cfg_edge`].
    fn add_cfg_edge(&mut self, from: BlockId, to: BlockId) -> EdgeId {
        let edge_id = self.push_edge(from.func, EdgeData { from, to });
        self.block_mut(from).edges.insert(edge_id);
        self.block_mut(to).edges.insert(edge_id);
        edge_id
    }

    /// Removes a CFG edge, unlinking it from both incident blocks. The backing
    /// `EdgeData` slot is left dangling. Mirrors [`Context::remove_cfg_edge`].
    fn remove_cfg_edge(&mut self, edge_id: EdgeId) {
        let EdgeData { from, to } = *self.read_host().edge(edge_id);
        self.block_mut(from).edges.remove(&edge_id);
        self.block_mut(to).edges.remove(&edge_id);
    }

    /// Replaces every use of `old` with `new` across its owning function's
    /// instructions and updates the reverse use-map. Mirrors
    /// [`Context::replace_all_uses_with`] (SSA defs only; `old` is intra-function).
    fn replace_all_uses_with(&mut self, old: ValueId, new: ValueId) {
        if old == new {
            return;
        }
        let Some(func) = old.owning_function() else {
            return;
        };
        let users: Vec<InstructionId> = self.function(func).users_of(old).to_vec();
        for user in users {
            self.instruction_mut(user)
                .mnemonic_mut()
                .replace_value(old, new);
            self.function_mut(func)
                .users
                .entry(new)
                .or_default()
                .push(user);
        }
        self.function_mut(func).users.remove(&old);
    }

    /// Removes an instruction from its block, unlinks its outgoing CFG edges if it
    /// was a terminator, clears its (function-local) name, tombstones it, and
    /// prunes it from its operands' use-lists. Mirrors [`Context::remove_instruction`]
    /// composed with [`crate::value::registry::ValueRegistry::remove_instructions`],
    /// minus the global `call_sites` write on the checked-out path (see
    /// [`forget_call_site`](HostMut::forget_call_site)).
    fn remove_instruction(&mut self, id: InstructionId) {
        // Read everything needed up front through the read view, so the exclusive
        // mutations below don't overlap the borrow.
        let (parent, name, is_terminator, args, target) = {
            let insn = self.read_host().instruction(id);
            (
                insn.parent,
                insn.name.clone(),
                insn.mnemonic().is_terminator(),
                insn.mnemonic().args().into_iter().collect::<Vec<_>>(),
                insn.mnemonic().call_target(),
            )
        };

        if let Some(block_id) = parent {
            self.block_mut(block_id).instructions.retain(|&i| i != id);
            if is_terminator {
                let succ: Vec<EdgeId> = {
                    let host = self.read_host();
                    let block = host.block(block_id);
                    block
                        .edges
                        .iter()
                        .copied()
                        .filter(|&e| host.edge(e).from == block_id)
                        .collect()
                };
                for edge_id in succ {
                    self.remove_cfg_edge(edge_id);
                }
            }
        }

        self.instruction_mut(id).parent = None;

        if let Some(n) = name {
            self.function_mut(id.func).names.forget(n.as_ref());
        }
        self.instruction_mut(id).name = None;

        // Tombstone + prune operand use-lists + call-site cache (mirrors
        // `remove_instructions` for a single id).
        self.instruction_mut(id).deleted = true;
        for arg in args {
            if let Some(users) = self.function_mut(id.func).users.get_mut(&arg) {
                users.retain(|u| *u != id);
            }
        }
        if let Some(target) = target {
            self.forget_call_site(target, id);
        }
    }

    // ---- names (function-local for block/instruction/param) ------------------

    /// Register `name` for `id` in the table that owns its kind — the *owning
    /// function's* local table for block/instruction/param, the global map
    /// otherwise. Mirrors [`crate::value::util::named::update_context_name`] but
    /// routes the function-local case to this host (so a checked-out function's
    /// names live in its owned arena, not the sentinel).
    fn register_local_name(
        &mut self,
        id: ValueId,
        name: std::borrow::Cow<'str, str>,
        old_name: Option<&str>,
    ) -> Result<()> {
        let existing = match id.name_scope_function() {
            Some(func) => self.function(func).names.get(&name),
            None => self.shared().get_named(&name),
        };
        if let Some(existing) = existing {
            return if existing == id {
                Ok(())
            } else {
                Err(Error::spanless(ErrorTy::DuplicateName(name.to_string())))
            };
        }
        match id.name_scope_function() {
            Some(func) => self.function_mut(func).names.register(name, id, old_name),
            None => self.shared_mut().update_name(name, id, old_name),
        }
    }
}

impl<'str> HostMut<'str> for &mut Context<'str> {
    fn function_mut(&mut self, f: FunctionId) -> &mut Function<'str> {
        &mut self.values.functions[f]
    }
    fn function(&self, f: FunctionId) -> &Function<'str> {
        &self.values.functions[f]
    }
    fn shared(&self) -> &Context<'str> {
        self
    }
    fn shared_mut(&mut self) -> &mut Context<'str> {
        self
    }
    fn read_host(&self) -> HostRef<'_, 'str> {
        HostRef::Module(self)
    }
    fn record_call_site(&mut self, target: FunctionId, site: InstructionId) {
        self.values.call_sites.entry(target).or_default().push(site);
    }
    fn forget_call_site(&mut self, target: FunctionId, site: InstructionId) {
        if let Some(sites) = self.values.call_sites.get_mut(&target) {
            sites.retain(|s| *s != site);
        }
    }
}

impl<'a, 'str> HostMut<'str> for CheckedOut<'a, 'str> {
    fn function_mut(&mut self, f: FunctionId) -> &mut Function<'str> {
        assert_eq!(
            f, self.id,
            "a checked-out function pass may not mutate another function"
        );
        self.fun
    }
    fn function(&self, f: FunctionId) -> &Function<'str> {
        assert_eq!(
            f, self.id,
            "a checked-out function pass may not read another function's arenas mutably"
        );
        self.fun
    }
    fn shared(&self) -> &Context<'str> {
        self.shared
    }
    fn shared_mut(&mut self) -> &mut Context<'str> {
        self.shared
    }
    fn read_host(&self) -> HostRef<'_, 'str> {
        HostRef::Checked {
            fun: &*self.fun,
            shared: &*self.shared,
            id: self.id,
        }
    }
    // A checked-out pass never touches the global call-site cache; the driver
    // rebuilds it by diffing outgoing calls at check-in.
    fn record_call_site(&mut self, _target: FunctionId, _site: InstructionId) {}
    fn forget_call_site(&mut self, _target: FunctionId, _site: InstructionId) {}
}
