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
        insn::{Instruction, InstructionId, Mnemonic},
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
    /// The rest of the module, **read-only**. A checked-out function pass reaches
    /// shared data (types, literals, spaces, registers, other functions'
    /// interface) immutably; it mints types/literals through the interners'
    /// `&self` paths, and mints no varnodes/temp-spaces (only the V1 argpromote
    /// pass does, and it runs on the module path).
    pub shared: &'a Context<'str>,
}

impl<'a, 'str> CheckedOut<'a, 'str> {
    /// Wrap `fun` (checked out of `shared` under `id`). Debug-asserts the function
    /// owns only self-stored, self-parented blocks (no reattribution).
    pub fn new(fun: &'a mut Function<'str>, id: FunctionId, shared: &'a Context<'str>) -> Self {
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
            shared: self.shared,
        }
    }
}

/// The write side of the arena-routing abstraction. Read access is provided by
/// [`read_host`](HostMut::read_host) (a [`HostRef`]); everything mutable — arena
/// writes, births, the CFG/use-map verbs, local-name registration — routes here.
pub trait HostMut<'str> {
    /// A shorter-lived reborrow of this host, so a value that owns its host by
    /// value — a [`Builder`](crate::builder::Builder) or a mutation `BaseRef` — can
    /// be built from `&mut self` without consuming the original. `&mut Context`
    /// reborrows to `&mut Context`; [`CheckedOut`] to a shorter [`CheckedOut`].
    type Reborrowed<'b>: HostMut<'str>
    where
        Self: 'b;
    /// Reborrow this host (see [`Reborrowed`](HostMut::Reborrowed)).
    fn reborrow_host(&mut self) -> Self::Reborrowed<'_>;

    /// The owning function's storage (write). Panics if `f` is not routable by
    /// this host (a checked-out host only owns its one function).
    fn function_mut<'b>(&'b mut self, f: FunctionId) -> &'b mut Function<'str>;
    /// The owning function's storage (read).
    fn function<'b>(&'b self, f: FunctionId) -> &'b Function<'str>;
    /// The module's shared data (read) — types, literals, spaces, registers, maps.
    /// Type and literal minting go through the interners' `&self` paths, so this
    /// read view suffices for everything a function pass mints.
    fn shared<'b>(&'b self) -> &'b Context<'str>;
    /// The module's shared data (write) — only for minting *varnodes*/temp-spaces,
    /// which require `&mut`. Available on the module path; a checked-out host has
    /// only immutable shared access and panics here (no function pass mints
    /// varnodes — only the V1 argpromote pass does, on the module path).
    fn shared_mut<'b>(&'b mut self) -> &'b mut Context<'str> {
        unimplemented!("a checked-out host has read-only shared access (mints via &self)")
    }
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

    /// Mint a fresh `Int(size)`-typed instruction with `mnemonic` into `func`'s
    /// arena (host-routed equivalent of `InstructionRef::from_mnemonic`): the type
    /// is minted in shared storage through the interner's `&self` path.
    fn push_mnemonic(
        &mut self,
        func: FunctionId,
        mnemonic: Mnemonic,
        size: usize,
    ) -> InstructionId {
        let type_id = self.shared().types.get_or_make_int(size);
        let insn = Instruction::new(type_id, mnemonic);
        self.push_insn(func, insn)
    }

    /// Mint an instruction with `mnemonic` and an explicit result `type_id` into
    /// `func`'s arena (host-routed equivalent of `InstructionRef::from_mnemonic_with_type`),
    /// preserving a non-`Int` result type (e.g. a hoisted load's aggregate type).
    fn push_mnemonic_with_type(
        &mut self,
        func: FunctionId,
        mnemonic: Mnemonic,
        type_id: crate::types::TypeId,
    ) -> InstructionId {
        let insn = Instruction::new(type_id, mnemonic);
        self.push_insn(func, insn)
    }

    /// Insert `insn` immediately before `before` in `block`, setting its parent.
    /// Panics if `before` is not in `block`.
    fn insert_insn_before(&mut self, block: BlockId, before: InstructionId, insn: InstructionId) {
        let index = self
            .read_host()
            .block(block)
            .instructions
            .iter()
            .position(|&i| i == before)
            .expect("before not in block");
        self.instruction_mut(insn).parent = Some(block);
        self.block_mut(block).instructions.insert(index, insn);
    }

    // ---- CFG / use-map verbs (mirror the `Context` inherent methods) ---------

    /// Whether this host may mutate `block`'s arena. Always `true` on the module
    /// path; on a checked-out host, only the checked-out function's own blocks.
    /// Used to skip the *other* endpoint of a cross-function CFG edge (a
    /// thunk/tail-call `Branch` into another function): a checked-out pass leaves
    /// the foreign block's edge set untouched rather than panicking. The module
    /// path (which owns every function) updates both, exactly as before.
    fn owns_block(&self, block: BlockId) -> bool {
        let _ = block;
        true
    }

    /// Adds a directed CFG edge `from -> to`, stored in `from`'s edge arena and
    /// linked into both incident blocks' edge sets. Mirrors
    /// [`Context::add_cfg_edge`]. `from` must be owned (the edge lives in its
    /// arena); the `to` endpoint is skipped if it is a foreign block.
    fn add_cfg_edge(&mut self, from: BlockId, to: BlockId) -> EdgeId {
        let edge_id = self.push_edge(from.func, EdgeData { from, to });
        if self.owns_block(from) {
            self.block_mut(from).edges.insert(edge_id);
        }
        if self.owns_block(to) {
            self.block_mut(to).edges.insert(edge_id);
        }
        edge_id
    }

    /// Removes a CFG edge, unlinking it from both incident blocks it owns. The
    /// backing `EdgeData` slot is left dangling. Mirrors [`Context::remove_cfg_edge`].
    fn remove_cfg_edge(&mut self, edge_id: EdgeId) {
        let EdgeData { from, to } = *self.read_host().edge(edge_id);
        if self.owns_block(from) {
            self.block_mut(from).edges.remove(&edge_id);
        }
        if self.owns_block(to) {
            self.block_mut(to).edges.remove(&edge_id);
        }
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

    /// Rehome `remove`'s outgoing CFG edges onto `keep` and drop the direct edge
    /// between them (host-routed mirror of `jstd`'s `Graph::merge_nodes`, which
    /// operates over the whole-context graph). The caller tombstones `remove`.
    fn merge_nodes(&mut self, keep: BlockId, remove: BlockId, direct_edge: EdgeId) {
        self.block_mut(keep).edges.remove(&direct_edge);
        self.block_mut(remove).edges.remove(&direct_edge);
        let outgoing: Vec<EdgeId> = {
            let host = self.read_host();
            host.block(remove)
                .edges
                .iter()
                .copied()
                .filter(|&e| host.edge(e).from == remove)
                .collect()
        };
        for eid in outgoing {
            self.function_mut(eid.func).edges[eid.local].from = keep;
            self.block_mut(keep).edges.insert(eid);
            self.block_mut(remove).edges.remove(&eid);
        }
    }

    /// Replace an instruction's mnemonic in place, keeping the reverse use-map and
    /// the call-site cache in sync (host-routed mirror of
    /// [`Context::replace_instruction_mnemonic`]).
    fn replace_instruction_mnemonic(&mut self, id: InstructionId, mnemonic: Mnemonic) {
        let func = id.func;
        let (old_args, old_target) = {
            let m = self.read_host().instruction(id).mnemonic();
            (m.args().into_iter().collect::<Vec<_>>(), m.call_target())
        };
        for arg in old_args {
            let now_empty = if let Some(users) = self.function_mut(func).users.get_mut(&arg) {
                users.retain(|&u| u != id);
                users.is_empty()
            } else {
                false
            };
            if now_empty {
                self.function_mut(func).users.remove(&arg);
            }
        }
        if let Some(target) = old_target {
            self.forget_call_site(target, id);
        }
        *self.instruction_mut(id).mnemonic_mut() = mnemonic;
        let (new_args, new_target) = {
            let m = self.read_host().instruction(id).mnemonic();
            (m.args().into_iter().collect::<Vec<_>>(), m.call_target())
        };
        for arg in new_args {
            self.function_mut(func)
                .users
                .entry(arg)
                .or_default()
                .push(id);
        }
        if let Some(target) = new_target {
            self.record_call_site(target, id);
        }
    }

    /// Drop `block` from the roster of whatever function owns it.
    fn unroster_block(&mut self, block: BlockId) {
        let owner = self.read_host().block(block).parent;
        if let Some(f) = owner {
            self.function_mut(f).roster.retain(|&b| b != block);
        }
        self.function_mut(block.func).roster.retain(|&b| b != block);
    }

    /// Remove `block` from its function: unlink every incident CFG edge, remove
    /// its instructions, detach its params, and tombstone it. Host-routed port of
    /// the former `BlockMutRef::delete`.
    fn delete_block(&mut self, block: BlockId, _function_id: FunctionId) {
        let edges: Vec<EdgeId> = self
            .read_host()
            .block(block)
            .edges
            .iter()
            .copied()
            .collect();
        for edge in edges {
            self.remove_cfg_edge(edge);
        }
        let insns: Vec<InstructionId> = self.read_host().block(block).instructions.clone();
        for insn in insns {
            self.remove_instruction(insn);
        }
        let params: Vec<BlockParamId> = self.read_host().block(block).params.clone();
        for param in params {
            self.function_mut(param.func)
                .users
                .remove(&ValueId::BlockParam(param));
            self.block_param_mut(param).parent = None;
        }
        self.unroster_block(block);
        let b = self.block_mut(block);
        b.parent = None;
        b.deleted = true;
    }

    /// Absorb `other` into `keep`: drop `keep`'s terminal branch, append `other`'s
    /// instructions, rehome its outgoing edges, and tombstone it. `edge_ab` is the
    /// direct edge `keep -> other`. Host-routed port of `BlockMutRef::absorb_block`.
    fn absorb_block(
        &mut self,
        keep: BlockId,
        other: BlockId,
        edge_ab: EdgeId,
        _function_id: FunctionId,
    ) {
        let branch_args = {
            let host = self.read_host();
            host.block(keep)
                .instructions
                .last()
                .and_then(|&id| match host.instruction(id).mnemonic() {
                    Mnemonic::Branch(branch) if branch.target == other => Some(branch.args.clone()),
                    _ => None,
                })
                .unwrap_or_default()
        };
        let other_params = self.read_host().block(other).params.clone();
        if !other_params.is_empty() {
            assert_eq!(
                other_params.len(),
                branch_args.len(),
                "cannot absorb block with {} params through branch with {} args",
                other_params.len(),
                branch_args.len()
            );
            for (param, arg) in other_params.into_iter().zip(branch_args) {
                self.replace_all_uses_with(ValueId::BlockParam(param), arg);
            }
        }
        self.block_mut(keep).instructions.pop();
        let b_insns = std::mem::take(&mut self.block_mut(other).instructions);
        for &insn_id in &b_insns {
            self.instruction_mut(insn_id).parent = Some(keep);
        }
        self.block_mut(keep).instructions.extend(b_insns);
        self.merge_nodes(keep, other, edge_ab);
        let (b_addr, b_extra) = {
            let b = self.read_host().block(other);
            (b.address, b.extra_addresses.clone())
        };
        self.unroster_block(other);
        {
            let ob = self.block_mut(other);
            ob.parent = None;
            ob.deleted = true;
        }
        if let Some(addr) = b_addr {
            self.block_mut(keep).extra_addresses.push(addr);
        }
        self.block_mut(keep).extra_addresses.extend(b_extra);
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
    type Reborrowed<'b>
        = &'b mut Context<'str>
    where
        Self: 'b;
    fn reborrow_host(&mut self) -> &mut Context<'str> {
        self
    }
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
    type Reborrowed<'b>
        = CheckedOut<'b, 'str>
    where
        Self: 'b;
    fn reborrow_host(&mut self) -> CheckedOut<'_, 'str> {
        self.reborrow()
    }
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
    fn owns_block(&self, block: BlockId) -> bool {
        block.func == self.id
    }
    // `shared_mut` intentionally not implemented: a checked-out host is read-only
    // on shared data (it uses the defaulted panic).
    fn read_host(&self) -> HostRef<'_, 'str> {
        HostRef::Checked {
            fun: &*self.fun,
            shared: self.shared,
            id: self.id,
        }
    }
    // A checked-out pass never touches the global call-site cache; the driver
    // rebuilds it by diffing outgoing calls at check-in.
    fn record_call_site(&mut self, _target: FunctionId, _site: InstructionId) {}
    fn forget_call_site(&mut self, _target: FunctionId, _site: InstructionId) {}
}
