//! The exclusive *mutation* host for a function pass ([`PassBacking`]).
//!
//! [`HostRef`](super::base_ref::HostRef) gives the read layer a `Copy` view that
//! routes arena reads to the pass's own function. [`PassBacking`] is its mutable
//! sibling: a single function's arenas borrowed `&mut` in place from
//! `Context.bodies[id]` for exclusive mutation by one worker (the driver's
//! `Context::split` hands out disjoint body borrows).
//!
//! All *shared* data (types, varnodes, spaces, registers, name map) stays behind
//! the `&Shared` view, reachable read-only. The inherent verb + read methods below
//! route each write to the borrowed function's arena; a function pass must not mutate another
//! function (asserted). The module-scope twin of every verb is an inherent method
//! on [`Context`](crate::context::Context); the shorter-lived reborrow needed to
//! hand the host to a value that owns it by value (a [`Builder`](crate::builder::Builder)
//! or a mutation `BaseRef`) is [`PassBacking::reborrow`].

use crate::{
    context::Context,
    error::{Error, ErrorTy, Result},
    value::{
        FunctionBody, FunctionId, ValueId,
        block::{BasicBlock, BlockId, EdgeData, EdgeId},
        block_param::{BlockParam, BlockParamId},
        insn::{Instruction, InstructionId, Mnemonic},
    },
};

use super::base_ref::HostRef;

/// A single function borrowed `&mut` in place from `Context.bodies[id]` for
/// exclusive mutation (its interface stays in `Context.interfaces[id]`,
/// reachable read-only through `interfaces`).
///
/// Out of scope (and asserted against on construction): a function with
/// *reattributed* blocks — a roster block stored in, or parented to, a different
/// function. Those functions go through the sequential (module) path.
pub struct PassBacking<'a, 'str> {
    pub fun: &'a mut FunctionBody<'str>,
    /// The module's shared IR state, **read-only**. A checked-out function pass
    /// reaches shared data (types, literals, spaces, registers) immutably; it
    /// mints types/literals through the interners' `&self` paths, and mints no
    /// varnodes/temp-spaces (only the V1 argpromote pass does, and it runs on
    /// the module path). Holds **no** `&Context` — bodies are out of reach by
    /// construction (context-split stage 5b-ii Pin B).
    pub shared: &'a crate::context::Shared<'str>,
    /// Every function's published interface (never checked out): the
    /// caller-reasoning surface a pass may consult about its callees.
    pub interfaces:
        &'a jstd::registry::Registry<FunctionId, crate::value::function::FunctionInterface<'str>>,
}

impl<'a, 'str> PassBacking<'a, 'str> {
    /// Wrap `fun` over the module's shared state and
    /// interface registry. Debug-asserts the function owns only self-stored,
    /// self-parented blocks (no reattribution).
    pub fn new(
        fun: &'a mut FunctionBody<'str>,
        shared: &'a crate::context::Shared<'str>,
        interfaces: &'a jstd::registry::Registry<
            FunctionId,
            crate::value::function::FunctionInterface<'str>,
        >,
    ) -> Self {
        let id = fun.id();
        assert!(
            fun.roster.iter().all(|&local| {
                // Stored in this function's own arena, and (if live) parented to it.
                let blk = &fun.blocks[local];
                blk.deleted || blk.parent == Some(id)
            }),
            "PassBacking requires a function with no reattributed blocks"
        );
        Self {
            fun,
            shared,
            interfaces,
        }
    }

    /// Wrap `fun` (checked out under `id`) over a whole module `&Context` — the
    /// module/test-scope convenience constructor (narrows to the shared state +
    /// interface registry).
    pub fn from_ctx(fun: &'a mut FunctionBody<'str>, ctx: &'a Context<'str>) -> Self {
        Self::new(fun, &ctx.shared, &ctx.interfaces)
    }

    /// A shorter-lived `PassBacking` reborrowing this one's exclusive references, so
    /// the host can be handed to a mutation ref (which owns its host by value)
    /// without consuming the original.
    pub fn reborrow(&mut self) -> PassBacking<'_, 'str> {
        PassBacking {
            fun: &mut *self.fun,
            shared: self.shared,
            interfaces: self.interfaces,
        }
    }
}

/// The verb + read surface of a checked-out function pass, delegating to the
/// owned `FunctionBody`'s inherent verbs and `self.shared`. The module-scope twin of
/// each verb is an inherent method on [`Context`](crate::context::Context); the
/// primitives below (`function{,_mut}`/`shared`/`read_host`, and the no-op
/// call-site cache) are the checked-out specializations.
impl<'a, 'str> PassBacking<'a, 'str> {
    // ---- primitives ---------------------------------------------------------

    /// The owned function's storage (write). Panics if `f` is not this function.
    pub fn function_mut(&mut self, f: FunctionId) -> &mut FunctionBody<'str> {
        assert_eq!(
            f,
            self.fun.id(),
            "a checked-out function pass may not mutate another function"
        );
        self.fun
    }
    /// The owned function's storage (read).
    pub fn function(&self, f: FunctionId) -> &FunctionBody<'str> {
        assert_eq!(
            f,
            self.fun.id(),
            "a checked-out function pass may not read another function's arenas mutably"
        );
        self.fun
    }
    /// The module's shared IR state ([`Shared`]) (read).
    ///
    /// [`Shared`]: crate::context::Shared
    pub fn shr(&self) -> &crate::context::Shared<'str> {
        self.shared
    }
    /// A `Copy` read view over this host, for the mutation refs' read methods.
    pub fn read_host(&self) -> HostRef<'_, 'str> {
        HostRef::Checked {
            fun: &*self.fun,
            shared: self.shared,
            interfaces: self.interfaces,
        }
    }
    /// A pass body never touches the global call-site cache; the driver
    /// rebuilds it by diffing outgoing calls at the barrier.
    fn record_call_site(&mut self, _target: FunctionId, _site: InstructionId) {}
    fn forget_call_site(&mut self, _target: FunctionId, _site: InstructionId) {}

    // ---- function-scoped read wrappers --------------------------------------

    /// A read [`BlockRef`](crate::value::BlockRef) over `id`, body-routed.
    pub fn block_ref(&self, id: BlockId) -> super::base_ref::BaseRef<HostRef<'_, 'str>, BlockId> {
        self.read_host().block_ref(id)
    }
    /// A read [`InstructionRef`](crate::value::InstructionRef) over `id`.
    pub fn insn_ref(
        &self,
        id: InstructionId,
    ) -> super::base_ref::BaseRef<HostRef<'_, 'str>, InstructionId> {
        self.read_host().insn_ref(id)
    }
    /// A read [`BlockParamRef`](crate::value::BlockParamRef) over `id`.
    pub fn param_ref(
        &self,
        id: BlockParamId,
    ) -> super::base_ref::BaseRef<HostRef<'_, 'str>, BlockParamId> {
        self.read_host().param_ref(id)
    }
    /// A read [`FunctionRef`](crate::value::FunctionRef) over `id`.
    pub fn function_ref(
        &self,
        id: FunctionId,
    ) -> super::base_ref::BaseRef<HostRef<'_, 'str>, FunctionId> {
        self.read_host().function_ref(id)
    }

    // ---- derived arena accessors --------------------------------------------

    pub fn instruction_mut(&mut self, id: InstructionId) -> &mut Instruction<'str> {
        &mut self.function_mut(id.func).insns[id.local]
    }
    pub fn block_mut(&mut self, id: BlockId) -> &mut BasicBlock<'str> {
        &mut self.function_mut(id.func).blocks[id.local]
    }
    pub fn block_param_mut(&mut self, id: BlockParamId) -> &mut BlockParam<'str> {
        &mut self.function_mut(id.func).params[id.local]
    }

    /// Physically removes a block parameter and its local bookkeeping.
    /// Positional block and edge-argument rewrites belong to the caller and may
    /// complete later in the same transformation.
    pub fn remove_block_param(&mut self, id: BlockParamId) {
        self.function_mut(id.func).remove_block_param(id);
    }

    // ---- births -------------------------------------------------------------

    pub fn push_edge(&mut self, func: FunctionId, edge: EdgeData) -> EdgeId {
        self.function_mut(func).edges.push(edge)
    }

    pub fn push_insn(&mut self, func: FunctionId, insn: Instruction<'str>) -> InstructionId {
        let args: Vec<crate::value::LocalValueId> = insn.mnemonic().args().into_iter().collect();
        let call_target = insn.mnemonic().call_target();
        let local = self.function_mut(func).insns.push(insn);
        let id = InstructionId::new(func, local);
        for arg in args {
            self.function_mut(func)
                .users
                .entry(arg)
                .or_default()
                .push(id.localize(func));
        }
        if let Some(target) = call_target {
            self.record_call_site(target, id);
        }
        id
    }

    pub fn push_block(&mut self, func: FunctionId, block: BasicBlock<'str>) -> BlockId {
        let local = self.function_mut(func).blocks.push(block);
        let id = BlockId::new(func, local);
        self.function_mut(func).roster.push(local);
        id
    }

    pub fn make_block(&mut self, func: FunctionId) -> BlockId {
        self.push_block(func, BasicBlock::detached(func))
    }

    pub fn push_block_param(&mut self, func: FunctionId, param: BlockParam<'str>) -> BlockParamId {
        let local = self.function_mut(func).params.push(param);
        BlockParamId::new(func, local)
    }

    pub fn push_mnemonic(
        &mut self,
        func: FunctionId,
        mnemonic: Mnemonic,
        size: usize,
    ) -> InstructionId {
        let type_id = self.shr().types.get_or_make_int(size);
        let insn = Instruction::new(type_id, mnemonic);
        self.push_insn(func, insn)
    }

    pub fn push_mnemonic_with_type(
        &mut self,
        func: FunctionId,
        mnemonic: Mnemonic,
        type_id: crate::types::TypeId,
    ) -> InstructionId {
        let insn = Instruction::new(type_id, mnemonic);
        self.push_insn(func, insn)
    }

    pub fn insert_insn_before(
        &mut self,
        block: BlockId,
        before: InstructionId,
        insn: InstructionId,
    ) {
        let index = self
            .read_host()
            .block(block)
            .instructions
            .iter()
            .position(|&local| InstructionId::new(block.func, local) == before)
            .expect("before not in block");
        self.instruction_mut(insn).parent = Some(block.local);
        self.block_mut(block)
            .instructions
            .insert(index, insn.localize(block.func));
    }

    // ---- CFG / use-map verbs ------------------------------------------------

    pub fn add_cfg_edge(&mut self, from: BlockId, to: BlockId) -> EdgeId {
        let edge_id = self.push_edge(from.func, EdgeData { from, to });
        self.block_mut(from).edges.insert(edge_id);
        self.block_mut(to).edges.insert(edge_id);
        edge_id
    }

    pub fn remove_cfg_edge(&mut self, func: FunctionId, edge_id: EdgeId) {
        let EdgeData { from, to } = *self.read_host().edge(func, edge_id);
        self.block_mut(from).edges.remove(&edge_id);
        self.block_mut(to).edges.remove(&edge_id);
        self.function_mut(func).edges.remove(edge_id);
    }

    pub fn replace_all_uses_with(&mut self, old: ValueId, new: ValueId) {
        if old == new {
            return;
        }
        let Some(func) = old.owning_function() else {
            return;
        };
        let users: Vec<_> = self
            .function(func)
            .local_users_of(old)
            .iter()
            .map(|&local| InstructionId::new(func, local))
            .collect();
        let old = old.localize(func);
        let new = new.localize(func);
        for user in users {
            self.instruction_mut(user)
                .mnemonic_mut()
                .replace_value(old, new);
            self.function_mut(func)
                .users
                .entry(new)
                .or_default()
                .push(user.localize(func));
        }
        self.function_mut(func).users.remove(&old);
    }

    pub fn remove_instruction(&mut self, id: InstructionId) {
        let (parent, name, is_terminator, args, target) = {
            let insn = self.read_host().instruction(id);
            (
                insn.parent.map(|l| BlockId::new(id.func, l)),
                insn.name.clone(),
                insn.mnemonic().is_terminator(),
                insn.mnemonic().args().into_iter().collect::<Vec<_>>(),
                insn.mnemonic().call_target(),
            )
        };

        if let Some(block_id) = parent {
            self.block_mut(block_id)
                .instructions
                .retain(|&local| local != id.localize(block_id.func));
            if is_terminator {
                let mut succ: Vec<EdgeId> = {
                    let host = self.read_host();
                    let block = host.block(block_id);
                    block
                        .edges
                        .iter()
                        .copied()
                        .filter(|&e| host.edge(block_id.func, e).from == block_id)
                        .collect()
                };
                succ.sort_unstable();
                for edge_id in succ {
                    self.remove_cfg_edge(block_id.func, edge_id);
                }
            }
        }

        if let Some(n) = name {
            self.function_mut(id.func).names.forget(n.as_ref());
        }
        for arg in args {
            if let Some(users) = self.function_mut(id.func).users.get_mut(&arg) {
                users.retain(|&local| local != id.localize(id.func));
            }
        }
        if let Some(target) = target {
            self.forget_call_site(target, id);
        }
        self.function_mut(id.func).insns.remove(id.local);
    }

    pub fn merge_nodes(&mut self, keep: BlockId, remove: BlockId, direct_edge: EdgeId) {
        let func = keep.func;
        self.remove_cfg_edge(func, direct_edge);
        let outgoing: Vec<EdgeId> = {
            let host = self.read_host();
            host.block(remove)
                .edges
                .iter()
                .copied()
                .filter(|&e| host.edge(func, e).from == remove)
                .collect()
        };
        for eid in outgoing {
            self.function_mut(func).edges[eid].from = keep;
            self.block_mut(keep).edges.insert(eid);
            self.block_mut(remove).edges.remove(&eid);
        }
    }

    pub fn replace_instruction_mnemonic(&mut self, id: InstructionId, mnemonic: Mnemonic) {
        let func = id.func;
        let (old_args, old_target) = {
            let m = self.read_host().instruction(id).mnemonic();
            (m.args().into_iter().collect::<Vec<_>>(), m.call_target())
        };
        for arg in old_args {
            let now_empty = if let Some(users) = self.function_mut(func).users.get_mut(&arg) {
                users.retain(|&local| local != id.localize(func));
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
                .push(id.localize(func));
        }
        if let Some(target) = new_target {
            self.record_call_site(target, id);
        }
    }

    pub fn unroster_block(&mut self, block: BlockId) {
        debug_assert!(
            self.read_host()
                .block(block)
                .parent
                .is_none_or(|owner| owner == block.func),
            "cross-arena block ownership is unsupported"
        );
        self.function_mut(block.func)
            .roster
            .retain(|&b| b != block.local);
    }

    pub fn delete_block(&mut self, block: BlockId, _function_id: FunctionId) {
        let mut edges: Vec<EdgeId> = self
            .read_host()
            .block(block)
            .edges
            .iter()
            .copied()
            .collect();
        edges.sort_unstable();
        for edge in edges {
            self.remove_cfg_edge(block.func, edge);
        }
        let insns: Vec<InstructionId> = self
            .read_host()
            .block(block)
            .instructions
            .iter()
            .map(|&local| InstructionId::new(block.func, local))
            .collect();
        for insn in insns {
            self.remove_instruction(insn);
        }
        let params: Vec<BlockParamId> = self
            .read_host()
            .block(block)
            .params
            .iter()
            .map(|&local| BlockParamId::new(block.func, local))
            .collect();
        for param in params {
            self.remove_block_param(param);
        }
        self.unroster_block(block);
        let b = self.block_mut(block);
        b.parent = None;
        b.deleted = true;
    }

    pub fn absorb_block(
        &mut self,
        keep: BlockId,
        other: BlockId,
        edge_ab: EdgeId,
        _function_id: FunctionId,
    ) {
        assert_eq!(
            keep.func, other.func,
            "cannot absorb across function arenas"
        );
        let branch_args = {
            let host = self.read_host();
            host.block(keep)
                .instructions
                .last()
                .and_then(|&local| {
                    match host
                        .instruction(InstructionId::new(keep.func, local))
                        .mnemonic()
                    {
                        Mnemonic::Branch(branch)
                            if BlockId::new(keep.func, branch.target) == other =>
                        {
                            Some(branch.args.clone())
                        }
                        _ => None,
                    }
                })
                .unwrap_or_default()
        };
        let other_params: Vec<_> = self
            .read_host()
            .block(other)
            .params
            .iter()
            .map(|&local| BlockParamId::new(other.func, local))
            .collect();
        if !other_params.is_empty() {
            assert_eq!(
                other_params.len(),
                branch_args.len(),
                "cannot absorb block with {} params through branch with {} args",
                other_params.len(),
                branch_args.len()
            );
            for (param, arg) in other_params.into_iter().zip(branch_args) {
                self.replace_all_uses_with(ValueId::BlockParam(param), arg.qualify(keep.func));
            }
        }
        self.block_mut(keep).instructions.pop();
        let b_insns = std::mem::take(&mut self.block_mut(other).instructions);
        for &local in &b_insns {
            self.instruction_mut(InstructionId::new(other.func, local))
                .parent = Some(keep.local);
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

    // ---- names --------------------------------------------------------------

    pub fn register_local_name(
        &mut self,
        id: ValueId,
        name: std::borrow::Cow<'str, str>,
        old_name: Option<&str>,
    ) -> Result<()> {
        let existing = match id.name_scope_function() {
            Some(func) => self.function(func).names.get(&name),
            None => self.shr().get_named(&name),
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
            None => {
                unimplemented!("a checked-out host has read-only shared access (mints via &self)")
            }
        }
    }
}
