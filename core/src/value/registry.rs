use crate::{
    assumption::{KnownContradiction, Proposition, Truth, Violation},
    types::TypeId,
    value::{
        ValueId,
        block::{BasicBlock, BlockId, EdgeData, EdgeId},
        block_param::{BlockParam, BlockParamId},
        function::{Function, FunctionId},
        insn::{Instruction, InstructionId},
        literal::{Literal, LiteralId},
        varnode::{Varnode, VarnodeId},
    },
};
use jstd::registry::Registry;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::collections::BTreeSet;

/// Central storage arena for all IR values in a [`Context`](crate::context::Context).
///
/// Each field is a typed arena ([`Registry`]) keyed by the corresponding ID
/// type. Values are append-only: once pushed, their ID is stable for the
/// lifetime of the registry and their data is never moved.
///
/// # Invariants
///
/// - **`push_insn` is final.** Instructions are immutable after insertion.
///   The `users` map is populated at push time from the instruction's operands
///   and is not updated if operands are later altered via interior mutation.
///   Use [`Context::replace_all_uses_with`](crate::context::Context::replace_all_uses_with)
///   to rewrite operands while keeping `users` consistent.
///
/// - **`users` is managed internally.** Do not access or mutate `users`
///   directly. Use [`users_of`](Self::users_of) to read and
///   [`remove_instructions`](Self::remove_instructions) to remove dead
///   instructions from the map.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ValueRegistry<'str> {
    /// Literal (constant) storage.
    pub literals: Registry<LiteralId, Literal>,

    /// Instruction storage.
    pub instructions: Registry<InstructionId, Instruction<'str>>,

    /// Basic block storage.
    pub basic_blocks: Registry<BlockId, BasicBlock<'str>>,

    /// Block parameter storage.
    pub block_params: Registry<BlockParamId, BlockParam<'str>>,

    /// Varnode storage.
    pub varnodes: Registry<VarnodeId, Varnode<'str>>,

    /// Control-flow edge storage.
    pub edges: Registry<EdgeId, EdgeData>,

    /// Function storage.
    pub functions: Registry<FunctionId, Function<'str>>,

    /// Truth map of the assumption system: what each [`Proposition`] is
    /// currently assumed or known to be (see [`crate::assumption`]). Accessed
    /// through [`Context::assume_true`](crate::context::Context::assume_true)
    /// and friends.
    pub(crate) truths: HashMap<Proposition, Truth>,

    /// Proven facts that contradicted an assumption this round; non-empty means
    /// the checkpoint+replay driver must discard this working copy and replay.
    pub(crate) violations: Vec<Violation>,

    /// Proven facts that contradicted an existing *known* fact this round (e.g. a
    /// user override the analysis disproved). A hard error for the driver, not a
    /// replay signal. Transient per round, so not serialized.
    #[serde(default, skip)]
    pub(crate) known_contradictions: Vec<KnownContradiction>,

    /// Reverse use-def map: for each `ValueId`, the list of instructions that
    /// use it as an operand. Kept in sync by [`push_insn`](Self::push_insn),
    /// [`remove_instructions`](Self::remove_instructions), and
    /// [`Context::replace_all_uses_with`].
    pub(crate) users: HashMap<ValueId, Vec<InstructionId>>,

    /// Reverse call graph: for each callee [`FunctionId`], the direct-call sites
    /// (instructions) that target it. Kept in sync alongside `users` by
    /// [`push_insn`](Self::push_insn),
    /// [`remove_instructions`](Self::remove_instructions), and
    /// [`Context::replace_instruction_mnemonic`](crate::context::Context::replace_instruction_mnemonic).
    /// Indirect calls have no static target and are not recorded here.
    #[serde(default)]
    pub(crate) call_sites: HashMap<FunctionId, Vec<InstructionId>>,

    /// Synthetic forward call-graph edges that are not backed by a direct `Call`
    /// instruction: `caller FunctionId → set of callee entry addresses`. Used for
    /// relationships a pass recovers but the IR can't express as a direct call —
    /// e.g. `entry → main`, where `main` is passed to `__libc_start_main` as a
    /// pointer argument rather than called. Keyed by address so the edge resolves
    /// once a function exists at the callee, independent of when it materializes.
    /// Merged into [`FunctionRef::callees`](crate::value::FunctionRef::callees).
    #[serde(default)]
    pub(crate) synthetic_callees: HashMap<FunctionId, BTreeSet<u64>>,

    /// Intern cache for non-symbolic literals: `(masked_value, TypeId) → LiteralId`.
    literal_cache: HashMap<(u64, TypeId), LiteralId>,
}

impl<'str> ValueRegistry<'str> {
    /// Returns a canonical [`LiteralId`] for the given typed constant.
    ///
    /// The value is masked to `type_id`'s size before lookup. Symbolic literals
    /// (created via [`push_literal`](Self::push_literal)) are not included in
    /// the intern cache and will not alias with constants produced here.
    ///
    /// Call [`Context::get_const`] for the common `Int(size)` case; use this
    /// method directly when you need to preserve a non-`Int` type (e.g.
    /// [`StackAddress`](crate::types::StackAddress)) through folding.
    pub fn get_or_make_typed_literal(
        &mut self,
        mut value: u64,
        type_id: TypeId,
        size: usize,
    ) -> LiteralId {
        value &= if size >= 8 {
            u64::MAX
        } else {
            (1u64 << (size * 8)) - 1
        };

        if let Some(&id) = self.literal_cache.get(&(value, type_id)) {
            return id;
        }

        let id = self.literals.push(Literal {
            value,
            type_id,
            symbolic: None,
        });
        self.literal_cache.insert((value, type_id), id);
        id
    }

    /// Pushes a [`Literal`] with arbitrary fields (e.g. with a symbolic ref)
    /// without interning. Use [`get_or_make_literal`](Self::get_or_make_literal)
    /// for plain integer constants.
    pub fn push_literal(&mut self, literal: Literal) -> LiteralId {
        self.literals.push(literal)
    }

    /// Appends an instruction and records all its operands in the `users` map.
    ///
    /// # Immutability invariant
    ///
    /// Instructions are considered immutable after this call. If you alter the
    /// operands of an instruction after insertion the `users` map will be
    /// stale. Rewrite operands through
    /// [`Context::replace_all_uses_with`](crate::context::Context::replace_all_uses_with)
    /// instead.
    pub fn push_insn(&mut self, insn: Instruction<'str>) -> InstructionId {
        let args = insn.mnemonic().args();
        let call_target = insn.mnemonic().call_target();
        let id = self.instructions.push(insn);
        for arg in args {
            self.users.entry(arg).or_default().push(id);
        }
        if let Some(target) = call_target {
            self.call_sites.entry(target).or_default().push(id);
        }
        id
    }

    /// Returns all instructions that use `value` as an operand.
    pub fn users_of(&self, value: ValueId) -> &[InstructionId] {
        self.users.get(&value).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Returns the direct-call sites (instructions) targeting `callee`.
    pub fn call_sites_of(&self, callee: FunctionId) -> &[InstructionId] {
        self.call_sites
            .get(&callee)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Records a synthetic forward call-graph edge `caller → callee_addr` (see
    /// [`synthetic_callees`](Self::synthetic_callees)). Returns `true` if the
    /// edge was newly added, so callers can drive a fixpoint without spinning.
    pub fn add_synthetic_callee(&mut self, caller: FunctionId, callee_addr: u64) -> bool {
        self.synthetic_callees
            .entry(caller)
            .or_default()
            .insert(callee_addr)
    }

    /// Returns the synthetic callee entry addresses recorded for `caller`.
    pub fn synthetic_callees_of(&self, caller: FunctionId) -> impl Iterator<Item = u64> + '_ {
        self.synthetic_callees
            .get(&caller)
            .into_iter()
            .flatten()
            .copied()
    }

    /// Removes a set of dead instructions from the use-def map.
    ///
    /// For every instruction in `dead`, each of its operands will have all
    /// members of `dead` pruned from their user lists. Call this after
    /// removing dead instructions from their basic blocks.
    pub fn remove_instructions(&mut self, dead: &HashSet<InstructionId>) {
        for &id in dead {
            let args = self.instructions[id].mnemonic().args();
            for arg in args {
                if let Some(users) = self.users.get_mut(&arg) {
                    users.retain(|u| !dead.contains(u));
                }
            }
            if let Some(target) = self.instructions[id].mnemonic().call_target()
                && let Some(sites) = self.call_sites.get_mut(&target)
            {
                sites.retain(|s| !dead.contains(s));
            }
        }
    }

    pub fn push_block(&mut self, block: BasicBlock<'str>) -> BlockId {
        self.basic_blocks.push(block)
    }

    pub fn push_block_param(&mut self, param: BlockParam<'str>) -> BlockParamId {
        self.block_params.push(param)
    }

    pub fn push_varnode(&mut self, varnode: Varnode<'str>) -> VarnodeId {
        self.varnodes.push(varnode)
    }

    pub fn push_function(&mut self, f: Function<'str>) -> FunctionId {
        self.functions.push(f)
    }
}
