use crate::{
    assumption::{Assumption, AssumptionId},
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
use std::collections::{HashMap, HashSet};

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

    /// Heuristic assumption ledger (see [`crate::assumption`]).
    pub assumptions: Registry<AssumptionId, Assumption>,

    /// Reverse index: callee `FunctionId` -> assumptions predicting it. Used by
    /// the verification pass to find every assumption to check when a function
    /// is analyzed. Kept in sync by
    /// [`Context::add_assumption`](crate::context::Context::add_assumption).
    pub(crate) assumptions_by_callee: HashMap<FunctionId, Vec<AssumptionId>>,

    /// Reverse use-def map: for each `ValueId`, the list of instructions that
    /// use it as an operand. Kept in sync by [`push_insn`](Self::push_insn),
    /// [`remove_instructions`](Self::remove_instructions), and
    /// [`Context::replace_all_uses_with`].
    pub(crate) users: HashMap<ValueId, Vec<InstructionId>>,

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
        let id = self.instructions.push(insn);
        for arg in args {
            self.users.entry(arg).or_default().push(id);
        }
        id
    }

    /// Returns all instructions that use `value` as an operand.
    pub fn users_of(&self, value: ValueId) -> &[InstructionId] {
        self.users.get(&value).map(Vec::as_slice).unwrap_or(&[])
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
