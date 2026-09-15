//! Storage for lifting instructions one at a time, each discarded before the
//! next.
//!
//! A [`ScratchStore`] owns a context and one host function, and lends them
//! out as a [`LiftTarget`] for exactly one instruction at a time. Between
//! instructions [`reset`](ScratchStore::reset) empties the host: its arenas
//! start a new epoch, reissuing ids from zero over the capacity the previous
//! instruction left, so ten thousand instructions cost one instruction's
//! allocations.
//!
//! That reuse is what makes the store the *only* owner of its handles. A
//! `BlockId` from one epoch names a different block in the next, so nothing
//! outside the store may keep one across a reset. The store therefore never
//! hands its context out mutably, and a facade over it — the scratch session
//! of the SLEIGH lifter — ties every view it gives out to the borrow of a
//! single lifted instruction.
//!
//! # What an instruction leaves in shared state
//!
//! The host function is emptied whole, but the module's shared arenas are
//! not: every distinct immediate an instruction mentions is interned as a
//! literal, and the interner only grows. The store watches that growth and,
//! past a [budget](ScratchStore::with_literal_budget), rebuilds its context
//! from the architecture base it was made from. Memory is bounded by the
//! budget, not by the corpus.

use crate::{
    address_index::AddressIndex,
    context::Context,
    lift::{LiftTarget, TargetError},
    value::{BodyArenaStats, FunctionId},
};

/// The default number of literals interned past the base before the context
/// is rebuilt.
pub const DEFAULT_LITERAL_BUDGET: usize = 1 << 16;

/// See the [module documentation](self).
pub struct ScratchStore {
    ctx: Context<'static>,
    /// The context as it was before any instruction: what a rebuild restores.
    base: Context<'static>,
    base_literals: usize,
    literal_budget: usize,
    addresses: AddressIndex,
    function: FunctionId,
    epoch: u64,
    rebuilds: u64,
}

impl ScratchStore {
    /// Makes a store over `base`, which carries the architecture (spaces,
    /// registers, user operations) and nothing an instruction depends on.
    pub fn new(base: Context<'static>) -> Self {
        let mut ctx = base.clone();
        let function = ctx.anon_function();
        Self {
            base_literals: base.shared.values.literals.len(),
            base,
            literal_budget: DEFAULT_LITERAL_BUDGET,
            addresses: AddressIndex::analyze(&ctx),
            ctx,
            function,
            epoch: 0,
            rebuilds: 0,
        }
    }

    /// Rebuilds the context once instructions have interned `budget` more
    /// literals than the base holds.
    pub fn with_literal_budget(mut self, budget: usize) -> Self {
        self.literal_budget = budget;
        self
    }

    /// The host every instruction is lifted into.
    pub fn function(&self) -> FunctionId {
        self.function
    }

    /// The context, holding at most the instruction lifted since the last
    /// reset.
    pub fn context(&self) -> &Context<'static> {
        &self.ctx
    }

    /// How many resets have happened. Every handle obtained before a reset
    /// belongs to an earlier epoch.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// How many times the context was rebuilt from its base.
    pub fn rebuilds(&self) -> u64 {
        self.rebuilds
    }

    /// The host's arena footprint.
    pub fn arena_stats(&self) -> BodyArenaStats {
        self.ctx.bodies[self.function].arena_stats()
    }

    /// Literals interned since the context was built or last rebuilt.
    pub fn interned_literals(&self) -> usize {
        self.ctx.shared.values.literals.len() - self.base_literals
    }

    /// Empties the host and starts a new epoch.
    pub fn reset(&mut self) {
        self.epoch += 1;
        if self.interned_literals() > self.literal_budget {
            self.ctx = self.base.clone();
            let function = self.ctx.anon_function();
            debug_assert_eq!(function, self.function, "the host keeps its id");
            self.function = function;
            self.rebuilds += 1;
        } else {
            self.ctx.bodies[self.function].start_epoch();
        }
        self.addresses.clear();
    }

    /// Binds the store for one instruction.
    pub fn target(&mut self) -> Result<LiftTarget<'_, 'static>, TargetError> {
        LiftTarget::bind_indexed(&mut self.ctx, &mut self.addresses, self.function)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        lift::{ExitArm, ExitKind, Recorder},
        value::{QCodeView, ValueId},
    };

    /// Lifts one two-instruction "instruction" at `address` storing
    /// `immediate`, which is what varies between real instructions.
    fn lift_one(store: &mut ScratchStore, address: u64, immediate: u64) {
        let mut target = store.target().unwrap();
        let mut construction = target.begin(address, 4).unwrap();
        let entry = construction.entry();
        let next = construction.block_at(address + 4).unwrap();
        let mut recorder = Recorder::new(address, 4, entry);
        let mut builder = construction.builder(entry);
        let temp = builder.make_temp(8);
        let value = builder.shr().get_const(immediate, 8);
        builder.push_copy(value, ValueId::Temp(temp));
        let label = builder.get_or_make_local_label(format!("local_{address:x}").into());
        builder.push_branch(label);
        builder.switch_to_block(label);
        let site = builder.push_branch(next).id;
        recorder.exit(site, ExitArm::Unconditional, ExitKind::Fallthrough);
        recorder.continue_in(label);
        construction.commit(recorder.finish()).unwrap();
    }

    #[test]
    fn a_reset_reissues_ids_over_the_same_capacity() {
        let mut store = ScratchStore::new(Context::new());
        lift_one(&mut store, 0x1000, 1);
        let first = store.arena_stats();
        assert_eq!(first.blocks.issued, 3, "entry, local label, fall-through");
        assert_eq!(first.instructions.issued, 3);
        let first_users = store.context().bodies[store.function()].users.len();
        assert_eq!(first_users, 2, "the literal and the temporary");

        for i in 0..10_000u64 {
            store.reset();
            lift_one(&mut store, 0x1000 + 4 * i, i);
        }
        let last = store.arena_stats();
        assert_eq!(store.epoch(), 10_000);
        assert_eq!(last.blocks.issued, first.blocks.issued);
        assert_eq!(last.instructions.issued, first.instructions.issued);
        assert_eq!(last.instructions.capacity, first.instructions.capacity);
        let body = &store.context().bodies[store.function()];
        assert_eq!(body.temps.len(), 1);
        assert_eq!(body.users.len(), first_users);
        let last_address = 0x1000 + 4 * 9_999;
        assert!(body.names.contains(&format!("local_{last_address:x}")));
        assert!(!body.names.contains("local_1000"));
        assert_eq!(store.context().block_ids().len(), 3);
    }

    #[test]
    fn varying_immediates_are_bounded_by_the_literal_budget() {
        let mut store = ScratchStore::new(Context::new()).with_literal_budget(100);
        for i in 0..10_000u64 {
            store.reset();
            lift_one(&mut store, 0x1000, 0x1_0000 + i);
            assert!(store.interned_literals() <= 101, "epoch {i}");
        }
        assert!(store.rebuilds() >= 90, "{}", store.rebuilds());
        assert_eq!(store.context().functions().count(), 1);
        assert_eq!(store.context().block_ids().len(), 3);
    }

    #[test]
    fn repeated_instructions_intern_nothing_new() {
        let mut store = ScratchStore::new(Context::new());
        for _ in 0..1_000 {
            store.reset();
            lift_one(&mut store, 0x1000, 42);
        }
        assert_eq!(store.rebuilds(), 0);
        assert_eq!(store.interned_literals(), 1);
    }
}
