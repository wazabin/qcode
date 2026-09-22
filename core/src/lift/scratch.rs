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
//! # What an instruction leaves in shared state, and the budget
//!
//! The host function is emptied whole, but the module's shared arenas are
//! not: every distinct immediate an instruction mentions is interned as a
//! literal, and the interner only grows. Recycling the per-instruction IR
//! cannot make total memory flat while that grows with the corpus, so the
//! store watches it and, past a [budget](ScratchStore::with_literal_budget),
//! rebuilds its context from the architecture base it was made from. Memory
//! is bounded by the budget, not by the number of distinct constants seen.
//!
//! A rebuild starts a new **literal epoch**: the interner restarts at the
//! base, so a `LiteralId` interned by an earlier instruction may now name a
//! different constant, or nothing. Literal ids are therefore epoch-scoped
//! exactly like block and instruction ids — valid while the instruction that
//! produced them is the current one, and never to be resolved after the next
//! reset. The store gives no way to resolve one at all:
//! [`context`](ScratchStore::context) is the raw handle a facade builds on, and the
//! facade resolves an operand only through the view of the live instruction
//! that holds it. Varnodes are not affected: an instruction interns none (a
//! lifter maps its registers when it is built), so a rebuild reproduces the
//! base varnodes id for id.
//!
//! # Failure
//!
//! A construction that fails rolls itself back; one whose rollback cannot
//! account for the host [poisons](crate::context::Context::is_poisoned) the
//! context. For scratch storage that is recoverable: the host holds nothing
//! but the failed instruction, so the next reset, which empties it, clears
//! the poison along with it.

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
        self.set_literal_budget(budget);
        self
    }

    /// See [`with_literal_budget`](Self::with_literal_budget). Takes effect
    /// at the next reset.
    pub fn set_literal_budget(&mut self, budget: usize) {
        self.literal_budget = budget;
    }

    /// The host every instruction is lifted into.
    pub fn function(&self) -> FunctionId {
        self.function
    }

    /// The context, holding at most the instruction lifted since the last
    /// reset.
    ///
    /// This is a raw handle onto scratch storage: any `BlockId`, `LiteralId`
    /// or other id read from it names storage that [`reset`](Self::reset)
    /// reissues, so nothing read here may be used after the next reset. The
    /// safe boundary is
    /// [`ScratchSession`](../../../wazabin_qcode_sleigh/session/struct.ScratchSession.html),
    /// whose views carry a borrow that forbids exactly that; prefer it, and
    /// treat this accessor as a building block for such a facade.
    pub fn context(&self) -> &Context<'static> {
        &self.ctx
    }

    /// How many resets have happened. Every handle obtained before a reset —
    /// block, instruction, temporary or literal — belongs to an earlier epoch
    /// and must not be used after it.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// How many times the context was rebuilt from its base, each starting a
    /// new literal epoch.
    pub fn rebuilds(&self) -> u64 {
        self.rebuilds
    }

    /// The host's arena footprint.
    pub fn arena_stats(&self) -> BodyArenaStats {
        self.ctx.bodies[self.function].arena_stats()
    }

    /// Literals interned since the context was built or last rebuilt. Never
    /// exceeds the budget for longer than one instruction.
    pub fn interned_literals(&self) -> usize {
        self.ctx.shared.values.literals.len() - self.base_literals
    }

    /// Empties the host and starts a new epoch, reusing its arena capacity —
    /// or, past the literal budget, rebuilding the context from its base. A
    /// poison left by a failed instruction is cleared with the instruction.
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
            self.ctx.clear_poison();
        }
        // Emptied by hand rather than rebuilt, to keep its capacity: the host
        // has no blocks now and the module has no addressed function, so an
        // empty index is the complete one.
        self.addresses.clear();
        self.addresses.mark_current(&self.ctx);
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
        lift::{ExitArm, ExitKind},
        value::ValueId,
    };

    /// Lifts one two-instruction "instruction" at `address` storing
    /// `immediate`, which is what varies between real instructions.
    fn lift_one(store: &mut ScratchStore, address: u64, immediate: u64) {
        let mut target = store.target().unwrap();
        let mut construction = target.begin(address, 4).unwrap();
        let next = construction.block_at(address + 4).unwrap();
        let mut emitter = construction.emitter();
        let temp = emitter.make_temp(8);
        let value = emitter.shr().get_const(immediate, 8);
        emitter.push_copy(value, ValueId::Temp(temp));
        let label = emitter.get_or_make_local_label(format!("local_{address:x}").into());
        emitter.push_branch(label);
        emitter.switch_to_block(label);
        let site = emitter.push_branch(next).id;
        emitter.exit(site, ExitArm::Unconditional, ExitKind::Fallthrough);
        emitter.continue_in(label);
        construction.commit().unwrap();
    }

    #[test]
    fn a_reset_reissues_ids_over_the_same_capacity() {
        let mut store = ScratchStore::new(Context::new());
        lift_one(&mut store, 0x1000, 1);
        let first = store.arena_stats();
        assert_eq!(first.blocks.issued, 3, "entry, local label, fall-through");
        assert_eq!(first.instructions.issued, 3);
        let first_uses = store.context().bodies[store.function()].uses.len();
        assert_eq!(first_uses, 2, "the literal and the temporary");

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
        assert_eq!(body.uses.len(), first_uses);
        let last_address = 0x1000 + 4 * 9_999;
        assert!(body.names.get(&format!("local_{last_address:x}")).is_some());
        assert!(body.names.get("local_1000").is_none());
        assert_eq!(store.context().block_ids().len(), 3);
    }

    #[test]
    fn varying_immediates_are_bounded_by_the_literal_budget() {
        let mut store = ScratchStore::new(Context::new()).with_literal_budget(100);
        let base_varnodes = store.context().shared.values.varnodes.len();
        let base_bytes = store.context().shared.values.bytes.len();
        for i in 0..10_000u64 {
            store.reset();
            lift_one(&mut store, 0x1000, 0x1_0000 + i);
            assert!(store.interned_literals() <= 101, "epoch {i}");
        }
        assert!(store.rebuilds() >= 90, "{}", store.rebuilds());
        assert_eq!(store.context().functions().count(), 1);
        assert_eq!(store.arena_stats().blocks.issued, 3);
        assert_eq!(store.context().block_ids().len(), 3);
        // Nothing but literals grows in shared state.
        assert_eq!(store.context().shared.values.varnodes.len(), base_varnodes);
        assert_eq!(store.context().shared.values.bytes.len(), base_bytes);
    }

    #[test]
    fn repeated_instructions_intern_nothing_new() {
        let mut store = ScratchStore::new(Context::new());
        for _ in 0..1_000 {
            store.reset();
            lift_one(&mut store, 0x1000, 42);
        }
        assert_eq!(store.interned_literals(), 1);
    }

    #[test]
    fn a_rebuild_starts_a_new_literal_epoch() {
        // Budget 0: every reset after an interning instruction rebuilds.
        let mut store = ScratchStore::new(Context::new()).with_literal_budget(0);
        lift_one(&mut store, 0x1000, 0xdead_beef);
        let stale = store
            .context()
            .shared
            .values
            .literals
            .iter()
            .find(|l| l.inner.value == 0xdead_beef)
            .expect("the immediate was interned")
            .id;
        store.reset();
        assert_eq!(store.rebuilds(), 1);
        // The id from the previous epoch names nothing now...
        assert_eq!(store.interned_literals(), 0);
        assert!(usize::from(stale) >= store.context().shared.values.literals.len());
        // ...and after the next instruction it names *its* constant, which
        // is why no id may be resolved across a reset.
        lift_one(&mut store, 0x1000, 0xcafe);
        assert_eq!(store.context().get_literal_value(stale), 0xcafe);
    }

    #[test]
    fn a_poisoned_instruction_is_discarded_with_its_epoch() {
        let mut store = ScratchStore::new(Context::new());
        lift_one(&mut store, 0x1000, 1);
        {
            // A second construction in the same epoch that writes into the
            // previous instruction's fall-through placeholder, which its
            // rollback cannot undo.
            let previous = *store
                .context()
                .block_ids()
                .iter()
                .find(|&&b| !store.context().block(b).has_insns())
                .expect("the fall-through placeholder is empty");
            let mut target = store.target().unwrap();
            let mut construction = target.begin(0x2000, 4).unwrap();
            let zero = construction.context().shared.get_const(0, 8);
            let mut emitter = construction.emitter();
            emitter.switch_to_block(previous);
            emitter.push_branchind(zero);
            construction.abort();
            assert!(target.is_poisoned());
        }
        assert!(store.context().is_poisoned());
        assert_eq!(store.target().err(), Some(TargetError::Poisoned));

        // The reset empties the host, and with it everything the poison
        // stood for.
        store.reset();
        assert!(!store.context().is_poisoned());
        lift_one(&mut store, 0x1000, 2);
        assert_eq!(store.arena_stats().blocks.issued, 3);
    }
}
