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
//! literal. The interner is **never rebuilt or reset** — it is append-only and
//! deduped — so a [`LiteralId`](crate::value::LiteralId) means the same value
//! for the whole life of the store, and resolving one obtained from an earlier
//! instruction is always sound, never an alias for a different value. The cost
//! is that the interner grows with the number of *distinct* constants seen
//! (see [`interned_literals`](ScratchStore::interned_literals)); the recycled
//! per-instruction IR is what keeps total memory flat regardless. A stable
//! interner is chosen over a budget-triggered rebuild precisely because a
//! rebuild would reissue literal ids and make a retained id alias.

use crate::{
    address_index::AddressIndex,
    context::Context,
    lift::{LiftTarget, TargetError},
    value::{BodyArenaStats, FunctionId},
};

/// See the [module documentation](self).
pub struct ScratchStore {
    ctx: Context<'static>,
    base_literals: usize,
    addresses: AddressIndex,
    function: FunctionId,
    epoch: u64,
}

impl ScratchStore {
    /// Makes a store over `base`, which carries the architecture (spaces,
    /// registers, user operations) and nothing an instruction depends on.
    pub fn new(base: Context<'static>) -> Self {
        let base_literals = base.shared.values.literals.len();
        let mut ctx = base;
        let function = ctx.anon_function();
        Self {
            base_literals,
            addresses: AddressIndex::analyze(&ctx),
            ctx,
            function,
            epoch: 0,
        }
    }

    /// The host every instruction is lifted into.
    pub fn function(&self) -> FunctionId {
        self.function
    }

    /// The context, holding at most the instruction lifted since the last
    /// reset.
    ///
    /// This is a raw handle onto scratch storage: any `BlockId` or other id
    /// read from it names storage that [`reset`](Self::reset) reissues, so
    /// nothing read here may be used after the next reset. The safe boundary is
    /// [`ScratchSession`](../../../wazabin_qcode_sleigh/session/struct.ScratchSession.html),
    /// whose views carry a borrow that forbids exactly that; prefer it, and
    /// treat this accessor as a building block for such a facade.
    pub fn context(&self) -> &Context<'static> {
        &self.ctx
    }

    /// How many resets have happened. Every block/instruction/temp handle
    /// obtained before a reset belongs to an earlier epoch and must not be
    /// used after it. (Literal ids are exempt: the interner is stable.)
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The host's arena footprint.
    pub fn arena_stats(&self) -> BodyArenaStats {
        self.ctx.bodies[self.function].arena_stats()
    }

    /// Distinct constants interned since the store was made. This is the only
    /// per-corpus growth; it rises with the number of distinct immediates and
    /// never shrinks, because the interner is stable (see the module docs).
    pub fn interned_literals(&self) -> usize {
        self.ctx.shared.values.literals.len() - self.base_literals
    }

    /// Empties the host function and starts a new epoch, reusing its arena
    /// capacity. The shared literal interner is deliberately left intact.
    pub fn reset(&mut self) {
        self.epoch += 1;
        self.ctx.bodies[self.function].start_epoch();
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
        value::ValueId,
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
    fn varying_immediates_keep_the_ir_flat_and_grow_only_the_interner() {
        let mut store = ScratchStore::new(Context::new());
        for i in 0..10_000u64 {
            store.reset();
            lift_one(&mut store, 0x1000, 0x1_0000 + i);
        }
        // The per-instruction IR never grows; only the deduped constant
        // interner does, by one per distinct immediate.
        assert_eq!(store.arena_stats().blocks.issued, 3);
        assert_eq!(store.context().block_ids().len(), 3);
        assert_eq!(store.interned_literals(), 10_000);
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
    fn a_literal_id_stays_valid_across_later_lifts() {
        let mut store = ScratchStore::new(Context::new());
        lift_one(&mut store, 0x1000, 0xdead_beef);
        // The literal id of the first instruction's immediate.
        let id = store
            .context()
            .shared
            .values
            .literals
            .iter()
            .find(|l| l.inner.value == 0xdead_beef)
            .expect("the immediate was interned")
            .id;

        // Many more instructions with distinct immediates — no rebuild, so the
        // interner never reissues that id.
        for i in 0..5_000u64 {
            store.reset();
            lift_one(&mut store, 0x2000, 0x10_0000 + i);
        }
        assert_eq!(
            store.context().get_literal_value(id),
            0xdead_beef,
            "a retained literal id still resolves to its original value"
        );
    }
}
