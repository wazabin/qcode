//! Generic infrastructure for composing GVN sub-passes.
//!
//! A [`SubPass`] owns one concern (constant folding, CSE, memory forwarding, …)
//! and its own [`SubPass::State`]. Sub-passes are composed statically as a tuple
//! (see [`SubPasses`]); per instruction, each member is tried in tuple order
//! until one returns [`Claim::Done`]. The drivers at the bottom of this module
//! own the block iteration and the dominator-tree state threading, so adding a
//! new sub-pass never touches the walk or the other sub-passes.

use std::any::Any;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use jstd::graph::analysis::{DominatorTree, compute_dominators};

use crate::AliasResult;

use super::affine::Numbering;

use qcode::context::Context;
use qcode::value::{
    ValueId,
    block::BlockId,
    function::FunctionId,
    insn::{InstructionId, Mnemonic},
    util::base_ref::HostRef,
};

/// What a sub-pass did with an instruction.
pub(super) enum Claim {
    /// Not mine / nothing to do — try the next sub-pass.
    Pass,
    /// Handled (rewrote, recorded, or consumed) — stop the chain for this insn.
    Done,
}

/// Read-only view of the instruction currently being visited.
pub(super) struct InsnCtx<'a> {
    pub block_id: BlockId,
    pub insn_id: InstructionId,
    /// The instruction's value id (what its uses refer to).
    pub id: ValueId,
    pub size: usize,
    pub mnemonic: &'a Mnemonic,
    pub aliases: Option<&'a AliasResult>,
    /// Read-only, function-wide affine view of every value, precomputed before
    /// the walk. Used for pointer base identity in memory forwarding.
    pub numbering: &'a Numbering,
}

/// Block-local rewrite bookkeeping shared by all sub-passes.
///
/// Instructions marked redundant stay in place (so later sub-pass state and
/// iteration order are unaffected) and are removed in one `retain_insns` sweep
/// when the block finishes.
pub(super) struct Editor {
    redundant: HashSet<InstructionId>,
}

impl Editor {
    fn new() -> Self {
        Editor {
            redundant: HashSet::default(),
        }
    }

    /// Forward all uses of `insn` to `with` and mark `insn` redundant.
    pub(super) fn replace<'str>(
        &mut self,
        host: &mut Context<'str>,
        insn: InstructionId,
        with: ValueId,
    ) {
        host.replace_all_uses_with(ValueId::Instruction(insn), with);
        self.redundant.insert(insn);
    }

    /// Materialize `mnemonic` as a new instruction inserted before `at`, then
    /// forward all uses of `at` to it and mark `at` redundant.
    pub(super) fn replace_with_new_insn<'str>(
        &mut self,
        host: &mut Context<'str>,
        block_id: BlockId,
        at: InstructionId,
        mnemonic: Mnemonic,
        size: usize,
    ) -> InstructionId {
        let type_id = host.shr().types.get_or_make_int(size);
        self.replace_with_new_insn_typed(host, block_id, at, mnemonic, type_id)
    }

    /// Like [`replace_with_new_insn`](Self::replace_with_new_insn) but with an
    /// explicit result [`TypeId`] — used when the new instruction is a comparison
    /// (which must be `bool`-typed, not a plain `iN`).
    pub(super) fn replace_with_new_insn_typed<'str>(
        &mut self,
        host: &mut Context<'str>,
        block_id: BlockId,
        at: InstructionId,
        mnemonic: Mnemonic,
        type_id: qcode::types::TypeId,
    ) -> InstructionId {
        let new_id = host.push_mnemonic_with_type(block_id.func, mnemonic, type_id);
        host.insert_insn_before(block_id, at, new_id);
        host.replace_all_uses_with(ValueId::Instruction(at), ValueId::Instruction(new_id));
        self.redundant.insert(at);
        new_id
    }

    /// Drop the redundant instructions from their block; returns whether
    /// anything was rewritten. Every redundant instruction has already had its
    /// uses forwarded, so removing it prunes only its own operand use-lists.
    fn finish<'str>(self, host: &mut Context<'str>) -> bool {
        let changed = !self.redundant.is_empty();
        for insn in self.redundant {
            host.remove_instruction(insn);
        }
        changed
    }
}

/// One composable concern of the GVN pipeline. Sub-passes are held as a
/// `&[Box<dyn SubPass>]` array and tried in order until one returns
/// [`Claim::Done`]. Per-sub-pass state is type-erased as `Box<dyn Any>`: each
/// pass mints it in [`SubPass::init_state`], clones it down the dominator tree
/// in [`SubPass::clone_state`], and downcasts it in the visit hooks.
///
/// The module-scope twin of [`SubPassC`]: one composable GVN concern driven over
/// the whole [`Context`] (`&mut Context`). It backs the whole-`Context` public
/// entry points ([`gvn_function`](super::gvn_function) &c) — which must tolerate
/// reattributed/orphan-region bodies the checked-out `SubPassC` path forbids —
/// and the module `concretize` pass (its three body-reading sub-passes read pure
/// callee bodies through `&*ctx`).
///
/// [`Context`]: qcode::context::Context
pub(super) trait ModuleSubPass<'str> {
    /// Fresh per-walk state, cloned down the dominator tree (a value available
    /// in a dominator is available in every block it dominates). Stateless
    /// sub-passes return `Box::new(())`.
    fn init_state(&self) -> Box<dyn Any>;

    /// Clone this pass's own state for a dominated child. The pass clones it
    /// (rather than a blanket `dyn` clone) because returning `Box<dyn Any>` from
    /// a `&self`-only method would tie the clone to the borrow's lifetime.
    fn clone_state(&self, state: &dyn Any) -> Box<dyn Any>;

    /// Called once per block before its instructions. `is_shared` marks blocks
    /// reachable from more than one walk entry, whose inherited dominance
    /// claims are invalid.
    #[allow(clippy::too_many_arguments)]
    fn on_block_entry(
        &self,
        _host: &mut Context<'str>,
        _state: &mut dyn Any,
        _block_id: BlockId,
        _tree: &DominatorTree<BlockId>,
        _aliases: Option<&AliasResult>,
        _numbering: &Numbering,
        _is_shared: bool,
    ) {
    }

    /// Visit one instruction. Rewrites go through `ed`.
    fn on_insn(
        &self,
        host: &mut Context<'str>,
        state: &mut dyn Any,
        ic: &InsnCtx,
        ed: &mut Editor,
    ) -> Claim;

    /// Called after the block's instructions, before its dominated children.
    fn after_block(
        &self,
        _host: &mut Context<'str>,
        _state: &mut dyn Any,
        _block_id: BlockId,
        _aliases: Option<&AliasResult>,
        _numbering: &Numbering,
    ) {
    }
}

/// One fresh erased state slot per sub-pass, in array order.
fn init_states<'str>(passes: &[Box<dyn ModuleSubPass<'str>>]) -> Vec<Box<dyn Any>> {
    passes.iter().map(|p| p.init_state()).collect()
}

/// Clone a state vector for a dominated child, each pass cloning its own slot.
fn clone_states<'str>(
    passes: &[Box<dyn ModuleSubPass<'str>>],
    states: &[Box<dyn Any>],
) -> Vec<Box<dyn Any>> {
    passes
        .iter()
        .zip(states)
        .map(|(p, s)| p.clone_state(s.as_ref()))
        .collect()
}

// ---------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------

/// Run the sub-pass chain over every instruction of `block_id`, then drop the
/// instructions it made redundant. Returns whether anything was rewritten.
fn run_block<'str>(
    host: &mut Context<'str>,
    block_id: BlockId,
    passes: &[Box<dyn ModuleSubPass<'str>>],
    states: &mut [Box<dyn Any>],
    aliases: Option<&AliasResult>,
    numbering: &Numbering,
) -> bool {
    let mut ed = Editor::new();
    let insns: Vec<InstructionId> = host.read_host().block(block_id).instructions.clone();

    for insn_id in insns {
        let (id, size, mnemonic) = {
            let insn = host.insn_ref(insn_id);
            (insn.id(), insn.size(), insn.mnemonic().clone())
        };
        let ic = InsnCtx {
            block_id,
            insn_id,
            id,
            size,
            mnemonic: &mnemonic,
            aliases,
            numbering,
        };
        // Try each sub-pass in array order; the first `Done` claims the insn.
        for (pass, state) in passes.iter().zip(states.iter_mut()) {
            if let Claim::Done = pass.on_insn(host, state.as_mut(), &ic, &mut ed) {
                break;
            }
        }
    }

    ed.finish(host)
}

/// Run the sub-passes over a single block with fresh state and no block-boundary
/// hooks (no dominator tree exists for a lone block).
pub(super) fn run_single_block<'str>(
    host: &mut Context<'str>,
    block_id: BlockId,
    passes: &[Box<dyn ModuleSubPass<'str>>],
    aliases: Option<&AliasResult>,
) -> bool {
    // No function context for a lone block: memory forwarding falls back to
    // degenerate per-pointer bases (exact-match only).
    let numbering = Numbering::default();
    let mut states = init_states(passes);
    run_block(host, block_id, passes, &mut states, aliases, &numbering)
}

/// Iterate the sub-passes over every block of `func_id` in flat order with
/// fresh per-block state, repeating until a full sweep changes nothing. No
/// block-boundary hooks run. Returns whether anything changed.
pub(super) fn run_flat_fixpoint<'str>(
    host: &mut Context<'str>,
    func_id: FunctionId,
    passes: &[Box<dyn ModuleSubPass<'str>>],
) -> bool {
    let block_ids: Vec<BlockId> = host
        .function_ref(func_id)
        .iter()
        .map(|block| block.id)
        .collect();

    let numbering = Numbering::default();
    let mut changed_any = false;
    loop {
        let mut changed = false;
        for &block_id in &block_ids {
            let mut states = init_states(passes);
            changed |= run_block(host, block_id, passes, &mut states, None, &numbering);
        }
        changed_any |= changed;
        if !changed {
            break;
        }
    }
    changed_any
}

/// The per-entry invariants of one dominator-tree walk.
struct Walk<'a, 'str> {
    passes: &'a [Box<dyn ModuleSubPass<'str>>],
    func_id: FunctionId,
    tree: &'a DominatorTree<BlockId>,
    aliases: Option<&'a AliasResult>,
    numbering: &'a Numbering,
    shared: &'a HashSet<BlockId>,
    /// The function being walked. A cross-function CFG successor (a jump-table
    /// case or tail-call/thunk `Branch` into another function) is **not**
    /// descended into: a function pass mutates only its own function, and a
    /// checked-out host cannot even reach a foreign block's arena for writing.
    /// The foreign block is optimized when its own function is walked.
    owner: FunctionId,
    changed: bool,
}

impl<'str> Walk<'_, 'str> {
    fn rec(&mut self, host: &mut Context<'str>, block_id: BlockId, inherited: &[Box<dyn Any>]) {
        // Stay inside the function being processed. A tail-call edge is a real CFG
        // edge, so the dominator tree can reach blocks owned by the callee — but
        // the per-function alias oracle does not describe them, and following a
        // foreign store here could invalidate (or fail to invalidate) a forwarded
        // load using the wrong facts. Each block is processed by its own owner's
        // walk, with its own alias oracle and dominator context.
        //
        // Do *not* prune the subtree here: an owned block can be dominated only
        // through a foreign block (the tail-callee sits between the root and a
        // later owned region), and the orphan sweep only rescues predecessor-less
        // blocks — so returning outright would leave that owned block with no GVN
        // at all. Skip processing the foreign block, but keep recursing (threading
        // the inherited state through unchanged) so owned descendants still run.
        if host.values.basic_blocks[block_id].parent != Some(self.func_id) {
            for &child in self.tree.children_of(block_id) {
                self.rec(host, child, inherited);
            }
            return;
        }
        let mut states = clone_states(self.passes, inherited);
        let is_shared = self.shared.contains(&block_id);
        for (pass, state) in self.passes.iter().zip(states.iter_mut()) {
            pass.on_block_entry(
                host,
                state.as_mut(),
                block_id,
                self.tree,
                self.aliases,
                self.numbering,
                is_shared,
            );
        }
        self.changed |= run_block(
            host,
            block_id,
            self.passes,
            &mut states,
            self.aliases,
            self.numbering,
        );
        for (pass, state) in self.passes.iter().zip(states.iter_mut()) {
            pass.after_block(host, state.as_mut(), block_id, self.aliases, self.numbering);
        }
        for &child in self.tree.children_of(block_id) {
            // Stay within the walked function: a dominator-tree child *owned* by
            // another function is a cross-function CFG target and is left to that
            // function's own walk. Keyed on ownership (`parent`), not storage
            // (`id.func`), so a reattributed own block (owner == walked function,
            // stored elsewhere) is still descended into.
            if host.read_host().block(child).parent == Some(self.owner) {
                self.rec(host, child, &states);
            }
        }
    }
}

/// All blocks reachable from `entry` via CFG successor edges (including `entry`),
/// **confined to `owner`'s own blocks**: a cross-function successor edge is not
/// crossed, so the reachable set — and every walk seeded from it — stays inside
/// the function being optimized.
fn reachable_from<'str>(
    host: HostRef<'_, 'str>,
    entry: BlockId,
    owner: FunctionId,
) -> HashSet<BlockId> {
    let mut seen = HashSet::from_iter([entry]);
    let mut stack = vec![entry];
    while let Some(block) = stack.pop() {
        for (_, succ) in host.block_ref(block).successors() {
            // Ownership, not storage: a reattributed own block (owner == the
            // walked function, stored in a foreign arena pre-normalization) is
            // followed; a block owned by another function is not.
            if host.block(succ).parent == Some(owner) && seen.insert(succ) {
                stack.push(succ);
            }
        }
    }
    seen
}

/// Run the sub-passes over an entire function, walking each entry's dominator
/// tree in pre-order and threading the cloned states from each block to its
/// dominated children.
///
/// Blocks unreachable from the function root are not covered by the root walk.
/// The common case is the fall-through after a `call`: a `Call` is a block
/// terminator, but the lifter records no CFG edge from the call site back to
/// its return block, so the entire post-call region (register reload chains,
/// the epilogue, ...) is orphaned. Each such region is walked with its own
/// dominator tree, rooted at its entry — an unreachable block with no
/// predecessor. Once calls grow a proper return edge this finds nothing extra.
///
/// Blocks reachable from more than one entry are flagged `is_shared` in
/// [`SubPass::on_block_entry`]: each walk's dominator tree only sees its own
/// entry's edges, so its dominance claims are invalid for blocks the other
/// entries can also reach.
pub(super) fn run_dominator_walk<'str>(
    host: &mut Context<'str>,
    func_id: FunctionId,
    passes: &[Box<dyn ModuleSubPass<'str>>],
    aliases: Option<&AliasResult>,
) -> bool {
    let root = match host.function_ref(func_id).root() {
        Some(r) => r.id,
        None => return false,
    };

    let root_reachable = reachable_from(host.read_host(), root, func_id);
    let entries: Vec<BlockId> = host
        .function_ref(func_id)
        .iter()
        .filter(|block| !root_reachable.contains(&block.id))
        .filter(|block| block.predecessors().next().is_none())
        .map(|block| block.id)
        .collect();

    let mut seen_count: HashMap<BlockId, u32> = HashMap::default();
    for &entry in std::iter::once(&root).chain(&entries) {
        for block in reachable_from(host.read_host(), entry, func_id) {
            *seen_count.entry(block).or_default() += 1;
        }
    }
    let shared: HashSet<BlockId> = seen_count
        .into_iter()
        .filter_map(|(block, count)| (count > 1).then_some(block))
        .collect();

    // Function-wide affine views, computed once and shared read-only with every
    // entry's walk (a value's arithmetic view is dominance-independent).
    let numbering = super::affine::precompute_forms(host.read_host(), func_id);

    let mut changed = false;
    for entry in std::iter::once(root).chain(entries) {
        let tree = compute_dominators(&host.function_ref(entry.func), entry);
        let mut walk = Walk {
            passes,
            func_id,
            tree: &tree,
            aliases,
            numbering: &numbering,
            shared: &shared,
            owner: func_id,
            changed: false,
        };
        walk.rec(host, entry, &init_states(passes));
        changed |= walk.changed;
    }
    changed
}

// ===========================================================================
// Concrete (host-free) twins — context-split stage 5b-ii(a).
//
// The generic `SubPass`/driver machinery above stays alive: it drives the module
// `concretize` pass (its three body-reading sub-passes run on `H = &mut Context`)
// and the whole-`Context` public entry points (`gvn_function`, `constant_fold_
// function`, `narrow_function`), which have external callers.
//
// These concrete twins drive the *function-pass* GVN chain over a checked-out
// `(&mut FunctionBody, ContextView)` with no threaded mutation host: reads route
// through `body.read_host(cx)`, shallow rewrites through the inherent
// `body.verb(cx, …)` surface (via `Editor`'s `_c` methods). The large shared
// mutation helpers (`materialize`, `MemForward::{record_store,try_load}`,
// `narrow_to`, `simplify_bitwise`/`simplify_compare`) are still reached through a
// scoped `body.host(cx)` — the same pattern the dce sibling (8cbc4b2) uses for
// `replace_terminator_with_branch`; they stay generic because the generic path
// above shares them.
// ===========================================================================

use crate::{ContextView, FunctionBody};

impl Editor {
    /// Concrete twin of [`replace`](Self::replace).
    pub(super) fn replace_c<'str>(
        &mut self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        insn: InstructionId,
        with: ValueId,
    ) {
        body.replace_all_uses_with(cx, ValueId::Instruction(insn), with);
        self.redundant.insert(insn);
    }

    /// Concrete twin of [`replace_with_new_insn`](Self::replace_with_new_insn).
    pub(super) fn replace_with_new_insn_c<'str>(
        &mut self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        block_id: BlockId,
        at: InstructionId,
        mnemonic: Mnemonic,
        size: usize,
    ) -> InstructionId {
        let type_id = body
            .read_host(cx)
            .shared()
            .shared
            .types
            .get_or_make_int(size);
        self.replace_with_new_insn_typed_c(body, cx, block_id, at, mnemonic, type_id)
    }

    /// Concrete twin of
    /// [`replace_with_new_insn_typed`](Self::replace_with_new_insn_typed).
    pub(super) fn replace_with_new_insn_typed_c<'str>(
        &mut self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        block_id: BlockId,
        at: InstructionId,
        mnemonic: Mnemonic,
        type_id: qcode::types::TypeId,
    ) -> InstructionId {
        // `func = body.id` (own function); `block_id.func == body.id` here, so this
        // is behaviour-identical to the generic `push_mnemonic_with_type(block_id.func, …)`.
        let new_id = body.push_mnemonic_with_type(cx, mnemonic, type_id);
        body.insert_insn_before(cx, block_id, at, new_id);
        body.replace_all_uses_with(cx, ValueId::Instruction(at), ValueId::Instruction(new_id));
        self.redundant.insert(at);
        new_id
    }

    /// Concrete twin of [`finish`](Self::finish).
    fn finish_c<'str>(self, body: &mut FunctionBody<'str>, cx: ContextView<'_, 'str>) -> bool {
        let changed = !self.redundant.is_empty();
        for insn in self.redundant {
            body.remove_instruction(cx, insn);
        }
        changed
    }
}

/// Concrete twin of [`SubPass`]: one composable GVN concern over a checked-out
/// `(&mut FunctionBody, ContextView)`. Implemented by the seven function-pass
/// sub-passes ([`Fold`](super::fold::Fold), [`NarrowTrunc`](super::narrow::NarrowTrunc),
/// [`Recognize`](super::intrinsics::Recognize), [`FlagIdiom`](super::flag_idiom::FlagIdiom),
/// [`Identities`](super::identity::Identities), [`MemoryForwarding`](super::memory::MemoryForwarding),
/// [`Cse`](super::cse::Cse)); the three body-reading sub-passes keep only the
/// generic [`SubPass`] impl (they run on the module `concretize` path).
pub(super) trait SubPassC<'str> {
    /// See [`SubPass::init_state`].
    fn init_state(&self) -> Box<dyn Any>;

    /// See [`SubPass::clone_state`].
    fn clone_state(&self, state: &dyn Any) -> Box<dyn Any>;

    /// See [`SubPass::on_block_entry`].
    #[allow(clippy::too_many_arguments)]
    fn on_block_entry(
        &self,
        _body: &mut FunctionBody<'str>,
        _cx: ContextView<'_, 'str>,
        _state: &mut dyn Any,
        _block_id: BlockId,
        _tree: &DominatorTree<BlockId>,
        _aliases: Option<&AliasResult>,
        _numbering: &Numbering,
        _is_shared: bool,
    ) {
    }

    /// See [`SubPass::on_insn`].
    fn on_insn(
        &self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        state: &mut dyn Any,
        ic: &InsnCtx,
        ed: &mut Editor,
    ) -> Claim;

    /// See [`SubPass::after_block`].
    fn after_block(
        &self,
        _body: &mut FunctionBody<'str>,
        _cx: ContextView<'_, 'str>,
        _state: &mut dyn Any,
        _block_id: BlockId,
        _aliases: Option<&AliasResult>,
        _numbering: &Numbering,
    ) {
    }
}

/// Concrete twin of [`init_states`].
fn init_states_c<'str>(passes: &[Box<dyn SubPassC<'str>>]) -> Vec<Box<dyn Any>> {
    passes.iter().map(|p| p.init_state()).collect()
}

/// Concrete twin of [`clone_states`].
fn clone_states_c<'str>(
    passes: &[Box<dyn SubPassC<'str>>],
    states: &[Box<dyn Any>],
) -> Vec<Box<dyn Any>> {
    passes
        .iter()
        .zip(states)
        .map(|(p, s)| p.clone_state(s.as_ref()))
        .collect()
}

/// Concrete twin of [`run_block`].
fn run_block_c<'str>(
    body: &mut FunctionBody<'str>,
    cx: ContextView<'_, 'str>,
    block_id: BlockId,
    passes: &[Box<dyn SubPassC<'str>>],
    states: &mut [Box<dyn Any>],
    aliases: Option<&AliasResult>,
    numbering: &Numbering,
) -> bool {
    let mut ed = Editor::new();
    let insns: Vec<InstructionId> = body.read_host(cx).block(block_id).instructions.clone();

    for insn_id in insns {
        let (id, size, mnemonic) = {
            let insn = body.insn_ref(cx, insn_id);
            (insn.id(), insn.size(), insn.mnemonic().clone())
        };
        let ic = InsnCtx {
            block_id,
            insn_id,
            id,
            size,
            mnemonic: &mnemonic,
            aliases,
            numbering,
        };
        for (pass, state) in passes.iter().zip(states.iter_mut()) {
            if let Claim::Done = pass.on_insn(body, cx, state.as_mut(), &ic, &mut ed) {
                break;
            }
        }
    }

    ed.finish_c(body, cx)
}

/// Concrete twin of [`run_flat_fixpoint`].
pub(super) fn run_flat_fixpoint_c<'str>(
    body: &mut FunctionBody<'str>,
    cx: ContextView<'_, 'str>,
    func_id: FunctionId,
    passes: &[Box<dyn SubPassC<'str>>],
) -> bool {
    let block_ids: Vec<BlockId> = body
        .read_host(cx)
        .function_ref(func_id)
        .iter()
        .map(|block| block.id)
        .collect();

    let numbering = Numbering::default();
    let mut changed_any = false;
    loop {
        let mut changed = false;
        for &block_id in &block_ids {
            let mut states = init_states_c(passes);
            changed |= run_block_c(body, cx, block_id, passes, &mut states, None, &numbering);
        }
        changed_any |= changed;
        if !changed {
            break;
        }
    }
    changed_any
}

/// Concrete twin of [`Walk`].
struct WalkC<'a, 'str> {
    passes: &'a [Box<dyn SubPassC<'str>>],
    tree: &'a DominatorTree<BlockId>,
    aliases: Option<&'a AliasResult>,
    numbering: &'a Numbering,
    shared: &'a HashSet<BlockId>,
    owner: FunctionId,
    changed: bool,
}

impl<'str> WalkC<'_, 'str> {
    fn rec(
        &mut self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        block_id: BlockId,
        inherited: &[Box<dyn Any>],
    ) {
        let mut states = clone_states_c(self.passes, inherited);
        let is_shared = self.shared.contains(&block_id);
        for (pass, state) in self.passes.iter().zip(states.iter_mut()) {
            pass.on_block_entry(
                body,
                cx,
                state.as_mut(),
                block_id,
                self.tree,
                self.aliases,
                self.numbering,
                is_shared,
            );
        }
        self.changed |= run_block_c(
            body,
            cx,
            block_id,
            self.passes,
            &mut states,
            self.aliases,
            self.numbering,
        );
        for (pass, state) in self.passes.iter().zip(states.iter_mut()) {
            pass.after_block(
                body,
                cx,
                state.as_mut(),
                block_id,
                self.aliases,
                self.numbering,
            );
        }
        for &child in self.tree.children_of(block_id) {
            if body.read_host(cx).block(child).parent == Some(self.owner) {
                self.rec(body, cx, child, &states);
            }
        }
    }
}

/// Concrete twin of [`run_dominator_walk`].
pub(super) fn run_dominator_walk_c<'str>(
    body: &mut FunctionBody<'str>,
    cx: ContextView<'_, 'str>,
    func_id: FunctionId,
    passes: &[Box<dyn SubPassC<'str>>],
    aliases: Option<&AliasResult>,
) -> bool {
    let root = match body.read_host(cx).function_ref(func_id).root() {
        Some(r) => r.id,
        None => return false,
    };

    let root_reachable = reachable_from(body.read_host(cx), root, func_id);
    let entries: Vec<BlockId> = body
        .read_host(cx)
        .function_ref(func_id)
        .iter()
        .filter(|block| !root_reachable.contains(&block.id))
        .filter(|block| block.predecessors().next().is_none())
        .map(|block| block.id)
        .collect();

    let mut seen_count: HashMap<BlockId, u32> = HashMap::default();
    for &entry in std::iter::once(&root).chain(&entries) {
        for block in reachable_from(body.read_host(cx), entry, func_id) {
            *seen_count.entry(block).or_default() += 1;
        }
    }
    let shared: HashSet<BlockId> = seen_count
        .into_iter()
        .filter_map(|(block, count)| (count > 1).then_some(block))
        .collect();

    let numbering = super::affine::precompute_forms(body.read_host(cx), func_id);

    let mut changed = false;
    for entry in std::iter::once(root).chain(entries) {
        let tree = compute_dominators(&body.read_host(cx).function_ref(entry.func), entry);
        let mut walk = WalkC {
            passes,
            tree: &tree,
            aliases,
            numbering: &numbering,
            shared: &shared,
            owner: func_id,
            changed: false,
        };
        walk.rec(body, cx, entry, &init_states_c(passes));
        changed |= walk.changed;
    }
    changed
}

#[cfg(test)]
mod tests {
    use crate::AliasResult;
    use crate::gvn::gvn_function;
    use qcode::{
        builder::Builder,
        context::Context,
        testing::TestContext,
        value::{BasicBlock, Function, Value, ValueId},
    };
    use qcode_macro::qcode;

    /// A `call` is a block terminator with no CFG edge to its fall-through, so
    /// the post-call block is unreachable from the function root. `gvn_function`
    /// must still optimize it — otherwise the register reload chains the lifter
    /// emits after every call survive untouched.
    #[test]
    fn test_register_forwarding_in_orphaned_post_call_block() {
        let mut tc = TestContext::new();
        let fun_id = Function::make(&mut tc.ctx, "test".into()).unwrap().id;
        let entry = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x1000, __f)
        };
        let post_call = {
            let __f = tc.ctx.anon_function();
            tc.ctx.get_or_make_block(0x2000, __f)
        };
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fun_id);
            f.set_root(entry).unwrap();
            f.add_block(entry);
            f.add_block(post_call);
        }

        let reg_space = tc.reg_space;
        let eax = ValueId::Varnode(tc.r0_lo32);
        let other = ValueId::Varnode(tc.r1);

        // Entry ends in a `call`; no edge links it to `post_call`.
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            b.push_call(fun_id);
            unsafe { b.dont_finalize() };
        }

        // Orphaned fall-through: store a register and read it straight back.
        let load_id;
        {
            let mut b = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, post_call));
            let c = b.context_mut().get_const(0x42, 4).id();
            b.push_store(c, eax, reg_space);
            let loaded = b.push_load::<false>(eax, 4, reg_space).id();
            b.push_store(loaded, other, reg_space);
            load_id = loaded;
            unsafe { b.dont_finalize() };
        }

        let aliases = AliasResult::simple_for_function(&tc.ctx, fun_id);
        gvn_function(&mut tc.ctx, fun_id, Some(&aliases));

        let ValueId::Instruction(load_insn) = load_id else {
            panic!("push_load should produce an instruction value");
        };
        assert!(
            !BasicBlock::from_id(&tc.ctx, post_call)
                .instruction_ids()
                .contains(&load_insn),
            "forwarding must reach the orphaned post-call block, got:\n{}",
            BasicBlock::from_id(&tc.ctx, post_call)
        );
    }

    /// A block reachable from two orphan-region entries must receive no forwarded
    /// values: each entry-local dominator tree's claims are invalid for it.
    #[test]
    fn test_gvn_function_does_not_forward_into_block_shared_by_orphan_regions() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
                varnode i32 A;
                varnode i32 B;

                fn shared_orphans:
                    <entry>
                        return at 0x1000;

                    <e1>
                        store(A:4, &A <- i32 1);
                        goto <shared>;

                    <e2>
                        store(A:4, &A <- i32 2);
                        goto <shared>;

                    <shared>
                        %v = load(A:4, &A);
                        store(B:4, &B <- %v);
                        return at 0x1001;
                "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        gvn_function(&mut ctx, shared_orphans, Some(&aliases));

        assert!(
            BasicBlock::from_id(&ctx, shared)
                .instruction_ids()
                .contains(&v),
            "the load in a block reachable from two orphan entries must not be forwarded \
             a store from either entry"
        );
    }

    /// `gvn_function` reports whether it rewrote anything, so the pass manager can
    /// see GVN's changes.
    #[test]
    fn gvn_function_reports_changes() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
                varnode i64 A;
                varnode i64 B;

                fn g:
                    <entry>
                        %a = load(A:8, &A);
                        %v1 = %a + %a;
                        %v2 = %a + %a;
                        store(B:8, &B <- %v2);
                        return at 0x1000;
                "
        );

        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        assert!(
            gvn_function(&mut ctx, g, Some(&aliases)),
            "eliminating the duplicate binop must be reported as a change"
        );
        assert!(
            !gvn_function(&mut ctx, g, Some(&aliases)),
            "a second run on the already-optimized function must report no change"
        );
    }
}
