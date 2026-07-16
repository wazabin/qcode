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

use qcode::value::{
    QCodeView, ValueId,
    block::BlockId,
    function::FunctionId,
    insn::{InstructionId, Mnemonic},
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
    pub type_id: qcode::types::TypeId,
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
}

/// All blocks reachable from `entry` via CFG successor edges (including `entry`),
/// **confined to `owner`'s own blocks**: a cross-function successor edge is not
/// crossed, so the reachable set — and every walk seeded from it — stays inside
/// the function being optimized.
fn reachable_from<'ctx, 'str: 'ctx>(
    host: impl QCodeView<'ctx, 'str>,
    entry: BlockId,
    owner: FunctionId,
) -> HashSet<BlockId> {
    let mut seen = HashSet::from_iter([entry]);
    let mut stack = vec![entry];
    while let Some(block) = stack.pop() {
        for (_, succ) in host.block_ref(block).successors() {
            // Ownership is derived from the storing arena (`succ.func`): a
            // successor in the walked function is followed; one owned by another
            // function is not.
            if succ.func == owner && seen.insert(succ) {
                stack.push(succ);
            }
        }
    }
    seen
}

// The body-local walker drives the function-pass GVN chain over a checked-out
// `(&mut FunctionBody, ContextView)` with no threaded mutation host: reads route
// through `cx.body_view(body)`, shallow rewrites through the inherent
// `body.verb(cx, …)` surface (via `Editor` methods). The large shared
// mutation helpers are reached through the same body-local mutation boundary.
// ===========================================================================

use crate::{ContextView, FunctionBody};

impl Editor {
    /// Forward all uses and mark the old instruction redundant.
    pub(super) fn replace<'str>(
        &mut self,
        body: &mut FunctionBody<'str>,
        _cx: ContextView<'_, 'str>,
        insn: InstructionId,
        with: ValueId,
    ) {
        body.replace_all_uses_with(ValueId::Instruction(insn), with);
        self.redundant.insert(insn);
    }

    /// Insert a replacement instruction with an integer result type.
    pub(super) fn replace_with_new_insn<'str>(
        &mut self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        block_id: BlockId,
        at: InstructionId,
        mnemonic: Mnemonic,
        size: usize,
    ) -> InstructionId {
        let type_id = cx.body_view(body).shared().types.get_int(size);
        self.replace_with_new_insn_typed(body, cx, block_id, at, mnemonic, type_id)
    }

    /// Insert a replacement instruction with an explicit result type.
    pub(super) fn replace_with_new_insn_typed<'str>(
        &mut self,
        body: &mut FunctionBody<'str>,
        _cx: ContextView<'_, 'str>,
        block_id: BlockId,
        at: InstructionId,
        mnemonic: Mnemonic,
        type_id: qcode::types::TypeId,
    ) -> InstructionId {
        // `func = body.id` (own function); `block_id.func == body.id` here, so this
        // is behaviour-identical to the generic `push_mnemonic_with_type(block_id.func, …)`.
        let new_id = body.push_mnemonic_with_type(mnemonic, type_id);
        body.insert_insn_before(block_id, at, new_id);
        body.replace_all_uses_with(ValueId::Instruction(at), ValueId::Instruction(new_id));
        self.redundant.insert(at);
        new_id
    }

    /// Remove every instruction made redundant during the block walk.
    fn finish<'str>(self, body: &mut FunctionBody<'str>, _cx: ContextView<'_, 'str>) -> bool {
        let changed = !self.redundant.is_empty();
        let mut redundant: Vec<_> = self.redundant.into_iter().collect();
        redundant.sort_unstable();
        for insn in redundant {
            body.remove_instruction(insn);
        }
        changed
    }
}

/// One composable GVN concern over a checked-out
/// `(&mut FunctionBody, ContextView)`. Implemented by the seven function-pass
/// sub-passes ([`Fold`](super::fold::Fold), [`NarrowTrunc`](super::narrow::NarrowTrunc),
/// [`Recognize`](super::intrinsics::Recognize), [`FlagIdiom`](super::flag_idiom::FlagIdiom),
/// [`Identities`](super::identity::Identities), [`MemoryForwarding`](super::memory::MemoryForwarding),
/// and [`Cse`](super::cse::Cse).
pub(super) trait SubPass<'str> {
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

/// Create one fresh erased state slot per sub-pass.
fn init_states<'str>(passes: &[Box<dyn SubPass<'str>>]) -> Vec<Box<dyn Any>> {
    passes.iter().map(|p| p.init_state()).collect()
}

/// Clone each sub-pass state for a dominated child.
fn clone_states<'str>(
    passes: &[Box<dyn SubPass<'str>>],
    states: &[Box<dyn Any>],
) -> Vec<Box<dyn Any>> {
    passes
        .iter()
        .zip(states)
        .map(|(p, s)| p.clone_state(s.as_ref()))
        .collect()
}

/// Run the chain over one block and remove redundant instructions afterwards.
fn run_block<'str>(
    body: &mut FunctionBody<'str>,
    cx: ContextView<'_, 'str>,
    block_id: BlockId,
    passes: &[Box<dyn SubPass<'str>>],
    states: &mut [Box<dyn Any>],
    aliases: Option<&AliasResult>,
    numbering: &Numbering,
) -> bool {
    let mut ed = Editor::new();
    let insns: Vec<InstructionId> = cx.body_view(body).block_ref(block_id).instruction_ids();

    for insn_id in insns {
        let (id, type_id, size, mnemonic) = {
            let insn = cx.body_view(body).insn_ref(insn_id);
            (
                insn.id(),
                insn.type_id(),
                insn.size(),
                insn.mnemonic().clone(),
            )
        };
        let ic = InsnCtx {
            block_id,
            insn_id,
            id,
            type_id,
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

    ed.finish(body, cx)
}

/// Run the sub-passes over a single block with fresh state and no block-boundary
/// hooks (no dominator tree exists for a lone block), over a checked-out
/// `(&mut FunctionBody, ContextView)`.
pub(super) fn run_single_block<'str>(
    body: &mut FunctionBody<'str>,
    cx: ContextView<'_, 'str>,
    block_id: BlockId,
    passes: &[Box<dyn SubPass<'str>>],
    aliases: Option<&AliasResult>,
) -> bool {
    // No function context for a lone block: memory forwarding falls back to
    // degenerate per-pointer bases (exact-match only).
    let numbering = Numbering::default();
    let mut states = init_states(passes);
    run_block(body, cx, block_id, passes, &mut states, aliases, &numbering)
}

/// Run one sub-pass chain to a flat per-function fixpoint.
pub(super) fn run_flat_fixpoint<'str>(
    body: &mut FunctionBody<'str>,
    cx: ContextView<'_, 'str>,
    func_id: FunctionId,
    passes: &[Box<dyn SubPass<'str>>],
) -> bool {
    let block_ids: Vec<BlockId> = cx
        .body_view(body)
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
            changed |= run_block(body, cx, block_id, passes, &mut states, None, &numbering);
        }
        changed_any |= changed;
        if !changed {
            break;
        }
    }
    changed_any
}

/// Per-entry dominator-walk state.
struct Walk<'a, 'str> {
    passes: &'a [Box<dyn SubPass<'str>>],
    tree: &'a DominatorTree<BlockId>,
    aliases: Option<&'a AliasResult>,
    numbering: &'a Numbering,
    shared: &'a HashSet<BlockId>,
    owner: FunctionId,
    changed: bool,
}

impl<'str> Walk<'_, 'str> {
    fn rec(
        &mut self,
        body: &mut FunctionBody<'str>,
        cx: ContextView<'_, 'str>,
        block_id: BlockId,
        mut states: Vec<Box<dyn Any>>,
    ) {
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
        self.changed |= run_block(
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
        let owner = self.owner;
        let mut children = self
            .tree
            .children_of(block_id)
            .iter()
            .copied()
            // Ownership is derived from the storing arena (`child.func`).
            .filter(|child| child.func == owner)
            .peekable();
        let mut remaining_states = Some(states);
        while let Some(child) = children.next() {
            // The parent snapshot is dead after its children have been seeded.
            // Clone it only for sibling branches; the final child can take the
            // original state. Linear dominator chains therefore allocate no
            // state clones at all.
            let child_states = if children.peek().is_some() {
                clone_states(
                    self.passes,
                    remaining_states.as_ref().expect("parent GVN state"),
                )
            } else {
                remaining_states.take().expect("parent GVN state")
            };
            self.rec(body, cx, child, child_states);
        }
    }
}

/// Run the canonical GVN chain over a function's dominator regions.
pub(super) fn run_dominator_walk<'str>(
    body: &mut FunctionBody<'str>,
    cx: ContextView<'_, 'str>,
    func_id: FunctionId,
    passes: &[Box<dyn SubPass<'str>>],
    aliases: Option<&AliasResult>,
) -> bool {
    let root = match cx.body_view(body).function_ref(func_id).root() {
        Some(r) => r.id,
        None => return false,
    };

    let root_reachable = reachable_from(cx.body_view(body), root, func_id);
    let entries: Vec<BlockId> = cx
        .body_view(body)
        .function_ref(func_id)
        .iter()
        .filter(|block| !root_reachable.contains(&block.id))
        .filter(|block| block.predecessors().next().is_none())
        .map(|block| block.id)
        .collect();

    let mut seen_count: HashMap<BlockId, u32> = HashMap::default();
    for &entry in std::iter::once(&root).chain(&entries) {
        for block in reachable_from(cx.body_view(body), entry, func_id) {
            *seen_count.entry(block).or_default() += 1;
        }
    }
    let shared: HashSet<BlockId> = seen_count
        .into_iter()
        .filter_map(|(block, count)| (count > 1).then_some(block))
        .collect();

    let numbering = super::affine::precompute_forms(cx.body_view(body), func_id);

    let mut changed = false;
    for entry in std::iter::once(root).chain(entries) {
        let tree = compute_dominators(&cx.body_view(body).function_ref(entry.func), entry);
        let mut walk = Walk {
            passes,
            tree: &tree,
            aliases,
            numbering: &numbering,
            shared: &shared,
            owner: func_id,
            changed: false,
        };
        walk.rec(body, cx, entry, init_states(passes));
        changed |= walk.changed;
    }
    changed
}

#[cfg(test)]
mod tests {
    use std::{any::Any, cell::Cell, rc::Rc};

    use super::{Claim, Editor, InsnCtx, SubPass, run_dominator_walk};
    use crate::AliasResult;
    use crate::gvn::gvn_function;
    use qcode::{
        context::Context,
        testing::TestContext,
        value::{BasicBlock, FunctionBody, Value, ValueId},
    };
    use qcode_macro::qcode;

    struct CountStateClones(Rc<Cell<usize>>);

    impl<'str> SubPass<'str> for CountStateClones {
        fn init_state(&self) -> Box<dyn Any> {
            Box::new(())
        }

        fn clone_state(&self, _state: &dyn Any) -> Box<dyn Any> {
            self.0.set(self.0.get() + 1);
            Box::new(())
        }

        fn on_insn(
            &self,
            _body: &mut FunctionBody<'str>,
            _cx: crate::ContextView<'_, 'str>,
            _state: &mut dyn Any,
            _ic: &InsnCtx,
            _ed: &mut Editor,
        ) -> Claim {
            Claim::Pass
        }
    }

    #[test]
    fn linear_dominator_walk_moves_state_without_cloning() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
                fn linear:
                    <entry>
                        goto <middle>;
                    <middle>
                        goto <exit>;
                    <exit>
                        return at 0x1000;
            "
        );

        let clones = Rc::new(Cell::new(0));
        let passes: Vec<Box<dyn SubPass<'_>>> =
            vec![Box::new(CountStateClones(Rc::clone(&clones)))];
        crate::with_body_mut(&mut ctx, linear, |body, cx| {
            run_dominator_walk(body, cx, linear, &passes, None);
        });

        assert_eq!(clones.get(), 0);
    }

    /// A `call` is a block terminator with no CFG edge to its fall-through, so
    /// the post-call block is unreachable from the function root. `gvn_function`
    /// must still optimize it — otherwise the register reload chains the lifter
    /// emits after every call survive untouched.
    #[test]
    fn test_register_forwarding_in_orphaned_post_call_block() {
        let mut tc = TestContext::new();
        let fun_id = FunctionBody::make(&mut tc.ctx, "test".into()).unwrap().id;
        // Both blocks are born into `fun_id`'s own arena (self-stored,
        // self-parented). The orphan-ness of `post_call` is a *CFG* property — the
        // `call` terminating `entry` grows no edge to it — not a storage one, so
        // this fixture exercises the same orphan-region walk without the legacy
        // reattributed-block shape (a block stored in a foreign arena).
        let entry = tc.ctx.get_or_make_block(0x1000, fun_id);
        let post_call = tc.ctx.get_or_make_block(0x2000, fun_id);
        {
            let mut f = FunctionBody::from_id_mut(&mut tc.ctx, fun_id);
            f.set_root(entry).unwrap();
            f.add_block(post_call);
        }

        let reg_space = tc.reg_space;
        let eax = ValueId::Varnode(tc.r0_lo32);
        let other = ValueId::Varnode(tc.r1);

        // Entry ends in a `call`; no edge links it to `post_call`.
        {
            let mut b = tc.ctx.builder(entry);
            b.push_call(fun_id);
        }

        // Orphaned fall-through: store a register and read it straight back.
        let load_id;
        {
            let mut b = tc.ctx.builder(post_call);
            let c = b.shr().get_const(0x42, 4);
            b.push_store(c, eax, reg_space);
            let loaded = b.push_load::<false>(eax, 4, reg_space).id();
            b.push_store(loaded, other, reg_space);
            load_id = loaded;
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
