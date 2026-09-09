//! The types that give a function pass its restricted, parallel-safe view of the
//! world (Stage 5 of the parallel-function-passes plan; see `PARALLEL_PASSES.md`).
//!
//! A function pass reads the module's *published interface* through a shared
//! [`ContextView`] and mutates *only its own function* through a `&mut`
//! [`FunctionBody`] — the body borrowed `&mut` in place from the bodies registry
//! by the driver's [`split`](ContextSplit::split). Global effects are **returned**:
//! self-renames, detached functions, and pre-mutation type requests are applied by
//! the driver at the post-run barrier. The pass itself touches no global mutable
//! state, which lets workers run in parallel with the `ContextView` `&`-shared and
//! the bodies disjoint `&mut`.
//!
//! The driver `split`s the context once, borrows every worklist body disjointly
//! (worklist order) via [`select_mut`](jstd::registry::Registry::select_mut), runs
//! each function's pass fixpoint — sequentially or on `std::thread::scope` workers
//! — then drops the split borrow and does the barrier work in worklist order. The
//! types are identical either way.

use std::borrow::Cow;

use jstd::registry::Registry;
use qcode::{
    context::{Context, Shared},
    types::TypeRequest,
    value::{
        BodyView, FunctionBody, FunctionId, FunctionKind,
        function::FunctionInterface,
        insn::Callee,
        util::{body_mut::BodyMut, detached::DetachedMut},
    },
};

use super::PipelineEnv;

/// A detached function minted by a pass this run.
///
/// `slot` is the pass-local [`Callee::Minted`] index used by the owner's IR;
/// it is not a reserved registry ID. The body is built fully *detached* (no
/// registry identity — [`FunctionBody::detached`]), storing only body-local IR
/// IDs, so no owner ID can leak into it; installation simply stamps the identity
/// through [`FunctionBody::install_id`].
pub struct Minted<'str> {
    slot: u32,
    pub interface: FunctionInterface<'str>,
    pub body: FunctionBody<'str>,
}

impl<'str> Minted<'str> {
    /// The pass-local placeholder carried by owner call-like instructions.
    pub const fn callee(&self) -> Callee {
        Callee::Minted(self.slot)
    }

    /// The placeholder's zero-based index in [`Outcome::minted`].
    pub const fn slot(&self) -> u32 {
        self.slot
    }

    /// Consume this detached function for installation under `installed`.
    /// No IR-local IDs are remapped; the detached body simply acquires its
    /// registry identity.
    pub fn into_installed_parts(
        mut self,
        installed: FunctionId,
    ) -> (u32, FunctionInterface<'str>, FunctionBody<'str>) {
        self.body.install_id(installed);
        (self.slot, self.interface, self.body)
    }
}

/// The result of one function-pass run: whether it changed the IR, an optional
/// self-rename claim, functions it minted, and types it needs published.
///
/// Returned **by value** from [`FunctionPass::run`](super::FunctionPass::run), so
/// a pass touches no wrapper scratch: `rename` subsumes the old `Effects` buffer
/// and `minted` subsumes the old analysis wrapper's minted buffer. The driver replays
/// `rename` and installs `minted` at the post-run barrier in worklist order,
/// exactly as it drained the wrapper before — the transport changes, the barrier
/// semantics do not.
pub struct Outcome<'str> {
    /// Whether the pass changed the function's IR (the old `Ok(bool)`).
    pub changed: bool,
    /// A buffered self-rename claim (from `cpp_demangle` / `name_thunks`), applied
    /// as a `get_unique_name` claim at the barrier. Last writer wins across a
    /// function's pass fixpoint, so replay is deterministic.
    pub rename: Option<Cow<'str, str>>,
    /// Functions this run minted (the loop outliners), installed by the driver at
    /// the barrier. Concatenated across a function's pass fixpoint; entry `k`
    /// carries slot `k`, allocated by the driver's per-owner stage cursor.
    pub minted: Vec<Minted<'str>>,
    /// Types this pass needs before it can safely rewrite its body. A requesting
    /// outcome must carry no body/global mutation: the driver publishes these at
    /// the barrier and reruns the owner against the new type generation.
    pub type_requests: Vec<TypeRequest>,
    /// Analyses preserved across the changing runs aggregated into this outcome.
    pub(crate) preserved_analyses: super::PreservedAnalyses,
}

impl Default for Outcome<'_> {
    fn default() -> Self {
        Self {
            changed: false,
            rename: None,
            minted: Vec::new(),
            type_requests: Vec::new(),
            preserved_analyses: super::PreservedAnalyses::all(),
        }
    }
}

impl<'str> Outcome<'str> {
    /// The analyses this particular invocation reported preserving.
    pub fn preserved_analyses(&self) -> &super::PreservedAnalyses {
        &self.preserved_analyses
    }

    /// An unchanged outcome — no rename, no minted functions.
    pub fn unchanged() -> Self {
        Self::default()
    }

    /// An outcome reporting `changed`, with no rename or minted functions (the
    /// common case for the mechanical `Ok(bool)` → `Ok(Outcome::changed(bool))`
    /// sweep).
    pub fn changed(changed: bool) -> Self {
        if changed {
            Self {
                changed: true,
                rename: None,
                minted: Vec::new(),
                type_requests: Vec::new(),
                preserved_analyses: super::PreservedAnalyses::none(),
            }
        } else {
            Self::default()
        }
    }

    /// A changed outcome carrying a self-rename claim (the driver uniquifies and
    /// applies it at the barrier).
    pub fn renamed(name: Cow<'str, str>) -> Self {
        Self {
            changed: true,
            rename: Some(name),
            minted: Vec::new(),
            type_requests: Vec::new(),
            preserved_analyses: super::PreservedAnalyses::none(),
        }
    }

    /// An outcome carrying detached functions produced by an outlining pass.
    pub fn with_minted(changed: bool, minted: Vec<Minted<'str>>) -> Self {
        let mut outcome = Self::changed(changed || !minted.is_empty());
        outcome.minted = minted;
        outcome
    }

    /// Ask the driver to publish `request` at the barrier and rerun this
    /// function. Call this before mutating the body.
    pub fn requesting_type(request: TypeRequest) -> Self {
        Self::requesting_types([request])
    }

    /// Batch sibling of [`requesting_type`](Self::requesting_type).
    pub fn requesting_types(requests: impl IntoIterator<Item = TypeRequest>) -> Self {
        Self {
            type_requests: requests.into_iter().collect(),
            ..Self::default()
        }
    }

    /// Report that this particular invocation preserved global analysis `A`.
    /// This is intentionally outcome-level: a different path through the same
    /// pass may report a different preservation set.
    pub fn preserving_global<A: super::GlobalAnalysis>(mut self) -> Self {
        self.preserved_analyses.preserve_global::<A>();
        self
    }

    /// Report that this particular invocation preserved local analysis `A` for
    /// the function being transformed.
    pub fn preserving_local<A: super::LocalAnalysis>(mut self) -> Self {
        self.preserved_analyses.preserve_local::<A>();
        self
    }
}

impl<'str> From<bool> for Outcome<'str> {
    fn from(changed: bool) -> Self {
        Self::changed(changed)
    }
}

#[cfg(test)]
mod preservation_tests {
    use super::*;
    use crate::{AliasAnalysis, CallGraphAnalysis};

    #[test]
    fn unchanged_function_outcome_preserves_everything() {
        let outcome = Outcome::changed(false);
        assert!(
            outcome
                .preserved_analyses()
                .preserves_global_analysis::<CallGraphAnalysis>()
        );
        assert!(
            outcome
                .preserved_analyses()
                .preserves_local_analysis::<AliasAnalysis>()
        );
    }

    #[test]
    fn changed_function_outcome_invalidates_unreported_analyses() {
        let outcome = Outcome::changed(true).preserving_global::<CallGraphAnalysis>();
        assert!(
            outcome
                .preserved_analyses()
                .preserves_global_analysis::<CallGraphAnalysis>()
        );
        assert!(
            !outcome
                .preserved_analyses()
                .preserves_local_analysis::<AliasAnalysis>()
        );
    }
}

/// The read-only module interface a function pass may consult: architecture /
/// ABI, the interners (mintable through `&self`), and every *other* function's
/// published interface (name, address, signature, purity, clobber/write
/// summaries) — but **not** other functions' in-flight bodies (except the
/// pure-bodies view, which is stage-invariant; added with the gvn port).
///
/// The **bodies-free** module view of the context-split design (stage 5b-ii):
/// `{shared, interfaces, env}`. A function pass structurally cannot read another
/// function's body through it — holding a `ContextView` is the proof that
/// regimes 1–3 are frozen while workers hold disjoint `&mut` bodies. It is
/// `Copy` (it holds only shared references), so a worker hands the same view to
/// every helper and sub-ref.
#[derive(Clone, Copy)]
pub struct ContextView<'ctx, 'str> {
    shared: &'ctx Shared<'str>,
    interfaces: &'ctx Registry<FunctionId, FunctionInterface<'str>>,
    env: &'ctx PipelineEnv,
}

impl<'ctx, 'str> ContextView<'ctx, 'str> {
    /// Build a view over `ctx`'s shared state + interfaces and the pipeline
    /// environment. Takes the whole `&Context` for caller convenience (the
    /// driver) and narrows; [`Context`]'s bodies are NOT captured —
    /// prefer [`split`](ContextSplit::split), which proves that with a
    /// simultaneous `&mut` bodies borrow.
    pub fn new(ctx: &'ctx Context<'str>, env: &'ctx PipelineEnv) -> Self {
        Self {
            shared: &ctx.shared,
            interfaces: &ctx.interfaces,
            env,
        }
    }

    /// The pipeline environment (register layout, ABI, OS, bitness, stack
    /// pointer, shared alias base).
    pub fn env(&self) -> &'ctx PipelineEnv {
        self.env
    }

    /// Narrow to the environment-free view the block-local passes in
    /// [`qcode_passes`] take. Those passes need only the shared IR state and
    /// the published interfaces; dropping the env is what lets them live in a
    /// crate below this one.
    pub fn pass(self) -> qcode_passes::PassCtx<'ctx, 'str> {
        qcode_passes::PassCtx::from_parts(self.shared, self.interfaces)
    }

    /// The published interface of function `f` (name, address, signature, purity,
    /// clobber/write summaries) — the caller-reasoning surface. Interfaces are
    /// never checked out, so this always reads the shared registry.
    pub fn interface(&self, f: FunctionId) -> &'ctx FunctionInterface<'str> {
        &self.interfaces[f]
    }

    /// The module's shared IR state (interners, spaces, registers, name/address
    /// maps, memory image, truths).
    pub fn shr(&self) -> &'ctx Shared<'str> {
        self.shared
    }

    /// The whole interface registry (for building a [`BodyMut`]/[`BodyView`]).
    pub fn interfaces(&self) -> &'ctx Registry<FunctionId, FunctionInterface<'str>> {
        self.interfaces
    }

    /// Build the mutation host for a pass's exclusively borrowed body.
    pub fn host<'body>(self, body: &'body mut FunctionBody<'str>) -> BodyMut<'body, 'str>
    where
        'ctx: 'body,
    {
        BodyMut::new(body, self.shared, self.interfaces)
    }

    /// Build the static read view for a pass's borrowed body.
    pub fn body_view<'body>(self, body: &'body FunctionBody<'str>) -> BodyView<'body, 'str>
    where
        'ctx: 'body,
    {
        BodyView::new(body, self.shared, self.interfaces)
    }
}

/// The driver-side disjoint borrow of the context-split design (00-overview):
/// `&mut bodies` alongside a frozen, bodies-free [`ContextView`]. Rust's
/// disjoint-field borrows prove the safety — no `unsafe`, no relocation.
pub trait ContextSplit<'str> {
    /// Split into the mutable bodies registry and the read-only module view.
    /// `env` is threaded as an argument so `Context` stays env-free.
    fn split<'a>(
        &'a mut self,
        env: &'a PipelineEnv,
    ) -> (
        &'a mut Registry<FunctionId, FunctionBody<'str>>,
        ContextView<'a, 'str>,
    );
}

impl<'str> ContextSplit<'str> for Context<'str> {
    fn split<'a>(
        &'a mut self,
        env: &'a PipelineEnv,
    ) -> (
        &'a mut Registry<FunctionId, FunctionBody<'str>>,
        ContextView<'a, 'str>,
    ) {
        (
            &mut self.bodies,
            ContextView {
                shared: &self.shared,
                interfaces: &self.interfaces,
                env,
            },
        )
    }
}

/// Mint a detached function for `owner`, advancing the driver-owned stage cursor.
pub fn mint_function<'str>(
    owner: &FunctionBody<'str>,
    next_minted: &mut u32,
    minted: &mut Vec<Minted<'str>>,
    name: Cow<'str, str>,
    kind: FunctionKind,
    pure: bool,
) -> Callee {
    let slot = *next_minted;
    *next_minted = next_minted
        .checked_add(1)
        .expect("more than u32::MAX functions minted by one function stage");
    let mut interface = FunctionInterface::new(name);
    interface.kind = kind;
    if pure {
        interface.signature.get_or_insert_default().is_pure = true;
        // A pure minted lambda's register channel is (vacuously) materialized —
        // the unified effect state that `is_reg_materialized` now reads.
        interface.effects.register = qcode::value::RegisterChannelState::Materialized(
            qcode::value::RegisterInterfaceMap::default(),
        );
    }
    let _ = owner;
    minted.push(Minted {
        slot,
        interface,
        body: FunctionBody::detached(),
    });
    Callee::Minted(slot)
}

/// Read the owner while mutating one of its detached minted bodies through the
/// body-local [`DetachedMut`] surface.
pub fn host_with_minted<'body, 'ctx, 'str>(
    owner: &'body FunctionBody<'str>,
    minted: &'body mut [Minted<'str>],
    cx: ContextView<'ctx, 'str>,
    callee: Callee,
) -> (BodyView<'body, 'str>, DetachedMut<'body, 'str>)
where
    'ctx: 'body,
{
    let slot = callee
        .minted()
        .expect("host_with_minted requires a minted callee placeholder");
    let entry = minted
        .iter_mut()
        .find(|entry| entry.slot == slot)
        .expect("host_with_minted: not a function minted this run");
    (
        cx.body_view(owner),
        DetachedMut::new(&mut entry.body, cx.shr(), cx.interfaces()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::dummy_env;
    use qcode::value::{
        QCodeView, ValueId,
        block::BlockId,
        block_param::BlockParamId,
        insn::{InstructionId, Mnemonic},
    };
    use wazabin_qcode_macro::qcode;

    #[test]
    fn mint_slots_are_infallible_ordered_and_persist_across_outcomes() {
        let mut ctx = Context::new();
        qcode!(ctx, "fn mint_owner: <entry> return at i64 0;");
        let installed_second = FunctionBody::make(&mut ctx, "installed_second".into())
            .unwrap()
            .id;
        let env = dummy_env();
        let (bodies, view) = ctx.split(&env);
        let mut slots = bodies.select_mut(&[mint_owner]);
        let (owner, _) = slots.split_first_mut().unwrap();
        let mut next_minted = 0;

        // Model two pass outcomes aggregated across one stage fixpoint. The
        // driver-owned cursor is shared even though each pass returns a fresh Vec.
        let mut first_outcome = Vec::new();
        let first = mint_function(
            owner,
            &mut next_minted,
            &mut first_outcome,
            "first".into(),
            FunctionKind::Lambda,
            true,
        );
        let mut second_outcome = Vec::new();
        let second = mint_function(
            owner,
            &mut next_minted,
            &mut second_outcome,
            "second".into(),
            FunctionKind::Lambda,
            true,
        );
        assert_eq!(first, Callee::Minted(0));
        assert_eq!(second, Callee::Minted(1));

        let mut next_stage = 0;
        let mut next_stage_outcome = Vec::new();
        assert_eq!(
            mint_function(
                owner,
                &mut next_stage,
                &mut next_stage_outcome,
                "next_stage".into(),
                FunctionKind::Lambda,
                true,
            ),
            Callee::Minted(0)
        );

        first_outcome.extend(second_outcome);
        assert_eq!(
            first_outcome.iter().map(Minted::slot).collect::<Vec<_>>(),
            [0, 1]
        );

        // A minted-to-minted call retains the placeholder and cannot silently
        // become a real call back to the owner.
        {
            let (_, mut host) = host_with_minted(owner, &mut first_outcome, view, first);
            let ty = view.shr().types.get_or_make_int(0);
            let call = host.push_mnemonic_with_type(
                Mnemonic::Call(qcode::value::insn::Call {
                    target: second,
                    args: Vec::new(),
                    clobbers: Vec::new(),
                    tag: Default::default(),
                }),
                ty,
            );
            assert!(matches!(
                host.mnemonic(call),
                Mnemonic::Call(call) if call.target == Callee::Minted(1)
            ));
        }
        assert_eq!(
            first_outcome[0]
                .body
                .resolve_minted_callee(1, installed_second),
            1
        );
    }

    /// A minted body is built fully detached (no registry identity) through the
    /// body-local [`DetachedMut`] surface; installation stamps the identity and the
    /// stored local ids qualify against it. The former "colliding-local" ambient
    /// test is obsolete: an owner id can no longer enter a minted body — there is
    /// no ambient qualifier in scope, and `id()` panics before install.
    #[test]
    fn minted_body_is_detached_then_installs_local_ids() {
        let mut ctx = Context::new();
        qcode!(ctx, "fn rebind_owner: <owner_block> return at i64 0;");
        let installed = FunctionBody::make(&mut ctx, "installed".into()).unwrap().id;

        let env = dummy_env();
        let (bodies, view) = ctx.split(&env);
        let mut slots = bodies.select_mut(&[rebind_owner]);
        let (owner_fun, _) = slots.split_first_mut().unwrap();
        let mut next_minted = 0;
        let mut outcome = Vec::new();
        let placeholder = mint_function(
            owner_fun,
            &mut next_minted,
            &mut outcome,
            "detached".into(),
            FunctionKind::Lambda,
            true,
        );
        let (root, child, edge);
        {
            let (_, mut host) = host_with_minted(owner_fun, &mut outcome, view, placeholder);
            // (a) The body has no registry identity while detached.
            assert_eq!(host.body.try_id(), None);
            root = host.make_block();
            child = host.make_block();
            edge = host.add_cfg_edge(root, child);
            host.set_root(root);
            host.rename_block(root, "minted_root".into()).unwrap();
        }

        // (b) Installation stamps the identity; the local ids resolve against it.
        let minted = outcome.pop().unwrap();
        let (slot, _, mut detached) = minted.into_installed_parts(installed);
        assert_eq!(slot, 0);
        assert_eq!(detached.try_id(), Some(installed));
        assert_eq!(detached.root_id(), Some(root));
        assert_eq!(detached.edge(edge).from, root);
        assert_eq!(detached.edge(edge).to, child);
        let root_q = BlockId::new(installed, root);
        let child_q = BlockId::new(installed, child);
        let host = BodyMut::new(&mut detached, view.shr(), view.interfaces());
        assert_eq!(
            host.view()
                .block_ref(root_q)
                .successors()
                .map(|(_, successor)| successor)
                .collect::<Vec<_>>(),
            [child_q]
        );
        assert_eq!(
            host.view()
                .function_ref(installed)
                .local_named("minted_root"),
            Some(ValueId::BasicBlock(root_q))
        );
    }

    fn two_functions_with_users(
        mut ctx: &mut Context<'static>,
    ) -> (FunctionId, FunctionId, InstructionId) {
        qcode!(
            ctx,
            "
            fn body_users_a:
                <a_entry>
                    %a_def = i64 1 + i64 2;
                    %a_user = %a_def + i64 3;
                    return at %a_user;

            fn body_users_b:
                <b_entry>
                    %b_def = i64 1 + i64 2;
                    %b_user = %b_def + i64 3;
                    return at %b_user;
            "
        );
        let foreign = ctx
            .function_ref(body_users_a)
            .root()
            .unwrap()
            .instruction_ids()[0];
        (body_users_a, body_users_b, foreign)
    }

    fn two_functions_with_params(
        mut ctx: &mut Context<'static>,
    ) -> (FunctionId, BlockParamId, BlockParamId) {
        qcode!(
            ctx,
            "
            fn body_params_a:
            <a_entry @a:i64>
                return at @a;

            fn body_params_b:
            <b_entry @b:i64>
                return at @b;
            "
        );
        let foreign = ctx
            .function_ref(body_params_a)
            .root()
            .unwrap()
            .params()
            .next()
            .unwrap()
            .id;
        let own = ctx
            .function_ref(body_params_b)
            .root()
            .unwrap()
            .params()
            .next()
            .unwrap()
            .id;
        assert_eq!(
            foreign.local, own.local,
            "regression setup requires colliding local parameter ids"
        );
        (body_params_b, foreign, own)
    }

    #[test]
    #[should_panic(expected = "block parameter belongs to another function")]
    fn function_body_block_param_mut_rejects_foreign_id_with_colliding_local() {
        let mut ctx = Context::new();
        let (own_id, foreign, _) = two_functions_with_params(&mut ctx);
        let env = dummy_env();
        let (bodies, _view) = ctx.split(&env);
        let mut slots = bodies.select_mut(&[own_id]);
        let (own, _) = slots.split_first_mut().unwrap();

        let _ = own.block_param_mut(foreign);
    }

    #[test]
    #[should_panic(expected = "block parameter belongs to another function")]
    fn function_body_block_param_read_rejects_foreign_id_with_colliding_local() {
        let mut ctx = Context::new();
        let (own_id, foreign, _) = two_functions_with_params(&mut ctx);
        let env = dummy_env();
        let (bodies, _view) = ctx.split(&env);
        let mut slots = bodies.select_mut(&[own_id]);
        let (own, _) = slots.split_first_mut().unwrap();

        let _ = own.block_param(foreign);
    }

    #[test]
    #[should_panic(expected = "attempted to read")]
    fn function_body_param_ref_rejects_foreign_id_with_colliding_local() {
        let mut ctx = Context::new();
        let (own_id, foreign, _) = two_functions_with_params(&mut ctx);
        let env = dummy_env();
        let (bodies, view) = ctx.split(&env);
        let mut slots = bodies.select_mut(&[own_id]);
        let (own, _) = slots.split_first_mut().unwrap();

        let _ = view.body_view(own).param_ref(foreign);
    }

    #[test]
    fn function_body_users_of_rejects_foreign_owned_values() {
        let mut ctx = Context::new();
        let (_, own_id, foreign) = two_functions_with_users(&mut ctx);
        let env = dummy_env();
        let (bodies, _view) = ctx.split(&env);
        let mut slots = bodies.select_mut(&[own_id]);
        let (own, _) = slots.split_first_mut().unwrap();

        assert!(own.users_of(ValueId::Instruction(foreign)).is_empty());
    }

    #[test]
    #[should_panic(expected = "cannot replace uses of a value owned by another function")]
    fn function_body_replace_all_uses_rejects_foreign_old_value() {
        let mut ctx = Context::new();
        let (_, own_id, foreign) = two_functions_with_users(&mut ctx);
        let env = dummy_env();
        let (bodies, _view) = ctx.split(&env);
        let mut slots = bodies.select_mut(&[own_id]);
        let (own, _) = slots.split_first_mut().unwrap();

        own.replace_all_uses_with(
            ValueId::Instruction(foreign),
            ValueId::Literal(0usize.into()),
        );
    }

    /// The split's borrow story: hold `&mut bodies[fid]` (and mutate through the
    /// inherent `FunctionBody` verbs) while simultaneously reading the module through
    /// the bodies-free `ContextView` — interners, interfaces, env. Rust's
    /// disjoint-field borrows prove no aliasing; no `unsafe` anywhere.
    #[test]
    fn split_borrows_bodies_and_view_disjointly() {
        let mut ctx = Context::new();
        qcode!(ctx, "fn callee: <c_entry> return at 0x2000;");
        qcode!(ctx, "fn f: <entry> return at 0x1000;");
        let env = dummy_env();

        let insn_count_before;
        {
            let (bodies, view) = ctx.split(&env);
            // Disjoint &mut borrows of two distinct bodies, worklist order.
            let mut slots = bodies.select_mut(&[f, callee]);
            let (own, rest) = slots.split_first_mut().unwrap();

            // View reads while the body borrows are live.
            assert_eq!(view.interface(callee).name.as_ref(), "callee");
            let k = view.shr().get_const(42, 8);

            // Mutate the own body through its inherent verbs, consuming the
            // view-minted literal — the exact pass-shaped usage.
            let root = BlockId::new(f, own.root_id().expect("root"));
            insn_count_before = own.block(root).instructions.len();
            let insn = own.push_mnemonic(
                view.shr(),
                Mnemonic::Zext(qcode::value::insn::Zext {
                    src: k.localize(f),
                    size: 8,
                }),
                8,
            );
            let first = InstructionId::new(root.func, own.block(root).instructions[0]);
            own.insert_insn_before(root, first, insn);
            let _ = rest;
        }

        // The split borrow has ended; the whole context is usable again.
        let root = BlockId::new(f, ctx.bodies[f].root_id().expect("root"));
        assert_eq!(
            ctx.bodies[f].block(root).instructions.len(),
            insn_count_before + 1
        );
        assert!(ctx.shared.values.literals.iter().any(|l| l.value == 42));
    }
}
