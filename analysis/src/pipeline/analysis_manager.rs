//! Typed caches for disposable module and function analyses.

use std::{
    any::{Any, TypeId},
    collections::HashMap,
};

use qcode::{
    context::Context,
    value::{FunctionBody, FunctionId},
};

use super::ContextView;

/// A derived analysis over the whole module.
pub trait GlobalAnalysis: 'static {
    type Result: Any + Send + Sync;

    fn analyze(ctx: &Context<'_>) -> Self::Result;
}

/// A derived analysis over one function body.
pub trait LocalAnalysis: 'static {
    type Result: Any + Send;

    fn analyze<'str>(body: &FunctionBody<'str>, cx: ContextView<'_, 'str>) -> Self::Result;
}

/// The analyses a changing pass leaves valid.
///
/// Local preservation always applies only to the function being run, or to the
/// functions named by a module pass's `changed_functions` outcome. It can never
/// invalidate another function as a side effect of a function pass.
#[derive(Clone, Debug)]
pub struct PreservedAnalyses {
    all_global: bool,
    globals: Vec<TypeId>,
    all_local: bool,
    locals: Vec<TypeId>,
}

impl PreservedAnalyses {
    pub fn all() -> Self {
        Self {
            all_global: true,
            globals: Vec::new(),
            all_local: true,
            locals: Vec::new(),
        }
    }

    pub fn none() -> Self {
        Self {
            all_global: false,
            globals: Vec::new(),
            all_local: false,
            locals: Vec::new(),
        }
    }

    pub fn preserve_global<A: GlobalAnalysis>(&mut self) {
        let id = TypeId::of::<A>();
        if !self.globals.contains(&id) {
            self.globals.push(id);
        }
    }

    /// Builder-style form of [`Self::preserve_global`], for concise pass
    /// preservation declarations.
    pub fn preserving_global<A: GlobalAnalysis>(mut self) -> Self {
        self.preserve_global::<A>();
        self
    }

    pub fn preserve_local<A: LocalAnalysis>(&mut self) {
        let id = TypeId::of::<A>();
        if !self.locals.contains(&id) {
            self.locals.push(id);
        }
    }

    /// Builder-style form of [`Self::preserve_local`], for concise pass
    /// preservation declarations.
    pub fn preserving_local<A: LocalAnalysis>(mut self) -> Self {
        self.preserve_local::<A>();
        self
    }

    fn preserves_global(&self, id: TypeId) -> bool {
        self.all_global || self.globals.contains(&id)
    }

    fn preserves_local(&self, id: TypeId) -> bool {
        self.all_local || self.locals.contains(&id)
    }

    /// Whether global analysis `A` was reported preserved.
    pub fn preserves_global_analysis<A: GlobalAnalysis>(&self) -> bool {
        self.preserves_global(TypeId::of::<A>())
    }

    /// Whether local analysis `A` was reported preserved.
    pub fn preserves_local_analysis<A: LocalAnalysis>(&self) -> bool {
        self.preserves_local(TypeId::of::<A>())
    }

    pub(crate) fn intersect(&mut self, other: &Self) {
        if self.all_global {
            self.all_global = other.all_global;
            self.globals = other.globals.clone();
        } else if !other.all_global {
            self.globals.retain(|id| other.globals.contains(id));
        }
        if self.all_local {
            self.all_local = other.all_local;
            self.locals = other.locals.clone();
        } else if !other.all_local {
            self.locals.retain(|id| other.locals.contains(id));
        }
    }
}

impl Default for PreservedAnalyses {
    fn default() -> Self {
        Self::none()
    }
}

/// Per-function analysis cache. Instances are moved with their function body
/// into parallel workers, so no locking or cross-function invalidation exists.
#[derive(Default)]
pub struct LocalAnalysisManager {
    cache: HashMap<TypeId, Box<dyn Any + Send>>,
}

impl LocalAnalysisManager {
    pub fn get<A: LocalAnalysis>(
        &mut self,
        body: &FunctionBody<'_>,
        cx: ContextView<'_, '_>,
    ) -> &A::Result {
        let value = self
            .cache
            .entry(TypeId::of::<A>())
            .or_insert_with(|| Box::new(A::analyze(body, cx)));
        value
            .downcast_ref::<A::Result>()
            .expect("local analysis marker returned inconsistent result type")
    }

    pub(crate) fn invalidate(&mut self, preserved: &PreservedAnalyses) {
        self.cache.retain(|id, _| preserved.preserves_local(*id));
    }
}

/// Pipeline-owned analysis caches. Nothing here is stored in or serialized with
/// core IR; a fresh manager is created for each pipeline execution boundary.
#[derive(Default)]
pub struct AnalysisManager {
    globals: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
    locals: HashMap<FunctionId, LocalAnalysisManager>,
}

impl AnalysisManager {
    pub fn global<A: GlobalAnalysis>(&mut self, ctx: &Context<'_>) -> &A::Result {
        let value = self
            .globals
            .entry(TypeId::of::<A>())
            .or_insert_with(|| Box::new(A::analyze(ctx)));
        value
            .downcast_ref::<A::Result>()
            .expect("global analysis marker returned inconsistent result type")
    }

    /// Remove a global analysis result from the manager, computing it first if
    /// necessary. Module passes that maintain derived state alongside mutations
    /// can own the result temporarily and return it with [`Self::put_global`].
    pub fn take_global<A: GlobalAnalysis>(&mut self, ctx: &Context<'_>) -> A::Result {
        let value = self
            .globals
            .remove(&TypeId::of::<A>())
            .unwrap_or_else(|| Box::new(A::analyze(ctx)));
        *value
            .downcast::<A::Result>()
            .expect("global analysis marker returned inconsistent result type")
    }

    /// Return a global analysis result previously removed with
    /// [`Self::take_global`].
    pub fn put_global<A: GlobalAnalysis>(&mut self, result: A::Result) {
        let previous = self.globals.insert(TypeId::of::<A>(), Box::new(result));
        assert!(
            previous.is_none(),
            "put_global called while the analysis is already cached"
        );
    }

    pub(crate) fn take_local(&mut self, function: FunctionId) -> LocalAnalysisManager {
        self.locals.remove(&function).unwrap_or_default()
    }

    pub(crate) fn put_local(&mut self, function: FunctionId, analyses: LocalAnalysisManager) {
        self.locals.insert(function, analyses);
    }

    pub(crate) fn invalidate_globals(&mut self, preserved: &PreservedAnalyses) {
        self.globals.retain(|id, _| preserved.preserves_global(*id));
    }

    pub(crate) fn invalidate_local(&mut self, function: FunctionId, preserved: &PreservedAnalyses) {
        if let Some(analyses) = self.locals.get_mut(&function) {
            analyses.invalidate(preserved);
        }
    }

    pub(crate) fn invalidate_all_locals(&mut self, preserved: &PreservedAnalyses) {
        for analyses in self.locals.values_mut() {
            analyses.invalidate(preserved);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use qcode::context::Context;

    use super::*;
    use crate::{ContextSplit, PipelineEnv};

    static GLOBAL_BUILDS: AtomicUsize = AtomicUsize::new(0);
    static LOCAL_BUILDS: AtomicUsize = AtomicUsize::new(0);
    static LOCAL_PRESERVATION_BUILDS: AtomicUsize = AtomicUsize::new(0);

    struct CountingGlobal;

    impl GlobalAnalysis for CountingGlobal {
        type Result = usize;

        fn analyze(_ctx: &Context<'_>) -> Self::Result {
            GLOBAL_BUILDS.fetch_add(1, Ordering::SeqCst) + 1
        }
    }

    struct CountingLocal;

    impl LocalAnalysis for CountingLocal {
        type Result = usize;

        fn analyze<'str>(_body: &FunctionBody<'str>, _cx: ContextView<'_, 'str>) -> Self::Result {
            LOCAL_BUILDS.fetch_add(1, Ordering::SeqCst) + 1
        }
    }

    struct CountingPreservedLocal;

    impl LocalAnalysis for CountingPreservedLocal {
        type Result = usize;

        fn analyze<'str>(_body: &FunctionBody<'str>, _cx: ContextView<'_, 'str>) -> Self::Result {
            LOCAL_PRESERVATION_BUILDS.fetch_add(1, Ordering::SeqCst) + 1
        }
    }

    #[test]
    fn global_cache_obeys_preservation() {
        GLOBAL_BUILDS.store(0, Ordering::SeqCst);
        let ctx = Context::new();
        let mut analyses = AnalysisManager::default();

        assert_eq!(*analyses.global::<CountingGlobal>(&ctx), 1);
        assert_eq!(*analyses.global::<CountingGlobal>(&ctx), 1);

        let mut preserved = PreservedAnalyses::none();
        preserved.preserve_global::<CountingGlobal>();
        analyses.invalidate_globals(&preserved);
        assert_eq!(*analyses.global::<CountingGlobal>(&ctx), 1);

        analyses.invalidate_globals(&PreservedAnalyses::none());
        assert_eq!(*analyses.global::<CountingGlobal>(&ctx), 2);
    }

    #[test]
    fn global_analysis_can_be_taken_updated_and_returned() {
        GLOBAL_BUILDS.store(0, Ordering::SeqCst);
        let ctx = Context::new();
        let mut analyses = AnalysisManager::default();

        let mut value = analyses.take_global::<CountingGlobal>(&ctx);
        assert_eq!(value, 1);
        value = 42;
        analyses.put_global::<CountingGlobal>(value);

        assert_eq!(*analyses.global::<CountingGlobal>(&ctx), 42);
        assert_eq!(GLOBAL_BUILDS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn local_invalidation_is_scoped_to_one_function() {
        LOCAL_BUILDS.store(0, Ordering::SeqCst);
        let mut ctx = Context::new();
        FunctionBody::make(&mut ctx, "a".into()).unwrap();
        FunctionBody::make(&mut ctx, "b".into()).unwrap();
        let ids = ctx.function_ids();
        let (a, b) = (ids[0], ids[1]);
        let env = PipelineEnv::headless(&ctx);
        let mut analyses = AnalysisManager::default();

        for function in [a, b] {
            let mut local = analyses.take_local(function);
            {
                let (bodies, view) = ctx.split(&env);
                let _ = local.get::<CountingLocal>(&bodies[function], view);
            }
            analyses.put_local(function, local);
        }
        assert_eq!(LOCAL_BUILDS.load(Ordering::SeqCst), 2);

        analyses.invalidate_local(a, &PreservedAnalyses::none());
        for function in [b, a] {
            let mut local = analyses.take_local(function);
            {
                let (bodies, view) = ctx.split(&env);
                let _ = local.get::<CountingLocal>(&bodies[function], view);
            }
            analyses.put_local(function, local);
        }
        assert_eq!(LOCAL_BUILDS.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn local_cache_obeys_positive_and_negative_preservation() {
        LOCAL_PRESERVATION_BUILDS.store(0, Ordering::SeqCst);
        let mut ctx = Context::new();
        let function = FunctionBody::make(&mut ctx, "f".into()).unwrap().id;
        let env = PipelineEnv::headless(&ctx);
        let mut local = LocalAnalysisManager::default();

        {
            let (bodies, view) = ctx.split(&env);
            assert_eq!(
                *local.get::<CountingPreservedLocal>(&bodies[function], view),
                1
            );
        }

        let preserved = PreservedAnalyses::none().preserving_local::<CountingPreservedLocal>();
        local.invalidate(&preserved);
        {
            let (bodies, view) = ctx.split(&env);
            assert_eq!(
                *local.get::<CountingPreservedLocal>(&bodies[function], view),
                1
            );
        }

        local.invalidate(&PreservedAnalyses::none());
        {
            let (bodies, view) = ctx.split(&env);
            assert_eq!(
                *local.get::<CountingPreservedLocal>(&bodies[function], view),
                2
            );
        }
    }
}
