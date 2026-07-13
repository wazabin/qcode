// TODO: maybe try to reuse some of num_ref's code here?

use crate::{
    context::{Context, Shared},
    value::{
        ValueId,
        block::{BasicBlock, BlockId, EdgeData, EdgeId},
        block_param::{BlockParam, BlockParamId},
        function::{FunctionBody, FunctionId},
        insn::{Instruction, InstructionId},
    },
};

/// A `Copy` read view of the module's IR, routing composite-id reads to the
/// arena that owns them (Stage 5.3a of the parallel-function-passes plan; see
/// `PARALLEL_PASSES.md`).
///
/// Immutable `*Ref`s over the per-function arena cluster (function, block,
/// instruction, block-param, edge) carry a `HostRef` instead of a bare
/// `&Context`, so the same ref code reads correctly whether the target function
/// lives in the module registry ([`Module`](Self::Module)) or has been *checked
/// out* by a pass ([`Checked`](Self::Checked)). Because a checked-out function's
/// body slot holds an empty body, reads of its arenas must come from the owned
/// `&FunctionBody`, not from `shared`.
///
/// It is `Copy` (it holds only shared references), which is what lets a read ref
/// hand the same host to every sub-ref it constructs.
#[derive(Clone, Copy)]
pub enum HostRef<'a, 'str> {
    /// The whole module; every function is read from its registry slot. The only
    /// variant used outside a checked-out pass, and the one the existing suite
    /// exercises — its routing is identical to the pre-`HostRef` `&Context` path.
    Module(&'a Context<'str>),
    /// A function checked out of the module for exclusive access: reads of `id`'s
    /// arenas come from `fun`; shared data from `shared`; interface queries from
    /// `interfaces`. Carries **no** `&Context` — in particular no reach into
    /// other functions' bodies, which is the parallel-safety contract
    /// (context-split stage 5b-ii Pin B).
    Checked {
        fun: &'a FunctionBody<'str>,
        shared: &'a crate::context::Shared<'str>,
        interfaces: &'a jstd::registry::Registry<
            FunctionId,
            crate::value::function::FunctionInterface<'str>,
        >,
    },
}

impl<'a, 'str> HostRef<'a, 'str> {
    /// The whole `&Context`, for **module-scope-only** reads (cross-function
    /// body walks such as `FunctionRef::callees`/`callers`). Panics on a
    /// checked-out host, which by design carries no `&Context`.
    pub fn module_ctx(self) -> &'a Context<'str> {
        match self {
            HostRef::Module(c) => c,
            HostRef::Checked { .. } => {
                panic!("whole-context read on a checked-out host: this query is module-scope only")
            }
        }
    }

    /// The module's shared IR state ([`Shared`]): varnodes, literals, types,
    /// spaces, registers, name/address maps. Both variants read these from the
    /// module — a checked-out function's *arenas* live in `fun`, but everything
    /// else stays shared.
    ///
    /// [`Shared`]: crate::context::Shared
    pub fn shr(self) -> &'a crate::context::Shared<'str> {
        match self {
            HostRef::Module(c) => &c.shared,
            HostRef::Checked { shared, .. } => shared,
        }
    }

    /// The function *body* `f`: the owned `fun` on a checked-out host, the
    /// registry slot on a module host. A checked-out host holds only its own
    /// body — a foreign-body read through it is the exact coupling the
    /// context-split forbids, and panics.
    pub fn function(self, f: FunctionId) -> &'a FunctionBody<'str> {
        match self {
            HostRef::Module(c) => &c.bodies[f],
            HostRef::Checked { fun, .. } if f == fun.id() => fun,
            HostRef::Checked { .. } => panic!(
                "foreign function-body read on a checked-out host: other bodies are module-scope only"
            ),
        }
    }

    /// The function *interface* of `f`. Interfaces are never checked out, so this
    /// always reads the shared registry — the real interface even for a
    /// co-checked-out function.
    pub fn interface(self, f: FunctionId) -> &'a crate::value::function::FunctionInterface<'str> {
        match self {
            HostRef::Module(c) => &c.interfaces[f],
            HostRef::Checked { interfaces, .. } => &interfaces[f],
        }
    }

    /// The instruction `id`, routed to its owning function's arena.
    pub fn instruction(self, id: InstructionId) -> &'a Instruction<'str> {
        &self.function(id.func).insns[id.local]
    }

    /// Whether `id` currently names a live instruction payload.
    pub fn contains_instruction(self, id: InstructionId) -> bool {
        self.function(id.func).insns.contains(id.local)
    }

    /// The block `id`, routed to its owning function's arena.
    pub fn block(self, id: BlockId) -> &'a BasicBlock<'str> {
        &self.function(id.func).blocks[id.local]
    }

    /// Whether `id` currently names a live block payload.
    pub fn contains_block(self, id: BlockId) -> bool {
        self.function(id.func).blocks.contains(id.local)
    }

    /// The block parameter `id`, routed to its owning function's arena.
    pub fn block_param(self, id: BlockParamId) -> &'a BlockParam<'str> {
        &self.function(id.func).params[id.local]
    }

    /// Whether `id` currently names a live block-parameter payload.
    pub fn contains_block_param(self, id: BlockParamId) -> bool {
        self.function(id.func).params.contains(id.local)
    }

    /// The CFG edge `id`, stored in function `func`'s edge arena.
    pub fn edge(self, func: FunctionId, id: EdgeId) -> &'a EdgeData {
        &self.function(func).edges[id]
    }

    /// The [`TypeId`] of `id`, routing arena-cluster values (instruction,
    /// block-param) to their owning function and reading shared leaves (literal,
    /// bytes, varnode) from the module. The [`HostRef`] mirror of
    /// [`Context::type_of`], so a checked-out pass resolves its own SSA values'
    /// types correctly.
    ///
    /// [`Context::type_of`]: crate::context::Context::type_of
    pub fn type_of(self, id: ValueId) -> crate::types::TypeId {
        let shared = self.shr();
        match id {
            ValueId::Literal(lid) => shared.values.literals[lid].type_id,
            ValueId::Bytes(bid) => shared.values.bytes[bid].type_id,
            ValueId::Instruction(iid) => self.instruction(iid).type_id,
            ValueId::BlockParam(pid) => self.block_param(pid).type_id,
            ValueId::Varnode(vid) => {
                if let Some(&ty) = shared.values.varnode_types.get(&vid) {
                    return ty;
                }
                shared
                    .types
                    .get_or_make_int(shared.values.varnodes[vid].size_bytes())
            }
            ValueId::BasicBlock(_) | ValueId::Function(_) => shared.types.get_or_make_int(0),
        }
    }

    /// The stored [`TypeId`] of `id` (routed like [`type_of`](Self::type_of)),
    /// never interning a fallback integer type. The [`HostRef`] mirror of
    /// [`Context::stored_type_of`]; varnodes without an override, blocks, and
    /// functions return `None`.
    ///
    /// [`Context::stored_type_of`]: crate::context::Context::stored_type_of
    pub fn stored_type_of(self, id: ValueId) -> Option<crate::types::TypeId> {
        let shared = self.shr();
        match id {
            ValueId::Literal(lid) => Some(shared.values.literals[lid].type_id),
            ValueId::Bytes(bid) => Some(shared.values.bytes[bid].type_id),
            ValueId::Instruction(iid) => Some(self.instruction(iid).type_id),
            ValueId::BlockParam(pid) => Some(self.block_param(pid).type_id),
            ValueId::Varnode(vid) => shared.values.varnode_types.get(&vid).copied(),
            ValueId::BasicBlock(_) | ValueId::Function(_) => None,
        }
    }
}

/// Function-scoped wrapper-ref constructors (context-split stage 5a).
///
/// These are the single convention for reading IR under the context-split
/// migration: at a call site they replace the *globally routed*
/// `BasicBlock::from_id(ctx, id)` / `BlockRef::new(host, id)` spellings with one
/// body-routed accessor, `view.block(id)` (and siblings), that returns the same
/// wrapper ref the rest of the code already speaks. The wrapper is unchanged —
/// only the routing behind it (module registry vs. checked-out body) is now
/// explicit in *where the `HostRef` came from*.
///
/// The call form is deliberately stable across the stage-5b split: after bodies
/// are held directly, the identical `.block(id)` / `.insn(id)` methods move onto
/// the body-view type, and only *how the receiver is obtained* changes
/// (`host.read_host()` → `&FunctionBody` + `ContextView`). Converted sites do not
/// otherwise move.
impl<'a, 'str> HostRef<'a, 'str> {
    /// A [`BlockRef`](crate::value::BlockRef) over `id`, routed to its owning
    /// function's arena through this view. Replaces `BasicBlock::from_id(ctx, id)`.
    pub fn block_ref(self, id: BlockId) -> BaseRef<HostRef<'a, 'str>, BlockId> {
        let _ = self.block(id);
        BaseRef::new(self, id)
    }

    /// An [`InstructionRef`](crate::value::InstructionRef) over `id`, body-routed.
    /// Replaces `Instruction::from_id(ctx, id)`.
    pub fn insn_ref(self, id: InstructionId) -> BaseRef<HostRef<'a, 'str>, InstructionId> {
        let _ = self.instruction(id);
        BaseRef::new(self, id)
    }

    /// A [`BlockParamRef`](crate::value::BlockParamRef) over `id`, body-routed.
    /// Replaces `BlockParam::from_id(ctx, id)`.
    pub fn param_ref(self, id: BlockParamId) -> BaseRef<HostRef<'a, 'str>, BlockParamId> {
        let _ = self.block_param(id);
        BaseRef::new(self, id)
    }

    /// A [`FunctionRef`](crate::value::FunctionRef) over `id`, body-routed.
    /// Replaces `FunctionBody::from_id(ctx, id)`.
    pub fn function_ref(self, id: FunctionId) -> BaseRef<HostRef<'a, 'str>, FunctionId> {
        BaseRef::new(self, id)
    }
}

impl<'a, 'str> From<&'a Context<'str>> for HostRef<'a, 'str> {
    fn from(ctx: &'a Context<'str>) -> Self {
        HostRef::Module(ctx)
    }
}

impl<'a, 'str> From<&'a mut Context<'str>> for HostRef<'a, 'str> {
    fn from(ctx: &'a mut Context<'str>) -> Self {
        // A read host only needs shared access; downgrade the exclusive borrow.
        HostRef::Module(ctx)
    }
}

impl<'a, 'str, Id: Copy> BaseRef<HostRef<'a, 'str>, Id> {
    /// Construct a [`HostRef`]-backed read ref (the arena-cluster refs) over the
    /// whole module. Takes a concrete `&Context` (a `&mut Context` auto-reborrows)
    /// so callers need no explicit downgrade; construct a *checked-out* ref with
    /// [`BaseRef::new`] over a [`HostRef::Checked`] instead.
    pub fn from_id(ctx: &'a Context<'str>, id: Id) -> Self {
        BaseRef::new(HostRef::Module(ctx), id)
    }
}

/// A generic wrapper struct for referencing an ID with a context.
/// This is the basis for all `*Ref` and `*MutRef`.
///
/// `Id` is any cheap `Copy` handle — either a `Registry` `Identifier` (varnode,
/// literal, function, …) or a composite IR id (`InstructionId`, `BlockId`, …).
/// The bound is intentionally just `Copy`; routing to storage is the concern of
/// the concrete ref impls, not this wrapper.
pub struct BaseRef<Ctx, Id: Copy> {
    pub id: Id,
    pub(in crate::value) ctx: Ctx,
}

impl<Ctx, Id: Copy> BaseRef<Ctx, Id> {
    pub fn new(ctx: Ctx, id: Id) -> Self {
        BaseRef { id, ctx }
    }

    /// The underlying backing handle (read). Crate-internal: the `Builder`
    /// (a sibling module) reaches its backing through this.
    pub(crate) fn host_ref(&self) -> &Ctx {
        &self.ctx
    }

    /// The underlying host/context handle (write).
    pub(crate) fn host_mut(&mut self) -> &mut Ctx {
        &mut self.ctx
    }
}

impl<Ctx, Id: Copy + Into<ValueId>> BaseRef<Ctx, Id> {
    pub fn id(&self) -> ValueId {
        self.id.into()
    }
}

impl<'str, 'ctx, Id: Copy> BaseRef<&'ctx Context<'str>, Id> {
    pub fn from_id(ctx: &'ctx Context<'str>, id: Id) -> Self {
        BaseRef { id, ctx }
    }
}

/// Construct a shared-leaf wrapper ref (literal, bytes, varnode) over the
/// module's [`Shared`] IR state, taking **either** a `&Context` or a `&Shared`
/// (via [`AsShared`]) so module- and pass-scope callers share one spelling
/// (context-split stage 5b-ii item #1).
impl<'a, 'str, Id: Copy> BaseRef<&'a Shared<'str>, Id> {
    pub fn from_id(src: impl AsShared<'a, 'str>, id: Id) -> Self {
        BaseRef {
            id,
            ctx: src.as_shared(),
        }
    }
}

impl<'str, 'ctx, Id: Copy> BaseRef<&'ctx mut Context<'str>, Id> {
    pub fn from_id(ctx: &'ctx mut Context<'str>, id: Id) -> Self {
        BaseRef { id, ctx }
    }
}

/// Convert mutable refs to immutable refs by cloning the ID and sharing the context reference.
impl<'str, 'ctx, Id: Copy> From<BaseRef<&'ctx mut Context<'str>, Id>>
    for BaseRef<&'ctx Context<'str>, Id>
{
    fn from(r: BaseRef<&'ctx mut Context<'str>, Id>) -> Self {
        BaseRef {
            id: r.id,
            ctx: r.ctx,
        }
    }
}

pub trait WithCtx<'s, 'ctx: 's, 'str: 'ctx> {
    fn ctx(&'s self) -> &'ctx Context<'str>;
}

/// Narrows a module handle to its [`Shared`] IR state, so the shared-leaf
/// wrapper-ref constructors (`Varnode::from_id`, `Space::from_id`,
/// `LiteralRef`/`BytesRef`) accept **either** a whole `&Context` (module scope)
/// **or** a bare `&Shared` (pass scope, via `HostRef::shr()` /
/// `PassBacking::shr()`) with no call-site churn (context-split stage 5b-ii item
/// #1). The leaf refs themselves store only a `&Shared`.
pub trait AsShared<'a, 'str> {
    // Implemented only for `Copy` reference types (`&Context`, `&Shared`), so
    // taking `self` by value is a cheap pointer copy, not an owning move.
    #[allow(clippy::wrong_self_convention)]
    fn as_shared(self) -> &'a Shared<'str>;
}

impl<'a, 'str> AsShared<'a, 'str> for &'a Shared<'str> {
    fn as_shared(self) -> &'a Shared<'str> {
        self
    }
}

impl<'a, 'str> AsShared<'a, 'str> for &'a Context<'str> {
    fn as_shared(self) -> &'a Shared<'str> {
        &self.shared
    }
}

/// The shared-leaf twin of [`WithCtx`]: a ref that can yield the module's
/// [`Shared`] IR state (interners, spaces, registers, name map) for its read
/// methods. Implemented by the shared-leaf wrapper refs (literal, bytes,
/// varnode) — which now carry only a `&Shared` — and by their `&mut Context`
/// mutation twins (which narrow through `self.ctx.shared`). This replaces the
/// leaf refs' old `WithCtx` dependency: a shared-leaf value never needs the
/// whole `&Context`, only its shared part (context-split stage 5b-ii item #1).
pub trait WithShared<'s, 'sh: 's, 'str: 'sh> {
    fn shared(&'s self) -> &'sh Shared<'str>;
}

pub trait WithCtxMut<'s, 'str: 's>: WithCtx<'s, 's, 'str> {
    fn ctx_mut(&'s mut self) -> &'s mut Context<'str>;
}

/// A ref that can yield a [`HostRef`] for routing arena reads. Implemented by the
/// per-function arena-cluster refs (function/block/instruction/block-param/edge),
/// both immutable (host lifetime is the independent `'ctx`, since the ref stores a
/// `Copy` `HostRef`) and mutable (host lifetime collapses to the `&self` borrow
/// `'s`, mirroring how the mut refs implement [`WithCtx<'s, 's, 'str>`]).
///
/// Read-ref method bodies route arena access through `self.host()` so they read
/// the owned function when it is checked out; shared reads still go through
/// [`WithCtx::ctx`] (== `self.host().shared()`).
pub trait WithHost<'s, 'ctx: 's, 'str: 'ctx>: WithCtx<'s, 'ctx, 'str> {
    fn host(&'s self) -> HostRef<'ctx, 'str>;
}
