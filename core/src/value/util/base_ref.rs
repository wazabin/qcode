// TODO: maybe try to reuse some of num_ref's code here?

use crate::{
    context::Context,
    value::{
        ValueId,
        block::{BasicBlock, BlockId, EdgeData, EdgeId},
        block_param::{BlockParam, BlockParamId},
        function::{Function, FunctionId},
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
/// registry slot holds a [`sentinel`](Function::sentinel), reads of its arenas
/// must come from the owned `&Function`, not from `shared`.
///
/// It is `Copy` (it holds only shared references), which is what lets a read ref
/// hand the same host to every sub-ref it constructs.
#[derive(Clone, Copy)]
pub enum HostRef<'a, 'str> {
    /// The whole module; every function is read from its registry slot. The only
    /// variant used outside a checked-out pass, and the one the existing suite
    /// exercises — its routing is identical to the pre-`HostRef` `&Context` path.
    Module(&'a Context<'str>),
    /// A function checked out of `shared` for exclusive access: reads of `id`'s
    /// arenas come from `fun`; every other function (and all shared data) from
    /// `shared`.
    Checked {
        fun: &'a Function<'str>,
        shared: &'a Context<'str>,
        id: FunctionId,
    },
}

impl<'a, 'str> HostRef<'a, 'str> {
    /// The module's shared data (varnodes, literals, types, spaces, registers,
    /// name/address maps). Both variants read these from the module — a
    /// checked-out function's *arenas* live in `fun`, but everything else stays
    /// in the shared context.
    pub fn shared(self) -> &'a Context<'str> {
        match self {
            HostRef::Module(c) => c,
            HostRef::Checked { shared, .. } => shared,
        }
    }

    /// The function `f`, from `fun` if it is the checked-out one, else from the
    /// shared registry.
    pub fn function(self, f: FunctionId) -> &'a Function<'str> {
        match self {
            HostRef::Module(c) => &c.values.functions[f],
            HostRef::Checked { fun, id, .. } if f == id => fun,
            HostRef::Checked { shared, .. } => &shared.values.functions[f],
        }
    }

    /// The instruction `id`, routed to its owning function's arena.
    pub fn instruction(self, id: InstructionId) -> &'a Instruction<'str> {
        &self.function(id.func).insns[id.local]
    }

    /// The block `id`, routed to its owning function's arena.
    pub fn block(self, id: BlockId) -> &'a BasicBlock<'str> {
        &self.function(id.func).blocks[id.local]
    }

    /// The block parameter `id`, routed to its owning function's arena.
    pub fn block_param(self, id: BlockParamId) -> &'a BlockParam<'str> {
        &self.function(id.func).params[id.local]
    }

    /// The CFG edge `id`, routed to its owning function's arena.
    pub fn edge(self, id: EdgeId) -> &'a EdgeData {
        &self.function(id.func).edges[id.local]
    }

    /// The [`TypeId`] of `id`, routing arena-cluster values (instruction,
    /// block-param) to their owning function and reading shared leaves (literal,
    /// bytes, varnode) from the module. The [`HostRef`] mirror of
    /// [`Context::type_of`], so a checked-out pass resolves its own SSA values'
    /// types correctly.
    ///
    /// [`Context::type_of`]: crate::context::Context::type_of
    pub fn type_of(self, id: ValueId) -> crate::types::TypeId {
        let shared = self.shared();
        match id {
            ValueId::Literal(lid) => shared.values.literals[lid].type_id,
            ValueId::Bytes(bid) => shared.values.bytes[bid].type_id,
            ValueId::Instruction(iid) => self.instruction(iid).type_id,
            ValueId::BlockParam(pid) => self.block_param(pid).type_id,
            ValueId::Varnode(vid) => {
                if let Some(&ty) = shared.values.varnode_types.get(&vid) {
                    return ty;
                }
                shared.types.get_or_make_int(shared.values.varnodes[vid].size_bytes())
            }
            ValueId::BasicBlock(_) | ValueId::Function(_) => shared.types.get_or_make_int(0),
        }
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

    /// The underlying host/context handle (read). Crate-internal: the `Builder`
    /// (a sibling module) reaches the `HostMut` through this.
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
