//! Static immutable providers for qcode IR reads.
//!
//! [`ModuleView`] resolves any function body in an unchanged [`Context`].
//! [`BodyView`] resolves exactly one body plus shared data and published
//! interfaces. Both implement [`QCodeView`], which is deliberately read-only.

use jstd::registry::Registry;

use crate::{
    context::{Context, Shared},
    types::TypeId,
    value::{
        BasicBlock, BlockId, BlockParamRef, BlockRef, FunctionBody, FunctionId, FunctionRef,
        Instruction, InstructionRef, ValueId,
        block::{EdgeData, EdgeId},
        block_param::{BlockParam, BlockParamId},
        function::FunctionInterface,
        insn::InstructionId,
    },
};

/// Read-only resolution capability shared by module and single-body views.
///
/// The explicit lifetimes let provider-generic refs return data with the
/// provider's underlying borrow lifetime, rather than tying results to a short
/// borrow of the thin provider value.
pub trait QCodeView<'ctx, 'str>: Copy
where
    'str: 'ctx,
{
    fn shared(self) -> &'ctx Shared<'str>;
    fn interface(self, id: FunctionId) -> &'ctx FunctionInterface<'str>;
    fn function(self, id: FunctionId) -> &'ctx FunctionBody<'str>;

    fn instruction(self, id: InstructionId) -> &'ctx Instruction<'str> {
        &self.function(id.func).insns[id.local]
    }

    fn contains_instruction(self, id: InstructionId) -> bool {
        self.function(id.func).insns.contains(id.local)
    }

    fn block(self, id: BlockId) -> &'ctx BasicBlock<'str> {
        &self.function(id.func).blocks[id.local]
    }

    fn contains_block(self, id: BlockId) -> bool {
        self.function(id.func).blocks.contains(id.local)
    }

    fn block_param(self, id: BlockParamId) -> &'ctx BlockParam<'str> {
        &self.function(id.func).params[id.local]
    }

    fn contains_block_param(self, id: BlockParamId) -> bool {
        self.function(id.func).params.contains(id.local)
    }

    fn edge(self, function: FunctionId, id: EdgeId) -> &'ctx EdgeData {
        &self.function(function).edges[id]
    }

    fn type_of(self, id: ValueId) -> TypeId {
        let shared = self.shared();
        match id {
            ValueId::Literal(id) => shared.values.literals[id].type_id,
            ValueId::Bytes(id) => shared.values.bytes[id].type_id,
            ValueId::Instruction(id) => self.instruction(id).type_id,
            ValueId::BlockParam(id) => self.block_param(id).type_id,
            ValueId::Varnode(id) => shared
                .values
                .varnode_types
                .get(&id)
                .copied()
                .unwrap_or_else(|| {
                    shared
                        .types
                        .get_or_make_int(shared.values.varnodes[id].size_bytes())
                }),
            ValueId::BasicBlock(_) | ValueId::Function(_) => shared.types.get_or_make_int(0),
        }
    }

    fn stored_type_of(self, id: ValueId) -> Option<TypeId> {
        let shared = self.shared();
        match id {
            ValueId::Literal(id) => Some(shared.values.literals[id].type_id),
            ValueId::Bytes(id) => Some(shared.values.bytes[id].type_id),
            ValueId::Instruction(id) => Some(self.instruction(id).type_id),
            ValueId::BlockParam(id) => Some(self.block_param(id).type_id),
            ValueId::Varnode(id) => shared.values.varnode_types.get(&id).copied(),
            ValueId::BasicBlock(_) | ValueId::Function(_) => None,
        }
    }

    fn block_ref(self, id: BlockId) -> BlockRef<'str, 'ctx, Self>
    where
        Self: Sized,
    {
        let _ = self.block(id);
        BlockRef::new(self, id)
    }

    fn insn_ref(self, id: InstructionId) -> InstructionRef<'str, 'ctx, Self>
    where
        Self: Sized,
    {
        let _ = self.instruction(id);
        InstructionRef::new(self, id)
    }

    fn param_ref(self, id: BlockParamId) -> BlockParamRef<'str, 'ctx, Self>
    where
        Self: Sized,
    {
        let _ = self.block_param(id);
        BlockParamRef::new(self, id)
    }

    fn function_ref(self, id: FunctionId) -> FunctionRef<'str, 'ctx, Self>
    where
        Self: Sized,
    {
        let _ = self.function(id);
        FunctionRef::new(self, id)
    }
}

/// Whole-module immutable provider.
#[derive(Clone, Copy)]
pub struct ModuleView<'ctx, 'str> {
    context: &'ctx Context<'str>,
}

impl<'ctx, 'str> ModuleView<'ctx, 'str> {
    pub fn new(context: &'ctx Context<'str>) -> Self {
        Self { context }
    }

    /// Whole-context access is intentionally module-only and absent from
    /// [`QCodeView`] / [`BodyView`].
    pub fn context(self) -> &'ctx Context<'str> {
        self.context
    }
}

impl<'ctx, 'str: 'ctx> QCodeView<'ctx, 'str> for ModuleView<'ctx, 'str> {
    fn shared(self) -> &'ctx Shared<'str> {
        &self.context.shared
    }

    fn interface(self, id: FunctionId) -> &'ctx FunctionInterface<'str> {
        &self.context.interfaces[id]
    }

    fn function(self, id: FunctionId) -> &'ctx FunctionBody<'str> {
        &self.context.bodies[id]
    }
}

/// Single-body immutable provider used by function passes.
#[derive(Clone, Copy)]
pub struct BodyView<'ctx, 'str> {
    body: &'ctx FunctionBody<'str>,
    shared: &'ctx Shared<'str>,
    interfaces: &'ctx Registry<FunctionId, FunctionInterface<'str>>,
}

impl<'ctx, 'str> BodyView<'ctx, 'str> {
    pub fn new(
        body: &'ctx FunctionBody<'str>,
        shared: &'ctx Shared<'str>,
        interfaces: &'ctx Registry<FunctionId, FunctionInterface<'str>>,
    ) -> Self {
        let id = body.id();
        assert!(
            body.roster
                .iter()
                .all(|&local| body.blocks[local].parent == Some(id)),
            "BodyView requires a function with no reattributed blocks"
        );
        Self {
            body,
            shared,
            interfaces,
        }
    }

    pub fn function_id(self) -> FunctionId {
        self.body.id()
    }
}

impl<'ctx, 'str: 'ctx> QCodeView<'ctx, 'str> for BodyView<'ctx, 'str> {
    fn shared(self) -> &'ctx Shared<'str> {
        self.shared
    }

    fn interface(self, id: FunctionId) -> &'ctx FunctionInterface<'str> {
        &self.interfaces[id]
    }

    fn function(self, id: FunctionId) -> &'ctx FunctionBody<'str> {
        assert_eq!(
            id,
            self.body.id(),
            "BodyView cannot read a foreign function body"
        );
        self.body
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        builder::Builder,
        context::Context,
        value::{BasicBlock, FunctionBody, ValueId, ValueRef},
    };

    use super::*;

    #[test]
    fn module_and_body_views_resolve_identical_local_data() {
        let mut ctx = Context::new();
        let function = FunctionBody::make(&mut ctx, "f".into()).unwrap().id;
        let block = BasicBlock::make(&mut ctx, function).id;
        let value = ctx.get_const(7, 8).id();
        let insn = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, block))
            .push_return(value)
            .id;

        let module = ModuleView::new(&ctx);
        let body = BodyView::new(&ctx.bodies[function], &ctx.shared, &ctx.interfaces);

        assert_eq!(module.block(block).address, body.block(block).address);
        assert!(std::ptr::eq(
            module.instruction(insn),
            body.instruction(insn)
        ));
        assert_eq!(
            module.type_of(ValueId::Instruction(insn)),
            body.type_of(ValueId::Instruction(insn))
        );
        assert_eq!(module.insn_ref(insn).id, body.insn_ref(insn).id);
        assert_eq!(
            module.insn_ref(insn).as_statement().to_string(),
            body.insn_ref(insn).as_statement().to_string()
        );
        assert_eq!(
            module
                .function_ref(function)
                .iter()
                .map(|block| block.id)
                .collect::<Vec<_>>(),
            body.function_ref(function)
                .iter()
                .map(|block| block.id)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            ValueRef::from_view(module, ValueId::Instruction(insn)).to_string(),
            ValueRef::from_view(body, ValueId::Instruction(insn)).to_string()
        );
    }

    #[test]
    fn body_view_reads_foreign_interfaces_but_not_foreign_bodies() {
        let mut ctx = Context::new();
        let own = FunctionBody::make(&mut ctx, "own".into()).unwrap().id;
        let foreign = FunctionBody::make(&mut ctx, "foreign".into()).unwrap().id;
        let view = BodyView::new(&ctx.bodies[own], &ctx.shared, &ctx.interfaces);

        assert_eq!(view.interface(foreign).name.as_ref(), "foreign");
        assert!(std::panic::catch_unwind(|| view.function(foreign)).is_err());
    }

    #[test]
    fn body_view_rejects_foreign_composite_ids() {
        let mut ctx = Context::new();
        let own = FunctionBody::make(&mut ctx, "own".into()).unwrap().id;
        let foreign = FunctionBody::make(&mut ctx, "foreign".into()).unwrap().id;
        let block = BasicBlock::make(&mut ctx, foreign).id;
        let view = BodyView::new(&ctx.bodies[own], &ctx.shared, &ctx.interfaces);

        assert!(std::panic::catch_unwind(|| view.block(block)).is_err());
    }
}
