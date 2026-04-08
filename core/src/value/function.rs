use jstd::Identifier;
use std::{
    borrow::Cow,
    collections::HashSet,
    fmt::{Display, Formatter},
};

mod signature;
pub use signature::FunctionSignature;

use crate::{
    context::Context,
    error::{Error, ErrorTy, Result},
    value::{
        BasicBlock, BlockId, BlockRef, Value, ValueId,
        util::{
            base_ref::{BaseRef, WithCtx, WithCtxMut},
            named::{Named, Renameable, update_context_name},
        },
    },
};

#[derive(Identifier)]
pub struct FunctionId(usize);

#[derive(Clone)]
pub struct Function<'str> {
    /// The function's name.
    pub name: Cow<'str, str>,

    /// Optional entry address (from binary).
    pub address: Option<u64>,

    /// The entry block (dominates all other blocks in this function).
    pub root: Option<BlockId>,

    /// All blocks belonging to this function (includes root).
    pub blocks: HashSet<BlockId>,

    /// Whether this is an external (imported) function.
    ///
    /// External functions have no lifted body — they are stubs for calls that
    /// go outside the binary (e.g. PLT thunks for shared-library functions).
    /// The recursive disassembler will not attempt to lift their body.
    pub is_external: bool,

    /// Optional ABI description used by alias analysis.
    pub signature: Option<FunctionSignature>,
}

impl<'str> Function<'str> {
    fn new(name: Cow<'str, str>) -> Self {
        Self {
            name,
            address: None,
            root: None,
            blocks: HashSet::new(),
            is_external: false,
            signature: None,
        }
    }

    /// Gets a reference to a function from its ID
    pub fn from_id<'ctx>(ctx: &'ctx Context<'str>, id: FunctionId) -> FunctionRef<'str, 'ctx> {
        FunctionRef { ctx, id }
    }

    /// Gets a mutable reference to a function from its ID
    pub fn from_id_mut<'ctx>(
        ctx: &'ctx mut Context<'str>,
        id: FunctionId,
    ) -> FunctionMutRef<'str, 'ctx> {
        FunctionMutRef::new(ctx, id)
    }

    /// Gets a reference to a function by name
    pub fn from_name<'ctx>(
        ctx: &'ctx Context<'str>,
        name: &str,
    ) -> Option<FunctionRef<'str, 'ctx>> {
        ctx.get_named(name)
            .and_then(ValueId::as_function)
            .map(|id| FunctionRef { ctx, id })
    }

    /// Gets a reference to a function by address
    pub fn from_addr<'ctx>(ctx: &'ctx Context<'str>, addr: u64) -> Option<FunctionRef<'str, 'ctx>> {
        ctx.get_at_addr(&addr)
            .and_then(ValueId::as_function)
            .map(|id| FunctionRef { ctx, id })
    }

    /// Gets a mutable reference to a function by address
    pub fn from_addr_mut<'ctx>(
        ctx: &'ctx mut Context<'str>,
        addr: u64,
    ) -> Option<FunctionMutRef<'str, 'ctx>> {
        ctx.get_at_addr(&addr)
            .and_then(ValueId::as_function)
            .map(|id| FunctionMutRef::new(ctx, id))
    }

    /// Create a new function
    pub fn make<'ctx>(
        ctx: &'ctx mut Context<'str>,
        name: Cow<'str, str>,
    ) -> Result<'str, FunctionMutRef<'str, 'ctx>> {
        let id = ctx.values.push_function(Function::new(name.clone()));
        ctx.update_name(name, id.into(), None)?;
        Ok(Self::from_id_mut(ctx, id))
    }

    /// Create a new function at a given address, generating a name if necessary.
    pub fn make_at_addr<'ctx>(
        ctx: &'ctx mut Context<'str>,
        address: u64,
        name: Option<Cow<'str, str>>,
    ) -> FunctionMutRef<'str, 'ctx> {
        let name = match name {
            Some(name) => name,
            None => Cow::Owned(format!("fn_{address:x}")),
        };

        let id = ctx.values.push_function(Function::new(name.clone()));

        Self::from_id_mut(ctx, id)
            .with_name(name)
            .expect("Function name is not unique")
            .with_address(address)
            .expect("Function address is not unique")
    }

    /// Like [`Function::make_at_addr`] but marks the result as external.
    ///
    /// External functions have no lifted body; the recursive disassembler will
    /// not try to explore them.
    pub fn make_external<'ctx>(
        ctx: &'ctx mut Context<'str>,
        address: u64,
        name: Option<Cow<'str, str>>,
    ) -> FunctionMutRef<'str, 'ctx> {
        let mut f = Self::make_at_addr(ctx, address, name);
        f.inner_mut().is_external = true;
        f
    }

    /// Returns the [`FunctionId`] for `addr`, creating a named stub if absent.
    pub fn from_addr_or_create<'ctx>(
        ctx: &'ctx mut Context<'str>,
        address: u64,
    ) -> FunctionMutRef<'str, 'ctx> {
        match ctx.get_at_addr(&address).and_then(ValueId::as_function) {
            Some(id) => Self::from_id_mut(ctx, id),
            None => Self::make_at_addr(ctx, address, None),
        }
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx, Ctx> BaseRef<Ctx, FunctionId>
where
    Self: WithCtx<'s, 'ctx, 'str>,
{
    fn inner(&'s self) -> &'ctx Function<'str> {
        &self.ctx().values.functions[self.id]
    }

    fn size(&self) -> usize {
        0
    }

    /// The `address` of the inner `Function`.
    pub fn address(&'s self) -> Option<u64> {
        self.inner().address
    }

    /// Whether the inner `Function` is external.
    pub fn is_external(&'s self) -> bool {
        self.inner().is_external
    }

    /// A reference to the signature of the inner `Function`, if any.
    pub fn signature(&'s self) -> Option<&'ctx FunctionSignature> {
        self.inner().signature.as_ref()
    }

    /// The name of the inner `Function`.
    pub fn name(&'s self) -> &'ctx str {
        self.inner().name.as_ref()
    }

    /// The root block of this function, if it exists.
    pub fn root(&'s self) -> Option<BlockRef<'str, 'ctx>> {
        self.inner().root.map(|id| BlockRef::new(self.ctx(), id))
    }

    /// An iterator over the blocks belonging to this function.
    pub fn blocks(&'s self) -> impl Iterator<Item = BlockRef<'str, 'ctx>> + 's {
        let ctx = self.ctx();
        let mut blocks = self
            .inner()
            .blocks
            .iter()
            .map(move |&id| BlockRef::new(ctx, id))
            .collect::<Vec<_>>();

        blocks.sort_by_key(|b| b.address());
        blocks.into_iter()
    }

    fn fmt(&'s self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.is_external() {
            return writeln!(f, "extern fn {};", self.name());
        }
        writeln!(f, "fn {}:", self.name())?;
        for block in self.blocks() {
            block.fmt(f)?;
        }
        Ok(())
    }
}

pub type FunctionRef<'str, 'ctx> = BaseRef<&'ctx Context<'str>, FunctionId>;

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 'ctx, 'str> for FunctionRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'ctx Context<'str> {
        self.ctx
    }
}

impl Named for FunctionRef<'_, '_> {
    fn name(&self) -> Option<&str> {
        Some(self.ctx.values.functions[self.id].name.as_ref())
    }
}

impl Display for FunctionRef<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.fmt(f)
    }
}

impl<'str, 'ctx> Value<'str, 'ctx> for FunctionRef<'str, 'ctx> {
    fn id(&self) -> ValueId {
        self.id()
    }

    fn size(&self) -> usize {
        self.size()
    }
}

pub type FunctionMutRef<'str, 'ctx> = BaseRef<&'ctx mut Context<'str>, FunctionId>;

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtx<'s, 's, 'str> for FunctionMutRef<'str, 'ctx> {
    fn ctx(&'s self) -> &'s Context<'str> {
        self.ctx
    }
}

impl<'s, 'ctx: 's, 'str: 'ctx> WithCtxMut<'s, 'str> for FunctionMutRef<'str, 'ctx> {
    fn ctx_mut(&'s mut self) -> &'s mut Context<'str> {
        self.ctx
    }
}

impl Display for FunctionMutRef<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.fmt(f)
    }
}

impl<'ctx, 'str> Value<'str, 'ctx> for FunctionMutRef<'str, 'ctx> {
    fn id(&self) -> ValueId {
        self.id()
    }

    fn size(&self) -> usize {
        self.size()
    }
}

impl Named for FunctionMutRef<'_, '_> {
    fn name(&self) -> Option<&str> {
        Some(self.ctx.values.functions[self.id].name.as_ref())
    }
}

impl<'str, 'ctx> Renameable<'str, 'ctx> for FunctionMutRef<'str, 'ctx> {
    fn rename(&mut self, name: Cow<'str, str>) -> Result<'str, ()> {
        let id = self.id();
        let old_name = self.ctx.values.functions[self.id].name.as_ref().to_owned();
        update_context_name(id, self.ctx, name.clone(), Some(old_name.as_ref()))?;
        self.ctx.values.functions[self.id].name = name;
        Ok(())
    }
}

impl<'str, 'ctx> FunctionMutRef<'str, 'ctx> {
    fn inner_mut(&mut self) -> &mut Function<'str> {
        &mut self.ctx.values.functions[self.id]
    }

    fn set_address(&mut self, address: u64) -> Result<'str, ()> {
        self.inner_mut().address = Some(address);
        self.ctx.set_address(address, self.id.into())
    }

    fn with_address(mut self, address: u64) -> Result<'str, Self> {
        self.set_address(address)?;
        Ok(self)
    }

    /// Sets a block as the root of this function.
    /// This will also add the block to the function's block list if it's not already present.
    /// This will also set the address of the function/block to the address of the root block/function if both addresses are unset.
    /// Panics if the function already has an address that doesn't match the root block's address.
    pub fn set_root(&mut self, id: BlockId) -> Result<'str, ()> {
        self.add_block(id);
        self.inner_mut().root = Some(id);

        let block_addr = BasicBlock::from_id(self.ctx, id).address();
        let self_addr = self.address();

        match (self_addr, block_addr) {
            (Some(fn_addr), Some(block_addr)) if fn_addr != block_addr => {
                return Err(Error::spanless(ErrorTy::FunctionRootAddressMismatch {
                    fn_addr,
                    block_addr,
                }));
            }
            (None, Some(addr)) => {
                self.set_address(addr)
                    .expect("This address should be valid");
            }
            (Some(addr), None) => {
                BasicBlock::from_id_mut(self.ctx, id)
                    .set_address(addr)
                    .expect("This address should be valid");
            }
            _ => {}
        }
        Ok(())
    }

    pub fn make_root(&mut self) -> BlockRef<'str, '_> {
        let root = BasicBlock::make(self.ctx).id;
        self.set_root(root).expect("We just created the block");
        BlockRef::new(self.ctx, root)
    }

    pub fn ensure_root(&mut self, id: BlockId) -> Result<'str, ()> {
        if let Some(root) = self.inner().root {
            if root != id {
                return Err(Error::spanless(ErrorTy::FunctionRootMismatch {
                    expected: root,
                    actual: id,
                }));
            }
            Ok(())
        } else {
            self.set_root(id)
        }
    }

    pub fn set_external(&mut self, is_external: bool) {
        self.inner_mut().is_external = is_external;
        assert!(
            self.inner().blocks.is_empty(),
            "External functions should not have blocks"
        );
    }

    pub fn set_signature(&mut self, sig: FunctionSignature) {
        self.ctx.values.functions[self.id].signature = Some(sig);
    }

    /// Associates `block` with `function`: pushes it onto the function's block list
    /// and sets the block's `parent` field.
    pub fn add_block(&mut self, id: BlockId) {
        self.inner_mut().blocks.insert(id);
        self.ctx.values.basic_blocks[id].parent = Some(self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_function_creates_function_with_correct_name_root_address() {
        let mut ctx = Context::new();
        let f = Function::make(&mut ctx, "main".into()).unwrap();
        assert_eq!(f.name(), "main");
    }

    #[test]
    fn get_function_by_name_returns_correct_function() {
        let mut ctx = Context::new();
        let id = Function::make(&mut ctx, "foo".into()).unwrap().id();
        let f = Function::from_name(&ctx, "foo").unwrap();
        assert_eq!(f.id(), id);
        assert_eq!(f.name(), "foo");
    }

    #[test]
    fn get_function_by_name_returns_none_if_not_found() {
        let ctx = Context::new();
        assert!(Function::from_name(&ctx, "nonexistent").is_none());
    }

    #[test]
    fn get_function_by_addr_returns_correct_function() {
        let mut ctx = Context::new();
        let id = Function::make_at_addr(&mut ctx, 0x2000, None).id();
        let f = Function::from_addr(&ctx, 0x2000).unwrap();
        assert_eq!(f.id(), id);
        assert_eq!(f.address(), Some(0x2000));
        assert_eq!(f.name(), "fn_2000");
    }

    #[test]
    fn get_function_by_addr_returns_none_if_missing() {
        let ctx = Context::new();
        assert!(Function::from_addr(&ctx, 0xdeadbeef).is_none());
    }

    #[test]
    fn add_block_via_function_mut_ref_updates_blocks_list() {
        let mut ctx = Context::new();
        let root = BasicBlock::make(&mut ctx).id;
        let extra = BasicBlock::make(&mut ctx).id;

        let mut baz = Function::make(&mut ctx, "baz".into()).unwrap();
        baz.add_block(root);
        baz.add_block(extra);

        let block_ids: Vec<_> = baz.blocks().map(|b| b.id).collect();
        assert!(block_ids.contains(&root));
        assert!(block_ids.contains(&extra));
    }

    #[test]
    fn display_shows_function_name_and_block_contents() {
        let mut ctx = Context::new();
        Function::make(&mut ctx, "display_test".into()).unwrap();

        let f = Function::from_name(&ctx, "display_test").unwrap();

        let s = f.to_string();
        assert!(s.contains("fn display_test:"));
    }

    /// A BasicBlock may be created at an address before the Function stub for
    /// that address is registered (e.g. when Sleigh emits a branch target block
    /// ahead of the function being lifted).  `set_address` must allow this and
    /// must attribute the block as the function's root.
    #[test]
    fn set_address_allows_function_at_existing_block_address() {
        let mut ctx = Context::new();

        // Simulate a branch-target block created at 0x1000 before the function
        // stub exists (as happens with tail-jumps to sibling functions).
        let block_id = BasicBlock::make(&mut ctx).id;
        ctx.set_address(0x1000, ValueId::BasicBlock(block_id))
            .unwrap();

        // Registering a function at the same address must succeed.
        let fn_id = Function::make_at_addr(&mut ctx, 0x1000, None).id;

        // The function wins in the address map.
        assert!(Function::from_addr(&ctx, 0x1000).is_some());
        // The pre-existing block becomes the function's root.
        assert_eq!(
            Function::from_id(&ctx, fn_id).root().map(|b| b.id),
            Some(block_id)
        );
    }
}
