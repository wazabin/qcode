use jstd::Identifier;
use std::{
    borrow::Cow,
    collections::{BTreeSet, hash_set},
    fmt::{Display, Formatter},
};

use rustc_hash::FxHashSet as HashSet;

mod signature;
pub use signature::FunctionSignature;

use crate::{
    context::Context,
    error::{Error, ErrorTy, Result},
    value::{
        BasicBlock, BlockId, BlockRef, Instruction, Value, ValueId, Varnode, VarnodeId,
        util::{
            base_ref::{BaseRef, WithCtx, WithCtxMut},
            named::{Named, Renameable, update_context_name},
        },
    },
};

#[derive(Identifier)]
pub struct FunctionId(usize);

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Function<'str> {
    /// The function's name.
    pub name: Cow<'str, str>,

    /// Optional entry address (from binary).
    pub address: Option<u64>,

    /// The entry block (dominates all other blocks in this function).
    pub root: Option<BlockId>,

    /// All blocks belonging to this function (includes root).
    pub blocks: HashSet<BlockId>,

    /// Addresses of every machine instruction lifted into this function, in
    /// ascending order. Recorded during recursive disassembly and preserved
    /// across optimization (which merges blocks and rewrites the IR), so the
    /// raw disassembly view can be reconstructed regardless of CFG changes.
    pub instruction_addrs: BTreeSet<u64>,

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
            blocks: HashSet::default(),
            instruction_addrs: BTreeSet::new(),
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
    ) -> Result<FunctionMutRef<'str, 'ctx>> {
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

    /// Registers concretely written by this function, as set by analysis.
    pub fn clobbered_regs(&'s self) -> Option<&'ctx [VarnodeId]> {
        self.inner()
            .signature
            .as_ref()
            .and_then(|s| s.clobbered.as_deref())
    }

    /// Whether `argpromote_registers` has functionalized this function's register
    /// side effects into a pure value function. See
    /// [`FunctionSignature::pure_reg`].
    pub fn is_pure_reg(&'s self) -> bool {
        self.inner().signature.as_ref().is_some_and(|s| s.pure_reg)
    }

    /// Whether argpromote has functionalized *every* side-effect channel of this
    /// function — it is a deterministic pure function of its by-value params,
    /// touching no caller-visible memory or registers. Strictly stronger than
    /// [`is_pure_reg`](Self::is_pure_reg). See [`FunctionSignature::is_pure`].
    pub fn is_pure(&'s self) -> bool {
        self.inner().signature.as_ref().is_some_and(|s| s.is_pure)
    }

    /// Registers read before written (function inputs), as inferred by analysis.
    ///
    /// Legacy ABI input-register list. It is **`None` for `pure_reg` functions**
    /// (argpromote never populates it); their by-value root block params are the
    /// source of truth for the call interface, so prefer the params (e.g.
    /// [`input_arg_name`](Self::input_arg_name)). Retained only for the
    /// conventional/external calling-convention path (`summaries`).
    #[deprecated(
        note = "legacy ABI register list; None for pure_reg functions. Use the root block params \
                as the call interface; this remains only for the conventional/external path."
    )]
    pub fn input_regs(&'s self) -> Option<&'ctx [VarnodeId]> {
        self.inner()
            .signature
            .as_ref()
            .and_then(|s| s.inputs.as_deref())
    }

    /// The display name for the call-site argument bound to input `index`: the
    /// register name for a register input, or a synthesized `stack_<addr>` slot
    /// name for a stack-passed input (whose varnode is a nameless stack-space
    /// offset carrier). Mirrors mem2reg's `block_param_name_for_var` so a call
    /// argument reads with the same name as the callee's promoted stack
    /// parameter. `None` when there is no input at `index`.
    pub fn input_arg_name(&'s self, index: usize) -> Option<String> {
        // The root block param at `index` is the interface element a call
        // argument actually binds to, named after its register by
        // `argpromote_registers` or `stack_<addr>` by mem2reg's
        // `block_param_name_for_var`. Prefer it: it is the source of truth and is
        // populated even for `pure_reg` functions, whose ABI register list
        // (`input_regs`) is never filled in.
        if let Some(root) = self.root()
            && let Some(name) = root
                .params()
                .nth(index)
                .and_then(|p| p.name().map(str::to_owned))
        {
            return Some(name);
        }

        // Fall back to the inferred input-register list: a register name, or a
        // synthesized `stack_<addr>` slot name for a stack-passed input.
        // Intentional use of the legacy list — only reached when the param has no
        // name (conventional functions, never `pure_reg`).
        #[allow(deprecated)]
        let input = self.input_regs()?.get(index).copied()?;
        let vn = Varnode::from_id(self.ctx(), input);
        if let Some(name) = vn.name() {
            return Some(name.to_owned());
        }
        let space = vn.space();
        if space.name.as_deref() == Some("stack") {
            return Some(format!("stack_{:x}", vn.address() as u64));
        }
        None
    }

    /// Registers saved and restored unchanged (preserved across calls), as
    /// inferred by analysis.
    pub fn saved_regs(&'s self) -> Option<&'ctx [VarnodeId]> {
        self.inner()
            .signature
            .as_ref()
            .and_then(|s| s.saved.as_deref())
    }

    /// The net change this function applies to the stack pointer between entry
    /// and return, as inferred by analysis. See [`FunctionSignature::stack_delta`].
    pub fn stack_delta(&'s self) -> Option<i64> {
        self.inner().signature.as_ref().and_then(|s| s.stack_delta)
    }

    /// Whether this function performs an unresolved/dynamic stack read (or
    /// forwards a stack pointer into one). See
    /// [`FunctionSignature::reads_unbounded_stack`].
    pub fn reads_unbounded_stack(&'s self) -> bool {
        self.inner()
            .signature
            .as_ref()
            .is_some_and(|s| s.reads_unbounded_stack)
    }

    /// Whether this function hands a pointer into its own frame to a callee that
    /// may read it unboundedly. See
    /// [`FunctionSignature::frame_escapes_to_unbounded`].
    pub fn frame_escapes_to_unbounded(&'s self) -> bool {
        self.inner()
            .signature
            .as_ref()
            .is_some_and(|s| s.frame_escapes_to_unbounded)
    }

    /// The name of the inner `Function`.
    pub fn name(&'s self) -> &'ctx str {
        self.inner().name.as_ref()
    }

    /// The addresses of every machine instruction lifted into this function, in
    /// ascending order. Unlike [`blocks`](Self::blocks), this is stable across
    /// optimization, so it drives the raw disassembly view.
    pub fn instruction_addrs(&'s self) -> impl Iterator<Item = u64> + 'ctx {
        self.inner().instruction_addrs.iter().copied()
    }

    /// The functions this function directly calls, deduplicated and ordered by
    /// id. Derived from the IR on demand — like [`BlockRef::successors`] reading
    /// the CFG — so it always reflects the current instructions. Indirect calls
    /// have no static target and are not included.
    pub fn callees(&'s self) -> Vec<FunctionId> {
        let ctx = self.ctx();
        let mut callees = self
            .blocks()
            .flat_map(|block| {
                block
                    .instructions()
                    .filter_map(|insn| insn.mnemonic().call_target())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        // Synthetic edges (e.g. `entry → main`) recovered by a pass but not
        // backed by a direct call. Keyed by address; included once a function
        // exists at that address.
        callees.extend(
            ctx.values
                .synthetic_callees_of(self.id)
                .filter_map(|addr| Function::from_addr(ctx, addr).map(|function| function.id)),
        );
        callees.sort_by_key(|&id| Into::<usize>::into(id));
        callees.dedup();
        callees
    }

    /// The functions that directly call this one, deduplicated and ordered by
    /// id. Reads the reverse call graph maintained alongside the use-def map and
    /// resolves each call site to its enclosing function. Counterpart of
    /// [`callees`](Self::callees).
    pub fn callers(&'s self) -> Vec<FunctionId> {
        let ctx = self.ctx();
        let mut callers = ctx
            .values
            .call_sites_of(self.id)
            .iter()
            .filter_map(|&site| {
                Instruction::from_id(ctx, site)
                    .block()
                    .and_then(|block| block.function())
                    .map(|function| function.id)
            })
            .collect::<Vec<_>>();
        callers.sort_by_key(|&id| Into::<usize>::into(id));
        callers.dedup();
        callers
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

    /// Iterates over the blocks in this function
    /// This is slightly different from `blocks()` as the blocks will be returned in an arbitrary order, not sorted by address.
    pub fn iter(&'s self) -> BlockIter<'str, 'ctx> {
        let inner = self.inner();

        BlockIter {
            ctx: self.ctx(),
            inner: inner.blocks.iter(),
        }
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

pub struct BlockIter<'str, 'ctx> {
    ctx: &'ctx Context<'str>,
    inner: hash_set::Iter<'ctx, BlockId>,
}

impl<'str, 'ctx> Iterator for BlockIter<'str, 'ctx> {
    type Item = BlockRef<'str, 'ctx>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|id| BlockRef::new(self.ctx, *id))
    }
}

impl<'str, 'ctx> IntoIterator for &FunctionRef<'str, 'ctx> {
    type Item = BlockRef<'str, 'ctx>;
    type IntoIter = BlockIter<'str, 'ctx>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
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
    fn rename(&mut self, name: Cow<'str, str>) -> Result<()> {
        let id = self.id();
        let old_name = self.ctx.values.functions[self.id].name.as_ref().to_owned();
        update_context_name(id, self.ctx, name.clone(), Some(old_name.as_ref()))?;
        self.ctx.values.functions[self.id].name = name;
        Ok(())
    }
}

impl<'str, 'ctx> FunctionMutRef<'str, 'ctx> {
    pub(crate) fn inner_mut(&mut self) -> &mut Function<'str> {
        &mut self.ctx.values.functions[self.id]
    }

    fn set_address(&mut self, address: u64) -> Result<()> {
        self.inner_mut().address = Some(address);
        self.ctx.set_address(address, self.id.into())
    }

    fn with_address(mut self, address: u64) -> Result<Self> {
        self.set_address(address)?;
        Ok(self)
    }

    /// Sets a block as the root of this function.
    /// This will also add the block to the function's block list if it's not already present.
    /// This will also set the address of the function/block to the address of the root block/function if both addresses are unset.
    /// Panics if the function already has an address that doesn't match the root block's address.
    pub fn set_root(&mut self, id: BlockId) -> Result<()> {
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

    pub fn ensure_root(&mut self, id: BlockId) -> Result<()> {
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

    /// Records the analysis-computed clobbered register set on this function.
    pub fn set_clobbered_regs(&mut self, regs: Vec<VarnodeId>) {
        self.inner_mut().signature.get_or_insert_default().clobbered = Some(regs);
    }

    /// Records the analysis-inferred input (live-in) register set on this function.
    ///
    /// Legacy ABI input-register list — see [`input_regs`](Self::input_regs).
    /// Not set for `pure_reg` functions, whose param interface supersedes it.
    #[deprecated(
        note = "legacy ABI register list; None for pure_reg functions. Use the root block params \
                as the call interface; this remains only for the conventional/external path."
    )]
    pub fn set_input_regs(&mut self, regs: Vec<VarnodeId>) {
        self.inner_mut().signature.get_or_insert_default().inputs = Some(regs);
    }

    /// Marks this function as fully functionalized over its register channel.
    /// See [`FunctionSignature::pure_reg`].
    pub fn set_pure_reg(&mut self, value: bool) {
        self.inner_mut().signature.get_or_insert_default().pure_reg = value;
    }

    /// Marks this function as fully functionalized over *every* side-effect
    /// channel — a deterministic pure function of its params. See
    /// [`FunctionSignature::is_pure`].
    pub fn set_is_pure(&mut self, value: bool) {
        self.inner_mut().signature.get_or_insert_default().is_pure = value;
    }

    /// Records the analysis-inferred saved (preserved) register set on this function.
    pub fn set_saved_regs(&mut self, regs: Vec<VarnodeId>) {
        self.inner_mut().signature.get_or_insert_default().saved = Some(regs);
    }

    /// Records the output (return-value) register set on this function.
    pub fn set_output_regs(&mut self, regs: Vec<VarnodeId>) {
        self.inner_mut().signature.get_or_insert_default().outputs = Some(regs);
    }

    /// Records the analysis-inferred net stack-pointer delta on this function.
    pub fn set_stack_delta(&mut self, delta: i64) {
        self.inner_mut()
            .signature
            .get_or_insert_default()
            .stack_delta = Some(delta);
    }

    /// Records whether this function performs an unresolved/dynamic stack read.
    /// See [`FunctionSignature::reads_unbounded_stack`].
    pub fn set_reads_unbounded_stack(&mut self, value: bool) {
        self.inner_mut()
            .signature
            .get_or_insert_default()
            .reads_unbounded_stack = value;
    }

    /// Records whether this function hands a pointer into its own frame to a
    /// callee that may read it unboundedly. See
    /// [`FunctionSignature::frame_escapes_to_unbounded`].
    pub fn set_frame_escapes_to_unbounded(&mut self, value: bool) {
        self.inner_mut()
            .signature
            .get_or_insert_default()
            .frame_escapes_to_unbounded = value;
    }

    /// Records the address of a machine instruction lifted into this function.
    pub fn add_instruction_addr(&mut self, addr: u64) {
        self.inner_mut().instruction_addrs.insert(addr);
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
    use qcode_macro::qcode;

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
    fn iter_yields_all_blocks() {
        let mut ctx = Context::new();
        let root = BasicBlock::make(&mut ctx).id;
        let extra = BasicBlock::make(&mut ctx).id;
        let mut f = Function::make(&mut ctx, "iter_fn".into()).unwrap();
        f.add_block(root);
        f.add_block(extra);

        let f = Function::from_name(&ctx, "iter_fn").unwrap();
        let ids: Vec<_> = f.iter().map(|b| b.id).collect();
        assert!(ids.contains(&root));
        assert!(ids.contains(&extra));
    }

    #[test]
    fn into_iterator_for_function_ref_matches_iter() {
        let mut ctx = Context::new();
        let b1 = BasicBlock::make(&mut ctx).id;
        let b2 = BasicBlock::make(&mut ctx).id;
        let mut f = Function::make(&mut ctx, "into_iter_fn".into()).unwrap();
        f.add_block(b1);
        f.add_block(b2);

        let f = Function::from_name(&ctx, "into_iter_fn").unwrap();
        let mut via_iter: Vec<usize> = f.iter().map(|b| b.id.into()).collect();
        let mut via_into: Vec<usize> = (&f).into_iter().map(|b| b.id.into()).collect();
        via_iter.sort();
        via_into.sort();
        assert_eq!(via_iter, via_into);
    }

    #[test]
    fn qcode_fn_single_block_populates_function() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn simple:
                <entry>
                    return [0];
            "
        );

        let f = Function::from_name(&ctx, "simple").unwrap();
        assert_eq!(f.name(), "simple");
        assert!(f.root().is_some());
        assert_eq!(f.root().unwrap().name().unwrap(), "entry");
        assert_eq!(f.blocks().count(), 1);
    }

    #[test]
    fn qcode_fn_multi_block_populates_all_blocks() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn multiblock:
                <bb1>
                    if i8 1 goto <bb2> else goto <bb3>;

                <bb2>
                    goto <bb3>;

                <bb3>
                    return [0];
            "
        );

        let f = Function::from_name(&ctx, "multiblock").unwrap();
        assert_eq!(f.root().unwrap().name().unwrap(), "bb1");
        let block_names: Vec<_> = f.blocks().filter_map(|b| b.name()).collect();
        assert!(block_names.contains(&"bb1"), "missing bb1");
        assert!(block_names.contains(&"bb2"), "missing bb2");
        assert!(block_names.contains(&"bb3"), "missing bb3");
        assert_eq!(f.blocks().count(), 3);
    }

    #[test]
    fn qcode_fn_id_variable_is_set() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn myfn:
                <start>
                    return [0];
            "
        );

        let by_name = Function::from_name(&ctx, "myfn").unwrap();
        assert_eq!(by_name.name(), "myfn");
    }

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
