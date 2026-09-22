//! Hooks as rewrites of the lifted code.
//!
//! A [`Hook`] names the *sites* in a block it cares about — the block's entry,
//! a guest address, every store to guest memory, every comparison — and, for
//! each, emits QCode through an [`Emitter`]. What it emits is ordinary IR: a
//! [`VM_INTERRUPT`] where the host must act, arithmetic to decide whether it
//! must, loads and stores that count. The interpreter and the JIT run the
//! result like any other code, so a hook whose condition is false costs a few
//! native instructions and never leaves compiled code.
//!
//! The two ways to stop:
//!
//! - [`Emitter::interrupt`] places an unconditional interrupt before the site.
//! - [`Emitter::interrupt_if`] places the interrupt on a detour: the block is
//!   split before the site, and a conditional branch chooses between a small
//!   block holding the interrupt and the rest of the code. Only the condition
//!   is evaluated on the fast path.
//!
//! Both pass *values* to the host — literals, or anything the block computes,
//! such as the address and datum of a store — which the exit reports as the
//! interrupt's arguments.
//!
//! Hooks also read and write guest memory inline:
//! [`Emitter::load`] and [`Emitter::store`] emit an ordinary RAM load or store
//! before the site, so a hook can keep a counter, a shadow byte or a visited
//! bitmap in the guest's own address space without ever leaving compiled code.
//!
//! [`HookInjector`] adapts a hook to the [`CodeInjector`] the machine runs,
//! and handles idempotence: a site is instrumented once, remembered by its
//! anchor instruction, which survives absorption and disappears with a
//! re-lift.

use qcode::{
    context::Context,
    space::MemorySpaceId,
    value::{
        BasicBlock, BlockId, InstructionId, Value, ValueId,
        insn::{Binop, IntBinop, Mnemonic, Store, VM_INTERRUPT},
    },
};
use rustc_hash::FxHashSet;

use crate::inject::CodeInjector;

/// A point in a block a hook may instrument. Every site is anchored to the
/// instruction it precedes, which is where the hook's code is inserted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Site {
    /// The block's first instruction; `address` is the block's.
    BlockEntry { address: u64, anchor: InstructionId },
    /// The first instruction lifted from a guest instruction.
    Address { address: u64, anchor: InstructionId },
    /// A store to guest memory.
    Store { insn: InstructionId },
    /// A load from guest memory.
    Load { insn: InstructionId },
    /// An integer comparison.
    Compare { insn: InstructionId },
}

impl Site {
    /// The instruction the hook's code goes before.
    pub fn anchor(&self) -> InstructionId {
        match self {
            Self::BlockEntry { anchor, .. } | Self::Address { anchor, .. } => *anchor,
            Self::Store { insn } | Self::Load { insn } | Self::Compare { insn } => *insn,
        }
    }
}

/// A read-only look at a block, for choosing sites.
pub struct BlockView<'a> {
    pub ctx: &'a Context<'static>,
    pub block: BlockId,
}

impl BlockView<'_> {
    /// The block's guest address, if it starts at one.
    pub fn address(&self) -> Option<u64> {
        BasicBlock::from_id(self.ctx, self.block).address()
    }

    /// The site at the block's entry, if the block starts at a guest address.
    ///
    /// Anchored on the first instruction that is not an interrupt an earlier
    /// hook placed at the entry, so hooks on the same entry fire in
    /// registration order and a hook recognises its own anchor when asked
    /// again.
    pub fn entry(&self) -> Option<Site> {
        let address = self.address()?;
        let anchor = BasicBlock::from_id(self.ctx, self.block)
            .instructions()
            .find(|insn| !is_interrupt_op(self.ctx, insn.id))
            .map(|insn| insn.id)?;
        Some(Site::BlockEntry { address, anchor })
    }

    /// Every guest instruction that *starts* in the block, in order, as a
    /// site on the first instruction lifted from it.
    ///
    /// One guest instruction's p-code may branch within itself, so a block
    /// with no address of its own can open with the tail of an instruction
    /// begun in its predecessor. Those instructions carry the predecessor's
    /// last address; they are a continuation, not a start, and get no site.
    pub fn addresses(&self) -> Vec<Site> {
        let block = BasicBlock::from_id(self.ctx, self.block);
        let continued = if block.address().is_none() {
            block
                .predecessors()
                .filter_map(|(_, pred)| {
                    BasicBlock::from_id(self.ctx, pred)
                        .instructions()
                        .last()
                        .and_then(|insn| insn.address())
                })
                .collect::<Vec<u64>>()
        } else {
            Vec::new()
        };
        let mut sites = Vec::new();
        let mut seen = None;
        for insn in block.instructions() {
            // An interrupt a hook placed carries the site's address so the
            // stop reports it, but it is the hook's instruction, not the
            // guest's: never the start of a run, never an anchor.
            if is_interrupt_op(self.ctx, insn.id) {
                continue;
            }
            let at = insn.address();
            if at.is_some() && at != seen {
                let first_run = seen.is_none();
                seen = at;
                let address = at.unwrap_or_default();
                if first_run && continued.contains(&address) {
                    continue;
                }
                sites.push(Site::Address {
                    address,
                    anchor: insn.id,
                });
            }
        }
        sites
    }

    /// The instruction addressed by the guest instruction at `address`, as a
    /// site, if the block covers it.
    pub fn at(&self, address: u64) -> Option<Site> {
        self.addresses()
            .into_iter()
            .find(|site| matches!(site, Site::Address { address: at, .. } if *at == address))
    }

    fn is_ram(&self, space: qcode::space::LocalMemorySpaceId) -> bool {
        space.qualify(self.block.func) == MemorySpaceId::Shared(self.ctx.shared.default_space)
    }

    /// Every store to guest memory, in order.
    pub fn stores(&self) -> Vec<Site> {
        BasicBlock::from_id(self.ctx, self.block)
            .instructions()
            .filter(|insn| matches!(insn.mnemonic(), Mnemonic::Store(store) if self.is_ram(store.space)))
            .map(|insn| Site::Store { insn: insn.id })
            .collect()
    }

    /// Every load from guest memory, in order.
    pub fn loads(&self) -> Vec<Site> {
        BasicBlock::from_id(self.ctx, self.block)
            .instructions()
            .filter(
                |insn| matches!(insn.mnemonic(), Mnemonic::Load(load) if self.is_ram(load.space)),
            )
            .map(|insn| Site::Load { insn: insn.id })
            .collect()
    }

    /// Every integer comparison, in order.
    pub fn compares(&self) -> Vec<Site> {
        BasicBlock::from_id(self.ctx, self.block)
            .instructions()
            .filter(|insn| {
                matches!(insn.mnemonic(), Mnemonic::Binop(binary) if binary.op.is_comparison()
                    && matches!(binary.op, Binop::Int(_)))
            })
            .map(|insn| Site::Compare { insn: insn.id })
            .collect()
    }
}

/// A rewrite of lifted code at chosen sites. See the [module docs](self).
pub trait Hook {
    /// The sites in `block` to instrument. Returned in block order; a hook
    /// that instruments none returns an empty list.
    fn sites(&mut self, block: &BlockView<'_>) -> Vec<Site>;

    /// Emits the hook's code at `site`.
    fn instrument(&mut self, site: &Site, emit: &mut Emitter<'_>);
}

/// Emits QCode before a site's anchor, on the hook's behalf.
///
/// Values are [`ValueId`]s: literals from [`constant`](Self::constant),
/// operands of the anchor from [`store_operands`](Self::store_operands) and
/// friends, or the results of arithmetic emitted here. Everything emitted
/// stays *before* the anchor, in emission order.
pub struct Emitter<'a> {
    ctx: &'a mut Context<'static>,
    /// The block the anchor currently sits in; a split moves it.
    block: BlockId,
    anchor: InstructionId,
    /// The guest address the emitted code is stamped with.
    address: Option<u64>,
}

impl<'a> Emitter<'a> {
    pub fn new(ctx: &'a mut Context<'static>, site: &Site) -> Self {
        let anchor = site.anchor();
        let block = qcode::value::Instruction::from_id(ctx, anchor)
            .parent()
            .map(|block| block.id)
            .expect("a site's anchor is in a block");
        let address = match site {
            Site::BlockEntry { address, .. } | Site::Address { address, .. } => Some(*address),
            _ => qcode::value::Instruction::from_id(ctx, anchor).address(),
        };
        Self {
            ctx,
            block,
            anchor,
            address,
        }
    }

    pub fn ctx(&self) -> &Context<'static> {
        self.ctx
    }

    /// The guest address of the site.
    pub fn address(&self) -> Option<u64> {
        self.address
    }

    /// An integer literal of `size` bytes.
    pub fn constant(&self, value: u64, size: usize) -> ValueId {
        self.ctx.shared.get_const(value, size)
    }

    /// The width in bytes of a value.
    pub fn size_of(&self, value: ValueId) -> usize {
        self.ctx
            .stored_type_of(value)
            .map(|ty| self.ctx.shared.types.size_of(ty))
            .unwrap_or(0)
    }

    /// The anchor's store operands: pointer, width and stored value, if the
    /// anchor is a store.
    pub fn store_operands(&self) -> Option<(ValueId, usize, ValueId)> {
        let insn = self.ctx.instruction(self.anchor);
        let Mnemonic::Store(Store { ptr, size, src, .. }) = insn.mnemonic() else {
            return None;
        };
        let func = self.anchor.func;
        Some((ptr.qualify(func), *size, src.qualify(func)))
    }

    /// The anchor's load operands: pointer and width, if the anchor is a load.
    pub fn load_operands(&self) -> Option<(ValueId, usize)> {
        let insn = self.ctx.instruction(self.anchor);
        let Mnemonic::Load(load) = insn.mnemonic() else {
            return None;
        };
        Some((load.ptr.qualify(self.anchor.func), load.size))
    }

    /// The anchor's binary operands, if the anchor is a binary operation.
    pub fn binop_operands(&self) -> Option<(Binop, ValueId, ValueId)> {
        let insn = self.ctx.instruction(self.anchor);
        let Mnemonic::Binop(binary) = insn.mnemonic() else {
            return None;
        };
        let func = self.anchor.func;
        Some((
            binary.op,
            binary.lhs.qualify(func),
            binary.rhs.qualify(func),
        ))
    }

    /// Emits an integer binary operation before the anchor.
    ///
    /// Emitted arithmetic carries no guest address: it is the hook's, not
    /// the guest instruction's, and stamping it would make it look like the
    /// start of that instruction to the next hook choosing sites.
    pub fn binop(&mut self, op: IntBinop, lhs: ValueId, rhs: ValueId) -> ValueId {
        let (block, anchor) = (self.block, self.anchor);
        let mut builder = self.ctx.builder(block);
        builder.set_insert_point_before(anchor);
        builder.push_binop(Binop::Int(op), lhs, rhs).id()
    }

    /// Zero-extends (or truncates) `value` to `size` bytes.
    pub fn zext(&mut self, value: ValueId, size: usize) -> ValueId {
        if self.size_of(value) == size {
            return value;
        }
        let (block, anchor) = (self.block, self.anchor);
        let mut builder = self.ctx.builder(block);
        builder.set_insert_point_before(anchor);
        builder.push_zext(value, size).id()
    }

    /// Loads `size` bytes of guest RAM at `ptr` before the anchor.
    ///
    /// Like [`binop`](Self::binop), the load carries no guest address: it is
    /// the hook's instruction, not the guest's.
    pub fn load(&mut self, ptr: ValueId, size: usize) -> ValueId {
        let (block, anchor) = (self.block, self.anchor);
        let space = self.ctx.shared.default_space;
        let mut builder = self.ctx.builder(block);
        builder.set_insert_point_before(anchor);
        builder.push_load::<true>(ptr, size, space).id()
    }

    /// Stores `value`, at its own width, to guest RAM at `ptr` before the
    /// anchor.
    ///
    /// Like [`binop`](Self::binop), the store carries no guest address: it is
    /// the hook's instruction, not the guest's, and a store to the default
    /// space that looked like the start of a guest instruction would be
    /// offered as a site to the next hook.
    pub fn store(&mut self, ptr: ValueId, value: ValueId) {
        let (block, anchor) = (self.block, self.anchor);
        let space = self.ctx.shared.default_space;
        let mut builder = self.ctx.builder(block);
        builder.set_insert_point_before(anchor);
        // A varnode pointer names a register, not an address in RAM; take its
        // value, as the load side does with `CHECK_LOCAL`.
        let ptr = builder.ensure_local(ptr);
        builder.push_store(value, ptr, space);
    }

    /// `value - begin < end - begin + 1`, as a one-byte condition: whether a
    /// 64-bit `value` lies in `begin..=end`.
    pub fn in_range(&mut self, value: ValueId, begin: u64, end: u64) -> ValueId {
        let value = self.zext(value, 8);
        let offset = self.binop(IntBinop::Sub, value, self.constant(begin, 8));
        let length = self.constant(end.wrapping_sub(begin).wrapping_add(1), 8);
        self.binop(IntBinop::Less, offset, length)
    }

    /// Stops unconditionally before the anchor with `vm.interrupt(code,
    /// args...)`.
    pub fn interrupt(&mut self, code: u64, args: &[ValueId]) -> InstructionId {
        let (block, anchor, address) = (self.block, self.anchor, self.address);
        crate::inject::insert_interrupt(self.ctx, block, Some(anchor), address, code, args)
    }

    /// Stops before the anchor only when `cond` is non-zero, without leaving
    /// compiled code otherwise.
    ///
    /// The block is split before the anchor; the code emitted so far stays in
    /// the first half, which ends in `cbranch cond -> hook, rest`, where
    /// `hook` holds the interrupt and falls through to `rest`, the second
    /// half. Later emissions go before the anchor in `rest`.
    pub fn interrupt_if(&mut self, cond: ValueId, code: u64, args: &[ValueId]) -> InstructionId {
        let (block, anchor, address) = (self.block, self.anchor, self.address);
        let func = block.func;
        let rest = self.ctx.split_block_before(block, anchor);
        let hook = self.ctx.body_mut(func).make_block();
        let interrupt = crate::inject::insert_interrupt(self.ctx, hook, None, address, code, args);
        {
            let mut builder = self.ctx.builder(hook);
            if let Some(address) = address {
                builder.set_address(address);
            }
            builder.finalize(rest);
        }
        {
            // No address on the branch either: the rest of the block then
            // starts the guest instruction it starts, rather than looking
            // like a continuation of one.
            let mut builder = self.ctx.builder(block);
            builder.push_cbranch(cond, hook, rest);
        }
        self.block = rest;
        interrupt
    }
}

/// The guest address the fall-through tail of an [`Emitter::interrupt_if`]
/// split opens with, if `block` is one.
///
/// The split leaves exactly this shape: the block it was cut from keeps the
/// address and ends in `cbranch cond -> hook, tail`, and `hook` holds the
/// interrupt and falls through to `tail`. Neither the branch nor the tail is
/// stamped with a guest address — a tail that looked like the start of a
/// guest instruction would be offered as a site to the next hook choosing
/// them — but the detour's own instructions are, and that address is the
/// guest instruction the tail opens with.
///
/// That is what makes such a tail a place the machine may go on folding a
/// guest basic block into, though it carries no address: what it holds is
/// the guest's code from `address` on, and nothing else. A hook anchored
/// *inside* an instruction's p-code — on a store, a load, a comparison —
/// splits in the middle of one, and its tail opens with a continuation
/// instead; the block it was cut from still holding instructions of that
/// same address is how that case is told apart.
pub fn split_tail_address(ctx: &Context<'static>, block: BlockId) -> Option<u64> {
    let view = BasicBlock::from_id(ctx, block);
    if view.address().is_some() {
        return None;
    }
    let mut address: Option<u64> = None;
    let mut origin: Option<BlockId> = None;
    for (_, pred) in view.predecessors() {
        let terminator = BasicBlock::from_id(ctx, pred).last_instruction()?;
        match ctx.instruction(terminator).mnemonic() {
            // The arm that chose between the two: the block the tail was cut
            // from. There is one, and a second would mean something else
            // branches here — which nothing can, the tail having no address
            // to be found by.
            Mnemonic::CBranch(_) if origin.replace(pred).is_none() => {}
            // The interrupt's block, stamped with the site's address.
            Mnemonic::Branch(_)
                if BasicBlock::from_id(ctx, pred)
                    .instructions()
                    .any(|insn| is_interrupt_op(ctx, insn.id)) =>
            {
                let at = qcode::value::Instruction::from_id(ctx, terminator).address()?;
                if address.replace(at).is_some_and(|earlier| earlier != at) {
                    return None;
                }
            }
            _ => return None,
        }
    }
    let (address, origin) = (address?, origin?);
    // The tail opens the guest instruction at `address`, so the block it was
    // cut from holds none of it — and anything of it left behind would sit at
    // the very end, where the cut was made. So walk back from there to the
    // first instruction the guest addressed, rather than over a block that
    // may hold a whole run: the hook's own instructions carry no address, and
    // an interrupt an earlier hook placed carries the site's without being
    // its code.
    let mut at = BasicBlock::from_id(ctx, origin).last_instruction();
    while let Some(insn) = at.map(|id| qcode::value::Instruction::from_id(ctx, id)) {
        if !is_interrupt_op(ctx, insn.id)
            && let Some(stamped) = insn.address()
        {
            return (stamped != address).then_some(address);
        }
        at = insn.prev().map(|previous| previous.id);
    }
    Some(address)
}

/// Runs a [`Hook`] as the machine's [`CodeInjector`], instrumenting each site
/// once.
pub struct HookInjector<H> {
    pub hook: H,
    done: FxHashSet<InstructionId>,
}

impl<H: Hook> HookInjector<H> {
    pub fn new(hook: H) -> Self {
        Self {
            hook,
            done: FxHashSet::default(),
        }
    }
}

impl<H: Hook> CodeInjector for HookInjector<H> {
    fn inject(&mut self, ctx: &mut Context<'static>, block: BlockId) {
        let sites = self.hook.sites(&BlockView { ctx, block });
        for site in sites {
            if !self.done.insert(site.anchor()) {
                continue;
            }
            // The emitter finds the anchor's block itself: an earlier site's
            // split may have moved this one.
            let mut emit = Emitter::new(ctx, &site);
            self.hook.instrument(&site, &mut emit);
        }
    }
}

/// Whether `insn` is a [`VM_INTERRUPT`] op.
pub fn is_interrupt_op(ctx: &Context<'static>, insn: InstructionId) -> bool {
    match ctx.instruction(insn).mnemonic() {
        Mnemonic::PCodeOp(op) => ctx.shared.pcode_ops[op.id].as_ref() == VM_INTERRUPT,
        _ => false,
    }
}

// ---- The built-in hooks: the shapes every hook layer needs, and the pattern
// ---- for writing more.

/// Whether `addr` lies in `begin..=end`, or anywhere when `begin > end` —
/// Unicorn's convention for "no range".
fn in_range(begin: u64, end: u64, addr: u64) -> bool {
    begin > end || (begin..=end).contains(&addr)
}

/// Stops at the entry of every block whose address lies in a range, with
/// `vm.interrupt(code, address)`.
#[derive(Debug, Clone)]
pub struct BlockEntryHook {
    pub begin: u64,
    pub end: u64,
    pub code: u64,
}

impl Hook for BlockEntryHook {
    fn sites(&mut self, block: &BlockView<'_>) -> Vec<Site> {
        block
            .entry()
            .filter(|site| matches!(site, Site::BlockEntry { address, .. } if in_range(self.begin, self.end, *address)))
            .into_iter()
            .collect()
    }

    fn instrument(&mut self, _site: &Site, emit: &mut Emitter<'_>) {
        let address = emit.constant(emit.address().unwrap_or_default(), 8);
        emit.interrupt(self.code, &[address]);
    }
}

/// Stops before the guest instruction at each of a set of addresses, with
/// `vm.interrupt(code, address)`.
#[derive(Debug, Clone)]
pub struct AddressHook {
    pub addresses: FxHashSet<u64>,
    pub code: u64,
}

impl AddressHook {
    pub fn new(addresses: impl IntoIterator<Item = u64>, code: u64) -> Self {
        Self {
            addresses: addresses.into_iter().collect(),
            code,
        }
    }
}

impl Hook for AddressHook {
    fn sites(&mut self, block: &BlockView<'_>) -> Vec<Site> {
        block
            .addresses()
            .into_iter()
            .filter(|site| matches!(site, Site::Address { address, .. } if self.addresses.contains(address)))
            .collect()
    }

    fn instrument(&mut self, _site: &Site, emit: &mut Emitter<'_>) {
        let address = emit.constant(emit.address().unwrap_or_default(), 8);
        emit.interrupt(self.code, &[address]);
    }
}

/// Stops before every store to guest memory whose address lies in
/// `begin..=end`, with `vm.interrupt(code, address, size, value)`.
///
/// The range check is emitted as IR before the store, so a store elsewhere
/// costs three native instructions and no exit. The stored value is passed
/// when it fits an interrupt argument (8 bytes or fewer).
#[derive(Debug, Clone)]
pub struct WriteWatch {
    pub begin: u64,
    pub end: u64,
    pub code: u64,
}

impl Hook for WriteWatch {
    fn sites(&mut self, block: &BlockView<'_>) -> Vec<Site> {
        block.stores()
    }

    fn instrument(&mut self, _site: &Site, emit: &mut Emitter<'_>) {
        let Some((ptr, size, value)) = emit.store_operands() else {
            return;
        };
        let cond = emit.in_range(ptr, self.begin, self.end);
        let mut args = vec![emit.zext(ptr, 8), emit.constant(size as u64, 8)];
        if size <= 8 {
            args.push(emit.zext(value, 8));
        }
        emit.interrupt_if(cond, self.code, &args);
    }
}

/// Stops before every integer comparison with `vm.interrupt(code, lhs, rhs)`:
/// the operands as the guest computed them, which is what a comparison
/// logger for a fuzzer wants.
#[derive(Debug, Clone)]
pub struct CompareHook {
    pub code: u64,
}

impl Hook for CompareHook {
    fn sites(&mut self, block: &BlockView<'_>) -> Vec<Site> {
        block.compares()
    }

    fn instrument(&mut self, _site: &Site, emit: &mut Emitter<'_>) {
        let Some((_, lhs, rhs)) = emit.binop_operands() else {
            return;
        };
        emit.interrupt(self.code, &[lhs, rhs]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::value::{FunctionBody, insn::CBranch};

    /// A module holding one block of two instructions at `0x1000`, an `add`
    /// and a comparison of its result, branching to a block at `0x1004`.
    fn module() -> (Context<'static>, BlockId) {
        let mut ctx = Context::new();
        let function = FunctionBody::make_at_addr(&mut ctx, 0x1000, None).id;
        let block = BasicBlock::make(&mut ctx, function).with_address(0x1000).id;
        let next = BasicBlock::make(&mut ctx, function).with_address(0x1004).id;
        let one = ctx.shared.get_const(1, 4);
        let mut builder = ctx.builder(block);
        builder.set_address(0x1000);
        let sum = builder.push_binop(Binop::Int(IntBinop::Add), one, one).id();
        builder.push_binop(Binop::Int(IntBinop::Equal), sum, one);
        builder.finalize(next);
        (ctx, block)
    }

    /// The block the split left behind, read off the `cbranch` it ends in.
    fn tail_of(ctx: &Context<'static>, block: BlockId) -> BlockId {
        let terminator = BasicBlock::from_id(ctx, block)
            .last_instruction()
            .expect("the block is terminated");
        let Mnemonic::CBranch(CBranch { failure_block, .. }) =
            ctx.instruction(terminator).mnemonic()
        else {
            panic!("a conditional hook ends the block it splits in a cbranch");
        };
        BlockId::new(block.func, *failure_block)
    }

    #[test]
    fn the_tail_of_a_split_at_a_site_opens_the_site_s_instruction() {
        let (mut ctx, block) = module();
        let site = BlockView { ctx: &ctx, block }
            .entry()
            .expect("the block starts at a guest address");
        let mut emit = Emitter::new(&mut ctx, &site);
        let cond = emit.constant(1, 1);
        emit.interrupt_if(cond, 7, &[]);

        let tail = tail_of(&ctx, block);
        assert_eq!(split_tail_address(&ctx, tail), Some(0x1000));
        // The block the tail was cut from keeps its address, and is not one.
        assert_eq!(split_tail_address(&ctx, block), None);
    }

    #[test]
    fn the_tail_of_a_split_inside_an_instruction_is_not_one() {
        let (mut ctx, block) = module();
        // A comparison is not the start of the guest instruction it belongs
        // to, so the tail opens with the rest of that instruction — which no
        // lift of its address would reproduce on its own.
        let site = BlockView { ctx: &ctx, block }
            .compares()
            .pop()
            .expect("the block compares");
        let mut emit = Emitter::new(&mut ctx, &site);
        let cond = emit.constant(1, 1);
        emit.interrupt_if(cond, 7, &[]);

        assert_eq!(split_tail_address(&ctx, tail_of(&ctx, block)), None);
    }
}
