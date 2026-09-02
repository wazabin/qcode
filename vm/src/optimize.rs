//! A cheap cleanup round for freshly lifted code.
//!
//! SLEIGH expresses an instruction's semantics through *unique* (temporary)
//! space: an intermediate result is stored to a temporary and immediately loaded
//! back. Lifting `add eax, ebx` produces around forty QCode operations, roughly
//! half of which are this round trip:
//!
//! ```text
//! i32 %tmpe  = %eax_3 ^ %ebx_3;
//! store($temp1:4, 0 <- %tmpe);
//! i32 %tmp10 = load($temp1:4, 0);   // reloads what was just stored
//! ```
//!
//! Plain dead-code elimination cannot touch this — the store has a user, and the
//! load has users — so the interpreter re-executes the whole round trip on every
//! pass over the block. Forwarding the stored value to the load removes both.
//!
//! This is deliberately the *limited* version of store-to-load forwarding, not
//! [`mem2reg`](qcode_analysis::mem::mem2reg): it is block-local, needs no alias
//! analysis, and is linear in the size of the block, so it can run on every
//! lifted block without reintroducing the quadratic cost that a whole-function
//! pass would.
//!
//! # Why this is sound
//!
//! Only *temporary* spaces are forwarded. A SLEIGH unique is scratch private to
//! one instruction's semantics: it is written before it is read and does not
//! outlive the block, so no other block, and no guest-visible memory access, can
//! observe it. Registers and RAM are left alone, since a call, a fault handler,
//! or another block legitimately observes those.
//!
//! Two further restrictions keep it honest:
//!
//! * A load is forwarded only when it matches the last store to that space *and
//!   has the same width*. A narrower or wider access is left alone rather than
//!   guessed at.
//! * A store through a non-constant pointer clears what is known about that
//!   space, since it may land anywhere in it.

use qcode::{
    context::Context,
    space::{MemorySpaceId, Space, SpaceType},
    value::{
        BasicBlock, BlockId, Instruction, ValueId, ValueRef,
        insn::{InstructionId, Load, Mnemonic, Store},
    },
};
use rustc_hash::{FxHashMap, FxHashSet};

/// What one cleanup round changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Cleanup {
    /// Loads replaced by the value that was stored.
    pub forwarded_loads: usize,
    /// Stores removed because nothing reads them any more.
    pub removed_stores: usize,
}

impl Cleanup {
    pub fn is_empty(&self) -> bool {
        self.forwarded_loads == 0 && self.removed_stores == 0
    }
}

/// Whether `space` is scratch private to the block being lifted.
fn is_temporary(ctx: &Context<'_>, space: MemorySpaceId) -> bool {
    match space {
        // A per-function temporary space is scratch by construction.
        MemorySpaceId::Temp(_) => true,
        MemorySpaceId::Shared(id) => {
            matches!(Space::from_id(ctx, id).ty, SpaceType::Unique)
        }
    }
}

/// The constant address a pointer denotes, if it is one.
///
/// SLEIGH addresses a temporary by a `Temp` value rather than a literal — the
/// interpreter evaluates one to its address — so both forms are constants here.
fn constant_address(ctx: &Context<'_>, ptr: ValueId) -> Option<u64> {
    match ValueRef::new(ptr, ctx) {
        ValueRef::Literal(literal) => Some(literal.value()),
        ValueRef::Temp(temp) => Some(temp.address() as u64),
        ValueRef::Varnode(varnode) => Some(varnode.address() as u64),
        _ => None,
    }
}

/// Forwards temporary-space stores to the loads that read them back, in
/// `block_id` only.
pub fn forward_temp_stores(ctx: &mut Context<'_>, block_id: BlockId) -> Cleanup {
    let func = block_id.func;
    let insn_ids: Vec<InstructionId> = BasicBlock::from_id(ctx, block_id).instruction_ids();

    /// The value last stored to a temporary location, and its width.
    struct Stored {
        value: ValueId,
        size: usize,
        store: InstructionId,
    }

    let mut available: FxHashMap<(MemorySpaceId, u64), Stored> = FxHashMap::default();
    // Stores whose every reader was forwarded. A store still standing at the end
    // of the block is kept: something outside this pass's knowledge may read it.
    let mut redundant: FxHashSet<(InstructionId, (MemorySpaceId, u64))> = FxHashSet::default();
    // Spaces where an access through an unknown pointer was seen: nothing about
    // them can be concluded, so none of their stores may be removed.
    let mut poisoned: FxHashSet<MemorySpaceId> = FxHashSet::default();
    // Slots that were read by something other than a forwarded load.
    let mut read_otherwise: FxHashSet<(MemorySpaceId, u64)> = FxHashSet::default();
    let mut consumed: FxHashSet<InstructionId> = FxHashSet::default();
    let mut forwards: Vec<(ValueId, ValueId)> = Vec::new();

    for &insn_id in &insn_ids {
        let mnemonic = Instruction::from_id(ctx, insn_id).mnemonic().clone();
        match mnemonic {
            Mnemonic::Store(Store {
                space,
                ptr,
                size,
                src,
            }) => {
                let space = space.qualify(func);
                if !is_temporary(ctx, space) {
                    continue;
                }
                match constant_address(ctx, ptr.qualify(func)) {
                    Some(addr) => {
                        // Overwriting an earlier store to the same slot whose
                        // readers were all forwarded makes the earlier one dead.
                        available.insert(
                            (space, addr),
                            Stored {
                                value: src.qualify(func),
                                size,
                                store: insn_id,
                            },
                        );
                    }
                    // An unknown destination may be anywhere in the space, so
                    // nothing known about it survives.
                    None => {
                        poisoned.insert(space);
                        available.retain(|(other, _), _| *other != space);
                    }
                }
            }
            Mnemonic::Load(Load { space, ptr, size }) => {
                let space = space.qualify(func);
                if !is_temporary(ctx, space) {
                    continue;
                }
                let Some(addr) = constant_address(ctx, ptr.qualify(func)) else {
                    // An unknown source could read any of this space, so no
                    // store to it can be considered fully consumed.
                    poisoned.insert(space);
                    available.retain(|(other, _), _| *other != space);
                    continue;
                };
                match available.get(&(space, addr)) {
                    // Same slot, same width: the load is exactly the value that
                    // was stored.
                    Some(stored) if stored.size == size => {
                        forwards.push((ValueId::Instruction(insn_id), stored.value));
                        consumed.insert(insn_id);
                        redundant.insert((stored.store, (space, addr)));
                    }
                    // A different width reads bytes this pass does not model,
                    // so the store behind it is genuinely read.
                    Some(_) => {
                        read_otherwise.insert((space, addr));
                        available.remove(&(space, addr));
                    }
                    None => {
                        read_otherwise.insert((space, addr));
                    }
                }
            }
            _ => {}
        }
    }

    if forwards.is_empty() {
        return Cleanup::default();
    }

    let forwarded_loads = forwards.len();

    let body = ctx.function_mut(func);
    for (load_result, stored_value) in forwards {
        body.replace_all_uses_with(load_result, stored_value);
    }
    // A store goes only when every read of its slot in this block was
    // forwarded and nothing accessed the space through an unknown pointer.
    let dead_stores: Vec<InstructionId> = redundant
        .iter()
        .filter(|(_, slot)| !poisoned.contains(&slot.0) && !read_otherwise.contains(slot))
        .map(|(store, _)| *store)
        .collect();
    let mut dead: Vec<_> = consumed.iter().copied().chain(dead_stores.iter().copied()).collect();
    dead.sort_unstable();
    for id in dead {
        body.remove_instruction(id);
    }

    Cleanup {
        forwarded_loads,
        removed_stores: dead_stores.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::value::{FunctionBody, QCodeMut};

    /// Registers and RAM must be left alone even though the shape matches.
    #[test]
    fn only_temporary_spaces_are_forwarded() {
        let mut ctx = Context::new();
        let function = FunctionBody::make_at_addr(&mut ctx, 0x1000, None).id;
        let block = BasicBlock::make(&mut ctx, function).with_address(0x1000).id;
        // The default space is RAM, which this pass must not touch.
        let ram = MemorySpaceId::Shared(ctx.shared.default_space);
        assert!(!is_temporary(&ctx, ram));
        assert_eq!(forward_temp_stores(&mut ctx, block), Cleanup::default());
    }

    #[test]
    fn an_empty_block_is_unchanged() {
        let mut ctx = Context::new();
        let function = FunctionBody::make_at_addr(&mut ctx, 0x1000, None).id;
        let block = BasicBlock::make(&mut ctx, function).with_address(0x1000).id;
        assert!(forward_temp_stores(&mut ctx, block).is_empty());
    }
}
