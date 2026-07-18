//! `promote_global_cells`: classify constant real-RAM globals and prepare them
//! for the register effect channel (see `GLOBALS_AS_VARNODES.md`).
//!
//! An early, whole-program pre-pass with two jobs, both keyed on a single
//! syntactic scan for constant-address accesses:
//!
//! 1. **Read-only fold.** A global that appears as a `store(ram, <const>)`
//!    *nowhere* in the program, and lives outside a section the loader may
//!    rewrite (`is_known_writable`), is immutable. Every `load(ram, <const>)`
//!    of it folds to the byte value in the binary image, guarded by a
//!    [`Proposition::ImmutableMemory`] assumption (a later-proven write refutes
//!    it → checkpoint+replay). Such a global never enters any interface.
//!
//! 2. **Pre-mint mutable cells.** Every *remaining* constant-RAM access (a
//!    global that is written somewhere) gets a stable identity varnode minted
//!    via [`Context::get_or_make_global_varnode`], so the immutable register
//!    effect scan can look it up and thread the global's **value** like a
//!    register.
//!
//! # Accepted unsoundness (read this before touching the write-set logic)
//!
//! ┌───────────────────────────────────────────────────────────────────────┐
//! │  A global counts as *written* ONLY when some instruction is a          │
//! │  syntactic `store(ram, <constant address>)`. A store through a         │
//! │  COMPUTED pointer — `store(ram, %p)` — is IGNORED: it neither marks a  │
//! │  global mutable nor clobbers a global cell, even though it *could*     │
//! │  alias one at runtime. This is deliberate, controllable unsoundness,   │
//! │  exactly symmetric to registers (which are never modelled as aliasable │
//! │  by memory). The checkpoint+replay net catches anything later *proven*.│
//! │  Do NOT "fix" this by having computed stores poison the global set —   │
//! │  that collapses the whole channel in obfuscated code full of computed  │
//! │  stores, which is the case this exists to handle.                      │
//! └───────────────────────────────────────────────────────────────────────┘

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use qcode::{
    assumption::Proposition,
    context::Context,
    space::{LocalMemorySpaceId, Space, SpaceType},
    value::{FunctionBody, FunctionId, LocalValueId, QCodeMut, ValueId, ValueRef, insn::Mnemonic},
};

use crate::{Pass, PipelineEnv};

pub struct PromoteGlobalCells;

impl Default for PromoteGlobalCells {
    fn default() -> Self {
        Self
    }
}

/// A constant real-RAM pointer: `(address, access-data-width)`. Returns `None`
/// for register/temporary/non-constant pointers.
fn const_ram_access(
    ctx: &Context,
    fid: FunctionId,
    ptr: LocalValueId,
    space: LocalMemorySpaceId,
    size: usize,
) -> Option<(u64, usize)> {
    if !space
        .shared()
        .is_some_and(|s| matches!(Space::from_id(ctx, s).ty, SpaceType::Ram))
    {
        return None;
    }
    let ValueRef::Literal(lit) = ValueRef::new(ptr.qualify(fid), ctx) else {
        return None;
    };
    Some((lit.value(), size))
}

impl Pass for PromoteGlobalCells {
    const NAME: &'static str = "promote_global_cells";

    fn description(&self) -> &'static str {
        "Folds read-only constant globals and pre-mints mutable global cells"
    }

    fn run(
        &self,
        ctx: &mut Context,
        env: &PipelineEnv,
        targets: &[FunctionId],
    ) -> Result<crate::ModulePassOutcome, String> {
        let fun_ids: Vec<FunctionId> = targets
            .iter()
            .copied()
            .filter(|&id| !FunctionBody::from_id(ctx, id).is_external())
            .collect();

        // Phase 1: whole-program syntactic write-set — addresses that appear as
        // a `store(ram, <const>)` anywhere. (See the accepted-unsoundness box:
        // computed stores are intentionally excluded.)
        let mut written: HashSet<u64> = HashSet::default();
        for &fid in &fun_ids {
            for block in FunctionBody::from_id(ctx, fid).blocks() {
                for insn in block.iter() {
                    if let Mnemonic::Store(s) = insn.mnemonic()
                        && let Some((addr, _)) = const_ram_access(ctx, fid, s.ptr, s.space, s.size)
                    {
                        written.insert(addr);
                    }
                }
            }
        }

        // Phase 2: per-function, collect read-only load folds and the set of
        // mutable global cells to pre-mint. Scan is immutable; mutations apply
        // after.
        let binary = env.binary.as_deref();
        let mut folds: HashMap<FunctionId, Vec<(qcode::value::insn::InstructionId, u64, usize)>> =
            HashMap::default();
        let mut mutable_cells: HashSet<(u64, usize)> = HashSet::default();
        for &fid in &fun_ids {
            for block in FunctionBody::from_id(ctx, fid).blocks() {
                for insn in block.iter() {
                    let (ptr, space, size) = match insn.mnemonic() {
                        Mnemonic::Load(l) => (l.ptr, l.space, l.size),
                        Mnemonic::Store(s) => (s.ptr, s.space, s.size),
                        _ => continue,
                    };
                    let Some((addr, data_size)) = const_ram_access(ctx, fid, ptr, space, size)
                    else {
                        continue;
                    };
                    if written.contains(&addr) {
                        // Written somewhere → a value-threaded cell.
                        mutable_cells.insert((addr, data_size));
                        continue;
                    }
                    // Read-only candidate: fold the load (only loads produce a
                    // value; a lone read-only store is dead and left to DCE).
                    if let Mnemonic::Load(_) = insn.mnemonic() {
                        folds
                            .entry(fid)
                            .or_default()
                            .push((insn.id, addr, data_size));
                    }
                }
            }
        }

        // Phase 3a: apply read-only folds (needs the binary image).
        let mut changed: HashSet<FunctionId> = HashSet::default();
        if let Some(binary) = binary {
            for (&fid, loads) in &folds {
                for &(load_id, addr, size) in loads {
                    // Section the loader may rewrite (GOT/PLT): not a reliable
                    // constant even absent a syntactic store.
                    if binary.is_known_writable(addr) {
                        continue;
                    }
                    let Some(value) = binary.read_uint(addr, size) else {
                        continue;
                    };
                    if !ctx.assume_true(Proposition::ImmutableMemory {
                        addr,
                        size: size as u8,
                    }) {
                        continue;
                    }
                    let konst = ctx.get_const(value, size).id();
                    ctx.replace_all_uses_with(ValueId::Instruction(load_id), konst);
                    ctx.remove_instruction(load_id);
                    changed.insert(fid);
                }
            }
        }

        // Phase 3b: pre-mint identity varnodes for every mutable global cell so
        // the register effect scan can look them up.
        for (addr, size) in mutable_cells {
            ctx.get_or_make_global_varnode(addr, size);
        }

        Ok(crate::ModulePassOutcome {
            module_changed: !changed.is_empty(),
            changed_functions: changed,
            type_requests: Vec::new(),
            preserved_analyses: crate::PreservedAnalyses::none(),
        })
    }
}

crate::register_module_pass!(PromoteGlobalCells);
