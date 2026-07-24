//! `harvest_code_pointers`: value-set discovery over constant pointers.
//!
//! The dominant indirect-call shape in real programs is **vtable / table
//! dispatch** — `call [obj->method]` where `obj` is a *dynamic* pointer, so the
//! call site itself never carries a constant address (measured: 98 of 99 indirect
//! calls in `as`, and neither the param path nor call-site load resolution
//! reaches them). But the function-pointer *values* are constants materialized in
//! code at their **table/setup sites** (`mov $bfd_elf64_x86_64_vec, …`).
//!
//! So instead of resolving call sites, this pass runs a small value-set fixpoint
//! over constants and the immutable image:
//!
//! 1. Every code literal that lands in **executable** memory is a function entry
//!    → seed a [`Discovery`].
//! 2. Every code literal that lands in **read-only data** is a candidate table →
//!    read consecutive pointer-width words: executable words are function entries;
//!    read-only words are nested tables (a vector-of-vtables), pushed onto the
//!    worklist. A word that is neither ends the table.
//!
//! Only tables the code actually *references* are read, which keeps this precise
//! (~0.99 on `as`) unlike a blind data scan. Shares the constant→`Discovery` leaf
//! and the `CodePointer` discovery reason with the param path
//! ([`propagate_code_pointer_args`](super::code_pointer_args)); both feed the
//! discovery fixpoint.

use std::collections::BTreeSet;

use rustc_hash::FxHashSet;

use qcode::{
    address_index::AddressIndex,
    discovery::{Discovery, FunctionDiscoveryReason},
    value::{FunctionBody, ValueId, insn::Mnemonic},
};

use crate::{Pass, PipelineEnv};

/// Cap on words read from a single table (stops runaway walks over zero regions).
const MAX_TABLE_WORDS: usize = 4096;
/// Global cap on image words read per pass invocation.
const READ_BUDGET: usize = 1_000_000;

#[derive(Default)]
pub struct HarvestCodePointers;

impl Pass for HarvestCodePointers {
    const NAME: &'static str = "harvest_code_pointers";

    fn description(&self) -> &'static str {
        "Seed functions from constant code pointers and read-only pointer tables"
    }

    fn run(
        &self,
        cone: &mut crate::ConeMut,
        env: &PipelineEnv,
    ) -> Result<crate::ModulePassOutcome, String> {
        let Some(binary) = env.binary.clone() else {
            return Ok(crate::ModulePassOutcome::module_if(false));
        };
        let binary = binary.as_ref();
        let psize = usize::from(env.cfg.bitness) / 8;
        if psize == 0 {
            return Ok(crate::ModulePassOutcome::module_if(false));
        }

        let ctx = cone.ctx();
        let addresses = AddressIndex::analyze(ctx);

        // A genuine code pointer targets either an unknown address (an
        // undiscovered function) or an existing function *entry*. An address that
        // is a known *non-entry* block start is internal code — a return address
        // (the lifter materializes the `call` push as a literal store) or a branch
        // target — never a function pointer.
        //
        // The stronger discriminator: a real entry begins with a recognizable
        // prologue. On x86-64 ~99.9% of functions open with `push %rbp` (0x55) or
        // `endbr64` (f3 0f 1e fa); a stored mid-function address (the 48% false
        // rate of the raw literal harvest) does not. Require both an entry-shaped
        // index position and a prologue.
        let has_prologue = |v: u64| {
            binary.read_uint(v, 1) == Some(0x55) || binary.read_uint(v, 4) == Some(0xfa1e_0ff3)
        };
        let is_code_ptr = |v: u64| {
            binary.is_executable(v)
                && (addresses.function_at(v).is_some() || addresses.block_at(v).is_none())
                && has_prologue(v)
        };

        // Whether `addr` names a readable, non-executable, non-writable image byte:
        // a constant table entry the loader will not rewrite.
        let is_ro = |addr: u64| {
            !binary.is_executable(addr)
                && !binary.is_known_writable(addr)
                && binary.read_uint(addr, psize).is_some()
        };

        // Phase 1: harvest constant seeds from literals in genuine pointer-
        // materialization positions — a value *stored to memory* (`mov $vec,(obj)`
        // table setup) or *passed as a call argument* (`f(handler)`). Iterating
        // the whole interned literal pool over-seeds catastrophically: the IR mints
        // thousands of intermediate constants that merely fall in the code address
        // range but are not pointers.
        let mut funcs: BTreeSet<u64> = BTreeSet::new();
        let mut table_worklist: Vec<u64> = Vec::new();
        let mut seen_tables: FxHashSet<u64> = FxHashSet::default();
        let mut consider = |v: u64, funcs: &mut BTreeSet<u64>, wl: &mut Vec<u64>| {
            if is_code_ptr(v) {
                funcs.insert(v);
            } else if is_ro(v) && seen_tables.insert(v) {
                wl.push(v);
            }
        };
        let literal = |arg: ValueId| match arg {
            ValueId::Literal(lid) if ctx.shared.values.literals[lid].symbolic.is_none() => {
                Some(ctx.shared.values.literals[lid].value)
            }
            _ => None,
        };
        for fid in cone.cone_functions() {
            let f = FunctionBody::from_id(ctx, fid);
            if f.is_external() {
                continue;
            }
            for block in f.blocks() {
                for insn in block.iter() {
                    match insn.mnemonic() {
                        Mnemonic::Store(s) => {
                            if let Some(v) = literal(s.src.qualify(fid)) {
                                consider(v, &mut funcs, &mut table_worklist);
                            }
                        }
                        Mnemonic::Call(c) => {
                            for &a in &c.args {
                                if let Some(v) = literal(a.qualify(fid)) {
                                    consider(v, &mut funcs, &mut table_worklist);
                                }
                            }
                        }
                        Mnemonic::TailCall(c) => {
                            for &a in &c.args {
                                if let Some(v) = literal(a.qualify(fid)) {
                                    consider(v, &mut funcs, &mut table_worklist);
                                }
                            }
                        }
                        Mnemonic::CallInd(c) => {
                            for &a in &c.args {
                                if let Some(v) = literal(a.qualify(fid)) {
                                    consider(v, &mut funcs, &mut table_worklist);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        // Phase 2: walk referenced read-only tables (and nested vector-of-tables).
        let mut budget = READ_BUDGET;
        while let Some(base) = table_worklist.pop() {
            let mut a = base;
            for _ in 0..MAX_TABLE_WORDS {
                if budget == 0 {
                    break;
                }
                budget -= 1;
                let Some(w) = binary.read_uint(a, psize) else {
                    break;
                };
                if binary.is_executable(w) {
                    if is_code_ptr(w) {
                        funcs.insert(w);
                    }
                } else if w == 0 {
                    // Alignment/padding gap: keep scanning within the word cap.
                } else if is_ro(w) {
                    if seen_tables.insert(w) {
                        table_worklist.push(w);
                    }
                } else {
                    // A word that is neither code, padding, nor a table pointer
                    // ends the table.
                    break;
                }
                a += psize as u64;
            }
        }

        // Phase 3: seed discoveries (idempotent; the lifter validates each address).
        let mut changed = false;
        for target in funcs {
            changed |= cone.discover(
                Discovery::function(target)
                    .with_function_reason(FunctionDiscoveryReason::CodePointer)
                    .from_addr(target),
            );
        }

        Ok(crate::ModulePassOutcome::module_if(changed))
    }
}

crate::register_module_pass!(HarvestCodePointers);
