//! Owning, caching and running compiled blocks.

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use qcode::{
    context::Context,
    value::{
        BlockId, ValueId,
        insn::{InstructionId, Mnemonic},
    },
};
use qcode_emulator::{EmulatorErrorKind, SizedValue, StandaloneEmulator};
use qcode_vm::{
    BlockExecutor, Executed, VmMemory, flat::FlatSpaces, qcode_jit_load, qcode_jit_overflow,
    qcode_jit_sdiv128, qcode_jit_srem128, qcode_jit_store, qcode_jit_udiv128, qcode_jit_urem128,
};
use rustc_hash::FxHashMap;

use crate::compile::{
    BLOCK_OK, BLOCK_OVERFLOW, BlockTranslator, Export, Helpers, SpaceTable, Unsupported,
};

/// What is known about one block, together with the block revision it was
/// learnt for.
type CacheEntry = Option<(u64, Known)>;

/// What the JIT knows about a block at a revision.
#[derive(Debug, Clone)]
enum Known {
    /// Entered this many times without being compiled; see
    /// [`Jit::set_warm_up`].
    Warming(u32),
    /// The index of its native code, or the reason the compiler declined it.
    Settled(Result<usize, Unsupported>),
}

/// A block that has been compiled to native code.
struct Compiled {
    /// The compiled body. Its arguments are the base of an array of space base
    /// pointers, in the order [`SpaceTable`] records, the base of the export
    /// buffer (one `u64` slot per entry of `exports`), the base of the guest's
    /// software TLB, and the `VmMemory` its slow paths call back into. It
    /// returns [`BLOCK_OK`], or [`BLOCK_FAULT`](crate::compile::BLOCK_FAULT) if
    /// an access faulted and the block stopped there.
    entry: extern "C" fn(*const *mut u8, *mut u64, *mut u8, *mut VmMemory) -> i32,
    table: SpaceTable,
    /// The results of earlier instructions this code reads on entry, in the
    /// first value-buffer slots. Empty for a block entered at its start.
    imports: Vec<Export>,
    /// The operands of what the interpreter runs next, in the slots after the
    /// imports.
    exports: Vec<Export>,
    /// The body index the interpreter continues from once the native code has
    /// run: the terminator's, or the first interrupting user op's. Taking it
    /// from here saves resolving the block through the module arena again.
    body_len: usize,
    /// Whether the body stops short at an interrupting op. Such a block ends
    /// in the interpreter's hands, so it is never chained past.
    interrupts: bool,
    /// Where each of `table`'s spaces lives in the machine's flat storage.
    ///
    /// Resolved on first execution and kept: a slot is stable for the life of
    /// the spaces, so re-entering a hot block costs an array index per space
    /// rather than a map lookup.
    slots: Vec<usize>,
}

/// How much work the JIT is taking, and how much it is declining.
#[derive(Debug, Default, Clone)]
pub struct JitStats {
    /// Blocks translated to native code.
    pub compiled: u64,
    /// Blocks the compiler declined; these run on the interpreter.
    pub declined: u64,
    /// Executions that ran as native code.
    pub native_runs: u64,
}

/// A JIT backend: compiles blocks on first use and runs them thereafter.
///
/// Holding the [`JITModule`] means compiled code lives as long as this does.
pub struct Jit {
    module: JITModule,
    /// The runtime's slow-path accessors, declared once and referenced by every
    /// compiled block.
    helpers: Helpers,
    /// Compiled blocks, indexed by the handles in `cache`.
    compiled: Vec<Compiled>,
    /// What is known about each block: an index into `compiled`, or the reason
    /// it was declined. Declining is cached too, so a block the compiler cannot
    /// take is only examined once.
    ///
    /// Each entry records the block revision it was made for, because a
    /// block is *not* immutable here: a VM that lifts on demand first presents
    /// an empty placeholder (which the compiler rightly declines), then fills
    /// it, runs a cleanup pass over it, and may later empty it and lift it
    /// again from the same bytes — as many instructions as before, under new
    /// ids. Trusting the entry regardless would freeze that first decline
    /// forever, and code compiled from the earlier ids would read and write
    /// the interpreter's value table by ids that no longer mean anything.
    ///
    /// Stored as slots indexed by the block id's two components rather than in
    /// a map: this is read on *every* block execution, and hashing a
    /// `(function, local)` pair each time was a measurable share of run time.
    /// Sparse ids cost only an unused slot.
    cache: Vec<Vec<CacheEntry>>,
    /// The same, for entries part-way into a block — the continuation after
    /// an interrupt. Rare enough to hash.
    partial: FxHashMap<(BlockId, usize), (u64, Result<usize, Unsupported>)>,
    /// Reused across runs so a hot block does not allocate to be entered.
    scratch: Vec<*mut u8>,
    /// Likewise for the export buffer compiled code writes its terminator
    /// operands into.
    exports: Vec<u64>,
    /// The entry at which a block is compiled; see [`set_warm_up`]
    /// (Self::set_warm_up).
    warm_up: u32,
    pub stats: JitStats,
}

impl Default for Jit {
    fn default() -> Self {
        Self::new()
    }
}

impl Jit {
    pub fn new() -> Self {
        let mut flags = settings::builder();
        // Compilation happens on the guest's critical path, so favour getting
        // through it over the last few percent of code quality.
        flags
            .set("opt_level", "speed")
            .expect("opt_level is a known flag");
        // The verifier re-checks Cranelift IR this backend has just built, on
        // the guest's critical path, for every block. It is a development aid
        // for the compiler itself; what guards *this* translation is the
        // divergence harness, which compares compiled code against the
        // interpreter block by block over whole programs.
        flags
            .set("enable_verifier", "false")
            .expect("enable_verifier is a known flag");
        let isa = cranelift_native::builder()
            .expect("host is a supported target")
            .finish(settings::Flags::new(flags))
            .expect("isa builds for the host");
        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        // The two calls compiled code makes. Registered by name because that is
        // how Cranelift resolves an external function; the addresses are this
        // process's own, so there is no dynamic loading involved.
        builder.symbol("qcode_jit_load", qcode_jit_load as *const u8);
        builder.symbol("qcode_jit_store", qcode_jit_store as *const u8);
        builder.symbol("qcode_jit_overflow", qcode_jit_overflow as *const u8);
        builder.symbol("qcode_jit_udiv128", qcode_jit_udiv128 as *const u8);
        builder.symbol("qcode_jit_urem128", qcode_jit_urem128 as *const u8);
        builder.symbol("qcode_jit_sdiv128", qcode_jit_sdiv128 as *const u8);
        builder.symbol("qcode_jit_srem128", qcode_jit_srem128 as *const u8);
        let mut module = JITModule::new(builder);

        let mut load_sig = module.make_signature();
        // (memory, address, size, out) -> status
        load_sig.params.push(AbiParam::new(types::I64));
        load_sig.params.push(AbiParam::new(types::I64));
        load_sig.params.push(AbiParam::new(types::I32));
        load_sig.params.push(AbiParam::new(types::I64));
        load_sig.returns.push(AbiParam::new(types::I32));
        let load = module
            .declare_function("qcode_jit_load", Linkage::Import, &load_sig)
            .expect("the load helper declares once");

        let mut store_sig = module.make_signature();
        // (memory, address, size, value) -> status
        store_sig.params.push(AbiParam::new(types::I64));
        store_sig.params.push(AbiParam::new(types::I64));
        store_sig.params.push(AbiParam::new(types::I32));
        store_sig.params.push(AbiParam::new(types::I64));
        store_sig.returns.push(AbiParam::new(types::I32));
        let store = module
            .declare_function("qcode_jit_store", Linkage::Import, &store_sig)
            .expect("the store helper declares once");

        let mut overflow_sig = module.make_signature();
        // (memory, address, size) -> ()
        overflow_sig.params.push(AbiParam::new(types::I64));
        overflow_sig.params.push(AbiParam::new(types::I64));
        overflow_sig.params.push(AbiParam::new(types::I32));
        let overflow = module
            .declare_function("qcode_jit_overflow", Linkage::Import, &overflow_sig)
            .expect("the overflow helper declares once");

        // (a low, a high, b low, b high, out) -> ()
        let mut divide_sig = module.make_signature();
        for _ in 0..5 {
            divide_sig.params.push(AbiParam::new(types::I64));
        }
        let mut wide_division = |name: &str| {
            module
                .declare_function(name, Linkage::Import, &divide_sig)
                .expect("a division helper declares once")
        };
        let divisions = [
            wide_division("qcode_jit_udiv128"),
            wide_division("qcode_jit_urem128"),
            wide_division("qcode_jit_sdiv128"),
            wide_division("qcode_jit_srem128"),
        ];

        Self {
            module,
            helpers: Helpers {
                load,
                store,
                overflow,
                divisions,
            },
            compiled: Vec::new(),
            cache: Vec::new(),
            partial: FxHashMap::default(),
            scratch: Vec::new(),
            exports: Vec::new(),
            warm_up: 2,
            stats: JitStats::default(),
        }
    }

    /// Compiles a block on its `entries`th entry, running it on the
    /// interpreter until then; `1` compiles on first sight, `0` is `1`.
    ///
    /// The default is 2. A block is compiled from the shape it has when it
    /// is entered, and a VM that lifts on demand grows a block by absorbing
    /// the instructions it discovers after it, each absorption a new
    /// revision that has to be compiled again: on straight-line code most
    /// blocks are compiled at least twice, and code that runs once — setup,
    /// most of a program's text — is compiled for nothing, at a cost of tens
    /// of microseconds per block against the microsecond of interpreting it.
    /// Waiting for the second entry compiles a block once it has been run
    /// through, and only when something comes back to it; on the Embench
    /// images that is a quarter to a third off the whole run.
    pub fn set_warm_up(&mut self, entries: u32) {
        self.warm_up = entries.max(1);
    }

    /// Whether `block` has native code, on one more entry to it: compiled
    /// once the entries reach the [warm-up](Self::set_warm_up), declined
    /// with [`Unsupported::Cold`] before that.
    ///
    /// A decline is remembered, so an unsupported block costs one compilation
    /// attempt over the life of the machine rather than one per execution.
    fn resolve(
        &mut self,
        ctx: &Context<'_>,
        flat: &FlatSpaces,
        block: BlockId,
        start: usize,
    ) -> Result<usize, Unsupported> {
        let revision = ctx.block(block).revision();
        if start != 0 {
            // A continuation is compiled at once: the block is one that is
            // running, and the interpreter has just been through part of it.
            if let Some((cached_revision, known)) = self.partial.get(&(block, start))
                && *cached_revision == revision
            {
                return known.clone();
            }
            let outcome = self.compile(ctx, flat, block, start);
            match &outcome {
                Ok(_) => self.stats.compiled += 1,
                Err(_) => self.stats.declined += 1,
            }
            self.partial
                .insert((block, start), (revision, outcome.clone()));
            return outcome;
        }
        let entries = match self.slot(block) {
            Some((cached_revision, known)) if *cached_revision == revision => match known {
                Known::Settled(known) => return known.clone(),
                Known::Warming(entries) => *entries + 1,
            },
            _ => 1,
        };
        if entries < self.warm_up {
            *self.slot_mut(block) = Some((revision, Known::Warming(entries)));
            return Err(Unsupported::Cold);
        }
        self.settle(ctx, flat, block, revision)
    }

    /// Compiles `block` at `revision` from its start, whatever its entries,
    /// and remembers the outcome.
    fn settle(
        &mut self,
        ctx: &Context<'_>,
        flat: &FlatSpaces,
        block: BlockId,
        revision: u64,
    ) -> Result<usize, Unsupported> {
        let outcome = self.compile(ctx, flat, block, 0);
        match &outcome {
            Ok(_) => self.stats.compiled += 1,
            Err(_) => self.stats.declined += 1,
        }
        *self.slot_mut(block) = Some((revision, Known::Settled(outcome.clone())));
        outcome
    }

    fn slot(&self, block: BlockId) -> Option<&(u64, Known)> {
        let func: usize = block.func.into();
        let local: usize = block.local.into();
        self.cache.get(func)?.get(local)?.as_ref()
    }

    fn slot_mut(&mut self, block: BlockId) -> &mut CacheEntry {
        let func: usize = block.func.into();
        let local: usize = block.local.into();
        if func >= self.cache.len() {
            self.cache.resize_with(func + 1, Vec::new);
        }
        let slots = &mut self.cache[func];
        if local >= slots.len() {
            slots.resize(local + 1, None);
        }
        &mut slots[local]
    }

    fn compile(
        &mut self,
        ctx: &Context<'_>,
        flat: &FlatSpaces,
        block: BlockId,
        start: usize,
    ) -> Result<usize, Unsupported> {
        let full_body = ctx.block(block).insn_count().saturating_sub(1);
        let mut signature = self.module.make_signature();
        // spaces, exports, tlb, memory.
        for _ in 0..4 {
            signature.params.push(AbiParam::new(types::I64));
        }
        signature.returns.push(AbiParam::new(types::I32));

        let name = format!("qcode_block_{}_{}", self.compiled.len(), self.cache.len());
        let id = self
            .module
            .declare_function(&name, Linkage::Export, &signature)
            .map_err(|_| Unsupported::Mnemonic("function declaration failed"))?;

        let mut context = self.module.make_context();
        context.func.signature = signature;
        let helpers = crate::compile::HelperRefs {
            load: self
                .module
                .declare_func_in_func(self.helpers.load, &mut context.func),
            store: self
                .module
                .declare_func_in_func(self.helpers.store, &mut context.func),
            overflow: self
                .module
                .declare_func_in_func(self.helpers.overflow, &mut context.func),
            divisions: self
                .helpers
                .divisions
                .map(|id| self.module.declare_func_in_func(id, &mut context.func)),
        };

        // A fresh builder context per attempt: a declined block abandons its
        // half-built function, which would leave a shared context dirty and
        // trip Cranelift's emptiness assertion on the next compilation.
        let mut builder_ctx = FunctionBuilderContext::new();
        let (table, imports, exports, body_len) = {
            let mut builder = FunctionBuilder::new(&mut context.func, &mut builder_ctx);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let mut translator = BlockTranslator::new(ctx, flat, builder, entry, helpers);
            match translator.translate_body(block, start) {
                Ok(body_len) => {
                    let compiled = (
                        translator.table.clone(),
                        translator.imports.clone(),
                        translator.exports.clone(),
                        body_len,
                    );
                    translator.finish();
                    compiled
                }
                Err(unsupported) => {
                    // The half-built function is simply dropped; nothing was
                    // defined in the module, so there is nothing to undo.
                    self.module.clear_context(&mut context);
                    return Err(unsupported);
                }
            }
        };

        self.module
            .define_function(id, &mut context)
            .map_err(|_| Unsupported::Mnemonic("function definition failed"))?;
        self.module.clear_context(&mut context);
        self.module
            .finalize_definitions()
            .map_err(|_| Unsupported::Mnemonic("finalization failed"))?;

        let code = self.module.get_finalized_function(id);
        // SAFETY: `code` is the entry point Cranelift just finalized for the
        // signature declared above — four pointer-width arguments and a 32-bit
        // status result.
        let entry = unsafe {
            std::mem::transmute::<
                *const u8,
                extern "C" fn(*const *mut u8, *mut u64, *mut u8, *mut VmMemory) -> i32,
            >(code)
        };

        self.compiled.push(Compiled {
            entry,
            table,
            imports,
            exports,
            body_len,
            interrupts: body_len < full_body,
            slots: Vec::new(),
        });
        Ok(self.compiled.len() - 1)
    }

    /// Compiles `block` without running it, reporting why if it is declined.
    ///
    /// For tooling that wants to report coverage over a module. With no
    /// machine in hand, no flat space is bounded, so a computed address into
    /// one is declined here where [`run_block`](Self::run_block) may compile
    /// it.
    pub fn try_compile(&mut self, ctx: &Context<'_>, block: BlockId) -> Result<(), Unsupported> {
        let revision = ctx.block(block).revision();
        match self.slot(block) {
            Some((cached_revision, Known::Settled(known))) if *cached_revision == revision => {
                known.clone().map(|_| ())
            }
            _ => self
                .settle(ctx, &FlatSpaces::default(), block, revision)
                .map(|_| ()),
        }
    }

    /// Runs `block` as native code from body index `start`, if it has any.
    ///
    /// Returns `Ok(None)` when the block is not compiled, which is the caller's
    /// signal to run it on the interpreter instead, and `Ok(Some(_))` when it
    /// ran, saying where the interpreter continues.
    pub fn run_block(
        &mut self,
        ctx: &Context<'_>,
        emu: &mut StandaloneEmulator<VmMemory>,
        block: BlockId,
        start: usize,
        chain: u64,
    ) -> Result<Option<Executed>, EmulatorErrorKind> {
        let mut current = block;
        let mut from = start;
        let mut retired = 0;
        loop {
            let Ok(index) = self.resolve(ctx, emu.memory.flat(), current, from) else {
                // Nothing compiled here. If earlier blocks ran, the machine is
                // already at `current`'s start and the interpreter takes over
                // from there; otherwise this call did nothing at all.
                return Ok((retired > 0).then_some(Executed {
                    block: current,
                    body: 0,
                    retired,
                }));
            };
            let Some((body, interrupts)) = self.enter(ctx, emu, index)? else {
                // An import the interpreter never produced: not this backend's
                // block to run right now.
                return Ok((retired > 0).then_some(Executed {
                    block: current,
                    body: 0,
                    retired,
                }));
            };
            retired += (body - from) as u64;
            self.stats.native_runs += 1;

            // Only continue while the successor is one this backend can also
            // run: deciding the branch here is what keeps control inside
            // compiled code, and the interpreter would otherwise redo it. A
            // body cut at an interrupting op has no successor to decide: the
            // interpreter takes over at the op. And only within the caller's
            // allowance, or a loop compiled whole would never hand back.
            // Nor past a block that wrote over lifted code: the successor may
            // be what it wrote, and the VM has to see the write first.
            let next = if retired < chain && !interrupts && !emu.memory.mmu.code_written() {
                self.next_block(ctx, emu, current)
            } else {
                None
            };
            let Some(next) = next.filter(|&next| self.is_compiled(ctx, next)) else {
                return Ok(Some(Executed {
                    block: current,
                    body,
                    retired,
                }));
            };
            // Chaining means this block's terminator was decided here rather
            // than by the interpreter, so it is retired work nobody else will
            // count. Only the *last* block's terminator is left to the caller.
            retired += 1;
            current = next;
            from = 0;
        }
    }

    /// Whether `block` has native code from its start, without compiling it
    /// or counting an entry: a block still warming is the interpreter's to
    /// enter, and that entry is what [`resolve`](Self::resolve) counts.
    fn is_compiled(&self, ctx: &Context<'_>, block: BlockId) -> bool {
        matches!(
            self.slot(block),
            Some((revision, Known::Settled(Ok(_)))) if *revision == ctx.block(block).revision()
        )
    }

    /// The successor this block's terminator selects, when that is a decision
    /// the backend can make: an argument-less branch, a conditional one whose
    /// condition the compiled body has just exported, or an indirect one whose
    /// pointer it exported and that resolves to a block already lifted.
    ///
    /// The indirect case is what guest calls and returns become under the
    /// VM's flat lifting — `ret` is a `branchind` on the popped address — so
    /// without it every return handed control back to the interpreter, and on
    /// call-heavy code that round trip was most of the run. The pointer is
    /// resolved through the emulator's own address index, the same lookup the
    /// interpreter's `branchind` makes; an address it does not know is left to
    /// the interpreter, whose failure to resolve it is what triggers discovery.
    ///
    /// `None` means "leave it to the interpreter" — a call, a return, or any
    /// edge that binds block arguments, all of which stay in one
    /// implementation.
    fn next_block(
        &self,
        ctx: &Context<'_>,
        emu: &mut StandaloneEmulator<VmMemory>,
        block: BlockId,
    ) -> Option<BlockId> {
        let terminator = ctx.block(block).last_insn()?;
        let terminator = InstructionId::new(block.func, terminator);
        let insn = qcode::value::Instruction::from_id(ctx, terminator);
        let target = match insn.mnemonic() {
            Mnemonic::Branch(branch) if branch.args.is_empty() => branch.target,
            Mnemonic::CBranch(cbranch)
                if cbranch.success_args.is_empty() && cbranch.failure_args.is_empty() =>
            {
                let ValueId::Instruction(condition) = cbranch.condition.qualify(block.func) else {
                    return None;
                };
                let taken = emu.insn_values.get(&condition)?.as_bits() != 0;
                if taken {
                    cbranch.success_block
                } else {
                    cbranch.failure_block
                }
            }
            Mnemonic::BranchInd(branchind) => {
                let ValueId::Instruction(ptr) = branchind.ptr.qualify(block.func) else {
                    return None;
                };
                let addr = emu.insn_values.get(&ptr)?.as_bits() as u64;
                return emu.block_at_address(ctx, addr);
            }
            _ => return None,
        };
        Some(BlockId::new(block.func, target))
    }

    /// Runs one compiled block, leaving the operands of whatever the
    /// interpreter runs next where it would have put them. Returns the body
    /// index the interpreter continues from, and whether the code stopped
    /// short at an interrupting op — or `None` when an import the code needs
    /// is missing from the interpreter's table, in which case nothing ran.
    fn enter(
        &mut self,
        _ctx: &Context<'_>,
        emu: &mut StandaloneEmulator<VmMemory>,
        index: usize,
    ) -> Result<Option<(usize, bool)>, EmulatorErrorKind> {
        let compiled = &mut self.compiled[index];
        // Taken as a raw pointer, and everything below derived from it: the
        // compiled block holds base pointers into the flat spaces *while*
        // calling back into this same `VmMemory` for the RAM accesses it could
        // not settle inline. A live `&mut` spanning the call would make those
        // base pointers ones the compiler is entitled to assume nothing else
        // reaches.
        let memory: *mut VmMemory = &raw mut emu.memory;

        // SAFETY: for every dereference of `memory` here — it points to the
        // emulator's own memory, which outlives this call, and no reference to
        // it is held across any of them.
        if compiled.slots.is_empty() {
            compiled.slots = compiled
                .table
                .entries()
                .iter()
                .map(|&(space, _)| unsafe { (*memory).flat_mut().slot(space) })
                .collect();
        }

        // Each space is grown to the size the block needs *before* its base
        // pointer is taken: growing reallocates, and compiled code holds these
        // pointers for the duration of the call.
        self.scratch.clear();
        for (&slot, &(_, required)) in compiled.slots.iter().zip(compiled.table.entries()) {
            self.scratch
                .push(unsafe { (*memory).flat_mut().base_ptr_at(slot, required)? });
        }
        // The value buffer: imports first, filled from the interpreter's
        // table, then room for the exports.
        self.exports.clear();
        for import in &compiled.imports {
            let Some(value) = emu.insn_values.get(&import.insn) else {
                return Ok(None);
            };
            self.exports.push(value.as_bits() as u64);
        }
        self.exports
            .resize(compiled.imports.len() + compiled.exports.len(), 0);
        // The TLB moves only with the memory itself, which is pinned for the
        // duration of the call.
        let tlb = unsafe { (*memory).mmu.tlb_ptr() };

        // SAFETY: the function was compiled from this block and reads and writes
        // only within the byte ranges recorded in its space table, each of which
        // has just been made addressable, plus the export buffer, which has just
        // been sized to the slot count that same compilation recorded, plus
        // guest RAM through the TLB and the memory it is handed.
        let status = (compiled.entry)(
            self.scratch.as_ptr(),
            self.exports.as_mut_ptr(),
            tlb,
            memory,
        );

        // A faulting access stopped the block where it happened. The fault is
        // left where an interpreted one would be, for the VM to turn into an
        // exit; what goes back from here is only the error the interpreter's
        // own signature can carry.
        if status == BLOCK_OVERFLOW as i32 {
            // SAFETY: as above.
            let (addr, size) = unsafe { (*memory).take_overflow() }.unwrap_or((0, 0));
            return Err(EmulatorErrorKind::AddressOverflow(addr, size));
        }
        if status != BLOCK_OK as i32 {
            // SAFETY: as above.
            let fault = unsafe { (*memory).fault() };
            let fault = fault.ok_or(EmulatorErrorKind::MemoryReadError(0))?;
            return Err(if fault.is_write() {
                EmulatorErrorKind::MemoryWriteError(fault.addr)
            } else {
                EmulatorErrorKind::MemoryReadError(fault.addr)
            });
        }

        // The terminator is still the interpreter's to run, so the operands it
        // reads have to look as though the interpreter had computed them.
        let outputs = &self.exports[compiled.imports.len()..];
        for (export, &bits) in compiled.exports.iter().zip(outputs) {
            emu.insn_values
                .insert(export.insn, SizedValue::new(bits, export.size));
        }

        Ok(Some((compiled.body_len, compiled.interrupts)))
    }
}

/// Lets a [`Jit`] be installed on a machine as its block executor.
impl BlockExecutor for Jit {
    fn run_block(
        &mut self,
        ctx: &Context<'_>,
        emu: &mut StandaloneEmulator<VmMemory>,
        block: BlockId,
        start: usize,
        chain: u64,
    ) -> Result<Option<Executed>, EmulatorErrorKind> {
        Jit::run_block(self, ctx, emu, block, start, chain)
    }

    fn invalidate(&mut self) {
        // The machine code itself stays: `compiled` is only ever reached
        // through these two tables, and a `JITModule` cannot free a function
        // anyway.
        self.cache.clear();
        self.partial.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_jit_has_compiled_nothing() {
        let jit = Jit::new();
        assert_eq!(jit.stats.compiled, 0);
        assert_eq!(jit.stats.declined, 0);
        assert_eq!(jit.stats.native_runs, 0);
    }
}
