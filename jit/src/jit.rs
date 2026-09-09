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
    BlockExecutor, Executed, VmMemory, qcode_jit_load, qcode_jit_sdiv128, qcode_jit_srem128,
    qcode_jit_store, qcode_jit_udiv128, qcode_jit_urem128,
};

use crate::compile::{BLOCK_OK, BlockTranslator, Export, Helpers, SpaceTable, Unsupported};

/// What is known about one block: the index of its native code, or the reason
/// the compiler declined it, together with the instruction count that answer
/// was reached for.
type CacheEntry = Option<(usize, Result<usize, Unsupported>)>;

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
    /// The terminator operands this block computes, in slot order.
    exports: Vec<Export>,
    /// How many body instructions the native code retires: everything but
    /// the terminator, or everything before the first interrupting user op.
    /// The caller needs it to position the interpreter, and taking it from
    /// here saves resolving the block through the module arena again.
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
    /// Each entry records the instruction count it was made for, because a
    /// block is *not* immutable here: a VM that lifts on demand first presents
    /// an empty placeholder (which the compiler rightly declines), then fills
    /// it, then runs a cleanup pass over it. Trusting the entry regardless of
    /// count would freeze that first decline forever and the block would never
    /// be compiled.
    ///
    /// Stored as slots indexed by the block id's two components rather than in
    /// a map: this is read on *every* block execution, and hashing a
    /// `(function, local)` pair each time was a measurable share of run time.
    /// Sparse ids cost only an unused slot.
    cache: Vec<Vec<CacheEntry>>,
    /// Reused across runs so a hot block does not allocate to be entered.
    scratch: Vec<*mut u8>,
    /// Likewise for the export buffer compiled code writes its terminator
    /// operands into.
    exports: Vec<u64>,
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
                divisions,
            },
            compiled: Vec::new(),
            cache: Vec::new(),
            scratch: Vec::new(),
            exports: Vec::new(),
            stats: JitStats::default(),
        }
    }

    /// Whether `block` has native code, compiling it on first sight.
    ///
    /// A decline is remembered, so an unsupported block costs one compilation
    /// attempt over the life of the machine rather than one per execution.
    fn resolve(&mut self, ctx: &Context<'_>, block: BlockId) -> Result<usize, Unsupported> {
        let count = ctx.block(block).instruction_ids().len();
        let func: usize = block.func.into();
        let local: usize = block.local.into();
        if let Some(Some((cached_count, known))) =
            self.cache.get(func).and_then(|slots| slots.get(local))
            && *cached_count == count
        {
            return known.clone();
        }

        let outcome = self.compile(ctx, block);
        match &outcome {
            Ok(_) => self.stats.compiled += 1,
            Err(_) => self.stats.declined += 1,
        }
        if func >= self.cache.len() {
            self.cache.resize_with(func + 1, Vec::new);
        }
        let slots = &mut self.cache[func];
        if local >= slots.len() {
            slots.resize(local + 1, None);
        }
        slots[local] = Some((count, outcome.clone()));
        outcome
    }

    fn compile(&mut self, ctx: &Context<'_>, block: BlockId) -> Result<usize, Unsupported> {
        let full_body = ctx.block(block).instruction_ids().len().saturating_sub(1);
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
            divisions: self
                .helpers
                .divisions
                .map(|id| self.module.declare_func_in_func(id, &mut context.func)),
        };

        // A fresh builder context per attempt: a declined block abandons its
        // half-built function, which would leave a shared context dirty and
        // trip Cranelift's emptiness assertion on the next compilation.
        let mut builder_ctx = FunctionBuilderContext::new();
        let (table, exports, body_len) = {
            let mut builder = FunctionBuilder::new(&mut context.func, &mut builder_ctx);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let mut translator = BlockTranslator::new(ctx, builder, entry, helpers);
            match translator.translate_body(block) {
                Ok(body_len) => {
                    let compiled = (
                        translator.table.clone(),
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
            exports,
            body_len,
            interrupts: body_len < full_body,
            slots: Vec::new(),
        });
        Ok(self.compiled.len() - 1)
    }

    /// Compiles `block` without running it, reporting why if it is declined.
    ///
    /// For tooling that wants to report coverage over a module.
    pub fn try_compile(&mut self, ctx: &Context<'_>, block: BlockId) -> Result<(), Unsupported> {
        self.resolve(ctx, block).map(|_| ())
    }

    /// Runs `block` as native code, if it has any.
    ///
    /// Returns `Ok(None)` when the block is not compiled, which is the caller's
    /// signal to run it on the interpreter instead, and `Ok(Some(n))` when it
    /// ran, where `n` is the number of body instructions it retired.
    pub fn run_block(
        &mut self,
        ctx: &Context<'_>,
        emu: &mut StandaloneEmulator<VmMemory>,
        block: BlockId,
        chain: bool,
    ) -> Result<Option<Executed>, EmulatorErrorKind> {
        let mut current = block;
        let mut retired = 0;
        loop {
            let Ok(index) = self.resolve(ctx, current) else {
                // Nothing compiled here. If earlier blocks ran, the machine is
                // already at `current`'s start and the interpreter takes over
                // from there; otherwise this call did nothing at all.
                return Ok((retired > 0).then_some(Executed {
                    block: current,
                    body: 0,
                    retired,
                }));
            };
            let (body, interrupts) = self.enter(ctx, emu, index)?;
            retired += body as u64;
            self.stats.native_runs += 1;

            // Only continue while the successor is one this backend can also
            // run: deciding the branch here is what keeps control inside
            // compiled code, and the interpreter would otherwise redo it. A
            // body cut at an interrupting op has no successor to decide: the
            // interpreter takes over at the op.
            let next = if chain && !interrupts {
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
        }
    }

    /// Whether `block` has native code, without compiling it.
    fn is_compiled(&mut self, ctx: &Context<'_>, block: BlockId) -> bool {
        self.resolve(ctx, block).is_ok()
    }

    /// The successor this block's terminator selects, when that is a decision
    /// the backend can make: an argument-less branch, or a conditional one
    /// whose condition the compiled body has just exported.
    ///
    /// `None` means "leave it to the interpreter" — an indirect branch, a call,
    /// a return, or any edge that binds block arguments, all of which stay in
    /// one implementation.
    fn next_block(
        &self,
        ctx: &Context<'_>,
        emu: &StandaloneEmulator<VmMemory>,
        block: BlockId,
    ) -> Option<BlockId> {
        let &terminator = ctx.block(block).instruction_ids().last()?;
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
            _ => return None,
        };
        Some(BlockId::new(block.func, target))
    }

    /// Runs one compiled block, leaving the operands of whatever the
    /// interpreter runs next where it would have put them. Returns how many
    /// body instructions it retired, and whether it stopped short at an
    /// interrupting op.
    fn enter(
        &mut self,
        _ctx: &Context<'_>,
        emu: &mut StandaloneEmulator<VmMemory>,
        index: usize,
    ) -> Result<(usize, bool), EmulatorErrorKind> {
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
        self.exports.clear();
        self.exports.resize(compiled.exports.len(), 0);
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
        for (export, &bits) in compiled.exports.iter().zip(&self.exports) {
            emu.insn_values
                .insert(export.insn, SizedValue::new(bits, export.size));
        }

        Ok((compiled.body_len, compiled.interrupts))
    }
}

/// Lets a [`Jit`] be installed on a machine as its block executor.
impl BlockExecutor for Jit {
    fn run_block(
        &mut self,
        ctx: &Context<'_>,
        emu: &mut StandaloneEmulator<VmMemory>,
        block: BlockId,
        chain: bool,
    ) -> Result<Option<Executed>, EmulatorErrorKind> {
        Jit::run_block(self, ctx, emu, block, chain)
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
