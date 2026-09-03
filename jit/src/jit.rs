//! Owning, caching and running compiled blocks.

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use qcode::{context::Context, value::BlockId};
use qcode_emulator::{EmulatorErrorKind, SizedValue, StandaloneEmulator};
use qcode_vm::{BlockExecutor, VmMemory};

use crate::compile::{BlockTranslator, Export, SpaceTable, Unsupported};

/// What is known about one block: the index of its native code, or the reason
/// the compiler declined it, together with the instruction count that answer
/// was reached for.
type CacheEntry = Option<(usize, Result<usize, Unsupported>)>;

/// A block that has been compiled to native code.
struct Compiled {
    /// The compiled body. Its arguments are the base of an array of space base
    /// pointers, in the order [`SpaceTable`] records, and the base of the
    /// export buffer, one `u64` slot per entry of `exports`.
    entry: extern "C" fn(*const *mut u8, *mut u64),
    table: SpaceTable,
    /// The terminator operands this block computes, in slot order.
    exports: Vec<Export>,
    /// How many body instructions this block has — everything but the
    /// terminator. The caller needs it to position the interpreter, and taking
    /// it from here saves resolving the block through the module arena again.
    body_len: usize,
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
        let isa = cranelift_native::builder()
            .expect("host is a supported target")
            .finish(settings::Flags::new(flags))
            .expect("isa builds for the host");
        let module = JITModule::new(JITBuilder::with_isa(
            isa,
            cranelift_module::default_libcall_names(),
        ));
        Self {
            module,
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
        let body_len = ctx.block(block).instruction_ids().len().saturating_sub(1);
        let mut signature = self.module.make_signature();
        signature.params.push(AbiParam::new(types::I64));
        signature.params.push(AbiParam::new(types::I64));

        let name = format!("qcode_block_{}_{}", self.compiled.len(), self.cache.len());
        let id = self
            .module
            .declare_function(&name, Linkage::Export, &signature)
            .map_err(|_| Unsupported::Mnemonic("function declaration failed"))?;

        let mut context = self.module.make_context();
        context.func.signature = signature;

        // A fresh builder context per attempt: a declined block abandons its
        // half-built function, which would leave a shared context dirty and
        // trip Cranelift's emptiness assertion on the next compilation.
        let mut builder_ctx = FunctionBuilderContext::new();
        let (table, exports) = {
            let mut builder = FunctionBuilder::new(&mut context.func, &mut builder_ctx);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let mut translator = BlockTranslator::new(ctx, builder, entry);
            match translator.translate_body(block) {
                Ok(()) => {
                    let compiled = (translator.table.clone(), translator.exports.clone());
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
        // signature declared above — two pointer arguments, no return value.
        let entry = unsafe {
            std::mem::transmute::<*const u8, extern "C" fn(*const *mut u8, *mut u64)>(code)
        };

        self.compiled.push(Compiled {
            entry,
            table,
            exports,
            body_len,
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
    ) -> Result<Option<usize>, EmulatorErrorKind> {
        let Ok(index) = self.resolve(ctx, block) else {
            return Ok(None);
        };

        let compiled = &mut self.compiled[index];
        let flat = emu.memory.flat_mut();
        if compiled.slots.is_empty() {
            compiled.slots = compiled
                .table
                .entries()
                .iter()
                .map(|&(space, _)| flat.slot(space))
                .collect();
        }

        // Each space is grown to the size the block needs *before* its base
        // pointer is taken: growing reallocates, and compiled code holds these
        // pointers for the duration of the call.
        self.scratch.clear();
        for (&slot, &(_, required)) in compiled.slots.iter().zip(compiled.table.entries()) {
            self.scratch.push(flat.base_ptr_at(slot, required)?);
        }
        self.exports.clear();
        self.exports.resize(compiled.exports.len(), 0);

        // SAFETY: the function was compiled from this block and reads and writes
        // only within the byte ranges recorded in its space table, each of which
        // has just been made addressable, plus the export buffer, which has just
        // been sized to the slot count that same compilation recorded.
        (compiled.entry)(self.scratch.as_ptr(), self.exports.as_mut_ptr());

        // The terminator is still the interpreter's to run, so the operands it
        // reads have to look as though the interpreter had computed them.
        for (export, &bits) in compiled.exports.iter().zip(&self.exports) {
            emu.insn_values
                .insert(export.insn, SizedValue::new(bits, export.size));
        }

        self.stats.native_runs += 1;
        Ok(Some(compiled.body_len))
    }
}

/// Lets a [`Jit`] be installed on a machine as its block executor.
impl BlockExecutor for Jit {
    fn run_block(
        &mut self,
        ctx: &Context<'_>,
        emu: &mut StandaloneEmulator<VmMemory>,
        block: BlockId,
    ) -> Result<Option<usize>, EmulatorErrorKind> {
        Jit::run_block(self, ctx, emu, block)
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
