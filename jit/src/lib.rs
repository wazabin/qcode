//! A Cranelift JIT backend for QCode: an execution strategy alongside the
//! interpreter, not a replacement for it.
//!
//! [`qcode_emulator`] interprets QCode one operation at a time, which means
//! every intermediate value is materialised into its value table and every
//! operand is resolved through the module. Compiled code does neither: a QCode
//! block is already SSA, so it maps onto Cranelift's SSA directly and values
//! consumed inside the block stay in machine registers.
//!
//! The backend is deliberately partial — see [`compile::Unsupported`]. A block
//! it declines is run by the interpreter instead, so coverage can grow without
//! ever being a correctness question.//!
//! # Installing it
//!
//! The JIT is a [`qcode_vm`] block executor; a machine runs identically with
//! or without it, only faster.
//!
//! ```no_run
//! use qcode_jit::Jit;
//! use qcode_vm::{Vm, VmMemory, perm};
//! use wazabin_qcode_sleigh::vm_source::SleighCodeSource;
//!
//! # fn run(code: &[u8]) {
//! let source = SleighCodeSource::new(sleigh_precompile::x64::spec());
//! let ctx = source.new_context();
//!
//! let mut memory = VmMemory::new();
//! memory.mmu.write_unchecked(0x1000, code, perm::READ | perm::EXEC);
//!
//! let mut vm = Vm::at_address(ctx, 0x1000, source, memory).expect("the entry decodes");
//! vm.set_block_executor(Box::new(Jit::new()));
//! vm.run(10_000);
//!
//! // How much the JIT actually took on, rather than handed back.
//! println!("compiled {} blocks natively", vm.stats.native_bodies);
//! # }
//! ```
//!
//! [`qcode_vm`]: https://docs.rs/qcode_vm

pub mod compile;
pub mod jit;

pub use compile::{Export, SpaceTable, Unsupported};
pub use jit::{Jit, JitStats};
