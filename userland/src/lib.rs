//! A Linux x86-64 userland environment on top of the [`qcode_vm`] emulator.
//!
//! The crate loads a static ELF executable into the VM's MMU, builds the SysV
//! initial process stack, and services Linux system calls on the host.
//!
//! # How a system call reaches the host
//!
//! The x86-64 `syscall` instruction lifts, in the SLEIGH x86 specification, to
//! a user-defined p-code operation (`define pcodeop syscall`). The QCode
//! emulator has no semantics for it and the VM stops with
//! [`VmExit::Interrupt`](qcode_vm::VmExit::Interrupt), the machine positioned
//! at the operation with everything before it retired. The environment
//! services the call and resumes with [`Vm::resume`](qcode_vm::Vm::resume).
//! See [`process`] for the exact rule.
//!
//! # Layout
//!
//! - [`loader`]: places an ELF64 image into guest memory.
//! - [`stack`]: builds `argc`/`argv`/`envp`/auxv.
//! - [`fs`]: the file-descriptor table and the (optionally sandboxed) host
//!   filesystem behind it.
//! - [`syscall`]: the Linux system call dispatcher.
//! - [`process`]: the run loop that ties the above to the VM.

pub mod errno;
pub mod fs;
pub mod guest;
pub mod loader;
pub mod process;
pub mod regs;
pub mod stack;
pub mod syscall;

pub use process::{Config, Crash, Process, ProcessExit};
