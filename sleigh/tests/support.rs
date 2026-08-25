#![allow(dead_code, unused_imports)]

use std::sync::OnceLock;

use qcode::{
    context::Context,
    value::{BlockId, FunctionId},
};
use sleigh::{Decoder, Instruction};
use wazabin_qcode_sleigh::{LiftError, SleighLifter};

pub mod x64 {
    use super::*;

    pub use sleigh_precompile::x64::regs::*;

    fn lifter() -> &'static SleighLifter<'static> {
        static LIFTER: OnceLock<SleighLifter<'static>> = OnceLock::new();
        LIFTER.get_or_init(|| SleighLifter::new(sleigh_precompile::x64::spec()))
    }

    pub fn make_context() -> Context<'static> {
        lifter().new_context()
    }

    pub fn lift(
        ctx: &mut Context<'static>,
        instruction: &Instruction<'_, '_>,
        function: Option<FunctionId>,
    ) -> Result<BlockId, LiftError> {
        lifter().lift_instruction(ctx, instruction, function)
    }

    pub struct Disassembler {
        address: u64,
        bytes: Vec<u8>,
        decoded: bool,
    }

    impl Disassembler {
        pub fn from_bytes(address: usize, bytes: &[u8]) -> Self {
            Self {
                address: address as u64,
                bytes: bytes.to_vec(),
                decoded: false,
            }
        }
    }

    impl Iterator for Disassembler {
        type Item = Instruction<'static, 'static>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.decoded {
                return None;
            }
            self.decoded = true;
            let spec = sleigh_precompile::x64::spec();
            let context = spec.new_context();
            Decoder::new(spec)
                .decode_one(self.address, &self.bytes, &context)
                .ok()
                .map(Instruction::into_owned_bytes)
        }
    }
}

pub mod x86 {
    use super::*;

    fn lifter() -> &'static SleighLifter<'static> {
        static LIFTER: OnceLock<SleighLifter<'static>> = OnceLock::new();
        LIFTER.get_or_init(|| SleighLifter::new(sleigh_precompile::x86::spec()))
    }

    pub fn make_context() -> Context<'static> {
        lifter().new_context()
    }

    pub fn lift(
        ctx: &mut Context<'static>,
        instruction: &Instruction<'_, '_>,
        function: Option<FunctionId>,
    ) -> Result<BlockId, LiftError> {
        lifter().lift_instruction(ctx, instruction, function)
    }

    pub struct Disassembler {
        address: u64,
        bytes: Vec<u8>,
        decoded: bool,
    }

    impl Disassembler {
        pub fn from_bytes(address: usize, bytes: &[u8]) -> Self {
            Self {
                address: address as u64,
                bytes: bytes.to_vec(),
                decoded: false,
            }
        }
    }

    impl Iterator for Disassembler {
        type Item = Instruction<'static, 'static>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.decoded {
                return None;
            }
            self.decoded = true;
            let spec = sleigh_precompile::x86::spec();
            let context = spec.new_context();
            Decoder::new(spec)
                .decode_one(self.address, &self.bytes, &context)
                .ok()
                .map(Instruction::into_owned_bytes)
        }
    }
}
