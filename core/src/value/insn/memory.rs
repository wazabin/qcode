use crate::{context::Context, space::SpaceId, value::ValueId};
use std::fmt::Formatter;

use super::mnemonic::MnemonicKind;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Load {
    pub space: SpaceId,
    pub ptr: ValueId,
    pub size: usize,
}

impl MnemonicKind for Load {
    fn opcode(&self) -> &'static str {
        "load"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        // if matches!(ctx.get_value(self.ptr), ValueRef::Varnode(_)) {
        //     write!(f, "{};", ctx.get_value(self.ptr))
        // } else {
        // let space = ctx.get_space(self.space).name.unwrap_or("space");
        // write!(f, "*[{}]:{} {};", space, self.size, ctx.get_value(self.ptr))
        // }
        write!(
            f,
            "*[{}]:{} {};",
            ctx.get_space(self.space)
                .name
                .unwrap_or(&format!("space: {}", self.space)),
            self.size,
            ctx.get_value(self.ptr)
        )
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.ptr]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Store {
    pub space: SpaceId,
    pub ptr: ValueId,
    pub src: ValueId,
    pub size: usize,
}

impl Store {
    /// Create a `Load` instruction that would read from the same memory location as this `Store`.
    pub fn get_matching_load(&self) -> Load {
        Load {
            space: self.space,
            ptr: self.ptr,
            size: self.size,
        }
    }
}

impl MnemonicKind for Store {
    fn opcode(&self) -> &'static str {
        "store"
    }

    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        // if matches!(ctx.get_value(self.ptr), ValueRef::Varnode(_)) {
        //     write!(
        //         f,
        //         "{} = {};",
        //         ctx.get_value(self.ptr),
        //         ctx.get_value(self.src)
        //     )
        // } else {
        //     let space = ctx.get_space(self.space).name.unwrap_or("space");
        //     write!(
        //         f,
        //         "*[{}]:{} {} = {};",
        //         space,
        //         self.size,
        //         ctx.get_value(self.ptr),
        //         ctx.get_value(self.src)
        //     )
        // }
        write!(
            f,
            "*[{}]:{} {} = {};",
            ctx.get_space(self.space)
                .name
                .unwrap_or(&format!("space: {}", self.space)),
            self.size,
            ctx.get_value(self.ptr),
            ctx.get_value(self.src)
        )
    }

    fn args(&self) -> Vec<ValueId> {
        vec![self.ptr, self.src]
    }
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use crate::{
        builder::Builder,
        value::{
            LiteralId, VarnodeId,
            insn::{Instruction, InstructionId, Mnemonic},
        },
    };

    use super::*;

    macro_rules! assert_memory {
        ($expr:literal, $match_pat:pat, $expected_stmt:literal, $size:expr) => {{
            let mut ctx = Context::new();
            let mut builder = Builder::from_context(&mut ctx, 0x1000);

            qcode!(builder, "local i32 v0 as V0");
            let v1: InstructionId = qcode!(builder, $expr);
            builder.finalize(0x1001);
            match ctx.values.instructions[v1].clone() {
                Instruction {
                    mnemonic: $match_pat,
                    size: $size,
                    ..
                } => {}

                _ => panic!("expected memory instruction"),
            }

            assert_eq!(ctx.get_insn(v1).as_statement().to_string(), $expected_stmt);
        }};
    }

    #[test]
    fn test_load_display() {
        assert_memory!(
            "ptr = i32 {v0} + i32 0x2; load(i32, ptr)",
            Mnemonic::Load(Load { .. }),
            "i32 %tmp2 = *[space: 1]:4 i32 %tmp1;",
            4
        );
    }

    #[test]
    fn test_store_display() {
        assert_memory!(
            "ptr = i32 {v0} + i32 0x2; store(ptr, i32 0x7)",
            Mnemonic::Store(Store { .. }),
            "*[space: 1]:4 i32 %tmp1 = 0x7;",
            0
        );
    }

    #[test]
    fn test_matching_load() {
        let store = Store {
            space: SpaceId::from(0),
            ptr: ValueId::Varnode(VarnodeId::from(1)),
            src: ValueId::Literal(LiteralId::from(2)),
            size: 4,
        };

        let load = store.get_matching_load();
        assert_eq!(load.space, store.space);
        assert_eq!(load.ptr, store.ptr);
        assert_eq!(load.size, store.size);
    }
}
