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
                .as_deref()
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
                .as_deref()
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

    use crate::value::{
        BasicBlock, LiteralId, VarnodeId,
        insn::{Instruction, Mnemonic},
    };

    use super::*;

    #[test]
    fn test_load_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 v0;
                %ptr = i32 %v0 + i32 0x2;
                %v = load(i32, %ptr);
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        if !matches!(v.mnemonic(), Mnemonic::Load(Load { .. })) {
            panic!("expected memory instruction");
        }

        assert_eq!(v.size(), 4);
        assert_eq!(v.as_statement().to_string(), "i32 %v = *[ram]:4 i32 %ptr;");
    }

    #[test]
    fn test_store_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                local i32 V0;
                %v0 = load(i32, V0);
                %ptr = i32 %v0 + i32 0x2;
                store(%ptr, i32 0x7);
                goto <0x1001>;
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let store = block.iter().nth(2).expect("expected store instruction");

        if !matches!(store.mnemonic(), Mnemonic::Store(Store { .. })) {
            panic!("expected memory instruction");
        }

        assert_eq!(store.size(), 0);
        assert_eq!(store.as_statement().to_string(), "*[ram]:4 i32 %ptr = 0x7;");
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
