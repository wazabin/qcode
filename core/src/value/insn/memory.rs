use crate::{space::SpaceId, value::ValueId};

use super::mnemonic::{Args, MnemonicKind};
use smallvec::smallvec;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Load {
    pub space: SpaceId,
    pub ptr: ValueId,
    pub size: usize,
}

impl MnemonicKind for Load {
    fn opcode(&self) -> &'static str {
        "load"
    }

    fn args(&self) -> Args {
        smallvec![self.ptr]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

    fn args(&self) -> Args {
        smallvec![self.ptr, self.src]
    }
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use crate::context::Context;
    use crate::value::{
        BasicBlock, LiteralId, Varnode, VarnodeId,
        insn::{Instruction, Mnemonic},
    };

    use super::*;

    #[test]
    fn test_load_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i32 v0;

            <block>
                %ptr = i64 &v0 + i64 0x2;
                %v = load(v0:4, %ptr);
                goto <0x1001>;
            "
        );

        let v = Instruction::from_id(&ctx, v);

        if !matches!(v.mnemonic(), Mnemonic::Load(Load { .. })) {
            panic!("expected memory instruction");
        }

        assert_eq!(v.size(), 4);
        assert!(v.space().is_none());
        assert_eq!(
            v.as_statement().to_string(),
            "i32 %v = load(v0:4, i32 %ptr);"
        );
    }

    #[test]
    fn test_store_display() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i32 V0;

            <block>
                %v0 = load(V0:4, V0);
                %ptr = i32 %v0 + i32 0x2;
                store(ram:4, %ptr <- i32 0x7);
                goto <0x1001>;
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let store = block.iter().nth(2).expect("expected store instruction");

        if !matches!(store.mnemonic(), Mnemonic::Store(Store { .. })) {
            panic!("expected memory instruction");
        }

        assert_eq!(store.size(), 0);
        assert!(store.space().is_none());
        assert_eq!(
            store.as_statement().to_string(),
            "store(ram:4, i32 %ptr <- i32 0x7);"
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

    #[test]
    fn test_computed_pointer_load_inherits_instruction_space() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 A;

            <block>
                %ptr = i64 &A + i64 0x2;
                %value = load(A:8, %ptr);
                goto <0x1001>;
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let a = Varnode::from_id(&ctx, A);
        let ptr = block.iter().next().expect("expected pointer arithmetic");
        let value = Instruction::from_id(&ctx, value);

        assert_eq!(ptr.space().and_then(|s| s.name.as_deref()), Some("A"));

        let Mnemonic::Load(load) = value.mnemonic() else {
            panic!("expected load instruction");
        };
        assert_eq!(load.space, a.space().id);
        assert_eq!(
            value.as_statement().to_string(),
            "i64 %value = load(A:8, i64 %ptr);"
        );
    }

    #[test]
    fn test_computed_pointer_store_inherits_instruction_space() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 A;

            <block>
                %ptr = i64 &A + i64 0x2;
                store(A:8, %ptr <- i64 0x7);
                goto <0x1001>;
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let a = Varnode::from_id(&ctx, A);
        let store = block.iter().nth(1).expect("expected store instruction");

        let Mnemonic::Store(store_mnemonic) = store.mnemonic() else {
            panic!("expected store instruction");
        };
        assert_eq!(store_mnemonic.space, a.space().id);
        assert_eq!(
            store.as_statement().to_string(),
            "store(A:8, i64 %ptr <- i64 0x7);"
        );
    }

    #[test]
    fn test_computed_pointer_without_provenance_falls_back_to_default_space() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                %ptr = i64 0x10 + i64 0x2;
                %v = load(ram:8, %ptr);
                goto <0x1001>;
            "
        );

        let value = Instruction::from_id(&ctx, v);
        let ptr = Instruction::from_id(&ctx, ptr);

        assert_eq!(
            ptr.space().map(|s| s.id),
            Some(ctx.shared.default_space),
            "expected default space for pointer arithmetic without provenance"
        );

        let Mnemonic::Load(load) = value.mnemonic() else {
            panic!("expected load instruction");
        };
        assert_eq!(load.space, ctx.shared.default_space);
        assert_eq!(
            value.as_statement().to_string(),
            "i64 %v = load(ram:8, i64 %ptr);"
        );
    }
}
