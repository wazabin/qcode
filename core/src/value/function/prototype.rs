//! Appending a run of operations from a prototype.
//!
//! A lift that replays a cached instruction knows every operation it will
//! emit before it emits the first: the operations are a recording, with
//! operands numbered relative to the recording. Emitting them one at a
//! time through the [`Builder`](crate::builder::Builder) pays the builder's
//! per-operation bookkeeping thousands of times per instruction of a
//! program. [`FunctionBody::append_prototype`] takes the whole run and
//! does, per operation, exactly what the body must: copy the mnemonic and
//! point its operands into the body, push it, link it, record its use
//! edges, connect its branches and name its result — in one pass, with
//! nothing resolved twice.

use smallvec::SmallVec;

use crate::{
    space::LocalMemorySpaceId,
    types::TypeId,
    value::{
        FunctionBody, Instruction, LocalBlockId, LocalTempId, LocalTempSpaceId, LocalValueId,
        VarnodeId,
        insn::{Callee, LocalInsnId, Mnemonic},
        name::BaseId,
    },
};

/// One operation of a prototype run. Its mnemonic's operands are indices
/// into the run's [`ProtoMap`]: `Literal(k)` the `k`th literal,
/// `Instruction(k)` the run's `k`th operation, `Temp(k)` the `k`th
/// temporary, `Varnode(k)` the `k`th varnode; a load's or store's temporary
/// space is the `k`th space, a branch's target the `k`th block and a call's
/// [minted](Callee::Minted) callee the `k`th callee.
#[derive(Debug, Clone)]
pub struct ProtoOp {
    /// The run's block the operation is in, an index into the map's blocks.
    pub block: u32,
    pub type_id: TypeId,
    /// The base the result is named after, made unique in the body.
    pub base: Option<BaseId>,
    pub mnemonic: Mnemonic,
}

/// What a prototype run's indices stand for in the body it is appended to.
#[derive(Debug, Clone, Copy)]
pub struct ProtoMap<'a> {
    pub literals: &'a [LocalValueId],
    pub varnodes: &'a [VarnodeId],
    pub temps: &'a [LocalTempId],
    pub spaces: &'a [LocalTempSpaceId],
    pub blocks: &'a [LocalBlockId],
    pub callees: &'a [Callee],
}

impl<'str> FunctionBody<'str> {
    /// Appends `ops`, resolved through `map`, at the end of their blocks,
    /// and pushes each operation's id onto `out`, in order. `address` is
    /// the machine address every operation is at; names are given only
    /// when `naming`.
    ///
    /// The run is what a lift recorded, so nothing here can fail: the
    /// blocks are the run's own or placeholders made for it, and a
    /// terminator is the last operation of its block.
    pub(crate) fn append_prototype(
        &mut self,
        ops: &[ProtoOp],
        map: &ProtoMap<'_>,
        address: Option<u64>,
        naming: bool,
        out: &mut Vec<LocalInsnId>,
    ) {
        let first = out.len();
        // The run's operations are grouped by block, so a block's list is
        // read and written once per run of its own rather than per
        // operation: `tail` is the last operation appended to `block`, and
        // `added` how many are not counted in its list yet.
        let mut block = map.blocks[ops.first().map_or(0, |op| op.block as usize)];
        let mut tail = self.blocks[block].instructions.last;
        let mut added = 0usize;
        for op in ops {
            let at = map.blocks[op.block as usize];
            if at != block {
                self.close_run(block, tail, added);
                block = at;
                tail = self.blocks[block].instructions.last;
                added = 0;
            }
            let mut mnemonic = op.mnemonic.clone();
            mnemonic.for_each_operand_mut(|operand| {
                *operand = match *operand {
                    LocalValueId::Literal(k) => map.literals[usize::from(k)],
                    LocalValueId::Instruction(k) => {
                        LocalValueId::Instruction(out[first + usize::from(k)])
                    }
                    LocalValueId::Temp(k) => LocalValueId::Temp(map.temps[usize::from(k)]),
                    LocalValueId::Varnode(k) => LocalValueId::Varnode(map.varnodes[usize::from(k)]),
                    other => other,
                }
            });
            match &mut mnemonic {
                Mnemonic::Load(load) => {
                    if let LocalMemorySpaceId::Temp(space) = &mut load.space {
                        *space = map.spaces[usize::from(*space)];
                    }
                }
                Mnemonic::Store(store) => {
                    if let LocalMemorySpaceId::Temp(space) = &mut store.space {
                        *space = map.spaces[usize::from(*space)];
                    }
                }
                Mnemonic::Branch(branch) => {
                    branch.target = map.blocks[usize::from(branch.target)];
                    self.add_cfg_edge_local(block, branch.target);
                }
                Mnemonic::CBranch(cbranch) => {
                    cbranch.success_block = map.blocks[usize::from(cbranch.success_block)];
                    cbranch.failure_block = map.blocks[usize::from(cbranch.failure_block)];
                    self.add_cfg_edge_local(block, cbranch.success_block);
                    self.add_cfg_edge_local(block, cbranch.failure_block);
                }
                Mnemonic::Call(call) => {
                    if let Some(k) = call.target.minted() {
                        call.target = map.callees[k as usize];
                    }
                }
                Mnemonic::TailCall(call) => {
                    if let Some(k) = call.target.minted() {
                        call.target = map.callees[k as usize];
                    }
                }
                _ => {}
            }
            // The operands, read before the mnemonic moves into the arena
            // whose slots the use edges' heads live in.
            let mut operands: SmallVec<[LocalValueId; 4]> = SmallVec::new();
            mnemonic.for_each_operand(|operand| operands.push(operand));

            let local = self.insns.push(Instruction::linked(
                op.type_id, mnemonic, block, tail, address,
            ));
            match tail {
                Some(tail) => self.insns[tail].next.set(Some(local)),
                None => self.blocks[block].instructions.first = Some(local),
            }
            tail = Some(local);
            added += 1;

            for (index, &value) in operands.iter().enumerate() {
                self.add_use(value, local, index);
            }
            if let Some(base) = op.base
                && naming
            {
                self.name_insn_unique(local, base);
            }
            out.push(local);
        }
        if !ops.is_empty() {
            self.close_run(block, tail, added);
        }
    }

    /// Records that `added` operations ending at `tail` joined `block`.
    fn close_run(&mut self, block: LocalBlockId, tail: Option<LocalInsnId>, added: usize) {
        let list = &mut self.blocks[block].instructions;
        list.last = tail;
        list.len += added;
    }
}
