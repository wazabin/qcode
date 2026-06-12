use std::collections::HashMap;

use qcode::{
    context::Context,
    value::{
        ValueId,
        function::FunctionSignature,
        insn::{Binop, FloatBinop, IntBinop, Mnemonic},
    },
};

use super::{AliasResult, NodeId};

// ---------------------------------------------------------------------------
// Internal union-find state
// ---------------------------------------------------------------------------

struct SteensgaardState {
    /// parent[i] for node Id(i). Id(i) == Id(i) means i is a root.
    parent: Vec<NodeId>,
    rank: Vec<u8>,
    /// Steensgaard's ref() field. None = no ref slot yet.
    points_to: Vec<Option<NodeId>>,
    value_to_node: HashMap<ValueId, NodeId>,
    next_id: usize,
}

impl SteensgaardState {
    fn new() -> Self {
        Self {
            parent: Vec::new(),
            rank: Vec::new(),
            points_to: Vec::new(),
            value_to_node: HashMap::new(),
            next_id: 0,
        }
    }

    fn alloc_node(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.parent.push(NodeId::Id(id));
        self.rank.push(0);
        self.points_to.push(None);
        NodeId::Id(id)
    }

    fn node_for(&mut self, value: ValueId) -> NodeId {
        if let Some(&n) = self.value_to_node.get(&value) {
            return n;
        }
        let n = self.alloc_node();
        self.value_to_node.insert(value, n);
        n
    }

    /// Path-halving find. Returns `Unknown` if the root of `x` has been
    /// merged into `Unknown`.
    fn find_mut(&mut self, x: NodeId) -> NodeId {
        let NodeId::Id(mut i) = x else {
            return NodeId::Unknown;
        };
        loop {
            match self.parent[i] {
                NodeId::Unknown => return NodeId::Unknown,
                NodeId::Id(p) if p == i => return NodeId::Id(i),
                NodeId::Id(p) => {
                    // Path halving: skip to grandparent (may be Unknown).
                    self.parent[i] = self.parent[p];
                    i = p;
                }
            }
        }
    }

    /// Union-by-rank merge. Merging into `Unknown` redirects the other node's
    /// parent; if both `Id` roots have a `points_to`, join those too.
    fn join(&mut self, a: NodeId, b: NodeId) {
        let ra = self.find_mut(a);
        let rb = self.find_mut(b);
        if ra == rb {
            return;
        }

        match (ra, rb) {
            (NodeId::Unknown, NodeId::Id(i)) | (NodeId::Id(i), NodeId::Unknown) => {
                let pt = self.points_to[i].take();
                self.parent[i] = NodeId::Unknown;
                if let Some(p) = pt {
                    self.join(p, NodeId::Unknown);
                }
            }
            (NodeId::Id(ri), NodeId::Id(rj)) => {
                let pti = self.points_to[ri];
                let ptj = self.points_to[rj];
                let (winner, loser) = if self.rank[ri] >= self.rank[rj] {
                    (ri, rj)
                } else {
                    (rj, ri)
                };
                self.parent[loser] = NodeId::Id(winner);
                if self.rank[winner] == self.rank[loser] {
                    self.rank[winner] += 1;
                }
                match (pti, ptj) {
                    (None, None) => {}
                    (Some(p), None) | (None, Some(p)) => {
                        self.points_to[winner] = Some(p);
                    }
                    (Some(pi), Some(pj)) => {
                        self.points_to[winner] = Some(pi);
                        self.join(pi, pj);
                    }
                }
            }
            (NodeId::Unknown, NodeId::Unknown) => unreachable!("covered by ra == rb"),
        }
    }

    /// Return (or lazily create) the node that `n` points to.
    /// `Unknown` always points to itself.
    fn get_or_create_ref(&mut self, n: NodeId) -> NodeId {
        match self.find_mut(n) {
            NodeId::Unknown => NodeId::Unknown,
            NodeId::Id(i) => match self.points_to[i] {
                Some(r) => r,
                None => {
                    let fresh = self.alloc_node();
                    self.points_to[i] = Some(fresh);
                    fresh
                }
            },
        }
    }

    /// Path-compress every mapped value to its canonical root.
    fn finalize(mut self) -> AliasResult {
        let pairs: Vec<(ValueId, NodeId)> =
            self.value_to_node.iter().map(|(&v, &n)| (v, n)).collect();
        let value_to_root = pairs
            .into_iter()
            .map(|(v, n)| (v, self.find_mut(n)))
            .collect();
        AliasResult {
            value_to_root,
            value_to_interval: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Constraint generation
// ---------------------------------------------------------------------------

fn apply_sig(state: &mut SteensgaardState, sig: &FunctionSignature) {
    for &vn in sig
        .outputs
        .iter()
        .flatten()
        .chain(sig.caller_saved.iter().flatten())
    {
        let n = state.node_for(ValueId::Varnode(vn));
        state.join(n, NodeId::Unknown);
    }
}

fn is_comparison(op: &Binop) -> bool {
    matches!(
        op,
        Binop::Int(
            IntBinop::Equal
                | IntBinop::NotEqual
                | IntBinop::Less
                | IntBinop::SLess
                | IntBinop::LessEqual
                | IntBinop::SLessEqual
        ) | Binop::Float(
            FloatBinop::Equal | FloatBinop::NotEqual | FloatBinop::Less | FloatBinop::LessEqual
        )
    )
}

/// Run Steensgaard's flow-insensitive alias analysis over all instructions in
/// `ctx` and return an [`AliasResult`] that answers [`AliasResult::may_alias`]
/// queries.
pub fn alias_analysis(ctx: &Context) -> AliasResult {
    use qcode::value::InstructionId;

    let mut state = SteensgaardState::new();

    let insns: Vec<(usize, usize, Mnemonic)> = ctx
        .values
        .instructions
        .iter()
        .map(|item| {
            let insn_ref = ctx.get_insn(item.id);
            (item.id.into(), insn_ref.size(), insn_ref.mnemonic().clone())
        })
        .collect();

    for (raw_id, size, mnemonic) in insns {
        let result_id = ValueId::Instruction(InstructionId::from(raw_id));

        match &mnemonic {
            // r = Load { ptr } => node(r) == get_or_create_ref(node(ptr))
            Mnemonic::Load(load) if size > 0 => {
                let r = state.node_for(result_id);
                let ptr_node = state.node_for(load.ptr);
                let ref_node = state.get_or_create_ref(ptr_node);
                state.join(r, ref_node);
            }

            // Store { ptr, src } => get_or_create_ref(node(ptr)) == node(src)
            Mnemonic::Store(store) => {
                let ptr_node = state.node_for(store.ptr);
                let ref_node = state.get_or_create_ref(ptr_node);
                let src_node = state.node_for(store.src);
                state.join(ref_node, src_node);
            }

            // r = Zext/Sext/Range { src } => node(r) == node(src)
            Mnemonic::Zext(z) if size > 0 => {
                let r = state.node_for(result_id);
                let s = state.node_for(z.src);
                state.join(r, s);
            }
            Mnemonic::Sext(s_op) if size > 0 => {
                let r = state.node_for(result_id);
                let s = state.node_for(s_op.src);
                state.join(r, s);
            }
            Mnemonic::Range(rng) if size > 0 => {
                let r = state.node_for(result_id);
                let s = state.node_for(rng.src);
                state.join(r, s);
            }

            // r = Binop { lhs, rhs } (non-comparison) => node(r) == node(lhs), node(r) == node(rhs)
            Mnemonic::Binop(bin) if size > 0 && !is_comparison(&bin.op) => {
                let r = state.node_for(result_id);
                let l = state.node_for(bin.lhs);
                let rh = state.node_for(bin.rhs);
                state.join(r, l);
                state.join(r, rh);
            }

            // r = PCodeOp (size > 0) => node(r) == Unknown
            Mnemonic::PCodeOp(_) if size > 0 => {
                let r = state.node_for(result_id);
                state.join(r, NodeId::Unknown);
            }

            Mnemonic::Call(call) => {
                if let Some(sig) = ctx.values.functions[call.target].signature.as_ref() {
                    apply_sig(&mut state, sig);
                }
                // Arguments escape: a value passed into the callee may be stored
                // through, so it joins Unknown. Likewise every recorded
                // clobbered/aliased location.
                for &arg in call.args.iter().chain(call.clobbers.iter()) {
                    let n = state.node_for(arg);
                    state.join(n, NodeId::Unknown);
                }
            }

            Mnemonic::CallInd(call) => {
                // No FunctionId, so no signature; conservatively escape the
                // (over-approximated) arguments through the unknown callee.
                for &arg in &call.args {
                    let n = state.node_for(arg);
                    state.join(n, NodeId::Unknown);
                }
            }

            _ => {}
        }
    }

    state.finalize()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use qcode::{
        builder::Builder,
        context::Context,
        testing::TestContext,
        value::{BlockId, Function, InstructionId, Value, ValueId, function::FunctionSignature},
    };

    fn build_block(f: impl FnOnce(&mut Builder<'static, '_>)) -> (Context<'static>, BlockId) {
        let mut ctx = TestContext::new().ctx;
        let block_id = ctx.get_or_make_block(0x1000);
        let mut builder = Builder::from_context(&mut ctx, 0x1000);
        f(&mut builder);
        unsafe { builder.dont_finalize() };
        drop(builder);
        (ctx, block_id)
    }

    #[test]
    fn test_zext_propagates_alias() {
        let src_cell: Cell<ValueId> = Cell::new(ValueId::Instruction(InstructionId::from(0)));
        let zext_cell: Cell<ValueId> = Cell::new(ValueId::Instruction(InstructionId::from(0)));

        let (ctx, _) = build_block(|b| {
            let src = b.context_mut().get_const(0xdeadu64, 8).id();
            src_cell.set(src);
            let z = b.push_zext(src, 8);
            zext_cell.set(z.id());
        });

        let result = alias_analysis(&ctx);
        assert!(
            result.may_alias(&ctx, zext_cell.get(), src_cell.get()),
            "zext result must alias its source"
        );
    }

    #[test]
    fn test_load_store_round_trip() {
        let v_cell: Cell<ValueId> = Cell::new(ValueId::Instruction(InstructionId::from(0)));
        let load_cell: Cell<ValueId> = Cell::new(ValueId::Instruction(InstructionId::from(0)));

        let (ctx, _) = build_block(|b| {
            let space = b.context().try_get_space("register").unwrap();
            let ptr = b.context_mut().get_const(0x1000u64, 8).id();
            let v = b.context_mut().get_const(42u64, 8).id();
            v_cell.set(v);
            b.push_store(v, ptr, space);
            let loaded = b.push_load::<false>(ptr, 8, space);
            load_cell.set(loaded.id());
        });

        let result = alias_analysis(&ctx);
        assert!(
            result.may_alias(&ctx, load_cell.get(), v_cell.get()),
            "load through same pointer must alias the stored value"
        );
    }

    #[test]
    fn test_distinct_values_dont_alias() {
        let a_cell: Cell<ValueId> = Cell::new(ValueId::Instruction(InstructionId::from(0)));
        let b_cell: Cell<ValueId> = Cell::new(ValueId::Instruction(InstructionId::from(0)));

        let (ctx, _) = build_block(|b| {
            let a = b.context_mut().get_const(1u64, 8).id();
            let bv = b.context_mut().get_const(2u64, 8).id();
            a_cell.set(a);
            b_cell.set(bv);
        });

        let result = alias_analysis(&ctx);
        assert!(
            !result.may_alias(&ctx, a_cell.get(), b_cell.get()),
            "unrelated constants must not alias"
        );
    }

    #[test]
    fn test_binop_propagates_alias() {
        let ptr_cell: Cell<ValueId> = Cell::new(ValueId::Instruction(InstructionId::from(0)));
        let add_cell: Cell<ValueId> = Cell::new(ValueId::Instruction(InstructionId::from(0)));

        let (ctx, _) = build_block(|b| {
            let ptr = b.context_mut().get_const(0x1000u64, 8).id();
            ptr_cell.set(ptr);
            let offset = b.context_mut().get_const(8u64, 8).id();
            let r = b.push_add(ptr, offset);
            add_cell.set(r.id());
        });

        let result = alias_analysis(&ctx);
        assert!(
            result.may_alias(&ctx, add_cell.get(), ptr_cell.get()),
            "binop result must alias its lhs operand"
        );
    }

    #[test]
    fn test_comparison_does_not_propagate() {
        let a_cell: Cell<ValueId> = Cell::new(ValueId::Instruction(InstructionId::from(0)));
        let cond_cell: Cell<ValueId> = Cell::new(ValueId::Instruction(InstructionId::from(0)));

        let (ctx, _) = build_block(|b| {
            let a = b.context_mut().get_const(1u64, 8).id();
            a_cell.set(a);
            let bv = b.context_mut().get_const(2u64, 8).id();
            let cond = b.push_eq(a, bv);
            cond_cell.set(cond.id());
        });

        let result = alias_analysis(&ctx);
        assert!(
            !result.may_alias(&ctx, cond_cell.get(), a_cell.get()),
            "comparison result must not alias its operands"
        );
    }

    #[test]
    fn test_transitive_join() {
        let v_cell: Cell<ValueId> = Cell::new(ValueId::Instruction(InstructionId::from(0)));
        let final_cell: Cell<ValueId> = Cell::new(ValueId::Instruction(InstructionId::from(0)));

        let (ctx, _) = build_block(|b| {
            let space = b.context().try_get_space("register").unwrap();
            let v = b.context_mut().get_const(0xaau64, 8).id();
            v_cell.set(v);
            let p = b.context_mut().get_const(0x100u64, 8).id();
            let q = b.context_mut().get_const(0x200u64, 8).id();

            b.push_store(v, p, space);
            let p_val = b.push_load::<false>(p, 8, space);
            let p_val_id = p_val.id();
            b.push_store(p_val_id, q, space);
            let final_load = b.push_load::<false>(q, 8, space);
            final_cell.set(final_load.id());
        });

        let result = alias_analysis(&ctx);
        assert!(
            result.may_alias(&ctx, final_cell.get(), v_cell.get()),
            "transitively chained load/store must alias original value"
        );
    }

    fn make_call_ctx(sig: Option<FunctionSignature>) -> (Context<'static>, ValueId) {
        let tc = TestContext::new();
        let rax_vn = tc.r0;
        let rax_vid = ValueId::Varnode(rax_vn);
        let mut ctx = tc.ctx;

        let fn_id = Function::make(&mut ctx, "target".into()).unwrap().id;
        ctx.values.functions[fn_id].signature = sig;

        let _caller_root = ctx.get_or_make_block(0x1000);
        let mut builder = Builder::from_context(&mut ctx, 0x1000);
        builder.push_call(fn_id);
        drop(builder);

        (ctx, rax_vid)
    }

    #[test]
    fn test_call_caller_saved_clobbered() {
        let rax_vn = TestContext::new().r0;
        let sig = FunctionSignature {
            caller_saved: Some(vec![rax_vn]),
            ..Default::default()
        };
        let (ctx, rax_vid) = make_call_ctx(Some(sig));
        let result = alias_analysis(&ctx);
        assert_eq!(
            result.alias_class(rax_vid),
            Some(NodeId::Unknown),
            "caller_saved register must be Unknown after call"
        );
    }

    #[test]
    fn test_call_output_clobbered() {
        let rax_vn = TestContext::new().r0;
        let sig = FunctionSignature {
            outputs: Some(vec![rax_vn]),
            ..Default::default()
        };
        let (ctx, rax_vid) = make_call_ctx(Some(sig));
        let result = alias_analysis(&ctx);
        assert_eq!(
            result.alias_class(rax_vid),
            Some(NodeId::Unknown),
            "output register must be Unknown after call"
        );
    }

    #[test]
    fn test_call_no_signature_no_effect() {
        let (ctx, rax_vid) = make_call_ctx(None);
        let result = alias_analysis(&ctx);
        assert_eq!(
            result.alias_class(rax_vid),
            None,
            "no signature must not introduce alias for registers"
        );
    }

    #[test]
    fn test_call_ind_no_effect() {
        let tc = TestContext::new();
        let rax_vid = ValueId::Varnode(tc.r0);
        let mut ctx = tc.ctx;

        let _block = ctx.get_or_make_block(0x3000);
        let mut builder = Builder::from_context(&mut ctx, 0x3000);
        let ptr = builder.context_mut().get_const(0x4000u64, 8).id();
        builder.push_call_ind(ptr);
        drop(builder);

        let result = alias_analysis(&ctx);
        assert_eq!(
            result.alias_class(rax_vid),
            None,
            "CallInd must not introduce alias for registers"
        );
    }

    #[test]
    fn test_two_loads_same_ptr_alias() {
        let r1_cell: Cell<ValueId> = Cell::new(ValueId::Instruction(InstructionId::from(0)));
        let r2_cell: Cell<ValueId> = Cell::new(ValueId::Instruction(InstructionId::from(0)));

        let (ctx, _) = build_block(|b| {
            let space = b.context().try_get_space("register").unwrap();
            let addr = b.context_mut().get_const(0x300u64, 8).id();
            let r1 = b.push_load::<false>(addr, 8, space);
            r1_cell.set(r1.id());
            let r2 = b.push_load::<false>(addr, 8, space);
            r2_cell.set(r2.id());
        });

        let result = alias_analysis(&ctx);
        assert!(
            result.may_alias(&ctx, r1_cell.get(), r2_cell.get()),
            "two loads through the same pointer ValueId must alias each other"
        );
    }
}
