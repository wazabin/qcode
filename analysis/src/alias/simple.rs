use std::collections::HashMap;

use qcode::{
    context::Context,
    space::{SpaceId, SpaceType},
    value::{
        insn::{Binop, IntBinop, Mnemonic},
        ValueId, ValueRef, Varnode,
    },
};

use super::{AliasResult, NodeId};

#[derive(Clone, Copy)]
struct SizedVarnode {
    root: NodeId,
    start: u64,
    end: u64,
}

#[derive(Clone, Copy)]
struct SizedRoot {
    root: NodeId,
    start: u64,
    end: u64,
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: Vec::new(),
            rank: Vec::new(),
        }
    }

    fn alloc_node(&mut self) -> NodeId {
        let id = self.parent.len();
        self.parent.push(id);
        self.rank.push(0);
        NodeId::Id(id)
    }

    fn find_mut(&mut self, node: NodeId) -> NodeId {
        let NodeId::Id(mut idx) = node else {
            return NodeId::Unknown;
        };

        loop {
            let parent = self.parent[idx];
            if parent == idx {
                return NodeId::Id(idx);
            }

            let grandparent = self.parent[parent];
            self.parent[idx] = grandparent;
            idx = parent;
        }
    }

    fn join(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let ra = self.find_mut(a);
        let rb = self.find_mut(b);

        if ra == rb {
            return ra;
        }

        let (NodeId::Id(ra), NodeId::Id(rb)) = (ra, rb) else {
            return NodeId::Unknown;
        };

        let (winner, loser) = if self.rank[ra] >= self.rank[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };

        self.parent[loser] = winner;
        if self.rank[winner] == self.rank[loser] {
            self.rank[winner] += 1;
        }

        NodeId::Id(winner)
    }
}

fn overlaps(start_a: u64, end_a: u64, start_b: u64, end_b: u64) -> bool {
    start_a < end_b && start_b < end_a
}

fn canonical_root(uf: &mut UnionFind, root: NodeId) -> NodeId {
    match root {
        NodeId::Unknown => NodeId::Unknown,
        NodeId::Id(_) => uf.find_mut(root),
    }
}

fn lookup_root(
    value_to_root: &HashMap<ValueId, NodeId>,
    uf: &mut UnionFind,
    value: ValueId,
) -> Option<NodeId> {
    value_to_root
        .get(&value)
        .copied()
        .map(|root| canonical_root(uf, root))
}

fn merge_roots(existing: NodeId, incoming: NodeId, uf: &mut UnionFind) -> NodeId {
    match (existing, incoming) {
        (NodeId::Unknown, _) | (_, NodeId::Unknown) => NodeId::Unknown,
        _ => uf.join(existing, incoming),
    }
}

fn record_value_root(
    value_to_root: &mut HashMap<ValueId, NodeId>,
    uf: &mut UnionFind,
    value: ValueId,
    root: NodeId,
) {
    match value_to_root.get_mut(&value) {
        Some(existing) => *existing = merge_roots(*existing, root, uf),
        None => {
            value_to_root.insert(value, root);
        }
    }
}

fn literal_interval(ctx: &Context, literal: ValueId, size: usize) -> Option<(u64, u64)> {
    let ValueRef::Literal(literal) = ctx.get_value(literal) else {
        unreachable!("literal_interval must only be called for literals");
    };

    let start = literal.value();
    let size = u64::try_from(size).ok()?;
    let end = start.checked_add(size)?;
    Some((start, end))
}

fn assign_literal_root(
    ctx: &Context,
    value_to_root: &HashMap<ValueId, NodeId>,
    by_space: &HashMap<SpaceId, Vec<SizedVarnode>>,
    literal_ranges: &mut HashMap<SpaceId, Vec<SizedRoot>>,
    uf: &mut UnionFind,
    literal: ValueId,
    space: SpaceId,
    size: usize,
) -> NodeId {
    let Some((start, end)) = literal_interval(ctx, literal, size) else {
        return NodeId::Unknown;
    };

    let mut root = lookup_root(value_to_root, uf, literal).unwrap_or_else(|| uf.alloc_node());

    if let Some(varnodes) = by_space.get(&space) {
        for varnode in varnodes {
            if overlaps(start, end, varnode.start, varnode.end) {
                let varnode_root = canonical_root(uf, varnode.root);
                root = uf.join(root, varnode_root);
            }
        }
    }

    if let Some(ranges) = literal_ranges.get(&space) {
        for range in ranges.iter().copied() {
            if overlaps(start, end, range.start, range.end) {
                let range_root = canonical_root(uf, range.root);
                root = uf.join(root, range_root);
            }
        }
    }

    literal_ranges
        .entry(space)
        .or_default()
        .push(SizedRoot { root, start, end });

    root
}

fn unresolved_pointer_root(
    ctx: &Context,
    value_to_root: &HashMap<ValueId, NodeId>,
    uf: &mut UnionFind,
    value: ValueId,
    space: SpaceId,
) -> NodeId {
    if matches!(ctx.get_space(space).ty, SpaceType::Register) {
        lookup_root(value_to_root, uf, value).unwrap_or_else(|| uf.alloc_node())
    } else {
        NodeId::Unknown
    }
}

fn resolve_pointer_root(
    ctx: &Context,
    value: ValueId,
    space: SpaceId,
    size: usize,
    value_to_root: &HashMap<ValueId, NodeId>,
    by_space: &HashMap<SpaceId, Vec<SizedVarnode>>,
    literal_ranges: &mut HashMap<SpaceId, Vec<SizedRoot>>,
    uf: &mut UnionFind,
) -> NodeId {
    match value {
        ValueId::Varnode(id) => {
            let varnode = Varnode::from_id(ctx, id);
            debug_assert_eq!(
                varnode.space().id,
                space,
                "load/store pointer varnodes must stay in the access space; \
                 builder.rs::push_load documents this IR invariant"
            );

            if varnode.space().id != space {
                return unresolved_pointer_root(ctx, value_to_root, uf, value, space);
            }

            lookup_root(value_to_root, uf, value).unwrap_or(NodeId::Unknown)
        }
        ValueId::Literal(_) => assign_literal_root(
            ctx,
            value_to_root,
            by_space,
            literal_ranges,
            uf,
            value,
            space,
            size,
        ),
        ValueId::Instruction(id) => {
            if ctx.value_space_id(value) != Some(space) {
                // TODO(ir-invariant): require every load/store pointer to carry
                // space provenance so this defensive Unknown path can go away.
                return unresolved_pointer_root(ctx, value_to_root, uf, value, space);
            }

            match ctx.get_insn(id).mnemonic() {
                Mnemonic::Binop(bin)
                    if matches!(bin.op, Binop::Int(IntBinop::Add | IntBinop::Sub)) =>
                {
                    let lhs_space = ctx.value_space_id(bin.lhs);
                    let rhs_space = ctx.value_space_id(bin.rhs);

                    if lhs_space == Some(space) && rhs_space != Some(space) {
                        resolve_pointer_root(
                            ctx,
                            bin.lhs,
                            space,
                            size,
                            value_to_root,
                            by_space,
                            literal_ranges,
                            uf,
                        )
                    } else if matches!(bin.op, Binop::Int(IntBinop::Add))
                        && rhs_space == Some(space)
                        && lhs_space != Some(space)
                    {
                        resolve_pointer_root(
                            ctx,
                            bin.rhs,
                            space,
                            size,
                            value_to_root,
                            by_space,
                            literal_ranges,
                            uf,
                        )
                    } else {
                        unresolved_pointer_root(ctx, value_to_root, uf, value, space)
                    }
                }
                _ => unresolved_pointer_root(ctx, value_to_root, uf, value, space),
            }
        }
        _ => unresolved_pointer_root(ctx, value_to_root, uf, value, space),
    }
}

impl AliasResult {
    /// A location-based aliasing result that only reasons about known varnode
    /// ranges. Overlapping varnodes in the same space are joined; only pointer
    /// values that actually participate in loads/stores are added.
    pub fn simple(ctx: &Context) -> Self {
        let mut value_to_root = HashMap::new();
        let mut by_space: HashMap<_, Vec<_>> = HashMap::new();
        let mut uf = UnionFind::new();

        for varnode in ctx.varnodes() {
            let address = varnode.address();
            debug_assert!(
                address >= 0,
                "simple alias analysis expects non-negative varnode addresses"
            );

            let start = address as u64;
            let end = start
                .checked_add(varnode.size() as u64)
                .expect("varnode range must fit in u64");
            let root = uf.alloc_node();

            value_to_root.insert(varnode.id.into(), root);
            by_space
                .entry(varnode.space().id)
                .or_default()
                .push(SizedVarnode { root, start, end });
        }

        for sized_nodes in by_space.values_mut() {
            sized_nodes.sort_by_key(|node| (node.start, node.end));

            let Some(first) = sized_nodes.first().copied() else {
                continue;
            };

            let mut component_root = first.root;
            let mut component_end = first.end;

            for node in sized_nodes.iter().skip(1).copied() {
                if node.start < component_end {
                    component_root = uf.join(component_root, node.root);
                    component_end = component_end.max(node.end);
                } else {
                    component_root = node.root;
                    component_end = node.end;
                }
            }
        }

        let pointer_uses: Vec<(ValueId, SpaceId, usize)> = ctx
            .instructions()
            .filter_map(|insn| match insn.mnemonic() {
                Mnemonic::Load(load) => Some((load.ptr, load.space, load.size)),
                Mnemonic::Store(store) => Some((store.ptr, store.space, store.size)),
                _ => None,
            })
            .collect();

        let mut literal_ranges: HashMap<SpaceId, Vec<SizedRoot>> = HashMap::new();
        let mut pointer_spaces: HashMap<ValueId, SpaceId> = HashMap::new();

        for (ptr, space, size) in pointer_uses {
            if let Some(existing_space) = pointer_spaces.insert(ptr, space) {
                assert_eq!(
                    existing_space, space,
                    "simple alias analysis invariant violated: pointer {ptr} used in multiple spaces ({existing_space:?} vs {space:?})"
                );
            }

            let root = resolve_pointer_root(
                ctx,
                ptr,
                space,
                size,
                &value_to_root,
                &by_space,
                &mut literal_ranges,
                &mut uf,
            );
            record_value_root(&mut value_to_root, &mut uf, ptr, root);
        }

        let value_to_root = value_to_root
            .into_iter()
            .map(|(value, root)| (value, canonical_root(&mut uf, root)))
            .collect();

        AliasResult { value_to_root }
    }
}

#[cfg(test)]
mod tests {
    use qcode::{
        builder::Builder,
        context::Context,
        space::{Space, SpaceId, SpaceType},
        testing::TestContext,
        value::{BasicBlock, BlockId, Varnode},
    };
    use qcode_macro::qcode;

    use crate::gvn::gvn;

    use super::{AliasResult, NodeId};

    fn make_space(ctx: &mut Context<'static>, name: &'static str) -> SpaceId {
        let mut space = Space::new(Some(name), 1, 8);
        space.ty = SpaceType::Register;
        let id = ctx.spaces.push(space);
        ctx.named_spaces.insert(name, id);
        id
    }

    fn build_in_custom_space(
        f: impl FnOnce(&mut Builder<'static, '_>, SpaceId),
    ) -> (Context<'static>, BlockId, SpaceId) {
        let mut ctx = Context::new();
        let space = make_space(&mut ctx, "register");
        let block_id = ctx.get_or_make_block(0x1000);
        let mut builder = Builder::from_context(&mut ctx, 0x1000);
        f(&mut builder, space);
        unsafe { builder.dont_finalize() };
        drop(builder);
        (ctx, block_id, space)
    }

    #[test]
    fn separate_varnodes_do_not_alias() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i32 A;
            varnode i32 B;

            <block>
                %a = load(i32, &A);
                %b = load(i32, &B);
                return [0];
        "
        );

        let result = AliasResult::simple(&ctx);
        assert!(
            !result.may_alias(A.into(), B.into()),
            "distinct non-overlapping varnodes in the same space must not alias"
        );
    }

    #[test]
    fn complex_operations_in_same_space_become_may_alias() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i32 A;
            varnode i32 B;

            <block>
                %ptr = &A + i32 4;
                %a = load(i32, %ptr);
                %b = load(i32, &A);
                return [0];
        "
        );

        let result = AliasResult::simple(&ctx);
        assert!(
            result.may_alias(ptr.into(), A.into()),
            "IR-derived pointer expressions should conservatively become may-alias"
        );
    }

    #[test]
    fn complex_operations_in_other_space_do_not_alias() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i32 A;
            varnode i32 B;

            <block>
                %ptr = &A + i32 4;
                %a = load(i32, %ptr);
                %b = load(i32, &A);
                return [0];
        "
        );

        let result = AliasResult::simple(&ctx);
        assert!(
            !result.may_alias(ptr.into(), B.into()),
            "IR-derived pointer expressions in other spaces should not become may-alias"
        );
    }

    #[test]
    fn overlapping_registers_alias() {
        let test_ctx = TestContext::new();

        let r0 = test_ctx.r0;
        let r0_lo32 = test_ctx.r0_lo32;

        let ctx = test_ctx.ctx;

        let result = AliasResult::simple(&ctx);
        assert!(
            result.may_alias(r0.into(), r0_lo32.into()),
            "overlapping registers in the same space must alias"
        );
    }

    #[test]
    fn irrelevant_instructions_and_constants_do_not_appear() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            <block>
                %sum = i32 1 + i32 2;
                return [0];
        "
        );

        let one = ctx.get_const(1, 4).id();

        let result = AliasResult::simple(&ctx);

        assert_eq!(
            result.alias_class(one),
            None,
            "plain arithmetic constants should not appear in alias analysis"
        );
        assert_eq!(
            result.alias_class(sum.into()),
            None,
            "non-pointer arithmetic instructions should not appear in alias analysis"
        );
        assert_eq!(
            result.alias_class(block.into()),
            None,
            "basic block should not appear in alias analysis"
        );
    }

    #[test]
    fn pointer_literals_are_tracked() {
        let test_ctx = TestContext::new();
        let reg_space = test_ctx.reg_space;
        let r0 = test_ctx.r0;
        let mut ctx = test_ctx.ctx;

        let _block_id = ctx.get_or_make_block(0x1000);
        let mut builder = Builder::from_context(&mut ctx, 0x1000);
        let literal_ptr = builder.context_mut().get_const(0, 8).id();
        builder.push_load::<false>(literal_ptr, 8, reg_space);
        unsafe { builder.dont_finalize() };
        drop(builder);

        let result = AliasResult::simple(&ctx);
        assert!(
            result.alias_class(literal_ptr).is_some(),
            "pointer literals used for memory accesses should appear in alias analysis"
        );
        assert!(
            result.may_alias(literal_ptr, r0.into()),
            "literal register address should alias the overlapping register varnode"
        );
    }

    #[test]
    fn literal_straddles_two_disjoint_varnode_classes_joins_them() {
        let mut a = None;
        let mut b = None;
        let mut literal_ptr = None;

        let (ctx, _, _) = build_in_custom_space(|builder, space| {
            let a_id = Varnode::make(builder.context_mut(), 0, 4, space).id;
            let b_id = Varnode::make(builder.context_mut(), 8, 4, space).id;
            let ptr = builder.context_mut().get_const(2, 8).id();

            builder.push_load::<false>(ptr, 8, space);

            a = Some(a_id);
            b = Some(b_id);
            literal_ptr = Some(ptr);
        });

        let a = a.unwrap();
        let b = b.unwrap();
        let literal_ptr = literal_ptr.unwrap();
        let result = AliasResult::simple(&ctx);

        assert!(result.may_alias(literal_ptr, a.into()));
        assert!(result.may_alias(literal_ptr, b.into()));
        assert!(result.may_alias(a.into(), b.into()));
    }

    #[test]
    fn two_literal_pointers_same_addr_alias_without_varnode() {
        let mut lit1 = None;
        let mut lit2 = None;

        let (ctx, _, _) = build_in_custom_space(|builder, space| {
            let first = builder.context_mut().get_const(0x10, 8).id();
            let second = builder
                .context_mut()
                .get_const(0xdead_beef_0000_0010, 4)
                .id();

            builder.push_load::<false>(first, 4, space);
            builder.push_load::<false>(second, 4, space);

            lit1 = Some(first);
            lit2 = Some(second);
        });

        let result = AliasResult::simple(&ctx);
        assert!(result.may_alias(lit1.unwrap(), lit2.unwrap()));
    }

    #[test]
    fn two_literal_pointers_overlapping_ranges_alias() {
        let mut lit1 = None;
        let mut lit2 = None;

        let (ctx, _, _) = build_in_custom_space(|builder, space| {
            let first = builder.context_mut().get_const(0x100, 8).id();
            let second = builder.context_mut().get_const(0x104, 8).id();

            builder.push_load::<false>(first, 8, space);
            builder.push_load::<false>(second, 4, space);

            lit1 = Some(first);
            lit2 = Some(second);
        });

        let result = AliasResult::simple(&ctx);
        assert!(result.may_alias(lit1.unwrap(), lit2.unwrap()));
    }

    #[test]
    fn two_literal_pointers_different_spaces_do_not_alias() {
        let mut ctx = Context::new();
        let reg_space = make_space(&mut ctx, "register");
        let alt_space = make_space(&mut ctx, "other");
        let _block_id = ctx.get_or_make_block(0x1000);
        let mut builder = Builder::from_context(&mut ctx, 0x1000);

        let lit1 = builder.context_mut().get_const(0x20, 8).id();
        let lit2 = builder
            .context_mut()
            .get_const(0xfeed_face_0000_0020, 4)
            .id();
        builder.push_load::<false>(lit1, 4, reg_space);
        builder.push_load::<false>(lit2, 4, alt_space);

        unsafe { builder.dont_finalize() };
        drop(builder);

        let result = AliasResult::simple(&ctx);
        assert!(!result.may_alias(lit1, lit2));
    }

    #[test]
    #[should_panic(expected = "used in multiple spaces")]
    fn same_pointer_used_in_multiple_spaces_panics() {
        let mut ctx = Context::new();
        let reg_space = make_space(&mut ctx, "register");
        let alt_space = make_space(&mut ctx, "other");
        let _block_id = ctx.get_or_make_block(0x1000);
        let mut builder = Builder::from_context(&mut ctx, 0x1000);

        let ptr = builder.context_mut().get_const(0x20, 8).id();
        builder.push_load::<false>(ptr, 4, reg_space);
        builder.push_load::<false>(ptr, 4, alt_space);

        unsafe { builder.dont_finalize() };
        drop(builder);

        let _ = AliasResult::simple(&ctx);
    }

    #[test]
    fn unresolvable_load_ptr_becomes_unknown_and_aliases_everything() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 A;

            <block>
                %ptr = i64 0x10 + i64 0x2;
                %v = load(i64, %ptr);
                return [0];
        "
        );

        let result = AliasResult::simple(&ctx);
        assert_eq!(result.alias_class(ptr.into()), Some(NodeId::Unknown));
        assert!(result.may_alias(ptr.into(), A.into()));
    }

    #[test]
    fn unresolvable_load_ptr_in_register_space_does_not_alias_registers() {
        let test_ctx = TestContext::new();
        let reg_space = test_ctx.reg_space;
        let r0 = test_ctx.r0;
        let mut ctx = test_ctx.ctx;

        let _block_id = ctx.get_or_make_block(0x1000);
        let mut builder = Builder::from_context(&mut ctx, 0x1000);
        let lhs = builder.context_mut().get_const(0x10, 8).id();
        let rhs = builder.context_mut().get_const(0x2, 8).id();
        let ptr = builder.push_add(lhs, rhs).id();
        builder.push_load::<false>(ptr, 8, reg_space);
        unsafe { builder.dont_finalize() };
        drop(builder);

        let result = AliasResult::simple(&ctx);
        let alias_class = result.alias_class(ptr);
        assert!(matches!(alias_class, Some(NodeId::Id(_))));
        assert!(
            !result.may_alias(ptr, r0.into()),
            "register-space built pointers must not alias register varnodes"
        );
    }

    #[test]
    fn store_then_load_invalidation_is_conservative_for_unknown_ptr() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 A;
            varnode i64 B;

            <block>
                %before = load(i64, &A);
                %base = load(i64, &B);
                %ptr = base * 2;
                store(%ptr, i64 0x7);
                %after = load(i64, &A);
                return [after];
        "
        );

        let aliases = AliasResult::simple(&ctx);
        assert_eq!(aliases.alias_class(ptr.into()), Some(NodeId::Unknown));

        let mut block = BasicBlock::from_id_mut(&mut ctx, block);
        assert!(block.instruction_ids().contains(&after));

        gvn(&mut block, Some(&aliases));

        assert!(
            block.instruction_ids().contains(&after),
            "unknown store pointers must conservatively invalidate cached loads"
        );
    }

    #[test]
    fn literal_with_high_bit_set_aliases_overlapping_literals() {
        let mut lit1 = None;
        let mut lit2 = None;

        let (ctx, _, _) = build_in_custom_space(|builder, space| {
            let first = builder
                .context_mut()
                .get_const(0x8000_0000_0000_0000, 8)
                .id();
            let second = builder
                .context_mut()
                .get_const(0x8000_0000_0000_0004, 8)
                .id();

            builder.push_load::<false>(first, 8, space);
            builder.push_load::<false>(second, 4, space);

            lit1 = Some(first);
            lit2 = Some(second);
        });

        let result = AliasResult::simple(&ctx);
        assert!(result.may_alias(lit1.unwrap(), lit2.unwrap()));
    }

    #[test]
    fn literal_with_upper_junk_bits_is_masked_to_size() {
        let mut a = None;
        let mut literal_ptr = None;

        let (ctx, _, _) = build_in_custom_space(|builder, space| {
            let a_id = Varnode::make(builder.context_mut(), 0x10, 4, space).id;
            let ptr = builder
                .context_mut()
                .get_const(0xdead_beef_0000_0010, 4)
                .id();

            builder.push_load::<false>(ptr, 4, space);

            a = Some(a_id);
            literal_ptr = Some(ptr);
        });

        let result = AliasResult::simple(&ctx);
        assert!(result.may_alias(literal_ptr.unwrap(), a.unwrap().into()));
    }

    #[test]
    fn many_overlapping_subregisters_still_join_in_one_class() {
        let mut varnodes = Vec::new();

        let (ctx, _, _) = build_in_custom_space(|builder, space| {
            for size in (1..=32).rev() {
                let id = Varnode::make(builder.context_mut(), 0, size, space).id;
                varnodes.push(id);
            }
        });

        let result = AliasResult::simple(&ctx);
        let first = varnodes[0];

        for &varnode in &varnodes[1..] {
            assert!(result.may_alias(first.into(), varnode.into()));
        }
    }

    #[test]
    fn pointer_arithmetic_with_non_add_sub_returns_unknown() {
        let mut ctx = Context::new();

        qcode!(
            ctx,
            "
            varnode i64 A;

            <block>
                %base = load(i64, &A);
                %ptr = base * 2;
                %v = load(i64, %ptr);
                return [v];
        "
        );

        let result = AliasResult::simple(&ctx);
        assert_eq!(result.alias_class(ptr.into()), Some(NodeId::Unknown));
        assert!(result.may_alias(ptr.into(), A.into()));
    }
}
