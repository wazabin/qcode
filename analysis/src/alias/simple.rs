use rustc_hash::FxHashMap as HashMap;
use std::marker::PhantomData;

use qcode::{
    context::{Context, Shared},
    space::{SpaceId, SpaceType},
    value::{
        FunctionId, QCodeView, ValueId, ValueRef, Varnode,
        insn::{Binop, IntBinop, Mnemonic},
        literal::LiteralRef,
    },
};

use super::{AliasResult, NodeId};

#[derive(Clone, Copy)]
struct SizedNode {
    root: NodeId,
    start: u64,
    end: u64,
}

/// Union-find with union-by-rank and half-path-compression (path splitting).
/// https://en.wikipedia.org/wiki/Disjoint-set_data_structure
#[derive(Clone)]
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

            // Path splitting: point each node to its grandparent on the way up.
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

        // Either operand being Unknown propagates: Unknown aliases everything.
        let (NodeId::Id(ra), NodeId::Id(rb)) = (ra, rb) else {
            return NodeId::Unknown;
        };

        let (ra, rb) = if self.rank[ra] >= self.rank[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };

        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] += 1;
        }

        NodeId::Id(ra)
    }
}

/// Mutable scratch state threaded through the analysis passes.
struct Analysis<'base, 'ctx, 'str, R> {
    host: R,
    marker: PhantomData<&'ctx &'str ()>,

    /// Maps each tracked value to its equivalence-class root. Seeded from the
    /// shared [`RegisterBase`] (varnode entries), then grown with this function's
    /// pointer values.
    value_to_root: HashMap<ValueId, NodeId>,

    /// All varnodes in each address space, sorted by address. Borrowed read-only
    /// from the shared [`RegisterBase`]; it is function-independent so it is built
    /// once and never rebuilt per function.
    by_space: &'base HashMap<SpaceId, Vec<SizedNode>>,

    /// Literal-pointer ranges encountered during pointer resolution; grows as loads/stores are processed.
    literal_ranges: HashMap<SpaceId, Vec<SizedNode>>,

    /// The canonical root assigned to each distinct literal pointer on its first
    /// sight, plus the widest access size scanned for it so far, so repeated uses
    /// of the same interned literal short-circuit instead of re-scanning every
    /// varnode and previously-seen range (the O(uses²) term). A *wider* later use
    /// covers addresses the earlier scan never looked at, so it re-runs the scan
    /// and widens the cache. The stored node is canonicalized through `uf` on
    /// lookup, so it reflects any merges recorded after first sight.
    literal_roots: HashMap<ValueId, (NodeId, usize)>,

    /// Exact `(space, byte_start, byte_end)` for pointer values whose location
    /// was resolved to a concrete varnode or literal address.
    value_to_interval: HashMap<ValueId, (SpaceId, u64, u64)>,

    uf: UnionFind,
}

impl<'base, 'ctx, 'str: 'ctx, R: QCodeView<'ctx, 'str>> Analysis<'base, 'ctx, 'str, R> {
    fn canonical_root(&mut self, root: NodeId) -> NodeId {
        match root {
            NodeId::Unknown => NodeId::Unknown,
            NodeId::Id(_) => self.uf.find_mut(root),
        }
    }

    fn lookup_root(&mut self, value: ValueId) -> Option<NodeId> {
        let root = *self.value_to_root.get(&value)?;
        Some(self.canonical_root(root))
    }

    fn set_value_root(&mut self, value: ValueId, root: NodeId) {
        match self.value_to_root.get_mut(&value) {
            // Merge with the existing class; either Unknown wins.
            Some(existing) => {
                let new_root = match (*existing, root) {
                    (NodeId::Unknown, _) | (_, NodeId::Unknown) => NodeId::Unknown,
                    (a, b) => self.uf.join(a, b),
                };
                *existing = new_root;
            }
            None => {
                self.value_to_root.insert(value, root);
            }
        }
    }

    /// Assigns a union-find root to `literal` as a pointer into `space`, merging it
    /// with any varnode or previously-seen literal range whose address interval overlaps.
    fn assign_literal_root(&mut self, literal: ValueId, space: SpaceId, size: usize) -> NodeId {
        // A literal seen before at this size (or wider) was already merged with
        // every overlapping varnode and range; its class is stable modulo later
        // joins, which `canonical_root` reflects. Short-circuit the full rescan.
        // A wider access falls through: its interval may overlap nodes the
        // narrower scan never touched.
        if let Some(&(root, cached_size)) = self.literal_roots.get(&literal)
            && size <= cached_size
        {
            return self.canonical_root(root);
        }

        let Some((start, end)) = literal_interval(self.host.shared(), literal, size) else {
            return NodeId::Unknown;
        };

        let root_opt = self.lookup_root(literal);
        let mut root = root_opt.unwrap_or_else(|| self.uf.alloc_node());

        // Merge with any varnode whose range overlaps this literal's interval.
        // We call self.uf.find_mut directly (rather than self.canonical_root) so
        // the borrow checker can see that self.by_space and self.uf are disjoint fields.
        if let Some(varnodes) = self.by_space.get(&space) {
            for varnode in varnodes.iter().copied() {
                if overlaps(start, end, varnode.start, varnode.end) {
                    let varnode_root = self.uf.find_mut(varnode.root);
                    root = self.uf.join(root, varnode_root);
                }
            }
        }

        // Merge with any previously-registered literal range that overlaps.
        if let Some(ranges) = self.literal_ranges.get(&space) {
            for range in ranges.iter().copied() {
                if overlaps(start, end, range.start, range.end) {
                    let range_root = self.uf.find_mut(range.root);
                    root = self.uf.join(root, range_root);
                }
            }
        }

        self.literal_ranges
            .entry(space)
            .or_default()
            .push(SizedNode { root, start, end });

        self.value_to_interval.insert(literal, (space, start, end));
        self.literal_roots.insert(literal, (root, size));

        root
    }

    /// Recursively resolves the alias root for a pointer value used in a load/store.
    ///
    /// * Varnode pointers look up their pre-seeded root.
    /// * Literal pointers call `assign_literal_root` to track address ranges.
    /// * Add/Sub instructions peel off the non-pointer operand and recurse on the
    ///   pointer-typed side; anything else is unresolvable.
    fn resolve_pointer_root(&mut self, value: ValueId, space: SpaceId, size: usize) -> NodeId {
        match value {
            ValueId::Varnode(id) => {
                let (varnode_space_id, start, vn_size) = {
                    let vn = Varnode::from_id(self.host.shared(), id);
                    (vn.space().id, vn.address() as u64, vn.size() as u64)
                };
                if varnode_space_id != space {
                    log::error!(
                        "alias: load/store pointer varnode {value} lives in space {varnode_space_id:?} \
                         but the access is in {space:?}; degrading to Unknown (IR invariant from \
                         builder::push_load violated)"
                    );
                    return NodeId::Unknown;
                }
                self.value_to_interval
                    .insert(value, (space, start, start + vn_size));
                let root = self.lookup_root(value);
                root.unwrap_or(NodeId::Unknown)
            }

            ValueId::Literal(_) => self.assign_literal_root(value, space, size),

            ValueId::Instruction(id) => {
                // Decide what to peel while the ctx borrow from `mnemonic()` is live,
                // copying out only the owned operands needed, then act after it ends
                // (the recursive resolve takes `&mut self`).
                let action = match self.host.insn_ref(id).mnemonic() {
                    Mnemonic::Binop(bin)
                        if matches!(bin.op, Binop::Int(IntBinop::Add | IntBinop::Sub)) =>
                    {
                        PeelAction::AddSub(
                            bin.op,
                            bin.lhs.qualify(id.func),
                            bin.rhs.qualify(id.func),
                        )
                    }
                    // A realigned pointer `p & -2^k` (a `sub rsp,-N` frame realign, or
                    // any align-down) names the same location as `p` for aliasing: the
                    // mask only lowers the address. Peel to the non-mask operand.
                    Mnemonic::Binop(bin) if matches!(bin.op, Binop::Int(IntBinop::And)) => {
                        match align_peel_target(
                            self.host.shared(),
                            bin.lhs.qualify(id.func),
                            bin.rhs.qualify(id.func),
                        ) {
                            Some(base) => PeelAction::Peel(base),
                            None => PeelAction::Unknown,
                        }
                    }
                    // A widened/narrowed or typed-arithmetic pointer names the same
                    // location as its source. Root propagation only *adds* may-alias
                    // edges, so this stays sound; no exact interval is recorded.
                    Mnemonic::Zext(z) => PeelAction::Peel(z.src.qualify(id.func)),
                    Mnemonic::Sext(s) => PeelAction::Peel(s.src.qualify(id.func)),
                    Mnemonic::Range(r) => PeelAction::Peel(r.src.qualify(id.func)),
                    Mnemonic::Gep(g) => PeelAction::Peel(g.base.qualify(id.func)),
                    _ => PeelAction::Unknown,
                };

                match action {
                    PeelAction::AddSub(op, lhs, rhs) => {
                        if ValueRef::from_view(self.host, value).space().map(|s| s.id)
                            != Some(space)
                        {
                            return NodeId::Unknown;
                        }
                        let lhs_space = ValueRef::from_view(self.host, lhs).space().map(|s| s.id);
                        let rhs_space = ValueRef::from_view(self.host, rhs).space().map(|s| s.id);

                        if lhs_space == Some(space) && rhs_space != Some(space) {
                            self.resolve_pointer_root(lhs, space, size)
                        } else if matches!(op, Binop::Int(IntBinop::Add))
                            && rhs_space == Some(space)
                            && lhs_space != Some(space)
                        {
                            self.resolve_pointer_root(rhs, space, size)
                        } else if lhs_space.is_none() && rhs_space.is_none() {
                            NodeId::Unknown
                        } else {
                            log::error!(
                                "Odd pointer arithmetic: {value} = {lhs} {op} {rhs} with mismatched spaces {lhs_space:?} vs {rhs_space:?}; degrading to Unknown"
                            );
                            NodeId::Unknown
                        }
                    }
                    // Peel only when the source shares the access space (a typed
                    // pointer) or has none (an untyped/scalar carrier); a source in a
                    // *different* space would be an unrelated location.
                    PeelAction::Peel(src) => {
                        let src_space = ValueRef::from_view(self.host, src).space().map(|s| s.id);
                        if src_space == Some(space) || src_space.is_none() {
                            self.resolve_pointer_root(src, space, size)
                        } else {
                            NodeId::Unknown
                        }
                    }
                    PeelAction::Unknown => NodeId::Unknown,
                }
            }

            _ => NodeId::Unknown,
        }
    }
}

/// What [`Analysis::resolve_pointer_root`] does with an instruction pointer,
/// computed while the `mnemonic()` borrow is live and acted on after it ends.
enum PeelAction {
    /// `add`/`sub` — resolve the pointer-typed operand (existing behavior).
    AddSub(Binop, ValueId, ValueId),
    /// A cast / gep / align-mask — propagate the root of this single source.
    Peel(ValueId),
    /// Unresolvable — degrade to [`NodeId::Unknown`].
    Unknown,
}

/// The literal value of `v`, if it is a literal.
fn literal_value(shared: &Shared, v: ValueId) -> Option<u64> {
    match v {
        ValueId::Literal(id) => Some(LiteralRef::from_id(shared, id).value()),
        _ => None,
    }
}

/// Whether `m` is a round-down alignment mask (`-2^k`): nonzero, with the bits it
/// clears forming a contiguous low run. `x & m ≤ x`, so the masked pointer names
/// the same region as `x`. Mirrors `gvn::affine::is_round_down_mask`.
fn is_align_mask(m: u64) -> bool {
    m != 0 && (!m).wrapping_add(1).is_power_of_two()
}

/// For an `and`, the non-mask operand when the other is an alignment mask
/// (`p & -2^k`). `None` if neither operand is such a mask.
fn align_peel_target(shared: &Shared, lhs: ValueId, rhs: ValueId) -> Option<ValueId> {
    match (literal_value(shared, lhs), literal_value(shared, rhs)) {
        (Some(m), _) if is_align_mask(m) => Some(rhs),
        (_, Some(m)) if is_align_mask(m) => Some(lhs),
        _ => None,
    }
}

fn overlaps(start_a: u64, end_a: u64, start_b: u64, end_b: u64) -> bool {
    start_a < end_b && start_b < end_a
}

fn literal_interval(shared: &Shared, literal: ValueId, size: usize) -> Option<(u64, u64)> {
    let ValueId::Literal(id) = literal else {
        unreachable!("literal_interval must only be called for literals");
    };

    let start = LiteralRef::from_id(shared, id).value();
    let size = u64::try_from(size).ok()?;
    let end = start.checked_add(size)?;
    Some((start, end))
}

/// Function-independent register/varnode aliasing — "Part A" of the simple alias
/// analysis. Seeding one union-find node per varnode and merging overlapping
/// (sub-)registers within each address space depends only on the varnode/register
/// layout, not on any function body, and is invariant under the constant-folding
/// GVN does before consulting the oracle. So it is built once per varnode set and
/// shared (via `PipelineEnv`) across the per-function GVN runs, instead of being
/// rebuilt — including its O(varnodes·log) sort — on every one of them.
pub struct RegisterBase {
    /// Varnodes per space, sorted by address and sub-register-merged. Read-only
    /// after construction; lent to each per-function [`Analysis`].
    by_space: HashMap<SpaceId, Vec<SizedNode>>,
    /// Union-find seeded with every varnode node and its sub-register merges.
    /// Cloned per function (Part B mutates it: literal nodes, joins, path
    /// compression).
    uf: UnionFind,
    /// Varnode value -> root. Cloned per function, then grown with pointer values.
    value_to_root: HashMap<ValueId, NodeId>,
    /// The varnode count this base was built from. The varnode registry is
    /// append-only, so an equal count means an unchanged varnode set — the cache
    /// validity key used by `PipelineEnv`.
    varnode_count: usize,
}

impl RegisterBase {
    /// Build Part A from the module's shared varnode set.
    pub fn build(shared: &Shared) -> Self {
        let mut uf = UnionFind::new();
        let mut value_to_root: HashMap<ValueId, NodeId> = HashMap::default();
        let mut by_space: HashMap<SpaceId, Vec<SizedNode>> = HashMap::default();
        let mut varnode_count = 0usize;

        // Seed one union-find node per varnode and group by address space.
        for varnode in shared.varnodes() {
            varnode_count += 1;
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

            // Each temporary lives alone in its own freshly-minted space (see
            // `Context::make_temp_space`), so it can never overlap another varnode
            // and no literal is ever resolved in a temporary space — a temporary is
            // therefore never read out of `by_space`. Skipping the insert avoids
            // allocating one tiny singleton `Vec` (and sorting it) per temporary,
            // which is the bulk of the varnodes on real programs. `value_to_root`
            // still carries it, so its equivalence class is unchanged.
            let space = varnode.space();
            if !matches!(space.ty, SpaceType::Temporary) {
                by_space
                    .entry(space.id)
                    .or_default()
                    .push(SizedNode { root, start, end });
            }
        }

        // Within each space, sort by address and sweep to merge overlapping varnodes
        // (e.g. al/ax/eax/rax all fall into one equivalence class).
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

        Self {
            by_space,
            uf,
            value_to_root,
            varnode_count,
        }
    }

    /// Number of varnodes this base was built from; see the field docs.
    pub fn varnode_count(&self) -> usize {
        self.varnode_count
    }

    /// Finish the analysis ("Part B") for the given load/store `pointer_uses`,
    /// resolving each against a private clone of this shared base.
    fn resolve<'ctx, 'str: 'ctx>(
        &self,
        host: impl QCodeView<'ctx, 'str>,
        pointer_uses: Vec<(ValueId, SpaceId, usize)>,
    ) -> AliasResult {
        let mut a = Analysis {
            host,
            marker: PhantomData,
            value_to_root: self.value_to_root.clone(),
            by_space: &self.by_space,
            literal_ranges: HashMap::default(),
            literal_roots: HashMap::default(),
            value_to_interval: HashMap::default(),
            uf: self.uf.clone(),
        };

        // pointer_spaces guards the invariant that a given pointer value always
        // refers to the same address space across all uses.
        let mut pointer_spaces: HashMap<ValueId, SpaceId> = HashMap::default();

        for (ptr, space, size) in pointer_uses {
            if let Some(existing_space) = pointer_spaces.insert(ptr, space)
                && existing_space != space
            {
                log::warn!(
                    "alias: pointer {ptr} used in multiple spaces ({existing_space:?} vs {space:?}); \
                     degrading to Unknown"
                );
                a.set_value_root(ptr, NodeId::Unknown);
                // Also drop any exact interval recorded under the other space — it no
                // longer identifies a unique location.
                a.value_to_interval.remove(&ptr);
                continue;
            }

            let root = a.resolve_pointer_root(ptr, space, size);
            a.set_value_root(ptr, root);
        }

        // Canonicalize all roots before returning so callers see stable IDs.
        let value_to_root = a
            .value_to_root
            .into_iter()
            .map(|(value, root)| (value, a.uf.find_mut(root)))
            .collect();

        AliasResult {
            value_to_root,
            value_to_interval: a.value_to_interval,
            frame: None,
        }
    }

    /// Finish Part B for a single function: resolve only the pointers used by
    /// `fun_id`'s own loads/stores, so the module-wide instruction scan is not
    /// repeated for every function. This is the per-function GVN entry point.
    pub fn for_function<'a, 'str: 'a>(
        &self,
        host: impl QCodeView<'a, 'str>,
        fun_id: FunctionId,
    ) -> AliasResult {
        let mut pointer_uses: Vec<(ValueId, SpaceId, usize)> = Vec::new();
        for block in host.function_ref(fun_id).blocks() {
            for iid in block.instruction_ids() {
                match host.insn_ref(iid).mnemonic() {
                    Mnemonic::Load(load) => {
                        pointer_uses.push((load.ptr.qualify(iid.func), load.space, load.size))
                    }
                    Mnemonic::Store(store) => {
                        pointer_uses.push((store.ptr.qualify(iid.func), store.space, store.size))
                    }
                    _ => {}
                }
            }
        }
        self.resolve(host, pointer_uses)
    }
}

impl AliasResult {
    /// A location-based aliasing result that only reasons about known varnode
    /// ranges. Overlapping varnodes in the same space are joined; only pointer
    /// values that actually participate in `function_id`'s own loads/stores are
    /// added.
    ///
    /// Function-scoped entry point: builds a fresh [`RegisterBase`] and resolves
    /// only `function_id`'s pointers, so a whole-module instruction scan is never
    /// performed. Convenience wrapper for callers (e.g. `mem2reg`, dead-load
    /// tests) that lack a shared base to reuse; the per-function GVN pass reuses a
    /// shared base directly via [`RegisterBase::for_function`].
    pub fn simple_for_function(ctx: &Context, function_id: FunctionId) -> Self {
        RegisterBase::build(&ctx.shared)
            .for_function(qcode::value::ModuleView::new(ctx), function_id)
    }
}

#[cfg(test)]
#[path = "simple_tests.rs"]
mod tests;
