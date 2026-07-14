//! Disposable whole-module call-relationship analysis.
//!
//! [`CallGraph`] is derived from one immutable [`Context`] snapshot. It owns no
//! IR and is never stored in or serialized with the context. Structural IR
//! mutation invalidates a graph; mutation consumers must extract the stable IDs
//! they need, drop the graph, and rebuild before their next relationship query.

use jstd::{Identifier, registry::Registry};
use qcode::{
    address_index::AddressIndex,
    context::Context,
    value::{
        FunctionBody, FunctionId, ValueId,
        insn::{InstructionId, Mnemonic},
    },
};
use rustc_hash::FxHashMap as HashMap;

/// Stable identifier of one edge inside a [`CallGraph`] snapshot.
#[derive(Identifier)]
pub struct CallEdgeId(usize);

/// The IR/discovery construct that contributes a call edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallKind {
    /// A statically targeted `Call`, `Apply`, `Map`, or `Scan`.
    Direct,
    /// A statically targeted function-level `TailCall` terminator.
    Tail,
    /// A `CallInd` through a computed value.
    Indirect,
    /// A semantic discovery edge with no backing instruction.
    Synthetic,
}

/// Target of a call edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallTarget {
    /// A statically known installed function.
    Function(FunctionId),
    /// The qualified pointer value used by an unresolved `CallInd`.
    Indirect(ValueId),
}

/// One canonical call relationship, stored exactly once in [`CallGraph::edges`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CallEdge {
    pub caller: FunctionId,
    /// The backing instruction, absent only for [`CallKind::Synthetic`].
    pub site: Option<InstructionId>,
    pub target: CallTarget,
    pub kind: CallKind,
}

/// Explicit disposable call-relationship snapshot over an unchanged [`Context`].
#[derive(Clone, Debug, Default)]
pub struct CallGraph {
    edges: Registry<CallEdgeId, CallEdge>,
    outgoing: HashMap<FunctionId, Vec<CallEdgeId>>,
    incoming: HashMap<FunctionId, Vec<CallEdgeId>>,
}

impl CallGraph {
    /// Derive the complete call graph from the context's current live IR and
    /// semantic synthetic-discovery inputs.
    pub fn analyze(ctx: &Context<'_>) -> Self {
        let mut graph = Self::default();
        let addresses = AddressIndex::analyze(ctx);

        for caller in ctx.function_ids() {
            for block in FunctionBody::from_id(ctx, caller).blocks() {
                for insn in block.instructions() {
                    let edge = match insn.mnemonic() {
                        Mnemonic::Call(call) => call.target.real().map(|target| CallEdge {
                            caller,
                            site: Some(insn.id),
                            target: CallTarget::Function(target),
                            kind: CallKind::Direct,
                        }),
                        Mnemonic::TailCall(call) => call.target.real().map(|target| CallEdge {
                            caller,
                            site: Some(insn.id),
                            target: CallTarget::Function(target),
                            kind: CallKind::Tail,
                        }),
                        Mnemonic::Apply(apply) => apply.target.real().map(|target| CallEdge {
                            caller,
                            site: Some(insn.id),
                            target: CallTarget::Function(target),
                            kind: CallKind::Direct,
                        }),
                        Mnemonic::Map(map) => map.body.real().map(|target| CallEdge {
                            caller,
                            site: Some(insn.id),
                            target: CallTarget::Function(target),
                            kind: CallKind::Direct,
                        }),
                        Mnemonic::Scan(scan) => scan.body.real().map(|target| CallEdge {
                            caller,
                            site: Some(insn.id),
                            target: CallTarget::Function(target),
                            kind: CallKind::Direct,
                        }),
                        Mnemonic::CallInd(call) => Some(CallEdge {
                            caller,
                            site: Some(insn.id),
                            target: CallTarget::Indirect(call.ptr.qualify(caller)),
                            kind: CallKind::Indirect,
                        }),
                        _ => None,
                    };

                    debug_assert!(
                        edge.is_some()
                            || !matches!(
                                insn.mnemonic(),
                                Mnemonic::Call(_)
                                    | Mnemonic::TailCall(_)
                                    | Mnemonic::Apply(_)
                                    | Mnemonic::Map(_)
                                    | Mnemonic::Scan(_)
                            ),
                        "unresolved minted callee escaped into CallGraph::analyze"
                    );
                    if let Some(edge) = edge {
                        graph.push(edge);
                    }
                }
            }

            for address in ctx.shared.values.synthetic_callees_of(caller) {
                if let Some(target) = addresses.function_at(address) {
                    graph.push(CallEdge {
                        caller,
                        site: None,
                        target: CallTarget::Function(target),
                        kind: CallKind::Synthetic,
                    });
                }
            }
        }

        graph
    }

    fn push(&mut self, edge: CallEdge) -> CallEdgeId {
        let caller = edge.caller;
        let target = match edge.target {
            CallTarget::Function(target) => Some(target),
            CallTarget::Indirect(_) => None,
        };
        let id = self.edges.push(edge);
        self.outgoing.entry(caller).or_default().push(id);
        if let Some(target) = target {
            self.incoming.entry(target).or_default().push(id);
        }
        id
    }

    /// Number of canonical edges in this snapshot.
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Iterate canonical edge payloads in deterministic construction order.
    pub fn edges(&self) -> impl Iterator<Item = (CallEdgeId, &CallEdge)> {
        self.edges.iter().map(|edge| (edge.id, *edge))
    }

    /// Resolve one edge ID.
    pub fn edge(&self, id: CallEdgeId) -> &CallEdge {
        &self.edges[id]
    }

    /// Raw outgoing edge IDs for `caller`, in deterministic construction order.
    pub fn outgoing_edges(&self, caller: FunctionId) -> &[CallEdgeId] {
        self.outgoing.get(&caller).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Raw incoming edge IDs for `callee`. Indirect edges have no incoming entry.
    pub fn incoming_edges(&self, callee: FunctionId) -> &[CallEdgeId] {
        self.incoming.get(&callee).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Known callees of `caller`, deduplicated and sorted by function ID.
    pub fn callees(&self, caller: FunctionId) -> Vec<FunctionId> {
        let mut functions = self
            .outgoing_edges(caller)
            .iter()
            .filter_map(|&edge| match self.edge(edge).target {
                CallTarget::Function(target) => Some(target),
                CallTarget::Indirect(_) => None,
            })
            .collect::<Vec<_>>();
        functions.sort_unstable();
        functions.dedup();
        functions
    }

    /// Known callers of `callee`, deduplicated and sorted by function ID.
    pub fn callers(&self, callee: FunctionId) -> Vec<FunctionId> {
        let mut functions = self
            .incoming_edges(callee)
            .iter()
            .map(|&edge| self.edge(edge).caller)
            .collect::<Vec<_>>();
        functions.sort_unstable();
        functions.dedup();
        functions
    }

    /// Instruction-backed incoming sites for `callee`, sorted by instruction ID.
    /// Synthetic edges have no site and are omitted.
    pub fn call_sites(&self, callee: FunctionId) -> Vec<InstructionId> {
        let mut sites = self
            .incoming_edges(callee)
            .iter()
            .filter_map(|&edge| self.edge(edge).site)
            .collect::<Vec<_>>();
        sites.sort_unstable();
        sites.dedup();
        sites
    }

    /// Whether `caller` has at least one unresolved indirect call edge.
    pub fn has_indirect_call(&self, caller: FunctionId) -> bool {
        self.outgoing_edges(caller)
            .iter()
            .any(|&edge| self.edge(edge).kind == CallKind::Indirect)
    }
}

#[cfg(test)]
mod tests {
    use qcode::{
        builder::Builder,
        context::Context,
        value::{BasicBlock, FunctionBody},
    };

    use super::*;

    fn function(ctx: &mut Context<'static>, name: &str) -> (FunctionId, qcode::value::BlockId) {
        let id = FunctionBody::make(ctx, name.to_owned().into()).unwrap().id;
        let block = BasicBlock::make(ctx, id).id;
        (id, block)
    }

    fn new_block(ctx: &mut Context<'static>, function: FunctionId) -> qcode::value::BlockId {
        BasicBlock::make(ctx, function).id
    }

    #[test]
    fn stores_each_ir_edge_once_and_indexes_only_known_targets_incoming() {
        let mut ctx = Context::new();
        let (caller, block) = function(&mut ctx, "caller");
        let (first, _) = function(&mut ctx, "first");
        let (second, _) = function(&mut ctx, "second");
        let ptr = ctx.get_const(0x1234, 8).id();

        let direct_second = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, block))
            .push_call(second)
            .id;
        let site = new_block(&mut ctx, caller);
        let direct_first = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, site))
            .push_call(first)
            .id;
        let site = new_block(&mut ctx, caller);
        let duplicate_first = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, site))
            .push_call(first)
            .id;
        let site = new_block(&mut ctx, caller);
        let tail = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, site))
            .push_tail_call(second)
            .id;
        let site = new_block(&mut ctx, caller);
        let indirect = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, site))
            .push_call_ind(ptr)
            .id;

        let graph = CallGraph::analyze(&ctx);

        assert_eq!(graph.len(), 5);
        assert_eq!(graph.callees(caller), vec![first, second]);
        assert_eq!(graph.callers(first), vec![caller]);
        assert_eq!(graph.callers(second), vec![caller]);
        assert!(graph.has_indirect_call(caller));
        assert_eq!(graph.incoming_edges(first).len(), 2);
        assert_eq!(graph.incoming_edges(second).len(), 2);
        assert_eq!(graph.outgoing_edges(caller).len(), 5);

        let mut first_sites = vec![direct_first, duplicate_first];
        first_sites.sort_unstable();
        assert_eq!(graph.call_sites(first), first_sites);
        let mut second_sites = vec![direct_second, tail];
        second_sites.sort_unstable();
        assert_eq!(graph.call_sites(second), second_sites);

        let (indirect_edge_id, indirect_edge) = graph
            .edges()
            .find(|(_, edge)| edge.site == Some(indirect))
            .unwrap();
        assert_eq!(indirect_edge.kind, CallKind::Indirect);
        assert_eq!(indirect_edge.target, CallTarget::Indirect(ptr));
        assert!(
            graph
                .incoming
                .values()
                .flatten()
                .all(|&id| id != indirect_edge_id)
        );

        let tail_edge = graph
            .edges()
            .map(|(_, edge)| edge)
            .find(|edge| edge.site == Some(tail))
            .unwrap();
        assert_eq!(tail_edge.kind, CallKind::Tail);
    }

    #[test]
    fn apply_map_and_scan_are_direct_like_edges() {
        let mut ctx = Context::new();
        let (caller, block) = function(&mut ctx, "caller");
        let (apply_body, _) = function(&mut ctx, "apply_body");
        let (map_body, _) = function(&mut ctx, "map_body");
        let (scan_body, _) = function(&mut ctx, "scan_body");
        let value = ctx.get_const(1, 8).id();

        let apply = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, block))
            .push_apply(apply_body, vec![value])
            .id;
        let map = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, block))
            .push_map(map_body, value, Vec::new())
            .id;
        let scan = Builder::from_block(BasicBlock::from_id_mut(&mut ctx, block))
            .push_scan(scan_body, value, value, Vec::new())
            .id;

        let graph = CallGraph::analyze(&ctx);
        assert_eq!(graph.callees(caller), vec![apply_body, map_body, scan_body]);
        for site in [apply, map, scan] {
            let edge = graph
                .edges()
                .map(|(_, edge)| edge)
                .find(|edge| edge.site == Some(site))
                .unwrap();
            assert_eq!(edge.kind, CallKind::Direct);
        }
    }

    #[test]
    fn synthetic_edge_appears_only_after_its_address_resolves() {
        let mut ctx = Context::new();
        let caller = FunctionBody::make_at_addr(&mut ctx, 0x1000, Some("caller".into())).id;
        assert!(ctx.shared.values.add_synthetic_callee(caller, 0x2000));

        let before = CallGraph::analyze(&ctx);
        assert!(before.is_empty());

        let target = FunctionBody::make_at_addr(&mut ctx, 0x2000, Some("target".into())).id;
        let after = CallGraph::analyze(&ctx);
        assert_eq!(after.callees(caller), vec![target]);
        assert_eq!(after.callers(target), vec![caller]);
        assert!(after.call_sites(target).is_empty());
        let edge = after.edges().next().unwrap().1;
        assert_eq!(edge.kind, CallKind::Synthetic);
        assert_eq!(edge.site, None);
    }

    #[test]
    fn callers_and_callees_are_deduplicated_and_sorted_including_self_recursion() {
        let mut ctx = Context::new();
        let (low, low_block) = function(&mut ctx, "low");
        let (middle, middle_block) = function(&mut ctx, "middle");
        let (high, _) = function(&mut ctx, "high");

        Builder::from_block(BasicBlock::from_id_mut(&mut ctx, middle_block)).push_call(high);
        let site = new_block(&mut ctx, middle);
        Builder::from_block(BasicBlock::from_id_mut(&mut ctx, site)).push_call(low);
        let site = new_block(&mut ctx, middle);
        Builder::from_block(BasicBlock::from_id_mut(&mut ctx, site)).push_call(high);
        let site = new_block(&mut ctx, middle);
        Builder::from_block(BasicBlock::from_id_mut(&mut ctx, site)).push_call(middle);
        Builder::from_block(BasicBlock::from_id_mut(&mut ctx, low_block)).push_call(high);

        let graph = CallGraph::analyze(&ctx);
        assert_eq!(graph.callees(middle), vec![low, middle, high]);
        assert_eq!(graph.callers(high), vec![low, middle]);
        assert_eq!(graph.callers(middle), vec![middle]);
    }

    #[test]
    fn snapshots_do_not_change_when_the_ir_changes() {
        let mut ctx = Context::new();
        let (caller, block) = function(&mut ctx, "caller");
        let (callee, _) = function(&mut ctx, "callee");
        let before = CallGraph::analyze(&ctx);

        Builder::from_block(BasicBlock::from_id_mut(&mut ctx, block)).push_call(callee);

        assert!(before.callees(caller).is_empty());
        assert_eq!(CallGraph::analyze(&ctx).callees(caller), vec![callee]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "unresolved minted callee escaped into CallGraph::analyze")]
    fn unresolved_minted_callee_is_rejected_at_the_snapshot_boundary() {
        use qcode::value::insn::Callee;

        let mut ctx = Context::new();
        let (_, block) = function(&mut ctx, "caller");
        Builder::from_block(BasicBlock::from_id_mut(&mut ctx, block)).push_call(Callee::Minted(0));

        let _ = CallGraph::analyze(&ctx);
    }
}
