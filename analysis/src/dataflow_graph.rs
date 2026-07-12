use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use qcode::{
    context::Context,
    value::{
        BasicBlock, BlockId, BlockParam, Function, FunctionId, Instruction, ValueId,
        insn::{Call, Mnemonic},
    },
};

use crate::AliasResult;

#[derive(Clone, Debug)]
pub struct DataflowOptions {
    pub no_expand: HashSet<FunctionId>,
    pub max_nodes: usize,
}

impl Default for DataflowOptions {
    fn default() -> Self {
        Self {
            no_expand: HashSet::default(),
            max_nodes: 200,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DataflowGraph {
    pub root: ValueId,
    pub nodes: Vec<DfNode>,
    pub edges: Vec<DfEdge>,
    pub truncated: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DfNode {
    Root(ValueId),
    Function(FunctionId),
    ExternCall(FunctionId),
    Truncated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DfEdge {
    pub from: usize,
    pub to: usize,
    pub carrier: ValueId,
    pub kind: DfEdgeKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DfEdgeKind {
    Return,
    Arg(usize),
    Memory,
    Operand,
}

pub fn build_dataflow(
    ctx: &Context,
    root: ValueId,
    alias: &AliasResult,
    opts: &DataflowOptions,
) -> DataflowGraph {
    Builder {
        ctx,
        alias,
        opts,
        graph: DataflowGraph {
            root,
            nodes: Vec::new(),
            edges: Vec::new(),
            truncated: false,
        },
        node_of: HashMap::default(),
        seen_values: HashSet::default(),
        seen_edges: HashSet::default(),
        truncated_node: None,
    }
    .build(root)
}

struct Builder<'a, 'ctx> {
    ctx: &'a Context<'ctx>,
    alias: &'a AliasResult,
    opts: &'a DataflowOptions,
    graph: DataflowGraph,
    node_of: HashMap<DfNode, usize>,
    seen_values: HashSet<ValueId>,
    seen_edges: HashSet<(usize, usize, ValueId, EdgeKey)>,
    truncated_node: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum EdgeKey {
    Return,
    Arg(usize),
    Memory,
    Operand,
}

impl From<DfEdgeKind> for EdgeKey {
    fn from(value: DfEdgeKind) -> Self {
        match value {
            DfEdgeKind::Return => EdgeKey::Return,
            DfEdgeKind::Arg(i) => EdgeKey::Arg(i),
            DfEdgeKind::Memory => EdgeKey::Memory,
            DfEdgeKind::Operand => EdgeKey::Operand,
        }
    }
}

impl<'a, 'ctx> Builder<'a, 'ctx> {
    fn build(mut self, root: ValueId) -> DataflowGraph {
        let root_node = self.node(DfNode::Root(root));
        self.walk_value(root, root_node);
        self.graph
    }

    fn node(&mut self, node: DfNode) -> usize {
        if let Some(&id) = self.node_of.get(&node) {
            return id;
        }
        if self.graph.nodes.len() >= self.opts.max_nodes {
            self.graph.truncated = true;
            return self.truncated_node();
        }
        let id = self.graph.nodes.len();
        self.graph.nodes.push(node);
        self.node_of.insert(node, id);
        id
    }

    fn truncated_node(&mut self) -> usize {
        if let Some(id) = self.truncated_node {
            return id;
        }
        let id = self.graph.nodes.len();
        self.graph.nodes.push(DfNode::Truncated);
        self.truncated_node = Some(id);
        id
    }

    fn edge(&mut self, from: usize, to: usize, carrier: ValueId, kind: DfEdgeKind) {
        if self.seen_edges.insert((from, to, carrier, kind.into())) {
            self.graph.edges.push(DfEdge {
                from,
                to,
                carrier,
                kind,
            });
        }
    }

    fn walk_value(&mut self, value: ValueId, sink: usize) {
        if !self.seen_values.insert(value) {
            return;
        }

        match value {
            ValueId::Instruction(id) => {
                let insn = Instruction::from_id(self.ctx, id);
                let local_sink = insn
                    .function()
                    .map(|function| self.node(DfNode::Function(function.id)))
                    .unwrap_or(sink);
                if local_sink != sink {
                    self.edge(local_sink, sink, value, DfEdgeKind::Operand);
                }

                match insn.mnemonic() {
                    Mnemonic::Load(load) => {
                        self.walk_value(load.ptr, local_sink);
                        if let Some(function) = insn.function() {
                            self.add_local_stores(load.ptr, function.id, local_sink);
                        }
                    }
                    Mnemonic::Call(call) => self.add_call_flow(call, local_sink),
                    Mnemonic::CallInd(call) => {
                        self.walk_value(call.ptr, local_sink);
                        for (i, &arg) in call.args.iter().enumerate() {
                            if is_const(arg) {
                                continue;
                            }
                            self.edge(local_sink, sink, arg, DfEdgeKind::Arg(i));
                            self.walk_value(arg, local_sink);
                        }
                    }
                    _ => {
                        for arg in insn.operands() {
                            if is_const(arg) {
                                continue;
                            }
                            self.edge(local_sink, sink, arg, DfEdgeKind::Operand);
                            self.walk_value(arg, local_sink);
                        }
                    }
                }
            }
            ValueId::BlockParam(id) => {
                let param = BlockParam::from_id(self.ctx, id);
                if let Some(block) = param.parent() {
                    let local_sink = block
                        .function()
                        .map(|function| self.node(DfNode::Function(function.id)))
                        .unwrap_or(sink);
                    if local_sink != sink {
                        self.edge(local_sink, sink, value, DfEdgeKind::Operand);
                    }
                    self.add_block_param_producers(block.id, param.index(), local_sink);
                }
            }
            ValueId::Literal(_) | ValueId::Bytes(_) => {
                // Constants are invisible leaves: they carry no provenance and
                // only clutter the graph.
            }
            ValueId::Varnode(_) => {
                // A bare varnode read is a leaf: under the no-side-effect model,
                // memory is function-local and cross-function value flow travels
                // only through call arguments and return values.
            }
            ValueId::BasicBlock(_) | ValueId::Function(_) => {}
            _ => {}
        }
    }

    fn add_block_param_producers(&mut self, block: BlockId, index: usize, sink: usize) {
        for (_edge, pred) in BasicBlock::from_id(self.ctx, block).predecessors() {
            let Some(term) = BasicBlock::from_id(self.ctx, pred).instructions().last() else {
                continue;
            };
            let q = |t| BlockId::new(pred.func, t);
            let arg = match term.mnemonic() {
                Mnemonic::Branch(branch) if q(branch.target) == block => branch.args.get(index),
                Mnemonic::CBranch(branch) if q(branch.success_block) == block => {
                    branch.success_args.get(index)
                }
                Mnemonic::CBranch(branch) if q(branch.failure_block) == block => {
                    branch.failure_args.get(index)
                }
                _ => None,
            };
            if let Some(&arg) = arg
                && !is_const(arg)
            {
                self.edge(sink, sink, arg, DfEdgeKind::Operand);
                self.walk_value(arg, sink);
            }
        }
    }

    /// Could a write to `other` affect a read from `ptr`? Uses precise byte
    /// intervals when both are statically resolved (distinct constant/global
    /// addresses never overlap), and otherwise falls back to the alias graph's
    /// `may_alias` — which already rejects different spaces, isolated values,
    /// and provably-disjoint pairs.
    fn may_overlap(&self, ptr: ValueId, other: ValueId) -> bool {
        if let (Some((sa, la, ha)), Some((sb, lb, hb))) =
            (self.alias.interval(ptr), self.alias.interval(other))
        {
            return sa == sb && la < hb && lb < ha;
        }
        self.alias.may_alias(self.ctx, ptr, other)
    }

    /// Memory flow for a load, scoped to the load's own function. Functions are
    /// assumed side-effect-free (their loads/stores address function-local
    /// memory), so a load's producers are the may-aliasing stores in the *same*
    /// function — never a whole-program store scan. Cross-function value flow is
    /// carried exclusively by call arguments and return values.
    fn add_local_stores(&mut self, ptr: ValueId, fid: FunctionId, sink: usize) {
        let function = Function::from_id(self.ctx, fid);
        let stores: Vec<ValueId> = function
            .blocks()
            .flat_map(|block| block.instructions().collect::<Vec<_>>())
            .filter_map(|insn| match insn.mnemonic() {
                Mnemonic::Store(store)
                    if !is_const(store.src) && self.may_overlap(ptr, store.ptr) =>
                {
                    Some(store.src)
                }
                _ => None,
            })
            .collect();
        for src in stores {
            self.edge(sink, sink, src, DfEdgeKind::Memory);
            self.walk_value(src, sink);
        }
    }

    fn add_call_flow(&mut self, call: &Call, sink: usize) {
        let callee = Function::from_id(self.ctx, call.target);
        let callee_node = if callee.is_external() {
            self.node(DfNode::ExternCall(call.target))
        } else {
            self.node(DfNode::Function(call.target))
        };
        self.edge(
            callee_node,
            sink,
            ValueId::Function(call.target),
            DfEdgeKind::Return,
        );

        if callee.is_external() || self.opts.no_expand.contains(&call.target) {
            return;
        }

        let param_indices = self.slice_callee_params(call.target);
        let indices: Vec<usize> = if param_indices.is_empty() {
            (0..call.args.len()).collect()
        } else {
            param_indices.into_iter().collect()
        };

        for index in indices {
            let Some(&arg) = call.args.get(index) else {
                continue;
            };
            if is_const(arg) {
                continue;
            }
            self.edge(callee_node, sink, arg, DfEdgeKind::Arg(index));
            self.walk_value(arg, callee_node);
        }
    }

    fn slice_callee_params(&self, fid: FunctionId) -> HashSet<usize> {
        let mut params = HashSet::default();
        let function = Function::from_id(self.ctx, fid);
        let Some(root) = function.root() else {
            return params;
        };
        let root_params: HashMap<ValueId, usize> = root
            .params()
            .map(|p| (ValueId::BlockParam(p.id), p.index()))
            .collect();
        let mut seen = HashSet::default();
        for block in function.blocks() {
            for insn in block.instructions() {
                if let Mnemonic::Return(ret) = insn.mnemonic() {
                    collect_root_params(self.ctx, ret.ptr, &root_params, &mut seen, &mut params);
                    if let Some(value) = ret.value {
                        collect_root_params(self.ctx, value, &root_params, &mut seen, &mut params);
                    }
                }
            }
        }
        params
    }
}

/// Constants carry no provenance and are excluded from the graph.
fn is_const(value: ValueId) -> bool {
    matches!(value, ValueId::Literal(_) | ValueId::Bytes(_))
}

fn collect_root_params(
    ctx: &Context,
    value: ValueId,
    root_params: &HashMap<ValueId, usize>,
    seen: &mut HashSet<ValueId>,
    out: &mut HashSet<usize>,
) {
    if !seen.insert(value) {
        return;
    }
    if let Some(&index) = root_params.get(&value) {
        out.insert(index);
        return;
    }
    match value {
        ValueId::Instruction(id) => {
            for arg in Instruction::from_id(ctx, id).operands() {
                collect_root_params(ctx, arg, root_params, seen, out);
            }
        }
        ValueId::BlockParam(id) => {
            let param = BlockParam::from_id(ctx, id);
            let Some(block) = param.parent() else {
                return;
            };
            let index = param.index();
            for (_edge, pred) in block.predecessors() {
                let Some(term) = BasicBlock::from_id(ctx, pred).instructions().last() else {
                    continue;
                };
                let q = |t| BlockId::new(pred.func, t);
                let arg = match term.mnemonic() {
                    Mnemonic::Branch(branch) if q(branch.target) == block.id => {
                        branch.args.get(index)
                    }
                    Mnemonic::CBranch(branch) if q(branch.success_block) == block.id => {
                        branch.success_args.get(index)
                    }
                    Mnemonic::CBranch(branch) if q(branch.failure_block) == block.id => {
                        branch.failure_args.get(index)
                    }
                    _ => None,
                };
                if let Some(&arg) = arg {
                    collect_root_params(ctx, arg, root_params, seen, out);
                }
            }
        }
        _ => {}
    }
}

pub fn value_label(ctx: &Context, value: ValueId) -> String {
    match value {
        ValueId::Instruction(id) => {
            let insn = Instruction::from_id(ctx, id);
            insn.name()
                .map(|name| format!("%{name}"))
                .unwrap_or_else(|| format!("%tmp{:x}", usize::from(id.local)))
        }
        ValueId::BlockParam(_) | ValueId::Varnode(_) | ValueId::Literal(_) | ValueId::Bytes(_) => {
            // Render through the whole-`&Context` token path so a symbolic
            // block/function literal still resolves its target name (a
            // `&Shared`-backed `ValueRef` cannot — context-split 5b-ii #1).
            qcode::value::insn::segment::value_tokens(ctx, value)
                .into_iter()
                .map(|t| t.text)
                .collect()
        }
        ValueId::Function(id) => Function::from_id(ctx, id).name().to_string(),
        ValueId::BasicBlock(id) => BasicBlock::from_id(ctx, id)
            .name()
            .map(|name| format!("<{name}>"))
            .unwrap_or_else(|| format!("<bb_{:x}>", usize::from(id.local))),
        _ => format!("{value}"),
    }
}

#[cfg(test)]
mod tests {
    use qcode::{context::Context, value::ValueId};
    use qcode_macro::qcode;

    use super::*;

    fn has_function(graph: &DataflowGraph, fid: FunctionId) -> bool {
        graph
            .nodes
            .iter()
            .any(|node| matches!(node, DfNode::Function(id) if *id == fid))
    }

    /// A store in an unrelated function must not be pulled into a load's
    /// provenance: functions are assumed side-effect-free, so a load only sees
    /// stores in its own function, and cross-function flow travels through the
    /// call interface — never a whole-program memory scan.
    #[test]
    fn unrelated_function_store_is_excluded() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 g;
            fn other:
                <o_entry>
                    store(g:8, &g <- i64 0xdead);
                    return i64 0;
            fn f:
                <entry>
                    %v = load(ram:8, i64 0x2000);
                    return %v;
            "
        );
        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        let graph = build_dataflow(
            &ctx,
            ValueId::Instruction(v),
            &aliases,
            &DataflowOptions::default(),
        );
        assert!(has_function(&graph, f), "load's own function present");
        assert!(
            !has_function(&graph, other),
            "unrelated function with only a global store must be excluded"
        );
    }

    #[test]
    fn straight_line_operands_reach_root() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
                <entry>
                    %a = i64 1 + i64 2;
                    %b = %a + i64 3;
                    return %b;
            "
        );
        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        let graph = build_dataflow(
            &ctx,
            ValueId::Instruction(b),
            &aliases,
            &DataflowOptions::default(),
        );
        assert!(
            graph
                .nodes
                .iter()
                .any(|node| matches!(node, DfNode::Function(id) if *id == f))
        );
        // Constants are never materialized as nodes.
        assert!(graph.nodes.iter().all(|node| matches!(
            node,
            DfNode::Function(_) | DfNode::Root(_) | DfNode::ExternCall(_) | DfNode::Truncated
        )));
    }

    #[test]
    fn load_includes_aliasing_store() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i64 cell;
            fn f:
                <entry>
                    %c = i64 0x1234 + i64 1;
                    store(cell:8, &cell <- %c);
                    %v = load(cell:8, &cell);
                    return %v;
            "
        );
        let aliases = AliasResult::simple_for_function(&ctx, ctx.function_ids()[0]);
        let graph = build_dataflow(
            &ctx,
            ValueId::Instruction(v),
            &aliases,
            &DataflowOptions::default(),
        );
        assert!(
            graph
                .edges
                .iter()
                .any(|edge| matches!(edge.kind, DfEdgeKind::Memory))
        );
    }
}
