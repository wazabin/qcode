//! Backward program slice (the *projection*) of a function onto one of its
//! return values.
//!
//! [`project_return`] computes which instructions, block params, and — crucially
//! — which **root (input) params** a given return-tuple field depends on,
//! following both **data** and **control** dependencies. Pure-function emulation
//! in constant propagation uses the input-param set as its soundness gate: a
//! field whose projection touches only constant call arguments can be emulated
//! and harvested (see `PURE_EMULATION_DESIGN.md`).
//!
//! The result is owned and self-contained so the GUI / headless inspector can
//! render a function's projection onto a chosen return value.
//!
//! ## Control dependence
//!
//! In this block-argument SSA, a value that differs by control-flow path is a
//! block param fed by differing branch arguments. The data slice already pulls
//! in every incoming argument, but the *choice* between predecessors is decided
//! by a `cbranch` condition that is not a data operand of the param. To stay
//! sound for poison-fill emulation, the projection therefore also includes the
//! condition of any `cbranch` that can route control to a block contributing to
//! the slice. This is a conservative over-approximation of true
//! (post-dominator-frontier) control dependence: it can pull in a condition that
//! does not actually affect the field, which only makes the gate *stricter*
//! (harvest less), never unsound. It can be tightened later without changing the
//! interface.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::collections::BTreeSet;

use qcode::{
    context::Context,
    value::{
        BasicBlock, Function, ValueId,
        block::BlockId,
        block_param::BlockParamId,
        function::FunctionId,
        insn::{Branch, CBranch, InstructionId, Mnemonic, Return, Tuple},
    },
};

/// The backward slice of a function onto one return value.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Projection {
    /// Instructions the return value transitively depends on (data + control).
    pub insns: HashSet<InstructionId>,
    /// Block params the return value transitively depends on.
    pub block_params: HashSet<BlockParamId>,
    /// Indices of the function's **root (input) params** the value depends on.
    /// This is the set the emulation gate checks against the literal call args.
    pub input_params: BTreeSet<usize>,
    /// `true` if the slice reached a value it could not resolve to a constant,
    /// an input param, or a tracked instruction (e.g. a raw `Varnode` read or an
    /// unmappable predecessor). Such a field cannot be proven independent of any
    /// symbolic input and must not be harvested.
    pub opaque: bool,
}

impl Projection {
    /// Whether this field can be emulated given the set of call-argument indices
    /// that are constant literals: the slice must be fully resolved and depend
    /// only on input params that are passed a literal.
    pub fn is_constant_over(&self, literal_arg_indices: &HashSet<usize>) -> bool {
        !self.opaque
            && self
                .input_params
                .iter()
                .all(|i| literal_arg_indices.contains(i))
    }
}

/// Project pure function `fid` onto field `field` of its returned aggregate.
///
/// Returns `None` if the function has no root or no return carries a value for
/// `field` (e.g. `field` is out of range of the returned tuple). A field that is
/// itself an aggregate is left to the caller's scalar-only handling; the slice is
/// still computed over its components.
pub fn project_return(ctx: &Context, fid: FunctionId, field: usize) -> Option<Projection> {
    let root = ctx.values.functions[fid].root?;

    // Map every instruction to its block once, and collect the block list, for
    // control-dependence reachability below.
    let mut insn_block: HashMap<InstructionId, BlockId> = HashMap::default();
    let mut blocks: Vec<BlockId> = Vec::new();
    for block in Function::from_id(ctx, fid).iter() {
        let bid = block.id;
        blocks.push(bid);
        for &iid in BasicBlock::from_id(ctx, bid).instruction_ids() {
            insn_block.insert(iid, bid);
        }
    }

    // Seed values: field `field` of every return block's returned aggregate.
    let mut seeds: Vec<ValueId> = Vec::new();
    let mut any_return = false;
    for &bid in &blocks {
        let Some(value) = return_value_of(ctx, bid) else {
            continue;
        };
        if let Some(seed) = aggregate_field(ctx, value, field) {
            any_return = true;
            seeds.push(seed);
        }
    }
    if !any_return {
        return None;
    }

    let mut proj = Projection::default();
    let mut visited: HashSet<ValueId> = HashSet::default();
    let mut worklist: Vec<ValueId> = seeds;

    // Blocks of values already in the slice — drives control-dependence seeding.
    let mut sliced_blocks: HashSet<BlockId> = HashSet::default();
    // cbranch condition values already fed into the worklist, to avoid repeats.
    let mut seeded_conditions: HashSet<ValueId> = HashSet::default();

    loop {
        // Drain the data/value worklist to a fixpoint.
        while let Some(v) = worklist.pop() {
            if !visited.insert(v) {
                continue;
            }
            match v {
                ValueId::Literal(_) => {} // constant leaf
                ValueId::Instruction(id) => {
                    proj.insns.insert(id);
                    if let Some(&b) = insn_block.get(&id) {
                        sliced_blocks.insert(b);
                    }
                    for arg in ctx.get_insn(id).mnemonic().args() {
                        worklist.push(arg);
                    }
                }
                ValueId::BlockParam(pid) => {
                    proj.block_params.insert(pid);
                    let param = &ctx.values.block_params[pid];
                    let Some(parent) = param.parent else {
                        proj.opaque = true;
                        continue;
                    };
                    sliced_blocks.insert(parent);
                    if parent == root {
                        // A function input.
                        proj.input_params.insert(param.index);
                        continue;
                    }
                    // A merge param: its value is whatever each predecessor passes
                    // at this param index.
                    for incoming in predecessor_args(ctx, parent, param.index) {
                        match incoming {
                            Some(arg) => worklist.push(arg),
                            None => proj.opaque = true,
                        }
                    }
                }
                // A pure function reads no registers/globals; a raw varnode (or a
                // function pointer / block value) cannot be proven constant.
                ValueId::Varnode(_) | ValueId::BasicBlock(_) | ValueId::Function(_) => {
                    proj.opaque = true;
                }
                _ => proj.opaque = true,
            }
        }

        // Control dependence: include the condition of every cbranch that can
        // route control to a block in the slice. Iterate until no new condition
        // is discovered (a condition can grow the slice with new blocks).
        let mut grew = false;
        for &bid in &blocks {
            let Some(Mnemonic::CBranch(CBranch { condition, .. })) = terminator_of(ctx, bid) else {
                continue;
            };
            if seeded_conditions.contains(&condition) {
                continue;
            }
            if reaches_any(ctx, bid, &sliced_blocks) {
                seeded_conditions.insert(condition);
                worklist.push(condition);
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }

    Some(proj)
}

/// Field `field` of the aggregate a block returns, if it ends in a value-carrying
/// `return`. Used to read a concrete return value back out after emulation.
pub fn return_field(ctx: &Context, bid: BlockId, field: usize) -> Option<ValueId> {
    let value = return_value_of(ctx, bid)?;
    aggregate_field(ctx, value, field)
}

/// The `value` of a block's `return`, if it ends in one carrying a value.
fn return_value_of(ctx: &Context, bid: BlockId) -> Option<ValueId> {
    match terminator_of(ctx, bid)? {
        Mnemonic::Return(Return { value, .. }) => value,
        _ => None,
    }
}

/// Field `field` of an aggregate value: the corresponding `Tuple` field, or the
/// value itself when `field == 0` and it is not a tuple.
fn aggregate_field(ctx: &Context, value: ValueId, field: usize) -> Option<ValueId> {
    if let ValueId::Instruction(id) = value
        && let Mnemonic::Tuple(Tuple { fields }) = ctx.get_insn(id).mnemonic()
    {
        return fields.get(field).copied();
    }
    (field == 0).then_some(value)
}

/// The mnemonic of a block's terminator (its last instruction), cloned.
fn terminator_of(ctx: &Context, bid: BlockId) -> Option<Mnemonic> {
    BasicBlock::from_id(ctx, bid)
        .iter()
        .last()
        .filter(|insn| insn.is_terminator())
        .map(|insn| insn.mnemonic().clone())
}

/// For each predecessor of `bid`, the argument it passes to param index
/// `param_index`. `None` for a predecessor whose terminator cannot be mapped to
/// a branch argument list (marks the projection opaque).
fn predecessor_args(ctx: &Context, bid: BlockId, param_index: usize) -> Vec<Option<ValueId>> {
    let preds: Vec<BlockId> = BasicBlock::from_id(ctx, bid)
        .predecessors()
        .map(|(_, p)| p)
        .collect();
    preds
        .into_iter()
        .map(|pred| match terminator_of(ctx, pred) {
            Some(Mnemonic::Branch(Branch { target, args })) if target == bid => {
                args.get(param_index).copied()
            }
            Some(Mnemonic::CBranch(CBranch {
                success_block,
                success_args,
                failure_block,
                failure_args,
                ..
            })) => {
                if success_block == bid {
                    success_args.get(param_index).copied()
                } else if failure_block == bid {
                    failure_args.get(param_index).copied()
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect()
}

/// Whether any block in `targets` is reachable from `from` via CFG successor
/// edges (excluding `from` itself, which is the cbranch's own block).
fn reaches_any(ctx: &Context, from: BlockId, targets: &HashSet<BlockId>) -> bool {
    let mut seen = HashSet::from_iter([from]);
    let mut stack = vec![from];
    while let Some(b) = stack.pop() {
        for (_, succ) in BasicBlock::from_id(ctx, b).successors() {
            if targets.contains(&succ) {
                return true;
            }
            if seen.insert(succ) {
                stack.push(succ);
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode::{
        builder::Builder,
        testing::TestContext,
        value::{BasicBlock, Function, Value, insn::Return},
    };

    /// Set a return instruction's value channel to `value` (the builder's
    /// `push_return` leaves it `None`; argpromote normally fills it).
    fn set_return_value(ctx: &mut Context, ret: ValueId, ptr: ValueId, value: ValueId) {
        let ValueId::Instruction(iid) = ret else {
            panic!("return must be an instruction");
        };
        ctx.replace_instruction_mnemonic(
            iid,
            Mnemonic::Return(Return {
                ptr,
                value: Some(value),
            }),
        );
    }

    fn param_index(ctx: &Context, v: ValueId) -> usize {
        let ValueId::BlockParam(pid) = v else {
            panic!("expected a block param");
        };
        ctx.values.block_params[pid].index
    }

    /// `foo(a, b) = (a, b*69 + 42)` — a straight-line pure function. Field 0 is
    /// the symbolic param `a`; field 1 depends only on `b`.
    #[test]
    fn projection_data_dependencies_are_per_field() {
        let mut tc = TestContext::new();
        let fid = Function::make(&mut tc.ctx, "foo".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }

        let (a, b);
        let (ret, ptr, tuple);
        {
            let mut bld = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            a = bld.push_param(8).id();
            b = bld.push_param(8).id();
            let c69 = bld.context_mut().get_const(69, 8).id();
            let c42 = bld.context_mut().get_const(42, 8).id();
            let b69 = bld.push_mul(b, c69).id();
            let body = bld.push_add(b69, c42).id();
            tuple = bld.push_tuple(vec![a, body]).id();
            ptr = bld.context_mut().get_const(0x2000, 8).id();
            ret = bld.push_return(ptr).id();
            unsafe { bld.dont_finalize() };
        }
        set_return_value(&mut tc.ctx, ret, ptr, tuple);

        let f0 = project_return(&tc.ctx, fid, 0).expect("field 0");
        let f1 = project_return(&tc.ctx, fid, 1).expect("field 1");

        assert!(!f0.opaque && !f1.opaque);
        assert_eq!(
            f0.input_params,
            BTreeSet::from([param_index(&tc.ctx, a)]),
            "field 0 is just `a`"
        );
        assert_eq!(
            f1.input_params,
            BTreeSet::from([param_index(&tc.ctx, b)]),
            "field 1 = b*69+42 depends only on `b`"
        );

        // Gate: field 1 is harvestable when `b` (index 1) is a literal arg.
        assert!(f1.is_constant_over(&HashSet::from_iter([1])));
        assert!(!f1.is_constant_over(&HashSet::from_iter([0])));
    }

    /// A merge value selected by a `cbranch` on `a` must pick up `a` through the
    /// **control** dependence, even though both incoming arguments are constants.
    #[test]
    fn projection_includes_control_dependence() {
        let mut tc = TestContext::new();
        let fid = Function::make(&mut tc.ctx, "g".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        let t = tc.ctx.get_or_make_block(0x1100);
        let fb = tc.ctx.get_or_make_block(0x1200);
        let m = tc.ctx.get_or_make_block(0x1300);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
            f.add_block(t);
            f.add_block(fb);
            f.add_block(m);
        }

        let a;
        {
            let mut bld = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            a = bld.push_param(8).id();
            let _b = bld.push_param(8).id();
            let zero = bld.context_mut().get_const(0, 8).id();
            let cond = bld.push_ne(a, zero).id();
            bld.push_cbranch(cond, t, fb);
            unsafe { bld.dont_finalize() };
        }
        {
            let mut bld = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, t));
            let one = bld.context_mut().get_const(1, 8).id();
            bld.push_branch_with_args(m, vec![one]);
            unsafe { bld.dont_finalize() };
        }
        {
            let mut bld = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, fb));
            let two = bld.context_mut().get_const(2, 8).id();
            bld.push_branch_with_args(m, vec![two]);
            unsafe { bld.dont_finalize() };
        }
        let (ret, ptr, tuple);
        {
            let mut bld = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, m));
            let x = bld.push_param(8).id();
            tuple = bld.push_tuple(vec![x]).id();
            ptr = bld.context_mut().get_const(0x2000, 8).id();
            ret = bld.push_return(ptr).id();
            unsafe { bld.dont_finalize() };
        }
        set_return_value(&mut tc.ctx, ret, ptr, tuple);

        let f0 = project_return(&tc.ctx, fid, 0).expect("field 0");
        assert!(!f0.opaque);
        assert_eq!(
            f0.input_params,
            BTreeSet::from([param_index(&tc.ctx, a)]),
            "the merge value is constant per path but its selection depends on `a`"
        );
    }

    /// A raw varnode read (a non-pure leaf) makes the projection opaque, so the
    /// field is never harvested.
    #[test]
    fn projection_opaque_on_varnode() {
        let mut tc = TestContext::new();
        let fid = Function::make(&mut tc.ctx, "h".into()).unwrap().id;
        let entry = tc.ctx.get_or_make_block(0x1000);
        {
            let mut f = Function::from_id_mut(&mut tc.ctx, fid);
            f.set_root(entry).unwrap();
            f.add_block(entry);
        }
        let reg = ValueId::Varnode(tc.r0);
        let (ret, ptr, tuple);
        {
            let mut bld = Builder::from_block(BasicBlock::from_id_mut(&mut tc.ctx, entry));
            tuple = bld.push_tuple(vec![reg]).id();
            ptr = bld.context_mut().get_const(0x2000, 8).id();
            ret = bld.push_return(ptr).id();
            unsafe { bld.dont_finalize() };
        }
        set_return_value(&mut tc.ctx, ret, ptr, tuple);

        let f0 = project_return(&tc.ctx, fid, 0).expect("field 0");
        assert!(f0.opaque, "a varnode read cannot be proven constant");
        assert!(!f0.is_constant_over(&HashSet::from_iter([0, 1])));
    }
}
