//! Shared loop recognition and induction-variable helpers.
//!
//! Several passes (`array_promote`, `loop_to_scan`, `loop_to_map`) each need the
//! same few facts about a loop: its preheader / header / body / exit blocks,
//! whether it is a rotated (do-while) or split (while) shape, its unit-step
//! induction variable with start and trip bound, and whether a value is
//! loop-invariant. This module computes those once so the passes can be thin
//! recognizers over the *values* they care about rather than re-deriving loop
//! structure inline.
//!
//! Scope (v1): the canonical single-body counted loop — a natural loop whose
//! nodes are exactly `{header, body}` (or a single self-looping block in the
//! rotated shape), with a unique preheader, a single exit block reached only from
//! the guard, and a header terminated by the guard `cbranch`. That is the shape
//! the lifter + `mem2reg` produce for `for`/`while` counted loops; anything more
//! complex is simply not recognized (the passes decline, they never miscompile).

use jstd::graph::analysis::compute_dominators;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use qcode::{
    context::Context,
    value::{
        BlockId, BlockRef, FunctionId, FunctionRef, ValueId,
        insn::{Binary, Binop, Branch, CBranch, InstructionId, IntBinop, Mnemonic},
        util::{base_ref::HostRef, host_mut::PassBacking},
    },
};

// ===========================================================================
// Small value helpers (shared by the loop passes)
// ===========================================================================

/// `c` if `v` is the integer literal `c`, else `None`.
pub(crate) fn literal<'a, 'str: 'a>(host: impl Into<HostRef<'a, 'str>>, v: ValueId) -> Option<u64> {
    let host = host.into();
    match v {
        ValueId::Literal(lid) => Some(host.shr().values.literals[lid].value),
        _ => None,
    }
}

/// `true` if `v` is `idx + 1` (either operand order) — a unit step of `idx`.
pub(crate) fn is_increment<'a, 'str: 'a>(
    host: impl Into<HostRef<'a, 'str>>,
    v: ValueId,
    idx: ValueId,
) -> bool {
    let host = host.into();
    let ValueId::Instruction(id) = v else {
        return false;
    };
    let Mnemonic::Binop(Binary { lhs, rhs, op }) = host.instruction(id).mnemonic() else {
        return false;
    };
    let one = |x: ValueId| literal(host, x) == Some(1);
    matches!(op, Binop::Int(IntBinop::Add))
        && ((*lhs == idx && one(*rhs)) || (*rhs == idx && one(*lhs)))
}

/// `true` if `v` is `idx - 1`, expressed either as `idx - 1` or as `idx + (-1)`
/// (the wrapping representation `array_promote` emits for a back-index).
pub(crate) fn is_decrement<'a, 'str: 'a>(
    host: impl Into<HostRef<'a, 'str>>,
    v: ValueId,
    idx: ValueId,
    idx_width: usize,
) -> bool {
    let host = host.into();
    let ValueId::Instruction(id) = v else {
        return false;
    };
    let Mnemonic::Binop(Binary { lhs, rhs, op }) = host.instruction(id).mnemonic() else {
        return false;
    };
    let (lhs, rhs, op) = (*lhs, *rhs, *op);
    let neg_one = if idx_width >= 8 {
        u64::MAX
    } else {
        (1u64 << (idx_width * 8)) - 1
    };
    match op {
        Binop::Int(IntBinop::Sub) => lhs == idx && literal(host, rhs) == Some(1),
        Binop::Int(IntBinop::Add) => {
            (lhs == idx && literal(host, rhs) == Some(neg_one))
                || (rhs == idx && literal(host, lhs) == Some(neg_one))
        }
        _ => false,
    }
}

/// Index of block-param `p` within `block`'s parameter list.
pub(crate) fn param_pos<'a, 'str: 'a>(
    host: impl Into<HostRef<'a, 'str>>,
    block: BlockId,
    p: ValueId,
) -> Option<usize> {
    BlockRef::new(host.into(), block)
        .params()
        .position(|q| q.id() == p)
}

/// Parent block of a block-param value (`None` if `v` is not a block param).
pub(crate) fn param_parent<'a, 'str: 'a>(
    host: impl Into<HostRef<'a, 'str>>,
    v: ValueId,
) -> Option<BlockId> {
    let ValueId::BlockParam(pid) = v else {
        return None;
    };
    host.into().block_param(pid).parent
}

/// Values feeding block-param index `k` of `block` from every predecessor edge.
pub(crate) fn incoming<'a, 'str: 'a>(
    host: impl Into<HostRef<'a, 'str>>,
    block: BlockId,
    k: usize,
) -> Vec<ValueId> {
    let host = host.into();
    let mut out = Vec::new();
    let preds: Vec<BlockId> = BlockRef::new(host, block)
        .predecessors()
        .map(|(_, p)| p)
        .collect();
    for pred in preds {
        let Some(term) = BlockRef::new(host, pred).iter().last() else {
            continue;
        };
        match term.mnemonic() {
            Mnemonic::Branch(b) => out.extend(b.args.get(k).copied()),
            Mnemonic::CBranch(cb) => {
                if cb.success_block == block {
                    out.extend(cb.success_args.get(k).copied());
                }
                if cb.failure_block == block {
                    out.extend(cb.failure_args.get(k).copied());
                }
            }
            _ => {}
        }
    }
    out
}

// ===========================================================================
// Loop-recognizer plumbing (shared by the carried-array recognizers)
// ===========================================================================

/// Exit discovery for a header terminated by the loop guard `cbranch`: the exit
/// is the successor that is **not** itself a header predecessor, and the "stay"
/// (loop-body side) is the other. `None` if the header does not end in a
/// `CBranch`, or neither/both successors are header predecessors.
pub(crate) fn cbranch_exit<'a, 'str: 'a>(
    host: impl Into<HostRef<'a, 'str>>,
    header: BlockId,
) -> Option<(BlockId, BlockId)> {
    let host = host.into();
    let hterm = BlockRef::new(host, header).iter().last()?;
    let Mnemonic::CBranch(CBranch {
        success_block: sb,
        failure_block: fb,
        ..
    }) = hterm.mnemonic()
    else {
        return None;
    };
    let (sb, fb) = (*sb, *fb);
    let preds: HashSet<BlockId> = BlockRef::new(host, header)
        .predecessors()
        .map(|(_, p)| p)
        .collect();
    if preds.contains(&sb) && !preds.contains(&fb) {
        Some((fb, sb))
    } else if preds.contains(&fb) && !preds.contains(&sb) {
        Some((sb, fb))
    } else {
        None
    }
}

/// The instructions using SSA value `v`, read from its owning function through
/// the host (empty for shared values — literals/varnodes — which the loop
/// helpers never define).
pub(crate) fn users_of<'a, 'str: 'a>(host: HostRef<'a, 'str>, v: ValueId) -> &'a [InstructionId] {
    match v.owning_function() {
        Some(f) => host.function(f).users_of(v),
        None => &[],
    }
}

/// `true` if the loop is wholly private: every value defined by `blocks` (params
/// and instructions) is used only inside `blocks`. Such a loop computes nothing
/// observable outside it once its escaping values have been rerouted, so it can
/// be deleted.
pub(crate) fn is_loop_private<'a, 'str: 'a>(
    host: impl Into<HostRef<'a, 'str>>,
    blocks: &[BlockId],
) -> bool {
    let host = host.into();
    let in_loop = |v: ValueId| {
        users_of(host, v).iter().all(|&u| {
            qcode::value::InstructionRef::new(host, u)
                .parent()
                .is_some_and(|b| blocks.contains(&b.id))
        })
    };
    blocks.iter().all(|&blk| {
        let b = BlockRef::new(host, blk);
        b.params().all(|p| in_loop(p.id())) && b.iter().all(|i| in_loop(ValueId::Instruction(i.id)))
    })
}

/// Delete a wholly-private loop: reroute `preheader`'s terminator to a plain
/// `Branch(exit, exit_args)` (re-feeding the exit's params from
/// preheader-available values), add the new CFG edge, then delete the loop
/// `blocks`. `preheader` is assumed to end in a terminator; if it somehow does
/// not, the reroute is skipped but the blocks are still deleted.
pub(crate) fn delete_private_loop<'str>(
    host: &mut PassBacking<'_, 'str>,
    fid: FunctionId,
    preheader: BlockId,
    blocks: &[BlockId],
    exit: BlockId,
    exit_args: Vec<ValueId>,
) {
    if let Some(term_id) = BlockRef::new(host.read_host(), preheader)
        .iter()
        .last()
        .map(|t| t.id)
    {
        host.replace_instruction_mnemonic(
            term_id,
            Mnemonic::Branch(Branch {
                target: exit,
                args: exit_args,
            }),
        );
        host.add_cfg_edge(preheader, exit);
    }
    for &blk in blocks {
        host.delete_block(blk, fid);
    }
}

// ===========================================================================
// Pass-through pointer resolution (union-find)
// ===========================================================================

/// Minimal union-find over `ValueId` for pass-through phi resolution.
#[derive(Default)]
struct Uf {
    parent: HashMap<ValueId, ValueId>,
}

impl Uf {
    fn find(&mut self, v: ValueId) -> ValueId {
        let p = *self.parent.get(&v).unwrap_or(&v);
        if p == v {
            return v;
        }
        let r = self.find(p);
        self.parent.insert(v, r);
        r
    }

    fn union(&mut self, a: ValueId, b: ValueId) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent.insert(ra, rb);
        }
    }
}

/// Map each value to its unique **root param** (a function-entry param), when its
/// pass-through phi class contains exactly one such root. Two threaded pointers
/// name the same loop-invariant base iff they map to the same root. A class with
/// zero or several roots is omitted (ambiguous → not resolvable).
pub fn value_roots<'a, 'str: 'a>(
    host: impl Into<HostRef<'a, 'str>>,
    fid: FunctionId,
) -> HashMap<ValueId, ValueId> {
    let host = host.into();
    let mut uf = Uf::default();
    // A representative is only ever stored as a parent *value*, never a key, so
    // track membership explicitly rather than relying on `parent.keys()`.
    let mut nodes: HashSet<ValueId> = HashSet::default();
    let blocks: Vec<BlockId> = FunctionRef::new(host, fid).iter().map(|b| b.id).collect();
    for &bid in &blocks {
        let params: Vec<ValueId> = BlockRef::new(host, bid).params().map(|p| p.id()).collect();
        for (k, p) in params.into_iter().enumerate() {
            for v in incoming(host, bid, k) {
                uf.union(p, v);
                nodes.insert(p);
                nodes.insert(v);
            }
        }
    }
    let roots: Vec<ValueId> = FunctionRef::new(host, fid)
        .root()
        .map(|r| r.params().map(|p| p.id()).collect())
        .unwrap_or_default();
    for &r in &roots {
        nodes.insert(r);
    }
    let mut rep_root: HashMap<ValueId, Option<ValueId>> = HashMap::default();
    for &r in &roots {
        let rep = uf.find(r);
        rep_root
            .entry(rep)
            .and_modify(|e| *e = None)
            .or_insert(Some(r));
    }
    let mut val_root: HashMap<ValueId, ValueId> = HashMap::default();
    for k in nodes {
        let rep = uf.find(k);
        if let Some(Some(r)) = rep_root.get(&rep) {
            val_root.insert(k, *r);
        }
    }
    val_root
}

// ===========================================================================
// Natural loop
// ===========================================================================

/// A recognized canonical counted loop.
///
/// In the **rotated** (do-while) shape `header == body`: a single block that ends
/// in the guard `cbranch`, one edge back to itself and one to `exit`. In the
/// **split** (while) shape `header` holds the guard and `body` is a distinct block
/// that ends in `goto header`.
pub struct NaturalLoop {
    pub preheader: BlockId,
    pub header: BlockId,
    pub body: BlockId,
    pub exit: BlockId,
    pub rotated: bool,
    nodes: HashSet<BlockId>,
}

/// A recognized unit-step induction variable.
pub struct Induction {
    /// The body induction parameter (`@i`), as used inside the body.
    pub var: ValueId,
    /// The value of `var` on the first body iteration.
    pub start: i64,
    /// The trip bound `N`: body indices run over `[start, N)`, last index `N-1`.
    pub count: i64,
}

/// Recognize every canonical counted loop in `fid`.
pub fn recognize_loops<'a, 'str: 'a>(
    host: impl Into<HostRef<'a, 'str>>,
    fid: FunctionId,
) -> Vec<NaturalLoop> {
    let host = host.into();
    let function = FunctionRef::new(host, fid);
    let Some(root) = function.root().map(|b| b.id) else {
        return Vec::new();
    };
    let block_ids: Vec<BlockId> = function.iter().map(|b| b.id).collect();
    let doms = compute_dominators(&function, root);

    let mut loops = Vec::new();
    for &latch in &block_ids {
        let succs: Vec<BlockId> = BlockRef::new(host, latch)
            .successors()
            .map(|(_, s)| s)
            .collect();
        for header in succs {
            if !doms.dominates(header, latch) {
                continue;
            }
            if let Some(l) = build_loop(host, latch, header) {
                loops.push(l);
            }
        }
    }
    loops
}

/// Build the [`NaturalLoop`] for a back-edge `latch -> header`, or `None` if it is
/// not the canonical single-body counted shape.
fn build_loop(host: HostRef, latch: BlockId, header: BlockId) -> Option<NaturalLoop> {
    // The natural loop: header + every block reaching the latch without going
    // *through* the header. In the rotated self-loop (`latch == header`) that is
    // just the header itself — the walk must not step back into the header's own
    // (preheader) predecessors.
    let mut nodes = HashSet::from_iter([header]);
    let mut worklist = Vec::new();
    if latch != header {
        nodes.insert(latch);
        worklist.push(latch);
    }
    while let Some(block) = worklist.pop() {
        for (_, pred) in BlockRef::new(host, block).predecessors() {
            if pred != header && nodes.insert(pred) {
                worklist.push(pred);
            }
        }
    }
    let rotated = header == latch;
    // v1: exactly the header and one body block (or a single self-looping block).
    let expected = if rotated { 1 } else { 2 };
    if nodes.len() != expected {
        return None;
    }
    // Unique preheader: the single out-of-loop predecessor of the header.
    let preheader = {
        let out_of_loop: Vec<BlockId> = BlockRef::new(host, header)
            .predecessors()
            .map(|(_, p)| p)
            .filter(|p| !nodes.contains(p))
            .collect();
        let [p] = out_of_loop[..] else {
            return None;
        };
        p
    };
    if !ends_with_goto(host, preheader, header) {
        return None;
    }
    // The guard lives on the header; its non-loop successor is the exit.
    let hterm = BlockRef::new(host, header).iter().last()?;
    let Mnemonic::CBranch(cb) = hterm.mnemonic() else {
        return None;
    };
    let (sb, fb) = (cb.success_block, cb.failure_block);
    let exit = match (nodes.contains(&sb), nodes.contains(&fb)) {
        (true, false) => fb,
        (false, true) => sb,
        _ => return None,
    };
    // The split-shape body must end in `goto header`.
    if !rotated && !ends_with_goto(host, latch, header) {
        return None;
    }
    // The exit must be entered only from the guard (header).
    let exit_preds: Vec<BlockId> = BlockRef::new(host, exit)
        .predecessors()
        .map(|(_, p)| p)
        .collect();
    if exit_preds != [header] {
        return None;
    }
    Some(NaturalLoop {
        preheader,
        header,
        body: latch,
        exit,
        rotated,
        nodes,
    })
}

/// `true` if `block`'s terminator is `goto target` (an unconditional branch).
fn ends_with_goto(host: HostRef, block: BlockId, target: BlockId) -> bool {
    matches!(
        BlockRef::new(host, block).iter().last().map(|t| t.mnemonic()),
        Some(Mnemonic::Branch(b)) if b.target == target
    )
}

impl NaturalLoop {
    /// `true` if `v` is loop-invariant: a literal, or defined (instruction or block
    /// param) outside the loop.
    pub fn is_invariant(&self, ctx: &Context, v: ValueId) -> bool {
        match v {
            ValueId::Literal(_)
            | ValueId::Bytes(_)
            | ValueId::Varnode(_)
            | ValueId::Function(_)
            | ValueId::BasicBlock(_) => true,
            ValueId::BlockParam(_) => param_parent(ctx, v).is_none_or(|b| !self.nodes.contains(&b)),
            ValueId::Instruction(i) => ctx
                .get_insn(i)
                .parent()
                .is_none_or(|b| !self.nodes.contains(&b.id)),
            _ => false,
        }
    }

    /// Verify that `var` is a unit-step induction over `[start, count)` and return
    /// `(start, count)`.
    ///
    /// `var` may be the induction as it is *used* — this reaches the loop in one of
    /// three shapes:
    ///
    /// - **rotated** self-loop: `var` is the header(==body) param; its own two
    ///   incomings are the preheader init and the back-edge increment, and the
    ///   guard tests the incremented value `i+1`.
    /// - **split, header-carried**: `var` is a header param the body reads directly
    ///   (the body takes no param of its own for it); the header param's incomings
    ///   carry init + increment, and the guard tests the header param.
    /// - **split, body-copied**: `var` is a body param that copies a header param;
    ///   the copied header param's incomings carry init + increment, and the guard
    ///   tests that header param.
    ///
    /// The trip bound `N` comes from the header guard.
    pub fn unit_induction<'a, 'str: 'a>(
        &self,
        host: impl Into<HostRef<'a, 'str>>,
        var: ValueId,
    ) -> Option<Induction> {
        let host = host.into();
        let parent = param_parent(host, var)?;
        // Resolve the feed set (init + increment), the value the guard compares,
        // and whether it compares the plain index or the incremented one.
        let (feeds, guard_key, guard_uses_inc) = if self.rotated {
            // The rotated body *is* the header; `var` must be its own param.
            if parent != self.header {
                return None;
            }
            let k = param_pos(host, self.header, var)?;
            (incoming(host, self.header, k), var, true)
        } else if parent == self.header {
            // Split loop whose body reads the header induction param directly.
            let k = param_pos(host, self.header, var)?;
            (incoming(host, self.header, k), var, false)
        } else if parent == self.body {
            // Split loop where `var` is a body param copying a header param.
            let [hp] = incoming(host, self.body, param_pos(host, self.body, var)?)[..] else {
                return None;
            };
            if param_parent(host, hp) != Some(self.header) {
                return None;
            }
            let kh = param_pos(host, self.header, hp)?;
            (incoming(host, self.header, kh), hp, false)
        } else {
            return None;
        };
        let inc = feeds
            .iter()
            .copied()
            .find(|&v| is_increment(host, v, var))?;
        let inits: Vec<i64> = feeds
            .iter()
            .filter(|&&v| v != inc)
            .filter_map(|&v| literal(host, v).map(|x| x as i64))
            .collect();
        let [start] = inits[..] else {
            return None;
        };
        // Rotated guards compare the *incremented* index (`i+1`) since the body has
        // already run at `i`; split guards compare the header param itself.
        let count = if guard_uses_inc {
            self.guard_bound(host, inc)?
        } else {
            self.guard_bound(host, guard_key)?
        };
        Some(Induction { var, start, count })
    }

    /// The trip bound `N` from the header guard `key <cmp> N`. Accepts the two
    /// canonical polarities: `key == N` exiting the loop on true, or `key < N`
    /// (unsigned/signed) continuing on true.
    fn guard_bound(&self, host: HostRef, key: ValueId) -> Option<i64> {
        let hterm = BlockRef::new(host, self.header).iter().last()?;
        let Mnemonic::CBranch(cb) = hterm.mnemonic() else {
            return None;
        };
        let exit_on_true = cb.success_block == self.exit;
        let ValueId::Instruction(id) = cb.condition else {
            return None;
        };
        let Mnemonic::Binop(Binary { lhs, rhs, op }) = host.instruction(id).mnemonic() else {
            return None;
        };
        let konst = if *lhs == key {
            *rhs
        } else if *rhs == key {
            *lhs
        } else {
            return None;
        };
        let ok = match op {
            Binop::Int(IntBinop::Equal) => exit_on_true,
            Binop::Int(IntBinop::Less | IntBinop::SLess) => !exit_on_true && *lhs == key,
            _ => false,
        };
        if !ok {
            return None;
        }
        Some(literal(host, konst)? as i64)
    }
}

#[cfg(test)]
mod tests {
    use qcode_macro::qcode;

    use super::*;
    use qcode::value::BasicBlock;

    fn param(ctx: &Context, block: BlockId, name: &str) -> ValueId {
        BasicBlock::from_id(ctx, block)
            .params()
            .find(|p| p.name() == Some(name))
            .expect("named param")
            .id()
    }

    #[test]
    fn split_counted_loop_recognized() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
            <entry @base:i64>
                goto <head @i=0 @b=@base>;
            <head @i:i64 @b:i64>
                %done = @i == 10;
                if %done goto <exit> else goto <body @j=@i @bb=@b>;
            <body @j:i64 @bb:i64>
                %off = @j * 4;
                %a = @bb + %off;
                %v = trunc(i32, @j);
                store(ram:4, %a <- %v);
                %j1 = @j + 1;
                goto <head @i=%j1 @b=@bb>;
            <exit>
                return at i64 0x0;
            "
        );
        let loops = recognize_loops(&ctx, f);
        assert_eq!(loops.len(), 1, "one loop recognized");
        let l = &loops[0];
        assert!(!l.rotated, "distinct guard header is the split shape");
        let ind = l
            .unit_induction(&ctx, param(&ctx, l.body, "j"))
            .expect("j is a unit induction var");
        assert_eq!((ind.start, ind.count), (0, 10));
    }

    #[test]
    fn rotated_counted_loop_recognized() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn g:
            <entry @base:i64>
                goto <body @i=1 @b=@base>;
            <body @i:i64 @b:i64>
                %off = @i * 4;
                %a = @b + %off;
                %v = trunc(i32, @i);
                store(ram:4, %a <- %v);
                %i1 = @i + 1;
                %done = %i1 < 10;
                if %done goto <body @i=%i1 @b=@b> else goto <exit>;
            <exit>
                return at i64 0x0;
            "
        );
        let loops = recognize_loops(&ctx, g);
        assert_eq!(loops.len(), 1, "one loop recognized");
        let l = &loops[0];
        assert!(l.rotated, "self-looping guard block is the rotated shape");
        let ind = l
            .unit_induction(&ctx, param(&ctx, l.body, "i"))
            .expect("i is a unit induction var");
        assert_eq!((ind.start, ind.count), (1, 10));
    }

    // A split loop whose body reads the *header* induction param directly, taking
    // no body param of its own for it. The induction is used as `@i` (a header
    // param) inside the body; `unit_induction` must still recognize it.
    #[test]
    fn split_header_carried_induction_recognized() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            fn f:
            <entry @base:i64>
                goto <head @i=0>;
            <head @i:i64>
                %done = @i < 10;
                if %done goto <body> else goto <exit>;
            <body>
                %off = @i * 4;
                %a = @base + %off;
                %v = trunc(i32, @i);
                store(ram:4, %a <- %v);
                %i1 = @i + 1;
                goto <head @i=%i1>;
            <exit>
                return at i64 0x0;
            "
        );
        let loops = recognize_loops(&ctx, f);
        assert_eq!(loops.len(), 1, "one loop recognized");
        let l = &loops[0];
        assert!(!l.rotated, "distinct guard header is the split shape");
        // The induction is the header param, referenced directly by the body.
        let ind = l
            .unit_induction(&ctx, param(&ctx, l.header, "i"))
            .expect("header param `i` is a unit induction var used in the body");
        assert_eq!((ind.start, ind.count), (0, 10));
    }
}
