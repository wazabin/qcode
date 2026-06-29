//! Recover counted/while loops as explicit tail-recursive lambdas.
//!
//! This is a purely *structural* transform: it does not interpret the loop's
//! arithmetic at all, so it works for any condition, any counter, any state
//! update, multiple carried values, and any return shape (a tuple counts as one
//! value). A natural loop in SSA already factors into three pieces:
//!
//! ```text
//! <entry @args...>            init : () -> S     (the branch into the header)
//!   ... goto <head @s0=...>
//! <head @s...>               latch : S -> bool  (the header's own terminator)
//!   if cond goto <exit ...> else goto <body @s...>
//! <body @s...>               iter  : S -> S      (the back-edge into the header)
//!   ... goto <head @s0=...>
//! <exit @r...>
//!   return @r
//! ```
//!
//! The header `<head>` together with everything reachable from it (the body
//! region and the exit/return blocks) is moved verbatim into a fresh lambda
//! whose root is `<head>` and whose parameters are the loop-carried state `S`.
//! Each back-edge `goto <head @s0=next...>` is rewritten into
//! `%r = apply rec(next...); return %r`, turning one loop iteration into one
//! recursive call. The original entry becomes `%r = apply rec(init...);
//! return %r`.
//!
//! Because the header/body/exit blocks are reused unchanged, the transform is
//! correct by construction — there is no recurrence to derive and get wrong.

use std::borrow::Cow;
use std::collections::HashSet;

use qcode::{
    builder::Builder,
    context::Context,
    value::{
        BasicBlock, Function, FunctionId, ValueId,
        block::BlockId,
        insn::{Branch, InstructionId, Mnemonic},
    },
};

use crate::{FunctionPass, PipelineEnv};

#[derive(Default)]
pub struct LoopToRecursion;

impl FunctionPass for LoopToRecursion {
    const NAME: &'static str = "loop_to_recursion";

    fn description(&self) -> &'static str {
        "Recover counted loops as recursive lambda applications"
    }

    fn run(
        &self,
        ctx: &mut Context,
        fun_id: FunctionId,
        _env: &PipelineEnv,
    ) -> Result<bool, String> {
        Ok(loop_to_recursion(ctx, fun_id))
    }
}

crate::register_function_pass!(LoopToRecursion);

#[derive(Debug, Clone)]
struct LoopModel {
    /// The host lambda's entry block (stays in the host, performs `init`).
    root: BlockId,
    /// The loop header; becomes the recursive lambda's root.
    head: BlockId,
    /// Arguments forwarded from `root` into `head` — the initial state `S`.
    init_args: Vec<ValueId>,
    /// `head` plus every block reachable from it; moved into the new lambda.
    region: Vec<BlockId>,
    /// Latch blocks: each ends in `goto head(next...)` and is rewritten into a
    /// recursive call. Paired with the state passed on the back-edge.
    back_edges: Vec<(BlockId, Vec<ValueId>)>,
}

pub fn loop_to_recursion(ctx: &mut Context, fun_id: FunctionId) -> bool {
    let Some(model) = recognize_loop(ctx, fun_id) else {
        return false;
    };
    transform(ctx, fun_id, &model);
    true
}

fn recognize_loop(ctx: &Context, fun_id: FunctionId) -> Option<LoopModel> {
    let fun = Function::from_id(ctx, fun_id);
    if !fun.is_lambda() {
        return None;
    }
    let root = fun.root()?.id;

    // The entry must unconditionally branch into the header, carrying `init`.
    let (head, init_args) = match terminator_mnemonic(ctx, root)? {
        Mnemonic::Branch(Branch { target, args }) => (*target, args.clone()),
        _ => return None,
    };
    if head == root {
        return None;
    }
    let arity = block_param_count(ctx, head);
    if init_args.len() != arity {
        return None;
    }

    // The loop is the header and everything it dominates. Since `root`'s only
    // successor is `head`, the region is reachable solely through the header, so
    // moving it out cannot strand any code the entry still needs.
    let region = reachable_from(ctx, fun_id, head);
    if region.contains(&root) {
        return None;
    }

    // Every predecessor of the header is either the entry or a back-edge. Each
    // back-edge must be a clean latch (`goto head(next...)`) so we can rewrite
    // it into a recursive call without disturbing other control flow.
    let mut back_edges = Vec::new();
    for (_edge, pred) in BasicBlock::from_id(ctx, head).predecessors() {
        if pred == root {
            continue;
        }
        if !region.contains(&pred) {
            // An entry into the loop from outside the region we are extracting.
            return None;
        }
        match terminator_mnemonic(ctx, pred)? {
            Mnemonic::Branch(Branch { target, args }) if *target == head => {
                if args.len() != arity {
                    return None;
                }
                back_edges.push((pred, args.clone()));
            }
            _ => return None,
        }
    }
    if back_edges.is_empty() {
        return None;
    }

    let mut region: Vec<BlockId> = region.into_iter().collect();
    region.sort_by_key(|&b| usize::from(b));

    Some(LoopModel {
        root,
        head,
        init_args,
        region,
        back_edges,
    })
}

fn transform(ctx: &mut Context, host: FunctionId, model: &LoopModel) {
    let host_name = Function::from_id(ctx, host).name().to_owned();
    let name = ctx.get_unique_name(Cow::Owned(format!("{host_name}_rec")));
    let rec = Function::make_lambda(ctx, name)
        .expect("recursive lambda name was deduplicated")
        .id;

    // Move the header/body/exit region out of the host and into the new lambda.
    for &block in &model.region {
        Function::from_id_mut(ctx, host).remove_block(block);
        Function::from_id_mut(ctx, rec).add_block(block);
    }
    Function::from_id_mut(ctx, rec)
        .set_root(model.head)
        .expect("header has no conflicting address");

    // Each back-edge becomes a recursive call: one loop iteration per frame.
    for (pred, next_args) in &model.back_edges {
        replace_terminator_with_apply(ctx, *pred, rec, next_args.clone());
    }

    // The host entry seeds the recursion with the initial state and returns it.
    replace_terminator_with_apply(ctx, model.root, rec, model.init_args.clone());
}

/// Drops `block`'s terminator and appends `%r = apply target(args); return %r`.
fn replace_terminator_with_apply(
    ctx: &mut Context,
    block: BlockId,
    target: FunctionId,
    args: Vec<ValueId>,
) {
    if let Some(term) = terminator_id(ctx, block) {
        ctx.remove_instruction(term);
    }
    let mut b = Builder::from_block(BasicBlock::from_id_mut(ctx, block));
    let out = b.push_apply(target, args).id();
    b.push_return_value(out);
}

/// Blocks reachable from `start` within `fun_id`, following CFG successors.
fn reachable_from(ctx: &Context, fun_id: FunctionId, start: BlockId) -> HashSet<BlockId> {
    let mut seen = HashSet::new();
    let mut stack = vec![start];
    while let Some(block) = stack.pop() {
        if !seen.insert(block) {
            continue;
        }
        for (_edge, succ) in BasicBlock::from_id(ctx, block).successors() {
            let same_fun = BasicBlock::from_id(ctx, succ)
                .function()
                .is_some_and(|f| f.id == fun_id);
            if same_fun {
                stack.push(succ);
            }
        }
    }
    seen
}

fn block_param_count(ctx: &Context, block: BlockId) -> usize {
    BasicBlock::from_id(ctx, block).params().count()
}

/// The id of `block`'s terminator instruction, if it ends in one.
fn terminator_id(ctx: &Context, block: BlockId) -> Option<InstructionId> {
    let &id = BasicBlock::from_id(ctx, block).instruction_ids().last()?;
    ctx.get_insn(id).is_terminator().then_some(id)
}

/// A borrow of `block`'s terminator mnemonic, avoiding a full clone.
fn terminator_mnemonic<'a>(ctx: &'a Context, block: BlockId) -> Option<&'a Mnemonic> {
    let id = terminator_id(ctx, block)?;
    Some(ctx.get_insn(id).mnemonic())
}

#[cfg(test)]
mod tests {
    use super::*;
    use qcode_emulator::{SizedValue, StandaloneEmulator};
    use qcode_macro::qcode;

    fn run(ctx: &Context, fun: FunctionId, n: u64) -> Option<u64> {
        let root = Function::from_id(ctx, fun).root().expect("root").id;
        let ret = match terminator_mnemonic(ctx, root)? {
            Mnemonic::ReturnValue(r) => r.value,
            _ => return None,
        };
        let mut emu = StandaloneEmulator::new(root);
        emu.run_pure(ctx, fun, &[SizedValue::new(n, 8)], 100_000)
            .expect("rewritten loop runs");
        emu.get_value(ctx, ret)
    }

    #[test]
    fn rewrites_iterative_fibonacci_loop_to_recursion() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda fib_loop:
            <entry @input:i64>
                goto <head @hn=@input @a=0 @b=1>;

            <head @hn:i64 @a:i64 @b:i64>
                %done = @hn == 0;
                if %done goto <exit @r=@a> else goto <body @m=@hn @x=@a @y=@b>;

            <body @m:i64 @x:i64 @y:i64>
                %next = @x + @y;
                %m1 = @m - 1;
                goto <head @hn=%m1 @a=@y @b=%next>;

            <exit @r:i64>
                return @r;
            "
        );

        assert!(loop_to_recursion(&mut ctx, fib_loop));

        // The host entry is now just `apply rec(init); return`.
        let fib = Function::from_id(&ctx, fib_loop);
        let root = fib.root().expect("root");
        let insns = root.instruction_ids();
        assert!(matches!(
            ctx.get_insn(*insns.last().unwrap()).mnemonic(),
            Mnemonic::ReturnValue(_)
        ));
        // The loop blocks were moved out of the host: only the entry remains.
        assert_eq!(fib.blocks().count(), 1);

        let rec = Function::from_name(&ctx, "fib_loop_rec").expect("recursive lambda exists");
        assert!(rec.is_lambda());
        assert!(rec.to_string().contains("apply fib_loop_rec"));

        assert_eq!(run(&ctx, fib_loop, 6), Some(8));
        assert_eq!(run(&ctx, fib_loop, 10), Some(55));
    }

    /// A *non-linear* recurrence (a running product) that the old linear-algebra
    /// recognizer could never have handled — the structural transform doesn't
    /// care about the arithmetic at all.
    #[test]
    fn rewrites_factorial_loop_to_recursion() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda fact:
            <entry @input:i64>
                goto <head @n=@input @acc=1>;

            <head @n:i64 @acc:i64>
                %done = @n == 0;
                if %done goto <exit @r=@acc> else goto <body @m=@n @a=@acc>;

            <body @m:i64 @a:i64>
                %na = @a * @m;
                %m1 = @m - 1;
                goto <head @n=%m1 @acc=%na>;

            <exit @r:i64>
                return @r;
            "
        );

        assert!(loop_to_recursion(&mut ctx, fact));
        assert!(Function::from_name(&ctx, "fact_rec").unwrap().is_lambda());

        assert_eq!(run(&ctx, fact, 5), Some(120));
        assert_eq!(run(&ctx, fact, 0), Some(1));
        assert_eq!(run(&ctx, fact, 6), Some(720));
    }

    /// A loop that counts *up* and compares against a carried bound — a
    /// different counter and a different comparison from the Fibonacci shape.
    #[test]
    fn rewrites_count_up_sum_loop() {
        let mut ctx = Context::new();
        // sum of 0..input via an ascending counter compared to the bound.
        qcode!(
            ctx,
            "
            lambda sum_up:
            <entry @input:i64>
                goto <head @i=0 @s=0 @bound=@input>;

            <head @i:i64 @s:i64 @bound:i64>
                %done = @i == @bound;
                if %done goto <exit @r=@s> else goto <body @j=@i @t=@s @bd=@bound>;

            <body @j:i64 @t:i64 @bd:i64>
                %s2 = @t + @j;
                %j1 = @j + 1;
                goto <head @i=%j1 @s=%s2 @bound=@bd>;

            <exit @r:i64>
                return @r;
            "
        );

        assert!(loop_to_recursion(&mut ctx, sum_up));
        // 0+1+2+3+4 = 10
        assert_eq!(run(&ctx, sum_up, 5), Some(10));
        assert_eq!(run(&ctx, sum_up, 1), Some(0));
    }

    #[test]
    fn leaves_non_loop_lambda_untouched() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda straight:
            <entry @input:i64>
                %x = @input + 1;
                return %x;
            "
        );
        assert!(!loop_to_recursion(&mut ctx, straight));
    }
}
