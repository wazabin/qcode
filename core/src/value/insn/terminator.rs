use crate::value::{LocalBlockId, LocalValueId, function::FunctionId};

use super::mnemonic::{Args, MnemonicKind};
use smallvec::{SmallVec, smallvec};

/// A statically named callee. `Real` refers to an installed function; `Minted`
/// is a pass-local placeholder that must be resolved before execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Callee {
    Real(FunctionId),
    Minted(u32),
}

impl Callee {
    pub const fn real(self) -> Option<FunctionId> {
        match self {
            Self::Real(id) => Some(id),
            Self::Minted(_) => None,
        }
    }

    pub const fn minted(self) -> Option<u32> {
        match self {
            Self::Real(_) => None,
            Self::Minted(slot) => Some(slot),
        }
    }

    /// Require an installed function at an execution-facing boundary.
    pub fn expect_real(self, operation: &str) -> FunctionId {
        match self {
            Self::Real(id) => id,
            Self::Minted(slot) => {
                panic!("{operation} requires a real callee; minted placeholder #{slot} escaped")
            }
        }
    }
}

impl From<FunctionId> for Callee {
    fn from(id: FunctionId) -> Self {
        Self::Real(id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Branch {
    /// The CFG successor, stored as a bare body-local block index. Strict IR
    /// locality (context-split ruling 2) guarantees the target lives in the same
    /// arena as this terminator, so its owning `FunctionId` is the terminator's
    /// own `id.func`.
    pub target: LocalBlockId,
    /// Arguments passed to the target block's parameters.
    pub args: Vec<LocalValueId>,
}

impl MnemonicKind for Branch {
    fn opcode(&self) -> &'static str {
        "branch"
    }

    fn is_terminator(&self) -> bool {
        true
    }

    fn args(&self) -> Args {
        SmallVec::from_vec(self.args.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BranchInd {
    pub ptr: LocalValueId,
}

impl MnemonicKind for BranchInd {
    fn opcode(&self) -> &'static str {
        "branchind"
    }

    fn is_terminator(&self) -> bool {
        true
    }

    fn args(&self) -> Args {
        smallvec![self.ptr]
    }
}

/// A tail call: an unconditional transfer of control to another *function's*
/// entry (a thunk `jmp realfunc`, or a tail `jmp`/`jcc` that the disassembler
/// resolved to a sibling function). Unlike [`Branch`], whose target is a
/// [`BlockId`] *within the same function*, a `TailCall` carries a [`Callee`]:
/// normally a real [`FunctionId`], or temporarily a pass-local minted
/// placeholder. It is a function-level terminator with no intra-function CFG
/// successor. This is the honest encoding of cross-function control flow — the
/// IR never stores a foreign [`BlockId`]. See the context-split design, ruling
/// 2 ("strict IR locality").
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TailCall {
    pub target: Callee,
    /// Values passed to the callee, one per inferred callee input, in order.
    /// Empty on the freshly-lifted IR; populated once the call interface is known.
    pub args: Vec<LocalValueId>,
}

impl MnemonicKind for TailCall {
    fn opcode(&self) -> &'static str {
        "tailcall"
    }

    fn is_terminator(&self) -> bool {
        true
    }

    fn args(&self) -> Args {
        SmallVec::from_vec(self.args.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Apply {
    pub target: Callee,
    /// Values passed to the lambda, one per root block param, in order.
    pub args: Vec<LocalValueId>,
}

impl MnemonicKind for Apply {
    fn opcode(&self) -> &'static str {
        "apply"
    }

    fn args(&self) -> Args {
        SmallVec::from_vec(self.args.clone())
    }
}

/// Per-call-site binding-convention tag (argpromote v2, `ARGPROMOTE_REGISTERS_V2.md`).
///
/// A materialized function supports two calling conventions selected per site;
/// this tag records which one a given `Call` uses and how much of the callee's
/// effect is already explicit at the site. Serialized to the `.harbinger` wire
/// so that rewritten regpure sites persist; older snapshots that predate the
/// field load as [`CallTag::Opaque`] via `#[serde(default)]`.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum CallTag {
    /// Implicit binding: the call reads its inputs from, and writes its outputs
    /// back to, the register file per the callee's interface mapping (or, for a
    /// non-materialized / ⊤ callee, clobbers conservatively). The default and
    /// the only convention on freshly-lifted IR.
    #[default]
    Opaque,
    /// The call's *register* interface is fully explicit at this site: inputs
    /// are passed as SSA `args`, outputs are read from the SSA return pack, and
    /// the call neither reads nor writes register space. Requires a materialized
    /// callee whose interface mapping the `args`/pack align with 1:1.
    RegPure,
    /// Additionally no implicit RAM effects — every effect is threaded through
    /// operands and results, so the call is a pure SSA operation. Strictly
    /// stronger than [`RegPure`](Self::RegPure). (Reserved; the RAM channel that
    /// sets it is out of scope for the register phases.)
    Pure,
}

impl CallTag {
    /// Whether the call's register interface is fully explicit at this site
    /// (`RegPure` or the stronger `Pure`): no implicit register reads/writes.
    pub fn is_regpure(self) -> bool {
        matches!(self, CallTag::RegPure | CallTag::Pure)
    }

    /// Whether the call is a fully pure SSA operation (no implicit RAM effects).
    pub fn is_pure(self) -> bool {
        matches!(self, CallTag::Pure)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Call {
    pub target: Callee,
    /// Values passed to the callee, one per inferred callee input, in order.
    pub args: Vec<LocalValueId>,
    /// Register / memory locations the call may write or alias (the callee's
    /// clobbered set plus escaping pointer arguments). These are *defs*, not
    /// reads: they are intentionally excluded from [`MnemonicKind::args`] so
    /// they do not participate in use-def bookkeeping.
    pub clobbers: Vec<LocalValueId>,
    /// Binding-convention tag (argpromote v2). Serialized so rewritten regpure
    /// sites persist; older snapshots default it to `Opaque` (see [`CallTag`]).
    #[serde(default)]
    pub tag: CallTag,
}

impl MnemonicKind for Call {
    fn opcode(&self) -> &'static str {
        "call"
    }

    fn is_terminator(&self) -> bool {
        true
    }

    fn args(&self) -> Args {
        SmallVec::from_vec(self.args.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CallInd {
    pub ptr: LocalValueId,
    pub args: Vec<LocalValueId>,
}

impl MnemonicKind for CallInd {
    fn opcode(&self) -> &'static str {
        "callind"
    }

    fn is_terminator(&self) -> bool {
        true
    }

    fn args(&self) -> Args {
        let mut args = smallvec![self.ptr];
        args.extend(self.args.clone());
        args
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CBranch {
    pub condition: LocalValueId,
    /// Taken-arm CFG successor (bare body-local index; same arena as this
    /// terminator — see [`Branch::target`]).
    pub success_block: LocalBlockId,
    /// Arguments passed to `success_block`'s parameters when the branch is taken.
    pub success_args: Vec<LocalValueId>,
    /// Fall-through CFG successor (bare body-local index; same arena as this
    /// terminator).
    pub failure_block: LocalBlockId,
    /// Arguments passed to `failure_block`'s parameters when the branch falls through.
    pub failure_args: Vec<LocalValueId>,
}

impl MnemonicKind for CBranch {
    fn opcode(&self) -> &'static str {
        "cbranch"
    }

    fn is_terminator(&self) -> bool {
        true
    }

    fn args(&self) -> Args {
        let mut args = smallvec![self.condition];
        args.extend_from_slice(&self.success_args);
        args.extend_from_slice(&self.failure_args);
        args
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Return {
    pub ptr: LocalValueId,
    pub value: Option<LocalValueId>,
}

impl MnemonicKind for Return {
    fn opcode(&self) -> &'static str {
        "return"
    }

    fn is_terminator(&self) -> bool {
        true
    }

    fn args(&self) -> Args {
        let mut args = smallvec![self.ptr];
        if let Some(value) = self.value {
            args.push(value);
        }
        args
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ReturnValue {
    pub value: LocalValueId,
}

impl MnemonicKind for ReturnValue {
    fn opcode(&self) -> &'static str {
        "returnvalue"
    }

    fn is_terminator(&self) -> bool {
        true
    }

    fn args(&self) -> Args {
        smallvec![self.value]
    }
}

/// Control flow reached bytes that do not decode to a valid instruction.
///
/// A terminator with **no successors** — the analogue of LLVM's `unreachable`.
/// It records an honest "we could not lift this" in the IR, so a failed decode
/// neither aborts the lift nor leaves a block empty and terminator-less for a
/// later pass to trip over. Dead-code elimination may prune a block ending here
/// once nothing reaches it.
///
/// Scope is deliberately narrow: **invalid bytes only**. A block that is merely
/// unlifted — a fall-through placeholder whose address a later discovery round
/// may still fill — is a different situation and must not be stamped with this,
/// or a healthy function would be poisoned mid-fixpoint.
///
/// Carries no payload: it is a marker, not a diagnostic. The reason a decode
/// failed belongs in the lifter's log and stats, where it can be counted, rather
/// than embedded in every rendered body.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BadInsn;

impl MnemonicKind for BadInsn {
    fn opcode(&self) -> &'static str {
        "badinsn"
    }

    fn is_terminator(&self) -> bool {
        true
    }

    fn args(&self) -> Args {
        smallvec![]
    }
}

#[cfg(test)]
mod tests {
    use crate::value::QCodeMut;
    use qcode_macro::qcode;

    use crate::{
        context::Context,
        testing::TestContext,
        value::{
            BasicBlock, FunctionBody, Instruction,
            insn::{Callee, Mnemonic},
        },
    };

    #[test]
    fn minted_callee_is_not_a_call_graph_target_and_renders_explicitly() {
        let mut ctx = Context::new();
        let func = FunctionBody::make(&mut ctx, "caller".into()).unwrap().id;
        let id = crate::value::InstructionRef::from_mnemonic(
            &mut ctx,
            func,
            Mnemonic::Call(super::Call {
                target: Callee::Minted(7),
                args: vec![],
                clobbers: vec![],
                tag: Default::default(),
            }),
            0,
        )
        .id;

        let insn = Instruction::from_id(&ctx, id);
        assert_eq!(insn.mnemonic().call_target(), None);
        assert_eq!(insn.as_statement().to_string(), "call fn <minted:7>();");
        assert_eq!(Callee::Minted(7).real(), None);
        assert_eq!(Callee::Minted(7).minted(), Some(7));
        assert_eq!(Callee::from(func).real(), Some(func));
    }

    #[test]
    #[should_panic(expected = "execution requires a real callee; minted placeholder #3 escaped")]
    fn execution_boundary_rejects_minted_callee() {
        Callee::Minted(3).expect_real("execution");
    }

    #[test]
    fn tail_call_is_a_function_level_terminator() {
        use crate::value::{BasicBlock, FunctionBody};

        let mut ctx = Context::new();
        let callee = FunctionBody::make_at_addr(&mut ctx, 0x2000, None).id;
        let block = {
            let f = ctx.anon_function();
            BasicBlock::make(&mut ctx, f).id
        };
        let insn = ctx.builder(block).push_tail_call(callee).id;

        let insn = Instruction::from_id(&ctx, insn);
        assert!(insn.is_terminator());
        // A tail call carries a FunctionId, is a call-graph edge, and exposes no
        // static block target (strict IR locality: no foreign BlockId).
        assert_eq!(insn.mnemonic().call_target(), Some(callee));
        assert!(insn.mnemonic().target_blocks().is_empty());
        assert_eq!(insn.as_statement().to_string(), "tailcall fn fn_2000();");
    }

    #[test]
    fn qcode_emits_branch() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <block>
                goto <done>;
            <done>
                goto <0x1001>;
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let last = block.iter().last().expect("block has instructions");
        assert!(matches!(last.mnemonic(), Mnemonic::Branch(_)));
        assert_eq!(last.as_statement().to_string(), "goto <done>;");
    }

    #[test]
    fn qcode_emits_branchind() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <block>
                local i64 ptr;
                goto [ptr];
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let last = block.iter().last().expect("block has instructions");
        assert!(matches!(last.mnemonic(), Mnemonic::BranchInd(_)));
    }

    #[test]
    fn qcode_emits_cbranch() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;

            <block>
                %c = load(cond:1, &cond);
                if %c goto <then_lbl> else goto <else_lbl>;

            <then_lbl>
                goto <0x1001>;

            <else_lbl>
                goto <0x1002>;
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let last = block.iter().last().expect("block has instructions");
        assert!(matches!(last.mnemonic(), Mnemonic::CBranch(_)));
    }

    #[test]
    fn qcode_emits_call() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <block>
                call <target>;
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let last = block.iter().last().expect("block has instructions");
        assert!(matches!(last.mnemonic(), Mnemonic::Call(_)));
    }

    #[test]
    fn call_display_shows_named_args_with_fallbacks() {
        let mut tc = TestContext::new();
        let callee = FunctionBody::make(&mut tc.ctx, "callee".into()).unwrap().id;
        FunctionBody::from_id_mut(&mut tc.ctx, callee).set_extern_interface(
            crate::value::ExternInterface {
                args: vec![crate::value::ExternArg {
                    slot: crate::value::ExternSlot::Reg(tc.r0, 8),
                    name: Some("r0".into()),
                    attrs: Default::default(),
                }],
            },
        );

        let block = {
            let __f = tc.ctx.anon_function();
            BasicBlock::make(&mut tc.ctx, __f)
        }
        .id;
        let call_id = {
            let mut builder = tc.ctx.builder(block);
            builder.push_call(callee).id
        };

        let first = tc.ctx.get_const(1u64, 8).id();
        let second = tc.ctx.get_const(2u64, 8).id();
        tc.ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(super::Call {
                target: Callee::Real(callee),
                args: vec![first.strip_func(), second.strip_func()],
                clobbers: vec![],
                tag: Default::default(),
            }),
        );

        let rendered = Instruction::from_id(&tc.ctx, call_id)
            .as_statement()
            .to_string();
        assert_eq!(rendered, "call fn callee(@r0=i64 0x1, @arg1=i64 0x2);");
    }

    #[test]
    fn call_display_names_stack_passed_arg() {
        use crate::{space::Space, value::Varnode};

        let mut tc = TestContext::new();

        // A "stack" space (addr_size = pointer width 4). A stack-passed parameter
        // is a nameless varnode in this space at the slot offset.
        let stack_space = tc.ctx.add_space(Space::new(Some("stack"), 1, 4));
        let stack_input = Varnode::make(&mut tc.ctx, 4, 4, stack_space).id;

        let callee = FunctionBody::make(&mut tc.ctx, "callee".into()).unwrap().id;
        // A stack-passed argument, named after its slot offset in the external
        // call interface (the source of truth for a bodyless callee's arg names).
        FunctionBody::from_id_mut(&mut tc.ctx, callee).set_extern_interface(
            crate::value::ExternInterface {
                args: vec![crate::value::ExternArg {
                    slot: crate::value::ExternSlot::Stack { offset: 4, size: 4 },
                    name: Some("stack_4".into()),
                    attrs: Default::default(),
                }],
            },
        );
        let _ = stack_input;

        let block = {
            let __f = tc.ctx.anon_function();
            BasicBlock::make(&mut tc.ctx, __f)
        }
        .id;
        let call_id = {
            let mut builder = tc.ctx.builder(block);
            builder.push_call(callee).id
        };

        let arg = tc.ctx.get_const(7u64, 4).id();
        tc.ctx.replace_instruction_mnemonic(
            call_id,
            Mnemonic::Call(super::Call {
                target: Callee::Real(callee),
                args: vec![arg.strip_func()],
                clobbers: vec![],
                tag: Default::default(),
            }),
        );

        let rendered = Instruction::from_id(&tc.ctx, call_id)
            .as_statement()
            .to_string();
        // The stack-passed input is named after its slot offset (varnode address 4).
        assert_eq!(rendered, "call fn callee(@stack_4=i32 0x7);");
    }

    #[test]
    fn qcode_emits_callind() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <block>
                local i64 ptr;
                call [ptr];
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let last = block.iter().last().expect("block has instructions");
        assert!(matches!(last.mnemonic(), Mnemonic::CallInd(_)));
    }

    #[test]
    fn qcode_emits_return() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <block>
                local i64 ptr;
                return at ptr;
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let last = block.iter().last().expect("block has instructions");
        assert!(matches!(last.mnemonic(), Mnemonic::Return(_)));
    }

    #[test]
    fn qcode_emits_lambda_apply_and_value_return() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            lambda rec:
            <entry @s:i64>
                %next = @s + 1;
                %out = apply rec(%next);
                return %out;
            "
        );

        let rec = FunctionBody::from_name(&ctx, "rec").expect("lambda exists");
        assert!(rec.is_lambda());
        let entry = rec.root().expect("lambda has root");
        let insns = entry.instruction_ids();
        let apply = ctx.get_insn(insns[1]);
        assert!(!apply.is_terminator(), "apply is a value instruction");
        assert!(matches!(apply.mnemonic(), Mnemonic::Apply(_)));
        assert!(matches!(
            ctx.get_insn(*insns.last().unwrap()).mnemonic(),
            Mnemonic::ReturnValue(_)
        ));
        assert!(apply.as_statement().to_string().contains("apply rec("));
        assert_eq!(
            ctx.get_insn(*insns.last().unwrap())
                .as_statement()
                .to_string(),
            "return i64 %out;"
        );
    }

    #[test]
    fn qcode_multi_block_with_label() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i32 V;

            <block>
                goto <body>;

            <body>
                %sum = i64 &V + i64 0x1;
                goto <0x1001>;
            "
        );

        // Entry block ends with a branch to "body".
        let entry = BasicBlock::from_id(&ctx, block);
        let entry_last = entry.iter().last().expect("entry has instructions");
        assert!(matches!(entry_last.mnemonic(), Mnemonic::Branch(_)));

        // "body" block contains the add instruction.
        let Mnemonic::Branch(branch) = entry_last.mnemonic() else {
            panic!("expected branch");
        };
        let body = BasicBlock::from_id(&ctx, crate::value::BlockId::new(block.func, branch.target));
        assert!(!body.is_empty());
    }

    #[test]
    fn qcode_cbranch_target_and_fallthrough_are_distinct_blocks() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            varnode i8 cond;

            <block>
                %c = load(cond:1, &cond);
                if %c goto <then_lbl> else goto <else_lbl>;

            <then_lbl>
                goto <0x1001>;

            <else_lbl>
                goto <0x1002>;
            "
        );

        let block = BasicBlock::from_id(&ctx, block);
        let last = block.iter().last().expect("block has instructions");
        let Mnemonic::CBranch(cbranch) = last.mnemonic() else {
            panic!("expected cbranch");
        };
        assert_ne!(
            cbranch.success_block, cbranch.failure_block,
            "target and fallthrough must be distinct"
        );
    }

    #[test]
    fn branch_with_args_stores_args() {
        use crate::value::ValueId;

        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <src @a>
                goto <dst @x=@a>;
            <dst @x>
                goto <0x1001>;
            "
        );

        let src_block = BasicBlock::from_id(&ctx, src);
        let last = src_block.iter().last().expect("block has instructions");
        let Mnemonic::Branch(branch) = last.mnemonic() else {
            panic!("expected branch");
        };
        assert_eq!(branch.target, dst.local);
        assert_eq!(branch.args.len(), 1);
        assert_eq!(branch.args[0], ValueId::BlockParam(a).strip_func());
    }

    #[test]
    fn cbranch_with_per_target_args_are_independent() {
        use crate::value::ValueId;

        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <src @cond:i8 @then_arg:i64 @else_arg:i64>
                if @cond goto <then_lbl @x=@then_arg> else goto <else_lbl @y=@else_arg>;
            <then_lbl @x:i64>
                goto <0x1001>;
            <else_lbl @y:i64>
                goto <0x1002>;
            "
        );

        let src_block = BasicBlock::from_id(&ctx, src);
        let insn = src_block.iter().last().expect("src has cbranch");
        let Mnemonic::CBranch(cbranch) = insn.mnemonic() else {
            panic!("expected cbranch");
        };
        assert_eq!(
            cbranch.success_args,
            [ValueId::BlockParam(then_arg).strip_func()]
        );
        assert_eq!(
            cbranch.failure_args,
            [ValueId::BlockParam(else_arg).strip_func()]
        );
        assert_ne!(cbranch.success_block, cbranch.failure_block);
    }

    #[test]
    fn branch_args_display() {
        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <src @a>
                goto <done @x=@a>;
            <done @x>
                goto <0x1001>;
            "
        );

        let src_block = BasicBlock::from_id(&ctx, src);
        let last = src_block.iter().last().expect("block has instructions");
        assert_eq!(last.as_statement().to_string(), "goto <done @x=i0 @a>;");
    }

    #[test]
    fn branch_args_are_ordered_by_target_params() {
        use crate::value::ValueId;

        let mut ctx = Context::new();
        qcode!(
            ctx,
            "
            <src @a @b>
                goto <done @y=@a @x=@b>;
            <done @x @y>
                goto <0x1001>;
            "
        );

        let src_block = BasicBlock::from_id(&ctx, src);
        let last = src_block.iter().last().expect("block has instructions");
        let Mnemonic::Branch(branch) = last.mnemonic() else {
            panic!("expected branch");
        };
        assert_eq!(
            branch.args,
            [
                ValueId::BlockParam(b).strip_func(),
                ValueId::BlockParam(a).strip_func()
            ]
        );
        assert_eq!(
            last.as_statement().to_string(),
            "goto <done @x=i0 @b @y=i0 @a>;"
        );
    }
}
