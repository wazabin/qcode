use crate::value::{
    LocalValueId,
    function::FunctionId,
    insn::{
        Apply, Assert, BadInsn, Binary, Branch, BranchInd, CBranch, Call, CallInd, Carry, Extract,
        FloatToFloat, FloatToInt, Gep, IntToFloat, IntrinsicApp, IsFloatNaN, Load, LzCount, Map,
        PCodeOp, PopCount, Range, Return, ReturnValue, SBorrow, SCarry, Scan, Sext, Store,
        TailCall, Tuple, Unary, Zext,
    },
};
use smallvec::SmallVec;

/// Operand list returned by [`MnemonicKind::args`]. Inline-stores up to two
/// operands (covering every fixed-arity instruction — binops, casts, loads,
/// flags, …), so the pervasive per-instruction operand walks in the analysis
/// passes don't heap-allocate. Variable-arity ops (calls, tuples, `scan`) spill
/// to the heap only when they exceed two operands.
pub type Args = SmallVec<[LocalValueId; 2]>;

/// Implemented by each concrete instruction type.
///
/// Provides the common interface that [`Mnemonic`] dispatches to: a short
/// opcode string, argument enumeration, and terminator status.
pub trait MnemonicKind {
    /// Short textual opcode, e.g. `"load"`, `"int_add"`, `"branch"`.
    fn opcode(&self) -> &'static str;

    // TODO: replace with a visitor pattern to avoid the need for this method
    /// Returns the [`LocalValueId`]s of all operands consumed by this instruction.
    fn args(&self) -> Args;

    /// Returns `true` if this instruction ends a basic block.
    ///
    /// Terminators are: [`Branch`], [`CBranch`], [`BranchInd`], [`Call`],
    /// [`CallInd`], and [`Return`].
    fn is_terminator(&self) -> bool {
        false
    }
}

/// The operation performed by an [`Instruction`](crate::value::Instruction).
///
/// `Mnemonic` is a closed enum over all supported IR operations.  It is
/// `#[non_exhaustive]` so that new operations can be added without requiring
/// downstream crates to update exhaustive match arms.
///
/// # Categories
///
/// | Variants | Category |
/// |---|---|
/// | [`Load`], [`Store`] | Memory access |
/// | [`Branch`], [`CBranch`], [`BranchInd`], [`Call`], [`CallInd`], [`Return`], [`ReturnValue`], [`BadInsn`] | Control flow (terminators) |
/// | [`Unop`](Mnemonic::Unop) | Unary integer/float/bool operations |
/// | [`Binop`](Mnemonic::Binop) | Binary integer/float/bool operations |
/// | [`Zext`], [`Sext`], [`Range`], [`IntToFloat`], [`FloatToInt`], [`FloatToFloat`] | Type casts and bit extraction |
/// | [`IsFloatNaN`], [`PopCount`], [`LzCount`], [`Carry`], [`SCarry`], [`SBorrow`] | Bit/flag operations |
/// | [`PCodeOp`] | User-defined or architecture-specific operation |
/// | [`Intrinsic`] | Pure named intrinsic function (e.g. `rol`, `ror`) |
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Mnemonic {
    /// Load a value from a memory space.
    Load(Load),
    /// Store a value to a memory space.
    Store(Store),
    /// Unconditional direct branch to a static target block.
    Branch(Branch),
    /// Conditional branch: taken when the condition operand is non-zero.
    CBranch(CBranch),
    /// Unconditional indirect branch to a dynamically-computed address.
    BranchInd(BranchInd),
    /// Direct call to a known function.
    Call(Call),
    /// Tail call: a function-level transfer of control to another function's
    /// entry (thunk / tail jump). Carries a [`FunctionId`], never a foreign
    /// block — see [`TailCall`].
    TailCall(TailCall),
    /// Value-level application of a pure lambda function.
    Apply(Apply),
    /// Indirect call through a computed function pointer.
    CallInd(CallInd),
    /// Return from the current function.
    Return(Return),
    /// Value return from a lambda function.
    ReturnValue(ReturnValue),
    /// Bytes that do not decode to a valid instruction. A terminator with no
    /// successors (see [`BadInsn`]).
    BadInsn(BadInsn),
    /// A unary integer, float, or boolean operation.
    Unop(Unary),
    /// A binary integer, float, or boolean operation.
    Binop(Binary),
    /// Extract a contiguous byte range from a value.
    Range(Range),
    /// Convert an integer to a floating-point value.
    IntToFloat(IntToFloat),
    /// Convert a floating-point value to a different float width.
    FloatToFloat(FloatToFloat),
    /// Convert a floating-point value to an integer (truncate toward zero).
    FloatToInt(FloatToInt),
    /// Zero-extend a value to a wider integer.
    Zext(Zext),
    /// Sign-extend a value to a wider integer.
    Sext(Sext),
    /// Test whether a floating-point value is NaN.
    IsFloatNaN(IsFloatNaN),
    /// Count the number of set bits (population count / Hamming weight).
    PopCount(PopCount),
    /// Count leading zero bits.
    LzCount(LzCount),
    /// Unsigned addition carry-out flag.
    Carry(Carry),
    /// Signed addition carry-out (overflow) flag.
    SCarry(SCarry),
    /// Signed subtraction borrow flag.
    SBorrow(SBorrow),
    /// Assert when resolving an execution trace
    Assert(Assert),
    /// A user-defined or architecture-specific p-code operation.
    PCodeOp(PCodeOp),
    /// A pure named intrinsic function (e.g. `rol`, `ror`). Categorically pure:
    /// no memory or observable side effects.
    Intrinsic(IntrinsicApp),
    /// Build an aggregate (tuple) value from ordered fields.
    Tuple(Tuple),
    /// Project a single field out of an aggregate value.
    Extract(Extract),
    /// Compute the address of a struct field (typed, named pointer arithmetic).
    Gep(Gep),
    /// Total element-wise map over an array value (a projectable loop).
    Map(Map),
    /// Total left-scan (prefix fold) over an array value: a projectable loop
    /// whose per-element write depends on the previous iteration's result.
    Scan(Scan),
}

impl Mnemonic {
    /// The unresolved direct-callee slot carried by this mnemonic, if any.
    pub fn minted_callee_slot(&self) -> Option<u32> {
        match self {
            Self::Call(call) => call.target.minted(),
            Self::TailCall(call) => call.target.minted(),
            Self::Apply(apply) => apply.target.minted(),
            Self::Map(map) => map.body.minted(),
            Self::Scan(scan) => scan.body.minted(),
            _ => None,
        }
    }

    /// Resolve one pass-local direct-callee placeholder to its installed
    /// function ID. Returns whether this mnemonic contained that placeholder.
    pub fn resolve_minted_callee(&mut self, slot: u32, real: FunctionId) -> bool {
        let callee = match self {
            Self::Call(call) => Some(&mut call.target),
            Self::TailCall(call) => Some(&mut call.target),
            Self::Apply(apply) => Some(&mut apply.target),
            Self::Map(map) => Some(&mut map.body),
            Self::Scan(scan) => Some(&mut scan.body),
            _ => None,
        };
        let Some(callee) = callee else {
            return false;
        };
        if *callee != super::Callee::Minted(slot) {
            return false;
        }
        *callee = super::Callee::Real(real);
        true
    }

    fn as_kind(&self) -> &dyn MnemonicKind {
        match self {
            Mnemonic::Load(m) => m,
            Mnemonic::Store(m) => m,
            Mnemonic::Branch(m) => m,
            Mnemonic::CBranch(m) => m,
            Mnemonic::BranchInd(m) => m,
            Mnemonic::Call(m) => m,
            Mnemonic::TailCall(m) => m,
            Mnemonic::Apply(m) => m,
            Mnemonic::CallInd(m) => m,
            Mnemonic::Return(m) => m,
            Mnemonic::ReturnValue(m) => m,
            Mnemonic::BadInsn(m) => m,
            Mnemonic::Range(m) => m,
            Mnemonic::Unop(m) => m,
            Mnemonic::Binop(m) => m,
            Mnemonic::IsFloatNaN(m) => m,
            Mnemonic::IntToFloat(m) => m,
            Mnemonic::FloatToFloat(m) => m,
            Mnemonic::FloatToInt(m) => m,
            Mnemonic::Zext(m) => m,
            Mnemonic::Sext(m) => m,
            Mnemonic::PopCount(m) => m,
            Mnemonic::LzCount(m) => m,
            Mnemonic::Carry(m) => m,
            Mnemonic::SCarry(m) => m,
            Mnemonic::SBorrow(m) => m,
            Mnemonic::Assert(m) => m,
            Mnemonic::PCodeOp(m) => m,
            Mnemonic::Intrinsic(m) => m,
            Mnemonic::Tuple(m) => m,
            Mnemonic::Extract(m) => m,
            Mnemonic::Gep(m) => m,
            Mnemonic::Map(m) => m,
            Mnemonic::Scan(m) => m,
        }
    }

    pub fn opcode(&self) -> &'static str {
        self.as_kind().opcode()
    }

    pub fn is_terminator(&self) -> bool {
        self.as_kind().is_terminator()
    }

    /// Whether an instruction must be kept even if its result has no users:
    /// it writes memory, transfers control, calls, asserts, or invokes an
    /// opaque p-code op. This is the single source of truth shared by DCE
    /// (which must not delete these) and the emitter (which must always print
    /// them); keep the two in agreement by routing both through here.
    pub fn has_side_effects(&self) -> bool {
        matches!(
            self,
            Mnemonic::Store(_)
                | Mnemonic::Call(_)
                | Mnemonic::CallInd(_)
                | Mnemonic::PCodeOp(_)
                | Mnemonic::Assert(_)
        ) || self.is_terminator()
    }

    /// The callee of a direct [`Call`] or body of a [`Map`], or `None`
    /// (including indirect [`CallInd`] calls, whose target is not statically
    /// known). Used to maintain the reverse call graph.
    pub fn call_target(&self) -> Option<FunctionId> {
        match self {
            Mnemonic::Call(call) => call.target.real(),
            Mnemonic::TailCall(tc) => tc.target.real(),
            Mnemonic::Apply(apply) => apply.target.real(),
            Mnemonic::Map(map) => map.body.real(),
            Mnemonic::Scan(scan) => scan.body.real(),
            _ => None,
        }
    }

    /// The statically-known CFG target blocks this mnemonic branches to — the
    /// `Branch` target and both `CBranch` arms — as bare body-local indices. Empty
    /// for non-branch or indirect terminators (a `BranchInd` resolves to computed
    /// addresses, not a static block). Strict IR locality (context-split ruling 2)
    /// guarantees each target lives in this terminator's own arena, so a caller
    /// with the owning function in hand recovers the full `BlockId` via
    /// `BlockId::new(func, local)`.
    pub fn target_blocks(&self) -> smallvec::SmallVec<[crate::value::LocalBlockId; 2]> {
        match self {
            Mnemonic::Branch(b) => smallvec::smallvec![b.target],
            Mnemonic::CBranch(c) => smallvec::smallvec![c.success_block, c.failure_block],
            _ => smallvec::SmallVec::new(),
        }
    }

    pub fn args(&self) -> Args {
        self.as_kind().args()
    }

    /// Replace every occurrence of `old` with `new` in this instruction's operands.
    pub fn replace_value(&mut self, old: LocalValueId, new: LocalValueId) {
        match self {
            Mnemonic::Load(m) => {
                if m.ptr == old {
                    m.ptr = new;
                }
            }
            Mnemonic::Store(m) => {
                if m.ptr == old {
                    m.ptr = new;
                }
                if m.src == old {
                    m.src = new;
                }
            }
            Mnemonic::CBranch(m) => {
                if m.condition == old {
                    m.condition = new;
                }
                m.success_args.iter_mut().for_each(|a| {
                    if *a == old {
                        *a = new;
                    }
                });
                m.failure_args.iter_mut().for_each(|a| {
                    if *a == old {
                        *a = new;
                    }
                });
            }
            Mnemonic::BranchInd(m) => {
                if m.ptr == old {
                    m.ptr = new;
                }
            }
            Mnemonic::Call(m) => {
                m.args.iter_mut().for_each(|a| {
                    if *a == old {
                        *a = new;
                    }
                });
            }
            Mnemonic::TailCall(m) => {
                m.args.iter_mut().for_each(|a| {
                    if *a == old {
                        *a = new;
                    }
                });
            }
            Mnemonic::Apply(m) => {
                m.args.iter_mut().for_each(|a| {
                    if *a == old {
                        *a = new;
                    }
                });
            }
            Mnemonic::CallInd(m) => {
                if m.ptr == old {
                    m.ptr = new;
                }
                m.args.iter_mut().for_each(|a| {
                    if *a == old {
                        *a = new;
                    }
                });
            }
            Mnemonic::Return(m) => {
                if m.ptr == old {
                    m.ptr = new;
                }
                if let Some(v) = m.value.as_mut()
                    && *v == old
                {
                    *v = new;
                }
            }
            Mnemonic::ReturnValue(m) => {
                if m.value == old {
                    m.value = new;
                }
            }
            // No operands to rewrite.
            Mnemonic::BadInsn(_) => {}
            Mnemonic::Unop(m) => {
                if m.src == old {
                    m.src = new;
                }
            }
            Mnemonic::Binop(m) => {
                if m.lhs == old {
                    m.lhs = new;
                }
                if m.rhs == old {
                    m.rhs = new;
                }
            }
            Mnemonic::Range(m) => {
                if m.src == old {
                    m.src = new;
                }
            }
            Mnemonic::Zext(m) => {
                if m.src == old {
                    m.src = new;
                }
            }
            Mnemonic::Sext(m) => {
                if m.src == old {
                    m.src = new;
                }
            }
            Mnemonic::IntToFloat(m) => {
                if m.src == old {
                    m.src = new;
                }
            }
            Mnemonic::FloatToFloat(m) => {
                if m.src == old {
                    m.src = new;
                }
            }
            Mnemonic::FloatToInt(m) => {
                if m.src == old {
                    m.src = new;
                }
            }
            Mnemonic::IsFloatNaN(m) => {
                if m.src == old {
                    m.src = new;
                }
            }
            Mnemonic::PopCount(m) => {
                if m.src == old {
                    m.src = new;
                }
            }
            Mnemonic::LzCount(m) => {
                if m.src == old {
                    m.src = new;
                }
            }
            Mnemonic::Carry(m) => {
                if m.lhs == old {
                    m.lhs = new;
                }
                if m.rhs == old {
                    m.rhs = new;
                }
            }
            Mnemonic::SCarry(m) => {
                if m.lhs == old {
                    m.lhs = new;
                }
                if m.rhs == old {
                    m.rhs = new;
                }
            }
            Mnemonic::SBorrow(m) => {
                if m.lhs == old {
                    m.lhs = new;
                }
                if m.rhs == old {
                    m.rhs = new;
                }
            }
            Mnemonic::PCodeOp(m) => {
                m.args.iter_mut().for_each(|a| {
                    if *a == old {
                        *a = new;
                    }
                });
                if let Some(v) = m.dst.as_mut()
                    && *v == old
                {
                    *v = new;
                }
            }
            Mnemonic::Branch(m) => {
                m.args.iter_mut().for_each(|a| {
                    if *a == old {
                        *a = new;
                    }
                });
            }
            Mnemonic::Intrinsic(m) => {
                m.args.iter_mut().for_each(|a| {
                    if *a == old {
                        *a = new;
                    }
                });
            }
            Mnemonic::Tuple(m) => {
                m.fields.iter_mut().for_each(|a| {
                    if *a == old {
                        *a = new;
                    }
                });
            }
            Mnemonic::Assert(m) => {
                if m.condition == old {
                    m.condition = new;
                }
            }
            Mnemonic::Extract(m) => {
                if m.agg == old {
                    m.agg = new;
                }
            }
            Mnemonic::Gep(m) => {
                if m.base == old {
                    m.base = new;
                }
            }
            Mnemonic::Map(m) => {
                // `body` is a function symbol, not a value operand — left intact.
                if m.src == old {
                    m.src = new;
                }
                m.captures.iter_mut().for_each(|a| {
                    if *a == old {
                        *a = new;
                    }
                });
            }
            Mnemonic::Scan(m) => {
                // `body` is a function symbol, not a value operand — left intact.
                if m.init == old {
                    m.init = new;
                }
                if m.src == old {
                    m.src = new;
                }
                m.captures.iter_mut().for_each(|a| {
                    if *a == old {
                        *a = new;
                    }
                });
            }
        }
    }
}
