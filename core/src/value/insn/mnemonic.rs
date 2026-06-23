use crate::{
    context::Context,
    value::{
        ValueId,
        function::FunctionId,
        insn::{
            Binary, Branch, BranchInd, CBranch, Call, CallInd, Carry, FloatToFloat, FloatToInt,
            IntToFloat, IsFloatNaN, Load, LzCount, PCodeOp, PopCount, Range, Return, SBorrow,
            SCarry, Sext, Store, Unary, Zext,
        },
    },
};
use std::fmt::Formatter;

/// Implemented by each concrete instruction type.
///
/// Provides the common interface that [`Mnemonic`] dispatches to: a short
/// opcode string, argument enumeration, terminator status, and a context-aware
/// display implementation.
pub trait MnemonicKind {
    /// Short textual opcode, e.g. `"load"`, `"int_add"`, `"branch"`.
    fn opcode(&self) -> &'static str;

    /// Formats the full instruction (opcode + operands) into `f`.
    ///
    /// Operands are displayed using their context-aware `Display` impls so that
    /// names, addresses, and symbolic labels are shown when available.
    fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result;

    // TODO: replace with a visitor pattern to avoid the need for this method
    /// Returns the [`ValueId`]s of all operands consumed by this instruction.
    fn args(&self) -> Vec<ValueId>;

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
/// | [`Branch`], [`CBranch`], [`BranchInd`], [`Call`], [`CallInd`], [`Return`] | Control flow (terminators) |
/// | [`Unop`](Mnemonic::Unop) | Unary integer/float/bool operations |
/// | [`Binop`](Mnemonic::Binop) | Binary integer/float/bool operations |
/// | [`Zext`], [`Sext`], [`Range`], [`IntToFloat`], [`FloatToInt`], [`FloatToFloat`] | Type casts and bit extraction |
/// | [`IsFloatNaN`], [`PopCount`], [`LzCount`], [`Carry`], [`SCarry`], [`SBorrow`] | Bit/flag operations |
/// | [`PCodeOp`] | User-defined or architecture-specific operation |
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
    /// Indirect call through a computed function pointer.
    CallInd(CallInd),
    /// Return from the current function.
    Return(Return),
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
    /// A user-defined or architecture-specific p-code operation.
    PCodeOp(PCodeOp),
}

impl Mnemonic {
    fn as_kind(&self) -> &dyn MnemonicKind {
        match self {
            Mnemonic::Load(m) => m,
            Mnemonic::Store(m) => m,
            Mnemonic::Branch(m) => m,
            Mnemonic::CBranch(m) => m,
            Mnemonic::BranchInd(m) => m,
            Mnemonic::Call(m) => m,
            Mnemonic::CallInd(m) => m,
            Mnemonic::Return(m) => m,
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
            Mnemonic::PCodeOp(m) => m,
        }
    }

    pub fn opcode(&self) -> &'static str {
        self.as_kind().opcode()
    }

    pub fn is_terminator(&self) -> bool {
        self.as_kind().is_terminator()
    }

    /// The callee of a direct [`Call`], or `None` for any other mnemonic
    /// (including indirect [`CallInd`] calls, whose target is not statically
    /// known). Used to maintain the reverse call graph.
    pub fn call_target(&self) -> Option<FunctionId> {
        match self {
            Mnemonic::Call(call) => Some(call.target),
            _ => None,
        }
    }

    pub fn fmt(&self, f: &mut Formatter<'_>, ctx: &Context<'_>) -> std::fmt::Result {
        self.as_kind().fmt(f, ctx)
    }

    pub fn args(&self) -> Vec<ValueId> {
        self.as_kind().args()
    }

    /// Replace every occurrence of `old` with `new` in this instruction's operands.
    pub fn replace_value(&mut self, old: ValueId, new: ValueId) {
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
        }
    }
}
