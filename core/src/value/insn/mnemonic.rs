use crate::LangRef;
use crate::value::{
    LocalValueId,
    function::FunctionId,
    insn::{
        Apply, Assert, BadInsn, Binary, Branch, BranchInd, CBranch, Call, CallInd, Carry, Extract,
        FloatToFloat, FloatToInt, Gep, IntToFloat, IntrinsicApp, IsFloatNaN, Load, LzCount, Map,
        PCodeOp, PopCount, Range, Return, ReturnValue, SBorrow, SCarry, Scan, Sext, Store, Switch,
        TailCall, Tuple, Unary, Zext,
    },
};
use smallvec::SmallVec;

/// Operand list returned by [`Mnemonic::args`]. Inline-stores up to two
/// operands (covering every fixed-arity instruction — binops, casts, loads,
/// flags, …), so the pervasive per-instruction operand walks in the analysis
/// passes don't heap-allocate. Variable-arity ops (calls, tuples, `scan`) spill
/// to the heap only when they exceed two operands.
pub type Args = SmallVec<[LocalValueId; 2]>;

/// Implemented by each concrete instruction type.
///
/// Provides the common interface that [`Mnemonic`] dispatches to: a short
/// opcode string and terminator status. Operands are enumerated by
/// [`Mnemonic::for_each_operand`], in one place for every kind.
pub trait MnemonicKind {
    /// Short textual opcode, e.g. `"load"`, `"int_add"`, `"branch"`.
    fn opcode(&self) -> &'static str;

    /// Returns `true` if this instruction ends a basic block.
    ///
    /// Terminators are: [`Branch`], [`CBranch`], [`BranchInd`], [`Switch`], [`Call`],
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
/// | [`Branch`], [`CBranch`], [`BranchInd`], [`Switch`], [`Call`], [`CallInd`], [`Return`], [`ReturnValue`], [`BadInsn`] | Control flow (terminators) |
/// | [`Unop`](Mnemonic::Unop) | Unary integer/float/bool operations |
/// | [`Binop`](Mnemonic::Binop) | Binary integer/float/bool operations |
/// | [`Zext`], [`Sext`], [`Range`], [`IntToFloat`], [`FloatToInt`], [`FloatToFloat`] | Type casts and bit extraction |
/// | [`IsFloatNaN`], [`PopCount`], [`LzCount`], [`Carry`], [`SCarry`], [`SBorrow`] | Bit/flag operations |
/// | [`PCodeOp`] | User-defined or architecture-specific operation |
/// | [`Intrinsic`](crate::value::insn::intrinsic::Intrinsic) | Pure named intrinsic function (e.g. `rol`, `ror`) |
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, LangRef)]
pub enum Mnemonic {
    /// Load a value from a memory space.
    ///
    /// Reads `N` bytes at address `ptr` in the named space (`ram`, a register
    /// space, or a `$tempK` scratch space) and produces them as an `N`-byte
    /// value, little-endian. The result type is the sized integer `iN` unless
    /// a later pass retypes it; the address type is the space's address width.
    ///
    /// Loads observe every earlier `store` to an aliasing address in program
    /// order; they are the only instruction other than `call` that reads
    /// memory.
    #[langref(
        category = "Memory",
        syntax = "T %r = load(space:N, ptr)",
        example = "i32 %v = load(ram:4, i64 @p);"
    )]
    Load(Load),
    /// Store a value to a memory space.
    ///
    /// Writes the `N`-byte value `v` at address `ptr` in the named space,
    /// little-endian. `N` must equal `v`'s byte width. A store has no result.
    #[langref(
        category = "Memory",
        syntax = "store(space:N, ptr <- v)",
        example = "store(ram:4, i64 @p <- i32 @x);"
    )]
    Store(Store),
    /// Unconditional direct branch to a static target block.
    ///
    /// Transfers control to `bb`, binding each of `bb`'s block parameters to
    /// the argument named for it. Every parameter of the target must be
    /// supplied; the argument's type must match the parameter's. See the
    /// *block arguments* concept page for how this replaces φ-nodes.
    #[langref(
        category = "Control flow",
        syntax = "goto <bb @param=v …>",
        example = "goto <next>;",
        example = "goto <loop @i=i32 @x @acc=i32 @y>;"
    )]
    Branch(Branch),
    /// Conditional branch: taken when the condition operand is non-zero.
    ///
    /// `c` is a `bool` (or any 1-byte value). Control goes to the first target
    /// when `c` is non-zero and to the second otherwise. Each target carries
    /// its own argument list, so the two successors may bind different values
    /// to their parameters.
    #[langref(
        category = "Control flow",
        syntax = "if c goto <bb1 @param=v …> else goto <bb2 @param=v …>",
        example = "bool %lt = i32 @x < i32 @y;\nif bool %lt goto <next> else goto <other>;",
        example = "bool %lt = i32 @x < i32 @y;\nif bool %lt goto <loop @i=i32 @x @acc=i32 @y> else goto <loop @i=i32 @y @acc=i32 @x>;"
    )]
    CBranch(CBranch),
    /// Unconditional indirect branch to a dynamically-computed address.
    ///
    /// Jumps to the code at address `ptr`. The successors are not encoded in
    /// the instruction; a CFG analysis may attach the edges it resolves as an
    /// `// -> <bb>, …` hint after the statement. A resolved jump table is
    /// rewritten to a [`Switch`](Self::Switch).
    #[langref(
        category = "Control flow",
        syntax = "goto [ptr]",
        example = "goto [i64 @p];"
    )]
    BranchInd(BranchInd),
    /// Multi-way dispatch on an integer scrutinee — a resolved jump table.
    ///
    /// Compares `v` against each case constant and transfers control to the
    /// matching arm's block, binding that block's parameters from the arm's
    /// argument list. `default` receives every unlisted value; it may be
    /// omitted when a preceding bounds check makes the listed cases total.
    /// Case constants are pairwise distinct.
    #[langref(
        category = "Control flow",
        syntax = "switch v { K => <bb @param=v …>, …, default => <bb …> }",
        example = "switch i32 @x { 0x0 => <next>, 0x1 => <other>, default => <next> };"
    )]
    Switch(Switch),
    /// Direct call to a known function.
    ///
    /// Transfers control to function `f`'s entry, passing one argument per
    /// callee parameter by name, and resumes at the fall-through block on
    /// return. A call is a terminator: the return point is the block that
    /// follows (or an explicit `// -> <bb>` hint). Freshly lifted calls have
    /// no arguments; interface inference fills them in. The callee's clobbered
    /// registers and any escaping pointer arguments are treated as written by
    /// the call.
    #[langref(
        category = "Control flow",
        syntax = "call fn f(@param=v, …)",
        example = "call fn callee(@a=i32 @x, @b=i32 @y);"
    )]
    Call(Call),
    /// Tail call: a function-level transfer of control to another function's
    /// entry (thunk / tail jump).
    ///
    /// Control leaves the current function for `f` and never returns to it;
    /// `f`'s return is this function's return. Arguments are positional, one
    /// per callee input. Unlike [`Branch`](Self::Branch), whose target is a
    /// block of the same function, the target here is a function: the IR
    /// never names a block of another function. See [`TailCall`].
    #[langref(
        category = "Control flow",
        syntax = "tailcall fn f(v, …)",
        example = "tailcall fn callee(i32 @x, i32 @y);"
    )]
    TailCall(TailCall),
    /// Value-level application of a pure lambda function.
    ///
    /// Evaluates the lambda `f` on the positional arguments and yields its
    /// [`return`](Self::ReturnValue) value as an ordinary SSA result. `apply`
    /// is not a terminator and has no effect on memory or registers: a lambda
    /// is a pure function of its inputs.
    #[langref(
        category = "Control flow",
        syntax = "T %r = apply f(v, …)",
        example = "i32 %r = apply inc(i32 @x);"
    )]
    Apply(Apply),
    /// Indirect call through a computed function pointer.
    ///
    /// Calls the code at address `ptr`, then resumes at the fall-through
    /// block. Positional arguments, when present, are the inferred inputs of
    /// the callee. Successor edges may be attached as an `// -> <bb>` hint.
    #[langref(
        category = "Control flow",
        syntax = "call [ptr](v, …)",
        example = "call [i64 @p];",
        example = "call [i64 @p](i32 @x, i32 @y);"
    )]
    CallInd(CallInd),
    /// Return from the current function.
    ///
    /// Transfers control to the return address `ptr` (the value popped from
    /// the stack or read from the link register). With a value, `return v at
    /// ptr` additionally makes `v` the function's SSA return value once the
    /// call interface is known.
    #[langref(
        category = "Control flow",
        syntax = "return at ptr",
        syntax = "return v at ptr",
        example = "return at i64 @p;",
        example = "return i32 @x at i64 @p;"
    )]
    Return(Return),
    /// Value return from a lambda function.
    ///
    /// Ends a lambda and yields `v` to the [`apply`](Self::Apply) (or `map` /
    /// `scanl`) that invoked it. Lambdas have no return address.
    #[langref(
        category = "Control flow",
        syntax = "return v",
        example = "return i32 @x;"
    )]
    ReturnValue(ReturnValue),
    /// Bytes that do not decode to a valid instruction. A terminator with no
    /// successors (see [`BadInsn`]).
    ///
    /// The analogue of LLVM's `unreachable`: it records that lifting could not
    /// continue past this point. Executing it is an error.
    #[langref(category = "Control flow", syntax = "badinsn", example = "badinsn;")]
    BadInsn(BadInsn),
    /// A unary integer, float, or boolean operation.
    ///
    /// The result has the operand's type. The operators are listed under
    /// *Unary operators*.
    #[langref(
        category = "Arithmetic",
        syntax = "T %r = <op> v",
        syntax = "T %r = <op>(v)",
        example = "i32 %neg = - i32 @x;",
        example = "i64 %a = abs(i64 %f);"
    )]
    Unop(Unary),
    /// A binary integer, float, or boolean operation.
    ///
    /// Both operands have the same width; a literal takes the width of the
    /// other operand. Arithmetic results have the operands' type;
    /// comparisons produce `bool`. The operators are listed under *Integer
    /// operators* and *Float operators*.
    #[langref(
        category = "Arithmetic",
        syntax = "T %r = a <op> b",
        example = "i32 %r = i32 @x + i32 @y;",
        example = "bool %lt = i32 @x s< i32 @y;"
    )]
    Binop(Binary),
    /// Extract a contiguous byte range from a value.
    ///
    /// `v[start:end]` is bytes `[start, end)` of `v` (little-endian, so
    /// `v[0:1]` is the least significant byte), as an `end - start` byte
    /// integer. `start` defaults to `0` and `end` to `v`'s width.
    #[langref(
        category = "Casts",
        syntax = "iN %r = v[start:end]",
        example = "i16 %lo = i32 @x[0:2];",
        example = "i8 %hi = i32 @x[3:4];"
    )]
    Range(Range),
    /// Convert an integer to a floating-point value.
    ///
    /// Interprets `v` as a signed integer and rounds it to the nearest `fN`
    /// (round to nearest even).
    #[langref(
        category = "Casts",
        syntax = "fN %r = int2float(fN, v)",
        example = "i64 %d = int2float(f64, i32 @x);"
    )]
    IntToFloat(IntToFloat),
    /// Convert a floating-point value to a different float width.
    ///
    /// Widening is exact; narrowing rounds to nearest even and may overflow to
    /// an infinity. NaN converts to NaN.
    #[langref(
        category = "Casts",
        syntax = "fN %r = float2float(fN, v)",
        example = "i32 %s = float2float(f32, i64 %f);"
    )]
    FloatToFloat(FloatToFloat),
    /// Convert a floating-point value to an integer (truncate toward zero).
    ///
    /// The result is the signed integer nearest to zero, `iN` wide. Values
    /// outside `iN`'s range and NaN produce an unspecified value.
    #[langref(
        category = "Casts",
        syntax = "iN %r = trunc(iN, v)",
        example = "i32 %int = trunc(i32, i64 %f);"
    )]
    FloatToInt(FloatToInt),
    /// Zero-extend a value to a wider integer.
    ///
    /// The upper `N - width(v)` bytes of the result are zero. `iN` must be at
    /// least as wide as `v`.
    #[langref(
        category = "Casts",
        syntax = "iN %r = zext(iN, v)",
        example = "i64 %w = zext(i64, i32 @x);"
    )]
    Zext(Zext),
    /// Sign-extend a value to a wider integer.
    ///
    /// The upper bytes of the result are copies of `v`'s sign bit. `iN` must
    /// be at least as wide as `v`.
    #[langref(
        category = "Casts",
        syntax = "iN %r = sext(iN, v)",
        example = "i64 %w = sext(i64, i32 @x);"
    )]
    Sext(Sext),
    /// Test whether a floating-point value is NaN.
    ///
    /// The result is the byte `1` when `v` is a NaN and `0` otherwise.
    #[langref(
        category = "Bit and flag operations",
        syntax = "i8 %r = nan(v)",
        example = "i8 %isnan = nan(i64 %f);"
    )]
    IsFloatNaN(IsFloatNaN),
    /// Count the number of set bits (population count / Hamming weight).
    ///
    /// The result width is chosen by the producer (the p-code output size);
    /// the text form makes it one byte.
    #[langref(
        category = "Bit and flag operations",
        syntax = "iN %r = popcount(v)",
        example = "i8 %bits = popcount(i32 @x);"
    )]
    PopCount(PopCount),
    /// Count leading zero bits.
    ///
    /// The number of zero bits above the most significant set bit of `v`;
    /// `width(v) * 8` when `v` is zero. The result width is chosen by the
    /// producer (the p-code output size); the text form makes it one byte.
    #[langref(
        category = "Bit and flag operations",
        syntax = "iN %r = lzcount(v)",
        example = "i8 %lz = lzcount(i32 @x);"
    )]
    LzCount(LzCount),
    /// Unsigned addition carry-out flag.
    ///
    /// The byte `1` when `a + b` does not fit in the operands' width, i.e.
    /// the unsigned addition carries out of the top bit, else `0`. This is
    /// the x86 `CF` after `add`.
    #[langref(
        category = "Bit and flag operations",
        syntax = "i8 %r = carry(a, b)",
        example = "i8 %cf = carry(i32 @x, i32 @y);"
    )]
    Carry(Carry),
    /// Signed addition carry-out (overflow) flag.
    ///
    /// The byte `1` when the two's-complement sum `a + b` overflows (both
    /// operands have the same sign and the result's sign differs), else `0`.
    /// This is the x86 `OF` after `add`.
    #[langref(
        category = "Bit and flag operations",
        syntax = "i8 %r = scarry(a, b)",
        example = "i8 %of = scarry(i32 @x, i32 @y);"
    )]
    SCarry(SCarry),
    /// Signed subtraction borrow flag.
    ///
    /// The byte `1` when the two's-complement difference `a - b` overflows,
    /// else `0`. This is the x86 `OF` after `sub` or `cmp`.
    #[langref(
        category = "Bit and flag operations",
        syntax = "i8 %r = sborrow(a, b)",
        example = "i8 %of = sborrow(i32 @x, i32 @y);"
    )]
    SBorrow(SBorrow),
    /// Assert when resolving an execution trace
    ///
    /// Declares that `c` holds on every execution reaching this point. It has
    /// no effect on the machine state; analyses may assume it, and the
    /// emulator checks it.
    #[langref(
        category = "Verification",
        syntax = "assert c",
        example = "bool %lt = i32 @x < i32 @y;\nassert bool %lt;"
    )]
    Assert(Assert),
    /// A user-defined or architecture-specific p-code operation.
    ///
    /// An opaque operation declared by the SLEIGH specification (a
    /// `define pcodeop`), such as a CPUID query or a system call. Its
    /// arguments and optional result are values; its semantics are whatever
    /// the environment provides. `vm.interrupt` is the reserved op that hands
    /// control to the host. There is no textual form for user ops.
    #[langref(
        category = "Extensions",
        syntax = "T %r = opname(v, …)",
        syntax = "opname(v, …)"
    )]
    PCodeOp(PCodeOp),
    /// A pure named intrinsic function (e.g. `rol`, `ror`). Categorically pure:
    /// no memory or observable side effects.
    ///
    /// Applies a registered intrinsic to its operands. Intrinsics either name
    /// an idiom recognized from machine code (`$rol`) or a sequence operation
    /// that has no p-code counterpart (`$iota`, `$at`). Each is listed under
    /// *Intrinsics*.
    #[langref(
        category = "Extensions",
        syntax = "T %r = $name(v, …)",
        example = "i32 %r = $rol(i32 @x, i32 0x5);"
    )]
    Intrinsic(IntrinsicApp),
    /// Build an aggregate (tuple) value from ordered fields.
    ///
    /// The result is the aggregate of the named fields, laid out in order.
    /// Fields are read back with [`extract`](Self::Extract). Tuples let one
    /// instruction produce several values (e.g. a call's return value together
    /// with its register write-set) without a multi-result instruction.
    #[langref(
        category = "Aggregates",
        syntax = "T %r = pack(field=v, …)",
        example = "i64 %t = pack(lo=i32 @x, hi=i32 @y);"
    )]
    Tuple(Tuple),
    /// Project a single field out of an aggregate value.
    ///
    /// The result is the named field, with that field's type.
    #[langref(
        category = "Aggregates",
        syntax = "T %r = extract(agg.field)",
        example = "i64 %t = pack(lo=i32 @x, hi=i32 @y);\ni32 %lo = extract(%t.lo);"
    )]
    Extract(Extract),
    /// Compute the address of a struct field (typed, named pointer arithmetic).
    ///
    /// `gep(base.field)` is `base + offset(field)`, typed as a pointer to the
    /// field. It performs no memory access: the field's value is a separate
    /// `load` of the result. `base` must be a pointer to a struct declared
    /// with `type name { field: size, … }`; a value is given that type by
    /// declaring it as `name* %v = …`.
    #[langref(
        category = "Aggregates",
        syntax = "T* %r = gep(base.field)",
        example = "point* %pt = i64 @p + i64 0x0;\n%py = gep(%pt.y);"
    )]
    Gep(Gep),
    /// Total element-wise map over an array value (a projectable loop).
    ///
    /// `body <$> src` is the array `out[i] = body(src[i], captures…)`, the
    /// same length as `src`. `body` is a unary lambda applied to each element;
    /// captures are loop-invariant values it also receives. The element type
    /// of the result is the body's return type.
    #[langref(
        category = "Sequences",
        syntax = "T %r = body <$> src",
        syntax = "T %r = (body capture …) <$> src",
        example = "%src = $iota(i64 0x8);\n%out = inc64 <$> %src;"
    )]
    Map(Map),
    /// Total left-scan (prefix fold) over an array value: a projectable loop
    /// whose per-element write depends on the previous iteration's result.
    ///
    /// `scanl @body init src` is the array `out[i] = acc(i+1)` where
    /// `acc(0) = init` and `acc(i+1) = body(acc(i), src[i], captures…)`.
    /// `body` is a binary lambda `(accumulator, element)`.
    #[langref(
        category = "Sequences",
        syntax = "T %r = scanl @body init src",
        syntax = "T %r = scanl (@body capture …) init src",
        example = "%src = $iota(i64 0x8);\n%out = scanl @step i64 @n %src;"
    )]
    Scan(Scan),
}

/// The canonical operand order, written once: `$f` is called on each operand
/// slot, as `&LocalValueId` by default and as `&mut LocalValueId` with a
/// trailing `mut`.
///
/// A `Callee`, a block target and a memory space are not operands — they
/// are not values — and neither is a p-code op's destination slot.
macro_rules! visit_operands {
    ($mnemonic:expr, $f:ident $(, $m:tt)?) => {
        match $mnemonic {
            Mnemonic::Load(m) => $f(& $($m)? m.ptr),
            Mnemonic::Store(m) => {
                $f(& $($m)? m.ptr);
                $f(& $($m)? m.src);
            }
            Mnemonic::Branch(m) => {
                for a in & $($m)? m.args {
                    $f(a);
                }
            }
            Mnemonic::CBranch(m) => {
                $f(& $($m)? m.condition);
                for a in & $($m)? m.success_args {
                    $f(a);
                }
                for a in & $($m)? m.failure_args {
                    $f(a);
                }
            }
            Mnemonic::BranchInd(m) => $f(& $($m)? m.ptr),
            Mnemonic::Switch(m) => {
                $f(& $($m)? m.scrutinee);
                for case in & $($m)? m.cases {
                    for a in & $($m)? case.args {
                        $f(a);
                    }
                }
                for a in & $($m)? m.default_args {
                    $f(a);
                }
            }
            Mnemonic::TailCall(m) => {
                for a in & $($m)? m.args {
                    $f(a);
                }
            }
            Mnemonic::Apply(m) => {
                for a in & $($m)? m.args {
                    $f(a);
                }
            }
            Mnemonic::Call(m) => {
                for a in & $($m)? m.args {
                    $f(a);
                }
            }
            Mnemonic::CallInd(m) => {
                $f(& $($m)? m.ptr);
                for a in & $($m)? m.args {
                    $f(a);
                }
            }
            Mnemonic::Return(m) => {
                $f(& $($m)? m.ptr);
                if let Some(v) = & $($m)? m.value {
                    $f(v);
                }
            }
            Mnemonic::ReturnValue(m) => $f(& $($m)? m.value),
            Mnemonic::BadInsn(_) => {}
            Mnemonic::Unop(m) => $f(& $($m)? m.src),
            Mnemonic::Binop(m) => {
                $f(& $($m)? m.lhs);
                $f(& $($m)? m.rhs);
            }
            Mnemonic::Range(m) => $f(& $($m)? m.src),
            Mnemonic::Zext(m) => $f(& $($m)? m.src),
            Mnemonic::Sext(m) => $f(& $($m)? m.src),
            Mnemonic::IntToFloat(m) => $f(& $($m)? m.src),
            Mnemonic::FloatToFloat(m) => $f(& $($m)? m.src),
            Mnemonic::FloatToInt(m) => $f(& $($m)? m.src),
            Mnemonic::IsFloatNaN(m) => $f(& $($m)? m.src),
            Mnemonic::PopCount(m) => $f(& $($m)? m.src),
            Mnemonic::LzCount(m) => $f(& $($m)? m.src),
            Mnemonic::Carry(m) => {
                $f(& $($m)? m.lhs);
                $f(& $($m)? m.rhs);
            }
            Mnemonic::SCarry(m) => {
                $f(& $($m)? m.lhs);
                $f(& $($m)? m.rhs);
            }
            Mnemonic::SBorrow(m) => {
                $f(& $($m)? m.lhs);
                $f(& $($m)? m.rhs);
            }
            Mnemonic::PCodeOp(m) => {
                for a in & $($m)? m.args {
                    $f(a);
                }
            }
            Mnemonic::Intrinsic(m) => {
                for a in & $($m)? m.args {
                    $f(a);
                }
            }
            Mnemonic::Tuple(m) => {
                for a in & $($m)? m.fields {
                    $f(a);
                }
            }
            Mnemonic::Assert(m) => $f(& $($m)? m.condition),
            Mnemonic::Extract(m) => $f(& $($m)? m.agg),
            Mnemonic::Gep(m) => $f(& $($m)? m.base),
            Mnemonic::Map(m) => {
                $f(& $($m)? m.src);
                for a in & $($m)? m.captures {
                    $f(a);
                }
            }
            Mnemonic::Scan(m) => {
                $f(& $($m)? m.init);
                $f(& $($m)? m.src);
                for a in & $($m)? m.captures {
                    $f(a);
                }
            }
        }
    };
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
            Mnemonic::Switch(m) => m,
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
            // Every arm plus the default, if it has one: a resolved dispatch
            // knows all of its successors statically.
            Mnemonic::Switch(s) => s
                .cases
                .iter()
                .map(|case| case.target)
                .chain(s.default)
                .collect(),
            _ => smallvec::SmallVec::new(),
        }
    }

    /// The operands of this mnemonic, in canonical order.
    ///
    /// Every use of an operand's position — recording a use edge when an
    /// instruction is created, rewriting one operand of it, checking the edges
    /// against the operands — goes through [`for_each_operand`](Self::for_each_operand)
    /// or its mutable twin, so an operand's index means the same thing
    /// everywhere. This builds the list those walk.
    pub fn args(&self) -> Args {
        let mut args = Args::new();
        self.for_each_operand(|operand| args.push(operand));
        args
    }

    /// Calls `f` on each operand, in canonical order. Allocation-free; the
    /// operand's position in this order is its *operand index*.
    pub fn for_each_operand(&self, mut f: impl FnMut(LocalValueId)) {
        let mut visit = |operand: &LocalValueId| f(*operand);
        visit_operands!(self, visit);
    }

    /// Calls `f` on each operand slot, in the same order as
    /// [`for_each_operand`](Self::for_each_operand).
    ///
    /// Crate-private: an installed instruction's operands are mirrored by its
    /// body's use edges, which only the body's use-maintaining verbs may
    /// desynchronize and repair. A detached mnemonic is free to change.
    pub(crate) fn for_each_operand_mut(&mut self, mut f: impl FnMut(&mut LocalValueId)) {
        visit_operands!(self, f, mut);
    }

    /// This mnemonic with every operand replaced by `f` of it, in canonical
    /// order. For a mnemonic being rebuilt in another body — a cached
    /// instruction replayed with its operands renumbered — which is why it
    /// takes the mnemonic by value: an installed instruction's operands are
    /// mirrored by use edges, and only its body may change those.
    pub fn map_operands(mut self, mut f: impl FnMut(LocalValueId) -> LocalValueId) -> Self {
        self.for_each_operand_mut(|operand| *operand = f(*operand));
        self
    }

    /// How many operands this mnemonic has.
    pub fn operand_count(&self) -> usize {
        let mut count = 0;
        self.for_each_operand(|_| count += 1);
        count
    }

    /// The operand at `index` in canonical order, if there is one.
    pub fn operand(&self, index: usize) -> Option<LocalValueId> {
        let mut at = 0;
        let mut found = None;
        self.for_each_operand(|operand| {
            if at == index {
                found = Some(operand);
            }
            at += 1;
        });
        found
    }

    /// Sets the operand at `index` in canonical order. Panics if there is no
    /// such operand.
    pub(crate) fn set_operand(&mut self, index: usize, value: LocalValueId) {
        let mut at = 0;
        let mut found = false;
        self.for_each_operand_mut(|operand| {
            if at == index {
                *operand = value;
                found = true;
            }
            at += 1;
        });
        assert!(found, "{} has no operand {index}", self.opcode());
    }

    /// Replace every occurrence of `old` with `new` in this instruction's
    /// operands (and a p-code op's destination slot, which is a result rather
    /// than an operand but is a value reference all the same).
    ///
    /// For a mnemonic that is not in a body. Once installed, rewrite operands
    /// through the body, which keeps the use edges in step
    /// ([`FunctionBody::replace_uses_where`](crate::value::FunctionBody::replace_uses_where)).
    pub fn replace_value(&mut self, old: LocalValueId, new: LocalValueId) {
        self.for_each_operand_mut(|operand| {
            if *operand == old {
                *operand = new;
            }
        });
        if let Mnemonic::PCodeOp(m) = self
            && let Some(dst) = m.dst.as_mut()
            && *dst == old
        {
            *dst = new;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{
        LiteralId, LocalBlockId,
        insn::{LocalInsnId, SwitchArm},
    };

    fn lit(n: usize) -> LocalValueId {
        LocalValueId::Literal(LiteralId::from(n))
    }

    fn insn(n: usize) -> LocalValueId {
        LocalValueId::Instruction(LocalInsnId::from(n))
    }

    /// The variable-arity and optional-operand shapes, where the two walks
    /// could most plausibly drift apart.
    fn samples() -> Vec<Mnemonic> {
        let block = LocalBlockId::from(0);
        vec![
            Mnemonic::Store(Store {
                space: crate::space::LocalMemorySpaceId::Temp(
                    crate::value::LocalTempSpaceId::from(0),
                ),
                ptr: lit(0),
                src: insn(1),
                size: 8,
            }),
            Mnemonic::CBranch(CBranch {
                condition: insn(0),
                success_block: block,
                success_args: vec![lit(1), insn(2)],
                failure_block: block,
                failure_args: vec![insn(3)],
            }),
            Mnemonic::Switch(Switch {
                scrutinee: insn(0),
                cases: vec![
                    SwitchArm {
                        value: 1,
                        target: block,
                        args: vec![lit(1)],
                    },
                    SwitchArm {
                        value: 2,
                        target: block,
                        args: vec![],
                    },
                    SwitchArm {
                        value: 3,
                        target: block,
                        args: vec![insn(2), insn(3)],
                    },
                ],
                default: Some(block),
                default_args: vec![lit(4)],
            }),
            Mnemonic::CallInd(CallInd {
                ptr: insn(0),
                args: vec![insn(1), insn(1), lit(2)],
            }),
            Mnemonic::Return(Return {
                ptr: insn(0),
                value: None,
            }),
            Mnemonic::Return(Return {
                ptr: insn(0),
                value: Some(insn(1)),
            }),
            Mnemonic::Scan(Scan {
                body: crate::value::insn::Callee::Minted(0),
                init: lit(0),
                src: insn(1),
                captures: vec![insn(2), lit(3)],
            }),
            Mnemonic::Map(Map {
                body: crate::value::insn::Callee::Minted(0),
                src: insn(0),
                captures: vec![],
            }),
            Mnemonic::BadInsn(BadInsn),
        ]
    }

    #[test]
    fn both_walks_visit_the_same_operands_in_the_same_order() {
        for mut mnemonic in samples() {
            let read: Vec<_> = mnemonic.args().to_vec();
            let mut written = Vec::new();
            mnemonic.for_each_operand_mut(|slot| written.push(*slot));
            assert_eq!(read, written, "{}", mnemonic.opcode());
            assert_eq!(mnemonic.operand_count(), read.len());
            for (index, &operand) in read.iter().enumerate() {
                assert_eq!(mnemonic.operand(index), Some(operand));
            }
            assert_eq!(mnemonic.operand(read.len()), None);
        }
    }

    #[test]
    fn setting_an_operand_changes_that_position_only() {
        let fresh = insn(99);
        for mut mnemonic in samples() {
            let before = mnemonic.args().to_vec();
            for index in 0..before.len() {
                let mut expected = before.clone();
                expected[index] = fresh;
                let mut edited = mnemonic.clone();
                edited.set_operand(index, fresh);
                assert_eq!(edited.args().to_vec(), expected, "{}", mnemonic.opcode());
            }
            // A repeated operand is replaced at every occurrence.
            if let Some(&first) = before.first() {
                mnemonic.replace_value(first, fresh);
                let expected: Vec<_> = before
                    .iter()
                    .map(|&v| if v == first { fresh } else { v })
                    .collect();
                assert_eq!(mnemonic.args().to_vec(), expected);
            }
        }
    }

    #[test]
    #[should_panic(expected = "has no operand 2")]
    fn setting_a_missing_operand_panics() {
        let mut mnemonic = Mnemonic::Return(Return {
            ptr: insn(0),
            value: Some(insn(1)),
        });
        mnemonic.set_operand(2, insn(3));
    }
}
