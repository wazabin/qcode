//! Exact IEEE/x87 80-bit floating-point operations for the concrete emulator.
//!
//! The emulator stores scalar values as raw little-endian bits.  Keep all
//! extended-precision interpretation here so the integer/register-memory
//! implementation remains byte-oriented.

use rustc_apfloat::{
    Float, FloatConvert, Round, Status,
    ieee::{Double, Single, X87DoubleExtended},
};

#[derive(Clone, Copy, Debug)]
pub(super) struct Result {
    pub bits: u128,
    pub status: Status,
}

fn result(value: rustc_apfloat::StatusAnd<X87DoubleExtended>) -> Result {
    Result {
        bits: bits(value.value),
        status: value.status,
    }
}

pub(super) fn round_from_control(control: u16) -> Round {
    match (control >> 10) & 3 {
        0 => Round::NearestTiesToEven,
        1 => Round::TowardNegative,
        2 => Round::TowardPositive,
        _ => Round::TowardZero,
    }
}

/// Apply x87's precision-control field after an extended operation.  Unlike
/// conversion through IEEE single/double, this rounds the f80 significand in
/// place and therefore retains x87's much wider exponent range.
fn apply_precision(value: X87DoubleExtended, control: u16, round: Round) -> Result {
    let precision = match (control >> 8) & 3 {
        0 => 24,
        2 => 53,
        // 11 is extended; 01 is reserved and treated as extended by this
        // non-trapping interpreter.
        _ => {
            return Result {
                bits: bits(value),
                status: Status::OK,
            };
        }
    };
    let raw = bits(value);
    let sign_exponent = (raw >> 64) as u16;
    let negative = sign_exponent & 0x8000 != 0;
    let exponent = sign_exponent & 0x7fff;
    // Precision control rounds normal finite extended values.  APFloat has
    // already handled exceptional and denormal results before this stage.
    if exponent == 0x7fff {
        return Result {
            bits: raw,
            status: Status::OK,
        };
    }
    // A denormal result is rounded like any other significand rather than
    // kept: an extended denormal sits far below what 24 or 53 bits of
    // precision can hold, so it normally rounds away to zero. Hardware
    // reports underflow alongside the inexact result either way.
    let denormal = exponent == 0;
    if denormal && raw as u64 == 0 {
        return Result {
            bits: raw,
            status: Status::OK,
        };
    }
    let significand = raw as u64;
    let discarded_bits = 64 - precision;
    let discarded_mask = (1u64 << discarded_bits) - 1;
    let discarded = significand & discarded_mask;
    if discarded == 0 {
        let status = if denormal {
            Status::UNDERFLOW | Status::INEXACT
        } else {
            Status::OK
        };
        return Result { bits: raw, status };
    }
    let retained = significand >> discarded_bits;
    let increment = match round {
        Round::NearestTiesToEven => {
            let halfway = 1u64 << (discarded_bits - 1);
            discarded > halfway || (discarded == halfway && retained & 1 != 0)
        }
        Round::TowardPositive => !negative,
        Round::TowardNegative => negative,
        Round::TowardZero => false,
        // x87 has no ties-away mode, but preserving APFloat's behavior here
        // makes this helper total if a caller uses it in the future.
        Round::NearestTiesToAway => discarded >= (1u64 << (discarded_bits - 1)),
    };
    let (retained, exponent) = if increment {
        let (rounded, carry) = retained.overflowing_add(1);
        if carry || rounded == (1u64 << precision) {
            (1u64 << (precision - 1), exponent.saturating_add(1))
        } else {
            (rounded, exponent)
        }
    } else {
        (retained, exponent)
    };
    let rounded = (u128::from(sign_exponent & 0x8000 | exponent) << 64)
        | (u128::from(retained) << discarded_bits);
    let mut status = Status::INEXACT;
    if denormal {
        status |= Status::UNDERFLOW;
    }
    if exponent == 0x7fff {
        status |= Status::OVERFLOW;
    }
    Result {
        bits: rounded,
        status,
    }
}

/// Round an 80-bit value's significand to a narrower precision while keeping
/// the extended exponent range.  This is the IEEE notion of "round to a
/// narrower significand precision within the same format", not a conversion
/// to single or double: the exponent range is unchanged, so a value that a
/// narrower format could not represent still keeps its exponent here.
///
/// `precision` counts significand bits including the explicit integer bit;
/// 64 (or more) is the identity.  Only the 80-bit format is supported, so
/// callers holding an f32 or f64 value get `None` from the p-code layer.
pub(super) fn round_to_precision(raw: u128, precision: u32, round: Round) -> Result {
    if precision >= 64 {
        return Result {
            bits: raw,
            status: Status::OK,
        };
    }
    let sign_exponent = (raw >> 64) as u16;
    let negative = sign_exponent & 0x8000 != 0;
    let exponent = sign_exponent & 0x7fff;
    // Infinities and NaNs carry no significand to narrow.
    if exponent == 0x7fff {
        return Result {
            bits: raw,
            status: Status::OK,
        };
    }
    let denormal = exponent == 0;
    if denormal && raw as u64 == 0 {
        return Result {
            bits: raw,
            status: Status::OK,
        };
    }
    let significand = raw as u64;
    let discarded_bits = 64 - precision;
    let discarded_mask = (1u64 << discarded_bits) - 1;
    let discarded = significand & discarded_mask;
    if discarded == 0 {
        // A subnormal input is already inexact with respect to the narrower
        // precision's exponent range, and rounding it reports tininess.
        let status = if denormal {
            Status::UNDERFLOW | Status::INEXACT
        } else {
            Status::OK
        };
        return Result { bits: raw, status };
    }
    let retained = significand >> discarded_bits;
    let increment = match round {
        Round::NearestTiesToEven => {
            let halfway = 1u64 << (discarded_bits - 1);
            discarded > halfway || (discarded == halfway && retained & 1 != 0)
        }
        Round::TowardPositive => !negative,
        Round::TowardNegative => negative,
        Round::TowardZero => false,
        Round::NearestTiesToAway => discarded >= (1u64 << (discarded_bits - 1)),
    };
    let (retained, exponent) = if increment {
        let (rounded, carry) = retained.overflowing_add(1);
        if carry || rounded == (1u64 << precision) {
            (1u64 << (precision - 1), exponent.saturating_add(1))
        } else {
            (rounded, exponent)
        }
    } else {
        (retained, exponent)
    };
    let rounded = (u128::from(sign_exponent & 0x8000 | exponent) << 64)
        | (u128::from(retained) << discarded_bits);
    let mut status = Status::INEXACT;
    if denormal {
        status |= Status::UNDERFLOW;
    }
    if exponent == 0x7fff {
        status |= Status::OVERFLOW;
    }
    Result {
        bits: rounded,
        status,
    }
}

/// x87's unsupported encodings: a non-zero exponent with the explicit integer
/// bit clear. An unnormal, and the pseudo-NaN and pseudo-infinity that share
/// that shape, are not values the FPU will operate on - it reports invalid and
/// delivers the indefinite rather than computing with them. A zero exponent
/// with the bit clear is an ordinary denormal and stays valid.
fn is_unsupported(raw: u128) -> bool {
    let exponent = (raw >> 64) & 0x7fff;
    let integer_bit = (raw >> 63) & 1;
    exponent != 0 && integer_bit == 0
}

/// The value a masked invalid operation delivers.
///
/// x87 propagates a NaN operand rather than the architectural indefinite: the
/// NaN is quieted in place, so its sign and payload survive. When both
/// operands are NaNs the one with the larger significand wins. The indefinite
/// is written only for an invalid operation that has no NaN operand at all -
/// 0*inf, inf-inf, 0/0, inf/inf and the like.
fn invalid_result(operands: &[u128]) -> u128 {
    const INDEFINITE: u128 = 0xffff_c000_0000_0000_0000;
    const QUIET_BIT: u128 = 1 << 62;
    let mut winner: Option<u128> = None;
    for &raw in operands {
        if !value(raw).is_nan() {
            continue;
        }
        winner = match winner {
            Some(current) if (current as u64) >= (raw as u64) => Some(current),
            _ => Some(raw),
        };
    }
    match winner {
        Some(raw) => raw | QUIET_BIT,
        None => INDEFINITE,
    }
}

fn arithmetic(
    value: rustc_apfloat::StatusAnd<X87DoubleExtended>,
    control: u16,
    round: Round,
    operands: &[u128],
) -> Result {
    let rounded = apply_precision(value.value, control, round);
    // APFloat computes with an unsupported encoding rather than rejecting it,
    // so the invalid it never raises has to be added here.
    let unsupported = operands.iter().copied().any(is_unsupported);
    let mut status = value.status | rounded.status;
    if unsupported {
        status |= Status::INVALID_OP;
    }
    // APFloat's operation-specific NaN sign/payload is not what x87 delivers.
    let bits = if unsupported {
        // An unsupported operand outranks NaN propagation: there is no
        // meaningful payload to carry, so the indefinite is delivered.
        0xffff_c000_0000_0000_0000
    } else if status.contains(Status::INVALID_OP) {
        invalid_result(operands)
    } else {
        rounded.bits
    };
    Result { bits, status }
}

fn value(bits: u128) -> X87DoubleExtended {
    X87DoubleExtended::from_bits(bits)
}

fn bits(value: X87DoubleExtended) -> u128 {
    value.to_bits()
}

pub(super) fn to_f64(raw: u128) -> f64 {
    let mut loses_info = false;
    let value: Double = value(raw)
        .convert_r(Round::NearestTiesToEven, &mut loses_info)
        .value;
    f64::from_bits(value.to_bits() as u64)
}

pub(super) fn from_f64(value: f64) -> u128 {
    let mut loses_info = false;
    let source = Double::from_bits(u128::from(value.to_bits()));
    bits(
        source
            .convert_r(Round::NearestTiesToEven, &mut loses_info)
            .value,
    )
}

pub(super) fn from_i128(value: i128) -> u128 {
    from_i128_contextual(value, 0x037f).bits
}

pub(super) fn from_i128_contextual(value: i128, control: u16) -> Result {
    let round = round_from_control(control);
    arithmetic(
        X87DoubleExtended::from_i128_r(value, round),
        control,
        round,
        &[],
    )
}

pub(super) fn from_f32_bits(raw: u32) -> u128 {
    let mut loses_info = false;
    bits(
        Single::from_bits(u128::from(raw))
            .convert_r(Round::NearestTiesToEven, &mut loses_info)
            .value,
    )
}

pub(super) fn to_i128(raw: u128, width: usize) -> i128 {
    truncate_to_i128(raw, width).value
}

/// Truncate toward zero into a `width`-bit signed integer.
///
/// x87 reaches this through SLEIGH's `round(); trunc()` pair, so rounding
/// control is applied by the preceding round and the truncation itself is
/// always toward zero. A source that is NaN, infinite, or outside the
/// destination range stores the *integer indefinite* value — the most negative
/// integer of that width — rather than APFloat's saturated bound.
pub(super) fn truncate_to_i128(raw: u128, width: usize) -> rustc_apfloat::StatusAnd<i128> {
    let mut exact = false;
    let converted = value(raw).to_i128_r(width, Round::TowardZero, &mut exact);
    if converted.status.contains(Status::INVALID_OP) && width > 0 {
        let indefinite = -(1i128 << (width - 1));
        return rustc_apfloat::StatusAnd {
            status: converted.status,
            value: indefinite,
        };
    }
    converted
}

/// The generic 80-bit arithmetic the interpreter falls back to for any
/// specification that has no explicit IEEE p-code operation.  It carries no
/// architectural policy: no precision control, no x87 payload selection, and
/// no status reporting.  x87 constructors use `float_{add,sub,mul,div}` with
/// an explicit rounding mode instead.
pub(super) fn add(lhs: u128, rhs: u128) -> u128 {
    bits(
        value(lhs)
            .add_r(value(rhs), Round::NearestTiesToEven)
            .value,
    )
}

pub(super) fn sub(lhs: u128, rhs: u128) -> u128 {
    bits(
        value(lhs)
            .sub_r(value(rhs), Round::NearestTiesToEven)
            .value,
    )
}

pub(super) fn mul(lhs: u128, rhs: u128) -> u128 {
    bits(
        value(lhs)
            .mul_r(value(rhs), Round::NearestTiesToEven)
            .value,
    )
}

pub(super) fn div(lhs: u128, rhs: u128) -> u128 {
    bits(
        value(lhs)
            .div_r(value(rhs), Round::NearestTiesToEven)
            .value,
    )
}

/// FSCALE: multiply by 2 raised to the truncated integer value of `factor`.
/// The scale is exact for a finite result, so only the destination rounding
/// applies. A NaN, infinite or zero source is returned by APFloat unchanged.
pub(super) fn scale_contextual(raw: u128, factor: u128, control: u16) -> Result {
    let round = round_from_control(control);
    let source = value(raw);
    // x87 truncates the scale operand toward zero, whatever the rounding
    // control says, and clamps far beyond the representable exponent range.
    let mut exact = false;
    let steps = value(factor)
        .to_i128_r(32, Round::TowardZero, &mut exact)
        .value
        .clamp(-0x8000, 0x7fff) as i32;
    arithmetic(
        rustc_apfloat::StatusAnd {
            status: Status::OK,
            value: source.scalbn_r(steps, round),
        },
        control,
        round,
        &[raw],
    )
}

pub(super) fn round_to_integral_contextual(raw: u128, control: u16) -> Result {
    let round = round_from_control(control);
    result(value(raw).round_to_integral(round))
}

/// The result of FPREM/FPREM1: an exact remainder plus the quotient bits the
/// condition codes report.
#[derive(Clone, Copy, Debug)]
pub(super) struct Remainder {
    pub bits: u128,
    pub status: Status,
    /// Low three bits of the quotient's magnitude, reported in C0/C3/C1.
    pub quotient: u64,
    /// Set when the reduction did not complete, which x87 reports as C2.
    pub incomplete: bool,
}

/// FPREM (`ieee` false, quotient truncated toward zero) and FPREM1 (`ieee`
/// true, quotient rounded to nearest even). The remainder itself is always
/// exact, so no rounding control applies.
///
/// x87 reduces at most 63 binary exponents at a time. Beyond that it performs
/// a partial reduction and sets C2 so the caller loops. No hardware capture in
/// the corpus reaches that path, so the exact number of exponents consumed per
/// partial step is not pinned down here; a vector that exercises it should be
/// captured before this branch is relied on.
pub(super) fn remainder(raw: u128, divisor: u128, ieee: bool) -> Remainder {
    let x = value(raw);
    let y = value(divisor);

    let indefinite = Remainder {
        bits: 0xffff_c000_0000_0000_0000,
        status: Status::INVALID_OP,
        quotient: 0,
        incomplete: false,
    };
    // A zero divisor or an infinite dividend is invalid, and x87's masked
    // result is the indefinite QNaN rather than APFloat's NaN.
    if y.is_zero() || x.is_infinite() {
        return indefinite;
    }
    if x.is_nan() || y.is_nan() {
        let status = if x.is_signaling() || y.is_signaling() {
            Status::INVALID_OP
        } else {
            Status::OK
        };
        return Remainder {
            status,
            ..indefinite
        };
    }
    if x.is_zero() || y.is_infinite() {
        return Remainder {
            bits: raw,
            status: Status::OK,
            quotient: 0,
            incomplete: false,
        };
    }

    let exponent_span = i32::from(x.ilogb()) - i32::from(y.ilogb());
    if exponent_span >= 64 {
        // Partial reduction: bring the dividend within reach of one more step.
        let scaled = y.scalbn(exponent_span - 32);
        let value = x.c_fmod(scaled);
        return Remainder {
            bits: bits(value.value),
            status: value.status,
            quotient: 0,
            incomplete: true,
        };
    }

    let value = if ieee { x.ieee_rem(y) } else { x.c_fmod(y) };
    // The quotient is exact and fits once the span is under 64 exponents.
    let mut exact = false;
    let quotient = x
        .div_r(y, if ieee { Round::NearestTiesToEven } else { Round::TowardZero })
        .value
        .to_i128_r(
            80,
            if ieee { Round::NearestTiesToEven } else { Round::TowardZero },
            &mut exact,
        )
        .value;
    Remainder {
        bits: bits(value.value),
        status: value.status,
        quotient: quotient.unsigned_abs() as u64,
        incomplete: false,
    }
}

/// The ten-byte packed-decimal indefinite, stored when a value cannot be
/// represented as eighteen decimal digits.
const BCD_INDEFINITE: u128 = 0xffff_c000_0000_0000_0000;

/// The largest magnitude eighteen packed decimal digits can hold.
const BCD_MAX: i128 = 999_999_999_999_999_999;

/// FBLD: eighteen packed decimal digits in bytes 0..9 with the sign in bit 7
/// of byte 9. Every such value is exactly representable in extended format.
pub(super) fn from_bcd(raw: u128) -> u128 {
    let mut magnitude: i128 = 0;
    let mut scale: i128 = 1;
    for byte in 0..9 {
        let packed = ((raw >> (byte * 8)) & 0xff) as i128;
        magnitude += (packed & 0xf) * scale + (packed >> 4) * scale * 10;
        scale *= 100;
    }
    if (raw >> 79) & 1 == 1 {
        magnitude = -magnitude;
    }
    bits(X87DoubleExtended::from_i128_r(magnitude, Round::NearestTiesToEven).value)
}

/// FBSTP: round to an integer under the rounding control, then pack it as
/// eighteen decimal digits. A value that will not fit — including a NaN or an
/// infinity — stores the packed-decimal indefinite and raises invalid.
pub(super) fn to_bcd(raw: u128, control: u16) -> Result {
    let rounded = round_to_integral_contextual(raw, control);
    let mut exact = false;
    let converted = value(rounded.bits).to_i128_r(80, Round::TowardZero, &mut exact);
    let magnitude = converted.value;
    if converted.status.contains(Status::INVALID_OP) || magnitude.abs() > BCD_MAX {
        return Result {
            bits: BCD_INDEFINITE,
            status: Status::INVALID_OP,
        };
    }

    let mut packed = 0u128;
    let mut remaining = magnitude.unsigned_abs();
    for byte in 0..9 {
        let low = remaining % 10;
        let high = (remaining / 10) % 10;
        packed |= ((low | (high << 4)) as u128) << (byte * 8);
        remaining /= 100;
    }
    // The sign comes from the operand, so a negative zero stays negative.
    if (raw >> 79) & 1 == 1 {
        packed |= 1u128 << 79;
    }
    Result {
        bits: packed,
        status: rounded.status,
    }
}

/// Correctly rounded 80-bit square root.
///
/// APFloat has no square root, and routing f80 through f64 loses eleven
/// significand bits. Compute it on the integer significand instead: shift so
/// the exponent is even, take an integer square root wide enough for 64
/// significand bits, and round the remainder under the rounding control.
pub(super) fn sqrt_contextual(raw: u128, control: u16) -> Result {
    let round = round_from_control(control);
    let source = value(raw);

    if source.is_nan() {
        return Result {
            bits: if source.is_signaling() {
                0xffff_c000_0000_0000_0000
            } else {
                raw
            },
            status: if source.is_signaling() {
                Status::INVALID_OP
            } else {
                Status::OK
            },
        };
    }
    // sqrt of a negative is invalid; negative zero returns itself.
    if source.is_negative() && !source.is_zero() {
        return Result {
            bits: 0xffff_c000_0000_0000_0000,
            status: Status::INVALID_OP,
        };
    }
    if source.is_zero() || source.is_infinite() {
        return Result {
            bits: raw,
            status: Status::OK,
        };
    }

    // Renormalize so the significand's integer bit is set. `ilogb` gives the
    // unbiased exponent of the leading bit for denormals too.
    let leading = i32::from(source.ilogb());
    let normalized = source.scalbn(-leading);
    let significand = bits(normalized) as u64;

    // `value == significand * 2^power`, and the significand's integer bit is
    // bit 63, so power accounts for the 63 fraction bits.
    let mut power = leading - 63;
    let mut wide = u128::from(significand);
    // The root must come out with its integer bit set, and shifting the
    // radicand by an even amount moves the root by half of it. A significand
    // in [2^63, 2^64) needs 64; making the power even first doubles it into
    // [2^64, 2^65), which needs 62. No single shift covers both.
    let shift = if power & 1 != 0 {
        wide <<= 1;
        power -= 1;
        62
    } else {
        64
    };

    let radicand = wide << shift;
    let root = radicand.isqrt();
    let remainder = radicand - root * root;

    let mut significand = root as u64;
    let mut exponent = power / 2 - shift / 2 + 63;
    let inexact = remainder != 0;
    if inexact {
        let round_up = match round {
            // (root + 1/2)^2 <= radicand exactly when remainder > root.
            Round::NearestTiesToEven | Round::NearestTiesToAway => remainder > root,
            Round::TowardPositive => true,
            Round::TowardNegative | Round::TowardZero => false,
        };
        if round_up {
            let (next, carry) = significand.overflowing_add(1);
            if carry {
                significand = 1 << 63;
                exponent += 1;
            } else {
                significand = next;
            }
        }
    }

    let biased = (exponent + 16383) as u128 & 0x7fff;
    Result {
        bits: (biased << 64) | u128::from(significand),
        status: if inexact { Status::INEXACT } else { Status::OK },
    }
}

/// FXTRACT's significand: the operand with its exponent forced to zero, so
/// the result lies in [1, 2) with the operand's sign. A zero, infinity or NaN
/// is returned unchanged.
pub(super) fn extract_significand(raw: u128) -> u128 {
    let source = value(raw);
    if source.is_zero() || source.is_infinite() || source.is_nan() {
        return raw;
    }
    bits(source.scalbn(-i32::from(source.ilogb())))
}

/// FXTRACT's exponent: the operand's unbiased base-two exponent as an integer
/// valued extended float. A denormal reports the exponent of its normalized
/// form. Zero yields negative infinity and raises divide-by-zero; an infinity
/// yields positive infinity, and a NaN is returned unchanged.
pub(super) fn extract_exponent(raw: u128) -> Result {
    let source = value(raw);
    if source.is_nan() {
        return Result {
            bits: raw,
            status: Status::OK,
        };
    }
    if source.is_zero() {
        return Result {
            bits: 0xffff_8000_0000_0000_0000,
            status: Status::DIV_BY_ZERO,
        };
    }
    if source.is_infinite() {
        return Result {
            bits: 0x7fff_8000_0000_0000_0000,
            status: Status::OK,
        };
    }
    Result {
        bits: bits(
            X87DoubleExtended::from_i128_r(
                i128::from(source.ilogb()),
                Round::NearestTiesToEven,
            )
            .value,
        ),
        status: Status::OK,
    }
}

pub(super) fn negate(raw: u128) -> u128 {
    bits(-value(raw))
}

pub(super) fn abs(raw: u128) -> u128 {
    bits(value(raw).abs())
}

pub(super) fn is_nan(raw: u128) -> bool {
    value(raw).is_nan()
}

/// A signalling NaN raises invalid even for a quiet comparison. x87's ordered
/// compares additionally raise it for a quiet NaN, which the FCOM
/// constructors express; the unordered FUCOM forms do not.
pub(super) fn is_signaling_nan(raw: u128) -> bool {
    value(raw).is_signaling()
}

pub(super) fn equal(lhs: u128, rhs: u128) -> bool {
    value(lhs) == value(rhs)
}

pub(super) fn less(lhs: u128, rhs: u128) -> bool {
    value(lhs) < value(rhs)
}

pub(super) fn less_equal(lhs: u128, rhs: u128) -> bool {
    value(lhs) <= value(rhs)
}
