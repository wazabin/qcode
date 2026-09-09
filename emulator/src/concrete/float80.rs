//! Exact IEEE-754 80-bit floating-point operations for the concrete emulator.
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

/// FILD's conversion. Every integer x87 loads fits the 64-bit significand
/// exactly, so this is an exact nearest-even conversion with no status.
pub(super) fn from_i128(value: i128) -> u128 {
    bits(X87DoubleExtended::from_i128_r(value, Round::NearestTiesToEven).value)
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
    bits(value(lhs).add_r(value(rhs), Round::NearestTiesToEven).value)
}

pub(super) fn sub(lhs: u128, rhs: u128) -> u128 {
    bits(value(lhs).sub_r(value(rhs), Round::NearestTiesToEven).value)
}

pub(super) fn mul(lhs: u128, rhs: u128) -> u128 {
    bits(value(lhs).mul_r(value(rhs), Round::NearestTiesToEven).value)
}

pub(super) fn div(lhs: u128, rhs: u128) -> u128 {
    bits(value(lhs).div_r(value(rhs), Round::NearestTiesToEven).value)
}

/// IEEE scaleB: multiply by two raised to an integral power, rounded once.
/// Overflow, underflow and inexact are derived from the value the format can
/// hold; every architectural special case belongs to the caller.
pub(super) fn scalb_ieee(raw: u128, steps: i32, round: Round) -> Result {
    let source = value(raw);
    let scaled = source.scalbn_r(steps, round);
    let mut status = Status::OK;
    if source.is_finite() && !source.is_zero() {
        // Scaling is exact unless the format cannot hold the answer, and the
        // direction of the scale says which end it ran off. Under a directed
        // rounding mode an overflow delivers the largest finite value rather
        // than an infinity, so the result's class cannot decide this.
        let restored = scaled.scalbn_r(-steps, Round::NearestTiesToEven);
        if scaled.is_infinite() || scaled.is_zero() || restored != source {
            status |= Status::INEXACT
                | if steps > 0 {
                    Status::OVERFLOW
                } else {
                    Status::UNDERFLOW
                };
        }
    }
    Result {
        bits: bits(scaled),
        status,
    }
}

/// Base-two logarithm.  rustc_apfloat has no transcendental functions, so the
/// value is computed in double precision; the special cases - which are the
/// only part the hardware corpus can check - are exact.
pub(super) fn log2_ieee(raw: u128) -> Result {
    let source = value(raw);
    if source.is_nan() {
        return Result {
            bits: raw,
            status: if source.is_signaling() {
                Status::INVALID_OP
            } else {
                Status::OK
            },
        };
    }
    if source.is_zero() {
        return Result {
            bits: 0xffff_8000_0000_0000_0000,
            status: Status::DIV_BY_ZERO,
        };
    }
    if source.is_negative() {
        return Result {
            bits: raw,
            status: Status::INVALID_OP,
        };
    }
    if source.is_infinite() {
        return Result {
            bits: 0x7fff_8000_0000_0000_0000,
            status: Status::OK,
        };
    }
    let exact_one = bits(source) == 0x3fff_8000_0000_0000_0000;
    Result {
        bits: from_f64(to_f64(raw).log2()),
        status: if exact_one {
            Status::OK
        } else {
            Status::INEXACT
        },
    }
}

/// Round to an integral value in the same format under an explicit rounding
/// mode.  Inexact is the only exception it can report.
pub(super) fn round_to_integral_ieee(raw: u128, round: Round) -> Result {
    result(value(raw).round_to_integral(round))
}

/// An exact partial remainder, plus the quotient facts a caller may report.
#[derive(Clone, Copy, Debug)]
pub(super) struct Remainder {
    pub bits: u128,
    /// Low three bits of the quotient's magnitude.
    pub quotient: u64,
    /// Set when the reduction was only partial, so no quotient bits exist.
    pub incomplete: bool,
}

/// The partial remainder. `ieee` false truncates the quotient toward zero;
/// `ieee` true rounds it to nearest even. The remainder is exact under either
/// rule, so no result rounding mode applies.
///
/// At most 63 quotient bits are produced per call, and they are produced 32 at
/// a time: when the operands' exponent span reaches 64 the reduction is partial
/// and consumes the largest multiple of 32 quotient bits that leaves at least
/// 32 of the span behind, so between 32 and 63 bits of span survive each step.
/// It computes the remainder modulo the divisor scaled by `2^bits`, sets
/// `incomplete`, and the caller repeats until it clears. The step size is
/// observable: it decides both the intermediate remainders and whether a
/// dividend that is an exact multiple of the divisor reduces to zero.
pub(super) fn remainder(raw: u128, divisor: u128, ieee: bool) -> Remainder {
    let x = value(raw);
    let y = value(divisor);

    let indefinite = Remainder {
        bits: 0xffff_c000_0000_0000_0000,
        quotient: 0,
        incomplete: false,
    };
    // A zero divisor or an infinite dividend has no remainder. The value a
    // caller substitutes for such an invalid operation is its own choice; the
    // default quiet NaN stands in until it does.
    if y.is_zero() || x.is_infinite() {
        return indefinite;
    }
    // The propagated NaN and the exception it does or does not raise are
    // architectural choices the specification makes from the operands.
    if x.is_nan() || y.is_nan() {
        return indefinite;
    }
    if x.is_zero() || y.is_infinite() {
        return Remainder {
            bits: raw,
            quotient: 0,
            incomplete: false,
        };
    }

    let exponent_span = x.ilogb() - y.ilogb();
    if exponent_span >= 64 {
        // Partial reduction: take as many whole 32-bit groups of quotient bits
        // as leave at least 32 exponents of span for the following steps.
        let scaled = y.scalbn((exponent_span - 32) / 32 * 32);
        let value = x.c_fmod(scaled);
        return Remainder {
            bits: bits(value.value),
            quotient: 0,
            incomplete: true,
        };
    }

    let value = if ieee { x.ieee_rem(y) } else { x.c_fmod(y) };
    // The quotient is exact and fits once the span is under 64 exponents.
    let mut exact = false;
    let quotient = x
        .div_r(
            y,
            if ieee {
                Round::NearestTiesToEven
            } else {
                Round::TowardZero
            },
        )
        .value
        .to_i128_r(
            80,
            if ieee {
                Round::NearestTiesToEven
            } else {
                Round::TowardZero
            },
            &mut exact,
        )
        .value;
    Remainder {
        bits: bits(value.value),
        quotient: quotient.unsigned_abs() as u64,
        incomplete: false,
    }
}

/// The ten-byte packed-decimal indefinite, stored when a value cannot be
/// represented as eighteen decimal digits.
const BCD_INDEFINITE: u128 = 0xffff_c000_0000_0000_0000;

/// The largest magnitude eighteen packed decimal digits can hold.
const BCD_MAX: i128 = 999_999_999_999_999_999;

/// Pack a value as eighteen signed decimal digits after rounding it to an
/// integer under `round`. The conversion is exact when it succeeds and
/// inexact when rounding discarded a fraction; a value that will not fit -
/// including a NaN or an infinity - is invalid and yields the packed-decimal
/// indefinite. Choosing what to do with either fact is the caller's.
pub(super) fn to_bcd(raw: u128, round: Round) -> Result {
    let rounded = round_to_integral_ieee(raw, round);
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
/// Correctly rounded square root of an extended value under an explicit
/// rounding mode.  The only exceptions it reports are the IEEE ones: invalid
/// for a negative operand or a signalling NaN, inexact for a rounded root.
/// Choosing what an invalid square root delivers is the caller's policy.
pub(super) fn sqrt_ieee(raw: u128, round: Round) -> Result {
    let source = value(raw);

    if source.is_nan() {
        return Result {
            bits: raw,
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
            bits: raw,
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
    let leading = source.ilogb();
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
    bits(source.scalbn(-source.ilogb()))
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
            X87DoubleExtended::from_i128_r(i128::from(source.ilogb()), Round::NearestTiesToEven)
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

pub(super) fn equal(lhs: u128, rhs: u128) -> bool {
    value(lhs) == value(rhs)
}

pub(super) fn less(lhs: u128, rhs: u128) -> bool {
    value(lhs) < value(rhs)
}

pub(super) fn less_equal(lhs: u128, rhs: u128) -> bool {
    value(lhs) <= value(rhs)
}
