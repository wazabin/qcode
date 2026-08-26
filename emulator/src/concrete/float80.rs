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
    if exponent == 0 || exponent == 0x7fff {
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
        return Result {
            bits: raw,
            status: Status::OK,
        };
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
    if exponent == 0x7fff {
        status |= Status::OVERFLOW;
    }
    Result {
        bits: rounded,
        status,
    }
}

fn arithmetic(
    value: rustc_apfloat::StatusAnd<X87DoubleExtended>,
    control: u16,
    round: Round,
) -> Result {
    let rounded = apply_precision(value.value, control, round);
    let status = value.status | rounded.status;
    // x87 masked invalid arithmetic writes the architectural indefinite QNaN,
    // rather than preserving APFloat's operation-specific NaN sign/payload.
    let bits = if status.contains(Status::INVALID_OP) {
        0xffff_c000_0000_0000_0000
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
    arithmetic(X87DoubleExtended::from_i128_r(value, round), control, round)
}

pub(super) fn from_f32_bits(raw: u32) -> u128 {
    let mut loses_info = false;
    bits(
        Single::from_bits(u128::from(raw))
            .convert_r(Round::NearestTiesToEven, &mut loses_info)
            .value,
    )
}

/// Narrow an f80 store using the x87 rounding-control field.  Precision
/// control governs arithmetic results, not the destination format; conversion
/// to f32/f64 is rounded directly to that format.
pub(super) fn to_float_contextual(raw: u128, size: usize, control: u16) -> Result {
    let round = round_from_control(control);
    let mut loses_info = false;
    match size {
        4 => {
            let value: rustc_apfloat::StatusAnd<Single> =
                value(raw).convert_r(round, &mut loses_info);
            Result {
                bits: value.value.to_bits(),
                status: value.status,
            }
        }
        8 => {
            let value: rustc_apfloat::StatusAnd<Double> =
                value(raw).convert_r(round, &mut loses_info);
            Result {
                bits: value.value.to_bits(),
                status: value.status,
            }
        }
        _ => Result {
            bits: raw,
            status: Status::OK,
        },
    }
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

pub(super) fn add(lhs: u128, rhs: u128) -> u128 {
    add_contextual(lhs, rhs, 0x037f).bits
}

pub(super) fn add_contextual(lhs: u128, rhs: u128, control: u16) -> Result {
    let round = round_from_control(control);
    arithmetic(value(lhs).add_r(value(rhs), round), control, round)
}

pub(super) fn sub(lhs: u128, rhs: u128) -> u128 {
    sub_contextual(lhs, rhs, 0x037f).bits
}

pub(super) fn sub_contextual(lhs: u128, rhs: u128, control: u16) -> Result {
    let round = round_from_control(control);
    arithmetic(value(lhs).sub_r(value(rhs), round), control, round)
}

pub(super) fn mul(lhs: u128, rhs: u128) -> u128 {
    mul_contextual(lhs, rhs, 0x037f).bits
}

pub(super) fn mul_contextual(lhs: u128, rhs: u128, control: u16) -> Result {
    let round = round_from_control(control);
    arithmetic(value(lhs).mul_r(value(rhs), round), control, round)
}

pub(super) fn div(lhs: u128, rhs: u128) -> u128 {
    div_contextual(lhs, rhs, 0x037f).bits
}

pub(super) fn div_contextual(lhs: u128, rhs: u128, control: u16) -> Result {
    let round = round_from_control(control);
    arithmetic(value(lhs).div_r(value(rhs), round), control, round)
}

pub(super) fn round_to_integral_contextual(raw: u128, control: u16) -> Result {
    let round = round_from_control(control);
    result(value(raw).round_to_integral(round))
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
