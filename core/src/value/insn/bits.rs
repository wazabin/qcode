/// Returns a bitmask covering the low `size` bytes (i.e. `size * 8` bits).
pub fn mask_for_size(size: usize) -> u128 {
    let bits = size.saturating_mul(8);
    if bits >= u128::BITS as usize {
        u128::MAX
    } else if bits == 0 {
        0u128
    } else {
        (1u128 << bits) - 1
    }
}

/// Interprets a raw bit pattern as a signed integer of the given byte width.
pub fn signed_value(bits: u128, size: usize) -> i128 {
    let nbits = size.saturating_mul(8);
    if nbits == 0 {
        return 0;
    }
    if nbits >= u128::BITS as usize {
        return bits as i128;
    }
    let sign_bit = 1u128 << (nbits - 1);
    let extended = if (bits & sign_bit) != 0 {
        bits | !mask_for_size(size)
    } else {
        bits
    };
    extended as i128
}
