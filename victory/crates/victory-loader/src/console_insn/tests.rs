//! Tests of the console-instruction module. Encodings are generated from
//! operand fields by a small encoder (`encode`), field values by a fixed
//! generator, so no vector is a hand-picked constant; expectations come
//! from the manuals' tables and text, a bit-by-bit reference, or the CPU.

mod decoding;
mod emulation;
mod encode;
mod fields;
mod loops;

/// xorshift64* (Marsaglia; Vigna): values of no one's choosing, the same
/// on every run.
pub(super) struct Values(u64);

impl Values {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub(super) fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}
