//! The console's float mode as the lowering reads it: denormal operands of
//! min, max, med3 and float selects are zeros of their sign, and -0 orders
//! below +0 (RDNA2; after the base fork's 77053a0c5).
//!
//! A child of `lower`: it builds on [`Lower`]'s private fields, and lives
//! apart to keep `lower.rs` under the policy's file-size cap.

use super::super::bitcode::{Bin, V, pred};
use super::Lower;

impl Lower<'_> {
    /// A 32-bit denormal as the console's float mode reads it: a zero of its
    /// sign, decided on the bits (the host's own denormal mode varies).
    pub(super) fn ftz(&mut self, f: V) -> V {
        let c = self.cur;
        let x = self.as_int(f);
        let (exp, sign, zero) = (
            self.s.m.i32(0x7F80_0000),
            self.s.m.i32(0x8000_0000),
            self.s.m.i32(0),
        );
        let e = self.s.m.bin(c, Bin::And, x, exp);
        let den = self.s.m.cmp(c, pred::EQ, e, zero);
        let s = self.s.m.bin(c, Bin::And, x, sign);
        let r = self.s.m.select(c, den, s, x);
        self.as_float(r)
    }

    /// The hardware's maximum (`max`) or minimum of two floats: as DXIL's
    /// FMax/FMin, but of the operands with denormals read as zeros and with
    /// -0 ordered below +0 (RDNA2 ISA, `v_max_f32`), where the compare and
    /// select DXIL specifies gives either zero.
    pub(super) fn max_min(&mut self, max: bool, a: V, b: V) -> V {
        let (a, b) = (self.ftz(a), self.ftz(b));
        self.max_min_flushed(max, a, b)
    }

    /// [`Self::max_min`] of operands without denormals.
    pub(super) fn max_min_flushed(&mut self, max: bool, a: V, b: V) -> V {
        let c = self.cur;
        let r = self.binary(if max { 35 } else { 36 }, a, b);
        let (ia, ib) = (self.as_int(a), self.as_int(b));
        let either = self.s.m.bin(c, Bin::Or, ia, ib);
        let magnitude = self.s.m.i32(0x7FFF_FFFF);
        let either = self.s.m.bin(c, Bin::And, either, magnitude);
        let zero = self.s.m.i32(0);
        let zeros = self.s.m.cmp(c, pred::EQ, either, zero);
        // Of two zeros, +0 unless both are -0 (max), -0 unless neither (min).
        let op = if max { Bin::And } else { Bin::Or };
        let sign = self.s.m.bin(c, op, ia, ib);
        let signed = self.as_float(sign);
        self.s.m.select(c, zeros, signed, r)
    }

    /// `v_med3_f32`: the median of three, of the operands flushed, a NaN in
    /// any of them giving the least (as the hardware does).
    pub(super) fn med3(&mut self, a: V, b: V, third: V) -> V {
        let (a, b, third) = (self.ftz(a), self.ftz(b), self.ftz(third));
        let c = self.cur;
        let lo = self.max_min_flushed(false, a, b);
        let hi = self.max_min_flushed(true, a, b);
        let t = self.max_min_flushed(false, hi, third);
        let median = self.max_min_flushed(true, lo, t);
        let least = self.max_min_flushed(false, lo, third);
        let nan_a = self.is_nan_f32(a);
        let nan_b = self.is_nan_f32(b);
        let nan_3 = self.is_nan_f32(third);
        let nan_ab = self.s.m.bin(c, Bin::Or, nan_a, nan_b);
        let nan = self.s.m.bin(c, Bin::Or, nan_ab, nan_3);
        self.s.m.select(c, nan, least, median)
    }

}
