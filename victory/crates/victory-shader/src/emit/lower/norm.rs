//! Exact normalized-format quotients: a UNORM or SNORM code converted to
//! the float nearest its exact quotient `a / (2^n - 1)`, as the console's
//! buffer unit gives it (after the base fork's da82e444e). The host's own
//! division lands one ulp off on some codes on hosts dividing through a
//! reciprocal (the RX 6700 XT gave 125 of the 256 8-bit codes one ulp high
//! and 16-bit codes such as 257 one ulp low), and titles truncate `x * d`
//! back to the code (a material index in a vertex colour's alpha), which a
//! quotient one ulp low turns into the code below.
//!
//! A child of `lower`: it builds on [`Lower`]'s private fields, and lives
//! apart to keep `lower.rs` under the policy's file-size cap.

use super::super::bitcode::{Bin, Cast, V, pred};
use super::Lower;

impl Lower<'_> {
    /// The bits of the float nearest to `a / d`, `d = 2^w - 1`, for codes
    /// `0 <= a < d` and `w <= 16`: what the console's buffer unit gives for
    /// a normalized code.
    ///
    /// `Q = floor(a * 2^32 / d) = a * K + floor(a * r / d)` with `K =
    /// floor(2^32 / d)` and `r = 2^32 mod d` (no division when `r` is 1:
    /// widths dividing 32). `a / d` is never exact for `0 < a < d` (`d`
    /// odd, `r` coprime to it), so `Q` rounds to 24 bits up exactly when
    /// its highest dropped bit is set, and the result is that integer (at
    /// most 2^24, so its conversion is exact on any host) scaled by a power
    /// of two; `Q` itself when it fits, the quotient's further bits then
    /// lying below half an ulp. That holds for every code of every width but
    /// 11 and 12 (checked for all codes of widths 2 to 16), which take
    /// [`Self::exact_norm`]. The host's conversion of a larger integer is
    /// not relied on: WARP rounds those above 2^31 otherwise.
    pub(super) fn norm_quotient(&mut self, a: V, w: u32) -> V {
        if matches!(w, 11 | 12) {
            return self.exact_norm(a, w);
        }
        let c = self.cur;
        let d = (1u64 << w) - 1;
        let (kk, r) = ((1u64 << 32) / d, (1u64 << 32) % d);
        let k = |l: &mut Self, v: u64| l.s.m.i32(v as u32);
        let (one, kv) = (k(self, 1), k(self, kk));
        let mut q = self.s.m.bin(c, Bin::Mul, a, kv);
        if r != 1 {
            let (rv, dv) = (k(self, r), k(self, d));
            let t = self.s.m.bin(c, Bin::Mul, a, rv);
            let more = self.s.m.bin(c, Bin::UDiv, t, dv);
            q = self.s.m.bin(c, Bin::Add, q, more);
        }
        // The bits to drop (1 to 8) from the exponent of `q >> 8`, which
        // converts exactly (fewer than 25 bits); none when `q` fits.
        let k8 = k(self, 8);
        let top = self.s.m.bin(c, Bin::LShr, q, k8);
        let top = self.s.m.cast(c, Cast::UiToFp, top, self.s.t.f32);
        let top = self.as_int(top);
        let k23 = k(self, 23);
        let e = self.s.m.bin(c, Bin::LShr, top, k23);
        let bias = k(self, 127 + 15);
        let shift = self.s.m.bin(c, Bin::Sub, e, bias);
        let (zero, b24) = (k(self, 0), k(self, 1 << 24));
        let fits = self.s.m.cmp(c, pred::IULT, q, b24);
        let shift = self.s.m.select(c, fits, zero, shift);
        // Rounded up when the highest dropped bit is set: `a / d` is never
        // exact, so a dropped half is more than half, never a tie.
        let up = self.s.m.bin(c, Bin::Shl, q, one);
        let up = self.s.m.bin(c, Bin::LShr, up, shift);
        let up = self.s.m.bin(c, Bin::And, up, one);
        let m = self.s.m.bin(c, Bin::LShr, q, shift);
        let m = self.s.m.bin(c, Bin::Add, m, up);
        // At most 2^24: exact as a float, and so its scale by 2^(shift - 32).
        let f = self.s.m.cast(c, Cast::UiToFp, m, self.s.t.f32);
        let b95 = k(self, 127 - 32);
        let scale = self.s.m.bin(c, Bin::Add, shift, b95);
        let scale = self.s.m.bin(c, Bin::Shl, scale, k23);
        let scale = self.as_float(scale);
        let f = self.s.m.bin(c, Bin::Mul, f, scale);
        self.as_int(f)
    }

    /// [`Self::norm_quotient`] in integers only, for any width up to 16: `Q`
    /// as there, 8 more bits when that leaves fewer than 25 (codes of wide
    /// formats near 0), then rounded to 24 bits to nearest even with the
    /// remainder as sticky bit, and assembled as float bits.
    fn exact_norm(&mut self, a: V, w: u32) -> V {
        let c = self.cur;
        let d = (1u64 << w) - 1;
        let (kk, r) = ((1u64 << 32) / d, (1u64 << 32) % d);
        let k = |l: &mut Self, v: u64| l.s.m.i32(v as u32);
        let (one, zero, dv, k8) = (k(self, 1), k(self, 0), k(self, d), k(self, 8));
        let kv = k(self, kk);
        let mut q = self.s.m.bin(c, Bin::Mul, a, kv);
        let mut rem = a;
        if r != 1 {
            let rv = k(self, r);
            let t = self.s.m.bin(c, Bin::Mul, a, rv);
            let more = self.s.m.bin(c, Bin::UDiv, t, dv);
            q = self.s.m.bin(c, Bin::Add, q, more);
            rem = self.s.m.bin(c, Bin::URem, t, dv);
        }
        // The exponent's bias less the 32 (or 40) fraction bits of `q`; the
        // shift below is added.
        let mut bias = k(self, 127 + 23 - 32);
        if kk < 1 << 24 {
            // Fewer than 25 bits for the smallest codes: 8 more.
            let t = self.s.m.bin(c, Bin::Shl, rem, k8);
            let more = self.s.m.bin(c, Bin::UDiv, t, dv);
            let rest = self.s.m.bin(c, Bin::URem, t, dv);
            let wide = self.s.m.bin(c, Bin::Shl, q, k8);
            let wide = self.s.m.bin(c, Bin::Or, wide, more);
            let limit = k(self, 1 << 24);
            let small = self.s.m.cmp(c, pred::IULT, q, limit);
            q = self.s.m.select(c, small, wide, q);
            rem = self.s.m.select(c, small, rest, rem);
            let b40 = k(self, 127 + 23 - 40);
            bias = self.s.m.select(c, small, b40, bias);
        }
        // Bits to drop: the bit length less 24, 1 to 8.
        let mut shift = one;
        for bit in 25..32 {
            let at = k(self, 1 << bit);
            let ge = self.s.m.cmp(c, pred::IUGE, q, at);
            let add = self.s.m.select(c, ge, one, zero);
            shift = self.s.m.bin(c, Bin::Add, shift, add);
        }
        let mant = self.s.m.bin(c, Bin::LShr, q, shift);
        let below = self.s.m.bin(c, Bin::Sub, shift, one);
        let half = self.s.m.bin(c, Bin::LShr, q, below);
        let half = self.s.m.bin(c, Bin::And, half, one);
        let low = self.s.m.bin(c, Bin::Shl, one, below);
        let low = self.s.m.bin(c, Bin::Sub, low, one);
        let low = self.s.m.bin(c, Bin::And, q, low);
        let rest = self.s.m.bin(c, Bin::Or, low, rem);
        let sticky = self.s.m.cmp(c, pred::NE, rest, zero);
        let odd = self.s.m.bin(c, Bin::And, mant, one);
        let tie = self.s.m.select(c, sticky, one, odd);
        let up = self.s.m.bin(c, Bin::And, half, tie);
        let mant = self.s.m.bin(c, Bin::Add, mant, up);
        // Rounding up to 2^24 moves to the next exponent.
        let k24 = k(self, 24);
        let carry = self.s.m.bin(c, Bin::LShr, mant, k24);
        let mant = self.s.m.bin(c, Bin::LShr, mant, carry);
        let e = self.s.m.bin(c, Bin::Add, bias, shift);
        let e = self.s.m.bin(c, Bin::Add, e, carry);
        let k23 = k(self, 23);
        let e = self.s.m.bin(c, Bin::Shl, e, k23);
        let fraction = k(self, 0x7F_FFFF);
        let m = self.s.m.bin(c, Bin::And, mant, fraction);
        let bits = self.s.m.bin(c, Bin::Or, e, m);
        let is_zero = self.s.m.cmp(c, pred::EQ, a, zero);
        self.s.m.select(c, is_zero, zero, bits)
    }

    /// A normalized (UNORM or SNORM) format's field converted to float as
    /// the format defines it: `field` is its bits, `w` its width; the
    /// format's ends are exactly 1.0 and -1.0, and its quotient is
    /// [`Self::norm_quotient`]'s where that applies.
    pub(super) fn normalized_field(&mut self, field: V, w: u32, unorm: bool) -> V {
        let c = self.cur;
        let f32t = self.s.t.f32;
        let signed = |l: &mut Self| {
            if w == 32 {
                return field;
            }
            let k = l.s.m.i32(32 - w);
            let t = l.s.m.bin(c, Bin::Shl, field, k);
            l.s.m.bin(c, Bin::AShr, t, k)
        };
        let q = if unorm {
            let top = (1u64 << w) - 1;
            let q = if w <= 16 {
                let q = self.norm_quotient(field, w);
                self.as_float(q)
            } else {
                let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                let max = self.s.m.f32((top as f32).to_bits());
                self.s.m.bin(c, Bin::SDiv, f, max)
            };
            let (high, one) = (self.s.m.i32(top as u32), self.s.m.f32(1.0f32.to_bits()));
            let at_top = self.s.m.cmp(c, pred::EQ, field, high);
            self.s.m.select(c, at_top, one, q)
        } else {
            let s = signed(self);
            let top = (1u32 << (w - 1)) - 1;
            let q = if w <= 17 {
                // The magnitude's quotient, signed (codes at and below -top
                // are -1.0 below).
                let (zero, sign) = (self.s.m.i32(0), self.s.m.i32(0x8000_0000));
                let neg = self.s.m.cmp(c, pred::SLT, s, zero);
                let minus = self.s.m.bin(c, Bin::Sub, zero, s);
                let m = self.s.m.select(c, neg, minus, s);
                let q = self.norm_quotient(m, w - 1);
                let sign = self.s.m.select(c, neg, sign, zero);
                let q = self.s.m.bin(c, Bin::Or, q, sign);
                self.as_float(q)
            } else {
                let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                let max = self.s.m.f32((top as f32).to_bits());
                self.s.m.bin(c, Bin::SDiv, f, max)
            };
            // The lowest value is -1.0 too, as is the one above it.
            let (high, one) = (self.s.m.i32(top), self.s.m.f32(1.0f32.to_bits()));
            let at_top = self.s.m.cmp(c, pred::SGE, s, high);
            let q = self.s.m.select(c, at_top, one, q);
            let low = self.s.m.i32(top.wrapping_neg());
            let minus_one = self.s.m.f32((-1.0f32).to_bits());
            let at_bottom = self.s.m.cmp(c, pred::SLE, s, low);
            self.s.m.select(c, at_bottom, minus_one, q)
        };
        self.as_int(q)
    }
}
