//! Float-to-integer conversions. DXIL leaves the cast of a NaN to the host
//! (WARP's and Intel's give 0), and a host may fold a float compare for NaN
//! away (`is_nan_f32` exists for that; on Intel the floor form's `fcmp uno`
//! select let the cast's 0 through), so a NaN is found on the operand's bits
//! and never reaches the cast: the lowering picks the console's result for
//! it.
//!
//! A child of `lower`: it builds on [`Lower`]'s private fields, and lives
//! apart to keep `lower.rs` under the policy's file-size cap.

use super::super::bitcode::{Bin, Cast, V, pred};
use super::{Lower, Why};
use crate::gfx10::Fields;
use crate::ir::Inst;

impl Lower<'_> {
    /// Float source `k` as its bits, with the abs and neg modifiers applied
    /// to the sign bit as the hardware applies them: a host `fabs` or
    /// `0 - x` may flush a denormal, and leaves a NaN's sign as it likes.
    pub(super) fn fsrc_bits(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let mut x = match self.src0 {
            Some(x) if k == 0 => x,
            _ => {
                let op = *v.src.get(k).ok_or("a missing operand")?;
                let x = self.operand(inst, &op)?;
                match Self::sdwa_src(inst, k) {
                    Some((sel, sext)) => self.select_bits(x, sel, sext),
                    None => x,
                }
            }
        };
        if v.abs >> k & 1 != 0 {
            let m = self.s.m.i32(0x7FFF_FFFF);
            x = self.s.m.bin(c, Bin::And, x, m);
        }
        if v.neg >> k & 1 != 0 {
            let m = self.s.m.i32(0x8000_0000);
            x = self.s.m.bin(c, Bin::Xor, x, m);
        }
        Ok(x)
    }

    /// `f` with a NaN replaced by +0, and the test that found it: what a
    /// cast, or a clamp before one, may be given whatever the host does
    /// with a NaN there. The caller that picks the console's own result for
    /// a NaN reuses the test rather than making a second one.
    pub(super) fn nan_to_zero(&mut self, f: V) -> (V, V) {
        let nan = self.is_nan_f32(f);
        let zero = self.s.m.f32(0);
        (self.s.m.select(self.cur, nan, zero, f), nan)
    }

    /// `v_cvt_i32_f32`, `v_cvt_u32_f32`, `v_cvt_flr_i32_f32` and
    /// `v_cvt_rpi_i32_f32` of source 0. The plain forms truncate and give 0
    /// for a NaN. The floor form takes floor(x), the rpi form floor(x + 0.5)
    /// with the half added exactly (x - floor(x) is exact in f32), both of
    /// x with denormals read as zeros, and both saturate a NaN by its sign:
    /// the lowest i32 for a negative one, the highest for a positive one
    /// (measured on RDNA2 by the base fork).
    // shortcut: a value past the integer's range is left to the host's
    // cast, which saturates on the hosts tried (D3D's own conversions do).
    pub(super) fn cvt_int(&mut self, inst: &Inst, name: &str) -> Result<V, Why> {
        let c = self.cur;
        let xi = self.fsrc_bits(inst, 0)?;
        let x = self.as_float(xi);
        let (mut a, nan) = self.nan_to_zero(x);
        let floor = matches!(name, "v_cvt_flr_i32_f32" | "v_cvt_rpi_i32_f32");
        if floor {
            let flushed = self.ftz(a);
            let fl = self.unary(27, flushed);
            a = fl;
            if name == "v_cvt_rpi_i32_f32" {
                let frac = self.s.m.bin(c, Bin::Sub, flushed, fl);
                let h = self.s.m.f32(0.5f32.to_bits());
                let up = self.s.m.cmp(c, pred::OGE, frac, h);
                let one = self.s.m.f32(1.0f32.to_bits());
                let next = self.s.m.bin(c, Bin::Add, fl, one);
                a = self.s.m.select(c, up, next, fl);
            }
        }
        let op = if name == "v_cvt_u32_f32" {
            Cast::FpToUi
        } else {
            Cast::FpToSi
        };
        let r = self.s.m.cast(c, op, a, self.s.t.i32);
        if !floor {
            return Ok(r);
        }
        let zero = self.s.m.i32(0);
        let neg = self.s.m.cmp(c, pred::SLT, xi, zero);
        let (lo, hi) = (self.s.m.i32(0x8000_0000), self.s.m.i32(0x7FFF_FFFF));
        let sat = self.s.m.select(c, neg, lo, hi);
        Ok(self.s.m.select(c, nan, sat, r))
    }
}
