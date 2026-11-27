//! A pixel or mesh program's reads of another lane of its quad: `ds_swizzle_b32` in its quad
//! and bitmask modes and `quad_perm` DPP (`Lower::ds`, `Lower::dpp`). A child module of the
//! lowering, so that `lower.rs` stays within the policy's size limit.
//!
//! Host and console number a quad alike (0 1 over 2 3), and the console's quad operations
//! read the helper lanes of a partial quad, as a DXIL quad read does and a wave read does not
//! (SM 6.0 to 6.6). In a pixel program a quad-local read is a quad read only where its source
//! is defined as the console has it:
//!
//! - **Every lane of the quad runs the block** (no per-lane branch or loop around it,
//!   [`quad_blocks`]): a quad read of a lane that is not there is undefined.
//! - **A source lane outside EXEC** (a guard, not a branch: its EXEC bit is the
//!   instruction's guard in that lane, read with the value) gives the swizzle 0; DPP with
//!   `fi` reads the value, with `bound_ctrl` 0. With neither, the readings differ (LLVM's
//!   GFX10 rules keep the destination, RDNA2's field table reads 0): refused where a guard is.
//! - **DPP row and bank masks** name the console's rows and banks; a host quad sits
//!   elsewhere in the host wave, so a pixel DPP with masks is refused.
//!
//! Mesh programs read through [`Lower::quad_pick`] as before: their lanes are the console's.

use super::super::bitcode::{Attr, Bin, Cast, V, pred};
use super::super::shader::flags;
use super::{Lower, Why};
use crate::gfx10::Dpp;
use crate::structure::{LaneExit, Structured};

/// How a read of another lane lowers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum QuadPath {
    /// A mesh program's quad: [`Lower::quad_pick`].
    Mesh,
    /// A pixel program's quad, every lane of it here: quad reads, EXEC kept.
    Pixel,
    /// A wave read.
    Wave,
}

/// Whether a `ds_swizzle_b32` offset reads within a quad. Bit 15 clear: the bitmask mode
/// (and, or, xor on the lane within 32), within the quad when the and keeps bits 2..4 and
/// the or and xor leave them. `0x80xx`: the quad mode. The rotate and FFT modes (the rest of
/// bit 15 set) are refused.
pub(super) fn swizzle_in_quad(offset: u32) -> Result<bool, Why> {
    if offset & 0x8000 == 0 {
        return Ok(offset & 0x1C == 0x1C && offset & 0x380 == 0 && offset >> 10 & 0x1F < 4);
    }
    if offset & 0xFF00 != 0x8000 {
        return Err("a ds_swizzle_b32 rotate or FFT mode");
    }
    Ok(true)
}

/// The IR blocks every lane runs: `all_lane_blocks`, less each per-lane branch a lane can
/// come back to (a loop's test: a lane that left is not there the next round), which is no
/// block of its own region. None with an indirect jump.
fn quad_blocks(st: &Structured) -> Option<Vec<bool>> {
    let mut all = super::all_lane_blocks(st)?;
    let to = |b: usize, d: usize| match st.exits[b] {
        LaneExit::Jump(t) => t == d,
        LaneExit::Branch { taken, not_taken, .. } => taken == d || not_taken == d,
        LaneExit::End | LaneExit::Indirect => false,
    };
    for (d, region) in super::lane_regions(st)? {
        if to(d, d) || (0..region.len()).any(|b| region[b] && to(b, d)) {
            all[d] = false;
        }
    }
    Some(all)
}

impl Lower<'_> {
    /// How a read of another lane (`quad`: of its quad) lowers here.
    pub(super) fn quad_path(&mut self, quad: bool) -> Result<QuadPath, Why> {
        if quad && self.s.mesh.is_some() {
            return Ok(QuadPath::Mesh);
        }
        if quad && self.s.pixel {
            let st = self.x.st;
            let all = self.quad_all.get_or_insert_with(|| quad_blocks(st));
            if all.as_ref().is_some_and(|a| a.get(self.cur_ir).copied().unwrap_or(false)) {
                return Ok(QuadPath::Pixel);
            }
            // shortcut: where not every lane of the quad runs, the wave read of before quad
            // reads (undefined for a helper or an absent source lane, as it was). Prove the
            // quad's lanes there, or reconverge the read, if such sites show.
            self.s.flags |= flags::WAVE_OPS;
            return Ok(QuadPath::Wave);
        }
        self.cross_lane()?;
        Ok(QuadPath::Wave)
    }

    /// `x` of lane `src` as `path` reads it; `guard`: the instruction's. `dpp`: the DPP form
    /// and whether it has row or bank masks; none for a swizzle.
    pub(super) fn quad_value(
        &mut self,
        path: QuadPath,
        x: V,
        src: V,
        guard: Option<V>,
        dpp: Option<(Dpp, bool)>,
    ) -> Result<V, Why> {
        if self.s.pixel && dpp.is_some_and(|d| d.1) {
            return Err("a DPP row or bank mask in a pixel shader");
        }
        match path {
            QuadPath::Mesh => return Ok(self.quad_pick(x, src)),
            QuadPath::Wave => return Ok(self.read_lane(x, src)),
            QuadPath::Pixel => {}
        }
        let r = self.quad_pick(x, src);
        // no guard: every lane of the quad is here and in EXEC
        let Some(g) = guard else {
            return Ok(r);
        };
        let (fi, zero) = match dpp {
            None => (false, true),
            Some((Dpp::Row { fi, bound_ctrl, .. }, _)) => (fi, bound_ctrl),
            Some((Dpp::Lanes { fi, .. }, _)) => (fi, false),
        };
        if fi {
            return Ok(r);
        }
        if !zero {
            return Err("a DPP read of a lane EXEC may leave out, without fi or bound_ctrl");
        }
        let c = self.cur;
        let gi = self.s.m.cast(c, Cast::ZExt, g, self.s.t.i32);
        let on = self.quad_pick(gi, src);
        let z = self.s.m.i32(0);
        let on = self.s.m.cmp(c, pred::NE, on, z);
        Ok(self.s.m.select(c, on, r, z))
    }

    /// `x` of lane `src` (in this lane's quad) through quad reads, whose
    /// lane must be a constant.
    fn quad_pick(&mut self, x: V, src: V) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let three = self.s.m.i32(3);
        let q = self.s.m.bin(c, Bin::And, src, three);
        let mut r = x;
        for n in 0..4 {
            let kn = self.s.m.i32(n);
            let read = self.s.op(
                c,
                "quadReadLaneAt.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                122,
                &[x, kn],
            );
            let is = self.s.m.cmp(c, pred::EQ, q, kn);
            r = self.s.m.select(c, is, read, r);
        }
        r
    }
}
