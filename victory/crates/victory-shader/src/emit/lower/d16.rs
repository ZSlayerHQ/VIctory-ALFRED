//! 16-bit typed buffer loads (`buffer_load_format_d16_*`) through views of
//! integer formats, with `VICTORY_D16_INTEGER_VIEWS=1` only
//! ([`super::super::d16_integer_views_on`]; plan K5, after the base fork's
//! 4243a96a7 and our own 6514f1b89, 9b82bbaeb and dbc6ecd74 on
//! job/fork-accuracy-r2). Without the switch `lower::buffer` lowers every
//! d16 typed load as `world` does, and nothing here runs.
//!
//! On GFX10 a typed buffer access converts by the buffer format of its
//! descriptor, and `_d16` only packs the 16-bit results: through a UINT or
//! SINT view the load gives each channel's low 16 bits, not a half. The
//! view's format is in the descriptor, not in the code, so the kind is read
//! at run time from the host's slot table (`VIEW_INTEGER` beside the
//! stride). A float or normalized view keeps `world`'s narrowing, the same
//! calls in the same order.
//!
//! A child of `lower`: it builds on [`Lower`]'s private fields, and lives
//! apart to keep `lower.rs` under the policy's file-size cap.

use super::super::bitcode::{Attr, Bin, V};
use super::super::shader::flags;
use super::{Lower, Why};
use crate::ir::{Inst, Reg};
use crate::resource::{Site, Source};

/// A d16 typed load as `Lower::buffer` hands it over: the instruction, its
/// descriptor, the element index, its kind of view (`u`: `0x1000` for an
/// unordered one), its VGPRs and the guard its writes run under.
pub(super) struct D16Load<'a> {
    pub inst: &'a Inst,
    pub site: Site,
    pub src: &'a Source,
    pub index: V,
    pub u: u32,
    pub reg: u8,
    pub count: u32,
    pub guard: Option<V>,
}

impl Lower<'_> {
    /// Lowers a d16 typed load with the view kind read: two 16-bit channels
    /// a VGPR, low half first (an odd last high half 0). The kind picks one
    /// of two branches, so each view is read only through a handle of its
    /// own component type: a float or normalized view through the
    /// float-typed handle (`0x409`), narrowed by `legacyF32ToF16` exactly as
    /// on `world`; an integer view through an integer-typed one (`0x405`,
    /// `bufferLoad.i32`), each channel's low 16 bits. (A SINT view read as
    /// unsigned gives the same low 16 bits.) The branches' results meet in
    /// a select by the kind.
    pub(super) fn d16_typed_load(&mut self, l: D16Load<'_>) -> Result<(), Why> {
        let D16Load { inst, site, src, index, u, reg, count, guard } = l;
        if u != 0 {
            self.s.flags |= flags::TYPED_UAV_LOADS;
        }
        let iu = self.s.m.undef(self.s.t.i32);
        let float_h = self.handle(site, src, false, (10 | u, 0x409))?;
        let int_h = self.handle(site, src, false, (10 | u, 0x405))?;
        let (int, _) = self.slot_integer(float_h)?;
        let i32t = self.s.t.i32;
        let zero = self.s.m.i32(0);
        let dwords = count.div_ceil(2) as usize;
        // The float view's branch.
        let yes = self.s.m.bool(true);
        let float = self.s.m.bin(self.cur, Bin::Xor, int, yes);
        let at = self.begin_guarded(Some(float));
        let c = self.cur;
        let r = self
            .s
            .op(c, "bufferLoad.f32", Attr::ReadOnly, self.s.t.res_ret, 68, &[float_h, index, iu]);
        let halves = self.d16_pack(count, |l, ch| {
            let f = l.s.m.extract(l.cur, r, ch);
            l.s.op(l.cur, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
        });
        let outside: Vec<(V, V)> = halves.iter().map(|&x| (x, zero)).collect();
        let halves = self.end_guarded(at, &outside);
        // The integer view's branch.
        let at = self.begin_guarded(Some(int));
        let c = self.cur;
        let ret = self.s.m.named_struct("dx.types.ResRet.i32", vec![i32t; 5]);
        let r = self.s.op(c, "bufferLoad.i32", Attr::ReadOnly, ret, 68, &[int_h, index, iu]);
        let low = self.s.m.i32(0xFFFF);
        let bits = self.d16_pack(count, |l, ch| {
            let x = l.s.m.extract(l.cur, r, ch);
            l.s.m.bin(l.cur, Bin::And, x, low)
        });
        let outside: Vec<(V, V)> = bits.iter().map(|&x| (x, zero)).collect();
        let bits = self.end_guarded(at, &outside);
        let c = self.cur;
        for k in 0..dwords {
            let x = self.s.m.select(c, int, bits[k], halves[k]);
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// `count` 16-bit channels, each `channel` of its index, packed two a
    /// dword, low half first, in the current block (the higher channel made
    /// first, as `world`'s d16 load does).
    fn d16_pack(&mut self, count: u32, mut channel: impl FnMut(&mut Self, u32) -> V) -> Vec<V> {
        let k16 = self.s.m.i32(16);
        let mut out = Vec::new();
        for k in 0..count.div_ceil(2) {
            let mut x = None;
            for ch in (2 * k..(2 * k + 2).min(count)).rev() {
                let h = channel(self, ch);
                let c = self.cur;
                x = Some(match x {
                    Some(hi) => {
                        let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                        self.s.m.bin(c, Bin::Or, h, hi)
                    }
                    None => h,
                });
            }
            out.push(x.expect("a channel"));
        }
        out
    }
}
