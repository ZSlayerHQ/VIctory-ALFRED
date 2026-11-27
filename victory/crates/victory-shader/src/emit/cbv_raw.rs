//! Constant buffers in the view heap as raw buffer reads, with the slot map on
//! (`VICTORY_SLOT_MAP=1`; design `jobs/2026-11-25-slot-lookup`, section 3).
//!
//! A console constant buffer record starts on 16 bytes; a D3D12 constant buffer view starts
//! on 256. Without the map the host copies each table that holds one into fresh slots, per
//! draw, with the constants moved to 256 bytes. With the map there are no per-draw tables:
//! the host makes such a record a raw view of its memory (from its own start, as long as
//! its size), and the shader reads it by byte where it read rows of a constant buffer. A
//! read past the record's size reads zero, as the console's does. The map's class for it is
//! the raw buffer's, so a buffer view the program reads through a constant buffer load is
//! read too, as the console reads any buffer descriptor's memory. Root constant buffers are
//! root descriptors, outside the heap, and stay constant buffers.

use super::{Lower, Site, Why};
use crate::emit::bitcode::{Attr, Bin, V, pred};
use crate::resource::Source;

impl Lower<'_> {
    /// The handle a constant buffer read of the descriptor `src` at `site` takes: with the
    /// map on, a raw buffer's for a descriptor in the heap (noted in `Shader::raw_cbvs`,
    /// which [`Lower::cbuffer_dword`] reads), else a constant buffer's.
    pub(super) fn cbv_handle(&mut self, site: Site, src: &Source) -> Result<V, Why> {
        let in_heap = !matches!(src, Source::Built { .. } | Source::Constant(_));
        if !(in_heap && super::super::slot_map::on()) {
            return self.handle(site, src, false, (13, 0x1_0000));
        }
        let h = self.handle(site, src, false, (11, 0))?;
        self.s.raw_cbvs.insert(h);
        Ok(h)
    }

    /// The dword at a computed byte address of a constant buffer: of a raw one
    /// ([`Lower::cbv_handle`]), the dword there (the address's low two bits ignored, as
    /// the console ignores them); else component `(a >> 2) & 3` of row `a >> 4`.
    pub(super) fn cbuffer_dword(&mut self, h: V, a: V) -> V {
        let c = self.cur;
        if self.s.raw_cbvs.contains(&h) {
            let low = self.s.m.i32(!3);
            let at = self.s.m.bin(c, Bin::And, a, low);
            return self.s.raw_dword(c, h, at);
        }
        let four = self.s.m.i32(4);
        let ri = self.s.m.bin(c, Bin::LShr, a, four);
        let r = self.s.op(
            c,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, ri],
        );
        let two = self.s.m.i32(2);
        let three = self.s.m.i32(3);
        let comp = self.s.m.bin(c, Bin::LShr, a, two);
        let comp = self.s.m.bin(c, Bin::And, comp, three);
        let mut x = self.s.m.extract(c, r, 0);
        for j in 1..4 {
            let e = self.s.m.extract(c, r, j);
            let jc = self.s.m.i32(j);
            let is = self.s.m.cmp(c, pred::EQ, comp, jc);
            x = self.s.m.select(c, is, e, x);
        }
        x
    }
}
