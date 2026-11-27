//! `s_cbranch_vccz` and `vccnz` as the console takes them, for the whole wave: VCCZ is "VCC is
//! zero in all 64 bits", so with one lane's VCC bit set every lane of EXEC runs the region the
//! branch skips otherwise. Off by default (`VICTORY_WAVE_VCC=1`, [`crate::emit::wave_vcc_on`]):
//! then `structure_wide` keeps every such branch for the wave and the lowering's `wave_wide`
//! decides it, through the parts of this child of the lowering. Without the switch the branch
//! is read per lane, as on `world`.
//!
//! Not exact even with the switch (Sol's third review, item 1): a vote of the threads there
//! misses VCC bits of lanes an earlier per-lane branch sent elsewhere and bits a scalar write
//! sets for lanes the wave does not have; pixel, vertex and mesh programs and each half-wave
//! on its own vote over the host's wave, not the console's ([`Lower::host_wave_wide`]). The
//! exchange of two 32-lane halves is declined for these branches ([`Lower::vcc_declined`]):
//! its barriers need every thread of the group in the same sequence of votes (item 2).

use super::{Attr, Lower, V, Why, flags};
use crate::structure::Bool;

impl Lower<'_> {
    /// A wave-wide branch in the meeting form: the wave's vote (`meet_wide`) only where it is
    /// asked for (`VICTORY_WAVE_MEET_WIDE=1`: group barriers inside branches and loops).
    pub(super) fn met_wide(&mut self, b: usize, when: Bool, any: bool) -> Result<V, Why> {
        const WHY: Why = "a wave-wide branch where not every lane runs";
        if !crate::emit::meet_wide_on() {
            return Err(WHY);
        }
        self.meet_wide(b, when, any).map_err(|w| if w == super::meet::NOT_EVERY { WHY } else { w })
    }

    /// Whether block `b`'s wave-wide branch is a VCC branch the switch made so, whose
    /// exchange between two host waves is declined.
    pub(super) fn vcc_declined(&self, b: usize) -> bool {
        crate::emit::wave_vcc_on()
            && self.x.p.blocks[b].insts.last().is_some_and(|i| matches!(i.name(), "s_cbranch_vccz" | "s_cbranch_vccnz"))
    }

    /// Whether a wave-wide branch is decided by [`Self::host_wave_wide`]: with
    /// `VICTORY_WAVE_VCC=1`, where no form has the console's wave to ask.
    pub(super) fn by_host_wave(&self) -> bool {
        crate::emit::wave_vcc_on() && (self.s.threads.is_none() || self.local || self.mask_data == super::MaskData::None)
    }

    /// A wave-wide branch as the host's wave answers it, for the stages whose host waves are
    /// not declared the console's (pixel, vertex, mesh programs: `MaskData::Host`, `None`)
    /// and for each half-wave on its own (`with_local_waves`): some lane's `when` (`any`), or
    /// every lane's.
    /// shortcut: right where VCC is alike in the console's wave or the host's wave is the
    /// console's; otherwise each host wave decides apart. Only with `VICTORY_WAVE_VCC=1`.
    pub(super) fn host_wave_wide(&mut self, when: Bool, any: bool) -> Result<V, Why> {
        let c = self.cur;
        let mine = self.boolean(when, c)?;
        let i1t = self.s.t.i1;
        self.s.flags |= flags::WAVE_OPS;
        let (name, code) = if any { ("waveAnyTrue", 113) } else { ("waveAllTrue", 114) };
        Ok(self.s.op(c, name, Attr::NoUnwind, i1t, code, &[mine]))
    }
}
