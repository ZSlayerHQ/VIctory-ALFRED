//! Lane masks read as data where a console wave may be partial: a child of the lowering.
//! A mask's bits come back as a ballot of the lanes' own bits (`Lower::mask_dword`), and a
//! partial wave (the last of a group whose threads are no whole number of waves) has no
//! lanes for the bits above its threads. For EXEC and compare results those bits are clear
//! on the console too; a number held as a mask (a pair restored from EXEC on one way and from
//! scalars on another, -1 included), a complement (`s_not_b64` of EXEC) or another constant
//! can have them set. `Structured::bounded` names the dwords that cannot (Sol's third
//! review, item 3). Off by default: with
//! `VICTORY_WAVE_GUARDS=1`, and with `VICTORY_WAVE_WEBS=1`, whose webs need it
//! ([`crate::emit::mask_bounds_on`]).

use super::Lower;
use crate::ir::Value;
use crate::native::StageIo;

impl Lower<'_> {
    /// Whether the mask dword `v` read as data may have bits above the wave's threads that a
    /// ballot loses: a wave of the group may be partial (any stage but a compute group of
    /// whole console waves) and `v` is not one of `Structured::bounded`. Never without
    /// `VICTORY_WAVE_GUARDS=1` or `VICTORY_WAVE_WEBS=1` ([`crate::emit::mask_bounds_on`]):
    /// every mask dword is then the ballot of the lanes that are there, as on `world`.
    pub(super) fn above_threads(&self, v: Value) -> bool {
        if !crate::emit::mask_bounds_on() {
            return false;
        }
        let lanes = if self.x.native.wave64 { 64 } else { 32 };
        // The console's group, not the host's: the paired form gives a host thread two
        // lanes, so its group (`s.threads`) is half the console's, and a group of an odd
        // number of console waves is no whole number of 64 there.
        let threads = match self.x.native.io {
            StageIo::Cs(cs) => Some(cs.threads),
            _ => self.s.threads,
        };
        let group = threads.and_then(|t| t.iter().try_fold(1u32, |n, &k| n.checked_mul(k)));
        !group.is_some_and(|n| n % lanes == 0) && !self.x.st.bounded.contains(&v)
    }
}
