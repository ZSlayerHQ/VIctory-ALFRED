//! Renderer port 1, the GPU side's half: absent from this tree.
//!
//! The private build has a module here that, with `VICTORY_PORT_CONSTANTS=1`,
//! reads the title's 32-bit graphics heap in place instead of copying it,
//! told where that heap's pieces are from what the title's code is. The
//! public tree leaves it out, and the switch is refused (`victory-hle`'s
//! `xmem` says so once). What is here only gives the callers what they need,
//! and does nothing: nothing is placed by it, so the base's own `inplace`
//! path (or a copy) places every buffer, and its notes and log are empty.

use crate::inplace::Placed;

/// A buffer of `_len` bytes at `_start` read in place: never by this
/// module; the caller goes on to `crate::inplace::place_for`.
pub(crate) fn place_for(_start: u64, _len: u64, _window: bool) -> Option<Placed> {
    None
}

/// A batched copy of the title's from `_src` to `_dst`: nothing to note.
pub(crate) fn copied(_dst: usize, _src: usize, _len: usize) {}

/// A batched fill of the title's at `_src`: nothing to note.
pub(crate) fn filled(_src: usize, _len: usize) {}

/// Title memory `[_at, _at + _len)` is let go: nothing of this module's there.
pub(crate) fn forget(_at: usize, _len: usize) {}

/// The frame log's line for it: none.
pub(crate) fn log(_frames: f64) {}
