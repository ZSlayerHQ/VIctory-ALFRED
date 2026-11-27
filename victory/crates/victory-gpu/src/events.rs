//! Debug events on the lists of our own (`BeginEvent`/`EndEvent`, the PIX-style markers that
//! PIX, RenderDoc and the vendors' GPU traces show), in a diagnostics build.
//!
//! The judges of the card's time are those tools, not our stamps; they can only name what a
//! list names itself. Our own lists and the groups of ours inside the title's lists are
//! named here, each kind by one stable text, so that an outside trace shows the same kinds
//! `crate::stamps` counts. The title's own events are not forwarded (its lists' markers are
//! ignored where they arrive), so its passes are unnamed to an outside tool: a job of its own.
//!
//! An event is two commands in a list and nothing else: no allocation, no lock, no wait. A
//! build without diagnostics records none.

use windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList;
use windows::core::{PCWSTR, w};

/// Our list that sends copies in the card's memory up and hashes them.
pub(crate) const HASH: PCWSTR = w!("victory: hash card copies");
/// Our compare of copies in the card's memory after a submission.
pub(crate) const COMPARE: PCWSTR = w!("victory: compare card copies");
/// The relay's fetch of card copies' pieces on the drawing queue.
pub(crate) const RELAY_FETCH: PCWSTR = w!("victory: relay fetch");
/// The relay's fetch of card copies' pieces on the copy queue.
pub(crate) const COPY_QUEUE_FETCH: PCWSTR = w!("victory: copy queue fetch");
/// A synchronous submission's fetch of card copies' pieces.
pub(crate) const FETCH: PCWSTR = w!("victory: synchronous fetch");
/// A host list's prelude.
pub(crate) const PRELUDE: PCWSTR = w!("victory: prelude");
/// A clamp-and-barrier group of ours inside a title list.
pub(crate) const CLAMP: PCWSTR = w!("victory: clamp group");
/// The copy of a frame to the swap chain's buffer.
pub(crate) const PRESENT: PCWSTR = w!("victory: present copy");
/// Our copy of a texture's staging buffer into its host copy, inside a title list.
pub(crate) const UPLOAD: PCWSTR = w!("victory: texture upload");

/// The hash check's own fetch of pieces the copy-back did not fetch (`VICTORY_HASH_CHECK=1`).
pub(crate) const HASH_CHECK: PCWSTR = w!("victory: hash check fetch");

/// Every name, for the fixture.
#[cfg(test)]
pub(crate) const ALL: [PCWSTR; 10] = [HASH, COMPARE, RELAY_FETCH, COPY_QUEUE_FETCH, FETCH, PRELUDE, CLAMP, PRESENT, UPLOAD, HASH_CHECK];

/// Begins the event `name` in `list` (an open list); [`end`] ends it.
#[inline]
pub(crate) fn begin(list: &ID3D12GraphicsCommandList, name: PCWSTR) {
    #[cfg(feature = "diagnostics")]
    // SAFETY: an open list and a static NUL-terminated UTF-16 text, copied when recorded;
    // metadata 0 says the data is such a text, and the size is in bytes with its NUL.
    unsafe {
        let bytes = (name.as_wide().len() + 1) * 2;
        list.BeginEvent(0, Some(name.as_ptr().cast()), bytes as u32);
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = (list, name);
}

/// Ends the event begun last in `list`.
#[inline]
pub(crate) fn end(list: &ID3D12GraphicsCommandList) {
    #[cfg(feature = "diagnostics")]
    // SAFETY: an open list in which an event was begun.
    unsafe {
        list.EndEvent();
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = list;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The names are one a kind, all under one prefix an outside tool can filter on.
    #[test]
    fn the_names_are_distinct_and_prefixed() {
        // SAFETY: static NUL-terminated texts.
        let names: Vec<String> = ALL.iter().map(|name| unsafe { name.to_string() }.expect("UTF-16")).collect();
        for (i, name) in names.iter().enumerate() {
            assert!(name.starts_with("victory: "), "{name}");
            assert!(!names[..i].contains(name), "{name} twice");
        }
    }
}
