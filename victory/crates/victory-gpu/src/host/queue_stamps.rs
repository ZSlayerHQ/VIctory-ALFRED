//! Diagnostics: GPU timestamps by queue (`VICTORY_GPU_STAMPS=1`).
//!
//! The host sends lists to two queues: the drawing queue, whose pairs `crate::stamps` reads
//! into windows of that queue's clock, and the copy queue of `VICTORY_FETCH_QUEUE`
//! (`super::fetch`), whose only lists are the relay's fetches of card copies' pieces. A fetch
//! on the copy queue gets a pair of its own here ([`Pair`]): a timestamp before its first copy
//! and one after its last, on the copy queue's clock, read when the fetch's ticket is reached.
//!
//! Each queue is reported by itself ([`line`], [`json`]): a span on the copy queue runs while
//! the drawing queue runs other work, so the two are never added into one busy figure, and
//! the copy queue's time is not taken out of the drawing queue's uncovered time.
//!
//! What is not stamped is said in a field of its own ([`not_stamped`]), by queue and kind of
//! list. As long as that field is not empty, time no pair covers on a queue is "uncovered",
//! never "idle".

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

/// The copy queue since the last [`take`]: fetch lists sent, lists whose pair was read,
/// nanoseconds between the pairs' two timestamps, pairs that could not be read.
static COPY: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Whether the copy queue's fetch lists are stamped: 0 not asked yet, 1 yes, 2 no (the
/// adapter has no timestamps on copy queues), 3 no (it has, and the pair could not be made).
static SUPPORTED: AtomicU64 = AtomicU64::new(0);

/// The pair of a copy-queue fetch list: its query heap and where the two values are read.
pub(super) struct Pair {
    heap: ID3D12QueryHeap,
    back: ID3D12Resource,
    /// Where the two values are mapped.
    data: usize,
}

impl Pair {
    /// A pair for the copy queue's list, if the stamps are on and the adapter has timestamps
    /// on copy queues (`D3D12_FEATURE_DATA_D3D12_OPTIONS3`); none otherwise, and
    /// [`not_stamped`] says so.
    pub(super) fn new(device: &ID3D12Device) -> Option<Pair> {
        if !crate::stamps::on() {
            return None;
        }
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
        // SAFETY: a plain query into a structure of the size said.
        let asked = unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS3,
                std::ptr::from_mut(&mut options).cast(),
                size_of_val(&options) as u32,
            )
        };
        let supported = asked.is_ok() && options.CopyQueueTimestampQueriesSupported.as_bool();
        SUPPORTED.store(if supported { 1 } else { 2 }, Relaxed);
        if !supported {
            return None;
        }
        let mut heap: Option<ID3D12QueryHeap> = None;
        let mut back: Option<ID3D12Resource> = None;
        let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 16,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        // SAFETY: plain object creation; read-back memory may stay mapped while it lives.
        unsafe {
            let made = (|| {
                let heap_desc = D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP, Count: 2, NodeMask: 0 };
                device.CreateQueryHeap(&heap_desc, &mut heap).ok()?;
                device
                    .CreateCommittedResource(&readback, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back)
                    .ok()?;
                let back = back?;
                let mut data = std::ptr::null_mut();
                back.Map(0, None, Some(&mut data)).ok()?;
                Some(Pair { heap: heap?, back, data: data as usize })
            })();
            // Not silent: the lists then run without a pair, and the report says why.
            if made.is_none() {
                SUPPORTED.store(3, Relaxed);
            }
            made
        }
    }

    /// The pair's first timestamp, before the list's copies.
    pub(super) fn begin(&self, list: &ID3D12GraphicsCommandList) {
        // SAFETY: an open copy list and our query heap.
        unsafe { list.EndQuery(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, 0) };
    }

    /// The pair's second timestamp, after the list's copies, and their resolve.
    pub(super) fn end(&self, list: &ID3D12GraphicsCommandList) {
        // SAFETY: an open copy list, our query heap and our read-back buffer.
        unsafe {
            list.EndQuery(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            list.ResolveQueryData(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.back, 0);
        }
    }

    /// The list has run (its ticket was reached on a live device): the nanoseconds between
    /// its two timestamps on `queue`'s clock; none if the clock or the pair is unusable.
    pub(super) fn read(&self, queue: &ID3D12CommandQueue) -> Option<u64> {
        // SAFETY: a plain query; the buffer is mapped and the card has written both values.
        let (frequency, from, to) = unsafe {
            let data = self.data as *const u64;
            (queue.GetTimestampFrequency().ok()?, data.read(), data.add(1).read())
        };
        (frequency > 0 && to >= from).then(|| ((to - from) as u128 * 1_000_000_000 / u128::from(frequency)) as u64)
    }
}

/// A fetch list went to the copy queue and ran; `span` is its pair's reading, if it had one
/// ([`Pair::read`]).
pub(super) fn copy_fetched(pair: bool, span: Option<u64>) {
    COPY[0].fetch_add(1, Relaxed);
    match span {
        Some(ns) => {
            COPY[1].fetch_add(1, Relaxed);
            COPY[2].fetch_add(ns, Relaxed);
        }
        None if pair => {
            COPY[3].fetch_add(1, Relaxed);
        }
        None => {}
    }
}

/// The copy queue's figures since the last take.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Copy {
    /// Fetch lists that ran on the copy queue.
    pub(crate) lists: u64,
    /// Of them, lists whose pair was read.
    pub(crate) stamped: u64,
    /// Nanoseconds between those pairs' timestamps, on the copy queue's clock.
    pub(crate) ns: u64,
    /// Pairs recorded and not readable.
    pub(crate) unread: u64,
}

/// Takes the copy queue's figures.
pub(crate) fn take() -> Copy {
    let [lists, stamped, ns, unread] = [0, 1, 2, 3].map(|i| COPY[i].swap(0, Relaxed));
    Copy { lists, stamped, ns, unread }
}

/// What is not stamped, by queue and kind of list. `on`: whether the stamps are on at all;
/// `copy_queue`: whether the adapter has timestamps on copy queues (none: not asked yet,
/// because no copy-queue list has been made).
pub(crate) fn not_stamped_with(on: bool, copy_queue: Option<bool>) -> Vec<&'static str> {
    if !on {
        return vec!["drawing queue: every list (VICTORY_GPU_STAMPS is off)", "copy queue: every list (VICTORY_GPU_STAMPS is off)"];
    }
    let mut kinds = vec![
        "drawing queue: the resolves of the stamps themselves",
        "drawing queue: lists of ours run under the submission lock",
        "drawing queue: our indirect, in-place and tile work lists",
        "drawing queue: the hash list a synchronous submission without a title list sends under VICTORY_GPU_TIMING",
        "drawing queue: the hash check's own fetch lists (VICTORY_HASH_CHECK=1)",
    ];
    if copy_queue == Some(false) {
        kinds.push("copy queue: the fetch lists (the adapter has no timestamps on copy queues)");
    }
    kinds
}

/// [`not_stamped_with`] by the copy queue's state: 0 not asked yet, 1 stamped, 2 the adapter
/// has no timestamps on copy queues, 3 it has and the pair could not be made.
pub(crate) fn not_stamped_state(on: bool, state: u64) -> Vec<&'static str> {
    let mut kinds = not_stamped_with(on, match state {
        0 => None,
        2 => Some(false),
        _ => Some(true),
    });
    if on && state == 3 {
        kinds.push("copy queue: the fetch lists (their pair of timestamps could not be made)");
    }
    kinds
}

/// [`not_stamped_with`], for this run.
pub(crate) fn not_stamped() -> Vec<&'static str> {
    not_stamped_state(crate::stamps::on(), SUPPORTED.load(Relaxed))
}

/// Why no figure here is called idle, said in the object's `idle` field: a queue's spans are
/// durations on that queue's own clock (its frequency, with no `GetClockCalibration` to a
/// common time base), so the two queues' spans cannot be laid on one time line, and lists
/// without stamps run in what no span covers. Until both are mended, time no pair covers is
/// "uncovered" and idle time is not a figure.
const IDLE: &str = "not derivable: the queues' spans are on separate clocks with no calibration to a common time base, and some lists carry no stamps";

/// The stats file's `gpu_stamps` object: the copy queue per frame over `frames`, and what is
/// not stamped.
pub(crate) fn json(copy: &Copy, frames: f64, not_stamped: &[&str]) -> String {
    let per = frames.max(1.0);
    let kinds: Vec<String> = not_stamped.iter().map(|kind| format!("\"{kind}\"")).collect();
    format!(
        "{{\"copy_queue\":{{\"fetch_lists\":{:.2},\"stamped\":{:.2},\"ms\":{:.3},\"pairs_unread\":{:.2}}},\"not_stamped\":[{}],\
         \"idle\":\"{IDLE}\"}}",
        copy.lists as f64 / per,
        copy.stamped as f64 / per,
        copy.ns as f64 / 1e6 / per,
        copy.unread as f64 / per,
        kinds.join(",")
    )
}

/// The log's line, once a second with the drawing queue's.
pub(crate) fn text(copy: &Copy, frames: f64, not_stamped: &[&str]) -> String {
    let per = frames.max(1.0);
    format!(
        "card queues by GPU timestamps, each queue by itself (VICTORY_GPU_STAMPS=1): copy queue per frame: {:.2} fetch lists, \
         {:.2} of them stamped, {:.3} ms between their first and last copy on the copy queue's clock ({:.2} pairs unread); \
         these spans overlap the drawing queue's and are not added to it or taken from its uncovered time; not stamped: {}",
        copy.lists as f64 / per,
        copy.stamped as f64 / per,
        copy.ns as f64 / 1e6 / per,
        copy.unread as f64 / per,
        if not_stamped.is_empty() { "nothing".to_owned() } else { not_stamped.join("; ") }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stamp read on the copy queue appears in the report, under the copy queue's own name,
    /// per frame; and the field of what is not stamped names queue and kind.
    #[test]
    fn a_copy_queue_stamp_appears_in_the_report() {
        let copy = Copy { lists: 4, stamped: 3, ns: 6_000_000, unread: 1 };
        let kinds = not_stamped_with(true, Some(true));
        let object = json(&copy, 2.0, &kinds);
        assert!(
            object.starts_with("{\"copy_queue\":{\"fetch_lists\":2.00,\"stamped\":1.50,\"ms\":3.000,\"pairs_unread\":0.50},\"not_stamped\":[\""),
            "{object}"
        );
        assert!(!object.contains("copy queue: the fetch lists"), "a stamped copy queue listed as not stamped: {object}");
        let line = text(&copy, 2.0, &kinds);
        assert!(line.contains("copy queue per frame: 2.00 fetch lists, 1.50 of them stamped, 3.000 ms"), "{line}");
        assert!(!line.contains("idle"), "{line}");
    }

    /// A pair that could not be made on an adapter that has timestamps on copy queues is not
    /// silent: the fetch lists are named as not stamped, with that reason.
    #[test]
    fn a_pair_that_could_not_be_made_is_named() {
        let kinds = not_stamped_state(true, 3);
        assert!(kinds.contains(&"copy queue: the fetch lists (their pair of timestamps could not be made)"), "{kinds:?}");
        assert!(!not_stamped_state(true, 1).iter().any(|kind| kind.starts_with("copy queue: ")), "a stamped copy queue is named");
    }

    /// No figure is called idle: the object says in a field of its own that idle time cannot
    /// be worked out from these stamps, and why.
    #[test]
    fn idle_is_said_not_to_be_derivable() {
        let object = json(&Copy::default(), 1.0, &not_stamped_with(true, Some(true)));
        assert!(
            object.contains("\"idle\":\"not derivable: the queues' spans are on separate clocks with no calibration to a common time base, and some lists carry no stamps\""),
            "{object}"
        );
    }

    /// Where the adapter has no timestamps on copy queues, or the stamps are off, the field
    /// says exactly which queue and which lists are not stamped.
    #[test]
    fn what_is_not_stamped_is_named_by_queue_and_kind() {
        let without = not_stamped_with(true, Some(false));
        assert!(without.contains(&"copy queue: the fetch lists (the adapter has no timestamps on copy queues)"), "{without:?}");
        assert!(without.iter().all(|kind| kind.starts_with("drawing queue: ") || kind.starts_with("copy queue: ")), "{without:?}");
        let off = not_stamped_with(false, None);
        assert_eq!(off.len(), 2, "{off:?}");
        assert!(off.iter().all(|kind| kind.contains("every list")), "{off:?}");
    }
}
