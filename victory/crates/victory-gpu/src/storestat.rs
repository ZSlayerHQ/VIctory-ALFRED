//! Stores into title memory the kernel refused (`crate::host::write_mapped`):
//! counted in every build, and the first of a run says where, so a release
//! run that drops landed bytes is not silent. Diagnostics' copy line reads
//! the same counters ([`crate::copystat`]), and the live stats file
//! ([`crate::host::stats`]) has them in every build.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Pieces dropped and their bytes, since the last [`take`]. A test build without diagnostics neither counts into
/// them (a fixture reads its own thread's counts) nor takes them, so they are left out of it.
#[cfg(any(feature = "diagnostics", not(test)))]
static FAILED: AtomicU64 = AtomicU64::new(0);
#[cfg(any(feature = "diagnostics", not(test)))]
static FAILED_BYTES: AtomicU64 = AtomicU64::new(0);

/// Pieces dropped and their bytes since the start, never taken: the live stats file's
/// `stores_failed_total` (the counters above are zeroed by the copy line in a diagnostics
/// build and never in another, so one name read from them meant two things).
static TOTAL: AtomicU64 = AtomicU64::new(0);
static TOTAL_BYTES: AtomicU64 = AtomicU64::new(0);

// Tests: the counters a fixture reads, per thread. The suite runs tests in
// parallel, and another test's refused store would land in a process-wide
// counter, so a fixture reads what its own thread counted.
#[cfg(test)]
thread_local! {
    static TEST: [std::cell::Cell<u64>; 2] = const { [const { std::cell::Cell::new(0) }; 2] };
}

/// A store of `bytes` into title memory was refused by the kernel and
/// dropped, not written (`crate::host::write_mapped`): counted, and the
/// first in the run says the offset in its page. Not the address: title
/// addresses are the process's memory, and the first is enough to say
/// where to look.
pub(crate) fn failed(at: usize, bytes: usize) {
    #[cfg(not(test))]
    {
        FAILED.fetch_add(1, Relaxed);
        FAILED_BYTES.fetch_add(bytes as u64, Relaxed);
        TOTAL.fetch_add(1, Relaxed);
        TOTAL_BYTES.fetch_add(bytes as u64, Relaxed);
    }
    #[cfg(test)]
    TEST.with(|counts| {
        counts[0].set(counts[0].get().wrapping_add(1));
        counts[1].set(counts[1].get().wrapping_add(bytes as u64));
    });
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SAID.swap(true, Relaxed) {
        tracing::warn!(
            page_offset = format_args!("{:#x}", at & 0xfff),
            "a kernel store into title memory failed and its bytes were dropped: later failures are counted"
        );
    }
}

/// Pieces dropped and their bytes since the start, in every build.
pub(crate) fn total() -> (u64, u64) {
    (TOTAL.load(Relaxed), TOTAL_BYTES.load(Relaxed))
}

/// The counters since the last copy line, taken: the copy line's (diagnostics has the line).
#[cfg(feature = "diagnostics")]
pub(crate) fn take() -> (u64, u64) {
    (FAILED.swap(0, Relaxed), FAILED_BYTES.swap(0, Relaxed))
}

/// Tests: the pieces and bytes this thread counted, for the fixture that
/// fails a store.
#[cfg(test)]
pub(crate) fn counted() -> (u64, u64) {
    TEST.with(|counts| (counts[0].get(), counts[1].get()))
}
