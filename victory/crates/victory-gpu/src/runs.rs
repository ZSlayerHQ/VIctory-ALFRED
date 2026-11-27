//! Diagnostics, with `VICTORY_GPU_STAMPS`: a census of runs of clamped
//! indirect calls. A run is the clamped calls of one host list (a title
//! list, or a part of one cut at a wait) with nothing between them that
//! can write buffer memory: one clamp pass placed before a run's first call
//! would clamp exactly what the calls' own clamps clamp now. Counted only;
//! nothing recorded or sent changes.
//!
//! What ends a run, after its last call:
//! - (a) a compute dispatch of the title's (`State::dispatch`); an indirect
//!   dispatch is a call of its run, and ends it;
//! - (b) a copy into a buffer: the title's buffer copies (`copy_buffer_region`,
//!   `copy_resource` of buffers, `copy_memory`), its texture-to-buffer read
//!   backs and its query resolves (`crate::list`); there is no clear of a
//!   buffer among the title's methods;
//! - (c) a draw or call whose pipeline has a stage whose code can store to
//!   buffer memory (`Translated::stores`, the test the runtime makes for
//!   copying shader writes back, `crate::draw`): a store goes through a
//!   view bound for writing, so this is every draw that can write a buffer,
//!   taken from the translated code instead of the views bound; an indirect
//!   call that can store is a call of its run, and ends it;
//! - (d) a refresh or copy of ours into an argument buffer's host copy in
//!   the middle of the list: none is ever recorded into a list (refreshes
//!   and title copies are done on the CPU when the list is submitted,
//!   before any of it runs, `HostList::run_deferred`; the card side of a
//!   copy in the card's memory is filled by a list of its own, before the
//!   submission's lists, `Host::card_before`), so the line says 0;
//! - (e) a cut of the host list or its end (`HostList::close`).
//!
//! (f): calls whose arguments are in another host copy than the run's first
//! call's are counted, and end nothing (one pass can clamp records of
//! several buffers). The runs there would be with (c) ignored are counted
//! too.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Whether the census counts: with `VICTORY_GPU_STAMPS` (and in tests).
pub(crate) fn on() -> bool {
    cfg!(test) || crate::stamps::on()
}

/// What ended a run (the index of its count).
#[derive(Clone, Copy)]
enum End {
    Dispatch = 0,
    Copy = 1,
    Stores = 2,
    Close = 3,
}

/// The counts, in [`Runs::tally`] and in [`TOTAL`]: clamped calls, runs,
/// runs by length ([`BUCKETS`]), runs by what ended them ([`End`]), calls
/// with another argument buffer than their run's first, runs with (c)
/// ignored.
const CALLS: usize = 0;
const RUNS: usize = 1;
const LENGTHS: usize = 2;
const BUCKETS: [u32; 5] = [2, 4, 8, 16, 64];
const ENDED: usize = LENGTHS + BUCKETS.len() + 1;
const OTHER_BUFFER: usize = ENDED + 4;
const RUNS_IGNORING_STORES: usize = OTHER_BUFFER + 1;
pub(crate) const COUNTS: usize = RUNS_IGNORING_STORES + 1;

/// Since the last frame line, all host lists.
static TOTAL: [AtomicU64; COUNTS] = [const { AtomicU64::new(0) }; COUNTS];

/// One host list's runs.
#[derive(Default)]
pub(crate) struct Runs {
    /// The run open: its calls, and the host copy its first call reads.
    open: Option<(u32, usize)>,
    /// Whether a run is open with (c) ignored.
    open_ignoring_stores: bool,
    /// The draw or call bound last can store to buffer memory, and nothing
    /// has been counted since.
    stores: bool,
    /// What this list counted ([`TOTAL`]'s indices).
    pub(crate) tally: [u64; COUNTS],
}

impl Runs {
    fn count(&mut self, at: usize, n: u64) {
        self.tally[at] += n;
        TOTAL[at].fetch_add(n, Relaxed);
    }

    /// A clamped indirect call whose arguments are in host copy `buffer`;
    /// `dispatch`: a compute dispatch, which ends its run.
    pub(crate) fn call(&mut self, buffer: usize, dispatch: bool) {
        self.stored();
        self.count(CALLS, 1);
        match &mut self.open {
            Some((calls, first)) => {
                *calls += 1;
                if *first != buffer {
                    self.count(OTHER_BUFFER, 1);
                }
            }
            None => self.open = Some((1, buffer)),
        }
        self.open_ignoring_stores = true;
        if dispatch {
            self.end(End::Dispatch);
        }
    }

    /// A draw or dispatch was bound; `stores`: its pipeline can store to
    /// buffer memory.
    pub(crate) fn bound(&mut self, stores: bool) {
        self.stored();
        self.stores = stores;
    }

    /// A compute dispatch of the title's was recorded (its own stores are
    /// the dispatch's: (a), not (c)).
    pub(crate) fn dispatch(&mut self) {
        self.stores = false;
        self.end(End::Dispatch);
    }

    /// A copy, read back or resolve into a buffer was recorded.
    pub(crate) fn copy(&mut self) {
        self.stored();
        self.end(End::Copy);
    }

    /// The host list is closed: cut, or at its end.
    pub(crate) fn close(&mut self) {
        self.stored();
        self.end(End::Close);
    }

    /// What the last draw or call bound may have stored ends the run.
    fn stored(&mut self) {
        if std::mem::take(&mut self.stores)
            && let Some((calls, _)) = self.open.take()
        {
            self.ended(calls, End::Stores);
        }
    }

    fn end(&mut self, why: End) {
        if let Some((calls, _)) = self.open.take() {
            self.ended(calls, why);
        }
        if std::mem::take(&mut self.open_ignoring_stores) {
            self.count(RUNS_IGNORING_STORES, 1);
        }
    }

    fn ended(&mut self, calls: u32, why: End) {
        self.count(RUNS, 1);
        let bucket = BUCKETS.iter().position(|&least| calls < least).unwrap_or(BUCKETS.len());
        self.count(LENGTHS + bucket, 1);
        self.count(ENDED + why as usize, 1);
    }
}

/// The frame line, the counts since the last one taken.
pub(crate) fn line(frames: f64) {
    let c: Vec<f64> = TOTAL.iter().map(|t| t.swap(0, Relaxed) as f64 / frames.max(1.0)).collect();
    let l = &c[LENGTHS..ENDED];
    let e = &c[ENDED..OTHER_BUFFER];
    tracing::info!(
        "clamped indirect runs per frame: {:.0} clamped calls in {:.1} runs; runs of 1 call {:.1}, 2-3 {:.1}, 4-7 {:.1}, 8-15 {:.1}, \
         16-63 {:.1}, 64 or more {:.1}; ended by (a) a title dispatch {:.1}, (b) a copy, read back or resolve into a buffer {:.1}, \
         (c) a draw or call that can store to buffer memory {:.1}, (d) a refresh or copy of ours mid-list 0, (e) a cut or the list's end {:.1}; \
         (f) calls reading another argument buffer than their run's first {:.1}; {:.1} runs with (c) ignored",
        c[CALLS],
        c[RUNS],
        l[0],
        l[1],
        l[2],
        l[3],
        l[4],
        l[5],
        e[End::Dispatch as usize],
        e[End::Copy as usize],
        e[End::Stores as usize],
        e[End::Close as usize],
        c[OTHER_BUFFER],
        c[RUNS_IGNORING_STORES]
    );
}

/// What a tally says, by name (tests).
#[cfg(test)]
pub(crate) fn named(tally: &[u64; COUNTS]) -> [(&'static str, u64); 9] {
    [
        ("calls", tally[CALLS]),
        ("runs", tally[RUNS]),
        ("runs of 1", tally[LENGTHS]),
        ("runs of 2-3", tally[LENGTHS + 1]),
        ("ended by a dispatch", tally[ENDED + End::Dispatch as usize]),
        ("ended by a copy", tally[ENDED + End::Copy as usize]),
        ("ended by a store", tally[ENDED + End::Stores as usize]),
        ("ended by the list's end", tally[ENDED + End::Close as usize]),
        ("other buffer", tally[OTHER_BUFFER]),
    ]
}

/// The runs with (c) ignored (tests).
#[cfg(test)]
pub(crate) fn ignoring_stores(tally: &[u64; COUNTS]) -> u64 {
    tally[RUNS_IGNORING_STORES]
}
