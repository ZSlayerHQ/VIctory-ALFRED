//! The PC-side half of a landing (`crate::host`'s relay thread), and the
//! switches that change how a landing goes: `VICTORY_LAND_SCAN=1` looks
//! the copies in the PC's memory through while the fetch from the card's
//! memory is in flight.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::shadow::Shadow;

use super::Declared;

/// Whether the copies in the PC's memory are looked through while the
/// card fetch is in flight (`VICTORY_LAND_SCAN=1`; off by default).
///
/// Without it `Host::relay_land` fetches what the card wrote to copies in
/// its own memory first, waits for it, merges it, and only then compares
/// the copies in the PC's memory ([`crate::shadow::find_all`]) and merges
/// those. The compare needs nothing the fetch brings, and the fetch spends
/// most of its time waiting for the card: with the switch on the compare
/// runs in that wait and its result is merged after both are done. The
/// result is the same: the compare reads without the snapshot's lock and
/// the merge looks again under it, so a piece that changes during the
/// fetch is found by this landing or by the later flight that wrote it.
pub(crate) fn scan_beside() -> bool {
    crate::live::land_scan()
}

/// Whether a merge compares 32 bytes at a time and stores only the 4-byte
/// words that differ (`VICTORY_EXACT_BACK=1`; off by default).
///
/// Without it a merge reads title memory from the first byte the card
/// changed to the last, puts the card's bytes over it and stores that whole
/// stretch back, and the relay's merge of a copy in the PC's memory
/// compares the piece three times over (to find its first difference, its
/// last, and to merge). With it one pass over 32 bytes finds both ends and
/// the word runs between them, title memory is still read only over the
/// stretch, and only the words that differ are stored, so a title byte the
/// card did not write beside them is left as it is. The result is the same
/// as without it: only the stores cover less.
pub(crate) fn exact() -> bool {
    crate::live::exact_back()
}

/// The declared ranges of `flights`, each copy once: the ranges the
/// flights' shaders may write, joined per copy where they overlap or
/// touch.
pub(super) fn declared(flights: &[&super::Flight]) -> Vec<Declared> {
    let mut back: Vec<Declared> = Vec::new();
    for flight in flights {
        for (shadow, ranges) in &flight.back {
            match back.iter_mut().find(|(s, _)| Arc::ptr_eq(s, shadow)) {
                Some((_, all)) => {
                    for range in ranges {
                        if !all.contains(range) {
                            all.push(*range);
                        }
                    }
                }
                None => back.push((shadow.clone(), ranges.clone())),
            }
        }
    }
    back
}

/// [`crate::host`]'s relay, for the copies not in the card's memory: what
/// their shaders wrote goes into title memory, each byte once, the pieces
/// that differ found on several threads. In two steps: [`look`] compares
/// the copies with their snapshots ([`crate::shadow::find_all`]), which
/// needs nothing of the card's fetch, and [`merge`] puts what differs into
/// title memory. With [`scan_beside`] the first step runs on the landing
/// thread while the fetch is in flight; without it [`merge`] runs it just
/// before merging, as the landing did before the switch.
///
/// [`look`]: RelayBack::look
/// [`merge`]: RelayBack::merge
pub(crate) struct RelayBack {
    /// Each copy and a range of it declared and not done with.
    work: Vec<(Arc<Shadow>, usize, usize)>,
    /// The 64 KiB pieces of each of `work` that differ, once looked for.
    found: Option<Vec<Vec<(usize, usize)>>>,
    /// The exact-merge mode this landing started under ([`exact`]): one landing keeps it
    /// from its first piece to its last, whatever the control file says meanwhile.
    exact: bool,
}

impl RelayBack {
    /// The work of landing `back`: for each copy not in the card's memory,
    /// its declared ranges joined, counted as looked through
    /// ([`Shadow::relay_compared`]), as the landing did before.
    pub(crate) fn new(back: &[Declared], exact: bool) -> Self {
        let mut work: Vec<(Arc<Shadow>, usize, usize)> = Vec::new();
        for (shadow, declared) in back {
            if shadow.carded() {
                continue;
            }
            let mut spans: Vec<(usize, usize)> = Vec::new();
            super::join_ranges(&mut spans, declared.iter().copied());
            for &(from, to) in &spans {
                shadow.relay_compared(from, to - from);
            }
            work.extend(spans.into_iter().map(|(from, to)| (shadow.clone(), from, to)));
        }
        Self { work, found: None, exact }
    }

    /// Looks the ranges through, once: the pieces that differ from their
    /// snapshots. Its time is the PC-side scan. May run while the card
    /// fetch is in flight ([`scan_beside`]); the copies are read without
    /// the snapshot's lock, and [`merge`](RelayBack::merge) looks again
    /// under it.
    pub(crate) fn look(&mut self) {
        if self.work.is_empty() || self.found.is_some() {
            return;
        }
        let scanning = std::time::Instant::now();
        self.found = Some(crate::shadow::find_all(&self.work));
        let us = scanning.elapsed().as_micros() as u64;
        super::RELAY_STATS[12].fetch_add(us, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackScanUs, us);
    }

    /// Looks the ranges through if that has not been done ([`look`]) and
    /// puts what differs into title memory, serially. Its time, without
    /// the look, is the PC-side merge. Counts the bytes stored into title
    /// memory (`COPY_BACK[1]`), the same figure in both modes of
    /// `VICTORY_EXACT_BACK`.
    pub(crate) fn merge(mut self) {
        self.look();
        let Some(found) = self.found.take() else {
            return;
        };
        let merging = std::time::Instant::now();
        let (mut now, mut title) = (Vec::new(), Vec::new());
        for ((shadow, from, to), found) in self.work.iter().zip(found) {
            let mut written = 0;
            for (at, n) in found {
                written += shadow.relay_merge(at, n, &mut now, &mut title, self.exact);
            }
            super::COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(*from, to - *from, written);
            let _ = (from, to);
        }
        super::RELAY_STATS[10].fetch_add(merging.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}

#[cfg(test)]
#[path = "tests/exact.rs"]
mod exact_tests;

#[cfg(test)]
#[path = "tests/land.rs"]
mod land_tests;
