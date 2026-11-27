//! F5's ledger (`VICTORY_GUARD_LEDGER=1`; off by default, and then nothing
//! here is used): per native page and per address, the protection the title
//! asked for under each write guard we laid, and the guard's generation
//! (`jobs/2026-11-26-native-memory/f5-design.md` §1).
//!
//! Entries exist only while a guard is laid; elsewhere the host's
//! protection is the title's. What the host shows under a guard is
//! [`guarded`] of what the title asked: write rights taken away, execute,
//! no-access and the modifiers kept. A release restores what the title
//! asked, never `PAGE_READWRITE` by default, and only for an entry still
//! laid ([`Ledger::release`]); a free, decommit or remap drops the entries
//! without touching the host ([`Ledger::forget`]).
//!
//! Pure bookkeeping: the host calls are the caller's (`crate::watch`), made
//! under the same lock as these changes.

use std::collections::BTreeMap;

/// The kernel's page, the unit of protection.
pub(crate) const NATIVE: usize = 4096;

const PAGE_READONLY: u32 = 0x02;
const PAGE_READWRITE: u32 = 0x04;
const PAGE_WRITECOPY: u32 = 0x08;
const PAGE_EXECUTE_READ: u32 = 0x20;
const PAGE_EXECUTE_READWRITE: u32 = 0x40;
const PAGE_EXECUTE_WRITECOPY: u32 = 0x80;
/// `PAGE_GUARD`, `PAGE_NOCACHE`, `PAGE_WRITECOMBINE` and the rest above the
/// base protection: kept as they are under a guard.
const MODIFIERS: u32 = !0xff;

/// Whether the ledger is on (`VICTORY_GUARD_LEDGER=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GUARD_LEDGER").is_some_and(|v| v == "1"))
}

/// What the host shows under a write guard over a page the title asked
/// `requested` for: the same, without write rights (copy-on-write included:
/// the release puts it back, and the retried write makes the copy).
pub(crate) fn guarded(requested: u32) -> u32 {
    let base = match requested & 0xff {
        PAGE_READWRITE | PAGE_WRITECOPY => PAGE_READONLY,
        PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY => PAGE_EXECUTE_READ,
        other => other,
    };
    base | (requested & MODIFIERS)
}

/// Whether a write to a page of protection `protect` goes through (no
/// `PAGE_GUARD`: its first touch faults whatever the rights).
pub(crate) fn writable(protect: u32) -> bool {
    const PAGE_GUARD: u32 = 0x100;
    protect & PAGE_GUARD == 0
        && matches!(
            protect & 0xff,
            PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY
        )
}

/// Pages `[start, end)` (addresses, native-page aligned) under one guard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Run {
    pub end: usize,
    /// What the title asked for these pages.
    pub requested: u32,
    /// The guard's: given when it was laid, never again.
    pub generation: u64,
}

/// A piece of a run: `[start, end)` and what it held.
pub(crate) type Piece = (usize, Run);

#[derive(Default)]
pub(crate) struct Ledger {
    /// By start address; runs do not overlap.
    runs: BTreeMap<usize, Run>,
    /// The last generation given.
    generations: u64,
}

impl Ledger {
    pub(crate) const fn new() -> Ledger {
        Ledger {
            runs: BTreeMap::new(),
            generations: 0,
        }
    }

    /// Whether nothing is guarded.
    pub(crate) fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// The run holding address `at`: its start and itself.
    pub(crate) fn at(&self, at: usize) -> Option<Piece> {
        self.runs
            .range(..=at)
            .next_back()
            .filter(|(_, run)| at < run.end)
            .map(|(&start, &run)| (start, run))
    }

    /// The start of the first run after `at` (exclusive), if any.
    pub(crate) fn next_after(&self, at: usize) -> Option<usize> {
        self.runs.range(at.saturating_add(1)..).next().map(|(&start, _)| start)
    }

    /// Cuts the run holding `at` in two there, so that runs start or end at
    /// `at`.
    fn split(&mut self, at: usize) {
        if let Some((start, run)) = self.at(at)
            && start < at
        {
            self.runs.insert(start, Run { end: at, ..run });
            self.runs.insert(at, run);
        }
    }

    /// The runs inside `[start, end)`, cut to it, in order (and the map cut
    /// at both ends).
    fn inside(&mut self, start: usize, end: usize) -> Vec<Piece> {
        if start >= end {
            return Vec::new();
        }
        self.split(start);
        self.split(end);
        self.runs.range(start..end).map(|(&s, &r)| (s, r)).collect()
    }

    /// The parts of `[start, end)` no guard holds, in order.
    pub(crate) fn gaps(&self, start: usize, end: usize) -> Vec<(usize, usize)> {
        let mut gaps = Vec::new();
        let mut at = start;
        if let Some((_, run)) = self.at(start) {
            at = run.end;
        }
        for (&s, run) in self.runs.range(start..end) {
            if s > at {
                gaps.push((at, s));
            }
            at = at.max(run.end);
        }
        if at < end {
            gaps.push((at, end));
        }
        gaps
    }

    /// A new generation, for one laying.
    pub(crate) fn generation(&mut self) -> u64 {
        self.generations += 1;
        self.generations
    }

    /// A guard of `generation` laid over `[start, end)`, which no guard
    /// held, where the title asked for `requested`.
    pub(crate) fn lay(&mut self, start: usize, end: usize, requested: u32, generation: u64) {
        debug_assert!(self.gaps(start, end) == [(start, end)]);
        if start < end {
            self.runs.insert(
                start,
                Run {
                    end,
                    requested,
                    generation,
                },
            );
        }
    }

    /// The title asked for `requested` over `[start, end)`: the guarded
    /// pieces there take it (their guard stays). Returns them, for the
    /// caller to lay [`guarded`] of it over them.
    pub(crate) fn request(&mut self, start: usize, end: usize, requested: u32) -> Vec<Piece> {
        let pieces = self.inside(start, end);
        for &(s, _) in &pieces {
            if let Some(run) = self.runs.get_mut(&s) {
                run.requested = requested;
            }
        }
        pieces
    }

    /// The guards over `[start, end)` end: of `generation` only, or
    /// whichever is laid (`None`, a fault at the address now). Returns the
    /// pieces taken away, for the caller to restore what the title asked
    /// where the host still shows the guard.
    pub(crate) fn release(&mut self, start: usize, end: usize, generation: Option<u64>) -> Vec<Piece> {
        let pieces: Vec<Piece> = self
            .inside(start, end)
            .into_iter()
            .filter(|(_, run)| generation.is_none_or(|g| g == run.generation))
            .collect();
        for (s, _) in &pieces {
            self.runs.remove(s);
        }
        pieces
    }

    /// `[start, end)` was freed, decommitted or remapped: its guards are
    /// gone with it, nothing to restore.
    pub(crate) fn forget(&mut self, start: usize, end: usize) {
        for (s, _) in self.inside(start, end) {
            self.runs.remove(&s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: usize = NATIVE;
    const PAGE_NOACCESS: u32 = 0x01;
    const PAGE_EXECUTE: u32 = 0x10;

    #[test]
    fn a_guard_takes_write_rights_only() {
        assert_eq!(guarded(PAGE_READWRITE), PAGE_READONLY);
        assert_eq!(guarded(PAGE_WRITECOPY), PAGE_READONLY);
        assert_eq!(guarded(PAGE_EXECUTE_READWRITE), PAGE_EXECUTE_READ);
        assert_eq!(guarded(PAGE_EXECUTE_WRITECOPY), PAGE_EXECUTE_READ);
        for kept in [PAGE_NOACCESS, PAGE_READONLY, PAGE_EXECUTE, PAGE_EXECUTE_READ] {
            assert_eq!(guarded(kept), kept);
        }
        // Modifiers stay: write-combined memory stays write-combined.
        assert_eq!(guarded(PAGE_READWRITE | 0x400), PAGE_READONLY | 0x400);
        assert!(writable(PAGE_WRITECOPY) && !writable(PAGE_READWRITE | 0x100));
    }

    #[test]
    fn a_request_inside_a_guard_cuts_it_and_keeps_the_rest() {
        let mut l = Ledger::new();
        let g = l.generation();
        l.lay(0, 4 * P, PAGE_READWRITE, g);
        let pieces = l.request(P, 2 * P, PAGE_NOACCESS);
        assert_eq!(pieces.len(), 1);
        assert_eq!(l.at(P).unwrap().1.requested, PAGE_NOACCESS);
        assert_eq!(l.at(0).unwrap().1.requested, PAGE_READWRITE);
        assert_eq!(l.at(3 * P).unwrap().1.requested, PAGE_READWRITE);
        assert_eq!(l.gaps(0, 5 * P), [(4 * P, 5 * P)]);
    }

    #[test]
    fn a_release_of_another_generation_restores_nothing() {
        let mut l = Ledger::new();
        let old = l.generation();
        l.lay(0, 2 * P, PAGE_READWRITE, old);
        l.forget(0, 2 * P);
        let new = l.generation();
        l.lay(0, 2 * P, PAGE_READONLY, new);
        assert!(l.release(0, 2 * P, Some(old)).is_empty(), "the old guard is gone");
        assert_eq!(l.at(0).unwrap().1.generation, new);
        let taken = l.release(0, 2 * P, Some(new));
        assert_eq!(taken.len(), 1);
        assert_eq!(taken[0].1.requested, PAGE_READONLY);
        assert!(l.is_empty());
    }

    #[test]
    fn a_partial_release_leaves_the_pages_around_it_guarded() {
        let mut l = Ledger::new();
        let g = l.generation();
        l.lay(0, 3 * P, PAGE_READWRITE, g);
        let taken = l.release(P, 2 * P, None);
        assert_eq!(taken, [(P, Run { end: 2 * P, requested: PAGE_READWRITE, generation: g })]);
        assert_eq!(l.gaps(0, 3 * P), [(P, 2 * P)]);
        assert_eq!(l.next_after(0), Some(2 * P));
    }
}
