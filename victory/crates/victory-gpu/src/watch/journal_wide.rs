//! The journal for many askers (the fork-parity plan's N1, stage S1; the N1 note's §3), behind
//! `VICTORY_JOURNAL_WIDE=1` ([`on`], read once, off by default). Every entry point of
//! [`super::journal`] comes here at its first line when the switch is on, and runs its old
//! body otherwise.
//!
//! What changes against the old journal: the book of stamps is read by many threads at once
//! (an `RwLock`), and written only to note an answer; a kernel asking (or a guard of view
//! memory) is made holding only a hold of the pages it resets, never the book, so askers of
//! disjoint pages ask at once. The rules the code keeps, each where it is kept:
//! - every number is drawn after the reset it stamps for, in the write section that writes
//!   the stamps (`note_answer`, `note_ahead`, `wrote`, `drain`), so stamps arrive in number
//!   order and a title write after a copy's read is stamped above that copy's mark;
//! - a hold covers exactly what its asking resets, and the hold of everything
//!   ([`Askers::hold_all`], the submission's asking) registers before it waits: no range
//!   hold is admitted while it waits;
//! - no guard of the book is alive while its thread waits for a hold or for the book, or
//!   takes the guards' lock (three thread-local asserts in tests);
//! - every reader of the book that hands out something a receipt rests on drains the
//!   published writes first ([`Askers::drain_first`]).

use std::collections::BTreeMap;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release, SeqCst};
use std::sync::{Condvar, Mutex, MutexGuard, RwLock};

use super::journal::{AHEAD_GAP, AHEAD_MAX, GROUP, PUBLISHED, PUBLISHED_N};
use super::{KERNEL_PAGE, PAGE};

/// Whether the journal for many askers is on (`VICTORY_JOURNAL_WIDE=1`; off by default), read
/// once.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_JOURNAL_WIDE").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                "VICTORY_JOURNAL_WIDE=1: the journal of title memory's writes is read by many threads at once, \
                 and asks the kernel holding only the pages it asks about"
            );
        }
        on
    })
}

/// The journal the entry points of [`super::journal`] use with the switch on.
pub(super) static WIDE: Askers = Askers::new(&Kernel);

/// What an asking asks: the kernel and the guards. Fixtures give their own [`Askers`] a fake.
pub(super) trait Source: Sync {
    /// The written pages of `[first, last)` (64 KiB pages) since the last reset, reset now
    /// and added to `found`; false, and nothing reset, if the range is not write-watched
    /// memory of one reservation.
    fn ask(&self, first: usize, last: usize, found: &mut Vec<usize>) -> bool;
    /// View memory: the pages written since they were last guarded, guarded again
    /// ([`super::guard`]); `None` if the memory is not a view.
    fn guard(&self, start: usize, end: usize) -> Option<Vec<usize>>;
    /// Whether `start` begins in guarded view memory ([`super::view_first`]).
    fn view_first(&self, start: usize) -> bool;
}

/// The kernel's write watch and our guards.
struct Kernel;

thread_local! {
    /// Room for the kernel's answer, a thread's own: askings run on many threads at once.
    static ANSWER: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl Source for Kernel {
    fn ask(&self, first: usize, last: usize, found: &mut Vec<usize>) -> bool {
        const WRITE_WATCH_FLAG_RESET: u32 = 1;
        ANSWER.with(|answer| {
            let mut answer = answer.borrow_mut();
            let mut count = (last - first) * PAGE / KERNEL_PAGE;
            if answer.len() < count {
                answer.resize(count, 0);
            }
            let mut granularity = 0;
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            crate::stalls::kernel(crate::stalls::Kernel::Watch);
            // The span asked of the kernel, answered or not (the memory census).
            #[cfg(feature = "diagnostics")]
            crate::memclass::queried(first * PAGE, last * PAGE);
            // SAFETY: the answer buffer holds `count` entries.
            let failed = unsafe {
                crate::kernelstat::get_write_watch(
                    WRITE_WATCH_FLAG_RESET,
                    (first * PAGE) as *const _,
                    (last - first) * PAGE,
                    answer.as_mut_ptr().cast(),
                    &mut count,
                    &mut granularity,
                )
            } != 0;
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                let took = started.elapsed().as_micros() as u64;
                // Failed requests counted apart from watches, as the old journal counts them.
                if failed {
                    add(C::AskFailed, 1);
                    add(C::AskFailedUs, took);
                } else {
                    add(C::Watches, 1);
                    add(C::WatchUs, took);
                    super::journal::slow(took);
                    super::journal::note_asked_pages(first, last);
                }
            }
            if failed {
                return false;
            }
            found.extend(answer[..count].iter().map(|&at| at / PAGE));
            true
        })
    }

    fn guard(&self, start: usize, end: usize) -> Option<Vec<usize>> {
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let written = super::guard(start, end);
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::Guards, 1);
            crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
        }
        written
    }

    fn view_first(&self, start: usize) -> bool {
        super::view_first(start)
    }
}

/// The stamps, and the asking that stands: the old journal's state without its counter
/// (now [`Askers::asked`]) and its answer buffer (now a thread's own).
struct Book {
    /// Page number → the number of the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// With receipts: per group of [`GROUP`] pages, its pages' highest stamp. Empty while the
    /// switch is off, as in the old journal, so that [`Askers::fresh_all`] answers as it does.
    groups: BTreeMap<usize, u64>,
    /// Page ranges a submission's asking asked about ([`Askers::ask_ahead`],
    /// [`Askers::ask_once`]), answered from the book until it ends, and its number.
    ahead: Vec<(usize, usize)>,
    ahead_asked: u64,
    /// The submission's one asking stands ([`Askers::ask_once`]) until
    /// [`Askers::ask_once_end`].
    kept: bool,
}

impl Book {
    fn covers(&self, first: usize, last: usize) -> bool {
        self.ahead.iter().any(|&(a, b)| a <= first && last <= b)
    }

    fn stamps(&self, first: usize, last: usize) -> Vec<(usize, u64)> {
        self.written.range(first..last).map(|(&p, &w)| (p, w)).collect()
    }
}

/// The page ranges held by askings under way, and how many holds of everything wait.
struct Holds {
    held: Vec<(usize, usize)>,
    all_waiting: usize,
}

impl Holds {
    /// Whether a hold of `[first, last)` (or of everything) may be taken now: a range while
    /// nothing held overlaps it and no hold of everything waits; everything when nothing is
    /// held.
    fn admits(&self, first: usize, last: usize, all: bool) -> bool {
        if all {
            return self.held.is_empty();
        }
        self.all_waiting == 0 && !self.held.iter().any(|&(a, b)| a < last && first < b)
    }
}

/// What a hold of everything holds.
const ALL: (usize, usize) = (0, usize::MAX);

/// The points a fixture can stop a thread at ([`Askers::pause`]). None is reached with a guard
/// of the book alive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) enum Point {
    /// Before a hold is taken (the book was read and did not answer).
    BeforeHold,
    /// A hold must wait: `[first, last)`, or everything.
    HoldWaits { first: usize, last: usize, all: bool },
    /// After a kernel asking or a guard (what it found is not in the book yet).
    AfterReset,
    /// Before the write section that draws the number and notes.
    BeforeNote,
}

/// Named mutations for the fixtures, each a rule of §3 broken (none outside tests).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) enum Mutation {
    /// A submission's asking draws its number at its hold, before its resets (F2).
    EarlyNumber = 1,
    /// `writes` asks without its hold (F3).
    NoHold,
    /// A stamp is overwritten by a lower one (F5).
    Insert,
    /// `ask_ahead` does not look at `kept` again once it holds everything (F7).
    NoSecondLook,
    /// The hold of everything does not register while it waits (F10).
    NoRegister,
}

/// The counters of the new path ([`Askers::counts`]): calls into it, notes lower than the stamp
/// they met (kept higher), resets outside the asking's hold, and notes whose number is not
/// above the counter read after their last reset. The last three are 0 by construction.
pub(super) const CALLS: usize = 0;
pub(super) const LOWER_NOTES: usize = 1;
pub(super) const UNHELD_RESETS: usize = 2;
pub(super) const EARLY_STAMPS: usize = 3;

/// The journal for many askers.
pub(super) struct Askers {
    book: RwLock<Book>,
    holds: Mutex<Holds>,
    freed: Condvar,
    /// Numbers the askings: drawn with `fetch_add`, only in the section that notes with it
    /// (or, for an answer that stamps nothing, under the read lock). 0 is "never".
    asked: AtomicU64,
    /// The newest number a section ended with ([`Askers::absent_number`]).
    seen: AtomicU64,
    source: &'static dyn Source,
    counts: [AtomicU64; 4],
    #[cfg(test)]
    hook: Mutex<Option<std::sync::Arc<dyn Fn(Point) -> bool + Send + Sync>>>,
    #[cfg(test)]
    mutation: std::sync::atomic::AtomicU8,
}

/// A hold of pages `span` (or of everything), let go when dropped.
pub(super) struct Hold<'a> {
    askers: &'a Askers,
    span: (usize, usize),
}

impl Hold<'_> {
    fn covers(&self, first: usize, last: usize) -> bool {
        self.span.0 <= first && last <= self.span.1
    }
}

impl Drop for Hold<'_> {
    fn drop(&mut self) {
        let mut h = self.askers.holds();
        if let Some(i) = h.held.iter().position(|&r| r == self.span) {
            h.held.swap_remove(i);
        }
        drop(h);
        self.askers.freed.notify_all();
        #[cfg(test)]
        HELD.with(|n| n.set(n.get() - 1));
    }
}

/// An asking's findings, kept outside the book until its write section.
#[derive(Default)]
struct Asking {
    found: Vec<usize>,
    ahead: Vec<(usize, usize)>,
    /// The counter read after the last reset ([`EARLY_STAMPS`]).
    after_reset: u64,
}

/// A guard of the book, read.
struct Read<'a>(std::sync::RwLockReadGuard<'a, Book>);
/// A guard of the book, written.
struct Write<'a>(std::sync::RwLockWriteGuard<'a, Book>);

impl std::ops::Deref for Read<'_> {
    type Target = Book;
    fn deref(&self) -> &Book {
        &self.0
    }
}

impl std::ops::Deref for Write<'_> {
    type Target = Book;
    fn deref(&self) -> &Book {
        &self.0
    }
}

impl std::ops::DerefMut for Write<'_> {
    fn deref_mut(&mut self) -> &mut Book {
        &mut self.0
    }
}

#[cfg(test)]
impl Drop for Read<'_> {
    fn drop(&mut self) {
        BOOK.with(|n| n.set(n.get() - 1));
    }
}

#[cfg(test)]
impl Drop for Write<'_> {
    fn drop(&mut self) {
        BOOK.with(|n| n.set(n.get() - 1));
    }
}

#[cfg(test)]
thread_local! {
    /// Tests: guards of a book alive on this thread (at most one).
    static BOOK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Tests: holds alive on this thread (at most one).
    static HELD: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Tests: a guard of the book is about to be taken; this thread holds none (the first of the
/// three asserts: depth at most one, so no thread waits for the book holding it).
#[cfg(test)]
fn taking_book() {
    BOOK.with(|n| {
        assert_eq!(n.get(), 0, "a second guard of the journal's book on one thread");
        n.set(1);
    });
}

/// Tests: the guards' lock is about to be taken (`super::guards`); this thread holds no guard
/// of a book (the second assert: the book and the guards' lock are never nested).
#[cfg(test)]
pub(crate) fn no_book_here() {
    BOOK.with(|n| assert_eq!(n.get(), 0, "the guards' lock taken with a guard of the journal's book alive"));
}

/// Tests: the pool's work is about to run on this thread (`crate::shadow`'s `wide`); it holds
/// no asking's hold (the third assert: a hold is never held across the pool).
#[cfg(test)]
pub(crate) fn no_hold_here() {
    HELD.with(|n| assert_eq!(n.get(), 0, "the pool's work run holding an asking's hold"));
}

impl Askers {
    pub(super) const fn new(source: &'static dyn Source) -> Askers {
        Askers {
            book: RwLock::new(Book {
                written: BTreeMap::new(),
                groups: BTreeMap::new(),
                ahead: Vec::new(),
                ahead_asked: 0,
                kept: false,
            }),
            holds: Mutex::new(Holds { held: Vec::new(), all_waiting: 0 }),
            freed: Condvar::new(),
            asked: AtomicU64::new(0),
            seen: AtomicU64::new(0),
            source,
            counts: [const { AtomicU64::new(0) }; 4],
            #[cfg(test)]
            hook: Mutex::new(None),
            #[cfg(test)]
            mutation: std::sync::atomic::AtomicU8::new(0),
        }
    }

    fn count(&self, what: usize) {
        self.counts[what].fetch_add(1, Relaxed);
    }

    fn read(&self) -> Read<'_> {
        #[cfg(test)]
        taking_book();
        Read(self.book.read().unwrap_or_else(|e| e.into_inner()))
    }

    fn write(&self) -> Write<'_> {
        #[cfg(test)]
        taking_book();
        Write(self.book.write().unwrap_or_else(|e| e.into_inner()))
    }

    fn holds(&self) -> MutexGuard<'_, Holds> {
        self.holds.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A new number. Drawn only where the stamps it numbers are written, after the resets
    /// that found them; the counter's one read-modify-write chain orders it after every
    /// number handed out before (a mark, a receipt).
    fn draw(&self) -> u64 {
        self.asked.fetch_add(1, SeqCst) + 1
    }

    #[cfg(test)]
    fn mutated(&self, m: Mutation) -> bool {
        self.mutation.load(Relaxed) == m as u8
    }

    #[cfg(not(test))]
    #[inline(always)]
    fn mutated(&self, _: Mutation) -> bool {
        false
    }

    /// Under [`Mutation::EarlyNumber`] only: the number drawn at the hold.
    fn early(&self) -> Option<u64> {
        self.mutated(Mutation::EarlyNumber).then(|| self.draw())
    }

    /// Tests: the fixture's hook at `at` (true: the fixture's scheduler gates the thread, and
    /// a hold that waits looks again instead of sleeping). Nothing outside tests.
    #[cfg(test)]
    fn pause(&self, at: Point) -> bool {
        BOOK.with(|n| assert_eq!(n.get(), 0, "{at:?} reached with a guard of the journal's book alive"));
        let hook = self.hook.lock().unwrap_or_else(|e| e.into_inner()).clone();
        hook.is_some_and(|hook| hook(at))
    }

    #[cfg(not(test))]
    #[inline(always)]
    fn pause(&self, _: Point) -> bool {
        false
    }

    /// Tests: the asserts before a hold is waited for: no guard of the book (a thread reading
    /// the book while it waits for a hold, and the holder waiting to write it, is a deadlock
    /// under SRW), and no other hold on this thread.
    #[cfg(test)]
    fn holding() {
        BOOK.with(|n| assert_eq!(n.get(), 0, "a hold waited for with a guard of the journal's book alive"));
        HELD.with(|n| {
            assert_eq!(n.get(), 0, "a second hold on one thread");
            n.set(1);
        });
    }

    /// Holds pages `[first, last)`: waits while a held range overlaps them or a hold of
    /// everything waits.
    fn hold(&self, first: usize, last: usize) -> Hold<'_> {
        #[cfg(test)]
        Self::holding();
        let mut h = self.holds();
        while !h.admits(first, last, false) {
            h = self.wait_hold(h, first, last, false);
        }
        h.held.push((first, last));
        Hold { askers: self, span: (first, last) }
    }

    /// Holds everything (a submission's asking). Registers first, so that no range hold is
    /// admitted while it waits: it waits only for the holds admitted before, each of which
    /// ends (they wait for nothing but the book and the guards' lock).
    fn hold_all(&self) -> Hold<'_> {
        #[cfg(test)]
        Self::holding();
        let register = !self.mutated(Mutation::NoRegister);
        let mut h = self.holds();
        if register {
            h.all_waiting += 1;
        }
        while !h.admits(ALL.0, ALL.1, true) {
            h = self.wait_hold(h, ALL.0, ALL.1, true);
        }
        if register {
            h.all_waiting -= 1;
        }
        h.held.push(ALL);
        Hold { askers: self, span: ALL }
    }

    fn wait_hold<'a>(&'a self, h: MutexGuard<'a, Holds>, first: usize, last: usize, all: bool) -> MutexGuard<'a, Holds> {
        #[cfg(test)]
        if self.hook.lock().unwrap_or_else(|e| e.into_inner()).is_some() {
            drop(h);
            let gated = self.pause(Point::HoldWaits { first, last, all });
            let h = self.holds();
            if gated || h.admits(first, last, all) {
                return h;
            }
            return self.freed.wait(h).unwrap_or_else(|e| e.into_inner());
        }
        #[cfg(not(test))]
        let _ = (first, last, all);
        self.freed.wait(h).unwrap_or_else(|e| e.into_inner())
    }

    /// One reset of `[first, last)` by the source, under `hold` (counted if it does not
    /// cover it); what it found goes into `asking`.
    fn reset(&self, hold: Option<&Hold<'_>>, first: usize, last: usize, asking: &mut Asking) -> bool {
        if !hold.is_some_and(|h| h.covers(first, last)) {
            self.count(UNHELD_RESETS);
        }
        let asked = self.source.ask(first, last, &mut asking.found);
        asking.after_reset = self.asked.load(SeqCst);
        self.pause(Point::AfterReset);
        asked
    }

    /// A guard of view memory `[first, last)` under `hold`: what it found goes into `asking`;
    /// false if the memory is not a view.
    fn guarded(&self, hold: Option<&Hold<'_>>, first: usize, last: usize, asking: &mut Asking) -> bool {
        if !hold.is_some_and(|h| h.covers(first, last)) {
            self.count(UNHELD_RESETS);
        }
        let written = self.source.guard(first * PAGE, last * PAGE);
        asking.after_reset = self.asked.load(SeqCst);
        self.pause(Point::AfterReset);
        let Some(written) = written else {
            return false;
        };
        asking.found.extend(written);
        true
    }

    /// Whether writes the runtime published wait to be stamped (receipts only, as the old
    /// journal drains only with them).
    fn published(&self) -> bool {
        crate::receipts::on() && PUBLISHED_N.load(Acquire) > 0
    }

    /// Stamps the published writes with one new number, in the write section the caller
    /// holds, before that section draws its own.
    fn drain(&self, b: &mut Book) {
        if !self.published() {
            return;
        }
        let done = std::mem::take(&mut *PUBLISHED.lock().unwrap_or_else(|e| e.into_inner()));
        PUBLISHED_N.fetch_sub(done.len(), AcqRel);
        if done.is_empty() {
            return;
        }
        let now = self.draw();
        for (at, end) in done {
            for page in at / PAGE..end.div_ceil(PAGE) {
                self.note(b, page, now);
            }
        }
        self.seen.fetch_max(now, Release);
    }

    /// A reader's first step: the published writes stamped, in a write section of their
    /// own let go before the reader reads. A publish after the look is concurrent with the
    /// read and is drained by the next reader, under a higher number.
    fn drain_first(&self) {
        if self.published() {
            let mut b = self.write();
            self.drain(&mut b);
        }
    }

    /// `page` was found written at `now`. A stamp never goes down: a lower note is counted
    /// and not written (none can come: every number is drawn in its noting section).
    fn note(&self, b: &mut Book, page: usize, now: u64) {
        let stamp = b.written.entry(page).or_insert(0);
        if self.mutated(Mutation::Insert) || now >= *stamp {
            *stamp = now;
        } else {
            self.count(LOWER_NOTES);
        }
        if crate::receipts::on() {
            let group = b.groups.entry(page / GROUP).or_insert(0);
            *group = (*group).max(now);
        }
    }

    fn early_check(&self, now: u64, after_reset: u64) {
        if now <= after_reset {
            self.count(EARLY_STAMPS);
        }
    }

    /// The book's answer for `[first, last)` if a submission's asking covers it: its number,
    /// the stamps, and the counter (every published write stamped at or below it).
    fn book_answer(&self, first: usize, last: usize) -> Option<(u64, Vec<(usize, u64)>, u64)> {
        let b = self.read();
        b.covers(first, last).then(|| (b.ahead_asked, b.stamps(first, last), self.asked.load(SeqCst)))
    }

    /// [`super::journal::writes`].
    pub(super) fn writes(&self, start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
        self.count(CALLS);
        self.writes_looked(start, end, guarded).map(|(now, written, _)| (now, written))
    }

    /// [`super::journal::writes_top`].
    pub(super) fn writes_top(&self, start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>, u64)> {
        self.count(CALLS);
        self.writes_looked(start, end, guarded)
    }

    fn writes_looked(&self, start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>, u64)> {
        let answer = self.writes_answer(start, end, guarded);
        // Every look at title memory, for the memory census (outside the book).
        #[cfg(feature = "diagnostics")]
        if let Some((_, written, _)) = &answer
            && crate::memclass::on()
        {
            crate::memclass::looked(start, end, written);
        }
        answer
    }

    /// Read: the book answers if it covers the range. Else the range is held, the book read
    /// again (a submission's asking may have ended meanwhile), the source asked holding only
    /// the hold, and the answer noted; the hold is let go after the note, so that no
    /// overlapping asker is told "not written" before the stamps are in.
    fn writes_answer(&self, start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>, u64)> {
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        self.drain_first();
        if let Some(answer) = self.book_answer(first, last) {
            return Some(answer);
        }
        self.pause(Point::BeforeHold);
        let hold = (!self.mutated(Mutation::NoHold)).then(|| self.hold(first, last));
        if let Some(answer) = self.book_answer(first, last) {
            return Some(answer);
        }
        let mut asking = Asking::default();
        // A range that begins in guarded view memory cannot be write-watched: the guards
        // answer, as they would after the ask failed.
        let view = self.source.view_first(first * PAGE);
        if view && !guarded {
            return None;
        }
        let found = if view {
            self.guarded(hold.as_ref(), first, last, &mut asking)
        } else {
            self.reset(hold.as_ref(), first, last, &mut asking)
                || (guarded && self.guarded(hold.as_ref(), first, last, &mut asking))
        };
        if !found {
            return None;
        }
        self.pause(Point::BeforeNote);
        Some(self.note_answer(first, last, asking))
    }

    /// The write section of an answer: the published writes drained, then a number drawn,
    /// after every reset of this asking, and its findings stamped with it. With nothing found
    /// and nothing published, the number is drawn under the read lock and nothing stamped.
    fn note_answer(&self, first: usize, last: usize, asking: Asking) -> (u64, Vec<(usize, u64)>, u64) {
        if asking.found.is_empty() && !self.published() {
            let b = self.read();
            let now = self.draw();
            let top = self.asked.load(SeqCst);
            self.seen.fetch_max(top, Release);
            return (now, b.stamps(first, last), top);
        }
        let mut b = self.write();
        self.drain(&mut b);
        let now = self.draw();
        self.early_check(now, asking.after_reset);
        for &page in &asking.found {
            self.note(&mut b, page, now);
        }
        let top = self.asked.load(SeqCst);
        self.seen.fetch_max(top, Release);
        (now, b.stamps(first, last), top)
    }

    /// `ranges` as page ranges, without the empty ones and those that begin in guarded view
    /// memory (left to `writes` and the guards), in order.
    fn pieces(&self, ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut pieces: Vec<(usize, usize)> = ranges
            .iter()
            .filter(|r| r.0 < r.1 && !self.source.view_first(r.0))
            .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
            .collect();
        pieces.sort_unstable();
        pieces
    }

    /// Ranges close together asked about as one, as the old `ask_ahead` joins them.
    fn ask_runs(&self, all: &Hold<'_>, pieces: &[(usize, usize)], asking: &mut Asking) {
        let mut i = 0;
        while i < pieces.len() {
            let first = pieces[i].0;
            let mut last = pieces[i].1;
            let mut k = i + 1;
            while k < pieces.len() && pieces[k].0 <= last + AHEAD_GAP && last.max(pieces[k].1) - first <= AHEAD_MAX {
                last = last.max(pieces[k].1);
                k += 1;
            }
            self.ask_run(all, &pieces[i..k], asking);
            i = k;
        }
    }

    /// One asking for `pieces`, by halves if it fails; a piece alone is left to `writes`.
    fn ask_run(&self, all: &Hold<'_>, pieces: &[(usize, usize)], asking: &mut Asking) {
        if pieces.len() < 2 {
            return;
        }
        let first = pieces[0].0;
        let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
        if !self.source.view_first(first * PAGE) && self.reset(Some(all), first, last, asking) {
            asking.ahead.push((first, last));
            return;
        }
        let (a, b) = pieces.split_at(pieces.len() / 2);
        self.ask_run(all, a, asking);
        self.ask_run(all, b, asking);
    }

    /// The write section of a submission's asking: drained, one number drawn after its last
    /// reset, everything found stamped with it, and the asking stands under it.
    fn note_ahead(&self, asking: Asking, early: Option<u64>, once: bool) {
        self.pause(Point::BeforeNote);
        let mut b = self.write();
        self.drain(&mut b);
        let now = early.unwrap_or_else(|| self.draw());
        self.early_check(now, asking.after_reset);
        for &page in &asking.found {
            self.note(&mut b, page, now);
        }
        b.ahead_asked = now;
        if once {
            b.kept = !asking.ahead.is_empty();
        }
        b.ahead = asking.ahead;
        self.seen.fetch_max(self.asked.load(SeqCst), Release);
    }

    /// [`super::journal::ask_ahead`]: holds everything, clears the last asking under the write
    /// lock before its first reset, and looks at `kept` again once it holds (a submission's
    /// one asking may have begun since the first look).
    pub(super) fn ask_ahead(&self, ranges: &[(usize, usize)]) {
        self.count(CALLS);
        let pieces = self.pieces(ranges);
        if pieces.len() < 2 || self.read().kept {
            return;
        }
        self.pause(Point::BeforeHold);
        let all = self.hold_all();
        let early = self.early();
        {
            let mut b = self.write();
            if b.kept && !self.mutated(Mutation::NoSecondLook) {
                return;
            }
            b.ahead.clear();
        }
        let mut asking = Asking::default();
        self.ask_runs(&all, &pieces, &mut asking);
        self.note_ahead(asking, early, false);
    }

    /// [`super::journal::ask_once`], as one hold: `kept` set at its start (so `wrote` notes
    /// from there on, and a flush's `ask_ahead` leaves it alone), the joined ranges asked,
    /// then each range they leave alone, and one number drawn after the last reset.
    pub(super) fn ask_once(&self, ranges: &[(usize, usize)]) {
        self.count(CALLS);
        if !super::once() {
            return;
        }
        let pieces = self.pieces(ranges);
        self.pause(Point::BeforeHold);
        let all = self.hold_all();
        let early = self.early();
        {
            let mut b = self.write();
            b.ahead.clear();
            b.kept = true;
        }
        let mut asking = Asking::default();
        if pieces.len() >= 2 {
            self.ask_runs(&all, &pieces, &mut asking);
        }
        // A range that lies alone is asked here, once for the submission (the old
        // `ask_rest`); what is not write-watched stays with `writes`.
        for &(start, end) in ranges {
            if start >= end || self.source.view_first(start) {
                continue;
            }
            let (first, last) = (start / PAGE, end.div_ceil(PAGE));
            if !asking.ahead.iter().any(|&(a, b)| a <= first && last <= b) && self.reset(Some(&all), first, last, &mut asking) {
                asking.ahead.push((first, last));
            }
        }
        self.note_ahead(asking, early, true);
    }

    /// [`super::journal::ask_ahead_done`] (the submitter's; write lock only).
    pub(super) fn ask_ahead_done(&self) {
        self.count(CALLS);
        let mut b = self.write();
        if !b.kept {
            b.ahead.clear();
        }
    }

    /// [`super::journal::ask_once_end`] (the submitter's; write lock only).
    pub(super) fn ask_once_end(&self) {
        self.count(CALLS);
        let mut b = self.write();
        if b.kept {
            b.kept = false;
            b.ahead.clear();
        }
    }

    /// [`super::journal::wrote`]: noted with a number of its own, under which the asking
    /// that stands then stands.
    pub(super) fn wrote(&self, at: usize, len: usize) {
        self.count(CALLS);
        let mut b = self.write();
        if !b.kept || len == 0 {
            return;
        }
        self.drain(&mut b);
        let now = self.draw();
        b.ahead_asked = now;
        for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
            self.note(&mut b, page, now);
        }
        self.seen.fetch_max(now, Release);
    }

    /// [`super::journal::asked`] (drains first: an undrained landing would answer "not stale").
    pub(super) fn asked(&self, start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
        self.count(CALLS);
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        self.drain_first();
        let b = self.read();
        b.covers(first, last).then(|| (b.ahead_asked, b.stamps(first, last)))
    }

    /// [`super::journal::recorded`] (drains first: the recheck compares stamps with a record).
    pub(super) fn recorded(&self, start: usize, end: usize) -> Vec<(usize, u64)> {
        self.count(CALLS);
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        self.drain_first();
        self.read().stamps(first, last)
    }

    /// [`super::journal::top`]: every write published so far stamped at or below it.
    pub(super) fn top(&self) -> u64 {
        self.count(CALLS);
        self.drain_first();
        self.asked.load(SeqCst)
    }

    /// [`super::journal::absent_number`]: a number at or below the newest, read without the
    /// book (a lower one is the safe side).
    pub(super) fn absent_number(&self) -> u64 {
        self.count(CALLS);
        self.seen.load(Acquire)
    }

    /// [`super::journal::fresh_all`] (drains first: a stamp above the receipt says "not fresh").
    pub(super) fn fresh_all(&self, ranges: &[(usize, usize, u64)]) -> Vec<bool> {
        self.count(CALLS);
        self.drain_first();
        let b = self.read();
        ranges
            .iter()
            .map(|&(start, end, receipt)| {
                let (first, last) = (start / PAGE, end.div_ceil(PAGE));
                if first >= last {
                    return true;
                }
                let groups = first / GROUP..=(last - 1) / GROUP;
                if b.groups.range(groups).all(|(_, &stamp)| stamp <= receipt) {
                    return true;
                }
                b.written.range(first..last).all(|(_, &stamp)| stamp <= receipt)
            })
            .collect()
    }

    /// Tests: the counters ([`CALLS`] and the others).
    #[cfg(test)]
    pub(super) fn counts(&self) -> [u64; 4] {
        [0, 1, 2, 3].map(|i| self.counts[i].load(Relaxed))
    }
}

#[cfg(test)]
mod tests {
    //! The N1 note's fixtures F2 to F11 (§5), against [`Askers`] of their own: each with a fake
    //! [`Source`] (pages as numbers, a dirty bit per page that a title store sets and an asking
    //! resets) or the real kernel, and a hook that stops named threads at named points. No
    //! sleeps; every wait is bounded by [`WAIT`], every stress or search by [`BOUND`].

    use super::*;
    use std::cell::Cell;
    use std::collections::{BTreeSet, HashMap};
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc;
    use std::thread::Scope;
    use std::time::{Duration, Instant};

    /// The longest a fixture waits for one thing.
    const WAIT: Duration = Duration::from_secs(5);
    /// The longest a stress fixture or the search of every order runs.
    const BOUND: Duration = Duration::from_secs(120);

    fn page(p: usize) -> (usize, usize) {
        (p * PAGE, (p + 1) * PAGE)
    }

    fn pages(first: usize, last: usize) -> (usize, usize) {
        (first * PAGE, last * PAGE)
    }

    fn mutate(askers: &Askers, mutation: Option<Mutation>) {
        askers.mutation.store(mutation.map_or(0, |m| m as u8), Relaxed);
    }

    /// Page `p`'s stamp, 0 if none.
    fn stamp(askers: &Askers, p: usize) -> u64 {
        askers.recorded(page(p).0, page(p).1).first().map_or(0, |&(_, s)| s)
    }

    thread_local! {
        /// F11: the actor this thread is, for the scheduler and the fake's reset log.
        static ACTOR: Cell<Option<usize>> = const { Cell::new(None) };
    }

    #[derive(Default)]
    struct Mem {
        dirty: BTreeSet<usize>,
        version: BTreeMap<usize, u64>,
        views: BTreeSet<usize>,
        /// View pages written since they were last guarded.
        opened: BTreeSet<usize>,
        /// Each page a reset covered: the actor (F11), the page, its version then.
        resets: Vec<(Option<usize>, usize, u64)>,
    }

    /// The fake kernel and guards.
    #[derive(Default)]
    struct Fake {
        mem: Mutex<Mem>,
        asks: AtomicUsize,
        /// F4: the next guard says it is inside on the first and waits on the second.
        inside_guard: Mutex<Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>>,
    }

    impl Fake {
        fn leak() -> &'static Fake {
            Box::leak(Box::default())
        }

        fn mem(&self) -> MutexGuard<'_, Mem> {
            self.mem.lock().unwrap_or_else(|e| e.into_inner())
        }

        /// The title stores into page `p`.
        fn store(&self, p: usize) {
            let mut m = self.mem();
            *m.version.entry(p).or_insert(0) += 1;
            if m.views.contains(&p) {
                m.opened.insert(p);
            } else {
                m.dirty.insert(p);
            }
        }

        fn version(&self, p: usize) -> u64 {
            self.mem().version.get(&p).copied().unwrap_or(0)
        }

        /// Pages `[first, last)` are view memory, not taken in yet (the first guard reports them).
        fn views(&self, first: usize, last: usize) {
            let mut m = self.mem();
            m.views.extend(first..last);
            m.opened.extend(first..last);
        }

        fn clear(&self) {
            *self.mem() = Mem::default();
            self.asks.store(0, SeqCst);
        }
    }

    impl Source for Fake {
        fn ask(&self, first: usize, last: usize, found: &mut Vec<usize>) -> bool {
            self.asks.fetch_add(1, SeqCst);
            let mut m = self.mem();
            if m.views.range(first..last).next().is_some() {
                return false;
            }
            let actor = ACTOR.with(Cell::get);
            for p in first..last {
                let v = m.version.get(&p).copied().unwrap_or(0);
                m.resets.push((actor, p, v));
            }
            let hit: Vec<usize> = m.dirty.range(first..last).copied().collect();
            for p in &hit {
                m.dirty.remove(p);
            }
            found.extend(hit);
            true
        }

        fn guard(&self, start: usize, end: usize) -> Option<Vec<usize>> {
            let (first, last) = (start / PAGE, end.div_ceil(PAGE));
            if !self.mem().views.contains(&first) {
                return None;
            }
            let stop = self.inside_guard.lock().unwrap_or_else(|e| e.into_inner()).take();
            if let Some((inside, go)) = stop {
                let _ = inside.send(());
                go.recv_timeout(WAIT).expect("the guard let go");
            }
            let mut m = self.mem();
            let hit: Vec<usize> = m.opened.range(first..last).copied().collect();
            for p in &hit {
                m.opened.remove(p);
            }
            Some(hit)
        }

        fn view_first(&self, start: usize) -> bool {
            self.mem().views.contains(&(start / PAGE))
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Kind {
        BeforeHold,
        HoldWaits,
        AfterReset,
        BeforeNote,
    }

    fn kind(at: Point) -> Kind {
        match at {
            Point::BeforeHold => Kind::BeforeHold,
            Point::HoldWaits { .. } => Kind::HoldWaits,
            Point::AfterReset => Kind::AfterReset,
            Point::BeforeNote => Kind::BeforeNote,
        }
    }

    /// Scripted stops: a thread (by name) reaching a point is noted; at a stop set for it, it
    /// waits to be let go ([`Gate::go`]).
    #[derive(Default)]
    struct Gate {
        state: Mutex<GateState>,
        moved: Condvar,
    }

    #[derive(Default)]
    struct GateState {
        reached: Vec<(String, Kind)>,
        stops: Vec<(String, Kind)>,
        go: HashMap<String, usize>,
    }

    impl Gate {
        fn on(askers: &Askers) -> Arc<Gate> {
            let gate = Arc::new(Gate::default());
            let g = gate.clone();
            *askers.hook.lock().unwrap() = Some(Arc::new(move |at| {
                g.reach(at);
                false
            }));
            gate
        }

        fn state(&self) -> MutexGuard<'_, GateState> {
            self.state.lock().unwrap_or_else(|e| e.into_inner())
        }

        /// `name` stops the next time it reaches `at`.
        fn stop(&self, name: &str, at: Kind) {
            self.state().stops.push((name.into(), at));
        }

        fn reach(&self, at: Point) {
            let name = std::thread::current().name().unwrap_or("").to_string();
            let at = kind(at);
            let mut s = self.state();
            s.reached.push((name.clone(), at));
            self.moved.notify_all();
            let Some(i) = s.stops.iter().position(|stop| stop.0 == name && stop.1 == at) else {
                return;
            };
            s.stops.remove(i);
            let deadline = Instant::now() + WAIT;
            while s.go.get(&name).copied().unwrap_or(0) == 0 {
                let left = deadline.saturating_duration_since(Instant::now());
                assert!(!left.is_zero(), "{name} was never let go at {at:?}");
                s = self.moved.wait_timeout(s, left).unwrap_or_else(|e| e.into_inner()).0;
            }
            *s.go.get_mut(&name).expect("let go") -= 1;
        }

        /// Waits until `name` has reached `at` (each arrival answers one wait).
        fn reached(&self, name: &str, at: Kind) -> Result<(), String> {
            let deadline = Instant::now() + WAIT;
            let mut s = self.state();
            loop {
                if let Some(i) = s.reached.iter().position(|r| r.0 == name && r.1 == at) {
                    s.reached.remove(i);
                    return Ok(());
                }
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return Err(format!("{name} did not reach {at:?} within {WAIT:?}"));
                }
                s = self.moved.wait_timeout(s, left).unwrap_or_else(|e| e.into_inner()).0;
            }
        }

        /// The arrivals so far are forgotten (one part of a fixture is over).
        fn forget(&self) {
            self.state().reached.clear();
        }

        fn go(&self, name: &str) {
            *self.state().go.entry(name.into()).or_insert(0) += 1;
            self.moved.notify_all();
        }
    }

    /// Runs `f` on a thread named `name`; its result comes back on the receiver.
    fn run<'s, T: Send + 's>(s: &'s Scope<'s, '_>, name: &str, f: impl FnOnce() -> T + Send + 's) -> mpsc::Receiver<T> {
        let (tell, told) = mpsc::channel();
        std::thread::Builder::new()
            .name(name.into())
            .spawn_scoped(s, move || {
                let _ = tell.send(f());
            })
            .expect("a thread");
        told
    }

    fn get<T>(told: &mpsc::Receiver<T>, what: &str) -> T {
        told.recv_timeout(WAIT).unwrap_or_else(|e| panic!("{what}: no answer within {WAIT:?} ({e})"))
    }

    /// Runs the ignored test `name` (its path from the crate's root) in a process of its own,
    /// with this journal and receipts switched as asked (each switch is read once a process).
    fn child(name: &str, wide: bool, receipts: bool) {
        let exe = std::env::current_exe().expect("the test binary");
        let mut command = std::process::Command::new(exe);
        command.args(["--exact", name, "--ignored", "--test-threads", "1", "--nocapture"]);
        if wide {
            command.env("VICTORY_JOURNAL_WIDE", "1");
        } else {
            command.env_remove("VICTORY_JOURNAL_WIDE");
        }
        if receipts {
            command.env("VICTORY_TEXTURE_RECEIPTS", "1");
        } else {
            command.env_remove("VICTORY_TEXTURE_RECEIPTS");
        }
        let out = command.output().expect("the child ran");
        let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        assert!(out.status.success() && text.contains("1 passed"), "{name}: {text}");
    }

    // The asserts of "No book guard across a wait" (§3), each made to fire.

    #[test]
    #[should_panic(expected = "a second guard of the journal's book on one thread")]
    fn a_second_guard_of_the_book_on_one_thread_panics() {
        let askers = Askers::new(Fake::leak());
        let _read = askers.read();
        let _write = askers.write();
    }

    #[test]
    #[should_panic(expected = "a hold waited for with a guard of the journal's book alive")]
    fn a_hold_waited_for_under_the_book_panics() {
        let askers = Askers::new(Fake::leak());
        let _read = askers.read();
        let _hold = askers.hold(0, 1);
    }

    #[test]
    #[should_panic(expected = "a second hold on one thread")]
    fn a_second_hold_on_one_thread_panics() {
        let askers = Askers::new(Fake::leak());
        let _one = askers.hold(0, 1);
        let _two = askers.hold(2, 3);
    }

    #[test]
    #[should_panic(expected = "the guards' lock taken with a guard of the journal's book alive")]
    fn the_guards_lock_under_the_book_panics() {
        let askers = Askers::new(Fake::leak());
        let _read = askers.read();
        drop(super::super::guards());
    }

    #[test]
    #[should_panic(expected = "the pool's work run holding an asking's hold")]
    fn the_pool_run_under_a_hold_panics() {
        let askers = Askers::new(Fake::leak());
        let _hold = askers.hold(0, 1);
        crate::shadow::wide_run(Vec::new());
    }

    /// F2: a receipt handed out while the submission's asking holds everything (`top` takes no
    /// hold), after its first reset and before a later one, is below the stamp of the title
    /// write that later reset finds; an overlapping asker waits for the asking and answers
    /// above its number. Mutation: the asking's number drawn at its hold, before its resets.
    #[test]
    fn a_receipt_taken_while_the_one_asking_holds_everything_is_below_what_it_finds_after() {
        child("watch::journal_wide::tests::receipt_while_holding_child", false, true);
    }

    #[test]
    #[ignore = "run by its parent, with receipts on"]
    fn receipt_while_holding_child() {
        f2(None);
    }

    fn f2(mutation: Option<Mutation>) {
        assert!(crate::receipts::on(), "run by its parent");
        let fake = Fake::leak();
        let askers = Askers::new(fake);
        mutate(&askers, mutation);
        let gate = Gate::on(&askers);
        // X1 and X2 together, one asking; R alone, more than `AHEAD_GAP` away, asked after it.
        let (x1, x2, r) = (0, 1, 200);
        const { assert!(200 > 2 + AHEAD_GAP) };
        let ranges = [page(x1), page(x2), page(r)];
        gate.stop("S", Kind::AfterReset);
        std::thread::scope(|s| {
            let (askers, gate) = (&askers, &gate);
            let one = run(s, "S", move || {
                askers.ask_once(&ranges);
                askers.asked(page(r).0, page(r).1).expect("R stands").0
            });
            gate.reached("S", Kind::AfterReset).unwrap();
            let (tell, told) = mpsc::channel();
            let b = run(s, "B", move || {
                crate::watch::publish(page(500).0, page(500).1);
                let _ = tell.send(askers.top());
                // R and the page after it: overlapping R, not inside the asking.
                askers.writes(pages(r, r + 2).0, pages(r, r + 2).1, false).expect("watched")
            });
            let t = get(&told, "top while the asking holds everything");
            gate.reached("B", Kind::HoldWaits).unwrap();
            // The title writes R after the read the receipt stands for.
            fake.store(r);
            gate.go("S");
            let number = get(&one, "the one asking");
            let (now, written) = get(&b, "B's asking");
            let stamp = stamp(askers, r);
            assert!(stamp > t, "the write after the receipt is stamped above it: {stamp} > {t}");
            assert!(written.contains(&(r, stamp)), "B's answer holds the asking's stamp: {written:?}");
            assert!(now > number, "B's answer, after the asking's, is above its number: {now} > {number}");
        });
    }

    /// F3: an asker of pages another holds waits for that asking's stamps (and does not ask
    /// meanwhile); an asker of other pages does not wait. Mutation: `writes` without its hold.
    #[test]
    fn an_overlapping_asker_waits_for_the_stamps_and_a_disjoint_one_does_not() {
        f3(None);
    }

    fn f3(mutation: Option<Mutation>) {
        let fake = Fake::leak();
        let askers = Askers::new(fake);
        mutate(&askers, mutation);
        let gate = Gate::on(&askers);
        fake.store(11);
        gate.stop("A", Kind::AfterReset);
        std::thread::scope(|s| {
            let (askers, gate) = (&askers, &gate);
            let a = run(s, "A", move || askers.writes(pages(10, 12).0, pages(10, 12).1, false));
            gate.reached("A", Kind::AfterReset).unwrap();
            let b = run(s, "B", move || askers.writes(pages(11, 13).0, pages(11, 13).1, false));
            gate.reached("B", Kind::HoldWaits).expect("B waits for A's hold");
            assert_eq!(fake.asks.load(SeqCst), 1, "B did not ask while A holds the pages");
            let c = run(s, "C", move || askers.writes(page(20).0, page(20).1, false));
            let (_, other) = get(&c, "C's asking of other pages, while A holds its own").expect("watched");
            assert!(other.is_empty());
            assert_eq!(fake.asks.load(SeqCst), 2);
            gate.go("A");
            let (a_now, a_written) = get(&a, "A").expect("watched");
            assert_eq!(a_written, [(11, a_now)]);
            let (b_now, b_written) = get(&b, "B").expect("watched");
            assert!(b_written.contains(&(11, a_now)) && a_now < b_now, "B's answer {b_written:?} at {b_now}, A's stamp {a_now}");
        });
        assert_eq!(askers.counts()[UNHELD_RESETS], 0);
    }

    /// F4: a guard of view memory is made outside the book: while one is under way the book
    /// answers, hands out a number and is written.
    #[test]
    fn a_guard_is_made_outside_the_book() {
        let fake = Fake::leak();
        let askers = Askers::new(fake);
        fake.views(40, 42);
        askers.ask_ahead(&[page(0), page(1)]);
        assert!(askers.asked(page(0).0, page(1).1).is_some(), "X1 and X2 stand");
        let (inside, stopped) = mpsc::channel();
        let (go, gone) = mpsc::channel();
        *fake.inside_guard.lock().unwrap() = Some((inside, gone));
        std::thread::scope(|s| {
            let askers = &askers;
            let a = run(s, "A", move || askers.writes(pages(40, 42).0, pages(40, 42).1, true));
            stopped.recv_timeout(WAIT).expect("A inside the guard");
            let b = run(s, "B", move || {
                let book = askers.writes(page(0).0, page(0).1, false).expect("the book");
                let top = askers.top();
                askers.ask_ahead_done();
                (book, top)
            });
            let ((_, written), _) = get(&b, "the book's answer and a write section, while a guard is under way");
            assert!(written.is_empty());
            go.send(()).unwrap();
            let (now, written) = get(&a, "the guarded asking").expect("view memory");
            assert_eq!(written, [(40, now), (41, now)], "the first guard takes the pages in");
        });
    }

    /// F4 with real views: the title stores through the other view of guarded pages while their
    /// guard's findings are not noted yet (the asking stopped after its guards' section); the
    /// next asking reports the store, above the first one's number.
    #[test]
    fn a_store_through_another_view_while_a_guard_is_noted_is_told_next() {
        use super::super::tests::{drop_views, two_views};
        const SIZE: usize = 2 * PAGE;
        let askers = Askers::new(&Kernel);
        let gate = Gate::on(&askers);
        let (section, guarded, other) = two_views(SIZE);
        let first = guarded / PAGE;
        // The first asking takes the blocks in (a kernel asking fails first: not guarded yet).
        let (taken, written) = askers.writes(guarded, guarded + SIZE, true).expect("view memory");
        assert_eq!(written, [(first, taken), (first + 1, taken)], "the first guard takes the blocks in");
        gate.forget();
        gate.stop("A", Kind::AfterReset);
        let now = std::thread::scope(|s| {
            let (askers, gate) = (&askers, &gate);
            let a = run(s, "A", move || askers.writes(guarded, guarded + SIZE, true));
            // After A's guards' section, before its note.
            gate.reached("A", Kind::AfterReset).unwrap();
            // SAFETY: the second view, read-write once the handler has opened it.
            unsafe { ((other + PAGE + 8) as *mut u8).write(1) };
            gate.go("A");
            let (now, written) = get(&a, "the guarded asking").expect("view memory");
            assert_eq!(written, [(first, taken), (first + 1, taken)], "nothing written before A's guard");
            now
        });
        let (later, again) = askers.writes(guarded, guarded + SIZE, true).expect("view memory");
        assert!(later > now && now > taken);
        assert_eq!(again, [(first, taken), (first + 1, later)], "the store is told at the guarded view");
        drop_views(section, &[guarded, other], SIZE);
    }

    /// F5: our own write noted (`wrote`) while an asking of the same page is under way, before
    /// and after its note: the stamp is the larger number both ways, and no note meets a higher
    /// stamp (every number is drawn in its noting section). A lower note given directly keeps
    /// the higher stamp and is counted. Mutation: a stamp overwritten by a lower note.
    #[test]
    fn our_write_noted_beside_an_asking_keeps_the_larger_number() {
        f5(None);
    }

    fn f5(mutation: Option<Mutation>) {
        let fake = Fake::leak();
        let askers = Askers::new(fake);
        mutate(&askers, mutation);
        let gate = Gate::on(&askers);
        // A submission's asking stands, so `wrote` notes.
        askers.ask_once(&[page(0)]);
        assert!(askers.read().kept);
        for wrote_first in [true, false] {
            fake.store(5);
            gate.stop("A", Kind::BeforeNote);
            std::thread::scope(|s| {
                let (askers, gate) = (&askers, &gate);
                let a = run(s, "A", move || askers.writes(pages(5, 7).0, pages(5, 7).1, false));
                gate.reached("A", Kind::BeforeNote).unwrap();
                let ours = |askers: &Askers| {
                    askers.wrote(5 * PAGE, PAGE);
                    askers.asked(page(0).0, page(0).1).expect("stands").0
                };
                let w = wrote_first.then(|| ours(askers));
                gate.go("A");
                let (now, _) = get(&a, "A").expect("watched");
                let w = w.unwrap_or_else(|| ours(askers));
                assert_eq!(stamp(askers, 5), now.max(w), "wrote first: {wrote_first}, at {w}; A at {now}");
            });
        }
        assert_eq!(askers.counts()[LOWER_NOTES], 0, "no note met a higher stamp");
        {
            let mut b = askers.write();
            askers.note(&mut b, 9, 5);
            askers.note(&mut b, 9, 3);
        }
        assert_eq!(stamp(&askers, 9), 5, "a lower note does not lower the stamp");
        assert_eq!(askers.counts()[LOWER_NOTES], 1, "and is counted");
    }

    /// F6: each reader of the receipts table (§3) stamps a write published just before it
    /// when it answers; `top` from another thread while an asking is under way is at or above
    /// the stamp of a write published meanwhile.
    #[test]
    fn every_reader_a_receipt_rests_on_drains_first() {
        child("watch::journal_wide::tests::readers_drain_child", false, true);
    }

    #[test]
    #[ignore = "run by its parent, with receipts on"]
    fn readers_drain_child() {
        assert!(crate::receipts::on(), "run by its parent");
        let fake = Fake::leak();
        let askers = Askers::new(fake);
        let publish = |p: usize| crate::watch::publish(page(p).0, page(p).1);
        let newest = || askers.asked.load(SeqCst);
        askers.ask_ahead(&[page(0), page(1)]);
        // `asked`
        let before = newest();
        publish(0);
        let (_, stamps) = askers.asked(page(0).0, page(0).1).expect("stands");
        assert!(stamps.iter().any(|&(p, s)| p == 0 && s > before), "asked: {stamps:?} after {before}");
        // `recorded`
        let before = newest();
        publish(3);
        let stamps = askers.recorded(page(3).0, page(3).1);
        assert!(matches!(stamps[..], [(3, s)] if s > before), "recorded: {stamps:?} after {before}");
        // A book answer.
        let before = newest();
        publish(1);
        let (_, stamps, top) = askers.writes_top(page(1).0, page(1).1, false).expect("the book");
        assert!(matches!(stamps[..], [(1, s)] if s > before && s <= top), "book: {stamps:?} after {before}, top {top}");
        // A kernel answer: drained before its number is drawn.
        let before = newest();
        publish(7);
        let (now, stamps, top) = askers.writes_top(page(7).0, page(7).1, false).expect("watched");
        assert!(matches!(stamps[..], [(7, s)] if s > before && s < now && now <= top), "asked: {stamps:?} at {now}");
        // `fresh_all`
        let receipt = newest();
        publish(4);
        assert_eq!(askers.fresh_all(&[(page(4).0, page(4).1, receipt)]), [false], "a receipt from before the publish");
        // `top`
        let before = newest();
        publish(6);
        let t = askers.top();
        let s = stamp(&askers, 6);
        assert!(before < s && s <= t, "top {t}, the stamp {s}, before {before}");
        // `absent_number`: no drain, at or below the newest.
        let n = askers.absent_number();
        assert!(n > 0 && n <= newest());
        // A publish while an asking is under way; `top` from another thread.
        let gate = Gate::on(&askers);
        gate.stop("A", Kind::AfterReset);
        std::thread::scope(|s| {
            let (askers, gate) = (&askers, &gate);
            let a = run(s, "A", move || askers.writes(page(20).0, page(20).1, false));
            gate.reached("A", Kind::AfterReset).unwrap();
            let b = run(s, "B", move || {
                crate::watch::publish(page(30).0, page(30).1);
                askers.top()
            });
            let t = get(&b, "top while an asking is under way");
            let s = stamp(askers, 30);
            assert!(s > 0 && s <= t, "top {t}, the stamp {s}");
            gate.go("A");
            get(&a, "A").expect("watched");
        });
    }

    /// F7: the one asking waits for a range hold admitted before it and draws above it; and a
    /// flush's `ask_ahead` that read `kept` false before the one asking began looks again once
    /// it holds everything, and leaves that asking standing. Mutation: no second look.
    #[test]
    fn the_one_asking_waits_and_a_flush_looks_at_kept_again() {
        f7(None);
    }

    fn f7(mutation: Option<Mutation>) {
        let fake = Fake::leak();
        let askers = Askers::new(fake);
        mutate(&askers, mutation);
        let gate = Gate::on(&askers);
        let ones = [page(10), page(11), page(300)];
        gate.stop("A", Kind::AfterReset);
        std::thread::scope(|s| {
            let (askers, gate) = (&askers, &gate);
            let a = run(s, "A", move || askers.writes(page(10).0, page(10).1, false));
            gate.reached("A", Kind::AfterReset).unwrap();
            let one = run(s, "S", move || {
                askers.ask_once(&ones);
                askers.asked(page(10).0, page(10).1).expect("stands").0
            });
            gate.reached("S", Kind::HoldWaits).expect("the one asking waits for A");
            gate.go("A");
            let (a_now, _) = get(&a, "A").expect("watched");
            let number = get(&one, "the one asking");
            assert!(number > a_now, "{number} > {a_now}");
        });
        askers.ask_once_end();
        gate.forget();
        gate.stop("F", Kind::BeforeHold);
        gate.stop("S", Kind::AfterReset);
        std::thread::scope(|s| {
            let (askers, gate) = (&askers, &gate);
            let flush = run(s, "F", move || askers.ask_ahead(&[page(100), page(101)]));
            gate.reached("F", Kind::BeforeHold).unwrap();
            let one = run(s, "S", move || askers.ask_once(&ones));
            gate.reached("S", Kind::AfterReset).unwrap();
            gate.go("F");
            gate.reached("F", Kind::HoldWaits).expect("the flush waits for the one asking");
            gate.go("S");
            get(&one, "the one asking");
            get(&flush, "the flush's asking");
        });
        let b = askers.read();
        assert!(b.kept, "the one asking stands");
        assert_eq!(b.ahead, [(10, 12), (300, 301)], "and is the one asking's");
    }

    // F8: stress, with the real kernel.

    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// A write-watched allocation of `bytes`, freed when dropped.
    struct Watched(usize);

    impl Watched {
        fn new(bytes: usize) -> Watched {
            use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
            // SAFETY: a fresh reservation of our own.
            let at = unsafe { VirtualAlloc(std::ptr::null(), bytes, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as usize;
            assert!(at != 0 && at.is_multiple_of(PAGE));
            Watched(at)
        }
    }

    impl Drop for Watched {
        fn drop(&mut self) {
            use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
            // SAFETY: the reservation made in `new`.
            unsafe { VirtualFree(self.0 as *mut _, 0, MEM_RELEASE) };
        }
    }

    /// The 8 bytes of page `p` at `at` that writer `t` writes.
    fn slot(at: usize, p: usize, t: usize) -> &'static std::sync::atomic::AtomicU64 {
        // SAFETY: committed memory of the fixture's, aligned, alive while its threads run.
        unsafe { &*((at + p * PAGE + t * 8) as *const std::sync::atomic::AtomicU64) }
    }

    /// Threads writing pages and asking about ranges around them at once: whoever wrote a
    /// page before asking is told of that write, above its last number for the page.
    fn many_askers(askers: &Askers, at: usize, pages: usize, threads: usize, rounds: usize, guarded: bool, write_at: &[usize]) {
        let started = Instant::now();
        std::thread::scope(|s| {
            for t in 0..threads {
                s.spawn(move || {
                    let mut last = vec![0u64; pages];
                    let mut seed = t as u64 * 7919 + 1;
                    let mut next = |n: usize| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        (seed >> 33) as usize % n
                    };
                    for round in 0..rounds {
                        assert!(started.elapsed() < BOUND, "past {BOUND:?}");
                        let p = next(pages);
                        let from = p - next(p.min(12) + 1);
                        let to = p + 1 + next((pages - p).min(12));
                        let view = write_at[round % write_at.len()];
                        slot(view, p, t).store(round as u64 + 1, Relaxed);
                        let (now, written) = askers.writes(at + from * PAGE, at + to * PAGE, guarded).expect("watched");
                        let number = written.iter().find(|w| w.0 == at / PAGE + p).map_or(0, |w| w.1);
                        assert!(number > last[p], "page {p}: {number} is not after {}", last[p]);
                        for l in &mut last[from..to] {
                            *l = now;
                        }
                    }
                });
            }
        });
        assert_eq!(askers.counts()[1..], [0, 0, 0], "lower notes, unheld resets, early stamps");
    }

    /// F8 (the fork's `no_write_is_lost_between_askers`): 8 threads, 4,000 askings each.
    #[test]
    fn no_write_is_lost_between_askers() {
        const PAGES: usize = 48;
        let askers = Askers::new(&Kernel);
        let memory = Watched::new(PAGES * PAGE);
        many_askers(&askers, memory.0, PAGES, 8, 4000, false, &[memory.0]);
    }

    /// F8 of view memory, guarded: the stores go through either view of the pages, the askings
    /// through one.
    #[test]
    fn no_write_to_views_is_lost_between_askers() {
        use super::super::tests::{drop_views, two_views};
        const PAGES: usize = 12;
        let askers = Askers::new(&Kernel);
        let (section, one, other) = two_views(PAGES * PAGE);
        many_askers(&askers, one, PAGES, 8, 1000, true, &[one, other]);
        drop_views(section, &[one, other], PAGES * PAGE);
    }

    /// F8 with the one asking: one thread stands a submission's asking over half the pages,
    /// copies them by the book's answer and ends it, again and again, while six others write
    /// and ask. Every write its last copy did not read is stamped above that copy's mark.
    #[test]
    fn the_one_asking_beside_many_askers_misses_no_write_its_copy_did_not_read() {
        const PAGES: usize = 48;
        const KEPT: usize = 24;
        const THREADS: usize = 6;
        let askers = Askers::new(&Kernel);
        let memory = Watched::new(PAGES * PAGE);
        let at = memory.0;
        let started = Instant::now();
        let copy = std::thread::scope(|s| {
            let askers = &askers;
            for t in 0..THREADS {
                s.spawn(move || {
                    let mut last = [0u64; PAGES];
                    let mut seed = t as u64 * 104729 + 3;
                    let mut next = |n: usize| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        (seed >> 33) as usize % n
                    };
                    for round in 0..3000 {
                        assert!(started.elapsed() < BOUND, "past {BOUND:?}");
                        let p = next(PAGES);
                        let from = p - next(p.min(12) + 1);
                        let to = p + 1 + next((PAGES - p).min(12));
                        slot(at, p, t).store(((t as u64 + 1) << 32) | (round as u64 + 1), Relaxed);
                        let (now, written) = askers.writes(at + from * PAGE, at + to * PAGE, false).expect("watched");
                        // Inside the asking's pages the book may answer: what was written
                        // after its reset belongs to the next submission (`once`).
                        if to > KEPT {
                            let number = written.iter().find(|w| w.0 == at / PAGE + p).map_or(0, |w| w.1);
                            assert!(number > last[p], "page {p}: {number} is not after {}", last[p]);
                        }
                        for l in &mut last[from..to] {
                            *l = now;
                        }
                    }
                });
            }
            let once = s.spawn(move || {
                let mut copy = (0, Vec::new());
                for _ in 0..300 {
                    assert!(started.elapsed() < BOUND, "past {BOUND:?}");
                    askers.ask_once(&[(at, at + KEPT * PAGE)]);
                    let (mark, _) = askers.writes(at, at + KEPT * PAGE, false).expect("the book");
                    let read = (0..KEPT).map(|p| (0..THREADS).map(|t| slot(at, p, t).load(Relaxed)).collect::<Vec<_>>()).collect();
                    copy = (mark, read);
                    askers.ask_once_end();
                }
                copy
            });
            once.join().unwrap()
        });
        askers.ask_once_end();
        askers.writes(at, at + PAGES * PAGE, false).expect("watched");
        let (mark, read) = copy;
        for (p, read) in read.iter().enumerate() {
            let now: Vec<u64> = (0..THREADS).map(|t| slot(at, p, t).load(Relaxed)).collect();
            if &now != read {
                let s = stamp(&askers, at / PAGE + p);
                assert!(s > mark, "page {p}, written after the copy at {mark}, is stamped {s}");
            }
        }
        assert_eq!(askers.counts()[1..], [0, 0, 0], "lower notes, unheld resets, early stamps");
    }

    // F9: off means off.

    #[test]
    fn with_the_switch_off_no_entry_point_reaches_the_new_journal() {
        child("watch::journal_wide::tests::switch_off_child", false, false);
    }

    #[test]
    fn with_the_switch_on_every_entry_point_goes_to_the_new_journal() {
        child("watch::journal_wide::tests::switch_on_child", true, false);
    }

    /// Every entry point of the journal once, through `crate::watch`, over an allocation of
    /// its own; the new journal's calls after each.
    fn every_entry_point() -> Vec<(&'static str, u64)> {
        use crate::watch as w;
        let memory = Watched::new(4 * PAGE);
        let p = memory.0;
        let calls = || WIDE.counts()[CALLS];
        let mut after = Vec::new();
        w::ask_ahead(&[(p, p + PAGE), (p + PAGE, p + 2 * PAGE)]);
        after.push(("ask_ahead", calls()));
        w::ask_ahead_done();
        after.push(("ask_ahead_done", calls()));
        w::ask_once(&[(p, p + 2 * PAGE)]);
        after.push(("ask_once", calls()));
        w::wrote(p, 8);
        after.push(("wrote", calls()));
        assert!(w::asked(p, p + PAGE).is_some(), "the one asking stands");
        after.push(("asked", calls()));
        w::recorded(p, p + PAGE);
        after.push(("recorded", calls()));
        w::ask_once_end();
        after.push(("ask_once_end", calls()));
        // SAFETY: committed memory of ours.
        unsafe { ((p + 3 * PAGE + 8) as *mut u8).write(1) };
        let (now, written) = w::writes(p + 3 * PAGE, p + 4 * PAGE, false).expect("watched");
        assert_eq!(written, [(p / PAGE + 3, now)], "the store is told");
        after.push(("writes", calls()));
        w::writes_top(p + 3 * PAGE, p + 4 * PAGE, false).expect("watched");
        after.push(("writes_top", calls()));
        w::top();
        after.push(("top", calls()));
        w::fresh_all(&[(p, p + PAGE, 0)]);
        after.push(("fresh_all", calls()));
        w::absent_number();
        after.push(("absent_number", calls()));
        after
    }

    #[test]
    #[ignore = "run by its parent, with the switch off"]
    fn switch_off_child() {
        assert!(!on(), "run by its parent");
        let after = every_entry_point();
        assert!(after.iter().all(|&(_, n)| n == 0), "the new journal was called: {after:?}");
    }

    #[test]
    #[ignore = "run by its parent, with the switch on"]
    fn switch_on_child() {
        assert!(on(), "run by its parent");
        let after = every_entry_point();
        assert_eq!(after.len(), 12);
        for (i, &(name, n)) in after.iter().enumerate() {
            assert_eq!(n, i as u64 + 1, "{name} did not reach the new journal once: {after:?}");
        }
    }

    /// F10: two threads take and let go range holds hand over hand (each lets its hold go only
    /// once the other has taken one since, so something is always held); the hold of
    /// everything, started among them, is admitted within a bounded wait. Mutation: it does
    /// not register (and then waits until the stream stops).
    #[test]
    fn the_one_asking_is_not_kept_out_by_a_stream_of_range_holds() {
        let (registered, admitted) = f10(None).unwrap();
        assert!(admitted <= registered + 2, "admitted {} holds after it registered", admitted - registered);
    }

    #[derive(Default)]
    struct Stream {
        /// Holds taken so far, and the number each thread's current one was taken at.
        taken: u64,
        at: [u64; 2],
        waiting: [bool; 2],
        stop: bool,
        /// Holds taken when the hold of everything first waited.
        registered: Option<u64>,
    }

    /// The holds taken when the hold of everything registered and when it was admitted.
    fn f10(mutation: Option<Mutation>) -> Result<(u64, u64), String> {
        let askers = Askers::new(Fake::leak());
        mutate(&askers, mutation);
        let stream = Arc::new((Mutex::new(Stream::default()), Condvar::new()));
        let lock = |stream: &Arc<(Mutex<Stream>, Condvar)>| stream.0.lock().unwrap_or_else(|e| e.into_inner()).taken;
        {
            let stream = stream.clone();
            *askers.hook.lock().unwrap() = Some(Arc::new(move |at| {
                if let Point::HoldWaits { .. } = at {
                    let mut st = stream.0.lock().unwrap_or_else(|e| e.into_inner());
                    match std::thread::current().name() {
                        Some("T0") => st.waiting[0] = true,
                        Some("T1") => st.waiting[1] = true,
                        Some("S") => {
                            let taken = st.taken;
                            st.registered.get_or_insert(taken);
                        }
                        _ => {}
                    }
                    stream.1.notify_all();
                }
                false
            }));
        }
        std::thread::scope(|s| {
            let askers = &askers;
            let mut streams = Vec::new();
            for me in 0..2 {
                let stream = stream.clone();
                streams.push(run(s, &format!("T{me}"), move || -> Result<(), String> {
                    let other = 1 - me;
                    for k in 0.. {
                        // T0 holds pages 0 or 2, T1 pages 1 or 3: never the other's.
                        let first = me + 2 * (k % 2);
                        let hold = askers.hold(first, first + 1);
                        let mut st = stream.0.lock().unwrap_or_else(|e| e.into_inner());
                        st.taken += 1;
                        st.at[me] = st.taken;
                        st.waiting[me] = false;
                        stream.1.notify_all();
                        let deadline = Instant::now() + WAIT;
                        while !(st.at[other] > st.at[me] || st.waiting[other] || st.stop) {
                            let left = deadline.saturating_duration_since(Instant::now());
                            if left.is_zero() {
                                return Err(format!("T{me}: the other took nothing within {WAIT:?}"));
                            }
                            st = stream.1.wait_timeout(st, left).unwrap_or_else(|e| e.into_inner()).0;
                        }
                        let stop = st.stop;
                        drop(st);
                        drop(hold);
                        if stop {
                            break;
                        }
                    }
                    Ok(())
                }));
            }
            let deadline = Instant::now() + WAIT;
            while lock(&stream) < 20 {
                assert!(Instant::now() < deadline, "the stream did not start");
                std::thread::yield_now();
            }
            let all = {
                let stream = stream.clone();
                run(s, "S", move || {
                    let hold = askers.hold_all();
                    let taken = stream.0.lock().unwrap_or_else(|e| e.into_inner()).taken;
                    drop(hold);
                    taken
                })
            };
            let admitted = all.recv_timeout(Duration::from_secs(2));
            {
                let mut st = stream.0.lock().unwrap_or_else(|e| e.into_inner());
                st.stop = true;
                stream.1.notify_all();
            }
            for t in &streams {
                get(t, "a stream's end")?;
            }
            let registered = stream.0.lock().unwrap_or_else(|e| e.into_inner()).registered;
            match admitted {
                Ok(admitted) => Ok((registered.unwrap_or(admitted), admitted)),
                Err(_) => {
                    get(&all, "the hold of everything, once the stream stopped");
                    Err(format!("the hold of everything was kept out for 2 s by {} holds", lock(&stream)))
                }
            }
        })
    }

    // F11: every order.

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Status {
        Running,
        /// Stopped at a point of the journal, or (`None`) at the actor's start.
        At(Option<Point>),
        Go,
        Done,
    }

    /// The scheduler: one actor runs at a time, from one point to its next.
    struct Sched {
        state: Mutex<Vec<Status>>,
        moved: Condvar,
    }

    impl Sched {
        fn state(&self) -> MutexGuard<'_, Vec<Status>> {
            self.state.lock().unwrap_or_else(|e| e.into_inner())
        }

        fn park(&self, me: usize, at: Option<Point>) {
            let mut s = self.state();
            s[me] = Status::At(at);
            self.moved.notify_all();
            let deadline = Instant::now() + WAIT;
            while s[me] != Status::Go {
                let left = deadline.saturating_duration_since(Instant::now());
                assert!(!left.is_zero(), "actor {me} was never let go at {at:?}");
                s = self.moved.wait_timeout(s, left).unwrap_or_else(|e| e.into_inner()).0;
            }
            s[me] = Status::Running;
        }

        /// The actors' states once none runs.
        fn settled(&self) -> Result<Vec<Status>, String> {
            let deadline = Instant::now() + WAIT;
            let mut s = self.state();
            while s.iter().any(|x| matches!(x, Status::Running | Status::Go)) {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return Err(format!("an actor neither reached a point nor ended within {WAIT:?}: {s:?}"));
                }
                s = self.moved.wait_timeout(s, left).unwrap_or_else(|e| e.into_inner()).0;
            }
            Ok(s.clone())
        }

        fn step(&self, who: usize) {
            self.state()[who] = Status::Go;
            self.moved.notify_all();
        }
    }

    /// An actor's end, also when it panics.
    struct Ends(Arc<Sched>, usize);

    impl Drop for Ends {
        fn drop(&mut self) {
            self.0.state()[self.1] = Status::Done;
            self.0.moved.notify_all();
        }
    }

    /// The submission's asking (`ask_once`), an asker with a copy of its answer (`writes_top`),
    /// the title's store, a copy by a receipt (`top`).
    const S: usize = 0;
    const B: usize = 1;
    const T: usize = 2;
    const C: usize = 3;
    /// The two pages: `ask_once` asks the first as a joined asking, the second alone after it.
    const PA: usize = 0;
    const PB: usize = 200;

    /// One search: the page the title stores into, and the pages B asks about.
    #[derive(Clone, Copy, Debug)]
    struct Cfg {
        store: usize,
        asks: (usize, usize),
    }

    const CFGS: [Cfg; 6] = [
        Cfg { store: PB, asks: (PB, PB + 1) },
        Cfg { store: PA, asks: (PB, PB + 1) },
        Cfg { store: PA, asks: (PA, PA + 1) },
        Cfg { store: PB, asks: (PA, PA + 1) },
        Cfg { store: PA, asks: (PA, PB + 1) },
        Cfg { store: PB, asks: (PA, PB + 1) },
    ];

    #[derive(Default)]
    struct Seen {
        /// The one asking's number.
        s: Option<u64>,
        /// B's number, its counter, the stamps it was told, and the versions its copy read.
        b: Option<(u64, u64, Vec<(usize, u64)>, Vec<(usize, u64)>)>,
        /// C's receipt, and the versions of both pages it read.
        c: Option<(u64, [u64; 2])>,
    }

    /// Runs one order: `path`'s choices, then the first actor that can go at each step. The
    /// choices made, and those there were, are left in `chosen` and `alts`.
    fn run_order(
        cfg: Cfg,
        mutation: Option<Mutation>,
        counters: bool,
        fake: &'static Fake,
        path: &[usize],
        alts: &mut Vec<Vec<usize>>,
        chosen: &mut Vec<usize>,
    ) -> Result<(), String> {
        fake.clear();
        let askers = Arc::new(Askers::new(fake));
        mutate(&askers, mutation);
        let sched = Arc::new(Sched { state: Mutex::new(vec![Status::Running; 4]), moved: Condvar::new() });
        {
            let sched = sched.clone();
            *askers.hook.lock().unwrap() = Some(Arc::new(move |at| match ACTOR.with(Cell::get) {
                Some(me) => {
                    sched.park(me, Some(at));
                    true
                }
                None => false,
            }));
        }
        let seen = Arc::new(Mutex::new(Seen::default()));
        let threads: Vec<_> = (0..4)
            .map(|me| {
                let (askers, sched, seen) = (askers.clone(), sched.clone(), seen.clone());
                std::thread::Builder::new()
                    .name(format!("actor {me}"))
                    .spawn(move || {
                        ACTOR.with(|a| a.set(Some(me)));
                        let _ends = Ends(sched.clone(), me);
                        let seen = |f: &mut dyn FnMut(&mut Seen)| f(&mut seen.lock().unwrap_or_else(|e| e.into_inner()));
                        match me {
                            S => {
                                askers.ask_once(&[page(PA), page(PA), page(PB)]);
                                let number = askers.asked(page(PA).0, page(PA).1).map(|a| a.0);
                                seen(&mut |s| s.s = number);
                            }
                            B => {
                                sched.park(B, None);
                                let (now, written, top) = askers.writes_top(pages(cfg.asks.0, cfg.asks.1).0, pages(cfg.asks.0, cfg.asks.1).1, false).expect("watched");
                                // The copy reads the pages told written since its last mark (0)
                                // at once: a later read could only be newer.
                                let read: Vec<(usize, u64)> = written.iter().filter(|w| w.1 > 0).map(|&(p, _)| (p, fake.version(p))).collect();
                                seen(&mut |s| s.b = Some((now, top, written.clone(), read.clone())));
                            }
                            T => {
                                sched.park(T, None);
                                fake.store(cfg.store);
                            }
                            _ => {
                                sched.park(C, None);
                                let t = askers.top();
                                let read = [fake.version(PA), fake.version(PB)];
                                seen(&mut |s| s.c = Some((t, read)));
                            }
                        }
                    })
                    .expect("a thread")
            })
            .collect();
        loop {
            let status = sched.settled()?;
            let enabled: Vec<usize> = (0..4)
                .filter(|&i| match status[i] {
                    Status::At(Some(Point::HoldWaits { first, last, all })) => askers.holds().admits(first, last, all),
                    Status::At(_) => true,
                    _ => false,
                })
                .collect();
            if enabled.is_empty() {
                if status.iter().all(|s| *s == Status::Done) {
                    break;
                }
                return Err(format!("no actor can go on: {status:?}"));
            }
            let depth = chosen.len();
            let pick = match path.get(depth) {
                Some(&p) if enabled.contains(&p) => p,
                Some(&p) => return Err(format!("the replay went another way at step {depth}: {p} not in {enabled:?}")),
                None => enabled[0],
            };
            alts.push(enabled);
            chosen.push(pick);
            sched.step(pick);
        }
        for t in threads {
            t.join().map_err(|_| "an actor panicked".to_string())?;
        }
        // What the kernel's bits still hold, stamped above every number handed out.
        askers.ask_once_end();
        for p in [PA, PB] {
            askers.writes(page(p).0, page(p).1, false).expect("watched");
        }
        let seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        let (t, c_read) = seen.c.ok_or("C did not copy")?;
        for (i, p) in [PA, PB].into_iter().enumerate() {
            if c_read[i] < fake.version(p) && stamp(&askers, p) <= t {
                return Err(format!(
                    "P1: C read page {p} at version {} under receipt {t}; it is at {} now, stamped {}",
                    c_read[i],
                    fake.version(p),
                    stamp(&askers, p)
                ));
            }
        }
        let (now, top, written, read) = seen.b.clone().ok_or("B did not answer")?;
        let resets = fake.mem().resets.clone();
        for p in [PA, PB].into_iter().filter(|p| (cfg.asks.0..cfg.asks.1).contains(p)) {
            let got = read.iter().find(|r| r.0 == p).map_or(0, |r| r.1);
            if got < fake.version(p) && (stamp(&askers, p) <= now || stamp(&askers, p) <= top) {
                return Err(format!("P1: B's copy read page {p} at version {got} under {now} (top {top}); stamped {}", stamp(&askers, p)));
            }
            // What the reset whose number B answers with found there was written before it.
            let by = |actor: usize| resets.iter().rev().find(|r| r.0 == Some(actor) && r.1 == p).map(|r| r.2);
            let before = by(B).or_else(|| (seen.s == Some(now)).then(|| by(S)).flatten()).unwrap_or(0);
            if before > got {
                return Err(format!("P2: B was told {written:?} at {now}: page {p}, at version {before} when its asking reset it, copied at {got}"));
            }
        }
        if counters && askers.counts()[1..] != [0, 0, 0] {
            return Err(format!("counters: lower notes, unheld resets, early stamps {:?}", &askers.counts()[1..]));
        }
        Ok(())
    }

    /// Every order of every [`CFGS`] search; the orders run and the time, or the first order
    /// that fails.
    fn explore(mutation: Option<Mutation>, counters: bool) -> Result<(Vec<usize>, Duration), String> {
        let fake = Fake::leak();
        let started = Instant::now();
        let mut orders = Vec::new();
        for cfg in CFGS {
            let mut path = Vec::new();
            let mut n = 0;
            loop {
                let (mut alts, mut chosen) = (Vec::new(), Vec::new());
                run_order(cfg, mutation, counters, fake, &path, &mut alts, &mut chosen)
                    .map_err(|e| format!("{cfg:?}, order {chosen:?}: {e}"))?;
                n += 1;
                if started.elapsed() > BOUND {
                    return Err(format!("past {BOUND:?} after {n} orders of {cfg:?}"));
                }
                // The deepest step with a choice not tried yet.
                let Some(d) = (0..chosen.len()).rev().find(|&d| alts[d].last() != Some(&chosen[d])) else {
                    break;
                };
                let i = alts[d].iter().position(|&a| a == chosen[d]).expect("chosen among them");
                path = chosen[..d].to_vec();
                path.push(alts[d][i + 1]);
            }
            orders.push(n);
        }
        Ok((orders, started.elapsed()))
    }

    /// F11: in every order of the hook points of the submission's asking and an asker, a title
    /// store and a copy by a receipt, over two pages: a copy's mark is below the stamp of every
    /// write it did not read (P1), and an asker is told of every write its asking's reset found
    /// there (P2); no note is lower, unheld or early.
    #[test]
    fn in_every_order_a_copy_s_mark_is_below_every_write_it_did_not_read() {
        let (orders, took) = explore(None, true).unwrap();
        println!("F11: {} orders ({orders:?} by search) in {took:?}", orders.iter().sum::<usize>());
    }

    /// F11 under F2's and F3's mutations: some order fails P1 (the number drawn at the hold)
    /// and P2 (`writes` without its hold).
    #[test]
    fn some_order_fails_without_the_late_number_or_the_hold() {
        for (mutation, property) in [(Mutation::EarlyNumber, "P1"), (Mutation::NoHold, "P2")] {
            let failed = explore(Some(mutation), false).expect_err("an order that fails");
            println!("F11 under {mutation:?}: {failed}");
            assert!(failed.contains(&format!("{property}:")), "{mutation:?}: {failed}");
        }
    }
}
