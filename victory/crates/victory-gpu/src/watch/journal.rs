use std::collections::BTreeMap;
use std::sync::Mutex;

use super::journal_wide;
use super::{KERNEL_PAGE, PAGE, guard, view_first};

pub(super) struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    pub(super) written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call.
    ahead: Vec<(usize, usize)>,
    ahead_asked: u64,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
    /// With receipts (`crate::receipts`): per group of [`GROUP`] pages, the
    /// highest stamp of any of its pages, so a receipt over a group stamped
    /// below it needs no look at the pages. Empty while the switch is off.
    groups: BTreeMap<usize, u64>,
}

pub(super) static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
    ahead: Vec::new(),
    ahead_asked: 0,
    kept: false,
    groups: BTreeMap::new(),
});

/// Pages per group of [`Journal::groups`]: 16 MiB.
pub(super) const GROUP: usize = (16 << 20) / PAGE;

/// The journal, locked. With receipts (`crate::receipts`), the writes
/// published since the last taking are stamped first ([`publish`]).
fn journal() -> std::sync::MutexGuard<'static, Journal> {
    #[cfg(test)]
    HOLDING.with(|held| assert!(!held.get(), "journal taken twice on one thread"));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if crate::receipts::on() {
        drain(&mut j);
        SEEN.store(j.asked, std::sync::atomic::Ordering::Release);
    }
    j
}

/// With receipts: the journal's newest number at its last taking ([`absent_number`]).
static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
thread_local! {
    /// Tests: this thread holds the journal in [`writes_top`], across the
    /// guards' checks, so a second taking panics instead of hanging.
    static HOLDING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `page` was found written at `now`. With receipts, a stamp never goes
/// down (a guard: each asking's number is drawn in the taking that stamps
/// with it, after that taking's drain), and its group's highest stamp
/// follows.
fn note(j: &mut Journal, page: usize, now: u64) {
    if !crate::receipts::on() {
        j.written.insert(page, now);
        return;
    }
    let stamp = j.written.entry(page).or_insert(0);
    *stamp = (*stamp).max(now);
    let group = j.groups.entry(page / GROUP).or_insert(0);
    *group = (*group).max(now);
}

/// Writes the runtime saw end (receipts, `crate::receipts`), not stamped
/// yet: `[start, end)` of title memory. A leaf lock, so a writer never
/// waits for the journal, which an asker holds across the kernel's answer
/// and the guards' checks; and a write that ends inside such a check (a
/// pending read settled by `guard`) does not take the journal its own
/// thread holds.
pub(super) static PUBLISHED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
pub(super) static PUBLISHED_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Since the start: ranges published, and of them those at another view of
/// a page than the address written (`landed`'s fan-out).
pub(crate) static PUBLISHES: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// With receipts: the runtime's write into title memory `[at, end)` has
/// ended, its bytes there. Stamped above every receipt taken before now at
/// the journal's next taking ([`journal`]), which comes before any receipt
/// taken after now.
pub(crate) fn publish(at: usize, end: usize) {
    if at >= end {
        return;
    }
    PUBLISHES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    PUBLISHED.lock().unwrap_or_else(|e| e.into_inner()).push((at, end));
    PUBLISHED_N.fetch_add(1, std::sync::atomic::Ordering::Release);
}

/// Stamps the published writes ([`PUBLISHED`]) with one new number, under
/// the journal's lock the caller holds.
fn drain(j: &mut Journal) {
    use std::sync::atomic::Ordering::{AcqRel, Acquire};
    if PUBLISHED_N.load(Acquire) == 0 {
        return;
    }
    let done = std::mem::take(&mut *PUBLISHED.lock().unwrap_or_else(|e| e.into_inner()));
    PUBLISHED_N.fetch_sub(done.len(), AcqRel);
    if done.is_empty() {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    for (at, end) in done {
        for page in at / PAGE..end.div_ceil(PAGE) {
            note(j, page, now);
        }
    }
}

/// With receipts: the journal's newest number now, every write published
/// so far stamped at or below it. A receipt for a read that starts after.
pub(crate) fn top() -> u64 {
    if journal_wide::on() { return journal_wide::WIDE.top(); }
    journal().asked
}

/// With receipts: a number at or below the journal's newest, read without taking it (the
/// submitting thread holds it across its kernel calls): the texture pass's number for a
/// texture it finds unmapped, where a lower one is the safe side (more counts as written
/// after it).
pub(crate) fn absent_number() -> u64 {
    if journal_wide::on() { return journal_wide::WIDE.absent_number(); }
    SEEN.load(std::sync::atomic::Ordering::Acquire)
}

/// With receipts: for each `(start, end, receipt)`, whether no page of
/// `[start, end)` is stamped above `receipt` (one taking of the journal for
/// all: a group stamped no higher answers for its pages).
pub(crate) fn fresh_all(ranges: &[(usize, usize, u64)]) -> Vec<bool> {
    if journal_wide::on() { return journal_wide::WIDE.fresh_all(ranges); }
    let j = journal();
    ranges
        .iter()
        .map(|&(start, end, receipt)| {
            let (first, last) = (start / PAGE, end.div_ceil(PAGE));
            if first >= last {
                return true;
            }
            let groups = first / GROUP..=(last - 1) / GROUP;
            if j.groups.range(groups).all(|(_, &stamp)| stamp <= receipt) {
                return true;
            }
            j.written.range(first..last).all(|(_, &stamp)| stamp <= receipt)
        })
        .collect()
}

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
pub(super) const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
pub(super) const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (thousands of askings a frame,
/// tenths of a second). Ranges that are not write-watched, that begin in
/// guarded view memory, or that lie alone, are left to `writes`.
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    if journal_wide::on() { return journal_wide::WIDE.ask_ahead(ranges); }
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        // A range that begins in view memory cannot be write-watched: asking
        // about it would fail, so it is left to `writes` and the guards.
        .filter(|r| r.0 < r.1 && !view_first(r.0))
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    if pieces.len() < 2 {
        return;
    }
    pieces.sort_unstable();
    let mut j = journal();
    let j = &mut *j;
    // The submission's one asking stands ([`ask_once`]): what it does not
    // hold is left to `writes`.
    if j.kept {
        return;
    }
    j.asked += 1;
    j.ahead_asked = j.asked;
    j.ahead.clear();
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        while k < pieces.len()
            && pieces[k].0 <= last + AHEAD_GAP
            && last.max(pieces[k].1) - first <= AHEAD_MAX
        {
            last = last.max(pieces[k].1);
            k += 1;
        }
        ask_run(j, &pieces[i..k]);
        i = k;
    }
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in one run the
/// askings of a frame took several times as long in the lines where the pass over the
/// heap's views ran long as in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
pub(super) fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// Diagnostics: the pages asked of the kernel since the last present, for
/// the frame line's distinct-pages figure beside the repeats-counted bytes
/// (finding 13). Cleared at each present ([`crate::copystat::presented`]).
#[cfg(feature = "diagnostics")]
static ASKED_PAGES: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// The pages `[first, last)` were asked about (successful kernel asks only:
/// they are the spans the "asked about" bytes count).
#[cfg(feature = "diagnostics")]
pub(super) fn note_asked_pages(first: usize, last: usize) {
    let mut set = ASKED_PAGES.lock().unwrap_or_else(|e| e.into_inner());
    let set = set.get_or_insert_with(std::collections::HashSet::new);
    set.extend(first..last);
}

/// Diagnostics: the distinct pages asked about since the last present.
#[cfg(feature = "diagnostics")]
pub(crate) fn distinct_asked_pages() -> usize {
    ASKED_PAGES.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(0, std::collections::HashSet::len)
}

/// Diagnostics: a frame was presented; the distinct-pages count starts again.
#[cfg(feature = "diagnostics")]
pub(crate) fn clear_asked_pages() {
    if let Some(set) = ASKED_PAGES.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        set.clear();
    }
}

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    if journal_wide::on() { return journal_wide::WIDE.ask_ahead_done(); }
    let mut j = journal();
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (on by default;
/// `VICTORY_ASK_ONCE=0` turns it off).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: thousands of askings a frame, dozens a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since a run showed it holds (with card buffers: a third
    // off the frame, the picture unchanged over minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if journal_wide::on() { return journal_wide::WIDE.ask_once(ranges); }
    if !once() {
        return;
    }
    ask_once_end();
    ask_ahead(ranges);
    ask_rest(ranges);
}

/// [`ask_once`]'s second taking of the journal, after [`ask_ahead`]: asks
/// about the ranges `ask_ahead` left alone, and lets the asking stand.
pub(super) fn ask_rest(ranges: &[(usize, usize)]) {
    let mut j = journal();
    let j = &mut *j;
    // A number of its own, whatever `ask_ahead` asked: a number handed out
    // since the first taking, drawn or read (another thread's asking, `top`,
    // `absent_number`), is a copy's mark or a receipt, and what the kernel
    // calls below find may be written after it. A number read can be the first
    // taking's own, so only a new one is above it. Nothing draws between
    // this and those calls: the same taking.
    j.asked += 1;
    j.ahead_asked = j.asked;
    // `ask_ahead` leaves a range that lies alone to `writes`, which asks
    // once per flush: with one asking for the submission that was every
    // such range asked again at each of its flushes (thousands of
    // askings a frame with the switch on, as without it). They are asked
    // here, once; what is not write-watched stays with `writes`.
    for &(start, end) in ranges {
        if start >= end || view_first(start) {
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if !j.ahead.iter().any(|&(a, b)| a <= first && last <= b) && ask_kernel(j, first, last) {
            j.ahead.push((first, last));
        }
    }
    j.kept = !j.ahead.is_empty();
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    if journal_wide::on() { return journal_wide::WIDE.ask_once_end(); }
    let mut j = journal();
    if j.kept {
        j.kept = false;
        j.ahead.clear();
    }
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    if journal_wide::on() { return journal_wide::WIDE.wrote(at, len); }
    let mut j = journal();
    if !j.kept || len == 0 {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    // The asking that stands moves with it: `asked` and `writes` give it as
    // "now", and a page noted written after "now" stays stale for every
    // copy that takes it (copied at `now`, written at `now + 1`): copied
    // again at every refresh, and a reason to settle flights
    // (fxceless/victory-fork#410).
    j.ahead_asked = now;
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        note(&mut j, page, now);
    }
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`].
fn ask_run(j: &mut Journal, pieces: &[(usize, usize)]) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    // A run that begins in view memory cannot be write-watched: the ask
    // would fail. Split it as a failed ask does, so the ranges after the
    // view are asked about as before.
    if !view_first(first * PAGE) && ask_kernel(j, first, last) {
        j.ahead.push((first, last));
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(j, a);
    ask_run(j, b);
}

/// Asks the kernel for the written pages of pages `[first, last)` and notes
/// them in the journal under the ahead asking; false (and nothing changed)
/// if the range is not write-watched memory of one reservation.
fn ask_kernel(j: &mut Journal, first: usize, last: usize) -> bool {
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    // View memory cannot be write-watched: the ask would fail. No asking is
    // recorded; `writes` answers from the guards.
    if view_first(first * PAGE) {
        return false;
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
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
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    #[cfg(feature = "diagnostics")]
    if failed {
        use crate::copystat::{C, add};
        add(C::AskFailed, 1);
        add(C::AskFailedUs, started.elapsed().as_micros() as u64);
    }
    if failed {
        return false;
    }
    #[cfg(feature = "diagnostics")]
    note_asked_pages(first, last);
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    let now = j.ahead_asked;
    for k in 0..count {
        let page = j.found[k] / PAGE;
        note(j, page, now);
    }
    true
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages written since tracking began and the asking that last found each
/// written, with the number of the asking that stands (as [`writes`] gives
/// it). `None` if the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    if journal_wide::on() { return journal_wide::WIDE.asked(start, end); }
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = journal();
    j.ahead
        .iter()
        .any(|&(a, b)| a <= first && last <= b)
        .then(|| (j.ahead_asked, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()))
}

/// What the journal has recorded for `[start, end)` without asking the
/// kernel: the pages written since tracking began and the asking that last
/// found each, in page order. For the upload recheck
/// (`crate::host::recheck`), which must not clear the kernel's bits.
pub(crate) fn recorded(start: usize, end: usize) -> Vec<(usize, u64)> {
    if journal_wide::on() { return journal_wide::WIDE.recorded(start, end); }
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = journal();
    j.written
        .range(first..last)
        .map(|(&page, &asked)| (page, asked))
        .collect()
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded). A range that begins
/// in guarded view memory is not asked about, the ask being sure to fail
/// (`VICTORY_VIEW_ASK=1` asks as before).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    if journal_wide::on() { return journal_wide::WIDE.writes(start, end, guarded); }
    writes_top(start, end, guarded).map(|(now, written, _)| (now, written))
}

/// [`writes`], and the journal's newest number when it answered, every write
/// published so far stamped at or below it: with receipts
/// (`crate::receipts`), the receipt of a read of the range that starts after
/// (in a submission's one asking, "now" is that asking's number, older than
/// what was published since).
pub(crate) fn writes_top(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>, u64)> {
    if journal_wide::on() { return journal_wide::WIDE.writes_top(start, end, guarded); }
    let mut j = journal();
    #[cfg(test)]
    HOLDING.with(|held| held.set(true));
    let answer = writes_locked(&mut j, start, end, guarded);
    #[cfg(test)]
    HOLDING.with(|held| held.set(false));
    let top = j.asked;
    drop(j);
    // Every look at title memory, for the memory census (outside the
    // journal's lock).
    #[cfg(feature = "diagnostics")]
    if let Some((_, written)) = &answer
        && crate::memclass::on()
    {
        crate::memclass::looked(start, end, written);
    }
    answer.map(|(now, written)| (now, written, top))
}

/// [`writes`] with the journal's lock held. The fixture holds it across two
/// calls, so that no other test's failing asking moves the counter between
/// them (every kernel asking is made under this lock).
pub(super) fn writes_locked(
    j: &mut Journal,
    start: usize,
    end: usize,
    guarded: bool,
) -> Option<(u64, Vec<(usize, u64)>)> {
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    // Asked ahead with the submission's other ranges: the journal has it.
    if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
        return Some((
            j.ahead_asked,
            j.written
                .range(first..last)
                .map(|(&p, &w)| (p, w))
                .collect(),
        ));
    }
    // A range that begins in a block we made read-only is a view, which
    // cannot be write-watched: the ask would fail. The guards answer, as
    // they would after it. The guards' lock is not held across the call.
    let view = view_first(first * PAGE);
    if view && !guarded {
        return None;
    }
    let mut count = 0;
    let written = if view {
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let written = guard(first * PAGE, last * PAGE);
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::Guards, 1);
            crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
        }
        written?
    } else {
        count = (last - first) * PAGE / KERNEL_PAGE;
        j.found.resize(count.max(j.found.len()), 0);
        let mut granularity = 0;
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        // The span asked of the kernel, answered or not (the memory census).
        #[cfg(feature = "diagnostics")]
        crate::memclass::queried(first * PAGE, last * PAGE);
        #[cfg(feature = "diagnostics")]
        crate::stalls::kernel(crate::stalls::Kernel::Watch);
        // SAFETY: the answer buffer holds `count` entries.
        let failed = unsafe {
            crate::kernelstat::get_write_watch(
                WRITE_WATCH_FLAG_RESET,
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                j.found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            let took = started.elapsed().as_micros() as u64;
            // Failed requests counted apart from watches, as ask_kernel
            // counts them (finding 11): a failure is not a watch.
            if failed {
                add(C::AskFailed, 1);
                add(C::AskFailedUs, took);
            } else {
                add(C::Watches, 1);
                add(C::WatchUs, took);
                slow(took);
                note_asked_pages(first, last);
            }
        }
        if failed {
            count = 0;
            if !guarded {
                return None;
            }
            #[cfg(feature = "diagnostics")]
            let guarding = std::time::Instant::now();
            let written = guard(first * PAGE, last * PAGE);
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::add(crate::copystat::C::Guards, 1);
                crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
            }
            written?
        } else {
            Vec::new()
        }
    };
    // With receipts: writes that ended while the guards were checked (a
    // pending read settled in `guard`) are stamped now, in the journal this
    // thread holds, below the number it answers with.
    if crate::receipts::on() {
        drain(j);
    }
    j.asked += 1;
    let now = j.asked;
    let pages = j.found[..count].iter().map(|&at| at / PAGE).chain(written);
    for page in pages.collect::<Vec<_>>() {
        note(j, page, now);
    }
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}
