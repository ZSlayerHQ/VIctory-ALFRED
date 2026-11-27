//! Diagnostics: what held the host up, per frame, in one line after the
//! five-second frame line ([`line`]; diagnostics builds only, always on in
//! them: each count is a relaxed add, a clock read or a short lock where it
//! happens). Each figure says what it covers; several overlap and do not add.
//!
//! - Host pipelines built by the workers ([`pipeline_made`]): a build that
//!   ran, failed or not (a call that found one built or waited for another
//!   thread's build is none), for a draw or dispatch that had asked for it
//!   before the build began, or ahead of any ask; of the first, the ones
//!   compiled rather than loaded from the library, and those that failed;
//!   the title's threads' waits for pipelines.
//! - BCPack decode on DirectStorage's workers, summed over them
//!   ([`storage_decoded`]); its requests waiting to start, running, or held
//!   back for their page arrays, on all queues, now and at most
//!   ([`storage_depth`], from `victory-hle`).
//! - Settles, once each, by why they settled and by what released them:
//!   the settling thread's own fence waits, nothing to wait for (signals
//!   and writes alone), or the relay's landing, with pieces of copies in
//!   the card's memory fetched or not ([`settled`]); the submission lock
//!   when another thread held it ([`lock`]); waits of our threads for a
//!   fence on the card by where ([`card_waited`]), which overlap each other
//!   and the settles; the title's frame pacing; the texture refreshes'
//!   waits.
//! - Title memory: watched ranges checked, and copy and fill steps asked
//!   for (`crate::copystat`'s counters, read here, not taken).
//! - Kernel calls about memory at the call sites counted ([`kernel`]), on
//!   every thread and inside submissions ([`submitting`]); the whole
//!   process's page faults.
//! - The title's D3D12 objects we hand out boxed: made, and freed at their
//!   last release ([`object_made`], [`object_gone`]).

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

/// Where a thread of ours waits for a fence on the card.
#[derive(Clone, Copy)]
pub(crate) enum Site {
    /// A settle of the flights in the air, without the relay.
    Settle = 0,
    /// A synchronous submission, from its send.
    Submission = 1,
    /// A synchronous submission's fetch of card copies' pieces.
    CardBack = 2,
    /// A list of ours run under the submission lock.
    OwnLocked = 3,
    /// The relay's fetch of card copies' pieces.
    RelayFetch = 4,
    /// The relay's thread, for the next flight to land.
    Relay = 5,
    /// The thread that settles flights sent ahead, for the card.
    LateHelper = 6,
    /// In-place work on title memory.
    InPlace = 7,
}

const SITES: [&str; 8] = [
    "settles",
    "synchronous submissions",
    "their fetches",
    "our lists under the lock",
    "the relay's fetches",
    "the relay's thread",
    "the settling thread",
    "in-place work",
];

/// By [`Site`]: waits, microseconds.
static CARD: [[AtomicU64; 2]; 8] = [const { [const { AtomicU64::new(0) }; 2] }; 8];

/// A wait for the card at `site`, begun at `since`, is over.
pub(crate) fn card_waited(site: Site, since: Instant) {
    let c = &CARD[site as usize];
    c[0].fetch_add(1, Relaxed);
    c[1].fetch_add(since.elapsed().as_micros() as u64, Relaxed);
}

/// What let a settle go on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Release {
    /// Its own waits for the card's fence.
    Fence = 0,
    /// Nothing: it had no flight to wait for (signals and writes alone).
    Nothing = 1,
    /// A landing of the relay's that fetched no pieces of copies in the
    /// card's memory.
    Landing = 2,
    /// A landing of the relay's that fetched some.
    Fetched = 3,
}

/// Why a settle settled, by the host's `WHY_` numbers (2 to 10).
const WHY: [&str; 11] = [
    "",
    "",
    "a synchronous submission or present",
    "the card had run them",
    "a wait for a memory value",
    "title memory changed under shader writes",
    "title memory read",
    "a step of ours",
    "lists in the air",
    "bottom-of-pipe writes",
    "a blocking fence wait",
];

/// By why and [`Release`]: settles, microseconds.
static SETTLES: [[[AtomicU64; 2]; 4]; 11] = [const { [const { [const { AtomicU64::new(0) }; 2] }; 4] }; 11];

/// A settle for `why` that waited `us` microseconds was released by
/// `release`.
pub(crate) fn settled(why: usize, release: Release, us: u64) {
    let c = &SETTLES[why.min(10)][release as usize];
    c[0].fetch_add(1, Relaxed);
    c[1].fetch_add(us, Relaxed);
}

/// The relay's landings so far, twice, plus one if the last fetched pieces
/// of copies in the card's memory ([`landed`]).
static LANDING: AtomicU64 = AtomicU64::new(0);

/// The relay landed flights (`fetched`: it fetched pieces of copies in the
/// card's memory), under the line's lock, before it wakes those waiting
/// for a landing.
pub(crate) fn landed(fetched: bool) {
    let n = LANDING.load(Relaxed) >> 1;
    LANDING.store((n + 1) << 1 | u64::from(fetched), Relaxed);
}

/// What released a wait for the relay that just woke: its last landing
/// (read under the line's lock).
pub(crate) fn landing() -> Release {
    release_of(LANDING.load(Relaxed))
}

/// The release a landing word says ([`LANDING`]).
fn release_of(word: u64) -> Release {
    if word & 1 != 0 { Release::Fetched } else { Release::Landing }
}

/// The submission lock, when another thread held it: waits, microseconds.
static LOCK: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// `lock`, timed if another thread holds it.
pub(crate) fn lock<T>(lock: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match lock.try_lock() {
        Ok(held) => held,
        Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => {
            let waiting = Instant::now();
            let held = lock.lock().unwrap_or_else(|e| e.into_inner());
            LOCK[0].fetch_add(1, Relaxed);
            LOCK[1].fetch_add(waiting.elapsed().as_micros() as u64, Relaxed);
            held
        }
    }
}

/// Builds for an ask (count, microseconds, compiled, failed), ahead
/// (count, microseconds).
static PIPELINES: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

thread_local! {
    /// Pipelines this thread compiled (not loaded from the library).
    static COMPILED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// The thread is inside a submission.
    static INSIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The pipeline library compiled a pipeline on this thread.
pub(crate) fn compiled() {
    COMPILED.with(|c| c.set(c.get() + 1));
}

/// Pipelines compiled on this thread so far.
pub(crate) fn compiles_here() -> u64 {
    COMPILED.with(std::cell::Cell::get)
}

/// A worker ran a host pipeline's build in `us` microseconds: `asked`, a
/// draw or dispatch had asked for it before the build began; `compiled`,
/// not loaded from the library; `failed`, it made none.
pub(crate) fn pipeline_made(asked: bool, compiled: bool, failed: bool, us: u64) {
    if asked {
        PIPELINES[0].fetch_add(1, Relaxed);
        PIPELINES[1].fetch_add(us, Relaxed);
        PIPELINES[2].fetch_add(u64::from(compiled), Relaxed);
        PIPELINES[3].fetch_add(u64::from(failed), Relaxed);
    } else {
        PIPELINES[4].fetch_add(1, Relaxed);
        PIPELINES[5].fetch_add(us, Relaxed);
    }
}

/// DirectStorage: BCPack decodes and their nanoseconds.
static DECODES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// DirectStorage's requests now, and the most since the last line: one
/// lock, so that the line's reset loses no update.
static DEPTH: Mutex<(u64, u64)> = Mutex::new((0, 0));

/// DirectStorage decoded one request's BCPack data in `ns` nanoseconds.
pub fn storage_decoded(ns: u64) {
    DECODES[0].fetch_add(1, Relaxed);
    DECODES[1].fetch_add(ns, Relaxed);
}

/// DirectStorage's requests waiting to start, running or held back for
/// their page arrays, on all queues, now.
pub fn storage_depth(depth: u64) {
    let mut d = DEPTH.lock().unwrap_or_else(|e| e.into_inner());
    *d = (depth, d.1.max(depth));
}

/// The depth now and the most since the last call; the most starts again
/// from the depth now.
fn take_depth() -> [u64; 2] {
    let mut d = DEPTH.lock().unwrap_or_else(|e| e.into_inner());
    let taken = [d.0, d.1.max(d.0)];
    d.1 = d.0;
    taken
}

/// Kernel calls about memory, at the call sites counted.
#[derive(Clone, Copy)]
pub enum Kernel {
    /// `VirtualQuery`.
    Query = 0,
    /// `VirtualProtect`.
    Protect = 1,
    /// Mapping and unmapping views.
    Map = 2,
    /// Reserving, committing and releasing virtual memory.
    Alloc = 3,
    /// Write-watch asks and resets.
    Watch = 4,
    /// Reads and writes of memory through the kernel (`ReadProcessMemory`,
    /// `WriteProcessMemory`, `NtWriteVirtualMemory`): title memory copied
    /// without a fault.
    Read = 5,
}

/// By [`Kernel`]: on every thread, and inside submissions.
static KERNEL: [[AtomicU64; 6]; 2] = [const { [const { AtomicU64::new(0) }; 6] }; 2];

/// A kernel call about memory (the title's own reservations and mappings
/// too, from `victory-hle`).
pub fn kernel(call: Kernel) {
    KERNEL[0][call as usize].fetch_add(1, Relaxed);
    if INSIDE.with(std::cell::Cell::get) {
        KERNEL[1][call as usize].fetch_add(1, Relaxed);
    }
}

/// This thread enters (`true`) or leaves a submission.
pub(crate) fn submitting(inside: bool) {
    INSIDE.with(|i| i.set(inside));
}

/// The title's D3D12 objects we hand out boxed, made and freed.
static OBJECTS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

pub(crate) fn object_made() {
    OBJECTS[0].fetch_add(1, Relaxed);
}

pub(crate) fn object_gone() {
    OBJECTS[1].fetch_add(1, Relaxed);
}

/// Texture refreshes' waits (microseconds): for another thread's refresh
/// of the same texture, asking the write watch and the guards, for title
/// memory to hold still. Added each second from the frame line's own.
static REFRESH: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// The texture refreshes' waits taken for the frame line, in microseconds.
pub(crate) fn refresh_waits(us: [u64; 3]) {
    for (c, us) in REFRESH.iter().zip(us) {
        c.fetch_add(us, Relaxed);
    }
}

/// What the line reads that others keep: watched ranges of title memory
/// checked (bytes), copy and fill steps of title memory asked for (bytes,
/// steps), the title's threads' waits for pipelines and in frame pacing
/// (microseconds), and the process's page faults so far.
#[derive(Default, Clone, Copy)]
pub(crate) struct Others {
    pub(crate) checked: u64,
    pub(crate) copied: u64,
    pub(crate) copies: u64,
    pub(crate) pipeline_wait_us: u64,
    pub(crate) pace_us: u64,
    pub(crate) faults: u64,
}

/// Everything the line says, since the last one.
#[derive(Default)]
struct Figures {
    card: [[u64; 2]; 8],
    settles: [[[u64; 2]; 4]; 11],
    lock: [u64; 2],
    pipelines: [u64; 6],
    decodes: [u64; 2],
    depth: [u64; 2],
    kernel: [[u64; 6]; 2],
    objects: [u64; 2],
    refresh: [u64; 3],
    others: Others,
}

fn take<const N: usize>(c: &[AtomicU64; N]) -> [u64; N] {
    std::array::from_fn(|i| c[i].swap(0, Relaxed))
}

impl Figures {
    fn take(others: Others) -> Figures {
        Figures {
            card: std::array::from_fn(|i| take(&CARD[i])),
            settles: std::array::from_fn(|i| std::array::from_fn(|j| take(&SETTLES[i][j]))),
            lock: take(&LOCK),
            pipelines: take(&PIPELINES),
            decodes: take(&DECODES),
            depth: take_depth(),
            kernel: std::array::from_fn(|i| take(&KERNEL[i])),
            objects: take(&OBJECTS),
            refresh: take(&REFRESH),
            others,
        }
    }
}

/// The line for `f` over `frames`: fixed words and numbers alone.
fn text(f: &Figures, frames: f64) -> String {
    use std::fmt::Write;
    let per = |n: u64| n as f64 / frames.max(1.0);
    let ms = |us: u64| us as f64 / 1e3 / frames.max(1.0);
    let p = &f.pipelines;
    let mut t = format!(
        "stalls per frame (diagnostics, over {frames:.0} frames since the last line): host pipelines built by the workers (a \
         build that ran, failed or not), for a draw or dispatch that asked before the build began {:.2} ({:.1} ms building, \
         {:.2} of them compiled rather than loaded from the library, {:.2} failed), ahead of any ask {:.2} ({:.1} ms); the \
         title's threads waited for pipelines {:.1} ms; BCPack decode on DirectStorage's workers {:.1} ms summed over them \
         ({:.1} requests); DirectStorage requests waiting to start, running or held back for their page arrays, on all \
         queues: {} at the line, {} at most since the last",
        per(p[0]),
        ms(p[1]),
        per(p[2]),
        per(p[3]),
        per(p[4]),
        ms(p[5]),
        ms(f.others.pipeline_wait_us),
        f.decodes[1] as f64 / 1e6 / frames.max(1.0),
        per(f.decodes[0]),
        f.depth[0],
        f.depth[1]
    );
    // Settles by why, each with what released them.
    let settles: u64 = f.settles.iter().flatten().map(|c| c[0]).sum();
    let settled_us: u64 = f.settles.iter().flatten().map(|c| c[1]).sum();
    let _ = write!(t, "; settles, once each, {:.2} ({:.1} ms waiting in them)", per(settles), ms(settled_us));
    for (why, by) in f.settles.iter().enumerate() {
        let (n, us) = (by.iter().map(|c| c[0]).sum::<u64>(), by.iter().map(|c| c[1]).sum::<u64>());
        if n == 0 {
            continue;
        }
        let _ = write!(
            t,
            ", for {} {:.2} ({:.1} ms; released by its own fence waits {:.2}, with no flight to wait for {:.2}, by a relay \
             landing that fetched pieces of card copies {:.2}, by one that fetched none {:.2})",
            WHY[why],
            per(n),
            ms(us),
            per(by[Release::Fence as usize][0]),
            per(by[Release::Nothing as usize][0]),
            per(by[Release::Fetched as usize][0]),
            per(by[Release::Landing as usize][0])
        );
    }
    let _ = write!(t, "; the submission lock found held {:.2} times ({:.1} ms waiting)", per(f.lock[0]), ms(f.lock[1]));
    let (waits, waited) = (f.card.iter().map(|c| c[0]).sum::<u64>(), f.card.iter().map(|c| c[1]).sum::<u64>());
    let _ = write!(
        t,
        "; waits of our threads for a fence on the card {:.2} ({:.1} ms; they overlap each other and the settles, whose own \
         fence waits are among them: these figures and the settles' do not add)",
        per(waits),
        ms(waited)
    );
    for (site, c) in f.card.iter().enumerate().filter(|(_, c)| c[0] > 0) {
        let _ = write!(t, ", {} {:.2} ({:.1} ms)", SITES[site], per(c[0]), ms(c[1]));
    }
    let [own, asking, still] = f.refresh;
    let _ = write!(
        t,
        "; the title's frame pacing {:.1} ms sleeping; texture refreshes waited {:.1} ms for another thread's, {:.1} ms asking \
         the write watch and the guards, {:.1} ms for title memory to hold still",
        ms(f.others.pace_us),
        ms(own),
        ms(asking),
        ms(still)
    );
    let o = &f.others;
    let [all, inside] = &f.kernel;
    let _ = write!(
        t,
        "; title memory: watched ranges checked {:.1} MB (unchanged bytes among them), copy and fill steps asked for {:.0} \
         ({:.1} MB as recorded, before they run; refreshes of watched ranges and write-backs are not among them); kernel calls \
         about memory at the call sites counted, on all threads (inside submissions): queries {:.0} ({:.0}), protection \
         changes {:.0} ({:.0}), mappings and unmappings {:.0} ({:.0}), reservations, commits and releases {:.0} ({:.0}), \
         write-watch asks {:.0} ({:.0}), reads and writes through the kernel {:.0} ({:.0}); page faults of the whole process \
         {:.0}; the title's D3D12 objects we hand out boxed: made {:.1}, freed at their last release {:.1}",
        o.checked as f64 / 1e6 / frames.max(1.0),
        per(o.copies),
        o.copied as f64 / 1e6 / frames.max(1.0),
        per(all[0]),
        per(inside[0]),
        per(all[1]),
        per(inside[1]),
        per(all[2]),
        per(inside[2]),
        per(all[3]),
        per(inside[3]),
        per(all[4]),
        per(inside[4]),
        per(all[5]),
        per(inside[5]),
        per(o.faults),
        per(f.objects[0]),
        per(f.objects[1]),
    );
    t
}

/// The line, every five seconds after the frame line: what was counted
/// since the last one over `frames`, with `others` (read, not taken, from
/// where they are kept; their page faults the process's total so far).
pub(crate) fn line(frames: f64, others: Others) {
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    let faults = others.faults.saturating_sub(FAULTS.swap(others.faults, Relaxed));
    let figures = Figures::take(Others { faults, ..others });
    tracing::info!("{}", text(&figures, frames));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The line names each kind of wait with its count and ms a frame, says
    /// what released each settle and that the fence waits overlap them,
    /// and leaves out the causes and sites that had none.
    #[test]
    fn the_line_names_each_wait_and_what_released_it() {
        let mut f = Figures {
            pipelines: [10, 50_000, 4, 1, 20, 10_000],
            decodes: [30, 15_000_000],
            depth: [7, 12],
            lock: [20, 4_000],
            kernel: [[100, 50, 20, 10, 40, 300], [60, 30, 0, 0, 40, 200]],
            objects: [8, 6],
            refresh: [1_000, 2_000, 3_000],
            others: Others { checked: 100_000_000, copied: 20_000_000, copies: 300, pipeline_wait_us: 9_000, pace_us: 7_000, faults: 1_000 },
            ..Figures::default()
        };
        f.settles[3][Release::Fence as usize] = [10, 20_000];
        f.settles[3][Release::Fetched as usize] = [5, 10_000];
        f.settles[9][Release::Nothing as usize] = [2, 0];
        f.card[Site::Submission as usize] = [10, 30_000];
        let t = text(&f, 10.0);
        for piece in [
            "stalls per frame (diagnostics, over 10 frames since the last line): host pipelines built by the workers (a build that ran, failed or not), for a draw or dispatch that asked before the build began 1.00 (5.0 ms building, 0.40 of them compiled rather than loaded from the library, 0.10 failed), ahead of any ask 2.00 (1.0 ms)",
            "BCPack decode on DirectStorage's workers 1.5 ms summed over them (3.0 requests); DirectStorage requests waiting to start, running or held back for their page arrays, on all queues: 7 at the line, 12 at most since the last",
            "; settles, once each, 1.70 (3.0 ms waiting in them), for the card had run them 1.50 (3.0 ms; released by its own fence waits 1.00, with no flight to wait for 0.00, by a relay landing that fetched pieces of card copies 0.50, by one that fetched none 0.00), for bottom-of-pipe writes 0.20 (0.0 ms; released by its own fence waits 0.00, with no flight to wait for 0.20,",
            "; the submission lock found held 2.00 times (0.4 ms waiting)",
            "; waits of our threads for a fence on the card 1.00 (3.0 ms; they overlap each other and the settles, whose own fence waits are among them: these figures and the settles' do not add), synchronous submissions 1.00 (3.0 ms);",
            "watched ranges checked 10.0 MB (unchanged bytes among them), copy and fill steps asked for 30 (2.0 MB as recorded, before they run;",
            "queries 10 (6), protection changes 5 (3), mappings and unmappings 2 (0), reservations, commits and releases 1 (0), write-watch asks 4 (4), reads and writes through the kernel 30 (20); page faults of the whole process 100; the title's D3D12 objects we hand out boxed: made 0.8, freed at their last release 0.6",
        ] {
            assert!(t.contains(piece), "missing {piece:?} in {t}");
        }
        assert!(!t.contains("for a wait for a memory value"), "a cause without settles: {t}");
        assert!(!t.contains("the relay's thread"), "a site without waits: {t}");
        assert!(!t.contains('\n'), "one line: {t}");
    }

    /// A wait for the relay is said to be released by its last landing, as
    /// that landing fetched pieces or not.
    #[test]
    fn a_landing_says_whether_it_fetched() {
        assert_eq!(release_of(7 << 1 | 1), Release::Fetched);
        assert_eq!(release_of(8 << 1), Release::Landing);
    }

    /// The depth at the line is the depth now, not zero, and the most
    /// starts again from it (Sol's review: the line swapped the gauge to
    /// zero and said 0 with requests outstanding).
    #[test]
    fn the_depth_at_the_line_is_the_depth_now() {
        storage_depth(9);
        storage_depth(4);
        assert_eq!(take_depth(), [4, 9]);
        assert_eq!(take_depth(), [4, 4], "nothing since: the depth stands, and it is the most");
    }

    /// Kernel calls inside a submission count twice: on all threads and
    /// inside; outside only once.
    #[test]
    fn kernel_calls_inside_submissions_are_counted_apart() {
        let before = [KERNEL[0][Kernel::Protect as usize].load(Relaxed), KERNEL[1][Kernel::Protect as usize].load(Relaxed)];
        kernel(Kernel::Protect);
        submitting(true);
        kernel(Kernel::Protect);
        submitting(false);
        let after = [KERNEL[0][Kernel::Protect as usize].load(Relaxed), KERNEL[1][Kernel::Protect as usize].load(Relaxed)];
        // Other threads may count too: at least these.
        assert!(after[0] >= before[0] + 2 && after[1] > before[1], "{before:?} {after:?}");
    }

    /// A lock another thread holds is timed; a free one is not.
    #[test]
    fn a_held_lock_is_timed() {
        let m = std::sync::Arc::new(Mutex::new(()));
        let before = LOCK[0].load(Relaxed);
        drop(lock(&m));
        let held = m.lock().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let m2 = m.clone();
        let waiter = std::thread::spawn(move || {
            tx.send(()).unwrap();
            drop(lock(&m2));
        });
        rx.recv().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        drop(held);
        waiter.join().unwrap();
        assert!(LOCK[0].load(Relaxed) > before, "the held lock's wait was not counted");
    }
}
