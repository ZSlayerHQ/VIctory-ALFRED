//! The frame budget: one scope type that says which phase a thread is in.
//!
//! This is the runtime's only phase timer. It is deliberately small: the judges of where a
//! thread's time goes are outside tools (a system trace, a profiler); these figures are the
//! cheap view that is always on in a diagnostics build, for the live switches, and they are to
//! be shown to agree with those tools before they are believed.
//!
//! # What it measures
//!
//! Every thread has a clock of its own. A thread is in exactly one [`Phase`] at any moment: a
//! scope that begins inside another suspends it until it ends, so a phase's time never holds a
//! nested phase's, and the time a thread spends between two moments is the sum of its phases'
//! by construction (each interval between two clock reads is added to one phase).
//!
//! The thread that presents is the budget's thread: on the title this is the thread that
//! submits its lists. Its frame runs from one present's count ([`frame`]) to the next, and
//! the frame is the sum of every phase, exactly, in nanoseconds of one clock. That sum proves
//! nothing by itself (it holds whatever the scopes are); what makes `title` the title's own
//! time is that every call the title makes into this runtime opens a scope:
//!
//! - [`entry`] at the top of every function the title calls through a table of ours (the
//!   lists', the device's, the descriptor heaps', the resources', the fences', the queues' and
//!   the other objects' methods) and of every export and object method of `victory-hle`
//!   ([`crate::hle_call`]): `list_calls`, `object_calls`, `hle_calls`. A fixture reads the
//!   sources and fails if a function named in such a table opens none.
//! - [`phase`] for what a queue call does (a submission, a present, a signal) and for our own
//!   threads' work; `unaccounted` is the time inside a queue call that no named phase covers.
//!
//! So `title` is the time outside every call of ours. Two figures let an outside trace check
//! that independently of the sum: the calls entered from the title a frame, and the largest
//! single stretch of `title` between two of them ([`Taken::calls`], [`Taken::largest_gap`]).
//!
//! Work on other threads never enters that budget: a thread that has not presented adds what
//! its scopes measured to [`Others`], by phase, summed over all such threads. Those figures
//! overlap the budget's in time and are not parts of the frame.
//!
//! # The clocks
//!
//! Wall time is `std::time::Instant` (the performance counter; 100 ns ticks on this
//! platform). Processor time is `QueryThreadCycleTime` for the calling thread: the cycles the
//! scheduler charged to the thread, in user and kernel mode, in units of the processor's
//! timestamp counter, exact at the moment of the call (not sampled at the timer interrupt as
//! `GetThreadTimes` is, whose resolution is the 15.6 ms tick). Time in interrupts and deferred
//! calls is not charged to a thread by Windows, so a phase's wall time less its processor time
//! is waiting, being preempted, and interrupts. Cycles are turned into time by the counter's
//! rate, measured against the wall clock over the interval reported ([`Taken::cycles_per_ns`]).
//!
//! The title's calls are too many to read the cycle counter at each (tens of thousands of
//! list calls a frame, a kernel call each time), so an [`entry`] scope reads the wall clock
//! alone. The processor time of `title`, `list_calls`, `object_calls` and `hle_calls` is
//! therefore known only together ([`Taken::entry_cycles`]), never one by one: none of them
//! has a processor time of its own, and `title`'s is not said to be what it is not.
//!
//! # Cost
//!
//! A [`phase`] scope reads each clock twice (when it begins and when it ends): four reads. An
//! [`entry`] scope reads the wall clock twice, and nothing when the thread is already inside
//! a scope. No allocation, no lock, thread-local cells only. A frame's end adds two reads and
//! moves the arrays. A thread that has not presented does one relaxed add per phase it used
//! when its outermost phase scope ends, or after a millisecond of entry scopes. [`lock`] opens
//! a scope only when the lock is held by another thread. The scopes and calls a frame entered
//! are counted, so the cost is stated by the figures themselves.
//!
//! # Outside tools
//!
//! [`begin`] and [`end`] are the single place a [`phase`] scope starts and stops. A profiler
//! zone (Tracy) and a Direct3D 12 debug event on the queue (`BeginEvent`/`EndEvent`, which PIX,
//! RenderDoc and the vendors' GPU traces show) attach there, by [`NAMES`], behind a feature
//! that does not exist yet; nothing is emitted today.

/// What a thread is doing. The names ([`NAMES`]) are stable: the stats file, the log and any
/// outside tool's zones use them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
#[cfg_attr(not(all(windows, feature = "diagnostics")), allow(dead_code))]
pub(crate) enum Phase {
    /// Outside every scope: the title's own code on this thread.
    Title = 0,
    /// Inside a list method the title called ([`entry`]).
    ListCalls = 1,
    /// Inside a queue call of ours, in no named phase.
    Unaccounted = 2,
    /// Acquiring the submission lock while another thread holds it.
    SubmissionLockWait = 3,
    /// Settling flights in the air: its own work outside the waits and the copy-back in it.
    Settle = 4,
    /// Waiting for the relay's thread to land flights.
    RelayWait = 5,
    /// This thread's own wait for a fence on one of the card's queues (not time on the card).
    CardWait = 6,
    /// The sweep of idle host objects.
    Sweep = 7,
    /// Making the descriptor slots that waited for a texture's host copy.
    Slots = 8,
    /// Bringing host copies of title memory up to date: copies, fills, write-watch asks.
    Refresh = 9,
    /// Refreshing textures: hashing and detiling.
    Textures = 10,
    /// Waiting for the helper pool to finish the jobs this thread gave it.
    PoolWait = 11,
    /// Snapshots of the ranges shaders may write.
    Snapshots = 12,
    /// Recording lists of ours (the card copies' hash and compare lists, the present's copy).
    OwnLists = 13,
    /// Handing lists to the card's queue (`ExecuteCommandLists` and the signal behind them).
    Send = 14,
    /// Copying back what shaders wrote: fetches, merges, the hand-back.
    CopyBack = 15,
    /// The swap chain's `Present`.
    PresentCall = 16,
    /// Our own reporting at a present: the log lines and the stats file.
    Report = 17,
    /// A job of the helper pool (its workers' threads).
    PoolWork = 18,
    /// The relay's thread landing flights, outside its waits.
    RelayLand = 19,
    /// Inside a method of another object of ours the title called: the device, a descriptor
    /// heap, a resource, a fence, a queue's lesser methods and the rest ([`entry`]).
    ObjectCalls = 20,
    /// Inside an export or an object method of `victory-hle` the title called
    /// ([`crate::hle_call`]).
    HleCalls = 21,
    /// Acquiring one of the queues' locks while another thread holds it: the line of flights,
    /// the queues that wait and what they hold ([`lock`]).
    QueueLockWait = 22,
    /// The title's frame pacing wait (`crate::frame`'s wait).
    PaceWait = 23,
}

/// How many phases there are.
#[cfg_attr(not(all(windows, feature = "diagnostics")), allow(dead_code))]
pub(crate) const PHASES: usize = 24;

/// The phases' names, by number.
#[cfg_attr(not(all(windows, feature = "diagnostics")), allow(dead_code))]
pub(crate) const NAMES: [&str; PHASES] = [
    "title",
    "list_calls",
    "unaccounted",
    "submission_lock_wait",
    "settle",
    "relay_wait",
    "card_wait",
    "sweep",
    "slots",
    "refresh",
    "textures",
    "pool_wait",
    "snapshots",
    "own_lists",
    "send",
    "copy_back",
    "present_call",
    "report",
    "pool_work",
    "relay_land",
    "object_calls",
    "hle_calls",
    "queue_lock_wait",
    "pace_wait",
];

/// The phases an [`entry`] scope measures, and `title` between them: the wall clock alone.
/// Their processor time is known only together.
#[cfg_attr(not(all(windows, feature = "diagnostics")), allow(dead_code))]
pub(crate) const ENTRY_PHASES: [Phase; 4] = [Phase::Title, Phase::ListCalls, Phase::ObjectCalls, Phase::HleCalls];

/// A phase under way on this thread; it ends when this is dropped, on the thread that began
/// it (it cannot be sent to another). Nothing at all in a build without diagnostics.
#[must_use = "the phase ends when the scope is dropped"]
pub struct Scope {
    /// 0: nothing to end; 1: a [`phase`] scope; 2: an [`entry`] scope.
    #[cfg(all(windows, feature = "diagnostics"))]
    kind: u8,
    /// A scope is its thread's: not `Send`.
    not_send: std::marker::PhantomData<*const ()>,
}

impl Scope {
    /// Ends the phase here: the scope is dropped (in a build without diagnostics it is nothing,
    /// and a `drop` of nothing is a lint).
    #[inline(always)]
    pub(crate) fn end(self) {}

    #[cfg(not(all(windows, feature = "diagnostics")))]
    #[inline(always)]
    const fn none() -> Scope {
        Scope { not_send: std::marker::PhantomData }
    }
}

/// Begins `phase` on this thread, suspending the phase it is in.
#[cfg(not(all(windows, feature = "diagnostics")))]
#[inline(always)]
pub(crate) fn phase(_: Phase) -> Scope {
    Scope::none()
}

/// The title called into the runtime on this thread: begins `phase` unless the thread is
/// already inside a scope (a call of ours from inside our own work is that work's).
#[cfg(not(all(windows, feature = "diagnostics")))]
#[inline(always)]
pub(crate) fn entry(_: Phase) -> Scope {
    Scope::none()
}

/// `lock`, its wait counted as `phase` when another thread holds it.
#[cfg(not(all(windows, feature = "diagnostics")))]
#[inline(always)]
pub(crate) fn lock<T>(lock: &std::sync::Mutex<T>, _: Phase) -> std::sync::MutexGuard<'_, T> {
    lock.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) use on::{Others, Taken, entry, frame, lock, phase, take, take_others};
#[cfg(all(test, windows, feature = "diagnostics"))]
pub(crate) use on::{clocks, hand, peek_open};

#[cfg(all(windows, feature = "diagnostics"))]
mod on {
    use super::{PHASES, Phase, Scope};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::time::Instant;

    /// Scopes nested deeper than this still count: their time stays with the phase around
    /// them.
    const DEPTH: usize = 16;

    struct Clock {
        /// The last moment charged ([`now`]); 0 before the first.
        mark: Cell<u64>,
        /// The thread's cycles when they were last read.
        cycles: Cell<u64>,
        depth: Cell<usize>,
        stack: [Cell<u8>; DEPTH],
        /// The frame (or, on a thread that has not presented, the time since it last sent)
        /// under way.
        wall: [Cell<u64>; PHASES],
        cpu: [Cell<u64>; PHASES],
        scopes: Cell<u64>,
        /// Calls entered from outside every scope, and the longest time outside between two.
        calls: Cell<u64>,
        gap: Cell<u64>,
        /// Whether this thread has presented: its time is the budget's.
        presents: Cell<bool>,
        /// When the frame under way began.
        frame_from: Cell<u64>,
        /// The frames ended since the last [`take`], and their sums.
        frames: Cell<u64>,
        closed_ns: Cell<u64>,
        closed_wall: [Cell<u64>; PHASES],
        closed_cpu: [Cell<u64>; PHASES],
        closed_scopes: Cell<u64>,
        closed_calls: Cell<u64>,
        closed_gap: Cell<u64>,
        /// The timestamp counter and the wall clock at the last [`take`].
        rate_from: Cell<(u64, u64)>,
        /// Entry scopes' time on a thread that has not presented, not yet sent to [`OTHERS`].
        unsent: Cell<u64>,
    }

    thread_local! {
        static CLOCK: Clock = const {
            Clock {
                mark: Cell::new(0),
                cycles: Cell::new(0),
                depth: Cell::new(0),
                stack: [const { Cell::new(0) }; DEPTH],
                wall: [const { Cell::new(0) }; PHASES],
                cpu: [const { Cell::new(0) }; PHASES],
                scopes: Cell::new(0),
                calls: Cell::new(0),
                gap: Cell::new(0),
                presents: Cell::new(false),
                frame_from: Cell::new(0),
                frames: Cell::new(0),
                closed_ns: Cell::new(0),
                closed_wall: [const { Cell::new(0) }; PHASES],
                closed_cpu: [const { Cell::new(0) }; PHASES],
                closed_scopes: Cell::new(0),
                closed_calls: Cell::new(0),
                closed_gap: Cell::new(0),
                rate_from: Cell::new((0, 0)),
                unsent: Cell::new(0),
            }
        };
    }

    /// By phase: wall nanoseconds and cycles of the threads that have not presented, since
    /// the last [`take_others`]; and the scopes they entered.
    static OTHERS: [[AtomicU64; 2]; PHASES] = [const { [const { AtomicU64::new(0) }; 2] }; PHASES];
    static OTHER_SCOPES: AtomicU64 = AtomicU64::new(0);

    /// Nanoseconds since the first call, never 0.
    fn now() -> u64 {
        #[cfg(test)]
        if let Some(at) = hand::wall() {
            return at;
        }
        static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        Instant::now().saturating_duration_since(*EPOCH.get_or_init(Instant::now)).as_nanos() as u64 + 1
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread() -> *mut core::ffi::c_void;
        fn QueryThreadCycleTime(thread: *mut core::ffi::c_void, cycles: *mut u64) -> i32;
    }

    /// The cycles charged to this thread so far.
    fn cycles() -> u64 {
        #[cfg(test)]
        if let Some(cycles) = hand::cycles() {
            return cycles;
        }
        let mut cycles = 0;
        // SAFETY: the calling thread's pseudo handle and a valid out pointer.
        unsafe { QueryThreadCycleTime(GetCurrentThread(), &mut cycles) };
        cycles
    }

    /// The processor's timestamp counter, the unit of [`cycles`].
    fn counter() -> u64 {
        #[cfg(test)]
        if let Some(ticks) = hand::counter() {
            return ticks;
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: every x86-64 processor has the instruction.
        return unsafe { core::arch::x86_64::_rdtsc() };
        #[cfg(not(target_arch = "x86_64"))]
        0
    }

    /// The phase this thread is in.
    fn current(c: &Clock) -> usize {
        match c.depth.get() {
            0 => Phase::Title as usize,
            depth => c.stack[depth.min(DEPTH) - 1].get() as usize,
        }
    }

    /// Charges the wall time since the last mark to the phase the thread is in, and marks
    /// now: the moment, and the time charged.
    fn lap_wall(c: &Clock) -> (u64, u64) {
        let at = now();
        let spent = if c.mark.get() != 0 { at.saturating_sub(c.mark.get()) } else { 0 };
        let phase = current(c);
        c.wall[phase].set(c.wall[phase].get() + spent);
        c.mark.set(at);
        (at, spent)
    }

    /// [`lap_wall`], and the cycles since they were last read go to the same phase.
    fn lap(c: &Clock) -> u64 {
        let spent = cycles();
        let phase = current(c);
        if c.mark.get() != 0 {
            c.cpu[phase].set(c.cpu[phase].get() + spent.saturating_sub(c.cycles.get()));
        }
        c.cycles.set(spent);
        lap_wall(c).0
    }

    /// The single place a [`phase`] scope begins: where a profiler zone and a debug event on
    /// the queue would begin too (see the module's text). Nothing is emitted today.
    #[inline(always)]
    fn begin(_: Phase) {}

    /// The single place a [`phase`] scope ends ([`begin`]).
    #[inline(always)]
    fn end(_: usize) {}

    fn push(c: &Clock, phase: Phase) {
        c.scopes.set(c.scopes.get() + 1);
        let depth = c.depth.get();
        if depth < DEPTH {
            c.stack[depth].set(phase as u8);
        }
        c.depth.set(depth + 1);
    }

    /// Begins `phase` on this thread, suspending the phase it is in.
    pub(crate) fn phase(phase: Phase) -> Scope {
        begin(phase);
        CLOCK.with(|c| {
            lap(c);
            push(c, phase);
        });
        Scope { kind: 1, not_send: std::marker::PhantomData }
    }

    /// The title called into the runtime on this thread: begins `phase` unless the thread is
    /// already inside a scope (a call of ours from inside our own work is that work's). The
    /// wall clock alone is read.
    pub(crate) fn entry(phase: Phase) -> Scope {
        let kind = CLOCK.with(|c| {
            if c.depth.get() != 0 {
                return 0;
            }
            // The time since the last scope ended was the title's.
            let (_, outside) = lap_wall(c);
            c.calls.set(c.calls.get() + 1);
            c.gap.set(c.gap.get().max(outside));
            push(c, phase);
            2
        });
        Scope { kind, not_send: std::marker::PhantomData }
    }

    impl Drop for Scope {
        fn drop(&mut self) {
            let kind = self.kind;
            if kind == 0 {
                return;
            }
            // A thread being torn down has no clock left to charge.
            let _ = CLOCK.try_with(|c| {
                let ended = current(c);
                let spent = if kind == 1 {
                    end(ended);
                    lap(c);
                    0
                } else {
                    lap_wall(c).1
                };
                let depth = c.depth.get().saturating_sub(1);
                c.depth.set(depth);
                if kind == 2 && ended == Phase::ListCalls as usize {
                    // The copies' "list calls" figures, from the same two reads.
                    crate::copystat::list_call(spent);
                }
                if depth == 0 && !c.presents.get() {
                    // An entry scope's time is sent a millisecond at a time.
                    let unsent = c.unsent.get() + spent;
                    if kind == 1 || unsent >= 1_000_000 {
                        c.unsent.set(0);
                        send(c);
                    } else {
                        c.unsent.set(unsent);
                    }
                }
            });
        }
    }

    /// A thread that has not presented: what its scopes measured goes to [`OTHERS`]. Its time
    /// outside every scope is nobody's.
    fn send(c: &Clock) {
        for phase in 0..PHASES {
            let (wall, cpu) = (c.wall[phase].replace(0), c.cpu[phase].replace(0));
            if phase != Phase::Title as usize && wall | cpu != 0 {
                OTHERS[phase][0].fetch_add(wall, Relaxed);
                OTHERS[phase][1].fetch_add(cpu, Relaxed);
            }
        }
        OTHER_SCOPES.fetch_add(c.scopes.replace(0), Relaxed);
        c.calls.set(0);
        c.gap.set(0);
    }

    /// `lock`, its wait counted as `phase` when another thread holds it; no scope, and no
    /// clock read, when it is free.
    pub(crate) fn lock<T>(lock: &std::sync::Mutex<T>, wait: Phase) -> std::sync::MutexGuard<'_, T> {
        match lock.try_lock() {
            Ok(held) => held,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                let _waiting = phase(wait);
                lock.lock().unwrap_or_else(|e| e.into_inner())
            }
        }
    }

    /// The present counts a frame on this thread: the frame under way ends here and the next
    /// begins. This thread is the budget's from now on; what it measured before its first
    /// frame began is dropped. The frame's length in nanoseconds, from the second call on.
    pub(crate) fn frame() -> Option<u64> {
        CLOCK.with(|c| {
            let at = lap(c);
            let first = !c.presents.replace(true);
            let from = c.frame_from.replace(at);
            if first {
                for phase in 0..PHASES {
                    c.wall[phase].set(0);
                    c.cpu[phase].set(0);
                }
                c.scopes.set(0);
                c.calls.set(0);
                c.gap.set(0);
                c.rate_from.set((counter(), at));
                return None;
            }
            for phase in 0..PHASES {
                c.closed_wall[phase].set(c.closed_wall[phase].get() + c.wall[phase].replace(0));
                c.closed_cpu[phase].set(c.closed_cpu[phase].get() + c.cpu[phase].replace(0));
            }
            c.closed_scopes.set(c.closed_scopes.get() + c.scopes.replace(0));
            c.closed_calls.set(c.closed_calls.get() + c.calls.replace(0));
            c.closed_gap.set(c.closed_gap.get().max(c.gap.replace(0)));
            let length = at.saturating_sub(from);
            c.closed_ns.set(c.closed_ns.get() + length);
            c.frames.set(c.frames.get() + 1);
            Some(length)
        })
    }

    /// The presenting thread's frames since the last take.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct Taken {
        pub(crate) frames: u64,
        /// From the first of those frames' beginning to the last one's end, in nanoseconds:
        /// the sum of `wall`, exactly.
        pub(crate) ns: u64,
        /// By phase: nanoseconds, and cycles of the thread (the entry phases' cycles are
        /// meaningful only together, [`Taken::entry_cycles`]).
        pub(crate) wall: [u64; PHASES],
        pub(crate) cpu: [u64; PHASES],
        /// Scopes entered, of both kinds.
        pub(crate) scopes: u64,
        /// Calls the title made into the runtime from outside every scope ([`entry`]).
        pub(crate) calls: u64,
        /// The longest single stretch outside every scope before such a call, in nanoseconds:
        /// the title's own code between two calls of ours, never a sum.
        pub(crate) largest_gap: u64,
        /// The timestamp counter's rate over the interval since the last take (0: unknown).
        pub(crate) cycles_per_ns: f64,
    }

    impl Taken {
        /// The cycles of `title` and the title's calls into the runtime, together: the only
        /// way they are known.
        pub(crate) fn entry_cycles(&self) -> u64 {
            super::ENTRY_PHASES.iter().map(|&phase| self.cpu[phase as usize]).sum()
        }
    }

    #[cfg(test)]
    impl Taken {
        /// The phases' sum: `ns`, by construction.
        pub(crate) fn sum(&self) -> u64 {
            self.wall.iter().sum()
        }

        /// A phase's processor time in nanoseconds; none where the rate is unknown, and none
        /// for `title` and the title's calls, which have none of their own.
        pub(crate) fn cpu_ns(&self, phase: usize) -> Option<f64> {
            let entry = super::ENTRY_PHASES.iter().any(|&p| p as usize == phase);
            (self.cycles_per_ns > 0.0 && !entry).then(|| self.cpu[phase] as f64 / self.cycles_per_ns)
        }

        /// The processor time of `title` and the title's calls together, in nanoseconds.
        pub(crate) fn entry_cpu_ns(&self) -> Option<f64> {
            (self.cycles_per_ns > 0.0).then(|| self.entry_cycles() as f64 / self.cycles_per_ns)
        }
    }

    /// Takes this thread's ended frames ([`frame`]).
    pub(crate) fn take() -> Taken {
        CLOCK.with(|c| {
            let (counter, at) = (counter(), now());
            let (counter_from, from) = c.rate_from.replace((counter, at));
            let cycles_per_ns = if at > from && counter > counter_from && from != 0 {
                (counter - counter_from) as f64 / (at - from) as f64
            } else {
                0.0
            };
            Taken {
                frames: c.frames.replace(0),
                ns: c.closed_ns.replace(0),
                wall: std::array::from_fn(|phase| c.closed_wall[phase].replace(0)),
                cpu: std::array::from_fn(|phase| c.closed_cpu[phase].replace(0)),
                scopes: c.closed_scopes.replace(0),
                calls: c.closed_calls.replace(0),
                largest_gap: c.closed_gap.replace(0),
                cycles_per_ns,
            }
        })
    }

    /// Tests: the frame under way on this thread, by phase, in nanoseconds (not taken).
    #[cfg(test)]
    pub(crate) fn peek_open() -> [u64; PHASES] {
        CLOCK.with(|c| {
            lap(c);
            std::array::from_fn(|phase| c.wall[phase].get())
        })
    }

    /// What the threads that have not presented measured since the last take: by phase,
    /// wall nanoseconds and cycles, summed over those threads; and their scopes.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct Others {
        pub(crate) wall: [u64; PHASES],
        pub(crate) cpu: [u64; PHASES],
        pub(crate) scopes: u64,
    }

    /// Takes the other threads' figures.
    pub(crate) fn take_others() -> Others {
        Others {
            wall: std::array::from_fn(|phase| OTHERS[phase][0].swap(0, Relaxed)),
            cpu: std::array::from_fn(|phase| OTHERS[phase][1].swap(0, Relaxed)),
            scopes: OTHER_SCOPES.swap(0, Relaxed),
        }
    }

    /// Tests: the three clocks a scope and a take read, read by the test itself: the wall
    /// clock's nanoseconds ([`now`]), this thread's cycles ([`cycles`]) and the timestamp
    /// counter ([`counter`]); the hand clock's where one is set on this thread.
    #[cfg(test)]
    pub(crate) fn clocks() -> (u64, u64, u64) {
        (now(), cycles(), counter())
    }

    /// Tests: a clock the test moves by hand, in place of the real ones on the threads it is
    /// set on, so that what a scope books is exact arithmetic and not the machine's speed or
    /// load. Its wall clock is shared by the threads it is set on, as the real one is; its
    /// processor time is each thread's own, as the real cycles are. Its timestamp counter runs
    /// at [`hand::RATE`] cycles a nanosecond of its wall clock. [`now`], [`cycles`] and
    /// [`counter`] look here first in the tests' build only: outside it this does not exist,
    /// and those functions are what they were.
    #[cfg(test)]
    pub(crate) mod hand {
        use std::cell::Cell;
        use std::sync::atomic::{AtomicU64, Ordering::SeqCst};

        /// The hand clock's cycles a nanosecond (a 3 GHz counter).
        pub(crate) const RATE: u64 = 3;

        /// Nanoseconds in a millisecond.
        const MS: u64 = 1_000_000;

        /// A wall clock moved by hand, and how many times it was read.
        pub(crate) struct Wall {
            ns: AtomicU64,
            reads: AtomicU64,
        }

        impl Wall {
            /// `ms` milliseconds pass on this wall clock, for every thread it is set on.
            pub(crate) fn advance(&self, ms: u64) {
                self.ns.fetch_add(ms * MS, SeqCst);
            }

            /// How many times this wall clock was read, by every thread it is set on. A read
            /// is counted after its value is taken: a thread that sees the count has the read.
            pub(crate) fn reads(&self) -> u64 {
                self.reads.load(SeqCst)
            }
        }

        thread_local! {
            static WALL: Cell<Option<&'static Wall>> = const { Cell::new(None) };
            static CYCLES: Cell<u64> = const { Cell::new(0) };
            static CYCLE_READS: Cell<u64> = const { Cell::new(0) };
        }

        /// Sets a new hand clock on this thread, before its first scope: its wall clock at a
        /// second (never 0, which a clock's mark reserves), its processor time at 0. The wall
        /// clock is returned for other threads to share ([`share`]); it lives to the process's
        /// end (a few bytes a test).
        pub(crate) fn set() -> &'static Wall {
            let wall: &'static Wall = Box::leak(Box::new(Wall { ns: AtomicU64::new(1_000 * MS), reads: AtomicU64::new(0) }));
            share(wall);
            wall
        }

        /// Sets `wall` on this thread, before its first scope, with a processor time of its
        /// own at 0.
        pub(crate) fn share(wall: &'static Wall) {
            WALL.with(|w| w.set(Some(wall)));
            CYCLES.with(|c| c.set(0));
            CYCLE_READS.with(|c| c.set(0));
        }

        /// This thread is off the processor for `ms` milliseconds (a sleep, a wait): its wall
        /// clock moves, its processor time does not.
        pub(crate) fn advance_wall(ms: u64) {
            WALL.with(Cell::get).expect("a hand clock set on this thread").advance(ms);
        }

        /// This thread is charged `ms` milliseconds of processor time, its wall clock unmoved
        /// ([`compute`] moves both).
        pub(crate) fn advance_cpu(ms: u64) {
            WALL.with(Cell::get).expect("a hand clock set on this thread");
            CYCLES.with(|c| c.set(c.get() + ms * MS * RATE));
        }

        /// This thread computes for `ms` milliseconds: wall time and processor time alike.
        pub(crate) fn compute(ms: u64) {
            advance_wall(ms);
            advance_cpu(ms);
        }

        /// The reads so far of this thread's wall clock (by every thread it is set on) and of
        /// this thread's processor time.
        pub(crate) fn reads() -> (u64, u64) {
            let wall = WALL.with(Cell::get).expect("a hand clock set on this thread");
            (wall.reads(), CYCLE_READS.with(Cell::get))
        }

        /// [`super::now`] on a thread with a hand clock.
        pub(super) fn wall() -> Option<u64> {
            let wall = WALL.with(Cell::get)?;
            let at = wall.ns.load(SeqCst);
            wall.reads.fetch_add(1, SeqCst);
            Some(at)
        }

        /// [`super::cycles`] on a thread with a hand clock.
        pub(super) fn cycles() -> Option<u64> {
            WALL.with(Cell::get)?;
            CYCLE_READS.with(|r| r.set(r.get() + 1));
            Some(CYCLES.with(Cell::get))
        }

        /// [`super::counter`] on a thread with a hand clock: not counted as a read.
        pub(super) fn counter() -> Option<u64> {
            WALL.with(Cell::get).map(|wall| wall.ns.load(SeqCst) * RATE)
        }
    }
}

#[cfg(all(test, windows, feature = "diagnostics"))]
mod tests {
    //! What a scope books is tested on a hand clock ([`hand`]): exact figures, whatever the
    //! machine's speed or load. The real clocks are tested apart, by bounds the test reads
    //! itself around a scope ([`the_real_clocks_bracket_what_a_scope_books`]).
    use super::*;
    use std::time::Duration;

    /// Nanoseconds in a millisecond.
    const MS: u64 = 1_000_000;

    fn ms(ns: u64) -> f64 {
        ns as f64 / 1e6
    }

    /// Burns the processor until this thread was charged at least `cycles` cycles (the real
    /// clock's): processor time, not wall time, so that load lengthens it and never shortens it.
    fn spin(cycles: u64) {
        let from = clocks().1;
        let mut n = 0u64;
        while clocks().1 - from < cycles {
            for _ in 0..10_000 {
                n = std::hint::black_box(n.wrapping_mul(6364136223846793005).wrapping_add(1));
            }
        }
    }

    /// The names are one a phase, snake_case, and no two alike: outside tools key on them.
    #[test]
    fn the_names_are_stable_words() {
        for (i, name) in NAMES.iter().enumerate() {
            assert!(name.bytes().all(|b| b.is_ascii_lowercase() || b == b'_'), "{name}");
            assert!(!NAMES[..i].contains(name), "{name} twice");
        }
        assert_eq!(NAMES[Phase::RelayLand as usize], "relay_land");
        assert_eq!(NAMES[Phase::Unaccounted as usize], "unaccounted");
    }

    /// A synthetic frame with nested phases, on a hand clock: the phases, the title's time and
    /// the unaccounted time add up to the frame exactly, and each phase holds its own time and
    /// not the phase nested in it, to the nanosecond.
    #[test]
    fn a_frame_s_phases_sum_to_the_frame() {
        std::thread::spawn(|| {
            hand::set();
            assert_eq!(frame(), None, "the first present begins the first frame");
            hand::advance_wall(30); // the title
            {
                let _call = phase(Phase::Unaccounted);
                hand::advance_wall(20);
                {
                    let _refresh = phase(Phase::Refresh);
                    hand::advance_wall(40);
                    {
                        let _textures = phase(Phase::Textures);
                        hand::advance_wall(60);
                    }
                    hand::advance_wall(10);
                }
            }
            let length = frame().expect("the second present ends a frame");
            let taken = take();
            assert_eq!(taken.frames, 1);
            assert_eq!(taken.ns, length);
            assert_eq!(length, 160 * MS, "the frame is the 160 ms the clock moved: {} ms", ms(length));
            assert_eq!(taken.sum(), taken.ns, "the phases do not add up to the frame: {taken:?}");
            let of = |p: Phase| taken.wall[p as usize];
            assert_eq!(of(Phase::Title), 30 * MS, "title {} ms for 30", ms(of(Phase::Title)));
            assert_eq!(of(Phase::Unaccounted), 20 * MS, "unaccounted {} ms for 20", ms(of(Phase::Unaccounted)));
            assert_eq!(of(Phase::Refresh), 50 * MS, "refresh without the textures in it {} ms for 50", ms(of(Phase::Refresh)));
            assert_eq!(of(Phase::Textures), 60 * MS, "textures {} ms for 60", ms(of(Phase::Textures)));
            assert_eq!(taken.scopes, 3);
            // A second take has nothing: no frame is counted twice.
            let again = take();
            assert_eq!((again.frames, again.ns, again.sum()), (0, 0, 0));
        })
        .join()
        .unwrap();
    }

    /// A wait inside a phase is counted once, on a hand clock: in the waiting phase's wall
    /// time, in no other phase's, and not as processor time; a phase that computes shows its
    /// processor time, exactly.
    #[test]
    fn a_wait_inside_a_phase_is_counted_once() {
        std::thread::spawn(|| {
            hand::set();
            frame();
            {
                let _refresh = phase(Phase::Refresh);
                hand::compute(60);
                {
                    let _wait = phase(Phase::CardWait);
                    hand::advance_wall(80);
                }
            }
            frame();
            let taken = take();
            assert_eq!(taken.sum(), taken.ns);
            let (refresh, wait) = (Phase::Refresh as usize, Phase::CardWait as usize);
            assert_eq!(taken.wall[wait], 80 * MS, "the wait {} ms for 80", ms(taken.wall[wait]));
            assert_eq!(taken.wall[refresh], 60 * MS, "the phase around it holds the wait: {} ms for 60", ms(taken.wall[refresh]));
            assert_eq!(taken.cycles_per_ns, hand::RATE as f64, "the counter's rate");
            let cpu = |phase| taken.cpu_ns(phase).expect("the counter's rate") / 1e6;
            assert_eq!(cpu(wait), 0.0, "a sleep counted as {} ms of processor time", cpu(wait));
            assert_eq!(cpu(refresh), 60.0, "60 ms of computing counted as {} ms of processor time", cpu(refresh));
        })
        .join()
        .unwrap();
    }

    /// Work on a helper thread does not enter the presenting thread's budget, on a hand clock
    /// the two threads share: the helper's phase goes to the other threads' figures, and the
    /// presenting thread's frame holds only its own wait for it. The helper begins its job once
    /// the wait has begun (a channel), so the wait holds the job's time exactly.
    #[test]
    fn a_helper_thread_s_work_is_not_in_the_budget() {
        std::thread::spawn(|| {
            let wall = hand::set();
            frame();
            let (give, given) = std::sync::mpsc::channel::<()>();
            let helper = std::thread::spawn(move || {
                hand::share(wall);
                given.recv().unwrap();
                let _job = phase(Phase::PoolWork);
                hand::compute(70);
            });
            {
                let _wait = phase(Phase::PoolWait);
                give.send(()).unwrap();
                helper.join().unwrap();
            }
            frame();
            let taken = take();
            assert_eq!(taken.sum(), taken.ns);
            let (work, wait) = (Phase::PoolWork as usize, Phase::PoolWait as usize);
            assert_eq!(taken.wall[work], 0, "the helper's work is in the presenting thread's budget");
            assert_eq!(taken.wall[wait], 70 * MS, "the wait for a 70 ms job {} ms ({taken:?})", ms(taken.wall[wait]));
            assert_eq!(taken.ns, 70 * MS, "the helper's 70 ms were added to the frame: {} ms", ms(taken.ns));
            assert_eq!(taken.cpu[wait], 0, "the helper's processor time is the waiting thread's");
            // Other tests' helper threads add to the same figures: at least ours.
            let others = take_others();
            assert!(others.wall[work] >= 70 * MS, "{others:?}");
            assert!(others.cpu[work] >= 70 * MS * hand::RATE, "{others:?}");
            assert!(others.scopes >= 1);
        })
        .join()
        .unwrap();
    }

    /// The real clocks, which the other tests here replace with a hand clock: what a scope
    /// books lies between what the test reads itself just outside it and just inside it, on
    /// the wall clock and in the thread's cycles alike, whatever the machine's speed or load.
    /// A sleep inside a scope is the wait's wall time and not its processor time, computing is
    /// processor time, and the thread is charged no more cycles than the timestamp counter
    /// moved (the unit the cycles are turned into time by). Each bound holds by the order of
    /// the reads, not by a duration.
    #[test]
    fn the_real_clocks_bracket_what_a_scope_books() {
        std::thread::spawn(|| {
            // Cycles the computing is charged at least: milliseconds at any counter's rate.
            const WORK: u64 = 30_000_000;
            frame();
            // The counter read before the first cycles read and after the last.
            let from = clocks().2;
            let r0 = clocks();
            let refresh = phase(Phase::Refresh);
            let r1 = clocks();
            spin(WORK);
            let r2 = clocks();
            let wait = phase(Phase::CardWait);
            let r3 = clocks();
            std::thread::sleep(Duration::from_millis(30));
            let r4 = clocks();
            wait.end();
            let r5 = clocks();
            refresh.end();
            let r6 = clocks();
            frame();
            let taken = take();
            assert_eq!(taken.sum(), taken.ns);
            let (refresh, wait) = (Phase::Refresh as usize, Phase::CardWait as usize);
            // The wait began between r2 and r3 and ended between r4 and r5; the phase around
            // it began between r0 and r1, and ended between r5 and r6.
            for (what, clock, got) in [("wall", 0, taken.wall), ("cycles", 1, taken.cpu)] {
                let at = |r: (u64, u64, u64)| [r.0, r.1][clock];
                let span = |from, to| at(to) - at(from);
                let (inner, outer) = (span(r3, r4), span(r2, r5));
                assert!((inner..=outer).contains(&got[wait]), "the wait's {what} {} outside {inner}..={outer}", got[wait]);
                let (inner, outer) = (span(r1, r2), span(r0, r3) + span(r4, r6));
                assert!((inner..=outer).contains(&got[refresh]), "the phase's {what} {} outside {inner}..={outer}", got[refresh]);
            }
            assert!(taken.cpu[refresh] >= WORK, "computing was not processor time: {taken:?}");
            assert!(
                taken.cpu[wait] < taken.cpu[refresh],
                "a sleep was charged {} cycles, more than computing's {}",
                taken.cpu[wait],
                taken.cpu[refresh]
            );
            let charged = r6.1 - r0.1;
            assert!(charged <= r6.2 - from, "charged {charged} cycles while the counter moved {}", r6.2 - from);
            assert!(taken.cpu_ns(refresh).is_some(), "the counter's rate");
        })
        .join()
        .unwrap();
    }

    /// The reads the module's text counts, exactly, on a hand clock that counts them: a phase
    /// scope reads each clock twice (four reads), an entry scope the wall clock twice and the
    /// cycles never, an entry inside a scope of ours and a free lock read nothing, and a
    /// present reads each clock once.
    #[test]
    fn the_scopes_read_the_clocks_the_text_says() {
        std::thread::spawn(|| {
            static FREE: std::sync::Mutex<()> = std::sync::Mutex::new(());
            hand::set();
            let reads = |what: &dyn Fn()| {
                let (wall, cycles) = hand::reads();
                what();
                let after = hand::reads();
                (after.0 - wall, after.1 - cycles)
            };
            assert_eq!(reads(&|| assert_eq!(frame(), None)), (1, 1), "a present: (wall, cycles) reads");
            assert_eq!(reads(&|| phase(Phase::Sweep).end()), (2, 2), "a phase scope: (wall, cycles) reads");
            assert_eq!(reads(&|| entry(Phase::ListCalls).end()), (2, 0), "an entry scope: (wall, cycles) reads");
            let ours = phase(Phase::Unaccounted);
            assert_eq!(reads(&|| entry(Phase::ObjectCalls).end()), (0, 0), "an entry inside a scope of ours: (wall, cycles) reads");
            assert_eq!(reads(&|| drop(lock(&FREE, Phase::QueueLockWait))), (0, 0), "a free lock: (wall, cycles) reads");
            ours.end();
            frame();
            assert_eq!(take().scopes, 3, "the scopes: a phase, an entry and ours");
        })
        .join()
        .unwrap();
    }

    /// The cost of the two kinds of scope on the real clocks, printed for the report: a phase
    /// scope reads both clocks twice, an entry scope the wall clock twice (counted exactly by
    /// [`the_scopes_read_the_clocks_the_text_says`]). The bounds are on the thread's processor
    /// time, which other threads' load does not lengthen, turned into time at a rate the test
    /// can only overstate (the counter read outside the wall clock at both ends).
    #[test]
    fn a_scope_s_cost() {
        std::thread::spawn(|| {
            frame();
            const N: u32 = 200_000;
            let from = clocks().2;
            let r0 = clocks();
            for _ in 0..N {
                let _scope = phase(Phase::Sweep);
            }
            let r1 = clocks();
            for _ in 0..N {
                let _scope = entry(Phase::ListCalls);
            }
            let r2 = clocks();
            frame();
            let taken = take();
            assert_eq!(taken.sum(), taken.ns);
            assert_eq!(taken.scopes, 2 * u64::from(N));
            let rate = (r2.2 - from) as f64 / (r2.0 - r0.0) as f64;
            let n = f64::from(N);
            let (each_wall, light_wall) = ((r1.0 - r0.0) as f64 / n, (r2.0 - r1.0) as f64 / n);
            let (each, light) = ((r1.1 - r0.1) as f64 / n / rate, (r2.1 - r1.1) as f64 / n / rate);
            eprintln!(
                "budget: a phase scope costs {each_wall:.0} ns (two reads of each clock), an entry scope {light_wall:.0} ns (two reads of the wall clock); on the processor {each:.0} and {light:.0} ns"
            );
            assert!(each < 20_000.0 && light < each, "a phase scope costs {each:.0} ns of processor time, an entry scope {light:.0} ns");
        })
        .join()
        .unwrap();
    }
}

#[cfg(all(test, windows, feature = "diagnostics"))]
#[path = "budget_tests.rs"]
mod entry_tests;
