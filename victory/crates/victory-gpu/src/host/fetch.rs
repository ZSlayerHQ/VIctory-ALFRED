//! The copy-back's fetch on a copy queue of its own (`VICTORY_FETCH_QUEUE=1`; off by default,
//! live).
//!
//! Without it the relay's fetch of the pieces the card's compare flagged
//! (`Host::relay_card_back`) goes to the drawing queue and is signalled on a fence of the
//! relay's own: it runs only once the card has run every submission sent since the flights it
//! lands (the threading review's 3.3: a wait tens of times as long as the copying).
//! With it the fetch goes to a copy queue of ours, which waits on the drawing queue's fence
//! only for those flights, and so runs beside the later submissions. (The idea is the other
//! fork's `VICTORY_RELAY_COPY`; this code is ours.)
//!
//! **The ordering rule: the later writer is held.** A fetch must read what the drawing queue
//! had written when the flights it lands ended, and nothing later. Each plan made with the
//! switch on takes a ticket, a value of the copy queue's fence, and notes it on every copy whose
//! ranges it declared ([`Book`]). A later submission that writes such a copy (declares a range
//! of it, or sends title bytes up to it) puts a wait for that copy's last ticket on the drawing
//! queue ahead of everything it sends, unless the ticket is reached already. The landing signals
//! its ticket on the copy queue right behind the copies, before it merges. So no declared write
//! to a fetch's source runs while the fetch reads it, and the merge counts the sends of title
//! bytes up to the batch's last plan as on the card and the later ones as not.
//!
//! **Writers through the address directory are held too** (Sol's review, finding 3). A shader
//! that finds title memory by address reaches every copy the directory numbers when it runs,
//! including one numbered after its list was recorded (`crate::memory` writes the directory
//! and the views while lists are in the air), so what its list declared does not bound what it
//! writes. The hold is therefore made by what a submission can reach when it is sent: one with
//! a list that writes by address (`crate::memory::WRITES_BY_ADDRESS`) is held behind the last
//! ticket of every copy numbered then ([`numbered_copies`]). A copy numbered after that send
//! could still be reached by it: its numbering waits for a copy-queue read of it under way, and
//! a landing whose ticket is not older than the numbering goes to the drawing queue
//! ([`numbered`], [`why::LATE`]). With `VICTORY_BVH_QUARANTINE=1` the builder stores through
//! views it does not declare at all: then no plan takes a ticket ([`Refused`]). Every plan that
//! does not qualify is landed as before, on the drawing queue, and counted ([`FALLBACK`]).
//!
//! **A drawing-queue fetch never waits behind a held writer.** Before one is sent, every ticket
//! given so far is signalled ([`Shared::release_all`]; nothing is being read on the copy queue
//! then), and the plans those tickets were for land on the drawing queue too: their writers may
//! have run.
//!
//! **Failures** (finding 1): every copy-queue signal goes through [`Shared::signal`] (in order,
//! never lower than the last). A fetch not sent stops the copy queue for the rest of the run:
//! every ticket is released, which is safe because nothing of ours is in flight on it, and the
//! landing goes to the drawing queue. A signal that fails after the copies were sent, or a wait
//! past [`BOUND`] on a live device, fails the run: nothing tells when that copy ends, so its
//! writers stay held and no landing is published.
//!
//! **The waits it adds, and what ends each** (none without the switch: no queue, fence or wait
//! is made, and the drawing queue gets exactly what it got before):
//!
//! 1. The drawing queue's wait for a ticket ([`Book::hold`]): ended by the copy queue's signal
//!    of that ticket or a higher one (its landing, with or without pieces; a release before a
//!    drawing-queue fetch; a stop); by a removed device (every fence reads `u64::MAX`); else the
//!    relay's bounded wait fails the run.
//! 2. The copy queue's wait for the drawing queue's fence ([`Relay::fetch`]): for a value the
//!    relay has seen reached before it sends, so it ends at once; by a removed device.
//! 3. The relay's wait for the copy queue's fence: by the signal behind the copies; by a removed
//!    device (the run fails as before); else at [`BOUND`] the run fails.
//! 4. A numbering in the address directory waits for a copy-queue read of that copy under way
//!    ([`numbered`]): ended when the relay's wait 3 ends.
//!
//! **Resource states** (Microsoft's "Using resource barriers" and "Multi-engine
//! synchronization" pages): a buffer decays to COMMON when the `ExecuteCommandLists` that used
//! it completes, whatever barriers it had; COMMON is promoted implicitly on a copy queue to
//! COPY_SOURCE (the card copy) and COPY_DEST (the PC-memory buffer the pieces come down to); and
//! what a copy queue used decays to COMMON when its work completes. A fence wait orders a
//! queue's write before another queue's read; only one queue writes a resource at a time.

#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::Interface;

use super::{CardPlan, Declared, Host, fail};
use crate::shadow::Shadow;

/// Whether a plan made now (with nothing out) fetches on the copy queue (live; starts from
/// `VICTORY_FETCH_QUEUE`, off).
pub(crate) fn on() -> bool {
    crate::live::fetch_queue()
}

/// Whether anything of the copy queue can matter to a submission now: the switch is on, or a
/// ticket given is not reached yet (a writer of its copies must still be held). With the switch
/// off and nothing out this is two loads, and nothing else of this module runs for a
/// submission: no lock, no scan of its lists.
pub(super) fn engaged() -> bool {
    on() || SHARED.get().and_then(Option::as_ref).is_some_and(|s| s.reached() < s.issued.load(Ordering::Acquire))
}

/// What a submission's lists may reach without declaring it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Reach {
    /// Only what they declare.
    Declared,
    /// Every copy numbered in the address directory (`crate::memory::WRITES_BY_ADDRESS`).
    Directory,
    /// Any copy: a store through a view picked from the heap by a computed index
    /// (`crate::memory::WRITES_BY_HEAP`).
    Any,
}

/// What `runs` may reach ([`Reach`]); `Declared` without a look at them unless [`engaged`].
pub(super) fn reach(runs: &[&super::HostList]) -> Reach {
    if !engaged() {
        return Reach::Declared;
    }
    let mut most = Reach::Declared;
    for list in runs {
        let marks = &list.deferred().textures;
        if marks.contains(&crate::memory::WRITES_BY_HEAP) {
            return Reach::Any;
        }
        if marks.contains(&crate::memory::WRITES_BY_ADDRESS) {
            most = Reach::Directory;
        }
    }
    most
}

/// The longest the relay waits for a fetch on the copy queue: 10 s. The drawing queue's own
/// waits have no bound of their own; they end because the system resets a card whose work runs
/// past its GPU timeout (2 s by default) and every fence then reads `u64::MAX`. A fetch copies
/// tens of megabytes (milliseconds), so a live device whose copy queue has not reached its
/// value five timeouts later lost the signal: waiting longer would only hang.
pub(super) const BOUND: Duration = Duration::from_secs(10);

/// Why a plan, or a landing, stays on the drawing queue ([`FALLBACK`]).
pub(super) mod why {
    /// Not counted since `job/fetch-queue-2`: a copy in the address directory qualifies (its
    /// writers by address are held). Kept so the stats keep their names.
    #[allow(dead_code)]
    pub(crate) const NUMBERED: usize = 0;
    /// `VICTORY_BVH_QUARANTINE=1`: the builder's stores run undeclared.
    pub(crate) const QUARANTINE: usize = 1;
    /// The copy queue was stopped by a failure.
    pub(crate) const STOPPED: usize = 2;
    /// Its ticket was released for a drawing-queue fetch: its writers may have run.
    pub(crate) const RELEASED: usize = 3;
    /// A copy of it was numbered in the directory while its ticket was out.
    pub(crate) const LATE: usize = 4;
    /// Landed with a plan that stays on the drawing queue.
    pub(crate) const SHARED: usize = 5;
    pub(crate) const COUNT: usize = 6;
    /// The names the line and the stats file give them.
    pub(crate) const NAMES: [&str; COUNT] = ["numbered", "quarantine", "stopped", "released", "late", "shared"];
}

/// Since the last frame-rate line: plans landed on the drawing queue with the switch on, by
/// [`why`].
pub(super) static FALLBACK: [AtomicU64; why::COUNT] = [const { AtomicU64::new(0) }; why::COUNT];

/// A piece to fetch: its copy, its offsets from and to, and where it comes down.
pub(super) type Piece = (Arc<Shadow>, usize, usize, u64);

/// What a plan made with the switch on carries to its landing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Ticket {
    /// The copy queue's fence value its landing signals; 0 if it declared no range (nothing of
    /// it is fetched and nothing waits for it).
    pub(super) value: u64,
    /// `QUEUE_ORDER`'s count of sends of title bytes when the plan was made.
    pub(super) order: u64,
}

/// The ticketed plans of one landing: the highest ticket (each lower one is either among them
/// or was signalled before), the lowest one that is not 0 (`u64::MAX` if none), the highest
/// order, how many.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Batch {
    value: u64,
    first: u64,
    order: u64,
    plans: u64,
}

/// Whether a batch's tickets were released for a drawing-queue fetch: by its lowest one, not its
/// highest (Sol's second review, 4: in a batch of an old ticket, released, and a newer one, the
/// old one's writers may be running).
pub(super) fn released(batch: &Batch, released: u64) -> bool {
    batch.first != u64::MAX && batch.first <= released
}

/// The landing's ticketed plans as one [`Batch`], or `None` if it has none.
pub(super) fn batch(plans: &[&CardPlan]) -> Option<Batch> {
    let mut batch: Option<Batch> = None;
    for ticket in plans.iter().filter_map(|plan| plan.fetch) {
        let b = batch.get_or_insert(Batch { value: 0, first: u64::MAX, order: 0, plans: 0 });
        b.value = b.value.max(ticket.value);
        if ticket.value > 0 {
            b.first = b.first.min(ticket.value);
        }
        b.order = b.order.max(ticket.order);
        b.plans += 1;
    }
    batch
}

/// The orders of the ticketed plans not landed yet, in the line's order. Pushed under
/// `QUEUE_ORDER`.
static ORDERS: Mutex<VecDeque<u64>> = Mutex::new(VecDeque::new());

/// How far the notes of sends of title bytes may go, after a fetch that read up to `after`:
/// not past the order of a ticketed plan still to land, whose fetch reads only up to it.
/// `after` itself when none is out (always, with the switch off).
pub(super) fn prune_bound(after: u64) -> u64 {
    // No copy queue was ever made: no ticket, and no lock taken.
    if SHARED.get().is_none() {
        return after;
    }
    ORDERS.lock().unwrap_or_else(|e| e.into_inner()).front().map_or(after, |&o| o.min(after))
}

/// `plans` ticketed plans are landing: they leave [`ORDERS`].
fn landing(plans: u64) {
    let mut orders = ORDERS.lock().unwrap_or_else(|e| e.into_inner());
    for _ in 0..plans {
        orders.pop_front();
    }
}

/// Since the last frame-rate line: fetches sent to the copy queue, their bytes, microseconds
/// from their send to their end, submissions held behind a fetch, microseconds from a hold
/// being put on the drawing queue to the relay seeing its ticket reached (at most that long on
/// the card), tickets given, stops of the copy queue, plans that qualified (a ticket for
/// declared ranges), and holds made only because the submission writes by address.
pub(super) static STATS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

/// The copy queue and its fence, made the first time a plan is made with the switch on; what
/// was given and signalled on the fence.
pub(super) struct Shared {
    pub(super) queue: ID3D12CommandQueue,
    pub(super) fence: ID3D12Fence,
    /// The highest ticket given.
    issued: AtomicU64,
    /// The highest value signalled on the queue (or, at a stop, from the CPU).
    signalled: AtomicU64,
    /// The highest ticket released for a drawing-queue fetch: plans up to it land there.
    released: AtomicU64,
    stopped: AtomicBool,
}

// SAFETY: D3D12 queues and fences are free-threaded.
unsafe impl Send for Shared {}
// SAFETY: as above.
unsafe impl Sync for Shared {}

impl Shared {
    pub(super) fn new(device: &ID3D12Device) -> windows::core::Result<Self> {
        // SAFETY: plain object creation.
        unsafe {
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COPY,
                ..Default::default()
            })?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            Ok(Self {
                queue,
                fence,
                issued: AtomicU64::new(0),
                signalled: AtomicU64::new(0),
                released: AtomicU64::new(0),
                stopped: AtomicBool::new(false),
            })
        }
    }

    fn reached(&self) -> u64 {
        // SAFETY: a plain query.
        unsafe { self.fence.GetCompletedValue() }
    }

    pub(super) fn stopped(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
    }

    /// Signals `value` on the copy queue, behind what it has, unless as much was signalled
    /// already (the fence never goes back). The relay's thread only.
    pub(super) fn signal(&self, value: u64) -> windows::core::Result<()> {
        if value <= self.signalled.load(Ordering::Acquire) {
            return Ok(());
        }
        #[cfg(test)]
        if FAULT.get() == Fault::Release {
            return Err(windows::Win32::Foundation::E_FAIL.into());
        }
        // SAFETY: our queue and fence.
        unsafe { self.queue.Signal(&self.fence, value)? };
        self.signalled.store(value, Ordering::Release);
        Ok(())
    }

    /// Before a fetch on the drawing queue: every ticket given so far is signalled, so that the
    /// fetch is not behind a writer held for one; their plans land on the drawing queue. Only
    /// when nothing is being read on the copy queue (the relay waited for its last fetch).
    pub(super) fn release_all(&self) -> windows::core::Result<()> {
        let issued = self.issued.load(Ordering::Acquire);
        if issued <= self.signalled.load(Ordering::Acquire) {
            return Ok(());
        }
        self.released.fetch_max(issued, Ordering::AcqRel);
        self.signal(issued)
    }

    /// The copy queue is not used again: every hold ends now (nothing of ours is in flight on
    /// it), first on the queue, else from the CPU. False if neither could signal.
    pub(super) fn stop(&self) -> bool {
        self.stopped.store(true, Ordering::Release);
        self.released.store(u64::MAX, Ordering::Release);
        if self.signal(u64::MAX).is_ok() {
            return true;
        }
        // SAFETY: our fence, from the CPU.
        let done = unsafe { self.fence.Signal(u64::MAX) }.is_ok();
        if done {
            self.signalled.store(u64::MAX, Ordering::Release);
        }
        done
    }
}

static SHARED: OnceLock<Option<Shared>> = OnceLock::new();

/// The copies numbered in the address directory, each with the highest ticket given when it
/// was numbered; and those the copy queue is reading now.
#[derive(Default)]
struct Guard {
    numbered: HashMap<usize, u64>,
    reading: Vec<usize>,
}

static GUARD: Mutex<Option<Guard>> = Mutex::new(None);
static READ_DONE: Condvar = Condvar::new();

/// `crate::memory`: `shadow` is numbered in the address directory, before its entries are
/// written. A submission sent before this may reach it without being held for it, so a landing
/// whose plans' tickets are not newer than this numbering does not read it on the copy queue
/// ([`start_reading`]); a read of it under way is waited for first (it ends within [`BOUND`]).
/// Submissions sent from here on that write by address are held for it ([`numbered_copies`]).
pub(crate) fn numbered(shadow: &Arc<Shadow>) {
    // No copy queue made: no ticket, read or hold to order against. The copies numbered so
    // far are taken from the directory when one is made ([`seed`]).
    let Some(shared) = SHARED.get().and_then(Option::as_ref) else {
        return;
    };
    let copy = Arc::as_ptr(shadow) as usize;
    let mut guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    // Read under the guard, which a submission's view of the numbered copies takes too.
    let issued = shared.issued.load(Ordering::Acquire);
    guard.get_or_insert_with(Guard::default).numbered.entry(copy).or_insert(issued);
    while guard.as_ref().is_some_and(|g| g.reading.contains(&copy)) {
        guard = READ_DONE.wait(guard).unwrap_or_else(|e| e.into_inner());
    }
}

/// Once the copy queue is made: the copies the directory numbered before, as numbered before
/// any ticket. A numbering from here on notes itself ([`numbered`]); one between the queue's
/// making and this is in the directory already, and `or_insert` keeps its own.
fn seed() {
    static SEEDED: std::sync::Once = std::sync::Once::new();
    SEEDED.call_once(|| {
        let copies = crate::memory::numbered_copies();
        let mut guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let g = guard.get_or_insert_with(Guard::default);
        for copy in &copies {
            g.numbered.entry(Arc::as_ptr(copy) as usize).or_insert(0);
        }
    });
}

/// The copies numbered in the address directory now: what a list that writes by address can
/// reach when it runs, if numbered before its submission is sent.
fn numbered_copies() -> Vec<usize> {
    GUARD
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or_else(Vec::new, |g| g.numbered.keys().copied().collect())
}

/// `copies` are to be read on the copy queue for plans whose lowest ticket is `first`: false
/// if one was numbered after that ticket was given (a writer by address sent before the
/// numbering is not held for it; the landing goes to the drawing queue), else they are marked
/// as being read until [`read_done`].
fn start_reading(copies: &[usize], first: u64) -> bool {
    let mut guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let g = guard.get_or_insert_with(Guard::default);
    if copies.iter().any(|c| g.numbered.get(c).is_some_and(|&at| at >= first)) {
        return false;
    }
    g.reading.extend_from_slice(copies);
    true
}

fn read_done() {
    if let Some(g) = GUARD.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        g.reading.clear();
    }
    READ_DONE.notify_all();
}

/// The tickets: which copy's last plan holds which, for the drawing queue's holds. Under the
/// submission lock and `QUEUE_ORDER` (`Host::card_before`); the relay takes it alone.
pub(super) struct Book {
    mode: bool,
    next: u64,
    /// Each copy (its address) and the last ticket of a plan that declared a range of it.
    last: Vec<(usize, u64)>,
    /// Holds put on the drawing queue and not yet seen reached: ticket, when.
    held: Vec<(u64, Instant)>,
}

impl Book {
    pub(super) const fn new() -> Self {
        Self {
            mode: false,
            next: 0,
            last: Vec::new(),
            held: Vec::new(),
        }
    }

    /// The mode of a plan made now. With `none_out` (no plan out) it follows `on`, and the
    /// tickets of plans landed are forgotten.
    pub(super) fn mode(&mut self, on: bool, none_out: bool) -> bool {
        if none_out {
            self.mode = on;
            self.last.clear();
        }
        self.mode
    }

    /// A submission writes the copies `written` (their addresses), and may reach `by_address`
    /// (the numbered copies, if it writes by address), or `any` copy (it stores through views
    /// picked by a computed heap index): if the last ticket of any of them is not
    /// reached, the drawing queue waits for it before what the submission sends. The ticket
    /// waited for, if any.
    pub(super) fn hold(
        &mut self,
        shared: &Shared,
        drawing: &ID3D12CommandQueue,
        written: impl Iterator<Item = usize>,
        by_address: &[usize],
        any: bool,
    ) -> windows::core::Result<Option<u64>> {
        if self.last.is_empty() {
            return Ok(None);
        }
        let reached = shared.reached();
        self.last.retain(|&(_, ticket)| ticket > reached);
        let last = |copy: usize| self.last.iter().find(|(c, _)| *c == copy).map_or(0, |&(_, ticket)| ticket);
        let declared = written.map(last).max().unwrap_or(0);
        let reachable = if any {
            self.last.iter().map(|&(_, ticket)| ticket).max().unwrap_or(0)
        } else {
            by_address.iter().map(|&c| last(c)).max().unwrap_or(0)
        };
        let most = reachable.max(declared);
        if most == 0 {
            return Ok(None);
        }
        // SAFETY: our queue and the copy queue's fence.
        unsafe { drawing.Wait(&shared.fence, most)? };
        self.held.push((most, Instant::now()));
        STATS[3].fetch_add(1, Ordering::Relaxed);
        if most > declared {
            STATS[8].fetch_add(1, Ordering::Relaxed);
        }
        Ok(Some(most))
    }

    /// A plan declared ranges of `declared` (their addresses): its ticket, noted on each.
    pub(super) fn issue(&mut self, shared: &Shared, declared: impl Iterator<Item = usize>) -> u64 {
        self.next += 1;
        let ticket = self.next;
        for copy in declared {
            match self.last.iter_mut().find(|(c, _)| *c == copy) {
                Some(entry) => entry.1 = ticket,
                None => self.last.push((copy, ticket)),
            }
        }
        shared.issued.fetch_max(ticket, Ordering::AcqRel);
        ticket
    }

    /// The relay saw `ticket` reached (or gave it up): the holds it ended are timed.
    fn reached(&mut self, ticket: u64) {
        let mut us = 0;
        self.held.retain(|(t, at)| {
            let ended = *t <= ticket;
            if ended {
                us += at.elapsed().as_micros() as u64;
            }
            !ended
        });
        STATS[4].fetch_add(us, Ordering::Relaxed);
    }
}

static BOOK: Mutex<Book> = Mutex::new(Book::new());

/// Why a plan made with the switch on does not take a ticket ([`why`]), if it does not.
pub(super) type Refused = Option<usize>;

/// [`Host::fetch_ticket`]'s rule, apart from the host: with `relay` (the relay lands flights),
/// the plan's mode; the hold for the copies it writes (and `by_address`, the copies it may
/// reach by address); its ticket if it is `flown`, made with the switch on, and not `refused`.
/// `shared` gives the copy queue, made if asked with `true`; it is never asked with the switch
/// off and nothing out.
#[allow(clippy::too_many_arguments)]
pub(super) fn ticket_for<'a>(
    book: &mut Book,
    on: bool,
    relay: bool,
    flown: bool,
    none_out: bool,
    order: u64,
    declared: &[usize],
    written: &[usize],
    by_address: &[usize],
    any: bool,
    refused: impl FnOnce(&Shared) -> Refused,
    shared: impl FnOnce(bool) -> Option<&'a Shared>,
    drawing: &ID3D12CommandQueue,
) -> windows::core::Result<Option<Ticket>> {
    if !relay {
        return Ok(None);
    }
    let mode = book.mode(on, none_out);
    if !mode && book.last.is_empty() {
        return Ok(None);
    }
    let Some(shared) = shared(mode) else {
        book.mode = false;
        return Ok(None);
    };
    book.hold(shared, drawing, written.iter().copied(), by_address, any)?;
    if !(mode && flown) {
        return Ok(None);
    }
    let value = if declared.is_empty() {
        0
    } else {
        if let Some(why) = refused(shared) {
            FALLBACK[why].fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        STATS[7].fetch_add(1, Ordering::Relaxed);
        book.issue(shared, declared.iter().copied())
    };
    ORDERS.lock().unwrap_or_else(|e| e.into_inner()).push_back(order);
    STATS[5].fetch_add(1, Ordering::Relaxed);
    Ok(Some(Ticket { value, order }))
}

/// A piece's copy as the copy queue takes it: from `source`, `len` bytes at `lo`, to `at`.
pub(super) struct Part<'a> {
    pub(super) source: &'a ID3D12Resource,
    pub(super) lo: u64,
    pub(super) len: u64,
    pub(super) at: u64,
}

/// Where a fetch stands in the queues' order: the drawing queue's fence and the value that
/// says the flights landed have run, and the ticket the fetch signals.
pub(super) struct Order<'a> {
    pub(super) fence: &'a ID3D12Fence,
    pub(super) ran: u64,
    pub(super) ticket: u64,
}

/// A fetch that ran: where its bytes are, and when it was recorded, sent and done.
pub(super) struct Fetched {
    pub(super) data: usize,
    pub(super) recorded: Instant,
    pub(super) sent: Instant,
    pub(super) done: Instant,
    /// With `VICTORY_GPU_STAMPS=1`, where the adapter has timestamps on copy queues: the
    /// nanoseconds between the list's first and last copy on the copy queue's clock
    /// (`super::queue_stamps`, which counts it; kept here for the fixtures).
    #[cfg(all(test, feature = "diagnostics"))]
    pub(super) on_queue: Option<u64>,
}

#[derive(Debug)]
pub(super) enum Failed {
    /// The device was removed while the fetch ran.
    Removed,
    /// Not recorded or not sent: nothing of it is on the copy queue, and its ticket was not
    /// signalled.
    NotSent(windows::core::Error),
    /// Sent, but its signal failed twice: nothing tells when its copies end.
    Unsignalled(windows::core::Error),
    /// Sent and signalled, and not reached within the bound on a live device.
    Timeout(Duration),
}

/// Tests: a fault made on this thread's next fetch or signal.
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Fault {
    None,
    /// The copy queue's wait for the drawing fence fails (before anything is sent).
    BeforeSend,
    /// The signal after the copies fails, and its retry.
    Signal,
    /// [`Shared::signal`] fails.
    Release,
}

#[cfg(test)]
thread_local! {
    pub(super) static FAULT: Cell<Fault> = const { Cell::new(Fault::None) };
    /// Tests: the relay's bound on this thread, if not [`BOUND`].
    pub(super) static TEST_BOUND: Cell<Option<Duration>> = const { Cell::new(None) };
    /// Tests: run on this thread between a fetch's send and its wait.
    pub(super) static BEFORE_WAIT: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
}

fn bound() -> Duration {
    #[cfg(test)]
    if let Some(bound) = TEST_BOUND.get() {
        return bound;
    }
    BOUND
}

/// Until `fence` reaches `value` on a live device, for at most `bound`: by `event` where it
/// can be set on the fence, then by asking the fence every millisecond.
fn bounded_wait(device: &ID3D12Device, fence: &ID3D12Fence, event: HANDLE, value: u64, bound: Duration) -> Result<(), Failed> {
    let _p = crate::budget::phase(crate::budget::Phase::CardWait);
    let deadline = Instant::now() + bound;
    // SAFETY: a live fence and an event of ours.
    unsafe {
        if fence.SetEventOnCompletion(value, event).is_ok() {
            let ms = u32::try_from(bound.as_millis()).unwrap_or(u32::MAX);
            WaitForSingleObject(event, ms);
        }
    }
    loop {
        // SAFETY: a plain query.
        let done = unsafe { fence.GetCompletedValue() };
        if done == u64::MAX && super::device_removed(device) {
            return Err(Failed::Removed);
        }
        if done >= value {
            return Ok(());
        }
        if super::device_removed(device) {
            return Err(Failed::Removed);
        }
        if Instant::now() >= deadline {
            return Err(Failed::Timeout(bound));
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The relay's own list for the copy queue, its event, and the memory the pieces come down to.
pub(super) struct Relay {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    event: HANDLE,
    buffer: Option<(ID3D12Resource, usize, u64)>,
    /// The list's pair of timestamps (`super::queue_stamps`).
    #[cfg(feature = "diagnostics")]
    pair: Option<super::queue_stamps::Pair>,
}

impl Relay {
    pub(super) fn new(device: &ID3D12Device) -> windows::core::Result<Self> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
            let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
            list.Close()?;
            let event = CreateEventW(None, false, false, None)?;
            Ok(Self {
                allocator,
                list,
                event,
                buffer: None,
                #[cfg(feature = "diagnostics")]
                pair: super::queue_stamps::Pair::new(device),
            })
        }
    }

    /// Memory for `total` bytes of pieces, from `make` (PC memory the CPU reads, created in the
    /// common state): whether there is.
    pub(super) fn room(&mut self, total: u64, make: impl FnOnce(u64) -> Option<(ID3D12Resource, *mut u8)>) -> bool {
        if self.buffer.as_ref().is_none_or(|b| b.2 < total) {
            let size = total.next_multiple_of(16 << 20);
            self.buffer = make(size).map(|(buffer, data)| {
                super::name(&buffer, || "pieces fetched on the copy queue".into());
                (buffer, data as usize, size)
            });
        }
        self.buffer.is_some()
    }

    /// Fetches `parts` into the memory [`Relay::room`] gave: the copy queue waits for the
    /// drawing queue's fence at `order.ran`, copies, and signals `order.ticket`; `beside` runs
    /// while the copies are in flight; then this thread waits for the ticket, at most
    /// [`BOUND`]. Nothing here signals the ticket other than behind the copies.
    pub(super) fn fetch(
        &mut self,
        device: &ID3D12Device,
        shared: &Shared,
        order: Order,
        parts: &[Part],
        beside: impl FnOnce(),
    ) -> Result<Fetched, Failed> {
        let Some((buffer, data, _)) = self.buffer.clone() else {
            return Err(Failed::NotSent(windows::core::Error::empty()));
        };
        let mut recorded = Instant::now();
        // SAFETY: the list's last run was waited for by this thread; live buffers; the fences
        // are ours.
        let sent = unsafe {
            (|| -> windows::core::Result<[Option<ID3D12CommandList>; 1]> {
                self.allocator.Reset()?;
                self.list.Reset(&self.allocator, None)?;
                crate::events::begin(&self.list, crate::events::COPY_QUEUE_FETCH);
                #[cfg(feature = "diagnostics")]
                if let Some(pair) = &self.pair {
                    pair.begin(&self.list);
                }
                for part in parts {
                    self.list.CopyBufferRegion(&buffer, part.at, part.source, part.lo, part.len);
                }
                #[cfg(feature = "diagnostics")]
                if let Some(pair) = &self.pair {
                    pair.end(&self.list);
                }
                crate::events::end(&self.list);
                self.list.Close()?;
                let lists = [Some(self.list.cast::<ID3D12CommandList>()?)];
                recorded = Instant::now();
                #[cfg(test)]
                if FAULT.get() == Fault::BeforeSend {
                    return Err(windows::Win32::Foundation::E_FAIL.into());
                }
                // What the flights wrote on the drawing queue, before the copies read it.
                shared.queue.Wait(order.fence, order.ran)?;
                Ok(lists)
            })()
        };
        let lists = sent.map_err(Failed::NotSent)?;
        // SAFETY: a closed list of ours; our queue and fence.
        let signalled = unsafe {
            shared.queue.ExecuteCommandLists(&lists);
            let once = || -> windows::core::Result<()> {
                #[cfg(test)]
                if FAULT.get() == Fault::Signal {
                    return Err(windows::Win32::Foundation::E_FAIL.into());
                }
                shared.queue.Signal(&shared.fence, order.ticket)
            };
            once().or_else(|_| once())
        };
        signalled.map_err(Failed::Unsignalled)?;
        shared.signalled.fetch_max(order.ticket, Ordering::AcqRel);
        let sent = Instant::now();
        beside();
        #[cfg(test)]
        if let Some(hook) = BEFORE_WAIT.with(|hook| hook.borrow_mut().take()) {
            hook();
        }
        bounded_wait(device, &shared.fence, self.event, order.ticket, bound())?;
        let done = Instant::now();
        // The ticket is reached on a live device: the list's two timestamps are there.
        #[cfg(feature = "diagnostics")]
        let on_queue = self.pair.as_ref().and_then(|pair| pair.read(&shared.queue));
        #[cfg(feature = "diagnostics")]
        super::queue_stamps::copy_fetched(self.pair.is_some(), on_queue);
        Ok(Fetched {
            data,
            recorded,
            sent,
            done,
            #[cfg(all(test, feature = "diagnostics"))]
            on_queue,
        })
    }
}

impl Drop for Relay {
    fn drop(&mut self) {
        // SAFETY: our event.
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(self.event);
        }
    }
}

thread_local! {
    /// The relay thread's [`Relay`], made at its first fetch with the switch on.
    static RELAY: RefCell<Option<Relay>> = const { RefCell::new(None) };
}

/// The pieces the compares of `plans` flagged (all of a range not compared), each once: as
/// `Host::relay_card_back` gathers them in either mode.
pub(super) fn pieces(plans: &[&CardPlan]) -> (Vec<Piece>, u64) {
    let differs = {
        let work = super::CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        work.as_ref().map(|work| (work.differs.1, (work.differs.2 / 4) as usize))
    };
    let mut pieces: Vec<Piece> = Vec::new();
    let mut total = 0u64;
    if let Some((differs, dwords)) = differs {
        // SAFETY: the mapping of the results. The compares of these
        // plans have run, and their dwords are given to no other plan
        // before these are done with (`card_done`).
        let differs = unsafe { std::slice::from_raw_parts(differs as *const u32, dwords) };
        let mut seen = std::collections::HashSet::new();
        for plan in plans {
            let flags: Vec<(usize, u32, bool)> = plan
                .ranges
                .iter()
                .flat_map(|r| {
                    (0..r.span.pieces).map(move |p| {
                        let flagged = !r.compared || differs.get((r.at + p) as usize).is_none_or(|&d| d != 0);
                        (Arc::as_ptr(&r.shadow) as usize, r.span.first + p, flagged)
                    })
                })
                .collect();
            let fetched = super::flagged_pieces(&flags);
            for r in &plan.ranges {
                let copy = Arc::as_ptr(&r.shadow) as usize;
                for p in 0..r.span.pieces {
                    if !fetched.contains(&(copy, r.span.first + p)) {
                        continue;
                    }
                    let (lo, hi) = r.shadow.card_piece(&r.span, p);
                    if lo < hi && seen.insert((copy, lo, hi)) {
                        pieces.push((r.shadow.clone(), lo, hi, total));
                        total += (hi - lo) as u64;
                    }
                }
                crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
            }
        }
    }
    (pieces, total)
}

/// The fetched pieces, `total` bytes at `data`, go into title memory, in either mode: counted,
/// merged (`Shadow::card_merge_relay`, with `after`, the last send of title bytes the card had
/// when they were read), and the notes of sends let go up to [`prune_bound`]. `us`: the
/// fetch's time.
pub(super) fn land_pieces(pieces: &[Piece], data: usize, total: u64, us: u64, after: u64, exact: bool) {
    {
        use crate::shadow::CARD_STATS;
        CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
        CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
        CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
        CARD_STATS[6].fetch_add(us, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        for (shadow, lo, hi, _) in pieces {
            shadow.census(crate::memclass::Count::Fetched, *lo, *hi);
        }
    }
    // The merge alone is timed as merging (Sol's review of the frame
    // timers: the statistics and the stamps' reading were in it).
    let merging = Instant::now();
    let mut title = Vec::new();
    let mut written = 0;
    for (shadow, lo, hi, at) in pieces {
        // SAFETY: the mapping of the fetched bytes, `total` of them;
        // the copy into it has run (its fence was seen to reach it).
        let gpu = unsafe { std::slice::from_raw_parts((data + *at as usize) as *const u8, hi - lo) };
        let merged = shadow.card_merge_relay(*lo, gpu, after, &mut title, exact);
        shadow.walls_merged(gpu.len(), merged);
        written += merged;
    }
    super::RELAY_STATS[7].fetch_add(merging.elapsed().as_micros() as u64, Ordering::Relaxed);
    super::COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
    // Sends up to `after` are of no use to any later merge (with tickets out: up to the
    // oldest one's order).
    crate::shadow::relay_fetched(prune_bound(after));
    super::RELAY_STATS[9].fetch_add(written as u64, Ordering::Relaxed);
}

/// Says a failure of the copy queue once, and counts it.
fn said(what: &str) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    STATS[6].fetch_add(1, Ordering::Relaxed);
    ONCE.call_once(|| tracing::warn!("{what}: the copy-back's fetch goes to the drawing queue for the rest of the run (VICTORY_FETCH_QUEUE)"));
}

#[cfg(test)]
thread_local! {
    /// Tests: run between the release before a drawing-queue fetch and the fetch's send.
    pub(super) static AFTER_RELEASE: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
}

/// The relay's wait for its fetch on the drawing queue (`Host::relay_card_back`): as before
/// (`fetch_waited`) unless [`engaged`]; then at most [`BOUND`], which fails the run: a writer
/// held for a ticket would otherwise wait for ever behind it. False if the device was removed.
pub(super) fn drawing_waited(device: &ID3D12Device, fence: &ID3D12Fence, event: HANDLE, value: u64) -> windows::core::Result<bool> {
    if !engaged() {
        return super::fetch_waited(device, fence, event, value);
    }
    match bounded_wait(device, fence, event, value, bound()) {
        Ok(()) => Ok(true),
        Err(Failed::Removed) => Ok(false),
        Err(failed) => {
            said("a fetch on the drawing queue did not end");
            fail(format_args!(
                "the relay's fetch on the drawing queue did not end ({failed:?}) with the copy queue's tickets out (VICTORY_FETCH_QUEUE=1)"
            ))
        }
    }
}

/// Before a fetch on the drawing queue, under `QUEUE_ORDER` and right before its send: no
/// writer is held for a ticket then ([`Shared::release_all`]), and none can be held for a newer
/// one until the fetch and its signal are on the queue.
pub(super) fn before_drawing_fetch() {
    let Some(shared) = SHARED.get().and_then(Option::as_ref) else {
        return;
    };
    if let Err(e) = shared.release_all() {
        said(&format!("the copy queue did not take a release ({e})"));
        if !shared.stop() {
            fail(format_args!("the copy queue's holds could not be ended ({e}, VICTORY_FETCH_QUEUE=1)"));
        }
        BOOK.lock().unwrap_or_else(|e| e.into_inner()).last.clear();
    }
    #[cfg(test)]
    if let Some(hook) = AFTER_RELEASE.with(|hook| hook.borrow_mut().take()) {
        hook();
    }
}

impl Host {
    /// `Host::card_before` for a submission without a plan (it declares no copy's range and
    /// sends nothing up), under `QUEUE_ORDER`: one that may still reach copies by address or by
    /// a heap index is held behind their tickets (Sol's second review, 2 and 3).
    pub(super) fn fetch_hold_only(&self, reach: Reach) {
        if reach == Reach::Declared {
            return;
        }
        let Some(shared) = SHARED.get().and_then(Option::as_ref) else {
            return;
        };
        let by_address = if reach == Reach::Directory { numbered_copies() } else { Vec::new() };
        let mut book = BOOK.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(e) = book.hold(shared, &self.queue, std::iter::empty(), &by_address, reach == Reach::Any) {
            fail(format_args!("a submission could not be held behind the copy queue's fetch ({e}, VICTORY_FETCH_QUEUE=1)"));
        }
    }

    /// `Host::card_before`'s part, under `QUEUE_ORDER` and before anything of the submission
    /// goes to the queue ([`ticket_for`]): `declared` are the copies whose ranges its shaders
    /// may write, `dirty` those its title bytes go up to; `by_address`: a list of it writes by
    /// address (it may reach every numbered copy); `none_out`: no plan is out.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fetch_ticket(
        &self,
        flown: bool,
        none_out: bool,
        order: u64,
        declared: &[Declared],
        dirty: &[Declared],
        reach: Reach,
    ) -> Option<Ticket> {
        // Off, with nothing out: nothing here runs, as before the copy queue was there.
        if !engaged() {
            return None;
        }
        let id = |(shadow, _): &Declared| Arc::as_ptr(shadow) as usize;
        let declared: Vec<usize> = declared.iter().map(id).collect();
        let written: Vec<usize> = declared.iter().copied().chain(dirty.iter().map(id)).collect();
        let mut book = BOOK.lock().unwrap_or_else(|e| e.into_inner());
        // What it can reach by address when it runs: every copy numbered by now (one numbered
        // later is kept off the copy queue by `numbered`).
        let by_address = if reach == Reach::Directory && SHARED.get().is_some() { numbered_copies() } else { Vec::new() };
        let shared = |make: bool| {
            if make {
                SHARED
                    .get_or_init(|| match Shared::new(&self.device) {
                        Ok(shared) => Some(shared),
                        Err(e) => {
                            tracing::warn!(error = %e, "no copy queue for the copy-back's fetch: it stays on the drawing queue (VICTORY_FETCH_QUEUE)");
                            None
                        }
                    })
                    .as_ref()
            } else {
                SHARED.get().and_then(Option::as_ref)
            }
        };
        let refused = |shared: &Shared| {
            if shared.stopped() {
                Some(why::STOPPED)
            } else if crate::pipeline::bvh_quarantine() {
                Some(why::QUARANTINE)
            } else {
                None
            }
        };
        let shared = |make: bool| {
            let made = shared(make);
            if made.is_some() {
                seed();
            }
            made
        };
        let any = reach == Reach::Any;
        match ticket_for(&mut book, on(), super::relay(), flown, none_out, order, &declared, &written, &by_address, any, refused, shared, &self.queue) {
            Ok(ticket) => ticket,
            Err(e) => fail(format_args!("a submission could not be held behind the copy queue's fetch ({e}, VICTORY_FETCH_QUEUE=1)")),
        }
    }

    /// `Host::relay_card_back` for a landing with ticketed plans: `pieces` (`total` bytes) are
    /// fetched on the copy queue and merged with the batch's order, the batch's ticket
    /// signalled behind the copies (or alone if there are none). `Some`: whether anything was
    /// fetched. `None`: the landing goes to the drawing queue, as before (no ticketed plan; a
    /// plan of it does not qualify; the copy queue is stopped or stops now).
    pub(super) fn fetch_on_copy_queue(
        &self,
        plans: &[&CardPlan],
        pieces: &[Piece],
        total: u64,
        beside: Option<&mut super::land::RelayBack>,
        exact: bool,
    ) -> Option<bool> {
        let batch = batch(plans)?;
        landing(batch.plans);
        let Some(shared) = SHARED.get().and_then(Option::as_ref) else {
            fail(format_args!("plans with tickets and no copy queue (VICTORY_FETCH_QUEUE=1)"));
        };
        // The landing's tickets are done with: the holds they ended are timed.
        let ended = || BOOK.lock().unwrap_or_else(|e| e.into_inner()).reached(batch.value);
        // To the drawing queue: the batch's tickets are let go first (its plans are fetched as
        // before, behind whatever ran meanwhile), so that no writer waits for them for ever.
        let fall_back = |why: usize| -> Option<bool> {
            FALLBACK[why].fetch_add(batch.plans, Ordering::Relaxed);
            if let Err(e) = shared.signal(batch.value) {
                // Nothing of ours is in flight on the copy queue: stopping ends every hold.
                said(&format!("the copy queue did not take a signal ({e})"));
                if !shared.stop() {
                    fail(format_args!("the copy queue's holds could not be ended ({e}, VICTORY_FETCH_QUEUE=1)"));
                }
            }
            ended();
            None
        };
        if shared.stopped() {
            return fall_back(why::STOPPED);
        }
        if plans.iter().any(|plan| plan.fetch.is_none() && !plan.ranges.is_empty()) {
            return fall_back(why::SHARED);
        }
        if released(&batch, shared.released.load(Ordering::Acquire)) {
            return fall_back(why::RELEASED);
        }
        if pieces.is_empty() {
            if let Err(e) = shared.signal(batch.value) {
                // Nothing is in flight on the copy queue: stopping ends every hold.
                said(&format!("the copy queue did not take a signal ({e})"));
                if !shared.stop() {
                    fail(format_args!("the copy queue's holds could not be ended ({e}, VICTORY_FETCH_QUEUE=1)"));
                }
            }
            crate::shadow::relay_fetched(prune_bound(batch.order));
            ended();
            return Some(false);
        }
        if batch.value == 0 {
            fail(format_args!("pieces to fetch for plans that declared nothing (VICTORY_FETCH_QUEUE=1)"));
        }
        let mut copies: Vec<usize> = pieces.iter().map(|(shadow, ..)| Arc::as_ptr(shadow) as usize).collect();
        copies.dedup();
        if !start_reading(&copies, batch.first) {
            return fall_back(why::LATE);
        }
        let waiting = Instant::now();
        let fetched = RELAY.with(|relay| {
            let mut relay = relay.borrow_mut();
            if relay.is_none() {
                *relay = Relay::new(&self.device).ok();
            }
            let relay = relay.as_mut()?;
            if !relay.room(total, |size| self.shared_buffer(size)) {
                return None;
            }
            let parts: Vec<Part> = pieces
                .iter()
                .map(|(shadow, lo, hi, at)| Part {
                    source: &shadow.buffer,
                    lo: *lo as u64,
                    len: (hi - lo) as u64,
                    at: *at,
                })
                .collect();
            // SAFETY: a plain query. The flights of this landing have run (the relay saw
            // their values reached), so this is at least theirs.
            let ran = unsafe { self.fence.GetCompletedValue() };
            let order = Order {
                fence: &self.fence,
                ran,
                ticket: batch.value,
            };
            // The PC-side scan runs while the copies are in flight (`VICTORY_LAND_SCAN=1`).
            Some(relay.fetch(&self.device, shared, order, &parts, || {
                if let Some(back) = beside {
                    back.look();
                }
            }))
        });
        read_done();
        let fetched = match fetched {
            Some(Ok(fetched)) => fetched,
            not_sent @ (None | Some(Err(Failed::NotSent(_)))) => {
                // Nothing of it is on the copy queue: every hold can end, and this landing
                // goes to the drawing queue.
                let e = match not_sent {
                    Some(Err(Failed::NotSent(e))) => e.to_string(),
                    _ => "no list, event or memory for it".to_owned(),
                };
                said(&format!("a fetch could not be recorded or sent on the copy queue ({e})"));
                if !shared.stop() {
                    fail(format_args!("the copy queue's holds could not be ended (VICTORY_FETCH_QUEUE=1)"));
                }
                BOOK.lock().unwrap_or_else(|e| e.into_inner()).last.clear();
                return fall_back(why::STOPPED);
            }
            Some(Err(Failed::Removed)) => fail(format_args!(
                "host GPU device removed during a fetch on the copy queue of what shaders wrote to the card's memory (VICTORY_FETCH_QUEUE=1)"
            )),
            Some(Err(Failed::Unsignalled(e))) => {
                said("a fetch on the copy queue could not be signalled");
                fail(format_args!(
                    "a fetch on the copy queue was sent and could not be signalled ({e}); its writers stay held and its \
                     landing is not published (VICTORY_FETCH_QUEUE=1)"
                ))
            }
            Some(Err(Failed::Timeout(bound))) => {
                said("a fetch on the copy queue did not end");
                fail(format_args!(
                    "a fetch on the copy queue did not end within {} s on a live device; its writers stay held and its \
                     landing is not published (VICTORY_FETCH_QUEUE=1)",
                    bound.as_secs_f64()
                ))
            }
        };
        #[cfg(feature = "diagnostics")]
        crate::stalls::card_waited(crate::stalls::Site::RelayFetch, fetched.sent);
        let us = fetched.done.saturating_duration_since(waiting).as_micros() as u64;
        for (stat, from, to) in [(3, waiting, fetched.recorded), (5, fetched.recorded, fetched.sent), (6, fetched.sent, fetched.done)] {
            super::RELAY_STATS[stat].fetch_add(to.saturating_duration_since(from).as_micros() as u64, Ordering::Relaxed);
        }
        super::RELAY_STATS[8].fetch_add(total, Ordering::Relaxed);
        STATS[0].fetch_add(1, Ordering::Relaxed);
        STATS[1].fetch_add(total, Ordering::Relaxed);
        STATS[2].fetch_add(fetched.done.saturating_duration_since(fetched.sent).as_micros() as u64, Ordering::Relaxed);
        land_pieces(pieces, fetched.data, total, us, batch.order, exact);
        ended();
        Some(true)
    }
}

/// The frame-rate line's part for the copy queue, per frame, and the same numbers kept for the
/// live stats file. Nothing while the switch is off and nothing was counted.
pub(super) fn line(frames: f64) {
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    let f: Vec<f64> = FALLBACK.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    let fallback: f64 = f.iter().sum();
    super::stats::remember_fetch_queue([s[0], s[1] / 1e6, s[2] / 1000.0, s[3], s[4] / 1000.0, fallback, s[6], s[7], s[8]], &f);
    if !on() && s.iter().chain(&f).all(|&n| n == 0.0) {
        return;
    }
    let reasons: Vec<String> = why::NAMES.iter().zip(&f).map(|(name, n)| format!("{name} {n:.1}")).collect();
    tracing::info!(
        "the copy-back's fetch on the copy queue per frame (VICTORY_FETCH_QUEUE=1): {:.1} fetches, {:.1} MB, {:.1} ms from \
         send to done (the copy queue's wait for the drawing queue's fence is for flights the relay saw run before it sent: \
         none of its own); {:.1} submissions held behind a fetch for {:.1} ms in all (until the relay saw the ticket reached: \
         on the card at most that); {:.1} plans given a ticket, {:.1} of them qualified for the copy queue; {:.1} holds \
         only for writing by address; {fallback:.1} plans landed on the drawing queue ({}); {:.1} stops",
        s[0],
        s[1] / 1e6,
        s[2] / 1000.0,
        s[3],
        s[4] / 1000.0,
        s[5],
        s[7],
        s[8],
        reasons.join(", "),
        s[6],
    );
}

#[cfg(test)]
#[path = "tests/fetch.rs"]
mod fetch_tests;
