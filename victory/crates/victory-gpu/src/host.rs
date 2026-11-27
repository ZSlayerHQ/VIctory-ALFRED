//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! Submissions go to the card without a wait for each by default ([`late`],
//! at stage 3 with the relay); with `VICTORY_LATE_WAIT=0` every submission
//! is waited for, so CPU and GPU never overlap.

use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

use crate::budget::{Phase as Ph, phase as ph};

use self::relay::{hand_back, relay_worker};
use self::refresh::Batch;
use self::waits::{blocked_by, relay_wait, stale_under_flights};

#[cfg(test)]
mod messages;
pub(crate) mod land;
mod upload;
mod waits;
mod relay;
mod refresh;
mod copy;
mod hud;
mod background;
pub(crate) mod fetch;
mod stats;
#[cfg(feature = "diagnostics")]
pub(crate) mod queue_stamps;
#[cfg(test)]
pub(crate) use hud::note_sample;
pub(crate) mod hash;
mod hash_check;
pub(crate) mod recheck;
pub(crate) use copy::{copy_mapped, read_for};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Option<Presenter>>,
    /// The adapter the device is on, for its memory budget
    /// ([`Host::video_memory`]); `None` if the system does not name it.
    adapter: Option<IDXGIAdapter3>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Host objects the title let go of while a submission may still use them
/// (a texture its descriptors name, a descriptor slot's old resource), by
/// the submission count they wait for. The title's descriptor writes and
/// releases reach the host at once, from its own threads, while the host
/// runs a submission: freeing there is a GPU page fault on the freed
/// texture (a device hang).
static RETIRED: Mutex<Vec<(u64, Retired)>> = Mutex::new(Vec::new());
/// Submissions started and finished (waited for).
static STARTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FINISHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A retired object, dropped on whatever thread finishes the submission.
struct Retired(#[allow(dead_code)] Box<dyn std::any::Any>);
// SAFETY: what is retired is D3D12 objects (free-threaded) and the
// reference-counted wrappers holding them, dropped once.
unsafe impl Send for Retired {}

/// Milliseconds since the first call.
pub(crate) fn now_ms() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// This moment, for telling later how long ago it was ([`long_ago`]): the
/// frames begun so far (high half, `crate::frame::current`) and tenths of a
/// second since the start. Host copies of title memory, staging buffers and
/// host textures note when they were last used with it.
pub(crate) fn moment() -> u64 {
    (crate::frame::current() << 32) | ((now_ms() / 100) & 0xffff_ffff)
}

/// Whether `moment` lies at least `ms` milliseconds and `frames` frames
/// back. Both: at seconds per frame, what every frame uses is not touched
/// for seconds, and at many frames per second not for many frames. (A title
/// that never waits for frames has no frame count: nothing is then idle.)
pub(crate) fn long_ago(moment: u64, ms: u64, frames: u64) -> bool {
    long_ago_at(self::moment(), moment, ms, frames)
}

/// [`long_ago`] at `now` (a [`moment`]), for a caller that reads one now
/// and asks it of many things; the sweep's tests ask at a later moment.
pub(crate) fn long_ago_at(now: u64, moment: u64, ms: u64, frames: u64) -> bool {
    let tenths = (now & 0xffff_ffff).saturating_sub(moment & 0xffff_ffff);
    (now >> 32).saturating_sub(moment >> 32) >= frames && tenths * 100 >= ms
}

/// Available physical RAM on the host in bytes, from `GlobalMemoryStatusEx`.
pub(crate) fn available_physical_ram() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    let ok = unsafe { GlobalMemoryStatusEx(&mut m) } != 0;
    ok.then_some(m.ullAvailPhys)
}

/// Effective idle timeout (in milliseconds) for VRAM-backed host textures,
/// scaled down under VRAM pressure. Never returns less than 1,000 ms so
/// handles not yet recorded into a list or slot remain protected.
pub(crate) fn vram_idle_ms(video_memory: Option<[u64; 4]>) -> u64 {
    const NORMAL_MS: u64 = 5_000;
    const MODERATE_MS: u64 = 2_500;
    const HIGH_MS: u64 = 1_000;
    let Some([budget, used, ..]) = video_memory else {
        return NORMAL_MS;
    };
    if budget == 0 {
        return NORMAL_MS;
    }
    let used_pct = (u128::from(used) * 100) / u128::from(budget);
    if used_pct >= 85 {
        HIGH_MS
    } else if used_pct >= 75 {
        MODERATE_MS
    } else {
        NORMAL_MS
    }
}

/// Effective idle timeout (in milliseconds) for system-RAM caches (staging
/// buffers, loose shadows, and the mapped pipeline library), scaled down
/// when free physical RAM is low. Never returns less than 2,000 ms.
pub(crate) fn ram_idle_ms(avail_phys: Option<u64>) -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;
    const NORMAL_MS: u64 = 10_000;
    const MODERATE_MS: u64 = 5_000;
    const HIGH_MS: u64 = 2_000;
    let Some(avail) = avail_phys else {
        return NORMAL_MS;
    };
    if avail < 2 * GIB {
        HIGH_MS
    } else if avail < 4 * GIB {
        MODERATE_MS
    } else {
        NORMAL_MS
    }
}

/// How many references are held on a host object (its COM count): ours
/// all, since the host's views and commands hold none. A buffer or texture
/// with one is held by its owner alone: no descriptor slot keeps it
/// (`HostHeap::kept`) and no list (`HostList::keep`).
pub(crate) fn references(object: &impl Interface) -> u32 {
    type Method = unsafe extern "system" fn(*mut std::ffi::c_void) -> u32;
    let raw = object.as_raw();
    // SAFETY: a live COM object, whose first three methods are those of
    // `IUnknown` (query, add a reference, release one); the pair leaves the
    // count as it was, and the release returns it. The first is not called.
    unsafe {
        let methods = *raw.cast::<*const [Method; 3]>();
        ((*methods)[1])(raw);
        ((*methods)[2])(raw)
    }
}

/// Once a second, before a submission starts: lets go of what nothing
/// names any more (copies of windows of title memory, staging buffers,
/// host copies of textures the title fills).
/// `VICTORY_SWEEP=0` keeps everything, as before.
fn sweep() {
    // The in-place probe and census, if asked for (`VICTORY_INPLACE_PROBE`).
    crate::inplace::tick();
    // Tests: a delay inside the step, before the check whether to sweep
    // (in tests nearly always too soon after the last sweep).
    #[cfg(test)]
    test_hook(HOOK_SWEEP);
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_SWEEP").is_none_or(|v| v != "0")) {
        return;
    }
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let now = now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 1000
        || LAST
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return;
    }
    crate::pipeline::trim_library();
    crate::shadow::let_go_idle();
    crate::texture::let_go_idle();
    crate::resource::let_go_idle();
}

/// Drops `object` once every submission running now has finished: at once
/// if none is.
pub(crate) fn retire<T: 'static>(object: T) {
    let started = STARTED.load(Ordering::Acquire);
    if started <= FINISHED.load(Ordering::Acquire) {
        RETIREMENTS[0].fetch_add(1, Ordering::Relaxed);
        drop(object);
        return;
    }
    RETIREMENTS[1].fetch_add(1, Ordering::Relaxed);
    RETIRED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((started, Retired(Box::new(object))));
}

/// Drops what waited for submissions up to `finished`.
fn bury(finished: u64) {
    let done: Vec<(u64, Retired)> = {
        let mut retired = RETIRED.lock().unwrap_or_else(|e| e.into_inner());
        if retired.is_empty() {
            return;
        }
        let (done, keep) = std::mem::take(&mut *retired)
            .into_iter()
            .partition(|(at, _)| *at <= finished);
        *retired = keep;
        done
    };
    drop(done);
}

/// Objects [`retire`] dropped at once (no submission was running) and
/// kept until a running submission finished, since the start.
static RETIREMENTS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// What the layer did to the host's objects lately, newest last (sweeps
/// that let host copies go, pipeline library calls, saves and trims): said
/// once when the host device is removed ([`log_removal_context`]), with
/// what the submitting thread and the pipeline workers were doing at that
/// moment. A device the driver removes "internally" names no command, and
/// the pass that happens to notice first names only the view it was at:
/// three such losses each came within milliseconds of a
/// pipeline library call that no line of the log tied to them.
/// `VICTORY_TRAIL=0`: nothing is kept and nothing is said.
static TRAIL: Mutex<std::collections::VecDeque<(u64, u32, String)>> = Mutex::new(std::collections::VecDeque::new());
/// The entries kept.
const TRAIL_MOST: usize = 96;

fn trail_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TRAIL").is_none_or(|v| v != "0"))
}

/// Microseconds since the first call.
pub(crate) fn now_us() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_micros() as u64
}

/// The calling thread, as the log's other lines name threads.
pub(crate) fn thread_id() -> u32 {
    // SAFETY: a plain query.
    unsafe { windows::Win32::System::Threading::GetCurrentThreadId() }
}

/// Notes something done to the host's objects ([`TRAIL`]); `what` is only
/// run when the trail is kept.
pub(crate) fn trail(what: impl FnOnce() -> String) {
    if !trail_on() {
        return;
    }
    let entry = (now_us(), thread_id(), what());
    let mut trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    if trail.len() >= TRAIL_MOST {
        trail.pop_front();
    }
    trail.push_back(entry);
}

/// What the submitting thread is doing (0 nothing: between submissions,
/// 1 preparing one, 2 waiting for the card to run it, 3 copying back what
/// it wrote), the lists of that submission, and since when ([`now_us`]).
static SUBMITTING: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn submitting(phase: u64, lists: usize) {
    SUBMITTING[1].store(lists as u64, Ordering::Relaxed);
    SUBMITTING[2].store(now_us(), Ordering::Relaxed);
    SUBMITTING[0].store(phase, Ordering::Relaxed);
}

/// What was going on when the host device went: whether a submission was
/// on the card, the pipeline library's state and the calls running in it,
/// and the trail.
pub(crate) fn removal_context() -> String {
    use std::fmt::Write;
    let now = now_us();
    let ms = |at: u64| now.saturating_sub(at) as f64 / 1000.0;
    let phase = match SUBMITTING[0].load(Ordering::Relaxed) {
        // The submitting thread's phase only: flights sent ahead may be on
        // the card in any of them (Sol's review of the frame timers).
        0 => "between submissions",
        1 => "preparing a submission",
        2 => "waiting for a submission's fence",
        _ => "copying back what a submission wrote",
    };
    let mut text = format!(
        "the submitting thread was {phase}, of {} lists, for {:.1} ms; {}; objects let go at once {} and after a running submission {}; submissions started {} finished {}",
        SUBMITTING[1].load(Ordering::Relaxed),
        ms(SUBMITTING[2].load(Ordering::Relaxed)),
        crate::pipeline::library_removal_note(),
        RETIREMENTS[0].load(Ordering::Relaxed),
        RETIREMENTS[1].load(Ordering::Relaxed),
        STARTED.load(Ordering::Relaxed),
        FINISHED.load(Ordering::Relaxed),
    );
    let trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    let _ = write!(text, "; the last {} things done, newest last:", trail.len());
    for (at, thread, what) in trail.iter() {
        let _ = write!(text, "\n  {:.1} ms before, thread {thread}: {what}", ms(*at));
    }
    text
}

/// Says [`removal_context`] in the log, once: called by whoever finds the
/// host device removed.
pub(crate) fn log_removal_context() {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !trail_on() || SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    tracing::error!("when the host device went: {}", removal_context());
}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling); then the host's waits, none of them the card's own time:
/// waits of any thread for the relay to land flights ([`relay_wait`]: the
/// relay's wait for the card, its fetches and merges), waits of the
/// submitting thread for flights' fence ([`Host::settle`] without the
/// relay), and synchronous submissions from their send to their retirement
/// (the fence wait, and the objects let go when it is reached).
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The frame line's host waits per frame ([`SUBMISSION_US`]): for the
/// relay, for flights' fence, synchronous submissions from send to
/// retirement; summed across threads. Made from the frame line's sums
/// (`stats::waits`) and formatted only where the line is logged: making
/// them allocates nothing (Sol's review of the frame timers: the unset
/// switch added a text on every reporting tick).
struct HostWaits([f64; 3]);

impl HostWaits {
    fn all(&self) -> f64 {
        self.0.iter().sum()
    }
}

impl std::fmt::Display for HostWaits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [relay, fence, sync] = self.0;
        write!(
            f,
            "{:.1} host waits (for the relay {relay:.1}, for flights' fence {fence:.1}, synchronous submissions sent to retired {sync:.1})",
            self.all()
        )
    }
}

#[cfg(test)]
thread_local! {
    /// Tests: what a submission on this thread does at a step before it
    /// prepares ([`HOOK_ENTRY`] and on), each taken by the submission.
    static HOOKS: std::cell::RefCell<[Option<Box<dyn FnOnce()>>; 9]> =
        const { std::cell::RefCell::new([None, None, None, None, None, None, None, None, None]) };
}

/// Tests: the steps of [`HOOKS`]: in the entry's accounting, just before
/// the lock is asked for, inside the settle, inside the sweep, inside
/// making the waiting slots, in the bookkeeping after the lock; and in a
/// wait for the relay, as it begins to wait (the line held).
#[cfg(test)]
const HOOK_ENTRY: usize = 0;
#[cfg(test)]
const HOOK_LOCK: usize = 1;
#[cfg(test)]
const HOOK_SETTLE: usize = 2;
#[cfg(test)]
const HOOK_SWEEP: usize = 3;
#[cfg(test)]
const HOOK_SLOTS: usize = 4;
#[cfg(test)]
const HOOK_OTHER: usize = 5;
#[cfg(test)]
const HOOK_RELAY: usize = 6;
/// Tests: inside the check whether the reporting thread runs
/// ([`started`]), with what it holds of the reporter's slot held.
#[cfg(all(test, feature = "diagnostics"))]
const HOOK_START_CHECK: usize = 7;
/// Tests: in a fence wait's polling, each time the fence is found below
/// its value ([`wait_fence`]).
#[cfg(test)]
const HOOK_POLL: usize = 8;

/// Tests: runs this thread's hook for `step`, if one is set.
#[cfg(test)]
fn test_hook(step: usize) {
    let hook = HOOKS.with(|h| h.borrow_mut()[step].take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Tests: sets this thread's hook for `step`.
#[cfg(test)]
fn set_hook(step: usize, hook: impl FnOnce() + 'static) {
    HOOKS.with(|h| h.borrow_mut()[step] = Some(Box::new(hook)));
}

/// Tests: the host's one queue had alone by one test ([`queue_alone`]):
/// every submission of another test waits here before it takes the
/// submission lock. A test that parks a wait on the queue (a gate) would
/// otherwise hang a parallel run: another test takes the submission lock,
/// sends behind the gate and waits for the card while it holds the lock,
/// and the gated test never gets the lock it needs to send (a real-card
/// gate of `a96d7e83e`). A test that times the submission lock would count
/// other tests' holds of it.
#[cfg(test)]
static QUEUE_TESTS: std::sync::RwLock<()> = std::sync::RwLock::new(());

#[cfg(test)]
thread_local! {
    /// Tests: `VICTORY_GPU_TIMING` for this thread's submissions, whatever
    /// the environment says (`None`: as it says).
    static TIMING_FORCED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    /// Tests: blocks allocated by this thread's last synchronous submission
    /// counting its signals, and making its groups.
    static GROUPING: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
    /// Tests: a fetch's event cannot be set on its fence on this thread
    /// ([`fetch_waited`]).
    static FETCH_REGISTRATION_FAILS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tests: the device counts as removed for this thread's fence waits
    /// ([`wait_fence`]).
    static DEVICE_GONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
thread_local! {
    /// Tests: this thread's test has the queue alone ([`queue_alone`]).
    static ALONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Tests: the queue alone for this thread's test while this lives
/// ([`QUEUE_TESTS`]). Threads it starts that take the submission lock
/// directly need nothing more; a thread of its that submits must not.
#[cfg(all(test, feature = "diagnostics"))]
pub(crate) struct QueueAlone(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>);

#[cfg(all(test, feature = "diagnostics"))]
pub(crate) fn queue_alone() -> QueueAlone {
    let held = QUEUE_TESTS.write().unwrap_or_else(|e| e.into_inner());
    ALONE.with(|a| a.set(true));
    QueueAlone(held)
}

#[cfg(all(test, feature = "diagnostics"))]
impl Drop for QueueAlone {
    fn drop(&mut self) {
        ALONE.with(|a| a.set(false));
    }
}

/// The submission lock held ([`Host::submission_lock`]); in tests with the
/// share of the queue that keeps it from a test that has it alone.
#[cfg(not(test))]
type Submission<'a> = std::sync::MutexGuard<'a, u64>;

#[cfg(test)]
pub(crate) struct Submission<'a> {
    // Let go first, then the share of the queue.
    lock: std::sync::MutexGuard<'a, u64>,
    _shared: Option<std::sync::RwLockReadGuard<'static, ()>>,
}

#[cfg(test)]
impl std::ops::Deref for Submission<'_> {
    type Target = u64;
    fn deref(&self) -> &u64 {
        &self.lock
    }
}

#[cfg(test)]
impl std::ops::DerefMut for Submission<'_> {
    fn deref_mut(&mut self) -> &mut u64 {
        &mut self.lock
    }
}

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Our queue as a watcher of its fence sees it, for the frame line of every
/// run, in microseconds since the last one: how long nothing of ours was
/// queued by the fence (a host-side estimate: the card may run other work
/// then, a stall with work queued counts as queued, and an empty stretch is
/// counted at the next send); how much of that while the thread that holds
/// the submission lock was working; and the submission side's host waits
/// and fetch latency.
///
/// Every send of lists to the queue is followed by a signal on a fence of
/// the clock's own ([`Host::card_sent`]). A thread of the clock waits for
/// the newest signal and notes the moment the card reached it with nothing
/// newer sent; the next send ends that idle stretch.
static CARD_CLOCK_US: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Of the thread that holds the submission lock: the microseconds it has
/// worked so far (preparing, copying back; not its waits for the card), and
/// since when it works now ([`now_us`] plus one; 0: it does not).
static WORKING: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The thread that holds the submission lock starts (`on`) or stops
/// working; whether it worked until now.
fn working(on: bool) -> bool {
    let now = now_us() + 1;
    let since = WORKING[1].swap(if on { now } else { 0 }, Ordering::Relaxed);
    if since != 0 {
        WORKING[0].fetch_add(now.saturating_sub(since), Ordering::Relaxed);
    }
    since != 0
}

/// [`WORKING`] as of now: microseconds worked so far.
fn worked() -> u64 {
    let since = WORKING[1].load(Ordering::Relaxed);
    let now = if since != 0 { (now_us() + 1).saturating_sub(since) } else { 0 };
    WORKING[0].load(Ordering::Relaxed) + now
}

/// A submission-side host wait or fetch from `since` until now
/// ([`CARD_CLOCK_US`]).
fn stood(since: std::time::Instant) {
    CARD_CLOCK_US[2].fetch_add(since.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// What [`CARD_CLOCK_US`] is measured with.
struct CardClock {
    fence: ID3D12Fence,
    /// Signals sent so far; and, while the card has reached the last of
    /// them, since when and what [`worked`] said then.
    state: Mutex<(u64, Option<(std::time::Instant, u64)>)>,
    sent: std::sync::Condvar,
}

// SAFETY: a fence (free-threaded) behind a lock.
unsafe impl Send for CardClock {}
// SAFETY: as above.
unsafe impl Sync for CardClock {}

/// The clock's thread: waits for the newest signal sent and notes when the
/// card got there with nothing newer behind it.
fn card_clock_watch(clock: &'static CardClock) {
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        return;
    };
    let mut seen = 0;
    loop {
        let target = {
            let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
            while state.0 == seen {
                state = clock.sent.wait(state).unwrap_or_else(|e| e.into_inner());
            }
            state.0
        };
        // SAFETY: the clock's fence and this thread's event.
        unsafe {
            if clock.fence.SetEventOnCompletion(target, event).is_ok() {
                WaitForSingleObject(event, INFINITE);
            }
        }
        seen = target;
        let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.0 == target && state.1.is_none() {
            state.1 = Some((std::time::Instant::now(), worked()));
        }
    }
}

/// For the frame-rate line: [`CARD_CLOCK_US`] per frame.
fn card_clock_line(frames: f64) {
    let [idle, behind, stood] = [0, 1, 2].map(|i| CARD_CLOCK_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
    tracing::info!(
        "our queue per frame, as a watcher of its fence sees it: nothing of ours queued {idle:.1} ms on the drawing queue (a \
         host-side estimate, not the card's load and not idle time: the copy queue's fetches are not watched), {behind:.1} ms of that while the thread holding the submission lock was working, \
         the rest while it was not (between submissions or in a host wait); submission-side host waits and fetch latency \
         {stood:.1} ms"
    );
}

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Host buffers made since the start, as (count, bytes): in the upload heap
/// (staging, list scratch), shared with the processor (host copies of title
/// memory) and the graphics card's own (indirect arguments, twins' copies).
/// For the memory line: against what is alive, they show what was let go.
pub(crate) static MADE: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// What of that is alive now, for the memory line (`MADE` counts what was
/// ever made: tens of thousands of upload buffers in minutes of a run,
/// and nothing said how many were alive): host lists; the lists' scratch
/// chunks and their bytes; of those, the chunks no list holds (kept for the
/// next list, [`Spare`]) and their bytes; chunks taken from those kept
/// instead of made, since the start; and the lists' other upload buffers
/// (texture data copied from buffers, scratch too big for a chunk) and
/// their bytes.
pub(crate) static ALIVE: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// About host lists, for the memory line: lists made by the host since the
/// start; preludes made (a list's second allocator and list, made when it
/// first needs one, [`lazy_prelude`]); and, in diagnostics builds, the
/// bytes the process's private memory grew by while lists were being made
/// (what a list costs: host objects with the driver's command memory;
/// hundreds of lists were alive in a run, each with a prelude it may never use).
/// Then the lists alive that were never reset since they were made: the
/// host sides of title lists the title made and has not used (a host list
/// costs most of a megabyte of private memory: hundreds of megabytes for the
/// lists a guest makes at its start).
pub(crate) static LISTS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The process's private bytes (diagnostics: around the making of lists).
#[cfg(feature = "diagnostics")]
fn private_bytes() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    c.PagefileUsage as u64
}

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
            log_removal_context();
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    // Tests have no log: the reason goes to their output.
    #[cfg(test)]
    eprintln!("host GPU failure: {text}");
    // The run ends here: the live segment under way is logged first.
    crate::live::finish();
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes). It never panics: the title's
/// calls (`extern "system"`, through which no panic can unwind) reach it
/// first. A test's own body asks with [`test_host`].
pub(crate) fn host() -> Option<&'static Host> {
    static HOST: OnceLock<Option<Host>> = OnceLock::new();
    HOST.get_or_init(|| match create() {
        Ok(host) => {
            // Before any list can complete: the walls trace and the refresh
            // check hand lines to the reporting thread from there and from
            // submissions, and never start it there.
            #[cfg(feature = "diagnostics")]
            if (walls_trace() || crate::shadow::refresh_check_on()) && !reports_start() {
                tracing::warn!("no diagnostics reporting thread: walls trace and refresh check lines are dropped and counted");
            }
            Some(host)
        }
        Err(NoHost::Refused(text)) => {
            refuse(&text);
            None
        }
        Err(NoHost::Failed(e)) => {
            // Tests: `VICTORY_REQUIRE_GPU` makes no device a failure, not a
            // skip; so does `VICTORY_D3D12_WARP=1`, whose test asked for the
            // software adapter and must not pass by skipping.
            #[cfg(test)]
            {
                if std::env::var_os("VICTORY_D3D12_WARP").is_some_and(|v| v == "1") {
                    refuse(&format!(
                        "VICTORY_D3D12_WARP=1 and no software device was made: the test did not run ({e})"
                    ));
                    return None;
                }
                if let Err(failed) = victory_shader::adapter::require(victory_shader::adapter::Got::Nothing) {
                    refuse(&format!("{failed} ({e})"));
                    return None;
                }
            }
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

/// Why the tests are not to have the host device: a refusal (`VICTORY_ADAPTER`) or a gate's
/// requirement not met (`VICTORY_REQUIRE_GPU`), set when the device was to be made. [`host`]
/// is `None` then, and [`test_host`] fails each test that asks for it.
#[cfg(test)]
static UNUSABLE: OnceLock<String> = OnceLock::new();

/// The host device for a test's own body, never for code the title's calls reach: [`host`],
/// and a failure of the test, not a skip, where the device was refused or a gate's
/// requirement is not met ([`UNUSABLE`]).
#[cfg(test)]
#[track_caller]
pub(crate) fn test_host() -> Option<&'static Host> {
    let host = host();
    if let Some(why) = UNUSABLE.get() {
        panic!("{why}");
    }
    host
}

/// Why there is no host device.
enum NoHost {
    /// `VICTORY_ADAPTER` names no adapter there is, or one that cannot run the host
    /// ([`victory_shader::adapter`]): the run does not start ([`refuse`]).
    Refused(String),
    /// The device could not be made: a run goes on without one, as before.
    Failed(String),
}

impl From<windows::core::Error> for NoHost {
    fn from(e: windows::core::Error) -> NoHost {
        NoHost::Failed(e.to_string())
    }
}

impl From<victory_shader::adapter::Unmade> for NoHost {
    fn from(unmade: victory_shader::adapter::Unmade) -> NoHost {
        match unmade {
            victory_shader::adapter::Unmade::Refused(text) => NoHost::Refused(text),
            victory_shader::adapter::Unmade::Failed(text) => NoHost::Failed(text),
        }
    }
}

/// `VICTORY_ADAPTER` names no adapter there is, or one that cannot run the host (its device
/// is not made at feature level 12_0 or lacks shader model 6.6): the run ends before the
/// title has a device, with the message (which lists the adapters' descriptions) in the log,
/// on standard error, and in the crash report where the runtime set one ([`on_failure`]).
/// Tests: the host is `None`, the message said once, and each test that asks for the host
/// fails with it ([`test_host`]); nothing panics here, where the title's calls may be.
fn refuse(text: &str) {
    tracing::error!("{text}");
    #[cfg(test)]
    {
        victory_shader::adapter::say_once(text);
        let _ = UNUSABLE.set(text.to_owned());
    }
    #[cfg(not(test))]
    {
        eprintln!("VIctory: {text}");
        match ON_FAILURE.get() {
            Some(report) => report(text),
            None => std::process::exit(1),
        }
    }
}

/// The adapter the host device was made on (by its LUID; `None` where DXGI does not describe
/// it) and its line, for the test that checks the line names the device's own adapter.
#[cfg(test)]
static CHOSEN: OnceLock<(Option<victory_shader::adapter::Adapter>, String)> = OnceLock::new();

#[cfg(feature = "diagnostics")]
mod observers;

fn create() -> Result<Host, NoHost> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // Whether GPU-based validation asked for actually came on, for the observers line.
        #[cfg(feature = "diagnostics")]
        let mut gbv_on = false;
        // The debug layer only in builds with diagnostics.
        let debug =
            cfg!(feature = "diagnostics") && crate::switches::switch_on("VICTORY_D3D12_DEBUG");
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned), unless
            // `VICTORY_D3D12_GBV=states`: then the states of the resources
            // shaders read and write are checked too.
            let gbv = crate::switches::switch_on("VICTORY_D3D12_GBV");
            if gbv
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                    #[cfg(feature = "diagnostics")]
                    {
                        gbv_on = true;
                    }
                }
                let states = std::env::var_os("VICTORY_D3D12_GBV").is_some_and(|v| v == "states");
                if !states && let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!(states, "GPU-based validation on");
            }
        }
        // Every diagnostic that costs frame time and is on, in one line, so a log says what
        // it measured under.
        #[cfg(feature = "diagnostics")]
        observers::say(debug, gbv_on);
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        // The adapter by the one rule the test hosts use too (`VICTORY_ADAPTER`; the default
        // rule's second try where its pick cannot run the host), said in one line, the adapter
        // the device's LUID names. Diagnostics: the software adapter (`VICTORY_D3D12_WARP`),
        // on which a shader's wild access is an access violation in this process (a crash
        // report with the address and instruction) instead of a GPU fault and driver reset.
        let made = victory_shader::adapter::make_device(on_warp().then_some("VICTORY_D3D12_WARP"))?;
        let line = made.line();
        tracing::info!("{line}");
        // Tests have no log: the line goes to their output, once; and a gate's
        // `VICTORY_REQUIRE_GPU` holds for the host's device, which is a test's request for
        // hardware (put on the software adapter by `VICTORY_D3D12_WARP`, which the requirement
        // then fails), an adapter DXGI does not describe failing it too: the device is not
        // the tests', which fail on it as on a refusal ([`test_host`]).
        #[cfg(test)]
        {
            victory_shader::adapter::say_once(&line);
            if let Err(failed) = victory_shader::adapter::require(made.adapter.as_ref().into()) {
                return Err(NoHost::Refused(failed));
            }
            let _ = CHOSEN.set((made.adapter.clone(), line));
        }
        let device = made.device;
        VRAM_ADAPTER.remember(&device, caption_on());
        // `VICTORY_D3D12_BACKGROUND`: the driver's background processing
        // (`ID3D12Device6::SetBackgroundProcessingMode`), unset by default
        // (`host/background.rs`).
        background::apply(&device);
        // the waves the GPU runs: a widest of 32 lanes (NVIDIA) gets console wave64 compute
        // programs whose cross-lane code stays in 32-lane halves translated for waves of 32,
        // and a program that fixes a wave size the device lacks is refused when translated
        let mut o1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        let answered = device
            .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, (&raw mut o1).cast(), size_of_val(&o1) as u32)
            .is_ok();
        let answer = answered.then(|| (o1.WaveLaneCountMin, o1.WaveLaneCountMax, o1.WaveOps.as_bool()));
        record_host_waves(victory_shader::emit::process_host(), answer);
        // Whether the card's memory is its own and the processor can write
        // it directly (GPU upload heaps), once.
        crate::memclass::log_card_memory(&device);
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        let adapter = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
            .ok();
        tracing::info!("host D3D12 device ready");
        let host = Host {
            device,
            queue,
            fence,
            submitted: Mutex::new(0),
            event,
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(None),
            adapter,
        };
        // What the layer said while the device was made: logged here, and in tests kept in
        // the start-up bucket, so a test reads only the messages its own work caused.
        host.log_messages_inner(true);
        Ok(host)
    }
}

/// Records what a new device said of its waves (`D3D12_FEATURE_DATA_D3D12_OPTIONS1`: the
/// fewest lanes, the most, wave operations) as the host the translator works for, and
/// returns it. `None`: the query failed, and the host is unknown from here on. It is said,
/// and what an earlier device answered does not stay behind: a second device was once
/// translated for with the first one's widest wave (the recheck of job/land-2e, Question 2).
pub(crate) fn record_host_waves(
    host: &victory_shader::emit::HostCell,
    answer: Option<(u32, u32, bool)>,
) -> victory_shader::emit::HostCaps {
    use victory_shader::emit::{HostCaps, HostWaves};
    let caps = match answer {
        Some((min, max, wave_ops)) => {
            tracing::info!(min, max, wave_ops, "host wave lanes");
            HostCaps::Known(HostWaves { min, max, wave_ops })
        }
        None => {
            tracing::warn!(
                "the device did not say its wave sizes: no host-specific form is taken, and a program that fixes a wave size is handed over unchecked"
            );
            HostCaps::Unknown
        }
    };
    host.set(caps);
    caps
}

/// Whether the host device is the software adapter (WARP): diagnostics
/// builds, and this crate's tests in any build, with `VICTORY_D3D12_WARP` on
/// ([`crate::switches::switch_on`]). The tests too: a plain-build test run
/// with the switch set had the real card, and the fixtures that stall a queue
/// on purpose reset its engine (2026-11-26).
/// (DXGI may give hardware for the software adapter, notably under Proton:
/// the device is then refused, `victory_shader::adapter::software_adapter`.)
pub(crate) fn on_warp() -> bool {
    (cfg!(feature = "diagnostics") || cfg!(test)) && crate::switches::switch_on("VICTORY_D3D12_WARP")
}

/// A device for a test of this crate on the software adapter, by the test's own request (not
/// a switch, and not held to `VICTORY_REQUIRE_GPU`), its line said once a process; `None`
/// without one.
#[cfg(test)]
pub(crate) fn test_warp_device() -> Option<ID3D12Device> {
    let made = victory_shader::adapter::test_device(victory_shader::adapter::Asked::Software, None);
    made.ok().map(|made| made.device)
}

#[cfg(test)]
mod tests {
    /// A device that answers the wave query and then one that does not: the translator's
    /// host is the second device's, unknown, not the first one's waves left behind.
    #[test]
    fn a_failed_wave_query_leaves_no_earlier_devices_waves_behind() {
        use victory_shader::emit::{HostCaps, HostCell, HostWaves};
        let host = HostCell::new();
        let first = HostCaps::Known(HostWaves { min: 32, max: 32, wave_ops: true });
        assert_eq!(super::record_host_waves(&host, Some((32, 32, true))), first);
        assert_eq!(host.get(), first);
        assert_eq!(super::record_host_waves(&host, None), HostCaps::Unknown);
        assert_eq!(host.get(), HostCaps::Unknown, "the first device's waves stayed for the second");
        // and a device's answer is taken as it is given, wave operations included
        let third = HostCaps::Known(HostWaves { min: 4, max: 4, wave_ops: false });
        assert_eq!(super::record_host_waves(&host, Some((4, 4, false))), third);
        assert_eq!(host.get(), third);
    }

    #[test]
    fn sparse_copies_preserve_islands_and_observe_remaps() {
        use windows_sys::Win32::System::Memory::*;
        const N: usize = 5 << 16;
        // SAFETY: we reserve a private region, commit selected pages inside it,
        // and only directly access the pages while they are writable.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), N, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            let mut expected = vec![0; N];
            for (offset, value) in [(0, 9), (2 << 16, 37), (N - 4096, 85)] {
                let at = base.add(offset);
                assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                at.write_bytes(value, 4096);
                expected[offset..offset + 4096].fill(value);
            }
            for remap in [false, true] {
                if remap {
                    let mut old = 0;
                    assert_ne!(VirtualProtect(base.cast(), 4096, PAGE_NOACCESS, &mut old), 0);
                    expected[..4096].fill(0);
                    let at = base.add(1 << 16);
                    assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                    at.write_bytes(0xc7, 4096);
                    expected[1 << 16..(1 << 16) + 4096].fill(0xc7);
                }
                let mut out = vec![0xee; N];
                super::copy_mapped(out.as_mut_ptr().add(27), base.add(27), N - 54, false);
                assert_eq!(out[27..N - 27], expected[27..N - 27]);
                assert!(out[..27].iter().chain(&out[N - 27..]).all(|&v| v == 0xee));
            }
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    use super::*;

    /// What a list of ours run under the submission lock records and runs
    /// before any other submission: another thread's, asked for while it
    /// records, waits until it has run (Sol's re-check, finding 9).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_list_run_under_the_lock_keeps_other_submissions_out() {
        let Some(host) = test_host() else {
            return; // no host GPU here
        };
        let (inside, recording) = std::sync::mpsc::channel::<()>();
        let other = std::thread::spawn(move || {
            recording.recv().unwrap();
            let list = host.new_list().unwrap();
            list.close();
            host.execute(&[&list]);
            Instant::now()
        });
        let mut recorded = None;
        assert!(host.run_own_locked(|_| {
            inside.send(()).unwrap();
            std::thread::sleep(Duration::from_millis(200));
            recorded = Some(Instant::now());
        }));
        let other_ran = other.join().unwrap();
        assert!(other_ran >= recorded.unwrap(), "the other submission ran while ours was recorded");
    }

    /// Host figures say what they measure (Sol's review of the frame
    /// timers, Question 5): with the relay the settles' flights each time
    /// are not a ratio that reads 0, and the removal report says what the
    /// submitting thread was doing without claiming the card was idle.
    #[test]
    fn host_figures_are_not_called_the_card_s() {
        assert_eq!(Each(Some(2.0)).to_string(), "2.0 flights each time");
        assert_eq!(Each(None).to_string(), "flights landed by the relay");
        let context = removal_context();
        assert!(context.starts_with("the submitting thread was "), "{context}");
        assert!(!context.contains("nothing of ours"), "{context}");
    }

    /// This thread's budget for the frame under way, by phase, in ms: the
    /// thread is made a presenting one first, so that its phases stay its own
    /// (`crate::budget`).
    #[cfg(feature = "diagnostics")]
    fn phases_open() -> [f64; crate::budget::PHASES] {
        crate::budget::peek_open().map(|ns| ns as f64 / 1e6)
    }

    /// The wait reported for the submission lock is its acquisition alone:
    /// another thread holds the lock and lets it go 150 ms after the
    /// submission asks for it (a barrier where it asks); the submission
    /// then works 100 ms in its bookkeeping, which is reported apart.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn the_lock_wait_is_its_acquisition_and_the_work_after_it_is_apart() {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        // No other test's hold of the lock in the wait timed.
        let _alone = queue_alone();
        let list = host.new_list().unwrap();
        list.close();
        crate::budget::frame();
        let (taken, held) = std::sync::mpsc::channel::<()>();
        let (asking, asked) = std::sync::mpsc::channel::<()>();
        let holder = std::thread::spawn(move || {
            let lock = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            taken.send(()).unwrap();
            asked.recv().unwrap();
            std::thread::sleep(Duration::from_millis(150));
            drop(lock);
        });
        held.recv().unwrap();
        set_hook(HOOK_LOCK, move || asking.send(()).unwrap());
        set_hook(HOOK_OTHER, || std::thread::sleep(Duration::from_millis(100)));
        host.execute(&[&list]);
        holder.join().unwrap();
        let phases = phases_open();
        let (lock, other) = (phases[Ph::SubmissionLockWait as usize], phases[Ph::Unaccounted as usize]);
        assert!(
            (150.0..250.0).contains(&lock),
            "the lock's wait reported as {lock:.1} ms; it was let go 150 ms after it was asked for"
        );
        assert!((100.0..200.0).contains(&other), "the 100 ms after the lock reported apart as {other:.1} ms");
    }

    /// Each moment from a submission's call to its preparing has one home:
    /// a delay in each part shows in that part alone, the entry's
    /// accounting included (Sol's review of the frame timers: it was in no
    /// part, and the bookkeeping after the lock was "waiting for flights").
    /// Corrected with the budget: the parts are its phases, and the entry
    /// and the bookkeeping, which had a part each, are its `unaccounted`.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn each_part_before_preparing_holds_its_own_delay() {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        // No other test's hold of the lock in the lock's part.
        let _alone = queue_alone();
        let list = host.new_list().unwrap();
        list.close();
        crate::budget::frame();
        // Delays 100 ms apart: one landing in a neighbouring part shows.
        let delays = [(HOOK_ENTRY, 50), (HOOK_SETTLE, 150), (HOOK_SWEEP, 250), (HOOK_SLOTS, 350), (HOOK_OTHER, 450)];
        for (step, ms) in delays {
            set_hook(step, move || std::thread::sleep(Duration::from_millis(ms)));
        }
        host.execute(&[&list]);
        let phases = phases_open();
        // The entry's accounting (50) and the bookkeeping after the lock (450)
        // are in no named phase: both are `unaccounted`, and nothing else is.
        let want = [(Ph::Unaccounted, 500.0), (Ph::SubmissionLockWait, 0.0), (Ph::Settle, 150.0), (Ph::Sweep, 250.0), (Ph::Slots, 350.0)];
        for (phase, want) in want {
            let (got, name) = (phases[phase as usize], crate::budget::NAMES[phase as usize]);
            assert!((want..want + 60.0).contains(&got), "{name}: {got:.1} ms for a delay of {want} ms ({phases:?})");
        }
        // And the submission is the sum of its phases: nothing counted twice.
        let all: f64 = phases.iter().sum::<f64>() - phases[Ph::Title as usize];
        assert!((1200.0..1500.0).contains(&all), "delays of 1200 ms in all, phases of {all:.1} ms ({phases:?})");
    }

    /// A wait for the relay is reported as one, and not as time on the
    /// card: the lander says the flights landed 120 ms after the waiter is
    /// inside its wait (a handshake there), and the frame line's text says
    /// a wait for the relay of that length. Only the relay's counter is
    /// read (nothing else adds to it here: the relay is off in tests).
    #[test]
    fn a_relay_wait_is_reported_as_one_not_as_time_on_the_card() {
        let from = SUBMISSION_US[1].load(Ordering::SeqCst);
        let landed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (inside, waiting) = std::sync::mpsc::channel::<()>();
        let lander = {
            let landed = landed.clone();
            std::thread::spawn(move || {
                waiting.recv().unwrap();
                std::thread::sleep(Duration::from_millis(120));
                let held = line();
                landed.store(true, Ordering::SeqCst);
                drop(held);
                LINE_LANDED.notify_all();
            })
        };
        set_hook(HOOK_RELAY, move || inside.send(()).unwrap());
        relay_wait(WHY_READ, |_| landed.load(Ordering::SeqCst));
        lander.join().unwrap();
        let relay_ms = (SUBMISSION_US[1].load(Ordering::SeqCst) - from) as f64 / 1000.0;
        assert!((120.0..220.0).contains(&relay_ms), "the relay waited for {relay_ms:.1} ms");
        // The frame line's part, as it is written into the line.
        let text = HostWaits([relay_ms, 0.0, 0.0]).to_string();
        assert!(!text.contains("GPU") && !text.contains("card"), "a wait for the relay reported as the card's time: {text}");
        assert_eq!(
            text,
            format!(
                "{relay_ms:.1} host waits (for the relay {relay_ms:.1}, for flights' fence 0.0, synchronous submissions sent to retired 0.0)"
            )
        );
    }

    /// Making the frame line's host waits and passing them to a line that
    /// is filtered out allocates nothing (Sol's review of the frame timers:
    /// the text was made on every reporting tick, the switch unset). From
    /// sums of its own: taking the run's raced the relay wait fixture's
    /// reading of them (Sol's third read). Corrected with the frame line:
    /// its waits are made from its interval's sums, no longer taken from
    /// the counters where it is logged.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn the_frame_line_s_waits_allocate_nothing_when_filtered() {
        // The frame line's sums, as the meter keeps them (made before counting).
        let mut line = stats::Interval::new();
        line.present(1, 100.0);
        line.sums = [1500; stats::sum::COUNT];
        struct WarnOnly;
        impl tracing::Subscriber for WarnOnly {
            fn enabled(&self, m: &tracing::Metadata<'_>) -> bool {
                *m.level() <= tracing::Level::WARN
            }
            fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
                Some(tracing::level_filters::LevelFilter::WARN)
            }
            fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
                tracing::span::Id::from_u64(1)
            }
            fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
            fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
            fn event(&self, _: &tracing::Event<'_>) {}
            fn enter(&self, _: &tracing::span::Id) {}
            fn exit(&self, _: &tracing::span::Id) {}
        }
        tracing::subscriber::with_default(WarnOnly, || {
            // The callsite's interest is settled before counting.
            tracing::info!("warm");
            let n = crate::copystat::allocations_in(|| {
                let waits = stats::waits(&line);
                tracing::info!("x, {waits}; y");
            });
            assert_eq!(n, 0, "blocks allocated taking the frame line's waits with the line filtered");
        });
    }

    /// The wait of both fetches of card copies' pieces, the relay's and a
    /// synchronous submission's (`Host::card_back`, which merged on its
    /// event's wake alone: Sol's third read of the frame timers), through
    /// each outcome: a failed registration fails at once; a wake before the
    /// fence has its value (the event set already) waits on until it has
    /// it; a removed device (its fence at `u64::MAX`) gives false, not
    /// done; a fence that has its value gives done. Nothing is timed: the
    /// fence gets its value from inside the wait's polling, which a wait
    /// that ended at the wake never reaches (the timed signaller of the
    /// fixture before could win the race).
    #[test]
    fn a_fetch_s_wait_ends_only_at_its_fence_on_a_live_device() {
        type Wait = fn(&ID3D12Device, &ID3D12Fence, HANDLE, u64) -> windows::core::Result<bool>;
        let waits: [(&str, Wait); 1] = [("the fetches' wait", fetch_waited)];
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        // SAFETY: plain objects of our own; an auto-reset event, as the
        // fetches' are.
        let fresh = || unsafe {
            let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            (fence, CreateEventW(None, false, false, None).unwrap())
        };
        // SAFETY: plain queries and signals of our fences, from the CPU;
        // our events.
        let done = |fence: &ID3D12Fence| unsafe { fence.GetCompletedValue() };
        let close = |event: HANDLE| unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(event);
        };
        for (name, wait) in waits {
            let (fence, event) = fresh();
            FETCH_REGISTRATION_FAILS.with(|f| f.set(true));
            let failed = wait(&host.device, &fence, event, 3);
            FETCH_REGISTRATION_FAILS.with(|f| f.set(false));
            assert!(failed.is_err(), "{name}: a failed registration gave {failed:?}");
            close(event);

            let (fence, event) = fresh();
            // SAFETY: our event: a wake before the fence moved.
            unsafe { windows::Win32::System::Threading::SetEvent(event) }.unwrap();
            let polled = std::rc::Rc::new(std::cell::Cell::new(None));
            {
                let (polled, fence) = (polled.clone(), fence.clone());
                set_hook(HOOK_POLL, move || {
                    polled.set(Some(done(&fence)));
                    // SAFETY: as above.
                    unsafe { fence.Signal(3) }.unwrap();
                });
            }
            let early = wait(&host.device, &fence, event, 3);
            HOOKS.with(|h| h.borrow_mut()[HOOK_POLL] = None);
            assert_eq!(early.ok(), Some(true), "{name}: an early wake, then the fence");
            assert_eq!(polled.get(), Some(0), "{name}: the wait ended at the early wake, its fence below the value");
            assert!(done(&fence) >= 3);
            close(event);

            let (fence, event) = fresh();
            // A removed device's fence reads `u64::MAX`, which wakes the
            // event as if every value had been reached.
            // SAFETY: as above.
            unsafe { fence.Signal(u64::MAX) }.unwrap();
            DEVICE_GONE.with(|g| g.set(true));
            let gone = wait(&host.device, &fence, event, 3);
            DEVICE_GONE.with(|g| g.set(false));
            assert_eq!(gone.ok(), Some(false), "{name}: a removed device taken for a fetch that ran");
            close(event);

            let (fence, event) = fresh();
            // SAFETY: as above.
            unsafe { fence.Signal(3) }.unwrap();
            assert_eq!(wait(&host.device, &fence, event, 3).ok(), Some(true), "{name}: a fence with its value");
            close(event);
        }
    }

    /// A list of ours run under the submission lock is not stamped with the
    /// stamps on: its pair would be recorded and resolved and never read
    /// (Sol's recheck of the frame timers, Question 1, item 2). No query
    /// heap is taken for it and no send noted.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_list_of_ours_run_under_the_lock_is_not_stamped() {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let sink = crate::stamps::Sink::new();
        let taken = crate::stamps::taken();
        assert!(host.run_own_locked(|_| {}));
        assert_eq!(crate::stamps::taken(), taken, "a query heap taken for a list whose stamps no one reads");
        assert!(sink.spans().is_empty() && sink.unread() == 0);
    }

    /// With the stamps off a synchronous submission counts its signals
    /// without allocating and makes its groups once (one block, as
    /// `job/land-2d` did), with `VICTORY_GPU_TIMING` off and on (Sol's
    /// third read of the frame timers: the count built the groups' ranges
    /// on the heap, and they were built again to send).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_submission_counts_its_signals_without_allocating() {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let _off = crate::stamps::Off::new();
        for timing in [false, true] {
            TIMING_FORCED.with(|t| t.set(Some(timing)));
            let list = host.new_list().unwrap();
            list.close();
            GROUPING.with(|g| g.set((u64::MAX, u64::MAX)));
            host.execute(&[&list]);
            TIMING_FORCED.with(|t| t.set(None));
            let (signals, groups) = GROUPING.with(std::cell::Cell::get);
            assert_eq!(
                (signals, groups),
                (0, 1),
                "timing {timing}: blocks allocated counting the signals, and making the groups"
            );
        }
    }

    /// Every group a synchronous submission sends has a signal after it.
    /// With `VICTORY_GPU_TIMING` and no title list there is none, as on
    /// `job/land-2d` (Sol's fourth read of the frame timers: one empty
    /// group went); a hash of copies in the card's memory sent then is not
    /// stamped ([`Host::card_before`]).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_submission_s_groups_each_have_a_signal() {
        let lists = [0, 1, 2, 3, 4];
        let groups = |timing: bool, firsts: &[usize], len: usize| -> Vec<Vec<i32>> {
            let made = send_groups(timing, firsts, &lists[..len]);
            assert_eq!(made.len() as u64, send_signals(timing, firsts), "the signals counted are the groups sent");
            made.into_iter().map(<[i32]>::to_vec).collect()
        };
        assert!(groups(true, &[], 0).is_empty(), "timing on, no list of the title's: no group, as job/land-2d");
        assert_eq!(groups(true, &[0, 2], 5), [vec![0, 1], vec![2, 3, 4]], "timing on: a group a list, the compare with the last");
        assert_eq!(groups(false, &[0, 2], 5), [vec![0, 1, 2, 3, 4]]);
        assert_eq!(groups(false, &[], 0), [vec![]]);
    }

    /// A synchronous submission with no title list, run: with
    /// `VICTORY_GPU_TIMING` nothing goes to the queue after it, no signal of
    /// the host's fence and no send to the card's clock (so no wait), as on
    /// `job/land-2d`, which iterated no group; without the switch one group
    /// goes and is signalled, as there (the control). Sol's fourth read of
    /// the frame timers: with the switch it sent an empty group, signalled
    /// and waited.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_timed_submission_without_a_title_list_sends_nothing() {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let _off = crate::stamps::Off::new();
        // No other test's submission between the counts.
        let _alone = queue_alone();
        let counts = || {
            let signals = *host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            let clock = host.card_clock().map_or(0, |c| c.state.lock().unwrap_or_else(|e| e.into_inner()).0);
            (signals, clock)
        };
        for (timing, sent) in [(true, 0), (false, 1)] {
            TIMING_FORCED.with(|t| t.set(Some(timing)));
            let before = counts();
            host.execute(&[]);
            TIMING_FORCED.with(|t| t.set(None));
            let after = counts();
            assert_eq!(
                (after.0 - before.0, after.1 - before.1),
                (sent, sent),
                "timing {timing}: signals of the host's fence, and sends to the card's clock"
            );
        }
    }

    /// Tests: a fence of ours the host's queue waits for, parked with the
    /// queue had alone ([`queue_alone`]: no other test sends behind it and
    /// holds the submission lock while it waits), and opened when the gate
    /// goes if not before, a failed assertion included (the queue never
    /// stays shut for the tests after).
    #[cfg(feature = "diagnostics")]
    struct Gate {
        fence: ID3D12Fence,
        /// Read before its watchdog's thread began: the watchdog's time runs
        /// from a later moment, so a gate opened by it has been shut at least
        /// that time since this.
        armed: Instant,
        // Let go when the gate goes: its watchdog then opens it at once.
        _release: std::sync::mpsc::Sender<()>,
        _alone: QueueAlone,
    }

    /// How long a test gate stays shut at most: its watchdog then opens it,
    /// whatever its test is doing (Sol's third read of the frame timers: a
    /// test that waits behind its own gate never reaches the drop that
    /// opens it).
    #[cfg(feature = "diagnostics")]
    const GATE_MOST: Duration = Duration::from_secs(30);

    #[cfg(feature = "diagnostics")]
    impl Gate {
        /// The queue alone, then the gate parked on it.
        fn park(host: &Host) -> Gate {
            Gate::park_for(host, GATE_MOST)
        }

        /// [`Gate::park`], its watchdog opening it after `most`.
        fn park_for(host: &Host, most: Duration) -> Gate {
            let alone = queue_alone();
            // SAFETY: a plain fence of our own; the queue waits for it.
            let fence: ID3D12Fence = unsafe { host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
            let (release, released) = std::sync::mpsc::channel::<()>();
            let armed = Instant::now();
            {
                let fence = fence.clone();
                std::thread::spawn(move || {
                    if released.recv_timeout(most) == Err(std::sync::mpsc::RecvTimeoutError::Timeout) {
                        eprintln!("a test gate was opened by its watchdog after {most:?}");
                    }
                    // SAFETY: as in `open`.
                    let _ = unsafe { fence.Signal(1) };
                });
            }
            // SAFETY: as above.
            unsafe { host.queue.Wait(&fence, 1) }.unwrap();
            Gate { fence, armed, _release: release, _alone: alone }
        }

        fn open(&self) {
            // SAFETY: a plain signal of our fence from the CPU.
            unsafe { self.fence.Signal(1) }.unwrap();
        }
    }

    #[cfg(feature = "diagnostics")]
    impl Drop for Gate {
        fn drop(&mut self) {
            // SAFETY: as in `open`; signalled again it stays open.
            let _ = unsafe { self.fence.Signal(1) };
        }
    }

    /// A gate its test never opens is opened by its watchdog: a submission
    /// sent behind it, and the test's wait for that submission, end. The
    /// time is from the moment read before the watchdog's thread began
    /// ([`Gate::armed`]), so the bound holds by the order of the reads
    /// however late the watchdog's thread or this one runs; a moment read
    /// after the gate was parked came after the watchdog's time began.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_gate_left_shut_is_opened_by_its_watchdog() {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let list = host.new_list().unwrap();
        list.close();
        let gate = Gate::park_for(host, Duration::from_millis(200));
        let (submitted, sent) = host.submit(&[&list], true).expect("sent ahead");
        let target = *submitted;
        drop((submitted, sent));
        assert!(wait_fence(&host.device, &host.fence, None, target), "a live device");
        let shut = gate.armed.elapsed();
        assert!(shut >= Duration::from_millis(200), "the submission ran with the gate shut: done {shut:?} after the gate was armed");
        drop(gate);
    }

    /// A list with real work, a clamp group inside it: nothing of its
    /// stamps is read while the queue stands before it (a fence of ours
    /// holds it), and once its fence has passed its pair and the group's
    /// are read, the group's strictly inside the list's (copies before and
    /// after it), in a send that went ahead as a flight's does.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_list_s_pairs_are_nested_and_read_only_after_its_fence() {
        use crate::stamps::{Kind, Sink};
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let sink = Sink::new();
        let (src, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: inside the mapping: two mesh records.
        unsafe { (data as *mut [u32; 6]).write([3, 1, 1, 0, 1, 1]) };
        let big: Vec<ID3D12Resource> = (0..4).map(|_| host.uav_buffer(8 << 20).unwrap()).collect();
        let list = host.new_list().unwrap();
        // SAFETY: an open list; live buffers in the common state.
        unsafe { list.list.CopyBufferRegion(&big[1], 0, &big[0], 0, 8 << 20) };
        let scratch = crate::indirect::bound(&list, (&src, 0), None, (10, 12, 2)).expect("the clamp");
        crate::indirect::done(&list, &scratch);
        // SAFETY: as above.
        unsafe { list.list.CopyBufferRegion(&big[3], 0, &big[2], 0, 8 << 20) };
        list.close();
        let gate = Gate::park(host);
        let (submitted, sent) = host.submit(&[&list], true).expect("sent ahead");
        let target = *submitted;
        drop((submitted, sent));
        std::thread::sleep(Duration::from_millis(50));
        list.stamps_landed();
        assert!(sink.spans().is_empty(), "read before the fence passed: {:?}", sink.spans());
        assert_eq!((sink.flaws().early, sink.unread()), (1, 1), "the early landing is said and the send stays unread");
        gate.open();
        assert!(wait_fence(&host.device, &host.fence, None, target));
        list.stamps_landed();
        let spans = sink.spans();
        let main = *spans.iter().find(|s| s.0 == Kind::List).expect("the list's pair");
        let group = *spans.iter().find(|s| s.0 == Kind::Clamp).expect("the clamp group's pair");
        assert!(main.1 < group.1 && group.1 <= group.2 && group.2 < main.2, "not nested: list {main:?}, group {group:?}");
        assert_eq!(sink.unread(), 0);
    }

    /// A closed list run twice, one run after the other, is read twice; sent
    /// again before its earlier send was read, it is left out for both, in
    /// so many words (its read-back memory holds the later run's stamps,
    /// which the reading cannot tell from the earlier's). The second send
    /// goes only after the card finished the first, as D3D12 asks of a list
    /// run again (Sol's recheck of the frame timers, Question 4: both went
    /// behind a gate, the first still pending).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_list_sent_again_before_it_was_read_is_left_out() {
        use crate::stamps::{Kind, Sink};
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let sink = Sink::new();
        let list = host.new_list().unwrap();
        list.close();
        host.execute(&[&list]);
        host.execute(&[&list]);
        let lists = |spans: &[(Kind, u64, u64)]| spans.iter().filter(|s| s.0 == Kind::List).count();
        assert_eq!(lists(&sink.spans()), 2, "run twice, read twice");
        for _ in 0..2 {
            let (submitted, sent) = host.submit(&[&list], true).expect("sent ahead");
            let target = *submitted;
            drop((submitted, sent));
            // Run, not read: the next send of the list is legal and finds
            // this one unread.
            assert!(wait_fence(&host.device, &host.fence, None, target));
        }
        list.stamps_landed();
        list.stamps_landed();
        assert_eq!(lists(&sink.spans()), 2, "nothing read of the two sends unread together");
        assert_eq!((sink.flaws().excluded, sink.unread()), (2, 0));
    }

    /// A pair across two query heaps (slots 255 and 256) is read whole; a
    /// list whose closing stamp found no query heap counts its clamp group
    /// and not itself, beside a good list, and the window says so.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_lost_closing_pair_makes_its_window_incomplete() {
        use crate::stamps::{Kind, Sink};
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let sink = Sink::new();
        let good = host.new_list().unwrap();
        good.close();
        let big: Vec<ID3D12Resource> = (0..2).map(|_| host.uav_buffer(4 << 20).unwrap()).collect();
        let bad = host.new_list().unwrap();
        for _ in 1..255 {
            bad.stamp_begin();
        }
        let begun = bad.stamp_begin();
        assert_eq!(begun, Some(255));
        // SAFETY: an open list; live buffers in the common state.
        unsafe { bad.list.CopyBufferRegion(&big[1], 0, &big[0], 0, 4 << 20) };
        bad.stamp_end(Kind::Clamp, begun, true);
        for _ in 257..512 {
            bad.stamp_begin();
        }
        crate::stamps::no_chunks(true);
        bad.close();
        crate::stamps::no_chunks(false);
        host.execute(&[&good, &bad]);
        let spans = sink.spans();
        assert_eq!(spans.iter().filter(|s| s.0 == Kind::List).count(), 1, "the good list alone: {spans:?}");
        let group = *spans.iter().find(|s| s.0 == Kind::Clamp).expect("the group across two heaps");
        assert!(group.2 > group.1, "the copy inside it took time: {group:?}");
        assert_eq!(sink.flaws().unclosed, 1);
        // The send's moment (the CPU's counter) on the card's clock as the
        // line puts it. Taken as the card's clock as it is, the flaw fell
        // after the window on a real card, whose ticks are not the CPU's
        // counter, and the window said complete (a real-card gate of
        // `a96d7e83e`); the software adapter's are.
        let (on_card, now) = crate::stamps::card_clock();
        let end = spans.iter().map(|s| s.2).max().unwrap();
        assert!(now >= end, "the card's clock now ({now}) before the list's end ({end})");
        let (from, to, window, flaws) = sink.window(now, &on_card).expect("a window");
        assert_eq!(window.ticks.iter().sum::<u64>() + window.uncovered, to - from);
        assert_eq!((window.pairs[Kind::List as usize], window.pairs[Kind::Clamp as usize]), (1, 1));
        assert_eq!(flaws.unclosed, 1, "the window is incomplete");
    }

    /// A present copy's pair counts as one, sent with the frame or ahead as
    /// a flight of its own (read when the flight lands, from the list the
    /// presenter keeps).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_present_copy_is_read_in_both_modes() {
        use crate::stamps::{Kind, Sink};
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let sink = Sink::new();
        let copy = host.new_list().unwrap();
        copy.stamps_as(Kind::Present);
        copy.close();
        host.execute(&[&copy]);
        copy.reset();
        copy.close();
        let (submitted, sent) = host.submit(&[&copy], true).expect("sent ahead");
        let flight = Flight {
            number: sent.number,
            fence: *submitted,
            lists: Vec::new(),
            order: Vec::new(),
            card: None,
            back: Vec::new(),
            own: copy.stamps_handle(),
        };
        drop(submitted);
        assert!(wait_fence(&host.device, &host.fence, None, flight.fence));
        flight_stamps_landed(&flight);
        let presents = sink.spans().iter().filter(|s| s.0 == Kind::Present).count();
        assert_eq!(presents, 2, "{:?}", sink.spans());
        assert_eq!(sink.unread(), 0);
    }

    /// A fetch whose fence is below its value, or at `u64::MAX` with the
    /// device gone, yields no stamps and no accounting: nothing is read.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_failed_fetch_yields_no_stamps() {
        use crate::stamps::{Kind, Sink, Stamps};
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let sink = Sink::new();
        // An open list to record into; it never runs.
        let list = host.new_list().unwrap();
        // SAFETY: a plain fence of our own.
        let fence: ID3D12Fence = unsafe { host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
        let mut stamps = Stamps::default();
        let begun = stamps.begin(&list.list);
        stamps.end(Kind::Relay, begun, &list.list, true);
        stamps.resolve(&list.list);
        stamps.sent(&fence, 5);
        stamps.landed();
        assert!(sink.spans().is_empty(), "read below the fence's value: {:?}", sink.spans());
        assert_eq!((sink.flaws().early, sink.unread()), (1, 1));
        // SAFETY: a plain signal of our fence from the CPU.
        unsafe { fence.Signal(u64::MAX) }.unwrap();
        crate::stamps::removed_for_test(true);
        stamps.landed();
        assert!(sink.spans().is_empty(), "read with the device gone: {:?}", sink.spans());
        assert_eq!((sink.flaws().unverified, sink.unread()), (1, 0));
        list.close();
    }

    /// The device goes with two lists' sends unread: each landing gives its
    /// send up unread as unverified, and reads nothing of either.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_removal_with_two_sends_unread_reads_neither() {
        use crate::stamps::{Kind, Sink, Stamps};
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let sink = Sink::new();
        // An open list to record into; it never runs.
        let list = host.new_list().unwrap();
        // SAFETY: a plain fence of our own.
        let fence: ID3D12Fence = unsafe { host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
        let (mut a, mut b) = (Stamps::default(), Stamps::default());
        for (stamps, value) in [(&mut a, 1), (&mut b, 2)] {
            let begun = stamps.begin(&list.list);
            stamps.end(Kind::Relay, begun, &list.list, true);
            stamps.resolve(&list.list);
            stamps.sent(&fence, value);
        }
        assert_eq!(sink.unread(), 2);
        // SAFETY: a plain signal of our fence from the CPU: a removed
        // device's fence reads `u64::MAX`.
        unsafe { fence.Signal(u64::MAX) }.unwrap();
        crate::stamps::removed_for_test(true);
        a.landed();
        b.landed();
        assert!(sink.spans().is_empty(), "read with the device gone: {:?}", sink.spans());
        assert_eq!((sink.flaws().unverified, sink.unread()), (2, 0));
        list.close();
    }

    /// With the switch off (forced off on this thread, whatever the run's
    /// environment) a list made, recorded, closed, run and let go takes no
    /// query heap, makes no stamp state and never locks the pool of spare
    /// chunks; stamp state that stamped nothing goes without allocating or
    /// locking (Sol's recheck of the frame timers, Question 3: each list
    /// made an `Arc<Mutex<Stamps>>`, and its drop locked the pool).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn with_the_switch_off_nothing_is_stamped() {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let _off = crate::stamps::Off::new();
        assert!(!crate::stamps::on());
        let (taken, locks) = (crate::stamps::taken(), crate::stamps::spare_locks());
        let list = host.new_list().unwrap();
        assert!(list.flight_stamps.is_none(), "stamp state made with the switch off");
        assert_eq!(list.stamp_begin(), None, "a stamp with the switch off");
        list.close();
        host.execute(&[&list]);
        drop(list);
        assert_eq!(crate::stamps::taken(), taken, "a query heap taken with the switch off");
        assert_eq!(crate::stamps::spare_locks(), locks, "the pool of spare chunks locked with the switch off");
        let unused = crate::stamps::Stamps::default();
        assert_eq!(crate::copystat::allocations_in(|| drop(unused)), 0);
        assert_eq!(crate::stamps::spare_locks(), locks, "the pool locked letting go of stamp state that stamped nothing");
    }

    /// With the switch off a title call recorded into a list looks at
    /// nothing but the list: no lock and no look at the switch's level
    /// (Sol's third read of the frame timers: the level was looked at after
    /// every title call). A list made at level 2 does look (the control).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn with_the_switch_off_a_title_call_looks_at_the_list_alone() {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        {
            let _off = crate::stamps::Off::new();
            let list = host.new_list().unwrap();
            let asked = crate::stamps::level_asks();
            for _ in 0..100 {
                list.stamp_program(1);
                assert!(!list.stamps_calls());
            }
            assert_eq!(crate::stamps::level_asks(), asked, "the level looked at for title calls with the switch off");
            list.close();
        }
        let _calls = crate::stamps::Sink::calls();
        let list = host.new_list().unwrap();
        let asked = crate::stamps::level_asks();
        assert!(list.stamps_calls());
        assert!(crate::stamps::level_asks() > asked, "the control: at level 2 the level is looked at");
        list.close();
    }

    /// The reporting thread says what was dropped also when no report
    /// follows the drop: after a bounded wait (Sol's second re-check, new
    /// fault 5). It ends once nothing can send to it.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn the_reporter_says_its_drops_without_a_report_after_them() {
        let (send, receive) = std::sync::mpsc::sync_channel::<Report>(1);
        let dropped = std::sync::atomic::AtomicU64::new(3);
        assert_eq!(reporter_step(&receive, &dropped, Duration::from_millis(10)), (true, 3), "no report: the drops");
        assert_eq!(dropped.load(Ordering::Relaxed), 0);
        let (ran, done) = std::sync::mpsc::channel();
        send.send(Box::new(move || ran.send(()).unwrap())).unwrap();
        assert_eq!(reporter_step(&receive, &dropped, Duration::from_millis(10)), (true, 0), "a report");
        assert!(done.try_recv().is_ok(), "the report was made");
        drop(send);
        assert!(!reporter_step(&receive, &dropped, Duration::from_millis(10)).0, "nothing can send");
    }

    /// A reporting thread that could not be started is started at the
    /// next chance, and one that runs is not started again (new fault 5).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_reporter_that_failed_to_start_is_tried_again() {
        let slot = std::sync::RwLock::new(None);
        assert!(!started(&slot, || None), "the start failed");
        assert!(started(&slot, || Some(1)), "tried again");
        assert!(started(&slot, || -> Option<i32> { panic!("started twice") }), "runs");
        assert_eq!(*slot.read().unwrap(), Some(1));
    }

    /// A report handed over while another thread checks whether the
    /// reporting thread runs is handed over, not dropped: the check only
    /// reads the reporter's slot. It held the slot's one lock, and a report
    /// made meanwhile found no queue and was dropped: the one failure in 11
    /// parallel runs of `capture::tests::records_not_read_are_logged_by_the_reporter`
    /// (whose line another test's report or recording raced), and the same
    /// loss in a run whenever two threads report at once.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_report_handed_over_during_a_start_check_is_not_dropped() {
        assert!(reports_start(), "the reporting thread runs");
        let (inside, checking) = std::sync::mpsc::channel::<()>();
        let (go, release) = std::sync::mpsc::channel::<()>();
        let checker = std::thread::spawn(move || {
            set_hook(HOOK_START_CHECK, move || {
                inside.send(()).unwrap();
                let _ = release.recv_timeout(Duration::from_secs(10));
            });
            reports_start()
        });
        checking.recv().unwrap();
        let (ran, done) = std::sync::mpsc::channel::<()>();
        report_after(vec![Box::new(move || {
            let _ = ran.send(());
        })]);
        go.send(()).unwrap();
        assert!(checker.join().unwrap(), "the checker found the thread running");
        assert!(
            done.recv_timeout(Duration::from_secs(10)).is_ok(),
            "the report was dropped while another thread checked whether the reporter runs"
        );
    }

    /// A full reporting queue, or none, counts the report it cannot take
    /// and returns at once: the completion path neither waits nor logs
    /// (Sol's re-check, new fault 3).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_full_report_queue_counts_and_returns() {
        const WATCHDOG: Duration = Duration::from_secs(30);
        let (send, receive) = std::sync::mpsc::sync_channel::<Report>(1);
        let send = ReportQueue(send);
        let dropped = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        hand_over(Some(&send), Box::new(|| {}), &dropped);
        assert_eq!(dropped.load(Ordering::Relaxed), 0, "the queue had room");
        // The handoff to the full queue, on its own thread under a watchdog: one that waited for room would not
        // come back (nothing drains this queue meanwhile), so the watchdog drains it and fails naming the wait.
        // No assertion compares elapsed time.
        let (returned, keep) = std::sync::mpsc::channel::<()>();
        let handing = {
            let (dropped, inner) = (dropped.clone(), send.clone());
            std::thread::spawn(move || {
                hand_over(Some(&inner), Box::new(|| {}), &dropped);
                let _ = returned.send(());
            })
        };
        if keep.recv_timeout(WATCHDOG).is_err() {
            // Drain one place so a waiting handoff can finish, and fail here and now: the timeout handler does
            // not wait for the handoff it guards (no join on this path).
            let _ = receive.recv_timeout(WATCHDOG);
            panic!("the handoff waited {WATCHDOG:?} for room in a full report queue");
        }
        assert!(handing.join().is_ok());
        hand_over(None, Box::new(|| {}), &dropped);
        assert_eq!(dropped.load(Ordering::Relaxed), 2, "full, then no thread");
        // Nothing took the queue's report: the full handoff did not wait to make room.
        assert!(receive.try_recv().is_ok(), "the queue still holds its report");
    }

    /// With the walls trace on, nothing it says is logged on the thread
    /// that notes it, so a sink that blocks holds up neither the completion
    /// of lists (the values a read-back hands the title, the frame's
    /// landings, the shader-written buffers merged) nor a recording thread
    /// with the census locked (the main pass by scissor rectangle): each
    /// line goes to the reporting thread (Sol's review of the second
    /// landing, hold 8).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn the_walls_trace_logs_nothing_on_the_thread_that_notes_it() {
        let logged = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        WALLS_TEST.set(true);
        tracing::subscriber::with_default(Logged(logged.clone()), || {
            // Frames no other test uses: each producer's line for the first
            // comes with its first note in the second.
            let frame = 4_000_000_000;
            for frame in [frame, frame + 1] {
                walls_read_back(frame, 0x1000, &1f32.to_le_bytes().repeat(5));
                walls_landed_in(frame, READ_BACK, 0x1000, 20, 20, 20, 0x2000, Some(frame));
                crate::shadow::Shadow::walls_merged_in(frame, 0x3000, 0x100, 64, 64);
                crate::draw::walls_census_in(frame, crate::draw::CensusNote::Draw([0, 0, 8, 8]));
            }
        });
        WALLS_TEST.set(false);
        let logged = logged.load(Ordering::SeqCst);
        assert_eq!(logged, 0, "{logged} walls trace lines logged on the thread that noted them");
    }

    /// Where lists are recorded the walls trace never asks for the
    /// reporting thread to be started, which takes the reporter's lock,
    /// waits for a start another thread is making and may make the thread
    /// itself: the census's line of a frame and a read-back recorded hand
    /// over without it (Sol's recheck of the second landing, new fault 3).
    /// The thread is started when the host is made.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn walls_recording_never_starts_the_reporting_thread() {
        let Some(host) = test_host() else {
            return; // no host GPU here
        };
        let list = host.new_list().unwrap();
        let (buffer, _) = host.upload_buffer(256).unwrap();
        let mut title = [0u8; 16];
        REPORT_STARTS.set(0);
        logged_here(|| {
            // Frames no other test uses: frame 1's line comes at frame 2.
            let frame = 4_100_000_000;
            for frame in [frame, frame + 1] {
                crate::draw::walls_census_in(frame, crate::draw::CensusNote::Draw([0, 0, 8, 8]));
            }
            list.defer_read_back(buffer, title.as_mut_ptr(), [16, 1, 16], std::ptr::null());
        });
        let asked = REPORT_STARTS.get();
        assert_eq!(asked, 0, "the reporting thread's start asked for {asked} times while recording");
        list.close();
    }

    /// The wait for a submission ends only when the fence has its value,
    /// also when no event can be set on it (finding 7): here the fence is
    /// signalled from another thread 50 ms on.
    #[test]
    fn a_fence_wait_without_an_event_ends_at_the_value() {
        let Some(host) = test_host() else {
            return; // no host GPU here
        };
        // SAFETY: plain creation.
        let fence: ID3D12Fence = unsafe { host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
        let signaller = fence.clone();
        let started = Instant::now();
        let signalled = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            // SAFETY: a live fence, set from the CPU.
            unsafe { signaller.Signal(1) }.unwrap();
        });
        assert!(wait_fence(&host.device, &fence, None, 1), "the device is live");
        let took = started.elapsed();
        // SAFETY: a plain query.
        assert!(unsafe { fence.GetCompletedValue() } >= 1);
        assert!(took >= Duration::from_millis(40), "{took:?}");
        signalled.join().unwrap();
    }

    /// Work a list leaves for once it has run (diagnostics: mapping a
    /// read-back buffer and logging what it holds) does not hold up the
    /// submission's completion, whose bottom-of-pipe writes tell the title
    /// its work is done (Sol's review, finding 6). The work runs on the
    /// reporting thread and blocks here until the test lets it go; the test
    /// proves the submission returned while that work was still unfinished,
    /// and that the work is not released until after the return — its entry
    /// may come before or after the return, so no assertion depends on
    /// which, and none compares elapsed time. A 30 s watchdog releases the
    /// work and fails naming the return if the submission never comes back;
    /// a stall elsewhere (a slow fence) trips it too, and its message names
    /// the return, not a cause — the lines around it in the log show which.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn work_after_a_list_runs_off_the_completion_path() {
        const WATCHDOG: Duration = Duration::from_secs(30);
        let Some(host) = test_host() else {
            return; // no host GPU here
        };
        let (release, wait) = std::sync::mpsc::channel::<()>();
        let (entered, seen) = std::sync::mpsc::channel::<()>();
        let (done, finished) = std::sync::mpsc::channel::<()>();
        let wait = std::sync::Arc::new(Mutex::new(wait));
        // The submission runs on its own thread, so the watchdog below is this thread: it can fail the test
        // itself, at WATCHDOG, without waiting for the return it guards.
        let (returned, outcome) = std::sync::mpsc::channel::<()>();
        let submitting = {
            let (wait, entered, done) = (wait.clone(), entered.clone(), done.clone());
            std::thread::spawn(move || {
                let list = host.new_list().unwrap();
                list.defer_after(move || {
                    let (wait, entered, done) = (wait.clone(), entered.clone(), done.clone());
                    Some(Box::new(move || {
                        let _ = entered.send(());
                        // Blocked until the test releases it, which only happens after the submission returned.
                        let _ = wait.lock().unwrap_or_else(|e| e.into_inner()).recv();
                        let _ = done.send(());
                    }))
                });
                list.close();
                host.execute(&[&list]);
                let _ = returned.send(());
            })
        };
        let waits_before = REPORT_WAITS.load(Ordering::Relaxed);
        if outcome.recv_timeout(WATCHDOG).is_err() {
            // The work is gated, so a submission that waited for it is parked here: release it, then fail now —
            // the wait, not a later return, is the finding. (A stall elsewhere in the submission trips this
            // too; which one it was is in the lines around it, not in this message.)
            let _ = release.send(());
            panic!("execute did not return in {WATCHDOG:?}: the submission waited, or stalled, before returning");
        }
        // The submission returned; the work reached the reporting thread and is still blocked in it.
        assert_eq!(
            REPORT_WAITS.load(Ordering::Relaxed) - waits_before,
            0,
            "the completion path waited on report progress"
        );
        assert!(seen.recv_timeout(WATCHDOG).is_ok(), "the work ran at all");
        assert!(finished.try_recv().is_err(), "the submission returned with the work still blocked");
        let _ = release.send(());
        assert!(finished.recv_timeout(WATCHDOG).is_ok(), "the work finished once released");
        assert!(submitting.join().is_ok());
    }

    /// The queue's waiting methods count as waits on report progress when made from the handoff, whatever calls
    /// them; the non-waiting handoff, and the same calls outside it, do not.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_wait_on_the_report_queue_from_the_handoff_is_counted() {
        let (send, _receive) = std::sync::mpsc::sync_channel::<Report>(4);
        let queue = ReportQueue(send);
        let before = REPORT_WAITS.load(Ordering::Relaxed);
        {
            let _handoff = Handoff::enter();
            assert!(queue.send(Box::new(|| {})), "room: sent");
            assert!(queue.send_timeout(Box::new(|| {}), Duration::from_millis(1)), "room: sent within the bound");
        }
        assert_eq!(REPORT_WAITS.load(Ordering::Relaxed) - before, 2, "two waits from the handoff, counted");
        let before = REPORT_WAITS.load(Ordering::Relaxed);
        assert!(queue.try_send(Box::new(|| {})).is_ok(), "the handoff's own send does not wait");
        assert!(queue.send_timeout(Box::new(|| {}), Duration::from_millis(1)), "outside the handoff: not counted");
        assert_eq!(REPORT_WAITS.load(Ordering::Relaxed), before, "neither counted");
    }

    /// A table constant buffer of 1 MiB made as a draw makes it (the
    /// draw's own function, with the real list and host heap): the scratch
    /// request and the fill take 65,536 bytes, both counted apart from the
    /// title copies, and the last vector a constant view can address
    /// (element 4,095, bytes 65,520 to 65,535) is read back by a shader
    /// through that slot, with a title write made after recording and
    /// before the fill (the fill reads the source at submission, as
    /// `world`'s `defer_fill` did: unchanged, only the length capped).
    /// Shown failing with the cap removed ("requested at the cap": 1,048,576
    /// against 65,536) and with the fill's share folded into the title
    /// copies.
    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    fn a_table_constant_buffer_at_a_draw_fills_64_kib() {
        use crate::copystat::{C, count};
        use windows::core::s;
        let Some(host) = test_host() else {
            return; // no host GPU here
        };
        let Some(heap) = crate::descriptor::host::HostHeap::new(false, 2) else {
            return; // no host device
        };
        // 1 MiB of title constant data, 256-byte aligned (Direct3D wants
        // constant views 256-byte aligned), the last readable constants
        // distinctive.
        let mut backing = vec![0u8; (1 << 20) + 256];
        let aligned = ((backing.as_mut_ptr() as usize + 255) & !255) as *mut u8;
        // SAFETY: inside the backing vector.
        let src = unsafe { std::slice::from_raw_parts_mut(aligned, 1 << 20) };
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i & 0xFF) as u8;
        }
        let mark = |from: u8| {
            let mut b = [0u8; 16];
            for (k, x) in b.iter_mut().enumerate() {
                *x = from.wrapping_add(k as u8);
            }
            b
        };
        src[65_520..65_536].copy_from_slice(&mark(0xA0));
        let before = (
            count(C::TableFillRequested),
            count(C::TableFillFilled),
            count(C::Copies),
            count(C::CopyBytes),
        );
        let list = host.new_list().unwrap();
        let Some((to, len)) =
            crate::draw::table_constant_scratch(&list, &heap, 0, aligned as u64, 1 << 20)
        else {
            panic!("the table slot was not made");
        };
        assert_eq!(len, 65_536, "requested at the cap");
        assert_eq!(
            count(C::TableFillRequested) - before.0,
            65_536,
            "the request is counted apart"
        );
        // A shader reads the last vector a constant view can address back
        // out of the slot the draw made, so the tail is checked where a
        // shader sees it, not only in scratch.
        const TAIL_READER: &str = r#"
#define RS "DescriptorTable(CBV(b0)), UAV(u0)"
cbuffer Tail : register(b0) { uint4 words[4096]; };
RWByteAddressBuffer seen : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    seen.Store4(0, words[4095]);
}
"#;
        let (hits, out) = host.shared_buffer(16).unwrap();
        // SAFETY: the buffer's mapped bytes.
        unsafe { std::ptr::write_bytes(out, 0, 16) };
        let code = crate::host::compile(TAIL_READER, s!("main"), s!("cs_5_1"))
            .expect("the tail reader compiles");
        // SAFETY: plain object creation from our own code.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).unwrap();
            let pso = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        };
        // A title write before the first fill: the fill reads the source at
        // submission, so this is what it copies.
        let expected = mark(0xB0);
        src[65_520..65_536].copy_from_slice(&expected);
        // SAFETY: an open list; our pipeline; a shader-visible heap and the
        // slot the draw's own function made; the buffer the shader writes.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list.SetComputeRootDescriptorTable(0, heap.gpu_handle(0));
            list.list.SetComputeRootUnorderedAccessView(1, hits.GetGPUVirtualAddress());
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);
        assert_eq!(
            count(C::TableFillFilled) - before.1,
            65_536,
            "the fill is counted apart"
        );
        assert_eq!(count(C::Copies) - before.2, 0, "a table fill is not a title copy");
        assert_eq!(count(C::CopyBytes) - before.3, 0, "its bytes are not a title copy's");
        // SAFETY: the shared buffer's mapped bytes, written by the dispatch.
        let seen = unsafe { std::slice::from_raw_parts(out, 16) };
        assert_eq!(seen, &expected[..], "the last vector, as a shader reads it");
        // SAFETY: the scratch piece, written by the submission's fill.
        let filled = unsafe { std::slice::from_raw_parts(to, len as usize) };
        assert_eq!(&filled[65_520..], &expected[..], "and the bytes the fill copied");
    }

    /// Two draws through the table path read one title buffer whose
    /// constants the title refreshes between them: the first list's fill
    /// runs at its submission, the refresh after it (a deferred write, in
    /// order with the copies), and the second list's fill takes the new
    /// bytes, both draws going to the card in the same submission. The
    /// first draw keeps the snapshot its own fill copied. `world` does the
    /// same (its fill also read the source at submission): unchanged, only
    /// the length capped.
    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    fn a_refresh_between_two_table_fills_leaves_the_first_draw_its_snapshot() {
        use windows::core::s;
        let Some(host) = test_host() else {
            return; // no host GPU here
        };
        let Some(heap) = crate::descriptor::host::HostHeap::new(false, 2) else {
            return; // no host device
        };
        let mut backing = vec![0u8; (1 << 20) + 256];
        let aligned = ((backing.as_mut_ptr() as usize + 255) & !255) as *mut u8;
        // SAFETY: inside the backing vector.
        let src = unsafe { std::slice::from_raw_parts_mut(aligned, 1 << 20) };
        src.fill(0);
        let mark = |from: u8| {
            let mut b = [0u8; 16];
            for (k, x) in b.iter_mut().enumerate() {
                *x = from.wrapping_add(k as u8);
            }
            b
        };
        let first = mark(0xA0);
        let second = mark(0xB0);
        src[65_520..65_536].copy_from_slice(&first);
        let (hits, out) = host.shared_buffer(32).unwrap();
        // SAFETY: the buffer's mapped bytes.
        unsafe { std::ptr::write_bytes(out, 0, 32) };
        const TAIL_READER: &str = r#"
#define RS "DescriptorTable(CBV(b0)), UAV(u0)"
cbuffer Tail : register(b0) { uint4 words[4096]; };
RWByteAddressBuffer seen : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    seen.Store4(0, words[4095]);
}
"#;
        let code = crate::host::compile(TAIL_READER, s!("main"), s!("cs_5_1"))
            .expect("the tail reader compiles");
        // SAFETY: plain object creation from our own code.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).unwrap();
            let pso = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        };
        // The first draw's list: its own fill of the tail, its shader reading
        // it into the front of the output.
        let a = host.new_list().unwrap();
        let Some((to_a, len_a)) =
            crate::draw::table_constant_scratch(&a, &heap, 0, aligned as u64, 1 << 20)
        else {
            panic!("the first table slot was not made");
        };
        assert_eq!(len_a, 65_536);
        // SAFETY: an open list; our pipeline; a shader-visible heap and the
        // slot the draw's own function made; the buffer the shader writes.
        unsafe {
            a.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
            a.list.SetComputeRootSignature(&root);
            a.list.SetPipelineState(&pso);
            a.list.SetComputeRootDescriptorTable(0, heap.gpu_handle(0));
            a.list.SetComputeRootUnorderedAccessView(1, hits.GetGPUVirtualAddress());
            a.list.Dispatch(1, 1, 1);
        }
        // The title refreshes the constants at this list's submission, in
        // order with its copies: after the first fill, before the second
        // list's (the lists are prepared in order).
        let at = aligned as usize + 65_520;
        a.defer(move || {
            // SAFETY: title memory the test owns.
            unsafe { std::ptr::copy_nonoverlapping(second.as_ptr(), at as *mut u8, 16) };
        });
        // The second draw's list, with its own slot and its output behind
        // the first's.
        let b = host.new_list().unwrap();
        let Some((to_b, len_b)) =
            crate::draw::table_constant_scratch(&b, &heap, 1, aligned as u64, 1 << 20)
        else {
            panic!("the second table slot was not made");
        };
        assert_eq!(len_b, 65_536);
        // SAFETY: as above, for the second list's slot.
        unsafe {
            b.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
            b.list.SetComputeRootSignature(&root);
            b.list.SetPipelineState(&pso);
            b.list.SetComputeRootDescriptorTable(0, heap.gpu_handle(1));
            b.list.SetComputeRootUnorderedAccessView(1, hits.GetGPUVirtualAddress() + 16);
            b.list.Dispatch(1, 1, 1);
        }
        b.close();
        a.close();
        host.execute(&[&a, &b]);
        // SAFETY: the shared buffer's mapped bytes, written by the dispatches.
        let seen = unsafe { std::slice::from_raw_parts(out, 32) };
        assert_eq!(&seen[..16], &first[..], "the first draw keeps its fill's snapshot");
        assert_eq!(&seen[16..], &second[..], "the second draw's fill took the refresh");
        // SAFETY: each list's scratch piece, filled at its submission.
        let filled_a = unsafe { std::slice::from_raw_parts(to_a, len_a as usize) };
        let filled_b = unsafe { std::slice::from_raw_parts(to_b, len_b as usize) };
        assert_eq!(&filled_a[65_520..], &first[..], "the first fill's bytes");
        assert_eq!(&filled_b[65_520..], &second[..], "the second fill's bytes");
    }

    /// The caption is written only to the window presented to: a handle
    /// that is no window now, or a window of another thread or process
    /// (the handle given again), is left alone (Sol's re-check, finding 10).
    #[test]
    fn the_caption_goes_only_to_the_window_presented_to() {
        let ours = std::process::id();
        assert!(same_window((ours, 7), (ours, 7), ours));
        assert!(!same_window((ours, 7), (0, 0), ours), "no window now");
        assert!(!same_window((ours, 7), (ours, 8), ours), "another thread's window");
        assert!(!same_window((ours, 7), (ours + 1, 7), ours), "another process's window");
        assert!(!same_window((ours + 1, 7), (ours + 1, 7), ours), "never ours");
        assert!(!same_window((0, 0), (0, 0), ours), "no window when presented to");
        assert_eq!(window_owner(HWND(std::ptr::null_mut())), (0, 0), "a null handle is no window");
    }

    /// `VICTORY_CAPTION` is on unless it says `off` (finding 10).
    #[test]
    fn the_caption_switch_is_on_unless_off() {
        assert!(caption_switch(None), "unset");
        for on in ["", "on", "1", "of", "offline"] {
            assert!(caption_switch(Some(on)), "{on:?}");
        }
        for off in ["off", "OFF", " Off ", "off\n"] {
            assert!(!caption_switch(Some(off)), "{off:?}");
        }
    }

    /// The caption names the host's waits as what they are, in ms a frame,
    /// not as the card's load or a share of the frame, and stops moving
    /// once presents stop (finding 10; the frame timers' review).
    #[test]
    fn the_caption_says_host_waits_and_stops_with_the_presents() {
        let figures = Figures { fps: 30.0, ms: 33.3, host_wait_ms: 13.9, draws: 1526.0 };
        let build = "ZSlayerHQ 45722cd diag";
        assert_eq!(
            figures.text(build, Some((12.4, 3500.2)), Some(87.4)),
            "ZSlayerHQ 45722cd diag | 30 fps 33.3 ms/frame | host waits 14 ms/frame gpu 87% cpu 12% ram 3500 MB | 1526 draws/frame"
        );
        assert_eq!(
            figures.text(build, None, Some(100.0)),
            "ZSlayerHQ 45722cd diag | 30 fps 33.3 ms/frame | host waits 14 ms/frame gpu 100% cpu - ram - | 1526 draws/frame"
        );
        assert!(caption_due(Duration::from_millis(4900)));
        assert!(!caption_due(Duration::from_secs(5)));
    }

    /// The caption names the build by the commit's short hash and the
    /// profile (Cargo's, `diag` for release code with the diagnostics
    /// feature), never by the branch, and probes for no changes; a name
    /// shows only where `VICTORY_BUILD_NAME` asks for one, a public-display
    /// opt-in (Sol's second review). The build script's own figures have
    /// that shape.
    #[test]
    fn the_caption_names_the_build_by_hash_and_profile() {
        use super::{build_identity, build_name, profile_word};

        assert_eq!(build_identity("45722cd", "diag", None), "ZSlayerHQ 45722cd diag");
        assert_eq!(build_identity("45722cd", "release", Some("neon")), "ZSlayerHQ neon 45722cd release");
        assert_eq!(
            build_identity("45722cd", "diag", Some(" neon-2.1\n")),
            "ZSlayerHQ neon-2.1 45722cd diag",
            "the name is trimmed"
        );
        for blank in ["", "  ", "\t\n"] {
            assert_eq!(
                build_identity("45722cd", "diag", Some(blank)),
                "ZSlayerHQ 45722cd diag",
                "a blank name {blank:?} is none"
            );
        }
        // The profile: Cargo's `PROFILE` (`debug` or `release`), `diag` for
        // release code with the diagnostics feature, never another word.
        assert_eq!(profile_word("release", true), "diag");
        assert_eq!(profile_word("release", false), "release");
        assert_eq!(profile_word("debug", true), "debug");
        assert_eq!(profile_word("debug", false), "debug");
        assert_eq!(profile_word("someones-profile", true), "build");
        assert_eq!(build_identity("45722cd", "custom", None), "ZSlayerHQ 45722cd build");
        // The commit: seven hex digits, else `unknown`.
        assert_eq!(build_identity("C:/x", "diag", None), "ZSlayerHQ unknown diag");
        assert_eq!(build_identity("45722cdef", "diag", None), "ZSlayerHQ unknown diag");
        assert_eq!(build_identity("45722cd+dirty", "diag", None), "ZSlayerHQ unknown diag");
        // The build script passes no branch and no change probe's answer.
        assert_eq!(option_env!("VICTORY_GPU_BUILD_BRANCH"), None, "the build script reads no branch");
        assert_eq!(option_env!("VICTORY_GPU_BUILD_DIRTY"), None, "the build script probes for no changes");
        // This build's own: the build script's commit and profile.
        let own: Vec<&str> = build_name().split(' ').collect();
        let named = std::env::var("VICTORY_BUILD_NAME").is_ok_and(|name| !name.trim().is_empty());
        assert_eq!(own.len(), if named { 4 } else { 3 }, "{own:?}");
        assert_eq!(own[0], "ZSlayerHQ", "{own:?}");
        let (commit, profile) = (own[own.len() - 2], own[own.len() - 1]);
        assert!(
            commit == "unknown" || (commit.len() == 7 && commit.bytes().all(|b| b.is_ascii_hexdigit())),
            "the commit's short hash: {own:?}"
        );
        assert!(["debug", "release"].contains(&env!("VICTORY_GPU_BUILD_PROFILE")), "Cargo's PROFILE");
        assert_eq!(profile, profile_word(env!("VICTORY_GPU_BUILD_PROFILE"), cfg!(feature = "diagnostics")), "{own:?}");
    }

    /// The caption can appear in a shared recording: a `VICTORY_BUILD_NAME` that is no short
    /// public label (a path, an address, a login) shows as `build` (Sol's
    /// review of the first landing, finding 1).
    #[test]
    fn the_caption_shows_only_public_labels() {
        use super::build_identity;

        let private = [
            r"C:\Users\someone\victory",
            "C:/Users/someone/victory",
            "/home/someone/victory",
            "home/someone/victory",
            "Users/someone",
            "https://example.org/someone/victory.git",
            "git@example.org:someone/victory",
            "example.org/someone/victory",
            "www.example.org",
            "someone@laptop/fix",
            "someone.else/fix",
            "job/../secret",
            "job//title",
            "job/",
            "-rf",
            ".hidden",
            "a label with spaces",
            "a\u{7}b",
            "na\u{ef}ve",
            "a-very-long-label-that-is-more-than-forty-chars",
        ];
        for text in private.iter().map(|t| t.to_string()).chain(["x".repeat(41)]) {
            assert_eq!(
                build_identity("45722cd", "diag", Some(&text)),
                "ZSlayerHQ build 45722cd diag",
                "{text:?} as the name"
            );
        }
        let forty = "x".repeat(40);
        for public in ["world", "main", "job/title-bar", "neon", "fork-2.1", "v0.62.2", "A_b.9", &forty] {
            assert_eq!(
                build_identity("45722cd", "diag", Some(public)),
                format!("ZSlayerHQ {public} 45722cd diag"),
                "{public:?}"
            );
        }
    }

    /// The GPU's share shows as `gpu -` when it cannot be read, never as a
    /// number.
    #[test]
    fn the_caption_says_gpu_dash_when_the_gpu_cannot_be_read() {
        let figures = Figures { fps: 30.0, ms: 33.3, host_wait_ms: 13.9, draws: 1526.0 };
        assert_eq!(
            figures.text("ZSlayerHQ 45722cd diag", Some((12.4, 3500.2)), None),
            "ZSlayerHQ 45722cd diag | 30 fps 33.3 ms/frame | host waits 14 ms/frame gpu - cpu 12% ram 3500 MB | 1526 draws/frame"
        );
    }

    /// The GPU's share from synthetic counters (each 3D engine's running
    /// time, in 100 ns): the busiest engine's share of the time between two
    /// samples, at most all of it; none for counters that went back, wrapped
    /// or do not match, or no time between (Sol's review, finding 3).
    #[test]
    fn the_gpu_share_is_the_busiest_engines_share_between_two_samples() {
        use super::gpu_share;

        let second = Duration::from_secs(1);
        assert_eq!(gpu_share(&[0], &[5_000_000], second), Some(50.0));
        assert_eq!(gpu_share(&[7], &[7], second), Some(0.0));
        assert_eq!(gpu_share(&[0], &[2_500_000], Duration::from_millis(500)), Some(50.0));
        assert_eq!(
            gpu_share(&[0, 0], &[6_000_000, 6_000_000], second),
            Some(60.0),
            "two engines busy 60% each are 60%, not 100%"
        );
        assert_eq!(gpu_share(&[0, 100], &[2_000_000, 7_000_100], second), Some(70.0), "the busiest");
        assert_eq!(gpu_share(&[0], &[10_500_000], second), Some(100.0), "a packet longer than the interval: all of it");
        assert_eq!(gpu_share(&[0], &[100_000_001], second), None, "more than ten intervals is no running time");
        assert_eq!(gpu_share(&[10], &[5], second), None, "a counter that went back");
        assert_eq!(gpu_share(&[i64::MIN], &[i64::MAX], second), None, "a difference beyond the counter's range");
        assert_eq!(gpu_share(&[-5], &[5], second), None, "a negative counter");
        assert_eq!(gpu_share(&[0], &[5], Duration::ZERO), None, "no time between the samples");
        assert_eq!(gpu_share(&[0], &[5], Duration::from_nanos(50)), None, "less than one count between the samples");
        assert_eq!(gpu_share(&[0, 0], &[5], second), None, "engines that do not match");
        assert_eq!(gpu_share(&[], &[], second), None, "no engines");
    }

    /// The 3D engines are found from every node's metadata or not at all
    /// (`gpu -`): a node unread leaves the discovery incomplete. The
    /// metadata is read as bytes, its engine type after the node's number,
    /// whatever the bytes of its two `BOOLEAN`s hold (Sol's review, finding 3).
    #[test]
    fn the_gpu_engines_come_from_every_nodes_metadata_or_none() {
        use super::{NODE_METADATA_SIZE, engine_type, engines_from};

        assert_eq!(engines_from(&[Some(1), Some(3), Some(1)]), Some(vec![0, 2]));
        assert_eq!(engines_from(&[Some(1), None, Some(1)]), None, "a node unread");
        assert_eq!(engines_from(&[Some(3), Some(4)]), None, "no 3D engine");
        assert_eq!(engines_from(&[]), None, "no nodes");
        let mut bytes = [0xff_u8; NODE_METADATA_SIZE];
        bytes[4..8].copy_from_slice(&1_i32.to_le_bytes());
        assert_eq!(engine_type(&bytes), Some(1));
        bytes[4..8].copy_from_slice(&3_i32.to_le_bytes());
        assert_eq!(engine_type(&bytes), Some(3));
        assert_eq!(engine_type(&bytes[..NODE_METADATA_SIZE - 1]), None, "a short read");
    }

    /// The GPU's share reads for the device's adapter where it is hardware:
    /// its 3D engines are found and read, engine by engine, and do not go
    /// back. A software one (WARP, whose engines count none of the work it
    /// runs on the CPU) is not read at all, and the caption says `gpu -`.
    #[test]
    fn the_caption_reads_the_gpu_share_of_a_hardware_adapter_only() {
        let Some(host) = test_host() else {
            return; // no host GPU here
        };
        // SAFETY: plain queries of a live device and its adapter.
        let (luid, desc) = unsafe { (host.device.GetAdapterLuid(), host.adapter.as_ref().map(|a| a.GetDesc1())) };
        let software = desc.and_then(Result::ok).map(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0);
        let engines = super::engines_3d(luid);
        match software {
            Some(true) => {
                assert_eq!(engines, None, "a software adapter's engines are not read");
                return;
            }
            Some(false) => {}
            None => return, // DXGI does not describe the adapter
        }
        let nodes = engines.expect("a hardware adapter's 3D engines are found");
        assert!(!nodes.is_empty());
        let first = super::running_times(luid, &nodes).expect("the 3D engines found are read");
        let second = super::running_times(luid, &nodes).expect("and read again");
        assert_eq!((first.len(), second.len()), (nodes.len(), nodes.len()));
        assert!(first.iter().zip(&second).all(|(a, b)| 0 <= *a && a <= b), "{first:?} then {second:?}");
    }

    #[test]
    fn the_caption_ends_with_how_long_the_process_has_run() {
        use super::{Duration, caption};

        let figures = "ZSlayerHQ 45722cd diag | 1 fps 953.5 ms/frame | 1526 draws/frame";
        let up = |secs| caption(figures, None, Duration::from_secs(secs));
        assert_eq!(up(0), format!("{figures} | up 0:00"));
        assert_eq!(up(59), format!("{figures} | up 0:59"));
        assert_eq!(up(754), format!("{figures} | up 12:34"));
        assert_eq!(up(3723), format!("{figures} | up 1:02:03"));
    }

    #[test]
    fn the_caption_names_video_memory_when_the_adapter_reports_it() {
        use super::{Duration, caption, vram_line};

        const GB: u64 = 1 << 30;
        assert_eq!(vram_line(GB, 2 * GB), "VRAM 1.0/2.0 GB");
        assert_eq!(
            vram_line((9.4 * GB as f64) as u64, (30.1 * GB as f64) as u64),
            "VRAM 9.4/30.1 GB"
        );
        let figures = "1 fps 953.5 ms/frame";
        assert_eq!(
            caption(figures, Some("VRAM 9.4/30.1 GB"), Duration::from_secs(65)),
            "1 fps 953.5 ms/frame | VRAM 9.4/30.1 GB | up 1:05"
        );
        assert_eq!(
            caption(figures, None, Duration::from_secs(65)),
            "1 fps 953.5 ms/frame | up 1:05"
        );
    }

    /// The caption's video memory field finds the adapter the device was
    /// made on, which reports use and budget.
    #[test]
    fn the_caption_finds_the_devices_adapter_for_video_memory() {
        let Some(host) = test_host() else {
            return; // no host GPU here
        };
        assert!(super::vram_adapter(&host.device, true).is_some(), "the device's adapter reports video memory");
    }

    /// `VICTORY_CAPTION=off`: the caption's video memory field asks the
    /// adapter nothing and says nothing, as nothing else is sampled for our
    /// caption then (Sol's re-check, finding 10).
    #[test]
    fn the_caption_off_asks_the_adapter_nothing() {
        let Some(host) = test_host() else {
            return; // no host GPU here
        };
        assert!(super::vram_adapter(&host.device, false).is_none(), "asked with the caption off");
    }

    /// The caption's video memory field looks up the device's adapter (a
    /// DXGI factory and a lookup by LUID, waited for without a bound) at
    /// the caption thread's first refresh, not when the device is made;
    /// with the caption off, never (Sol's review of the second landing,
    /// follow-up 1).
    #[test]
    fn the_caption_looks_up_its_adapter_at_its_first_refresh() {
        let Some(host) = test_host() else {
            return; // no host GPU here
        };
        let vram = super::VramAdapter::new();
        vram.remember(&host.device, true);
        assert!(!vram.looked_up(), "looked up when the device was made");
        assert!(vram.adapter().is_some(), "the device's adapter reports video memory");
        assert!(vram.looked_up());
        let off = super::VramAdapter::new();
        off.remember(&host.device, false);
        assert!(off.adapter().is_none(), "an adapter with the caption off");
        assert!(!off.looked_up(), "looked up with the caption off");
    }

    #[test]
    fn the_running_time_counts_from_the_process_start_and_moves_on() {
        use super::{Duration, process_age, process_uptime};

        // The test process was created before this test ran, and not long ago.
        let age = process_age().expect("the kernel's creation time");
        assert!(age > Duration::ZERO && age < Duration::from_secs(3600), "{age:?}");
        let first = process_uptime();
        std::thread::sleep(Duration::from_millis(20));
        let later = process_uptime();
        assert!(first >= age, "{first:?} is before the process's age {age:?}");
        assert!(later >= first + Duration::from_millis(20), "{first:?} then {later:?}");
    }

    /// The list's indirect-work budget (`crate::indirect`) is in the card's
    /// memory, where the clamp's atomics on it stay on the card (on an RTX
    /// 5080 the benchmark's p1 against p: 4.61 against 7.80 microseconds a
    /// call), and keeps its meaning there: charged across the list's calls,
    /// a record past it runs empty with its other words kept, and it is at
    /// zero again for the next recording of the list and for each run of
    /// the same recording.
    #[test]
    fn the_indirect_budget_is_on_the_card_and_starts_every_run_at_zero() {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        // Indexed draws, 20 bytes a record: at 0 one that asks for the most
        // (2^24 indices, its instances cut to 2^26 vertex instances: half
        // the budget), at 32 one of three indices.
        let (src, data) = host.shared_buffer(1 << 16).unwrap();
        let (out, seen) = host.shared_buffer(1 << 16).unwrap();
        let (big, small, mesh) = (0u64, 32u64, 64u64);
        // SAFETY: inside the mapping; at 64 a mesh dispatch of 40,000 groups.
        unsafe {
            data.cast::<[u32; 5]>().write([u32::MAX, u32::MAX, 0, 0, 0]);
            data.add(small as usize).cast::<[u32; 5]>().write([3, 1, 7, 8, 9]);
            data.add(mesh as usize).cast::<[u32; 3]>().write([40_000, 1, 1]);
        }
        // SAFETY (both): inside the mapping, read once the card is done.
        let read = |at: usize| unsafe { seen.add(at).cast::<[u32; 5]>().read_volatile() };
        let clear = |at: usize| unsafe { seen.add(at).cast::<[u32; 5]>().write_volatile([0xEE; 5]) };
        // One record clamped through the runtime's path; the clamped record
        // copied to `out` at `at`, if given.
        let clamp = |list: &HostList, from: u64, at: Option<u64>| {
            let spec = if from == mesh { (10, 12, 1) } else { (1, 20, 1) };
            let scratch = crate::indirect::bound(list, (&src, from), None, spec).expect("the clamp");
            if let Some(at) = at {
                let (argument, source) = (D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
                // SAFETY: an open list; the scratch buffer as `bound` left it.
                unsafe {
                    list.list.ResourceBarrier(&[transition(&scratch, argument, source)]);
                    list.list.CopyBufferRegion(&out, at, &scratch, 0, 20);
                    list.list.ResourceBarrier(&[transition(&scratch, source, argument)]);
                }
            }
            crate::indirect::done(list, &scratch);
        };
        let list = host.new_list().unwrap();
        clamp(&list, big, None);
        clamp(&list, big, None);
        clamp(&list, small, Some(0));
        clamp(&list, mesh, None);
        list.close();
        host.execute(&[&list]);
        assert_eq!(read(0), [0, 0, 7, 8, 9], "two records of half the budget spend it: the third runs empty, its other words kept");
        let (budget, notes) = list.budget_buffer().expect("the list's budget");
        // What the frame line takes at the list's next submission: the
        // budget spent, and the mesh record's ask (noted though it ran empty).
        assert_eq!(notes, [1 << 27, 40_000, 40_000], "the run's words, copied to the PC's memory where the list ends");
        let mut heap = D3D12_HEAP_PROPERTIES::default();
        // SAFETY: a plain query of a live buffer.
        unsafe { budget.GetHeapProperties(Some(&mut heap), None) }.unwrap();
        assert_eq!(
            heap.Type,
            D3D12_HEAP_TYPE_DEFAULT,
            "the budget is in memory the CPU sees (pages {:?}, pool {:?}): the clamp's atomics cross the bus",
            heap.CPUPageProperty,
            heap.MemoryPoolPreference
        );
        // The next recording starts at zero: its first record is clamped as
        // asked, and the two after it spend the budget again.
        list.reset();
        clamp(&list, small, Some(32));
        clamp(&list, big, None);
        clamp(&list, big, None);
        list.close();
        host.execute(&[&list]);
        assert_eq!(read(32), [3, 1, 7, 8, 9], "the next recording of the list starts with what the last one spent");
        let notes = list.budget_buffer().expect("the list's budget").1;
        assert_eq!(notes, [1 << 27, 0, 0], "the notes start at zero with the budget: the last recording's mesh ask is not counted again");
        // So does the same recording run again.
        clear(32);
        host.execute(&[&list]);
        assert_eq!(read(32), [3, 1, 7, 8, 9], "a second run of the recording starts with what its first run spent");
        list.reset();
    }

    /// The census of runs of clamped calls (`crate::runs`) on lists recorded
    /// as the title's methods record them: the calls clamped through
    /// `crate::indirect::bound`, and between them what `State::bind`,
    /// `State::dispatch` and the copy methods of `crate::list` tell the
    /// census.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn the_census_of_runs_counts_known_lists() {
        use crate::runs::{Runs, ignoring_stores, named};
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let (src, data) = host.shared_buffer(1 << 16).unwrap();
        let (other, other_data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: inside the mappings: a record of one group in each.
        unsafe {
            data.cast::<[u32; 3]>().write([1, 1, 1]);
            other_data.cast::<[u32; 3]>().write([1, 1, 1]);
        }
        // A mesh dispatch's record clamped, or a compute dispatch's.
        let call = |list: &HostList, from: &ID3D12Resource, dispatch: bool| {
            let spec = if dispatch { (2, 12, 1) } else { (10, 12, 1) };
            let scratch = crate::indirect::bound(list, (from, 0), None, spec).expect("the clamp");
            crate::indirect::done(list, &scratch);
        };
        // A title dispatch (bound, its program storing, then recorded),
        // three calls, a copy into a buffer, two calls, the end.
        let list = host.new_list().unwrap();
        list.runs(|runs| runs.bound(true));
        list.runs(Runs::dispatch);
        for _ in 0..3 {
            call(&list, &src, false);
        }
        list.runs(Runs::copy);
        for _ in 0..2 {
            call(&list, &src, false);
        }
        list.close();
        let tally = list.runs_tally();
        assert_eq!(
            named(&tally),
            [
                ("calls", 5),
                ("runs", 2),
                ("runs of 1", 0),
                ("runs of 2-3", 2),
                ("ended by a dispatch", 0),
                ("ended by a copy", 1),
                ("ended by a store", 0),
                ("ended by the list's end", 1),
                ("other buffer", 0)
            ]
        );
        assert_eq!(ignoring_stores(&tally), 2);
        // A call, one reading another buffer, a draw that can store, a
        // call, an indirect dispatch (whose own program stores), a call,
        // the end: runs of 2 (ended by the store), 2 (by the dispatch) and
        // 1 (by the end); with (c) ignored, 4 and 1.
        let list = host.new_list().unwrap();
        call(&list, &src, false);
        call(&list, &other, false);
        list.runs(|runs| runs.bound(true));
        call(&list, &src, false);
        call(&list, &src, true);
        list.runs(|runs| runs.bound(true));
        call(&list, &src, false);
        list.close();
        let tally = list.runs_tally();
        assert_eq!(
            named(&tally),
            [
                ("calls", 5),
                ("runs", 3),
                ("runs of 1", 1),
                ("runs of 2-3", 2),
                ("ended by a dispatch", 1),
                ("ended by a copy", 0),
                ("ended by a store", 1),
                ("ended by the list's end", 1),
                ("other buffer", 1)
            ]
        );
        assert_eq!(ignoring_stores(&tally), 2);
        // Reset, the list counts anew.
        list.reset();
        assert_eq!(list.runs_tally(), [0; crate::runs::COUNTS]);
    }

    /// Runs [`a_list_named_twice_child`] in a process of its own, on
    /// `path`: "sync" (`VICTORY_LATE_WAIT=0`, every submission waited for)
    /// or "late" (`VICTORY_LATE_WAIT=3`, sent ahead); with the debug layer
    /// asked for, which must be on before the first device is made and
    /// which a machine may not have (the child says so, loudly, and checks
    /// everything else).
    #[cfg(feature = "diagnostics")]
    fn named_twice(path: &str) {
        let name = "host::tests::a_list_named_twice_child";
        let exe = std::env::current_exe().expect("the test binary");
        let out = std::process::Command::new(exe)
            .args(["--exact", name, "--ignored", "--test-threads", "1", "--nocapture"])
            .env("VICTORY_D3D12_DEBUG", "1")
            .env("VICTORY_NAMED_TWICE", path)
            .env("VICTORY_LATE_WAIT", if path == "late" { "3" } else { "0" })
            .env_remove("VICTORY_GPU_TIMING")
            .output()
            .expect("the child ran");
        let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        for line in text.lines().filter(|l| l.contains("NOT RUN")) {
            eprintln!("{path}: {line}");
        }
        assert!(out.status.success() && text.contains("1 passed"), "{path}: {text}");
    }

    /// A title submission that names the same list twice, through the
    /// title's queue, waited for: see [`a_list_named_twice_child`].
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_list_named_twice_in_one_submission_runs_twice_with_a_budget_each() {
        named_twice("sync");
    }

    /// As [`a_list_named_twice_in_one_submission_runs_twice_with_a_budget_each`],
    /// with submissions sent ahead (`VICTORY_LATE_WAIT=3`).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_list_named_twice_in_one_late_submission_runs_twice_with_a_budget_each() {
        named_twice("late");
    }

    /// The child of the two above, in its own process. One title list,
    /// named twice in one submission to a title queue (its table's
    /// `ExecuteCommandLists`), needs a different prelude each time (Sol's
    /// second review of the clamp in the card's memory): its texture
    /// enters the first execution in copy destination and the second in
    /// copy source, where the first left it; the list clears it as a render
    /// target and copies a texel of it out. It also clamps a draw record
    /// that asks for half the budget (2^24 indices, cut to 2^26 vertex
    /// instances) and a dispatch of three groups that it runs through
    /// `ExecuteIndirect` on a program that counts its groups: two
    /// executions sharing one allowance would leave the second's dispatch
    /// empty (3 groups counted, not 6). Each execution goes as a submission
    /// of its own, the first waited for before the second is prepared.
    /// Checked without the debug layer: the groups, the texel, the
    /// texture's state, the submissions; with it, also that it has nothing
    /// to say.
    #[cfg(feature = "diagnostics")]
    #[test]
    #[ignore = "a child process of the tests above"]
    fn a_list_named_twice_child() {
        use crate::object::{This, slot};
        use windows::core::s;
        const RENDER_TARGET: u32 = 0x4;
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let path = std::env::var("VICTORY_NAMED_TWICE").expect("run by its parents");
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let layer = host.messages.is_some();
        if !layer {
            eprintln!("NOT RUN (no debug layer): the D3D12 debug layer's check of a list named twice; the rest is checked");
        }
        host.log_messages();
        let before = std::mem::take(&mut *super::messages::DEBUG_ERRORS.lock().unwrap());
        if !before.is_empty() {
            eprintln!("before the test, the debug layer said: {before:?}");
        }
        const COUNTER: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer hits : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint prev;
    hits.InterlockedAdd(0, 1, prev);
}
"#;
        let code = compile(COUNTER, s!("main"), s!("cs_5_1")).expect("the group counter compiles");
        // SAFETY: plain object creation from our own compiled code and
        // descriptions.
        let (root, pso, signature, texture) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
                    ..Default::default()
                })
                .unwrap();
            let arg = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, ..Default::default() };
            let desc = D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 12, NumArgumentDescs: 1, pArgumentDescs: &arg, NodeMask: 0 };
            let mut signature: Option<ID3D12CommandSignature> = None;
            host.device.CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut signature).unwrap();
            let mut texture: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: 64,
                        Height: 64,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATES(COPY_DEST as i32),
                    None,
                    &mut texture,
                )
                .unwrap();
            (root, pso, signature.unwrap(), texture.unwrap())
        };
        let texture = std::sync::Arc::new(crate::resource::HostTexture::for_test(texture, COPY_DEST));
        let (src, data) = host.shared_buffer(1 << 16).unwrap();
        let (hits, counted) = host.shared_buffer(256).unwrap();
        let (texel, seen) = host.shared_buffer(64 * 256).unwrap();
        // SAFETY: inside the mappings.
        unsafe {
            data.cast::<[u32; 5]>().write([u32::MAX, u32::MAX, 0, 0, 0]);
            data.add(64).cast::<[u32; 3]>().write([3, 1, 1]);
            counted.cast::<u32>().write_volatile(0);
            std::ptr::write_bytes(seen, 0, 64 * 256);
        }
        let title = crate::list::test_list();
        let list = crate::list::host_list(title).expect("the title list's host list");
        // The texture as a render target, cleared, then copied from.
        list.settle(&texture, RENDER_TARGET);
        let rtv = host.rtv(Some(&texture.texture), None);
        // SAFETY: an open list; live resources; the copy locations name
        // them as they are laid out.
        unsafe {
            list.list.ClearRenderTargetView(rtv, &[0.2, 0.4, 0.6, 1.0], None);
            list.settle(&texture, COPY_SOURCE);
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&texel),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                            Width: 64,
                            Height: 64,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&texture.texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        let big = crate::indirect::bound(list, (&src, 0), None, (1, 20, 1)).expect("the draw's clamp");
        crate::indirect::done(list, &big);
        let small = crate::indirect::bound(list, (&src, 64), None, (2, 12, 1)).expect("the dispatch's clamp");
        // SAFETY: an open list; our pipeline and signature, the clamped
        // record as `bound` left it.
        unsafe {
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list.SetComputeRootUnorderedAccessView(0, hits.GetGPUVirtualAddress());
            list.list.ExecuteIndirect(&signature, 1, &small, 0, None, 0);
        }
        crate::indirect::done(list, &small);
        let close = slot::<extern "system" fn(This) -> windows::core::HRESULT>(title, 0x48);
        assert!(close(title).is_ok(), "the title's list closes");
        // The title's submission: its queue's `ExecuteCommandLists`.
        let queue = crate::queue::test_queue();
        let started = STARTED.load(Ordering::Acquire);
        let counted_twice = NAMED_TWICE.load(Ordering::Relaxed);
        let named = [title, title];
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 2, named.as_ptr());
        let sent = STARTED.load(Ordering::Acquire) - started;
        // Sent ahead, it is in the air: settled as at a present.
        settle_for_present();
        host.log_messages();
        let said = std::mem::take(&mut *super::messages::DEBUG_ERRORS.lock().unwrap());
        // SAFETY: the card is done with the mappings.
        let (groups, first) = unsafe { (counted.cast::<u32>().read_volatile(), seen.cast::<[u8; 4]>().read_volatile()) };
        for said in &said {
            eprintln!("{path}: the debug layer: {said}");
        }
        assert_eq!(sent, 2, "{path}: the two executions, each a submission of its own");
        let twice = NAMED_TWICE.load(Ordering::Relaxed) - counted_twice;
        assert_eq!(twice, 1, "{path}: counted for the frame line");
        assert_eq!(groups, 6, "{path}: each execution's dispatch runs its three groups, charged from a budget of its own");
        let cleared = first.iter().zip([51u8, 102, 153, 255]).all(|(&a, b)| a.abs_diff(b) <= 1);
        assert!(cleared, "{path}: the texel cleared and copied out: {first:?}");
        assert_eq!(texture.state.load(Ordering::Relaxed), COPY_SOURCE, "{path}: the texture as the list leaves it");
        if layer {
            assert!(said.is_empty(), "{path}: the debug layer: {said:#?}");
        }
    }

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        // The software adapter the host's device takes with `VICTORY_D3D12_WARP` (the one
        // rule's, `victory_shader::adapter`): software by DXGI's flag, or the Basic Render
        // Driver's ids.
        assert_eq!(
            victory_shader::adapter::software_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0
                || (desc.VendorId == 0x1414 && desc.DeviceId == 0x8c),
        );
    }

    /// The host's device is made by the one rule, its line logged (and printed here, once):
    /// the line names the adapter the device is on, found by the device's own LUID, and how
    /// it was chosen. Off the card (`VICTORY_D3D12_WARP=1`) that is the software adapter;
    /// on the real-card gate, the card the rule picks.
    #[test]
    fn the_host_says_the_adapter_its_device_is_on() {
        let Some(host) = test_host() else {
            eprintln!("no host device here; skipped");
            return;
        };
        let (chosen, line) = super::CHOSEN.get().expect("the host's device was made by create");
        let own = victory_shader::adapter::device_adapter(&host.device).expect("the device's adapter, by its LUID");
        let chosen = chosen.as_ref().expect("the host's adapter described");
        assert_eq!(chosen.luid, own.luid, "{line}");
        assert_eq!(chosen.description, own.description, "{line}");
        assert!(line.starts_with(&format!("host adapter: {} (vendor ", own.description)), "{line}");
        let how = line.split_once("), chosen by ").expect("how the adapter was chosen").1;
        assert_eq!(*line, victory_shader::adapter::line(&own, how), "the line is the device's own adapter's");
        if on_warp() {
            assert!(own.software, "{line}");
            assert!(how.starts_with("VICTORY_D3D12_WARP"), "{line}");
        }
    }

    /// A `VICTORY_ADAPTER` value no adapter can match: longer than the 128 characters DXGI
    /// gives a description, so that it is refused before any device is made.
    const NO_ADAPTER: &str = "no adapter has this description, which is longer than the 128 characters DXGI gives a \
                              description, so that none matches it and no device is made";

    /// A refused adapter fails each test that asks for the host, and aborts no test process
    /// (the hybrid laptop's gate, 2026-11-25: `victory-gpu` died with "thread caused
    /// non-unwinding panic"). A child of this binary runs with a value no adapter matches: a
    /// test that asks for the host in its body, and one whose title calls (`extern "system"`,
    /// through which no panic can unwind) reach the host first. The child makes no device
    /// (the value is refused before one is made); the software switch is taken from it so
    /// that the refusal is reached, as on a gate on the card.
    #[test]
    fn a_refused_adapter_fails_the_tests_that_ask_for_the_host_and_aborts_none() {
        assert!(NO_ADAPTER.chars().count() > 128);
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "host::tests::the_host_says_the_adapter_its_device_is_on",
                "list::tests::every_inline_command_falls_back_to_the_table",
                "--test-threads=1",
            ])
            .env("VICTORY_ADAPTER", NO_ADAPTER)
            .env_remove("VICTORY_D3D12_WARP")
            .env_remove("VICTORY_REQUIRE_GPU")
            .output()
            .unwrap();
        let (stdout, stderr) = (String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        let all = format!("{stdout}\n{stderr}");
        assert_ne!(out.status.code(), Some(0xc000_0409_u32 as i32), "the child aborted: {all}");
        assert!(stdout.contains("test result: FAILED. 1 passed; 1 failed"), "{all}");
        assert!(stdout.contains("test list::tests::every_inline_command_falls_back_to_the_table ... ok"), "{all}");
        assert!(stdout.contains("test host::tests::the_host_says_the_adapter_its_device_is_on ... FAILED"), "{all}");
        let refusal = format!("VICTORY_ADAPTER={NO_ADAPTER}: no hardware adapter's description contains");
        assert!(stdout.contains(&refusal), "the test fails with the refusal: {all}");
        assert_eq!(out.status.code(), Some(101), "{all}");
    }

    /// `Present`'s result decides whether a frame counts: a success code is a frame shown
    /// (`DXGI_STATUS_OCCLUDED` is one: DXGI took it); a failure is not, and is said with
    /// its code; a device removed, reset or hung, or a failure on a device its removal reason
    /// says is gone, takes the fatal path submissions take.
    #[test]
    fn a_frame_not_presented_does_not_count() {
        use windows::Win32::Foundation::{DXGI_STATUS_OCCLUDED, E_FAIL, S_OK};
        let gone = || true;
        let there = || false;
        assert_eq!(shown(S_OK, gone), Shown::Yes);
        assert_eq!(shown(DXGI_STATUS_OCCLUDED, there), Shown::Yes);
        assert_eq!(shown(DXGI_ERROR_INVALID_CALL, there), Shown::No);
        assert_eq!(shown(E_FAIL, there), Shown::No);
        for removed in [DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_RESET, DXGI_ERROR_DEVICE_HUNG] {
            assert_eq!(shown(removed, there), Shown::Removed, "{removed:?}");
        }
        assert_eq!(shown(E_FAIL, gone), Shown::Removed);
        // The first failure and every hundredth after it are said.
        let said: Vec<u64> = (1..=301).filter(|&n| said_not_presented(n)).collect();
        assert_eq!(said, [1, 101, 201, 301]);
        assert_eq!(
            not_presented_line(7, DXGI_ERROR_INVALID_CALL, 1),
            "frame 7 not presented: IDXGISwapChain::Present returned 0x887a0001; not counted as a frame; \
             1 not presented so far (the first and every hundredth after it are logged)"
        );
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// What the list does with depth textures that have a colour twin, by
    /// twin ([`TwinUse`]).
    twins: Mutex<std::collections::HashMap<usize, TwinUse>>,
    /// The list that runs those transitions just before this one, made
    /// when the list first has such transitions ([`lazy_prelude`]).
    prelude: OnceLock<Option<Prelude>>,
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// The work its indirect executions may still do (`indirect`): words in
    /// the card's memory, zeroed there where the list first uses them
    /// ([`HostList::budget`]).
    budget: OnceLock<Option<Budget>>,
    /// Device-local scratch buffer for clamped indirect arguments (`indirect`),
    /// grown to the largest indirect execution recorded on the list.
    indirect_args: Mutex<Option<(ID3D12Resource, u64)>>,
    /// The clamp of runs of indirect calls (`indirect`, `VICTORY_CLAMP_RUNS`):
    /// the batch open, and the slot buffer, kept across resets.
    batching: Mutex<crate::indirect::Batching>,
    /// The list's enhanced barriers, where the clamps order their reads of
    /// the title's copies with them ([`Host::enhanced_barriers`]).
    enhanced: Option<ID3D12GraphicsCommandList7>,
    /// Diagnostics: the runs of its clamped indirect calls (`crate::runs`).
    #[cfg(feature = "diagnostics")]
    runs: Mutex<crate::runs::Runs>,
    /// Counts the list in [`ALIVE`] while it lives.
    _alive: Counted,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
    /// With `VICTORY_GPU_STAMPS=1`: the pairs of timestamps around the
    /// list, its clamp groups and its preludes, read for each send once the
    /// card has run it ([`crate::stamps`]); shared with a flight that has no
    /// host list of its own (a present copy sent ahead). None with the
    /// switch off: nothing made, nothing locked.
    #[cfg(feature = "diagnostics")]
    flight_stamps: Option<std::sync::Arc<Mutex<crate::stamps::Stamps>>>,
}

/// A list's prelude: an allocator and a closed list.
type Prelude = (ID3D12CommandAllocator, ID3D12GraphicsCommandList);

/// Whether a host list's prelude is made when the list first needs one. A
/// prelude runs the texture transitions and twin copies a list needs before
/// it; lists that name no texture never have one, and every host list was
/// made with one all the same: twice the host objects per list, for the
/// title's 630 lists and for every list made anew.
/// `VICTORY_LAZY_PRELUDE=0`: made with the list, as before.
fn lazy_prelude() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAZY_PRELUDE").is_none_or(|v| v != "0"))
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<
    usize,
    (u32, u32, std::sync::Arc<crate::resource::HostTexture>),
>;

/// What a list does with a depth texture that has a colour twin
/// ([`crate::resource::Twin`]). Lists are recorded on many threads and run
/// in submission order, so a list cannot know which side an earlier list
/// leaves behind: it notes which sides it uses as it finds them, and its
/// prelude copies one side into the other at submission if that side is
/// behind then. From its first use of a side on, the list knows.
struct TwinUse {
    /// The depth texture (the title resource's host copy) and its twin.
    depth: std::sync::Arc<crate::resource::HostTexture>,
    twin: std::sync::Arc<crate::resource::Twin>,
    /// The sides (depth texture, twin) the list uses as it finds them.
    needs: [bool; 2],
    /// Whether each side lacks what the other was written last, where the
    /// list is now; `None`: as the list found it.
    behind: [Option<bool>; 2],
    /// The draw or dispatch being prepared writes the twin.
    writing: bool,
    /// The depth plane's twin holds what the plane lacks although the list
    /// copied the twins since: the plane's pass waits for the list's close.
    late: bool,
}

/// The pass that draws the twin of a depth plane into the plane: one
/// triangle over each mip of each slice, its pixel shader writing the
/// twin's texel as the depth, with the depth test "always" and no stencil
/// (the stencil plane is copied through a buffer and not touched here).
///
/// A 16-bit twin holds whole steps of 1/65535, and the depth written is
/// the step times the float nearest 1/65535 (0x37800080): a draw that
/// computes its depth as a whole number times that float, and tests
/// "equal", then gives the host the same float as this pass, and
/// the host turns the same float into the same 24 bits. The step divided
/// by 65535 is another float for 512 of the 65,536 steps (none under 257),
/// and lands in another 24-bit step for 127 of them (none under 385).
const PLANE_CORNERS: &str = r#"
float4 corner(uint i : SV_VertexID) : SV_Position {
    return float4(float(i & 1) * 4.0 - 1.0, float(i >> 1) * 4.0 - 1.0, 0.0, 1.0);
}
"#;
const PLANE_STEPS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<uint> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return float(twin.Load(int4(int2(p.xy), 0, 0))) * asfloat(0x37800080);
}
"#;
const PLANE_FLOATS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<float> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return twin.Load(int4(int2(p.xy), 0, 0));
}
"#;

/// Views of twins the pass reads, in a heap of ours used as a ring.
/// shortcut: a view must last until its list has run; with one or two
/// passes a frame the ring is far from coming round.
const PLANE_VIEWS: u32 = 256;

/// What the pass draws with: the heap of its views of twins, and per kind
/// of twin (16-bit steps or floats) and depth format the root signature
/// and pipeline, made when first needed (`None`: the host refused).
struct PlanePass {
    views: ID3D12DescriptorHeap,
    cpu: usize,
    gpu: u64,
    step: u32,
    next: AtomicU32,
    made: Mutex<Vec<(bool, i32, Option<PlanePipeline>)>>,
}

/// The pass's root signature and pipeline for one kind of twin and one
/// depth format.
type PlanePipeline = (ID3D12RootSignature, ID3D12PipelineState);

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for PlanePass {}
// SAFETY: as above.
unsafe impl Sync for PlanePass {}

fn plane_pass() -> Option<&'static PlanePass> {
    static PASS: OnceLock<Option<PlanePass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let host = host()?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // SAFETY: plain heap creation and queries.
        unsafe {
            let views: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: PLANE_VIEWS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .ok()?;
            Some(PlanePass {
                cpu: views.GetCPUDescriptorHandleForHeapStart().ptr,
                gpu: views.GetGPUDescriptorHandleForHeapStart().ptr,
                step: host.device.GetDescriptorHandleIncrementSize(kind),
                views,
                next: AtomicU32::new(0),
                made: Mutex::default(),
            })
        }
    })
    .as_ref()
}

/// Compiles function `entry` of `source` for `target` with the system's
/// compiler, as the indirect-argument clamp is (`crate::indirect`).
pub(crate) fn compile(
    source: &str,
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
) -> windows::core::Result<Vec<u8>> {
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            windows::core::s!("victory"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("a shader of ours: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
    };
    Ok(bytes.to_vec())
}

impl PlanePass {
    /// The root signature and pipeline for a twin of 16-bit steps (`steps`)
    /// or of floats and the depth format `format`.
    fn pipeline(&self, steps: bool, format: DXGI_FORMAT) -> Option<PlanePipeline> {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, pipeline)) = made.iter().find(|m| (m.0, m.1) == (steps, format.0)) {
            return pipeline.clone();
        }
        let pipeline = match Self::make(steps, format) {
            Ok(pipeline) => Some(pipeline),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    steps,
                    format = format.0,
                    "no pipeline to draw a depth plane from its twin: such planes stay as they are"
                );
                None
            }
        };
        made.push((steps, format.0, pipeline.clone()));
        pipeline
    }

    fn make(steps: bool, format: DXGI_FORMAT) -> windows::core::Result<PlanePipeline> {
        use windows::core::s;
        let device = &host().ok_or_else(windows::core::Error::empty)?.device;
        let vs = compile(PLANE_CORNERS, s!("corner"), s!("vs_5_1"))?;
        let source = if steps { PLANE_STEPS } else { PLANE_FLOATS };
        let ps = compile(source, s!("plane"), s!("ps_5_1"))?;
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        // SAFETY: the root signature is inside the pixel shader; the
        // description points at live data and never releases its pointer.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &ps[..])?;
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs[..]),
                PS: code(&ps[..]),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Every texel is written; stencil stays off.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                    DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                DSVFormat: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let pipeline: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc)?;
            Ok((root, pipeline))
        }
    }
}

/// Records into `list` the pass that draws `plane`, the twin of the depth
/// plane of `depth`, into that plane: `depth` in the depth-write state,
/// `plane` readable by pixel shaders. The pass sets the list's descriptor
/// heap, graphics root signature, pipeline, depth target, viewport,
/// scissor and topology and puts none of them back: it is for a list that
/// records nothing of the title's after it.
fn record_plane_pass(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    plane: &ID3D12Resource,
) {
    let (Some(host), Some(pass)) = (host(), plane_pass()) else {
        return;
    };
    // SAFETY: live resources.
    let (of, twin) = unsafe { (depth.GetDesc(), plane.GetDesc()) };
    // An R16 typeless twin (53) holds steps, read as R16_UINT (57); an R32
    // typeless one floats, read as R32_FLOAT (41). The depth view names
    // the depth-stencil format of the texture's family (44: 45, 19: 20).
    let steps = twin.Format.0 == 53;
    let format = DXGI_FORMAT(match of.Format.0 {
        44 => 45,
        19 => 20,
        typed => typed,
    });
    let Some((root, pipeline)) = pass.pipeline(steps, format) else {
        return;
    };
    for slice in 0..u32::from(of.DepthOrArraySize) {
        for mip in 0..u32::from(of.MipLevels) {
            let slot = pass.next.fetch_add(1, Ordering::Relaxed) % PLANE_VIEWS;
            let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT(if steps { 57 } else { 41 }),
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                        MostDetailedMip: mip,
                        MipLevels: 1,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                        ..Default::default()
                    },
                },
            };
            let target = D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                        MipSlice: mip,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                    },
                },
                ..Default::default()
            };
            let target = host.dsv(depth, Some(&target));
            let (width, height) = ((of.Width >> mip).max(1) as u32, (of.Height >> mip).max(1));
            let cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: pass.cpu + (slot * pass.step) as usize,
            };
            let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: pass.gpu + u64::from(slot * pass.step),
            };
            // SAFETY: an open list; a slot of our heap; live resources and
            // objects (the pass's own live as long as the process).
            unsafe {
                host.device
                    .CreateShaderResourceView(plane, Some(&view), cpu);
                list.SetDescriptorHeaps(&[Some(pass.views.clone())]);
                list.SetGraphicsRootSignature(&root);
                list.SetPipelineState(&pipeline);
                list.SetGraphicsRootDescriptorTable(0, gpu);
                list.OMSetRenderTargets(0, None, false, Some(&target));
                list.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: width as f32,
                    Height: height as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                list.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                }]);
                list.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY(4),
                );
                list.DrawInstanced(3, 1, 0, 0);
            }
            crate::resource::PLANES[0].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Records into `list` the copy of a depth texture into its colour twin
/// (`colour`) or of the twin into the depth texture, both already in their
/// copy states: whole textures of one size in one format family, which the
/// host copies as they are; or, for the twin of a stencil plane, that plane
/// alone, through the twin's buffer.
fn record_twin_copy(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    twin: &crate::resource::Twin,
    colour: bool,
) {
    let (from, to) = if colour {
        (depth, &twin.colour.texture)
    } else {
        (&twin.colour.texture, depth)
    };
    crate::resource::TWINS[2 + usize::from(colour)].fetch_add(1, Ordering::Relaxed);
    // shortcut: the depth plane's twin is not given the plane (that is a
    // pass the other way: the plane read by a shader, the twin as its
    // target). The twin then lacks what draws and clears wrote to the
    // plane since shaders last wrote the twin. In the title
    // nothing writes that plane but the twin's pass (its 192 draws against
    // it only test, run 29), so the twin lacks nothing. Counted.
    if colour && twin.plane.is_some() {
        crate::resource::PLANES[1].fetch_add(1, Ordering::Relaxed);
    }
    let Some(through) = &twin.through else {
        // SAFETY: an open list and two live textures.
        unsafe { list.CopyResource(to, from) };
        return;
    };
    // A stencil plane and its twin: each mip and slice of the plane goes
    // into the buffer, laid out as the host lays the plane out for copies,
    // and from the buffer into the other texture, whole subresources both
    // times. In the depth texture the stencil plane's subresources follow
    // the depth plane's; the twin has the one plane.
    let Some(host) = host() else { return };
    // SAFETY: a live texture.
    let of = unsafe { depth.GetDesc() };
    let planes = u32::from(of.MipLevels) * u32::from(of.DepthOrArraySize);
    let mut rows = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); planes as usize];
    // SAFETY: a plain query writing `planes` footprints.
    unsafe {
        host.device.GetCopyableFootprints(
            &of,
            planes,
            planes,
            0,
            Some(rows.as_mut_ptr()),
            None,
            None,
            None,
        );
    }
    let (first_from, first_to) = if colour { (planes, 0) } else { (0, planes) };
    let subresource = |texture: &ID3D12Resource, index: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index,
        },
    };
    let placed = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(through) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, source, dest) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: an open list and live resources; the buffer was made as big
    // as these footprints, and rests in the common state between copies.
    unsafe {
        list.ResourceBarrier(&[transition(through, common, dest)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_from + i as u32;
            list.CopyTextureRegion(&placed(row), 0, 0, 0, &subresource(from, index), None);
        }
        list.ResourceBarrier(&[transition(through, dest, source)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_to + i as u32;
            list.CopyTextureRegion(&subresource(to, index), 0, 0, 0, &placed(row), None);
        }
        list.ResourceBarrier(&[transition(through, source, common)]);
    }
}

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "diagnostics") && crate::switches::switch_on("VICTORY_D3D12_DRED")
    })
}

/// Indirect mesh dispatches since the last frame line: the largest group
/// count a record asked for, and the groups asked in all (from the copies
/// of the lists' budgets in the PC's memory). A diagnostic, and not exact:
/// a host list's notes are taken at its next submission that clamps, so
/// they come a submission late, possibly in a later frame or for another
/// title list (the pool hands host lists on); and a list never used again
/// takes its last notes with it.
static MESH_GROUPS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Submissions that named a host list twice since the last frame line
/// ([`Host::execute`]).
static NAMED_TWICE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether `runs` names a host list twice.
fn names_twice(runs: &[&HostList]) -> bool {
    runs.iter().enumerate().any(|(i, r)| runs[..i].iter().any(|q| std::ptr::eq(*q, *r)))
}

/// Whether `order` (places in a submission's taken lists) names one twice.
fn names_twice_in(order: &[usize]) -> bool {
    order.iter().enumerate().any(|(i, n)| order[..i].contains(n))
}

/// A submission of `lists` host lists named one twice: counted for the
/// frame line, and said the first time.
fn named_twice(lists: usize) {
    if NAMED_TWICE.fetch_add(1, Ordering::Relaxed) == 0 {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            tracing::warn!(
                lists,
                "a submission names a host list twice: it is carried out as submissions of its own, one after the other, each \
                 waited for (not sent ahead); counted on the frame line"
            );
        });
    }
}

/// Tests: whether the host device has the debug layer's messages
/// (`VICTORY_D3D12_DEBUG=1` before it was made, and a machine that has the
/// layer).
#[cfg(all(test, feature = "diagnostics"))]
pub(crate) fn debug_layer_on() -> bool {
    host().is_some_and(|h| h.debug_layer_on())
}

/// Tests: the debug layer's errors since the last call, its stored
/// messages taken first ([`Host::log_messages`]).
#[cfg(all(test, feature = "diagnostics"))]
pub(crate) fn take_debug_errors() -> Vec<messages::DebugMessage> {
    if let Some(h) = host() {
        h.log_messages();
    }
    std::mem::take(&mut *messages::DEBUG_ERRORS.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Tests: the debug layer's errors since the last call on a device of the
/// test's own (`None` without the layer); the shared host's are
/// [`take_debug_errors`].
#[cfg(test)]
pub(crate) fn take_host_debug_errors(host: &Host) -> Option<Vec<messages::DebugMessage>> {
    #[cfg(feature = "diagnostics")]
    {
        host.messages.as_ref()?;
        host.log_messages();
        Some(std::mem::take(&mut *messages::DEBUG_ERRORS.lock().unwrap_or_else(|e| e.into_inner())))
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = host;
        None
    }
}

/// A list's indirect-work budget ([`HostList::budget`]).
struct Budget {
    /// The words the clamps charge and note in (`crate::indirect`), in the
    /// card's memory: in the PC's, every clamp's atomics on them crossed
    /// the bus.
    card: ID3D12Resource,
    /// Where they are copied when the list ends, in the PC's memory, and
    /// its mapping: what the clamps noted for the frame line.
    notes: ID3D12Resource,
    data: usize,
}

/// Zeros in upload memory, made once: what lists' budgets are reset from.
fn budget_zeros(host: &Host) -> Option<&'static ID3D12Resource> {
    static ZEROS: OnceLock<Option<ID3D12Resource>> = OnceLock::new();
    ZEROS
        .get_or_init(|| {
            let (buffer, data) = host.upload_buffer(256)?;
            // SAFETY: the buffer's own mapping of 256 bytes.
            unsafe { std::ptr::write_bytes(data, 0, 256) };
            name(&buffer, || "indirect budget zeros".to_string());
            Some(buffer)
        })
        .as_ref()
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
pub(crate) fn timing() -> bool {
    #[cfg(test)]
    if let Some(on) = TIMING_FORCED.with(std::cell::Cell::get) {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switches::switch_on("VICTORY_GPU_TIMING"))
}

/// The groups of a synchronous submission's `lists`, each sent with a
/// signal of the host's fence after it: with `VICTORY_GPU_TIMING`
/// (`timing`) each title list on its own (`firsts`: where each begins; what
/// follows the last, the compare of copies in the card's memory, goes with
/// it), so none without a title list, as on `job/land-2d` (Sol's fourth read
/// of the frame timers: one empty group went, signalled and waited for);
/// else all together. One block, as `job/land-2d` made.
#[cfg(feature = "diagnostics")]
fn send_groups<'a, T>(timing: bool, firsts: &[usize], lists: &'a [T]) -> Vec<&'a [T]> {
    if !timing {
        return vec![lists];
    }
    firsts.iter().zip(firsts.iter().skip(1).chain([&lists.len()])).map(|(&a, &b)| &lists[a..b]).collect()
}

/// How many groups [`send_groups`] makes of a submission's lists, without
/// making them (Sol's third read of the frame timers: the count allocated
/// on every synchronous submission, the switch unset). With timing and no
/// title list, none: nothing signals after the submission, and a hash list
/// it sends for copies in the card's memory is not stamped
/// ([`Host::card_before`]: no value it could be read at).
#[cfg(feature = "diagnostics")]
fn send_signals(timing: bool, firsts: &[usize]) -> u64 {
    if timing { firsts.len() as u64 } else { 1 }
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// One in [`ALIVE`]`[0]` while it lives (a host list), and one in
/// [`LISTS`]`[3]` until it is first reset.
struct Counted {
    /// Not reset since it was made.
    fresh: std::sync::atomic::AtomicBool,
}

impl Counted {
    fn new() -> Counted {
        ALIVE[0].fetch_add(1, Ordering::Relaxed);
        LISTS[3].fetch_add(1, Ordering::Relaxed);
        Counted {
            fresh: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// The list is reset: it is in use.
    fn used(&self) {
        if self.fresh.swap(false, Ordering::Relaxed) {
            LISTS[3].fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        ALIVE[0].fetch_sub(1, Ordering::Relaxed);
        self.used();
    }
}

/// A chunk of upload memory for lists' scratch data: the buffer, its
/// mapping and its host GPU address.
struct Chunk {
    _buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
}

// SAFETY: a D3D12 resource (free-threaded) and its persistent mapping,
// written only by the list that holds the chunk.
unsafe impl Send for Chunk {}

/// Upload chunks handed out front to back: the chunks (each with its
/// bytes), the one in use and the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(Chunk, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

impl Drop for Scratch {
    fn drop(&mut self) {
        give_chunks(std::mem::take(&mut self.chunks));
    }
}

const SCRATCH_CHUNK: u64 = 4 << 20;
/// A list's first chunk when the chunks are shared ([`scratch_pool`]): most
/// lists put a few constants or a rectangle's corners there, and every
/// list that did took 4 MiB of the PC's memory for it.
const SCRATCH_FIRST: u64 = 256 << 10;
/// The most bytes of chunks kept while no list holds them.
const SPARE_MOST: u64 = 128 << 20;

/// Whether lists share their scratch chunks: a list takes a chunk when it
/// first needs one, a small one first, and gives all of them back when it
/// is reset or dropped, for the next list. Before, a list kept every chunk
/// it ever took (4 MiB each) for as long as it lived, used or not, and the
/// host lists of held submissions, which live for one submission, each
/// made theirs anew: dozens of upload buffers of 4 MiB made, zeroed and
/// freed per frame of gameplay.
/// `VICTORY_SCRATCH_POOL=0`: chunks of 4 MiB kept by their list, as before.
fn scratch_pool() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SCRATCH_POOL").is_none_or(|v| v != "0"))
}

/// Things of some sizes that nothing holds now, kept for whoever next asks
/// for one of the same size.
struct Spare<T> {
    kept: Vec<(T, u64)>,
    bytes: u64,
}

impl<T> Spare<T> {
    const fn new() -> Self {
        Spare { kept: Vec::new(), bytes: 0 }
    }

    /// A kept thing of `size` bytes, the one given last.
    fn take(&mut self, size: u64) -> Option<T> {
        let at = self.kept.iter().rposition(|(_, s)| *s == size)?;
        self.bytes -= size;
        Some(self.kept.remove(at).0)
    }

    /// Keeps `thing` (`size` bytes) unless that takes the kept bytes past
    /// `most`: then it comes back.
    fn give(&mut self, thing: T, size: u64, most: u64) -> Option<T> {
        if self.bytes + size > most {
            return Some(thing);
        }
        self.bytes += size;
        self.kept.push((thing, size));
        None
    }
}

/// Scratch chunks no list holds.
static SPARE: Mutex<Spare<Chunk>> = Mutex::new(Spare::new());

/// A scratch chunk of `size` bytes: one kept ([`SPARE`]) or a new one.
fn take_chunk(host: &Host, size: u64) -> Option<Chunk> {
    if scratch_pool()
        && let Some(chunk) = SPARE.lock().unwrap_or_else(|e| e.into_inner()).take(size)
    {
        ALIVE[3].fetch_sub(1, Ordering::Relaxed);
        ALIVE[4].fetch_sub(size, Ordering::Relaxed);
        ALIVE[5].fetch_add(1, Ordering::Relaxed);
        return Some(chunk);
    }
    let (buffer, data) = host.upload_buffer(size)?;
    name(&buffer, || "list scratch".into());
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    ALIVE[1].fetch_add(1, Ordering::Relaxed);
    ALIVE[2].fetch_add(size, Ordering::Relaxed);
    Some(Chunk {
        _buffer: buffer,
        data: data as usize,
        gpu,
    })
}

/// A list is done with `chunks` (it was reset after its submissions ran,
/// or dropped): kept for the next list, or freed past [`SPARE_MOST`] and
/// without the pool.
fn give_chunks(chunks: Vec<(Chunk, u64)>) {
    if chunks.is_empty() {
        return;
    }
    let mut freed = Vec::new();
    {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        for (chunk, size) in chunks {
            let back = if scratch_pool() {
                spare.give(chunk, size, SPARE_MOST)
            } else {
                Some(chunk)
            };
            match back {
                None => {
                    ALIVE[3].fetch_add(1, Ordering::Relaxed);
                    ALIVE[4].fetch_add(size, Ordering::Relaxed);
                }
                Some(chunk) => {
                    ALIVE[1].fetch_sub(1, Ordering::Relaxed);
                    ALIVE[2].fetch_sub(size, Ordering::Relaxed);
                    freed.push(chunk);
                }
            }
        }
    }
    // Freed outside the lock.
    drop(freed);
}

/// What the host keeps for copies in the card's own memory
/// ([`crate::shadow::cards`]): three lists of ours and the compare.
struct CardWork {
    /// Before a submission's lists: what changed on the copies' CPU side
    /// goes to their card side. Those not in use: a plan takes one and
    /// gives it back when its submission has run (`Host::card_done`), so
    /// that submissions sent ahead ([`late`]) each have their own.
    sync: Vec<Prelude>,
    /// After them, in the same submission: the compare of the declared
    /// ranges with their bytes from before. As `sync`.
    check: Vec<Prelude>,
    /// After it ran: the pieces that differ come down.
    fetch: Prelude,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// One dword per piece compared, set where the piece differs: PC
    /// memory, mapped (buffer, mapping, bytes).
    differs: (ID3D12Resource, usize, u64),
    /// Where the pieces that differ are fetched to: PC memory, mapped.
    fetched: Option<(ID3D12Resource, usize, u64)>,
    /// The dwords of `differs` taken by plans whose results were not read
    /// yet.
    taken: u32,
    /// Plans made and not done with (`Host::card_done`).
    out: u32,
    /// With `VICTORY_GPU_STAMPS=1`: the pair around `fetch`'s copies.
    #[cfg(feature = "diagnostics")]
    stamps: crate::stamps::Stamps,
}

// SAFETY: D3D12 objects (free-threaded) and persistent mappings, used only
// under the submission lock.
unsafe impl Send for CardWork {}

static CARD_WORK: Mutex<Option<CardWork>> = Mutex::new(None);

/// Whether the pieces hashed again after ranges were sent up to a copy in
/// the card's memory are hashed in runs of pieces that touch, each piece
/// once (`VICTORY_CARD_REHASH_RUNS=1`; off by default), and not in a pass
/// per range sent up.
///
/// These passes are ours, they run on the card before every submission's
/// lists, and no timing so far names them: each walks its 64 KiB pieces
/// 16 bytes at a time, 4,096 steps one after the other, so a pass takes
/// the time of one piece however few its pieces are. A run sent up hundreds of
/// ranges a frame: as many such passes. The same pieces are hashed either way;
/// a piece hashed when it need not be, or not hashed when it should, costs
/// a fetch that finds nothing and loses no shader write.
fn rehash_runs() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_REHASH_RUNS").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line, of our own passes over copies in the
/// card's memory: hash passes after ranges were sent up, the pieces they
/// hashed, and compare passes after the lists.
static CARD_PASSES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Whether what shaders wrote to copies in the card's memory is merged
/// into title memory on several threads (`VICTORY_WIDE_MERGE=1`; off by
/// default). The pieces the card's compare found differing come down in
/// one fetch and were then merged one after the other on the thread that
/// settles: hundreds of pieces and tens of megabytes a frame, each compared
/// with the copy's CPU side byte by byte and put into title memory through
/// the kernel. A piece is its own bytes of its copy and of title memory,
/// so eight threads take an eighth of them each.
fn wide_merge() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_MERGE").is_some_and(|v| v == "1"))
}

/// One declared range of a copy in the card's memory, in a submission.
struct CardRange {
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    /// As declared (title addresses), for the statistics.
    declared: (usize, usize),
    span: crate::shadow::CardSpan,
    /// Its first piece's dword in the compare's results.
    at: u32,
    /// Whether the card compares it; else every piece is fetched.
    compared: bool,
}

/// What a submission does for copies in the card's memory.
struct CardPlan {
    ranges: Vec<CardRange>,
    /// The compare, run after the submission's lists.
    check: Option<ID3D12CommandList>,
    /// The copies whose card side was brought up to date, alive until the
    /// submission has run.
    _synced: Vec<std::sync::Arc<crate::shadow::Shadow>>,
    /// The lists of ours the plan recorded (whether the compare, the
    /// list), given back by `Host::card_done`.
    lists: Vec<(bool, Prelude)>,
    /// With `VICTORY_GPU_STAMPS=1`: the pairs around the hash list and the
    /// compare, read in `Host::card_done`.
    #[cfg(feature = "diagnostics")]
    stamps: Option<Box<crate::stamps::Stamps>>,
    /// Made with `VICTORY_FETCH_QUEUE` on: its ticket ([`fetch`]).
    fetch: Option<fetch::Ticket>,
    /// Sampled for `VICTORY_HASH_CHECK=1`: after its copy-back, what that
    /// did not fetch is fetched apart and compared ([`hash_check`]).
    hash_check: bool,
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(upload::Fill),
    /// A run of such copies with distinct destinations and sources outside
    /// their span: run together, on the pool with `VICTORY_UPLOAD_POOL=1`
    /// ([`run_fills`]).
    Fills(Vec<upload::Fill>),
    /// A table constant buffer's fill into list scratch, counted apart
    /// (the native-memory plan, step 2).
    TableFill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy.
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>),
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    /// The bytes of `uploads` (counted in [`ALIVE`] while they are held).
    upload_bytes: u64,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// Title textures (by resource) the list uploads, and the first mip
    /// its upload covers: noted once the list holds the copies, or the
    /// staging buffer its host copy is known to hold
    /// ([`crate::texture::upload_from`]). The texture's shared staging
    /// entry says nothing about this list: another list may have put one
    /// of more mips there.
    mips: std::collections::HashMap<usize, u32>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Textures read back (texture to buffer copies), done once the list
    /// has run: the host buffer the list copies the texels into, the title
    /// memory they go to, the bytes of a row, the rows, the title's row
    /// pitch; and, for the walls trace only, the title texture read and
    /// the frame the copy was recorded in.
    backs: Vec<(ID3D12Resource, usize, [usize; 3], usize, u64)>,
    /// Whether the list resets its indirect-work budget on the card (and a
    /// step takes what the last run noted in it).
    budget_reset: bool,
    /// Diagnostics, each time the list has run (`HostList::defer_after`):
    /// what takes the bytes of read-back buffers of ours, kept until the
    /// list is reset, and gives the report made of them, which
    /// [`report_after`]'s thread makes and logs.
    #[cfg(feature = "diagnostics")]
    after: Vec<Box<dyn Fn() -> Option<Report> + Send>>,
    /// Diagnostics: the read-backs of ours the list holds for `after`, and
    /// their bytes, up to [`LIST_READS_MOST`] and [`LIST_READ_BYTES_MOST`].
    #[cfg(feature = "diagnostics")]
    reads: (u32, u64),
    /// Title textures the list renders to or writes in a listed frame, for
    /// that frame's target dump once the list is submitted (diagnostics:
    /// `crate::capture::note_target`): the frame, the title resource, its
    /// host copy, held.
    #[cfg(feature = "diagnostics")]
    targets: Vec<(u64, usize, std::sync::Arc<crate::resource::HostTexture>)>,
}

impl Deferred {
    /// Holds `buffer` (`size` bytes of upload memory) until the list is
    /// reset.
    fn hold_upload(&mut self, buffer: ID3D12Resource, size: u64) {
        ALIVE[6].fetch_add(1, Ordering::Relaxed);
        ALIVE[7].fetch_add(size, Ordering::Relaxed);
        self.upload_bytes += size;
        self.uploads.push(buffer);
    }
}

impl Drop for Deferred {
    fn drop(&mut self) {
        ALIVE[6].fetch_sub(self.uploads.len() as u64, Ordering::Relaxed);
        ALIVE[7].fetch_sub(self.upload_bytes, Ordering::Relaxed);
    }
}

/// Whether a queue's submissions go to the card without a wait for each
/// (on by default at stage 3, which the relay asks for when
/// `VICTORY_LATE_WAIT` is unset; `=1`, `2` or `3` names the stage, any
/// other value turns it off, and so does `VICTORY_RELAY=0` with it unset;
/// off in the timing run, which waits for every list on its own).
///
/// Every submission was prepared, sent, waited for and copied back before
/// the next one was looked at, so the card stood idle while we prepared
/// and we stood idle while it ran: a frame was the sum of its
/// preparing, its waiting (the largest part) and its copying back, one after the
/// other, on the one thread that submits.
///
/// With the switch a queue's submission is sent and joins a line of
/// "flights" ([`Flight`]); the thread goes on to the next one while the
/// card runs. The flights are settled together ([`Host::settle`]: one wait
/// for the card, what their shaders wrote copied back once per byte, then
/// their bottom-of-pipe writes and the fence signals behind them, in
/// order) at the next point where the title, or our own preparing, can
/// observe what they do. Those points, exactly:
///
/// 1. a synchronous submission of ours ([`Host::execute`]): a present, a
///    read-back or map of a texture, a clear or upload run at once;
/// 2. the next submission reads title memory that a flight's shaders may
///    write: a title buffer copy, a fill of upload memory, a texture's
///    memory, any step that runs code of ours on title memory
///    ([`late_blocked`]);
/// 3. the next submission refreshes a range of a host copy that a flight's
///    shaders may write, and the title wrote a page of it since it was last
///    copied ([`stale_under_flights`]): the refresh leaves such a range
///    alone until the flight's writes are back, as it always did within a
///    submission;
/// 4. a queue stands at a wait for a memory value that no flight's
///    bottom-of-pipe write is known to satisfy, at an address a flight may
///    write ([`late_met`]);
/// 5. the title blocks on a fence without an event;
/// 6. the card has run every flight (a thread of ours waits for that:
///    [`late_helper`]), or the flights hold [`late_most`] host lists.
///
/// What the title is told (bottom-of-pipe writes, fence values) is told
/// when the card has really run the work, later than before and never
/// earlier; a fence signalled on a queue while flights are in the air is
/// set behind them. A wait for a memory value that a flight's
/// bottom-of-pipe write satisfies counts as met: the waiting part goes to
/// the card behind that flight, which is the order the wait asks for.
///
/// What it assumes beyond today: memory the title rewrites between two of
/// its submissions, without waiting for the first, may be seen by the
/// first's shaders (as on the console, where the card reads title memory
/// when it runs, not when the list is submitted).
pub(crate) fn late() -> bool {
    late_stage() > 0
}

/// `VICTORY_LATE_WAIT`: 0 (off), 1, 2 ([`late_exact`]) or 3
/// ([`late_free_present`]); unset, 3 with the relay and 0 without it.
fn late_stage() -> u8 {
    static STAGE: OnceLock<u8> = OnceLock::new();
    *STAGE.get_or_init(|| {
        if crate::switches::switch_on("VICTORY_GPU_TIMING") {
            return 0;
        }
        match std::env::var("VICTORY_LATE_WAIT").as_deref() {
            Ok("1") => 1,
            Ok("2") => 2,
            Ok("3") => 3,
            // The relay alone: on top of stage 3.
            Err(_) if crate::shadow::relay() => 3,
            _ => 0,
        }
    })
}

/// Stage 2 of [`late`] (`VICTORY_LATE_WAIT=2`): point 3 of its list is no
/// longer a point. Where the title wrote a page that lies in a range a
/// flight's shaders may write, the bytes the title changed are found (the
/// page as it is now against what the copy held when the flight was
/// prepared) and put into the copy, and only those: what the flight's
/// shaders write beside them stays (`Shadow::merge_under_writes`). The
/// flights are settled for it only where that cannot be done: memory that
/// is not write-watched, or a copy in the card's memory with more changed
/// runs than a submission should send up one by one.
///
/// At first this point settled the flights several times a frame,
/// each time with the card's work still to wait for: most of the
/// frame's waiting was left.
///
/// What differs from settling first: a byte both the title and a flight's
/// shader wrote is the title's if the shader wrote it first, the shader's
/// if it wrote it afterwards; settled first it was always the shader's.
/// The title writing a byte a shader of its own is writing is a race on
/// the console too.
fn late_exact() -> bool {
    late_stage() > 1
}

/// Stage 3 of [`late`] (`VICTORY_LATE_WAIT=3`: stage 2, and the present is
/// no longer a point where flights are settled). The copy to the back
/// buffer is sent as a flight of its own (`Host::execute_own`, a list of
/// its own for each present in the air) and the swap chain is told to
/// present behind it; the queue's call returns, and the card finishes the
/// frame while the title records the next one. What the title is told
/// about the frame (bottom-of-pipe writes, fences) is told when the card
/// has run it, by the settling thread.
///
/// What is left out with it: the look whether the presented frame is black
/// (diagnostics; it reads the frame back at once). A frame that is saved
/// or listed is presented the old way. Once every second and a half a
/// submission settles the flights all the same, for the sweep that lets
/// idle host objects go (it needs nothing in the air).
pub(crate) fn late_free_present() -> bool {
    late_stage() > 2
}

/// The relay's first stage (`VICTORY_RELAY=1`; the relay is on by default,
/// at its second stage, [`relay_exact`], and off for any value but `1` or
/// `2`; it turns [`late`] on at stage 3 unless `VICTORY_LATE_WAIT` says
/// another stage, and is off without [`late`], in the timing run and with
/// kept snapshots).
///
/// With [`late`] alone every settle does three things on the thread that
/// holds the submission lock: it waits for everything in the air, copies
/// back with the card standing, and only then goes on preparing. The frame
/// stays a relay: the card works about half of it and one thread
/// prepares, waits and copies back for about as long.
///
/// With the switch a thread of its own ([`relay_worker`]) lands flights as
/// soon as the card has run them, in the order they were sent and without
/// the submission lock: it fetches and merges what their shaders wrote to
/// copies in the card's memory, compares and merges the other copies,
/// frees the flights' ranges, and then does their read-backs, their
/// bottom-of-pipe writes and the fence signals behind them, in the line's
/// order. The submitting thread meanwhile prepares and sends the next
/// submissions. [`Host::settle`] no longer settles anything: it waits until
/// that thread has landed everything in the air, at the same points as
/// before (the list in [`late`]), so whoever can observe title memory, a
/// fence or a present still does so only after the flights before it are
/// merged.
///
/// What makes that safe is in `crate::shadow::relay` (ranges counted per
/// flight; a merge that never stores into memory the card may be writing;
/// sends to card copies numbered) and here: flights land in the order of
/// [`LINE`] by one thread, so a later flight's results never reach title
/// memory before an earlier flight's; the compare on the card stores the
/// hash it made, so each flight's compare flags exactly what changed since
/// the compare before it, however late the pieces are fetched; the fetch
/// goes to the queue under [`QUEUE_ORDER`], which says which sends of
/// title bytes the card had received when it ran.
pub(crate) fn relay() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| late_stage() > 0 && crate::shadow::relay())
}

/// The relay's second stage (`VICTORY_RELAY=2`): a submission that reads
/// title memory a flight's shaders may write (a title buffer copy, a fill
/// of upload memory, a texture) waits only until the flights that declared
/// that memory have landed, not for everything in the air; and at
/// [`late_most`] lists in the air it waits until there are fewer. At `1`
/// both wait for everything.
fn relay_exact() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    // Stage 2 unless `VICTORY_RELAY=1` asks for the first one (the default
    // since an interior's walls were found drawn in every run with
    // this set and in fewer than half without it).
    *ON.get_or_init(|| relay() && std::env::var_os("VICTORY_RELAY").is_none_or(|v| v == "2"))
}

/// Told when something joins [`LINE`], for the thread that lands flights
/// ([`relay`]); [`LINE_MOVED`] stays the word for the thread that looks at
/// what queues hold.
static RELAY_MOVED: std::sync::Condvar = std::sync::Condvar::new();

/// Told when flights have landed ([`relay`]): who waits for that looks
/// again.
static LINE_LANDED: std::sync::Condvar = std::sync::Condvar::new();

/// With [`relay`], the order of what we put on the queue: the number of
/// sends of changed ranges to copies in the card's memory so far
/// (`Host::card_before`). Held from the taking of the ranges until their
/// list is on the queue, and while a fetch of pieces goes to the queue: a
/// fetch sent with the count at N ran behind sends 1..N and ahead of every
/// later one.
static QUEUE_ORDER: Mutex<u64> = Mutex::new(0);

/// Since the last frame-rate line, of the thread that lands flights
/// ([`relay`]): landings, flights landed, microseconds it waited for the
/// card with nothing to land, microseconds recording fetches of card
/// copies' pieces, microseconds of landing in all; then of the fetches:
/// microseconds sending them (the queue's order, the list, the signals),
/// waiting for the card to run them (and what was sent before them), and
/// merging what they brought; bytes fetched, and of them merged into title
/// memory; then the parts of "copying back" the frame line does not split
/// ([`copy_back_parts`]): microseconds merging the copies in the PC's
/// memory, microseconds of the flights' ranges being done with
/// (`copy_back_done`), and microseconds looking the copies in the PC's
/// memory through (the same as `copystat::C::BackScanUs`; kept here so
/// that one line has all five parts).
static RELAY_STATS: [std::sync::atomic::AtomicU64; 13] = [const { std::sync::atomic::AtomicU64::new(0) }; 13];

/// The parts of the frame line's "copying back" per frame, as milliseconds,
/// from [`RELAY_STATS`] as [`late_line`] took them: waiting for the fetch,
/// the card merge, the PC-side scan, the PC-side merge, and the flights'
/// ranges being done with. With `VICTORY_LAND_SCAN` the scan runs under the
/// fetch, so those two add up to more than the landings' time (they
/// overlap); with `VICTORY_EXACT_BACK` the merges store fewer bytes, which
/// the frame line's megabytes show.
fn copy_back_parts(r: &[f64]) {
    tracing::info!(
        "copying back per frame, in parts: {back_fetch_ms:.1} ms waiting for fetches, {back_card_ms:.1} ms card merge, \
         {back_scan_ms:.1} ms PC-side scan, {back_merge_ms:.1} ms PC-side merge, {back_done_ms:.1} ms done with ranges",
        back_fetch_ms = r[6] / 1000.0,
        back_card_ms = r[7] / 1000.0,
        back_scan_ms = r[12] / 1000.0,
        back_merge_ms = r[10] / 1000.0,
        back_done_ms = r[11] / 1000.0,
    );
}

/// Of the line's things in order (a flight: the value signalled on the
/// host's fence behind it; a write or a signal behind a flight: `None`):
/// how many, from the front, can be landed when the card has reached
/// `completed`: everything ahead of the first flight it has not run. A
/// later flight never lands before an earlier one, and what follows a
/// flight (its bottom-of-pipe writes, signals) never before the flight.
fn landable(line: &[Option<u64>], completed: u64) -> usize {
    line.iter().position(|f| f.is_some_and(|f| f > completed)).unwrap_or(line.len())
}

/// For one plan of copies in the card's memory, with [`relay`]: `flags`
/// holds, for every piece of every range of the plan, the copy, the
/// piece's number in the copy and whether the compare flagged it there (or
/// did not run). The pieces to fetch, under whichever range: all of them
/// flagged under any. Two ranges of a plan may hold parts of the same
/// 64 KiB piece; the compare hashes the whole piece for each and stores
/// the hash, so the second of them may find it stored already and not
/// flag it.
fn flagged_pieces(flags: &[(usize, u32, bool)]) -> std::collections::HashSet<(usize, u32)> {
    flags.iter().filter(|f| f.2).map(|f| (f.0, f.1)).collect()
}

#[cfg(test)]
mod relay_tests {
    use super::{flagged_pieces, landable};

    /// Two flights and what follows each: nothing of the second lands
    /// before the card has run it, and nothing behind a flight lands
    /// before the flight.
    #[test]
    fn flights_land_in_the_order_they_were_sent() {
        // Flight (fence 5), its write, a signal, flight (fence 6), its write.
        let line = [Some(5), None, None, Some(6), None];
        assert_eq!(landable(&line, 4), 0);
        assert_eq!(landable(&line, 5), 3);
        assert_eq!(landable(&line, 6), 5);
        assert_eq!(landable(&line, u64::MAX), 5);
        // Signals with no flight ahead land at once; behind one they wait.
        assert_eq!(landable(&[None, None], 0), 2);
        assert_eq!(landable(&[None, Some(9), None], 8), 1);
        assert_eq!(landable(&[], 3), 0);
        // After the first three landed, the rest when its turn comes.
        assert_eq!(landable(&line[3..], 5), 0);
        assert_eq!(landable(&line[3..], 6), 2);
    }

    /// A piece flagged under one range of a plan is fetched under every
    /// range that holds a part of it; pieces of other copies are not.
    #[test]
    fn a_piece_flagged_under_one_range_is_fetched_under_all() {
        let (a, b) = (0x1000usize, 0x2000usize);
        // Copy a: piece 3 under two ranges, flagged under the first only;
        // piece 4 not flagged; copy b: piece 3 not flagged, piece 9 flagged.
        let flags = [(a, 3, true), (a, 4, false), (a, 3, false), (b, 3, false), (b, 9, true)];
        let fetch = flagged_pieces(&flags);
        assert!(fetch.contains(&(a, 3)));
        assert!(!fetch.contains(&(a, 4)));
        assert!(!fetch.contains(&(b, 3)));
        assert!(fetch.contains(&(b, 9)));
        assert_eq!(fetch.len(), 2);
    }
}

/// The most host lists in the air before the flights are settled
/// (`VICTORY_LATE_WAIT_LISTS`, 256; `=128` is the value before: with the
/// relay it stopped the submitting thread twice a frame for most of a
/// tenth of a second while the card stood idle about as long;
/// 32 settled the flights three times a frame and 64 still more than once,
/// with no table refused; watch "tables asked, N refused"): their dynamic
/// descriptor slots, upload
/// memory and scratch chunks are held until then, and a frame's lists held
/// all at once ran the slots out before.
fn late_most() -> usize {
    static MOST: OnceLock<usize> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_LATE_WAIT_LISTS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(256).max(1)
    })
}

/// Whether a submission with bottom-of-pipe writes may fly
/// (`VICTORY_LATE_WAIT_WRITES=0`: it is settled as soon as it is sent, so
/// the title is told at once, as before; for telling a title that polls
/// such a value from one that does not).
fn late_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LATE_WAIT_WRITES").is_none_or(|v| v != "0"))
}

/// Per host copy, the ranges a submission's shaders may write, as declared
/// (address, end).
type Declared = (std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize)>);

/// What [`Host::submit`] hands back for a submission it sent and did not
/// wait for.
struct Sent {
    /// Its number among the submissions started.
    number: u64,
    card: Option<CardPlan>,
    back: Vec<Declared>,
    /// Its bottom-of-pipe writes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
}

/// A submission on the card that has not been waited for ([`late`]): its
/// host lists, taken from the title's (which it may reset meanwhile), and
/// what is to be done once it has run.
struct Flight {
    number: u64,
    /// The value signalled on the host's fence behind it.
    fence: u64,
    lists: Vec<crate::list::Detached>,
    order: Vec<usize>,
    card: Option<CardPlan>,
    back: Vec<Declared>,
    /// With `VICTORY_GPU_STAMPS=1`, for a flight of a list of ours the
    /// caller keeps (a present copy): that list's stamps.
    #[cfg(feature = "diagnostics")]
    own: Option<std::sync::Arc<Mutex<crate::stamps::Stamps>>>,
}

/// The flight has landed: the stamps of its lists are read
/// (`VICTORY_GPU_STAMPS=1`; each only if its fence says so).
#[cfg(feature = "diagnostics")]
fn flight_stamps_landed(flight: &Flight) {
    if !crate::stamps::on() {
        return;
    }
    for &i in &flight.order {
        flight.lists[i].host().stamps_landed();
    }
    if let Some(own) = &flight.own {
        crate::stamps::landed_in(own);
    }
}

/// One thing in the line of what waits for the card, in order.
enum Late {
    Run(Flight),
    /// A bottom-of-pipe write of the flight before it: address, what, bytes.
    Write(usize, crate::queue::Written, usize),
    /// A fence (its address) signalled behind the flights before it.
    Signal(usize, u64),
}

struct Line {
    line: Vec<Late>,
    /// The title memory the flights' shaders may write (address, end), in
    /// order and joined.
    written: Vec<(usize, usize)>,
    /// Host lists in the flights.
    lists: usize,
    /// The value signalled on the host's fence behind the last flight.
    target: u64,
    /// A settle has taken the line and is not done: a signal still goes
    /// behind it.
    settling: bool,
    /// Something was settled away from a queue's own calls: the helper
    /// thread looks at what queues hold.
    poke: bool,
    /// Host lists of settled flights, reset and given back outside the
    /// submission lock ([`land`]: a settle may run under a caller's own
    /// locks).
    landed: Vec<(Vec<crate::list::Detached>, Vec<usize>)>,
    /// With [`relay`]: the bottom-of-pipe writes of the flights being
    /// landed (taken out of `line`, not done yet), for [`late_met`].
    in_hand: Vec<(usize, crate::queue::Written, usize)>,
}

// SAFETY: D3D12 objects (free-threaded) and host lists nothing records
// into; touched under the lock.
unsafe impl Send for Line {}

static LINE: Mutex<Line> = Mutex::new(Line {
    line: Vec::new(),
    written: Vec::new(),
    lists: 0,
    target: 0,
    settling: false,
    poke: false,
    landed: Vec::new(),
    in_hand: Vec::new(),
});
/// Told when a flight joins [`LINE`] or it is poked.
static LINE_MOVED: std::sync::Condvar = std::sync::Condvar::new();

fn line() -> std::sync::MutexGuard<'static, Line> {
    crate::budget::lock(&LINE, Ph::QueueLockWait)
}

/// Why flights were settled ([`LATE_STATS`]' places from 2 on).
const WHY_SYNC: usize = 2;
const WHY_CARD_DONE: usize = 3;
const WHY_VALUE: usize = 4;
const WHY_STALE: usize = 5;
const WHY_READ: usize = 6;
const WHY_RUN: usize = 7;
const WHY_MOST: usize = 8;
const WHY_WRITES: usize = 9;
const WHY_FENCE: usize = 10;

/// Since the last frame-rate line: submissions sent without a wait;
/// settles; settles by reason ([`WHY_SYNC`]..); flights settled;
/// microseconds settles waited for the card; value waits met by a flight's
/// bottom-of-pipe write; fence signals set behind flights.
/// Then: the settles for title memory read, by what read it (a buffer
/// copy, a fill of upload memory, a texture).
static LATE_STATS: [std::sync::atomic::AtomicU64; 18] = [const { std::sync::atomic::AtomicU64::new(0) }; 18];

/// Since the last frame-rate line: the microseconds settles waited for the
/// card, by why they settled (the `WHY_` places).
static LATE_WAITED: [std::sync::atomic::AtomicU64; 11] = [const { std::sync::atomic::AtomicU64::new(0) }; 11];

/// Whether `[from, to)` touches one of `ranges` (in order, joined).
fn touches(ranges: &[(usize, usize)], from: usize, to: usize) -> bool {
    if from >= to {
        return false;
    }
    let at = ranges.partition_point(|r| r.1 <= from);
    ranges.get(at).is_some_and(|r| r.0 < to)
}

/// `ranges` with `more`, in order, joined where they overlap or touch.
fn join_ranges(ranges: &mut Vec<(usize, usize)>, more: impl IntoIterator<Item = (usize, usize)>) {
    ranges.extend(more.into_iter().filter(|r| r.0 < r.1));
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(from, to) in ranges.iter() {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    *ranges = out;
}

/// Whether anything is in the air.
fn flying() -> bool {
    if !late() {
        return false;
    }
    let line = line();
    // (With the relay: also while flights are being landed.)
    line.settling || line.line.iter().any(|l| matches!(l, Late::Run(_)))
}

/// With flights in the air ([`late`]): why `runs` cannot be prepared
/// before they are settled, if so. Preparing reads title memory on the
/// CPU (the title's buffer copies, fills of upload memory, textures) and
/// runs steps of ours that may read or write it anywhere; what the
/// flights' shaders write is not there yet.
fn late_blocked(runs: &[&HostList]) -> Option<usize> {
    let written = {
        let line = line();
        if line.written.is_empty() {
            return None;
        }
        line.written.clone()
    };
    blocked_by(&written, runs, true)
}

/// A fence signal on a queue ([`late`]): goes behind the flights in the
/// air, if there are any, and is set when they are settled. Whether it
/// did; if not the caller sets the fence now.
pub(crate) fn late_signal(fence: usize, value: u64) -> bool {
    if !late() {
        return false;
    }
    let mut line = line();
    if line.line.is_empty() && !line.settling {
        return false;
    }
    line.line.push(Late::Signal(fence, value));
    LATE_STATS[14].fetch_add(1, Ordering::Relaxed);
    drop(line);
    // A signal alone (behind a settle under way) has no flight to wake
    // the helper thread for it.
    LINE_MOVED.notify_one();
    RELAY_MOVED.notify_one();
    true
}

/// A queue stands at `wait`, which memory does not meet now ([`late`]):
/// whether it is met once the flights in the air have run. A flight's
/// bottom-of-pipe write of a value that meets it says yes without a wait
/// for the card: the part behind the wait goes to the card behind that
/// flight, which is the order the wait asks for. If the flights may write
/// the value in a way not known here (shaders, a timestamp, another
/// width), they are settled and memory is asked again.
pub(crate) fn late_met(wait: &crate::list::ValueWait) -> bool {
    let (address, bytes) = wait.place();
    {
        let line = line();
        if line.line.is_empty() && line.in_hand.is_empty() {
            return false;
        }
        let mut unsure = touches(&line.written, address, address + bytes);
        // (With the relay: first the writes of flights being landed.)
        let writes = line.in_hand.iter().map(|(at, what, n)| (at, what, n)).chain(line.line.iter().filter_map(|item| match item {
            Late::Write(at, what, n) => Some((at, what, n)),
            _ => None,
        }));
        for (at, what, n) in writes {
            if *at >= address + bytes || address >= *at + *n {
                continue;
            }
            match what {
                crate::queue::Written::Value(v) if *at == address && *n >= bytes => {
                    if wait.holds_for(*v) {
                        LATE_STATS[13].fetch_add(1, Ordering::Relaxed);
                        return true;
                    }
                }
                _ => unsure = true,
            }
        }
        if !unsure {
            return false;
        }
    }
    settle_now(WHY_VALUE);
    wait.met()
}

/// Settles the flights in the air, if any ([`Host::settle`]), for reason
/// `why` (one of the `WHY_` places). Not from inside a submission.
pub(crate) fn settle_now(why: usize) {
    if !late() {
        return;
    }
    let Some(host) = host() else {
        return;
    };
    let _call = ph(Ph::Unaccounted);
    let mut submitted = host.submission_lock();
    host.settle(&mut submitted, why);
}

/// [`settle_now`] at a present, and for a fence the title blocks on.
pub(crate) fn settle_for_present() {
    settle_now(WHY_SYNC);
}
pub(crate) fn settle_for_fence() {
    settle_now(WHY_FENCE);
}

/// Resets the host lists of settled flights and keeps them for reuse, as
/// a queue does for the lists it ran (`crate::list::recycle`).
fn land() {
    let landed = std::mem::take(&mut line().landed);
    for (lists, order) in landed {
        crate::list::recycle(lists, &order);
    }
}

/// The thread that settles the flights once the card has run them all
/// ([`late`]): without it a bottom-of-pipe write would reach the title only
/// at the next call that settles, and a title thread that polls such a
/// value while the submitting thread waits for that thread would never
/// see it.
fn late_helper() {
    let Some(host) = host() else {
        return;
    };
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        tracing::warn!("no event for the thread that settles submissions sent ahead");
        return;
    };
    loop {
        let target = {
            let mut line = line();
            loop {
                if line.poke {
                    line.poke = false;
                    break None;
                }
                // (With the relay the flights are another thread's.)
                if !relay() && !line.line.is_empty() && !line.settling {
                    break Some(line.target);
                }
                line = LINE_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
            }
        };
        let Some(target) = target else {
            land();
            crate::queue::release_held(false);
            continue;
        };
        // The card first, without the submission lock: the submitting
        // thread prepares and sends meanwhile.
        #[cfg(feature = "diagnostics")]
        let waiting = Instant::now();
        // SAFETY: our fence and event.
        unsafe {
            let _p = ph(Ph::CardWait);
            if host.fence.SetEventOnCompletion(target, event).is_ok() {
                WaitForSingleObject(event, INFINITE);
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::stalls::card_waited(crate::stalls::Site::LateHelper, waiting);
        {
            let mut submitted = host.submission_lock();
            // Something newer went up meanwhile: the card is waited for
            // again, the submitting thread is not held up for it.
            if line().target != target {
                continue;
            }
            host.settle(&mut submitted, WHY_CARD_DONE);
        }
        land();
        crate::queue::release_held(false);
    }
}

/// Starts the threads of [`late`], once: the one that settles flights when
/// the card has run them and looks at what queues hold, and with [`relay`]
/// the one that lands flights.
fn late_threads() {
    static HELPER: std::sync::Once = std::sync::Once::new();
    HELPER.call_once(|| {
        if let Err(e) = std::thread::Builder::new().name("victory-late-wait".into()).spawn(late_helper) {
            tracing::warn!(error = %e, "no thread to settle submissions sent ahead: they are settled by the calls that follow");
        }
        tracing::info!(
            most = late_most(),
            writes = late_writes(),
            "submissions are sent to the card without a wait for each (VICTORY_LATE_WAIT=1)"
        );
        if relay() {
            // Without it nothing would ever land.
            if let Err(e) = std::thread::Builder::new().name("victory-relay".into()).spawn(relay_worker) {
                fail(format_args!("no thread to land submissions sent ahead (VICTORY_RELAY=1): {e}"));
            }
            tracing::info!(
                exact = relay_exact(),
                "what shaders wrote is copied back by a thread of its own, beside the submitting thread (VICTORY_RELAY=1)"
            );
        }
    });
}

/// What the thread that lands flights fetches card copies' pieces with
/// ([`relay`]): a list, a fence and an event of its own (the host's are
/// behind the submission lock), and the memory the pieces come down to.
struct RelayFetch {
    list: Prelude,
    fence: ID3D12Fence,
    event: HANDLE,
    /// The last value signalled on `fence`.
    sent: u64,
    /// Buffer, mapping, bytes.
    buffer: Option<(ID3D12Resource, usize, u64)>,
    /// With `VICTORY_GPU_STAMPS=1`: the pair around a fetch's copies.
    #[cfg(feature = "diagnostics")]
    stamps: crate::stamps::Stamps,
}

/// The settles' flights each time, or, with the relay (whose thread lands
/// them, not the settles), that it did.
struct Each(Option<f64>);

impl std::fmt::Display for Each {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(n) => write!(f, "{n:.1} flights each time"),
            None => f.write_str("flights landed by the relay"),
        }
    }
}

/// For the frame-rate line, with [`late`]: what the line of flights did
/// per frame since the last one.
fn late_line(frames: f64) {
    if !late() {
        return;
    }
    let s: Vec<f64> = LATE_STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "submissions sent ahead per frame: {:.1} in {:.1} settles ({}), {:.1} ms host time waiting for settles; \
         settled for: a synchronous submission or present {:.1}, the card had run them {:.1}, a wait for a memory value {:.1}, \
         title memory changed under shader writes {:.1}, title memory read {:.1} (by a buffer copy {:.1}, a fill of upload memory {:.1}, \
         a texture {:.1}), a step of ours {:.1}, {} lists in the air {:.1}, \
         bottom-of-pipe writes (VICTORY_LATE_WAIT_WRITES=0) {:.1}, a blocking fence wait {:.1}; \
         {:.1} waits for memory values met by a flight's bottom-of-pipe write, {:.1} fence signals set behind flights",
        s[0],
        s[1],
        // With the relay the settles land nothing: its thread does (Sol's
        // review of the frame timers: the ratio read 0 there).
        Each(if relay() { None } else { Some(if s[1] > 0.0 { s[11] / s[1] } else { 0.0 }) }),
        s[12] / 1000.0,
        s[WHY_SYNC],
        s[WHY_CARD_DONE],
        s[WHY_VALUE],
        s[WHY_STALE],
        s[WHY_READ],
        s[15],
        s[16],
        s[17],
        s[WHY_RUN],
        late_most(),
        s[WHY_MOST],
        s[WHY_WRITES],
        s[WHY_FENCE],
        s[13],
        s[14],
    );
    {
        let w: Vec<f64> = LATE_WAITED.iter().map(|w| w.swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames).collect();
        tracing::info!(
            "of the host time waiting for settles, per frame, by what settled: a synchronous submission or present {:.1} ms, \
             the card had run them {:.1}, a wait for a memory value {:.1}, title memory changed under shader writes {:.1}, \
             title memory read {:.1}, a step of ours {:.1}, lists in the air {:.1}, bottom-of-pipe writes {:.1}, a blocking fence wait {:.1}",
            w[WHY_SYNC],
            w[WHY_CARD_DONE],
            w[WHY_VALUE],
            w[WHY_STALE],
            w[WHY_READ],
            w[WHY_RUN],
            w[WHY_MOST],
            w[WHY_WRITES],
            w[WHY_FENCE],
        );
    }
    if relay() {
        let r: Vec<f64> = RELAY_STATS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        tracing::info!(
            "the relay per frame (VICTORY_RELAY=1): {:.1} flights landed in {:.1} landings by the thread that copies back, \
             beside the submitting thread; that thread waited {:.1} ms for the card with nothing to land; fetching card \
             copies' pieces took it {:.1} ms recording, {:.1} ms sending, {:.1} ms waiting for the card, then {:.1} ms \
             merging ({:.1} MB fetched, {:.1} MB of it merged into title memory); it was {:.1} ms landing in all, of which \
             the frame line's \"copying back\" is the part up to the flights' ranges being done with (the fetches, the \
             merges, the hand-back), not the read-backs, title writes, signals, retirement and logging after it; \
             \"settles\" above are waits of other threads for it",
            r[1],
            r[0],
            r[2] / 1000.0,
            r[3] / 1000.0,
            r[5] / 1000.0,
            r[6] / 1000.0,
            r[7] / 1000.0,
            r[8] / 1e6,
            r[9] / 1e6,
            r[4] / 1000.0,
        );
        // Where "copying back" goes, per run: the fetch's wait, the card
        // merge, the PC-side scan and merge, and the ranges being done
        // with (the frame line's fields are kept as they were).
        copy_back_parts(&r);
        fetch::line(frames);
        crate::watch::holds_line(frames);
        // The same parts, for the live stats file: the frame line has no room for them
        // and the stats keep them for a second write (`host/stats.rs`).
        crate::host::stats::remember_back_parts([
            r[6] as u64,
            r[7] as u64,
            r[10] as u64,
            r[11] as u64,
            r[12] as u64,
        ]);
    }
    if late_exact() {
        let [pages, runs, bytes] = crate::shadow::merged_stats().map(|n| n as f64 / frames);
        tracing::info!(
            "title bytes put in under flights' shader writes per frame (VICTORY_LATE_WAIT=2): {pages:.0} pages looked at, \
             {runs:.0} changed runs, {:.2} MB",
            bytes / (1 << 20) as f64
        );
    }
}

#[cfg(test)]
mod late_tests {
    use super::{continues, join_ranges, touches};

    /// Copies join only when both ends continue and the whole does not
    /// overlap itself.
    #[test]
    fn copies_join_only_end_to_end_and_apart() {
        // (destination, source, bytes)
        assert!(continues((1000, 5000, 16), (1016, 5016, 32)));
        assert!(continues((5000, 1000, 16), (5016, 1016, 32)));
        // A gap, or only one side continues.
        assert!(!continues((1000, 5000, 16), (1020, 5016, 32)));
        assert!(!continues((1000, 5000, 16), (1016, 5020, 32)));
        assert!(!continues((1000, 5000, 16), (1000, 5000, 16)));
        // Together they would overlap: the second reads what the first wrote.
        assert!(!continues((1000, 1016, 16), (1016, 1032, 16)));
        assert!(!continues((1016, 1000, 16), (1032, 1016, 16)));
        // They touch without overlapping.
        assert!(continues((1000, 1032, 16), (1016, 1048, 16)));
        assert!(!continues((usize::MAX - 8, 0, 8), (0, 8, 8)));
    }

    /// Ranges are found by what they hold, not by where they start, and a
    /// range that only touches another's end does not touch it.
    #[test]
    fn ranges_touch_where_they_overlap() {
        let mut ranges = Vec::new();
        join_ranges(&mut ranges, [(40, 50), (10, 20), (20, 25), (45, 60), (7, 7)]);
        assert_eq!(ranges, [(10, 25), (40, 60)]);
        join_ranges(&mut ranges, [(25, 40)]);
        assert_eq!(ranges, [(10, 60)]);
        let ranges = [(10usize, 25usize), (40, 60), (100, 101)];
        for (from, to, hit) in [
            (0, 10, false),
            (0, 11, true),
            (24, 25, true),
            (25, 40, false),
            (25, 41, true),
            (59, 200, true),
            (60, 100, false),
            (100, 101, true),
            (101, 500, false),
            (30, 30, false),
            (50, 50, false),
        ] {
            assert_eq!(touches(&ranges, from, to), hit, "{from}..{to}");
        }
        assert!(!touches(&[], 0, usize::MAX));
    }
}

/// Whether a title buffer copy that continues the one recorded before it
/// in its list becomes one step with it (`VICTORY_JOIN_COPIES=1`; off by
/// default).
///
/// Each copy is done at submission through the kernel, which reports a
/// page that is not there instead of faulting (`copy_mapped`), tells the
/// write watch and the guards, and ends the refreshes gathered so far: a
/// call into the kernel for a few hundred bytes, thousands of times a frame
/// (the line's "moves", with the refreshes' own reads). How many of them
/// continue the one before is counted either way and said on its own line
/// with the frame rate; the switch is worth that many kernel calls.
///
/// Joined, the same bytes move: [`continues`] joins only copies whose
/// ranges, taken together, do not overlap, and nothing is recorded between
/// two steps that follow each other. What differs: a page that is not
/// mapped now fails the joined copy as a whole (it then goes page by page,
/// zeros for what cannot be read, as a single copy does), where before
/// only the copy that held it did.
fn join_copies() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_JOIN_COPIES").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line: title buffer copies recorded, and those
/// that continue the one before ([`join_copies`]).
static COPIES_JOINED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether the copy `next` (destination, source, bytes) goes on exactly
/// where `last` ended, source and destination both, and the two as one do
/// not overlap (source over destination): then one copy of both moves the
/// bytes the two would move in turn.
fn continues(last: (usize, usize, usize), next: (usize, usize, usize)) -> bool {
    let ((d, s, n), (dst, src, len)) = (last, next);
    let Some(all) = n.checked_add(len) else {
        return false;
    };
    d.checked_add(n) == Some(dst)
        && s.checked_add(n) == Some(src)
        && (d.checked_add(all).is_some_and(|end| end <= s) || s.checked_add(all).is_some_and(|end| end <= d))
}

/// Until `fence` has reached `value`: by `event` when it can be set on the
/// fence, else by asking the fence every millisecond. A submission's
/// read-backs are mapped, its bottom-of-pipe writes made and its lists
/// reset for reuse only after this. A failed event registration or wait
/// was taken as done and only logged (Sol's review, finding 7). False if
/// the device was removed: its fence reads `u64::MAX`, which is no value
/// reached (Sol's re-check, finding 7), and the caller takes the device
/// failure path before it reads anything the submission wrote.
#[must_use]
fn wait_fence(device: &ID3D12Device, fence: &ID3D12Fence, event: Option<HANDLE>, value: u64) -> bool {
    let _p = ph(Ph::CardWait);
    // SAFETY: a live device and fence and, if given, an event of ours.
    unsafe {
        if let Some(event) = event
            && fence.SetEventOnCompletion(value, event).is_ok()
        {
            WaitForSingleObject(event, INFINITE);
        }
        loop {
            let done = fence.GetCompletedValue();
            if done == u64::MAX && device_removed(device) {
                return false;
            }
            if done >= value {
                return true;
            }
            #[cfg(test)]
            test_hook(HOOK_POLL);
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

/// Whether `device` was removed (tests: or counts as removed on this
/// thread, [`DEVICE_GONE`]).
fn device_removed(device: &ID3D12Device) -> bool {
    #[cfg(test)]
    if DEVICE_GONE.with(std::cell::Cell::get) {
        return true;
    }
    // SAFETY: a plain query.
    unsafe { device.GetDeviceRemovedReason() }.is_err()
}

/// A fetch of card copies' pieces was sent, the relay's or a synchronous
/// submission's ([`Host::card_back`]): waits until `fence` has reached
/// `value` on a live device, whatever wakes `event`; false if the device
/// was removed. Nothing the fetch wrote (its pieces, its stamps) is read
/// before, and nothing is merged (Sol's review of the frame timers: a
/// failed or early wake fell through to the reads; its third read: the
/// synchronous fetch merged on its event's wake alone, as on the base). A
/// failed registration of the event fails the fetch at once, as it did
/// before (Sol's recheck: it polled instead). After a wake the fence is
/// asked until it has its value: as long as the base's wait on the event,
/// which has no end either while the device lives and the fence does not
/// move (a hung card is removed by the system, and the fence then reads
/// `u64::MAX`).
fn fetch_waited(device: &ID3D12Device, fence: &ID3D12Fence, event: HANDLE, value: u64) -> windows::core::Result<bool> {
    let _p = ph(Ph::CardWait);
    #[cfg(test)]
    if FETCH_REGISTRATION_FAILS.with(std::cell::Cell::get) {
        return Err(windows::Win32::Foundation::E_FAIL.into());
    }
    // SAFETY: a live fence and an event of ours.
    unsafe {
        fence.SetEventOnCompletion(value, event)?;
        WaitForSingleObject(event, INFINITE);
    }
    Ok(wait_fence(device, fence, None, value))
}

/// Hands one report to [`report_after`]'s thread: lines made where waiting
/// or logging could hold something up (diagnostics). Never waits or logs.
#[cfg(feature = "diagnostics")]
pub(crate) fn report(work: Report) {
    reports_start();
    report_after(vec![work]);
}

/// Tests: waits until the work handed to [`report_after`] so far has run.
#[cfg(all(test, feature = "diagnostics"))]
pub(crate) fn reports_settled() {
    // One thread runs the work in the order handed over.
    let (done, wait) = std::sync::mpsc::channel::<()>();
    report(Box::new(move || {
        let _ = done.send(());
    }));
    let _ = wait.recv_timeout(Duration::from_secs(10));
}

/// Most pieces of diagnostics work waiting for [`report_after`]'s thread:
/// past them, more are dropped, and counted ([`REPORTS_DROPPED`]). Four
/// lists' worth: one list's completion hands over up to
/// [`LIST_READS_MOST`] at once.
#[cfg(feature = "diagnostics")]
const REPORTS_QUEUED: usize = 4 * LIST_READS_MOST as usize;

/// Reports dropped since the reporting thread last said so: counted on the
/// completion path, which logs nothing (Sol's re-check, new fault 3: the
/// first drop's warning there could wait for the sink that held the
/// reporter up), and logged by the reporting thread.
#[cfg(feature = "diagnostics")]
static REPORTS_DROPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The most read-backs of ours a list holds for diagnostics reports, and
/// their most bytes (`HostList::read_budget`): past them its lines say so
/// and nothing more is allocated (Sol's re-check, finding 6).
#[cfg(feature = "diagnostics")]
pub(crate) const LIST_READS_MOST: u32 = 256;
#[cfg(feature = "diagnostics")]
pub(crate) const LIST_READ_BYTES_MOST: u64 = 16 << 20;

/// Waits the completion path makes on report progress: the path makes none (Sol's review, finding 6), and the
/// ordering fixture asserts this stayed zero across a submission. The count is taken at the thing waited on —
/// the report queue's waiting methods (`send`, `send_timeout`) — so a wait added by anyone, marked or not,
/// counts when it is made from the handoff.
#[cfg(all(test, feature = "diagnostics"))]
static REPORT_WAITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// Tests: this thread is inside the completion path's handoff, the one place a wait on report progress is the
// path's (the reporting thread's own waits are its work).
#[cfg(all(test, feature = "diagnostics"))]
thread_local! {
    static IN_HANDOFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(all(test, feature = "diagnostics"))]
fn count_report_wait() {
    if IN_HANDOFF.with(std::cell::Cell::get) {
        REPORT_WAITS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Tests: inside the handoff until dropped.
#[cfg(all(test, feature = "diagnostics"))]
struct Handoff;

#[cfg(all(test, feature = "diagnostics"))]
impl Handoff {
    fn enter() -> Handoff {
        IN_HANDOFF.with(|f| f.set(true));
        Handoff
    }
}

#[cfg(all(test, feature = "diagnostics"))]
impl Drop for Handoff {
    fn drop(&mut self) {
        IN_HANDOFF.with(|f| f.set(false));
    }
}

/// The reporting thread's queue as the completion path sees it: the handoff uses the non-waiting send, and the
/// waiting methods exist so that a wait on report progress counts at the primitive itself — whoever writes the
/// call — while the handoff is on this thread.
#[cfg(feature = "diagnostics")]
#[derive(Clone)]
struct ReportQueue(std::sync::mpsc::SyncSender<Report>);

#[cfg(feature = "diagnostics")]
impl ReportQueue {
    fn try_send(&self, work: Report) -> Result<(), std::sync::mpsc::TrySendError<Report>> {
        self.0.try_send(work)
    }

    #[cfg(test)]
    fn send(&self, work: Report) -> bool {
        count_report_wait();
        self.0.send(work).is_ok()
    }

    /// A bounded wait on room in the queue, made of the stable calls (the queue's own send_timeout is unstable
    /// on this toolchain): the wait counts as a wait on report progress either way.
    #[cfg(test)]
    fn send_timeout(&self, work: Report, timeout: Duration) -> bool {
        count_report_wait();
        let began = std::time::Instant::now();
        let mut work = Some(work);
        while began.elapsed() < timeout {
            match self.0.try_send(work.take().expect("one send")) {
                Ok(()) => return true,
                Err(std::sync::mpsc::TrySendError::Full(back)) => {
                    work = Some(back);
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(std::sync::mpsc::TrySendError::Disconnected(_)) => return false,
            }
        }
        false
    }
}

/// Hands `work` to the reporting thread's queue `reports`, or counts it in
/// `dropped` when the queue is full or there is no thread. Never waits and
/// never logs: it runs on the completion path. (Tests: a waiting method on
/// the queue counts itself there, so a wait added here fails the ordering
/// fixture without any marking.)
#[cfg(feature = "diagnostics")]
fn hand_over(reports: Option<&ReportQueue>, work: Report, dropped: &std::sync::atomic::AtomicU64) {
    if !reports.is_some_and(|r| r.try_send(work).is_ok()) {
        dropped.fetch_add(1, Ordering::Relaxed);
    }
}

/// A report made of bytes taken from a list's read-back buffers once it
/// has run: it owns them, and makes and logs its lines.
#[cfg(feature = "diagnostics")]
pub(crate) type Report = Box<dyn FnOnce() + Send>;

/// Diagnostics: the reports made of what lists' read-back buffers held once
/// they had run (`HostList::defer_after`: making a line, logging it) are
/// made on a thread of their own. The completion's bottom-of-pipe writes
/// and fence signals, which tell the title its work is done, never wait for
/// them (Sol's review, finding 6). Each owns the bytes it reports.
#[cfg(feature = "diagnostics")]
fn report_after(work: Vec<Report>) {
    if work.is_empty() {
        return;
    }
    // Never waits: only while a start writes the queue's slot is there
    // none. Checks whether the thread runs only read it (a test's report
    // was dropped while another thread checked).
    let reports = match REPORTS.try_read() {
        Ok(slot) => slot.clone(),
        Err(std::sync::TryLockError::Poisoned(slot)) => slot.into_inner().clone(),
        Err(std::sync::TryLockError::WouldBlock) => None,
    };
    #[cfg(test)]
    let _handoff = Handoff::enter();
    for work in work {
        hand_over(reports.as_ref(), work, &REPORTS_DROPPED);
    }
}

/// The reporting thread's queue, once the thread runs: written only by a
/// start ([`started`]), read by everything else.
#[cfg(feature = "diagnostics")]
static REPORTS: std::sync::RwLock<Option<ReportQueue>> = std::sync::RwLock::new(None);

/// Starts the reporting thread unless it runs: where lists are recorded
/// (`HostList::defer_after`, [`report`]), never on the completion path.
/// A start that failed is tried again the next time (Sol's second
/// re-check, new fault 5: a failed start stayed failed for good).
/// Whether it runs.
#[cfg(feature = "diagnostics")]
fn reports_start() -> bool {
    #[cfg(test)]
    REPORT_STARTS.with(|n| n.set(n.get() + 1));
    started(&REPORTS, || {
        let (send, receive) = std::sync::mpsc::sync_channel::<Report>(REPORTS_QUEUED);
        let send = ReportQueue(send);
        let spawned = std::thread::Builder::new().name("victory-reports".into()).spawn(move || {
            loop {
                let (alive, dropped) = reporter_step(&receive, &REPORTS_DROPPED, REPORTS_IDLE);
                if dropped > 0 {
                    tracing::warn!(
                        dropped,
                        most = REPORTS_QUEUED,
                        "diagnostics reports dropped after lists had run: their thread was behind"
                    );
                }
                if !alive {
                    return;
                }
            }
        });
        spawned.ok().map(|_| send)
    })
}

/// Whether `slot` holds what `start` makes, starting it now if it does
/// not (and keeping nothing if it fails, to be tried again). Once it does,
/// the check only reads the slot: a report handed over meanwhile
/// ([`report_after`]) finds the queue.
#[cfg(feature = "diagnostics")]
fn started<T>(slot: &std::sync::RwLock<Option<T>>, start: impl FnOnce() -> Option<T>) -> bool {
    {
        let running = slot.read().unwrap_or_else(|e| e.into_inner());
        #[cfg(test)]
        test_hook(HOOK_START_CHECK);
        if running.is_some() {
            return true;
        }
    }
    let mut slot = slot.write().unwrap_or_else(|e| e.into_inner());
    if slot.is_none() {
        *slot = start();
    }
    slot.is_some()
}

/// How long the reporting thread waits for a report before it says what
/// was dropped meanwhile.
#[cfg(feature = "diagnostics")]
const REPORTS_IDLE: Duration = Duration::from_secs(1);

/// One turn of the reporting thread: the next report made, or `wait` with
/// none; then the reports dropped since the last turn, to be logged. A drop
/// after the last report is said at the next turn, waited for a bounded
/// time (Sol's second re-check, new fault 5: it waited for a report that
/// might never come). False once no queue can send any more.
#[cfg(feature = "diagnostics")]
fn reporter_step(
    receive: &std::sync::mpsc::Receiver<Report>,
    dropped: &std::sync::atomic::AtomicU64,
    wait: Duration,
) -> (bool, u64) {
    let alive = match receive.recv_timeout(wait) {
        Ok(work) => {
            work();
            true
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => true,
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => false,
    };
    (alive, dropped.swap(0, Ordering::Relaxed))
}

/// The ranges the shaders of `runs` may write, per host copy, as declared.
/// One list's lock at a time: a submission may name a list twice.
fn declared_writes(runs: &[&HostList]) -> Vec<Declared> {
    let mut back: Vec<Declared> = Vec::new();
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::GpuWrite(shadow, address, len) = step {
                let range = (*address, *address + *len);
                match back.iter_mut().find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow)) {
                    Some((_, ranges)) => {
                        if !ranges.contains(&range) {
                            ranges.push(range);
                        }
                    }
                    None => back.push((shadow.clone(), vec![range])),
                }
            }
        }
    }
    back
}

/// The walls trace (`VICTORY_WALLS_TRACE=1` or `2`; off by default): log
/// lines that say what the title's CPU was given while it decided whether
/// the camera is in a room. The values of the small read-backs
/// ([`read_backs`]), the frame's main pass by scissor rectangle with its
/// culling dispatches (`draw.rs`, `walls_census`), the shader-written
/// buffers merged into title memory (`shadow.rs`, `walls_merged`), host
/// copies found stale after their refresh (`shadow.rs`; it turns
/// `VICTORY_REFRESH_CHECK` on, which costs about 100 ms a frame), and, in
/// the runtime, the read requests that fail (with 2: every request).
///
/// In builds with diagnostics only: its lines are made where lists complete
/// and where they are recorded, and go to the reporting thread
/// ([`walls_line`]), which only those builds have (Sol's review of the
/// second landing, hold 8: logged in place, a sink that blocked held up the
/// read-backs' completion and the fences after them).
pub(crate) fn walls_trace() -> bool {
    if !cfg!(feature = "diagnostics") {
        return false;
    }
    #[cfg(test)]
    if WALLS_TEST.with(std::cell::Cell::get) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WALLS_TRACE").is_some_and(|v| v == "1" || v == "2"))
}

/// Hands a walls trace line (or the refresh check's, which the trace turns
/// on: `Shadow::refresh_check`), made of what it owns, to the reporting
/// thread from where lists are recorded, submitted or complete: never
/// waits, never logs, never starts the thread (it is started when the host
/// is made with the trace or the check on; a start that failed there is
/// said, and tried again only where the diagnostics reports of
/// `HostList::defer_after` are recorded); a line it cannot take is counted
/// as dropped.
pub(crate) fn walls_line(line: impl FnOnce() + Send + 'static) {
    #[cfg(feature = "diagnostics")]
    report_after(vec![Box::new(line)]);
    #[cfg(not(feature = "diagnostics"))]
    drop(line);
}

/// Tests: counts the events logged on the thread it is the default of.
/// With diagnostics only, as what uses it: the walls trace exists only
/// there ([`walls_trace`] is false before its test switch is asked).
#[cfg(all(test, feature = "diagnostics"))]
pub(crate) struct Logged(pub(crate) std::sync::Arc<std::sync::atomic::AtomicUsize>);

#[cfg(all(test, feature = "diagnostics"))]
impl tracing::Subscriber for Logged {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, _: &tracing::Event<'_>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// Tests: runs `f` with the walls trace on, on this thread, and gives the
/// events it logged on this thread.
#[cfg(all(test, feature = "diagnostics"))]
pub(crate) fn logged_here(f: impl FnOnce()) -> usize {
    let logged = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    WALLS_TEST.set(true);
    tracing::subscriber::with_default(Logged(logged.clone()), f);
    WALLS_TEST.set(false);
    logged.load(Ordering::SeqCst)
}

#[cfg(all(test, feature = "diagnostics"))]
thread_local! {
    /// Tests: the times this thread asked for the reporting thread to be
    /// started ([`reports_start`]), which takes the reporter's lock and may
    /// make the thread.
    pub(crate) static REPORT_STARTS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
thread_local! {
    /// Tests: the walls trace on, on this thread alone.
    pub(crate) static WALLS_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// After `list` has run: the texels its read-backs copied into host
/// buffers of ours go to the title's memory (`HostList::defer_read_back`).
///
/// A submission that is waited for did this itself. One sent ahead
/// ([`late`]) did not, at any stage: the flights' lists were settled
/// without it, so the title read a buffer nobody wrote, as before read-backs
/// were made (the exposure metering's two small float targets, every frame).
fn read_backs(list: &HostList) {
    #[cfg(feature = "diagnostics")]
    // The bytes are taken now: the submission lock is held (or, sent ahead,
    // the list is its flight's until it is recycled), so no later run of
    // the list writes them meanwhile. Each run is reported (Sol's re-check,
    // new fault 2: the work was handed over whole, raced the next run's
    // copy into its buffer, and the next run had none).
    #[cfg(feature = "diagnostics")]
    {
        let reports: Vec<Report> = list.deferred().after.iter().filter_map(|take| take()).collect();
        report_after(reports);
    }
    for (buffer, at, [row, rows, pitch], source, recorded) in &list.deferred().backs {
        let mut landed = false;
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer of ours, written by the list that has
        // run; the title's destination holds every row (checked when the
        // copy was recorded).
        unsafe {
            if buffer.Map(0, None, Some(&mut data)).is_ok() && !data.is_null() {
                let from = data.cast::<u8>().cast_const();
                for line in 0..*rows {
                    std::ptr::copy_nonoverlapping(from.add(line * row.next_multiple_of(256)), (*at as *mut u8).add(line * pitch), *row);
                }
                // The walls trace: what the title is handed (one row of at
                // most 24 values: small float targets).
                if walls_trace() && *rows == 1 && *row <= 96 {
                    walls_read_back(crate::frame::current(), *at, std::slice::from_raw_parts(from, *row));
                }
                buffer.Unmap(0, None);
                landed = true;
            }
        }
        let bytes = row * rows;
        walls_landed(READ_BACK, *at, pitch * (rows - 1) + row, bytes, if landed { bytes } else { 0 }, *source, Some(*recorded));
    }
}

/// The walls trace: the values a read-back of one row (`bytes`, at most 96)
/// hands the title at `at` in `frame`.
fn walls_read_back(frame: u64, at: usize, bytes: &[u8]) {
    let mut values = [0f32; 24];
    for (value, b) in values.iter_mut().zip(bytes.chunks_exact(4)) {
        *value = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    let len = bytes.len().min(96);
    walls_line(move || {
        tracing::info!(
            frame,
            to = format_args!("{at:#x}"),
            bytes = len,
            values = ?&values[..len / 4],
            "walls trace: a read-back lands in title memory"
        );
    });
}

/// The walls trace's kinds of what reaches title memory from the card.
pub(crate) const READ_BACK: &str = "read-back";
pub(crate) const CARD_COPY: &str = "card copy";
pub(crate) const HOST_COPY: &str = "host copy";

/// The walls trace: one line a frame of what the card wrote that reached
/// title memory in it, each summed per kind and destination. It lists
/// every read-back, with the title texture it copies and the frame it was
/// recorded in, and every card copy merged back and host copy shaders
/// changed (those compared and found unchanged only counted). It shows
/// when results a guest's CPU reads reach title memory, and against which
/// frame event of the same frame. Logging only.
pub(crate) fn walls_landed(kind: &'static str, at: usize, len: usize, bytes: usize, written: usize, source: usize, recorded: Option<u64>) {
    if !walls_trace() {
        return;
    }
    walls_landed_in(crate::frame::current(), kind, at, len, bytes, written, source, recorded);
}

/// [`walls_landed`] in `frame`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn walls_landed_in(
    frame: u64,
    kind: &'static str,
    at: usize,
    len: usize,
    bytes: usize,
    written: usize,
    source: usize,
    recorded: Option<u64>,
) {
    static LANDINGS: Mutex<Option<Landings>> = Mutex::new(None);
    let line = LANDINGS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(Landings::default)
        .add(frame, kind, at, len, bytes, written, source, recorded);
    // Taken with the lock, which is let go before the line goes to the
    // reporting thread: it runs where lists complete.
    if let Some((frame, count, unchanged, landed)) = line {
        walls_line(move || {
            tracing::info!(
                frame,
                count,
                unchanged,
                landed = ?landed,
                "walls trace: what the card wrote reached title memory in this frame"
            );
        });
    }
}

/// One frame's [`walls_landed`].
#[derive(Default)]
struct Landings {
    frame: u64,
    entries: std::collections::BTreeMap<(&'static str, usize), Landing>,
    /// Copies compared and found unchanged.
    unchanged: u64,
}

#[derive(Clone, Copy)]
struct Landing {
    len: usize,
    count: u32,
    bytes: u64,
    written: u64,
    source: usize,
    /// The earliest frame one was recorded in, for read-backs.
    recorded: Option<u64>,
}

impl Landings {
    /// Notes one in `frame`; if it is the first of a new frame, the line of
    /// the frame before: its number, its entries, how many were unchanged,
    /// and the entries (at most 256).
    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        frame: u64,
        kind: &'static str,
        at: usize,
        len: usize,
        bytes: usize,
        written: usize,
        source: usize,
        recorded: Option<u64>,
    ) -> Option<(u64, usize, u64, Vec<String>)> {
        let line = if self.frame == frame { None } else { self.take() };
        self.frame = frame;
        if written == 0 && kind != READ_BACK {
            self.unchanged += 1;
            return line;
        }
        let e = self.entries.entry((kind, at)).or_insert(Landing {
            len,
            count: 0,
            bytes: 0,
            written: 0,
            source,
            recorded,
        });
        e.len = e.len.max(len);
        e.count += 1;
        e.bytes += bytes as u64;
        e.written += written as u64;
        e.recorded = match (e.recorded, recorded) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        line
    }

    /// The line of the frame noted so far, if anything was noted.
    fn take(&mut self) -> Option<(u64, usize, u64, Vec<String>)> {
        if self.entries.is_empty() && self.unchanged == 0 {
            return None;
        }
        let landed = self
            .entries
            .iter()
            .take(256)
            .map(|((kind, at), l)| {
                let mut s = format!("{kind} {at:#x}+{:#x}: x{}, {} bytes, {} written", l.len, l.count, l.bytes, l.written);
                if l.source != 0 {
                    s += &format!(", from {:#x}", l.source);
                }
                if let Some(r) = l.recorded {
                    s += &format!(", recorded in frame {r}");
                }
                s
            })
            .collect();
        let line = (self.frame, self.entries.len(), self.unchanged, landed);
        self.entries.clear();
        self.unchanged = 0;
        Some(line)
    }
}

#[cfg(test)]
mod landings_tests {
    use super::{CARD_COPY, HOST_COPY, Landings, READ_BACK};

    /// A frame's line comes when the next frame's first landing does: read
    /// backs summed per destination with their texture and the earliest
    /// frame they were recorded in, a card copy merged back, and a host
    /// copy found unchanged only counted.
    #[test]
    fn a_frames_landings_are_one_line_once_the_next_frame_begins() {
        let mut l = Landings::default();
        assert!(l.add(5, READ_BACK, 0x1000, 0x200, 0x100, 0x100, 0xabc, Some(4)).is_none());
        assert!(l.add(5, READ_BACK, 0x1000, 0x200, 0x100, 0x100, 0xabc, Some(3)).is_none());
        assert!(l.add(5, CARD_COPY, 0x2000, 0x40, 0x40, 0x10, 0, None).is_none());
        assert!(l.add(5, HOST_COPY, 0x3000, 0x40, 0x40, 0, 0, None).is_none());
        let (frame, count, unchanged, landed) = l.add(6, CARD_COPY, 0x2000, 0x40, 0x40, 0x40, 0, None).expect("frame 5's line");
        assert_eq!((frame, count, unchanged), (5, 2, 1));
        assert_eq!(
            landed,
            [
                "card copy 0x2000+0x40: x1, 64 bytes, 16 written",
                "read-back 0x1000+0x200: x2, 512 bytes, 512 written, from 0xabc, recorded in frame 3",
            ]
        );
        // Frame 6 holds its own.
        assert_eq!(l.take().map(|t| (t.0, t.1)), Some((6, 1)));
        assert!(l.take().is_none());
    }
}

impl Host {
    /// Whether the clamps of indirect arguments read the title's copies
    /// ordered by enhanced barriers (`crate::indirect`'s comment): the
    /// device has them (`D3D12_FEATURE_D3D12_OPTIONS12`), and
    /// `VICTORY_ENHANCED_BARRIERS=0` does not turn them off. Without, each
    /// indirect call is clamped alone, its copies read in the state the
    /// title's commands left them. Asked when the first list is made;
    /// logged once.
    pub(crate) fn enhanced_barriers(&self) -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
            // SAFETY: the query writes its own structure of the given size.
            let supported = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS12,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            } && options.EnhancedBarriersSupported.as_bool();
            let on = supported && std::env::var_os("VICTORY_ENHANCED_BARRIERS").is_none_or(|v| v != "0");
            tracing::info!(
                supported,
                on,
                "enhanced barriers (VICTORY_ENHANCED_BARRIERS=0 turns them off): the clamps of indirect arguments                  read the title's copies ordered by them, or each call is clamped alone"
            );
            on
        })
    }

    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// Whether the host GPU supports the depth bounds test; logged once.
    pub(crate) fn depth_bounds(&self) -> bool {
        static BOUNDS: OnceLock<bool> = OnceLock::new();
        *BOUNDS.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
            // SAFETY: writes `options` of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS2,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let supported = ok && options.DepthBoundsTestSupported.as_bool();
            tracing::info!(supported, "host depth bounds support");
            supported
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        self.new_list_stamped(true)
    }

    /// A new list, open for recording, never stamped whatever
    /// `VICTORY_GPU_STAMPS` says: a list of ours whose stamps no send would
    /// read ([`Host::run_own_locked`]).
    #[cfg(feature = "diagnostics")]
    fn new_list_unstamped(&self) -> windows::core::Result<HostList> {
        self.new_list_stamped(false)
    }

    fn new_list_stamped(&self, stamped: bool) -> windows::core::Result<HostList> {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let before = private_bytes();
        let list = self.make_list(stamped);
        LISTS[0].fetch_add(1, Ordering::Relaxed);
        // (May shrink while a list is made, on another thread's account:
        // added as it is, the sum is what counts.)
        #[cfg(feature = "diagnostics")]
        LISTS[2].fetch_add(private_bytes().wrapping_sub(before), Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::HostLists, 1);
            add(C::HostListNs, started.elapsed().as_nanos() as u64);
        }
        list
    }

    /// An allocator and a closed list for a prelude.
    fn make_prelude(&self) -> Option<Prelude> {
        // SAFETY: plain object creation.
        let made: windows::core::Result<Prelude> = unsafe {
            (|| {
                let allocator: ID3D12CommandAllocator =
                    self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList =
                    self.device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                list.Close()?;
                Ok((allocator, list))
            })()
        };
        LISTS[1].fetch_add(1, Ordering::Relaxed);
        match made {
            Ok(prelude) => Some(prelude),
            Err(e) => {
                tracing::warn!(error = %e, "no host list for a prelude");
                None
            }
        }
    }

    /// A list; `stamped`: its opening stamped if the stamps are on.
    fn make_list(&self, stamped: bool) -> windows::core::Result<HostList> {
        #[cfg(not(feature = "diagnostics"))]
        let _ = stamped;
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude = OnceLock::new();
            if !lazy_prelude() {
                let _ = prelude.set(self.make_prelude());
            }
            let enhanced = if self.enhanced_barriers() { list.cast().ok() } else { None };
            // The list is open: its opening is stamped (`VICTORY_GPU_STAMPS=1`).
            #[cfg(feature = "diagnostics")]
            let flight_stamps = (stamped && crate::stamps::on()).then(|| {
                let mut stamps = crate::stamps::Stamps::default();
                stamps.open(&list);
                std::sync::Arc::new(Mutex::new(stamps))
            });
            Ok(HostList {
                allocator,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                twins: Mutex::default(),
                prelude,
                pipelines: Mutex::default(),
                budget: OnceLock::new(),
                indirect_args: Mutex::default(),
                batching: Mutex::default(),
                enhanced,
                #[cfg(feature = "diagnostics")]
                runs: Mutex::default(),
                _alive: Counted::new(),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
                #[cfg(feature = "diagnostics")]
                flight_stamps,
            })
        }
    }

    /// The submission lock (tests: after any test that has the queue alone
    /// is done, [`queue_alone`]).
    fn submission_lock(&self) -> Submission<'_> {
        let _p = ph(Ph::SubmissionLockWait);
        #[cfg(test)]
        {
            let shared = (!ALONE.with(std::cell::Cell::get)).then(|| QUEUE_TESTS.read().unwrap_or_else(|e| e.into_inner()));
            Submission { lock: self.submitted.lock().unwrap_or_else(|e| e.into_inner()), _shared: shared }
        }
        #[cfg(all(not(test), feature = "diagnostics"))]
        return crate::stalls::lock(&self.submitted);
        #[cfg(all(not(test), not(feature = "diagnostics")))]
        self.submitted.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The queue's clock: its ticks a second, and one moment on it and on
    /// the CPU's performance counter (`VICTORY_GPU_STAMPS=1`).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn clock_calibration(&self) -> windows::core::Result<crate::stamps::Calibration> {
        let (mut gpu, mut cpu) = (0, 0);
        // SAFETY: plain queries of our queue into our locals.
        unsafe {
            let frequency = self.queue.GetTimestampFrequency()?;
            self.queue.GetClockCalibration(&mut gpu, &mut cpu)?;
            Ok(crate::stamps::Calibration { frequency, gpu, cpu })
        }
    }

    /// The card's clock ([`CARD_CLOCK_US`]), made at the first send.
    fn card_clock(&self) -> Option<&'static CardClock> {
        static CLOCK: OnceLock<Option<CardClock>> = OnceLock::new();
        static WATCH: std::sync::Once = std::sync::Once::new();
        let clock = CLOCK
            .get_or_init(|| {
                // SAFETY: a plain fence on our device.
                let fence = unsafe { self.device.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) }.ok()?;
                Some(CardClock {
                    fence,
                    state: Mutex::new((0, None)),
                    sent: std::sync::Condvar::new(),
                })
            })
            .as_ref()?;
        WATCH.call_once(|| {
            if let Err(e) = std::thread::Builder::new().name("victory-card-clock".into()).spawn(move || card_clock_watch(clock)) {
                tracing::warn!(error = %e, "no thread for the card's clock: the frame line's queue-empty estimates stay 0");
            }
        });
        Some(clock)
    }

    /// Lists of ours just went to the queue: the clock's signal goes
    /// behind them, and an idle stretch of the card, if one was under way,
    /// ends here ([`CARD_CLOCK_US`]).
    fn card_sent(&self) {
        let Some(clock) = self.card_clock() else {
            return;
        };
        let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
        state.0 += 1;
        // SAFETY: our queue and the clock's fence.
        let _ = unsafe { self.queue.Signal(&clock.fence, state.0) };
        if let Some((since, worked_then)) = state.1.take() {
            CARD_CLOCK_US[0].fetch_add(since.elapsed().as_micros() as u64, Ordering::Relaxed);
            CARD_CLOCK_US[1].fetch_add(worked().saturating_sub(worked_then), Ordering::Relaxed);
        }
        drop(state);
        clock.sent.notify_one();
    }

    /// Whether the host has what copies in the card's memory need (three
    /// lists and the compare); made at the first ask. A copy goes to the
    /// card's memory only if it has.
    pub(crate) fn card_ready(&self) -> bool {
        static READY: OnceLock<bool> = OnceLock::new();
        *READY.get_or_init(|| match self.make_card_work() {
            Ok(work) => {
                *CARD_WORK.lock().unwrap_or_else(|e| e.into_inner()) = Some(work);
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "no host copies in the card's memory: the compare could not be made");
                false
            }
        })
    }

    fn make_card_work(&self) -> windows::core::Result<CardWork> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let mut code: Option<ID3DBlob> = None;
        let program = hash::program();
        // SAFETY: the source and names live through the call; the out
        // pointer is ours.
        unsafe {
            D3DCompile(
                program.as_ptr().cast(),
                program.len(),
                windows::core::s!("card compare"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        let list = || self.make_prelude().ok_or_else(windows::core::Error::empty);
        // A megabyte of results: 262,144 pieces, 16 GB of declared ranges
        // in one submission.
        const RESULTS: u64 = 1 << 20;
        let (differs, data) = self.shared_buffer(RESULTS).ok_or_else(windows::core::Error::empty)?;
        name(&differs, || "card compare results".into());
        // SAFETY: the blob's own bytes; the root signature is inside the
        // shader.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
            let pso: ID3D12PipelineState = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(CardWork {
                sync: vec![list()?],
                check: vec![list()?],
                fetch: list()?,
                root,
                pso,
                differs: (differs, data as usize, RESULTS),
                fetched: None,
                taken: 0,
                out: 0,
                #[cfg(feature = "diagnostics")]
                stamps: Default::default(),
            })
        }
    }

    /// Before a submission's lists run, for copies in the card's memory
    /// ([`crate::shadow::cards`]): what changed on their CPU side since
    /// the last submission goes to their card side (a list of ours, run
    /// now; the submission's lists follow it on the queue), and the ranges
    /// this submission's shaders may write get a compare to run after its
    /// lists. `None` if there is nothing to do.
    ///
    /// `flown`: the submission is sent ahead with [`relay`]. Its compare
    /// then stores the hashes it makes (see [`flagged_pieces`] and
    /// `relay`). `target`: the value the host's fence reaches once the
    /// submission has run (diagnostics: the plan's stamps are read then);
    /// `None` if the submission signals nothing after its lists
    /// (`VICTORY_GPU_TIMING` without a title list, [`send_signals`]), and
    /// then the plan is not stamped.
    fn card_before(&self, runs: &[&HostList], flown: bool, target: Option<u64>) -> Option<CardPlan> {
        use std::sync::Arc;
        #[cfg(not(feature = "diagnostics"))]
        let _ = target;
        if !crate::shadow::cards() {
            return None;
        }
        // With `VICTORY_GPU_STAMPS=1`: pairs around the hash list and the
        // compare, sent (noted) before the first of them goes to the queue,
        // if a signal follows them.
        #[cfg(feature = "diagnostics")]
        let target = target.filter(|_| crate::stamps::on());
        #[cfg(feature = "diagnostics")]
        let mut stamps = target.map(|_| crate::stamps::Stamps::default());
        // The ranges this submission's shaders may write, per copy.
        let mut declared: Vec<(Arc<crate::shadow::Shadow>, Vec<(usize, usize)>)> = Vec::new();
        for list in runs {
            for step in &list.deferred().copies {
                if let Step::GpuWrite(shadow, address, len) = step
                    && shadow.carded()
                {
                    let range = (*address, *address + *len);
                    match declared.iter_mut().find(|(s, _)| Arc::ptr_eq(s, shadow)) {
                        Some((_, ranges)) => {
                            if !ranges.contains(&range) {
                                ranges.push(range);
                            }
                        }
                        None => declared.push((shadow.clone(), vec![range])),
                    }
                }
            }
        }
        // With the relay: the order of the queue is held from here until
        // what is taken is on the queue, and the taking is numbered.
        let mut order = relay().then(|| QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        let sent_as = match order.as_mut() {
            Some(order) => {
                **order += 1;
                **order
            }
            None => 0,
        };
        let mut dirty = crate::shadow::card_dirty(sent_as);
        if declared.is_empty() && dirty.is_empty() {
            // No plan, but it may still write copies by address or by a heap index.
            self.fetch_hold_only(fetch::reach(runs));
            return None;
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let work = work.as_mut()?;
        // Before anything of it goes to the queue.
        let ticket = self.fetch_ticket(flown, work.out == 0, sent_as, &declared, &dirty, fetch::reach(runs));
        // A copy whose shader writes are declared for the first time gets
        // its bytes from before, whole.
        let mut whole: Vec<Arc<crate::shadow::Shadow>> = Vec::new();
        for (shadow, _) in &declared {
            if shadow.card_make_before(|size| {
                let before = self.uav_buffer(crate::shadow::card_hash_bytes(size))?;
                name(&before, || "title buffer on the card, hashes of its pieces".into());
                Some(before)
            }) {
                whole.push(shadow.clone());
                if !dirty.iter().any(|(s, _)| Arc::ptr_eq(s, shadow)) {
                    dirty.push((shadow.clone(), Vec::new()));
                }
            }
        }
        let mut used: Vec<(bool, Prelude)> = Vec::new();
        if !dirty.is_empty() {
            let Some(sync) = work.sync.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!("no host list to bring host copies in the card's memory up to date"));
            };
            let (allocator, list) = &sync;
            // SAFETY: the list's last run finished (it ran ahead of a
            // submission that was waited for, and came back after that);
            // live buffers, and the ranges lie in them.
            let recorded = unsafe {
                (|| -> windows::core::Result<()> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    crate::events::begin(list, crate::events::HASH);
                    #[cfg(feature = "diagnostics")]
                    let begun = stamps.as_mut().and_then(|s| s.begin(list));
                    for (shadow, ranges) in &dirty {
                        let Some((staging, before, size)) = shadow.card_sides() else {
                            continue;
                        };
                        for &(a, b) in ranges {
                            list.CopyBufferRegion(&shadow.buffer, a as u64, staging, a as u64, (b - a) as u64);
                            crate::shadow::CARD_STATS[0].fetch_add((b - a) as u64, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            shadow.census(crate::memclass::Count::Sent, a, b);
                        }
                        crate::shadow::CARD_STATS[1].fetch_add(ranges.len() as u64, Ordering::Relaxed);
                        let Some(before) = before else {
                            continue;
                        };
                        // No twin (it cost the card the copy's size again,
                        // hundreds of megabytes): `before` holds a 128-bit hash
                        // of each 64 KiB piece. The pieces just sent up are
                        // hashed again, all of them the first time; the
                        // compare after the lists hashes the declared
                        // pieces and flags those whose hash is another.
                        if !ranges.is_empty() {
                            list.ResourceBarrier(&[transition(
                                &shadow.buffer,
                                D3D12_RESOURCE_STATE_COPY_DEST,
                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                            )]);
                        }
                        list.SetComputeRootSignature(&work.root);
                        list.SetPipelineState(&work.pso);
                        list.SetComputeRootShaderResourceView(1, shadow.buffer.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                        let runs: Vec<crate::shadow::CardSpan> = if whole.iter().any(|s| Arc::ptr_eq(s, shadow)) {
                            vec![shadow.card_run(0, usize::MAX)]
                        } else {
                            ranges.iter().map(|&(a, b)| shadow.card_run(a, b)).collect()
                        };
                        // The same pieces in fewer passes (`rehash_runs`).
                        let runs = if rehash_runs() { crate::shadow::joined_piece_runs(runs) } else { runs };
                        CARD_PASSES[0].fetch_add(runs.len() as u64, Ordering::Relaxed);
                        CARD_PASSES[1].fetch_add(runs.iter().map(|r| u64::from(r.pieces)).sum::<u64>(), Ordering::Relaxed);
                        for run in runs {
                            // Hashed on the card (in the same list as the
                            // sends, and in its stamp; the bytes apart).
                            #[cfg(feature = "diagnostics")]
                            shadow.census(crate::memclass::Count::ComparedCard, run.from as usize, run.to as usize);
                            for (offset, groups) in hash::dispatch_parts(run.pieces) {
                                let constants = [run.first + offset, run.skew, size as u32, run.pieces - offset, 0, 0];
                                list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                                list.Dispatch(groups, 1, 1);
                            }
                        }
                    }
                    crate::events::end(list);
                    #[cfg(feature = "diagnostics")]
                    if let Some(s) = stamps.as_mut() {
                        s.end(crate::stamps::Kind::Hash, begun, list, true);
                        s.resolve(list);
                    }
                    list.Close()?;
                    let lists = [Some(list.cast::<ID3D12CommandList>()?)];
                    #[cfg(feature = "diagnostics")]
                    if let (Some(s), Some(target)) = (stamps.as_mut(), target) {
                        s.sent(&self.fence, target);
                    }
                    self.queue.ExecuteCommandLists(&lists);
                    Ok(())
                })()
            };
            if let Err(e) = recorded {
                fail(format_args!("host copies in the card's memory could not be brought up to date ({e})"));
            }
            used.push((false, sync));
        }
        drop(order);
        let mut plan = CardPlan {
            ranges: Vec::new(),
            check: None,
            _synced: dirty.into_iter().map(|(shadow, _)| shadow).collect(),
            lists: used,
            #[cfg(feature = "diagnostics")]
            stamps: None,
            fetch: ticket,
            hash_check: !declared.is_empty() && hash_check::sample(),
        };
        // Each byte once, as the copy-back joins the declared ranges. The
        // results go behind those of plans still in the air.
        let mut at = work.taken;
        for (shadow, ranges) in &declared {
            let mut sorted = ranges.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            let twin = shadow.card_sides().is_some_and(|(_, before, _)| before.is_some());
            for (from, to) in spans {
                let Some(span) = shadow.card_span(from, to) else {
                    continue;
                };
                let compared = twin && u64::from(at + span.pieces) * 4 <= work.differs.2;
                // Compared on the card after the lists.
                #[cfg(feature = "diagnostics")]
                if compared {
                    shadow.census(crate::memclass::Count::ComparedCard, span.from as usize, span.to as usize);
                }
                plan.ranges.push(CardRange {
                    shadow: shadow.clone(),
                    declared: (from, to),
                    span,
                    at,
                    compared,
                });
                if compared {
                    at += span.pieces;
                }
            }
        }
        work.taken = at;
        if plan.ranges.iter().any(|r| r.compared) {
            let Some(check) = work.check.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!("no host list for the compare of host copies in the card's memory"));
            };
            let (allocator, list) = &check;
            // SAFETY: the list's last run finished (its submission was
            // waited for, and it came back after that); live buffers; the
            // compare's own root signature.
            let recorded = unsafe {
                (|| -> windows::core::Result<ID3D12CommandList> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    crate::events::begin(list, crate::events::COMPARE);
                    #[cfg(feature = "diagnostics")]
                    let begun = stamps.as_mut().and_then(|s| s.begin(list));
                    // What the submission's shaders wrote, first.
                    list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                pResource: std::mem::ManuallyDrop::new(None),
                            }),
                        },
                    }]);
                    list.SetComputeRootSignature(&work.root);
                    list.SetPipelineState(&work.pso);
                    list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                    for r in plan.ranges.iter().filter(|r| r.compared) {
                        let Some((_, Some(before), size)) = r.shadow.card_sides() else {
                            continue;
                        };
                        for (offset, groups) in hash::dispatch_parts(r.span.pieces) {
                            let constants = [r.span.first + offset, r.span.skew, size as u32, r.span.pieces - offset, r.at * 4 + offset * 4, if flown { 2 } else { 1 }];
                            list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                            list.SetComputeRootShaderResourceView(1, r.shadow.buffer.GetGPUVirtualAddress());
                            list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                            list.Dispatch(groups, 1, 1);
                        }
                        CARD_PASSES[2].fetch_add(1, Ordering::Relaxed);
                    }
                    crate::events::end(list);
                    #[cfg(feature = "diagnostics")]
                    if let Some(s) = stamps.as_mut() {
                        s.end(crate::stamps::Kind::Compare, begun, list, true);
                        s.resolve(list);
                    }
                    list.Close()?;
                    list.cast()
                })()
            };
            match recorded {
                Ok(list) => plan.check = Some(list),
                Err(e) => fail(format_args!("the compare of host copies in the card's memory could not be recorded ({e})")),
            }
            plan.lists.push((true, check));
        }
        work.out += 1;
        // The compare goes with the submission's lists: noted now if the
        // hash list did not go first.
        #[cfg(feature = "diagnostics")]
        if let (Some(mut s), Some(target)) = (stamps, target) {
            if s.unsent() {
                s.sent(&self.fence, target);
            }
            plan.stamps = Some(Box::new(s));
        }
        Some(plan)
    }

    /// With [`relay`]: whether the compare's result area is half taken by
    /// plans in the air. It starts from the beginning only when no plan is
    /// out, so the caller lets the flights land first.
    fn card_results_low(&self) -> bool {
        CARD_WORK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|work| u64::from(work.taken) * 8 > work.differs.2)
    }

    /// The plan's submission has run and its results were read
    /// ([`Host::card_back`]): its lists are free for the next plan. Called
    /// when nothing of ours is on the card, so the compare's results start
    /// from the beginning again.
    fn card_done(&self, plan: CardPlan) {
        // Its hash list and compare have run: their pairs are read
        // (`VICTORY_GPU_STAMPS=1`; nothing if their fence says otherwise).
        #[cfg(feature = "diagnostics")]
        let plan = {
            let mut plan = plan;
            if let Some(stamps) = plan.stamps.as_mut() {
                stamps.landed();
            }
            plan
        };
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        for (check, list) in plan.lists {
            if check { work.check.push(list) } else { work.sync.push(list) }
        }
        work.out = work.out.saturating_sub(1);
        // With the relay other plans may be on the card, their results
        // not read yet: the area starts again only when none is out.
        if !relay() || work.out == 0 {
            work.taken = 0;
        }
    }

    /// The relay's own list, fence, event for [`Host::relay_card_back`].
    fn make_relay_fetch(&self) -> Option<RelayFetch> {
        let list = self.make_prelude()?;
        // SAFETY: plain objects of our own.
        unsafe {
            let fence: ID3D12Fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            Some(RelayFetch {
                list,
                fence,
                event,
                sent: 0,
                buffer: None,
                #[cfg(feature = "diagnostics")]
                stamps: crate::stamps::Stamps::default(),
            })
        }
    }

    /// The relay's fetch on the drawing queue, in the queue's order: the
    /// sends of title bytes so far are ahead of it, every later one behind
    /// it (the count of them comes back). With the copy queue's tickets
    /// out (`fetch::engaged`), every ticket is released, the list sent and
    /// its fence signalled all under `QUEUE_ORDER`, so no submission can put
    /// a hold for a newer ticket between them (Sol's second review, 1).
    fn send_drawing_fetch(&self, lists: &[Option<ID3D12CommandList>], fence: &ID3D12Fence, sent: u64) -> windows::core::Result<u64> {
        let order = QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner());
        if fetch::engaged() {
            fetch::before_drawing_fetch();
            // SAFETY: a closed list of ours; the relay's fence.
            unsafe {
                self.queue.ExecuteCommandLists(lists);
                self.queue.Signal(fence, sent)?;
            }
            let after = *order;
            drop(order);
            self.card_sent();
            return Ok(after);
        }
        // SAFETY: as above.
        unsafe { self.queue.ExecuteCommandLists(lists) };
        let after = *order;
        drop(order);
        self.card_sent();
        // SAFETY: as above.
        unsafe { self.queue.Signal(fence, sent)? };
        Ok(after)
    }

    /// After a submission ran, for copies in the card's memory: the pieces
    /// of the declared ranges that the compare found differing (all of a
    /// range it did not compare) are fetched, with one more wait for the
    /// card, and what shaders wrote in them goes into title memory.
    /// `submitted` is the last value signalled on the fence. `seen` holds
    /// the pieces fetched already since the card was last waited for
    /// (several submissions settled together name the same ones).
    fn card_back(
        &self,
        plan: &CardPlan,
        submitted: &mut u64,
        seen: &mut std::collections::HashSet<(usize, usize, usize)>,
    ) {
        if plan.ranges.is_empty() {
            return;
        }
        // The landing's merge mode, read once here and kept to its last piece
        // (`VICTORY_EXACT_BACK`, live).
        let exact = crate::host::land::exact();
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        // SAFETY: the mapping of the results, which the compare (it has
        // run) wrote.
        let differs = unsafe { std::slice::from_raw_parts(work.differs.1 as *const u32, (work.differs.2 / 4) as usize) };
        // The pieces to fetch: the range, the offsets in its copy, and
        // where the bytes come down.
        let mut pieces: Vec<(usize, usize, usize, u64)> = Vec::new();
        let mut total = 0u64;
        for (i, r) in plan.ranges.iter().enumerate() {
            for p in 0..r.span.pieces {
                if r.compared && differs[(r.at + p) as usize] == 0 {
                    continue;
                }
                let (lo, hi) = r.shadow.card_piece(&r.span, p);
                if lo < hi && seen.insert((std::sync::Arc::as_ptr(&r.shadow) as usize, lo, hi)) {
                    pieces.push((i, lo, hi, total));
                    total += (hi - lo) as u64;
                }
            }
            crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
        }
        let mut written = vec![0usize; plan.ranges.len()];
        if !pieces.is_empty() {
            if work.fetched.as_ref().is_none_or(|f| f.2 < total) {
                let size = total.next_multiple_of(16 << 20);
                work.fetched = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
            }
            let Some((fetched, data, _)) = &work.fetched else {
                fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes)"));
            };
            let waiting = std::time::Instant::now();
            let (allocator, list) = &work.fetch;
            #[cfg(feature = "diagnostics")]
            let mut stamps = crate::stamps::on().then_some(&mut work.stamps);
            // SAFETY: the list's last run was waited for; live buffers; the
            // fence and event are ours, under the submission lock.
            let ran = unsafe {
                (|| -> windows::core::Result<bool> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    crate::events::begin(list, crate::events::FETCH);
                    // A pair around the copies (`VICTORY_GPU_STAMPS=1`).
                    #[cfg(feature = "diagnostics")]
                    let begun = stamps.as_deref_mut().and_then(|s| {
                        s.restart();
                        s.begin(list)
                    });
                    for &(i, lo, hi, at) in &pieces {
                        list.CopyBufferRegion(fetched, at, &plan.ranges[i].shadow.buffer, lo as u64, (hi - lo) as u64);
                    }
                    crate::events::end(list);
                    #[cfg(feature = "diagnostics")]
                    if let Some(s) = stamps.as_deref_mut() {
                        s.end(crate::stamps::Kind::Fetch, begun, list, true);
                        s.resolve(list);
                    }
                    list.Close()?;
                    let lists = [Some(list.cast::<ID3D12CommandList>()?)];
                    #[cfg(feature = "diagnostics")]
                    if let Some(s) = stamps.as_deref_mut() {
                        s.sent(&self.fence, *submitted + 1);
                    }
                    self.queue.ExecuteCommandLists(&lists);
                    self.card_sent();
                    *submitted += 1;
                    self.queue.Signal(&self.fence, *submitted)?;
                    let was_working = working(false);
                    // Until the fence has the value on a live device, not
                    // until the event wakes (Sol's third read of the frame
                    // timers: a removal or an early wake merged as done).
                    #[cfg(feature = "diagnostics")]
                    let fetching = Instant::now();
                    let reached = fetch_waited(&self.device, &self.fence, self.event, *submitted);
                    #[cfg(feature = "diagnostics")]
                    crate::stalls::card_waited(crate::stalls::Site::CardBack, fetching);
                    stood(waiting);
                    working(was_working);
                    reached
                })()
            };
            match ran {
                Ok(true) => {}
                Ok(false) => fail(format_args!("host GPU device removed during a fetch of what shaders wrote to the card's memory")),
                Err(e) => fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e})")),
            }
            // Its pair is read if its fence has reached its value on a live
            // device (the stamps look; the wait above does not).
            #[cfg(feature = "diagnostics")]
            if crate::stamps::on() {
                work.stamps.landed();
            }
            {
                use crate::shadow::CARD_STATS;
                CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
                CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
                CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
                CARD_STATS[6].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
                #[cfg(feature = "diagnostics")]
                for &(i, lo, hi, _) in &pieces {
                    plan.ranges[i].shadow.census(crate::memclass::Count::Fetched, lo, hi);
                }
            }
            if wide_merge() && pieces.len() >= 8 {
                // The pieces are apart from one another (each its own bytes
                // of its copy and of title memory): several at a time.
                let fetched = *data;
                let counts: std::sync::Arc<Vec<std::sync::atomic::AtomicUsize>> =
                    std::sync::Arc::new((0..plan.ranges.len()).map(|_| std::sync::atomic::AtomicUsize::new(0)).collect());
                let each = pieces.len().div_ceil(8);
                let jobs: Vec<Box<dyn FnOnce() + Send>> = pieces
                    .chunks(each)
                    .map(|chunk| {
                        let chunk: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, usize, u64)> =
                            chunk.iter().map(|&(i, lo, hi, at)| (plan.ranges[i].shadow.clone(), i, lo, hi, at)).collect();
                        let counts = counts.clone();
                        Box::new(move || {
                            let mut title = Vec::new();
                            for (shadow, i, lo, hi, at) in chunk {
                                // SAFETY: the mapping of the fetched bytes;
                                // the copy into it has run, and the caller
                                // waits for this job.
                                let gpu = unsafe { std::slice::from_raw_parts((fetched + at as usize) as *const u8, hi - lo) };
                                let merged = shadow.card_merge(lo, gpu, &mut title, exact);
                                shadow.walls_merged(gpu.len(), merged);
                                counts[i].fetch_add(merged, Ordering::Relaxed);
                            }
                        }) as Box<dyn FnOnce() + Send>
                    })
                    .collect();
                crate::shadow::wide_run(jobs);
                for (written, count) in written.iter_mut().zip(counts.iter()) {
                    *written += count.load(Ordering::Relaxed);
                }
            } else {
                let mut title = Vec::new();
                for &(i, lo, hi, at) in &pieces {
                    // SAFETY: the mapping of the fetched bytes, `total` of
                    // them; the copy into it has run.
                    let gpu = unsafe { std::slice::from_raw_parts((*data + at as usize) as *const u8, hi - lo) };
                    let merged = plan.ranges[i].shadow.card_merge(lo, gpu, &mut title, exact);
                    plan.ranges[i].shadow.walls_merged(gpu.len(), merged);
                    written[i] += merged;
                }
            }
        }
        hash_check::look(self, &[plan], differs, false);
        for (r, written) in plan.ranges.iter().zip(written) {
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            let (from, to) = r.declared;
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(from, to - from, written);
            let _ = (from, to);
        }
    }

    /// Runs closed lists and waits for them, and with [`late`] for
    /// everything sent to the card before them, which is settled first.
    ///
    /// A submission that names a host list twice (the title may name a
    /// list twice; `crate::queue` and `crate::list::detach` pass that on) is
    /// carried out as submissions of its own, in order: cut before each
    /// list the piece so far holds already, each piece prepared (its
    /// steps, its lists' preludes), sent and waited for before the next is
    /// prepared. What a host list keeps for one execution (its prelude
    /// list, its budget, its scratch buffer) is then never made again while
    /// an execution of it is pending, and D3D12 runs a list again only
    /// once its last execution is done.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        if !names_twice(runs) {
            let _ = self.submit(runs, false);
            return;
        }
        named_twice(runs.len());
        let mut from = 0;
        for i in 1..=runs.len() {
            if i == runs.len() || runs[from..i].iter().any(|r| std::ptr::eq(*r, runs[i])) {
                let _ = self.submit(&runs[from..i], false);
                from = i;
            }
        }
    }

    /// A queue's submission with [`late`]: the host lists `lists`, taken
    /// from the title's, go to the card in `order` and are not waited for.
    /// They join the line of flights and are settled, with the others in
    /// the air, at the next point where what they do can be observed
    /// ([`late`] lists those points); their host lists are reset and kept
    /// for reuse then, as a queue does for the lists it ran.
    ///
    /// One that names a host list twice is not sent ahead: it takes the
    /// waited path ([`Host::execute`], as with `VICTORY_LATE_WAIT=0`).
    pub(crate) fn execute_late(&self, lists: Vec<crate::list::Detached>, order: Vec<usize>) {
        if order.is_empty() {
            crate::list::recycle(lists, &order);
            return;
        }
        if names_twice_in(&order) {
            {
                let runs: Vec<&HostList> = order.iter().map(|&i| lists[i].host()).collect();
                self.execute(&runs);
            }
            crate::list::recycle(lists, &order);
            return;
        }
        late_threads();
        let _call = ph(Ph::Unaccounted);
        let sent = {
            let runs: Vec<&HostList> = order.iter().map(|&i| lists[i].host()).collect();
            self.submit(&runs, true)
        };
        let Some((mut submitted, sent)) = sent else {
            // (Not reached: a submission sent ahead always comes back.)
            crate::list::recycle(lists, &order);
            return;
        };
        let told = !sent.writes.is_empty();
        let most = {
            let mut line = line();
            let Line { line: items, written, lists: flying, target, .. } = &mut *line;
            join_ranges(written, sent.back.iter().flat_map(|(_, ranges)| ranges.iter().copied()));
            *flying += order.len();
            *target = *submitted;
            items.push(Late::Run(Flight {
                number: sent.number,
                fence: *target,
                lists,
                order,
                card: sent.card,
                back: sent.back,
                #[cfg(feature = "diagnostics")]
                own: None,
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
            *flying >= late_most()
        };
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        RELAY_MOVED.notify_one();
        if told && !late_writes() {
            self.settle(&mut submitted, WHY_WRITES);
        } else if most && relay_exact() {
            // Until there are fewer, not until there are none.
            relay_wait(WHY_MOST, |line| line.lists < late_most());
        } else if most {
            self.settle(&mut submitted, WHY_MOST);
        } else {
            LINE_MOVED.notify_one();
        }
        drop(submitted);
        land();
    }

    /// Sends `list`, a closed list of ours (the present's copy), as a
    /// flight ([`late_free_present`]): not waited for, settled with the
    /// others. The caller keeps the list and does not reset it before the
    /// submission whose number comes back has finished ([`FINISHED`]); 0 if
    /// it ran and was waited for after all.
    /// Diagnostics: records a list of ours with `record`, runs it and
    /// waits for it, holding the submission lock throughout. A host
    /// texture's tracked state (`HostTexture::state`) changes only in a
    /// submission's prelude, under that lock: what `record` reads of it
    /// holds until this list has run, so barriers made from it are valid.
    /// Lists sent ahead ([`late`]) made their preludes when they were sent,
    /// and run before this one in the queue's order. False if no list could
    /// be made.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn run_own_locked(&self, record: impl FnOnce(&HostList)) -> bool {
        let mut submitted = self.submission_lock();
        let Ok(list) = self.new_list_unstamped() else {
            return false;
        };
        record(&list);
        list.close();
        *submitted += 1;
        // SAFETY: a closed list of ours; the fence is ours, under the
        // submission lock.
        unsafe {
            self.queue.ExecuteCommandLists(&[list.list.cast().ok()]);
            // As a submission does: the card is not idle while this runs
            // (Sol's second re-check, new fault 8).
            self.card_sent();
            if let Err(e) = self.queue.Signal(&self.fence, *submitted) {
                fail(format_args!("a list of ours could not be followed by a signal ({e})"));
            }
        }
        #[cfg(feature = "diagnostics")]
        let waiting = Instant::now();
        if !wait_fence(&self.device, &self.fence, Some(self.event), *submitted) {
            fail(format_args!("host GPU device removed during a list of ours{}", last_notes(&[&list])));
        }
        #[cfg(feature = "diagnostics")]
        crate::stalls::card_waited(crate::stalls::Site::OwnLocked, waiting);
        true
    }

    fn execute_own(&self, list: &HostList) -> u64 {
        late_threads();
        let Some((submitted, sent)) = self.submit(&[list], true) else {
            return 0;
        };
        let number = sent.number;
        {
            let mut line = line();
            let Line { line: items, written, lists: flying, target, .. } = &mut *line;
            join_ranges(written, sent.back.iter().flat_map(|(_, ranges)| ranges.iter().copied()));
            *flying += 1;
            *target = *submitted;
            items.push(Late::Run(Flight {
                number,
                fence: *target,
                lists: Vec::new(),
                order: Vec::new(),
                card: sent.card,
                back: sent.back,
                // Its stamps are read when it lands (Sol's review of the
                // frame timers: they were never collected).
                #[cfg(feature = "diagnostics")]
                own: list.stamps_handle(),
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
        }
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        LINE_MOVED.notify_one();
        RELAY_MOVED.notify_one();
        drop(submitted);
        number
    }

    /// Waits for the card to run every flight in the air ([`late`]) and
    /// does what they left to do, as [`Host::execute`] does for one
    /// submission after its wait: the pieces their shaders wrote in copies
    /// in the card's memory are fetched and merged (each once), the ranges
    /// they declared in the other copies are compared and copied back
    /// (each byte once, whichever flights declared it), and then, in the
    /// order they were given: their bottom-of-pipe writes and the fence
    /// signals behind them. `submitted` is the submission lock's value;
    /// `why` one of the `WHY_` places. Nothing happens if nothing flies.
    ///
    /// Everything in the air is settled, never the first flights alone:
    /// the merge writes the copies' CPU side, which a flight still on the
    /// card may be writing too.
    fn settle(&self, submitted: &mut u64, why: usize) {
        if !late() {
            return;
        }
        if relay() {
            // Nothing is settled here: the thread that lands flights does
            // it, in order, and this waits until nothing is in the air and
            // nothing is being landed. Every point that settled before
            // comes here, so each still sees title memory, fences and
            // bottom-of-pipe values as after a settle.
            relay_wait(why, |line| line.line.is_empty() && !line.settling);
            return;
        }
        let _p = ph(Ph::Settle);
        let was_working = working(true);
        let mut counted = false;
        // Its waits for the fence, over every batch it settles, if any.
        #[cfg(feature = "diagnostics")]
        let mut fenced: Option<u64> = None;
        loop {
            let items = {
                let mut line = line();
                if line.line.is_empty() {
                    line.settling = false;
                    break;
                }
                line.settling = true;
                line.written.clear();
                line.lists = 0;
                std::mem::take(&mut line.line)
            };
            let flights = items.iter().filter(|i| matches!(i, Late::Run(_))).count();
            if !counted {
                counted = true;
                LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
                LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
            }
            LATE_STATS[11].fetch_add(flights as u64, Ordering::Relaxed);
            if flights > 0 {
                submitting(2, flights);
                let waiting = std::time::Instant::now();
                // `submitted` is the last value signalled, under the
                // submission lock.
                working(false);
                let reached = wait_fence(&self.device, &self.fence, Some(self.event), *submitted);
                stood(waiting);
                working(true);
                let gpu = waiting.elapsed();
                // Released by its own wait for the fence.
                #[cfg(feature = "diagnostics")]
                {
                    crate::stalls::card_waited(crate::stalls::Site::Settle, waiting);
                    *fenced.get_or_insert(0) += gpu.as_micros() as u64;
                }
                LATE_STATS[12].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                LATE_WAITED[why.min(10)].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                SUBMISSION_US[2].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                #[cfg(feature = "diagnostics")]
                crate::copystat::host_waited(gpu.as_micros() as u64, || {
                    format!("{flights} submissions sent ahead, settled together")
                });
                // SAFETY: a plain query.
                if !reached || unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = items
                        .iter()
                        .filter_map(|i| match i {
                            Late::Run(f) => Some(f),
                            _ => None,
                        })
                        .flat_map(|f| f.order.iter().map(|&i| f.lists[i].host()))
                        .collect();
                    fail(format_args!(
                        "host GPU device removed after {flights} submissions sent ahead (VICTORY_LATE_WAIT=1) of {} lists{}{}",
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                submitting(3, flights);
                let _back = ph(Ph::CopyBack);
                let copying = std::time::Instant::now();
                let mut seen = std::collections::HashSet::new();
                let mut back: Vec<Declared> = Vec::new();
                for item in &items {
                    let Late::Run(flight) = item else {
                        continue;
                    };
                    if let Some(card) = &flight.card {
                        self.card_back(card, submitted, &mut seen);
                    }
                    for (shadow, ranges) in &flight.back {
                        match back.iter_mut().find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow)) {
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
                hand_back(&back);
                COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            // In the order given: what tells the title the card is done.
            let mut last = 0;
            let mut landed = Vec::new();
            for item in items {
                match item {
                    Late::Run(flight) => {
                        last = flight.number;
                        for &i in &flight.order {
                            read_backs(flight.lists[i].host());
                        }
                        #[cfg(feature = "diagnostics")]
                        flight_stamps_landed(&flight);
                        if let Some(card) = flight.card {
                            self.card_done(card);
                        }
                        landed.push((flight.lists, flight.order));
                    }
                    Late::Write(at, what, bytes) => what.write(at as *mut u8, bytes),
                    Late::Signal(fence, value) => crate::fence::set(fence as crate::object::This, value),
                }
            }
            if last > 0 {
                FINISHED.store(last, Ordering::Release);
                bury(last);
            }
            line().landed.extend(landed);
        }
        if counted {
            // Once a settle: released by its own fence waits, or with no
            // flight to wait for (signals and writes alone).
            #[cfg(feature = "diagnostics")]
            match fenced {
                Some(us) => crate::stalls::settled(why, crate::stalls::Release::Fence, us),
                None => crate::stalls::settled(why, crate::stalls::Release::Nothing, 0),
            }
            self.log_messages();
            submitting(0, 0);
            // What was settled may be what a queue stands at, and the
            // caller may not be one that looks (a read-back, a present).
            if why != WHY_CARD_DONE {
                line().poke = true;
                LINE_MOVED.notify_one();
            }
        }
        working(was_working);
    }

    /// [`Host::execute`], or with `late` the first half of
    /// [`Host::execute_late`]: the lists are prepared and sent, and what is
    /// to be done once they have run comes back with the submission lock
    /// still held.
    fn submit<'a>(&'a self, runs: &[&HostList], late: bool) -> Option<(Submission<'a>, Sent)> {
        // `VICTORY_AVOID_CPUS`: the thread that first submits is kept off
        // the listed processors once (a thread-local hook, not per submit).
        crate::cpu::once_before_submit();
        // From the call on each moment is in one phase of this thread's
        // budget (`crate::budget`): the lock's acquisition, the settle, the
        // sweep, the waiting slots, and so on; what no named phase covers,
        // the entry's accounting and the bookkeeping between them, is
        // `unaccounted` (Sol's review of the frame timers: the parts did not
        // add up to the one figure they replaced).
        let _call = ph(Ph::Unaccounted);
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_begins();
        #[cfg(test)]
        test_hook(HOOK_ENTRY);
        #[cfg(test)]
        test_hook(HOOK_LOCK);
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = self.submission_lock();
        let was_working = working(true);
        submitting(1, runs.len());
        // Flights in the air ([`late`]): a submission that is waited for
        // settles them first, one sent ahead only if preparing it reads
        // what they write.
        let blocked = if late { late_blocked(runs) } else { None };
        #[cfg(test)]
        test_hook(HOOK_OTHER);
        {
            let _p = ph(Ph::Settle);
            if !late {
                #[cfg(test)]
                test_hook(HOOK_SETTLE);
                self.settle(&mut submitted, WHY_SYNC);
            } else if let Some(why) = blocked {
                if relay_exact() && why == WHY_READ {
                    // Only until the flights that may write what is read have
                    // landed: their copy-back's end, not the card's other work.
                    relay_wait(why, |line| blocked_by(&line.written, runs, false).is_none());
                } else {
                    self.settle(&mut submitted, why);
                }
            }
        }
        let behind = late && flying();
        // Before this submission counts as started: what the sweep lets go
        // of is named by no list, and nothing runs now (behind flights the
        // sweep waits for a submission that finds none in the air: there is
        // one at every present).
        static SWEPT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if !behind {
            let _p = ph(Ph::Sweep);
            sweep();
            SWEPT.store(now_ms(), Ordering::Relaxed);
        } else if late_free_present() && now_ms() > SWEPT.load(Ordering::Relaxed) + 1500 {
            // With the present sent ahead nothing else brings a moment
            // with nothing in the air: one is made for the sweep.
            self.settle(&mut submitted, WHY_SYNC);
            let _p = ph(Ph::Sweep);
            sweep();
            SWEPT.store(now_ms(), Ordering::Relaxed);
        }
        let behind = late && flying();
        // The slots that waited for a texture's host copy, before the
        // lists that may read them run.
        {
            let _p = ph(Ph::Slots);
            #[cfg(test)]
            test_hook(HOOK_SLOTS);
            crate::descriptor::make_waiting();
        }
        let this_submission = STARTED.fetch_add(1, Ordering::AcqRel) + 1;
        // The ranges shaders write are counted for it (with the relay).
        crate::shadow::preparing(this_submission);
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        // Bringing host copies up to date, until the lists are ready to go.
        let refresh = ph(Ph::Refresh);
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut batch = Batch::default();
        // The time a settle of flights takes inside this preparing: the
        // wait for the card and the copy-back have their own places on the
        // frame line (a run showed them twice, in "preparing" as well).
        let mut settled = std::time::Duration::ZERO;
        // One write-watch asking for everything the submission refreshes
        // (`VICTORY_ASK_ONCE`): per copy, the span of its refreshed ranges.
        if crate::watch::once() {
            let mut spans: std::collections::HashMap<usize, (usize, usize)> = std::collections::HashMap::new();
            for list in runs {
                for step in &list.deferred().copies {
                    if let Step::Refresh(shadow, address, len) = step
                        && !shadow.unwatched()
                    {
                        let span = spans
                            .entry(std::sync::Arc::as_ptr(shadow) as usize)
                            .or_insert((*address, *address + *len));
                        *span = (span.0.min(*address), span.1.max(*address + *len));
                    }
                }
            }
            let ranges: Vec<(usize, usize)> = spans.into_values().collect();
            crate::watch::ask_once(&ranges);
            // Behind flights: the title wrote memory that a flight's shaders
            // may write too, and this submission refreshes it. The flights
            // are settled, and the asking made again: the copy-back wrote
            // title memory, which other copies of it must see.
            if behind && stale_under_flights(runs) {
                let settling = std::time::Instant::now();
                self.settle(&mut submitted, WHY_STALE);
                settled += settling.elapsed();
                crate::watch::ask_once(&ranges);
            }
        } else if behind && stale_under_flights(runs) {
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_STALE);
            settled += settling.elapsed();
        }
        for list in runs {
            // The targets its commands write join their frame's now: they
            // are written once this submission has run.
            #[cfg(feature = "diagnostics")]
            crate::capture::targets_submitted(std::mem::take(&mut list.deferred().targets));
            list.run_deferred(&mut batch);
            firsts.push(lists.len());
            if let Some(prelude) = list.prelude() {
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
        }
        batch.flush();
        crate::watch::ask_once_end();
        refresh.end();
        // The snapshots of the ranges shaders may write, if they were left
        // for now (`crate::shadow::wide_back`): the refreshes are done, the
        // lists have not gone to the card.
        {
            let _p = ph(Ph::Snapshots);
            #[cfg(feature = "diagnostics")]
            let snapshots = std::time::Instant::now();
            crate::shadow::take_snapshots();
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::SnapshotUs, snapshots.elapsed().as_micros() as u64);
        }
        // Copies in the card's memory: what changed on their CPU side goes
        // up ahead of the lists, and a compare of what the shaders may
        // write follows them in the same submission.
        let flown = late && relay();
        if flown && self.card_results_low() {
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_MOST);
            settled += settling.elapsed();
        }
        // The signals this submission sends on the host's fence, one after
        // each group it sends ([`send_groups`]): the last says it has run.
        #[cfg(all(test, feature = "diagnostics"))]
        let counting = crate::copystat::allocated_here();
        #[cfg(feature = "diagnostics")]
        let signals = if !late { send_signals(timing(), &firsts) } else { 1 };
        #[cfg(not(feature = "diagnostics"))]
        let signals = 1;
        #[cfg(all(test, feature = "diagnostics"))]
        GROUPING.with(|g| g.set((crate::copystat::allocated_here() - counting, 0)));
        let card = {
            let _p = ph(Ph::OwnLists);
            self.card_before(runs, flown, (signals > 0).then(|| *submitted + signals))
        };
        if let Some(check) = card.as_ref().and_then(|c| c.check.clone()) {
            lists.push(Some(check));
        }
        // With `VICTORY_GPU_STAMPS=1` each list's send is noted before it
        // goes, with the value that says it has run.
        #[cfg(feature = "diagnostics")]
        if crate::stamps::on() {
            for list in runs {
                list.stamps_sent(&self.fence, *submitted + signals);
            }
        }
        let prepared = std::time::Instant::now();
        if late {
            *submitted += 1;
            // SAFETY: closed lists; the fence is ours.
            let signalled = unsafe {
                let _p = ph(Ph::Send);
                self.queue.ExecuteCommandLists(&lists);
                self.queue.Signal(&self.fence, *submitted)
            };
            self.card_sent();
            if let Err(e) = signalled {
                fail(format_args!("a submission sent ahead could not be followed by a signal ({e})"));
            }
            let back = declared_writes(runs);
            let writes = runs.iter().flat_map(|l| l.deferred().writes.clone()).collect();
            SUBMISSION_US[0].fetch_add((prepared - started).saturating_sub(settled).as_micros() as u64, Ordering::Relaxed);
            submitting(0, 0);
            #[cfg(feature = "diagnostics")]
            crate::copystat::submission_ends();
            let sent = Sent {
                number: this_submission,
                card,
                back,
                writes,
            };
            working(was_working);
            return Some((submitted, sent));
        }
        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(all(test, feature = "diagnostics"))]
        let grouping = crate::copystat::allocated_here();
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = send_groups(timing(), &firsts, &lists);
        #[cfg(not(feature = "diagnostics"))]
        let groups = [&lists[..]];
        #[cfg(all(test, feature = "diagnostics"))]
        GROUPING.with(|g| g.set((g.get().0, crate::copystat::allocated_here() - grouping)));
        submitting(2, runs.len());
        for (i, group) in groups.iter().enumerate() {
            *submitted += 1;
            let started = std::time::Instant::now();
            // SAFETY: closed lists; the fence and event are ours.
            unsafe {
                let _p = ph(Ph::Send);
                self.queue.ExecuteCommandLists(group);
                self.card_sent();
                // Nothing to wait for without the signal: what follows maps
                // and recycles what the card may still be using.
                if let Err(e) = self.queue.Signal(&self.fence, *submitted) {
                    fail(format_args!("a host submission could not be followed by a signal ({e})"));
                }
            }
            working(false);
            #[cfg(feature = "diagnostics")]
            let waiting = Instant::now();
            let reached = wait_fence(&self.device, &self.fence, Some(self.event), *submitted);
            #[cfg(feature = "diagnostics")]
            crate::stalls::card_waited(crate::stalls::Site::Submission, waiting);
            stood(started);
            working(true);
            // A removed device fails here, before its timing data is read
            // or the submission is published as finished (Sol's re-check,
            // finding 7).
            if !reached {
                fail(format_args!(
                    "host GPU device removed during a submission of {} lists{}{}",
                    runs.len(),
                    self.removal_data(runs),
                    last_notes(runs)
                ));
            }
            // With timing on each group is one of `runs` (and the compare
            // of copies in the card's memory, with the last): every
            // command's time on the card goes to the sums by pipeline.
            #[cfg(feature = "diagnostics")]
            if timing() && i < runs.len() {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                for (ms, op) in runs[i].slowest(frequency, usize::MAX) {
                    crate::copystat::command_took(&op, ms);
                }
            }
            // A submission of one list was never named here (`groups.len()
            // > 1` was the test): the two long lists of the gameplay
            // frame, each a submission of its own, printed nothing.
            #[cfg(feature = "diagnostics")]
            if timing() && i < runs.len() && started.elapsed() > std::time::Duration::from_millis(50) {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                let commands = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner()).len();
                tracing::warn!(ms = started.elapsed().as_millis(), commands, "slow host list");
                for (ms, op) in runs[i].slowest(frequency, 6) {
                    tracing::warn!("slow list command, {ms:.1} ms: {op}");
                }
            }
            let _ = (i, started);
        }
        let _ = &firsts;
        FINISHED.store(this_submission, Ordering::Release);
        bury(this_submission);
        let gpu = prepared.elapsed();
        SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
        SUBMISSION_US[3].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::host_waited(gpu.as_micros() as u64, || {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            let text = format!("{} lists, pipelines {pipelines:?}", runs.len());
            text.chars().take(1200).collect()
        });
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        // The lists have run on a device still there: their timestamps are
        // read (`VICTORY_GPU_STAMPS=1`; below the removal check, Sol's
        // review of the frame timers).
        #[cfg(feature = "diagnostics")]
        if crate::stamps::on() {
            for list in runs {
                list.stamps_landed();
            }
        }
        submitting(3, runs.len());
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let back_phase = ph(Ph::CopyBack);
        let copying = std::time::Instant::now();
        // For copies in the card's memory the compare has run there: the
        // pieces that differ are fetched and merged.
        if let Some(card) = &card {
            self.card_back(card, &mut submitted, &mut std::collections::HashSet::new());
        }
        let back = declared_writes(runs);
        hand_back(&back);
        if let Some(card) = card {
            self.card_done(card);
        }
        #[cfg(feature = "diagnostics")]
        if !back.is_empty() {
            crate::copystat::add(crate::copystat::C::BackSubmissions, 1);
        }
        // Nothing of this submission goes back to the title: no range of
        // shader writes, no value written when it is done. The wait for
        // such a one is only for order and for what the card still holds.
        #[cfg(feature = "diagnostics")]
        if back.is_empty() && runs.iter().all(|l| l.deferred().writes.is_empty()) {
            use crate::copystat::{C, add};
            add(C::NothingBack, 1);
            add(C::NothingBackUs, gpu.as_micros() as u64);
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        back_phase.end();
        for list in runs {
            // Read backs first: a value written at the bottom of the pipe
            // may be what tells the title that they are there.
            read_backs(list);
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
        self.log_messages();
        submitting(0, 0);
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_ends();
        // With the relay: nothing is in the air now, so no fetch still to
        // be merged is older than the sends to card copies so far, and
        // their notes can go.
        if relay() {
            crate::shadow::relay_fetched(*QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        }
        working(was_working);
        None
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let ours = p.pCommandList.as_ref().map(|l| l.as_raw());
                let index = runs.iter().position(|r| Some(r.list.as_raw()) == ours);
                progress.push(format!("{done}/{} {index:?}", p.BreadcrumbCount));
                node = p.pNext;
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let list = p.pCommandList.as_ref().map(|l| l.as_raw());
                    let run = runs.iter().find(|r| Some(r.list.as_raw()) == list);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// Whether the debug layer is on (it has an info queue to read).
    pub(crate) fn debug_layer_on(&self) -> bool {
        self.messages.is_some()
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        self.log_messages_inner(false)
    }

    /// As [`Host::log_messages`]; with `startup`, the tests' store is left alone and the
    /// errors go to the start-up bucket instead (the device's own making, [`messages`]).
    fn log_messages_inner(&self, startup: bool) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        #[cfg(not(test))]
        let _ = startup;
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    #[cfg(test)]
                    messages::keep_error(startup, m.ID.0, m.Severity.0, text);
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(format!("id {}, severity {}: {text}", m.ID.0, m.Severity.0));
                    }
                }
            }
            queue.ClearStoredMessages();
            // The queue keeps a limited number of stored messages; what overran its limit is
            // counted here (the layer's count is for the queue's life), so a log says when
            // messages were lost.
            let discarded = queue.GetNumMessagesDiscardedByMessageCountLimit();
            static DISCARDED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let before = DISCARDED.swap(discarded, Ordering::Relaxed);
            if discarded > before {
                tracing::warn!(
                    discarded = discarded - before,
                    "the debug layer's queue discarded messages past its storage limit"
                );
            }
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: Option<&ID3D12Resource>,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource, or none with a
        // description (a typed null view).
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    pub(crate) fn close(&self) {
        // Its last run ends here (`crate::runs`: (e)).
        #[cfg(feature = "diagnostics")]
        self.runs(crate::runs::Runs::close);
        // Its batch of clamped calls ends, and their slots go back to the
        // common state (`crate::indirect::Batching`).
        crate::indirect::batch_close(&self.list, &mut self.batching());
        // The budget's words, if the list reset it, go to the PC's memory
        // for the frame line ([`HostList::budget`]).
        let reset = self.deferred().budget_reset;
        if reset && let Some(Some(budget)) = self.budget.get() {
            crate::indirect::note_budget_on(&self.list, &budget.card, &budget.notes);
        }
        self.twins_to_depth();
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // The list's pair ends here, and its stamps are resolved
        // (`VICTORY_GPU_STAMPS=1`).
        #[cfg(feature = "diagnostics")]
        if let Some(mut stamps) = self.flight_stamps() {
            stamps.close(&self.list);
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self._alive.used();
        *self.deferred() = Deferred::default();
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.batching().reset();
        #[cfg(feature = "diagnostics")]
        {
            *self.runs.lock().unwrap_or_else(|e| e.into_inner()) = crate::runs::Runs::default();
        }
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
            // Every submission of the list has run: nothing reads its
            // scratch memory any more, and the next list to ask takes it.
            if scratch_pool() {
                give_chunks(std::mem::take(&mut scratch.chunks));
            }
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
        // The list is open again: its opening is stamped.
        #[cfg(feature = "diagnostics")]
        if let Some(mut stamps) = self.flight_stamps() {
            stamps.open(&self.list);
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::ResetNs, started.elapsed().as_nanos() as u64);
    }

    /// With `VICTORY_GPU_STAMPS=1`: a timestamp in the list that begins a
    /// pair ([`HostList::stamp_end`]).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn stamp_begin(&self) -> Option<u32> {
        self.flight_stamps()?.begin(&self.list)
    }

    /// The timestamp that ends a pair of `kind` begun by `begun`
    /// ([`HostList::stamp_begin`]); read once the list has run. No pair if
    /// the work between `recorded` nothing.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn stamp_end(&self, kind: crate::stamps::Kind, begun: Option<u32>, recorded: bool) {
        if let Some(mut stamps) = self.flight_stamps() {
            stamps.end(kind, begun, &self.list, recorded);
        }
    }

    /// With `VICTORY_GPU_STAMPS=2`: a timestamp after the title call just
    /// recorded, which ran `p` ([`crate::stamps`]). Nothing at all with the
    /// switch unset or at 1.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn stamp_call(&self, p: &crate::pipeline::host::HostPipeline) {
        if self.stamps_calls() {
            self.stamp_program(p.program());
        }
    }

    /// With `VICTORY_GPU_PASSES=1`: a title call that runs `p` is about to
    /// be recorded, cut by `key` (0 for a dispatch; for a draw, which
    /// binding of targets it follows), in a title list whose label hashes
    /// to `label` if it has one, in `fragment` of its recording: where it
    /// begins a pass, a timestamp ([`crate::passes`]). A pass counts only if
    /// a call of it is recorded ([`HostList::pass_called`]). With the
    /// stamps unset the list has no stamp state, and this is that field's
    /// test.
    #[cfg(feature = "diagnostics")]
    #[inline]
    pub(crate) fn pass_mark(&self, p: &crate::pipeline::host::HostPipeline, key: u64, label: Option<u32>, fragment: u32) {
        if self.flight_stamps.is_none() || !crate::stamps::passes_on() {
            return;
        }
        if let Some(mut stamps) = self.flight_stamps() {
            stamps.pass(&self.list, key, p.program(), !p.name.is_empty(), label, fragment);
        }
    }

    /// With `VICTORY_GPU_PASSES=1`: a title call was recorded into the
    /// list, in the pass under way.
    #[cfg(feature = "diagnostics")]
    #[inline]
    pub(crate) fn pass_called(&self) {
        if self.flight_stamps.is_none() || !crate::stamps::passes_on() {
            return;
        }
        if let Some(mut stamps) = self.flight_stamps() {
            stamps.pass_called();
        }
    }

    /// With `VICTORY_GPU_PASSES=1`: the timestamp that begins a pair around
    /// a texture upload of ours in the list ([`HostList::upload_stamp_end`]):
    /// `Some` while the passes are on, with the slot or none if it had no
    /// query heap (the pair is then lost, and said so).
    #[cfg(feature = "diagnostics")]
    #[inline]
    pub(crate) fn upload_stamp_begin(&self) -> Option<Option<u32>> {
        if self.flight_stamps.is_none() || !crate::stamps::passes_on() {
            return None;
        }
        Some(self.stamp_begin())
    }

    /// The end of the pair [`HostList::upload_stamp_begin`] began; one whose
    /// beginning had no query heap counts as lost.
    #[cfg(feature = "diagnostics")]
    #[inline]
    pub(crate) fn upload_stamp_end(&self, begun: Option<Option<u32>>) {
        if let Some(begun) = begun {
            self.stamp_end(crate::stamps::Kind::Upload, begun, true);
        }
    }

    /// Whether a title call recorded into the list is stamped
    /// (`VICTORY_GPU_STAMPS=2`). With the switch unset the list has no stamp
    /// state, and this is that field's test alone: no lock, no look at the
    /// switch (Sol's third read of the frame timers: a check of the switch
    /// at every title call).
    #[cfg(feature = "diagnostics")]
    #[inline]
    pub(crate) fn stamps_calls(&self) -> bool {
        self.flight_stamps.is_some() && crate::stamps::calls_on()
    }

    /// Tests: the slots the list's stamps used since it was opened (0
    /// without stamp state).
    #[cfg(all(test, feature = "diagnostics"))]
    pub(crate) fn stamp_slots(&self) -> u32 {
        self.flight_stamps().map_or(0, |s| s.slots())
    }

    /// [`HostList::stamp_call`] for a call that ran `program`.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn stamp_program(&self, program: crate::stamps::Program) {
        if self.stamps_calls()
            && let Some(mut stamps) = self.flight_stamps()
        {
            stamps.call(&self.list, program);
        }
    }

    /// With `VICTORY_GPU_STAMPS=1`: what the list's own pair is counted as
    /// (a title list unless said), from its next close.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn stamps_as(&self, kind: crate::stamps::Kind) {
        if let Some(mut stamps) = self.flight_stamps() {
            stamps.class(kind);
        }
    }

    /// The list goes to the queue (called before it does), and the card
    /// has run it once the host's fence reaches `value`
    /// (`VICTORY_GPU_STAMPS=1`).
    #[cfg(feature = "diagnostics")]
    fn stamps_sent(&self, fence: &ID3D12Fence, value: u64) {
        if let Some(mut stamps) = self.flight_stamps() {
            stamps.sent(fence, value);
        }
    }

    /// The list's oldest send (with its prelude) has run: its pairs are
    /// read, if its fence says so (`VICTORY_GPU_STAMPS=1`).
    #[cfg(feature = "diagnostics")]
    fn stamps_landed(&self) {
        if let Some(stamps) = &self.flight_stamps {
            crate::stamps::landed_in(stamps);
        }
    }

    /// The list's stamps, for a flight that carries no host list (none
    /// with the switch off).
    #[cfg(feature = "diagnostics")]
    fn stamps_handle(&self) -> Option<std::sync::Arc<Mutex<crate::stamps::Stamps>>> {
        self.flight_stamps.clone()
    }

    /// The list's stamps, locked, if it was made with the switch on.
    #[cfg(feature = "diagnostics")]
    fn flight_stamps(&self) -> Option<std::sync::MutexGuard<'_, crate::stamps::Stamps>> {
        self.flight_stamps.as_ref().map(|s| s.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        let texture = crate::resource::host_resource(resource)?;
        let copy = crate::resource::host_copy(resource)?;
        let wanted = legal_state(texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(&copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some((texture, before))
            }
        }
    }

    /// Moves host texture `copy` to `wanted` (a host state) in this list,
    /// as [`HostList::transition`] does a title resource's host copy, and
    /// records the barrier: for textures that are not a title resource's
    /// host copy (colour twins) and for the copies between the two.
    fn settle(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) {
        let wanted = legal_state(&copy.texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let before = match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        };
        drop(states);
        if let Some(before) = before {
            let barrier = transition(
                &copy.texture,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(wanted as i32),
            );
            // SAFETY: an open list and a live texture.
            unsafe { self.list.ResourceBarrier(&[barrier]) };
        }
    }

    /// A use of title texture `resource` about to be recorded, if it is a
    /// depth texture with a colour twin (false if it is not): of the twin
    /// (`colour`: unordered access) or of the depth texture (depth target,
    /// clear, shader read, copy, resolve), writing it or only reading. The
    /// side used is first brought up to date: by a copy recorded here if
    /// this list wrote the other side since, by the list's prelude if the
    /// list uses the side as it finds it. The twin is left in the
    /// unordered access state; the caller moves the depth texture to the
    /// state its use needs, as for any texture. A twin still ahead when the
    /// list closes is copied then (`twins_to_depth`).
    /// shortcut: every depth target bound for a draw counts as written
    /// (also when the draw only tests), and every unordered access view a
    /// shader names as written; views a shader picks from the heap by a
    /// computed index are not seen here at all.
    pub(crate) fn twin_use(
        &self,
        resource: crate::object::This,
        colour: bool,
        write: bool,
    ) -> bool {
        const UNORDERED_ACCESS: u32 = 0x8;
        // Asked for every texture a draw reads: most have no twin.
        let Some(twin) = crate::resource::twin(resource) else {
            return false;
        };
        let Some(depth) = crate::resource::host_copy(resource) else {
            return false;
        };
        let side = usize::from(colour);
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let u = twins
            .entry(std::sync::Arc::as_ptr(&twin) as usize)
            .or_insert_with(|| TwinUse {
                depth: depth.clone(),
                twin: twin.clone(),
                needs: [false; 2],
                behind: [None; 2],
                writing: false,
                late: false,
            });
        let copy = match u.behind[side] {
            Some(behind) => behind,
            None => {
                u.needs[side] = true;
                false
            }
        };
        u.behind[side] = Some(false);
        match (write, colour) {
            // Shaders write the twin when the draw or dispatch runs: what
            // the same one reads of the depth texture is prepared before
            // (`twins_written`).
            (true, true) => u.writing = true,
            (true, false) => u.behind[1] = Some(true),
            (false, _) => {}
        }
        // shortcut: the depth plane's twin is not drawn into the plane
        // here: the pass sets a pipeline and a target of ours, which a list
        // still recording the title's commands cannot take. This use sees
        // the plane as it was; the pass runs when the list closes
        // (`twins_to_depth`). Counted.
        if copy && !colour && twin.plane.is_some() {
            u.late = true;
            crate::resource::PLANES[2].fetch_add(1, Ordering::Relaxed);
        }
        drop(twins);
        if copy {
            self.twin_copy(&depth, &twin, colour);
        } else if colour {
            self.settle(&twin.colour, UNORDERED_ACCESS);
        }
        // The depth plane's twin is written through unordered access too.
        if colour && let Some(plane) = &twin.plane {
            self.settle(plane, UNORDERED_ACCESS);
        }
        true
    }

    /// Records the pass that draws the twin of the depth plane of `depth`
    /// into the plane, if `twin` has one, with the moves to the states the
    /// pass needs; the twin is left in the unordered access state. Only
    /// for a list that records nothing of the title's afterwards
    /// (`record_plane_pass`).
    fn pass_plane(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const DEPTH_WRITE: u32 = 0x10;
        const PIXEL_SHADER_RESOURCE: u32 = 0x80;
        let Some(plane) = &twin.plane else { return };
        self.settle(depth, DEPTH_WRITE);
        self.settle(plane, PIXEL_SHADER_RESOURCE);
        record_plane_pass(&self.list, &depth.texture, &plane.texture);
        self.settle(plane, UNORDERED_ACCESS);
    }

    /// Records the copy of a depth texture into its colour twin (`colour`)
    /// or of the twin into the depth texture, with the moves to the copy
    /// states. The twin is left in the unordered access state.
    fn twin_copy(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
        colour: bool,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let (from, to) = if colour {
            (depth, &twin.colour)
        } else {
            (&twin.colour, depth)
        };
        self.settle(from, COPY_SOURCE);
        self.settle(to, COPY_DEST);
        record_twin_copy(&self.list, &depth.texture, twin, colour);
        self.settle(&twin.colour, UNORDERED_ACCESS);
    }

    /// Before the list closes: the depth textures whose twins it wrote last
    /// are given what was written, and left readable by shaders. Shaders
    /// that pick a view from the heap by a computed index are not seen by
    /// `twin_use` and read the depth texture in lists that never name it.
    fn twins_to_depth(&self) {
        const SHADER_RESOURCE: u32 = 0x40 | 0x80;
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let written: Vec<_> = twins
            .values_mut()
            .filter(|u| u.behind[0] == Some(true) || u.late)
            .map(|u| {
                // Not copied again if the list copied since the last write
                // and only the depth plane's pass is left.
                let copy = u.behind[0] == Some(true);
                u.behind[0] = Some(false);
                u.late = false;
                (u.depth.clone(), u.twin.clone(), copy)
            })
            .collect();
        drop(twins);
        for (depth, twin, copy) in &written {
            if *copy {
                self.twin_copy(depth, twin, false);
            }
            // Nothing of the title's is recorded after this: the pass may
            // leave its own pipeline and target set.
            self.pass_plane(depth, twin);
            self.settle(depth, SHADER_RESOURCE);
        }
    }

    /// The draw or dispatch just prepared is recorded: the twins it writes
    /// now hold what their depth textures lack.
    pub(crate) fn twins_written(&self) {
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.values_mut().filter(|u| u.writing) {
            u.writing = false;
            u.behind[0] = Some(true);
        }
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with. For depth textures with a
    /// colour twin it first copies the side the list uses as it finds it
    /// from the other, if a list submitted earlier left it behind.
    fn prelude(&self) -> Option<ID3D12CommandList> {
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let host_state = |state: u32| D3D12_RESOURCE_STATES(state as i32);
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        // The copies (depth texture, twin, whether into the twin); the
        // sides are then as this list leaves them.
        let mut syncs = Vec::new();
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.values() {
            let real = u.twin.behind.load(Ordering::Relaxed);
            let mut behind = [real & 1 != 0, real & 2 != 0];
            for side in [0, 1] {
                if u.needs[side] && behind[side] {
                    syncs.push((u.depth.clone(), u.twin.clone(), side == 1));
                    behind = [false; 2];
                }
            }
            let left = [0, 1].map(|side| u.behind[side].unwrap_or(behind[side]));
            let left = u32::from(left[0]) | (u32::from(left[1]) << 1);
            u.twin.behind.store(left, Ordering::Relaxed);
        }
        drop(twins);
        // A copy runs with its textures in the copy states: each moves
        // there from its real state, and from there to the state the list
        // starts with (back to its real state if the list does not use it).
        let mut to_copy = Vec::new();
        let mut barriers = Vec::new();
        let mut copying = std::collections::HashMap::new();
        for (depth, twin, colour) in &syncs {
            let (from, to) = if *colour {
                (depth, &twin.colour)
            } else {
                (&twin.colour, depth)
            };
            for (texture, state) in [(from, COPY_SOURCE), (to, COPY_DEST)] {
                let key = std::sync::Arc::as_ptr(texture) as usize;
                let real = texture.state.load(Ordering::Relaxed);
                copying.insert(key, state);
                if real != state {
                    to_copy.push(transition(
                        &texture.texture,
                        host_state(real),
                        host_state(state),
                    ));
                    if !states.contains_key(&key) {
                        barriers.push(transition(
                            &texture.texture,
                            host_state(state),
                            host_state(real),
                        ));
                    }
                }
            }
        }
        barriers.extend(states.iter().filter_map(|(key, (first, last, copy))| {
            let real = copy.state.swap(*last, Ordering::Relaxed);
            let real = copying.get(key).copied().unwrap_or(real);
            (real != *first)
                .then(|| transition(&copy.texture, host_state(real), host_state(*first)))
        }));
        if barriers.is_empty() && syncs.is_empty() {
            return None;
        }
        let Some((allocator, list)) = self
            .prelude
            .get_or_init(|| host().and_then(|h| h.make_prelude()))
            .as_ref()
        else {
            fail(format_args!("no host list for a state prelude"));
        };
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                // A pair around the prelude, resolved in it
                // (`VICTORY_GPU_STAMPS=1`; the list's own were resolved
                // when it closed).
                #[cfg(feature = "diagnostics")]
                let mut stamps = self.flight_stamps();
                crate::events::begin(list, crate::events::PRELUDE);
                #[cfg(feature = "diagnostics")]
                let begun = stamps.as_mut().and_then(|s| s.begin(list));
                if !to_copy.is_empty() {
                    list.ResourceBarrier(&to_copy);
                }
                for (depth, twin, colour) in &syncs {
                    record_twin_copy(list, &depth.texture, twin, *colour);
                }
                if !barriers.is_empty() {
                    list.ResourceBarrier(&barriers);
                }
                crate::events::end(list);
                #[cfg(feature = "diagnostics")]
                if let Some(stamps) = stamps.as_mut() {
                    stamps.end(crate::stamps::Kind::Prelude, begun, list, true);
                    stamps.resolve(list);
                }
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if !dred_on() {
            crate::capture::op(std::ptr::from_ref(self) as usize, what);
            return;
        }
        {
            let started = std::time::Instant::now();
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            crate::capture::op(std::ptr::from_ref(self) as usize, || what.clone());
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
            crate::copystat::add(crate::copystat::C::NoteNs, started.elapsed().as_nanos() as u64);
        }
    }

    /// Whether nothing recorded since the list was opened does any work:
    /// no draw or dispatch (the first keeps its pipeline), no copy, clear,
    /// upload or resolve (they defer a step or move a texture's state), no
    /// bottom-of-pipe write. State set on the list does not count.
    pub(crate) fn untouched(&self) -> bool {
        {
            let deferred = self.deferred();
            if !(deferred.copies.is_empty()
                && deferred.uploads.is_empty()
                && deferred.kept.is_empty()
                && deferred.textures.is_empty()
                && deferred.writes.is_empty()
                && deferred.backs.is_empty())
            {
                return false;
            }
        }
        self.states.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.twins.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Notes a pipeline the list binds.
    pub(crate) fn used_pipeline(&self, name: &str) {
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
    }

    /// The list's enhanced barriers, where the clamps order their reads of
    /// the title's copies with them ([`Host::enhanced_barriers`]; tests:
    /// none on a thread that forces the legacy fallback).
    pub(crate) fn enhanced(&self) -> Option<&ID3D12GraphicsCommandList7> {
        #[cfg(test)]
        if crate::indirect::LEGACY_FORCED.with(std::cell::Cell::get) {
            return None;
        }
        self.enhanced.as_ref()
    }

    /// The list's batch of clamped calls (`crate::indirect::Batching`).
    pub(crate) fn batching(&self) -> std::sync::MutexGuard<'_, crate::indirect::Batching> {
        self.batching.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Something that may write buffer memory is recorded: the batch of
    /// clamped calls open ends, and the next call begins another
    /// (`crate::indirect::batched`).
    pub(crate) fn batch_end(&self) {
        self.batching().end();
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Tests: the copies of title memory into the list's own upload memory
    /// it makes at submission ([`HostList::defer_fill`]: constants a fresh
    /// descriptor table names, among others; a table constant buffer's copy
    /// is [`HostList::defer_table_fill`], the same copy counted apart).
    #[cfg(all(test, feature = "diagnostics"))]
    pub(crate) fn fills(&self) -> usize {
        self.deferred()
            .copies
            .iter()
            .map(|step| match step {
                Step::Fill(_) | Step::TableFill(..) => 1,
                Step::Fills(group) => group.len(),
                _ => 0,
            })
            .sum()
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        COPIES_JOINED[0].fetch_add(1, Ordering::Relaxed);
        // A copy that goes on where the one recorded just before it ended,
        // source and destination both: with `join_copies` the two are one
        // step.
        if let Some(Step::Copy(d, s, n)) = deferred.copies.last_mut()
            && continues((*d, *s, *n), (dst as usize, src as usize, len))
        {
            COPIES_JOINED[1].fetch_add(1, Ordering::Relaxed);
            if join_copies() {
                *n += len;
                return;
            }
        }
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.refreshed.clear();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted. With
    /// `VICTORY_UPLOAD_POOL=1` a run of independent fills is grouped and run
    /// together ([`run_fills`]); with the switch off each is its own step,
    /// as before.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        let fill = upload::Fill { dst: dst as usize, src: src as usize, len };
        self.deferred().append_fill(fill, upload::fill_workers());
    }

    /// As [`Self::defer_fill`], for a table constant buffer's fill: counted
    /// apart from the title's other copies (the native-memory plan, step 2).
    pub(crate) fn defer_table_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::TableFill(dst as usize, src as usize, len));
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run.
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        self.deferred()
            .copies
            .push(Step::GpuWrite(shadow.clone(), address, len));
    }

    /// A read back: once the list has run, `rows` rows of `row` bytes go
    /// from `buffer` (rows 256-byte pitched, see `PlacedFootprint::host_rows`)
    /// to title memory at `dst`, pitched `pitch`. The buffer lives until
    /// the list is reset.
    pub(crate) fn defer_read_back(&self, buffer: ID3D12Resource, dst: *mut u8, [row, rows, pitch]: [usize; 3], source: crate::object::This) {
        // The frame it was recorded in, for the walls trace alone: not
        // asked for with the trace off (Sol's recheck of the second
        // landing, follow-up on disabled tracing).
        let recorded = if walls_trace() { crate::frame::current() } else { 0 };
        // The walls trace's lines of this read-back go to the reporting
        // thread when the list has run, without starting it: it was started
        // when the host was made (Sol's recheck of the second landing, new
        // fault 3: a start here waited for the reporter's lock).
        self.deferred().backs.push((buffer, dst as usize, [row, rows, pitch], source as usize, recorded));
    }

    /// Runs `take` each time the list has run, with its read-backs, until
    /// the list is reset (diagnostics: `take` copies what a read-back
    /// buffer of ours holds and gives the report made of it, if any).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn defer_after(&self, take: impl Fn() -> Option<Report> + Send + 'static) {
        reports_start();
        self.deferred().after.push(Box::new(take));
    }

    /// Whether the list may hold one more read-back of ours of `bytes` for
    /// a diagnostics report, within [`LIST_READS_MOST`] and
    /// [`LIST_READ_BYTES_MOST`] until it is reset; if so it is counted.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn read_budget(&self, bytes: u64) -> bool {
        let reads = &mut self.deferred().reads;
        let fits = reads.0 < LIST_READS_MOST && reads.1.saturating_add(bytes) <= LIST_READ_BYTES_MOST;
        if fits {
            *reads = (reads.0 + 1, reads.1 + bytes);
        }
        fits
    }

    /// Notes title texture `resource` (host copy `copy`) as one the list
    /// writes in listed frame `frame` (diagnostics: `crate::capture::note_target`).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_target(&self, frame: u64, resource: usize, copy: std::sync::Arc<crate::resource::HostTexture>) {
        let targets = &mut self.deferred().targets;
        if !targets.iter().any(|t| t.0 == frame && t.1 == resource) {
            targets.push((frame, resource, copy));
        }
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.upload_buffer(size)?;
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().hold_upload(buffer.clone(), size);
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred().hold_upload(buffer, len);
            return Some((gpu, data));
        }
        // The chunk in use while `len` fits in what is left of it, else the
        // next one (a list's first chunk may be a small one).
        while s.current < s.chunks.len() && s.used + len > s.chunks[s.current].1 {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let size = if scratch_pool() && s.chunks.is_empty() && len <= SCRATCH_FIRST {
                SCRATCH_FIRST
            } else {
                SCRATCH_CHUNK
            };
            let chunk = take_chunk(host, size)?;
            s.chunks.push((chunk, size));
            s.used = 0;
        }
        let (data, gpu) = (s.chunks[s.current].0.data, s.chunks[s.current].0.gpu);
        let at = s.used;
        s.used += len;
        Some((gpu + at, (data + at as usize) as *mut u8))
    }

    /// The bytes of scratch chunks the list holds now (after a reset:
    /// none while lists share their chunks).
    pub(crate) fn scratch_kept(&self) -> u64 {
        let scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        scratch.chunks.iter().map(|c| c.1).sum()
    }

    /// The host address of the list's indirect-work budget, in the card's
    /// memory. Where the list first uses it after a reset, a copy that
    /// zeroes it is recorded ([`crate::indirect::reset_budget_on`]): every
    /// run of the recording starts at zero, in order with the list itself
    /// on the queue. Where the list ends its words go to the PC's memory
    /// ([`HostList::close`]), and the list's next submission takes what
    /// they noted.
    pub(crate) fn budget(&self) -> Option<u64> {
        let host = host()?;
        let budget = self
            .budget
            .get_or_init(|| {
                let card = host.uav_buffer(256)?;
                name(&card, || "indirect budget".to_string());
                let (notes, data) = host.shared_buffer(256)?;
                name(&notes, || "indirect budget notes".to_string());
                Some(Budget { card, notes, data: data as usize })
            })
            .as_ref()?;
        let mut deferred = self.deferred();
        if !deferred.budget_reset {
            crate::indirect::reset_budget_on(&self.list, &budget.card, budget_zeros(host)?);
            deferred.budget_reset = true;
            let at = budget.data;
            // SAFETY: the list's own mapped words, which its last run's
            // copy wrote: what the clamps noted about that run (mesh
            // dispatches: the largest group count asked for, the groups
            // asked in all), taken, then zeroed.
            deferred.copies.push(Step::Host(Box::new(move || unsafe {
                let words = at as *mut u32;
                let (largest, all) = (words.add(1).read_volatile(), words.add(2).read_volatile());
                MESH_GROUPS[0].fetch_max(u64::from(largest), Ordering::Relaxed);
                MESH_GROUPS[1].fetch_add(u64::from(all), Ordering::Relaxed);
                for k in 0..3 {
                    words.add(k).write_volatile(0);
                }
            })));
        }
        // SAFETY: a live buffer.
        Some(unsafe { budget.card.GetGPUVirtualAddress() })
    }

    /// Diagnostics: `count` on the list's census of runs of clamped
    /// indirect calls, when it counts (`crate::runs`).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn runs(&self, count: impl FnOnce(&mut crate::runs::Runs)) {
        if crate::runs::on() {
            count(&mut self.runs.lock().unwrap_or_else(|e| e.into_inner()));
        }
    }

    /// What the list's census counted since its reset (tests).
    #[cfg(all(test, feature = "diagnostics"))]
    pub(crate) fn runs_tally(&self) -> [u64; crate::runs::COUNTS] {
        self.runs.lock().unwrap_or_else(|e| e.into_inner()).tally
    }

    /// The buffer the list's clamps charge, once one has asked for it, and
    /// the words its last run's copy left in the PC's memory.
    #[cfg(test)]
    pub(crate) fn budget_buffer(&self) -> Option<(ID3D12Resource, [u32; 3])> {
        let budget = self.budget.get()?.as_ref()?;
        // SAFETY: the list's own mapped words; read once it has run.
        let notes = unsafe { (budget.data as *const [u32; 3]).read_volatile() };
        Some((budget.card.clone(), notes))
    }

    /// A device-local unordered-access buffer of at least `size` bytes for
    /// clamped indirect arguments, reused across executions and resets of the
    /// list. Smaller buffers replaced mid-list stay alive until
    /// [`HostList::reset`].
    pub(crate) fn indirect_scratch(&self, size: u64) -> Option<ID3D12Resource> {
        let needed = size.max(64 << 10);
        let mut slot = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, cap)) = &*slot
            && *cap >= needed
        {
            return Some(buffer.clone());
        }
        let prev = slot.as_ref().map_or(0, |(_, cap)| *cap);
        let alloc = prev
            .saturating_mul(2)
            .max(needed)
            .next_multiple_of(64 << 10);
        let buffer = host()?.uav_buffer(alloc)?;
        name(&buffer, || format!("indirect arguments {alloc:#x}"));
        let old = slot.replace((buffer.clone(), alloc));
        drop(slot);
        if let Some((old_buf, _)) = old {
            self.keep(&old_buf);
        }
        Some(buffer)
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// The first mip of title texture `resource` that this list's uploads
    /// cover, if it uploads the texture at all.
    pub(crate) fn uploaded_from(&self, resource: usize) -> Option<u32> {
        self.deferred().mips.get(&resource).copied()
    }

    /// Notes that this list's uploads cover title texture `resource` from
    /// mip `first` on (the smallest such mip is kept).
    pub(crate) fn note_uploaded(&self, resource: usize, first: u32) {
        self.deferred()
            .mips
            .entry(resource)
            .and_modify(|f| *f = (*f).min(first))
            .or_insert(first);
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Run(Box::new(work)));
        self.deferred().refreshed.clear();
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted, keeping it alive until the list is reset.
    pub(crate) fn defer_texture(&self, staging: std::sync::Arc<crate::texture::Staging>) {
        self.deferred().copies.push(Step::Texture(staging));
    }
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len(), crate::kernelstat::Caller::ReadInto)
}

/// [`read_into`] with the caller the kernel-call counters name
/// (`readable`'s one-byte probe).
pub(crate) fn read_into_as(src: *const u8, dst: &mut [u8], caller: crate::kernelstat::Caller) -> bool {
    read(src, dst.as_mut_ptr(), dst.len(), caller)
}

/// Copies `len` bytes from `src` to `dst` unless a source page is not
/// readable; counted for the kernel-call figures by `caller`.
fn read(src: *const u8, dst: *mut u8, len: usize, caller: crate::kernelstat::Caller) -> bool {
    #[cfg(feature = "diagnostics")]
    crate::stalls::kernel(crate::stalls::Kernel::Read);
    // SAFETY: the destination is mapped writable memory of `len` bytes; the
    // kernel checks the source.
    unsafe { crate::kernelstat::read_memory(caller, src, dst, len) }
}

/// Whether title memory is written with the kernel's own call
/// (`NtWriteVirtualMemory`) and not `WriteProcessMemory`, which first asks
/// what the pages are (as `VirtualQuery` does: about 2 ms a call in the
/// title's big pools, see [`copy_mapped`]): putting back what shaders wrote
/// cost over a second a frame for hundreds of pieces, milliseconds each whatever
/// their size. On by default; `VICTORY_DIRECT_WRITES=0` writes as before.
fn direct_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DIRECT_WRITES").is_none_or(|v| v != "0"))
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    let direct = direct_writes();
    // Returns the bytes the kernel did not write: a call that fails may have
    // written a prefix, and one that succeeds may have written less than
    // asked. Only what landed is the card's (the census); the rest is
    // retried, and what still fails is counted (`crate::storestat`).
    let write = |at: usize, n: usize| -> usize {
        let from = src.wrapping_add(at - dst as usize);
        #[cfg(feature = "diagnostics")]
        crate::stalls::kernel(crate::stalls::Kernel::Read);
        let kind = if direct { crate::kernelstat::Writeback::Nt } else { crate::kernelstat::Writeback::Process };
        // SAFETY: the source is our mapped host copy; the kernel checks the
        // destination and writes nothing it cannot (the caller held its
        // blocks open: `crate::watch::kernel_writing`).
        let written = unsafe { crate::kernelstat::write_memory(kind, at, from, n) };
        // The card's mark, and its bytes by allocation (`VICTORY_LANDED_CLASSES`).
        #[cfg(feature = "diagnostics")]
        if written > 0 {
            crate::memclass::landed(at, written);
        }
        n - written
    };
    if len == 0 {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::memclass::landing_asked(len);
    let left = write(dst as usize, len);
    if left == 0 {
        return;
    }
    for_pieces(dst as usize + len - left, left, 64 << 10, |at, n| {
        let left = write(at, n);
        if left == 0 {
            return;
        }
        for_pieces(at + n - left, left, 4 << 10, |at, n| {
            let left = write(at, n);
            if left != 0 {
                // Dropped: counted in every build, and said once a run.
                crate::storestat::failed(at + n - left, left);
            }
        });
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

impl Host {
    /// The queue everything of ours runs on, for a call that must go to it
    /// in order with the lists (`crate::tiles`: a texture's levels are
    /// given memory ahead of the list that uploads them).
    pub(crate) fn queue(&self) -> &ID3D12CommandQueue {
        &self.queue
    }

    /// A buffer in the upload heap, mapped for writing.
    /// A buffer of `size` bytes the card writes and we read (a read back),
    /// in the state copies into it need.
    pub(crate) fn read_back_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a read-back buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no read-back buffer");
                None
            }
        }
    }

    /// Tests: the card runs nothing sent to the queue after this until
    /// `fence` reaches `value` (work held in flight).
    #[cfg(all(test, feature = "diagnostics"))]
    pub(crate) fn hold_queue(&self, fence: &ID3D12Fence, value: u64) {
        // SAFETY: a fence of this device.
        unsafe { self.queue.Wait(fence, value) }.expect("the queue waits on the fence");
    }

    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        MADE[0].fetch_add(1, Ordering::Relaxed);
        MADE[1].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    /// With presents sent ahead ([`late_free_present`]): a list for each
    /// present in the air, and the submission it last flew as (it is used
    /// again once that one has finished).
    ahead: Vec<(HostList, u64)>,
    hwnd: HWND,
    meter: Meter,
}

/// The process's CPU time in microseconds, kernel and user.
#[cfg(windows)]
fn process_cpu_us() -> Option<u64> {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    // SAFETY: plain C structs, zero is a valid value for each.
    let (mut creation, mut exit, mut kernel, mut user) = unsafe {
        (
            std::mem::zeroed::<FILETIME>(),
            std::mem::zeroed::<FILETIME>(),
            std::mem::zeroed::<FILETIME>(),
            std::mem::zeroed::<FILETIME>(),
        )
    };
    // SAFETY: the current-process pseudo handle and four writable FILETIMEs.
    let ok = unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            &mut creation,
            &mut exit,
            &mut kernel,
            &mut user,
        )
    };
    if ok == 0 {
        return None;
    }
    let ticks = |f: FILETIME| (u64::from(f.dwHighDateTime) << 32) | u64::from(f.dwLowDateTime);
    Some((ticks(kernel) + ticks(user)) / 10)
}

/// The process's working set in bytes.
#[cfg(windows)]
fn process_working_set() -> Option<u64> {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: a plain C struct, zero is a valid value.
    let mut counters = unsafe { std::mem::zeroed::<PROCESS_MEMORY_COUNTERS>() };
    // SAFETY: the current-process pseudo handle and a counter of the size passed.
    let ok = unsafe {
        GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut counters,
            std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        )
    };
    (ok != 0).then_some(counters.WorkingSetSize as u64)
}

/// The adapter whose video memory the caption reports, when the device's
/// adapter offers it ([`vram_adapter`]), and the device it is looked up by.
/// Looked up at the caption thread's first refresh, not when the device is
/// made: making a DXGI factory and finding the adapter by LUID are waited
/// for without a bound, and the field is optional (Sol's review of the
/// second landing, follow-up 1).
struct VramAdapter {
    device: OnceLock<ID3D12Device>,
    adapter: OnceLock<Option<IDXGIAdapter3>>,
}

impl VramAdapter {
    const fn new() -> VramAdapter {
        VramAdapter { device: OnceLock::new(), adapter: OnceLock::new() }
    }

    /// When the device is made, with the caption on (`caption`) or off:
    /// the device is kept for the lookup, and nothing is asked.
    fn remember(&self, device: &ID3D12Device, caption: bool) {
        if caption {
            let _ = self.device.set(device.clone());
        }
    }

    /// The adapter, for a caption refresh: looked up at the first.
    fn adapter(&self) -> Option<&IDXGIAdapter3> {
        let device = self.device.get()?;
        self.adapter.get_or_init(|| vram_adapter(device, true)).as_ref()
    }

    /// Tests: whether the adapter has been looked up.
    #[cfg(test)]
    fn looked_up(&self) -> bool {
        self.adapter.get().is_some()
    }
}

/// The caption's video memory field's adapter.
static VRAM_ADAPTER: VramAdapter = VramAdapter::new();

/// The adapter `device` was made on, for the caption's video memory field,
/// found by the device's LUID as the host's own is (`Host::adapter`): a
/// D3D12 device is not a DXGI device, and asked for one it gave none, so
/// the field was always left out. An adapter without the interface leaves
/// the field out, said once here rather than on every caption refresh.
/// With `caption` off (`VICTORY_CAPTION=off`) nothing is asked or said: the
/// field is not shown, and the caption's thread, which would ask the
/// adapter for its figures, is not started.
fn vram_adapter(device: &ID3D12Device, caption: bool) -> Option<IDXGIAdapter3> {
    if !caption {
        return None;
    }
    // SAFETY: plain DXGI object creation, and a lookup by the LUID of a
    // live device.
    let adapter = unsafe {
        CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
    };
    if adapter.is_err() {
        tracing::info!("video memory: this adapter does not report use and budget; the caption leaves VRAM out");
    }
    adapter.ok()
}

/// The process's video memory against the budget Windows gives it, in GB to one decimal.
fn vram_line(usage: u64, budget: u64) -> String {
    const GB: f64 = (1u64 << 30) as f64;
    format!("VRAM {:.1}/{:.1} GB", usage as f64 / GB, budget as f64 / GB)
}

/// [`vram_line`] from the adapter, when it can be read; none when it cannot.
fn vram_figures() -> Option<String> {
    let adapter = VRAM_ADAPTER.adapter()?;
    let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    // SAFETY: the adapter is alive and `info` is a writable struct of the right type.
    unsafe { adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }.ok()?;
    Some(vram_line(info.CurrentUsage, info.Budget))
}

/// The machine's logical processors.
#[cfg(windows)]
fn logical_processors() -> u32 {
    use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
    // SAFETY: the call fills the plain struct.
    let mut info = unsafe { std::mem::zeroed::<SYSTEM_INFO>() };
    // SAFETY: as above.
    unsafe { GetSystemInfo(&mut info) };
    info.dwNumberOfProcessors.max(1)
}

/// How long ago the kernel created this process.
#[cfg(windows)]
fn process_age() -> Option<Duration> {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    // A FILETIME counts 100 ns from 1601; the Unix epoch is this many later.
    const UNIX_EPOCH_TICKS: u64 = 116_444_736_000_000_000;
    // SAFETY: plain C structs, zero is a valid value for each.
    let mut times = unsafe { std::mem::zeroed::<[FILETIME; 4]>() };
    let [creation, exit, kernel, user] = &mut times;
    // SAFETY: the current-process pseudo handle and four writable FILETIMEs.
    if unsafe { GetProcessTimes(GetCurrentProcess(), creation, exit, kernel, user) } == 0 {
        return None;
    }
    let ticks = (u64::from(creation.dwHighDateTime) << 32) | u64::from(creation.dwLowDateTime);
    let created = Duration::from_nanos(ticks.checked_sub(UNIX_EPOCH_TICKS)?.checked_mul(100)?);
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .checked_sub(created)
}

/// How long the process has been running: its age when first asked, plus the
/// time since, so the wall clock being set does not move it.
fn process_uptime() -> Duration {
    static FIRST: std::sync::OnceLock<(Instant, Duration)> = std::sync::OnceLock::new();
    let (at, age) = *FIRST.get_or_init(|| (Instant::now(), process_age().unwrap_or_default()));
    age + at.elapsed()
}

/// The window's caption: the meter's figures, the video memory when the adapter reports it, then how long the
/// process has run, as minutes and seconds, with hours once there are any.
fn caption(figures: &str, vram: Option<&str>, up: Duration) -> String {
    let vram = vram.map(|vram| format!(" | {vram}")).unwrap_or_default();
    let s = up.as_secs();
    let (h, m, s) = (s / 3600, s / 60 % 60, s % 60);
    if h == 0 {
        format!("{figures}{vram} | up {m}:{s:02}")
    } else {
        format!("{figures}{vram} | up {h}:{m:02}:{s:02}")
    }
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    /// When it reports, and over which frames ([`stats::Gate`]).
    gate: stats::Gate,
    #[cfg(feature = "diagnostics")]
    copy_stats_frames: u64,
    drawn: u64,
    dispatched: u64,
    /// Draws run on the host when the last frame was presented, for the
    /// live segments' draws a frame.
    live_drawn: u64,
    /// The host's waits over the last report, summed across threads, in ms
    /// a frame ([`Figures`]).
    host_wait_ms: f64,
    /// Whether `VICTORY_LIVE` is set: the live switches' frame boundary is
    /// this present (`crate::live::frame`), with the time since the last.
    live: bool,
    /// When the last presented frame happened, for every frame's time.
    last_present: Option<std::time::Instant>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            gate: stats::Gate::new(now),
            #[cfg(feature = "diagnostics")]
            copy_stats_frames: 0,
            drawn: crate::draw::drawn(),
            dispatched: crate::draw::dispatched(),
            live_drawn: crate::draw::drawn(),
            host_wait_ms: 0.0,
            live: crate::live::on(),
            last_present: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::presented();
            crate::passes::presented();
            crate::memclass::landed_presented();
            // The presenting thread's frame ends here and the next begins
            // (`crate::budget`).
            crate::budget::frame();
        }
        // Our own reporting from here on is a phase of the frame that begins.
        let _report = ph(Ph::Report);
        // The time since the last present is this frame's: the stats file's
        // and the frame line's frame times and, a presented frame being the
        // live switches' frame boundary, the segment lines'. One clock read
        // a present, the one that looked whether a second had passed.
        let now = Instant::now();
        if let Some(previous) = self.last_present.replace(now) {
            let (number, ms) = (crate::queue::frames(), (now - previous).as_secs_f64() * 1000.0);
            self.gate.second.present(number, ms);
            if self.live {
                let drawn = crate::draw::drawn();
                crate::live::frame(number, ms, drawn - std::mem::replace(&mut self.live_drawn, drawn));
            }
        } else {
            // The first present: what was counted before it is no frame's.
            for counter in SUBMISSION_US.iter().chain(&PREPARING).chain(&COPY_BACK) {
                counter.store(0, Ordering::Relaxed);
            }
            crate::draw::take_dropped();
            #[cfg(feature = "diagnostics")]
            crate::budget::take_others();
            (self.drawn, self.dispatched) = (crate::draw::drawn(), crate::draw::dispatched());
            self.live_drawn = self.drawn;
        }
        // `VICTORY_TEXTURE_RECHECK`: a presented frame is the boundary the
        // uploads are looked at from (`crate::host::recheck`); off, one
        // atomic load.
        let frame = crate::frame::current();
        crate::host::recheck::frame(frame);
        crate::kernelstat::frame(frame);
        // `VICTORY_TEXTURE_MADE`: its lines go by this clock, not by a make.
        #[cfg(feature = "diagnostics")]
        crate::resource::made::frame(frame);
        // Pipeline states made, every 600 frames (`host/background.rs`):
        // once a scene has settled, any is one made again.
        crate::host::background::frame(crate::frame::current());
        // `VICTORY_HASH_CHECK=1`: every 600th presented frame says the
        // check's counts; no work with the switch off.
        if hash_check::on() {
            hash_check::frame();
        }
        #[cfg(feature = "diagnostics")]
        {
            self.copy_stats_frames += 1;
        }
        if !self.gate.second_due(now) {
            return;
        }
        // The second's sums, over every frame since the last take: for the
        // stats file's object and, added up, the frame line
        // ([`stats::Interval`]). Summed across threads, so they overlap and
        // are not parts of the frame: preparing submissions; the host's
        // waits (for the relay, for flights' fence, synchronous submissions:
        // not the card's time, which only the stamps measure, and the
        // relay's landings included, Sol's reviews, finding 10 and the frame
        // timers); copies; the copy-back. The process's CPU share and
        // working set are sampled on the caption's thread ([`Usage`]).
        let (drawn, dispatched) = (crate::draw::drawn(), crate::draw::dispatched());
        {
            use stats::sum::{BACK_US, COPIES_US, DISPATCHED, DRAWN, DROPPED, PREPARING_US, WAITS_US};
            let s = &mut self.gate.second.sums;
            s[PREPARING_US] = SUBMISSION_US[0].swap(0, Ordering::Relaxed);
            for i in 0..3 {
                s[WAITS_US + i] = SUBMISSION_US[1 + i].swap(0, Ordering::Relaxed);
            }
            for i in 0..4 {
                s[COPIES_US + i] = PREPARING[i].swap(0, Ordering::Relaxed);
            }
            for i in 0..2 {
                s[BACK_US + i] = COPY_BACK[i].swap(0, Ordering::Relaxed);
            }
            (s[DRAWN], s[DISPATCHED], s[DROPPED]) = (drawn - self.drawn, dispatched - self.dispatched, crate::draw::take_dropped());
        }
        // The presenting thread's budget over the same frames, and what the
        // other threads measured meanwhile (`crate::budget`).
        #[cfg(feature = "diagnostics")]
        {
            self.gate.second.budget = stats::Budget::take();
        }
        (self.drawn, self.dispatched) = (drawn, dispatched);
        let frames = self.gate.second.frames as f64;
        let (fps, ms) = (self.gate.second.fps(), self.gate.second.frame_ms());
        let draws = self.gate.second.per_frame(stats::sum::DRAWN, 1.0);
        self.host_wait_ms = stats::waits(&self.gate.second).all();
        // A diagnostic, a submission late and not exact ([`MESH_GROUPS`]).
        let mesh_largest = MESH_GROUPS[0].swap(0, Ordering::Relaxed);
        let mesh_groups = MESH_GROUPS[1].swap(0, Ordering::Relaxed) as f64 / frames;
        tracing::info!(
            "indirect mesh dispatches: the largest group count asked for {mesh_largest} (run: at most 65535), {mesh_groups:.0} groups asked per frame"
        );
        // With `VICTORY_GPU_STAMPS`: the census of runs of clamped calls.
        #[cfg(feature = "diagnostics")]
        if crate::stamps::on() {
            crate::runs::line(frames);
        }
        // The clamp of runs (on unless `VICTORY_CLAMP_RUNS=0`).
        crate::indirect::runs_line(frames);
        // With `VICTORY_GPU_STAMPS`, and whenever there were any.
        let twice = NAMED_TWICE.swap(0, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        let stamped = crate::stamps::on();
        #[cfg(not(feature = "diagnostics"))]
        let stamped = false;
        if twice > 0 || stamped {
            tracing::info!("submissions naming a list twice: {twice} (since the last line; each carried out as several, waited for)");
        }
        late_line(frames);
        card_clock_line(frames);
        #[cfg(feature = "diagnostics")]
        if crate::stamps::on() {
            crate::stamps::line();
        }
        crate::shadow::back_line(frames);
        {
            let [passes, pieces, compares] = [0, 1, 2].map(|i| CARD_PASSES[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if passes + compares > 0.0 {
                tracing::info!(
                    "our passes over copies in the card's memory per frame: {passes:.0} hash passes over {pieces:.0} pieces after \
                     ranges were sent up (VICTORY_CARD_REHASH_RUNS=1 joins them: now {}), {compares:.0} compare passes",
                    if rehash_runs() { "on" } else { "off" }
                );
            }
        }
        {
            let us = [0, 1, 2, 3].map(|i| crate::texture::REFRESH_WAITS[i].swap(0, Ordering::Relaxed));
            // And for the stall line, over its five seconds.
            #[cfg(feature = "diagnostics")]
            crate::stalls::refresh_waits([us[0], us[1], us[2]]);
            let [own, asking, still, all] = us.map(|us| us as f64 / 1000.0 / frames);
            if all >= 1.0 {
                tracing::info!(
                    "texture refreshes per frame, all threads: {all:.1} ms, of them {own:.1} ms waiting for a texture another thread \
                     is refreshing, {asking:.1} ms asking the write watch and the guards, {still:.1} ms waiting for title memory to \
                     hold still (the runtime remapping it); the rest is reading, hashing and detiling"
                );
            }
            let [through, left] = [0, 1].map(|i| crate::texture::UNLOCKED[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if through + left > 0.0 {
                tracing::info!(
                    "texture refreshes that did not wait for title memory to hold still, per frame (VICTORY_TEXTURE_NO_WAIT=1): \
                     {through:.1} read through the kernel, {left:.1} left for the next refresh (a page was not there)"
                );
            }
        }
        // The other side of "waiting for title memory to hold still": the
        // runtime's changes of mapping, what they waited for and held.
        #[cfg(feature = "diagnostics")]
        {
            let [changes, waiting, holding] =
                [0, 1, 2].map(|i| crate::watch::REMAPS[i].swap(0, Ordering::Relaxed) as f64 / [1.0, 1e3, 1e3][i] / frames);
            let [waited_most, held_most] = [3, 4].map(|i| crate::watch::REMAPS[i].swap(0, Ordering::Relaxed) as f64 / 1e3);
            if changes > 0.0 {
                tracing::info!(
                    "changes of title memory's mapping per frame: {changes:.0}, {waiting:.1} ms of them waiting for reads in place \
                     (longest {waited_most:.1} ms), {holding:.1} ms holding {} (longest {held_most:.1} ms; \
                     VICTORY_MAPPING_RANGES=0 holds all of title memory)",
                    if crate::watch::by_range() { "the bytes each changes" } else { "all of title memory" }
                );
            }
        }
        {
            let [copies, joined] = [0, 1].map(|i| COPIES_JOINED[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if copies > 0.0 {
                tracing::info!(
                    "title buffer copies recorded per frame: {copies:.0}, of them {joined:.0} continue the one before \
                     (VICTORY_JOIN_COPIES=1 makes those one copy with it: now {})",
                    if join_copies() { "on" } else { "off" }
                );
            }
        }
        // Every second: the card's queues by themselves, and the stats file's
        // object, of every frame since the one before it ([`stats`]); the
        // second's frames then join the frame line's.
        #[cfg(feature = "diagnostics")]
        let stamps = {
            let (copy, kinds) = (queue_stamps::take(), queue_stamps::not_stamped());
            if crate::stamps::on() {
                tracing::info!("{}", queue_stamps::text(&copy, frames, &kinds));
            }
            move || queue_stamps::json(&copy, frames, &kinds)
        };
        #[cfg(not(feature = "diagnostics"))]
        let stamps = || "null".to_owned();
        let vram = || crate::host::host().and_then(|h| h.video_memory()).map(|m| m[1] >> 20);
        if self.gate.second_done(now, |second| stats::write(second, vram, stamps)) {
            // From here: per frame over every frame since the last frame line.
            let frames = self.gate.line.frames.max(1) as f64;
            let indirect = crate::list::take_indirect() as f64 / frames;
            // Passes over the heap's buffer views for shaders that pick them by index.
            let picked = crate::draw::HEAP_BUFFER_PASSES.swap(0, Ordering::Relaxed) as f64 / frames;
            let picked_ms =
                crate::draw::HEAP_BUFFER_MICROS.swap(0, Ordering::Relaxed) as f64 / 1e3 / frames;
            // Copies between depth textures and their colour twins, each way.
            let [twin_to_depth, twin_to_colour] =
                [2, 3].map(|i| crate::resource::TWINS[i].swap(0, Ordering::Relaxed) as f64 / frames);
            // Depth planes drawn from their twins.
            let planes = crate::resource::PLANES[0].swap(0, Ordering::Relaxed) as f64 / frames;
            // Copies in the card's memory (`crate::shadow::cards`).
            let card = crate::shadow::card_stats();
            #[cfg(feature = "diagnostics")]
            let slow = crate::watch::slow_asks();
            // Over the same frames as the counts above.
            #[cfg(feature = "diagnostics")]
            let drops = crate::pipeline::take_drops();
            tracing::info!(
                "placed texture views: {} writes of another view of the texture a slot showed (it kept the old view until the new was made)",
                crate::descriptor::host::KEPT_VIEWS.load(Ordering::Relaxed)
            );
            crate::tiles::status();
            {
                let [same, changed, waiting] =
                    [0, 1, 2].map(|i| crate::draw::HEAP_TEXTURES_RECHECKED[i].load(Ordering::Relaxed));
                tracing::info!(
                    "textures picked by index, looked at again after their upload: {same} looks found them unchanged, \
                     {changed} changed (uploaded again), {waiting} wait for a look"
                );
                let [queued, due, away, partly, oldest] =
                    [0, 1, 2, 3, 4].map(|i| crate::draw::HEAP_TEXTURES_QUEUE[i].load(Ordering::Relaxed));
                let [same, parts, tails] =
                    [0, 1, 2].map(|i| crate::draw::HEAP_TEXTURES_PARTLY[i].load(Ordering::Relaxed));
                tracing::info!(
                    "textures picked by index, the queue: {queued} slots, {due} of them due (the oldest written {oldest} frames ago), \
                     {away} waiting for their memory, {partly} uploaded in part; since the start: {} slots found uploaded already by \
                     the draws that read them through tables, {} writes of tables' slots (due some frames later), {} slots taken out \
                     as read through a table, {parts} uploads of textures not all mapped ({tails} of them without their first page), \
                     {same} looks that found no more mapped; {} slots made stale by a newer view of other memory there, \
                     {} ranges of memory mapped, unmapped or written under texture views ({} dropped), {} slots made due by them, \
                     {} made slots found unmapped and made null",
                    crate::draw::HEAP_TEXTURES_TABLED.load(Ordering::Relaxed),
                    crate::descriptor::host::TABLED_WRITES[0].load(Ordering::Relaxed),
                    crate::descriptor::host::TABLED_WRITES[1].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[0].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[1].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[2].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[3].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[4].load(Ordering::Relaxed)
                );
                let [views, shown, waiting, tables, gone, own, free, start, other, shows, back] =
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(|i: usize| crate::descriptor::host::SWEPT[i].load(Ordering::Relaxed));
                tracing::info!(
                    "placed texture views in the heap at the last look: {views}, of them {shown} shown, {waiting} in the queue, \
                     {tables} read through tables, {gone} without memory; with memory and nothing to bring them up to date: \
                     {own} of their own memory, {free} of memory no other view describes, {start} of memory another texture \
                     begins at, {other} of another's memory ({shows} of all these show another view); {back} put back into the \
                     queue since the start"
                );
            }
            {
                let mode = crate::watch::page_writes();
                let [writes, bytes, guarded, plain] = [0, 1, 2, 3].map(|i| crate::watch::WRITTEN[i].load(Ordering::Relaxed));
                let [kept, dropped, slots, uploaded] =
                    [0, 1, 2, 3].map(|i| crate::descriptor::host::PAGE_WRITES[i].load(Ordering::Relaxed));
                tracing::info!(
                    "writes behind the guards, since the start: {writes} DirectStorage writes into physical pages at addresses the                      title had mapped ({} MB); {guarded} guarded 64 KiB blocks of host copies lay under them ({}); {plain} reads to                      plain addresses finished ({}); {kept} ranges under texture views kept for the texture pass ({dropped} dropped);                      {slots} slots of placed textures held their bytes, {uploaded} of them uploaded before the write ({})                      (VICTORY_PAGE_WRITES: unset or 1 acts, 0 counts, guards, slots, all)",
                    bytes >> 20,
                    if mode & 1 != 0 { "opened, taken again" } else { "left as they were" },
                    if mode & 4 != 0 { "told as well" } else { "counted only" },
                    if mode & 2 != 0 { "made due again" } else { "left as they were" }
                );
            }
            #[cfg(feature = "diagnostics")]
            tracing::info!(
                "write-watch askings of a millisecond or more: {:.1} a frame, {:.1} ms a frame; the longest asking {:.1} ms",
                slow[0] as f64 / frames,
                slow[1] as f64 / 1e3 / frames,
                slow[2] as f64 / 1e3
            );
            if crate::shadow::cards() {
                let per_frame = |i: usize| card[i] as f64 / frames;
                tracing::info!(
                    "host copies in the card's memory: {} copies {} MB, their bytes from before {} MB; per frame: \
                     {:.1} MB sent up in {:.0} ranges, {:.0} pieces examined for fetch (compared on the card, or fetched \
                     whole without a compare), {:.0} of them fetched ({:.1} MB) in {:.1} fetch batches ({:.1} ms from \
                     recording to completion)",
                    card[7],
                    card[8] >> 20,
                    card[9] >> 20,
                    per_frame(0) / 1e6,
                    per_frame(1),
                    per_frame(2),
                    per_frame(3),
                    per_frame(4) / 1e6,
                    per_frame(5),
                    per_frame(6) / 1e3
                );
            }
            let beside = stats::Beside { fetched_mb: card[4] as f64 / 1e6 / frames, indirect, picked, picked_ms, twin_to_depth, twin_to_colour, planes };
            stats::frame_line(&self.gate.line, &beside);
            // The live segment under way: a run killed from outside logs no last segment.
            if self.live && let Some(open) = crate::live::open_line() {
                tracing::info!("{open}");
            }
            self.gate.line_done();
            #[cfg(feature = "diagnostics")]
            {
                // What held the host up since the last such line, over the
                // frames the copies' counters count (taken just after).
                let (counted, others) = crate::copystat::stall_figures();
                crate::stalls::line(if counted > 0 { counted as f64 } else { self.copy_stats_frames as f64 }, others);
                crate::draw::stuck_unsaid_line();
                crate::copystat::log(std::mem::take(&mut self.copy_stats_frames) as f64);
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
            }
        }
        // `VICTORY_CAPTION=off`: the title's caption stays, and nothing is
        // sampled for ours.
        if !caption_on() {
            return;
        }
        let figures = Figures { fps, ms, host_wait_ms: self.host_wait_ms, draws };
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest figures when free,
        // samples the process's CPU time, working set and GPU share when they
        // come, and keeps the running time moving while frames are slow,
        // until presents stop ([`caption_due`]).
        // The window's owner now, at a present to it: the setter writes to
        // the handle only while it is still that window (Sol's re-check,
        // finding 10: a handle freed and given to another window passed
        // `IsWindow`).
        static PENDING: Mutex<Option<(usize, (u32, u32), Figures, Instant)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        let owner = window_owner(hwnd);
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, owner, figures, Instant::now()));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    let (mut newest, mut shown, mut usage) = (None, String::new(), Usage::new());
                    let (mut gpu, mut gpu_busy) = (GpuBusy::new(), None);
                    loop {
                        // Taken in a statement of its own: the lock is let
                        // go before the sampling, which a present waiting
                        // for it would wait for (Sol's re-check, new fault 5).
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        if let Some(next) = next {
                            usage.sample();
                            gpu_busy = gpu.sample();
                            newest = Some(next);
                        }
                        if let Some((hwnd, owner, figures, at)) = &newest
                            && caption_due(at.elapsed())
                        {
                            let hwnd = HWND(*hwnd as *mut _);
                            // Asked only while the caption is set: not once
                            // presents stop, and never with the caption off
                            // (this thread does not run then).
                            let vram = vram_figures();
                            let text = caption(
                                &figures.text(build_name(), Usage::latest(), gpu_busy),
                                vram.as_deref(),
                                process_uptime(),
                            );
                            // SAFETY: a plain query of the process's id.
                            let ours = unsafe { windows::Win32::System::Threading::GetCurrentProcessId() };
                            if text != shown && same_window(*owner, window_owner(hwnd), ours) {
                                let wide: Vec<u16> = text.encode_utf16().chain([0]).collect();
                                // SAFETY: a window handle and a NUL-terminated text.
                                unsafe {
                                    let _ = SetWindowTextW(hwnd, PCWSTR(wide.as_ptr()));
                                }
                                shown = text;
                            }
                        }
                        std::thread::sleep(Duration::from_millis(100));
                    }
                });
        });
    }
}

/// `VICTORY_CAPTION`: on unless `off`. On, the window's caption shows the
/// build ([`build_name`]), the frame rate, our wait for the card, the share
/// of the GPU's 3D engines our work took ([`GpuBusy`]), the process's CPU
/// share and working set, its video memory against its budget, and how long
/// it has run. `off` leaves the title's own caption, and the CPU time,
/// working set, GPU share and video memory are not sampled.
fn caption_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| caption_switch(std::env::var("VICTORY_CAPTION").ok().as_deref()))
}

/// Whether `VICTORY_CAPTION` set to `value` (or unset) leaves our caption
/// on: everything but `off`, in any case, with spaces around.
fn caption_switch(value: Option<&str>) -> bool {
    !value.is_some_and(|v| v.trim().eq_ignore_ascii_case("off"))
}

/// The process and thread that own window `hwnd`; (0, 0) for a handle that
/// is no window.
fn window_owner(hwnd: HWND) -> (u32, u32) {
    let mut process = 0;
    // SAFETY: a plain query of a handle, which may be no window's.
    let thread = unsafe { GetWindowThreadProcessId(hwnd, Some(&mut process)) };
    if thread == 0 { (0, 0) } else { (process, thread) }
}

/// Whether a handle whose window was owned by `then` (process, thread) when
/// it was presented to, and is owned by `now`, is still that window of this
/// process (`ours`): a window gone, or its handle given to another
/// window, is not written to.
fn same_window(then: (u32, u32), now: (u32, u32), ours: u32) -> bool {
    then.1 != 0 && then.0 == ours && now == then
}

/// How long after the last figures (a second of presents) the caption's
/// running time still moves: frames can be slow, but once presents stop
/// the window is left alone (Sol's review, finding 10).
const CAPTION_STALE: Duration = Duration::from_secs(5);

/// Whether the caption is still set, figures `age` old.
fn caption_due(age: Duration) -> bool {
    age < CAPTION_STALE
}

/// A second of presents, for the caption: frames a second, ms a frame, the
/// host's waits in ms a frame (summed across threads), draws a frame.
#[derive(Clone, Copy)]
struct Figures {
    fps: f64,
    ms: f64,
    host_wait_ms: f64,
    draws: f64,
}

impl Figures {
    /// The caption's figures, before the running time: the build (`build`,
    /// [`build_name`]), then these, the GPU's share (`gpu`, [`GpuBusy`])
    /// and the process's CPU and memory (`usage`, [`Usage`]).
    fn text(&self, build: &str, usage: Option<(f64, f64)>, gpu: Option<f64>) -> String {
        let Figures { fps, ms, host_wait_ms, draws } = *self;
        let gpu = gpu.map_or_else(|| "gpu -".to_owned(), |share| format!("gpu {share:.0}%"));
        format!(
            "{build} | {fps:.0} fps {ms:.1} ms/frame | host waits {host_wait_ms:.0} ms/frame {gpu} {} | {draws:.0} draws/frame",
            Usage::text(usage)
        )
    }
}

/// What the caption calls this build: `ZSlayerHQ`, the commit's short hash
/// and the profile ([`profile_word`]), taken when it was built (`build.rs`),
/// never the branch. A name shows before the hash only where
/// `VICTORY_BUILD_NAME` (read once, at the first caption) asks for one, an
/// opt-in to its public display (the harness sets it for a fork's run); the
/// caption can appear in a shared recording, so one that is no public label
/// ([`crate::build_label::public`]) shows as `build`, said in the log
/// without its text.
fn build_name() -> &'static str {
    static NAME: OnceLock<String> = OnceLock::new();
    NAME.get_or_init(|| {
        let name = std::env::var("VICTORY_BUILD_NAME").ok();
        if name.as_deref().map(str::trim).is_some_and(|name| !name.is_empty() && !crate::build_label::public(name)) {
            tracing::info!(
                "VICTORY_BUILD_NAME is not a public label (1 to 40 of A-Z a-z 0-9 . _ / -, no path, address or \
                 login); the caption says {}",
                crate::build_label::FALLBACK
            );
        }
        build_identity(
            env!("VICTORY_GPU_BUILD_COMMIT"),
            profile_word(env!("VICTORY_GPU_BUILD_PROFILE"), cfg!(feature = "diagnostics")),
            name.as_deref(),
        )
    })
}

/// The caption's word for the profile, from Cargo's `PROFILE` (`debug` or
/// `release`) and whether the diagnostics feature is on: `diag` for release
/// code with it (the workspace's `diag` profile, as the harness builds it),
/// `release` and `debug` as they are; any other word is `build`.
fn profile_word(profile: &str, diagnostics: bool) -> &'static str {
    match profile {
        "release" if diagnostics => "diag",
        "release" => "release",
        "debug" => "debug",
        _ => crate::build_label::FALLBACK,
    }
}

/// [`build_name`] from its parts: `name` (`VICTORY_BUILD_NAME`), trimmed,
/// shows before the commit unless it is blank or none, and only if it is a
/// public label, else as `build`. The commit shows only as seven hex digits
/// (else `unknown`), the profile only as [`profile_word`]'s words.
fn build_identity(commit: &str, profile: &str, name: Option<&str>) -> String {
    use crate::build_label::{FALLBACK, public};
    let name = match name.map(str::trim) {
        Some(name) if public(name) => format!("{name} "),
        Some(name) if !name.is_empty() => format!("{FALLBACK} "),
        _ => String::new(),
    };
    let commit = if commit.len() == 7 && commit.bytes().all(|b| b.is_ascii_hexdigit()) { commit } else { "unknown" };
    let profile = if ["debug", "release", "diag"].contains(&profile) { profile } else { FALLBACK };
    format!("ZSlayerHQ {name}{commit} {profile}")
}

/// The share of the last sample's time this process's work kept the
/// device's adapter's busiest 3D engine busy ([`gpu_share`]), from the
/// kernel's graphics statistics (`D3DKMTQueryStatistics`): the running time
/// it counts for a process on each engine, the figure Task Manager's GPU
/// column and the `GPU Engine` performance counters are made from (an
/// interface Microsoft documents for system use: where it fails, the
/// caption says `gpu -`). Asked of this process and this adapter alone, it
/// is one kernel query per 3D engine (one, on most cards) a second, on the
/// caption's thread, never the render thread; the performance counters
/// would gather every process's engines on every adapter at each
/// collection, and their instance names match a pid's prefix as well
/// (`pid_12*` takes `pid_123`). Release builds read it too.
struct GpuBusy {
    /// The adapter and its 3D engines' node numbers, looked up at the first
    /// sample; none when they cannot all be read (said once).
    engines: Option<Option<(windows::Win32::Foundation::LUID, Vec<u32>)>>,
    /// Each engine's running time, in 100 ns, at the last sample, and when
    /// it was read.
    last: Option<(Vec<i64>, Instant)>,
    /// Whether a running time that could not be read has been said.
    said: bool,
}

impl GpuBusy {
    const fn new() -> GpuBusy {
        GpuBusy { engines: None, last: None, said: false }
    }

    /// The share since the last sample, in percent; none at the first, or
    /// when the adapter's statistics cannot be read.
    fn sample(&mut self) -> Option<f64> {
        let (luid, nodes) = self
            .engines
            .get_or_insert_with(|| {
                let device = VRAM_ADAPTER.device.get();
                // SAFETY: plain queries of a live device.
                let (luid, linked) = match device.map(|device| unsafe { (device.GetAdapterLuid(), device.GetNodeCount()) }) {
                    Some((luid, linked)) => (Some(luid), linked),
                    None => (None, 0),
                };
                // Linked adapters (several physical ones behind one device)
                // number their engines by physical adapter as well; not read.
                let engines =
                    luid.filter(|_| linked == 1).and_then(|luid| engines_3d(luid).map(|nodes| (luid, nodes)));
                match &engines {
                    Some((_, nodes)) => tracing::info!(
                        "GPU share: the caption reads this process's running time on {} 3D engine(s) of the device's \
                         adapter, the busiest engine's share",
                        nodes.len()
                    ),
                    None => tracing::info!(
                        "GPU share: the device's adapter's 3D engines cannot all be found (no statistics, a software \
                         adapter such as WARP, a node unread, or {linked} linked adapters); the caption says gpu -"
                    ),
                }
                engines
            })
            .as_ref()?;
        let (running, at) = (running_times(*luid, nodes), Instant::now());
        let Some(running) = running else {
            if !std::mem::replace(&mut self.said, true) {
                tracing::info!("GPU share: this process's running time on the 3D engines could not be read; the caption says gpu -");
            }
            self.last = None;
            return None;
        };
        let (before, then) = self.last.replace((running, at))?;
        let (running, _) = self.last.as_ref()?;
        gpu_share(&before, running, at.duration_since(then))
    }
}

/// Running time between two samples beyond this many times the wall time
/// between them is no running time: a packet's time is counted when it
/// completes, so one that began before the interval can put more than the
/// interval in it (the engine was busy all of it), but not ten intervals'
/// worth, packets being bounded by the GPU's timeout detection.
const MOST_INTERVALS: u128 = 10;

/// The share of `wall` that the busiest 3D engine was busy with this
/// process's work, in percent, from each engine's running time (in 100 ns)
/// at the two samples (`before`, `now`, engine by engine): the busiest
/// engine's share, as Task Manager's process GPU column takes it, so two
/// engines each busy 60% of the time are 60%, not a sum of 120% (or 100%
/// capped). At most 100 (a packet longer than the interval). None for
/// engines that do not match or none, a negative counter, one that went
/// back (a wrap or a reset), more than [`MOST_INTERVALS`] intervals of
/// running time, or less than one count (100 ns) between the samples.
fn gpu_share(before: &[i64], now: &[i64], wall: Duration) -> Option<f64> {
    let wall = wall.as_nanos() / 100;
    if before.is_empty() || before.len() != now.len() || wall == 0 {
        return None;
    }
    let mut busiest = 0u64;
    for (&then, &now) in before.iter().zip(now) {
        if then < 0 || now < then {
            return None;
        }
        // Both at least zero: the difference fits.
        busiest = busiest.max(u64::try_from(now - then).ok()?);
    }
    if u128::from(busiest) > wall.saturating_mul(MOST_INTERVALS) {
        return None;
    }
    Some((busiest as f64 * 100.0 / wall as f64).min(100.0))
}

/// `D3DKMT_NODEMETADATA`'s size: the node's number (4 bytes) and
/// `DXGK_NODEMETADATA` (engine type 4, friendly name 64, flags 4, two
/// `BOOLEAN`s), packed.
const NODE_METADATA_SIZE: usize = 78;

/// Where the engine type lies in `D3DKMT_NODEMETADATA`.
const ENGINE_TYPE_AT: usize = std::mem::offset_of!(windows::Wdk::Graphics::Direct3D::D3DKMT_NODEMETADATA, NodeData)
    + std::mem::offset_of!(windows::Wdk::Graphics::Direct3D::DXGK_NODEMETADATA, EngineType);

const _: () = {
    assert!(std::mem::size_of::<windows::Wdk::Graphics::Direct3D::D3DKMT_NODEMETADATA>() == NODE_METADATA_SIZE);
    assert!(ENGINE_TYPE_AT == 4);
    assert!(std::mem::size_of::<windows::Wdk::Graphics::Direct3D::D3DKMT_ADAPTERTYPE>() == 4);
};

/// The engine type in a node's metadata as the kernel wrote it (`bytes`,
/// [`NODE_METADATA_SIZE`] of them): read as bytes, as the binding's `bool`s
/// for its two `BOOLEAN`s would make a byte other than 0 or 1 undefined
/// (Sol's review, finding 3); none for a read of another size.
fn engine_type(bytes: &[u8]) -> Option<i32> {
    if bytes.len() != NODE_METADATA_SIZE {
        return None;
    }
    Some(i32::from_le_bytes(bytes[ENGINE_TYPE_AT..ENGINE_TYPE_AT + 4].try_into().ok()?))
}

/// The 3D engines' node numbers from every node's engine type (`types`, in
/// node order; none for a node whose metadata could not be read): none if a
/// node was not read (the discovery incomplete), or none is a 3D engine.
fn engines_from(types: &[Option<i32>]) -> Option<Vec<u32>> {
    let mut nodes = Vec::new();
    for (node, kind) in types.iter().enumerate() {
        if *kind.as_ref()? == windows::Wdk::Graphics::Direct3D::DXGK_ENGINE_TYPE_3D.0 {
            nodes.push(u32::try_from(node).ok()?);
        }
    }
    (!nodes.is_empty()).then_some(nodes)
}

/// The most nodes an adapter's count is believed for.
const MOST_NODES: u32 = 64;

/// The node numbers of the 3D engines of the adapter `luid` names, as the
/// kernel numbers them for its statistics; none if it keeps no statistics
/// for the adapter, a node's metadata cannot be read ([`engines_from`]), it
/// names no 3D engine, or it is a software adapter (WARP's: our work runs
/// on the CPU, and its engines count none of it).
fn engines_3d(luid: windows::Win32::Foundation::LUID) -> Option<Vec<u32>> {
    use windows::Wdk::Graphics::Direct3D::*;
    /// `D3DKMT_ADAPTERTYPE`'s `SoftwareDevice` bit.
    const SOFTWARE_DEVICE: u32 = 1 << 2;
    let mut open = D3DKMT_OPENADAPTERFROMLUID { AdapterLuid: luid, hAdapter: 0 };
    // SAFETY: a struct the call fills with the adapter's handle.
    if unsafe { D3DKMTOpenAdapterFromLuid(&mut open) }.0 < 0 {
        return None;
    }
    // Asks the open adapter for `kind`, into the bytes of `what`.
    let ask = |kind: KMTQUERYADAPTERINFOTYPE, what: &mut [u8]| {
        let mut info = D3DKMT_QUERYADAPTERINFO {
            hAdapter: open.hAdapter,
            Type: kind,
            pPrivateDriverData: what.as_mut_ptr().cast(),
            PrivateDriverDataSize: u32::try_from(what.len()).unwrap_or(0),
        };
        // SAFETY: an open adapter, and a buffer of the size given for the
        // kind asked, filled in place.
        unsafe { D3DKMTQueryAdapterInfo(&mut info) }.0 >= 0
    };
    let nodes = (|| {
        let mut kind = [0u8; 4];
        if !ask(KMTQAITYPE_ADAPTERTYPE, &mut kind) || u32::from_le_bytes(kind) & SOFTWARE_DEVICE != 0 {
            return None;
        }
        let mut adapter =
            D3DKMT_QUERYSTATISTICS { Type: D3DKMT_QUERYSTATISTICS_ADAPTER, AdapterLuid: luid, ..Default::default() };
        // SAFETY: a statistics query, filled in place.
        if unsafe { D3DKMTQueryStatistics(&raw mut adapter) }.0 < 0 {
            return None;
        }
        // SAFETY: an adapter query fills the adapter's information, integers
        // all (any bits valid).
        let count = unsafe { adapter.QueryResult.AdapterInformation.NodeCount };
        if count > MOST_NODES {
            return None;
        }
        let types: Vec<Option<i32>> = (0..count)
            .map(|node| {
                // The node's number, physical adapter 0 in the high 16 bits.
                let mut meta = [0u8; NODE_METADATA_SIZE];
                meta[..4].copy_from_slice(&node.to_le_bytes());
                ask(KMTQAITYPE_NODEMETADATA, &mut meta).then(|| engine_type(&meta)).flatten()
            })
            .collect();
        engines_from(&types)
    })();
    // SAFETY: the handle opened above, closed once.
    unsafe {
        let _ = D3DKMTCloseAdapter(&D3DKMT_CLOSEADAPTER { hAdapter: open.hAdapter });
    }
    nodes
}

/// This process's running time on each of engines `nodes` of the adapter
/// `luid` names, in 100 ns; none if one cannot be read or is negative.
fn running_times(luid: windows::Win32::Foundation::LUID, nodes: &[u32]) -> Option<Vec<i64>> {
    use windows::Wdk::Graphics::Direct3D::*;
    // SAFETY: the current-process pseudo handle.
    let process = unsafe { windows::Win32::System::Threading::GetCurrentProcess() };
    nodes
        .iter()
        .map(|&node| {
            let mut query = D3DKMT_QUERYSTATISTICS {
                Type: D3DKMT_QUERYSTATISTICS_PROCESS_NODE,
                AdapterLuid: luid,
                hProcess: process,
                Anonymous: D3DKMT_QUERYSTATISTICS_0 {
                    QueryProcessNode: D3DKMT_QUERYSTATISTICS_QUERY_NODE { NodeId: node },
                },
                ..Default::default()
            };
            // SAFETY: a statistics query, filled in place.
            if unsafe { D3DKMTQueryStatistics(&raw mut query) }.0 < 0 {
                return None;
            }
            // SAFETY: a process-node query fills the node's information,
            // integers all (any bits valid).
            let running = unsafe { query.QueryResult.ProcessNodeInformation.RunningTime };
            (running >= 0).then_some(running)
        })
        .collect()
}

/// The process's share of the machine's CPU and its working set, sampled on
/// the caption's thread when figures come (once a second while presenting),
/// never on the present path.
struct Usage {
    cpu_us: u64,
    at: Instant,
}

/// The newest sample: percent of the machine's CPU, working set in MB.
static USAGE: Mutex<Option<(f64, f64)>> = Mutex::new(None);

impl Usage {
    fn new() -> Usage {
        Usage { cpu_us: process_cpu_us().unwrap_or(0), at: Instant::now() }
    }

    /// Samples the CPU time since the last sample and the working set.
    fn sample(&mut self) {
        let (now, at) = (process_cpu_us().unwrap_or(self.cpu_us), Instant::now());
        let wall = at.duration_since(self.at).as_secs_f64().max(1e-3);
        let cpu = now.saturating_sub(self.cpu_us) as f64 / 1e6 / wall / f64::from(logical_processors()) * 100.0;
        let ram = process_working_set().unwrap_or(0) as f64 / (1024.0 * 1024.0);
        (self.cpu_us, self.at) = (now, at);
        *USAGE.lock().unwrap_or_else(|e| e.into_inner()) = Some((cpu, ram));
    }

    /// The newest sample, if the caption's thread has made one.
    fn latest() -> Option<(f64, f64)> {
        *USAGE.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn text(usage: Option<(f64, f64)>) -> String {
        match usage {
            Some((cpu, ram)) => format!("cpu {cpu:.0}% ram {ram:.0} MB"),
            None => "cpu - ram -".to_owned(),
        }
    }
}

/// What `IDXGISwapChain::Present`'s result means for a frame ([`shown`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shown {
    /// Presented (a success code, `DXGI_STATUS_OCCLUDED` among them): the frame counts.
    Yes,
    /// Not presented, the device still there: not counted, said with its code.
    No,
    /// The device is gone: the fatal path submissions take ([`fail`]).
    Removed,
}

/// What `result` means for the frame; `removed`: whether the device's removal reason says it
/// is gone, asked of a failure only.
fn shown(result: windows::core::HRESULT, removed: impl FnOnce() -> bool) -> Shown {
    if result.is_ok() {
        Shown::Yes
    } else if [DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_RESET, DXGI_ERROR_DEVICE_HUNG].contains(&result)
        || removed()
    {
        Shown::Removed
    } else {
        Shown::No
    }
}

/// Frames `Present` did not present, of the run.
static NOT_PRESENTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether the `n`th frame not presented is said: the first, and every hundredth after it.
fn said_not_presented(n: u64) -> bool {
    n % 100 == 1
}

/// The line that says frame `frame` was not presented, the `n`th of the run.
fn not_presented_line(frame: u64, result: windows::core::HRESULT, n: u64) -> String {
    format!(
        "frame {frame} not presented: IDXGISwapChain::Present returned {:#010x}; not counted as a frame; \
         {n} not presented so far (the first and every hundredth after it are logged)",
        result.0 as u32
    )
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on). `None` if nothing was
    /// shown (`Present` failed too: the frame is not counted); else, in
    /// diagnostics builds, whether the frame was black. A device `Present`
    /// finds removed ends the run as a submission's does.
    pub(crate) fn present(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = match self.presenter(width, height, format) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // Presents sent ahead hold back buffers until they have run.
            settle_now(WHY_SYNC);
            // SAFETY: no back buffer is held; every submission was waited for.
            let resized = unsafe {
                p.swap_chain
                    .ResizeBuffers(BUFFERS, width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
            };
            match resized {
                Ok(()) => {
                    (p.width, p.height, p.format) = (width, height, format);
                    tracing::info!(
                        width,
                        height,
                        format = format.0,
                        frame,
                        "the swap chain's buffers were made again for another size or format"
                    );
                    self.log_video_memory("right after the swap chain's buffers were made again");
                }
                Err(e) => {
                    tracing::warn!(error = %e, width, height, "swap chain not resized");
                    return None;
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and a live source texture.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            // Sent ahead ([`late_free_present`]) unless the frame is to be
            // read back at once (a saved or listed frame): then a list of
            // its own, one whose last flight has finished or a new one.
            // A listed frame only with a capture folder: without one nothing
            // is read back (Sol's review, finding 9).
            #[cfg(feature = "diagnostics")]
            let read_now = crate::capture::wanted(frame)
                || crate::capture::folder().is_some() && crate::capture::listed(frame);
            #[cfg(not(feature = "diagnostics"))]
            let read_now = false;
            let slot = if late_free_present() && !read_now {
                let finished = FINISHED.load(Ordering::Acquire);
                p.ahead.iter().position(|(_, flown)| *flown <= finished).or_else(|| {
                    let list = self.new_list().ok()?;
                    list.close();
                    p.ahead.push((list, 0));
                    Some(p.ahead.len() - 1)
                })
            } else {
                None
            };
            let copy = match slot {
                Some(i) => &p.ahead[i].0,
                None => &p.copy,
            };
            let own = ph(Ph::OwnLists);
            copy.reset();
            let list = &copy.list;
            crate::events::begin(list, crate::events::PRESENT);
            list.ResourceBarrier(&[
                transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ),
            ]);
            list.CopyResource(&back, texture);
            // The live panel, when its switch is on: a copy of its box onto the back buffer,
            // still in the copy-destination state this list put it in; its upload buffer is
            // held by this list until it resets (`crate::host::hud`).
            hud::hook(self, copy, &back, format, width, height);
            #[cfg(feature = "diagnostics")]
            let capture = crate::capture::wanted(frame)
                .then(|| crate::capture::read_back(&self.device, list, texture))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = if slot.is_some() { None } else { crate::capture::probe(&self.device, list, texture) };
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            list.ResourceBarrier(&[
                transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                ),
            ]);
            // Its pair counts as a present copy, which ends a frame on the
            // card's clock (`VICTORY_GPU_STAMPS=1`).
            crate::events::end(list);
            #[cfg(feature = "diagnostics")]
            copy.stamps_as(crate::stamps::Kind::Present);
            copy.close();
            own.end();
            match slot {
                Some(i) => {
                    let flown = self.execute_own(copy);
                    p.ahead[i].1 = flown;
                }
                None => self.execute(&[copy]),
            }
            #[cfg(feature = "diagnostics")]
            if let Some(read) = capture {
                crate::capture::save_read_back(frame, &read);
            }
            let presented = {
                let _p = ph(Ph::PresentCall);
                p.swap_chain.Present(0, DXGI_PRESENT(0))
            };
            // SAFETY (the closure): a plain query.
            match shown(presented, || self.device.GetDeviceRemovedReason().is_err()) {
                Shown::Yes => p.meter.frame(p.hwnd),
                Shown::No => {
                    let n = NOT_PRESENTED.fetch_add(1, Ordering::Relaxed) + 1;
                    if said_not_presented(n) {
                        tracing::warn!("{}", not_presented_line(frame, presented, n));
                    }
                    return None;
                }
                Shown::Removed => fail(format_args!(
                    "host GPU device removed at the present of frame {frame}: IDXGISwapChain::Present returned {:#010x}{}",
                    presented.0 as u32,
                    self.removal_data(&[])
                )),
            }
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> windows::core::Result<Presenter> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height)?,
        };
        self.log_video_memory("right before the swap chain is made");
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)?
                .cast::<IDXGISwapChain3>()?;
            let copy = self.new_list()?;
            copy.close();
            tracing::info!(width, height, format = format.0, "presenting to a window");
            self.log_video_memory("right after the swap chain is made");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                ahead: Vec::new(),
                hwnd,
                meter: Meter::new(),
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// The card's memory as the system accounts it to this process, in
    /// bytes: the budget and the use of the card's own memory, then of the
    /// PC's memory shared with the card (where the system moves what does
    /// not fit on the card).
    pub(crate) fn video_memory(&self) -> Option<[u64; 4]> {
        let adapter = self.adapter.as_ref()?;
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: plain queries that fill our structures.
        unsafe {
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut own)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
                .ok()?;
        }
        // The card's budget moved by 256 MB or more since the last look:
        // said at once, with what was in use at the last look, so the log
        // has the moment. (It fell by about a tenth with
        // the first world frame in several runs, and the memory line,
        // seconds later, was the first to show it. We reserve nothing
        // and make the swap chain once, at the start.)
        static LAST: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
        let now = [own.Budget, own.CurrentUsage, shared.CurrentUsage];
        let last = [0, 1, 2].map(|i| LAST[i].swap(now[i], Ordering::Relaxed));
        if last[0] != 0 && last[0].abs_diff(own.Budget) >= (256u64 << 20) {
            tracing::info!(
                "the system changed the budget of the card's memory from {} MB to {} MB (frame {}); at the last look {} MB of the card's and {} MB of the PC's shared memory were in use",
                last[0] >> 20,
                own.Budget >> 20,
                crate::frame::current(),
                last[1] >> 20,
                last[2] >> 20
            );
            self.log_video_memory("right after that change");
        }
        Some([own.Budget, own.CurrentUsage, shared.Budget, shared.CurrentUsage])
    }

    /// Both of the system's memory groups for this process, in bytes: for
    /// the card's own memory and then for the PC's memory shared with the
    /// card, the budget, the use, what could still be reserved and what
    /// is reserved (the layer reserves none).
    pub(crate) fn video_memory_groups(&self) -> Option<[[u64; 4]; 2]> {
        let adapter = self.adapter.as_ref()?;
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: plain queries that fill our structures.
        unsafe {
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut own)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
                .ok()?;
        }
        let all = |m: &DXGI_QUERY_VIDEO_MEMORY_INFO| {
            [m.Budget, m.CurrentUsage, m.AvailableForReservation, m.CurrentReservation]
        };
        Some([all(&own), all(&shared)])
    }

    /// Logs [`Self::video_memory_groups`], with `when` it is said.
    pub(crate) fn log_video_memory(&self, when: &str) {
        let Some([own, shared]) = self.video_memory_groups() else {
            return;
        };
        tracing::info!(
            "video memory as the system counts it, {}: the card's own: budget {} MB, in use {} MB, {} MB could still be reserved, {} MB reserved; the PC's shared with the card: budget {} MB, in use {} MB, {} MB could still be reserved, {} MB reserved (the layer reserves none)",
            when,
            own[0] >> 20,
            own[1] >> 20,
            own[2] >> 20,
            own[3] >> 20,
            shared[0] >> 20,
            shared[1] >> 20,
            shared[2] >> 20,
            shared[3] >> 20
        );
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        MADE[2].fetch_add(1, Ordering::Relaxed);
        MADE[3].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }

    /// A device-local buffer shaders may write (unordered access) and indirect
    /// execution may read, starting in the common state.
    pub(crate) fn uav_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        MADE[4].fetch_add(1, Ordering::Relaxed);
        MADE[5].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a device-local buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no unordered-access buffer");
                None
            }
        }
    }
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    // From just before the command the GPU had not finished: the first
    // twelve after the barrier named draws long done when a list of 79
    // mesh draws stopped at its 45th (run 41).
    for i in from.max(done.saturating_sub(4))..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod trail {
    /// The trail keeps the newest things, and the report made when the
    /// host device is removed says them and the pipeline library's state
    /// (no card is asked for any of it).
    #[test]
    fn the_removal_report_says_the_newest_things_done() {
        // A library call held open, as another test's pipeline work would hold one: the note reads the
        // process-wide calls, so the fixture must not depend on any other test leaving them idle.
        let _call = crate::pipeline::hold_call_for_tests();
        for n in 0..super::TRAIL_MOST + 5 {
            super::trail(|| format!("thing {n} done"));
        }
        let text = super::removal_context();
        assert!(text.contains("the submitting thread was"));
        assert!(
            text.contains("pipeline library:") && text.contains("test of held"),
            "the library's line names the call held open, whatever else is running: {text}"
        );
        assert!(text.contains(&format!("thing {} done", super::TRAIL_MOST + 4)));
        assert!(!text.contains("thing 4 done"), "the oldest entries are dropped");
    }
}

#[cfg(test)]
mod spare {
    use super::Spare;

    /// Kept things come back by size, the last given first, and nothing is
    /// kept past the limit.
    #[test]
    fn kept_things_come_back_by_size_within_the_limit() {
        let mut spare: Spare<u32> = Spare::new();
        assert_eq!(spare.take(4), None);
        assert_eq!(spare.give(1, 4, 10), None);
        assert_eq!(spare.give(2, 1, 10), None);
        assert_eq!(spare.give(3, 4, 10), None);
        assert_eq!(spare.bytes, 9);
        // Past the limit: handed back, nothing changes.
        assert_eq!(spare.give(4, 4, 10), Some(4));
        assert_eq!(spare.bytes, 9);
        assert_eq!(spare.take(2), None, "no thing of that size");
        assert_eq!(spare.take(4), Some(3));
        assert_eq!(spare.take(4), Some(1));
        assert_eq!(spare.take(4), None);
        assert_eq!(spare.take(1), Some(2));
        assert_eq!(spare.bytes, 0);
        assert!(spare.kept.is_empty());
    }
}

#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::refresh::Batch;
    use std::sync::Arc;

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }

    #[test]
    fn vram_idle_ms_scales_with_pressure() {
        use super::vram_idle_ms;
        let gib = 1024 * 1024 * 1024;
        // Normal (< 75%)
        assert_eq!(vram_idle_ms(Some([10 * gib, 0, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 7 * gib, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([100, 74, 0, 0])), 5_000);
        // Moderate (75%..85%)
        assert_eq!(vram_idle_ms(Some([100, 75, 0, 0])), 2_500);
        assert_eq!(vram_idle_ms(Some([100, 84, 0, 0])), 2_500);
        // High (>= 85%, including over-budget)
        assert_eq!(vram_idle_ms(Some([100, 85, 0, 0])), 1_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 12 * gib, 0, 0])), 1_000);
        // Missing or zero budget falls back to normal
        assert_eq!(vram_idle_ms(None), 5_000);
        assert_eq!(vram_idle_ms(Some([0, 100, 0, 0])), 5_000);
    }

    #[test]
    fn ram_idle_ms_scales_with_pressure() {
        use super::ram_idle_ms;
        let gib = 1024 * 1024 * 1024;
        // Normal (>= 4 GiB)
        assert_eq!(ram_idle_ms(Some(8 * gib)), 10_000);
        assert_eq!(ram_idle_ms(Some(4 * gib)), 10_000);
        // Moderate (2 GiB..4 GiB)
        assert_eq!(ram_idle_ms(Some(4 * gib - 1)), 5_000);
        assert_eq!(ram_idle_ms(Some(2 * gib)), 5_000);
        // High (< 2 GiB)
        assert_eq!(ram_idle_ms(Some(2 * gib - 1)), 2_000);
        assert_eq!(ram_idle_ms(Some(0)), 2_000);
        // Missing falls back to normal
        assert_eq!(ram_idle_ms(None), 10_000);
    }
}

#[cfg(test)]
#[path = "host/tests/hash_model.rs"]
mod hash_model_tests;

#[cfg(test)]
#[path = "host/tests/wide.rs"]
mod wide_tests;
