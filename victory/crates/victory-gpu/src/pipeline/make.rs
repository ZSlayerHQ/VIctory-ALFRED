//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs tenths of a second, some driver compiles seconds), and
//! the title's first world frame wants over a thousand new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! The default budget is 12 s per title frame and kind, a window the
//! recording threads wait in together ([`DEFAULT_WAIT_MS`]); it was half
//! a second. A pass the title runs only once and whose pipeline takes
//! longer is lost: while a world loads the title dispatches the passes
//! that unpack its models once. With a budget set below the defaults, a
//! frame may wait longer for a pipeline while a run's loading allowance
//! lasts ([`frame_budget`]), and never for one the driver has held for
//! long ([`held_too_long`]).
//! Ported from the sibling fork `joshuu/victory-amd`'s 4436f8f1 (its
//! loading waits only).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queue_state: AtomicU8,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw or dispatch has requested it yet.
    key: String,
    used: AtomicBool,
    /// Why it is not made, if it is not ([`left_out`]).
    left_out: Option<&'static str>,
    /// When a worker began it ([`slow`]).
    begun: OnceLock<Instant>,
}

/// Why a pipeline with these shaders is not made, if it is not.
fn left_out(shaders: &[Arc<Shader>]) -> Option<&'static str> {
    if without_structures(shaders) {
        Some("shades ray-tracing hits, left out")
    } else if for_structures(shaders) {
        Some("feeds or traces ray-tracing structures, which are not built; left out")
    } else if too_dear(shaders) {
        Some("too dear to compile, left out")
    } else {
        None
    }
}

/// Whether some compute shader of `shaders` holds `name`.
fn named(shaders: &[Arc<Shader>], name: &[u8]) -> bool {
    shaders
        .iter()
        .any(|s| s.stage == "compute" && s.container.windows(name.len()).any(|w| w == name))
}

/// Whether a shader is left out for what it costs the host's driver:
/// `CS_ScreenProbes_ActivateRays` (39f0e9aeb156a61c, a very large
/// program). As translated before the clean-up passes
/// (`emit::shader_opt`), AMD's took a core and ever more memory for it and
/// did not finish; cleaned up, it is slow to compile, once.
/// `VICTORY_ACTIVATE_RAYS=0` leaves it out.
fn too_dear(shaders: &[Arc<Shader>]) -> bool {
    static OUT: OnceLock<bool> = OnceLock::new();
    let out = *OUT.get_or_init(|| std::env::var_os("VICTORY_ACTIVATE_RAYS").is_some_and(|v| v == "0"));
    out && named(shaders, b"CS_ScreenProbes_ActivateRays")
}

/// Whether a shader only shades what the console's ray-tracing structures
/// were hit by: `CS_RaytracingResolve`, bound together early and each
/// costly in AMD's compiler. With the structure builder off there is no hit
/// to shade. Made by default, the picture before loading time (with the
/// builder on, they read the structures it built);
/// `VICTORY_RT_RESOLVE=0` leaves them out while the builder is off.
fn without_structures(shaders: &[Arc<Shader>]) -> bool {
    static OUT: OnceLock<bool> = OnceLock::new();
    let out = *OUT.get_or_init(|| {
        let builder = std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1");
        std::env::var_os("VICTORY_RT_RESOLVE").is_some_and(|v| v == "0") && !builder
    });
    out && named(shaders, b"CS_RaytracingResolve")
}

/// The pipeline names `VICTORY_SKIP_RT_FEED=1` leaves out while the
/// structure builder is off: `CSComputeSkinning*` (every program of that
/// name) and `CS_ScreenProbes_TraceRays_*_BVH_*` (the `Fallback` and
/// `Voxels` tracers stay). They take a good part of the card's time a world
/// frame. Off by default: with the switch unset they are made as before.
const RT_FEED_NAMES: [&str; 2] = ["CSComputeSkinning*", "CS_ScreenProbes_TraceRays*_BVH_*"];

/// Whether `VICTORY_SKIP_RT_FEED=1` asks to leave those pipelines out
/// ([`RT_FEED_NAMES`]); `VICTORY_BVH_BUILDER=1` keeps them, since the
/// structures are built then. Read once, and said once at info.
fn skip_rt_feed() -> bool {
    static OUT: OnceLock<bool> = OnceLock::new();
    *OUT.get_or_init(|| {
        let builder = std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1");
        let skip = std::env::var_os("VICTORY_SKIP_RT_FEED").is_some_and(|v| v == "1") && !builder;
        if skip {
            tracing::info!(
                count = RT_FEED_NAMES.len(),
                names = ?RT_FEED_NAMES,
                "VICTORY_SKIP_RT_FEED leaves these pipelines out while the ray-tracing structure builder is off"
            );
        }
        skip
    })
}

/// Whether a shader only feeds or traces the console's ray-tracing
/// structures: the title's `bvh_skinning` passes and the screen-probe ray
/// tracers that walk them.
fn feeds_structures(shaders: &[Arc<Shader>], skip: bool) -> bool {
    skip && (named(shaders, b"CSComputeSkinning")
        || (named(shaders, b"CS_ScreenProbes_TraceRays") && named(shaders, b"_BVH_")))
}

/// Whether these shaders are left out by the switch ([`skip_rt_feed`]).
fn for_structures(shaders: &[Arc<Shader>]) -> bool {
    feeds_structures(shaders, skip_rt_feed())
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(graphics.as_ref().map(|g| g.describe()).unwrap_or_default().as_bytes());
        let left_out = left_out(&shaders);
        Pending {
            shaders,
            graphics,
            left_out,
            begun: OnceLock::new(),
            queue_state: AtomicU8::new(NOT_QUEUED),
            made: OnceLock::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
        }
    }

    /// Whether it is a compute pipeline.
    fn computes(&self) -> bool {
        matches!(self.shaders.as_slice(), [only] if only.stage == "compute")
    }

    /// Whether the host's driver has been at it for longer than a frame
    /// waits for ([`held_too_long`]).
    fn slow(&self) -> bool {
        self.begun.get().is_some_and(|at| held_too_long(at.elapsed(), self.computes(), wait_budget()))
    }

    fn make(&self) {
        let _ = self.begun.set(Instant::now());
        // Whether a draw or dispatch asked before the build began.
        #[cfg(feature = "diagnostics")]
        let asked = self.used.load(Ordering::Relaxed);
        self.made.get_or_init(|| {
            #[cfg(feature = "diagnostics")]
            let (started, compiles) = (Instant::now(), crate::stalls::compiles_here());
            let made = HostPipeline::new(&self.shaders, self.graphics.as_deref());
            // Counted only where the build ran: not a call that found it
            // built, or waited for another thread's build.
            #[cfg(feature = "diagnostics")]
            crate::stalls::pipeline_made(
                asked,
                crate::stalls::compiles_here() > compiles,
                made.is_none(),
                started.elapsed().as_micros() as u64,
            );
            made
        });
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some(p.left_out.map_or_else(
            || format!("being made {:?}", p.shaders),
            str::to_owned,
        )),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
fn wait_budget() -> Duration {
    #[cfg(test)]
    if let Some(budget) = TEST_BUDGET.get() {
        return budget;
    }
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_PIPELINE_WAIT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_WAIT_MS);
        tracing::info!(millis, "host pipeline wait budget per frame, per kind");
        Duration::from_millis(millis)
    })
}

/// The default of [`wait_budget`]: the picture before frame rate. A frame
/// waits for pipelines still being made rather than drop their draws and
/// dispatches, up to 12 s for each kind (graphics, compute), the threads
/// together ([`Sharing::Window`]). At worst a frame then waits 24 s, both
/// kinds' windows used up one after the other, under the title's GPU
/// watchdog, which ends a run whose frame takes more than 30 s (decision
/// 0007). Only a pipeline still not made after its frame's window loses
/// that frame's draws. `VICTORY_PIPELINE_WAIT_MS=500` waits as before.
const DEFAULT_WAIT_MS: u64 = 12_000;

/// The title's GPU watchdog: one frame of more than this ends the run.
const TITLE_WATCHDOG: Duration = Duration::from_secs(30);

// Both kinds' windows, one after the other, stay under the watchdog.
const _: () = assert!(2 * DEFAULT_WAIT_MS < TITLE_WATCHDOG.as_millis() as u64);

#[cfg(test)]
thread_local! {
    /// A test's own frame budget, on its thread alone.
    static TEST_BUDGET: std::cell::Cell<Option<Duration>> = const { std::cell::Cell::new(None) };
}

/// What a frame may wait for one kind of pipeline while a world loads: they
/// are asked for by the hundred then, and a pass the title runs once is
/// lost if its pipeline is not there. Compute pipelines get more, apart:
/// the passes that unpack a world's models are dispatched once (the
/// sibling fork counted over a hundred compute pipelines dropped while a world loaded,
/// and walls, floors, ceilings and faces missing for the session).
const LOADING_FRAME: Duration = Duration::from_secs(4);
const LOADING_FRAME_COMPUTE: Duration = Duration::from_secs(12);

/// `VICTORY_PIPELINE_GATE_S` (120): the most a run waits past
/// [`wait_budget`] for each kind, in seconds, all frames together; 0 waits
/// as before. The title's GPU watchdog bounds one frame (30 s).
fn loading_allowance() -> Duration {
    static TOTAL: OnceLock<Duration> = OnceLock::new();
    *TOTAL.get_or_init(|| {
        let seconds = std::env::var("VICTORY_PIPELINE_GATE_S")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(120);
        tracing::info!(seconds, "host pipeline loading allowance per kind");
        Duration::from_secs(seconds)
    })
}

/// A frame's budget for a pipeline of one kind: the frame budget, or while
/// the run's allowance for that kind lasts (`spent` of it gone) up to
/// [`LOADING_FRAME`] or [`LOADING_FRAME_COMPUTE`].
fn frame_budget(base: Duration, compute: bool, allowance: Duration, spent: Duration) -> Duration {
    let loading = if compute { LOADING_FRAME_COMPUTE } else { LOADING_FRAME };
    base.max(loading.min(allowance.saturating_sub(spent)))
}

/// Whether the driver has held a pipeline for longer than most take, so no
/// frame waits for it (one compute program of the title takes over a minute): 10 s
/// for a compute pipeline, 3 s for others. Only while the frame budget
/// `base` (`VICTORY_PIPELINE_WAIT_MS`) is shorter than that: a budget set
/// at or above it is the bound alone, so a pipeline still being made is
/// waited for as long as the frame's budget lasts.
fn held_too_long(held: Duration, compute: bool, base: Duration) -> bool {
    let most = Duration::from_secs(if compute { 10 } else { 3 });
    base < most && held > most
}

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if p.left_out.is_some() {
        return None;
    }
    // A still-pending pipeline was requested by the title too. Prewarm it
    // on the next launch, even if this run never got to use it.
    if !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    if p.made.get().is_none() {
        start(p, REQUESTED);
        wait(p);
    }
    p.made.get()?.as_ref()
}

/// Makes `p` on this thread if it is not made yet (waiting for a worker
/// that is making it); whether it has a host pipeline.
pub(crate) fn now(p: &Arc<Pending>) -> bool {
    if p.left_out.is_some() {
        return false;
    }
    p.make();
    p.made.get().is_some_and(|made| made.is_some())
}

/// Translates the containers of compute programs and makes their host
/// pipelines, for the caches on disk; whether it has one (`None`: left out).
pub(crate) fn warm(shaders: Vec<Arc<Shader>>) -> Option<bool> {
    let p = Pending::new(shaders, None);
    if p.left_out.is_some() {
        return None;
    }
    Some(HostPipeline::new(&p.shaders, None).is_some())
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if p.left_out.is_none() && !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, EARLY);
    }
}

/// How the threads that wait in one frame share its budget
/// (`VICTORY_PIPELINE_WAIT_SHARED`): by default a window of waiting time
/// ([`Sharing::Window`]); `reserve` the way before it, `0` the way before
/// that. Counted only when a wait had ended, the budget was one per
/// thread: each of the title's five recording threads began its first
/// wait of a frame with nothing counted yet and waited the whole half
/// second, seconds of waiting in every frame in which pipelines were being
/// made (the frame line's "waiting for pipelines" in each such frame).
/// Reserved, a thread that waited took the rest of the budget, and every
/// other thread's new pipeline in that frame was dropped at once: a run
/// of hundreds of drops as "being made" whose frame lines show at most 9 ms
/// waited (run fork-essentials).
fn shared_budget() -> Sharing {
    static ON: OnceLock<Sharing> = OnceLock::new();
    *ON.get_or_init(|| sharing_from(std::env::var("VICTORY_PIPELINE_WAIT_SHARED").ok().as_deref()))
}

/// How the threads that wait in one frame count its budget for one kind
/// of pipeline (`VICTORY_PIPELINE_WAIT_SHARED`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sharing {
    /// The budget is a window of waiting time: every thread may wait while
    /// the time in which some thread has waited in this frame is shorter,
    /// and nothing is reserved, so no thread drops a pipeline because
    /// another is waiting. The threads' waits together take at most the
    /// budget.
    Window,
    /// `reserve`: a thread that waits takes what is left of the budget and
    /// gives back what it did not use; another thread asking meanwhile
    /// gets nothing.
    Reserve,
    /// `0`: each thread counts its waits once they have ended.
    PerThread,
}

/// The way of counting the switch's value asks for.
fn sharing_from(value: Option<&str>) -> Sharing {
    match value.map(str::trim) {
        Some("0") => Sharing::PerThread,
        Some("reserve") => Sharing::Reserve,
        _ => Sharing::Window,
    }
}

/// One frame's waiting for one kind of pipeline.
#[derive(Debug)]
struct FrameWait {
    frame: u64,
    /// The waiting counted against the budget so far (the window: time in
    /// which some thread waited, up to `since`).
    spent: Duration,
    /// The window: threads waiting now, and since when one has been.
    waiting: u32,
    since: Option<Instant>,
}

impl FrameWait {
    /// Frame 0's, nothing waited yet.
    const fn new() -> FrameWait {
        FrameWait { frame: 0, spent: Duration::ZERO, waiting: 0, since: None }
    }

    /// A thread is to wait in `frame` at `now`: how long it may. Frames
    /// only move on: a thread that comes with a frame older than the one
    /// waited in last gets nothing, and the newer frame keeps what it has
    /// spent (made anew for the older one, the newer frame's next asking
    /// would find a fresh window).
    fn begin(&mut self, sharing: Sharing, frame: u64, budget: Duration, now: Instant) -> Duration {
        if frame < self.frame {
            return Duration::ZERO;
        }
        if frame > self.frame {
            *self = FrameWait { frame, ..FrameWait::new() };
        }
        if sharing == Sharing::Window {
            let open = self.since.map_or(Duration::ZERO, |at| now.saturating_duration_since(at));
            let left = budget.saturating_sub(self.spent + open);
            if !left.is_zero() {
                if self.waiting == 0 {
                    self.since = Some(now);
                }
                self.waiting += 1;
            }
            return left;
        }
        let left = budget.saturating_sub(self.spent);
        if sharing == Sharing::Reserve {
            self.spent += left;
        }
        left
    }

    /// The thread that [`begin`](Self::begin) gave `left` (more than
    /// nothing) in `frame` has waited `waited`, and stops at `now`. A frame
    /// that began meanwhile has its own budget.
    fn end(&mut self, sharing: Sharing, frame: u64, left: Duration, waited: Duration, now: Instant) {
        if self.frame != frame {
            return;
        }
        match sharing {
            Sharing::Window => {
                self.waiting = self.waiting.saturating_sub(1);
                if self.waiting == 0
                    && let Some(at) = self.since.take()
                {
                    self.spent += now.saturating_duration_since(at);
                }
            }
            Sharing::Reserve => self.spent = self.spent.saturating_sub(left - waited),
            Sharing::PerThread => self.spent += waited,
        }
    }
}

/// Lets a thread wait for a pipeline of the kind whose frame slot is
/// `slot`: the frame (`frame`) and the time (`clock`) it waits in, and
/// how long it may. Both are read with the slot locked, so threads come to
/// it in the order of their frames and times: read before, a thread held
/// up between reading and locking came with an older frame, or with a time
/// before the window another thread opened meanwhile, which then counted
/// for nothing.
fn admit(
    slot: &Mutex<FrameWait>,
    sharing: Sharing,
    budget: Duration,
    frame: impl FnOnce() -> u64,
    clock: impl FnOnce() -> Instant,
) -> (u64, Instant, Duration) {
    let mut slot = slot.lock().unwrap_or_else(|e| e.into_inner());
    let (frame, now) = (frame(), clock());
    let left = slot.begin(sharing, frame, budget, now);
    (frame, now, left)
}

/// Waits for `p` while the frame has budget left.
fn wait(p: &Pending) {
    let base = wait_budget();
    if base.is_zero() || p.slow() {
        return;
    }
    // What was waited past the base budget since the start, in
    // milliseconds, by kind (others, compute).
    static PAST: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
    let compute = p.computes();
    let past = &PAST[usize::from(compute)];
    let budget = frame_budget(base, compute, loading_allowance(), Duration::from_millis(past.load(Ordering::Relaxed)));
    // The frame the budget belongs to, and what was waited of it (or is
    // taken by a thread waiting now), by kind.
    static FRAMES: [Mutex<FrameWait>; 2] = [const { Mutex::new(FrameWait::new()) }; 2];
    let frame_wait = &FRAMES[usize::from(compute)];
    let sharing = shared_budget();
    let (frame, started, left) = admit(frame_wait, sharing, budget, crate::frame::current, Instant::now);
    if left.is_zero() {
        return;
    }
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() && !p.slow() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        // Woken now and then to see whether it has become a slow one.
        let rest = rest.min(Duration::from_millis(250));
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    drop(held);
    let waited = started.elapsed().min(left);
    past.fetch_add(waited.saturating_sub(base).as_millis() as u64, Ordering::Relaxed);
    frame_wait.lock().unwrap_or_else(|e| e.into_inner()).end(sharing, frame, left, waited, Instant::now());
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        crate::copystat::C::PipelineWaitUs,
        started.elapsed().as_micros() as u64,
    );
}

const NOT_QUEUED: u8 = 0;
const EARLY: u8 = 1;
const REQUESTED: u8 = 2;
const MAKING: u8 = 3;

struct Queue {
    early: VecDeque<Arc<Pending>>,
    requested: VecDeque<Arc<Pending>>,
}

impl Queue {
    const fn new() -> Self {
        Self {
            early: VecDeque::new(),
            requested: VecDeque::new(),
        }
    }

    /// Queues an unrequested pipeline or promotes one queued at creation.
    /// Returns whether it is new to the queue, or `None` if nothing changed.
    fn push(&mut self, p: &Arc<Pending>, priority: u8) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev >= priority {
            return None;
        }
        p.queue_state.store(priority, Ordering::Relaxed);
        if priority == REQUESTED {
            self.requested.push_back(p.clone());
        } else {
            self.early.push_back(p.clone());
        }
        Some(prev == NOT_QUEUED)
    }

    fn pop(&mut self) -> Option<Arc<Pending>> {
        let p = self.requested.pop_front().or_else(|| {
            // An early entry can have been promoted and claimed already.
            std::iter::from_fn(|| self.early.pop_front())
                .find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY)
        })?;
        p.queue_state.store(MAKING, Ordering::Relaxed);
        Some(p)
    }
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue::new());
static AVAILABLE: Condvar = Condvar::new();

/// Start or promote work for a draw without waiting behind the entire early
/// list. Queue changes and worker claims happen under the same lock.
fn start(p: &Arc<Pending>, priority: u8) {
    if p.queue_state.load(Ordering::Relaxed) >= priority {
        return;
    }
    if !workers() {
        // No worker could be started: the pipeline is made here, on the
        // title's thread, for as long as the driver takes. That is outside
        // the frame's budget ([`wait`] then finds it made), and the frame
        // can outlast the title's watchdog; the README says so (Sol's
        // review of the second landing, follow-up 4).
        if p.queue_state.swap(MAKING, Ordering::Relaxed) == NOT_QUEUED {
            p.make();
        }
        return;
    }
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(new) = queue.push(p, priority) {
        if new {
            PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        }
        AVAILABLE.notify_one();
    }
}

fn next() -> Arc<Pending> {
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(p) = queue.pop() {
            return p;
        }
        queue = AVAILABLE.wait(queue).unwrap_or_else(|e| e.into_inner());
    }
}

/// Starts the workers (half the logical processors, 2 to 8) on first use.
fn workers() -> bool {
    static RUNNING: OnceLock<bool> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        let mut running = false;
        let workers = std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8));
        for i in 0..workers {
            let spawned = std::thread::Builder::new()
                .name(format!("victory-pipelines-{i}"))
                .spawn(|| {
                    // Below the title's own threads (loading, recording).
                    // SAFETY: a plain call on our own thread.
                    unsafe {
                        use windows::Win32::System::Threading::*;
                        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
                    }
                    loop {
                        next().make();
                        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                            drained();
                        } else {
                            super::library::save(false);
                        }
                    }
                });
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "no pipeline worker");
            } else {
                running = true;
            }
        }
        running
    })
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod loading_tests {
    use super::{
        FrameWait, LOADING_FRAME, LOADING_FRAME_COMPUTE, Sharing, admit, frame_budget, held_too_long, sharing_from,
    };
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    /// Two threads ask for pipelines still being made in one frame, the
    /// second 1 ms after the first: by default both may wait, inside one
    /// window of the frame's budget, and neither drops its pipeline
    /// because the other is waiting. The window bounds their waiting
    /// together: once some thread has waited its whole length, a thread
    /// that asks gets nothing. A new frame has a window of its own; a
    /// thread that comes late with an older frame gets nothing and leaves
    /// the newer frame's window as it is.
    #[test]
    fn two_threads_in_one_frame_both_wait_inside_the_window() {
        let sharing = sharing_from(None);
        let (budget, t0, ms) = (Duration::from_secs(12), Instant::now(), Duration::from_millis(1));
        let mut w = FrameWait::new();
        let a = w.begin(sharing, 7, budget, t0);
        let b = w.begin(sharing, 7, budget, t0 + ms);
        assert_eq!(a, budget);
        assert_eq!(b, budget - ms, "the second thread drops its pipeline at once: {sharing:?} gives it {b:?}");
        // Both made after 2 s: 2 s of the window gone.
        w.end(sharing, 7, a, Duration::from_secs(2), t0 + Duration::from_secs(2));
        w.end(sharing, 7, b, Duration::from_secs(2) - ms, t0 + Duration::from_secs(2));
        assert_eq!(w.begin(sharing, 7, budget, t0 + Duration::from_secs(5)), Duration::from_secs(10));
        // That thread waits out the rest: the window is used up.
        w.end(sharing, 7, Duration::from_secs(10), Duration::from_secs(10), t0 + Duration::from_secs(15));
        assert_eq!(w.begin(sharing, 7, budget, t0 + Duration::from_secs(16)), Duration::ZERO);
        // The next frame.
        assert_eq!(w.begin(sharing, 8, budget, t0 + Duration::from_secs(17)), budget);
        // That thread waits out frame 8's window.
        w.end(sharing, 8, budget, budget, t0 + Duration::from_secs(29));
        // A thread that read frame 7 before frame 8 began and was held up
        // until now asks: frame 7 is over, and frame 8 gets no second
        // window (two graphics windows and a compute one would be 36 s,
        // past the title's 30 s watchdog).
        let late = w.begin(sharing, 7, budget, t0 + Duration::from_secs(30));
        assert_eq!(late, Duration::ZERO, "a thread with an older frame waits {late:?}");
        let again = w.begin(sharing, 8, budget, t0 + Duration::from_secs(31));
        assert_eq!(again, Duration::ZERO, "frame 8 waits {again:?} more after its window");
    }

    /// A thread reads the frame it waits in and the time under the lock of
    /// the frame's budget: read before it, a thread held up between the
    /// reading and the lock comes with an older frame or an earlier time,
    /// and the window another thread opened meanwhile counts for nothing.
    #[test]
    fn a_wait_reads_its_frame_and_the_clock_under_the_budget_lock() {
        let slot = Mutex::new(FrameWait::new());
        let (budget, t0) = (Duration::from_secs(4), Instant::now());
        let held = |slot: &Mutex<FrameWait>| slot.try_lock().is_err();
        let (mut frame_held, mut clock_held) = (false, false);
        let frame = || {
            frame_held = held(&slot);
            3
        };
        let clock = || {
            clock_held = held(&slot);
            t0
        };
        assert_eq!(admit(&slot, Sharing::Window, budget, frame, clock), (3, t0, budget));
        assert!(frame_held, "the frame is read with the budget unlocked");
        assert!(clock_held, "the clock is read with the budget unlocked");
    }

    /// `VICTORY_PIPELINE_WAIT_SHARED=reserve`, the default before the
    /// window: the second thread gets nothing while the first waits. `0`:
    /// each thread counts only waits that have ended.
    #[test]
    fn the_old_ways_of_counting_stay_behind_the_switch() {
        let (budget, t0) = (Duration::from_secs(12), Instant::now());
        let mut w = FrameWait::new();
        assert_eq!(w.begin(Sharing::Reserve, 1, budget, t0), budget);
        assert_eq!(w.begin(Sharing::Reserve, 1, budget, t0), Duration::ZERO);
        let mut w = FrameWait::new();
        assert_eq!(w.begin(Sharing::PerThread, 1, budget, t0), budget);
        assert_eq!(w.begin(Sharing::PerThread, 1, budget, t0), budget);
        w.end(Sharing::PerThread, 1, budget, Duration::from_secs(3), t0);
        assert_eq!(w.begin(Sharing::PerThread, 1, budget, t0), Duration::from_secs(9));
        assert_eq!(sharing_from(Some("0")), Sharing::PerThread);
        assert_eq!(sharing_from(Some("reserve")), Sharing::Reserve);
    }

    /// While a run's allowance lasts a frame may wait 4 s for a graphics
    /// pipeline and 12 s for a compute one, then only the frame budget; a
    /// budget set above them stays.
    #[test]
    fn a_frame_waits_longer_while_the_loading_allowance_lasts() {
        let (base, allowance) = (Duration::from_millis(500), Duration::from_secs(120));
        assert_eq!(frame_budget(base, false, allowance, Duration::ZERO), LOADING_FRAME);
        assert_eq!(frame_budget(base, true, allowance, Duration::ZERO), LOADING_FRAME_COMPUTE);
        // 1 s of the allowance left: 1 s.
        assert_eq!(frame_budget(base, true, allowance, Duration::from_secs(119)), Duration::from_secs(1));
        // Spent: the frame budget, as before.
        assert_eq!(frame_budget(base, true, allowance, Duration::from_secs(130)), base);
        assert_eq!(frame_budget(base, false, Duration::ZERO, Duration::ZERO), base, "VICTORY_PIPELINE_GATE_S=0");
        assert_eq!(frame_budget(Duration::from_secs(20), true, allowance, Duration::ZERO), Duration::from_secs(20));
    }

    /// No frame waits for a pipeline the driver has held past 10 s
    /// (compute) or 3 s (others), while the frame budget is shorter; a
    /// budget at or above that is the bound alone.
    #[test]
    fn a_pipeline_held_too_long_is_not_waited_for() {
        let base = Duration::from_millis(500);
        assert!(!held_too_long(Duration::from_secs(10), true, base));
        assert!(held_too_long(Duration::from_millis(10_001), true, base));
        assert!(!held_too_long(Duration::from_secs(3), false, base));
        assert!(held_too_long(Duration::from_millis(3_001), false, base));
        // VICTORY_PIPELINE_WAIT_MS=12000: waited for however long it is held.
        let long = Duration::from_secs(12);
        assert!(!held_too_long(Duration::from_secs(60), true, long));
        assert!(!held_too_long(Duration::from_secs(60), false, long));
        // 5 s: past it for compute pipelines only.
        let between = Duration::from_secs(5);
        assert!(held_too_long(Duration::from_secs(60), true, between));
        assert!(!held_too_long(Duration::from_secs(60), false, between));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    #[test]
    fn pending_pipeline_is_recorded_for_the_next_launch() {
        let mut p = Pending::new(Vec::new(), None);
        p.key = "pending-pipeline-test".to_owned();
        // Simulate a worker that has taken the job but cannot finish it yet.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);
        // No frame budget on this thread (`VICTORY_PIPELINE_WAIT_MS=0`):
        // the request is not waited for, where the default would wait out
        // a 12 s window for a pipeline that is never made.
        super::TEST_BUDGET.set(Some(std::time::Duration::ZERO));

        assert!(!super::super::library::wanted_early(&p.key));
        assert!(ready(&p).is_none());
        assert!(p.made.get().is_none());
        assert!(super::super::library::wanted_early(&p.key));
    }

    #[test]
    fn draws_promote_early_pipelines_without_claiming_them_twice() {
        let earlier = Arc::new(Pending::new(Vec::new(), None));
        let promoted = Arc::new(Pending::new(Vec::new(), None));
        let requested = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();

        assert_eq!(queue.push(&earlier, EARLY), Some(true));
        assert_eq!(queue.push(&promoted, EARLY), Some(true));
        assert_eq!(queue.push(&requested, REQUESTED), Some(true));
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        assert_eq!(queue.push(&promoted, REQUESTED), None);

        assert!(Arc::ptr_eq(&queue.pop().unwrap(), &requested));
        assert!(Arc::ptr_eq(&queue.pop().unwrap(), &promoted));
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        assert!(Arc::ptr_eq(&queue.pop().unwrap(), &earlier));
        assert!(queue.pop().is_none());
    }

    fn shader(stage: &'static str, name: &str) -> Arc<Shader> {
        Arc::new(Shader {
            stage,
            hash: name.to_owned(),
            container: name.as_bytes().to_vec(),
            translated: OnceLock::new(),
        })
    }

    fn one(name: &str) -> Vec<Arc<Shader>> {
        vec![shader("compute", name)]
    }

    /// `VICTORY_SKIP_RT_FEED=1` leaves out exactly the feeder and tracer
    /// names, and nothing else; off, it leaves out nothing.
    #[test]
    fn the_skip_switch_leaves_out_only_the_feeder_and_tracer_names() {
        let left = [
            "CSComputeSkinningCharacters",
            "prefix_CSComputeSkinning_suffix",
            "CS_ScreenProbes_TraceRays_Characters_BVH_Walk",
        ];
        let kept = [
            "CS_RaytracingResolve",
            "CS_ScreenProbes_ActivateRays",
            "CS_ScreenProbes_TraceRays_Fallback",
            "CS_ScreenProbes_TraceRays_Voxels",
            "CS_ScreenProbes_TraceRays_Characters",
            "CS_Unrelated",
        ];
        for name in left {
            assert!(feeds_structures(&one(name), true), "{name}");
            assert!(!feeds_structures(&one(name), false), "{name}");
        }
        for name in kept {
            assert!(!feeds_structures(&one(name), true), "{name}");
            assert!(!feeds_structures(&one(name), false), "{name}");
        }
        // A name in another stage is no compute pipeline.
        let pixel = vec![shader("pixel", "CSComputeSkinningCharacters")];
        assert!(!feeds_structures(&pixel, true));
    }

    /// The switch through the environment, in processes of its own: unset,
    /// the pipelines are made as before; set with the builder off, only the
    /// feeder and tracer names are left out; set with the builder on, none
    /// are.
    #[test]
    fn the_skip_switch_reads_its_environment() {
        for (envs, why) in [
            (vec![], "unset"),
            (vec![("VICTORY_SKIP_RT_FEED", "1")], "set"),
            (vec![("VICTORY_SKIP_RT_FEED", "1"), ("VICTORY_BVH_BUILDER", "1")], "set with the builder"),
        ] {
            let name = "pipeline::make::tests::skip_rt_feed_child";
            let mut command = std::process::Command::new(std::env::current_exe().expect("the test binary"));
            command
                .args(["--exact", name, "--ignored", "--test-threads", "1", "--nocapture"])
                .env_remove("VICTORY_SKIP_RT_FEED")
                .env_remove("VICTORY_BVH_BUILDER")
                .env_remove("VICTORY_RT_RESOLVE")
                .env_remove("VICTORY_ACTIVATE_RAYS");
            for (key, value) in envs {
                command.env(key, value);
            }
            let out = command.output().expect("the child ran");
            let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
            assert!(out.status.success() && text.contains("1 passed"), "{why}: {text}");
        }
    }

    /// The child of the test above: which pipelines the switch leaves out,
    /// and whether the builder being on keeps them.
    #[test]
    #[ignore = "a child process of the test above"]
    fn skip_rt_feed_child() {
        let builder = std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1");
        let skipped = std::env::var_os("VICTORY_SKIP_RT_FEED").is_some_and(|v| v == "1") && !builder;
        let reason = "feeds or traces ray-tracing structures, which are not built; left out";
        for name in ["CSComputeSkinningCharacters", "CS_ScreenProbes_TraceRays_Characters_BVH_Walk"] {
            assert_eq!(left_out(&one(name)), skipped.then_some(reason), "{name}");
        }
        for name in [
            "CS_RaytracingResolve",
            "CS_ScreenProbes_ActivateRays",
            "CS_ScreenProbes_TraceRays_Fallback",
            "CS_ScreenProbes_TraceRays_Voxels",
            "CS_Unrelated",
        ] {
            assert_eq!(left_out(&one(name)), None, "{name}");
        }
        assert_eq!(left_out(&[shader("pixel", "CSComputeSkinningCharacters")]), None);
    }
}
