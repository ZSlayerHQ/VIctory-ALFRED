//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and
//! the title's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! The default budget is half a second per title frame. A pass the title
//! runs only once and whose pipeline takes longer is lost, but for the title's
//! character bakes: they are started at creation and a bake draw waits for its
//! own pipeline, outside the budget, until 10 s after the frame's first bake
//! wait ([`wait_if_bake`], [`BAKE_CEILING`]). Passes over the
//! screen (no vertex input, no depth) have workers and a longer budget of
//! their own ([`screen_budget`]): the title fills its lighting tables with
//! such draws once, at the world's first frame, and behind a world's worth of
//! pipelines they were dropped; that run had no lamps and no diffuse light to
//! its end (a measured run, the first after a new translation; another
//! fork found the same).

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
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
    /// Taken by a worker from [`Queue::slow`].
    was_slow: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw or dispatch has requested it yet.
    key: String,
    used: AtomicBool,
    /// A pass over the screen ([`GraphicsState::screen_pass`]) or a confirmed
    /// opening-scene / bake producer that must never sit behind cold RT shaders.
    screen: bool,
    /// Confirmed first-scene / bake producer prepared early even on a cold cache.
    scene_input: bool,
    /// Why this pipeline is intentionally skipped (e.g. RT resolve without BVH).
    left_out: Option<&'static str>,
    /// What a bake draw's wait for it has logged ([`BAKE_SAID`] ...).
    bake: AtomicU8,
    /// The milliseconds bake draws have waited for it, in all.
    bake_waited: AtomicU64,
    /// Tests: what its making gives instead of [`HostPipeline::new`], once
    /// the delay has passed on the worker making it ([`Pending::held_back`]).
    #[cfg(test)]
    held: Mutex<Option<(Duration, Option<HostPipeline>)>>,
}

fn named(shaders: &[Arc<Shader>], name: &[u8]) -> bool {
    shaders
        .iter()
        .any(|s| s.stage == "compute" && s.container.windows(name.len()).any(|w| w == name))
}

/// Shader hashes whose pipelines are left out in every build
/// (`VICTORY_LEAVE_OUT=<16 hex digits>[,..]`; `VICTORY_SKIP_PIPELINES` does
/// the same in diagnostics builds only). The decal heap's
/// `CS_DefragHeap_ComputeRelocations` (02384a60fda6f2d0), translatable for
/// 32-lane hosts through host waves that meet by spinning, took 11.85 ms of
/// the card a frame in the first gameplay area (timing run, 2026-11-26).
fn leave_out() -> &'static [String] {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    HASHES.get_or_init(|| {
        std::env::var("VICTORY_LEAVE_OUT")
            .unwrap_or_default()
            .split(',')
            .map(|h| h.trim().to_ascii_lowercase())
            .filter(|h| h.len() == 16 && h.bytes().all(|b| b.is_ascii_hexdigit()))
            .collect()
    })
}

/// The light in the water's volume, in tiles (`water_shading:CS_TransmissionVolumetrics_Tiled`
/// and `_Tiled_LocalLights`): programs of one 64-lane wave a group, which a
/// host of 32-lane waves gets in the wave-wide form only (`victory_shader::emit`, 4 and 7
/// branches taken wave-wide). In that form they leave whole tiles of open water dark, around
/// boats, along shores and under bridges, and take about 15 ms of the card a frame (RTX 5070,
/// measured runs of 2026-11-26: 25 fps with them, 30 without, the blocks gone;
/// another fork's notes of 2026-11-27 name the same two programs and form). Left
/// out there, the water has no light of its own depth (darker, bluer) and no blocks.
/// foxhound 0.2.0-pre.3 fixed the blocks (64-lane programs on 32-lane waves:
/// `VICTORY_LOOPS_WAVE`, `VICTORY_LDS_FIRST_INIT`), so they run now unless
/// `VICTORY_WATER_LIGHT=0` (an empty value, an earlier version's saved OFF, runs them too).
const WATER_LIGHT: [&str; 2] = ["5dc3b25986622949", "6f225b641c5b0708"];

/// Whether this start leaves the water's tiled light out ([`WATER_LIGHT`]; the F1 menu asks too).
pub(super) fn water_light_left_out_here() -> bool {
    static ASKED: OnceLock<bool> = OnceLock::new();
    *ASKED.get_or_init(|| std::env::var_os("VICTORY_WATER_LIGHT").is_some_and(|v| v == "0"))
        && victory_shader::emit::host_settings()[0] == 32
}

fn water_light_left_out(shaders: &[Arc<Shader>]) -> bool {
    water_light_left_out_here() && shaders.iter().any(|s| WATER_LIGHT.contains(&s.hash.as_str()))
}

/// Whether this start leaves the ray-tracing resolve out (the F1 menu asks too).
pub(super) fn rt_resolve_left_out() -> bool {
    static SKIP_RT_RESOLVE: OnceLock<bool> = OnceLock::new();
    *SKIP_RT_RESOLVE.get_or_init(|| {
        let builder = std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1");
        if builder {
            return false;
        }
        // Without BVH, ~1,000 CS_RaytracingResolve shaders shade no hits while
        // clogging the cold compiler queue; skip unless VICTORY_RT_RESOLVE=1.
        std::env::var_os("VICTORY_RT_RESOLVE").is_none_or(|v| v == "0")
    })
}

fn left_out(shaders: &[Arc<Shader>]) -> Option<&'static str> {
    if !leave_out().is_empty() && shaders.iter().any(|s| leave_out().contains(&s.hash)) {
        return Some("named in VICTORY_LEAVE_OUT, left out");
    }
    if water_light_left_out(shaders) {
        return Some("the water's tiled light on a host of 32-lane waves, left out (VICTORY_WATER_LIGHT=0)");
    }
    if rt_resolve_left_out() && named(shaders, b"CS_RaytracingResolve") {
        return Some("shades ray-tracing hits without BVH, left out");
    }
    // The screen probes' ray activation runs without the structure builder
    // too: left out, the probes kept stale values and surfaces lit mostly by
    // them took a magenta or violet cast (papers on a wall, irises), at no
    // frame-rate gain (bisected and measured by a tester, issue #2 of the
    // public repository, 2026-11-27). `VICTORY_ACTIVATE_RAYS=0`: left out,
    // as before.
    static SKIP_ACTIVATE_RAYS: OnceLock<bool> = OnceLock::new();
    let skip_activate =
        *SKIP_ACTIVATE_RAYS.get_or_init(|| std::env::var_os("VICTORY_ACTIVATE_RAYS").is_some_and(|v| v == "0"));
    if skip_activate && named(shaders, b"CS_ScreenProbes_ActivateRays") {
        return Some("ray activation shader without BVH, left out");
    }
    None
}

/// Confirmed first-scene producers. Prepare these at creation even on a cold
/// launch so cutscene lighting, tiling, and bakes never miss their first frame.
/// Every bake is one ([`is_bake`]): its bake geometry stage too, not only the
/// seven known pixel shaders (`VICTORY_BAKE_WAIT=0`: those alone, as before).
fn scene_input(shaders: &[Arc<Shader>]) -> bool {
    shaders.iter().any(|s| {
        matches!(
            s.hash.as_str(),
            "970c4fbb3b017c0e"
                | "068c90badc86f1fd"
                | "e4a7e088c7780ce8"
                | "ebe4e53f031f5249"
                | "2cc4c197ee29e291"
                | "369cbe906ea72764"
        ) || crate::draw::BAKE_PIXEL_SHADERS.contains(&s.hash.as_str())
    }) || (bake_wait() && is_bake(shaders))
}

/// Whether a pipeline of `shaders` is one of the title's character or
/// decoration bakes by its shaders (`crate::draw::bake_shader`): known before
/// its host pipeline is made.
fn is_bake(shaders: &[Arc<Shader>]) -> bool {
    shaders.iter().any(|s| crate::draw::bake_shader(&s.hash))
}

/// Whether `VICTORY_BAKE_WAIT`'s value leaves it on: on unless `0`.
fn bake_wait_from(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_none_or(|v| v != "0")
}

/// Whether a bake draw waits for its own pipeline ([`wait_if_bake`]) and
/// every bake starts at creation ([`scene_input`]). The title bakes each
/// character's skin into a texture of its own once, while it loads: in 3 of
/// 8 cold starts (an empty pipeline library) the player's head had no skin
/// for the whole run, its bake dropped while its pipeline was being made
/// ("dropped 6788 without a host pipeline (1057 being made)" at the bakes),
/// and the alpha test removed the whole head. On by default;
/// `VICTORY_BAKE_WAIT=0`: dropped as before.
fn bake_wait() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_BAKE_WAIT.get() {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| bake_wait_from(std::env::var_os("VICTORY_BAKE_WAIT").as_deref()))
}

/// How long the bake waits of one frame take at most, all together: the
/// frame's deadline is this long after its first bake wait began
/// ([`bake_deadline`]), and the ordinary waits after it in that frame end
/// there too ([`wait`]). Several bakes in one frame, or an ordinary pass the
/// structural rule takes for one, stop the title's recording for 10 s at
/// most, a third of the title's 30 s watchdog. It is also the most one
/// pipeline is waited for over all frames ([`BAKE_CAPPED`]).
const BAKE_CEILING: Duration = Duration::from_secs(10);

/// The frame of the latest bake wait and its deadline ([`BAKE_CEILING`]).
static BAKE_DEADLINE: Mutex<Option<(u64, Instant)>> = Mutex::new(None);

/// The deadline of frame `frame`'s bake waits in `slot`: set by the first,
/// `ceiling` from now.
fn deadline_in(slot: &Mutex<Option<(u64, Instant)>>, frame: u64, ceiling: Duration) -> Instant {
    let mut deadline = slot.lock().unwrap_or_else(|e| e.into_inner());
    match *deadline {
        Some((f, until)) if f == frame => until,
        _ => {
            let until = Instant::now() + ceiling;
            *deadline = Some((frame, until));
            until
        }
    }
}

/// The deadline of frame `frame` in `slot`, if a bake wait of it set one.
fn deadline_of(slot: &Mutex<Option<(u64, Instant)>>, frame: u64) -> Option<Instant> {
    match *slot.lock().unwrap_or_else(|e| e.into_inner()) {
        Some((f, until)) if f == frame => Some(until),
        _ => None,
    }
}

/// [`deadline_in`] of the title's present frame.
fn bake_deadline() -> Instant {
    deadline_in(&BAKE_DEADLINE, this_frame(), BAKE_CEILING)
}

/// The title's frame waits are counted by.
fn this_frame() -> u64 {
    #[cfg(test)]
    if let Some(frame) = TEST_FRAME.get() {
        return frame;
    }
    crate::frame::current()
}

#[cfg(test)]
thread_local! {
    /// Tests, on their own thread: the frame waits count by, so a test's
    /// deadline is its own; `VICTORY_BAKE_WAIT` as if set; no ordinary wait
    /// (both budgets zero), so only the bake wait can find a pipeline made.
    static TEST_FRAME: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
    static TEST_BAKE_WAIT: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static TEST_NO_BUDGET: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Tests: this thread's frame, `VICTORY_BAKE_WAIT` (`None`: the variable's),
/// and whether ordinary waits are none ([`TEST_FRAME`] ...).
#[cfg(test)]
pub(crate) fn test_settings(frame: Option<u64>, bake_wait: Option<bool>, no_ordinary_wait: bool) {
    TEST_FRAME.set(frame);
    TEST_BAKE_WAIT.set(bake_wait);
    TEST_NO_BUDGET.set(no_ordinary_wait);
}

/// [`Pending::bake`]: the wait was logged; it was waited for [`BAKE_CEILING`]
/// in all (later draws do not wait again); its failure was logged; a frame's
/// deadline passed before it was made.
const BAKE_SAID: u8 = 1;
const BAKE_CAPPED: u8 = 2;
const BAKE_FAILED: u8 = 4;
const BAKE_LATE: u8 = 8;

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
        let scene_input = scene_input(&shaders);
        let screen = (graphics.as_ref().is_some_and(|g| g.screen_pass())
            && shaders.iter().all(|s| s.stage != "mesh"))
            || scene_input;
        let left_out = left_out(&shaders);
        Pending {
            screen,
            scene_input,
            left_out,
            shaders,
            graphics,
            queue_state: AtomicU8::new(NOT_QUEUED),
            was_slow: AtomicBool::new(false),
            made: OnceLock::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
            bake: AtomicU8::new(0),
            bake_waited: AtomicU64::new(0),
            #[cfg(test)]
            held: Mutex::new(None),
        }
    }

    /// Tests: a pending pipeline of `shaders` whose making takes `delay` and
    /// gives `made`, a host pipeline made by the test itself.
    #[cfg(test)]
    pub(crate) fn held_back(shaders: Vec<Arc<Shader>>, delay: Duration, made: HostPipeline) -> Pending {
        let p = Pending::new(shaders, None);
        *p.held.lock().unwrap() = Some((delay, Some(made)));
        p
    }

    pub(super) fn key(&self) -> &str {
        &self.key
    }

    fn make(&self) {
        self.make_with(|| {
            // Tests: held back, then the test's own pipeline (inside the catch, as a real build).
            #[cfg(test)]
            if let Some((delay, made)) = self.held.lock().unwrap_or_else(|e| e.into_inner()).take() {
                std::thread::sleep(delay);
                return made;
            }
            let began = Instant::now();
            let made = HostPipeline::new(&self.shaders, self.graphics.as_deref());
            super::library::note_time(&self.key, began.elapsed().as_millis() as u64);
            made
        });
    }

    fn make_with(&self, build: impl FnOnce() -> Option<HostPipeline>) {
        self.made.get_or_init(|| {
            // A failed translation must complete this queue entry and wake its waiters:
            // a panic left `made` empty for ever and the worker thread gone with it.
            match catch_unwind(AssertUnwindSafe(build)) {
                Ok(pipeline) => pipeline,
                Err(_) => {
                    tracing::warn!(shaders = ?self.shaders, "host pipeline build panicked");
                    None
                }
            }
        });
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// Why a pipeline has no host pipeline (diagnostics): left out on purpose
/// ([`left_out`]), its shaders that do not translate, else "refused" or
/// "being made"; `None` if it has one. (A pipeline left out is never made:
/// it was counted as "being made" in every frame, some 630 indirect
/// ray-tracing resolves a frame in gameplay, while no pipeline was queued.)
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    if let Some(why) = p.left_out {
        return Some(format!("left out on purpose: {why}"));
    }
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
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

/// The hashes of the shaders `p` is made of.
pub(crate) fn hashes(p: &Pending) -> String {
    let hashes: Vec<&str> = p.shaders.iter().map(|s| s.hash.as_str()).collect();
    hashes.join("+")
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
fn wait_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_PIPELINE_WAIT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(500);
        tracing::info!(millis, "host pipeline wait budget per frame");
        Duration::from_millis(millis)
    })
}

/// As [`wait_budget`], for passes over the screen (`VICTORY_SCREEN_WAIT_MS`,
/// 10 s; 0: like the others). Below the title's 30 s watchdog.
fn screen_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_SCREEN_WAIT_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(10_000);
        Duration::from_millis(millis)
    })
}

/// Whether passes over the screen go before the others.
fn screens_first() -> bool {
    !screen_budget().is_zero()
}

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Draws and dispatches (direct and indirect) asked of a pipeline left out on
/// purpose ([`left_out`]), since the last [`take_left_out`].
static LEFT_OUT: AtomicU64 = AtomicU64::new(0);

/// See [`LEFT_OUT`] (for the frame line).
pub(crate) fn take_left_out() -> u64 {
    LEFT_OUT.swap(0, Ordering::Relaxed)
}

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if p.left_out.is_some() {
        LEFT_OUT.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    // A still-pending pipeline was requested by the title too. Prewarm it
    // on the next launch, even if this run never got to use it.
    if !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    if p.made.get().is_none() {
        start(p, REQUESTED);
        wait(p, this_frame());
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

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it),
/// or if it is a confirmed opening-scene / bake producer.
pub(crate) fn early(p: &Arc<Pending>) {
    if p.left_out.is_some() {
        return;
    }
    if !p.shaders.is_empty() && (p.scene_input || super::library::wanted_early(&p.key)) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, if p.scene_input { REQUESTED } else { EARLY });
    }
}

/// Whether the frame's budget is one for all threads together: a thread
/// that waits takes what is left of it and gives back what it did not
/// use. Counted only when a wait had ended, the budget was one per thread:
/// each of the title's five recording threads began its first wait of a
/// frame with nothing counted yet and waited the whole half second, 2.5 s
/// of waiting in every frame in which pipelines were being made (runs
/// 34-49: "2504 ms waiting for pipelines" in each such frame).
/// On by default; `VICTORY_PIPELINE_WAIT_SHARED=0` counts as before.
fn shared_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_SHARED").is_none_or(|v| v != "0"))
}

/// Waits for `p` while frame `frame` has budget left, and not past the
/// deadline of the frame's bake waits if one began ([`BAKE_CEILING`]).
fn wait(p: &Pending, frame: u64) {
    wait_in(p, frame, &BAKE_DEADLINE);
}

/// [`wait`], with the bake deadlines in `deadlines`.
fn wait_in(p: &Pending, frame: u64, deadlines: &Mutex<Option<(u64, Instant)>>) {
    #[cfg(test)]
    if TEST_NO_BUDGET.get() {
        return;
    }
    let screen = p.screen && screens_first();
    let budget = if screen { screen_budget() } else { wait_budget() };
    if budget.is_zero() {
        return;
    }
    let until_deadline = deadline_of(deadlines, frame).map(|until| until.saturating_duration_since(Instant::now()));
    // The frame the budget belongs to, and what was waited of it (or is
    // taken by a thread waiting now); of the passes over the screen too.
    static SPENT: [Mutex<(u64, Duration)>; 2] = [const { Mutex::new((u64::MAX, Duration::ZERO)) }; 2];
    let spent_of = &SPENT[usize::from(screen)];
    let shared = shared_budget();
    let left = {
        let mut spent = spent_of.lock().unwrap_or_else(|e| e.into_inner());
        if spent.0 != frame {
            *spent = (frame, Duration::ZERO);
        }
        let left = budget.saturating_sub(spent.1).min(until_deadline.unwrap_or(Duration::MAX));
        if shared {
            spent.1 += left;
        }
        left
    };
    if left.is_zero() {
        return;
    }
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    drop(held);
    let waited = started.elapsed().min(left);
    if screen && waited > Duration::from_millis(500) {
        tracing::info!(
            ms = waited.as_millis() as u64,
            made = p.made.get().is_some(),
            shaders = ?p.shaders,
            "a pass over the screen was waited for (VICTORY_SCREEN_WAIT_MS=0: dropped past half a second a frame)"
        );
    }
    {
        let mut spent = spent_of.lock().unwrap_or_else(|e| e.into_inner());
        // A frame that began meanwhile has its own budget.
        if spent.0 == frame {
            if shared {
                spent.1 = spent.1.saturating_sub(left - waited);
            } else {
                spent.1 += waited;
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        crate::copystat::C::PipelineWaitUs,
        started.elapsed().as_micros() as u64,
    );
}

/// Before a draw with `p`: if it is not made yet and the draw is one of the
/// title's bakes (by `p`'s shaders, [`is_bake`], else `by_targets`), waits
/// for it ([`wait_for_bake`]); a bake whose pipeline failed to make is
/// named once. Nothing with `VICTORY_BAKE_WAIT=0` ([`bake_wait`]).
pub(crate) fn wait_if_bake(p: &Arc<Pending>, by_targets: impl FnOnce() -> bool) {
    if matches!(p.made.get(), Some(Some(_))) || p.left_out.is_some() || !bake_wait() {
        return;
    }
    if !is_bake(&p.shaders) && !by_targets() {
        return;
    }
    wait_for_bake(p, bake_deadline, BAKE_CEILING);
}

/// Waits for `p`, a bake's pipeline, to be made, until `deadline` (the
/// frame's, shared by all its bake waits: [`BAKE_CEILING`]), with no budget
/// of the frame's: a bake the title draws once and that is dropped stays
/// missing for the run. Moves `p` to the front of the queue first. A pipeline
/// waited for `cap` in all is not waited for again.
///
/// What this does not save (as before): a bake whose pipeline is still not
/// made at the deadline is dropped, and the draw is not replayed when the
/// pipeline is made later; so is a bake whose pipeline failed to make (named
/// once). And with no worker thread, [`hurry`] makes the pipeline on this
/// thread, outside any deadline.
fn wait_for_bake(p: &Arc<Pending>, deadline: impl FnOnce() -> Instant, cap: Duration) {
    if p.made.get().is_none() && p.bake.load(Ordering::Relaxed) & BAKE_CAPPED == 0 {
        let until = deadline();
        let started = Instant::now();
        hurry(p);
        let (lock, done) = &DONE;
        let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
        while p.made.get().is_none() {
            let rest = until.saturating_duration_since(Instant::now());
            if rest.is_zero() {
                break;
            }
            held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
        }
        drop(held);
        let ms = started.elapsed().as_millis() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::PipelineWaitUs, started.elapsed().as_micros() as u64);
        if p.made.get().is_none() {
            let all = p.bake_waited.fetch_add(ms, Ordering::Relaxed) + ms;
            if all >= cap.as_millis() as u64 {
                if p.bake.fetch_or(BAKE_CAPPED, Ordering::Relaxed) & BAKE_CAPPED == 0 {
                    tracing::warn!(
                        ms = all,
                        shaders = %hashes(p),
                        "a character bake's pipeline was not made in {all} ms of waits: its draws are dropped and not waited for again"
                    );
                }
            } else if p.bake.fetch_or(BAKE_LATE, Ordering::Relaxed) & BAKE_LATE == 0 {
                tracing::warn!(
                    ms,
                    shaders = %hashes(p),
                    "a character bake's pipeline was not made by the frame's deadline ({} ms for all bake waits of a frame): this draw is dropped",
                    cap.as_millis()
                );
            }
            return;
        }
        if matches!(p.made.get(), Some(Some(_))) && p.bake.fetch_or(BAKE_SAID, Ordering::Relaxed) & BAKE_SAID == 0 {
            tracing::info!(
                ms,
                shaders = %hashes(p),
                "a character bake waited {ms} ms for its pipeline (VICTORY_BAKE_WAIT=0: dropped)"
            );
        }
    }
    if matches!(p.made.get(), Some(None)) && p.bake.fetch_or(BAKE_FAILED, Ordering::Relaxed) & BAKE_FAILED == 0 {
        tracing::warn!(shaders = %hashes(p), "a character bake's pipeline could not be made: its draws are dropped");
    }
}

/// Moves `p` to the front of the queue, before every other, if no worker is
/// making it yet. With no workers (none of their threads started) it makes
/// it here, on the title's recording thread, as before: that compile is not
/// bounded by the frame's deadline ([`BAKE_CEILING`]).
fn hurry(p: &Arc<Pending>) {
    if !workers() {
        start(p, REQUESTED);
        return;
    }
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(new) = queue.put_first(p) {
        if new {
            PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        }
        AVAILABLE.notify_all();
    }
}

const NOT_QUEUED: u8 = 0;
const EARLY: u8 = 1;
const REQUESTED: u8 = 2;
const MAKING: u8 = 3;

/// How many workers may make pipelines at once that are known to take long
/// and that no draw has asked for: a quarter of them. On a fresh cache (a
/// change of the translator or the driver) the early list begins with
/// programs that take tens of seconds each: all 16 workers were in them
/// through the whole of the title's loading, 97 pipelines made in 20 s, and
/// the first minute and a half of the world ran at 0.5 to 1.6 s a frame,
/// dropping up to 2,900 draws a frame (a measured run).
/// `VICTORY_PIPELINE_SLOW=0`: in the order the title creates them, as before.
fn slow_limit() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        if std::env::var_os("VICTORY_PIPELINE_SLOW").is_some_and(|v| v == "0") {
            usize::MAX
        } else {
            (worker_count(None, std::thread::available_parallelism().map_or(2, |n| n.get())) / 4).max(1)
        }
    })
}

struct Queue {
    /// Requested passes over the screen: before all others, and the only ones
    /// [`SCREEN_WORKERS`] take.
    screen: VecDeque<Arc<Pending>>,
    early: VecDeque<Arc<Pending>>,
    /// Those of the early ones known to take long, and how many of them are
    /// being made ([`slow_limit`]).
    slow: VecDeque<Arc<Pending>>,
    slow_now: usize,
    requested: VecDeque<Arc<Pending>>,
}

impl Queue {
    const fn new() -> Self {
        Self {
            screen: VecDeque::new(),
            early: VecDeque::new(),
            slow: VecDeque::new(),
            slow_now: 0,
            requested: VecDeque::new(),
        }
    }

    /// Queues an unrequested pipeline or promotes one queued at creation.
    /// Returns whether it is new to the queue, or `None` if nothing changed.
    fn push(&mut self, p: &Arc<Pending>, priority: u8) -> Option<bool> {
        self.push_as(p, priority, slow_limit() != usize::MAX && super::library::slow(&p.key))
    }

    fn push_as(&mut self, p: &Arc<Pending>, priority: u8, slow: bool) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev >= priority {
            return None;
        }
        p.queue_state.store(priority, Ordering::Relaxed);
        if priority == REQUESTED && p.screen && screens_first() {
            self.screen.push_back(p.clone());
        } else if priority == REQUESTED {
            self.requested.push_back(p.clone());
        } else if slow {
            self.slow.push_back(p.clone());
        } else {
            self.early.push_back(p.clone());
        }
        Some(prev == NOT_QUEUED)
    }

    /// Puts `p` before every other pipeline, out of whichever list held it
    /// (requested, on the list workers take first; once only, so it is made
    /// and counted once). Returns whether it is new to the queue, or `None`
    /// if a worker is making it already.
    fn put_first(&mut self, p: &Arc<Pending>) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev == MAKING {
            return None;
        }
        if prev != NOT_QUEUED {
            for list in [&mut self.screen, &mut self.requested, &mut self.early, &mut self.slow] {
                list.retain(|q| !Arc::ptr_eq(q, p));
            }
        }
        p.queue_state.store(REQUESTED, Ordering::Relaxed);
        if screens_first() {
            self.screen.push_front(p.clone());
        } else {
            self.requested.push_front(p.clone());
        }
        Some(prev == NOT_QUEUED)
    }

    fn pop(&mut self, screens_only: bool, slow_limit: usize) -> Option<Arc<Pending>> {
        // An early entry can have been promoted and claimed already.
        let unclaimed = |list: &mut VecDeque<Arc<Pending>>| {
            std::iter::from_fn(|| list.pop_front()).find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY)
        };
        let mut p = self.screen.pop_front();
        if p.is_none() && !screens_only {
            p = self.requested.pop_front().or_else(|| unclaimed(&mut self.early));
            if p.is_none() && self.slow_now < slow_limit {
                p = unclaimed(&mut self.slow);
                if let Some(p) = &p {
                    p.was_slow.store(true, Ordering::Relaxed);
                    self.slow_now += 1;
                }
            }
        }
        let p = p?;
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
        // (One woken could be a worker that takes passes over the screen
        // alone: it would leave the pipeline to nobody.)
        AVAILABLE.notify_all();
    }
}

/// Workers that make passes over the screen alone, besides the others.
const SCREEN_WORKERS: usize = 2;

fn next(screens_only: bool) -> Arc<Pending> {
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(p) = queue.pop(screens_only, slow_limit()) {
            return p;
        }
        queue = AVAILABLE.wait(queue).unwrap_or_else(|e| e.into_inner());
    }
}

/// How many workers make pipelines: `asked` (`VICTORY_PIPELINE_WORKERS`, 1 to
/// 64), else two thirds of the logical processors, 2 to 16. They run below the
/// title's own threads, and a cold cache queues hundreds at once.
fn worker_count(asked: Option<usize>, processors: usize) -> usize {
    worker_count_for(asked, processors, false)
}

fn worker_count_for(asked: Option<usize>, processors: usize, limited_ram: bool) -> usize {
    match asked {
        Some(n) => n.clamp(1, 64),
        None if limited_ram => (processors * 3 / 4).clamp(2, 8),
        None => (processors * 2 / 3).clamp(2, 16),
    }
}

/// Starts the workers ([`worker_count`]) on first use.
fn workers() -> bool {
    static RUNNING: OnceLock<bool> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        let mut running = false;
        let workers = worker_count_for(
            std::env::var("VICTORY_PIPELINE_WORKERS")
                .ok()
                .and_then(|v| v.trim().parse().ok()),
            std::thread::available_parallelism().map_or(2, |n| n.get()),
            crate::residency::limited_host_ram(),
        );
        tracing::info!(workers, "pipeline workers");
        let screens = if screens_first() { SCREEN_WORKERS } else { 0 };
        for i in 0..workers + screens {
            let screens_only = i >= workers;
            let spawned = std::thread::Builder::new()
                .name(format!("victory-pipelines-{i}"))
                .spawn(move || {
                    // Below the title's own threads (loading, recording),
                    // and the streaming one, which is below normal itself
                    // (`VICTORY_PIPELINE_LOWEST=0`: level with it, as before).
                    let lowest = !screens_only && std::env::var_os("VICTORY_PIPELINE_LOWEST").is_none_or(|v| v != "0");
                    // SAFETY: a plain call on our own thread.
                    unsafe {
                        use windows::Win32::System::Threading::*;
                        let priority = if lowest { THREAD_PRIORITY_LOWEST } else { THREAD_PRIORITY_BELOW_NORMAL };
                        let _ = SetThreadPriority(GetCurrentThread(), priority);
                    }
                    loop {
                        let p = next(screens_only);
                        p.make();
                        if p.was_slow.load(Ordering::Relaxed) {
                            let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
                            queue.slow_now -= 1;
                            AVAILABLE.notify_all();
                        }
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
mod tests {
    use super::*;

    #[test]
    fn worker_count_follows_the_processors_and_the_switch() {
        assert_eq!(worker_count(None, 1), 2);
        assert_eq!(worker_count(None, 24), 16);
        assert_eq!(worker_count(None, 64), 16);
        assert_eq!(worker_count(Some(0), 24), 1);
        assert_eq!(worker_count(Some(20), 24), 20);
        assert_eq!(worker_count_for(None, 8, true), 6);
        assert_eq!(worker_count_for(None, 16, true), 8);
        assert_eq!(worker_count_for(None, 24, true), 8);
        assert_eq!(worker_count_for(Some(12), 8, true), 12);
    }

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

    /// A panic inside a build must not take the worker thread with it, nor
    /// leave the queue entry open for ever.
    #[test]
    fn a_panicking_pipeline_is_refused_and_the_worker_continues() {
        let bad = Arc::new(Pending::new(Vec::new(), None));
        let good = Arc::new(Pending::new(Vec::new(), None));
        let failed = bad.clone();
        let next = good.clone();
        let worker = std::thread::spawn(move || {
            failed.make_with(|| panic!("synthetic pipeline build panic"));
            next.make_with(|| None);
        });

        worker.join().expect("the pipeline worker should survive");
        assert!(matches!(bad.made.get(), Some(None)));
        assert!(matches!(good.made.get(), Some(None)));

        bad.make_with(|| panic!("the refused pipeline should stay complete"));
        assert!(matches!(bad.made.get(), Some(None)));
    }

    /// A pipeline left out on purpose (a ray-tracing resolve without the structure builder)
    /// is never queued, so never made: it is counted as left out, and named so, not as
    /// "being made" (some 630 indirect dispatches a frame were, in every gameplay frame).
    #[test]
    fn pipelines_left_out_are_counted_and_never_queued() {
        let mut p = Pending::new(Vec::new(), None);
        p.left_out = Some("shades ray-tracing hits without BVH, left out");
        let p = Arc::new(p);
        let before = LEFT_OUT.load(Ordering::Relaxed);
        assert!(ready(&p).is_none());
        assert!(ready(&p).is_none());
        assert!(!now(&p));
        assert_eq!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        assert!(p.made.get().is_none());
        assert!(LEFT_OUT.load(Ordering::Relaxed) >= before + 2);
        #[cfg(feature = "diagnostics")]
        assert_eq!(
            missing(&p).as_deref(),
            Some("left out on purpose: shades ray-tracing hits without BVH, left out")
        );
    }

    #[test]
    fn pending_pipeline_is_recorded_for_the_next_launch() {
        let mut p = Pending::new(Vec::new(), None);
        p.key = "pending-pipeline-test".to_owned();
        // Simulate a worker that has taken the job but cannot finish it yet.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);

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

        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &requested));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &promoted));
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &earlier));
        assert!(queue.pop(false, 1).is_none());
    }

    #[test]
    fn pipelines_known_to_take_long_leave_workers_for_the_others() {
        let new = || Arc::new(Pending::new(Vec::new(), None));
        let (long, longer, asked, quick) = (new(), new(), new(), new());
        let mut queue = Queue::new();
        for p in [&long, &longer, &asked] {
            assert_eq!(queue.push_as(p, EARLY, true), Some(true));
        }
        assert_eq!(queue.push_as(&quick, EARLY, false), Some(true));
        // The quick one first, then one long one at a time; one a draw asks
        // for waits for nobody.
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &quick));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &long));
        assert!(queue.pop(false, 1).is_none());
        assert_eq!(queue.push_as(&asked, REQUESTED, true), Some(false));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &asked));
        assert!(!asked.was_slow.load(Ordering::Relaxed));
        assert!(queue.pop(false, 1).is_none());
        queue.slow_now -= 1;
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &longer));
        assert!(queue.pop(false, 1).is_none());
    }

    /// Every permutation of the title's character bake is started at the
    /// title's creation of it, not only those of the seven known pixel
    /// shaders: one with a bake geometry stage (the ped baker's vertex
    /// stage, the decoration projection's mesh stage) and another pixel
    /// shader is too. A plain pipeline is not.
    #[test]
    fn every_bake_starts_early() {
        use super::super::named_shader;
        let of = |stage, hash| Pending::new(vec![named_shader(stage, hash), named_shader("pixel", "00000000000000b1")], None);
        assert!(of("vertex", "089d6271be6650ef").scene_input, "the ped baker's vertex stage");
        assert!(of("mesh", "d4111a6e6ea49d91").scene_input, "the decoration projection's mesh stage");
        assert!(!of("vertex", "00000000000000a1").scene_input, "a plain pipeline");
    }

    /// A pending pipeline is a bake by its shaders, before its host pipeline
    /// exists: one of the seven pixel shaders or a bake geometry stage.
    #[test]
    fn a_pending_bake_is_known_by_its_shaders() {
        use super::super::named_shader;
        let with = |stage, hash| [named_shader(stage, hash), named_shader("pixel", "00000000000000b1")];
        assert!(is_bake(&with("vertex", "089d6271be6650ef")));
        assert!(is_bake(&with("mesh", "d4111a6e6ea49d91")));
        assert!(is_bake(&[named_shader("vertex", "00000000000000a1"), named_shader("pixel", "c0a7fe77b1862e10")]));
        assert!(!is_bake(&with("vertex", "00000000000000a1")));
        assert!(!is_bake(&[]));
    }

    #[test]
    fn the_bake_wait_is_on_unless_zero() {
        use std::ffi::OsStr;
        assert!(bake_wait_from(None));
        for on in ["1", "", "yes", "00", " 0"] {
            assert!(bake_wait_from(Some(OsStr::new(on))), "{on:?}");
        }
        assert!(!bake_wait_from(Some(OsStr::new("0"))));
    }

    /// A bake waited for goes before every other pipeline, out of whichever
    /// list held it, and once only; one being made stays where it is.
    #[test]
    fn a_bake_waited_for_goes_to_the_front_once() {
        let new = || Arc::new(Pending::new(Vec::new(), None));
        let (asked, early, bake, slow, making) = (new(), new(), new(), new(), new());
        let mut queue = Queue::new();
        assert_eq!(queue.push_as(&asked, REQUESTED, false), Some(true));
        assert_eq!(queue.push_as(&early, EARLY, false), Some(true));
        assert_eq!(queue.push_as(&bake, EARLY, false), Some(true));
        assert_eq!(queue.push_as(&slow, EARLY, true), Some(true));
        assert_eq!(queue.put_first(&bake), Some(false));
        assert_eq!(queue.put_first(&slow), Some(false));
        let fresh = new();
        assert_eq!(queue.put_first(&fresh), Some(true));
        making.queue_state.store(MAKING, Ordering::Relaxed);
        assert_eq!(queue.put_first(&making), None);
        let order: Vec<Arc<Pending>> = std::iter::from_fn(|| queue.pop(false, 0)).collect();
        let expected = [&fresh, &slow, &bake, &asked, &early];
        assert_eq!(order.len(), expected.len(), "each once");
        for (got, want) in order.iter().zip(expected) {
            assert!(Arc::ptr_eq(got, want));
        }
        assert!(!slow.was_slow.load(Ordering::Relaxed));
    }

    /// A pipeline a worker has and will not finish (for the waits below).
    fn never_made() -> Arc<Pending> {
        let mut p = Pending::new(Vec::new(), None);
        p.queue_state = AtomicU8::new(MAKING);
        Arc::new(p)
    }

    /// A bake whose pipeline is waited for the cap in all, over waits, is not
    /// waited for again: its later draws do not stop the title's thread.
    #[test]
    fn a_bake_past_the_cap_is_not_waited_for_again() {
        let p = never_made();
        let cap = Duration::from_millis(100);
        let started = Instant::now();
        wait_for_bake(&p, || Instant::now() + Duration::from_millis(60), cap);
        assert_eq!(p.bake.load(Ordering::Relaxed) & BAKE_CAPPED, 0, "60 of 100 ms");
        wait_for_bake(&p, || Instant::now() + Duration::from_millis(60), cap);
        assert!(started.elapsed() >= Duration::from_millis(120));
        assert_ne!(p.bake.load(Ordering::Relaxed) & BAKE_CAPPED, 0);
        let again = Instant::now();
        wait_for_bake(&p, || Instant::now() + Duration::from_secs(5), cap);
        assert!(again.elapsed() < Duration::from_secs(1));
        assert!(p.made.get().is_none());
    }

    /// One frame has one deadline for its bake waits, set by the first;
    /// the next frame gets its own.
    #[test]
    fn a_frames_bake_waits_share_one_deadline() {
        let slot = Mutex::new(None);
        assert!(deadline_of(&slot, 7).is_none());
        let first = deadline_in(&slot, 7, Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(deadline_in(&slot, 7, Duration::from_secs(10)), first);
        assert_eq!(deadline_of(&slot, 7), Some(first));
        assert!(deadline_of(&slot, 8).is_none());
        let next = deadline_in(&slot, 8, Duration::from_secs(10));
        assert!(next > first);
        assert!(deadline_of(&slot, 7).is_none());
    }

    /// Two bakes in one frame wait until its deadline together, not each for
    /// as long: the second finds the deadline the first used up.
    #[test]
    fn two_bakes_of_a_frame_wait_one_deadline() {
        let slot = Mutex::new(None);
        let (a, b) = (never_made(), never_made());
        let started = Instant::now();
        for p in [&a, &b] {
            wait_for_bake(p, || deadline_in(&slot, 3, Duration::from_millis(300)), Duration::from_secs(10));
        }
        let took = started.elapsed();
        assert!(took >= Duration::from_millis(300), "{took:?}");
        assert!(took < Duration::from_millis(550), "each waited its own 300 ms: {took:?}");
        assert_eq!(b.bake.load(Ordering::Relaxed) & BAKE_CAPPED, 0);
        assert_ne!(b.bake.load(Ordering::Relaxed) & BAKE_LATE, 0);
    }

    /// The ordinary wait after a frame's bake deadline ends there: a pass over
    /// the screen (10 s of budget a frame) waits for no more than is left.
    #[test]
    fn the_ordinary_wait_honours_the_frames_bake_deadline() {
        let mut p = Pending::new(Vec::new(), None);
        p.queue_state = AtomicU8::new(MAKING);
        p.screen = true;
        let frame = 0xba4e_0000_0000_0003;
        let slot = Mutex::new(Some((frame, Instant::now() + Duration::from_millis(100))));
        let started = Instant::now();
        wait_in(&p, frame, &slot);
        assert!(started.elapsed() < Duration::from_secs(2), "{:?}", started.elapsed());
    }

    /// Passes over the screen go before all others, and workers of their own
    /// take nothing else.
    #[test]
    fn passes_over_the_screen_go_first() {
        let of = |screen| {
            let mut p = Pending::new(Vec::new(), None);
            p.screen = screen;
            Arc::new(p)
        };
        let (world, screen) = (of(false), of(true));
        let mut queue = Queue::new();
        assert_eq!(queue.push(&world, REQUESTED), Some(true));
        assert_eq!(queue.push(&screen, REQUESTED), Some(true));
        assert!(Arc::ptr_eq(&queue.pop(true, 1).unwrap(), &screen));
        assert!(queue.pop(true, 1).is_none());
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &world));
    }
}
