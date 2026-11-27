//! The live stats file and the frame line: what the meter took between two presents' counts,
//! said with the frames it covers.
//!
//! With `VICTORY_LIVE_STATS=<file>` the runtime writes one JSON object about once a second,
//! at the first present a second or more after the object before it, atomically (a temporary
//! file beside it and a rename). Every present is in exactly one object: an object covers the
//! frames since the object before it, says which (`interval`), and its figures are means over
//! all of them. Everything is per frame unless its name ends in `_total` (a count since the
//! start of the run) or `_per_minute`, or is a state now (`vram_mb`, `live`, `segment_open`,
//! `words`, `minute_*`, `landed_classes`, a heap's last sweep). A reader tells a repeated or a stale object
//! from the object alone: `seq` goes up by one an object, `interval.first_frame` is the frame
//! after the object before's `interval.last_frame`, `t_ms` is the runtime's monotonic clock
//! and `unix_ms` the wall clock when it was written.
//!
//! Writing once a second, not once every five or six as before: the object is a few
//! kilobytes, and its making and writing are inside the budget's `report` phase, so their
//! cost is in the figures. Unset: nothing is written and nothing changes.
//!
//! The same sums, added up over the seconds since the last frame line, make the log's frame
//! line every five seconds or more ([`frame_line`]): its figures cover every frame since the
//! line before it.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// The split copy-back parts of the last second, in microseconds a frame: waiting for
/// fetches, card merge, PC-side scan, PC-side merge, ranges done with (`RELAY_STATS` 6, 7, 12,
/// 10, 11). Kept here because `late_line` takes them, in the same present, before the object
/// is written.
static BACK_PARTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Records the parts `late_line` took, for this second's object.
pub(crate) fn remember_back_parts(parts: [u64; 5]) {
    for (slot, value) in BACK_PARTS.iter().zip(parts) {
        slot.store(value, Ordering::Relaxed);
    }
}

/// The copy queue's fetches of the last second, per frame (`VICTORY_FETCH_QUEUE`,
/// `host/fetch.rs`): fetches, megabytes, milliseconds from send to done, submissions held
/// behind a fetch, milliseconds they were held (until the relay saw the ticket reached), plans
/// landed on the drawing queue with the switch on, stops of the copy queue, plans qualified,
/// and holds made only for writing by address; then the plans landed on the drawing queue by
/// reason (`fetch::why`). As bit patterns of `f64`.
static FETCH_QUEUE: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
static FETCH_FALLBACK: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Records the copy queue's numbers the second's line took, for this second's object.
pub(crate) fn remember_fetch_queue(numbers: [f64; 9], reasons: &[f64]) {
    for (slot, value) in FETCH_QUEUE.iter().zip(numbers) {
        slot.store(value.to_bits(), Ordering::Relaxed);
    }
    for (slot, value) in FETCH_FALLBACK.iter().zip(reasons) {
        slot.store(value.to_bits(), Ordering::Relaxed);
    }
}

/// The sums an [`Interval`] holds, each summed across every thread that added to it: they
/// overlap one another and the frame, and are not parts of it.
pub(crate) mod sum {
    /// Microseconds preparing submissions, on whichever threads submitted.
    pub(crate) const PREPARING_US: usize = 0;
    /// Title memory copies: microseconds, bytes.
    pub(crate) const COPIES_US: usize = 1;
    pub(crate) const COPIES_BYTES: usize = 2;
    /// Texture refreshes: microseconds, bytes hashed.
    pub(crate) const TEXTURES_US: usize = 3;
    pub(crate) const TEXTURES_BYTES: usize = 4;
    /// The host's waits, three in a row: for the relay, for flights' fence, synchronous
    /// submissions from send to retirement (microseconds).
    pub(crate) const WAITS_US: usize = 5;
    /// Copying back what shaders wrote: microseconds, bytes merged into title memory.
    pub(crate) const BACK_US: usize = 8;
    pub(crate) const BACK_BYTES: usize = 9;
    /// Draws and dispatches run on the host, and draws dropped without a host pipeline.
    pub(crate) const DRAWN: usize = 10;
    pub(crate) const DISPATCHED: usize = 11;
    pub(crate) const DROPPED: usize = 12;
    pub(crate) const COUNT: usize = 13;
}

/// The frame times an interval keeps for its percentiles and its list: a second at 240
/// frames a second, or the five seconds of a frame line at 48.
const TIMES: usize = 240;

/// The presenting thread's budget and the other threads' figures over an interval
/// (`crate::budget`): nanoseconds by phase; processor time where the counter's rate was known.
#[cfg(feature = "diagnostics")]
#[derive(Clone, Copy, Default)]
pub(crate) struct Budget {
    pub(crate) frames: u64,
    pub(crate) ns: u64,
    pub(crate) wall: [u64; crate::budget::PHASES],
    pub(crate) cpu_ns: [f64; crate::budget::PHASES],
    pub(crate) scopes: u64,
    pub(crate) others_wall: [u64; crate::budget::PHASES],
    pub(crate) others_cpu_ns: [f64; crate::budget::PHASES],
    pub(crate) others_scopes: u64,
    /// Whether a processor time above was left out: the counter's rate was not known.
    pub(crate) cpu_missing: bool,
    /// Calls the title made into the runtime from outside every scope, and the largest single
    /// stretch of `title` before one, in nanoseconds (the largest of the interval, not a sum).
    pub(crate) calls: u64,
    pub(crate) largest_gap: u64,
    /// The processor time of `title` and the title's calls together: they read no cycle
    /// counter one by one (`crate::budget::ENTRY_PHASES`).
    pub(crate) entry_cpu_ns: f64,
}

#[cfg(feature = "diagnostics")]
impl Budget {
    /// What the presenting thread (this one) and the others measured since the last take.
    pub(crate) fn take() -> Budget {
        Budget::of(&crate::budget::take(), &crate::budget::take_others())
    }

    fn of(taken: &crate::budget::Taken, others: &crate::budget::Others) -> Budget {
        let rate = taken.cycles_per_ns;
        let time = |cycles: u64| if rate > 0.0 { cycles as f64 / rate } else { 0.0 };
        Budget {
            frames: taken.frames,
            ns: taken.ns,
            wall: taken.wall,
            cpu_ns: taken.cpu.map(time),
            scopes: taken.scopes,
            others_wall: others.wall,
            others_cpu_ns: others.cpu.map(time),
            others_scopes: others.scopes,
            cpu_missing: rate <= 0.0,
            calls: taken.calls,
            largest_gap: taken.largest_gap,
            entry_cpu_ns: time(taken.entry_cycles()),
        }
    }

    fn add(&mut self, o: &Budget) {
        self.frames += o.frames;
        self.ns += o.ns;
        self.scopes += o.scopes;
        self.others_scopes += o.others_scopes;
        for phase in 0..crate::budget::PHASES {
            self.wall[phase] += o.wall[phase];
            self.cpu_ns[phase] += o.cpu_ns[phase];
            self.others_wall[phase] += o.others_wall[phase];
            self.others_cpu_ns[phase] += o.others_cpu_ns[phase];
        }
        self.cpu_missing |= o.cpu_missing;
        self.calls += o.calls;
        self.largest_gap = self.largest_gap.max(o.largest_gap);
        self.entry_cpu_ns += o.entry_cpu_ns;
    }

    /// `"name":{"ms":..,"cpu_ms":..}` by phase, per frame over `frames`; `skip` leaves phases
    /// out. `title` and the title's calls have no processor time of their own: it is known
    /// only for all of them together (`title_and_calls_cpu_ms`).
    fn phases(&self, wall: &[u64], cpu_ns: &[f64], frames: f64, skip: &[usize]) -> String {
        let mut text = String::new();
        for (phase, name) in crate::budget::NAMES.iter().enumerate() {
            if skip.contains(&phase) {
                continue;
            }
            if !text.is_empty() {
                text.push(',');
            }
            let entry = crate::budget::ENTRY_PHASES.iter().any(|&p| p as usize == phase);
            let cpu = if self.cpu_missing || entry {
                "null".to_owned()
            } else {
                format!("{:.3}", cpu_ns[phase] / 1e6 / frames)
            };
            text.push_str(&format!("\"{name}\":{{\"ms\":{:.3},\"cpu_ms\":{cpu}}}", wall[phase] as f64 / 1e6 / frames));
        }
        text
    }

    /// The stats file's `budget` and `other_threads` objects. The budget is per frame over
    /// its own frames; the others' figures are per frame over `frames`, the interval's.
    fn json(&self, frames: u64) -> (String, String) {
        use crate::budget::Phase;
        let budget = if self.frames == 0 {
            "null".to_owned()
        } else {
            let per = self.frames as f64;
            let sum: u64 = self.wall.iter().sum();
            format!(
                "{{\"thread\":\"presenting\",\"frames\":{},\"frame_ms\":{:.3},\"sum_ms\":{:.3},\"scopes\":{:.1},\
                 \"calls\":{:.1},\"largest_title_gap_ms\":{:.3},\"title_and_calls_cpu_ms\":{},\"phases\":{{{}}}}}",
                self.frames,
                self.ns as f64 / 1e6 / per,
                sum as f64 / 1e6 / per,
                self.scopes as f64 / per,
                self.calls as f64 / per,
                self.largest_gap as f64 / 1e6,
                if self.cpu_missing { "null".to_owned() } else { format!("{:.3}", self.entry_cpu_ns / 1e6 / per) },
                self.phases(&self.wall, &self.cpu_ns, per, &[Phase::PoolWork as usize, Phase::RelayLand as usize]),
            )
        };
        let per = frames.max(1) as f64;
        let others = format!(
            "{{\"scopes\":{:.1},\"phases\":{{{}}}}}",
            self.others_scopes as f64 / per,
            self.phases(&self.others_wall, &self.others_cpu_ns, per, &[Phase::Title as usize]),
        );
        (budget, others)
    }

    /// The log's budget line, with the frame line.
    fn line(&self, first: u64, last: u64, frames: u64) -> Option<String> {
        use crate::budget::{NAMES, PHASES, Phase};
        if self.frames == 0 {
            return None;
        }
        let per = self.frames as f64;
        let ms = |ns: u64| ns as f64 / 1e6 / per;
        let parts: Vec<String> = (0..PHASES)
            .filter(|&p| self.wall[p] > 0 || p == Phase::Title as usize || p == Phase::Unaccounted as usize)
            .map(|p| format!("{} {:.2}", NAMES[p], ms(self.wall[p])))
            .collect();
        let entry = |p: usize| crate::budget::ENTRY_PHASES.iter().any(|&e| e as usize == p);
        let mut cpu: Vec<String> = (0..PHASES)
            .filter(|&p| self.cpu_ns[p] > 0.0 && !entry(p))
            .map(|p| format!("{} {:.2}", NAMES[p], self.cpu_ns[p] / 1e6 / per))
            .collect();
        cpu.insert(0, format!("the title and its calls together {:.2}", self.entry_cpu_ns / 1e6 / per));
        let over = frames.max(1) as f64;
        let others: Vec<String> = (0..PHASES)
            .filter(|&p| self.others_wall[p] > 0)
            .map(|p| format!("{} {:.2} (cpu {:.2})", NAMES[p], self.others_wall[p] as f64 / 1e6 / over, self.others_cpu_ns[p] / 1e6 / over))
            .collect();
        Some(format!(
            "frame budget, the presenting thread, frames {first}..{last} ({} frames): {:.2} ms/frame = {} (sum {:.2}); of it on \
             the processor{}: {}; {:.0} scopes a frame, {:.0} calls from the title a frame, the largest stretch of title between \
             two {:.2} ms; other threads per frame, overlapping, not parts of the frame: {}",
            self.frames,
            ms(self.ns),
            parts.join(" + "),
            ms(self.wall.iter().sum()),
            if self.cpu_missing { " (the counter's rate was unknown for part of it: left out)" } else { "" },
            cpu.join(", "),
            self.scopes as f64 / per,
            self.calls as f64 / per,
            self.largest_gap as f64 / 1e6,
            if others.is_empty() { "nothing measured".to_owned() } else { others.join(", ") },
        ))
    }
}

/// What the meter took over the presents `first_frame..=last_frame`: sums, not per frame.
#[derive(Clone, Default)]
pub(crate) struct Interval {
    /// The first and last frame counted (0: none yet).
    pub(crate) first_frame: u64,
    pub(crate) last_frame: u64,
    pub(crate) frames: u64,
    /// From the present before `first_frame` to `last_frame`'s, in milliseconds: the sum of
    /// the frames' times.
    pub(crate) wall_ms: f64,
    /// The frames' times in milliseconds, the first [`TIMES`] of them.
    times: Vec<f64>,
    /// See [`sum`].
    pub(crate) sums: [u64; sum::COUNT],
    #[cfg(feature = "diagnostics")]
    pub(crate) budget: Budget,
}

impl Interval {
    /// An empty interval with room for its frame times: nothing is allocated at a present.
    pub(crate) fn new() -> Interval {
        Interval { times: Vec::with_capacity(TIMES), ..Default::default() }
    }

    /// A present counted frame `frame`, `ms` after the present before it.
    pub(crate) fn present(&mut self, frame: u64, ms: f64) {
        if self.frames == 0 {
            self.first_frame = frame;
        }
        self.last_frame = frame;
        self.frames += 1;
        self.wall_ms += ms;
        if self.times.len() < TIMES {
            self.times.push(ms);
        }
    }

    /// Adds `o`, the frames after this interval's.
    pub(crate) fn add(&mut self, o: &Interval) {
        if o.frames == 0 {
            return;
        }
        if self.frames == 0 {
            self.first_frame = o.first_frame;
        }
        self.last_frame = o.last_frame;
        self.frames += o.frames;
        self.wall_ms += o.wall_ms;
        let room = TIMES - self.times.len();
        self.times.extend(o.times.iter().take(room));
        for (all, more) in self.sums.iter_mut().zip(o.sums) {
            *all += more;
        }
        #[cfg(feature = "diagnostics")]
        self.budget.add(&o.budget);
    }

    /// Empties it, keeping its room.
    pub(crate) fn clear(&mut self) {
        let mut times = std::mem::take(&mut self.times);
        times.clear();
        *self = Interval { times, ..Default::default() };
    }

    /// Sum `i` per frame, divided by `scale` (1e3 for milliseconds of microseconds, 1e6 for
    /// megabytes of bytes).
    pub(crate) fn per_frame(&self, i: usize, scale: f64) -> f64 {
        self.sums[i] as f64 / scale / self.frames.max(1) as f64
    }

    /// The mean frame time in milliseconds.
    pub(crate) fn frame_ms(&self) -> f64 {
        self.wall_ms / self.frames.max(1) as f64
    }

    /// Frames a second over the interval.
    pub(crate) fn fps(&self) -> f64 {
        if self.wall_ms > 0.0 { self.frames as f64 * 1000.0 / self.wall_ms } else { 0.0 }
    }

    /// The longest of the kept frame times (0 with none): with the four or five frames of a
    /// second, the only honest thing to say beside the median.
    pub(crate) fn worst(&self) -> f64 {
        self.times.iter().copied().fold(0.0, f64::max)
    }

    /// The p50 and p95 of the kept frame times, by nearest rank (0 with none).
    pub(crate) fn percentiles(&self) -> (f64, f64) {
        if self.times.is_empty() {
            return (0.0, 0.0);
        }
        let mut sorted = self.times.clone();
        sorted.sort_by(f64::total_cmp);
        let rank = |percent: usize| sorted[(sorted.len() * percent).div_ceil(100).max(1) - 1];
        (rank(50), rank(95))
    }

    /// Frames whose time is in the sums and not in the list or the percentiles.
    fn times_missing(&self) -> u64 {
        self.frames - self.times.len() as u64
    }
}

/// When the meter reports: every present joins the second under way; a second or more
/// after the last take the second's object is written and its frames join the frame line's;
/// five seconds or more after the last frame line the line is logged over all of them. The
/// order is here, not in the meter, because it was the defect: the object was written inside
/// the frame line's gate, one second's sample about every six.
pub(crate) struct Gate {
    since: std::time::Instant,
    logged: std::time::Instant,
    /// The frames since the last second's take, and since the last frame line: every present
    /// is in both, once.
    pub(crate) second: Interval,
    pub(crate) line: Interval,
}

impl Gate {
    pub(crate) fn new(now: std::time::Instant) -> Gate {
        Gate { since: now, logged: now, second: Interval::new(), line: Interval::new() }
    }

    /// Whether the second's take is due at a present at `now`: a second or more since the
    /// last, and a frame to report.
    pub(crate) fn second_due(&self, now: std::time::Instant) -> bool {
        now.saturating_duration_since(self.since) >= std::time::Duration::from_secs(1) && self.second.frames > 0
    }

    /// The second's sums are in: `write` gets it (the stats file's object, every second), its
    /// frames join the frame line's, and the next second begins. Whether the frame line is
    /// due: five seconds or more since the last ([`Gate::line_done`] after it is logged).
    pub(crate) fn second_done(&mut self, now: std::time::Instant, write: impl FnOnce(&Interval)) -> bool {
        write(&self.second);
        self.line.add(&self.second);
        self.second.clear();
        self.since = now;
        let due = now.saturating_duration_since(self.logged) >= std::time::Duration::from_secs(5);
        if due {
            self.logged = now;
        }
        due
    }

    /// The frame line was logged: the next one begins.
    pub(crate) fn line_done(&mut self) {
        self.line.clear();
    }
}

/// The file `VICTORY_LIVE_STATS` names (an empty value is off); said once.
fn path() -> Option<&'static PathBuf> {
    static PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
    PATH.get_or_init(|| {
        let path = std::env::var_os("VICTORY_LIVE_STATS").filter(|value| !value.is_empty()).map(PathBuf::from);
        if let Some(path) = &path {
            tracing::info!(
                file = %path.display(),
                "live stats: one JSON object about a second is written here, each covering every frame since the one before it"
            );
        }
        path
    })
    .as_ref()
}

/// What an object says besides its interval: read where it is written.
pub(crate) struct Now {
    /// The object's number: one more than the object before it.
    pub(crate) seq: u64,
    /// The runtime's monotonic clock and the wall clock, in milliseconds.
    pub(crate) t_ms: f64,
    pub(crate) unix_ms: u64,
    pub(crate) vram_mb: Option<u64>,
    pub(crate) live: Vec<(&'static str, String)>,
    pub(crate) segment: u64,
    /// The segment under way (`crate::live::open_json`), or `null`.
    pub(crate) segment_open: String,
    /// The card's queues by timestamps (`host/queue_stamps.rs`), or `null`.
    pub(crate) gpu_stamps: String,
}

/// The number of the last object made.
static SEQ: AtomicU64 = AtomicU64::new(0);

impl Now {
    /// The next object's: its number is taken here.
    fn next(vram_mb: Option<u64>, gpu_stamps: String) -> Now {
        static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
        Now {
            seq: SEQ.fetch_add(1, Ordering::Relaxed) + 1,
            t_ms: EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1000.0,
            unix_ms: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64),
            vram_mb,
            live: crate::live::pairs(),
            segment: crate::live::segment(),
            segment_open: crate::live::open_json(),
            gpu_stamps,
        }
    }
}

/// Writes the object of `interval` when the stats file is asked for; nothing otherwise
/// (`vram_mb` and `gpu_stamps` are asked for only then).
pub(crate) fn write(interval: &Interval, vram_mb: impl FnOnce() -> Option<u64>, gpu_stamps: impl FnOnce() -> String) {
    // The live panel's first line, with the switch on (`crate::host::hud`): the same numbers.
    super::hud::note_sample(interval.frame_ms(), interval.fps());
    let Some(path) = path() else {
        return;
    };
    let text = json(interval, &Now::next(vram_mb(), gpu_stamps()));
    if let Err(e) = write_atomic(path, &text) {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, Ordering::Relaxed) {
            tracing::warn!(error = %e, file = %path.display(), "live stats: the file could not be written");
        }
    }
}

/// A counter the copies line zeroes every few seconds, since this was last asked: its value
/// less the value then, or its value alone once it has been zeroed in between.
#[cfg(feature = "diagnostics")]
fn since_last(now: u64, last: &AtomicU64) -> u64 {
    let before = last.swap(now, Ordering::Relaxed);
    if now >= before { now - before } else { now }
}

/// One JSON object, keys in a fixed order.
fn json(interval: &Interval, now: &Now) -> String {
    let frames = interval.frames.max(1) as f64;
    let parts = [0, 1, 2, 3, 4].map(|i| BACK_PARTS[i].load(Ordering::Relaxed) as f64 / 1000.0);
    let queue = [0, 1, 2, 3, 4, 5, 6, 7, 8].map(|i| f64::from_bits(FETCH_QUEUE[i].load(Ordering::Relaxed)));
    let mut reasons = String::new();
    for (i, name) in super::fetch::why::NAMES.iter().enumerate() {
        let n = f64::from_bits(FETCH_FALLBACK[i].load(Ordering::Relaxed));
        reasons.push_str(&format!("{}\"{name}\":{n:.2}", if i > 0 { "," } else { "" }));
    }
    // The upload pool's counters live in the diagnostics' counters, which the copies line
    // zeroes; a release build has neither the pool nor the counters.
    let [pool_bytes, pool_join] = {
        #[cfg(feature = "diagnostics")]
        {
            static LAST: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
            [
                since_last(crate::copystat::peek(crate::copystat::C::UploadPoolBytes), &LAST[0]),
                since_last(crate::copystat::peek(crate::copystat::C::UploadPoolJoinUs), &LAST[1]),
            ]
        }
        #[cfg(not(feature = "diagnostics"))]
        {
            [0, 0]
        }
    };
    // Stores into title memory the kernel refused since the start (`crate::storestat`): a
    // run that drops landed bytes says so here even when the log's copy line is not read.
    let (stores_failed, stores_failed_bytes) = crate::storestat::total();
    let mut live_text = String::new();
    for (i, (name, value)) in now.live.iter().enumerate() {
        if i > 0 {
            live_text.push(',');
        }
        live_text.push_str(&format!("\"{}\":\"{}\"", escape(name), escape(value)));
    }
    // Pipeline states made in all, and per minute since the previous object
    // (`host/background.rs`): once a scene has settled, any made is one made
    // again (the library missed it by name).
    let (pipelines_created, pipelines_per_minute) = crate::host::background::pipeline_figures();
    // The median and the worst frame: an object's few frames have no 95th percentile (an
    // arm's percentiles come from the objects' frame times joined).
    let (p50, worst) = (interval.percentiles().0, interval.worst());
    let times: Vec<String> = interval.times.iter().map(|ms| format!("{ms:.2}")).collect();
    #[cfg(feature = "diagnostics")]
    let (budget, others) = interval.budget.json(interval.frames);
    #[cfg(not(feature = "diagnostics"))]
    let (budget, others) = ("null", "null");
    format!(
        "{{\"seq\":{seq},\"t_ms\":{t_ms:.1},\"unix_ms\":{unix_ms},\
         \"interval\":{{\"first_frame\":{first},\"last_frame\":{last},\"frames\":{count},\"wall_ms\":{wall:.2}}},\
         \"frame\":{last},\"fps\":{fps:.2},\"frame_ms\":{ms:.2},\"frame_ms_p50\":{p50:.2},\"frame_ms_worst\":{worst:.2},\
         \"frame_ms_list\":[{times}],\"frame_ms_list_missing\":{missing},\
         \"budget\":{budget},\"other_threads\":{others},\
         \"summed_across_threads\":{{\"preparing_ms\":{prep:.2},\"copies_ms\":{copy_ms:.2},\"copies_mb\":{copy_mb:.2},\
         \"textures_ms\":{tex_ms:.2},\"textures_mb\":{tex_mb:.2},\"host_wait_ms\":{wait:.2},\
         \"back_ms\":{back_ms:.2},\"back_mb\":{back_mb:.2},\"back_fetch_ms\":{fetch:.2},\"back_card_ms\":{card:.2},\
         \"back_scan_ms\":{scan:.2},\"back_merge_ms\":{merge:.2},\"back_done_ms\":{done:.2},\
         \"pool_mb\":{pool_mb:.2},\"pool_join_ms\":{pool_join:.2},\
         \"fetch_queue_fetches\":{fq_n:.2},\"fetch_queue_mb\":{fq_mb:.2},\"fetch_queue_ms\":{fq_ms:.2},\
         \"fetch_queue_held\":{fq_held:.2},\"fetch_queue_held_ms\":{fq_held_ms:.2},\
         \"fetch_queue_fallback\":{fq_fallback:.2},\"fetch_queue_fallback_by\":{{{reasons}}},\
         \"fetch_queue_stops\":{fq_stops:.2},\"fetch_queue_qualified\":{fq_qualified:.2},\
         \"fetch_queue_held_by_address\":{fq_by_address:.2}}},\
         \"gpu_stamps\":{gpu_stamps},\
         \"draws\":{draws:.1},\"dispatched\":{dispatched:.1},\"dropped\":{dropped:.1},\
         \"stores_failed_total\":{stores_failed},\"stores_failed_bytes_total\":{stores_failed_bytes},\
         \"pipelines_created_total\":{pipelines_created},\"pipelines_per_minute\":{pipelines_per_minute:.1},\
         \"vram_mb\":{vram},\"live\":{{{live_text}}},\"segments_closed_total\":{segment},\
         \"segment_open\":{segment_open},\
         \"kernel_calls\":{kernel_calls}{optional}}}",
        seq = now.seq,
        t_ms = now.t_ms,
        unix_ms = now.unix_ms,
        first = interval.first_frame,
        last = interval.last_frame,
        count = interval.frames,
        wall = interval.wall_ms,
        fps = interval.fps(),
        ms = interval.frame_ms(),
        times = times.join(","),
        missing = interval.times_missing(),
        prep = interval.per_frame(sum::PREPARING_US, 1e3),
        copy_ms = interval.per_frame(sum::COPIES_US, 1e3),
        copy_mb = interval.per_frame(sum::COPIES_BYTES, 1e6),
        tex_ms = interval.per_frame(sum::TEXTURES_US, 1e3),
        tex_mb = interval.per_frame(sum::TEXTURES_BYTES, 1e6),
        wait = (0..3).map(|i| interval.per_frame(sum::WAITS_US + i, 1e3)).sum::<f64>(),
        back_ms = interval.per_frame(sum::BACK_US, 1e3),
        back_mb = interval.per_frame(sum::BACK_BYTES, 1e6),
        fetch = parts[0],
        card = parts[1],
        scan = parts[2],
        merge = parts[3],
        done = parts[4],
        pool_mb = pool_bytes as f64 / 1e6 / frames,
        pool_join = pool_join as f64 / 1e3 / frames,
        fq_n = queue[0],
        fq_mb = queue[1],
        fq_ms = queue[2],
        fq_held = queue[3],
        fq_held_ms = queue[4],
        fq_fallback = queue[5],
        fq_stops = queue[6],
        fq_qualified = queue[7],
        fq_by_address = queue[8],
        gpu_stamps = now.gpu_stamps,
        draws = interval.per_frame(sum::DRAWN, 1.0),
        dispatched = interval.per_frame(sum::DISPATCHED, 1.0),
        dropped = interval.per_frame(sum::DROPPED, 1.0),
        vram = now.vram_mb.map_or("null".to_owned(), |mb| mb.to_string()),
        segment = now.segment,
        segment_open = now.segment_open,
        kernel_calls = crate::kernelstat::text(interval.frames),
        optional = optional(),
    )
}

/// The stale-upload trap's figures as an object (`VICTORY_TEXTURE_RECHECK`),
/// counts since the switch last came on (`_total`) and what it follows now:
/// `stale_total` is the fault (sampled after the change was seen, still
/// registered, not uploaded again within 600 frames), `stale_textures_total`
/// its distinct textures, and the four `stale_*_total` after it split it by
/// writer (storage request, copy-back landing, no writer seen, writer unknown).
fn recheck_text(f: &crate::host::recheck::Figures) -> String {
    format!(
        "{{\"uploads_total\":{},\"too_large_total\":{},\"again_total\":{},\"stale_total\":{},\"stale_textures_total\":{},\
         \"unused_total\":{},\"released_total\":{},\"following\":{},\"stale_storage_total\":{},\"stale_landing_total\":{},\
         \"stale_no_writer_seen_total\":{},\"stale_writer_unknown_total\":{},\"not_looked_total\":{},\
         \"follows_dropped_total\":{},\"lines_capped_total\":{},\"frames_skipped_total\":{}}}",
        f.uploads,
        f.too_large,
        f.again.iter().sum::<u64>(),
        f.stale,
        f.stale_textures,
        f.unused,
        f.released,
        f.following,
        f.writers[0],
        f.writers[1],
        f.writers[2],
        f.writers[3],
        f.not_looked,
        f.follows_dropped,
        f.lines_capped,
        f.frames_skipped
    )
}

/// The textures-made object (`VICTORY_TEXTURE_MADE`) while its switch is on;
/// none with it off, and in a build without the diagnostic.
fn made_object() -> Option<String> {
    #[cfg(feature = "diagnostics")]
    {
        crate::live::texture_made().then(made_text)
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        None
    }
}

/// The object's optional parts, after `kernel_calls`: each is there only while its switch is
/// on, so with every switch off the object is the same for every run. A part's names follow
/// the object's rule: per frame, or `_total`, `_per_minute`, or a state now.
///
/// ADD AN OPTIONAL OBJECT HERE: one line in `objects`, its name and its JSON text while its
/// switch is on (`None` while it is off).
fn optional() -> String {
    // `VICTORY_HASH_CHECK=1`: the check's three counts since the start, as fields of the
    // object itself (the check's own text, with its leading comma); nothing with it off.
    let mut text = crate::host::hash_check::fields(crate::host::hash_check::on());
    let objects: [(&str, Option<String>); 4] = [
        // `VICTORY_TEXTURE_RECEIPTS=1`: counts since the start and each heap's last sweep.
        ("texture_receipts", crate::receipts::on().then(crate::receipts::stats_json)),
        // `VICTORY_LANDED_CLASSES=1`: the last finished 600-frame window (`crate::memclass`).
        ("landed_classes", crate::memclass::landed_object()),
        // `VICTORY_TEXTURE_RECHECK=1`: the stale-upload trap, since the switch last came on.
        ("texture_recheck", crate::host::recheck::on().then(|| recheck_text(&crate::host::recheck::figures()))),
        // `VICTORY_TEXTURE_MADE=1`: new keys and storage requests (diagnostics builds).
        ("textures_made", made_object()),
        // <-- the next optional object goes on a line of its own here (and the array's length goes up by one).
    ];
    for (name, object) in objects {
        if let Some(object) = object {
            text.push_str(&format!(",\"{name}\":{object}"));
        }
    }
    text
}

/// What the frame line says besides an interval's sums, per frame over the same frames:
/// megabytes fetched from copies in the card's memory, draws left to indirect execution,
/// passes over the heap's buffer views and their milliseconds, copies into depth textures
/// and into their twins, depth planes drawn from their twins.
pub(super) struct Beside {
    pub(super) fetched_mb: f64,
    pub(super) indirect: f64,
    pub(super) picked: f64,
    pub(super) picked_ms: f64,
    pub(super) twin_to_depth: f64,
    pub(super) twin_to_colour: f64,
    pub(super) planes: f64,
}

/// The frame line's host waits per frame, from an interval's sums.
pub(super) fn waits(line: &Interval) -> super::HostWaits {
    super::HostWaits([0, 1, 2].map(|i| line.per_frame(sum::WAITS_US + i, 1e3)))
}

/// The clause the frame line ends with: the frames its means are over.
fn window_text(line: &Interval) -> String {
    let (p50, p95) = line.percentiles();
    format!(
        "means over frames {}..{} ({} frames in {:.0} ms, p50 {p50:.1}, p95 {p95:.1} ms/frame): every frame since the line \
         before; the figures in the parentheses are summed across threads and overlap, they are not parts of the frame (the \
         frame budget line's are)",
        line.first_frame, line.last_frame, line.frames, line.wall_ms
    )
}

/// Logs the frame line for `line`, every frame since the line before it; then the presenting
/// thread's budget over the same frames, in a diagnostics build.
pub(super) fn frame_line(line: &Interval, beside: &Beside) {
    use Ordering::Relaxed;
    let waits = waits(line);
    let Beside { fetched_mb, indirect, picked, picked_ms, twin_to_depth, twin_to_colour, planes } = *beside;
    tracing::info!(
        "frame {frame}, {fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
         {copy_mb:.1} MB, textures hashed {texture_ms:.1} ms {texture_mb:.1} MB; {waits}; \
         {back_ms:.1} copying back {back_mb:.1} MB merged into title memory (host time, fetches included), \
         {fetched_mb:.1} MB fetched from copies in the card's memory), {draws:.0} draws/frame, \
         dropped {dropped:.0} without a host pipeline ({making} being made) and \
         {indirect:.0} indirect; heap buffers {picked:.1} passes {picked_ms:.1} ms; \
         texture slots written so far: {placed} placed, {objects} of objects; \
         textures picked by index: {picked_textures} uploaded so far ({picked_mb} MB), \
         {picked_waiting} waiting, {picked_away} passed over unmapped; depth textures with \
         unordered access: {du0} made, bound as depth {du1}x, cleared as depth {du2}x, \
         unordered views refused {du3}x; buffer views left null as the host would refuse them: {refused}; \
         host waits summed across threads {host_wait_ms:.1} ms a frame, {usage}; {window}",
        frame = line.last_frame,
        fps = line.fps(),
        ms = line.frame_ms(),
        cpu = line.per_frame(sum::PREPARING_US, 1e3),
        copy_ms = line.per_frame(sum::COPIES_US, 1e3),
        copy_mb = line.per_frame(sum::COPIES_BYTES, 1e6),
        texture_ms = line.per_frame(sum::TEXTURES_US, 1e3),
        texture_mb = line.per_frame(sum::TEXTURES_BYTES, 1e6),
        back_ms = line.per_frame(sum::BACK_US, 1e3),
        back_mb = line.per_frame(sum::BACK_BYTES, 1e6),
        draws = line.per_frame(sum::DRAWN, 1.0),
        dropped = line.per_frame(sum::DROPPED, 1.0),
        picked_textures = crate::draw::HEAP_TEXTURES.load(Relaxed),
        picked_mb = crate::draw::HEAP_TEXTURE_BYTES.load(Relaxed) >> 20,
        picked_waiting = crate::draw::HEAP_TEXTURES_WAITING.load(Relaxed),
        picked_away = crate::draw::HEAP_TEXTURES_AWAY.load(Relaxed),
        du0 = crate::resource::DEPTH_UNORDERED[0].load(Relaxed),
        du1 = crate::resource::DEPTH_UNORDERED[1].load(Relaxed),
        du2 = crate::resource::DEPTH_UNORDERED[2].load(Relaxed),
        du3 = crate::resource::DEPTH_UNORDERED[3].load(Relaxed),
        refused = format!(
            "{}; depth twins: {} alive, {} depth textures left without one; twin copies per frame: {twin_to_depth:.1} into depth textures, {twin_to_colour:.1} into twins; depth planes: {planes:.1} drawn from their twins per frame, {} times not given back to a twin, {} times used before the pass",
            crate::descriptor::host::REFUSED_VIEWS.load(Relaxed),
            crate::resource::TWINS[0].load(Relaxed),
            crate::resource::TWINS[1].load(Relaxed),
            crate::resource::PLANES[1].load(Relaxed),
            crate::resource::PLANES[2].load(Relaxed)
        ),
        making = crate::pipeline::making(),
        placed = crate::descriptor::host::PLACED_TEXTURE_SLOTS.load(Relaxed),
        objects = crate::descriptor::host::OBJECT_TEXTURE_SLOTS.load(Relaxed),
        host_wait_ms = waits.all(),
        usage = super::Usage::text(super::Usage::latest()),
        window = window_text(line),
    );
    #[cfg(feature = "diagnostics")]
    if let Some(text) = line.budget.line(line.first_frame, line.last_frame, line.frames) {
        tracing::info!("{text}");
    }
}

/// The textures-made diagnostic's counts as an object (`VICTORY_TEXTURE_MADE`):
/// new keys and storage requests since the start (`_total`), and the last
/// wall-clock minute before now (a state: its number and that one minute's
/// counts; `null` before the first minute has passed).
#[cfg(feature = "diagnostics")]
fn made_text() -> String {
    let (keys, requests, last) = crate::resource::made::counts();
    let last = last.map_or_else(
        || "null".to_owned(),
        |(minute, requests, ranges, keys)| {
            format!("{{\"minute\":{minute},\"storage_requests\":{requests},\"distinct_ranges\":{ranges},\"new_keys\":{keys}}}")
        },
    );
    format!("{{\"new_keys_total\":{keys},\"storage_requests_total\":{requests},\"last_minute\":{last}}}")
}

/// A JSON string's contents: quotes, backslashes and control characters escaped.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Writes `text` to `path` atomically: a temporary file beside it, then a rename, so a
/// reader never sees half an object.
fn write_atomic(path: &std::path::Path, text: &str) -> std::io::Result<()> {
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = PathBuf::from(temporary);
    std::fs::write(&temporary, text)?;
    std::fs::rename(&temporary, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Five frames, 1230 to 1234, 220 ms each but one of 420: 1300 ms, with the sums a
    /// meter would have taken over them.
    fn interval() -> Interval {
        let mut interval = Interval::new();
        for (frame, ms) in [(1230, 220.0), (1231, 220.0), (1232, 420.0), (1233, 220.0), (1234, 220.0)] {
            interval.present(frame, ms);
        }
        interval.sums[sum::PREPARING_US] = 5 * 96_000;
        interval.sums[sum::COPIES_US] = 5 * 40_000;
        interval.sums[sum::COPIES_BYTES] = 5 * 156_000_000;
        interval.sums[sum::TEXTURES_US] = 5 * 4_900;
        interval.sums[sum::TEXTURES_BYTES] = 5 * 16_700_000;
        interval.sums[sum::WAITS_US] = 5 * 70_000;
        interval.sums[sum::WAITS_US + 1] = 5 * 5_000;
        interval.sums[sum::WAITS_US + 2] = 5 * 3_000;
        interval.sums[sum::BACK_US] = 5 * 122_000;
        interval.sums[sum::BACK_BYTES] = 5 * 45_300_000;
        interval.sums[sum::DRAWN] = 5 * 6500;
        interval.sums[sum::DISPATCHED] = 60;
        interval.sums[sum::DROPPED] = 10;
        interval
    }

    fn now(seq: u64) -> Now {
        Now {
            seq,
            t_ms: 1000.0 * seq as f64,
            unix_ms: 1_800_000_000_000 + 1000 * seq,
            vram_mb: Some(18600),
            live: vec![("VICTORY_LAND_SCAN", "off".to_owned()), ("VICTORY_EXACT_BACK", "on".to_owned())],
            segment: 3,
            segment_open: "null".to_owned(),
            gpu_stamps: "null".to_owned(),
        }
    }

    /// The object holds every field the README names, per frame over its interval, and
    /// numbers and the live map as JSON. (Corrected with the object: the parts are under
    /// `summed_across_threads`, the totals say `_total`, and `dispatched` is per frame; and
    /// with its check: the remaining totals say `_total` too, and the largest of an
    /// object's frame times is `frame_ms_worst`, where it was called a 95th percentile.)
    #[test]
    fn the_object_has_every_field() {
        remember_back_parts([1000, 2000, 3000, 4000, 5000]);
        remember_fetch_queue([11.5, 74.25, 3.5, 9.0, 12.75, 4.0, 0.0, 7.5, 2.25], &[3.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let text = json(&interval(), &now(7));
        for key in [
            "\"seq\":7",
            "\"interval\":{\"first_frame\":1230,\"last_frame\":1234,\"frames\":5,\"wall_ms\":1300.00}",
            "\"frame\":1234",
            "\"fps\":3.85",
            "\"frame_ms\":260.00",
            "\"frame_ms_p50\":220.00",
            "\"frame_ms_worst\":420.00",
            "\"frame_ms_list\":[220.00,220.00,420.00,220.00,220.00],\"frame_ms_list_missing\":0",
            "\"summed_across_threads\":{\"preparing_ms\":96.00",
            "\"copies_ms\":40.00",
            "\"copies_mb\":156.00",
            "\"textures_ms\":4.90",
            "\"host_wait_ms\":78.00",
            "\"back_ms\":122.00",
            "\"back_fetch_ms\":1.00",
            "\"back_card_ms\":2.00",
            "\"back_scan_ms\":3.00",
            "\"back_merge_ms\":4.00",
            "\"back_done_ms\":5.00",
            "\"fetch_queue_fetches\":11.50",
            "\"fetch_queue_mb\":74.25",
            "\"fetch_queue_ms\":3.50",
            "\"fetch_queue_held\":9.00",
            "\"fetch_queue_held_ms\":12.75",
            "\"fetch_queue_fallback\":4.00",
            "\"fetch_queue_fallback_by\":{\"numbered\":3.00,\"quarantine\":0.00,\"stopped\":0.00,\"released\":1.00,\"late\":0.00,\"shared\":0.00}",
            "\"fetch_queue_stops\":0.00",
            "\"fetch_queue_qualified\":7.50",
            "\"fetch_queue_held_by_address\":2.25",
            "\"gpu_stamps\":null",
            "\"draws\":6500.0",
            "\"dropped\":2.0",
            "\"stores_failed_total\":0",
            "\"stores_failed_bytes_total\":0",
            "\"pipelines_created_total\":",
            "\"pipelines_per_minute\":",
            "\"vram_mb\":18600",
            "\"VICTORY_LAND_SCAN\":\"off\"",
            "\"VICTORY_EXACT_BACK\":\"on\"",
            "\"segments_closed_total\":3",
            "\"segment_open\":null",
            "\"kernel_calls\":{\"per_frame\":",
            "\"reads_total\":",
            "\"reads_by_caller_total\":{\"copy_mapped\":[",
            "\"protects_by_place_total\":{\"guard\":",
        ] {
            assert!(text.contains(key), "{key} missing: {text}");
        }
        // The names whose meaning changed are gone, not kept beside the new ones.
        for gone in ["\"parts\":", "\"dispatches\":", "\"stores_failed\":", "\"pipelines_created\":"] {
            assert!(!text.contains(gone), "{gone} is still in the object: {text}");
        }
        // The hash check's fields are there only with its switch on.
        if crate::host::hash_check::on() {
            assert!(text.contains("\"hash_missed_pieces_total\":"), "{text}");
            assert!(text.contains("\"hash_checked_pieces_total\":"), "{text}");
            assert!(text.contains("\"hash_check_skipped_total\":"), "{text}");
        } else {
            assert!(!text.contains("hash_missed_pieces"), "{text}");
            assert!(!text.contains("hash_checked_pieces"), "{text}");
            assert!(!text.contains("hash_check_skipped"), "{text}");
        }
        // Receipts are off in the test process: the object is as it was without them.
        assert!(!text.contains("texture_receipts"), "{text}");
        // So are the stale-upload trap and the textures-made counter: optional objects,
        // left out while their switches are off.
        if !crate::host::recheck::on() {
            assert!(!text.contains("texture_recheck"), "{text}");
        }
        #[cfg(feature = "diagnostics")]
        if !crate::live::texture_made() {
            assert!(!text.contains("textures_made"), "{text}");
        }
        // `VICTORY_LANDED_CLASSES`: no field while it is off (off changes nothing).
        if std::env::var_os("VICTORY_LANDED_CLASSES").is_none_or(|v| v != "1") {
            assert!(!text.contains("landed_classes"), "{text}");
        }
        // Well-formed enough to parse: braces balance and no raw control characters.
        assert_eq!(text.matches('{').count(), text.matches('}').count());
        assert!(!text.chars().any(|c| (c as u32) < 0x20));
    }

    /// The textures-made object holds only what the diagnostic can know: the
    /// new keys made and the storage requests on a clock. The classes it
    /// could not tell apart, and the range sums that meant nothing, are gone.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn the_texture_made_object_holds_no_classes() {
        let text = made_text();
        for gone in ["\"moved\"", "\"changed\"", "\"overlap\"", "\"again\"", "\"new\"", "\"words\"", "\"ranges\"", "\"minute_requests\"", "\"minute_ranges\""] {
            assert!(!text.contains(gone), "{gone}: {text}");
        }
        assert!(text.contains("\"new_keys_total\":") && text.contains("\"storage_requests_total\":"), "{text}");
    }

    /// `dispatched` is per frame over the object's frames, like its neighbours: 60 dispatches
    /// in five frames are 12.0, not 60.
    #[test]
    fn dispatches_are_per_frame() {
        let text = json(&interval(), &now(1));
        assert!(text.contains("\"draws\":6500.0,\"dispatched\":12.0,\"dropped\":2.0"), "{text}");
    }

    /// The window fields across two writes: each object covers every frame since the one
    /// before it and no other, its means are over all of them (not its last second), and a
    /// reader tells a repeated object from a new one by `seq` and the frames alone.
    #[test]
    fn an_object_says_the_frames_it_covers_and_a_repeat_is_told() {
        // What a reader takes from an object: seq, first and last frame, frames, wall ms.
        fn window(text: &str) -> [f64; 5] {
            ["\"seq\":", "\"first_frame\":", "\"last_frame\":", "\"frames\":", "\"wall_ms\":"].map(|key| {
                let from = text.find(key).unwrap_or_else(|| panic!("{key} missing: {text}")) + key.len();
                let to = text[from..].find([',', '}']).expect("a value's end") + from;
                text[from..to].parse().expect("a number")
            })
        }
        let mut meter = Interval::new();
        for (frame, ms) in [(10, 100.0), (11, 300.0), (12, 200.0)] {
            meter.present(frame, ms);
        }
        meter.sums[sum::COPIES_US] = 90_000;
        let first = json(&meter, &Now { seq: 1, ..now(1) });
        // The second object: the frames after the first's, six of them in a longer wait.
        meter.clear();
        for frame in 13..19 {
            meter.present(frame, 1000.0);
        }
        meter.sums[sum::COPIES_US] = 60_000;
        let second = json(&meter, &Now { seq: 2, ..now(2) });
        assert_eq!(window(&first), [1.0, 10.0, 12.0, 3.0, 600.0]);
        assert_eq!(window(&second), [2.0, 13.0, 18.0, 6.0, 6000.0]);
        // Means over the whole interval: 90 ms of copies in three frames, 60 ms in six.
        assert!(first.contains("\"frame_ms\":200.00") && first.contains("\"copies_ms\":30.00"), "{first}");
        assert!(second.contains("\"frame_ms\":1000.00") && second.contains("\"copies_ms\":10.00"), "{second}");
        // Contiguous: nothing between two objects is in neither.
        assert_eq!(window(&second)[1], window(&first)[2] + 1.0);
        // A repeated read is the same object: its seq did not move. A new one's did, by one.
        let again = first.clone();
        assert_eq!(window(&again)[0], window(&first)[0], "a repeat has the same seq");
        assert_eq!(window(&second)[0], window(&first)[0] + 1.0, "the next object is one later");
        // Its clocks moved too: the monotonic one and the wall clock.
        assert!(second.contains("\"t_ms\":2000.0") && first.contains("\"t_ms\":1000.0"));
        assert!(second.contains("\"unix_ms\":1800000002000"), "{second}");
    }

    /// The numbers objects take go up by one each: a reader that sees `seq` jump knows it
    /// missed objects, and how many.
    #[test]
    fn each_object_made_takes_the_next_number() {
        let a = Now::next(None, "null".to_owned()).seq;
        let b = Now::next(None, "null".to_owned()).seq;
        // Other tests make objects too: later, never the same.
        assert!(b > a, "{a} then {b}");
    }

    /// A frame line's sums are the seconds' added up, and its frames the first second's
    /// first to the last one's last: a line covers every frame since the line before it.
    #[test]
    fn a_frame_line_covers_every_second_since_the_last() {
        let mut line = Interval::new();
        let mut second = Interval::new();
        for (frame, ms) in [(1, 250.0), (2, 250.0), (3, 250.0), (4, 250.0)] {
            second.present(frame, ms);
        }
        second.sums[sum::COPIES_US] = 4 * 50_000;
        line.add(&second);
        second.clear();
        for (frame, ms) in [(5, 500.0), (6, 500.0)] {
            second.present(frame, ms);
        }
        second.sums[sum::COPIES_US] = 2 * 20_000;
        line.add(&second);
        assert_eq!((line.first_frame, line.last_frame, line.frames), (1, 6, 6));
        assert_eq!(line.wall_ms, 2000.0);
        // (4 x 50 + 2 x 20) / 6, not the last second's 20.
        assert!((line.per_frame(sum::COPIES_US, 1e3) - 40.0).abs() < 1e-9);
        assert_eq!(line.percentiles(), (250.0, 500.0));
        let window = window_text(&line);
        assert!(window.starts_with("means over frames 1..6 (6 frames in 2000 ms, p50 250.0, p95 500.0 ms/frame)"), "{window}");
    }

    /// The budget in the object: the presenting thread's phases per frame, their sum beside
    /// the frame, and the other threads' figures apart, never added into it. (Corrected with
    /// the phases: `recording` is `list_calls`, and `title` has no processor time of its own.)
    #[cfg(feature = "diagnostics")]
    #[test]
    fn the_budget_is_in_the_object_and_other_threads_are_apart() {
        use crate::budget::{PHASES, Phase};
        let mut wall = [0u64; PHASES];
        wall[Phase::Title as usize] = 2 * 100_000_000;
        wall[Phase::RelayWait as usize] = 2 * 80_000_000;
        wall[Phase::Unaccounted as usize] = 2 * 20_000_000;
        let mut others = [0u64; PHASES];
        others[Phase::RelayLand as usize] = 2 * 150_000_000;
        let mut interval = interval();
        interval.budget = Budget {
            frames: 2,
            ns: 2 * 200_000_000,
            wall,
            cpu_ns: wall.map(|ns| ns as f64 / 2.0),
            scopes: 60,
            others_wall: others,
            others_cpu_ns: others.map(|ns| ns as f64 / 4.0),
            others_scopes: 10,
            cpu_missing: false,
            ..Default::default()
        };
        let text = json(&interval, &now(1));
        assert!(
            text.contains("\"budget\":{\"thread\":\"presenting\",\"frames\":2,\"frame_ms\":200.000,\"sum_ms\":200.000,\"scopes\":30.0,\"calls\":0.0,\"largest_title_gap_ms\":0.000,\"title_and_calls_cpu_ms\":0.000,\"phases\":{\"title\":{\"ms\":100.000,\"cpu_ms\":null},\"list_calls\":{\"ms\":0.000,\"cpu_ms\":null},\"unaccounted\":{\"ms\":20.000,\"cpu_ms\":10.000}"),
            "{text}"
        );
        assert!(text.contains("\"relay_wait\":{\"ms\":80.000,\"cpu_ms\":40.000}"), "{text}");
        // The relay's 150 ms a frame are the other threads', over the interval's five frames.
        let (budget, rest) = text.split_once("\"other_threads\":").expect("the other threads' object");
        assert!(!budget.contains("relay_land"), "the relay's landing is in the presenting thread's budget: {budget}");
        assert!(rest.starts_with("{\"scopes\":2.0,\"phases\":{\"list_calls\":"), "{rest}");
        assert!(rest.contains("\"relay_land\":{\"ms\":60.000,\"cpu_ms\":15.000}"), "{rest}");
        let line = interval.budget.line(1230, 1234, 5).expect("a budget line");
        assert!(
            line.starts_with("frame budget, the presenting thread, frames 1230..1234 (2 frames): 200.00 ms/frame = title 100.00 + unaccounted 20.00 + relay_wait 80.00 (sum 200.00)"),
            "{line}"
        );
        assert!(line.ends_with("not parts of the frame: relay_land 60.00 (cpu 15.00)"), "{line}");
    }

    /// The meter's gate, which was the defect: an object is written at every second's take,
    /// each over every frame since the one before it, and a frame line every five seconds over
    /// every frame since the line before it (not the object inside the frame line's gate, one
    /// second's sample about every six). Thirteen seconds of presents 250 ms apart.
    #[test]
    fn every_present_is_in_one_object_and_in_one_frame_line() {
        use std::time::{Duration, Instant};
        let start = Instant::now();
        let mut gate = Gate::new(start);
        let (mut objects, mut lines) = (Vec::new(), Vec::new());
        for frame in 1..=52u64 {
            let now = start + Duration::from_millis(250 * frame);
            gate.second.present(frame, 250.0);
            if !gate.second_due(now) {
                continue;
            }
            // What the meter takes at a second: 10 ms of copies a frame.
            gate.second.sums[sum::COPIES_US] = gate.second.frames * 10_000;
            if gate.second_done(now, |second| objects.push((second.first_frame, second.last_frame, second.frames))) {
                lines.push((gate.line.first_frame, gate.line.last_frame, gate.line.frames));
                assert!((gate.line.per_frame(sum::COPIES_US, 1e3) - 10.0).abs() < 1e-9, "a frame line's means are over all its frames");
                gate.line_done();
            }
        }
        let every_second: Vec<(u64, u64, u64)> = (0..13).map(|second| (4 * second + 1, 4 * second + 4, 4)).collect();
        assert_eq!(objects, every_second, "an object a second, end to end");
        assert_eq!(lines, [(1, 20, 20), (21, 40, 20)], "a frame line over every frame since the one before");
    }

    /// `title` is only the title: the title's calls into the runtime are phases of their own,
    /// none of them (nor `title`) is given a processor time it was not measured with, the
    /// processor time of all of them together is said once, and the frame says how many calls
    /// it held and the largest stretch of `title` between two (what an outside trace checks,
    /// the sum being true by arithmetic).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn title_is_only_the_title_and_the_calls_are_counted() {
        use crate::budget::{NAMES, PHASES};
        let at = |name: &str| NAMES.iter().position(|n| *n == name).unwrap_or_else(|| panic!("no phase is called {name}"));
        let mut wall = [0u64; PHASES];
        wall[at("title")] = 2 * 60_000_000;
        wall[at("list_calls")] = 2 * 30_000_000;
        wall[at("object_calls")] = 2 * 6_000_000;
        wall[at("hle_calls")] = 2 * 4_000_000;
        wall[at("unaccounted")] = 2 * 100_000_000;
        let mut cpu_ns = [0f64; PHASES];
        cpu_ns[at("unaccounted")] = 2.0 * 50e6;
        let mut interval = interval();
        interval.budget = Budget {
            frames: 2,
            ns: 2 * 200_000_000,
            wall,
            cpu_ns,
            scopes: 60,
            calls: 4000,
            largest_gap: 3_500_000,
            entry_cpu_ns: 2.0 * 90e6,
            ..Default::default()
        };
        let text = json(&interval, &now(1));
        for key in [
            "\"title\":{\"ms\":60.000,\"cpu_ms\":null}",
            "\"list_calls\":{\"ms\":30.000,\"cpu_ms\":null}",
            "\"object_calls\":{\"ms\":6.000,\"cpu_ms\":null}",
            "\"hle_calls\":{\"ms\":4.000,\"cpu_ms\":null}",
            "\"unaccounted\":{\"ms\":100.000,\"cpu_ms\":50.000}",
            "\"title_and_calls_cpu_ms\":90.000",
            "\"calls\":2000.0",
            "\"largest_title_gap_ms\":3.500",
        ] {
            assert!(text.contains(key), "{key} missing: {text}");
        }
        let line = interval.budget.line(1230, 1234, 5).expect("a budget line");
        assert!(line.contains("the title and its calls together 90.00"), "{line}");
        assert!(line.contains("2000 calls from the title a frame, the largest stretch of title between two 3.50 ms"), "{line}");
    }

    /// Counts since the start say `_total`, in every object of the file, and the largest of
    /// an object's few frame times is called the worst frame, not a percentile.
    #[test]
    fn totals_say_total_and_the_worst_frame_is_called_that() {
        let text = json(&interval(), &now(7));
        for key in [
            "\"frame_ms_p50\":220.00,\"frame_ms_worst\":420.00",
            "\"segments_closed_total\":3",
            // The kernel-call counters hold a per-frame part and counts since the start, so the
            // object keeps its plain name and the counts inside it say `_total`.
            "\"kernel_calls\":{\"per_frame\":",
            "\"reads_total\":",
        ] {
            assert!(text.contains(key), "{key} missing: {text}");
        }
        // The trap's and the classifier's objects are there only while their switches are on
        // (`optional`): with both off, as here, the object has neither key.
        for gone in [
            "\"frame_ms_p95\":",
            "\"segment\":",
            "\"kernel_calls_total\":",
            "\"texture_recheck\":",
            "\"textures_made\":",
            "\"uploads\":",
            "\"moved\":",
        ] {
            assert!(!text.contains(gone), "{gone} is still in the object: {text}");
        }
    }

    /// The write is atomic and replaces: the file holds the newest object whole, and no
    /// temporary file is left beside it.
    #[test]
    fn the_write_replaces_whole() {
        let dir = std::env::temp_dir().join(format!("victory-live-stats-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a directory");
        let path = dir.join("stats.json");
        let first = json(&interval(), &now(1));
        write_atomic(&path, &first).expect("the first write");
        assert_eq!(std::fs::read_to_string(&path).expect("the file"), first);
        let second = json(&interval(), &now(2));
        write_atomic(&path, &second).expect("the second write");
        assert_eq!(std::fs::read_to_string(&path).expect("the file"), second, "replaced whole");
        let temporary = dir.join("stats.json.tmp");
        assert!(!temporary.exists(), "no temporary file is left");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// With `VICTORY_LIVE_STATS` unset, nothing is written: the path is none, and [`write`]
    /// returns at once, asking for neither the video memory nor the stamps. The gate runs
    /// with it unset.
    #[test]
    fn unset_is_inert() {
        if std::env::var_os("VICTORY_LIVE_STATS").is_some() {
            return;
        }
        assert!(path().is_none());
        write(&interval(), || panic!("the video memory was asked for"), || panic!("the stamps were asked for"));
    }

    /// With the panel off its switch is not an entry of the live map, so the object is the one
    /// a run without the panel writes.
    #[test]
    fn the_panel_is_not_in_the_live_map_while_it_is_off() {
        if crate::live::live_hud() {
            return; // a run with the panel on is not this test
        }
        let text = json(&interval(), &Now { live: crate::live::pairs(), ..now(1) });
        assert!(!text.contains("VICTORY_LIVE_HUD"), "{text}");
    }

    /// A counter the copies line zeroes is read as what was added since the last object.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_zeroed_counter_is_read_since_the_last_object() {
        let last = AtomicU64::new(0);
        assert_eq!(since_last(30, &last), 30);
        assert_eq!(since_last(50, &last), 20);
        // Zeroed by the copies line, then 7 added.
        assert_eq!(since_last(7, &last), 7);
    }

    /// A value with a quote or a backslash is escaped, so the object stays one.
    #[test]
    fn values_are_escaped() {
        assert_eq!(escape("a\"b\\c"), "a\\\"b\\\\c");
    }
}
