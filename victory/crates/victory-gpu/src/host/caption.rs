//! `VICTORY_CAPTION_STATS` ([`on`], on unless `0`): the window's caption with the
//! process's figures beside the frame rate. Off, the caption is this line's
//! own as before (`Meter::frame`'s text, [`posted`]) and nothing of this
//! module's is sampled or asked; the F3 panel and the line at the top of
//! the picture are not this module's either way. Taken from another line
//! of this project's work, which adapted it from the submitter's own
//! VIctory fork.
//!
//! On, after the window's own text:
//!
//! ` | VIctory-ALFRED 1a2b3c4 diag | 7 fps 137.8 ms/frame | gpu 3D 47%, cpu 17% of all cores, ram 22481 MB private
//! | 6396 draws/frame | VRAM 17.8 of 22.3 GB budget | up 6:13 | preparing 12.3, host waits 98.1 ms/frame
//! | RTX 5090, 32 GB, driver 591.44`
//!
//! - the build's name first ([`build_name`]): this line's own name,
//!   `VIctory-ALFRED`, and the commit's short hash as the build script read
//!   it (`unknown` outside a checkout), or `VICTORY_BUILD_NAME` in place of
//!   both, so two builds' bars tell them apart.
//! - `diag`, `release` or `debug`: how the build was made ([`kind`]).
//! - gpu 3D: this process's share of the device's adapter's busiest 3D engine
//!   ([`GpuBusy`]); copy engines are not counted. `gpu 3D -` where it cannot be read.
//! - cpu: the process's CPU time over the last sample, as a share of all
//!   logical processors together, not the busiest thread.
//! - ram: the process's private bytes now, not the peak.
//! - VRAM: the card's own memory this process uses, of the budget the
//!   system gives it (not the card's total); left out when the adapter
//!   does not report it.
//! - up: how long the process has run.
//! - preparing, host waits, last so that a narrow window cuts them first:
//!   the frame line's own figures, in ms a frame: submissions prepared on
//!   the CPU, and the host's waits for the card's work to finish (the frame
//!   line's "on the GPU"). Both are summed over the threads that spend
//!   them, so neither is a share of the frame. (This line counts them as
//!   the team line does, `SUBMISSION_US`.)
//! - the card, after them: the device's adapter's name (a leading "NVIDIA
//!   GeForce " left out), its dedicated memory in whole GB, and the
//!   user-mode driver's version (NVIDIA's own form on NVIDIA; left out when
//!   the system does not say it), so a screenshot says what it was measured
//!   on. Read once when the host is made, with the switch on only
//!   ([`card`], [`Adapter`]); left out without an adapter.
//!
//! A figure that could not be read shows as a dash.
//!
//! Presenting only hands the window's text and the second's frame figures
//! to the caption's thread, the one that sets the text with the switch off
//! too ([`posted`]); with the switch on, that thread samples the rest when
//! they come, once a second, and adds them ([`finish`], [`Sampler`]).

use std::ffi::OsStr;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use windows::Win32::Foundation::LUID;
use windows::Win32::Graphics::Dxgi::{DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO};

/// `VICTORY_CAPTION_STATS` (on unless `0`), read once: the caption shows, after the
/// window's own text and ` | VIctory`, the build's kind, frames a second
/// and ms a frame, this process's share of the GPU's busiest 3D engine,
/// its share of all the CPU's cores together (not the busiest thread,
/// which the results template asks for as a figure of its own), its
/// private bytes now (not the peak), draws a frame, its video memory
/// of its budget, its running time, and the ms a frame preparing and
/// waiting, sampled once a second on the caption's thread. Unset or
/// anything but `1`: off, the caption as before, and nothing more is
/// sampled or asked: the adapter's facts and their one-line `adapter` log
/// at start-up are read and written with the switch on only ([`card`]).
pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| switch(std::env::var_os(SWITCH).as_deref()))
}

/// The switch's name, as [`on`] reads it.
const SWITCH: &str = "VICTORY_CAPTION_STATS";

/// Whether `VICTORY_CAPTION_STATS` set to `value` (or unset) is on: on unless `0`
/// (the maintainer, 2026-11-27: the bar is part of the build; `0` gives the base's own text).
fn switch(value: Option<&OsStr>) -> bool {
    value.is_none_or(|v| v != "0")
}

/// A second of presents: frames a second, ms a frame, of a frame the ms
/// preparing submissions and the ms the host waited for the card (both
/// summed over threads), and draws a frame.
#[derive(Clone, Copy, Debug)]
pub(super) struct Figures {
    /// The rate's window: frames over the last second or more, reset at
    /// each read (`Meter::frame`, at the first present a second or more
    /// after the last read); the same count and window as the base's
    /// on-screen line (`METER_FPS_TENTHS`, stored at the same read).
    pub(super) fps: f64,
    pub(super) ms: f64,
    pub(super) preparing: f64,
    pub(super) waits: f64,
    pub(super) draws: f64,
}

/// What the caption's thread read for one caption: the GPU's and the CPU's
/// shares in percent, the private bytes, the card's memory in use and its
/// budget in bytes, and the running time; none for what it could not read.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Sample {
    gpu: Option<f64>,
    cpu: Option<f64>,
    ram: Option<u64>,
    vram: Option<(u64, u64)>,
    up: Option<Duration>,
    /// The device's adapter as [`Adapter::caption`] says it, read once when
    /// the host was made; none without an adapter.
    card: Option<&'static str>,
}

/// The build's kind: `debug` for a debug build, else `diag` with the
/// diagnostics feature (the `diag` profile) and `release` without.
fn kind(debug: bool, diagnostics: bool) -> &'static str {
    match (debug, diagnostics) {
        (true, _) => "debug",
        (false, true) => "diag",
        (false, false) => "release",
    }
}

/// The caption's text after the window's own, with the switch on: the
/// build's `name` ([`build_name`]) first; the time preparing and waiting
/// last, the first a narrow window cuts (only the card after them).
fn text(name: &str, kind: &str, figures: &Figures, sample: &Sample) -> String {
    const MB: f64 = (1u64 << 20) as f64;
    let Figures { fps, ms, preparing, waits, draws } = *figures;
    // A figure's name and value, or a dash for one that could not be read.
    let named = |name: &str, value: Option<String>| value.map_or_else(|| format!("{name} -"), |v| format!("{name} {v}"));
    let gpu = named("gpu 3D", sample.gpu.map(|share| format!("{share:.0}%")));
    let cpu = named("cpu", sample.cpu.map(|share| format!("{share:.0}% of all cores")));
    let ram = named("ram", sample.ram.map(|bytes| format!("{:.0} MB private", bytes as f64 / MB)));
    let up = named("up", sample.up.map(uptime));
    let vram = sample.vram.map(|(usage, budget)| format!(" | {}", vram(usage, budget))).unwrap_or_default();
    let card = sample.card.map(|card| format!(" | {card}")).unwrap_or_default();
    format!(
        " | {name} {kind} | {fps:.0} fps {ms:.1} ms/frame | {gpu}, {cpu}, {ram} | {draws:.0} draws/frame{vram} \
         | {up} | preparing {preparing:.1}, host waits {waits:.1} ms/frame{card}"
    )
}

/// This line's own name for itself on the bar (the public repository's
/// spelling). With the switch off the title bar keeps the base's own text,
/// `VIctory-NVIDIA` (its caption text in `host.rs`).
const OWN_NAME: &str = "VIctory-ALFRED";

/// What stands in for a `VICTORY_BUILD_NAME` that is no public label.
const FALLBACK: &str = "build";

/// The build's name at the start of the caption, read once, at the first
/// caption with the switch on (never with it off): [`OWN_NAME`] and the
/// commit's short hash the build script read (`build.rs`), or
/// `VICTORY_BUILD_NAME` in place of both where it is set and not blank. As
/// `world` reads that switch: trimmed, and, as the caption can
/// appear in a shared recording, one that is no short public label
/// ([`public`]) shows as [`FALLBACK`], said in the log without its text.
fn build_name() -> &'static str {
    static NAME: OnceLock<String> = OnceLock::new();
    NAME.get_or_init(|| {
        let name = std::env::var("VICTORY_BUILD_NAME").ok();
        if name.as_deref().map(str::trim).is_some_and(|name| !name.is_empty() && !public(name)) {
            tracing::info!(
                "VICTORY_BUILD_NAME is not a public label (1 to 40 of A-Z a-z 0-9 . _ / -, no path, address or \
                 login); the caption says {FALLBACK}"
            );
        }
        build_identity(env!("VICTORY_GPU_BUILD_COMMIT"), name.as_deref())
    })
}

/// [`build_name`] from its parts: `name` (`VICTORY_BUILD_NAME`), trimmed,
/// in place of the default if it is not blank or none, and only if it is a
/// public label, else [`FALLBACK`]; the default is [`OWN_NAME`] and the
/// commit, shown only as seven hex digits (else `unknown`).
fn build_identity(commit: &str, name: Option<&str>) -> String {
    match name.map(str::trim) {
        Some(name) if public(name) => name.to_owned(),
        Some(name) if !name.is_empty() => FALLBACK.to_owned(),
        _ => {
            let commit = if commit.len() == 7 && commit.bytes().all(|b| b.is_ascii_hexdigit()) { commit } else { "unknown" };
            format!("{OWN_NAME} {commit}")
        }
    }
}

/// The longest label that passes [`public`].
const LONGEST: usize = 40;

/// Segments that name a place rather than a build: URL schemes, a host's
/// first word, home folders.
const PLACES: &[&str] = &["http", "https", "ftp", "file", "git", "ssh", "www", "users", "home"];

/// Whether `text` may be shown as the build's name (`world`'s
/// `build_label::public`): 1 to 40 of `A-Z a-z 0-9 . _ / -`,
/// beginning with a letter or digit, not ending with `/`, without `..` or
/// `//`, and no segment (between `/`s) a place ([`PLACES`]) or of a host's
/// or a dotted login's shape (letters only after its last `.`). A
/// backslash, a drive's colon, `@`, `:`, a space and anything not ASCII
/// fail on the characters alone.
fn public(text: &str) -> bool {
    let bytes = text.as_bytes();
    let allowed = |b: &u8| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'/' | b'-');
    if bytes.is_empty()
        || bytes.len() > LONGEST
        || !bytes.iter().all(allowed)
        || !bytes[0].is_ascii_alphanumeric()
        || text.ends_with('/')
        || text.contains("..")
        || text.contains("//")
    {
        return false;
    }
    text.split('/').all(|segment| {
        let place = PLACES.iter().any(|place| segment.eq_ignore_ascii_case(place));
        let host = segment
            .rsplit_once('.')
            .is_some_and(|(_, last)| !last.is_empty() && last.bytes().all(|b| b.is_ascii_alphabetic()));
        !place && !host
    })
}

/// NVIDIA's PCI vendor id.
const NVIDIA: u32 = 0x10DE;

/// The most characters of the card's name the caption shows.
const NAME_MOST: usize = 29;

/// The three facts of the adapter the device was made on, read once when
/// the host is made ([`Adapter::read`]): its name as it describes itself,
/// its dedicated video memory in whole GB, and the user-mode driver's
/// version ([`driver_version`]; none when the query failed). Nothing
/// else of it (no ids, no LUID) is kept.
pub(super) struct Adapter {
    pub(super) name: String,
    pub(super) vram_gb: u64,
    pub(super) driver: Option<String>,
}

impl Adapter {
    /// The facts of `adapter`: its description and one driver query; none
    /// when it cannot describe itself.
    pub(super) fn read(adapter: &windows::Win32::Graphics::Dxgi::IDXGIAdapter3) -> Option<Adapter> {
        use windows::Win32::Graphics::Dxgi::IDXGIDevice;
        use windows::core::Interface;
        // SAFETY: plain queries of a live adapter.
        let (desc, driver) = unsafe { (adapter.GetDesc1().ok()?, adapter.CheckInterfaceSupport(&IDXGIDevice::IID).ok()) };
        Some(Adapter::from(&desc.Description, desc.VendorId, desc.DedicatedVideoMemory as u64, driver))
    }

    /// The facts from a description's name (UTF-16, NUL-padded), vendor id
    /// and dedicated memory in bytes, and the driver query's answer.
    fn from(name: &[u16], vendor: u32, dedicated: u64, driver: Option<i64>) -> Adapter {
        let end = name.iter().position(|&c| c == 0).unwrap_or(name.len());
        Adapter {
            name: String::from_utf16_lossy(&name[..end]).trim().to_owned(),
            vram_gb: whole_gb(dedicated),
            driver: driver_version(vendor, driver),
        }
    }

    /// What the caption says of it, last ([`card_part`]).
    pub(super) fn caption(&self) -> String {
        card_part(&card_name(&self.name), self.vram_gb, self.driver.as_deref())
    }
}

/// The card's name for the caption: as the adapter gives it, without the
/// NULs after it, the spaces around it or a leading "NVIDIA GeForce ", and
/// at most [`NAME_MOST`] characters.
fn card_name(name: &str) -> String {
    let name = name.trim_matches(|c: char| c == '\0' || c.is_whitespace());
    let name = name.strip_prefix("NVIDIA GeForce ").unwrap_or(name);
    name.chars().take(NAME_MOST).collect::<String>().trim_end().to_owned()
}

/// Bytes in whole GB (GiB), to the nearest.
fn whole_gb(bytes: u64) -> u64 {
    const GB: u64 = 1 << 30;
    bytes / GB + u64::from(bytes % GB >= GB / 2)
}

/// The user-mode driver's version from the 64-bit form the system gives
/// (four 16-bit parts, product.version.subversion.build): for NVIDIA
/// (`vendor` 0x10DE) its own form, the last digit of the third part and
/// the four of the fourth with a point before the last two and the part
/// before it padded to three digits (32.0.15.9144 is 591.44, 32.0.10.7 is
/// 000.07); for any other the four parts dotted; none for a failed query.
fn driver_version(vendor: u32, raw: Option<i64>) -> Option<String> {
    let raw = raw? as u64;
    let part = |at: u32| (raw >> at) & 0xffff;
    let (product, version, sub, build) = (part(48), part(32), part(16), part(0));
    Some(if vendor == NVIDIA {
        let n = sub % 10 * 10_000 + build % 10_000;
        format!("{:03}.{:02}", n / 100, n % 100)
    } else {
        format!("{product}.{version}.{sub}.{build}")
    })
}

/// The caption's card part: the name, the memory, and the driver where it
/// is known.
fn card_part(name: &str, vram_gb: u64, driver: Option<&str>) -> String {
    match driver {
        Some(driver) => format!("{name}, {vram_gb} GB, driver {driver}"),
        None => format!("{name}, {vram_gb} GB"),
    }
}

/// A running time as minutes and seconds, with hours once there are any.
fn uptime(up: Duration) -> String {
    let s = up.as_secs();
    let (h, m, s) = (s / 3600, s / 60 % 60, s % 60);
    if h == 0 { format!("{m}:{s:02}") } else { format!("{h}:{m:02}:{s:02}") }
}

/// The card's memory in use of its budget (not the card's total), in GB to one decimal.
fn vram(usage: u64, budget: u64) -> String {
    const GB: f64 = (1u64 << 30) as f64;
    format!("VRAM {:.1} of {:.1} GB budget", usage as f64 / GB, budget as f64 / GB)
}

/// A caption handed from presenting to the caption's thread: with the
/// switch off the whole text, NUL-terminated, as before; on, the window's
/// own text and the second's figures, for the thread to finish.
pub(super) struct Posted {
    text: Vec<u16>,
    figures: Option<Figures>,
}

/// What presenting hands the caption's thread for the window's own text
/// `base` and the second's `figures`, with the switch `on` or off: off,
/// `base` then this line's own text after it (`today`, `Meter::frame`'s,
/// called only then), the caption as before.
pub(super) fn posted(on: bool, base: &[u16], figures: Figures, today: impl FnOnce() -> String) -> Posted {
    if on {
        return Posted { text: base.to_vec(), figures: Some(figures) };
    }
    let mut text: Vec<u16> = base.iter().copied().chain(today().encode_utf16()).collect();
    text.push(0);
    Posted { text, figures: None }
}

/// The NUL-terminated text the caption's thread sets for `posted`: off,
/// the text as posted, and `sample` is not called; on, the window's text
/// and then [`text`] of one sample.
pub(super) fn finish(posted: Posted, sample: impl FnOnce() -> Sample) -> Vec<u16> {
    let Posted { mut text, figures } = posted;
    if let Some(figures) = figures {
        let kind = kind(cfg!(debug_assertions), cfg!(feature = "diagnostics"));
        text.extend(self::text(build_name(), kind, &figures, &sample()).encode_utf16());
        text.push(0);
    }
    text
}

/// The host once it is made; never makes it.
fn made() -> Option<&'static super::Host> {
    super::HOST.get()?.as_ref()
}

/// The card's part for the host being made ([`Adapter::caption`]): with the
/// switch `on`, the facts of `adapter` read once and said in one `adapter`
/// line; off, nothing is asked or said.
pub(super) fn card(on: bool, adapter: Option<&windows::Win32::Graphics::Dxgi::IDXGIAdapter3>) -> Option<String> {
    if !on {
        return None;
    }
    // One event shape either way: `adapter`, with `described` and, when it
    // is, `card`, `vram_gb` and `driver`.
    let facts = adapter.and_then(Adapter::read);
    match &facts {
        Some(a) => tracing::info!(
            described = true,
            card = %a.name,
            vram_gb = a.vram_gb,
            driver = a.driver.as_deref(),
            "adapter"
        ),
        None => tracing::info!(described = false, "adapter"),
    }
    facts.map(|a| a.caption())
}

/// What one sample read from the system, as the system gave it, before any
/// figure is made of it ([`Sampler::step`]); none for a reading that failed.
struct Readings {
    /// The logical processors (0: the query failed), asked only while the
    /// sampler has no count ([`Sampler::wants_processors`]); none when not asked.
    processors: Option<u32>,
    /// The process's creation time and CPU time, in 100 ns ([`process_times`]).
    times: Option<(u64, u64)>,
    /// The process's memory counters ([`memory_counters`]).
    memory: Option<windows_sys::Win32::System::ProcessStatus::PROCESS_MEMORY_COUNTERS_EX>,
    /// This process's running time on each 3D engine, in 100 ns ([`GpuBusy::running`]).
    gpu: Option<Vec<i64>>,
    /// The card's own memory as the adapter reports it for this process.
    vram: Option<DXGI_QUERY_VIDEO_MEMORY_INFO>,
    /// The monotonic clock when the readings were taken, and the wall clock
    /// (since the Unix epoch).
    at: Instant,
    wall: Option<Duration>,
}

/// The caption thread's samples, one set for each caption with the switch
/// on. Made with the thread, it asks nothing until its first sample.
pub(super) struct Sampler {
    gpu: GpuBusy,
    /// Each 3D engine's running time at the last sample, and when.
    gpu_last: Option<(Vec<i64>, Instant)>,
    /// The process's CPU time at the last sample, in 100 ns, and when.
    cpu: Option<(u64, Instant)>,
    /// The machine's logical processors, asked at the first sample and
    /// again at each while the count is not known (0, the query's failure).
    processors: u32,
    up: Uptime,
    /// Whether a device without an adapter for its video memory was said.
    vram_said: bool,
}

impl Sampler {
    pub(super) const fn new() -> Sampler {
        Sampler {
            gpu: GpuBusy::new(),
            gpu_last: None,
            cpu: None,
            processors: 0,
            up: Uptime::new(),
            vram_said: false,
        }
    }

    /// One caption's figures: the readings ([`Self::read`]), then the
    /// figures made of them ([`Self::step`]).
    pub(super) fn sample(&mut self) -> Sample {
        let readings = self.read();
        // The card's part, made once with the host: no query here.
        Sample { card: made().and_then(|host| host.card.as_deref()), ..self.step(readings) }
    }

    /// Whether the next sample asks for the processor count: until one
    /// that is not 0 has been read.
    fn wants_processors(&self) -> bool {
        self.processors == 0
    }

    /// One sample's readings: a statistics query for each 3D engine (one,
    /// on most cards; the engines are found at the first), the processor
    /// count while it is not known, a process time query, a memory counter
    /// query and a video memory query.
    fn read(&mut self) -> Readings {
        let gpu = self.gpu.running();
        let processors = self.wants_processors().then(processors);
        let (times, at) = (process_times(), Instant::now());
        let wall = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok();
        Readings { processors, times, memory: memory_counters(), gpu, vram: self.video_memory(), at, wall }
    }

    /// The caption's figures from one sample's readings, asking nothing: the
    /// CPU's and GPU's shares over the time since the last sample (none at
    /// the first), none for the CPU while the processor count is not known.
    fn step(&mut self, readings: Readings) -> Sample {
        let Readings { processors, times, memory, gpu, vram, at, wall } = readings;
        if let Some(count) = processors {
            self.processors = count;
        }
        let cpu = match (self.cpu, times) {
            (Some((before, then)), Some((_, now))) => {
                cpu_share(before, now, at.duration_since(then), self.processors)
            }
            _ => None,
        };
        self.cpu = times.map(|(_, cpu)| (cpu, at));
        let gpu = match (self.gpu_last.take(), gpu) {
            (Some((before, then)), Some(now)) => {
                let share = gpu_share(&before, &now, at.duration_since(then));
                self.gpu_last = Some((now, at));
                share
            }
            (None, Some(now)) => {
                self.gpu_last = Some((now, at));
                None
            }
            (_, None) => None,
        };
        let up = self.up.at(times.map(|(created, _)| created), wall, at);
        let ram = memory.map(|counters| counters.PrivateUsage as u64);
        Sample { gpu, cpu, ram, vram: vram.map(|info| (info.CurrentUsage, info.Budget)), up, card: None }
    }

    /// The card's own memory as the adapter reports it for this process:
    /// none when it does not.
    fn video_memory(&mut self) -> Option<DXGI_QUERY_VIDEO_MEMORY_INFO> {
        let host = made()?;
        let Some(adapter) = host.adapter.as_ref() else {
            if !std::mem::replace(&mut self.vram_said, true) {
                tracing::info!("caption: the system names no adapter for the device; VRAM is left out");
            }
            return None;
        };
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: a plain query that fills our structure.
        unsafe { adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }.ok()?;
        Some(info)
    }
}

/// The share of all `processors` that the CPU time between two samples
/// (`before`, `now`, in 100 ns) took over `wall`, in percent, at most 100;
/// none for a counter that went back, no processors, or less than one
/// count (100 ns) between the samples.
fn cpu_share(before: u64, now: u64, wall: Duration, processors: u32) -> Option<f64> {
    let wall = wall.as_nanos() / 100;
    if now < before || processors == 0 || wall == 0 {
        return None;
    }
    Some(((now - before) as f64 * 100.0 / wall as f64 / f64::from(processors)).min(100.0))
}

/// The process's creation time and its CPU time, kernel and user, in 100 ns.
fn process_times() -> Option<(u64, u64)> {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: our own process and four valid out pointers.
    if unsafe { GetProcessTimes(GetCurrentProcess(), &mut created, &mut exited, &mut kernel, &mut user) } == 0 {
        return None;
    }
    let ticks = |f: FILETIME| (u64::from(f.dwHighDateTime) << 32) | u64::from(f.dwLowDateTime);
    Some((ticks(created), ticks(kernel) + ticks(user)))
}

/// The process's memory counters, the extended form: the caption shows its
/// private bytes now (`PrivateUsage`, the commit charge: what the results
/// template's private RAM counts, which can differ from the working set by
/// gigabytes here), not the peak.
fn memory_counters() -> Option<windows_sys::Win32::System::ProcessStatus::PROCESS_MEMORY_COUNTERS_EX> {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS_EX { cb: size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32, ..Default::default() };
    // SAFETY: our own process and the extended counters, of the size given
    // (the call fills the extended form when given its size).
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut c).cast::<PROCESS_MEMORY_COUNTERS>(), c.cb) };
    (ok != 0).then_some(c)
}

/// The machine's logical processors, in every processor group; 0 when the
/// query fails, which [`cpu_share`] shows as no share.
fn processors() -> u32 {
    use windows_sys::Win32::System::Threading::{ALL_PROCESSOR_GROUPS, GetActiveProcessorCount};
    // SAFETY: a plain query.
    unsafe { GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) }
}

/// A FILETIME counts 100 ns from 1601; the Unix epoch is this many later.
const UNIX_EPOCH_TICKS: u64 = 116_444_736_000_000_000;

/// The process's age from its creation time (`created`, a FILETIME's 100 ns)
/// and the wall clock (`wall`, since the Unix epoch); none for a creation
/// time after the clock or before the epoch.
fn age(created: u64, wall: Duration) -> Option<Duration> {
    let created = Duration::from_nanos(created.checked_sub(UNIX_EPOCH_TICKS)?.checked_mul(100)?);
    wall.checked_sub(created)
}

/// How long the process has been running: unknown until a sample reads its
/// creation time and the wall clock (asked again at each until then), then
/// its age at that sample plus the monotonic clock since, so the queries
/// failing or the wall clock being set after do not move it.
struct Uptime {
    since: Option<(Instant, Duration)>,
}

impl Uptime {
    const fn new() -> Uptime {
        Uptime { since: None }
    }

    /// The running time at `now`, from the creation time and the wall clock
    /// as this sample read them (none for one it could not read).
    fn at(&mut self, created: Option<u64>, wall: Option<Duration>, now: Instant) -> Option<Duration> {
        if self.since.is_none() {
            self.since = Some((now, age(created?, wall?)?));
        }
        let (then, age) = self.since?;
        Some(age + now.saturating_duration_since(then))
    }
}

/// The caption's "gpu": this process's share of the device's adapter's
/// busiest 3D engine over the last sample's time ([`gpu_share`]); work on
/// copy engines (and the card's other engines) is not counted. It comes
/// from the kernel's graphics statistics (`D3DKMTQueryStatistics`): the running time
/// it counts for a process on each engine, which Task Manager's GPU column
/// and the `GPU Engine` performance counters are made from. Asked of this
/// process and this adapter alone, it is one kernel query per 3D engine a
/// sample; the performance counters would gather every process's engines
/// on every adapter at each collection. Where the statistics cannot be
/// read (a software adapter such as WARP, linked adapters), `gpu 3D -`.
struct GpuBusy {
    /// The adapter and its 3D engines' node numbers, looked up at the first
    /// sample with the host made; none when they cannot all be read (said once).
    engines: Option<Option<(LUID, Vec<u32>)>>,
    /// Whether a running time that could not be read has been said.
    said: bool,
}

impl GpuBusy {
    const fn new() -> GpuBusy {
        GpuBusy { engines: None, said: false }
    }

    /// This process's running time on each 3D engine now, in 100 ns, for
    /// the share between two samples ([`Sampler::step`]); none when the
    /// adapter's statistics cannot be read.
    fn running(&mut self) -> Option<Vec<i64>> {
        if self.engines.is_none() {
            let device = &made()?.device;
            // SAFETY: plain queries of a live device.
            let (luid, linked) = unsafe { (device.GetAdapterLuid(), device.GetNodeCount()) };
            // Linked adapters (several physical ones behind one device)
            // number their engines by physical adapter as well; not read.
            let engines = (linked == 1).then(|| engines_3d(luid).map(|nodes| (luid, nodes))).flatten();
            match &engines {
                Some((_, nodes)) => tracing::info!(
                    "caption: the GPU share is this process's running time on the busiest of {} 3D engine(s) \
                     of the device's adapter",
                    nodes.len()
                ),
                None => tracing::info!(
                    "caption: the device's adapter's 3D engines cannot all be found (no statistics, a software \
                     adapter such as WARP, a node unread, or {linked} linked adapters); it says gpu 3D -"
                ),
            }
            self.engines = Some(engines);
        }
        let (luid, nodes) = self.engines.as_ref()?.as_ref()?;
        let running = running_times(*luid, nodes);
        if running.is_none() && !std::mem::replace(&mut self.said, true) {
            tracing::info!("caption: this process's running time on the 3D engines could not be read; it says gpu 3D -");
        }
        running
    }
}

/// Running time between two samples beyond this many times the wall time
/// between them is no running time: a packet's time is counted when it
/// completes, so one that began before the interval can put more than the
/// interval in it (the engine was busy all of it), but not ten intervals'
/// worth, packets being bounded by the GPU's timeout detection.
const MOST_INTERVALS: u128 = 10;

/// The share of `wall` that the busiest 3D engine was busy with this
/// process's work, in percent, from each 3D engine's running time (in 100 ns)
/// at the two samples (`before`, `now`, engine by engine): this process's
/// share of the busiest 3D engine, copy engines not counted, so two 3D
/// engines each busy 60% of the time are 60%, not 120%. At most 100 (a
/// packet longer than the interval). None for engines that do not match or
/// none, a negative counter, one that went back (a wrap or a reset), more
/// than [`MOST_INTERVALS`] intervals of running time, or less than one
/// count (100 ns) between the samples.
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
/// for its two `BOOLEAN`s would make a byte other than 0 or 1 undefined;
/// none for a read of another size.
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
/// names no 3D engine, or it is a software adapter (WARP's: the work runs
/// on the CPU, and its engines count none of it).
fn engines_3d(luid: LUID) -> Option<Vec<u32>> {
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
fn running_times(luid: LUID, nodes: &[u32]) -> Option<Vec<i64>> {
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
            // SAFETY: a statistics query, filled in place (the binding
            // takes a const pointer; the kernel writes the result).
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

/// The newest of what presenting hands the caption's thread, and a wake
/// for it: with the switch on the thread waits here ([`Mailbox::wait`])
/// and is woken by the post itself, so the bar is set in the same second's
/// work as the line at the top of the picture, not up to 100 ms later;
/// off, it looks ([`Mailbox::take`]) and sleeps as before.
pub(super) struct Mailbox<T> {
    slot: std::sync::Mutex<Option<T>>,
    ready: std::sync::Condvar,
}

impl<T> Mailbox<T> {
    pub(super) const fn new() -> Mailbox<T> {
        Mailbox { slot: std::sync::Mutex::new(None), ready: std::sync::Condvar::new() }
    }

    /// Puts `value` in place of any not yet taken, and wakes a waiter.
    pub(super) fn post(&self, value: T) {
        *self.slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(value);
        self.ready.notify_one();
    }

    /// What was posted and not yet taken, without waiting.
    pub(super) fn take(&self) -> Option<T> {
        self.slot.lock().unwrap_or_else(|e| e.into_inner()).take()
    }

    /// What was posted, waiting for a post if none is there; the lock is
    /// let go before it returns (so before a sample).
    pub(super) fn wait(&self) -> T {
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(value) = slot.take() {
                return value;
            }
            slot = self.ready.wait(slot).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// A text set that took longer than this is counted ([`time_set`]): a
/// frame at 60 frames a second.
#[cfg(feature = "diagnostics")]
const SLOW_SET: Duration = Duration::from_millis(16);

/// Since the last frame-rate line: text sets that took over [`SLOW_SET`],
/// and the longest set in microseconds (diagnostics builds, the switch on).
#[cfg(feature = "diagnostics")]
static SETS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Runs `set` (the window's text set, which waits for the window's thread)
/// and, in diagnostics builds with the switch on, counts how long it took
/// ([`slow_sets_line`]); how the text is set is not changed.
pub(super) fn time_set(set: impl FnOnce()) {
    #[cfg(feature = "diagnostics")]
    if on() {
        let began = Instant::now();
        set();
        note_set(began.elapsed());
        return;
    }
    set();
}

/// Counts one text set that took `took`.
#[cfg(feature = "diagnostics")]
fn note_set(took: Duration) {
    use std::sync::atomic::Ordering::Relaxed;
    if took > SLOW_SET {
        SETS[0].fetch_add(1, Relaxed);
    }
    SETS[1].fetch_max(u64::try_from(took.as_micros()).unwrap_or(u64::MAX), Relaxed);
}

/// The text sets since the last call, at debug level beside the frame-rate
/// line (diagnostics builds, the switch on): whether the window's thread
/// holds the bar back.
#[cfg(feature = "diagnostics")]
pub(super) fn slow_sets_line() {
    use std::sync::atomic::Ordering::Relaxed;
    if !on() {
        return;
    }
    let [slow, longest] = [0, 1].map(|i| SETS[i].swap(0, Relaxed));
    tracing::debug!(
        target: "victory_gpu::caption",
        slow,
        longest_ms = longest as f64 / 1e3,
        "caption: window text sets over 16 ms since the last frame-rate line, and the longest"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: f64 = (1u64 << 30) as f64;

    fn figures() -> Figures {
        Figures { fps: 7.0, ms: 137.8, preparing: 12.3, waits: 98.1, draws: 6396.0 }
    }

    fn sample() -> Sample {
        Sample {
            gpu: Some(47.0),
            cpu: Some(17.0),
            ram: Some(22481 << 20),
            vram: Some(((17.8 * GB) as u64, (22.3 * GB) as u64)),
            up: Some(Duration::from_secs(373)),
            card: None,
        }
    }

    /// The build's name in the tests' expected texts: the default for a
    /// commit. `1a2b3c4` and `1a2b3c4d5` are invented
    /// hashes, not commits of any repository.
    const NAME: &str = "VIctory-ALFRED 1a2b3c4";

    /// Readings at `at` with nothing read.
    fn readings(at: Instant) -> Readings {
        Readings { processors: None, times: None, memory: None, gpu: None, vram: None, at, wall: None }
    }

    /// A processor count of 0 (the query's failure) is no count: no CPU
    /// share, and the count is asked for again at each sample until one
    /// comes, then kept.
    #[test]
    fn a_failed_processor_count_is_no_cpu_share_and_is_asked_again() {
        let s = Duration::from_secs;
        let start = Instant::now();
        // The process's CPU time so far, in 100 ns, created at the Unix epoch.
        let times = |cpu: u64| Some((UNIX_EPOCH_TICKS, cpu));
        let mut sampler = Sampler::new();
        assert!(sampler.wants_processors(), "asked at the first sample");
        let first = sampler.step(Readings { processors: Some(0), times: times(0), ..readings(start) });
        assert_eq!(first.cpu, None, "the first sample");
        assert!(sampler.wants_processors(), "asked again after a count of 0");
        // A second of CPU time in a second: all of one processor, 12.5% of eight.
        let failed = sampler.step(Readings { processors: Some(0), times: times(10_000_000), ..readings(start + s(1)) });
        assert_eq!(failed.cpu, None, "no share without a count");
        assert!(sampler.wants_processors(), "asked again after a second count of 0");
        let counted = sampler.step(Readings { processors: Some(8), times: times(20_000_000), ..readings(start + s(2)) });
        assert_eq!(counted.cpu, Some(12.5), "a share once the count came");
        assert!(!sampler.wants_processors(), "not asked once known");
        let kept = sampler.step(Readings { times: times(60_000_000), ..readings(start + s(3)) });
        assert_eq!(kept.cpu, Some(50.0), "the count kept");
    }

    /// The caption's ram is the memory counters' private bytes, not their
    /// working set, which can differ from them by gigabytes.
    #[test]
    fn the_caption_shows_the_private_bytes_not_the_working_set() {
        use windows_sys::Win32::System::ProcessStatus::PROCESS_MEMORY_COUNTERS_EX;
        let counters =
            PROCESS_MEMORY_COUNTERS_EX { WorkingSetSize: 9_000 << 20, PrivateUsage: 22_481 << 20, ..Default::default() };
        let sample = Sampler::new().step(Readings { memory: Some(counters), ..readings(Instant::now()) });
        assert_eq!(sample.ram, Some(22_481 << 20));
        let text = text(NAME, "diag", &figures(), &sample);
        assert!(text.contains(", ram 22481 MB private |"), "{text}");
    }

    /// The GPU's share through the step: none at the first reading and at
    /// the first after one that failed, else the busiest 3D engine's share
    /// between two readings.
    #[test]
    fn the_gpu_share_is_between_two_readings() {
        let s = Duration::from_secs;
        let start = Instant::now();
        let gpu = |running: &[i64]| Some(running.to_vec());
        let mut sampler = Sampler::new();
        assert_eq!(sampler.step(Readings { gpu: gpu(&[0, 0]), ..readings(start) }).gpu, None, "the first");
        let busy = sampler.step(Readings { gpu: gpu(&[5_000_000, 1_000_000]), ..readings(start + s(1)) });
        assert_eq!(busy.gpu, Some(50.0));
        assert_eq!(sampler.step(readings(start + s(2))).gpu, None, "a reading that failed");
        let after = sampler.step(Readings { gpu: gpu(&[6_000_000, 1_000_000]), ..readings(start + s(3)) });
        assert_eq!(after.gpu, None, "the first after it");
        let again = sampler.step(Readings { gpu: gpu(&[8_500_000, 1_000_000]), ..readings(start + s(4)) });
        assert_eq!(again.gpu, Some(25.0));
    }

    /// The live readers fill what the step reads: the memory counters in
    /// their extended form (the private bytes, which the plain form lacks),
    /// and a creation time before now.
    #[test]
    fn the_live_readers_fill_what_the_step_reads() {
        let counters = memory_counters().expect("the process's memory counters");
        assert!(counters.PrivateUsage > 0, "the extended form filled");
        let (created, _) = process_times().expect("the process's times");
        let wall = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("the wall clock");
        assert!(age(created, wall).is_some(), "created before now");
    }

    /// This line's own text after the window's, as `Meter::frame` makes it.
    fn today(fps: f64, ms: f64, draws: f64, shown_per_frame: u32) -> String {
        super::super::meter_text(fps, ms, draws, shown_per_frame)
    }

    /// Off, the caption's thread sets this line's text as before and
    /// samples nothing; on, it samples once for each caption and adds the
    /// figures after the window's own text, and this line's text is not
    /// made.
    #[test]
    fn off_the_caption_thread_samples_nothing() {
        let base: Vec<u16> = "Window".encode_utf16().collect();
        let (mut made, mut calls) = (0, 0);
        let off = finish(
            posted(false, &base, figures(), || {
                made += 1;
                today(7.0, 137.8, 6396.0, 1)
            }),
            || {
                calls += 1;
                sample()
            },
        );
        assert_eq!((made, calls), (1, 0), "the off path sampled, or did not make this line's text");
        assert_eq!(String::from_utf16_lossy(&off), "Window | VIctory-NVIDIA 7 fps 137.8 ms/frame 6396 draws/frame\0");
        let on = finish(
            posted(true, &base, figures(), || {
                made += 1;
                today(7.0, 137.8, 6396.0, 1)
            }),
            || {
                calls += 1;
                sample()
            },
        );
        assert_eq!((made, calls), (1, 1), "one sample for a caption with the switch on, and not this line's text");
        let kind = kind(cfg!(debug_assertions), cfg!(feature = "diagnostics"));
        let on = String::from_utf16_lossy(&on);
        assert_eq!(on, format!("Window{}\0", text(build_name(), kind, &figures(), &sample())));
        // On, the bar after the window's own text begins with the build's
        // name: by default this line's own and the commit's short hash.
        assert!(on.starts_with(&format!("Window | {} {kind} | 7 fps", build_name())), "{on}");
        if std::env::var("VICTORY_BUILD_NAME").is_ok_and(|name| !name.trim().is_empty()) {
            return;
        }
        let commit = on.strip_prefix("Window | VIctory-ALFRED ").and_then(|rest| rest.split(' ').next());
        assert!(
            commit.is_some_and(|c| c == "unknown" || (c.len() == 7 && c.bytes().all(|b| b.is_ascii_hexdigit()))),
            "the default name: this line's own and the commit's short hash: {on}"
        );
    }

    /// The build's name: by default this line's own and the commit's short
    /// hash (seven hex digits, else `unknown`); `VICTORY_BUILD_NAME`,
    /// trimmed, in place of both where it is not blank.
    #[test]
    fn the_build_is_named_by_this_lines_name_and_the_commit() {
        assert_eq!(build_identity("1a2b3c4", None), NAME);
        assert_eq!(build_identity("1a2b3c4", Some("neon")), "neon");
        assert_eq!(build_identity("1a2b3c4", Some(" neon-2.1\n")), "neon-2.1", "the name is trimmed");
        for blank in ["", "  ", "\t\n"] {
            assert_eq!(build_identity("1a2b3c4", Some(blank)), "VIctory-ALFRED 1a2b3c4", "a blank name {blank:?} is none");
        }
        assert_eq!(build_identity("C:/x", None), "VIctory-ALFRED unknown");
        assert_eq!(build_identity("1a2b3c4d5", None), "VIctory-ALFRED unknown");
        assert_eq!(build_identity("1a2b3c4+dirty", None), "VIctory-ALFRED unknown");
        assert_eq!(option_env!("VICTORY_GPU_BUILD_BRANCH"), None, "the build script reads no branch");
        let own = env!("VICTORY_GPU_BUILD_COMMIT");
        assert!(own == "unknown" || (own.len() == 7 && own.bytes().all(|b| b.is_ascii_hexdigit())), "{own:?}");
    }

    /// The caption can appear in a shared recording: a `VICTORY_BUILD_NAME`
    /// that is no short public label (a path, an address, a login) shows as
    /// `build` (`world`'s cases).
    #[test]
    fn the_caption_shows_only_public_labels() {
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
            assert_eq!(build_identity("1a2b3c4", Some(&text)), "build", "{text:?} as the name");
        }
        let forty = "x".repeat(40);
        for public in ["world", "main", "job/title-bar", "neon", "fork-2.1", "v0.62.2", "A_b.9", &forty] {
            assert_eq!(build_identity("1a2b3c4", Some(public)), public, "{public:?}");
        }
    }

    /// Off, with no adapter facts read (the switch off at the host's
    /// making), and on without an adapter, there is no card.
    #[test]
    fn off_no_adapter_is_asked() {
        assert_eq!(card(false, None), None, "off: no card");
        assert_eq!(card(true, None), None, "on without an adapter: no card");
    }

    /// A post wakes the caption's thread waiting for it at once, not at its
    /// next look of the old 100 ms: every one of five rounds is taken well
    /// within 50 ms of its post (a look every 100 ms would miss that in
    /// about half of them). Each post is taken before the next is made: the
    /// mailbox keeps only the newest, so a second post before the waiter
    /// ran would replace the first, and a waiter counting five would then
    /// wait for ever (it hung a loaded workspace run). A post not taken
    /// within 10 s fails the test instead.
    #[test]
    fn a_post_wakes_the_waiting_thread_at_once() {
        static BOX: Mailbox<(u32, Instant)> = Mailbox::new();
        let (told, taken_rx) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            for _ in 0..5 {
                let (n, at) = BOX.wait();
                if told.send((n, at.elapsed())).is_err() {
                    return;
                }
            }
        });
        let mut taken = Vec::new();
        for n in 0..5 {
            // The thread is waiting by now, or takes the post as soon as it
            // waits (a post before the wait is not lost).
            std::thread::sleep(Duration::from_millis(30));
            BOX.post((n, Instant::now()));
            taken.push(taken_rx.recv_timeout(Duration::from_secs(10)).expect("the post taken within 10 s"));
        }
        waiter.join().expect("the waiting thread");
        assert_eq!(taken.iter().map(|(n, _)| *n).collect::<Vec<_>>(), [0, 1, 2, 3, 4], "each post taken, in order");
        for (n, after) in &taken {
            assert!(*after < Duration::from_millis(50), "post {n} taken {after:?} after it");
        }
    }

    /// A post made before the thread waits is taken at once (no wake is
    /// lost), the newest in place of an older one not taken, and nothing is
    /// left after it.
    #[test]
    fn a_post_before_the_wait_is_not_lost() {
        let mailbox = Mailbox::new();
        assert_eq!(mailbox.take(), None);
        mailbox.post(1);
        mailbox.post(2);
        assert_eq!(mailbox.wait(), 2, "the newest");
        assert_eq!(mailbox.take(), None, "taken once");
    }

    /// Text sets over 16 ms are counted and the longest kept, until the
    /// next frame-rate line.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn slow_text_sets_are_counted() {
        use std::sync::atomic::Ordering::Relaxed;
        note_set(Duration::from_millis(17));
        note_set(Duration::from_millis(5));
        note_set(Duration::from_millis(16));
        note_set(Duration::from_millis(40));
        let [slow, longest] = [0, 1].map(|i| SETS[i].swap(0, Relaxed));
        assert_eq!((slow, longest), (2, 40_000), "over 16 ms only, the longest in microseconds");
    }

    /// The switch's name is the one its comment and the docs give.
    #[test]
    fn the_switch_is_read_by_its_name() {
        assert_eq!(SWITCH, "VICTORY_CAPTION_STATS");
    }

    /// `VICTORY_CAPTION_STATS` is on unless `0`: unset or any other word shows
    /// the full caption; `0` alone leaves the caption as the base had it.
    #[test]
    fn the_switch_is_on_unless_0() {
        assert!(switch(None), "unset");
        assert!(!switch(Some(OsStr::new("0"))), "0");
        for other in ["", "1", "on", "true", "yes", " 0", "0 ", "00", "2"] {
            assert!(switch(Some(OsStr::new(other))), "{other:?}");
        }
    }

    /// Off, the caption's text after the window's own is exactly this
    /// line's before the switch: frames a second (and, with frame
    /// generation, the frames shown), ms a frame, draws a frame.
    #[test]
    fn off_the_caption_is_the_one_before() {
        assert_eq!(today(7.0, 137.8, 6396.0, 1), " | VIctory-NVIDIA 7 fps 137.8 ms/frame 6396 draws/frame");
        assert_eq!(today(29.6, 33.84, 1525.6, 1), " | VIctory-NVIDIA 30 fps 33.8 ms/frame 1526 draws/frame");
        assert_eq!(
            today(29.6, 33.84, 1525.6, 2),
            " | VIctory-NVIDIA 30 fps (frame gen 59 fps) 33.8 ms/frame 1526 draws/frame"
        );
        assert_eq!(
            today(29.6, 33.84, 1525.6, 3),
            " | VIctory-NVIDIA 30 fps (frame gen 89 fps) 33.8 ms/frame 1526 draws/frame"
        );
    }

    /// On, the caption keeps the prefix and names the build's kind, then
    /// frames a second and ms a frame, the GPU's share, the CPU's of all
    /// cores, the private bytes, draws a frame, video memory against its
    /// budget, the running time, and last (the first a narrow window cuts)
    /// the time preparing and waiting.
    #[test]
    fn on_the_caption_says_the_processs_figures() {
        assert_eq!(
            text(NAME, "diag", &figures(), &sample()),
            " | VIctory-ALFRED 1a2b3c4 diag | 7 fps 137.8 ms/frame | gpu 3D 47%, cpu 17% of all cores, ram 22481 MB private \
             | 6396 draws/frame | VRAM 17.8 of 22.3 GB budget | up 6:13 | preparing 12.3, host waits 98.1 ms/frame"
        );
    }

    /// Video memory is left out where the adapter does not report it.
    #[test]
    fn on_the_caption_leaves_video_memory_out_when_it_is_not_reported() {
        let sample = Sample { vram: None, ..sample() };
        assert_eq!(
            text(NAME, "release", &figures(), &sample),
            " | VIctory-ALFRED 1a2b3c4 release | 7 fps 137.8 ms/frame | gpu 3D 47%, cpu 17% of all cores, ram 22481 MB private \
             | 6396 draws/frame | up 6:13 | preparing 12.3, host waits 98.1 ms/frame"
        );
    }

    /// The running time shows hours once there are any.
    #[test]
    fn on_the_caption_shows_hours_once_there_are_any() {
        let sample = Sample { up: Some(Duration::from_secs(3723)), ..sample() };
        let text = text(NAME, "diag", &figures(), &sample);
        assert!(text.ends_with(" | VRAM 17.8 of 22.3 GB budget | up 1:02:03 | preparing 12.3, host waits 98.1 ms/frame"), "{text}");
        assert_eq!(uptime(Duration::ZERO), "0:00");
        assert_eq!(uptime(Duration::from_secs(65)), "1:05");
        assert_eq!(uptime(Duration::from_secs(754)), "12:34");
        assert_eq!(uptime(Duration::from_secs(3600)), "1:00:00");
        assert_eq!(uptime(Duration::from_secs(36_000 + 59)), "10:00:59");
    }

    /// A figure that could not be read shows as a dash, never as a number.
    #[test]
    fn on_the_caption_says_a_dash_for_what_it_could_not_read() {
        let sample = Sample { gpu: None, cpu: None, ram: None, vram: None, up: None, card: None };
        assert_eq!(
            text(NAME, "diag", &figures(), &sample),
            " | VIctory-ALFRED 1a2b3c4 diag | 7 fps 137.8 ms/frame | gpu 3D -, cpu -, ram - | 6396 draws/frame | up - \
             | preparing 12.3, host waits 98.1 ms/frame"
        );
    }

    /// On, the adapter's card, memory and driver come last, after the
    /// frame line's figures, so a narrow window cuts them first.
    #[test]
    fn on_the_caption_ends_with_the_card() {
        let sample = Sample { card: Some("RTX 5090, 32 GB, driver 591.44"), ..sample() };
        assert_eq!(
            text(NAME, "diag", &figures(), &sample),
            " | VIctory-ALFRED 1a2b3c4 diag | 7 fps 137.8 ms/frame | gpu 3D 47%, cpu 17% of all cores, ram 22481 MB private \
             | 6396 draws/frame | VRAM 17.8 of 22.3 GB budget | up 6:13 | preparing 12.3, host waits 98.1 ms/frame \
             | RTX 5090, 32 GB, driver 591.44"
        );
    }

    /// Off, the card is not in the caption, whatever the host knows of it.
    #[test]
    fn off_the_caption_has_no_card() {
        let base: Vec<u16> = "Window".encode_utf16().collect();
        let off = finish(posted(false, &base, figures(), || today(7.0, 137.8, 6396.0, 1)), || Sample {
            card: Some("RTX 5090, 32 GB"),
            ..sample()
        });
        assert_eq!(String::from_utf16_lossy(&off), "Window | VIctory-NVIDIA 7 fps 137.8 ms/frame 6396 draws/frame\0");
    }

    /// The card's name as the adapter gives it, a leading "NVIDIA GeForce "
    /// left out, the NULs after it and spaces around it dropped, and at
    /// most [`NAME_MOST`] characters.
    #[test]
    fn the_cards_name_is_short() {
        assert_eq!(card_name("NVIDIA GeForce RTX 5090"), "RTX 5090");
        assert_eq!(card_name("NVIDIA GeForce RTX 4070 Laptop GPU\0\0\0"), "RTX 4070 Laptop GPU");
        assert_eq!(card_name("  NVIDIA GeForce GTX 1080 Ti "), "GTX 1080 Ti");
        assert_eq!(card_name("NVIDIA RTX A6000"), "NVIDIA RTX A6000", "not a GeForce: kept");
        assert_eq!(card_name("AMD Radeon RX 7900 XTX"), "AMD Radeon RX 7900 XTX");
        assert_eq!(card_name("Microsoft Basic Render Driver"), "Microsoft Basic Render Driver");
        let long = card_name("A card with a name far longer than any title bar wants");
        assert_eq!(long, "A card with a name far longer");
        assert_eq!(long.chars().count(), NAME_MOST);
        assert_eq!(card_name("Карта Карта Карта Карта Карта Карта"), "Карта Карта Карта Карта Карта", "cut by characters");
        assert_eq!(card_name("\0"), "");
    }

    /// Dedicated video memory in whole GB, to the nearest: a 32 GB card
    /// reports a little under 32 GiB.
    #[test]
    fn dedicated_memory_is_in_whole_gigabytes() {
        let mib = |m: u64| m << 20;
        assert_eq!(whole_gb(mib(32_607)), 32, "an RTX 5090");
        assert_eq!(whole_gb(mib(24_156)), 24, "an RTX 4090");
        assert_eq!(whole_gb(8 << 30), 8);
        assert_eq!(whole_gb(mib(1_535)), 1);
        assert_eq!(whole_gb(mib(1_536)), 2, "a half rounds up");
        assert_eq!(whole_gb(0), 0, "the software adapter has none of its own");
        assert_eq!(whole_gb(u64::MAX), (u64::MAX >> 30) + 1, "rounds up without overflow");
    }

    /// The user-mode driver's version from its 64-bit form: NVIDIA's own
    /// form (the last digit of the third part and the four of the fourth,
    /// a point before the last two) for vendor 0x10DE, else the four parts
    /// dotted; none when the query failed.
    #[test]
    fn the_drivers_version_reads_as_its_vendor_writes_it() {
        let raw = |a: i64, b: i64, c: i64, d: i64| (a << 48) | (b << 32) | (c << 16) | d;
        assert_eq!(driver_version(NVIDIA, Some(raw(32, 0, 15, 9144))).as_deref(), Some("591.44"));
        assert_eq!(driver_version(NVIDIA, Some(raw(31, 0, 15, 5222))).as_deref(), Some("552.22"));
        assert_eq!(driver_version(NVIDIA, Some(raw(32, 0, 15, 612))).as_deref(), Some("506.12"), "a fourth part under 1000");
        assert_eq!(driver_version(NVIDIA, Some(raw(32, 0, 15, 7))).as_deref(), Some("500.07"));
        assert_eq!(
            driver_version(NVIDIA, Some(raw(32, 0, 10, 7))).as_deref(),
            Some("000.07"),
            "the major part padded to three digits"
        );
        assert_eq!(driver_version(0x1002, Some(raw(32, 0, 21025, 1000))).as_deref(), Some("32.0.21025.1000"), "AMD");
        assert_eq!(driver_version(0x1414, Some(raw(10, 0, 26100, 1))).as_deref(), Some("10.0.26100.1"), "the software adapter");
        assert_eq!(
            driver_version(0x8086, Some(raw(0xffff, 0xffff, 0xffff, 0xffff))).as_deref(),
            Some("65535.65535.65535.65535"),
            "every part unsigned"
        );
        assert_eq!(driver_version(NVIDIA, None), None, "the query failed");
        assert_eq!(driver_version(0x1002, None), None, "the query failed");
    }

    /// The caption's card part: the name, the memory, and the driver
    /// where it is known; left out without it.
    #[test]
    fn the_card_part_leaves_out_an_unknown_driver() {
        assert_eq!(card_part("RTX 5090", 32, Some("591.44")), "RTX 5090, 32 GB, driver 591.44");
        assert_eq!(card_part("RTX 5090", 32, None), "RTX 5090, 32 GB");
        assert_eq!(
            card_part("Microsoft Basic Render Driver", 0, Some("10.0.26100.1")),
            "Microsoft Basic Render Driver, 0 GB, driver 10.0.26100.1"
        );
        assert!(card_part("RTX 5090", 32, Some("591.44")).len() <= 30, "short enough for the bar");
    }

    /// The facts from an adapter's description and its driver query.
    #[test]
    fn the_adapters_facts_come_from_its_description() {
        let mut name = [0u16; 128];
        for (to, from) in name.iter_mut().zip("NVIDIA GeForce RTX 5090".encode_utf16()) {
            *to = from;
        }
        let facts = Adapter::from(&name, NVIDIA, 32_607 << 20, Some((32 << 48) | (15 << 16) | 9144));
        assert_eq!(facts.name, "NVIDIA GeForce RTX 5090", "the log's name is the adapter's own");
        assert_eq!(facts.vram_gb, 32);
        assert_eq!(facts.driver.as_deref(), Some("591.44"));
        assert_eq!(facts.caption(), "RTX 5090, 32 GB, driver 591.44");
        let failed = Adapter::from(&name, NVIDIA, 32_607 << 20, None);
        assert_eq!(failed.caption(), "RTX 5090, 32 GB");
    }

    /// The build's kind: `diag` for release code with the diagnostics
    /// feature (the `diag` profile), `release` without, `debug` for a
    /// debug build.
    #[test]
    fn the_build_is_named_by_its_kind() {
        assert_eq!(kind(false, true), "diag");
        assert_eq!(kind(false, false), "release");
        assert_eq!(kind(true, true), "debug");
        assert_eq!(kind(true, false), "debug");
    }

    #[test]
    fn video_memory_is_in_gigabytes_to_one_decimal() {
        assert_eq!(vram(1 << 30, 2 << 30), "VRAM 1.0 of 2.0 GB budget");
        assert_eq!(vram((9.4 * GB) as u64, (30.1 * GB) as u64), "VRAM 9.4 of 30.1 GB budget");
    }

    /// The process's CPU time (in 100 ns) over the wall time between two
    /// samples, as a share of all logical processors.
    #[test]
    fn the_cpu_share_is_of_all_processors() {
        let second = Duration::from_secs(1);
        assert_eq!(cpu_share(0, 10_000_000, second, 1), Some(100.0));
        assert_eq!(cpu_share(0, 80_000_000, second, 16), Some(50.0), "8 s of CPU time in a second of 16 processors");
        assert_eq!(cpu_share(5, 5, second, 8), Some(0.0));
        assert_eq!(cpu_share(0, 10_000_000, Duration::from_millis(500), 4), Some(50.0));
        assert_eq!(cpu_share(0, 30_000_000, second, 2), Some(100.0), "at most all of them");
        assert_eq!(cpu_share(10, 5, second, 8), None, "a counter that went back");
        assert_eq!(cpu_share(0, 5, Duration::ZERO, 8), None, "no time between the samples");
        assert_eq!(cpu_share(0, 5, second, 0), None, "no processors");
    }

    /// The GPU's share from synthetic counters (each 3D engine's running
    /// time, in 100 ns): the busiest engine's share of the time between two
    /// samples, at most all of it; none for counters that went back, wrapped
    /// or do not match, or no time between.
    #[test]
    fn the_gpu_share_is_the_busiest_engines_share_between_two_samples() {
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
        assert_eq!(gpu_share(&[-5], &[5], second), None, "a negative counter");
        assert_eq!(gpu_share(&[0], &[5], Duration::ZERO), None, "no time between the samples");
        assert_eq!(gpu_share(&[0], &[5], Duration::from_nanos(50)), None, "less than one count between the samples");
        assert_eq!(gpu_share(&[0, 0], &[5], second), None, "engines that do not match");
        assert_eq!(gpu_share(&[], &[], second), None, "no engines");
    }

    /// The 3D engines are found from every node's metadata or not at all
    /// (`gpu 3D -`): a node unread leaves the discovery incomplete. The
    /// metadata is read as bytes, its engine type after the node's number,
    /// whatever the bytes of its two `BOOLEAN`s hold.
    #[test]
    fn the_gpu_engines_come_from_every_nodes_metadata_or_none() {
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

    /// The running time, from controlled times: none (`up -`) until the
    /// creation time and the wall clock are read, asked again at each
    /// sample; then the age at that sample plus the monotonic clock,
    /// whatever the queries or the wall clock do after.
    #[test]
    fn the_running_time_waits_for_the_creation_time_then_counts_on() {
        let s = Duration::from_secs;
        let start = Instant::now();
        // Created 1000 s after the Unix epoch, in FILETIME's 100 ns from 1601.
        let created = UNIX_EPOCH_TICKS + 1_000 * 10_000_000;
        let mut up = Uptime::new();
        assert_eq!(up.at(None, Some(s(1_100)), start), None, "the creation time not read");
        assert_eq!(up.at(Some(created), None, start + s(1)), None, "the wall clock not read");
        assert_eq!(up.at(Some(created), Some(s(1_102)), start + s(2)), Some(s(102)), "read at last");
        assert_eq!(up.at(None, None, start + s(7)), Some(s(107)), "the queries failing after");
        assert_eq!(up.at(Some(created), Some(s(5)), start + s(12)), Some(s(112)), "the wall clock set back");
    }

    /// The process's age from its creation time and the wall clock; none
    /// for a creation time after the clock or before the Unix epoch.
    #[test]
    fn the_age_is_from_the_creation_time_and_the_wall_clock() {
        let s = Duration::from_secs;
        let created = UNIX_EPOCH_TICKS + 1_000 * 10_000_000;
        assert_eq!(age(created, s(1_373)), Some(s(373)));
        assert_eq!(age(created, s(1_000)), Some(Duration::ZERO));
        assert_eq!(age(created, s(999)), None, "created after the clock");
        assert_eq!(age(UNIX_EPOCH_TICKS - 1, s(10)), None, "created before 1970");
    }
}
