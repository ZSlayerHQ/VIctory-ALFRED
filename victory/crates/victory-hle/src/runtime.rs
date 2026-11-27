//! Process-wide state the implementations need.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

static MAIN_IMAGE_BASE: AtomicUsize = AtomicUsize::new(0);

/// Records where the title's main image is mapped. The title expects
/// `GetModuleHandle(NULL)` and friends to return this, not our own exe.
pub fn set_main_image(base: usize) {
    MAIN_IMAGE_BASE.store(base, Ordering::Release);
}

pub fn main_image() -> Option<usize> {
    match MAIN_IMAGE_BASE.load(Ordering::Acquire) {
        0 => None,
        base => Some(base),
    }
}

/// Original file of each title image we mapped, by base address.
static IMAGE_PATHS: Mutex<Vec<(usize, PathBuf)>> = Mutex::new(Vec::new());

/// Records the file a title image was mapped from: the title asks for its
/// own path (`GetModuleFileName`) to find its data, and must not get our
/// cache copy.
pub fn set_image_path(base: usize, path: &Path) {
    let path = std::path::absolute(path).unwrap_or_else(|_| path.to_owned());
    let mut paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths.retain(|(b, _)| *b != base);
    paths.push((base, path));
}

/// Forgets a title image that was unmapped again.
pub fn forget_image(base: usize) {
    let mut paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths.retain(|(b, _)| *b != base);
}

pub fn image_path(base: usize) -> Option<PathBuf> {
    let paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths
        .iter()
        .find(|(b, _)| *b == base)
        .map(|(_, p)| p.clone())
}

/// The title image whose original file is called `file` (any case).
pub fn image_named(file: &str) -> Option<usize> {
    let paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths
        .iter()
        .find(|(_, p)| p.file_name().is_some_and(|n| n.eq_ignore_ascii_case(file)))
        .map(|(b, _)| *b)
}

/// Maps a title DLL the title loads itself (`LoadLibrary`) and returns its
/// base; `Ok(None)` if `file` is not one of the title's DLLs.
pub type TitleDllLoader = Box<dyn Fn(&str) -> Result<Option<usize>, String> + Send + Sync>;

static TITLE_DLL_LOADER: std::sync::OnceLock<TitleDllLoader> = std::sync::OnceLock::new();

/// Set once, by whoever mapped the title (it knows the folders and the
/// resolver); later calls are ignored.
pub fn set_title_dll_loader(loader: TitleDllLoader) {
    let _ = TITLE_DLL_LOADER.set(loader);
}

pub fn load_title_dll(file: &str) -> Result<Option<usize>, String> {
    TITLE_DLL_LOADER.get().map_or(Ok(None), |load| load(file))
}

/// What the title's manifest (`MicrosoftGame.config` beside its executable)
/// says about it, and where its persistent local storage lives.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Title {
    pub id: u32,
    /// The store's product id (empty if the manifest has none).
    pub store_id: String,
    /// Persistent local storage: our folder for it and the title's quota.
    /// Game saves live beside it ([`Title::saves`]).
    pub storage: PathBuf,
    pub storage_bytes: u64,
}

impl Title {
    /// Id, store id and storage quota from the manifest (zero or empty
    /// where it has none); the storage folder is left for the caller.
    pub fn from_manifest(text: &str) -> Title {
        let elements = elements(text);
        let get = |path: &str| {
            elements
                .iter()
                .find(|(p, _)| p == path)
                .map(|(_, text)| text.as_str())
        };
        let id = get("Game/TitleId").and_then(|v| u32::from_str_radix(v, 16).ok());
        let storage_mb = get("Game/PersistentLocalStorage/SizeMB").and_then(|v| v.parse().ok());
        Title {
            id: id.unwrap_or(0),
            store_id: get("Game/StoreId").unwrap_or_default().to_owned(),
            storage: PathBuf::new(),
            storage_bytes: storage_mb.unwrap_or(0) << 20,
        }
    }

    /// Where game saves live: next to persistent local storage, under the
    /// data directory, never the cache.
    pub fn saves(&self) -> PathBuf {
        self.storage.with_file_name("saves")
    }
}

/// Each element of an XML text as its path from the root (`Game/TitleId`)
/// and its trimmed text up to the first child, the predefined entities
/// decoded; comments, CDATA sections, declarations and attributes are
/// skipped.
fn elements(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut path: Vec<&str> = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find('<') {
        rest = &rest[open..];
        let end = if rest.starts_with("<!--") {
            rest.find("-->").map(|e| e + 3)
        } else if rest.starts_with("<![CDATA[") {
            rest.find("]]>").map(|e| e + 3)
        } else {
            rest.find('>').map(|e| e + 1)
        };
        let Some(end) = end else { break };
        let tag = &rest[1..end - 1];
        rest = &rest[end..];
        if tag.starts_with(['!', '?']) {
            continue;
        }
        if let Some(name) = tag.strip_prefix('/') {
            if let Some(at) = path.iter().rposition(|n| *n == name.trim()) {
                path.truncate(at);
            }
            continue;
        }
        let closed = tag.ends_with('/');
        let name = tag
            .trim_end_matches('/')
            .split_whitespace()
            .next()
            .unwrap_or("");
        path.push(name);
        let body = rest[..rest.find('<').unwrap_or(rest.len())].trim();
        let body = [("&lt;", "<"), ("&gt;", ">"), ("&quot;", "\""), ("&apos;", "'"), ("&amp;", "&")]
            .iter()
            .fold(body.to_owned(), |b, (entity, c)| b.replace(entity, c));
        out.push((path.join("/"), body));
        if closed {
            path.pop();
        }
    }
    out
}

static TITLE: Mutex<Option<Title>> = Mutex::new(None);

pub fn set_title(title: Title) {
    *TITLE.lock().unwrap_or_else(|e| e.into_inner()) = Some(title);
}

/// The running title; the default (id 0, no storage) until set.
pub fn title() -> Title {
    TITLE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .unwrap_or_default()
}

/// Gives the loader's emulation of the console's instructions what only the
/// runtime can decide: the write tracking's opener, so that a `clzero` into
/// a page the tracking holds read-only is noted as a title write would be,
/// and whether `rdpru` may run and what it reads on a processor without it
/// ([`rdpru_register`]).
/// Once, after `victory_loader::report_crashes` and before title code.
#[cfg(windows)]
pub fn hook_console_instructions() {
    victory_loader::set_console_insn_hooks(victory_loader::ConsoleInsnHooks {
        open_tracked: victory_gpu::kernel_will_write,
        rdpru_enabled: victory_loader::time_stamp_counter_readable,
        rdpru_register,
    });
}

/// What `rdpru` reads here when the processor lacks it (every Intel one, and
/// AMD's before Zen 2), where the loader emulates it. User code on a PC can
/// read neither register, so both are the time-stamp counter, read at the
/// call:
///
/// - MPERF counts at the P0 frequency while the core runs (PPR 55803 p. 97);
///   on the console's Zen 2 the TSC counts at that same rate in every P- and
///   C-state (pp. 43, 96). So the TSC keeps MPERF's rate, as the title's own
///   RDTSC sees it, though it also counts while a core sleeps.
/// - APERF counts the clocks the core actually ran (p. 97), which nothing
///   here can see: the TSC again, as if the core always ran at P0.
///
/// A title that divides one by the other, as AMD's effective frequency does
/// (APERF's count over MPERF's, times the P0 frequency: p. 39), divides two
/// spans of one constant-rate counter that cover nearly the same time: close
/// to one, and so P0. The divisor is never zero, since two readings are two
/// exceptions apart (microseconds) and the TSC rises in between, and on one
/// core the TSC never goes back, so neither span is negative. What is lost
/// is the change in frequency the real pair would show. Where this PC denies
/// user code the TSC as well (CR4.TSD, which the loader's start-up probe
/// finds), the `rdpru_enabled` hook (that probe's answer) says no and the
/// loader refuses the instruction for every ECX, as the console's CPU would
/// under CR4.TSD (#UD, APM3 p. 308); this hook then gives nothing either.
#[cfg(windows)]
fn rdpru_register(register: victory_loader::RdpruRegister) -> Option<u64> {
    rdpru_stand_in(register, victory_loader::time_stamp_counter_readable(), || {
        // SAFETY: RDTSC runs in user code here: the loader's probe said so.
        unsafe { std::arch::x86_64::_rdtsc() }
    })
}

/// [`rdpru_register`]'s choice, given whether user code may read the TSC
/// and the reading.
#[cfg(windows)]
fn rdpru_stand_in(register: victory_loader::RdpruRegister, readable: bool, tsc: impl FnOnce() -> u64) -> Option<u64> {
    use victory_loader::RdpruRegister::{Aperf, Mperf};
    match register {
        Mperf | Aperf => readable.then(tsc),
    }
}

/// Gives the process the console's millisecond timers. Titles pace
/// themselves with short waits (an audio mixer sleeping a few
/// milliseconds per block, say) and cannot tune the timer on the console;
/// a desktop process waits in 15.6 ms ticks unless it asks for finer ones,
/// and Windows 11 drops that request while the window is hidden unless the
/// process opts out. Process-wide, for the whole run.
#[cfg(windows)]
pub fn use_console_timers() {
    use windows_sys::Win32::Media::timeBeginPeriod;
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, PROCESS_POWER_THROTTLING_STATE,
        ProcessPowerThrottling, SetProcessInformation,
    };
    let keep = PROCESS_POWER_THROTTLING_STATE {
        Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
        StateMask: 0,
    };
    // SAFETY: plain process settings; `keep` outlives the call.
    unsafe {
        SetProcessInformation(
            GetCurrentProcess(),
            ProcessPowerThrottling,
            (&raw const keep).cast(),
            size_of_val(&keep) as u32,
        );
        timeBeginPeriod(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_title_id_store_id_and_storage_quota() {
        let text = "\u{feff}<?xml version=\"1.0\"?>\n<Game configVersion=\"1\">\
                    <!-- <TitleId>00000000</TitleId> -->\
                    <Identity Name=\"x\"/>\n  <TitleId>0A1B2C3D</TitleId>\n\
                    <![CDATA[ <TitleId>1</TitleId> ]]>                    <StoreId>9&amp;C</StoreId><Other><SizeMB>7</SizeMB></Other>\
                    <PersistentLocalStorage>\n  <SizeMB>1024</SizeMB> <!-- c -->\n\
                    </PersistentLocalStorage></Game>";
        let t = Title::from_manifest(text);
        assert_eq!((t.id, t.storage_bytes), (0x0A1B_2C3D, 1024 << 20));
        assert_eq!(t.store_id, "9&C");
        assert_eq!(Title::from_manifest("").storage_bytes, 0);
    }

    #[cfg(windows)]
    #[test]
    fn rdpru_reads_the_time_stamp_counter_for_both_registers_or_nothing() {
        use victory_loader::RdpruRegister::{Aperf, Mperf};
        // SAFETY: Windows lets user code run RDTSC on the machines these
        // tests run on (the loader's probe tests show it).
        let tsc = || unsafe { std::arch::x86_64::_rdtsc() };
        // Two readings of each, as a title takes them for the effective
        // frequency: both spans grow, and their ratio is near one.
        let read = |register| rdpru_stand_in(register, true, tsc).expect("readable");
        let (m1, a1) = (read(Mperf), read(Aperf));
        std::thread::sleep(std::time::Duration::from_millis(5));
        let (m2, a2) = (read(Mperf), read(Aperf));
        assert!(m2 > m1 && a2 > a1 && a1 >= m1 && m2 > a1);
        let ratio = (a2 - a1) as f64 / (m2 - m1) as f64;
        assert!((0.9..1.1).contains(&ratio), "{ratio}");
        // Where the probe found the TSC denied to user code (CR4.TSD), no
        // reading at all: the loader then refuses the instruction.
        for register in [Mperf, Aperf] {
            assert_eq!(rdpru_stand_in(register, false, || unreachable!("not read")), None);
        }
    }
}
