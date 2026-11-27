//! Kernel calls the runtime makes on memory, counted so that the
//! kernel-free copies work and the threading design are sized by numbers
//! (`jobs/2026-11-26-cbj-review`, the review's "count first";
//! `jobs/2026-11-26-live-switches/opus-review-runtime-diagnostics.md`).
//!
//! Every such call in `victory-gpu` and `victory-hle` is made by a wrapper
//! here, which counts it: reads of memory through the kernel
//! (`ReadProcessMemory`) by caller, size class and outcome, with the bytes
//! asked and the bytes moved apart; write-backs (`NtWriteVirtualMemory`,
//! `WriteProcessMemory`); protection changes (`VirtualProtect`) and
//! questions (`VirtualQuery`, `VirtualQueryEx`, `QueryWorkingSetEx`) by
//! place; and write watching (`GetWriteWatch`, `ResetWriteWatch`) with the
//! pages asked and found. A source scan in the tests fails if any of these
//! calls is made anywhere else (`tests::no_memory_call_bypasses_the_counters`).
//!
//! Not seen: calls the title makes straight to the system. With the guard
//! ledger off (`VICTORY_GUARD_LEDGER`, the default) its imports of
//! `VirtualProtect` and `VirtualQuery` bind to the host's and are not ours
//! to count; with it on, and for `XMemVirtualQuery` always, the runtime makes
//! them and counts them under `title`. Every thread is counted, not only the
//! submitting one. A `WriteProcessMemory` (`VICTORY_DIRECT_WRITES=0`) makes
//! queries and protection changes of its own inside the system, which are
//! not counted here.
//!
//! Always on: a few relaxed atomic adds a call, no lock, no allocation. A
//! line every 600 frames (per frame over the last 600, and since the start)
//! and the same figures in the live stats file under `kernel_calls`.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Who reads memory through the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Caller {
    /// A submission's whole copy (`copy_mapped`).
    CopyMapped = 0,
    /// One region of a failed copy, cut by `VirtualQuery` (`read_sparse`).
    ReadSparse = 1,
    /// One 64 KiB or 4 KiB piece of a failed copy (`read_pieces`).
    ReadPieces = 2,
    /// A caller outside the copy path (`read_into`).
    ReadInto = 3,
    /// `readable`'s one-byte probe of a page the working set does not hold.
    Readable = 4,
    /// The stale-upload trap's own reads (`VICTORY_TEXTURE_RECHECK`).
    Recheck = 5,
    /// A pending kernel write's I/O status (`watch::io_done`).
    IoStatus = 6,
}
pub(crate) const CALLERS: usize = 7;
const CALLER_LIST: [Caller; CALLERS] = [
    Caller::CopyMapped,
    Caller::ReadSparse,
    Caller::ReadPieces,
    Caller::ReadInto,
    Caller::Readable,
    Caller::Recheck,
    Caller::IoStatus,
];

/// The size classes of a read: one byte, up to 4 KiB, up to 64 KiB, larger.
const CLASSES: usize = 4;
const CLASS_NAMES: [&str; CLASSES] = ["1B", "<=4K", "<=64K", ">64K"];

/// The class of a read of `len` bytes.
pub(crate) fn class_of(len: usize) -> usize {
    match len {
        1 => 0,
        2..=4096 => 1,
        4097..=65536 => 2,
        _ => 3,
    }
}

/// The caller's name as the line and the stats file print it.
fn caller_name(caller: Caller) -> &'static str {
    match caller {
        Caller::CopyMapped => "copy_mapped",
        Caller::ReadSparse => "read_sparse",
        Caller::ReadPieces => "read_pieces",
        Caller::ReadInto => "read_into",
        Caller::Readable => "readable",
        Caller::Recheck => "recheck",
        Caller::IoStatus => "io_status",
    }
}

/// One (caller, size class): calls that worked and their bytes; calls that
/// failed, the bytes they asked for and the bytes they moved before failing.
struct Counts {
    ok_calls: AtomicU64,
    ok_bytes: AtomicU64,
    failed_calls: AtomicU64,
    failed_asked: AtomicU64,
    failed_moved: AtomicU64,
}

impl Counts {
    const fn new() -> Self {
        Self {
            ok_calls: AtomicU64::new(0),
            ok_bytes: AtomicU64::new(0),
            failed_calls: AtomicU64::new(0),
            failed_asked: AtomicU64::new(0),
            failed_moved: AtomicU64::new(0),
        }
    }
}

/// One `(caller, class)` a row, flattened at `caller * CLASSES + class`.
static READS: [Counts; CALLERS * CLASSES] = [const { Counts::new() }; CALLERS * CLASSES];

/// A failed whole read that fell back to pieces, by the caller that failed.
static FALLBACKS: [AtomicU64; CALLERS] = [const { AtomicU64::new(0) }; CALLERS];

/// Write-backs, `NtWriteVirtualMemory` (0) and `WriteProcessMemory` (1):
/// calls, bytes asked, bytes written, calls that wrote less than asked.
static WRITEBACKS: [[AtomicU64; 4]; 2] = [const { [const { AtomicU64::new(0) }; 4] }; 2];

#[cfg(test)]
thread_local! {
    /// Tests: every count this thread made, by (kind, bucket), so a fixture
    /// sees its own calls whatever other tests count meanwhile.
    static HERE: std::cell::RefCell<std::collections::HashMap<(&'static str, &'static str), u64>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Tests: adds `n` to this thread's (kind, bucket).
#[cfg(test)]
fn tally(kind: &'static str, bucket: &'static str, n: u64) {
    HERE.with(|h| *h.borrow_mut().entry((kind, bucket)).or_default() += n);
}

/// Tests: what this thread counted under (kind, bucket).
#[cfg(test)]
pub(crate) fn here(kind: &'static str, bucket: &'static str) -> u64 {
    HERE.with(|h| h.borrow().get(&(kind, bucket)).copied().unwrap_or(0))
}

/// One kernel read by `caller` of `len` bytes, of which `moved` arrived;
/// `ok` says whether the call worked.
fn count_read(caller: Caller, len: usize, moved: usize, ok: bool) {
    #[cfg(test)]
    {
        tally("read", caller_name(caller), 1);
        if !ok {
            tally("read_failed_asked", caller_name(caller), len as u64);
            tally("read_failed_moved", caller_name(caller), moved as u64);
        }
    }
    let counts = &READS[caller as usize * CLASSES + class_of(len)];
    if ok {
        counts.ok_calls.fetch_add(1, Relaxed);
        counts.ok_bytes.fetch_add(moved as u64, Relaxed);
    } else {
        counts.failed_calls.fetch_add(1, Relaxed);
        counts.failed_asked.fetch_add(len as u64, Relaxed);
        counts.failed_moved.fetch_add(moved as u64, Relaxed);
    }
}

/// A failed whole read by `caller` went to its pieces.
pub(crate) fn fallback(caller: Caller) {
    FALLBACKS[caller as usize].fetch_add(1, Relaxed);
}

/// Which call wrote title memory back.
#[derive(Clone, Copy)]
pub(crate) enum Writeback {
    /// `NtWriteVirtualMemory` (the default path).
    Nt,
    /// `WriteProcessMemory` (`VICTORY_DIRECT_WRITES=0`).
    Process,
}

/// One write-back of `len` bytes of which `written` landed.
fn count_writeback(kind: Writeback, len: usize, written: usize) {
    let counts = &WRITEBACKS[match kind {
        Writeback::Nt => 0,
        Writeback::Process => 1,
    }];
    counts[0].fetch_add(1, Relaxed);
    counts[1].fetch_add(len as u64, Relaxed);
    counts[2].fetch_add(written as u64, Relaxed);
    if written < len {
        counts[3].fetch_add(1, Relaxed);
    }
}

/// Where the runtime changes or asks a page's state.
#[derive(Clone, Copy)]
pub(crate) enum Place {
    /// The write guards (`watch`): laying, opening and asking a block.
    Guard = 0,
    /// In-place title views (`inplace`), and its probe's own memory.
    Inplace = 1,
    /// A failed copy's sparse fallback (`read_sparse`).
    Copy = 2,
    /// The texture path: a committed range asked page by page.
    Texture = 3,
    /// The title's own calls, which the runtime makes for it
    /// (`title_virtual_protect`, `title_virtual_query`, `XMemVirtualQuery`).
    Title = 4,
    /// The copy statistics' own questions (diagnostics builds): the walk of
    /// the address space for the memory map, and the copied-ranges line.
    /// Without diagnostics nothing asks there, and the place stays at 0.
    #[cfg(feature = "diagnostics")]
    CopyStats = 5,
}
const PLACES: usize = 6;
const PLACE_NAMES: [&str; PLACES] = ["guard", "inplace", "copy", "texture", "title", "copystat"];

/// `VirtualProtect` calls by place, `VirtualQuery` calls by place, and
/// `QueryWorkingSetEx` calls and the 4 KiB pages they asked about by place.
static PROTECT: [AtomicU64; PLACES] = [const { AtomicU64::new(0) }; PLACES];
static QUERY: [AtomicU64; PLACES] = [const { AtomicU64::new(0) }; PLACES];
static WORKING_SET: [[AtomicU64; 2]; PLACES] = [const { [const { AtomicU64::new(0) }; 2] }; PLACES];

/// Write watching: `GetWriteWatch` calls, the ones that failed, the 4 KiB
/// pages asked about and the pages reported written; `ResetWriteWatch`
/// calls and their pages.
static WRITE_WATCH: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
const WRITE_WATCH_NAMES: [&str; 6] = ["calls", "failed", "pages_asked", "pages_written", "resets", "reset_pages"];

fn count_protect(place: Place) {
    #[cfg(test)]
    tally("protect", PLACE_NAMES[place as usize], 1);
    PROTECT[place as usize].fetch_add(1, Relaxed);
}

fn count_query(place: Place) {
    #[cfg(test)]
    tally("query", PLACE_NAMES[place as usize], 1);
    QUERY[place as usize].fetch_add(1, Relaxed);
}

/// One write-watch counter `which` ([`WRITE_WATCH_NAMES`]) up by `n`.
fn count_write_watch(which: usize, n: u64) {
    #[cfg(test)]
    tally("write_watch", WRITE_WATCH_NAMES[which], n);
    WRITE_WATCH[which].fetch_add(n, Relaxed);
}

/// The Windows calls, made here alone.
#[cfg(windows)]
mod sys {
    pub(super) use windows_sys::Win32::Foundation::HANDLE;
    pub(super) use windows_sys::Win32::System::Diagnostics::Debug::{ReadProcessMemory, WriteProcessMemory};
    pub(super) use windows_sys::Win32::System::Memory::{
        GetWriteWatch, MEMORY_BASIC_INFORMATION, ResetWriteWatch, VirtualProtect, VirtualQuery, VirtualQueryEx,
    };
    pub(super) use windows_sys::Win32::System::ProcessStatus::K32QueryWorkingSetEx;
    pub(super) use windows_sys::Win32::System::Threading::GetCurrentProcess;

    #[link(name = "ntdll", kind = "raw-dylib")]
    unsafe extern "system" {
        /// The kernel's write into a process's memory, `(process, address,
        /// bytes, length, written out)`: an NTSTATUS, negative when a page of
        /// the range is not there or not writable. It changes no protection.
        pub(super) fn NtWriteVirtualMemory(
            process: *mut std::ffi::c_void,
            base: *mut std::ffi::c_void,
            buffer: *const std::ffi::c_void,
            size: usize,
            written: *mut usize,
        ) -> i32;
    }
}

/// `ReadProcessMemory` of our own process for `caller`: `len` bytes from
/// `src` into `dst`. True if all were read; the kernel checks the source and
/// reports a page that is not there instead of faulting.
///
/// # Safety
///
/// `dst` is writable memory of `len` bytes.
#[cfg(windows)]
pub(crate) unsafe fn read_memory(caller: Caller, src: *const u8, dst: *mut u8, len: usize) -> bool {
    let mut done = 0;
    // SAFETY: the caller's destination; the kernel checks the source.
    let ok = unsafe { sys::ReadProcessMemory(sys::GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) } != 0;
    count_read(caller, len, done.min(len), ok);
    ok
}

/// Writes `n` bytes from `from` into our own process's memory at `at` by
/// `kind`'s call; the bytes that landed (a call that fails may have written
/// a prefix, one that succeeds may have written less than asked).
///
/// # Safety
///
/// `from` is readable memory of `n` bytes; the kernel checks the
/// destination and writes nothing it cannot.
#[cfg(windows)]
pub(crate) unsafe fn write_memory(kind: Writeback, at: usize, from: *const u8, n: usize) -> usize {
    let mut done = 0;
    // SAFETY: as the caller vouches.
    match kind {
        Writeback::Nt => unsafe {
            sys::NtWriteVirtualMemory(sys::GetCurrentProcess(), at as _, from.cast(), n, &mut done);
        },
        Writeback::Process => unsafe {
            sys::WriteProcessMemory(sys::GetCurrentProcess(), at as _, from.cast(), n, &mut done);
        },
    }
    let written = done.min(n);
    count_writeback(kind, n, written);
    written
}

/// `VirtualQuery` of the page at `at` for `place`: the answer's length, 0
/// if the system has none (`info` then says nothing).
#[cfg(windows)]
pub(crate) fn virtual_query(place: Place, at: usize, info: &mut sys::MEMORY_BASIC_INFORMATION) -> usize {
    count_query(place);
    // SAFETY: any address may be asked about; the answer goes to `info`,
    // which is the size given.
    unsafe { sys::VirtualQuery(at as *const _, info, size_of::<sys::MEMORY_BASIC_INFORMATION>()) }
}

/// [`virtual_query`] with the caller's pointer and length, as the title
/// passes them.
///
/// # Safety
///
/// As `VirtualQuery`'s: `out` points at room for `len` bytes.
#[cfg(windows)]
pub(crate) unsafe fn virtual_query_raw(
    place: Place,
    at: *const std::ffi::c_void,
    out: *mut sys::MEMORY_BASIC_INFORMATION,
    len: usize,
) -> usize {
    count_query(place);
    // SAFETY: as the caller vouches.
    unsafe { sys::VirtualQuery(at, out, len) }
}

/// The title's `VirtualQueryEx` (`victory-hle`'s `XMemVirtualQuery` without
/// the guard ledger), counted under `title`.
///
/// # Safety
///
/// As `VirtualQueryEx`'s: `out` points at room for `len` bytes.
#[cfg(windows)]
pub unsafe fn title_virtual_query_ex(
    process: sys::HANDLE,
    at: *const std::ffi::c_void,
    out: *mut sys::MEMORY_BASIC_INFORMATION,
    len: usize,
) -> usize {
    count_query(Place::Title);
    // SAFETY: as the caller vouches.
    unsafe { sys::VirtualQueryEx(process, at, out, len) }
}

/// `VirtualProtect` for `place`; the call's result.
///
/// # Safety
///
/// As `VirtualProtect`'s: only the protection of `[at, at + len)` changes,
/// and the caller knows nothing of its own needs a write there; `old` is
/// null or points at room for a `u32`.
#[cfg(windows)]
pub(crate) unsafe fn virtual_protect(place: Place, at: *const std::ffi::c_void, len: usize, new: u32, old: *mut u32) -> i32 {
    count_protect(place);
    // SAFETY: as the caller vouches.
    unsafe { sys::VirtualProtect(at, len, new, old) }
}

/// `GetWriteWatch` (0 when it worked), counted with the 4 KiB pages of the
/// range asked about and, when it worked, the pages it reported written.
///
/// # Safety
///
/// As `GetWriteWatch`'s: `addresses` has room for `*count` entries.
#[cfg(windows)]
pub(crate) unsafe fn get_write_watch(
    flags: u32,
    base: *const std::ffi::c_void,
    size: usize,
    addresses: *mut *mut std::ffi::c_void,
    count: &mut usize,
    granularity: &mut u32,
) -> u32 {
    // SAFETY: as the caller vouches.
    let failed = unsafe { sys::GetWriteWatch(flags, base, size, addresses, count, granularity) };
    count_write_watch(0, 1);
    count_write_watch(2, size.div_ceil(4096) as u64);
    if failed == 0 {
        count_write_watch(3, *count as u64);
    } else {
        count_write_watch(1, 1);
    }
    failed
}

/// `ResetWriteWatch` (0 when it worked), counted with its 4 KiB pages.
///
/// # Safety
///
/// As `ResetWriteWatch`'s: `[base, base + size)` is write-watched memory.
#[cfg(windows)]
pub(crate) unsafe fn reset_write_watch(base: *const std::ffi::c_void, size: usize) -> u32 {
    count_write_watch(4, 1);
    count_write_watch(5, size.div_ceil(4096) as u64);
    // SAFETY: as the caller vouches.
    unsafe { sys::ResetWriteWatch(base, size) }
}

/// `QueryWorkingSetEx` of our own process for `place`, over `pages` entries
/// of `cb` bytes in all at `info`; whether it worked.
///
/// # Safety
///
/// `info` holds `cb` bytes of entries, each with the address to ask about.
#[cfg(windows)]
pub(crate) unsafe fn working_set(place: Place, info: *mut std::ffi::c_void, cb: u32, pages: usize) -> bool {
    #[cfg(test)]
    tally("working_set", PLACE_NAMES[place as usize], 1);
    WORKING_SET[place as usize][0].fetch_add(1, Relaxed);
    WORKING_SET[place as usize][1].fetch_add(pages as u64, Relaxed);
    // SAFETY: as the caller vouches.
    unsafe { sys::K32QueryWorkingSetEx(sys::GetCurrentProcess(), info, cb) != 0 }
}

/// The calls since the start in the classes a system trace counts them by:
/// reads (`NtReadVirtualMemory`), writes (`NtWriteVirtualMemory`, and
/// `WriteProcessMemory`'s write), protects (`NtProtectVirtualMemory`),
/// queries (`NtQueryVirtualMemory`: `VirtualQuery` and `QueryWorkingSetEx`)
/// and write watching (`NtGetWriteWatch`, `NtResetWriteWatch`).
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct Totals {
    reads: u64,
    writes: u64,
    protects: u64,
    queries: u64,
    write_watch: u64,
}

fn totals() -> Totals {
    let reads = READS.iter().map(|c| c.ok_calls.load(Relaxed) + c.failed_calls.load(Relaxed)).sum();
    Totals {
        reads,
        writes: WRITEBACKS.iter().map(|w| w[0].load(Relaxed)).sum(),
        protects: PROTECT.iter().map(|c| c.load(Relaxed)).sum(),
        queries: QUERY.iter().map(|c| c.load(Relaxed)).sum::<u64>()
            + WORKING_SET.iter().map(|w| w[0].load(Relaxed)).sum::<u64>(),
        write_watch: WRITE_WATCH[0].load(Relaxed) + WRITE_WATCH[4].load(Relaxed),
    }
}

/// The totals at the last stats object, for the next one's figures per frame.
static LAST_OBJECT: std::sync::Mutex<Option<Totals>> = std::sync::Mutex::new(None);

/// The figures as the stats file's object (fixed key order), by the
/// object's rule: `per_frame`, the calls a frame over the object's `frames`
/// (all threads) in a system trace's classes, `null` in the first object;
/// then counts since the start (`_total`): the same classes, every caller
/// with its four size classes of [`READ_FIELDS`], the fallbacks by caller,
/// the write-backs, protects, queries and working-set questions by place,
/// and write watching.
pub(crate) fn text(frames: u64) -> String {
    let t = totals();
    let before = LAST_OBJECT.lock().unwrap_or_else(|e| e.into_inner()).replace(t);
    let per_frame = before.map_or_else(
        || "null".to_owned(),
        |before| {
            let [r, w, p, q, g] = per_frame(t, before, frames);
            format!("{{\"reads\":{r:.1},\"writes\":{w:.1},\"protects\":{p:.1},\"queries\":{q:.1},\"write_watch\":{g:.1}}}")
        },
    );
    let rows: Vec<String> = CALLER_LIST
        .iter()
        .map(|&caller| {
            let classes: Vec<String> = (0..CLASSES)
                .map(|class| {
                    let c = &READS[caller as usize * CLASSES + class];
                    format!(
                        "[{},{},{},{},{}]",
                        c.ok_calls.load(Relaxed),
                        c.ok_bytes.load(Relaxed),
                        c.failed_calls.load(Relaxed),
                        c.failed_asked.load(Relaxed),
                        c.failed_moved.load(Relaxed)
                    )
                })
                .collect();
            format!("\"{}\":[{}]", caller_name(caller), classes.join(","))
        })
        .collect();
    let fallbacks: Vec<String> = CALLER_LIST
        .iter()
        .map(|&c| format!("\"{}\":{}", caller_name(c), FALLBACKS[c as usize].load(Relaxed)))
        .collect();
    let by_place = |counts: &[AtomicU64; PLACES]| -> String {
        let pairs: Vec<String> = (0..PLACES).map(|i| format!("\"{}\":{}", PLACE_NAMES[i], counts[i].load(Relaxed))).collect();
        format!("{{{}}}", pairs.join(","))
    };
    let working: Vec<String> = (0..PLACES)
        .map(|i| format!("\"{}\":[{},{}]", PLACE_NAMES[i], WORKING_SET[i][0].load(Relaxed), WORKING_SET[i][1].load(Relaxed)))
        .collect();
    let watch: Vec<String> = (0..WRITE_WATCH.len())
        .map(|i| format!("\"{}\":{}", WRITE_WATCH_NAMES[i], WRITE_WATCH[i].load(Relaxed)))
        .collect();
    let write = |i: usize, name: &str| {
        let w = &WRITEBACKS[i];
        format!(
            "\"write_{name}_calls_total\":{},\"write_{name}_bytes_asked_total\":{},\"write_{name}_bytes_written_total\":{},\"write_{name}_short_total\":{}",
            w[0].load(Relaxed),
            w[1].load(Relaxed),
            w[2].load(Relaxed),
            w[3].load(Relaxed)
        )
    };
    format!(
        "{{\"per_frame\":{per_frame},\"reads_total\":{},\"writes_total\":{},\"protects_total\":{},\"queries_total\":{},\
         \"write_watch_total\":{},\"classes\":[{names}],\"read_fields\":[{fields}],\"reads_by_caller_total\":{{{reads}}},\
         \"fallbacks_by_caller_total\":{{{fallbacks}}},{nt},{process},\"protects_by_place_total\":{protect},\
         \"queries_by_place_total\":{query},\"working_set_by_place_total\":{{{working}}},\"write_watch_parts_total\":{{{watch}}}}}",
        t.reads,
        t.writes,
        t.protects,
        t.queries,
        t.write_watch,
        names = CLASS_NAMES.iter().map(|n| format!("\"{n}\"")).collect::<Vec<_>>().join(","),
        reads = rows.join(","),
        fields = READ_FIELDS.iter().map(|n| format!("\"{n}\"")).collect::<Vec<_>>().join(","),
        fallbacks = fallbacks.join(","),
        nt = write(0, "nt"),
        process = write(1, "process"),
        protect = by_place(&PROTECT),
        query = by_place(&QUERY),
        working = working.join(","),
        watch = watch.join(","),
    )
}

/// The five numbers of each (caller, class) in the stats file, in order.
const READ_FIELDS: [&str; 5] = ["ok_calls", "ok_bytes", "failed_calls", "failed_asked", "failed_moved"];

/// The totals and the frame at the last line, for the per-frame figures.
static LAST_LINE: std::sync::Mutex<Option<(u64, Totals)>> = std::sync::Mutex::new(None);

/// What `now` adds to `then` over `frames` frames, per frame.
fn per_frame(now: Totals, then: Totals, frames: u64) -> [f64; 5] {
    let f = frames.max(1) as f64;
    [
        now.reads.saturating_sub(then.reads) as f64 / f,
        now.writes.saturating_sub(then.writes) as f64 / f,
        now.protects.saturating_sub(then.protects) as f64 / f,
        now.queries.saturating_sub(then.queries) as f64 / f,
        now.write_watch.saturating_sub(then.write_watch) as f64 / f,
    ]
}

/// One line every 600 frames: the calls per frame since the last line (all
/// threads), in a system trace's classes, then since the start: reads by
/// caller (calls, failed calls, megabytes moved), fallbacks, write-backs,
/// protects and queries by place, working-set questions and write watching.
pub(crate) fn frame(frame: u64) {
    if !frame.is_multiple_of(600) {
        return;
    }
    let now = totals();
    let then = {
        let mut last = LAST_LINE.lock().unwrap_or_else(|e| e.into_inner());
        match last.replace((frame, now)) {
            Some((at, _)) if at == frame => return,
            Some(then) => Some(then),
            None => None,
        }
    };
    let window = match then {
        Some((at, before)) if at < frame => {
            let [r, w, p, q, g] = per_frame(now, before, frame - at);
            format!(
                "per frame over the last {} frames: reads {r:.0}, writes {w:.0}, reads and writes {:.0}, protects {p:.0}, queries {q:.0}, write watching {g:.0}",
                frame - at,
                r + w
            )
        }
        _ => "per frame: from the next line".to_owned(),
    };
    let mut parts: Vec<String> = Vec::new();
    for caller in CALLER_LIST {
        let (mut calls, mut failed, mut bytes) = (0u64, 0u64, 0u64);
        for class in 0..CLASSES {
            let c = &READS[caller as usize * CLASSES + class];
            calls += c.ok_calls.load(Relaxed) + c.failed_calls.load(Relaxed);
            failed += c.failed_calls.load(Relaxed);
            bytes += c.ok_bytes.load(Relaxed) + c.failed_moved.load(Relaxed);
        }
        parts.push(format!("{} {calls} ({failed} failed, {} MB moved)", caller_name(caller), bytes / (1 << 20)));
    }
    let places = |counts: &[AtomicU64; PLACES]| -> String {
        (0..PLACES).map(|i| format!("{} {}", PLACE_NAMES[i], counts[i].load(Relaxed))).collect::<Vec<_>>().join(", ")
    };
    let fallbacks: u64 = FALLBACKS.iter().map(|c| c.load(Relaxed)).sum();
    let working: (u64, u64) = WORKING_SET.iter().fold((0, 0), |(c, p), w| (c + w[0].load(Relaxed), p + w[1].load(Relaxed)));
    tracing::info!(
        "kernel calls on memory, all threads, {window}; since the start: reads {} ({}), fallbacks {fallbacks}; write-backs Nt {} ({} short) WriteProcess {} ({} short); protects {} ({}); queries {} ({}; working set {} calls over {} pages); write watching {} askings ({} failed, {} pages asked, {} written) and {} resets",
        now.reads,
        parts.join(", "),
        WRITEBACKS[0][0].load(Relaxed),
        WRITEBACKS[0][3].load(Relaxed),
        WRITEBACKS[1][0].load(Relaxed),
        WRITEBACKS[1][3].load(Relaxed),
        now.protects,
        places(&PROTECT),
        now.queries,
        places(&QUERY),
        working.0,
        working.1,
        WRITE_WATCH[0].load(Relaxed),
        WRITE_WATCH[1].load(Relaxed),
        WRITE_WATCH[2].load(Relaxed),
        WRITE_WATCH[3].load(Relaxed),
        WRITE_WATCH[4].load(Relaxed),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The size classes are told apart at their edges.
    #[test]
    fn the_classes_of_a_read_are_told_apart() {
        assert_eq!([1, 2, 4096].map(class_of), [0, 1, 1]);
        assert_eq!([4097, 65536].map(class_of), [2, 2]);
        assert_eq!([65537, 1 << 20].map(class_of), [3, 3]);
    }

    /// A counted read lands in its caller's class, ok or failed, and the
    /// figures print every caller in order.
    #[test]
    fn a_read_is_counted_where_it_was_made() {
        count_read(Caller::Readable, 1, 0, false);
        let counts = &READS[Caller::Readable as usize * CLASSES + class_of(1)];
        assert!(counts.failed_calls.load(Relaxed) >= 1);
        let text = text(1);
        assert!(text.contains("\"readable\":["), "{text}");
        assert!(text.contains("\"classes\":[\"1B\",\"<=4K\",\"<=64K\",\">64K\"]"), "{text}");
    }

    /// The stats object's figures follow its rule: per frame over its frames
    /// once an object came before (here: 30 reads more in 5 frames), and
    /// counts since the start with `_total`.
    #[test]
    fn the_stats_figures_are_per_frame_or_totals() {
        let then = Totals { reads: 100, writes: 10, protects: 4, queries: 0, write_watch: 5 };
        let now = Totals { reads: 130, writes: 10, protects: 9, queries: 20, write_watch: 5 };
        assert_eq!(per_frame(now, then, 5), [6.0, 0.0, 1.0, 4.0, 0.0]);
        text(5);
        let text = text(5);
        assert!(text.starts_with("{\"per_frame\":{\"reads\":"), "{text}");
        for key in ["\"reads_total\":", "\"write_watch_total\":", "\"reads_by_caller_total\":", "\"write_nt_calls_total\":", "\"protects_by_place_total\":"] {
            assert!(text.contains(key), "{key} missing: {text}");
        }
    }

    /// A read that fails part-way is counted with the bytes it asked for and
    /// the bytes it moved, apart: the second page of the range is reserved
    /// and not committed, so at most the first page can move.
    #[cfg(windows)]
    #[test]
    fn a_failed_read_says_bytes_asked_and_moved_apart() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        // SAFETY: a fresh reservation of two pages, the first committed.
        let base = unsafe {
            let base = VirtualAlloc(std::ptr::null(), 8192, MEM_RESERVE, PAGE_READWRITE);
            assert!(!base.is_null());
            assert_eq!(VirtualAlloc(base, 4096, MEM_COMMIT, PAGE_READWRITE), base);
            base.cast::<u8>()
        };
        let (asked, moved) = (here("read_failed_asked", "read_into"), here("read_failed_moved", "read_into"));
        let mut into = vec![0u8; 8192];
        assert!(!crate::host::read_into(base, &mut into), "the second page cannot be read");
        assert_eq!(here("read_failed_asked", "read_into") - asked, 8192);
        assert!(here("read_failed_moved", "read_into") - moved <= 4096);
        let text = text(1);
        assert!(text.contains("\"read_fields\":[\"ok_calls\",\"ok_bytes\",\"failed_calls\",\"failed_asked\",\"failed_moved\"]"), "{text}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) }, 0);
    }

    /// The Windows calls that read, write, protect, ask about or watch memory.
    const MEMORY_CALLS: [&str; 16] = [
        "ReadProcessMemory",
        "WriteProcessMemory",
        "VirtualProtect",
        "VirtualProtectEx",
        "VirtualQuery",
        "VirtualQueryEx",
        "QueryWorkingSetEx",
        "K32QueryWorkingSetEx",
        "GetWriteWatch",
        "ResetWriteWatch",
        "NtReadVirtualMemory",
        "NtWriteVirtualMemory",
        "NtProtectVirtualMemory",
        "NtQueryVirtualMemory",
        "NtGetWriteWatch",
        "NtResetWriteWatch",
    ];

    /// The user copy's leaf and its module (`safecopy.rs`), whose checked copy reads and stores
    /// title memory with no kernel call: made outside that file by this module's wrappers
    /// alone. The module's name itself is refused, so no import (braced, glob, aliased) can
    /// reach the copy, but where [`SAFECOPY_NAMED`] says.
    const COPY_CALLS: [&str; 2] = ["victory_user_copy", "safecopy"];
    /// The files that may name `safecopy`: its module line, the handler's `redirect`, itself.
    const SAFECOPY_NAMED: [&str; 3] = ["victory-gpu/src/lib.rs", "victory-gpu/src/watch.rs", "victory-gpu/src/safecopy.rs"];

    /// Files that may name the calls outside this module, and why. Each is
    /// checked: the reason is a fact the test asserts.
    const ALLOWED: [(&str, &str); 1] = [(
        "victory-gpu/src/crash_report.rs",
        "compiled only for tests (lib.rs declares it under cfg(all(windows, test))): its own fixture's pages",
    )];

    /// `text` with comments, strings and character literals blanked (every
    /// line kept, so positions still name lines).
    fn code_only(text: &str) -> Vec<char> {
        let c: Vec<char> = text.chars().collect();
        let mut out: Vec<char> = Vec::with_capacity(c.len());
        let blank = |out: &mut Vec<char>, ch: char| out.push(if ch == '\n' { '\n' } else { ' ' });
        let ident = |ch: char| ch.is_alphanumeric() || ch == '_';
        let mut i = 0;
        while i < c.len() {
            let next = c.get(i + 1).copied();
            if c[i] == '/' && next == Some('/') {
                while i < c.len() && c[i] != '\n' {
                    out.push(' ');
                    i += 1;
                }
                continue;
            }
            if c[i] == '/' && next == Some('*') {
                let mut depth = 0;
                while i < c.len() {
                    if c[i] == '/' && c.get(i + 1) == Some(&'*') {
                        depth += 1;
                        out.extend([' ', ' ']);
                        i += 2;
                    } else if c[i] == '*' && c.get(i + 1) == Some(&'/') {
                        depth -= 1;
                        out.extend([' ', ' ']);
                        i += 2;
                        if depth == 0 {
                            break;
                        }
                    } else {
                        blank(&mut out, c[i]);
                        i += 1;
                    }
                }
                continue;
            }
            // Raw strings: `r"…"`, `r#"…"#`, and their byte and C forms.
            if (i == 0 || !ident(c[i - 1])) && (c[i] == 'r' || (matches!(c[i], 'b' | 'c') && next == Some('r'))) {
                let mut j = if c[i] == 'r' { i + 1 } else { i + 2 };
                let mut hashes = 0;
                while c.get(j) == Some(&'#') {
                    hashes += 1;
                    j += 1;
                }
                if c.get(j) == Some(&'"') {
                    let mut end = j + 1;
                    while end < c.len() && !(c[end] == '"' && (1..=hashes).all(|k| c.get(end + k) == Some(&'#'))) {
                        end += 1;
                    }
                    for &ch in &c[i..(end + 1 + hashes).min(c.len())] {
                        blank(&mut out, ch);
                    }
                    i = end + 1 + hashes;
                    continue;
                }
            }
            if c[i] == '"' {
                out.push(' ');
                i += 1;
                while i < c.len() && c[i] != '"' {
                    if c[i] == '\\' && i + 1 < c.len() {
                        blank(&mut out, c[i]);
                        i += 1;
                    }
                    blank(&mut out, c[i]);
                    i += 1;
                }
                out.push(' ');
                i += 1;
                continue;
            }
            if c[i] == '\'' {
                // A character literal (`'x'`, `'\''`, `'\u{..}'`), not a lifetime.
                let end = if next == Some('\\') {
                    (i + 3..c.len()).find(|&j| c[j] == '\'')
                } else if c.get(i + 2) == Some(&'\'') {
                    Some(i + 2)
                } else {
                    None
                };
                if let Some(end) = end {
                    for &ch in &c[i..=end] {
                        blank(&mut out, ch);
                    }
                    i = end + 1;
                    continue;
                }
            }
            out.push(c[i]);
            i += 1;
        }
        out
    }

    /// Whether an attribute's text (`cfg(…)`, no brackets) compiles its item
    /// for tests alone: `cfg(test)`, or `cfg(all(…))` with `test` among its
    /// arguments.
    fn test_only(attribute: &str) -> bool {
        let a: String = attribute.chars().filter(|c| !c.is_whitespace()).collect();
        a == "cfg(test)"
            || a.strip_prefix("cfg(all(").and_then(|rest| rest.strip_suffix("))")).is_some_and(|args| args.split(',').any(|arg| arg == "test"))
    }

    /// `code` with every item compiled only for tests blanked: the attribute
    /// and the item after it, to its `;` or its closing brace.
    fn without_tests(code: &mut [char]) {
        let mut i = 0;
        while i + 2 < code.len() {
            if !(code[i] == '#' && code[i + 1] == '[') {
                i += 1;
                continue;
            }
            let mut depth = 0;
            let mut end = i + 1;
            while end < code.len() {
                match code[end] {
                    '[' => depth += 1,
                    ']' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                end += 1;
            }
            let attribute: String = code[i + 2..end.min(code.len())].iter().collect();
            if !test_only(&attribute) {
                i = end;
                continue;
            }
            let mut j = end + 1;
            let mut depth = 0i32;
            while j < code.len() {
                match code[j] {
                    '(' | '[' | '{' => depth += 1,
                    ')' | ']' => depth -= 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    ';' if depth == 0 => break,
                    _ => {}
                }
                j += 1;
            }
            let last = j.min(code.len() - 1);
            for ch in &mut code[i..=last] {
                if *ch != '\n' {
                    *ch = ' ';
                }
            }
            i = j + 1;
        }
    }

    /// The `.rs` files under `dir`, recursively, as (path from the crates
    /// folder, text).
    fn sources(crates: &std::path::Path, dir: &str) -> Vec<(String, String)> {
        let mut out = Vec::new();
        let mut pending = vec![crates.join(dir)];
        while let Some(at) = pending.pop() {
            for entry in std::fs::read_dir(&at).unwrap_or_else(|e| panic!("{}: {e}", at.display())) {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    pending.push(path);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    let name = path.strip_prefix(crates).unwrap().to_string_lossy().replace('\\', "/");
                    out.push((name, std::fs::read_to_string(&path).unwrap()));
                }
            }
        }
        out.sort();
        out
    }

    /// Every place in `code` (comments, strings and tests blanked) that
    /// names one of the calls, as (line, name), but a definition (`fn
    /// VirtualProtect`) or an export (`hle_export!(VirtualProtect)`): the
    /// title's imports `victory-hle` serves itself, which are not the
    /// Windows calls.
    fn named_calls(code: &[char]) -> Vec<(usize, &'static str)> {
        let ident = |ch: char| ch.is_alphanumeric() || ch == '_';
        let text: String = code.iter().collect();
        let mut out = Vec::new();
        for name in MEMORY_CALLS.into_iter().chain(COPY_CALLS) {
            for (at, _) in text.match_indices(name) {
                let before = text[..at].chars().next_back();
                let after = text[at + name.len()..].chars().next();
                if before.is_some_and(ident) || after.is_some_and(ident) {
                    continue;
                }
                let lead = text[..at].trim_end();
                if lead.ends_with(" fn") || lead.ends_with("\nfn") || lead.trim_end_matches('(').trim_end().ends_with("hle_export!") && lead.ends_with('(') {
                    continue;
                }
                out.push((text[..at].matches('\n').count() + 1, name));
            }
        }
        out.sort();
        out
    }

    /// No call that reads, writes, protects, asks about or watches memory is
    /// made in `victory-gpu` or `victory-hle` outside this module's counted
    /// wrappers: a source scan, so a call added anywhere else fails here
    /// before it can go uncounted. Tests' own calls are left out (items
    /// under `cfg(test)`), and so is each file of [`ALLOWED`], whose reason
    /// is checked.
    #[test]
    fn no_memory_call_bypasses_the_counters() {
        let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
        let lib = std::fs::read_to_string(crates.join("victory-gpu/src/lib.rs")).unwrap();
        assert!(lib.contains("#[cfg(all(windows, test))]\nmod crash_report;"), "the allow-list's reason for crash_report.rs");
        let mut found = Vec::new();
        for (name, text) in sources(&crates, "victory-gpu/src").into_iter().chain(sources(&crates, "victory-hle/src")) {
            if name == "victory-gpu/src/kernelstat.rs" || ALLOWED.iter().any(|(file, _)| *file == name) {
                continue;
            }
            let mut code = code_only(&text);
            without_tests(&mut code);
            for (line, call) in named_calls(&code) {
                if call == "safecopy" && SAFECOPY_NAMED.contains(&name.as_str())
                    || call == "victory_user_copy" && name == "victory-gpu/src/safecopy.rs"
                {
                    continue;
                }
                found.push(format!("{name}:{line} {call}"));
            }
        }
        assert!(found.is_empty(), "memory calls outside the counted wrappers (kernelstat.rs):\n{}", found.join("\n"));
    }

    /// The scan's own parts: comments, strings and tests are left out, and a
    /// call, an import or a call through a path is found.
    #[test]
    fn the_scan_finds_calls_and_leaves_out_comments_strings_and_tests() {
        let text = "use x::{VirtualQuery};\n// VirtualProtect in a comment\nlet a = \"GetWriteWatch\";\n\
                    let b = r#\"ResetWriteWatch\"#; let q = '\"';\nmem::VirtualProtect(p);\n\
                    #[cfg(test)]\nmod tests { fn f() { VirtualQuery(p); } }\n\
                    #[cfg(all(windows, test))]\nfn g() { K32QueryWorkingSetEx(p) }\n\
                    pub extern \"system\" fn VirtualProtect(a: u8) {}\nhle_export!(VirtualProtect),\nlet n = NtReadVirtualMemory2(p);\n\
                    #[cfg(not(test))]\nfn h() { ReadProcessMemory(p); }\n\
                    crate::safecopy::copy(d, s, n, side); victory_user_copy(d, s, n, f); victory_user_copy_at();\n\
                    use crate::safecopy::{copy, Side}; use crate::safecopy::*; use crate::safecopy as s; s::copy(d);\n";
        let mut code = code_only(text);
        without_tests(&mut code);
        assert_eq!(
            named_calls(&code),
            vec![
                (1, "VirtualQuery"),
                (5, "VirtualProtect"),
                (14, "ReadProcessMemory"),
                (15, "safecopy"),
                (15, "victory_user_copy"),
                (16, "safecopy"),
                (16, "safecopy"),
                (16, "safecopy")
            ]
        );
    }
}
