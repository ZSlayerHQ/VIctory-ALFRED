//! Diagnostics: where the submission copies of title memory go (which
//! ranges, how often, of what memory) and what limits their rate, logged
//! with the frame-rate line. `VICTORY_COPY_MODE=plain` copies without the
//! mapped-page guard, `none` skips the copies (both for timing only).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Counters since the last line.
pub(crate) enum C {
    Submissions,
    Copies,
    CopyBytes,
    Refreshes,
    RefreshBytes,
    Skipped,
    SkippedBytes,
    Runs,
    RunUs,
    Queries,
    QueryUs,
    MoveUs,
    Unmapped,
    /// Write-watch askings and their time.
    Watches,
    WatchUs,
    /// Write-watch askings the kernel refused (the range is not
    /// write-watched: view memory, say), and their time.
    AskFailed,
    AskFailedUs,
    /// Bytes of watched ranges checked, and of those found unchanged.
    Checked,
    Unchanged,
    /// Copies that met an unmapped page.
    Fallbacks,
    /// Writes taken by the guard on view memory.
    GuardFaults,
    /// Texture bytes detiled after a watched write; hashed; detiled after a
    /// hash change.
    TexWatched,
    TexHashed,
    TexChanged,
    /// Frames presented since the last line (what the line divides by).
    Frames,
    /// Calls of list methods that record into a host list, and the
    /// nanoseconds they took, summed over the title's recording threads.
    RecordCalls,
    RecordNs,
    /// Waiting for host pipelines being made (`pipeline::make`).
    PipelineWaitUs,
    /// Dynamic descriptor slots: tables asked for, refused (no free run),
    /// and the nanoseconds of the searches.
    DynamicAsked,
    DynamicRefused,
    DynamicNs,
    /// Marker texts for device-removed data (`HostList::note`).
    NoteNs,
    /// Ranges given to shaders (steps), their snapshots, and the bytes
    /// compared after the submission.
    GpuWrites,
    SnapshotUs,
    BackCompared,
    /// Reading around unmapped pages after a failed bulk read.
    SparseUs,
    /// Guard operations whose time is in [`GuardUs`]: repeated asks on
    /// view memory (the ask would fail, the guards answer) and the blocks a
    /// write to guarded memory opened (the fault handler).
    Guards,
    /// Guarding view memory: every guard path times itself here, the
    /// fault handler's included.
    GuardUs,
    /// Bytes copied whole because their memory is neither write-watched
    /// nor a view (`Shadow::update_free`).
    WholeBytes,
    /// Submissions with ranges shaders write (they need the GPU's results
    /// before they return).
    BackSubmissions,
    /// The title's frame waits (`frame::wait`) and the time slept in them,
    /// over all its threads.
    PaceWaits,
    PaceUs,
    /// After a present: until the title's first list call and until its
    /// first submission; and from its last submission to its next present.
    UntilRecordUs,
    UntilSubmitUs,
    TailUs,
    /// Resetting host lists (`HostList::reset`).
    ResetNs,
    /// What shaders wrote: finding the 64 KiB pieces that differ, how many
    /// do, and putting them into title memory.
    BackScanUs,
    BackDiffer,
    BackSyncUs,
    /// Texture uploads recorded, and left out as unchanged.
    TexUploads,
    TexSkipped,
    /// Held submissions a present ran although their waits were not met.
    Forced,
    /// Ranges named through unordered access views by shaders that cannot
    /// store to buffers, not given to shaders as written, and their bytes.
    GpuWritesLeftOut,
    GpuWritesLeftOutBytes,
    /// Waits for memory values looked at (`list::ValueWait::met`) and the
    /// nanoseconds the looks took, on every thread.
    WaitChecks,
    WaitCheckNs,
    /// Host lists made (`Host::new_list`: for the lists a queue holds and
    /// for the parts of lists cut at waits) and the nanoseconds that took.
    HostLists,
    HostListNs,
    /// Parts of held submissions that ran while the part behind them could
    /// have run with them (its waits were met already).
    PartsTogether,
    /// Microseconds of the copies counted in `WholeBytes`.
    WholeUs,
    /// Submissions by how long the card was waited for (under 2 ms, 2-8,
    /// 8-32, longer), and the microseconds waited in each class.
    CardUnder2,
    Card2To8,
    Card8To32,
    CardOver32,
    CardUnder2Us,
    Card2To8Us,
    Card8To32Us,
    CardOver32Us,
    /// Submissions that hand nothing back to the title (no range shaders
    /// write, no value written when done), and the microseconds the card
    /// was waited for them.
    NothingBack,
    NothingBackUs,
    /// Waits for every earlier shader write recorded into host lists: for
    /// the title's barrier groups, for its barriers on buffers, and the
    /// indirect commands bounded (each a dispatch of ours between two such
    /// waits).
    WaitsForGroups,
    WaitsForBuffers,
    IndirectBounded,
    /// Bytes table constant buffers asked their scratch fill for, bytes
    /// filled, and the fills' microseconds: reported apart from other
    /// copies (at most 4,096 vectors each; the native-memory plan, step 2).
    /// With `VICTORY_COPY_MODE=none` nothing is filled, so nothing is
    /// counted as filled then; the request stands, whatever the mode.
    TableFillRequested,
    TableFillFilled,
    TableFillUs,
    /// Upload fills run on the pool at submission (`VICTORY_UPLOAD_POOL=1`)
    /// and the bytes they moved, counted apart from the serial fills so an
    /// on/off pair reads the split directly.
    UploadPoolBytes,
    /// The join's time: how long the submitting thread waited in
    /// [`crate::host`]'s `run_fills` for every fill of a step to finish
    /// before the next step, the send to the card, or a release of title
    /// memory could go on.
    UploadPoolJoinUs,
    /// The descriptor mirror dispatcher's work for records the title copies
    /// or creates through the API: nanoseconds and records, one batch per
    /// call (`copy_simple` and `record` in `descriptor.rs`; Sol's note on
    /// the deferred-records fork).
    DescriptorCopyNs,
    DescriptorCopyRecords,
    /// Host copies of textures the title fills kept past their idle time
    /// because the card was under the low watermark, and let go by the
    /// pressure sweep (`resource::let_go_idle`).
    IdleKept,
    PressureGone,
}
const COUNTERS: usize = 90;
static COUNT: [AtomicU64; COUNTERS] = [const { AtomicU64::new(0) }; COUNTERS];

// Tests: the counters a fixture reads, per thread. The suite runs tests in
// parallel and every submission of every test would land in a process-wide
// counter, so a fixture's delta would include other tests' copies; a test
// reads what its own thread counted (its submissions run on its own
// thread). The line's counters are not exercised in tests.
#[cfg(test)]
thread_local! {
    static TEST_COUNT: [std::cell::Cell<u64>; COUNTERS] =
        const { [const { std::cell::Cell::new(0) }; COUNTERS] };
}

pub(crate) fn add(c: C, n: u64) {
    let at = c as usize;
    #[cfg(not(test))]
    COUNT[at].fetch_add(n, Relaxed);
    #[cfg(test)]
    TEST_COUNT.with(|counts| counts[at].set(counts[at].get().wrapping_add(n)));
}

/// The counter's value so far, for fixtures (the line takes and zeroes them).
#[cfg(test)]
pub(crate) fn count(c: C) -> u64 {
    TEST_COUNT.with(|counts| counts[c as usize].get())
}

/// A counter since the last line, read and not taken (`crate::stalls`).
pub(crate) fn peek(c: C) -> u64 {
    COUNT[c as usize].load(Relaxed)
}

/// What the stall line reads of these counters ([`crate::stalls::line`]),
/// and the process's page faults so far.
pub(crate) fn stall_figures() -> (u64, crate::stalls::Others) {
    let others = crate::stalls::Others {
        checked: peek(C::Checked),
        copied: peek(C::CopyBytes),
        copies: peek(C::Copies),
        pipeline_wait_us: peek(C::PipelineWaitUs),
        pace_us: peek(C::PaceUs),
        faults: page_faults(),
    };
    (peek(C::Frames), others)
}

/// How copies run: 0 guarded (as always), 1 plain, 2 not at all.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_COPY_MODE").as_deref() {
        Ok("plain") => 1,
        Ok("none") => 2,
        _ => 0,
    })
}

/// Per source range (start, length): copies, bytes, microseconds, memory type.
type Ranges = HashMap<(usize, usize), (u64, u64, u64, u32)>;
static RANGES: Mutex<Option<Ranges>> = Mutex::new(None);

/// Per range copied whole because its memory is not watched (start,
/// length): copies and microseconds.
static WHOLE: Mutex<Option<HashMap<(usize, usize), (u64, u64)>>> = Mutex::new(None);

/// `len` bytes at `src`, memory that is not watched, were copied whole in
/// `us`.
pub(crate) fn whole(src: usize, len: usize, us: u64) {
    add(C::WholeUs, us);
    let mut map = WHOLE.lock().unwrap_or_else(|e| e.into_inner());
    let e = map.get_or_insert_with(HashMap::new).entry((src, len)).or_insert((0, 0));
    e.0 += 1;
    e.1 += us;
}

/// The longest wait for the card since the last line: microseconds, and
/// what ran (lists and pipelines).
static LONGEST: Mutex<(u64, String)> = Mutex::new((0, String::new()));

/// The waits for the card since the last line by what ran (lists and
/// pipelines): how many, and their microseconds.
static WAITS_BY_WORK: Mutex<Option<HashMap<String, (u64, u64)>>> = Mutex::new(None);

/// With `VICTORY_GPU_TIMING`: the commands' time on the card since the last
/// line by what they ran (a pipeline, or the kind of command): how many,
/// their microseconds, and the longest one's.
static COMMANDS_BY_WORK: Mutex<Option<HashMap<String, (u64, u64, u64)>>> = Mutex::new(None);

/// A command took `ms` on the card (timing run). `op` is its note: the
/// kind and size first, then the pipeline in brackets, then what it binds.
pub(crate) fn command_took(op: &str, ms: f64) {
    let us = (ms * 1e3) as u64;
    let mut by = COMMANDS_BY_WORK.lock().unwrap_or_else(|e| e.into_inner());
    let e = by.get_or_insert_with(HashMap::new).entry(command_key(op)).or_insert((0, 0, 0));
    e.0 += 1;
    e.1 += us;
    e.2 = e.2.max(us);
}

/// What a command's note is summed under: its pipeline (`[compute ...]`,
/// `[mesh ..., pixel ...]`) with the command's first word, or without a
/// pipeline its first two words.
fn command_key(op: &str) -> String {
    let head = op.split(" | ").next().unwrap_or(op);
    let word = head.split(' ').next().unwrap_or("");
    match (head.find('['), head.find(']')) {
        (Some(a), Some(b)) if a < b => format!("{word} {}", &head[a..=b]),
        _ => head.split(' ').take(2).collect::<Vec<_>>().join(" "),
    }
}

/// A host wait of `us`: a synchronous submission from its send to its
/// retirement, or a settle's wait for flights' fence (not time on the card:
/// earlier work queued and the objects let go are in it); `what` names what
/// was waited for, and the longest since the last line.
pub(crate) fn host_waited(us: u64, what: impl FnOnce() -> String) {
    let (n, t) = match us {
        0..2_000 => (C::CardUnder2, C::CardUnder2Us),
        2_000..8_000 => (C::Card2To8, C::Card2To8Us),
        8_000..32_000 => (C::Card8To32, C::Card8To32Us),
        _ => (C::CardOver32, C::CardOver32Us),
    };
    add(n, 1);
    add(t, us);
    // Every submission is named (some forty a frame): the same lists run
    // every frame, and the eight longest waits of a line were one list
    // eight times.
    let text = what();
    {
        let mut by = WAITS_BY_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let e = by.get_or_insert_with(HashMap::new).entry(text.clone()).or_insert((0, 0));
        e.0 += 1;
        e.1 += us;
    }
    let mut longest = LONGEST.lock().unwrap_or_else(|e| e.into_inner());
    if us > longest.0 {
        *longest = (us, text);
    }
}

/// Whether an experiment on barriers is on (`VICTORY_BARRIER_TEST`): bit 0
/// for `groups` (group barriers record no wait for earlier shader writes),
/// bit 1 for `indirect` (the path that bounds an indirect command runs
/// without both such waits; its state changes stay), both bits for `both`. For timing only: without the waits a shader may read
/// what another has not finished writing.
pub(crate) fn barrier_test() -> u8 {
    static TEST: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *TEST.get_or_init(|| match std::env::var("VICTORY_BARRIER_TEST").as_deref() {
        Ok("groups") => 1,
        Ok("indirect") => 2,
        Ok("both") => 3,
        _ => 0,
    })
}

/// Per range compared after a submission (start, length): times, and the
/// bytes of it written back to the title.
static COMPARED: Mutex<Option<HashMap<(usize, usize), (u64, u64)>>> = Mutex::new(None);

/// `len` bytes at `src` were compared after a submission and `written` of
/// them put back into title memory.
pub(crate) fn compared(src: usize, len: usize, written: usize) {
    let mut map = COMPARED.lock().unwrap_or_else(|e| e.into_inner());
    let e = map.get_or_insert_with(HashMap::new).entry((src, len)).or_insert((0, 0));
    e.0 += 1;
    e.1 += written as u64;
}

/// A copy of `len` bytes from `src` (memory type `kind`) took `us`.
pub(crate) fn range(src: usize, len: usize, us: u64, kind: u32) {
    let mut map = RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let e = map
        .get_or_insert_with(HashMap::new)
        .entry((src, len))
        .or_insert((0, 0, 0, kind));
    e.0 += 1;
    e.1 += len as u64;
    e.2 += us;
}

/// Our own allocations alive now, by size of block (under 4 KiB, to 64 KiB,
/// to 1 MiB, larger): bytes, then blocks. "Our own" is the Rust code of
/// every crate in the process: in diagnostics builds [`Counting`] stands in
/// front of the system's heap. The memory map's "process heap segments"
/// hold the blocks under about half a megabyte of ours, of the graphics
/// runtime and of the driver alike (gigabytes in gameplay, growing every
/// frame, and nothing said whose): this says which part is
/// ours. Bigger blocks are allocations of their own in the map.
static OWN: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// The class of [`OWN`] a block of `size` bytes counts in.
fn block_class(size: usize) -> usize {
    match size {
        0..0x1000 => 0,
        0x1000..0x1_0000 => 1,
        0x1_0000..0x10_0000 => 2,
        _ => 3,
    }
}

fn own(size: usize, made: bool) {
    let class = block_class(size);
    if made {
        OWN[class].fetch_add(size as u64, Relaxed);
        OWN[4 + class].fetch_add(1, Relaxed);
    } else {
        OWN[class].fetch_sub(size as u64, Relaxed);
        OWN[4 + class].fetch_sub(1, Relaxed);
    }
}

/// Our own blocks of [`BIG`] bytes and more alive now: where each is (0:
/// none) and its bytes. The memory map lists the process's big allocations
/// by size without saying whose they are (several of hundreds of
/// megabytes in a run); the sizes here say which of them
/// are ours.
static BIG_AT: [std::sync::atomic::AtomicUsize; 64] = [const { std::sync::atomic::AtomicUsize::new(0) }; 64];
static BIG_BYTES: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];
const BIG: usize = 32 << 20;

/// Big blocks made while all 64 of [`BIG_AT`] were listed: they are not in
/// the list, so its sizes and count come up short without this (finding
/// 23). Cumulative since start.
static BIG_MISSING: AtomicU64 = AtomicU64::new(0);

/// Takes the first free slot of `slots` for `at` and stores `size`; false
/// when every slot is taken (the caller counts the block as not listed).
fn big_claim(
    slots: &[std::sync::atomic::AtomicUsize],
    sizes: &[AtomicU64],
    at: usize,
    size: usize,
) -> bool {
    for (place, bytes) in slots.iter().zip(sizes) {
        if place.compare_exchange(0, at, Relaxed, Relaxed).is_ok() {
            bytes.store(size as u64, Relaxed);
            return true;
        }
    }
    false
}

/// A block of [`BIG`] bytes or more was made at `at`, or released there.
fn big(at: *mut u8, size: usize, made: bool) {
    if size < BIG {
        return;
    }
    if made {
        if !big_claim(&BIG_AT, &BIG_BYTES, at as usize, size) {
            BIG_MISSING.fetch_add(1, Relaxed);
        }
        return;
    }
    for (place, bytes) in BIG_AT.iter().zip(&BIG_BYTES) {
        if place.load(Relaxed) == at as usize {
            bytes.store(0, Relaxed);
            place.store(0, Relaxed);
            return;
        }
    }
}

/// The sizes of our own big blocks alive, largest first, in MB, and how
/// many blocks made since the start found no slot to be listed in.
fn big_blocks() -> String {
    let mut sizes: Vec<u64> = BIG_BYTES.iter().map(|b| b.load(Relaxed)).filter(|&b| b != 0).collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let text: Vec<String> = sizes.iter().map(|b| format!("{:.2}", *b as f64 / (1 << 20) as f64)).collect();
    let list = if text.is_empty() { "none".to_owned() } else { text.join(", ") };
    let missing = BIG_MISSING.load(Relaxed);
    if missing == 0 {
        list
    } else {
        // Not "alive now": made since the start while the list was full.
        format!("{list} ({missing} more made since the start with all 64 listed, not listed)")
    }
}

/// The system's allocator with every block counted in [`OWN`] (two atomic
/// additions per allocation and per release: diagnostics builds only).
struct Counting;

// SAFETY: every call goes to the system's allocator unchanged; only
// counters are kept beside it.
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        // SAFETY: the caller's contract, passed on.
        let at = unsafe { std::alloc::System.alloc(layout) };
        if !at.is_null() {
            own(layout.size(), true);
            big(at, layout.size(), true);
            #[cfg(test)]
            let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        }
        at
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        // SAFETY: the caller's contract, passed on.
        let at = unsafe { std::alloc::System.alloc_zeroed(layout) };
        if !at.is_null() {
            own(layout.size(), true);
            big(at, layout.size(), true);
            #[cfg(test)]
            let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        }
        at
    }

    unsafe fn dealloc(&self, at: *mut u8, layout: std::alloc::Layout) {
        own(layout.size(), false);
        big(at, layout.size(), false);
        // SAFETY: the caller's contract, passed on.
        unsafe { std::alloc::System.dealloc(at, layout) };
    }

    unsafe fn realloc(&self, at: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        // SAFETY: the caller's contract, passed on.
        let moved = unsafe { std::alloc::System.realloc(at, layout, size) };
        if !moved.is_null() {
            own(layout.size(), false);
            own(size, true);
            big(at, layout.size(), false);
            big(moved, size, true);
        }
        moved
    }
}

#[global_allocator]
static HEAP: Counting = Counting;

#[cfg(test)]
thread_local! {
    /// Tests: blocks allocated on this thread (reallocations not counted).
    static ALLOCATIONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Tests: blocks allocated on this thread so far.
#[cfg(test)]
pub(crate) fn allocated_here() -> u64 {
    ALLOCATIONS.with(std::cell::Cell::get)
}

/// Tests: how many blocks `f` allocated on this thread.
#[cfg(test)]
pub(crate) fn allocations_in(f: impl FnOnce()) -> u64 {
    let before = ALLOCATIONS.with(std::cell::Cell::get);
    f();
    ALLOCATIONS.with(std::cell::Cell::get) - before
}

#[cfg(test)]
mod command_tests {
    /// Commands are summed by pipeline and first word; those without a
    /// pipeline by their first two words. The lines are written for this
    /// test in the shape the host's notes have (`Draw::op_text`, the
    /// texture notes); every id, size, index and address in them is made up.
    #[test]
    fn commands_are_named_by_their_pipeline() {
        use super::command_key;
        assert_eq!(
            command_key("dispatch 4x3x2 [compute 1111222233334444] | 1111222233334444: u2=0x40 [heap 7: x]"),
            "dispatch [compute 1111222233334444]"
        );
        assert_eq!(
            command_key("indirect kind 7 max 5 at 0x1000 [mesh 5555666677778888, pixel 9999aaaabbbbcccc] | t"),
            "indirect [mesh 5555666677778888, pixel 9999aaaabbbbcccc]"
        );
        assert_eq!(command_key("texture upload 0x2000 (64x32x1 f28 mips 3) sub 1"), "texture upload");
        assert_eq!(command_key("alias copy of 0x3000 into 0x4000"), "alias copy");
    }
}

#[cfg(test)]
mod own_tests {
    use super::{OWN, block_class};
    use std::sync::atomic::Ordering::Relaxed;

    #[test]
    fn blocks_count_in_their_size_class() {
        assert_eq!(block_class(0), 0);
        assert_eq!(block_class(4095), 0);
        assert_eq!(block_class(4096), 1);
        assert_eq!(block_class(65535), 1);
        assert_eq!(block_class(65536), 2);
        assert_eq!(block_class((1 << 20) - 1), 2);
        assert_eq!(block_class(1 << 20), 3);
    }

    /// A block of 32 MiB or more is listed by its size while it lives.
    #[test]
    fn big_blocks_are_listed_while_they_live() {
        let size = (48usize << 20) + 4096;
        let listed = |at: usize| {
            super::BIG_AT
                .iter()
                .zip(&super::BIG_BYTES)
                .any(|(place, bytes)| place.load(Relaxed) == at && bytes.load(Relaxed) == size as u64)
        };
        let block = vec![0u8; size];
        let at = block.as_ptr() as usize;
        assert!(listed(at));
        assert!(super::big_blocks().contains("48.00"));
        drop(block);
        assert!(!listed(at));
    }

    /// A block made while every slot is taken is counted, not dropped: the
    /// claim fails, and the list says how many did (finding 23).
    #[test]
    fn blocks_past_the_list_are_counted() {
        let slots = [const { std::sync::atomic::AtomicUsize::new(0) }; 2];
        let sizes = [const { std::sync::atomic::AtomicU64::new(0) }; 2];
        assert!(super::big_claim(&slots, &sizes, 0x1000, 48 << 20));
        assert!(super::big_claim(&slots, &sizes, 0x2000, 48 << 20));
        assert!(!super::big_claim(&slots, &sizes, 0x3000, 48 << 20), "both slots taken");
        super::BIG_MISSING.fetch_add(2, Relaxed);
        let text = super::big_blocks();
        super::BIG_MISSING.fetch_sub(2, Relaxed);
        assert!(text.contains("2 more"), "{text}");
    }

    /// A big block shows in the counters while it lives (other tests
    /// allocate small blocks meanwhile: only the big class is looked at,
    /// and only for at least this block).
    #[test]
    fn a_block_is_counted_while_it_lives() {
        let block = vec![1u8; 8 << 20];
        assert!(OWN[3].load(Relaxed) >= 8 << 20);
        assert!(OWN[7].load(Relaxed) >= 1);
        drop(block);
        assert!(OWN[7].load(Relaxed) < u64::MAX / 2, "no count below zero");
    }
}

/// The process's memory in bytes: in physical memory, and committed
/// (private bytes).
fn process_memory() -> (u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    (c.WorkingSetSize as u64, c.PagefileUsage as u64)
}

/// The PC's memory in bytes: physical memory free, and what is left to
/// commit.
fn system_memory() -> (u64, u64) {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    unsafe { GlobalMemoryStatusEx(&mut m) };
    (m.ullAvailPhys, m.ullAvailPageFile)
}

/// Whether the address space is walked for the memory map line
/// ([`log_memory_map`]). On by default (this module is in diagnostics
/// builds only); `VICTORY_MEMORY_MAP=0` leaves the walk out.
pub(crate) fn memory_map() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MEMORY_MAP").is_none_or(|v| v != "0"))
}

/// Asks which of the pages in `asked` (address, answer) are in physical
/// memory and adds the bytes each stands for (kind, bytes) to its kind.
fn in_memory(asked: &mut Vec<[usize; 2]>, stands: &mut Vec<(usize, u64)>, present: &mut [u64; 10]) {
    // SAFETY: our own process; `asked` holds that many entries of two
    // words each, as the call fills them in.
    let ok = unsafe {
        crate::kernelstat::working_set(
            crate::kernelstat::Place::CopyStats,
            asked.as_mut_ptr().cast(),
            (asked.len() * 16) as u32,
            asked.len(),
        )
    };
    if ok {
        for (a, &(kind, bytes)) in asked.iter().zip(stands.iter()) {
            // Bit 0 of the answer: the page is in the process's memory.
            if a[1] & 1 != 0 {
                present[kind] += bytes;
            }
        }
    }
    asked.clear();
    stands.clear();
}

/// Logs what the process has committed by kind, from a walk of its address
/// space, and how much of each kind is in physical memory (one page in
/// sixteen is asked about). The kinds tell the title's own memory from
/// ours: the title's heaps and page views lie at console addresses (below
/// 1 TiB), everything the host allocates lies far above. A run seen
/// from outside had its private memory in the title's heaps, in the
/// segments of the process heap, in hundreds of allocations of 4 MiB
/// (windows of title memory) and in the pipeline library's file; mapped
/// and so not private: the title's page pool and the driver's shader
/// cache file.
/// shortcut: the walk takes about 0.6 s on a thread of its own, and a
/// query of a region of gigabytes holds up other threads' memory calls
/// for its length (tens of milliseconds for the title's page pool), so it runs with every
/// sixth memory line only.
fn log_memory_map() {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_FREE, MEM_IMAGE, MEM_MAPPED, MEMORY_BASIC_INFORMATION, PAGE_WRITECOMBINE,
    };
    const KINDS: [&str; 10] = [
        "the title's heaps (private, console addresses)",
        "views of the title's pages (mapped, console addresses)",
        "the title's page pool (mapped, 1 GB and more)",
        "process heap segments (private, 15.8 MB each)",
        "windows of title memory and other 4 MiB allocations",
        "write-combined (buffers shared with the card)",
        "other private allocations of 1 MB and more",
        "other private allocations under 1 MB",
        "files and other mappings at host addresses",
        "programs and libraries",
    ];
    /// Console addresses lie below this.
    const HOST: usize = 1 << 40;
    /// One page asked about stands for this much.
    const STEP: usize = 64 << 10;
    let started = std::time::Instant::now();
    // Allocation, start, bytes, committed, type, protection.
    type Region = (usize, usize, usize, bool, u32, u32);
    let mut regions: Vec<Region> = Vec::new();
    let mut at = 0usize;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if crate::kernelstat::virtual_query(crate::kernelstat::Place::CopyStats, at, &mut m) == 0 {
            break;
        }
        if m.State != MEM_FREE {
            regions.push((
                m.AllocationBase as usize,
                m.BaseAddress as usize,
                m.RegionSize,
                m.State == MEM_COMMIT,
                m.Type,
                m.Protect,
            ));
        }
        let next = (m.BaseAddress as usize).saturating_add(m.RegionSize);
        if next <= at {
            break;
        }
        at = next;
    }
    // Each allocation's whole size, reserved parts included.
    let mut whole: HashMap<usize, usize> = HashMap::new();
    for r in &regions {
        *whole.entry(r.0).or_default() += r.2;
    }
    let kind = |r: &Region| -> usize {
        let size = whole.get(&r.0).copied().unwrap_or(0);
        let console = r.0 < HOST;
        if r.4 == MEM_IMAGE {
            9
        } else if r.4 == MEM_MAPPED {
            match (console, size >= 1 << 30) {
                (false, _) => 8,
                (true, true) => 2,
                (true, false) => 1,
            }
        } else if console {
            0
        } else if size == 0xfd_0000 {
            3
        } else if size == 4 << 20 {
            4
        } else if r.5 & PAGE_WRITECOMBINE != 0 {
            5
        } else if size >= 1 << 20 {
            6
        } else {
            7
        }
    };
    let mut committed = [0u64; 10];
    let mut present = [0u64; 10];
    let mut asked: Vec<[usize; 2]> = Vec::with_capacity(4096);
    let mut stands: Vec<(usize, u64)> = Vec::with_capacity(4096);
    for r in regions.iter().filter(|r| r.3) {
        let k = kind(r);
        committed[k] += r.2 as u64;
        let mut offset = 0;
        while offset < r.2 {
            asked.push([r.1 + offset, 0]);
            stands.push((k, STEP.min(r.2 - offset) as u64));
            if asked.len() == 4096 {
                in_memory(&mut asked, &mut stands, &mut present);
            }
            offset += STEP;
        }
    }
    if !asked.is_empty() {
        in_memory(&mut asked, &mut stands, &mut present);
    }
    let text: Vec<String> = (0..KINDS.len())
        .filter(|&k| committed[k] > 0)
        .map(|k| format!("{} {} MB ({} MB in memory)", KINDS[k], committed[k] >> 20, present[k] >> 20))
        .collect();
    tracing::info!(
        "memory map ({} regions, {} ms): {}",
        regions.len(),
        started.elapsed().as_millis(),
        text.join("; ")
    );
    // The private allocations the kinds above leave as "other" (and the
    // write-combined ones), grouped by their size: the sizes name their
    // owners (the pipeline library's file to the byte, a snapshot buffer
    // of a title heap's size, the driver's blocks). Committed bytes by
    // allocation first, then allocations by (kind, whole size).
    let mut by_allocation: HashMap<usize, (usize, u64)> = HashMap::new();
    for r in regions.iter().filter(|r| r.3) {
        let k = kind(r);
        if k == 5 || k == 6 {
            by_allocation.entry(r.0).or_insert((k, 0)).1 += r.2 as u64;
        }
    }
    let mut groups: HashMap<(usize, usize), (u64, u64)> = HashMap::new();
    for (base, &(k, bytes)) in &by_allocation {
        let group = groups.entry((k, whole.get(base).copied().unwrap_or(0))).or_default();
        group.0 += 1;
        group.1 += bytes;
    }
    let mut groups: Vec<((usize, usize), (u64, u64))> = groups.into_iter().collect();
    groups.sort_by_key(|g| std::cmp::Reverse(g.1.1));
    let biggest: Vec<String> = groups
        .iter()
        .take(20)
        .map(|&((k, size), (n, bytes))| {
            format!(
                "{n} of {:.2} MB{}: {} MB",
                size as f64 / (1 << 20) as f64,
                if k == 5 { " write-combined" } else { "" },
                bytes >> 20
            )
        })
        .collect();
    tracing::info!(
        "memory map, other private allocations by size, the {} biggest groups of {} (committed): {}",
        biggest.len(),
        groups.len(),
        biggest.join("; ")
    );
}

/// Logs where the memory is: the process and the PC, then what this crate
/// holds by kind (alive now), then the host buffers made since the start.
fn log_memory() {
    use std::sync::atomic::Ordering::Relaxed;
    let mb = |v: u64| v as f64 / (1 << 20) as f64;
    let (working, private) = process_memory();
    let (free, left) = system_memory();
    let copies = crate::shadow::memory();
    let staged = crate::texture::memory();
    let textures = [0, 1].map(|i| crate::resource::HOST_TEXTURES[i].load(Relaxed));
    let made: Vec<u64> = crate::host::MADE.iter().map(|m| m.load(Relaxed)).collect();
    let windows = crate::shadow::windows();
    let stagings = crate::texture::let_go();
    let census = crate::resource::census();
    let kinds: Vec<u64> = crate::resource::KINDS.iter().map(|k| k.load(Relaxed)).collect();
    let card = crate::host::host()
        .and_then(|h| h.video_memory())
        .unwrap_or_default();
    // Everything the system says of the two memory groups, beside the
    // budget and use the lines below print.
    if let Some(h) = crate::host::host() {
        h.log_video_memory("with the memory line");
    }
    // What else holds memory: the map, on a thread of its own, with every
    // sixth line; and what the shader table and the pipeline library keep.
    static LINES: AtomicU64 = AtomicU64::new(0);
    static WALKING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if memory_map() && LINES.fetch_add(1, Relaxed) % 6 == 5 && !WALKING.swap(true, Relaxed) {
        let walk = std::thread::Builder::new().name("victory-memory-map".into()).spawn(|| {
            log_memory_map();
            WALKING.store(false, Relaxed);
        });
        if walk.is_err() {
            WALKING.store(false, Relaxed);
        }
    }
    let kept = crate::pipeline::kept();
    tracing::info!(
        "memory kept beside the lines below: {} shader containers {:.0} MB, {} translations \
         {:.0} MB; the pipeline library's file {:.0} MB (its last save wrote {:.0} MB)",
        kept[0],
        mb(kept[1]),
        kept[2],
        mb(kept[3]),
        mb(kept[4]),
        mb(kept[5]),
    );
    tracing::info!(
        "memory of textures: {} title textures have no host copy yet ({:.0} MB of title memory); \
         host copies the host renders to or writes: {} alive {:.0} MB; host copies the title \
         fills: {} alive {:.0} MB ({} of them placed), {} let go {:.0} MB and {} made again \
         {:.0} MB since the start; {} host copies made in all; passes that made none for lack of \
         room: {}; the card: {:.0} MB in use of a budget of {:.0} MB, {:.0} MB of the PC's memory \
         shared with it in use",
        census[2],
        mb(census[3]),
        census[0],
        mb(census[1]),
        census[4],
        mb(census[5]),
        census[6],
        kinds[2],
        mb(kinds[3]),
        kinds[4],
        mb(kinds[5]),
        kinds[6],
        crate::draw::HEAP_TEXTURES_FULL.load(Relaxed),
        mb(card[1]),
        mb(card[0]),
        mb(card[3]),
    );
    let targets = crate::resource::targets_census();
    tracing::info!(
        "host copies the sweep never lets go: {} the host renders to or writes {:.0} MB ({} of them placed {:.0} MB);          of them {} have title memory that is not mapped now {:.0} MB, {} were not used for 30 s and are held by no          list and no slot {:.0} MB, {} are both {:.0} MB; and {} copies of textures the title fills that the host          copied into {:.0} MB",
        targets[0],
        mb(targets[1]),
        targets[10],
        mb(targets[11]),
        targets[2],
        mb(targets[3]),
        targets[4],
        mb(targets[5]),
        targets[6],
        mb(targets[7]),
        targets[8],
        mb(targets[9]),
    );
    let spare = crate::texture::spare();
    tracing::info!(
        "memory let go since the start: {} windows outside buffers {:.0} MB, {} staging buffers \
         {:.0} MB; of the windows alive, {:.0} MB lie between the lowest and highest address \
         asked for in each; staging buffers taken from spare ones instead of made: {} {:.0} MB, \
         spare now {} {:.0} MB",
        windows[0],
        mb(windows[1]),
        stagings[0],
        mb(stagings[1]),
        mb(windows[2]),
        spare[0],
        mb(spare[1]),
        spare[2],
        mb(spare[3]),
    );
    tracing::info!(
        "memory: {:.0} MB private ({:.0} MB of it in memory; the PC has {:.0} MB free and {:.0} MB \
         left to commit); host copies of title memory: {} alive {:.0} MB, of them {} windows \
         outside buffers {:.0} MB; snapshot buffers kept: {} reserving {:.0} MB; staging of \
         textures: {} alive {:.0} MB, {} kept by their texture; host textures: {} alive {:.0} MB, \
         {} placed in title memory; made so far: {} upload buffers {:.0} MB, {} shared buffers \
         {:.0} MB, {} card buffers {:.0} MB",
        mb(private),
        mb(working),
        mb(free),
        mb(left),
        copies[0],
        mb(copies[1]),
        copies[2],
        mb(copies[3]),
        copies[4],
        mb(copies[5]),
        staged[0],
        mb(staged[1]),
        staged[2],
        textures[0],
        mb(textures[1]),
        crate::resource::placed_textures(),
        made[0],
        mb(made[1]),
        made[2],
        mb(made[3]),
        made[4],
        mb(made[5]),
    );
    tracing::info!(
        "memory, placed textures: {} of them were asked for again with other filler words in their description          (one resource and host copy each since the filler left the key; VICTORY_PLACED_FILLER=1 keeps it)",
        crate::resource::placed_filler_twins()
    );
    // What the line above leaves open: what is alive of the buffers "made
    // so far", what the windows hold and what holds them, and what our own
    // code has allocated.
    let alive: Vec<u64> = crate::host::ALIVE.iter().map(|a| a.load(Relaxed)).collect();
    let window = crate::shadow::window_use();
    let views = crate::descriptor::views_kept();
    let own: Vec<u64> = OWN.iter().map(|o| o.load(Relaxed)).collect();
    tracing::info!(
        "memory, what holds it: {} host lists alive; their scratch memory: {} chunks {:.0} MB, of          them {} chunks {:.0} MB spare ({} taken from the spare ones so far); their other upload          buffers: {} alive {:.0} MB; windows outside buffers (grid {} KiB): {:.0} MB of their          pages ever copied, {} shown by descriptor slots {:.0} MB, {} held by lists or the pass          over the heap {:.0} MB, {} asked for without a length {:.0} MB; buffer views of the heap          whose slot showed another copy than the one brought up to date: {}; distinct views          kept: {} ({} free) {:.0} MB; our own allocations: {:.0} MB in {} blocks (under 4 KiB          {:.0} MB in {}, to 64 KiB {:.0} MB in {}, to 1 MiB {:.0} MB in {}, larger {:.0} MB in {})",
        alive[0],
        alive[1],
        mb(alive[2]),
        alive[3],
        mb(alive[4]),
        alive[5],
        alive[6],
        mb(alive[7]),
        window[7] >> 10,
        mb(window[0]),
        window[1],
        mb(window[2]),
        window[3],
        mb(window[4]),
        window[5],
        mb(window[6]),
        crate::draw::HEAP_VIEWS_ELSEWHERE.load(Relaxed),
        views[0],
        views[1],
        mb(views[2]),
        mb(own[..4].iter().sum()),
        own[4..].iter().sum::<u64>(),
        mb(own[0]),
        own[4],
        mb(own[1]),
        own[5],
        mb(own[2]),
        own[6],
        mb(own[3]),
        own[7],
    );
    // Round 2: what the view table still holds and what goes around it,
    // what host lists cost, the card's share that is the layer's own or
    // the title's no more, and which big blocks are ours.
    let lists: Vec<u64> = crate::host::LISTS.iter().map(|l| l.load(Relaxed)).collect();
    tracing::info!(
        "memory, more: views named whole in their records since the start: {} constant buffer \
         views, {} views of placed memory; of the view table's entries {} are constant buffer \
         views and {} views of placed memory; shapes of placed views kept: {}; host lists: {} \
         made by the host, {} preludes made for them, private bytes grew {:.0} MB while \
         lists were made; twins of depth textures: {:.0} MB; host copies of \
         placed textures whose title memory is not mapped any more (estimate): {:.0} MB, of \
         them still shown by a descriptor slot {:.0} MB; the pipeline library's mapped file \
         let go from memory so far: {:.0} MB; our own blocks of 32 MB and more, in MB: {}",
        views[6],
        views[7],
        views[3],
        views[4],
        views[5],
        lists[0],
        lists[1],
        lists[2] as i64 as f64 / (1 << 20) as f64,
        mb(census[7]),
        mb(census[8]),
        mb(census[9]),
        mb(kept[6]),
        big_blocks(),
    );
    // Round 3: whether windows hold the same memory more than once (they
    // can, once they are cut finer than 4 MiB), and how many host lists
    // the title made and never used.
    tracing::info!(
        "memory, windows: the {:.0} MB of windows lie over {:.0} MB of title memory (the rest is \
         memory that is in more than one window, and is copied once for each); host lists never \
         reset since they were made: {}",
        mb(copies[3]),
        mb(crate::shadow::window_cover()),
        lists[3],
    );
}

/// Microseconds since the first call.
fn now_us() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_micros() as u64
}

/// When the last frame was presented and when the last submission returned
/// ([`now_us`]), and whether a list call and a submission followed that
/// present yet.
static PRESENTED_AT: AtomicU64 = AtomicU64::new(0);
static SUBMITTED_AT: AtomicU64 = AtomicU64::new(0);
static RECORDED_SINCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static SUBMITTED_SINCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// By thread: list-call time (nanoseconds, calls) of every recording thread.
/// What a submitting thread does between its submissions and in them is the
/// frame budget's (`crate::budget`): its phases, with the thread's processor
/// time, replaced the laps and thread times that were kept here.
static RECORDERS: Mutex<Vec<(u32, u64, u64)>> = Mutex::new(Vec::new());

thread_local! {
    /// List-call time not yet added to `RECORDERS` (nanoseconds, calls).
    static UNSENT: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}

fn thread_id() -> u32 {
    // SAFETY: a plain query.
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
}

fn micros(t: windows_sys::Win32::Foundation::FILETIME) -> u64 {
    (u64::from(t.dwHighDateTime) << 32 | u64::from(t.dwLowDateTime)) / 10
}

/// The process's processor time so far (user, kernel), in microseconds.
fn process_times() -> (u64, u64) {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: our own process and four valid out pointers.
    unsafe { GetProcessTimes(GetCurrentProcess(), &mut created, &mut exited, &mut kernel, &mut user) };
    (micros(user), micros(kernel))
}

/// A list method the title called on this thread took `ns` nanoseconds: told by the
/// method's entry scope (`crate::budget::entry`), from its own two clock reads. Every list
/// method counts (the methods that take the draw state alone did before).
pub(crate) fn list_call(ns: u64) {
    add(C::RecordCalls, 1);
    add(C::RecordNs, ns);
    if !RECORDED_SINCE.load(Relaxed) && !RECORDED_SINCE.swap(true, Relaxed) {
        let began = now_us().saturating_sub(ns / 1000);
        add(C::UntilRecordUs, began.saturating_sub(PRESENTED_AT.load(Relaxed)));
    }
    let (unsent, calls) = UNSENT.with(|u| u.get());
    let (unsent, calls) = (unsent + ns, calls + 1);
    // By thread, a couple of milliseconds at a time.
    if unsent < 2_000_000 {
        UNSENT.with(|u| u.set((unsent, calls)));
        return;
    }
    UNSENT.with(|u| u.set((0, 0)));
    let id = thread_id();
    let mut all = RECORDERS.lock().unwrap_or_else(|e| e.into_inner());
    match all.iter_mut().find(|r| r.0 == id) {
        Some(r) => (r.1, r.2) = (r.1 + unsent, r.2 + calls),
        None => all.push((id, unsent, calls)),
    }
}

/// A submission is called on this thread (before it waits for the
/// submission lock).
pub(crate) fn submission_begins() {
    crate::stalls::submitting(true);
    let now = now_us();
    if !SUBMITTED_SINCE.load(Relaxed) && !SUBMITTED_SINCE.swap(true, Relaxed) {
        add(C::UntilSubmitUs, now.saturating_sub(PRESENTED_AT.load(Relaxed)));
    }
}

/// The submission returns.
pub(crate) fn submission_ends() {
    crate::stalls::submitting(false);
    SUBMITTED_AT.store(now_us(), Relaxed);
}

/// The title calls its present (before our own copy to the window runs).
pub(crate) fn present_begins() {
    add(C::TailUs, now_us().saturating_sub(SUBMITTED_AT.load(Relaxed)));
}

/// A frame was presented.
pub(crate) fn presented() {
    add(C::Frames, 1);
    PRESENTED_AT.store(now_us(), Relaxed);
    RECORDED_SINCE.store(false, Relaxed);
    SUBMITTED_SINCE.store(false, Relaxed);
    crate::watch::clear_asked_pages();
}

fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// Logs the counters per frame, and the ranges that took most bytes. The
/// counters run from line to line, so they are divided by the frames
/// presented since the last line, not by `frames` (the frames of the last
/// second or more: a frame of seconds after an unlogged one showed two
/// frames' counts as one's).
pub(crate) fn log(frames: f64) {
    let counted = COUNT[C::Frames as usize].swap(0, Relaxed);
    let frames = if counted > 0 { counted as f64 } else { frames };
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    let faults = page_faults();
    let faults = faults - FAULTS.swap(faults, Relaxed).min(faults);
    let c: Vec<f64> = COUNT
        .iter()
        .map(|a| a.swap(0, Relaxed) as f64 / frames)
        .collect();
    let mb = |v: f64| v / 1e6;
    let ms = |v: f64| v / 1e3;
    // Said only when there were any: a healthy run's line does not change.
    // The counters are every build's (`crate::storestat`).
    let (failed, failed_bytes) = crate::storestat::take();
    let failed_stores = if failed > 0 {
        format!(", {:.0} stores failed ({:.0} bytes)", failed as f64 / frames, failed_bytes as f64 / frames)
    } else {
        String::new()
    };
    tracing::info!(
        "copies per frame: {:.1} submissions; {:.0} title copies {:.1} MB; table fills {:.1} MB asked, {:.1} MB filled in {:.2} ms; \
         uploads on the pool {:.1} MB, join {:.2} ms; descriptor-copy sync {:.2} ms ({:.0} records); {:.0} refreshes {:.1} MB, \
         {:.0} skipped {:.1} MB; {:.0} runs {:.1} ms; {:.0} queries {:.2} ms, moves {:.2} ms, \
         {:.2} MB unmapped; {:.0} page faults; {:.0} watches {:.2} ms ({:.0} failed {:.2} ms), \
         {:.1} MB asked about (repeats counted), {:.1} MB unchanged, {:.0} distinct pages since \
         the last present; {:.1} fallbacks; {:.0} guard faults; {:.0} idle \
         copies kept, {:.0} let go by pressure; textures {:.1} MB watched and \
         changed, {:.1} MB hashed, {:.1} MB of them changed{failed_stores}",
        c[C::Submissions as usize],
        c[C::Copies as usize],
        mb(c[C::CopyBytes as usize]),
        mb(c[C::TableFillRequested as usize]),
        mb(c[C::TableFillFilled as usize]),
        ms(c[C::TableFillUs as usize]),
        mb(c[C::UploadPoolBytes as usize]),
        ms(c[C::UploadPoolJoinUs as usize]),
        c[C::DescriptorCopyNs as usize] / 1e6,
        c[C::DescriptorCopyRecords as usize],
        c[C::Refreshes as usize],
        mb(c[C::RefreshBytes as usize]),
        c[C::Skipped as usize],
        mb(c[C::SkippedBytes as usize]),
        c[C::Runs as usize],
        ms(c[C::RunUs as usize]),
        c[C::Queries as usize],
        ms(c[C::QueryUs as usize]),
        ms(c[C::MoveUs as usize]),
        mb(c[C::Unmapped as usize]),
        faults as f64 / frames,
        c[C::Watches as usize],
        ms(c[C::WatchUs as usize]),
        c[C::AskFailed as usize],
        ms(c[C::AskFailedUs as usize]),
        mb(c[C::Checked as usize]),
        mb(c[C::Unchanged as usize]),
        crate::watch::distinct_asked_pages() as f64,
        c[C::Fallbacks as usize],
        c[C::GuardFaults as usize],
        c[C::IdleKept as usize],
        c[C::PressureGone as usize],
        mb(c[C::TexWatched as usize]),
        mb(c[C::TexHashed as usize]),
        mb(c[C::TexChanged as usize]),
        failed_stores = failed_stores,
    );
    tracing::info!(
        "recording per frame ({frames:.0} frames): {:.0} list calls {:.1} ms (all recording \
         threads); {:.1} ms waiting for pipelines; dynamic slots: {:.0} tables asked, {:.0} \
         refused, {:.1} ms searching; notes {:.1} ms; shader-written ranges: {:.0} steps in \
         {:.1} submissions, snapshots {:.1} ms, {:.1} MB compared back; sparse reads {:.1} ms; \
         {:.0} guards in {:.1} ms (the fault handler's included); {:.1} MB copied whole (memory not watched); {:.1} frame waits \
         sleeping {:.1} ms (all threads); ranges of shaders that cannot store to buffers: \
         {:.0} steps {:.1} MB left out",
        c[C::RecordCalls as usize],
        c[C::RecordNs as usize] / 1e6,
        ms(c[C::PipelineWaitUs as usize]),
        c[C::DynamicAsked as usize],
        c[C::DynamicRefused as usize],
        c[C::DynamicNs as usize] / 1e6,
        c[C::NoteNs as usize] / 1e6,
        c[C::GpuWrites as usize],
        c[C::BackSubmissions as usize],
        ms(c[C::SnapshotUs as usize]),
        mb(c[C::BackCompared as usize]),
        ms(c[C::SparseUs as usize]),
        c[C::Guards as usize],
        ms(c[C::GuardUs as usize]),
        mb(c[C::WholeBytes as usize]),
        c[C::PaceWaits as usize],
        ms(c[C::PaceUs as usize]),
        c[C::GpuWritesLeftOut as usize],
        mb(c[C::GpuWritesLeftOutBytes as usize]),
    );
    let [together, written, mapped] =
        [0, 1, 2].map(|n| crate::watch::ALIASES[n].load(std::sync::atomic::Ordering::Relaxed));
    tracing::info!(
        "guards over several views of a page, since the start: {together} blocks guarded with their other views, \
         {written} views opened by a write through another, {mapped} ranges opened because a view of them was mapped"
    );
    // Where the wall clock of a frame goes: the phases of a frame, the
    // processor time of the whole process, and what each submitting and
    // recording thread did.
    static CPU: Mutex<(u64, u64)> = Mutex::new((0, 0));
    let (user, kernel) = process_times();
    let (user, kernel) = {
        let mut last = CPU.lock().unwrap_or_else(|e| e.into_inner());
        let spent = (user.saturating_sub(last.0), kernel.saturating_sub(last.1));
        *last = (user, kernel);
        (spent.0 as f64 / 1e3 / frames, spent.1 as f64 / 1e3 / frames)
    };
    tracing::info!(
        "frame account per frame: after a present {:.1} ms until the first list call and {:.1} ms \
         until the first submission; {:.1} ms from the last submission to the present call; \
         lists reset in {:.1} ms; the process ran {user:.0} ms user + {kernel:.0} ms kernel on \
         all threads; what shaders wrote: {:.1} ms finding {:.0} pieces that differ, {:.1} ms \
         putting them back; texture uploads: {:.0} recorded, {:.0} left out; {:.1} held \
         submissions run at the present with their waits not met; waits for memory values: \
         {:.0} looks {:.1} ms; {:.0} host lists made {:.1} ms; {:.0} parts of held submissions \
         ran alone that the next part could have run with",
        ms(c[C::UntilRecordUs as usize]),
        ms(c[C::UntilSubmitUs as usize]),
        ms(c[C::TailUs as usize]),
        c[C::ResetNs as usize] / 1e6,
        ms(c[C::BackScanUs as usize]),
        c[C::BackDiffer as usize],
        ms(c[C::BackSyncUs as usize]),
        c[C::TexUploads as usize],
        c[C::TexSkipped as usize],
        c[C::Forced as usize],
        c[C::WaitChecks as usize],
        c[C::WaitCheckNs as usize] / 1e6,
        c[C::HostLists as usize],
        c[C::HostListNs as usize] / 1e6,
        c[C::PartsTogether as usize],
    );
    // What is copied whole, largest first: memory nobody tells us about.
    let whole = WHOLE.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut whole: Vec<_> = whole.into_iter().flatten().collect();
    whole.sort_by_key(|&((_, len), (n, _))| std::cmp::Reverse(len as u64 * n));
    tracing::info!(
        "  copied whole (memory not watched): {:.1} ms per frame in {} ranges; the largest follow",
        ms(c[C::WholeUs as usize]),
        whole.len()
    );
    for ((src, len), (n, us)) in whole.into_iter().take(6) {
        tracing::info!(
            "  copied whole from {src:#x}+{len:#x}: {:.1}x/frame, {:.2} MB/frame, {:.2} ms/frame",
            n as f64 / frames,
            (len as u64 * n) as f64 / 1e6 / frames,
            us as f64 / 1e3 / frames
        );
    }
    // The host's waits for submissions and settles: many short ones or a
    // few long ones, how much of it is for submissions the title gets
    // nothing back from, and the longest one.
    let longest = std::mem::take(&mut *LONGEST.lock().unwrap_or_else(|e| e.into_inner()));
    tracing::info!(
        "  host waits for submissions and settles per frame (a synchronous submission from its \
         send to its retirement, a settle its fence wait alone; not time on the card): {:.1} wait samples under 2 ms ({:.1} ms), {:.1} of 2-8 ms ({:.1} ms), {:.1} \
         of 8-32 ms ({:.1} ms), {:.1} longer ({:.1} ms); {:.1} of them for submissions that hand \
         nothing back to the title ({:.1} ms); the longest since the last line: {:.1} ms, {}",
        c[C::CardUnder2 as usize],
        ms(c[C::CardUnder2Us as usize]),
        c[C::Card2To8 as usize],
        ms(c[C::Card2To8Us as usize]),
        c[C::Card8To32 as usize],
        ms(c[C::Card8To32Us as usize]),
        c[C::CardOver32 as usize],
        ms(c[C::CardOver32Us as usize]),
        c[C::NothingBack as usize],
        ms(c[C::NothingBackUs as usize]),
        longest.0 as f64 / 1e3,
        longest.1
    );
    let by = WAITS_BY_WORK.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut by: Vec<_> = by.into_iter().flatten().collect();
    by.sort_by_key(|&(_, (_, us))| std::cmp::Reverse(us));
    for (what, (n, us)) in by.into_iter().take(12) {
        tracing::info!(
            "  host waits by what was waited for (send to retirement, or a settle; not time on the card): \
             {:.1} ms a frame in {:.1} wait samples a frame: {}",
            us as f64 / 1e3 / frames,
            n as f64 / frames,
            what.chars().take(700).collect::<String>()
        );
    }
    let by = COMMANDS_BY_WORK.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut by: Vec<_> = by.into_iter().flatten().collect();
    by.sort_by_key(|&(_, (_, us, _))| std::cmp::Reverse(us));
    // All of them together, and by their first word: the forty lines below
    // are the heaviest only (under half of the time waited in one run), and what the
    // commands do not account for is the layer's own lists (the card
    // copies' passes, preludes) and the waits themselves.
    if !by.is_empty() {
        let mut kinds: Vec<(String, u64, u64)> = Vec::new();
        for (what, (n, us, _)) in &by {
            let word = what.split(' ').next().unwrap_or("").to_owned();
            match kinds.iter_mut().find(|k| k.0 == word) {
                Some(k) => (k.1, k.2) = (k.1 + n, k.2 + us),
                None => kinds.push((word, *n, *us)),
            }
        }
        kinds.sort_by_key(|k| std::cmp::Reverse(k.2));
        let all: u64 = kinds.iter().map(|k| k.2).sum();
        let text: Vec<String> =
            kinds.iter().map(|(word, n, us)| format!("{word} {:.1} ms in {:.0}", *us as f64 / 1e3 / frames, *n as f64 / frames)).collect();
        tracing::info!(
            "  the card's time by command (timing run), all {} kinds together: {:.1} ms a frame; by first word: {}",
            by.len(),
            all as f64 / 1e3 / frames,
            text.join(", ")
        );
    }
    for (what, (n, us, longest)) in by.into_iter().take(40) {
        tracing::info!(
            "  the card's time by command (timing run): {:.2} ms a frame in {:.1} commands a frame, the longest {:.2} ms: {what}",
            us as f64 / 1e3 / frames,
            n as f64 / frames,
            longest as f64 / 1e3
        );
    }
    tracing::info!(
        "  waits for every earlier shader write, per frame: {:.0} for the title's barrier groups, {:.0} \
         for its barriers on buffers; {:.0} indirect commands bounded (a dispatch of ours between two \
         such waits each); barrier experiment {}",
        c[C::WaitsForGroups as usize],
        c[C::WaitsForBuffers as usize],
        c[C::IndirectBounded as usize],
        barrier_test()
    );
    // What is compared after submissions, largest first, and how much of
    // it shaders had written.
    let compared = COMPARED.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut compared: Vec<_> = compared.into_iter().flatten().collect();
    compared.sort_by_key(|&((_, len), (n, _))| std::cmp::Reverse(len as u64 * n));
    tracing::info!("  compared after submissions: {} ranges; the largest follow", compared.len());
    for ((src, len), (n, written)) in compared.into_iter().take(8) {
        tracing::info!(
            "  compared {src:#x}+{len:#x}: {:.1}x/frame, {:.2} MB/frame, {:.3} MB/frame of it written back",
            n as f64 / frames,
            (len as u64 * n) as f64 / 1e6 / frames,
            written as f64 / 1e6 / frames
        );
    }
    let mut recorders = std::mem::take(&mut *RECORDERS.lock().unwrap_or_else(|e| e.into_inner()));
    recorders.sort_by_key(|r| std::cmp::Reverse(r.1));
    let busiest: Vec<String> = recorders
        .iter()
        .take(6)
        .map(|r| format!("thread {} {:.0} ms in {:.0} calls", r.0, r.1 as f64 / 1e6 / frames, r.2 as f64 / frames))
        .collect();
    tracing::info!("  list calls by thread: {} threads; {}", recorders.len(), busiest.join(", "));
    log_memory();
    let ranges = RANGES.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut ranges: Vec<_> = ranges.into_iter().flatten().collect();
    ranges.sort_by_key(|&(_, (_, bytes, _, _))| std::cmp::Reverse(bytes));
    for ((src, len), (n, bytes, us, _)) in ranges.into_iter().take(10) {
        // SAFETY: plain data filled in by the query.
        let mut info: windows_sys::Win32::System::Memory::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        crate::kernelstat::virtual_query(crate::kernelstat::Place::CopyStats, src, &mut info);
        let kind = info.Type;
        tracing::info!(
            "  copied from {src:#x}+{len:#x} (memory {kind:#x}): {:.1}x/frame, {:.2} MB/frame, {:.2} ms/frame",
            n as f64 / frames,
            bytes as f64 / 1e6 / frames,
            us as f64 / 1e3 / frames
        );
    }
}
