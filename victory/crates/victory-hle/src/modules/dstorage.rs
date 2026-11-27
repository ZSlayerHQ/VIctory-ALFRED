//! The console's DirectStorage (`XCRDAPI.dll`): a factory that opens files
//! and creates request queues and status arrays the title polls. Its
//! interfaces have grown since 2020 (new interface ids, new slots); the
//! slots here are those of the console interfaces the runtime implements.
//! A slot without a method logs its offset.
//!
//! Submitted requests of every queue run on one pool of workers, as the
//! console's hardware runs them: in parallel, started in queue order,
//! queues served by weighted round robin over their priorities (about 10x
//! per level, counted in bytes), with at most the staging buffer's worth of
//! compressed source in flight. Each request reads its source (a file or
//! memory), inflates zlib data and/or unpacks BCPack data, and writes the
//! destination (an address, or physical pages). Completion is what stays
//! in order: a status entry is written once every request enqueued before
//! it on its queue has finished, with the first failure (in queue order)
//! since the previous status entry, or success.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ffi::{CStr, c_char, c_void};
use std::os::windows::fs::FileExt;
use std::os::windows::io::FromRawHandle;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};

use victory_formats::bcpack::{self, Tables};

use super::xasync::{E_INVALIDARG, E_NOTIMPL, E_PENDING};
use super::xgameruntime::{Guid, HRESULT, Methods, Object, S_OK, add_ref, missing, put};
use crate::{HleModule, hle_export};

/// The console factory interface the runtime supports.
const FACTORY: Guid = Guid::new(
    0x6295_9d11,
    0xd45f,
    0x4884,
    [0xa0, 0xa3, 0xf7, 0x86, 0x2e, 0xa2, 0x43, 0xae],
);
const E_NOINTERFACE: HRESULT = 0x8000_4002_u32 as i32;
const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;
/// `HRESULT_FROM_WIN32(ERROR_HANDLE_EOF)`: the source ends early.
const END_OF_FILE: HRESULT = 0x8007_0026_u32 as i32;
/// `HRESULT_FROM_WIN32(ERROR_INVALID_DATA)`: a bad compressed stream.
const INVALID_DATA: HRESULT = 0x8007_000D_u32 as i32;

type Out = *mut *mut c_void;

static FACTORY_OBJECT: Object = Object::new("dstorage factory", &FACTORY_VTABLE);

pub extern "system" fn DStorageGetFactory(iid: *const Guid, out: Out) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    // SAFETY: the title passes a GUID (null tolerated).
    match unsafe { iid.as_ref() } {
        Some(&iid) if iid == FACTORY => {
            put(out, std::ptr::from_ref(&FACTORY_OBJECT).cast_mut().cast())
        }
        iid => {
            tracing::warn!(iid = ?iid.map(Guid::to_string), "DStorageGetFactory: unknown interface");
            E_NOINTERFACE
        }
    }
}

/// One of our objects with state: the runtime's object header (method
/// table first), a reference count, then `T`.
#[repr(C)]
struct Instance<T> {
    base: Object,
    refs: AtomicU32,
    state: T,
}

impl<T> Instance<T> {
    fn create(base: Object, state: T) -> *mut c_void {
        Box::into_raw(Box::new(Instance {
            base,
            refs: AtomicU32::new(1),
            state,
        }))
        .cast()
    }
}

extern "system" fn add_ref_instance<T>(this: *const Instance<T>) -> u32 {
    let _entry = victory_gpu::hle_call();
    // SAFETY: an instance we created and the title still holds.
    unsafe { &*this }.refs.fetch_add(1, Ordering::Relaxed) + 1
}

extern "system" fn release<T>(this: *mut Instance<T>) -> u32 {
    let _entry = victory_gpu::hle_call();
    // SAFETY: as above.
    let left = unsafe { &*this }.refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if left == 0 {
        // SAFETY: the last reference; created by `Instance::create`.
        drop(unsafe { Box::from_raw(this) });
    }
    left
}

fn iid_text(iid: *const Guid) -> String {
    // SAFETY: the title passes a GUID (null tolerated).
    unsafe { iid.as_ref() }.map_or_else(|| "null".into(), Guid::to_string)
}

/// `HRESULT_FROM_WIN32` of a host error.
fn from_win32(error: &std::io::Error) -> HRESULT {
    error
        .raw_os_error()
        .map_or(E_INVALIDARG, |e| (0x8007_0000 | (e as u32 & 0xffff)) as i32)
}

// The factory.

type Factory = *const Object;

/// Opens `path` for reading; the file object owns the handle.
extern "system" fn open_file(_: Factory, path: *const u16, iid: *const Guid, out: Out) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    const GENERIC_READ: u32 = 0x8000_0000;
    const FILE_SHARE_READ: u32 = 1;
    const OPEN_EXISTING: u32 = 3;
    if path.is_null() {
        return E_INVALIDARG;
    }
    // Goes through `files`, so paths on our virtual drives work too.
    let handle = super::files::CreateFileW(
        path,
        GENERIC_READ,
        FILE_SHARE_READ,
        std::ptr::null(),
        OPEN_EXISTING,
        0,
        std::ptr::null_mut(),
    );
    // SAFETY: the title passes a NUL-terminated path.
    let name = unsafe {
        let len = (0..).take_while(|&i| *path.add(i) != 0).count();
        String::from_utf16_lossy(std::slice::from_raw_parts(path, len))
    };
    if handle as isize == -1 {
        let error = std::io::Error::last_os_error();
        tracing::warn!(path = name, %error, "dstorage: opening a file failed");
        return from_win32(&error);
    }
    // The walls trace names a request's file by it, while the trace is on.
    let kept = (walls_trace() != 0).then(|| name.clone());
    tracing::info!(path = name, iid = iid_text(iid), "dstorage: file opened");
    // SAFETY: a handle we just opened and nobody else owns.
    let file = unsafe { std::fs::File::from_raw_handle(handle) };
    let file = Instance::create(Object::new("dstorage file", &FILE_VTABLE), File { file });
    if let Some(kept) = kept {
        WALLS_FILES.lock().unwrap_or_else(|e| e.into_inner()).push((file as usize, kept));
    }
    put(out, file)
}

/// `(capacity, name, iid, out)`.
extern "system" fn create_status_array(
    _: Factory,
    capacity: u32,
    name: *const c_char,
    iid: *const Guid,
    out: Out,
) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    tracing::info!(
        capacity,
        name = text(name),
        iid = iid_text(iid),
        "dstorage: status array"
    );
    let entries = (0..capacity).map(|_| AtomicI32::new(S_OK)).collect();
    let array = Instance::create(
        Object::new("dstorage status array", &STATUS_VTABLE),
        StatusArray { entries },
    );
    put(out, array)
}

/// DSTORAGE_QUEUE_DESC1, passed to IDStorageFactoryX2::CreateQueue1 (after
/// the base fork's 7f0156c09).
#[repr(C)]
struct QueueDesc {
    source_type: u64,
    capacity: u16,
    priority: i8,
    name: *const c_char,
    options: u64,
}

const AUTO_SUBMIT_DISABLED: u64 = 1;

extern "system" fn create_queue(
    _: Factory,
    desc: *const QueueDesc,
    iid: *const Guid,
    out: Out,
) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    // SAFETY: the title passes its description (null tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let name = text(desc.name);
    let auto_submit = desc.options & AUTO_SUBMIT_DISABLED == 0;
    tracing::info!(
        source = desc.source_type,
        capacity = desc.capacity,
        priority = desc.priority,
        name,
        auto_submit,
        iid = iid_text(iid),
        "dstorage: queue"
    );
    let Some(scheduler) = scheduler() else {
        return E_OUTOFMEMORY;
    };
    let mut lanes = scheduler.lock();
    let id = lanes.next_id;
    lanes.next_id += 1;
    let lane = Lane::new(desc.priority, desc.capacity);
    lanes.lanes.insert(id, lane);
    drop(lanes);
    put(
        out,
        Instance::create(
            Object::new("dstorage queue", &QUEUE_VTABLE),
            Queue { id, auto_submit },
        ),
    )
}

/// A title that asks for a staging size stops if this fails.
extern "system" fn set_staging_buffer_size(_: Factory, size: u32) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    tracing::info!(size, "dstorage: staging buffer size");
    if let Some(scheduler) = scheduler() {
        scheduler.lock().staging = u64::from(size);
    }
    S_OK
}

fn text(name: *const c_char) -> String {
    if name.is_null() {
        return String::new();
    }
    // SAFETY: a NUL-terminated name from the title.
    unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

static FACTORY_VTABLE: Methods<16> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    missing::<3> as *const (),            // 0x18 (2020: queue creation)
    open_file as *const (),               // 0x20
    create_status_array as *const (),     // 0x28
    missing::<6> as *const (),            // 0x30 (2020: CPU affinity)
    missing::<7> as *const (),            // 0x38 (2020: debug flags)
    set_staging_buffer_size as *const (), // 0x40
    missing::<9> as *const (),
    create_queue as *const (), // 0x50
    missing::<11> as *const (),
    missing::<12> as *const (),
    missing::<13> as *const (),
    missing::<14> as *const (),
    missing::<15> as *const (),
]);

// Files.

struct File {
    file: std::fs::File,
}

static FILE_VTABLE: Methods<8> = Methods([
    missing::<0> as *const (),
    add_ref_instance::<File> as *const (),
    release::<File> as *const (),
    missing::<3> as *const (),
    missing::<4> as *const (),
    missing::<5> as *const (),
    missing::<6> as *const (),
    missing::<7> as *const (),
]);

// Status arrays.

struct StatusArray {
    entries: Box<[AtomicI32]>,
}

type Array = *mut Instance<StatusArray>;

/// The result of the requests before entry `index`: pending, the first
/// failure, or success.
extern "system" fn status_result(this: Array, index: u32) -> HRESULT {
    let _entry = victory_gpu::hle_call();
    // SAFETY: an array we created and the title still holds.
    let entries = &unsafe { &*this }.state.entries;
    entries
        .get(index as usize)
        .map_or(E_INVALIDARG, |e| e.load(Ordering::Acquire))
}

/// The title polls this (entry 0, a `Sleep(1)` loop) and never
/// reads the result itself.
extern "system" fn is_complete(this: Array, index: u32) -> bool {
    let _entry = victory_gpu::hle_call();
    status_result(this, index) != E_PENDING
}

static STATUS_VTABLE: Methods<5> = Methods([
    missing::<0> as *const (),
    add_ref_instance::<StatusArray> as *const (),
    release::<StatusArray> as *const (),
    is_complete as *const (),   // 0x18
    status_result as *const (), // 0x20
]);

// Queues.

/// A queue object: its lane in the scheduler.
struct Queue {
    id: u64,
    /// Whether the queue submits on its own once half of it is enqueued: a
    /// queue the title created with AUTO_SUBMIT_DISABLED submits only when
    /// asked, as DirectStorage's own queues do.
    auto_submit: bool,
}

impl Drop for Queue {
    fn drop(&mut self) {
        close_lane(self.id);
    }
}

type QueueRef = *const Instance<Queue>;

/// Every queue's requests, run by a pool of workers.
struct Scheduler {
    lanes: Mutex<Lanes>,
    /// Requests to start, or staging space freed.
    work: Condvar,
    /// A request finished (for full queues and `close`).
    finished: Condvar,
}

/// Priority weights per class (low, normal, high and real-time), as the
/// console's: about 10x per level, counted against request sizes.
const COST: [u64; 3] = [100, 10, 1];
/// The console's default staging buffer.
const STAGING: u64 = 32 << 20;

struct Lanes {
    lanes: HashMap<u64, Lane>,
    next_id: u64,
    /// Submission order over all queues.
    next_seq: u64,
    /// Bytes of staging held by running requests, and the limit.
    staged: u64,
    staging: u64,
    /// A start waited for staging space since the last finish.
    starved: bool,
    /// Weighted round robin: each class's virtual time (bytes × cost), and
    /// the time of the last start; within a class, the lane clock of the
    /// last start.
    clock: [u64; 3],
    now: u64,
    lane_now: [u64; 3],
    /// Since the last debug summary: requests, bytes written, time spent
    /// running them.
    done: (u64, u64, std::time::Duration),
    refused: u64,
    /// BCPack bytes written and time spent on them.
    bcpack: (u64, std::time::Duration),
    since: Option<std::time::Instant>,
    /// Since the last debug summary: times a request was held back for its
    /// page array ([`Lanes::hold_back`]); and when that was last said.
    held_back: u64,
    held_said: Option<std::time::Instant>,
}

/// A queue's requests in the scheduler.
struct Lane {
    class: usize,
    capacity: usize,
    /// Bytes started, for round robin among the lanes of a class.
    clock: u64,
    /// Enqueued, not yet submitted.
    unsubmitted: Vec<Command>,
    /// Submitted, by sequence number: reads not started, status entries
    /// not written, reads running, and failures not yet reported.
    reads: VecDeque<(u64, Read)>,
    statuses: VecDeque<(u64, Status)>,
    running: BTreeSet<u64>,
    failures: BTreeMap<u64, HRESULT>,
    closed: bool,
    /// Reads held back until their page arrays are filled
    /// ([`Lanes::hold_back`]), by sequence number: started, not finished.
    held: BTreeMap<u64, Held>,
}

/// A read held back for its page array: due to be looked at again from
/// `due`; held since `since`, `times` times so far.
struct Held {
    request: Option<Read>,
    due: std::time::Instant,
    since: std::time::Instant,
    times: u32,
}

// SAFETY: requests point into title memory, which stays valid until the
// title has seen them complete, and at our file objects, on which a read
// holds a reference (`Read`); file objects and status arrays count their
// references atomically.
unsafe impl Send for Lane {}

/// A read a worker runs.
struct Job {
    lane: u64,
    seq: u64,
    request: Read,
    staging: u64,
}

// SAFETY: as for `Lane`.
unsafe impl Send for Job {}

/// The scheduler, with its workers started on first use; `None` if not
/// even one worker could start.
fn scheduler() -> Option<&'static Scheduler> {
    static SCHEDULER: OnceLock<Option<Scheduler>> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let scheduler = SCHEDULER
        .get_or_init(|| {
            Some(Scheduler {
                lanes: Mutex::new(Lanes {
                    lanes: HashMap::new(),
                    next_id: 0,
                    next_seq: 0,
                    staged: 0,
                    staging: STAGING,
                    starved: false,
                    clock: [0; 3],
                    now: 0,
                    lane_now: [0; 3],
                    done: Default::default(),
                    refused: 0,
                    bcpack: Default::default(),
                    since: None,
                    held_back: 0,
                    held_said: None,
                }),
                work: Condvar::new(),
                finished: Condvar::new(),
            })
        })
        .as_ref()?;
    STARTED.call_once(|| {
        walls_start();
        let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
        let started = (0..workers)
            .filter(|i| {
                std::thread::Builder::new()
                    .name(format!("dstorage {i}"))
                    .spawn(|| work(scheduler))
                    .is_ok()
            })
            .count();
        tracing::info!(workers = started, "dstorage: workers");
    });
    Some(scheduler)
}

impl Scheduler {
    fn lock(&self) -> MutexGuard<'_, Lanes> {
        self.lanes.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The walls trace (`VICTORY_WALLS_TRACE`, diagnostics builds): 0 or unset:
/// off; 1: a line for every read request that fails, with its file; 2: for
/// every request (the list of a launch's reads, by file and offset, is what
/// a launch with an interior's walls drawn and one without are compared by).
#[cfg(any(test, feature = "diagnostics"))]
fn walls_trace() -> u8 {
    #[cfg(test)]
    {
        let level = tests::WALLS_LEVEL.load(Ordering::Relaxed);
        if level != 0 {
            return level;
        }
    }
    static ON: OnceLock<u8> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("VICTORY_WALLS_TRACE").ok().and_then(|v| v.trim().parse::<u8>().ok()).unwrap_or(0)
    })
}

/// Off in a plain build.
#[cfg(not(any(test, feature = "diagnostics")))]
fn walls_trace() -> u8 {
    0
}

/// The files opened while the walls trace is on, by the address of our file
/// object, with their paths (a request names its file by that address).
static WALLS_FILES: Mutex<Vec<(usize, String)>> = Mutex::new(Vec::new());

/// The walls trace's line for a request that is done, owned: the worker
/// makes it, and hands it to the trace's thread once the request's
/// completion is published ([`walls_send`]).
struct WallsLine {
    file: String,
    offset: u64,
    source_size: u32,
    destination_size: u32,
    options: u64,
    ms: u64,
    hr: u32,
}

/// The walls trace's line for a request that is done, if the trace wants it.
fn walls_request(request: &Request, result: &Result<(), HRESULT>, started: std::time::Instant) -> Option<WallsLine> {
    let level = walls_trace();
    if level == 0 || (level == 1 && result.is_ok()) {
        return None;
    }
    let file = if request.options & MEMORY_SOURCE != 0 {
        "memory".to_string()
    } else {
        let files = WALLS_FILES.lock().unwrap_or_else(|e| e.into_inner());
        let source = request.source as usize;
        files.iter().find(|f| f.0 == source).map_or_else(|| "?".to_string(), |f| f.1.clone())
    };
    Some(WallsLine {
        file,
        offset: request.offset,
        source_size: request.source_size,
        destination_size: request.destination_size,
        options: request.options,
        ms: started.elapsed().as_millis() as u64,
        hr: result.err().map_or(0, |hr| hr as u32),
    })
}

/// The walls trace's queue to its thread (`None`: the thread did not
/// start), set when the workers start ([`walls_start`]).
static WALLS_OUT: OnceLock<Option<std::sync::mpsc::SyncSender<WallsLine>>> = OnceLock::new();
/// Lines dropped since the last one written.
static WALLS_DROPPED: AtomicU64 = AtomicU64::new(0);

/// Starts the walls trace's own thread, with the workers and before any of
/// them runs (`scheduler`), when the trace is on (and in tests). A thread
/// that cannot start is said so, once; the trace's lines are then dropped
/// and counted.
fn walls_start() {
    use std::sync::mpsc::sync_channel;
    if !cfg!(test) && walls_trace() == 0 {
        return;
    }
    let out = WALLS_OUT.get_or_init(|| {
        let (out, lines) = sync_channel::<WallsLine>(4096);
        let write = move || {
            for line in lines {
                #[cfg(test)]
                tests::walls_sink(&line.file);
                tracing::info!(
                    file = line.file,
                    offset = line.offset,
                    source_size = line.source_size,
                    destination_size = line.destination_size,
                    to_pages = line.options & PAGE_DESTINATION != 0,
                    options = format_args!("{:#x}", line.options),
                    ms = line.ms,
                    ok = line.hr == 0,
                    hr = format_args!("{:#010x}", line.hr),
                    dropped_before = WALLS_DROPPED.swap(0, Ordering::Relaxed),
                    "walls trace: a read request is done"
                );
            }
        };
        std::thread::Builder::new().name("dstorage walls".into()).spawn(write).ok().map(|_| out)
    });
    if out.is_none() {
        tracing::warn!("dstorage: the walls trace's thread did not start; its lines are dropped");
    }
}

/// Hands `line` to the walls trace's own thread, which writes it: a worker
/// only tries the queue, never waits for the log's output (a stalled one
/// would hold the worker, and with it the requests after its own) nor
/// starts the thread ([`walls_start`]). Lines past the queue's room are
/// dropped and counted; the next line written says how many.
fn walls_send(line: WallsLine) {
    let sent = WALLS_OUT.get().and_then(Option::as_ref).is_some_and(|out| out.try_send(line).is_ok());
    if !sent {
        WALLS_DROPPED.fetch_add(1, Ordering::Relaxed);
    }
}

/// A worker: runs the next read of the most deserving queue.
fn work(scheduler: &'static Scheduler) {
    let mut lanes = scheduler.lock();
    loop {
        let Some(mut job) = lanes.start() else {
            // A read held back is looked at again when it is due.
            lanes = match lanes.next_due() {
                Some(due) => {
                    let wait = due.saturating_duration_since(std::time::Instant::now());
                    match scheduler.work.wait_timeout(lanes, wait) {
                        Ok((lanes, _)) => lanes,
                        Err(e) => e.into_inner().0,
                    }
                }
                None => scheduler.work.wait(lanes).unwrap_or_else(|e| e.into_inner()),
            };
            continue;
        };
        drop(lanes);
        let started = std::time::Instant::now();
        let Ran::Done(result) = execute(&job.request) else {
            // Its page array is not filled yet: back to the scheduler, with
            // its file, and this worker takes the next read.
            lanes = scheduler.lock();
            lanes.hold_back(job);
            if std::mem::take(&mut lanes.starved) {
                scheduler.work.notify_all();
            }
            // It is no longer running: a `close_lane` waiting for the
            // queue's running reads goes on.
            scheduler.finished.notify_all();
            continue;
        };
        let walls = walls_request(&job.request, &result, started);
        // The read's file goes before its completion is published: a title
        // that sees the status may delete the file, or expect it closed.
        job.request.file = None;
        lanes = scheduler.lock();
        lanes.finish(&job, result);
        #[cfg(feature = "diagnostics")]
        victory_gpu::stalls::storage_depth(lanes.depth());
        lanes.count(&job, started);
        if std::mem::take(&mut lanes.starved) {
            scheduler.work.notify_all();
        }
        scheduler.finished.notify_all();
        // The completion is published: now the trace's line.
        if let Some(line) = walls {
            drop(lanes);
            walls_send(line);
            lanes = scheduler.lock();
        }
    }
}

impl Lanes {
    /// Requests waiting to start, running, or held back for their page
    /// arrays, on every queue (diagnostics: the stall line's depth). A held
    /// read put back to start again, or running, keeps an entry without its
    /// request among the held: it is counted where it is, once.
    #[cfg(feature = "diagnostics")]
    fn depth(&self) -> u64 {
        self.lanes
            .values()
            .map(|l| (l.reads.len() + l.running.len() + l.held.values().filter(|h| h.request.is_some()).count()) as u64)
            .sum()
    }

    /// Counts a finished request; every 5 s a debug line sums them up.
    fn count(&mut self, job: &Job, started: std::time::Instant) {
        let (requests, bytes, busy) = &mut self.done;
        *requests += 1;
        *bytes += u64::from(job.request.destination_size);
        *busy += started.elapsed();
        if job.request.options & BCPACK != 0 {
            self.bcpack.0 += u64::from(job.request.destination_size);
            self.bcpack.1 += started.elapsed();
        }
        let since = *self.since.get_or_insert(started);
        let elapsed = since.elapsed();
        if elapsed.as_secs() >= 5 {
            let waiting: usize = self.lanes.values().map(|l| l.reads.len()).sum();
            let reading = READING.swap(0, Ordering::Relaxed) as f64 / 1e9;
            tracing::debug!(
                requests = *requests,
                mb_per_s = format_args!("{:.1}", *bytes as f64 / elapsed.as_secs_f64() / 1e6),
                busy_workers = format_args!("{:.2}", busy.as_secs_f64() / elapsed.as_secs_f64()),
                reading_workers = format_args!("{:.2}", reading / elapsed.as_secs_f64()),
                staging_waits = std::mem::take(&mut self.refused),
                bcpack_mb = self.bcpack.0 / 1_000_000,
                bcpack_workers = format_args!("{:.2}", self.bcpack.1.as_secs_f64() / elapsed.as_secs_f64()),
                waiting,
                held_back = std::mem::take(&mut self.held_back),
                held_now = self.lanes.values().map(|l| l.held.len()).sum::<usize>(),
                page_failures = format_args!(
                    "{} never filled, {} unknown page, {} host failure",
                    PAGE_FAILURES[0].load(Ordering::Relaxed),
                    PAGE_FAILURES[1].load(Ordering::Relaxed),
                    PAGE_FAILURES[2].load(Ordering::Relaxed)
                ),
                "dstorage: last {:.1} s",
                elapsed.as_secs_f64()
            );
            self.done = Default::default();
            self.bcpack = Default::default();
            self.since = None;
        }
    }

    /// Starts the next read of the class with the least virtual time, from
    /// its lane that has started the fewest bytes, if the staging buffer
    /// has room for it (or nothing else holds any). Reads held back whose
    /// look is due go first in their lanes again.
    fn start(&mut self) -> Option<Job> {
        let now = std::time::Instant::now();
        for lane in self.lanes.values_mut() {
            // Held reads were started before every read still in `reads`:
            // put back in front, oldest first, the lane's order stays.
            let due: Vec<u64> = lane
                .held
                .iter()
                .filter(|(_, h)| h.request.is_some() && h.due <= now)
                .map(|(&seq, _)| seq)
                .collect();
            for seq in due.into_iter().rev() {
                if let Some(request) = lane.held.get_mut(&seq).and_then(|h| h.request.take()) {
                    lane.reads.push_front((seq, request));
                }
            }
        }
        let busy = || self.lanes.iter().filter(|(_, l)| !l.reads.is_empty());
        let (_, class) = busy().map(|(_, l)| (self.clock[l.class], l.class)).min()?;
        let (_, _, id) = busy()
            .filter(|(_, l)| l.class == class)
            .filter_map(|(&id, l)| Some((l.clock, l.reads.front()?.0, id)))
            .min()?;
        let lane = self.lanes.get_mut(&id)?;
        let (seq, staging) = lane.reads.front().map(|(seq, read)| (*seq, staging_bytes(read)))?;
        if self.staged > 0 && self.staged + staging > self.staging {
            self.starved = true;
            self.refused += 1;
            return None;
        }
        let (_, request) = lane.reads.pop_front()?;
        lane.running.insert(seq);
        self.staged += staging;
        let size = u64::from(request.destination_size.max(request.source_size)).max(1);
        self.now = self.clock[class];
        self.clock[class] += size * COST[class];
        self.lane_now[class] = lane.clock;
        lane.clock += size;
        Some(Job {
            lane: id,
            seq,
            request,
            staging,
        })
    }

    /// A read whose page array is not filled yet (some entry still 0, the
    /// title fills it after it enqueued the request): held back, looked at
    /// again later (2 ms, doubling to 50 ms), with no give-up while its
    /// queue lives; closing the queue drops it ([`close_lane`]). It gives
    /// back its staging and the bytes it was counted as started, and stays
    /// unfinished: the status entries after it wait for it, as
    /// DirectStorage has them (a status is written once every request
    /// enqueued before it is done; requests themselves may finish in any
    /// order), while the reads after it run. It keeps its file ([`Read`]).
    /// Said once a second.
    fn hold_back(&mut self, job: Job) {
        self.staged -= job.staging;
        self.held_back += 1;
        let now = std::time::Instant::now();
        let size = u64::from(job.request.destination_size.max(job.request.source_size)).max(1);
        let Some(lane) = self.lanes.get_mut(&job.lane) else {
            return;
        };
        self.clock[lane.class] = self.clock[lane.class].saturating_sub(size * COST[lane.class]);
        lane.clock = lane.clock.saturating_sub(size);
        lane.running.remove(&job.seq);
        let held = lane.held.entry(job.seq).or_insert(Held { request: None, due: now, since: now, times: 0 });
        held.times += 1;
        held.due = now + std::time::Duration::from_millis((2u64 << held.times.min(5)).min(50));
        held.request = Some(job.request);
        if self.held_said.is_none_or(|said| said.elapsed() >= std::time::Duration::from_secs(1)) {
            self.held_said = Some(now);
            let held = self.lanes.values().flat_map(|l| l.held.values());
            let (n, longest) = held.fold((0, std::time::Duration::ZERO), |(n, most), h| {
                (n + 1, most.max(h.since.elapsed()))
            });
            tracing::warn!(
                requests = n,
                longest_ms = longest.as_millis() as u64,
                "dstorage: requests wait for their page arrays to be filled"
            );
        }
    }

    /// When the next held read is due to be looked at again, if any is held.
    fn next_due(&self) -> Option<std::time::Instant> {
        self.lanes.values().flat_map(|l| l.held.values()).filter(|h| h.request.is_some()).map(|h| h.due).min()
    }

    fn finish(&mut self, job: &Job, result: Result<(), HRESULT>) {
        self.staged -= job.staging;
        if let Some(lane) = self.lanes.get_mut(&job.lane) {
            lane.running.remove(&job.seq);
            lane.held.remove(&job.seq);
            if let Err(hr) = result {
                lane.failures.insert(job.seq, hr);
            }
            lane.flush();
        }
    }

    /// Moves a queue's enqueued commands to the workers.
    fn submit(&mut self, id: u64) {
        let Some(class) = self.lanes.get(&id).map(|l| l.class) else {
            return;
        };
        // A class that had nothing waiting starts from now, not from the
        // credit it left behind.
        if !self
            .lanes
            .values()
            .any(|l| l.class == class && !l.reads.is_empty())
        {
            self.clock[class] = self.clock[class].max(self.now);
        }
        let Some(lane) = self.lanes.get_mut(&id) else {
            return;
        };
        if lane.reads.is_empty() {
            lane.clock = lane.clock.max(self.lane_now[class]);
        }
        for command in std::mem::take(&mut lane.unsubmitted) {
            let seq = self.next_seq;
            self.next_seq += 1;
            match command {
                Command::Read(request) => lane.reads.push_back((seq, request)),
                Command::Status(status) => lane.statuses.push_back((seq, status)),
            }
        }
        lane.flush();
    }
}

impl Lane {
    fn new(priority: i8, capacity: u16) -> Lane {
        Lane {
            class: priority.clamp(-1, 1).wrapping_add(1) as usize,
            capacity: usize::from(capacity),
            clock: 0,
            unsubmitted: Vec::new(),
            reads: VecDeque::new(),
            statuses: VecDeque::new(),
            running: BTreeSet::new(),
            failures: BTreeMap::new(),
            closed: false,
            held: BTreeMap::new(),
        }
    }

    /// Requests and status entries held, submitted or not.
    fn entries(&self) -> usize {
        let held = self.held.values().filter(|h| h.request.is_some()).count();
        self.unsubmitted.len() + self.reads.len() + self.statuses.len() + self.running.len() + held
    }

    /// Writes the status entries whose earlier reads have all finished.
    fn flush(&mut self) {
        while let Some(&(seq, _)) = self.statuses.front() {
            let waiting = self.reads.front().map(|r| r.0);
            let held = self.held.keys().next().copied();
            if waiting.into_iter().chain(self.running.first().copied()).chain(held).any(|s| s < seq) {
                return;
            }
            let Some((_, status)) = self.statuses.pop_front() else {
                return;
            };
            let later = self.failures.split_off(&seq);
            let first = std::mem::replace(&mut self.failures, later)
                .into_values()
                .next();
            status
                .entry()
                .store(first.unwrap_or(S_OK), Ordering::Release);
        }
    }
}

/// Staging bytes a request holds while it runs: what it buffers between
/// the file and the destination. Raw reads into an address no larger than
/// a worker's scratch ([`SCRATCH`]) go through it, and memory sources need
/// none (as on the console); a larger raw read into an address, and every
/// raw read into pages, is read whole beside ([`read_whole`]).
fn staging_bytes(request: &Request) -> u64 {
    let buffered = request.options & (ZLIB | BCPACK | PAGE_DESTINATION) != 0
        || request.source_size as usize > SCRATCH;
    if buffered && request.options & MEMORY_SOURCE == 0 {
        u64::from(request.source_size)
    } else {
        0
    }
}

/// Ends a queue: what was not started is dropped (its reads release their
/// files, its status entries their arrays), and this waits for the reads
/// already running (they write title memory).
fn close_lane(id: u64) {
    let Some(scheduler) = scheduler() else {
        return;
    };
    let mut lanes = scheduler.lock();
    let Some(lane) = lanes.lanes.get_mut(&id) else {
        return;
    };
    lane.closed = true;
    lane.unsubmitted.clear();
    lane.reads.clear();
    lane.statuses.clear(); // releases their arrays
    // Reads still waiting for their page arrays go with the queue.
    let unfilled = lane.held.values().filter(|h| h.request.is_some()).count();
    lane.held.retain(|_, h| h.request.is_none());
    if unfilled > 0 {
        PAGE_FAILURES[0].fetch_add(unfilled as u64, Ordering::Relaxed);
        tracing::warn!(
            requests = unfilled,
            "dstorage: a queue closed with requests whose page arrays were never filled; dropped"
        );
    }
    while lanes.lanes.get(&id).is_some_and(|l| !l.running.is_empty()) {
        lanes = scheduler
            .finished
            .wait(lanes)
            .unwrap_or_else(|e| e.into_inner());
    }
    lanes.lanes.remove(&id);
    #[cfg(feature = "diagnostics")]
    victory_gpu::stalls::storage_depth(lanes.depth());
    scheduler.finished.notify_all();
}

/// A read request in the console's layout, 0x48 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct Request {
    options: u64,
    /// An address, or with `PAGE_DESTINATION` an array of 64 KiB page
    /// numbers.
    destination: *mut c_void,
    /// With `PAGE_DESTINATION`, the low 16 bits are where in the first page
    /// the destination starts.
    page_offset: u64,
    destination_size: u32,
    /// Our file object, or with `MEMORY_SOURCE` an address.
    source: *const c_void,
    offset: u64,
    source_size: u32,
    /// Compressed size between two decompression stages; unused here.
    intermediate_size: u32,
    /// Matched by cancellation; a caller that wants no match leaves it 0.
    tag: u64,
    name: *const c_char,
}

// Request options.
/// Inflate zlib (RFC 1950) data.
const ZLIB: u64 = 1;
/// Bits 1-4: decode BCPack data of this BC format number (0: none).
const BCPACK: u64 = 0xf << 1;
const PAGE_DESTINATION: u64 = 1 << 21;
const MEMORY_SOURCE: u64 = 1 << 22;
// Bits 5-20 select a further transform, bit 23 a source in physical pages;
// neither is supported, so requests with them fail.
/// The options a request may have and run.
const SUPPORTED: u64 = ZLIB | BCPACK | PAGE_DESTINATION | MEMORY_SOURCE;

const PAGE: usize = 64 * 1024;

enum Command {
    Read(Read),
    Status(Status),
}

/// A read request with a reference of its own on its file, if it reads one
/// (a request this runs, without `MEMORY_SOURCE`): the title may release
/// its reference once it has enqueued the request (after the base fork's
/// 6d8ceb6ef). One owner from enqueue to the end, never copied: enqueued,
/// submitted, running, held back for its page array and looked at again
/// ([`Lanes::hold_back`]). The reference goes when the read is done, before
/// its completion is published ([`work`]), or with the read when its queue
/// drops it ([`close_lane`], or [`queue`] on a closed queue).
struct Read {
    request: Request,
    file: Option<FileRef>,
}

impl Read {
    fn new(request: Request) -> Read {
        let reads_file =
            request.options & !SUPPORTED == 0 && request.options & MEMORY_SOURCE == 0 && !request.source.is_null();
        let file = reads_file.then(|| {
            let file = request.source.cast::<Instance<File>>().cast_mut();
            add_ref_instance(file);
            FileRef(file)
        });
        Read { request, file }
    }
}

impl std::ops::Deref for Read {
    type Target = Request;

    fn deref(&self) -> &Request {
        &self.request
    }
}

/// A reference on one of our file objects, released when dropped.
struct FileRef(*mut Instance<File>);

impl Drop for FileRef {
    fn drop(&mut self) {
        release(self.0);
    }
}

/// A status entry to write, holding a reference on its array.
struct Status {
    array: Array,
    index: u32,
}

impl Drop for Status {
    fn drop(&mut self) {
        release(self.array);
    }
}

impl Status {
    fn entry(&self) -> &AtomicI32 {
        // SAFETY: we hold a reference; the index was checked when enqueued.
        &unsafe { &*self.array }.state.entries[self.index as usize]
    }
}

extern "system" fn enqueue_request(this: QueueRef, request: *const Request) {
    let _entry = victory_gpu::hle_call();
    // SAFETY: the title passes its request (null tolerated).
    let Some(&request) = (unsafe { request.as_ref() }) else {
        return;
    };
    queue(this, Command::Read(Read::new(request)));
}

extern "system" fn enqueue_status(this: QueueRef, array: Array, index: u32) {
    let _entry = victory_gpu::hle_call();
    // SAFETY: the title passes an array we created (null tolerated).
    let Some(entries) = (unsafe { array.as_ref() }).map(|a| &a.state.entries) else {
        return;
    };
    let Some(entry) = entries.get(index as usize) else {
        tracing::warn!(index, "dstorage: status entry out of range");
        return;
    };
    entry.store(E_PENDING, Ordering::Release);
    add_ref_instance(array);
    queue(this, Command::Status(Status { array, index }));
}

/// Adds a command to the queue; blocks while the queue is full, and (for a
/// queue that was not created with AUTO_SUBMIT_DISABLED) submits on its own
/// once half of it is enqueued.
fn queue(this: QueueRef, command: Command) {
    // SAFETY: a queue we created and the title still holds.
    let Queue { id, auto_submit } = unsafe { &*this }.state;
    let Some(scheduler) = scheduler() else {
        return;
    };
    let mut lanes = scheduler.lock();
    loop {
        let Some(lane) = lanes.lanes.get(&id).filter(|l| !l.closed) else {
            return; // closed: dropped (a status releases its array, a read its file)
        };
        // One slot stays free, as the console's count of empty slots says.
        if lane.entries() + 1 < lane.capacity.max(2) {
            break;
        }
        lanes = scheduler
            .finished
            .wait(lanes)
            .unwrap_or_else(|e| e.into_inner());
    }
    let Some(lane) = lanes.lanes.get_mut(&id) else {
        return;
    };
    lane.unsubmitted.push(command);
    if auto_submit && lane.unsubmitted.len() >= (lane.capacity / 2).max(1) {
        lanes.submit(id);
        #[cfg(feature = "diagnostics")]
        victory_gpu::stalls::storage_depth(lanes.depth());
        scheduler.work.notify_all();
    }
}

extern "system" fn submit(this: QueueRef) {
    let _entry = victory_gpu::hle_call();
    // SAFETY: as above.
    let id = unsafe { &*this }.state.id;
    let Some(scheduler) = scheduler() else {
        return;
    };
    let mut lanes = scheduler.lock();
    let pending = lanes.lanes.get(&id).map_or(0, |l| l.unsubmitted.len());
    tracing::debug!(commands = pending, "dstorage: submit");
    lanes.submit(id);
    #[cfg(feature = "diagnostics")]
    victory_gpu::stalls::storage_depth(lanes.depth());
    scheduler.work.notify_all();
    // Status entries with no read before them are written at once
    // (`Lane::flush`), which frees their slots: an enqueuer waiting for
    // room goes on. On a queue that does not submit itself that is the
    // only wake it gets when no read is left to finish.
    scheduler.finished.notify_all();
}

/// Ends the queue whatever its reference count: what was not started never
/// completes, and this returns once the reads already running have
/// finished. A caller closes the queue this way, not through `Release`.
extern "system" fn close(this: QueueRef) {
    let _entry = victory_gpu::hle_call();
    // SAFETY: as above.
    close_lane(unsafe { &*this }.state.id);
}

/// What running a request came to: done, or not run yet because its page
/// array is not filled ([`Lanes::hold_back`]).
enum Ran {
    Done(Result<(), HRESULT>),
    NotFilled,
}

/// Since the start: requests into physical pages that did not get them, by
/// why: their page array never filled (dropped when the queue closed); a
/// page number we do not know (freed, or never ours: numbers are not given
/// out twice, so no wait brings it back); the host failing to write known
/// pages after its retries.
static PAGE_FAILURES: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// Runs a request ([`execute_now`]), unless its destination's page array
/// still holds a 0 (no page is 0): the console interface reads the array
/// only when it gets to the request, so a caller may fill it after the
/// request is enqueued; here a busy machine can get to it first, so the
/// request is deferred until the array is filled.
fn execute(request: &Request) -> Ran {
    if request.options & PAGE_DESTINATION != 0 && !request.destination.is_null() {
        let len = if request.options & (ZLIB | BCPACK) == 0 { request.source_size } else { request.destination_size };
        let n = ((request.page_offset & 0xffff) as usize + len as usize).div_ceil(PAGE);
        let array = request.destination.cast::<usize>();
        // SAFETY: the caller passes a page number per 64 KiB of destination
        // and keeps the array until the request completes; it may write it
        // from another thread meanwhile.
        if (0..n).any(|i| unsafe { array.add(i).read_volatile() } == 0) {
            return Ran::NotFilled;
        }
    }
    Ran::Done(execute_now(request))
}

/// Reads, inflates and/or unpacks, and writes one request.
fn execute_now(request: &Request) -> Result<(), HRESULT> {
    let options = request.options;
    let fail = |hr: HRESULT, why: &str| {
        tracing::warn!(
            options = format_args!("{options:#x}"),
            offset = request.offset,
            source_size = request.source_size,
            destination_size = request.destination_size,
            intermediate_size = request.intermediate_size,
            tag = request.tag,
            name = text(request.name),
            hr = format_args!("{:#010x}", hr as u32),
            "dstorage: request failed: {why}"
        );
        hr
    };
    if options & !SUPPORTED != 0 {
        return Err(fail(E_NOTIMPL, "unsupported options"));
    }
    if request.source.is_null() || request.destination.is_null() {
        return Err(fail(E_INVALIDARG, "no source or destination"));
    }
    let bc = ((options & BCPACK) >> 1) as u8;
    let size = request.destination_size as usize;
    if options & (ZLIB | BCPACK) == 0 {
        if request.source_size > request.destination_size {
            return Err(fail(E_INVALIDARG, "source larger than destination"));
        }
        let size = request.source_size as usize;
        // A file read into a title address is the kernel's write, and the
        // kernel does not fault into the GPU side's guard: a block guarded
        // again between `kernel_will_write` and the read (the guard over all
        // the views of a page protects other views from other threads) made
        // the read fail with ERROR_INVALID_USER_BUFFER, and the title stops
        // at the failed request (runs amd-C and amd-C2, seconds after the
        // world's first frame). It is read beside, a piece at a time, and
        // copied in (`read_beside`): our copy faults into the guard like the
        // title's own writes.
        let beside = options & (PAGE_DESTINATION | MEMORY_SOURCE) == 0;
        // The file holds the whole range, asked before any of it is written
        // (the pieces are copied in as they come).
        if options & MEMORY_SOURCE == 0 {
            file_holds(request).map_err(|hr| fail(hr, "reading"))?;
        }
        // A file read into physical pages is read whole first, before the
        // writable page window opens: read in place there, a short read the
        // host then failed left new bytes in the first pages and old ones in
        // the rest. Whole or not at all, as for an address (`read_beside`).
        if options & (PAGE_DESTINATION | MEMORY_SOURCE) == PAGE_DESTINATION {
            return read_whole(request, size, |bytes| {
                into_destination(request, size, |out| {
                    out.copy_from_slice(bytes);
                    Ok(size)
                })
            })
            .map_err(|hr| fail(hr, "copying"));
        }
        return into_destination(request, size, |out| {
            if beside { read_beside(request, out) } else { read_into(request, out) }.map(|()| size)
        })
        .map_err(|hr| fail(hr, "copying"));
    }
    let tables = if bc != 0 { Some(bcpack_tables()?) } else { None };
    let source = read_source(request).map_err(|hr| fail(hr, "reading the source"))?;
    // The console's order: zlib first, then BCPack.
    let source = if bc != 0 && options & ZLIB != 0 {
        let mut packed = vec![0; request.intermediate_size as usize];
        let n = inflate(&source, &mut packed).map_err(|hr| fail(hr, "zlib"))?;
        packed.truncate(n);
        Cow::Owned(packed)
    } else {
        source
    };
    into_destination(request, size, |out| match tables {
        Some(tables) => {
            #[cfg(feature = "diagnostics")]
            let decoding = std::time::Instant::now();
            let decoded = unpack_bcpack(bc, &source, out, tables)
                .map(|()| out.len())
                .map_err(|e| fail(INVALID_DATA, &e.to_string()));
            #[cfg(feature = "diagnostics")]
            victory_gpu::stalls::storage_decoded(decoding.elapsed().as_nanos() as u64);
            decoded
        }
        None => inflate(&source, out).map_err(|hr| fail(hr, "zlib")),
    })?;
    tracing::trace!(
        options = format_args!("{options:#x}"),
        offset = request.offset,
        size,
        "dstorage: request done"
    );
    Ok(())
}

/// Inflates a zlib stream into `out`, returning the bytes written. A stream
/// that ends short of `out` passes: the title's chunks fill it exactly, and
/// nothing says that is required.
fn inflate(source: &[u8], out: &mut [u8]) -> Result<usize, HRESULT> {
    thread_local! {
        static INFLATE: std::cell::RefCell<zlib_rs::Inflate> =
            std::cell::RefCell::new(zlib_rs::Inflate::new(true, 15));
    }
    INFLATE.with_borrow_mut(|state| {
        state.reset(true);
        // `Inflate::decompress` cannot distinguish a valid empty stream from
        // an insufficient buffer when `avail_out == 0`, so use a 1-byte scratch
        // destination and verify `written <= out_len` afterward.
        let out_len = out.len();
        let mut scratch = [0u8; 1];
        let dest = if out_len == 0 { &mut scratch[..] } else { out };
        let (mut in_pos, mut out_pos) = (0usize, 0usize);
        let status = loop {
            match state.decompress(
                &source[in_pos..],
                &mut dest[out_pos..],
                zlib_rs::InflateFlush::NoFlush,
            ) {
                Ok(zlib_rs::Status::StreamEnd) => break Ok(zlib_rs::Status::StreamEnd),
                Ok(zlib_rs::Status::Ok) => {
                    let next_in = state.total_in() as usize;
                    let next_out = state.total_out() as usize;
                    if next_in == in_pos && next_out == out_pos {
                        break Ok(zlib_rs::Status::BufError);
                    }
                    in_pos = next_in;
                    out_pos = next_out;
                }
                other => break other,
            }
        };
        let written = state.total_out() as usize;
        match status {
            Ok(zlib_rs::Status::StreamEnd) if written <= out_len => Ok(written),
            other => {
                tracing::debug!(?other, "dstorage: zlib stream");
                Err(INVALID_DATA)
            }
        }
    })
}

/// Whether BCPack data is unpacked as it was first (the default): one lane step and one block
/// after another on the worker, at the worker's priority. `VICTORY_BCPACK_SERIAL` set to
/// `0`, `false`, `off`, `no` or an empty value selects [`unpack_bcpack`]'s parallel way,
/// which writes the same bytes: off by default until a game run shows the single black
/// texels on fine surface detail are its (2026-11-26).
fn bcpack_serial() -> bool {
    #[cfg(test)]
    if let Some(serial) = tests::bcpack_serial_choice() {
        return serial;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_BCPACK_SERIAL").is_none_or(|v| !switch_off(&v))
    })
}

/// Whether a switch value selects the parallel unpack: `0`, `false`, `off` or `no` (any case),
/// with surrounding space ignored, and an empty value too.
fn switch_off(value: &std::ffi::OsStr) -> bool {
    let text = value.to_string_lossy();
    let text = text.trim();
    text.is_empty()
        || text.eq_ignore_ascii_case("0")
        || text.eq_ignore_ascii_case("false")
        || text.eq_ignore_ascii_case("off")
        || text.eq_ignore_ascii_case("no")
}

/// Whether every BCPack request is unpacked three ways and compared
/// (`VICTORY_BCPACK_CHECK=1`; off by default, read once). With the switch
/// off nothing here runs.
fn bcpack_check() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let value = std::env::var_os("VICTORY_BCPACK_CHECK");
        let on = bcpack::diag::enabled(value.as_deref());
        if on {
            static AT_EXIT: std::sync::Once = std::sync::Once::new();
            AT_EXIT.call_once(|| {
                // SAFETY: registers a function with the host CRT's exit table.
                unsafe { atexit(bcpack_check_at_exit); }
            });
        }
        on
    })
}

unsafe extern "C" {
    fn atexit(f: extern "C" fn()) -> i32;
}

/// Requests the check compared, and how the paths differed.
static BCPACK_CHECKED: AtomicU64 = AtomicU64::new(0);
static BCPACK_WIDE_DIFFS: AtomicU64 = AtomicU64::new(0);
static BCPACK_ONE_LANE_DIFFS: AtomicU64 = AtomicU64::new(0);
/// Differing requests logged so far (the first 40, then only counted).
static BCPACK_DIFF_LINES: AtomicU64 = AtomicU64::new(0);
/// Differing requests offered to the keep folder (the first 16).
static BCPACK_KEPT: AtomicU64 = AtomicU64::new(0);
static BCPACK_KEEP_FAILED: AtomicBool = AtomicBool::new(false);

fn bcpack_check_summary(why: &str) {
    tracing::info!(
        checked = BCPACK_CHECKED.load(Ordering::Relaxed),
        differ_wide = BCPACK_WIDE_DIFFS.load(Ordering::Relaxed),
        differ_one_lane = BCPACK_ONE_LANE_DIFFS.load(Ordering::Relaxed),
        "dstorage: BCPack check: {why}"
    );
}

extern "C" fn bcpack_check_at_exit() {
    bcpack_check_summary("at shutdown");
}

/// `VICTORY_BCPACK_CHECK_KEEP`, read once with the check on: where the first
/// differing requests are kept, if it is outside every repository.
fn bcpack_check_keep() -> Option<&'static std::path::Path> {
    static DIR: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::path::PathBuf::from(std::env::var_os("VICTORY_BCPACK_CHECK_KEEP")?);
        if !bcpack::diag::keep_allowed(&dir) {
            tracing::warn!(
                "dstorage: BCPack check: VICTORY_BCPACK_CHECK_KEEP is not under the captures root; nothing will be kept"
            );
            return None;
        }
        if std::fs::create_dir_all(&dir).is_err() {
            tracing::warn!(
                "dstorage: BCPack check: the keep folder cannot be made; nothing will be kept"
            );
            return None;
        }
        Some(dir)
    })
    .as_deref()
}

/// [`unpack_bcpack`] with `VICTORY_BCPACK_CHECK=1`: the serial decode (the
/// bytes handed on, so the picture stays right), `unpack` with the lanes it
/// chooses today and `unpack` forced to one lane. Differing requests log
/// one line (the first 40) and the first 16 can be kept.
fn unpack_bcpack_checked(
    bc: u8,
    source: &[u8],
    out: &mut [u8],
    tables: &Tables,
) -> Result<(), bcpack::Error> {
    let ordinal = BCPACK_CHECKED.fetch_add(1, Ordering::Relaxed) + 1;
    if ordinal.is_multiple_of(2000) {
        bcpack_check_summary("every 2000 requests");
    }
    // The serial path is the result handed on; it writes `out` first and
    // nothing after it touches those bytes.
    let serial = bcpack::decode(bc, source, out, tables);
    let helpers = bcpack_helpers();
    let mut wide_out = vec![0u8; out.len()];
    let wide = bcpack::unpack(bc, source, &mut wide_out, tables, helpers);
    let mut one_out = vec![0u8; out.len()];
    let one = bcpack::unpack_one_lane(bc, source, &mut one_out, tables, helpers);
    serial?;
    let block = bcpack::diag::layout(bc).map_or(1, |(block, _, _)| block);
    let mut comparison = bcpack::diag::compare(out, &wide_out, &one_out, block);
    if let Err(error) = &wide {
        comparison.wide = None;
        comparison.wide_error = Some(error.0);
    }
    if let Err(error) = &one {
        comparison.one_lane = None;
        comparison.one_lane_error = Some(error.0);
    }
    let wide_differs = comparison.wide.is_some() || comparison.wide_error.is_some();
    let one_differs = comparison.one_lane.is_some() || comparison.one_lane_error.is_some();
    if wide_differs {
        BCPACK_WIDE_DIFFS.fetch_add(1, Ordering::Relaxed);
    }
    if one_differs {
        BCPACK_ONE_LANE_DIFFS.fetch_add(1, Ordering::Relaxed);
    }
    if comparison.differs() {
        let hash = blake3::hash(source).to_hex().to_string();
        let report = bcpack::diag::Report {
            ordinal,
            bc,
            source_size: source.len(),
            destination_size: out.len(),
            hash: &hash,
            mode0: bcpack::diag::mode0(bc, out),
            comparison,
        };
        if BCPACK_DIFF_LINES.fetch_add(1, Ordering::Relaxed) < 40 {
            tracing::warn!("{}", report.line());
        }
        if let Some(dir) = bcpack_check_keep() {
            let kept = BCPACK_KEPT.fetch_add(1, Ordering::Relaxed);
            let helpers = helpers.map_or(0, bcpack::Pool::threads);
            if kept < 16
                && bcpack::diag::keep_request(dir, &report, helpers, source).is_err()
                && !BCPACK_KEEP_FAILED.swap(true, Ordering::Relaxed)
            {
                tracing::warn!("dstorage: BCPack check: a kept request could not be written");
            }
        }
    }
    Ok(())
}

/// Unpacks a request's BCPack data. BCPack is what textures arrive in, and
/// unpacking ran at 15-20 MB/s of BC7 a worker, all of the console's work
/// on the processor; a camera cut has many workers at it at once, beside
/// the title's threads (all at normal priority). With
/// [`bcpack::unpack`] (`VICTORY_BCPACK_SERIAL=0`; the serial way is the
/// default until a game run shows the black texels on fine detail are its,
/// 2026-11-26): a round's lanes at once (AVX-512, about 8x), the blocks
/// packed by [`bcpack_helpers`] too, and the worker below normal priority
/// while it unpacks, so that the title's threads come first.
fn unpack_bcpack(bc: u8, source: &[u8], out: &mut [u8], tables: &Tables) -> Result<(), bcpack::Error> {
    if bcpack_check() {
        return unpack_bcpack_checked(bc, source, out, tables);
    }
    if bcpack_serial() {
        return bcpack::decode(bc, source, out, tables);
    }
    let result = {
        let _below = BelowNormal::now();
        #[cfg(test)]
        tests::note_bcpack_priority(true);
        bcpack::unpack(bc, source, out, tables, bcpack_helpers())
    };
    #[cfg(test)]
    tests::note_bcpack_priority(false);
    result
}

/// The threads that help pack BCPack blocks: a quarter of the processors,
/// at most 4, below normal priority. `None` if none started.
fn bcpack_helpers() -> Option<&'static bcpack::Pool> {
    static POOL: OnceLock<bcpack::Pool> = OnceLock::new();
    let pool = POOL.get_or_init(|| {
        let n = std::thread::available_parallelism().map_or(1, |n| n.get() / 4).clamp(1, 4);
        let pool = bcpack::Pool::new(n, "dstorage bcpack", below_normal);
        tracing::info!(helpers = pool.threads(), "dstorage: BCPack helpers");
        pool
    });
    (pool.threads() > 0).then_some(pool)
}

/// Puts the calling thread below normal priority.
fn below_normal() {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL};
    // SAFETY: a plain call on the calling thread's pseudo handle.
    unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL) };
}

/// The calling thread below normal priority until this is dropped, then
/// back at the priority it had; left as it is when that priority could not
/// be read (there would be nothing to go back to).
struct BelowNormal(Option<i32>);

impl BelowNormal {
    fn now() -> BelowNormal {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadPriority};
        // SAFETY: a plain query of the calling thread.
        BelowNormal::after(unsafe { GetThreadPriority(GetCurrentThread()) })
    }

    /// Lowers the calling thread, whose priority the query gave as `before`
    /// (`THREAD_PRIORITY_ERROR_RETURN`, `i32::MAX`: it failed, not lowered).
    fn after(before: i32) -> BelowNormal {
        if before == i32::MAX {
            return BelowNormal(None);
        }
        below_normal();
        BelowNormal(Some(before))
    }
}

impl Drop for BelowNormal {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority};
        if let Some(before) = self.0 {
            // SAFETY: as in `below_normal`.
            unsafe { SetThreadPriority(GetCurrentThread(), before) };
        }
    }
}

/// The BCPack format's constant tables, parsed once for the process. If
/// they do not parse every BCPack request fails: one warning, then debug.
fn bcpack_tables() -> Result<&'static Tables, HRESULT> {
    static TABLES: OnceLock<Result<Tables, String>> = OnceLock::new();
    static WARNED: AtomicBool = AtomicBool::new(false);
    match TABLES.get_or_init(load_bcpack_tables) {
        Ok(tables) => Ok(tables),
        Err(why) if !WARNED.swap(true, Ordering::Relaxed) => {
            tracing::warn!("dstorage: BCPack requests fail: {why} (more at debug)");
            Err(E_NOTIMPL)
        }
        Err(_) => {
            tracing::debug!("dstorage: BCPack request failed: no tables");
            Err(E_NOTIMPL)
        }
    }
}

/// The tables VIctory ships with.
#[cfg(not(test))]
fn load_bcpack_tables() -> Result<Tables, String> {
    Tables::builtin().map_err(|e| format!("the built-in BCPack tables: {e}"))
}

/// Tests decode streams made against tables of their own, from this folder.
#[cfg(test)]
static BCPACK_DIR: OnceLock<std::path::PathBuf> = OnceLock::new();

#[cfg(test)]
fn use_bcpack_tables(dir: std::path::PathBuf) {
    let _ = BCPACK_DIR.set(dir);
}

#[cfg(test)]
fn load_bcpack_tables() -> Result<Tables, String> {
    let dir = BCPACK_DIR.get().ok_or("no folder set for the test tables")?;
    let bytes = std::fs::read(dir.join("tables.bin")).map_err(|e| e.to_string())?;
    Tables::parse(&bytes).map_err(|e| e.to_string())
}

/// A compressed request's source: borrowed from memory, or read from the
/// file.
fn read_source(request: &Request) -> Result<Cow<'static, [u8]>, HRESULT> {
    let size = request.source_size as usize;
    if request.options & MEMORY_SOURCE != 0 {
        // SAFETY: the title's source holds `size` bytes until completion.
        let bytes = unsafe { std::slice::from_raw_parts(request.source.cast(), size) };
        return Ok(Cow::Borrowed(bytes));
    }
    let mut bytes = vec![0; size];
    read_into(request, &mut bytes)?;
    Ok(Cow::Owned(bytes))
}

/// Nanoseconds workers spent reading files, for the debug summary.
static READING: AtomicU64 = AtomicU64::new(0);

/// Fills `out` (the source size) from the request's file or memory.
fn read_into(request: &Request, out: &mut [u8]) -> Result<(), HRESULT> {
    read_at(request, 0, out)
}

/// What a worker reads a raw read into a title address into, when the read
/// is no larger (its scratch, kept for the next): such reads hold no
/// staging. A larger one is read into a buffer of its own, which the
/// scheduler counts against the staging budget ([`staging_bytes`]).
const SCRATCH: usize = 1 << 20;

/// Fills `out`, a title address, from the request's file, whole or not at
/// all: read beside first, into the worker's scratch or, for a read larger
/// than that, into a buffer of its own (counted against the staging budget,
/// [`staging_bytes`]: one larger than the whole budget runs alone), and
/// copied in from user mode once all of it is read, where a block guarded
/// again meanwhile faults into the GPU side's guard. A read that fails
/// leaves the destination as it was, as at 0e8374f0: a title that polls for
/// completion only (`IsComplete` says done for a failed request too) must
/// not find some new bytes and some old there. Without room for the bytes
/// beside, the read fails (E_OUTOFMEMORY) before writing anything.
fn read_beside(request: &Request, out: &mut [u8]) -> Result<(), HRESULT> {
    thread_local! {
        static BESIDE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    #[cfg(test)]
    let refused = tests::NO_ROOM.get();
    #[cfg(not(test))]
    let refused = false;
    if out.len() > SCRATCH {
        return read_whole(request, out.len(), |bytes| {
            out.copy_from_slice(bytes);
            Ok(())
        });
    }
    BESIDE.with_borrow_mut(|scratch| {
        if scratch.len() < out.len() {
            if refused || scratch.try_reserve_exact(out.len() - scratch.len()).is_err() {
                return Err(E_OUTOFMEMORY);
            }
            scratch.resize(out.len(), 0);
        }
        #[cfg(test)]
        tests::BESIDE_MOST.fetch_max(scratch.len(), Ordering::Relaxed);
        let piece = &mut scratch[..out.len()];
        read_into(request, piece)?;
        out.copy_from_slice(piece);
        Ok(())
    })
}

/// Whether the request's file holds its whole source range: one past the
/// end fails as reading it would (END_OF_FILE), and one whose end does not
/// fit in an offset with E_INVALIDARG. After it, the offsets of the range's
/// pieces cannot overflow.
fn file_holds(request: &Request) -> Result<(), HRESULT> {
    let end = request
        .offset
        .checked_add(u64::from(request.source_size))
        .ok_or(E_INVALIDARG)?;
    // SAFETY: a file object we created; the read holds a reference on it
    // (`Read`).
    let file = &unsafe { &*request.source.cast::<Instance<File>>() }.state.file;
    let len = file.metadata().map_err(|e| from_win32(&e))?.len();
    if end > len { Err(END_OF_FILE) } else { Ok(()) }
}

/// Reads the request's first `len` source bytes whole into a buffer of its
/// own, which the scheduler counts against the staging budget
/// ([`staging_bytes`]), and hands them to `then`: nothing is written where
/// the title sees it before all of the read succeeded. Without room for the
/// buffer, E_OUTOFMEMORY.
fn read_whole<T>(request: &Request, len: usize, then: impl FnOnce(&[u8]) -> Result<T, HRESULT>) -> Result<T, HRESULT> {
    #[cfg(test)]
    let refused = tests::NO_ROOM.get();
    #[cfg(not(test))]
    let refused = false;
    let mut staged = Vec::new();
    if refused || staged.try_reserve_exact(len).is_err() {
        return Err(E_OUTOFMEMORY);
    }
    staged.resize(len, 0);
    #[cfg(test)]
    let _held = tests::Staged::held(len);
    read_into(request, &mut staged)?;
    then(&staged)
}

/// Fills `out` from the request's file or memory, `skip` bytes into its
/// source.
fn read_at(request: &Request, skip: usize, out: &mut [u8]) -> Result<(), HRESULT> {
    if request.options & MEMORY_SOURCE != 0 {
        // SAFETY: the title's source holds `skip + out.len()` bytes (its
        // source size) until completion.
        let source = unsafe { request.source.cast::<u8>().add(skip) };
        // SAFETY: as above.
        out.copy_from_slice(unsafe { std::slice::from_raw_parts(source, out.len()) });
        return Ok(());
    }
    // SAFETY: a file object we created; the read holds a reference on it
    // (`Read`).
    let file = &unsafe { &*request.source.cast::<Instance<File>>() }
        .state
        .file;
    let started = std::time::Instant::now();
    let mut done = 0;
    let result = loop {
        if done == out.len() {
            break Ok(());
        }
        let rest = &mut out[done..];
        // Tests have the host read short for one file (`tests::SHORT_READS`).
        #[cfg(test)]
        let rest = match tests::SHORT_READS.load(Ordering::Relaxed) {
            source if source == request.source as usize => {
                let n = rest.len().min(SCRATCH);
                &mut rest[..n]
            }
            _ => rest,
        };
        match file.seek_read(rest, request.offset + (skip + done) as u64) {
            Ok(0) => break Err(END_OF_FILE),
            Ok(n) => done += n,
            Err(error) => break Err(from_win32(&error)),
        }
    };
    READING.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    result
}

/// The title addresses a running request's bytes are being written to, for
/// the texture upload recheck (`VICTORY_TEXTURE_RECHECK`): in flight from
/// begin to drop. Empty and inert when the switch is off.
struct StorageWrite(Vec<(usize, usize)>);

impl StorageWrite {
    /// The request writes title address `at` for `len` bytes.
    fn at(at: usize, len: usize) -> StorageWrite {
        let mut this = StorageWrite(Vec::new());
        if victory_gpu::texture_recheck_on() {
            victory_gpu::storage_write_begin(at, len);
            this.0.push((at, len));
        }
        this
    }

    /// The request writes the physical pages, which the title has mapped at
    /// these addresses (`super::xmem::shown_at`).
    fn pages(pages: &[usize], offset: usize, len: usize) -> StorageWrite {
        let mut this = StorageWrite(Vec::new());
        if victory_gpu::texture_recheck_on() {
            for (at, bytes) in super::xmem::shown_at(pages, offset, len) {
                victory_gpu::storage_write_begin(at, bytes);
                this.0.push((at, bytes));
            }
        }
        this
    }
}

impl Drop for StorageWrite {
    fn drop(&mut self) {
        for &(at, len) in &self.0 {
            victory_gpu::storage_write_end(at, len);
        }
    }
}

/// Runs `fill` on the destination's first `len` bytes; `fill` returns how
/// many it produced. GPU-visible destinations are host memory like the
/// rest (GPU address = CPU address), so an address is written in place;
/// physical pages are too when they are consecutive pages of one
/// allocation, and otherwise go through a buffer.
fn into_destination(
    request: &Request,
    len: usize,
    fill: impl FnOnce(&mut [u8]) -> Result<usize, HRESULT>,
) -> Result<(), HRESULT> {
    if request.options & PAGE_DESTINATION == 0 {
        // The churn's books (`VICTORY_TEXTURE_MADE`): the destination memory
        // this request fills, counted with the distinct ranges a minute.
        #[cfg(feature = "diagnostics")]
        victory_gpu::storage_destination((request.destination as u64, len as u64));
        // Opens the guarded blocks and tells the GPU side a placed texture
        // there changes. `fill` writes from user mode (a file read is read
        // beside first, see `execute`), so a block guarded again meanwhile
        // faults into the guard instead of failing the write.
        victory_gpu::kernel_will_write(request.destination as usize, len);
        let writing = StorageWrite::at(request.destination as usize, len);
        // SAFETY: the title's destination holds `destination_size` bytes, at
        // least `len`, and nothing else writes it until completion.
        let out = unsafe { std::slice::from_raw_parts_mut(request.destination.cast::<u8>(), len) };
        let filled = fill(out).map(drop);
        drop(writing);
        // Said again now that the bytes are there (upstream's c75fdba holds
        // the blocks open over the read and marks them once more at its
        // end): a copy or a texture pass that took the memory while the
        // read was under way took what was there before. The texture pass
        // is told in any case; the guards are opened again only with
        // `VICTORY_PAGE_WRITES=all`.
        victory_gpu::title_memory_written(request.destination as usize, len, false);
        return filled;
    }
    let offset = (request.page_offset & 0xffff) as usize;
    let pages = &filled(request.destination.cast(), (offset + len).div_ceil(PAGE))[..];
    #[cfg(feature = "diagnostics")]
    {
        // The churn's books: a page destination's range, by its first and last
        // page and its length.
        let first = pages.first().copied().unwrap_or(0) as u64;
        let last = pages.last().copied().unwrap_or(0) as u64;
        victory_gpu::storage_destination((first, last << 32 | len as u64));
    }
    // Consecutive pages of one allocation (the usual case): written in
    // place through one view.
    if let Some(mut window) = super::xmem::page_window(pages, offset, len) {
        let writing = StorageWrite::pages(pages, offset, len);
        let filled = fill(window.bytes()).map(drop);
        drop(window);
        drop(writing);
        // Pages are mapped where the card reads them and filled
        // afterwards: the GPU side's copies of those addresses learn of
        // the bytes only from this (upstream c75fdba, fork PR #571). After
        // a decode that failed part way too: what it wrote is there.
        super::xmem::pages_written(pages, offset, len);
        return filled;
    }
    let mut data = vec![0; len];
    let n = fill(&mut data)?;
    // A view the host refuses for a moment (known pages) comes right a
    // little later: the write is tried again before the request fails
    // (`VICTORY_DSTORAGE_RETRIES`, 12 tries, about 0.9 s in all; 0: not
    // tried again, as before). Runs 175-178 lost 12 to 446 reads a launch
    // here, all in one second of the loading; the one build that tried
    // again (run 165) lost none of 13. A page array not filled yet never
    // gets here (`execute` holds the request back), and an unknown page
    // fails above: the tries rewrite the same pages, and could not help.
    static RETRIES: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let retries = *RETRIES.get_or_init(|| {
        std::env::var("VICTORY_DSTORAGE_RETRIES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(12)
    });
    // Since the start: writes that succeeded on a later try, and writes
    // given up.
    static RETRIED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
    let _writing = StorageWrite::pages(pages, offset, n);
    let mut written = super::xmem::write_physical(pages, offset, &data[..n]);
    if !written && !super::xmem::pages_known(pages, offset, len) {
        // A page number we do not know: freed since it was handed in, or
        // never ours (0 is never filled: see `execute`). Numbers are not
        // given out twice, so no wait brings it back.
        let why = if pages.contains(&0) { 0 } else { 1 };
        PAGE_FAILURES[why].fetch_add(1, Ordering::Relaxed);
        tracing::warn!(
            offset,
            len,
            pages = pages.len(),
            first = format_args!("{:#x}", pages.first().copied().unwrap_or(0)),
            options = format_args!("{:#x}", request.options),
            reason = if why == 0 { "never filled" } else { "unknown page" },
            "dstorage: a destination's physical pages could not be written; its request fails"
        );
        return Err(E_INVALIDARG);
    }
    let mut tries = 0u32;
    while !written && tries < retries {
        std::thread::sleep(std::time::Duration::from_millis((2u64 << tries.min(6)).min(128)));
        tries += 1;
        written = super::xmem::write_physical(pages, offset, &data[..n]);
    }
    if written {
        if tries > 0 {
            let on_a_retry = RETRIED[0].fetch_add(1, Ordering::Relaxed) + 1;
            tracing::info!(
                offset,
                len,
                tries,
                on_a_retry,
                given_up = RETRIED[1].load(Ordering::Relaxed),
                "dstorage: a destination's physical pages written after a retry"
            );
        }
        // As above, after the write that succeeded.
        super::xmem::pages_written(pages, offset, n);
        Ok(())
    } else {
        let given_up = RETRIED[1].fetch_add(1, Ordering::Relaxed) + 1;
        tracing::warn!(
            known = super::xmem::pages_known(pages, offset, len),
            tries,
            given_up,
            on_a_retry = RETRIED[0].load(Ordering::Relaxed),
            "dstorage: retries used up (VICTORY_DSTORAGE_RETRIES)"
        );
        // Said aloud: the interface has no result to read (a caller polls
        // for completion only), so a destination left unwritten silently
        // holds whatever its pages held before.
        PAGE_FAILURES[2].fetch_add(1, Ordering::Relaxed);
        tracing::warn!(
            offset,
            len,
            pages = pages.len(),
            first = format_args!("{:#x}", pages.first().copied().unwrap_or(0)),
            options = format_args!("{:#x}", request.options),
            reason = "host failure",
            "dstorage: a destination's physical pages could not be written; its request fails"
        );
        Err(E_INVALIDARG)
    }
}

/// The page numbers of a destination. The array may be filled after the
/// request is enqueued (a copy taken then can hold zeros, or the numbers of
/// the array's last use), so it is read when the request runs; a request
/// whose array still holds a zero is held back before it runs (`execute`,
/// [`Lanes::hold_back`]), and a number that is zero here all the same (the
/// array emptied again) is waited for, a tenth of a second at most.
fn filled(array: *const usize, n: usize) -> Vec<usize> {
    static LATE: AtomicU64 = AtomicU64::new(0);
    let read = || -> Vec<usize> {
        // SAFETY: the caller passes a page number per 64 KiB of destination
        // and keeps the array until the request completes; it may write it
        // from another thread meanwhile.
        (0..n).map(|i| unsafe { array.add(i).read_volatile() }).collect()
    };
    let mut pages = read();
    if !pages.contains(&0) {
        return pages;
    }
    let began = std::time::Instant::now();
    while pages.contains(&0) && began.elapsed() < std::time::Duration::from_millis(100) {
        std::thread::sleep(std::time::Duration::from_millis(2));
        pages = read();
    }
    let late = LATE.fetch_add(1, Ordering::Relaxed) + 1;
    if late <= 8 || late.is_power_of_two() {
        tracing::warn!(
            requests = late,
            pages = n,
            waited_ms = began.elapsed().as_millis() as u64,
            filled = !pages.contains(&0),
            "dstorage: a request ran before its page array was filled"
        );
    }
    pages
}

/// Slots in the console ABI's 2020 order; a slot without a method of
/// ours is `missing` and logs its offset.
static QUEUE_VTABLE: Methods<12> = Methods([
    missing::<0> as *const (),
    add_ref_instance::<Queue> as *const (),
    release::<Queue> as *const (),
    enqueue_request as *const (), // 0x18
    enqueue_status as *const (),  // 0x20
    missing::<5> as *const (),    // 0x28 (2020: fence signal)
    submit as *const (),          // 0x30
    missing::<7> as *const (),    // 0x38 (2020: cancel by tag)
    close as *const (),           // 0x40
    missing::<9> as *const (),    // 0x48 (2020: error event)
    missing::<10> as *const (),   // 0x50 (2020: error record)
    missing::<11> as *const (),   // 0x58 (2020: queue info)
]);

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "dstorage",
        dlls: &["xcrdapi.dll"],
        exports: vec![hle_export!(DStorageGetFactory)],
    }
}

#[cfg(test)]
mod tests {
    use std::os::windows::ffi::OsStrExt;
    use std::time::{Duration, Instant};

    use super::*;
    use crate::modules::xgameruntime::method;

    fn call<F: Copy>(object: *mut c_void, slot: usize) -> F {
        // SAFETY: every object here starts with the runtime header.
        method(unsafe { &*object.cast::<Object>() }, slot)
    }

    fn factory() -> *mut c_void {
        let mut factory = std::ptr::null_mut();
        assert_eq!(DStorageGetFactory(&FACTORY, &mut factory), S_OK);
        factory
    }

    fn open(path: &std::path::Path) -> *mut c_void {
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
        let open: extern "system" fn(*mut c_void, *const u16, *const Guid, Out) -> HRESULT =
            call(factory(), 0x20);
        let mut file = std::ptr::null_mut();
        assert_eq!(
            open(factory(), wide.as_ptr(), std::ptr::null(), &mut file),
            S_OK
        );
        file
    }

    fn status_array(capacity: u32) -> *mut c_void {
        let create: extern "system" fn(
            *mut c_void,
            u32,
            *const c_char,
            *const Guid,
            Out,
        ) -> HRESULT = call(factory(), 0x28);
        let mut array = std::ptr::null_mut();
        let name = c"test".as_ptr();
        assert_eq!(
            create(factory(), capacity, name, std::ptr::null(), &mut array),
            S_OK
        );
        array
    }

    fn new_queue(source_type: u64) -> *mut c_void {
        queue_with_options(source_type, 0x80, AUTO_SUBMIT_DISABLED)
    }

    fn queue_with_options(source_type: u64, capacity: u16, options: u64) -> *mut c_void {
        let create: extern "system" fn(*mut c_void, *const QueueDesc, *const Guid, Out) -> HRESULT =
            call(factory(), 0x50);
        let desc = QueueDesc {
            source_type,
            capacity,
            priority: 0,
            name: c"test".as_ptr(),
            options,
        };
        let mut queue = std::ptr::null_mut();
        assert_eq!(create(factory(), &desc, std::ptr::null(), &mut queue), S_OK);
        queue
    }

    fn request(
        options: u64,
        destination: *mut c_void,
        destination_size: usize,
        source: *const c_void,
        offset: u64,
        source_size: usize,
    ) -> Request {
        Request {
            options,
            destination,
            page_offset: 0,
            destination_size: destination_size as u32,
            source,
            offset,
            source_size: source_size as u32,
            intermediate_size: 0,
            tag: 0,
            name: std::ptr::null(),
        }
    }

    fn enqueue(queue: *mut c_void, request: &Request) {
        let f: extern "system" fn(*mut c_void, *const Request) = call(queue, 0x18);
        f(queue, request);
    }

    fn enqueue_status(queue: *mut c_void, array: *mut c_void, index: u32) {
        let f: extern "system" fn(*mut c_void, *mut c_void, u32) = call(queue, 0x20);
        f(queue, array, index);
    }

    fn void(object: *mut c_void, slot: usize) {
        let f: extern "system" fn(*mut c_void) = call(object, slot);
        f(object);
    }

    fn result(array: *mut c_void, index: u32) -> HRESULT {
        let f: extern "system" fn(*mut c_void, u32) -> HRESULT = call(array, 0x20);
        f(array, index)
    }

    /// Polls until the entry completes, then returns its result.
    fn wait(array: *mut c_void, index: u32) -> HRESULT {
        let done: extern "system" fn(*mut c_void, u32) -> bool = call(array, 0x18);
        let start = Instant::now();
        while !done(array, index) {
            assert!(start.elapsed() < Duration::from_secs(10), "never completed");
            std::thread::sleep(Duration::from_millis(1));
        }
        result(array, index)
    }

    fn file_with(bytes: &[u8]) -> (tempfile::TempDir, *mut c_void) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        std::fs::write(&path, bytes).unwrap();
        let file = open(&path);
        (dir, file)
    }

    fn pattern(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i * 7 + i / 251) as u8).collect()
    }

    /// The largest scratch a raw read into a title address has used.
    pub(super) static BESIDE_MOST: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);
    thread_local! {
        /// Raw reads on this thread find no room for what they read beside,
        /// as on a host short of memory.
        pub(super) static NO_ROOM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    /// Bytes of raw reads' own buffers held now, and the most held at once.
    static STAGED: [std::sync::atomic::AtomicUsize; 2] = [const { std::sync::atomic::AtomicUsize::new(0) }; 2];

    /// A raw read's own buffer, counted while it is held.
    pub(super) struct Staged(usize);

    impl Staged {
        pub(super) fn held(bytes: usize) -> Staged {
            let now = STAGED[0].fetch_add(bytes, Ordering::SeqCst) + bytes;
            STAGED[1].fetch_max(now, Ordering::SeqCst);
            Staged(bytes)
        }
    }

    impl Drop for Staged {
        fn drop(&mut self) {
            STAGED[0].fetch_sub(self.0, Ordering::SeqCst);
        }
    }

    /// Raw reads into title addresses, larger than a worker's scratch and
    /// more of them than the staging budget holds, all enqueued at once:
    /// each lands whole; none uses more scratch than a worker's, and what
    /// their own buffers hold at once stays within the budget.
    #[test]
    fn raw_reads_larger_than_the_scratch_stay_within_the_staging_budget() {
        const SIZE: usize = 3 * SCRATCH + 12_345;
        const READS: usize = 12;
        let contents = pattern(READS * SIZE);
        let (_dir, file) = file_with(&contents);
        let (queue, array) = (new_queue(0), status_array(1));
        let mut out: Vec<Vec<u8>> = (0..READS).map(|_| vec![0u8; SIZE]).collect();
        for (i, o) in out.iter_mut().enumerate() {
            let at = (i * SIZE) as u64;
            enqueue(queue, &request(0, o.as_mut_ptr().cast(), SIZE, file, at, SIZE));
        }
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        for (i, o) in out.iter().enumerate() {
            assert!(o[..] == contents[i * SIZE..(i + 1) * SIZE], "read {i}");
        }
        let most = BESIDE_MOST.load(Ordering::Relaxed);
        assert!(most <= SCRATCH, "a raw read used {most} bytes of scratch");
        let staged = STAGED[1].load(Ordering::SeqCst);
        assert!(staged > 0 && staged as u64 <= STAGING, "raw reads held {staged} bytes of their own at once");
        void(queue, 0x40);
    }

    /// The walls trace's level while a test sets one (0: as the switch says).
    pub(super) static WALLS_LEVEL: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
    /// Held by a test, the walls trace's output stalls (as a sink that does
    /// not take its lines) for the reads of files named `walls-stall`.
    pub(super) static WALLS_STALL: Mutex<()> = Mutex::new(());

    /// The threads that unpacked BCPack data: name, priority while
    /// unpacking, priority after.
    static BCPACK_PRIORITIES: Mutex<Vec<(String, i32, i32)>> = Mutex::new(Vec::new());

    thread_local! {
        static UNPACKING: std::cell::Cell<i32> = const { std::cell::Cell::new(i32::MIN) };
    }

    fn priority() -> i32 {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadPriority};
        // SAFETY: a plain query of the calling thread.
        unsafe { GetThreadPriority(GetCurrentThread()) }
    }

    /// A test's choice for [`bcpack_serial`]: the unpack runs on a worker, so a thread-local
    /// would not reach it. One test's choice at a time ([`bcpack_serial_for_test`]) — -1
    /// none, 0 parallel, 1 serial.
    static SERIAL_CHOICE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

    /// Serializes the tests that override [`bcpack_serial`].
    static SERIAL_TESTS: Mutex<()> = Mutex::new(());

    /// The choice [`bcpack_serial`] reads: `Some(serial)` while a test holds one.
    pub(super) fn bcpack_serial_choice() -> Option<bool> {
        match SERIAL_CHOICE.load(std::sync::atomic::Ordering::Relaxed) {
            -1 => None,
            0 => Some(false),
            _ => Some(true),
        }
    }

    /// Makes [`bcpack_serial`] read `serial` on every thread until the guard drops; the guard
    /// keeps another test's choice out meanwhile.
    fn bcpack_serial_for_test(serial: bool) -> SerialChoice {
        let held = SERIAL_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        SERIAL_CHOICE.store(i8::from(serial), std::sync::atomic::Ordering::Relaxed);
        SerialChoice(held)
    }

    struct SerialChoice(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

    impl Drop for SerialChoice {
        fn drop(&mut self) {
            SERIAL_CHOICE.store(-1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// `unpack_bcpack` on a worker: its priority while it unpacks (`during`)
    /// and once it is done.
    pub(super) fn note_bcpack_priority(during: bool) {
        if during {
            UNPACKING.with(|p| p.set(priority()));
            return;
        }
        let name = std::thread::current().name().unwrap_or("").to_string();
        let during = UNPACKING.with(|p| p.get());
        BCPACK_PRIORITIES.lock().unwrap().push((name, during, priority()));
    }

    /// Where the walls trace writes a line.
    pub(super) fn walls_sink(file: &str) {
        if file.contains("walls-stall") {
            drop(WALLS_STALL.lock());
        }
    }

    /// Run by the next test in a child process (statics of its own); does
    /// nothing otherwise.
    #[test]
    fn walls_child() {
        if std::env::var_os("VICTORY_WALLS_CHILD").is_none() {
            return;
        }
        let (offset, source_size, destination_size, options, ms, hr) = (0, 0, 0, 0, 0, 0);
        let line = WallsLine { file: "child".into(), offset, source_size, destination_size, options, ms, hr };
        walls_send(line);
        assert!(WALLS_OUT.get().is_none(), "a worker's hand-off started the trace's thread");
        assert!(scheduler().is_some());
        assert!(WALLS_OUT.get().is_some_and(Option::is_some), "the trace's thread starts with the workers");
        println!("FINISHED");
    }

    /// The walls trace's thread starts with the workers, before any of them
    /// hands it a line; a hand-off only tries the queue.
    #[test]
    fn the_walls_trace_starts_with_the_workers() {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "modules::dstorage::tests::walls_child", "--nocapture", "--test-threads=1"])
            .env("VICTORY_WALLS_CHILD", "1")
            .env("VICTORY_WALLS_TRACE", "2")
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success() && text.contains("FINISHED"), "{text}{}", String::from_utf8_lossy(&out.stderr));
    }

    /// The walls trace's output stalls: a read's completion is published all
    /// the same, its status written and its queue free to close.
    #[test]
    fn a_stalled_walls_trace_does_not_hold_completion() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("walls-stall.bin");
        std::fs::write(&path, pattern(4096)).unwrap();
        // On before the file opens: the trace notes the files opened while on.
        WALLS_LEVEL.store(2, Ordering::Relaxed);
        let file = open(&path);
        let (queue, array) = (new_queue(0), status_array(1));
        let mut out = vec![0u8; 4096];
        let stalled = WALLS_STALL.lock().unwrap_or_else(|e| e.into_inner());
        enqueue(queue, &request(0, out.as_mut_ptr().cast(), 4096, file, 0, 4096));
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        let done: extern "system" fn(*mut c_void, u32) -> bool = call(array, 0x18);
        let start = Instant::now();
        while !done(array, 0) && start.elapsed() < Duration::from_secs(2) {
            std::thread::sleep(Duration::from_millis(1));
        }
        let published = done(array, 0);
        drop(stalled);
        WALLS_LEVEL.store(0, Ordering::Relaxed);
        assert!(published, "the read's completion waited for the trace's output");
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, pattern(4096));
        void(queue, 0x40);
    }

    /// A raw read whose file range runs past the file's end, after more than
    /// a piece of it is there, or whose end does not fit in an offset: it
    /// fails before anything of it is copied in.
    #[test]
    fn a_raw_read_past_the_end_fails_before_writing() {
        let contents = pattern(SCRATCH + 100);
        let (_dir, file) = file_with(&contents);
        let (queue, array) = (new_queue(0), status_array(2));
        let mut past_end = vec![0xaau8; 2 * SCRATCH];
        let mut wraps = vec![0xaau8; 100];
        let to = |out: &mut Vec<u8>| out.as_mut_ptr().cast();
        enqueue(queue, &request(0, to(&mut past_end), 2 * SCRATCH, file, 0, 2 * SCRATCH));
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), END_OF_FILE);
        assert!(past_end.iter().all(|&b| b == 0xaa), "nothing copied in");
        enqueue(queue, &request(0, to(&mut wraps), 100, file, u64::MAX - 10, 100));
        enqueue_status(queue, array, 1);
        void(queue, 0x30);
        assert_ne!(wait(array, 1), S_OK);
        assert!(wraps.iter().all(|&b| b == 0xaa), "nothing copied in");
        void(queue, 0x40);
    }

    /// A raw read that fails after more than a scratch of it was read (its
    /// second half locked by another handle): the destination is as it was,
    /// for a title that polls for completion only sees a failed read done.
    #[test]
    fn a_raw_read_that_fails_part_way_writes_nothing() {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::{LockFile, UnlockFile};
        const SIZE: usize = 2 * SCRATCH;
        let contents = pattern(SIZE);
        let (dir, file) = file_with(&contents);
        let other = std::fs::File::open(dir.path().join("data.bin")).unwrap();
        let (from, locked) = ((SCRATCH + 4096) as u32, (SCRATCH - 4096) as u32);
        // SAFETY: our own handle; a lock on bytes of the file.
        assert_ne!(unsafe { LockFile(other.as_raw_handle(), from, 0, locked, 0) }, 0);
        let (queue, array) = (new_queue(0), status_array(1));
        let mut out = vec![0xaau8; SIZE];
        enqueue(queue, &request(0, out.as_mut_ptr().cast(), SIZE, file, 0, SIZE));
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        let hr = wait(array, 0);
        // SAFETY: as above.
        unsafe { UnlockFile(other.as_raw_handle(), from, 0, locked, 0) };
        assert_eq!(hr, 0x8007_0021_u32 as i32, "ERROR_LOCK_VIOLATION");
        assert!(out.iter().all(|&b| b == 0xaa), "nothing copied in");
        void(queue, 0x40);
    }

    /// With no room for what it reads beside, a raw read fails before it
    /// writes anything: one within a scratch, and one larger.
    #[test]
    fn a_raw_read_without_room_beside_fails_before_writing() {
        const SIZE: usize = 2 * SCRATCH + 777;
        let contents = pattern(SIZE + 4096);
        let (_dir, file) = file_with(&contents);
        let file = file as usize;
        // A thread with no scratch yet, as a fresh worker: no room is found.
        let reads = std::thread::spawn(move || {
            NO_ROOM.set(true);
            [100, SIZE].map(|len| {
                let mut out = vec![0xaau8; len];
                let to = out.as_mut_ptr().cast();
                let r = request(0, to, len, file as *const c_void, 4096, len);
                (read_beside(&r, &mut out), out.iter().all(|&b| b == 0xaa))
            })
        })
        .join()
        .unwrap();
        assert_eq!(reads, [(Err(E_OUTOFMEMORY), true), (Err(E_OUTOFMEMORY), true)]);
    }

    #[test]
    fn a_page_array_filled_late_is_waited_for() {
        let array: &'static [std::sync::atomic::AtomicUsize] = Box::leak(Box::new([std::sync::atomic::AtomicUsize::new(7), std::sync::atomic::AtomicUsize::new(0)]));
        let title = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            array[1].store(8, Ordering::Release);
        });
        assert_eq!(filled(array.as_ptr().cast(), 2), [7, 8]);
        title.join().unwrap();
        // One never filled is given up on.
        array[0].store(0, Ordering::Release);
        assert_eq!(filled(array.as_ptr().cast(), 2), [0, 8]);
    }

    #[test]
    fn request_layout_is_the_titles() {
        assert_eq!(size_of::<Request>(), 0x48);
        assert_eq!(std::mem::offset_of!(Request, destination_size), 0x18);
        assert_eq!(std::mem::offset_of!(Request, source), 0x20);
        assert_eq!(std::mem::offset_of!(Request, source_size), 0x30);
        assert_eq!(std::mem::offset_of!(Request, name), 0x40);
        assert_eq!(std::mem::offset_of!(QueueDesc, options), 0x18);
    }

    fn compress_zlib(plain: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; zlib_rs::compress_bound(plain.len())];
        let (packed, rc) =
            zlib_rs::compress_slice(&mut buf, plain, zlib_rs::DeflateConfig::new(6));
        assert_eq!(rc, zlib_rs::ReturnCode::Ok);
        packed.to_vec()
    }

    #[test]
    fn raw_and_zlib_reads_from_a_file() {
        let plain = pattern(300_000);
        let packed = compress_zlib(&plain);
        let mut contents = plain.clone();
        contents.extend_from_slice(&packed);
        let (_dir, file) = file_with(&contents);
        let (queue, array) = (new_queue(0), status_array(1));

        let mut raw = vec![0u8; 1000];
        let mut inflated = vec![0u8; plain.len()];
        let at = |b: &mut Vec<u8>| b.as_mut_ptr().cast();
        enqueue(queue, &request(0, at(&mut raw), 1000, file, 5000, 1000));
        let offset = plain.len() as u64;
        let zlib = request(
            ZLIB,
            at(&mut inflated),
            plain.len(),
            file,
            offset,
            packed.len(),
        );
        enqueue(queue, &zlib);
        enqueue_status(queue, array, 0);
        assert_eq!(result(array, 0), E_PENDING, "pending until done");
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(raw, plain[5000..6000]);
        assert_eq!(inflated, plain);
        void(queue, 0x40);
    }

    #[test]
    fn memory_sources_inflate_too() {
        let plain = pattern(70_000);
        let packed = compress_zlib(&plain);
        let (queue, array) = (new_queue(1), status_array(1));
        let mut out = vec![0u8; plain.len()];
        let options = ZLIB | MEMORY_SOURCE;
        let r = request(
            options,
            out.as_mut_ptr().cast(),
            out.len(),
            packed.as_ptr().cast(),
            0,
            packed.len(),
        );
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, plain);
    }

    #[test]
    fn statuses_complete_in_order_with_the_first_failure() {
        let contents = pattern(4096);
        let (_dir, file) = file_with(&contents);
        let (queue, array) = (new_queue(0), status_array(3));
        // Requests in flight together run in any order, so none of them
        // shares a destination (failed ones leave theirs undefined).
        let mut buffers = [[0u8; 16]; 5];
        let [failed, failed_too, out, other, bcpack] = buffers.each_mut().map(|b| b.as_mut_ptr().cast());

        // Segment 0: a read past the end, then a bad stream; the first wins.
        enqueue(queue, &request(0, failed, 16, file, 4090, 16));
        enqueue(queue, &request(ZLIB, failed_too, 16, file, 0, 16));
        enqueue_status(queue, array, 0);
        // Segment 1: two reads.
        enqueue(queue, &request(0, other, 16, file, 0, 16));
        enqueue(queue, &request(0, out, 16, file, 100, 16));
        enqueue_status(queue, array, 1);
        void(queue, 0x30);
        // Segment 2: a BCPack request (BC7) whose stream is empty fails.
        with_test_tables();
        enqueue(queue, &request(7 << 1, bcpack, 16, file, 0, 0));
        enqueue_status(queue, array, 2);
        void(queue, 0x30);

        assert_eq!(wait(array, 2), INVALID_DATA);
        assert_eq!(result(array, 0), END_OF_FILE);
        assert_eq!(result(array, 1), S_OK);
        assert_eq!(buffers[2], contents[100..116]);
        assert_eq!(buffers[3], contents[..16]);
    }

    #[test]
    fn half_a_queue_submits_itself() {
        let (_dir, file) = file_with(&pattern(64));
        let create: extern "system" fn(*mut c_void, *const QueueDesc, *const Guid, Out) -> HRESULT =
            call(factory(), 0x50);
        let desc = QueueDesc {
            source_type: 0,
            capacity: 4,
            priority: 1,
            name: c"small".as_ptr(),
            options: 0, // auto-submit on: the test checks it
        };
        let mut queue = std::ptr::null_mut();
        assert_eq!(create(factory(), &desc, std::ptr::null(), &mut queue), S_OK);
        let array = status_array(1);
        let mut out = [0u8; 16];
        enqueue(queue, &request(0, out.as_mut_ptr().cast(), 16, file, 8, 16));
        enqueue_status(queue, array, 0); // the second of four: no submit needed
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, pattern(64)[8..24]);
    }

    /// Lanes without workers: what `start` picks, and when statuses land.
    fn lanes(staging: u64) -> Lanes {
        Lanes {
            lanes: HashMap::new(),
            next_id: 0,
            next_seq: 0,
            staged: 0,
            staging,
            starved: false,
            clock: [0; 3],
            now: 0,
            lane_now: [0; 3],
            done: Default::default(),
            refused: 0,
            bcpack: Default::default(),
            since: None,
            held_back: 0,
            held_said: None,
        }
    }

    fn read(lanes: &mut Lanes, id: u64, options: u64, size: u32) {
        let mut r = request(options, std::ptr::null_mut(), 0, std::ptr::null(), 0, 0);
        (r.destination_size, r.source_size) = (size, size);
        let lane = lanes.lanes.get_mut(&id).unwrap();
        lane.unsubmitted.push(Command::Read(Read::new(r)));
    }

    #[test]
    fn priorities_share_by_bytes_about_tenfold() {
        let mut l = lanes(STAGING);
        l.lanes.insert(0, Lane::new(0, 4096));
        l.lanes.insert(1, Lane::new(1, 4096));
        l.lanes.insert(2, Lane::new(2, 4096)); // real-time: as high
        for _ in 0..400 {
            for id in 0..3 {
                read(&mut l, id, 0, 4096);
            }
        }
        (0..3).for_each(|id| l.submit(id));
        let mut started = [0u32; 3];
        for _ in 0..330 {
            let job = l.start().unwrap();
            started[job.lane as usize] += 1;
            l.finish(&job, Ok(()));
        }
        // Normal gets about a tenth of what high and real-time get together.
        assert_eq!(started[1] + started[2], 300, "{started:?}");
        assert_eq!(started[0], 30, "{started:?}");
        assert!(started[1].abs_diff(started[2]) <= 1, "{started:?}");
    }

    /// The stall line's depth counts a read once whatever it is doing:
    /// waiting, running, held back for its page array, put back and running
    /// again (Sol's review of the run diagnostics: its held entry, kept
    /// without its request, was counted again).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_held_read_counts_once_in_the_depth() {
        let mut l = lanes(1 << 20);
        l.lanes.insert(0, Lane::new(0, 64));
        read(&mut l, 0, 0, 4096);
        l.submit(0);
        assert_eq!(l.depth(), 1, "waiting");
        let job = l.start().unwrap();
        assert_eq!(l.depth(), 1, "running");
        l.hold_back(job);
        assert_eq!(l.depth(), 1, "held back");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let again = l.start().expect("due again");
        assert_eq!(l.depth(), 1, "running again, its held entry kept without its request");
        l.finish(&again, Ok(()));
        assert_eq!(l.depth(), 0);
    }

    #[test]
    fn staging_bounds_compressed_bytes_in_flight() {
        let mut l = lanes(100);
        l.lanes.insert(0, Lane::new(0, 64));
        read(&mut l, 0, ZLIB, 60);
        read(&mut l, 0, ZLIB, 60);
        read(&mut l, 0, 0, 1000); // raw: straight to its destination
        l.submit(0);
        let first = l.start().unwrap();
        assert!(l.start().is_none(), "60 + 60 > 100");
        l.finish(&first, Ok(()));
        let second = l.start().unwrap();
        let raw = l.start().unwrap();
        assert_eq!((second.staging, raw.staging), (60, 0));
        // Alone, a request larger than the buffer still runs.
        read(&mut l, 0, ZLIB, 500);
        l.submit(0);
        assert!(l.start().is_none());
        l.finish(&second, Ok(()));
        assert!(l.start().is_some());
    }

    #[test]
    fn a_status_waits_for_every_earlier_read() {
        let mut l = lanes(STAGING);
        l.lanes.insert(0, Lane::new(0, 64));
        let array = status_array(2).cast::<Instance<StatusArray>>();
        let status = |index| {
            add_ref_instance(array);
            Command::Status(Status { array, index })
        };
        read(&mut l, 0, 0, 16);
        read(&mut l, 0, 0, 16);
        l.lanes.get_mut(&0).unwrap().unsubmitted.push(status(0));
        read(&mut l, 0, 0, 16);
        l.lanes.get_mut(&0).unwrap().unsubmitted.push(status(1));
        // SAFETY: an array we created.
        for entry in unsafe { &(*array).state.entries }.iter() {
            entry.store(E_PENDING, Ordering::Release);
        }
        l.submit(0);
        let jobs: Vec<Job> = (0..3).map(|_| l.start().unwrap()).collect();
        // The last read and the first finish; the second then fails.
        l.finish(&jobs[2], Ok(()));
        l.finish(&jobs[0], Ok(()));
        let results = || (result(array.cast(), 0), result(array.cast(), 1));
        assert_eq!(results(), (E_PENDING, E_PENDING));
        l.finish(&jobs[1], Err(INVALID_DATA));
        assert_eq!(results(), (INVALID_DATA, S_OK));
    }

    #[test]
    fn zlib_then_bcpack() {
        with_test_tables();
        let packed = compress_zlib(&BC4_STREAM);
        let (_dir, file) = file_with(&packed);
        let (queue, array) = (new_queue(0), status_array(1));
        let mut out = [0u8; 16];
        let options = ZLIB | 4 << 1;
        let mut r = request(options, out.as_mut_ptr().cast(), 16, file, 0, packed.len());
        r.intermediate_size = BC4_STREAM.len() as u32;
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, bc4_blocks());
    }

    /// BCPack tables we made (uniform models; BC4 index pairs: low 3 bits,
    /// next 3 bits), in a folder kept for the test process.
    fn with_test_tables() {
        static DIR: OnceLock<()> = OnceLock::new();
        DIR.get_or_init(|| {
            let mut t = Vec::new();
            for m in 0..128 {
                let g = m / 16;
                for b in 0..9 {
                    let f: u16 = if b < g + 2 { 1 << (15 - g) } else { 0 };
                    t.extend_from_slice(&f.to_le_bytes());
                }
            }
            t.extend([0; 256 + 64 + 256]);
            t.extend((0..64u8).map(|v| (v & 7) | (v >> 3) << 4));
            t.resize(bcpack::TABLES_SIZE, 0);
            let dir = tempfile::tempdir().unwrap().keep();
            std::fs::write(dir.join("tables.bin"), t).unwrap();
            use_bcpack_tables(dir);
        });
    }

    /// Two BC4 blocks from the format's test encoder: endpoints 10, -3 with
    /// indices p -> p & 7, then the same block plus 1 everywhere.
    const BC4_STREAM: [u8; 88] = [
        0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 4, 4, 4, 4, 1, 1, 1, 1, 0x10, 0x10, 0x10, 0x10, 2, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0x14, 5, 0x20, 0x68, 0xb0, 0xf8, 3, 3, 2, 2, 0x24, 0x24, 0x24,
        0x24, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0x10, 0x10, 0x10, 0x10, 0, 0, 0, 0,
        0x20, 0x68, 0xb0, 0xf8, 0x24, 0x24, 0x24, 0x24, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    /// What `BC4_STREAM` unpacks to.
    fn bc4_blocks() -> [u8; 16] {
        let code = [0u64, 2, 3, 4, 5, 6, 7, 1];
        let block = |e0: u64, e1: u64, add: usize| {
            let bits = (0..16).fold(0, |a, p| a | code[(p + add) & 7] << (16 + 3 * p));
            (e0 | e1 << 8 | bits).to_le_bytes()
        };
        let mut out = [0; 16];
        out[..8].copy_from_slice(&block(138, 125, 0));
        out[8..].copy_from_slice(&block(139, 126, 1));
        out
    }

    /// The check path hands on the serial decode's bytes (the other two
    /// results are only compared).
    #[test]
    fn the_check_path_hands_on_the_serial_bytes() {
        with_test_tables();
        let tables = bcpack_tables().unwrap();
        let mut out = [0u8; 16];
        unpack_bcpack_checked(4, &BC4_STREAM, &mut out, tables).unwrap();
        assert_eq!(out, bc4_blocks());
    }

    #[test]
    fn bcpack_requests_unpack_into_the_destination() {
        with_test_tables();
        let mut contents = pattern(100);
        contents.extend_from_slice(&BC4_STREAM);
        let (_dir, file) = file_with(&contents);
        let (queue, array) = (new_queue(0), status_array(1));
        let mut out = [0u8; 16];
        enqueue(queue, &request(4 << 1, out.as_mut_ptr().cast(), 16, file, 100, 88));
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, bc4_blocks());
    }

    /// The worker's guard lowers a thread only when its priority could be
    /// read, and gives that priority back: after a failed query
    /// (`THREAD_PRIORITY_ERROR_RETURN`) there is nothing to go back to, and
    /// the thread is left as it is (Sol's review of round one).
    #[test]
    fn a_failed_priority_query_leaves_the_thread_alone() {
        use windows_sys::Win32::System::Threading::{THREAD_PRIORITY_BELOW_NORMAL, THREAD_PRIORITY_NORMAL};
        std::thread::spawn(|| {
            assert_eq!(priority(), THREAD_PRIORITY_NORMAL);
            {
                let _guard = BelowNormal::after(i32::MAX);
                assert_eq!(priority(), THREAD_PRIORITY_NORMAL, "lowered with no priority to go back to");
            }
            assert_eq!(priority(), THREAD_PRIORITY_NORMAL);
            {
                let _below = BelowNormal::now();
                assert_eq!(priority(), THREAD_PRIORITY_BELOW_NORMAL);
            }
            assert_eq!(priority(), THREAD_PRIORITY_NORMAL);
        })
        .join()
        .unwrap();
    }

    /// A DirectStorage worker that unpacks a BCPack request runs below normal
    /// priority while it does (the title's threads run at normal) and is back
    /// at normal once it is done.
    #[test]
    fn a_worker_unpacks_below_normal_and_gets_its_priority_back() {
        use windows_sys::Win32::System::Threading::{THREAD_PRIORITY_BELOW_NORMAL, THREAD_PRIORITY_NORMAL};
        // The parallel unpack is the one that lowers a worker's priority; the serial one is
        // the default, so the test selects the parallel path for itself.
        let _parallel = bcpack_serial_for_test(false);
        with_test_tables();
        let (_dir, file) = file_with(&BC4_STREAM);
        let (queue, array) = (new_queue(0), status_array(1));
        let mut out = [0u8; 16];
        enqueue(queue, &request(4 << 1, out.as_mut_ptr().cast(), 16, file, 0, 88));
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, bc4_blocks());
        let seen = BCPACK_PRIORITIES.lock().unwrap().clone();
        assert!(seen.iter().any(|(name, ..)| name.starts_with("dstorage ")), "a worker unpacked: {seen:?}");
        for (name, during, after) in &seen {
            assert!(name.starts_with("dstorage "), "unpacked on {name}");
            assert_eq!((*during, *after), (THREAD_PRIORITY_BELOW_NORMAL, THREAD_PRIORITY_NORMAL), "{name}");
        }
    }

    /// The helpers that pack BCPack blocks run below normal priority: work
    /// handed to the runtime's pool is run by them (and by the thread that
    /// hands it out), and every piece a helper ran saw below normal.
    #[test]
    fn the_bcpack_helpers_run_below_normal() {
        use windows_sys::Win32::System::Threading::THREAD_PRIORITY_BELOW_NORMAL;
        let helpers = bcpack_helpers().expect("BCPack helpers");
        assert!((1..=4).contains(&helpers.threads()));
        let seen = std::sync::Arc::new(Mutex::new(Vec::new()));
        let pieces = seen.clone();
        helpers.run(64, move |_| {
            let name = std::thread::current().name().unwrap_or("").to_string();
            pieces.lock().unwrap().push((name, priority()));
            std::thread::sleep(Duration::from_millis(2));
        });
        let seen = seen.lock().unwrap();
        let helped: Vec<_> = seen.iter().filter(|(name, _)| name.starts_with("dstorage bcpack")).collect();
        assert!(!helped.is_empty(), "no helper ran a piece: {seen:?}");
        assert!(helped.iter().all(|(_, p)| *p == THREAD_PRIORITY_BELOW_NORMAL), "{helped:?}");
    }

    #[test]
    fn a_closed_queue_completes_nothing() {
        let (_dir, file) = file_with(&pattern(64));
        let (queue, array) = (new_queue(0), status_array(1));
        let mut out = [0u8; 16];
        enqueue(queue, &request(0, out.as_mut_ptr().cast(), 16, file, 0, 16));
        enqueue_status(queue, array, 0);
        void(queue, 0x40);
        void(queue, 0x30);
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(result(array, 0), E_PENDING);
        assert_eq!(out, [0; 16]);
    }

    #[test]
    fn physical_page_destinations() {
        use crate::modules::xmem;
        let mut pages = [0usize; 2];
        let mut n = 2;
        assert_eq!(
            xmem::XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()),
            1
        );
        // Where the title would see them: a reservation of 64 KiB pages.
        let mapped = xmem::XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            0x2000 | 0x2040_0000, // reserve, 64 KiB pages
            0,
            0x04,
        );
        assert!(!mapped.is_null());
        assert_eq!(
            xmem::XMemMapPhysicalPages(mapped, 2, pages.as_ptr()),
            mapped
        );

        let plain = pattern(100);
        let (queue, array) = (new_queue(1), status_array(1));
        let mut r = request(
            PAGE_DESTINATION | MEMORY_SOURCE,
            pages.as_mut_ptr().cast(),
            plain.len(),
            plain.as_ptr().cast(),
            0,
            plain.len(),
        );
        r.page_offset = (PAGE - 40) as u64; // across the page boundary
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        // SAFETY: both pages are mapped read-write.
        let seen = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>().add(PAGE - 40), 100) };
        assert_eq!(seen, plain);
    }
    /// The file object whose reads the host returns short, a scratch's worth
    /// at a time (0: none).
    pub(super) static SHORT_READS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    /// A raw read into consecutive physical pages (written in place through
    /// one view) that fails after the host read part of it (a short read,
    /// then the rest locked by another handle): the pages are as they were,
    /// as for an address.
    #[test]
    fn a_raw_read_into_pages_that_fails_part_way_writes_nothing() {
        use crate::modules::xmem;
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::{LockFile, UnlockFile};
        const SIZE: usize = 2 * SCRATCH;
        let n_pages = SIZE / PAGE;
        let (mut pages, mut n) = (vec![0usize; n_pages], n_pages);
        assert_eq!(xmem::XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let mapped = xmem::XMemVirtualAlloc(std::ptr::null_mut(), SIZE, 0x2000 | 0x2040_0000, 0, 0x04);
        assert!(!mapped.is_null());
        assert_eq!(xmem::XMemMapPhysicalPages(mapped, n_pages, pages.as_ptr()), mapped);
        // SAFETY: the pages are mapped read-write there.
        let seen = unsafe { std::slice::from_raw_parts_mut(mapped.cast::<u8>(), SIZE) };
        seen.fill(0xaa);
        let contents = pattern(SIZE);
        let (dir, file) = file_with(&contents);
        let other = std::fs::File::open(dir.path().join("data.bin")).unwrap();
        let (from, locked) = ((SCRATCH + 4096) as u32, (SCRATCH - 4096) as u32);
        // SAFETY: our own handle; a lock on bytes of the file.
        assert_ne!(unsafe { LockFile(other.as_raw_handle(), from, 0, locked, 0) }, 0);
        let (queue, array) = (new_queue(0), status_array(1));
        let r = request(PAGE_DESTINATION, pages.as_mut_ptr().cast(), SIZE, file, 0, SIZE);
        SHORT_READS.store(file as usize, Ordering::Relaxed);
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        let hr = wait(array, 0);
        SHORT_READS.store(0, Ordering::Relaxed);
        // SAFETY: as above.
        unsafe { UnlockFile(other.as_raw_handle(), from, 0, locked, 0) };
        assert_eq!(hr, 0x8007_0021_u32 as i32, "ERROR_LOCK_VIOLATION");
        assert!(seen.iter().all(|&b| b == 0xaa), "nothing written into the pages");
        void(queue, 0x40);
    }

    /// A request into a physical page enqueued with its page array not
    /// filled yet, filled two seconds later by another thread (as the title
    /// on a slow machine does): it waits for the array and completes with
    /// the right bytes. A request enqueued after it runs meanwhile, and the
    /// status entry after both waits for both.
    #[test]
    fn a_page_array_filled_seconds_late_is_waited_for() {
        use crate::modules::xmem;
        let (mut pages, mut n) = ([0usize; 1], 1);
        assert_eq!(xmem::XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let mapped = xmem::XMemVirtualAlloc(std::ptr::null_mut(), PAGE, 0x2000 | 0x2040_0000, 0, 0x04);
        assert!(!mapped.is_null());
        assert_eq!(xmem::XMemMapPhysicalPages(mapped, 1, pages.as_ptr()), mapped);
        let plain = pattern(1000);
        let (queue, array) = (new_queue(1), status_array(1));
        // The array the request names: empty until the other thread fills it.
        let late = Box::new(std::sync::atomic::AtomicUsize::new(0));
        let source = plain.as_ptr().cast();
        enqueue(queue, &request(PAGE_DESTINATION | MEMORY_SOURCE, late.as_ptr().cast(), 1000, source, 0, 1000));
        let next = pattern(100).into_iter().rev().collect::<Vec<u8>>();
        let mut after = vec![0u8; 100];
        enqueue(queue, &request(MEMORY_SOURCE, after.as_mut_ptr().cast(), 100, next.as_ptr().cast(), 0, 100));
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        let began = Instant::now();
        // SAFETY: the later request's destination, written by a worker.
        let landed = || (0..100).all(|i| unsafe { after.as_ptr().add(i).read_volatile() } == next[i]);
        while !landed() && began.elapsed() < Duration::from_millis(1500) {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(landed(), "the request after it waited");
        assert_eq!(result(array, 0), E_PENDING, "the status waits for both");
        let page = pages[0];
        let filler = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(2).saturating_sub(began.elapsed()));
            late.store(page, Ordering::SeqCst);
            late
        });
        assert_eq!(wait(array, 0), S_OK);
        // SAFETY: the page is mapped read-write there.
        let seen = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), 1000) };
        assert_eq!(seen, plain);
        drop(filler.join());
        void(queue, 0x40);
    }

    /// A request into a physical page the title freed before it ran: no
    /// wait brings it back, so it fails, as an unknown page.
    #[test]
    fn a_request_into_a_freed_page_fails_as_an_unknown_page() {
        use crate::modules::xmem;
        let (mut pages, mut n) = ([0usize; 1], 1);
        assert_eq!(xmem::XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(xmem::XMemFreePhysicalPages(1, pages.as_ptr()), 1);
        let plain = pattern(100);
        let (queue, array) = (new_queue(1), status_array(1));
        let unknown = PAGE_FAILURES[1].load(Ordering::SeqCst);
        let to = pages.as_mut_ptr().cast();
        enqueue(queue, &request(PAGE_DESTINATION | MEMORY_SOURCE, to, 100, plain.as_ptr().cast(), 0, 100));
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), E_INVALIDARG);
        assert!(PAGE_FAILURES[1].load(Ordering::SeqCst) > unknown, "counted as an unknown page");
        void(queue, 0x40);
    }

    #[test]
    fn zlib_into_pages_in_place_and_across_allocations() {
        use crate::modules::xmem;
        let allocate = |n: usize| {
            let (mut pages, mut given) = (vec![0usize; n], n);
            assert_eq!(
                xmem::XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
                1
            );
            pages
        };
        let (a, b) = (allocate(3), allocate(1));
        let plain = pattern(2 * PAGE + 100);
        let packed = compress_zlib(&plain);
        // One allocation's pages in order (one window), then pages of two
        // (through a buffer), both from 50 bytes into the first.
        let mut destinations = [a.clone(), vec![a[0], a[1], b[0]]];
        let (queue, array) = (new_queue(1), status_array(1));
        for pages in &mut destinations {
            let mut r = request(
                ZLIB | PAGE_DESTINATION | MEMORY_SOURCE,
                pages.as_mut_ptr().cast(),
                plain.len(),
                packed.as_ptr().cast(),
                0,
                packed.len(),
            );
            r.page_offset = 50;
            enqueue(queue, &r);
        }
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        for pages in &destinations {
            let mapped = xmem::XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                0x2000 | 0x2040_0000, // reserve, 64 KiB pages
                0,
                0x04,
            );
            assert!(!mapped.is_null());
            assert_eq!(
                xmem::XMemMapPhysicalPages(mapped, 3, pages.as_ptr()),
                mapped
            );
            // SAFETY: all three pages are mapped read-write.
            let seen = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), 3 * PAGE) };
            assert!(seen[..50].iter().all(|&b| b == 0));
            assert_eq!(seen[50..50 + plain.len()], plain);
            assert_eq!(
                xmem::XMemMapPhysicalPages(mapped, 3, std::ptr::null()),
                mapped
            );
        }
        assert_eq!(xmem::XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(xmem::XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn zlib_inflate_handles_short_exact_empty_and_bad_streams() {
        let plain = pattern(65_537);
        let packed = compress_zlib(&plain);

        // Exact fit.
        let mut exact = vec![0u8; plain.len()];
        assert_eq!(inflate(&packed, &mut exact), Ok(plain.len()));
        assert_eq!(exact, plain);

        // Short stream into a larger buffer passes and returns bytes written;
        // guard bytes outside the slice passed to `inflate` stay untouched.
        let mut larger = vec![0xAA; plain.len() + 64];
        assert_eq!(
            inflate(&packed, &mut larger[..plain.len() + 32]),
            Ok(plain.len())
        );
        assert_eq!(&larger[..plain.len()], &plain[..]);
        assert!(larger[plain.len() + 32..].iter().all(|&b| b == 0xAA));

        // Empty valid stream passes for empty or non-empty destinations.
        let empty = compress_zlib(&[]);
        assert_eq!(inflate(&empty, &mut []), Ok(0));
        let mut scratch = [0x55u8; 4];
        assert_eq!(inflate(&empty, &mut scratch), Ok(0));

        // Oversized stream (destination too small, including empty) fails.
        let mut small = vec![0u8; plain.len() - 1];
        assert_eq!(inflate(&packed, &mut small), Err(INVALID_DATA));
        assert_eq!(inflate(&packed, &mut []), Err(INVALID_DATA));

        // Truncated stream (missing trailer or payload bytes) fails.
        let mut out = vec![0u8; plain.len()];
        assert_eq!(
            inflate(&packed[..packed.len() - 1], &mut out),
            Err(INVALID_DATA)
        );
        assert_eq!(
            inflate(&packed[..packed.len() / 2], &mut out),
            Err(INVALID_DATA)
        );
        assert_eq!(inflate(&[], &mut out), Err(INVALID_DATA));

        // Corrupt Adler-32 trailer fails.
        let mut bad_adler = packed.clone();
        *bad_adler.last_mut().unwrap() ^= 1;
        assert_eq!(inflate(&bad_adler, &mut out), Err(INVALID_DATA));

        // Corrupt header/payload fails, and subsequent valid inflate on the
        // same thread still succeeds (verifies thread-local state reset).
        let mut bad_body = packed.clone();
        bad_body[2] ^= 0xFF;
        assert_eq!(inflate(&bad_body, &mut out), Err(INVALID_DATA));
        assert_eq!(inflate(&packed, &mut out), Ok(plain.len()));
        assert_eq!(out, plain);
    }
    /// DirectStorage queues submit by halves unless the title created them
    /// with AUTO_SUBMIT_DISABLED (DSTORAGE_QUEUE_DESC1's low bit): the flag
    /// is honored (after the base fork's 7f0156c09).
    #[test]
    fn auto_submit_disabled_queues_wait_for_submit() {
        // (options, whether the one enqueued read goes by itself at capacity 2)
        for (options, auto) in [(0u64, true), (AUTO_SUBMIT_DISABLED, false)] {
            let queue = queue_with_options(0, 2, options);
            let mut buf = [0u8; 64];
            enqueue(
                queue,
                &request(0, buf.as_mut_ptr().cast(), buf.len(), std::ptr::null(), 0, buf.len()),
            );
            let id = unsafe { &*queue.cast::<Instance<Queue>>() }.state.id;
            let pending = scheduler()
                .expect("a scheduler")
                .lock()
                .lanes
                .get(&id)
                .map_or(0, |l| l.unsubmitted.len());
            assert_eq!(pending, if auto { 0 } else { 1 }, "options {options:#x}");
            void(queue, 0x40);
        }
    }

    /// The scheduler's id of a queue object.
    fn lane_id(queue: *mut c_void) -> u64 {
        // SAFETY: a queue the test created and holds.
        unsafe { &*queue.cast::<Instance<Queue>>() }.state.id
    }

    /// A queue with AUTO_SUBMIT_DISABLED that its enqueuer fills with status
    /// entries, the last enqueue waiting for room: another thread's Submit
    /// writes them all at once (no reads before them), and the waiting
    /// enqueuer goes on, as the console's contract for such queues has it
    /// (Sol's review of the fork fixes, condition 3). Before, Submit woke
    /// the workers only and the enqueuer slept on.
    #[test]
    fn a_full_manual_queue_goes_on_once_submit_frees_it() {
        let queue = queue_with_options(0, 4, AUTO_SUBMIT_DISABLED);
        let array = status_array(4);
        for index in 0..3 {
            enqueue_status(queue, array, index);
        }
        let (q, a) = (queue as usize, array as usize);
        let (went, gone) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            enqueue_status(q as *mut c_void, a as *mut c_void, 3);
            let _ = went.send(());
        });
        // One slot stays free: the fourth waits.
        assert!(gone.recv_timeout(Duration::from_millis(100)).is_err(), "a full queue waits");
        void(queue, 0x30);
        let woke = gone.recv_timeout(Duration::from_secs(2)).is_ok();
        if !woke {
            void(queue, 0x40); // lets the enqueuer go
        }
        assert!(woke, "Submit freed the queue, and its enqueuer went on");
        assert_eq!((0..3).map(|i| result(array, i)).collect::<Vec<_>>(), [S_OK; 3]);
        void(queue, 0x30);
        assert_eq!(wait(array, 3), S_OK);
        void(queue, 0x40);
    }

    /// The title releases its file after it enqueued two reads of it and
    /// before it submits them: each read holds the file until it is done,
    /// and lets it go before its completion is published, so the handle is
    /// closed by the time the title sees the status (Sol's review of the
    /// fork fixes, condition 3; after the base fork's 6d8ceb6ef).
    #[test]
    fn a_file_released_after_enqueue_is_kept_until_its_reads_are_done() {
        let plain = pattern(8192);
        let packed = compress_zlib(&plain);
        let mut contents = plain.clone();
        contents.extend_from_slice(&packed);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("released.bin");
        std::fs::write(&path, &contents).unwrap();
        let file = open(&path);
        let (queue, array) = (new_queue(0), status_array(1));
        let (mut raw, mut inflated) = (vec![0u8; 512], vec![0u8; plain.len()]);
        enqueue(queue, &request(0, raw.as_mut_ptr().cast(), raw.len(), file, 128, raw.len()));
        let offset = plain.len() as u64;
        enqueue(queue, &request(ZLIB, inflated.as_mut_ptr().cast(), plain.len(), file, offset, packed.len()));
        let release: extern "system" fn(*mut c_void) -> u32 = call(file, 0x10);
        assert_eq!(release(file), 2, "each queued read holds the file");
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(raw, plain[128..640]);
        assert_eq!(inflated, plain);
        std::fs::remove_file(&path).expect("the reads let the file go before their completion was published");
        void(queue, 0x40);
    }

    /// A file read into physical pages whose page array the title has not
    /// filled, after which the title releases the file: the read is held
    /// back and looked at again, holding the file all the while, and reads
    /// it once the array is filled; another such read, still held when its
    /// queue closes, lets the file go with the queue (Sol's review of the
    /// fork fixes, condition 3).
    #[test]
    fn a_held_read_keeps_its_file_through_page_array_retries_and_queue_close() {
        use crate::modules::xmem;
        let (mut pages, mut n) = ([0usize; 1], 1);
        assert_eq!(xmem::XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let mapped = xmem::XMemVirtualAlloc(std::ptr::null_mut(), PAGE, 0x2000 | 0x2040_0000, 0, 0x04);
        assert!(!mapped.is_null());
        assert_eq!(xmem::XMemMapPhysicalPages(mapped, 1, pages.as_ptr()), mapped);
        let plain = pattern(1000);
        let dir = tempfile::tempdir().unwrap();
        // Page arrays not filled yet; leaked, as a worker may look at one
        // until its request is done or dropped.
        let unfilled = || -> &'static std::sync::atomic::AtomicUsize {
            Box::leak(Box::new(std::sync::atomic::AtomicUsize::new(0)))
        };
        let held = |queue: *mut c_void| {
            let (id, start) = (lane_id(queue), Instant::now());
            let held = || scheduler().unwrap().lock().lanes.get(&id).is_some_and(|l| !l.held.is_empty());
            while !held() && start.elapsed() < Duration::from_secs(5) {
                std::thread::sleep(Duration::from_millis(1));
            }
            held()
        };
        let file_at = |name: &str| {
            let path = dir.path().join(name);
            std::fs::write(&path, &plain).unwrap();
            let file = open(&path);
            let release: extern "system" fn(*mut c_void) -> u32 = call(file, 0x10);
            (path, file, release)
        };

        // Retried, then filled.
        let (path, file, release) = file_at("retried.bin");
        let (queue, array) = (new_queue(0), status_array(1));
        let late = unfilled();
        enqueue(queue, &request(PAGE_DESTINATION, late.as_ptr().cast(), 1000, file, 0, 1000));
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert!(held(queue), "held back for its page array");
        assert_eq!(release(file), 1, "the held read keeps the file");
        std::thread::sleep(Duration::from_millis(20)); // looked at again meanwhile
        late.store(pages[0], Ordering::SeqCst);
        assert_eq!(wait(array, 0), S_OK);
        // SAFETY: the page is mapped read-write there.
        let seen = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), 1000) };
        assert_eq!(seen, plain);
        std::fs::remove_file(&path).expect("the read let the file go before its completion was published");
        void(queue, 0x40);

        // Still held when its queue closes.
        let (path, file, release) = file_at("closed.bin");
        let queue = new_queue(0);
        enqueue(queue, &request(PAGE_DESTINATION, unfilled().as_ptr().cast(), 1000, file, 0, 1000));
        void(queue, 0x30);
        assert!(held(queue), "held back for its page array");
        assert_eq!(release(file), 1, "the held read keeps the file");
        assert!(std::fs::remove_file(&path).is_err(), "open while the read holds it");
        void(queue, 0x40);
        std::fs::remove_file(&path).expect("the queue's close let the file go");
    }
}
