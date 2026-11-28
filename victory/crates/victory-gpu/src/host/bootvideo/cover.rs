//! The loading cover (`VICTORY_BOOT_VIDEO_COVER=1`, with `VICTORY_BOOT_VIDEO`;
//! off unless set): once the boot video has played, the same video covers the
//! title's world-loading screen, forward to its last frame and back to its
//! first, until the world is drawn.
//!
//! It is a surface of its own: a click-through window of ours over the title
//! window's client area, with its own queue, fence and composition swap
//! chain. It shares nothing with the title's presents but the device (free
//! threaded): no host lock, no host submission, no retirement ticket.
//!
//! Two threads. The keeper owns the window: it watches the title's presents
//! (a ring written where they are counted, `crate::queue`), decides the
//! begin and the end, shows and hides the window, and owns every timeout. It
//! takes no lock the presenter holds (only `try_lock`), calls no DXGI or
//! D3D12, and hides before anything is let go; a guard hides and destroys
//! the window however the keeper leaves, a panic too. The presenter owns the
//! card's objects and presents without waiting (a waitable object asked
//! with no timeout, `DXGI_PRESENT_DO_NOT_WAIT`); it lets go of them only
//! after the keeper has said the window is hidden and the fence has passed
//! everything sent, and otherwise leaves them alone.
//!
//! Unset, nothing is made or asked: the presenter's window is not even
//! looked at, and the title's lookup of its window asks nothing more.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{CloseHandle, DXGI_STATUS_OCCLUDED, HANDLE, HWND, LPARAM, LRESULT, POINT, RECT, WAIT_OBJECT_0, WAIT_TIMEOUT, WPARAM};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::DirectComposition::{DCompositionCreateDevice2, IDCompositionDesktopDevice, IDCompositionTarget, IDCompositionVisual2};
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::Graphics::Gdi::{ClientToScreen, MONITOR_DEFAULTTONEAREST, MonitorFromWindow};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

use super::{Plan, Player, Rows, Step, decode_file, fit, qoi, write_frame};

/// Decode workers (as the boot video's).
const WORKERS: usize = 2;
/// Upload buffers, each reused once the copy out of it has run.
const UPLOADS: usize = 3;
/// Allocators, each reset once its list has run.
const ALLOCATORS: usize = 2;
/// The ceiling: the cover ends this long after it began, whatever else.
const CEILING_S: f64 = 120.0;
/// The title stopped presenting this long: the cover ends.
const STALL_S: f64 = 5.0;
/// Frame 0 is presented twice and has run on the card within this long of
/// the begin, or the cover ends without being shown.
const READY_S: f64 = 1.0;
/// How long either thread waits for the other at the end; past it, what the
/// presenter holds is left alone for the life of the process.
const STOP_S: f64 = 2.0;
/// The keeper's look while covering, and while watching.
const COVER_TICK: Duration = Duration::from_millis(10);
const WATCH_TICK: Duration = Duration::from_millis(100);
/// Messages the keeper handles at most per look.
const MESSAGES: usize = 64;
/// The window class (`title_window` skips windows of it).
const CLASS: PCWSTR = w!("victory loading cover");

/// Whether `VICTORY_BOOT_VIDEO_COVER` asks for the cover: exactly `1`.
fn asked(value: Option<OsString>) -> bool {
    value.is_some_and(|v| v == "1")
}

/// [`asked`] of the environment, read once.
fn switch_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| asked(std::env::var_os("VICTORY_BOOT_VIDEO_COVER")))
}

// ---- the path: forward, then backward, never from the end to the start ----

/// The frame of a video of `n` frames shown at position `pos` of the path
/// 0, 1, .., n - 1, n - 2, .., 1, 0, 1, ..: the same frame is never shown
/// twice at a turn, and the last is never followed by the first.
pub(super) fn path_frame(pos: u64, n: u32) -> u32 {
    if n <= 1 {
        return 0;
    }
    let period = 2 * u64::from(n - 1);
    let k = pos % period;
    (if k < u64::from(n) { k } else { period - k }) as u32
}

// ---- the begin and the end, from the title's presents ----

/// Why the cover ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Why {
    World,
    Light,
    Stall,
    Ceiling,
    /// The record of the presents had a gap or went back in time: what the
    /// title shows is not known, so it is shown.
    Unsure,
}

impl Why {
    fn text(self) -> &'static str {
        match self {
            Why::World => "the world is drawn",
            Why::Light => "a light screen (a prompt or a menu)",
            Why::Stall => "the title stopped presenting",
            Why::Ceiling => "the ceiling",
            Why::Unsure => "the record of the title's presents had a gap",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Event {
    /// The world-loading section has begun.
    Begin,
    /// A second of the world was seen before any loading: the cover never begins.
    Disarmed,
    End(Why),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Watch {
    Watching,
    Covering,
    Over,
}

/// The rule over the title's presents (times in seconds, draws counted by the
/// host between one present and the next): a pure function of them.
///
/// - Begins when in the last 1.5 s every present carried at least 1,500
///   draws and each of its two seconds held 2 to 7 presents.
/// - The world (the design's rule): a second with 8 or more presents
///   averaging 1,000 draws or more. While covering it ends the cover. Before
///   a begin it disarms the cover for good, if the median of that second is
///   1,000 or more too: a second of light start screens with the loading's
///   first heavy frames averages heavy (the `transition` test).
/// - A light screen: 3 s of presents each under 1,000 draws, ends it.
/// - No present for 5 s, or 120 s since the begin, ends it ([`Detector::tick`];
///   the keeper keeps its own ceiling too).
/// - A present missing from the record, or one earlier than the last: before
///   a begin, what is known is cleared; while covering, the cover ends.
pub(super) struct Detector {
    last: Option<(u64, f64, u64)>,
    /// (time, draws) of the presents of the last few seconds.
    seen: VecDeque<(f64, u64)>,
    watch: Watch,
    began: f64,
    last_present: f64,
}

impl Detector {
    pub(super) fn new() -> Detector {
        Detector { last: None, seen: VecDeque::new(), watch: Watch::Watching, began: 0.0, last_present: 0.0 }
    }

    /// The title's present number `frame` at `t`, with `drawn` draws recorded so far.
    pub(super) fn present(&mut self, frame: u64, t: f64, drawn: u64) -> Option<Event> {
        let whole = match self.last {
            Some((f, at, d)) if frame == f + 1 && t >= at && drawn >= d => {
                self.seen.push_back((t, drawn - d));
                true
            }
            Some(_) => false,
            None => true,
        };
        self.last = Some((frame, t, drawn));
        self.last_present = t;
        if !whole {
            self.seen.clear();
            if self.watch == Watch::Covering {
                return self.end(Why::Unsure);
            }
            return None;
        }
        while self.seen.front().is_some_and(|&(at, _)| at < t - 4.0) {
            self.seen.pop_front();
        }
        let second: Vec<u64> = self.seen.iter().filter(|&&(at, _)| at > t - 1.0).map(|&(_, d)| d).collect();
        let world = second.len() >= 8 && second.iter().sum::<u64>() >= 1000 * second.len() as u64;
        match self.watch {
            Watch::Watching if world && median(second) >= 1000 => {
                self.watch = Watch::Over;
                Some(Event::Disarmed)
            }
            Watch::Watching if self.loading(t) => {
                self.watch = Watch::Covering;
                self.began = t;
                Some(Event::Begin)
            }
            Watch::Covering if world => self.end(Why::World),
            Watch::Covering if self.light(t) => self.end(Why::Light),
            _ => None,
        }
    }

    /// The time is `now` (whether or not the title presented).
    pub(super) fn tick(&mut self, now: f64) -> Option<Event> {
        if self.watch != Watch::Covering {
            return None;
        }
        if now - self.began >= CEILING_S {
            return self.end(Why::Ceiling);
        }
        if now - self.last_present >= STALL_S {
            return self.end(Why::Stall);
        }
        None
    }

    fn end(&mut self, why: Why) -> Option<Event> {
        self.watch = Watch::Over;
        Some(Event::End(why))
    }

    /// Presents in `(from, to]`.
    fn within(&self, from: f64, to: f64) -> usize {
        self.seen.iter().filter(|&&(at, _)| at > from && at <= to).count()
    }

    fn loading(&self, t: f64) -> bool {
        // What is known reaches back 1.5 s.
        if self.seen.front().is_none_or(|&(at, _)| at > t - 1.5) {
            return false;
        }
        let heavy = self.seen.iter().filter(|&&(at, _)| at > t - 1.5).all(|&(_, d)| d >= 1500);
        heavy && (2..=7).contains(&self.within(t - 1.0, t)) && (2..=7).contains(&self.within(t - 1.5, t - 0.5))
    }

    fn light(&self, t: f64) -> bool {
        if self.seen.front().is_none_or(|&(at, _)| at > t - 3.0) {
            return false;
        }
        let recent: Vec<u64> = self.seen.iter().filter(|&&(at, _)| at > t - 3.0).map(|&(_, d)| d).collect();
        !recent.is_empty() && recent.iter().all(|&d| d < 1000)
    }
}

fn median(mut values: Vec<u64>) -> u64 {
    values.sort_unstable();
    values.get(values.len() / 2).copied().unwrap_or(0)
}

/// Feeds the records after `*after` one at a time and stops at the first
/// event: the records behind it stay for the next look (a begin and an end
/// in one look are two looks, so the end is never lost).
fn feed(detector: &mut Detector, records: &[(u64, f64, u64)], after: &mut u64) -> Option<Event> {
    let from = *after;
    for &(frame, t, drawn) in records.iter().filter(|r| r.0 > from) {
        *after = frame;
        if let Some(event) = detector.present(frame, t, drawn) {
            return Some(event);
        }
    }
    None
}

// ---- the title's window, as the host's presenter chose it ----

/// The window the host presents to (0: none yet), its thread, and which
/// presenter it is (a new one each time `Host::presenter` makes one).
/// Written only with the cover asked for.
static TITLE_HWND: AtomicU64 = AtomicU64::new(0);
static TITLE_THREAD: AtomicU32 = AtomicU32::new(0);
static TITLE_MADE: AtomicU64 = AtomicU64::new(0);
/// The cover's window class has been registered (`is_cover` asks nothing before).
static CLASS_MADE: AtomicBool = AtomicBool::new(false);

/// The host's presenter is made for `hwnd`: with the cover asked for, its
/// window, thread and number are published (else nothing is asked or written).
pub(in crate::host) fn title_window_is(hwnd: HWND) {
    if !switch_on() {
        return;
    }
    // SAFETY: a plain query of a window handle.
    let thread = unsafe { GetWindowThreadProcessId(hwnd, None) };
    let made = TITLE_MADE.load(Ordering::Relaxed) + 1;
    // Odd while it is written.
    TITLE_MADE.store(2 * made - 1, Ordering::Relaxed);
    std::sync::atomic::fence(Ordering::Release);
    TITLE_THREAD.store(thread, Ordering::Relaxed);
    TITLE_HWND.store(hwnd.0 as usize as u64, Ordering::Relaxed);
    TITLE_MADE.store(2 * made, Ordering::Release);
}

/// Whether `hwnd` is the cover's own window (nothing is asked before the
/// cover ever made its class).
pub(in crate::host) fn is_cover(hwnd: HWND) -> bool {
    if !switch_on() || !CLASS_MADE.load(Ordering::Acquire) {
        return false;
    }
    let mut name = [0u16; 32];
    // SAFETY: a plain query into our buffer.
    let n = unsafe { GetClassNameW(hwnd, &mut name) };
    // SAFETY: a static wide string.
    n > 0 && &name[..n as usize] == unsafe { CLASS.as_wide() }
}

/// The title's window, pinned: the presenter it was published by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Title {
    hwnd: usize,
    thread: u32,
    made: u64,
}

impl Title {
    fn hwnd(self) -> HWND {
        HWND(self.hwnd as *mut _)
    }
}

/// The title's window as published now, if it is whole and alive.
fn title_now() -> Option<Title> {
    let made = TITLE_MADE.load(Ordering::Acquire);
    let (hwnd, thread) = (TITLE_HWND.load(Ordering::Relaxed) as usize, TITLE_THREAD.load(Ordering::Relaxed));
    std::sync::atomic::fence(Ordering::Acquire);
    if made == 0 || made % 2 == 1 || TITLE_MADE.load(Ordering::Relaxed) != made || hwnd == 0 {
        return None;
    }
    let title = Title { hwnd, thread, made };
    let mut process = 0;
    // SAFETY: plain queries of a window handle.
    let (alive, of) = unsafe { (IsWindow(Some(title.hwnd())).as_bool(), GetWindowThreadProcessId(title.hwnd(), Some(&mut process))) };
    (alive && of == thread && process == std::process::id()).then_some(title)
}

// (In user32 since Windows 10 1607; the `windows` crate's feature for them is not on.)
#[link(name = "user32")]
unsafe extern "system" {
    fn GetDpiForWindow(hwnd: *mut core::ffi::c_void) -> u32;
    fn GetWindowDpiAwarenessContext(hwnd: *mut core::ffi::c_void) -> isize;
    fn SetThreadDpiAwarenessContext(context: isize) -> isize;
}

/// Where the title's client area is on the screen, its monitor and its DPI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Place {
    at: (i32, i32),
    size: (u32, u32),
    monitor: isize,
    dpi: u32,
}

fn place_of(hwnd: HWND) -> Option<Place> {
    let mut rect = RECT::default();
    let mut origin = POINT::default();
    // SAFETY: plain queries of a live window into our structures.
    unsafe {
        GetClientRect(hwnd, &mut rect).ok()?;
        if !ClientToScreen(hwnd, &mut origin).as_bool() {
            return None;
        }
        let monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
        let dpi = GetDpiForWindow(hwnd.0);
        let size = ((rect.right - rect.left).max(0) as u32, (rect.bottom - rect.top).max(0) as u32);
        (size.0 > 0 && size.1 > 0 && dpi > 0).then_some(Place { at: (origin.x, origin.y), size, monitor: monitor.0 as isize, dpi })
    }
}

/// Whether the title's window is the one in front and not minimised.
fn in_front(hwnd: HWND) -> bool {
    // SAFETY: plain queries.
    unsafe { GetForegroundWindow() == hwnd && !IsIconic(hwnd).as_bool() }
}

/// F1 presses so far, as the settings menu's key thread latches them where
/// it reads the keys (`crate::F1_PRESSES`): a press is seen even when the
/// title stops presenting, or F1 and Esc reach the menu in one batch.
fn f1_presses() -> u32 {
    crate::F1_PRESSES.load(Ordering::Acquire)
}

// ---- the title's presents, read from the ring `crate::queue` writes ----

/// Seconds of the performance counter.
fn seconds(ticks: u64) -> f64 {
    static FREQUENCY: OnceLock<f64> = OnceLock::new();
    let f = *FREQUENCY.get_or_init(|| {
        let mut f = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceFrequency(&mut f) };
        f.max(1) as f64
    });
    ticks as f64 / f
}

fn now() -> f64 {
    let mut count = 0;
    // SAFETY: a valid out pointer.
    unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut count) };
    seconds(count as u64)
}

/// The presents in the ring (number, time, draws so far), in order of number.
fn records() -> Vec<(u64, f64, u64)> {
    let mut found: Vec<(u64, f64, u64)> = crate::queue::cover_presents().map(|(f, t, d)| (f, seconds(t), d)).collect();
    found.sort_unstable_by_key(|p| p.0);
    found
}

// ---- the card's side: a queue, a composition swap chain, uploads ----

/// A handle, closed when it goes (owned from the moment it is had).
struct Owned(HANDLE);

impl Drop for Owned {
    fn drop(&mut self) {
        // SAFETY: our own handle, closed once.
        unsafe {
            let _ = CloseHandle(self.0);
        }
    }
}

/// What a send is, for what may be let go after a failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sent {
    /// Nothing of it went to the card.
    No,
    /// Given to the queue, its fence value not yet signalled: kept for good.
    Unfenced,
    /// Done once the fence reaches the value.
    At(u64),
}

struct Upload {
    buffer: ID3D12Resource,
    data: *mut u8,
    sent: Sent,
}

struct Allocator {
    allocator: ID3D12CommandAllocator,
    sent: Sent,
}

/// The cover's objects on the card: made by the keeper, used by the presenter.
struct Parts {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    chain: IDXGISwapChain3,
    waitable: Owned,
    allocators: Vec<Allocator>,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    event: Owned,
    value: u64,
    uploads: Vec<Upload>,
    size: (u32, u32),
    pitch: u32,
    /// Set by the keeper once the window is hidden: nothing is let go before.
    hidden: Arc<AtomicBool>,
}

// SAFETY: free-threaded D3D12 and DXGI objects, handles, and mapped upload
// memory written by the one thread that owns the surface at a time.
unsafe impl Send for Parts {}

/// [`Parts`], let go of only once the keeper has hidden the window and the
/// card has run everything sent with them; else left alone for the life of
/// the process. On an end, a failure or an unwinding panic alike.
struct Surface(std::mem::ManuallyDrop<Parts>);

impl std::ops::Deref for Surface {
    type Target = Parts;
    fn deref(&self) -> &Parts {
        &self.0
    }
}

impl std::ops::DerefMut for Surface {
    fn deref_mut(&mut self) -> &mut Parts {
        &mut self.0
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        let deadline = Instant::now() + Duration::from_secs_f64(STOP_S);
        while !self.hidden.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(2));
        }
        if self.hidden.load(Ordering::Acquire) && self.settled(Duration::from_secs_f64(STOP_S)) {
            // SAFETY: dropped once, here; hidden, and nothing sent with them is still on the card.
            unsafe { std::mem::ManuallyDrop::drop(&mut self.0) };
        } else {
            tracing::warn!("loading cover: not known to be hidden and run; its objects are left alone");
        }
    }
}

const FORMAT: DXGI_FORMAT = DXGI_FORMAT_B8G8R8A8_UNORM;

fn named(object: &impl Interface, text: &str) {
    crate::host::name(object, || format!("victory cover: {text}"));
}

fn failed() -> windows::core::Error {
    windows::core::Error::from_hresult(windows::Win32::Foundation::E_FAIL)
}

impl Parts {
    /// A surface `size` for `hwnd` (whose window association is turned off), on `device`.
    fn make(device: &ID3D12Device, hwnd: HWND, size: (u32, u32), hidden: Arc<AtomicBool>) -> windows::core::Result<Surface> {
        // SAFETY: plain object creation on a live device; the mapped
        // pointers stay valid while the buffers live (in the surface).
        unsafe {
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })?;
            named(&queue, "queue");
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            // As `CreateSwapChainForComposition` documents it: flip-sequential, stretched.
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: size.0,
                Height: size.1,
                Format: FORMAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: 2,
                Scaling: DXGI_SCALING_STRETCH,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
                AlphaMode: DXGI_ALPHA_MODE_IGNORE,
                Flags: DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT.0 as u32,
                ..Default::default()
            };
            let chain: IDXGISwapChain3 = factory.CreateSwapChainForComposition(&queue, &desc, None)?.cast()?;
            factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_WINDOW_CHANGES | DXGI_MWA_NO_ALT_ENTER)?;
            chain.SetMaximumFrameLatency(1)?;
            let waitable = Owned(chain.GetFrameLatencyWaitableObject());
            if waitable.0.is_invalid() {
                return Err(failed());
            }
            let event = Owned(CreateEventW(None, false, false, None)?);
            let allocators = (0..ALLOCATORS)
                .map(|i| {
                    let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                    named(&allocator, &format!("allocator {i}"));
                    Ok(Allocator { allocator, sent: Sent::No })
                })
                .collect::<windows::core::Result<Vec<_>>>()?;
            let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocators[0].allocator, None)?;
            list.Close()?;
            named(&list, "list");
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            named(&fence, "fence");
            let pitch = (size.0 * 4).next_multiple_of(256);
            let bytes = u64::from(pitch) * u64::from(size.1);
            let mut uploads = Vec::with_capacity(UPLOADS);
            for i in 0..UPLOADS {
                let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: bytes,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                let mut buffer: Option<ID3D12Resource> = None;
                device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut buffer)?;
                let buffer = buffer.ok_or_else(failed)?;
                named(&buffer, &format!("upload {i}"));
                let mut data = std::ptr::null_mut();
                buffer.Map(0, None, Some(&mut data))?;
                uploads.push(Upload { buffer, data: data.cast(), sent: Sent::No });
            }
            let parts = Parts { device: device.clone(), queue, chain, waitable, allocators, list, fence, event, value: 0, uploads, size, pitch, hidden };
            Ok(Surface(std::mem::ManuallyDrop::new(parts)))
        }
    }

    /// The bytes it holds: the uploads on the processor's side, the swap chain's two buffers on the card's.
    fn bytes(&self) -> (u64, u64) {
        let one = u64::from(self.pitch) * u64::from(self.size.1);
        (one * self.uploads.len() as u64, u64::from(self.size.0) * u64::from(self.size.1) * 4 * 2)
    }

    /// The fence has passed `v` (never once the device is removed: `u64::MAX`).
    fn passed(&self, v: u64) -> bool {
        // SAFETY: a plain query.
        let done = unsafe { self.fence.GetCompletedValue() };
        done != u64::MAX && done >= v
    }

    fn done(&self, sent: Sent) -> bool {
        match sent {
            Sent::No => true,
            Sent::Unfenced => false,
            Sent::At(v) => self.passed(v),
        }
    }

    /// Whether everything sent has run (waited for up to `wait`).
    fn settled(&self, wait: Duration) -> bool {
        if self.allocators.iter().any(|a| a.sent == Sent::Unfenced) || self.uploads.iter().any(|u| u.sent == Sent::Unfenced) {
            return false;
        }
        if self.value == 0 || self.passed(self.value) {
            return true;
        }
        // SAFETY: our fence and event.
        let woke = unsafe {
            self.fence.SetEventOnCompletion(self.value, self.event.0).is_ok() && WaitForSingleObject(self.event.0, wait.as_millis() as u32) == WAIT_OBJECT_0
        };
        // (A removal also wakes the event: the value is asked again.)
        woke && self.passed(self.value)
    }
}

/// What a present tick did.
#[derive(Debug, PartialEq, Eq)]
enum Tick {
    Presented,
    /// Busy (the swap chain, an allocator or an upload): skipped.
    Busy,
    /// The present went, and the window is occluded: not counted as shown.
    Occluded,
    /// Asked to stop before the send: nothing sent.
    Stopped,
}

/// The presenter's counts, read by the keeper.
#[derive(Default, Debug, Clone)]
struct Counts {
    presented: u64,
    busy: u64,
    occluded: u64,
    slowest_scale_ms: f64,
    slowest_present_ms: f64,
    held: (u64, u64),
    /// Of the decoders: frames shown, frames due and never shown, the slowest decode.
    decoded_shown: u64,
    skipped: u64,
    slowest_decode_ms: f64,
}

/// Writes `image` into a free upload, copies it to the next back buffer and
/// presents without waiting; sends nothing once `stop` is set.
fn present(s: &mut Surface, image: &qoi::Image, rows: &mut Rows, counts: &mut Counts, stop: &AtomicBool) -> Result<Tick, String> {
    // SAFETY: our own waitable object, asked without waiting.
    match unsafe { WaitForSingleObject(s.waitable.0, 0) } {
        WAIT_OBJECT_0 => {}
        WAIT_TIMEOUT => {
            counts.busy += 1;
            return Ok(Tick::Busy);
        }
        other => return Err(format!("the swap chain's waitable object: {other:?}")),
    }
    let free = (0..s.uploads.len()).find(|&i| s.done(s.uploads[i].sent)).zip((0..s.allocators.len()).find(|&i| s.done(s.allocators[i].sent)));
    let Some((slot, a)) = free else {
        counts.busy += 1;
        return Ok(Tick::Busy);
    };
    let began = Instant::now();
    let fit = fit((image.width, image.height), s.size);
    let len = s.pitch as usize * s.size.1 as usize;
    // SAFETY: a mapped upload buffer of this length, whose last copy has run.
    let out = unsafe { std::slice::from_raw_parts_mut(s.uploads[slot].data, len) };
    write_frame(image, &fit, FORMAT, out, s.pitch as usize, rows);
    counts.slowest_scale_ms = counts.slowest_scale_ms.max(began.elapsed().as_secs_f64() * 1e3);
    // SAFETY: our list, allocator (its last list has run), live resources.
    unsafe {
        let back: ID3D12Resource = s.chain.GetBuffer(s.chain.GetCurrentBackBufferIndex()).map_err(|e| e.to_string())?;
        s.allocators[a].allocator.Reset().map_err(|e| e.to_string())?;
        s.list.Reset(&s.allocators[a].allocator, None).map_err(|e| e.to_string())?;
        s.list.ResourceBarrier(&[crate::host::transition(&back, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST)]);
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&s.uploads[slot].buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: FORMAT, Width: s.size.0, Height: s.size.1, Depth: 1, RowPitch: s.pitch },
                },
            },
        };
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&back),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        s.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        s.list.ResourceBarrier(&[crate::host::transition(&back, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT)]);
        s.list.Close().map_err(|e| e.to_string())?;
        drop(back);
        if stop.load(Ordering::Acquire) {
            return Ok(Tick::Stopped);
        }
        // From here until the signal, what the list uses is the card's.
        s.allocators[a].sent = Sent::Unfenced;
        s.uploads[slot].sent = Sent::Unfenced;
        let list: ID3D12CommandList = s.list.cast().map_err(|e| e.to_string())?;
        s.queue.ExecuteCommandLists(&[Some(list)]);
        let value = s.value + 1;
        s.queue.Signal(&s.fence, value).map_err(|e| format!("signal: {e}"))?;
        s.value = value;
        s.allocators[a].sent = Sent::At(value);
        s.uploads[slot].sent = Sent::At(value);
        let called = Instant::now();
        let shown = s.chain.Present(0, DXGI_PRESENT_DO_NOT_WAIT);
        counts.slowest_present_ms = counts.slowest_present_ms.max(called.elapsed().as_secs_f64() * 1e3);
        if shown == DXGI_ERROR_WAS_STILL_DRAWING {
            counts.busy += 1;
            return Ok(Tick::Busy);
        }
        if shown == DXGI_STATUS_OCCLUDED {
            counts.occluded += 1;
            return Ok(Tick::Occluded);
        }
        if shown.is_err() {
            return Err(format!("present: {}", windows::core::Error::from_hresult(shown)));
        }
    }
    counts.presented += 1;
    Ok(Tick::Presented)
}

// ---- the two threads ----

/// What the keeper and the presenter share. The keeper only reads the
/// atomics and `try_lock`s the rest.
#[derive(Default)]
struct Shared {
    /// The keeper asks the presenter to stop.
    stop: AtomicBool,
    /// The window is not up: no presents (the clock runs on).
    paused: AtomicBool,
    /// The window is hidden (the keeper says so): the presenter may let go.
    hidden: Arc<AtomicBool>,
    /// Frame 0 has been presented twice and has run on the card.
    ready: AtomicBool,
    /// The presenter stopped by itself (why: `why`).
    failed: AtomicBool,
    why: Mutex<Option<String>>,
    /// The presenter has let go of everything, or left it alone.
    finished: AtomicBool,
    counts: Mutex<Counts>,
}

impl Shared {
    fn fail(&self, why: String) {
        if let Ok(mut w) = self.why.lock() {
            w.get_or_insert(why);
        }
        self.failed.store(true, Ordering::Release);
    }
}

/// Says the presenter failed if it unwinds, before anything else of it is
/// let go (it is made last, so it goes first).
struct FailOnUnwind(Arc<Shared>);

impl Drop for FailOnUnwind {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.fail("the presenter panicked".to_owned());
        }
    }
}

/// The presenter: decodes ahead, presents frame 0 twice for the show, then
/// by the clock, until asked to stop or a failure.
fn presenter(mut surface: Surface, plan: Plan, shared: Arc<Shared>) {
    let frames = plan.frames;
    let mut along = plan.clone();
    along.loops = u32::MAX;
    let decoded = plan.clone();
    let player = Player::start_with(along, WORKERS, Arc::new(move |pos| decode_file(&decoded, u64::from(path_frame(pos, frames)))));
    let _unwinding = FailOnUnwind(Arc::clone(&shared));
    let mut player = match player {
        Ok(p) => p,
        Err(e) => {
            shared.fail(format!("no decoder: {e}"));
            drop(surface);
            shared.finished.store(true, Ordering::Release);
            return;
        }
    };
    if let Some(host) = crate::host::host() {
        host.log_video_memory("the loading cover begins");
    }
    let mut rows = Rows::default();
    let mut counts = Counts { held: surface.bytes(), ..Counts::default() };
    let mut began: Option<Instant> = None;
    let mut last: Option<u64> = None;
    let mut image: Option<qoi::Image> = None;
    // Presents of frame 0 before the show, and the fence value of the second.
    let mut primed = 0;
    let mut primed_at = 0;
    while !shared.stop.load(Ordering::Acquire) {
        std::thread::sleep(Duration::from_millis(4));
        if primed == 2 && !shared.ready.load(Ordering::Relaxed) && surface.passed(primed_at) {
            shared.ready.store(true, Ordering::Release);
        }
        let elapsed = began.map_or(0.0, |b| b.elapsed().as_secs_f64());
        match player.step(elapsed) {
            Step::Show(pos, frame) if last.is_none_or(|l| pos > l) => {
                last = Some(pos);
                image = Some(frame);
            }
            Step::End(why) => {
                shared.fail(why.unwrap_or_else(|| "the video ran out".to_owned()));
                break;
            }
            _ => {}
        }
        // (Hidden for the show: frame 0 is presented while paused too.)
        if primed == 2 && shared.paused.load(Ordering::Acquire) {
            continue;
        }
        let Some(frame) = image.as_ref() else { continue };
        match present(&mut surface, frame, &mut rows, &mut counts, &shared.stop) {
            Ok(Tick::Presented) if primed < 2 => {
                primed += 1;
                primed_at = surface.value;
                if primed == 2 {
                    image = None;
                    began = Some(Instant::now());
                }
            }
            Ok(Tick::Presented) => image = None,
            Ok(_) => {}
            Err(e) => {
                // The device removed, or the cover's own error: either ends
                // the cover only (the title's path reports a removal).
                // SAFETY: a plain query.
                let removed = unsafe { surface.device.GetDeviceRemovedReason() };
                shared.fail(match removed {
                    Err(reason) => format!("the device was removed ({reason}); {e}"),
                    Ok(()) => format!("the cover's own error: {e}"),
                });
                break;
            }
        }
        if let Ok(mut c) = shared.counts.lock() {
            *c = counts.clone();
        }
    }
    player.stop();
    let (shown, skipped, decode_ms, ring) = player.counts();
    counts.held.0 += ring as u64 + rows.bytes() as u64;
    (counts.decoded_shown, counts.skipped, counts.slowest_decode_ms) = (shown, skipped, decode_ms);
    if let Ok(mut c) = shared.counts.lock() {
        *c = counts;
    }
    for worker in std::mem::take(&mut player.workers) {
        let _ = worker.join();
    }
    // (Let go of once hidden and run: `Surface`'s drop.)
    drop(surface);
    if let Some(host) = crate::host::host() {
        host.log_video_memory("the loading cover ended");
    }
    shared.finished.store(true, Ordering::Release);
}

/// A panic in the window procedure: the keeper ends the cover.
static WINDOW_PANICKED: AtomicBool = AtomicBool::new(false);

extern "system" fn window_proc(hwnd: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    let answered = std::panic::catch_unwind(|| match message {
        // Never activated, never hit: clicks and the cursor are the title window's.
        WM_MOUSEACTIVATE => Some(LRESULT(MA_NOACTIVATE as isize)),
        WM_NCHITTEST => Some(LRESULT(HTTRANSPARENT as isize)),
        _ => None,
    });
    match answered {
        Ok(Some(r)) => r,
        Ok(None) => {
            // SAFETY: the default handling of our own window's message.
            unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
        }
        Err(_) => {
            WINDOW_PANICKED.store(true, Ordering::Release);
            LRESULT(0)
        }
    }
}

/// The cover's window, hidden: click-through, never activated, no taskbar
/// entry, no owner and no parent.
fn window(place: Place) -> windows::core::Result<HWND> {
    static REGISTERED: OnceLock<bool> = OnceLock::new();
    // SAFETY: registering our class and making our window, hidden.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let ok = *REGISTERED.get_or_init(|| {
            CLASS_MADE.store(true, Ordering::Release);
            let class = WNDCLASSEXW {
                cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                lpfnWndProc: Some(window_proc),
                hInstance: instance.into(),
                lpszClassName: CLASS,
                ..Default::default()
            };
            RegisterClassExW(&class) != 0
        });
        if !ok {
            return Err(windows::core::Error::from_thread());
        }
        CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_NOREDIRECTIONBITMAP | WS_EX_TOPMOST,
            CLASS,
            w!(""),
            WS_POPUP,
            place.at.0,
            place.at.1,
            place.size.0 as i32,
            place.size.1 as i32,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

/// Shows the window at `place`, over everything, never activated.
fn show(hwnd: HWND, place: Place) -> windows::core::Result<()> {
    // SAFETY: our own window.
    unsafe {
        SetWindowPos(
            hwnd,
            Some(HWND_TOPMOST),
            place.at.0,
            place.at.1,
            place.size.0 as i32,
            place.size.1 as i32,
            SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )
    }
}

/// Hides the window: true once it is not visible.
fn hide(hwnd: HWND) -> bool {
    // SAFETY: our own window.
    unsafe {
        let _ = ShowWindow(hwnd, SW_HIDE);
        !IsWindowVisible(hwnd).as_bool()
    }
}

/// The window's picture: the presenter's swap chain, through DirectComposition.
struct Composition {
    device: IDCompositionDesktopDevice,
    _target: IDCompositionTarget,
    visual: IDCompositionVisual2,
}

fn composition(hwnd: HWND, chain: &IDXGISwapChain3) -> windows::core::Result<Composition> {
    // SAFETY: plain DirectComposition objects for our window.
    unsafe {
        let device: IDCompositionDesktopDevice = DCompositionCreateDevice2(None)?;
        let target = device.CreateTargetForHwnd(hwnd, true)?;
        let visual = device.CreateVisual()?;
        visual.SetContent(chain)?;
        target.SetRoot(&visual)?;
        device.Commit()?;
        Ok(Composition { device, _target: target, visual })
    }
}

/// The keeper's hold on its window and picture: however the keeper leaves
/// (an end, a failure, a panic), the window is hidden first; then, if the
/// presenter is known to be done (or never had anything), the picture is
/// detached and the window destroyed, else both are left hidden for good.
struct Held {
    hwnd: HWND,
    composition: Option<Composition>,
    shared: Option<Arc<Shared>>,
}

impl Drop for Held {
    fn drop(&mut self) {
        let hidden = hide(self.hwnd);
        if let Some(shared) = &self.shared {
            shared.hidden.store(hidden, Ordering::Release);
            shared.stop.store(true, Ordering::Release);
            let deadline = Instant::now() + Duration::from_secs_f64(STOP_S);
            while !shared.finished.load(Ordering::Acquire) && Instant::now() < deadline {
                pump();
                std::thread::sleep(Duration::from_millis(2));
            }
            if !shared.finished.load(Ordering::Acquire) {
                tracing::warn!("loading cover: the presenter did not return; its window is left hidden");
                std::mem::forget(self.composition.take());
                return;
            }
        }
        if let Some(c) = self.composition.take() {
            // SAFETY: our own composition objects.
            unsafe {
                if let Err(e) = c.visual.SetContent(None).and_then(|()| c.device.Commit()) {
                    tracing::warn!(error = %e, "loading cover: the picture was not detached");
                }
            }
        }
        // SAFETY: our own window, on its own thread.
        if let Err(e) = unsafe { DestroyWindow(self.hwnd) } {
            tracing::warn!(error = %e, "loading cover: the window was not destroyed (it stays hidden)");
        }
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: this thread's own messages, at most `MESSAGES` a look.
    unsafe {
        for _ in 0..MESSAGES {
            if !PeekMessageW(&mut message, None, 0, 0, PM_REMOVE).as_bool() {
                break;
            }
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

fn wait(tick: Duration) {
    // SAFETY: a wait on this thread's message queue only.
    unsafe {
        let _ = MsgWaitForMultipleObjects(None, false, tick.as_millis() as u32, QS_ALLINPUT);
    }
    pump();
}

/// Whether the frame recorder is on (it exists in diagnostics builds only).
#[cfg(feature = "diagnostics")]
fn recording() -> bool {
    crate::replay::record::on()
}

/// Whether the frame recorder is on (it exists in diagnostics builds only).
#[cfg(not(feature = "diagnostics"))]
fn recording() -> bool {
    false
}

/// What rules the cover out now (the live state, not settings).
fn refused() -> Option<&'static str> {
    if recording() {
        Some("the frame recorder is on (it hooks the device's queues)")
    } else if crate::streamline::swap_chain_wrapped() {
        Some("the title's swap chain is Streamline's or FSR 3's (not until a measured run says otherwise)")
    } else if super::super::two_queues::on() {
        Some("two queues are on")
    } else {
        None
    }
}

/// The keeper: watches the title's presents, covers the loading section,
/// and ends it. Never returns an error to the title.
fn keeper(plan: Plan) {
    // Parked until the boot video has ended (the title's present only stores that).
    loop {
        match BOOT.load(Ordering::Acquire) {
            PLAYED => break,
            NOT_PLAYED => return,
            _ => std::thread::sleep(WATCH_TICK),
        }
    }
    if let Some(why) = refused() {
        tracing::info!(why, "loading cover: not used");
        return;
    }
    let Some(pinned) = title_now() else {
        tracing::info!("loading cover: not used: the title's window is not known");
        return;
    };
    // This thread maps coordinates as the title's window does.
    // SAFETY: plain queries and this thread's own setting.
    unsafe { SetThreadDpiAwarenessContext(GetWindowDpiAwarenessContext(pinned.hwnd().0)) };
    let mut detector = Detector::new();
    let mut after = 0;
    crate::queue::COVER_WATCHING.store(true, Ordering::Release);
    tracing::info!(video = %plan.name(), "loading cover: watching the title's presents for the world-loading section");
    let event = loop {
        wait(WATCH_TICK);
        if title_now() != Some(pinned) {
            break None;
        }
        match feed(&mut detector, &records(), &mut after) {
            Some(Event::Begin) => break Some(Event::Begin),
            Some(e @ Event::Disarmed) => break Some(e),
            _ => {}
        }
    };
    let why = match event {
        Some(Event::Begin) => cover(&plan, &mut detector, &mut after, pinned),
        Some(_) => "a second of the world was seen first; it does not begin in this run".to_owned(),
        None => "the title's presenter changed".to_owned(),
    };
    crate::queue::COVER_WATCHING.store(false, Ordering::Release);
    tracing::info!(why, "loading cover: over");
}

/// From the begin to the end. Whatever ends it, the window is hidden first
/// ([`Held`]).
fn cover(plan: &Plan, detector: &mut Detector, after: &mut u64, pinned: Title) -> String {
    let began = now();
    if let Some(why) = refused() {
        return why.to_owned();
    }
    let Some(host) = crate::host::host() else { return "no host".to_owned() };
    let Some(mut place) = place_of(pinned.hwnd()) else { return "the title's window has no client area".to_owned() };
    let Ok(hwnd) = window(place) else { return "no window".to_owned() };
    let mut held = Held { hwnd, composition: None, shared: None };
    let shared = Arc::new(Shared::default());
    let surface = match Parts::make(&host.device, hwnd, place.size, Arc::clone(&shared.hidden)) {
        Ok(s) => s,
        Err(e) => {
            // (Nothing was sent: it may go at once.)
            shared.hidden.store(true, Ordering::Release);
            return format!("no swap chain: {e}");
        }
    };
    match composition(hwnd, &surface.chain) {
        Ok(c) => held.composition = Some(c),
        Err(e) => {
            shared.hidden.store(true, Ordering::Release);
            return format!("no composition: {e}");
        }
    }
    held.shared = Some(Arc::clone(&shared));
    let presenting = {
        let (plan, shared) = (plan.clone(), Arc::clone(&shared));
        std::thread::Builder::new().name("victory-cover".into()).spawn(move || {
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| presenter(surface, plan, Arc::clone(&shared))));
            if caught.is_err() {
                shared.fail("the presenter panicked".to_owned());
            }
            shared.finished.store(true, Ordering::Release);
        })
    };
    if presenting.is_err() {
        // (The surface went with the closure, unsent.)
        shared.finished.store(true, Ordering::Release);
        return "no presenter thread".to_owned();
    }
    tracing::info!(w = place.size.0, h = place.size.1, "loading cover: begins");
    let f1_at = f1_presses();
    let mut shown = false;
    let why = loop {
        wait(COVER_TICK);
        let t = now();
        if let Some(why) = look(detector, after, t, began, &shared, f1_at, pinned, place) {
            break why;
        }
        let Some(here) = place_of(pinned.hwnd()) else { break "the title's window has no client area".to_owned() };
        if !in_front(pinned.hwnd()) {
            if shown {
                shown = !hide(hwnd);
                if shown {
                    break "the window could not be hidden".to_owned();
                }
            }
            shared.paused.store(true, Ordering::Release);
            continue;
        }
        if !shared.ready.load(Ordering::Acquire) {
            continue;
        }
        if shown && here.at != place.at {
            // Moved: hidden first, shown again at the new place next look.
            shown = !hide(hwnd);
            if shown {
                break "the window could not be hidden".to_owned();
            }
            shared.paused.store(true, Ordering::Release);
            place = here;
            continue;
        }
        if !shown {
            place = here;
            if let Err(e) = show(hwnd, place) {
                break format!("the window could not be shown: {e}");
            }
            shown = true;
            shared.paused.store(false, Ordering::Release);
        }
    };
    drop(held);
    let counts = shared.counts.try_lock().map(|c| c.clone()).unwrap_or_default();
    tracing::info!(
        why = %why,
        presented = counts.presented,
        busy = counts.busy,
        occluded = counts.occluded,
        shown = counts.decoded_shown,
        skipped = counts.skipped,
        slowest_decode_ms = counts.slowest_decode_ms,
        slowest_scale_ms = counts.slowest_scale_ms,
        slowest_present_ms = counts.slowest_present_ms,
        processor_bytes = counts.held.0,
        card_bytes = counts.held.1,
        covered_s = now() - began,
        "loading cover: ended"
    );
    why
}

/// One look of the keeper while covering: why it ends, if it does. Takes no
/// lock the presenter holds.
#[allow(clippy::too_many_arguments)]
fn look(detector: &mut Detector, after: &mut u64, t: f64, began: f64, shared: &Shared, f1_at: u32, pinned: Title, place: Place) -> Option<String> {
    if WINDOW_PANICKED.load(Ordering::Acquire) {
        return Some("the window's procedure panicked".to_owned());
    }
    // The ceiling, whatever the rule's state.
    if t - began >= CEILING_S {
        return Some(Why::Ceiling.text().to_owned());
    }
    if let Some(Event::End(why)) = feed(detector, &records(), after).or_else(|| detector.tick(t)) {
        return Some(why.text().to_owned());
    }
    if shared.failed.load(Ordering::Acquire) {
        let why = shared.why.try_lock().ok().and_then(|w| w.clone());
        return Some(why.unwrap_or_else(|| "the presenter failed".to_owned()));
    }
    if !shared.ready.load(Ordering::Acquire) && t - began > READY_S {
        return Some("frame 0 was not ready in time".to_owned());
    }
    if f1_presses() != f1_at || crate::SETTINGS_MENU_OPEN.load(Ordering::Relaxed) {
        return Some("F1 (the settings menu)".to_owned());
    }
    if title_now() != Some(pinned) {
        return Some("the title's presenter or window changed".to_owned());
    }
    match place_of(pinned.hwnd()) {
        Some(here) if (here.size, here.monitor, here.dpi) == (place.size, place.monitor, place.dpi) => None,
        Some(_) => Some("the title's window changed size, monitor or DPI".to_owned()),
        None => Some("the title's window has no client area".to_owned()),
    }
}

/// Where the boot video is: not ended, played to its end, or ended otherwise.
const PENDING: u8 = 0;
const PLAYED: u8 = 1;
const NOT_PLAYED: u8 = 2;
static BOOT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(PENDING);
/// A keeper is parked, waiting for [`BOOT`].
static PARKED: AtomicBool = AtomicBool::new(false);

/// The boot video starts: with `VICTORY_BOOT_VIDEO_COVER=1` the keeper starts
/// now, parked until the video has ended (its checks run on its own thread).
/// Unset, nothing at all.
pub(super) fn boot_starts(plan: &Plan) {
    if switch_on() && start(plan, true).is_ok() {
        PARKED.store(true, Ordering::Release);
    }
}

/// The boot video ended (on the title's present): one atomic store for a
/// parked keeper, and nothing else; unset, not even that.
pub(super) fn boot_ended(played: bool) {
    if PARKED.load(Ordering::Acquire) {
        BOOT.store(if played { PLAYED } else { NOT_PLAYED }, Ordering::Release);
    }
}

/// Keepers started (for the test that nothing starts unset).
static STARTED: AtomicU32 = AtomicU32::new(0);

/// Starts the keeper if `asked`.
fn start(plan: &Plan, asked: bool) -> Result<(), Option<&'static str>> {
    if !asked {
        return Err(None);
    }
    let plan = plan.clone();
    let started = std::thread::Builder::new().name("victory-cover-keeper".into()).spawn(move || {
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| keeper(plan))).is_err() {
            crate::queue::COVER_WATCHING.store(false, Ordering::Release);
        }
    });
    match started {
        Ok(_) => {
            STARTED.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        Err(_) => {
            tracing::warn!("loading cover: no thread; not used");
            Err(Some("no thread"))
        }
    }
}

#[cfg(test)]
mod tests;
