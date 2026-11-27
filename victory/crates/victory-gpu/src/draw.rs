//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::resource::RootIndexedView;
use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

#[cfg(all(test, feature = "diagnostics"))]
impl IndexView {
    /// Tests: the title's view of `size` bytes of indices at `address`, in
    /// `format` (PC numbering).
    pub(crate) fn test(address: u64, size: u32, format: u32) -> IndexView {
        IndexView { address, size, format }
    }
}

/// Host states of a depth target, and of a texture shaders read.
const DEPTH_WRITE: u32 = 0x10;
const DEPTH_READ: u32 = 0x20;
const SHADER_RESOURCE: u32 = 0x40 | 0x80;

thread_local! {
    /// While a draw is prepared (`State::bind`, then its targets' states):
    /// the depth target the draw only tests (0: none, or one it writes),
    /// and whether a view its shaders read is of that same texture.
    static TESTED_DEPTH: std::cell::Cell<(usize, bool)> =
        const { std::cell::Cell::new((0, false)) };
    /// While a stage's tables are walked (`State::tables`): whether that
    /// stage's shader can store to buffers (`Translated::stores`).
    static STORES: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    /// While `State::early_textures` runs the texture pass: the
    /// milliseconds and the bytes that run may take, in place of the
    /// pass's own per frame.
    static PASS_BUDGET: std::cell::Cell<Option<(u64, u64)>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
thread_local! {
    /// Tests: what `State::bind` takes the bound pipeline's classification
    /// (`Translated::stores`) to be where it ends the batch of clamped calls,
    /// on this thread, whatever the programs are classed as.
    pub(crate) static STORES_FORCED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// The milliseconds of every frame the pass that uploads textures picked
/// by index takes before a shader that picks by index has asked for it
/// (`VICTORY_EARLY_TEXTURES_MS`, 12; 0: never, as before), and the extra
/// milliseconds one draw may take (`VICTORY_BAKE_TEXTURES_MS`, 1500; 0:
/// nothing special).
///
/// An indexed read needs its textures on the card before the draw that makes
/// it: whatever is not there is read as nothing, once, and stays so for the
/// run. Staging them by then is the runtime's job. (On the console a texture
/// is there when its bytes are.)
fn early_budgets() -> (u64, u64) {
    static MS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        let ms = |name: &str, default: u64| {
            std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default)
        };
        (ms("VICTORY_EARLY_TEXTURES_MS", 12), ms("VICTORY_BAKE_TEXTURES_MS", 1500))
    })
}

/// The listed pixel shaders (`PS_BakeDecorations_*`), by the first 16 hex
/// digits of their hash as the frame listings print it.
const BAKE_PIXEL_SHADERS: [&str; 4] =
    ["c0a7fe77b1862e10", "0d1575a396dfe513", "b7cfd8dda91ebb6b", "280967ad158cee0b"];

/// Since the start: early runs of the texture pass, and those of them
/// made for a bake draw.
pub(crate) static EARLY_TEXTURE_PASSES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Whether a buffer range a shader names through an unordered access view
/// is given to shaders as written (snapshot before the submission, compare
/// and write back after) only if that shader's code can store to buffers.
/// A guest may bind buffers its shaders only read through such views, and
/// every range declared costs a snapshot and a compare of its bytes:
/// gigabytes a frame, most of a second of snapshots and
/// comparing. On by default; `VICTORY_WRITE_SKIP=0` declares
/// every such range, as before.
fn write_skip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WRITE_SKIP").is_none_or(|v| v != "0"))
}

/// A run of indirect records clamped in one pass ([`windows`]): the host
/// copy they came from (its number), their kind and stride, the offset of
/// the first in that copy, how many, and the list's scratch buffer that
/// holds them clamped, record `i` at `i * step`.
#[derive(Clone)]
struct Window {
    copy: u64,
    kind: u32,
    stride: u32,
    /// How far apart the records are in the copy, and in the scratch
    /// buffer: what the title walks them by ([`windows`]).
    step: u32,
    base: u64,
    records: u32,
    scratch: ID3D12Resource,
}

/// The most records clamped in one pass ([`windows`]).
const WINDOW_MOST: u64 = 64;

/// Whether indirect draws that read record after record of one argument
/// buffer are clamped a window at a time (`VICTORY_INDIRECT_WINDOW=1`; off
/// by default).
///
/// A title can issue its mesh draws one record per call, each from the next
/// slot of the same argument buffer: thousands a frame, and before
/// each our clamp ran as a compute dispatch of its own between two waits,
/// so the card went from drawing to compute and back for every record. In
/// a timing run of a lit scene a shadow draw of next to
/// nothing cost over ten microseconds, thousands of records a frame.
///
/// With the switch, when a draw asks for a record a short step after the
/// one drawn before, the clamp takes that record and the ones that step
/// apart after it (16, then twice as many each time, [`WINDOW_MOST`] at
/// most, never past the buffer) into
/// the list's scratch buffer in one pass, and the draws that follow read
/// their record there with no pass of ours, until one of these: a compute
/// dispatch, a direct draw, an indirect call with a count buffer or more
/// than one record, a draw whose shaders can store to buffers, a record
/// outside the window, another host list.
///
/// The step is the one the caller is seen to take from one call to the
/// next, not the stride its command signature names: a call that draws one
/// record never uses the stride, so nothing ties it to how far apart the
/// caller keeps its records, and a step that changes ends a window. Walked
/// by the named stride the window never opened: a run with the switch
/// still clamped thousands of records a frame one at a time.
///
/// What it costs the guard against garbage records: nothing against a
/// hang (every record is bounded as before and charged to the list's
/// budget as before), but the budget is charged when a window is clamped,
/// so also for records of the window's tail that are never drawn. Those
/// can hold garbage (slots the caller has not filled: run 93 spent lists'
/// budgets on them and emptied the draws after), so in a window a record
/// that asks for more than can run is left empty and charges nothing
/// (`crate::indirect::bound_ahead`); a tail record that looks real is
/// charged what it asks, a few thousand of 2^27. And arguments the title's CPU side changes are seen as
/// before, at the next submission.
fn windows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_WINDOW").is_some_and(|v| v == "1"))
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// The host state the bound depth view needs ([`depth_state`]).
    depth_state: u32,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the title set that goes straight onto the host list, kept for
    /// the host list that continues this one after a cut ([`State::replay`]):
    /// whether a topology was set, the render target and depth handles, the
    /// viewports, the scissors, the blend factor and the stencil reference.
    topology_set: bool,
    target_handles: Vec<u64>,
    depth_handle: u64,
    targets_set: bool,
    viewports: Vec<D3D12_VIEWPORT>,
    scissors: Vec<RECT>,
    blend: Option<[f32; 4]>,
    stencil: Option<u32>,
    depth_bounds: Option<[f32; 2]>,
    /// The indirect records clamped last, if later draws may still read
    /// them ([`windows`]).
    window: Option<Window>,
    /// A hash of the title's label of the list (its `SetName`; the label
    /// itself is never kept), kept when the list is reset
    /// ([`State::take`]): its passes are named by it
    /// (`VICTORY_GPU_PASSES=1`). And the fragment of the recording the
    /// list is in (from 0; one more at each cut, [`State::replay`]), and
    /// the times the title has bound targets: each binding begins a pass.
    #[cfg(feature = "diagnostics")]
    pub(crate) label: Option<u32>,
    #[cfg(feature = "diagnostics")]
    fragment: u32,
    #[cfg(feature = "diagnostics")]
    bindings: u64,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, slot count, contents) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u32, u64), u32>,
    /// The dynamic slots the list holds: title heap, first, count; given
    /// back when the list has run, is reset or released.
    dynamic: Vec<(usize, u32, u32)>,
    /// Root SRVs bound as one-view tables in this list: (title heap, title
    /// address; 0 for none) → dynamic slot, and the host copy the slot
    /// shows: later draws of the list bring that one up to date, whatever
    /// copy the address would find by then (`shadow.rs` `tails`).
    root_srv_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// Root UAVs bound as one-view tables in this list, likewise.
    root_uav_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// The list ran and its dynamic slots were given back.
    tables_gone: bool,
    /// Under the slot map: each host list's directory of each view heap's map (host
    /// list, map) → its GPU address ([`State::map_directory`]).
    map_dirs: std::collections::HashMap<(usize, usize), u64>,
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, first, n) in &self.dynamic {
            crate::descriptor::release_dynamic(heap, first, n);
        }
    }
}

/// The most a shader can read through a constant view: 4,096 vectors
/// (65,536 bytes, D3D12_REQ_CONSTANT_BUFFER_ELEMENT_COUNT). The same cap
/// constant and rounding order as the descriptor host's constant size
/// (job/fork-fixes 58e254355), so the two merge cleanly.
const TABLE_CB_FILL_CAP: u32 = 4096 * 16;

/// The bytes a table constant buffer's scratch request and fill take: its
/// declared size, at most [`TABLE_CB_FILL_CAP`]. A table constant buffer is
/// copied into list scratch at its whole declared size today although a
/// shader can read no more than the cap, so an oversized one costs copy
/// bytes every draw for data nothing can read (the native-memory plan,
/// step 2). The 256-byte step a short view is rounded to stays where it
/// was (the scratch and the descriptor both round).
pub(crate) fn table_cbv_fill_len(size: u32) -> u32 {
    size.min(TABLE_CB_FILL_CAP)
}

/// One table constant buffer's dynamic slot as the draw makes it: list
/// scratch of the capped length, the fill deferred, and the host view made
/// from the same capped length (so the view never runs past the scratch
/// piece; the descriptor host rounds it to 256-byte steps and caps it at
/// 4,096 vectors). The scratch's CPU address and the length are returned
/// for fixtures; the draw itself ignores them.
pub(crate) fn table_constant_scratch(
    list: &crate::host::HostList,
    heap: &crate::descriptor::host::HostHeap,
    index: u32,
    address: u64,
    size: u32,
) -> Option<(*mut u8, u32)> {
    let len = table_cbv_fill_len(size);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::TableFillRequested, u64::from(len));
    let (gpu, to) = list.scratch(u64::from(len))?;
    list.defer_table_fill(to, address as *const u8, len as usize);
    heap.constant_at(index, gpu, len);
    Some((to, len))
}

impl State {
    /// The list's submission has run (submissions are waited for): the
    /// dynamic slots its tables were made in are given back, not kept
    /// until the title resets the list. A frame of a scene with heavy table
    /// pressure asks for thousands of tables; held until the resets they ran the 131,072
    /// slots out, and thousands of draws a frame went without their tables.
    /// shortcut: a list run a second time without a reset then reads slots
    /// others may have taken (logged once).
    pub(crate) fn tables_done(&mut self) {
        if self.tables_gone && self.dynamic.is_empty() {
            return;
        }
        if self.tables_gone {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("a list ran again after its descriptor tables were given back"));
        }
        for (heap, first, n) in self.dynamic.drain(..) {
            crate::descriptor::release_dynamic(heap, first, n);
        }
        self.fresh_tables.clear();
        self.root_srv_slots.clear();
        self.root_uav_slots.clear();
        self.map_dirs.clear();
        self.tables_gone = true;
    }
}

/// An experiment on the title's scissor rectangles (`VICTORY_SCISSOR_TEST=full`):
/// they are all widened to the whole target. Where part of a scene is
/// missing and what is left looks cut by one rectangle, one run with this on
/// says whether the missing part is drawn and cut away by such a rectangle, or
/// not drawn at all. Off by default: a rectangle the title sets is part of
/// its picture.
fn scissor_test() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_SCISSOR_TEST").as_deref() == Ok("full"))
}

/// An experiment on mesh draws (`VICTORY_MESH_TEST`): 1 for `skip` (none
/// is issued), 2 for `scissor` (each runs its shaders but rasterises one
/// pixel), 0 otherwise. It tells a card hung by a mesh shader from one hung
/// by what the shader's triangles cover.
fn mesh_test() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_MESH_TEST").as_deref() {
        Ok("skip") => 1,
        Ok("scissor") => 2,
        _ => 0,
    })
}

/// `VICTORY_MESH_TEST=keep:<hex digits>[:<hex digits>]`: only the mesh
/// draws whose mesh shader's hash ends in one of the first digits (and
/// whose digit before it is one of the second, if given) are issued: a
/// search for the shaders that stop the card.
fn mesh_kept(pipeline: usize) -> bool {
    static KEEP: std::sync::OnceLock<Option<(String, String)>> = std::sync::OnceLock::new();
    let Some((last, before)) = KEEP.get_or_init(|| {
        let value = std::env::var("VICTORY_MESH_TEST").ok()?;
        let mut parts = value.strip_prefix("keep:")?.split(':');
        Some((parts.next()?.to_owned(), parts.next().unwrap_or("").to_owned()))
    }) else {
        return true;
    };
    let Some(p) = crate::pipeline::host_pipeline(pipeline as This) else {
        return true;
    };
    let Some(at) = p.name.find("mesh ") else {
        return true;
    };
    let hash = p.name[at + 5..].as_bytes();
    if hash.len() < 16 {
        return true;
    }
    last.as_bytes().contains(&hash[15]) && (before.is_empty() || before.as_bytes().contains(&hash[14]))
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
/// A root SRV's dynamic slot in a list, and the host copy it shows.
type RootSlot = (u32, Option<std::sync::Arc<crate::shadow::Shadow>>);

/// Takes `more` from the time a run may spend bringing in a long queue of
/// textures at more than the usual pace; whether some is left.
/// `VICTORY_TEXTURE_GATE_S` sets it (0: none).
fn catching_up(more: std::time::Duration) -> bool {
    static SPENT: AtomicU64 = AtomicU64::new(0);
    static TOTAL: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let total = *TOTAL.get_or_init(|| {
        let seconds = std::env::var("VICTORY_TEXTURE_GATE_S")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(60);
        seconds * 1000
    });
    let more = more.as_millis() as u64;
    SPENT.fetch_add(more, Ordering::Relaxed) + more < total
}

/// Buffer views of the heap whose slot showed another host buffer than
/// the copy the pass brought up to date for them, since the start
/// (diagnostics; for the memory line). It stays 0 while the pass and the
/// slots agree; with `VICTORY_VIEW_EXTENT=0` it counts the views that were
/// read stale.
#[cfg(feature = "diagnostics")]
pub(crate) static HEAP_VIEWS_ELSEWHERE: AtomicU64 = AtomicU64::new(0);
/// Passes over the view heap's buffer views for shaders that pick them by
/// index ([`State::heap_buffers`]), and the microseconds they took.
pub(crate) static HEAP_BUFFER_PASSES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_BUFFER_MICROS: AtomicU64 = AtomicU64::new(0);
/// Textures brought up to date for shaders that pick them by index
/// ([`State::heap_textures`]), the bytes of their memory, and the slots
/// still waiting.
pub(crate) static HEAP_TEXTURES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURE_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURES_WAITING: AtomicU64 = AtomicU64::new(0);
/// Passes that made no texture because the card had no room
/// (`crate::resource::room`), for the memory line.
pub(crate) static HEAP_TEXTURES_FULL: AtomicU64 = AtomicU64::new(0);

/// Looks at uploaded textures picked by index that found the memory as it
/// was, that found it changed (uploaded again), since the start; and the
/// slots that wait for such a look now.
pub(crate) static HEAP_TEXTURES_RECHECKED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Whether textures picked by index are looked at again after their upload
/// (`VICTORY_HEAP_RECHECK=0`: no).
fn heap_recheck() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_RECHECK").is_none_or(|v| v != "0"))
}

/// Uploads of textures already on the card made while it had no room for
/// new ones, since the start.
pub(crate) static HEAP_TEXTURES_REUPLOADED_FULL: AtomicU64 = AtomicU64::new(0);

/// Whether a pass over the textures picked by index on a full card holds
/// back the slots that need a new host copy (`VICTORY_FULL_CARD_PASS=0`:
/// no, as before: they stay due, each pass looks at the same ones again
/// and stops at the 257th). Held back, such a slot is looked at again
/// after about four seconds or as soon as the card has room, and the pass
/// goes on to the slots behind it: the ones whose host copy exists and
/// only wants bringing up to date, and the ones whose memory went (letting
/// their host copies go is what makes room). What is uploaded while the
/// card has room is the same either way.
fn full_card_pass() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FULL_CARD_PASS").is_none_or(|v| v != "0"))
}

/// Whether a texture that has its host copy is still brought up to date
/// while the card has no room for new ones (`VICTORY_REUPLOAD_WHEN_FULL=0`:
/// no).
fn reupload_when_full() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REUPLOAD_WHEN_FULL").is_none_or(|v| v != "0"))
}
/// Times a placed texture's slot was passed over because its memory is not
/// mapped.
pub(crate) static HEAP_TEXTURES_AWAY: AtomicU64 = AtomicU64::new(0);
/// Slots the pass found up to date already: their texture is uploaded by
/// the draws that read it through a table (it has a staging buffer).
pub(crate) static HEAP_TEXTURES_TABLED: AtomicU64 = AtomicU64::new(0);
/// The queue of textures picked by index at the last pass
/// (`TextureQueue::census`): slots in it, due, waiting for their memory,
/// uploaded in part, and the frames the oldest due one has waited.
pub(crate) static HEAP_TEXTURES_QUEUE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Since the start: looks at a slot uploaded in part that found no more of
/// its memory mapped; uploads of textures not all mapped; and of those,
/// the ones whose first page was not there (asked by the first page alone,
/// as before, they counted as not there at all).
pub(crate) static HEAP_TEXTURES_PARTLY: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Lines at most in any second about textures picked by index entering a
/// state the frame line counts (`State::say_stuck`): a world's first frames
/// put tens of thousands of slots there at once.
#[cfg(feature = "diagnostics")]
const STUCK_PER_SECOND: usize = 100;

/// The moments (ms) of the lines said in the last second, oldest first.
#[cfg(feature = "diagnostics")]
static STUCK_SAID: std::sync::Mutex<std::collections::VecDeque<u64>> = std::sync::Mutex::new(std::collections::VecDeque::new());

/// Lines not said for the limit since the frame line last said so.
#[cfg(feature = "diagnostics")]
static STUCK_UNSAID: AtomicU64 = AtomicU64::new(0);

/// Whether a line about a slot entering a state may be said now, on any
/// thread: the clock is read under the lock, and at most
/// [`STUCK_PER_SECOND`] lines lie in any second ([`stuck_admits`]).
#[cfg(feature = "diagnostics")]
fn stuck_due() -> bool {
    let mut said = STUCK_SAID.lock().unwrap_or_else(|e| e.into_inner());
    let now = crate::host::now_ms();
    if stuck_admits(&mut said, now) {
        return true;
    }
    STUCK_UNSAID.fetch_add(1, Ordering::Relaxed);
    false
}

/// Whether a line may be said at `now` (ms; never before the last said)
/// with `said` the moments of the lines said lately: at most
/// [`STUCK_PER_SECOND`] in any 1,000 ms.
#[cfg(feature = "diagnostics")]
fn stuck_admits(said: &mut std::collections::VecDeque<u64>, now: u64) -> bool {
    let now = now.max(said.back().copied().unwrap_or(0));
    while said.front().is_some_and(|&t| t + 1000 <= now) {
        said.pop_front();
    }
    if said.len() >= STUCK_PER_SECOND {
        return false;
    }
    said.push_back(now);
    true
}

/// With the frame line, every five seconds: the lines about slots that
/// were not said for the limit since, if any.
#[cfg(feature = "diagnostics")]
pub(crate) fn stuck_unsaid_line() {
    let n = STUCK_UNSAID.swap(0, Ordering::Relaxed);
    if n > 0 {
        tracing::info!(
            "texture slots picked by index that entered a state the frame line counts, not said: {n} since the last line \
             (at most {STUCK_PER_SECOND} such lines in any second)"
        );
    }
}

/// The line for a title slot `slot` that entered `state` in `frame`: its
/// texture as the pass saw it when it decided (`texture`: the description's
/// words, the start of its memory and its bytes from the first mip the view
/// shows), `mapped` of those bytes there if known. Fixed words and numbers
/// alone, one line. Which draw last bound it is unknown: a slot picked by
/// index is bound through the heap, and no draw is tracked to it.
#[cfg(feature = "diagnostics")]
fn stuck_text(state: &str, slot: u32, texture: Option<&([u32; 14], u64, u64)>, mapped: Option<u64>, frame: u64) -> String {
    use std::fmt::Write;
    let mut t = format!("texture slot {state}: slot {slot}");
    match texture {
        Some((r, address, len)) => {
            let width = u64::from(r[4]) | u64::from(r[5]) << 32;
            // D3D12's dimension 4 is a volume: its third size is a depth.
            let third = if r[0] == 4 { "depth" } else { "slices" };
            let _ = write!(
                t,
                ", {width}x{} (dimension {}), {} {third}, {} mips, format {}, memory from {address:#x} ({len:#x} bytes from the \
                 first mip its view shows",
                r[6],
                r[0],
                r[7] & 0xffff,
                r[7] >> 16,
                r[8]
            );
            match mapped {
                Some(m) => {
                    let _ = write!(t, ", {m:#x} of them mapped)");
                }
                None => t.push(')'),
            }
        }
        None => t.push_str(", its texture not known (no title texture, or a view of no memory)"),
    }
    let _ = write!(
        t,
        "; the program and draw that last bound it: unknown (a slot picked by index is bound through the heap, and no draw \
         is tracked to it); found by the pass over the heap in frame {frame}"
    );
    t
}

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

/// Title draws and mesh dispatches recorded so far, direct and indirect,
/// whether they ran on the host or were dropped; never compute dispatches,
/// run or dropped. What the heavy scene's detector counts
/// (`crate::capture`, `VICTORY_CAPTURE_HEAVY_FRAMES`): one count however a
/// build fares with the frame's pipelines. Counted only in diagnostics
/// builds with that list and a capture folder ([`attempt`]).
#[cfg(feature = "diagnostics")]
static ATTEMPTS: AtomicU64 = AtomicU64::new(0);

/// See [`ATTEMPTS`].
#[cfg(feature = "diagnostics")]
pub(crate) fn attempted() -> u64 {
    ATTEMPTS.load(Ordering::Relaxed)
}

/// A title draw or mesh dispatch is recorded: counted for the heavy
/// scene's detector when it is on (`crate::capture::heavy_counting`).
/// Otherwise the draw path pays nothing for it: no shared atomic add in a
/// build without diagnostics or a run without the list (Sol's review,
/// finding 8).
#[inline]
fn attempt() {
    #[cfg(feature = "diagnostics")]
    if crate::capture::heavy_counting() {
        ATTEMPTS.fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// Whether a live skip switch (`VICTORY_LIVE`) drops this call: a compute
/// dispatch asks `LIVE_SKIP_DISPATCH`, a draw `LIVE_SKIP_DRAW`, of the
/// pipeline id the log names. One check at the top of the path: off unless
/// the run set `VICTORY_LIVE`.
#[inline]
fn live_skip(pipeline: This, dispatch: bool) -> bool {
    // An empty list skips nothing: no pipeline id is made for it (a live
    // session's frame times are not to pay for the tool that measures them).
    if !crate::live::on() || !crate::live::skipping(dispatch) {
        return false;
    }
    let Some(id) = crate::pipeline::live_id(pipeline) else {
        return false;
    };
    if dispatch {
        crate::live::skip_dispatch(&id)
    } else {
        crate::live::skip_draw(&id)
    }
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB
/// (cut at their buffer's end), other views as all that is left of their
/// buffer. (They were taken as 1 MiB: most of the mesh vertex
/// streams of a frame with heavy geometry streaming reach further, and what lay past the first
/// megabyte was never brought up to date - meshes without bodies, exploded
/// triangles. `VICTORY_ROOT_VIEW_MB` sets a limit again.)
fn root_view(list: &HostList, address: u64, srv: bool) -> Option<u64> {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("VICTORY_ROOT_VIEW_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(u64::MAX, |mb| mb << 20)
    });
    let shadow = crate::shadow::find(address, 1)?;
    let len = shadow
        .left(address)
        .min(if srv { limit } else { 1 << 16 });
    let gpu = shadow.gpu(address);
    if srv || gpu % 256 == 0 {
        shadow.refresh(list, address, len);
        return Some(gpu);
    }
    let (gpu, to) = list.scratch(len)?;
    list.defer_fill(to, address as *const u8, len as usize);
    Some(gpu)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// Binding under the slot map (`VICTORY_SLOT_MAP=1`).
#[path = "draw_map.rs"]
mod map;

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        self.topology_set = true;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    /// What the list set, taken for its recording's end; what the title
    /// sets is gone for the next recording, the list's label stays.
    pub(crate) fn take(&mut self) -> State {
        let taken = std::mem::take(self);
        #[cfg(feature = "diagnostics")]
        {
            self.label = taken.label;
        }
        taken
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.target_handles.clear();
        self.target_handles.extend_from_slice(handles);
        self.depth_handle = depth;
        self.targets_set = true;
        #[cfg(feature = "diagnostics")]
        {
            self.bindings += 1;
        }
        self.targets.clear();
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                // A typed null view (or a target without a host copy) still
                // takes its slot: left out, the next live target would move
                // into it and the pixel shader's outputs reach the wrong
                // targets.
                // shortcut: a view without a description or a format has no
                // null view to stand for it and is left out, as before.
                let null = record
                    .desc()
                    .filter(|d| d.Format.0 != 0 && d.ViewDimension.0 != 0);
                if let Some(desc) = null {
                    rtvs.push(host.rtv(None, Some(&desc)));
                }
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(list, record.resource);
            rtvs.push(host.rtv(Some(texture), record.desc().as_ref()));
        }
        self.depth = 0;
        self.depth_state = DEPTH_WRITE;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            self.depth_state = depth_state(texture, record.depth_desc().as_ref());
            if crate::resource::depth_and_unordered(record.resource) {
                crate::resource::DEPTH_UNORDERED[1].fetch_add(1, Ordering::Relaxed);
            }
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(list, record.resource);
            Some(host.dsv(texture, record.depth_desc().as_ref()))
        });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// Viewports, scissors, the blend factor and the stencil reference: set
    /// on the host list at once, and kept for [`State::replay`].
    pub(crate) fn set_viewports(&mut self, list: &HostList, viewports: &[D3D12_VIEWPORT]) {
        self.viewports.clear();
        self.viewports.extend_from_slice(viewports);
        crate::draw::viewports(list, viewports);
    }

    pub(crate) fn set_scissors(&mut self, list: &HostList, rects: &[RECT]) {
        self.scissors.clear();
        self.scissors.extend_from_slice(rects);
        if scissor_test() {
            // The experiment: every rectangle the title sets is widened to
            // the largest target there is, so nothing is cut by one. What
            // the listing shows (`op_text`) stays the title's own.
            let full = RECT {
                left: 0,
                top: 0,
                right: 16384,
                bottom: 16384,
            };
            crate::draw::scissors(list, &vec![full; rects.len()]);
            return;
        }
        crate::draw::scissors(list, rects);
    }

    pub(crate) fn set_blend_factor(&mut self, list: &HostList, factor: Option<&[f32; 4]>) {
        self.blend = factor.copied();
        crate::draw::blend_factor(list, factor);
    }

    pub(crate) fn set_stencil_ref(&mut self, list: &HostList, reference: u32) {
        self.stencil = Some(reference);
        crate::draw::stencil_ref(list, reference);
    }

    pub(crate) fn set_depth_bounds(&mut self, list: &HostList, min: f32, max: f32) {
        self.depth_bounds = Some([min, max]);
        crate::draw::depth_bounds(list, min, max);
    }

    /// `list` is a new host list that continues the title's list (it was
    /// cut at a wait for a memory value, `crate::list::cut`): a new host
    /// list has nothing set, so what the title set so far is set again.
    /// The pipeline, the root signatures and the heaps are set by the next
    /// draw or dispatch (`bind`, which is told here that the host list has
    /// none); root arguments, vertex and index buffers are set by every
    /// draw anyway. The dynamic descriptor slots and the tables made in
    /// them stay: the parts before the cut have not run yet, and they are
    /// given back when the title's list is reset.
    /// shortcut: the render target views are made again from the handles'
    /// descriptors as they are now, not as they were when the title set
    /// the targets (a descriptor rewritten in between names another view).
    pub(crate) fn replay(&mut self, list: &HostList) {
        // A cut: the recording goes on in another fragment.
        #[cfg(feature = "diagnostics")]
        {
            self.fragment += 1;
        }
        self.host_pipeline = 0;
        self.host_roots = [0; 2];
        self.host_heaps = [0; 2];
        // The tables made afresh name constants copied into the list before
        // (its upload memory, reset with it, possibly before this one runs):
        // this list makes its own. Their slots stay held (`dynamic`) until
        // the title's list has run.
        self.fresh_tables.clear();
        if self.topology_set {
            self.set_topology(list, self.topology);
        }
        if self.targets_set {
            let handles = std::mem::take(&mut self.target_handles);
            self.set_targets(list, &handles, self.depth_handle);
        }
        if !self.viewports.is_empty() {
            crate::draw::viewports(list, &self.viewports);
        }
        if !self.scissors.is_empty() {
            let rects = std::mem::take(&mut self.scissors);
            self.set_scissors(list, &rects);
        }
        if let Some(factor) = self.blend {
            crate::draw::blend_factor(list, Some(&factor));
        }
        if let Some(reference) = self.stencil {
            crate::draw::stencil_ref(list, reference);
        }
        if let Some([min, max]) = self.depth_bounds {
            crate::draw::depth_bounds(list, min, max);
        }
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            // The host's own ([`Self::host_word`]): no root argument.
            (_, RootType::Other(_)) => 0,
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// A user-data dword the host computes for a translated shader
    /// (`victory_shader::emit::USER_MEMORY_BASE`, `USER_VIEW_ADDRESS`).
    fn host_word(
        &self,
        compute: bool,
        kind: u32,
        slot: u8,
        dword: u32,
        memory: Option<&crate::memory::Ready>,
    ) -> u32 {
        use victory_shader::emit::{USER_GROUP_BASE, USER_MEMORY_BASE, USER_VIEW_ADDRESS, USER_VIEW_BIAS};
        // Set between the parts of a large dispatch (`group_base`).
        if kind == USER_GROUP_BASE {
            return 0;
        }
        if kind == USER_MEMORY_BASE {
            return memory.map_or(0, |m| m.base);
        }
        if kind == USER_VIEW_BIAS {
            // The view begins on 16 bytes of the copy the address finds
            // (`root_uav`, `root_srv_table`).
            // shortcut: found again here, not taken from the list's slot.
            return match self.roots_ref(compute).get(usize::from(slot)) {
                Some(&Root::View(address)) => crate::shadow::find(address, 1).map_or(0, |s| {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    ((s.gpu(address) - base) & 15) as u32
                }),
                _ => 0,
            };
        }
        if kind != USER_VIEW_ADDRESS && kind != USER_VIEW_ADDRESS + 1 {
            once!(kind, "a shader reads a host value this build does not know");
            return 0;
        }
        // The title address of the buffer view at that dword of the table.
        let address = match self.roots_ref(compute).get(usize::from(slot)) {
            Some(&Root::Table(table)) => crate::descriptor::view(table + u64::from(dword) * 4)
                .and_then(|view| crate::descriptor::host::buffer_range(&view))
                .map_or(0, |range| range.0),
            _ => 0,
        };
        if address == 0 {
            once!(slot, dword, "a shader takes the address of a buffer view that is not there: it reads 0");
            // Counted once per address: its low dword.
            #[cfg(feature = "diagnostics")]
            if kind == USER_VIEW_ADDRESS {
                crate::null_trace::note(crate::null_trace::Kind::NoAddress, || {
                    format!(
                        "{} root {slot} dword {dword}",
                        crate::pipeline::shaders_of(self.pipeline as This)
                    )
                });
            }
        }
        if kind == USER_VIEW_ADDRESS {
            address as u32
        } else {
            (address >> 32) as u32
        }
    }

    /// For a shader that reads title memory by address
    /// ([`crate::memory`]): its directory, fault notes and view slots,
    /// with the pages of the root views the list has set made ready.
    fn memory_ready(&self, list: &HostList, compute: bool, writes: bool) -> Option<crate::memory::Ready> {
        let heap = match self.heaps[0] {
            0 => None,
            h => crate::descriptor::heap(h as This).host(),
        }?;
        let (mut addresses, mut constants) = (Vec::new(), Vec::new());
        for root in self.roots_ref(compute) {
            match root {
                Root::View(address) => addresses.push(*address),
                Root::Constants(values) => constants.extend_from_slice(values),
                _ => {}
            }
        }
        crate::memory::ready(list, heap, &addresses, &constants, writes)
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            return false;
        }
        if p.rects && self.topology != RECTANGLES {
            // Its geometry stage would turn each triangle into a rectangle.
            once!("a rectangle pipeline drawn with another topology");
        }
        if !self.bind(list, p, false) {
            return false;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find_any(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find_any(v.address, u64::from(v.size)) else {
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(self.depth as This, false, true);
            // A view that writes neither plane keeps the texture readable:
            // the depth read state, together with the shader read states
            // if this draw's shaders read the same texture (the write state
            // put it out of the state those views need, and a move to it
            // and back for every such draw).
            let state = if self.depth_state == DEPTH_READ {
                let (tested, read) = TESTED_DEPTH.get();
                if read && tested == self.depth {
                    DEPTH_READ | SHADER_RESOURCE
                } else {
                    DEPTH_READ
                }
            } else {
                DEPTH_WRITE
            };
            crate::list::ensure_state(list, self.depth as This, state);
        }
        true
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a root SRV it reads
    /// could not be given a view.
    pub(crate) fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        // Where a pass begins, before anything of ours for this call
        // (`VICTORY_GPU_PASSES=1`): a dispatch after draws, or a draw after
        // a dispatch or after the title bound targets again.
        #[cfg(feature = "diagnostics")]
        list.pass_mark(p, if compute { 0 } else { self.bindings + 1 }, self.label, self.fragment);
        let l = &list.list;
        // A shader translated with `VICTORY_SLOT_MAP=1` reads the slot map: it
        // runs with a view heap under the map bound, and is skipped otherwise,
        // never bound to slots it would read as other views.
        if p.stages.iter().any(|s| crate::descriptor::slot_map::reads_map(&s.translated().host.root_srvs))
            && self.mapped_heap().is_none()
        {
            map::skipped(&p.name, "no view heap under the map is bound");
            return false;
        }
        // Textures picked by index, before anything asks for them.
        self.early_textures(list, p, compute);
        if crate::host::walls_trace() {
            self.walls_census(p, compute);
        }
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        if self.host_heaps != self.heaps {
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { l.SetDescriptorHeaps(&heaps) };
            self.host_heaps = self.heaps;
            self.host_roots = [0; 2];
        }
        // The depth target a draw only tests, for the views of it that the
        // draw's shaders read (`readable`).
        let tested = if !compute && self.depth_state == DEPTH_READ {
            self.depth
        } else {
            0
        };
        TESTED_DEPTH.set((tested, false));
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        for stage in &p.stages {
            let t = stage.translated();
            STORES.set(t.stores);
            let made = self.tables(list, &t.used, &t.host.root_indexed_views, compute);
            STORES.set(true);
            if made.iter().any(|m| m.1 == u32::MAX) {
                once!("a draw whose descriptor table found no free slots is dropped");
                return false;
            }
            // What the shader stores through root views and raw addresses is
            // copied back into title memory afterwards.
            let writes = (t.stores || !write_skip())
                && !crate::pipeline::builder_quarantined(self.pipeline as This);
            // Title memory by address: what the shader finds it by.
            let memory = if t.host.root_srvs.contains(&victory_shader::emit::SLOT_MEMORY) {
                let Some(ready) = self.memory_ready(list, compute, writes) else {
                    once!("a shader that reads title memory by address without a view heap to hold its views: the draw is dropped");
                    return false;
                };
                Some(ready)
            } else {
                None
            };
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    if let RootType::Other(kind) = u.root {
                        values[u.index as usize] =
                            self.host_word(compute, kind, u.slot, u.dword, memory.as_ref());
                        continue;
                    }
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None => self.user_value(compute, u.slot, u.root, u.dword),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, srv) in [(&stage.root_cbvs, false), (&stage.root_srvs, true)] {
                for &(slot, param) in views {
                    if srv && slot == victory_shader::emit::SLOT_MEMORY {
                        // Ours: the directory of title memory by address.
                        if let Some(m) = &memory {
                            // SAFETY: an open list; the directory lives as
                            // long as the process and the list holds it.
                            unsafe {
                                if compute {
                                    l.SetComputeRootShaderResourceView(param, m.directory);
                                } else {
                                    l.SetGraphicsRootShaderResourceView(param, m.directory);
                                }
                            }
                        }
                        continue;
                    }
                    if srv
                        && (slot == victory_shader::emit::slot_map::MAP_DIR
                            || slot == victory_shader::emit::slot_map::MAP_POOL)
                    {
                        // Ours: the slot map's pool, and this list's directory
                        // of it (filled when the list is submitted).
                        let gpu = self.mapped_heap().and_then(|(_, _, m)| {
                            if slot == victory_shader::emit::slot_map::MAP_POOL {
                                Some(m.pool_gpu())
                            } else {
                                self.map_directory(list, m)
                            }
                        });
                        let Some(gpu) = gpu else {
                            map::skipped(&p.name, "no directory of the map could be had for the list");
                            return false;
                        };
                        // SAFETY: an open list; the list holds the pool and its
                        // scratch until it is reset.
                        unsafe {
                            if compute {
                                l.SetComputeRootShaderResourceView(param, gpu);
                            } else {
                                l.SetGraphicsRootShaderResourceView(param, gpu);
                            }
                        }
                        continue;
                    }
                    if srv && slot == victory_shader::emit::SLOT_STRIDES && stage.strides_table {
                        // The same, as a one-view table: the root
                        // descriptor did not fit the root signature
                        // (`VICTORY_D16_INTEGER_VIEWS=1`).
                        if !self.strides_table(list, writes, param, compute) {
                            return false;
                        }
                        continue;
                    }
                    if srv && slot == victory_shader::emit::SLOT_STRIDES {
                        // Ours, not the title's: the shader picks buffers
                        // from the view heap by an index it computes.
                        match self.heap_buffers(list, writes) {
                            // SAFETY: an open list; the table lives as long
                            // as the heap the list holds.
                            Some(gpu) => unsafe {
                                if compute {
                                    l.SetComputeRootShaderResourceView(param, gpu);
                                } else {
                                    l.SetGraphicsRootShaderResourceView(param, gpu);
                                }
                            },
                            None => once!("a shader picks buffers from a view heap that is not set"),
                        }
                        continue;
                    }
                    let tabled = srv && crate::pipeline::host::bounded_root_srvs();
                    let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                    else {
                        once!(slot, "a shader reads a root view the title did not set");
                        // A table left unset is undefined: a null view.
                        if tabled && !self.root_srv_table(list, None, param, compute, false) {
                            return false;
                        }
                        continue;
                    };
                    // The shader adds the bytes a raw view begins early by.
                    let biased = srv
                        && t.host.user_data.iter().any(|u| {
                            u.slot == slot
                                && u.root == RootType::Other(victory_shader::emit::USER_VIEW_BIAS)
                        });
                    if tabled {
                        if !self.root_srv_table(list, Some(address), param, compute, biased) {
                            return false;
                        }
                        continue;
                    }
                    let Some(mut gpu) = root_view(list, address, srv) else {
                        continue;
                    };
                    if biased {
                        gpu -= u64::from(self.host_word(
                            compute,
                            victory_shader::emit::USER_VIEW_BIAS,
                            slot,
                            0,
                            None,
                        ));
                    }
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, srv) {
                            (false, false) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, true) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (true, false) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, true) => l.SetComputeRootShaderResourceView(param, gpu),
                        }
                    }
                }
            }
            for &(slot, param) in &stage.root_uavs {
                if slot == victory_shader::emit::SLOT_FAULTS {
                    // Ours: where shaders note the pages they found empty.
                    if let Some(m) = &memory {
                        // SAFETY: an open list; the buffer lives as long as
                        // the process and the list holds it.
                        unsafe {
                            if compute {
                                l.SetComputeRootUnorderedAccessView(param, m.faults);
                            } else {
                                l.SetGraphicsRootUnorderedAccessView(param, m.faults);
                            }
                        }
                    }
                    continue;
                }
                let address = match self.roots_ref(compute).get(usize::from(slot)) {
                    Some(&Root::View(address)) => Some(address),
                    _ => {
                        once!(slot, "a shader writes through a root view the title did not set");
                        None
                    }
                };
                if !self.root_uav(list, address, param, compute, writes) {
                    return false;
                }
            }
        }
        // Depth textures it writes through unordered access: their colour
        // twins hold what they lack from here on.
        list.twins_written();
        // What the draw's textures are, for the pixel shaders the binding
        // trace names (reads only; logged at the next present).
        #[cfg(feature = "diagnostics")]
        if !compute && crate::bind_trace::on() {
            self.bind_trace(p);
        }
        // The texture pack's log (`VICTORY_TEXTURE_OVERRIDE_LOG`): the large
        // textures a draw over most of the screen reads (reads only).
        #[cfg(feature = "diagnostics")]
        if !compute && crate::texture_pack::logging() {
            self.pack_log(p);
        }
        // Whether what is bound can store to buffer memory: the census of
        // runs of clamped calls (`crate::runs`, (c)), and the clamp of runs,
        // whose batch ends here (the calls after it read records it may
        // have written; a call of the batch that is this draw has its own).
        let stores = p.stages.iter().any(|stage| stage.translated().stores);
        #[cfg(test)]
        let stores = STORES_FORCED.get().unwrap_or(stores);
        #[cfg(feature = "diagnostics")]
        list.runs(|runs| runs.bound(stores));
        if stores {
            list.batch_end();
        }
        true
    }

    /// Binds a root UAV: a raw unordered access view of the title buffer's
    /// host copy from `address` on, as a one-view table (a null view
    /// without an address, without a host copy shaders can write, or at an
    /// address not on 16 bytes), or as a root descriptor with
    /// `VICTORY_ROOT_SRV_DIRECT=1` (the dispatch is dropped then if there
    /// is nothing to bind). With `writes`, what the shader stored is copied
    /// back into title memory when the list has run. False if the draw
    /// must be dropped.
    /// shortcut: a root view has no size: the view, the refresh and the
    /// copy back cover all that is left of the title's buffer
    /// (`VICTORY_ROOT_UAV_MB` sets a limit for all three).
    fn root_uav(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        writes: bool,
    ) -> bool {
        const UNORDERED: i32 = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
        static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let limit = *LIMIT.get_or_init(|| {
            std::env::var("VICTORY_ROOT_UAV_MB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map_or(u64::MAX, |mb| mb << 20)
        });
        let find = |a: u64| {
            let s = crate::shadow::find(a, 1)?;
            // SAFETY: a live buffer.
            let flags = unsafe { s.buffer.GetDesc() }.Flags;
            if flags.0 & UNORDERED == 0 {
                once!("a root UAV of a buffer whose host copy shaders cannot write");
                return None;
            }
            Some(s)
        };
        let ready = |a: u64, s: &std::sync::Arc<crate::shadow::Shadow>| {
            let len = s.left(a).min(limit);
            s.refresh(list, a, len);
            if writes {
                s.gpu_writes(list, a, len);
            }
        };
        if !crate::pipeline::host::bounded_root_srvs() {
            let Some((a, s)) = address.and_then(|a| Some((a, find(a)?))) else {
                once!("a root UAV with nothing to bind: the draw is dropped");
                return false;
            };
            ready(a, &s);
            // From the 16 bytes below: the shader adds the rest.
            // SAFETY: a live buffer.
            let base = unsafe { s.buffer.GetGPUVirtualAddress() };
            let gpu = base + ((s.gpu(a) - base) & !15);
            // SAFETY: an open list; the address is in a live buffer.
            unsafe {
                if compute {
                    list.list.SetComputeRootUnorderedAccessView(param, gpu);
                } else {
                    list.list.SetGraphicsRootUnorderedAccessView(param, gpu);
                }
            }
            return true;
        }
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root UAV without a bound view heap: the draw is dropped");
            return false;
        };
        let key = (std::ptr::from_ref(heap) as usize, address.unwrap_or(0));
        let (index, shadow) = match self.root_uav_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| find(a).map(|s| (a, s)));
                let Some(index) = host.dynamic(1) else {
                    return false;
                };
                self.dynamic.push((key.0, index, 1));
                if let Some((a, s)) = &shadow {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    let offset = s.gpu(*a) - base;
                    // Raw views start on 16 bytes: from the 16 bytes
                    // below, and the shader adds the rest to its offsets
                    // (`victory_shader::emit::USER_VIEW_BIAS`). (An address
                    // not on 16 bytes got a null view at first: the
                    // builder's counters are at such addresses.)
                    let bias = offset & 15;
                    host.raw_uav_at(index, &s.buffer, offset - bias, s.left(*a).min(limit) + bias);
                }
                if shadow.is_none() {
                    host.null_raw_uav_at(index);
                }
                self.root_uav_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                (index, shadow)
            }
        };
        if let Some((a, s)) = &shadow {
            ready(*a, s);
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// Binds a root SRV as a one-view table
    /// ([`crate::pipeline::host::bounded_root_srvs`]): a raw view of the
    /// title buffer's host copy from `address` to the copy's end, or a null
    /// view (no address, no host copy, or an address not on 16 bytes). One
    /// dynamic slot per address per list, given back with the list's
    /// others. False if no slot could be had or no view heap is bound.
    fn root_srv_table(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        biased: bool,
    ) -> bool {
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root SRV without a bound view heap: the draw is dropped");
            return false;
        };
        // A view that begins early is another view than one that does not
        // (title addresses stay below bit 63).
        let key = (
            std::ptr::from_ref(heap) as usize,
            address.unwrap_or(0) | u64::from(biased) << 63,
        );
        let (index, shadow) = match self.root_srv_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| crate::shadow::find(a, 1).map(|s| (a, s)));
                let Some(index) = host.dynamic(1) else {
                    return false;
                };
                self.dynamic.push((key.0, index, 1));
                self.root_srv_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                match &shadow {
                    Some((a, s)) => {
                        // SAFETY: a live buffer.
                        let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                        let offset = s.gpu(*a) - base;
                        if biased {
                            // From the 16 bytes below; the shader adds the
                            // rest (`victory_shader::emit::USER_VIEW_BIAS`).
                            let bias = offset & 15;
                            host.raw_at(index, &s.buffer, offset - bias, s.left(*a) + bias);
                        } else if offset.is_multiple_of(16) {
                            host.raw_at(index, &s.buffer, offset, s.left(*a));
                        } else {
                            // Raw views start on 16 bytes; one from the
                            // aligned start would shift every read.
                            once!(
                                address = format_args!("{a:#x}"),
                                "a root SRV not on 16 bytes reads zeros"
                            );
                            host.null_raw_at(index);
                        }
                    }
                    None => host.null_raw_at(index),
                }
                (index, shadow)
            }
        };
        // As for root descriptors (`root_view`): all that is left of the
        // buffer is brought up to date. The first 1 MiB only left what lay
        // past it stale (most mesh vertex streams reach further).
        if let Some((a, s)) = &shadow {
            s.refresh(list, *a, s.left(*a));
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// Binds the view heap's slot table at root parameter `param` as a
    /// one-view table (`VICTORY_D16_INTEGER_VIEWS=1`, where its root
    /// descriptor did not fit the root signature: `pipeline::host`'s
    /// `fit_strides`): a raw view of the whole table in a dynamic slot of the
    /// heap, made once per heap for this state, after [`Self::heap_buffers`]
    /// has made the heap's buffers ready as for the root descriptor. False
    /// (the draw is dropped) without a view heap or a free slot.
    fn strides_table(&mut self, list: &HostList, writes: bool, param: u32, compute: bool) -> bool {
        if self.heap_buffers(list, writes).is_none() {
            once!("a shader picks buffers from a view heap that is not set");
            return false;
        }
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return false;
        };
        let Some((host, buffer)) = heap.host().and_then(|h| Some((h, h.strides_buffer()?.clone()))) else {
            return false;
        };
        // Its own key: title addresses stay below bit 63, and a biased one
        // sets bit 63 alone (`root_srv_table`).
        let key = (std::ptr::from_ref(heap) as usize, u64::MAX);
        let index = match self.root_srv_slots.get(&key) {
            Some((index, _)) => *index,
            None => {
                let Some(index) = host.dynamic(1) else {
                    once!("no free dynamic slot for the slot table's view: the draw is dropped");
                    return false;
                };
                self.dynamic.push((key.0, index, 1));
                self.root_srv_slots.insert(key, (index, None));
                let bytes = u64::from(victory_shader::emit::SLOT_STRIDES_COUNT) * 4;
                host.raw_at(index, &buffer, 0, bytes);
                index
            }
        };
        // (The list holds the table's buffer: `heap_buffers`.)
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// For a shader that picks buffers from the view heap by an index it
    /// computes (SM 6.6 dynamic resources): the GPU address of the heap's
    /// table of element strides, with every buffer view of the heap given
    /// its host slot and its memory brought up to date when `list` is
    /// submitted. Tables tell us which views a draw reads; nothing tells
    /// us which of the heap's a computed index names, so all of them are
    /// made ready, once per list.
    /// shortcut: the buffers are read, not written: what such a shader
    /// writes through a picked unordered access view stays in the host
    /// copy. A list refreshes them where it first runs such a shader;
    /// title writes recorded later in the same list are not seen.
    fn heap_buffers(&self, list: &HostList, writes: bool) -> Option<u64> {
        // What it may store through a picked view is declared nowhere: the copy-queue fetch
        // holds the list's submission behind every ticket out (`host/fetch.rs`).
        if writes {
            list.first_use(crate::memory::WRITES_BY_HEAP);
        }
        let heap = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())?;
        let host = heap.host()?;
        let gpu = host.strides_gpu()?;
        // One key per list, as textures have: no title resource is there.
        const ONCE: usize = usize::MAX - 0x5e;
        if !list.first_use(ONCE) {
            return Some(gpu);
        }
        // The list may outlive the title's heap object, which owns the table.
        if let Some(buffer) = host.strides_buffer() {
            list.keep(buffer);
        }
        // The pass itself once per frame: per list it ran dozens of times a
        // frame over tens of thousands of views and thousands of host copies,
        // many seconds of every frame.
        // shortcut: what the title writes into such buffers between lists
        // of one frame is seen a frame late.
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        let frame = crate::frame::current();
        if FRAME.swap(frame, Ordering::Relaxed) == frame {
            return Some(gpu);
        }
        let started = std::time::Instant::now();
        // Per host heap: the host copy and range each slot's buffer view
        // reads, kept from pass to pass. Only the slots written since the
        // last pass are looked at again (tens of thousands of slots a pass
        // otherwise); all of them every 16 passes, because a view of a
        // released resource goes stale without a write, and a slot written
        // before its memory had a host copy is only made then.
        type Ranges = std::collections::HashMap<u32, (std::sync::Arc<crate::shadow::Shadow>, u64, u64)>;
        static RANGES: std::sync::Mutex<Option<std::collections::HashMap<usize, (u64, Ranges)>>> =
            std::sync::Mutex::new(None);
        let mut kept = RANGES.lock().unwrap_or_else(|e| e.into_inner());
        let (passes, ranges) = kept
            .get_or_insert_with(Default::default)
            .entry(std::ptr::from_ref(host) as usize)
            .or_insert_with(|| (0, Ranges::new()));
        let changed = host.buffer_changes();
        let slots = if *passes % 16 == 0 {
            ranges.clear();
            host.buffer_slots()
        } else {
            changed
        };
        *passes += 1;
        // Whether the device was there before the pass: a view that removes
        // it is named below.
        static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let mut alive = !GONE.load(Ordering::Relaxed) && crate::descriptor::host::device_removed().is_none();
        for (n, &index) in slots.iter().enumerate() {
            ranges.remove(&index);
            let Some(handle) = heap.handle(index) else {
                continue;
            };
            let Some(view) = host.mirror(index, handle, true) else {
                continue;
            };
            // Every 64 views while the device lives (a cheap query).
            if alive
                && n % 64 == 63
                && let Some(why) = crate::descriptor::host::device_removed()
            {
                alive = false;
                GONE.store(true, Ordering::Relaxed);
                let last: Vec<String> = slots[n - 63..=n]
                    .iter()
                    .filter_map(|&i| crate::descriptor::view(heap.handle(i)?).map(|v| format!("{i}: {v:?}")))
                    .map(|s| s.chars().take(200).collect())
                    .collect();
                tracing::error!(why, slot = index, "the host device went while buffer views of the heap were made; one of: {last:#?}");
                crate::host::log_removal_context();
            }
            // The copy the slot shows (`buffer_copy`), not the one the
            // view's range alone would find.
            let Some((shadow, address, len)) = crate::descriptor::host::buffer_copy(&view) else {
                continue;
            };
            #[cfg(feature = "diagnostics")]
            if host.shows(index, &shadow.buffer) == Some(false) {
                HEAP_VIEWS_ELSEWHERE.fetch_add(1, Ordering::Relaxed);
            }
            ranges.insert(index, (shadow, address, address + len));
        }
        let views = ranges.len() as u32;
        // Per host copy, the span its picked views cover.
        let mut spans: std::collections::HashMap<usize, (std::sync::Arc<crate::shadow::Shadow>, u64, u64)> =
            std::collections::HashMap::new();
        for (shadow, from, to) in ranges.values() {
            let key = std::sync::Arc::as_ptr(shadow) as usize;
            let span = spans.entry(key).or_insert_with(|| (shadow.clone(), *from, *to));
            span.1 = span.1.min(*from);
            span.2 = span.2.max(*to);
        }
        drop(kept);
        let bytes: u64 = spans.values().map(|s| s.2 - s.1).sum();
        // Memory that is not write-watched is copied whole by every refresh.
        // It was once refreshed only every so many frames (unmapped pages read
        // one by one cost gigabytes a frame before they were
        // remembered): streamed geometry then drew from stale buffers, and
        // the garbage ended runs (the card stopped answering early in
        // a scene with heavy geometry streaming; with every frame it ran
        // on). The pass
        // costs tens of milliseconds a frame this way.
        static WHOLE: std::sync::Mutex<Option<std::collections::HashMap<usize, u64>>> = std::sync::Mutex::new(None);
        let mut whole = WHOLE.lock().unwrap_or_else(|e| e.into_inner());
        let whole = whole.get_or_insert_with(Default::default);
        for (shadow, from, to) in spans.values() {
            if shadow.unwatched() {
                // By the copy's number: a copy made where one let go was
                // (`shadow::let_go_idle`) has never been refreshed.
                let key = shadow.id() as usize;
                // `VICTORY_UNWATCHED_FRAMES` sets the interval (1: every frame).
                static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                let every = *EVERY.get_or_init(|| {
                    std::env::var("VICTORY_UNWATCHED_FRAMES")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1)
                });
                match whole.get(&key) {
                    Some(&last) if frame < last + every => continue,
                    _ => {
                        whole.insert(key, frame);
                    }
                }
            }
            shadow.refresh(list, *from, to - from);
        }
        self.heap_textures(list, heap, host);
        HEAP_BUFFER_PASSES.fetch_add(1, Ordering::Relaxed);
        HEAP_BUFFER_MICROS.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        static LOGGED: AtomicU64 = AtomicU64::new(0);
        // The first pass and every growth of the set by half.
        let seen = LOGGED.load(Ordering::Relaxed);
        if u64::from(views) > seen + seen / 2 {
            LOGGED.store(u64::from(views), Ordering::Relaxed);
            tracing::info!(
                slots = slots.len(),
                views,
                copies = spans.len(),
                mb = bytes >> 20,
                ms = started.elapsed().as_millis(),
                "buffer views of the heap made ready for shaders that pick them by index"
            );
        }
        Some(gpu)
    }

    /// Runs the pass that uploads textures picked by index
    /// ([`State::heap_textures`]) where nothing has asked for it yet
    /// ([`early_budgets`]): once a frame for a few milliseconds, at the
    /// frame's first draw with a view heap set, and with a larger budget
    /// of its own before every draw whose pixel shader is one of the
    /// title's bake programs. The uploads are recorded into `list` ahead
    /// of the draw being bound, as the world's pass records them.
    fn early_textures(&self, list: &HostList, p: &HostPipeline, compute: bool) {
        let (early_ms, bake_ms) = early_budgets();
        if compute || (early_ms == 0 && bake_ms == 0) {
            return;
        }
        let bake = bake_ms > 0
            && p.stages.iter().any(|s| BAKE_PIXEL_SHADERS.contains(&s.shader.hash.as_str()));
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        // The milliseconds this frame's bake draws have taken so far: a
        // frame has several, and together they get `bake_ms`.
        static BAKED_MS: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        let first = FRAME.swap(frame, Ordering::Relaxed) != frame;
        if first {
            BAKED_MS.store(0, Ordering::Relaxed);
        }
        let left = bake_ms.saturating_sub(BAKED_MS.load(Ordering::Relaxed));
        let budget = if bake && left > 0 {
            // What a bake may read must be there now: up to 2 GiB.
            (left, 2 << 30)
        } else if first && early_ms > 0 {
            (early_ms, 64 << 20)
        } else {
            return;
        };
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return;
        };
        let Some(host) = heap.host() else {
            return;
        };
        let bake = bake && left > 0;
        let n = EARLY_TEXTURE_PASSES[usize::from(bake)].fetch_add(1, Ordering::Relaxed) + 1;
        if bake && n.is_power_of_two() {
            let [slots, due, ..] = HEAP_TEXTURES_QUEUE.each_ref().map(|c| c.load(Ordering::Relaxed));
            tracing::info!(
                frame,
                bakes = n,
                queued_at_the_last_pass = slots,
                due_at_the_last_pass = due,
                uploaded_so_far = HEAP_TEXTURES.load(Ordering::Relaxed),
                "a draw of the title's decoration bake: textures picked by index are uploaded first (VICTORY_BAKE_TEXTURES_MS=0: not)"
            );
        }
        let began = std::time::Instant::now();
        PASS_BUDGET.set(Some(budget));
        self.heap_textures(list, heap, host);
        PASS_BUDGET.set(None);
        if bake {
            BAKED_MS.fetch_add(began.elapsed().as_millis() as u64, Ordering::Relaxed);
        }
    }

    /// The queue's side of the card accounting line
    /// (`crate::resource::card_accounting`, which says what the line
    /// holds): the slots that are due, by what their texture has on the
    /// card. Off unless `VICTORY_CARD_ACCOUNTING=1`; then at the first pass
    /// that finds the card without room, and every five seconds after it.
    /// Nothing is made, uploaded or counted as used here. For 18,000 due
    /// slots: 18,000 looks at a title slot and at the texture it views, a
    /// lock each, some milliseconds of the thread that records the frame.
    fn card_accounting(
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
        frame: u64,
        room: bool,
    ) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ACCOUNTING").is_some_and(|v| v == "1")) {
            return;
        }
        // When the next line is due (0: none yet, the card has had room).
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let now = crate::host::now_ms().max(1);
        let next = NEXT.load(Ordering::Relaxed);
        // (With levels on demand the card may never be without room: the
        // lines then start with the first texture made that way.)
        let by_level = crate::tiles::STATS[4].load(Ordering::Relaxed) > 0;
        let due_now = if next == 0 { !room || by_level } else { now >= next };
        if !due_now || NEXT.compare_exchange(next, now + 5000, Ordering::Relaxed, Ordering::Relaxed).is_err() {
            return;
        }
        // Due slots; of them uploaded before; of them viewing nothing the
        // title fills; and the textures (each once) whose host copy is
        // alive, with their bytes on the card.
        let mut due = [0u64; 5];
        // Textures (each once) that never had a host copy, and ones whose
        // host copy was let go: where their memory is, and its bytes.
        let (mut never, mut gone) = (Vec::new(), Vec::new());
        let mut seen = std::collections::BTreeSet::new();
        for (index, uploaded) in host.texture_slots_all_due(frame) {
            due[0] += 1;
            due[1] += u64::from(uploaded);
            let state = match heap.handle(index).and_then(crate::descriptor::view) {
                Some(View::Placed(false, resource, desc)) if resource[0] != 1 => {
                    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                    match crate::resource::placed_known(address, &resource) {
                        Some(texture) => crate::resource::copy_state(texture),
                        // No title texture was made for it: the pass never
                        // got as far as making one.
                        None => Some((0, crate::resource::placed_bytes(&resource).unwrap_or(0), address)),
                    }
                }
                Some(View::Shader(object, _)) if crate::resource::title_fills(object as This) => {
                    crate::resource::copy_state(object as This)
                }
                _ => None,
            };
            let Some((state, bytes, memory)) = state else {
                due[2] += 1;
                continue;
            };
            if !seen.insert(memory) {
                continue;
            }
            match state {
                2 => {
                    due[3] += 1;
                    due[4] += bytes;
                }
                1 => gone.push((memory, bytes)),
                _ => never.push((memory, bytes)),
            }
        }
        crate::resource::card_accounting(room, due, never, gone);
    }

    /// The texture side of [`State::heap_buffers`]: textures the title fills
    /// itself (streamed pages, layout-library copies) whose views sit in the
    /// view heap are created (placed textures have no host texture until
    /// something reads them) and uploaded, as a table read would do, for
    /// shaders that pick them by a computed index. Within a budget of bytes
    /// per frame and in all; the rest waits for later lists and frames.
    /// The most recently written slots first (`TextureQueue::due`); slots
    /// draws read through tables are not in the queue.
    /// shortcut: uploaded once, and again only when the title rewrites the
    /// slot. A placed texture not all mapped yet is uploaded with what is
    /// there and looked at again (1, 2, 4, ... frames later, at most about
    /// a second): uploaded again when more of it is mapped. Pages mapped
    /// in place of others, and contents changed with no descriptor write,
    /// are missed (hook the page-mapping events for those).
    /// `VICTORY_HEAP_TEXTURES=0` turns it off.
    fn heap_textures(
        &self,
        list: &HostList,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
    ) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_TEXTURES").is_none_or(|v| v != "0")) {
            return;
        }
        /// Bytes of texture memory staged per frame.
        // `VICTORY_TEXTURE_STAGE_MB` raises it: a backlog of tens of thousands of
        // textures (a scene's first minutes) then drains in fewer frames.
        static PER_FRAME_MB: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let per_frame: u64 = *PER_FRAME_MB.get_or_init(|| {
            std::env::var("VICTORY_TEXTURE_STAGE_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(192)
        }) << 20;
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        static SPENT: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        if FRAME.swap(frame, Ordering::Relaxed) != frame {
            SPENT.store(0, Ordering::Relaxed);
        }
        // No more of them are made while the card has no room
        // (`resource::room`): by what is alive now, not by what was ever
        // staged (4 GB in all, after which later textures stayed empty for
        // the run). The ones nothing names any more are let go, which
        // makes room again: so the slots are still looked at (the pass
        // returned here, and a slot whose texture had been streamed out
        // kept its host copy for as long as the card was full).
        let room = crate::resource::room();
        if !room {
            HEAP_TEXTURES_FULL.fetch_add(1, Ordering::Relaxed);
            once!("textures picked by index: the card has no room for more; they wait");
        }
        let mut refused = 0;
        // Room again: the slots held back while the card was full are due
        // now (`full_card_pass`).
        if room {
            host.texture_slots_room(frame);
        }
        // Slots whose texture's memory the runtime mapped, unmapped or
        // wrote since the last pass are due again.
        host.texture_slots_remapped(frame);
        // With receipts (`crate::receipts`): and the slots whose host copy is
        // older than what was written there since, whoever took the ranges.
        if crate::receipts::on() {
            host.receipts_walk(frame);
        }
        host.texture_sweep(heap, frame);
        if heap_recheck() {
            let [same, changed, waiting] = host.texture_slots_rechecked(frame);
            HEAP_TEXTURES_RECHECKED[0].fetch_add(same, Ordering::Relaxed);
            HEAP_TEXTURES_RECHECKED[1].fetch_add(changed, Ordering::Relaxed);
            HEAP_TEXTURES_RECHECKED[2].store(waiting, Ordering::Relaxed);
        }
        // Levels on demand: the slots of textures whose first level with
        // memory changed since the last pass are made again from it.
        host.texture_views_remade();
        // And with room, the textures whose upload left a held level
        // without the card's memory are uploaded again.
        if room {
            host.texture_slots_waiting(frame);
        }
        let (due, census) = host.texture_slots_due(frame, 4096);
        HEAP_TEXTURES_WAITING.store(census[0], Ordering::Relaxed);
        for (count, now) in HEAP_TEXTURES_QUEUE.iter().zip(census) {
            count.store(now, Ordering::Relaxed);
        }
        // Where the card's memory is, and what the due slots would need
        // (`VICTORY_CARD_ACCOUNTING=1`; a line when the card first has no
        // room, and every five seconds after it).
        Self::card_accounting(heap, host, frame, room);
        // The frames in about four seconds, between 16 and 256: how long a
        // slot with nothing to upload waits at most for its next look. In
        // frames alone it was 120: four minutes at the half a frame per
        // second of a slow warm-up, while the title streams by the clock.
        // (About a second before the runtime told of the memory it maps:
        // the looks are now only for what that does not tell.)
        static PASSED: AtomicU64 = AtomicU64::new(0);
        let now = crate::host::now_ms();
        let took = now.saturating_sub(PASSED.swap(now, Ordering::Relaxed)).max(1);
        let most = (4000 / took).clamp(16, 256);
        // What a table entry of a texture a shader samples says.
        let sampled = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        // Also within a time per pass: the pass runs on a thread that
        // records the frame, and making and staging the textures took
        // seconds of each of the world's first frames.
        // The rest waits for the next frames, as it does past the bytes.
        static SPAN_MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let span = std::time::Duration::from_millis(*SPAN_MS.get_or_init(|| {
            std::env::var("VICTORY_TEXTURE_STAGE_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(250)
        }));
        // A long queue (the world's first minutes: tens of thousands of slots, which
        // at the usual pace took three and a half minutes with the materials
        // black) gets 512 MB and 1.5 s a pass while the run's allowance for
        // catching up lasts (`catching_up`, `VICTORY_TEXTURE_GATE_S`, 60 s):
        // joshuu/victory-amd's pace, which brought more in than tripling the
        // usual budget once 1,024 were due. `VICTORY_TEXTURE_CATCHUP=0`
        // keeps the plain budget.
        const PER_FRAME_CATCHING_UP: u64 = 512 << 20;
        const SPAN_CATCHING_UP: std::time::Duration = std::time::Duration::from_millis(1500);
        const LONG: u64 = 2_000;
        static CATCHUP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let catchup = *CATCHUP.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_CATCHUP").is_none_or(|v| v != "0"));
        let plain = span;
        let (per_frame, span) = if catchup && census[0] > LONG && catching_up(std::time::Duration::ZERO) {
            (per_frame.max(PER_FRAME_CATCHING_UP), span.max(SPAN_CATCHING_UP))
        } else {
            (per_frame, span)
        };
        // A run made early (`State::early_textures`) has a budget of its
        // own: its bytes on top of what the frame has spent so far.
        // (Not `most`: that is the longest wait between looks, used below.)
        let (span, bytes_at_most) = match PASS_BUDGET.get() {
            Some((ms, bytes)) => {
                (std::time::Duration::from_millis(ms), SPENT.load(Ordering::Relaxed).saturating_add(bytes))
            }
            None => (span, per_frame),
        };
        let began = std::time::Instant::now();
        // With receipts: a number at or below the journal's newest, read
        // without its lock, the receipt of a texture found with nothing
        // mapped (an earlier number is the safe one: more counts as written
        // after it).
        let unmapped_at = crate::receipts::on().then(crate::watch::absent_number);
        // The due slots the loop reached (with receipts: the rest the budget
        // left are named so).
        let mut looked = 0;
        for &(index, write) in &due {
            if SPENT.load(Ordering::Relaxed) >= bytes_at_most || began.elapsed() > span {
                break;
            }
            looked += 1;
            // A placed texture whose memory is not mapped gets no host
            // texture; it is looked at again soon. One of which some is
            // mapped is uploaded with what is there, and again when more
            // is. (It was uploaded again every 120 frames, changed or not;
            // and asked by its first page alone, one whose first page is
            // the one missing counted as not there.)
            let mut partly = None;
            // What the pass saw of a placed view when it decided, for the
            // line about the slot.
            #[cfg(feature = "diagnostics")]
            let mut seen = None;
            if let Some(View::Placed(false, resource, desc)) = heap.handle(index).and_then(crate::descriptor::view) {
                let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                // Of the mips the view shows: what lies behind them is
                // another texture's.
                let (len, mapped) = crate::texture::view_mapped(address, &resource, &desc);
                #[cfg(feature = "diagnostics")]
                {
                    seen = Some((resource, address, len));
                }
                if mapped == 0 {
                    HEAP_TEXTURES_AWAY.fetch_add(1, Ordering::Relaxed);
                    // The pipeline bound when the pass ran, not necessarily
                    // one that reads the slot.
                    #[cfg(feature = "diagnostics")]
                    crate::null_trace::note(crate::null_trace::Kind::TextureAway, || {
                        format!(
                            "pipeline {} slot {index} address {address:#x}",
                            crate::pipeline::shaders_of(self.pipeline as This)
                        )
                    });
                    // Streamed out: what the slot showed is let go.
                    host.texture_slot_gone(index);
                    // With receipts: owed nothing until a map or a write
                    // there is stamped above this.
                    if let Some(seq) = unmapped_at {
                        crate::descriptor::host::receipts::absent(address, &resource, &desc, len, seq);
                    }
                    let entered = host.texture_slot_later(index, write, frame, most);
                    #[cfg(feature = "diagnostics")]
                    if entered {
                        Self::say_stuck(index, "waiting for its memory", seen.as_ref(), Some(0), frame);
                    }
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = entered;
                    continue;
                }
                if mapped < len {
                    if mapped <= host.texture_slot_mapped(index) {
                        HEAP_TEXTURES_PARTLY[0].fetch_add(1, Ordering::Relaxed);
                        host.texture_slot_later(index, write, frame, most);
                        continue;
                    }
                    if !crate::texture::committed(address as *const u8, 1) {
                        HEAP_TEXTURES_PARTLY[2].fetch_add(1, Ordering::Relaxed);
                    }
                    partly = Some(mapped);
                }
            }
            // A texture object the host renders to or writes holds the
            // host's own contents: nothing to bring up to date, and this
            // pass is no use of it (its host copy is made when something
            // uses it, and its slots then: `descriptor::make_waiting`).
            if let Some(View::Shader(object, _)) = heap.handle(index).and_then(crate::descriptor::view)
                && crate::resource::buffer_bytes(object as This).is_none()
                && !crate::resource::title_fills(object as This)
            {
                host.texture_slot_done(index, write, None);
                continue;
            }
            // From here on a host texture is made: not without room. The
            // slot stays due.
            // Not so a slot already made from this very view: its host
            // texture exists, and uploading it again takes none of the
            // card's memory. It was refused all the same, so a texture filled
            // after its first upload kept its empty host copy once the card
            // was full: the same slots were found changed again at every
            // pass and never uploaded (tens of thousands of such findings in
            // a run, measured). `VICTORY_REUPLOAD_WHEN_FULL=0`: refused, as
            // before.
            if !room {
                let has_its_texture = reupload_when_full()
                    && heap.handle(index).is_some_and(|handle| host.texture_slot_made_from(index, handle));
                if !has_its_texture {
                    refused += 1;
                    if full_card_pass() {
                        // Held back for the frames of about four seconds
                        // (`most`), or until the card has room; the pass
                        // goes on, within a count that bounds one pass.
                        host.texture_slot_held(index, write, frame + most);
                        if refused > 1024 {
                            break;
                        }
                    } else if refused > 256 {
                        break;
                    }
                    continue;
                }
                let n = HEAP_TEXTURES_REUPLOADED_FULL.fetch_add(1, Ordering::Relaxed) + 1;
                if n == 1 || n.is_multiple_of(500) {
                    tracing::info!(
                        so_far = n,
                        slot = index,
                        "a texture already on the card is brought up to date although the card has no room for new ones"
                    );
                }
            }
            let view = heap.handle(index).and_then(|handle| host.mirror(index, handle, true));
            let placed = view.as_ref().and_then(crate::descriptor::host::placed_texture);
            let resource = match placed.as_ref().or(view.as_ref()) {
                Some(&View::Shader(resource, _)) => resource as This,
                _ => {
                    host.texture_slot_done(index, write, None);
                    continue;
                }
            };
            let title_fills = crate::resource::host_resource(resource).is_some_and(crate::texture::title_written);
            if !title_fills {
                // a render target, a depth or unordered access texture: the host's own
                host.texture_slot_done(index, write, None);
                continue;
            }
            // A texture that has a staging buffer now is one draws read
            // through a table: they upload it, and check it at every
            // submission that samples it.
            let tabled = crate::texture::is_staged(resource);
            if tabled {
                crate::texture::upload_again(resource);
            }
            if let Some(view) = &view {
                readable(list, sampled.kind, view);
            }
            // A texture view the host refuses removes the device: name it.
            static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !GONE.load(Ordering::Relaxed)
                && let Some(why) = crate::descriptor::host::device_removed()
            {
                GONE.store(true, Ordering::Relaxed);
                let text: String = format!("{view:?}").chars().take(300).collect();
                tracing::error!(why, slot = index, "the host device went while textures picked by index were made ready; last: {text}");
                crate::host::log_removal_context();
                return;
            }
            let (bytes, whole) = crate::texture::staged(resource).unwrap_or((0, true));
            // Not all of it there: looked at again next frame, and from
            // then on as a slot with nothing new (a placed texture: until
            // more of it is mapped; a texture object at an address the
            // title mapped: again in 120 frames, as before).
            let again = match partly {
                Some(mapped) => Some((frame + 1, mapped)),
                None => (!whole).then_some((frame + 120, 0)),
            };
            if again.is_some() {
                HEAP_TEXTURES_PARTLY[1].fetch_add(1, Ordering::Relaxed);
            }
            if tabled {
                // Nothing was staged here, so nothing is counted or spent,
                // and the staging buffer stays: taking it away made the
                // next draw that reads the texture through a table stage,
                // detile and upload all of it once more.
                HEAP_TEXTURES_TABLED.fetch_add(1, Ordering::Relaxed);
                let entered = host.texture_slot_done(index, write, again);
                #[cfg(feature = "diagnostics")]
                if entered {
                    Self::say_stuck(index, "uploaded in part", seen.as_ref().or(Self::object_texture(resource).as_ref()), partly, frame);
                }
                #[cfg(not(feature = "diagnostics"))]
                let _ = entered;
                continue;
            }
            // Uploaded: the list holds the staging buffer until it has run;
            // a table read, or the next look at a texture not all mapped
            // yet, makes it anew. (Kept for those until now: every partly
            // streamed texture of the heap held a staging buffer of its full
            // size for good.)
            let memory = if again.is_none() && heap_recheck() {
                crate::texture::staged_memory(resource)
            } else {
                None
            };
            crate::texture::unstage(resource);
            SPENT.fetch_add(bytes, Ordering::Relaxed);
            HEAP_TEXTURE_BYTES.fetch_add(bytes, Ordering::Relaxed);
            HEAP_TEXTURES.fetch_add(1, Ordering::Relaxed);
            let entered = host.texture_slot_done(index, write, again);
            #[cfg(feature = "diagnostics")]
            if entered {
                Self::say_stuck(index, "uploaded in part", seen.as_ref().or(Self::object_texture(resource).as_ref()), partly, frame);
            }
            #[cfg(not(feature = "diagnostics"))]
            let _ = entered;
            // Whole and uploaded: looked at again (`texture_slots_rechecked`).
            if let Some((at, len, whole)) = memory {
                host.texture_slot_recheck(index, frame, (at, len, whole), crate::texture::fingerprint(at, len, whole));
            }
        }
        if crate::receipts::on() {
            host.receipts_left(&due[looked..]);
        }
        catching_up(began.elapsed().saturating_sub(plain));
    }

    /// For the frame listing, after a dropped dispatch's line: its group
    /// counts and the compute root arguments it was recorded with (root
    /// constants as hex dwords, root views with the title buffer they
    /// point into). A program that does not translate has no bindings to
    /// list; these say which memory it would have read and written (the
    /// ray-tracing structure builder passes its buffers this way).
    #[cfg(feature = "diagnostics")]
    fn list_dropped_roots(&self, groups: [u32; 3]) {
        crate::capture::op(0, || {
            use std::fmt::Write;
            let [x, y, z] = groups;
            let mut text = format!("  its roots ({x}x{y}x{z}):");
            for (slot, root) in self.compute.iter().enumerate() {
                match root {
                    Root::None => {}
                    Root::Table(a) => {
                        let _ = write!(text, " {slot}=table {a:#x}");
                    }
                    Root::Constants(v) => {
                        let _ = write!(text, " {slot}=constants {:x?}", &v[..v.len().min(48)]);
                    }
                    Root::View(a) => {
                        let _ = write!(text, " {slot}=view {a:#x}");
                        match crate::shadow::extent(*a) {
                            Some((start, len)) => {
                                let _ = write!(text, " in buffer {start:#x}+{len:#x}");
                            }
                            None => text.push_str(" in no buffer"),
                        }
                    }
                }
            }
            text
        });
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        // It may write the arguments a window holds clamped.
        self.window = None;
        if live_skip(self.pipeline as This, true) {
            crate::live::skipped(true);
            return;
        }
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        // The ray-tracing structure builder's stages run all or none.
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_why("a stage of the ray-tracing structure builder, which is off".to_owned());
                self.list_dropped_roots([x, y, z]);
            }
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_drop(self.pipeline as This);
                self.list_dropped_roots([x, y, z]);
            }
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            // A stage of the structure builder that cannot be bound is a
            // dropped stage: the others stop too.
            if crate::pipeline::is_builder(self.pipeline as This) {
                crate::pipeline::builder_off("one of its dispatches could not be bound");
            }
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        match group_base(p) {
            Some((param, dword)) if x > crate::indirect::PART => {
                // More than the host runs in one dimension: in parts.
                // SAFETY: an open list with everything set; the parameter
                // and dword are the pipeline's.
                unsafe {
                    let mut base = 0;
                    while base < x {
                        list.list.SetComputeRoot32BitConstant(param, base, dword);
                        list.list.Dispatch((x - base).min(crate::indirect::PART), y, z);
                        base += crate::indirect::PART;
                    }
                    list.list.SetComputeRoot32BitConstant(param, 0, dword);
                }
            }
            // SAFETY: an open list with everything set.
            _ => unsafe { list.list.Dispatch(x, y, z) },
        }
        #[cfg(feature = "diagnostics")]
        {
            list.pass_called();
            list.stamp_call(p);
        }
        // The census of runs of clamped calls (`crate::runs`, (a)), and the
        // clamp of runs: a dispatch ends the batch open.
        #[cfg(feature = "diagnostics")]
        list.runs(crate::runs::Runs::dispatch);
        list.batch_end();
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        // What it draws into (a dispatch writes through its views below).
        if !compute {
            let described = |t: usize| {
                crate::resource::host_made(t as This).map(|r| {
                    // SAFETY: a live resource.
                    let d = unsafe { r.GetDesc() };
                    format!("{t:#x} {}x{} f{}", d.Width, d.Height, d.Format.0)
                })
            };
            let targets: Vec<String> = self.targets.iter().filter_map(|&t| described(t)).collect();
            let _ = write!(text, " targets {targets:?} depth {:?} state [{}]", described(self.depth), p.state);
            // The scissor and viewport at this draw (the first of each): a
            // guest may draw its view through a portal (a window, from
            // inside a room) under the portal's rectangle, and the listing
            // did not say which draws those are.
            if let Some(r) = self.scissors.first() {
                let _ = write!(text, " scissor [{} {} {} {}]", r.left, r.top, r.right, r.bottom);
            }
            if let Some(v) = self.viewports.first() {
                let _ = write!(
                    text,
                    " viewport [{} {} {} {} depth {}..{}]",
                    v.TopLeftX, v.TopLeftY, v.Width, v.Height, v.MinDepth, v.MaxDepth
                );
            }
        }
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = match u.root {
                    RootType::Other(kind) => self.host_word(compute, kind, u.slot, u.dword, None),
                    _ => self.user_value(compute, u.slot, u.root, u.dword),
                };
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(slot, _) in stage.root_cbvs.iter().chain(&stage.root_srvs).chain(&stage.root_uavs) {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        if !compute && !list_slots().is_empty() {
            text.push_str(&self.slots_text());
        }
        text
    }

    /// For the frame listing, after a draw whose shader `VICTORY_LIST_SLOTS`
    /// names: the heap slots it will pick textures from by index, and what
    /// the title and the host hold there (one `~slot` line each), with the
    /// words they were read from (`~raw` lines, to decode offline what is
    /// not decoded here).
    ///
    /// Where the slots are: in the small buffers and constants of the
    /// draw's own table (every word of those that names a texture view gets
    /// a line), which needs no setting; and in records of larger buffers,
    /// at places that belong to one program of whatever title is run and
    /// are given by `VICTORY_LIST_SLOTS_AT` ([`SlotPlaces`]), never built
    /// in. Without them, or with a value that does not parse, one line says
    /// so and no record is looked for. A word that is a small number by
    /// chance and names a texture gets a line too.
    #[cfg(feature = "diagnostics")]
    fn slots_text(&self) -> String {
        use std::fmt::Write;
        let places = slot_places();
        let wanted = list_slots();
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return String::new();
        };
        let Some(stage) = p.stages.iter().find(|s| wanted.iter().any(|w| s.shader.hash.starts_with(w.as_str())))
        else {
            return String::new();
        };
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return String::new();
        };
        let t = stage.translated();
        let mut text = String::new();
        let entry = |slot: u8, dwords: u32| -> Option<View> {
            let Some(Root::Table(table)) = self.roots_ref(false).get(usize::from(slot)) else {
                return None;
            };
            crate::descriptor::view(table + u64::from(dwords) * 4)
        };
        let mut material = None;
        for u in t.used.iter().filter(|u| u.root == RootType::Table && u.kind != ResourceType::Sampler) {
            let label = format!("s{}+{}", u.slot, u.table_offset_dwords);
            let Some(view) = entry(u.slot, u.table_offset_dwords) else {
                let _ = write!(text, "\n    ~table {label} no view");
                continue;
            };
            let range = match &view {
                View::Constant(address, size) => Some((*address, u64::from(*size))),
                other => crate::descriptor::host::buffer_range(other),
            };
            match range {
                // A small buffer or constants: its words.
                Some((address, len)) if len <= 2048 => {
                    Self::slot_words(&mut text, heap, &label, address, len as usize);
                    if let Ok(places) = places
                        && (u.slot, u.table_offset_dwords) == places.constants
                        && len >= places.offset.saturating_add(4)
                    {
                        let mut word = [0u8; 4];
                        if crate::host::read_into((address + places.offset) as *const u8, &mut word) {
                            material = Some(u32::from_le_bytes(word));
                        }
                    }
                }
                Some(_) => {}
                // A texture read through the table: the title slot it sits in.
                None => {
                    let Some(Root::Table(table)) = self.roots_ref(false).get(usize::from(u.slot)) else {
                        continue;
                    };
                    match heap.index(table + u64::from(u.table_offset_dwords) * 4) {
                        Some(index) => text.push_str(&Self::slot_line(heap, &format!("table:{label}"), 0, index)),
                        None => {
                            let _ = write!(text, "\n    ~table {label} title=[{}]", view_state(Some(&view)));
                        }
                    }
                }
            }
        }
        let uses = |slot: u8, dwords: u32| {
            t.used.iter().any(|u| u.root == RootType::Table && u.slot == slot && u.table_offset_dwords == dwords)
        };
        let places = match places {
            Ok(places) => places,
            Err(why) => {
                text.push_str(&places_not_given(why));
                return text;
            }
        };
        let (first, second) = (places.first, places.second);
        if let Some(at) = material
            && uses(first.0, first.1)
            && let Some((base, len)) = entry(first.0, first.1).as_ref().and_then(crate::descriptor::host::buffer_range)
        {
            let at = u64::from(at);
            let _ = write!(text, "\n    ~material record at {at:#x} of the buffer at {base:#x} bytes {len}");
            if at + 64 <= len {
                Self::slot_words(&mut text, heap, "M", base + at, 64);
            }
            if let Some(further) = places.further
                && at.saturating_add(further).saturating_add(512) <= len
            {
                Self::slot_words(&mut text, heap, &format!("M+{further}"), base + at + further, 512);
            }
        }
        if uses(second.0, second.1)
            && let Some(slot) = t.used.iter().find(|u| u.root == RootType::Constants).map(|u| u.slot)
            && let Some(Root::Constants(words)) = self.roots_ref(false).get(usize::from(slot))
            && let Some(&at) = words.get(places.root_constant)
            && let Some((base, len)) = entry(second.0, second.1).as_ref().and_then(crate::descriptor::host::buffer_range)
        {
            let at = u64::from(at);
            let _ = write!(text, "\n    ~root constants {words:08x?}: record at {at:#x} of the buffer at {base:#x} bytes {len}");
            if at + 64 <= len {
                Self::slot_words(&mut text, heap, "G", base + at, 64);
            }
        }
        text
    }

    /// Up to 512 bytes of title memory at `address` as hex words (a `~raw`
    /// line), and a `~slot` line for every word of them that names a
    /// texture view in `heap`.
    #[cfg(feature = "diagnostics")]
    fn slot_words(text: &mut String, heap: &crate::descriptor::Heap, label: &str, address: u64, len: usize) {
        use std::fmt::Write;
        let mut bytes = vec![0u8; len.min(512) & !3];
        if bytes.is_empty() || !crate::host::read_into(address as *const u8, &mut bytes) {
            let _ = write!(text, "\n    ~raw {label} at {address:#x} bytes {len}: unreadable");
            return;
        }
        let words: Vec<u32> =
            bytes.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
        let _ = write!(text, "\n    ~raw {label} at {address:#x} bytes {len}:");
        for w in &words {
            let _ = write!(text, " {w:08x}");
        }
        for (i, &index) in words.iter().enumerate() {
            let names = index != 0
                && heap
                    .handle(index)
                    .and_then(crate::descriptor::view)
                    .as_ref()
                    .is_some_and(crate::descriptor::host::is_texture_view);
            if names {
                text.push_str(&Self::slot_line(heap, label, i * 4, index));
            }
        }
    }

    /// One `~slot` line: where the slot number was read (`from`, byte
    /// `at`), what the title's heap holds in slot `index` with the state
    /// of that texture ([`view_state`]), what the host slot was made from,
    /// and what the queue of textures picked by index knows of it.
    #[cfg(feature = "diagnostics")]
    fn slot_line(heap: &crate::descriptor::Heap, from: &str, at: usize, index: u32) -> String {
        let title = heap.handle(index).and_then(crate::descriptor::view);
        let host: String = heap
            .host()
            .and_then(|h| h.describe_slot(index))
            .unwrap_or_else(|| "none".into())
            .chars()
            .take(90)
            .collect();
        let queue = match heap.host() {
            Some(h) => h.texture_slot_state(index, crate::frame::current()),
            None => "no host heap".into(),
        };
        format!(
            "\n    ~slot from={from}+{at:#x} slot={index} title=[{}] host=[{host}] queue=[{queue}]",
            view_state(title.as_ref())
        )
    }

    /// A mesh draw under `VICTORY_MESH_TEST=scissor` rasterises one pixel:
    /// the scissors before it (`on`), the title's after it.
    fn mesh_scissor(&self, list: &HostList, on: bool) {
        if mesh_test() != 2 {
            return;
        }
        if on {
            let one = RECT { left: 0, top: 0, right: 1, bottom: 1 };
            crate::draw::scissors(list, &vec![one; self.scissors.len().max(1)]);
        } else if !self.scissors.is_empty() {
            crate::draw::scissors(list, &self.scissors);
        }
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        attempt();
        self.window = None;
        if live_skip(self.pipeline as This, false) {
            crate::live::skipped(false);
            return;
        }
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = mesh_test() != 1
            && mesh_kept(self.pipeline)
            && self.prepare_for(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    self.mesh_scissor(list, true);
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    #[cfg(feature = "diagnostics")]
                    self.stamp_call(list);
                    self.mesh_scissor(list, false);
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// The walls trace (`crate::host::walls_trace`): one log line per
    /// frame with the draws of its main pass (four or more colour targets)
    /// by scissor rectangle, and how often each of the title's culling
    /// programs was dispatched. The title clips what it draws
    /// through a window or door to that opening's rectangle while it holds
    /// the camera to be in a room (a scissor rectangle inside the frame's
    /// bounds, shared by the main pass's draws); in a launch without the
    /// room's walls the same shot has no such rectangle. The line shows
    /// the frame in which that changes.
    fn walls_census(&self, p: &HostPipeline, compute: bool) {
        let note = if compute {
            CensusNote::Dispatch(
                WALLS_CULL.map(|c| p.stages.iter().any(|s| s.shader.hash.as_str() == c.0)),
            )
        } else if self.targets.iter().filter(|&&t| t != 0).count() >= 4 {
            CensusNote::Draw(self.scissors.first().map_or([0; 4], |r| [r.left, r.top, r.right, r.bottom]))
        } else {
            CensusNote::Other
        };
        walls_census_in(crate::frame::current(), note);
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        let mirrored = host.mirror(index, handle, true);
        #[cfg(feature = "diagnostics")]
        if crate::null_trace::on() {
            let why = match &mirrored {
                None => Some("no view there"),
                Some(view)
                    if crate::descriptor::host::buffer_range(view).is_some() && host.shows_nothing(index) =>
                {
                    Some("a buffer view the host refused")
                }
                Some(_) => None,
            };
            if let Some(why) = why {
                crate::null_trace::note(crate::null_trace::Kind::BufferNull, || {
                    format!(
                        "{} root {} entry {} ({:?}) title slot {index}: {why}",
                        crate::pipeline::shaders_of(self.pipeline as This),
                        u.slot,
                        u.table_offset_dwords / 8,
                        u.kind
                    )
                });
            }
        }
        if let Some(view) = mirrored {
            readable(list, u.kind, &view);
            // Up to date by this draw: not the texture pass's to make.
            if crate::descriptor::host::is_texture_view(&view) {
                host.texture_slot_tabled(index, true);
            }
        }
    }

    /// The null-read trace's first kind (`crate::null_trace`): the view
    /// table entries a draw names, and those its shader picks by index,
    /// whose title slot lies past the ones the host view heap mirrors.
    /// Reads only: what [`Self::tables`] makes of them is its own.
    #[cfg(feature = "diagnostics")]
    fn past_heap_trace(&self, used: &[UsedResource], indexed: &[RootIndexedView], compute: bool) {
        // (root slot, bytes into the table, picked by index)
        let named = used
            .iter()
            .filter(|u| u.root == RootType::Table)
            .map(|u| (u.slot, u64::from(u.table_offset_dwords) * 4, false));
        let picked = indexed.iter().map(|index| {
            let constant = self.user_value(compute, index.constant_slot, RootType::Constants, index.constant_dword);
            // As `tables` finds the entry: i32 byte offsets, shifted by 5.
            let entry = index.byte_offset.wrapping_add(constant << index.shift) / 32;
            (index.table_slot, u64::from(entry) * 32, true)
        });
        for (slot, offset, by_index) in named.chain(picked) {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            if heap.samplers() {
                continue;
            }
            let (Some(host), Some(index)) = (heap.host(), table.checked_add(offset).and_then(|h| heap.index(h))) else {
                continue;
            };
            if index >= host.count() {
                crate::null_trace::note(crate::null_trace::Kind::PastHeap, || {
                    format!(
                        "{} root {slot} entry {}{} title slot {index}, host heap mirrors {}",
                        crate::pipeline::shaders_of(self.pipeline as This),
                        offset / 32,
                        if by_index { " (picked by index)" } else { "" },
                        host.count()
                    )
                });
            }
        }
    }

    /// The binding trace (`crate::bind_trace`): if this graphics draw's
    /// pixel shader is one the trace names and its bindings are new, a
    /// line for the draw and one per texture its shaders read, kept for the
    /// next present. Called once the draw is bound, so the host slots are
    /// as the draw reads them (its upload from title memory runs at
    /// submission: the present's line says how that went). Reads only.
    #[cfg(feature = "diagnostics")]
    fn bind_trace(&self, p: &HostPipeline) {
        use std::fmt::Write;
        use std::hash::{Hash, Hasher};
        /// Texture lines per draw at most.
        const MOST: usize = 64;
        let Some(pixel) = p.stages.iter().find(|s| s.shader.stage == "pixel") else {
            return;
        };
        let id = pixel.shader.hash.as_str();
        if !crate::bind_trace::wants(id) {
            return;
        }
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return;
        };
        let host = heap.host();
        let mirrored = host.map_or(0, |h| h.count());
        let roots = self.roots_ref(false);
        let table_of = |slot: u8| match roots.get(usize::from(slot)) {
            Some(&Root::Table(table)) => Some(table),
            _ => None,
        };
        // Per texture: stage, where the shader finds it, the title handle,
        // and whether the shader reads a dynamic slot made for the draw.
        let mut found: Vec<(&'static str, String, u64, bool)> = Vec::new();
        let mut uses = String::new();
        for stage in &p.stages {
            let t = stage.translated();
            let name = stage.shader.stage;
            let tables = |u: &&UsedResource| u.root == RootType::Table;
            // As `tables` decides: a table holding a constant buffer or
            // reaching past the mirrored slots is made afresh in dynamic
            // slots, from the title's views at this draw.
            let mut fresh: Vec<u8> =
                t.used.iter().filter(tables).filter(|u| u.kind == ResourceType::Cbv).map(|u| u.slot).collect();
            for u in t.used.iter().filter(tables) {
                if let Some(table) = table_of(u.slot)
                    && heap.index(table + u64::from(u.table_offset_dwords) * 4).is_some_and(|i| i >= mirrored)
                {
                    fresh.push(u.slot);
                }
            }
            let _ = write!(uses, " | {name} {} used [", stage.shader.hash);
            for u in &t.used {
                let _ = match u.root {
                    RootType::Table => write!(uses, " s{}+{} {:?}", u.slot, u.table_offset_dwords, u.kind),
                    other => write!(uses, " s{} {other:?} {:?}", u.slot, u.kind),
                };
            }
            let _ = write!(uses, " ] picked by a root constant: {}", t.host.root_indexed_views.len());
            // Table entries the shader names.
            for u in t.used.iter().filter(tables).filter(|u| matches!(u.kind, ResourceType::Srv | ResourceType::Uav)) {
                let Some(table) = table_of(u.slot) else {
                    continue;
                };
                let source = format!("s{}+{} ({:?}, entry {})", u.slot, u.table_offset_dwords, u.kind, u.table_offset_dwords / 8);
                found.push((name, source, table + u64::from(u.table_offset_dwords) * 4, fresh.contains(&u.slot)));
            }
            // Entries picked by a root constant (as `tables` finds them).
            for index in &t.host.root_indexed_views {
                let Some(table) = table_of(index.table_slot) else {
                    continue;
                };
                let constant = self.user_value(false, index.constant_slot, RootType::Constants, index.constant_dword);
                let entry = index.byte_offset.wrapping_add(constant << index.shift) / 32;
                let source = format!(
                    "s{} entry {entry} (picked by index: constant s{}[{}])",
                    index.table_slot, index.constant_slot, index.constant_dword
                );
                found.push((name, source, table + u64::from(entry) * 32, fresh.contains(&index.table_slot)));
            }
            // Slots named by words of the draw's small constants and
            // buffers, and of the records `VICTORY_LIST_SLOTS_AT` gives
            // places for ([`SlotPlaces`]; none looked for without it).
            let places = slot_places().as_ref().ok();
            let mut words: Vec<(String, u64, u64)> = Vec::new();
            let mut material = None;
            for u in t.used.iter().filter(tables).filter(|u| u.kind != ResourceType::Sampler) {
                let Some(view) = table_of(u.slot).and_then(|table| crate::descriptor::view(table + u64::from(u.table_offset_dwords) * 4))
                else {
                    continue;
                };
                let range = match &view {
                    View::Constant(address, size) => Some((*address, u64::from(*size))),
                    other => crate::descriptor::host::buffer_range(other),
                };
                let Some((address, len)) = range.filter(|&(_, len)| len <= 2048) else {
                    continue;
                };
                words.push((format!("s{}+{}", u.slot, u.table_offset_dwords), address, len));
                if let Some(places) = places
                    && (u.slot, u.table_offset_dwords) == places.constants
                    && len >= places.offset.saturating_add(4)
                {
                    let mut word = [0u8; 4];
                    if crate::host::read_into((address + places.offset) as *const u8, &mut word) {
                        material = Some(u64::from(u32::from_le_bytes(word)));
                    }
                }
            }
            let named = |slot: u8, dwords: u32| t.used.iter().any(|u| u.root == RootType::Table && u.slot == slot && u.table_offset_dwords == dwords);
            let buffer_at = |slot: u8, dwords: u32| {
                table_of(slot)
                    .and_then(|table| crate::descriptor::view(table + u64::from(dwords) * 4))
                    .as_ref()
                    .and_then(crate::descriptor::host::buffer_range)
            };
            if let Some(places) = places
                && let Some(at) = material
                && named(places.first.0, places.first.1)
                && let Some((base, len)) = buffer_at(places.first.0, places.first.1)
            {
                let (slot, dword) = places.first;
                if at + 64 <= len {
                    words.push((format!("the material record (s{slot}+{dword} at {at:#x})"), base + at, 64));
                }
                if let Some(further) = places.further
                    && at.saturating_add(further).saturating_add(512) <= len
                {
                    words.push((format!("the material record (s{slot}+{dword} at {at:#x}+{further})"), base + at + further, 512));
                }
            }
            if let Some(places) = places
                && named(places.second.0, places.second.1)
                && let Some(slot) = t.used.iter().find(|u| u.root == RootType::Constants).map(|u| u.slot)
                && let Some(Root::Constants(values)) = roots.get(usize::from(slot))
                && let Some(&at) = values.get(places.root_constant)
                && let Some((base, len)) = buffer_at(places.second.0, places.second.1)
                && u64::from(at) + 64 <= len
            {
                let (table, dword) = places.second;
                let index = places.root_constant;
                words.push((format!("the record s{table}+{dword} at {at:#x} (root constant s{slot}[{index}])"), base + u64::from(at), 64));
            }
            for (label, address, len) in words {
                for (at, index) in Self::texture_words(heap, address, len) {
                    let Some(handle) = heap.handle(index) else {
                        continue;
                    };
                    if found.iter().any(|f| f.0 == name && f.2 == handle) {
                        continue;
                    }
                    found.push((name, format!("named by a word of {label} at +{at:#x}"), handle, false));
                }
            }
        }
        found.truncate(MOST);
        // The draw's bindings: which views it reads, in which slots.
        let key = {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            self.pipeline.hash(&mut h);
            for (_, _, handle, _) in &found {
                handle.hash(&mut h);
                // SAFETY: a descriptor slot in title memory.
                unsafe { (*handle as *const [u32; 3]).read_unaligned() }.hash(&mut h);
            }
            h.finish()
        };
        let Some(n) = crate::bind_trace::take(id, key) else {
            return;
        };
        let frame = crate::frame::current();
        let targets: Vec<String> =
            self.targets.iter().map(|&t| format!("{t:#x} {}", crate::resource::host_copy_state(t as This))).collect();
        let mut lines = vec![(
            format!(
                "{id} draw {n} frame {frame}: {} textures, {} | targets {targets:?}{uses}",
                found.len(),
                p.name
            ),
            0,
        )];
        for (stage, source, handle, dynamic) in found {
            let index = heap.index(handle);
            let view = crate::descriptor::view(handle);
            let host_slot = match (index, host) {
                _ if dynamic => "a dynamic slot of the draw's, made from this view at this draw".to_string(),
                (None, _) => "none: the entry is not in the bound view heap".to_string(),
                (Some(index), Some(host)) => format!("the title slot's: {}", host.shows_record(index, handle)),
                (Some(_), None) => "none (no host heap)".to_string(),
            };
            let queue = match (index, host) {
                (Some(index), Some(host)) if index < mirrored => Some(host.texture_slot_state(index, frame)),
                _ => None,
            };
            let (view_text, texture) = Self::trace_view(view.as_ref());
            let resource = texture.as_ref().map_or(0, |t| t.resource);
            let bound = crate::bind_trace::Bound {
                stage,
                source,
                slot: index,
                mirrored,
                host_slot,
                view: view_text,
                texture,
                queue,
            };
            lines.push((bound.text(id, n), resource));
        }
        crate::bind_trace::queue(lines);
    }

    /// The texture pack's log (`crate::texture_pack`): if this graphics
    /// draw covers most of the screen (its first viewport and its first
    /// target), the textures its shaders read through tables, and those
    /// picked by a root constant, are noted; a large one's identity is
    /// logged once. Called once the draw is bound (its views were made
    /// ready). Reads only.
    #[cfg(feature = "diagnostics")]
    fn pack_log(&self, p: &HostPipeline) {
        let Some(target) = self.targets.first().and_then(|&t| crate::resource::host_made(t as This)) else {
            return;
        };
        let Some(v) = self.viewports.first() else {
            return;
        };
        // SAFETY: a live resource.
        let d = unsafe { target.GetDesc() };
        if !crate::texture_pack::covers_screen(d.Width, d.Height, [v.TopLeftX, v.TopLeftY, v.Width, v.Height]) {
            return;
        }
        let roots = self.roots_ref(false);
        let table_of = |slot: u8| match roots.get(usize::from(slot)) {
            Some(&Root::Table(table)) => Some(table),
            _ => None,
        };
        let mut handles = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in t.used.iter().filter(|u| u.root == RootType::Table && u.kind == ResourceType::Srv) {
                if let Some(table) = table_of(u.slot) {
                    handles.push(table + u64::from(u.table_offset_dwords) * 4);
                }
            }
            for index in &t.host.root_indexed_views {
                let Some(table) = table_of(index.table_slot) else {
                    continue;
                };
                let constant = self.user_value(false, index.constant_slot, RootType::Constants, index.constant_dword);
                let entry = index.byte_offset.wrapping_add(constant << index.shift) / 32;
                handles.push(table + u64::from(entry) * 32);
            }
        }
        // Only whole slots of the bound view heap are read (an index a
        // constant picks may point anywhere).
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return;
        };
        for handle in handles {
            if heap.slot(handle).is_none() {
                continue;
            }
            // A placed view's texture only if one was made for it already.
            let resource = match crate::descriptor::view(handle) {
                Some(View::Shader(resource, _)) => resource,
                Some(View::Placed(false, r, d)) if r[0] != 1 => {
                    let address = u64::from(d[10]) | u64::from(d[11]) << 32;
                    match crate::resource::texture_at_made(address, &r) {
                        Some(texture) => texture as usize,
                        None => continue,
                    }
                }
                _ => continue,
            };
            crate::texture_pack::drawn_over_screen(resource);
        }
    }

    /// The words of the `len` bytes (at most 2 KiB) of title memory at
    /// `address` that name a texture view in `heap`: byte offset and slot.
    #[cfg(feature = "diagnostics")]
    fn texture_words(heap: &crate::descriptor::Heap, address: u64, len: u64) -> Vec<(usize, u32)> {
        let mut bytes = vec![0u8; (len.min(2048) as usize) & !3];
        if bytes.is_empty() || !crate::host::read_into(address as *const u8, &mut bytes) {
            return Vec::new();
        }
        bytes
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .enumerate()
            .filter(|&(_, index)| {
                index != 0
                    && heap
                        .handle(index)
                        .and_then(crate::descriptor::view)
                        .as_ref()
                        .is_some_and(crate::descriptor::host::is_texture_view)
            })
            .map(|(i, index)| (i * 4, index))
            .collect()
    }

    /// Diagnostics: title slot `index` entered `state`, one the frame line
    /// counts ("waiting for its memory", "uploaded in part"), found by the
    /// pass over the heap in `frame`, its texture as the pass saw it when it
    /// decided (`texture`), `mapped` of its bytes there if known: one line,
    /// at most [`STUCK_PER_SECOND`] in any second ([`stuck_due`]). Nothing is
    /// read again, made or counted as used.
    #[cfg(feature = "diagnostics")]
    fn say_stuck(index: u32, state: &str, texture: Option<&([u32; 14], u64, u64)>, mapped: Option<u64>, frame: u64) {
        if stuck_due() {
            tracing::info!("{}", stuck_text(state, index, texture, mapped, frame));
        }
    }

    /// Diagnostics: a texture object's description, the start of its memory
    /// and its bytes, for [`State::say_stuck`].
    #[cfg(feature = "diagnostics")]
    fn object_texture(object: This) -> Option<([u32; 14], u64, u64)> {
        crate::resource::texture_memory(object).map(|(at, resource)| {
            let r = resource.words();
            (r, at as u64, crate::resource::placed_bytes_from(&r, 0).unwrap_or(0))
        })
    }

    /// A view for the binding trace: in a few words, and the texture it
    /// shows as the title describes it, with its host side. Nothing is
    /// made (a placed view's texture may not be there yet).
    #[cfg(feature = "diagnostics")]
    fn trace_view(view: Option<&View>) -> (String, Option<crate::bind_trace::Texture>) {
        match view {
            None => ("none (no view in the title's slot)".into(), None),
            Some(View::Placed(unordered, r, d)) if r[0] != 1 => {
                let address = u64::from(d[10]) | u64::from(d[11]) << 32;
                let this = crate::resource::texture_at_made(address, r).unwrap_or(std::ptr::null());
                let words: [u32; 10] = d[..10].try_into().unwrap_or([0; 10]);
                let kind = if *unordered { "placed unordered access view (the host does not mirror these: null)" } else { "placed shader view" };
                (format!("{kind} dim {}", d[1]), Some(Self::trace_texture(this, address, r, &words)))
            }
            Some(View::Shader(object, desc) | View::Unordered(object, _, desc))
                if crate::resource::texture_memory(*object as This).is_some() =>
            {
                let this = *object as This;
                let Some((at, resource)) = crate::resource::texture_memory(this) else {
                    return ("a texture object".into(), None);
                };
                let r = resource.words();
                let unordered = matches!(view, Some(View::Unordered(..)));
                let text = match desc {
                    Some(d) => format!("texture object {} view dim {}", if unordered { "unordered access" } else { "shader" }, d[1]),
                    None => format!("texture object {} view (the resource's own)", if unordered { "unordered access" } else { "shader" }),
                };
                (text, Some(Self::trace_texture(this, at as u64, &r, &desc.unwrap_or([0; 10]))))
            }
            Some(v) => (format!("not a texture: {}", format!("{v:?}").chars().take(100).collect::<String>()), None),
        }
    }

    /// The texture of a view for the binding trace (`this` null: none made
    /// for the placed memory yet): its description `r` and the view's `d`.
    #[cfg(feature = "diagnostics")]
    fn trace_texture(this: This, address: u64, r: &[u32; 14], d: &[u32; 10]) -> crate::bind_trace::Texture {
        let first = crate::resource::first_mip(r, d);
        let len = crate::resource::placed_bytes_from(r, first).unwrap_or(0);
        let made = !this.is_null();
        crate::bind_trace::Texture {
            resource: this as usize,
            address,
            dimension: r[0],
            width: u64::from(r[4]) | u64::from(r[5]) << 32,
            height: r[6],
            depth: r[7] & 0xffff,
            mips: r[7] >> 16,
            samples: r[9],
            format: r[8],
            layout: r[11],
            flags: r[12],
            mode: crate::resource::tile_mode(r),
            view_format: d[0],
            first_mip: d[4],
            view_mips: d[5],
            host: if made { crate::resource::host_copy_state(this) } else { "none (no title texture yet)".into() },
            title_fills: made && crate::resource::title_fills(this),
            host_wrote: made && crate::texture::host_wrote(this as usize),
            alias: if made { crate::resource::alias_of(this).map(|a| a as usize) } else { None },
            under: crate::resource::target_over(this, address, len),
            staged: if made { crate::texture::staged_state(this) } else { None },
            upload: if made { crate::bind_trace::upload(this as usize) } else { None },
            memory: memory_state(address, len),
        }
    }

    /// The descriptors a shader reads through tables. Tables holding a
    /// constant buffer are made afresh for the draw in dynamic slots, with
    /// the constant data copied to upload memory when the list is
    /// submitted: the host wants constant buffers 256-byte aligned, the
    /// console only 16. Returns those tables' new byte offsets by root
    /// parameter; the other tables' slots are made from the title's.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once: its constant data is copied at submission either way, and the
    /// big tables the world binds would otherwise take hundreds of slots per
    /// draw, wrapping the dynamic slots onto lists not yet submitted.
    pub(crate) fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        indexed: &[RootIndexedView],
        compute: bool,
    ) -> Vec<(u8, u32)> {
        #[cfg(feature = "diagnostics")]
        if crate::null_trace::on() {
            self.past_heap_trace(used, indexed, compute);
        }
        // Under the slot map no table is made afresh: shaders read the title's
        // own offsets through the map.
        if self.mapped_heap().is_some() {
            self.mapped_tables(list, used, indexed, compute);
            return Vec::new();
        }
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = used
            .iter()
            .filter(tables)
            .filter(|u| u.kind == ResourceType::Cbv)
            .map(|u| u.slot)
            .collect();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            if heap.samplers() || (table - heap.start) % 32 != 0 {
                continue;
            }
            let Some(host) = heap.host() else {
                continue;
            };
            let mut selected: Vec<u32> = indexed
                .iter()
                .filter(|index| index.table_slot == slot)
                .map(|index| {
                    let constant = self.user_value(
                        compute,
                        index.constant_slot,
                        RootType::Constants,
                        index.constant_dword,
                    );
                    // The shader adds i32 byte offsets and shifts by 5.
                    index.byte_offset.wrapping_add(constant << index.shift) / 32
                })
                .collect();
            selected.sort_unstable();
            selected.dedup();
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8)
                .chain(selected.last().copied())
                .max()
                .unwrap_or(0)
                + 1;
            let last = table.checked_add(u64::from(n - 1) * 32);
            if n > crate::descriptor::host::DYNAMIC
                || last.and_then(|handle| heap.index(handle)).is_none()
            {
                once!(slot, n, "fresh descriptor table exceeds the available span");
                continue;
            }
            // The table and what its entries hold (record words).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    unsafe { (at as *const [u32; 3]).read_unaligned() }.hash(&mut h);
                }
                (table, n, h.finish())
            };
            if let Some(&first) = self.fresh_tables.get(&key) {
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        readable(list, u.kind, &view);
                    }
                }
                for k in selected {
                    if entries.iter().any(|u| u.table_offset_dwords / 8 == k) {
                        continue;
                    }
                    let handle = table + u64::from(k) * 32;
                    if let Some(view) = crate::descriptor::view(handle)
                        && let Some(kind) = view_kind(&view)
                    {
                        readable(list, kind, &view);
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = host.dynamic(n) else {
                // No room: the draw must not run with the title's own
                // offsets (slots never made for it); `bind` drops it.
                made.push((slot, u32::MAX));
                continue;
            };
            self.fresh_tables.insert(key, first);
            self.dynamic
                .push((std::ptr::from_ref(heap) as usize, first, n));
            // The title's own slots of this table: what the draw reads of
            // them it brings up to date here, in the dynamic slots.
            let title_first = heap.index(table).filter(|first| first + n <= host.count());
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    // No bytes: the null view, and no scratch (whose bytes an
                    // earlier list may have filled).
                    Some(View::Constant(_, 0)) => host.constant_at(index, 0, 0),
                    Some(View::Constant(address, size)) => {
                        if table_constant_scratch(list, host, index, address, size).is_none() {
                            host.make_null(index);
                        }
                    }
                    Some(view) => {
                        host.make_at(index, &view);
                        let kind = used.map(|u| u.kind).or_else(|| {
                            selected.binary_search(&k).ok().and_then(|_| view_kind(&view))
                        });
                        if let Some(kind) = kind {
                            readable(list, kind, &view);
                        }
                        // The title's slot is a table's: out of the queue
                        // of textures picked by index if the draw read it
                        // (its texture is up to date now), and made in
                        // place as the texture pass would have made it,
                        // for a shader that picks it by index all the same.
                        if let Some(title) = title_first
                            && crate::descriptor::host::is_texture_view(&view)
                        {
                            host.texture_slot_tabled(title + k, kind.is_some());
                            if kind.is_some() {
                                host.mirror(title + k, table + u64::from(k) * 32, true);
                            }
                        }
                    }
                    None => {
                        host.make_null(index);
                        if used.is_some() {
                            once!(slot, "a table entry without a view");
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_made(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        attempt();
        self.window = None;
        if live_skip(self.pipeline as This, false) {
            crate::live::skipped(false);
            return;
        }
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let ready = self.prepare(list, false);
        // A pipeline with the rectangle geometry stage draws the title's
        // vertices as they are; any other gets its rectangles expanded.
        let rects = self.topology == RECTANGLES && !self.completes_rects();
        let ran = ready && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            #[cfg(feature = "diagnostics")]
            self.stamp_call(list);
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// With `VICTORY_GPU_STAMPS=2`: a timestamp after the call just
    /// recorded, with the bound pipeline's program ([`HostList::stamp_call`]).
    #[cfg(feature = "diagnostics")]
    fn stamp_call(&self, list: &HostList) {
        // The pass under way ran a call (`VICTORY_GPU_PASSES=1`).
        list.pass_called();
        if list.stamps_calls()
            && let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This)
        {
            list.stamp_call(p);
        }
    }

    /// Whether the bound pipeline completes rectangles itself, in the
    /// geometry stage rectangle pipelines get (`HostPipeline::rects`).
    fn completes_rects(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.rects)
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || u64::from(first + count) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // The vertices are read in place, at submission: no change
                // of mapping under the read (see
                // `crate::watch::title_memory_reading`).
                let _mapped = crate::watch::title_memory_reading(from, n * 3 * stride);
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        attempt();
        self.window = None;
        if live_skip(self.pipeline as This, false) {
            crate::live::skipped(false);
            return;
        }
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        // Indexed rectangles are drawn by the rectangle geometry stage
        // alone (nothing expands them on the CPU).
        let lost = self.topology == RECTANGLES && !self.completes_rects();
        if lost {
            once!("indexed rectangles not drawn");
        }
        let ran = !lost && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            #[cfg(feature = "diagnostics")]
            self.stamp_call(list);
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        let ran = self.run_indirect(list, (kind, stride, max), arguments, count);
        // A listed frame: a dropped execution is named (one of many meshes
        // is a whole class of geometry missing). Its records are not read:
        // they are in a title buffer, whose state in the list is not
        // tracked, so no copy from it is known valid (Sol's review,
        // finding 1).
        #[cfg(feature = "diagnostics")]
        if !ran && crate::capture::recording_listed() {
            let name = crate::pipeline::shaders_of(self.pipeline as This);
            let why = crate::capture::Unread::TitleBuffer;
            crate::capture::report_indirect(crate::capture::unread_line((kind, stride, max), count.is_some(), true, why, &name));
        }
        ran
    }

    fn run_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        // A dispatch's kind (2) is compute.
        if kind != 2 {
            attempt();
        }
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        };
        // A live skip switch drops the whole execution, before the argument
        // clamp can write the indirect buffer. It ran no call, but it is not
        // an untranslated pipeline's fallback: say it ran, so the drop
        // reasons stay those of missing host pipelines.
        let live_kind = match kind {
            DISPATCH => Some(true),
            DRAW | INDEXED | MESH => Some(false),
            _ => None,
        };
        if crate::live::on()
            && let Some(dispatch) = live_kind
        {
            let skip = if dispatch {
                crate::live::skip_dispatch(&p.name)
            } else {
                crate::live::skip_draw(&p.name)
            };
            if skip {
                crate::live::skipped(dispatch);
                return true;
            }
        }
        // Indirect rectangles (the title's tiled lighting, 256
        // indirect draws a frame, makes their corners from the vertex
        // number) are completed by the pipeline's geometry stage: the
        // arguments stay the title's, three vertices a rectangle. They
        // cannot be expanded on the CPU, which does not know their count.
        // shortcut: in a pipeline without that stage, a rectangle with no
        // per-vertex inputs is drawn as its first triangle (three vertices,
        // half of it), and one with vertex buffers is not drawn.
        let made_rects = !p.rects
            && kind == DRAW
            && self.topology == RECTANGLES
            && !p.inputs.iter().any(|e| !e.per_instance);
        if made_rects {
            once!("indirect rectangles without vertex inputs drawn as their first triangle");
        }
        let matches_kind = match kind {
            DRAW | INDEXED => {
                (self.topology != RECTANGLES || made_rects || p.rects) && !p.compute && !p.mesh
            }
            MESH => !p.compute && p.mesh,
            DISPATCH => p.compute,
            _ => false,
        };
        if !matches_kind {
            why(&format!(
                "pipeline mismatch (compute {}, mesh {}, topology {})",
                p.compute, p.mesh, self.topology
            ));
            return false;
        }
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find_any(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find_any(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, stride))
            .flatten()
        else {
            why("no host signature");
            return false;
        };
        if kind == MESH && (mesh_test() == 1 || !mesh_kept(self.pipeline)) {
            why("mesh draws are switched off (VICTORY_MESH_TEST=skip)");
            return false;
        }
        // A window of records clamped in one pass (`windows`): for single
        // records without a count buffer, drawn by shaders that store to no
        // buffer. Anything else ends the window: it uses the scratch buffer
        // or may write the arguments.
        const WINDOW_USE: usize = usize::MAX - 0x77;
        let offset = args.offset(arguments);
        // A dispatch that may have more groups than the host runs in one
        // dimension, by a program that takes the number of its first group.
        let parts = if kind == DISPATCH && max == 1 && count.is_none() {
            group_base(p).zip(host_signature(DISPATCH, crate::indirect::PART_BYTES as u32))
        } else {
            None
        };
        // A one-record call without a count buffer, with the clamp of runs
        // (on unless `VICTORY_CLAMP_RUNS=0`): its record is clamped by its
        // run's pass (`crate::indirect::batched`), which takes the windows'
        // place.
        let runs = crate::indirect::runs_on() && parts.is_none() && kind != DISPATCH && max == 1 && count.is_none();
        let windowed = !runs
            && windows()
            && kind != DISPATCH
            && max == 1
            && count.is_none()
            && !p.stages.iter().any(|stage| stage.translated().stores);
        // Another host list (a new one after a cut, or this one after its
        // reset) has another scratch buffer, or this one's with old bytes.
        if !windowed || list.first_use(WINDOW_USE) {
            self.window = None;
        }
        let same = |w: &Window| w.copy == args.id() && w.kind == kind && w.stride == stride;
        let hit = self
            .window
            .as_ref()
            .filter(|w| same(w) && offset >= w.base)
            .filter(|w| (offset - w.base) % u64::from(w.step) == 0 && (offset - w.base) / u64::from(w.step) < u64::from(w.records))
            .map(|w| (w.scratch.clone(), offset - w.base, false));
        // In a batch: the record's slot stays in the indirect-argument
        // state for the batch's other calls (no `done`). Not batched (none
        // could be made): clamped alone, as without the switch.
        let batch = if runs {
            args.refresh(list, arguments, len);
            crate::indirect::in_run(list, (&args.buffer, offset), (kind, stride))
        } else {
            None
        };
        let (bounded, at, in_batch) = match (hit, batch) {
            (Some(hit), _) => hit,
            (None, Some((slots, at, recorded))) => {
                if recorded {
                    self.host_pipeline = 0;
                    self.host_roots[1] = 0;
                }
                (slots, at, true)
            }
            (None, None) => {
                // A record a short step after the last window's last one
                // (the same step as that window's, if it had one): the
                // title walks the buffer, so the ones that step apart
                // after it are clamped with it.
                let (step, records) = match &self.window {
                    Some(w) if windowed && same(w) => {
                        let last = w.base + u64::from(w.records - 1) * u64::from(w.step);
                        let step = offset.saturating_sub(last);
                        let walks = step >= u64::from(stride)
                            && step <= 256
                            && step % 4 == 0
                            && (w.records == 1 || step == u64::from(w.step));
                        if walks {
                            let fit = (args.left(arguments) / step).min(WINDOW_MOST) as u32;
                            (step as u32, (if w.records == 1 { 16 } else { w.records * 2 }).min(fit).max(1))
                        } else {
                            (stride, max)
                        }
                    }
                    _ => (stride, max),
                };
                args.refresh(list, arguments, len.max(u64::from(step) * u64::from(records)));
                if let Some((s, at)) = &counter {
                    s.refresh(list, *at, 4);
                }
                // Bounded first: the clamp binds a compute pipeline of its
                // own and copies clamped active records into a scratch
                // buffer at offset 0.
                let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
                // More than the record asked for: a window clamped ahead,
                // whose unfilled slots must not spend the list's budget.
                let bounded = if parts.is_some() {
                    crate::indirect::bound_split(list, (&args.buffer, offset))
                } else if windowed && records > 1 {
                    crate::indirect::bound_ahead(list, (&args.buffer, offset), (kind, step, records))
                } else {
                    crate::indirect::bound(list, (&args.buffer, offset), count_arg, (kind, step, records))
                };
                let Some(bounded) = bounded else {
                    self.window = None;
                    why("not bounded (no clamp, or arguments not 4-byte aligned)");
                    return false;
                };
                self.host_pipeline = 0;
                self.host_roots[1] = 0;
                self.window = windowed.then(|| Window {
                    copy: args.id(),
                    kind,
                    stride,
                    step,
                    base: offset,
                    records,
                    scratch: bounded.clone(),
                });
                (bounded, 0, false)
            }
        };
        let ran = match kind {
            DRAW | INDEXED => self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true),
            DISPATCH => self.bind(list, p, true),
            _ => false,
        };
        if !ran {
            if !in_batch {
                crate::indirect::done(list, &bounded);
            }
            why("prepare failed");
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)
        });
        if kind == MESH {
            self.mesh_scissor(list, true);
        }
        // SAFETY: an open list with everything set; the scratch buffer and
        // host copies are kept alive by the list, and the signature is cached.
        unsafe {
            if let Some(((param, dword), signature)) = &parts {
                let parts = crate::indirect::split();
                crate::indirect::execute_parts(&list.list, signature, &bounded, (*param, *dword), parts);
            } else {
                // With a count buffer the call was clamped alone (`bound`:
                // no window, no batch), and reads the count the clamp left
                // after its records, not the title's.
                list.list.ExecuteIndirect(
                    &signature,
                    max,
                    &bounded,
                    at,
                    counter.as_ref().map(|_| &bounded),
                    counter.as_ref().map_or(0, |_| crate::indirect::count_at(stride, max)),
                )
            }
        };
        #[cfg(feature = "diagnostics")]
        {
            list.pass_called();
            list.stamp_call(p);
        }
        if kind == MESH {
            self.mesh_scissor(list, false);
        }
        // A listed frame: what the clamped records asked for. With a count
        // they are not read: the clamp writes no record past it, and how
        // many it wrote is known only on the card.
        #[cfg(feature = "diagnostics")]
        if crate::capture::recording_listed() {
            let name = crate::pipeline::shaders_of(self.pipeline as This);
            if counter.is_none() {
                crate::capture::read_indirect(list, &bounded, at, (kind, stride, max), name);
            } else {
                let why = crate::capture::Unread::TitleBuffer;
                crate::capture::report_indirect(crate::capture::unread_line((kind, stride, max), true, false, why, &name));
            }
        }
        if !in_batch {
            crate::indirect::done(list, &bounded);
        }
        if kind == DISPATCH {
            // It may write the records of the calls after it: the batch
            // open ends (the census's (a)).
            list.batch_end();
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// The title's culling programs by hash, each with what kind of pass it is
/// (described here, not named: names are the title's), for the walls
/// trace's census (`State::walls_census`).
const WALLS_CULL: [(&str, &str); 9] = [
    ("55679a0547649d0b", "tree instance culling"),
    ("89d08e561897d4e5", "culling of trees drawn as distant cards"),
    ("9209777e8face9dd", "grass instance culling"),
    ("17af15c4015cb4b6", "prop instance culling"),
    ("fdd56d97a806bcca", "prop batching without culling"),
    ("42f717f1b4d18e94", "plain instance culling"),
    ("1c4b3c0547e3e6fa", "far low-detail culling"),
    ("9146291b38ea9ce1", "meshlet culling argument setup"),
    ("42683eebec081520", "depth pyramid for CPU occlusion tests"),
];

/// What one bind adds to the walls trace's census: a dispatch (which of
/// [`WALLS_CULL`] it runs), a draw of the main pass under its scissor
/// rectangle, or neither.
pub(crate) enum CensusNote {
    Dispatch([bool; 9]),
    Draw([i32; 4]),
    Other,
}

/// The walls trace's census (`State::walls_census`) in `frame`.
pub(crate) fn walls_census_in(frame: u64, note: CensusNote) {
    type Census = (u64, std::collections::BTreeMap<[i32; 4], u32>, [u32; 9]);
    static CENSUS: std::sync::Mutex<Census> =
        std::sync::Mutex::new((u64::MAX, std::collections::BTreeMap::new(), [0; 9]));
    let mut census = CENSUS.lock().unwrap_or_else(|e| e.into_inner());
    // The frame before, taken with the lock; its line is made and logged on
    // the reporting thread, with neither the census nor the recording
    // thread held up by the sink.
    let before = (census.0 != frame).then(|| {
        (
            std::mem::replace(&mut census.0, frame),
            std::mem::take(&mut census.1),
            std::mem::take(&mut census.2),
        )
    });
    match note {
        CensusNote::Dispatch(runs) => {
            for (count, ran) in census.2.iter_mut().zip(runs) {
                *count += u32::from(ran);
            }
        }
        CensusNote::Draw(r) => *census.1.entry(r).or_insert(0) += 1,
        CensusNote::Other => {}
    }
    drop(census);
    let Some(line) = before.and_then(|(frame, rects, culling)| census_line(frame, rects, culling)) else {
        return;
    };
    crate::host::walls_line(move || {
        let scissors: Vec<String> = line
            .most()
            .iter()
            .map(|(r, n)| format!("{} {} {} {} x{n}", r[0], r[1], r[2], r[3]))
            .collect();
        let culling: Vec<String> = WALLS_CULL
            .iter()
            .zip(line.culling)
            .filter(|(_, n)| *n > 0)
            .map(|(c, n)| format!("{} ({}): {n}", c.1, c.0))
            .collect();
        tracing::info!(
            frame = line.frame,
            main_pass_draws = line.draws,
            rectangles = line.rectangles,
            scissors = ?scissors,
            culling = ?culling,
            "walls trace: the frame's main pass by scissor rectangle, and its culling dispatches"
        );
    });
}

/// The scissor rectangles the census's line names: those most drawn under.
const CENSUS_MOST: usize = 5;

/// The census's line of a frame, as it waits for the reporting thread.
#[derive(Debug, PartialEq)]
pub(crate) struct CensusLine {
    frame: u64,
    /// The main pass's draws, and the scissor rectangles they were under.
    draws: u32,
    rectangles: usize,
    /// The rectangles most drawn under, most first, with their draws: the
    /// first `kept` of them.
    most: [([i32; 4], u32); CENSUS_MOST],
    kept: usize,
    /// The dispatches of each of [`WALLS_CULL`].
    culling: [u32; 9],
}

impl CensusLine {
    /// The rectangles most drawn under, most first, with their draws.
    fn most(&self) -> &[([i32; 4], u32)] {
        &self.most[..self.kept]
    }
}

/// The census's line of `frame` from its draws by scissor rectangle and
/// its culling dispatches: exact totals and the [`CENSUS_MOST`] rectangles
/// most drawn under; none without a main-pass draw.
fn census_line(frame: u64, rects: std::collections::BTreeMap<[i32; 4], u32>, culling: [u32; 9]) -> Option<CensusLine> {
    // One pass, the best kept in place: on the recording thread, and the
    // line may find the reporting thread's queue full (Sol's second recheck
    // of the second landing, new fault 4: every rectangle of the frame was
    // collected first).
    let (mut draws, mut most, mut kept) = (0u32, [([0; 4], 0); CENSUS_MOST], 0);
    // Most draws first, then the lower rectangle.
    let key = |&(r, n): &([i32; 4], u32)| (std::cmp::Reverse(n), r);
    for (&r, &n) in &rects {
        draws += n;
        let at = most[..kept].partition_point(|held| key(held) < key(&(r, n)));
        if at < CENSUS_MOST {
            let end = kept.min(CENSUS_MOST - 1);
            most.copy_within(at..end, at + 1);
            most[at] = (r, n);
            kept = (kept + 1).min(CENSUS_MOST);
        }
    }
    (draws != 0).then_some(CensusLine { frame, draws, rectangles: rects.len(), most, kept, culling })
}

/// Where a compute pipeline's program takes the number of its first thread
/// group (`victory_shader::emit::USER_GROUP_BASE`): the root parameter of
/// its user data and the dword in it.
fn group_base(p: &crate::pipeline::host::HostPipeline) -> Option<(u32, u32)> {
    let stage = p.stages.first().filter(|_| p.compute)?;
    let (param, _) = stage.user_data?;
    let base = RootType::Other(victory_shader::emit::USER_GROUP_BASE);
    let word = stage.translated().host.user_data.iter().find(|u| u.root == base)?;
    Some((param, word.index))
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
fn view_kind(view: &View) -> Option<ResourceType> {
    match view {
        View::Shader(..) | View::Placed(false, ..) => Some(ResourceType::Srv),
        View::Unordered(..) | View::Placed(true, ..) => Some(ResourceType::Uav),
        _ => None,
    }
}

/// The host state a depth view needs of its texture: the read state if the
/// view writes no plane the format has (the title's read-only flags), else
/// the write state.
/// shortcut: whole textures have one state, so a view that writes one of
/// two planes counts as writing.
fn depth_state(texture: &ID3D12Resource, desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>) -> u32 {
    // D16_UNORM and D32_FLOAT: depth formats without a stencil plane.
    const ONE_PLANE: [i32; 2] = [55, 40];
    let Some(desc) = desc else {
        return DEPTH_WRITE;
    };
    let format = if desc.Format.0 == 0 {
        // SAFETY: a live texture; a view without a format has its own.
        unsafe { texture.GetDesc() }.Format.0
    } else {
        desc.Format.0
    };
    let mut needed = D3D12_DSV_FLAG_READ_ONLY_DEPTH.0;
    if !ONE_PLANE.contains(&format) {
        needed |= D3D12_DSV_FLAG_READ_ONLY_STENCIL.0;
    }
    if desc.Flags.0 & needed == needed {
        DEPTH_READ
    } else {
        DEPTH_WRITE
    }
}

fn readable(list: &HostList, kind: ResourceType, view: &View) {
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    // Of a texture the title placed, the mips the view shows.
    let first = match (&placed, view) {
        (Some(_), View::Placed(_, resource, desc)) => crate::resource::first_mip(resource, desc),
        _ => 0,
    };
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((shadow, address, len)) = crate::descriptor::host::buffer_copy(view) {
        shadow.refresh(list, address, len);
        if kind == ResourceType::Uav {
            // A shader that cannot store to buffers leaves the range as
            // the title has it: nothing to find afterwards.
            if STORES.get() || !write_skip() {
                shadow.gpu_writes(list, address, len);
            } else {
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOut, 1);
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOutBytes, len);
                }
            }
        }
    }
    match (view, kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            // A view the title placed, in another format, in the memory of
            // a texture the host writes: that texture's texels, copied on
            // the host (title memory does not have them).
            let aliased = crate::resource::alias_of(*resource as This)
                .is_some_and(|source| crate::texture::alias(list, *resource as This, source));
            if !aliased
                && let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                crate::texture::upload_from(list, *resource as This, t, first);
            }
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(*resource as This, false, false);
            // The depth target the draw only tests is read in both ways at
            // once: both read states, which its preparation asks for too.
            let (tested, _) = TESTED_DEPTH.get();
            let state = if tested != 0 && tested == *resource {
                TESTED_DEPTH.set((tested, true));
                DEPTH_READ | SHADER_RESOURCE
            } else {
                SHADER_RESOURCE
            };
            crate::list::ensure_state(list, *resource as This, state);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(list, *resource as This);
            // Unordered access to a depth texture is its colour twin's.
            if !list.twin_use(*resource as This, true, true) {
                crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
            }
        }
        _ => {}
    }
}

/// The pixel shaders whose draws get their slots listed
/// ([`State::slots_text`]): `VICTORY_LIST_SLOTS`, hash prefixes separated
/// by commas. None by default.
#[cfg(feature = "diagnostics")]
fn list_slots() -> &'static [String] {
    static LIST: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        std::env::var("VICTORY_LIST_SLOTS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    })
}

/// Where a listed shader keeps texture slot numbers besides the draw's own
/// small buffers and constants ([`State::slots_text`], and the binding
/// trace's lines for the same draws). Such places belong to one program of
/// whatever title is run, so none is built in: they come from
/// `VICTORY_LIST_SLOTS_AT` (diagnostics builds), read once:
///
/// `<first buffer>,<second buffer>,<constants>,<offset>,<root constant>[,<further>]`
///
/// The first three are table entries, `<slot>:<dword>` (the root parameter
/// slot of a table and the entry's dword in it); every number is decimal or
/// `0x` hex. Two kinds of place can be given. A record in the first buffer,
/// whose byte offset is the word at byte `<offset>` of the draw's constants
/// at `<constants>`: its first 64 bytes are listed, and 512 more from byte
/// `<further>` of it where that is given. And a record in the second
/// buffer, whose byte offset is root constant `<root constant>` of the
/// program's block: its first 64 bytes.
#[cfg(feature = "diagnostics")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SlotPlaces {
    first: (u8, u32),
    second: (u8, u32),
    constants: (u8, u32),
    offset: u64,
    root_constant: usize,
    further: Option<u64>,
}

/// `VICTORY_LIST_SLOTS_AT`'s value as places ([`SlotPlaces`]), or why it
/// gives none. Nothing is assumed for a field that is missing or no number.
#[cfg(feature = "diagnostics")]
fn parse_slot_places(value: &str) -> Result<SlotPlaces, String> {
    fn number(field: &str) -> Result<u64, String> {
        let field = field.trim();
        let parsed = match field.strip_prefix("0x").or_else(|| field.strip_prefix("0X")) {
            Some(hex) => u64::from_str_radix(hex, 16),
            None => field.parse(),
        };
        parsed.map_err(|_| format!("`{field}` is not a number"))
    }
    fn place(field: &str) -> Result<(u8, u32), String> {
        let field = field.trim();
        let (slot, dword) = field.split_once(':').ok_or_else(|| format!("`{field}` is not <slot>:<dword>"))?;
        let slot = u8::try_from(number(slot)?).map_err(|_| format!("the slot of `{field}` is too large"))?;
        let dword = u32::try_from(number(dword)?).map_err(|_| format!("the dword of `{field}` is too large"))?;
        Ok((slot, dword))
    }
    let fields: Vec<&str> = value.split(',').collect();
    if !(5..=6).contains(&fields.len()) {
        return Err(format!("{} fields where 5 or 6 are wanted", fields.len()));
    }
    Ok(SlotPlaces {
        first: place(fields[0])?,
        second: place(fields[1])?,
        constants: place(fields[2])?,
        offset: number(fields[3])?,
        root_constant: usize::try_from(number(fields[4])?).map_err(|_| format!("`{}` is too large", fields[4].trim()))?,
        further: fields.get(5).map(|f| number(f)).transpose()?,
    })
}

/// The places `VICTORY_LIST_SLOTS_AT` gives ([`SlotPlaces`]), or why there
/// are none: the variable unset, or what its value lacks.
#[cfg(feature = "diagnostics")]
fn slot_places() -> &'static Result<SlotPlaces, String> {
    static PLACES: std::sync::OnceLock<Result<SlotPlaces, String>> = std::sync::OnceLock::new();
    PLACES.get_or_init(|| slot_places_of(std::env::var("VICTORY_LIST_SLOTS_AT").ok().as_deref()))
}

/// [`slot_places`] for the variable's value (`None`: unset).
#[cfg(feature = "diagnostics")]
fn slot_places_of(value: Option<&str>) -> Result<SlotPlaces, String> {
    match value {
        None => Err("VICTORY_LIST_SLOTS_AT is not set".into()),
        Some(value) => parse_slot_places(value).map_err(|why| format!("VICTORY_LIST_SLOTS_AT: {why}")),
    }
}

/// The listing's line where no record place is known ([`slot_places`]'s
/// reason): the records are not looked for, and nothing is guessed.
#[cfg(feature = "diagnostics")]
fn places_not_given(why: &str) -> String {
    format!("\n    ~record places not given ({why}): only the draw's own small buffers and constants are listed")
}

/// A fingerprint of the 4 KiB of title memory at `at` (FNV-1a, 64 bits; of
/// 4 KiB of zeros it is always the same value, which `slots.py` knows), or
/// "unreadable".
#[cfg(feature = "diagnostics")]
fn print_4k(at: u64) -> String {
    let mut bytes = [0u8; 4096];
    if at == 0 || !crate::host::read_into(at as *const u8, &mut bytes) {
        return "unreadable".into();
    }
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for b in bytes {
        hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// The `bytes` of title memory at `at`: how many are mapped, and the
/// fingerprints of the first and the last 4 KiB (a texture's smallest
/// levels lie first, its largest last).
#[cfg(feature = "diagnostics")]
fn memory_state(at: u64, bytes: u64) -> String {
    let mapped = crate::texture::mapped_bytes(at as *const u8, bytes as usize, bytes as usize);
    let last = at + bytes.saturating_sub(4096);
    format!("memory={at:#x} mapped={mapped}/{bytes} first4k={} last4k={}", print_4k(at), print_4k(last))
}

/// Title texture `this` for the frame listing: its size, whether it has a
/// host copy (not made here), its staging buffer's state, whether the host
/// copied into it, and its memory ([`memory_state`]).
#[cfg(feature = "diagnostics")]
fn texture_state(this: This) -> String {
    let Some((at, desc)) = crate::resource::texture_memory(this) else {
        return "not a texture".into();
    };
    let words = desc.words();
    let bytes = crate::resource::placed_bytes(&words).unwrap_or(0);
    let staged = match crate::texture::staged_state(this) {
        Some((uploaded, dirty)) => format!("staged=yes uploaded={uploaded} dirty={dirty}"),
        None => "staged=no".into(),
    };
    format!(
        "{}x{} f{} levels={} host_copy={} {staged} host_written={} title_fills={} {}",
        u64::from(words[4]) | u64::from(words[5]) << 32,
        words[6],
        words[8],
        words[7] >> 16,
        crate::resource::host_made(this).is_some(),
        crate::texture::host_wrote(this as usize),
        crate::resource::title_fills(this),
        memory_state(at as u64, bytes)
    )
}

/// A view of the title's heap for the frame listing: a texture with its
/// state ([`texture_state`]; a placed one that has no resource yet with
/// its memory alone), anything else in a few words.
#[cfg(feature = "diagnostics")]
fn view_state(view: Option<&View>) -> String {
    match view {
        None => "no view".into(),
        Some(View::Placed(false, r, d)) if r[0] != 1 => {
            let address = u64::from(d[10]) | u64::from(d[11]) << 32;
            let state = match crate::resource::texture_at_made(address, r) {
                Some(this) => texture_state(this),
                None => format!(
                    "{}x{} f{} levels={} resource=none {}",
                    u64::from(r[4]) | u64::from(r[5]) << 32,
                    r[6],
                    r[8],
                    r[7] >> 16,
                    memory_state(address, crate::resource::placed_bytes(r).unwrap_or(0))
                ),
            };
            // (Last, so that readers of the older lines still find
            // theirs: the level the view starts at, and how many it shows.)
            format!("placed texture view_format={} {state} view_levels={}+{}", d[0], d[4], d[5] as i32)
        }
        Some(View::Shader(r, d)) if crate::resource::texture_memory(*r as This).is_some() => {
            format!(
                "texture object {r:#x} view_format={} {} view_levels={}",
                d.as_ref().map_or(0, |d| d[0]),
                texture_state(*r as This),
                d.as_ref().map_or("all".to_owned(), |d| format!("{}+{}", d[4], d[5] as i32))
            )
        }
        Some(v) => format!("{v:?}").chars().take(80).collect(),
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_made(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!("{}x{} f{} flags {:#x} of {r:?}", d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        Some(v @ View::Unordered(r, ..)) => match texture(*r as This) {
            Some(t) => format!("uav texture {t}"),
            None => format!(
                "uav resource {r:#x} buffer {:?}",
                crate::descriptor::host::buffer_range(v)
            ),
        },
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || format!("STENCIL REF {reference:#x}"));
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

pub(crate) fn depth_bounds(list: &HostList, min: f32, max: f32) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || {
        format!("DEPTH BOUNDS {min}..{max}")
    });
    if crate::host::host().is_some_and(|h| h.depth_bounds())
        && let Ok(l1) = list.list.cast::<ID3D12GraphicsCommandList1>()
    {
        // SAFETY: an open list on a device that supports depth bounds.
        unsafe { l1.OMSetDepthBounds(min, max) };
    }
}

#[cfg(test)]
mod census_tests {
    use super::{CENSUS_MOST, census_line, table_cbv_fill_len};

    /// What waits for the reporting thread of a frame's census is its exact
    /// totals and the rectangles most drawn under, never the frame's whole
    /// map: a sink that stalls leaves many such lines queued (Sol's recheck
    /// of the second landing, new fault 4). Rectangle `i` drawn under `i`
    /// times, for 1,000 rectangles.
    #[test]
    fn a_queued_census_line_holds_totals_and_the_most_drawn_rectangles() {
        let rects = (1..=1000).map(|i| ([i, 0, i + 1, 1], i as u32)).collect();
        let line = census_line(7, rects, [3; 9]).unwrap();
        assert_eq!((line.frame, line.draws, line.rectangles, line.culling), (7, 500_500, 1000, [3; 9]));
        let most: Vec<_> = (996..=1000).rev().map(|i| ([i, 0, i + 1, 1], i as u32)).collect();
        assert_eq!(line.most().len(), CENSUS_MOST, "rectangles queued");
        assert_eq!(line.most(), most.as_slice());
        assert_eq!(census_line(8, Default::default(), [0; 9]), None, "no main-pass draw");
    }

    /// A table constant buffer larger than a shader can read (the most
    /// is 4,096 vectors, 65,536 bytes) is requested and filled at 65,536
    /// bytes, not at its whole declared size; the last readable constant
    /// (bytes 65,520 to 65,535) is inside what is copied, and a short or
    /// empty buffer behaves as before (the 256-byte rounding stays).
    #[test]
    fn a_table_constant_buffer_fills_at_most_64_kib() {
        assert_eq!(table_cbv_fill_len(1 << 20), 65_536, "1 MiB declared");
        assert_eq!(table_cbv_fill_len(2 << 20), 65_536, "a large declared size");
        assert_eq!(table_cbv_fill_len(65_536), 65_536, "exactly the most");
        assert_eq!(table_cbv_fill_len(65_537), 65_536, "one past it");
        assert_eq!(table_cbv_fill_len(200), 200, "short, as before");
        assert_eq!(table_cbv_fill_len(0), 0, "empty, as before");
        // The fill is one contiguous copy of the length asked:
        // 65,520..65,535 are its last 16 bytes and reach untouched.
        let len = table_cbv_fill_len(1 << 20) as usize;
        let src: Vec<u8> = (0..len + 16).map(|i| (i & 0xFF) as u8).collect();
        let mut dst = vec![0u8; len];
        dst.copy_from_slice(&src[..len]);
        assert_eq!(&dst[len - 16..], &src[len - 16..len], "the last readable constants");
    }

    /// A frame of fewer rectangles than a line names keeps them all; draws
    /// tied at the line's end keep the lower rectangles; ties are told
    /// apart by the rectangle (the line is kept in one pass, in place).
    #[test]
    fn a_census_line_keeps_ties_in_rectangle_order() {
        let r = |i: i32| [i, 0, i + 1, 1];
        let few = [(r(3), 2), (r(1), 5), (r(2), 2)].into_iter().collect();
        let line = census_line(1, few, [0; 9]).unwrap();
        assert_eq!((line.draws, line.rectangles), (9, 3));
        assert_eq!(line.most(), [(r(1), 5), (r(2), 2), (r(3), 2)]);
        // Seven tied at one draw, an eighth with two, given highest first.
        let tied = (1..=7).rev().map(|i| (r(i), 1)).chain([(r(9), 2)]).collect();
        let line = census_line(2, tied, [0; 9]).unwrap();
        assert_eq!((line.draws, line.rectangles), (9, 8));
        assert_eq!(line.most(), [(r(9), 2), (r(1), 1), (r(2), 1), (r(3), 1), (r(4), 1)]);
    }
}

/// The line about a texture slot picked by index entering a state the
/// frame line counts, and its limit.
#[cfg(all(test, feature = "diagnostics"))]
mod stuck_tests {
    /// It says the slot, the texture's size, format, mip and slice counts,
    /// where its memory begins and how much of it is there, as the pass saw
    /// them, and that the draw that last bound it is not known: fixed words
    /// and numbers on one line (Sol's review of the run diagnostics: the
    /// draw it named was the scan's, not one that read the slot).
    #[test]
    fn a_stuck_slot_says_its_texture_and_that_its_last_bind_is_unknown() {
        let mut r = [0u32; 14];
        (r[0], r[4], r[6], r[7], r[8]) = (3, 1024, 512, 11 << 16 | 6, 71);
        let line = super::stuck_text("uploaded in part", 4711, Some(&(r, 0x2f00_0000, 0x2c_0000)), Some(0x4_0000), 812);
        assert_eq!(
            line,
            "texture slot uploaded in part: slot 4711, 1024x512 (dimension 3), 6 slices, 11 mips, format 71, memory from \
             0x2f000000 (0x2c0000 bytes from the first mip its view shows, 0x40000 of them mapped); the program and draw that \
             last bound it: unknown (a slot picked by index is bound through the heap, and no draw is tracked to it); found by \
             the pass over the heap in frame 812"
        );
        r[0] = 4;
        let volume = super::stuck_text("waiting for its memory", 1, Some(&(r, 0, 0)), None, 0);
        assert!(volume.contains(", 6 depth, ") && volume.contains("(0x0 bytes from the first mip its view shows)"), "{volume}");
        let none = super::stuck_text("waiting for its memory", 2, None, None, 1);
        assert!(none.contains("slot 2, its texture not known") && !none.contains('\n'), "{none}");
    }

    /// At most 100 lines lie in any second, across the edge of a clock's
    /// second too, and a moment before the last said counts as it (Sol's
    /// review: 100 a clock second let 200 through about an edge, and a
    /// thread that read the clock before the lock set the count back).
    #[test]
    fn at_most_a_hundred_lines_lie_in_any_second() {
        let mut said = std::collections::VecDeque::new();
        assert!((0..100).all(|_| super::stuck_admits(&mut said, 999)));
        assert!(!super::stuck_admits(&mut said, 1_000), "a new clock second, the same 1,000 ms");
        assert!(!super::stuck_admits(&mut said, 500), "an earlier moment than the last said");
        assert!(!super::stuck_admits(&mut said, 1_998));
        assert!(super::stuck_admits(&mut said, 1_999), "a second after the first");
        assert_eq!(said.len(), 1);
    }
}

#[cfg(all(test, feature = "diagnostics"))]
mod slot_places_tests {
    use super::{SlotPlaces, parse_slot_places, places_not_given, slot_places_of};

    /// Made-up places: none is any program's.
    const GOOD: &str = "2:24,3:40,5:8,12,3";

    #[test]
    fn five_fields_are_two_buffers_the_constants_an_offset_and_a_root_constant() {
        let want = SlotPlaces {
            first: (2, 24),
            second: (3, 40),
            constants: (5, 8),
            offset: 12,
            root_constant: 3,
            further: None,
        };
        assert_eq!(parse_slot_places(GOOD), Ok(want));
        // Spaces around a field, and a sixth field: the further window.
        assert_eq!(
            parse_slot_places(" 2:24 , 3:40 ,5:8, 12 ,3, 100 "),
            Ok(SlotPlaces {
                further: Some(100),
                ..want
            })
        );
    }

    #[test]
    fn numbers_are_decimal_or_hex() {
        assert_eq!(
            parse_slot_places("0x2:0x18,3:0X28,5:8,0xc,3,0x64"),
            parse_slot_places("2:24,3:40,5:8,12,3,100")
        );
        let places = parse_slot_places("0x2:0x18,3:0X28,5:8,0xc,3,0x64").expect("hex");
        assert_eq!((places.first, places.second, places.offset, places.further), ((2, 24), (3, 40), 12, Some(100)));
    }

    #[test]
    fn too_few_or_too_many_fields_give_no_places() {
        for (value, fields) in [("", 1), ("2:24,3:40,5:8,12", 4), ("2:24,3:40,5:8,12,3,100,7", 7)] {
            let why = parse_slot_places(value).expect_err(value);
            assert_eq!(why, format!("{fields} fields where 5 or 6 are wanted"), "{value:?}");
        }
    }

    #[test]
    fn a_field_that_is_no_number_or_no_place_gives_no_places() {
        for (value, why) in [
            ("2:24,3:40,5:8,twelve,3", "`twelve` is not a number"),
            ("2:24,3:40,5:8,12,", "`` is not a number"),
            ("2:24,3:40,5:8,12,3,0xzz", "`0xzz` is not a number"),
            ("24,3:40,5:8,12,3", "`24` is not <slot>:<dword>"),
            ("2:24,3:x,5:8,12,3", "`x` is not a number"),
            ("300:24,3:40,5:8,12,3", "the slot of `300:24` is too large"),
            ("2:24,3:40,5:-8,12,3", "`-8` is not a number"),
        ] {
            assert_eq!(parse_slot_places(value), Err(why.to_owned()), "{value:?}");
        }
    }

    /// Unset or malformed, the listing says the record places were not given, and why.
    #[test]
    fn without_places_the_listing_says_so() {
        let unset = slot_places_of(None).expect_err("unset");
        assert_eq!(unset, "VICTORY_LIST_SLOTS_AT is not set");
        assert_eq!(
            places_not_given(&unset),
            "\n    ~record places not given (VICTORY_LIST_SLOTS_AT is not set): only the draw's own small buffers and constants are listed"
        );
        let short = slot_places_of(Some("2:24,3:40")).expect_err("short");
        assert_eq!(short, "VICTORY_LIST_SLOTS_AT: 2 fields where 5 or 6 are wanted");
        assert!(places_not_given(&short).contains("(VICTORY_LIST_SLOTS_AT: 2 fields where 5 or 6 are wanted)"));
        assert_eq!(slot_places_of(Some(GOOD)).map(|p| p.root_constant), Ok(3));
    }
}
