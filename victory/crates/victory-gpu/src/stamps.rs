//! Diagnostics: where the card's queue spends its time, from GPU timestamps
//! (`VICTORY_GPU_STAMPS=1`, or `2` for that and the title lists' time by
//! program, unverified and taken only with `VICTORY_GPU_STAMPS_UNVERIFIED=1`,
//! else 1; off when unset or any other value). `VICTORY_GPU_PASSES=1` adds
//! a stamp where each of the title's passes begins and a pair around each
//! of our texture uploads in its lists, read with the rest, for the lines
//! by pass ([`crate::passes`]); alone it turns level 1 on too.
//!
//! `VICTORY_GPU_TIMING` measures another program: it sends no submission
//! ahead and runs each list alone, waited for. With this switch nothing is
//! scheduled otherwise. Pairs of timestamps go into the lists as they are
//! recorded: around each host list from its opening to its close (a title
//! list, or a present copy), around each of our clamp-and-barrier groups
//! that bound an indirect command inside them (`indirect`), around each
//! list's prelude, around each fetch of card copies' pieces by the relay,
//! and around our hash, compare and synchronous fetch lists for copies in
//! the card's memory. The same lists resolve them into read-back memory.
//!
//! Each send of such a list is noted when it goes to the queue, with the
//! fence value that says the card has run it; its pairs are read only once
//! that fence has reached the value on a live device ([`Stamps::landed`]).
//! A pair is a span of the queue's clock: elapsed time between two points
//! on one queue, not time the card spent on that work alone (a span holds
//! the stalls and barriers inside it, and work of other queues may run).
//!
//! The line is made of windows of the card's clock ([`line`]). A window's
//! end is an estimate of where the sends not yet read begin: the moment the
//! oldest of them went to the queue, on the CPU's counter, put on the
//! card's clock by the queue's calibration, less 1 ms ([`Seen::window`]).
//! A send's work can begin before that, and a send unread after 10 s ages
//! out and no longer holds the windows (below). In a closed window each
//! tick of the clock is counted once: for the first kind of span read by
//! its close over it in [`FIRST`], or as uncovered queue time when none is,
//! so the parts add up to the window by construction. A span that crosses
//! the window's end is split there; a pair is counted in the window its end
//! lies in, and the window's frames are the present copies read with their
//! end in it.
//!
//! A flaw found with a send (a pair lost, a list without its own, a send
//! left out or not read) counts in every window that closes after it is
//! found and ends after the moment the send went, until one ends past the
//! moment it was found: a send read after its fence ran in that stretch,
//! as far as the calibration places it; one given up on with the device
//! removed, its fence never reached, is not known to have run at all. A
//! send unread after 10 s is given up on: windows close past it, and each
//! that does after it went says so until it is read. A window with a flaw
//! found by its close says it is incomplete, and gives no figures; a flaw
//! found after a line was printed does not reach it, and that line's
//! figures can miss the send's work. A window without one says only that
//! no loss was detected within estimated bounds at its close: a send's
//! work can begin before the end of a window that closed without it, and
//! the line that reads it later says how much lies before its window, in
//! none ([`Seen::deliver`]).
//!
//! Not stamped, so uncovered queue time when they run: the resolves of the
//! stamps themselves, lists of ours run under the submission lock
//! (diagnostics), our other lists (indirect, in-place and tile work), and
//! the hash list of copies in the card's memory that a synchronous
//! submission without a title list sends under `VICTORY_GPU_TIMING` (no
//! signal follows it there).
//!
//! # By program (`VICTORY_GPU_STAMPS=2`)
//!
//! Unverified, and deferred: taken only with
//! `VICTORY_GPU_STAMPS_UNVERIFIED=1`, else the level is 1 and one line says
//! so ([`switch`]), until it is compared with level 1 on matched hardware
//! and these are fixed (Sol's fifth read of the frame timers):
//! - a late read after a second summary: its bounds replace the first's, whose loss goes unsaid; a read after the tally is taken and before its lines are logged is not in "up to this line";
//! - a late flaw with no readable span leaves a kept window it lies in in the tally;
//! - a late zero-length end at the first window's beginning leaves that window in the tally (`read_late` needs `from < end`);
//! - a tally whose last kept window a late read removes has no fixture of its own;
//! - the attribution sentence names only our clamp groups' priority: our other lists over a title list take its ticks too ([`FIRST`]).
//!
//! Everything above, and one more timestamp in a title list after each
//! draw or dispatch the title records ([`Stamps::call`]), with the program
//! the call ran: the host pipeline's shader ids and its stage ([`program`]).
//! A title list's span is cut at those stamps into consecutive parts, one
//! a call, each from the stamp before the call (the list's opening for the
//! first) to the call's own ([`parts_of`]); the part after the last call
//! (copies, clears and barriers recorded after it, our closing work, a list
//! with no call) is no program's ([`AFTER`]). A stamp is used only inside
//! its list's span and not before the stamp before it: a lost or unusable
//! one leaves the stretch up to the next usable stamp in no part, and the
//! window says so; the call after it still counts, at its own stamp. So
//! every tick of a list's span has at most one owner and the parts add up
//! to the span with what is in no part. Our clamp groups inside the list
//! keep their own pairs, and their ticks stay theirs ([`FIRST`]). The parts
//! go through the same windows, so the programs' ticks in a window add up
//! to its title-list ticks exactly ([`ByProgram`]); a call counts in the
//! window its stamp lies in. The lines by program take the windows of five
//! seconds together that had no flaw at their close and in which no span
//! or part read later lies, up to the lines; one read after the lines that
//! lies in a window they counted is said in the next ([`Tally`],
//! [`Seen::read_late`], [`programs_text`]).
//!
//! What this is: title-list queue time attributed to calls between
//! completion marks (bottom-of-pipe timestamps on one queue) in the windows
//! counted, after clipping to the windows, our clamp groups' priority and
//! the rule for overlapping lists' parts below; not the time the card spent
//! running a program. A stamp is written once the work recorded before it
//! has finished, so the stamp before a call already waited for the call
//! before: a call whose work overlapped that one's is partly done by then
//! and can show a short interval or none, while the call before holds its
//! whole tail; contention between overlapping calls can lengthen either.
//! Whatever the queue does between two stamps is charged to the call after
//! the first: a barrier, a copy, a clear or a pipeline change the title
//! records before a call (the stage "pixel" means a pipeline with a pixel
//! shader, and its whole interval is counted), our own work there (texture
//! transitions, rectangles, a twin pass), and a stall waiting for earlier
//! work or for another queue. The first call's part begins at the list's
//! opening stamp, with whatever the list does before it; the part after
//! the last call runs to the list's closing stamp, our closing work
//! included; the stamps' resolves come after that, in no list. A call is
//! one title call: a dispatch we split into parts, and an indirect
//! execution, split or of many records or of none, count once. The stamps
//! can themselves keep calls from overlapping, and change what they
//! measure. A program is a host pipeline's shader ids and stage: pipelines
//! that differ only in their state, targets or sample count count as one,
//! and one pipeline's calls in different passes count together. Where two
//! lists' parts overlap on the queue the tick goes to the part begun last,
//! by its own beginning whatever windows cut it, then to the one that ends
//! first, then to the lower program number.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Mutex, OnceLock};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

/// The line the log gets once when `VICTORY_GPU_STAMPS=2` is asked for
/// without `VICTORY_GPU_STAMPS_UNVERIFIED=1` ([`switch`]).
const DEFERRED: &str = "GPU timestamps: VICTORY_GPU_STAMPS=2 is deferred and runs as 1: the time by program is unverified until \
                        its comparison with level 1 on matched hardware and its reporting fixes are done \
                        (VICTORY_GPU_STAMPS_UNVERIFIED=1 asks for it anyway)";

/// What the switches ask for, and the line to log: `VICTORY_GPU_STAMPS`
/// `1` the windows by kind, `2` those and the title lists' time by program,
/// nothing else (0: off). Level 2 is unverified (Sol's fifth read of the
/// frame timers): it is taken only with `VICTORY_GPU_STAMPS_UNVERIFIED=1`
/// (`unverified`), else the level is 1 and [`DEFERRED`] says why.
fn switch(value: Option<&std::ffi::OsStr>, unverified: Option<&std::ffi::OsStr>) -> (u8, Option<&'static str>) {
    match value {
        Some(v) if v == "1" => (1, None),
        Some(v) if v == "2" && unverified.is_some_and(|u| u == "1") => (2, None),
        Some(v) if v == "2" => (1, Some(DEFERRED)),
        _ => (0, None),
    }
}

/// The stamps' level: 0 off, 1 or 2 (`VICTORY_GPU_STAMPS`, and
/// `VICTORY_GPU_STAMPS_UNVERIFIED` for 2, [`switch`]).
fn level() -> u8 {
    #[cfg(test)]
    LEVEL_ASKS.with(|n| n.set(n.get() + 1));
    #[cfg(test)]
    if let Some(level) = FORCED.with(std::cell::Cell::get) {
        return level;
    }
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        let (level, said) = switch(
            std::env::var_os("VICTORY_GPU_STAMPS").as_deref(),
            std::env::var_os("VICTORY_GPU_STAMPS_UNVERIFIED").as_deref(),
        );
        if let Some(said) = said {
            tracing::warn!("{said}");
        }
        // The passes stand on level 1 (`VICTORY_GPU_PASSES=1` alone asks
        // for both).
        if level == 0 && passes_asked() { 1 } else { level }
    })
}

/// Whether `VICTORY_GPU_PASSES=1` asks for the card's time by pass
/// ([`crate::passes`]).
fn passes_asked() -> bool {
    #[cfg(test)]
    if let Some(on) = FORCED_PASSES.with(std::cell::Cell::get) {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_PASSES").is_some_and(|v| v == "1"))
}

/// Whether the title's passes are stamped and our texture uploads in its
/// lists (`VICTORY_GPU_PASSES=1`, with the stamps on).
pub(crate) fn passes_on() -> bool {
    passes_asked() && on()
}

/// Whether the stamps are on (`VICTORY_GPU_STAMPS=1` or `2`).
pub(crate) fn on() -> bool {
    level() >= 1
}

/// Whether each title call is stamped too (`VICTORY_GPU_STAMPS=2`).
pub(crate) fn calls_on() -> bool {
    level() >= 2
}

#[cfg(test)]
thread_local! {
    /// Tests: the stamps' level for this thread, whatever the environment
    /// says (`None`: as it says).
    static FORCED: std::cell::Cell<Option<u8>> = const { std::cell::Cell::new(None) };
    /// Tests: the pool of spare chunks locked on this thread.
    static SPARE_LOCKS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Tests: what this thread notes and reads goes here, not to the line.
    static SINK: std::cell::RefCell<Option<Seen>> = const { std::cell::RefCell::new(None) };
    /// Tests: no query heap can be had on this thread.
    static NO_CHUNKS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tests: chunks taken on this thread (made or from the spare ones).
    static TAKEN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Tests: the device counts as removed for this thread's landings.
    static REMOVED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Tests: the times this thread looked at the stamps' level.
    static LEVEL_ASKS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Tests: the passes on or off for this thread, whatever the
    /// environment says (`None`: as it says).
    static FORCED_PASSES: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// What a pair of timestamps brackets.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    /// A title's host list from its opening to its close.
    List = 0,
    /// A clamp-and-barrier group of ours inside a host list.
    Clamp = 1,
    /// A host list's prelude.
    Prelude = 2,
    /// A fetch of card copies' pieces by the relay.
    Relay = 3,
    /// The copy of a frame to the swap chain's buffer (a host list of ours).
    Present = 4,
    /// Our list that sends copies in the card's memory up and hashes them.
    Hash = 5,
    /// Our compare of copies in the card's memory after a submission.
    Compare = 6,
    /// A fetch of card copies' pieces after a synchronous submission.
    Fetch = 7,
    /// Our copy of a texture's staging buffer into its host copy, with its
    /// barrier, inside a title list (stamped only with
    /// `VICTORY_GPU_PASSES=1`, [`crate::passes`]).
    Upload = 8,
}

pub(crate) const KINDS: usize = 9;

/// Where spans overlap, a tick counts once, for the first of these with a
/// span over it: a clamp group or a texture upload before the list around
/// it, the lists of ours before a title list whose end overlaps them.
const FIRST: [Kind; KINDS] = [
    Kind::Clamp,
    Kind::Upload,
    Kind::Prelude,
    Kind::Relay,
    Kind::Fetch,
    Kind::Hash,
    Kind::Compare,
    Kind::Present,
    Kind::List,
];

/// What a title call ran (`VICTORY_GPU_STAMPS=2`): a number for a host
/// pipeline's shader ids and stage ([`program`]); [`AFTER`] for the part of
/// a list after its last call.
pub(crate) type Program = u32;

/// The part of a title list after its last call: no program.
pub(crate) const AFTER: Program = 0;

/// A program's stage as the lines by program print it: what the host
/// pipeline is.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum ProgramStage {
    /// A compute pipeline.
    Compute,
    /// A vertex pipeline with a pixel shader.
    Pixel,
    /// A vertex pipeline without one.
    Vertex,
    /// A mesh pipeline (with a pixel shader or not).
    Mesh,
}

impl ProgramStage {
    fn word(self) -> &'static str {
        match self {
            ProgramStage::Compute => "compute",
            ProgramStage::Pixel => "pixel",
            ProgramStage::Vertex => "vertex",
            ProgramStage::Mesh => "mesh",
        }
    }
}

/// The programs by number (the first is number 1) and their numbers.
struct Names {
    by_number: Vec<(ProgramStage, String)>,
    numbers: BTreeMap<(ProgramStage, String), Program>,
}

static NAMES: Mutex<Names> = Mutex::new(Names { by_number: Vec::new(), numbers: BTreeMap::new() });

/// The number of the program with `stage` and the shader ids `shaders` (as
/// the log prints a host pipeline's: `[vertex …, pixel …]`), the same for
/// every host pipeline with both. Asked once per host pipeline.
pub(crate) fn program(stage: ProgramStage, shaders: &str) -> Program {
    let mut names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    let key = (stage, shaders.to_owned());
    if let Some(&n) = names.numbers.get(&key) {
        return n;
    }
    names.by_number.push(key.clone());
    let n = names.by_number.len() as Program;
    names.numbers.insert(key, n);
    n
}

/// The stage and shader ids of program `n` ([`program`]).
fn program_name(n: Program) -> Option<(ProgramStage, String)> {
    let names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    names.by_number.get((n as usize).checked_sub(1)?).cloned()
}

/// Timestamps in one query heap.
const CHUNK: u32 = 256;

/// A query heap of [`CHUNK`] timestamps and the read-back memory they are
/// resolved into, mapped for its life.
struct Chunk {
    heap: ID3D12QueryHeap,
    back: ID3D12Resource,
    data: *const u64,
}

// SAFETY: D3D12 objects (free-threaded) and the mapping of their read-back
// memory, read by whoever holds the chunk.
unsafe impl Send for Chunk {}

/// Chunks no list holds (a list's go back here when it goes), for the next
/// list that needs one; at most [`SPARE_MOST`].
static SPARE: Mutex<Vec<Chunk>> = Mutex::new(Vec::new());
const SPARE_MOST: usize = 1024;

fn chunk() -> Option<Chunk> {
    #[cfg(test)]
    TAKEN.with(|n| n.set(n.get() + 1));
    #[cfg(test)]
    if NO_CHUNKS.with(std::cell::Cell::get) {
        return None;
    }
    #[cfg(test)]
    SPARE_LOCKS.with(|n| n.set(n.get() + 1));
    if let Some(chunk) = SPARE.lock().unwrap_or_else(|e| e.into_inner()).pop() {
        return Some(chunk);
    }
    make_chunk()
}

fn make_chunk() -> Option<Chunk> {
    let device = &crate::host::host()?.device;
    let mut heap: Option<ID3D12QueryHeap> = None;
    let mut back: Option<ID3D12Resource> = None;
    let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: u64::from(CHUNK) * 8,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    // SAFETY: plain object creation; the read-back buffer stays mapped
    // while it lives (read-back memory may be).
    unsafe {
        device
            .CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: CHUNK, NodeMask: 0 }, &mut heap)
            .ok()?;
        device
            .CreateCommittedResource(&readback, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back)
            .ok()?;
        let back = back?;
        let mut data = std::ptr::null_mut();
        back.Map(0, None, Some(&mut data)).ok()?;
        Some(Chunk { heap: heap?, back, data: data.cast::<u64>().cast_const() })
    }
}

/// The CPU's performance counter now, and its ticks a second.
fn cpu_clock() -> (u64, u64) {
    use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
    let (mut now, mut frequency) = (0i64, 0i64);
    // SAFETY: two plain queries into our locals.
    unsafe {
        QueryPerformanceCounter(&mut now);
        QueryPerformanceFrequency(&mut frequency);
    }
    (now.max(0) as u64, frequency.max(0) as u64)
}

/// What went wrong with pairs, counted where it was found: a window it
/// falls in is incomplete.
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Flaws {
    /// Pairs begun whose end, or whose beginning, had no query heap.
    pub(crate) lost: u64,
    /// Lists whose own pair (opening to close) is missing: their clamp
    /// groups count, the rest of their time is uncovered.
    pub(crate) unclosed: u64,
    /// Sends of a list sent again before its earlier send was read: the
    /// list's pairs are not read for either (its read-back memory holds one
    /// of them, which is not known).
    pub(crate) excluded: u64,
    /// Sends whose fence went to `u64::MAX` with the device removed:
    /// nothing read.
    pub(crate) unverified: u64,
    /// Landings asked for before the send's fence reached its value:
    /// nothing read then.
    pub(crate) early: u64,
    /// Sends forgotten unread (the list was reset or let go first).
    pub(crate) dropped: u64,
    /// Sends still unread after [`STALE_S`]: a window may close past them.
    pub(crate) stale: u64,
    /// Pairs whose end came before their beginning (not counted).
    pub(crate) reversed: u64,
    /// Clamp groups not inside their list's own pair (counted as found).
    pub(crate) outside: u64,
    /// Stamps after a title call that had no query heap
    /// (`VICTORY_GPU_STAMPS=2`): the part they would have ended goes to no
    /// program, and their call is not counted.
    pub(crate) calls_lost: u64,
    /// Stamps after a title call before the stamp before them or outside
    /// their list's span (`VICTORY_GPU_STAMPS=2`): as a lost one.
    pub(crate) calls_bad: u64,
}

impl Flaws {
    fn add(&mut self, o: Flaws) {
        self.calls_lost += o.calls_lost;
        self.calls_bad += o.calls_bad;
        self.lost += o.lost;
        self.unclosed += o.unclosed;
        self.excluded += o.excluded;
        self.unverified += o.unverified;
        self.early += o.early;
        self.dropped += o.dropped;
        self.stale += o.stale;
        self.reversed += o.reversed;
        self.outside += o.outside;
    }

    fn any(&self) -> bool {
        *self != Flaws::default()
    }

    /// Whether the window's figures by kind cannot be established: a flaw
    /// other than of the stamps after title calls.
    fn kinds(&self) -> bool {
        Flaws { calls_lost: 0, calls_bad: 0, ..*self }.any()
    }
}

/// A send not read yet.
struct Sending {
    id: u64,
    /// When it went to the queue, on the CPU's counter.
    at: u64,
    /// What says the card has run it.
    fence: ID3D12Fence,
    value: u64,
    /// The pairs recorded for this send alone (a prelude's), and those lost.
    once: Vec<(Kind, u32, u32)>,
    once_lost: u64,
    /// The frame it went in, if the passes count it
    /// ([`crate::passes::sent`]): what it reads goes to that frame.
    frame: Option<u64>,
}

/// The timestamps of one list (with its preludes), or of one of our lists:
/// the pairs stamped since it was opened, read for each of its sends once
/// the card has run it.
#[derive(Default)]
pub(crate) struct Stamps {
    chunks: Vec<Chunk>,
    /// Slots stamped since the list was opened, and of them resolved.
    used: u32,
    resolved: u32,
    /// What the list's own pair is ([`Kind::List`] unless said).
    class: Option<Kind>,
    /// The list was opened with the stamps on: its own pair is owed, and
    /// the slot stamped at the opening.
    owed: bool,
    opened: Option<u32>,
    /// The list's own pair, once it is closed.
    main: Option<(u32, u32)>,
    /// Pairs inside the list (or of our lists), kept until it is opened
    /// again: every send of the recording reads them. And those lost.
    kept: Vec<(Kind, u32, u32)>,
    kept_lost: u64,
    /// Pairs for the next send alone (preludes), and those lost.
    once: Vec<(Kind, u32, u32)>,
    once_lost: u64,
    /// Sends not read yet, oldest first.
    sends: VecDeque<Sending>,
    /// A send went while an earlier one was unread ([`Flaws::excluded`]).
    clash: bool,
    /// The list was opened with a stamp owed after each title call
    /// (`VICTORY_GPU_STAMPS=2`).
    per_call: bool,
    /// The stamps after the title's calls since the list was opened, in
    /// recorded order, each with the program its call ran; `None` where the
    /// stamp had no query heap.
    calls: Vec<(Option<u32>, Program)>,
    /// The list was opened with the passes on (`VICTORY_GPU_PASSES=1`):
    /// the stamps that begin its passes since, in recorded order, each with
    /// its pass's name (`None` where the stamp had no query heap); what cut
    /// the pass now under way ([`Stamps::pass`]), and the passes begun.
    passes: bool,
    marks: Vec<(Option<u32>, crate::passes::PassName, u32)>,
    pass_key: Option<u64>,
    /// The passes begun so far by what names them, for each one's place.
    named: Vec<(crate::passes::By, u32)>,
}

/// A part of a title list between two of its stamps
/// (`VICTORY_GPU_STAMPS=2`): the program of the call it ends, the calls it
/// counts (one, none for [`AFTER`]), and its beginning and end on the
/// card's clock.
pub(crate) type Part = (Program, u32, u64, u64);

/// The parts of a title list's own span `(begun, end)` between the stamps
/// after its calls, `calls` in recorded order (each the stamp's value,
/// `None` where it was lost, and its call's program): a part for each call,
/// from the stamp before it (the list's opening for the first) to its own;
/// then [`AFTER`]'s, from the last stamp to the list's close. A stamp is
/// used only if it lies inside the span and not before the stamp before it
/// (Sol's third read of the frame timers: one past the list's close made a
/// part outside it, which took another list's ticks). A lost or unusable
/// stamp ends no part: the stretch from the stamp before it to the next
/// usable one is in no part, and the call after that still counts, at its
/// own stamp, with no time (its end and program are known: it was not
/// counted). Equal stamps make a part of no time that counts its call.
fn parts_of((begun, end): (u64, u64), calls: impl IntoIterator<Item = (Option<u64>, Program)>, flaws: &mut Flaws) -> Vec<Part> {
    let mut parts = Vec::new();
    // Where the next part begins (none after a lost or unusable stamp), and
    // the last usable stamp.
    let (mut from, mut last) = (Some(begun), begun);
    for (at, program) in calls {
        match at {
            None => {
                flaws.calls_lost += 1;
                from = None;
            }
            Some(at) if at < last || at > end => {
                flaws.calls_bad += 1;
                from = None;
            }
            Some(at) => {
                parts.push((program, 1, from.unwrap_or(at), at));
                (from, last) = (Some(at), at);
            }
        }
    }
    if let Some(a) = from {
        parts.push((AFTER, 0, a, end));
    }
    parts
}

// SAFETY: as [`Chunk`]; the fences are free-threaded.
unsafe impl Send for Stamps {}

impl Drop for Stamps {
    fn drop(&mut self) {
        let unread = !self.sends.is_empty();
        self.forget();
        // Nothing stamped (the switch off, or our lists' stamps unused): no
        // lock (Sol's recheck of the frame timers, Question 3).
        if self.chunks.is_empty() {
            return;
        }
        let chunks = std::mem::take(&mut self.chunks);
        if unread {
            // The card may still resolve into them: let go after a running
            // submission.
            crate::host::retire(chunks);
        } else {
            #[cfg(test)]
            SPARE_LOCKS.with(|n| n.set(n.get() + 1));
            let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
            let room = SPARE_MOST.saturating_sub(spare.len());
            spare.extend(chunks.into_iter().take(room));
        }
    }
}

impl Stamps {
    /// A timestamp into `list` at the next slot: its slot.
    fn stamp(&mut self, list: &ID3D12GraphicsCommandList) -> Option<u32> {
        let at = self.used;
        while self.chunks.len() <= (at / CHUNK) as usize {
            self.chunks.push(chunk()?);
        }
        let chunk = &self.chunks[(at / CHUNK) as usize];
        // SAFETY: an open list and our query heap.
        unsafe { list.EndQuery(&chunk.heap, D3D12_QUERY_TYPE_TIMESTAMP, at % CHUNK) };
        self.used += 1;
        Some(at)
    }

    /// The beginning of a pair, stamped into `list`: its slot.
    pub(crate) fn begin(&mut self, list: &ID3D12GraphicsCommandList) -> Option<u32> {
        self.stamp(list)
    }

    /// Sends not read are forgotten ([`Flaws::dropped`]).
    fn forget(&mut self) {
        for send in std::mem::take(&mut self.sends) {
            // A list is reset or let go only after its sends have run.
            let now = cpu_clock().0;
            with_seen(|seen| seen.deliver(send.id, send.at, now, Vec::new(), Flaws { dropped: 1, ..Flaws::default() }));
            if let Some(frame) = send.frame {
                crate::passes::lost(frame);
            }
        }
        self.clash = false;
    }

    /// Starts again (the list was reset or made, or is recorded anew): what
    /// it stamped before is forgotten. What it is stays.
    pub(crate) fn restart(&mut self) {
        self.forget();
        (self.used, self.resolved, self.owed, self.opened, self.main) = (0, 0, false, None, None);
        self.kept.clear();
        self.once.clear();
        (self.kept_lost, self.once_lost) = (0, 0);
        self.per_call = false;
        self.calls.clear();
        (self.passes, self.pass_key) = (false, None);
        self.marks.clear();
        self.named.clear();
    }

    /// The list was opened: [`Stamps::restart`], and the opening stamped.
    /// With `VICTORY_GPU_STAMPS=2` a stamp is owed after each title call;
    /// with `VICTORY_GPU_PASSES=1` one where each pass begins.
    pub(crate) fn open(&mut self, list: &ID3D12GraphicsCommandList) {
        self.restart();
        self.owed = true;
        self.per_call = calls_on();
        self.passes = passes_on();
        self.opened = self.stamp(list);
    }

    /// A title call is about to be recorded into `list` (a list opened with
    /// `VICTORY_GPU_PASSES=1`), cut by `key` ([`crate::passes`]), running
    /// `program` (`named`: its pipeline has shader ids), in a title list
    /// whose label hashes to `label` if it has one, in `fragment` of its
    /// recording: if the key is not the pass's under way, a timestamp ends
    /// that pass and begins one, named by the label, else by the program,
    /// else not, with its fragment and its place among the passes so named
    /// in this list. It has no call until [`Stamps::pass_called`].
    pub(crate) fn pass(
        &mut self,
        list: &ID3D12GraphicsCommandList,
        key: u64,
        program: Program,
        named: bool,
        label: Option<u32>,
        fragment: u32,
    ) {
        use crate::passes::{By, PassName};
        if !self.passes || self.pass_key == Some(key) {
            return;
        }
        self.pass_key = Some(key);
        let by = match label {
            Some(label) => By::Label(label),
            None if named => By::Program(program),
            None => By::Unnamed,
        };
        let nth = match self.named.iter_mut().find(|n| n.0 == by) {
            Some(n) => {
                n.1 += 1;
                n.1
            }
            None => {
                self.named.push((by, 1));
                1
            }
        };
        let slot = self.stamp(list);
        self.marks.push((slot, PassName { by, fragment, nth }, 0));
    }

    /// A title call was recorded in the pass under way.
    pub(crate) fn pass_called(&mut self) {
        if let Some(mark) = self.marks.last_mut() {
            mark.2 += 1;
        }
    }

    /// A title call that ran `program` was just recorded into `list` (a
    /// list opened with `VICTORY_GPU_STAMPS=2`): a timestamp after it ends
    /// the part of the list that is the call's.
    pub(crate) fn call(&mut self, list: &ID3D12GraphicsCommandList, program: Program) {
        if self.per_call {
            let slot = self.stamp(list);
            self.calls.push((slot, program));
        }
    }

    /// The parts of the list's own span between its call stamps
    /// (`VICTORY_GPU_STAMPS=2`, [`parts_of`]).
    fn parts(&self, span: (u64, u64), flaws: &mut Flaws) -> Vec<Part> {
        parts_of(span, self.calls.iter().map(|&(slot, program)| (slot.and_then(|s| self.value(s)), program)), flaws)
    }

    /// Slots stamped since the list was opened (tests).
    #[cfg(test)]
    pub(crate) fn slots(&self) -> u32 {
        self.used
    }

    /// What the list's own pair is.
    pub(crate) fn class(&mut self, kind: Kind) {
        self.class = Some(kind);
    }

    /// The end of a pair of `kind` that `begun` started, stamped in `list`;
    /// if `recorded` is false the work between recorded nothing and there is
    /// no pair (its beginning stays unread).
    pub(crate) fn end(&mut self, kind: Kind, begun: Option<u32>, list: &ID3D12GraphicsCommandList, recorded: bool) {
        if !recorded {
            return;
        }
        let end = begun.and_then(|_| self.stamp(list));
        let (pairs, lost) = if kind == Kind::Prelude {
            (&mut self.once, &mut self.once_lost)
        } else {
            (&mut self.kept, &mut self.kept_lost)
        };
        match (begun, end) {
            (Some(begun), Some(end)) => pairs.push((kind, begun, end)),
            _ => *lost += 1,
        }
    }

    /// Resolves what was stamped since the last resolve into the read-back
    /// memory, recorded in `list` behind the stamps.
    pub(crate) fn resolve(&mut self, list: &ID3D12GraphicsCommandList) {
        while self.resolved < self.used {
            let (n, first) = ((self.resolved / CHUNK) as usize, self.resolved % CHUNK);
            let count = (self.used - self.resolved).min(CHUNK - first);
            let Some(chunk) = self.chunks.get(n) else {
                break;
            };
            // SAFETY: an open list; our heap and read-back buffer, which
            // holds the chunk's timestamps.
            unsafe {
                list.ResolveQueryData(&chunk.heap, D3D12_QUERY_TYPE_TIMESTAMP, first, count, &chunk.back, u64::from(first) * 8)
            };
            self.resolved += count;
        }
    }

    /// The list is closed: its own pair ends here, and everything stamped
    /// is resolved.
    pub(crate) fn close(&mut self, list: &ID3D12GraphicsCommandList) {
        if self.owed
            && let Some(opened) = self.opened.take()
        {
            self.main = self.stamp(list).map(|end| (opened, end));
        }
        self.resolve(list);
    }

    /// The list goes to the queue (called before it does), and the card
    /// has run it once `fence` reaches `value`: until this send is read, or
    /// given up on unread after [`STALE_S`], no window closes past now as
    /// the queue's calibration puts it on the card's clock, less 1 ms (an
    /// estimate: its work can begin before; given up on, windows close past
    /// it, each saying so while it stays unread). The pairs recorded for
    /// this send alone go with it.
    pub(crate) fn sent(&mut self, fence: &ID3D12Fence, value: u64) {
        if !self.sends.is_empty() {
            self.clash = true;
        }
        let (now, _) = cpu_clock();
        let id = with_seen(|seen| seen.register(now));
        // With the passes on every send counts in its frame, ours too: a
        // frame's figures wait for all of them (its number taken under the
        // book's lock, as the lines take frames).
        let frame = passes_on().then(|| crate::passes::sent(now));
        self.sends.push_back(Sending {
            id,
            at: now,
            fence: fence.clone(),
            value,
            once: std::mem::take(&mut self.once),
            once_lost: std::mem::take(&mut self.once_lost),
            frame,
        });
    }

    /// Whether no send is unread.
    pub(crate) fn unsent(&self) -> bool {
        self.sends.is_empty()
    }

    /// The value of slot `at`, as the card resolved it.
    fn value(&self, at: u32) -> Option<u64> {
        let chunk = self.chunks.get((at / CHUNK) as usize)?;
        // SAFETY: the mapping holds the chunk's timestamps; the caller has
        // seen the send's fence reach its value (the card has run the list
        // and its resolve).
        Some(unsafe { chunk.data.add((at % CHUNK) as usize).read_volatile() })
    }

    /// The card has run the oldest send: its pairs are read and handed to
    /// the line. Nothing is read unless its fence has reached its value on a
    /// live device; the send stays unread if the fence is below it, and is
    /// given up if the device went.
    pub(crate) fn landed(&mut self) {
        let Some(send) = self.sends.front() else {
            return;
        };
        match reached(&send.fence, send.value) {
            Reached::Yes => {}
            Reached::No => {
                let at = send.at;
                with_seen(|seen| seen.flaws.push((at, at, Flaws { early: 1, ..Flaws::default() })));
                return;
            }
            Reached::Removed => {
                let send = self.sends.pop_front().expect("a send");
                if self.sends.is_empty() {
                    self.clash = false;
                }
                let now = cpu_clock().0;
                with_seen(|seen| seen.deliver(send.id, send.at, now, Vec::new(), Flaws { unverified: 1, ..Flaws::default() }));
                if let Some(frame) = send.frame {
                    crate::passes::lost(frame);
                }
                return;
            }
        }
        let send = self.sends.pop_front().expect("a send");
        let mut flaws = Flaws::default();
        let mut read = Vec::new();
        let mut parts = Vec::new();
        let mut pair = |kind: Kind, begun: u32, end: u32, flaws: &mut Flaws| -> Option<(u64, u64)> {
            match (self.value(begun), self.value(end)) {
                (Some(b), Some(e)) if e >= b => {
                    read.push((kind, b, e));
                    Some((b, e))
                }
                (Some(_), Some(_)) => {
                    flaws.reversed += 1;
                    None
                }
                _ => {
                    flaws.lost += 1;
                    None
                }
            }
        };
        // A title list's own span, read (for the passes).
        let mut title = None;
        if self.clash {
            flaws.excluded += 1;
        } else {
            let parent = match self.main {
                Some((b, e)) => pair(self.class.unwrap_or(Kind::List), b, e, &mut flaws),
                None => {
                    if self.owed {
                        flaws.unclosed += 1;
                    }
                    None
                }
            };
            for &(kind, b, e) in &self.kept {
                if let Some((b, e)) = pair(kind, b, e, &mut flaws)
                    && kind == Kind::Clamp
                    && parent.is_some_and(|(pb, pe)| b < pb || e > pe)
                {
                    flaws.outside += 1;
                }
            }
            flaws.lost += self.kept_lost;
            // A title list's parts by program (`VICTORY_GPU_STAMPS=2`),
            // only with its own pair read.
            if let Some(span) = parent
                && self.per_call
                && self.class.unwrap_or(Kind::List) == Kind::List
            {
                parts = self.parts(span, &mut flaws);
            }
            title = parent.filter(|_| self.class.unwrap_or(Kind::List) == Kind::List);
        }
        for &(kind, b, e) in &send.once {
            pair(kind, b, e, &mut flaws);
        }
        flaws.lost += send.once_lost;
        if self.sends.is_empty() {
            self.clash = false;
        }
        if let Some(frame) = send.frame {
            crate::passes::read(frame, self.pass_reading(title, &read, &flaws));
        }
        let now = cpu_clock().0;
        with_seen(|seen| {
            seen.parts_read(parts);
            seen.deliver(send.id, send.at, now, read, flaws);
        });
    }

    /// What a send read gives the passes ([`crate::passes`]): the passes of
    /// a title list whose own span `title` was read (the list was opened
    /// with the passes on), with our clamp groups and texture uploads in
    /// them left out; every read pair of ours, by kind; and whether the
    /// send had a flaw that leaves its frame out.
    fn pass_reading(&self, title: Option<(u64, u64)>, read: &[(Kind, u64, u64)], flaws: &Flaws) -> crate::passes::Reading {
        let mut reading = crate::passes::Reading { flawed: flaws.kinds(), ..Default::default() };
        for &(kind, b, e) in read {
            if kind != Kind::List {
                reading.ours[kind as usize] += e - b;
            }
        }
        if let Some(span) = title
            && self.passes
        {
            let inside: Vec<(u64, u64)> =
                read.iter().filter(|s| matches!(s.0, Kind::Clamp | Kind::Upload)).map(|&(_, b, e)| (b, e)).collect();
            let marks = self.marks.iter().map(|&(slot, name, calls)| (slot.and_then(|s| self.value(s)), name, calls));
            let cut = crate::passes::parts(span, marks, &inside);
            reading.passes = cut.passes;
            reading.unattributed = cut.none;
            reading.uncalled = cut.uncalled;
            reading.flawed |= cut.bad > 0;
        }
        reading
    }
}

enum Reached {
    Yes,
    No,
    Removed,
}

/// Whether `fence` has reached `value` on a live device.
fn reached(fence: &ID3D12Fence, value: u64) -> Reached {
    // SAFETY: a plain query of a live fence.
    let done = unsafe { fence.GetCompletedValue() };
    if done == u64::MAX && removed() {
        Reached::Removed
    } else if done >= value {
        Reached::Yes
    } else {
        Reached::No
    }
}

fn removed() -> bool {
    #[cfg(test)]
    if REMOVED.with(std::cell::Cell::get) {
        return true;
    }
    // SAFETY: a plain query.
    crate::host::host().is_some_and(|h| unsafe { h.device.GetDeviceRemovedReason() }.is_err())
}

/// A send's stamps that live elsewhere (a present copy sent as a flight of
/// its own): its oldest send has run.
pub(crate) fn landed_in(stamps: &Mutex<Stamps>) {
    stamps.lock().unwrap_or_else(|e| e.into_inner()).landed();
}

/// A closed window of the card's clock, in its ticks.
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Window {
    /// Ticks by [`Kind`], each tick once ([`FIRST`]).
    pub(crate) ticks: [u64; KINDS],
    /// Ticks no span covers.
    pub(crate) uncovered: u64,
    /// Pairs by [`Kind`] that ended in the window.
    pub(crate) pairs: [u64; KINDS],
}

/// A closed window's title-list ticks by program (`VICTORY_GPU_STAMPS=2`).
#[derive(Default, Clone, PartialEq, Eq, Debug)]
pub(crate) struct ByProgram {
    /// Ticks and calls by [`Program`] (its number is the index): each tick
    /// a window gives title lists once, the calls of the parts that ended
    /// in the window.
    pub(crate) parts: Vec<(u64, u64)>,
    /// Title-list ticks in no part: lists without call stamps, or a stretch
    /// a lost stamp left without a known program.
    pub(crate) unattributed: u64,
}

impl ByProgram {
    fn add(&mut self, program: Program, ticks: u64, calls: u64) {
        let at = program as usize;
        if self.parts.len() <= at {
            self.parts.resize(at + 1, (0, 0));
        }
        self.parts[at].0 += ticks;
        self.parts[at].1 += calls;
    }

    /// Ticks and calls of `program`.
    pub(crate) fn of(&self, program: Program) -> (u64, u64) {
        self.parts.get(program as usize).copied().unwrap_or((0, 0))
    }

    /// All ticks: the programs', [`AFTER`]'s and those of no part.
    #[cfg(test)]
    pub(crate) fn ticks(&self) -> u64 {
        self.parts.iter().map(|p| p.0).sum::<u64>() + self.unattributed
    }

    fn join(&mut self, o: &ByProgram) {
        for (n, &(ticks, calls)) in o.parts.iter().enumerate() {
            self.add(n as Program, ticks, calls);
        }
        self.unattributed += o.unattributed;
    }
}

/// [`close_all`] without parts of title lists (tests of the windows).
#[cfg(test)]
fn close(spans: &mut Vec<(Kind, u64, u64)>, from: u64, to: u64) -> Window {
    close_all(spans, None, from, to).0
}

/// The window `[from, to)` of the card's clock: each tick counted once, for
/// the first kind in [`FIRST`] with a span over it, or as uncovered; each
/// pair counted in the window its end lies in, over `(from, to]` at
/// successive cuts (an end exactly at `to` counts here, with no tick here if
/// the pair began there). What reaches past `to` stays in `spans` for the
/// next window.
///
/// With `VICTORY_GPU_STAMPS=2`, `parts` of title lists: a tick the window
/// gives title lists goes to the part over it (the one begun last where two
/// are), or to none; a part's calls count where its end lies, as a pair's,
/// and what reaches past `to` stays in `parts`. Without them (level 1) no
/// work is done by program, and the second result is empty.
fn close_all(spans: &mut Vec<(Kind, u64, u64)>, parts: Option<&mut Vec<Part>>, from: u64, to: u64) -> (Window, ByProgram) {
    let mut window = Window::default();
    let mut by = ByProgram::default();
    let by_program = parts.is_some();
    let mut no_parts = Vec::new();
    let parts = parts.unwrap_or(&mut no_parts);
    // Edges at a moment: where they end before where they begin; a kind's
    // number below [`KINDS`], a part's [`KINDS`] and up.
    let mut edges: Vec<(u64, i8, usize)> = Vec::with_capacity(2 * (spans.len() + parts.len()));
    let mut left = Vec::new();
    for &(kind, begun, end) in spans.iter() {
        if end > to {
            left.push((kind, begun.max(to), end));
        } else {
            window.pairs[kind as usize] += 1;
        }
        let (a, b) = (begun.max(from), end.min(to));
        if a < b {
            edges.push((a, 1, kind as usize));
            edges.push((b, -1, kind as usize));
        }
    }
    let mut parts_left = Vec::new();
    for (i, &(program, calls, begun, end)) in parts.iter().enumerate() {
        if end > to {
            // Its own beginning kept: which of two overlapping parts has a
            // tick is the same however windows cut them (Sol's third read
            // of the frame timers: cut here, two parts began together, and
            // the order they were read in chose).
            parts_left.push((program, calls, begun, end));
        } else {
            by.add(program, 0, u64::from(calls));
        }
        let (a, b) = (begun.max(from), end.min(to));
        if a < b {
            edges.push((a, 1, KINDS + i));
            edges.push((b, -1, KINDS + i));
        }
    }
    edges.sort_unstable();
    let mut over = [0u32; KINDS];
    // The parts over the moment; the one begun last has it, then the one
    // that ends first (inside the other), then the lower program number.
    let mut under: Vec<usize> = Vec::new();
    let rank = |i: usize| {
        let (program, _, begun, end) = parts[i];
        (begun, std::cmp::Reverse(end), std::cmp::Reverse(program))
    };
    let mut at = from;
    for (edge, step, what) in edges {
        if edge > at {
            match FIRST.iter().find(|k| over[**k as usize] > 0) {
                Some(&k) => {
                    window.ticks[k as usize] += edge - at;
                    if by_program && k == Kind::List {
                        match under.iter().copied().max_by_key(|&i| rank(i)) {
                            Some(i) => by.add(parts[i].0, edge - at, 0),
                            None => by.unattributed += edge - at,
                        }
                    }
                }
                None => window.uncovered += edge - at,
            }
            at = edge;
        }
        if what < KINDS {
            over[what] = over[what].wrapping_add_signed(i32::from(step));
        } else if step > 0 {
            under.push(what - KINDS);
        } else if let Some(i) = under.iter().rposition(|&i| i == what - KINDS) {
            under.remove(i);
        }
    }
    window.uncovered += to - at;
    *spans = left;
    *parts = parts_left;
    (window, by)
}

/// How long a send may stay unread before windows close past it anyway.
const STALE_S: u64 = 10;

/// What the stamps read and noted, not yet in a closed window.
#[derive(Default)]
struct Seen {
    /// The next send's number.
    next: u64,
    /// Sends not read yet, by number: when they went (CPU's counter).
    out: BTreeMap<u64, u64>,
    /// Sends given up on unread after [`STALE_S`], by number: when they
    /// went. Windows close past them, and each window that ends after one
    /// went is incomplete while it is unread ([`Flaws::stale`]).
    given: BTreeMap<u64, u64>,
    /// Spans read, not yet wholly in a closed window (kind, begin, end).
    spans: Vec<(Kind, u64, u64)>,
    /// Flaws, each with the stretch in which the work it concerns may have
    /// run (CPU's counter): from the moment its send went to the moment it
    /// was read, its fence passed. Every window over that stretch is
    /// incomplete, not only the one the send went in.
    flaws: Vec<(u64, u64, Flaws)>,
    /// Where the closed windows end, on the card's clock.
    cut: Option<u64>,
    /// Since the last window: spans read after the window they began in
    /// had closed, and their ticks before its end (in no window).
    late: (u64, u64),
    /// The calibration at the last line (the card's clock, the CPU's).
    last: Option<(u64, u64)>,
    /// Parts of title lists read (`VICTORY_GPU_STAMPS=2`), not yet wholly
    /// in a closed window.
    parts: Vec<Part>,
    /// The windows closed since the last lines by program.
    tally: Tally,
    /// The windows the last lines by program counted (their beginnings and
    /// ends on the card's clock), until work read later is found in one.
    printed: Vec<(u64, u64)>,
}

static SEEN: Mutex<Seen> = Mutex::new(Seen {
    next: 0,
    out: BTreeMap::new(),
    given: BTreeMap::new(),
    spans: Vec::new(),
    flaws: Vec::new(),
    cut: None,
    late: (0, 0),
    last: None,
    parts: Vec::new(),
    tally: Tally::EMPTY,
    printed: Vec::new(),
});

/// How long the lines by program take windows together, in seconds.
const TALLY_S: u64 = 5;

/// Closed windows taken together for the lines by program
/// (`VICTORY_GPU_STAMPS=2`).
#[derive(Default, Clone, PartialEq, Eq, Debug)]
pub(crate) struct Tally {
    /// When the first of them closed (CPU's counter).
    since: Option<u64>,
    /// Those whose samples establish their figures, which the rest are of
    /// (with the four after it, the sums of [`Tally::kept`], made when the
    /// tally is due).
    pub(crate) windows: u64,
    /// Those left out at their close: with a flaw, their lines say which.
    pub(crate) incomplete: u64,
    /// Those counted at their close and left out since: a span or part read
    /// after it lies in them ([`Seen::read_late`]).
    pub(crate) late: u64,
    /// Windows the last lines by program counted that a span or part read
    /// since lies in: those lines' figures miss it.
    pub(crate) earlier: u64,
    /// Their ticks of the card's clock.
    pub(crate) wall: u64,
    /// The frames that ended in them (present copies).
    pub(crate) frames: u64,
    /// The ticks they gave title lists, and those by program.
    pub(crate) list: u64,
    pub(crate) by: ByProgram,
    /// The windows counted, each with its figures, until the tally is due.
    kept: Vec<Kept>,
}

/// A window a tally counted at its close: where it lies on the card's
/// clock, and its figures.
#[derive(Default, Clone, PartialEq, Eq, Debug)]
struct Kept {
    from: u64,
    to: u64,
    frames: u64,
    list: u64,
    by: ByProgram,
}

impl Tally {
    const EMPTY: Tally = Tally {
        since: None,
        windows: 0,
        incomplete: 0,
        late: 0,
        earlier: 0,
        wall: 0,
        frames: 0,
        list: 0,
        by: ByProgram { parts: Vec::new(), unattributed: 0 },
        kept: Vec::new(),
    };

    /// A closed window joins the tally, at `now` (CPU's counter): its
    /// figures if its samples establish them, else only its count as
    /// incomplete (Sol's third read of the frame timers: an incomplete
    /// window's figures were summed with the rest).
    fn add(&mut self, closed: &Closed, now: u64) {
        self.since.get_or_insert(now);
        if closed.flaws.any() {
            self.incomplete += 1;
            return;
        }
        self.kept.push(Kept {
            from: closed.from,
            to: closed.to,
            frames: closed.window.pairs[Kind::Present as usize],
            list: closed.window.ticks[Kind::List as usize],
            by: closed.programs.clone(),
        });
    }

    /// The tally, once its first window closed `every` or more before `now`
    /// (CPU's counter), with the sums of the windows it kept; a new one
    /// begins.
    fn due(&mut self, now: u64, every: u64) -> Option<Tally> {
        if !self.since.is_some_and(|since| now.saturating_sub(since) >= every) {
            return None;
        }
        let mut tally = std::mem::take(self);
        for k in &tally.kept {
            tally.windows += 1;
            tally.wall += k.to - k.from;
            tally.frames += k.frames;
            tally.list += k.list;
            tally.by.join(&k.by);
        }
        Some(tally)
    }
}

/// `f` on what the stamps saw (tests: this thread's sink, if it has one).
fn with_seen<R>(f: impl FnOnce(&mut Seen) -> R) -> R {
    #[cfg(test)]
    {
        let mut f = Some(f);
        let mine = SINK.with(|sink| sink.borrow_mut().as_mut().map(|seen| (f.take().expect("once"))(seen)));
        if let Some(r) = mine {
            return r;
        }
        (f.take().expect("once"))(&mut SEEN.lock().unwrap_or_else(|e| e.into_inner()))
    }
    #[cfg(not(test))]
    f(&mut SEEN.lock().unwrap_or_else(|e| e.into_inner()))
}

/// A closed window and what came with it.
struct Closed {
    from: u64,
    to: u64,
    window: Window,
    flaws: Flaws,
    late: (u64, u64),
    programs: ByProgram,
}

impl Seen {
    /// A send went at `at` (CPU's counter): its number.
    fn register(&mut self, at: u64) -> u64 {
        self.next += 1;
        self.out.insert(self.next, at);
        self.next
    }

    /// Send `id`, which went at `at`, was read at `until` (CPU's counter;
    /// its fence had passed, or it is let go): its spans and flaws.
    fn deliver(&mut self, id: u64, at: u64, until: u64, read: Vec<(Kind, u64, u64)>, flaws: Flaws) {
        self.out.remove(&id);
        self.given.remove(&id);
        for (kind, begun, end) in read {
            match self.cut {
                // Begun before the windows' end, or ended at it (a pair
                // ending exactly at a window's end counts in that window,
                // [`close_all`]): the window's end is an estimate, and its
                // part there is in no window, and said; its pair too if it
                // ended there (Sol's fourth read of the frame timers: a
                // span of no length at the end counted in the next window).
                Some(cut) if begun < cut || end <= cut => {
                    self.late.0 += 1;
                    self.late.1 += end.min(cut) - begun;
                    self.read_late(begun, end, cut, true);
                    if end > cut {
                        self.spans.push((kind, cut, end));
                    }
                }
                _ => self.spans.push((kind, begun, end)),
            }
        }
        if flaws.any() {
            self.flaws.push((at, until.max(at), flaws));
        }
    }

    /// A span or part `[begun, end)` read after the windows closed at
    /// `cut`, with its end counted where it lies if `ends` (a pair, or a
    /// part with a call): a closed window its ticks before the cut, or that
    /// end, lie in misses them. Such a window leaves the tally (it was
    /// counted only as checked at its close), or, if the last lines by
    /// program counted it, is said in the next (Sol's fourth read of the
    /// frame timers: a window the next line showed short of a span stayed in
    /// the tally, said to have no loss detected).
    fn read_late(&mut self, begun: u64, end: u64, cut: u64, ends: bool) {
        let before = end.min(cut);
        let lies_in = |from: u64, to: u64| (begun < before && begun < to && from < before) || (ends && end <= cut && from < end && end <= to);
        let kept = self.tally.kept.len();
        self.tally.kept.retain(|k| !lies_in(k.from, k.to));
        self.tally.late += (kept - self.tally.kept.len()) as u64;
        let printed = self.printed.len();
        self.printed.retain(|&(from, to)| !lies_in(from, to));
        self.tally.earlier += (printed - self.printed.len()) as u64;
    }

    /// The window closed by [`Seen::window`] joins the tally
    /// (`VICTORY_GPU_STAMPS=2`), under the same lock as it closed (a span
    /// read between the two would miss it, [`Seen::read_late`]); the tally
    /// if it is due at `now` (CPU's counter, `every` its ticks), and the
    /// windows it counted are kept as the last printed.
    fn tally_closed(&mut self, closed: &Closed, now: u64, every: u64) -> Option<Tally> {
        self.tally.add(closed, now);
        let tally = self.tally.due(now, every)?;
        self.printed = tally.kept.iter().map(|k| (k.from, k.to)).collect();
        Some(tally)
    }

    /// Parts of a title list read with it (`VICTORY_GPU_STAMPS=2`): what
    /// lies before the windows' end is lost, as its list's is (and said),
    /// and so is the call of a part that ends at it (its window has
    /// closed: Sol's fourth read of the frame timers, two calls marked at
    /// the end, the second counted in the next window); a part across it
    /// keeps its own beginning, which orders it among parts that overlap it
    /// ([`close_all`]).
    fn parts_read(&mut self, parts: Vec<Part>) {
        for (program, calls, begun, end) in parts {
            match self.cut {
                Some(cut) if begun < cut || end <= cut => {
                    self.read_late(begun, end, cut, calls > 0);
                    if end > cut {
                        self.parts.push((program, calls, begun, end));
                    }
                }
                _ => self.parts.push((program, calls, begun, end)),
            }
        }
    }

    /// Sends unread since before `before` (CPU's counter) are given up on:
    /// windows close past them, incomplete while they are unread
    /// ([`Flaws::stale`]).
    fn stale(&mut self, before: u64) {
        let old: Vec<(u64, u64)> = self.out.iter().filter(|(_, at)| **at < before).map(|(id, at)| (*id, *at)).collect();
        for (id, at) in old {
            self.out.remove(&id);
            self.given.insert(id, at);
        }
    }

    /// The line when no window closed: the sends not read yet at `now`
    /// (CPU's counter, `cpu_frequency` ticks a second).
    fn waiting_text(&self, now: u64, cpu_frequency: u64) -> String {
        let ms = |at: u64| now.saturating_sub(at) as f64 * 1e3 / cpu_frequency.max(1) as f64;
        // Both groups: a send given up on is still unread (Sol's third read
        // of the frame timers: the line said 0 with one given up on).
        let oldest = self.out.values().chain(self.given.values()).min();
        format!(
            "card queue by GPU timestamps (VICTORY_GPU_STAMPS=1): no window closed since the last line: {} sends not read yet \
             ({} of them given up on after {STALE_S} s), the oldest sent {:.0} ms ago",
            self.out.len() + self.given.len(),
            self.given.len(),
            oldest.map_or(0.0, |&at| ms(at))
        )
    }

    /// Closes the next window, up to `now` on the card's clock and to
    /// `margin` ticks before the moment the oldest unread send went
    /// (`gpu_of` puts the CPU's counter on the card's clock); `None` if that
    /// is not past the last window's end.
    fn window(&mut self, now: u64, margin: u64, gpu_of: &dyn Fn(u64) -> u64) -> Option<Closed> {
        let held = self.out.values().map(|&at| gpu_of(at)).min();
        let to = held.map_or(now, |held| held.min(now)).saturating_sub(margin);
        let from = self.cut.or_else(|| self.spans.iter().map(|s| s.1).min()).unwrap_or(to);
        if to <= from {
            return None;
        }
        // By program only at level 2 (Sol's third read of the frame timers:
        // level 1 kept the title lists' ticks in no part).
        let by_program = calls_on() || !self.parts.is_empty();
        let (window, programs) = close_all(&mut self.spans, by_program.then_some(&mut self.parts), from, to);
        self.cut = Some(to);
        // A send given up on that went before the window's end may have
        // run in it; a flaw counts in every window its stretch reaches, and
        // stays for the next while it reaches the end.
        let mut flaws = Flaws { stale: self.given.values().filter(|&&at| gpu_of(at) < to).count() as u64, ..Flaws::default() };
        self.flaws.retain(|&(at, until, f)| {
            if gpu_of(at) < to {
                flaws.add(f);
                gpu_of(until) >= to
            } else {
                true
            }
        });
        Some(Closed { from, to, window, flaws, late: std::mem::take(&mut self.late), programs })
    }
}

/// The queue's clock: ticks a second, and a moment on it with the same
/// moment on the CPU's performance counter.
#[derive(Clone, Copy)]
pub(crate) struct Calibration {
    pub(crate) frequency: u64,
    pub(crate) gpu: u64,
    pub(crate) cpu: u64,
}

/// The line's text for a closed window: `frequency` the card's ticks a
/// second, `moved` how far the card's and the CPU's clocks went since the
/// last line (ms), if there was one. A window with a flaw found by its
/// close that its figures cannot be established without says it is
/// incomplete, and why, and gives no frames and no parts (a flaw found
/// later does not reach a line already printed); one without says that no
/// loss was detected
/// within estimated bounds at its close, never that it is complete: its end
/// and each send's moment are CPU moments put on the card's clock by a
/// calibration, and work may lie on the other side of them (Sol's third
/// read of the frame timers). What it did not read it says as such, not as
/// what did not happen, and a ratio over the pairs that ended in it is
/// unavailable when none did (Sol's fourth read: 0, with time in the
/// window).
fn text(closed: &Closed, frequency: u64, moved: Option<(f64, f64)>) -> String {
    text_with(closed, frequency, moved, passes_on())
}

/// [`text`], with our texture uploads in title lists given (`uploads`:
/// they are stamped, `VICTORY_GPU_PASSES=1`); without them the line is as
/// it was.
fn text_with(closed: &Closed, frequency: u64, moved: Option<(f64, f64)>, uploads: bool) -> String {
    use std::fmt::Write;
    let Closed { from, to, window: w, flaws, late, .. } = closed;
    let ms = |ticks: u64| ticks as f64 * 1e3 / frequency as f64;
    let frames = w.pairs[Kind::Present as usize];
    let per = frames.max(1) as f64;
    let wall = ms(to - from);
    let part = |k: Kind| ms(w.ticks[k as usize]) / per;
    let n = |k: Kind| w.pairs[k as usize] as f64 / per;
    // The window's ticks of a kind over its pairs that ended in it (not a
    // mean of spans: a span's time and its end can lie in two windows).
    let each = |k: Kind, places: usize| match w.pairs[k as usize] {
        0 => "unavailable".to_owned(),
        ended => format!("{:.places$}", ms(w.ticks[k as usize]) / ended as f64),
    };
    let mut t = format!("card queue by GPU timestamps (VICTORY_GPU_STAMPS=1): a window of {wall:.1} ms on the card's clock");
    if let Some((card, cpu)) = moved {
        let _ = write!(t, " (since the last line the card's clock moved {card:.1} ms, the CPU's {cpu:.1} ms)");
    }
    let said = [
        (flaws.lost, "pairs lost (no query heap)"),
        (flaws.unclosed, "lists without their own pair"),
        (flaws.excluded, "sends left out (a list sent again before its earlier send was read)"),
        (flaws.unverified, "sends unread (the device went before their fence)"),
        (flaws.early, "landings asked for before their fence (not read then)"),
        (flaws.dropped, "sends forgotten unread"),
        (flaws.stale, "sends given up on unread after 10 s that went before the window's end"),
        (flaws.reversed, "pairs that ended before they began"),
        (flaws.outside, "clamp groups outside their list"),
    ];
    let said: Vec<String> = said.iter().filter(|(n, _)| *n > 0).map(|(n, what)| format!("{n} {what}")).collect();
    if flaws.kinds() {
        let _ = write!(
            t,
            "; the window is incomplete: {}; its frames and parts are not given (its samples cannot establish them)",
            said.join(", ")
        );
    } else {
        if frames > 0 {
            let _ = write!(
                t,
                ", {frames} present copies read with their end in it (frames are counted by them), {:.2} ms a frame; elapsed on the \
                 queue a frame",
                wall / per
            );
        } else {
            t.push_str(", no present copy read with its end in it (frames are counted by them); elapsed on the queue in the whole window");
        }
        let _ = write!(
            t,
            ", each tick counted once: title lists {:.2} ms ({:.1} lists ended, {} ms a list (the window's title-list time over \
             the lists that ended in it), without the clamps in them), our clamp groups {:.2} ms ({:.0} ended, {} ms a group \
             (the window's clamp time over the groups that ended in it), their barriers included), preludes {:.2} ms ({:.1}), \
             relay fetches {:.2} ms ({:.1} ended, {} ms a fetch (the window's relay time over the fetches that ended in it)), \
             present copies {:.2} ms, hash lists {:.2} ms ({:.1} lists), compare lists {:.2} ms ({:.1} lists), synchronous \
             fetches {:.2} ms ({:.1}), uncovered queue time {:.2} ms (no stamped span read by the window's close covers it: the \
             queue idle, work not stamped, our resolves of these stamps among it, or work whose stamps were read after)",
            part(Kind::List),
            n(Kind::List),
            each(Kind::List, 3),
            part(Kind::Clamp),
            n(Kind::Clamp),
            each(Kind::Clamp, 4),
            part(Kind::Prelude),
            n(Kind::Prelude),
            part(Kind::Relay),
            n(Kind::Relay),
            each(Kind::Relay, 3),
            part(Kind::Present),
            part(Kind::Hash),
            n(Kind::Hash),
            part(Kind::Compare),
            n(Kind::Compare),
            part(Kind::Fetch),
            n(Kind::Fetch),
            ms(w.uncovered) / per,
        );
        if uploads {
            let _ = write!(
                t,
                ", and out of the title lists' time: our texture uploads in them {:.2} ms ({:.1} ended, their barriers included; \
                 VICTORY_GPU_PASSES=1)",
                part(Kind::Upload),
                n(Kind::Upload)
            );
        }
        t.push_str(
            "; no losses detected within estimated bounds at its close (the window's ends and each send's moment are the \
             CPU's, put on the card's clock by the queue's calibration)",
        );
    }
    let by_program = [
        (flaws.calls_lost, "stamps after a title call lost (no query heap)"),
        (flaws.calls_bad, "stamps after a title call before the one before them or outside their list"),
    ];
    let by_program: Vec<String> = by_program.iter().filter(|(n, _)| *n > 0).map(|(n, what)| format!("{n} {what}")).collect();
    if !by_program.is_empty() {
        let _ = write!(
            t,
            "; its parts by program are incomplete: {}: their calls not counted, the time up to the next usable stamp in \
             no program",
            by_program.join(", ")
        );
    }
    if late.0 > 0 {
        let _ = write!(
            t,
            "; {} spans read after an earlier window had closed: their parts before this window, {:.2} ms summed (overlaps \
             counted twice), are in no window (an earlier line that gave figures for those ticks misses them)",
            late.0,
            ms(late.1)
        );
    }
    if crate::host::timing() {
        t.push_str("; VICTORY_GPU_TIMING is on too: its resolves lie inside title lists");
    }
    t
}

/// The moment `at` of the CPU's counter (`cpu_frequency` ticks a second) on
/// the card's clock, by `calibration`. The two are other clocks: on a real
/// card the queue's ticks are not the CPU's counter (the software adapter's
/// may be).
fn on_card(calibration: &Calibration, cpu_frequency: u64, at: u64) -> u64 {
    let Calibration { frequency, gpu, cpu } = *calibration;
    let ticks = (i128::from(at) - i128::from(cpu)) * i128::from(frequency) / i128::from(cpu_frequency.max(1));
    (i128::from(gpu) + ticks).max(0) as u64
}

/// Tests: the CPU's counter put on the card's clock as the line puts it
/// ([`on_card`]), and the card's clock now.
#[cfg(test)]
pub(crate) fn card_clock() -> (impl Fn(u64) -> u64, u64) {
    let host = crate::host::host().expect("a host device");
    let calibration = host.clock_calibration().expect("the queue's clock calibration");
    let (now, cpu_frequency) = cpu_clock();
    let now = on_card(&calibration, cpu_frequency, now);
    (move |at| on_card(&calibration, cpu_frequency, at), now)
}

/// The lines by program for `tally` (`frequency` the card's ticks a
/// second; `name` a program's stage and shader ids): one a program for the
/// [`TOP`] with the most elapsed time, then one with what they and the rest
/// of the title lists' time are. Plain numbers in a fixed order. With no
/// window counted, one line that says so, and no figures (Sol's fourth read
/// of the frame timers: zero-valued ms a frame).
fn programs_text(tally: &Tally, frequency: u64, name: &dyn Fn(Program) -> Option<(ProgramStage, String)>) -> Vec<String> {
    use std::fmt::Write;
    const HEAD: &str = "card time by program (VICTORY_GPU_STAMPS=2)";
    // Work read since the last lines that lies in a window they counted.
    let earlier = |line: &mut String| {
        if tally.earlier > 0 {
            let _ = write!(
                line,
                "; {} windows the lines before counted have work read since that lies in them (those lines' figures miss it)",
                tally.earlier
            );
        }
    };
    if tally.windows == 0 {
        let mut line = format!(
            "{HEAD}: no usable windows: {} windows left out, incomplete at their close (their lines say why), and {} left out \
             after their close (work read later lies in them); no figures by program",
            tally.incomplete, tally.late
        );
        earlier(&mut line);
        return vec![line];
    }
    let per = tally.frames.max(1) as f64;
    let ms = |ticks: u64| ticks as f64 * 1e3 / frequency.max(1) as f64 / per;
    let share = |ticks: u64| if tally.list > 0 { ticks as f64 * 100.0 / tally.list as f64 } else { 0.0 };
    let mut ran: Vec<(Program, u64, u64)> = tally
        .by
        .parts
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, (ticks, calls))| *ticks > 0 || *calls > 0)
        .map(|(n, &(ticks, calls))| (n as Program, ticks, calls))
        .collect();
    ran.sort_by_key(|&(n, ticks, _)| (std::cmp::Reverse(ticks), n));
    let mut lines = Vec::with_capacity(TOP + 1);
    for (rank, &(n, ticks, calls)) in ran.iter().take(TOP).enumerate() {
        let (stage, shaders) = name(n).map_or(("unknown", "[unknown]".to_owned()), |(s, shaders)| (s.word(), shaders));
        // Window time over the ends counted in the windows: not a mean of
        // calls' intervals (a call's time and its end can fall in two
        // windows). None without an end.
        let each = if calls > 0 {
            format!("{:.4}", ticks as f64 * 1e3 / frequency.max(1) as f64 / calls as f64)
        } else {
            "unavailable".to_owned()
        };
        lines.push(format!(
            "{HEAD}: rank {}: stage {stage}, {:.3} ms a frame, {:.1} calls a frame, {each} ms a call (the windows' time over \
             the calls that ended in them), {:.2}% of title lists, shaders={shaders}",
            rank + 1,
            ms(ticks),
            calls as f64 / per,
            share(ticks),
        ));
    }
    let top: u64 = ran.iter().take(TOP).map(|r| r.1).sum();
    let all: u64 = ran.iter().map(|r| r.1).sum();
    let calls: u64 = ran.iter().map(|r| r.2).sum();
    let mut last = format!(
        "{HEAD}: over {} windows of {:.1} ms on the card's clock, {} frames ended in them{}; title lists {:.3} ms a frame; the \
         top {} programs {:.3} ms a frame ({:.2}%) of the {} with time attributed or a call counted; all programs {:.3} ms a \
         frame, {:.1} calls a frame; after the last call stamp of a list, or in a list with none, {:.3} ms a frame; in no \
         part {:.3} ms a frame",
        tally.windows,
        tally.wall as f64 * 1e3 / frequency.max(1) as f64,
        tally.frames,
        if tally.frames == 0 { " (none: a frame here is the whole time)" } else { "" },
        ms(tally.list),
        ran.len().min(TOP),
        ms(top),
        share(top),
        ran.len(),
        ms(all),
        calls as f64 / per,
        ms(tally.by.of(AFTER).0),
        ms(tally.by.unattributed),
    );
    // What the figure is (Sol's fourth read of the frame timers: "from the
    // stamp before each of its calls to the stamp after it" left out the
    // clipping, the clamp groups' priority and the overlap rule, and
    // "overlap with the call before included" hid where it begins).
    last.push_str(
        "; a program's time is the title-list queue time attributed to its calls between completion marks in the windows \
         counted: each call has the stretch from the stamp before it (after the call before, or at the list's opening) to \
         the stamp after it, clipped to the windows, less our clamp groups' time in it, and a tick where two lists' parts \
         overlap goes to one of them (the one begun last); so it holds the stalls, barriers and other work between the \
         marks, and is not the program's own time on the card; summed over host pipelines with the same shader ids and \
         stage; a call is one title call (an indirect execution once, whatever its records), counted in the window its \
         stamp lies in",
    );
    if tally.incomplete > 0 {
        let _ = write!(last, "; {} more windows left out, incomplete at their close (their lines say why)", tally.incomplete);
    }
    if tally.late > 0 {
        let _ = write!(last, "; {} more windows left out after their close: work read later lies in them", tally.late);
    }
    // What was checked, and when: a window counts if no flaw was found at
    // its close, and no work read after it, up to now, lies in it; work
    // read after this line can still (Sol's fourth read of the frame
    // timers: the check at close was given as the whole of it).
    last.push_str(
        "; in the windows counted no loss was detected within estimated bounds, checked at each one's close and against what \
         was read after it up to this line",
    );
    earlier(&mut last);
    lines.push(last);
    lines
}

/// Programs the lines by program name.
const TOP: usize = 30;

/// For the frame line, once a second: the next window of the card's clock
/// that can close, or why none did.
pub(crate) fn line() {
    let Some(host) = crate::host::host() else {
        return;
    };
    let (now, cpu_frequency) = cpu_clock();
    let calibration = match host.clock_calibration() {
        Ok(c) if c.frequency > 0 && cpu_frequency > 0 => c,
        Ok(c) => {
            tracing::info!(
                "card queue by GPU timestamps (VICTORY_GPU_STAMPS=1): no window: the queue's clock runs at {} ticks a second \
                 and the CPU's at {cpu_frequency}",
                c.frequency
            );
            return;
        }
        Err(e) => {
            tracing::info!("card queue by GPU timestamps (VICTORY_GPU_STAMPS=1): no window: the queue's clock gave no calibration ({e})");
            return;
        }
    };
    let gpu_of = |at: u64| on_card(&calibration, cpu_frequency, at);
    let Calibration { frequency, gpu, cpu } = calibration;
    let ms_cpu = |ticks: u64| ticks as f64 * 1e3 / cpu_frequency as f64;
    let (closed, moved, waiting, due) = with_seen(|seen| {
        seen.stale(now.saturating_sub(STALE_S * cpu_frequency));
        let moved = seen.last.replace((gpu, cpu)).map(|(g, c)| {
            (gpu.saturating_sub(g) as f64 * 1e3 / frequency as f64, ms_cpu(cpu.saturating_sub(c)))
        });
        let waiting = seen.waiting_text(now, cpu_frequency);
        let closed = seen.window(gpu_of(now), frequency / 1000, &gpu_of);
        // With `VICTORY_GPU_STAMPS=2` the window joins the tally before
        // the lock goes, and every five seconds the tally is due.
        let due = match &closed {
            Some(closed) if calls_on() => seen.tally_closed(closed, now, TALLY_S * cpu_frequency),
            _ => None,
        };
        (closed, moved, waiting, due)
    });
    match &closed {
        Some(closed) => tracing::info!("{}", text(closed, frequency, moved)),
        None => tracing::info!("{waiting}"),
    }
    // The windows since by program, after the window's own line.
    if let Some(tally) = due {
        for line in programs_text(&tally, frequency, &program_name) {
            tracing::info!("{line}");
        }
    }
    // Every five seconds, the frames read since by pass.
    if passes_on() {
        crate::passes::line(now, cpu_frequency, frequency);
    }
}

/// The stage and shader ids of program `n`, for the lines by pass.
pub(crate) fn program_text(n: Program) -> Option<(&'static str, String)> {
    program_name(n).map(|(stage, shaders)| (stage.word(), shaders))
}

/// Tests: this thread's stamps on and its own sink for what they read, both
/// until this goes.
#[cfg(test)]
pub(crate) struct Sink;

#[cfg(test)]
impl Sink {
    pub(crate) fn new() -> Sink {
        FORCED.with(|f| f.set(Some(1)));
        SINK.with(|s| *s.borrow_mut() = Some(Seen::default()));
        Sink
    }

    /// The spans read so far (kind, begin, end), in the order read.
    pub(crate) fn spans(&self) -> Vec<(Kind, u64, u64)> {
        with_seen(|seen| seen.spans.clone())
    }

    /// The flaws noted so far, together.
    pub(crate) fn flaws(&self) -> Flaws {
        with_seen(|seen| {
            let mut all = Flaws::default();
            for (_, _, f) in &seen.flaws {
                all.add(*f);
            }
            all
        })
    }

    /// Sends not read yet.
    pub(crate) fn unread(&self) -> usize {
        with_seen(|seen| seen.out.len())
    }

    /// Closes a window up to `now` on the card's clock, the CPU's counter
    /// taken as the card's clock with `gpu_of`: its parts and flaws.
    pub(crate) fn window(&self, now: u64, gpu_of: &dyn Fn(u64) -> u64) -> Option<(u64, u64, Window, Flaws)> {
        with_seen(|seen| seen.window(now, 0, gpu_of)).map(|c| (c.from, c.to, c.window, c.flaws))
    }

    /// The stamps at level 2 on this thread (`VICTORY_GPU_STAMPS=2`), and
    /// its own sink.
    pub(crate) fn calls() -> Sink {
        let sink = Sink::new();
        FORCED.with(|f| f.set(Some(2)));
        sink
    }

    /// [`Sink::window`] with the title lists' ticks by program, and the
    /// window's line.
    pub(crate) fn window_by(&self, now: u64, gpu_of: &dyn Fn(u64) -> u64, frequency: u64) -> Option<(Window, Flaws, ByProgram, String)> {
        with_seen(|seen| seen.window(now, 0, gpu_of)).map(|c| {
            let line = text(&c, frequency, None);
            (c.window, c.flaws, c.programs, line)
        })
    }

    /// The parts of title lists read so far (program, calls, begin, end).
    pub(crate) fn parts(&self) -> Vec<Part> {
        with_seen(|seen| seen.parts.clone())
    }
}

#[cfg(test)]
impl Drop for Sink {
    fn drop(&mut self) {
        FORCED.with(|f| f.set(None));
        FORCED_PASSES.with(|f| f.set(None));
        SINK.with(|s| *s.borrow_mut() = None);
        NO_CHUNKS.with(|f| f.set(false));
        REMOVED.with(|f| f.set(false));
    }
}

/// Tests: the passes on (or off) for this thread while the sink lives
/// (`VICTORY_GPU_PASSES=1`).
#[cfg(test)]
pub(crate) fn passes_for_test(on: bool) {
    FORCED_PASSES.with(|f| f.set(Some(on)));
}

/// Tests: no query heap can be had on this thread (`no`), or again.
#[cfg(test)]
pub(crate) fn no_chunks(no: bool) {
    NO_CHUNKS.with(|f| f.set(no));
}

/// Tests: the device counts as removed for this thread's landings.
#[cfg(test)]
pub(crate) fn removed_for_test(gone: bool) {
    REMOVED.with(|f| f.set(gone));
}

/// Tests: chunks taken on this thread so far.
#[cfg(test)]
pub(crate) fn taken() -> u64 {
    TAKEN.with(std::cell::Cell::get)
}

/// Tests: the times this thread looked at the stamps' level so far.
#[cfg(test)]
pub(crate) fn level_asks() -> u64 {
    LEVEL_ASKS.with(std::cell::Cell::get)
}

/// Tests: the pool of spare chunks locked on this thread so far.
#[cfg(test)]
pub(crate) fn spare_locks() -> u64 {
    SPARE_LOCKS.with(std::cell::Cell::get)
}

/// Tests: the stamps off for this thread while this lives, whatever the
/// environment says.
#[cfg(test)]
pub(crate) struct Off;

#[cfg(test)]
impl Off {
    pub(crate) fn new() -> Off {
        FORCED.with(|f| f.set(Some(0)));
        Off
    }
}

#[cfg(test)]
impl Drop for Off {
    fn drop(&mut self) {
        FORCED.with(|f| f.set(None));
    }
}

#[cfg(test)]
mod tests {
    use super::{Closed, Flaws, Kind, Seen, Window, close, switch, text, text_with};

    fn sum(w: &Window) -> u64 {
        w.ticks.iter().sum::<u64>() + w.uncovered
    }

    /// `1` turns the stamps on, `2` them and the calls' stamps only with
    /// `VICTORY_GPU_STAMPS_UNVERIFIED=1` (else level 1, and a line says
    /// level 2 is deferred), nothing else anything (Sol's fifth read of the
    /// frame timers: level 2 deferred until its comparison and reporting
    /// fixes are done).
    #[test]
    fn the_switch_is_one_or_two_and_two_only_if_asked_for_unverified() {
        let os = |v: Option<&'static str>| v.map(std::ffi::OsStr::new);
        let values = [(None, 0), (Some("0"), 0), (Some("3"), 0), (Some("1 "), 0), (Some(""), 0), (Some("1"), 1)];
        for (value, level) in values {
            for unverified in [None, Some("1")] {
                assert_eq!(switch(os(value), os(unverified)), (level, None), "{value:?} {unverified:?}");
            }
        }
        // Level 2 deferred: to 1, and the line that says so.
        for unverified in [None, Some("0"), Some(""), Some("yes")] {
            assert_eq!(switch(os(Some("2")), os(unverified)), (1, Some(super::DEFERRED)), "{unverified:?}");
        }
        assert_eq!(switch(os(Some("2")), os(Some("1"))), (2, None), "asked for unverified");
        assert!(super::DEFERRED.starts_with("GPU timestamps: VICTORY_GPU_STAMPS=2 is deferred and runs as 1: "), "{}", super::DEFERRED);
    }

    /// Each tick of a window counts once: a clamp group inside its list
    /// counts as the group's, overlapping lists as one, the rest as
    /// uncovered; the parts are the window. A span crossing the end is
    /// split there and its pair counted where it ends.
    #[test]
    fn each_tick_of_a_window_counts_once() {
        let mut spans = vec![
            (Kind::List, 10, 60),
            (Kind::Clamp, 20, 30),
            (Kind::List, 50, 80),
            (Kind::Prelude, 45, 52),
            (Kind::Relay, 90, 140),
        ];
        let w = close(&mut spans, 0, 100);
        assert_eq!(sum(&w), 100);
        assert_eq!(w.ticks[Kind::Clamp as usize], 10);
        assert_eq!(w.ticks[Kind::Prelude as usize], 7);
        // 10..20, 30..45, 52..80.
        assert_eq!(w.ticks[Kind::List as usize], 10 + 15 + 28);
        assert_eq!(w.ticks[Kind::Relay as usize], 10);
        // 0..10, 80..90.
        assert_eq!(w.uncovered, 20);
        assert_eq!(w.pairs[Kind::List as usize], 2);
        assert_eq!(w.pairs[Kind::Relay as usize], 0, "the fetch ends in the next window");
        assert_eq!(spans, [(Kind::Relay, 100, 140)]);
        let w = close(&mut spans, 100, 200);
        assert_eq!((w.ticks[Kind::Relay as usize], w.uncovered, w.pairs[Kind::Relay as usize]), (40, 60, 1));
        assert_eq!(sum(&w), 100);
        assert!(spans.is_empty());
    }

    /// Two windows of busy and empty stretches, with one send's reading
    /// held back across the first window's end: the first closes only
    /// before that send went, nothing it reads later falls before the cut,
    /// and each window's parts are its length exactly.
    #[test]
    fn a_window_closes_only_behind_the_sends_not_read() {
        let mut seen = Seen::default();
        let same = |at: u64| at;
        let a = seen.register(100);
        let b = seen.register(150);
        // Busy 100..200 (a clamp group 120..140), empty 200..210.
        seen.deliver(a, 100, 205, vec![(Kind::List, 100, 200), (Kind::Clamp, 120, 140)], Flaws::default());
        let first = seen.window(400, 0, &same).expect("a window before b went");
        assert_eq!((first.from, first.to), (100, 150), "held at b's sending");
        assert_eq!(sum(&first.window), 50);
        assert_eq!(first.window.ticks[Kind::Clamp as usize], 20);
        assert_eq!(first.window.ticks[Kind::List as usize], 30);
        assert_eq!(first.window.pairs[Kind::List as usize], 0, "the list ends in the next window");
        assert!(seen.window(400, 0, &same).is_none(), "b still holds the windows");
        // b read late: busy 210..260, then nothing to 400.
        seen.deliver(b, 150, 270, vec![(Kind::List, 210, 260), (Kind::Present, 255, 262)], Flaws::default());
        let second = seen.window(400, 0, &same).expect("a window once b was read");
        assert_eq!((second.from, second.to), (150, 400));
        assert_eq!(sum(&second.window), 250);
        assert_eq!(second.window.ticks[Kind::List as usize], 50 + 45);
        assert_eq!(second.window.ticks[Kind::Present as usize], 7);
        assert_eq!(second.window.uncovered, 10 + 138);
        assert_eq!(second.window.pairs[Kind::List as usize], 2);
        assert_eq!(second.window.pairs[Kind::Present as usize], 1);
        assert_eq!(second.late, (0, 0));
    }

    /// A span read after its window closed (a send given up on as stale):
    /// its part before the cut is in no window and said; the rest counts.
    #[test]
    fn a_span_read_after_its_window_is_split_and_said() {
        let mut seen = Seen::default();
        let same = |at: u64| at;
        let a = seen.register(100);
        seen.stale(101);
        assert!(seen.out.is_empty());
        let first = seen.window(300, 0, &same);
        assert!(first.is_none(), "nothing read: the first window starts at the first span");
        seen.cut = Some(200);
        seen.deliver(a, 100, 270, vec![(Kind::List, 150, 260)], Flaws::default());
        let w = seen.window(300, 0, &same).expect("a window");
        assert_eq!(w.late, (1, 50));
        assert_eq!(w.window.ticks[Kind::List as usize], 60);
        assert_eq!(sum(&w.window), 100);
        // Read before this window closed: its spans are in it, and the
        // window is incomplete for the part before its beginning, said as
        // late. The windows closed while it was given up were the stale
        // ones (`a_send_given_up_on_leaves_every_window_after_it_incomplete`).
        assert_eq!(w.flaws.stale, 0);
    }

    /// A send given up on unread (sent at 140, its work 220..280, read at
    /// 290): every window that closes after it went says so while it is
    /// unread, the second as well as the first; read, its part before the
    /// windows' end is late and said in the next (Sol's recheck of the
    /// frame timers, Question 2: the second window said complete).
    #[test]
    fn a_send_given_up_on_leaves_every_window_after_it_incomplete() {
        let mut seen = Seen::default();
        let same = |at: u64| at;
        let a = seen.register(100);
        seen.deliver(a, 100, 130, vec![(Kind::List, 100, 120)], Flaws::default());
        let b = seen.register(140);
        seen.stale(141);
        let first = seen.window(200, 0, &same).expect("a window past the send given up on");
        assert_eq!((first.from, first.to, first.flaws.stale), (100, 200, 1));
        let second = seen.window(300, 0, &same).expect("the next window");
        assert_eq!((second.from, second.to), (200, 300));
        assert_eq!(second.flaws.stale, 1, "the second window said complete with the send still unread");
        seen.deliver(b, 140, 290, vec![(Kind::List, 220, 280)], Flaws::default());
        let third = seen.window(400, 0, &same).expect("the window after it was read");
        assert_eq!((third.flaws.stale, third.late), (0, (1, 60)));
    }

    /// A flaw counts in every window the work it concerns may have run in:
    /// a list sent at 100 whose closing pair was lost, read at 290, with
    /// another send (at 200) holding the first window's end there: the
    /// window after says it too (Sol's recheck of the frame timers,
    /// Question 2: the flaw was charged once, at the moment the send went).
    #[test]
    fn a_flaw_counts_in_every_window_its_work_may_have_run_in() {
        let mut seen = Seen::default();
        let same = |at: u64| at;
        let a = seen.register(100);
        let c = seen.register(200);
        seen.deliver(a, 100, 290, Vec::new(), Flaws { unclosed: 1, ..Flaws::default() });
        let first = seen.window(400, 0, &same);
        assert!(first.is_none(), "nothing read before c went: no span begins a window");
        seen.deliver(c, 200, 210, vec![(Kind::List, 205, 210)], Flaws::default());
        seen.cut = Some(100);
        let w = seen.window(250, 0, &same).expect("a window to 250");
        assert_eq!((w.from, w.to, w.flaws.unclosed), (100, 250, 1));
        let w = seen.window(350, 0, &same).expect("a window to 350");
        assert_eq!((w.from, w.to), (250, 350));
        assert_eq!(w.flaws.unclosed, 1, "the window after the first said complete with the list's work in it");
        let w = seen.window(450, 0, &same).expect("a window past the list's reading");
        assert_eq!(w.flaws, Flaws::default(), "a window after the stretch is complete again");
    }

    /// The calibration that puts the CPU's moments on the card's clock moves
    /// between two windows, and a send's work began on the card 15 ticks
    /// before the first window's end, more than the margin: the first line
    /// cannot know it, and never says that it is complete, only that no loss
    /// was detected within estimated bounds; the next line says what of the
    /// send lay before it (Sol's third read of the frame timers).
    #[test]
    fn a_window_s_end_is_an_estimate_and_its_line_says_so() {
        let mut seen = Seen::default();
        let a = seen.register(100);
        seen.deliver(a, 100, 120, vec![(Kind::List, 100, 120)], Flaws::default());
        let b = seen.register(150);
        // The first mapping: the CPU's moment as it is, a margin of 1.
        let first = seen.window(400, 1, &|at| at).expect("a window before b went");
        assert_eq!(first.to, 149);
        let line = text(&first, 1_000, None);
        assert!(!line.contains("complete"), "{line}");
        assert!(
            line.contains("; no losses detected within estimated bounds at its close (the window's ends and each send's moment"),
            "{line}"
        );
        // b's work began at 134 on the card: the mapping was 16 ticks off.
        seen.deliver(b, 150, 200, vec![(Kind::List, 134, 160)], Flaws::default());
        let second = seen.window(400, 1, &|at| at - 16).expect("the next window");
        assert_eq!(second.late, (1, 15));
        let line = text(&second, 1_000, None);
        // Said as what is missing where: an earlier line may have withheld
        // its figures, or the span may begin before the first window (Sol's
        // fifth read of the frame timers: "an earlier line's figures miss
        // them" whatever that line gave).
        assert!(
            line.contains(
                "; 1 spans read after an earlier window had closed: their parts before this window, 15.00 ms summed (overlaps \
                 counted twice), are in no window (an earlier line that gave figures for those ticks misses them)"
            ),
            "{line}"
        );
        assert!(!line.contains("an earlier line's figures miss them"), "{line}");
    }

    /// A span that ends exactly at a window's end counts there, pair and
    /// ticks; one that begins exactly there has no tick in it and counts
    /// in the next.
    #[test]
    fn a_span_at_a_window_s_end_counts_once() {
        let mut spans = vec![(Kind::List, 10, 50), (Kind::Present, 50, 60)];
        let w = close(&mut spans, 0, 50);
        assert_eq!((w.ticks[Kind::List as usize], w.pairs[Kind::List as usize]), (40, 1));
        assert_eq!((w.ticks[Kind::Present as usize], w.pairs[Kind::Present as usize]), (0, 0));
        assert_eq!(spans, [(Kind::Present, 50, 60)]);
        let w = close(&mut spans, 50, 100);
        assert_eq!((w.ticks[Kind::Present as usize], w.pairs[Kind::Present as usize]), (10, 1));
        assert!(spans.is_empty());
    }

    /// Sends read out of the order they went: the later one first. The
    /// windows close only behind the earlier one, then take both whole,
    /// nothing late.
    #[test]
    fn sends_read_out_of_order_are_both_counted_whole() {
        let mut seen = Seen::default();
        let same = |at: u64| at;
        let a = seen.register(100);
        let b = seen.register(150);
        seen.deliver(b, 150, 300, vec![(Kind::List, 160, 200)], Flaws::default());
        assert!(seen.window(400, 0, &same).is_none(), "a, unread, holds the windows; nothing read before it went");
        seen.deliver(a, 100, 310, vec![(Kind::List, 110, 150)], Flaws::default());
        let w = seen.window(400, 0, &same).expect("a window");
        assert_eq!((w.from, w.to, w.late), (110, 400, (0, 0)));
        assert_eq!((w.window.ticks[Kind::List as usize], w.window.pairs[Kind::List as usize]), (80, 2));
    }

    /// Two spans read after their window closed, overlapping before its
    /// end: their parts before it are summed, the overlap twice, as the
    /// line says.
    #[test]
    fn overlapping_late_prefixes_are_summed_as_said() {
        let mut seen = Seen::default();
        let same = |at: u64| at;
        let a = seen.register(100);
        seen.deliver(a, 100, 110, vec![(Kind::List, 100, 105)], Flaws::default());
        let b = seen.register(140);
        seen.stale(141);
        seen.window(200, 0, &same).expect("a window past the send given up on");
        seen.deliver(b, 140, 260, vec![(Kind::List, 150, 230), (Kind::Clamp, 170, 220)], Flaws::default());
        let w = seen.window(300, 0, &same).expect("the next window");
        assert_eq!(w.late, (2, 50 + 30), "the prefixes 150..200 and 170..200, summed");
        let line = text(&w, 1_000, None);
        assert!(line.contains("; 2 spans read after an earlier window had closed: their parts before this window, 80.00 ms summed (overlaps"), "{line}");
    }

    /// At startup, one send given up on unread (10 s, here at 10 ticks a
    /// second) and nothing read: no window closes, and the line counts the
    /// send as not read, given up on (Sol's third read of the frame timers:
    /// the line counted only the sends not given up on, and said 0).
    #[test]
    fn the_no_window_line_counts_sends_given_up_on() {
        let mut seen = Seen::default();
        let same = |at: u64| at;
        seen.register(100);
        seen.stale(250 - super::STALE_S * 10);
        assert!(seen.out.is_empty() && seen.given.len() == 1);
        assert!(seen.window(250, 0, &same).is_none(), "no span read: no window");
        let line = seen.waiting_text(250, 10);
        assert!(
            line.ends_with(
                "no window closed since the last line: 1 sends not read yet (1 of them given up on after 10 s), the oldest \
                 sent 15000 ms ago"
            ),
            "{line}"
        );
    }

    /// At level 1 a window does no work by program: its title-list ticks
    /// are in no part of the by-program results, which stay empty (Sol's
    /// third read of the frame timers: level 1 counted them as in no part).
    #[test]
    fn level_one_does_no_work_by_program() {
        let _level = super::Sink::new();
        assert!(!super::calls_on());
        let mut seen = Seen::default();
        let same = |at: u64| at;
        let a = seen.register(100);
        seen.deliver(a, 100, 205, vec![(Kind::List, 100, 200)], Flaws::default());
        let w = seen.window(300, 0, &same).expect("a window");
        assert_eq!(w.window.ticks[Kind::List as usize], 100);
        assert_eq!(w.programs, super::ByProgram::default(), "work by program at level 1");
    }

    /// A window that a title list, a clamp group and a relay fetch cross
    /// with time in it and no end, and no present copy: each ratio of time
    /// over the pairs that ended in it is unavailable, not 0, and the line
    /// says what it did not read, not what did not happen (Sol's fourth read
    /// of the frame timers: "0.000 ms in the window for each that ended",
    /// "no frame ended in it (no copy to the swap chain)", "no stamped list
    /// on the queue").
    #[test]
    fn a_ratio_with_no_pair_ended_is_unavailable() {
        let mut spans = vec![(Kind::List, 0, 300), (Kind::Relay, 30, 250), (Kind::Clamp, 60, 200)];
        let window = close(&mut spans, 0, 100);
        assert_eq!(window.pairs, [0; super::KINDS]);
        assert_eq!([Kind::List, Kind::Relay, Kind::Clamp].map(|k| window.ticks[k as usize]), [30, 30, 40]);
        let closed = Closed { from: 0, to: 100, window, flaws: Flaws::default(), late: (0, 0), programs: super::ByProgram::default() };
        let line = text(&closed, 1_000, None);
        assert!(!line.contains(" 0.000 ms") && !line.contains(" 0.0000 ms"), "a ratio of no pair given as 0: {line}");
        for (one, what, all) in [("list", "title-list", "lists"), ("group", "clamp", "groups"), ("fetch", "relay", "fetches")] {
            let said = format!("unavailable ms a {one} (the window's {what} time over the {all} that ended in it)");
            assert!(line.contains(&said), "{said} in {line}");
        }
        assert!(line.contains(", no present copy read with its end in it (frames are counted by them); elapsed on"), "{line}");
        assert!(line.contains("uncovered queue time 0.00 ms (no stamped span read by the window's close covers it: "), "{line}");
        assert!(!line.contains("no stamped list on the queue") && !line.contains("no frame ended"), "{line}");
    }

    /// The same line with ends read in the window: two present copies, two
    /// lists, a clamp group and a relay fetch. The count of frames is the
    /// present copies read with their end in it, said so, and each ratio is
    /// the window's time of its kind over the pairs that ended (Sol's fifth
    /// read of the frame timers: "{frames} frames ended in it (by their
    /// copies to the swap chain)" gave the count as a total of frames, and
    /// only the zero branch had a fixture).
    #[test]
    fn a_ratio_with_pairs_ended_divides_by_them() {
        let mut spans = vec![
            (Kind::List, 0, 40),
            (Kind::Clamp, 10, 20),
            (Kind::Relay, 40, 50),
            (Kind::List, 50, 90),
            (Kind::Present, 90, 95),
            (Kind::Present, 95, 98),
        ];
        let window = close(&mut spans, 0, 100);
        assert_eq!([Kind::List, Kind::Clamp, Kind::Relay, Kind::Present].map(|k| window.pairs[k as usize]), [2, 1, 1, 2]);
        assert_eq!([Kind::List, Kind::Clamp, Kind::Relay].map(|k| window.ticks[k as usize]), [70, 10, 10]);
        let closed = Closed { from: 0, to: 100, window, flaws: Flaws::default(), late: (0, 0), programs: super::ByProgram::default() };
        let line = text(&closed, 1_000, None);
        assert!(
            line.contains(
                ", 2 present copies read with their end in it (frames are counted by them), 50.00 ms a frame; elapsed on the \
                 queue a frame, "
            ),
            "{line}"
        );
        for said in [
            "35.000 ms a list (the window's title-list time over the lists that ended in it)",
            "10.0000 ms a group (the window's clamp time over the groups that ended in it)",
            "10.000 ms a fetch (the window's relay time over the fetches that ended in it)",
        ] {
            assert!(line.contains(said), "{said} in {line}");
        }
        assert!(!line.contains("frames ended in it") && !line.contains("unavailable"), "{line}");
    }

    /// The line's figures add up to its ms a frame within their rounding,
    /// and a flaw makes the window incomplete in so many words.
    #[test]
    fn the_line_adds_up_and_says_what_is_missing() {
        let mut spans = vec![
            (Kind::List, 0, 4_000_000),
            (Kind::Clamp, 1_000_000, 1_500_000),
            (Kind::Present, 4_000_000, 4_100_000),
            (Kind::List, 5_000_000, 8_000_000),
            (Kind::Present, 8_000_000, 8_050_000),
        ];
        let window = close(&mut spans, 0, 10_000_000);
        let closed = Closed { from: 0, to: 10_000_000, window, flaws: Flaws::default(), late: (0, 0), programs: super::ByProgram::default() };
        let line = text(&closed, 10_000_000, None);
        let per: f64 = line.split(" ms a frame").next().unwrap().rsplit(", ").next().unwrap().parse().unwrap();
        assert_eq!(per, 500.0, "{line}");
        let parts: f64 = [
            "title lists ",
            "our clamp groups ",
            "preludes ",
            "relay fetches ",
            // Its own figure, not the count of those read ("2 present
            // copies read with their end in it").
            ", present copies ",
            "hash lists ",
            "compare lists ",
            "synchronous fetches ",
            "uncovered queue time ",
        ]
        .iter()
        .map(|label| line.split(label).nth(1).unwrap().split(' ').next().unwrap().parse::<f64>().unwrap())
        .sum();
        assert!((parts - per).abs() < 0.05, "{parts} of {per}: {line}");
        assert!(line.ends_with("put on the card's clock by the queue's calibration)"), "{line}");
        assert!(!line.contains("complete"), "{line}");
        let flawed = Closed { flaws: Flaws { unclosed: 1, ..Flaws::default() }, ..closed };
        let line = text(&flawed, 10_000_000, None);
        assert!(
            line.ends_with(
                "; the window is incomplete: 1 lists without their own pair; its frames and parts are not given (its samples \
                 cannot establish them)"
            ),
            "{line}"
        );
        assert!(!line.contains("title lists") && !line.contains("frames ended"), "figures given for an incomplete window: {line}");
    }

    /// With the passes on our texture uploads in title lists have their own
    /// figure, out of the lists', and the line still adds up; without them
    /// the line is as before.
    #[test]
    fn our_uploads_are_out_of_the_title_lists_and_the_line_adds_up() {
        let mut spans = vec![
            (Kind::List, 0, 4_000_000),
            (Kind::Upload, 1_000_000, 1_250_000),
            (Kind::Clamp, 2_000_000, 2_500_000),
            (Kind::Present, 4_000_000, 4_100_000),
        ];
        let window = close(&mut spans, 0, 5_000_000);
        assert_eq!(window.ticks[Kind::Upload as usize], 250_000);
        assert_eq!(window.ticks[Kind::List as usize], 4_000_000 - 250_000 - 500_000);
        let closed = Closed { from: 0, to: 5_000_000, window, flaws: Flaws::default(), late: (0, 0), programs: super::ByProgram::default() };
        let line = text_with(&closed, 10_000_000, None, true);
        let per: f64 = line.split(" ms a frame").next().unwrap().rsplit(", ").next().unwrap().parse().unwrap();
        let parts: f64 = [
            "title lists ",
            "our clamp groups ",
            "preludes ",
            "relay fetches ",
            ", present copies ",
            "hash lists ",
            "compare lists ",
            "synchronous fetches ",
            "uncovered queue time ",
            "our texture uploads in them ",
        ]
        .iter()
        .map(|label| line.split(label).nth(1).unwrap().split(' ').next().unwrap().parse::<f64>().unwrap())
        .sum();
        assert!((parts - per).abs() < 0.05, "{parts} of {per}: {line}");
        assert!(line.contains("our texture uploads in them 25.00 ms (1.0 ended"), "{line}");
        assert!(!text_with(&closed, 10_000_000, None, false).contains("texture uploads"), "uploads said with the passes off");
    }
}

/// The lines by program (`VICTORY_GPU_STAMPS=2`): the sweep, the lines and
/// the tally, off the card.
#[cfg(test)]
mod program_tests {
    use super::{AFTER, ByProgram, Closed, Flaws, Kind, Part, ProgramStage, Seen, Tally, Window, close_all, parts_of, programs_text};

    /// A list read after a cut at 100, spanning 90 to 120, with two calls
    /// marked at 100, and a span of no length at 100: read in time, both
    /// calls and the span end in the window that closes at 100; read late,
    /// they count in no later window, and the spans are said as late (Sol's
    /// fourth read of the frame timers: the 100-to-100 part and the span
    /// counted in the window after the cut).
    #[test]
    fn a_late_end_at_the_cut_counts_in_no_later_window() {
        let (p1, p2) = (1, 2);
        let same = |at: u64| at;
        let read = |seen: &mut Seen| {
            let a = seen.register(90);
            let mut flaws = Flaws::default();
            let parts = parts_of((90, 120), [(Some(100), p1), (Some(100), p2)], &mut flaws);
            assert_eq!(flaws, Flaws::default());
            seen.parts_read(parts);
            seen.deliver(a, 90, 130, vec![(Kind::List, 90, 120), (Kind::Present, 100, 100)], flaws);
        };
        let mut timely = Seen::default();
        read(&mut timely);
        let first = timely.window(100, 0, &same).expect("a window to 100");
        assert_eq!((first.from, first.to), (90, 100));
        let ends = |c: &Closed| (c.programs.of(p1).1, c.programs.of(p2).1, c.window.pairs[Kind::Present as usize]);
        assert_eq!(ends(&first), (1, 1, 1), "read in time: {:?}", first.programs);
        let mut late = Seen { cut: Some(100), ..Seen::default() };
        read(&mut late);
        let next = late.window(200, 0, &same).expect("the window after the cut");
        assert_eq!((next.from, next.to), (100, 200));
        assert_eq!(ends(&next), (0, 0, 0), "an end at the cut counted after it: {:?}", next.programs);
        assert_eq!(next.window.pairs[Kind::List as usize], 1, "the list ends at 120");
        assert_eq!(next.late, (2, 10), "the list's 90..100, and the span at the cut");
    }

    /// A window counted at its close, then a span read after it closed that
    /// lies partly in it (its send's moment put on the card's clock 10 ticks
    /// late): the tally leaves the window out and says so (Sol's fourth read
    /// of the frame timers: the tally kept it, and said no losses were
    /// detected in the windows counted).
    #[test]
    fn a_window_that_work_read_later_lies_in_leaves_the_tally() {
        let same = |at: u64| at;
        let mut seen = Seen::default();
        let a = seen.register(100);
        seen.deliver(a, 100, 150, vec![(Kind::List, 100, 140), (Kind::Present, 140, 145)], Flaws::default());
        let b = seen.register(190);
        let first = seen.window(400, 0, &same).expect("a window before b went");
        assert_eq!((first.from, first.to, first.flaws), (100, 190, Flaws::default()));
        seen.tally.add(&first, 1_000);
        // b's work began at 180 on the card, before the first window's end.
        seen.deliver(b, 190, 260, vec![(Kind::List, 180, 250), (Kind::Present, 250, 255)], Flaws::default());
        let second = seen.window(400, 0, &same).expect("the next window");
        assert_eq!((second.late, second.flaws), ((1, 10), Flaws::default()));
        seen.tally.add(&second, 1_060);
        let t = seen.tally.due(1_060, 50).expect("due");
        assert_eq!((t.windows, t.incomplete, t.frames, t.wall), (1, 0, 1, 210), "the first window kept: {t:?}");
        let last = programs_text(&t, 1_000, &|_| None).pop().expect("a line");
        assert!(last.contains("; 1 more windows left out after their close: work read later lies in them"), "{last}");
    }

    /// The lines by program printed with a window, then a call marked at
    /// that window's end read: its part (of no length, its list begun at
    /// the end) lies in the printed window, which the next lines say; the
    /// window after it stays counted.
    #[test]
    fn work_read_after_the_lines_by_program_is_said_in_the_next() {
        let same = |at: u64| at;
        let mut seen = Seen::default();
        let a = seen.register(100);
        seen.deliver(a, 100, 150, vec![(Kind::List, 100, 140)], Flaws::default());
        let b = seen.register(190);
        let first = seen.window(400, 0, &same).expect("a window before b went");
        assert_eq!(seen.tally_closed(&first, 1_000, 0).expect("due at once").windows, 1);
        let mut flaws = Flaws::default();
        seen.parts_read(parts_of((190, 230), [(Some(190), 7)], &mut flaws));
        seen.deliver(b, 190, 260, vec![(Kind::List, 190, 230)], flaws);
        let second = seen.window(400, 0, &same).expect("the next window");
        assert_eq!((second.late, second.programs.of(7)), ((0, 0), (0, 0)), "the list begins at the cut; its call is before it");
        let next = seen.tally_closed(&second, 1_060, 0).expect("due");
        assert_eq!((next.windows, next.late, next.earlier), (1, 0, 1), "{next:?}");
        let last = programs_text(&next, 1_000, &|_| None).pop().expect("a line");
        assert!(
            last.ends_with(
                "; 1 windows the lines before counted have work read since that lies in them (those lines' figures miss it)"
            ),
            "{last}"
        );
    }

    /// Every window of a tally left out: the lines by program say there is
    /// no usable window and give no figures (Sol's fourth read of the frame
    /// timers: they gave zero-valued ms a frame).
    #[test]
    fn a_tally_with_no_usable_window_says_so() {
        let tally = Tally { since: Some(0), incomplete: 3, ..Tally::default() };
        let lines = programs_text(&tally, 10_000_000, &|_| None);
        assert_eq!(lines.len(), 1, "{lines:#?}");
        assert!(!lines[0].contains("ms a frame"), "figures for no usable window: {}", lines[0]);
        assert!(
            lines[0].starts_with(
                "card time by program (VICTORY_GPU_STAMPS=2): no usable windows: 3 windows left out, incomplete at their close"
            ),
            "{}",
            lines[0]
        );
    }

    /// Each tick a window gives title lists goes to one part: the call's
    /// program, the part after a list's last call, or none (a list without
    /// call stamps); a clamp group's ticks stay the group's; where two lists'
    /// parts overlap the part begun last has the tick. The parts add up to
    /// the title-list ticks; a part crossing the window's end is split there
    /// and its calls counted in the window its end lies in.
    #[test]
    fn each_tick_of_a_title_list_goes_to_one_part() {
        let (p1, p2) = (1, 2);
        let mut spans = vec![
            (Kind::List, 0, 100),
            (Kind::Clamp, 30, 40),
            (Kind::List, 90, 150),
            (Kind::List, 160, 170),
        ];
        let mut parts: Vec<Part> = vec![(p1, 1, 0, 20), (p2, 2, 20, 60), (AFTER, 0, 60, 100), (p1, 1, 90, 130), (AFTER, 0, 130, 150)];
        let (w, by) = close_all(&mut spans, Some(&mut parts), 0, 120);
        assert_eq!(w.ticks[Kind::Clamp as usize], 10);
        assert_eq!(w.ticks[Kind::List as usize], 110);
        assert_eq!(by.ticks(), w.ticks[Kind::List as usize], "{by:?}");
        // 0..20, and 90..120 of the second list (begun after the first's
        // part after its last call).
        assert_eq!(by.of(p1), (20 + 30, 1));
        assert_eq!(by.of(p2), (40 - 10, 2), "without the clamp group");
        assert_eq!(by.of(AFTER), (30, 0));
        assert_eq!(by.unattributed, 0);
        assert_eq!(parts, [(p1, 1, 90, 130), (AFTER, 0, 130, 150)], "carried with their own beginnings");
        let (w, by) = close_all(&mut spans, Some(&mut parts), 120, 200);
        assert_eq!(w.ticks[Kind::List as usize], 30 + 10);
        assert_eq!(by.of(p1), (10, 1), "the call counted where its part ends");
        assert_eq!(by.of(AFTER), (20, 0));
        assert_eq!(by.unattributed, 10, "a list without call stamps");
        assert_eq!(by.ticks(), w.ticks[Kind::List as usize]);
        assert!(parts.is_empty());
    }

    /// A stamp after a call that lies past its list's close, or before the
    /// stamp before it, makes no part, and the call after it counts at its
    /// own stamp with no time: no part of list A lies outside it, and list
    /// B, begun earlier and still running, keeps its own ticks (Sol's third
    /// read of the frame timers: a part past A's close, begun after B's,
    /// took B's ticks by time).
    #[test]
    fn a_call_stamp_out_of_place_makes_no_part() {
        let (p1, p2, p3, p4, b1) = (1, 2, 3, 4, 5);
        let mut flaws = Flaws::default();
        // A, 100..200: p1 at 150, p2 at 140 (before p1's), p3 at 170, p4 at
        // 250 (past A's close).
        let a = parts_of((100, 200), [(Some(150), p1), (Some(140), p2), (Some(170), p3), (Some(250), p4)], &mut flaws);
        assert_eq!(flaws.calls_bad, 2, "{flaws:?}");
        assert!(a.iter().all(|p| 100 <= p.2 && p.2 <= p.3 && p.3 <= 200), "a part outside its list: {a:?}");
        // B, 120..300: one call, at 290.
        let b = parts_of((120, 300), [(Some(290), b1)], &mut Flaws::default());
        let mut spans = vec![(Kind::List, 100, 200), (Kind::List, 120, 300)];
        let mut parts: Vec<Part> = a.into_iter().chain(b).collect();
        let (w, by) = close_all(&mut spans, Some(&mut parts), 0, 400);
        assert_eq!(by.ticks(), w.ticks[Kind::List as usize]);
        assert_eq!(by.of(p4), (0, 0), "p4's stamp past A's close: {by:?}");
        assert_eq!(by.of(p2), (0, 0), "p2's stamp before p1's: {by:?}");
        assert_eq!(by.of(p3), (0, 1), "p3 after a stamp out of place: counted, no time: {by:?}");
        // A's p1 alone 100..120, then B's (begun later) to 290, then B's
        // part after its last call.
        assert_eq!((by.of(p1), by.of(b1), by.of(AFTER)), ((20, 1), (170, 1), (10, 0)), "{by:?}");
        assert_eq!(by.unattributed, 0);
    }

    /// Two lists' parts overlap: B `[20, 45)`, read first, and A `[10, 50)`.
    /// Cut at 30 or swept whole, B (begun later) has `[20, 45)` (Sol's third
    /// read of the frame timers: cut, both carried parts began at the cut,
    /// and A, after B in the order read, took `[30, 45)`).
    #[test]
    fn splitting_a_window_keeps_the_owner_of_an_overlap() {
        let (a, b) = (1, 2);
        let parts0: Vec<Part> = vec![(b, 1, 20, 45), (a, 1, 10, 50)];
        let spans0 = vec![(Kind::List, 20, 45), (Kind::List, 10, 50)];
        let (mut spans, mut parts) = (spans0.clone(), parts0.clone());
        let (_, whole) = close_all(&mut spans, Some(&mut parts), 0, 100);
        assert_eq!((whole.of(a), whole.of(b)), ((15, 1), (25, 1)));
        let (mut spans, mut parts) = (spans0, parts0);
        let (_, mut split) = close_all(&mut spans, Some(&mut parts), 0, 30);
        let (_, second) = close_all(&mut spans, Some(&mut parts), 30, 100);
        split.join(&second);
        assert_eq!(split, whole, "cut at 30");
    }

    /// Each call counts in the window its own stamp lies in, a run of one
    /// program's calls too, and two stamps at the same tick make a part of
    /// no time that counts; after a lost stamp the next call counts at its
    /// own (Sol's third read of the frame timers: a run's calls all counted
    /// at its last stamp, in the window after a cut, and the call after a
    /// lost stamp was not counted).
    #[test]
    fn each_call_counts_at_its_own_stamp() {
        let p = 7;
        let mut flaws = Flaws::default();
        let mut parts = parts_of((0, 40), [(Some(10), p), (Some(10), p), (Some(30), p)], &mut flaws);
        assert_eq!(flaws, Flaws::default());
        let mut spans = vec![(Kind::List, 0, 40)];
        let (_, first) = close_all(&mut spans, Some(&mut parts), 0, 25);
        let (_, second) = close_all(&mut spans, Some(&mut parts), 25, 50);
        assert_eq!(first.of(p), (25, 2), "two calls' stamps before the cut");
        assert_eq!((second.of(p), second.of(AFTER)), ((5, 1), (10, 0)));
        let mut flaws = Flaws::default();
        let parts = parts_of((0, 40), [(Some(10), 1), (None, 2), (Some(30), 3)], &mut flaws);
        assert_eq!(flaws.calls_lost, 1);
        assert_eq!(parts, [(1, 1, 0, 10), (3, 1, 30, 30), (AFTER, 0, 30, 40)]);
    }

    fn closed(list: u64, frames: u64, by: ByProgram) -> Closed {
        let mut window = Window::default();
        window.ticks[Kind::List as usize] = list;
        window.pairs[Kind::Present as usize] = frames;
        window.uncovered = 1_000_000 - list;
        Closed { from: 0, to: 1_000_000, window, flaws: Flaws::default(), late: (0, 0), programs: by }
    }

    /// A tally of windows is due once its first window closed five seconds
    /// before (here the CPU's counter at 10 ticks a second), with their sums;
    /// then a new one begins. A flawed window counts as incomplete.
    #[test]
    fn a_tally_is_due_five_seconds_after_its_first_window() {
        let mut tally = Tally::default();
        let mut by = ByProgram::default();
        by.add(3, 400_000, 7);
        by.add(AFTER, 100_000, 0);
        tally.add(&closed(500_000, 2, by.clone()), 100);
        assert!(tally.due(140, 50).is_none(), "4 s after its first window");
        let mut flawed = closed(500_000, 3, by);
        flawed.flaws.calls_lost = 1;
        tally.add(&flawed, 145);
        let t = tally.due(150, 50).expect("5 s after its first window");
        assert_eq!((t.windows, t.incomplete, t.frames, t.wall, t.list), (1, 1, 2, 1_000_000, 500_000), "the flawed window left out");
        assert_eq!(t.by.of(3), (400_000, 7));
        assert_eq!(t.by.of(AFTER), (100_000, 0));
        assert_eq!(tally, Tally::default(), "a new tally begins");
    }

    /// The lines by program: one a program for the 30 with the most time, in
    /// order, each with its stage and shaders, ms and calls a frame, ms a
    /// call and its share of the title lists; then one with what they cover,
    /// how many programs ran, the part after lists' last calls and the part
    /// in no part. Their numbers add up to the title lists' ms a frame.
    #[test]
    fn the_lines_by_program_rank_parse_and_add_up() {
        let mut by = ByProgram::default();
        // 35 programs: program n has n * 1,000 ticks and n calls.
        for n in 1..=35 {
            by.add(n, u64::from(n) * 1_000, u64::from(n));
        }
        by.add(AFTER, 20_000, 0);
        by.unattributed = 5_000;
        let list = by.ticks();
        let tally = Tally { since: Some(0), windows: 5, wall: 50_000_000, frames: 10, list, by, ..Tally::default() };
        let name = |n: u32| Some((if n.is_multiple_of(2) { ProgramStage::Pixel } else { ProgramStage::Compute }, format!("[test program {n}]")));
        let lines = programs_text(&tally, 10_000_000, &name);
        assert_eq!(lines.len(), 31, "{lines:#?}");
        // Rank 1: program 35, 35,000 ticks at 10 MHz over 10 frames.
        let first = &lines[0];
        assert!(first.starts_with("card time by program (VICTORY_GPU_STAMPS=2): rank 1: stage compute, "), "{first}");
        let numbers = |line: &str| -> Vec<f64> {
            line.split(", shaders=")
                .next()
                .unwrap()
                .split(' ')
                .filter_map(|w| w.trim_end_matches([',', '%', ':']).parse::<f64>().ok())
                .collect()
        };
        // rank, ms a frame, calls a frame, ms a call, share.
        assert!(first.contains(" ms a call (the windows' time over the calls that ended in them), "), "{first}");
        let n = numbers(first);
        assert_eq!(n[..4], [1.0, 0.35, 3.5, 0.1], "{first}");
        assert!(n.len() == 5 && (n[4] - 100.0 * 35_000.0 / list as f64).abs() < 0.006, "{first}");
        assert!(first.ends_with(", shaders=[test program 35]"), "{first}");
        assert!(lines[1].contains("rank 2: stage pixel, ") && lines[1].ends_with("[test program 34]"), "{}", lines[1]);
        assert!(lines[29].contains("rank 30: ") && lines[29].ends_with("[test program 6]"), "{}", lines[29]);
        let sum: f64 = lines[..30].iter().map(|l| numbers(l)[1]).sum();
        let top: f64 = (6..=35).map(|n| f64::from(n) * 1_000.0).sum::<f64>() * 1e3 / 1e7 / 10.0;
        assert!((sum - top).abs() < 0.001 * 30.0, "{sum} against {top}");
        let last = &lines[30];
        assert!(last.starts_with("card time by program (VICTORY_GPU_STAMPS=2): over 5 windows "), "{last}");
        let field = |label: &str| -> f64 {
            last.split(label).nth(1).unwrap_or_else(|| panic!("{label} in {last}")).split([' ', ',', ';', '%']).next().unwrap().parse().unwrap()
        };
        assert_eq!(field("title lists "), list as f64 * 1e3 / 1e7 / 10.0);
        assert!((field("the top 30 programs ") - top).abs() < 0.001, "{last}");
        assert!((field("the top 30 programs ") / field("title lists ") * 100.0 - field("ms a frame (")).abs() < 0.01, "{last}");
        assert_eq!(field("of the "), 35.0, "programs with time or a call: {last}");
        assert!(last.contains(" of the 35 with time attributed or a call counted; "), "{last}");
        let after = "after the last call stamp of a list, or in a list with none, ";
        assert_eq!(field(after), 0.2);
        assert_eq!(field("in no part "), 0.05);
        let all = field("all programs ") + field(after) + field("in no part ");
        assert!((all - field("title lists ")).abs() < 0.002, "{last}");
        assert!(
            last.ends_with(
                "; in the windows counted no loss was detected within estimated bounds, checked at each one's close and \
                 against what was read after it up to this line"
            ),
            "{last}"
        );
        assert!(
            last.contains(
                "; a program's time is the title-list queue time attributed to its calls between completion marks in the \
                 windows counted: each call has the stretch from the stamp before it (after the call before, or at the \
                 list's opening) to the stamp after it, clipped to the windows, less our clamp groups' time in it, and a \
                 tick where two lists' parts overlap goes to one of them (the one begun last); "
            ),
            "{last}"
        );
        assert!(!last.contains("that ran") && !last.contains("overlap with the call before"), "{last}");
        assert!(!last.contains("complete"), "{last}");
        // A program with time and no call that ended in the windows.
        let mut by = ByProgram::default();
        by.add(1, 5_000, 0);
        let lone = Tally { since: Some(0), windows: 1, wall: 10_000_000, frames: 1, list: 5_000, by, ..Tally::default() };
        let line = &programs_text(&lone, 10_000_000, &name)[0];
        assert!(line.contains(", unavailable ms a call ("), "{line}");
        // Windows left out at their close and after it, and work read
        // since the lines before in windows they counted: each said.
        let flawed = Tally { incomplete: 2, late: 1, earlier: 1, ..tally };
        let last = &programs_text(&flawed, 10_000_000, &name)[30];
        assert!(
            last.ends_with(
                "; 2 more windows left out, incomplete at their close (their lines say why); 1 more windows left out after \
                 their close: work read later lies in them; in the windows counted no loss was detected within estimated \
                 bounds, checked at each one's close and against what was read after it up to this line; 1 windows the \
                 lines before counted have work read since that lies in them (those lines' figures miss it)"
            ),
            "{last}"
        );
    }
}

/// The lines by program (`VICTORY_GPU_STAMPS=2`) on the software adapter:
/// real lists, real stamps, read after their fence.
#[cfg(test)]
pub(crate) mod program_tests_on_the_card {
    use super::{AFTER, Kind, Sink, card_clock, program, ProgramStage};
    use crate::host::{HostList, test_host};
    use crate::pipeline::host::HostPipeline;
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::s;

    /// A compute program of ours: `iterations` steps of a hash a thread,
    /// stored through a root view.
    pub(crate) fn compute(iterations: u32, name: &str) -> HostPipeline {
        let source = format!(
            "#define RS \"UAV(u0)\"\nRWByteAddressBuffer b : register(u0);\n[RootSignature(RS)]\n[numthreads(64, 1, 1)]\n\
             void main(uint3 id : SV_DispatchThreadID) {{ uint x = id.x; [loop] for (uint i = 0; i < {iterations}u; ++i) \
             {{ x = x * 1664525u + 1013904223u; }} b.Store((id.x % 1024u) * 4u, x); }}\n"
        );
        let host = test_host().expect("a host device");
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; then the
        // blob's own bytes, which hold the root signature too.
        unsafe {
            D3DCompile(source.as_ptr().cast(), source.len(), s!("test"), None, None, s!("main"), s!("cs_5_1"), 0, 0, &mut code, None)
                .expect("compiled");
            let code = code.expect("code");
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                    ..Default::default()
                })
                .unwrap();
            pipeline(true, pso, root, name)
        }
    }

    fn pipeline(compute: bool, pso: ID3D12PipelineState, root: ID3D12RootSignature, name: &str) -> HostPipeline {
        HostPipeline {
            compute,
            mesh: false,
            pso,
            root,
            stages: Vec::new(),
            slots: 0,
            inputs: Vec::new(),
            rects: false,
            name: name.to_owned(),
            state: String::new(),
            program: std::sync::OnceLock::new(),
        }
    }

    /// `groups` groups of `p` into `buffer`, then the call's stamp as a
    /// title call's (`HostList::stamp_call`).
    pub(crate) fn dispatch(list: &HostList, p: &HostPipeline, buffer: &ID3D12Resource, groups: u32) {
        // SAFETY: an open list; our pipeline and a live buffer.
        unsafe {
            list.list.SetComputeRootSignature(&p.root);
            list.list.SetPipelineState(&p.pso);
            list.list.SetComputeRootUnorderedAccessView(0, buffer.GetGPUVirtualAddress());
            list.list.Dispatch(groups, 1, 1);
        }
        list.stamp_call(p);
    }

    /// The list's window: closed at the card's clock now, by the queue's
    /// calibration, with its ticks by program and its line.
    fn window(sink: &Sink) -> (super::Window, super::Flaws, super::ByProgram, String) {
        let frequency = test_host().unwrap().clock_calibration().unwrap().frequency;
        let (on_card, now) = card_clock();
        sink.window_by(now, &on_card, frequency).expect("a window")
    }

    /// A list with a heavy dispatch and a trivial one after it: the heavy
    /// one's part is most of the list, each program has its one call, and
    /// the parts add up to the list's ticks.
    #[test]
    fn the_heavy_dispatch_gets_the_larger_share() {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let sink = Sink::calls();
        let (heavy, light) = (compute(20_000, "[compute heavy test]"), compute(1, "[compute light test]"));
        let buffer = host.uav_buffer(4096).unwrap();
        let list = host.new_list().unwrap();
        dispatch(&list, &heavy, &buffer, 64);
        dispatch(&list, &light, &buffer, 1);
        list.close();
        host.execute(&[&list]);
        let (w, flaws, by, line) = window(&sink);
        assert_eq!(flaws, super::Flaws::default(), "{line}");
        let list_ticks = w.ticks[Kind::List as usize];
        assert_eq!(by.ticks(), list_ticks, "{by:?}");
        let ((h, hn), (l, ln)) = (by.of(heavy.program()), by.of(light.program()));
        assert_eq!((hn, ln), (1, 1), "{by:?}");
        assert!(h > 4 * l && h * 2 > list_ticks, "heavy {h} ticks, light {l}, the list {list_ticks}: {by:?}");
    }

    /// A list that interleaves three programs, with a clamp group of ours
    /// in it: each program's calls are counted, the parts and the group add
    /// up to the list's span exactly, and nothing is in no part.
    #[test]
    fn three_interleaved_programs_add_up_to_their_list() {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let sink = Sink::calls();
        let ps = [compute(50, "[compute a test]"), compute(100, "[compute b test]"), compute(200, "[compute c test]")];
        let buffer = host.uav_buffer(4096).unwrap();
        let (src, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: inside the mapping: two mesh records.
        unsafe { (data as *mut [u32; 6]).write([3, 1, 1, 0, 1, 1]) };
        let list = host.new_list().unwrap();
        for (i, at) in [0, 1, 2, 0, 0, 1, 2, 2, 0].into_iter().enumerate() {
            if i == 6 {
                let scratch = crate::indirect::bound(&list, (&src, 0), None, (10, 12, 2)).expect("the clamp");
                crate::indirect::done(&list, &scratch);
            }
            dispatch(&list, &ps[at], &buffer, 8);
        }
        list.close();
        host.execute(&[&list]);
        let span = sink.spans().iter().find(|s| s.0 == Kind::List).map(|s| s.2 - s.1).expect("the list's pair");
        let (w, flaws, by, line) = window(&sink);
        assert_eq!(flaws, super::Flaws::default(), "{line}");
        assert_eq!(w.pairs[Kind::Clamp as usize], 1);
        assert_eq!(by.ticks() + w.ticks[Kind::Clamp as usize], span, "the parts and the group against the list's span: {by:?}");
        assert_eq!(by.ticks(), w.ticks[Kind::List as usize]);
        assert_eq!(by.unattributed, 0);
        let calls: Vec<u64> = ps.iter().map(|p| by.of(p.program()).1).collect();
        assert_eq!(calls, [4, 2, 3]);
        assert_eq!(by.of(AFTER).1, 0);
    }

    /// With the stamps at level 1, or off, a title call is not stamped: the
    /// list's slots are its opening and its close (none off), no part is
    /// read and no query heap is taken off. At level 2 the same calls are
    /// (the control).
    #[test]
    fn level_one_and_off_make_no_call_stamps() {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let p = compute(1, "[compute level test]");
        let buffer = host.uav_buffer(4096).unwrap();
        {
            let sink = Sink::calls();
            let list = host.new_list().unwrap();
            let opened = list.stamp_slots();
            dispatch(&list, &p, &buffer, 1);
            dispatch(&list, &p, &buffer, 1);
            assert_eq!(list.stamp_slots(), opened + 2, "the control: level 2 stamps each call");
            list.close();
            host.execute(&[&list]);
            assert_eq!(sink.parts().iter().map(|part| part.1).sum::<u32>(), 2, "{:?}", sink.parts());
        }
        {
            let sink = Sink::new();
            let list = host.new_list().unwrap();
            let opened = list.stamp_slots();
            dispatch(&list, &p, &buffer, 1);
            dispatch(&list, &p, &buffer, 1);
            assert_eq!(list.stamp_slots(), opened, "a call stamped at level 1");
            list.close();
            host.execute(&[&list]);
            assert_eq!(sink.spans().iter().filter(|s| s.0 == Kind::List).count(), 1);
            assert!(sink.parts().is_empty(), "parts read at level 1: {:?}", sink.parts());
        }
        let _off = super::Off::new();
        let taken = super::taken();
        let list = host.new_list().unwrap();
        dispatch(&list, &p, &buffer, 1);
        list.close();
        host.execute(&[&list]);
        assert_eq!((super::taken(), list.stamp_slots()), (taken, 0), "a stamp with the switch off");
    }

    /// A stamp after a call that finds no query heap: the window says so,
    /// the stretch it would have cut goes to no program, and the parts still
    /// add up to the list's ticks.
    #[test]
    fn a_lost_call_stamp_makes_its_window_incomplete() {
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let sink = Sink::calls();
        let [p1, p2, p3] = [1, 2, 3].map(|n| program(ProgramStage::Compute, &format!("[compute lost test {n}]")));
        let list = host.new_list().unwrap();
        for _ in 1..255 {
            list.stamp_begin();
        }
        list.stamp_program(p1);
        super::no_chunks(true);
        list.stamp_program(p2);
        super::no_chunks(false);
        list.stamp_program(p3);
        list.close();
        host.execute(&[&list]);
        let (w, flaws, by, line) = window(&sink);
        assert_eq!(flaws.calls_lost, 1, "{line}");
        assert!(line.contains("; its parts by program are incomplete: 1 stamps after a title call lost"), "{line}");
        assert!(line.contains("; no losses detected within estimated bounds"), "the window's figures by kind stand: {line}");
        assert_eq!((by.of(p1).1, by.of(p2).1, by.of(p3).1), (1, 0, 1), "the call after the lost stamp counts: {by:?}");
        assert_eq!(by.ticks(), w.ticks[Kind::List as usize]);
    }

    /// Diagnostics: what a list of 5,000 trivial draws costs with the
    /// stamps at level 2 against level 1 on this adapter (recording, the
    /// submission waited for and read, the list's span on the queue, query
    /// heaps taken). Run by name with `--ignored --nocapture`.
    #[test]
    #[ignore = "a measurement: run it by name with --ignored --nocapture"]
    fn level_two_cost() {
        const DRAWS: u32 = 5_000;
        let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let source = "#define RS \"RootFlags(0)\"\n[RootSignature(RS)]\nfloat4 main(uint v : SV_VertexID) : SV_Position { return float4(0, 0, 0, 1); }\n";
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: as in `compute`; a vertex-only pipeline with no target.
        let p = unsafe {
            D3DCompile(source.as_ptr().cast(), source.len(), s!("test"), None, None, s!("main"), s!("vs_5_1"), 0, 0, &mut code, None)
                .expect("compiled");
            let code = code.expect("code");
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let pso: ID3D12PipelineState = host.device.CreateGraphicsPipelineState(&desc).unwrap();
            pipeline(false, pso, root, "[vertex cost test]")
        };
        let frequency = host.clock_calibration().unwrap().frequency as f64;
        let run = |level: u8| -> [f64; 5] {
            let sink = if level == 2 { Sink::calls() } else { Sink::new() };
            let taken = super::taken();
            let started = std::time::Instant::now();
            let list = host.new_list().unwrap();
            // SAFETY: an open list; our pipeline.
            unsafe {
                list.list.SetGraphicsRootSignature(&p.root);
                list.list.SetPipelineState(&p.pso);
                list.list.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                list.list.RSSetViewports(&[D3D12_VIEWPORT { Width: 1.0, Height: 1.0, MaxDepth: 1.0, ..Default::default() }]);
                list.list.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: 1, bottom: 1 }]);
            }
            for _ in 0..DRAWS {
                // SAFETY: as above.
                unsafe { list.list.DrawInstanced(3, 1, 0, 0) };
                list.stamp_call(&p);
            }
            list.close();
            let recorded = started.elapsed().as_secs_f64() * 1e3;
            let sent = std::time::Instant::now();
            host.execute(&[&list]);
            let ran = sent.elapsed().as_secs_f64() * 1e3;
            let span = sink.spans().iter().find(|s| s.0 == Kind::List).map_or(0, |s| s.2 - s.1) as f64 * 1e3 / frequency;
            let closing = std::time::Instant::now();
            let _ = window(&sink);
            let closed = closing.elapsed().as_secs_f64() * 1e3;
            [recorded, ran, span, closed, (super::taken() - taken) as f64]
        };
        run(1);
        run(2);
        let mut results: [Vec<[f64; 5]>; 2] = [Vec::new(), Vec::new()];
        for _ in 0..5 {
            results[0].push(run(1));
            results[1].push(run(2));
        }
        for (level, runs) in [1, 2].into_iter().zip(&results) {
            let median = |i: usize| {
                let mut v: Vec<f64> = runs.iter().map(|r| r[i]).collect();
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            };
            eprintln!(
                "level {level}, {DRAWS} trivial draws in one list (median of 5): recording {:.2} ms, sent to read {:.2} ms, the \
                 list's span on the queue {:.2} ms, the window closed in {:.3} ms, {} query heaps of 256 taken",
                median(0),
                median(1),
                median(2),
                median(3),
                median(4)
            );
        }
    }
}
