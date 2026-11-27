//! Presented frames written to PNG files, to see what reaches the screen:
//! with `VICTORY_CAPTURE=<folder>`, every frame whose number is a power of
//! two (the frames the log reports) is saved as `frame-<n>.png`, at half
//! size. Only in builds with diagnostics.

use std::path::PathBuf;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// The folder frames go to, if capturing.
pub(crate) fn folder() -> Option<&'static PathBuf> {
    static FOLDER: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    FOLDER
        .get_or_init(|| std::env::var_os("VICTORY_CAPTURE").map(PathBuf::from))
        .as_ref()
}

/// Frames listed in `VICTORY_CAPTURE_FRAMES` (comma-separated): saved
/// too, and their draws traced at debug level. `VICTORY_CAPTURE_HEAVY_FRAMES`
/// lists frames counted from the heavy scene's first ([`heavy_frame`]).
pub(crate) fn listed(n: u64) -> bool {
    static LIST: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    let named = LIST
        .get_or_init(|| {
            let list = std::env::var("VICTORY_CAPTURE_FRAMES").unwrap_or_default();
            list.split(',')
                .filter_map(|f| f.trim().parse().ok())
                .collect()
        })
        .contains(&n);
    // `VICTORY_CAPTURE_WORLD`: also the 2nd and the 6th frame after the first
    // one that ran 500 draws or more (a title's first 3D frames: their
    // numbers depend on how long the menus were shown), and the 30th and
    // the 80th (later frames, in case the first ones are dark by design).
    //
    // `VICTORY_CAPTURE_WORLD_ALSO` (comma-separated): more frames counted
    // from that first one, to pick out a frame where a fault of interest
    // shows (a fault that appears only once streamed content has arrived
    // may be in none of the four above).
    static ALSO: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    let also = ALSO.get_or_init(|| {
        let list = std::env::var("VICTORY_CAPTURE_WORLD_ALSO").unwrap_or_default();
        list.split(',')
            .filter_map(|f| f.trim().parse().ok())
            .collect()
    });
    let first = WORLD.load(std::sync::atomic::Ordering::Relaxed);
    let after = n.wrapping_sub(first);
    named || first != u64::MAX && ([2, 6, 30, 80].contains(&after) || also.contains(&after)) || heavy_listed(n)
}

/// `VICTORY_CAPTURE_HEAVY_FRAMES` (comma-separated): frames counted from
/// the heavy scene's first, whose number differs from build to build. Only
/// with a capture folder: without one nothing is saved, and a listed frame
/// still changed how it was presented (Sol's review, finding 9).
///
/// Offset 0 is the heavy frame itself, known as such only at its present:
/// its image is saved, but what is noted while frames are recorded (their
/// ops, targets and indirect lines) starts at offset 1.
fn heavy_offsets() -> &'static [u64] {
    static OFFSETS: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    OFFSETS.get_or_init(|| {
        if folder().is_none() {
            return Vec::new();
        }
        let list = std::env::var("VICTORY_CAPTURE_HEAVY_FRAMES").unwrap_or_default();
        let offsets: Vec<u64> = list.split(',').filter_map(|f| f.trim().parse().ok()).collect();
        if offsets.contains(&0) {
            tracing::info!(
                "VICTORY_CAPTURE_HEAVY_FRAMES offset 0: the heavy frame is known at its present, so only its image is saved"
            );
        }
        offsets
    })
}

/// Whether title draws and mesh dispatches are counted for the heavy
/// scene's detector (`crate::draw::attempted`): only with
/// [`heavy_offsets`]. Asked on every draw, a load once the list is read.
pub(crate) fn heavy_counting() -> bool {
    #[cfg(test)]
    if COUNTING_IN_TESTS.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    !heavy_offsets().is_empty()
}

/// Tests: counts draws for the heavy scene's detector without the list.
#[cfg(test)]
static COUNTING_IN_TESTS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The heavy scene's first presented frame, once seen.
#[derive(Debug, Default)]
struct Heavy {
    first: Option<u64>,
}

static HEAVY: std::sync::Mutex<Heavy> = std::sync::Mutex::new(Heavy { first: None });

/// The title draws and mesh dispatches a frame records, run or dropped
/// (`crate::draw::attempted`), that make it the heavy scene's first.
const HEAVY_ATTEMPTS: u64 = 1_000;

/// One presented frame against the heavy list: `attempts` is the title
/// draws and mesh dispatches it recorded, whether they ran or were dropped
/// (never compute dispatches: counted only when dropped, they made a
/// compute-heavy frame heavy on one build and not on another, Sol's
/// review), `state` the frames so far. Sets the scene's first frame when
/// this one reaches [`HEAVY_ATTEMPTS`], and returns whether `n` is one of
/// the listed frames after it.
fn heavy_frame(n: u64, attempts: u64, state: &mut Heavy, offsets: &[u64]) -> bool {
    if state.first.is_none() && attempts >= HEAVY_ATTEMPTS {
        state.first = Some(n);
    }
    state.first.is_some_and(|first| offsets.iter().any(|&k| first.saturating_add(k) == n))
}

/// Whether `n` is one of `VICTORY_CAPTURE_HEAVY_FRAMES` after the heavy
/// scene's first frame; before that frame nothing is.
fn heavy_listed(n: u64) -> bool {
    let offsets = heavy_offsets();
    !offsets.is_empty() && heavy_frame(n, 0, &mut HEAVY.lock().unwrap_or_else(|e| e.into_inner()), offsets)
}

/// With a capture folder, for the frames [`listed`]: `what` is added to
/// the frame's `frame-<n>-ops.txt`, one line per draw, dispatch, copy or
/// drop in the order recorded (the title records on several threads, so
/// lines of different lists interleave; each names its list). The file
/// says which pass writes which target, to find the pass whose output
/// stays empty.
#[cfg(feature = "diagnostics")]
pub(crate) fn op(list: usize, what: impl FnOnce() -> String) {
    use std::io::Write;
    let frame = crate::queue::presented() + 1;
    if !listed(frame) {
        // Before the world: the last lines are kept, and written out when
        // it starts (`before-world-ops.txt`): what the title drew just
        // before (the title shows a title card there, whose text
        // did not show).
        if world_on() && WORLD.load(std::sync::atomic::Ordering::Relaxed) == u64::MAX {
            // At most 300 lines of a frame: the world's first frame alone
            // (thousands of operations, recorded before it is known to be
            // the world's) pushed everything before it out.
            static IN_FRAME: std::sync::Mutex<(u64, u32)> = std::sync::Mutex::new((0, 0));
            {
                let mut in_frame = IN_FRAME.lock().unwrap_or_else(|e| e.into_inner());
                if in_frame.0 != frame {
                    *in_frame = (frame, 0);
                }
                in_frame.1 += 1;
                if in_frame.1 > 300 {
                    return;
                }
            }
            let what = what();
            let mut ring = RING.lock().unwrap_or_else(|e| e.into_inner());
            if ring.len() >= RING_LINES {
                ring.pop_front();
            }
            ring.push_back(format!("frame {frame} list {list:#x} {what}"));
        }
        return;
    }
    let Some(folder) = folder() else {
        return;
    };
    static FILE: std::sync::Mutex<Option<(u64, u64, std::fs::File)>> = std::sync::Mutex::new(None);
    let what = what();
    let mut file = FILE.lock().unwrap_or_else(|e| e.into_inner());
    if file.as_ref().is_none_or(|f| f.0 != frame) {
        let Ok(made) = std::fs::File::create(folder.join(format!("frame-{frame}-ops.txt"))) else {
            return;
        };
        *file = Some((frame, 0, made));
    }
    if let Some((_, n, f)) = file.as_mut() {
        *n += 1;
        let _ = writeln!(f, "{n} list {list:#x} {what}");
    }
}

/// The last lines of [`op`] before the world.
#[cfg(feature = "diagnostics")]
static RING: std::sync::Mutex<std::collections::VecDeque<String>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());
#[cfg(feature = "diagnostics")]
const RING_LINES: usize = 6000;

/// Whether `VICTORY_CAPTURE_WORLD` is on ([`crate::switches::switch_on`]).
pub(crate) fn world_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switches::switch_on("VICTORY_CAPTURE_WORLD"))
}

/// The first frame that ran 500 draws or more (`VICTORY_CAPTURE_WORLD`).
static WORLD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

/// At the present of `frame`: the draws run on the host so far.
pub(crate) fn frame_drew(frame: u64, drawn: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static LAST_ATTEMPTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let offsets = heavy_offsets();
    if !offsets.is_empty() {
        let attempted = crate::draw::attempted();
        let attempts = attempted.saturating_sub(LAST_ATTEMPTS.swap(attempted, Relaxed));
        let mut heavy = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
        let before = heavy.first;
        heavy_frame(frame, attempts, &mut heavy, offsets);
        if before.is_none() && heavy.first.is_some() {
            tracing::info!(frame, attempts, "the heavy scene's first frame (VICTORY_CAPTURE_HEAVY_FRAMES counts from it)");
        }
    }
    let before = LAST.swap(drawn, Relaxed);
    if world_on()
        && drawn.saturating_sub(before) >= 500
        && WORLD.compare_exchange(u64::MAX, frame, Relaxed, Relaxed).is_ok()
    {
        tracing::info!(frame, "the first frame of 500 draws or more: frames {} and {} are captured", frame + 2, frame + 6);
        #[cfg(feature = "diagnostics")]
        if let Some(folder) = folder() {
            let lines: Vec<String> = std::mem::take(&mut *RING.lock().unwrap_or_else(|e| e.into_inner())).into();
            let _ = std::fs::write(folder.join("before-world-ops.txt"), lines.join("
"));
        }
    }
}

/// Whether the frame being recorded is listed and a folder is set: then
/// indirect executions' records are read ([`read_indirect`]).
pub(crate) fn recording_listed() -> bool {
    folder().is_some() && listed(crate::queue::presented() + 1)
}

/// The most bytes of records an indirect line reads back: a title's
/// executions hold a few thousand records of a few dozen bytes each.
const RECORDS_MOST: u64 = 1 << 22;

/// The bytes of records an indirect line reads for `(kind, stride, max)`
/// at `at` of a buffer `width` bytes long: `None` unless the clamp takes
/// the layout (`crate::indirect`'s kinds and strides) and the records lie
/// inside the buffer. Read as given, stride 1 with max 4 mapped two words
/// and showed a third (Sol's review, finding 2).
fn records_span((kind, stride, max): (u32, u32, u32), (at, width): (u64, u64)) -> Option<u64> {
    let size = match kind {
        0 => 16,
        1 => 20,
        2 | 10 => 12,
        _ => return None,
    };
    if stride < size || !stride.is_multiple_of(4) {
        return None;
    }
    let len = u64::from(stride).checked_mul(u64::from(max.max(1)))?;
    (len <= RECORDS_MOST && at.checked_add(len)? <= width).then_some(len)
}

/// The work `records` of `kind` ask for, in `words` (`stride` bytes
/// apart): how many ask for some, and how much in all. Draws: vertices
/// (or indices) times instances; dispatches and mesh dispatches: groups.
/// In 128 bits: three group counts of 2^32 - 1 multiply past 64.
fn records_with_work(words: &[u32], kind: u32, stride: u32, records: usize) -> (usize, u128) {
    let step = (stride as usize / 4).max(1);
    let (mut busy, mut total) = (0, 0u128);
    for r in 0..records {
        let w = r.checked_mul(step).and_then(|at| words.get(at..)).unwrap_or(&[]);
        let word = |i: usize, or: u32| u128::from(w.get(i).copied().unwrap_or(or));
        let work = match kind {
            0 | 1 => word(0, 0) * word(1, 1),
            _ if w.is_empty() => 0,
            _ => word(0, 1) * word(1, 1) * word(2, 1),
        };
        busy += usize::from(work > 0);
        total = total.saturating_add(work);
    }
    (busy, total)
}

/// The line for `words` read back for an indirect execution of `(kind,
/// stride, max)` without a count: its records. Reads only what `words`
/// holds.
fn indirect_line(words: &[u32], (kind, stride, max): (u32, u32, u32), name: &str) -> String {
    let step = (stride as usize / 4).max(1);
    let records = max as usize;
    let (busy, total) = records_with_work(words, kind, stride, records);
    let shown: Vec<&[u32]> = (0..records.min(4))
        .filter_map(|r| words.get(r * step..).map(|w| &w[..w.len().min(step.min(4))]))
        .collect();
    format!("indirect kind {kind} max {max} count none: {busy} of {records} records with work, {total} in all; first {shown:?} {name}")
}

/// Why an indirect execution's line has no records.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Unread {
    /// They, or the count that says how many ran, are in a title buffer:
    /// one that did not run (`dropped`) has its records there, and one with
    /// a count has it there. A title buffer's state in the list is not
    /// tracked: a shader may have left it in the unordered access state,
    /// from which no copy is valid (Sol's review, finding 1).
    TitleBuffer,
    /// They are these many bytes, more than a line reads ([`RECORDS_MOST`];
    /// Sol's re-check, new fault 7).
    Size(u64),
    /// The list holds as many read-backs for lines as it may
    /// (`HostList::read_budget`; Sol's re-check, finding 6).
    Spent,
}

impl Unread {
    fn text(self) -> String {
        match self {
            Unread::TitleBuffer => "a title buffer's state is not tracked".to_owned(),
            Unread::Size(bytes) => format!("{bytes} bytes of records, more than the {RECORDS_MOST} a line reads"),
            Unread::Spent => format!(
                "the list's read-backs are spent: {} lines or {} bytes",
                crate::host::LIST_READS_MOST,
                crate::host::LIST_READ_BYTES_MOST
            ),
        }
    }
}

/// The line for an indirect execution whose records are not read, and
/// `why`. A count the title gives is reported as unavailable, never as
/// absent (finding 11).
pub(crate) fn unread_line((kind, _, max): (u32, u32, u32), counted: bool, dropped: bool, why: Unread, name: &str) -> String {
    let count = if counted { "unavailable" } else { "none" };
    let dropped = if dropped { "DROPPED " } else { "" };
    format!("indirect kind {kind} max {max} count {count}: records not read ({}); {dropped}{name}", why.text())
}

/// Logs an indirect execution's line (tests also keep it, [`LINES`], and
/// those logged off the reporting thread, [`OFF_REPORTER`]).
pub(crate) fn log_indirect(line: String) {
    tracing::info!(target: "victory_gpu::trace", "{line}");
    #[cfg(test)]
    if std::thread::current().name() != Some("victory-reports") {
        OFF_REPORTER.lock().unwrap_or_else(|e| e.into_inner()).push(line.clone());
    }
    #[cfg(test)]
    LINES.lock().unwrap_or_else(|e| e.into_inner()).push(line);
}

/// Logs an indirect execution's line made where it is recorded, on the
/// reporting thread ([`log_indirect`] there): recording holds the list's
/// state, which a log that blocks would hold too (Sol's second re-check,
/// new fault 6). Dropped and counted when the reporter is behind.
pub(crate) fn report_indirect(line: String) {
    crate::host::report(Box::new(move || log_indirect(line)));
}

/// Tests: the lines [`log_indirect`] logged on a thread other than the
/// reporting thread.
#[cfg(test)]
static OFF_REPORTER: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Tests: the lines [`read_indirect`] made.
#[cfg(test)]
static LINES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// In a listed frame: copies the records an indirect execution without a
/// count ran, as the argument clamp left them in its scratch buffer (at
/// `at`, in the indirect-argument state, `crate::indirect::bound`), to
/// read-back memory, logged once the list has run, as our line logs it:
/// "indirect kind K max M count none: B of R records with work, W in all;
/// first [...] <pipeline>". An execution whose records ask for nothing
/// draws nothing, which a dropped one does for another reason; the line
/// tells the two apart. Nothing is recorded for a layout [`records_span`]
/// refuses. Executions with a count, and dropped ones, have
/// [`unread_line`]: the clamp writes no record past the count, which is
/// in a title buffer.
pub(crate) fn read_indirect(
    list: &crate::host::HostList,
    scratch: &ID3D12Resource,
    at: u64,
    spec: (u32, u32, u32),
    name: String,
) {
    struct Readback(ID3D12Resource);
    // SAFETY: D3D12 objects are free-threaded.
    unsafe impl Send for Readback {}
    // Too many to read: the line says so, and nothing is copied.
    let (_, stride, max) = spec;
    let bytes = u64::from(stride) * u64::from(max.max(1));
    if bytes > RECORDS_MOST {
        report_indirect(unread_line(spec, false, false, Unread::Size(bytes), &name));
        return;
    }
    let Some(host) = crate::host::host() else { return };
    // SAFETY: a live buffer.
    let width = unsafe { scratch.GetDesc() }.Width;
    let Some(len) = records_span(spec, (at, width)) else {
        return;
    };
    if !list.read_budget(len) {
        report_indirect(unread_line(spec, false, false, Unread::Spent, &name));
        return;
    }
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: len,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let mut buffer: Option<ID3D12Resource> = None;
    let l = &list.list;
    // SAFETY: plain creation; an open list; the range is inside the buffer
    // (`records_span`). The clamp's scratch buffer, ours, goes from the
    // indirect-argument state to copy source and back.
    unsafe {
        if host
            .device
            .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buffer)
            .is_err()
        {
            return;
        }
        let Some(buffer) = buffer else { return };
        let (args, copy) = (D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
        l.ResourceBarrier(&[crate::host::transition(scratch, args, copy)]);
        l.CopyBufferRegion(&buffer, 0, scratch, at, len);
        l.ResourceBarrier(&[crate::host::transition(scratch, copy, args)]);
        list.keep(&buffer);
        let buffer = Readback(buffer);
        // Each time the list has run: the records are copied out of the
        // mapping there, and the line is made of the copy on the reporting
        // thread.
        list.defer_after(move || {
            let mut data = std::ptr::null_mut();
            if buffer.0.Map(0, None, Some(&mut data)).is_err() || data.is_null() {
                return None;
            }
            // The buffer's own `len` bytes, a multiple of 4.
            let words = std::slice::from_raw_parts(data.cast::<u32>(), len as usize / 4).to_vec();
            buffer.0.Unmap(0, None);
            let name = name.clone();
            Some(Box::new(move || log_indirect(indirect_line(&words, spec, &name))))
        });
    }
}

/// Whether frame `n` is saved.
pub(crate) fn wanted(n: u64) -> bool {
    // Two frames in a row each time: a guest may draw to two buffers in
    // turn, and a lone 8th or 16th frame would show one of them only.
    // Before the world, with `VICTORY_CAPTURE_WORLD` and
    // `VICTORY_CAPTURE_MENUS=1`: every 16th frame too (small pictures of
    // menus, loading screens and title cards). Asked for apart: saving them
    // makes the menus and their music stutter.
    static MENUS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let menus = *MENUS.get_or_init(|| std::env::var_os("VICTORY_CAPTURE_MENUS").is_some_and(|v| v == "1"));
    let before_world = menus && world_on() && n % 16 < 2 && WORLD.load(std::sync::atomic::Ordering::Relaxed) == u64::MAX;
    // In the world: every 8th frame's picture, a timeline to compare runs
    // by (what is on screen does not fall on the same frame numbers from run
    // to run, so single frames cannot be compared between runs).
    let first = WORLD.load(std::sync::atomic::Ordering::Relaxed);
    let in_world = world_on() && first != u64::MAX && n > first && (n - first) % 8 < 2;
    folder().is_some() && (n.is_power_of_two() || listed(n) || before_world || in_world)
}

/// Saves 8-bit RGBA or BGRA rows (`pitch` bytes apart) as frame `n`.
pub(crate) fn save(n: u64, width: u32, height: u32, pitch: usize, bgra: bool, rows: &[u8]) {
    save_as(&format!("frame-{n}"), width, height, pitch, bgra, rows);
}

/// Saves 8-bit RGBA or BGRA rows as `<name>.png`, at half size.
fn save_as(name: &str, width: u32, height: u32, pitch: usize, bgra: bool, rows: &[u8]) {
    let Some(folder) = folder() else { return };
    let (w, h) = ((width / 2).max(1), (height / 2).max(1));
    // Filter byte 0 per row, then RGB of every other pixel of every other row.
    let mut raw = Vec::with_capacity((w as usize * 3 + 1) * h as usize);
    for y in 0..h as usize {
        raw.push(0);
        let row = &rows[(2 * y).min(height as usize - 1) * pitch..];
        for x in 0..w as usize {
            let p = &row[(2 * x).min(width as usize - 1) * 4..][..3];
            if bgra {
                raw.extend([p[2], p[1], p[0]]);
            } else {
                raw.extend_from_slice(p);
            }
        }
    }
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::new();
    header.extend(w.to_be_bytes());
    header.extend(h.to_be_bytes());
    header.extend([8, 2, 0, 0, 0]); // 8-bit RGB
    chunk(&mut png, b"IHDR", &header);
    let mut compressed = vec![0u8; zlib_rs::compress_bound(raw.len())];
    let (idat, rc) = zlib_rs::compress_slice(&mut compressed, &raw, zlib_rs::DeflateConfig::new(3));
    debug_assert_eq!(rc, zlib_rs::ReturnCode::Ok);
    chunk(&mut png, b"IDAT", idat);
    chunk(&mut png, b"IEND", &[]);
    let path = folder.join(format!("{name}.png"));
    match std::fs::write(&path, png) {
        Ok(()) => tracing::info!(path = %path.display(), "frame saved"),
        Err(e) => tracing::warn!(error = %e, path = %path.display(), "frame not saved"),
    }
}

fn chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend((data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend(kind);
    png.extend(data);
    let crc = crc32(&png[start..]);
    png.extend(crc.to_be_bytes());
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

/// The bytes a mapped texture read-back is read as: what the copy wrote,
/// `rows` rows `RowPitch` apart, the last without its padding
/// (`row_bytes`). That is GetCopyableFootprints' total, the buffer's size.
/// Before, `RowPitch * Height`: 256 bytes for a 1x1 RGBA8 target whose
/// copy is 4, and for block-compressed formats rows are fewer than texels
/// high (Sol's review, finding 3).
fn mapped_len(footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, rows: u32, row_bytes: u64) -> usize {
    match rows.checked_sub(1) {
        Some(before_last) => before_last as usize * footprint.Footprint.RowPitch as usize + row_bytes as usize,
        None => 0,
    }
}

/// A subresource copied into a read-back buffer ([`read_back`]): its
/// footprint, and its rows and the unpadded bytes of one, as
/// GetCopyableFootprints gave them.
pub(crate) struct ReadBack {
    pub(crate) buffer: ID3D12Resource,
    pub(crate) footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    rows: u32,
    row_bytes: u64,
}

impl ReadBack {
    /// Whether the footprint has a row for each texel row: not for
    /// block-compressed formats, whose rows are rows of blocks.
    fn rows_are_texels(&self) -> bool {
        self.rows == self.footprint.Footprint.Height
    }

    /// `read` of the bytes the copy wrote, mapped: the copy has finished.
    fn map<R>(&self, read: impl FnOnce(&[u8]) -> R) -> Option<R> {
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer of ours whose copy has run; the slice
        // is the copy's bytes, which the buffer holds (`mapped_len`).
        unsafe {
            self.buffer.Map(0, None, Some(&mut data)).ok()?;
            let read = (!data.is_null()).then(|| {
                let len = mapped_len(&self.footprint, self.rows, self.row_bytes);
                read(std::slice::from_raw_parts(data.cast::<u8>(), len))
            });
            self.buffer.Unmap(0, None);
            read
        }
    }
}

/// Records a copy of `texture` (in the copy-source state) into a
/// readback buffer.
pub(crate) fn read_back(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<ReadBack> {
    read_back_of(device, list, texture, 0)
}

/// As [`read_back`], of subresource `sub` (the stencil plane of a
/// one-level, one-slice depth-stencil texture is subresource 1).
fn read_back_of(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
    sub: u32,
) -> Option<ReadBack> {
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let (mut rows, mut row_bytes, mut total) = (0, 0, 0);
    // SAFETY: a live texture; outputs for one subresource.
    unsafe {
        let desc = texture.GetDesc();
        device.GetCopyableFootprints(
            &desc,
            sub,
            1,
            0,
            Some(&mut footprint),
            Some(&mut rows),
            Some(&mut row_bytes),
            Some(&mut total),
        );
    }
    // A subresource the call does not describe (it reports u64::MAX).
    if total == u64::MAX || mapped_len(&footprint, rows, row_bytes) as u64 != total {
        return None;
    }
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; an open list; locations naming live resources.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
        let buffer = buffer?;
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(texture),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: sub,
            },
        };
        list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        Some(ReadBack { buffer, footprint, rows, row_bytes })
    }
}

/// Saves a frame read back by [`read_back`] (8-bit color only).
pub(crate) fn save_read_back(frame: u64, read: &ReadBack) {
    let f = read.footprint.Footprint;
    let bgra = match f.Format {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => false,
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => true,
        other => {
            tracing::warn!(format = other.0, "frame not saved: not 8-bit color");
            return;
        }
    };
    // The copy finished: every submission that saves a frame is waited for.
    read.map(|rows| save(frame, f.Width, f.Height, f.RowPitch as usize, bgra, rows));
}

/// With `VICTORY_DUMP_TARGETS` set: title textures the frames listed in
/// `VICTORY_CAPTURE_FRAMES` render to or write in lists submitted before
/// the frame's present ([`targets_submitted`]), in first-use order, saved
/// at that frame's present as `frame-<n>-<i>-<resource>-<w>x<h>-f<format>.png`
/// (what each holds at the end of the frame; formats normalized to show
/// structure: floats scaled by their largest value, depth by its range).
/// The host copies are held: the title may release a target before the
/// frame is presented.
/// Each with the frame it was noted for: a frame's dump takes its own
/// only (Sol's re-check, finding 9: the untagged list could mix frames).
#[allow(clippy::type_complexity)]
static TARGETS: std::sync::Mutex<Vec<(u64, usize, std::sync::Arc<crate::resource::HostTexture>)>> =
    std::sync::Mutex::new(Vec::new());

/// Whether the end-of-frame target dump runs: asked for (a capture folder
/// and `VICTORY_DUMP_TARGETS`) and allowed ([`dump_allowed`]). Refused, it
/// says so once, naming the switch.
pub(crate) fn dump_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let asked = folder().is_some() && crate::switches::switch_on("VICTORY_DUMP_TARGETS");
        let allowed = dump_allowed(asked, std::env::var("VICTORY_DUMP_UNSAFE").ok().as_deref());
        if asked && !allowed {
            tracing::warn!(
                "VICTORY_DUMP_TARGETS refused: the target dump can race a present's change of a texture's state \
                 and record an invalid barrier; VICTORY_DUMP_UNSAFE=1 runs it anyway"
            );
        }
        allowed
    })
}

/// Whether target dumps run: `asked` (a capture folder and
/// `VICTORY_DUMP_TARGETS`) and `VICTORY_DUMP_UNSAFE` set to `unsafe_switch`,
/// which must be 1. The dump is the fork's, which the heavy frames reach:
/// a present can publish a texture's copy-source state before its
/// transition is queued, and the dump then leaves a barrier out (Sol's
/// second re-check, finding 9). Without the switch the landing never
/// issues it; the harness sets it when it asks for target dumps.
fn dump_allowed(asked: bool, unsafe_switch: Option<&str>) -> bool {
    asked && unsafe_switch.is_some_and(|s| s.trim() == "1")
}

/// A title texture `list` renders to or writes in the frame being built:
/// noted in the list, and one of the frame's targets once the list is
/// submitted ([`targets_submitted`]). A list recorded and never submitted
/// adds none: its writes never ran (Sol's review, finding 9).
pub(crate) fn note_target(list: &crate::host::HostList, resource: crate::object::This) {
    let frame = crate::queue::presented() + 1;
    if resource.is_null() || !dump_on() || !listed(frame) {
        return;
    }
    if let Some(copy) = crate::resource::host_copy(resource) {
        list.note_target(frame, resource as usize, copy);
    }
}

/// At a list's submission: the targets it noted ([`note_target`]) join
/// their frame's, if that frame has not been presented (a list recorded
/// before a present and submitted after it is too late for that frame,
/// and adds nothing to the next).
pub(crate) fn targets_submitted(noted: Vec<(u64, usize, std::sync::Arc<crate::resource::HostTexture>)>) {
    if noted.is_empty() {
        return;
    }
    let next = crate::queue::presented() + 1;
    admit(&mut TARGETS.lock().unwrap_or_else(|e| e.into_inner()), noted, next);
}

/// Adds to `targets` those of `noted` noted for frame `next` (the one being
/// built) or later, each resource once a frame, in first-use order, with
/// the frame they were noted for.
fn admit<T>(targets: &mut Vec<(u64, usize, T)>, noted: Vec<(u64, usize, T)>, next: u64) {
    for (noted_for, resource, copy) in noted {
        if noted_for >= next && !targets.iter().any(|t| t.0 == noted_for && t.1 == resource) {
            targets.push((noted_for, resource, copy));
        }
    }
}

/// Takes from `targets` frame `frame`'s, in order; those of earlier frames
/// go, and those of later ones stay for their frames.
fn take_frame<T>(targets: &mut Vec<(u64, usize, T)>, frame: u64) -> Vec<(usize, T)> {
    let (mine, later): (Vec<_>, Vec<_>) = std::mem::take(targets).into_iter().filter(|t| t.0 >= frame).partition(|t| t.0 == frame);
    *targets = later;
    mine.into_iter().map(|(_, resource, copy)| (resource, copy)).collect()
}

/// At the present of frame `frame`: saves the noted targets.
pub(crate) fn dump_targets(frame: u64) {
    if !dump_on() {
        return;
    }
    let targets = take_frame(&mut TARGETS.lock().unwrap_or_else(|e| e.into_inner()), frame);
    let Some(host) = crate::host::host() else { return };
    for (i, (resource, copy)) in targets.iter().enumerate() {
        // SAFETY: a live resource.
        let d = unsafe { copy.texture.GetDesc() };
        if d.SampleDesc.Count > 1 || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
            continue;
        }
        dump_one(host, copy, frame, i, *resource as crate::object::This, &d);
    }
}

/// Copies host texture `copy` out and saves it. Its tracked state is read,
/// and the barriers to the copy-source state and back are made from it,
/// under the submission lock, where the list also runs
/// (`Host::run_own_locked`): no submission's prelude moves the texture in
/// between. Read before the lock, a submission in between made the
/// dump's barrier invalid (Sol's review and re-check, finding 9).
fn dump_one(
    host: &crate::host::Host,
    copy: &std::sync::Arc<crate::resource::HostTexture>,
    frame: u64,
    i: usize,
    resource: crate::object::This,
    d: &D3D12_RESOURCE_DESC,
) {
    let texture = &copy.texture;
    let (mut read, mut stencil) = (None, None);
    let ran = host.run_own_locked(|list| {
        let state = copy.state.load(std::sync::atomic::Ordering::Relaxed);
        let state = D3D12_RESOURCE_STATES(crate::host::legal_state(texture, state) as i32);
        let copy_source = D3D12_RESOURCE_STATE_COPY_SOURCE;
        // SAFETY: an open list of ours and a live texture.
        unsafe {
            if state != copy_source {
                list.list.ResourceBarrier(&[crate::host::transition(texture, state, copy_source)]);
            }
            read = read_back(&host.device, &list.list, texture);
            // The stencil plane of a depth-stencil target, which the title's
            // lighting reads (material classes and flags): the plane after
            // the depth plane of a one-level, one-slice texture.
            stencil = (matches!(d.Format.0, 19..=22 | 44..=47) && d.MipLevels == 1 && d.DepthOrArraySize == 1)
                .then(|| read_back_of(&host.device, &list.list, texture, 1))
                .flatten();
            if state != copy_source {
                list.list.ResourceBarrier(&[crate::host::transition(texture, copy_source, state)]);
            }
        }
    });
    if !ran {
        return;
    }
    if let Some(stencil) = stencil {
        stencil_values(&stencil, frame, resource);
    }
    let Some(read) = read else { return };
    let f = read.footprint.Footprint;
    // Formats are turned away before the mapping: those `to_rgba` does not
    // convert, and footprints whose rows are not texel rows (block formats).
    let depth24 = matches!(d.Format.0, 44..=47);
    if !read.rows_are_texels() || (rgba_texel_bytes(f.Format.0 as u32).is_none() && !depth24) {
        tracing::info!(format = f.Format.0, ?resource, "target not dumped: format");
        return;
    }
    let pitch = f.RowPitch as usize;
    // The copy finished: the dump's list was waited for.
    read.map(|rows| {
        // A 24-bit depth plane comes back as 32-bit words: which depth values
        // (low 24 bits) and which top bytes it holds, the most common first.
        if depth24 {
            let mut depths = std::collections::HashMap::<u32, u32>::new();
            let mut tops = std::collections::HashMap::<u32, u32>::new();
            for y in 0..f.Height as usize {
                for x in 0..f.Width as usize {
                    let at = y * pitch + x * 4;
                    let Some(w) = rows.get(at..at + 4) else { continue };
                    let w = u32::from_le_bytes(w.try_into().unwrap_or([0; 4]));
                    *depths.entry(w & 0xff_ffff).or_default() += 1;
                    *tops.entry(w >> 24).or_default() += 1;
                }
            }
            let most = |m: std::collections::HashMap<u32, u32>| {
                let mut v: Vec<(u32, u32)> = m.into_iter().collect();
                v.sort_by_key(|e| std::cmp::Reverse(e.1));
                (v.len(), v.into_iter().take(12).collect::<Vec<_>>())
            };
            tracing::info!(?resource, depths = ?most(depths), top_bytes = ?most(tops), "24-bit depth plane values (value, texels)");
        }
        if let Some(rgba) = to_rgba(f.Format.0 as u32, f.Width as usize, f.Height as usize, pitch, rows) {
            let name = format!(
                "frame-{frame}-{i:02}-{resource:?}-{}x{}-f{}",
                d.Width, d.Height, d.Format.0
            );
            save_as(&name, f.Width, f.Height, f.Width as usize * 4, false, &rgba);
        } else {
            tracing::info!(format = f.Format.0, ?resource, "target not dumped: format");
        }
    });
}

/// Logs which values a stencil plane read back by [`read_back_of`] holds
/// (the most common first), how many texels have each of its bits, and the
/// box of the texels with bit 3 (0x08).
fn stencil_values(read: &ReadBack, frame: u64, resource: crate::object::This) {
    let f = read.footprint.Footprint;
    let (w, h, pitch) = (f.Width as usize, f.Height as usize, f.RowPitch as usize);
    // One byte a texel, a row for each texel row.
    if h == 0 || w == 0 || !read.rows_are_texels() || read.row_bytes != w as u64 {
        return;
    }
    // The copy finished: the dump's list was waited for.
    read.map(|rows| {
        let mut values = [0u32; 256];
        let mut bits = [0u32; 8];
        // Left, top, right, bottom of the texels with bit 3.
        let mut inside = [u32::MAX, u32::MAX, 0, 0];
        for y in 0..h {
            for (x, &v) in rows[y * pitch..y * pitch + w].iter().enumerate() {
                values[usize::from(v)] += 1;
                for (bit, count) in bits.iter_mut().enumerate() {
                    *count += u32::from(v >> bit & 1);
                }
                if v & 8 != 0 {
                    inside = [
                        inside[0].min(x as u32),
                        inside[1].min(y as u32),
                        inside[2].max(x as u32),
                        inside[3].max(y as u32),
                    ];
                }
            }
        }
        let mut most: Vec<(usize, u32)> = values
            .iter()
            .copied()
            .enumerate()
            .filter(|v| v.1 != 0)
            .collect();
        most.sort_by_key(|v| std::cmp::Reverse(v.1));
        most.truncate(16);
        tracing::info!(
            frame,
            ?resource,
            width = w,
            height = h,
            values = ?most,
            texels_with_bit = ?bits,
            bit3_box = ?inside,
            "stencil plane values (value, texels)"
        );
    });
}

fn half(h: u16) -> f32 {
    let (sign, exp, man) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f32);
    let v = match exp {
        0 => man * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    };
    if sign == 1 { -v } else { v }
}

/// A float packed in `bits` with a 5-bit exponent and `man` mantissa bits
/// (the 11- and 10-bit floats of R11G11B10).
fn small_float(bits: u32, man: u32) -> f32 {
    let (exp, m) = ((bits >> man) & 0x1f, (bits & ((1 << man) - 1)) as f32);
    match exp {
        0 => m / (1 << man) as f32 * 2f32.powi(-14),
        31 => f32::INFINITY,
        e => (1.0 + m / (1 << man) as f32) * 2f32.powi(e as i32 - 15),
    }
}

/// The bytes of a texel of the formats [`to_rgba`] converts; `None` for the
/// others (block-compressed formats among them).
fn rgba_texel_bytes(format: u32) -> Option<usize> {
    Some(match format {
        1..=2 => 16,
        9..=16 | 19..=22 => 8,
        23..=47 | 87..=93 => 4,
        48..=59 => 2,
        60..=65 => 1,
        _ => return None,
    })
}

/// Rows of a host format as 8-bit RGBA: 8-bit color as is, everything else
/// as channel values scaled so the largest finite one is white (single
/// channels: lowest black, highest white, as grey). `None` for formats not
/// handled, and for `rows` shorter than `h` rows of `w` texels.
fn to_rgba(format: u32, w: usize, h: usize, pitch: usize, rows: &[u8]) -> Option<Vec<u8>> {
    let n = rgba_texel_bytes(format)?;
    let end = h.checked_sub(1).map_or(Some(0), |last| last.checked_mul(pitch)?.checked_add(w.checked_mul(n)?))?;
    if end > rows.len() {
        return None;
    }
    let mut out = Vec::with_capacity(w * h * 4);
    let px = |x: usize, y: usize, n: usize| &rows[y * pitch + x * n..][..n];
    let u16_at = |b: &[u8], i: usize| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
    let u32_at = |b: &[u8], i: usize| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap_or([0; 4]));
    // Channel values as floats, then scaled.
    type Decode = Box<dyn Fn(&[u8]) -> [f32; 3]>;
    let (n, decode): (usize, Decode) = match format {
        27..=32 => {
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y, 4);
                    out.extend([p[0], p[1], p[2], 255]);
                }
            }
            return Some(out);
        }
        // B8G8R8A8 and B8G8R8X8 (89 is R10G10B10_XR_BIAS_A2, below).
        87 | 88 | 90..=93 => {
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y, 4);
                    out.extend([p[2], p[1], p[0], 255]);
                }
            }
            return Some(out);
        }
        23..=25 | 89 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            [(v & 0x3ff) as f32, (v >> 10 & 0x3ff) as f32, (v >> 20 & 0x3ff) as f32]
        })),
        26 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            [small_float(v & 0x7ff, 6), small_float(v >> 11 & 0x7ff, 6), small_float(v >> 22, 5)]
        })),
        9..=10 => (8, Box::new(move |b| [half(u16_at(b, 0)), half(u16_at(b, 1)), half(u16_at(b, 2))])),
        11..=14 => (8, Box::new(move |b| [0, 1, 2].map(|i| f32::from(u16_at(b, i))))),
        33..=34 => (4, Box::new(move |b| [half(u16_at(b, 0)), half(u16_at(b, 1)), 0.0])),
        1..=2 => (16, Box::new(move |b| {
            [0, 1, 2].map(|i| f32::from_bits(u32_at(b, i)))
        })),
        15..=16 => (8, Box::new(move |b| [f32::from_bits(u32_at(b, 0)), f32::from_bits(u32_at(b, 1)), 0.0])),
        35..=38 => (4, Box::new(move |b| [f32::from(u16_at(b, 0)), f32::from(u16_at(b, 1)), 0.0])),
        48..=52 => (2, Box::new(move |b| [f32::from(b[0]), f32::from(b[1]), 0.0])),
        // Single channels (and depth planes): shown as grey.
        39..=43 => (4, Box::new(move |b| [f32::from_bits(u32_at(b, 0)); 3])),
        19..=22 => (8, Box::new(move |b| [f32::from_bits(u32_at(b, 0)); 3])),
        44..=47 => (4, Box::new(move |b| [(u32_at(b, 0) & 0xff_ffff) as f32; 3])),
        53..=54 => (2, Box::new(move |b| [half(u16_at(b, 0)); 3])),
        55..=59 => (2, Box::new(move |b| [f32::from(u16_at(b, 0)); 3])),
        60..=65 => (1, Box::new(move |b| [f32::from(b[0]); 3])),
        _ => return None,
    };
    let values: Vec<[f32; 3]> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| decode(px(x, y, n)))
        .collect();
    let grey = matches!(format, 19..=22 | 39..=47 | 53..=65);
    let finite = values.iter().flatten().copied().filter(|v| v.is_finite());
    let (low, high) = if grey {
        finite.fold((f32::MAX, f32::MIN), |(l, h), v| (l.min(v), h.max(v)))
    } else {
        (0.0, finite.fold(0.0f32, f32::max))
    };
    let range = (high - low).max(1e-20);
    // How much of the target holds anything: a lit scene and one bright
    // lamp on black have the same highest value.
    let above = values.iter().filter(|v| v.iter().any(|c| c.is_finite() && *c > low)).count();
    let nonzero_percent = above as f32 * 100.0 / values.len().max(1) as f32;
    tracing::info!(format, low, high, nonzero_percent, "target values");
    // Small float colour targets:
    // every texel's four channels as numbers. The picture has no alpha and
    // eight bits a channel.
    if matches!(format, 1..=2 | 9..=10) && w * h <= 16 {
        for y in 0..h {
            for x in 0..w {
                let b = px(x, y, n);
                let texel: [f32; 4] = if n == 16 {
                    [0, 1, 2, 3].map(|i| f32::from_bits(u32_at(b, i)))
                } else {
                    [0, 1, 2, 3].map(|i| half(u16_at(b, i)))
                };
                tracing::info!(format, x, y, ?texel, "target texel");
            }
        }
    }
    // An R32_UINT target's words have no one reading, and as floats their
    // low, high and share above the lowest say nothing. Optional readings
    // instead: how many texels hold any bit, the highest channel value if
    // each word is read as R9G9B9E5 (three 9-bit channels, a shared 5-bit
    // exponent), and the words as plain numbers.
    if format == 42 {
        let (mut set, mut packed_high) = (0usize, 0f32);
        for y in 0..h {
            for x in 0..w {
                let v = u32_at(px(x, y, 4), 0);
                set += usize::from(v != 0);
                let scale = 2f32.powi((v >> 27) as i32 - 24);
                for k in 0..3 {
                    packed_high = packed_high.max((v >> (9 * k) & 0x1ff) as f32 * scale);
                }
            }
        }
        let nonzero_bits_percent = set as f32 * 100.0 / (w * h).max(1) as f32;
        tracing::info!(format, nonzero_bits_percent, packed_high, "target values as R9G9B9E5");
        // The same words as plain numbers: a histogram by bit length
        // ([`words_by_bit_length`]) and the highest word.
        let words = || (0..h).flat_map(move |y| (0..w).map(move |x| u32_at(px(x, y, 4), 0)));
        let by_bit_length = words_by_bit_length(words());
        let highest = words().max().unwrap_or(0);
        tracing::info!(format, width = w, height = h, ?by_bit_length, highest, "target words by bit length");
    }
    // Float colours go through a cube root. Scaled linearly to the highest
    // value, a texel at one percent of it lands on level 2 of 255, near
    // black; its cube root lands on level 54, so dim detail stays visible
    // beside a few bright texels.
    let curved = matches!(format, 1..=2 | 9..=10 | 26);
    for v in values {
        let c = v.map(|c| {
            let t = ((c - low) / range).clamp(0.0, 1.0);
            ((if curved { t.cbrt() } else { t }) * 255.0) as u8
        });
        out.extend([c[0], c[1], c[2], 255]);
    }
    Some(out)
}

/// How many of `words` need each number of bits: entry 0 counts the zeros,
/// entry `k` the words from `2^(k-1)` to `2^k - 1`. Buckets are powers of
/// two alone; no value is counted apart from its bucket.
fn words_by_bit_length(words: impl IntoIterator<Item = u32>) -> [usize; 33] {
    let mut counts = [0usize; 33];
    for w in words {
        counts[(u32::BITS - w.leading_zeros()) as usize] += 1;
    }
    counts
}

/// Pixels per side of the grid [`probe`] reads.
const GRID: u32 = 4;
/// Bytes between the grid's pixels in the readback buffer (placement
/// alignment).
const SLOT: u64 = 512;

/// Records copies of a grid of pixels of `texture` (8-bit color, in the
/// copy-source state) into a readback buffer, to tell black frames.
pub(crate) fn probe(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<ID3D12Resource> {
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    if !matches!(d.Format.0, 27..=29 | 87 | 90 | 91) {
        return None;
    }
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: SLOT * u64::from(GRID * GRID),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; an open list; locations naming live resources.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
        let buffer = buffer?;
        for i in 0..GRID * GRID {
            let (x, y) = (
                (2 * (i % GRID) + 1) * d.Width as u32 / (2 * GRID),
                (2 * (i / GRID) + 1) * d.Height / (2 * GRID),
            );
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&buffer),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: SLOT * u64::from(i),
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: d.Format,
                            Width: 1,
                            Height: 1,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let pixel = D3D12_BOX {
                left: x,
                top: y,
                front: 0,
                right: x + 1,
                bottom: y + 1,
                back: 1,
            };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, Some(&pixel));
        }
        Some(buffer)
    }
}

/// Whether every pixel [`probe`] read is (near) black.
pub(crate) fn black(buffer: &ID3D12Resource) -> bool {
    let mut data = std::ptr::null_mut();
    // SAFETY: the copies finished (every submission is waited for).
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return false;
        }
        let bytes =
            std::slice::from_raw_parts(data.cast::<u8>(), (SLOT * u64::from(GRID * GRID)) as usize);
        let level = bytes
            .chunks(SLOT as usize)
            .flat_map(|pixel| pixel[..3].iter().copied())
            .max()
            .unwrap_or(0);
        buffer.Unmap(0, None);
        let mut levels = LEVELS.lock().unwrap_or_else(|e| e.into_inner());
        if levels.len() == 12 {
            levels.pop_front();
        }
        levels.push_back(level);
        level < 8
    }
}

/// The brightest probed channel of the last frames, oldest first: a fade
/// darkens over several, a glitch drops at once.
static LEVELS: std::sync::Mutex<std::collections::VecDeque<u8>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());

pub(crate) fn recent_levels() -> Vec<u8> {
    let levels = LEVELS.lock().unwrap_or_else(|e| e.into_inner());
    levels.iter().copied().collect()
}

#[cfg(test)]
mod tests {
    /// The records an indirect execution logs as having work: a draw's
    /// vertices times instances, a dispatch's or mesh dispatch's groups,
    /// only as many records as the count allows.
    #[test]
    fn records_with_work_count_what_each_kind_asks_for() {
        // Draws (vertices, instances, first vertex, first instance), 16 bytes.
        let draws = [3, 2, 0, 0, 0, 5, 0, 0, 4, 1, 0, 0];
        assert_eq!(super::records_with_work(&draws, 0, 16, 3), (2, 10), "6 and 4, and one of no vertices");
        assert_eq!(super::records_with_work(&draws, 0, 16, 1), (1, 6), "a count of one");
        // Mesh dispatches (x, y, z), 12 bytes: an empty one and two of work.
        let meshes = [2, 1, 1, 0, 1, 1, 8, 2, 1];
        assert_eq!(super::records_with_work(&meshes, 10, 12, 3), (2, 18));
        assert_eq!(super::records_with_work(&meshes, 10, 12, 0), (0, 0), "a count of zero");
    }

    /// Raw records may ask for anything: three group counts of 2^32 - 1
    /// multiply past 64 bits, and so does a sum of large draws. The totals
    /// neither wrap nor panic (Sol's review, finding 2).
    #[test]
    fn records_with_work_totals_do_not_overflow() {
        let most = u128::from(u32::MAX);
        let (busy, total) = super::records_with_work(&[u32::MAX; 6], 2, 12, 2);
        assert_eq!((busy, total), (2, 2 * most.pow(3)), "two dispatches of (2^32 - 1)^3 groups");
        let (busy, total) = super::records_with_work(&[u32::MAX; 12], 0, 16, 3);
        assert_eq!((busy, total), (3, 3 * most.pow(2)), "three draws of (2^32 - 1)^2 vertex instances");
    }

    /// A texture read back is read as the bytes the copy writes, which
    /// is what its buffer holds (GetCopyableFootprints' total): the last
    /// row has no padding, and a block-compressed footprint has fewer rows
    /// than texels high (Sol's review, finding 3). A 1x1 RGBA8 target is 4
    /// bytes, not a 256-byte row; 8x8 BC1 is two rows of 16 bytes.
    #[test]
    fn texture_read_backs_map_only_the_bytes_the_copy_writes() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        let Some(host) = crate::host::test_host() else {
            return; // no host GPU here
        };
        for (format, size, bytes) in [(DXGI_FORMAT_R8G8B8A8_UNORM, 1, 4), (DXGI_FORMAT_BC1_UNORM, 8, 256 + 16)] {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: size,
                Height: size as u32,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let (mut rows, mut row_bytes, mut total) = (0, 0, 0);
            // SAFETY: outputs for one subresource.
            unsafe {
                host.device.GetCopyableFootprints(
                    &desc,
                    0,
                    1,
                    0,
                    Some(&mut footprint),
                    Some(&mut rows),
                    Some(&mut row_bytes),
                    Some(&mut total),
                )
            };
            assert_eq!(total, bytes, "{format:?}: the copy's bytes");
            assert_eq!(super::mapped_len(&footprint, rows, row_bytes) as u64, total, "{format:?}: what is mapped");
        }
        // Block formats are turned away before the mapping; rows shorter
        // than the texels asked for convert to nothing, not to a panic.
        assert_eq!(super::rgba_texel_bytes(DXGI_FORMAT_BC1_UNORM.0 as u32), None);
        assert_eq!(super::to_rgba(28, 2, 2, 256, &[0; 4]), None);
        // A real 1x1 target read back: the mapping is its 4 bytes.
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 1,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            host.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_SOURCE, None, &mut texture)
        }
        .unwrap();
        let list = host.new_list().unwrap();
        let read = super::read_back(&host.device, &list.list, texture.as_ref().unwrap()).expect("a read-back");
        list.close();
        host.execute(&[&list]);
        assert!(read.rows_are_texels());
        assert_eq!(read.map(|bytes| bytes.len()), Some(4));
    }

    /// The bytes an indirect line reads: only layouts the clamp takes
    /// (kinds 0, 1, 2 and 10, whole records, 4-byte strides), inside their
    /// buffer, and 4 MiB at most. (The count is not read: finding 1.)
    #[test]
    fn records_span_takes_only_layouts_inside_their_buffers() {
        let span = super::records_span;
        assert_eq!(span((10, 12, 2), (0, 24)), Some(24));
        assert_eq!(span((1, 20, 3), (40, 100)), Some(60));
        assert_eq!(span((10, 12, 0), (0, 12)), Some(12), "max 0 reads one record");
        assert_eq!(span((10, 1, 4), (0, 1 << 16)), None, "stride 1");
        assert_eq!(span((10, 0, 2), (0, 1 << 16)), None, "stride 0");
        assert_eq!(span((0, 12, 1), (0, 1 << 16)), None, "a draw's record is 16 bytes");
        assert_eq!(span((10, 14, 1), (0, 1 << 16)), None, "not a multiple of 4");
        assert_eq!(span((5, 12, 1), (0, 1 << 16)), None, "no such kind");
        assert_eq!(span((10, 12, 2), (4, 24)), None, "past the buffer's end");
        assert_eq!(span((10, 12, 2), (u64::MAX - 8, u64::MAX)), None, "an offset that wraps");
        assert_eq!(span((10, 12, 1 << 30), (0, u64::MAX)), None, "12 GiB of records");
    }

    /// The line reads only the words it is given: a short read gives a
    /// shorter line, not a panic.
    #[test]
    fn indirect_line_reads_only_the_words_it_has() {
        let line = super::indirect_line(&[2, 1, 1, 0, 1, 1], (10, 12, 2), "p");
        assert_eq!(line, "indirect kind 10 max 2 count none: 1 of 2 records with work, 2 in all; first [[2, 1, 1], [0, 1, 1]] p");
        let line = super::indirect_line(&[7], (10, 12, 3), "p");
        assert_eq!(line, "indirect kind 10 max 3 count none: 1 of 3 records with work, 7 in all; first [[7]] p");
    }

    /// An execution whose records are not read says so: a count the title
    /// gives is unavailable, not absent, and a dropped one is named
    /// (Sol's review, findings 1 and 11).
    #[test]
    fn unread_records_say_why_and_never_claim_no_count() {
        let title = super::Unread::TitleBuffer;
        let line = super::unread_line((10, 12, 64), true, true, title, "[mesh f5e5]");
        assert_eq!(
            line,
            "indirect kind 10 max 64 count unavailable: records not read (a title buffer's state is not tracked); DROPPED [mesh f5e5]"
        );
        let line = super::unread_line((0, 16, 4), false, true, title, "[vertex 1234]");
        assert!(line.contains("count none:") && line.contains("DROPPED [vertex 1234]"), "{line}");
        let line = super::unread_line((10, 12, 64), true, false, title, "[mesh f5e5]");
        assert!(line.contains("count unavailable:") && !line.contains("DROPPED"), "{line}");
    }

    /// A layout the clamp takes no execution of (here stride 1 with max 4)
    /// records nothing: read as given, the line mapped two words and showed
    /// a third, a panic in the submission's completion (Sol's review,
    /// finding 2).
    #[test]
    fn read_indirect_declines_a_layout_the_clamp_would_refuse() {
        let Some(host) = crate::host::test_host() else {
            return; // no host GPU here
        };
        let (src, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: inside the mapping: two mesh records.
        unsafe { (data as *mut [u32; 6]).write([2, 1, 1, 3, 1, 1]) };
        let list = host.new_list().unwrap();
        let scratch = crate::indirect::bound(&list, (&src, 0), None, (10, 12, 2)).expect("the clamp");
        for spec in [(10, 1, 4), (10, 0, 2), (5, 12, 2), (0, 12, 1)] {
            super::read_indirect(&list, &scratch, 0, spec, "test".to_owned());
        }
        crate::indirect::done(&list, &scratch);
        list.close();
        host.execute(&[&list]);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// A line for records not read is made when the execution is recorded,
    /// and logged by the reporting thread, never there: recording holds
    /// the list's state, and a log that blocks would hold it too (Sol's
    /// second re-check, new fault 6).
    #[test]
    fn records_not_read_are_logged_by_the_reporter() {
        let Some(host) = crate::host::test_host() else {
            return; // no host GPU here
        };
        let (scratch, _) = host.shared_buffer(1 << 16).unwrap();
        let list = host.new_list().unwrap();
        super::read_indirect(&list, &scratch, 0, (10, 12, 500_000), "reported".to_owned());
        list.close();
        crate::host::reports_settled();
        let line = |l: &String| l.contains("records not read") && l.ends_with(" reported");
        assert!(super::LINES.lock().unwrap_or_else(|e| e.into_inner()).iter().any(line), "the line");
        let off = super::OFF_REPORTER.lock().unwrap_or_else(|e| e.into_inner());
        assert!(!off.iter().any(line), "logged where it was recorded: {off:?}");
    }

    /// One list reads back at most 256 executions' records (and 16 MiB):
    /// past them, each line says the list's read-backs are spent, and
    /// nothing is allocated for it (Sol's re-check, finding 6).
    #[test]
    fn a_list_reads_back_a_bounded_number_of_executions() {
        let Some(host) = crate::host::test_host() else {
            return; // no host GPU here
        };
        let (src, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: inside the mapping: two mesh records.
        unsafe { (data as *mut [u32; 6]).write([4, 1, 1, 0, 1, 1]) };
        let list = host.new_list().unwrap();
        let scratch = crate::indirect::bound(&list, (&src, 0), None, (10, 12, 2)).expect("the clamp");
        for _ in 0..300 {
            super::read_indirect(&list, &scratch, 0, (10, 12, 2), "many".to_owned());
        }
        crate::indirect::done(&list, &scratch);
        list.close();
        host.execute(&[&list]);
        crate::host::reports_settled();
        let lines = super::LINES.lock().unwrap_or_else(|e| e.into_inner());
        let count = |text: &str| lines.iter().filter(|l| l.contains(text) && l.ends_with(" many")).count();
        assert_eq!(count("records not read (the list's read-backs are spent"), 44);
        assert_eq!(count("1 of 2 records with work"), 256);
    }

    /// An execution without a count whose records are more than a line
    /// reads (4 MiB) is not left out: its line says why its records were
    /// not read (Sol's re-check, new fault 7). Nothing is copied for it.
    #[test]
    fn records_past_the_cap_say_so() {
        let Some(host) = crate::host::test_host() else {
            return; // no host GPU here
        };
        let (scratch, _) = host.shared_buffer(1 << 16).unwrap();
        let list = host.new_list().unwrap();
        super::read_indirect(&list, &scratch, 0, (10, 12, 400_000), "big".to_owned());
        list.close();
        crate::host::reports_settled();
        let lines = super::LINES.lock().unwrap_or_else(|e| e.into_inner());
        let line = "indirect kind 10 max 400000 count none: records not read \
                    (4800000 bytes of records, more than the 4194304 a line reads); big";
        assert!(lines.iter().any(|l| l == line), "{lines:?}");
    }

    /// A closed list may run again without a reset, as the title submits
    /// it again: each run reports what that run copied, from bytes taken
    /// while the next run cannot write them (Sol's re-check, new fault 2:
    /// the first report took the work, and the second run had none).
    #[test]
    fn a_list_that_runs_twice_reports_twice() {
        let Some(host) = crate::host::test_host() else {
            return; // no host GPU here
        };
        let (src, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: inside the mapping: two mesh records.
        unsafe { (data as *mut [u32; 6]).write([3, 1, 1, 0, 1, 1]) };
        let list = host.new_list().unwrap();
        let scratch = crate::indirect::bound(&list, (&src, 0), None, (10, 12, 2)).expect("the clamp");
        super::read_indirect(&list, &scratch, 0, (10, 12, 2), "twice".to_owned());
        crate::indirect::done(&list, &scratch);
        list.close();
        host.execute(&[&list]);
        host.execute(&[&list]);
        crate::host::reports_settled();
        let lines = super::LINES.lock().unwrap_or_else(|e| e.into_inner());
        let line = "indirect kind 10 max 2 count none: 1 of 2 records with work, 3 in all; first [[3, 1, 1], [0, 1, 1]] twice";
        assert_eq!(lines.iter().filter(|l| *l == line).count(), 2, "{lines:?}");
    }

    /// The reader records its copy of the clamp's scratch buffer into a
    /// real list behind a real clamp, which runs, and its line is made once
    /// the list has run, from what the clamp wrote: two mesh records, one
    /// of two groups and one of none.
    #[test]
    fn read_indirect_copies_clamped_records_into_a_list_that_runs() {
        let Some(host) = crate::host::test_host() else {
            return; // no host GPU here
        };
        let (src, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: inside the mapping: two mesh records.
        unsafe { (data as *mut [u32; 6]).write([2, 1, 1, 0, 1, 1]) };
        let list = host.new_list().unwrap();
        let scratch = crate::indirect::bound(&list, (&src, 0), None, (10, 12, 2)).expect("the clamp");
        super::read_indirect(&list, &scratch, 0, (10, 12, 2), "clamped".to_owned());
        crate::indirect::done(&list, &scratch);
        list.close();
        host.execute(&[&list]);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        crate::host::reports_settled();
        let lines = super::LINES.lock().unwrap_or_else(|e| e.into_inner());
        let line = "indirect kind 10 max 2 count none: 1 of 2 records with work, 2 in all; first [[2, 1, 1], [0, 1, 1]] clamped";
        assert!(lines.iter().any(|l| l == line), "{lines:?}");
    }

    /// Target dumps are refused unless VICTORY_DUMP_UNSAFE=1 is set too:
    /// the inherited dump can still race a present's state change (Sol's
    /// second re-check, finding 9).
    #[test]
    fn target_dumps_need_the_unsafe_switch() {
        assert!(!super::dump_allowed(true, None), "asked, no switch");
        assert!(!super::dump_allowed(true, Some("0")), "asked, switch 0");
        assert!(!super::dump_allowed(true, Some("yes")), "asked, switch not 1");
        assert!(super::dump_allowed(true, Some("1")), "asked, switch 1");
        assert!(super::dump_allowed(true, Some(" 1 ")), "asked, switch 1 with spaces");
        assert!(!super::dump_allowed(false, Some("1")), "not asked");
    }

    /// The frame's targets are those of submitted lists, noted for that
    /// frame, each once; a frame's dump takes its own only (finding 9).
    #[test]
    fn targets_join_the_frame_they_were_noted_for_once() {
        let mut targets = vec![(7, 1, 'a')];
        super::admit(&mut targets, vec![(7, 1, 'b'), (7, 2, 'c'), (6, 3, 'd'), (7, 2, 'e'), (8, 2, 'f')], 7);
        assert_eq!(targets, [(7, 1, 'a'), (7, 2, 'c'), (8, 2, 'f')], "3 was noted for the frame before");
        targets.insert(0, (5, 9, 'z'));
        assert_eq!(super::take_frame(&mut targets, 7), [(1, 'a'), (2, 'c')], "frame 7's");
        assert_eq!(targets, [(8, 2, 'f')], "frame 8's stays; frame 5's goes");
    }

    /// The target dump copies a texture out with barriers made from its
    /// tracked state under the submission lock, and leaves it in that
    /// state: here a 1x1 target the tracker has in the common state
    /// (finding 9).
    #[test]
    fn a_dumped_target_is_left_in_its_tracked_state() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        let Some(host) = crate::host::test_host() else {
            return; // no host GPU here
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 1,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe { host.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut texture) }
            .unwrap();
        let copy = crate::resource::HostTexture::for_tests(texture.unwrap(), 0);
        super::dump_one(host, &copy, 1, 0, std::ptr::null_mut(), &desc);
        assert_eq!(copy.state.load(std::sync::atomic::Ordering::Relaxed), 0);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// A title draw or mesh dispatch counts for the heavy scene's detector
    /// whether it runs or is dropped, and a compute dispatch never: here
    /// all are dropped (no pipeline). Before, the detector took draws run
    /// plus drops, so a dropped compute dispatch counted and a run one did
    /// not (Sol's review).
    #[test]
    fn draw_attempts_count_title_draws_run_or_not_and_no_dispatches() {
        let Some(host) = crate::host::test_host() else {
            return; // no host GPU here
        };
        let list = host.new_list().unwrap();
        let mut state = crate::draw::State::default();
        // No heavy list (tests have none, nor a capture folder): the draw
        // path counts nothing (Sol's review, finding 8).
        let before = crate::draw::attempted();
        state.draw(&list, 3, 1, 0, 0);
        state.dispatch_mesh(&list, 1);
        assert_eq!(crate::draw::attempted(), before, "no heavy list: nothing counted");
        super::COUNTING_IN_TESTS.store(true, std::sync::atomic::Ordering::Relaxed);
        let before = crate::draw::attempted();
        state.draw(&list, 3, 1, 0, 0);
        state.draw_indexed(&list, 3, 1, 0, 0, 0);
        state.dispatch_mesh(&list, 1);
        assert_eq!(crate::draw::attempted() - before, 3, "a draw, an indexed draw and a mesh dispatch, all dropped");
        state.dispatch(&list, 1, 1, 1);
        assert_eq!(crate::draw::attempted() - before, 3, "a compute dispatch, dropped, is not one");
        list.close();
    }

    /// `VICTORY_CAPTURE_HEAVY_FRAMES` counts presented frames from the first
    /// frame that recorded 1,000 title draws and mesh dispatches, run or
    /// dropped. A light frame later does not move that start.
    #[test]
    fn heavy_frames_count_from_the_first_heavy_one() {
        let offsets = [2, 5];
        let mut state = super::Heavy::default();
        assert!(!super::heavy_frame(10, 999, &mut state, &offsets), "999 attempts: not heavy");
        assert_eq!(state.first, None);
        // 1,000: the scene starts here; the list counts from the frame after.
        assert!(!super::heavy_frame(11, 1_000, &mut state, &offsets));
        assert_eq!(state.first, Some(11));
        for (n, listed) in [(12, false), (13, true), (14, false), (15, false), (16, true), (17, false)] {
            assert_eq!(super::heavy_frame(n, 0, &mut state, &offsets), listed, "{n}");
        }
        assert!(!super::heavy_frame(40, 10, &mut state, &offsets), "a later light frame");
        assert_eq!(state.first, Some(11));
    }

    #[test]
    fn crc_matches_the_png_check_value() {
        assert_eq!(super::crc32(b"IEND"), 0xae42_6082);
    }

    /// R16G16B16A16_UNORM is integers, not half floats; 89 is 10:10:10:2,
    /// not B8G8R8A8.
    #[test]
    fn unorm16_and_xr_bias_targets_decode_as_their_formats() {
        // Red 0x8000 (a half-float -0), green 0x4000, blue 0.
        let unorm16 = [0x00, 0x80, 0x00, 0x40, 0, 0, 0xff, 0xff];
        assert_eq!(super::to_rgba(11, 1, 1, 8, &unorm16), Some(vec![255, 127, 0, 255]));
        // Red 0x3ff, green 0x1ff, blue 0, alpha 3.
        let xr = (0x3ffu32 | 0x1ff << 10 | 3 << 30).to_le_bytes();
        assert_eq!(super::to_rgba(89, 1, 1, 4, &xr), Some(vec![255, 127, 0, 255]));
        // B8G8R8A8 still swaps to RGBA.
        assert_eq!(super::to_rgba(87, 1, 1, 4, &[1, 2, 3, 4]), Some(vec![3, 2, 1, 255]));
    }

    /// A word target's words are counted by the bits each needs: zeros in
    /// bucket 0, then one bucket per power of two, every word in exactly one
    /// bucket and no value counted apart from its bucket.
    #[test]
    fn target_words_are_counted_by_bit_length() {
        // Both edges of several buckets, the lowest and highest word among them.
        let words = [0, 1, 2, 3, 4, 7, 8, 0xff, 0x100, 0xffff, 0x1_0000, 0x7fff_ffff, 0x8000_0000, u32::MAX];
        let counts = super::words_by_bit_length(words);
        // Expected by comparison alone: a word's bucket is the fewest bits `k`
        // with `word < 2^k`.
        let mut expected = [0usize; 33];
        for w in words {
            let k = (0..=32).find(|&k| u64::from(w) < 1u64 << k).expect("a 32-bit word");
            expected[k] += 1;
        }
        assert_eq!(counts, expected);
        assert_eq!(counts.iter().sum::<usize>(), words.len());
        // The highest word shares its bucket with the other words of 32 bits.
        assert_eq!(counts[32], 2);
        assert_eq!(counts[0], 1);
        assert_eq!(super::words_by_bit_length([]), [0; 33]);
    }
}
