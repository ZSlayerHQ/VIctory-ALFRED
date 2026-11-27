//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    #[cfg(feature = "diagnostics")]
    total: u64,
    /// The title texture it is for (the binding trace's notes).
    #[cfg(feature = "diagnostics")]
    resource: usize,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    /// The first of them that is uploaded, and the bytes of title memory
    /// read for it and the smaller ones.
    first: u32,
    size: usize,
    /// The host texture's DXGI format (`D3D12_RESOURCE_DESC::Format`), for
    /// the upload recheck's line.
    format: u32,
    /// Whether the runtime made the texture for a placed view (`true`) or the
    /// title made the object (`false`): the texture recheck's slot kind.
    placed: bool,
    /// With `VICTORY_TEXTURE_RECHECK`: the hash of the bytes last detiled
    /// into the staging buffer, the bytes an upload copies, and the trap's
    /// epoch it was taken in (`crate::host::recheck::epoch`).
    detiled_hash: Mutex<Option<(u64, blake3::Hash)>>,
    /// The staging buffer's bytes (counted in `STAGED` while it lives).
    bytes: u64,
    memory: *const u8,
    last: Mutex<Last>,
    /// Whether the staging buffer was detiled since a submitted list last
    /// uploaded it, and whether one has uploaded it at all: a list records
    /// the upload only then. (Every list that sampled a title texture
    /// uploaded all of it again, changed or not: thousands of subresource copies
    /// of over a thousand textures in one frame.)
    dirty: std::sync::atomic::AtomicBool,
    uploaded: std::sync::atomic::AtomicBool,
    /// When a list last recorded a use of it (`crate::host::moment`).
    used: std::sync::atomic::AtomicU64,
    /// Levels on demand (`crate::tiles`): the texture's host copy was made
    /// without memory of its own when the staging buffer was made.
    on_demand: bool,
    /// For such a texture: the levels that were the title's at the last
    /// detiling (a bit a level; the staging buffer holds zeros for the
    /// others), and what that was judged by ([`Staging::judgement`]).
    detiled: std::sync::atomic::AtomicU32,
    judged: [std::sync::atomic::AtomicU64; 2],
    /// For such a texture: when an upload last left a held level without
    /// tiles (`crate::host::moment`, never 0; 0: none did). The upload is
    /// recorded again about a second later if the card has room by then
    /// ([`upload`]).
    waiting: std::sync::atomic::AtomicU64,
    /// A 2-D texture of one slice, as the texture pack sees it
    /// (`crate::texture_pack`); none for any other, and for every texture
    /// while the pack and its log are off.
    shape: Option<crate::texture_pack::Shape>,
    /// The pack's count of finished preparations (`generation`) plus one
    /// when this texture's picture was still being prepared at its last
    /// detiling; 0: none waited for. Once the count moves the texture is
    /// detiled anew ([`Staging::refresh`]).
    pack_wait: std::sync::atomic::AtomicU64,
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after.
#[derive(Default)]
struct Last {
    hash: Option<Sum>,
    asked: u64,
    /// The frame a partly mapped texture was last read in.
    partial: Option<u64>,
    /// The texture was found changed without a write the guard saw
    /// ([`Staging::unseen_write`]): it is hashed at every check until this
    /// frame.
    unguarded: u64,
    /// With receipts (`crate::receipts`): the journal's number taken before
    /// the bytes the staging buffer holds were read (or last found unchanged
    /// since): every write stamped at or below it is in them. 0: none.
    /// Not `asked`, which the partial path's skip moves past what was read.
    receipt: u64,
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// The buffer a texture the host writes is copied through into the texture
/// of a view the title placed in its memory in another format
/// ([`alias`]): every level of every layer as each of the two formats lays
/// it out for a copy (the same bytes: the texels are as wide), the written
/// texture's layout first.
struct Through {
    buffer: ID3D12Resource,
    parts: Vec<(
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    )>,
}

/// The most levels times layers [`alias`] copies (a cube of nine levels).
const ALIAS_PARTS: u32 = 54;

// SAFETY: a device-local buffer only lists record copies with.
unsafe impl Send for Through {}
// SAFETY: as above.
unsafe impl Sync for Through {}

/// The copy buffers of [`alias`], by the placed view's texture.
static THROUGH: Mutex<Option<HashMap<usize, Arc<Through>>>> = Mutex::new(None);

/// Brings `placed`, the texture of a view the title placed in the memory
/// of `source` in another format (`crate::resource::alias_of`), up to date
/// with what the host wrote to `source`: the texels of every level and
/// layer are copied as they are, through a buffer (the host copies between
/// two formats of one texel width in no other way). Recorded the first time
/// the list reads the view
/// and again whenever the list used `source` in another way since (wrote
/// it, or read it itself). Leaves `placed` in the copy destination state,
/// as [`upload`] does. False if the copy cannot be made: the caller fills
/// the texture from title memory, as before.
pub(crate) fn alias(list: &HostList, placed: This, source: This) -> bool {
    const TARGET_OR_UNORDERED: u32 = 0x4 | 0x8;
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let (Some(to), Some(from)) = (
        crate::resource::host_resource(placed),
        crate::resource::host_resource(source),
    ) else {
        return false;
    };
    let Some(host) = host() else { return false };
    let through = {
        let mut map = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
        let map = map.get_or_insert_with(HashMap::new);
        match map.get(&(placed as usize)) {
            Some(through) => through.clone(),
            None => {
                // SAFETY: live textures.
                let (of_from, of_to) = unsafe { (from.GetDesc(), to.GetDesc()) };
                // Levels times layers (a volume's slices are one part).
                let count = |d: &D3D12_RESOURCE_DESC| {
                    let layers = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                        1
                    } else {
                        u32::from(d.DepthOrArraySize)
                    };
                    u32::from(d.MipLevels) * layers
                };
                let parts = count(&of_from);
                if parts == 0 || parts > ALIAS_PARTS || parts != count(&of_to) {
                    return false;
                }
                let mut rows = [(); 2]
                    .map(|()| vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); parts as usize]);
                let mut bytes = [0u64; 2];
                // SAFETY: plain queries; the out pointers are ours and
                // hold `parts` layouts each.
                unsafe {
                    host.device.GetCopyableFootprints(
                        &of_from,
                        0,
                        parts,
                        0,
                        Some(rows[0].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[0]),
                    );
                    host.device.GetCopyableFootprints(
                        &of_to,
                        0,
                        parts,
                        0,
                        Some(rows[1].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[1]),
                    );
                }
                let alike = rows[0].iter().zip(&rows[1]).all(|(a, b)| {
                    let (fa, fb) = (a.Footprint, b.Footprint);
                    a.Offset == b.Offset
                        && (fa.Width, fa.Height, fa.Depth, fa.RowPitch)
                            == (fb.Width, fb.Height, fb.Depth, fb.RowPitch)
                });
                if bytes[0] == 0 || bytes[0] != bytes[1] || !alike {
                    return false;
                }
                let Some(buffer) = host.uav_buffer(bytes[0]) else {
                    return false;
                };
                crate::host::name(&buffer, || {
                    format!(
                        "alias copy {}x{} f{} to f{}",
                        of_from.Width, of_from.Height, of_from.Format.0, of_to.Format.0
                    )
                });
                tracing::info!(
                    view = ?placed,
                    texture = ?source,
                    width = of_from.Width,
                    height = of_from.Height,
                    parts,
                    written_as = of_from.Format.0,
                    read_as = of_to.Format.0,
                    "a view placed in the memory of a texture the host writes, in another format: read from that texture by a copy on the host"
                );
                let [of_written, of_read] = rows;
                let through = Arc::new(Through {
                    buffer,
                    parts: of_written.into_iter().zip(of_read).collect(),
                });
                map.insert(placed as usize, through.clone());
                through
            }
        }
    };
    let first = list.first_use(placed as usize);
    // The source's first use in the list records nothing (its prelude
    // brings it to the state); a use in another way since the last copy
    // shows as a change of state.
    let before = list.transition(source, COPY_SOURCE).map(|(texture, before)| {
        let state = crate::host::legal_state(texture, COPY_SOURCE);
        let barrier = crate::host::transition(
            texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(state as i32),
        );
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
        before
    });
    if !first && before.is_none() {
        return true;
    }
    card_writes(list, placed, to, None);
    crate::list::ensure_state(list, placed, COPY_DEST);
    let part = |texture: &ID3D12Resource, index: usize| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index as u32,
        },
    };
    let rows = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(&through.buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, read, written) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("alias copy of {source:?} into {placed:?}"));
    // SAFETY: an open list and live resources; the buffer is as big as
    // the footprints and rests in the common state between copies.
    unsafe {
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, common, written)]);
        for (index, (of_written, _)) in through.parts.iter().enumerate() {
            list.list
                .CopyTextureRegion(&rows(of_written), 0, 0, 0, &part(from, index), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, written, read)]);
        for (index, (_, of_read)) in through.parts.iter().enumerate() {
            list.list
                .CopyTextureRegion(&part(to, index), 0, 0, 0, &rows(of_read), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, read, common)]);
    }
    list.keep(&through.buffer);
    // A draw that also writes the source (a view of it handled before
    // this one) needs it in that state again; the next read of the alias
    // then copies once more, as it must after a write.
    if let Some(before) = before
        && before & TARGET_OR_UNORDERED != 0
    {
        crate::list::ensure_state(list, source, before);
    }
    true
}

/// Whether a texture refresh that finds the runtime remapping title memory
/// goes on without waiting for it (`VICTORY_TEXTURE_NO_WAIT=1`; off by
/// default).
///
/// A texture whose memory is all mapped is read in place, holding its
/// bytes against the runtime's changes of their mapping
/// (`crate::watch::title_memory_reading`; all of title memory with
/// `VICTORY_MAPPING_RANGES=0`), so that no page goes away under the read. While the world streams in the runtime remaps all the time,
/// and each of the frame's hundreds of refreshes waited for it: several runs
/// have frame lines with tenths of a second of "textures", up to
/// seconds, and on every one of them the wait for this lock is the whole
/// of it (nearly all of the time; the write-watch askings take milliseconds,
/// the texture's own lock none).
///
/// With the switch the refresh asks for the lock without waiting. If the
/// runtime has it, the texture is read through the kernel instead, which
/// needs no lock: it fails if a page is not there, and then the texture is
/// left as it is and read at its next refresh, a frame later. A read
/// through the kernel costs about 0.6 ms a megabyte against 0.3 in place.
///
/// What changes for the picture: a texture that arrives while the runtime
/// is remapping may show a frame later. It never shows partly read.
fn no_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_NO_WAIT").is_some_and(|v| v == "1"))
}

/// Whether the checks of "did this texture's memory change" hash with
/// [`crate::digest::digest`] instead of blake3 (`VICTORY_TEXTURE_DIGEST=1`;
/// off by default). The checks hash tens of megabytes a frame to find under a megabyte
/// changed, on the submitting thread; in the timing test (one thread, bytes
/// in cache) the digest takes 0.031 ms a megabyte, blake3 0.08. Off, every
/// check computes and compares what it did before. Not for the trap's hashes (`detiled_hash`, `host::recheck`),
/// which stay blake3.
///
/// Read once for the run: every [`Sum`] a staging buffer keeps and every
/// fingerprint the texture queue keeps is of the same kind as the one it is
/// compared with.
fn texture_digest() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_DIGEST").is_some_and(|v| v == "1"))
}

/// What a check of a texture's memory keeps of its bytes: blake3's whole
/// hash, or with [`texture_digest`] the digest. A value of one kind is never
/// compared with one of the other ([`texture_digest`] is read once); if it
/// were, they are unequal (derived `PartialEq`: different variants), so the
/// texture counts as changed and is detiled anew, never passed over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sum {
    Blake3(blake3::Hash),
    Digest(u128),
}

impl Sum {
    /// The sum of `bytes`: the digest if `digest`, else blake3's hash.
    fn of(bytes: &[u8], digest: bool) -> Self {
        if digest {
            Self::Digest(crate::digest::digest(bytes))
        } else {
            Self::Blake3(blake3::hash(bytes))
        }
    }
}

/// Since the last frame-rate line, with [`no_wait`]: refreshes that read
/// through the kernel because title memory was being remapped, and those
/// left for the next refresh because a page was not there.
pub(crate) static UNLOCKED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Since the last frame-rate line, of texture refreshes on every thread, in
/// microseconds: waiting for the texture's own lock (another thread is
/// refreshing the same texture), asking the write watch and the guards,
/// waiting for title memory to hold still (the runtime remapping the
/// texture's bytes, `crate::watch::title_memory_reading`; its side is
/// `crate::watch::REMAPS`), and the refreshes in all.
///
/// While the world streams in, the frame line's "textures" is seconds a
/// frame in some runs and tenths in others for the same megabytes, and the
/// submitting thread is not running while it lasts (a fraction of the
/// "textures" time is its own): it waits.
/// These say for what.
pub(crate) static REFRESH_WAITS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// `f`'s result, its time added to [`REFRESH_WAITS`]' place `slot`.
fn waited<T>(slot: usize, f: impl FnOnce() -> T) -> T {
    let started = std::time::Instant::now();
    let out = f();
    REFRESH_WAITS[slot].fetch_add(started.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    out
}

thread_local! {
    /// Room to read a texture's memory into: one buffer per submitting
    /// thread. Each texture kept its own, a second copy in memory of every
    /// write-watched or partly mapped texture ever sampled, for the run.
    static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Calls `f` with `size` bytes of this thread's read buffer (their
/// contents are whatever the last texture left).
fn with_read<R>(size: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
    READ.with(|read| {
        let mut read = read.borrow_mut();
        if read.len() < size {
            read.resize(size, 0);
        }
        f(&mut read[..size])
    })
}

/// Whether a texture all of whose memory is mapped is read through the
/// kernel's copy instead of in place (`VICTORY_TEXTURE_COPY=1`; off by
/// default). Round 3 of the memory work read every such texture that way
/// to be safe from pages taken away under the read; it cost 0.6 ms per MB
/// against 0.3 in place (several times the milliseconds a frame while the world
/// streams in), and both runs with it lost the graphics
/// device, for a reason not found. The read in place is safe now for
/// another reason (`crate::watch::title_memory_reading`); this switch is
/// left to tell whether the copy was what lost the device.
fn through_copy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_COPY").is_some_and(|v| v == "1"))
}

/// Calls `f` with the `size` bytes of title memory at `memory`, read
/// through the kernel into this thread's read buffer; pages that are not
/// there read as zeros. The bytes are those a read in place would see.
fn with_title<R>(memory: *const u8, size: usize, f: impl FnOnce(&[u8]) -> R) -> R {
    with_read(size, |read| {
        crate::host::copy_mapped(read.as_mut_ptr(), memory, size, false);
        f(read)
    })
}

/// Staging buffers alive (also those only a list still holds) and their
/// bytes.
static STAGED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Staging {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        STAGED[0].fetch_sub(1, Relaxed);
        STAGED[1].fetch_sub(self.bytes, Relaxed);
        // No list holds it (each that recorded an upload from it held it
        // until it had run): the next staging of its size may have it.
        keep_spare(self.buffer.clone(), self.data, self.bytes);
    }
}

/// Whether a staging buffer no texture holds is kept for the next one of
/// its size (`VICTORY_STAGING_SPARE=0`: not kept, every staging buffer
/// made at its exact size, as before).
///
/// A staging buffer is let go a few idle frames after its texture was last
/// sampled ([`let_go_idle`]), and the next list that samples the texture
/// makes one anew: a run made tens of thousands of upload buffers in
/// minutes. Making one is a committed resource in the upload heap: pages
/// the system zeroes and the card maps, on the recording thread (another
/// fork measured 674 µs for 512 KiB made, against 20 µs for one kept).
fn spare_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_STAGING_SPARE").is_none_or(|v| v != "0"))
}

/// Staging buffers no texture holds, by their bytes; their bytes in all.
struct Spare {
    by_size: std::collections::BTreeMap<u64, Vec<Kept>>,
    bytes: u64,
}

/// A spare buffer, still mapped at `data`, kept since `since`
/// (`crate::host::moment`).
struct Kept {
    buffer: ID3D12Resource,
    data: *mut u8,
    since: u64,
}

// SAFETY: a buffer no list uses, and its mapping, which only the thread
// that takes it back writes.
unsafe impl Send for Spare {}

static SPARE: Mutex<Spare> = Mutex::new(Spare {
    by_size: std::collections::BTreeMap::new(),
    bytes: 0,
});

/// The most bytes of spare buffers kept; one that would go past it is let go.
const SPARE_MOST: u64 = 512 << 20;

/// Since the start: staging buffers taken from the spare ones instead of
/// made, and their bytes.
static SPARE_TAKEN: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The bytes a staging buffer for `total` is made with: rounded up to an
/// eighth of its power of two, and to 64 KiB, so that textures of nearby
/// sizes take each other's (an eighth more memory at most).
fn staging_bytes(total: u64) -> u64 {
    if !spare_on() {
        return total;
    }
    let step = (total.max(1).next_power_of_two() / 8).max(64 << 10);
    total.next_multiple_of(step)
}

/// Keeps a staging buffer of `bytes` for the next staging of that size, if
/// there is room.
fn keep_spare(buffer: ID3D12Resource, data: *mut u8, bytes: u64) {
    if !spare_on() {
        return;
    }
    let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
    if spare.bytes + bytes > SPARE_MOST {
        return;
    }
    spare.bytes += bytes;
    spare.by_size.entry(bytes).or_default().push(Kept {
        buffer,
        data,
        since: crate::host::moment(),
    });
}

/// A staging buffer of `bytes` ([`staging_bytes`]) and its mapping: a spare
/// one, its first `zero` bytes zeroed as a new one's are, else a new one.
fn take_staging(host: &crate::host::Host, bytes: u64, zero: u64) -> Option<(ID3D12Resource, *mut u8)> {
    let kept = {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        let kept = spare.by_size.get_mut(&bytes).and_then(Vec::pop);
        if kept.is_some() {
            spare.bytes -= bytes;
            if spare.by_size.get(&bytes).is_some_and(Vec::is_empty) {
                spare.by_size.remove(&bytes);
            }
        }
        kept
    };
    let Some(kept) = kept else {
        return host.upload_buffer(bytes);
    };
    use std::sync::atomic::Ordering::Relaxed;
    SPARE_TAKEN[0].fetch_add(1, Relaxed);
    SPARE_TAKEN[1].fetch_add(bytes, Relaxed);
    // SAFETY: the buffer's mapping, `bytes` long; no list uses the buffer.
    unsafe { kept.data.write_bytes(0, zero.min(bytes) as usize) };
    Some((kept.buffer, kept.data))
}

/// Lets go of the spare buffers kept for `idle_ms` and `frames` frames.
fn let_go_spare(idle_ms: u64, frames: u64) {
    let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
    let mut freed = 0;
    spare.by_size.retain(|&bytes, kept| {
        kept.retain(|k| {
            let keep = !crate::host::long_ago(k.since, idle_ms, frames);
            if !keep {
                freed += bytes;
            }
            keep
        });
        !kept.is_empty()
    });
    spare.bytes -= freed;
}

/// For the memory line: staging buffers taken from the spare ones since
/// the start and their bytes, then the spare ones now and their bytes.
#[cfg(feature = "diagnostics")]
pub(crate) fn spare() -> [u64; 4] {
    use std::sync::atomic::Ordering::Relaxed;
    let spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
    let count = spare.by_size.values().map(|k| k.len() as u64).sum();
    [SPARE_TAKEN[0].load(Relaxed), SPARE_TAKEN[1].load(Relaxed), count, spare.bytes]
}

/// For the memory line: staging buffers alive, their bytes, and how many
/// of them their texture still has (the others wait for a list's reset).
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 3] {
    use std::sync::atomic::Ordering::Relaxed;
    let kept = STAGING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or(0, |map| map.len() as u64);
    [STAGED[0].load(Relaxed), STAGED[1].load(Relaxed), kept]
}

/// Staging buffers let go as idle since the start, and their bytes.
static LET_GO: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// For the memory line: staging buffers let go as idle, and their bytes.
#[cfg(feature = "diagnostics")]
pub(crate) fn let_go() -> [u64; 2] {
    use std::sync::atomic::Ordering::Relaxed;
    [LET_GO[0].load(Relaxed), LET_GO[1].load(Relaxed)]
}

/// Lets go of the staging buffers no list recorded a use of for `IDLE_MS`
/// and `IDLE_FRAMES` frames and no list holds: a staging buffer is system memory the size of its
/// texture, kept so that an unchanged texture is not uploaded again, and
/// those of textures no longer sampled were kept for the run (thousands of
/// them, gigabytes, by a run's end). The next list that samples such a
/// texture makes a staging buffer anew, detiles and uploads once more.
/// Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    use std::sync::atomic::Ordering::Relaxed;
    const IDLE_FRAMES: u64 = 4;
    // Spare buffers no staging has taken for as long go too.
    let_go_spare(crate::host::ram_idle_ms(crate::host::available_physical_ram()), IDLE_FRAMES);
    // The stale-upload trap keeps the last sample of each staging let go
    // (after the table's lock is left).
    let recheck = crate::host::recheck::on();
    let mut sampled: Vec<(usize, u64)> = Vec::new();
    let (mut went, mut bytes) = (0u64, 0u64);
    {
        let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        let Some(map) = map.as_mut() else {
            return;
        };
        if map.is_empty() {
            return;
        }
        let idle_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
        // A list that recorded an upload or a check holds the staging until it
        // is reset: one only the table holds is named by no list, recorded or
        // running.
        map.retain(|&id, s| {
            let keep = Arc::strong_count(s) > 1
                || !crate::host::long_ago(s.used.load(Relaxed), idle_ms, IDLE_FRAMES);
            if !keep {
                LET_GO[0].fetch_add(1, Relaxed);
                LET_GO[1].fetch_add(s.bytes, Relaxed);
                went += 1;
                bytes += s.bytes;
                if recheck {
                    sampled.push((id, s.used.load(Relaxed) >> 32));
                }
            }
            keep
        });
    }
    for (id, frame) in sampled {
        crate::host::recheck::let_go(id, frame);
    }
    if went > 0 {
        crate::host::trail(|| format!("sweep: {went} staging buffers of textures let go ({} KiB), held by no list", bytes >> 10));
    }
}

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let new = {
        let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        set.get_or_insert_with(Default::default).insert(resource as usize)
    };
    // With receipts: when, so that the title's writes over its memory after it are
    // counted (`crate::receipts::Summary::host_written_newer`).
    if crate::receipts::on() {
        crate::receipts::host_wrote(resource as usize, crate::watch::top());
    }
    // Diagnostics (run 177): where its title memory is, for placed views
    // the title reads that memory through ([`placed_in_host_written`]).
    if new && let Some((at, desc)) = crate::resource::texture_memory(resource) {
        let words = desc.words();
        let bytes = crate::resource::placed_bytes(&words).unwrap_or(0);
        if !at.is_null() && bytes > 0 {
            HOST_WRITTEN_AT
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(at as u64, (bytes, words[4], words[6], words[7], words[8], resource as usize));
        }
    }
}

/// The title memory of textures the host copied into, by address: bytes
/// and description words 4, 6, 7 and 8 (width, height, layers and levels,
/// format).
static HOST_WRITTEN_AT: Mutex<std::collections::BTreeMap<u64, (u64, u32, u32, u32, u32, usize)>> =
    Mutex::new(std::collections::BTreeMap::new());

/// A texture the title placed at `address` (description words `desc`) was
/// made: says so once if it lies in the memory of a texture the host
/// copied into ([`host_written`]). Its own host copy is filled from title
/// memory, which the copy never reached: the copied texels are on the
/// card in the other texture, where no shader that reads this view finds
/// them.
pub(crate) fn placed_in_host_written(address: u64, desc: &[u32; 14]) {
    let hit = {
        let map = HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner());
        map.range(..=address)
            .next_back()
            .filter(|(at, d)| address < **at + d.0)
            .map(|(at, d)| (*at, *d))
    };
    if let Some((at, (bytes, width, height, layers_levels, format, _))) = hit {
        static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::warn!(
                view_address = format_args!("{address:#x}"),
                view_offset = address - at,
                view_width = desc[4],
                view_height = desc[6],
                view_layers_levels = format_args!("{:#x}", desc[7]),
                view_format = desc[8],
                copied_address = format_args!("{at:#x}"),
                copied_bytes = bytes,
                copied_width = width,
                copied_height = height,
                copied_layers_levels = format_args!("{layers_levels:#x}"),
                copied_format = format,
                "HERO LEAD: a placed texture view lies in the memory of a texture the host copied into: it is filled from title memory, which the copy never reached"
            );
        }
    }
}

/// Whether the host GPU wrote `resource` by a copy or resolve
/// ([`host_written`]): its host copy cannot be made again from title
/// memory.
pub(crate) fn host_wrote(resource: usize) -> bool {
    HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&resource))
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    // The stale-upload trap ends what it follows of it (no lock is held).
    crate::host::recheck::released(resource);
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
    // Its memory may be given to other textures now (run 177: most of the
    // "HERO LEAD" lines were views in memory a released texture had held).
    HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner()).retain(|_, d| d.5 != resource);
    let mut through = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(through) = through.as_mut() {
        through.remove(&resource);
    }
    #[cfg(feature = "diagnostics")]
    crate::bind_trace::forget(resource);
    #[cfg(feature = "diagnostics")]
    crate::texture_pack::forget(resource);
    // With receipts: its receipt goes too (its own leaf locks).
    if crate::receipts::on() {
        crate::receipts::forget(resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    upload_from(list, resource, texture, 0);
}

/// [`upload`] of the mips from `first` on, the ones a view shows: of a
/// placed texture the title may have no memory for the larger ones, and
/// what lies there is another texture's.
pub(crate) fn upload_from(list: &HostList, resource: This, texture: &ID3D12Resource, first: u32) {
    let host_written = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)));
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    // What this list already uploads is the list's own to know: the
    // texture's staging entry may hold more mips because another list,
    // not submitted yet, staged them (`HostList::uploaded_from`).
    if host_written || list.uploaded_from(resource as usize).is_some_and(|from| from <= first) {
        return;
    }
    let Some(staging) = staging(resource, texture, first) else {
        return;
    };
    use std::sync::atomic::Ordering::Relaxed;
    staging.used.store(crate::host::moment(), Relaxed);
    // Levels on demand: a view of the texture now starts at another
    // level, or another texture now starts inside its span, with no byte
    // of its memory changed. The levels are judged anew: detiled again at
    // this list's submission, and uploaded by this list.
    if staging.on_demand && staging.judgement() != staging.judged() {
        staging.forget();
        staging.dirty.store(true, Relaxed);
    }
    // Levels on demand: the last upload left a held level without tiles
    // (the card had no room, or no heap could be had), and nothing else
    // would record an upload again until the title writes the texture: the
    // level stayed empty, the texture blurred or black. Asked again about
    // once a second, and only when the card has room; the staging buffer
    // still holds the level's bytes.
    if staging.on_demand {
        let since = staging.waiting.load(Relaxed);
        if since != 0 && crate::host::long_ago(since, 1000, 2) {
            if crate::resource::room_lately() {
                staging.waiting.store(0, Relaxed);
                staging.dirty.store(true, Relaxed);
                crate::tiles::STATS[10].fetch_add(1, Relaxed);
            } else {
                staging.waiting.store(crate::host::moment().max(1), Relaxed);
            }
        }
    }
    // The host texture has the staging buffer's contents: only the check
    // at submission is left, and what it finds changed the next list that
    // samples the texture uploads (a frame late where every list of the
    // frame was recorded before that check).
    if staging.uploaded.load(Relaxed) && !staging.dirty.load(Relaxed) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexSkipped, 1);
        list.note_uploaded(resource as usize, staging.first);
        list.defer_texture(staging);
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::TexUploads, 1);
    // Its time on the card, apart from the title's (`VICTORY_GPU_PASSES=1`).
    crate::events::begin(&list.list, crate::events::UPLOAD);
    #[cfg(feature = "diagnostics")]
    let stamped = list.upload_stamp_begin();
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.total,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: staging.subresource(i),
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    #[cfg(feature = "diagnostics")]
    list.upload_stamp_end(stamped);
    crate::events::end(&list.list);
    // When the list is submitted, its copy takes what the staging buffer
    // holds by then: noted before the check below, which sets `dirty`
    // again if it detiles (the next list then uploads once more).
    {
        let staging = staging.clone();
        list.defer_host(move || {
            staging.uploaded.store(true, Relaxed);
            staging.dirty.store(false, Relaxed);
        });
    }
    // The list now holds the copies of these mips.
    list.note_uploaded(resource as usize, staging.first);
    // Levels on demand (`crate::tiles`): after the check below, in the
    // order of this list's steps, and before the list is executed, the
    // levels the staging buffer holds as the title's are given the card's
    // memory if they have none. The list copies every level, so each one
    // given here is written whole by it (the rest are copies into no
    // memory, which the card drops).
    let on_demand = if staging.on_demand { crate::resource::host_tiled(resource) } else { None };
    // `VICTORY_TEXTURE_RECHECK`: remember the source, its hash and the list's
    // refresh, in this order, so the hash is what the copies below upload.
    let recheck = crate::host::recheck::on();
    let Some(on_demand) = on_demand else {
        // Once per submission however many of its lists sample the texture.
        list.defer_texture(staging.clone());
        if recheck {
            let staging = staging.clone();
            let id = resource as usize;
            list.defer_host(move || {
                if let Some(source) = staging.recheck_source() {
                    crate::host::recheck::uploaded(id, source);
                }
            });
        }
        if crate::receipts::on() {
            install_receipt(list, resource, &staging);
        }
        return;
    };
    list.defer_texture(staging.clone());
    if recheck {
        let staging = staging.clone();
        let id = resource as usize;
        list.defer_host(move || {
            if let Some(source) = staging.recheck_source() {
                crate::host::recheck::uploaded(id, source);
            }
        });
    }
    let receipt = crate::receipts::on().then(|| staging.clone());
    list.defer_host(move || {
        let waiting = on_demand.uploaded(staging.detiled.load(Relaxed), staging.memory as u64);
        staging.waiting.store(if waiting { crate::host::moment().max(1) } else { 0 }, Relaxed);
        // A texture picked by index has no staging buffer after this
        // upload: its slots are made due again by the texture pass.
        if waiting {
            crate::descriptor::host::levels_waiting(staging.memory as u64);
        }
    });
    // With receipts: after the levels were given their tiles, so a level
    // left without them is known.
    if let Some(staging) = receipt {
        install_receipt(list, resource, &staging);
    }
}

/// With receipts (`crate::receipts`): when `list` is submitted, after the
/// refresh its copy of `staging` follows, the receipt of what that copy
/// writes into `resource`'s host copy is installed. In the list's order of
/// steps, so on the queue's: an older copy submitted later lowers it.
fn install_receipt(list: &HostList, resource: This, staging: &Arc<Staging>) {
    let staging = staging.clone();
    let host = crate::resource::host_weak(resource);
    let this = resource as usize;
    list.defer_host(move || {
        use std::sync::atomic::Ordering::Relaxed;
        let short = staging.on_demand && staging.waiting.load(Relaxed) != 0;
        let read = (staging.memory as usize, staging.size, staging.first);
        crate::receipts::installed(this, staging.receipt(), read, host.clone(), short);
    });
}

/// Levels on demand: the levels a list claimed for its zeros
/// (`crate::tiles::Tiled::claim`), held by the list's submission-time
/// step. A list that is reset without having been submitted gives the
/// claim up, or the next list that writes those levels would send no zeros
/// and what its write does not cover would be what the tiles' last owner
/// left.
struct Claimed {
    on_demand: crate::resource::OnDemand,
    fresh: u32,
    ran: std::sync::atomic::AtomicBool,
}

impl Drop for Claimed {
    fn drop(&mut self) {
        if self.fresh != 0
            && !*self.ran.get_mut()
            && let Some(tiled) = self.on_demand.tiled()
        {
            tiled.unclaim(self.fresh);
        }
    }
}

/// Levels on demand (`crate::tiles`): the card itself is about to write
/// title texture `resource` in `list` (a copy or a resolve into level
/// `level`, or into every level). If its host copy `texture` was made
/// without memory, the levels written get theirs when the list is
/// submitted, ahead of it on the queue; the ones that have none yet are
/// first written with zeros, here in the same list, so that what the
/// card's write does not cover is not what the tiles' last owner left.
/// To be called before the list is brought to the state of that write.
/// shortcut: two lists that write the same new level, submitted in the
/// other order than recorded, lose the first one's write under the zeros
/// of the other; so does an upload recorded and not yet submitted.
pub(crate) fn card_writes(list: &HostList, resource: This, texture: &ID3D12Resource, level: Option<u32>) {
    let Some(on_demand) = crate::resource::host_tiled(resource) else {
        return;
    };
    let Some(tiled) = on_demand.tiled() else {
        return;
    };
    let levels = match level {
        Some(level) => tiled.with_level(level),
        None => tiled.every_level(),
    };
    if levels == 0 {
        return;
    }
    let fresh = tiled.claim(levels);
    // Of them, the ones whose zeros could not be recorded.
    let mut unzeroed = if host().is_none() { fresh } else { 0 };
    if fresh != 0
        && let Some(host) = host()
    {
        const COPY_DEST: u32 = 0x400;
        crate::list::ensure_state(list, resource, COPY_DEST);
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        for level in (0..u32::from(d.MipLevels).min(32)).filter(|&level| fresh & (1u32 << level) != 0) {
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut bytes = 0u64;
            // SAFETY: outputs for one subresource.
            unsafe {
                host.device
                    .GetCopyableFootprints(&d, level, 1, 0, Some(&mut footprint), None, None, Some(&mut bytes));
            }
            let Some((buffer, data)) = list.upload(host, bytes) else {
                // (With the levels that share its tiles, if it is one of
                // the small ones.)
                unzeroed |= tiled.with_level(level);
                continue;
            };
            // SAFETY: a mapped buffer of `bytes` bytes, just made.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: level },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above; the list holds the buffer until it is reset.
                pResource: unsafe { std::mem::transmute_copy(&buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
    }
    // A new level without its zeros gets no tiles from this list: the
    // card's write into it is lost (the level reads zeros, as before the
    // write), which is less wrong than a part write over what the tiles'
    // last owner left. The next list that writes it claims it again.
    unzeroed &= fresh;
    if unzeroed != 0 {
        tiled.unclaim(unzeroed);
    }
    let levels = levels & !unzeroed;
    if levels == 0 {
        return;
    }
    let memory = crate::resource::texture_memory(resource).map_or(0, |(memory, _)| memory as u64);
    let step = Claimed {
        on_demand,
        fresh: fresh & !unzeroed,
        ran: std::sync::atomic::AtomicBool::new(false),
    };
    list.defer_host(move || {
        step.ran.store(true, std::sync::atomic::Ordering::Relaxed);
        step.on_demand.card_wrote(levels, memory);
    });
}

impl Staging {
    /// Levels on demand: what says, beside the mapping, which levels of
    /// the texture are the title's now. First the lowest level a live view
    /// of the texture shows (0 without a view that says): the title's own
    /// word, a texture it holds from level 3 on is viewed from level 3.
    /// Then where the next texture placed in this one's span starts, in
    /// bytes from this one's start (`u64::MAX`: none does): for a texture
    /// viewed from level 0 although a neighbour sits where its large
    /// levels would be.
    fn judgement(&self) -> [u64; 2] {
        let start = self.memory as u64;
        let first = crate::descriptor::host::first_viewed(start).map_or(0, u64::from);
        let limit = crate::descriptor::host::next_start(start, start.saturating_add(self.layout.size))
            .map_or(u64::MAX, |at| at - start);
        [first, limit]
    }

    /// What [`Staging::judgement`] said at the last detiling.
    fn judged(&self) -> [u64; 2] {
        use std::sync::atomic::Ordering::Relaxed;
        [self.judged[0].load(Relaxed), self.judged[1].load(Relaxed)]
    }

    /// Levels on demand: the levels that are the title's now, a bit a
    /// level. A level is held if no view starts above it, its bytes end
    /// before the next texture starts, and all of them are mapped. The
    /// smallest levels share the texture's first block and are held
    /// together. (The console layout: that block first, then the other
    /// levels from the smallest up, level 0 last:
    /// `victory_formats::texture::Layout`.) A texture this cannot tell
    /// the levels of holds them all.
    fn held_levels(&self, [first, limit]: [u64; 2]) -> u32 {
        let mips = &self.layout.mips;
        if self.layout.slices != 1 || mips.is_empty() || mips.len() > 32 {
            return u32::MAX;
        }
        let tail = (self.layout.first_tail_mip as usize).min(mips.len());
        let there = |from: u64, to: u64| {
            to > from && to <= limit && committed(self.memory.wrapping_add(from as usize), (to - from) as usize)
        };
        let mut held = 0u32;
        for (level, m) in mips.iter().enumerate().take(tail) {
            let end = if level == 0 { self.layout.slice_size } else { mips[level - 1].offset };
            if level as u64 >= first && there(m.offset, end) {
                held |= 1 << level;
            }
        }
        if tail < mips.len() && there(0, self.layout.alignment.min(self.layout.slice_size)) {
            for level in tail..mips.len() {
                if level as u64 >= first {
                    held |= 1 << level;
                }
            }
        }
        held
    }

    /// Forgets what the staging buffer was last filled from: the next
    /// refresh detiles anew, whether the memory changed or not.
    fn forget(&self) {
        *self.last.lock().unwrap_or_else(|e| e.into_inner()) = Last::default();
    }

    /// The title memory the texture is read from: address and bytes.
    pub(crate) fn memory(&self) -> (usize, usize) {
        (self.memory as usize, self.size)
    }

    /// The write tracking's asking at the last check (0: not tracked).
    pub(crate) fn asked(&self) -> u64 {
        self.last.lock().unwrap_or_else(|e| e.into_inner()).asked
    }

    /// With receipts: the staging buffer's receipt ([`Last::receipt`]).
    fn receipt(&self) -> u64 {
        self.last.lock().unwrap_or_else(|e| e.into_inner()).receipt
    }

    /// What the upload recheck remembers of this staging buffer, or none for
    /// one whose staging buffer is not a whole copy of its source range
    /// (levels on demand, the texture pack), and for one not detiled while
    /// the trap was on in its present epoch (what it holds is not known).
    pub(crate) fn recheck_source(&self) -> Option<crate::host::recheck::Source> {
        self.recheck_source_in(crate::host::recheck::epoch())
    }

    /// [`Staging::recheck_source`] in the trap's `epoch`.
    fn recheck_source_in(&self, epoch: u64) -> Option<crate::host::recheck::Source> {
        if self.on_demand || self.shape.is_some() {
            return None;
        }
        let hash = match *self.detiled_hash.lock().unwrap_or_else(|e| e.into_inner()) {
            Some((taken, hash)) if taken == epoch => hash,
            _ => return None,
        };
        Some(crate::host::recheck::Source {
            // The caller knows the resource; the record fills it in.
            id: 0,
            start: self.memory as usize,
            size: self.size,
            format: self.format,
            mip: self.first,
            asked: self.asked(),
            hash: Some(hash),
            placed: Some(self.placed),
        })
    }

    /// The mip and the array slice of the `i`th part of the staging buffer.
    fn part(&self, i: usize) -> (u32, u32) {
        let mips = self.mips - self.first;
        (self.first + i as u32 % mips, i as u32 / mips)
    }

    /// The texture's subresource the `i`th part of the staging buffer is.
    fn subresource(&self, i: usize) -> u32 {
        let (mip, slice) = self.part(i);
        slice * self.mips + mip
    }

    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        #[cfg(feature = "diagnostics")]
        if crate::bind_trace::on() {
            crate::bind_trace::note_check(self.resource);
        }
        let started = std::time::Instant::now();
        let read = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        let whole = std::time::Instant::now();
        let read = self.refresh_timed();
        REFRESH_WAITS[3].fetch_add(whole.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
        read
    }

    fn refresh_timed(&self) -> usize {
        self.refresh_as(texture_digest(), through_copy())
    }

    /// [`Staging::refresh_timed`], its checks hashing with the digest if
    /// `digest` ([`texture_digest`]), and the all-mapped check reading
    /// through the kernel's copy if `copy` ([`through_copy`]); both given
    /// here so a test can choose.
    fn refresh_as(&self, digest: bool, copy: bool) -> usize {
        let size = self.size;
        let start = self.memory as usize;
        let mut last = waited(0, || self.last.lock().unwrap_or_else(|e| e.into_inner()));
        let last = &mut *last;
        // The texture pack finished a picture since this texture's last
        // detiling found it being prepared: detiled anew, whatever changed.
        if self.shape.is_some() {
            use std::sync::atomic::Ordering::Relaxed;
            let wait = self.pack_wait.load(Relaxed);
            if wait != 0 && crate::texture_pack::generation() + 1 != wait {
                self.pack_wait.store(0, Relaxed);
                *last = Last::default();
            }
        }
        // Receipts (`crate::receipts`): the journal's newest number when the
        // range was asked about is the receipt of a read that follows; a
        // check that finds nothing written since the last receipt moves it
        // there too (nothing stamped in between).
        let receipts = crate::receipts::on();
        // With receipts a check skips the read only if nothing is stamped above
        // the receipt: `asked` can stand past bytes never read (the partial
        // path's wait), and a skip by it copied the old staging buffer again.
        let unread = |written: &[(usize, u64)], asked: u64, receipt: u64| {
            if receipts {
                receipt != 0 && written.iter().all(|&(_, w)| w <= receipt)
            } else {
                asked != 0 && written.iter().all(|&(_, w)| w <= asked)
            }
        };
        if let Some((now, written, top)) = waited(1, || crate::watch::writes_top(start, start + size, false)) {
            if unread(&written, last.asked, last.receipt) {
                if receipts {
                    last.receipt = top;
                }
                return 0;
            }
            last.asked = now;
            if receipts {
                last.receipt = top;
            }
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            with_read(size, |read| {
                crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                self.detile(read);
            });
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return size;
        }
        // View memory has no write-watch, and a texture there was hashed
        // whole at every check: hundreds of megabytes and tens of milliseconds a frame
        // in a gameplay view where two textures change. It is guarded now, as buffers
        // in view memory are: a write of the title's to it faults once and
        // is noted, a mapping is announced by the runtime. While neither
        // happened the texture is as it was. What the guard cannot see (a
        // write to the same pages through another view) is still found by
        // the hash, which is taken every eighth frame only, by the
        // texture's address, so the textures take turns.
        // `VICTORY_TEXTURE_GUARD=0` hashes at every check, as before.
        static GUARD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        // A texture that changed without the guard seeing it is hashed at
        // every check for a while (during video playback: three pictures in turn
        // that only changed every eighth frame, one old picture after
        // another on the screen).
        let mut unseen = false;
        // With receipts: the receipt of the read below (0: not asked yet).
        let mut read_at = 0;
        if *GUARD.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_GUARD").is_none_or(|v| v != "0"))
            && let Some((now, written, top)) = waited(1, || crate::watch::writes_top(start, start + size, true))
        {
            const HASH_EVERY: u64 = 8;
            let unwritten = unread(&written, last.asked, last.receipt);
            last.asked = now;
            let frame = crate::frame::current();
            let turn = frame.wrapping_add((start >> 16) as u64).is_multiple_of(HASH_EVERY);
            // Linear pictures of one level are what the processor draws (the
            // planes of a movie), through a view of its own.
            let desc = &self.layout.desc;
            let film = desc.swizzle.0 == 0 && desc.mips == 1;
            if unwritten && !turn && !film && frame >= last.unguarded {
                if receipts {
                    last.receipt = top;
                }
                return 0;
            }
            unseen = unwritten;
            read_at = top;
        }
        if receipts && read_at == 0 {
            // Memory nobody watches: the number before the read. The
            // runtime's writes are stamped; the title's own stores are seen
            // only by the hash below.
            read_at = crate::watch::top();
            crate::receipts::STATS[crate::receipts::UNWATCHED].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // From here on the texture's memory may be read in place. Held until
        // that read is done: the runtime changes the mapping of these bytes
        // only when no read holds them, so what `committed` finds mapped
        // stays mapped (see `crate::watch::title_memory_reading`; without
        // it a page went between the asking and the read in runs 71 and
        // 74).
        let _mapped = if no_wait() {
            // Not waited for: while the runtime remaps title memory the
            // texture is read through the kernel instead, whole or not at
            // all (`no_wait`).
            let Some(mapped) = crate::watch::title_memory_reading_now(start, size) else {
                return self.refresh_unlocked(last, size, read_at, digest);
            };
            mapped
        } else {
            waited(2, || crate::watch::title_memory_reading(start, size))
        };
        if !committed(self.memory, size) {
            // Partly mapped: a streamed texture with only some mips' pages
            // there, the usual state while it streams in. Read through the
            // copy, which stands unmapped pages (zeros), and upload what
            // is there; nothing at all mapped changes nothing.
            // shortcut: read again at most every 30 frames (reading around
            // unmapped pages is slow, and nothing tells us when pages
            // arrive): a mip that streams in shows up to 30 frames late.
            let frame = crate::frame::current();
            if last.partial.is_some_and(|at| frame < at + 30) {
                // With receipts: a named blocker, the receipt left as it was.
                if receipts {
                    crate::receipts::STATS[crate::receipts::PARTIAL_SKIPS]
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                return 0;
            }
            last.partial = Some(frame);
            if receipts {
                last.receipt = read_at;
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
            let changed = with_read(size, |read| {
                crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                let hash = Sum::of(read, digest);
                if last.hash == Some(hash) {
                    return false;
                }
                last.hash = Some(hash);
                if read.iter().all(|&b| b == 0) {
                    return false;
                }
                self.detile(read);
                true
            });
            #[cfg(feature = "diagnostics")]
            if changed {
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
            }
            let _ = changed;
            return size;
        }
        // With receipts: what follows reads all of it (a hash the same as
        // the last one leaves bytes that are what memory holds).
        if receipts {
            last.receipt = read_at;
        }
        if copy {
            return with_title(self.memory, size, |tiled| {
                let hash = Sum::of(tiled, digest);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
                if last.hash == Some(hash) {
                    return size;
                }
                if unseen && last.hash.is_some() {
                    self.unseen_write(last);
                }
                last.hash = Some(hash);
                self.detile(tiled);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
                size
            });
        }
        // SAFETY: committed memory of the texture's layout size, and it
        // stays committed while `_mapped` is held.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = Sum::of(tiled, digest);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return size;
        }
        if unseen && last.hash.is_some() {
            self.unseen_write(last);
        }
        last.hash = Some(hash);
        self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        size
    }

    /// The texture's memory changed and the guard says nothing wrote it: it
    /// is hashed at every check for the next frames, the longer the more
    /// often this happens (a texture streamed in once is soon guarded
    /// again, a movie's is not).
    fn unseen_write(&self, last: &mut Last) {
        let frame = crate::frame::current();
        let more = if frame < last.unguarded { 120 } else { 16 };
        #[cfg(feature = "diagnostics")]
        {
            use windows_sys::Win32::System::Memory::MEMORY_BASIC_INFORMATION;
            static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 24 {
                // SAFETY: plain data filled in by the query.
                let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                #[cfg(feature = "diagnostics")]
                crate::stalls::kernel(crate::stalls::Kernel::Query);
                crate::kernelstat::virtual_query(crate::kernelstat::Place::Texture, self.memory as usize, &mut info);
                tracing::info!(
                    frame,
                    memory = ?self.memory,
                    size = self.layout.size,
                    protection = format_args!("{:#x}", info.Protect),
                    kind = format_args!("{:#x}", info.Type),
                    run = info.RegionSize,
                    desc = ?self.layout.desc,
                    "a guarded texture changed and no write to it was seen"
                );
            }
        }
        last.unguarded = frame + more;
    }

    /// A refresh while the runtime is remapping title memory ([`no_wait`]):
    /// the texture is read through the kernel, which fails if a page of it
    /// is not there at this moment. Then nothing is taken and the texture
    /// is looked at again at its next refresh (what was learnt about writes
    /// to it is forgotten, so that it is not passed over as unwritten).
    /// Returns the bytes read. `receipt`: with receipts, the number taken
    /// before this read (0: none), the staging buffer's receipt if it reads
    /// (a failed read keeps the old one). `digest`: [`texture_digest`].
    fn refresh_unlocked(&self, last: &mut Last, size: usize, receipt: u64, digest: bool) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let read = with_read(size, |read| {
            if !crate::host::read_into(self.memory, read) {
                return false;
            }
            let hash = Sum::of(read, digest);
            if last.hash != Some(hash) {
                last.hash = Some(hash);
                self.detile(read);
            }
            true
        });
        if read {
            UNLOCKED[0].fetch_add(1, Relaxed);
            if receipt != 0 {
                last.receipt = receipt;
            }
            size
        } else {
            UNLOCKED[1].fetch_add(1, Relaxed);
            last.asked = 0;
            0
        }
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        self.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
        if crate::host::recheck::on() {
            *self.detiled_hash.lock().unwrap_or_else(|e| e.into_inner()) =
                Some((crate::host::recheck::epoch(), blake3::hash(tiled)));
        }
        // Levels on demand: which levels are the title's is judged here,
        // once, for what the staging buffer holds from now on. The step
        // that gives levels their memory goes by this answer.
        let held = if self.on_demand {
            use std::sync::atomic::Ordering::Relaxed;
            let judged = self.judgement();
            self.judged[0].store(judged[0], Relaxed);
            self.judged[1].store(judged[1], Relaxed);
            // The mips before the staging buffer's first are not in it
            // and no list copies them (`staged_part`): never held, so
            // never given the card's memory for a copy that is not made.
            let held = self.held_levels(judged) & (u32::MAX << self.first.min(31));
            self.detiled.store(held, Relaxed);
            held
        } else {
            u32::MAX
        };
        #[cfg(feature = "diagnostics")]
        let zero = tiled.iter().all(|&b| b == 0);
        #[cfg(feature = "diagnostics")]
        if zero {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        #[cfg(feature = "diagnostics")]
        let mut failed = false;
        // The texture pack, on only when configured (`crate::texture_pack`;
        // off, a texture has no shape), for a texture it maps or logs: a
        // pass of its own. Any other texture: the loop as without it.
        if let Some((look, shape)) = self.shape.and_then(|shape| Some((crate::texture_pack::look(shape)?, shape))) {
            let _failed = self.detile_packed(&look, &shape, tiled, held);
            #[cfg(feature = "diagnostics")]
            {
                failed = _failed;
            }
        } else {
            for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
                let (mip, slice) = self.part(i);
                let f = footprint.Footprint;
                let pitch = f.RowPitch as usize;
                let len = pitch * rows as usize * f.Depth as usize;
                // SAFETY: the subresource's part of the mapped staging buffer.
                let linear = unsafe {
                    std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
                };
                if mip < 32 && held & (1 << mip) == 0 {
                    // Not the title's: zeros, not what lies in that memory.
                    linear.fill(0);
                    continue;
                }
                if let Err(e) =
                    self.layout
                        .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
                {
                    tracing::warn!(?e, mip, slice, "texture not detiled");
                    #[cfg(feature = "diagnostics")]
                    {
                        failed = true;
                    }
                    break;
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        if crate::bind_trace::on() {
            crate::bind_trace::note_detile(self.resource, tiled.len() as u64, zero, failed);
        }
    }

    /// [`Staging::detile`] with the texture pack, for a texture of `shape`
    /// it maps or logs: every part is detiled into a buffer of the pack's
    /// laid out as the staging buffer, which gives the texture's identity
    /// (only if every level from level 0 is the title's and was detiled). A
    /// mapped identity whose picture is ready and takes every part has the
    /// picture written into the staging buffer, every part of it; else the
    /// parts detiled go in as the title's texels, as without the pack. A
    /// picture still being prepared: the texture is detiled anew once it is
    /// ([`Staging::pack_wait`]). True if a part was not detiled.
    fn detile_packed(
        &self,
        look: &crate::texture_pack::Look,
        shape: &crate::texture_pack::Shape,
        tiled: &[u8],
        held: u32,
    ) -> bool {
        use crate::texture_pack::{Identity, Replacement};
        use std::sync::atomic::Ordering::Relaxed;
        struct Part {
            mip: u32,
            at: usize,
            len: usize,
            pitch: usize,
            rows: usize,
        }
        let parts: Vec<Part> = self
            .footprints
            .iter()
            .zip(&self.rows)
            .enumerate()
            .map(|(i, (footprint, &rows))| {
                let f = footprint.Footprint;
                let pitch = f.RowPitch as usize;
                Part {
                    mip: self.part(i).0,
                    at: footprint.Offset as usize,
                    len: pitch * rows as usize * f.Depth as usize,
                    pitch,
                    rows: rows as usize,
                }
            })
            .collect();
        // (The last part's rows may end short of a whole pitch in the staging
        // buffer's bytes; the buffer laid out like it is as long as the parts.)
        let len = parts.iter().map(|p| p.at + p.len).max().unwrap_or(0);
        crate::texture_pack::with_scratch(len, |scratch| {
            let (mut done, mut whole, mut failed) = (0, self.first == 0, false);
            for (i, part) in parts.iter().enumerate() {
                let (mip, slice) = self.part(i);
                let linear = &mut scratch[part.at..part.at + part.len];
                if mip < 32 && held & (1 << mip) == 0 {
                    // Not the title's: zeros, not what lies in that memory.
                    linear.fill(0);
                    whole = false;
                    done = i + 1;
                    continue;
                }
                if let Err(e) = self.layout.detile(tiled, mip, slice, linear, part.pitch, part.pitch * part.rows) {
                    tracing::warn!(?e, mip, slice, "texture not detiled");
                    failed = true;
                    break;
                }
                done = i + 1;
            }
            // Every level's texels, rows without their padding.
            let element = self.layout.desc.element_bytes as usize;
            let levels: Option<Vec<(&[u8], usize, usize, usize)>> = parts
                .iter()
                .map(|part| {
                    let m = self.layout.mips.get(part.mip as usize)?;
                    let row = (m.width as usize * element).min(part.pitch);
                    Some((&scratch[part.at..part.at + part.len], part.pitch, part.rows.min(m.height as usize), row))
                })
                .collect();
            let identity = levels.filter(|_| whole && !failed).map(|levels| Identity::of(shape, levels));
            #[cfg(feature = "diagnostics")]
            if let Some(id) = &identity {
                crate::texture_pack::detiled(self.resource, id);
            }
            let asked = crate::texture_pack::generation();
            let replacement = identity.map_or(Replacement::None, |id| look.replacement(&id));
            let picture = match replacement {
                Replacement::Ready(picture) => {
                    let takes = picture.fits(parts.iter().map(|p| (p.mip, p.len, p.pitch, p.rows)));
                    if !takes {
                        once!(?identity, "a texture pack picture not of the staging buffer's sizes: the title's texels kept");
                    }
                    takes.then_some(picture)
                }
                Replacement::Pending => {
                    self.pack_wait.store(asked + 1, Relaxed);
                    None
                }
                Replacement::None => None,
            };
            for part in &parts[..done] {
                // SAFETY: the part of the mapped staging buffer.
                let linear = unsafe { std::slice::from_raw_parts_mut(self.data.add(part.at), part.len) };
                match &picture {
                    Some(picture) => {
                        picture.write(part.mip, linear, part.pitch, part.rows);
                    }
                    None => linear.copy_from_slice(&scratch[part.at..part.at + part.len]),
                }
            }
            failed
        })
    }
}

/// Lets go of a title texture's staging buffer (lists that recorded an
/// upload from it hold it until they have run); the next upload makes one
/// anew.
pub(crate) fn unstage(resource: This) {
    let gone = {
        let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        map.as_mut().and_then(|map| map.remove(&(resource as usize)))
    };
    // The stale-upload trap keeps its last sample (after the table's lock).
    if let Some(staging) = gone
        && crate::host::recheck::on()
    {
        crate::host::recheck::let_go(resource as usize, staging.used.load(std::sync::atomic::Ordering::Relaxed) >> 32);
    }
}

/// The frame a list last sampled title texture `resource` (recorded an
/// upload of it, `upload_from`'s `used`), while it has a staging buffer: for
/// the stale-upload trap.
pub(crate) fn last_sampled(resource: usize) -> Option<u64> {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let staging = map.as_ref()?.get(&resource)?;
    Some(staging.used.load(std::sync::atomic::Ordering::Relaxed) >> 32)
}

/// Has the next list that uploads a title texture with a staging buffer
/// copy it whole, changed or not since: what the check at a submission
/// finds changed is copied by the next list that samples the texture
/// through a table, and there may be none for one that shaders now pick by
/// index alone.
pub(crate) fn upload_again(resource: This) {
    let staging = {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        map.as_ref().and_then(|map| map.get(&(resource as usize)).cloned())
    };
    if let Some(staging) = staging {
        staging.uploaded.store(false, std::sync::atomic::Ordering::Relaxed);
        staging.last.lock().unwrap_or_else(|e| e.into_inner()).partial = None;
    }
}

/// Where a staged title texture's memory is: address and bytes, and the
/// bytes all its mips take there (the staging buffer may hold the smaller
/// ones alone: `staged_part`).
pub(crate) fn staged_memory(resource: This) -> Option<(u64, u64, u64)> {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let staging = map.as_ref()?.get(&(resource as usize))?;
    let (at, len) = staging.memory();
    Some((at as u64, len as u64, staging.layout.size))
}

/// A fingerprint of the title memory `[at, at + len)`, never 0: of all of
/// it up to 48 KiB, else of twelve pieces of 4 KiB spread evenly over it,
/// the first and the last included. Read through the kernel, so pages not
/// there count as zeros. (A texture whose change falls between the pieces
/// is missed.) `whole`: the bytes all the texture's mips take from `at`,
/// of which `len` are the smaller mips a view shows; where a fingerprint
/// of `whole` takes a piece inside `len`, so does this one (up to twelve
/// more), so that a view's mips are looked at wherever they were while a
/// view's span was all of the texture.
pub(crate) fn fingerprint(at: u64, len: u64, whole: u64) -> u64 {
    fingerprint_as(at, len, whole, texture_digest())
}

/// [`fingerprint`], of blake3's hash of the pieces one after another, or
/// with `digest` ([`texture_digest`]) of their digest. The texture queue
/// compares a fingerprint only with one taken in the same run, so of the
/// same kind (the switch is read once).
fn fingerprint_as(at: u64, len: u64, whole: u64, digest: bool) -> u64 {
    const PIECE: usize = 4 << 10;
    const PIECES: usize = 12;
    thread_local! {
        static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    // Where the pieces of a fingerprint of `len` bytes start.
    let starts = |len: usize| {
        (0..PIECES - 1).map(move |n| ((len - PIECE) / (PIECES - 1) * n) & !0xfff).chain([len - PIECE])
    };
    let (len, whole) = (len as usize, whole as usize);
    READ.with(|read| {
        let mut read = read.borrow_mut();
        // With the digest the pieces lie one after another (up to twice
        // twelve), digested at the end; blake3 takes each as it is read.
        read.resize(2 * PIECES * PIECE, 0);
        let mut hasher = (!digest).then(blake3::Hasher::new);
        let mut filled = 0;
        let mut take = |from: usize, bytes: usize| {
            let piece = &mut read[filled..filled + bytes];
            piece.fill(0);
            crate::host::copy_mapped(piece.as_mut_ptr(), (at as usize + from) as *const u8, bytes, false);
            match &mut hasher {
                Some(hasher) => {
                    hasher.update(piece);
                }
                None => filled += bytes,
            }
        };
        if len <= PIECES * PIECE {
            take(0, len);
        } else if whole <= len {
            for from in starts(len) {
                take(from, PIECE);
            }
        } else {
            let mut from: Vec<usize> = starts(len).chain(starts(whole).filter(|&s| s < len)).collect();
            from.sort_unstable();
            from.dedup();
            for from in from {
                take(from, PIECE.min(len - from));
            }
        }
        // 0 says "none" to the texture queue (`Recheck`, `checked`).
        let Some(hasher) = hasher else {
            let d = crate::digest::digest(&read[..filled]);
            return (d as u64 ^ (d >> 64) as u64) | 1;
        };
        let hash = hasher.finalize();
        let mut first = [0u8; 8];
        first.copy_from_slice(&hash.as_bytes()[..8]);
        u64::from_le_bytes(first) | 1
    })
}

/// Whether a title texture has a staging buffer: a list recorded its
/// upload, and the lists that sample it check it at submission.
/// For the frame listing: whether a list has uploaded `resource`'s staging
/// buffer, and whether it was detiled since. None: it has none.
#[cfg(feature = "diagnostics")]
pub(crate) fn staged_state(resource: This) -> Option<(bool, bool)> {
    use std::sync::atomic::Ordering::Relaxed;
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let staging = map.as_ref()?.get(&(resource as usize))?;
    Some((staging.uploaded.load(Relaxed), staging.dirty.load(Relaxed)))
}

pub(crate) fn is_staged(resource: This) -> bool {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    map.as_ref().is_some_and(|map| map.contains_key(&(resource as usize)))
}

/// The bytes of a title texture's memory, and whether all of it is mapped,
/// if the texture has been staged for upload.
pub(crate) fn staged(resource: This) -> Option<(u64, bool)> {
    let s = {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        map.as_ref()?.get(&(resource as usize))?.clone()
    };
    Some((s.size as u64, committed(s.memory, s.size)))
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    if by_query() {
        return committed_by_query(at, len);
    }
    if len == 0 {
        return true;
    }
    let (there, of) = readable(at as usize, len, true);
    there == of
}

/// Whether the two questions above and below are put to `VirtualQuery`, as
/// they were (`VICTORY_COMMITTED_QUERY=1`). `VirtualQuery` answers with the
/// length of the run of pages in the same state, which it finds by walking
/// the page tables to the run's end: milliseconds for one question in the title's
/// largest block (round 6). The pass over the heap's views asks three or
/// more of them for every texture slot that is due, thousands of slots a frame
/// when the card has no room: that pass then ran tenths of a second, and
/// while it ran the submitting thread's write-watch askings waited behind
/// the walks (some askings a frame of many milliseconds each).
fn by_query() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COMMITTED_QUERY").is_some_and(|v| v == "1"))
}

/// Of the kernel's pages that hold `[at, at + len)` (`len > 0`): how many
/// can be read, and how many there are; with `all`, it stops at the first
/// that cannot. One question to the kernel for up to 256 pages at once
/// (`QueryWorkingSetEx`: the state of each page asked about, no walk); a
/// page not in the working set may still be there (never touched, or
/// trimmed), and a one-byte read through the kernel says.
/// A page that is committed and may not be read counts as not there, which
/// is what the callers ask for: they read or write the memory next.
fn readable(at: usize, len: usize, all: bool) -> (usize, usize) {
    use windows_sys::Win32::System::ProcessStatus::PSAPI_WORKING_SET_EX_INFORMATION;
    const KERNEL_PAGE: usize = 4096;
    let (first, last) = (at / KERNEL_PAGE, at.saturating_add(len - 1) / KERNEL_PAGE);
    let of = last - first + 1;
    let mut there = 0;
    // SAFETY: plain data.
    let mut info: [PSAPI_WORKING_SET_EX_INFORMATION; 256] = unsafe { std::mem::zeroed() };
    let mut page = first;
    while page <= last {
        let n = (last - page + 1).min(info.len());
        for (i, entry) in info[..n].iter_mut().enumerate() {
            entry.VirtualAddress = ((page + i) * KERNEL_PAGE) as *mut _;
        }
        // SAFETY: `n` entries of the size given; our own process.
        let asked = unsafe {
            crate::kernelstat::working_set(
                crate::kernelstat::Place::Texture,
                info.as_mut_ptr().cast(),
                (n * size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32,
                n,
            )
        };
        for (i, entry) in info[..n].iter().enumerate() {
            // Bit 0 of the answer: the page is in the working set.
            // SAFETY: plain data the query filled in.
            let valid = asked && unsafe { entry.VirtualAttributes.Flags } & 1 != 0;
            let mut byte = [0u8];
            if valid
                || crate::host::read_into_as(
                    ((page + i) * KERNEL_PAGE) as *const u8,
                    &mut byte,
                    crate::kernelstat::Caller::Readable,
                )
            {
                there += 1;
            } else if all {
                return (there, of);
            }
        }
        page += n;
    }
    (there, of)
}

fn committed_by_query(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        #[cfg(feature = "diagnostics")]
        crate::stalls::kernel(crate::stalls::Kernel::Query);
        let n = crate::kernelstat::virtual_query(crate::kernelstat::Place::Texture, p, &mut info);
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// How many bytes of `[at, at + len)` are committed memory: all of a
/// texture that is there, some of one streaming in (its smaller mips lie
/// at the start of its memory, and their pages are mapped first), none of
/// one not there. With neither the first nor the last page committed the
/// answer is none without a look at the pages between (two queries for the
/// slots whose texture is away, which are asked again and again), unless
/// the last page of `[at, at + reach)` is: `len` bytes are the smaller mips
/// a view shows of a texture whose mips take `reach`, and the pages between
/// are looked at whenever they were while a view's span was all of it.
pub(crate) fn mapped_bytes(at: *const u8, len: usize, reach: usize) -> u64 {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION};
    if len == 0 {
        return 0;
    }
    let start = at as usize;
    let end = start.saturating_add(len);
    if !committed(at, 1)
        && !committed((end - 1) as *const u8, 1)
        && (reach <= len || !committed(start.saturating_add(reach - 1) as *const u8, 1))
    {
        return 0;
    }
    if !by_query() {
        static BY_PAGE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *BY_PAGE.get_or_init(|| std::env::var_os("VICTORY_MAPPED_BY_PAGE").is_some_and(|v| v == "1")) {
            // By page, without the walk: whole pages, cut to the range.
            let (there, _) = readable(start, len, false);
            return (there as u64 * 4096).min(len as u64);
        }
        // By 64 KiB block, one page asked of each: the title's memory is
        // mapped in blocks of that size (the console's page), and the
        // answer is a measure of how much of a texture has arrived, not a
        // promise that every byte can be read (the upload reads through the
        // kernel). By page, a texture half there cost a failed kernel read
        // for each of its missing pages, every frame it stayed due: over a thousand
        // textures "uploaded in part" kept the pass over the heap's views at
        // a tenth of a second a frame. `VICTORY_MAPPED_BY_PAGE=1`: by page.
        const BLOCK: usize = 64 << 10;
        let mut there = 0u64;
        let mut block = start & !(BLOCK - 1);
        while block < end {
            let (lo, hi) = (block.max(start), block.saturating_add(BLOCK).min(end));
            if readable(lo, 1, true).0 == 1 {
                there += (hi - lo) as u64;
            }
            block = block.saturating_add(BLOCK);
        }
        return there;
    }
    let (mut p, mut there) = (start, 0u64);
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        #[cfg(feature = "diagnostics")]
        crate::stalls::kernel(crate::stalls::Kernel::Query);
        let n = crate::kernelstat::virtual_query(crate::kernelstat::Place::Texture, p, &mut info);
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize).min(end);
        if n == 0 || next <= p {
            break;
        }
        if info.State == MEM_COMMIT {
            there += (next - p) as u64;
        }
        p = next;
    }
    there
}

/// Of the mips a placed texture's view shows (the texture's resource
/// description and the view, as words) at `address`: their bytes, and how
/// many of them are mapped ([`mapped_bytes`], which also looks between
/// their ends when the end of all the texture's memory is mapped).
pub(crate) fn view_mapped(address: u64, resource: &[u32; 14], view: &[u32]) -> (u64, u64) {
    let len = crate::resource::placed_bytes_from(resource, crate::resource::first_mip(resource, view)).unwrap_or(1);
    let whole = crate::resource::placed_bytes(resource).unwrap_or(len).max(len);
    (len, mapped_bytes(address as *const u8, len as usize, whole as usize))
}

/// The first mip a staging buffer holds and the bytes of title memory read
/// for it and the smaller ones, for a texture of `mips` mips a view shows
/// from `first` on: from `first` only where the layout can end behind the
/// smaller mips (one 2-D slice: `bytes_from` is the whole layout for any
/// other), else all of it.
fn staged_part(layout: &victory_formats::texture::Layout, mips: u32, first: u32) -> (u32, usize) {
    let first = if first < mips && layout.bytes_from(first) < layout.size { first } else { 0 };
    (first, layout.bytes_from(first) as usize)
}

/// Tests: no staging buffer is to be had ([`take_staging`] refuses), as on a
/// card or a process out of memory.
#[cfg(test)]
pub(crate) static FAIL_STAGING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The staging state of a title texture from mip `first` on, made on first
/// use (and made anew for larger mips than the one there holds).
fn staging(resource: This, texture: &ID3D12Resource, first: u32) -> Option<Arc<Staging>> {
    // Asked before the table is locked: the sweep lets go of a staging
    // buffer with its texture's host side locked, in that order.
    let on_demand = crate::resource::host_tiled(resource).is_some();
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize))
        && s.first <= first
    {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        // With receipts: never uploaded from title memory, so not owed.
        if crate::receipts::on() {
            crate::receipts::never(resource as usize);
        }
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        if crate::receipts::on() {
            crate::receipts::never(resource as usize);
        }
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let (first, size) = staged_part(&layout, u32::from(d.MipLevels), first);
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize) - first
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            first,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let bytes = staging_bytes(total);
    #[cfg(test)]
    let staged = if FAIL_STAGING.load(std::sync::atomic::Ordering::Relaxed) { None } else { take_staging(host, bytes, total) };
    #[cfg(not(test))]
    let staged = take_staging(host, bytes, total);
    let Some((buffer, data)) = staged else {
        // With receipts: a named blocker of the texture's debt, until an install.
        if crate::receipts::on() {
            crate::receipts::no_staging(resource as usize);
        }
        return None;
    };
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    STAGED[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    STAGED[1].fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
    let s = Arc::new(Staging {
        buffer,
        #[cfg(feature = "diagnostics")]
        total,
        #[cfg(feature = "diagnostics")]
        resource: resource as usize,
        data,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        first,
        size,
        format: d.Format.0 as u32,
        detiled_hash: Mutex::new(None),
        placed: crate::resource::placed_resource(resource).unwrap_or(false),
        bytes,
        memory,
        last: Mutex::default(),
        dirty: std::sync::atomic::AtomicBool::new(true),
        uploaded: std::sync::atomic::AtomicBool::new(false),
        used: std::sync::atomic::AtomicU64::new(crate::host::moment()),
        on_demand,
        detiled: std::sync::atomic::AtomicU32::new(0),
        judged: [const { std::sync::atomic::AtomicU64::new(0) }; 2],
        waiting: std::sync::atomic::AtomicU64::new(0),
        pack_wait: std::sync::atomic::AtomicU64::new(0),
        shape: (crate::texture_pack::on() && d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.DepthOrArraySize == 1).then(|| {
            crate::texture_pack::Shape {
                width: d.Width as u32,
                height: d.Height,
                format: d.Format.0 as u32,
                mips: u32::from(d.MipLevels),
            }
        }),
    });
    map.insert(resource as usize, s.clone());
    Some(s)
}

#[cfg(test)]
mod spare_tests {
    use super::*;

    /// A staging buffer is made at most an eighth larger than asked (or 64
    /// KiB), never smaller, and nearby sizes come out the same.
    #[test]
    fn staging_sizes_round_up_by_an_eighth_at_most() {
        for total in [1u64, 4096, 65_536, 65_537, 300_000, 1 << 20, (1 << 20) + 1, 5_640_000, 16 << 20, (16 << 20) + 3] {
            let bytes = staging_bytes(total);
            let step = (total.next_power_of_two() / 8).max(64 << 10);
            assert!(bytes >= total && bytes - total < step, "{total}: {bytes}");
        }
        assert_eq!(staging_bytes(290_000), staging_bytes(300_000));
        assert_eq!(staging_bytes(5_600_000), staging_bytes(5_640_000));
    }

    /// A staging buffer no texture holds is taken again by the next staging
    /// of its size: the same buffer and mapping, zeroed where the new
    /// texture's bytes go, as a new buffer is. A size of another class makes
    /// a new one; spare buffers idle long enough are let go.
    #[test]
    fn a_staging_buffer_let_go_is_taken_again_zeroed() {
        let host = crate::host::test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        // A size no other test stages, so that none takes it in between.
        let (total, again) = (9_000_001u64, 8_900_000u64);
        let bytes = staging_bytes(total);
        assert_eq!(staging_bytes(again), bytes);
        let (buffer, data) = take_staging(host, bytes, total).expect("a staging buffer");
        // SAFETY: a live buffer; its mapping holds `bytes`.
        let address = unsafe { buffer.GetGPUVirtualAddress() };
        unsafe { data.write_bytes(0xab, bytes as usize) };
        keep_spare(buffer, data, bytes);
        let (taken, mapping) = take_staging(host, bytes, again).expect("a staging buffer");
        // SAFETY: as above.
        assert_eq!(unsafe { taken.GetGPUVirtualAddress() }, address, "the spare buffer");
        assert_eq!(mapping, data);
        // SAFETY: the mapping, `bytes` long.
        let mapped = unsafe { std::slice::from_raw_parts(mapping, bytes as usize) };
        assert!(mapped[..again as usize].iter().all(|&b| b == 0), "zeroed as a new one is");
        let (other, _) = take_staging(host, staging_bytes(2 * total), 2 * total).expect("a staging buffer");
        // SAFETY: as above.
        assert_ne!(unsafe { other.GetGPUVirtualAddress() }, address, "another size: a new buffer");
        keep_spare(taken, mapping, bytes);
        assert!(SPARE.lock().unwrap().by_size.get(&bytes).is_some_and(|k| !k.is_empty()));
        let_go_spare(0, 0);
        assert!(!SPARE.lock().unwrap().by_size.contains_key(&bytes), "let go when idle");
    }

    /// The card's fence, signalled when this is dropped (also when the test
    /// fails first): what the held queue waits for.
    #[cfg(feature = "diagnostics")]
    struct Gate(ID3D12Fence);

    // SAFETY: D3D12 fences are free-threaded.
    #[cfg(feature = "diagnostics")]
    unsafe impl Send for Gate {}

    #[cfg(feature = "diagnostics")]
    impl Drop for Gate {
        fn drop(&mut self) {
            // SAFETY: a live fence.
            let _ = unsafe { self.0.Signal(1) };
        }
    }

    /// A staging buffer is not taken while an upload from it is in flight.
    /// A texture's staging (in the table, as `staging` makes it) is recorded
    /// as a list's upload, a copy out of its buffer, and the list holds it
    /// as `upload_from` does; the queue is held on a fence, so the card has
    /// not run the copy. The texture lets go of its staging (`unstage`):
    /// the buffer is not spare, and a staging of its size gets another.
    /// Once the fence is signalled the list runs, and until it is reset the
    /// buffer is still not spare. The copy read the bytes written before it
    /// was sent. After the reset the next staging of its size takes the
    /// buffer, zeroed.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_staging_buffer_is_not_taken_while_an_upload_from_it_is_in_flight() {
        use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
        let host = crate::host::test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let _alone = crate::host::queue_alone();
        // A size no other test stages.
        let total = 7_000_003u64;
        let bytes = staging_bytes(total);
        let spare = move || SPARE.lock().unwrap().by_size.get(&bytes).map_or(0, Vec::len);
        assert_eq!(spare(), 0);
        let (buffer, data) = take_staging(host, bytes, total).expect("a staging buffer");
        // SAFETY: the buffer's mapping holds `bytes`.
        unsafe { data.write_bytes(0x5c, total as usize) };
        // SAFETY: a live buffer.
        let address = unsafe { buffer.GetGPUVirtualAddress() };
        let layout = victory_formats::texture::Layout::new(&victory_formats::texture::Desc {
            dimension: victory_formats::texture::Dimension::Tex2d,
            width: 64,
            height: 64,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: victory_formats::texture::Swizzle::LINEAR,
        })
        .unwrap();
        STAGED[0].fetch_add(1, Relaxed);
        STAGED[1].fetch_add(bytes, Relaxed);
        let staging = Arc::new(Staging {
            buffer,
            total,
            resource: 0,
            data,
            footprints: Vec::new(),
            rows: Vec::new(),
            layout,
            mips: 1,
            first: 0,
            size: 0,
            format: 0,
            detiled_hash: Mutex::new(None),
            placed: false,
            bytes,
            memory: std::ptr::null(),
            last: Mutex::default(),
            dirty: AtomicBool::new(true),
            uploaded: AtomicBool::new(false),
            used: AtomicU64::new(crate::host::moment()),
            on_demand: false,
            detiled: AtomicU32::new(0),
            judged: [const { AtomicU64::new(0) }; 2],
            waiting: AtomicU64::new(0),
            pack_wait: std::sync::atomic::AtomicU64::new(0),
            shape: None,
        });
        // The texture's: in the table under a key no title object has.
        let texture = 0x7e57_5a9e_usize;
        STAGING.lock().unwrap().get_or_insert_with(HashMap::new).insert(texture, staging.clone());
        // The upload: the buffer's last 4 KiB copied out, the list holding
        // the staging until it is reset.
        let back = host.read_back_buffer(4096).expect("a read-back buffer");
        let list = host.new_list().unwrap();
        // SAFETY: an open list; both buffers live, the ranges inside them.
        unsafe { list.list.CopyBufferRegion(&back, 0, &staging.buffer, total - 4096, 4096) };
        list.defer_host(move || {
            let _ = &staging;
        });
        list.close();
        // SAFETY: plain creation.
        let gate = Gate(unsafe { host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap());
        host.hold_queue(&gate.0, 1);
        let (sending, sent) = std::sync::mpsc::channel::<()>();
        let checker = std::thread::spawn(move || {
            let gate = gate;
            sent.recv().unwrap();
            // The list is with the card, held behind the fence.
            std::thread::sleep(std::time::Duration::from_millis(100));
            unstage(texture as This);
            let early = spare();
            let (other, _) = take_staging(host, bytes, total).expect("a staging buffer");
            // SAFETY: a live buffer.
            let other = unsafe { other.GetGPUVirtualAddress() };
            drop(gate);
            (early, other)
        });
        sending.send(()).unwrap();
        host.execute(&[&list]);
        let (early, other) = checker.join().unwrap();
        assert_eq!(early, 0, "spare while the upload was in flight");
        assert_ne!(other, address, "taken while the upload was in flight");
        assert_eq!(spare(), 0, "spare after the list ran, before its reset");
        let mut copied = std::ptr::null_mut();
        // SAFETY: a read-back buffer of 4 KiB, mapped whole.
        let copied = unsafe {
            back.Map(0, None, Some(&mut copied)).unwrap();
            std::slice::from_raw_parts(copied.cast::<u8>(), 4096)
        };
        assert!(copied.iter().all(|&b| b == 0x5c), "the copy read the bytes written before it was sent");
        list.reset();
        assert_eq!(spare(), 1, "spare once the list was reset");
        let (taken, mapping) = take_staging(host, bytes, total).expect("a staging buffer");
        // SAFETY: a live buffer; its mapping holds `bytes`.
        assert_eq!(unsafe { taken.GetGPUVirtualAddress() }, address, "the buffer taken again");
        let mapped = unsafe { std::slice::from_raw_parts(mapping, total as usize) };
        assert!(mapped.iter().all(|&b| b == 0), "zeroed as a new one is");
    }

    /// What a staging buffer costs made and taken from the spare ones (the
    /// software adapter: the card's driver maps the pages too):
    /// `VICTORY_D3D12_WARP=1 cargo test --release -p victory-gpu --features diagnostics staging_buffer_cost -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark"]
    fn staging_buffer_cost() {
        let host = crate::host::test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        for total in [512u64 << 10, 5_640_000, 22_400_000] {
            let bytes = staging_bytes(total);
            let runs = 20;
            let started = std::time::Instant::now();
            for _ in 0..runs {
                let (buffer, data) = host.upload_buffer(bytes).unwrap();
                // A new buffer's pages are the system's until first written.
                // SAFETY: its mapping, `total` long.
                unsafe { data.write_bytes(1, total as usize) };
                drop(buffer);
            }
            let made = started.elapsed().as_secs_f64() / f64::from(runs);
            let (buffer, data) = host.upload_buffer(bytes).unwrap();
            keep_spare(buffer, data, bytes);
            let started = std::time::Instant::now();
            for _ in 0..runs {
                let (buffer, data) = take_staging(host, bytes, total).unwrap();
                // SAFETY: as above.
                unsafe { data.write_bytes(1, total as usize) };
                keep_spare(buffer, data, bytes);
            }
            let kept = started.elapsed().as_secs_f64() / f64::from(runs);
            println!(
                "{:.2} MB staging: made and written {:.3} ms, taken from the spare ones and written {:.3} ms",
                total as f64 / 1e6,
                made * 1e3,
                kept * 1e3
            );
        }
        let_go_spare(0, 0);
    }
}

#[cfg(test)]
mod first_mip_tests {
    use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};

    fn layout(depth_or_array: u32) -> victory_formats::texture::Layout {
        let desc = TextureDesc {
            dimension: Dimension::Tex2d,
            width: 1024,
            height: 1024,
            depth_or_array,
            mips: 9,
            format: 71,
            samples: 1,
            swizzle: Swizzle::SW_4KB_S,
        };
        Texture::new(&desc).unwrap().planes[0].layout.clone()
    }

    /// A view from mip 4 of a 1024x1024 BC1 texture of 9 mips stages mips
    /// 4 to 8 and reads only the memory they take; a view of all of it, an
    /// array's view, or a first mip past the chain stage all of it.
    #[test]
    fn a_view_of_the_smaller_mips_stages_only_their_memory() {
        let l = layout(1);
        assert_eq!(super::staged_part(&l, 9, 4), (4, l.bytes_from(4) as usize));
        assert!(l.bytes_from(4) < l.size);
        assert_eq!(super::staged_part(&l, 9, 0), (0, l.size as usize));
        assert_eq!(super::staged_part(&l, 9, 9), (0, l.size as usize), "past the chain");
        let array = layout(2);
        assert_eq!(super::staged_part(&array, 9, 4), (0, array.size as usize), "slices each hold their chain");
    }

    /// A 2-D view's first mip is its fifth word, and so is a 2-D array
    /// view's of a texture of one slice; other views show all mips.
    #[test]
    fn a_2d_view_names_its_first_mip() {
        use crate::resource::first_mip;
        // 1024x1024 BC1 of 9 mips, of one slice and of two.
        let one = [3, 0, 0, 0, 1024, 0, 1024, 0x9_0001, 71, 1, 0, 0x105, 0, 0];
        let mut two = one;
        two[7] = 0x9_0002;
        assert_eq!(first_mip(&one, &[71, 4, 0x1688, 0, 4, 5, 0, 0, 0, 0]), 4);
        assert_eq!(first_mip(&two, &[71, 5, 0x1688, 0, 4, 5, 0, 0, 0, 0]), 0, "an array view");
        assert_eq!(first_mip(&one, &[71, 4, 0x1688, 0, 99, 5, 0, 0, 0, 0]), 15);
        // A made-up texture of the same shape as a streamed one: a 2-D
        // array view of one slice from mip 3 of a 256x256 BC7 texture of 7
        // mips, the next texture 8 KiB on. It shows mips 3 to 6 and takes
        // the memory a 2-D view from mip 3 takes, not the whole chain's.
        // (The words need only be a valid description of that shape: the
        // test reads the dimension, the mip count, the slice count and the
        // view's fifth word, and measures memory through `layout()`.)
        let streamed = [3, 0, 0, 0, 256, 0, 256, 0x7_0001, 98, 1, 0, 0x105, 0, 0];
        let array_view = [0x47, 5, 0x1688, 0, 3, 5, 0, 1, 0, 0];
        assert_eq!(first_mip(&streamed, &array_view), 3);
        let claimed = crate::resource::placed_bytes_from(&streamed, first_mip(&streamed, &array_view)).unwrap();
        let from_3 = crate::resource::placed_bytes_from(&streamed, first_mip(&streamed, &[0x47, 4, 0x1688, 0, 3, 5, 0, 0, 0, 0]));
        assert_eq!(Some(claimed), from_3, "what a 2-D view from mip 3 takes");
        assert!(claimed <= 0x2000, "mips 3 to 6 end before the next texture: {claimed:#x}");
        assert!(claimed < crate::resource::placed_bytes(&streamed).unwrap());
        // A cube's faces and a 3-D texture's depth hold the chain each: a
        // cube view (of six slices) and a 3-D view show all mips.
        let mut cube = streamed;
        cube[7] = 0x7_0006;
        assert_eq!(first_mip(&cube, &[0x47, 9, 0x1688, 0, 3, 5, 0, 0, 0, 0]), 0, "a cube view");
        assert_eq!(first_mip(&cube, &[0x47, 5, 0x1688, 0, 3, 5, 0, 1, 0, 0]), 0, "one face's array view");
        let volume = [4, 0, 0, 0, 64, 0, 64, 0x7_0040, 28, 1, 0, 0x105, 0, 0];
        assert_eq!(first_mip(&volume, &[28, 8, 0x1688, 0, 2, 5, 0, 0, 0, 0]), 0, "a 3-D view");
    }
}

#[cfg(test)]
mod title_memory_tests {
    use super::with_title;

    /// Title memory whose tail is reserved and not committed reads through
    /// the copy as its bytes and then zeros, without a fault; so does
    /// memory a page of which is taken away between two reads, and the
    /// zeros are zeros whatever the read buffer held before.
    #[test]
    fn an_uncommitted_tail_reads_as_zeros() {
        use windows_sys::Win32::System::Memory::*;
        const HEAD: usize = 0xb_0000;
        const SIZE: usize = 0x20_0000;
        // SAFETY: a private reservation of our own; only committed pages
        // are written in place, and nothing else reads them in place.
        unsafe {
            // The read buffer holds something else first.
            let other = vec![0x77u8; SIZE];
            with_title(other.as_ptr(), SIZE, |bytes| assert!(bytes.iter().all(|&b| b == 0x77)));
            let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert_eq!(VirtualAlloc(base.cast(), HEAD, MEM_COMMIT, PAGE_READWRITE), base.cast());
            base.write_bytes(0x5a, HEAD);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..HEAD].iter().all(|&b| b == 0x5a), "what is mapped");
                assert!(bytes[HEAD..].iter().all(|&b| b == 0), "what is not: zeros");
            });
            // The title takes a page in the middle away: zeros from then
            // on, the rest as before.
            assert_ne!(VirtualFree(base.add(0x4_0000).cast(), 0x1_0000, MEM_DECOMMIT), 0);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..0x4_0000].iter().all(|&b| b == 0x5a));
                assert!(bytes[0x4_0000..0x5_0000].iter().all(|&b| b == 0));
                assert!(bytes[0x5_0000..HEAD].iter().all(|&b| b == 0x5a));
                assert!(bytes[HEAD..].iter().all(|&b| b == 0));
            });
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    /// The mips a one-slice 2-D array view from mip 1 shows (1024x1024
    /// RGBA8 of two mips: mip 1 is the first 1 MiB of 5), their first and
    /// last pages not mapped, a 64 KiB block between them mapped: found
    /// mapped once the end of all the texture's memory is, as the probe of
    /// all of it found them when the view's span was all of it. Nothing
    /// mapped at any of those ends: none, without a look between.
    #[test]
    fn the_mapped_middle_of_a_views_mips_is_found() {
        use windows_sys::Win32::System::Memory::*;
        let resource = [3, 0, 0, 0, 1024, 0, 1024, 0x2_0001, 28, 1, 0, 0x105, 0, 0];
        let view = [28, 5, 0x1688, 0, 1, 1, 0, 1, 0, 0];
        let len = crate::resource::placed_bytes_from(&resource, 1).unwrap();
        let whole = crate::resource::placed_bytes(&resource).unwrap();
        assert!(len >= 0x3_0000 && len < whole, "{len:#x} of {whole:#x}");
        // SAFETY: a private reservation of our own, read only through the
        // kernel and the page queries.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), whole as usize, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            let commit = |at: u64| assert!(!VirtualAlloc(base.add(at as usize).cast(), 0x1000, MEM_COMMIT, PAGE_READWRITE).is_null());
            let at = base as u64;
            assert_eq!(super::view_mapped(at, &resource, &view), (len, 0), "nothing mapped");
            commit(0x1_0000);
            assert_eq!(super::view_mapped(at, &resource, &view), (len, 0), "no end mapped");
            commit(whole - 0x1000);
            let (_, mapped) = super::view_mapped(at, &resource, &view);
            assert!(mapped > 0 && mapped <= len, "the middle block: {mapped:#x}");
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    /// A made-up small texture the detiling path accepts: 24 texels wide, 6
    /// or 10 high, format 10, the console layout field naming swizzle mode 0,
    /// no flags. It is placed 0x300, 0x700 or 0xf00 into a page here, so
    /// that a page line runs through it: what the detiling is given and
    /// what it writes are the same read in place and read through the copy,
    /// and the rows it writes end where the staging buffer the host sizes
    /// for it ends (the last row's 192 bytes after the rows before it, 256
    /// apart).
    #[test]
    fn a_small_texture_detiles_the_same_both_ways() {
        use windows_sys::Win32::System::Memory::*;
        const WIDTH: usize = 24;
        for (height, within) in [(6u32, 0x300usize), (10, 0x700), (6, 0xf00), (10, 0xf00)] {
            let desc = crate::resource::ResourceDesc::test_texture_2d(WIDTH as u64, height, 10, 0x100, 0);
            let layout = crate::resource::texture_layout(&desc)
                .expect("a console layout")
                .planes[0]
                .layout
                .clone();
            let size = layout.size as usize;
            let (row, pitch) = (WIDTH * 8, 256usize);
            assert!(size >= row * height as usize, "the layout holds every texel");
            // Committed pages and a reserved one after them; the texture
            // begins `within` bytes into the first, as the title placed it.
            // SAFETY: our own reservation; the committed part is written
            // and read in place, by this thread alone.
            unsafe {
                let pages = size.div_ceil(4096) + 1;
                let base = VirtualAlloc(std::ptr::null(), (pages + 1) * 4096, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
                assert!(!base.is_null());
                assert_eq!(VirtualAlloc(base.cast(), pages * 4096, MEM_COMMIT, PAGE_READWRITE), base.cast());
                for i in 0..pages * 4096 {
                    base.add(i).write((i * 7 + i / 251) as u8);
                }
                let memory = base.add(within);
                let in_place = std::slice::from_raw_parts(memory, size);
                // The bytes: the same in place and through the copy.
                with_title(memory, size, |copied| assert_eq!(copied, in_place));
                // What the detiling writes: the same, and inside the bytes
                // the host's staging buffer for this texture has.
                let staged = (height as usize - 1) * pitch + row;
                assert_eq!(staged, if height == 6 { 1472 } else { 2496 });
                let detile = |tiled: &[u8]| {
                    // The slice `Staging::detile` hands over: whole rows.
                    let mut linear = vec![0xeeu8; pitch * height as usize];
                    layout
                        .detile(tiled, 0, 0, &mut linear, pitch, pitch * height as usize)
                        .expect("detiled");
                    linear
                };
                let direct = detile(in_place);
                let through = with_title(memory, size, |copied| detile(copied));
                assert_eq!(direct, through);
                for y in 0..height as usize {
                    assert!(direct[y * pitch..][..row].iter().any(|&b| b != 0xee), "row {y} written");
                    assert!(direct[y * pitch + row..(y + 1) * pitch].iter().all(|&b| b == 0xee), "nothing past row {y}");
                }
                // A buffer of exactly the staged bytes is enough.
                let mut exact = vec![0u8; staged];
                layout
                    .detile(in_place, 0, 0, &mut exact, pitch, pitch * height as usize)
                    .expect("the staging buffer's bytes are enough");
                assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
            }
        }
    }
}

/// A texture refresh and the runtime's changes of title memory's mapping
/// (`crate::watch::title_memory_remapping`): the refresh reads the texture
/// in place, and a change of its bytes must not happen under that read.
#[cfg(test)]
mod remap_tests {
    use super::Staging;
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
    use std::sync::{Arc, Mutex, mpsc};
    use std::time::Duration;
    use windows::Win32::Graphics::Direct3D12::{D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_SUBRESOURCE_FOOTPRINT};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_UINT;

    /// The fixture texture's size in texels: 8 bytes a texel, so a row of 160
    /// bytes in a staging pitch of 256.
    const WIDTH: u32 = 20;
    const ROWS: u32 = 6;

    /// A texture of the fixture's own, small and made up ([`WIDTH`] texels by
    /// [`ROWS`], format 10, the console layout field naming swizzle mode 0,
    /// no flags), in private committed memory, which is neither
    /// write-watched nor a view, so that its refresh reads it in place. Its
    /// staging state and its memory's address and bytes. The memory stays
    /// reserved after the test: no later test's memory takes its addresses.
    pub(super) fn texture() -> (Arc<Staging>, usize, usize) {
        use windows_sys::Win32::System::Memory::*;
        let host = crate::host::test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let desc = crate::resource::ResourceDesc::test_texture_2d(u64::from(WIDTH), ROWS, 10, 0x100, 0);
        let layout = crate::resource::texture_layout(&desc).expect("a console layout").planes[0].layout.clone();
        let size = layout.size as usize;
        // SAFETY: memory of our own, written here before anything reads it.
        let memory = unsafe {
            let base = VirtualAlloc(std::ptr::null(), size.div_ceil(4096) * 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
                .cast::<u8>();
            assert!(!base.is_null());
            for i in 0..size {
                base.add(i).write((i * 7 + 1) as u8);
            }
            base
        };
        let (pitch, rows) = (256u32, ROWS);
        let bytes = u64::from(pitch * rows);
        let (buffer, data) = host.upload_buffer(bytes).expect("a staging buffer");
        let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Format: DXGI_FORMAT_R16G16B16A16_UINT,
                Width: WIDTH,
                Height: rows,
                Depth: 1,
                RowPitch: pitch,
            },
        };
        // As `staging` counts it (and `Staging`'s drop takes away).
        super::STAGED[0].fetch_add(1, Relaxed);
        super::STAGED[1].fetch_add(bytes, Relaxed);
        let staging = Arc::new(Staging {
            buffer,
            #[cfg(feature = "diagnostics")]
            total: bytes,
            #[cfg(feature = "diagnostics")]
            resource: 0,
            data,
            footprints: vec![footprint],
            rows: vec![rows],
            pack_wait: std::sync::atomic::AtomicU64::new(0),
            shape: None,
            layout,
            mips: 1,
            first: 0,
            size,
            format: 0,
            detiled_hash: Mutex::new(None),
            placed: false,
            bytes,
            memory,
            last: Mutex::default(),
            dirty: AtomicBool::new(false),
            uploaded: AtomicBool::new(false),
            used: AtomicU64::new(crate::host::moment()),
            on_demand: false,
            detiled: AtomicU32::new(0),
            judged: [const { AtomicU64::new(0) }; 2],
            waiting: AtomicU64::new(0),
        });
        (staging, memory as usize, size)
    }

    /// `staging` refreshed on a thread of its own; told on the channel when
    /// the refresh is done.
    fn refreshing(staging: &Arc<Staging>) -> (std::thread::JoinHandle<()>, mpsc::Receiver<()>) {
        let (done, refreshed) = mpsc::channel();
        let staging = staging.clone();
        let thread = std::thread::spawn(move || {
            staging.refresh();
            done.send(()).unwrap();
        });
        (thread, refreshed)
    }

    /// A staging buffer whose bytes were not hashed while the trap was on
    /// (`VICTORY_TEXTURE_RECHECK`) is not watched: what it holds is not
    /// known, and a hash of zeros or of an older detiling would make a false
    /// change.
    #[test]
    fn a_staging_not_detiled_while_the_trap_is_on_is_not_watched() {
        let (staging, _, _) = texture();
        assert!(staging.recheck_source().is_none(), "no hash of what it holds");
        // A hash taken in this epoch stands; one from before the switch last
        // went off does not.
        let hash = blake3::hash(b"detiled");
        *staging.detiled_hash.lock().unwrap() = Some((7, hash));
        assert_eq!(staging.recheck_source_in(7).and_then(|s| s.hash), Some(hash));
        assert!(staging.recheck_source_in(8).is_none(), "taken before the switch went off");
    }

    /// While the runtime changes the mapping of other title memory, a
    /// texture refresh goes on: it reads the texture in place (detiles it
    /// into its staging buffer) without waiting for the change to end. With
    /// one lock over all of title memory it waited (the 17 s freeze at
    /// world arrival: texture work waiting for title memory to hold still).
    #[test]
    fn a_texture_refresh_goes_on_while_other_memory_is_remapped() {
        if !crate::watch::by_range() {
            return; // VICTORY_MAPPING_RANGES=0: one lock over all of it
        }
        let (staging, at, _) = texture();
        // Elsewhere: a megabyte after the texture, a 64 KiB page.
        let changing = crate::watch::title_memory_remapping(at + (1 << 20), 64 << 10);
        let (thread, refreshed) = refreshing(&staging);
        let went_on = refreshed.recv_timeout(Duration::from_secs(10)).is_ok();
        drop(changing);
        thread.join().unwrap();
        assert!(went_on, "the refresh waited for a change of other memory's mapping");
        assert!(staging.dirty.load(Relaxed), "the refresh read the texture");
    }

    /// While the runtime changes the mapping of the page the texture lies
    /// in, its refresh waits, and reads it once the change is done.
    #[test]
    fn a_texture_refresh_waits_for_a_remap_of_its_own_memory() {
        let (staging, at, _) = texture();
        let page = at & !((64 << 10) - 1);
        let changing = crate::watch::title_memory_remapping(page, 64 << 10);
        let (thread, refreshed) = refreshing(&staging);
        assert!(refreshed.recv_timeout(Duration::from_millis(200)).is_err(), "the texture was read while its mapping changed");
        assert!(!staging.dirty.load(Relaxed));
        drop(changing);
        assert!(refreshed.recv_timeout(Duration::from_secs(10)).is_ok(), "the refresh went on after the change");
        thread.join().unwrap();
        assert!(staging.dirty.load(Relaxed), "the refresh read the texture");
    }
}

/// The change checks with the switch's value given (`VICTORY_TEXTURE_DIGEST`
/// is read once, so the tests do not go through it): off, what is kept is
/// blake3's whole hash of the same bytes, as before; on, the digest, which
/// flags a changed byte and passes over unchanged bytes.
#[cfg(test)]
mod digest_tests {
    use super::{Staging, Sum};
    use std::sync::Arc;
    use std::sync::atomic::Ordering::Relaxed;

    /// The fixture's texture (`remap_tests::texture`) reading `size` bytes
    /// at `memory` instead, with no part to detile into: a refresh that
    /// finds a change only marks the staging buffer detiled (`dirty`).
    fn texture_over(memory: *const u8, size: usize) -> Arc<Staging> {
        let (staging, _, _) = super::remap_tests::texture();
        let mut staging = Arc::try_unwrap(staging).ok().expect("the fixture's only owner");
        staging.memory = memory;
        staging.size = size;
        staging.footprints.clear();
        staging.rows.clear();
        Arc::new(staging)
    }

    fn kept(staging: &Staging) -> Option<Sum> {
        staging.last.lock().unwrap().hash
    }

    /// Refreshed with `digest`, the all-mapped check in place: whether it
    /// detiled.
    fn detiled(staging: &Staging, digest: bool) -> bool {
        detiled_as(staging, digest, false)
    }

    /// Refreshed with `digest`, the all-mapped check through the kernel's
    /// copy if `copy` (`VICTORY_TEXTURE_COPY`'s path): whether it detiled.
    fn detiled_as(staging: &Staging, digest: bool, copy: bool) -> bool {
        staging.dirty.store(false, Relaxed);
        staging.refresh_as(digest, copy);
        staging.dirty.load(Relaxed)
    }

    #[test]
    fn a_sum_off_is_blake3s_whole_hash_and_kinds_never_match() {
        let bytes: Vec<u8> = (0..10_000u32).map(|i| (i * 7 + 3) as u8).collect();
        assert_eq!(Sum::of(&bytes, false), Sum::Blake3(blake3::hash(&bytes)));
        assert_eq!(Sum::of(&bytes, true), Sum::Digest(crate::digest::digest(&bytes)));
        // Two kinds compared: changed, not a panic.
        assert_ne!(Sum::of(&bytes, false), Sum::of(&bytes, true));
    }

    /// The in-place check (memory all mapped, neither watched nor guarded).
    #[test]
    fn the_in_place_check_keeps_blake3_off_and_the_digest_on() {
        for digest in [false, true] {
            let (staging, at, size) = super::remap_tests::texture();
            let memory = at as *mut u8;
            // SAFETY: the fixture's committed memory of `size` bytes.
            let bytes = || unsafe { std::slice::from_raw_parts(memory, size) }.to_vec();
            assert!(detiled(&staging, digest), "first look");
            assert_eq!(kept(&staging), Some(Sum::of(&bytes(), digest)));
            if !digest {
                assert_eq!(kept(&staging), Some(Sum::Blake3(blake3::hash(&bytes()))));
            }
            assert!(!detiled(&staging, digest), "unchanged bytes (digest: {digest})");
            // SAFETY: as above.
            unsafe { *memory.add(size / 2) ^= 0x80 };
            assert!(detiled(&staging, digest), "a changed byte (digest: {digest})");
            assert_eq!(kept(&staging), Some(Sum::of(&bytes(), digest)));
            assert!(!detiled(&staging, digest));
        }
    }

    /// The copied check (memory all mapped, read through the kernel's copy:
    /// `VICTORY_TEXTURE_COPY=1`), as the in-place one.
    #[test]
    fn the_copied_check_keeps_blake3_off_and_the_digest_on() {
        for digest in [false, true] {
            let (staging, at, size) = super::remap_tests::texture();
            let memory = at as *mut u8;
            // SAFETY: the fixture's committed memory of `size` bytes.
            let bytes = || unsafe { std::slice::from_raw_parts(memory, size) }.to_vec();
            assert!(detiled_as(&staging, digest, true), "first look");
            assert_eq!(kept(&staging), Some(Sum::of(&bytes(), digest)));
            if !digest {
                assert_eq!(kept(&staging), Some(Sum::Blake3(blake3::hash(&bytes()))));
            }
            assert!(!detiled_as(&staging, digest, true), "unchanged bytes (digest: {digest})");
            // SAFETY: as above.
            unsafe { *memory.add(size / 3) ^= 0x40 };
            assert!(detiled_as(&staging, digest, true), "a changed byte (digest: {digest})");
            assert_eq!(kept(&staging), Some(Sum::of(&bytes(), digest)));
            assert!(!detiled_as(&staging, digest, true));
        }
    }

    /// The partly mapped check: 64 KiB mapped of 128, read with zeros for
    /// the rest (every 30 frames; the wait is cleared here between looks).
    #[test]
    fn the_partly_mapped_check_keeps_blake3_off_and_the_digest_on() {
        use windows_sys::Win32::System::Memory::*;
        const SIZE: usize = 0x2_0000;
        const HEAD: usize = 0x1_0000;
        for digest in [false, true] {
            // SAFETY: a private reservation of our own, its head committed;
            // it stays reserved after the test.
            let memory = unsafe {
                let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
                assert!(!base.is_null());
                assert_eq!(VirtualAlloc(base.cast(), HEAD, MEM_COMMIT, PAGE_READWRITE), base.cast());
                for i in 0..HEAD {
                    base.add(i).write((i * 13 + 5) as u8);
                }
                base
            };
            let staging = texture_over(memory, SIZE);
            let read = || {
                let mut read = vec![0u8; SIZE];
                // SAFETY: the committed head.
                read[..HEAD].copy_from_slice(unsafe { std::slice::from_raw_parts(memory, HEAD) });
                read
            };
            let look = |staging: &Staging| {
                staging.last.lock().unwrap().partial = None;
                detiled(staging, digest)
            };
            assert!(look(&staging), "first look");
            assert_eq!(kept(&staging), Some(Sum::of(&read(), digest)));
            if !digest {
                assert_eq!(kept(&staging), Some(Sum::Blake3(blake3::hash(&read()))));
            }
            assert!(!look(&staging), "unchanged bytes (digest: {digest})");
            // SAFETY: the committed head.
            unsafe { *memory.add(HEAD - 1) ^= 1 };
            assert!(look(&staging), "a changed byte (digest: {digest})");
            assert!(!look(&staging));
        }
    }

    /// The check of a read through the kernel (`VICTORY_TEXTURE_NO_WAIT`'s
    /// path while title memory is remapped).
    #[test]
    fn the_unlocked_check_keeps_blake3_off_and_the_digest_on() {
        for digest in [false, true] {
            let (staging, at, size) = super::remap_tests::texture();
            let memory = at as *mut u8;
            // SAFETY: the fixture's committed memory of `size` bytes.
            let bytes = || unsafe { std::slice::from_raw_parts(memory, size) }.to_vec();
            let look = |staging: &Staging| {
                staging.dirty.store(false, Relaxed);
                let mut last = staging.last.lock().unwrap();
                assert_eq!(staging.refresh_unlocked(&mut last, size, 0, digest), size);
                drop(last);
                staging.dirty.load(Relaxed)
            };
            assert!(look(&staging));
            assert_eq!(kept(&staging), Some(Sum::of(&bytes(), digest)));
            if !digest {
                assert_eq!(kept(&staging), Some(Sum::Blake3(blake3::hash(&bytes()))));
            }
            assert!(!look(&staging), "unchanged bytes (digest: {digest})");
            // SAFETY: as above.
            unsafe { *memory ^= 0x80 };
            assert!(look(&staging), "a changed byte (digest: {digest})");
        }
    }

    /// The fingerprint: off, blake3's first eight bytes (bit 0 set) of the
    /// bytes read, whole up to 48 KiB, else of the twelve 4 KiB pieces one
    /// after another; on, the 128-bit digest of the same bytes folded to 64
    /// bits (bit 0 set), so a change in a piece moves it and the same bytes
    /// keep it.
    #[test]
    fn the_fingerprint_keeps_blake3_off_and_the_digest_on() {
        const PIECE: usize = 4 << 10;
        let off = |bytes: &[u8]| {
            let mut first = [0u8; 8];
            first.copy_from_slice(&blake3::hash(bytes).as_bytes()[..8]);
            u64::from_le_bytes(first) | 1
        };
        // On: the digest, not blake3 (a `fingerprint_as` that ignored
        // `digest` would fail here).
        let on = |bytes: &[u8]| {
            let d = crate::digest::digest(bytes);
            (d as u64 ^ (d >> 64) as u64) | 1
        };
        let print =
            |m: &[u8], digest: bool| super::fingerprint_as(m.as_ptr() as u64, m.len() as u64, m.len() as u64, digest);
        let mut small: Vec<u8> = (0..40_000u32).map(|i| (i * 11 + 1) as u8).collect();
        assert_eq!(print(&small, false), off(&small));
        assert_eq!(print(&small, true), on(&small));
        let before = print(&small, true);
        assert_eq!(print(&small, true), before);
        small[20_000] ^= 4;
        assert_ne!(print(&small, true), before);
        assert_eq!(print(&small, true), on(&small));
        assert_eq!(print(&small, false), off(&small));

        // 1 MiB: twelve pieces, the last at the end.
        let mut big: Vec<u8> = (0..1u32 << 20).map(|i| (i * 29 + (i >> 9)) as u8).collect();
        let len = big.len();
        let pieces = |big: &[u8]| -> Vec<u8> {
            let mut starts: Vec<usize> = (0..11).map(|n| ((len - PIECE) / 11 * n) & !0xfff).collect();
            starts.push(len - PIECE);
            starts.iter().flat_map(|&s| big[s..s + PIECE].to_vec()).collect()
        };
        assert_eq!(print(&big, false), off(&pieces(&big)));
        assert_eq!(print(&big, true), on(&pieces(&big)));
        let before = print(&big, true);
        assert_eq!(print(&big, true), before);
        big[len - 1] ^= 0x80;
        assert_ne!(print(&big, true), before);
        assert_eq!(print(&big, true), on(&pieces(&big)));
        assert_eq!(print(&big, false), off(&pieces(&big)));

        // `whole > len` with the digest, at the widest buffer this case
        // fills: the view's twelve pieces and those of the whole texture's
        // that start inside the view (the last of them cut at its end), 23
        // distinct of the 24 the buffer has room for (a search over lengths
        // to 4 MiB found none with 24). Digested end to end.
        let (len, whole) = (858_677usize, 862_099usize);
        let wide: Vec<u8> = (0..len as u32).map(|i| (i * 13 + (i >> 11)) as u8).collect();
        let starts = |l: usize| -> Vec<usize> {
            let mut s: Vec<usize> = (0..11).map(|n| ((l - PIECE) / 11 * n) & !0xfff).collect();
            s.push(l - PIECE);
            s
        };
        let mut from: Vec<usize> = starts(len).into_iter().chain(starts(whole).into_iter().filter(|&s| s < len)).collect();
        from.sort_unstable();
        from.dedup();
        assert_eq!(from.len(), 23, "the widest case");
        let laid: Vec<u8> = from.iter().flat_map(|&s| wide[s..s + PIECE.min(len - s)].to_vec()).collect();
        assert!(laid.len() > 22 * PIECE, "the buffer is nearly full");
        let got = super::fingerprint_as(wide.as_ptr() as u64, len as u64, whole as u64, true);
        assert_eq!(got, on(&laid));
    }
}
