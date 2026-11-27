//! `VICTORY_TEXTURE_RECHECK=1` (a live switch too): the stale-upload trap.
//! After a title texture is uploaded, its source memory is read again when
//! the upload is 8, 30 and 120 frames old. A change with no upload of the
//! texture since is *followed*, not counted yet, and resolved:
//!
//! - *again*: the texture was uploaded again (late, counted by how late);
//! - *released*: the title released the texture (`crate::texture::forget`);
//! - *unused*: 600 frames after the upload no list had sampled it since the
//!   change was seen;
//! - *stale*, the fault: a list sampled it after the change was seen, it is
//!   still registered, and it was not uploaded again within 600 frames.
//!
//! "Sampled" is the runtime's own record of a use, the moment a list last
//! recorded an upload of the texture (`Staging::used`, stamped by
//! `crate::texture::upload_from`), kept across a staging buffer let go as
//! idle ([`let_go`]). A draw the runtime does not record as a use leaves a
//! texture *unused*: a miss, never a false fault.
//!
//! The first look is at 8 frames, past the two designed delays a fresh write
//! has: a page destination's completion notice re-uploads two frames later
//! (`PAGE_WRITE_AFTER`) and a staged texture's refresh turns every eighth
//! frame (`HASH_EVERY`). A change seen at 8 frames is therefore the memory's
//! and not our own delay.
//!
//! The source range, its hash and the way it is read are `crate::texture`'s
//! (`Staging::refresh_timed` and `Staging::detile`): the bytes detiled into
//! the staging buffer are the bytes an upload copies. Not watched: a texture
//! whose host copy is made level by level (levels on demand), one the texture
//! pack prepares (with the pack or its log on, every 2-D texture of one
//! slice), one whose staging buffer was not detiled while the switch was on,
//! and a source larger than the looks' byte budget. Their staging buffer is
//! not a whole-picture copy of the source range, or what it holds is not
//! known, so a change there would be a false fault.
//!
//! The records are bounded: the `MAX_UPLOADS` most recent uploads, with a
//! copy of the uploaded bytes for diffs while they fit `KEPT_BYTES`. Every
//! loss is counted: records dropped before their last look, followed changes
//! dropped past `TRACKED_MOST`, fault lines not printed, frame boundaries
//! skipped while the last looks still ran.
//!
//! What the runtime knows of writers is kept beside each look: a storage
//! request covering the range, in flight at the upload or completed since
//! (`crate::watch::title_memory_written` and the destinations
//! `victory-hle`'s `dstorage` reports); a copy-back landing into it since
//! (`crate::shadow`); and the write tracking's own record
//! (`crate::watch::recorded`). A fault with none of them is *no writer seen*
//! only when the evidence since its upload is whole; else *writer unknown*.
//! The title's own CPU writes are no writer the runtime sees.
//!
//! The reads: the looks run on a thread of their own, never on the present
//! path, at most `LOOKS_PER_FRAME` ranges and `LOOK_BYTES_PER_FRAME` bytes a
//! frame boundary; the copies kept for diffs are taken at upload time, on
//! the submitting thread, at most `KEPT_PER_FRAME` bytes a frame. Both are
//! counted under the kernel-call counters' own caller, `recheck`.
//!
//! With the switch off nothing is recorded, read or logged: every entry
//! point asks `on` first, and a frame boundary costs one atomic swap once
//! the switch has a way in. Switching it off forgets everything, and the
//! staging buffers' hashes taken before (`epoch`), so that a change made
//! while it was off is never read as one made while it was on.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

/// The most uploads remembered.
const MAX_UPLOADS: usize = 8_192;
/// The most uploaded bytes kept for exact diffs, oldest records' copies
/// dropped first.
const KEPT_BYTES: usize = 64 << 20;
/// One upload larger than this keeps no copy.
const KEPT_UPLOAD: usize = 8 << 20;
/// Looks at an upload's source range after this many frames, once each. The
/// first is past the designed delays (the module's doc).
const LOOK_AGES: [u64; 3] = [8, 30, 120];
/// A change followed is resolved this many frames after its upload.
const STALE_FRAMES: u64 = 600;
/// The most records followed at once.
const TRACKED_MOST: usize = 4_096;
/// The most lines the stale group prints, and the most any one (format,
/// size) group of it takes.
const STALE_LINES: u64 = 40;
const STALE_PER_GROUP: u64 = 3;
/// The most ranges read and hashed at one frame boundary, and the most
/// bytes; a source larger than the bytes is not watched.
const LOOKS_PER_FRAME: usize = 512;
const LOOK_BYTES_PER_FRAME: usize = 32 << 20;
/// The most bytes copied at upload time for exact diffs in one frame.
const KEPT_PER_FRAME: usize = 8 << 20;
/// Landings and completed storage writes kept, and for how many frames: at
/// least the last look's age, with as much again for looks the budget puts
/// off. A look later than that, or past an event dropped by count, finds
/// its evidence not whole and says so.
const WRITER_EVENTS: usize = 1 << 16;
const WRITER_FRAMES: u64 = 2 * LOOK_AGES[2];
/// How long a storage write may stay "in flight" without an end.
const IN_FLIGHT_FRAMES: u64 = 120;

/// Whether the switch has a way in at all (`VICTORY_LIVE` or
/// `VICTORY_TEXTURE_RECHECK`): without either, do not even set up the live
/// registry, and a run is exactly as before.
fn armed() -> bool {
    static ARMED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ARMED.get_or_init(|| {
        std::env::var_os("VICTORY_LIVE").is_some() || std::env::var_os("VICTORY_TEXTURE_RECHECK").is_some()
    })
}

/// Whether the switch is on (`VICTORY_TEXTURE_RECHECK`, live).
pub(crate) fn on() -> bool {
    armed() && crate::live::texture_recheck()
}

/// The staging buffers' hashes are good for this epoch only: it moves each
/// time the switch goes off, so a hash taken before (or none, for a detiling
/// while off) never stands for what the buffer holds now.
static EPOCH: AtomicU64 = AtomicU64::new(1);

/// The epoch a staging buffer's hash is taken in (`crate::texture`).
pub(crate) fn epoch() -> u64 {
    EPOCH.load(Relaxed)
}

/// A staging buffer's source, as `crate::texture` reads and hashes it.
#[derive(Clone, Copy)]
pub(crate) struct Source {
    /// The title texture's id (its address, as the log names it).
    pub(crate) id: usize,
    pub(crate) start: usize,
    pub(crate) size: usize,
    pub(crate) format: u32,
    pub(crate) mip: u32,
    /// The write tracking's asking when the staging buffer was last checked
    /// (0: not tracked).
    pub(crate) asked: u64,
    /// The hash of the bytes detiled into the staging buffer, taken while
    /// the switch was on; none: not known, and the upload is not watched.
    pub(crate) hash: Option<blake3::Hash>,
    /// Whether the runtime made the texture for a placed view (`true`) or the
    /// title made the object (`false`); `None` unknown.
    pub(crate) placed: Option<bool>,
}

/// One upload's source and hash, kept for the looks.
struct Record {
    id: usize,
    start: usize,
    size: usize,
    format: u32,
    mip: u32,
    frame: u64,
    seq: u64,
    hash: blake3::Hash,
    asked: u64,
    /// The bytes uploaded, while the byte budget holds them.
    bytes: Option<Vec<u8>>,
    /// The `LOOK_AGES` already looked at (bit `i` for age `i`).
    looked: u8,
    /// Found changed: followed in [`Book::tracked`], not looked at again.
    detected: bool,
    /// Whether a storage write covered the range when the upload was
    /// recorded, and whether the book of writes in flight was whole then.
    storage_in_flight: bool,
    in_flight_whole: bool,
    /// The texture's kind ([`Source::placed`]).
    placed: Option<bool>,
}

/// Every look taken.
const ALL_LOOKS: u8 = (1 << LOOK_AGES.len()) - 1;

/// A source range touched by the runtime's own writers.
#[derive(Clone, Copy)]
struct Event {
    start: usize,
    end: usize,
    frame: u64,
}

fn touches(event: Event, start: usize, end: usize) -> bool {
    event.start < end && start < event.end
}

/// One range to read again at this boundary.
struct Candidate {
    seq: u64,
    id: usize,
    start: usize,
    size: usize,
    hash: blake3::Hash,
}

/// A change found and followed.
#[derive(Clone, Copy)]
struct Tracked {
    seq: u64,
    id: usize,
    format: u32,
    mip: u32,
    size: usize,
    placed: Option<bool>,
    start: usize,
    upload_frame: u64,
    /// The frame once the bytes that showed the change had been read: a
    /// sample at a later frame came after the change was seen.
    detected_frame: u64,
    differing: Option<u64>,
    first: Option<usize>,
    uploaded_zero: Option<bool>,
    storage_in_flight: bool,
    storage_done_since: bool,
    landing_since: bool,
    /// Whether every writer event since the upload was still kept when the
    /// change was seen.
    evidence_whole: bool,
    write_tracking: Option<bool>,
}

/// A fault's writer group: each fault is in exactly one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Writer {
    /// A storage request in flight at the upload or completed since.
    Storage,
    /// No storage request; a copy-back landing since.
    Landing,
    /// Neither, with the evidence since the upload whole.
    NoneSeen,
    /// Neither, with events since the upload dropped: not known.
    Unknown,
}

impl Tracked {
    fn writer(&self) -> Writer {
        if self.storage_in_flight || self.storage_done_since {
            Writer::Storage
        } else if self.landing_since {
            Writer::Landing
        } else if self.evidence_whole {
            Writer::NoneSeen
        } else {
            Writer::Unknown
        }
    }
}

#[derive(Default)]
struct Book {
    records: VecDeque<Record>,
    /// The last upload of each texture: frame and sequence number.
    last: HashMap<usize, (u64, u64)>,
    next_seq: u64,
    kept_bytes: usize,
    /// The frame `kept_this_frame` counts, and the bytes copied for diffs
    /// in it.
    byte_frame: u64,
    kept_this_frame: usize,
    uploads: u64,
    /// Uploads not watched: a source larger than the looks' byte budget.
    too_large: u64,
    /// The changes followed and not yet resolved.
    tracked: Vec<Tracked>,
    /// Resolved: uploaded again, by frames after the upload ([8, 30),
    /// [30, 120), [120, 600)); the fault; not sampled since seen; released.
    again: [u64; 3],
    stale: u64,
    unused: u64,
    released: u64,
    /// The distinct textures of the faults.
    stale_ids: HashSet<usize>,
    /// The faults by writer group ([`Writer`]): storage, landing, none,
    /// unknown.
    writers: [u64; 4],
    /// Losses: records dropped from the ring before their last look,
    /// followed changes dropped past `TRACKED_MOST`, fault lines not printed.
    not_looked: u64,
    follows_dropped: u64,
    lines_capped: u64,
    /// The last frame a list was known to sample each followed texture, kept
    /// here when its staging buffer goes ([`let_go`]) and when a change is
    /// resolved.
    sampled: HashMap<usize, u64>,
    /// Textures the title released: their records below this sequence are
    /// the released texture's (a new one may come at the same address).
    gone: HashMap<usize, u64>,
    /// The fault lines printed, and the ones each (format, size) took.
    warn_lines: u64,
    groups: HashMap<(u32, usize), u32>,
    /// The wall-clock minute last said.
    minute: Option<u64>,
    in_flight: Vec<Event>,
    storage_done: VecDeque<Event>,
    landings: VecDeque<Event>,
    /// Writer events from this frame on are all kept; an upload before it
    /// has lost some.
    evidence_from: u64,
    /// Uploads up to this frame may have missed a storage write in flight
    /// (one was dropped from the book of writes in flight).
    in_flight_short_until: Option<u64>,
}

/// The figures the summary line and the live stats file take.
#[derive(Clone, Copy, Default)]
pub(crate) struct Figures {
    pub(crate) uploads: u64,
    pub(crate) too_large: u64,
    pub(crate) again: [u64; 3],
    pub(crate) stale: u64,
    pub(crate) stale_textures: u64,
    pub(crate) unused: u64,
    pub(crate) released: u64,
    pub(crate) following: u64,
    /// The faults by writer group: storage, landing, none seen, unknown.
    pub(crate) writers: [u64; 4],
    pub(crate) not_looked: u64,
    pub(crate) follows_dropped: u64,
    pub(crate) lines_capped: u64,
    pub(crate) frames_skipped: u64,
}

static BOOK: Mutex<Option<Book>> = Mutex::new(None);

/// Frame boundaries whose looks were skipped because the last ones still
/// ran (since the switch last came on).
static SKIPPED: AtomicU64 = AtomicU64::new(0);

/// Runs `f` on the book, made on first use.
fn with_book<R>(f: impl FnOnce(&mut Book) -> R) -> R {
    let mut book = BOOK.lock().unwrap_or_else(|e| e.into_inner());
    f(book.get_or_insert_with(Book::default))
}

/// Drops events from the front of `events` while `drop` says so, and moves
/// `from` past each one dropped: the evidence from there on is whole.
fn drop_front(events: &mut VecDeque<Event>, from: &mut u64, mut drop: impl FnMut(&Event, usize) -> bool) {
    while let Some(front) = events.front() {
        if !drop(front, events.len()) {
            break;
        }
        *from = (*from).max(front.frame + 1);
        events.pop_front();
    }
}

impl Book {
    /// Whether a copy of `size` bytes still fits the budgets (the frame's
    /// resets when it moves).
    fn wants_bytes(&mut self, size: usize, frame: u64) -> bool {
        if self.byte_frame != frame {
            self.byte_frame = frame;
            self.kept_this_frame = 0;
        }
        size > 0
            && size <= KEPT_UPLOAD
            && self.kept_bytes + size <= KEPT_BYTES
            && self.kept_this_frame + size <= KEPT_PER_FRAME
    }

    /// Whether a record of `id` with sequence `seq` is a texture the title
    /// has released since.
    fn released_since(&self, id: usize, seq: u64) -> bool {
        self.gone.get(&id).is_some_and(|&horizon| horizon > seq)
    }

    /// The storage writes in flight over `[start, end)` now, and whether the
    /// book of them is whole for an upload at `frame`.
    fn in_flight_over(&self, start: usize, end: usize, frame: u64) -> (bool, bool) {
        let covered = self.in_flight.iter().any(|&w| touches(w, start, end));
        (covered, self.in_flight_short_until.is_none_or(|until| frame > until))
    }

    /// Remembers one upload. An upload of the same texture in the same
    /// frame replaces the earlier one: the lists of a submission copy the
    /// same staging buffer. A source with no known hash, or larger than the
    /// looks' byte budget, is not watched.
    fn add(&mut self, new: Source, frame: u64, bytes: Option<Vec<u8>>) {
        let Some(hash) = new.hash else {
            return;
        };
        if new.size == 0 {
            return;
        }
        if new.size > LOOK_BYTES_PER_FRAME {
            self.too_large += 1;
            return;
        }
        let end = new.start.saturating_add(new.size);
        let (storage_in_flight, in_flight_whole) = self.in_flight_over(new.start, end, frame);
        // The lists of a submission upload the same staging buffer: only a
        // duplicate scans the records, so a frame of unique textures pays
        // one hash-map lookup each.
        let same_frame = self.last.get(&new.id).is_some_and(|&(last, _)| last == frame);
        if same_frame && let Some(index) = self.records.iter().rposition(|r| r.id == new.id && r.frame == frame) {
            let mut record = self.records.remove(index).expect("the record found");
            self.forget_bytes(&record);
            record.bytes = None;
            record.start = new.start;
            record.size = new.size;
            record.format = new.format;
            record.mip = new.mip;
            record.hash = hash;
            record.asked = new.asked;
            record.placed = new.placed;
            record.looked = 0;
            record.detected = false;
            record.storage_in_flight = storage_in_flight;
            record.in_flight_whole = in_flight_whole;
            self.keep_bytes(&mut record, bytes, frame);
            self.records.insert(index, record);
            return;
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        let mut record = Record {
            id: new.id,
            start: new.start,
            size: new.size,
            format: new.format,
            mip: new.mip,
            frame,
            seq,
            hash,
            asked: new.asked,
            bytes: None,
            looked: 0,
            detected: false,
            storage_in_flight,
            in_flight_whole,
            placed: new.placed,
        };
        self.keep_bytes(&mut record, bytes, frame);
        self.records.push_back(record);
        self.last.insert(new.id, (frame, seq));
        self.uploads += 1;
        while self.records.len() > MAX_UPLOADS {
            let Some(old) = self.records.pop_front() else {
                break;
            };
            self.forget_bytes(&old);
            if !old.detected && old.looked != ALL_LOOKS && !self.released_since(old.id, old.seq) {
                self.not_looked += 1;
            }
            if self.last.get(&old.id).is_some_and(|&(_, last)| last == old.seq) {
                self.last.remove(&old.id);
            }
        }
        while self.kept_bytes > KEPT_BYTES {
            let Some(old) = self.records.iter_mut().find(|r| r.bytes.is_some()) else {
                break;
            };
            if let Some(bytes) = old.bytes.take() {
                self.kept_bytes = self.kept_bytes.saturating_sub(bytes.len());
            }
        }
    }

    fn keep_bytes(&mut self, record: &mut Record, bytes: Option<Vec<u8>>, frame: u64) {
        if let Some(bytes) = bytes
            && bytes.len() == record.size
            && self.wants_bytes(bytes.len(), frame)
        {
            self.kept_bytes += bytes.len();
            self.kept_this_frame += bytes.len();
            record.bytes = Some(bytes);
        }
    }

    fn forget_bytes(&mut self, record: &Record) {
        if let Some(bytes) = &record.bytes {
            self.kept_bytes = self.kept_bytes.saturating_sub(bytes.len());
        }
    }

    /// The records to look at now: for each, the first age it has reached
    /// that it has not been looked at. (The tests take them all; [`look`]
    /// gives them a byte budget.)
    #[cfg(test)]
    fn candidates(&mut self, frame: u64) -> Vec<Candidate> {
        self.candidates_within(frame, usize::MAX)
    }

    /// [`Book::candidates`] whose ranges together take at most `bytes`, and
    /// at most `LOOKS_PER_FRAME` of them.
    fn candidates_within(&mut self, frame: u64, bytes: usize) -> Vec<Candidate> {
        let mut out = Vec::new();
        let mut taken = 0usize;
        let gone = &self.gone;
        for record in &mut self.records {
            if record.detected || gone.get(&record.id).is_some_and(|&horizon| horizon > record.seq) {
                continue;
            }
            let age = frame.saturating_sub(record.frame);
            let Some(index) = (0..LOOK_AGES.len()).find(|&i| age >= LOOK_AGES[i] && record.looked & (1 << i) == 0) else {
                continue;
            };
            if taken + record.size > bytes {
                continue;
            }
            // Ages already passed are looked at by this one look.
            for (i, &at) in LOOK_AGES.iter().enumerate().skip(index) {
                if age >= at {
                    record.looked |= 1 << i;
                }
            }
            taken += record.size;
            out.push(Candidate {
                seq: record.seq,
                id: record.id,
                start: record.start,
                size: record.size,
                hash: record.hash,
            });
            if out.len() >= LOOKS_PER_FRAME || taken >= bytes {
                break;
            }
        }
        out
    }

    /// Whether some event of `events` since `frame` touches `[start, end)`.
    /// Events are pushed in frame order give or take a frame (threads read
    /// the frame before they take the lock), so the walk from the newest end
    /// goes two frames past.
    fn touched_since(events: &VecDeque<Event>, frame: u64, start: usize, end: usize) -> bool {
        events
            .iter()
            .rev()
            .take_while(|e| e.frame + 2 >= frame)
            .any(|&e| e.frame >= frame && touches(e, start, end))
    }

    /// A candidate's source no longer hashes to the upload's bytes: the change
    /// is followed (not counted yet), with what the runtime knows of its
    /// writers. `frame` is the frame once the bytes had been read.
    fn detect(&mut self, candidate: &Candidate, hash: blake3::Hash, bytes: &[u8], frame: u64) -> Option<Tracked> {
        let index = self.records.iter().position(|r| r.seq == candidate.seq && r.id == candidate.id)?;
        let record = &self.records[index];
        // A record the same frame replaced since the candidates were taken
        // has the newer hash: compare with that.
        if record.detected || hash == record.hash || self.released_since(record.id, record.seq) {
            return None;
        }
        if self.last.get(&record.id).is_some_and(|&(_, last)| last > record.seq) {
            return None;
        }
        let (mut differing, mut first, mut uploaded_zero) = (None, None, None);
        if let Some(old) = record.bytes.as_deref().filter(|old| old.len() == bytes.len()) {
            let mut count = 0u64;
            let mut at = None;
            let mut zero = true;
            for (i, (&before, &after)) in old.iter().zip(bytes).enumerate() {
                if before != after {
                    count += 1;
                    at.get_or_insert(i);
                    if before != 0 {
                        zero = false;
                    }
                }
            }
            differing = Some(count);
            first = at;
            uploaded_zero = Some(zero);
        }
        let start = record.start;
        let end = start.saturating_add(record.size);
        let storage_done_since = Self::touched_since(&self.storage_done, record.frame, start, end);
        let landing_since = Self::touched_since(&self.landings, record.frame, start, end);
        let write_tracking = (record.asked != 0)
            .then(|| crate::watch::recorded(start, end).iter().any(|&(_, asked)| asked > record.asked));
        let tracked = Tracked {
            seq: record.seq,
            id: record.id,
            format: record.format,
            mip: record.mip,
            size: record.size,
            placed: record.placed,
            start,
            upload_frame: record.frame,
            detected_frame: frame,
            differing,
            first,
            uploaded_zero,
            storage_in_flight: record.storage_in_flight,
            storage_done_since,
            landing_since,
            evidence_whole: record.frame >= self.evidence_from && record.in_flight_whole,
            write_tracking,
        };
        self.records[index].detected = true;
        if self.tracked.len() >= TRACKED_MOST {
            self.tracked.remove(0);
            self.follows_dropped += 1;
        }
        self.tracked.push(tracked);
        Some(tracked)
    }

    /// A list sampled texture `id` at `frame` (the latest known is kept).
    fn note_sampled(&mut self, id: usize, frame: u64) {
        let at = self.sampled.entry(id).or_insert(frame);
        *at = (*at).max(frame);
    }

    /// The title released texture `id`: what is followed of it, and its
    /// records' looks, end (a texture made later at the same address is
    /// another).
    fn released(&mut self, id: usize) {
        self.gone.insert(id, self.next_seq);
        self.sampled.remove(&id);
    }

    /// The followed textures due at `frame`, released or uploaded again
    /// left out: the ones whose samples decide.
    fn due(&self, frame: u64) -> Vec<usize> {
        self.tracked
            .iter()
            .filter(|item| frame.saturating_sub(item.upload_frame) >= STALE_FRAMES)
            .filter(|item| !self.released_since(item.id, item.seq))
            .filter(|item| !self.last.get(&item.id).is_some_and(|&(_, last)| last > item.seq))
            .map(|item| item.id)
            .collect()
    }

    /// Resolves what is followed: released, uploaded again (counted by how
    /// late), and at 600 frames after the upload either the fault (sampled
    /// after the change was seen) or unused. The faults are returned to be
    /// itemised.
    fn resolve(&mut self, frame: u64) -> Vec<Tracked> {
        let mut stale = Vec::new();
        let mut still = Vec::new();
        for item in std::mem::take(&mut self.tracked) {
            if self.released_since(item.id, item.seq) {
                self.released += 1;
                continue;
            }
            if self.last.get(&item.id).is_some_and(|&(_, last)| last > item.seq) {
                let age = frame.saturating_sub(item.upload_frame);
                let bucket = if age < 30 {
                    0
                } else if age < 120 {
                    1
                } else {
                    2
                };
                self.again[bucket] += 1;
                continue;
            }
            if frame.saturating_sub(item.upload_frame) < STALE_FRAMES {
                still.push(item);
                continue;
            }
            if !self.sampled.get(&item.id).is_some_and(|&at| at > item.detected_frame) {
                self.unused += 1;
                continue;
            }
            self.stale += 1;
            self.stale_ids.insert(item.id);
            self.writers[item.writer() as usize] += 1;
            stale.push(item);
        }
        self.tracked = still;
        let followed: HashSet<usize> = self.tracked.iter().map(|item| item.id).collect();
        self.sampled.retain(|id, _| followed.contains(id));
        stale
    }

    /// Drops writer events outside the window a look can care about, and the
    /// releases no record or followed change is below any more.
    fn prune(&mut self, frame: u64) {
        let keep = frame.saturating_sub(WRITER_FRAMES);
        drop_front(&mut self.storage_done, &mut self.evidence_from, |e, _| e.frame < keep);
        drop_front(&mut self.landings, &mut self.evidence_from, |e, _| e.frame < keep);
        let old = frame.saturating_sub(IN_FLIGHT_FRAMES);
        if self.in_flight.iter().any(|w| w.frame < old) {
            self.in_flight.retain(|w| w.frame >= old);
            self.in_flight_short_until = Some(frame + IN_FLIGHT_FRAMES);
        }
        let oldest = self
            .records
            .front()
            .map(|r| r.seq)
            .into_iter()
            .chain(self.tracked.iter().map(|t| t.seq))
            .min()
            .unwrap_or(self.next_seq);
        self.gone.retain(|_, horizon| *horizon > oldest);
    }

    fn storage_begin(&mut self, start: usize, end: usize, frame: u64) {
        self.in_flight.push(Event { start, end, frame });
        if self.in_flight.len() > WRITER_EVENTS {
            self.in_flight.remove(0);
            self.in_flight_short_until = Some(frame + IN_FLIGHT_FRAMES);
        }
    }

    fn storage_end(&mut self, start: usize, end: usize, frame: u64) {
        self.in_flight.retain(|w| w.start != start || w.end != end);
        self.completed(start, end, frame);
    }

    fn completed(&mut self, start: usize, end: usize, frame: u64) {
        self.storage_done.push_back(Event { start, end, frame });
        drop_front(&mut self.storage_done, &mut self.evidence_from, |_, len| len > WRITER_EVENTS);
    }

    fn landing(&mut self, start: usize, end: usize, frame: u64) {
        self.landings.push_back(Event { start, end, frame });
        drop_front(&mut self.landings, &mut self.evidence_from, |_, len| len > WRITER_EVENTS);
    }

    /// Whether this fault's line may be printed: `STALE_LINES` over the run
    /// and `STALE_PER_GROUP` for its (format, size); one that may not is
    /// counted.
    fn may_log(&mut self, item: &Tracked) -> bool {
        let group = self.groups.entry((item.format, item.size)).or_insert(0);
        if self.warn_lines >= STALE_LINES || *group >= STALE_PER_GROUP as u32 {
            self.lines_capped += 1;
            return false;
        }
        *group += 1;
        self.warn_lines += 1;
        true
    }

    /// The minute line's figures, when the wall clock's minute has moved.
    fn minute_summary(&mut self, minute: u64) -> Option<Figures> {
        if self.minute == Some(minute) {
            return None;
        }
        let said = self.minute.is_some();
        self.minute = Some(minute);
        // The first call marks the minute it starts in; nothing to say yet.
        said.then(|| self.figures())
    }

    fn figures(&self) -> Figures {
        Figures {
            uploads: self.uploads,
            too_large: self.too_large,
            again: self.again,
            stale: self.stale,
            stale_textures: self.stale_ids.len() as u64,
            unused: self.unused,
            released: self.released,
            following: self.tracked.len() as u64,
            writers: self.writers,
            not_looked: self.not_looked,
            follows_dropped: self.follows_dropped,
            lines_capped: self.lines_capped,
            frames_skipped: SKIPPED.load(Relaxed),
        }
    }
}

/// Reads title memory the way an upload does: through the kernel, so pages
/// that are not there read as zeros. Counted under the trap's own caller
/// (`recheck`), never as a submission's copy (`copy_mapped`).
fn read_range(start: usize, size: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; size];
    crate::host::read_for(crate::kernelstat::Caller::Recheck, start as *const u8, bytes.as_mut_ptr(), size);
    bytes
}

/// Remembers one upload of title texture `id` from `source`. Called after
/// the list's submission-time refresh, so the hash is the staging buffer's
/// content when its copies run.
pub(crate) fn uploaded(id: usize, source: Source) {
    if !on() || source.size == 0 || source.hash.is_none() {
        return;
    }
    let frame = crate::frame::current();
    let keep = with_book(|book| book.wants_bytes(source.size, frame));
    let mut bytes = keep.then(|| read_range(source.start, source.size));
    if bytes.as_ref().is_some_and(|bytes| Some(blake3::hash(bytes)) != source.hash) {
        // The source changed between the detiling and this record: staging
        // still holds the detiled bytes, which we no longer have.
        bytes = None;
    }
    with_book(|book| book.add(Source { id, ..source }, frame, bytes));
}

/// The title released texture `id` (`crate::texture::forget`).
pub(crate) fn released(id: usize) {
    if !on() {
        return;
    }
    with_book(|book| book.released(id));
}

/// Texture `id`'s staging buffer is let go (idle, or its host copy let go);
/// a list last sampled it at `frame`, which is kept for the resolve.
pub(crate) fn let_go(id: usize, frame: u64) {
    if !on() {
        return;
    }
    with_book(|book| {
        if book.tracked.iter().any(|item| item.id == id) {
            book.note_sampled(id, frame);
        }
    });
}

/// A storage request's write into title memory `[start, start + size)` is
/// about to run (DirectStorage's destination, from `victory-hle`).
pub(crate) fn storage_begin(start: usize, size: usize) {
    if !on() || size == 0 {
        return;
    }
    let frame = crate::frame::current();
    with_book(|book| book.storage_begin(start, start.saturating_add(size), frame));
}

/// The write `storage_begin` announced has ended: its bytes are there.
pub(crate) fn storage_end(start: usize, size: usize) {
    if !on() || size == 0 {
        return;
    }
    let frame = crate::frame::current();
    with_book(|book| book.storage_end(start, start.saturating_add(size), frame));
}

/// A storage write's bytes are there (`crate::watch::title_memory_written`).
pub(crate) fn storage_completed(start: usize, size: usize) {
    if !on() || size == 0 {
        return;
    }
    let frame = crate::frame::current();
    with_book(|book| book.completed(start, start.saturating_add(size), frame));
}

/// A copy-back landing wrote title memory `[start, start + size)`
/// (`crate::shadow`).
pub(crate) fn landing(start: usize, size: usize) {
    if !on() || size == 0 {
        return;
    }
    let frame = crate::frame::current();
    with_book(|book| book.landing(start, start.saturating_add(size), frame));
}

/// Whether the switch was on at the last frame boundary.
static SEEN_ON: AtomicBool = AtomicBool::new(false);

/// Whether the switch went off since the last boundary that saw it on:
/// what `seen` says of the last boundary, brought up to `on`.
fn went_off(seen: &AtomicBool, on: bool) -> bool {
    seen.swap(on, Relaxed) && !on
}

/// The switch went off: everything recorded goes, and the staging buffers'
/// hashes with it ([`EPOCH`]), so that nothing seen before it went off, nor
/// a detiling made while it is off, is resolved once it is on again.
fn switched_off() {
    EPOCH.fetch_add(1, Relaxed);
    *BOOK.lock().unwrap_or_else(|e| e.into_inner()) = None;
    SKIPPED.store(0, Relaxed);
}

/// A frame boundary (the present path): hands the looks to the trap's own
/// thread and returns. Off, one atomic swap once the switch has a way in.
pub(crate) fn frame(frame: u64) {
    if !armed() {
        return;
    }
    let on = crate::live::texture_recheck();
    if went_off(&SEEN_ON, on) {
        switched_off();
    }
    if !on {
        return;
    }
    static LOOKS: std::sync::OnceLock<Option<std::sync::mpsc::SyncSender<u64>>> = std::sync::OnceLock::new();
    let looks = LOOKS.get_or_init(|| {
        let (send, take) = std::sync::mpsc::sync_channel::<u64>(1);
        std::thread::Builder::new()
            .name("texture recheck".into())
            .spawn(move || {
                for frame in take {
                    look(frame);
                }
            })
            .ok()
            .map(|_| send)
    });
    if let Some(looks) = looks
        && let Err(std::sync::mpsc::TrySendError::Full(_)) = looks.try_send(frame)
    {
        SKIPPED.fetch_add(1, Relaxed);
    }
}

/// The looks of frame boundary `frame`, on the trap's own thread: reads the
/// uploads that are 8, 30 and 120 frames old within the budget, follows what
/// changed, resolves what is due and itemises the faults; a minute line on
/// the wall clock.
fn look(frame: u64) {
    if !on() {
        return;
    }
    let candidates = with_book(|book| {
        book.prune(frame);
        book.candidates_within(frame, LOOK_BYTES_PER_FRAME)
    });
    for candidate in candidates {
        let bytes = read_range(candidate.start, candidate.size);
        let hash = blake3::hash(&bytes);
        if hash == candidate.hash {
            continue;
        }
        // Read after the bytes: a sample at a later frame came after the
        // change was seen.
        let seen = crate::frame::current();
        with_book(|book| book.detect(&candidate, hash, &bytes, seen));
    }
    // The samples of what is due, outside the book's lock (the staging
    // table's lock is taken here).
    let due = with_book(|book| book.due(frame));
    let samples: Vec<(usize, u64)> =
        due.into_iter().filter_map(|id| crate::texture::last_sampled(id).map(|at| (id, at))).collect();
    let (stale, summary) = with_book(|book| {
        for (id, at) in samples {
            book.note_sampled(id, at);
        }
        let stale = book.resolve(frame);
        (stale, book.minute_summary(crate::host::now_ms() / 60_000))
    });
    for item in &stale {
        let registered = crate::descriptor::host::registered(item.start as u64);
        if !with_book(|book| book.may_log(item)) {
            continue;
        }
        tracing::warn!(
            texture = format_args!("{:#x}", item.id),
            format = item.format,
            mip = item.mip,
            size = item.size,
            placed = match item.placed {
                Some(true) => "yes",
                Some(false) => "no",
                None => "unknown",
            },
            registered_span = registered,
            frames_since_detected = frame.saturating_sub(item.detected_frame),
            differing = item.differing.map_or_else(|| "not kept".to_owned(), |n| n.to_string()),
            first_offset = item.first.map_or_else(|| "not kept".to_owned(), |at| format!("{at:#x}")),
            uploaded_zero = item.uploaded_zero.map_or("not kept", |zero| if zero { "yes" } else { "no" }),
            storage_in_flight = item.storage_in_flight,
            storage_done_since = item.storage_done_since,
            landing = item.landing_since,
            writer_evidence = if item.evidence_whole { "whole" } else { "not whole" },
            write_tracking = item.write_tracking.map_or("untracked", |wrote| if wrote { "yes" } else { "no" }),
            "texture recheck: stale upload: the source changed after the upload, a list sampled the texture after the change was seen, it is still registered, and it was not uploaded again within 600 frames"
        );
    }
    if let Some(f) = summary {
        tracing::info!(
            "texture recheck since the switch came on: uploads watched {} ({} too large to watch); changes followed {}: uploaded again {} ({} under 30 frames, {} 30 to 120, {} 120 to 600), stale {} ({} textures; sampled after the change was seen, still registered, not uploaded again within 600 frames), unused {} (not sampled since the change was seen), released {}, still followed {}; the stale by writer: storage request {}, copy-back landing {}, no writer seen {}, writer unknown (evidence not whole) {}; lost: {} uploads dropped before their last look, {} follows dropped, {} stale lines not printed, {} frames' looks skipped. Not watched: levels-on-demand textures, texture-pack textures (2-D, one slice, with the pack or its log on), staging buffers not detiled while on, and sources over 32 MB; the title's own CPU writes are no writer the runtime sees",
            f.uploads,
            f.too_large,
            f.again.iter().sum::<u64>() + f.stale + f.unused + f.released + f.following,
            f.again.iter().sum::<u64>(),
            f.again[0],
            f.again[1],
            f.again[2],
            f.stale,
            f.stale_textures,
            f.unused,
            f.released,
            f.following,
            f.writers[Writer::Storage as usize],
            f.writers[Writer::Landing as usize],
            f.writers[Writer::NoneSeen as usize],
            f.writers[Writer::Unknown as usize],
            f.not_looked,
            f.follows_dropped,
            f.lines_capped,
            f.frames_skipped,
        );
    }
}

/// The figures for the live stats file; zero with nothing recorded.
pub(crate) fn figures() -> Figures {
    BOOK.lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or_else(|| Figures { frames_skipped: SKIPPED.load(Relaxed), ..Figures::default() }, Book::figures)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(id: usize, bytes: &[u8]) -> Source {
        Source {
            id,
            start: id * 0x1000,
            size: bytes.len(),
            format: 71,
            mip: 0,
            asked: 0,
            hash: Some(blake3::hash(bytes)),
            placed: Some(true),
        }
    }

    fn add(book: &mut Book, id: usize, frame: u64, bytes: &[u8]) {
        book.add(source(id, bytes), frame, Some(bytes.to_vec()));
    }

    fn candidate_for(book: &mut Book, id: usize, now: u64) -> Candidate {
        book.candidates(now).into_iter().find(|c| c.id == id).expect("a candidate")
    }

    /// Texture `id` uploaded at 100 from `aaaa`, found changed to `bbbb` at
    /// its 8-frame look.
    fn followed(book: &mut Book, id: usize) {
        add(book, id, 100, b"aaaa");
        let candidate = candidate_for(book, id, 108);
        book.detect(&candidate, blake3::hash(b"bbbb"), b"bbbb", 108).expect("tracked");
    }

    /// The same bytes hash the same: nothing is followed.
    #[test]
    fn unchanged_is_not_detected() {
        let mut book = Book::default();
        add(&mut book, 1, 100, b"aaaa");
        let candidate = candidate_for(&mut book, 1, 108);
        assert_eq!(candidate.hash, blake3::hash(b"aaaa"));
        assert!(book.detect(&candidate, blake3::hash(b"aaaa"), b"aaaa", 108).is_none());
        assert!(book.resolve(200).is_empty());
        assert_eq!(book.figures().stale, 0);
    }

    /// A change found is followed; the texture uploaded again later is late,
    /// not stale, and counted by how late (21 frames after its upload: the
    /// first bucket).
    #[test]
    fn changed_and_uploaded_again_later_is_late_not_stale() {
        let mut book = Book::default();
        followed(&mut book, 7);
        add(&mut book, 7, 120, b"bbbb");
        assert!(book.resolve(121).is_empty());
        assert_eq!(book.figures().again, [1, 0, 0]);
        assert_eq!(book.figures().stale, 0);
    }

    /// The fault: a change sampled after it was seen, never uploaded again
    /// within 600 frames, counted once. (This test held the old rule, that a
    /// change never uploaded again is the fault whether or not anything drew
    /// the texture since: corrected to sample it after the change was seen.)
    #[test]
    fn changed_sampled_and_never_uploaded_again_is_stale() {
        let mut book = Book::default();
        add(&mut book, 3, 100, b"aaaa");
        let candidate = candidate_for(&mut book, 3, 108);
        let tracked = book.detect(&candidate, blake3::hash(b"bbbb"), b"bbbb", 108).expect("tracked");
        assert_eq!((tracked.differing, tracked.first, tracked.uploaded_zero), (Some(4), Some(0), Some(false)));
        assert_eq!(tracked.placed, Some(true), "the kind is the upload's");
        book.note_sampled(3, 109);
        assert!(book.resolve(699).is_empty(), "not yet 600 frames");
        let stale = book.resolve(700);
        assert_eq!(stale.len(), 1);
        assert_eq!((book.figures().stale, book.figures().stale_textures), (1, 1));
        // Once: it is out of the following set.
        assert!(book.resolve(701).is_empty());
        assert_eq!(book.figures().stale, 1);
    }

    /// A change found and followed that no list sampled after it was seen
    /// is not the fault: nothing drew the texture from the stale copy. A
    /// sample in the frame it was seen is not after it.
    #[test]
    fn a_change_not_sampled_after_it_was_seen_is_not_stale() {
        let mut book = Book::default();
        add(&mut book, 3, 100, b"aaaa");
        let candidate = candidate_for(&mut book, 3, 108);
        book.detect(&candidate, blake3::hash(b"bbbb"), b"bbbb", 108).expect("tracked");
        book.note_sampled(3, 108);
        book.resolve(700);
        assert_eq!(book.figures().stale, 0, "never drawn again: no stale picture");
        assert_eq!(book.figures().unused, 1);
    }

    /// A texture the title released while followed is not the fault, though
    /// it was sampled; a texture made later at the same address is another,
    /// followed on its own.
    #[test]
    fn a_released_texture_is_not_stale() {
        let mut book = Book::default();
        followed(&mut book, 8);
        book.note_sampled(8, 200);
        book.released(8);
        book.resolve(700);
        assert_eq!((book.figures().stale, book.figures().released), (0, 1));
        // The new texture at the address: uploaded, changed, sampled.
        add(&mut book, 8, 800, b"cccc");
        let candidate = candidate_for(&mut book, 8, 808);
        book.detect(&candidate, blake3::hash(b"dddd"), b"dddd", 808).expect("the new texture is followed");
        book.note_sampled(8, 900);
        book.resolve(1400);
        assert_eq!((book.figures().stale, book.figures().released), (1, 1));
    }

    /// A texture whose staging buffer was let go keeps its last sample: let
    /// go after a list sampled it past the change, and not uploaded again,
    /// it is the fault.
    #[test]
    fn a_sample_before_a_let_go_still_counts() {
        let mut book = Book::default();
        followed(&mut book, 9);
        book.note_sampled(9, 150);
        assert_eq!(book.due(699), Vec::<usize>::new());
        assert_eq!(book.due(700), vec![9]);
        book.resolve(700);
        assert_eq!(book.figures().stale, 1);
    }

    /// Two faults of one texture are two events and one texture.
    #[test]
    fn faults_are_counted_as_events_and_as_textures() {
        let mut book = Book::default();
        followed(&mut book, 10);
        book.note_sampled(10, 200);
        book.resolve(700);
        add(&mut book, 10, 800, b"cccc");
        let candidate = candidate_for(&mut book, 10, 808);
        book.detect(&candidate, blake3::hash(b"dddd"), b"dddd", 808).expect("tracked");
        book.note_sampled(10, 900);
        book.resolve(1400);
        assert_eq!((book.figures().stale, book.figures().stale_textures), (2, 1));
    }

    /// Writer evidence lasts as long as the last look: a storage write 40
    /// frames after the upload, first seen at the 120-frame look, is the
    /// storage's.
    #[test]
    fn writer_evidence_lasts_as_long_as_the_last_look() {
        let mut book = Book::default();
        add(&mut book, 4, 100, b"aaaa");
        book.completed(4 * 0x1000, 4 * 0x1000 + 4, 140);
        book.prune(220);
        let candidate = candidate_for(&mut book, 4, 220);
        let tracked = book.detect(&candidate, blake3::hash(b"bbbb"), b"bbbb", 220).expect("tracked");
        assert!(tracked.storage_done_since);
        assert_eq!(tracked.writer(), Writer::Storage);
    }

    /// Each of the three looks is taken: after the 8- and 30-frame looks the
    /// upload is looked at again at 120 frames. (The looks were marked by
    /// the age's lowest set bit, which 8 and 120 share, so the 120-frame
    /// look was never taken.)
    #[test]
    fn the_last_look_is_taken_at_120_frames() {
        let mut book = Book::default();
        add(&mut book, 6, 100, b"aaaa");
        candidate_for(&mut book, 6, 108);
        candidate_for(&mut book, 6, 130);
        assert!(book.candidates(219).is_empty(), "nothing due between the looks");
        assert_eq!(book.candidates(220).len(), 1, "the 120-frame look");
        assert!(book.candidates(10_000).is_empty(), "every look taken");
    }

    /// A change whose writer evidence was dropped before it was seen is not
    /// one with no writer: an event after the upload went (here by age) and
    /// may have been its writer.
    #[test]
    fn a_change_whose_writer_evidence_was_dropped_is_not_without_a_writer() {
        let mut book = Book::default();
        add(&mut book, 5, 100, b"aaaa");
        book.landing(0x9_0000, 0x9_0004, 105);
        book.prune(346);
        let candidate = candidate_for(&mut book, 5, 346);
        book.detect(&candidate, blake3::hash(b"bbbb"), b"bbbb", 346).expect("tracked");
        book.note_sampled(5, 400);
        book.resolve(700);
        assert_eq!(book.figures().writers, [0, 0, 0, 1], "the evidence since the upload is not whole");
        // With the evidence whole the same change has no writer seen.
        let mut book = Book::default();
        add(&mut book, 5, 100, b"aaaa");
        let candidate = candidate_for(&mut book, 5, 108);
        book.detect(&candidate, blake3::hash(b"bbbb"), b"bbbb", 108).expect("tracked");
        book.note_sampled(5, 400);
        book.resolve(700);
        assert_eq!(book.figures().writers, [0, 0, 1, 0]);
    }

    /// Every loss is counted: records dropped before their last look, follows
    /// dropped past the most followed, lines not printed.
    #[test]
    fn losses_are_counted() {
        let mut book = Book::default();
        for i in 0..MAX_UPLOADS + 10 {
            add(&mut book, i, 1, b"a");
        }
        assert_eq!(book.records.len(), MAX_UPLOADS);
        assert_eq!(book.records.front().expect("a record").id, 10);
        assert_eq!(book.figures().uploads, (MAX_UPLOADS + 10) as u64);
        assert_eq!(book.figures().not_looked, 10, "evicted before any look");
        let mut book = Book::default();
        for i in 0..=TRACKED_MOST {
            add(&mut book, i, 100, b"a");
        }
        // At most `LOOKS_PER_FRAME` a call: taken until none is left.
        loop {
            let candidates = book.candidates(108);
            if candidates.is_empty() {
                break;
            }
            for candidate in candidates {
                book.detect(&candidate, blake3::hash(b"b"), b"b", 108).expect("tracked");
            }
        }
        assert_eq!(book.figures().follows_dropped, 1);
        let item = book.tracked[0];
        for _ in 0..STALE_PER_GROUP {
            assert!(book.may_log(&item));
        }
        assert!(!book.may_log(&item));
        assert_eq!(book.figures().lines_capped, 1);
    }

    /// A source larger than the looks' byte budget is not watched, and said;
    /// one with no hash known is not watched either.
    #[test]
    fn sources_the_budget_cannot_read_or_with_no_hash_are_not_watched() {
        let mut book = Book::default();
        let mut big = source(1, b"a");
        big.size = LOOK_BYTES_PER_FRAME + 1;
        book.add(big, 1, None);
        let mut unknown = source(2, b"a");
        unknown.hash = None;
        book.add(unknown, 1, None);
        assert!(book.records.is_empty());
        assert_eq!((book.figures().uploads, book.figures().too_large), (0, 1));
    }

    /// Changed bytes that were zero in the uploaded copy say so.
    #[test]
    fn uploaded_bytes_that_differed_were_zero() {
        let mut book = Book::default();
        add(&mut book, 5, 100, &[0u8; 4]);
        let candidate = candidate_for(&mut book, 5, 108);
        let tracked = book.detect(&candidate, blake3::hash(b"bbbb"), b"bbbb", 108).expect("tracked");
        assert_eq!((tracked.differing, tracked.first, tracked.uploaded_zero), (Some(4), Some(0), Some(true)));
    }

    /// The switch going off is seen once, at the first boundary that finds it
    /// off; it forgets the book and moves the staging buffers' epoch, so a
    /// change followed before is never resolved after it comes on again.
    #[test]
    fn the_switch_going_off_forgets_what_was_followed() {
        let seen = AtomicBool::new(false);
        assert!(!went_off(&seen, true), "on from the start");
        assert!(!went_off(&seen, true));
        assert!(went_off(&seen, false), "off");
        assert!(!went_off(&seen, false), "once");
        assert!(!went_off(&seen, true), "on again");
        let before = epoch();
        with_book(|book| followed(book, 11));
        switched_off();
        assert!(epoch() > before);
        assert!(BOOK.lock().unwrap().is_none(), "nothing kept");
        assert!(with_book(|book| book.resolve(700)).is_empty());
    }

    /// With the switch off nothing is recorded.
    #[test]
    fn the_switch_off_records_nothing() {
        if std::env::var_os("VICTORY_TEXTURE_RECHECK").is_some() {
            return; // a run with the switch is not this test
        }
        uploaded(1, source(1, b"aaaa"));
        frame(2);
        let figures = figures();
        assert_eq!(figures.stale, 0);
        assert!(BOOK.lock().unwrap().as_ref().is_none_or(|book| book.records.iter().all(|r| r.id != 1)));
    }

    /// The trap's own reads of title memory are counted under a caller of
    /// their own, never as a submission's copy.
    #[test]
    fn the_traps_reads_are_counted_under_their_own_caller() {
        use crate::kernelstat::here;
        let bytes = vec![7u8; 8192];
        let (own, copies) = (here("read", "recheck"), here("read", "copy_mapped"));
        assert_eq!(read_range(bytes.as_ptr() as usize, bytes.len()), bytes);
        assert_eq!(here("read", "recheck") - own, 1);
        assert_eq!(here("read", "copy_mapped") - copies, 0);
    }
}
