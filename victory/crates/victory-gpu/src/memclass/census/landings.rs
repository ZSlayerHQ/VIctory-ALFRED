//! The copy-back's landed bytes by the title allocation they land in, with
//! diagnostics and `VICTORY_LANDED_CLASSES=1` (read once; it turns the
//! memory census on, whose lookup and classes it uses). The question it
//! answers: of the bytes the runtime stores into title memory after the
//! card wrote them (every store of `crate::host::write_mapped`), how many
//! land in write-combined allocations, which a console processor writes and
//! would not sensibly read back (`jobs/2026-11-26-cbj-review`, the review of
//! the back-reads design, step 1).
//!
//! **Per store** (`[at, at + len)`, the bytes the kernel wrote): its bytes
//! go to the allocation holding them as the census's own tables give it
//! ([`State::holder`]: the newest `XMemVirtualAlloc` call over the byte,
//! else the newest `XMemAlloc` block, else none), split exactly where the
//! allocations change; bytes in none go to the `unknown` row. A row is the
//! allocation's span as the census holds it (a call's range less what newer
//! calls took), with its signature (kind, XMem flags and protection, or
//! `XMemAlloc`'s attributes). *Write-combined* is the protection's 0x400
//! (`PAGE_WRITECOMBINE`); an `XMemAlloc` block or no allocation is "flags
//! not known". The census class is the page's at the store
//! ([`Page::class`]; after the store's own card mark, so mostly *card only*
//! or *unknown* split as the census splits it).
//!
//! **Per window** of [`WINDOW`] presented frames (`crate::queue::presented`,
//! the census's frame), started again at each boundary, the last finished
//! one kept: bytes and stores a frame, the 64 KiB pieces a frame (what the
//! fetch's cost follows) and the distinct pages of the window, in all, by
//! write-combined or not, by census class and by allocation; and beside
//! them the bytes and calls the merges asked `write_mapped` to store (the
//! frame line's "MB merged into title memory"; the kernel may write less).
//! A page counts once in a window and once in each frame: in the row, class
//! and protection of its first landing there (so pages and pieces, like
//! bytes, sum to the whole); a store counts once in each row, class and
//! protection it touches. Also, per page landed in the window: whether the
//! census saw the title's processor write it in the same window (its look
//! found it changed with no card or storage mark since the last look). That
//! is a lower bound: a processor write between a landing and the next look
//! counts as the card's (the census's own rule), so on a page landed before
//! every look it is never seen, and 0 proves nothing.
//!
//! **Aliases**: a page the title's views of its physical pages show at
//! another address too (the census's `shown`, turned round in
//! [`Landed`]'s `views`, one entry for each of its own) is counted again by
//! the pair of protections, the one stored through and the other
//! address's, so a reader knows how much of "write-combined" is also
//! reachable through memory that is not. Private memory has no alias here.
//!
//! **What the share is**: of the bytes stored, by the protection of the
//! reservation they were stored through. It is not evidence the title
//! never reads them, nor the fetch's cost (the pieces are). Read a window
//! wholly inside the scene, never the first (it starts at boot) nor the one
//! at exit.
//!
//! **Bound**: [`ROWS`] allocations a window; stores into others go to the
//! `other` row, counted, so no byte is dropped and the splits by protection
//! and class stay whole. Two windows' tables are made once, at the start.
//!
//! **Locks**: none of its own. Every count is made under the census's lock
//! (`STATE`, a leaf lock), which the landing already takes for its card
//! mark; the stores from the relay thread, the submitting thread
//! (`VICTORY_RELAY=0`) and any other are serialized by it. A window is
//! finished under that lock and its report made there whole (the log lines
//! and the stats file's object), so a reader never sees half a window; the
//! lines are logged after the lock is let go, at the next present (never on
//! the landing path).
//!
//! **Output**: lines starting "landed classes" each [`WINDOW`] frames and at
//! the title's exit (the window so far); the last finished window as
//! `landed_classes` in the live stats file (no such key with the switch
//! off). No address is printed (the census prints none): an allocation is
//! its signature and size.

use super::{Backing, Class, Page, State, pages, with};
use crate::watch::PAGE;
use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Frames in a window.
pub(crate) const WINDOW: u64 = 600;
/// Allocations tabled in a window; stores into more go to `other`.
pub(crate) const ROWS: usize = 256;
/// Allocations a report names, most bytes first.
pub(crate) const TOP: usize = 12;

/// Row numbers past the table: stores into allocations it had no room for,
/// and into none the census holds.
const OTHER: u16 = u16::MAX - 1;
const UNKNOWN: u16 = u16::MAX;

/// By the allocation's protection.
pub(crate) const WC: usize = 0;
pub(crate) const NOT_WC: usize = 1;
pub(crate) const FLAGS_NOT_KNOWN: usize = 2;
const WC_NAMES: [&str; 3] = ["write-combined", "not write-combined", "flags not known"];
const WC_KEYS: [&str; 3] = ["write_combined", "not_write_combined", "flags_not_known"];

/// By the page's census class: its four classes, unknown split as the
/// census splits it, and pages it never looked at.
pub(crate) const CLASSES: usize = 7;
pub(crate) const CARD_ONLY: usize = 2;
#[cfg(test)]
pub(crate) const BOTH_WRITE: usize = 4;
pub(crate) const NOT_LOOKED: usize = 6;
const CLASS_NAMES: [&str; CLASSES] = [
    "loaded once",
    "changes every frame",
    "card only",
    "unknown (changes sometimes)",
    "unknown (the card and the processor both write)",
    "unknown (copied whole)",
    "not looked at",
];

/// `PAGE_WRITECOMBINE`, in the protection the title passes.
const WRITE_COMBINE: u32 = 0x400;

/// Whether the table runs: `VICTORY_LANDED_CLASSES=1`, read once.
pub(crate) fn switch() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_LANDED_CLASSES").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                "landed classes on (VICTORY_LANDED_CLASSES=1): the copy-back's landed bytes by allocation, \
                 a report each {WINDOW} frames; the memory census runs with it"
            );
        }
        on
    })
}

/// A page's landings, in its census record. Window fields hold the window
/// plus one (0: none); frame the frame plus one.
#[derive(Clone, Copy, Default, Debug)]
pub(super) struct Mark {
    /// The frame it last landed in (pages a frame).
    frame: u64,
    /// The window it last landed in, and where its first landing there was
    /// counted: row, class, protection.
    window: u32,
    row: u16,
    class: u8,
    wc: u8,
    /// The window the census last saw the processor write it in.
    processor: u32,
}

/// Bytes, stores, distinct pages, and of those the pages the processor was
/// seen writing in the window; and pieces: the 64 KiB pages landed in each
/// frame, summed over the window's frames (a page once a frame, where its
/// first landing in that frame is counted), what the fetch's cost follows.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct Tally {
    pub(crate) bytes: u64,
    pub(crate) stores: u64,
    pub(crate) pages: u64,
    pub(crate) pieces: u64,
    pub(crate) processor: u64,
}

/// What the merges asked to store (`write_mapped`'s calls and their
/// bytes, the frame line's "MB merged into title memory"), beside what the
/// kernel wrote ([`Tally`]).
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct Asked {
    pub(crate) bytes: u64,
    pub(crate) calls: u64,
}

/// Landed bytes on pages the title's views also show at another address:
/// bytes, pieces (as [`Tally`]'s) and distinct pages.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct Alias {
    pub(crate) bytes: u64,
    pub(crate) pieces: u64,
    pub(crate) pages: u64,
}

/// An allocation: layer (0 `XMemVirtualAlloc`, 1 `XMemAlloc`), span start
/// and end, signature.
type Key = (u8, usize, usize, u32);

struct Row {
    key: Key,
    wc: u8,
    tally: Tally,
    /// Bytes by class.
    classes: [u64; CLASSES],
}

/// One window's counts.
#[derive(Default)]
struct Window {
    /// The window (frame / [`WINDOW`]), the first frame counted in it and
    /// the last seen.
    number: u64,
    from: u64,
    last: u64,
    rows: Vec<Row>,
    index: HashMap<Key, u16>,
    other: Tally,
    unknown: Tally,
    all: Tally,
    wc: [Tally; 3],
    classes: [Tally; CLASSES],
    asked: Asked,
    /// By the protection stored through, then the other address's.
    aliases: [[Alias; 3]; 3],
    /// Nanoseconds spent counting.
    own_ns: u64,
}

/// The table: the window being counted, the room of the one before (made
/// once: no allocation after the start), the protection of each
/// `XMemVirtualAlloc` signature, the last finished window's stats object,
/// and its lines until they are logged.
pub(crate) struct Landed {
    started: bool,
    current: Window,
    spare: Window,
    protections: HashMap<u32, u32>,
    /// The title's views, by view page then address page: the census's
    /// `shown` turned round (one entry for each of its own), kept from
    /// the start so aliases are known.
    views: BTreeSet<(Backing, usize)>,
    previous: Option<String>,
    pending: Vec<String>,
}

/// A window's figures, as counted (not per frame).
#[derive(Debug, Default)]
pub(crate) struct Figures {
    pub(crate) from: u64,
    pub(crate) to: u64,
    pub(crate) all: Tally,
    pub(crate) wc: [Tally; 3],
    pub(crate) classes: [Tally; CLASSES],
    /// Every tabled allocation, most bytes first.
    pub(crate) rows: Vec<RowFigures>,
    pub(crate) other: Tally,
    pub(crate) unknown: Tally,
    pub(crate) asked: Asked,
    pub(crate) aliases: [[Alias; 3]; 3],
    pub(crate) own_ns: u64,
}

#[derive(Debug)]
pub(crate) struct RowFigures {
    pub(crate) layer: u8,
    pub(crate) signature: String,
    pub(crate) size: usize,
    pub(crate) wc: u8,
    pub(crate) tally: Tally,
    pub(crate) classes: [u64; CLASSES],
}

/// The layers' names (`super::LAYERS`' first two).
const LAYER_NAMES: [&str; 2] = ["XMemVirtualAlloc", "XMemAlloc"];

/// A page's class bucket ([`CLASS_NAMES`]), as the census would report it
/// now: a page it never looked at is "not looked at".
fn bucket(p: &Page) -> usize {
    if p.looks == 0 {
        return NOT_LOOKED;
    }
    match p.class() {
        Class::LoadedOnce => 0,
        Class::EveryFrame => 1,
        Class::CardOnly => CARD_ONLY,
        Class::Unknown => 3 + p.detail(),
    }
}

/// What one page's share of a store is, beside its bytes: its first landing
/// in the window (a distinct page), its first in the frame (a piece), and
/// seen written by the processor in the window before.
#[derive(Clone, Copy)]
struct Firsts {
    page: bool,
    piece: bool,
    processor: bool,
}

/// `bytes` of one page into `t` ([`Firsts`]).
fn put(t: &mut Tally, bytes: u64, firsts: Firsts) {
    t.bytes += bytes;
    t.pages += u64::from(firsts.page);
    t.pieces += u64::from(firsts.piece);
    t.processor += u64::from(firsts.processor);
}

/// The bucket of the protection of the allocation `holder` names: an
/// `XMemVirtualAlloc` call's 0x400, else not known.
fn protection_of(holder: Option<Key>, protections: &HashMap<u32, u32>) -> usize {
    match holder.map(|key| (key.0, protections.get(&key.3))) {
        Some((0, Some(protect))) if protect & WRITE_COMBINE != 0 => WC,
        Some((0, Some(_))) => NOT_WC,
        _ => FLAGS_NOT_KNOWN,
    }
}

impl Window {
    /// Tables with room for [`ROWS`] allocations, made once.
    fn with_room() -> Window {
        Window {
            rows: Vec::with_capacity(ROWS),
            index: HashMap::with_capacity(ROWS),
            ..Window::default()
        }
    }

    /// Empty again, for window `number` from frame `from`, keeping its room.
    fn reset(&mut self, number: u64, from: u64) {
        let (mut rows, mut index) = (std::mem::take(&mut self.rows), std::mem::take(&mut self.index));
        rows.clear();
        index.clear();
        *self = Window {
            number,
            from,
            last: from,
            rows,
            index,
            ..Window::default()
        };
    }

    /// The window's mark in a page's [`Mark`]: its number plus one (a
    /// window is 600 frames: no wrap in any run).
    fn mark(&self) -> u32 {
        self.number as u32 + 1
    }

    /// The row of the allocation `holder` names (a new one while there is
    /// room, else `other`; `unknown` for none), and its protection.
    fn row(&mut self, holder: Option<Key>, protections: &HashMap<u32, u32>) -> (u16, usize) {
        let wc = protection_of(holder, protections);
        let Some(key) = holder else {
            return (UNKNOWN, wc);
        };
        if let Some(&row) = self.index.get(&key) {
            return (row, wc);
        }
        if self.rows.len() >= ROWS {
            return (OTHER, wc);
        }
        let row = self.rows.len() as u16;
        self.rows.push(Row {
            key,
            wc: wc as u8,
            tally: Tally::default(),
            classes: [0; CLASSES],
        });
        self.index.insert(key, row);
        (row, wc)
    }

    fn tally(&mut self, row: u16) -> &mut Tally {
        match row {
            OTHER => &mut self.other,
            UNKNOWN => &mut self.unknown,
            row => &mut self.rows[usize::from(row)].tally,
        }
    }

    /// `bytes` of one page landed in `row`, of protection `wc` and class
    /// `class` ([`put`]).
    fn add(&mut self, row: u16, wc: usize, class: usize, bytes: u64, firsts: Firsts) {
        for t in [&mut self.all, &mut self.wc[wc], &mut self.classes[class]] {
            put(t, bytes, firsts);
        }
        put(self.tally(row), bytes, firsts);
        if let Some(r) = self.rows.get_mut(usize::from(row)) {
            r.classes[class] += bytes;
        }
    }

    /// The same page's bytes, stored through protection `wc`, on a page the
    /// title's views also show where `shown` says (by protection).
    fn alias(&mut self, wc: usize, shown: [bool; 3], bytes: u64, firsts: Firsts) {
        for (cell, shown) in self.aliases[wc].iter_mut().zip(shown) {
            if shown {
                cell.bytes += bytes;
                cell.pieces += u64::from(firsts.piece);
                cell.pages += u64::from(firsts.page);
            }
        }
    }

    /// A page landed in this window, counted at `row`, `wc`, `class`, seen
    /// written by the processor.
    fn processor(&mut self, row: u16, wc: usize, class: usize) {
        for t in [&mut self.all, &mut self.wc[wc], &mut self.classes[class]] {
            t.processor += 1;
        }
        self.tally(row).processor += 1;
    }

    /// The figures, the window ending before frame `to`.
    fn figures(&self, to: u64, signatures: &[String]) -> Figures {
        let mut rows: Vec<RowFigures> = self
            .rows
            .iter()
            .map(|r| RowFigures {
                layer: r.key.0,
                signature: signatures.get(r.key.3 as usize).cloned().unwrap_or_default(),
                size: r.key.2 - r.key.1,
                wc: r.wc,
                tally: r.tally,
                classes: r.classes,
            })
            .collect();
        rows.sort_by(|a, b| b.tally.bytes.cmp(&a.tally.bytes).then_with(|| a.signature.cmp(&b.signature)));
        Figures {
            from: self.from,
            to: to.max(self.from + 1),
            all: self.all,
            wc: self.wc,
            classes: self.classes,
            rows,
            other: self.other,
            unknown: self.unknown,
            asked: self.asked,
            aliases: self.aliases,
            own_ns: self.own_ns,
        }
    }
}

impl Landed {
    pub(crate) fn new() -> Landed {
        Landed {
            started: false,
            current: Window::with_room(),
            spare: Window::with_room(),
            protections: HashMap::new(),
            views: BTreeSet::new(),
            previous: None,
            pending: Vec::new(),
        }
    }

    /// The census saw `XMemVirtualAlloc` signature `id` made with `protect`.
    pub(super) fn protection(&mut self, id: u32, protect: u32) {
        self.protections.insert(id, protect);
    }

    /// Address page `page` shows view page `backing` now, in place of `old`.
    pub(super) fn shown(&mut self, page: usize, old: Option<Backing>, backing: Backing) {
        if let Some(old) = old {
            self.views.remove(&(old, page));
        }
        self.views.insert((backing, page));
    }

    /// Address page `page` no longer shows `backing`.
    pub(super) fn unshown(&mut self, page: usize, backing: Backing) {
        self.views.remove(&(backing, page));
    }

    /// Counting at `frame`: a window that ended before it is finished, its
    /// lines and stats object made now, under the census's lock, whole.
    pub(super) fn at(&mut self, frame: u64, signatures: &[String]) {
        let number = frame / WINDOW;
        if !self.started {
            self.started = true;
            self.current.reset(number, frame);
            return;
        }
        // Frames only go forward (the census's frame is the presents').
        if number <= self.current.number {
            self.current.last = self.current.last.max(frame);
            return;
        }
        let to = (self.current.number + 1) * WINDOW;
        std::mem::swap(&mut self.current, &mut self.spare);
        let figures = self.spare.figures(to, signatures);
        self.pending = figures.lines(&format!("every {WINDOW} frames"));
        self.previous = Some(figures.json());
        self.current.reset(number, number * WINDOW);
    }

    /// The census saw the processor write the page whose mark is `mark`:
    /// counted once a window, if the page landed in it (else at its
    /// landing, if one follows in the window).
    pub(super) fn processor_wrote(&mut self, mark: &mut Mark) {
        let window = self.current.mark();
        if mark.processor == window {
            return;
        }
        mark.processor = window;
        if mark.window == window {
            self.current.processor(mark.row, usize::from(mark.wc), usize::from(mark.class));
        }
    }
}

impl State {
    /// The allocation holding byte `at` as the census's tables give it
    /// ([`super::Layer::span`], the lookup its report uses): the newest
    /// `XMemVirtualAlloc` call over it, else the newest `XMemAlloc` block,
    /// else none; and where that answer ends, before `end` (`at < end`).
    fn holder(&self, at: usize, end: usize) -> (usize, Option<Key>) {
        if let Some((start, stop, id)) = self.layers[0].span(at) {
            return (stop.min(end), Some((0, start, stop, id)));
        }
        let limit = self.layers[0].next(at, end).unwrap_or(end);
        if let Some((start, stop, id)) = self.layers[1].span(at) {
            return (stop.min(limit), Some((1, start, stop, id)));
        }
        (self.layers[1].next(at, limit).unwrap_or(limit), None)
    }

    /// The bytes the kernel stored at `[at, at + len)` after the card wrote
    /// them (with the card's mark made first: [`landed`]), split by the
    /// allocations holding them and by page. Nothing with the table off.
    pub(crate) fn landed(&mut self, frame: u64, at: usize, len: usize) {
        if len == 0 {
            return;
        }
        let Some(mut landed) = self.landed.take() else {
            return;
        };
        let started = std::time::Instant::now();
        landed.at(frame, &self.signatures);
        let l = &mut *landed;
        let now = frame + 1;
        let window = l.current.mark();
        let end = at.saturating_add(len);
        let (mut wcs, mut classes) = ([false; 3], [false; CLASSES]);
        let mut last_row = None;
        l.current.all.stores += 1;
        let mut cursor = at;
        while cursor < end {
            let (stop, holder) = self.holder(cursor, end);
            let (row, wc) = l.current.row(holder, &l.protections);
            if last_row != Some(row) {
                l.current.tally(row).stores += 1;
                last_row = Some(row);
            }
            wcs[wc] = true;
            for page in pages(cursor, stop - cursor) {
                let bytes = (stop.min((page + 1) * PAGE) - cursor.max(page * PAGE)) as u64;
                let backing = self.backing(page);
                let p = self.page(page);
                let class = bucket(p);
                classes[class] = true;
                let m = &mut p.landed;
                let first = m.window != window;
                let firsts = Firsts {
                    page: first,
                    piece: m.frame != now,
                    processor: first && m.processor == window,
                };
                m.frame = now;
                if first {
                    (m.window, m.row, m.class, m.wc) = (window, row, class as u8, wc as u8);
                }
                l.current.add(row, wc, class, bytes, firsts);
                if let Backing::View(..) = backing {
                    let shown = self.shown_elsewhere(&l.views, &l.protections, backing, page);
                    l.current.alias(wc, shown, bytes, firsts);
                }
            }
            cursor = stop;
        }
        for (t, touched) in l.current.wc.iter_mut().zip(wcs) {
            t.stores += u64::from(touched);
        }
        for (t, touched) in l.current.classes.iter_mut().zip(classes) {
            t.stores += u64::from(touched);
        }
        l.current.own_ns += started.elapsed().as_nanos() as u64;
        self.landed = Some(landed);
    }

    /// By protection, whether a title view shows view page `backing` at an
    /// address page other than `page` (its allocation as [`State::holder`]
    /// gives it).
    fn shown_elsewhere(
        &self,
        views: &BTreeSet<(Backing, usize)>,
        protections: &HashMap<u32, u32>,
        backing: Backing,
        page: usize,
    ) -> [bool; 3] {
        let mut shown = [false; 3];
        for &(_, other) in views.range((backing, 0)..=(backing, usize::MAX)) {
            if other != page {
                let address = other * PAGE;
                shown[protection_of(self.holder(address, address + 1).1, protections)] = true;
            }
        }
        shown
    }

    /// A merge asked `write_mapped` to store `len` bytes at `frame`.
    pub(crate) fn landed_asked(&mut self, frame: u64, len: usize) {
        let signatures = &self.signatures;
        if let Some(l) = self.landed.as_mut() {
            l.at(frame, signatures);
            l.current.asked.bytes += len as u64;
            l.current.asked.calls += 1;
        }
    }

    /// A present at `frame`: the lines of a window finished, to log.
    pub(crate) fn landed_tick(&mut self, frame: u64) -> Vec<String> {
        let signatures = &self.signatures;
        match self.landed.as_mut() {
            Some(l) => {
                l.at(frame, signatures);
                std::mem::take(&mut l.pending)
            }
            None => Vec::new(),
        }
    }

    /// The window so far, as the title exits, after any finished one's.
    pub(crate) fn landed_exit_lines(&mut self, frame: u64) -> Vec<String> {
        let signatures = &self.signatures;
        let Some(l) = self.landed.as_mut() else {
            return Vec::new();
        };
        l.at(frame, signatures);
        let mut lines = std::mem::take(&mut l.pending);
        lines.extend(l.current.figures(frame, signatures).lines("at exit"));
        lines
    }

    /// The window so far; `None` with the table off (tests).
    #[cfg(test)]
    pub(crate) fn landed_figures(&self) -> Option<Figures> {
        let l = self.landed.as_ref()?;
        Some(l.current.figures(l.current.last + 1, &self.signatures))
    }

    /// The last finished window's stats object, if one finished.
    pub(crate) fn landed_previous(&self) -> Option<String> {
        self.landed.as_ref()?.previous.clone()
    }

    /// The lines of the last finished window not yet logged (tests; the
    /// process's are taken by [`State::landed_tick`]).
    #[cfg(test)]
    pub(crate) fn landed_pending(&mut self) -> Vec<String> {
        self.landed.as_mut().map(|l| std::mem::take(&mut l.pending)).unwrap_or_default()
    }
}

/// A JSON string's contents (signatures hold neither, but the object stays
/// one whatever they hold).
fn escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

impl Figures {
    fn frames(&self) -> f64 {
        (self.to - self.from).max(1) as f64
    }

    /// The report's lines; `why` says when. Megabytes are 10^6 bytes, as
    /// the frame line's copy-back megabytes.
    pub(crate) fn lines(&self, why: &str) -> Vec<String> {
        let frames = self.frames();
        let mb = |bytes: u64| bytes as f64 / 1e6 / frames;
        let total = self.all.bytes;
        let share = |bytes: u64| if total == 0 { 0.0 } else { bytes as f64 * 100.0 / total as f64 };
        let per = |n: u64| n as f64 / frames;
        let part = |t: &Tally| {
            format!(
                "{:.2} MB a frame ({:.1}% of the bytes stored, {:.1} pieces of 64 KiB a frame, {} distinct pages, \
                 {} seen written by the processor)",
                mb(t.bytes),
                share(t.bytes),
                per(t.pieces),
                t.pages,
                t.processor
            )
        };
        let (from, to) = (self.from, self.to);
        let mut lines = vec![format!(
            "landed classes ({why}, frames {from}-{to}, {} frames; its own time {:.2} ms a frame): asked to store \
             {:.2} MB a frame in {:.1} calls a frame (the frame line's MB merged into title memory); stored {:.2} MB \
             a frame in {:.1} stores a frame, {:.1} pieces of 64 KiB a frame, {} distinct pages, {} of them seen \
             written by the processor in the window (a lower bound: 0 proves nothing); by the protection of the \
             reservation they were stored through (a share of the bytes stored, not evidence the title never reads \
             them): write-combined {}, not write-combined {}, flags not known {} (an XMemAlloc block, or no \
             allocation)",
            frames as u64,
            self.own_ns as f64 / 1e6 / frames,
            mb(self.asked.bytes),
            per(self.asked.calls),
            mb(total),
            per(self.all.stores),
            per(self.all.pieces),
            self.all.pages,
            self.all.processor,
            part(&self.wc[WC]),
            part(&self.wc[NOT_WC]),
            part(&self.wc[FLAGS_NOT_KNOWN]),
        )];
        let classes: Vec<String> = CLASS_NAMES
            .iter()
            .zip(&self.classes)
            .filter(|(_, t)| t.bytes > 0)
            .map(|(name, t)| format!("{name} {}", part(t)))
            .collect();
        lines.push(format!(
            "landed classes by census class (frames {from}-{to}): {}",
            if classes.is_empty() { "nothing landed".to_owned() } else { classes.join("; ") }
        ));
        let mut aliases = Vec::new();
        for (stored, cells) in self.aliases.iter().enumerate() {
            for (shown, a) in cells.iter().enumerate().filter(|(_, a)| a.bytes > 0) {
                aliases.push(format!(
                    "stored through {}, also shown {}: {:.2} MB a frame ({:.1}% of the bytes stored), {:.1} pieces \
                     of 64 KiB a frame, {} distinct pages",
                    WC_NAMES[stored],
                    WC_NAMES[shown],
                    mb(a.bytes),
                    share(a.bytes),
                    per(a.pieces),
                    a.pages
                ));
            }
        }
        lines.push(format!(
            "landed classes, pages also shown at another address (frames {from}-{to}; by the title's views of its \
             physical pages: the bytes stored through one are the other's; a page shown at two others counts under \
             each): {}",
            if aliases.is_empty() { "none".to_owned() } else { aliases.join("; ") }
        ));
        for (rank, r) in self.rows.iter().take(TOP).enumerate() {
            let classes: Vec<String> = CLASS_NAMES
                .iter()
                .zip(&r.classes)
                .filter(|&(_, &bytes)| bytes > 0)
                .map(|(name, &bytes)| format!("{name} {:.2}", mb(bytes)))
                .collect();
            lines.push(format!(
                "landed classes: allocation {} of {}: {} {}, {:.1} MB, {}: {}; {:.1} stores a frame; MB a frame by class: {}",
                rank + 1,
                self.rows.len(),
                LAYER_NAMES[usize::from(r.layer)],
                r.signature,
                r.size as f64 / 1e6,
                WC_NAMES[usize::from(r.wc)],
                part(&r.tally),
                r.tally.stores as f64 / frames,
                classes.join(", ")
            ));
        }
        lines.push(format!(
            "landed classes: {} allocations tabled (room for {ROWS}); past the table's {ROWS} allocations: {}, \
             {:.1} stores a frame; in no allocation the census holds: {}, {:.1} stores a frame",
            self.rows.len(),
            part(&self.other),
            self.other.stores as f64 / frames,
            part(&self.unknown),
            self.unknown.stores as f64 / frames
        ));
        lines
    }

    /// The stats file's object (fixed key order): megabytes, calls, stores
    /// and pieces a frame; pages distinct over the window.
    pub(crate) fn json(&self) -> String {
        let frames = self.frames();
        let fields = |t: &Tally| {
            format!(
                "\"mb_a_frame\":{:.3},\"stores_a_frame\":{:.1},\"pieces_a_frame\":{:.1},\"pages_in_window\":{},\
                 \"pages_processor_written_in_window\":{}",
                t.bytes as f64 / 1e6 / frames,
                t.stores as f64 / frames,
                t.pieces as f64 / frames,
                t.pages,
                t.processor
            )
        };
        let mut aliases = Vec::new();
        for (stored, cells) in self.aliases.iter().enumerate() {
            for (shown, a) in cells.iter().enumerate().filter(|(_, a)| a.bytes > 0) {
                aliases.push(format!(
                    "{{\"stored\":\"{}\",\"shown\":\"{}\",\"mb_a_frame\":{:.3},\"pieces_a_frame\":{:.1},\"pages_in_window\":{}}}",
                    WC_NAMES[stored],
                    WC_NAMES[shown],
                    a.bytes as f64 / 1e6 / frames,
                    a.pieces as f64 / frames,
                    a.pages
                ));
            }
        }
        let wc: Vec<String> = WC_KEYS.iter().zip(&self.wc).map(|(k, t)| format!("\"{k}\":{{{}}}", fields(t))).collect();
        let classes: Vec<String> =
            CLASS_NAMES.iter().zip(&self.classes).map(|(k, t)| format!("\"{k}\":{{{}}}", fields(t))).collect();
        let top: Vec<String> = self
            .rows
            .iter()
            .take(TOP)
            .map(|r| {
                format!(
                    "{{\"layer\":\"{}\",\"signature\":\"{}\",\"size\":{},\"protection\":\"{}\",{}}}",
                    LAYER_NAMES[usize::from(r.layer)],
                    escape(&r.signature),
                    r.size,
                    WC_NAMES[usize::from(r.wc)],
                    fields(&r.tally)
                )
            })
            .collect();
        format!(
            "{{\"from\":{},\"to\":{},\"frames\":{},\"own_ms_a_frame\":{:.3},\
             \"asked\":{{\"mb_a_frame\":{:.3},\"calls_a_frame\":{:.1}}},\"all\":{{{}}},{},\
             \"classes\":{{{}}},\"top\":[{}],\"tabled\":{},\"table_rows\":{ROWS},\"other\":{{{}}},\"unknown\":{{{}}},\
             \"aliases\":[{}]}}",
            self.from,
            self.to,
            frames as u64,
            self.own_ns as f64 / 1e6 / frames,
            self.asked.bytes as f64 / 1e6 / frames,
            self.asked.calls as f64 / frames,
            fields(&self.all),
            wc.join(","),
            classes.join(","),
            top.join(","),
            self.rows.len(),
            fields(&self.other),
            fields(&self.unknown),
            aliases.join(",")
        )
    }
}

/// The landing's store (`crate::host::write_mapped`): the card's mark, as
/// [`super::card`], and with the switch on its bytes in the table, under
/// the same lock. Off, the table is `None` and this is the card's mark.
pub(crate) fn landed(at: usize, len: usize) {
    with(|s, f| {
        s.card(f, at, len);
        s.landed(f, at, len);
    });
}

/// A merge asks `crate::host::write_mapped` to store `len` bytes. Off: one
/// load, no lock.
pub(crate) fn landing_asked(len: usize) {
    if !counting() {
        return;
    }
    with(|s, f| s.landed_asked(f, len));
}

/// Whether the table counts: its switch; in tests also this thread's own
/// census with it on ([`super::TestCensus::landed_on`]).
fn counting() -> bool {
    #[cfg(test)]
    if super::TEST.with(|t| t.borrow().as_ref().is_some_and(|(s, _)| s.landed.is_some())) {
        return true;
    }
    switch()
}

/// The live stats file's `landed_classes` object: `None` with the switch
/// off, so the object is as without the table.
pub(crate) fn landed_object() -> Option<String> {
    switch().then(landed_json)
}

/// Each present: a window finished is logged here. Off: one load, no lock.
pub(crate) fn landed_presented() {
    if !switch() {
        return;
    }
    static SEEN: AtomicU64 = AtomicU64::new(0);
    let window = crate::queue::presented() / WINDOW;
    if SEEN.swap(window, Relaxed) == window {
        return;
    }
    let mut lines = Vec::new();
    with(|s, f| lines = s.landed_tick(f));
    for line in lines {
        tracing::info!("{line}");
    }
}

/// The title exits: the window so far.
pub(crate) fn landed_exit() {
    if !switch() {
        return;
    }
    let frame = crate::queue::presented();
    let lines = match &mut *super::STATE.lock().unwrap_or_else(|e| e.into_inner()) {
        Some(state) => state.landed_exit_lines(frame),
        None => return,
    };
    for line in lines {
        tracing::info!("{line}");
    }
}

/// The live stats file's `landed_classes`: the last finished window, or
/// `null` (off, or none finished yet).
pub(crate) fn landed_json() -> String {
    if !switch() {
        return "null".to_owned();
    }
    super::STATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(State::landed_previous)
        .unwrap_or_else(|| "null".to_owned())
}

#[cfg(test)]
mod tests {
    use super::super::{TestCensus, TitleAllocation};
    use super::*;

    /// A synthetic base page number of the tests' own (10 GiB, as 64 KiB
    /// pages); nothing here depends on where it is.
    const P: usize = 0x2_8000_0000 / PAGE;
    /// Two allocations of the tests' own: one whose protection carries
    /// `PAGE_WRITECOMBINE` (0x400), and one whose protection carries the
    /// bit beside it (`PAGE_NOCACHE`, 0x200) but not that one, so it is
    /// not write-combined.
    const COMBINED: TitleAllocation = TitleAllocation::Virtual { kind: 0x1000, flags: 0x11, protect: 0x604 };
    const PLAIN: TitleAllocation = TitleAllocation::Virtual { kind: 0x3000, flags: 0x9, protect: 0x204 };
    const COMBINED_NAME: &str = "kind 0x1000 flags 0x11 protect 0x604";
    const PLAIN_NAME: &str = "kind 0x3000 flags 0x9 protect 0x204";

    /// A census with the table on, as the switch makes the process's.
    fn census() -> State {
        State {
            landed: Some(Box::new(Landed::new())),
            ..State::default()
        }
    }

    /// A landing's store as `write_mapped` makes it: asked ([`landing_asked`]),
    /// then the card's mark and the table ([`landed`]).
    fn land(s: &mut State, frame: u64, at: usize, len: usize) {
        s.landed_asked(frame, len);
        s.card(frame, at, len);
        s.landed(frame, at, len);
    }

    fn tally(bytes: u64, stores: u64, pages: u64, pieces: u64, processor: u64) -> Tally {
        Tally { bytes, stores, pages, pieces, processor }
    }

    /// Every byte is in a row, `other` or `unknown`; the splits by
    /// protection and by class are whole.
    fn whole(f: &Figures) {
        let rows: u64 = f.rows.iter().map(|r| r.tally.bytes).sum();
        assert_eq!(rows + f.other.bytes + f.unknown.bytes, f.all.bytes, "no byte dropped: {f:#?}");
        assert_eq!(f.wc.iter().map(|t| t.bytes).sum::<u64>(), f.all.bytes, "by protection: {f:#?}");
        assert_eq!(f.classes.iter().map(|t| t.bytes).sum::<u64>(), f.all.bytes, "by class: {f:#?}");
        let pages: u64 = f.rows.iter().map(|r| r.tally.pages).sum();
        assert_eq!(pages + f.other.pages + f.unknown.pages, f.all.pages, "pages: {f:#?}");
        assert_eq!(f.wc.iter().map(|t| t.pages).sum::<u64>(), f.all.pages, "pages by protection: {f:#?}");
        assert_eq!(f.classes.iter().map(|t| t.pages).sum::<u64>(), f.all.pages, "pages by class: {f:#?}");
        let pieces: u64 = f.rows.iter().map(|r| r.tally.pieces).sum();
        assert_eq!(pieces + f.other.pieces + f.unknown.pieces, f.all.pieces, "pieces: {f:#?}");
        assert_eq!(f.wc.iter().map(|t| t.pieces).sum::<u64>(), f.all.pieces, "pieces by protection: {f:#?}");
        assert_eq!(f.classes.iter().map(|t| t.pieces).sum::<u64>(), f.all.pieces, "pieces by class: {f:#?}");
    }

    /// A store inside a write-combined allocation counts there, with the
    /// allocation's signature and size and the page's census class; one in
    /// a plain allocation counts as not write-combined.
    #[test]
    fn a_landed_store_in_a_write_combined_allocation_counts_there() {
        let mut s = census();
        s.allocated(P * PAGE, 4 * PAGE, COMBINED);
        s.allocated((P + 8) * PAGE, 4 * PAGE, PLAIN);
        s.looked(0, P, P + 12, &[]);
        land(&mut s, 1, P * PAGE + 100, 1000);
        let f = s.landed_figures().expect("on");
        assert_eq!(f.all, tally(1000, 1, 1, 1, 0));
        assert_eq!(f.wc[WC], tally(1000, 1, 1, 1, 0));
        assert_eq!(f.wc[NOT_WC], Tally::default());
        assert_eq!(f.classes[CARD_ONLY], tally(1000, 1, 1, 1, 0), "looked at, written by the card only: {f:#?}");
        assert_eq!(f.rows.len(), 1, "{f:#?}");
        assert_eq!((f.rows[0].layer, f.rows[0].signature.as_str()), (0, COMBINED_NAME));
        assert_eq!((f.rows[0].size, f.rows[0].wc), (4 * PAGE, WC as u8));
        assert_eq!(f.rows[0].tally, tally(1000, 1, 1, 1, 0));
        assert_eq!(f.rows[0].classes[CARD_ONLY], 1000);
        land(&mut s, 2, (P + 8) * PAGE, 10);
        let f = s.landed_figures().expect("on");
        assert_eq!(f.wc[NOT_WC], tally(10, 1, 1, 1, 0));
        assert_eq!(f.all, tally(1010, 2, 2, 2, 0));
        assert_eq!(f.rows[1].signature, PLAIN_NAME);
        whole(&f);
    }

    /// A store across two allocations is split by bytes; a page the two
    /// share counts once, in the row of its first landing.
    #[test]
    fn a_store_spanning_two_allocations_is_split_by_bytes() {
        let mut s = census();
        s.allocated(P * PAGE, 2 * PAGE, COMBINED);
        s.allocated((P + 2) * PAGE, 2 * PAGE, PLAIN);
        land(&mut s, 1, (P + 2) * PAGE - 300, 1000);
        let f = s.landed_figures().expect("on");
        assert_eq!(f.all, tally(1000, 1, 2, 2, 0));
        assert_eq!(f.wc[WC], tally(300, 1, 1, 1, 0));
        assert_eq!(f.wc[NOT_WC], tally(700, 1, 1, 1, 0));
        let rows: Vec<(&str, u64)> = f.rows.iter().map(|r| (r.signature.as_str(), r.tally.bytes)).collect();
        assert_eq!(rows, [(PLAIN_NAME, 700), (COMBINED_NAME, 300)], "most bytes first");
        whole(&f);
        // Two allocations inside one page: the bytes split, the page once.
        let mut s = census();
        let at = (P + 20) * PAGE;
        s.allocated(at, PAGE / 2, COMBINED);
        s.allocated(at + PAGE / 2, PAGE / 2, PLAIN);
        land(&mut s, 1, at + PAGE / 2 - 0x1000, 0x2000);
        let f = s.landed_figures().expect("on");
        assert_eq!(f.all, tally(0x2000, 1, 1, 1, 0));
        assert_eq!(f.wc[WC], tally(0x1000, 1, 1, 1, 0), "the page's first landing was here");
        assert_eq!(f.wc[NOT_WC], tally(0x1000, 1, 0, 0, 0));
        whole(&f);
    }

    /// A newer allocation call over part of a span mid-window: the stores
    /// after it go to its own row; the old row keeps what it had.
    #[test]
    fn a_span_retaken_mid_window_is_a_row_of_its_own() {
        let mut s = census();
        s.allocated(P * PAGE, 4 * PAGE, COMBINED);
        land(&mut s, 1, P * PAGE, 100);
        s.allocated(P * PAGE, PAGE, PLAIN);
        land(&mut s, 2, P * PAGE, 30);
        let f = s.landed_figures().expect("on");
        let rows: Vec<(&str, usize, u64)> = f.rows.iter().map(|r| (r.signature.as_str(), r.size, r.tally.bytes)).collect();
        assert_eq!(rows, [(COMBINED_NAME, 4 * PAGE, 100), (PLAIN_NAME, PAGE, 30)], "{f:#?}");
        assert_eq!((f.wc[WC].bytes, f.wc[NOT_WC].bytes), (100, 30));
        whole(&f);
    }

    /// A store in no allocation the census holds is `unknown`, reported,
    /// never dropped; one half in an allocation is split; an `XMemAlloc`
    /// block outside every `XMemVirtualAlloc` call is its own row, its
    /// protection not known.
    #[test]
    fn a_store_outside_every_allocation_is_unknown() {
        let mut s = census();
        s.allocated(P * PAGE, PAGE, COMBINED);
        land(&mut s, 1, (P + 5) * PAGE, 64);
        let f = s.landed_figures().expect("on");
        assert_eq!(f.unknown, tally(64, 1, 1, 1, 0));
        assert_eq!(f.wc[FLAGS_NOT_KNOWN], tally(64, 1, 1, 1, 0));
        assert!(f.rows.is_empty(), "{f:#?}");
        land(&mut s, 2, (P + 1) * PAGE - 32, 64);
        let f = s.landed_figures().expect("on");
        assert_eq!(f.unknown, tally(96, 2, 2, 2, 0));
        assert_eq!(f.wc[WC], tally(32, 1, 1, 1, 0));
        whole(&f);
        s.allocated((P + 10) * PAGE, PAGE, TitleAllocation::General { attributes: 3 << 29 });
        land(&mut s, 3, (P + 10) * PAGE, 8);
        let f = s.landed_figures().expect("on");
        let block = f.rows.iter().find(|r| r.layer == 1).expect("the XMemAlloc block's row");
        assert_eq!((block.wc, block.tally), (FLAGS_NOT_KNOWN as u8, tally(8, 1, 1, 1, 0)));
        let lines = f.lines("test");
        assert_eq!(lines.len(), 3 + 2 + 1, "head, classes, aliases, two rows, the rest: {lines:#?}");
        whole(&f);
    }

    /// The counts start again each window; the finished one is kept for the
    /// stats file and its lines wait for the present.
    #[test]
    fn the_window_starts_again_each_600_frames_and_the_last_is_kept() {
        let mut s = census();
        s.allocated(P * PAGE, 4 * PAGE, COMBINED);
        land(&mut s, 10, P * PAGE, 100);
        land(&mut s, 599, P * PAGE, 100);
        let f = s.landed_figures().expect("on");
        assert_eq!((f.all, f.from), (tally(200, 2, 1, 2, 0), 10), "one page, two frames");
        assert!(s.landed_previous().is_none() && s.landed_pending().is_empty(), "none finished yet");
        land(&mut s, 600, (P + 1) * PAGE, 50);
        let f = s.landed_figures().expect("on");
        assert_eq!((f.all, f.from), (tally(50, 1, 1, 1, 0), 600), "the new window holds only its own");
        assert_eq!(f.asked, Asked { bytes: 50, calls: 1 }, "asked, in the same window as stored");
        let previous = s.landed_previous().expect("the finished window");
        assert!(previous.contains("\"from\":10,\"to\":600,\"frames\":590"), "{previous}");
        let lines = s.landed_pending();
        assert!(lines[0].starts_with("landed classes (every 600 frames, frames 10-600, 590 frames"), "{lines:#?}");
        assert!(s.landed_pending().is_empty(), "taken once");
        // A present in a later window finishes this one with nothing landed after.
        let lines = s.landed_tick(1300);
        assert!(lines[0].starts_with("landed classes (every 600 frames, frames 600-1200, 600 frames"), "{lines:#?}");
        assert_eq!(s.landed_figures().expect("on").all, Tally::default());
        assert!(s.landed_previous().expect("kept").contains("\"from\":600,"));
        assert!(s.landed_tick(1301).is_empty(), "once a window");
        // At exit: the window so far.
        let lines = s.landed_exit_lines(1500);
        assert!(lines[0].starts_with("landed classes (at exit, frames 1200-1500, 300 frames"), "{lines:#?}");
    }

    /// The table holds [`ROWS`] allocations a window; stores into more go
    /// to `other`, counted, and the splits stay whole.
    #[test]
    fn past_the_table_s_bound_stores_go_to_other() {
        let mut s = census();
        let n = ROWS + 3;
        for k in 0..n {
            s.allocated((P + 2 * k) * PAGE, PAGE, COMBINED);
        }
        for k in 0..n {
            land(&mut s, 1, (P + 2 * k) * PAGE, 10);
        }
        let f = s.landed_figures().expect("on");
        assert_eq!(f.rows.len(), ROWS);
        assert_eq!(f.other, tally(30, 3, 3, 3, 0));
        let n64 = n as u64;
        assert_eq!(f.wc[WC], tally(10 * n64, n64, n64, n64, 0), "the protection is known past the table");
        whole(&f);
        let lines = f.lines("test");
        assert_eq!(lines.len(), 3 + TOP + 1, "{lines:#?}");
        assert!(
            lines.last().expect("the rest").contains(
                "past the table's 256 allocations: 0.00 MB a frame (1.2% of the bytes stored, 3.0 pieces of 64 KiB a frame, 3 distinct pages"
            ),
            "{lines:#?}"
        );
        // A new window has room again.
        land(&mut s, 600, (P + 2 * (n - 1)) * PAGE, 10);
        assert_eq!(s.landed_figures().expect("on").rows.len(), 1);
    }

    /// The pages of a window's landings the census saw the processor write
    /// in the same window, before or after the landing, once a page; not
    /// the landing's own store, nor a write in another window.
    #[test]
    fn landed_pages_the_processor_writes_in_the_window_are_counted() {
        let mut s = census();
        s.allocated(P * PAGE, 8 * PAGE, COMBINED);
        s.looked(0, P, P + 2, &[]);
        land(&mut s, 1, P * PAGE, 100);
        // The landing's own write, at the next look: the card's.
        s.looked(2, P, P + 2, &[(P, 2)]);
        assert_eq!(s.landed_figures().expect("on").all.processor, 0);
        // Changed again with no mark since: the processor's (P + 1 never landed).
        s.looked(3, P, P + 2, &[(P, 3), (P + 1, 3)]);
        s.looked(4, P, P + 2, &[(P, 4)]);
        let f = s.landed_figures().expect("on");
        assert_eq!((f.all.processor, f.wc[WC].processor, f.classes[CARD_ONLY].processor), (1, 1, 1), "{f:#?}");
        assert_eq!(f.rows[0].tally.processor, 1);
        // Written by the processor first, landed after: counted at the landing.
        s.looked(5, P + 2, P + 3, &[(P + 2, 5)]);
        land(&mut s, 6, (P + 2) * PAGE, 8);
        assert_eq!(s.landed_figures().expect("on").all.processor, 2);
        // The landing after the processor wrote P: "both write" now.
        land(&mut s, 7, P * PAGE, 4);
        assert_eq!(s.landed_figures().expect("on").classes[BOTH_WRITE].bytes, 4);
        // A write in the window before does not count in this one.
        s.looked(10, P + 3, P + 4, &[(P + 3, 10)]);
        land(&mut s, 600, (P + 3) * PAGE, 10);
        assert_eq!(s.landed_figures().expect("on").all, tally(10, 1, 1, 1, 0));
    }

    /// The fetch's cost follows the 64 KiB pieces landed a frame: they are
    /// given a frame by protection beside the bytes (a page once a frame,
    /// however many stores), not only as distinct pages over the window.
    #[test]
    fn pieces_and_bytes_a_frame_are_given_by_protection() {
        let mut s = census();
        s.allocated(P * PAGE, 8 * PAGE, COMBINED);
        s.allocated((P + 8) * PAGE, 8 * PAGE, PLAIN);
        for f in 0..600 {
            land(&mut s, f, P * PAGE, 60_000);
            land(&mut s, f, (P + 1) * PAGE, 60_000);
            land(&mut s, f, P * PAGE + 60_000, 10);
            if f % 2 == 0 {
                land(&mut s, f, (P + 8) * PAGE, 40_000);
            }
        }
        land(&mut s, 600, P * PAGE, 1);
        let json = s.landed_previous().expect("finished");
        for key in [
            "\"all\":{\"mb_a_frame\":0.140,\"stores_a_frame\":3.5,\"pieces_a_frame\":2.5,\"pages_in_window\":3,",
            "\"write_combined\":{\"mb_a_frame\":0.120,\"stores_a_frame\":3.0,\"pieces_a_frame\":2.0,\"pages_in_window\":2,",
            "\"not_write_combined\":{\"mb_a_frame\":0.020,\"stores_a_frame\":0.5,\"pieces_a_frame\":0.5,\"pages_in_window\":1,",
        ] {
            assert!(json.contains(key), "{key} missing: {json}");
        }
        let lines = s.landed_pending();
        for part in [
            "write-combined 0.12 MB a frame (85.7% of the bytes stored, 2.0 pieces of 64 KiB a frame, 2 distinct pages",
            "not write-combined 0.02 MB a frame (14.3% of the bytes stored, 0.5 pieces of 64 KiB a frame, 1 distinct pages",
        ] {
            assert!(lines[0].contains(part), "{part} missing: {lines:#?}");
        }
        assert_eq!(s.landed_figures().expect("on").all.pieces, 1);
    }

    /// The bytes the merges asked to store are given beside the bytes the
    /// kernel wrote (a refused stretch is retried, then dropped).
    #[test]
    fn bytes_asked_are_given_beside_bytes_stored() {
        let mut s = census();
        s.allocated(P * PAGE, 64 * PAGE, COMBINED);
        for f in 0..600 {
            s.landed_asked(f, 3_000_000);
            s.card(f, P * PAGE, 2_000_000);
            s.landed(f, P * PAGE, 2_000_000);
        }
        let lines = s.landed_tick(600);
        assert!(
            lines[0].contains("asked to store 3.00 MB a frame in 1.0 calls a frame")
                && lines[0].contains("stored 2.00 MB a frame in 1.0 stores a frame"),
            "{lines:#?}"
        );
        let json = s.landed_previous().expect("finished");
        assert!(json.contains("\"asked\":{\"mb_a_frame\":3.000,\"calls_a_frame\":1.0},"), "{json}");
    }

    /// A page the title's views show at two addresses: a store through one
    /// is counted by the pair of protections, the one stored through and
    /// the other address's, so "write-combined" can be read with what of
    /// it is also reachable through memory that is not. A view taken away
    /// is no alias after; private memory never is one.
    #[test]
    fn a_landed_page_shown_at_another_address_is_counted_by_the_pair_of_protections() {
        let mut s = census();
        let (a, b, c, d) = (P * PAGE, (P + 100) * PAGE, (P + 200) * PAGE, (P + 300) * PAGE);
        s.allocated(a, 4 * PAGE, COMBINED);
        s.allocated(b, 4 * PAGE, PLAIN);
        s.allocated(c, 4 * PAGE, COMBINED);
        s.allocated(d, 4 * PAGE, COMBINED);
        s.section(0x10, 0x380);
        // Section pages 0 and 1 at a; page 0 also at b, page 1 also at c + PAGE.
        s.mapped(a, 2 * PAGE, 0x10, 0);
        s.mapped(b, PAGE, 0x10, 0);
        s.mapped(c + PAGE, PAGE, 0x10, PAGE as u64);
        land(&mut s, 1, a + 100, 1000);
        land(&mut s, 1, a + PAGE, 10);
        land(&mut s, 2, b + 200, 5);
        land(&mut s, 2, d, 64);
        let f = s.landed_figures().expect("on");
        let alias = |bytes, pieces, pages| Alias { bytes, pieces, pages };
        assert_eq!(f.aliases[WC][NOT_WC], alias(1000, 1, 1), "stored write-combined, also shown plain: {f:#?}");
        assert_eq!(f.aliases[WC][WC], alias(10, 1, 1));
        assert_eq!(f.aliases[NOT_WC][WC], alias(5, 1, 0), "the page was counted at its first landing, through a");
        let all: u64 = f.aliases.iter().flatten().map(|x| x.bytes).sum();
        assert_eq!(all, 1015, "the private page is no alias: {f:#?}");
        // The view at b taken away: page 0 is shown once now.
        s.remapped(b, PAGE);
        land(&mut s, 3, a + 300, 7);
        let f = s.landed_figures().expect("on");
        assert_eq!(f.aliases[WC][NOT_WC], alias(1000, 1, 1));
        let lines = f.lines("test");
        assert!(
            lines[2].starts_with("landed classes, pages also shown at another address")
                && lines[2].contains("stored through write-combined, also shown not write-combined:"),
            "{lines:#?}"
        );
        let json = f.json();
        assert!(json.contains("\"aliases\":[{\"stored\":\"write-combined\",\"shown\":\"write-combined\","), "{json}");
        assert!(json.contains("{\"stored\":\"write-combined\",\"shown\":\"not write-combined\",\"mb_a_frame\":"), "{json}");
    }

    /// `with` reads the frame before it takes the census's lock, so a store
    /// that read frame 599 can take the lock after one of frame 600 finished
    /// the window: it counts in the new window, and the finished one stays
    /// as it was reported.
    #[test]
    fn a_store_that_read_its_frame_before_a_window_ended_counts_in_the_next() {
        let mut s = census();
        s.allocated(P * PAGE, 8 * PAGE, COMBINED);
        land(&mut s, 599, P * PAGE, 10);
        land(&mut s, 600, (P + 1) * PAGE, 20);
        let finished = s.landed_previous().expect("finished");
        land(&mut s, 599, (P + 2) * PAGE, 30);
        assert_eq!(s.landed_previous().as_deref(), Some(finished.as_str()), "the finished window is not changed");
        let f = s.landed_figures().expect("on");
        assert_eq!((f.from, f.all.bytes, f.all.stores, f.asked.bytes), (600, 50, 2, 50), "{f:#?}");
        let lines = s.landed_pending();
        assert!(lines[0].contains("frames 599-600, 1 frames") && lines[0].contains("stored 0.00 MB a frame in 1.0 stores"), "{lines:#?}");
    }

    /// The lines and the stats object: the first line answers whether the
    /// bytes are write-combined; neither prints an address.
    #[test]
    fn the_report_says_the_write_combined_share_and_no_address() {
        let mut s = census();
        let base = 0x1234_5678_0000usize;
        s.allocated(base, 32 * PAGE, COMBINED);
        s.allocated(base + 32 * PAGE, 32 * PAGE, PLAIN);
        for f in 0..600 {
            land(&mut s, f, base + 0x100, 3 * 1_000_000 / 2);
            land(&mut s, f, base + 32 * PAGE + 0x20, 500_000);
        }
        land(&mut s, 600, base, 1);
        let lines = s.landed_pending();
        for part in [
            "asked to store 2.00 MB a frame in 2.0 calls a frame (the frame line's MB merged into title memory)",
            "stored 2.00 MB a frame in 2.0 stores a frame, 31.0 pieces of 64 KiB a frame",
            "by the protection of the reservation they were stored through (a share of the bytes stored, \
             not evidence the title never reads them): write-combined 1.50 MB a frame (75.0% of the bytes stored",
            "not write-combined 0.50 MB a frame (25.0% of the bytes stored",
        ] {
            assert!(lines[0].contains(part), "{part} missing: {lines:#?}");
        }
        assert!(lines.iter().any(|l| l.contains(COMBINED_NAME) && l.contains("2.1 MB")), "{lines:#?}");
        let json = s.landed_previous().expect("finished");
        for key in [
            "\"frames\":600",
            "\"asked\":{\"mb_a_frame\":2.000,\"calls_a_frame\":2.0},",
            "\"all\":{\"mb_a_frame\":2.000,\"stores_a_frame\":2.0,\"pieces_a_frame\":31.0,",
            "\"write_combined\":{\"mb_a_frame\":1.500,\"stores_a_frame\":1.0,\"pieces_a_frame\":23.0,\"pages_in_window\":23,",
            "\"aliases\":[]",
            "\"classes\":{\"loaded once\":",
            "\"top\":[{\"layer\":\"XMemVirtualAlloc\",\"signature\":\"kind 0x1000 flags 0x11 protect 0x604\"",
            "\"tabled\":2,\"table_rows\":256,",
            "\"other\":{",
            "\"unknown\":{",
        ] {
            assert!(json.contains(key), "{key} missing: {json}");
        }
        assert_eq!(json.matches('{').count(), json.matches('}').count(), "{json}");
        for text in lines.iter().chain([&json]) {
            for address in [base, base + 0x100, base + 32 * PAGE, base + 32 * PAGE + 0x20] {
                assert!(!text.contains(&format!("{address:x}")) && !text.contains(&address.to_string()), "{text}");
            }
        }
    }

    /// The real landing site (`crate::host::write_mapped`) feeds the table:
    /// a store into a write-combined allocation of the census's counts
    /// there.
    #[test]
    fn a_landing_through_write_mapped_is_counted_by_allocation() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        // SAFETY: a fresh reservation of ours, released below.
        let at = unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(at, 0);
        let census = TestCensus::new();
        census.landed_on();
        crate::memclass::memory_census_allocated(at, PAGE, COMBINED);
        crate::host::write_mapped((at + 100) as *mut u8, b"landed bytes".as_ptr(), 12);
        let f = census.landed().expect("on");
        assert_eq!(f.all, tally(12, 1, 1, 1, 0), "{f:#?}");
        assert_eq!(f.asked, Asked { bytes: 12, calls: 1 });
        assert_eq!(f.wc[WC].bytes, 12);
        assert_eq!(f.rows[0].signature, COMBINED_NAME);
        drop(census);
        // SAFETY: the reservation made above.
        assert_eq!(unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) }, 1);
    }

    /// One merge end to end: a host copy the card changed, merged back by
    /// its default store path, which stores through `write_mapped` into a
    /// write-combined allocation of the census's. What the merge returns
    /// (the frame line's "MB merged into title memory") is what was asked
    /// and what landed, all of it write-combined, in one 64 KiB piece.
    #[test]
    fn a_merge_lands_through_write_mapped_in_its_allocation() {
        use crate::shadow;
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        const LEN: usize = 64 << 10;
        const RESOURCE: usize = 0x7a5e;
        let base: Vec<u8> = (0..LEN).map(|i| (i * 13 + i / 251) as u8).collect();
        let mut card = base.clone();
        for at in [100, 200, 40_000] {
            card[at] ^= 0x5a;
        }
        // SAFETY: a fresh reservation of ours, released below.
        let at = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(at, 0);
        // SAFETY: `LEN` bytes are committed there.
        unsafe { std::ptr::copy_nonoverlapping(base.as_ptr(), at as *mut u8, LEN) };
        shadow::register(RESOURCE, at as u64, LEN as u64);
        let census = TestCensus::new();
        census.landed_on();
        crate::memclass::memory_census_allocated(at, LEN, COMBINED);
        let copy = shadow::find(at as u64, LEN as u64).expect("a copy of the title buffer");
        copy.test_card_write(0, &base);
        copy.writing(at, LEN);
        shadow::take_snapshots();
        copy.test_card_write(0, &card);
        let merged = copy.copy_back(at, LEN) as u64;
        copy.copy_back_done(&[(at, at + LEN)]);
        // SAFETY: still committed.
        let landed = unsafe { std::slice::from_raw_parts(at as *const u8, LEN) }.to_vec();
        assert_eq!(landed, card, "the card's bytes are in title memory");
        let f = census.landed().expect("on");
        assert!(merged > 0, "the merge stored something");
        assert_eq!((f.asked.bytes, f.all.bytes, f.wc[WC].bytes), (merged, merged, merged), "{f:#?}");
        // One page; a test census's frame advances with each event, so each
        // store is a piece of its own frame.
        assert_eq!((f.all.pages, f.all.pieces), (1, f.all.stores), "{f:#?}");
        assert_eq!(f.rows[0].signature, COMBINED_NAME);
        drop(copy);
        drop(census);
        shadow::unregister(RESOURCE, at as u64);
        // SAFETY: the reservation made above.
        assert_eq!(unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) }, 1);
    }

    /// Off: the landing site's store makes no table and counts nothing;
    /// the present's tick and the stats object do not touch the census.
    #[test]
    fn off_nothing_is_recorded() {
        if switch() {
            return; // a run with the switch on is not this test
        }
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        // SAFETY: a fresh reservation of ours, released below.
        let at = unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(at, 0);
        let census = TestCensus::new();
        crate::memclass::memory_census_allocated(at, PAGE, COMBINED);
        let events = census.events();
        crate::host::write_mapped((at + 100) as *mut u8, b"landed bytes".as_ptr(), 12);
        assert_eq!(census.events(), events + 1, "the card's mark only: the ask takes no lock");
        assert!(census.landed().is_none(), "no table made, nothing counted");
        let events = census.events();
        crate::memclass::landed_presented();
        assert_eq!(census.events(), events, "the present's tick returns before the census's lock");
        assert_eq!(crate::memclass::landed_object(), None, "no object in the stats file");
        assert_eq!(census.events(), events);
        // A state as the process makes it, with the switch off: no table.
        assert!(State::new().landed.is_none());
        drop(census);
        // SAFETY: the reservation made above.
        assert_eq!(unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) }, 1);
    }
}
