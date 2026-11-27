//! What title memory is for, as a native PC port would hold it (D-076: the
//! card has memory of its own, and the title's memory is not pretended to
//! be one with it). Two parts:
//!
//! - At start-up, what the card offers ([`log_card_memory`], every build):
//!   whether its memory is the PC's (`UMA`, `CacheCoherentUMA`), and whether
//!   the processor can write card memory directly (GPU upload heaps,
//!   `D3D12_FEATURE_D3D12_OPTIONS16`'s `GPUUploadHeapSupported`: card memory
//!   the processor maps through resizable BAR). Logged once.
//! - With diagnostics and `VICTORY_MEMORY_CENSUS=1`, the census (below).
//! - With diagnostics and `VICTORY_LANDED_CLASSES=1`, the copy-back's
//!   landed bytes by the title allocation they land in (`census::landings`),
//!   which turns the census on: it is the census's lookup and classes.
//!
//! # The census
//!
//! Watches, per 64 KiB page of title memory ([`crate::watch::PAGE`]) as
//! physical backing (`census::Backing`: a private page by its address, a
//! page of the title's page pool by its section and index, so a remap
//! starts the address on the new page's history and every view of one
//! physical page shares one), every range the runtime compares: each look
//! of the write watch or the guards
//! ([`crate::watch::writes`]: host copies' refreshes, their merges under
//! shader writes, textures), and each range copied whole because neither
//! sees it. For each page, over the run: the frames it was looked at and
//! found changed, who changed it, and the allocation the title made it in.
//!
//! **Who wrote it.** A change found at a look belongs to the writer marked
//! on the page since its last look: a storage read that completed there
//! (`ReadFile`'s end, `fread`'s copy, a DirectStorage read to an address or
//! to physical pages: [`storage`]) first, else the card (what shaders wrote,
//! merged back, and the title's copy commands, which the runtime runs on
//! the processor: [`card`]), else the title's processor. Shader-written
//! ranges declared for a submission are counted too ([`declared`]): the
//! hash and compare passes look through each, changed or not.
//!
//! **Classes** (per page; [`Page::class`]):
//! - *card only*: the card wrote it, and the title's processor never wrote
//!   it after a card write. Reads are not watched (that needs read traps,
//!   route plan step 8), so *read back* (card-written and read by the
//!   processor) is inside this class and is reported as not watched.
//! - *changes every frame*: changed at a look (by storage or the
//!   processor) in at least [`EVERY_SHARE`]'s share of the frames it was
//!   looked at, and in [`EVERY_MIN`] frames or more.
//! - *loaded once*: changed in at most one frame, or only ever by storage
//!   reads (each fill an upload at the read's end), split by its first
//!   writer: storage, the processor, or never written.
//! - *unknown*: the rest: changed now and then by the processor; written by
//!   both the card and the processor; or copied whole (not watched).
//!
//! **What it cannot tell.** A processor write between a merge and the next
//! look counts as the card's. Values the runtime writes for the card besides
//! merges and copies (completion values) count as the processor's. The
//! write watch answers per address: a write through a view the census
//! hears of only at a look through that view (the guards' view of it).
//! A physical page's section comes from the runtime's own mappings
//! (`crate::inplace::title_memory_view`); the card's page-mapping copies
//! go through those too. Allocations are the title's calls as made
//! (newest wins where they overlap; frees are not followed), `XMemAlloc`
//! blocks under 64 KiB are not kept. The virtual calls' tables take a
//! view page by the address it was last seen at.
//!
//! **Output**: lines starting "memory census" every 30 s and at the
//! title's exit, cumulative over the run: bytes, ranges (runs of adjacent
//! backing pages of one class) and pages looked at per frame, by class;
//! the bytes remapped (a backing that left an address) and of those no
//! address shows now; then the
//! same by allocation, for each of the title's allocation calls, with the
//! flags it passed. The census's own time is in the first line. Then five
//! byte counts a frame since the last report, kept apart
//! (`census::Count`): bytes looked at as asked, the kernel's write-watch
//! query spans, bytes compared (on the processor and on the card), bytes
//! sent up to copies in the card's memory and bytes fetched back. Each is
//! counted where the frame lines count the same bytes, split exactly over
//! the backing pages it touches and given by class at the report, with
//! bytes on no backing looked at as "not looked at": its parts sum to its
//! whole. (The card's "hash" stamp still times the sends and the hashing
//! together; their bytes are apart here.)

use windows::Win32::Graphics::Direct3D12::{
    D3D12_FEATURE_ARCHITECTURE1, D3D12_FEATURE_D3D12_OPTIONS16, D3D12_FEATURE_DATA_ARCHITECTURE1,
    D3D12_FEATURE_DATA_D3D12_OPTIONS16, ID3D12Device,
};

/// What the card answers of its memory: `(UMA, cache-coherent UMA)`, and
/// whether it has GPU upload heaps; `None` where the query was refused (a
/// runtime too old to know it).
pub(crate) fn card_memory(device: &ID3D12Device) -> (Option<(bool, bool)>, Option<bool>) {
    let mut arch = D3D12_FEATURE_DATA_ARCHITECTURE1::default();
    // SAFETY: each query writes its own structure of the given size.
    let arch = unsafe {
        device.CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, (&raw mut arch).cast(), size_of_val(&arch) as u32)
    }
    .is_ok()
    .then(|| (arch.UMA.as_bool(), arch.CacheCoherentUMA.as_bool()));
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    // SAFETY: as above.
    let upload = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS16,
            (&raw mut options).cast(),
            size_of_val(&options) as u32,
        )
    }
    .is_ok()
    .then(|| options.GPUUploadHeapSupported.as_bool());
    (arch, upload)
}

/// The start-up line of [`card_memory`]'s answer.
pub(crate) fn card_memory_line(arch: Option<(bool, bool)>, upload: Option<bool>) -> String {
    let arch = match arch {
        Some((false, _)) => "the card's memory is its own (not UMA)".to_string(),
        Some((true, coherent)) => format!("the card's memory is the PC's (UMA, cache-coherent {coherent})"),
        None => "the card did not say whether its memory is the PC's (architecture query refused)".to_string(),
    };
    let upload = match upload {
        Some(true) => "GPU upload heaps supported (the processor writes card memory directly: on a card of its own, through resizable BAR)",
        Some(false) => "GPU upload heaps not supported (no processor-writable card memory: resizable BAR off, or the driver lacks it)",
        None => "GPU upload heaps unknown (OPTIONS16 query refused: the D3D12 runtime predates it)",
    };
    format!("card memory: {arch}; {upload}")
}

/// Logs [`card_memory_line`] for the host device, once (at its making).
pub(crate) fn log_card_memory(device: &ID3D12Device) {
    let (arch, upload) = card_memory(device);
    tracing::info!("{}", card_memory_line(arch, upload));
}

#[cfg(feature = "diagnostics")]
pub use census::TitleAllocation;
#[cfg(feature = "diagnostics")]
pub(crate) use census::{Count, card, counted, declared, looked, mapped, on, queried, remapped, storage, unwatched};
#[cfg(feature = "diagnostics")]
pub(crate) use census::{landed, landed_object, landed_presented, landing_asked};

/// The live stats file's `landed_classes` object (`VICTORY_LANDED_CLASSES=1`);
/// none without diagnostics.
#[cfg(not(feature = "diagnostics"))]
pub(crate) fn landed_object() -> Option<String> {
    None
}
#[cfg(all(test, feature = "diagnostics"))]
pub(crate) use census::TestCensus;

/// The title made `[at, at + len)` with `what` (`victory_hle`'s xmem): for
/// the census's table by allocation.
#[cfg(feature = "diagnostics")]
pub fn memory_census_allocated(at: usize, len: usize, what: TitleAllocation) {
    census::allocated(at, len, what);
}

/// The runtime made physical-page `section` for the title's
/// `XMemAllocatePhysicalPages` with `flags`.
#[cfg(feature = "diagnostics")]
pub fn memory_census_physical(section: usize, flags: u64) {
    census::section(section, flags);
}

/// Storage wrote `[at, at + len)` in a way the GPU side is not told of
/// otherwise (`fread`'s copy from beside).
#[cfg(feature = "diagnostics")]
pub fn memory_census_storage(at: usize, len: usize) {
    census::storage(at, len);
}

/// The title is ending the process: the census's last lines.
#[cfg(feature = "diagnostics")]
pub fn memory_census_exit() {
    census::log("at exit");
    census::landed_exit();
}

#[cfg(feature = "diagnostics")]
mod census {
    use crate::watch::PAGE;
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    /// The copy-back's landed bytes by allocation (`VICTORY_LANDED_CLASSES`).
    mod landings;
    pub(crate) use landings::{landed, landed_exit, landed_object, landed_presented, landing_asked};

    /// Changed (by storage or the processor) in at least this share of the
    /// frames a page was looked at: one in four, so constants in a ring of
    /// up to four frames count.
    pub(crate) const EVERY_SHARE: u32 = 4;
    /// ... and in at least this many frames.
    pub(crate) const EVERY_MIN: u32 = 3;

    /// Writers marked on a page since its last look.
    const STORAGE: u8 = 1;
    const CARD: u8 = 2;

    /// What a 64 KiB page of title memory is, physically: the census keeps
    /// its history by this, not by the address that shows it, so a remap
    /// starts the address on another page's history and the views of one
    /// physical page share one.
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
    pub(crate) enum Backing {
        /// Private memory (the write-watched reservations): its own page,
        /// by address page, and how many times what the address shows
        /// was replaced before.
        Private(usize, u32),
        /// A page of a physical-page section: the section's census number
        /// (a new one for each section made, so a handle used again is
        /// another section) and the page's index in it.
        View(u32, u64),
    }

    impl Backing {
        /// The backing page after this one, as runs of one class join them.
        fn next(self) -> Backing {
            match self {
                Backing::Private(page, generation) => Backing::Private(page.wrapping_add(1), generation),
                Backing::View(section, index) => Backing::View(section, index.wrapping_add(1)),
            }
        }
    }

    /// A page's census. Frame fields hold the frame plus one (0: never).
    #[derive(Clone, Copy, Default, Debug)]
    pub(crate) struct Page {
        /// The address page it was last seen at: the allocation tables of
        /// the title's virtual calls are by address.
        at: usize,
        /// Looks, and the frames with one.
        looks: u32,
        looked: u32,
        last_look: u64,
        /// The write watch's asking last seen for the page.
        seen: u64,
        /// Writers marked since the last look ([`STORAGE`], [`CARD`]).
        since: u8,
        /// Copied whole: neither the write watch nor the guards see it.
        unwatched: bool,
        /// Frames a look found it changed, by storage, processor, card.
        fills: [u32; 3],
        last_change: u64,
        /// The first change's writer that was not the card: 0 none, 1
        /// storage, 2 the processor.
        first: u8,
        /// Frames the card wrote it, storage wrote it, shaders declared it.
        card: u32,
        last_card: u64,
        storage: u32,
        last_storage: u64,
        declared: u32,
        last_declared: u64,
        /// The processor changed it after a card write.
        cpu_after_card: bool,
        /// Times it stopped being shown at an address (unmapped or mapped
        /// over).
        remaps: u32,
        /// Since the last report, its bytes of each [`Count`].
        counts: [u64; COUNTS],
        /// Its landings' marks (`VICTORY_LANDED_CLASSES`; untouched off).
        landed: landings::Mark,
    }

    /// The byte counts the census keeps apart, each by class and in all.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub(crate) enum Count {
        /// Bytes the runtime asked about, as asked (a look of the write
        /// watch or the guards: `crate::watch::writes`; the copy refresh's
        /// share of it is the frame line's "MB checked").
        Looked,
        /// Bytes of the kernel's write-watch askings, as asked of it
        /// (joined ranges with the gaps between them, and those it turned
        /// down: view memory, two reservations).
        Queried,
        /// Bytes compared on the processor: copies with their snapshots
        /// after shaders ran (the frame line's "MB compared back"), title
        /// memory with a copy under shader writes, and what came back from
        /// the card with the copy's side.
        ComparedCpu,
        /// Bytes the card hashed or compared for copies in its memory: the
        /// pieces hashed after ranges were sent up, and the declared ranges
        /// compared after the lists.
        ComparedCard,
        /// Bytes sent up to copies in the card's memory (the card line's
        /// "MB sent up").
        Sent,
        /// Bytes fetched back from them (the card line's fetched pieces).
        Fetched,
    }

    pub(crate) const COUNTS: usize = 6;

    /// Where a [`Count`]'s bytes are reported: the four classes, then
    /// "not looked at" (no look of the census's at its backing, or no
    /// backing the census holds).
    pub(crate) const NOT_LOOKED: usize = 4;

    /// A page's class ([`Page::class`]).
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub(crate) enum Class {
        LoadedOnce,
        EveryFrame,
        CardOnly,
        Unknown,
    }

    const CLASSES: [Class; 4] = [Class::LoadedOnce, Class::EveryFrame, Class::CardOnly, Class::Unknown];

    impl Class {
        fn name(self) -> &'static str {
            match self {
                Class::LoadedOnce => "loaded once",
                Class::EveryFrame => "changes every frame",
                Class::CardOnly => "card only",
                Class::Unknown => "unknown",
            }
        }
    }

    impl Page {
        pub(crate) fn class(&self) -> Class {
            if self.unwatched {
                return Class::Unknown;
            }
            if self.card > 0 {
                return if self.cpu_after_card { Class::Unknown } else { Class::CardOnly };
            }
            let fills = self.fills[0] + self.fills[1];
            if fills >= EVERY_MIN && fills * EVERY_SHARE >= self.looked {
                return Class::EveryFrame;
            }
            if fills <= 1 || self.fills[1] == 0 {
                return Class::LoadedOnce;
            }
            Class::Unknown
        }

        /// Within its class: loaded once by storage 0, the processor 1,
        /// never written 2; unknown as changed sometimes 0, card and
        /// processor both 1, copied whole 2.
        fn detail(&self) -> usize {
            match self.class() {
                Class::LoadedOnce => match self.first {
                    1 => 0,
                    2 => 1,
                    _ => 2,
                },
                Class::Unknown if self.unwatched => 2,
                Class::Unknown if self.card > 0 => 1,
                _ => 0,
            }
        }
    }

    /// The title's allocation calls, one layer per call ([`LAYERS`]):
    /// address spans, newest wins, each with its flags' signature.
    #[derive(Default)]
    struct Layer {
        spans: BTreeMap<usize, (usize, u32)>,
    }

    impl Layer {
        fn insert(&mut self, at: usize, end: usize, signature: u32) {
            if at >= end {
                return;
            }
            if let Some((&s, &(e, id))) = self.spans.range(..at).next_back()
                && e > at
            {
                self.spans.insert(s, (at, id));
                if e > end {
                    self.spans.insert(end, (e, id));
                }
            }
            let inside: Vec<usize> = self.spans.range(at..end).map(|(&s, _)| s).collect();
            for s in inside {
                if let Some((e, id)) = self.spans.remove(&s)
                    && e > end
                {
                    self.spans.insert(end, (e, id));
                }
            }
            self.spans.insert(at, (end, signature));
        }

        /// The signature of the span holding `address`, if one does.
        fn at(&self, address: usize) -> Option<u32> {
            self.span(address).map(|(_, _, id)| id)
        }

        /// The span holding `address`, if one does: its start, end and
        /// signature.
        fn span(&self, address: usize) -> Option<(usize, usize, u32)> {
            self.spans
                .range(..=address)
                .next_back()
                .filter(|(_, (end, _))| *end > address)
                .map(|(&start, &(end, id))| (start, end, id))
        }

        /// The start of the first span after `address` and before `end`
        /// (`address < end`), if one starts there.
        fn next(&self, address: usize, end: usize) -> Option<usize> {
            self.spans.range(address + 1..end).next().map(|(&start, _)| start)
        }
    }

    /// What the title said when it made memory (`victory_hle`'s xmem).
    #[derive(Clone, Copy, Debug)]
    pub enum TitleAllocation {
        /// `XMemVirtualAlloc`'s allocation type, XMem flags and protection,
        /// as the title passed them.
        Virtual { kind: u32, flags: u64, protect: u32 },
        /// `XMemAlloc`'s attributes.
        General { attributes: u64 },
    }

    /// The layers, in the order the lines give them. The first two are the
    /// title's virtual calls, by address; the third is by backing (the
    /// section a view page belongs to).
    const LAYERS: [&str; 3] = ["XMemVirtualAlloc", "XMemAlloc", "XMemAllocatePhysicalPages"];

    #[derive(Default)]
    pub(crate) struct State {
        /// By physical backing.
        pages: BTreeMap<Backing, Page>,
        /// Address pages that show a view now, and what they show; the
        /// others are private memory, their own backing.
        shown: BTreeMap<usize, Backing>,
        /// Private address pages whose memory was replaced, and how many
        /// times ([`Backing::Private`]'s second number).
        replaced: HashMap<usize, u32>,
        /// The title's virtual calls by address ([`LAYERS`]' first two).
        layers: [Layer; 2],
        signatures: Vec<String>,
        ids: HashMap<String, u32>,
        /// Physical-page sections by handle: the census number of the one
        /// made last with it.
        sections: HashMap<usize, u32>,
        /// By section number: the signature of its call, if the census saw
        /// it made.
        section_signatures: Vec<Option<u32>>,
        /// The first frame seen, and the frame of the last report.
        start: Option<u64>,
        reported: Option<u64>,
        /// Since the last report, by [`Count`]: bytes in all, and bytes on
        /// no page the census holds.
        totals: [u64; COUNTS],
        outside: [u64; COUNTS],
        /// The landed bytes by allocation, with `VICTORY_LANDED_CLASSES=1`
        /// (made with the state, never otherwise: [`State::new`]).
        landed: Option<Box<landings::Landed>>,
    }

    /// The census's figures for one moment ([`State::report`]).
    #[derive(Default, Debug)]
    pub(crate) struct Report {
        pub(crate) frames: u64,
        /// By class: bytes, ranges, page looks.
        pub(crate) bytes: [u64; 4],
        pub(crate) ranges: [u64; 4],
        pub(crate) looks: [u64; 4],
        /// Loaded once by first writer; unknown by why ([`Page::detail`]).
        pub(crate) loaded: [u64; 3],
        pub(crate) unknown: [u64; 3],
        /// Declared by shaders; of those, never changed by the card.
        pub(crate) declared: u64,
        pub(crate) declared_unchanged: u64,
        /// Written by storage and never looked at; remapped at least once.
        pub(crate) storage_unlooked: u64,
        pub(crate) remapped: u64,
        /// Looked at, of a section's page no address shows now.
        pub(crate) unshown: u64,
        /// By layer: signature, its bytes by class; largest first.
        pub(crate) by_allocation: [Vec<(String, [u64; 4])>; 3],
        /// Since the last report: its frames, and by [`Count`] the bytes in
        /// all and by class (then [`NOT_LOOKED`]); the parts sum to the
        /// whole.
        pub(crate) interval: u64,
        pub(crate) totals: [u64; COUNTS],
        pub(crate) counts: [[u64; 5]; COUNTS],
    }

    impl State {
        /// What address page `page` shows.
        fn backing(&self, page: usize) -> Backing {
            if let Some(&shown) = self.shown.get(&page) {
                return shown;
            }
            let generation = if self.replaced.is_empty() { 0 } else { self.replaced.get(&page).copied().unwrap_or(0) };
            Backing::Private(page, generation)
        }

        /// The census of what address page `page` shows.
        fn page(&mut self, page: usize) -> &mut Page {
            let backing = self.backing(page);
            let p = self.pages.entry(backing).or_default();
            p.at = page;
            p
        }

        fn frame(&mut self, frame: u64) -> u64 {
            self.start.get_or_insert(frame);
            frame + 1
        }

        /// A look at pages `[first, last)`: `written` is the write watch's
        /// answer, each page written since tracking began with the asking
        /// that last found it written, in page order.
        pub(crate) fn looked(&mut self, frame: u64, first: usize, last: usize, written: &[(usize, u64)]) {
            let now = self.frame(frame);
            // The landed table, taken out for the pages' borrows and put
            // back at the end; `None` (no work) with its switch off.
            let mut landed = self.landed.take();
            if let Some(l) = landed.as_mut() {
                l.at(frame, &self.signatures);
            }
            let mut written = written.iter().peekable();
            for page in first..last {
                while written.peek().is_some_and(|w| w.0 < page) {
                    written.next();
                }
                let asked = match written.peek() {
                    Some(&&(p, w)) if p == page => w,
                    _ => 0,
                };
                let p = self.page(page);
                p.looks += 1;
                if p.last_look != now {
                    p.looked += 1;
                    p.last_look = now;
                }
                let changed = asked > p.seen || p.since != 0;
                p.seen = p.seen.max(asked);
                if !changed {
                    continue;
                }
                let who = if p.since & STORAGE != 0 {
                    0
                } else if p.since & CARD != 0 {
                    2
                } else {
                    1
                };
                p.since = 0;
                if who == 1 && p.card > 0 {
                    p.cpu_after_card = true;
                }
                if who == 1
                    && let Some(l) = landed.as_mut()
                {
                    l.processor_wrote(&mut p.landed);
                }
                if p.first == 0 && who != 2 {
                    p.first = who as u8 + 1;
                }
                if p.last_change != now {
                    p.last_change = now;
                    p.fills[who] += 1;
                }
            }
            self.landed = landed;
        }

        /// `[at, at + len)` was copied whole (not watched).
        pub(crate) fn unwatched(&mut self, frame: u64, at: usize, len: usize) {
            let now = self.frame(frame);
            for page in pages(at, len) {
                let p = self.page(page);
                p.unwatched = true;
                p.looks += 1;
                if p.last_look != now {
                    p.looked += 1;
                    p.last_look = now;
                }
            }
        }

        /// Storage wrote `[at, at + len)`.
        pub(crate) fn storage(&mut self, frame: u64, at: usize, len: usize) {
            let now = self.frame(frame);
            for page in pages(at, len) {
                let p = self.page(page);
                p.since |= STORAGE;
                if p.last_storage != now {
                    p.last_storage = now;
                    p.storage += 1;
                }
            }
        }

        /// The card's results (or its copy commands) were written to
        /// `[at, at + len)`.
        pub(crate) fn card(&mut self, frame: u64, at: usize, len: usize) {
            let now = self.frame(frame);
            for page in pages(at, len) {
                let p = self.page(page);
                p.since |= CARD;
                if p.last_card != now {
                    p.last_card = now;
                    p.card += 1;
                }
            }
        }

        /// Shaders of a submission may write `[at, at + len)`.
        pub(crate) fn declared(&mut self, frame: u64, at: usize, len: usize) {
            let now = self.frame(frame);
            for page in pages(at, len) {
                let p = self.page(page);
                if p.last_declared != now {
                    p.last_declared = now;
                    p.declared += 1;
                }
            }
        }

        /// What `[at, at + len)` showed is gone: the backings shown there
        /// are counted as remapped, and the addresses show nothing of the
        /// census's until mapped again ([`State::mapped`]).
        pub(crate) fn remapped(&mut self, at: usize, len: usize) {
            // Only what the census holds there is walked (a range can be
            // all of memory).
            let span = pages(at, len);
            // Private memory looked at there is replaced: what the address
            // shows next is a new page.
            let mut private: Vec<usize> = self
                .pages
                .range(Backing::Private(span.start, 0)..Backing::Private(span.end, 0))
                .filter_map(|(b, _)| match *b {
                    Backing::Private(page, _) => (!self.shown.contains_key(&page)).then_some(page),
                    Backing::View(..) => None,
                })
                .collect();
            private.dedup();
            for page in private {
                let backing = self.backing(page);
                if let Some(p) = self.pages.get_mut(&backing) {
                    p.remaps += 1;
                    *self.replaced.entry(page).or_insert(0) += 1;
                }
            }
            let views: Vec<usize> = self.shown.range(span).map(|(&k, _)| k).collect();
            for page in views {
                let Some(backing) = self.shown.remove(&page) else {
                    continue;
                };
                if let Some(l) = self.landed.as_mut() {
                    l.unshown(page, backing);
                }
                if let Some(p) = self.pages.get_mut(&backing) {
                    p.remaps += 1;
                }
            }
        }

        /// `what` took bytes `[at, end)`: each backing page they touch gets
        /// its share, exactly; bytes on pages the census does not hold are
        /// [`NOT_LOOKED`]'s. Only the census's own entries are walked (a
        /// kernel asking can span a gigabyte).
        pub(crate) fn spread(&mut self, what: Count, at: usize, end: usize) {
            if at >= end {
                return;
            }
            let k = what as usize;
            let all = (end - at) as u64;
            self.totals[k] += all;
            let span = at / PAGE..end.div_ceil(PAGE);
            let share = |page: usize| (end.min((page + 1) * PAGE) - at.max(page * PAGE)) as u64;
            let mut given = 0;
            for (&page, backing) in self.shown.range(span.clone()) {
                if let Some(p) = self.pages.get_mut(backing) {
                    p.counts[k] += share(page);
                    given += share(page);
                }
            }
            let (shown, replaced) = (&self.shown, &self.replaced);
            let private = self.pages.range_mut(Backing::Private(span.start, 0)..Backing::Private(span.end, 0));
            for (backing, p) in private {
                let Backing::Private(page, generation) = *backing else {
                    continue;
                };
                // The page's memory now, not a replaced one's.
                if shown.contains_key(&page) || replaced.get(&page).copied().unwrap_or(0) != generation {
                    continue;
                }
                p.counts[k] += share(page);
                given += share(page);
            }
            self.outside[k] += all - given;
        }

        fn signature(&mut self, text: String) -> u32 {
            if let Some(&id) = self.ids.get(&text) {
                return id;
            }
            let id = self.signatures.len() as u32;
            self.signatures.push(text.clone());
            self.ids.insert(text, id);
            id
        }

        /// The title made `[at, at + len)` with `what`.
        pub(crate) fn allocated(&mut self, at: usize, len: usize, what: TitleAllocation) {
            let (layer, text) = match what {
                TitleAllocation::Virtual { kind, flags, protect } => {
                    (0, format!("kind {kind:#x} flags {flags:#x} protect {protect:#x}"))
                }
                TitleAllocation::General { attributes } => (1, general(attributes)),
            };
            let id = self.signature(text);
            if let (TitleAllocation::Virtual { protect, .. }, Some(l)) = (what, self.landed.as_mut()) {
                l.protection(id, protect);
            }
            self.layers[layer].insert(at, at.saturating_add(len), id);
        }

        /// The census number of the section of handle `section` (one made
        /// now if the census did not see it made).
        fn section_number(&mut self, section: usize) -> u32 {
            if let Some(&n) = self.sections.get(&section) {
                return n;
            }
            let n = self.section_signatures.len() as u32;
            self.section_signatures.push(None);
            self.sections.insert(section, n);
            n
        }

        /// The runtime made physical-page `section` for the title's call
        /// with `flags`: a new section, even if the handle was used before.
        pub(crate) fn section(&mut self, section: usize, flags: u64) {
            let id = self.signature(format!("flags {flags:#x}"));
            let n = self.section_signatures.len() as u32;
            self.section_signatures.push(Some(id));
            self.sections.insert(section, n);
        }

        /// `[at, at + len)` shows the pages of `section` from byte `offset`
        /// on now.
        pub(crate) fn mapped(&mut self, at: usize, len: usize, section: usize, offset: u64) {
            let n = self.section_number(section);
            let first = offset / PAGE as u64;
            for (k, page) in pages(at, len).enumerate() {
                let backing = Backing::View(n, first + k as u64);
                let old = self.shown.insert(page, backing);
                if let Some(l) = self.landed.as_mut() {
                    l.shown(page, old, backing);
                }
            }
        }

        /// The census now; the byte counts since the last report, which
        /// start again.
        pub(crate) fn report(&mut self, frame: u64) -> Report {
            let mut r = Report {
                frames: frame.saturating_sub(self.start.unwrap_or(frame)).max(1),
                interval: frame.saturating_sub(self.reported.or(self.start).unwrap_or(frame)).max(1),
                totals: std::mem::take(&mut self.totals),
                ..Default::default()
            };
            self.reported = Some(frame);
            for (k, outside) in std::mem::take(&mut self.outside).into_iter().enumerate() {
                r.counts[k][NOT_LOOKED] = outside;
            }
            for p in self.pages.values_mut() {
                let bucket = if p.looks == 0 { NOT_LOOKED } else { p.class() as usize };
                for (k, n) in std::mem::take(&mut p.counts).into_iter().enumerate() {
                    r.counts[k][bucket] += n;
                }
            }
            // By layer, by signature (0: in none, else its id plus one).
            let mut tables: [Vec<[u64; 4]>; 3] = std::array::from_fn(|_| vec![[0; 4]; self.signatures.len() + 1]);
            let shown: std::collections::HashSet<Backing> = self.shown.values().copied().collect();
            let mut last: Option<(Backing, Class)> = None;
            let page_bytes = PAGE as u64;
            for (&backing, p) in &self.pages {
                if p.declared > 0 {
                    r.declared += page_bytes;
                    if p.card == 0 {
                        r.declared_unchanged += page_bytes;
                    }
                }
                if p.remaps > 0 {
                    r.remapped += page_bytes;
                }
                if p.looks == 0 {
                    if p.storage > 0 {
                        r.storage_unlooked += page_bytes;
                    }
                    continue;
                }
                let class = p.class();
                let k = class as usize;
                r.bytes[k] += page_bytes;
                r.looks[k] += u64::from(p.looks);
                if last.is_none_or(|(b, c)| b.next() != backing || c != class) {
                    r.ranges[k] += 1;
                }
                last = Some((backing, class));
                match class {
                    Class::LoadedOnce => r.loaded[p.detail()] += page_bytes,
                    Class::Unknown => r.unknown[p.detail()] += page_bytes,
                    _ => {}
                }
                if let Backing::View(..) = backing
                    && !shown.contains(&backing)
                {
                    r.unshown += page_bytes;
                }
                let address = p.at * PAGE;
                let rows = [
                    self.layers[0].at(address),
                    self.layers[1].at(address),
                    match backing {
                        Backing::View(n, _) => self.section_signatures.get(n as usize).copied().flatten(),
                        Backing::Private(..) => None,
                    },
                ];
                for (table, row) in tables.iter_mut().zip(rows) {
                    table[row.map_or(0, |id| id as usize + 1)][k] += page_bytes;
                }
            }
            for (out, table) in r.by_allocation.iter_mut().zip(tables) {
                let mut rows: Vec<(String, [u64; 4])> = table
                    .into_iter()
                    .enumerate()
                    .filter(|(_, b)| b.iter().any(|&n| n > 0))
                    .map(|(row, b)| {
                        let name = match row {
                            0 => "not in one",
                            row => self.signatures[row - 1].as_str(),
                        };
                        (name.to_string(), b)
                    })
                    .collect();
                rows.sort_by(|a, b| b.1.iter().sum::<u64>().cmp(&a.1.iter().sum::<u64>()).then(a.0.cmp(&b.0)));
                *out = rows;
            }
            r
        }
    }

    /// `XMemAlloc`'s attributes without the alignment (bits 24-28, which
    /// differ by call): the memory type (bits 29-32), the 32-bit address
    /// bit (33), and the low 24 bits as passed.
    fn general(attributes: u64) -> String {
        let kind = attributes >> 29 & 0xf;
        let low = attributes & 0xff_ffff;
        let wide = if attributes >> 33 & 1 != 0 { " 32-bit" } else { "" };
        format!("memory type {kind} low bits {low:#x}{wide} (high {:#x})", attributes >> 34)
    }

    /// The pages `[at, at + len)` touches.
    fn pages(at: usize, len: usize) -> std::ops::Range<usize> {
        if len == 0 {
            return 0..0;
        }
        at / PAGE..at.saturating_add(len).div_ceil(PAGE)
    }

    impl Report {
        /// The lines logged ([`log`]).
        pub(crate) fn lines(&self, why: &str, census_ms: f64, rows: usize) -> Vec<String> {
            let mb = |b: u64| b as f64 / (1u64 << 20) as f64;
            let frames = self.frames as f64;
            let class = |k: usize| {
                format!(
                    "{:.0} MB in {} ranges, {:.0} MB a frame looked at",
                    mb(self.bytes[k]),
                    self.ranges[k],
                    mb(self.looks[k] * PAGE as u64) / frames
                )
            };
            let total: u64 = self.bytes.iter().sum();
            let ranges: u64 = self.ranges.iter().sum();
            let mut lines = vec![format!(
                "memory census ({why}, {} frames; the census {census_ms:.2} ms a frame): {:.0} MB compared in {ranges} \
                 ranges; loaded once {} ({:.0} MB filled by storage, {:.0} MB by the title's processor, {:.0} MB never \
                 written); changes every frame {}; card only {} (reads are not watched: read back is in it); read back: \
                 not watched; unknown {} ({:.0} MB changes sometimes, {:.0} MB the card and the processor both write, \
                 {:.0} MB copied whole, not watched); declared for shaders {:.0} MB, {:.0} MB of it never changed by the \
                 card; written by storage and never compared {:.0} MB; remapped {:.0} MB (physical pages that left an \
                 address), {:.0} MB of it looked at and shown at no address now",
                self.frames,
                mb(total),
                class(0),
                mb(self.loaded[0]),
                mb(self.loaded[1]),
                mb(self.loaded[2]),
                class(1),
                class(2),
                class(3),
                mb(self.unknown[0]),
                mb(self.unknown[1]),
                mb(self.unknown[2]),
                mb(self.declared),
                mb(self.declared_unchanged),
                mb(self.storage_unlooked),
                mb(self.remapped),
                mb(self.unshown),
            )];
            for (layer, table) in LAYERS.iter().zip(&self.by_allocation) {
                for (name, b) in table.iter().take(rows) {
                    let parts: Vec<String> = CLASSES
                        .iter()
                        .zip(b)
                        .filter(|&(_, &n)| n > 0)
                        .map(|(c, &n)| format!("{} {:.0} MB", c.name(), mb(n)))
                        .collect();
                    lines.push(format!(
                        "memory census by {layer}: {name}: {:.0} MB: {}",
                        mb(b.iter().sum()),
                        parts.join(", ")
                    ));
                }
            }
            lines.extend(self.count_lines());
            lines
        }

        /// One line for each byte count ([`Count`]) since the last report,
        /// a frame: its whole and its parts by class, which sum to it.
        pub(crate) fn count_lines(&self) -> Vec<String> {
            let frames = self.interval as f64;
            let mb = |b: u64| b as f64 / (1u64 << 20) as f64 / frames;
            let parts = |k: usize| {
                let c = &self.counts[k];
                format!(
                    "loaded once {:.2} + changes every frame {:.2} + card only {:.2} + unknown {:.2} + not looked at {:.2}",
                    mb(c[0]),
                    mb(c[1]),
                    mb(c[2]),
                    mb(c[3]),
                    mb(c[NOT_LOOKED])
                )
            };
            let total = |k: usize| mb(self.totals[k]);
            let (cpu, card) = (Count::ComparedCpu as usize, Count::ComparedCard as usize);
            let compared: [u64; 5] = std::array::from_fn(|b| self.counts[cpu][b] + self.counts[card][b]);
            let compared_parts = format!(
                "loaded once {:.2} + changes every frame {:.2} + card only {:.2} + unknown {:.2} + not looked at {:.2}",
                mb(compared[0]),
                mb(compared[1]),
                mb(compared[2]),
                mb(compared[3]),
                mb(compared[NOT_LOOKED])
            );
            let n = self.interval;
            vec![
                format!(
                    "memory census bytes a frame ({n} frames): looked at, as asked: {:.2} MB = {}",
                    total(Count::Looked as usize),
                    parts(Count::Looked as usize)
                ),
                format!(
                    "memory census bytes a frame ({n} frames): kernel write-watch query spans: {:.2} MB = {}",
                    total(Count::Queried as usize),
                    parts(Count::Queried as usize)
                ),
                format!(
                    "memory census bytes a frame ({n} frames): compared: {:.2} MB (on the processor {:.2}, on the card {:.2}) = {}",
                    total(cpu) + total(card),
                    total(cpu),
                    total(card),
                    compared_parts
                ),
                format!(
                    "memory census bytes a frame ({n} frames): sent up to the card: {:.2} MB = {}",
                    total(Count::Sent as usize),
                    parts(Count::Sent as usize)
                ),
                format!(
                    "memory census bytes a frame ({n} frames): fetched from the card: {:.2} MB = {}",
                    total(Count::Fetched as usize),
                    parts(Count::Fetched as usize)
                ),
            ]
        }
    }

    /// Whether the census runs: diagnostics and `VICTORY_MEMORY_CENSUS=1`,
    /// or `VICTORY_LANDED_CLASSES=1`, which is built on it.
    pub(crate) fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var_os("VICTORY_MEMORY_CENSUS").is_some_and(|v| v == "1") || landings::switch()
        })
    }

    impl State {
        /// The process's census: with the landed table only when its
        /// switch is on.
        fn new() -> State {
            State {
                landed: landings::switch().then(|| Box::new(landings::Landed::new())),
                ..State::default()
            }
        }
    }

    static STATE: Mutex<Option<State>> = Mutex::new(None);
    /// Nanoseconds spent in the census.
    static SPENT: AtomicU64 = AtomicU64::new(0);
    /// Kernel askings not yet in the census ([`queried`]): noted under the
    /// write watch's journal lock, taken in by the next [`with`] or
    /// [`log`] (a leaf lock, held only to push or take).
    static QUERIES: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

    /// [`QUERIES`] into `state`.
    fn take_queries(state: &mut State) {
        let queued = std::mem::take(&mut *QUERIES.lock().unwrap_or_else(|e| e.into_inner()));
        for (at, end) in queued {
            state.spread(Count::Queried, at, end);
        }
    }

    // Tests: a census a fixture drives on its own thread (`TestCensus`), in
    // place of the process's: parallel fixtures share one otherwise, and the
    // switch (`VICTORY_MEMORY_CENSUS`) is not set for tests. The frame
    // advances with each event, as a run's frames would.
    #[cfg(test)]
    thread_local! {
        static TEST: std::cell::RefCell<Option<(State, u64)>> = const { std::cell::RefCell::new(None) };
    }

    /// Tests: a census of this thread's own, from here until it is dropped.
    #[cfg(test)]
    pub(crate) struct TestCensus(());

    #[cfg(test)]
    impl TestCensus {
        pub(crate) fn new() -> Self {
            TEST.with(|t| {
                assert!(t.borrow().is_none(), "one census at a time on a thread");
                *t.borrow_mut() = Some((State::default(), 0));
            });
            Self(())
        }

        /// The frames a look of page `at`'s found it changed, by storage,
        /// the processor, the card ([`Page::fills`]).
        pub(crate) fn fills(&self, at: usize) -> [u32; 3] {
            TEST.with(|t| {
                let mut t = t.borrow_mut();
                let (state, _) = t.as_mut().expect("the census is installed");
                state.page(at / PAGE).fills
            })
        }

        /// Turns this census's landed table on, as `VICTORY_LANDED_CLASSES=1`
        /// does the process's (before the allocations it is to know).
        pub(crate) fn landed_on(&self) {
            TEST.with(|t| {
                let mut t = t.borrow_mut();
                let (state, _) = t.as_mut().expect("the census is installed");
                state.landed = Some(Box::new(landings::Landed::new()));
            });
        }

        /// The landed table's window so far; `None` while it is off.
        pub(crate) fn landed(&self) -> Option<landings::Figures> {
            TEST.with(|t| t.borrow().as_ref().expect("the census is installed").0.landed_figures())
        }

        /// The census's events so far (one a [`with`]: its test frame).
        pub(crate) fn events(&self) -> u64 {
            TEST.with(|t| t.borrow().as_ref().expect("the census is installed").1)
        }
    }

    #[cfg(test)]
    impl Drop for TestCensus {
        fn drop(&mut self) {
            TEST.with(|t| *t.borrow_mut() = None);
        }
    }

    /// Runs `f` on the census (a leaf lock: `f` takes no other), counting
    /// its time; starts the reporter the first time. Tests: on the thread's
    /// own census instead, when it has one ([`TestCensus`]).
    fn with(f: impl FnOnce(&mut State, u64)) {
        #[cfg(test)]
        if TEST.with(|t| t.borrow().is_some()) {
            TEST.with(|t| {
                let mut t = t.borrow_mut();
                let (state, frame) = t.as_mut().expect("checked above");
                *frame += 1;
                f(state, *frame);
            });
            return;
        }
        if !on() {
            return;
        }
        let started = std::time::Instant::now();
        static REPORTER: std::sync::Once = std::sync::Once::new();
        REPORTER.call_once(|| {
            let _ = std::thread::Builder::new().name("victory-census".into()).spawn(|| {
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(30));
                    log("every 30 s");
                }
            });
        });
        let frame = crate::queue::presented();
        let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
        let state = state.get_or_insert_with(State::new);
        take_queries(state);
        f(state, frame);
        SPENT.fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
    }

    /// Logs the census's lines: `why` says when.
    pub(crate) fn log(why: &str) {
        if !on() {
            return;
        }
        let frame = crate::queue::presented();
        let report = match &mut *STATE.lock().unwrap_or_else(|e| e.into_inner()) {
            Some(state) => {
                take_queries(state);
                state.report(frame)
            }
            None => return,
        };
        let ms = SPENT.load(Relaxed) as f64 / 1e6 / report.frames as f64;
        for line in report.lines(why, ms, 8) {
            tracing::info!("{line}");
        }
    }

    /// A look at `[start, end)` ([`State::looked`] of its pages), and its
    /// bytes as asked.
    pub(crate) fn looked(start: usize, end: usize, written: &[(usize, u64)]) {
        with(|s, f| {
            s.looked(f, start / PAGE, end.div_ceil(PAGE), written);
            s.spread(Count::Looked, start, end);
        });
    }

    /// `what` took the `len` bytes at `at` ([`State::spread`]).
    pub(crate) fn counted(what: Count, at: usize, len: usize) {
        with(|s, _| s.spread(what, at, at.saturating_add(len)));
    }

    /// The kernel was asked about the written pages of `[at, end)`; called
    /// under the write watch's journal lock, so only noted here.
    pub(crate) fn queried(at: usize, end: usize) {
        if on() && at < end {
            QUERIES.lock().unwrap_or_else(|e| e.into_inner()).push((at, end));
        }
    }

    pub(crate) fn unwatched(at: usize, len: usize) {
        with(|s, f| s.unwatched(f, at, len));
    }

    pub(crate) fn storage(at: usize, len: usize) {
        with(|s, f| s.storage(f, at, len));
    }

    pub(crate) fn card(at: usize, len: usize) {
        with(|s, f| s.card(f, at, len));
    }

    pub(crate) fn declared(at: usize, len: usize) {
        with(|s, f| s.declared(f, at, len));
    }

    pub(crate) fn remapped(at: usize, len: usize) {
        with(|s, _| s.remapped(at, len));
    }

    pub(crate) fn allocated(at: usize, len: usize, what: TitleAllocation) {
        with(|s, _| s.allocated(at, len, what));
    }

    pub(crate) fn section(section: usize, flags: u64) {
        with(|s, _| s.section(section, flags));
    }

    pub(crate) fn mapped(at: usize, len: usize, section: usize, offset: u64) {
        with(|s, _| s.mapped(at, len, section, offset));
    }
}

#[cfg(all(test, feature = "diagnostics"))]
mod census_tests {
    use super::census::{Class, State, TitleAllocation};
    use crate::watch::PAGE;

    /// A page's class after the events `f` gives a census.
    fn class_of(page: usize, f: impl FnOnce(&mut State)) -> Class {
        let mut s = State::default();
        f(&mut s);
        let r = s.report(100);
        let classes = [Class::LoadedOnce, Class::EveryFrame, Class::CardOnly, Class::Unknown];
        let found: Vec<Class> = classes.into_iter().filter(|&c| r.bytes[c as usize] > 0).collect();
        assert_eq!(found.len(), 1, "one page, one class: {r:?} ({page})");
        found[0]
    }

    const P: usize = 0x1_0000_0000 / PAGE;

    /// Storage fills the page in frame 1; it is looked at in 40 frames and
    /// never changes again.
    #[test]
    fn a_page_storage_filled_and_never_changed_is_loaded_once_by_storage() {
        let mut s = State::default();
        s.storage(1, P * PAGE, PAGE);
        s.looked(1, P, P + 1, &[(P, 1)]);
        for f in 2..40 {
            s.looked(f, P, P + 1, &[(P, 1)]);
        }
        let r = s.report(40);
        assert_eq!(r.bytes[Class::LoadedOnce as usize], PAGE as u64);
        assert_eq!(r.loaded[0], PAGE as u64, "filled by storage: {r:?}");
        // Refilled by storage now and then: still an upload at each read's end.
        for f in [50, 70, 90] {
            s.storage(f, P * PAGE, 100);
            s.looked(f, P, P + 1, &[(P, 1)]);
        }
        for f in 91..200 {
            s.looked(f, P, P + 1, &[(P, 1)]);
        }
        assert_eq!(s.report(200).bytes[Class::LoadedOnce as usize], PAGE as u64);
    }

    /// The processor writes the page before every look.
    #[test]
    fn a_page_the_processor_writes_every_frame_changes_every_frame() {
        let class = class_of(P, |s| {
            for f in 0..30 {
                s.looked(f, P, P + 1, &[(P, f + 1)]);
            }
        });
        assert_eq!(class, Class::EveryFrame);
        // A ring of three frames: each page written every third frame.
        let class = class_of(P, |s| {
            for f in 0..30 {
                s.looked(f, P, P + 1, &[(P, f / 3 * 3 + 1)]);
            }
        });
        assert_eq!(class, Class::EveryFrame);
    }

    /// The card writes the page (a merge, whose own write the next look
    /// sees), and the processor does not write it after.
    #[test]
    fn a_page_only_the_card_writes_is_card_only() {
        let class = class_of(P, |s| {
            s.looked(0, P, P + 1, &[(P, 1)]);
            for f in 1..20 {
                s.declared(f, P * PAGE, PAGE);
                s.card(f, P * PAGE + 64, 128);
                s.looked(f + 1, P, P + 1, &[(P, f + 1)]);
            }
        });
        assert_eq!(class, Class::CardOnly);
    }

    /// Read back is card-written memory the processor reads. Reads are not
    /// watched (no read traps), so such a page is in card only, and the
    /// line says read back is not watched rather than giving a figure.
    #[test]
    fn read_back_is_not_watched_and_stays_in_card_only() {
        let mut s = State::default();
        s.looked(0, P, P + 1, &[(P, 1)]);
        s.card(1, P * PAGE, PAGE);
        // The title reads it here: nothing to see.
        s.looked(2, P, P + 1, &[(P, 2)]);
        let r = s.report(3);
        assert_eq!(r.bytes[Class::CardOnly as usize], PAGE as u64);
        let line = &r.lines("test", 0.0, 4)[0];
        assert!(line.contains("read back: not watched"), "{line}");
        assert!(line.contains("reads are not watched: read back is in it"), "{line}");
    }

    /// Unknown: changed by the processor now and then; written by both the
    /// card and the processor; copied whole.
    #[test]
    fn pages_of_no_clear_use_are_unknown() {
        // Changed in 3 of 40 frames (its fill and twice more), by the
        // processor.
        let class = class_of(P, |s| {
            for f in 0..40 {
                let w = if f < 10 { 1 } else if f < 30 { 2 } else { 3 };
                s.looked(f, P, P + 1, &[(P, w)]);
            }
        });
        assert_eq!(class, Class::Unknown);
        // The card writes it, then the processor.
        let mut s = State::default();
        s.looked(0, P, P + 1, &[(P, 1)]);
        s.card(1, P * PAGE, 16);
        s.looked(2, P, P + 1, &[(P, 2)]);
        s.looked(3, P, P + 1, &[(P, 3)]);
        let r = s.report(4);
        assert_eq!(r.bytes[Class::Unknown as usize], PAGE as u64);
        assert_eq!(r.unknown[1], PAGE as u64, "both write: {r:?}");
        // Neither watched nor guarded.
        let mut s = State::default();
        s.unwatched(0, P * PAGE, 3 * PAGE);
        let r = s.report(1);
        assert_eq!(r.unknown[2], 3 * PAGE as u64);
        assert_eq!(r.ranges[Class::Unknown as usize], 1);
    }

    /// The page the write watch never reported is loaded once, "never
    /// written"; the one written once by the processor is loaded once by
    /// it; adjacent pages of one class are one range.
    #[test]
    fn ranges_and_first_writers_are_counted() {
        let mut s = State::default();
        for f in 0..10 {
            // P: never written; P+1, P+2: once by the processor; P+3: storage.
            if f == 0 {
                s.storage(0, (P + 3) * PAGE, PAGE);
            }
            s.looked(f, P, P + 4, &[(P + 1, 1), (P + 2, 1), (P + 3, 1)]);
        }
        let r = s.report(10);
        assert_eq!(r.bytes[Class::LoadedOnce as usize], 4 * PAGE as u64);
        assert_eq!(r.ranges[Class::LoadedOnce as usize], 1);
        assert_eq!(r.loaded, [PAGE as u64, 2 * PAGE as u64, PAGE as u64]);
        assert_eq!(r.looks[Class::LoadedOnce as usize], 40);
    }

    /// The allocation table: each page's bytes go to the signature of the
    /// call that made it (the newest where calls overlap), by class; pages
    /// in none are "not in one".
    #[test]
    fn classes_are_tabled_by_the_allocation_that_made_them() {
        let mut s = State::default();
        let pool = TitleAllocation::Virtual { kind: 0x2000_3000, flags: 3, protect: 0x2004 };
        let ring = TitleAllocation::Virtual { kind: 0x3000, flags: 1, protect: 0x404 };
        s.allocated(P * PAGE, 8 * PAGE, pool);
        // A commit inside the reservation, with other flags: its own row.
        s.allocated((P + 2) * PAGE, 2 * PAGE, ring);
        s.allocated((P + 20) * PAGE, PAGE, TitleAllocation::General { attributes: 3 << 29 | 7 << 24 | 0x12 });
        s.section(0xabc, 0x40);
        s.mapped((P + 4) * PAGE, 2 * PAGE, 0xabc, 0);
        for f in 0..20 {
            // P..P+8 loaded once, except P+2, P+3 changed every frame.
            s.looked(f, P, P + 8, &[(P, 1), (P + 2, f + 2), (P + 3, f + 2)]);
            s.looked(f, P + 20, P + 21, &[]);
            s.looked(f, P + 30, P + 31, &[]);
        }
        let r = s.report(20);
        let row = |layer: usize, name: &str| {
            r.by_allocation[layer]
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("{name} in {:?}", r.by_allocation[layer]))
                .1
        };
        let page = PAGE as u64;
        assert_eq!(row(0, "kind 0x20003000 flags 0x3 protect 0x2004"), [6 * page, 0, 0, 0]);
        assert_eq!(row(0, "kind 0x3000 flags 0x1 protect 0x404"), [0, 2 * page, 0, 0]);
        assert_eq!(row(0, "not in one"), [2 * page, 0, 0, 0]);
        assert_eq!(row(1, "memory type 3 low bits 0x12 (high 0x0)"), [page, 0, 0, 0]);
        assert_eq!(row(2, "flags 0x40"), [2 * page, 0, 0, 0]);
        let lines = r.lines("test", 0.0, 8);
        assert!(lines.iter().any(|l| l.starts_with("memory census by XMemVirtualAlloc: kind 0x3000 flags 0x1 protect 0x404: 0 MB: changes every frame 0 MB")), "{lines:#?}");
    }

    /// A page of the title's page pool shows one physical page, then
    /// (remapped) another: each backing keeps its own history, and the
    /// address's looks after the remap are the new backing's.
    #[test]
    fn a_remapped_page_is_reported_under_its_new_backing() {
        let mut s = State::default();
        s.section(0x10, 0x40);
        s.section(0x20, 0x80);
        s.mapped(P * PAGE, PAGE, 0x10, 0);
        // The processor writes it before every look: changes every frame.
        for f in 0..30 {
            s.looked(f, P, P + 1, &[(P, f + 1)]);
        }
        // The address shows the other section's first page now, filled once
        // by storage and never changed after.
        s.remapped(P * PAGE, PAGE);
        s.mapped(P * PAGE, PAGE, 0x20, 0);
        s.storage(30, P * PAGE, PAGE);
        for f in 30..60 {
            s.looked(f, P, P + 1, &[(P, 31)]);
        }
        let r = s.report(60);
        let page = PAGE as u64;
        assert_eq!(r.bytes[Class::EveryFrame as usize], page, "the first backing keeps its class: {r:?}");
        assert_eq!(r.bytes[Class::LoadedOnce as usize], page, "the second has its own: {r:?}");
        assert_eq!(r.loaded[0], page, "filled by storage: {r:?}");
        assert_eq!(r.remapped, page, "the first backing left the address once: {r:?}");
        let row = |name: &str| r.by_allocation[2].iter().find(|(n, _)| n == name).map(|(_, b)| *b);
        assert_eq!(row("flags 0x40"), Some([0, page, 0, 0]), "{:?}", r.by_allocation[2]);
        assert_eq!(row("flags 0x80"), Some([page, 0, 0, 0]), "{:?}", r.by_allocation[2]);
    }

    /// One physical page shown at two addresses is one page: a storage
    /// write through one view is found at a look through the other, and
    /// its bytes count once.
    #[test]
    fn a_page_shown_in_two_views_is_one_backing() {
        let mut s = State::default();
        s.section(0x10, 0x40);
        s.mapped(P * PAGE, PAGE, 0x10, 0);
        s.mapped((P + 5) * PAGE, PAGE, 0x10, 0);
        s.storage(0, P * PAGE, PAGE);
        s.looked(0, P + 5, P + 6, &[(P + 5, 1)]);
        for f in 1..20 {
            s.looked(f, P, P + 1, &[(P, 1)]);
            s.looked(f, P + 5, P + 6, &[(P + 5, 1)]);
        }
        let r = s.report(20);
        assert_eq!(r.bytes, [PAGE as u64, 0, 0, 0], "one backing: {r:?}");
        assert_eq!(r.loaded[0], PAGE as u64, "the change at the other view was storage's: {r:?}");
        assert_eq!(r.ranges[Class::LoadedOnce as usize], 1);
    }

    /// The five byte counts, each kept apart: every byte fed is in its
    /// count's whole once, and the whole is the sum of its parts by class
    /// (with the bytes on no backing looked at). The counts start again at
    /// each report.
    #[test]
    fn the_five_byte_counts_sum_to_their_totals_by_class() {
        use super::census::{Count, NOT_LOOKED};
        let mut s = State::default();
        let page = PAGE as u64;
        // P changes every frame, P + 1 is loaded once; P + 10 is a view
        // the card writes.
        s.section(0x10, 0x40);
        s.mapped((P + 10) * PAGE, PAGE, 0x10, 0);
        let (start, end) = (P * PAGE + 100, (P + 2) * PAGE - 50);
        let mut fed = [0u64; 6];
        for f in 0..10 {
            s.looked(f, P, P + 2, &[(P, f + 1), (P + 1, 1)]);
            s.spread(Count::Looked, start, end);
            fed[0] += (end - start) as u64;
            s.card(f, (P + 10) * PAGE, 64);
            s.looked(f + 1, P + 10, P + 11, &[(P + 10, f + 1)]);
            // A kernel asking over the two pages and four pages around
            // them that the census never looked at.
            s.spread(Count::Queried, (P - 3) * PAGE, (P + 3) * PAGE);
            fed[1] += 6 * page;
            s.spread(Count::ComparedCpu, (P + 1) * PAGE, (P + 1) * PAGE + 4096);
            fed[2] += 4096;
            s.spread(Count::ComparedCard, (P + 10) * PAGE, (P + 11) * PAGE);
            fed[3] += page;
            s.spread(Count::Sent, P * PAGE + 10, P * PAGE + 1010);
            fed[4] += 1000;
            // A piece across the view and the address after it.
            s.spread(Count::Fetched, (P + 10) * PAGE + page as usize / 2, (P + 11) * PAGE + 8);
            fed[5] += page / 2 + 8;
        }
        let r = s.report(10);
        assert_eq!(r.totals, fed, "each count's whole is what was fed");
        for (k, parts) in r.counts.iter().enumerate() {
            assert_eq!(parts.iter().sum::<u64>(), r.totals[k], "count {k}: its parts are its whole: {parts:?}");
        }
        let (every, once, card) = (Class::EveryFrame as usize, Class::LoadedOnce as usize, Class::CardOnly as usize);
        let looked = r.counts[Count::Looked as usize];
        assert_eq!((looked[every], looked[once]), (10 * (page - 100), 10 * (page - 50)), "{looked:?}");
        let queried = r.counts[Count::Queried as usize];
        assert_eq!((queried[every], queried[once], queried[NOT_LOOKED]), (10 * page, 10 * page, 40 * page), "{queried:?}");
        assert_eq!(r.counts[Count::ComparedCard as usize][card], 10 * page);
        assert_eq!(r.counts[Count::Fetched as usize][card], 10 * page / 2, "the view's share");
        assert_eq!(r.counts[Count::Fetched as usize][NOT_LOOKED], 80, "the address after it");
        let lines = r.count_lines();
        assert_eq!(lines.len(), 5, "{lines:#?}");
        assert!(lines.iter().all(|l| l.starts_with("memory census bytes a frame (10 frames): ")), "{lines:#?}");
        // Since the last report: nothing yet.
        let again = s.report(11);
        assert_eq!(again.totals, [0; 6]);
        assert_eq!(again.interval, 1);
    }

    /// Declared ranges the card never changed, and storage writes nothing
    /// looked at, are counted beside the classes.
    #[test]
    fn declared_but_unchanged_and_unlooked_storage_are_counted() {
        let mut s = State::default();
        s.declared(0, P * PAGE, 2 * PAGE);
        s.card(0, P * PAGE, 10);
        s.looked(1, P, P + 2, &[(P, 1), (P + 1, 1)]);
        s.storage(1, (P + 9) * PAGE, PAGE);
        s.remapped((P + 1) * PAGE, PAGE);
        let r = s.report(2);
        assert_eq!(r.declared, 2 * PAGE as u64);
        assert_eq!(r.declared_unchanged, PAGE as u64);
        assert_eq!(r.storage_unlooked, PAGE as u64);
        assert_eq!(r.remapped, PAGE as u64);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_card_memory_line_says_each_answer() {
        use super::card_memory_line as line;
        assert!(line(Some((false, false)), Some(true)).contains("not UMA"));
        assert!(line(Some((false, false)), Some(true)).contains("GPU upload heaps supported"));
        assert!(line(Some((true, true)), Some(false)).contains("not supported"));
        assert!(line(None, None).contains("query refused"));
    }

    /// The host device's answer (on the software adapter in tests).
    #[test]
    fn the_card_memory_query_is_answered() {
        let host = crate::host::test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
        let (arch, upload) = super::card_memory(&host.device);
        println!("{}", super::card_memory_line(arch, upload));
        assert!(arch.is_some(), "every D3D12 runtime answers ARCHITECTURE1");
    }
}
