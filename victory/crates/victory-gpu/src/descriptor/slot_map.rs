//! The title-slot to host-slot map on the CPU (design
//! `jobs/2026-11-25-slot-lookup`, `VICTORY_SLOT_MAP=1`; the layout the card
//! reads is `victory_shader::emit::slot_map`).
//!
//! A translated shader forms a title slot with the console's arithmetic and
//! reads this map for the host slot to use. Host slots are not the title's
//! numbering: one is made per distinct view in the title's heap ([`Key`]:
//! the record and the host resource behind it), shared by every title slot
//! that holds that view, and never rewritten while any version of the map
//! can name it. The host heap so holds the live views, however large the
//! title's heap is, and a title slot with no live view reads the typed null
//! of the kind its access declares, never another draw's descriptor.
//!
//! - **Host slots** come from the pool past the typed nulls and the host's
//!   own slots ([`POOL_BASE`]). Each lists the title slots that map to it
//!   (its reference count), to unmap them when its backing goes. With none
//!   left it is a zombie, stamped with the newest version published then,
//!   and free once every version up to that one is let go
//!   ([`SlotMap::take_freed`]). A zombie whose view comes back before then is
//!   taken again as it was made.
//! - **Versions**: [`SlotMap::publish`] copies the chunks written since the
//!   last publish into fresh chunks of the card's pool and returns the
//!   directory that names them. A published chunk is never written again,
//!   and a list holds the version it was filled with until it is reset. A
//!   chunk a publish replaced is free once no older version is held. Chunk
//!   0 is all misses and shared: the caller makes it zeros.
//! - **Refusals**: a view with no host slot free reads its null and is
//!   counted ([`Counts::refused`]); live views are never evicted for it. A
//!   publish with no free chunk is counted and fails ([`Full`]), or waits for
//!   a version to be let go ([`SlotMap::publish_waiting`]); it never names a
//!   chunk a held version reads.
//!
//! CPU state only: the descriptors in the host heap and the pool's buffer
//! are the caller's, made and written from what this returns.
#![cfg_attr(not(test), allow(dead_code))]

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Weak};

use victory_shader::emit::slot_map::{CHUNK_SHIFT, CHUNK_SLOTS, CLASSES, DIR_ENTRIES, HOST_BITS, MAP_SLOTS, entry};

/// Host view heap slots of the typed nulls, one per class: slot `c` is the
/// null of class `c` (`victory_shader::emit::slot_map::NULL_BASE` is 0).
pub(crate) const TYPED_NULLS: u32 = CLASSES;
/// Host view heap slots after the typed nulls that are the host's own: the
/// memory copies' views and the one-view tables of root views.
pub(crate) const HOST_OWN: u32 = 1 << 16;
/// The first slot of the shared pool.
pub(crate) const POOL_BASE: u32 = TYPED_NULLS + HOST_OWN;

const HOST_MASK: u32 = (1 << HOST_BITS) - 1;

/// `VICTORY_SLOT_MAP=1`, the translator's switch: its shaders read the map, and the host's
/// view heaps keep one (`HostHeap::new`).
pub(crate) fn on() -> bool {
    victory_shader::emit::slot_map::on()
}

/// `VICTORY_SLOT_MAP_POOL=<n>`: at most `n` shared slots in a view heap under the map
/// (capacity checks: a host heap far smaller than the title's); by default as many as
/// the host's heap holds.
pub(crate) fn pool_cap() -> u32 {
    static CAP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("VICTORY_SLOT_MAP_POOL")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(u32::MAX)
    })
}

/// Whether a shader with these root SRVs reads the map: its directory or
/// pool. Its draws and dispatches run only with a view heap under the map
/// bound (`State::bind`).
pub(crate) fn reads_map(root_srvs: &[u8]) -> bool {
    use victory_shader::emit::slot_map::{MAP_DIR, MAP_POOL};
    root_srvs.iter().any(|&s| s == MAP_DIR || s == MAP_POOL)
}

/// What a host slot is made from: the title's record (`generation << 32 |
/// i`, as `HostHeap::mirror` keys slots) and the host resource or copy the
/// descriptor views. A new backing for the same record is a new host slot.
pub(crate) type Key = (u64, u64);

/// What [`SlotMap::resolve`] did with a title slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Resolved {
    /// The title slot reads host slot `host`. `fresh`: the slot is new, and
    /// the caller makes its descriptor there before the next publish.
    Mapped { host: u32, fresh: bool },
    /// No host slot is free: the title slot reads its null (counted).
    Refused,
    /// The host could not make the view ([`SlotMap::resolve_with`]): the
    /// title slot reads its null, and the slot it was to take stays free
    /// (counted).
    Unmade,
    /// The slot lies past the title's heap or the map: it reads a miss.
    Outside,
}

/// A publish found no free chunk in the card's pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Full;

/// The map as one publish left it: the directory a list is filled with.
/// The chunks it names, and the host slots they name, stay as they are
/// while it is held.
pub(crate) struct Version {
    serial: u64,
    directory: Box<[u32]>,
}

impl Version {
    /// Which publish made it (0: the empty map).
    pub(crate) fn serial(&self) -> u64 {
        self.serial
    }

    /// `DIR_ENTRIES` dwords: the pool chunk of each title chunk.
    pub(crate) fn directory(&self) -> &[u32] {
        &self.directory
    }
}

/// The map's counters.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Counts {
    /// Host slots some title slot maps to.
    pub(crate) live: u32,
    /// Host slots no title slot maps to that a held version may name.
    pub(crate) zombies: u32,
    /// The most pool slots taken at once (live and zombies).
    pub(crate) peak: u32,
    /// Views refused for want of a host slot.
    pub(crate) refused: u64,
    /// Views the host could not make.
    pub(crate) unmade: u64,
    /// Publishes that made a version.
    pub(crate) publishes: u64,
    /// Chunks written to the card's pool.
    pub(crate) chunks_written: u64,
    /// Publishes that found no free chunk.
    pub(crate) no_chunk: u64,
}

/// A host slot of the pool.
struct Slot {
    key: Key,
    /// The class of the descriptor made there.
    class: u32,
    /// The title slots that map here: its reference count is the length.
    titles: Vec<u32>,
    /// With no title slot left: the newest serial published then.
    zombie: Option<u64>,
    /// Whether its key still finds it: no longer once its backing is gone.
    keyed: bool,
}

/// See the module.
pub(crate) struct SlotMap {
    /// The entry of each title slot (0: a miss).
    map: Vec<u32>,
    /// Each mapped title slot's place in its host slot's `titles`.
    place: Vec<u32>,
    /// One bit per chunk written since the last publish.
    dirty: Vec<u64>,
    /// Pool slots by host slot less [`POOL_BASE`], as far as any was taken
    /// (`None`: free).
    slots: Vec<Option<Slot>>,
    /// Free pool slots below `slots.len()`.
    unused: Vec<u32>,
    /// Pool slots in all.
    pool: u32,
    by_key: HashMap<Key, u32>,
    by_backing: HashMap<u64, HashSet<u32>>,
    /// Zombies in the order they became so, with their stamps.
    zombies: VecDeque<(u64, u32)>,
    zombie_count: u32,
    /// Host slots freed since [`SlotMap::take_freed`].
    freed: Vec<u32>,
    /// The pool chunk of each title chunk now (0: the chunk of misses).
    directory: Box<[u32]>,
    free_chunks: Vec<u32>,
    /// Chunks from here on were never taken.
    next_chunk: u32,
    /// Chunks of the card's pool in all, chunk 0 included.
    chunks: u32,
    /// Chunks a publish replaced, with that publish's serial.
    retired: VecDeque<(u64, u32)>,
    /// The newest publish.
    serial: u64,
    /// The newest version, always held: lists filled later take it.
    current: Arc<Version>,
    /// Every version made, oldest first, as far as the oldest still held.
    versions: VecDeque<(u64, Weak<Version>)>,
    counts: Counts,
}

impl SlotMap {
    /// A map of `title_slots` title slots (at most `MAP_SLOTS - 1`: the last
    /// entry is always a miss), `pool` host slots from [`POOL_BASE`], and a
    /// card pool of `chunks` chunks.
    pub(crate) fn new(title_slots: u32, pool: u32, chunks: u32) -> Self {
        assert!(
            u64::from(POOL_BASE) + u64::from(pool) <= 1 << HOST_BITS,
            "host slots past the entry's bits"
        );
        assert!(chunks >= 1, "chunk 0 is the misses'");
        let len = title_slots.min(MAP_SLOTS - 1) as usize;
        let title_chunks = len.div_ceil(CHUNK_SLOTS as usize);
        let current = Arc::new(Version {
            serial: 0,
            directory: vec![0; DIR_ENTRIES as usize].into_boxed_slice(),
        });
        SlotMap {
            map: vec![0; len],
            place: vec![0; len],
            dirty: vec![0; title_chunks.div_ceil(64)],
            slots: Vec::new(),
            unused: Vec::new(),
            pool,
            by_key: HashMap::new(),
            by_backing: HashMap::new(),
            zombies: VecDeque::new(),
            zombie_count: 0,
            freed: Vec::new(),
            directory: current.directory.clone(),
            free_chunks: Vec::new(),
            next_chunk: 1,
            chunks,
            retired: VecDeque::new(),
            serial: 0,
            versions: VecDeque::from([(0, Arc::downgrade(&current))]),
            current,
            counts: Counts::default(),
        }
    }

    /// Title slot `t` holds the view `key`, whose descriptor is of class
    /// `class`: it maps to that view's host slot, which is made if there is
    /// none (`fresh`) or refused if the pool is full.
    pub(crate) fn resolve(&mut self, t: u32, key: Key, class: u32) -> Resolved {
        debug_assert!(class != 0 && class < CLASSES, "class {class}");
        self.resolve_with(t, key, |_| Some(class))
    }

    /// [`SlotMap::resolve`] for a view whose class is known once it is made:
    /// a new host slot `h` is made by `make(h)`, which gives the class of the
    /// descriptor it made there, or `None` if it made none (the slot stays
    /// free, and the title slot reads a miss: [`Resolved::Unmade`]).
    pub(crate) fn resolve_with(&mut self, t: u32, key: Key, make: impl FnOnce(u32) -> Option<u32>) -> Resolved {
        let Some(&e) = self.map.get(t as usize) else {
            return Resolved::Outside;
        };
        if e != 0 {
            let h = e & HOST_MASK;
            if self.slot(h).is_some_and(|s| s.keyed && s.key == key) {
                return Resolved::Mapped { host: h, fresh: false };
            }
            self.unmap(t);
        }
        let (h, fresh) = match self.by_key.get(&key) {
            Some(&h) => (h, false),
            None => {
                let Some(h) = self.take() else {
                    self.counts.refused += 1;
                    return Resolved::Refused;
                };
                let class = match make(h) {
                    Some(class) if class != 0 && class < CLASSES => class,
                    _ => {
                        self.unused.push(h - POOL_BASE);
                        self.counts.unmade += 1;
                        return Resolved::Unmade;
                    }
                };
                self.slots[(h - POOL_BASE) as usize] = Some(Slot {
                    key,
                    class,
                    titles: Vec::new(),
                    zombie: None,
                    keyed: true,
                });
                self.by_key.insert(key, h);
                self.by_backing.entry(key.1).or_default().insert(h);
                (h, true)
            }
        };
        let slot = self.slots[(h - POOL_BASE) as usize]
            .as_mut()
            .expect("a keyed host slot is taken");
        if slot.zombie.take().is_some() {
            self.zombie_count -= 1;
        }
        self.place[t as usize] = slot.titles.len() as u32;
        slot.titles.push(t);
        self.map[t as usize] = entry(h, slot.class);
        self.mark(t);
        Resolved::Mapped { host: h, fresh }
    }

    /// Title slot `t` holds no live view (never written, or its view or the
    /// resource behind it is gone): it reads a miss.
    pub(crate) fn unmap(&mut self, t: u32) {
        let Some(&e) = self.map.get(t as usize) else {
            return;
        };
        if e == 0 {
            return;
        }
        let h = e & HOST_MASK;
        let slot = self.slots[(h - POOL_BASE) as usize]
            .as_mut()
            .expect("a mapped host slot is taken");
        let at = self.place[t as usize] as usize;
        slot.titles.swap_remove(at);
        if let Some(&moved) = slot.titles.get(at) {
            self.place[moved as usize] = at as u32;
        }
        if slot.titles.is_empty() {
            slot.zombie = Some(self.serial);
            self.zombies.push_back((self.serial, h));
            self.zombie_count += 1;
        }
        self.map[t as usize] = 0;
        self.mark(t);
    }

    /// The host resource `backing` is gone or replaced: every title slot
    /// whose view it backs reads a miss, and no later view finds those host
    /// slots again. Returns those title slots, to resolve them anew.
    pub(crate) fn invalidate_backing(&mut self, backing: u64) -> Vec<u32> {
        let Some(hosts) = self.by_backing.remove(&backing) else {
            return Vec::new();
        };
        let mut titles = Vec::new();
        for h in hosts {
            if let Some(slot) = self.slots[(h - POOL_BASE) as usize].as_mut() {
                if slot.keyed {
                    slot.keyed = false;
                    self.by_key.remove(&slot.key);
                }
                titles.extend_from_slice(&slot.titles);
            }
        }
        for &t in &titles {
            self.unmap(t);
        }
        titles
    }

    /// The entry of title slot `t` now (0: a miss).
    pub(crate) fn entry(&self, t: u32) -> u32 {
        self.map.get(t as usize).copied().unwrap_or(0)
    }

    /// The view at title slot `t` is to be made again (its texture shows other levels
    /// now): its host slot's key no longer finds it, so the next resolve of the view makes
    /// a fresh slot, and the old one, never rewritten, goes once its title slots move.
    /// Returns the title slots that map to it, to resolve anew.
    pub(crate) fn unkey(&mut self, t: u32) -> Vec<u32> {
        let e = self.entry(t);
        if e == 0 {
            return Vec::new();
        }
        let Some(slot) = self.slots.get_mut(((e & HOST_MASK) - POOL_BASE) as usize).and_then(Option::as_mut) else {
            return Vec::new();
        };
        if slot.keyed {
            slot.keyed = false;
            self.by_key.remove(&slot.key);
        }
        slot.titles.clone()
    }

    /// The view host slot `host` was made from, while it is live or a zombie.
    pub(crate) fn key_of(&self, host: u32) -> Option<Key> {
        self.slot(host).map(|s| s.key)
    }

    /// Host slots freed since the last call that are still free: no version
    /// names them, and what their descriptors held may go.
    pub(crate) fn take_freed(&mut self) -> Vec<u32> {
        let mut freed = std::mem::take(&mut self.freed);
        freed.retain(|&h| self.slot(h).is_none());
        freed.sort_unstable();
        freed.dedup();
        freed
    }

    /// The version a list is filled with: the chunks written since the last
    /// publish, each into a free chunk of the card's pool by `write(chunk,
    /// entries)` (`CHUNK_SLOTS` entries), and a directory naming them. With
    /// nothing written since, the version before.
    pub(crate) fn publish(&mut self, write: &mut dyn FnMut(u32, &[u32])) -> Result<Arc<Version>, Full> {
        self.collect();
        if self.dirty.iter().all(|&w| w == 0) {
            return Ok(Arc::clone(&self.current));
        }
        let dirty: Vec<usize> = (0..self.dirty.len() * 64)
            .filter(|&c| self.dirty[c / 64] >> (c % 64) & 1 != 0)
            .collect();
        // A chunk of misses only is the shared one.
        let misses: Vec<bool> = dirty.iter().map(|&c| self.chunk(c).iter().all(|&e| e == 0)).collect();
        let need = misses.iter().filter(|&&m| !m).count();
        let have = self.free_chunks.len() + (self.chunks - self.next_chunk) as usize;
        if need > have {
            self.counts.no_chunk += 1;
            return Err(Full);
        }
        self.serial += 1;
        let mut entries = vec![0u32; CHUNK_SLOTS as usize];
        for (&c, &miss) in dirty.iter().zip(&misses) {
            let chunk = if miss {
                0
            } else {
                let p = self.free_chunks.pop().unwrap_or_else(|| {
                    self.next_chunk += 1;
                    self.next_chunk - 1
                });
                let from = self.chunk(c);
                entries[..from.len()].copy_from_slice(from);
                entries[from.len()..].fill(0);
                write(p, &entries);
                self.counts.chunks_written += 1;
                p
            };
            let old = std::mem::replace(&mut self.directory[c], chunk);
            if old != 0 {
                self.retired.push_back((self.serial, old));
            }
        }
        self.dirty.fill(0);
        self.counts.publishes += 1;
        let v = Arc::new(Version {
            serial: self.serial,
            directory: self.directory.clone(),
        });
        self.versions.push_back((self.serial, Arc::downgrade(&v)));
        self.current = Arc::clone(&v);
        Ok(v)
    }

    /// [`SlotMap::publish`], calling `wait` each time it finds no free
    /// chunk (to wait for the oldest flight, whose lists then let their
    /// versions go) until it publishes; `None` when `wait` says nothing is
    /// left to wait for.
    pub(crate) fn publish_waiting(
        &mut self,
        write: &mut dyn FnMut(u32, &[u32]),
        wait: &mut dyn FnMut() -> bool,
    ) -> Option<Arc<Version>> {
        loop {
            match self.publish(write) {
                Ok(v) => return Some(v),
                Err(Full) if wait() => {}
                Err(Full) => return None,
            }
        }
    }

    pub(crate) fn counts(&self) -> Counts {
        let taken = (self.slots.len() - self.unused.len()) as u32;
        Counts {
            live: taken - self.zombie_count,
            zombies: self.zombie_count,
            ..self.counts
        }
    }

    fn slot(&self, host: u32) -> Option<&Slot> {
        self.slots.get(host.checked_sub(POOL_BASE)? as usize)?.as_ref()
    }

    /// The entries of title chunk `c`.
    fn chunk(&self, c: usize) -> &[u32] {
        let start = c * CHUNK_SLOTS as usize;
        &self.map[start..(start + CHUNK_SLOTS as usize).min(self.map.len())]
    }

    fn mark(&mut self, t: u32) {
        let c = (t >> CHUNK_SHIFT) as usize;
        self.dirty[c / 64] |= 1 << (c % 64);
    }

    /// A free pool slot; zombies no version names are freed first if none is.
    fn take(&mut self) -> Option<u32> {
        if self.unused.is_empty() && self.slots.len() as u32 == self.pool {
            self.collect();
        }
        let i = match self.unused.pop() {
            Some(i) => i,
            None if (self.slots.len() as u32) < self.pool => {
                self.slots.push(None);
                self.slots.len() as u32 - 1
            }
            None => return None,
        };
        let taken = (self.slots.len() - self.unused.len()) as u32;
        self.counts.peak = self.counts.peak.max(taken);
        Some(POOL_BASE + i)
    }

    /// The serial of the oldest version still held.
    fn oldest_held(&mut self) -> u64 {
        while self.versions.front().is_some_and(|(_, v)| v.strong_count() == 0) {
            self.versions.pop_front();
        }
        // Never empty: the map holds the newest itself.
        self.versions.front().map_or(self.serial, |&(serial, _)| serial)
    }

    /// Frees the zombies and chunks no held version names.
    fn collect(&mut self) {
        let oldest = self.oldest_held();
        while let Some(&(stamp, h)) = self.zombies.front() {
            if stamp >= oldest {
                break;
            }
            self.zombies.pop_front();
            let i = (h - POOL_BASE) as usize;
            // Taken again since, or freed by an earlier entry of the same stamp.
            if !self.slots[i].as_ref().is_some_and(|s| s.zombie == Some(stamp)) {
                continue;
            }
            let slot = self.slots[i].take().expect("a zombie is taken");
            if slot.keyed {
                self.by_key.remove(&slot.key);
            }
            if let Some(hosts) = self.by_backing.get_mut(&slot.key.1) {
                hosts.remove(&h);
                if hosts.is_empty() {
                    self.by_backing.remove(&slot.key.1);
                }
            }
            self.zombie_count -= 1;
            self.unused.push(i as u32);
            self.freed.push(h);
        }
        while let Some(&(replaced, chunk)) = self.retired.front() {
            if replaced > oldest {
                break;
            }
            self.retired.pop_front();
            self.free_chunks.push(chunk);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use victory_shader::emit::slot_map::{NULL_BASE, lookup};

    /// Texture2D SRV and raw-buffer UAV (`victory_shader::emit::slot_map::class`).
    const TEX: u32 = 2;
    const RAW_UAV: u32 = 11 | 32;

    /// The pool as the card sees it: what [`SlotMap::publish`] wrote.
    struct Card {
        pool: Vec<u32>,
    }

    impl Card {
        fn new(chunks: u32) -> Self {
            Card {
                pool: vec![0; (chunks * CHUNK_SLOTS) as usize],
            }
        }

        fn write(&mut self) -> impl FnMut(u32, &[u32]) + '_ {
            |chunk, entries| {
                let at = (chunk * CHUNK_SLOTS) as usize;
                self.pool[at..at + entries.len()].copy_from_slice(entries);
            }
        }

        fn publish(&mut self, map: &mut SlotMap) -> Result<Arc<Version>, Full> {
            map.publish(&mut self.write())
        }

        /// What an access of class `want` reads for title slot `t` under `v`.
        fn read(&self, v: &Version, t: u32, want: u32) -> u32 {
            lookup(v.directory(), &self.pool, t, want)
        }
    }

    /// Four chunks of title slots, `pool` host slots, `chunks` pool chunks.
    fn map(pool: u32, chunks: u32) -> SlotMap {
        SlotMap::new(4 * CHUNK_SLOTS, pool, chunks)
    }

    fn host(r: Resolved) -> u32 {
        match r {
            Resolved::Mapped { host, .. } => host,
            r => panic!("not mapped: {r:?}"),
        }
    }

    #[test]
    fn title_slots_holding_one_view_share_one_host_slot() {
        let (mut m, mut card) = (map(64, 8), Card::new(8));
        let hosts: BTreeSet<u32> = (0..100).map(|t| host(m.resolve(t * 37, (7, 1), TEX))).collect();
        assert_eq!(hosts.len(), 1, "100 title slots of one view take one host slot");
        let h = *hosts.first().unwrap();
        assert!(h >= POOL_BASE, "a shared slot comes from the pool, not the title's numbering");
        assert_eq!(m.counts().live, 1);
        let v = card.publish(&mut m).unwrap();
        for t in 0..100 {
            assert_eq!(card.read(&v, t * 37, TEX), h);
        }
        // A slot between them that the title never wrote reads its null.
        assert_eq!(card.read(&v, 38, TEX), NULL_BASE + TEX);
        // One of them rewritten to nothing leaves the other 99 on the slot:
        // they, and only they, go with its backing.
        m.unmap(50 * 37);
        let mut gone = m.invalidate_backing(1);
        gone.sort_unstable();
        assert_eq!(gone, (0..100).filter(|&t| t != 50).map(|t| t * 37).collect::<Vec<_>>());
        let v = card.publish(&mut m).unwrap();
        assert!((0..100).all(|t| card.read(&v, t * 37, TEX) == NULL_BASE + TEX));
        assert_eq!((m.counts().live, m.counts().zombies), (0, 1));
    }

    #[test]
    fn views_past_the_pool_read_their_null_and_are_counted() {
        let (mut m, mut card) = (map(64, 8), Card::new(8));
        for t in 0..64 {
            assert!(matches!(m.resolve(t, (u64::from(t), 1), TEX), Resolved::Mapped { fresh: true, .. }));
        }
        assert_eq!(m.resolve(64, (64, 1), TEX), Resolved::Refused);
        assert_eq!(m.entry(64), 0, "a refused view reads a miss");
        assert_eq!(m.counts().refused, 1);
        assert_eq!(m.counts().peak, 64);
        // A view held already still shares its slot while the pool is full.
        assert!(matches!(m.resolve(100, (5, 1), TEX), Resolved::Mapped { fresh: false, .. }));
        let v = card.publish(&mut m).unwrap();
        assert_eq!(card.read(&v, 64, TEX), NULL_BASE + TEX);
        // Live views are never evicted for it.
        for t in 0..64 {
            assert!(card.read(&v, t, TEX) >= POOL_BASE);
        }
    }

    #[test]
    fn a_held_version_reads_the_old_view_and_keeps_its_slot_until_let_go() {
        let (mut m, mut card) = (map(4, 8), Card::new(8));
        let old = host(m.resolve(10, (1, 1), TEX));
        // List A is filled with this version and held behind a fence.
        let a = card.publish(&mut m).unwrap();
        // The title rewrites slot 10; list B takes the next version.
        let new = host(m.resolve(10, (2, 1), TEX));
        assert_ne!(new, old, "a live host slot is never rewritten");
        let b = card.publish(&mut m).unwrap();
        assert_eq!(card.read(&a, 10, TEX), old);
        assert_eq!(card.read(&b, 10, TEX), new);
        // The rest of the pool is taken; the old slot is not given out
        // while A can still name it.
        host(m.resolve(20, (3, 1), TEX));
        host(m.resolve(21, (4, 1), TEX));
        assert_eq!(m.resolve(22, (5, 1), TEX), Resolved::Refused);
        assert_eq!(m.key_of(old), Some((1, 1)));
        let c = card.publish(&mut m).unwrap();
        assert_eq!(card.read(&a, 10, TEX), old, "a later publish leaves A's chunks alone");
        assert!(m.take_freed().is_empty());
        // A is reset: its slot is free for the next view.
        drop(a);
        let again = card.publish(&mut m).unwrap();
        assert!(Arc::ptr_eq(&again, &c), "nothing written since: the version before");
        assert_eq!(m.take_freed(), [old]);
        assert_eq!(m.resolve(22, (5, 1), TEX), Resolved::Mapped { host: old, fresh: true });
        assert_eq!(card.read(&b, 10, TEX), new);
    }

    #[test]
    fn a_backing_let_go_unmaps_its_title_slots() {
        let (mut m, mut card) = (map(8, 8), Card::new(8));
        host(m.resolve(1, (1, 9), TEX));
        host(m.resolve(2, (1, 9), TEX));
        host(m.resolve(3, (2, 9), RAW_UAV));
        let kept = host(m.resolve(4, (3, 8), TEX));
        let mut gone = m.invalidate_backing(9);
        gone.sort_unstable();
        assert_eq!(gone, [1, 2, 3]);
        let v = card.publish(&mut m).unwrap();
        assert_eq!(card.read(&v, 1, TEX), NULL_BASE + TEX);
        assert_eq!(card.read(&v, 3, RAW_UAV), NULL_BASE + RAW_UAV);
        assert_eq!(card.read(&v, 4, TEX), kept);
        // The record over its new backing is a new host slot, and the old key
        // never finds the slot it had again.
        assert!(matches!(m.resolve(1, (1, 10), TEX), Resolved::Mapped { fresh: true, .. }));
        assert!(matches!(m.resolve(5, (1, 9), TEX), Resolved::Mapped { fresh: true, .. }));
        assert_eq!(m.counts().live, 3);
    }

    /// A view made again (its texture shows other levels): its title slots move to one
    /// fresh host slot, and the old one, which a held version names, is not rewritten
    /// or given out until that version goes.
    #[test]
    fn a_view_made_again_moves_its_title_slots_to_a_fresh_slot() {
        let (mut m, mut card) = (map(2, 8), Card::new(8));
        let old = host(m.resolve(1, (1, 1), TEX));
        host(m.resolve(2, (1, 1), TEX));
        let held = card.publish(&mut m).unwrap();
        let mut titles = m.unkey(2);
        titles.sort_unstable();
        assert_eq!(titles, [1, 2]);
        let mut made = 0;
        for t in titles {
            let r = m.resolve_with(t, (1, 1), |_| {
                made += 1;
                Some(TEX)
            });
            assert_ne!(host(r), old, "title slot {t} on a fresh slot");
        }
        assert_eq!(made, 1, "made once for both");
        let v = card.publish(&mut m).unwrap();
        assert_ne!(card.read(&v, 1, TEX), old);
        assert_eq!(card.read(&held, 1, TEX), old, "the held version reads the old slot");
        assert_eq!(m.resolve(9, (9, 1), TEX), Resolved::Refused, "the old slot is not given out while held");
        drop(held);
        assert_eq!(m.resolve(9, (9, 1), TEX), Resolved::Mapped { host: old, fresh: true });
    }

    #[test]
    fn a_view_that_comes_back_takes_its_slot_as_it_was_made() {
        let mut m = map(4, 8);
        let h = host(m.resolve(1, (1, 1), TEX));
        m.unmap(1);
        assert_eq!(m.counts().zombies, 1);
        assert_eq!(m.resolve(2, (1, 1), TEX), Resolved::Mapped { host: h, fresh: false });
        assert_eq!(m.counts().zombies, 0);
        assert_eq!(m.counts().live, 1);
    }

    #[test]
    fn a_slot_no_title_slot_maps_to_is_free_once_no_version_names_it() {
        let (mut m, mut card) = (map(1, 8), Card::new(8));
        let h = host(m.resolve(5, (1, 1), TEX));
        let v1 = card.publish(&mut m).unwrap();
        m.unmap(5);
        let v2 = card.publish(&mut m).unwrap();
        // A chunk of misses only is the shared one.
        assert_eq!(v2.directory()[0], 0);
        assert_eq!(card.read(&v1, 5, TEX), h);
        assert_eq!(m.resolve(6, (2, 1), TEX), Resolved::Refused, "v1 still names the only slot");
        drop(v1);
        assert_eq!(m.resolve(6, (2, 1), TEX), Resolved::Mapped { host: h, fresh: true });
        assert_eq!((m.counts().live, m.counts().zombies), (1, 0));
    }

    #[test]
    fn a_publish_with_no_free_chunk_waits_for_the_oldest_version_and_is_counted() {
        // Chunk 0 (all misses) and two more.
        let (mut m, mut card) = (map(64, 3), Card::new(3));
        host(m.resolve(0, (1, 1), TEX));
        let a = card.publish(&mut m).unwrap();
        let h1 = host(m.resolve(1, (2, 1), TEX));
        let b = card.publish(&mut m).unwrap();
        let h2 = host(m.resolve(2, (3, 1), TEX));
        assert_eq!(card.publish(&mut m).err(), Some(Full), "both chunks are named by held versions");
        assert_eq!(m.counts().no_chunk, 1);
        let mut held = vec![a];
        let c = m
            .publish_waiting(&mut card.write(), &mut || held.pop().is_some())
            .expect("published once A let go");
        assert_eq!(m.counts().no_chunk, 2);
        assert_eq!(c.serial(), b.serial() + 1);
        assert_eq!((m.counts().publishes, m.counts().chunks_written), (3, 3));
        assert_eq!(card.read(&c, 2, TEX), h2);
        assert_eq!(card.read(&b, 1, TEX), h1);
        assert_eq!(card.read(&b, 2, TEX), NULL_BASE + TEX, "B keeps the map it was filled with");
    }

    #[test]
    fn a_view_the_host_cannot_make_reads_its_null_and_leaves_its_slot_free() {
        let (mut m, mut card) = (map(1, 8), Card::new(8));
        let mut made = Vec::new();
        assert_eq!(
            m.resolve_with(3, (1, 1), |h| {
                made.push(h);
                None
            }),
            Resolved::Unmade
        );
        assert_eq!((m.counts().unmade, m.counts().live), (1, 0));
        // The one slot is still there for the next view, made once, with the class
        // its making gave; a second title slot of it takes it as made.
        let h = host(m.resolve_with(4, (2, 1), |h| {
            made.push(h);
            Some(RAW_UAV)
        }));
        assert_eq!(made, [h, h]);
        assert_eq!(m.resolve_with(5, (2, 1), |_| panic!("made twice")), Resolved::Mapped { host: h, fresh: false });
        let v = card.publish(&mut m).unwrap();
        assert_eq!(card.read(&v, 3, TEX), NULL_BASE + TEX);
        assert_eq!(card.read(&v, 5, RAW_UAV), h);
    }

    #[test]
    fn a_shader_that_reads_the_map_is_told_apart() {
        use victory_shader::emit::{SLOT_MEMORY, SLOT_STRIDES};
        assert!(reads_map(&[3, victory_shader::emit::slot_map::MAP_DIR]));
        assert!(reads_map(&[victory_shader::emit::slot_map::MAP_POOL]));
        assert!(!reads_map(&[0, 7, SLOT_STRIDES, SLOT_MEMORY]));
    }

    #[test]
    fn the_last_entry_and_slots_past_the_title_heap_are_never_mapped() {
        let mut m = SlotMap::new(MAP_SLOTS + 5, 64, 8);
        assert_eq!(m.resolve(MAP_SLOTS - 1, (1, 1), TEX), Resolved::Outside);
        assert!(matches!(m.resolve(MAP_SLOTS - 2, (1, 1), TEX), Resolved::Mapped { .. }));
        let mut m = map(64, 8);
        assert_eq!(m.resolve(4 * CHUNK_SLOTS, (1, 1), TEX), Resolved::Outside);
    }
}
