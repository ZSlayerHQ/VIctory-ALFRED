//! Receipts for the host GPU's copies of title memory (`VICTORY_TEXTURE_RECEIPTS=1`, off by
//! default; design `jobs/2026-11-26-seam-dashes/receipts-design.md`).
//!
//! A texture's host copy is filled from title memory; a guest may stream the bytes in
//! after it wrote the view, so an upload taken before they arrived keeps the bytes from
//! before (solid black for the run). The notices meant to bring such a slot back
//! lived in queues the first heap's pass took for itself. Here freshness is a property of the
//! copy instead:
//!
//! - the runtime's writes into title memory are stamped in the write-watch journal when they
//!   end (`crate::watch::publish`);
//! - a texture's staging buffer carries a receipt, the journal's number taken before its
//!   bytes were read (`crate::texture`);
//! - each upload, at its submission, installs that receipt for the texture it wrote
//!   ([`installed`]), the GPU's own order being the install order;
//! - each view heap's pass walks all its texture slots, table and index alike, and owes the
//!   ones whose copy is older than a stamp over their memory
//!   (`crate::descriptor::host::receipts`).
//!
//! Off, nothing here runs and nothing elsewhere changes.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Whether receipts are on (`VICTORY_TEXTURE_RECEIPTS=1`), read once.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_TEXTURE_RECEIPTS").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                "VICTORY_TEXTURE_RECEIPTS=1: host copies of title textures carry receipts, and each view heap's \
                 pass walks its texture slots for copies older than the memory"
            );
        }
        on
    })
}

/// What the GPU copy of a title texture was filled from.
#[derive(Clone)]
pub(crate) struct Receipt {
    /// The journal's number taken before the bytes were read: every write stamped at or
    /// below it is in the copy.
    pub(crate) seq: u64,
    /// The title memory read: address and bytes (the levels from `first` on).
    pub(crate) start: usize,
    pub(crate) len: usize,
    pub(crate) first: u32,
    /// The host copy the upload wrote. `None`: an absent receipt, the memory found not
    /// mapped at `seq` (the slot shows nothing, and owes nothing until the memory changes).
    pub(crate) host: Option<std::sync::Weak<crate::resource::HostTexture>>,
    /// Levels on demand: the upload left a held level without tiles; `world`'s WAITING
    /// asks again once the card has room.
    pub(crate) short: bool,
    /// Never in debt: a texture the upload cannot read from title memory at all.
    pub(crate) never: bool,
}

/// The receipts, by title texture (`This`).
static TABLE: Mutex<Option<HashMap<usize, Receipt>>> = Mutex::new(None);

/// Textures a walk found in debt, until an install for them: an install that finds its
/// texture here is an upload the receipts caused ([`STATS`]).
static OWED: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// Since the start: receipts installed by uploads; of them, installs that lowered the
/// texture's receipt (an older copy submitted after a newer one); installs for textures a
/// walk had found in debt; absent receipts; receipts taken over memory nobody watches;
/// partly mapped refreshes skipped for the 30 frames (`Staging::refresh_timed`); textures
/// the upload cannot read.
pub(crate) static STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
pub(crate) const INSTALLS: usize = 0;
pub(crate) const LOWERED: usize = 1;
pub(crate) const CAUSED: usize = 2;
pub(crate) const ABSENT: usize = 3;
pub(crate) const UNWATCHED: usize = 4;
pub(crate) const PARTIAL_SKIPS: usize = 5;
pub(crate) const NEVER: usize = 6;

/// Tests: runs the ignored test `name` (its path from the crate's root) in a process of its
/// own with receipts on (the switch is read once a process), and fails if it does.
#[cfg(test)]
pub(crate) fn child(name: &str) {
    let exe = std::env::current_exe().expect("the test binary");
    let out = std::process::Command::new(exe)
        .args(["--exact", name, "--ignored", "--test-threads", "1", "--nocapture"])
        .env("VICTORY_TEXTURE_RECEIPTS", "1")
        .output()
        .expect("the child ran");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success() && text.contains("1 passed"), "{name}: {text}");
}

/// Tests: an install stops here until let go: it says it has arrived on the first sender
/// and waits on the receiver.
#[cfg(test)]
pub(crate) static INSTALL_GATE: Mutex<Option<(std::sync::mpsc::Sender<()>, std::sync::mpsc::Receiver<()>)>> =
    Mutex::new(None);

/// An upload of title texture `this` has its submission: its copy holds what its staging
/// buffer held when read at `seq`, `len` bytes from `start`, levels from `first`, into
/// host copy `host`. Overwrites what was there (install order is the GPU's).
pub(crate) fn installed(
    this: usize,
    seq: u64,
    (start, len, first): (usize, usize, u32),
    host: Option<std::sync::Weak<crate::resource::HostTexture>>,
    short: bool,
) {
    #[cfg(test)]
    if let Some((arrived, wait)) = INSTALL_GATE.lock().unwrap_or_else(|e| e.into_inner()).take() {
        let _ = arrived.send(());
        let _ = wait.recv();
    }
    STATS[INSTALLS].fetch_add(1, Relaxed);
    let receipt = Receipt { seq, start, len, first, host, short, never: false };
    let old = TABLE.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(this, receipt);
    if old.is_some_and(|old| old.host.is_some() && old.seq > seq) {
        STATS[LOWERED].fetch_add(1, Relaxed);
    }
    if OWED.lock().unwrap_or_else(|e| e.into_inner()).as_mut().is_some_and(|owed| owed.remove(&this)) {
        STATS[CAUSED].fetch_add(1, Relaxed);
    }
    if let Some(set) = NO_STAGING.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        set.remove(&this);
    }
}

/// The pass found the memory of placed texture `this` (`len` bytes from `start`, the levels
/// from `first`) not mapped, at or after `seq`: its slots show nothing until a map or a
/// write there is stamped above it.
pub(crate) fn absent(this: usize, seq: u64, (start, len, first): (usize, usize, u32)) {
    STATS[ABSENT].fetch_add(1, Relaxed);
    let receipt = Receipt { seq, start, len, first, host: None, short: false, never: false };
    TABLE.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(this, receipt);
}

/// Title texture `this` cannot be uploaded from title memory at all (no console layout, an
/// XOR mode): the walk does not owe it.
pub(crate) fn never(this: usize) {
    let mut table = TABLE.lock().unwrap_or_else(|e| e.into_inner());
    let table = table.get_or_insert_with(HashMap::new);
    if table.get(&this).is_none_or(|r| !r.never) {
        STATS[NEVER].fetch_add(1, Relaxed);
        let receipt = Receipt { seq: u64::MAX, start: 0, len: 0, first: 0, host: None, short: false, never: true };
        table.insert(this, receipt);
    }
}

/// The receipt of title texture `this`, if it has one.
pub(crate) fn get(this: usize) -> Option<Receipt> {
    TABLE.lock().unwrap_or_else(|e| e.into_inner()).as_ref()?.get(&this).cloned()
}

/// A walk found title texture `this` in debt.
pub(crate) fn owed(this: usize) {
    OWED.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashSet::new).insert(this);
}

/// The title released `this`: its receipt goes (a texture made later at the same address is
/// another).
pub(crate) fn forget(this: usize) {
    if let Some(table) = TABLE.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        table.remove(&this);
    }
    if let Some(owed) = OWED.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        owed.remove(&this);
    }
    if let Some(set) = NO_STAGING.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        set.remove(&this);
    }
    if let Some(wrote) = HOST_WROTE.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        wrote.remove(&this);
    }
}

/// Textures whose staging buffer could not be had at their last upload (`crate::texture`,
/// `take_staging` refused: the card or the process out of memory), until an install: their
/// debt is the named blocker `no_staging`, and they stay owed.
static NO_STAGING: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// The upload of title texture `this` had no staging buffer.
pub(crate) fn no_staging(this: usize) {
    NO_STAGING.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashSet::new).insert(this);
}

/// Whether title texture `this`'s last upload had no staging buffer.
pub(crate) fn staging_refused(this: usize) -> bool {
    NO_STAGING.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|set| set.contains(&this))
}

/// Title textures the host GPU wrote by a copy or resolve (`crate::texture::host_written`),
/// with the journal's number at the last such write: what the title writes over their
/// memory after it is in no copy (`world` uploads such a texture from memory no more).
static HOST_WROTE: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);

/// The host GPU writes title texture `this`; the journal's number is `seq`.
pub(crate) fn host_wrote(this: usize, seq: u64) {
    HOST_WROTE.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(this, seq);
}

/// The journal's number at the host GPU's last write of title texture `this`, if it wrote it.
pub(crate) fn host_wrote_at(this: usize) -> Option<u64> {
    HOST_WROTE.lock().unwrap_or_else(|e| e.into_inner()).as_ref()?.get(&this).copied()
}

/// One heap's last full sweep, for the stats file (kept by the heap's walk at the end of
/// each sweep: `crate::descriptor::host::receipts`).
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct Summary {
    /// Texture slots the walk holds, and of them visited in the sweep.
    pub(crate) slots: u64,
    pub(crate) visited: u64,
    /// Slots found in debt, and the bytes of their memory.
    pub(crate) debt: u64,
    pub(crate) debt_bytes: u64,
    /// The frames the oldest debt has stood, at the sweep's end.
    pub(crate) oldest: u64,
    /// Slots in debt by what holds them ([`BLOCKERS`]).
    pub(crate) blockers: [u64; 7],
    /// Slots left out, by why ([`EXCLUDED`]).
    pub(crate) excluded: [u64; 3],
    /// Of the `host_written` ones, those whose memory was written after the host GPU's last
    /// write and after their last upload: bytes the title put there that no copy has (such a
    /// texture may be one to upload again, not one the host owns).
    pub(crate) host_written_newer: u64,
    /// Slots the walk put in the queue; queued slots whose part-upload gate it opened.
    pub(crate) queued: u64,
    pub(crate) ungated: u64,
    /// Fresh slots of absent receipts (memory not mapped, nothing owed).
    pub(crate) unmapped: u64,
    /// The sweep period: the wall milliseconds the sweep took, and its passes. Paced to a
    /// second, but a pass walks at most 2 ms (at least one slot), so a large heap takes
    /// longer: about slots x `slot_ns` / (2 ms x passes a second) once passes are cut.
    pub(crate) sweep_ms: u64,
    pub(crate) passes: u64,
    /// The walk's microseconds a pass, mean and worst; nanoseconds a slot; passes cut by the
    /// time a pass may take.
    pub(crate) pass_us: u64,
    pub(crate) pass_us_worst: u64,
    pub(crate) slot_ns: u64,
    pub(crate) cut: u64,
}

/// The names of [`Summary::blockers`]: held for room on the card; in the queue but not due
/// (its backoff, a table's grace); a placed view with no title texture made yet; levels on
/// demand left without tiles; left by the pass's byte, time or refusal budget (still in debt
/// at the sweep's end); a host copy the host refused to make (not owed: it never will be);
/// a staging buffer that could not be had (owed again).
pub(crate) const BLOCKERS: [&str; 7] =
    ["no_room", "not_due", "never_made", "levels_short", "budget", "refused", "no_staging"];
pub(crate) const BUDGET: usize = 4;
pub(crate) const REFUSED: usize = 5;
pub(crate) const NO_STAGING_BLOCKER: usize = 6;
/// The names of [`Summary::excluded`]: views `alias` fills, textures the host GPU wrote,
/// textures the host renders to or writes (target, depth, unordered access).
pub(crate) const EXCLUDED: [&str; 3] = ["alias", "host_written", "target"];

/// Each view heap's last sweep, by the heap's number.
static SUMMARIES: Mutex<std::collections::BTreeMap<u32, Summary>> = Mutex::new(std::collections::BTreeMap::new());

/// Heap `id` finished a sweep.
pub(crate) fn swept(id: u32, summary: Summary) {
    SUMMARIES.lock().unwrap_or_else(|e| e.into_inner()).insert(id, summary);
}

/// `name: n` (or `"name":n`) for each blocker or exclusion.
fn named(names: &[&str], counts: &[u64], json: bool) -> String {
    let q = if json { "\"" } else { "" };
    let pairs: Vec<String> = names.iter().zip(counts).map(|(name, n)| format!("{q}{name}{q}:{n}")).collect();
    pairs.join(if json { "," } else { ", " })
}

/// Heap `id`'s line, every 600 frames.
pub(crate) fn line(id: u32, s: &Summary) {
    let n = |i: usize| STATS[i].load(Relaxed);
    tracing::info!(
        "texture receipts (VICTORY_TEXTURE_RECEIPTS=1), heap {id}'s last sweep: {} slots, {} visited; {} in debt \
         ({:.1} MB), the oldest for {} frames; blockers: {}; left out: {} (host written, and written again since: \
         {}); {} queued, {} gates opened, {} unmapped; the sweep took {} ms (paced to 1 s, a pass walking at most \
         2 ms) in {} passes, {} us a pass (worst {}), {} ns a slot, {} cut. Since the start: {} installs ({} \
         lowered, {} after a debt), {} absent, {} unwatched, {} partial skips, {} publishes ({} at other views)",
        s.slots,
        s.visited,
        s.debt,
        s.debt_bytes as f64 / f64::from(1 << 20),
        s.oldest,
        named(&BLOCKERS, &s.blockers, false),
        named(&EXCLUDED, &s.excluded, false),
        s.host_written_newer,
        s.queued,
        s.ungated,
        s.unmapped,
        s.sweep_ms,
        s.passes,
        s.pass_us,
        s.pass_us_worst,
        s.slot_ns,
        s.cut,
        n(INSTALLS),
        n(LOWERED),
        n(CAUSED),
        n(ABSENT),
        n(UNWATCHED),
        n(PARTIAL_SKIPS),
        crate::watch::PUBLISHES[0].load(Relaxed),
        crate::watch::PUBLISHES[1].load(Relaxed),
    );
}

/// The stats file's `texture_receipts`: `null` while off; else the counts since the start
/// (named `_total`, the stats file's rule) and each heap's last sweep (states as of it).
pub(crate) fn stats_json() -> String {
    if !on() {
        return "null".to_owned();
    }
    let n = |i: usize| STATS[i].load(Relaxed);
    let mut heaps = String::new();
    for (i, (id, s)) in SUMMARIES.lock().unwrap_or_else(|e| e.into_inner()).iter().enumerate() {
        heaps.push_str(&format!(
            "{}{{\"heap\":{id},\"slots\":{},\"visited\":{},\"debt\":{},\"debt_mb\":{:.2},\"oldest_frames\":{},\
             \"blockers\":{{{}}},\"excluded\":{{{}}},\"host_written_newer\":{},\"queued\":{},\"ungated\":{},\
             \"unmapped\":{},\"sweep_period_ms\":{},\"passes\":{},\"pass_us\":{},\"pass_us_worst\":{},\
             \"slot_ns\":{},\"cut\":{}}}",
            if i > 0 { "," } else { "" },
            s.slots,
            s.visited,
            s.debt,
            s.debt_bytes as f64 / f64::from(1 << 20),
            s.oldest,
            named(&BLOCKERS, &s.blockers, true),
            named(&EXCLUDED, &s.excluded, true),
            s.host_written_newer,
            s.queued,
            s.ungated,
            s.unmapped,
            s.sweep_ms,
            s.passes,
            s.pass_us,
            s.pass_us_worst,
            s.slot_ns,
            s.cut,
        ));
    }
    format!(
        "{{\"installs_total\":{},\"lowered_total\":{},\"caused_total\":{},\"absent_total\":{},\"unwatched_total\":{},\
         \"partial_skips_total\":{},\"never_total\":{},\"publishes_total\":{},\"fanned_out_total\":{},\"heaps\":[{heaps}]}}",
        n(INSTALLS),
        n(LOWERED),
        n(CAUSED),
        n(ABSENT),
        n(UNWATCHED),
        n(PARTIAL_SKIPS),
        n(NEVER),
        crate::watch::PUBLISHES[0].load(Relaxed),
        crate::watch::PUBLISHES[1].load(Relaxed),
    )
}
