//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! Writes into physical pages through another view are seen too: the
//! runtime tells us the views of the pages it wrote itself ([`writing`],
//! xmem's `write_physical` and DirectStorage's page destinations), and a
//! write by the title through any view of a guarded page faults like one
//! at the guarded address (the guard makes every view of the page
//! read-only, [`open`] gives them all back; not a page of more than
//! [`MANY_VIEWS`] views). A
//! system call writing into a guarded block fails (none is known to).
//!
//! With `VICTORY_GUARD_LEDGER=1` ([`guard_ledger`], F5's first slice) a
//! guard takes write rights only, keeping what the title asked per 4 KiB
//! page and per view ([`crate::ledger`]), and a release gives back exactly
//! that, not `PAGE_READWRITE`; a write the title's own protection refuses
//! is passed on to its handlers; the title's `VirtualProtect`,
//! `VirtualQuery`, `VirtualAlloc`, `VirtualFree` and `UnmapViewOfFile` are
//! answered here ([`title_virtual_protect`] and the others), under the
//! guards' lock, so that it sees its own protection and its frees end our
//! guards. Off, nothing of this runs.

use std::collections::BTreeSet;
use std::sync::Mutex;

mod journal;
mod journal_wide;
#[cfg(test)]
pub(crate) use journal_wide::no_hold_here;
pub(crate) use journal::{
    PUBLISHES, absent_number, ask_ahead, ask_ahead_done, ask_once, ask_once_end, asked, fresh_all, once, publish,
    recorded, top, writes, writes_top, wrote,
};
#[cfg(feature = "diagnostics")]
pub(crate) use journal::{clear_asked_pages, distinct_asked_pages, slow_asks};

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: BTreeSet<usize>,
    /// Blocks whose physical page was guarded together with its other views
    /// (those views' blocks are in it too): a write to one opens them all.
    /// A write range can be huge (`fread`), so only these are looked up.
    aliased: BTreeSet<usize>,
    /// Blocks `[first, last)` of the kernel's writes under way for the
    /// runtime ([`kernel_writing`]), one entry a write: left writable, and
    /// the other views of their pages left open.
    kernel: Vec<(usize, usize)>,
    /// Of those, the writes their call left pending ([`KernelWrite::pending`]).
    pending: Vec<Pending>,
    /// With `VICTORY_GUARD_LEDGER=1`: per native page, what the title asked
    /// under each guard ([`crate::ledger`]). Empty otherwise.
    ledger: crate::ledger::Ledger,
}

/// A kernel write its call left pending: under way until the status its
/// I/O reports at `status` (an `OVERLAPPED`'s first field) is final.
struct Pending {
    at: usize,
    end: usize,
    blocks: (usize, usize),
    status: usize,
    /// Bytes of the range a storage read's (`KernelWrite::storage`).
    storage: usize,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
    absent: BTreeSet::new(),
    aliased: BTreeSet::new(),
    kernel: Vec::new(),
    pending: Vec::new(),
    ledger: crate::ledger::Ledger::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    #[cfg(test)]
    journal_wide::no_book_here();
    let mut g = GUARDS.lock().unwrap_or_else(|e| e.into_inner());
    // Landings' stores done since: their holds end first (`landing_writing`).
    end_landed(&mut g);
    g
}

/// What the holds of kernel writes cost, since the start: walks of the pending reads
/// ([`kernel_writes_settled`] with some pending), reads of a pending read's status
/// ([`io_done`], one system call each), notes to the texture pass
/// (`crate::descriptor::host::remapped`), and takings of the guards' lock by a hold.
pub(crate) static HOLDS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
pub(crate) const SETTLES: usize = 0;
pub(crate) const STATUS_READS: usize = 1;
pub(crate) const NOTES: usize = 2;
pub(crate) const HOLD_LOCKS: usize = 3;

/// The frame-rate line's part for the holds ([`HOLDS`]), per frame since the last line.
pub(crate) fn holds_line(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    static LAST: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
    let n: Vec<f64> = HOLDS.iter().zip(&LAST).map(|(now, last)| {
        let now = now.load(Relaxed);
        (now - last.swap(now, Relaxed)) as f64 / frames
    }).collect();
    tracing::info!(
        "kernel-write holds per frame: {:.1} settles of pending reads, {:.1} status reads (one system call each), \
         {:.1} notes to the texture pass, {:.1} takings of the guards' lock",
        n[SETTLES], n[STATUS_READS], n[NOTES], n[HOLD_LOCKS]
    );
}

#[cfg(test)]
thread_local! {
    /// Tests: [`HOLDS`] for this thread alone.
    pub(crate) static HOLDS_HERE: std::cell::Cell<[u64; 4]> = const { std::cell::Cell::new([0; 4]) };
}

fn counted(what: usize) {
    HOLDS[what].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    #[cfg(test)]
    HOLDS_HERE.with(|h| {
        let mut n = h.get();
        n[what] += 1;
        h.set(n);
    });
}

/// Whether `start` begins in a block a copy made read-only and the runtime
/// has not remapped since ([`Guards::protected`]), and the runtime still
/// knows as a view ([`crate::inplace::pages_of`]): still a mapped view,
/// which Windows cannot write-watch, so the ask would fail. A block whose
/// view went without the runtime saying (the title's own views) is asked
/// about as before, its guard no longer covering what is there. Not the
/// blocks noted `absent`: those are not there to be watched either, but
/// asking about them is left as it was. `VICTORY_VIEW_ASK=1` asks as
/// before.
fn view_first(start: usize) -> bool {
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ASK.get_or_init(|| std::env::var_os("VICTORY_VIEW_ASK").is_some_and(|v| v == "1")) {
        return false;
    }
    let block = start / BLOCK;
    let protected = guards().protected.contains(&block);
    protected && !crate::inplace::pages_of(block * BLOCK).is_empty()
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    // A range whose first block the guards hold already (read-only by our
    // hand, or noted absent) is a view: no question to Windows. The
    // question walks the page tables to the end of the pages that share the
    // first one's state, and a guarded view is one long run of read-only
    // pages: most of a millisecond for the title's big heap, many times a frame, and most
    // of a frame's "guards" time.
    // `VICTORY_GUARD_ASK=1` asks every time, as before.
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let known = !*ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1")) && {
        let g = guards();
        let first = start / BLOCK;
        g.protected.contains(&first) || g.absent.contains(&first)
    };
    if !known && !is_view(start, end) {
        return None;
    }
    install();
    kernel_writes_settled();
    let mut g = guards();
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // The range's clean blocks, in order: read off the set in one pass,
    // not looked up one block at a time. A frame asks about gigabytes of
    // view memory, over a hundred thousand blocks, nearly all clean: the lookups were most
    // of a frame's "guards" time. What the loop adds to
    // the set meanwhile is the block it stands at or ones before it.
    let clean: Vec<usize> = g.clean.range(first..last).copied().collect();
    if clean.len() == last - first {
        // Nothing written, mapped or new since the range was last taken.
        return Some(written);
    }
    let mut next_clean = 0;
    let mut uncommitted_until = first;
    for block in first..=last {
        while next_clean < clean.len() && clean[next_clean] < block {
            next_clean += 1;
        }
        let is_clean = next_clean < clean.len() && clean[next_clean] == block;
        let mut open = block < last && !is_clean;
        if open && g.absent.contains(&block) {
            // Ordinary commits need no runtime mapping notification. Reuse
            // this asking's query for the following blocks in the same region.
            if block >= uncommitted_until {
                uncommitted_until = uncommitted_end(block);
            }
            open = block >= uncommitted_until;
            if open {
                g.absent.remove(&block);
            }
        }
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    Some(written)
}

/// Tests: arms the guard over `[at, at + len)` as a refresh before a copy
/// does, and gives the pages written since the last check; `None` if the
/// memory is not a view. A fixture's way to race a store that is about to
/// be made.
#[cfg(test)]
pub(crate) fn test_guard(at: usize, len: usize) -> Option<Vec<usize>> {
    guard(at, at.saturating_add(len))
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (hundreds of such ranges, hundreds of megabytes
/// copied a frame for them, most of them reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION};
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        #[cfg(feature = "diagnostics")]
        crate::stalls::kernel(crate::stalls::Kernel::Query);
        let n = crate::kernelstat::virtual_query(crate::kernelstat::Place::Guard, at, &mut info);
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole (a placeholder, or free): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting those not committed (unmapped) absent. The same physical pages
/// at every other view of them are made read-only too: a write through
/// one of those is this memory written, and only a fault can say so.
/// `VICTORY_VIEW_GUARD=0` leaves the other views as they are, as before.
///
/// Blocks the kernel is writing ([`kernel_writing`]) are left as they are,
/// at this view or another: the kernel's write would fail on a read-only
/// block instead of faulting into the guard. Not noted clean, they are
/// reported written at the next check, and guarded then.
fn protect(g: &mut Guards, from: usize, to: usize) {
    if g.kernel.is_empty() {
        return protect_views(g, from, to);
    }
    let mut writes: Vec<(usize, usize)> = g
        .kernel
        .iter()
        .map(|&(first, last)| (first.max(from), last.min(to)))
        .filter(|(first, last)| first < last)
        .collect();
    writes.sort_unstable();
    let mut at = from;
    for (first, last) in writes {
        if at < first {
            protect_views(g, at, first);
        }
        at = at.max(last);
    }
    if at < to {
        protect_views(g, at, to);
    }
}

/// [`protect`] for blocks the kernel is not writing (its other views may be).
/// A block whose pages the kernel is writing through another view is opened
/// again: guarded and noted clean, it would not see those bytes land (the
/// kernel's write faults nowhere), and its copy would keep the ones from
/// before; open, it is reported written at every check until the write ends.
fn protect_views(g: &mut Guards, from: usize, to: usize) {
    protect_here(g, from, to);
    if !view_guard() {
        return;
    }
    for (base, bytes) in crate::inplace::views_between(from.saturating_mul(BLOCK), to.saturating_mul(BLOCK)) {
        let first = (base / BLOCK).max(from);
        let last = (base.saturating_add(bytes).div_ceil(BLOCK)).min(to);
        for block in first..last {
            let shown = crate::inplace::pages_of(block * BLOCK);
            let written_elsewhere = |view: &usize| view / BLOCK != block && g.kernel_writes(view / BLOCK);
            if !g.kernel.is_empty() && shown.iter().any(written_elsewhere) {
                open_here(g, block * BLOCK, (block + 1) * BLOCK);
                continue;
            }
            // One view (nothing else to guard), or the page the title shows
            // wherever it has nothing (thousands of views): each view of
            // that one is guarded by itself.
            if shown.len() < 2 || shown.len() > MANY_VIEWS {
                continue;
            }
            #[cfg(feature = "diagnostics")]
            ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for alias in shown {
                let alias = alias / BLOCK;
                g.aliased.insert(alias);
                if alias != block && !g.kernel_writes(alias) {
                    protect_alias(g, alias);
                }
            }
        }
    }
}

impl Guards {
    /// Whether the kernel is writing `block` ([`kernel_writing`]).
    fn kernel_writes(&self, block: usize) -> bool {
        self.kernel.iter().any(|&(first, last)| (first..last).contains(&block))
    }
}

/// Views of one page past which its views are not guarded together.
pub(crate) const MANY_VIEWS: usize = 64;

/// Whether all the views of a guarded page are guarded together
/// (`VICTORY_VIEW_GUARD`).
fn view_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

/// Diagnostics: blocks guarded together with their other views, blocks
/// opened because another view of theirs was written, and blocks opened
/// because a view of theirs was mapped; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static ALIASES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Makes `block` read-only: another view of pages a copy has just taken
/// through its own view, so that a write through this one faults too. It is
/// not noted clean. The copies of title memory at this view have not taken
/// the pages: noted clean, the block would not be reported written at their
/// next check and they would keep the bytes from before the write (`guard`
/// reports a block that is read-only and not clean as written, then notes it
/// clean). A block that is clean stays so: had the pages been written since
/// its copies took them, `open` would have opened it.
fn protect_alias(g: &mut Guards, block: usize) {
    use windows_sys::Win32::System::Memory::PAGE_READONLY;
    if g.protected.contains(&block) {
        return;
    }
    if crate::ledger::on() {
        if lay(g, block * BLOCK, (block + 1) * BLOCK) {
            g.protected.insert(block);
        }
        return;
    }
    let mut old = 0;
    #[cfg(feature = "diagnostics")]
    crate::stalls::kernel(crate::stalls::Kernel::Protect);
    // SAFETY: title view memory; only its protection changes.
    let guarded = unsafe {
        crate::kernelstat::virtual_protect(crate::kernelstat::Place::Guard, (block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old)
    };
    if guarded != 0 {
        g.protected.insert(block);
    }
}

/// [`protect`] for one range, without the other views of its pages.
fn protect_here(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::PAGE_READONLY;
    if crate::ledger::on() {
        return protect_here_ledger(g, from, to);
    }
    let set = |from: usize, to: usize| {
        let mut old = 0;
        #[cfg(feature = "diagnostics")]
        crate::stalls::kernel(crate::stalls::Kernel::Protect);
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            crate::kernelstat::virtual_protect(
                crate::kernelstat::Place::Guard,
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
        } else if block < uncommitted_end(block) {
            g.absent.insert(block);
        }
    }
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    #[cfg(feature = "diagnostics")]
    crate::stalls::kernel(crate::stalls::Kernel::Query);
    let n = crate::kernelstat::virtual_query(crate::kernelstat::Place::Guard, block * BLOCK, &mut info);
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// What the host says of the page at `at`, if anything; counted for
/// `place` (the guards', or the title's when its own call is answered).
fn query(place: crate::kernelstat::Place, at: usize) -> Option<windows_sys::Win32::System::Memory::MEMORY_BASIC_INFORMATION> {
    // SAFETY: plain data filled in by the query.
    let mut info: windows_sys::Win32::System::Memory::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let n = crate::kernelstat::virtual_query(place, at, &mut info);
    (n != 0).then_some(info)
}

/// [`protect_here`] with the ledger: blocks `[from, to)` guarded and noted
/// clean where each is committed and guarded whole, block by block if not
/// all are; wholly uncommitted ones noted absent.
fn protect_here_ledger(g: &mut Guards, from: usize, to: usize) {
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
    };
    if lay(g, from * BLOCK, to * BLOCK) {
        return note(g, from, to);
    }
    for block in from..to {
        if lay(g, block * BLOCK, (block + 1) * BLOCK) {
            note(g, block, block + 1);
        } else if block < uncommitted_end(block) {
            g.absent.insert(block);
        }
    }
}

/// With the ledger: a guard over the committed pages of `[start, end)` that
/// none holds, each at [`crate::ledger::guarded`] of what the host shows
/// there now, which is what the title asked (outside a guard nothing of ours
/// changes it). Execute, no-access and the modifiers stay; a page the title
/// keeps from writes needs no change and is noted all the same, so that a
/// later request of the title's to write it keeps the guard. True if every
/// page of the range is committed and guarded.
fn lay(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::MEM_COMMIT;
    let generation = g.ledger.generation();
    let mut whole = true;
    for (gap, gap_end) in g.ledger.gaps(start, end) {
        let mut at = gap;
        while at < gap_end {
            let Some(info) = query(crate::kernelstat::Place::Guard, at) else {
                whole = false;
                break;
            };
            let region_end = (info.BaseAddress as usize).saturating_add(info.RegionSize).min(gap_end);
            if region_end <= at {
                whole = false;
                break;
            }
            if info.State == MEM_COMMIT {
                let requested = info.Protect;
                let applied = crate::ledger::guarded(requested);
                let mut old = 0;
                // SAFETY: title memory; only its protection changes.
                let guarded = applied == requested
                    || (unsafe {
                        crate::kernelstat::virtual_protect(crate::kernelstat::Place::Guard, at as *const _, region_end - at, applied, &mut old)
                    }) != 0;
                if guarded {
                    g.ledger.lay(at, region_end, requested, generation);
                } else {
                    whole = false;
                }
            } else {
                whole = false;
            }
            at = region_end;
        }
    }
    whole
}

/// With the ledger: the guards over the blocks of `[start, end)` end, and
/// each page gets back what the title asked for it, not `PAGE_READWRITE`.
/// Only where the host still shows the guard: a page whose protection
/// changed by a route we do not see is left as it is (and said once). True
/// if a guard was there.
fn open_here_ledger(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::MEM_COMMIT;
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    let blocks: Vec<usize> = g.protected.range(first..last).copied().collect();
    for block in &blocks {
        g.protected.remove(block);
        g.clean.remove(block);
    }
    let pieces = g.ledger.release(first * BLOCK, last.saturating_mul(BLOCK), None);
    for &(from, run) in &pieces {
        let applied = crate::ledger::guarded(run.requested);
        if applied == run.requested {
            continue;
        }
        let mut at = from;
        while at < run.end {
            let Some(info) = query(crate::kernelstat::Place::Guard, at) else { break };
            let region_end = (info.BaseAddress as usize).saturating_add(info.RegionSize).min(run.end);
            if region_end <= at {
                break;
            }
            if info.State == MEM_COMMIT && info.Protect == applied {
                let mut old = 0;
                // SAFETY: pages we guarded, still showing our guard.
                unsafe {
                    crate::kernelstat::virtual_protect(crate::kernelstat::Place::Guard, at as *const _, region_end - at, run.requested, &mut old)
                };
            } else {
                unmediated(at);
            }
            at = region_end;
        }
    }
    !blocks.is_empty() || !pieces.is_empty()
}

/// Pages under a guard whose protection changed by a route the runtime does
/// not see (F5's "unmediated route"), found at a release: left as they are.
pub(crate) static UNMEDIATED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn unmediated(at: usize) {
    if UNMEDIATED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
        tracing::warn!(
            address = format_args!("{at:#x}"),
            "a guarded page's protection changed by a route the guard ledger does not see; left as it is"
        );
    }
}

/// With the ledger: blocks `[start, end)` count as written at their next
/// check, here and at the other views of their pages, though their guards
/// stay (a write may have landed while the title's own call had lifted one).
fn note_written(g: &mut Guards, start: usize, end: usize) {
    for block in start / BLOCK..end.div_ceil(BLOCK) {
        g.clean.remove(&block);
        if g.aliased.contains(&block) {
            for alias in crate::inplace::pages_of(block * BLOCK) {
                g.clean.remove(&(alias / BLOCK));
            }
        }
    }
}

/// With the ledger: the title asked for `requested` over pages
/// `[start, end)` and the host shows it now. The guards there stay, at
/// [`crate::ledger::guarded`] of it, and their blocks count as written: a
/// write between the title's call and this one went past them. A guard
/// that cannot be laid again ends, its blocks reported written until the
/// next check guards them.
fn reguard(g: &mut Guards, start: usize, end: usize, requested: u32) {
    let applied = crate::ledger::guarded(requested);
    for (from, run) in g.ledger.request(start, end, requested) {
        let mut old = 0;
        let changed = applied != requested;
        // SAFETY: title pages under our guard; only their protection changes.
        if changed
            && unsafe {
                crate::kernelstat::virtual_protect(crate::kernelstat::Place::Guard, from as *const _, run.end - from, applied, &mut old)
            } == 0
        {
            g.ledger.forget(from, run.end);
            for block in from / BLOCK..run.end.div_ceil(BLOCK) {
                g.protected.remove(&block);
            }
        }
        note_written(g, from, run.end);
    }
}

/// Forgets the guards over `[at, at + len)`: what it shows is new (mapped,
/// unmapped, freed or decommitted), so no guard is ours there any more and
/// nothing is restored.
fn forget(g: &mut Guards, at: usize, len: usize) {
    let range = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    let absent: Vec<usize> = g.absent.range(range.clone()).copied().collect();
    for block in absent {
        g.absent.remove(&block);
    }
    let blocks: Vec<usize> = g.protected.range(range.clone()).copied().collect();
    for block in blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
    }
    let aliased: Vec<usize> = g.aliased.range(range).copied().collect();
    for block in aliased {
        g.aliased.remove(&block);
    }
    if crate::ledger::on() {
        use crate::ledger::NATIVE;
        g.ledger.forget(at / NATIVE * NATIVE, at.saturating_add(len).div_ceil(NATIVE).saturating_mul(NATIVE));
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded. The same physical pages at
/// every other view of them are opened too: the write that faulted (or is
/// ours, `crate::host::copy_mapped`) goes through one view, and every
/// guard over the pages must copy them again.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    let any = open_here(g, start, end);
    if start >= end || g.aliased.is_empty() {
        return any;
    }
    let aliased: Vec<usize> = g.aliased.range(start / BLOCK..end.div_ceil(BLOCK)).copied().collect();
    for block in aliased {
        g.aliased.remove(&block);
        for alias in crate::inplace::pages_of(block * BLOCK) {
            if alias / BLOCK != block && open_here(g, alias, alias.saturating_add(BLOCK)) {
                #[cfg(feature = "diagnostics")]
                ALIASES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            g.aliased.remove(&(alias / BLOCK));
        }
    }
    any
}

/// [`open`] for one range, without the other views of its pages.
fn open_here(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    if crate::ledger::on() {
        return open_here_ledger(g, start, end);
    }
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
        let mut old = 0;
        #[cfg(feature = "diagnostics")]
        crate::stalls::kernel(crate::stalls::Kernel::Protect);
        // SAFETY: a block we made read-only.
        unsafe {
            crate::kernelstat::virtual_protect(crate::kernelstat::Place::Guard, (block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old)
        };
    }
    !blocks.is_empty()
}

/// Before our own write into title memory `[at, at + len)`: opens its
/// guarded blocks. The runtime calls it (as `kernel_will_write`) before it
/// writes title memory from user mode, where a block guarded again meanwhile
/// faults into the guard. A write of the kernel's, which fails on a guarded
/// block instead, is [`kernel_writing`]. True if any block was guarded.
pub fn writing(at: usize, len: usize) -> bool {
    // The length is the title's (`fread`'s size times count, saturated).
    let opened = open(&mut guards(), at, at.saturating_add(len));
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
    opened
}

/// A write of the kernel's into title memory under way ([`kernel_writing`]);
/// dropped when the call that writes returns, or handed on to the I/O
/// ([`KernelWrite::pending`]) when the call leaves it pending.
#[must_use = "the kernel's write is under way until this is dropped"]
pub struct KernelWrite {
    at: usize,
    end: usize,
    blocks: (usize, usize),
    /// Bytes of the range a storage read's (`KernelWrite::storage`); 0 for
    /// a landing's store, whose bytes are the card's.
    storage: usize,
}

/// Before a call that has the kernel write title memory `[at, at + len)`
/// (`ReadFile` into it): opens its guarded blocks, as [`writing`], and
/// leaves them writable until what it returns is dropped. Meanwhile a copy
/// of the same pages at another view, which guards all their views, leaves
/// these as they are: the kernel's write fails on a read-only block instead
/// of faulting into the guard (`ERROR_INVALID_USER_BUFFER`, or
/// `ERROR_NOACCESS`); and those other views stay open, reported written at
/// every check. Dropped, it opens the blocks again, so a copy that took the
/// pages before the write ended takes them again.
///
/// Neither its end nor the opening calls change the thread's last error:
/// the title reads the one the writing call left.
pub fn kernel_writing(at: usize, len: usize) -> KernelWrite {
    let end = at.saturating_add(len);
    let blocks = (at / BLOCK, end.div_ceil(BLOCK));
    {
        let _error = LastError::kept();
        kernel_writes_settled();
        let mut g = guards();
        counted(HOLD_LOCKS);
        g.kernel.push(blocks);
        open(&mut g, at, end);
    }
    // A placed texture there changes with no descriptor written.
    counted(NOTES);
    crate::descriptor::host::remapped(at, len);
    KernelWrite { at, end, blocks, storage: 0 }
}

impl KernelWrite {
    /// `bytes` of the range are a storage read's that brought them
    /// (`ReadFile`, a DirectStorage read into the title's memory): the
    /// memory census counts them as storage's ([`crate::memclass`]). A
    /// landing's store marks none; its bytes are the card's, counted where
    /// they land (`crate::host::write_mapped`). A read that brought
    /// nothing marks none.
    pub fn storage(&mut self, bytes: usize) {
        self.storage = bytes.min(self.end - self.at);
    }
    /// The call left the write pending (an overlapped read that returned
    /// `ERROR_IO_PENDING`): it stays under way past the call, until the
    /// status its I/O reports at `status` (the first field of the call's
    /// `OVERLAPPED`, `STATUS_PENDING` until the I/O is done) is final. Then
    /// it ends as if dropped, all of `[at, at + len)` opened whatever the
    /// read brought (a partial or cancelled read too).
    ///
    /// The runtime does not see the title learn of the completion (by
    /// `GetOverlappedResult`, a completion port, its event or an APC: all
    /// the host's), so the guards look at the status at each check; until
    /// they see it final, the range and the other views of its pages are
    /// reported written at every check, so no copy keeps bytes from before.
    /// A status they cannot read keeps it under way ([`io_done`]): a title
    /// that frees the structure with its page leaves the range changed for
    /// good, which costs copies, not bytes.
    pub fn pending(self, status: *const std::ffi::c_void) {
        let _error = LastError::kept();
        let this = std::mem::ManuallyDrop::new(self);
        let mut g = guards();
        g.pending.push(Pending {
            at: this.at,
            end: this.end,
            blocks: this.blocks,
            status: status as usize,
            storage: this.storage,
        });
        PENDING_WRITES.fetch_add(1, std::sync::atomic::Ordering::Release);
    }
}

impl Drop for KernelWrite {
    fn drop(&mut self) {
        let _error = LastError::kept();
        counted(HOLD_LOCKS);
        end_write(&mut guards(), self.at, self.end, self.blocks);
        landed(self.at, self.end, self.storage);
    }
}

/// Ends a kernel write: its blocks leave the list, and are opened again, so
/// another view of these pages guarded meanwhile, taken before the write
/// ended, takes the pages again at its next check. The caller then tells
/// the texture pass ([`landed`]), without the guards' lock.
fn end_write(g: &mut Guards, at: usize, end: usize, blocks: (usize, usize)) {
    if let Some(i) = g.kernel.iter().position(|&b| b == blocks) {
        g.kernel.swap_remove(i);
    }
    open(g, at, end);
}

/// A landing's store into title memory under way ([`landing_writing`]).
#[must_use = "the landing's store is under way until this is dropped"]
pub struct LandingWrite {
    at: usize,
    end: usize,
    blocks: (usize, usize),
}

/// Holds whose stores are done, to be ended at the next taking of the guards'
/// lock ([`end_landed`]): their blocks stay open meanwhile, which costs a copy
/// at most (an open block counts as written), never a byte.
static LANDED: Mutex<Vec<(usize, usize, (usize, usize))>> = Mutex::new(Vec::new());
static LANDED_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Ends the holds of landings whose stores are done ([`LANDED`]), under the guards' lock the
/// caller holds.
fn end_landed(g: &mut Guards) {
    use std::sync::atomic::Ordering::{AcqRel, Acquire};
    if LANDED_N.load(Acquire) == 0 {
        return;
    }
    let done = std::mem::take(&mut *LANDED.lock().unwrap_or_else(|e| e.into_inner()));
    LANDED_N.fetch_sub(done.len(), AcqRel);
    for (at, end, blocks) in done {
        end_write(g, at, end, blocks);
    }
}

/// [`kernel_writing`] for a landing's store (the relay's or a submission's merge of what the
/// card wrote, `crate::shadow`), lighter: it keeps what a kernel write's hold is for (a guard
/// armed again between the opening and the store cannot refuse it), and leaves out three
/// things a landing does not need.
///
/// - The settle of pending reads ([`kernel_writes_settled`]): that ends *other* writes whose
///   I/O is done; this store's blocks are opened here whatever those are. A read left pending
///   is ended at the next guard check or texture pass, which settle before they look, so the
///   delay keeps its blocks open a little longer (copies, not bytes).
/// - The note to the texture pass at the start: a pass that took it before the bytes landed
///   would upload the old ones, so only the note at the end counts ([`landed`]), once.
/// - A second taking of the guards' lock: the end is put aside and done at the next taking
///   by anyone (the next piece's start, a check), so a landing takes it once per piece.
pub fn landing_writing(at: usize, len: usize) -> LandingWrite {
    let end = at.saturating_add(len);
    let blocks = (at / BLOCK, end.div_ceil(BLOCK));
    let _error = LastError::kept();
    let mut g = guards();
    counted(HOLD_LOCKS);
    g.kernel.push(blocks);
    open(&mut g, at, end);
    LandingWrite { at, end, blocks }
}

impl Drop for LandingWrite {
    fn drop(&mut self) {
        let _error = LastError::kept();
        {
            // Counted under the list's lock, so the count never runs below the list.
            let mut landed = LANDED.lock().unwrap_or_else(|e| e.into_inner());
            landed.push((self.at, self.end, self.blocks));
            LANDED_N.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
        landed(self.at, self.end, 0);
    }
}

/// Pending kernel writes ([`Guards::pending`]), counted apart so that a
/// check finds none without the guards' lock.
static PENDING_WRITES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Ends the pending kernel writes whose I/O is done ([`KernelWrite::pending`])
/// and tells the texture pass of each ([`landed`]). Asked at each check of
/// guarded memory, at each kernel write's start, and by the texture pass
/// before it takes its ranges (`HostHeap::texture_slots_remapped`), so a
/// read that settles is seen there even when no guarded memory is checked.
pub(crate) fn kernel_writes_settled() {
    use std::sync::atomic::Ordering::Acquire;
    if PENDING_WRITES.load(Acquire) == 0 {
        return;
    }
    counted(SETTLES);
    let ended: Vec<(usize, usize, usize)> = {
        let _error = LastError::kept();
        let mut g = guards();
        let mut ended = Vec::new();
        let mut i = 0;
        while i < g.pending.len() {
            if io_done(g.pending[i].status) {
                let p = g.pending.swap_remove(i);
                PENDING_WRITES.fetch_sub(1, std::sync::atomic::Ordering::Release);
                end_write(&mut g, p.at, p.end, p.blocks);
                ended.push((p.at, p.end, p.storage));
            } else {
                i += 1;
            }
        }
        ended
    };
    for (at, end, storage) in ended {
        landed(at, end, storage);
    }
}

/// A kernel write into `[at, end)` has ended: the placed textures there and
/// at every other view of its pages changed with no descriptor written. The
/// note at its start (`kernel_writing`) may have been taken by a texture
/// pass that uploaded what was there before the bytes landed (a read left
/// pending, or one that took longer than a frame), so the pass is told again
/// (`crate::descriptor::host::remapped`). `storage` is the part of the range
/// a storage read brought ([`KernelWrite::storage`]); only that is the
/// census's as storage's, a landing's store being the card's where it landed
/// (`crate::host::write_mapped`), so a refused stretch is attributed to
/// nobody. Not the other views of a page shown in more than [`MANY_VIEWS`],
/// as the guards leave those too.
fn landed(at: usize, end: usize, storage: usize) {
    // A storage read's bytes are there (`ReadFile`, a DirectStorage read
    // to an address).
    #[cfg(feature = "diagnostics")]
    if storage > 0 {
        crate::memclass::storage(at, storage);
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = storage;
    // With receipts: the bytes are there, at this address and at every other
    // view of its pages, however many (`crate::receipts`).
    if crate::receipts::on() {
        publish_landed(at, end);
    }
    let note = |at: usize, len: usize| {
        counted(NOTES);
        crate::descriptor::host::remapped(at, len);
    };
    note(at, end - at);
    for block in at / BLOCK..end.div_ceil(BLOCK) {
        // Counted first: a page of thousands of views is not copied for nothing.
        if !(2..=MANY_VIEWS).contains(&crate::inplace::views_of_page(block * BLOCK)) {
            continue;
        }
        let shown = crate::inplace::pages_of(block * BLOCK);
        if shown.len() < 2 || shown.len() > MANY_VIEWS {
            continue;
        }
        // The part of this block the write covers, at each other view.
        let (lo, hi) = (at.max(block * BLOCK) - block * BLOCK, end.min((block + 1) * BLOCK) - block * BLOCK);
        for view in shown.into_iter().filter(|&view| view != block * BLOCK) {
            note(view + lo, hi - lo);
        }
    }
}

/// With receipts: [`landed`]'s write published at `[at, end)` and at every
/// other view of each of its blocks, with no cap on the views (a receipt
/// over any of them must see it; the journal is a map, nothing overflows).
fn publish_landed(at: usize, end: usize) {
    publish(at, end);
    for block in at / BLOCK..end.div_ceil(BLOCK) {
        if crate::inplace::views_of_page(block * BLOCK) < 2 {
            continue;
        }
        let (lo, hi) = (at.max(block * BLOCK) - block * BLOCK, end.min((block + 1) * BLOCK) - block * BLOCK);
        for view in crate::inplace::pages_of(block * BLOCK).into_iter().filter(|&view| view != block * BLOCK) {
            PUBLISHES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            publish(view + lo, view + hi);
        }
    }
}

/// Whether the I/O status at `status` is known final: read whole, and not
/// `STATUS_PENDING`. Read without faulting (the title may free its
/// `OVERLAPPED` once the I/O is done). A status that cannot be read, or not
/// whole, says nothing (its page may be away for a moment while the I/O is
/// under way), and one that reads `STATUS_PENDING` again (the structure used
/// for the next read) is not final either: the write stays under way, which
/// costs copies only.
fn io_done(status: usize) -> bool {
    const STATUS_PENDING: usize = 0x103;
    let mut now = 0usize;
    counted(STATUS_READS);
    #[cfg(feature = "diagnostics")]
    crate::stalls::kernel(crate::stalls::Kernel::Read);
    // SAFETY: into a local of the size read; the kernel checks the source
    // and fails instead of faulting. A read that worked moved it whole.
    let read = unsafe {
        crate::kernelstat::read_memory(
            crate::kernelstat::Caller::IoStatus,
            status as *const u8,
            (&raw mut now).cast(),
            size_of::<usize>(),
        )
    };
    read && now != STATUS_PENDING
}

/// The thread's last error, put back when dropped.
struct LastError(u32);

impl LastError {
    fn kept() -> LastError {
        // SAFETY: no arguments.
        LastError(unsafe { windows_sys::Win32::Foundation::GetLastError() })
    }
}

impl Drop for LastError {
    fn drop(&mut self) {
        // SAFETY: plain value.
        unsafe { windows_sys::Win32::Foundation::SetLastError(self.0) };
    }
}

/// What `VICTORY_PAGE_WRITES` asks to be done with a write the guards do
/// not see once its bytes are there ([`title_memory_written`]), as bits:
/// 1, the guarded blocks under it are opened and count as written (host
/// copies of buffers and textures read through tables take them again);
/// 2, the placed textures picked by index that lie under it are due again;
/// 4, both also for reads to plain addresses when they have finished.
/// Unset or `1`: bits 1 and 2. `0`: nothing, counted only. `guards`: bit 1
/// alone. `slots`: bit 2 alone. `all`: everything.
///
/// On by default on this line: before this function came in, every write
/// into physical pages through a view of the runtime's own opened the
/// guards at each title address showing the pages and made the textures
/// there due again (the victory-amd fork family, commit 4d64efe1), and this
/// does the same by the bytes written instead of whole pages. victory-fork's
/// main branch has it off.
pub(crate) fn page_writes() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| parse_page_writes(std::env::var("VICTORY_PAGE_WRITES").ok().as_deref()))
}

/// [`page_writes`] of the switch's value (`None`: unset). Pure, for
/// testing.
fn parse_page_writes(value: Option<&str>) -> u8 {
    match value.map(str::trim) {
        Some("0") => 0,
        Some("guards") => 1,
        Some("slots") => 2,
        Some("all") => 7,
        _ => 3,
    }
}

/// Since the start: writes into physical pages through the runtime's own
/// views that landed at addresses the title has mapped (one for each run
/// of addresses); their bytes; guarded blocks that lay under them (host
/// copies that would otherwise keep the bytes from before); reads to
/// plain addresses that finished.
pub(crate) static WRITTEN: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The bytes of a write the guards do not see are there: title memory
/// `[at, at + len)`. `pages`: DirectStorage wrote them into physical pages
/// through a view of the runtime's own, and this is an address where the
/// title has those pages mapped (no title address faulted, nothing was
/// mapped or unmapped; the shortcut this file's header names). Otherwise a
/// read to a plain address has finished, of which [`writing`] was told
/// before it began.
///
/// Upstream's c75fdba does the same for its trackers (its `Writing` marks
/// the pages once more when dropped; page destinations are reported at
/// each address showing them). Here it is counted always and acted on as
/// [`page_writes`] says. Called without the runtime's page-table lock;
/// takes the guards' lock, then, alone, the lock of the ranges kept for
/// the texture pass.
///
/// A read to a plain address tells the texture pass in any case, at the
/// address and at every other view of its pages ([`landed`]): a pass that
/// took the note at its start ([`writing`]) while the bytes were being
/// read uploaded what was there before them.
pub fn title_memory_written(at: usize, len: usize, pages: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    // For the upload recheck: a storage request's bytes are there.
    crate::host::recheck::storage_completed(at, len);
    if len == 0 {
        return;
    }
    let mode = page_writes();
    let end = at.saturating_add(len);
    if pages {
        WRITTEN[0].fetch_add(1, Relaxed);
        WRITTEN[1].fetch_add(len as u64, Relaxed);
        // DirectStorage's physical-page destination (a read to an address
        // is the census's in `landed`).
        #[cfg(feature = "diagnostics")]
        crate::memclass::storage(at, len);
        // With receipts: its bytes are there, at this address (the runtime
        // calls this at every address that shows the pages).
        if crate::receipts::on() {
            publish(at, end);
        }
    } else {
        WRITTEN[3].fetch_add(1, Relaxed);
        // A read to a plain address that finished: its bytes are storage's.
        landed(at, end, len);
        if mode & 4 == 0 {
            return;
        }
    }
    {
        let mut g = guards();
        let guarded = g.protected.range(at / BLOCK..end.div_ceil(BLOCK)).count() as u64;
        if pages {
            WRITTEN[2].fetch_add(guarded, Relaxed);
        }
        if mode & 1 != 0 && guarded != 0 {
            open(&mut g, at, end);
        }
    }
    // A placed texture there changed with no descriptor written.
    crate::descriptor::host::page_written(at, len);
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again — and are, the pages noted written, which also has the next check
/// guard the new view (`protect`).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    if ranges.is_empty() {
        return;
    }
    let mut g = guards();
    for &(at, n) in &ranges {
        if open_here(&mut g, at, at.saturating_add(n)) {
            #[cfg(feature = "diagnostics")]
            ALIASES[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Held shared by a read of title memory in place, and alone by the runtime
/// while it changes what title memory shows.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
///
/// Only with `VICTORY_MAPPING_RANGES=0` ([`by_range`]); otherwise reads and
/// changes hold the bytes they touch ([`SPANS`]).
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Whether a read in place and a change of mapping keep apart only when
/// their bytes meet ([`SPANS`]), not whatever memory each touches
/// ([`MAPPING`]); `VICTORY_MAPPING_RANGES=0` holds all of title memory for
/// each, as before. On by default.
///
/// With one lock over all of title memory each texture refresh waited for
/// every change of mapping anywhere, and each change for the refresh under
/// way. While the world streams in the title changes mappings hundreds of
/// times a frame, many a page at a time (`XMemMapPhysicalPagesScatter`
/// maps each page on its own). With the one lock the refreshes waited a
/// quarter of a frame for title memory to hold still; holding only the
/// bytes touched, none of a shorter frame (medians, a pair of runs on
/// another branch, `job/speed-all`). On `world` the same wait was the long
/// freeze at world arrival (nearly all of a frame's texture work).
pub(crate) fn by_range() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MAPPING_RANGES").is_none_or(|v| v != "0"))
}

/// Reads of title memory in place under way, and changes of its mapping,
/// by the bytes they touch ([`by_range`]).
struct Spans {
    /// To tell them apart.
    next: u64,
    /// Reads under way: number, start and end.
    reads: Vec<(u64, usize, usize)>,
    /// Changes: number, start, end, and whether under way (else waiting for
    /// the reads over its bytes to end; no read begins over them meanwhile,
    /// so that one read after another cannot keep a change waiting).
    changes: Vec<(u64, usize, usize, bool)>,
    /// Threads waiting on [`SPANS_LEFT`].
    waiting: usize,
}

static SPANS: Mutex<Spans> = Mutex::new(Spans {
    next: 0,
    reads: Vec::new(),
    changes: Vec::new(),
    waiting: 0,
});

/// Told when a read or a change ends while a thread waits.
static SPANS_LEFT: std::sync::Condvar = std::sync::Condvar::new();

fn spans() -> std::sync::MutexGuard<'static, Spans> {
    SPANS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Waits until a read or a change ends.
fn spans_wait(mut s: std::sync::MutexGuard<'static, Spans>) -> std::sync::MutexGuard<'static, Spans> {
    s.waiting += 1;
    let mut s = SPANS_LEFT.wait(s).unwrap_or_else(|e| e.into_inner());
    s.waiting -= 1;
    s
}

/// Whether `[a, b)` and `[c, d)` share a byte.
fn meet(a: usize, b: usize, c: usize, d: usize) -> bool {
    a < d && c < b
}

impl Spans {
    fn read(&mut self, at: usize, end: usize) -> Span {
        self.next += 1;
        self.reads.push((self.next, at, end));
        Span(self.next, false)
    }
}

/// A read's or a change's place in [`SPANS`] (its number, and whether a
/// change), given up when dropped.
struct Span(u64, bool);

impl Drop for Span {
    fn drop(&mut self) {
        let mut s = spans();
        if self.1 {
            s.changes.retain(|c| c.0 != self.0);
        } else if let Some(i) = s.reads.iter().position(|r| r.0 == self.0) {
            s.reads.swap_remove(i);
        }
        if s.waiting > 0 {
            SPANS_LEFT.notify_all();
        }
    }
}

/// What a read or a change of mapping holds: its bytes ([`by_range`]) or
/// the lock over all of title memory.
enum Held<G> {
    All(#[allow(dead_code)] G),
    Bytes(#[allow(dead_code)] Span),
}

/// Diagnostics, since the last frame-rate line: changes of title memory's
/// mapping; the microseconds they waited for reads in place to end, and
/// those they held title memory (from then to their end, which is how long
/// reads of their bytes, texture refreshes among them, could be kept
/// waiting); and the longest wait and the longest hold of one change.
#[cfg(feature = "diagnostics")]
pub(crate) static REMAPS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping {
    _held: Held<std::sync::RwLockWriteGuard<'static, ()>>,
    /// Diagnostics: when it began to hold ([`REMAPS`]).
    #[cfg(feature = "diagnostics")]
    since: std::time::Instant,
}

#[cfg(feature = "diagnostics")]
impl Drop for Remapping {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        let us = self.since.elapsed().as_micros() as u64;
        REMAPS[2].fetch_add(us, Relaxed);
        REMAPS[4].fetch_max(us, Relaxed);
    }
}

/// The runtime is about to map, unmap or move physical pages of title
/// memory, changing what `[at, at + len)` shows: until what it returns is
/// dropped, no read of those bytes in place is under way and none begins
/// ([`title_memory_reading`]). Taken around the whole change, the unmapping
/// and the mapping again, over all the bytes whose mapping it changes.
pub fn title_memory_remapping(at: usize, len: usize) -> Remapping {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let held = if by_range() {
        let end = at.saturating_add(len);
        let mut s = spans();
        s.next += 1;
        let id = s.next;
        s.changes.push((id, at, end, false));
        while s.reads.iter().any(|r| meet(r.1, r.2, at, end))
            || s.changes.iter().any(|c| c.3 && meet(c.1, c.2, at, end))
        {
            s = spans_wait(s);
        }
        if let Some(c) = s.changes.iter_mut().find(|c| c.0 == id) {
            c.3 = true;
        }
        Held::Bytes(Span(id, true))
    } else {
        Held::All(MAPPING.write().unwrap_or_else(|e| e.into_inner()))
    };
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        let us = started.elapsed().as_micros() as u64;
        REMAPS[0].fetch_add(1, Relaxed);
        REMAPS[1].fetch_add(us, Relaxed);
        REMAPS[3].fetch_max(us, Relaxed);
    }
    Remapping {
        _held: held,
        #[cfg(feature = "diagnostics")]
        since: std::time::Instant::now(),
    }
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] Held<std::sync::RwLockReadGuard<'static, ()>>);

/// For reading title memory `[at, at + len)` in place: until what it
/// returns is dropped, the runtime changes no mapping of those bytes, so
/// memory found mapped stays mapped. Not to be taken twice by one thread,
/// nor held across anything that maps title memory.
pub fn title_memory_reading(at: usize, len: usize) -> Reading {
    if !by_range() {
        return Reading(Held::All(MAPPING.read().unwrap_or_else(|e| e.into_inner())));
    }
    let end = at.saturating_add(len);
    let mut s = spans();
    while s.changes.iter().any(|c| meet(c.1, c.2, at, end)) {
        s = spans_wait(s);
    }
    Reading(Held::Bytes(s.read(at, end)))
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping of those bytes (or waits to), for a caller
/// that has another way to read and would rather not wait.
pub fn title_memory_reading_now(at: usize, len: usize) -> Option<Reading> {
    if !by_range() {
        return match MAPPING.try_read() {
            Ok(guard) => Some(Reading(Held::All(guard))),
            Err(std::sync::TryLockError::Poisoned(e)) => Some(Reading(Held::All(e.into_inner()))),
            Err(std::sync::TryLockError::WouldBlock) => None,
        };
    }
    let end = at.saturating_add(len);
    let mut s = spans();
    if s.changes.iter().any(|c| meet(c.1, c.2, at, end)) {
        return None;
    }
    Some(Reading(Held::Bytes(s.read(at, end))))
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    #[cfg(feature = "diagnostics")]
    crate::memclass::remapped(at, len);
    // With receipts: what it shows is new (called after a map and before an
    // unmap, with the change's span held), so a copy taken before is in debt.
    if crate::receipts::on() {
        publish(at, at.saturating_add(len));
    }
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    forget(&mut guards(), at, len);
}

/// Whether F5's guard ledger is on (`VICTORY_GUARD_LEDGER=1`; off by
/// default). With it the runtime answers the title's protection calls
/// itself ([`title_virtual_protect`], [`title_virtual_query`],
/// [`title_virtual_alloc`], [`title_virtual_free`], [`title_unmap_view`]),
/// so that the title sees its own protection under our guards and its
/// frees end them; without it they pass to the host unchanged.
pub fn guard_ledger() -> bool {
    crate::ledger::on()
}

/// The title's `VirtualProtect`. The host's call with the title's values
/// first (its checks, errors and last error stand), then, under our guards
/// in the range, the guard laid again over what the title asked
/// ([`reguard`]); `old` says what the title asked for the first page, not
/// our guard. Held under the guards' lock throughout, so a fault, a laying
/// or another call of the title's comes before or after it; `old` is
/// written after the lock is left (it may lie in guarded memory).
///
/// # Safety
///
/// As `VirtualProtect`'s: `old` is null or points at room for a `u32`.
pub unsafe fn title_virtual_protect(at: *const std::ffi::c_void, len: usize, new: u32, old: *mut u32) -> i32 {
    use crate::kernelstat::{Place, virtual_protect};
    use crate::ledger::NATIVE;
    if !crate::ledger::on() {
        // SAFETY: the title's arguments, as the host takes them.
        return unsafe { virtual_protect(Place::Title, at, len, new, old) };
    }
    let (done, first) = {
        let mut g = guards();
        let page = at as usize / NATIVE * NATIVE;
        let first = g.ledger.at(page).map(|(_, run)| run.requested);
        // SAFETY: the title's arguments, as the host takes them.
        let done = unsafe { virtual_protect(Place::Title, at, len, new, old) };
        if done == 0 {
            return done;
        }
        let _error = LastError::kept();
        let end = (at as usize).saturating_add(len.max(1)).div_ceil(NATIVE).saturating_mul(NATIVE);
        reguard(&mut g, page, end, new);
        (done, first)
    };
    if let Some(requested) = first
        && !old.is_null()
    {
        // SAFETY: the host just wrote it; the title's room for the answer.
        unsafe { old.write(requested) };
    }
    done
}

/// The title's `VirtualQuery`: the host's answer, with what the title asked
/// in place of our guards, and the region as long as the title would see
/// it (state, type, allocation and its protection the same), not cut where
/// a guard begins or ends. `AllocationProtect` is the host's: a guard never
/// changes it.
///
/// # Safety
///
/// As `VirtualQuery`'s: `out` points at room for `len` bytes.
pub unsafe fn title_virtual_query(
    at: *const std::ffi::c_void,
    out: *mut windows_sys::Win32::System::Memory::MEMORY_BASIC_INFORMATION,
    len: usize,
) -> usize {
    use crate::kernelstat::{Place, virtual_query_raw};
    use windows_sys::Win32::System::Memory::MEM_COMMIT;
    let size = size_of::<windows_sys::Win32::System::Memory::MEMORY_BASIC_INFORMATION>();
    if !crate::ledger::on() || len < size {
        // SAFETY: the title's arguments, as the host takes them.
        return unsafe { virtual_query_raw(Place::Title, at, out, len) };
    }
    let answer = {
        let g = guards();
        let Some(mut info) = query(Place::Title, at as usize) else {
            drop(g);
            // SAFETY: as above: the host's own failure and last error.
            return unsafe { virtual_query_raw(Place::Title, at, out, len) };
        };
        if info.State == MEM_COMMIT && !g.ledger.is_empty() {
            let _error = LastError::kept();
            let start = info.BaseAddress as usize;
            let want = g.ledger.at(start).map_or(info.Protect, |(_, run)| run.requested);
            let (mut end, mut cursor) = (start, start);
            'walk: while let Some(host) = query(Place::Title, cursor) {
                if host.State != info.State || host.Type != info.Type || host.AllocationBase != info.AllocationBase {
                    break;
                }
                let host_end = (host.BaseAddress as usize).saturating_add(host.RegionSize);
                if host_end <= cursor {
                    break;
                }
                let mut p = cursor;
                while p < host_end {
                    let (protect, next) = match g.ledger.at(p) {
                        Some((_, run)) => (run.requested, run.end.min(host_end)),
                        None => (host.Protect, g.ledger.next_after(p).map_or(host_end, |s| s.min(host_end))),
                    };
                    if protect != want {
                        end = p;
                        break 'walk;
                    }
                    p = next;
                }
                (end, cursor) = (host_end, host_end);
            }
            info.Protect = want;
            info.RegionSize = end - start;
        }
        info
    };
    // SAFETY: the title's room for the answer, written after the lock is
    // left (it may lie in guarded memory).
    unsafe { out.write(answer) };
    size
}

/// The bytes a free or an unmap at `at` of `len` takes away: its pages, or
/// with `len` 0 the whole allocation `at` begins (`MEM_RELEASE`'s and a
/// view's).
fn extent(at: usize, len: usize) -> (usize, usize) {
    use crate::ledger::NATIVE;
    use windows_sys::Win32::System::Memory::MEM_FREE;
    if len != 0 {
        return (at / NATIVE * NATIVE, at.saturating_add(len).div_ceil(NATIVE).saturating_mul(NATIVE));
    }
    let Some(info) = query(crate::kernelstat::Place::Guard, at).filter(|info| info.State != MEM_FREE) else {
        return (at, at);
    };
    let base = info.AllocationBase as usize;
    let mut end = (info.BaseAddress as usize).saturating_add(info.RegionSize);
    while let Some(next) = query(crate::kernelstat::Place::Guard, end) {
        if next.State == MEM_FREE || next.AllocationBase as usize != base || next.RegionSize == 0 {
            break;
        }
        end = (next.BaseAddress as usize).saturating_add(next.RegionSize);
    }
    (base, end)
}

/// The title's `VirtualFree`: a release or decommit ends our guards there
/// with it, under the guards' lock, so that no release of ours restores a
/// protection over memory allocated there afterwards.
///
/// # Safety
///
/// As `VirtualFree`'s.
pub unsafe fn title_virtual_free(at: *mut std::ffi::c_void, len: usize, kind: u32) -> i32 {
    use windows_sys::Win32::System::Memory::{MEM_DECOMMIT, MEM_RELEASE, VirtualFree};
    if !crate::ledger::on() {
        // SAFETY: the title's arguments, as the host takes them.
        return unsafe { VirtualFree(at, len, kind) };
    }
    let mut g = guards();
    let (start, end) = {
        let _error = LastError::kept();
        extent(at as usize, len)
    };
    // SAFETY: as above.
    let done = unsafe { VirtualFree(at, len, kind) };
    if done != 0 && kind & (MEM_RELEASE | MEM_DECOMMIT) != 0 {
        forget(&mut g, start, end - start);
    }
    done
}

/// The title's `VirtualAlloc`: committing pages already committed sets
/// their protection, as `VirtualProtect` does, so our guards there are laid
/// again over it ([`reguard`]).
///
/// # Safety
///
/// As `VirtualAlloc`'s.
pub unsafe fn title_virtual_alloc(
    at: *const std::ffi::c_void,
    len: usize,
    kind: u32,
    protect: u32,
) -> *mut std::ffi::c_void {
    use crate::ledger::NATIVE;
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, VirtualAlloc};
    if !crate::ledger::on() {
        // SAFETY: the title's arguments, as the host takes them.
        return unsafe { VirtualAlloc(at, len, kind, protect) };
    }
    let mut g = guards();
    // SAFETY: as above.
    let p = unsafe { VirtualAlloc(at, len, kind, protect) };
    if !p.is_null() && !at.is_null() && kind & MEM_COMMIT != 0 && !g.ledger.is_empty() {
        let _error = LastError::kept();
        let start = at as usize / NATIVE * NATIVE;
        let end = (at as usize).saturating_add(len).div_ceil(NATIVE).saturating_mul(NATIVE);
        reguard(&mut g, start, end, protect);
    }
    p
}

/// The title's `UnmapViewOfFile`: the view's guards end with it, under the
/// guards' lock.
///
/// # Safety
///
/// As `UnmapViewOfFile`'s.
pub unsafe fn title_unmap_view(at: *const std::ffi::c_void) -> i32 {
    use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
    let view = MEMORY_MAPPED_VIEW_ADDRESS { Value: at as *mut _ };
    if !crate::ledger::on() {
        // SAFETY: the title's argument, as the host takes it.
        return unsafe { UnmapViewOfFile(view) };
    }
    let mut g = guards();
    let (start, end) = {
        let _error = LastError::kept();
        extent(at as usize, 0)
    };
    // SAFETY: as above.
    let done = unsafe { UnmapViewOfFile(view) };
    if done != 0 {
        forget(&mut g, start, end - start);
    }
    done
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
pub(crate) fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // The user copy's leaf faulted: to its exit, and nothing else (no lock, no call).
        // Any other fault goes on below as before.
        // SAFETY: the system passes the exception being dispatched.
        if unsafe { crate::safecopy::redirect(info) } {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if crate::ledger::on() {
            return match g.ledger.at(at) {
                // Our guard over a page the title may write: the title's
                // write, noted, and let through.
                Some((_, run)) if crate::ledger::writable(run.requested) => {
                    open(&mut g, at, at + 1);
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::GuardFaults, 1);
                    EXCEPTION_CONTINUE_EXECUTION
                }
                // The title's own protection refuses the write: its fault,
                // for its handlers, and the guard stays.
                Some(_) => EXCEPTION_CONTINUE_SEARCH,
                // Opened by another thread since this one faulted: again.
                None if query(crate::kernelstat::Place::Guard, at).is_some_and(|info| {
                    info.State == windows_sys::Win32::System::Memory::MEM_COMMIT
                        && crate::ledger::writable(info.Protect)
                }) =>
                {
                    EXCEPTION_CONTINUE_EXECUTION
                }
                None => EXCEPTION_CONTINUE_SEARCH,
            };
        }
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::add(crate::copystat::C::GuardFaults, 1);
                crate::copystat::add(crate::copystat::C::Guards, 1);
                crate::copystat::add(
                    crate::copystat::C::GuardUs,
                    guarding.elapsed().as_micros() as u64,
                );
            }
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    #[cfg(feature = "diagnostics")]
    crate::stalls::kernel(crate::stalls::Kernel::Query);
    let n = crate::kernelstat::virtual_query(crate::kernelstat::Place::Guard, at, &mut info);
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::journal::{AHEAD_GAP, JOURNAL, ask_rest};
    #[cfg(feature = "diagnostics")]
    use super::journal::writes_locked;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (first, last) = (p / PAGE, (p + 4 * PAGE).div_ceil(PAGE));
        // The journal keeps a page's last write across allocations, so a
        // freed allocation of an earlier test can answer here: these pages
        // start clean, as the guard tests below clean theirs.
        JOURNAL.lock().unwrap_or_else(|e| e.into_inner()).written.retain(|&at, _| !(first..last).contains(&at));
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // The journal keeps pages written since tracking began: these must
        // not answer for memory another test allocates here.
        JOURNAL.lock().unwrap_or_else(|e| e.into_inner()).written.retain(|&at, _| !(first..last).contains(&at));
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// The kernel-call counters see the write tracking's asking: one
    /// write-watch call, and the 4 KiB pages it asked about.
    #[test]
    fn a_write_watch_asking_is_counted_with_its_pages() {
        use crate::kernelstat::here;
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe { VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as usize;
        assert_ne!(p, 0);
        let (first, last) = (p / PAGE, p / PAGE + 2);
        let (calls, asked) = (here("write_watch", "calls"), here("write_watch", "pages_asked"));
        writes(p, p + 2 * PAGE, false).expect("watched");
        assert_eq!(here("write_watch", "calls") - calls, 1);
        assert_eq!(here("write_watch", "pages_asked") - asked, (2 * PAGE / KERNEL_PAGE) as u64);
        JOURNAL.lock().unwrap_or_else(|e| e.into_inner()).written.retain(|&at, _| !(first..last).contains(&at));
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// The default guard laying (the ledger off) is counted: one protect
    /// under the guards' place for the two blocks it makes read-only.
    #[test]
    fn the_default_guard_laying_is_counted() {
        use crate::kernelstat::here;
        if crate::ledger::on() {
            return; // the ledger lays guards by its own route
        }
        // SAFETY: a fresh allocation of two blocks, freed at the end.
        let p = unsafe { VirtualAlloc(std::ptr::null(), 2 * BLOCK, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(p, 0);
        let before = here("protect", "guard");
        protect_here(&mut guards(), p / BLOCK, p / BLOCK + 2);
        assert_eq!(here("protect", "guard") - before, 1, "one call for the two blocks");
        // The guards are told before the memory goes (as the view tests do).
        title_memory_remapped(p, 2 * BLOCK);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A pending read's status, read through the kernel, is counted under a
    /// caller of its own.
    #[test]
    fn an_io_status_read_is_counted_under_its_own_caller() {
        use crate::kernelstat::here;
        let status = 0usize;
        let before = here("read", "io_status");
        assert!(io_done(&raw const status as usize), "0 is final");
        assert_eq!(here("read", "io_status") - before, 1);
    }

    /// The title's own protect and query, which the runtime makes for it,
    /// are counted under `title`, not under the guards.
    #[test]
    fn the_titles_own_protection_calls_are_counted_under_title() {
        use crate::kernelstat::here;
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, PAGE_READONLY};
        if crate::ledger::on() {
            return;
        }
        // SAFETY: a fresh page, freed at the end.
        let p = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!p.is_null());
        let counts = || (here("protect", "title"), here("protect", "guard"), here("query", "title"), here("query", "guard"));
        let before = counts();
        let mut old = 0;
        // SAFETY: our page, and room for the answers.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        unsafe {
            assert_ne!(title_virtual_protect(p, 4096, PAGE_READONLY, &mut old), 0);
            assert_ne!(title_virtual_query(p, &mut info, size_of_val(&info)), 0);
        }
        assert_eq!(info.Protect, PAGE_READONLY);
        let after = counts();
        assert_eq!((after.0 - before.0, after.1 - before.1, after.2 - before.2, after.3 - before.3), (1, 0, 1, 0));
        // SAFETY: our page.
        unsafe { VirtualFree(p, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // The guards are told before the memory goes: another test may map
        // memory at these addresses as soon as it has, and its blocks must not
        // find ours there (nor have ours cleared from under it).
        title_memory_remapped(view, 2 * BLOCK);
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
    }

    /// A store of the kernel's into a block the guards hold read-only is
    /// refused; its retries by 64 KiB and 4 KiB pieces all fail and are
    /// dropped. Each dropped piece is counted with its bytes (in every
    /// build: `crate::storestat`), and the title's bytes there stay what
    /// they were: the drop is no longer silent.
    #[test]
    fn a_store_the_kernel_refuses_is_counted_and_its_bytes_stay() {
        const SIZE: usize = 2 * BLOCK;
        let (section, view, _other) = two_views(SIZE);
        let pages = |block: usize| (view + block * BLOCK) / PAGE;
        // The title's bytes there, before the store and before the guard
        // (a write through the guard would open it).
        let before: Vec<u8> = (0..BLOCK).map(|i| (i * 29 + 7) as u8).collect();
        // SAFETY: inside the first view, read-write until the guard below.
        unsafe { std::ptr::copy_nonoverlapping(before.as_ptr(), view as *mut u8, BLOCK) };
        assert_eq!(guard(view, view + SIZE), Some(vec![pages(0), pages(1)]), "both blocks read-only");
        let src = vec![0xa5u8; BLOCK];
        let (stores, bytes) = crate::storestat::counted();
        // No opening here: the block is read-only, as a guard re-armed
        // between an opening and the store leaves it.
        crate::host::write_mapped(view as *mut u8, src.as_ptr(), BLOCK);
        let (stores, bytes) = (crate::storestat::counted().0 - stores, crate::storestat::counted().1 - bytes);
        assert_eq!(stores, (BLOCK / 4096) as u64, "every 4 KiB piece that still failed is counted");
        assert_eq!(bytes, BLOCK as u64, "and its bytes");
        // SAFETY: the first view, still holding the title's bytes.
        let landed = unsafe { std::slice::from_raw_parts(view as *const u8, BLOCK) };
        assert_eq!(landed, &before[..], "the refused store changed nothing");
        drop_views(section, &[view, _other], SIZE);
    }

    /// The memory census tells the writers apart: a landing's store is the
    /// card's (marked where it landed, `crate::host::write_mapped`), a
    /// storage read's bytes are storage's ([`KernelWrite::storage`]), and a
    /// stretch whose store the kernel refused is attributed to nobody. The
    /// generic end of a kernel write marked every stretch storage's, a
    /// landing's failed ones included, and masked the card's.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_landing_is_the_cards_a_storage_read_is_storages_and_a_refused_store_is_neither() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        const SIZE: usize = 2 * BLOCK;
        let (section, view, other) = two_views(SIZE);
        let census = crate::memclass::TestCensus::new();
        // A landing's store: the card's, not storage's.
        let landing = kernel_writing(view + 100, 13);
        crate::host::write_mapped((view + 100) as *mut u8, b"landed bytes".as_ptr(), 13);
        drop(landing);
        crate::memclass::looked(view, view + SIZE, &[]);
        assert_eq!(census.fills(view), [0, 0, 1], "a landing's bytes are the card's, not storage's");
        // A storage read's bytes: storage's.
        let mut read = kernel_writing(view + 200, 13);
        read.storage(13);
        drop(read);
        crate::memclass::looked(view, view + SIZE, &[]);
        assert_eq!(census.fills(view), [1, 0, 1], "a storage read's bytes are storage's");
        // A landing whose page went while the write was under way: every
        // piece refused, and nothing attributed.
        // SAFETY: a fresh reservation of ours, released below (and here).
        let at = unsafe { VirtualAlloc(std::ptr::null(), BLOCK, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(at, 0);
        let refused = kernel_writing(at, 13);
        // SAFETY: the reservation just made.
        assert_eq!(unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) }, 1, "the page went");
        let before = crate::storestat::counted();
        crate::host::write_mapped(at as *mut u8, b"refused bytes".as_ptr(), 13);
        assert!(crate::storestat::counted().0 > before.0, "the store was refused");
        drop(refused);
        crate::memclass::looked(at, at + PAGE, &[]);
        assert_eq!(census.fills(at), [0, 0, 0], "a refused store is not attributed as written");
        drop_views(section, &[view, other], SIZE);
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        // The guards are told before the memory goes: another test may map
        // memory at these addresses as soon as it has, and its blocks must not
        // find ours there (nor have ours cleared from under it).
        title_memory_remapped(p, 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // The guards are told before the memory goes: another test may map
        // memory at these addresses as soon as it has, and its blocks must not
        // find ours there (nor have ours cleared from under it).
        title_memory_remapped(p, 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// A pagefile section mapped twice: two views of the same physical
    /// pages, the second left for the test; the section handle, the two
    /// addresses.
    pub(super) fn two_views(bytes: usize) -> (usize, usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a fresh pagefile section, mapped twice; the views keep
        // the section alive and the handle is closed by the caller.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(one, 0);
            assert_ne!(two, 0);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The section of [`two_views`] is closed and its views unmapped; the
    /// guard is told they are gone first.
    pub(super) fn drop_views(section: usize, views: &[usize], bytes: usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, bytes);
            // SAFETY: a view of the section mapped by `two_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
        // SAFETY: the section's last reference.
        unsafe { CloseHandle(section as *mut std::ffi::c_void) };
    }

    /// Receipts' deadlock chain (review finding 1): a check of guarded view memory holds
    /// the journal (`writes`), and its guard settles a pending kernel read that has ended
    /// (`guard` → `kernel_writes_settled` → `landed`), whose publish must not take the
    /// journal again. The check returns, and the read's pages, at its address and at the
    /// other view, are stamped above the number before it and at or below the one the
    /// check answers with.
    #[test]
    fn a_read_settled_inside_a_guarded_check_is_stamped_in_the_journal_held() {
        crate::receipts::child("watch::tests::settled_inside_a_check_child");
    }

    #[test]
    #[ignore = "run by its parent, with receipts on"]
    fn settled_inside_a_check_child() {
        assert!(crate::receipts::on(), "run by its parent");
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        assert!(writes(guarded, guarded + SIZE, true).is_some(), "view memory, guarded now");
        let before = top();
        let status = std::sync::atomic::AtomicUsize::new(0x103);
        kernel_writing(other + 100, 13).pending((&raw const status).cast());
        // The read is done: the next check of guarded memory settles it.
        status.store(0, std::sync::atomic::Ordering::SeqCst);
        let (now, _, answered) = writes_top(guarded, guarded + SIZE, true).expect("guarded");
        let stamp = |at: usize| recorded(at, at + 1).first().map(|&(_, stamp)| stamp);
        for at in [other + 100, guarded + 100] {
            let stamp = stamp(at).expect("the read's page is stamped");
            assert!(stamp > before, "above the number before the read ended: {stamp} > {before}");
            assert!(stamp <= now && stamp <= answered, "in the journal the check held: {stamp} <= {now}");
        }
        drop_views(section, &[guarded, other], SIZE);
    }

    /// The texture pass's number for a texture it finds unmapped is read while another
    /// thread holds the journal, without waiting, and is no higher than the journal's newest.
    #[test]
    fn the_absent_number_is_read_without_waiting_for_the_journal() {
        crate::receipts::child("watch::tests::absent_number_child");
    }

    #[test]
    #[ignore = "run by its parent, with receipts on"]
    fn absent_number_child() {
        assert!(crate::receipts::on(), "run by its parent");
        publish(PAGE, 2 * PAGE);
        let newest = top();
        assert!(newest > 0, "the publish is stamped");
        let held = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
        let (tell, told) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tell.send(absent_number());
        });
        let got = told.recv_timeout(std::time::Duration::from_secs(2));
        drop(held);
        let got = got.expect("the number, while another thread holds the journal");
        assert!(got > 0 && got <= newest, "{got} against the newest, {newest}");
    }

    /// A number drawn between `ask_once`'s two takings of the journal is a
    /// copy's mark or a receipt: here a refresh's own asking of a range the
    /// first taking left alone; with receipts, any thread's `top` (a copy
    /// recorded on a title thread takes one: `list.rs`'s copy command,
    /// `texture::host_written`). A title write the second taking finds is
    /// stamped above it, or the copy keeps the old bytes and nothing says
    /// so (the N1 note's F1).
    #[test]
    fn a_write_the_second_taking_finds_is_stamped_above_a_number_drawn_between() {
        crate::receipts::child("watch::tests::second_taking_child");
    }

    #[test]
    #[ignore = "run by its parent, with receipts on"]
    fn second_taking_child() {
        assert!(crate::receipts::on(), "run by its parent");
        const SIZE: usize = 16 << 20;
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe { VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) }
            as usize;
        assert_ne!(p, 0);
        let at = |page: usize| p + page * PAGE;
        // X1 and X2 close together, asked about in one asking ahead; R more
        // than `AHEAD_GAP` away, alone, left to the second taking.
        let (x1, x2, r) = ((at(0), at(1)), (at(2), at(3)), (at(200), at(204)));
        const { assert!(204 <= SIZE / PAGE && 200 > 3 + AHEAD_GAP) };
        let ranges = [x1, x2, r];
        ask_once_end();
        ask_ahead(&ranges);
        assert!(asked(x1.0, x2.1).is_some(), "X1 and X2 asked about together");
        assert!(asked(r.0, r.1).is_none(), "R left alone");
        // Between the takings: a copy of R asks about it and takes the
        // number as its mark, then reads R.
        let (mark, written) = writes(r.0, r.1, false).expect("watched");
        assert!(written.is_empty(), "R not written yet");
        // The title writes R after that read.
        // SAFETY: committed pages of ours.
        unsafe { ((at(201) + 8) as *mut u8).write(1) };
        ask_rest(&ranges);
        let stamp = recorded(at(201), at(201) + 1).first().map(|&(_, stamp)| stamp).expect("the write is found");
        assert!(stamp > mark, "the write after the copy's read is stamped above its mark: {stamp} > {mark}");
        assert_eq!(fresh_all(&[(r.0, r.1, mark)]), [false], "a read at {mark} is not fresh");
        ask_once_end();
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A number only read between `ask_once`'s two takings is a receipt
    /// too: with nothing published, a title thread's `top` (`list.rs`'s copy
    /// command, `texture::host_written`) draws nothing and hands out the
    /// first taking's number. A title write the second taking finds is
    /// stamped above it, or the receipt calls the memory fresh. A second
    /// taking that draws only when someone drew since the first passes
    /// [`second_taking_child`] and fails here.
    #[test]
    fn a_write_the_second_taking_finds_is_stamped_above_a_number_read_between() {
        crate::receipts::child("watch::tests::second_taking_read_child");
    }

    #[test]
    #[ignore = "run by its parent, with receipts on"]
    fn second_taking_read_child() {
        assert!(crate::receipts::on(), "run by its parent");
        const SIZE: usize = 16 << 20;
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe { VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) }
            as usize;
        assert_ne!(p, 0);
        let at = |page: usize| p + page * PAGE;
        // As in `second_taking_child`: X1 and X2 asked about ahead, R alone.
        let (x1, x2, r) = ((at(0), at(1)), (at(2), at(3)), (at(200), at(204)));
        const { assert!(204 <= SIZE / PAGE && 200 > 3 + AHEAD_GAP) };
        let ranges = [x1, x2, r];
        ask_once_end();
        ask_ahead(&ranges);
        let (first, _) = asked(x1.0, x2.1).expect("X1 and X2 asked about together");
        assert!(asked(r.0, r.1).is_none(), "R left alone");
        // Between the takings: a receipt, nothing published, so nothing drawn.
        let t = top();
        assert_eq!((t, top()), (first, first), "top read the first taking's number and drew none");
        // The title writes R after the read the receipt stands for.
        // SAFETY: committed pages of ours.
        unsafe { ((at(201) + 8) as *mut u8).write(1) };
        ask_rest(&ranges);
        let stamp = recorded(at(201), at(201) + 1).first().map(|&(_, stamp)| stamp).expect("the write is found");
        assert!(stamp > t, "the write after the receipt is stamped above it: {stamp} > {t}");
        assert_eq!(fresh_all(&[(r.0, r.1, t)]), [false], "a receipt at {t} is not fresh");
        ask_once_end();
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// The title wrote the pages of a guarded texture through another view
    /// of them: the fault at that view is the guard's to see, and the
    /// guarded view's next check reports the pages written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        // The first check takes it in (and every view of its pages).
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        // Writing through the other view faults into the handler, which
        // opens that view and the guarded one: the next check sees it.
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A range that begins in a block the guards made read-only is view
    /// memory, which Windows cannot write-watch: checking it again must not
    /// ask the kernel again (the ask fails, and the guards answer).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_guarded_view_range_is_not_asked_about_again() {
        use crate::copystat::{C, count};
        const SIZE: usize = 2 * BLOCK;
        if std::env::var_os("VICTORY_VIEW_ASK").is_some_and(|v| v == "1") {
            return; // the switch asks about view memory, as before
        }
        let (section, view, other) = two_views(SIZE);
        assert_eq!(guard(view, view + SIZE), Some(vec![view / PAGE, view / PAGE + 1]));
        // Every kernel asking is made under the journal's lock: held here,
        // no other test's failing asking can move the counter between the
        // checks.
        let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
        let before = count(C::AskFailed);
        assert!(writes_locked(&mut j, view, view + SIZE, true).is_some());
        let first = count(C::AskFailed);
        assert!(writes_locked(&mut j, view, view + SIZE, true).is_some());
        let second = count(C::AskFailed);
        drop(j);
        assert_eq!(second, first, "the second check of a guarded view is not asked about");
        assert_eq!(first, before, "a guarded view is answered by the guards only");
        drop_views(section, &[view, other], SIZE);
    }

    /// The runtime announces a view anew: the blocks are no longer taken
    /// for ones we made read-only, so `writes` asks about the range again
    /// (the ask fails, and the guards answer). Write-watched memory is asked
    /// about all the same, and its written pages are found.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_remapped_view_is_asked_about_again() {
        use crate::copystat::{C, count};
        const SIZE: usize = 2 * BLOCK;
        let (section, view, other) = two_views(SIZE);
        assert!(guard(view, view + SIZE).is_some());
        title_memory_remapped(view, SIZE);
        let before = count(C::AskFailed);
        assert!(writes(view, view + SIZE, true).is_some());
        assert!(count(C::AskFailed) > before, "a remapped view is asked about");
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        // SAFETY: a page of ours.
        unsafe { ((p + 8) as *mut u8).write(1) };
        let (now, written) = writes(p, p + PAGE, true).expect("write-watched");
        assert_eq!(written, [(p / PAGE, now)]);
        // The journal keeps pages written since tracking began: this one
        // must not answer for memory another test allocates here.
        JOURNAL.lock().unwrap_or_else(|e| e.into_inner()).written.remove(&(p / PAGE));
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
        drop_views(section, &[view, other], SIZE);
    }

    /// A block the guards made read-only at one of the title's own views,
    /// unmapped without the runtime's telling (`title_memory_remapped`):
    /// write-watched memory at the address is asked about, not answered
    /// clean from a guard that no longer covers it.
    #[test]
    fn write_watched_memory_reused_at_a_guarded_view_is_asked_about() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            UnmapViewOfFile,
        };
        const SIZE: usize = BLOCK;
        // SAFETY: a fresh pagefile section of one block, mapped once.
        let (section, view) = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE).Value as usize;
            assert_ne!(view, 0);
            (section, view)
        };
        assert!(guard(view, view + SIZE).is_some());
        // The title's own view goes, and no runtime call says so.
        // SAFETY: our view of the section.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            })
        };
        // Write-watched memory takes the address and is written.
        // SAFETY: the address is free; the allocation is freed at the end.
        let p = unsafe {
            VirtualAlloc(
                view as *const _,
                SIZE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_eq!(p, view, "the write-watched memory takes the view's address");
        // SAFETY: a page of ours.
        unsafe { ((p + 8) as *mut u8).write_volatile(1) };
        let (now, written) = writes(p, p + SIZE, true).expect("write-watched");
        assert_eq!(written, [(p / PAGE, now)]);
        // The journal keeps pages written since tracking began: this one
        // must not answer for memory another test allocates here.
        JOURNAL.lock().unwrap_or_else(|e| e.into_inner()).written.remove(&(p / PAGE));
        title_memory_remapped(p, SIZE);
        // SAFETY: our allocation and the section's handle.
        unsafe {
            VirtualFree(p as *mut _, 0, MEM_RELEASE);
            CloseHandle(section);
        }
    }

    /// A view of guarded pages is mapped anew: it is writable, so the
    /// guarded view checks its pages again (and guards the new view).
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        title_memory_remapped(other, SIZE);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A write of the runtime's own, told at one view of the pages
    /// (`writing`): the other views' next checks report the pages written.
    #[test]
    fn a_write_told_at_one_view_is_reported_at_the_others() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        writing(other + BLOCK, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Half a page and the next: both blocks, not just the first's.
        writing(other + BLOCK / 2, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// Both views of the pages are checked, each for a copy of its own
    /// (the title shows one physical page at several addresses, and each
    /// address has its host copy): a write through either is reported at
    /// both, whichever is checked first. Guarding the pages for the first
    /// view's check must not note the second view clean, or its copy would
    /// keep the bytes from before the write.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0), pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0), pages(two, 1)]), "the second view's first check");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()), "nothing written yet");
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()), "nothing written yet");
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((two + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 1)]), "written through this view, checked second");
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0)]));
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0)]), "written through this view, checked second");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        drop_views(section, &[one, two], SIZE);
    }

    /// A pipe whose reads stay pending until its server end writes: the
    /// server's handle and the client's, opened for overlapped reads.
    fn pending_pipe() -> (usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        // One name a pipe in the process: the clock alone can give two tests
        // starting together the same one.
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let name: Vec<u16> = format!(r"\\.\pipe\victory-watch-{}-{nanos}-{n}", std::process::id())
            .encode_utf16()
            .chain([0])
            .collect();
        const PIPE_ACCESS_OUTBOUND: u32 = 2;
        const GENERIC_READ: u32 = 0x8000_0000;
        const OPEN_EXISTING: u32 = 3;
        const FILE_FLAG_OVERLAPPED: u32 = 0x4000_0000;
        // SAFETY: a NUL-terminated name; plain arguments.
        unsafe {
            let (out, null) = (PIPE_ACCESS_OUTBOUND, std::ptr::null());
            let server = pipe::CreateNamedPipeW(name.as_ptr(), out, 0, 1, 4096, 4096, 0, null);
            assert_ne!(server, INVALID_HANDLE_VALUE);
            let client = pipe::CreateFileW(
                name.as_ptr(),
                GENERIC_READ,
                0,
                std::ptr::null(),
                OPEN_EXISTING,
                FILE_FLAG_OVERLAPPED,
                std::ptr::null_mut(),
            );
            assert_ne!(client, INVALID_HANDLE_VALUE);
            (server as usize, client as usize)
        }
    }

    /// What the pipe fixture calls of kernel32 (features this crate does not
    /// otherwise need).
    mod pipe {
        use std::ffi::c_void;
        pub(super) type Handle = *mut c_void;
        /// `OVERLAPPED`.
        #[repr(C)]
        pub(super) struct Overlapped {
            pub(super) internal: usize,
            pub(super) internal_high: usize,
            pub(super) offset: u64,
            pub(super) event: Handle,
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            pub(super) fn CreateNamedPipeW(
                name: *const u16,
                open: u32,
                mode: u32,
                instances: u32,
                out: u32,
                into: u32,
                timeout: u32,
                security: *const c_void,
            ) -> Handle;
            pub(super) fn CreateFileW(
                name: *const u16,
                access: u32,
                share: u32,
                security: *const c_void,
                disposition: u32,
                flags: u32,
                template: Handle,
            ) -> Handle;
            pub(super) fn ReadFile(file: Handle, at: *mut u8, len: u32, n: *mut u32, o: *mut Overlapped) -> i32;
            pub(super) fn WriteFile(file: Handle, at: *const u8, len: u32, n: *mut u32, o: *mut Overlapped) -> i32;
            pub(super) fn GetOverlappedResult(file: Handle, o: *const Overlapped, n: *mut u32, wait: i32) -> i32;
            pub(super) fn CancelIoEx(file: Handle, o: *const Overlapped) -> i32;
        }
    }

    /// A read of the kernel's left pending by its call (an overlapped
    /// `ReadFile`), a copy of another view of its pages taken meanwhile, then
    /// the read completes and the title learns so: until then that view is
    /// reported written at every check, and after, its copy is taken again,
    /// with the bytes the read brought.
    #[test]
    fn a_pending_kernel_read_keeps_the_other_views_changed_until_it_completes() {
        use windows_sys::Win32::Foundation::{CloseHandle, GetLastError};
        const SIZE: usize = 2 * BLOCK;
        let (section, other, dest) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(other, other + SIZE), Some(vec![pages(other, 0), pages(other, 1)]));
        assert_eq!(guard(other, other + SIZE), Some(Vec::new()), "nothing written yet");
        let (server, client) = pending_pipe();
        let mut o = pipe::Overlapped { internal: 0, internal_high: 0, offset: 0, event: std::ptr::null_mut() };
        let write = kernel_writing(dest + 100, 13);
        // SAFETY: 13 bytes of the destination view, kept until the read is done.
        let started = unsafe { pipe::ReadFile(client as _, (dest + 100) as *mut u8, 13, std::ptr::null_mut(), &mut o) };
        // SAFETY: no arguments.
        assert_eq!((started, unsafe { GetLastError() }), (0, 997), "left pending");
        write.pending((&raw const o).cast());
        // A copy of the other view while the read is pending.
        assert_eq!(guard(other, other + SIZE), Some(vec![pages(other, 0)]));
        assert_eq!(guard(other, other + SIZE), Some(vec![pages(other, 0)]), "still changed while pending");
        // The server writes: the read completes, and the title learns so.
        let mut n = 0;
        // SAFETY: 13 bytes of ours; a synchronous write.
        let wrote =
            unsafe { pipe::WriteFile(server as _, b"from the pipe".as_ptr(), 13, &mut n, std::ptr::null_mut()) };
        assert_eq!((wrote, n), (1, 13));
        // SAFETY: the read's `OVERLAPPED`; waits for it.
        assert_eq!(unsafe { pipe::GetOverlappedResult(client as _, &o, &mut n, 1) }, 1);
        assert_eq!(n, 13);
        // The other view's copy is taken again, with the read's bytes.
        assert_eq!(guard(other, other + SIZE), Some(vec![pages(other, 0)]), "taken after the read landed");
        // SAFETY: the bytes just read, through the other view.
        let seen = unsafe { std::slice::from_raw_parts((other + 100) as *const u8, 13) };
        assert_eq!(seen, b"from the pipe");
        assert_eq!(guard(other, other + SIZE), Some(Vec::new()), "settled once done");
        // SAFETY: our pipe's handles.
        unsafe {
            CloseHandle(client as _);
            CloseHandle(server as _);
        }
        drop_views(section, &[other, dest], SIZE);
    }

    /// A pending read that brings less than it asked for, and one cancelled
    /// before anything came: each ends with its I/O, the other view's copy
    /// taken again once, then settled.
    #[test]
    fn a_short_or_cancelled_pending_read_ends_with_its_io() {
        use windows_sys::Win32::Foundation::{CloseHandle, GetLastError};
        const SIZE: usize = 2 * BLOCK;
        let (section, other, dest) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(other, other + SIZE), Some(vec![pages(other, 0), pages(other, 1)]));
        assert_eq!(guard(other, other + SIZE), Some(Vec::new()), "nothing written yet");
        let (server, client) = pending_pipe();
        let read = |at: usize, o: &mut pipe::Overlapped| {
            let write = kernel_writing(at, 13);
            // SAFETY: 13 bytes of the destination view, kept until the read is done.
            let started = unsafe { pipe::ReadFile(client as _, at as *mut u8, 13, std::ptr::null_mut(), o) };
            // SAFETY: no arguments.
            assert_eq!((started, unsafe { GetLastError() }), (0, 997), "left pending");
            write.pending((&raw const *o).cast());
        };
        let overlapped = || pipe::Overlapped { internal: 0, internal_high: 0, offset: 0, event: std::ptr::null_mut() };
        let (mut n, mut short) = (0, overlapped());
        read(dest + 100, &mut short);
        // SAFETY: 4 bytes of ours; a synchronous write.
        assert_eq!(unsafe { pipe::WriteFile(server as _, b"part".as_ptr(), 4, &mut n, std::ptr::null_mut()) }, 1);
        // SAFETY: the read's `OVERLAPPED`; waits for it.
        assert_eq!(unsafe { pipe::GetOverlappedResult(client as _, &short, &mut n, 1) }, 1);
        assert_eq!(n, 4, "short");
        assert_eq!(guard(other, other + SIZE), Some(vec![pages(other, 0)]), "taken after the read landed");
        // SAFETY: the bytes just read, through the other view.
        assert_eq!(unsafe { std::slice::from_raw_parts((other + 100) as *const u8, 4) }, b"part");
        assert_eq!(guard(other, other + SIZE), Some(Vec::new()), "settled once done");

        let mut cancelled = overlapped();
        read(dest + 200, &mut cancelled);
        assert_eq!(guard(other, other + SIZE), Some(vec![pages(other, 0)]));
        // SAFETY: the pending read's `OVERLAPPED`.
        assert_eq!(unsafe { pipe::CancelIoEx(client as _, &cancelled) }, 1);
        // SAFETY: as above; waits for it.
        assert_eq!(unsafe { pipe::GetOverlappedResult(client as _, &cancelled, &mut n, 1) }, 0);
        // SAFETY: no arguments.
        assert_eq!(unsafe { GetLastError() }, 995, "ERROR_OPERATION_ABORTED");
        assert_eq!(guard(other, other + SIZE), Some(vec![pages(other, 0)]), "ended with its I/O");
        assert_eq!(guard(other, other + SIZE), Some(Vec::new()), "settled once done");
        // SAFETY: our pipe's handles.
        unsafe {
            CloseHandle(client as _);
            CloseHandle(server as _);
        }
        drop_views(section, &[other, dest], SIZE);
    }

    /// A pending read whose status cannot be read for a while (its page not
    /// there): it is not taken for done, and the other view stays changed;
    /// once the status reads final, the write ends.
    #[test]
    fn a_status_that_cannot_be_read_keeps_the_write_under_way() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const SIZE: usize = 2 * BLOCK;
        let (section, other, dest) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(other, other + SIZE), Some(vec![pages(other, 0), pages(other, 1)]));
        assert_eq!(guard(other, other + SIZE), Some(Vec::new()), "nothing written yet");
        // SAFETY: a fresh reservation, nothing committed: reading it fails.
        let status = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE, PAGE_READWRITE) };
        assert!(!status.is_null());
        kernel_writing(dest + 100, 13).pending(status.cast_const());
        assert_eq!(guard(other, other + SIZE), Some(vec![pages(other, 0)]));
        assert_eq!(guard(other, other + SIZE), Some(vec![pages(other, 0)]), "still changed, status unread");
        // The status is there, and final.
        // SAFETY: committing the reserved page; it reads 0 (success).
        assert!(!unsafe { VirtualAlloc(status, 4096, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        assert_eq!(guard(other, other + SIZE), Some(vec![pages(other, 0)]), "ended");
        assert_eq!(guard(other, other + SIZE), Some(Vec::new()), "settled once done");
        // SAFETY: our reservation.
        unsafe { VirtualFree(status, 0, MEM_RELEASE) };
        drop_views(section, &[other, dest], SIZE);
    }

    /// Whatever the end of a kernel write calls of the host, the thread's
    /// last error is the one the write's call left (the title reads it next,
    /// `ERROR_IO_PENDING` say): here a block it must open has gone.
    #[test]
    fn the_end_of_a_kernel_write_keeps_the_last_error() {
        use windows_sys::Win32::Foundation::{GetLastError, SetLastError};
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        // SAFETY: a fresh reservation, nothing committed: a protection
        // change there fails.
        let at = unsafe { VirtualAlloc(std::ptr::null(), 2 * BLOCK, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(at, 0);
        assert_eq!(at % BLOCK, 0, "reservations start at 64 KiB");
        let write = kernel_writing(at, 100);
        // Guarded meanwhile, and gone since.
        guards().protected.insert(at / BLOCK);
        // SAFETY: plain calls.
        unsafe { SetLastError(997) };
        drop(write);
        // SAFETY: no arguments.
        assert_eq!(unsafe { GetLastError() }, 997);
        assert!(!guards().protected.contains(&(at / BLOCK)));
        // SAFETY: our reservation.
        unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) };
    }

    /// The runtime opens a block for a read of the kernel's into it (the
    /// title's `ReadFile`), and before the kernel writes, a copy of another
    /// view of the same pages takes them and guards them again, with every
    /// view of them: the read lands all the same, and that view's copy,
    /// taken before it did, is told the pages were written.
    #[test]
    fn a_kernel_read_lands_though_another_view_is_guarded_meanwhile() {
        use std::io::{Read, Seek, Write};
        const SIZE: usize = 2 * BLOCK;
        let (section, other, dest) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(other, other + SIZE), Some(vec![pages(other, 0), pages(other, 1)]));
        assert_eq!(guard(other, other + SIZE), Some(Vec::new()), "nothing written yet");
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        file.rewind().unwrap();
        // The runtime's `ReadFile`: the destination is opened for the kernel.
        let write = kernel_writing(dest + 100, 13);
        // A copy of the other view meanwhile: the pages were opened, so it
        // takes them and guards every view of them again.
        assert_eq!(guard(other, other + SIZE), Some(vec![pages(other, 0)]));
        // And a copy of the destination's view itself.
        assert_eq!(guard(dest, dest + SIZE), Some(vec![pages(dest, 0), pages(dest, 1)]));
        // SAFETY: 13 bytes of the destination view, nothing else refers to them.
        let into = unsafe { std::slice::from_raw_parts_mut((dest + 100) as *mut u8, 13) };
        assert_eq!(file.read(into).map_err(|e| e.raw_os_error()), Ok(13));
        drop(write);
        // SAFETY: the bytes just read, through the other view.
        let seen = unsafe { std::slice::from_raw_parts((other + 100) as *const u8, 13) };
        assert_eq!(seen, b"from the disk");
        let before = "taken before the read landed";
        assert_eq!(guard(other, other + SIZE), Some(vec![pages(other, 0)]), "{before}");
        assert_eq!(guard(other, other + SIZE), Some(Vec::new()));
        assert_eq!(guard(dest, dest + SIZE), Some(vec![pages(dest, 0)]), "{before}");
        assert_eq!(guard(dest, dest + SIZE), Some(Vec::new()));
        drop_views(section, &[other, dest], SIZE);
    }

    /// A page shown in more than `MANY_VIEWS` views is guarded at each by
    /// itself: the others stay writable, and nothing looks through them all.
    #[test]
    fn a_page_of_many_views_is_guarded_one_view_at_a_time() {
        let (section, one, two) = two_views(BLOCK);
        let filler = one + (1 << 45);
        for n in 0..MANY_VIEWS {
            crate::inplace::title_memory_view(filler + n * BLOCK, BLOCK, section, 0);
        }
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        assert!(writable(two), "another view of it is left alone");
        // SAFETY: the guarded view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        title_memory_remapped(filler, MANY_VIEWS * BLOCK);
        drop_views(section, &[one, two], BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }

    /// Unset, a write into physical pages through a view of our own opens
    /// the guards under it and makes the textures there due again, as
    /// every such write did on this line before the switch came in; `0`
    /// only counts.
    #[test]
    fn page_writes_act_by_default() {
        assert_eq!(parse_page_writes(None), 3);
        assert_eq!(parse_page_writes(Some("1")), 3);
        assert_eq!(parse_page_writes(Some("0")), 0);
        assert_eq!(parse_page_writes(Some(" guards ")), 1);
        assert_eq!(parse_page_writes(Some("slots")), 2);
        assert_eq!(parse_page_writes(Some("all")), 7);
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{PAGE, by_range, title_memory_reading, title_memory_remapping};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// Addresses only these tests hold (nothing is read there).
    const HERE: usize = 0x7e00_0000_0000;

    /// A change of mapping waits for a read in place of its bytes that is
    /// under way, and such a read waits for the change.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading(HERE + PAGE / 2, PAGE);
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping(HERE + PAGE, PAGE);
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping(HERE, 2 * PAGE);
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading(HERE + PAGE + 16, 16);
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!read.load(Ordering::SeqCst), "a read in place began during a change of mapping");
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }

    /// Reads and changes whose bytes do not meet do not wait for each other,
    /// and a change waiting for a read keeps new reads of its bytes out (but
    /// not of others).
    #[test]
    fn reads_and_remappings_of_other_bytes_go_on() {
        if !by_range() {
            return; // VICTORY_MAPPING_RANGES=0: one lock over all of it
        }
        let at = HERE + 16 * PAGE;
        let reading = title_memory_reading(at, PAGE);
        // Another page changes while the read is under way, and another
        // read goes on while that change is.
        let (done, finished) = mpsc::channel();
        std::thread::spawn(move || {
            let changing = title_memory_remapping(at + PAGE, PAGE);
            let (read, was_read) = mpsc::channel();
            std::thread::spawn(move || {
                let _reading = title_memory_reading(at + 2 * PAGE, PAGE);
                read.send(()).unwrap();
            });
            let other = was_read.recv_timeout(Duration::from_secs(10)).is_ok();
            drop(changing);
            done.send(other).unwrap();
        });
        assert_eq!(finished.recv_timeout(Duration::from_secs(10)), Ok(true), "kept apart though their bytes do not meet");
        // A change of the read's page waits, and keeps a new read of its
        // bytes out meanwhile; a read elsewhere is not kept out.
        let changed = Arc::new(AtomicBool::new(false));
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                let _changing = title_memory_remapping(at, PAGE);
                changed.store(true, Ordering::SeqCst);
            })
        };
        while !super::spans().changes.iter().any(|c| c.1 == at) {
            std::thread::yield_now();
        }
        assert!(super::title_memory_reading_now(at + 8, 8).is_none(), "a read began over a change that waits");
        assert!(super::title_memory_reading_now(at + 3 * PAGE, 8).is_some());
        std::thread::sleep(Duration::from_millis(50));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
    }
}

/// The loader's console-instruction emulation writing title memory through
/// the opener ([`writing`]), against the real guards.
#[cfg(test)]
mod console_writes_tests;

/// F5's guard ledger (`VICTORY_GUARD_LEDGER=1`, [`crate::ledger`]). The
/// switch is read once a process, so each fixture runs in a child of this
/// test binary with it on (and the other tests here see it off). Run in a
/// child without the switch, each fails as today's guards do.
#[cfg(test)]
mod ledger_tests {
    use super::*;
    use crate::ledger::NATIVE;
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE};

    const CHILD: &str = "VICTORY_LEDGER_TEST_CHILD";

    /// `body` in a child of this binary with the ledger on (`name`: this
    /// test's path), or, in the child, `body` itself. At most a minute.
    fn in_child(name: &str, body: fn()) {
        if std::env::var_os(CHILD).is_some() {
            return body();
        }
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--nocapture", "--test-threads=1"])
            .env(CHILD, "1")
            .env("VICTORY_GUARD_LEDGER", "1")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let started = std::time::Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if started.elapsed() > std::time::Duration::from_secs(60) {
                let _ = child.kill();
                panic!("{name}: the child did not end in a minute (a fault that never ends?)");
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        };
        let out = child.wait_with_output().unwrap();
        let said = String::from_utf8_lossy(&out.stdout);
        assert!(
            status.success() && said.contains("1 passed"),
            "{name} with the ledger on (the child must run this one test):\n{said}{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// A pagefile section of `bytes` mapped once, told to the runtime as a
    /// title view: the section and the view.
    fn one_view(bytes: usize) -> (usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
        // SAFETY: a fresh pagefile section, mapped once.
        unsafe {
            let section =
                CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, bytes as u32, std::ptr::null());
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(view, 0);
            crate::inplace::title_memory_view(view, bytes, section as usize, 0);
            (section as usize, view)
        }
    }

    fn close(section: usize) {
        // SAFETY: a section handle of ours.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(section as *mut std::ffi::c_void) };
    }

    fn unmap(view: usize) {
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        // SAFETY: a view of ours.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
    }

    /// What the host shows at `at`.
    fn shown(at: usize) -> u32 {
        query(crate::kernelstat::Place::Guard, at).expect("mapped").Protect
    }

    /// What the title is told of `at`: protection and region size.
    fn title_sees(at: usize) -> (u32, usize) {
        // SAFETY: plain data filled in by the query.
        let mut info: windows_sys::Win32::System::Memory::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: room for the answer.
        let n = unsafe { title_virtual_query(at as *const _, &mut info, size_of_val(&info)) };
        assert_ne!(n, 0);
        (info.Protect, info.RegionSize)
    }

    /// The title's `VirtualProtect` of `[at, at + len)`: its old value.
    fn title_protect(at: usize, len: usize, new: u32) -> u32 {
        let mut old = 0;
        // SAFETY: our pages, room for the answer.
        assert_ne!(unsafe { title_virtual_protect(at as *const _, len, new, &mut old) }, 0);
        old
    }

    /// The title's own handler, after ours: it takes write faults on the
    /// page at [`TITLE_AT`], counts them, and makes the page writable (as
    /// a title tracking its own writes would).
    static TITLE_AT: AtomicUsize = AtomicUsize::new(0);
    static TITLE_FAULTS: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "system" fn title_handler(
        info: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
    ) -> i32 {
        use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
        use windows_sys::Win32::System::Diagnostics::Debug::{
            EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        };
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        let at = TITLE_AT.load(SeqCst);
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != 1
            || at == 0
            || r.ExceptionInformation[1] / NATIVE != at / NATIVE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        TITLE_FAULTS.fetch_add(1, SeqCst);
        title_protect(at, 1, PAGE_READWRITE);
        EXCEPTION_CONTINUE_EXECUTION
    }

    /// Runs `f` with [`title_handler`] taking faults on the page at `at`:
    /// how many it took.
    fn title_faults(at: usize, f: impl FnOnce()) -> usize {
        use windows_sys::Win32::System::Diagnostics::Debug::{
            AddVectoredExceptionHandler, RemoveVectoredExceptionHandler,
        };
        TITLE_AT.store(at, SeqCst);
        TITLE_FAULTS.store(0, SeqCst);
        // SAFETY: a handler of ours, last in line, removed below.
        let handler = unsafe { AddVectoredExceptionHandler(0, Some(title_handler)) };
        assert!(!handler.is_null());
        f();
        // SAFETY: the handler added above.
        unsafe { RemoveVectoredExceptionHandler(handler) };
        TITLE_AT.store(0, SeqCst);
        TITLE_FAULTS.load(SeqCst)
    }

    /// The title changes the protection of guarded pages and reads back its
    /// own values, from `VirtualProtect` and `VirtualQuery`, while the guard
    /// stays and still sees its writes. Today's guards answer `PAGE_READONLY`.
    #[test]
    fn a_title_protection_under_a_guard_reads_back_as_its_own() {
        in_child("watch::ledger_tests::a_title_protection_under_a_guard_reads_back_as_its_own", || {
            const SIZE: usize = 2 * BLOCK;
            let (section, view) = one_view(SIZE);
            let pages = |block: usize| (view + block * BLOCK) / PAGE;
            assert_eq!(guard(view, view + SIZE), Some(vec![pages(0), pages(1)]));
            assert_eq!(shown(view), PAGE_READONLY, "guarded");
            assert_eq!(title_protect(view, NATIVE, PAGE_READWRITE), PAGE_READWRITE, "the title's, not the guard's");
            assert_eq!(shown(view), PAGE_READONLY, "the guard stays");
            assert_eq!(title_protect(view + NATIVE, NATIVE, PAGE_NOACCESS), PAGE_READWRITE);
            assert_eq!(title_sees(view + NATIVE), (PAGE_NOACCESS, NATIVE), "its page, as the title cut it");
            assert_eq!(shown(view + NATIVE), PAGE_NOACCESS, "no access stays no access");
            assert_eq!(title_protect(view + NATIVE, NATIVE, PAGE_READWRITE), PAGE_NOACCESS);
            // The title's calls may have let a write past: the block counts as written.
            assert_eq!(guard(view, view + SIZE), Some(vec![pages(0)]));
            assert_eq!(title_sees(view), (PAGE_READWRITE, SIZE), "one region, the guards' edges unseen");
            // The guard still sees the title's writes.
            // SAFETY: our view, read-write to the title.
            unsafe { ((view + BLOCK + 8) as *mut u8).write(1) };
            assert_eq!(shown(view + BLOCK), PAGE_READWRITE);
            assert_eq!(title_sees(view), (PAGE_READWRITE, SIZE), "one region though the host splits it");
            assert_eq!(guard(view, view + SIZE), Some(vec![pages(1)]));
            title_memory_remapped(view, SIZE);
            unmap(view);
            close(section);
        });
    }

    /// The title gives a guarded view back and allocates the same addresses
    /// read-only while a kernel read into it is under way: neither the
    /// read's end nor a late writer of ours restores the old guard's
    /// protection over the new memory. Today's guards open it read-write.
    #[test]
    fn a_guarded_range_freed_and_allocated_again_keeps_the_new_protection() {
        in_child("watch::ledger_tests::a_guarded_range_freed_and_allocated_again_keeps_the_new_protection", || {
            use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE};
            const SIZE: usize = 2 * BLOCK;
            let (section, view) = one_view(SIZE);
            assert!(guard(view, view + SIZE).is_some());
            let read = kernel_writing(view + 100, 13);
            // SAFETY: our view, given back by the title's route.
            assert_ne!(unsafe { title_unmap_view(view as *const _) }, 0);
            close(section);
            // SAFETY: the addresses just given back.
            let again = unsafe { title_virtual_alloc(view as *const _, SIZE, MEM_RESERVE | MEM_COMMIT, PAGE_READONLY) };
            assert_eq!(again as usize, view, "the same addresses");
            drop(read);
            writing(view, SIZE);
            assert_eq!(shown(view), PAGE_READONLY, "the title's protection, not a guard's release");
            assert_eq!(shown(view + BLOCK), PAGE_READONLY, "nor at the block the read did not touch");
            assert_eq!(guard(view, view + SIZE), None, "new private memory: nothing guarded is left");
            // SAFETY: the allocation above.
            assert_ne!(unsafe { title_virtual_free(again, 0, MEM_RELEASE) }, 0);
        });
    }

    /// The title made a guarded page read-only itself and writes it: the
    /// fault is its own, its handler sees it, and the guard stays; a write
    /// only the guard refuses is ours and the title's handler never sees
    /// it. Today the guard opens the page and the title's fault is lost.
    #[test]
    fn a_title_write_to_its_own_read_only_page_goes_to_the_title() {
        in_child("watch::ledger_tests::a_title_write_to_its_own_read_only_page_goes_to_the_title", || {
            let (section, view) = one_view(BLOCK);
            assert!(guard(view, view + BLOCK).is_some());
            title_protect(view, NATIVE, PAGE_READONLY);
            // SAFETY: our view; the title's handler makes the page writable.
            let faults = title_faults(view, || unsafe { (view as *mut u8).write(5) });
            assert_eq!(faults, 1, "the title's own protection refused it: its fault");
            // SAFETY: our view.
            assert_eq!(unsafe { (view as *const u8).read() }, 5, "then let through as the title chose");
            assert!(guard(view, view + BLOCK).is_some_and(|written| !written.is_empty()));
            let at = view + 2 * NATIVE;
            // SAFETY: our view, read-write to the title.
            let faults = title_faults(at, || unsafe { (at as *mut u8).write(6) });
            assert_eq!(faults, 0, "the guard's fault is not the title's");
            assert_eq!(guard(view, view + BLOCK), Some(vec![view / PAGE]));
            title_memory_remapped(view, BLOCK);
            unmap(view);
            close(section);
        });
    }

    /// One page shown at two addresses, guarded at both; a write through the
    /// second faults into the guard, which opens both, each to what the
    /// title asked there: a page the title keeps read-only at the first
    /// stays so. Today the release makes the whole block read-write.
    #[test]
    fn a_guard_fault_at_one_view_restores_each_view_its_own() {
        in_child("watch::ledger_tests::a_guard_fault_at_one_view_restores_each_view_its_own", || {
            const SIZE: usize = BLOCK;
            let (section, one, two) = super::tests::two_views(SIZE);
            title_protect(one + NATIVE, NATIVE, PAGE_READONLY);
            assert!(guard(one, one + SIZE).is_some());
            assert_eq!(shown(two), PAGE_READONLY, "the other view is guarded with it");
            assert_eq!(title_sees(two).0, PAGE_READWRITE, "and the title is not told");
            // SAFETY: the second view, read-write to the title.
            let faults = title_faults(two, || unsafe { ((two + 8) as *mut u8).write(1) });
            assert_eq!(faults, 0, "a guard's fault, not the title's");
            assert_eq!((shown(one), shown(two)), (PAGE_READWRITE, PAGE_READWRITE));
            assert_eq!(shown(one + NATIVE), PAGE_READONLY, "the title's read-only page stays so");
            assert_eq!(guard(one, one + SIZE), Some(vec![one / PAGE]), "written, at the guarded view");
            super::tests::drop_views(section, &[one, two], SIZE);
        });
    }
}
