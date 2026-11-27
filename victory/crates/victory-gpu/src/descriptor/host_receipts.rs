//! The receipts walk (`crate::receipts`, `VICTORY_TEXTURE_RECEIPTS=1`): each view heap's
//! texture pass looks at all its texture slots in turn, tables' and index-only alike, and
//! owes the ones whose host copy is older than what was written over their memory since
//! (the copy's receipt below a stamp in the journal), whatever heap took the notices of
//! those writes.
//!
//! A slot in debt reaches `upload_from` within one sweep, or sits behind a named, counted
//! blocker ([`crate::receipts::BLOCKERS`]): the walk puts it in the queue due now, or opens
//! the part-upload gate of a queued one and makes it due; what holds it then is the card's
//! room, its backoff or a table's grace, a title texture not made yet, or levels on demand
//! without tiles. Only an install clears debt.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::Ordering::Relaxed;

use super::{HostHeap, TABLED_GRACE, TextureQueue, Waiting};
use crate::descriptor::{View, view_at};
use crate::object::This;
use crate::receipts::Summary;

/// A sweep's pace: all of a heap's texture slots in about this much wall time.
const SWEEP_MS: u64 = 1000;
/// The most one pass walks for (it moves at least one slot).
const PASS: std::time::Duration = std::time::Duration::from_millis(2);
/// Frames between a heap's lines.
const LINE_FRAMES: u64 = 600;

/// Numbers for the heaps' lines.
static HEAPS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// A view heap's texture slots and the walk over them (under its own lock, a leaf:
/// the early, bake and heap-buffer passes record on threads of their own).
#[derive(Default)]
pub(crate) struct Walk {
    /// Title slot → the record (generation << 32 | index) of the texture view its host
    /// slot was last made from, placed or object, tables' or not.
    sources: BTreeMap<u32, u64>,
    /// The next slot to look at.
    cursor: u32,
    /// When the last pass walked, and when this sweep began (`crate::host::now_ms`).
    last_ms: u64,
    began_ms: u64,
    /// This sweep so far, and the last whole one.
    sweep: Summary,
    last: Summary,
    /// Slots in debt, and the frame each was first found so.
    debt_since: HashMap<u32, u64>,
    /// Slots in debt the pass's budget left due since they were last found settled: the
    /// blocker `budget` of those still in debt at a sweep's end.
    left: std::collections::HashSet<u32>,
    /// The frame (in [`LINE_FRAMES`]) of the last line; the heap's number.
    line: u64,
    id: u32,
}

/// Host-written textures found written since, said by name at most this often in a run.
const NEWER_SAID: u64 = 24;
static NEWER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Why a slot's copy is in debt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Debt {
    /// No receipt: never installed.
    NoReceipt,
    /// The receipt is of another host copy than the one made now.
    OtherCopy,
    /// The view shows levels below the receipt's first.
    Levels,
    /// A stamp over the memory above the receipt.
    Written,
    /// A placed view whose title texture is not made yet: the pass makes it.
    NeverMade,
    /// Levels on demand left without tiles: `world`'s WAITING asks again with room.
    Short,
    /// The host refused to make the host copy: not owed (it never will be made).
    Refused,
    /// The last upload had no staging buffer: owed again, under this name.
    NoStaging,
}

/// What a look at one slot found.
enum Look {
    /// The record names no texture view any more.
    Gone,
    /// Left out ([`crate::receipts::EXCLUDED`]).
    Excluded(usize),
    /// Left out as host-written; its memory and the number of the later of its last upload
    /// and the host GPU's last write, to look for writes since.
    HostWritten(usize, usize, usize, u64),
    /// Never in debt.
    Never,
    Debt(Debt, Option<usize>, u64),
    /// To be compared with the journal: the texture, its memory and the receipt, and
    /// whether the receipt is an absent one.
    Check(usize, usize, usize, u64, bool),
}

/// What the queue did with a slot in debt ([`TextureQueue::owed`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Owe {
    /// Put in, as a write due at the frame given.
    Queued,
    /// In the queue, held for room on the card.
    Held,
    /// In the queue, due later (its backoff, a table's grace).
    NotDue,
    /// In the queue and due; `true` if its part-upload gate was shut.
    Due(bool),
}

impl TextureQueue {
    /// Slot `index` is in debt in `frame`: out of the queue, it is put in as a write due
    /// at `due`; in it, if `actionable`, its part-upload gate opens (`mapped`, which the
    /// pass compares with what is mapped now) and it is due by `due`, unless it is held for
    /// the card's room. Nothing else of it changes: its looks and its hold stay, so a full
    /// card's churn does not come back. (`changed` would reset both.)
    pub(super) fn owed(&mut self, index: u32, frame: u64, due: u64, actionable: bool) -> Owe {
        let held = self.held.contains_key(&index);
        if let Some(w) = self.slots.get_mut(&index) {
            let gated = w.mapped != 0;
            if actionable {
                w.mapped = 0;
                if !held {
                    w.due = w.due.min(due);
                }
            }
            return if held {
                Owe::Held
            } else if w.due > frame {
                Owe::NotDue
            } else {
                Owe::Due(gated && actionable)
            };
        }
        self.writes += 1;
        let write = self.writes;
        self.rechecks.remove(&index);
        self.slots.insert(index, Waiting { due, written: frame, write, looks: 0, mapped: 0 });
        Owe::Queued
    }
}

/// The pass found the memory of a placed view not mapped at all, at or after `seq` (the
/// journal's number before its looks): the view's title texture, if it is made, owes
/// nothing until a map or a write there is stamped above it. `len`: the bytes of the levels
/// the view shows.
pub(crate) fn absent(address: u64, resource: &[u32; 14], desc: &[u32; 16], len: u64, seq: u64) {
    let Some(this) = crate::resource::placed_known(address, resource) else {
        return;
    };
    let first = crate::resource::first_mip(resource, desc);
    crate::receipts::absent(this as usize, seq, (address as usize, len as usize, first));
}

/// One slot's look, by its record `key`, with no lock of the heap's held.
fn look(key: u64) -> Look {
    let Some(view) = view_at(key as u32, (key >> 32) as u32) else {
        return Look::Gone;
    };
    // The texture, the first level the view shows, and the view's memory (address, bytes).
    let (this, first, memory) = match &view {
        View::Placed(false, resource, desc) if resource[0] != 1 => {
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let first = crate::resource::first_mip(resource, desc);
            let bytes = crate::resource::placed_bytes_from(resource, first).unwrap_or(0);
            let this = crate::resource::placed_known(address, resource).map(|t| t as usize);
            (this, first, Some((address as usize, bytes as usize)))
        }
        View::Shader(object, _) if crate::resource::buffer_bytes(*object as This).is_none() => (Some(*object), 0, None),
        _ => return Look::Gone,
    };
    let bytes = memory.map_or(0, |m| m.1 as u64);
    let Some(this) = this else {
        return Look::Debt(Debt::NeverMade, None, bytes);
    };
    let texture = this as This;
    // Left out: a view `alias` fills from a texture the host writes; a texture the host
    // GPU wrote by a copy or resolve; a texture the host renders to or writes. None of
    // them is filled from title memory.
    if crate::resource::alias_of(texture).is_some() {
        return Look::Excluded(0);
    }
    let receipt = crate::receipts::get(this);
    if crate::texture::host_wrote(this) {
        // Looked at for writes after the later of its last upload and the host GPU's
        // last write: what no copy has.
        let since = receipt.as_ref().filter(|r| r.host.is_some()).map_or(0, |r| r.seq);
        let since = since.max(crate::receipts::host_wrote_at(this).unwrap_or(0));
        let memory = memory.or_else(|| object_memory(texture)).filter(|m| m.1 > 0);
        return match memory {
            Some((start, len)) if since > 0 => Look::HostWritten(this, start, start.saturating_add(len), since),
            _ => Look::Excluded(1),
        };
    }
    if !crate::resource::title_fills(texture) {
        return Look::Excluded(2);
    }
    // Named blockers: a host copy the host refused, and an upload that had no staging
    // buffer (owed again under its name).
    if crate::resource::host_refused(texture) {
        return Look::Debt(Debt::Refused, Some(this), bytes);
    }
    if crate::receipts::staging_refused(this) {
        return Look::Debt(Debt::NoStaging, Some(this), bytes);
    }
    let Some(r) = receipt else {
        return Look::Debt(Debt::NoReceipt, Some(this), bytes);
    };
    if r.never {
        return Look::Never;
    }
    if r.short {
        return Look::Debt(Debt::Short, Some(this), r.len as u64);
    }
    if let Some(host) = &r.host
        && !crate::resource::host_weak(texture).is_some_and(|now| std::sync::Weak::ptr_eq(&now, host))
    {
        return Look::Debt(Debt::OtherCopy, Some(this), r.len as u64);
    }
    if first < r.first {
        return Look::Debt(Debt::Levels, Some(this), bytes);
    }
    Look::Check(this, r.start, r.start.saturating_add(r.len), r.seq, r.host.is_none())
}

/// A texture object's title memory: address and the bytes its description takes there.
fn object_memory(texture: This) -> Option<(usize, usize)> {
    let (at, desc) = crate::resource::texture_memory(texture)?;
    // From its fields: the description's padding is no data.
    let words = desc.words();
    Some((at as usize, crate::resource::placed_bytes(&words)? as usize))
}

/// Adds a pass's counts to a sweep's.
fn add(to: &mut Summary, from: &Summary) {
    to.visited += from.visited;
    to.debt += from.debt;
    to.debt_bytes += from.debt_bytes;
    for (to, n) in to.blockers.iter_mut().zip(from.blockers) {
        *to += n;
    }
    for (to, n) in to.excluded.iter_mut().zip(from.excluded) {
        *to += n;
    }
    to.host_written_newer += from.host_written_newer;
    to.queued += from.queued;
    to.ungated += from.ungated;
    to.unmapped += from.unmapped;
    to.cut += from.cut;
    to.passes += from.passes;
    to.pass_us += from.pass_us;
    to.pass_us_worst = to.pass_us_worst.max(from.pass_us_worst);
}

impl HostHeap {
    /// Title slot `index`'s host slot is now made from texture view `key` (a record's
    /// `generation << 32 | index`), or from no texture view.
    pub(super) fn walk_note(&self, index: u32, key: Option<u64>) {
        let mut w = self.walk.lock().unwrap_or_else(|e| e.into_inner());
        match key {
            Some(key) => {
                w.sources.insert(index, key);
            }
            None => {
                w.sources.remove(&index);
                w.debt_since.remove(&index);
            }
        }
    }

    /// The walk's part of the heap's texture pass in `frame` ([`self`](super::receipts)):
    /// the next slots, as many as the time since the last pass is of a sweep's (at least
    /// one, for at most [`PASS`]); the ones in debt owed ([`TextureQueue::owed`]).
    pub(crate) fn receipts_walk(&self, frame: u64) {
        if self.sampler {
            return;
        }
        let began = std::time::Instant::now();
        let now_ms = crate::host::now_ms();
        // The slots from the cursor on; where in them the sweep went round, if it did.
        let (chunk, wrap) = {
            let mut w = self.walk.lock().unwrap_or_else(|e| e.into_inner());
            if w.id == 0 {
                w.id = HEAPS.fetch_add(1, Relaxed) + 1;
            }
            let total = w.sources.len();
            if total == 0 {
                return;
            }
            if w.began_ms == 0 {
                w.began_ms = now_ms;
            }
            let since = if w.last_ms == 0 { 1 } else { now_ms.saturating_sub(w.last_ms).clamp(1, SWEEP_MS) };
            w.last_ms = now_ms;
            let n = ((total as u64 * since).div_ceil(SWEEP_MS) as usize).clamp(1, total);
            let mut chunk: Vec<(u32, u64)> = w.sources.range(w.cursor..).take(n).map(|(&i, &k)| (i, k)).collect();
            let wrap = (chunk.len() < n).then_some(chunk.len());
            if let Some(at) = wrap {
                chunk.extend(w.sources.range(..w.cursor).take(n - at).map(|(&i, &k)| (i, k)));
            }
            w.cursor = chunk.last().map_or(0, |&(i, _)| i.saturating_add(1));
            (chunk, wrap)
        };
        // What each slot's look found, counted for the sweep it belongs to: the one under
        // way (0), or past the point where the sweep went round, the next (1).
        let side = |k: usize| usize::from(wrap.is_some_and(|at| k >= at));
        let mut tally = [Summary::default(); 2];
        let mut stop = chunk.len();
        let (mut gone, mut settled, mut debts, mut checks) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut probes = Vec::new();
        for (k, &(index, key)) in chunk.iter().enumerate() {
            if k > 0 && began.elapsed() > PASS {
                stop = k;
                break;
            }
            let t = &mut tally[side(k)];
            t.visited += 1;
            match look(key) {
                Look::Gone => gone.push((index, key)),
                Look::Excluded(why) => {
                    t.excluded[why] += 1;
                    settled.push(index);
                }
                Look::HostWritten(this, start, end, since) => {
                    t.excluded[1] += 1;
                    settled.push(index);
                    probes.push((index, this, start, end, since, side(k)));
                }
                Look::Never => settled.push(index),
                Look::Debt(why, this, bytes) => debts.push((index, why, this, bytes, side(k))),
                Look::Check(this, start, end, seq, absent) => checks.push((index, this, start, end, seq, absent, side(k))),
            }
        }
        // One taking of the journal for the chunk: the receipts, then the host-written
        // textures' memory since the later of their last upload and the host's last write.
        let mut ranges: Vec<(usize, usize, u64)> = checks.iter().map(|c| (c.2, c.3, c.4)).collect();
        ranges.extend(probes.iter().map(|p| (p.2, p.3, p.4)));
        let mut fresh = if ranges.is_empty() { Vec::new() } else { crate::watch::fresh_all(&ranges) };
        let unwritten = fresh.split_off(checks.len());
        for (p, unwritten) in probes.into_iter().zip(unwritten) {
            if !unwritten {
                tally[p.5].host_written_newer += 1;
                if NEWER.fetch_add(1, Relaxed) < NEWER_SAID {
                    tracing::info!(
                        slot = p.0,
                        texture = format_args!("{:#x}", p.1),
                        address = format_args!("{:#x}", p.2),
                        bytes = p.3 - p.2,
                        "texture receipts: a texture the host GPU wrote was written in title memory since; it is not                          uploaded from memory again (host_written_newer)"
                    );
                }
            }
        }
        for (c, fresh) in checks.into_iter().zip(fresh) {
            if fresh {
                tally[c.6].unmapped += u64::from(c.5);
                settled.push(c.0);
            } else {
                debts.push((c.0, Debt::Written, Some(c.1), (c.3 - c.2) as u64, c.6));
            }
        }
        // One taking of the queue for the debts.
        if !debts.is_empty() {
            let lately: Vec<bool> = debts.iter().map(|d| self.tabled_lately(d.0, frame)).collect();
            let mut queue = self.textures.lock().unwrap_or_else(|e| e.into_inner());
            for (&(index, why, _, bytes, side), lately) in debts.iter().zip(lately) {
                let t = &mut tally[side];
                t.debt += 1;
                t.debt_bytes += bytes;
                // Named, and not owed: levels on demand without tiles (`world`'s WAITING asks
                // again), a host copy the host refused (it never will be made).
                if why == Debt::Short {
                    t.blockers[3] += 1;
                    continue;
                }
                if why == Debt::Refused {
                    t.blockers[crate::receipts::REFUSED] += 1;
                    continue;
                }
                // A table's slot lately: its draw goes first, as with a write of it.
                let due = if lately { frame + TABLED_GRACE } else { frame };
                match queue.owed(index, frame, due, why != Debt::NeverMade) {
                    _ if why == Debt::NoStaging => t.blockers[crate::receipts::NO_STAGING_BLOCKER] += 1,
                    Owe::Queued => t.queued += 1,
                    Owe::Held => t.blockers[0] += 1,
                    Owe::NotDue | Owe::Due(_) if why == Debt::NeverMade => t.blockers[2] += 1,
                    Owe::NotDue => t.blockers[1] += 1,
                    Owe::Due(ungated) => t.ungated += u64::from(ungated),
                }
            }
        }
        for &(_, why, this, ..) in &debts {
            if let Some(this) = this
                && why != Debt::Short
                && why != Debt::Refused
            {
                crate::receipts::owed(this);
            }
        }
        let us = began.elapsed().as_micros() as u64;
        let mut w = self.walk.lock().unwrap_or_else(|e| e.into_inner());
        // Cut short: the rest of the chunk is the next pass's, unless another pass has
        // moved the cursor since.
        if stop < chunk.len() {
            tally[side(stop)].cut += 1;
            let ours = chunk.last().map_or(0, |&(i, _)| i.saturating_add(1));
            if w.cursor == ours {
                w.cursor = chunk[stop].0;
            }
        }
        for (index, key) in gone {
            if w.sources.get(&index) == Some(&key) {
                w.sources.remove(&index);
            }
            w.debt_since.remove(&index);
            w.left.remove(&index);
        }
        for index in settled {
            w.debt_since.remove(&index);
            w.left.remove(&index);
        }
        for &(index, ..) in &debts {
            w.debt_since.entry(index).or_insert(frame);
        }
        tally[0].passes = 1;
        tally[0].pass_us = us;
        tally[0].pass_us_worst = us;
        add(&mut w.sweep, &tally[0]);
        // The sweep went round: what it found is the heap's to show; the slots past the
        // point are the next sweep's.
        if wrap.is_some_and(|at| stop >= at) {
            let mut done = std::mem::replace(&mut w.sweep, tally[1]);
            done.slots = w.sources.len() as u64;
            done.oldest = w.debt_since.values().map(|&since| frame.saturating_sub(since)).max().unwrap_or(0);
            done.blockers[crate::receipts::BUDGET] =
                w.left.iter().filter(|index| w.debt_since.contains_key(index)).count() as u64;
            done.sweep_ms = now_ms.saturating_sub(w.began_ms);
            done.slot_ns = done.pass_us * 1000 / done.visited.max(1);
            done.pass_us /= done.passes.max(1);
            w.began_ms = now_ms;
            w.last = done;
            crate::receipts::swept(w.id, done);
        }
        if frame / LINE_FRAMES != w.line {
            w.line = frame / LINE_FRAMES;
            crate::receipts::line(w.id, &w.last);
        }
    }

    /// The texture pass's loop stopped at its byte, time or refusal budget with `left` (title
    /// slot, write) still due: those in debt are held by the budget ([`crate::receipts::BUDGET`]).
    pub(crate) fn receipts_left(&self, left: &[(u32, u64)]) {
        if left.is_empty() {
            return;
        }
        let mut w = self.walk.lock().unwrap_or_else(|e| e.into_inner());
        let w = &mut *w;
        for &(index, _) in left {
            if w.debt_since.contains_key(&index) {
                w.left.insert(index);
            }
        }
    }

    /// Tests: the walk's last whole sweep.
    #[cfg(test)]
    fn walk_last(&self) -> Summary {
        self.walk.lock().unwrap_or_else(|e| e.into_inner()).last
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::Host;
    use windows::Win32::Graphics::Direct3D12::*;

    /// A placed 2-D texture of one level at `address`, BC7 (98) or another format, `flags`
    /// 1 for a render target: its resource description as words, and a view of it.
    fn placed(address: usize, width: u32, height: u32, format: u32, flags: u32) -> ([u32; 14], View) {
        let resource = [3, 0, 0, 0, width, 0, height, 0x1_0001, format, 1, 0, 0x105, flags, 0];
        // A 2-D view (dimension 4) of the format, the default component mapping, from level
        // 0, one level; then the memory's address.
        let mut desc = [format, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        (desc[10], desc[11]) = (address as u32, (address as u64 >> 32) as u32);
        (resource, View::Placed(false, resource, desc))
    }

    /// Title memory at `at` (no other fixture's, and each runs in a process of its own):
    /// `bytes` reserved and write-watched, as the runtime reserves title memory, the first
    /// `committed` of them committed.
    fn memory(at: usize, bytes: usize, committed: usize) {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        // SAFETY: new memory at an address no one holds, left to the process.
        let base = unsafe { VirtualAlloc(at as *const _, bytes, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE) };
        assert_eq!(base as usize, at, "the fixture's address is free");
        if committed > 0 {
            // SAFETY: as above, the start of what was just reserved.
            let done = unsafe { VirtualAlloc(at as *const _, committed, MEM_COMMIT, PAGE_READWRITE) };
            assert!(!done.is_null(), "committed");
        }
    }

    /// A texture object's memory does not depend on its description's padding: two
    /// textures made through the console interface (device slot 0x428) from descriptions
    /// written field by field, one with its padding words (1, after the dimension, and
    /// 13, at the end) zero and one with them set to a pattern, take the same bytes, those the
    /// description's words with the padding blank give.
    #[test]
    fn an_objects_memory_takes_no_padding_of_its_description() {
        // D3D12_RESOURCE_DESC's fields at their places, as 8-byte words: dimension and
        // padding, alignment, width, height with depth and mips, format and sample
        // count, sample quality and layout, flags and padding.
        let written = |pad_1: u32, pad_13: u32| -> [u64; 7] {
            let (dimension, width, height, depth, mips, format) = (3u64, 256u64, 128u64, 1u64, 1u64, 98u64);
            let (samples, quality, layout, flags) = (1u64, 0u64, 0x105u64, 0u64);
            [
                dimension | u64::from(pad_1) << 32,
                0,
                width,
                height | (depth | mips << 16) << 32,
                format | samples << 32,
                quality | layout << 32,
                flags | u64::from(pad_13) << 32,
            ]
        };
        let (zeroed_at, patterned_at) = (0x7e_7000_0000u64, 0x7e_7100_0000u64);
        let zeroed = crate::resource::test_texture_at(zeroed_at, &written(0, 0));
        let patterned = crate::resource::test_texture_at(patterned_at, &written(0xa5a5_a5a5, 0x5a5a_5a5a));
        let blank: [u32; 14] = [3, 0, 0, 0, 256, 0, 128, 0x1_0001, 98, 1, 0, 0x105, 0, 0];
        let bytes = crate::resource::placed_bytes(&blank).expect("a texture's bytes") as usize;
        assert!(bytes > 0);
        assert_eq!(object_memory(zeroed), Some((zeroed_at as usize, bytes)), "padding zero");
        assert_eq!(object_memory(patterned), Some((patterned_at as usize, bytes)), "padding set");
    }

    /// The title fills `[at, at + len)` with `byte`.
    fn fill(at: usize, len: usize, byte: u8) {
        // SAFETY: committed fixture memory.
        unsafe { std::ptr::write_bytes(at as *mut u8, byte, len) };
    }

    /// DirectStorage reads `len` bytes of `byte` to the address `at`: told before, the
    /// bytes, told when they are there (`dstorage.rs`' order).
    fn storage(at: usize, len: usize, byte: u8) {
        crate::watch::writing(at, len);
        fill(at, len, byte);
        crate::watch::title_memory_written(at, len, false);
    }

    /// A title heap's descriptor memory: eight slots.
    struct Title(Vec<[u32; 8]>);

    impl Title {
        fn new() -> Self {
            Self(vec![[0u32; 8]; 8])
        }
        fn handle(&self, index: u32) -> u64 {
            self.0.as_ptr() as u64 + u64::from(index) * 32
        }
        /// The title writes `view` into slot `index`, and the heap mirrors it.
        fn write(&self, heap: &HostHeap, index: u32, view: View) {
            super::super::super::record(self.handle(index), view);
            heap.mirror(index, self.handle(index), false);
        }
    }

    /// A draw uploads what `index` shows, as `readable` does for a table's entry or the
    /// pass for a slot it serves: `alias` for a view it fills, else `upload_from` if the
    /// title fills the texture; the list submitted (the steps at submission run, then the
    /// card). The title texture.
    fn upload(host: &Host, heap: &HostHeap, title: &Title, index: u32) -> Option<This> {
        let view = heap.mirror(index, title.handle(index), true)?;
        let shown = super::super::placed_texture(&view);
        let Some(&View::Shader(this, _)) = shown.as_ref().or(Some(&view)) else {
            return None;
        };
        let this = this as This;
        let first = match &view {
            View::Placed(_, resource, desc) => crate::resource::first_mip(resource, desc),
            _ => 0,
        };
        let list = host.new_list().expect("a list");
        let aliased = crate::resource::alias_of(this).is_some_and(|source| crate::texture::alias(&list, this, source));
        if !aliased
            && let Some(texture) = crate::resource::host_resource(this)
            && crate::texture::title_written(texture)
        {
            crate::texture::upload_from(&list, this, texture, first);
        }
        list.close();
        host.execute(&[&list]);
        Some(this)
    }

    /// The heap's texture pass at `frame`, through the calls `State::heap_textures` makes
    /// for each due slot (without its budget, room and counters): the notices, the walk,
    /// then each due slot's look at its memory, the part-upload gate, the upload and its
    /// hand-back. The slots it uploaded.
    fn pass(host: &Host, heap: &HostHeap, title: &Title, frame: u64) -> Vec<u32> {
        pass_within(host, heap, title, frame, usize::MAX)
    }

    /// [`pass`] with a budget of `most` slots looked at, as the pass's bytes and time are:
    /// past it, the loop stops and hands the rest back (`receipts_left`).
    fn pass_within(host: &Host, heap: &HostHeap, title: &Title, frame: u64, most: usize) -> Vec<u32> {
        heap.texture_slots_remapped(frame);
        if crate::receipts::on() {
            heap.receipts_walk(frame);
        }
        let unmapped_at = crate::receipts::on().then(crate::watch::absent_number);
        let mut uploaded = Vec::new();
        let due = heap.texture_slots_due(frame, 64).0;
        let mut looked = 0;
        for &(index, write) in &due {
            if looked >= most {
                break;
            }
            looked += 1;
            let mut partly = None;
            if let Some(View::Placed(false, resource, desc)) = crate::descriptor::view(title.handle(index)) {
                let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                let (len, mapped) = crate::texture::view_mapped(address, &resource, &desc);
                if mapped == 0 {
                    heap.texture_slot_gone(index);
                    if let Some(seq) = unmapped_at {
                        absent(address, &resource, &desc, len, seq);
                    }
                    heap.texture_slot_later(index, write, frame, 16);
                    continue;
                }
                if mapped < len {
                    if mapped <= heap.texture_slot_mapped(index) {
                        heap.texture_slot_later(index, write, frame, 16);
                        continue;
                    }
                    partly = Some(mapped);
                }
            }
            // A texture with a staging buffer is one tables read: copied whole again, and
            // its staging buffer kept.
            let staged = heap
                .mirror(index, title.handle(index), true)
                .and_then(|view| super::super::placed_texture(&view).or(Some(view)))
                .and_then(|view| match view {
                    View::Shader(this, _) => Some(this as This),
                    _ => None,
                })
                .filter(|&this| crate::texture::is_staged(this));
            if let Some(this) = staged {
                crate::texture::upload_again(this);
            }
            let staged = staged.is_some();
            let Some(this) = upload(host, heap, title, index) else {
                heap.texture_slot_done(index, write, None);
                continue;
            };
            if !staged {
                crate::texture::unstage(this);
            }
            heap.texture_slot_done(index, write, partly.map(|mapped| (frame + 1, mapped)));
            uploaded.push(index);
        }
        if crate::receipts::on() {
            heap.receipts_left(&due[looked..]);
        }
        uploaded
    }

    /// G-006: the receipts fixtures' host device, or their failure: a fixture that skipped
    /// for want of one would prove nothing.
    fn required(host: Option<&'static Host>) -> &'static Host {
        host.expect("no host device (VICTORY_D3D12_WARP=1 makes WARP's): the fixture fails rather than skip")
    }

    /// A walk pass at `frame` a sweep's time after the last (the walk is paced by the
    /// clock), twice: the second's summary is one whole sweep, each slot looked at once.
    fn sweep(heap: &HostHeap, frame: u64) -> Summary {
        for frame in frame..frame + 2 {
            std::thread::sleep(std::time::Duration::from_millis(1100));
            heap.receipts_walk(frame);
        }
        heap.walk_last()
    }

    /// View memory as xmem maps it: a placeholder of `blocks` 64 KiB blocks at `at`, the
    /// first `mapped` of them a view of a section of their own (told as xmem tells it), the
    /// rest left unmapped.
    fn view_memory(at: usize, blocks: usize, mapped: usize) {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc2, VirtualFree,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        const BLOCK: usize = 64 << 10;
        // SAFETY: a fresh placeholder at an address no one holds, split, its first part
        // replaced by a view of a fresh pagefile section; all left to the process.
        let section = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                at as *const _,
                blocks * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_eq!(p, at, "the fixture's address is free");
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, (mapped * BLOCK) as u32, std::ptr::null());
            assert!(!section.is_null());
            if mapped < blocks {
                VirtualFree(at as *mut _, mapped * BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
            }
            let view = MapViewOfFile3(
                section,
                GetCurrentProcess(),
                at as *const _,
                0,
                mapped * BLOCK,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(view.Value as usize, at, "mapped");
            section as usize
        };
        crate::watch::title_memory_remapped(at, mapped * BLOCK);
        crate::inplace::title_memory_view(at, mapped * BLOCK, section, 0);
    }

    /// Subresource 0 of `this`'s host copy, read back: its rows' bytes, one after another.
    fn read_back(host: &Host, this: This) -> Vec<u8> {
        let texture = crate::resource::host_resource(this).expect("a host copy");
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        let mut placed = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let (mut rows, mut row_bytes, mut total) = (0u32, 0u64, 0u64);
        // SAFETY: outputs for one subresource.
        unsafe {
            host.device.GetCopyableFootprints(
                &d,
                0,
                1,
                0,
                Some(&mut placed),
                Some(&mut rows),
                Some(&mut row_bytes),
                Some(&mut total),
            );
        }
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: total,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback)
                .expect("a readback buffer");
        }
        let readback = readback.expect("a readback buffer");
        let back = host.new_list().expect("a list");
        crate::list::ensure_state(&back, this, 0x800);
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: placed },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        // SAFETY: an open list and live resources.
        unsafe { back.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        back.close();
        host.execute(&[&back]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy finished (`execute` waits); the buffer is `total` bytes.
        let read = unsafe {
            readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: total as usize }), Some(&mut mapped)).expect("mapped");
            std::slice::from_raw_parts(mapped.cast::<u8>(), total as usize)
        };
        let pitch = placed.Footprint.RowPitch as usize;
        let bytes = (0..rows as usize).flat_map(|row| read[row * pitch..row * pitch + row_bytes as usize].iter().copied()).collect();
        // SAFETY: mapped above.
        unsafe { readback.Unmap(0, None) };
        bytes
    }

    /// The texture `upload` made for the placed view at `address` (`resource` its words).
    fn texture(address: usize, resource: &[u32; 14]) -> This {
        crate::resource::placed_known(address as u64, resource).expect("a title texture")
    }

    /// Two heaps. Heap A's first upload of its slot reads the texture's bytes and stops
    /// before its install; meanwhile a storage read lands new bytes there, and heap B's
    /// pass takes the notice (B has no slot there); A's upload resumes. A's next passes
    /// find its copy older than the write and upload it again: its readback has the new
    /// bytes. (On `world`, B's pass took the only notice, and A's copy kept the old bytes
    /// for good: the flag and the chair.)
    #[test]
    fn a_write_another_heap_took_the_notice_of_is_uploaded_again() {
        crate::receipts::child("descriptor::host::receipts::tests::two_heaps_child");
    }

    #[test]
    #[ignore = "run by its parent, with receipts on"]
    fn two_heaps_child() {
        let host = required(crate::host::test_host());
        assert!(crate::receipts::on(), "run by its parent");
        let (a_at, b_at) = (0x7d_6000_0000usize, 0x7d_6100_0000usize);
        memory(a_at, 1 << 20, 1 << 20);
        memory(b_at, 1 << 20, 1 << 20);
        fill(a_at, 4096, 0x11);
        fill(b_at, 4096, 0x22);
        let (a, b) = (HostHeap::new(false, 8).expect("heap A"), HostHeap::new(false, 8).expect("heap B"));
        let (title_a, title_b) = (Title::new(), Title::new());
        let (resource, view) = placed(a_at, 64, 64, 98, 0);
        title_a.write(&a, 1, view);
        title_b.write(&b, 1, placed(b_at, 64, 64, 98, 0).1);
        assert_eq!(pass(host, &b, &title_b, 10), [1], "B uploads its own");
        let (arrived, at_install) = std::sync::mpsc::channel();
        let (go, wait) = std::sync::mpsc::channel();
        *crate::receipts::INSTALL_GATE.lock().unwrap() = Some((arrived, wait));
        std::thread::scope(|s| {
            let first = s.spawn(|| pass(host, &a, &title_a, 10));
            at_install.recv().expect("A's upload read the bytes and waits to install");
            storage(a_at, 4096, 0x77);
            assert_eq!(pass(host, &b, &title_b, 11), [] as [u32; 0], "B takes the notice, and has nothing there");
            go.send(()).expect("A's upload goes on");
            assert_eq!(first.join().expect("A's first pass"), [1]);
        });
        let this = texture(a_at, &resource);
        assert!(read_back(host, this).iter().all(|&byte| byte == 0x11), "A's first copy has the bytes from before");
        let again: Vec<u32> = (12..16).flat_map(|frame| pass(host, &a, &title_a, frame)).collect();
        assert_eq!(again, [1], "A uploads its slot once more");
        assert!(read_back(host, this).iter().all(|&byte| byte == 0x77), "A's copy has the storage read's bytes");
    }

    /// A slot whose texture is partly mapped is uploaded with what is there; the pass then
    /// passes it over until more is mapped. Storage fills the mapped part afterwards (heap
    /// B takes the notice): the walk opens the gate, and the slot is uploaded again with
    /// the bytes.
    #[test]
    fn a_partly_mapped_slot_whose_bytes_land_later_is_uploaded_again() {
        crate::receipts::child("descriptor::host::receipts::tests::partly_mapped_child");
    }

    #[test]
    #[ignore = "run by its parent, with receipts on"]
    fn partly_mapped_child() {
        let host = required(crate::host::test_host());
        assert!(crate::receipts::on(), "run by its parent");
        // 512x256 BC7, one level: 128 KiB, of which the first 64 KiB are mapped.
        let (a_at, b_at) = (0x7d_6200_0000usize, 0x7d_6300_0000usize);
        memory(a_at, 256 << 10, 64 << 10);
        memory(b_at, 1 << 20, 1 << 20);
        let (a, b) = (HostHeap::new(false, 8).expect("heap A"), HostHeap::new(false, 8).expect("heap B"));
        let (title_a, title_b) = (Title::new(), Title::new());
        let (resource, view) = placed(a_at, 512, 256, 98, 0);
        title_a.write(&a, 1, view);
        title_b.write(&b, 1, placed(b_at, 64, 64, 98, 0).1);
        assert_eq!(pass(host, &b, &title_b, 10), [1]);
        assert_eq!(pass(host, &a, &title_a, 10), [1], "uploaded with what is mapped");
        assert_eq!(a.texture_slot_mapped(1), 64 << 10, "in part");
        assert_eq!(pass(host, &a, &title_a, 11), [] as [u32; 0], "no more mapped: passed over");
        storage(a_at, 64 << 10, 0x77);
        assert_eq!(pass(host, &b, &title_b, 12), [] as [u32; 0], "B takes the notice");
        let this = texture(a_at, &resource);
        assert!(!read_back(host, this).contains(&0x77), "the copy is from before the bytes");
        let again: Vec<u32> = (13..20).flat_map(|frame| pass(host, &a, &title_a, frame)).collect();
        assert_eq!(again, [1], "uploaded again");
        assert!(read_back(host, this).contains(&0x77), "with the bytes that landed");
    }

    /// The title writes a slot's view while a table holds the slot (no span is kept for
    /// it), its draw reads it through the table and uploads it (out of the queue); then
    /// storage writes the texture's memory, and from then on shaders pick the slot by index
    /// alone. The walk finds the copy older than the write and the pass uploads it again.
    #[test]
    fn a_tabled_slot_picked_by_index_later_is_refreshed_after_a_write() {
        crate::receipts::child("descriptor::host::receipts::tests::tabled_then_index_child");
    }

    #[test]
    #[ignore = "run by its parent, with receipts on"]
    fn tabled_then_index_child() {
        let host = required(crate::host::test_host());
        assert!(crate::receipts::on(), "run by its parent");
        let at = 0x7d_6400_0000usize;
        memory(at, 1 << 20, 1 << 20);
        fill(at, 4096, 0x11);
        let heap = HostHeap::new(false, 8).expect("a heap");
        let title = Title::new();
        let view = placed(at, 64, 64, 98, 0).1;
        // A table holds the slot (frame 0, the frame the title's writes see) when the
        // title writes it: no span.
        heap.tabled_in(1, false, crate::frame::current());
        title.write(&heap, 1, view);
        assert!(!heap.placed.lock().unwrap().spanned_at(at as u64), "no span for a table's slot");
        // Its draw reads it through the table and uploads it.
        heap.tabled_in(1, true, crate::frame::current());
        assert_eq!(heap.texture_slots_due(u64::MAX, 8).0, [] as [(u32, u64); 0], "out of the queue");
        let this = upload(host, &heap, &title, 1).expect("the draw's upload");
        assert!(read_back(host, this).iter().all(|&byte| byte == 0x11));
        storage(at, 4096, 0x77);
        // Long after the last table: picked by index alone.
        let frame = crate::frame::current() + 300;
        let again: Vec<u32> = (frame..frame + 4).flat_map(|frame| pass(host, &heap, &title, frame)).collect();
        assert_eq!(again, [1], "uploaded again");
        assert!(read_back(host, this).iter().all(|&byte| byte == 0x77));
    }

    /// Four slots written over by storage, the notices taken by heap B: a view `alias`
    /// fills, a texture the host GPU wrote, a render target, and an eligible texture. The
    /// walk owes the last alone, and counts each of the others by name.
    #[test]
    fn an_alias_a_host_written_texture_and_a_target_are_not_owed() {
        crate::receipts::child("descriptor::host::receipts::tests::exclusions_child");
    }

    #[test]
    #[ignore = "run by its parent, with receipts on"]
    fn exclusions_child() {
        let host = required(crate::host::test_host());
        assert!(crate::receipts::on(), "run by its parent");
        let base = 0x7d_6500_0000usize;
        let at = |k: usize| base + k * (1 << 20);
        for k in 0..5 {
            memory(at(k), 1 << 20, 1 << 20);
            fill(at(k), 16 << 10, 0x11);
        }
        let (a, b) = (HostHeap::new(false, 8).expect("heap A"), HostHeap::new(false, 8).expect("heap B"));
        let (title_a, title_b) = (Title::new(), Title::new());
        // 1: eligible. 2: an R9G9B9E5 view in the memory of an R32_UINT target. 3: written
        // by the host GPU after its upload. 4: a render target's own view.
        title_a.write(&a, 1, placed(at(1), 64, 64, 98, 0).1);
        let target = placed(at(2), 64, 64, 42, 1).0;
        crate::resource::texture_at(at(2) as u64, &target).expect("the target");
        let (aliased, view) = placed(at(2), 64, 64, 67, 0);
        title_a.write(&a, 2, view);
        let (host_written, view) = placed(at(3), 64, 64, 98, 0);
        title_a.write(&a, 3, view);
        title_a.write(&a, 4, placed(at(4), 64, 64, 42, 1).1);
        title_b.write(&b, 1, placed(at(0), 64, 64, 98, 0).1);
        pass(host, &b, &title_b, 10);
        let mut first = pass(host, &a, &title_a, 10);
        first.sort_unstable();
        assert_eq!(first, [1, 2, 3, 4], "each slot handled once");
        assert!(crate::resource::alias_of(texture(at(2), &aliased)).is_some(), "an alias");
        crate::texture::host_written(texture(at(3), &host_written));
        for k in 1..5 {
            storage(at(k), 4096, 0x77);
        }
        pass(host, &b, &title_b, 11);
        // Two passes a sweep's time apart: each walks all of the heap's slots.
        let mut owed = Vec::new();
        for frame in 12..14 {
            std::thread::sleep(std::time::Duration::from_millis(1100));
            a.receipts_walk(frame);
            owed.extend(a.texture_slots_due(frame, 8).0.into_iter().map(|(slot, _)| slot));
        }
        owed.sort_unstable();
        owed.dedup();
        assert_eq!(owed, [1], "the eligible slot alone is owed");
        assert_eq!(a.walk_last().excluded, [1, 1, 1], "alias, host written, target");
        // And so says the stats file's object.
        let stats = crate::receipts::stats_json();
        assert!(stats.contains("\"excluded\":{\"alias\":1,\"host_written\":1,\"target\":1}"), "{stats}");
    }

    /// A texture tables read, in view memory with only its first block mapped (its draws
    /// keep a staging buffer). Storage writes the mapped block, and the draw's next refresh,
    /// inside a partly mapped texture's 30-frame wait, reads nothing although the guard saw
    /// the write. The walk owes the slot and the pass uploads it again: that refresh must
    /// read, judged by the receipt. (Judged by the check before, it skipped, copied the old
    /// staging buffer and installed the old receipt again, at every sweep.)
    #[test]
    fn a_view_texture_owed_after_a_skipped_read_is_read_again() {
        crate::receipts::child("descriptor::host::receipts::tests::skipped_read_child");
    }

    #[test]
    #[ignore = "run by its parent, with receipts on"]
    fn skipped_read_child() {
        let host = required(crate::host::test_host());
        assert!(crate::receipts::on(), "run by its parent");
        // Not a hash turn at the tests' frame 0: the address's 64 KiB number is not a
        // multiple of 8 (`Staging::refresh_timed`).
        let at = 0x7d_7201_0000usize;
        view_memory(at, 2, 1);
        fill(at, 64 << 10, 0x11);
        let heap = HostHeap::new(false, 8).expect("a heap");
        let title = Title::new();
        // 512x256 BC7, one level: 128 KiB, its first block mapped. A table's slot: out of
        // the queue once its draw reads it.
        let view = placed(at, 512, 256, 98, 0).1;
        heap.tabled_in(1, false, crate::frame::current());
        title.write(&heap, 1, view);
        heap.tabled_in(1, true, crate::frame::current());
        let this = upload(host, &heap, &title, 1).expect("the draw's upload");
        let back = read_back(host, this);
        assert!(back.contains(&0x11) && !back.contains(&0x77), "the first copy");
        storage(at, 64 << 10, 0x77);
        upload(host, &heap, &title, 1).expect("the draw's next upload");
        assert!(!read_back(host, this).contains(&0x77), "the refresh inside the wait read nothing");
        let frame = crate::frame::current() + 300;
        let again: Vec<u32> = (frame..frame + 4).flat_map(|frame| pass(host, &heap, &title, frame)).collect();
        assert!(read_back(host, this).contains(&0x77), "read again: the copy has the storage read's bytes");
        assert_eq!(again, [1], "uploaded once, not at every sweep");
    }

    /// Debt that nothing named held: a texture whose host copy the host refused (not owed
    /// again), one whose staging buffer could not be had (owed again), one the pass's budget
    /// left; each is counted by name. Of two textures the host GPU wrote, the one whose
    /// memory storage wrote again after is counted as host-written and written since.
    #[test]
    fn blockers_are_named_and_host_written_textures_written_since_are_counted() {
        crate::receipts::child("descriptor::host::receipts::tests::named_blockers_child");
    }

    #[test]
    #[ignore = "run by its parent, with receipts on"]
    fn named_blockers_child() {
        use crate::receipts::{BUDGET, NO_STAGING_BLOCKER, REFUSED};
        let host = required(crate::host::test_host());
        assert!(crate::receipts::on(), "run by its parent");
        let base = 0x7d_7300_0000usize;
        let at = |k: usize| base + k * (1 << 20);
        for k in 0..6 {
            memory(at(k), 1 << 20, 1 << 20);
            fill(at(k), 128 << 10, 0x11);
        }
        let none: [(u32, u64); 0] = [];
        // BC7 of four samples, which the host does not make: its host copy is refused.
        let refused = HostHeap::new(false, 8).expect("a heap");
        let t = Title::new();
        let (mut resource, _) = placed(at(0), 64, 64, 98, 0);
        resource[9] = 4;
        let mut desc = [98, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        (desc[10], desc[11]) = (at(0) as u32, (at(0) as u64 >> 32) as u32);
        t.write(&refused, 1, View::Placed(false, resource, desc));
        pass(host, &refused, &t, 10);
        assert!(crate::resource::host_resource(texture(at(0), &resource)).is_none(), "refused");
        assert_eq!(sweep(&refused, 11).blockers[REFUSED], 1, "a refused host copy");
        assert_eq!(refused.texture_slots_due(u64::MAX, 8).0, none, "not owed again");
        // No staging buffer to be had at its upload.
        let lean = HostHeap::new(false, 8).expect("a heap");
        let t = Title::new();
        t.write(&lean, 1, placed(at(1), 64, 64, 98, 0).1);
        crate::texture::FAIL_STAGING.store(true, Relaxed);
        pass(host, &lean, &t, 10);
        crate::texture::FAIL_STAGING.store(false, Relaxed);
        assert_eq!(sweep(&lean, 11).blockers[NO_STAGING_BLOCKER], 1, "no staging buffer");
        assert!(lean.texture_slots_due(u64::MAX, 8).0.iter().any(|&(slot, _)| slot == 1), "owed again");
        // Two slots written over by storage; the pass's budget serves one.
        let busy = HostHeap::new(false, 8).expect("a heap");
        let t = Title::new();
        t.write(&busy, 1, placed(at(2), 64, 64, 98, 0).1);
        t.write(&busy, 2, placed(at(3), 64, 64, 98, 0).1);
        assert_eq!(pass(host, &busy, &t, 10).len(), 2);
        storage(at(2), 4096, 0x77);
        storage(at(3), 4096, 0x77);
        std::thread::sleep(std::time::Duration::from_millis(1100));
        assert_eq!(pass_within(host, &busy, &t, 20, 1).len(), 1, "one served");
        assert_eq!(sweep(&busy, 21).blockers[BUDGET], 1, "the other held by the budget");
        // Two textures the host GPU wrote after their uploads; storage writes one again.
        let copied = HostHeap::new(false, 8).expect("a heap");
        let t = Title::new();
        let (first, view) = placed(at(4), 64, 64, 98, 0);
        t.write(&copied, 1, view);
        let (second, view) = placed(at(5), 64, 64, 98, 0);
        t.write(&copied, 2, view);
        assert_eq!(pass(host, &copied, &t, 10).len(), 2);
        crate::texture::host_written(texture(at(4), &first));
        crate::texture::host_written(texture(at(5), &second));
        storage(at(4), 4096, 0x77);
        let s = sweep(&copied, 11);
        assert_eq!(s.excluded[1], 2, "both host-written");
        assert_eq!(s.host_written_newer, 1, "one written since");
    }

    /// G-006: without a host device a receipts fixture fails; it does not pass by skipping.
    #[test]
    #[should_panic(expected = "no host device")]
    fn a_receipts_fixture_without_a_device_fails() {
        let _ = required(None);
    }
}
