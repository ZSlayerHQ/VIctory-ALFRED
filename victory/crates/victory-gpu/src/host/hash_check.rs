//! `VICTORY_HASH_CHECK=1`: a run-time check of the card hash against the
//! bytes themselves.
//!
//! The card's compare hashes every declared piece and flags a piece whose
//! hash changed, and the copy-back fetches only what it flagged. A hash
//! collision would leave a shader's write unfetched: it would never reach
//! title memory, and nothing else would say so.
//!
//! With this switch on (read once at start), about one flight in 16 with
//! declared ranges is sampled, picked by a small generator of a counter
//! ([`picked`]), so the sample cannot fall in step with the number of
//! flights a frame. Its copy-back lands exactly as with the switch off: the
//! check adds no piece to it and merges nothing. After it, the pieces of
//! the sampled flight's ranges that the copy-back did not fetch (by its own
//! rule: a piece flagged under no range of the plan) are fetched by a list
//! of the check's own into memory of its own and compared with the CPU
//! side's bytes, which are what the last fetch or send left there.
//! [`CHECKED`] counts the pieces compared, [`MISSED`] those whose bytes
//! differ, and [`SKIPPED`] those not compared: over the budget (at most
//! [`BUDGET`] bytes a landing; over it, pieces are sampled within the
//! flight, [`within`]); no memory, list or fence for the check (a failed
//! allocation skips, it never ends the process); the check's memory in use
//! by the other thread; on the relay, the copy queue engaged
//! (`VICTORY_FETCH_QUEUE` on or its tickets out: a fetch on the drawing
//! queue could wait behind a writer held for a ticket that only the relay,
//! waiting for it, would land); or a fetch that could not be sent or ended
//! on a removed device.
//!
//! What a count means. With the relay on (the default), the check's fetch
//! runs behind whatever was sent since the flight, as the relay's own fetch
//! does, so it can carry a later flight's writes (and title bytes sent up
//! since) and count them as misses. A non-zero `hash_missed_pieces_total`
//! from such a run is a lead to look into, not a collision shown. A clean proof needs
//! a run with the relay off (`VICTORY_RELAY=0`, `VICTORY_LATE_WAIT` unset):
//! each submission is then waited for alone, and the fetch follows its own
//! fetch with nothing of the card's in between (with `VICTORY_LATE_WAIT`,
//! flights settled together share one wait, and the same caveat holds). In
//! either mode the title's own CPU writes to those bytes since the last
//! send would count too.
//!
//! The numbers are in the live stats file (`hash_checked_pieces_total`,
//! `hash_missed_pieces_total`, `hash_check_skipped_total`: counts since the
//! start), and a line is logged every 600 presented frames. With the switch off nothing is sampled, made,
//! fetched or counted, and the stats file has no such fields.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::CreateEventW;
use windows::core::Interface;

use super::{CardPlan, Host};
use crate::shadow::Shadow;

/// Whether the check is on: `VICTORY_HASH_CHECK=1`, read once at start, off
/// by default.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HASH_CHECK").is_some_and(|v| v == "1"))
}

/// The most bytes one landing's check fetches: 8 MiB, 128 whole pieces.
pub(super) const BUDGET: u64 = 8 << 20;

/// Flights with declared ranges offered so far ([`sample`]).
static FLIGHTS: AtomicU64 = AtomicU64::new(0);
/// Checks over the budget so far: where [`within`] starts.
static STARTS: AtomicU64 = AtomicU64::new(0);
/// Pieces fetched by the check and compared with the CPU side.
static CHECKED: AtomicU64 = AtomicU64::new(0);
/// Of those, the pieces whose bytes differ there.
static MISSED: AtomicU64 = AtomicU64::new(0);
/// Pieces of sampled flights the copy-back did not fetch and the check did
/// not compare either (see the module's note).
static SKIPPED: AtomicU64 = AtomicU64::new(0);
/// Presented frames, for the line every 600.
static FRAMES: AtomicU64 = AtomicU64::new(0);

/// splitmix64 of `n`: a small generator, one 64-bit value a step.
fn mix(n: u64) -> u64 {
    let mut z = n.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Whether the `n`-th flight with declared ranges is sampled: one in 16 on
/// average, by the generator's top four bits, so no number of flights a
/// frame takes the same passes every frame (a count `% 16` would: with 16
/// flights a frame it sampled the same pass in each).
fn picked(n: u64) -> bool {
    mix(n) >> 60 == 0
}

/// Whether a flight with declared ranges is one of the sampled ones. Off:
/// never, and no count moves.
pub(crate) fn sample() -> bool {
    on() && picked(FLIGHTS.fetch_add(1, Ordering::Relaxed))
}

/// Which of pieces of `sizes` bytes a check fetches within `budget`, in
/// order: all if they fit; else from `start` on with a stride prime to their
/// count (7,919 or 7,907; a count both divide would be 62 million pieces), so
/// that every piece is visited once and those taken are spread over the
/// flight, each taken while it fits.
fn within(sizes: &[u64], budget: u64, start: u64) -> Vec<usize> {
    let n = sizes.len();
    if sizes.iter().sum::<u64>() <= budget {
        return (0..n).collect();
    }
    let stride = [7919usize, 7907].into_iter().find(|&p| !n.is_multiple_of(p)).unwrap_or(1);
    let first = (start % n as u64) as usize;
    let mut used = 0u64;
    let mut taken = Vec::new();
    for j in 0..n {
        let i = (first + j * stride) % n;
        if used + sizes[i] <= budget {
            used += sizes[i];
            taken.push(i);
        }
    }
    taken.sort_unstable();
    taken
}

/// A piece the copy-back did not fetch: its copy and its offsets from and to.
type Unfetched = (Arc<Shadow>, usize, usize);

/// The pieces of the sampled ones of `plans` that their copy-back does not
/// fetch: under every range, the pieces flagged under no range of the plan
/// (`super::flagged_pieces`, read as `fetch::pieces` reads the results;
/// a range not compared is fetched whole), each once.
fn unfetched(plans: &[&CardPlan], differs: &[u32]) -> Vec<Unfetched> {
    let mut out: Vec<Unfetched> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for plan in plans.iter().filter(|plan| plan.hash_check) {
        let flags: Vec<(usize, u32, bool)> = plan
            .ranges
            .iter()
            .flat_map(|r| {
                (0..r.span.pieces).map(move |p| {
                    let flagged = !r.compared || differs.get((r.at + p) as usize).is_none_or(|&d| d != 0);
                    (Arc::as_ptr(&r.shadow) as usize, r.span.first + p, flagged)
                })
            })
            .collect();
        let fetched = super::flagged_pieces(&flags);
        for r in &plan.ranges {
            let copy = Arc::as_ptr(&r.shadow) as usize;
            for p in 0..r.span.pieces {
                if fetched.contains(&(copy, r.span.first + p)) {
                    continue;
                }
                let (lo, hi) = r.shadow.card_piece(&r.span, p);
                if lo < hi && seen.insert((copy, lo, hi)) {
                    out.push((r.shadow.clone(), lo, hi));
                }
            }
        }
    }
    out
}

/// The check's own list, fence, event and [`BUDGET`] bytes of PC memory the
/// pieces come down to; made at the first sample that needs them.
struct Room {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    event: HANDLE,
    sent: u64,
    buffer: ID3D12Resource,
    data: usize,
}

// SAFETY: D3D12 objects (free-threaded), an event and a persistent mapping,
// used only under `ROOM`'s lock.
unsafe impl Send for Room {}

impl Drop for Room {
    fn drop(&mut self) {
        // SAFETY: our event.
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(self.event);
        }
    }
}

/// One landing at a time has the room; the other thread skips.
static ROOM: Mutex<Option<Room>> = Mutex::new(None);

/// Tests: the next room cannot be made, as if its memory could not be had.
#[cfg(test)]
static NO_ROOM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn make_room(host: &Host) -> Option<Room> {
    #[cfg(test)]
    if NO_ROOM.swap(false, Ordering::Relaxed) {
        return None;
    }
    let (allocator, list) = host.make_prelude()?;
    let (buffer, data) = host.shared_buffer(BUDGET)?;
    super::name(&buffer, || "hash check fetch (VICTORY_HASH_CHECK=1)".into());
    // SAFETY: plain object creation.
    let fence: ID3D12Fence = unsafe { host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.ok()?;
    // SAFETY: plain object creation.
    let event = unsafe { CreateEventW(None, false, false, None) }.ok()?;
    Some(Room {
        allocator,
        list,
        fence,
        event,
        sent: 0,
        buffer,
        data: data as usize,
    })
}

/// What became of a check's fetch.
enum Fetch {
    /// It ran: its bytes are in the room.
    Done,
    /// Nothing of it went to the queue: the room is free again.
    NotSent,
    /// Sent, and not seen to end on a live device: the room is never used
    /// (nor freed) again.
    Lost,
}

/// Sends the copies of `pieces` (each with where it comes down in the room)
/// on the drawing queue and waits for them. `relay`: on the relay's thread,
/// where the send is made under the queue's order and only with the copy
/// queue not engaged.
fn fetch(host: &Host, room: &mut Room, pieces: &[(&Unfetched, u64)], relay: bool) -> Fetch {
    // SAFETY: the room's list ran to its fence before (or never ran); live
    // buffers (the plans hold the copies); the offsets lie in them.
    let recorded = unsafe {
        (|| -> windows::core::Result<ID3D12CommandList> {
            room.allocator.Reset()?;
            room.list.Reset(&room.allocator, None)?;
            // Named for outside GPU tools (`crate::events`); no timestamps of ours cover it.
            crate::events::begin(&room.list, crate::events::HASH_CHECK);
            for ((shadow, lo, hi), at) in pieces {
                room.list.CopyBufferRegion(&room.buffer, *at, &shadow.buffer, *lo as u64, (hi - lo) as u64);
            }
            crate::events::end(&room.list);
            room.list.Close()?;
            room.list.cast::<ID3D12CommandList>()
        })()
    };
    let Ok(list) = recorded else {
        return Fetch::NotSent;
    };
    let order = relay.then(|| super::QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
    if relay && super::fetch::engaged() {
        return Fetch::NotSent;
    }
    room.sent += 1;
    // SAFETY: a closed list of ours; the host's queue and our fence.
    let signalled = unsafe {
        host.queue.ExecuteCommandLists(&[Some(list)]);
        host.queue.Signal(&room.fence, room.sent)
    };
    drop(order);
    if signalled.is_err() {
        return Fetch::Lost;
    }
    match super::fetch_waited(&host.device, &room.fence, room.event, room.sent) {
        Ok(true) => Fetch::Done,
        _ => Fetch::Lost,
    }
}

/// After the copy-back of `plans` has landed (`relay`: on the relay's
/// thread; else a submission settled under the submission lock), with
/// `differs` their compare's results: the check's look at the sampled ones.
/// Nothing with no plan sampled.
pub(super) fn look(host: &Host, plans: &[&CardPlan], differs: &[u32], relay: bool) {
    if !plans.iter().any(|plan| plan.hash_check) {
        return;
    }
    let all = unfetched(plans, differs);
    if all.is_empty() {
        return;
    }
    let sizes: Vec<u64> = all.iter().map(|(_, lo, hi)| (hi - lo) as u64).collect();
    let taken = within(&sizes, BUDGET, mix(STARTS.fetch_add(1, Ordering::Relaxed)));
    SKIPPED.fetch_add((all.len() - taken.len()) as u64, Ordering::Relaxed);
    let skip = || {
        SKIPPED.fetch_add(taken.len() as u64, Ordering::Relaxed);
    };
    let Ok(mut room) = ROOM.try_lock() else {
        return skip();
    };
    if room.is_none() {
        *room = make_room(host);
    }
    let Some(r) = room.as_mut() else {
        return skip();
    };
    let data = r.data;
    let mut at = 0u64;
    let pieces: Vec<(&Unfetched, u64)> = taken
        .iter()
        .map(|&i| {
            let here = at;
            at += sizes[i];
            (&all[i], here)
        })
        .collect();
    match fetch(host, r, &pieces, relay) {
        Fetch::Done => {}
        Fetch::NotSent => return skip(),
        Fetch::Lost => {
            std::mem::forget(room.take());
            return skip();
        }
    }
    for ((shadow, lo, hi), at) in &pieces {
        // SAFETY: the room's mapping; the copy into it has run.
        let gpu = unsafe { std::slice::from_raw_parts((data + *at as usize) as *const u8, hi - lo) };
        count_piece(shadow.card_bytes_differ(*lo, gpu));
    }
}

/// [`look`] for a relay landing: the results read as `fetch::pieces` reads
/// them. The plans' dwords are theirs until `Host::card_done`.
pub(super) fn look_landed(host: &Host, plans: &[&CardPlan]) {
    if !plans.iter().any(|plan| plan.hash_check) {
        return;
    }
    let differs = {
        let work = super::CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        work.as_ref().map(|work| (work.differs.1, (work.differs.2 / 4) as usize))
    };
    if let Some((differs, dwords)) = differs {
        // SAFETY: the mapping of the results; the compares of these plans
        // have run, and their dwords are given to no other plan before
        // `card_done`.
        look(host, plans, unsafe { std::slice::from_raw_parts(differs as *const u32, dwords) }, true);
    }
}

/// One piece the check compared: `differs`, whether its fetched bytes
/// differ from the CPU side's.
fn count_piece(differs: bool) {
    CHECKED.fetch_add(1, Ordering::Relaxed);
    if miss_hook(differs) {
        MISSED.fetch_add(1, Ordering::Relaxed);
    }
}

/// The three figures (missed, checked, skipped), for the stats file and the
/// line.
pub(crate) fn figures() -> (u64, u64, u64) {
    (MISSED.load(Ordering::Relaxed), CHECKED.load(Ordering::Relaxed), SKIPPED.load(Ordering::Relaxed))
}

/// The stats file's fields, when the check is on: counts since the start, so named
/// `_total` (`host/stats.rs`); nothing when it is off.
pub(crate) fn fields(on: bool) -> String {
    if !on {
        return String::new();
    }
    let (missed, checked, skipped) = figures();
    format!(",\"hash_missed_pieces_total\":{missed},\"hash_checked_pieces_total\":{checked},\"hash_check_skipped_total\":{skipped}")
}

/// One presented frame: every 600th says the figures.
pub(crate) fn frame() {
    let n = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_multiple_of(600) {
        let (missed, checked, skipped) = figures();
        tracing::info!(
            "hash check (VICTORY_HASH_CHECK=1): {checked} pieces the copy-back did not fetch fetched apart and compared with \
             the CPU side, {missed} of them differ, {skipped} not compared (budget, memory, or the copy queue engaged); with \
             the relay on a difference can be a later flight's write, a lead and not a proof"
        );
    }
}

/// Tests: the next [`count_piece`] reports a difference even if the bytes
/// did not, so the counting path is exercised without a card.
#[cfg(test)]
static FORCE_MISS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
pub(crate) fn force_miss(once: bool) {
    FORCE_MISS.store(once, Ordering::Relaxed);
}

/// The difference the counter sees: the caller's, or the test hook's once.
fn miss_hook(differs: bool) -> bool {
    #[cfg(test)]
    if FORCE_MISS.swap(false, Ordering::Relaxed) {
        return true;
    }
    differs
}

#[cfg(test)]
#[path = "tests/hash_check.rs"]
mod tests;
