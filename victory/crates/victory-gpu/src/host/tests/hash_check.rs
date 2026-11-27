//! `VICTORY_HASH_CHECK=1` (`host::hash_check`): the counters and the stats fields; the flights
//! sampled (never in step with a frame's count of them); a flight over the budget sampled
//! within it; and, on the software adapter in a process of its own, the check's look itself:
//! it compares only what the copy-back did not fetch, never merges, and skips with a count
//! where it has no memory.

use super::*;
use std::sync::Mutex;

/// One test at a time over the process-wide counters.
static LOCK: Mutex<()> = Mutex::new(());

/// A forced miss (the test hook) is counted, an equal piece is checked but
/// not missed, and a difference is a miss.
#[test]
fn a_forced_miss_is_counted() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (missed, checked, skipped) = figures();
    force_miss(true);
    count_piece(false);
    assert_eq!(figures(), (missed + 1, checked + 1, skipped), "the forced miss");
    count_piece(false);
    assert_eq!(figures(), (missed + 1, checked + 2, skipped), "an equal piece");
    count_piece(true);
    assert_eq!(figures(), (missed + 2, checked + 3, skipped), "a difference");
}

/// The stats file's fields appear only with the switch on, all three.
#[test]
fn the_fields_are_only_said_when_on() {
    assert_eq!(fields(false), "");
    let text = fields(true);
    assert!(text.contains("\"hash_missed_pieces_total\":"), "{text}");
    assert!(text.contains("\"hash_checked_pieces_total\":"), "{text}");
    assert!(text.contains("\"hash_check_skipped_total\":"), "{text}");
}

/// Off, no flight is sampled and the count of flights does not move.
#[test]
fn off_no_flight_is_sampled() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if on() {
        return; // a run with the switch on is not this test
    }
    assert!(!sample(), "off: nothing is sampled");
    assert_eq!(FLIGHTS.load(Ordering::Relaxed), 0, "off: nothing is counted");
}

/// Opus's review of `job/hash-wide-fix`, 5: a count `% 16` falls in step with a fixed number of
/// flights a frame and samples the same passes every frame. For every count of 1 to 64 flights
/// a frame, over 640 frames, every place in the frame is sampled at least once, and about one
/// flight in 16 overall.
#[test]
fn the_sample_cannot_fall_in_step_with_the_flights_a_frame() {
    for per_frame in 1..=64u64 {
        let mut places = vec![0u32; per_frame as usize];
        let mut n = 0u64;
        for _ in 0..640 {
            for place in &mut places {
                *place += u32::from(picked(n));
                n += 1;
            }
        }
        let never: Vec<usize> = (0..places.len()).filter(|&i| places[i] == 0).collect();
        assert!(never.is_empty(), "{per_frame} flights a frame: places {never:?} never sampled");
        let all: u32 = places.iter().sum();
        let flights = 640 * per_frame;
        assert!(
            u64::from(all) * 32 >= flights && u64::from(all) * 8 <= flights,
            "{per_frame} a frame: {all} of {flights} sampled"
        );
    }
}

/// Opus's review, 5: a sampled flight fetched its whole declared ranges, up to about 1 GiB. Over
/// the budget the pieces are sampled within the flight: at most the budget's bytes, each piece
/// once, spread over the flight, and from another place the next time; under it, all of them.
#[test]
fn a_flight_over_the_budget_is_sampled_within_it() {
    let piece = 65_536u64;
    assert_eq!(within(&[piece; 100], BUDGET, 7), (0..100).collect::<Vec<_>>(), "under the budget: all");
    for n in [300usize, 1000, 7919, 16_384] {
        let sizes: Vec<u64> = (0..n).map(|i| if i % 3 == 0 { piece } else { piece / 2 }).collect();
        let first = within(&sizes, BUDGET, 0);
        let bytes: u64 = first.iter().map(|&i| sizes[i]).sum();
        assert!(bytes <= BUDGET, "{n} pieces: {bytes} bytes");
        assert!(bytes + piece > BUDGET, "{n} pieces: the budget is filled ({bytes} bytes)");
        assert!(first.windows(2).all(|w| w[0] < w[1]), "{n} pieces: each once, in order");
        assert!(*first.last().unwrap() > n / 2, "{n} pieces: spread over the flight, not its front");
        assert_ne!(within(&sizes, BUDGET, mix(1)), first, "{n} pieces: another start, other pieces");
    }
}

/// Opus's review, 1 and 5, on the software adapter: the check's look at a sampled plan compares
/// only the pieces the copy-back did not fetch, counts a difference, never merges, skips with a
/// count where it has no memory and samples within the budget. In a process of its own (it
/// owns dwords of the compare's results and adds a title buffer).
#[test]
fn the_look_compares_what_was_not_fetched_and_merges_nothing() {
    let name = "host::hash_check::tests::look_child";
    let exe = std::env::current_exe().expect("the test binary");
    let out = std::process::Command::new(exe)
        .args(["--exact", name, "--ignored", "--test-threads", "1", "--nocapture"])
        .env_remove("VICTORY_LIVE")
        .env_remove("VICTORY_FETCH_QUEUE")
        .output()
        .expect("the child ran");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success() && text.contains("1 passed"), "the child failed:\n{text}");
}

/// The child of [`the_look_compares_what_was_not_fetched_and_merges_nothing`].
#[test]
#[ignore = "a child process of the_look_compares_what_was_not_fetched_and_merges_nothing"]
fn look_child() {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    let host = crate::host::test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
    assert!(host.card_ready(), "no compare: the fixture proves nothing");
    // A zeroed title buffer of 16 MiB (256 pieces, twice the budget), its copy in the card's
    // memory: both sides zero.
    let len = 16usize << 20;
    // SAFETY: our own process's memory, kept to the end of the process.
    let at = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
    assert_ne!(at, 0);
    crate::shadow::register(0x5EED_0003, at as u64, len as u64);
    let copy = crate::shadow::find(at as u64, len as u64).expect("a copy");
    assert!(copy.carded(), "the copy is not in the card's memory: the fixture proves nothing");
    // Dwords of the compare's results, taken as a plan takes them.
    let (differs, slot) = {
        let mut work = crate::host::CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let work = work.as_mut().expect("the card's work");
        let slot = work.taken as usize;
        work.taken += 2 + 256;
        (work.differs.1 as *mut u32, slot)
    };
    // SAFETY: inside the results' mapping; these dwords are ours.
    let set = |i: usize, flag: u32| unsafe { *differs.add(slot + i) = flag };
    // SAFETY: as above, read only between the test's own writes.
    let results = || unsafe { std::slice::from_raw_parts(differs as *const u32, slot + 2 + 256) };
    let range = |from: usize, to: usize, i: usize| crate::host::CardRange {
        shadow: copy.clone(),
        declared: (at + from, at + to),
        span: copy.card_span(at + from, at + to).expect("inside the copy"),
        at: (slot + i) as u32,
        compared: true,
    };
    let plan = |ranges: Vec<crate::host::CardRange>, sampled: bool| CardPlan {
        ranges,
        check: None,
        _synced: Vec::new(),
        lists: Vec::new(),
        #[cfg(feature = "diagnostics")]
        stamps: None,
        fetch: None,
        hash_check: sampled,
    };
    // SAFETY: our own title buffer.
    let title = |offset: usize| unsafe { (at as *mut u8).add(offset) };
    // Range A in piece 0, flagged (the copy-back fetches it); range B in piece 1, not flagged.
    set(0, 1);
    set(1, 0);
    let two = plan(vec![range(0x100, 0x200, 0), range(0x1_0100, 0x1_0200, 1)], true);
    let (m, c, s) = figures();
    look(host, &[&two], results(), false);
    assert_eq!(figures(), (m, c + 1, s), "B compared and equal; A, which the copy-back fetches, not");
    // B's CPU side moves away from the card's, as a write the compare missed would leave them.
    // SAFETY: our own title buffer.
    unsafe { *title(0x1_0180) = 0x5A };
    assert!(copy.update(at + 0x1_0180, 1) >= 1, "the CPU side took the byte");
    look(host, &[&two], results(), false);
    assert_eq!(figures(), (m + 1, c + 2, s), "the difference is counted");
    // SAFETY: as above.
    assert_eq!(unsafe { *title(0x1_0180) }, 0x5A, "and nothing was merged");
    // A plan not sampled: nothing at all.
    look(host, &[&plan(vec![range(0x1_0100, 0x1_0200, 1)], false)], results(), false);
    assert_eq!(figures(), (m + 1, c + 2, s), "not sampled");
    // The relay's form (the copy queue not engaged here): the same.
    look(host, &[&two], results(), true);
    assert_eq!(figures(), (m + 2, c + 3, s), "the relay's form");
    // No memory for the check: skipped and counted, and the process goes on.
    *ROOM.lock().unwrap_or_else(|e| e.into_inner()) = None;
    NO_ROOM.store(true, Ordering::Relaxed);
    look(host, &[&two], results(), false);
    assert_eq!(figures(), (m + 2, c + 3, s + 1), "no memory: one piece skipped");
    // The whole copy, nothing flagged: 256 pieces, 16 MiB; 128 compared, 128 skipped. The
    // CPU side is put back first, so none differs.
    // SAFETY: as above.
    unsafe { *title(0x1_0180) = 0 };
    assert!(copy.update(at + 0x1_0180, 1) >= 1);
    for i in 0..256 {
        set(2 + i, 0);
    }
    let whole = plan(vec![range(0, len, 2)], true);
    assert_eq!(whole.ranges[0].span.pieces, 256);
    look(host, &[&whole], results(), false);
    assert_eq!(figures(), (m + 2, c + 3 + 128, s + 1 + 128), "over the budget: half compared, half skipped");
}
