//! Frame pacing: a guest sets a frame interval and waits for each frame
//! to begin. Ours is a clock started by the first interval set, ticking
//! once per interval; the token is the frame number. Each event kind
//! (origin, vblank, flip) happens once per tick, so two waits for the same
//! kind never end on the same tick. A wait may return on the token's
//! time-out (pacing by polling) or block until the tick (pacing by the wait).
//!
//! shortcut: event offsets are ignored (every event is at the tick); tie
//! the clock to presentation if a title paces by them.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::object::{HRESULT, S_FALSE, S_OK, This};

/// What a wait returns when its time is up first.
const TIMEOUT: HRESULT = S_FALSE;
/// The frame-origin event: the one that starts frames and makes tokens.
const ORIGIN: u32 = 0;
/// Make the frame's token even if the wait times out (the title starts the
/// frame early).
const TOKEN_ON_TIMEOUT: u32 = 2;
const SIXTIETH: Duration = Duration::from_micros(16_667);

struct Clock {
    start: Instant,
    interval: Duration,
    /// The tick number at `start` (ticks go on across interval changes).
    first: u64,
}

static CLOCK: Mutex<Option<Clock>> = Mutex::new(None);
static FRAME: AtomicU64 = AtomicU64::new(0);

fn clock() -> std::sync::MutexGuard<'static, Option<Clock>> {
    CLOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Device method: a frame every `length` microseconds. The period is how
/// many intervals a frame may take to reach the display (pipeline depth),
/// not part of the rate.
pub(crate) extern "system" fn set_interval(
    _: This,
    _output: This,
    length: u32,
    _period: u32,
    _flags: u32,
) -> HRESULT {
    let _entry = crate::budget::entry(crate::budget::Phase::ObjectCalls);
    let interval = match length {
        0 => SIXTIETH,
        us => Duration::from_micros(u64::from(us)),
    };
    tracing::info!(?interval, "frame interval");
    // A new interval starts from now, after the last tick handed out.
    let first = LAST
        .iter()
        .map(|t| t.load(Ordering::Relaxed))
        .max()
        .unwrap_or(0);
    *clock() = Some(Clock {
        start: Instant::now(),
        interval,
        first,
    });
    S_OK
}

/// Device method: events are scheduled at every tick.
pub(crate) extern "system" fn schedule(
    _: This,
    _kind: u32,
    _offset: u32,
    _list: *const u8,
    _flags: u32,
) -> HRESULT {
    let _entry = crate::budget::entry(crate::budget::Phase::ObjectCalls);
    S_OK
}

/// Frames begun so far (frame waits that ended).
pub(crate) fn current() -> u64 {
    FRAME.load(Ordering::Relaxed)
}

/// Per event kind (origin, vblank, flip and later kinds): the last tick a
/// wait for it ended on.
static LAST: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

fn last(kind: u32) -> &'static AtomicU64 {
    &LAST[(kind as usize).min(LAST.len() - 1)]
}

/// The next tick after now and after the last one a wait for `kind` ended
/// on (an event happens once per tick: a wait woken a little early must not
/// let the next one end on the same tick), and how long until it.
fn next_tick(kind: u32) -> (u64, Duration) {
    let mut clock = clock();
    let c = clock.get_or_insert_with(|| Clock {
        start: Instant::now(),
        interval: SIXTIETH,
        first: 0,
    });
    let elapsed = c.start.elapsed().as_nanos();
    let interval = c.interval.as_nanos().max(1);
    let tick =
        (c.first + (elapsed / interval + 1) as u64).max(last(kind).load(Ordering::Relaxed) + 1);
    let at = u128::from(tick - c.first) * interval;
    (
        tick,
        Duration::from_nanos(at.saturating_sub(elapsed) as u64),
    )
}

/// Device method: blocks until the next tick (or the time-out); an origin
/// wait then hands out the new frame's token. Other events keep the token
/// the title passes. Ticks are absolute (start plus whole intervals), so
/// lateness never accumulates.
pub(crate) extern "system" fn wait(
    _: This,
    kind: u32,
    timeout_ms: u32,
    _list: *const u8,
    flags: u32,
    token: *mut u64,
) -> HRESULT {
    // The title waits here for its frame interval: a wait of its own in the budget.
    let _entry = crate::budget::phase(crate::budget::Phase::PaceWait);
    let origin = kind == ORIGIN;
    let (mut tick, mut wait) = next_tick(kind);
    let limit = Duration::from_millis(u64::from(timeout_ms));
    // A blocking wait for an event that has happened since the last wait
    // for it ended (a frame took longer than a tick) ends at once, on the
    // tick that passed: sleeping on to the next one cost a tick of every
    // frame (two such waits a frame). Polls (a
    // time-out shorter than a tick) answer as before.
    // `VICTORY_LATE_WAITS=1` sleeps to the next tick, as before; with
    // `VICTORY_LIVE` the switch is read at every wait, so a change at a
    // frame boundary reaches the next wait.
    if !crate::live::late_waits()
        && limit >= wait
        && tick > last(kind).load(Ordering::Relaxed) + 1
    {
        tick -= 1;
        wait = Duration::ZERO;
    }
    let hr = if wait > limit {
        std::thread::sleep(limit);
        TIMEOUT
    } else {
        std::thread::sleep(wait);
        last(kind).fetch_max(tick, Ordering::Relaxed);
        S_OK
    };
    #[cfg(all(windows, feature = "diagnostics"))]
    {
        use crate::copystat::{C, add};
        add(C::PaceWaits, 1);
        add(C::PaceUs, wait.min(limit).as_micros() as u64);
    }
    // A frame begun on a time-out takes no tick: its flip still does.
    if origin && (hr == S_OK || flags & TOKEN_ON_TIMEOUT != 0) {
        // Tokens count frames (consecutive), not ticks.
        let frame = FRAME.fetch_add(1, Ordering::Relaxed) + 1;
        if !token.is_null() {
            // SAFETY: checked non-null; the title passes its token.
            unsafe { token.write(frame) };
        }
    }
    #[cfg(feature = "diagnostics")]
    tracing::debug!(
        kind,
        timeout_ms,
        flags,
        hr,
        tick,
        wait_us = wait.as_micros() as u64,
        frame = current(),
        "frame wait"
    );
    hr
}

/// Device method: no statistics are kept, so none are returned.
pub(crate) extern "system" fn statistics(
    _: This,
    _token: u64,
    _kinds: u32,
    count: *mut u32,
    _out: *mut u8,
) -> HRESULT {
    let _entry = crate::budget::entry(crate::budget::Phase::ObjectCalls);
    if !count.is_null() {
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
    }
    S_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether the flips' ticks hold: each later than the one before (no two flips share a
    /// tick, [`next_tick`]) and, where `consecutive`, each the very next one.
    fn flips_hold(flips: &[u64], consecutive: bool) -> bool {
        flips
            .windows(2)
            .all(|w| if consecutive { w[1] == w[0] + 1 } else { w[1] > w[0] })
    }

    /// Whether the flips must take consecutive ticks here. Ticks are absolute ([`wait`]), so a
    /// flip wait woken more than one interval late skips ticks by design; only a CI runner
    /// without a card (`allow_skip`: VICTORY_TESTS_ALLOW_SOFTWARE_SKIP=1; no hardware adapter
    /// listed), whose scheduler wakes the test that late, is let off that one part.
    fn consecutive_here(allow_skip: bool, hardware_listed: bool) -> bool {
        !allow_skip || hardware_listed
    }

    /// Whether DXGI lists a hardware adapter here (a listing that fails counts as one: the
    /// exact check).
    fn hardware_listed() -> bool {
        match victory_shader::adapter::listing() {
            Ok((listing, _)) => listing.adapters.iter().any(|a| !a.software),
            Err(_) => true,
        }
    }

    #[test]
    fn flips_hold_on_later_ticks_and_on_the_next_one_where_consecutive() {
        assert!(flips_hold(&[1, 2, 3, 4], true));
        assert!(!flips_hold(&[9, 17, 22, 23], true), "a skipped tick fails the consecutive check");
        assert!(flips_hold(&[9, 17, 22, 23], false), "later ticks hold without it");
        assert!(!flips_hold(&[3, 3, 4], false), "two flips on one tick fail either way");
        assert!(!flips_hold(&[4, 3], false), "an earlier tick fails either way");
    }

    #[test]
    fn only_a_ci_runner_without_a_card_may_skip_ticks() {
        assert!(consecutive_here(false, true));
        assert!(consecutive_here(false, false), "no card, no variable: exact");
        assert!(consecutive_here(true, true), "the variable on a machine with a card: exact");
        assert!(!consecutive_here(true, false), "a CI runner without a card");
    }

    #[test]
    fn waits_end_on_ticks_and_count_frames() {
        let null = std::ptr::null();
        assert_eq!(set_interval(null, null, 2_000, 1, 0), S_OK);
        let mut first = 0;
        assert_eq!(wait(null, 0, 1_000, std::ptr::null(), 0, &mut first), S_OK);
        let before = Instant::now();
        let mut second = 0;
        assert_eq!(wait(null, 0, 1_000, std::ptr::null(), 0, &mut second), S_OK);
        assert!(second > first);
        assert!(before.elapsed() <= Duration::from_millis(50));
        assert_eq!(wait(null, 0, 0, std::ptr::null(), 0, &mut second), TIMEOUT);
        // Tokens count frames, a frame started early on a time-out too.
        let mut early = 0;
        let flags = TOKEN_ON_TIMEOUT;
        assert_eq!(
            wait(null, 0, 0, std::ptr::null(), flags, &mut early),
            TIMEOUT
        );
        let mut next = 0;
        assert_eq!(wait(null, 0, 1_000, std::ptr::null(), 0, &mut next), S_OK);
        assert!(early == second + 1 && next == early + 1);
        assert!(LAST[0].load(Ordering::Relaxed) >= 2);

        // The title's loop (the clock is global, so in the same test): an
        // origin poll that times out (and starts the frame), then a
        // blocking flip wait. Each flip takes its own tick.
        const FLIP: u32 = 2;
        assert_eq!(set_interval(null, null, 5_000, 3, 0), S_OK);
        let mut token = 0;
        let flips: Vec<u64> = (0..4)
            .map(|_| {
                let _ = wait(null, 0, 0, std::ptr::null(), TOKEN_ON_TIMEOUT, &mut token);
                assert_eq!(
                    wait(null, FLIP, u32::MAX, std::ptr::null(), 0, &mut token),
                    S_OK
                );
                LAST[2].load(Ordering::Relaxed)
            })
            .collect();
        // On a CI runner without a card the scheduler may wake a flip wait more than one
        // interval late: there the flips are held to later ticks than the last (no two on one
        // tick), and the next check (a wait begun on a just-ended tick takes the next) holds.
        let consecutive = consecutive_here(
            std::env::var("VICTORY_TESTS_ALLOW_SOFTWARE_SKIP").is_ok_and(|v| v == "1"),
            hardware_listed(),
        );
        if !consecutive {
            eprintln!(
                "a CI runner without a card: flips held to later ticks, not to the next one (VICTORY_TESTS_ALLOW_SOFTWARE_SKIP=1)"
            );
        }
        assert!(flips_hold(&flips, consecutive), "{flips:?}");
        // Even when a flip wait begins on the tick another just ended on.
        let (tick, _) = next_tick(FLIP);
        assert!(tick > flips[3]);
    }
}
