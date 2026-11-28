//! Whether the shaders are ready, said on the frame-rate chip, the details
//! panel and the window's top bar. The count is the runtime's own queue:
//! host pipelines queued and not made yet (`crate::pipeline::making`, the
//! frame line's `making=`); making one translates its shaders and has the
//! driver compile it, so it counts both.
//!
//! Only a wave is said, not a trickle: `PREPARING SHADERS (n left)` once more
//! than [`ABOVE`] have waited for [`HOLD`] on end, then its count down to 0;
//! then `SHADERS READY` for [`READY_FOR`]; then nothing. In play the queue
//! takes one or two new pipelines every few seconds and makes them in about a
//! second, so a status that followed every one of them flipped between
//! preparing and ready all the time; those say nothing now, and neither do
//! they take a ready away. The window's top bar says the same as the chip.

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How long `SHADERS READY` stays.
pub(in crate::host) const READY_FOR: Duration = Duration::from_secs(5);

/// More waiting than this is a wave; this many or fewer is a trickle.
pub(in crate::host) const ABOVE: u64 = 8;

/// How long a wave has stayed above [`ABOVE`] before it is said.
pub(in crate::host) const HOLD: Duration = Duration::from_secs(2);

/// Where the status is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::host) enum Status {
    /// No wave, and none finished in the last [`READY_FOR`].
    #[default]
    Hidden,
    /// A wave: that many pipelines wait.
    Preparing(u64),
    /// The wave came down to nothing at that moment.
    Ready(Instant),
}

impl Status {
    /// Its colour on a chip: preparing pink, ready cyan (the settings panel's heading and choice colours).
    pub(in crate::host) fn colour(self) -> [u8; 3] {
        match self {
            Status::Ready(_) => super::art::VALUE,
            _ => super::art::HEADING,
        }
    }

    /// What is said, if anything.
    pub(in crate::host) fn words(self) -> Option<String> {
        match self {
            Status::Hidden => None,
            Status::Preparing(n) => Some(format!("PREPARING SHADERS ({n} left)")),
            Status::Ready(_) => Some("SHADERS READY".to_owned()),
        }
    }
}

/// The status said, and since when the queue has been above [`ABOVE`].
#[derive(Clone, Copy, Debug, Default)]
struct Steady {
    said: Status,
    above_since: Option<Instant>,
}

impl Steady {
    const NEW: Steady = Steady { said: Status::Hidden, above_since: None };

    /// The status with `left` pipelines waiting at `now`.
    fn next(&mut self, left: u64, now: Instant) -> Status {
        self.above_since = (left > ABOVE).then(|| self.above_since.unwrap_or(now));
        let wave = self.above_since.is_some_and(|since| now.duration_since(since) >= HOLD);
        self.said = match self.said {
            // A wave said is followed down to its end.
            Status::Preparing(_) if left > 0 => Status::Preparing(left),
            Status::Preparing(_) => Status::Ready(now),
            _ if wave => Status::Preparing(left),
            // A trickle neither shows nor takes a ready away.
            Status::Ready(since) if now.duration_since(since) < READY_FOR => self.said,
            _ => Status::Hidden,
        };
        self.said
    }
}

/// The status now, from the queue as it is now.
pub(in crate::host) fn now() -> Status {
    static STEADY: Mutex<Steady> = Mutex::new(Steady::NEW);
    let mut steady = STEADY.lock().unwrap_or_else(|e| e.into_inner());
    steady.next(crate::pipeline::making(), Instant::now())
}

/// What the status says now, if anything.
pub(in crate::host) fn line() -> Option<String> {
    now().words()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trickle_says_nothing_a_wave_is_said_down_to_ready_then_nothing() {
        let t0 = Instant::now();
        let at = |s: f64| t0 + Duration::from_secs_f64(s);
        let mut steady = Steady::NEW;
        let mut next = |left: u64, s: f64| steady.next(left, at(s));
        // Nothing, then a trickle of one or two: nothing said.
        assert_eq!(next(0, 0.0), Status::Hidden);
        assert_eq!(next(2, 0.5), Status::Hidden);
        assert_eq!(next(1, 1.0), Status::Hidden);
        assert_eq!(next(0, 1.5), Status::Hidden);
        // Above eight for less than the hold, then down: nothing said.
        assert_eq!(next(12, 2.0), Status::Hidden);
        assert_eq!(next(12, 3.9), Status::Hidden);
        assert_eq!(next(3, 4.0), Status::Hidden);
        // A dip to eight starts the hold again.
        assert_eq!(next(400, 5.0), Status::Hidden);
        assert_eq!(next(8, 6.0), Status::Hidden);
        assert_eq!(next(1200, 6.5), Status::Hidden);
        assert_eq!(next(1200, 8.4), Status::Hidden);
        // Held above eight for two seconds: said, with its count, down to its last one.
        assert_eq!(next(1100, 8.5).words().as_deref(), Some("PREPARING SHADERS (1100 left)"));
        assert_eq!(next(3, 30.0).words().as_deref(), Some("PREPARING SHADERS (3 left)"));
        assert_eq!(next(1, 31.0), Status::Preparing(1));
        // Down to nothing: ready for five seconds, a trickle meanwhile changing nothing.
        assert_eq!(next(0, 32.0), Status::Ready(at(32.0)));
        assert_eq!(next(2, 33.0), Status::Ready(at(32.0)));
        assert_eq!(next(0, 36.9).words().as_deref(), Some("SHADERS READY"));
        assert_eq!(next(1, 37.0), Status::Hidden);
        assert_eq!(next(0, 60.0), Status::Hidden);
        // A later wave is said again, and ends the same way.
        assert_eq!(next(40, 61.0), Status::Hidden);
        assert_eq!(next(30, 63.0), Status::Preparing(30));
        assert_eq!(next(0, 64.0), Status::Ready(at(64.0)));
        // A wave that comes during "ready" is said once it has held.
        assert_eq!(next(50, 65.0), Status::Ready(at(64.0)));
        assert_eq!(next(50, 67.0), Status::Preparing(50));
    }

    const FRAME_LINE: &str = r"
        0.0:0 5.3:0 10.3:0 15.3:0 20.3:0 25.3:0 30.4:0 35.4:0 40.4:0 45.4:0 50.4:0 55.4:0 60.4:0
        65.4:0 70.4:0 75.4:0 80.5:0 85.5:0 90.5:0 95.5:0 100.5:0 105.5:0 110.5:0 115.5:0 120.5:0
        125.5:0 130.5:0 135.5:0 140.5:0 145.5:0 150.5:0 155.5:0 160.6:0 165.6:0 170.7:0 175.7:0
        180.7:0 185.7:0 190.8:0 195.8:0 200.8:0 205.8:0 210.9:0 215.9:0 220.9:0 225.9:0 230.9:0
        235.9:0 240.9:0 246.0:0 251.0:0 256.1:0 261.2:0 266.2:0 271.2:0 276.2:0 281.3:0 286.4:0
        291.5:0 296.6:0 308.9:1057 314.6:756 320.7:695 326.9:638 332.0:537 337.1:491 343.1:570
        349.5:444 355.1:277 360.7:96 366.8:68 372.8:0 378.1:0 383.4:0 388.7:0 394.3:0 399.6:0
        404.8:0 410.0:0 415.2:0 420.5:0 425.6:0 430.8:0 436.0:0 441.1:0 446.3:0
    ";
    const FINISHED: &str = r"
        -4.69 -4.66 -4.4 -4.16 -4.14 -4.12 -4.12 -3.87 -2.64 -2.63 -2.51 -2.49 -2.47 -2.44 -2.44
        -2.35 -2.34 -2.32 -2.29 -2.26 -2.23 -2.2 -1.23 2.59 2.61 267.56 267.57 267.68 267.73 282.65
        282.68 290.7 290.76 295.93 297.6 297.75 297.77 297.79 298.0 299.72 299.8 299.85 299.88
        299.89 299.91 299.92 299.94 299.97 300.0 300.13 300.14 300.14 300.14 300.14 300.14 300.14
        300.16 300.17 300.19 300.19 300.19 300.21 300.23 300.25 300.25 300.26 300.27 300.27 300.27
        300.29 300.3 300.3 300.31 300.31 300.31 300.33 300.33 300.33 300.33 300.33 300.34 300.34
        300.34 300.36 300.37 300.39 300.39 300.4 300.4 300.43 300.43 300.44 300.44 300.46 300.46
        300.47 300.48 300.48 300.48 300.48 300.49 300.5 300.51 300.51 300.51 300.52 300.54 300.54
        300.55 300.57 300.57 300.61 300.61 300.61 300.64 300.64 300.66 300.68 300.69 300.7 300.71
        300.71 300.73 300.73 300.74 300.78 300.78 300.79 300.81 300.82 300.82 300.82 300.82 300.85
        300.85 300.85 300.86 300.87 300.87 300.89 300.89 300.89 300.9 300.91 300.91 300.95 300.98
        300.98 300.98 300.98 300.98 300.99 301.01 301.01 301.02 301.02 301.03 301.15 301.35 301.44
        301.46 301.48 301.6 301.69 301.78 301.8 301.84 301.88 301.96 301.97 302.05 302.08 302.13
        302.15 302.16 302.22 302.22 302.27 302.31 302.34 302.35 302.4 302.41 302.43 302.46 302.54
        302.62 302.62 302.62 302.88 302.88 302.9 302.91 302.93 302.95 303.03 303.04 303.54 303.55
        303.6 303.6 303.66 303.66 303.67 303.74 303.76 303.78 303.86 303.86 303.9 303.91 303.92
        304.01 304.01 304.01 304.03 304.04 304.05 304.07 304.11 304.13 304.15 304.16 304.18 304.29
        304.29 304.33 304.35 304.39 304.4 304.46 304.49 304.5 304.52 304.54 304.54 304.58 304.61
        304.62 304.72 304.87 304.87 304.88 304.89 304.9 304.94 305.0 305.04 305.1 305.1 305.14
        305.14 305.14 305.14 305.16 305.16 305.37 305.47 305.55 305.56 305.57 305.57 305.57 305.6
        305.61 305.63 305.64 305.64 305.66 305.66 305.67 305.69 305.7 305.7 305.7 305.71 305.72
        305.72 305.72 305.72 305.91 305.98 306.07 306.07 306.08 306.26 306.26 306.29 306.3 306.3
        306.39 306.44 306.49 306.53 306.57 306.64 306.64 306.64 306.64 306.64 306.65 306.68 306.68
        306.69 306.72 306.73 306.73 306.73 306.74 306.74 306.74 306.78 306.8 306.8 306.81 306.87
        306.88 306.89 306.9 306.98 306.98 306.98 306.98 306.98 306.98 307.0 307.01 307.03 307.12
        307.16 307.18 307.19 307.2 307.2 307.22 307.36 307.4 307.41 307.55 307.59 307.61 307.61
        307.61 307.61 307.78 307.79 307.84 307.87 307.91 307.92 307.94 307.94 307.96 307.97 307.97
        308.01 308.09 308.14 308.14 308.23 308.26 308.32 308.33 308.33 308.42 308.45 308.57 308.57
        308.63 308.63 308.63 308.68 308.69 308.7 308.71 308.72 308.73 308.75 308.77 308.8 308.8
        366.93 367.07 367.07 367.07 367.08 367.08 367.09 367.09 367.09 367.1 367.11 367.11 367.11
        367.15 367.16 367.17 367.17 367.17 367.19 367.2 367.2 367.23 367.34 367.36 367.44 367.49
        367.49 367.49 367.5 367.5 367.5 367.5 367.5 367.51 367.51 367.58 367.58 367.58 367.58
        367.58 367.59 367.6 367.6 367.6 367.61 367.62 367.62 367.63 367.64 367.64 367.71 367.71
        367.71 367.71 367.71 367.72 367.72 367.74 367.8 367.8 367.83 367.96 367.99 368.11 368.39
        368.46 368.48 368.84 368.99 369.57 369.94 369.98 370.54 370.84 371.0 371.45 374.43 380.38
        381.94 382.5 383.3 384.31 385.25 385.38 385.73 386.51 386.77 386.78 387.81 388.65 388.66
        388.66 388.89 390.14 390.84 391.15 394.1 394.86 395.73 396.76 396.76 397.06 397.09 397.46
        398.76 399.24 401.47 402.41 431.21 431.22 431.27 431.3 431.31 431.31 431.32 431.32 431.32
        431.33 431.34 431.34 431.39 431.77 431.88 432.34 433.42 433.45 442.56 445.11 445.92
    ";

    /// Replays an open session's queue (a recorded run with a cold
    /// pipeline cache): the frame line's counts (`seconds:count`, one about
    /// every five seconds), and between its counts above 0 and around them
    /// the moments a pipeline finished (seconds from the frame line's first),
    /// each taken as waiting [`WAITED`] before it. At the chip's rate (four a
    /// second), the old status flipped every few seconds after the long wave;
    /// this one says the long wave, its ready, and nothing in play after it.
    #[test]
    fn a_real_session_says_its_long_wave_and_its_ready_and_nothing_in_play() {
        const WAITED: f64 = 2.0;
        let frame: Vec<(f64, u64)> = FRAME_LINE
            .split_whitespace()
            .map(|s| s.split_once(':').map(|(t, n)| (t.parse().unwrap(), n.parse().unwrap())).unwrap())
            .collect();
        let finished: Vec<f64> = FINISHED.split_whitespace().map(|s| s.parse().unwrap()).collect();
        let queue = |t: f64| {
            let line = frame.iter().take_while(|(at, _)| *at <= t).last().map_or(0, |(_, n)| *n);
            let made = finished.iter().filter(|&&f| t < f && f <= t + WAITED).count() as u64;
            line.max(made)
        };
        let t0 = Instant::now();
        let at = |s: f64| t0 + Duration::from_secs_f64(s + 30.0);
        let (start, end) = (-20.0, 450.0);
        let steps = ((end - start) * 4.0) as usize;
        let mut steady = Steady::NEW;
        let mut said: Vec<(f64, Option<&'static str>)> = vec![];
        let mut trickles = 0;
        let mut was = 0;
        for i in 0..=steps {
            let t = start + i as f64 * 0.25;
            let left = queue(t);
            if t >= 380.0 && was == 0 && left > 0 {
                trickles += 1;
            }
            was = left;
            let status = steady.next(left, at(t));
            if t == 330.0 {
                assert!(matches!(status, Status::Preparing(n) if n >= 600), "{status:?}");
            }
            let words = match status {
                Status::Hidden => None,
                Status::Preparing(_) => Some("preparing"),
                Status::Ready(_) => Some("ready"),
            };
            if said.last().map(|(_, w)| *w) != Some(words) {
                said.push((t, words));
            }
        }
        // In play after the wave the queue still comes and goes (five times here), and the old status said each.
        assert!(trickles >= 4, "{trickles}");
        // A short wave at the start, before the first frame line; the long
        // wave from about 300 s, ready once it is made, and then nothing.
        let what: Vec<_> = said.iter().map(|(_, w)| *w).collect();
        assert_eq!(
            what,
            [None, Some("preparing"), Some("ready"), None, Some("preparing"), Some("ready"), None],
            "{said:?}"
        );
        let (ready, gone) = (said[5].0, said[6].0);
        assert!((372.0..377.0).contains(&ready), "{said:?}");
        assert!(gone - ready >= READY_FOR.as_secs_f64() - 0.25, "{said:?}");
        assert!(gone < 383.0, "{said:?}");
    }
}
