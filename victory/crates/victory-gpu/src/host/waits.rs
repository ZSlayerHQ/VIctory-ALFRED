//! With flights in the air: why preparing is blocked by them, whether a
//! refresh races one, and the wait for flights to land.

use std::sync::atomic::Ordering;

use crate::budget::{Phase as Ph, phase as ph};

use super::{
    HostList, LATE_STATS, LATE_WAITED, LINE_LANDED, Line, SUBMISSION_US, Step, WHY_READ, WHY_RUN, late_exact, line,
    stood, touches, working,
};
#[cfg(test)]
use super::{HOOK_RELAY, test_hook};

/// [`late_blocked`] against `written`, the title memory flights' shaders
/// may write (in order, joined); `count` whether the kind of read is
/// counted for the frame line.
pub(super) fn blocked_by(written: &[(usize, usize)], runs: &[&HostList], count: bool) -> Option<usize> {
    if written.is_empty() {
        return None;
    }
    for list in runs {
        for step in &list.deferred().copies {
            match step {
                &Step::Copy(dst, src, len) => {
                    if touches(written, src, src + len) || touches(written, dst, dst + len) {
                        if count {
                            LATE_STATS[15].fetch_add(1, Ordering::Relaxed);
                        }
                        return Some(WHY_READ);
                    }
                }
                Step::Fill(_) | Step::Fills(_) => {
                    if step.fills().iter().any(|f| touches(written, f.src, f.src + f.len)) {
                        if count {
                            LATE_STATS[16].fetch_add(1, Ordering::Relaxed);
                        }
                        return Some(WHY_READ);
                    }
                }
                &Step::TableFill(_, src, len) => {
                    if touches(written, src, src + len) {
                        if count {
                            LATE_STATS[16].fetch_add(1, Ordering::Relaxed);
                        }
                        return Some(WHY_READ);
                    }
                }
                Step::Texture(staging) => {
                    let (at, len) = staging.memory();
                    if touches(written, at, at + len) {
                        if count {
                            LATE_STATS[17].fetch_add(1, Ordering::Relaxed);
                        }
                        return Some(WHY_READ);
                    }
                }
                Step::Run(_) => return Some(WHY_RUN),
                Step::Refresh(..) | Step::GpuWrite(..) | Step::Host(_) => {}
            }
        }
    }
    None
}

/// With flights in the air, after the submission's write-watch asking:
/// whether `runs` refresh a range of a host copy that a flight's shaders
/// may write and that the title wrote since it was last copied. The
/// refresh leaves ranges shaders are writing alone (`Shadow::update`), so
/// the title's bytes there would reach the card a submission late.
pub(super) fn stale_under_flights(runs: &[&HostList]) -> bool {
    use std::sync::Arc;
    // Per copy, the span of what is refreshed (a flush brings that much
    // up to date at most).
    let mut spans: Vec<(Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::Refresh(shadow, address, len) = step
                && shadow.writes_pending()
            {
                match spans.iter_mut().find(|s| Arc::ptr_eq(&s.0, shadow)) {
                    Some(s) => (s.1, s.2) = (s.1.min(*address), s.2.max(*address + *len)),
                    None => spans.push((shadow.clone(), *address, *address + *len)),
                }
            }
        }
    }
    if late_exact() {
        // The title's bytes are put in beside the flights' writes; a
        // settle is still due where that could not be done. At most this
        // many changed runs go up to copies in the card's memory, each a
        // copy of its own there.
        let mut runs_left = 512;
        return spans.iter().any(|(shadow, from, to)| shadow.merge_under_writes(*from, *to - *from, &mut runs_left));
    }
    spans.iter().any(|(shadow, from, to)| shadow.stale_under_writes(*from, *to - *from))
}

/// With [`relay`]: waits until `done` says so of the line, looked at each
/// time flights have landed. Counted as a settle for `why`, and its time
/// as a wait for the relay ([`SUBMISSION_US`]; the card clock's "stood
/// waiting for the card" counts it too). The caller may hold the submission
/// lock: the thread that lands flights never takes it.
pub(super) fn relay_wait(why: usize, done: impl Fn(&Line) -> bool) {
    let mut line = line();
    if done(&line) {
        return;
    }
    LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
    LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
    let was_working = working(false);
    let _p = ph(Ph::RelayWait);
    let waiting = std::time::Instant::now();
    #[cfg(test)]
    test_hook(HOOK_RELAY);
    while !done(&line) {
        line = LINE_LANDED.wait(line).unwrap_or_else(|e| e.into_inner());
    }
    // The landing that released it, read before the line goes (the relay
    // notes it under the same lock).
    #[cfg(feature = "diagnostics")]
    let released = crate::stalls::landing();
    drop(line);
    stood(waiting);
    working(was_working);
    let us = waiting.elapsed().as_micros() as u64;
    LATE_STATS[12].fetch_add(us, Ordering::Relaxed);
    LATE_WAITED[why.min(10)].fetch_add(us, Ordering::Relaxed);
    SUBMISSION_US[1].fetch_add(us, Ordering::Relaxed);
    #[cfg(feature = "diagnostics")]
    crate::stalls::settled(why, released, us);
}
