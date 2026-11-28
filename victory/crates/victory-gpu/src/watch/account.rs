//! Who holds the guards' lock, and who waits behind whom
//! (`VICTORY_GUARDS_ACCOUNT=1`; off unless asked for; diagnostics builds).
//!
//! The frame line's "guards" (`super::WAITS`) sums the waits of all threads
//! for the lock and says nothing of who held it meanwhile, nor whether the
//! thread that submits was among the waiters. With the switch, each take of
//! the lock is recorded by its caller's place in the code (the site), each
//! hold is timed from the take to the release (the guards' settling of
//! faults that took no lock inside it, timed too), and each wait is split
//! among the holders it waited behind: from its start the holder then, from
//! each following take the one who took it, until its own take. So the
//! waits behind holders add up to the waits, by the waiter's role: in a
//! submission, a submitting thread between its submissions, or any other
//! thread. (A holder's share runs to the next take, the hand-over with it.)
//!
//! Nothing here takes a lock: the takes are numbered and kept in a ring of
//! atomics, written by whoever holds the guards' lock and read by a waiter
//! once it holds it; the counts are atomics by site.

use std::panic::Location;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicPtr, AtomicU64};

/// Whether the switch is on (for this thread, in tests).
pub(super) fn on() -> bool {
    #[cfg(test)]
    if let Some(on) = ON_IN_TEST.get() {
        return on;
    }
    crate::asked_for!("VICTORY_GUARDS_ACCOUNT")
}

#[cfg(test)]
thread_local! {
    /// [`on`], for this thread (`None`: the switch).
    pub(super) static ON_IN_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Roles of a thread: in a submission, a thread that submits between its
/// submissions, any other.
pub(crate) const ROLES: usize = 3;
const IN_SUBMISSION: u8 = 0;
const SUBMITTER: u8 = 1;
const OTHER: u8 = 2;

thread_local! {
    static ROLE: std::cell::Cell<u8> = const { std::cell::Cell::new(OTHER) };
}

/// This thread begins (`true`) or ends a submission.
pub(crate) fn submitting(inside: bool) {
    ROLE.set(if inside { IN_SUBMISSION } else { SUBMITTER });
}

fn role() -> usize {
    usize::from(ROLE.get())
}

/// Nanoseconds since the first call.
fn ns() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64
}

/// Sites, the last for those that found no room (and for waits behind a
/// holder not known).
const SITES: usize = 64;
const NOT_KNOWN: usize = SITES - 1;

/// What is counted of a site.
#[derive(Clone, Copy)]
#[repr(usize)]
pub(super) enum F {
    Takes,
    /// Takes that waited for the lock, and the nanoseconds of it.
    Waits,
    WaitNs,
    /// Nanoseconds held; the longest hold; of the holds, settling.
    HoldNs,
    HoldMax,
    SettleNs,
    /// Nanoseconds of others' waits behind this site's holds, and the holds
    /// waited behind, by the waiter's role.
    Behind,
    BehindN = F::Behind as usize + ROLES,
    Fields = F::BehindN as usize + ROLES,
}

struct Site {
    at: AtomicPtr<Location<'static>>,
    n: [AtomicU64; F::Fields as usize],
}

static TABLE: [Site; SITES] = [const {
    Site {
        at: AtomicPtr::new(std::ptr::null_mut()),
        n: [const { AtomicU64::new(0) }; F::Fields as usize],
    }
}; SITES];

/// The waits by the waiter's role: number, nanoseconds.
static WAITED: [[AtomicU64; 2]; ROLES] = [const { [const { AtomicU64::new(0) }; 2] }; ROLES];

/// The site's slot.
fn slot(at: &'static Location<'static>) -> usize {
    let ptr = std::ptr::from_ref(at).cast_mut();
    for (i, site) in TABLE[..NOT_KNOWN].iter().enumerate() {
        let there = site.at.load(Relaxed);
        if there.is_null() {
            match site.at.compare_exchange(std::ptr::null_mut(), ptr, Relaxed, Relaxed) {
                Ok(_) => return i,
                // SAFETY: only `&'static Location`s are stored.
                Err(other) if unsafe { *other == *at } => return i,
                Err(_) => continue,
            }
        }
        // SAFETY: as above.
        if there == ptr || unsafe { *there == *at } {
            return i;
        }
    }
    NOT_KNOWN
}

fn add(slot: usize, field: usize, n: u64) {
    TABLE[slot].n[field].fetch_add(n, Relaxed);
}

/// The takes, numbered from 1, as `(number, site and role, when)`.
const RING: usize = 1024;
static RING_AT: [[AtomicU64; 3]; RING] = [const { [const { AtomicU64::new(0) }; 3] }; RING];
static TAKEN: AtomicU64 = AtomicU64::new(0);

/// A wait for the lock begins: when, and the last take then (whose holder
/// is presumably the one waited for).
pub(super) fn wait_begins() -> (u64, u64) {
    (ns(), TAKEN.load(Relaxed))
}

/// What a hold of the lock has recorded so far.
pub(super) struct Hold {
    slot: usize,
    at: u64,
    pub(super) settling: u64,
}

/// The lock was taken at `site`, after the wait `waited` began if there was
/// one. Called holding the lock.
pub(super) fn taken(site: &'static Location<'static>, waited: Option<(u64, u64)>) -> Hold {
    let now = ns();
    let slot = slot(site);
    let role = role();
    // The lock orders these writes, and a waiter's reads after its own take.
    let number = TAKEN.load(Relaxed) + 1;
    let entry = &RING_AT[number as usize % RING];
    entry[0].store(number, Relaxed);
    entry[1].store((slot as u64) << 8 | role as u64, Relaxed);
    entry[2].store(now, Relaxed);
    TAKEN.store(number, Relaxed);
    add(slot, F::Takes as usize, 1);
    if let Some((began, last)) = waited {
        let wait = now.saturating_sub(began);
        add(slot, F::Waits as usize, 1);
        add(slot, F::WaitNs as usize, wait);
        WAITED[role][0].fetch_add(1, Relaxed);
        WAITED[role][1].fetch_add(wait, Relaxed);
        behind(role, began, last, number, now);
    }
    Hold { slot, at: now, settling: 0 }
}

/// Splits a wait from `began` to `now` among the holders of takes `last`
/// to `number - 1`; what cannot be told goes to [`NOT_KNOWN`].
fn behind(role: usize, began: u64, last: u64, number: u64, now: u64) {
    let entry = |k: u64| {
        let e = &RING_AT[k as usize % RING];
        (k != 0 && e[0].load(Relaxed) == k).then(|| ((e[1].load(Relaxed) >> 8) as usize, e[2].load(Relaxed)))
    };
    let mut charged = 0;
    // (Takes before the ring's oldest have left it: what they held goes to
    // `NOT_KNOWN` either way, and the walk under the lock stays short.)
    let mut k = last.max(number.saturating_sub(RING as u64));
    while k < number {
        let from = if k == last { began } else { entry(k).map_or(began, |e| e.1).max(began) };
        let to = if k + 1 == number { now } else { entry(k + 1).map_or(now, |e| e.1).min(now) };
        if let Some((slot, _)) = entry(k)
            && to > from
        {
            add(slot, F::Behind as usize + role, to - from);
            add(slot, F::BehindN as usize + role, 1);
            charged += to - from;
        }
        k += 1;
    }
    let rest = now.saturating_sub(began).saturating_sub(charged);
    if rest > 0 {
        add(NOT_KNOWN, F::Behind as usize + role, rest);
        add(NOT_KNOWN, F::BehindN as usize + role, 1);
    }
}

/// The lock is let go.
pub(super) fn released(hold: &Hold) {
    let held = ns().saturating_sub(hold.at);
    add(hold.slot, F::HoldNs as usize, held);
    TABLE[hold.slot].n[F::HoldMax as usize].fetch_max(held, Relaxed);
    add(hold.slot, F::SettleNs as usize, hold.settling);
}

/// Nanoseconds since `from` ([`ns`]'s clock).
pub(super) fn since(from: u64) -> u64 {
    ns().saturating_sub(from)
}

pub(super) fn now() -> u64 {
    ns()
}

/// One site's counts.
#[derive(Clone, Copy, Default, Debug)]
pub(super) struct Counts {
    pub(super) at: Option<&'static Location<'static>>,
    pub(super) n: [u64; F::Fields as usize],
}

/// The counts of every site and the waits by role; `take` sets them to
/// zero.
pub(super) fn counts(take: bool) -> (Vec<Counts>, [[u64; 2]; ROLES]) {
    let get = |a: &AtomicU64| if take { a.swap(0, Relaxed) } else { a.load(Relaxed) };
    let sites = TABLE
        .iter()
        .enumerate()
        .map(|(i, site)| {
            let at = site.at.load(Relaxed);
            Counts {
                // SAFETY: only `&'static Location`s are stored.
                at: (i != NOT_KNOWN && !at.is_null()).then(|| unsafe { &*at }),
                n: std::array::from_fn(|f| get(&site.n[f])),
            }
        })
        .collect();
    (sites, std::array::from_fn(|r| [get(&WAITED[r][0]), get(&WAITED[r][1])]))
}

/// The line for the frame-rate lines, per frame; `None` with the switch off.
///
/// Reading it:
/// - "behind it" is summed over the waiters, and runs from each wait's start
///   (or the holder's take) to the next take, the hand-over included: three
///   threads waiting behind one 50 ms hold make about 150 ms behind it, and
///   it can be more than the site's "held".
/// - A submission that ends on the late path is told ended at
///   `copystat::submission_ends` (host.rs 6102) while its thread still holds
///   the submission lock: its waits after that count as "between
///   submissions".
/// - A thread that has submitted once is a submitting thread for good.
/// - Takes and waits are numbers per frame, times milliseconds per frame,
///   "longest" milliseconds; the ten rows the submissions waited behind
///   most; "not known" also holds sites past the 63rd.
pub(crate) fn line(frames: f64) -> Option<String> {
    if !on() {
        return None;
    }
    let (sites, waited) = counts(true);
    let ms = |ns: u64| ns as f64 / 1e6 / frames;
    let each = |n: u64| n as f64 / frames;
    let roles = ["in submissions", "submitting threads between them", "other threads"];
    let waits: Vec<String> = (0..ROLES)
        .map(|r| format!("{} {:.0} waits {:.2} ms", roles[r], each(waited[r][0]), ms(waited[r][1])))
        .collect();
    let behind = |c: &Counts| c.n[F::Behind as usize..F::BehindN as usize].iter().sum::<u64>();
    let mut shown: Vec<&Counts> = sites.iter().filter(|c| c.n[F::Takes as usize] != 0 || behind(c) != 0).collect();
    // Those the submissions waited behind most first.
    let in_submissions = |c: &Counts| c.n[F::Behind as usize + usize::from(IN_SUBMISSION)];
    shown.sort_by_key(|c| std::cmp::Reverse((in_submissions(c), behind(c), c.n[F::HoldNs as usize])));
    let holders: Vec<String> = shown
        .iter()
        .take(10)
        .map(|c| {
            let place = c.at.map_or_else(
                || "not known".to_string(),
                |at| format!("{}:{}", at.file().rsplit(['/', '\\']).next().unwrap_or(at.file()), at.line()),
            );
            let b = |r: usize| ms(c.n[F::Behind as usize + r]);
            format!(
                "{place} {:.0} takes held {:.2} ms (longest {:.2} ms, settling {:.2} ms), {:.0} waited here {:.2} ms; \
                 behind it {:.2} / {:.2} / {:.2} ms",
                each(c.n[F::Takes as usize]),
                ms(c.n[F::HoldNs as usize]),
                c.n[F::HoldMax as usize] as f64 / 1e6,
                ms(c.n[F::SettleNs as usize]),
                each(c.n[F::Waits as usize]),
                ms(c.n[F::WaitNs as usize]),
                b(0),
                b(1),
                b(2)
            )
        })
        .collect();
    Some(format!(
        "the guards' lock per frame (VICTORY_GUARDS_ACCOUNT=1): waits {}; by holder, waits behind it in submissions / \
         submitting threads between them / other threads: {}",
        waits.join(", "),
        holders.join("; ")
    ))
}

#[cfg(test)]
mod tests {
    use super::super::guards;
    use super::{Counts, F, NOT_KNOWN, ON_IN_TEST, ROLES, counts, submitting};
    use std::sync::Mutex;
    use std::time::Duration;

    /// The tests here, one at a time: the counts are the process's.
    static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

    fn of(sites: &[Counts], line: u32) -> Counts {
        sites
            .iter()
            .copied()
            .find(|c| c.at.is_some_and(|at| at.line() == line && at.file().ends_with("account.rs")))
            .unwrap_or_default()
    }

    /// Per site and role, the counts added since `before`.
    fn since(before: &(Vec<Counts>, [[u64; 2]; ROLES])) -> (Vec<Counts>, [[u64; 2]; ROLES]) {
        let (now, waited) = counts(false);
        let sites = now
            .iter()
            .zip(&before.0)
            .map(|(a, b)| Counts { at: a.at, n: std::array::from_fn(|f| a.n[f].wrapping_sub(b.n[f])) })
            .collect();
        (sites, std::array::from_fn(|r| [waited[r][0] - before.1[r][0], waited[r][1] - before.1[r][1]]))
    }

    /// An uncontended take; a walk's retake at another site; a wait of a
    /// thread in a submission behind one holder and then another (whoever
    /// of the two waiters comes first): every take and hold is the site's,
    /// and the waits behind holders add up to the waits, role by role.
    #[test]
    fn waits_add_up_to_the_holders_waited_behind() {
        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        ON_IN_TEST.set(Some(true));
        let before = counts(false);
        // Uncontended, then retaken elsewhere, as a walk does between runs.
        let (g, alone) = (guards(), line!());
        drop(g);
        let (g, retake) = (guards(), line!());
        std::thread::sleep(Duration::from_millis(5));
        drop(g);
        let (held, waiting) = (std::sync::mpsc::channel(), std::sync::mpsc::channel());
        let (first, second, submitter) = std::thread::scope(|s| {
            let (held_tx, held_rx) = held;
            let (waiting_tx, waiting_rx) = waiting;
            let holder = s.spawn(move || {
                ON_IN_TEST.set(Some(true));
                let (g, line) = (guards(), line!());
                held_tx.send(()).unwrap();
                // Both waiters wait (as far as a sleep can tell).
                waiting_rx.recv().unwrap();
                waiting_rx.recv().unwrap();
                std::thread::sleep(Duration::from_millis(60));
                drop(g);
                line
            });
            held_rx.recv().unwrap();
            let waiting_two = waiting_tx.clone();
            let other = s.spawn(move || {
                ON_IN_TEST.set(Some(true));
                waiting_two.send(()).unwrap();
                let (g, line) = (guards(), line!());
                std::thread::sleep(Duration::from_millis(40));
                drop(g);
                line
            });
            let waiter = s.spawn(move || {
                ON_IN_TEST.set(Some(true));
                submitting(true);
                waiting_tx.send(()).unwrap();
                let (g, line) = (guards(), line!());
                drop(g);
                submitting(false);
                line
            });
            (holder.join().unwrap(), other.join().unwrap(), waiter.join().unwrap())
        });
        let (sites, waited) = since(&before);
        ON_IN_TEST.set(None);
        let site = |line| of(&sites, line);
        for line in [retake, first, second, submitter] {
            assert_eq!(site(line).n[F::Takes as usize], 1, "line {line} taken once");
        }
        // (Uncontended unless another test's thread held the lock just then.)
        let alone = site(alone);
        assert_eq!(alone.n[F::Takes as usize], 1);
        if alone.n[F::Waits as usize] == 0 {
            assert_eq!(alone.n[F::WaitNs as usize], 0, "no wait counted for an uncontended take");
        }
        assert!(site(retake).n[F::HoldNs as usize] >= 5_000_000, "the retake's own hold");
        assert!(site(first).n[F::HoldNs as usize] >= 60_000_000);
        for c in [alone, site(retake), site(first), site(second)] {
            assert!(c.n[F::SettleNs as usize] <= c.n[F::HoldNs as usize], "settling is inside the hold");
        }
        // The submission waited 60 ms behind the first holder at least, and
        // 40 more behind the second if that one came first.
        let wait = site(submitter).n[F::WaitNs as usize];
        assert_eq!(site(submitter).n[F::Waits as usize], 1);
        assert!(wait >= 55_000_000, "waited {wait} ns");
        let behind_first = site(first).n[F::Behind as usize];
        let behind_second = site(second).n[F::Behind as usize];
        assert!(behind_first >= 50_000_000, "{behind_first} ns behind the first holder");
        if wait >= 100_000_000 {
            assert!(behind_second >= 35_000_000, "{behind_second} ns behind the second");
        }
        // Role by role, the waits are all waited behind someone.
        for role in 0..ROLES {
            let split: u64 = sites.iter().map(|c| c.n[F::Behind as usize + role]).sum();
            assert_eq!(split, waited[role][1], "role {role}: the waits behind holders add up to the waits");
        }
        assert_eq!(waited[0][0], 1, "one wait in a submission");
        assert_eq!(waited[0][1], wait);
        // (Not known only if a take went by unrecorded: the ring holds 1,024.)
        assert_eq!(sites[NOT_KNOWN].n[F::Behind as usize], 0);
    }

    /// Three threads wait behind one 50 ms hold: the account's waits and
    /// their time are the frame line's "guards" (`super::super::WAITS`), to
    /// the tens of nanoseconds a wait's own counting takes. In a child
    /// process with the switch on (every thread of it counted, no other
    /// test's beside it).
    #[test]
    fn the_account_agrees_with_the_guards_figure() {
        const NAME: &str = "watch::account::tests::the_account_agrees_with_the_guards_figure";
        if std::env::var_os("VICTORY_ACCOUNT_CHILD").is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", NAME, "--nocapture", "--test-threads", "1"])
                .env("VICTORY_ACCOUNT_CHILD", "1")
                .env("VICTORY_GUARDS_ACCOUNT", "1")
                .output()
                .expect("the test binary again");
            let (stdout, stderr) = (String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
            assert!(out.status.success() && stdout.contains("1 passed"), "the child:
{stdout}
{stderr}");
            return;
        }
        use std::sync::atomic::Ordering::Relaxed;
        assert!(super::on(), "the switch, in the child");
        let figure = || [0, 1].map(|i| super::super::WAITS[3][i].load(Relaxed));
        let (before, figure_before) = (counts(false), figure());
        std::thread::scope(|s| {
            let (held_tx, held_rx) = std::sync::mpsc::channel();
            let (waiting_tx, waiting_rx) = std::sync::mpsc::channel();
            s.spawn(move || {
                let g = guards();
                held_tx.send(()).unwrap();
                for _ in 0..3 {
                    waiting_rx.recv().unwrap();
                }
                std::thread::sleep(Duration::from_millis(50));
                drop(g);
            });
            held_rx.recv().unwrap();
            for _ in 0..3 {
                let waiting = waiting_tx.clone();
                s.spawn(move || {
                    waiting.send(()).unwrap();
                    drop(guards());
                });
            }
        });
        let (_, waited) = since(&before);
        let figure_now = figure();
        let (n, ns) = (figure_now[0] - figure_before[0], figure_now[1] - figure_before[1]);
        let account_n: u64 = waited.iter().map(|w| w[0]).sum();
        let account_ns: u64 = waited.iter().map(|w| w[1]).sum();
        eprintln!("account: {account_n} waits {account_ns} ns; guards: {n} waits {ns} ns");
        assert_eq!((account_n, n), (3, 3), "three waits on both");
        assert!(account_ns >= 3 * 45_000_000, "three waits behind 50 ms: {account_ns} ns");
        assert!(account_ns.abs_diff(ns) < 1_000_000, "the account {account_ns} ns, the guards figure {ns} ns");
    }

    /// With the switch off nothing is counted, a wait included, and there
    /// is no line.
    #[test]
    fn off_counts_nothing() {
        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        ON_IN_TEST.set(Some(false));
        let before = counts(false);
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        std::thread::scope(|s| {
            s.spawn(move || {
                ON_IN_TEST.set(Some(false));
                let g = guards();
                held_tx.send(()).unwrap();
                std::thread::sleep(Duration::from_millis(20));
                drop(g);
            });
            held_rx.recv().unwrap();
            drop(guards());
        });
        let (sites, waited) = since(&before);
        assert!(sites.iter().all(|c| c.n.iter().all(|&n| n == 0)), "a count moved with the switch off");
        assert_eq!(waited, [[0; 2]; ROLES]);
        assert!(super::line(1.0).is_none());
        ON_IN_TEST.set(None);
    }
}
