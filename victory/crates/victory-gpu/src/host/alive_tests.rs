//! On this tree: a thread of ours that dies must end the run, not leave
//! what waits on it waiting for good.
//!
//! - The thread that lands flights (`relay_worker`, `VICTORY_RELAY=2`)
//!   panics by a test hook ([`RELAY_PANICS`]) as it takes a flight from the
//!   line, so it holds that flight and the line says a landing is under way;
//!   then the title settles, as its next present does ([`settle_now`]): the
//!   wait for landings (`relay_wait`) is what waits on it.
//! - The late helper (`late_helper`, the relay off) panics by a test hook
//!   ([`HELPER_PANICS`]) as it wakes with a flight to settle. No wait of
//!   ours waits on it: what its death leaves waiting is the title's own
//!   wait, here on the event of a title fence whose signal went behind the
//!   flight (`late_signal`), which only a settle sets.
//!
//! Either way the run must end loudly within [`BOUND`]: `fail`'s line ("host
//! GPU failure: ...", the tests' copy of the run's crash report) naming the
//! dead thread; never by a wait that returns as if the flight had settled.
//!
//! The relay's own end fails the run (`relay_died`); with
//! `RELAY_DIES_QUIETLY` it only marks its death and wakes the waiters, so
//! the two quiet scenarios show that the wait for landings ends the run by
//! its own check: woken while it waits, and before it waits.
//!
//! Each scenario runs in a child process of its own (the relay's switches
//! are read once a process), on the software adapter only; the parent kills
//! a child that has not ended within the bound and fails.

use super::*;
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::Ordering::SeqCst;
use std::time::{Duration, Instant};

/// The most a child may take from its start to its end. Ended loudly it
/// takes a few seconds on the software adapter, the device's making with
/// them.
const BOUND: Duration = Duration::from_secs(30);

/// Set by the parent: a child's body runs only in a child.
const CHILD: &str = "VICTORY_ALIVE_CHILD";

/// What a child must not inherit: switches that turn the sent-ahead path off
/// or change it, settings a hardware gate sets, and a live control file.
const CLEARED: &[&str] = &[
    "VICTORY_GPU_TIMING",
    "VICTORY_LATE_WAIT",
    "VICTORY_LIVE",
    "VICTORY_RELAY",
    "VICTORY_RELAY_COPY",
    "VICTORY_CARD_STEPS",
    "VICTORY_CARD_ALL",
    "VICTORY_THIN",
    "VICTORY_ASYNC_SUBMIT",
    "VICTORY_FETCH_QUEUE",
    "VICTORY_D3D12_DEBUG",
    "VICTORY_D3D12_GBV",
    "VICTORY_REQUIRE_GPU",
    "VICTORY_ADAPTER",
];

/// Whether the host device is on the software adapter: a diagnostics build
/// with `VICTORY_D3D12_WARP=1`, whose device `warp_adapter` refuses to make
/// on hardware. These fixtures hold a queue and kill processes that hang;
/// elsewhere they say so and run nothing.
fn software() -> bool {
    cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP")
}

/// In a child: the host device on the software adapter. Run by hand (not a
/// child) it says so and the body passes without running.
fn in_child() -> Option<&'static Host> {
    if std::env::var_os(CHILD).is_none() {
        eprintln!("NOT RUN: a child of a dying-thread fixture, run by its parent");
        return None;
    }
    assert!(software(), "a child runs on the software adapter only");
    Some(host().expect("a host device on the software adapter"))
}

/// Runs the module's `name` child in a process of its own with `switches`;
/// its output, and how it ended (`None`: killed at [`BOUND`]).
fn run_child(name: &str, switches: &[(&str, &str)]) -> (String, Option<ExitStatus>, Duration) {
    let path = format!("host::alive_tests::{name}");
    let mut command = Command::new(std::env::current_exe().expect("the test binary"));
    command.args(["--exact", path.as_str(), "--ignored", "--test-threads", "1", "--nocapture"]);
    for var in CLEARED {
        command.env_remove(var);
    }
    command.env(CHILD, "1").env("VICTORY_WARP", "1").env("VICTORY_D3D12_WARP", "1");
    for &(var, value) in switches {
        command.env(var, value);
    }
    let started = Instant::now();
    let mut child = command.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().expect("the child ran");
    let (mut out, mut err) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
    let out = std::thread::spawn(move || {
        let mut s = String::new();
        std::io::Read::read_to_string(&mut out, &mut s).ok();
        s
    });
    let err = std::thread::spawn(move || {
        let mut s = String::new();
        std::io::Read::read_to_string(&mut err, &mut s).ok();
        s
    });
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break Some(status);
        }
        if started.elapsed() > BOUND {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let took = started.elapsed();
    // The two streams apart: a test's name, printed without a line's end
    // before it runs, would join the other stream's first line.
    let text = format!("{}\n{}", out.join().unwrap(), err.join().unwrap());
    for line in text.lines().filter(|l| l.contains("said: ") || l.contains("NOT RUN")) {
        eprintln!("{name}: {line}");
    }
    (text, status, took)
}

/// The first line of `text` that ends the run naming `dead` as died:
/// `fail`'s. The hook's own panic is not one, nor another failure.
fn ends_naming<'a>(text: &'a str, dead: &str) -> Option<&'a str> {
    text.lines()
        .filter_map(|line| line.find("host GPU failure: ").map(|at| &line[at..]))
        .find(|said| said.contains(dead) && said.contains("died"))
}

/// Runs child `name` with `switches`: it must end within [`BOUND`], not
/// passing, its thread killed by the hook (`hooked`, the panic's text), and
/// say in `fail`'s line that `dead` died; where `must`, that line says it.
fn ends_loudly(name: &str, switches: &[(&str, &str)], hooked: &str, dead: &str, must: Option<&str>) {
    if !software() {
        eprintln!("SKIPPED {name}: the software adapter only (VICTORY_WARP=1 VICTORY_D3D12_WARP=1, diagnostics)");
        return;
    }
    let (text, status, took) = run_child(name, switches);
    let Some(status) = status else {
        panic!("{name}: the child did not end within {} s: it still waits on {dead}, which died\n{text}", BOUND.as_secs());
    };
    assert!(text.contains(hooked), "{name}: the thread died by the hook (exit {:?}):\n{text}", status.code());
    assert!(!text.contains("said: the wait returned"), "{name}: the wait returned as if the flight had settled:\n{text}");
    assert!(!status.success(), "{name}: the child passed:\n{text}");
    let Some(end) = ends_naming(&text, dead) else {
        panic!("{name}: the run ended without naming {dead} dead (exit {:?}):\n{text}", status.code());
    };
    eprintln!("{name}: ended in {:.1} s: {end}", took.as_secs_f64());
    if let Some(must) = must {
        assert!(end.contains(must), "{name}: the loud end does not say \"{must}\": {end}\n{text}");
    }
}

/// The hook's own panic is no loud end; `fail`'s line naming the dead
/// thread is; another failure is not.
#[test]
fn a_loud_end_is_told_from_the_hooks_own_panic() {
    let hook = "thread 'victory-relay' (1) panicked at crates\\victory-gpu\\src\\host.rs:1:1:\n\
                test: the thread that lands flights panics, holding 1 of the line's items\n";
    assert_eq!(ends_naming(hook, "relay thread"), None, "the hook's panic");
    let failed = format!("{hook}host GPU failure: the relay thread, which lands submissions sent ahead, died: it panicked\n");
    assert_eq!(
        ends_naming(&failed, "relay thread"),
        Some("host GPU failure: the relay thread, which lands submissions sent ahead, died: it panicked"),
        "fail's line"
    );
    assert_eq!(ends_naming(&failed, "late helper"), None, "another thread's death");
    let other = format!("{hook}host GPU failure: host GPU device removed during a list of ours\n");
    assert_eq!(ends_naming(&other, "relay thread"), None, "another failure");
}

/// An empty list of ours sent ahead: one flight, not waited for. The caller
/// keeps the list.
fn one_flight(host: &Host) -> HostList {
    let list = host.new_list().expect("a list");
    list.close();
    assert_ne!(host.execute_own(&list), 0, "the flight went up without a wait");
    list
}

/// The card held: the host's queue waits for a fence of ours, which a
/// thread signals from the CPU after `after`.
fn hold_the_card(host: &Host, after: Duration) {
    // SAFETY: a plain fence of our own.
    let fence: ID3D12Fence = unsafe { host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.expect("a fence");
    let gate = fence.clone();
    std::thread::spawn(move || {
        std::thread::sleep(after);
        eprintln!("said: the card is let go");
        // SAFETY: a signal of our fence from the CPU.
        unsafe { gate.Signal(1) }.expect("the card let go");
    });
    // SAFETY: a fence of this device; the thread above signals it.
    unsafe { host.queue.Wait(&fence, 1) }.expect("the queue waits on the fence");
}

/// Until `what` holds, for at most 10 s (`said` if not).
fn until(what: impl Fn() -> bool, said: &str) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !what() {
        assert!(Instant::now() < deadline, "{said} within 10 s");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The relay's switches for a child.
const RELAY_ON: &[(&str, &str)] = &[("VICTORY_RELAY", "2")];
/// What the relay's hook says as it panics.
const RELAY_HOOKED: &str = "test: the thread that lands flights panics";

/// What the wait for landings adds to the relay's death as it fails the run
/// (`relay_wait`): the quiet scenarios' loud end must be its own.
const WAITERS_END: &str = "a wait for its landings";

/// A child: the relay dies holding a flight; then the title settles, once
/// the relay has taken the flight and panicked (its unwinding under way or
/// done; either must be found). `quiet`: the relay's end does not fail the
/// run (`RELAY_DIES_QUIETLY`), and the settle begins only once the death is
/// marked, so the wait's check before it waits must end the run.
fn dies_before_the_settle(quiet: bool) {
    let Some(host) = in_child() else {
        return;
    };
    assert!(relay(), "the child runs with VICTORY_RELAY=2");
    RELAY_DIES_QUIETLY.store(quiet, SeqCst);
    RELAY_PANICS.store(true, SeqCst);
    let _list = one_flight(host);
    // The hook clears the switch as it panics.
    until(|| !RELAY_PANICS.load(SeqCst), "the relay took the flight");
    if quiet {
        until(|| line().relay_gone.is_some(), "the relay's death was marked");
        eprintln!("said: settling, the relay's death marked");
    } else {
        eprintln!("said: settling, the relay's hook having fired with the flight in hand");
    }
    settle_now(WHY_SYNC);
    eprintln!("said: the wait returned");
    panic!("the settle returned: the flight in hand never landed");
}

/// A child: the flight goes up behind a held card, let go half a second
/// later, so the settle is (most likely) inside its wait when the relay
/// takes the flight and dies. Either order must end the run. `quiet`: the
/// relay's end does not fail the run, so the wait, woken by the relay's end
/// or finding its death marked, must end it by its own check.
fn dies_under_a_waiting_settle(quiet: bool) {
    let Some(host) = in_child() else {
        return;
    };
    assert!(relay(), "the child runs with VICTORY_RELAY=2");
    RELAY_DIES_QUIETLY.store(quiet, SeqCst);
    hold_the_card(host, Duration::from_millis(500));
    RELAY_PANICS.store(true, SeqCst);
    let _list = one_flight(host);
    eprintln!("said: settling, one flight behind the held card");
    settle_now(WHY_SYNC);
    eprintln!("said: the wait returned");
    panic!("the settle returned: the flight in hand never landed");
}

/// The relay dies holding a flight; then the title settles.
#[test]
fn a_settle_after_the_relay_died_holding_a_flight_ends_the_run_loudly() {
    ends_loudly("a_settle_after_the_relay_died_holding_a_flight_child", RELAY_ON, RELAY_HOOKED, "relay thread", None);
}

#[test]
#[ignore = "a child process of a_settle_after_the_relay_died_holding_a_flight_ends_the_run_loudly"]
fn a_settle_after_the_relay_died_holding_a_flight_child() {
    dies_before_the_settle(false);
}

/// The settle is waiting for the flight when the relay takes it and dies.
#[test]
fn a_settle_waiting_when_the_relay_dies_ends_the_run_loudly() {
    ends_loudly("a_settle_waiting_when_the_relay_dies_child", RELAY_ON, RELAY_HOOKED, "relay thread", None);
}

#[test]
#[ignore = "a child process of a_settle_waiting_when_the_relay_dies_ends_the_run_loudly"]
fn a_settle_waiting_when_the_relay_dies_child() {
    dies_under_a_waiting_settle(false);
}

/// As [`a_settle_waiting_when_the_relay_dies_ends_the_run_loudly`], the
/// relay's end not failing the run: the waiting settle ends it.
#[test]
fn a_settle_waiting_when_a_quiet_relay_dies_ends_the_run_by_its_own_check() {
    ends_loudly(
        "a_settle_waiting_when_a_quiet_relay_dies_child",
        RELAY_ON,
        RELAY_HOOKED,
        "relay thread",
        Some(WAITERS_END),
    );
}

#[test]
#[ignore = "a child process of a_settle_waiting_when_a_quiet_relay_dies_ends_the_run_by_its_own_check"]
fn a_settle_waiting_when_a_quiet_relay_dies_child() {
    dies_under_a_waiting_settle(true);
}

/// As [`a_settle_after_the_relay_died_holding_a_flight_ends_the_run_loudly`],
/// the relay's end not failing the run: the settle, beginning after the
/// death is marked, ends it before it waits.
#[test]
fn a_settle_after_a_quiet_relay_died_ends_the_run_by_its_own_check() {
    ends_loudly("a_settle_after_a_quiet_relay_died_child", RELAY_ON, RELAY_HOOKED, "relay thread", Some(WAITERS_END));
}

#[test]
#[ignore = "a child process of a_settle_after_a_quiet_relay_died_ends_the_run_by_its_own_check"]
fn a_settle_after_a_quiet_relay_died_child() {
    dies_before_the_settle(true);
}

/// The late helper dies woken with a flight to settle; a title thread waits
/// on a fence's event that the flight's settle would set.
#[test]
fn a_title_wait_on_a_fence_the_dead_late_helper_would_set_ends_the_run_loudly() {
    ends_loudly(
        "a_title_wait_on_a_fence_the_dead_late_helper_would_set_child",
        &[("VICTORY_RELAY", "0"), ("VICTORY_LATE_WAIT", "1")],
        "test: the late helper panics",
        "late helper",
        None,
    );
}

/// The parent's child: a title fence's signal goes behind one flight sent
/// ahead, and this thread waits, as the title does, on the fence's event.
#[test]
#[ignore = "a child process of a_title_wait_on_a_fence_the_dead_late_helper_would_set_ends_the_run_loudly"]
fn a_title_wait_on_a_fence_the_dead_late_helper_would_set_child() {
    use crate::object::{HRESULT, S_OK, This, slot};
    use windows_sys::Win32::System::Threading as sys;
    let Some(host) = in_child() else {
        return;
    };
    assert!(late() && !relay(), "the child runs with VICTORY_LATE_WAIT=1 VICTORY_RELAY=0");
    let mut fence: This = std::ptr::null();
    assert_eq!(crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut fence), S_OK);
    // SAFETY: a plain auto-reset event.
    let event = unsafe { sys::CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
    assert!(!event.is_null(), "an event");
    HELPER_PANICS.store(true, SeqCst);
    let _list = one_flight(host);
    // The helper cannot have settled the flight: it panics as it wakes.
    assert!(late_signal(fence as usize, 1), "the fence's signal goes behind the flight");
    // ID3D12Fence::SetEventOnCompletion.
    let on_completion = slot::<extern "system" fn(This, u64, *mut std::ffi::c_void) -> HRESULT>(fence, 0x48);
    assert_eq!(on_completion(fence, 1, event), S_OK);
    eprintln!("said: waiting, as the title does, on the event of the fence the late helper would set");
    // SAFETY: our event.
    unsafe { sys::WaitForSingleObject(event, sys::INFINITE) };
    eprintln!("said: the wait returned");
    panic!("the fence was set: the flight was settled after all");
}
