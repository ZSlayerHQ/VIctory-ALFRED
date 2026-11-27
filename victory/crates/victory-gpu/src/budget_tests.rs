//! Fixtures for what the check of the frame budget asked for (`jobs/2026-11-26-live-switches`
//! in the outer repository): `title` is only the title, every title-facing entry opens a
//! scope, other locks' waits are lock waits, a scope stays on its thread, and the frame says
//! how many calls it held and its largest stretch outside them.

use super::*;

/// Nanoseconds in a millisecond: the hand clock's figures ([`hand`]) are exact.
const MS: u64 = 1_000_000;

fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

/// The phase called `name`.
fn named(name: &str) -> usize {
    NAMES.iter().position(|n| *n == name).unwrap_or_else(|| panic!("no phase is called {name}: {NAMES:?}"))
}

/// The phases are named for what they cover: the title's calls by the kind of table they
/// come through, the submission lock's wait apart from the queues' locks'. The names that
/// said more than they measured are gone.
#[test]
fn the_phases_are_named_for_what_they_cover() {
    for name in ["title", "list_calls", "object_calls", "hle_calls", "submission_lock_wait", "queue_lock_wait", "pace_wait"] {
        named(name);
    }
    for gone in ["recording", "lock_wait"] {
        assert!(!NAMES.contains(&gone), "{gone} is still a phase: {NAMES:?}");
    }
}

/// A call the title makes into the runtime is not the title's time: an entry scope's time is
/// its phase's, `title` holds only the time outside every scope, a queue call's phase inside
/// an entry suspends it, and an entry inside a scope of ours changes nothing. The frame says
/// how many calls came from the title and the longest stretch of `title` between two. No
/// processor time is given for `title` or a kind of call alone: only for all of them together.
/// On a hand clock: the figures are exact.
#[test]
fn a_title_facing_call_is_not_the_title_s() {
    std::thread::spawn(|| {
        hand::set();
        frame();
        hand::advance_wall(30); // the title
        {
            let _call = entry(Phase::ObjectCalls);
            hand::compute(40);
        }
        hand::advance_wall(20); // the title again
        {
            let _call = entry(Phase::HleCalls);
            hand::advance_wall(25);
            {
                let _ours = phase(Phase::Unaccounted);
                hand::advance_wall(10);
                // A call of ours from inside our own work is that work's.
                let _inner = entry(Phase::ObjectCalls);
                hand::advance_wall(15);
            }
        }
        frame();
        let taken = take();
        assert_eq!(taken.wall.iter().sum::<u64>(), taken.ns, "{taken:?}");
        let of = |name: &str| taken.wall[named(name)];
        assert_eq!(of("title"), 50 * MS, "title holds a call of ours: {} ms for 50 ({taken:?})", ms(of("title")));
        assert_eq!(of("object_calls"), 40 * MS, "object_calls {} ms for 40", ms(of("object_calls")));
        assert_eq!(of("hle_calls"), 25 * MS, "hle_calls {} ms for 25", ms(of("hle_calls")));
        assert_eq!(of("unaccounted"), 25 * MS, "unaccounted {} ms for 25", ms(of("unaccounted")));
        // Two calls came from the title; the longest stretch of title before one was 30 ms.
        assert_eq!(taken.calls, 2, "{taken:?}");
        assert_eq!(taken.largest_gap, 30 * MS, "the largest gap {} ms for 30", ms(taken.largest_gap));
        // Processor time: none for the title alone, and the 40 ms of computing in a call are
        // in the figure for the title and its calls together, and in no phase of ours.
        for name in ["title", "object_calls", "hle_calls"] {
            assert_eq!(taken.cpu_ns(named(name)), None, "{name} is given a processor time of its own");
        }
        let together = taken.entry_cpu_ns().expect("the counter's rate") / 1e6;
        assert_eq!(together, 40.0, "the title and its calls on the processor: {together} ms for 40");
        assert_eq!(taken.cpu[named("unaccounted")], 0, "a call's computing is in the queue call's processor time");
    })
    .join()
    .unwrap();
}

/// A scope ends on the thread that began it: the type cannot be sent to another.
#[test]
fn a_scope_cannot_be_sent() {
    // The inherent method is chosen for a type that can be sent, the trait's for one that cannot.
    struct Probe<T>(std::marker::PhantomData<T>);
    #[allow(dead_code)]
    trait NotSend {
        fn sendable(&self) -> bool {
            false
        }
    }
    impl<T> NotSend for Probe<T> {}
    #[allow(dead_code)]
    impl<T: Send> Probe<T> {
        fn sendable(&self) -> bool {
            true
        }
    }
    assert!(!Probe::<Scope>(std::marker::PhantomData).sendable(), "a scope can be sent to another thread");
}

/// A wait for a lock another thread holds is a lock wait of its own, not unaccounted time;
/// a lock that is free opens no scope and reads no clock. The wait is a real one (this thread
/// blocks on the lock) on a hand clock: the holder lets the lock go only once this thread has
/// read the clock as its wait begins, and 80 ms after that on the clock; so the wait is 80 ms
/// exactly, and the time outside it none.
#[test]
fn a_wait_for_a_queue_lock_is_a_lock_wait() {
    std::thread::spawn(|| {
        static HELD: std::sync::Mutex<u32> = std::sync::Mutex::new(0);
        let wall = hand::set();
        frame();
        let call = phase(Phase::Unaccounted);
        // Free: taken at once, with no clock read.
        let before = peek_open();
        let reads = hand::reads();
        drop(lock(&HELD, Phase::QueueLockWait));
        assert_eq!(hand::reads(), reads, "a free lock read a clock");
        // Held by another thread until this one waits for it.
        let (taken, held) = std::sync::mpsc::channel::<()>();
        let (asking, asked) = std::sync::mpsc::channel::<u64>();
        let holder = std::thread::spawn(move || {
            let guard = HELD.lock().unwrap();
            taken.send(()).unwrap();
            let read = asked.recv().unwrap();
            // The wait has begun when the waiter has read the wall clock once more. The bound
            // only ends the test if the wait never reads it (it opened no scope): no lock
            // would be let go otherwise, and the test would hang.
            let until = std::time::Instant::now() + std::time::Duration::from_secs(60);
            while wall.reads() == read && std::time::Instant::now() < until {
                std::thread::yield_now();
            }
            let began = wall.reads() > read;
            wall.advance(80);
            drop(guard);
            began
        });
        held.recv().unwrap();
        asking.send(wall.reads()).unwrap();
        drop(lock(&HELD, Phase::QueueLockWait));
        assert!(holder.join().unwrap(), "the wait for a held lock read no clock in a minute: it opened no scope");
        let after = peek_open();
        let waited = after[named("queue_lock_wait")] - before[named("queue_lock_wait")];
        assert_eq!(waited, 80 * MS, "80 ms waiting for a lock counted as {} ms of queue_lock_wait", ms(waited));
        let unaccounted = after[named("unaccounted")] - before[named("unaccounted")];
        assert_eq!(unaccounted, 0, "the wait for the lock is unaccounted: {} ms", ms(unaccounted));
        call.end();
        frame();
        assert_eq!(take().scopes, 2, "the scopes: ours and the one wait (a free lock opens none)");
    })
    .join()
    .unwrap();
}

/// Every `.rs` file under `dir`, with its path from the workspace's root.
fn sources(dir: &std::path::Path, out: &mut Vec<(String, String)>) {
    for item in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let path = item.expect("a directory entry").path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            let text = std::fs::read_to_string(&path).expect("a source file");
            let name = path.to_string_lossy().replace('\\', "/");
            let name = name.rsplit_once("/crates/").map_or(name.clone(), |(_, tail)| tail.to_owned());
            out.push((name, text.replace("\r\n", "\n")));
        }
    }
}

/// The functions `text` puts into a table the title calls through: each `name as *const ()`
/// (a method table's entry) and each `hle_export!(name)`.
fn table_entries(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    for (at, _) in text.match_indices(" as *const ()") {
        let mut before = &text[..at];
        // A generic entry: `name::<A, B>`.
        if before.ends_with('>') {
            before = before.rfind("::<").map_or(before, |open| &before[..open]);
        }
        let start = before.rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).map_or(0, |i| i + 1);
        if start < before.len() {
            names.push(before[start..].to_owned());
        }
    }
    for (at, found) in text.match_indices("hle_export!(") {
        let rest = &text[at + found.len()..];
        if let Some(end) = rest.find(')') {
            names.push(rest[..end].trim().to_owned());
        }
    }
    names
}

/// Where each definition of an `extern` function called `name` begins in `text`, as the
/// first line of its body (the line after the one its signature's `{` ends); `Err` with the
/// signature's line for a body on one line.
fn bodies<'a>(text: &'a str, name: &str) -> Vec<Result<&'a str, &'a str>> {
    let mut found = Vec::new();
    for pattern in [format!("fn {name}("), format!("fn {name}<")] {
        for (at, _) in text.match_indices(&pattern) {
            let line_start = text[..at].rfind('\n').map_or(0, |i| i + 1);
            let head = &text[line_start..at];
            if !head.contains("extern \"") || head.trim_start().starts_with("//") {
                continue;
            }
            // A declaration (`fn name(...);`) has no body.
            let Some(open) = text[at..].find('{') else { continue };
            if text[at..at + open].contains(';') {
                continue;
            }
            let after = &text[at + open + 1..];
            let signature = text[line_start..].lines().next().unwrap_or("");
            if !after.starts_with('\n') {
                found.push(Err(signature));
                continue;
            }
            // The first line of the body that is neither empty, a comment nor an attribute.
            let first = after
                .lines()
                .map(str::trim)
                .find(|line| !line.is_empty() && !line.starts_with("//") && !line.starts_with("#["))
                .unwrap_or("");
            found.push(Ok(first));
        }
    }
    found
}

/// Every function the title calls through a table of ours opens a scope first, so its time is
/// a phase's and never `title`'s: the methods in `victory-gpu`'s object tables (lists, the
/// device, descriptor heaps, resources, fences, queues and the rest), and the exports and
/// object methods of `victory-hle`. Read from the sources, so a method added to a table
/// without a scope fails here.
#[test]
fn every_title_facing_entry_opens_a_scope() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("crates");
    let mut missing = Vec::new();
    let mut checked = 0;
    for (krate, opens) in [("victory-gpu", ["budget::entry(", "budget::phase("]), ("victory-hle", ["victory_gpu::hle_call(", "hle_call("])] {
        let mut files = Vec::new();
        sources(&root.join(krate).join("src"), &mut files);
        let mut names: Vec<String> = files.iter().flat_map(|(_, text)| table_entries(text)).collect();
        names.sort();
        names.dedup();
        for name in &names {
            for (file, text) in &files {
                for body in bodies(text, name) {
                    checked += 1;
                    match body {
                        Ok(first) if opens.iter().any(|open| first.contains(open)) => {}
                        Ok(first) => missing.push(format!("{file}: {name} begins with `{first}`")),
                        Err(signature) => missing.push(format!("{file}: {name} has its body on one line: `{}`", signature.trim())),
                    }
                }
            }
        }
    }
    assert!(checked > 300, "only {checked} table entries were found: the tables are not being read");
    assert!(
        missing.is_empty(),
        "{} of {checked} title-facing entries open no scope (the first 12): {:#?}",
        missing.len(),
        &missing[..missing.len().min(12)]
    );
}
