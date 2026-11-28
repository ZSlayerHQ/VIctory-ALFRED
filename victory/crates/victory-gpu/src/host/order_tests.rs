//! Whether the title's buffer copies keep the list's order, and whether each
//! submission reads its own bytes (the other tree's five failing
//! fixtures, rebuilt on this tree's own API).
//!
//! Each test runs its body in a child process of its own, once per set of
//! switches in [`WAYS`]: the relay's statics are read once a process, and
//! the crate's tests run without the relay unless asked. The copies go the
//! way a title list's do (`list.rs`'s `buffer_copy`): at their place in the
//! list ([`HostList::place_copy`]), else at submission
//! ([`HostList::defer_copy`]). Every check is of bytes read back from the
//! card. On the software adapter only: elsewhere a test says so and returns
//! without submitting anything.

use super::*;
use crate::shadow::Shadow;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;

/// Set in a child: run the body.
const CHILD: &str = "VICTORY_ORDER_CHILD";

/// The switches of each way a body runs: the relay's second stage with the
/// crate's defaults, and the play preset's copy machinery
/// (`apps/victory/src/play.rs`, "Submission and copy machinery").
const WAYS: [(&str, &[(&str, &str)]); 2] = [
    ("relay", &[("VICTORY_RELAY", "2")]),
    (
        "play preset",
        &[
            ("VICTORY_THIN", "1"),
            ("VICTORY_JOIN_PARTS", "8"),
            ("VICTORY_RELAY", "2"),
            ("VICTORY_RELAY_COPY", "1"),
            ("VICTORY_WIDE_BACK", "1"),
            ("VICTORY_CARD_REHASH_RUNS", "1"),
            ("VICTORY_JOIN_COPIES", "1"),
            ("VICTORY_CARD_STEPS", "2"),
            ("VICTORY_CARD_AWAKE", "1"),
            ("VICTORY_CARD_BUFFERS_MIN_KB", "64"),
            ("VICTORY_UPLOAD_AHEAD", "1"),
            ("VICTORY_WC_NO_BACK", "1"),
            ("VICTORY_WC_NO_BACK_MIN_KB", "1024"),
        ],
    ),
];

/// What a child must not inherit: switches that turn the relay's sent-ahead
/// path off or change it, and a live control file.
const CLEARED: &[&str] = &[
    "VICTORY_GPU_TIMING",
    "VICTORY_LATE_WAIT",
    "VICTORY_LIVE",
    "VICTORY_RELAY",
    "VICTORY_RELAY_COPY",
    "VICTORY_CARD_STEPS",
    "VICTORY_CARD_ALL",
    "VICTORY_CARD_BUFFERS_MIN_KB",
    "VICTORY_COPIES_IN_PLACE",
    "VICTORY_SEND_OWN",
];

/// Runs `test` (its full name under this module) in a child process for each
/// of [`WAYS`], with `drop` taken out of the way's switches; fails with the
/// child's output if one fails, does not run the test, or takes over two
/// minutes.
fn children(test: &str, drop: &[&str]) {
    children_with(test, drop, &[]);
}

/// [`children`], with `extra` set in every way besides its switches.
fn children_with(test: &str, drop: &[&str], extra: &[(&str, &str)]) {
    if !(cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP")) {
        eprintln!("SKIPPED {test}: the software adapter only (VICTORY_WARP=1 VICTORY_D3D12_WARP=1, diagnostics)");
        return;
    }
    let name = format!("host::order_tests::{test}");
    for (way, switches) in WAYS {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command.args([name.as_str(), "--exact", "--nocapture", "--test-threads=1"]);
        for var in CLEARED {
            command.env_remove(var);
        }
        // The least size of a copy in the card's memory scales with the
        // adapter's memory; 1 MiB makes the sizes below decide.
        command.env("VICTORY_CARD_BUFFERS_MIN_MB", "1").env(CHILD, "1");
        for &(var, value) in switches.iter().filter(|(var, _)| !drop.contains(var)).chain(extra) {
            command.env(var, value);
        }
        let mut child = command
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let (mut out, mut err) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
        let read = std::thread::spawn(move || {
            let mut text = String::new();
            std::io::Read::read_to_string(&mut out, &mut text).ok();
            text
        });
        let read_err = std::thread::spawn(move || {
            let mut text = String::new();
            std::io::Read::read_to_string(&mut err, &mut text).ok();
            text
        });
        let began = std::time::Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break Some(status);
            }
            if began.elapsed() > std::time::Duration::from_secs(120) {
                child.kill().ok();
                child.wait().ok();
                break None;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        };
        let text = format!("{}{}", read.join().unwrap(), read_err.join().unwrap());
        let ran = text.contains("test result: ok. 1 passed");
        assert!(
            status.is_some_and(|s| s.success()) && ran,
            "{test}, {way}: the child {}:\n{text}",
            match status {
                None => "took over two minutes".to_owned(),
                Some(s) if s.success() => "did not run the test".to_owned(),
                Some(s) => format!("failed ({s})"),
            }
        );
        eprintln!("{test}, {way}: passed");
    }
}

/// The host, in a child on the software adapter; `None` (said) elsewhere.
fn software_host() -> Option<&'static Host> {
    std::env::var_os(CHILD)?;
    let host = host().expect("a host device in the child");
    let software = host.adapter.as_ref().is_some_and(|a| {
        // SAFETY: a plain query of the device's adapter.
        let desc = unsafe { a.cast::<IDXGIAdapter1>().and_then(|a| a.GetDesc1()) };
        desc.is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0)
    });
    assert!(software, "the child's device is not on the software adapter: refused");
    assert!(late() && relay_exact(), "the child runs with the relay's second stage");
    Some(host)
}

/// Title memory of our own, write-watched, and the host copy of the title
/// buffer it is.
fn title_of(resource: usize, len: usize) -> (&'static mut [u8], usize, Arc<Shadow>) {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    const WRITE_WATCH: u32 = 0x20_0000;
    // SAFETY: memory of our own, never freed (the child ends).
    let base = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT | WRITE_WATCH, PAGE_READWRITE) };
    assert!(!base.is_null());
    let base = base.cast::<u8>();
    crate::shadow::register(resource, base as u64, len as u64);
    let shadow = crate::shadow::find(base as u64, 64).expect("a host copy");
    // SAFETY: `len` bytes just committed.
    (unsafe { std::slice::from_raw_parts_mut(base, len) }, base as usize, shadow)
}

/// The whole of `shadow` taken once, nothing in the air after.
fn taken_whole(host: &Host, shadow: &Arc<Shadow>, lists: &mut Vec<HostList>) {
    let (start, len) = shadow.span();
    let all = host.new_list().unwrap();
    shadow.refresh(&all, start, len);
    all.close();
    host.execute_own(&all);
    settle_now(WHY_SYNC);
    lists.push(all);
}

/// A title buffer copy, as a title list records it. Whether it was placed.
fn title_copy(list: &HostList, dst: usize, src: usize, len: usize) -> bool {
    let placed = list.place_copy(dst, src, len);
    if !placed {
        list.defer_copy(dst as *mut u8, src as *const u8, len);
    }
    placed
}

/// Work for the card that takes some time.
struct Slow(ID3D12Resource, ID3D12Resource);

impl Slow {
    fn new(host: &Host) -> Slow {
        Slow(host.uav_buffer(64 << 20).unwrap(), host.uav_buffer(64 << 20).unwrap())
    }

    fn record(&self, list: &HostList, times: usize) {
        // SAFETY: an open list; live buffers.
        unsafe {
            for _ in 0..times {
                list.list.CopyBufferRegion(&self.1, 0, &self.0, 0, 64 << 20);
                list.list.ResourceBarrier(&[transition(&self.1, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
            }
        }
    }
}

/// A shader's write, stood in for: `bytes` go to offset `at` of the copy,
/// declared written first.
fn write(host: &Host, list: &HostList, shadow: &Arc<Shadow>, at: usize, bytes: &[u8]) {
    shadow.gpu_writes(list, shadow.span().0 + at as u64, bytes.len() as u64);
    let (upload, data) = host.upload_buffer(bytes.len() as u64).unwrap();
    // SAFETY: the mapping; an open list and live buffers.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len());
        list.list.CopyBufferRegion(&shadow.buffer, at as u64, &upload, 0, bytes.len() as u64);
        list.list.ResourceBarrier(&[transition(&shadow.buffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
    }
    list.keep(&upload);
}

/// What a draw of `list` reads at `at` of the copy: `n` bytes to `out`.
fn read_out(list: &HostList, shadow: &Arc<Shadow>, at: usize, n: usize, out: (&ID3D12Resource, u64)) {
    shadow.refresh(list, shadow.span().0 + at as u64, n as u64);
    // SAFETY: an open list; live buffers.
    unsafe {
        list.list.CopyBufferRegion(out.0, out.1, &shadow.buffer, at as u64, n as u64);
        list.list.ResourceBarrier(&[transition(&shadow.buffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
    }
}

/// Scenario 1: in each of six rounds, three submissions copy another value
/// to the same bytes and read them, prepared while slow card work ahead of
/// them is in the air. Each must read its own value.
fn each_reads_its_own(len: usize, resource: usize, carded: bool) {
    let Some(host) = software_host() else {
        return;
    };
    let (title, start, shadow) = title_of(resource, len);
    assert_eq!(shadow.carded(), carded, "where the host copy is");
    let slow = Slow::new(host);
    let (out, out_data) = host.shared_buffer(1 << 16).unwrap();
    let mut lists = Vec::new();
    taken_whole(host, &shadow, &mut lists);
    const AT: usize = 8192;
    const FROM: usize = 65_536;
    let (mut wrong, mut placed, mut in_air) = (Vec::new(), 0, 0);
    for round in 0..6u8 {
        for v in 0..3u8 {
            title[FROM + usize::from(v) * 256..][..256].fill(round * 16 + v + 1);
        }
        let busy = host.new_list().unwrap();
        slow.record(&busy, 40);
        busy.close();
        host.execute_own(&busy);
        lists.push(busy);
        for v in 0..3usize {
            let list = host.new_list().unwrap();
            placed += usize::from(title_copy(&list, start + AT, start + FROM + v * 256, 256));
            read_out(&list, &shadow, AT, 256, (&out, (v * 256) as u64));
            list.close();
            in_air += usize::from(flying());
            host.execute_own(&list);
            lists.push(list);
        }
        settle_now(WHY_SYNC);
        // SAFETY: a mapping of 64 KiB; the copies into it have run.
        let read = unsafe { std::slice::from_raw_parts(out_data, 768) };
        for v in 0..3u8 {
            let got = read[usize::from(v) * 256];
            if got != round * 16 + v + 1 {
                wrong.push((round, v, got.wrapping_sub(round * 16)));
            }
        }
    }
    eprintln!("{placed} of 18 copies placed in their list; {in_air} of 18 submissions prepared with flights in the air");
    assert!(wrong.is_empty(), "each submission must read its own bytes (round, submission, whose + 1): {wrong:?}");
    assert_eq!(in_air, 18, "every submission was prepared with flights in the air");
    assert_eq!(title[AT], 5 * 16 + 3, "title memory holds the last copy");
}

#[test]
fn each_submission_reads_its_own_copy_on_the_card() {
    if std::env::var_os(CHILD).is_some() {
        return each_reads_its_own(4 << 20, 0x0de1, true);
    }
    children("each_submission_reads_its_own_copy_on_the_card", &[]);
}

#[test]
fn each_submission_reads_its_own_copy_in_pc_memory() {
    if std::env::var_os(CHILD).is_some() {
        return each_reads_its_own(256 << 10, 0x0de2, false);
    }
    // Under the preset's 64 KiB this buffer would be in the card's memory.
    children("each_submission_reads_its_own_copy_in_pc_memory", &["VICTORY_CARD_BUFFERS_MIN_KB"]);
}

const D: usize = 0x10000;
const S: usize = 0x20000;

/// Scenario 2: one list, its steps recorded by `record` on a 4 MiB title
/// buffer whose first 192 bytes are set by `set`; the bytes read at the
/// offsets given come back in order, the whole list submitted with nothing
/// in the air ahead of it.
fn one_list(
    resource: usize,
    set: impl FnOnce(&mut [u8]),
    record: impl FnOnce(&Host, &HostList, usize, &Arc<Shadow>, &ID3D12Resource),
) -> (Vec<u8>, &'static mut [u8]) {
    let host = software_host().expect("a child");
    let (title, start, shadow) = title_of(resource, 4 << 20);
    assert!(shadow.carded(), "the host copy is in the card's memory");
    let (out, out_data) = host.shared_buffer(1 << 16).unwrap();
    let mut lists = Vec::new();
    taken_whole(host, &shadow, &mut lists);
    set(title);
    let before = super::place::STATS[0].load(Relaxed);
    let list = host.new_list().unwrap();
    record(host, &list, start, &shadow, &out);
    list.close();
    host.execute_own(&list);
    settle_now(WHY_SYNC);
    eprintln!("{} copies placed in their list", super::place::STATS[0].load(Relaxed) - before);
    // SAFETY: the mapping; the copies into it have run.
    (unsafe { std::slice::from_raw_parts(out_data, 256) }.to_vec(), title)
}

/// Copy 1 to D, a draw reads D, copy 2 to D, a draw reads D.
#[test]
fn copy_draw_copy_draw() {
    if std::env::var_os(CHILD).is_none() {
        return children("copy_draw_copy_draw", &[]);
    }
    let (read, title) = one_list(
        0x0de3,
        |title| {
            title[0..64].fill(1);
            title[64..128].fill(2);
        },
        |_, list, start, shadow, out| {
            title_copy(list, start + D, start, 64);
            read_out(list, shadow, D, 64, (out, 0));
            title_copy(list, start + D, start + 64, 64);
            read_out(list, shadow, D, 64, (out, 64));
        },
    );
    assert_eq!([read[0], read[64]], [1, 2], "what the draws read");
    assert_eq!(title[D], 2, "title memory");
}

/// A shader of the list writes S, and S is copied on: the read behind the
/// copy, and title memory after the landing, hold the shader's value.
#[test]
fn a_copy_takes_what_a_shader_wrote_before_it() {
    if std::env::var_os(CHILD).is_none() {
        return children("a_copy_takes_what_a_shader_wrote_before_it", &[]);
    }
    let (read, title) = one_list(
        0x0de4,
        |_| {},
        |host, list, start, shadow, out| {
            write(host, list, shadow, S, &[7; 64]);
            title_copy(list, start + D + 64, start + S, 64);
            read_out(list, shadow, D + 64, 64, (out, 0));
        },
    );
    assert_eq!(read[0], 7, "what the draw reads");
    assert_eq!([title[S], title[D + 64]], [7, 7], "title memory");
}

/// A counter holding 3 is copied to zero, a shader counts it to 5, and it is
/// copied on: the read behind the copy, and title memory, hold 5.
#[test]
fn a_copy_takes_a_counter_a_shader_counted_up() {
    if std::env::var_os(CHILD).is_none() {
        return children("a_copy_takes_a_counter_a_shader_counted_up", &[]);
    }
    let (read, title) = one_list(
        0x0de5,
        |title| {
            title[128..192].fill(0);
            title[D + 128..D + 192].fill(3);
        },
        |host, list, start, shadow, out| {
            title_copy(list, start + D + 128, start + 128, 64);
            write(host, list, shadow, D + 128, &[5; 64]);
            title_copy(list, start + D + 192, start + D + 128, 64);
            read_out(list, shadow, D + 192, 64, (out, 0));
        },
    );
    assert_eq!(read[0], 5, "what the draw reads");
    assert_eq!([title[D + 128], title[D + 192]], [5, 5], "title memory");
}

/// The count of copies kept under flights ([`kept_under_flights`]): its change over the
/// submission of `later`, sent with slow card work of our own in the air
/// ahead of it (which declares no write), or with nothing in the air.
fn counted(host: &Host, lists: &mut Vec<HostList>, behind: Option<&Slow>, later: HostList) -> [u64; 3] {
    if let Some(slow) = behind {
        let busy = host.new_list().unwrap();
        slow.record(&busy, 40);
        busy.close();
        host.execute_own(&busy);
        lists.push(busy);
    }
    later.close();
    assert_eq!(flying(), behind.is_some(), "what is in the air when the later submission goes");
    let before = kept_under_flights();
    host.execute_own(&later);
    let after = kept_under_flights();
    lists.push(later);
    settle_now(WHY_SYNC);
    [0, 1, 2].map(|i| after[i] - before[i])
}

/// The case counted: a copy kept for submission (a title buffer copy, and
/// `copy_memory`'s step) into a host copy in the PC's memory, behind a
/// flight. `big`: a copy of 9 MiB on default settings, kept as it is over
/// 8 MiB; else 64 bytes with `VICTORY_COPIES_IN_PLACE=0`.
fn kept_under_a_flight_counts_one(resource: usize, len: usize, big: bool) {
    let Some(host) = software_host() else {
        return;
    };
    let (title, start, shadow) = title_of(resource, len);
    assert!(!shadow.carded(), "the host copy is in the PC's memory");
    let slow = Slow::new(host);
    let mut lists = Vec::new();
    taken_whole(host, &shadow, &mut lists);
    let n = if big { 9 << 20 } else { 64 };
    let (d, s) = (0x2000, 0x2000 + n);
    title[s..s + n].fill(8);
    let later = host.new_list().unwrap();
    assert!(!title_copy(&later, start + d, start + s, n), "the copy is kept for submission");
    assert_eq!(counted(host, &mut lists, Some(&slow), later), [1, n as u64, u64::from(big)], "a title buffer copy");
    let later = host.new_list().unwrap();
    later.defer_writing(start + d, n, || {});
    assert_eq!(counted(host, &mut lists, Some(&slow), later), [1, n as u64, u64::from(big)], "copy_memory's step");
}

#[test]
fn m14_count_a_copy_kept_under_a_flight_counts_one() {
    if std::env::var_os(CHILD).is_some() {
        return kept_under_a_flight_counts_one(0x9b10, 256 << 10, false);
    }
    children_with(
        "m14_count_a_copy_kept_under_a_flight_counts_one",
        &["VICTORY_CARD_BUFFERS_MIN_KB"],
        &[("VICTORY_COPIES_IN_PLACE", "0")],
    );
}

#[test]
fn m14_count_a_big_copy_under_a_flight_counts_one() {
    if std::env::var_os(CHILD).is_some() {
        return kept_under_a_flight_counts_one(0x9b11, 32 << 20, true);
    }
    children_with(
        "m14_count_a_big_copy_under_a_flight_counts_one",
        &["VICTORY_CARD_BUFFERS_MIN_KB"],
        &[("VICTORY_CARD_BUFFERS", "0")],
    );
}

/// Not counted, on default settings: a copy placed in its list (PC memory),
/// a copy kept into a host copy in the card's memory, both behind a flight,
/// and a kept copy into the PC's memory with nothing in the air.
fn ordinary_copies_count_nothing() {
    let Some(host) = software_host() else {
        return;
    };
    let (pc, pc_start, pc_shadow) = title_of(0x9b12, 256 << 10);
    let (card, card_start, card_shadow) = title_of(0x9b13, 4 << 20);
    assert!(!pc_shadow.carded() && card_shadow.carded(), "one host copy in each memory");
    let slow = Slow::new(host);
    let mut lists = Vec::new();
    taken_whole(host, &pc_shadow, &mut lists);
    taken_whole(host, &card_shadow, &mut lists);
    pc[0x6000..0x6040].fill(8);
    card[0x6000..0x6040].fill(8);
    let later = host.new_list().unwrap();
    assert!(title_copy(&later, pc_start + 0x2000, pc_start + 0x6000, 64), "the small copy is placed");
    assert_eq!(counted(host, &mut lists, Some(&slow), later), [0; 3], "a copy placed in its list");
    let later = host.new_list().unwrap();
    later.defer_copy((card_start + 0x2000) as *mut u8, (card_start + 0x6000) as *const u8, 64);
    assert_eq!(counted(host, &mut lists, Some(&slow), later), [0; 3], "a kept copy into the card's memory");
    let later = host.new_list().unwrap();
    later.defer_copy((pc_start + 0x2000) as *mut u8, (pc_start + 0x6000) as *const u8, 64);
    assert_eq!(counted(host, &mut lists, None, later), [0; 3], "a kept copy with nothing in the air");
}

#[test]
fn m14_count_ordinary_copies_count_nothing() {
    if std::env::var_os(CHILD).is_some() {
        return ordinary_copies_count_nothing();
    }
    children("m14_count_ordinary_copies_count_nothing", &["VICTORY_CARD_BUFFERS_MIN_KB"]);
}
