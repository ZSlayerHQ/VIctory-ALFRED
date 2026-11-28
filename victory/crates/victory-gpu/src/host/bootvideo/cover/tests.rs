use super::*;

/// The tests that look at, or set, where the boot video is take turns.
static BOOT_TURN: Mutex<()> = Mutex::new(());

#[test]
fn the_path_turns_at_both_ends_and_never_jumps() {
    let n = 241;
    assert_eq!(path_frame(0, n), 0);
    assert_eq!(path_frame(240, n), 240);
    assert_eq!(path_frame(241, n), 239);
    assert_eq!(path_frame(479, n), 1);
    assert_eq!(path_frame(480, n), 0);
    assert_eq!(path_frame(481, n), 1);
    for pos in 0..2000 {
        let (a, b) = (path_frame(pos, n), path_frame(pos + 1, n));
        assert_eq!(a.abs_diff(b), 1, "at {pos}: {a} then {b}");
    }
    assert_eq!((0..10).map(|p| path_frame(p, 1)).collect::<Vec<_>>(), vec![0; 10]);
    assert_eq!((0..6).map(|p| path_frame(p, 2)).collect::<Vec<_>>(), vec![0, 1, 0, 1, 0, 1]);
}

/// Records of presents at `times` (seconds) with `draws` each, numbered from `first`.
fn records_of(first: u64, start_drawn: u64, presents: &[(f64, u64)]) -> Vec<(u64, f64, u64)> {
    let mut drawn = start_drawn;
    presents
        .iter()
        .enumerate()
        .map(|(i, &(t, draws))| {
            drawn += draws;
            (first + i as u64, t, drawn)
        })
        .collect()
}

/// Feeds them one by one (with a tick after each); the events.
fn feed_all(d: &mut Detector, records: &[(u64, f64, u64)]) -> Vec<(f64, Event)> {
    let mut events = Vec::new();
    for &(frame, t, drawn) in records {
        if let Some(e) = d.present(frame, t, drawn) {
            events.push((t, e));
        }
        if let Some(e) = d.tick(t) {
            events.push((t, e));
        }
    }
    events
}

fn run(presents: &[(f64, u64)]) -> Vec<(f64, Event)> {
    feed_all(&mut Detector::new(), &records_of(1, 0, presents))
}

fn steady(from: f64, seconds: f64, per_second: f64, draws: u64) -> Vec<(f64, u64)> {
    let n = (seconds * per_second) as usize;
    (1..=n).map(|i| (from + i as f64 / per_second, draws)).collect()
}

/// Start screens (30 a second, a dozen draws), the loading section (3.5 a
/// second, 10,000 draws), the world (20 a second, 7,000 draws).
fn real_start() -> Vec<(f64, u64)> {
    let mut p = steady(0.0, 20.0, 30.0, 12);
    p.extend(steady(20.0, 40.0, 3.5, 10_000));
    p.extend(steady(60.0, 10.0, 20.0, 7_000));
    p
}

#[test]
fn a_real_start_begins_once_in_the_loading_and_ends_on_the_world() {
    let events = run(&real_start());
    assert_eq!(events.len(), 2, "{events:?}");
    assert_eq!(events[0].1, Event::Begin);
    assert!((21.0..22.0).contains(&events[0].0), "begins about 1.5 s into the loading: {events:?}");
    assert_eq!(events[1].1, Event::End(Why::World));
    assert!((60.0..61.5).contains(&events[1].0), "ends within a second or so of the world: {events:?}");
}

#[test]
fn the_start_screens_into_the_loading_do_not_disarm() {
    // The second where light start screens meet the loading's first heavy
    // frames averages heavy at 8 or more presents: the median keeps it from
    // disarming (the reason the disarm asks for it).
    let mut p = steady(0.0, 5.0, 30.0, 12);
    p.extend(steady(5.0, 4.0, 3.5, 10_000));
    let events = run(&p);
    assert_eq!(events.first().map(|e| e.1), Some(Event::Begin), "{events:?}");
}

#[test]
fn a_mixed_world_ends_it_by_the_design_rule() {
    // A world whose frames alternate 300 and 3,000 draws (averaging 1,650) at 20 a second.
    let mut p = steady(0.0, 3.0, 3.5, 10_000);
    p.extend((1..=40).map(|i| (3.0 + f64::from(i) / 20.0, if i % 2 == 0 { 300 } else { 3000 })));
    let events = run(&p);
    assert_eq!(events.last().map(|e| e.1), Some(Event::End(Why::World)), "{events:?}");
}

#[test]
fn a_missed_loading_disarms_on_the_first_world_second_and_a_later_dip_never_begins() {
    let mut p = steady(0.0, 10.0, 30.0, 12);
    p.extend(steady(10.0, 5.0, 9.0, 10_000));
    p.extend(steady(15.0, 2.0, 20.0, 7_000));
    p.extend(steady(17.0, 5.0, 3.0, 7_000));
    let events = run(&p);
    assert_eq!(events.first().map(|e| e.1), Some(Event::Disarmed), "{events:?}");
    assert!(events.iter().all(|e| e.1 != Event::Begin), "{events:?}");
    // The same without the world: the dip begins it.
    let mut p = steady(0.0, 10.0, 30.0, 12);
    p.extend(steady(10.0, 5.0, 3.0, 7_000));
    assert!(run(&p).iter().any(|e| e.1 == Event::Begin));
}

#[test]
fn eight_presents_in_any_one_second_are_seen() {
    let mut p = steady(0.0, 3.0, 3.0, 10_000);
    p.extend([0.11, 0.30, 0.40, 0.52, 0.64, 0.76, 0.87, 1.09].map(|t| (3.0 + t, 10_000)));
    assert_eq!(run(&p).last().map(|e| e.1), Some(Event::End(Why::World)));
}

#[test]
fn each_present_must_be_heavy_not_only_the_sum() {
    let even: Vec<(f64, u64)> = (1..=8).map(|i| (f64::from(i) * 0.5, 1600)).collect();
    assert!(run(&even).iter().any(|e| e.1 == Event::Begin));
    let uneven: Vec<(f64, u64)> = (1..=8).map(|i| (f64::from(i) * 0.5, if i % 2 == 0 { 100 } else { 3100 })).collect();
    assert!(run(&uneven).is_empty());
}

#[test]
fn a_stall_a_light_screen_and_the_ceiling_end_it() {
    let loading = steady(0.0, 3.0, 3.5, 10_000);
    let mut d = Detector::new();
    assert!(feed_all(&mut d, &records_of(1, 0, &loading)).iter().any(|e| e.1 == Event::Begin));
    assert_eq!(d.tick(3.0 + STALL_S + 0.1), Some(Event::End(Why::Stall)));
    let mut p = loading.clone();
    p.extend(steady(3.0, 4.0, 30.0, 12));
    assert!(matches!(run(&p).last().map(|e| e.1), Some(Event::End(Why::Light | Why::World))));
    let mut p = loading.clone();
    p.extend(steady(3.0, 4.0, 5.0, 12));
    assert_eq!(run(&p).last().map(|e| e.1), Some(Event::End(Why::Light)));
    // A slow world, which cannot be told from loading: the ceiling.
    let mut p = loading;
    p.extend(steady(3.0, 125.0, 3.0, 7_000));
    assert_eq!(run(&p).last().map(|e| e.1), Some(Event::End(Why::Ceiling)));
}

#[test]
fn a_gap_or_a_present_back_in_time_shows_the_title() {
    // Before a begin: what is known is cleared, no begin from it.
    let mut d = Detector::new();
    let mut events = feed_all(&mut d, &records_of(1, 0, &steady(0.0, 1.2, 3.5, 10_000)));
    events.extend(feed_all(&mut d, &records_of(100, 1_000_000, &steady(1.2, 1.0, 3.5, 10_000))));
    assert!(events.is_empty(), "{events:?}");
    // While covering: a gap ends it.
    let loading = steady(0.0, 3.0, 3.5, 10_000);
    let mut d = Detector::new();
    let mut r = records_of(1, 0, &loading);
    assert!(feed_all(&mut d, &r).iter().any(|e| e.1 == Event::Begin));
    let last = *r.last().expect("records");
    assert_eq!(d.present(last.0 + 2, last.1 + 0.3, last.2 + 10_000), Some(Event::End(Why::Unsure)));
    // While covering: a present earlier than the last ends it too.
    let mut d = Detector::new();
    feed_all(&mut d, &r);
    r.push((last.0 + 1, last.1 - 0.5, last.2 + 10_000));
    assert_eq!(d.present(last.0 + 1, last.1 - 0.5, last.2 + 10_000), Some(Event::End(Why::Unsure)));
}

#[test]
fn a_begin_and_an_end_in_one_look_are_both_seen() {
    // The loading, then the world, all in the ring at one look.
    let mut p = steady(0.0, 3.0, 3.5, 10_000);
    p.extend(steady(3.0, 2.0, 20.0, 7_000));
    let records = records_of(1, 0, &p);
    let (mut d, mut after) = (Detector::new(), 0);
    assert_eq!(feed(&mut d, &records, &mut after), Some(Event::Begin));
    assert!(after < records.last().expect("records").0, "what is behind the begin stays for the next look");
    assert_eq!(feed(&mut d, &records, &mut after), Some(Event::End(Why::World)));
}

#[test]
fn the_keepers_ceiling_holds_whatever_the_rules_state() {
    // The rule is over (it ended, or never began): the keeper's own ceiling still ends it.
    let mut d = Detector::new();
    let _ = d.end(Why::Stall);
    let shared = Shared::default();
    shared.ready.store(true, Ordering::Relaxed);
    let pinned = Title { hwnd: 1, thread: 1, made: 2 };
    let place = Place { at: (0, 0), size: (1, 1), monitor: 0, dpi: 96 };
    let mut after = u64::MAX;
    let why = look(&mut d, &mut after, 1000.0 + CEILING_S, 1000.0, &shared, f1_presses(), pinned, place);
    assert_eq!(why.as_deref(), Some(Why::Ceiling.text()));
}

#[test]
fn unset_nothing_starts_nothing_is_asked_or_written() {
    let _turn = BOOT_TURN.lock().unwrap_or_else(|e| e.into_inner());
    let plan = Plan::parse(std::path::Path::new("v"), "{\"frames\": 4, \"fps\": 24, \"width\": 8, \"height\": 8, \"pattern\": \"%d.qoi\"}", 1)
        .expect("a plan");
    for unset in [None, Some(OsString::from("")), Some(OsString::from("0")), Some(OsString::from("yes"))] {
        assert!(!asked(unset));
    }
    assert!(!switch_on(), "the test environment does not set it");
    assert_eq!(start(&plan, false), Err(None));
    boot_starts(&plan);
    boot_ended(true);
    assert_eq!(STARTED.load(Ordering::Relaxed), 0, "no keeper: no window class, no watch, no ring writes");
    assert!(!PARKED.load(Ordering::Relaxed));
    assert_eq!(BOOT.load(Ordering::Relaxed), PENDING, "unset, the end of the boot video writes nothing");
    assert!(!crate::queue::COVER_WATCHING.load(Ordering::Relaxed));
    // The presenter's window: not looked at, nothing written.
    // SAFETY: a plain query.
    title_window_is(unsafe { GetDesktopWindow() });
    assert_eq!(TITLE_MADE.load(Ordering::Relaxed), 0);
    // The title's lookup asks nothing of a window, even of one of the cover's class.
    let hwnd = window(Place { at: (-32000, -32000), size: (8, 8), monitor: 0, dpi: 96 }).expect("a hidden window");
    assert!(!is_cover(hwnd), "unset, no class name is asked");
    // SAFETY: our own window.
    unsafe { DestroyWindow(hwnd) }.expect("destroyed");
}

#[test]
fn the_ring_takes_one_writer_at_a_time() {
    let mine = |frame: u64| crate::queue::cover_presents().any(|(f, ..)| f == frame);
    let (a, b) = (7_000_000_001, 7_000_000_002);
    // Another writer is in the ring: this present is not recorded (a gap).
    crate::queue::COVER_WRITING.store(true, Ordering::Release);
    crate::queue::cover_note(a);
    crate::queue::COVER_WRITING.store(false, Ordering::Release);
    assert!(!mine(a));
    crate::queue::cover_note(b);
    assert!(mine(b));
}

/// A plain window of another thread, off every monitor and never activated,
/// standing for the title's (the cover's input must reach it).
fn title_stand_in() -> (HWND, Arc<AtomicBool>, std::thread::JoinHandle<()>) {
    extern "system" fn proc(hwnd: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        // SAFETY: the default handling of the test's own window.
        unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
    }
    let (tx, rx) = std::sync::mpsc::channel();
    let quit = Arc::new(AtomicBool::new(false));
    let quitting = Arc::clone(&quit);
    let thread = std::thread::spawn(move || {
        // SAFETY: the test's own class and window, off every monitor.
        unsafe {
            let instance = GetModuleHandleW(None).expect("module");
            let class = WNDCLASSEXW {
                cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                lpfnWndProc: Some(proc),
                hInstance: instance.into(),
                lpszClassName: w!("victory cover test stand-in"),
                ..Default::default()
            };
            RegisterClassExW(&class);
            let hwnd = CreateWindowExW(
                WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
                w!("victory cover test stand-in"),
                w!(""),
                WS_POPUP,
                -32000,
                -32000,
                64,
                36,
                None,
                None,
                Some(instance.into()),
                None,
            )
            .expect("a stand-in");
            let _ = ShowWindow(hwnd, SW_SHOWNOACTIVATE);
            tx.send(hwnd.0 as usize).expect("sent");
            while !quitting.load(Ordering::Acquire) {
                wait(Duration::from_millis(5));
            }
            let _ = DestroyWindow(hwnd);
        }
    });
    let hwnd = HWND(rx.recv().expect("the stand-in") as *mut _);
    (hwnd, quit, thread)
}

/// The visible path, off every monitor (at -32000, -32000: the user's screen never
/// shows it), on the software adapter: shown with its picture attached, it
/// takes no focus and the point under it hits the other thread's window; a
/// panic after the show hides and destroys it. What the desktop's composer
/// draws of it cannot be seen off every monitor: that is the first game run's.
#[test]
fn shown_off_screen_it_is_click_through_never_in_front_and_a_panic_takes_it_down() {
    let _turn = crate::indirect::tests::warp_turn();
    let Some(device) = crate::inplace::tests::warp() else {
        eprintln!("SKIPPED: no software adapter");
        return;
    };
    let (title, quit, thread) = title_stand_in();
    let place = Place { at: (-32000, -32000), size: (64, 36), monitor: 0, dpi: 96 };
    let hwnd = window(place).expect("a hidden window");
    let hidden = Arc::new(AtomicBool::new(true));
    let surface = Parts::make(&device, hwnd, place.size, Arc::clone(&hidden)).expect("a surface");
    let composition = composition(hwnd, &surface.chain).expect("a composition");
    // SAFETY: plain queries.
    let before = unsafe { GetForegroundWindow() };
    show(hwnd, place).expect("shown off screen");
    pump();
    // SAFETY: plain queries.
    unsafe {
        assert!(IsWindowVisible(hwnd).as_bool());
        let after = GetForegroundWindow();
        assert!(after != hwnd && after != title, "neither window took the focus");
        assert!(before == after || before != hwnd, "the focus did not move to the cover");
        let under = WindowFromPoint(POINT { x: place.at.0 + 10, y: place.at.1 + 10 });
        assert_eq!(under, title, "the point under the cover hits the other thread's window");
    }
    // A panic after the show: the hold hides and destroys the window.
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _held = Held { hwnd, composition: Some(composition), shared: None };
        panic!("injected after the show");
    }));
    assert!(caught.is_err());
    // SAFETY: a plain query.
    assert!(!unsafe { IsWindow(Some(hwnd)) }.as_bool(), "destroyed by the hold");
    drop(surface);
    quit.store(true, Ordering::Release);
    thread.join().expect("the stand-in's thread");
}

/// The card's side on the software adapter, into a hidden window that is
/// never shown: frames run, a stop before a send sends nothing, and all is
/// let go once the window is said hidden and the fence has passed.
#[test]
fn the_surface_presents_on_warp_into_a_hidden_window_and_lets_go() {
    let _turn = crate::indirect::tests::warp_turn();
    let Some(device) = crate::inplace::tests::warp() else {
        eprintln!("SKIPPED: no software adapter");
        return;
    };
    let place = Place { at: (-32000, -32000), size: (64, 36), monitor: 0, dpi: 96 };
    let hwnd = window(place).expect("a hidden window");
    // SAFETY: a plain query.
    assert!(!unsafe { IsWindowVisible(hwnd) }.as_bool(), "the window is never shown in this test");
    let hidden = Arc::new(AtomicBool::new(false));
    let mut surface = Parts::make(&device, hwnd, place.size, Arc::clone(&hidden)).expect("a surface");
    let composition = composition(hwnd, &surface.chain).expect("a composition");
    let image = qoi::Image { width: 16, height: 9, channels: 3, pixels: [200u8, 100, 50].repeat(16 * 9) };
    let (mut rows, mut counts, go) = (Rows::default(), Counts::default(), AtomicBool::new(false));
    let mut presented = 0;
    let began = Instant::now();
    while presented < 4 && began.elapsed() < Duration::from_secs(10) {
        match present(&mut surface, &image, &mut rows, &mut counts, &go).expect("no failure") {
            Tick::Presented => presented += 1,
            _ => std::thread::sleep(Duration::from_millis(2)),
        }
    }
    assert_eq!(presented, 4, "{counts:?}");
    let sent = surface.value;
    let stop = AtomicBool::new(true);
    let mut tick = present(&mut surface, &image, &mut rows, &mut counts, &stop).expect("no failure");
    while tick == Tick::Busy && began.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(2));
        tick = present(&mut surface, &image, &mut rows, &mut counts, &stop).expect("no failure");
    }
    assert_eq!(tick, Tick::Stopped);
    assert_eq!(surface.value, sent, "nothing sent once asked to stop");
    assert!(surface.settled(Duration::from_secs(5)), "the fence reached the last send");
    assert!(surface.allocators.iter().all(|a| matches!(a.sent, Sent::At(_) | Sent::No)));
    assert!(surface.uploads.iter().all(|u| matches!(u.sent, Sent::At(_) | Sent::No)));
    let (cpu, card) = surface.bytes();
    assert_eq!(cpu, 256 * 36 * UPLOADS as u64);
    assert_eq!(card, 64 * 36 * 4 * 2);
    hidden.store(true, Ordering::Release);
    drop(surface);
    drop(composition);
    // SAFETY: our own window.
    unsafe { DestroyWindow(hwnd) }.expect("destroyed");
}

/// The presenter's tick at 3840 by 2160 on the software adapter (its card
/// time is not the card's: the scaling and the calls are what this measures).
#[test]
#[ignore = "a measurement: cargo test ... cover::tests::cost -- --ignored --nocapture"]
fn cost() {
    let _turn = crate::indirect::tests::warp_turn();
    let Some(device) = crate::inplace::tests::warp() else { return };
    let place = Place { at: (-32000, -32000), size: (3840, 2160), monitor: 0, dpi: 96 };
    let hwnd = window(place).expect("a hidden window");
    let hidden = Arc::new(AtomicBool::new(true));
    let mut surface = Parts::make(&device, hwnd, place.size, hidden).expect("a surface");
    let composition = composition(hwnd, &surface.chain).expect("a composition");
    let image = qoi::Image { width: 1920, height: 1080, channels: 3, pixels: (0..1920 * 1080 * 3).map(|i| (i % 251) as u8).collect() };
    let (mut rows, mut counts, go) = (Rows::default(), Counts::default(), AtomicBool::new(false));
    let mut ticks = Vec::new();
    let began = Instant::now();
    while counts.presented < 48 && began.elapsed() < Duration::from_secs(60) {
        let t = Instant::now();
        if present(&mut surface, &image, &mut rows, &mut counts, &go).expect("no failure") == Tick::Presented {
            ticks.push(t.elapsed().as_secs_f64() * 1e3);
        } else {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    ticks.sort_by(f64::total_cmp);
    println!(
        "3840 x 2160: {} presents, a tick median {:.2} ms, slowest {:.2} ms; scaling slowest {:.2} ms, the present call slowest {:.2} ms; busy {}; held {:?}",
        ticks.len(),
        ticks[ticks.len() / 2],
        ticks[ticks.len() - 1],
        counts.slowest_scale_ms,
        counts.slowest_present_ms,
        counts.busy,
        surface.bytes()
    );
    drop(surface);
    drop(composition);
    // SAFETY: our own window.
    unsafe { DestroyWindow(hwnd) }.expect("destroyed");
}

#[test]
fn f1_then_esc_in_one_batch_is_latched() {
    // The menu's key thread saw F1 and then Esc between two title presents:
    // the menu drains both at once and stays closed, but the press was latched
    // where the keys are read.
    let shared = Shared::default();
    shared.ready.store(true, Ordering::Relaxed);
    let pinned = Title { hwnd: 1, thread: 1, made: 2 };
    let place = Place { at: (0, 0), size: (1, 1), monitor: 0, dpi: 96 };
    let (mut d, mut after) = (Detector::new(), u64::MAX);
    let at_begin = f1_presses();
    crate::F1_PRESSES.fetch_add(1, Ordering::AcqRel);
    assert!(!crate::SETTINGS_MENU_OPEN.load(Ordering::Relaxed), "F1 then Esc: the menu is closed again");
    let why = look(&mut d, &mut after, 1000.5, 1000.0, &shared, at_begin, pinned, place);
    assert_eq!(why.as_deref(), Some("F1 (the settings menu)"));
}

#[test]
fn switched_on_the_hook_on_the_title_present_only_stores() {
    let _turn = BOOT_TURN.lock().unwrap_or_else(|e| e.into_inner());
    // A keeper parked since the boot video began (here a stand-in flag): the
    // hook at its end starts nothing and makes nothing; it stores where the
    // video is (two atomics, `boot_ended`), and the keeper goes on from there.
    let started = STARTED.load(Ordering::Relaxed);
    PARKED.store(true, Ordering::Release);
    boot_ended(true);
    assert_eq!(BOOT.load(Ordering::Acquire), PLAYED);
    assert_eq!(STARTED.load(Ordering::Relaxed), started, "no thread started by the hook");
    PARKED.store(false, Ordering::Release);
    BOOT.store(PENDING, Ordering::Release);
}
