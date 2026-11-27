//! `VICTORY_FETCH_QUEUE=1`: the copy queue's fetch reads what the flight wrote and nothing
//! later; its waits end on a removed device, at a bound, or by a release, and never by a false
//! completion; with the switch off nothing of it is made or waited for. On the software
//! adapter, with a drawing queue of the test's own (the host's queue is left to the other
//! tests).

use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;

use super::{BEFORE_WAIT, Book, FAULT, Failed, Fault, Order, Part, Relay, Shared, TEST_BOUND, Ticket, batch, ticket_for, why};
use crate::host::{CardPlan, DEVICE_GONE, Host, test_host, wait_fence};

const LEN: u64 = 1 << 16;
const FLIGHT: u8 = 0xA5;
const LATER: u8 = 0x5A;

/// A buffer on `device`: in PC memory the CPU maps (`pc`), else in the card's memory, shaders
/// may write it; created in the common state, as the host's are.
fn buffer(device: &ID3D12Device, size: u64, pc: bool) -> (ID3D12Resource, *mut u8) {
    let heap = if pc {
        D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        }
    } else {
        D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        }
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut made: Option<ID3D12Resource> = None;
    let mut data = std::ptr::null_mut();
    // SAFETY: plain creation, and mapping of a CPU-visible buffer.
    unsafe {
        device
            .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut made)
            .expect("a buffer");
        let made = made.expect("a buffer");
        if pc {
            made.Map(0, None, Some(&mut data)).expect("a mapping");
        }
        (made, data.cast())
    }
}

/// A closed direct list and its allocator on `device`.
fn prelude(device: &ID3D12Device) -> (ID3D12CommandAllocator, ID3D12GraphicsCommandList) {
    // SAFETY: plain object creation.
    unsafe {
        let allocator: ID3D12CommandAllocator =
            device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("an allocator");
        let list: ID3D12GraphicsCommandList =
            device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).expect("a list");
        list.Close().expect("closed");
        (allocator, list)
    }
}

/// Until `fence` reaches `value`, for at most ten seconds: whether it did.
fn reaches(fence: &ID3D12Fence, value: u64) -> bool {
    let deadline = Instant::now() + Duration::from_secs(10);
    // SAFETY: a plain query.
    while unsafe { fence.GetCompletedValue() } < value {
        if Instant::now() > deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    true
}

/// A queue has run everything sent to it: a fresh fence signalled behind it, waited for.
fn idle(device: &ID3D12Device, queue: &ID3D12CommandQueue) -> bool {
    // SAFETY: plain object creation; our queue.
    unsafe {
        let Ok(fence) = device.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) else {
            return false;
        };
        queue.Signal(&fence, 1).is_ok() && reaches(&fence, 1)
    }
}

/// A device of the fixture's own on the software adapter, a drawing queue and its fence,
/// source buffers in the card's memory as card copies' are, two buffers of PC memory holding
/// what a flight and a later submission write into them, and the relay's list. These
/// fixtures stall queues on purpose, so never on the host's device (other tests share it)
/// and never on a card: the device is refused unless it is the software adapter. Dropped in
/// order: every wait let go, both queues seen idle on fences, then the lists and buffers, the
/// queues, the device.
struct Fixture {
    relay: std::cell::RefCell<Option<Relay>>,
    lists: Vec<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    sources: Vec<ID3D12Resource>,
    flight: ID3D12Resource,
    later: ID3D12Resource,
    shared: Shared,
    drawing: ID3D12CommandQueue,
    fence: ID3D12Fence,
    device: ID3D12Device,
}

impl Fixture {
    fn new(sources: usize) -> Self {
        let made = victory_shader::adapter::make_device(Some("the fetch fixtures' own software device"))
            .unwrap_or_else(|e| panic!("a software device for the fetch fixtures: {e}"));
        assert!(
            made.adapter.as_ref().is_some_and(|a| a.software),
            "the fetch fixtures stall queues: refused on a device that is not the software adapter"
        );
        let device = made.device;
        let shared = Shared::new(&device).expect("a copy queue");
        // SAFETY: plain object creation.
        let (drawing, fence) = unsafe {
            let drawing: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .expect("a drawing queue");
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("a fence");
            (drawing, fence)
        };
        let filled = |byte: u8| {
            let (buffer, data) = buffer(&device, LEN, true);
            // SAFETY: the buffer's mapping, `LEN` bytes.
            unsafe { std::ptr::write_bytes(data, byte, LEN as usize) };
            buffer
        };
        Self {
            relay: std::cell::RefCell::new(None),
            lists: Vec::new(),
            sources: (0..sources).map(|_| buffer(&device, LEN, false).0).collect(),
            flight: filled(FLIGHT),
            later: filled(LATER),
            shared,
            drawing,
            fence,
            device,
        }
    }

    /// The drawing queue writes `from` over the whole of source `i`, then signals `value`.
    fn write(&mut self, i: usize, later: bool, value: u64) {
        let (allocator, list) = prelude(&self.device);
        let from = if later { &self.later } else { &self.flight };
        // SAFETY: a fresh list; live buffers; our queue and fence.
        unsafe {
            allocator.Reset().unwrap();
            list.Reset(&allocator, None).unwrap();
            list.CopyBufferRegion(&self.sources[i], 0, from, 0, LEN);
            list.Close().unwrap();
            self.drawing.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            self.drawing.Signal(&self.fence, value).unwrap();
        }
        self.lists.push((allocator, list));
    }

    fn reached(&self) -> u64 {
        // SAFETY: a plain query.
        unsafe { self.fence.GetCompletedValue() }
    }

    /// Until the drawing queue's fence reaches `value`, for at most ten seconds (a test that
    /// would wait for ever fails instead).
    fn until(&self, value: u64) -> bool {
        reaches(&self.fence, value)
    }

    /// The drawing queue has not reached `value` after time enough to run it: a check that
    /// something does not happen, the one wait here that is a time.
    fn still_held(&self, value: u64) -> bool {
        std::thread::sleep(Duration::from_millis(200));
        self.reached() < value
    }

    /// The fixture's relay list, kept by the fixture so that it outlives every list it sent.
    fn relay(&self) -> std::cell::RefMut<'_, Relay> {
        std::cell::RefMut::map(self.relay.borrow_mut(), |relay| {
            relay.get_or_insert_with(|| {
                let mut relay = Relay::new(&self.device).expect("a copy list");
                let size = LEN * self.sources.len() as u64;
                assert!(relay.room(size, |size| Some(buffer(&self.device, size, true))), "memory to fetch into");
                relay
            })
        })
    }

    /// Every source, whole, one after the other in the fetch's memory.
    fn parts(&self) -> Vec<Part<'_>> {
        (0..self.sources.len())
            .map(|i| Part {
                source: &self.sources[i],
                lo: 0,
                len: LEN,
                at: i as u64 * LEN,
            })
            .collect()
    }

    fn order(&self, ran: u64, ticket: u64) -> Order<'_> {
        Order {
            fence: &self.fence,
            ran,
            ticket,
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        FAULT.set(Fault::None);
        TEST_BOUND.set(None);
        // What a test left waiting is let go: the copy queue's wait for the drawing fence, the
        // drawing queue's holds for the copy fence.
        // SAFETY: our fences, from the CPU.
        unsafe {
            let _ = self.fence.Signal(u64::MAX);
            let _ = self.shared.fence.Signal(u64::MAX);
        }
        // Both queues seen idle before anything they may use goes.
        if idle(&self.device, &self.drawing) && idle(&self.device, &self.shared.queue) {
            return;
        }
        // A queue still has work: nothing it may use is freed (kept for the process's life,
        // rather than run freed, which removes the device).
        std::mem::forget(self.relay.take());
        std::mem::forget(std::mem::take(&mut self.lists));
        std::mem::forget(self.sources.clone());
        std::mem::forget((self.flight.clone(), self.later.clone(), self.device.clone()));
        std::mem::forget((self.drawing.clone(), self.shared.queue.clone()));
        if !std::thread::panicking() {
            panic!("a fetch fixture's queues did not go idle: its objects are kept");
        }
    }
}

/// The copies the test calls copy 7 and copy 8.
const COPY: usize = 7;
const OTHER: usize = 8;

/// The fetched bytes of source `i`, as `fetched` left them.
fn fetched_bytes(data: usize, i: usize) -> &'static [u8] {
    // SAFETY: the fetch's memory, written by copies that have run; kept by the fixture's relay
    // for the test's length.
    unsafe { std::slice::from_raw_parts((data + i * LEN as usize) as *const u8, LEN as usize) }
}

/// The ordering rule: a later submission writes the source right after the flight; the
/// fetch, sent after the flight has run, reads the flight's bytes, not the later ones, because
/// the later writer was held behind the fetch's ticket. The hold ends with the fetch.
#[test]
fn a_fetch_never_reads_a_later_write() {
    let mut f = Fixture::new(1);
    let mut book = Book::new();
    // The flight: nothing out yet, so nothing held; it takes a ticket for the copy it declares.
    assert_eq!(book.hold(&f.shared, &f.drawing, [COPY].into_iter(), &[], false).unwrap(), None);
    let ticket = book.issue(&f.shared, [COPY].into_iter());
    f.write(0, false, 1);
    // The next submission writes the same copy, right behind the flight.
    let held = book.hold(&f.shared, &f.drawing, [COPY].into_iter(), &[], false).unwrap();
    f.write(0, true, 2);
    // The relay: the flight has run.
    assert!(wait_fence(&f.device, &f.fence, None, 1));
    let behind = f.still_held(2);
    let mut relay = f.relay();
    let fetched = relay.fetch(&f.device, &f.shared, f.order(1, ticket), &f.parts(), || {}).expect("the fetch ran");
    let bytes = fetched_bytes(fetched.data, 0);
    let later_bytes = bytes.iter().filter(|&&b| b == LATER).count();
    assert!(
        bytes.iter().all(|&b| b == FLIGHT),
        "the fetch read a later write: {later_bytes} of {LEN} bytes are the later submission's"
    );
    assert_eq!(held, Some(ticket), "the later writer was held behind the ticket");
    assert!(behind, "the later writer ran before the fetch");
    // The fetch's signal ended the hold.
    assert!(f.until(2), "the held submission never ran after the fetch");
}

/// A landing of several plans: two flights write two copies, a writer of each is held behind
/// its plan's ticket; one fetch of both, signalling the batch's highest ticket, reads both
/// flights' bytes and lets both writers go. `batch` takes the highest ticket and order and
/// counts every ticketed plan, one that declared nothing too.
#[test]
fn one_fetch_lands_a_batch_of_plans() {
    let mut f = Fixture::new(2);
    let mut book = Book::new();
    let first = book.issue(&f.shared, [COPY].into_iter());
    f.write(0, false, 1);
    let second = book.issue(&f.shared, [OTHER].into_iter());
    f.write(1, false, 2);
    assert_eq!(book.hold(&f.shared, &f.drawing, [COPY].into_iter(), &[], false).unwrap(), Some(first));
    f.write(0, true, 3);
    assert_eq!(book.hold(&f.shared, &f.drawing, [OTHER].into_iter(), &[], false).unwrap(), Some(second));
    f.write(1, true, 4);
    assert!(wait_fence(&f.device, &f.fence, None, 2));
    assert!(f.still_held(3), "a writer ran before the batch's fetch");
    let plan = |ticket: Option<Ticket>| CardPlan {
        ranges: Vec::new(),
        check: None,
        _synced: Vec::new(),
        lists: Vec::new(),
        #[cfg(feature = "diagnostics")]
        stamps: None,
        fetch: ticket,
        hash_check: false,
    };
    let plans = [
        plan(Some(Ticket { value: first, order: 4 })),
        plan(Some(Ticket { value: 0, order: 5 })),
        plan(Some(Ticket { value: second, order: 6 })),
    ];
    let batch = batch(&plans.iter().collect::<Vec<_>>()).expect("a ticketed batch");
    assert_eq!(format!("{batch:?}"), format!("Batch {{ value: {second}, first: {first}, order: 6, plans: 3 }}"));
    let mut relay = f.relay();
    let fetched = relay.fetch(&f.device, &f.shared, f.order(2, second), &f.parts(), || {}).expect("the fetch ran");
    for i in 0..2 {
        assert!(fetched_bytes(fetched.data, i).iter().all(|&b| b == FLIGHT), "copy {i}: a later write was read");
    }
    assert!(f.until(4), "a held writer never ran after the batch's fetch");
}

/// A landing with nothing to fetch signals its ticket alone, and its writer goes; a signal
/// never takes the fence back (a lower one after it is not sent).
#[test]
fn an_empty_fetch_lets_its_writer_go() {
    let mut f = Fixture::new(1);
    let mut book = Book::new();
    let first = book.issue(&f.shared, [COPY].into_iter());
    let ticket = book.issue(&f.shared, [COPY].into_iter());
    f.write(0, false, 1);
    assert_eq!(book.hold(&f.shared, &f.drawing, [COPY].into_iter(), &[], false).unwrap(), Some(ticket));
    f.write(0, true, 2);
    assert!(f.still_held(2));
    f.shared.signal(ticket).expect("the signal");
    assert!(f.until(2), "the held writer never ran after the empty landing's signal");
    f.shared.signal(first).expect("a lower one is not sent");
    // Whatever the copy queue had is run: then the fence holds the ticket, not the lower one.
    assert!(idle(&f.device, &f.shared.queue), "the copy queue did not go idle");
    assert_eq!(f.shared.reached(), ticket, "the fence went back");
}

/// Before a fetch on the drawing queue every ticket given is released: a writer held for one
/// goes, and the plans up to it count as released (they land on the drawing queue).
#[test]
fn a_drawing_queue_fetch_releases_every_ticket_first() {
    let mut f = Fixture::new(1);
    let mut book = Book::new();
    let ticket = book.issue(&f.shared, [COPY].into_iter());
    f.write(0, false, 1);
    book.hold(&f.shared, &f.drawing, [COPY].into_iter(), &[], false).unwrap();
    f.write(0, true, 2);
    assert!(f.still_held(2));
    f.shared.release_all().expect("the release");
    assert!(f.until(2), "the held writer never ran after the release");
    assert!(f.shared.released.load(std::sync::atomic::Ordering::Acquire) >= ticket);
}

/// A device removed during a fetch: the relay's wait gives `Removed` (the run then fails, as
/// on the drawing queue), and a submission held behind a ticket no landing will signal is let
/// go by the removed device's fences, which read `u64::MAX`.
#[test]
fn a_lost_device_during_a_fetch_ends_every_wait() {
    let mut f = Fixture::new(1);
    let mut book = Book::new();
    let ticket = book.issue(&f.shared, [COPY].into_iter());
    f.write(0, false, 1);
    // A later plan's ticket, for the same copy, which its landing will never signal.
    let never = book.issue(&f.shared, [COPY].into_iter());
    assert_eq!(book.hold(&f.shared, &f.drawing, [COPY].into_iter(), &[], false).unwrap(), Some(never));
    f.write(0, true, 2);
    assert!(wait_fence(&f.device, &f.fence, None, 1));
    let mut relay = f.relay();
    // Between the send and the wait the device goes: once the copies have run, the fence
    // reads `u64::MAX` and the device counts as removed on this thread.
    let fence = f.shared.fence.clone();
    BEFORE_WAIT.with(|hook| {
        *hook.borrow_mut() = Some(Box::new(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            // SAFETY: plain queries and a signal from the CPU of our fence.
            unsafe {
                while fence.GetCompletedValue() < ticket && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(1));
                }
                fence.Signal(u64::MAX).unwrap();
            }
            DEVICE_GONE.with(|g| g.set(true));
        }))
    });
    let result = relay.fetch(&f.device, &f.shared, f.order(1, ticket), &f.parts(), || {});
    DEVICE_GONE.with(|g| g.set(false));
    assert!(matches!(result, Err(Failed::Removed)), "a removed device taken for a fetch that ran: {:?}", result.err());
    assert!(f.until(2), "the held submission still waits on a removed device");
}

/// A copy queue that never reaches the fetch's value on a live device: the relay's wait ends
/// at the bound with `Timeout`, and its writer stays held (no false completion).
#[test]
fn a_fetch_that_never_ends_times_out_and_keeps_its_writer_held() {
    let mut f = Fixture::new(1);
    let mut book = Book::new();
    let ticket = book.issue(&f.shared, [COPY].into_iter());
    f.write(0, false, 1);
    book.hold(&f.shared, &f.drawing, [COPY].into_iter(), &[], false).unwrap();
    f.write(0, true, 2);
    let mut relay = f.relay();
    TEST_BOUND.set(Some(Duration::from_millis(300)));
    // The copy queue waits for a drawing value nothing signals: its copies and signal never run.
    let started = Instant::now();
    let result = relay.fetch(&f.device, &f.shared, f.order(99, ticket), &f.parts(), || {});
    assert!(matches!(result, Err(Failed::Timeout(_))), "{:?}", result.err());
    assert!(started.elapsed() < Duration::from_secs(5), "the bound did not end the wait");
    assert!(f.shared.reached() < ticket, "the ticket was reached without its copies");
    assert!(f.still_held(2), "the writer was let go before the copy ran");
}

/// A faulted signal. After the copies were sent (and its retry): `Unsignalled`, and nothing
/// lets the writer go, since nothing tells when the copies end. Before anything was sent:
/// `NotSent`, and the stop that follows lets every writer go (nothing is in flight).
#[test]
fn a_faulted_signal_never_publishes_a_completion() {
    let mut f = Fixture::new(1);
    let mut book = Book::new();
    let ticket = book.issue(&f.shared, [COPY].into_iter());
    f.write(0, false, 1);
    book.hold(&f.shared, &f.drawing, [COPY].into_iter(), &[], false).unwrap();
    f.write(0, true, 2);
    assert!(wait_fence(&f.device, &f.fence, None, 1));
    let mut relay = f.relay();
    FAULT.set(Fault::Signal);
    let result = relay.fetch(&f.device, &f.shared, f.order(1, ticket), &f.parts(), || {});
    FAULT.set(Fault::None);
    assert!(matches!(result, Err(Failed::Unsignalled(_))), "{:?}", result.err());
    assert!(f.shared.reached() < ticket && f.still_held(2), "a writer was let go without a signal behind the copies");

    let mut g = Fixture::new(1);
    let mut book = Book::new();
    let ticket = book.issue(&g.shared, [COPY].into_iter());
    g.write(0, false, 1);
    book.hold(&g.shared, &g.drawing, [COPY].into_iter(), &[], false).unwrap();
    g.write(0, true, 2);
    assert!(wait_fence(&g.device, &g.fence, None, 1));
    let mut relay = g.relay();
    FAULT.set(Fault::BeforeSend);
    let result = relay.fetch(&g.device, &g.shared, g.order(1, ticket), &g.parts(), || {});
    FAULT.set(Fault::None);
    assert!(matches!(result, Err(Failed::NotSent(_))), "{:?}", result.err());
    assert!(g.still_held(2), "nothing but the stop lets the writer go");
    // The release itself failing too: the stop signals from the CPU.
    FAULT.set(Fault::Release);
    assert!(g.shared.stop(), "the stop could not end the holds");
    FAULT.set(Fault::None);
    assert!(g.shared.stopped());
    assert!(g.until(2), "the writer still waits after the stop");
}

/// A copy numbered in the address directory: the numbering waits for a copy-queue read of it
/// under way; a landing whose ticket is not newer than the numbering (a writer by address sent
/// before it was not held for it) does not read it on the copy queue, a later one does; and
/// it is among the copies a writer by address is held for.
#[test]
fn a_copy_numbered_late_stays_on_the_drawing_queue() {
    let copy = 0x7_0000usize;
    assert!(super::start_reading(&[copy], 1), "not numbered yet");
    let numbering = std::thread::spawn(move || {
        let mut guard = super::GUARD.lock().unwrap_or_else(|e| e.into_inner());
        // As `numbered` does, with five tickets given by then.
        guard.get_or_insert_with(Default::default).numbered.insert(copy, 5);
        while guard.as_ref().is_some_and(|g| g.reading.contains(&copy)) {
            guard = super::READ_DONE.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
        Instant::now()
    });
    std::thread::sleep(Duration::from_millis(100));
    let done = Instant::now();
    super::read_done();
    let numbered_at = numbering.join().expect("the numbering");
    assert!(numbered_at >= done, "the numbering did not wait for the read under way");
    assert!(!super::start_reading(&[copy], 5), "a copy numbered after its ticket was read on the copy queue");
    assert!(super::start_reading(&[copy], 6), "a ticket newer than the numbering is safe");
    super::read_done();
    assert!(super::numbered_copies().contains(&copy));
}

/// A writer through the address directory racing a fetch: the later submission declares
/// nothing of the copy (its list was recorded before the copy was numbered) but writes by
/// address, so it is held for every numbered copy's ticket, and the fetch reads the flight's
/// bytes. Without the rule (`by_address` empty) it is not held: the same fixture then reads
/// the later write.
#[test]
fn a_writer_by_address_is_held_behind_the_fetch() {
    for rule in [true, false] {
        let mut f = Fixture::new(1);
        let mut book = Book::new();
        let ticket = book.issue(&f.shared, [COPY].into_iter());
        f.write(0, false, 1);
        // Declares another copy only; may reach COPY by address.
        let reach: &[usize] = if rule { &[COPY, OTHER] } else { &[] };
        let before = super::STATS[8].load(std::sync::atomic::Ordering::Relaxed);
        let held = book.hold(&f.shared, &f.drawing, [OTHER].into_iter(), reach, false).unwrap();
        f.write(0, true, 2);
        assert!(wait_fence(&f.device, &f.fence, None, 1));
        let behind = f.still_held(2);
        let mut relay = f.relay();
        let fetched = relay.fetch(&f.device, &f.shared, f.order(1, ticket), &f.parts(), || {}).expect("the fetch ran");
        let whole = fetched_bytes(fetched.data, 0).iter().all(|&b| b == FLIGHT);
        if rule {
            assert_eq!(held, Some(ticket), "the writer by address was not held");
            assert!(super::STATS[8].load(std::sync::atomic::Ordering::Relaxed) > before, "the hold was not counted as by address");
            assert!(behind && whole, "a writer by address ran during the fetch");
            assert!(f.until(2), "the writer never ran after the fetch");
        } else {
            assert_eq!(held, None);
            assert!(!whole, "the fixture does not race without the rule (it proves nothing)");
        }
    }
}

/// With the switch off and nothing out, nothing of the copy queue is made or asked for, no
/// hold goes on the drawing queue and no ticket is given: the drawing queue's path is the old
/// one. A plan without a ticket lands on the drawing queue (`batch` is none).
#[test]
fn with_the_switch_off_nothing_is_made_or_waited_for() {
    let f = Fixture::new(1);
    let mut book = Book::new();
    for (relay, flown, none_out) in [(true, true, true), (true, false, true), (true, true, false), (false, true, true)] {
        let ticket = ticket_for(
            &mut book,
            false,
            relay,
            flown,
            none_out,
            5,
            &[COPY],
            &[COPY, 9],
            &[COPY],
            false,
            |_| panic!("asked why with the switch off"),
            |_| panic!("the copy queue was asked for with the switch off"),
            &f.drawing,
        )
        .expect("no error");
        assert_eq!(ticket, None);
    }
    let plan = CardPlan {
        ranges: Vec::new(),
        check: None,
        _synced: Vec::new(),
        lists: Vec::new(),
        #[cfg(feature = "diagnostics")]
        stamps: None,
        fetch: None,
        hash_check: false,
    };
    assert_eq!(batch(&[&plan]), None);
}

/// The mode follows the switch only while no plan is out; with it on a flown plan takes a
/// ticket unless it is refused (then it is counted by reason and takes none).
#[test]
fn the_mode_changes_only_with_nothing_out() {
    let f = Fixture::new(1);
    let mut book = Book::new();
    assert!(!book.mode(true, false), "plans out: it stays off");
    assert!(book.mode(true, true));
    assert!(book.mode(false, false), "plans out: it stays on");
    let given = |book: &mut Book, flown: bool, refused: Option<usize>| {
        ticket_for(book, true, true, flown, false, 3, &[COPY], &[COPY], &[], false, |_| refused, |_| Some(&f.shared), &f.drawing).expect("no error")
    };
    let numbered = super::FALLBACK[why::NUMBERED].load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(given(&mut book, true, Some(why::NUMBERED)), None, "a refused plan takes no ticket");
    assert!(super::FALLBACK[why::NUMBERED].load(std::sync::atomic::Ordering::Relaxed) > numbered);
    assert_eq!(given(&mut book, true, None), Some(Ticket { value: 1, order: 3 }));
    // A synchronous plan takes none (nothing is in the air when it is made).
    assert_eq!(given(&mut book, false, None), None);
    // The ticket's landing would take it back; this test has none.
    super::landing(1);
    assert!(!book.mode(false, true), "nothing out: it follows the switch");
}

/// Sol's second review, 4: a batch whose lowest ticket was released lands on the drawing queue,
/// whatever its highest one is.
#[test]
fn a_batch_with_a_released_ticket_falls_back() {
    let plan = |value: u64| CardPlan {
        ranges: Vec::new(),
        check: None,
        _synced: Vec::new(),
        lists: Vec::new(),
        #[cfg(feature = "diagnostics")]
        stamps: None,
        fetch: Some(Ticket { value, order: 1 }),
        hash_check: false,
    };
    let (old, new, empty) = (plan(3), plan(9), plan(0));
    let mixed = batch(&[&old, &new]).expect("a batch");
    assert!(super::released(&mixed, 3), "a released old ticket kept the batch on the copy queue");
    assert!(!super::released(&mixed, 2));
    assert!(!super::released(&batch(&[&empty]).expect("a batch"), 5), "a batch without tickets to release");
}

/// With the switch off and no ticket out, a submission's lists are not looked at.
#[test]
fn off_nothing_is_scanned() {
    if super::on() {
        return;
    }
    assert!(super::SHARED.get().is_none(), "the parent process never makes the copy queue");
    assert!(!super::engaged());
    let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
    let list = host.make_list(false).expect("a list");
    list.first_use(crate::memory::WRITES_BY_HEAP);
    assert_eq!(super::reach(&[&list]), super::Reach::Declared, "a list was looked at with the switch off");
}

/// Sol's second review, 6: the holds through the production path (`Host::card_before` as a
/// submission calls it, and the relay's `Host::send_drawing_fetch`), in a process of its own
/// with the relay and the switch on.
#[test]
fn the_production_path_holds_and_never_deadlocks() {
    let name = "host::fetch::fetch_tests::production_child";
    let exe = std::env::current_exe().expect("the test binary");
    let out = std::process::Command::new(exe)
        .args(["--exact", name, "--ignored", "--test-threads", "1", "--nocapture"])
        .env("VICTORY_RELAY", "2")
        .env("VICTORY_FETCH_QUEUE", "1")
        .env_remove("VICTORY_LIVE")
        .env_remove("VICTORY_GPU_TIMING")
        .output()
        .expect("the child ran");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success() && text.contains("1 passed"), "the child failed:\n{text}");
}

/// A fence of the test's own, signalled on the host's queue behind what is there: whether the
/// queue gets past it within `wait`.
fn queue_passes(host: &Host, value: u64, fence: &ID3D12Fence, wait: Duration) -> bool {
    // SAFETY: the host's queue and our fence.
    unsafe { host.queue.Signal(fence, value) }.unwrap();
    let deadline = Instant::now() + wait;
    // SAFETY: a plain query.
    while unsafe { fence.GetCompletedValue() } < value {
        if Instant::now() > deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    true
}

/// The child of [`the_production_path_holds_and_never_deadlocks`].
#[test]
#[ignore = "a child process of the_production_path_holds_and_never_deadlocks"]
fn production_child() {
    use std::sync::Arc;
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
    // It holds the host's queue on purpose: on a card that is an engine reset. Refused there
    // unless `VICTORY_TESTS_ALLOW_GPU_STALL=1` says so.
    let software = crate::host::CHOSEN.get().and_then(|(adapter, _)| adapter.as_ref()).is_some_and(|a| a.software);
    if !software && std::env::var_os("VICTORY_TESTS_ALLOW_GPU_STALL").is_none_or(|v| v != "1") {
        eprintln!("production_child: refused on a hardware adapter (it stalls the queue; VICTORY_D3D12_WARP=1 for the software one)");
        return;
    }
    assert!(crate::host::relay() && super::on(), "the child runs with the relay and the switch on");
    // A title buffer of 2 MiB whose copy is in the card's memory.
    let len = 2usize << 20;
    // SAFETY: our own process's memory, kept to the end of the process.
    let at = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as u64;
    assert_ne!(at, 0);
    crate::shadow::register(0x5EED_0001, at, len as u64);
    let copy: Arc<crate::shadow::Shadow> = crate::shadow::find(at, len as u64).expect("a copy");
    assert!(copy.carded(), "the copy is not in the card's memory: the fixture proves nothing");
    let list = || host.make_list(false).expect("a list");
    let mut plans = Vec::new();
    let mut lists = Vec::new();
    // A submission whose shaders write the copy: its plan takes a ticket.
    let plan_of = |plans: &mut Vec<CardPlan>, lists: &mut Vec<crate::host::HostList>| {
        let a = list();
        copy.gpu_writes(&a, at, len as u64);
        let plan = host.card_before(&[&a], true, None).expect("a plan");
        let ticket = plan.fetch.expect("a ticket").value;
        assert!(ticket > 0, "the plan did not qualify");
        plans.push(plan);
        lists.push(a);
        ticket
    };
    // SAFETY: plain object creation.
    let fence: ID3D12Fence = unsafe { host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
    let mut fence_value = 0;

    // 2. A list that writes by address and declares nothing of the copy (recorded before the
    // copy was numbered): no plan, held all the same.
    let first = plan_of(&mut plans, &mut lists);
    let shared = super::SHARED.get().and_then(Option::as_ref).expect("the copy queue");
    super::numbered(&copy);
    let b = list();
    b.first_use(crate::memory::WRITES_BY_ADDRESS);
    assert!(host.card_before(&[&b], true, None).is_none(), "it has no plan");
    fence_value += 1;
    assert!(!queue_passes(host, fence_value, &fence, Duration::from_millis(300)), "a writer by address without a plan was not held");
    shared.signal(first).unwrap();
    assert!(queue_passes(host, fence_value, &fence, Duration::from_secs(10)), "the hold did not end at its ticket");

    // 3. A list that may store through a view picked by a computed heap index: held behind
    // every ticket out.
    let second = plan_of(&mut plans, &mut lists);
    let c = list();
    c.first_use(crate::memory::WRITES_BY_HEAP);
    assert!(host.card_before(&[&c], true, None).is_none());
    fence_value += 1;
    assert!(!queue_passes(host, fence_value, &fence, Duration::from_millis(300)), "a writer by heap index was not held");
    shared.signal(second).unwrap();
    assert!(queue_passes(host, fence_value, &fence, Duration::from_secs(10)));

    // 1. The relay's drawing-queue fetch: between its release and its send, two submissions
    // try to take a fresh ticket and hold a writer behind it. They must come after the fetch
    // and its signal, or the relay waits behind a writer whose ticket only it can land.
    plan_of(&mut plans, &mut lists);
    let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
    let (end_tx, end_rx) = std::sync::mpsc::channel::<()>();
    let racing = Arc::new(std::sync::Mutex::new(None));
    let raced = Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let racing = racing.clone();
        let raced = raced.clone();
        let copy = copy.clone();
        super::AFTER_RELEASE.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move || {
                let thread = std::thread::spawn(move || {
                    let mut kept = Vec::new();
                    for _ in 0..2 {
                        let x = host.make_list(false).expect("a list");
                        copy.gpu_writes(&x, at, len as u64);
                        kept.push((host.card_before(&[&x], true, None), x));
                    }
                    raced.store(true, std::sync::atomic::Ordering::Release);
                    let _ = done_tx.send(());
                    // Kept until the queue has run them.
                    let _ = end_rx.recv();
                    drop(kept);
                });
                // As long as it takes them, up to a while.
                let _ = done_rx.recv_timeout(Duration::from_millis(300));
                *racing.lock().unwrap() = Some(thread);
            }))
        });
    }
    let (allocator, fetch_list) = host.make_prelude().expect("a list");
    // SAFETY: a fresh list of ours.
    unsafe {
        allocator.Reset().unwrap();
        fetch_list.Reset(&allocator, None).unwrap();
        fetch_list.Close().unwrap();
    }
    // SAFETY: plain object creation.
    let relay_fence: ID3D12Fence = unsafe { host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
    host.send_drawing_fetch(&[Some(fetch_list.cast::<ID3D12CommandList>().unwrap())], &relay_fence, 1).expect("sent");
    let deadline = Instant::now() + Duration::from_secs(5);
    // SAFETY: a plain query.
    while unsafe { relay_fence.GetCompletedValue() } < 1 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    // SAFETY: as above.
    let reached = unsafe { relay_fence.GetCompletedValue() } >= 1;
    // Everything held ends, then the racing submissions are joined.
    assert!(shared.stop());
    let thread = racing.lock().unwrap().take().expect("the race ran");
    // The racing submissions have their holds on the queue once they are done.
    let started = Instant::now();
    while !raced.load(std::sync::atomic::Ordering::Acquire) && started.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(10));
    }
    fence_value += 1;
    assert!(queue_passes(host, fence_value, &fence, Duration::from_secs(10)));
    let _ = end_tx.send(());
    thread.join().expect("the racing submissions");
    drop((plans, lists, allocator));
    assert!(reached, "the relay's fetch waited behind a writer held for a ticket taken after its release");
}

/// With `VICTORY_GPU_STAMPS=1` a fetch on the copy queue carries a pair of timestamps of its
/// own, read on the copy queue's clock, and the report shows it under the copy queue; where
/// the adapter has no timestamps on copy queues the fetch has none and the report's field of
/// what is not stamped names the copy queue's fetch lists. Either way nothing is silent.
#[cfg(feature = "diagnostics")]
#[test]
fn a_fetch_on_the_copy_queue_is_stamped_or_said_not_to_be() {
    use crate::host::queue_stamps;
    // The stamps on for this thread: the relay's pair is made with its list.
    let _stamps = crate::stamps::Sink::new();
    let mut f = Fixture::new(1);
    let mut book = Book::new();
    assert_eq!(book.hold(&f.shared, &f.drawing, [COPY].into_iter(), &[], false).unwrap(), None);
    let ticket = book.issue(&f.shared, [COPY].into_iter());
    f.write(0, false, 1);
    assert!(wait_fence(&f.device, &f.fence, None, 1));
    let mut relay = f.relay();
    let fetched = relay.fetch(&f.device, &f.shared, f.order(1, ticket), &f.parts(), || {}).expect("the fetch ran");
    assert!(fetched_bytes(fetched.data, 0).iter().all(|&b| b == FLIGHT), "the stamped fetch read other bytes");
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
    // SAFETY: a plain query into a structure of the size said.
    let asked = unsafe {
        f.device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, std::ptr::from_mut(&mut options).cast(), size_of_val(&options) as u32)
    };
    let supported = asked.is_ok() && options.CopyQueueTimestampQueriesSupported.as_bool();
    eprintln!("copy-queue timestamps on this adapter: {supported}; the fetch's span: {:?} ns", fetched.on_queue);
    let kinds = queue_stamps::not_stamped_with(true, Some(supported));
    if supported {
        let span = fetched.on_queue.expect("a fetch on the copy queue with no pair read, on an adapter that has them");
        let object = queue_stamps::json(&queue_stamps::Copy { lists: 1, stamped: 1, ns: span, unread: 0 }, 1.0, &kinds);
        assert!(object.contains("\"copy_queue\":{\"fetch_lists\":1.00,\"stamped\":1.00,"), "{object}");
        assert!(!object.contains("copy queue: the fetch lists"), "{object}");
    } else {
        assert_eq!(fetched.on_queue, None);
        assert!(kinds.contains(&"copy queue: the fetch lists (the adapter has no timestamps on copy queues)"), "{kinds:?}");
    }
}

/// Opus's review of `job/hash-wide-fix`, 1: a flown plan (compare mode 2) declares two separate
/// ranges inside one 64 KiB piece. The compare of the first flags the piece and stores its new
/// hash, so the compare of the second, unordered with it on the card, may find that hash stored
/// and not flag the piece (on the software adapter it always does). The piece is fetched under
/// both ranges whichever compare flagged it (`host::flagged_pieces`), and a plan sampled for
/// `VICTORY_HASH_CHECK=1` fetches exactly the same. In a process of its own (it owns the
/// compare's results and adds a title buffer).
#[test]
fn two_ranges_in_one_piece_are_fetched_under_both() {
    let name = "host::fetch::fetch_tests::two_ranges_child";
    let exe = std::env::current_exe().expect("the test binary");
    let out = std::process::Command::new(exe)
        .args(["--exact", name, "--ignored", "--test-threads", "1", "--nocapture"])
        .env_remove("VICTORY_LIVE")
        .output()
        .expect("the child ran");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success() && text.contains("1 passed"), "the child failed:\n{text}");
}

/// The child of [`two_ranges_in_one_piece_are_fetched_under_both`].
#[test]
#[ignore = "a child process of two_ranges_in_one_piece_are_fetched_under_both"]
fn two_ranges_child() {
    use std::sync::Arc;
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    let host = test_host().expect("a host device (the tests run with VICTORY_D3D12_WARP=1)");
    assert!(host.card_ready(), "no compare: the fixture proves nothing");
    // A title buffer of 2 MiB (64 KiB aligned, so its piece 0 starts at its offset 0).
    let len = 2usize << 20;
    // SAFETY: our own process's memory, kept to the end of the process.
    let at = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
    assert_ne!(at, 0);
    crate::shadow::register(0x5EED_0002, at as u64, len as u64);
    let copy: Arc<crate::shadow::Shadow> = crate::shadow::find(at as u64, len as u64).expect("a copy");
    // Two dwords of the compare's results, taken as a plan takes them.
    let (differs, slot) = {
        let mut work = crate::host::CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let work = work.as_mut().expect("the card's work");
        let slot = work.taken;
        work.taken += 2;
        (work.differs.1 as *mut u32, slot as usize)
    };
    let range = |from: usize, to: usize, slot: usize| crate::host::CardRange {
        shadow: copy.clone(),
        declared: (at + from, at + to),
        span: copy.card_span(at + from, at + to).expect("inside the copy"),
        at: slot as u32,
        compared: true,
    };
    let (a, b) = ((0x100, 0x200), (0x8000, 0x8100));
    for check in [false, true] {
        for flags in [[1u32, 0], [0, 1], [1, 1], [0, 0]] {
            // SAFETY: inside the results' mapping; these two dwords are ours.
            unsafe {
                *differs.add(slot) = flags[0];
                *differs.add(slot + 1) = flags[1];
            }
            let plan = CardPlan {
                ranges: vec![range(a.0, a.1, slot), range(b.0, b.1, slot + 1)],
                check: None,
                _synced: Vec::new(),
                lists: Vec::new(),
                #[cfg(feature = "diagnostics")]
                stamps: None,
                fetch: None,
                hash_check: check,
            };
            assert!(plan.ranges.iter().all(|r| r.span.first == 0 && r.span.pieces == 1), "both in piece 0");
            let (pieces, total) = super::pieces(&[&plan]);
            let mut got: Vec<(usize, usize)> = pieces.iter().map(|p| (p.1, p.2)).collect();
            got.sort_unstable();
            let want = if flags == [0, 0] { vec![] } else { vec![a, b] };
            assert_eq!(got, want, "flags {flags:?} (sampled for the check: {check})");
            assert_eq!(total, want.iter().map(|w| (w.1 - w.0) as u64).sum::<u64>(), "flags {flags:?} (check: {check})");
        }
    }
}
