//! `VICTORY_LAND_SCAN=1`: the same landing gives the same title memory with
//! the PC-side scan run under the card fetch as with it run after the
//! fetch and the card merge, as the landing did before the switch.

use std::sync::Arc;

use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};

use crate::host::Declared;
use crate::host::land::RelayBack;
use crate::shadow::{self, Shadow};

/// A region of title memory with a registered buffer, so that
/// `shadow::find` makes a copy of exactly its length: the title's bytes,
/// which the title may write at any moment (a test writes them directly,
/// as the title would).
struct Title {
    at: *mut u8,
    len: usize,
    resource: usize,
}

impl Title {
    fn new(resource: usize, bytes: &[u8]) -> Option<Self> {
        let len = bytes.len();
        // SAFETY: our own process's memory, released in Drop.
        let at = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) }.cast::<u8>();
        if at.is_null() {
            return None;
        }
        // SAFETY: `len` bytes are committed.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), at, len) };
        shadow::register(resource, at as u64, len as u64);
        Some(Self { at, len, resource })
    }

    /// The copy of this title memory (the title buffer's, as `find` makes
    /// it).
    fn copy(&self) -> Option<Arc<Shadow>> {
        shadow::find(self.at as u64, self.len as u64)
    }

    fn address(&self) -> usize {
        self.at as usize
    }

    fn end(&self) -> usize {
        self.at as usize + self.len
    }

    /// The title wrote `bytes` here (after the copy was made).
    fn poke(&self, at: usize, bytes: &[u8]) {
        // SAFETY: inside the region, which the test owns.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.at.add(at), bytes.len()) };
    }

    fn bytes(&self) -> Vec<u8> {
        // SAFETY: the region is committed until Drop.
        unsafe { std::slice::from_raw_parts(self.at, self.len) }.to_vec()
    }
}

impl Drop for Title {
    fn drop(&mut self) {
        shadow::unregister(self.resource, self.at as u64);
        // SAFETY: the region made in `new`.
        unsafe { VirtualFree(self.at.cast(), 0, MEM_RELEASE) };
    }
}

/// Lands `back` as `Host::relay_land` does, with the production calls in
/// the order the switch asks: with `beside` the copies in the PC's memory
/// are looked through where the card fetch is in flight
/// (`Host::relay_card_back`), and the merge comes after both; without it
/// the scan comes just before the merge, as the landing did before the
/// switch. Then each flight's ranges are done with, as the landing does.
fn land(back: &[Declared], beside: bool) {
    land_exact(back, beside, false);
}

/// [`land`] with the merge mode the landing started under, as `Host::relay_land` passes it.
fn land_exact(back: &[Declared], beside: bool, exact: bool) {
    // Both modes of the merge are the same bytes per piece, so one mode serves the
    // check either way.
    let mut hand = RelayBack::new(back, exact);
    if beside {
        // Here `Host::relay_card_back` has sent the fetch and waits for the
        // card; nothing of it touches the copies in the PC's memory.
        hand.look();
    }
    hand.merge();
    for (shadow, declared) in back {
        shadow.copy_back_done(declared);
    }
}

/// The same landing with the scan run under the fetch and with it run after
/// it gives identical title memory, and so does the exact merge: the card's
/// bytes where they differ from the snapshot, and the title's bytes the
/// title wrote during the flight where the card did not write.
#[test]
fn the_same_landing_scans_under_the_fetch_and_after_it_alike() {
    // The copies are made on a host device (WARP in the gates); without one
    // the fixture fails: a skip would let a broken landing pass unseen.
    crate::host::test_host().expect("a host device for the landing (the software adapter is enough)");
    const LEN: usize = 192 << 10;
    // The title's bytes before the flights.
    let base: Vec<u8> = (0..LEN).map(|i| (i * 37 + i / 251) as u8).collect();
    // What the card wrote: the first and last bytes, both sides of a 64 KiB
    // piece's cut, a 32-byte line's, and an unaligned one.
    let mut card = base.clone();
    for at in [0usize, 1, 31, 32, 33, 63, 64, 65, 4095, 4096, 4097, 65_535, 65_536, LEN - 1] {
        card[at] = card[at].wrapping_add(0x5a);
    }
    let one = Title::new(0x7a11, &base).expect("title memory for the landing off the fetch");
    let two = Title::new(0x7a12, &base).expect("title memory for the landing under the fetch");
    let three = Title::new(0x7a13, &base).expect("title memory for the exact merge");
    let first = one.copy().expect("a copy of the first title buffer");
    let second = two.copy().expect("a copy of the second title buffer");
    let third = three.copy().expect("a copy of the third title buffer");
    // The refresh before the submission gives the shaders the title's bytes
    // (in the title it is a list's copy, made as the list is brought up to
    // date).
    first.test_card_write(0, &base);
    second.test_card_write(0, &base);
    third.test_card_write(0, &base);
    // Each submission declares its range and its snapshot is taken before
    // its lists go to the card (`Shadow::writing`, `take_snapshots`).
    first.writing(one.address(), LEN);
    second.writing(two.address(), LEN);
    third.writing(three.address(), LEN);
    shadow::take_snapshots();
    // The title wrote during the flights, where the card did not.
    for (title, at, byte) in [
        (&one, 100usize, 0x99u8),
        (&two, 100, 0x99),
        (&three, 100, 0x99),
        (&one, 1000, 0x98),
        (&two, 1000, 0x98),
        (&three, 1000, 0x98),
    ] {
        title.poke(at, &[byte]);
    }
    // The card wrote both copies (shaders write the copies' memory).
    first.test_card_write(0, &card);
    second.test_card_write(0, &card);
    third.test_card_write(0, &card);
    let off = vec![(first.clone(), vec![(one.address(), one.end())])];
    let on = vec![(second.clone(), vec![(two.address(), two.end())])];
    // Without the switch the scan comes after the card fetch and its merge;
    // with it, under the fetch.
    land(&off, false);
    land(&on, true);
    // The third copy lands with the exact merge on (the landing keeps the mode it starts
    // under: `Host::relay_land` passes it once, `RelayBack` carries it to every piece).
    land_exact(&[(third.clone(), vec![(three.address(), three.end())])], true, true);
    let off_bytes = one.bytes();
    let on_bytes = two.bytes();
    let exact_bytes = three.bytes();
    assert_eq!(off_bytes, on_bytes, "the same title memory whether the scan ran under the fetch or after it");
    assert_eq!(off_bytes, exact_bytes, "the same title memory with either merge mode");
    for at in [0usize, 1, 31, 32, 33, 63, 64, 65, 4095, 4096, 4097, 65_535, 65_536, LEN - 1] {
        assert_eq!(off_bytes[at], card[at], "the card's byte at {at}");
    }
    assert_eq!(off_bytes[100], 0x99, "the title's byte where the card did not write");
    assert_eq!(off_bytes[1000], 0x98, "the title's byte where the card did not write");
}

/// A region of title memory that is a mapped view, so that the guards can
/// hold a block of it read-only, with a registered buffer: a fixture arms a
/// guard over it in the gap before a merge's kernel store.
struct ViewTitle {
    at: *mut u8,
    len: usize,
    resource: usize,
    section: *mut std::ffi::c_void,
}

impl ViewTitle {
    fn new(resource: usize, bytes: &[u8]) -> Option<Self> {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE};
        let len = bytes.len();
        // SAFETY: a fresh pagefile section of `len` bytes, mapped once; the
        // view keeps the section alive, and Drop closes both.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            );
            if section.is_null() {
                return None;
            }
            let at = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as *mut u8;
            if at.is_null() {
                windows_sys::Win32::Foundation::CloseHandle(section);
                return None;
            }
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), at, len);
            shadow::register(resource, at as u64, len as u64);
            Some(Self { at, len, resource, section })
        }
    }

    fn copy(&self) -> Option<Arc<Shadow>> {
        shadow::find(self.at as u64, self.len as u64)
    }

    fn address(&self) -> usize {
        self.at as usize
    }

    fn bytes(&self) -> Vec<u8> {
        // SAFETY: the view is mapped until Drop.
        unsafe { std::slice::from_raw_parts(self.at, self.len) }.to_vec()
    }
}

impl Drop for ViewTitle {
    fn drop(&mut self) {
        shadow::unregister(self.resource, self.at as u64);
        // The guards are told before the view goes, as the runtime tells
        // them.
        crate::watch::title_memory_remapped(self.at as usize, self.len);
        // SAFETY: the view and section made in `new`.
        unsafe {
            windows_sys::Win32::System::Memory::UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: self.at as *mut _ },
            );
            windows_sys::Win32::Foundation::CloseHandle(self.section);
        }
    }
}

/// The module's rule (`watch.rs`, the kernel-writing docs): a user-mode write
/// may open its blocks with `writing`, because a block guarded again meanwhile
/// faults into the handler; a kernel write fails on such a block, so it must
/// hold `kernel_writing`, which `protect` respects. A landing's merge opens
/// the block it stores into and a guard is armed again in the gap before its
/// kernel store (a refresh on another thread's `guard()`); the store must
/// still land and title memory must hold the card's bytes.
///
/// On world the merge opened with `writing`, the store was refused, retried
/// by 64 KiB and 4 KiB pieces, and dropped (commit 1's counter non-zero);
/// the snapshot already held the card's bytes, so no later landing saw a
/// difference and the loss stayed.
#[test]
fn a_guard_armed_before_a_merges_store_does_not_drop_the_landed_bytes() {
    // The landing's copy needs a host device (WARP in the gates); without
    // one the fixture fails: a skip would hide a dropped landing.
    crate::host::test_host().expect("a host device for the landing (the software adapter is enough)");
    const LEN: usize = 64 << 10;
    // The title's bytes before the flight; the card writes the first and
    // last bytes and three in between.
    let base: Vec<u8> = (0..LEN).map(|i| (i * 41 + i / 97) as u8).collect();
    let changed = [0usize, 100, 4096, 40_000, LEN - 1];
    let mut card = base.clone();
    for &at in &changed {
        card[at] ^= 0x5a;
    }
    let title = ViewTitle::new(0x7a21, &base).expect("title memory as a view");
    let copy = title.copy().expect("a copy of the title buffer");
    // The copy holds the title's bytes before the flight, and the snapshot
    // is taken; then the card writes.
    copy.test_card_write(0, &base);
    copy.writing(title.address(), LEN);
    shadow::take_snapshots();
    copy.test_card_write(0, &card);
    // The block is armed again in the gap between the merge's opening and
    // its kernel store.
    shadow::STORE_GAP_AT.store(title.address(), std::sync::atomic::Ordering::Relaxed);
    let before = crate::storestat::counted();
    land_exact(&[(copy.clone(), vec![(title.address(), title.address() + LEN)])], false, false);
    shadow::STORE_GAP_AT.store(0, std::sync::atomic::Ordering::Relaxed);
    assert_eq!(crate::storestat::counted().0 - before.0, 0, "no piece of the store failed");
    let bytes = title.bytes();
    for &at in &changed {
        assert_eq!(bytes[at], card[at], "the card's byte at {at} landed");
    }
    assert_eq!(bytes, card, "title memory holds what landed, and the title's other bytes");
}

/// What a landing's holds cost, and a landing racing a pending overlapped
/// read. Eight 64 KiB pieces land while a read into the same title memory is
/// pending: the hold of a landing's store does not walk the pending reads (no
/// settle, no status read: a guard check and the texture pass settle them),
/// notes each piece to the texture pass once (at its end, when the bytes are
/// there), and takes the guards' lock once per piece. On `world` at
/// 29cf5bb58 the same landing made 8 settles and 8 status reads, 16 notes
/// and 16 lock takings. Then the read completes: title memory holds the
/// card's bytes and the read's, and the read ends at the next settle.
#[test]
fn a_landing_holds_lightly_and_a_pending_read_still_ends() {
    use crate::watch::{HOLD_LOCKS, HOLDS_HERE, NOTES, SETTLES, STATUS_READS};
    crate::host::test_host().expect("a host device for the landing (the software adapter is enough)");
    const PIECES: usize = 8;
    const LEN: usize = PIECES << 16;
    let base: Vec<u8> = (0..LEN).map(|i| (i * 37 + i / 89) as u8).collect();
    let mut card = base.clone();
    for piece in 0..PIECES {
        card[piece << 16 | 300] ^= 0x3c;
    }
    let title = ViewTitle::new(0x7a22, &base).expect("title memory as a view");
    let copy = title.copy().expect("a copy of the title buffer");
    copy.test_card_write(0, &base);
    copy.writing(title.address(), LEN);
    shadow::take_snapshots();
    copy.test_card_write(0, &card);
    // An overlapped read into the first piece's block, at bytes the card did
    // not write, still pending (its status reads STATUS_PENDING).
    let status = Box::new(0x103usize);
    let read_at = title.address() + 8192;
    crate::watch::kernel_writing(read_at, 4096).pending((&raw const *status).cast());
    HOLDS_HERE.with(|h| h.set([0; 4]));
    land_exact(&[(copy.clone(), vec![(title.address(), title.address() + LEN)])], false, false);
    let n = HOLDS_HERE.with(std::cell::Cell::get);
    assert_eq!((n[SETTLES], n[STATUS_READS]), (0, 0), "a landing's hold walked the pending reads: {n:?}");
    assert!(n[NOTES] <= PIECES as u64, "{} notes to the texture pass for {PIECES} pieces", n[NOTES]);
    assert!(n[HOLD_LOCKS] <= PIECES as u64 + 1, "{} takings of the guards' lock for {PIECES} pieces", n[HOLD_LOCKS]);
    // The read completes: the kernel wrote its bytes, the status is final.
    let read: Vec<u8> = (0..4096).map(|i| (i as u8) ^ 0xa7).collect();
    // SAFETY: inside the view, which the test owns; the blocks are open (the
    // read's hold).
    unsafe { std::ptr::copy_nonoverlapping(read.as_ptr(), read_at as *mut u8, read.len()) };
    // SAFETY: the status the read's hold looks at; written as the I/O would.
    unsafe { std::ptr::write_volatile((&raw const *status).cast_mut(), 0) };
    HOLDS_HERE.with(|h| h.set([0; 4]));
    crate::watch::kernel_writes_settled();
    let n = HOLDS_HERE.with(std::cell::Cell::get);
    assert!(n[STATUS_READS] >= 1, "the read was not looked at by the settle");
    let mut expected = card.clone();
    expected[8192..8192 + 4096].copy_from_slice(&read);
    assert_eq!(title.bytes(), expected, "title memory holds the card's bytes and the read's");
    drop(status);
}

/// The default store path (`copy_back`, the switch-off merge; the exact
/// switch does not reach it) stores only the runs the card changed. A title
/// byte written in a gap between two changed runs on one page, after the
/// merge read title memory and before its store, stays: a store of the
/// page's first-to-last changed byte puts it back to the value read.
/// Commit 986e590e1 batched the page here and reverted that byte; a fixture
/// for any later batching.
#[test]
fn copy_back_leaves_a_title_byte_written_in_a_gap_of_one_page() {
    const LEN: usize = 64 << 10;
    let base: Vec<u8> = (0..LEN).map(|i| (i * 17 + i / 233) as u8).collect();
    let mut card = base.clone();
    // Two changed bytes on one 4 KiB page, with a gap between them.
    card[100] ^= 0x5a;
    card[200] ^= 0x5a;
    let title = Title::new(0x7a23, &base).expect("title memory for the copy-back");
    let copy = title.copy().expect("a copy of the title buffer");
    // The copy holds the title's bytes before the flight, and the snapshot
    // is taken; then the card writes.
    copy.test_card_write(0, &base);
    copy.writing(title.address(), LEN);
    shadow::take_snapshots();
    copy.test_card_write(0, &card);
    // The title writes in the gap while the store is about to run, after
    // the merge's read of title memory (`shadow::store_gap`).
    shadow::STORE_POKE_AT.store(title.address() + 150, std::sync::atomic::Ordering::Relaxed);
    shadow::STORE_POKE_BYTE.store(0x99, std::sync::atomic::Ordering::Relaxed);
    let written = copy.copy_back(title.address(), LEN);
    shadow::STORE_POKE_AT.store(0, std::sync::atomic::Ordering::Relaxed);
    copy.copy_back_done(&[(title.address(), title.address() + LEN)]);
    let bytes = title.bytes();
    assert_eq!(bytes[100], card[100], "the card's byte at 100 landed");
    assert_eq!(bytes[200], card[200], "the card's byte at 200 landed");
    assert_eq!(bytes[150], 0x99, "the title's gap byte stayed: the store is the changed runs, not the page");
    assert_eq!(written, 2, "the merge stored the two changed bytes, not the page: {written}");
}
