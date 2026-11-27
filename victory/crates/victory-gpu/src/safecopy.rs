//! The user-mode copy (stage S0 of `jobs/2026-11-26-fork-parity/a2-a1-copy-landing-design.md`,
//! §3.1): title memory copied by a hand-written leaf, not by a kernel call, and a fault in it
//! does not end the run.
//!
//! The title may unmap, guard or protect its memory at any time, so the runtime reads and
//! writes it through the kernel (`kernelstat`), which reports a page it cannot reach instead
//! of faulting. The leaf ([`victory_user_copy`]) is one `rep movsb`. A fault there reaches our
//! vectored handler (`watch`'s `on_write`), whose first lines ([`redirect`]) look at the
//! faulting *instruction*: if it is the leaf's `rep movsb` and the code an access violation, a
//! guard page or an in-page error, they write the fault into the caller's [`Fault`] and send
//! the thread to the leaf's failing exit, taking no lock, allocating nothing and making no
//! system call. Every other fault goes on as before, the title's own on the same page
//! included (its instruction is the title's). [`copy`] then accepts the fault only if it lay
//! in the range it was asked to read (or store) at or after the bytes done, and ends the run
//! otherwise. [`on`] shows, once a run, that the labels and the handler work in this build.
//!
//! Nothing calls it yet but its fixtures; reads (S1, `VICTORY_USER_COPY`) and stores (S2,
//! `VICTORY_USER_STORE`) reach it through `kernelstat`'s counted wrappers only, which the
//! memory-call scan (`kernelstat`'s tests) enforces.

use std::sync::OnceLock;

// The leaf, Win64: `victory_user_copy(dst: rcx, src: rdx, len: r8, fault: r9) -> eax`.
//
// It changes rax, rcx and r8 (volatile), and rsi and rdi, which its prologue saves in the
// caller's home space (`[rsp+8]`, `[rsp+16]`, the callee's to use) and both exits load back;
// rdx and r9 stay as given. It never touches rbx, rbp, r12-r15, rsp, any xmm or ymm register
// or the direction flag (0 on entry by the ABI, never set here). Its unwind data (`.seh_*`:
// a function entry with two saves, no stack allocation) tells the system where rsi and rdi
// are, so an unwind through a fault in it gives the caller's (fixture S0-4's unwind arm).
//
// The failing exit's convention: the handler changes only `Rip` (to `_failed`) and the
// `Fault` at `r9` (its `code`, `access` and `address`); it leaves every other register as the
// fault found it. rsp is the body's, so the exit finds the saves. At a `rep movsb` fault rcx
// counts the bytes not done, so the exit's `r8 - rcx` is the bytes done. eax: 1, all `len`
// bytes (`done = len`), or 0, the `Fault` filled.
std::arch::global_asm!(
    ".globl victory_user_copy",
    ".globl victory_user_copy_at",
    ".globl victory_user_copy_after",
    ".globl victory_user_copy_failed",
    ".seh_proc victory_user_copy",
    "victory_user_copy:",
    "mov qword ptr [rsp + 8], rsi",
    ".seh_savereg rsi, 8",
    "mov qword ptr [rsp + 16], rdi",
    ".seh_savereg rdi, 16",
    ".seh_endprologue",
    "mov rdi, rcx",
    "mov rsi, rdx",
    "mov rcx, r8",
    "victory_user_copy_at:",
    "rep movsb",
    "victory_user_copy_after:",
    "mov qword ptr [r9], r8",
    "mov eax, 1",
    "mov rsi, qword ptr [rsp + 8]",
    "mov rdi, qword ptr [rsp + 16]",
    "ret",
    // `done` from the registers, never from the fault address: the architecture lets the
    // registers of an interrupted `rep movsb` lag the fault (fast strings: every byte before
    // rcx's count is copied, the fault may lie further on), so a count taken from the
    // address could overstate the bytes copied, and stale bytes would pass for fresh. The
    // registers' count can only understate them.
    "victory_user_copy_failed:",
    "sub r8, rcx",
    "mov qword ptr [r9], r8",
    "xor eax, eax",
    "mov rsi, qword ptr [rsp + 8]",
    "mov rdi, qword ptr [rsp + 16]",
    "ret",
    ".seh_endproc",
);

unsafe extern "C" {
    /// The leaf (above): copies `len` bytes from `src` to `dst`; 1 if all were, 0 if a fault
    /// stopped it (then `fault` says where).
    fn victory_user_copy(dst: *mut u8, src: *const u8, len: usize, fault: *mut Fault) -> u32;
    /// Labels, never called: the leaf's `rep movsb`, the instruction after it, and the exit a
    /// fault in the first is sent to.
    fn victory_user_copy_at();
    fn victory_user_copy_after();
    fn victory_user_copy_failed();
}

/// What the leaf reports: `done` from either exit, the rest from the handler at a fault.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Fault {
    /// Bytes copied: all of them, or, at a fault, bytes certainly copied, the fault (at
    /// `address`) at or after them. Resume from `done`; never treat `[done, address)` as
    /// unreadable (the registers may lag the fault, see the failing exit).
    pub(crate) done: usize,
    /// The exception code.
    pub(crate) code: u32,
    /// `ExceptionInformation[0]`: 0 a read, 1 a write.
    pub(crate) access: u32,
    /// `ExceptionInformation[1]`: the data address the fault was at.
    pub(crate) address: usize,
}

// The leaf and the handler write these offsets.
const _: () = assert!(
    std::mem::offset_of!(Fault, done) == 0
        && std::mem::offset_of!(Fault, code) == 8
        && std::mem::offset_of!(Fault, access) == 12
        && std::mem::offset_of!(Fault, address) == 16
);

/// The codes a fault of the leaf's is sent to its exit for; any other runs as before.
const ACCESS_VIOLATION: u32 = windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION as u32;
const GUARD_PAGE: u32 = windows_sys::Win32::Foundation::STATUS_GUARD_PAGE_VIOLATION as u32;
const IN_PAGE: u32 = windows_sys::Win32::Foundation::STATUS_IN_PAGE_ERROR as u32;
/// `ExceptionInformation[0]` of a read and of a write.
const READ: u32 = 0;
const WRITE: u32 = 1;

/// Which of the two ranges is title memory, the one a fault may be in.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Side {
    /// A read of title memory (`src`) into ours.
    Read,
    /// A store of ours into title memory (`dst`).
    Store,
}

/// Why title memory could not be copied.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// An access violation: not committed, freed, or its protection refuses the access.
    NoAccess,
    /// A `PAGE_GUARD` page: the system cleared that guard as it raised the fault.
    Guard,
    /// The page could not be brought in.
    InPage,
}

/// What [`copy`] did.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Copied {
    /// All the bytes.
    Whole,
    /// A fault of `kind` in title memory at or after its byte `done`; the first `done` bytes
    /// are certainly copied, and nothing is said of the destination's bytes past them. `done`
    /// is a lower bound: callers resume from it and never treat the bytes from `done` to the
    /// fault as unreadable or fill them (S0-2 finds it exact on its machine; the architecture
    /// does not promise that).
    Faulted { done: usize, kind: Kind },
}

/// Where a fault at instruction `rip` is to go on: the leaf's failing exit if `rip` is its
/// `rep movsb`, else none.
pub(crate) fn exit_for(rip: usize) -> Option<usize> {
    let at = victory_user_copy_at as *const () as usize;
    let after = victory_user_copy_after as *const () as usize;
    (at..after).contains(&rip).then_some(victory_user_copy_failed as *const () as usize)
}

/// The vectored handler's first lines: a fault of one of the three codes whose instruction is
/// the leaf's goes to its exit, the `Fault` at the context's `r9` filled and `Rip` changed,
/// nothing else (true: the handler continues execution). Takes no lock, allocates nothing,
/// makes no system call and uses no thread state. Any other fault: false, untouched.
///
/// # Safety
///
/// `info` is the exception being dispatched, as the system passes it to a vectored handler.
pub(crate) unsafe fn redirect(
    info: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
) -> bool {
    // SAFETY: the system's record and the faulting thread's context, for this dispatch.
    let (r, c) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
    let code = r.ExceptionCode as u32;
    if !matches!(code, ACCESS_VIOLATION | GUARD_PAGE | IN_PAGE) || r.NumberParameters < 2 {
        return false;
    }
    let Some(exit) = exit_for(c.Rip as usize) else {
        return false;
    };
    let fault = c.R9 as usize as *mut Fault;
    // SAFETY: the leaf is entered from `leaf` alone, whose `r9` (which the leaf keeps) points
    // at that call's own `Fault`; the exit writes `done`.
    unsafe {
        (&raw mut (*fault).code).write(code);
        (&raw mut (*fault).access).write(r.ExceptionInformation[0] as u32);
        (&raw mut (*fault).address).write(r.ExceptionInformation[1]);
    }
    c.Rip = exit as u64;
    true
}

/// The leaf: `None` if all `len` bytes were copied, else the fault, as the handler and the
/// exit wrote it.
///
/// # Safety
///
/// As [`copy`]'s.
#[cfg_attr(not(test), allow(dead_code))]
unsafe fn leaf(dst: *mut u8, src: *const u8, len: usize) -> Option<Fault> {
    let mut fault = Fault::default();
    // SAFETY: the caller's; a fault in the leaf's copy leaves through its failing exit.
    let whole = unsafe { victory_user_copy(dst, src, len, &mut fault) } != 0;
    (!whole).then_some(fault)
}

/// Whether `fault` is one [`copy`] may report: of the three codes, and an access of `side`'s
/// kind at title memory's byte `done` or later, inside the `len` bytes. Otherwise why not.
#[cfg_attr(not(test), allow(dead_code))]
fn check(fault: &Fault, dst: usize, src: usize, len: usize, side: Side) -> Result<Kind, String> {
    let (start, access) = match side {
        Side::Read => (src, READ),
        Side::Store => (dst, WRITE),
    };
    let kind = match fault.code {
        ACCESS_VIOLATION => Some(Kind::NoAccess),
        GUARD_PAGE => Some(Kind::Guard),
        IN_PAGE => Some(Kind::InPage),
        _ => None,
    };
    let inside = fault.done < len && (start + fault.done..start + len).contains(&fault.address);
    match kind {
        Some(kind) if fault.access == access && inside => Ok(kind),
        _ => Err(format!(
            "the user copy's {side:?} of {len} bytes faulted where it may not: code {:#010x}, \
             access {} ({}), at {:+} bytes from the source and {:+} from the destination, \
             {} bytes done",
            fault.code,
            fault.access,
            match fault.access {
                READ => "a read",
                WRITE => "a write",
                _ => "neither read nor write",
            },
            fault.address.wrapping_sub(src) as isize,
            fault.address.wrapping_sub(dst) as isize,
            fault.done,
        )),
    }
}

/// Copies `len` bytes from `src` to `dst`, one side of which is title memory (`side`). A
/// fault in title memory at or after the bytes done gives [`Copied::Faulted`]; any other
/// (a fault on our own side, a misplaced label) ends the run (`crate::host::fail`).
///
/// # Safety
///
/// The side that is not title memory is ours, valid for `len` bytes, and no Rust reference
/// covers the destination; the two do not overlap. [`on`] has answered true (without the
/// handler, a fault ends the process).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) unsafe fn copy(dst: *mut u8, src: *const u8, len: usize, side: Side) -> Copied {
    if len == 0 {
        return Copied::Whole;
    }
    // SAFETY: the caller's.
    match unsafe { leaf(dst, src, len) } {
        None => Copied::Whole,
        Some(fault) => match check(&fault, dst.addr(), src.addr(), len, side) {
            Ok(kind) => Copied::Faulted { done: fault.done, kind },
            Err(why) => crate::host::fail(format_args!("{why}")),
        },
    }
}

/// Whether the user copy can be used in this run: asked once, by its probe ([`probe`]); a
/// probe that fails leaves it off, with one warning saying why. With S0 only the fixtures ask;
/// S1 and S2 ask at their switch's first read.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match probe() {
        Ok(()) => true,
        Err(why) => {
            tracing::warn!("the user copy stays off for this run: {why}");
            false
        }
    })
}

/// Tests: where the probe reads instead of its own reservation (0: there).
#[cfg(test)]
static PROBE_AT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The self-probe: reserves 64 KiB with no access, copies 16 bytes from it and frees it. The
/// answer must be a fault with no byte done, a read at the region's start; that shows the
/// labels and the handler work in this exact build. A fault the handler does not take ends
/// the process with the loader's report.
#[cfg_attr(not(test), allow(dead_code))]
fn probe() -> Result<(), String> {
    use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, VirtualAlloc, VirtualFree};
    const PROBE: usize = 64 << 10;
    crate::watch::install();
    // SAFETY: a fresh reservation, freed below.
    let region = unsafe { VirtualAlloc(std::ptr::null(), PROBE, MEM_RESERVE, PAGE_NOACCESS) };
    if region.is_null() {
        return Err("no address space for its probe".into());
    }
    #[cfg(test)]
    let at = match PROBE_AT.load(std::sync::atomic::Ordering::SeqCst) {
        0 => region as usize,
        hooked => hooked,
    };
    #[cfg(not(test))]
    let at = region as usize;
    let mut into = [0u8; 16];
    // SAFETY: our buffer; the source is the reservation (no access: the fault is the probe).
    let answer = unsafe { leaf(into.as_mut_ptr(), at as *const u8, into.len()) };
    // SAFETY: the reservation above.
    unsafe { VirtualFree(region, 0, MEM_RELEASE) };
    match answer {
        Some(f) if f.done == 0 && f.code == ACCESS_VIOLATION && f.access == READ && f.address == at => Ok(()),
        Some(f) => Err(format!(
            "its probe of a page with no access faulted as {f:?}, not as a read at the page's start with no byte done"
        )),
        None => Err("its probe of a page with no access read all 16 bytes".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS,
        PAGE_READONLY, PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualProtect, VirtualQuery,
    };

    const PAGE: usize = 4096;
    /// Set in a fixture's child process: a child run by hand without it does nothing.
    const CHILD: &str = "VICTORY_SAFECOPY_CHILD";

    /// Pages of ours: `pages` of 4 KiB reserved, the first `committed` committed read-write.
    struct Region {
        at: usize,
    }

    impl Region {
        fn new(pages: usize, committed: usize) -> Region {
            // SAFETY: a fresh reservation, freed on drop.
            let at = unsafe { VirtualAlloc(std::ptr::null(), pages * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as usize;
            assert_ne!(at, 0, "no address space");
            if committed > 0 {
                // SAFETY: pages of the reservation.
                let done = unsafe { VirtualAlloc(at as *const _, committed * PAGE, MEM_COMMIT, PAGE_READWRITE) };
                assert!(!done.is_null(), "no memory");
            }
            Region { at }
        }
        fn page(&self, n: usize) -> usize {
            self.at + n * PAGE
        }
        /// Byte `i` of a region's pattern: never 0xEE, the destinations' fill.
        fn pattern(i: usize) -> u8 {
            (i % 200) as u8
        }
        /// The pattern into the first `bytes`, which are committed and writable.
        fn fill(&self, bytes: usize) {
            for i in 0..bytes {
                // SAFETY: committed pages of ours.
                unsafe { ((self.at + i) as *mut u8).write(Self::pattern(i)) };
            }
        }
        fn protect(&self, page: usize, protection: u32) {
            let mut old = 0;
            // SAFETY: a committed page of ours.
            assert_ne!(unsafe { VirtualProtect(self.page(page) as *const _, PAGE, protection, &mut old) }, 0);
        }
    }

    impl Drop for Region {
        fn drop(&mut self) {
            // SAFETY: our reservation.
            unsafe { VirtualFree(self.at as *mut _, 0, MEM_RELEASE) };
        }
    }

    /// The protection of the page at `at`.
    fn protection(at: usize) -> u32 {
        // SAFETY: plain data, filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: room for the answer.
        assert_ne!(unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) }, 0);
        info.Protect
    }

    /// A small fast generator, seeded: the same 2,000 copies each run.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// Whether this is a fixture's child; a child run by hand says so and passes.
    fn in_child() -> bool {
        let child = std::env::var_os(CHILD).is_some();
        if !child {
            eprintln!("NOT RUN: a child of a fixture, run by its parent");
        }
        child
    }

    /// The child test `name` (this module's) in a process of its own: its exit code and its
    /// output. A child that has not ended within 30 s is killed and fails the parent.
    fn child(name: &str) -> (Option<i32>, String) {
        let name = format!("safecopy::tests::{name}");
        let mut child = std::process::Command::new(std::env::current_exe().expect("the test binary"))
            .args(["--exact", &name, "--ignored", "--test-threads", "1", "--nocapture"])
            .env(CHILD, "1")
            // `crate::host::fail` (S0-5's own-side arm) creates a device if none is there: on
            // WARP, however the parent was run.
            .env("VICTORY_WARP", "1")
            .env("VICTORY_D3D12_WARP", "1")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("the child ran");
        // Read as it comes, so a full pipe cannot stall the child.
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
        let started = std::time::Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if started.elapsed() > std::time::Duration::from_secs(30) {
                let _ = child.kill();
                panic!("{name}: the child did not end within 30 s (a fault that never ends?)");
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        };
        (status.code(), format!("{}{}", out.join().unwrap(), err.join().unwrap()))
    }

    /// The child `name` passed: exit code 0 and its one test passed. Its output, for the record.
    fn child_passes(name: &str) -> String {
        let (code, text) = child(name);
        assert!(code == Some(0) && text.contains("1 passed"), "{name}: the child failed (exit {code:?}):\n{text}");
        text
    }

    /// What a child recorded: each line's text from `S0-` on (the first follows the harness's
    /// `test … ` on its line).
    fn recorded(text: &str) -> Vec<&str> {
        text.lines().filter_map(|l| l.find("S0-").map(|at| &l[at..])).collect()
    }

    // S0-1 -----------------------------------------------------------------------------------

    /// 2,000 copies of up to 256 KiB, destination and source 0 to 63 bytes past a 4 KiB edge
    /// less 64 (so short ones cross it too), all readable: each `Whole`, the bytes
    /// `ptr::copy` gives, and no byte outside the destination touched. And a copy of no bytes
    /// is `Whole`.
    #[test]
    fn s0_1_a_readable_copy_is_whole_and_exact() {
        const MOST: usize = 256 << 10;
        const PAGES: usize = MOST / PAGE + 4;
        let (src, dst) = (Region::new(PAGES, PAGES), Region::new(PAGES, PAGES));
        src.fill(PAGES * PAGE);
        let size = PAGES * PAGE;
        // SAFETY: the two regions, committed, ours alone.
        let (s, d) = unsafe {
            (std::slice::from_raw_parts(src.at as *const u8, size), std::slice::from_raw_parts_mut(dst.at as *mut u8, size))
        };
        d.fill(0xEE);
        let fill = vec![0xEEu8; size];
        let mut expected = vec![0u8; MOST];
        let mut rng = Rng(0x5afe_c0de_2026_1126);
        for n in 0..2000 {
            let len = match rng.below(4) {
                0 => rng.below(300),
                1 => (1 + rng.below(64)) * PAGE - 64 + rng.below(128),
                _ => rng.below(MOST + 1),
            }
            .min(MOST);
            let (from, to) = (PAGE - 64 + rng.below(64), PAGE - 64 + rng.below(64));
            // SAFETY: inside the two regions (an edge less 64, plus 63, plus 256 KiB).
            let copied = unsafe { copy(d.as_mut_ptr().add(to), s.as_ptr().add(from), len, Side::Read) };
            assert_eq!(copied, Copied::Whole, "copy {n}: {len} bytes from +{from} to +{to}");
            // SAFETY: the source and our own buffer, apart.
            unsafe { std::ptr::copy(s.as_ptr().add(from), expected.as_mut_ptr(), len) };
            assert!(d[to..to + len] == expected[..len], "copy {n}: {len} bytes from +{from} to +{to}: other bytes");
            assert!(d[..to] == fill[..to] && d[to + len..] == fill[to + len..], "copy {n}: a byte outside the destination");
            d[to..to + len].fill(0xEE);
        }
        // SAFETY: no byte is read or written.
        assert_eq!(unsafe { copy(d.as_mut_ptr(), s.as_ptr(), 0, Side::Read) }, Copied::Whole);
    }

    // S0-2 -----------------------------------------------------------------------------------

    /// A copy that runs into a hole stops at its first byte, and says so exactly: committed
    /// pages then a reserved one, and the third of five pages decommitted. Every length
    /// crossing the hole by 1 to 64 bytes, every source alignment 0 to 15, from four distances
    /// before it (0 to 8,015 bytes): `Faulted { NoAccess }`, a read at the hole's first byte,
    /// `done` exactly the bytes before it, and those bytes the source's. Whether bytes past
    /// `done` changed is recorded, not asserted.
    #[test]
    fn s0_2_a_fault_reports_exactly_the_bytes_before_the_hole() {
        let text = child_passes("s0_2_child");
        for line in recorded(&text) {
            println!("{line}");
        }
    }

    #[test]
    #[ignore = "a child of s0_2_a_fault_reports_exactly_the_bytes_before_the_hole"]
    fn s0_2_child() {
        if !in_child() {
            return;
        }
        assert!(on());
        // Committed pages then a reserved one: the hole at page 3.
        let reserved = Region::new(4, 3);
        reserved.fill(3 * PAGE);
        // Five pages, the third decommitted: the hole at page 2.
        let decommitted = Region::new(5, 5);
        decommitted.fill(5 * PAGE);
        // SAFETY: a page of ours.
        assert_ne!(unsafe { VirtualFree(decommitted.page(2) as *mut _, PAGE, MEM_DECOMMIT) }, 0);
        for (arm, region, hole) in [("reserved", &reserved, reserved.page(3)), ("decommitted", &decommitted, decommitted.page(2))] {
            let (mut copies, mut wrong, mut past) = (0, Vec::new(), 0);
            let mut into = vec![0xEEu8; 8192 + 64 + 64];
            for align in 0..16 {
                let c = (16 - align) % 16;
                for before in [c, c + 48, c + PAGE + 16, c + 8000] {
                    for over in 1..=64 {
                        let len = before + over;
                        let src = hole - before;
                        assert_eq!(src % 16, align);
                        let to = (align * 7) % 64;
                        into.fill(0xEE);
                        // SAFETY: our buffer, large enough; the source is ours, with the hole in it.
                        let (fault, copied) = unsafe {
                            let dst = into.as_mut_ptr().add(to);
                            (leaf(dst, src as *const u8, len), copy(dst, src as *const u8, len, Side::Read))
                        };
                        copies += 1;
                        let ok = fault.is_some_and(|f| {
                            f.done == before && f.code == ACCESS_VIOLATION && f.access == READ && f.address == hole
                        }) && copied == (Copied::Faulted { done: before, kind: Kind::NoAccess });
                        if !ok {
                            wrong.push(format!("{len} bytes from {before} before the hole: {fault:?}, {copied:?}"));
                            continue;
                        }
                        let start = region.at;
                        let same = (0..before).all(|i| into[to + i] == Region::pattern(src - start + i));
                        if !same {
                            wrong.push(format!("{len} bytes from {before} before the hole: the bytes before it differ"));
                        }
                        if into[to + before..to + len].iter().any(|&b| b != 0xEE) {
                            past += 1;
                        }
                    }
                }
            }
            println!("S0-2 {arm}: {copies} copies, {} wrong, {past} with bytes changed past done", wrong.len());
            assert!(wrong.is_empty(), "{arm}: {} of {copies} wrong, the first:\n{}", wrong.len(), wrong[..wrong.len().min(8)].join("\n"));
        }
    }

    // S0-3 -----------------------------------------------------------------------------------

    /// What the kernel answers for one read or write: whether it said it worked, the bytes it
    /// moved, the error or status, and the protection of the guard page after.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Kernel {
        ok: bool,
        moved: usize,
        error: u32,
        after: u32,
    }

    #[link(name = "ntdll", kind = "raw-dylib")]
    unsafe extern "system" {
        fn NtWriteVirtualMemory(
            process: *mut std::ffi::c_void,
            base: *mut std::ffi::c_void,
            buffer: *const std::ffi::c_void,
            size: usize,
            written: *mut usize,
        ) -> i32;
    }

    /// Today's read (`kernelstat::read_memory`'s call): `ReadProcessMemory` of our own process.
    fn kernel_read(src: usize, into: &mut [u8], guard: usize) -> Kernel {
        use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let mut moved = 0;
        // SAFETY: our buffer; the kernel checks the source.
        let ok = unsafe {
            ReadProcessMemory(GetCurrentProcess(), src as *const _, into.as_mut_ptr().cast(), into.len(), &mut moved)
        } != 0;
        // SAFETY: a plain call.
        let error = if ok { 0 } else { unsafe { windows_sys::Win32::Foundation::GetLastError() } };
        Kernel { ok, moved, error, after: protection(guard) }
    }

    /// Today's store (`kernelstat::write_memory`'s call): `NtWriteVirtualMemory`.
    fn kernel_write(dst: usize, from: &[u8], guard: usize) -> Kernel {
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let mut moved = 0;
        // SAFETY: our buffer; the kernel checks the destination.
        let status = unsafe { NtWriteVirtualMemory(GetCurrentProcess(), dst as *mut _, from.as_ptr().cast(), from.len(), &mut moved) };
        Kernel { ok: status >= 0, moved, error: status as u32, after: protection(guard) }
    }

    /// The leaf at a test `PAGE_GUARD` page: `Faulted { Guard }`, `done` exact, a read at the
    /// page's first byte, and the guard gone after. The kernel's own answer at a fresh guard
    /// page, for `ReadProcessMemory` and `NtWriteVirtualMemory` (the result, the bytes moved,
    /// `PAGE_GUARD` after, and a second call's), recorded and printed, not asserted. And the
    /// parity S1 builds on: after the leaf's fault, with the guard put back, the kernel's read
    /// answers exactly as at a fresh guard page.
    #[test]
    fn s0_3_a_guard_page() {
        let text = child_passes("s0_3_child");
        for line in recorded(&text) {
            println!("{line}");
        }
    }

    #[test]
    #[ignore = "a child of s0_3_a_guard_page"]
    fn s0_3_child() {
        if !in_child() {
            return;
        }
        assert!(on());
        const GUARDED: u32 = PAGE_READWRITE | PAGE_GUARD;
        let region = Region::new(3, 3);
        region.fill(3 * PAGE);
        let guard = region.page(1);
        let (src, len, before) = (guard - 96, 200, 96);
        let mut into = vec![0xEEu8; len];

        // The leaf.
        region.protect(1, GUARDED);
        // SAFETY: our buffer; our pages, the second guarded.
        let fault = unsafe { leaf(into.as_mut_ptr(), src as *const u8, len) };
        assert_eq!(
            fault,
            Some(Fault { done: before, code: GUARD_PAGE, access: READ, address: guard }),
            "a read at the guard page's first byte, the bytes before it done"
        );
        assert!((0..before).all(|i| into[i] == Region::pattern(src - region.at + i)));
        assert_eq!(protection(guard), PAGE_READWRITE, "the system cleared the guard as it raised the fault");
        region.protect(1, GUARDED);
        into.fill(0xEE);
        // SAFETY: as above.
        let copied = unsafe { copy(into.as_mut_ptr(), src as *const u8, len, Side::Read) };
        assert_eq!(copied, Copied::Faulted { done: before, kind: Kind::Guard });
        println!("S0-3 leaf at a guard page: {fault:?}, guard after {:#x}", protection(guard));

        // The kernel's read at a fresh guard page, then once more.
        region.protect(1, GUARDED);
        into.fill(0xEE);
        let fresh = kernel_read(src, &mut into, guard);
        let fresh_bytes = into.clone();
        let again = kernel_read(src, &mut into, guard);
        println!("S0-3 kernel read at a fresh guard page: {fresh:?}; the same read again: {again:?}");
        println!(
            "S0-3 kernel read's bytes: the {} before the page {}, the {} from it {}",
            before,
            if (0..before).all(|i| fresh_bytes[i] == Region::pattern(src - region.at + i)) { "the source's" } else { "not the source's" },
            len - before,
            if (before..len).all(|i| fresh_bytes[i] == Region::pattern(src - region.at + i)) {
                "the source's"
            } else if fresh_bytes[before..].iter().all(|&b| b == 0xEE) {
                "untouched"
            } else {
                "partly written"
            }
        );

        // The kernel's write at a fresh guard page, then once more.
        let from: Vec<u8> = (0..len).map(|i| 0x40 ^ i as u8).collect();
        region.protect(1, GUARDED);
        let fresh_write = kernel_write(src, &from, guard);
        let again_write = kernel_write(src, &from, guard);
        println!("S0-3 kernel write at a fresh guard page: {fresh_write:?}; the same write again: {again_write:?}");
        // The guard may still be there: off before the test's own stores.
        region.protect(1, PAGE_READWRITE);
        region.fill(3 * PAGE);

        // Parity: the leaf's fault, the guard put back (S1's put-back), then the kernel.
        region.protect(1, GUARDED);
        // SAFETY: as above.
        assert_eq!(
            unsafe { copy(into.as_mut_ptr(), src as *const u8, len, Side::Read) },
            Copied::Faulted { done: before, kind: Kind::Guard }
        );
        region.protect(1, GUARDED);
        into.fill(0xEE);
        let put_back = kernel_read(src, &mut into, guard);
        assert_eq!(put_back, fresh, "after the leaf's fault and the guard put back, the kernel answers as at a fresh guard page");
        assert!(into == fresh_bytes, "and moves the same bytes");
        // And what the put-back prevents: the kernel after the leaf's fault without it.
        region.protect(1, GUARDED);
        // SAFETY: as above.
        unsafe { copy(into.as_mut_ptr(), src as *const u8, len, Side::Read) };
        into.fill(0xEE);
        let not_put_back = kernel_read(src, &mut into, guard);
        println!("S0-3 kernel read after the leaf's fault, the guard not put back: {not_put_back:?}");
    }

    // S0-4 -----------------------------------------------------------------------------------

    // A shim that puts known values in rbx, rbp, rsi, rdi and r12-r15, calls the leaf with its
    // own four arguments, and returns in its low 32 bits which of them changed (bit 0 rbx,
    // 1 rbp, 2 rsi, 3 rdi, 4-7 r12-r15, 8 the direction flag set, 9 rsp moved) and in its high
    // 32 bits the leaf's eax.
    std::arch::global_asm!(
        ".globl victory_user_copy_shim",
        "victory_user_copy_shim:",
        "push rbx",
        "push rbp",
        "push rsi",
        "push rdi",
        "push r12",
        "push r13",
        "push r14",
        "push r15",
        "sub rsp, 40",
        "mov rax, rsp",
        "mov qword ptr [rsp + 32], rax",
        "mov rbx, 0x1111111111111111",
        "mov rbp, 0x2222222222222222",
        "mov rsi, 0x3333333333333333",
        "mov rdi, 0x4444444444444444",
        "mov r12, 0x5555555555555555",
        "mov r13, 0x6666666666666666",
        "mov r14, 0x7777777777777777",
        "mov r15, 0x0123456789abcdef",
        "call victory_user_copy",
        "mov edx, eax",
        "xor r10d, r10d",
        "mov r11, 0x1111111111111111",
        "cmp rbx, r11",
        "je 2f",
        "or r10d, 1",
        "2:",
        "mov r11, 0x2222222222222222",
        "cmp rbp, r11",
        "je 3f",
        "or r10d, 2",
        "3:",
        "mov r11, 0x3333333333333333",
        "cmp rsi, r11",
        "je 4f",
        "or r10d, 4",
        "4:",
        "mov r11, 0x4444444444444444",
        "cmp rdi, r11",
        "je 5f",
        "or r10d, 8",
        "5:",
        "mov r11, 0x5555555555555555",
        "cmp r12, r11",
        "je 6f",
        "or r10d, 16",
        "6:",
        "mov r11, 0x6666666666666666",
        "cmp r13, r11",
        "je 7f",
        "or r10d, 32",
        "7:",
        "mov r11, 0x7777777777777777",
        "cmp r14, r11",
        "je 8f",
        "or r10d, 64",
        "8:",
        "mov r11, 0x0123456789abcdef",
        "cmp r15, r11",
        "je 9f",
        "or r10d, 128",
        "9:",
        "mov r11, qword ptr [rsp + 32]",
        "cmp r11, rsp",
        "je 22f",
        "or r10d, 512",
        "22:",
        "pushfq",
        "pop rax",
        "shr eax, 10",
        "and eax, 1",
        "shl eax, 8",
        "or r10d, eax",
        "mov eax, r10d",
        "shl rdx, 32",
        "or rax, rdx",
        "add rsp, 40",
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rdi",
        "pop rsi",
        "pop rbp",
        "pop rbx",
        "ret",
    );

    unsafe extern "C" {
        fn victory_user_copy_shim(dst: *mut u8, src: *const u8, len: usize, fault: *mut Fault) -> u64;
    }

    /// The shim's answer: (registers changed and the direction flag, the leaf's eax).
    fn through_shim(dst: *mut u8, src: usize, len: usize, fault: &mut Fault) -> (u32, u32) {
        // SAFETY: the callers' buffers; the shim gives back every register the ABI keeps.
        let r = unsafe { victory_user_copy_shim(dst, src as *const u8, len, fault) };
        (r as u32, (r >> 32) as u32)
    }

    /// The leaf keeps rbx, rbp, rsi, rdi, r12-r15 and rsp and leaves the direction flag 0,
    /// whole (here) and faulting (in a child).
    #[test]
    fn s0_4_the_leaf_keeps_its_callers_registers() {
        let src = Region::new(1, 1);
        src.fill(PAGE);
        let mut into = [0u8; 100];
        let mut fault = Fault::default();
        let (changed, eax) = through_shim(into.as_mut_ptr(), src.at, into.len(), &mut fault);
        assert_eq!((changed, eax, fault.done), (0, 1, 100), "whole: changed {changed:#b}");
        assert!((0..100).all(|i| into[i] == Region::pattern(i)));
        child_passes("s0_4_child");
    }

    #[test]
    #[ignore = "a child of s0_4_the_leaf_keeps_its_callers_registers"]
    fn s0_4_child() {
        if !in_child() {
            return;
        }
        assert!(on());
        let src = Region::new(2, 1);
        src.fill(PAGE);
        let mut into = [0u8; 100];
        let mut fault = Fault::default();
        let (changed, eax) = through_shim(into.as_mut_ptr(), src.page(1) - 40, into.len(), &mut fault);
        assert_eq!((changed, eax), (0, 0), "faulting: changed {changed:#b}");
        assert_eq!(fault, Fault { done: 40, code: ACCESS_VIOLATION, access: READ, address: src.page(1) });
    }

    /// What the unwind of [`unwind_observer`] found: whether it ran, whether the leaf has a
    /// function entry, and the caller's rsi, rdi and rip as the unwind gives them.
    static UNWOUND: [AtomicUsize; 5] = [const { AtomicUsize::new(0) }; 5];

    /// Ahead of `on_write`: at a fault on the leaf's `rep movsb`, unwinds one frame from the
    /// faulting context as the system's frame search would (the function entry and
    /// `RtlVirtualUnwind`, or, with no entry, as a leaf: rip from `[rsp]`, registers as they
    /// are), and leaves the fault to `on_write`.
    unsafe extern "system" fn unwind_observer(
        info: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
    ) -> i32 {
        use windows_sys::Win32::System::Diagnostics::Debug::{
            EXCEPTION_CONTINUE_SEARCH, RtlLookupFunctionEntry, RtlVirtualUnwind, UNW_FLAG_NHANDLER,
        };
        // SAFETY: the system's context for this dispatch, copied.
        let mut c = unsafe { *(*info).ContextRecord };
        if c.Rip as usize != victory_user_copy_at as *const () as usize || UNWOUND[0].load(SeqCst) != 0 {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let pc = c.Rip;
        let mut base = 0u64;
        // SAFETY: a lookup in our own image's function table.
        let entry = unsafe { RtlLookupFunctionEntry(pc, &mut base, std::ptr::null_mut()) };
        if entry.is_null() {
            // SAFETY: the faulting thread's stack, its return address on top in a leaf.
            c.Rip = unsafe { (c.Rsp as *const u64).read() };
            c.Rsp += 8;
        } else {
            let (mut data, mut frame) = (std::ptr::null_mut(), 0u64);
            // SAFETY: the entry found for `pc`, a copy of the faulting context.
            unsafe { RtlVirtualUnwind(UNW_FLAG_NHANDLER, base, pc, entry, &mut c, &mut data, &mut frame, std::ptr::null_mut()) };
        }
        UNWOUND[1].store(usize::from(!entry.is_null()), SeqCst);
        UNWOUND[2].store(c.Rsi as usize, SeqCst);
        UNWOUND[3].store(c.Rdi as usize, SeqCst);
        UNWOUND[4].store(c.Rip as usize, SeqCst);
        UNWOUND[0].store(1, SeqCst);
        EXCEPTION_CONTINUE_SEARCH
    }

    /// An unwind through a fault inside the leaf (one no handler has taken yet) finds the
    /// leaf's function entry and gives the caller's rsi and rdi (the shim's values) and the
    /// shim's return address: what a frame-based handler search, `RtlUnwindEx`, a debugger or
    /// a crash dump sees above the leaf.
    #[test]
    fn s0_4_an_unwind_through_a_leaf_fault_gives_the_callers_rsi_and_rdi() {
        let text = child_passes("s0_4_unwind_child");
        for line in recorded(&text) {
            println!("{line}");
        }
    }

    #[test]
    #[ignore = "a child of s0_4_an_unwind_through_a_leaf_fault_gives_the_callers_rsi_and_rdi"]
    fn s0_4_unwind_child() {
        use windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler;
        if !in_child() {
            return;
        }
        assert!(on());
        // SAFETY: a handler of this child's, first in line, for the rest of the process.
        assert!(!unsafe { AddVectoredExceptionHandler(1, Some(unwind_observer)) }.is_null());
        let src = Region::new(2, 1);
        src.fill(PAGE);
        let mut into = [0u8; 100];
        let mut fault = Fault::default();
        let (changed, eax) = through_shim(into.as_mut_ptr(), src.page(1) - 40, into.len(), &mut fault);
        assert_eq!((changed, eax, fault.done), (0, 0, 40), "the fault went on to the exit as before");
        let [seen, entry, rsi, rdi, rip] = UNWOUND.each_ref().map(|a| a.load(SeqCst));
        let shim = victory_user_copy_shim as *const () as usize;
        println!(
            "S0-4 unwind from the fault at the leaf's rep movsb: function entry {}, caller's rsi {rsi:#x}, rdi {rdi:#x}, rip {}",
            if entry == 1 { "found" } else { "none" },
            if (shim..shim + 0x200).contains(&rip) { "in the shim" } else { "elsewhere" }
        );
        assert_eq!(seen, 1, "the observer saw the fault");
        assert_eq!(entry, 1, "the leaf has a function entry");
        assert_eq!((rsi, rdi), (0x3333_3333_3333_3333, 0x4444_4444_4444_4444), "the caller's rsi and rdi");
        assert!((shim..shim + 0x200).contains(&rip), "the unwound rip is the shim's return address");
    }

    // S0-5 -----------------------------------------------------------------------------------

    /// Only the leaf's faults are taken: a raw read of a page with no access outside the leaf,
    /// with the handler installed, ends the child with an access violation, as it would
    /// without us: the test crash reporter's unhandled line, for a read of that page's first
    /// byte. (The exit code alone is not enough: a handler that redirected this fault sent the
    /// child into a stack overflow, which ends with the same code.)
    #[test]
    fn s0_5_a_fault_outside_the_leaf_is_not_ours() {
        let (code, text) = child("s0_5_outside_child");
        let at = text.split("S0-5 reading ").nth(1).and_then(|rest| rest.split_whitespace().next());
        let reported = at.is_some_and(|at| {
            text.contains("unhandled exception 0xc0000005 at ") && text.contains(&format!(", a read of {at} "))
        });
        assert!(
            code == Some(ACCESS_VIOLATION as i32) && reported && !text.contains("1 passed"),
            "the child must end with an access violation, reported as a read of {at:?} (exit {code:?}):\n{text}"
        );
    }

    #[test]
    #[ignore = "a child of s0_5_a_fault_outside_the_leaf_is_not_ours"]
    fn s0_5_outside_child() {
        if !in_child() {
            return;
        }
        assert!(on());
        let hole = Region::new(1, 0);
        println!("S0-5 reading {:#x} ", hole.at);
        // SAFETY: none: the fault is the fixture.
        let byte = unsafe { (hole.at as *const u8).read_volatile() };
        println!("S0-5 a raw read of a page with no access came back with {byte}");
    }

    /// A fault of the leaf's on its own destination (a read-only page of ours, the copy a
    /// read) is no title fault: the run ends through `fail`, naming it.
    #[test]
    fn s0_5_a_fault_on_our_own_side_ends_the_run() {
        let (code, text) = child("s0_5_own_side_child");
        assert!(
            code != Some(0)
                && !text.contains("1 passed")
                && text.contains("host GPU failure: the user copy's Read of 16 bytes faulted where it may not")
                && text.contains("access 1 (a write)")
                && text.contains("+0 from the destination"),
            "the child must end through fail (exit {code:?}):\n{text}"
        );
    }

    #[test]
    #[ignore = "a child of s0_5_a_fault_on_our_own_side_ends_the_run"]
    fn s0_5_own_side_child() {
        if !in_child() {
            return;
        }
        assert!(on());
        let ours = Region::new(1, 1);
        ours.protect(0, PAGE_READONLY);
        let from = [7u8; 16];
        // SAFETY: none: the fault on our own side is the fixture.
        let copied = unsafe { copy(ours.at as *mut u8, from.as_ptr(), from.len(), Side::Read) };
        println!("S0-5 a fault on our own destination came back as {copied:?}");
    }

    /// Four threads fault 10,000 times each at once, and a thread made with `CreateThread`
    /// (no Rust thread state) as many times beside them: every fault redirected, each with
    /// the exact bytes done.
    #[test]
    fn s0_5_many_threads_fault_at_once() {
        let text = child_passes("s0_5_threads_child");
        for line in recorded(&text) {
            println!("{line}");
        }
    }

    const ROUNDS: usize = 10_000;
    static HOLE: AtomicUsize = AtomicUsize::new(0);
    static GO: AtomicUsize = AtomicUsize::new(0);

    /// `ROUNDS` copies running 1 to 64 bytes into the hole at [`HOLE`]: how many faulted as
    /// they must.
    fn fault_rounds(seed: usize) -> usize {
        let hole = HOLE.load(SeqCst);
        let mut into = [0u8; 128];
        let mut right = 0;
        for n in 0..ROUNDS {
            let before = (n * 7 + seed) % 64;
            // SAFETY: our buffer; the source runs into the hole.
            let copied = unsafe { copy(into.as_mut_ptr(), (hole - before) as *const u8, before + 64, Side::Read) };
            right += usize::from(copied == Copied::Faulted { done: before, kind: Kind::NoAccess });
        }
        right
    }

    unsafe extern "system" fn raw_thread(_: *mut std::ffi::c_void) -> u32 {
        while GO.load(SeqCst) == 0 {
            std::hint::spin_loop();
        }
        fault_rounds(5) as u32
    }

    #[test]
    #[ignore = "a child of s0_5_many_threads_fault_at_once"]
    fn s0_5_threads_child() {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{CreateThread, GetExitCodeThread, INFINITE, WaitForSingleObject};
        if !in_child() {
            return;
        }
        assert!(on());
        let region = Region::new(2, 1);
        HOLE.store(region.page(1), SeqCst);
        // SAFETY: a thread of ours running `raw_thread`, joined below.
        let raw = unsafe { CreateThread(std::ptr::null(), 0, Some(raw_thread), std::ptr::null(), 0, std::ptr::null_mut()) };
        assert!(!raw.is_null());
        let start = std::sync::Arc::new(std::sync::Barrier::new(4));
        let threads: Vec<_> = (0..4)
            .map(|k| {
                let start = start.clone();
                std::thread::spawn(move || {
                    start.wait();
                    if k == 0 {
                        GO.store(1, SeqCst);
                    }
                    fault_rounds(k)
                })
            })
            .collect();
        let rust: Vec<usize> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        let mut code = 0;
        // SAFETY: the thread above.
        unsafe {
            WaitForSingleObject(raw, INFINITE);
            assert_ne!(GetExitCodeThread(raw, &mut code), 0);
            CloseHandle(raw);
        }
        println!("S0-5 faults redirected as they must, of {ROUNDS} a thread: {rust:?} and {code} on the CreateThread one");
        assert_eq!((rust, code as usize), (vec![ROUNDS; 4], ROUNDS));
    }

    // S0-6 -----------------------------------------------------------------------------------

    /// The self-probe answers true in this build.
    #[test]
    fn s0_6_the_self_probe_answers_true() {
        child_passes("s0_6_true_child");
    }

    #[test]
    #[ignore = "a child of s0_6_the_self_probe_answers_true"]
    fn s0_6_true_child() {
        if !in_child() {
            return;
        }
        assert_eq!(probe(), Ok(()));
        assert!(on());
    }

    /// A probe pointed at committed memory (the test hook) reads it whole: `on()` is false.
    #[test]
    fn s0_6_a_probe_that_does_not_fault_leaves_it_off() {
        child_passes("s0_6_false_child");
    }

    #[test]
    #[ignore = "a child of s0_6_a_probe_that_does_not_fault_leaves_it_off"]
    fn s0_6_false_child() {
        if !in_child() {
            return;
        }
        let committed = Region::new(1, 1);
        PROBE_AT.store(committed.at, SeqCst);
        assert_eq!(probe(), Err("its probe of a page with no access read all 16 bytes".to_string()));
        assert!(!on());
    }

    // The parts ---------------------------------------------------------------------------------

    /// The exit is given for the `rep movsb` alone.
    #[test]
    fn the_exit_is_for_the_leaf_s_copy_alone() {
        let at = victory_user_copy_at as *const () as usize;
        let after = victory_user_copy_after as *const () as usize;
        let failed = victory_user_copy_failed as *const () as usize;
        assert_eq!(after - at, 2, "rep movsb: two bytes");
        // SAFETY: the two bytes of our own code at the label.
        let code = unsafe { (at as *const [u8; 2]).read() };
        assert_eq!(code, [0xF3, 0xA4], "the range is exactly rep movsb (F3 A4)");
        // One function entry, the leaf's from its first byte, covers the copy and both exits.
        let entry = victory_user_copy as *const () as usize;
        for label in [entry, at, after, failed] {
            let mut base = 0u64;
            // SAFETY: a lookup in our own image's function table.
            let found = unsafe {
                windows_sys::Win32::System::Diagnostics::Debug::RtlLookupFunctionEntry(label as u64, &mut base, std::ptr::null_mut())
            };
            assert!(!found.is_null(), "a function entry at {:+} from the leaf", label - entry);
            // SAFETY: the entry the system found.
            let (begin, end) = unsafe { ((*found).BeginAddress as usize, (*found).EndAddress as usize) };
            assert_eq!(base as usize + begin, entry, "the leaf's own entry, at {:+}", label - entry);
            assert!(base as usize + end > failed + 15, "it runs to the failing exit's ret");
        }
        assert_eq!(exit_for(at), Some(failed));
        assert_eq!(exit_for(at - 1), None);
        assert_eq!(exit_for(after), None);
        assert_eq!(exit_for(failed), None);
    }

    /// The check takes a fault of the three codes of the side's access in its range at or
    /// after `done`, and nothing else.
    #[test]
    fn the_check_takes_only_a_fault_in_its_own_range() {
        let (dst, src, len) = (0x10_0000, 0x20_0000, 100);
        let fault = |done, code, access, address| Fault { done, code, access, address };
        assert_eq!(check(&fault(10, ACCESS_VIOLATION, READ, src + 10), dst, src, len, Side::Read), Ok(Kind::NoAccess));
        assert_eq!(check(&fault(10, GUARD_PAGE, READ, src + 99), dst, src, len, Side::Read), Ok(Kind::Guard));
        assert_eq!(check(&fault(0, IN_PAGE, WRITE, dst), dst, src, len, Side::Store), Ok(Kind::InPage));
        for (f, side) in [
            (fault(10, ACCESS_VIOLATION, READ, src + 9), Side::Read),
            (fault(10, ACCESS_VIOLATION, READ, src + 100), Side::Read),
            (fault(10, ACCESS_VIOLATION, WRITE, src + 10), Side::Read),
            (fault(10, ACCESS_VIOLATION, WRITE, dst + 10), Side::Read),
            (fault(10, ACCESS_VIOLATION, READ, src + 10), Side::Store),
            (fault(10, 0xC000_001D, READ, src + 10), Side::Read),
            (fault(100, ACCESS_VIOLATION, READ, src + 100), Side::Read),
        ] {
            assert!(check(&f, dst, src, len, side).is_err(), "{f:?} {side:?}");
        }
    }
}
