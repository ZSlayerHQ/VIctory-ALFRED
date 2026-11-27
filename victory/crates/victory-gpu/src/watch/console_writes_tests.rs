//! The loader's console-instruction emulation against the real guards: a
//! `clzero` whose line the kernel refuses to write goes to the opener the
//! runtime registers ([`writing`]), and the emulation then makes the same
//! checked write again. Each fixture runs alone in a child of this test
//! binary, the guard ledger on or off as it says: some leave a guard over
//! memory given back without the guards hearing (as a title can with the
//! ledger off), and that address must not reach another test meanwhile.

use super::*;
use crate::ledger::NATIVE;
use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, CONTEXT_CONTROL_AMD64, CONTEXT_INTEGER_AMD64};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READONLY, PAGE_READWRITE, VirtualAlloc, VirtualFree,
};

const CHILD: &str = "VICTORY_CONSOLE_WRITES_TEST_CHILD";

/// `body` alone in a child of this binary (`name`: this test's path), with
/// the guard ledger on or off; in the child, `body` itself. At most a minute.
fn alone(name: &str, ledger: bool, body: fn()) {
    if std::env::var_os(CHILD).is_some() {
        assert_eq!(crate::ledger::on(), ledger);
        return body();
    }
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .env("VICTORY_GUARD_LEDGER", if ledger { "1" } else { "0" })
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let started = std::time::Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if started.elapsed() > std::time::Duration::from_secs(60) {
            let _ = child.kill();
            panic!("{name}: the child did not end in a minute");
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    let out = child.wait_with_output().unwrap();
    let said = String::from_utf8_lossy(&out.stdout);
    assert!(
        status.success() && said.contains("1 passed"),
        "{name} (ledger {}), ended with {status}:\n{said}{}",
        if ledger { "on" } else { "off" },
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A pagefile section of one block, mapped once, every byte 0xff: the
/// section and the view.
fn view() -> (usize, usize) {
    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
    // SAFETY: a fresh pagefile section, mapped once and filled.
    unsafe {
        let section =
            CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
        assert!(!section.is_null());
        let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK).Value as usize;
        assert_ne!(view, 0);
        (view as *mut u8).write_bytes(0xff, BLOCK);
        (section as usize, view)
    }
}

fn close(section: usize) {
    // SAFETY: a section handle of ours.
    unsafe { windows_sys::Win32::Foundation::CloseHandle(section as *mut std::ffi::c_void) };
}

/// The protection the host shows at `at`.
fn shown(at: usize) -> u32 {
    query(crate::kernelstat::Place::Guard, at).expect("mapped").Protect
}

/// Whether the `len` bytes at `at` are all `byte`.
fn all(at: usize, len: usize, byte: u8) -> bool {
    // SAFETY: readable memory of the test's.
    unsafe { std::slice::from_raw_parts(at as *const u8, len) }.iter().all(|&b| b == byte)
}

/// A page holding `clzero` (0F 01 FC), allocated before a test gives any
/// memory back, so that it cannot land where that memory was.
struct Clzero(usize);

impl Clzero {
    fn new() -> Self {
        // SAFETY: a fresh page of ours for the instruction.
        let code = unsafe { VirtualAlloc(std::ptr::null(), 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(code, 0);
        // SAFETY: our page.
        unsafe { (code as *mut u8).copy_from_nonoverlapping([0x0f, 0x01, 0xfc].as_ptr(), 3) };
        Self(code)
    }

    /// The instruction with RAX at `at`, through the loader's emulation as
    /// its handler runs it where the CPU lacks the instruction, the opener
    /// registered as the runtime registers it: whether it was done.
    fn at(&self, at: usize) -> bool {
        self.through(writing, at)
    }

    /// [`Clzero::at`] with `open` registered as the opener (the first
    /// registration of the process stands: each fixture runs alone).
    fn through(&self, open: fn(usize, usize) -> bool, at: usize) -> bool {
        victory_loader::set_console_insn_hooks(victory_loader::ConsoleInsnHooks {
            open_tracked: open,
            rdpru_enabled: || false,
            rdpru_register: |_| None,
        });
        // SAFETY: a zeroed CONTEXT is valid.
        let mut context: CONTEXT = unsafe { std::mem::zeroed() };
        context.ContextFlags = CONTEXT_CONTROL_AMD64 | CONTEXT_INTEGER_AMD64;
        context.Rip = self.0 as u64;
        context.Rax = at as u64;
        // SAFETY: the instruction is in our page; the line is the test's.
        let done = unsafe { victory_loader::emulate_console_instruction(&mut context) };
        assert_eq!(context.Rip, self.0 as u64 + if done { 3 } else { 0 });
        done
    }
}

impl Drop for Clzero {
    fn drop(&mut self) {
        // SAFETY: our page.
        unsafe { VirtualFree(self.0 as *mut _, 0, MEM_RELEASE) };
    }
}

/// A guarded block the kernel may not write: the opener opens it, the
/// write is made, and the block counts as written at the next check.
#[test]
fn a_line_in_a_guarded_block_is_opened_noted_and_zeroed() {
    alone("watch::console_writes_tests::a_line_in_a_guarded_block_is_opened_noted_and_zeroed", false, || {
        let clzero = Clzero::new();
        let (section, view) = view();
        assert!(guard(view, view + BLOCK).is_some());
        assert_eq!(guard(view, view + BLOCK), Some(vec![]), "guarded and clean");
        assert_eq!(shown(view), PAGE_READONLY);
        assert!(clzero.at(view + 0x1234));
        assert!(all(view + 0x1200, 64, 0));
        assert!(all(view, 0x1200, 0xff) && all(view + 0x1240, BLOCK - 0x1240, 0xff));
        assert_eq!(guard(view, view + BLOCK), Some(vec![view / PAGE]), "noted written");
        title_memory_remapped(view, BLOCK);
        // SAFETY: our view.
        unsafe { title_unmap_view(view as *const _) };
        close(section);
    });
}

/// The title gives a guarded view back by its own route, which the guards
/// do not hear of with the ledger off: the opener still holds the block and
/// says so, but cannot open what is not there. Refused, with no fault.
#[test]
fn a_line_in_a_guarded_block_given_back_is_refused() {
    alone("watch::console_writes_tests::a_line_in_a_guarded_block_given_back_is_refused", false, || {
        let clzero = Clzero::new();
        let (section, view) = view();
        assert!(guard(view, view + BLOCK).is_some());
        // SAFETY: our view, given back as the title would.
        assert_ne!(unsafe { title_unmap_view(view as *const _) }, 0);
        close(section);
        assert!(!clzero.at(view + 0x40));
        title_memory_remapped(view, BLOCK);
    });
}

/// As above, and the addresses then hold a fresh page of the title's,
/// read-only: the opener, holding the old block, says so but cannot open
/// it (the block is no longer one allocation). Refused; the new page keeps
/// its bytes and its protection.
#[test]
fn a_line_in_a_fresh_page_where_a_guarded_block_was_is_refused() {
    alone("watch::console_writes_tests::a_line_in_a_fresh_page_where_a_guarded_block_was_is_refused", false, || {
        let clzero = Clzero::new();
        let (section, view) = view();
        assert!(guard(view, view + BLOCK).is_some());
        // SAFETY: our view, given back as the title would.
        assert_ne!(unsafe { title_unmap_view(view as *const _) }, 0);
        close(section);
        let page = fresh_read_only(view, 0x1000);
        assert!(!clzero.at(page + 0x40));
        assert!(all(page, 0x1000, 0x5a), "the new page's bytes");
        assert_eq!(shown(page), PAGE_READONLY, "and its protection");
        title_memory_remapped(view, BLOCK);
        // SAFETY: the allocation above.
        unsafe { title_virtual_free(page as *mut _, 0, MEM_RELEASE) };
    });
}

/// `len` bytes of 0x5a at `at`, allocated by the title's route and made
/// read-only by it.
fn fresh_read_only(at: usize, len: usize) -> usize {
    // SAFETY: the addresses the test just gave back.
    let page = unsafe { title_virtual_alloc(at as *const _, len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
    assert_eq!(page, at, "the same addresses");
    // SAFETY: the allocation just made.
    unsafe { (page as *mut u8).write_bytes(0x5a, len) };
    let mut old = 0;
    // SAFETY: as above.
    assert_ne!(unsafe { title_virtual_protect(page as *const _, len, PAGE_READONLY, &mut old) }, 0);
    page
}

/// With the ledger: the title makes a page under the guard read-only
/// itself. The opener ends the guard and gives each page back what the
/// title asked: that page stays read-only, the opener still says it held
/// the block, and the write is refused with no fault.
#[test]
fn a_line_in_a_page_the_title_made_read_only_under_the_guard_is_refused() {
    alone(
        "watch::console_writes_tests::a_line_in_a_page_the_title_made_read_only_under_the_guard_is_refused",
        true,
        || {
            let clzero = Clzero::new();
            let (section, view) = view();
            assert!(guard(view, view + BLOCK).is_some());
            let mut old = 0;
            // SAFETY: our view, its first native page.
            assert_ne!(unsafe { title_virtual_protect(view as *const _, NATIVE, PAGE_READONLY, &mut old) }, 0);
            assert!(!clzero.at(view + 0x40));
            assert_eq!(shown(view), PAGE_READONLY, "the title's protection");
            assert!(all(view, BLOCK, 0xff));
            title_memory_remapped(view, BLOCK);
            // SAFETY: our view.
            unsafe { title_unmap_view(view as *const _) };
            close(section);
        },
    );
}

/// With the ledger: a guarded view given back and a whole block of the
/// title's allocated there again, read-only. The guards heard of the
/// unmapping, so the opener holds nothing there: refused, the new block
/// unchanged.
#[test]
fn a_line_in_a_fresh_block_where_a_guarded_block_was_is_refused() {
    alone(
        "watch::console_writes_tests::a_line_in_a_fresh_block_where_a_guarded_block_was_is_refused",
        true,
        || {
            let clzero = Clzero::new();
            let (section, view) = view();
            assert!(guard(view, view + BLOCK).is_some());
            // SAFETY: our view, given back as the title would.
            assert_ne!(unsafe { title_unmap_view(view as *const _) }, 0);
            close(section);
            let block = fresh_read_only(view, BLOCK);
            assert!(!clzero.at(block + 0x40));
            assert!(all(block, BLOCK, 0x5a));
            assert_eq!(shown(block), PAGE_READONLY);
            // SAFETY: the allocation above.
            unsafe { title_virtual_free(block as *mut _, 0, MEM_RELEASE) };
        },
    );
}

/// The block a copy guards again right after the opener's next call, before
/// the emulation's checked retry: once, then 0.
static GUARD_AGAIN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The runtime's opener, after which a copy guards [`GUARD_AGAIN`] again.
fn writing_then_guarded_again(at: usize, len: usize) -> bool {
    let held = writing(at, len);
    let block = GUARD_AGAIN.swap(0, std::sync::atomic::Ordering::SeqCst);
    if block != 0 {
        assert!(guard(block, block + BLOCK).is_some());
        assert_eq!(shown(block), PAGE_READONLY, "guarded again");
    }
    held
}

/// A copy guards the block again between the opener and the checked
/// retry: the retry is refused, the opener opens it once more, and the
/// second retry writes the line, noted written.
#[test]
fn a_line_whose_block_is_guarded_again_before_the_retry_is_opened_again() {
    alone("watch::console_writes_tests::a_line_whose_block_is_guarded_again_before_the_retry_is_opened_again", false, || {
        use std::sync::atomic::Ordering::SeqCst;
        let clzero = Clzero::new();
        let (section, view) = view();
        assert!(guard(view, view + BLOCK).is_some());
        GUARD_AGAIN.store(view, SeqCst);
        assert!(clzero.through(writing_then_guarded_again, view + 0x80));
        assert_eq!(GUARD_AGAIN.load(SeqCst), 0, "the copy guarded it again in between");
        assert!(all(view + 0x80, 64, 0));
        assert!(all(view, 0x80, 0xff) && all(view + 0xc0, BLOCK - 0xc0, 0xff));
        assert_eq!(guard(view, view + BLOCK), Some(vec![view / PAGE]), "noted written");
        title_memory_remapped(view, BLOCK);
        // SAFETY: our view.
        unsafe { title_unmap_view(view as *const _) };
        close(section);
    });
}
