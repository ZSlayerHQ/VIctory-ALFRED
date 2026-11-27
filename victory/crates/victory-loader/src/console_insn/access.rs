//! Title memory touched from the exception handler. The kernel makes each
//! access (`ReadProcessMemory`, `NtWriteVirtualMemory`) and checks it as it
//! makes it, so a page that is not there, may not be read or written, or
//! changes meanwhile, gives a refusal instead of a fault in the handler,
//! and no page query (which costs milliseconds in a big resident region)
//! is needed before.

use std::ffi::c_void;
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// How a page took an access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Access {
    Granted,
    /// The runtime's write tracking held the page read-only; its opener
    /// opened it, noting the write, and the checked access made again then
    /// went through.
    Reopened,
    Denied,
}

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    /// The kernel's write into a process's memory, `(process, address,
    /// bytes, length, written out)`: an NTSTATUS, negative where a page of
    /// the range is not there or not writable. It changes no protection,
    /// where `WriteProcessMemory` makes a read-only page writable to write
    /// it, which would pass the runtime's write tracking by.
    fn NtWriteVirtualMemory(
        process: *mut c_void,
        base: *mut c_void,
        buffer: *const c_void,
        size: usize,
        written: *mut usize,
    ) -> i32;
}

/// `bytes` written at `at` by the kernel, all of them, or none taken.
fn kernel_write(at: usize, bytes: &[u8]) -> bool {
    let mut written = 0;
    #[cfg(test)]
    crate::crash::count_kernel_call(|n| n.writes += 1);
    // SAFETY: `bytes` is readable for its length; the kernel checks the
    // destination and writes nothing it may not.
    let status = unsafe {
        NtWriteVirtualMemory(GetCurrentProcess(), at as *mut c_void, bytes.as_ptr().cast(), bytes.len(), &mut written)
    };
    status >= 0 && written == bytes.len()
}

/// Zeroes the 64-byte line at `line` (64-byte aligned, so in one page);
/// `open_tracked` is the runtime's opener ([`super::ConsoleInsnHooks`]).
pub(super) fn zero_line(line: usize, open_tracked: fn(usize, usize) -> bool) -> Access {
    const ZEROS: [u8; 64] = [0; 64];
    if kernel_write(line, &ZEROS) {
        return Access::Granted;
    }
    // Refused: perhaps a page the runtime's write tracking holds read-only,
    // which a title write would open through its write-fault handler. The
    // same opener, which notes the write; then the same checked write
    // again. The opener's true says only that the tracking held the page:
    // the program may have changed its protection, freed it or mapped
    // something else there since, so the page may still refuse the write,
    // and the handler never stores into it itself. A page the tracking does
    // not hold stays refused: a guard page among them, whose one-time guard
    // the kernel's attempt used up as a store would (the title sees the
    // refused instruction reported instead of the guard's exception).
    //
    // Two openings, not a loop. A copy may guard the block again between
    // the opener and the write (a title store would then fault into the
    // guard and be opened again): the second opening covers that. For it to
    // fail as well, a second guard pass would have to land in a second
    // window of microseconds; a page still refused after two openings is
    // refused for another reason, which a third would not change. A loop,
    // run on the title's thread inside its exception, would have no bound
    // where the opener keeps saying it held a page it cannot open.
    for _ in 0..2 {
        if !open_tracked(line, ZEROS.len()) {
            break;
        }
        if kernel_write(line, &ZEROS) {
            return Access::Reopened;
        }
    }
    Access::Denied
}

/// Whether the byte at `at` can be read.
pub(super) fn byte_readable(at: u64) -> bool {
    usize::try_from(at).is_ok_and(|at| crate::crash::page_bytes(at, &mut [0]) == 1)
}

/// Whether the page holding `at` can be read: one byte of it.
pub(super) fn page_readable(at: u64) -> bool {
    byte_readable(at & !0xfff)
}
