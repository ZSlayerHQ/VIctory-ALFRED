//! Tests: an exception nothing handles is described on standard error
//! before the test process ends (its code, the instruction and the module
//! it is in, the address it touched, the thread, and a backtrace), and the
//! process then ends as it would have. A test process that faulted said
//! only "exit 139", without the test or the place (Sol's recheck of the
//! second landing, the parallel test crash). Installed before `main`; no
//! registry setting, no debugger. Compiled for tests only.

use std::io::Write;

use windows_sys::Win32::Foundation::HMODULE;
use windows_sys::Win32::System::Diagnostics::Debug::{EXCEPTION_POINTERS, SetUnhandledExceptionFilter};
use windows_sys::Win32::System::LibraryLoader::{
    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleFileNameW,
    GetModuleHandleExW,
};

/// The file name of the module holding `at`, if any.
fn module_of(at: usize) -> String {
    let mut module: HMODULE = std::ptr::null_mut();
    // SAFETY: a lookup by address; the out pointers are ours.
    let found = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            at as *const u16,
            &mut module,
        )
    } != 0;
    if !found {
        return "no module".to_owned();
    }
    let mut name = [0u16; 260];
    // SAFETY: a module handle and a buffer of the length given.
    let len = unsafe { GetModuleFileNameW(module, name.as_mut_ptr(), name.len() as u32) } as usize;
    let path = String::from_utf16_lossy(&name[..len.min(name.len())]);
    let base = module as usize;
    let file = path.rsplit('\\').next().unwrap_or(&path).to_owned();
    format!("{file}+{:#x}", at.wrapping_sub(base))
}

/// What the memory at `at` is: free, reserved or committed; private,
/// mapped or an image; its allocation and protection.
fn memory_at(at: usize) -> String {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 {
        return "not queried".to_owned();
    }
    let state = match info.State {
        0x1000 => "committed",
        0x2000 => "reserved",
        0x10000 => "free",
        _ => "?",
    };
    let kind = match info.Type {
        0x20000 => " private",
        0x40000 => " mapped",
        0x100_0000 => " image",
        _ => "",
    };
    format!(
        "{state}{kind}, allocation {:#x}, region {:#x}+{:#x}, protection {:#x}",
        info.AllocationBase as usize, info.BaseAddress as usize, info.RegionSize, info.Protect
    )
}

/// Says what faulted, then lets the default handling end the process.
unsafe extern "system" fn unhandled(info: *const EXCEPTION_POINTERS) -> i32 {
    const ACCESS_VIOLATION: i32 = 0xC000_0005_u32 as i32;
    // SAFETY: the system passes a live record.
    let record = unsafe { &*(*info).ExceptionRecord };
    let at = record.ExceptionAddress as usize;
    let touched = (record.ExceptionCode == ACCESS_VIOLATION && record.NumberParameters >= 2).then(|| {
        let how = ["read", "write", "", "", "", "", "", "", "execute"].get(record.ExceptionInformation[0]).copied();
        (how.unwrap_or("?"), record.ExceptionInformation[1])
    });
    let thread = std::thread::current();
    let mut text = format!(
        "\n=== unhandled exception {:#010x} at {at:#x} ({}) on thread {:?}",
        record.ExceptionCode as u32,
        module_of(at),
        thread.name().unwrap_or("unnamed")
    );
    if let Some((how, target)) = touched {
        text += &format!(", a {how} of {target:#x} ({})", memory_at(target));
    }
    text += &format!("\n{}\n===\n", std::backtrace::Backtrace::force_capture());
    // Not `eprint!`: a test's output is captured, and lost with the process.
    let _ = std::io::stderr().write_all(text.as_bytes());
    0 // EXCEPTION_CONTINUE_SEARCH: the process ends as before.
}

unsafe extern "C" fn install() {
    // SAFETY: a plain call; the filter lives as long as the process.
    unsafe { SetUnhandledExceptionFilter(Some(unhandled)) };
}

/// Run by the C runtime before `main`.
#[used]
#[unsafe(link_section = ".CRT$XCU")]
static INSTALL: unsafe extern "C" fn() = install;
