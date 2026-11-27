//! Overrides of shared Windows APIs whose console meaning differs.
//!
//! Only the differences live here; everything else passes through to the
//! host (see `HostPolicy`).
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{CStr, c_void};

use crate::{HleModule, HostPolicy, hle_export, registry, runtime};

/// `GetModuleHandleW(NULL)` must return the title's image, not our exe;
/// title images and our console modules are found by name.
pub extern "system" fn GetModuleHandleW(name: *const u16) -> *mut c_void {
    let _entry = victory_gpu::hle_call();
    if name.is_null() {
        return runtime::main_image().unwrap_or(0) as *mut c_void;
    }
    find_module(&wide(name), false).unwrap_or_else(|| host::get_module_handle_w(name))
}

/// See [`GetModuleHandleW`].
pub extern "system" fn GetModuleHandleA(name: *const u8) -> *mut c_void {
    let _entry = victory_gpu::hle_call();
    if name.is_null() {
        return runtime::main_image().unwrap_or(0) as *mut c_void;
    }
    find_module(&narrow(name), false).unwrap_or_else(|| host::get_module_handle_a(name))
}

/// A NUL-terminated string from the title.
fn narrow(text: *const u8) -> String {
    // SAFETY: the title passes a NUL-terminated string.
    unsafe { CStr::from_ptr(text.cast()) }
        .to_string_lossy()
        .into_owned()
}

/// See [`narrow`].
fn wide(text: *const u16) -> String {
    // SAFETY: the title passes a NUL-terminated string.
    unsafe {
        let len = (0..).take_while(|&i| *text.add(i) != 0).count();
        String::from_utf16_lossy(std::slice::from_raw_parts(text, len))
    }
}

/// A title image's path is its original file, not our cache copy;
/// `NULL` means the title's exe.
pub extern "system" fn GetModuleFileNameW(module: *mut c_void, out: *mut u16, size: u32) -> u32 {
    let _entry = victory_gpu::hle_call();
    let Some(path) = title_path(module) else {
        return host::get_module_file_name_w(module, out, size);
    };
    let wide: Vec<u16> = path.encode_utf16().collect();
    // SAFETY: the title passes a buffer of `size` characters.
    copy_truncated(&wide, unsafe {
        std::slice::from_raw_parts_mut(out, size as usize)
    })
}

/// See [`GetModuleFileNameW`].
pub extern "system" fn GetModuleFileNameA(module: *mut c_void, out: *mut u8, size: u32) -> u32 {
    let _entry = victory_gpu::hle_call();
    let Some(path) = title_path(module) else {
        return host::get_module_file_name_a(module, out, size);
    };
    // SAFETY: the title passes a buffer of `size` bytes.
    copy_truncated(path.as_bytes(), unsafe {
        std::slice::from_raw_parts_mut(out, size as usize)
    })
}

fn title_path(module: *mut c_void) -> Option<String> {
    let base = if module.is_null() {
        runtime::main_image()?
    } else {
        module as usize
    };
    Some(runtime::image_path(base)?.to_string_lossy().into_owned())
}

/// `GetModuleFileName`'s contract: copy with a NUL; if it does not fit,
/// truncate to the buffer, report `ERROR_INSUFFICIENT_BUFFER` and return
/// the buffer size.
fn copy_truncated<T: Copy + Default>(text: &[T], out: &mut [T]) -> u32 {
    let Some(room) = out.len().checked_sub(1) else {
        host::set_last_error(ERROR_INSUFFICIENT_BUFFER);
        return 0;
    };
    let n = text.len().min(room);
    out[..n].copy_from_slice(&text[..n]);
    out[n] = T::default();
    if n < text.len() {
        host::set_last_error(ERROR_INSUFFICIENT_BUFFER);
        return out.len() as u32;
    }
    n as u32
}

const ERROR_INSUFFICIENT_BUFFER: u32 = 122;

/// Console-only DLLs we implement load as a pseudo handle (titles load
/// `xgameruntime.dll` this way), the title's DLLs as the images we map
/// (on first use if nothing imports them); anything else loads on the host.
pub extern "system" fn LoadLibraryExA(
    name: *const u8,
    file: *mut c_void,
    flags: u32,
) -> *mut c_void {
    let _entry = victory_gpu::hle_call();
    if !name.is_null()
        && let Some(handle) = find_module(&narrow(name), runs_code(flags))
    {
        return handle;
    }
    host::load_library_ex_a(name, file, flags)
}

/// See [`LoadLibraryExA`].
pub extern "system" fn LoadLibraryExW(
    name: *const u16,
    file: *mut c_void,
    flags: u32,
) -> *mut c_void {
    let _entry = victory_gpu::hle_call();
    if !name.is_null()
        && let Some(handle) = find_module(&wide(name), runs_code(flags))
    {
        return handle;
    }
    host::load_library_ex_w(name, file, flags)
}

/// False for loads as data or resources (`DONT_RESOLVE_DLL_REFERENCES`,
/// `LOAD_LIBRARY_AS_DATAFILE(_EXCLUSIVE)`, `AS_IMAGE_RESOURCE`): those run
/// no code, so a title DLL we have not mapped can go to the host.
fn runs_code(flags: u32) -> bool {
    flags & (0x1 | 0x2 | 0x20 | 0x40) == 0
}

pub extern "system" fn LoadLibraryA(name: *const u8) -> *mut c_void {
    let _entry = victory_gpu::hle_call();
    LoadLibraryExA(name, std::ptr::null_mut(), 0)
}

pub extern "system" fn LoadLibraryW(name: *const u16) -> *mut c_void {
    let _entry = victory_gpu::hle_call();
    LoadLibraryExW(name, std::ptr::null_mut(), 0)
}

/// Looks up our implementations behind a pseudo handle; real modules go to
/// the host.
pub extern "system" fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void {
    let _entry = victory_gpu::hle_call();
    // `NULL` is the process's exe, which for the title is its own image.
    let module = if module.is_null() {
        runtime::main_image().unwrap_or(0) as *mut c_void
    } else {
        module
    };
    let Some(m) = registry().iter().find(|m| handle(m) == module) else {
        return host_override(module, name).unwrap_or_else(|| host::get_proc_address(module, name));
    };
    // Values below 0x10000 are ordinals; we export by name only.
    if (name as usize) < 0x1_0000 {
        tracing::warn!(
            module = m.name,
            ordinal = name as usize,
            "GetProcAddress by ordinal"
        );
        return std::ptr::null_mut();
    }
    // SAFETY: not an ordinal, so the title passes a NUL-terminated name.
    let name = unsafe { CStr::from_ptr(name.cast()) }.to_string_lossy();
    match m.exports.iter().find(|e| e.name == name) {
        Some(e) => e.address as *mut c_void,
        None => {
            tracing::warn!(module = m.name, %name, "GetProcAddress: not implemented");
            std::ptr::null_mut()
        }
    }
}

/// What a host module's `name` (or ordinal) is bound to at import binding
/// when we override it: a title that looks a function up by name must not
/// get past our overrides (the network guard, path rewriting).
fn host_override(module: *mut c_void, name: *const u8) -> Option<*mut c_void> {
    if module.is_null() || runtime::image_path(module as usize).is_some() {
        return None;
    }
    let symbol = if (name as usize) < 0x1_0000 {
        format!("#{}", name as usize)
    } else {
        narrow(name)
    };
    let mut path = [0u16; 260];
    let len = host::get_module_file_name_w(module, path.as_mut_ptr(), path.len() as u32) as usize;
    let path = String::from_utf16_lossy(&path[..len.min(path.len())]);
    let file = path.rsplit('\\').next()?;
    crate::registry::lookup(file, &symbol).map(|address| address as *mut c_void)
}

/// Our pseudo modules and the title's images stay loaded: `LoadLibrary`
/// took no reference on them, so the host's count must not drop.
pub extern "system" fn FreeLibrary(module: *mut c_void) -> i32 {
    let _entry = victory_gpu::hle_call();
    if registry().iter().any(|m| handle(m) == module)
        || runtime::image_path(module as usize).is_some()
    {
        return 1;
    }
    host::free_library(module)
}

/// Title images hear of thread starts and ends from us, not from the
/// Windows loader (`victory_loader::disable_thread_calls`).
pub extern "system" fn DisableThreadLibraryCalls(module: *mut c_void) -> i32 {
    let _entry = victory_gpu::hle_call();
    host::disable_thread_library_calls(module)
}

/// Title debug output goes to our log (no debugger needs to be attached).
pub extern "system" fn OutputDebugStringA(text: *const u8) {
    let _entry = victory_gpu::hle_call();
    if !text.is_null() {
        let text = narrow(text);
        host::note_title_output(text.trim());
        tracing::info!(target: "title", "{}", text.trim_end());
    }
}

/// See [`OutputDebugStringA`].
pub extern "system" fn OutputDebugStringW(text: *const u16) {
    let _entry = victory_gpu::hle_call();
    if !text.is_null() {
        let text = wide(text);
        host::note_title_output(text.trim());
        tracing::info!(target: "title", "{}", text.trim_end());
    }
}

/// The line logged when the title ends the process, and the code the host
/// then exits with. A function so the text and the code are tested without
/// ending the test process.
fn title_exit(call: &str, code: u32, from: &str) -> (String, i32) {
    (
        format!("title ended the process: call {call}, code {code}, from {from}"),
        code as i32,
    )
}

/// The line for the CRT's abort family (`abort`, `terminate`, the
/// invalid-parameter exit), which ends through the host CRT's `abort`: 3 is
/// that call's own code, but a SIGABRT handler or Windows error reporting
/// can end the process with another (Sol's review, finding 13). The label
/// follows the part the harness reads ("call <call>, code <code>, from ").
fn title_abort(call: &str, from: &str) -> String {
    let (line, _) = title_exit(call, 3, from);
    format!(
        "{line} (3 is abort's own code: a SIGABRT handler or error reporting can end the process with another)"
    )
}

/// The longest a title-chosen exit waits for its line to be logged.
const EXIT_LOG_WAIT: std::time::Duration = std::time::Duration::from_secs(1);

/// Makes the line with `line` and logs it from a thread of its own, and
/// waits for both at most `wait`: a log writer that blocks (the standard
/// error a full pipe, or held by a thread that is) must not keep a
/// title-chosen exit from happening while the title's other threads run
/// (Sol's review, finding 12). Nor must naming the callers, whose module
/// lookup takes the loader's lock (Sol's re-check, finding 12): `line`
/// names them on that thread. Best effort: without a thread, or late, the
/// line is not logged. The standard error is unbuffered, so nothing is
/// flushed. Returns whether it was logged in time.
fn log_bounded(line: impl FnOnce() -> String + Send + 'static, wait: std::time::Duration) -> bool {
    let (logged, wait_for) = std::sync::mpsc::sync_channel::<()>(1);
    let spawned = std::thread::Builder::new().name("victory-exit-log".into()).spawn(move || {
        tracing::warn!("{}", line());
        let _ = logged.send(());
    });
    spawned.is_ok() && wait_for.recv_timeout(wait).is_ok()
}

/// Logs a title-chosen exit ([`log_bounded`]), its callers `from` the
/// return addresses taken on the exiting thread, and returns the code the
/// host exits with.
fn log_title_exit(call: &'static str, code: u32, from: Vec<usize>) -> i32 {
    // The memory census's last lines (`VICTORY_MEMORY_CENSUS=1`; nothing
    // otherwise), bounded as the exit's own line is ([`log_bounded`]).
    #[cfg(feature = "diagnostics")]
    {
        let (done, wait_for) = std::sync::mpsc::sync_channel::<()>(1);
        let census = std::thread::Builder::new().name("victory-census-exit".into()).spawn(move || {
            victory_gpu::memory_census_exit();
            let _ = done.send(());
        });
        if census.is_ok() {
            let _ = wait_for.recv_timeout(EXIT_LOG_WAIT);
        }
    }
    log_bounded(move || title_exit(call, code, &host::named(&from)).0, EXIT_LOG_WAIT);
    code as i32
}

/// [`log_title_exit`] for the abort family ([`title_abort`]).
fn log_title_abort(call: &'static str, from: Vec<usize>) {
    log_bounded(move || title_abort(call, &host::named(&from)), EXIT_LOG_WAIT);
}

/// The CRT's `exit`, through which the title's entry point returns: logs
/// the code and the caller, then exits on the host.
pub extern "C" fn exit(code: i32) -> ! {
    let _entry = victory_gpu::hle_call();
    let from = host::caller_addresses();
    host::exit(log_title_exit("exit", code as u32, from))
}

/// The kernel's `ExitProcess`: logged as [`exit`] is, then the host's own
/// with the title's code (without this export it was bound silently).
/// Not the CRT's `exit`: that runs the exit table on the calling thread
/// while the title's other threads live, where an exit function waiting
/// for a lock the caller or another thread holds hung the process before
/// anything ended it. Windows ends the other threads first and runs no
/// exit table of the program; its loader detaches the DLLs it loaded.
/// shortcut: title images we mapped get no `DLL_PROCESS_DETACH` this way
/// (`victory_loader`'s detach runs from the CRT's exit only): a title that
/// needs its images' detach code at a direct `ExitProcess` would need it
/// called here after the other threads end.
pub extern "system" fn ExitProcess(code: u32) -> ! {
    let _entry = victory_gpu::hle_call();
    let from = host::caller_addresses();
    log_title_exit("ExitProcess", code, from);
    host::exit_process(code)
}

/// The CRT's `abort` (also reached from `terminate`): logged like [`exit`],
/// with the CRT's abort code, labelled as such ([`title_abort`]).
pub extern "C" fn abort() -> ! {
    let _entry = victory_gpu::hle_call();
    log_title_abort("abort", host::caller_addresses());
    host::abort()
}

/// The CRT's `terminate` and the STL's invalid-argument exit: logged, then
/// the host's `abort`.
pub extern "C" fn terminate() -> ! {
    let _entry = victory_gpu::hle_call();
    log_title_abort("terminate", host::caller_addresses());
    host::abort()
}

/// See [`terminate`].
pub extern "C" fn _invalid_parameter_noinfo_noreturn() -> ! {
    let _entry = victory_gpu::hle_call();
    log_title_abort("_invalid_parameter_noinfo_noreturn", host::caller_addresses());
    host::abort()
}

/// The title's unhandled-exception filter. Ours stands in front of it and
/// reports the crash first; the title still gets its own back.
static TITLE_FILTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub extern "system" fn SetUnhandledExceptionFilter(filter: usize) -> usize {
    let _entry = victory_gpu::hle_call();
    host::set_unhandled_exception_filter(report_unhandled);
    TITLE_FILTER.swap(filter, std::sync::atomic::Ordering::AcqRel)
}

extern "system" fn report_unhandled(info: *const c_void) -> i32 {
    host::report_unhandled(info);
    match TITLE_FILTER.load(std::sync::atomic::Ordering::Acquire) {
        0 => 0, // EXCEPTION_CONTINUE_SEARCH
        filter => {
            type Filter = extern "system" fn(*const c_void) -> i32;
            // SAFETY: the title registered this filter.
            let filter: Filter = unsafe { std::mem::transmute(filter) };
            filter(info)
        }
    }
}

/// See [`exit`]; only the current process is logged.
pub extern "system" fn TerminateProcess(process: *mut c_void, code: u32) -> i32 {
    let _entry = victory_gpu::hle_call();
    if process as isize == -1 {
        log_title_exit("TerminateProcess", code, host::caller_addresses());
        // The run ends here as it does at `ExitProcess`: the live segment under way is logged.
        victory_gpu::live_finish();
    }
    host::terminate_process(process, code)
}

/// The pseudo handle of one of our modules: its address, never a real image.
fn handle(m: &HleModule) -> *mut c_void {
    m as *const HleModule as *mut c_void
}

/// The title's images and our console modules by name; `None` leaves the
/// name to the host, which only DLLs `HostPolicy` passes through get (as at
/// import binding); others fail. With `load`, a title DLL not mapped yet is
/// mapped now.
fn find_module(path: &str, load: bool) -> Option<*mut c_void> {
    let file = path.rsplit(['\\', '/']).next().unwrap_or(path);
    // As the Windows loader does, a name without extension means a DLL.
    let file = if file.contains('.') {
        file.to_owned()
    } else {
        format!("{file}.dll")
    };
    // Loads go through the title DLL loader first: it waits for a DLL that
    // another thread is still initialising, as the Windows loader lock does.
    if load {
        match runtime::load_title_dll(&file) {
            Ok(Some(base)) => return Some(base as *mut c_void),
            Ok(None) => {}
            // Not the host: it would load the unpatched DLL from the title's folder.
            Err(error) => {
                tracing::error!(dll = file, %error, "mapping a title DLL failed");
                host::set_last_error(ERROR_MOD_NOT_FOUND);
                return Some(std::ptr::null_mut());
            }
        }
    }
    if let Some(base) = runtime::image_named(&file) {
        return Some(base as *mut c_void);
    }
    // DLLs the host has too are real host modules, whatever `--strict` says.
    let host_dll = HostPolicy::default().allows(&file);
    if !host_dll && let Some(m) = registry().iter().find(|m| m.serves(&file)) {
        if load {
            tracing::info!(dll = file, "loaded as a pseudo module");
        }
        return Some(handle(m));
    }
    if host_dll {
        return None;
    }
    // Any other host DLL (wininet, urlmon, ...) is beyond what we guard, and
    // the console does not have it.
    refuse(&file);
    host::set_last_error(ERROR_MOD_NOT_FOUND);
    Some(std::ptr::null_mut())
}

/// Logs a DLL name the title is refused, once.
fn refuse(file: &str) {
    static REFUSED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let file = file.to_ascii_lowercase();
    let mut refused = REFUSED.lock().unwrap_or_else(|e| e.into_inner());
    if !refused.contains(&file) {
        tracing::warn!(dll = file, "a host DLL outside the host policy: refused");
        refused.push(file);
    }
}

const ERROR_MOD_NOT_FOUND: u32 = 126;

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "kernel",
        dlls: &[
            "kernel32.dll",
            "kernelbase.dll",
            "api-ms-win-core-libraryloader-*",
            "api-ms-win-core-processthreads-*",
            "api-ms-win-core-debug-*",
            "api-ms-win-core-errorhandling-*",
            "api-ms-win-crt-runtime-*",
        ],
        exports: vec![
            hle_export!(GetModuleHandleW),
            hle_export!(GetModuleHandleA),
            hle_export!(GetModuleFileNameW),
            hle_export!(GetModuleFileNameA),
            hle_export!(LoadLibraryExA),
            hle_export!(LoadLibraryExW),
            hle_export!(LoadLibraryA),
            hle_export!(LoadLibraryW),
            hle_export!(GetProcAddress),
            hle_export!(FreeLibrary),
            hle_export!(DisableThreadLibraryCalls),
            hle_export!(OutputDebugStringA),
            hle_export!(OutputDebugStringW),
            hle_export!(exit),
            hle_export!(ExitProcess),
            hle_export!(abort),
            hle_export!(terminate),
            hle_export!(_invalid_parameter_noinfo_noreturn),
            hle_export!(TerminateProcess),
            hle_export!(SetUnhandledExceptionFilter),
        ],
    }
}

/// The return addresses above the caller, as `module+rva`, for logs.
#[cfg(feature = "diagnostics")]
pub(crate) fn callers() -> String {
    host::callers()
}

#[cfg(windows)]
mod host {
    use std::ffi::c_void;

    use windows_sys::Win32::Foundation::{FreeLibrary, SetLastError};
    use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
    use windows_sys::Win32::System::LibraryLoader::{
        DisableThreadLibraryCalls, GetModuleFileNameA, GetModuleFileNameW, GetModuleHandleA,
        GetModuleHandleW, GetProcAddress, LoadLibraryExA, LoadLibraryExW,
    };
    use windows_sys::Win32::System::Threading::TerminateProcess;

    unsafe extern "C" {
        #[link_name = "exit"]
        fn crt_exit(code: i32) -> !;
        #[link_name = "abort"]
        fn crt_abort() -> !;
    }

    // SAFETY (every function): forwarding the title's own arguments to the
    // host API of the same name.

    pub(super) fn get_module_handle_w(name: *const u16) -> *mut c_void {
        // SAFETY: see above.
        unsafe { GetModuleHandleW(name) }
    }

    pub(super) fn get_module_handle_a(name: *const u8) -> *mut c_void {
        // SAFETY: see above.
        unsafe { GetModuleHandleA(name) }
    }

    pub(super) fn get_module_file_name_w(module: *mut c_void, out: *mut u16, size: u32) -> u32 {
        // SAFETY: see above.
        unsafe { GetModuleFileNameW(module, out, size) }
    }

    pub(super) fn get_module_file_name_a(module: *mut c_void, out: *mut u8, size: u32) -> u32 {
        // SAFETY: see above.
        unsafe { GetModuleFileNameA(module, out, size) }
    }

    pub(super) fn set_last_error(code: u32) {
        // SAFETY: see above.
        unsafe { SetLastError(code) }
    }

    pub(super) fn load_library_ex_a(name: *const u8, file: *mut c_void, flags: u32) -> *mut c_void {
        // SAFETY: see above.
        unsafe { LoadLibraryExA(name, file, flags) }
    }

    pub(super) fn load_library_ex_w(
        name: *const u16,
        file: *mut c_void,
        flags: u32,
    ) -> *mut c_void {
        // SAFETY: see above.
        unsafe { LoadLibraryExW(name, file, flags) }
    }

    pub(super) fn get_proc_address(module: *mut c_void, name: *const u8) -> *mut c_void {
        // SAFETY: see above.
        unsafe { GetProcAddress(module, name) }.map_or(std::ptr::null_mut(), |f| f as *mut c_void)
    }

    pub(super) fn free_library(module: *mut c_void) -> i32 {
        // SAFETY: see above.
        unsafe { FreeLibrary(module) }
    }

    pub(super) fn disable_thread_library_calls(module: *mut c_void) -> i32 {
        match victory_loader::disable_thread_calls(module as usize) {
            Some(done) => i32::from(done),
            // SAFETY: see above.
            None => unsafe { DisableThreadLibraryCalls(module) },
        }
    }

    pub(super) fn exit(code: i32) -> ! {
        victory_gpu::live_finish();
        // SAFETY: see above.
        unsafe { crt_exit(code) }
    }

    pub(super) fn exit_process(code: u32) -> ! {
        victory_gpu::live_finish();
        // SAFETY: see above.
        unsafe { windows_sys::Win32::System::Threading::ExitProcess(code) }
    }

    pub(super) fn abort() -> ! {
        // SAFETY: see above.
        unsafe { crt_abort() }
    }

    pub(super) fn terminate_process(process: *mut c_void, code: u32) -> i32 {
        // SAFETY: see above.
        unsafe { TerminateProcess(process, code) }
    }

    pub(super) fn set_unhandled_exception_filter(filter: extern "system" fn(*const c_void) -> i32) {
        // SAFETY: our filter takes the exception pointers as an opaque pointer.
        unsafe {
            windows_sys::Win32::System::Diagnostics::Debug::SetUnhandledExceptionFilter(
                std::mem::transmute::<
                    extern "system" fn(*const c_void) -> i32,
                    windows_sys::Win32::System::Diagnostics::Debug::LPTOP_LEVEL_EXCEPTION_FILTER,
                >(filter),
            )
        };
    }

    pub(super) fn report_unhandled(info: *const c_void) {
        // SAFETY: the system's exception pointers, passed to the filter.
        unsafe { victory_loader::report_unhandled(info.cast()) };
    }

    pub(super) fn note_title_output(line: &str) {
        victory_loader::note_title_output(line);
    }

    /// Our caller's callers, as `module+rva` (diagnostics' logs).
    #[cfg(feature = "diagnostics")]
    pub(super) fn callers() -> String {
        let mut frames = [std::ptr::null_mut(); 12];
        // SAFETY: fills at most `frames.len()` return addresses.
        let n = unsafe {
            RtlCaptureStackBackTrace(
                2,
                frames.len() as u32,
                frames.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        let names: Vec<_> = frames[..usize::from(n)]
            .iter()
            .map(|&f| victory_loader::locate(f as usize))
            .collect();
        names.join(" < ")
    }

    /// Our caller's callers' return addresses, as [`callers`] finds them,
    /// unnamed: taking them takes no lock, so an exit takes them on its
    /// own thread and names them on its logging thread ([`named`]).
    #[inline(never)]
    pub(super) fn caller_addresses() -> Vec<usize> {
        let mut frames = [std::ptr::null_mut(); 12];
        // SAFETY: fills at most `frames.len()` return addresses.
        let n = unsafe {
            RtlCaptureStackBackTrace(
                2,
                frames.len() as u32,
                frames.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        frames[..usize::from(n)].iter().map(|&f| f as usize).collect()
    }

    /// `addresses` as `module+rva`, as [`callers`] names them: the module
    /// lookup takes the loader's lock.
    pub(super) fn named(addresses: &[usize]) -> String {
        let names: Vec<_> = addresses.iter().map(|&f| victory_loader::locate(f)).collect();
        names.join(" < ")
    }
}

#[cfg(not(windows))]
mod host {
    use std::ffi::c_void;

    pub(super) fn get_module_handle_w(_: *const u16) -> *mut c_void {
        std::ptr::null_mut()
    }

    pub(super) fn get_module_handle_a(_: *const u8) -> *mut c_void {
        std::ptr::null_mut()
    }

    pub(super) fn load_library_ex_a(_: *const u8, _: *mut c_void, _: u32) -> *mut c_void {
        std::ptr::null_mut()
    }

    pub(super) fn get_module_file_name_w(_: *mut c_void, _: *mut u16, _: u32) -> u32 {
        0
    }

    pub(super) fn get_module_file_name_a(_: *mut c_void, _: *mut u8, _: u32) -> u32 {
        0
    }

    pub(super) fn set_last_error(_: u32) {}

    pub(super) fn load_library_ex_w(_: *const u16, _: *mut c_void, _: u32) -> *mut c_void {
        std::ptr::null_mut()
    }

    pub(super) fn get_proc_address(_: *mut c_void, _: *const u8) -> *mut c_void {
        std::ptr::null_mut()
    }

    pub(super) fn free_library(_: *mut c_void) -> i32 {
        0
    }

    pub(super) fn disable_thread_library_calls(_: *mut c_void) -> i32 {
        0
    }

    pub(super) fn exit(code: i32) -> ! {
        victory_gpu::live_finish();
        std::process::exit(code)
    }

    pub(super) fn exit_process(code: u32) -> ! {
        victory_gpu::live_finish();
        std::process::exit(code as i32)
    }

    pub(super) fn abort() -> ! {
        std::process::abort()
    }

    pub(super) fn terminate_process(_: *mut c_void, code: u32) -> i32 {
        std::process::exit(code as i32)
    }

    pub(super) fn set_unhandled_exception_filter(_: extern "system" fn(*const c_void) -> i32) {}

    pub(super) fn report_unhandled(_: *const c_void) {}

    pub(super) fn note_title_output(_: &str) {}

    #[cfg(feature = "diagnostics")]
    pub(super) fn callers() -> String {
        String::new()
    }

    pub(super) fn caller_addresses() -> Vec<usize> {
        Vec::new()
    }

    pub(super) fn named(_: &[usize]) -> String {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A title-chosen exit's line names the call, the code and the caller,
    /// and the code the host exits with. The text is the one the harness
    /// matches, so it is fixed.
    #[test]
    fn a_title_exit_line_names_the_call_code_and_caller() {
        assert_eq!(
            title_exit("exit", 127, "game.exe+0x1234"),
            (
                "title ended the process: call exit, code 127, from game.exe+0x1234".to_owned(),
                127
            )
        );
        // An exit code that arrives as a negative i32 keeps its bits.
        assert_eq!(title_exit("exit", (-1i32) as u32, "").1, -1);
    }

    /// The abort family's line says its code is abort's own, which a
    /// SIGABRT handler or error reporting may replace, after the part the
    /// harness reads ("call <call>, code <code>, from "): Sol's review,
    /// finding 13.
    #[test]
    fn an_abort_line_says_its_code_is_aborts_own() {
        assert_eq!(
            title_abort("terminate", "game.exe+0x1234"),
            "title ended the process: call terminate, code 3, from game.exe+0x1234 \
             (3 is abort's own code: a SIGABRT handler or error reporting can end the process with another)"
        );
    }

    /// The two tests below run a child process of this test binary.
    #[cfg(windows)]
    const CHILD: &str = "VICTORY_TEST_EXIT_PROCESS_CHILD";
    #[cfg(windows)]
    const CODE: u32 = 0x8765_4321;

    /// The child's log: each event's message on the standard error, through
    /// its lock, as the runtime's log writes it.
    #[cfg(windows)]
    struct ToStderr;

    #[cfg(windows)]
    impl tracing::Subscriber for ToStderr {
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
        fn event(&self, event: &tracing::Event<'_>) {
            struct Message(String);
            impl tracing::field::Visit for Message {
                fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                    if field.name() == "message" {
                        self.0 = format!("{value:?}");
                    }
                }
            }
            let mut message = Message(String::new());
            event.record(&mut message);
            eprintln!("{}", message.0);
        }
        fn enter(&self, _: &tracing::span::Id) {}
        fn exit(&self, _: &tracing::span::Id) {}
    }

    /// The child: the title's state at its `ExitProcess`. A lock the caller
    /// holds, which an exit function registered with the CRT wants; another
    /// thread alive; the log on the standard error. `blocked`: another
    /// thread holds the standard error's lock, as a log writer blocked on a
    /// full pipe does.
    #[cfg(windows)]
    fn exit_process_child(blocked: bool) -> ! {
        static HELD: std::sync::Mutex<()> = std::sync::Mutex::new(());
        extern "C" fn wants_the_lock() {
            drop(HELD.lock());
        }
        unsafe extern "C" {
            fn atexit(f: extern "C" fn()) -> i32;
        }
        let _ = tracing::subscriber::set_global_default(ToStderr);
        let _held = HELD.lock().unwrap();
        // SAFETY: registers a function with the CRT's exit table.
        assert_eq!(unsafe { atexit(wants_the_lock) }, 0);
        // Another thread alive, as the title's are.
        drop(std::thread::spawn(|| {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        }));
        if blocked {
            let (holding, held) = std::sync::mpsc::channel();
            drop(std::thread::spawn(move || {
                let _stderr = std::io::stderr().lock();
                let _ = holding.send(());
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                }
            }));
            held.recv().unwrap();
        }
        ExitProcess(CODE)
    }

    /// Runs this binary's test `name` as the child, with `scenario`: its
    /// exit code and standard error, waited for at most 30 seconds.
    #[cfg(windows)]
    fn run_exit_child(name: &str, scenario: &str) -> (Option<u32>, String) {
        use std::io::Read;
        let exe = std::env::current_exe().unwrap();
        let mut child = std::process::Command::new(exe)
            .args(["--exact", name, "--nocapture"])
            .env(CHILD, scenario)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut stderr = child.stderr.take().unwrap();
        let read = std::thread::spawn(move || {
            let mut text = String::new();
            let _ = stderr.read_to_string(&mut text);
            text
        });
        let started = std::time::Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if started.elapsed() > std::time::Duration::from_secs(30) {
                let _ = child.kill();
                panic!("the title's ExitProcess did not end the process in 30 s ({scenario})");
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        };
        (status.code().map(|c| c as u32), read.join().unwrap())
    }

    /// The title's `ExitProcess` ends the process as Windows does: the
    /// CRT's exit table does not run on the calling thread while the
    /// title's other threads live, and the process ends with the title's
    /// code. Through the CRT's `exit` instead, an exit function that wants
    /// a lock the caller holds hung the process before anything else ran
    /// (Sol, integration review). The line is logged first, and the test
    /// reads it (Sol's review of the first landing, finding 12: the child's
    /// standard error went to null).
    #[cfg(windows)]
    #[test]
    fn a_title_exit_process_ends_as_windows_does_with_its_code() {
        if std::env::var_os(CHILD).is_some() {
            exit_process_child(false);
        }
        let name = "modules::kernel::tests::a_title_exit_process_ends_as_windows_does_with_its_code";
        let (code, stderr) = run_exit_child(name, "plain");
        assert_eq!(code, Some(CODE), "the title's code");
        let line = format!("title ended the process: call ExitProcess, code {CODE}, from ");
        assert!(stderr.contains(&line), "the line, in {stderr:?}");
        // The callers are named, on the logging thread, from the addresses
        // the exiting thread took (finding 12): this binary's frames.
        let exe = std::env::current_exe().unwrap();
        let exe = exe.file_name().unwrap().to_string_lossy();
        let named = format!("{line}{exe}+0x");
        assert!(stderr.contains(&named), "{named:?} in {stderr:?}");
    }

    /// A log that cannot be written does not keep the process from ending:
    /// the line is best effort, waited for a bounded time (finding 12).
    /// Here another thread holds the standard error's lock for good.
    #[cfg(windows)]
    #[test]
    fn a_title_exit_process_ends_while_the_log_is_blocked() {
        if std::env::var_os(CHILD).is_some() {
            exit_process_child(true);
        }
        let name = "modules::kernel::tests::a_title_exit_process_ends_while_the_log_is_blocked";
        let started = std::time::Instant::now();
        let (code, _) = run_exit_child(name, "blocked");
        assert_eq!(code, Some(CODE), "the title's code");
        assert!(started.elapsed() < std::time::Duration::from_secs(20), "{:?}", started.elapsed());
    }

    #[test]
    fn title_images_report_their_original_path() {
        let base = 0x7000_0000_0000;
        runtime::set_image_path(base, std::path::Path::new(r"C:\title\game.exe"));
        let module = base as *mut c_void;
        let mut buf = [0xFFFFu16; 32];
        let n = GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32);
        assert_eq!(
            String::from_utf16_lossy(&buf[..n as usize]),
            r"C:\title\game.exe"
        );
        assert_eq!(buf[n as usize], 0);

        // Too small: truncated, NUL-terminated, the buffer size returned.
        let mut small = [0u8; 4];
        assert_eq!(GetModuleFileNameA(module, small.as_mut_ptr(), 4), 4);
        assert_eq!(&small, &[b'C', b':', b'\\', 0]);
    }

    fn w(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(Some(0)).collect()
    }

    #[cfg(windows)]
    #[test]
    fn host_lookups_by_name_get_our_overrides() {
        let kernel32 = GetModuleHandleW(w("kernel32.dll").as_ptr());
        assert!(!kernel32.is_null());
        let ours = crate::registry::lookup("kernel32.dll", "CreateFileW").unwrap();
        assert_eq!(
            GetProcAddress(kernel32, c"CreateFileW".as_ptr().cast()) as usize,
            ours
        );
        // A plain name stays the host's.
        let host = host::get_proc_address(kernel32, c"GetTickCount".as_ptr().cast());
        assert!(!host.is_null());
        assert_eq!(
            GetProcAddress(kernel32, c"GetTickCount".as_ptr().cast()),
            host
        );
        // Host DLLs outside the host policy do not exist for the title.
        assert!(LoadLibraryW(w("wininet.dll").as_ptr()).is_null());
        assert!(LoadLibraryA(c"C:\\Windows\\System32\\urlmon".as_ptr().cast()).is_null());
        assert!(GetModuleHandleW(w("WININET.DLL").as_ptr()).is_null());
        assert_eq!(
            // SAFETY: reads this thread's last error.
            unsafe { windows_sys::Win32::Foundation::GetLastError() },
            ERROR_MOD_NOT_FOUND
        );
        assert!(!LoadLibraryW(w("kernel32").as_ptr()).is_null());
        // By ordinal too (ws2_32 #4 is `connect`).
        let ws2 = LoadLibraryW(w("ws2_32.dll").as_ptr());
        assert_eq!(
            GetProcAddress(ws2, 4 as *const u8) as usize,
            crate::registry::lookup("ws2_32.dll", "connect").unwrap()
        );
    }

    #[test]
    fn title_dlls_and_pseudo_modules_are_found_by_name() {
        let base = 0x7100_0000_0000usize;
        runtime::set_image_path(base, std::path::Path::new(r"C:\title\Mapped.dll"));
        runtime::set_title_dll_loader(Box::new(|file| match file {
            "late.dll" => Ok(Some(0x7200_0000_0000)),
            "broken.dll" => Err("bad image".into()),
            _ => Ok(None),
        }));
        let mapped = base as *mut c_void;
        assert_eq!(GetModuleHandleW(w("MAPPED").as_ptr()), mapped);
        assert_eq!(GetModuleHandleA(c"sub\\mapped.dll".as_ptr().cast()), mapped);
        assert_eq!(LoadLibraryW(w("mapped.dll").as_ptr()), mapped);
        assert_eq!(FreeLibrary(mapped), 1, "title images stay loaded");

        // Not mapped yet: `LoadLibrary` maps it, `GetModuleHandle` does not.
        assert!(GetModuleHandleW(w("late.dll").as_ptr()).is_null());
        assert_eq!(
            LoadLibraryA(c"late".as_ptr().cast()) as usize,
            0x7200_0000_0000
        );
        assert!(LoadLibraryA(c"broken.dll".as_ptr().cast()).is_null());
        // As data it runs no code, so the host loads it (and finds nothing).
        let data = LoadLibraryExW(w("late.dll").as_ptr(), std::ptr::null_mut(), 0x2);
        assert_ne!(data as usize, 0x7200_0000_0000);

        let runtime = LoadLibraryA(c"xgameruntime.dll".as_ptr().cast());
        assert!(!runtime.is_null());
        assert_eq!(GetModuleHandleW(w("XGameRuntime.dll").as_ptr()), runtime);
    }
}
