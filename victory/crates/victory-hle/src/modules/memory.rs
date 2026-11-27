//! The title's protection calls, with F5's guard ledger on
//! (`VICTORY_GUARD_LEDGER=1`, `victory_gpu::guard_ledger`): answered by the
//! GPU side, which keeps what the title asked under its write guards, so
//! that the title sees its own protection and its frees end the guards
//! (`jobs/2026-11-26-native-memory/f5-design.md`). Off, nothing is served
//! here and these names bind to the host as before.
//!
//! Only the `kernel32`, `kernelbase` and memory API-set routes; dynamic
//! lookups of them reach these too (`kernel::GetProcAddress`). `ntdll`'s
//! and the other routes the note lists stay unmediated.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;

use windows_sys::Win32::System::Memory::MEMORY_BASIC_INFORMATION;

use crate::{HleModule, hle_export};

pub extern "system" fn VirtualProtect(at: *const c_void, len: usize, new: u32, old: *mut u32) -> i32 {
    let _entry = victory_gpu::hle_call();
    // SAFETY: the title's arguments, which the host checks as its own.
    unsafe { victory_gpu::title_virtual_protect(at, len, new, old) }
}

pub extern "system" fn VirtualQuery(at: *const c_void, out: *mut MEMORY_BASIC_INFORMATION, len: usize) -> usize {
    let _entry = victory_gpu::hle_call();
    // SAFETY: as above.
    unsafe { victory_gpu::title_virtual_query(at, out, len) }
}

pub extern "system" fn VirtualAlloc(at: *const c_void, len: usize, kind: u32, protect: u32) -> *mut c_void {
    let _entry = victory_gpu::hle_call();
    // SAFETY: as above.
    unsafe { victory_gpu::title_virtual_alloc(at, len, kind, protect) }
}

pub extern "system" fn VirtualFree(at: *mut c_void, len: usize, kind: u32) -> i32 {
    let _entry = victory_gpu::hle_call();
    // SAFETY: as above.
    unsafe { victory_gpu::title_virtual_free(at, len, kind) }
}

pub extern "system" fn UnmapViewOfFile(at: *const c_void) -> i32 {
    let _entry = victory_gpu::hle_call();
    // SAFETY: as above.
    unsafe { victory_gpu::title_unmap_view(at) }
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "memory",
        dlls: &["kernel32.dll", "kernelbase.dll", "api-ms-win-core-memory-*"],
        exports: if victory_gpu::guard_ledger() {
            vec![
                hle_export!(VirtualProtect),
                hle_export!(VirtualQuery),
                hle_export!(VirtualAlloc),
                hle_export!(VirtualFree),
                hle_export!(UnmapViewOfFile),
            ]
        } else {
            Vec::new()
        },
    }
}

#[cfg(test)]
mod tests {
    /// Off, nothing binds here: the title's imports go to the host as before.
    #[test]
    fn off_the_host_serves_the_protection_calls() {
        if victory_gpu::guard_ledger() {
            return;
        }
        assert!(crate::registry::lookup("kernel32.dll", "VirtualProtect").is_none());
        assert!(crate::registry::lookup("api-ms-win-core-memory-l1-1-0.dll", "VirtualFree").is_none());
    }

    /// On, every route of the first slice binds here: imports from
    /// `kernel32`, `kernelbase` and the memory API set, and a lookup by name
    /// on the host's module. Run in a child with the switch (read once a
    /// process); without it each lookup finds the host's.
    #[test]
    fn on_imports_and_lookups_reach_the_ledger() {
        const CHILD: &str = "VICTORY_LEDGER_TEST_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let name = "modules::memory::tests::on_imports_and_lookups_reach_the_ledger";
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", name, "--test-threads=1"])
                .env(CHILD, "1")
                .env("VICTORY_GUARD_LEDGER", "1")
                .output()
                .unwrap();
            let said = String::from_utf8_lossy(&out.stdout);
            assert!(out.status.success() && said.contains("1 passed"), "{said}{}", String::from_utf8_lossy(&out.stderr));
            return;
        }
        let ours = super::VirtualProtect as *const () as usize;
        for dll in ["kernel32.dll", "KERNELBASE.dll", "api-ms-win-core-memory-l1-1-0.dll"] {
            assert_eq!(crate::registry::lookup(&dll.to_ascii_lowercase(), "VirtualProtect"), Some(ours), "{dll}");
        }
        let kernelbase = crate::modules::kernel::GetModuleHandleA(c"kernelbase.dll".as_ptr().cast());
        assert!(!kernelbase.is_null());
        let found = crate::modules::kernel::GetProcAddress(kernelbase, c"VirtualProtect".as_ptr().cast());
        assert_eq!(found as usize, ours, "a lookup by name reaches the ledger too");
    }
}
