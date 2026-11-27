//! Title threads' affinity and ideal processors, kept off the host
//! processors the GPU's interrupts use (`VICTORY_AVOID_CPUS`).
//!
//! On the console a title's "core" number is a processor number, and the
//! runtime passes these calls through to the host unchanged: there is no
//! console-to-host mapping (see `HostPolicy` and `HleResolver`). With the
//! variable unset these wrappers forward the title's own arguments, exactly
//! as the host pass-through did; with it set they drop the listed
//! processors from the thread's mask and move an ideal processor that is
//! one of them (`victory_gpu::cpu`).

#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::{HleModule, hle_export};

#[cfg(windows)]
mod host {
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::Kernel::PROCESSOR_NUMBER;
    use windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY;

    /// `SetThreadAffinityMask`: the listed processors leave `mask`.
    pub(super) fn set_thread_affinity_mask(thread: HANDLE, mask: usize) -> usize {
        victory_gpu::cpu::set_thread_affinity_mask(thread, mask)
    }

    /// `SetThreadIdealProcessor`: a listed ideal moves to the next
    /// processor in the thread's mask.
    pub(super) fn set_thread_ideal_processor(thread: HANDLE, ideal: u32) -> u32 {
        victory_gpu::cpu::set_thread_ideal_processor(thread, ideal)
    }

    /// `SetThreadIdealProcessorEx`: as above, with a `PROCESSOR_NUMBER`.
    pub(super) fn set_thread_ideal_processor_ex(
        thread: HANDLE,
        ideal: *const PROCESSOR_NUMBER,
        previous: *mut PROCESSOR_NUMBER,
    ) -> i32 {
        victory_gpu::cpu::set_thread_ideal_processor_ex(thread, ideal, previous)
    }

    /// `SetThreadGroupAffinity`: the listed processors leave the mask.
    pub(super) fn set_thread_group_affinity(
        thread: HANDLE,
        affinity: *const GROUP_AFFINITY,
        previous: *mut GROUP_AFFINITY,
    ) -> i32 {
        victory_gpu::cpu::set_thread_group_affinity(thread, affinity, previous)
    }
}

#[cfg(windows)]
pub extern "system" fn SetThreadAffinityMask(
    thread: windows_sys::Win32::Foundation::HANDLE,
    mask: usize,
) -> usize {
    let _entry = victory_gpu::hle_call();
    host::set_thread_affinity_mask(thread, mask)
}

#[cfg(windows)]
pub extern "system" fn SetThreadIdealProcessor(
    thread: windows_sys::Win32::Foundation::HANDLE,
    ideal: u32,
) -> u32 {
    let _entry = victory_gpu::hle_call();
    host::set_thread_ideal_processor(thread, ideal)
}

#[cfg(windows)]
pub extern "system" fn SetThreadIdealProcessorEx(
    thread: windows_sys::Win32::Foundation::HANDLE,
    ideal: *const windows_sys::Win32::System::Kernel::PROCESSOR_NUMBER,
    previous: *mut windows_sys::Win32::System::Kernel::PROCESSOR_NUMBER,
) -> i32 {
    let _entry = victory_gpu::hle_call();
    host::set_thread_ideal_processor_ex(thread, ideal, previous)
}

#[cfg(windows)]
pub extern "system" fn SetThreadGroupAffinity(
    thread: windows_sys::Win32::Foundation::HANDLE,
    affinity: *const windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY,
    previous: *mut windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY,
) -> i32 {
    let _entry = victory_gpu::hle_call();
    host::set_thread_group_affinity(thread, affinity, previous)
}

#[cfg(windows)]
pub(crate) fn module() -> HleModule {
    HleModule {
        name: "thread",
        dlls: &[
            "kernel32.dll",
            "kernelbase.dll",
            "api-ms-win-core-processthreads-*",
        ],
        exports: vec![
            hle_export!(SetThreadAffinityMask),
            hle_export!(SetThreadIdealProcessor),
            hle_export!(SetThreadIdealProcessorEx),
            hle_export!(SetThreadGroupAffinity),
        ],
    }
}

#[cfg(all(test, windows))]
mod tests {
    use victory_loader::Provenance;
    use victory_pe::ImportSymbol;

    use crate::{HleResolver, HostPolicy};

    /// The title's affinity and ideal-processor imports reach this module
    /// (which passes them through when `VICTORY_AVOID_CPUS` is unset), not
    /// the host's own functions directly.
    #[test]
    fn the_thread_processor_imports_resolve_to_the_hle() {
        let resolver = HleResolver::new(HostPolicy::default());
        for name in [
            "SetThreadAffinityMask",
            "SetThreadIdealProcessor",
            "SetThreadIdealProcessorEx",
            "SetThreadGroupAffinity",
        ] {
            let symbol = ImportSymbol::Name {
                hint: 0,
                name: name.into(),
            };
            assert_eq!(
                resolver.classify("api-ms-win-core-processthreads-l1-1-0.dll", &symbol),
                Some(Provenance::Hle),
                "{name}"
            );
        }
    }
}
