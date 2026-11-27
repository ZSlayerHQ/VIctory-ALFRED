//! `VICTORY_AVOID_CPUS`: keep title threads off the host logical processors
//! that take the GPU's interrupts.
//!
//! The CPU trace of 2026-11-26 (`jobs/2026-11-26-cbj-review/trace-findings.md`)
//! found the thread that submits the title's command lists with ideal
//! processor 1, and processor 1 taking nearly all of the card's interrupt
//! and DPC work: tens of milliseconds a frame landed on that thread as kernel time.
//! With this variable set to a comma list of host processor numbers, every
//! affinity or ideal-processor call a title thread makes drops those
//! processors from the thread's mask (never to an empty one), and an ideal
//! processor that is one of them moves to the next processor still in the
//! mask. The first thread to submit work is treated the same way once,
//! where `crate::host::Host::submit` begins.
//!
//! Unset (or empty), nothing changes and no extra call is made.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// The processor numbers in a `VICTORY_AVOID_CPUS` value: `"1, 3,5"`.
/// Entries that are not numbers, or that no mask can name on this host, are
/// left out; the result is sorted and without repeats.
pub fn parse_list(text: &str) -> Vec<u32> {
    let mut list: Vec<u32> = text
        .split(',')
        .filter_map(|part| part.trim().parse::<u32>().ok())
        .filter(|&cpu| u64::from(cpu) < u64::from(usize::BITS))
        .collect();
    list.sort_unstable();
    list.dedup();
    list
}

/// The host processors to keep title threads off, read once from
/// `VICTORY_AVOID_CPUS`. Empty when it is unset.
pub fn avoided() -> &'static [u32] {
    static LIST: OnceLock<Vec<u32>> = OnceLock::new();
    LIST.get_or_init(|| {
        std::env::var("VICTORY_AVOID_CPUS").map_or_else(|_| Vec::new(), |text| parse_list(&text))
    })
}

/// `mask` without the listed processors.
pub fn remaining(mask: usize, list: &[u32]) -> usize {
    list.iter().fold(mask, |mask, &cpu| {
        let bit = 1usize.checked_shl(cpu).unwrap_or(0);
        mask & !bit
    })
}

/// `mask` without the listed processors, when that leaves a different,
/// non-empty mask. `None` for no change, and for a mask the list would
/// empty (the caller then leaves the mask alone and warns).
pub fn without(mask: usize, list: &[u32]) -> Option<usize> {
    let kept = remaining(mask, list);
    (kept != mask && kept != 0).then_some(kept)
}

/// Whether the listed processors are every processor of `mask`.
pub fn would_empty(mask: usize, list: &[u32]) -> bool {
    mask != 0 && remaining(mask, list) == 0
}

/// `ideal` when it is not one of the listed processors; otherwise the next
/// unlisted processor in `mask` after it, wrapping around, or `ideal` when
/// the mask has none.
pub fn ideal_after(ideal: u32, mask: usize, list: &[u32]) -> u32 {
    if !list.contains(&ideal) {
        return ideal;
    }
    let bits = usize::BITS as usize;
    (1..=bits)
        .map(|step| (ideal as usize + step) % bits)
        .find(|&cpu| mask & (1usize << cpu) != 0 && !list.contains(&(cpu as u32)))
        .map_or(ideal, |cpu| cpu as u32)
}

/// A mask as processor numbers, for the log line: `"1,3"`; `"-"` when it is
/// empty.
pub fn numbers(mask: usize) -> String {
    let cpus: Vec<String> = (0..usize::BITS as usize)
        .filter(|&cpu| mask & (1usize << cpu) != 0)
        .map(|cpu| cpu.to_string())
        .collect();
    if cpus.is_empty() {
        "-".to_owned()
    } else {
        cpus.join(",")
    }
}

/// Warns, once per process, that a mask every processor of which is listed
/// was left as it was.
pub fn warn_would_empty() {
    static SAID: AtomicBool = AtomicBool::new(false);
    if !SAID.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            "VICTORY_AVOID_CPUS names every processor of a thread's mask: the mask is left as it was"
        );
    }
}

/// The filter to a thread's mask: `asked` without the listed processors,
/// or `asked` unchanged where that would empty it (with the one warning).
pub fn filtered_mask(asked: usize, list: &[u32]) -> usize {
    match without(asked, list) {
        Some(mask) => mask,
        None => {
            if would_empty(asked, list) {
                warn_would_empty();
            }
            asked
        }
    }
}

#[cfg(windows)]
pub use host::{
    restrict_current_thread, restrict_thread, set_thread_affinity_mask, set_thread_group_affinity,
    set_thread_ideal_processor, set_thread_ideal_processor_ex,
};

#[cfg(windows)]
mod host {
    use std::ffi::c_void;

    use windows_sys::Win32::Foundation::{HANDLE, LocalFree};
    use windows_sys::Win32::System::Kernel::PROCESSOR_NUMBER;
    use windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY;
    use windows_sys::Win32::System::Threading::{
        GetCurrentThread, GetThreadDescription, GetThreadGroupAffinity, GetThreadId,
        GetThreadIdealProcessorEx, SetThreadAffinityMask, SetThreadGroupAffinity,
        SetThreadIdealProcessor, SetThreadIdealProcessorEx,
    };

    use super::{avoided, filtered_mask, ideal_after, numbers};

    /// The thread's group affinity, as the system reports it now.
    fn affinity_of(thread: HANDLE) -> Option<GROUP_AFFINITY> {
        let mut affinity = GROUP_AFFINITY::default();
        // SAFETY: a plain query into a struct of the system's layout.
        (unsafe { GetThreadGroupAffinity(thread, &mut affinity) } != 0).then_some(affinity)
    }

    fn ideal_of(thread: HANDLE) -> Option<PROCESSOR_NUMBER> {
        let mut ideal = PROCESSOR_NUMBER {
            Group: 0,
            Number: 0,
            Reserved: 0,
        };
        // SAFETY: as above.
        (unsafe { GetThreadIdealProcessorEx(thread, &mut ideal) } != 0).then_some(ideal)
    }

    /// Sets the thread's ideal processor in `group`; whether it took.
    fn set_ideal(thread: HANDLE, group: u16, ideal: u32) -> bool {
        let processor = PROCESSOR_NUMBER {
            Group: group,
            Number: u8::try_from(ideal).unwrap_or(u8::MAX),
            Reserved: 0,
        };
        // SAFETY: a struct of the system's layout, alive for the call.
        (unsafe { SetThreadIdealProcessorEx(thread, &processor, std::ptr::null_mut()) }) != 0
    }

    /// The thread's name as its owner set it, if any.
    fn name_of(thread: HANDLE) -> Option<String> {
        let mut text: *mut u16 = std::ptr::null_mut();
        // SAFETY: the system allocates the string; it is freed below and
        // read only while allocated.
        unsafe {
            if GetThreadDescription(thread, &mut text) < 0 || text.is_null() {
                return None;
            }
            let len = (0..).take_while(|&i| *text.add(i) != 0).count();
            let name = String::from_utf16_lossy(std::slice::from_raw_parts(text, len));
            LocalFree(text.cast::<c_void>());
            (!name.is_empty()).then_some(name)
        }
    }

    /// One line for a thread whose mask or ideal processor this changed.
    fn log_change(
        thread: HANDLE,
        role: &str,
        mask_before: usize,
        mask_after: usize,
        ideal_before: Option<u32>,
        ideal_after: Option<u32>,
    ) {
        let thread_name = name_of(thread).unwrap_or_else(|| role.to_owned());
        // SAFETY: a plain query of a live thread handle.
        let id = unsafe { GetThreadId(thread) };
        let ideal = |ideal: Option<u32>| ideal.map_or_else(|| "-".to_owned(), |n| n.to_string());
        tracing::info!(
            thread = %thread_name,
            id,
            role,
            mask_before = %numbers(mask_before),
            mask_after = %numbers(mask_after),
            ideal_before = %ideal(ideal_before),
            ideal_after = %ideal(ideal_after),
            "VICTORY_AVOID_CPUS: the thread's processors changed"
        );
    }

    /// Applies `list` to `thread`: the mask without the listed processors,
    /// and an ideal processor that was listed moved to the next one in the
    /// mask. `role` names the thread when it has no name of its own.
    /// Whether anything changed.
    pub fn restrict_thread(thread: HANDLE, role: &str, list: &[u32]) -> bool {
        if list.is_empty() {
            return false;
        }
        let Some(before) = affinity_of(thread) else {
            return false;
        };
        let before_ideal = ideal_of(thread).map(|ideal| u32::from(ideal.Number));
        let mut asked = before;
        asked.Mask = filtered_mask(before.Mask, list);
        let mask_changed = asked.Mask != before.Mask;
        if mask_changed {
            // SAFETY: a struct of the system's layout, alive for the call.
            unsafe { SetThreadGroupAffinity(thread, &asked, std::ptr::null_mut()) };
        }
        let after = affinity_of(thread).unwrap_or(before);
        let mut after_ideal = before_ideal;
        if let Some(ideal) = before_ideal {
            let moved = ideal_after(ideal, after.Mask, list);
            if moved != ideal && set_ideal(thread, after.Group, moved) {
                after_ideal = ideal_of(thread)
                    .map(|ideal| u32::from(ideal.Number))
                    .or(Some(moved));
            }
        }
        let changed = after.Mask != before.Mask || after_ideal != before_ideal;
        if changed {
            log_change(
                thread,
                role,
                before.Mask,
                after.Mask,
                before_ideal,
                after_ideal,
            );
        }
        changed
    }

    /// [`restrict_thread`] for the calling thread, from the
    /// `VICTORY_AVOID_CPUS` list.
    pub fn restrict_current_thread(role: &str) -> bool {
        // SAFETY: the pseudo handle of the calling thread.
        let thread = unsafe { GetCurrentThread() };
        restrict_thread(thread, role, avoided())
    }

    /// `SetThreadAffinityMask`, with the listed processors dropped from the
    /// mask and a listed ideal processor moved; the previous mask, as the
    /// host's call returns it.
    pub fn set_thread_affinity_mask(thread: HANDLE, mask: usize) -> usize {
        let list = avoided();
        if list.is_empty() {
            // SAFETY: forwarding the title's own arguments to the host API.
            return unsafe { SetThreadAffinityMask(thread, mask) };
        }
        let applied = filtered_mask(mask, list);
        let before_ideal = ideal_of(thread).map(|ideal| u32::from(ideal.Number));
        let before_mask = affinity_of(thread).map_or(mask, |affinity| affinity.Mask);
        // SAFETY: forwarding the title's own arguments, filtered.
        let previous = unsafe { SetThreadAffinityMask(thread, applied) };
        if previous != 0 {
            let mut after_ideal = before_ideal;
            if let Some(ideal) = before_ideal {
                let moved = ideal_after(ideal, applied, list);
                if moved != ideal
                    && let Some(group) = affinity_of(thread).map(|a| a.Group)
                    && set_ideal(thread, group, moved)
                {
                    after_ideal = ideal_of(thread)
                        .map(|ideal| u32::from(ideal.Number))
                        .or(Some(moved));
                }
            }
            if applied != mask || after_ideal != before_ideal {
                log_change(
                    thread,
                    "SetThreadAffinityMask",
                    before_mask,
                    applied,
                    before_ideal,
                    after_ideal,
                );
            }
        }
        previous
    }

    /// `SetThreadIdealProcessor`, with a listed processor moved to the next
    /// one in the thread's mask and the mask filtered; the previous ideal,
    /// as the host's call returns it.
    pub fn set_thread_ideal_processor(thread: HANDLE, ideal: u32) -> u32 {
        let list = avoided();
        if list.is_empty() {
            // SAFETY: forwarding the title's own arguments to the host API.
            return unsafe { SetThreadIdealProcessor(thread, ideal) };
        }
        let before = affinity_of(thread);
        let before_mask = before.map_or(0, |affinity| affinity.Mask);
        if let Some(before) = before {
            let asked = filtered_mask(before.Mask, list);
            if asked != before.Mask {
                let mut set = before;
                set.Mask = asked;
                // SAFETY: a struct of the system's layout, alive for the call.
                unsafe { SetThreadGroupAffinity(thread, &set, std::ptr::null_mut()) };
            }
        }
        let mask = affinity_of(thread).map_or(before_mask, |affinity| affinity.Mask);
        let moved = ideal_after(ideal, mask, list);
        let before_ideal = ideal_of(thread).map(|ideal| u32::from(ideal.Number));
        // SAFETY: forwarding the title's own argument, filtered.
        let previous = unsafe { SetThreadIdealProcessor(thread, moved) };
        let after_ideal = ideal_of(thread)
            .map(|ideal| u32::from(ideal.Number))
            .or(before_ideal);
        if moved != ideal || mask != before_mask {
            log_change(
                thread,
                "SetThreadIdealProcessor",
                before_mask,
                mask,
                before_ideal,
                after_ideal,
            );
        }
        previous
    }

    /// `SetThreadIdealProcessorEx`, as [`set_thread_ideal_processor`].
    pub fn set_thread_ideal_processor_ex(
        thread: HANDLE,
        ideal: *const PROCESSOR_NUMBER,
        previous: *mut PROCESSOR_NUMBER,
    ) -> i32 {
        let list = avoided();
        if list.is_empty() || ideal.is_null() {
            // SAFETY: forwarding the title's own arguments to the host API.
            return unsafe { SetThreadIdealProcessorEx(thread, ideal, previous) };
        }
        // SAFETY: the title passes a PROCESSOR_NUMBER, by the API's contract.
        let asked = unsafe { *ideal };
        if asked.Group != 0 {
            // SAFETY: the title's own arguments, for a group this host is not
            // filtering.
            return unsafe { SetThreadIdealProcessorEx(thread, ideal, previous) };
        }
        let mask = affinity_of(thread).map_or(0, |affinity| affinity.Mask);
        let moved = ideal_after(u32::from(asked.Number), mask, list);
        let mut wanted = asked;
        wanted.Number = u8::try_from(moved).unwrap_or(u8::MAX);
        let before_ideal = ideal_of(thread).map(|ideal| u32::from(ideal.Number));
        // SAFETY: `wanted` is a PROCESSOR_NUMBER of the system's layout,
        // alive for the call; `previous` is the title's out pointer.
        let ok = unsafe { SetThreadIdealProcessorEx(thread, &wanted, previous) };
        let after_ideal = ideal_of(thread)
            .map(|ideal| u32::from(ideal.Number))
            .or(before_ideal);
        if ok != 0 && moved != u32::from(asked.Number) {
            log_change(
                thread,
                "SetThreadIdealProcessorEx",
                mask,
                mask,
                before_ideal,
                after_ideal,
            );
        }
        ok
    }

    /// `SetThreadGroupAffinity`, with the listed processors dropped from the
    /// mask and a listed ideal processor moved; one for `previous`, as the
    /// host's call gives it.
    pub fn set_thread_group_affinity(
        thread: HANDLE,
        affinity: *const GROUP_AFFINITY,
        previous: *mut GROUP_AFFINITY,
    ) -> i32 {
        let list = avoided();
        if list.is_empty() || affinity.is_null() {
            // SAFETY: forwarding the title's own arguments to the host API.
            return unsafe { SetThreadGroupAffinity(thread, affinity, previous) };
        }
        // SAFETY: the title passes a GROUP_AFFINITY, by the API's contract.
        let asked = unsafe { *affinity };
        let before_ideal = ideal_of(thread).map(|ideal| u32::from(ideal.Number));
        let mut set = asked;
        set.Mask = filtered_mask(asked.Mask, list);
        let before_mask = affinity_of(thread).map_or(asked.Mask, |a| a.Mask);
        // SAFETY: `set` is a GROUP_AFFINITY of the system's layout, alive for
        // the call; `previous` is the title's out pointer.
        let ok = unsafe { SetThreadGroupAffinity(thread, &set, previous) };
        if ok != 0 {
            let mut after_ideal = before_ideal;
            if let Some(ideal) = before_ideal {
                let moved = ideal_after(ideal, set.Mask, list);
                if moved != ideal && set_ideal(thread, asked.Group, moved) {
                    after_ideal = ideal_of(thread)
                        .map(|ideal| u32::from(ideal.Number))
                        .or(Some(moved));
                }
            }
            if set.Mask != asked.Mask || after_ideal != before_ideal {
                log_change(
                    thread,
                    "SetThreadGroupAffinity",
                    before_mask,
                    set.Mask,
                    before_ideal,
                    after_ideal,
                );
            }
        }
        ok
    }
}

/// Applies `VICTORY_AVOID_CPUS` once per thread, on the thread that first
/// calls `crate::host::Host::submit`.
pub(crate) fn once_before_submit() {
    if avoided().is_empty() {
        return;
    }
    std::thread_local! {
        static DONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    DONE.with(|done| {
        if !done.replace(true) {
            #[cfg(windows)]
            restrict_current_thread("submit");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The listed processors are dropped from a mask; unset is the identity;
    /// a mask the list would empty is left alone.
    #[test]
    fn mask_arithmetic() {
        assert_eq!(without(0b1010, &[1]), Some(0b1000));
        assert_eq!(without(0b1110, &[1, 3]), Some(0b0100));
        assert_eq!(without(0b10, &[1]), None, "never an empty mask");
        assert!(would_empty(0b10, &[1]));
        assert!(!would_empty(0b10, &[2]));
        assert_eq!(without(0b1010, &[]), None, "unset is the identity");
        assert_eq!(filtered_mask(0b10, &[1]), 0b10, "the mask is left alone");
        assert_eq!(numbers(0b1010), "1,3");
        assert_eq!(numbers(0), "-");
    }

    /// A listed ideal processor moves to the next processor in the mask,
    /// wrapping around; an unlisted one stays; unset is the identity.
    #[test]
    fn ideal_moves_to_the_next_unlisted_processor() {
        assert_eq!(ideal_after(1, 0b1100, &[1]), 2);
        assert_eq!(ideal_after(1, 0b0110, &[1]), 2);
        assert_eq!(ideal_after(3, 0b1001, &[3]), 0, "wraps around");
        assert_eq!(ideal_after(2, 0b1100, &[1]), 2, "not listed: it stays");
        assert_eq!(ideal_after(1, 0b1100, &[]), 1, "unset is the identity");
        assert_eq!(
            ideal_after(1, 0b10, &[1]),
            1,
            "no other processor to move to"
        );
        assert_eq!(
            ideal_after(0, 0b11, &[0, 1]),
            0,
            "no unlisted processor to move to"
        );
    }

    /// The variable's list: numbers only, sorted, without repeats; bad
    /// entries and processors no mask can name are left out.
    #[test]
    fn the_list_parses() {
        assert_eq!(parse_list(" 3, 1,3, x, 64 "), vec![1, 3]);
        assert!(parse_list("").is_empty());
        assert!(parse_list("nonsense").is_empty());
    }

    /// A thread that applies the list reports the new ideal processor (on a
    /// host with two logical processors to give it).
    #[cfg(windows)]
    #[test]
    fn a_thread_reports_its_new_ideal_processor() {
        use windows_sys::Win32::System::Kernel::PROCESSOR_NUMBER;
        use windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY;
        use windows_sys::Win32::System::Threading::{
            GetCurrentThread, GetThreadGroupAffinity, GetThreadIdealProcessorEx,
            SetThreadGroupAffinity, SetThreadIdealProcessorEx,
        };

        let (tx, rx) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            // SAFETY: the calling thread's pseudo handle, and structs of the
            // system's layout, alive for the calls.
            unsafe {
                let me = GetCurrentThread();
                let mut affinity = GROUP_AFFINITY::default();
                if GetThreadGroupAffinity(me, &mut affinity) == 0 {
                    return;
                }
                let mut wanted = affinity;
                wanted.Mask = 0b11;
                if SetThreadGroupAffinity(me, &wanted, std::ptr::null_mut()) == 0 {
                    return; // fewer than two processors this thread may use
                }
                let zero = PROCESSOR_NUMBER {
                    Group: 0,
                    Number: 0,
                    Reserved: 0,
                };
                if SetThreadIdealProcessorEx(me, &zero, std::ptr::null_mut()) == 0 {
                    return;
                }
                let changed = restrict_thread(me, "test", &[0]);
                let mut ideal = PROCESSOR_NUMBER {
                    Group: 0,
                    Number: 0,
                    Reserved: 0,
                };
                let _ = GetThreadIdealProcessorEx(me, &mut ideal);
                let mut now = GROUP_AFFINITY::default();
                let _ = GetThreadGroupAffinity(me, &mut now);
                tx.send((changed, ideal.Number, now.Mask)).unwrap();
            }
        });
        let (changed, ideal, mask) = match rx.recv() {
            Ok(result) => result,
            Err(_) => return, // this host gave the thread fewer than two processors
        };
        thread.join().unwrap();
        assert!(changed, "the restriction changed the thread");
        assert_eq!(ideal, 1, "the ideal processor moved off the listed one");
        assert_eq!(mask & 0b11, 0b10, "the listed processor left the mask");
    }
}
