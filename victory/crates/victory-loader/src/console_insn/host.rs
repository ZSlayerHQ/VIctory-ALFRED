//! What this PC lets user code read, learned once by trying.
//!
//! RDTSC and RDTSCP fault (#GP) in user code where the system set CR4.TSD
//! (SDM Vol. 2 RDTSC, RDTSCP; APM3 p. 445), RDTSCP is #UD on a CPU without
//! it, and RDFSBASE is #UD unless the system set CR4.FSGSBASE (SDM Vol. 2
//! RDFSBASE). None of that can be asked from user code, so each is tried
//! once, at start, in a probe whose fault the vectored handler takes back
//! ([`take_probe_fault`]); the emulation then reads only what worked.

use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering::SeqCst};
use windows_sys::Win32::System::Diagnostics::Debug::CONTEXT;

/// The readings `emulate` may take on this PC, and RDTSC, which only the
/// runtime's hook takes ([`time_stamp_counter_readable`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct Host {
    /// RDTSC.
    pub(super) time_stamp: bool,
    /// RDTSCP, for TSC_AUX.
    pub(super) tsc_aux: bool,
    /// RDFSBASE, for an FS-relative address.
    pub(super) fs_base: bool,
}

/// Where a probe's instruction is while it runs, and where it resumes if
/// the instruction faults.
struct ProbeSite {
    at: AtomicUsize,
    resume: AtomicUsize,
}

impl ProbeSite {
    const fn new() -> Self {
        Self { at: AtomicUsize::new(0), resume: AtomicUsize::new(0) }
    }
}

/// One site per probe: RDTSC, RDTSCP, RDFSBASE, and HLT for tests.
static SITES: [ProbeSite; 4] = [ProbeSite::new(), ProbeSite::new(), ProbeSite::new(), ProbeSite::new()];

/// Runs one instruction (given as `.byte` text) in a probe at site `$site`:
/// the probe stores its own instruction's address and its resume label
/// in the site, so the handler knows the fault as the probe's whatever
/// copy of the code runs; R11 is 1 if the instruction completed, and stays
/// 0 if it faulted and was resumed past.
macro_rules! probe {
    ($site:expr, $instruction:literal) => {{
        let site = &SITES[$site];
        let completed: u64;
        // SAFETY: the instruction writes at most RAX, RCX and RDX (marked
        // clobbered) or faults; a fault is taken back by the handler, which
        // resumes at label 3 with R11 still 0. The stores go to the site's
        // own atomics, read only by this thread's handler.
        unsafe {
            std::arch::asm!(
                "lea {t}, [rip + 3f]",
                "mov qword ptr [{resume}], {t}",
                "lea {t}, [rip + 2f]",
                "mov qword ptr [{at}], {t}",
                "xor r11d, r11d",
                "2:",
                $instruction,
                "mov r11d, 1",
                "3:",
                "mov qword ptr [{at}], 0",
                at = in(reg) site.at.as_ptr(),
                resume = in(reg) site.resume.as_ptr(),
                t = out(reg) _,
                out("r11") completed,
                out("rax") _,
                out("rcx") _,
                out("rdx") _,
                options(nostack),
            );
        }
        completed == 1
    }};
}

/// What the probes found: bit 7 once they ran, then bits 0-2.
static LEARNED: AtomicU8 = AtomicU8::new(0);

/// Learns what this PC allows; call once, with the vectored handler
/// installed (`crash::report_crashes`).
pub(crate) fn probe_host() {
    // RDTSC 0F 31; RDTSCP 0F 01 F9 (APM3 p. 445); RDFSBASE RAX, F3 REX.W
    // 0F AE /0 with mod 11 (SDM Vol. 2 RDFSBASE).
    let time_stamp = probe!(0, ".byte 0x0f, 0x31");
    let tsc_aux = probe!(1, ".byte 0x0f, 0x01, 0xf9");
    let fs_base = probe!(2, ".byte 0xf3, 0x48, 0x0f, 0xae, 0xc0");
    let bits = u8::from(time_stamp) | u8::from(tsc_aux) << 1 | u8::from(fs_base) << 2;
    LEARNED.store(0x80 | bits, SeqCst);
}

/// What [`probe_host`] learned; nothing allowed before it ran.
pub(super) fn host() -> Host {
    let bits = LEARNED.load(SeqCst);
    let known = bits & 0x80 != 0;
    Host {
        time_stamp: known && bits & 1 != 0,
        tsc_aux: known && bits & 2 != 0,
        fs_base: known && bits & 4 != 0,
    }
}

/// Takes back a fault of one of the probes: steps `context` to the
/// probe's resume label. The vectored handler asks this first.
pub(crate) fn take_probe_fault(context: &mut CONTEXT) -> bool {
    let rip = context.Rip as usize;
    for site in &SITES {
        let at = site.at.load(SeqCst);
        if at != 0 && at == rip {
            context.Rip = site.resume.load(SeqCst) as u64;
            return true;
        }
    }
    false
}

/// HLT, which user code may never run (#GP), in a probe: false once the
/// handler took its fault back. For tests of the probes' containment.
#[cfg(test)]
pub(super) fn try_privileged() -> bool {
    probe!(3, "hlt")
}

/// Whether user code on this PC may read the time-stamp counter (RDTSC
/// faults where the system set CR4.TSD), as the start-up probe found
/// (`report_crashes` runs it); false before it ran. For a runtime whose
/// `rdpru` hook ([`super::ConsoleInsnHooks`]) reads the counter: the module
/// reads none itself.
pub fn time_stamp_counter_readable() -> bool {
    host().time_stamp
}

/// TSC_AUX as RDTSCP returns it in ECX (APM3 p. 445), where the caller
/// found [`Host::tsc_aux`].
pub(super) fn read_tsc_aux(allowed: Host) -> Option<u32> {
    allowed.tsc_aux.then(|| {
        let mut aux = 0;
        // SAFETY: RDTSCP ran in user code here (probed).
        unsafe { std::arch::x86_64::__rdtscp(&mut aux) };
        aux
    })
}

/// This thread's FS base, where the caller found [`Host::fs_base`].
pub(super) fn read_fs_base(allowed: Host) -> Option<u64> {
    allowed.fs_base.then(|| {
        let base: u64;
        // SAFETY: RDFSBASE ran in user code here (probed); it writes RAX.
        unsafe {
            std::arch::asm!(".byte 0xf3, 0x48, 0x0f, 0xae, 0xc0", out("rax") base, options(nostack, nomem));
        }
        base
    })
}

/// This thread's GS base: its TEB, whose self pointer is at gs:[0x30]
/// (Microsoft's `NtCurrentTeb` for x64 reads it so). The handler runs on
/// the thread that faulted.
pub(super) fn read_gs_base() -> u64 {
    let base: u64;
    // SAFETY: reads this thread's TEB self pointer.
    unsafe { std::arch::asm!("mov {}, gs:[0x30]", out(reg) base, options(nostack, readonly, preserves_flags)) };
    base
}
