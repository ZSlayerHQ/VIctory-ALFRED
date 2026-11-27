//! The console CPU's instructions that some PC CPUs lack, done in software.
//!
//! The console's CPU is an AMD Zen 2, and its instruction set has a few
//! instructions that a PC CPU may not have: `clzero`, `extrq`/`insertq`
//! (SSE4A), `monitorx`/`mwaitx` and `rdpru` are AMD's alone; `clwb`,
//! `clflushopt` and `rdpid` are missing from older CPUs of either maker.
//! Where one is missing, the CPU raises an illegal-instruction exception;
//! the vectored handler (`crash.rs`) calls [`emulate`], which decodes the
//! instruction at `Rip`, does its work on the saved context and memory,
//! steps `Rip` past it and lets the thread go on. Anything else, and any
//! access the kernel refuses, is left exactly as it was and goes to the
//! handler's report like any other exception: there is no other emulation
//! behind this one.
//!
//! Written from the processor vendors' public manuals:
//!
//! - AMD64 Architecture Programmer's Manual, Volume 3: General-Purpose and
//!   System Instructions, publication 24594, revision 3.39 ("APM3");
//! - AMD64 Architecture Programmer's Manual, Volume 4: 128-Bit, 256-Bit and
//!   512-Bit Media Instructions, publication 26568, revision 3.27 ("APM4");
//! - Processor Programming Reference for AMD Family 17h Model 31h, the Zen
//!   2 generation, publication 55803, revision 0.93 ("PPR 55803");
//! - Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 2,
//!   for what both vendors document (`clwb`, `clflushopt`, `rdpid`,
//!   `rdtsc`, `rdtscp`, `rdfsbase`, prefixes) ("SDM").
//!
//! The pieces: `decode` (the encodings, from the manuals' tables),
//! `bitfield` (EXTRQ and INSERTQ), `access` (memory touched by the kernel,
//! never by a load or store of the handler's own), `host` (which counters
//! and segment bases this PC lets user code read, learned by probes at
//! start), `flush_loop` (a write-back loop done in one exception). Where a
//! manual leaves a result undefined, the choice is stated where it is made.
//! What only the runtime embedding the module can decide (its write
//! tracking, and what stands for a register the PC does not show user
//! code) it registers as [`ConsoleInsnHooks`].
//!
//! What the PC does not get, by design: `clwb` and `clflushopt` write back
//! and flush nothing (the PC's caches are coherent with its memory, so no
//! data is lost; a title that flushed for persistence or for a device
//! would see no difference here, and none was found to); `monitorx` arms no
//! monitor and `mwaitx` returns at once, so a wait loop around them spins
//! on its own check instead of sleeping (an exception per round on a CPU
//! that lacks them). Each still checks its address as a one-byte read, as
//! the manual says the real ones do, and is refused where that fails.

mod access;
mod bitfield;
mod decode;
mod flush_loop;
mod host;
#[cfg(test)]
mod tests;

pub use host::time_stamp_counter_readable;
pub(crate) use host::{probe_host, take_probe_fault};

use access::Access;
use bitfield::{BitRange, Xmm};
use decode::{Field, MemoryOperand, Op, Refusal, Segment, decode};
use flush_loop::{MAX_PASSES, STATUS_FLAGS, compare_flags};
use host::Host;
use std::sync::OnceLock;
use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, CONTEXT_CONTROL_AMD64, CONTEXT_FLOATING_POINT_AMD64, CONTEXT_INTEGER_AMD64,
    EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, M128A,
};

/// What the runtime embedding the emulation decides for it, registered once
/// with [`set_console_insn_hooks`]. Until then the emulation refuses what
/// needs them: a write into a page the kernel will not write, and `rdpru`.
#[derive(Debug, Clone, Copy)]
pub struct ConsoleInsnHooks {
    /// The runtime's opener for pages its write tracking holds read-only:
    /// called with `[at, at + len)` where the kernel refused an emulated
    /// write, it opens those pages and notes the write as its write-fault
    /// handler does for a write of the title's, and says whether it held
    /// any. The emulation then makes the same checked write again, whose
    /// answer stands.
    pub open_tracked: fn(usize, usize) -> bool,
    /// Whether `rdpru` may run at all, asked first whatever ECX names: the
    /// console's CPU refuses it to user code under CR4.TSD (#UD, APM3
    /// p. 308), and the runtime says no where this PC denies user code the
    /// time-stamp counter. No, and the instruction is refused.
    pub rdpru_enabled: fn() -> bool,
    /// What `rdpru` reads for `register`, the processor lacking it: the
    /// value for EDX:EAX, or `None` where the runtime has none to give, and
    /// the instruction is refused.
    pub rdpru_register: fn(RdpruRegister) -> Option<u64>,
}

/// A register `rdpru` reads, as ECX names it: "ECX Value 0 = Register
/// MPERF", "ECX Value 1 = Register APERF" (APM3 p. 308), the two the
/// console's Zen 2 supports (PPR 55803 p. 86: "RdpruMax ... 01h").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RdpruRegister {
    /// Counts at the P0 frequency while the core runs (PPR 55803 p. 97).
    Mperf,
    /// Counts the clocks the core actually runs (PPR 55803 p. 97).
    Aperf,
}

impl RdpruRegister {
    /// The register ECX names; `None` for a value the console's CPU does not
    /// support, which reads as zero ("Any unsupported ECX values return
    /// zero", APM3 p. 308).
    fn named_by(ecx: u32) -> Option<Self> {
        match ecx {
            0 => Some(RdpruRegister::Mperf),
            1 => Some(RdpruRegister::Aperf),
            _ => None,
        }
    }
}

/// The hooks before any are registered: nothing opened, nothing read.
const NO_HOOKS: ConsoleInsnHooks =
    ConsoleInsnHooks { open_tracked: |_, _| false, rdpru_enabled: || false, rdpru_register: |_| None };

static HOOKS: OnceLock<ConsoleInsnHooks> = OnceLock::new();

/// Registers what the runtime decides for the emulation; once, before
/// title code (a second call changes nothing).
pub fn set_console_insn_hooks(hooks: ConsoleInsnHooks) {
    let _ = HOOKS.set(hooks);
}

/// Performs the instruction at `context.Rip` if it is one of the console's
/// that this CPU lacks, and returns `EXCEPTION_CONTINUE_EXECUTION` (-1)
/// with `Rip` past it; for anything else returns `EXCEPTION_CONTINUE_SEARCH`
/// (0) having changed nothing.
///
/// # Safety
/// `context` is the saved context of the thread whose illegal-instruction
/// exception is being dispatched, on that thread.
pub(crate) unsafe fn emulate(context: &mut CONTEXT) -> i32 {
    let hooks = HOOKS.get().copied().unwrap_or(NO_HOOKS);
    // SAFETY: forwarded from the caller.
    unsafe { emulate_on(context, host::host(), hooks) }
}

/// What the vectored handler does at an illegal-instruction exception, on
/// `context`: true if the instruction was done (`Rip` past it), false if
/// it was left alone. Public for an embedding's tests of what it registers
/// against the real emulation (victory-gpu's write tracking has them).
///
/// # Safety
/// The bytes at `context.Rip` and the memory its registers name are the
/// caller's, or none.
#[doc(hidden)]
pub unsafe fn emulate_console_instruction(context: &mut CONTEXT) -> bool {
    // SAFETY: forwarded from the caller.
    unsafe { emulate(context) == EXCEPTION_CONTINUE_EXECUTION }
}

/// [`emulate`], reading only what `host` allows, with `hooks`.
///
/// # Safety
/// As [`emulate`]'s.
unsafe fn emulate_on(context: &mut CONTEXT, host: Host, hooks: ConsoleInsnHooks) -> i32 {
    // Rip and the general registers must be in the context, or a change to
    // them would not come back.
    let needed = CONTEXT_CONTROL_AMD64 | CONTEXT_INTEGER_AMD64;
    if context.ContextFlags & needed != needed {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let (code, len) = fetch(context.Rip);
    let Ok(decoded) = decode(&code[..len]) else {
        return EXCEPTION_CONTINUE_SEARCH;
    };
    let next = context.Rip.wrapping_add(u64::from(decoded.len));
    let window = Window { code: &code[..len], len: usize::from(decoded.len), next };
    match perform(decoded.op, context, &window, host, hooks) {
        Some(step) => {
            context.Rip = context.Rip.wrapping_add(step as u64);
            EXCEPTION_CONTINUE_EXECUTION
        }
        None => EXCEPTION_CONTINUE_SEARCH,
    }
}

/// The instruction's bytes, up to 15: those of its page, then those of the
/// next page if it is cut there and that page can be read.
fn fetch(rip: u64) -> ([u8; 15], usize) {
    let mut code = [0; 15];
    let at = rip as usize;
    let mut len = crate::crash::page_bytes(at, &mut code);
    if len > 0 && len < code.len() && decode(&code[..len]) == Err(Refusal::Truncated) {
        len += crate::crash::page_bytes(at + len, &mut code[len..]);
    }
    (code, len)
}

/// The fetched bytes around a decoded instruction.
struct Window<'a> {
    /// From the instruction on.
    code: &'a [u8],
    /// The instruction's length.
    len: usize,
    /// The next instruction's address.
    next: u64,
}

/// Does `op`'s work on `context` and memory; returns how far `Rip` moves,
/// or `None` having changed nothing.
fn perform(op: Op, context: &mut CONTEXT, window: &Window, host: Host, hooks: ConsoleInsnHooks) -> Option<usize> {
    match op {
        Op::Clzero(operand) => {
            // "Clears the cache line specified by the logical address in
            // rAX by writing a zero to every byte in the line" (APM3
            // p. 156), a 64-byte line with RAX[5:0] ignored (PPR 55803
            // p. 85).
            let line = linear_address(operand, context, window.next, host)? & !63;
            match access::zero_line(usize::try_from(line).ok()?, hooks.open_tracked) {
                Access::Granted | Access::Reopened => {}
                Access::Denied => return None,
            }
        }
        Op::Clwb(operand) | Op::Clflushopt(operand) => {
            // Each makes "all the segmentation and paging checks that a
            // 1-byte read would perform" (APM3 pp. 152, 154).
            let address = linear_address(operand, context, window.next, host)?;
            if !access::byte_readable(address) {
                return None;
            }
            if let Some(step) = write_back_loop(operand, context, window, address) {
                return Some(step);
            }
        }
        Op::Monitorx(operand) => {
            // "performs the same segmentation and paging checks as a 1-byte
            // read"; any bit set in ECX (32 bits) is #GP (APM3 p. 238).
            let address = linear_address(operand, context, window.next, host)?;
            if context.Rcx as u32 != 0 || !access::byte_readable(address) {
                return None;
            }
        }
        // "the same effect on architectural state as the NOP instruction";
        // ECX bit 0 (interrupts wake it: the console's Zen 2 reports IBE,
        // PPR 55803 p. 65) and bit 1 (a timer) are defined, bits 31:2 are
        // #GP (APM3 p. 268).
        Op::Mwaitx if context.Rcx as u32 & !3 != 0 => return None,
        Op::Mwaitx => {}
        Op::Rdpru => {
            // First whether it may run at all, whatever ECX names (#UD under
            // CR4.TSD, APM3 p. 308): the runtime says, and with no hooks
            // registered it may not.
            if !(hooks.rdpru_enabled)() {
                return None;
            }
            // ECX, 32 bits, names the register; the value is the runtime's
            // (its hook), or the instruction is refused where it has none.
            let value = match RdpruRegister::named_by(context.Rcx as u32) {
                Some(register) => (hooks.rdpru_register)(register)?,
                None => 0,
            };
            // EDX:EAX, RAX's and RDX's upper halves cleared: a 32-bit result
            // is zero-extended in 64-bit mode (APM3 2.3.1).
            context.Rax = value & 0xffff_ffff;
            context.Rdx = value >> 32;
            // Every flag as it was. APM3 p. 308 marks OF, SF, ZF, AF and PF
            // cleared and CF modified with no rule for its value; the AMD
            // CPU this was written on (Family 1Ah) leaves all six as they
            // were (`rdpru_on_this_cpu_keeps_every_flag`), and no Zen 2 was
            // at hand to try. LLVM, which compiles for Zen 2, lists RDPRU as
            // writing EAX and EDX only (X86InstrSystem.td: `Defs = [EAX,
            // EDX]`), so code it compiled may keep a flag live across it:
            // keeping them all is the one choice such code cannot be broken
            // by. Code written to the manual can expect no value of CF, and
            // has no reason to test the other five after reading a counter.
        }
        Op::Rdpid { reg } => {
            // "RDPID reads the value of TSC_AUX MSR used by the RDTSCP
            // instruction into the specified destination register", 64 bits
            // in 64-bit mode (APM3 p. 307; SDM RDPID: "DEST := IA32_TSC_AUX").
            let tsc_aux = host::read_tsc_aux(host)?;
            *crate::crash::gpr(context, reg) = u64::from(tsc_aux);
        }
        Op::Extrq { dst, field } => {
            let mut xmm = saved_xmm(context)?;
            let range = match field {
                Field::Immediate { length, index } => BitRange::of(length, index),
                Field::Register(src) => bitfield::extract_range_held_in(xmm.get(src)),
            };
            xmm.set(dst, bitfield::extract(xmm.get(dst), range));
        }
        Op::Insertq { dst, src, field } => {
            let mut xmm = saved_xmm(context)?;
            let source = xmm.get(src);
            let range = match field {
                Field::Immediate { length, index } => BitRange::of(length, index),
                Field::Register(_) => bitfield::insert_range_held_in(source),
            };
            xmm.set(dst, bitfield::insert(xmm.get(dst), source, range));
        }
    }
    Some(window.len)
}

/// The linear address `operand` names in `context`, the next instruction
/// at `next` (APM3 1.4.2: scale * index + base + offset; 1.7: RIP-relative
/// from the next instruction; 1.2.3 and 1.7.3: truncated to 32 bits with a
/// 32-bit address size; 1.2.4: FS or GS adds its base). `None` where the
/// segment's base cannot be read here.
fn linear_address(operand: MemoryOperand, context: &mut CONTEXT, next: u64, host: Host) -> Option<u64> {
    let mut sum = operand.displacement as i64 as u64;
    if let Some(base) = operand.base {
        sum = sum.wrapping_add(*crate::crash::gpr(context, base));
    }
    if let Some((index, scale)) = operand.index {
        sum = sum.wrapping_add(crate::crash::gpr(context, index).wrapping_mul(u64::from(scale)));
    }
    if operand.rip_relative {
        sum = sum.wrapping_add(next);
    }
    if operand.address32 {
        sum &= 0xffff_ffff;
    }
    let segment_base = match operand.segment {
        Segment::Flat => 0,
        Segment::Gs => host::read_gs_base(),
        Segment::Fs => host::read_fs_base(host)?,
    };
    Some(sum.wrapping_add(segment_base))
}

/// The saved XMM registers, where the context has them.
struct SavedXmm<'a>(&'a mut [M128A; 16]);

impl SavedXmm<'_> {
    fn get(&self, n: u8) -> Xmm {
        let r = self.0[usize::from(n)];
        [r.Low, r.High as u64]
    }

    fn set(&mut self, n: u8, value: Xmm) {
        self.0[usize::from(n)] = M128A { Low: value[0], High: value[1] as i64 };
    }
}

fn saved_xmm(context: &mut CONTEXT) -> Option<SavedXmm<'_>> {
    let flags = CONTEXT_FLOATING_POINT_AMD64;
    if context.ContextFlags & flags != flags {
        return None;
    }
    // SAFETY: the context holds its XMM registers (the flag says so).
    Some(SavedXmm(unsafe { &mut context.Anonymous.FltSave.XmmRegisters }))
}

/// A write-back loop around the hint at `window`, whose line at `start`
/// was checked, done in one exception (`flush_loop`); the step for `Rip`,
/// or `None` for the hint alone.
fn write_back_loop(operand: MemoryOperand, context: &mut CONTEXT, window: &Window, start: u64) -> Option<usize> {
    // The hint on a base register alone, as such loops write it.
    let plain = MemoryOperand { index: None, displacement: 0, rip_relative: false, address32: false, segment: Segment::Flat, ..operand };
    let base = operand.base.filter(|_| operand == plain)?;
    let found = flush_loop::recognise(window.code, window.len, base)?;
    let limit = *crate::crash::gpr(context, found.limit);
    let passes = found.passes(start, limit)?;
    // The passes whose lines can be read, up to the most one exception
    // does: one read per page after the first line's.
    let last_line = start + (passes.min(MAX_PASSES) - 1) * 64;
    let first_page = start & !0xfff;
    let mut done = passes.min(MAX_PASSES);
    let mut page = first_page + 0x1000;
    while page <= last_line {
        if !access::page_readable(page) {
            done = (page - start).div_ceil(64);
            break;
        }
        page += 0x1000;
    }
    let end = start + done * 64;
    *crate::crash::gpr(context, base) = end;
    context.EFlags = context.EFlags & !STATUS_FLAGS | compare_flags(end, limit);
    // All passes: past the branch. Fewer: at the hint, which goes on (or is
    // refused at an unreadable line) at the next exception.
    Some(if done == passes { found.len } else { 0 })
}
