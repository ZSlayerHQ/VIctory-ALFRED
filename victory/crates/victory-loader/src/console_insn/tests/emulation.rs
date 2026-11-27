//! `emulate` end to end, on contexts built here and memory of the tests'
//! own, with the handler installed and the host probed as the runtime does.

use super::super::bitfield::{BitRange, extract, extract_range_held_in, insert, insert_range_held_in};
use super::super::{ConsoleInsnHooks, NO_HOOKS, RdpruRegister, emulate_on};
use super::super::host::{Host, host, try_privileged};
use super::Values;
use super::encode::{Disp, Mem, insn, modrm, rex};
use crate::crash::{gpr, kernel_calls, report_crashes};
use std::sync::Mutex;
use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, CONTEXT_CONTROL_AMD64, CONTEXT_FLOATING_POINT_AMD64, CONTEXT_INTEGER_AMD64,
    EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, M128A,
};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_GUARD, PAGE_NOACCESS, PAGE_READONLY,
    PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualProtect,
};

/// The handler installed and the host probed, as the runtime starts.
pub(super) fn setup() -> Host {
    report_crashes();
    host()
}

/// Fresh read-write pages of ours, released on drop.
pub(super) struct Pages(*mut u8, usize);

impl Pages {
    pub(super) fn new(pages: usize) -> Self {
        // SAFETY: a fresh allocation, which nothing else uses.
        let at = unsafe {
            VirtualAlloc(std::ptr::null(), pages << 12, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
        };
        assert!(!at.is_null());
        let this = Self(at.cast(), pages);
        this.fill(0, pages << 12, 0xcc);
        this
    }

    /// Pages of ours below 2 GiB, for 32-bit and absolute addresses.
    pub(super) fn low(pages: usize) -> Self {
        for at in (0x1000_0000..0x8000_0000usize - (pages << 12)).step_by(0x100_0000) {
            // SAFETY: asks for a fresh allocation at `at`; fails if taken.
            let got = unsafe {
                VirtualAlloc(at as *const _, pages << 12, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
            };
            if !got.is_null() {
                let this = Self(got.cast(), pages);
                this.fill(0, pages << 12, 0xcc);
                return this;
            }
        }
        panic!("no free pages below 2 GiB");
    }

    pub(super) fn addr(&self, offset: usize) -> u64 {
        assert!(offset < self.1 << 12);
        self.0 as u64 + offset as u64
    }

    pub(super) fn put(&self, offset: usize, bytes: &[u8]) {
        assert!(offset + bytes.len() <= self.1 << 12);
        // SAFETY: inside our pages, which the caller left writable.
        unsafe { self.0.add(offset).copy_from_nonoverlapping(bytes.as_ptr(), bytes.len()) };
    }

    pub(super) fn fill(&self, offset: usize, len: usize, byte: u8) {
        assert!(offset + len <= self.1 << 12);
        // SAFETY: as `put`.
        unsafe { self.0.add(offset).write_bytes(byte, len) };
    }

    pub(super) fn get(&self, offset: usize, len: usize) -> Vec<u8> {
        assert!(offset + len <= self.1 << 12);
        // SAFETY: inside our pages, which the caller left readable.
        unsafe { std::slice::from_raw_parts(self.0.add(offset), len).to_vec() }
    }

    pub(super) fn protect(&self, page: usize, protect: u32) {
        let mut old = 0;
        // SAFETY: one of our pages.
        let ok = unsafe { VirtualProtect(self.0.add(page << 12).cast(), 0x1000, protect, &mut old) };
        assert_ne!(ok, 0);
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: releases our own allocation.
        unsafe { VirtualFree(self.0.cast(), 0, MEM_RELEASE) };
    }
}

/// The context parts an exception's saved context has, of those `emulate`
/// needs.
const SAVED: u32 = CONTEXT_CONTROL_AMD64 | CONTEXT_INTEGER_AMD64 | CONTEXT_FLOATING_POINT_AMD64;

/// A context as an exception saves it, at `rip`, its registers filled
/// with generated values.
pub(super) fn context_at(rip: u64) -> CONTEXT {
    // SAFETY: a zeroed CONTEXT is valid.
    let mut context: CONTEXT = unsafe { std::mem::zeroed() };
    context.ContextFlags = SAVED;
    context.Rip = rip;
    let mut values = Values::new(rip);
    for n in 0..16 {
        *gpr(&mut context, n) = values.next();
    }
    context.EFlags = 0x202; // IF and the always-one bit
    for n in 0..16 {
        set_xmm(&mut context, n, [values.next(), values.next()]);
    }
    context
}

pub(super) fn get_xmm(context: &CONTEXT, n: u8) -> [u64; 2] {
    // SAFETY: the saved XMM registers.
    let r = unsafe { context.Anonymous.FltSave.XmmRegisters[usize::from(n)] };
    [r.Low, r.High as u64]
}

pub(super) fn set_xmm(context: &mut CONTEXT, n: u8, value: [u64; 2]) {
    // SAFETY: the saved XMM registers.
    unsafe {
        context.Anonymous.FltSave.XmmRegisters[usize::from(n)] = M128A { Low: value[0], High: value[1] as i64 };
    }
}

fn bytes(context: &CONTEXT) -> &[u8] {
    // SAFETY: a CONTEXT is plain data; the slice is its own bytes.
    unsafe { std::slice::from_raw_parts((context as *const CONTEXT).cast(), size_of::<CONTEXT>()) }
}

/// The hooks the tests' runtime registers: the stand-in write tracking's
/// opener ([`open_tracked`]), and no `rdpru` register.
pub(super) const TEST_HOOKS: ConsoleInsnHooks =
    ConsoleInsnHooks { open_tracked, rdpru_enabled: || false, rdpru_register: |_| None };

/// Runs `emulate` on a context at `rip` after `setup`, with `hooks`;
/// returns the result with the context before and after.
fn run_hooked_at(rip: u64, host: Host, hooks: ConsoleInsnHooks, setup: impl FnOnce(&mut CONTEXT)) -> (i32, CONTEXT, CONTEXT) {
    let mut context = context_at(rip);
    setup(&mut context);
    let before = context;
    // SAFETY: the instruction is in a page of the caller's, and the memory
    // its registers name is the caller's or none.
    let result = unsafe { emulate_on(&mut context, host, hooks) };
    (result, before, context)
}

/// [`run_hooked_at`] with [`TEST_HOOKS`].
pub(super) fn run_at(rip: u64, host: Host, setup: impl FnOnce(&mut CONTEXT)) -> (i32, CONTEXT, CONTEXT) {
    run_hooked_at(rip, host, TEST_HOOKS, setup)
}

/// Puts `code` at the start of a fresh page and runs it as [`run_hooked_at`].
fn run_hooked(code: &[u8], host: Host, hooks: ConsoleInsnHooks, setup: impl FnOnce(&mut CONTEXT)) -> (i32, CONTEXT, CONTEXT) {
    let page = Pages::new(1);
    page.put(0, code);
    run_hooked_at(page.addr(0), host, hooks, setup)
}

/// Puts `code` at the start of a fresh page and runs it as [`run_at`].
pub(super) fn run(code: &[u8], host: Host, setup: impl FnOnce(&mut CONTEXT)) -> (i32, CONTEXT, CONTEXT) {
    run_hooked(code, host, TEST_HOOKS, setup)
}

/// Done: `after` is `before` with `Rip` stepped by `len` and `change` made.
#[track_caller]
pub(super) fn done(result: (i32, CONTEXT, CONTEXT), len: usize, change: impl FnOnce(&mut CONTEXT)) {
    let (r, before, after) = result;
    assert_eq!(r, EXCEPTION_CONTINUE_EXECUTION);
    let mut want = before;
    want.Rip += len as u64;
    change(&mut want);
    assert!(bytes(&want) == bytes(&after), "the context changed beyond the instruction's effect");
}

/// Refused: not ours, and the context exactly as it was.
#[track_caller]
pub(super) fn left_alone(result: (i32, CONTEXT, CONTEXT)) {
    let (r, before, after) = result;
    assert_eq!(r, EXCEPTION_CONTINUE_SEARCH);
    assert!(bytes(&before) == bytes(&after), "a refused instruction changed the context");
}

/// CLZERO, 0F 01 with ModRM 11 111 100 (APM3 Table A-8, p. 156), after
/// `prefixes`.
fn clzero(prefixes: &[u8]) -> Vec<u8> {
    insn(prefixes, None, 0x01, &[modrm(3, 7, 4)], &[])
}

/// This thread's GS base, its TEB (Microsoft's `NtCurrentTeb` reads the
/// self pointer at gs:[0x30]).
fn teb() -> u64 {
    let at: u64;
    // SAFETY: reads this thread's TEB self pointer.
    unsafe { std::arch::asm!("mov {}, gs:[0x30]", out(reg) at, options(nostack, readonly)) };
    at
}

#[test]
fn clzero_zeroes_the_line_holding_its_address() {
    let host = setup();
    let data = Pages::new(1);
    let mut values = Values::new(0xc1e0);
    let lines = |data: &Pages, line: usize| {
        let all = data.get(0, 0x1000);
        all[line..line + 64].iter().all(|&b| b == 0)
            && all.iter().enumerate().all(|(i, &b)| (line..line + 64).contains(&i) || b == 0xff)
    };
    // Any byte of the line names it (PPR 55803 p. 85: "RAX[5:0] is ignored").
    for _ in 0..32 {
        let offset = (values.next() % 0x1000) as usize;
        data.fill(0, 0x1000, 0xff);
        done(run(&clzero(&[]), host, |c| c.Rax = data.addr(offset)), 3, |_| {});
        assert!(lines(&data, offset & !63), "{offset:#x}");
    }
    // 67: the address is EAX, zero-extended (APM3 1.2.3).
    let low = Pages::low(1);
    let offset = (values.next() % 0x1000) as usize;
    low.fill(0, 0x1000, 0xff);
    let rax = values.next() << 32 | low.addr(offset);
    done(run(&clzero(&[0x67]), host, |c| c.Rax = rax), 4, |_| {});
    assert!(lines(&low, offset & !63));
    // GS adds its base, this thread's TEB (APM3 1.2.4).
    let offset = (values.next() % 0x1000) as usize;
    data.fill(0, 0x1000, 0xff);
    let rax = data.addr(offset).wrapping_sub(teb());
    done(run(&clzero(&[0x65]), host, |c| c.Rax = rax), 4, |_| {});
    assert!(lines(&data, offset & !63));
    // FS adds its base, read with RDFSBASE where the system allows it.
    if host.fs_base {
        let fs: u64;
        // SAFETY: RDFSBASE rax (F3 REX.W 0F AE /0), allowed (probed).
        unsafe { std::arch::asm!(".byte 0xf3, 0x48, 0x0f, 0xae, 0xc0", out("rax") fs, options(nostack, nomem)) };
        data.fill(0, 0x1000, 0xff);
        let rax = data.addr(0x40).wrapping_sub(fs);
        done(run(&clzero(&[0x64]), host, |c| c.Rax = rax), 4, |_| {});
        assert!(lines(&data, 0x40));
    } else {
        left_alone(run(&clzero(&[0x64]), host, |c| c.Rax = data.addr(0)));
    }
}

#[test]
fn clzero_leaves_a_line_it_cannot_write() {
    let host = setup();
    let data = Pages::new(4);
    data.fill(0, 0x4000, 0xff);
    data.protect(0, PAGE_READONLY);
    data.protect(1, PAGE_READWRITE | PAGE_GUARD);
    data.protect(2, PAGE_NOACCESS);
    // SAFETY: decommits the last of our pages.
    assert_ne!(unsafe { VirtualFree((data.addr(0x3000) as usize) as *mut _, 0x1000, MEM_DECOMMIT) }, 0);
    let unwritable = [
        data.addr(0x0040),
        data.addr(0x1040),
        data.addr(0x2040),
        data.addr(0x3040),
        0x40,                  // never mapped
        0xffff_8000_0000_0000, // the system's half
        0x0000_8000_0000_0000, // not canonical
    ];
    for rax in unwritable {
        left_alone(run(&clzero(&[]), host, |c| c.Rax = rax));
    }
    for page in 0..3 {
        data.protect(page, PAGE_READWRITE);
        assert!(data.get(page << 12, 0x1000).iter().all(|&b| b == 0xff), "page {page}");
    }
}

/// The pages a test's write tracking holds read-only, and the writes it
/// let through: a stand-in for the runtime's (victory-gpu's watch, whose
/// own tests run the emulation against the real one).
static TRACKED: Mutex<Vec<(u64, usize)>> = Mutex::new(Vec::new());
static NOTED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
/// Pages the tracking held, whose protection it can no longer restore (the
/// program changed or freed them since): it says it held them, and they
/// stay as they are.
static HELD_SHUT: Mutex<Vec<(u64, usize)>> = Mutex::new(Vec::new());

/// The tracking's opener: a write about to land in a tracked page opens
/// the page and is noted, as the runtime's write-fault handler does; true
/// if it held the page.
fn open_tracked(at: usize, len: usize) -> bool {
    let within = |&&(s, n): &&(u64, usize)| at as u64 >= s && (at + len) as u64 <= s + n as u64;
    if HELD_SHUT.lock().unwrap().iter().any(|entry| within(&entry)) {
        NOTED.lock().unwrap().push((at, len));
        return true;
    }
    let tracked = TRACKED.lock().unwrap();
    let held = tracked.iter().find(within);
    if let Some(&(start, size)) = held {
        let mut old = 0;
        // SAFETY: a test page the tracking holds.
        unsafe { VirtualProtect(start as *const _, size, PAGE_READWRITE, &mut old) };
        NOTED.lock().unwrap().push((at, len));
    }
    held.is_some()
}

#[test]
fn a_line_the_opener_says_it_held_but_cannot_open_is_refused() {
    // The opener's true says the tracking held the page, not that the page
    // is writable now: the program may have changed its protection or freed
    // it since. Nothing is written and nothing faults; the refusal goes to
    // the handler's report, after two openings and no more.
    let host = setup();
    let data = Pages::new(2);
    data.fill(0, 0x2000, 0xff);
    data.protect(0, PAGE_READONLY);
    data.protect(1, PAGE_NOACCESS);
    HELD_SHUT.lock().unwrap().extend([(data.addr(0), 0x1000), (data.addr(0x1000), 0x1000)]);
    for line in [data.addr(0x240), data.addr(0x1fc0)] {
        let asked = || NOTED.lock().unwrap().iter().filter(|&&n| n == (line as usize, 64)).count();
        let before = asked();
        left_alone(run(&clzero(&[]), host, |c| c.Rax = line + 5));
        assert_eq!(asked() - before, 2, "the opener asked twice");
    }
    HELD_SHUT.lock().unwrap().retain(|&(start, _)| start != data.addr(0) && start != data.addr(0x1000));
    data.protect(0, PAGE_READWRITE);
    data.protect(1, PAGE_READWRITE);
    assert!(data.get(0, 0x2000).iter().all(|&b| b == 0xff));
}

#[test]
fn a_line_on_a_tracked_read_only_page_is_opened_noted_and_zeroed() {
    let host = setup();
    let data = Pages::new(2);
    data.fill(0, 0x2000, 0xff);
    data.protect(0, PAGE_READONLY);
    data.protect(1, PAGE_READONLY);
    TRACKED.lock().unwrap().push((data.addr(0), 0x1000));
    let line = data.addr(0x7c0);
    done(run(&clzero(&[]), host, |c| c.Rax = line + 9), 3, |_| {});
    assert!(NOTED.lock().unwrap().contains(&(line as usize, 64)));
    assert!(data.get(0x7c0, 64).iter().all(|&b| b == 0));
    // The page the tracking does not hold stays the title's: refused.
    left_alone(run(&clzero(&[]), host, |c| c.Rax = data.addr(0x1040)));
    // The tracking lets the page go before it is freed: an allocation
    // there later is another test's.
    TRACKED.lock().unwrap().retain(|&(start, _)| start != data.addr(0));
    data.protect(1, PAGE_READWRITE);
    assert!(data.get(0x1000, 0x1000).iter().all(|&b| b == 0xff));
}

#[test]
fn a_megabyte_zeroed_line_by_line_takes_one_kernel_write_a_line_and_no_page_query() {
    // Each line: one read of the instruction's bytes and one write of the
    // line, counted at the kernel calls themselves; no page query, which
    // costs milliseconds in a big resident region.
    let host = setup();
    let data = Pages::new(256);
    data.fill(0, 1 << 20, 0xff); // resident, where a page query costs most
    let code = Pages::new(1);
    code.put(0, &clzero(&[]));
    let before = kernel_calls();
    for line in (0..1 << 20).step_by(64) {
        done(run_at(code.addr(0), host, |c| c.Rax = data.addr(line)), 3, |_| {});
    }
    let after = kernel_calls();
    let lines = (1 << 20) / 64;
    assert_eq!(
        (after.reads - before.reads, after.writes - before.writes, after.queries - before.queries),
        (lines, lines, 0),
        "kernel reads, writes and page queries for 16,384 lines"
    );
    assert!(data.get(0, 1 << 20).iter().all(|&b| b == 0));
}

/// Register values for `mem` (registers `base` and `index`, scale and
/// displacement chosen by the encoder) so that its address is `target`.
fn aim(context: &mut CONTEXT, mem: Mem, target: u64, extra: u64) {
    match mem {
        Mem::Base { base, disp } => {
            let d = match disp { Disp::None => 0, Disp::Byte(d) => i64::from(d), Disp::Dword(d) => i64::from(d) };
            *gpr(context, base) = target.wrapping_sub(d as u64);
        }
        Mem::Indexed { base, index, scale, disp } => {
            let d = match disp { Disp::None => 0, Disp::Byte(d) => i64::from(d), Disp::Dword(d) => i64::from(d) };
            *gpr(context, index) = extra;
            *gpr(context, base) = target.wrapping_sub(extra.wrapping_mul(u64::from(scale))).wrapping_sub(d as u64);
        }
        Mem::NoBase { index: Some((index, scale)), disp } => {
            let rest = target.wrapping_sub(disp as i64 as u64);
            assert_eq!(rest % u64::from(scale), 0);
            *gpr(context, index) = rest / u64::from(scale);
        }
        // An absolute or RIP-relative address is in the displacement.
        Mem::NoBase { index: None, .. } | Mem::Rip(_) => {}
    }
}

#[test]
fn cache_hints_and_monitorx_check_their_address() {
    // CLWB, CLFLUSHOPT and MONITORX make the checks of a one-byte read
    // (APM3 pp. 152, 154, 238): a byte that cannot be read is refused.
    let host = setup();
    // Code, a readable page, a page with no access: all below 2 GiB, so
    // absolute and 32-bit addresses reach them too.
    let pages = Pages::low(3);
    pages.protect(2, PAGE_NOACCESS);
    let mut values = Values::new(0xadd5);
    for target_page in [1, 2] {
        // Eight-byte aligned, so [index * scale + disp] reaches it.
        let target = pages.addr(target_page << 12) + ((values.next() % 0x1000) & !7);
        let readable = target_page == 1;
        for digit in [6, 7] {
            let base = (values.next() % 16) as u8;
            let index = [0, 1, 2, 3, 5, 6, 7, 8, 9, 12, 15].into_iter().find(|&i| i != base).unwrap();
            let scale = [1, 2, 4, 8][(values.next() % 4) as usize];
            let byte_disp = (values.next() as i8) & !7;
            let forms = [
                Mem::Base { base, disp: Disp::None },
                Mem::Base { base, disp: Disp::Byte(byte_disp) },
                Mem::Indexed { base, index, scale, disp: Disp::Dword(0x40) },
                Mem::NoBase { index: Some((index, scale)), disp: -0x40 },
                Mem::NoBase { index: None, disp: target as i32 },
                Mem::Rip(0),
            ];
            for mem in forms {
                for address32 in [false, true] {
                    let encoded = mem.encode(digit);
                    let mut prefixes = vec![0x66];
                    if address32 {
                        prefixes.push(0x67);
                    }
                    let needs = (encoded.x | encoded.b) >= 8;
                    let mut code = insn(&prefixes, needs.then(|| rex(false, 0, encoded.x, encoded.b)), 0xae, &encoded.bytes, &[]);
                    let next_rip = pages.addr(0) + code.len() as u64;
                    if let Mem::Rip(_) = mem {
                        let disp = (target as i64 - next_rip as i64) as i32;
                        let len = code.len();
                        code[len - 4..].copy_from_slice(&disp.to_le_bytes());
                    }
                    pages.fill(0, 0x40, 0xcc);
                    pages.put(0, &code);
                    let extra = values.next() % 0x100;
                    let result = run_at(pages.addr(0), host, |c| {
                        aim(c, mem, target, extra);
                        if address32 {
                            // Junk above bit 31 of every register: truncated away.
                            for n in 0..16 {
                                let v = *gpr(c, n);
                                *gpr(c, n) = v & 0xffff_ffff | 0xdead_0000_0000_0000;
                            }
                        }
                    });
                    if readable {
                        done(result, code.len(), |_| {});
                    } else {
                        left_alone(result);
                    }
                }
            }
        }
        // MONITORX at [rAX].
        let monitorx = insn(&[], None, 0x01, &[modrm(3, 7, 2)], &[]);
        let result = run(&monitorx, host, |c| {
            c.Rax = target;
            c.Rcx = 0;
        });
        if readable {
            done(result, 3, |_| {});
        } else {
            left_alone(result);
        }
    }
}

#[test]
fn monitorx_and_mwaitx_take_only_the_ecx_bits_defined() {
    // MONITORX: "setting any bits in ECX will result in a #GP" (APM3
    // p. 238). MWAITX: ECX bit 0 lets interrupts wake it, bit 1 sets the
    // timer, bits 31:2 are #GP (p. 268); bit 0 needs IBE, which the
    // console's Zen 2 reports (PPR 55803 p. 65, "IBE ... Fixed,1"). Both are
    // 32-bit operands: RCX's upper half does not count.
    let host = setup();
    let data = Pages::new(1);
    let monitorx = insn(&[], None, 0x01, &[modrm(3, 7, 2)], &[]);
    let mwaitx = insn(&[], None, 0x01, &[modrm(3, 7, 3)], &[]);
    let mut values = Values::new(0x3a17);
    for bit in 0..64 {
        let rcx = 1u64 << bit | if bit >= 32 { 0 } else { values.next() & 0xffff_ffff_0000_0000 };
        let result = run(&monitorx, host, |c| {
            c.Rax = data.addr(0);
            c.Rcx = rcx;
        });
        if bit < 32 { left_alone(result) } else { done(result, 3, |_| {}) }
        let result = run(&mwaitx, host, |c| c.Rcx = rcx);
        if (2..32).contains(&bit) { left_alone(result) } else { done(result, 3, |_| {}) }
    }
}

/// The status flags: OF, SF, ZF, AF, PF, CF.
const STATUS: u32 = 0x8d5;

thread_local! {
    /// What the test's `rdpru` hook gives, and the registers it was asked.
    static GIVES: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
    static ASKED: std::cell::RefCell<Vec<RdpruRegister>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The test's runtime's `rdpru` hook: [`GIVES`], noting the register.
fn rdpru_hook(register: RdpruRegister) -> Option<u64> {
    ASKED.with_borrow_mut(|asked| asked.push(register));
    GIVES.get()
}

/// A counter hook for tests where it must not be asked.
fn rdpru_never(register: RdpruRegister) -> Option<u64> {
    panic!("the counter was read for {register:?}")
}

#[test]
fn rdpru_reads_what_the_runtime_gives() {
    // RDPRU (APM3 p. 308): "Copy register specified by ECX into EDX:EAX",
    // RAX's and RDX's upper halves cleared (32-bit results, APM3 2.3.1).
    // The runtime first says whether it may run at all (#UD under CR4.TSD),
    // for every ECX. Then ECX, 32 bits: 0 MPERF, 1 APERF, the two the
    // console's Zen 2 has (PPR 55803 p. 86, "RdpruMax ... 01h"), whose value
    // the runtime's hook gives, or the instruction is refused where it gives
    // none; "Any unsupported ECX values return zero", the counter not read.
    // Every flag as it was, set or clear: the emulation's choice, after the
    // CPU at hand (`rdpru_on_this_cpu_keeps_every_flag`) and LLVM's
    // description of the instruction (`Defs = [EAX, EDX]`).
    use RdpruRegister::{Aperf, Mperf};
    let host = setup();
    let granted = ConsoleInsnHooks { rdpru_enabled: || true, rdpru_register: rdpru_hook, ..TEST_HOOKS };
    let denied = ConsoleInsnHooks { rdpru_enabled: || false, ..granted };
    let rdpru = insn(&[], None, 0x01, &[modrm(3, 7, 5)], &[]);
    let mut values = Values::new(0x7d50);
    let unsupported = |values: &mut Values| values.next() << 32 | u64::from((values.next() as u32).max(2));
    for (ecx, register) in [(0, Mperf), (1, Aperf)] {
        for status in [0, STATUS] {
            let (value, rcx) = (values.next(), values.next() << 32 | ecx);
            let given = |c: &mut CONTEXT| {
                c.Rcx = rcx;
                c.EFlags |= status;
            };
            // Granted: the register's value, or refused where it has none.
            GIVES.set(Some(value));
            done(run_hooked(&rdpru, host, granted, given), 3, |c| (c.Rax, c.Rdx) = (value & 0xffff_ffff, value >> 32));
            assert_eq!(ASKED.take(), [register]);
            GIVES.set(None);
            left_alone(run_hooked(&rdpru, host, granted, given));
            assert_eq!(ASKED.take(), [register]);
            // Denied, a supported ECX: refused, the counter not asked.
            GIVES.set(Some(value));
            left_alone(run_hooked(&rdpru, host, denied, given));
            assert!(ASKED.take().is_empty());
        }
    }
    for n in 0..32u64 {
        let rcx = unsupported(&mut values);
        let given = |c: &mut CONTEXT| {
            c.Rcx = rcx;
            c.EFlags |= if n % 2 == 0 { STATUS } else { 0 };
        };
        // Granted, an unsupported ECX: zero, and no counter read.
        let never = ConsoleInsnHooks { rdpru_register: rdpru_never, ..granted };
        done(run_hooked(&rdpru, host, never, given), 3, |c| (c.Rax, c.Rdx) = (0, 0));
        // Denied, an unsupported ECX: refused all the same.
        left_alone(run_hooked(&rdpru, host, ConsoleInsnHooks { rdpru_register: rdpru_never, ..denied }, given));
        // No hooks registered: refused, supported or not.
        left_alone(run_hooked(&rdpru, host, NO_HOOKS, given));
        left_alone(run_hooked(&rdpru, host, NO_HOOKS, |c| c.Rcx = rcx & !0xffff_ffff | (n & 1)));
    }
}

/// Whether this CPU has RDPRU (CPUID Fn8000_0008_EBX[4], APM3 p. 308).
fn has_rdpru() -> bool {
    use std::arch::x86_64::__cpuid;
    __cpuid(0x8000_0000).eax >= 0x8000_0008 && __cpuid(0x8000_0008).ebx & 1 << 4 != 0
}

#[test]
fn rdpru_on_this_cpu_keeps_every_flag() {
    // The hardware behind the emulation's flags: APM3 p. 308 marks OF, SF,
    // ZF, AF and PF cleared and CF modified with no rule for its value; the
    // AMD CPU this was written on (Family 1Ah) leaves all six as they were,
    // set or clear, for supported and unsupported ECX alike.
    if !has_rdpru() {
        eprintln!(
            "rdpru_on_this_cpu_keeps_every_flag: skipped, this CPU has no RDPRU (CPUID Fn8000_0008_EBX[4] is 0) \
             to compare with; the emulation's flags are asserted in the test of what rdpru reads"
        );
        return;
    }
    for ecx in [0u32, 1, 2, 0x8000_0001] {
        for status in [0, STATUS] {
            let before = 0x202 | u64::from(status);
            let after: u64;
            // SAFETY: RDPRU (0F 01 FD) writes EAX and EDX and the flags; the
            // flags are set before and read after through the stack.
            unsafe {
                std::arch::asm!(
                    "push {before}",
                    "popfq",
                    ".byte 0x0f, 0x01, 0xfd",
                    "pushfq",
                    "pop {after}",
                    before = in(reg) before,
                    after = lateout(reg) after,
                    in("ecx") ecx,
                    out("eax") _,
                    out("edx") _,
                );
            }
            assert_eq!(after & u64::from(STATUS), u64::from(status), "ECX {ecx:#x}, status flags {status:#x} before");
        }
    }
}

#[test]
fn rdpid_reads_tsc_aux_where_rdtscp_is_allowed() {
    use std::arch::x86_64::__rdtscp;
    // RDPID (APM3 p. 307): TSC_AUX, as RDTSCP returns it in ECX (p. 445),
    // into the register, zero-extended in 64-bit mode.
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcessorNumber, GetCurrentThread, SetThreadAffinityMask,
    };
    let host = setup();
    // SAFETY: holds this test's thread on the processor it is on, until the
    // end of the test.
    let held = unsafe { SetThreadAffinityMask(GetCurrentThread(), 1 << GetCurrentProcessorNumber()) };
    assert_ne!(held, 0);
    for reg in 0..16 {
        let code = insn(&[0xf3], (reg >= 8).then(|| rex(false, 0, 0, reg)), 0xc7, &[modrm(3, 7, reg)], &[]);
        // Without RDTSCP allowed, refused: the CPU without it, or a system
        // that forbids it to user code.
        left_alone(run(&code, Host { tsc_aux: false, ..host }, |_| {}));
        if !host.tsc_aux {
            continue;
        }
        let tsc_aux = || {
            let mut aux = 0;
            // SAFETY: RDTSCP may run here (probed).
            unsafe { __rdtscp(&mut aux) };
            aux
        };
        // Held on one processor, so TSC_AUX is the same before and in the
        // emulation.
        let aux = tsc_aux();
        let result = run(&code, host, |_| {});
        done(result, code.len(), |c| *gpr(c, reg) = u64::from(aux));
    }
    // SAFETY: gives the thread its processors back.
    unsafe { SetThreadAffinityMask(GetCurrentThread(), held) };
}

#[test]
fn extrq_and_insertq_work_on_the_saved_registers() {
    // The four forms (APM4 pp. 173, 196) on generated registers and fields;
    // the result as `bitfield` computes it (tested against a bitwise
    // reference in `fields`), the destination's upper quadword zeroed.
    let host = setup();
    let mut values = Values::new(0x55e4);
    for _ in 0..200 {
        let dst = (values.next() % 16) as u8;
        let src = (values.next() % 16) as u8;
        let (length, index) = (values.next() as u8, values.next() as u8);
        let pair = ((dst | src) >= 8).then(|| rex(false, dst, 0, src));
        let forms: [(Vec<u8>, bool, bool); 4] = [
            (insn(&[0x66], (dst >= 8).then(|| rex(false, 0, 0, dst)), 0x78, &[modrm(3, 0, dst)], &[length, index]), false, true),
            (insn(&[0x66], pair, 0x79, &[modrm(3, dst, src)], &[]), false, false),
            (insn(&[0xf2], pair, 0x78, &[modrm(3, dst, src)], &[length, index]), true, true),
            (insn(&[0xf2], pair, 0x79, &[modrm(3, dst, src)], &[]), true, false),
        ];
        for (code, inserting, immediate) in forms {
            let result = run(&code, host, |_| {});
            let before = result.1;
            let (d, s) = (get_xmm(&before, dst), get_xmm(&before, src));
            let want = match (inserting, immediate) {
                (false, true) => extract(d, BitRange::of(length, index)),
                (false, false) => extract(d, extract_range_held_in(s)),
                (true, true) => insert(d, s, BitRange::of(length, index)),
                (true, false) => insert(d, s, insert_range_held_in(s)),
            };
            assert_eq!(want[1], 0);
            done(result, code.len(), |c| set_xmm(c, dst, want));
        }
    }
}

#[test]
fn mixed_prefixes_agree_with_the_cpu() {
    // F2 with 66 either side of it: INSERTQ on the CPU (an AMD Family 1Ah
    // Model 44h here); the manuals give no row for the pair (APM3 Table
    // A-4; SDM Vol. 2A 2.1.1: "Other use of the 66H prefix is reserved").
    // The CPU runs the very bytes the emulation is given, where it has
    // SSE4A; elsewhere that half is skipped, said on stderr, and the
    // emulation is held to the bitfield reference alone.
    use super::fields::cpu;
    let host = setup();
    let sse4a = std::arch::is_x86_feature_detected!("sse4a");
    if !sse4a {
        eprintln!(
            "mixed_prefixes_agree_with_the_cpu: the CPU's own run skipped, this CPU has no SSE4A \
             (CPUID Fn8000_0001_ECX[6] is 0); the emulation is checked against the reference only"
        );
    }
    let mut values = Values::new(0xf266);
    for prefixes in [[0xf2, 0x66], [0x66, 0xf2]] {
        for _ in 0..16 {
            let code = insn(&prefixes, None, 0x79, &[modrm(3, 0, 1)], &[]);
            let fields = values.next() & 0x3f3f;
            let (xmm0, xmm1) = ([values.next(), values.next()], [values.next(), fields]);
            let want = if sse4a {
                let native = cpu_with_prefixes(prefixes, xmm0, xmm1);
                assert_eq!(native, cpu(true, xmm0, xmm1), "the CPU runs {prefixes:02x?} 0F 79 as INSERTQ");
                native
            } else {
                insert(xmm0, xmm1, insert_range_held_in(xmm1))
            };
            let result = run(&code, host, |c| {
                set_xmm(c, 0, xmm0);
                set_xmm(c, 1, xmm1);
            });
            done(result, code.len(), |c| set_xmm(c, 0, want));
        }
    }
}

/// The CPU's own run of `prefixes` then 0F 79 C1 (ModRM 11 000 001) on xmm0
/// and xmm1: xmm0 after. The CPU must have SSE4A.
fn cpu_with_prefixes(prefixes: [u8; 2], xmm0: [u64; 2], xmm1: [u64; 2]) -> [u64; 2] {
    use std::arch::x86_64::{__m128i, _mm_set_epi64x};
    assert!(std::arch::is_x86_feature_detected!("sse4a"));
    // SAFETY: SSE2 is part of x86-64 and SSE4A is present (asserted); the
    // instruction touches the two registers named.
    unsafe {
        let mut a = _mm_set_epi64x(xmm0[1] as i64, xmm0[0] as i64);
        let b = _mm_set_epi64x(xmm1[1] as i64, xmm1[0] as i64);
        match prefixes {
            [0xf2, 0x66] => std::arch::asm!(".byte 0xf2, 0x66, 0x0f, 0x79, 0xc1", inout("xmm0") a, in("xmm1") b,
                options(pure, nomem, nostack)),
            [0x66, 0xf2] => std::arch::asm!(".byte 0x66, 0xf2, 0x0f, 0x79, 0xc1", inout("xmm0") a, in("xmm1") b,
                options(pure, nomem, nostack)),
            _ => unreachable!("{prefixes:02x?}"),
        }
        std::mem::transmute::<__m128i, [u64; 2]>(a)
    }
}

#[test]
fn a_context_without_the_parts_needed_is_left_alone() {
    let host = setup();
    let data = Pages::new(1);
    let extrq = insn(&[0x66], None, 0x79, &[modrm(3, 0, 1)], &[]);
    // No XMM registers: extrq has nothing to work on; clzero still works.
    let partial = CONTEXT_CONTROL_AMD64 | CONTEXT_INTEGER_AMD64;
    left_alone(run(&extrq, host, |c| c.ContextFlags = partial));
    done(run(&clzero(&[]), host, |c| { c.ContextFlags = partial; c.Rax = data.addr(0) }), 3, |_| {});
    // No control or no integer registers: Rip or RAX would not come back.
    for flags in [CONTEXT_INTEGER_AMD64, CONTEXT_CONTROL_AMD64 | CONTEXT_FLOATING_POINT_AMD64] {
        left_alone(run(&clzero(&[]), host, |c| { c.ContextFlags = flags; c.Rax = data.addr(0) }));
    }
}

#[test]
fn an_instruction_across_a_page_end() {
    let host = setup();
    let pages = Pages::new(3);
    // clwb [rip+disp32] aimed at page 2: 8 bytes, the last 3 on page 1.
    let mut code = insn(&[0x66], None, 0xae, &Mem::Rip(0).encode(6).bytes, &[]);
    let at = 0x1000 - 5;
    let next_rip = pages.addr(at) + code.len() as u64;
    let disp = (pages.addr(0x2000) as i64 - next_rip as i64) as i32;
    let len = code.len();
    code[len - 4..].copy_from_slice(&disp.to_le_bytes());
    pages.put(at, &code);
    pages.protect(1, PAGE_NOACCESS);
    left_alone(run_at(pages.addr(at), host, |_| {}));
    pages.protect(1, PAGE_READONLY);
    done(run_at(pages.addr(at), host, |_| {}), code.len(), |_| {});
    // One that ends at its page's end needs nothing after it.
    let last = Pages::new(1);
    let clz = clzero(&[]);
    last.put(0x1000 - clz.len(), &clz);
    let rax = pages.addr(0x2000);
    done(run_at(last.addr(0x1000 - clz.len()), host, |c| c.Rax = rax), clz.len(), |_| {});
    last.protect(0, PAGE_NOACCESS);
    left_alone(run_at(last.addr(0x1000 - clz.len()), host, |c| c.Rax = rax));
}

#[test]
fn other_instructions_are_left_alone() {
    // Generated byte strings that do not decode as ours, through `emulate`.
    use super::super::decode::decode;
    let host = setup();
    let data = Pages::new(1);
    data.fill(0, 0x1000, 0xff);
    let mut values = Values::new(0x07e5);
    let mut tried = 0;
    while tried < 2000 {
        let mut code = [0u8; 15];
        for byte in code.iter_mut() {
            *byte = values.next() as u8;
        }
        // Half of them in the 0F map, where ours are.
        if tried % 2 == 0 {
            code[0] = 0x0f;
            code[1] = [0x01, 0xae, 0xc7, 0x78, 0x79][(values.next() % 5) as usize];
        }
        if decode(&code).is_ok() {
            continue;
        }
        tried += 1;
        left_alone(run(&code, host, |c| c.Rax = data.addr(0)));
    }
    assert!(data.get(0, 0x1000).iter().all(|&b| b == 0xff));
}

#[test]
fn probes_take_back_their_own_faults() {
    // HLT is privileged: in user code it faults (#GP), and the handler
    // takes the probe's fault back instead of reporting it. It stands for
    // RDTSC under CR4.TSD, which faults the same way (#GP at CPL 3) but
    // cannot be set from user code on Windows: a probe that faulted reads
    // as not allowed, the runtime's `rdpru` hook then gives nothing
    // (victory-hle's runtime tests), and the instruction is refused
    // (`rdpru_reads_what_the_runtime_gives`).
    setup();
    assert!(!try_privileged());
}
