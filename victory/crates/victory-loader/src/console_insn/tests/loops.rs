//! A write-back loop done in one exception, against the same loop run pass
//! by pass as the CPU would.

use super::Values;
use super::emulation::{Pages, done, run_at, setup};
use super::encode::{Disp, Mem, insn, modrm, rex};
use crate::crash::gpr;
use windows_sys::Win32::System::Memory::PAGE_NOACCESS;

/// The passes one exception does at most: 16 MiB of lines.
const MAX_PASSES: u64 = 1 << 18;

/// How a generated loop decides to go round again.
#[derive(Debug, Clone, Copy)]
enum Again {
    /// JB rel8 (72): CF.
    Below,
    /// JBE rel8 (76): CF or ZF.
    BelowOrEqual,
    /// JNE rel8 (75): not ZF.
    NotEqual,
}

/// `hint [base]; add base, 0x40; cmp; jcc` back to the hint, the CMP with
/// `base` in r/m (CMP r/m64, r64: REX.W 39 /r) or in reg (CMP r64, r/m64:
/// REX.W 3B /r), both `base - limit` (APM3 CMP, ADD; Jcc). Returns the
/// bytes and the hint's length.
fn write_back_loop(digit: u8, base: u8, limit: u8, base_in_rm: bool, again: Again) -> (Vec<u8>, usize) {
    let hint = Mem::Base { base, disp: Disp::None }.encode(digit);
    let mut code = insn(&[0x66], (hint.b >= 8).then(|| rex(false, 0, 0, hint.b)), 0xae, &hint.bytes, &[]);
    let hint_len = code.len();
    // ADD r/m64, imm8: REX.W 83 /0 ib.
    code.extend([rex(true, 0, 0, base), 0x83, modrm(3, 0, base), 0x40]);
    if base_in_rm {
        code.extend([rex(true, limit, 0, base), 0x39, modrm(3, limit, base)]);
    } else {
        code.extend([rex(true, base, 0, limit), 0x3b, modrm(3, base, limit)]);
    }
    let jcc = match again {
        Again::Below => 0x72,
        Again::BelowOrEqual => 0x76,
        Again::NotEqual => 0x75,
    };
    let back = -(code.len() as i64 + 2);
    code.extend([jcc, back as i8 as u8]);
    (code, hint_len)
}

/// The status flags CMP leaves for `a - b` (OF, SF, ZF, AF, PF, CF), by
/// wide and signed arithmetic.
fn cmp_flags(a: u64, b: u64) -> u32 {
    let r = a.wrapping_sub(b);
    u32::from((a as u128) < (b as u128))
        | u32::from((r as u8).count_ones().is_multiple_of(2)) << 2
        | u32::from((a & 0xf) < (b & 0xf)) << 4
        | u32::from(r == 0) << 6
        | ((r >> 63) as u32) << 7
        | u32::from((a as i64).overflowing_sub(b as i64).1) << 11
}

const STATUS: u32 = 0x8d5;

/// Where one exception leaves the loop.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    /// Gone round `passes` times and out: base and flags.
    Left(u64, u32),
    /// Stopped at the hint before a line it may not touch, or at the most
    /// passes one exception does: base and the last CMP's flags.
    AtHint(u64, u32),
}

/// The loop pass by pass from `start`: the hint's line checked, then ADD,
/// CMP and the branch.
fn pass_by_pass(start: u64, limit: u64, again: Again, readable: impl Fn(u64) -> bool) -> Outcome {
    let (mut base, mut flags, mut passes) = (start, 0, 0);
    loop {
        if passes == MAX_PASSES || !readable(base) {
            assert!(passes > 0, "the first line is readable in these tests");
            return Outcome::AtHint(base, flags);
        }
        base = base.wrapping_add(64);
        flags = cmp_flags(base, limit);
        passes += 1;
        let round = match again {
            Again::Below => base < limit,
            Again::BelowOrEqual => base <= limit,
            Again::NotEqual => base != limit,
        };
        if !round {
            return Outcome::Left(base, flags);
        }
    }
}

/// Runs the loop's first hint through `emulate` and checks the context
/// against `want`.
fn check(code: &[u8], base: u8, limit: u8, start: u64, limit_value: u64, want: Outcome) {
    let host = setup();
    let page = Pages::new(1);
    page.put(0, code);
    let result = run_at(page.addr(0), host, |c| {
        *gpr(c, base) = start;
        *gpr(c, limit) = limit_value;
    });
    match want {
        Outcome::Left(end, flags) => done(result, code.len(), |c| {
            *gpr(c, base) = end;
            c.EFlags = c.EFlags & !STATUS | flags;
        }),
        Outcome::AtHint(end, flags) => done(result, 0, |c| {
            *gpr(c, base) = end;
            c.EFlags = c.EFlags & !STATUS | flags;
        }),
    }
}

#[test]
fn write_back_loops_run_in_one_exception() {
    let data = Pages::new(64);
    let mut values = Values::new(0x100b);
    for digit in [6, 7] {
        for base in 0..16 {
            for again in [Again::Below, Again::BelowOrEqual, Again::NotEqual] {
                for base_in_rm in [true, false] {
                    let limit = (base + 1 + (values.next() % 15) as u8) % 16;
                    let (code, _) = write_back_loop(digit, base, limit, base_in_rm, again);
                    let start = data.addr((values.next() % 0x10000) as usize & !63);
                    let limit_value = match again {
                        Again::NotEqual => start + 64 * (1 + values.next() % 2000),
                        _ => start + values.next() % 0x20000,
                    };
                    let want = pass_by_pass(start, limit_value, again, |_| true);
                    check(&code, base, limit, start, limit_value, want);
                }
            }
        }
    }
}

#[test]
fn a_loop_stops_before_a_line_it_may_not_touch() {
    // Page 5 of 8 has no access: the passes up to it are done; the hint
    // then stands at its first line, where it will be refused.
    let data = Pages::new(8);
    data.protect(5, PAGE_NOACCESS);
    let page5 = data.addr(0x5000);
    let readable = |line: u64| !(page5..page5 + 0x1000).contains(&line);
    for again in [Again::Below, Again::BelowOrEqual] {
        let (code, _) = write_back_loop(6, 1, 0, true, again);
        let (start, limit_value) = (data.addr(0x40), data.addr(0x7fc0));
        let want = pass_by_pass(start, limit_value, again, readable);
        assert_eq!(want, Outcome::AtHint(page5, cmp_flags(page5, limit_value)));
        check(&code, 1, 0, start, limit_value, want);
    }
}

#[test]
fn a_loop_longer_than_one_exception_goes_on_from_where_it_stopped() {
    let data = Pages::new((17 << 20) >> 12);
    let (code, _) = write_back_loop(7, 3, 9, false, Again::Below);
    let (start, limit_value) = (data.addr(0), data.addr((17 << 20) - 64));
    let want = pass_by_pass(start, limit_value, Again::Below, |_| true);
    assert_eq!(want, Outcome::AtHint(start + 64 * MAX_PASSES, cmp_flags(start + 64 * MAX_PASSES, limit_value)));
    check(&code, 3, 9, start, limit_value, want);
}

#[test]
fn a_loop_that_would_not_end_is_done_one_hint_at_a_time() {
    // JNE to a limit no multiple of 64 away: the CPU would go round until
    // the base wrapped. Only the hint is done; the loop runs on the CPU.
    let host = setup();
    let data = Pages::new(4);
    let (code, hint_len) = write_back_loop(6, 2, 5, true, Again::NotEqual);
    let page = Pages::new(1);
    page.put(0, &code);
    let start = data.addr(0x80);
    let result = run_at(page.addr(0), host, |c| {
        *gpr(c, 2) = start;
        *gpr(c, 5) = start + 0x1001;
    });
    done(result, hint_len, |_| {});
}
