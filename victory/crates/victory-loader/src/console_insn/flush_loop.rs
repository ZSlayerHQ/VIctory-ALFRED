//! A write-back loop over a buffer, done in one exception.
//!
//! A title writes back whole buffers a line at a time with a loop like
//!
//! ```text
//! top: clwb [base]          (or clflushopt)
//!      add  base, 0x40
//!      cmp  base, limit     (CMP r/m64, r64 or CMP r64, r/m64)
//!      jb / jbe / jne top
//! ```
//!
//! Where the PC lacks the hint, each pass is an exception (about a
//! microsecond against a line's few nanoseconds). The hint changes nothing
//! on the PC but its check that the line can be read, so the loop is done
//! in one: the lines' pages are checked (one read each), the base goes to
//! where the loop leaves it, and the flags are those of the last CMP. It
//! stops early at a page it may not read (the hint then stands at that
//! page's first line, where it is refused) or after [`MAX_PASSES`].

use super::decode::Gpr;

/// The passes one exception does at most: 16 MiB of lines, 4,096 page
/// checks.
pub(super) const MAX_PASSES: u64 = 1 << 18;

/// The status flags (OF, SF, ZF, AF, PF, CF) in EFLAGS.
pub(super) const STATUS_FLAGS: u32 = 0x8d5;

/// When the loop goes round again: the branch at its end (APM3 Jcc,
/// p. 203; Table A-5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GoesRound {
    /// JB rel8, 72: while CF.
    Below,
    /// JBE rel8, 76: while CF or ZF.
    BelowOrEqual,
    /// JNE rel8, 75: while not ZF.
    NotEqual,
}

/// A recognised loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WriteBackLoop {
    /// The register holding the limit.
    pub(super) limit: Gpr,
    goes_round: GoesRound,
    /// Bytes from the hint to past the branch.
    pub(super) len: usize,
}

/// The loop whose hint, on the register `base` alone, takes `hint_len`
/// bytes at the start of `code`, if the bytes after it are the loop's ADD,
/// CMP and branch back to the hint.
pub(super) fn recognise(code: &[u8], hint_len: usize, base: Gpr) -> Option<WriteBackLoop> {
    let rest = code.get(hint_len..)?;
    // ADD r/m64, imm8 (APM3 p. 89): REX.W 83 /0 ib, r/m the base, 0x40.
    let &[add_rex, 0x83, add_modrm, 0x40, ..] = rest else { return None };
    if !is_rex_w(add_rex) || add_modrm >> 3 != 0b11_000 || register(add_modrm, add_rex, false) != base {
        return None;
    }
    // CMP (APM3 p. 162), "as the SUB instruction": REX.W 39 /r is r/m - reg,
    // REX.W 3B /r is reg - r/m; the base must be the first operand.
    let &[cmp_rex, cmp_opcode, cmp_modrm, ..] = rest.get(4..)? else { return None };
    if !is_rex_w(cmp_rex) || cmp_modrm >> 6 != 3 {
        return None;
    }
    let (in_reg, in_rm) = (register(cmp_modrm, cmp_rex, true), register(cmp_modrm, cmp_rex, false));
    let limit = match cmp_opcode {
        0x39 if in_rm == base => in_reg,
        0x3b if in_reg == base => in_rm,
        _ => return None,
    };
    if limit == base {
        return None;
    }
    // Jcc rel8 back to the hint.
    let &[jcc, displacement, ..] = rest.get(7..)? else { return None };
    let goes_round = match jcc {
        0x72 => GoesRound::Below,
        0x76 => GoesRound::BelowOrEqual,
        0x75 => GoesRound::NotEqual,
        _ => return None,
    };
    let len = hint_len + 9;
    (i64::from(displacement as i8) == -(len as i64)).then_some(WriteBackLoop { limit, goes_round, len })
}

/// A REX with W set (APM3 1.2.7): a 64-bit operand.
fn is_rex_w(byte: u8) -> bool {
    byte & 0xf8 == 0x48
}

/// ModRM's reg (`reg` true) or r/m register, with REX's extension.
fn register(modrm: u8, rex: u8, reg: bool) -> Gpr {
    if reg { modrm >> 3 & 7 | (rex >> 2 & 1) << 3 } else { modrm & 7 | (rex & 1) << 3 }
}

impl WriteBackLoop {
    /// How many passes the loop makes from `start` before it leaves; `None`
    /// where it would not leave before the base wrapped round.
    pub(super) fn passes(self, start: u64, limit: u64) -> Option<u64> {
        let after_one = start.checked_add(64)?;
        let passes = match self.goes_round {
            // Round while base < limit: the first pass that reaches it.
            GoesRound::Below if after_one >= limit => 1,
            GoesRound::Below => (limit - start).div_ceil(64),
            // Round while base <= limit: the first pass beyond it.
            GoesRound::BelowOrEqual if after_one > limit => 1,
            GoesRound::BelowOrEqual => (limit - start) / 64 + 1,
            // Round while base != limit: it must land on it.
            GoesRound::NotEqual if limit > start && (limit - start).is_multiple_of(64) => (limit - start) / 64,
            GoesRound::NotEqual => return None,
        };
        start.checked_add(passes.checked_mul(64)?)?;
        Some(passes)
    }
}

/// The status flags CMP leaves for `a - b` (APM3 p. 162, "in the same
/// manner as the SUB instruction"): CF the borrow, PF an even count of ones
/// in the result's low byte, AF the borrow out of bit 3, ZF a zero result,
/// SF its top bit, OF the signed overflow.
pub(super) fn compare_flags(a: u64, b: u64) -> u32 {
    let (difference, borrow) = a.overflowing_sub(b);
    let signed_overflow = (a as i64).checked_sub(b as i64).is_none();
    let flags = [
        (borrow, 1 << 0),
        ((difference as u8).count_ones().is_multiple_of(2), 1 << 2),
        (a & 0xf < b & 0xf, 1 << 4),
        (difference == 0, 1 << 6),
        (difference >> 63 == 1, 1 << 7),
        (signed_overflow, 1 << 11),
    ];
    flags.iter().filter(|(set, _)| *set).fold(0, |all, (_, bit)| all | bit)
}
