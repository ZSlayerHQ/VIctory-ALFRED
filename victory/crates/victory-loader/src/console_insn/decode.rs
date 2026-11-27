//! Decodes the console instructions `emulate` performs, in 64-bit mode,
//! by looking the bytes up in a table of their encodings ([`ENCODINGS`]).
//!
//! Prefixes are read as APM3 1.2 gives them: at most one of each group
//! (1.2.1: more is undefined), the CS, DS, ES and SS overrides as null
//! prefixes (1.2.4), LOCK refused (#UD with any of these, 1.2.5), one REX
//! right before the 0F escape (1.2.7), 15 bytes in all (1.1.2). F2 or F3
//! picks the opcode map's row whether 66 is there or not: the manuals give
//! no row for the pair (APM3 Table A-4; SDM Vol. 2A 2.1.1 calls other uses
//! of 66 reserved), and the AMD CPU this was checked on runs `F2 66 0F 79`
//! as INSERTQ and faults on `66 F3 0F 79` (`emulation` tests). 66 picks a
//! row only where the map has one. A prefix that changes nothing for an
//! instruction is accepted and ignored: 66 where there is no operand size,
//! 67 and the segment overrides where nothing is addressed, REX where it
//! names no extended register or would extend an opcode digit (1.4.4).

/// A general register as instructions number it, 0-15.
pub(super) type Gpr = u8;

/// The segment whose base an address adds (APM3 1.2.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Segment {
    /// No base: DS by default, and the CS, DS, ES and SS overrides, are
    /// flat in 64-bit mode.
    Flat,
    Fs,
    Gs,
}

/// A memory operand as the instruction names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MemoryOperand {
    pub(super) base: Option<Gpr>,
    /// The index register and its scale (1, 2, 4 or 8).
    pub(super) index: Option<(Gpr, u8)>,
    pub(super) displacement: i32,
    /// Relative to the next instruction's address (APM3 1.7).
    pub(super) rip_relative: bool,
    /// A 32-bit address size (67): the sum truncated to 32 bits.
    pub(super) address32: bool,
    pub(super) segment: Segment,
}

/// One of the instructions, with its operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Op {
    /// Zeroes the 64-byte line holding the address, [rAX] (or [EAX]).
    Clzero(MemoryOperand),
    Clwb(MemoryOperand),
    Clflushopt(MemoryOperand),
    /// Arms a monitor at [rAX] (or [EAX]).
    Monitorx(MemoryOperand),
    Mwaitx,
    Rdpru,
    /// Reads TSC_AUX into this general register.
    Rdpid { reg: Gpr },
    /// Extracts a field of XMM register `dst`'s low quadword into it.
    Extrq { dst: u8, field: Field },
    /// Inserts XMM register `src`'s low bits into `dst`'s low quadword.
    Insertq { dst: u8, src: u8, field: Field },
}

/// Where an `extrq` or `insertq` takes its field's length and index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Field {
    /// The two immediate bytes as encoded: the first the length, the
    /// second the index.
    Immediate { length: u8, index: u8 },
    /// This XMM register: `extrq` reads bits 5:0 and 13:8, `insertq` bits
    /// 69:64 and 77:72.
    Register(u8),
}

/// A decoded instruction and its length in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Decoded {
    pub(super) op: Op,
    pub(super) len: u8,
}

/// Why `decode` did not decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Refusal {
    /// Not one of the instructions, or an encoding the manuals leave
    /// undefined, reserved or faulting.
    NotOurs,
    /// The bytes ended before the instruction could be told apart: more
    /// may be one of ours.
    Truncated,
}

/// A row of the 0F opcode map, as the prefixes select it (APM3 1.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Row {
    Plain,
    With66,
    WithF2,
    WithF3,
}

/// What an encoding's ModRM byte holds (APM3 Appendix A, pp. 524-525).
#[derive(Debug, Clone, Copy)]
enum ModRmUse {
    /// The whole byte: mod 11, reg the /digit, r/m naming the instruction
    /// (Table A-8).
    Whole(u8),
    /// mod 11, reg the /digit, r/m a register ("R", "U").
    RegisterInRm { digit: u8 },
    /// mod 11, reg and r/m both registers ("V", "U").
    RegisterPair,
    /// mod not 11, reg the /digit, r/m with SIB and displacement a memory
    /// operand ("M").
    Memory { digit: u8 },
}

impl ModRmUse {
    fn admits(self, modrm: u8) -> bool {
        let (mode, reg) = (modrm >> 6, modrm >> 3 & 7);
        match self {
            ModRmUse::Whole(byte) => modrm == byte,
            ModRmUse::RegisterInRm { digit } => mode == 3 && reg == digit,
            ModRmUse::RegisterPair => mode == 3,
            ModRmUse::Memory { digit } => mode != 3 && reg == digit,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Mnemonic {
    Monitorx,
    Mwaitx,
    Clzero,
    Rdpru,
    Clwb,
    Clflushopt,
    Rdpid,
    Extrq,
    Insertq,
}

/// An encoding: 0F `opcode` in `row`, what its ModRM holds, and how many
/// immediate bytes follow the ModRM byte and any memory operand.
struct Encoding {
    opcode: u8,
    row: Row,
    modrm: ModRmUse,
    immediates: usize,
    mnemonic: Mnemonic,
}

const fn encoding(opcode: u8, row: Row, modrm: ModRmUse, immediates: usize, mnemonic: Mnemonic) -> Encoding {
    Encoding { opcode, row, modrm, immediates, mnemonic }
}

/// The instructions' encodings, from the manuals.
const ENCODINGS: [Encoding; 11] = [
    // APM3 Table A-8 (p. 540), 0F 01 row /7, prefix "none": MONITORX 0F 01
    // FA (p. 238), MWAITX FB (p. 268), CLZERO FC (p. 156), RDPRU FD (p. 308).
    encoding(0x01, Row::Plain, ModRmUse::Whole(0xfa), 0, Mnemonic::Monitorx),
    encoding(0x01, Row::Plain, ModRmUse::Whole(0xfb), 0, Mnemonic::Mwaitx),
    encoding(0x01, Row::Plain, ModRmUse::Whole(0xfc), 0, Mnemonic::Clzero),
    encoding(0x01, Row::Plain, ModRmUse::Whole(0xfd), 0, Mnemonic::Rdpru),
    // Table A-7, Group 15, row 66, note 6 (mod not 11): CLWB 66 0F AE /6
    // (p. 154; SDM CLWB), CLFLUSHOPT 66 0F AE /7 (p. 152; SDM CLFLUSHOPT,
    // "NFx": no F2 or F3).
    encoding(0xae, Row::With66, ModRmUse::Memory { digit: 6 }, 0, Mnemonic::Clwb),
    encoding(0xae, Row::With66, ModRmUse::Memory { digit: 7 }, 0, Mnemonic::Clflushopt),
    // Table A-7, Group 9, row F3, "RDPID Rd/q": F3 0F C7 /7 (p. 307; SDM
    // RDPID, "ModRM.MOD = 011B required").
    encoding(0xc7, Row::WithF3, ModRmUse::RegisterInRm { digit: 7 }, 0, Mnemonic::Rdpid),
    // Table A-4, row 7: 66 0F 78 is Group 17, whose /0 is EXTRQ xmm, imm8,
    // imm8 ("/0": the xmm is r/m, APM3 2.5.2); 66 0F 79 EXTRQ xmm1, xmm2
    // (APM4 p. 173); F2 0F 78 INSERTQ xmm1, xmm2, imm8, imm8 and F2 0F 79
    // INSERTQ xmm1, xmm2 (APM4 p. 196).
    encoding(0x78, Row::With66, ModRmUse::RegisterInRm { digit: 0 }, 2, Mnemonic::Extrq),
    encoding(0x79, Row::With66, ModRmUse::RegisterPair, 0, Mnemonic::Extrq),
    encoding(0x78, Row::WithF2, ModRmUse::RegisterPair, 2, Mnemonic::Insertq),
    encoding(0x79, Row::WithF2, ModRmUse::RegisterPair, 0, Mnemonic::Insertq),
];

/// The opcodes among ours whose map has a row for 66 (Tables A-4, A-7):
/// there 66 picks the row; elsewhere it is an operand size.
const HAS_66_ROW: [u8; 4] = [0x78, 0x79, 0xae, 0xc7];

/// Displacement bytes, and whether a base register is used, by ModRM.mod
/// (00, 01, 10) and by whether the base field (r/m, or SIB.base under r/m
/// 100) is 101 (APM3 Tables 1-10 note 4, 1-12 note 2, 1-13, 1-16).
const DISPLACEMENT: [[(usize, bool); 2]; 3] = [
    // mod 00: none; base 101 is no base, with a disp32 (RIP-relative when
    // it is r/m, 1.7.1).
    [(0, true), (4, false)],
    // mod 01: disp8.
    [(1, true), (1, true)],
    // mod 10: disp32.
    [(4, true), (4, true)],
];

/// SIB.scale's factors (APM3 Table 1-11).
const SCALE: [u8; 4] = [1, 2, 4, 8];

/// The longest instruction (APM3 1.1.2).
const MAX_LEN: usize = 15;

/// REX's extension bits for ModRM.reg, SIB.index, and r/m or SIB.base
/// (APM3 Table 1-15).
const REX_R: u8 = 2;
const REX_X: u8 = 1;
const REX_B: u8 = 0;

/// The byte at `i`: `Truncated` past the bytes there are, `NotOurs` past 15.
fn byte(code: &[u8], i: usize) -> Result<u8, Refusal> {
    if i >= MAX_LEN {
        return Err(Refusal::NotOurs);
    }
    code.get(i).copied().ok_or(Refusal::Truncated)
}

/// A legacy prefix's group (APM3 Table 1-1).
enum Group {
    OperandSize,
    AddressSize,
    Segment(Segment),
    /// CS, DS, ES, SS: no group's in 64-bit mode (1.2.4).
    Null,
    Lock,
    Repeat(Row),
}

fn group_of(byte: u8) -> Option<Group> {
    Some(match byte {
        0x66 => Group::OperandSize,
        0x67 => Group::AddressSize,
        0x2e | 0x36 | 0x3e | 0x26 => Group::Null,
        0x64 => Group::Segment(Segment::Fs),
        0x65 => Group::Segment(Segment::Gs),
        0xf0 => Group::Lock,
        0xf2 => Group::Repeat(Row::WithF2),
        0xf3 => Group::Repeat(Row::WithF3),
        _ => return None,
    })
}

/// The prefixes in front of an instruction.
#[derive(Default)]
struct Prefixes {
    operand_size: bool,
    address_size: bool,
    segment: Option<Segment>,
    repeat: Option<Row>,
    rex: u8,
}

impl Prefixes {
    /// Reads them up to the 0F escape; returns them and where it is.
    fn read(code: &[u8]) -> Result<(Self, usize), Refusal> {
        let mut seen = Self::default();
        let mut at = 0;
        loop {
            let b = byte(code, at)?;
            if b == 0x0f {
                return Ok((seen, at));
            }
            if b & 0xf0 == 0x40 {
                seen.rex = b;
                return match byte(code, at + 1)? {
                    0x0f => Ok((seen, at + 1)),
                    _ => Err(Refusal::NotOurs),
                };
            }
            let twice = match group_of(b).ok_or(Refusal::NotOurs)? {
                Group::Null => false,
                Group::Lock => true,
                Group::OperandSize => std::mem::replace(&mut seen.operand_size, true),
                Group::AddressSize => std::mem::replace(&mut seen.address_size, true),
                Group::Segment(s) => seen.segment.replace(s).is_some(),
                Group::Repeat(row) => seen.repeat.replace(row).is_some(),
            };
            if twice {
                return Err(Refusal::NotOurs);
            }
            at += 1;
        }
    }

    fn row(&self, opcode: u8) -> Row {
        match self.repeat {
            Some(row) => row,
            None if self.operand_size && HAS_66_ROW.contains(&opcode) => Row::With66,
            None => Row::Plain,
        }
    }

    fn segment(&self) -> Segment {
        self.segment.unwrap_or(Segment::Flat)
    }

    /// A 3-bit register field with REX's extension bit `bit` in front.
    fn extend(&self, field: u8, bit: u8) -> u8 {
        field & 7 | (self.rex >> bit & 1) << 3
    }
}

/// Decodes the instruction at the start of `code`, the bytes that could be
/// read (up to 15).
pub(super) fn decode(code: &[u8]) -> Result<Decoded, Refusal> {
    let (prefixes, escape) = Prefixes::read(code)?;
    let opcode = byte(code, escape + 1)?;
    if !ENCODINGS.iter().any(|e| e.opcode == opcode) {
        return Err(Refusal::NotOurs);
    }
    let row = prefixes.row(opcode);
    let modrm = byte(code, escape + 2)?;
    let encoding = ENCODINGS
        .iter()
        .find(|e| e.opcode == opcode && e.row == row && e.modrm.admits(modrm))
        .ok_or(Refusal::NotOurs)?;
    let mut end = escape + 3;
    let memory = match encoding.modrm {
        ModRmUse::Memory { .. } => {
            let (operand, after) = memory_operand(code, end, modrm, &prefixes)?;
            end = after;
            Some(operand)
        }
        _ => None,
    };
    let mut immediates = [0; 2];
    for (i, slot) in immediates.iter_mut().enumerate().take(encoding.immediates) {
        *slot = byte(code, end + i)?;
    }
    end += encoding.immediates;
    let op = operands(encoding, modrm, memory, immediates, &prefixes).ok_or(Refusal::NotOurs)?;
    Ok(Decoded { op, len: end as u8 })
}

/// The instruction `encoding` names, with its operands.
fn operands(
    encoding: &Encoding,
    modrm: u8,
    memory: Option<MemoryOperand>,
    immediates: [u8; 2],
    p: &Prefixes,
) -> Option<Op> {
    let (reg, rm) = (p.extend(modrm >> 3, REX_R), p.extend(modrm, REX_B));
    // [rAX], or [EAX] with a 32-bit address size (APM3 pp. 156, 238; 1.2.3).
    let at_rax = MemoryOperand {
        base: Some(0),
        index: None,
        displacement: 0,
        rip_relative: false,
        address32: p.address_size,
        segment: p.segment(),
    };
    let field = match encoding.immediates {
        2 => Field::Immediate { length: immediates[0], index: immediates[1] },
        _ => Field::Register(rm),
    };
    Some(match encoding.mnemonic {
        Mnemonic::Monitorx => Op::Monitorx(at_rax),
        Mnemonic::Mwaitx => Op::Mwaitx,
        Mnemonic::Clzero => Op::Clzero(at_rax),
        Mnemonic::Rdpru => Op::Rdpru,
        Mnemonic::Clwb => Op::Clwb(memory?),
        Mnemonic::Clflushopt => Op::Clflushopt(memory?),
        Mnemonic::Rdpid => Op::Rdpid { reg: rm },
        // The immediate form's one register is r/m; the other's
        // destination is reg.
        Mnemonic::Extrq => match encoding.modrm {
            ModRmUse::RegisterInRm { .. } => Op::Extrq { dst: rm, field },
            _ => Op::Extrq { dst: reg, field },
        },
        Mnemonic::Insertq => Op::Insertq { dst: reg, src: rm, field },
    })
}

/// The memory operand of ModRM byte `modrm` (mod not 11), its SIB and
/// displacement from `at`; returns it and where it ends.
fn memory_operand(
    code: &[u8],
    at: usize,
    modrm: u8,
    p: &Prefixes,
) -> Result<(MemoryOperand, usize), Refusal> {
    let mode = usize::from(modrm >> 6);
    let rm = modrm & 7;
    // r/m 100: a SIB names base, index and scale; index 100 (with REX.X
    // clear) is none (Table 1-12, note 1; Table 1-17).
    let (mut at, base_field, index) = if rm == 4 {
        let sib = byte(code, at)?;
        let index = p.extend(sib >> 3, REX_X);
        (at + 1, sib & 7, (index != 4).then(|| (index, SCALE[usize::from(sib >> 6)])))
    } else {
        (at, rm, None)
    };
    let (size, has_base) = DISPLACEMENT[mode][usize::from(base_field == 5)];
    let mut raw = [0; 4];
    for (i, slot) in raw.iter_mut().enumerate().take(size) {
        *slot = byte(code, at + i)?;
    }
    at += size;
    let displacement = match size {
        1 => i32::from(raw[0] as i8),
        _ => i32::from_le_bytes(raw),
    };
    let operand = MemoryOperand {
        base: has_base.then(|| p.extend(base_field, REX_B)),
        index,
        displacement,
        // mod 00, r/m 101 itself (not under a SIB) is RIP-relative, REX.B
        // or not (1.7.1, 1.7.2).
        rip_relative: rm == 5 && !has_base,
        address32: p.address_size,
        segment: p.segment(),
    };
    Ok((operand, at))
}
