//! A small x86-64 encoder: instructions built from their operand fields as
//! APM3 lays the bytes out (1.2 prefixes, 1.4 ModRM and SIB, 1.5
//! displacements, 1.7 RIP-relative), so the decoder is checked against
//! bytes it did not choose and no vector is picked by hand.

use super::super::decode::{MemoryOperand, Segment};

/// REX = 0100WRXB (APM3 1.2.7, Figure 1-3); `r`, `x` and `b` are full
/// register numbers, of which REX keeps bit 3.
pub(super) fn rex(w: bool, r: u8, x: u8, b: u8) -> u8 {
    0x40 | u8::from(w) << 3 | (r >> 3 & 1) << 2 | (x >> 3 & 1) << 1 | (b >> 3 & 1)
}

/// ModRM = mod (2 bits), reg (3), r/m (3) (APM3 1.4.1).
pub(super) fn modrm(mode: u8, reg: u8, rm: u8) -> u8 {
    assert!(mode < 4);
    mode << 6 | (reg & 7) << 3 | (rm & 7)
}

/// SIB = scale (2 bits: 1, 2, 4, 8), index (3), base (3) (APM3 1.4.2,
/// Table 1-11).
pub(super) fn sib(scale: u8, index: u8, base: u8) -> u8 {
    let scale = [1, 2, 4, 8].iter().position(|&s| s == scale).expect("a scale") as u8;
    scale << 6 | (index & 7) << 3 | (base & 7)
}

/// A displacement as an operand asks for it.
#[derive(Debug, Clone, Copy)]
pub(super) enum Disp {
    None,
    Byte(i8),
    Dword(i32),
}

impl Disp {
    fn value(self) -> i32 {
        match self {
            Disp::None => 0,
            Disp::Byte(d) => i32::from(d),
            Disp::Dword(d) => d,
        }
    }
}

/// A memory operand in its own terms.
#[derive(Debug, Clone, Copy)]
pub(super) enum Mem {
    /// [base + disp].
    Base { base: u8, disp: Disp },
    /// [base + index * scale + disp]; the index is not rsp.
    Indexed { base: u8, index: u8, scale: u8, disp: Disp },
    /// [index * scale + disp32], or [disp32] with no index.
    NoBase { index: Option<(u8, u8)>, disp: i32 },
    /// [rip + disp32].
    Rip(i32),
}

/// The encoded form of a memory operand.
pub(super) struct Encoded {
    /// ModRM, then SIB and displacement as needed.
    pub(super) bytes: Vec<u8>,
    /// The register numbers whose bit 3 REX.X and REX.B must carry.
    pub(super) x: u8,
    pub(super) b: u8,
}

impl Mem {
    /// Encodes the operand with `reg` in ModRM.reg.
    pub(super) fn encode(self, reg: u8) -> Encoded {
        // mod by displacement (Table 1-10, note 2). A base of rbp or r13 has
        // no form without one: it takes a zero disp8 (Table 1-17).
        let fit = |base: u8, disp: Disp| match disp {
            Disp::None if base & 7 == 5 => Disp::Byte(0),
            d => d,
        };
        let mode = |disp: Disp| match disp {
            Disp::None => 0,
            Disp::Byte(_) => 1,
            Disp::Dword(_) => 2,
        };
        let tail = |disp: Disp| match disp {
            Disp::None => vec![],
            Disp::Byte(d) => vec![d as u8],
            Disp::Dword(d) => d.to_le_bytes().to_vec(),
        };
        let mut bytes;
        let (x, b);
        match self {
            Mem::Base { base, disp } => {
                let disp = fit(base, disp);
                if base & 7 == 4 {
                    // rsp and r12 need a SIB with no index (Table 1-17).
                    bytes = vec![modrm(mode(disp), reg, 4), sib(1, 4, base)];
                } else {
                    bytes = vec![modrm(mode(disp), reg, base)];
                }
                bytes.extend(tail(disp));
                (x, b) = (0, base);
            }
            Mem::Indexed { base, index, scale, disp } => {
                assert_ne!(index, 4, "rsp is no index");
                let disp = fit(base, disp);
                bytes = vec![modrm(mode(disp), reg, 4), sib(scale, index, base)];
                bytes.extend(tail(disp));
                (x, b) = (index, base);
            }
            Mem::NoBase { index, disp } => {
                // mod 00, SIB base 101: no base, disp32 (Table 1-16).
                let (index, scale) = index.unwrap_or((4, 1));
                bytes = vec![modrm(0, reg, 4), sib(scale, index, 5)];
                bytes.extend(disp.to_le_bytes());
                (x, b) = (index, 0);
            }
            Mem::Rip(disp) => {
                // mod 00, r/m 101 (1.7.1).
                bytes = vec![modrm(0, reg, 5)];
                bytes.extend(disp.to_le_bytes());
                (x, b) = (0, 0);
            }
        }
        Encoded { bytes, x, b }
    }

    /// What the decoder should report for it.
    pub(super) fn expected(self, address32: bool, segment: Segment) -> MemoryOperand {
        let (base, index, displacement, rip_relative) = match self {
            Mem::Base { base, disp } => (Some(base), None, disp.value(), false),
            Mem::Indexed { base, index, scale, disp } => {
                (Some(base), Some((index, scale)), disp.value(), false)
            }
            Mem::NoBase { index, disp } => (None, index, disp, false),
            Mem::Rip(disp) => (None, None, disp, true),
        };
        MemoryOperand { base, index, displacement, rip_relative, address32, segment }
    }
}

/// An instruction in the 0F map: legacy prefixes, an optional REX, 0F, the
/// opcode, then `tail` (ModRM and what follows) and immediates.
pub(super) fn insn(prefixes: &[u8], rex: Option<u8>, opcode: u8, tail: &[u8], imm: &[u8]) -> Vec<u8> {
    let mut bytes = prefixes.to_vec();
    bytes.extend(rex);
    bytes.extend([0x0f, opcode]);
    bytes.extend_from_slice(tail);
    bytes.extend_from_slice(imm);
    bytes
}

/// Every memory form, over all bases and indexes, with displacements of
/// each size and sign.
pub(super) fn memory_forms() -> Vec<Mem> {
    let mut forms = vec![];
    let disps = [Disp::None, Disp::Byte(-0x80), Disp::Byte(0x7f), Disp::Dword(i32::MIN), Disp::Dword(0x7654_3210)];
    for base in 0..16 {
        for disp in disps {
            forms.push(Mem::Base { base, disp });
        }
        for index in (0..16).filter(|&i| i != 4) {
            for scale in [1, 2, 4, 8] {
                for disp in [Disp::None, Disp::Byte(-8), Disp::Dword(0x1000)] {
                    forms.push(Mem::Indexed { base, index, scale, disp });
                }
            }
        }
    }
    for index in (0..16).filter(|&i| i != 4) {
        for scale in [1, 2, 4, 8] {
            forms.push(Mem::NoBase { index: Some((index, scale)), disp: -0x40 });
        }
    }
    forms.push(Mem::NoBase { index: None, disp: 0x0040_0000 });
    forms.push(Mem::Rip(i32::MIN));
    forms.push(Mem::Rip(0x10));
    forms
}
