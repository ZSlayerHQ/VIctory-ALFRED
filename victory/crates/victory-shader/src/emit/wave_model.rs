//! A CPU model of one console wave of 64 lanes (GFX10 in wave64) or 32 (wave32), the reference
//! the translation of a console wave is checked against (`wave_model_tests`): into two host waves
//! of 32, or into one host wave of its own size.
//!
//! In wave32 a lane mask is one dword: EXEC and VCC are `exec_lo` and `vcc_lo`, mask writes
//! (compares, saveexec) write that dword alone, and the `_b32` mask instructions are the ones
//! the compiler uses.
//!
//! It decodes its own words and depends on nothing else in the crate: it is the console as
//! the RDNA ISA reference states it, not as the translator reads it. Every lane runs every
//! instruction in step; EXEC says which lanes a vector instruction writes, and a lane
//! outside EXEC keeps its old value ("masked writes keep old values"). A lane that has left
//! the program is a lane whose EXEC bit was cleared before an `s_cbranch_execz` to the end:
//! the console keeps no other record of it, and a later `s_or_saveexec_b64 ..., -1` runs it
//! again with what its registers hold.
//!
//! Only the instructions the fixtures use are modelled; any other word stops the model
//! with its encoding and opcode, as does a cross-lane read whose semantics the model does
//! not state (a DPP or permlane source lane outside EXEC).
//!
//! The assembler below writes the same encodings, so a fixture's program and its reference
//! come from one list of words.

/// Lanes of a console wave in wave64 (the most a wave has: the model's VGPR width).
pub(super) const LANES: usize = 64;

/// Scalar operand codes.
const VCC: usize = 106;
const EXEC: usize = 126;
/// `null`: reads as 0, writes are dropped.
const NULL: usize = 125;
/// A 32-bit literal follows the instruction.
const LITERAL: u32 = 255;
/// VALU source codes of the DPP forms (the DPP word follows).
const DPP8: u32 = 233;
const DPP8_FI: u32 = 234;
const DPP16: u32 = 250;

/// One console wave: its lane count (32 or 64), 128 scalar dwords (VCC at 106/107, EXEC at
/// 126/127), SCC, 256 VGPRs of 64 lanes (a wave32 uses the first 32), and the raw UAV the
/// program stores to, in dwords.
pub(super) struct Wave {
    lanes: usize,
    s: [u32; 128],
    scc: bool,
    v: Vec<[u32; LANES]>,
    pub mem: Vec<u32>,
}

/// A source operand as it arrives: a 9-bit VALU code (256 up: VGPRs) or an 8-bit SALU one.
#[derive(Clone, Copy)]
struct Src {
    code: u32,
    literal: u32,
}

impl Wave {
    /// The wave of `lanes` lanes (32 or 64) at the program's start: the root constants in s0
    /// up, v0 the thread index (one wave of that many threads, so the lane), every lane in
    /// EXEC, every other register 0.
    pub(super) fn with_lanes(constants: &[u32], mem_dwords: usize, lanes: usize) -> Self {
        assert!(lanes == 32 || lanes == 64, "a wave of 32 or 64 lanes");
        let mut s = [0u32; 128];
        s[..constants.len()].copy_from_slice(constants);
        s[EXEC] = u32::MAX;
        s[EXEC + 1] = if lanes == 64 { u32::MAX } else { 0 };
        let mut v = vec![[0u32; LANES]; 256];
        for (l, x) in v[0].iter_mut().enumerate() {
            *x = l as u32;
        }
        Self { lanes, s, scc: false, v, mem: vec![0; mem_dwords] }
    }

    fn exec(&self) -> u64 {
        self.mask(EXEC)
    }

    /// The lane mask at `code`: a register pair in wave64, its low dword in wave32.
    fn mask(&self, code: usize) -> u64 {
        let hi = if self.lanes == 64 { u64::from(self.s[code + 1]) << 32 } else { 0 };
        u64::from(self.s[code]) | hi
    }

    /// Writes the lane mask `bits` to `code` (one dword in wave32).
    fn set_mask(&mut self, code: usize, bits: u64) {
        if self.lanes == 64 {
            self.set64(code, bits);
        } else {
            self.set32(code, bits as u32);
        }
    }

    fn set64(&mut self, code: usize, x: u64) {
        if code != NULL {
            self.s[code] = x as u32;
            self.s[code + 1] = (x >> 32) as u32;
        }
    }

    fn set32(&mut self, code: usize, x: u32) {
        if code != NULL {
            self.s[code] = x;
        }
    }

    /// A scalar source, 32 bits.
    fn sread(&self, src: Src) -> u32 {
        match src.code {
            0..=105 | 106 | 107 | 124 | 126 | 127 => self.s[src.code as usize],
            125 => 0,
            128..=192 => src.code - 128,
            193..=208 => (-((src.code - 192) as i32)) as u32,
            LITERAL => src.literal,
            c => panic!("the model has no scalar source {c}"),
        }
    }

    /// A scalar source, 64 bits: a register pair, or an integer constant sign-extended.
    fn sread64(&self, src: Src) -> u64 {
        match src.code {
            0..=104 | 106 | 126 => u64::from(self.s[src.code as usize]) | u64::from(self.s[src.code as usize + 1]) << 32,
            125 => 0,
            128..=208 => i64::from(self.sread(src) as i32) as u64,
            LITERAL => u64::from(src.literal),
            c => panic!("the model has no 64-bit scalar source {c}"),
        }
    }

    /// A VALU source in lane `l`.
    fn vread(&self, src: Src, l: usize) -> u32 {
        match src.code {
            256..=511 => self.v[src.code as usize - 256][l],
            _ => self.sread(src),
        }
    }

    /// Writes `f(lane)` to VGPR `d` in the lanes of EXEC; the others keep their value.
    fn vwrite(&mut self, d: usize, f: impl Fn(&Self, usize) -> u32) {
        self.vwrite_but(d, f, 0);
    }

    /// [`Self::vwrite`], the lanes of `keep` left as they are too (a DPP16 source past the
    /// row without `bound_ctrl` writes nothing).
    fn vwrite_but(&mut self, d: usize, f: impl Fn(&Self, usize) -> u32, keep: u64) {
        let exec = self.exec() & !keep;
        let new: Vec<u32> = (0..self.lanes).map(|l| f(self, l)).collect();
        for (l, x) in new.into_iter().enumerate() {
            if exec >> l & 1 == 1 {
                self.v[d][l] = x;
            }
        }
    }

    /// Runs `words` from the first to `s_endpgm`.
    pub(super) fn run(&mut self, words: &[u32]) {
        let mut pc = 0usize;
        for _ in 0..100_000 {
            let w = *words.get(pc).unwrap_or_else(|| panic!("the program runs past its end at word {pc}"));
            let word = |i: usize| words.get(pc + i).copied().unwrap_or(0);
            let src = |code: u32, at: usize| Src { code, literal: word(at) };
            // (top bits as in the RDNA ISA reference's encoding table)
            if w >> 23 == 0x17F {
                // SOPP
                let op = w >> 16 & 0x7F;
                let simm = w as u16 as i16 as isize;
                let target = (pc as isize + 1 + simm) as usize;
                let take = match op {
                    1 => return, // s_endpgm
                    2 => true,
                    4 => !self.scc,
                    5 => self.scc,
                    6 => self.mask(VCC) == 0,
                    7 => self.mask(VCC) != 0,
                    8 => self.exec() == 0,
                    9 => self.exec() != 0,
                    _ => panic!("the model has no SOPP opcode {op}"),
                };
                pc = if take { target } else { pc + 1 };
                continue;
            }
            let len;
            if w >> 23 == 0x17E {
                // SOPC
                let op = w >> 16 & 0x7F;
                let (a, b) = (src(w & 0xFF, 1), src(w >> 8 & 0xFF, 1));
                len = 1 + usize::from(a.code == LITERAL || b.code == LITERAL);
                let (x, y) = (self.sread(a), self.sread(b));
                self.scc = match op {
                    6 => x == y,
                    7 => x != y,
                    8 => x > y,
                    9 => x >= y,
                    10 => x < y,
                    11 => x <= y,
                    _ => panic!("the model has no SOPC opcode {op}"),
                };
            } else if w >> 23 == 0x17D {
                // SOP1
                let op = w >> 8 & 0xFF;
                let d = (w >> 16 & 0x7F) as usize;
                let a = src(w & 0xFF, 1);
                len = 1 + usize::from(a.code == LITERAL);
                match op {
                    3 => self.set32(d, self.sread(a)),
                    4 => self.set64(d, self.sread64(a)),
                    0x10 => {
                        // s_bcnt1_i32_b64
                        let n = self.sread64(a).count_ones();
                        self.set32(d, n);
                        self.scc = n != 0;
                    }
                    0x14 => {
                        // s_ff1_i32_b64: -1 when no bit is set
                        let x = self.sread64(a);
                        self.set32(d, if x == 0 { u32::MAX } else { x.trailing_zeros() });
                    }
                    0x24 | 0x25 => {
                        // s_and_saveexec_b64, s_or_saveexec_b64
                        assert_eq!(self.lanes, 64, "a 64-bit saveexec in wave32");
                        let (old, x) = (self.exec(), self.sread64(a));
                        let new = if op == 0x24 { x & old } else { x | old };
                        self.set64(d, old);
                        self.set64(EXEC, new);
                        self.scc = new != 0;
                    }
                    0x3C | 0x3D => {
                        // s_and_saveexec_b32, s_or_saveexec_b32 (wave32)
                        assert_eq!(self.lanes, 32, "a 32-bit saveexec in wave64");
                        let (old, x) = (self.exec(), u64::from(self.sread(a)));
                        let new = if op == 0x3C { x & old } else { x | old };
                        self.set32(d, old as u32);
                        self.set32(EXEC, new as u32);
                        self.scc = new != 0;
                    }
                    _ => panic!("the model has no SOP1 opcode {op:#x}"),
                }
            } else if w >> 28 == 0xB {
                panic!("the model has no SOPK instruction ({w:#010x})");
            } else if w >> 30 == 0b10 {
                // SOP2
                let op = w >> 23 & 0x7F;
                let d = (w >> 16 & 0x7F) as usize;
                let (a, b) = (src(w & 0xFF, 1), src(w >> 8 & 0xFF, 1));
                len = 1 + usize::from(a.code == LITERAL || b.code == LITERAL);
                match op {
                    0 => {
                        let (r, carry) = self.sread(a).overflowing_add(self.sread(b));
                        self.set32(d, r);
                        self.scc = carry;
                    }
                    0xA => self.set32(d, if self.scc { self.sread(a) } else { self.sread(b) }),
                    0xB => self.set64(d, if self.scc { self.sread64(a) } else { self.sread64(b) }),
                    0xE | 0x10 => {
                        // s_and_b32, s_or_b32
                        let (x, y) = (self.sread(a), self.sread(b));
                        let r = if op == 0xE { x & y } else { x | y };
                        self.set32(d, r);
                        self.scc = r != 0;
                    }
                    0xF | 0x11 => {
                        let (x, y) = (self.sread64(a), self.sread64(b));
                        let r = if op == 0xF { x & y } else { x | y };
                        self.set64(d, r);
                        self.scc = r != 0;
                    }
                    0x1E => {
                        let r = self.sread(a) << (self.sread(b) & 31);
                        self.set32(d, r);
                        self.scc = r != 0;
                    }
                    _ => panic!("the model has no SOP2 opcode {op:#x}"),
                }
            } else if w >> 26 == 0x3D {
                // SMEM: the descriptor load; the model's UAV is `mem` whatever it reads
                len = 2;
            } else if w >> 25 == 0x3F || w >> 25 == 0x3E || w >> 31 == 0 {
                len = self.vop12c(w, word(1));
            } else if w >> 26 == 0x35 {
                self.vop3(w, word(1));
                len = 2;
            } else if w >> 26 == 0x38 {
                // MUBUF
                // buffer_store_dword, buffer_atomic_add (lanes in order; with glc the old value
                // comes back)
                let op = w >> 18 & 0x7F;
                assert!(op == 0x1C || op == 0x32, "the model has only buffer_store_dword and buffer_atomic_add of MUBUF");
                let glc = w >> 14 & 1 == 1;
                assert_eq!(w >> 13 & 1, 0, "the model has no idxen");
                let w1 = word(1);
                let (offset, offen) = (w & 0xFFF, w >> 12 & 1 == 1);
                let (vaddr, vdata) = ((w1 & 0xFF) as usize, (w1 >> 8 & 0xFF) as usize);
                let soffset = self.sread(Src { code: w1 >> 24, literal: 0 });
                let exec = self.exec();
                for l in (0..LANES).filter(|&l| exec >> l & 1 == 1) {
                    let at = if offen { self.v[vaddr][l] } else { 0 } + offset + soffset;
                    assert_eq!(at % 4, 0, "lane {l} stores to byte {at}");
                    let slot = self.mem.get_mut(at as usize / 4).unwrap_or_else(|| panic!("lane {l} stores past the UAV (byte {at})"));
                    if op == 0x1C {
                        *slot = self.v[vdata][l];
                    } else {
                        let old = *slot;
                        *slot = old.wrapping_add(self.v[vdata][l]);
                        if glc {
                            self.v[vdata][l] = old;
                        }
                    }
                }
                len = 2;
            } else {
                panic!("the model has no instruction {w:#010x}");
            }
            pc += len;
        }
        panic!("the program did not end in 100000 instructions");
    }

    /// VOP1, VOP2 and VOPC (with their DPP forms); the words it took.
    fn vop12c(&mut self, w: u32, w1: u32) -> usize {
        let code0 = w & 0x1FF;
        let (src0, len) = match code0 {
            DPP8 | DPP8_FI | DPP16 => (Src { code: 256 + (w1 & 0xFF), literal: 0 }, 2),
            LITERAL => (Src { code: LITERAL, literal: w1 }, 2),
            c => (Src { code: c, literal: 0 }, 1),
        };
        let exec = self.exec();
        // the lane src0 is read from, None for "write 0" (DPP16's bound_ctrl past the row)
        let source = |l: usize| -> Option<usize> {
            match code0 {
                DPP8 | DPP8_FI => Some(l & !7 | (w1 >> (8 + 3 * (l & 7)) & 7) as usize),
                DPP16 => {
                    let ctrl = w1 >> 8 & 0x1FF;
                    assert_eq!(w1 >> 24, 0xFF, "the model has DPP16 with every row and bank");
                    assert_eq!(w1 >> 20 & 0xF, 0, "the model has no DPP16 source modifiers");
                    match ctrl {
                        0x140 => Some(l & !15 | (15 - (l & 15))), // row_mirror
                        // row_shr:n, nothing past the row's first lane
                        0x111..=0x11F => (l & 15).checked_sub((ctrl - 0x110) as usize).map(|j| l & !15 | j),
                        _ => panic!("the model has no DPP16 control {ctrl:#x}"),
                    }
                }
                _ => Some(l),
            }
        };
        let dpp = matches!(code0, DPP8 | DPP8_FI | DPP16);
        // a DPP16 source past the row: 0 with bound_ctrl, else the lane is not written
        let bound = w1 >> 19 & 1 == 1;
        let keep: u64 = if code0 == DPP16 && !bound {
            (0..LANES).filter(|&l| source(l).is_none()).fold(0, |m, l| m | 1 << l)
        } else {
            0
        };
        let read0 = |me: &Self, l: usize| -> u32 {
            match source(l) {
                Some(from) if dpp => {
                    assert!(exec >> from & 1 == 1, "a DPP read of lane {from} outside EXEC (not modelled)");
                    me.v[src0.code as usize - 256][from]
                }
                Some(from) => me.vread(src0, from),
                None => 0,
            }
        };
        if w >> 25 == 0x3F {
            // VOP1
            let op = w >> 9 & 0xFF;
            let d = (w >> 17 & 0xFF) as usize;
            match op {
                1 => self.vwrite_but(d, read0, keep),
                2 => {
                    // v_readfirstlane_b32: the first lane of EXEC, lane 0 when EXEC is empty
                    assert!(!dpp);
                    let l = if exec == 0 { 0 } else { exec.trailing_zeros() as usize };
                    let x = self.vread(src0, l);
                    self.set32(d, x);
                }
                _ => panic!("the model has no VOP1 opcode {op:#x}"),
            }
        } else if w >> 25 == 0x3E {
            // VOPC: v_cmp to VCC, v_cmpx to EXEC (GFX10: EXEC only); lanes outside EXEC give 0
            let op = w >> 17 & 0xFF;
            assert!(!dpp);
            let vsrc1 = (w >> 9 & 0xFF) as usize;
            let bits = self.compare(op & 0xEF, |me, l| (me.vread(src0, l), me.v[vsrc1][l]));
            self.set_mask(if op & 0xF0 == 0xD0 { EXEC } else { VCC }, bits);
        } else {
            // VOP2
            let op = w >> 25 & 0x3F;
            let d = (w >> 17 & 0xFF) as usize;
            let vsrc1 = (w >> 9 & 0xFF) as usize;
            let vcc = self.mask(VCC);
            let f = |a: u32, b: u32, l: usize| -> u32 {
                match op {
                    1 => if vcc >> l & 1 == 1 { b } else { a }, // v_cndmask_b32
                    0x16 => b >> (a & 31),                     // v_lshrrev_b32
                    0x1A => b << (a & 31),                     // v_lshlrev_b32
                    0x1C => a | b,                             // v_or_b32
                    0x25 => a.wrapping_add(b),                 // v_add_nc_u32
                    0x26 => a.wrapping_sub(b),                 // v_sub_nc_u32
                    0x27 => b.wrapping_sub(a),                 // v_subrev_nc_u32
                    _ => panic!("the model has no VOP2 opcode {op:#x}"),
                }
            };
            self.vwrite_but(d, |me, l| f(read0(me, l), me.v[vsrc1][l], l), keep);
        }
        len
    }

    /// The lane mask of an unsigned compare (`op` with the `x` bit cleared: 0xC1 up), the
    /// lanes outside EXEC clear.
    fn compare(&self, op: u32, ab: impl Fn(&Self, usize) -> (u32, u32)) -> u64 {
        let exec = self.exec();
        let mut bits = 0u64;
        for l in (0..self.lanes).filter(|&l| exec >> l & 1 == 1) {
            let (a, b) = ab(self, l);
            let t = match op {
                0xC1 => a < b,
                0xC2 => a == b,
                0xC3 => a <= b,
                0xC4 => a > b,
                0xC5 => a != b,
                0xC6 => a >= b,
                _ => panic!("the model has no compare opcode {op:#x}"),
            };
            bits |= u64::from(t) << l;
        }
        bits
    }

    /// VOP3 (two words; no literal, modifiers or op_sel).
    fn vop3(&mut self, w: u32, w1: u32) {
        let op = w >> 16 & 0x3FF;
        let d = (w & 0xFF) as usize;
        // (v_permlanex16_b32's op_sel[0] is `fi`: a source lane outside EXEC is read too)
        let fi = op == 0x378 && w >> 11 & 1 == 1;
        assert_eq!(w >> 8 & 0xFF & !(u32::from(fi) << 3), 0, "the model has no VOP3 abs, op_sel or clamp");
        assert_eq!(w1 >> 29, 0, "the model has no VOP3 neg");
        let s = |i: u32| Src { code: w1 >> (9 * i) & 0x1FF, literal: 0 };
        let (a, b, c) = (s(0), s(1), s(2));
        let exec = self.exec();
        match op {
            0xC1..=0xC6 => {
                // v_cmp_*_u32 to an SGPR pair
                let bits = self.compare(op, |me, l| (me.vread(a, l), me.vread(b, l)));
                self.set_mask(d, bits);
            }
            0x101 => {
                // v_cndmask_b32 with a mask from SGPRs
                let m = if self.lanes == 64 { self.sread64(c) } else { u64::from(self.sread(c)) };
                self.vwrite(d, |me, l| if m >> l & 1 == 1 { me.vread(b, l) } else { me.vread(a, l) });
            }
            0x360 => {
                // v_readlane_b32: whatever EXEC is
                let l = self.sread(b) as usize & (self.lanes - 1);
                let x = self.vread(a, l);
                self.set32(d, x);
            }
            0x365 | 0x366 => {
                // v_mbcnt_lo/hi_u32_b32: the bits of the mask's half below this lane
                let mask = self.sread(a);
                self.vwrite(d, |me, l| {
                    let below = if op == 0x365 { l.min(32) } else { l.saturating_sub(32) };
                    let low = if below >= 32 { mask } else { mask & ((1u32 << below) - 1) };
                    low.count_ones().wrapping_add(me.vread(b, l))
                });
            }
            0x378 => {
                // v_permlanex16_b32: each lane reads the other row of its 32 by the selects
                let (lo, hi) = (self.sread(b), self.sread(c));
                self.vwrite(d, |me, l| {
                    let j = l & 15;
                    let sel = (if j < 8 { lo >> (4 * j) } else { hi >> (4 * (j - 8)) } & 15) as usize;
                    let from = l & !31 | (l & 16 ^ 16) | sel;
                    assert!(fi || exec >> from & 1 == 1, "a permlane read of lane {from} outside EXEC without fi (not modelled)");
                    me.vread(a, from)
                });
            }
            _ => panic!("the model has no VOP3 opcode {op:#x}"),
        }
    }
}

/// The UAV after one console wave ran `words` with root constants `constants`.
pub(super) fn console(words: &[u32], constants: &[u32], mem_dwords: usize) -> Vec<u32> {
    console_of(words, constants, mem_dwords, LANES)
}

/// [`console`] for a console wave of `lanes` lanes (32: a wave32 program).
pub(super) fn console_of(words: &[u32], constants: &[u32], mem_dwords: usize, lanes: usize) -> Vec<u32> {
    let mut wave = Wave::with_lanes(constants, mem_dwords, lanes);
    wave.run(words);
    wave.mem
}

/// Operands for the assembler.
pub(super) mod asm {
    /// A source: an SGPR, a VGPR, an inline integer (-16..=64) or a literal.
    #[derive(Clone, Copy)]
    pub enum Op {
        S(u8),
        V(u8),
        I(i32),
        Lit(u32),
    }
    pub use Op::*;

    pub const VCC: u8 = 106;
    pub const EXEC: u8 = 126;

    impl Op {
        /// (9-bit code, literal word)
        fn code(self) -> (u32, Option<u32>) {
            match self {
                S(n) => (u32::from(n), None),
                V(n) => (256 + u32::from(n), None),
                I(i @ 0..=64) => (128 + i as u32, None),
                I(i @ -16..=-1) => (192 + (-i) as u32, None),
                I(i) => (255, Some(i as u32)),
                Lit(x) => (255, Some(x)),
            }
        }
    }

    /// A program being written, with labels for its branches.
    #[derive(Default)]
    pub struct Asm {
        pub words: Vec<u32>,
        labels: Vec<(&'static str, usize)>,
        fixups: Vec<(usize, &'static str)>,
    }

    impl Asm {
        fn push(&mut self, w: u32, lit: Option<u32>) -> &mut Self {
            self.words.push(w);
            self.words.extend(lit);
            self
        }

        /// Raw words (a sequence written out elsewhere).
        pub fn raw(&mut self, words: &[u32]) -> &mut Self {
            self.words.extend_from_slice(words);
            self
        }

        pub fn label(&mut self, name: &'static str) -> &mut Self {
            self.labels.push((name, self.words.len()));
            self
        }

        /// SOPP branch `op` (2 s_branch, 4 scc0, 5 scc1, 6 vccz, 7 vccnz, 8 execz, 9 execnz).
        pub fn branch(&mut self, op: u32, to: &'static str) -> &mut Self {
            self.fixups.push((self.words.len(), to));
            self.push(0xBF80_0000 | op << 16, None)
        }

        pub fn endpgm(&mut self) -> &mut Self {
            self.push(0xBF81_0000, None)
        }

        pub fn finish(&mut self) -> Vec<u32> {
            for &(at, to) in &self.fixups {
                let target = self.labels.iter().find(|l| l.0 == to).unwrap_or_else(|| panic!("no label {to}")).1;
                let simm = target as isize - at as isize - 1;
                self.words[at] |= u32::from(simm as i16 as u16);
            }
            std::mem::take(&mut self.words)
        }

        /// SOP1 `op` (3 mov_b32, 4 mov_b64, 0x10 bcnt1_i32_b64, 0x14 ff1_i32_b64,
        /// 0x24 and_saveexec_b64, 0x25 or_saveexec_b64, 0x3C and_saveexec_b32,
        /// 0x3D or_saveexec_b32).
        pub fn sop1(&mut self, op: u32, d: u8, a: Op) -> &mut Self {
            let (c, lit) = a.code();
            assert!(c < 256, "a scalar source");
            self.push(0xBE80_0000 | u32::from(d) << 16 | op << 8 | c, lit)
        }

        /// SOP2 `op` (0 add_u32, 0xA cselect_b32, 0xE and_b32, 0xF and_b64, 0x10 or_b32,
        /// 0x11 or_b64, 0x1E lshl_b32).
        pub fn sop2(&mut self, op: u32, d: u8, a: Op, b: Op) -> &mut Self {
            let ((ca, la), (cb, lb)) = (a.code(), b.code());
            assert!(ca < 256 && cb < 256 && (la.is_none() || lb.is_none()));
            self.push(0x8000_0000 | op << 23 | u32::from(d) << 16 | cb << 8 | ca, la.or(lb))
        }

        /// SOPC `op` (6 eq_u32, 7 lg_u32, 9 ge_u32, 10 lt_u32).
        pub fn sopc(&mut self, op: u32, a: Op, b: Op) -> &mut Self {
            let ((ca, la), (cb, lb)) = (a.code(), b.code());
            assert!(ca < 256 && cb < 256 && (la.is_none() || lb.is_none()));
            self.push(0xBF00_0000 | op << 16 | cb << 8 | ca, la.or(lb))
        }

        /// VOP1 `op` (1 mov_b32, 2 readfirstlane_b32).
        pub fn vop1(&mut self, op: u32, d: u8, a: Op) -> &mut Self {
            let (c, lit) = a.code();
            self.push(0x7E00_0000 | u32::from(d) << 17 | op << 9 | c, lit)
        }

        /// VOP2 `op` (0x16 lshrrev, 0x1A lshlrev, 0x1C or, 0x25 add_nc_u32), `b` a VGPR.
        pub fn vop2(&mut self, op: u32, d: u8, a: Op, b: u8) -> &mut Self {
            let (c, lit) = a.code();
            self.push(op << 25 | u32::from(d) << 17 | u32::from(b) << 9 | c, lit)
        }

        /// VOP2 `op` with a DPP16 `row_shr:n` source `a` (every row and bank), `bound_ctrl`
        /// giving 0 past the row's first lane instead of leaving the lane unwritten.
        pub fn vop2_row_shr(&mut self, op: u32, d: u8, a: u8, b: u8, n: u32, bound_ctrl: bool) -> &mut Self {
            assert!((1..16).contains(&n));
            self.push(op << 25 | u32::from(d) << 17 | u32::from(b) << 9 | 250, None);
            self.push(0xFF << 24 | u32::from(bound_ctrl) << 19 | (0x110 + n) << 8 | u32::from(a), None)
        }

        /// `v_permlanex16_b32 v<d>, v<a>, sel_lo, sel_hi` with `fi` (fetch inactive) set.
        pub fn permlanex16_fi(&mut self, d: u8, a: u8, lo: Op, hi: Op) -> &mut Self {
            let ((cl, ll), (ch, lh)) = (lo.code(), hi.code());
            assert!(ll.is_none() && lh.is_none());
            self.push(0xD400_0000 | 0x378 << 16 | 1 << 11 | u32::from(d), None);
            self.push(ch << 18 | cl << 9 | (256 + u32::from(a)), None)
        }

        /// VOPC `op` (0xC1 lt .. 0xC6 ge of u32 to VCC; 0xD1 .. 0xD6 the v_cmpx forms).
        pub fn vopc(&mut self, op: u32, a: Op, b: u8) -> &mut Self {
            let (c, lit) = a.code();
            self.push(0x7C00_0000 | op << 17 | u32::from(b) << 9 | c, lit)
        }

        /// VOP3 `op` (0xC1-0xC6 compares to an SGPR pair, 0x360 readlane, 0x101 cndmask).
        pub fn vop3(&mut self, op: u32, d: u8, a: Op, b: Op, c: Op) -> &mut Self {
            let (ca, la) = a.code();
            let (cb, lb) = b.code();
            let (cc, lc) = c.code();
            assert!(la.is_none() && lb.is_none() && lc.is_none(), "no literal in VOP3 here");
            self.push(0xD400_0000 | op << 16 | u32::from(d), None);
            self.push(cc << 18 | cb << 9 | ca, None)
        }

        /// `buffer_atomic_add v<data>, v<addr>, s[8:11], 0 offen glc`: the old value back.
        pub fn atomic_add(&mut self, data: u8, addr: u8) -> &mut Self {
            self.push(0xE000_0000 | 0x32 << 18 | 1 << 14 | 1 << 12, None);
            self.push(0x80 << 24 | 2 << 16 | u32::from(data) << 8 | u32::from(addr), None)
        }

        /// `buffer_store_dword v<data>, v<addr>, s[8:11], 0 offen offset:<offset>`.
        pub fn store(&mut self, data: u8, addr: u8, offset: u32) -> &mut Self {
            assert!(offset < 4096);
            self.push(0xE070_1000 | offset, None);
            self.push(0x80 << 24 | 2 << 16 | u32::from(data) << 8 | u32::from(addr), None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::asm::{Asm, EXEC, I, Lit, S, V};
    use super::console;

    /// The model's own reading of the console's rules, on words whose effect the ISA
    /// reference states.
    #[test]
    fn the_model_keeps_masked_lanes_and_reads_lane_0_with_no_lane_left() {
        let mut a = Asm::default();
        a.vop2(0x1A, 3, I(2), 0) // v3 = 4 t
            .vop1(1, 1, I(7)) // v1 = 7
            .sop1(4, 12, S(EXEC)) // s[12:13] = exec
            .vopc(0xD3, S(0), 0) // exec = s0 <= t
            .vop1(1, 1, I(11)) // v1 = 11 in those lanes
            .vop1(2, 5, V(1)) // s5 = first lane of v1
            .vop3(0x360, 6, V(1), I(31), I(0)) // s6 = lane 31 of v1
            .sop1(4, EXEC, S(12))
            .vop1(1, 2, S(5))
            .store(2, 3, 0)
            .vop1(1, 2, S(6))
            .store(2, 3, 256)
            .endpgm();
        let words = a.finish();
        for from in [0u32, 31, 32, 63, 64] {
            let mem = console(&words, &[from], 128);
            let lane = |l: u32| if l >= from { 11 } else { 7 };
            assert!(mem[..64].iter().all(|&x| x == if from < 64 { 11 } else { lane(0) }), "from {from}");
            assert!(mem[64..].iter().all(|&x| x == lane(31)), "from {from}");
        }
    }

    #[test]
    fn lanes_that_left_store_nothing_and_a_full_saveexec_runs_them_again() {
        let mut a = Asm::default();
        a.vop2(0x1A, 3, I(2), 0)
            .vop2(0x25, 7, Lit(0xBEEF_0000), 0) // v7 = 0xbeef0000 + t in every lane
            .vopc(0xD4, S(0), 0) // exec = t < s0
            .branch(8, "end")
            .vop2(0x25, 7, I(1), 0) // v7 = t + 1 in the lanes left
            .sop1(0x25, 4, I(-1)) // all lanes on again
            .vop1(1, 2, V(7))
            .store(2, 3, 0)
            .sop1(4, EXEC, S(4))
            .label("end")
            .endpgm();
        let words = a.finish();
        // the lanes that came back store what they held before they left: their v7 was
        // written before the masked write and kept through it
        let mem = console(&words, &[40], 64);
        for (l, &x) in mem.iter().enumerate() {
            assert_eq!(x, if l < 40 { l as u32 + 1 } else { 0xBEEF_0000 + l as u32 }, "lane {l}");
        }
        assert!(console(&words, &[0], 64).iter().all(|&x| x == 0));
    }
}
