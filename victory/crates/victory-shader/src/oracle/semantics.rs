//! What each GFX10 instruction does, for [`super`]'s interpreter: written from AMD's published
//! "RDNA 2" Instruction Set Architecture reference guide (2020), its section and table numbers
//! cited per group, and from nothing of the translator (`emit/`), so that the two can disagree.
//!
//! One function per instruction group. Each takes plain values (a lane's sources, the MODE
//! register) and gives plain values; which lanes run, which registers are read and written and
//! which lane a cross-lane operation reads are the machine's ([`super::machine`]), from the
//! lane-selection functions here. Anything the reference does not define, or defines in two ways
//! that disagree, is refused with the reason (`Err`): the oracle never guesses.
//!
//! Instructions the decoder names but the reference does not define are refused by name. The
//! reference's own list of removals (its "Instruction Changes", printed page 3) names
//! `V_MAC_F32`, `V_MADMK_F32`, `V_MADAK_F32`, `V_MAC_LEGACY_F32` and `V_MAD_LEGACY_F32`, and it
//! gives no entry for `V_MAD_F32`. 6.3's Table 22 still lists `V_MAC_{F16,F32}`, a stale entry:
//! 12.7 defines only the fused replacement, and AMD's machine-readable RDNA 2 ISA has no F32
//! form. The console's programs use these, so the console's ISA is not exactly the
//! reference's there, and their semantics wait for a reference that defines them.

/// The MODE register (3.5, Table 4; 6.4, Table 24).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mode(pub u32);

impl Mode {
    /// Round mode of single precision ([1:0]) and of double and half ([3:2]); 0 is nearest even.
    fn round(self, wide: bool) -> u32 {
        let shift = if wide { 2 } else { 0 };
        (self.0 >> shift) & 3
    }
    /// Denormal mode ([5:4] single, [7:6] double and half): bit 0 allows input denormals, bit 1
    /// output denormals (0 flush both, 1 allow input, 2 allow output, 3 allow both).
    fn denorm(self, wide: bool) -> u32 {
        let shift = if wide { 6 } else { 4 };
        (self.0 >> shift) & 3
    }
    /// IEEE (bit 9): signaling NaNs are quieted and propagated by min and max.
    pub fn ieee(self) -> bool {
        self.0 >> 9 & 1 == 1
    }
    /// DX10_CLAMP (bit 8): the clamp bit turns NaN into 0.
    fn dx10_clamp(self) -> bool {
        self.0 >> 8 & 1 == 1
    }
    /// FP16_OVFL (bit 23).
    fn fp16_ovfl(self) -> bool {
        self.0 >> 23 & 1 == 1
    }
    /// Only round-to-nearest-even is modelled: the reference names the other modes (6.4) and
    /// every float result below would need them.
    fn nearest(self, wide: bool) -> Result<(), String> {
        if self.round(wide) == 0 {
            Ok(())
        } else {
            Err(format!("round mode {} (only nearest even is modelled)", self.round(wide)))
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Floating point helpers (6.4)
// ---------------------------------------------------------------------------------------------

fn denormal32(b: u32) -> bool {
    b & 0x7F80_0000 == 0 && b & 0x007F_FFFF != 0
}

fn nan32(b: u32) -> bool {
    b & 0x7F80_0000 == 0x7F80_0000 && b & 0x007F_FFFF != 0
}

fn snan32(b: u32) -> bool {
    nan32(b) && b & 0x0040_0000 == 0
}

/// `Quiet(x)` of the pseudocode: the NaN with its quiet bit set.
fn quiet32(b: u32) -> u32 {
    b | 0x0040_0000
}

fn denormal64(b: u64) -> bool {
    b & 0x7FF0_0000_0000_0000 == 0 && b & 0x000F_FFFF_FFFF_FFFF != 0
}

fn nan64(b: u64) -> bool {
    b & 0x7FF0_0000_0000_0000 == 0x7FF0_0000_0000_0000 && b & 0x000F_FFFF_FFFF_FFFF != 0
}

/// An input under the denormal mode: flushed to a zero of its sign where input denormals are
/// not allowed (6.4).
fn in32(mode: Mode, b: u32) -> f32 {
    if denormal32(b) && mode.denorm(false) & 1 == 0 {
        f32::from_bits(b & 0x8000_0000)
    } else {
        f32::from_bits(b)
    }
}

/// A result under the denormal mode: flushed to a zero of its sign where output denormals are
/// not allowed (6.4).
fn out32(mode: Mode, x: f32) -> u32 {
    let b = x.to_bits();
    if denormal32(b) && mode.denorm(false) & 2 == 0 { b & 0x8000_0000 } else { b }
}

fn in64(mode: Mode, b: u64) -> f64 {
    if denormal64(b) && mode.denorm(true) & 1 == 0 {
        f64::from_bits(b & 0x8000_0000_0000_0000)
    } else {
        f64::from_bits(b)
    }
}

fn out64(mode: Mode, x: f64) -> u64 {
    let b = x.to_bits();
    if denormal64(b) && mode.denorm(true) & 2 == 0 { b & 0x8000_0000_0000_0000 } else { b }
}

/// Always flushed, whatever the mode: the entries of `V_RCP_F32`, `V_RSQ_F32`, `V_SQRT_F32`,
/// `V_EXP_F32` and `V_LOG_F32` say "denormals are flushed" (12.8).
fn flush32(b: u32) -> u32 {
    if denormal32(b) { b & 0x8000_0000 } else { b }
}

/// IEEE single to half, round to nearest even, with half denormals where the half denormal
/// mode allows them (`V_CVT_F16_F32`, 12.8: "0.5ULP accuracy ... creates FP16 denormals when
/// appropriate"). With FP16_OVFL (3.5) an overflow gives the largest half, not infinity.
pub fn f32_to_f16(mode: Mode, b: u32) -> u16 {
    let sign = (b >> 16 & 0x8000) as u16;
    let exp = (b >> 23 & 0xFF) as i32;
    let man = b & 0x007F_FFFF;
    if exp == 0xFF {
        return if man != 0 { sign | 0x7E00 | (man >> 13) as u16 } else { sign | 0x7C00 };
    }
    if exp == 0 {
        return sign; // single denormals are far below the half range
    }
    let e = exp - 127;
    let overflow = if mode.fp16_ovfl() { sign | 0x7BFF } else { sign | 0x7C00 };
    if e > 15 {
        return overflow;
    }
    let sig = man | 0x0080_0000;
    let (q, rem, half) = if e >= -14 {
        // normal: drop 13 bits of the significand
        ((((e + 15) as u32) << 10) | (man >> 13), man & 0x1FFF, 0x1000)
    } else {
        // half denormal: units of 2^-24
        let shift = (-(e + 1)) as u32;
        if shift >= 32 {
            (0, 1, u32::MAX)
        } else {
            (sig >> shift, sig & ((1 << shift) - 1), 1 << (shift - 1))
        }
    };
    let up = rem > half || (rem == half && q & 1 == 1);
    let h = q + u32::from(up);
    if h >= 0x7C00 {
        return overflow;
    }
    let h = h as u16;
    if h & 0x7C00 == 0 && h != 0 && mode.denorm(true) & 2 == 0 {
        return sign; // output half denormals flushed
    }
    sign | h
}

/// Half to IEEE single, exact (`V_CVT_F32_F16`: "0ULP accuracy, FP16 denormal inputs are
/// accepted").
pub fn f16_to_f32(h: u16) -> u32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from(h >> 10 & 0x1F);
    let man = u32::from(h & 0x3FF);
    match (exp, man) {
        (0, 0) => sign,
        (0, _) => {
            let lead = 31 - man.leading_zeros(); // the top set bit, 0..=9
            sign | ((lead + 103) << 23) | ((man << (23 - lead)) & 0x007F_FFFF)
        }
        (0x1F, 0) => sign | 0x7F80_0000,
        (0x1F, _) => sign | 0x7FC0_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    }
}

// ---------------------------------------------------------------------------------------------
// Scalar ALU (5.4-5.8; 12.1 SOP2, 12.2 SOPK, 12.3 SOP1, 12.4 SOPC)
// ---------------------------------------------------------------------------------------------

/// What a scalar ALU instruction leaves: its destination (none for compares), SCC where it
/// sets it, and EXEC where it writes it (the `SAVEEXEC` and `WREXEC` forms).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Salu {
    pub d: Option<u64>,
    pub scc: Option<bool>,
    pub exec: Option<u64>,
}

/// Inputs of [`salu`]: the sources (zero-extended to 64 bits; 64-bit forms whole), SCC, EXEC,
/// SOPK's 16-bit immediate and the destination's value before (`S_CMOV`, `S_ADDK`, ...).
#[derive(Debug, Clone, Copy, Default)]
pub struct SaluIn {
    pub s0: u64,
    pub s1: u64,
    pub scc: bool,
    pub exec: u64,
    pub imm: u16,
    pub d: u64,
}

/// The scalar ALU. Pseudocode per opcode as 12.1-12.4 give it; the SAVEEXEC and WREXEC forms
/// from 5.7, Table 13 ("D = EXEC; EXEC = S0 <op> EXEC; SCC = (exec != 0)").
pub fn salu(name: &str, i: SaluIn) -> Result<Salu, String> {
    let (s0, s1) = (i.s0, i.s1);
    let (a, b) = (s0 as u32, s1 as u32);
    let d32 = |v: u32| Salu { d: Some(u64::from(v)), scc: Some(v != 0), exec: None };
    let d64 = |v: u64| Salu { d: Some(v), scc: Some(v != 0), exec: None };
    let mov = |v: u64| Salu { d: Some(v), scc: None, exec: None };
    let set = |c: bool| Salu { d: None, scc: Some(c), exec: None };
    let both = |v: u32, c: bool| Salu { d: Some(u64::from(v)), scc: Some(c), exec: None };
    let simm = i32::from(i.imm as i16);
    if let Some(op) = name.strip_suffix("_saveexec_b64").or(name.strip_suffix("_saveexec_b32")) {
        let op = op.trim_start_matches("s_");
        let wide = name.ends_with("b64");
        let e = if wide { i.exec } else { i.exec & 0xFFFF_FFFF };
        let v = bitwise(op, s0, e).ok_or_else(|| format!("{name}: no such operation"))?;
        let v = if wide { v } else { v & 0xFFFF_FFFF };
        return Ok(Salu { d: Some(e), scc: Some(v != 0), exec: Some(v) });
    }
    if let Some(op) = name.strip_suffix("_wrexec_b64").or(name.strip_suffix("_wrexec_b32")) {
        // 5.7: "N1: EXEC, D = ~S0 & EXEC; N2: EXEC, D = S0 & ~EXEC; SCC = (result != 0)".
        let e = if name.ends_with("b64") { i.exec } else { i.exec & 0xFFFF_FFFF };
        let v = match op {
            "s_andn1" => !s0 & e,
            "s_andn2" => s0 & !e,
            _ => return Err(format!("{name}: no such operation")),
        };
        let v = if name.ends_with("b64") { v } else { v & 0xFFFF_FFFF };
        return Ok(Salu { d: Some(v), scc: Some(v != 0), exec: Some(v) });
    }
    if let Some(rest) = name.strip_prefix("s_cmp_").or(name.strip_prefix("s_cmpk_")) {
        let k = name.starts_with("s_cmpk_");
        let (op, ty) = rest.rsplit_once('_').ok_or_else(|| format!("{name}: no type"))?;
        let (x, y): (i64, i64) = match (ty, k) {
            ("i32", false) => (i64::from(a as i32), i64::from(b as i32)),
            ("u32", false) => (i64::from(a), i64::from(b)),
            ("i32", true) => (i64::from(a as i32), i64::from(simm)),
            ("u32", true) => (i64::from(a), i64::from(i.imm)),
            ("u64", false) => (s0 as i64, s1 as i64), // only eq and lg: equality alone
            _ => return Err(format!("{name}: no such compare")),
        };
        return Ok(set(match op {
            "eq" => x == y,
            "lg" => x != y,
            "gt" => x > y,
            "ge" => x >= y,
            "lt" => x < y,
            "le" => x <= y,
            _ => return Err(format!("{name}: no such compare")),
        }));
    }
    Ok(match name {
        "s_mov_b32" | "s_mov_b64" => mov(s0),
        "s_movk_i32" => mov(u64::from(simm as u32)),
        "s_cmov_b32" | "s_cmov_b64" => mov(if i.scc { s0 } else { i.d }),
        "s_cmovk_i32" => mov(if i.scc { u64::from(simm as u32) } else { i.d }),
        "s_not_b32" => d32(!a),
        "s_not_b64" => d64(!s0),
        "s_brev_b32" => mov(u64::from(a.reverse_bits())),
        "s_brev_b64" => mov(s0.reverse_bits()),
        "s_bcnt0_i32_b32" => d32(32 - a.count_ones()),
        "s_bcnt0_i32_b64" => d32(64 - s0.count_ones()),
        "s_bcnt1_i32_b32" => d32(a.count_ones()),
        "s_bcnt1_i32_b64" => d32(s0.count_ones()),
        "s_ff1_i32_b32" => mov(u64::from(if a == 0 { u32::MAX } else { a.trailing_zeros() })),
        "s_ff1_i32_b64" => mov(u64::from(if s0 == 0 { u32::MAX } else { s0.trailing_zeros() })),
        "s_ff0_i32_b32" => mov(u64::from(if !a == 0 { u32::MAX } else { (!a).trailing_zeros() })),
        "s_ff0_i32_b64" => mov(u64::from(if !s0 == 0 { u32::MAX } else { (!s0).trailing_zeros() })),
        "s_flbit_i32_b32" => mov(u64::from(if a == 0 { u32::MAX } else { a.leading_zeros() })),
        "s_flbit_i32_b64" => mov(u64::from(if s0 == 0 { u32::MAX } else { s0.leading_zeros() })),
        "s_sext_i32_i8" => mov(u64::from(i32::from(a as i8) as u32)),
        "s_sext_i32_i16" => mov(u64::from(i32::from(a as i16) as u32)),
        "s_bitset0_b32" => mov(i.d & 0xFFFF_FFFF & !(1 << (a & 31))),
        "s_bitset1_b32" => mov((i.d & 0xFFFF_FFFF) | 1 << (a & 31)),
        "s_bitset0_b64" => mov(i.d & !(1 << (a & 63))),
        "s_bitset1_b64" => mov(i.d | 1 << (a & 63)),
        "s_abs_i32" => d32((a as i32).wrapping_abs() as u32),
        "s_add_u32" => {
            let r = u64::from(a) + u64::from(b);
            both(r as u32, r >> 32 != 0)
        }
        "s_addc_u32" => {
            let r = u64::from(a) + u64::from(b) + u64::from(i.scc);
            both(r as u32, r >> 32 != 0)
        }
        "s_sub_u32" => both(a.wrapping_sub(b), b > a),
        "s_subb_u32" => {
            let r = a.wrapping_sub(b).wrapping_sub(u32::from(i.scc));
            both(r, u64::from(b) + u64::from(i.scc) > u64::from(a))
        }
        "s_add_i32" => {
            let r = a.wrapping_add(b);
            both(r, a >> 31 == b >> 31 && a >> 31 != r >> 31)
        }
        "s_sub_i32" => {
            let r = a.wrapping_sub(b);
            both(r, a >> 31 != b >> 31 && a >> 31 != r >> 31)
        }
        "s_addk_i32" => {
            let t = i.d as u32;
            let r = t.wrapping_add(simm as u32);
            both(r, t >> 31 == u32::from(i.imm >> 15) && t >> 31 != r >> 31)
        }
        "s_mulk_i32" => mov(u64::from((i.d as u32 as i32).wrapping_mul(simm) as u32)),
        "s_min_i32" => both(if (a as i32) < (b as i32) { a } else { b }, (a as i32) < (b as i32)),
        "s_min_u32" => both(if a < b { a } else { b }, a < b),
        "s_max_i32" => both(if (a as i32) > (b as i32) { a } else { b }, (a as i32) > (b as i32)),
        "s_max_u32" => both(if a > b { a } else { b }, a > b),
        "s_cselect_b32" | "s_cselect_b64" => mov(if i.scc { s0 } else { s1 }),
        "s_lshl_b32" => d32(a << (b & 31)),
        "s_lshl_b64" => d64(s0 << (b & 63)),
        "s_lshr_b32" => d32(a >> (b & 31)),
        "s_lshr_b64" => d64(s0 >> (b & 63)),
        "s_ashr_i32" => d32(((a as i32) >> (b & 31)) as u32),
        "s_ashr_i64" => d64(((s0 as i64) >> (b & 63)) as u64),
        "s_bfm_b32" => mov(u64::from((((1u64 << (a & 31)) - 1) << (b & 31)) as u32)),
        "s_bfm_b64" => mov(((1u64 << (a & 63)) - 1) << (b & 63)),
        "s_mul_i32" => mov(u64::from((a as i32).wrapping_mul(b as i32) as u32)),
        "s_mul_hi_u32" => mov((u64::from(a) * u64::from(b)) >> 32),
        "s_mul_hi_i32" => mov(u64::from(((i64::from(a as i32) * i64::from(b as i32)) >> 32) as u32)),
        "s_bfe_u32" => {
            // D.u = (S0.u >> S1.u[4:0]) & ((1 << S1.u[22:16]) - 1)
            let w = (b >> 16) & 0x7F;
            let mask = if w >= 32 { u32::MAX } else { (1 << w) - 1 };
            d32((a >> (b & 31)) & mask)
        }
        "s_bfe_i32" => {
            // D.i = (S0.i >> S1.u[4:0]) & ((1 << S1.u[22:16]) - 1), sign-extended: the shift is
            // of the signed source, so a field past bit 31 takes its sign
            let w = (b >> 16) & 0x7F;
            let shifted = ((a as i32) >> (b & 31)) as u32;
            let field = if w >= 32 { shifted } else { shifted & ((1 << w) - 1) };
            let v = if w == 0 || w >= 32 { field } else { ((field << (32 - w)) as i32 >> (32 - w)) as u32 };
            d32(v)
        }
        "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
            let n = u32::from(name.as_bytes()[6] - b'0');
            let r = (u64::from(a) << n) + u64::from(b);
            both(r as u32, r >> 32 != 0)
        }
        "s_bitcmp0_b32" => set(a >> (b & 31) & 1 == 0),
        "s_bitcmp1_b32" => set(a >> (b & 31) & 1 == 1),
        "s_bitcmp0_b64" => set(s0 >> (b & 63) & 1 == 0),
        "s_bitcmp1_b64" => set(s0 >> (b & 63) & 1 == 1),
        _ => {
            let (op, wide) = match (name.strip_suffix("_b32"), name.strip_suffix("_b64")) {
                (Some(op), _) => (op, false),
                (_, Some(op)) => (op, true),
                _ => return Err(format!("{name}: not in the oracle's scalar ALU")),
            };
            let v = bitwise(op.trim_start_matches("s_"), s0, s1)
                .ok_or_else(|| format!("{name}: not in the oracle's scalar ALU"))?;
            if wide { d64(v) } else { d32(v as u32) }
        }
    })
}

/// The two-input bitwise operations of 12.1 and 5.7 (`and`, `or`, `xor`, `andn1`, `andn2`,
/// `orn1`, `orn2`, `nand`, `nor`, `xnor`; "N1" negates S0, "N2" S1).
fn bitwise(op: &str, x: u64, y: u64) -> Option<u64> {
    Some(match op {
        "and" => x & y,
        "or" => x | y,
        "xor" => x ^ y,
        "andn1" => !x & y,
        "andn2" => x & !y,
        "orn1" => !x | y,
        "orn2" => x | !y,
        "nand" => !(x & y),
        "nor" => !(x | y),
        "xnor" => !(x ^ y),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------------------------
// Vector ALU, one lane (6.2-6.5; 12.7 VOP2, 12.8 VOP1, 12.9 VOPC, 12.10 VOP3P, 12.12 VOP3)
// ---------------------------------------------------------------------------------------------

/// What kind of value a VALU source is: it decides the inline constants (6.2, Table 20), the
/// expansion of a literal to 64 bits (6.2.1) and whether `abs`/`neg` apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    B32,
    F32,
    F16,
    U64,
    I64,
    F64,
}

impl Kind {
    pub fn float(self) -> bool {
        matches!(self, Kind::F32 | Kind::F16 | Kind::F64)
    }
    pub fn wide(self) -> bool {
        matches!(self, Kind::U64 | Kind::I64 | Kind::F64)
    }
}

fn kind_of(token: &str) -> Kind {
    match token {
        "f32" => Kind::F32,
        "f16" => Kind::F16,
        "f64" => Kind::F64,
        "u64" | "b64" => Kind::U64,
        "i64" => Kind::I64,
        _ => Kind::B32,
    }
}

/// The kind of source `i` of a VALU opcode, by its name: the last type in the name, the source
/// type of a conversion (`v_cvt_<dst>_<src>`), and the exceptions whose entries name other
/// sizes for some of their sources.
pub fn source_kind(name: &str, i: usize) -> Kind {
    let parts: Vec<&str> = name.split('_').collect();
    match (name, i) {
        ("v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64", 0) => Kind::B32,
        ("v_ldexp_f32" | "v_cmp_class_f32" | "v_cmpx_class_f32", 1) => Kind::B32,
        ("v_ldexp_f64" | "v_cmp_class_f64" | "v_cmpx_class_f64", 1) => Kind::B32,
        ("v_mad_u64_u32", 0 | 1) | ("v_mad_i64_i32", 0 | 1) => Kind::B32,
        ("v_mad_u64_u32", 2) => Kind::U64,
        ("v_mad_i64_i32", 2) => Kind::I64,
        ("v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16", _) => Kind::F32,
        ("v_cvt_pk_u8_f32", 1 | 2) => Kind::B32,
        ("v_readlane_b32" | "v_writelane_b32", _) => Kind::B32,
        _ if parts.len() == 4 && parts[1] == "cvt" => kind_of(parts[3]),
        _ if name.starts_with("v_cvt_f32_ubyte") => Kind::B32,
        _ => kind_of(parts.last().copied().unwrap_or("")),
    }
}

/// An inline constant (6.2, Table 20) as a source of `kind`: 128-192 0..64, 193-208 -1..-16,
/// 240-248 the floats 0.5, -0.5, 1, -1, 2, -2, 4, -4 and 1/(2 pi) in the precision of the
/// instruction ("Single, double, or half-precision inline floats"), with the exact values the
/// table gives for 1/(2 pi). Integers widen to 64 bits with their sign.
pub fn inline_constant(code: u8, kind: Kind) -> Option<u64> {
    let int: Option<i64> = match code {
        128..=192 => Some(i64::from(code) - 128),
        193..=208 => Some(192 - i64::from(code)),
        _ => None,
    };
    if let Some(n) = int {
        return Some(if kind.wide() { n as u64 } else { u64::from(n as u32) });
    }
    let index = code.checked_sub(240).filter(|&i| i <= 8)? as usize;
    const F32: [u32; 9] = [
        0x3F00_0000, 0xBF00_0000, 0x3F80_0000, 0xBF80_0000, 0x4000_0000, 0xC000_0000, 0x4080_0000,
        0xC080_0000, 0x3E22_F983,
    ];
    const F16: [u16; 9] = [0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118];
    const F64: [u64; 9] = [
        0x3FE0_0000_0000_0000,
        0xBFE0_0000_0000_0000,
        0x3FF0_0000_0000_0000,
        0xBFF0_0000_0000_0000,
        0x4000_0000_0000_0000,
        0xC000_0000_0000_0000,
        0x4010_0000_0000_0000,
        0xC010_0000_0000_0000,
        0x3FC4_5F30_6DC9_C882,
    ];
    Some(match kind {
        Kind::F16 => u64::from(F16[index]),
        Kind::F64 | Kind::U64 | Kind::I64 => F64[index],
        Kind::B32 | Kind::F32 => u64::from(F32[index]),
    })
}

/// A 32-bit literal as a source of `kind` (6.2.1, "Literal Expansion to 64 bits": a 64-bit float
/// takes it as its high half, a 64-bit unsigned integer zero-extends it, a signed one
/// sign-extends it).
pub fn literal(value: u32, kind: Kind) -> u64 {
    match kind {
        Kind::F64 => u64::from(value) << 32,
        Kind::I64 => i64::from(value as i32) as u64,
        _ => u64::from(value),
    }
}

/// `abs` and `neg` of a float source (6.2.1: VOP3 "applying absolute value (ABS field) or negate
/// (NEG field) to any of the input operands"); abs first.
pub fn modify(value: u64, kind: Kind, abs: bool, neg: bool) -> u64 {
    let sign = match kind {
        Kind::F16 => 0x8000,
        Kind::F64 => 0x8000_0000_0000_0000,
        _ => 0x8000_0000,
    };
    let v = if abs { value & !sign } else { value };
    if neg { v ^ sign } else { v }
}

/// One lane's sources and context for [`valu`].
#[derive(Debug, Clone, Copy)]
pub struct Lane {
    /// Sources after modifiers, 32-bit kinds in the low half.
    pub s: [u64; 3],
    /// The lane's bit of the carry-in or select mask (`v_cndmask_b32`, `v_add_co_ci_u32`).
    pub mask: bool,
    /// The destination's value before (`v_fmac_f32`, `v_cvt_pk_u8_f32` keep parts of it).
    pub d: u64,
    /// The lane's number in its wave (`v_mbcnt_*`).
    pub lane: u32,
    pub mode: Mode,
    pub clamp: bool,
    /// Output modifier: 0 none, 1 times 2, 2 times 4, 3 half.
    pub omod: u8,
}

/// One lane's result of [`valu`]: the destination, the carry-out bit (`v_add_co_u32` ...) and,
/// for compares, the bit for the lane mask.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LaneOut {
    pub d: Option<u64>,
    pub carry: Option<bool>,
    pub cmp: Option<bool>,
}

fn out(d: u64) -> LaneOut {
    LaneOut { d: Some(d), carry: None, cmp: None }
}

/// The vector ALU on one lane: compares (12.9), then integer and bit operations, then floats.
/// Cross-lane operations (`v_readlane_b32`, DPP, `v_permlane*`) are the machine's.
pub fn valu(name: &str, l: &Lane) -> Result<LaneOut, String> {
    if name.starts_with("v_cmp_") || name.starts_with("v_cmpx_") {
        return compare(name, l).map(|c| LaneOut { d: None, carry: None, cmp: Some(c) });
    }
    if let Some(r) = valu_integer(name, l)? {
        return Ok(r);
    }
    if name.ends_with("_f64") || name.starts_with("v_cvt_f64_") {
        return valu_f64(name, l);
    }
    if name.contains("f16") {
        return valu_f16(name, l);
    }
    valu_f32(name, l)
}

/// Integer, bit and move operations of 12.7, 12.8 and 12.12, with their pseudocode; `None` for
/// any other opcode. The clamp bit saturates the additions and subtractions that name it (6.5:
/// "For integer operations, it clamps the result to the largest and smallest representable
/// value"); elsewhere a set clamp bit on an integer opcode is refused.
fn valu_integer(name: &str, l: &Lane) -> Result<Option<LaneOut>, String> {
    let [x, y, z] = l.s.map(|v| v as u32);
    let (xi, yi, zi) = (x as i32, y as i32, z as i32);
    let saturating = matches!(
        name,
        "v_add_nc_u32" | "v_sub_nc_u32" | "v_subrev_nc_u32" | "v_add_nc_i32" | "v_sub_nc_i32"
            | "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32"
    );
    let r = |v: u32| Ok(Some(out(u64::from(v))));
    let carry = |v: u32, c: bool| Ok(Some(LaneOut { d: Some(u64::from(v)), carry: Some(c), cmp: None }));
    let s24 = |v: u32| i64::from(((v << 8) as i32) >> 8);
    let u24 = |v: u32| u64::from(v & 0x00FF_FFFF);
    let result = match name {
        "v_mov_b32" => r(x),
        // 6.3 lists V_MOV_FED_B32 and no section gives what it does
        "v_mov_fed_b32" => return Err("v_mov_fed_b32: the reference lists it (6.3) without an operation".into()),
        "v_not_b32" => r(!x),
        "v_bfrev_b32" => r(x.reverse_bits()),
        "v_ffbh_u32" => r(if x == 0 { u32::MAX } else { x.leading_zeros() }),
        "v_ffbl_b32" => r(if x == 0 { u32::MAX } else { x.trailing_zeros() }),
        "v_ffbh_i32" => r((1..32).find(|&i| (x >> (31 - i) & 1) != x >> 31).unwrap_or(u32::MAX)),
        "v_cndmask_b32" => r(if l.mask { y } else { x }),
        "v_and_b32" => r(x & y),
        "v_or_b32" => r(x | y),
        "v_xor_b32" => r(x ^ y),
        "v_xnor_b32" => r(!(x ^ y)),
        "v_or3_b32" => r(x | y | z),
        "v_xor3_b32" => r(x ^ y ^ z),
        "v_and_or_b32" => r((x & y) | z),
        "v_lshl_or_b32" => r((x << (y & 31)) | z),
        "v_lshl_add_u32" => r((x << (y & 31)).wrapping_add(z)),
        "v_add_lshl_u32" => r(x.wrapping_add(y) << (z & 31)),
        "v_lshlrev_b32" => r(y << (x & 31)),
        "v_lshrrev_b32" => r(y >> (x & 31)),
        "v_ashrrev_i32" => r((yi >> (x & 31)) as u32),
        "v_lshlrev_b64" => Ok(Some(out(l.s[1] << (x & 63)))),
        "v_lshrrev_b64" => Ok(Some(out(l.s[1] >> (x & 63)))),
        "v_ashrrev_i64" => Ok(Some(out(((l.s[1] as i64) >> (x & 63)) as u64))),
        "v_bfe_u32" => r((x >> (y & 31)) & ((1u32 << (z & 31)) - 1)),
        "v_bfe_i32" => {
            return Err("v_bfe_i32: the reference's pseudocode (12.12) has no sign extension, which the \
                        name implies: two readings"
                .into());
        }
        "v_bfi_b32" => r((x & y) | (!x & z)),
        "v_bfm_b32" => r(((1u32 << (x & 31)) - 1) << (y & 31)),
        "v_alignbit_b32" => r(((u64::from(x) << 32 | u64::from(y)) >> (z & 31)) as u32),
        "v_alignbyte_b32" => {
            // ({S0, S1} >> (8 * count)) & 0xffffffff, the count S2[4:0] in RDNA1/2 12.12 and S2[1:0]
            // in RDNA3 16.12 (and GCN3 12.10): the two agree only for counts 0..3
            if z & 31 >= 4 {
                return Err(format!("v_alignbyte_b32 of a count of {}: the references take S2[4:0] and S2[1:0],                                     which disagree from 4 on", z & 31));
            }
            r(((u64::from(x) << 32 | u64::from(y)) >> (8 * (z & 3))) as u32)
        }
        "v_perm_b32" => r(perm(x, y, z)),
        "v_bcnt_u32_b32" => r(y.wrapping_add(x.count_ones())),
        "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
            // ThreadMask = (1LL << ThreadPosition) - 1, its low or high half against S0.
            let mask = (1u64 << l.lane) - 1;
            let half = if name.contains("_lo_") { mask as u32 } else { (mask >> 32) as u32 };
            r(y.wrapping_add((x & half).count_ones()))
        }
        "v_add_nc_u32" => r(if l.clamp { x.saturating_add(y) } else { x.wrapping_add(y) }),
        "v_sub_nc_u32" => r(if l.clamp { x.saturating_sub(y) } else { x.wrapping_sub(y) }),
        "v_subrev_nc_u32" => r(if l.clamp { y.saturating_sub(x) } else { y.wrapping_sub(x) }),
        "v_add_nc_i32" => r(if l.clamp { xi.saturating_add(yi) } else { xi.wrapping_add(yi) } as u32),
        "v_sub_nc_i32" => r(if l.clamp { xi.saturating_sub(yi) } else { xi.wrapping_sub(yi) } as u32),
        "v_add3_u32" => r(x.wrapping_add(y).wrapping_add(z)),
        "v_xad_u32" => r((x ^ y).wrapping_add(z)),
        "v_add_co_u32" => {
            let s = u64::from(x) + u64::from(y);
            carry(if l.clamp { s.min(0xFFFF_FFFF) as u32 } else { s as u32 }, s >> 32 != 0)
        }
        "v_sub_co_u32" => carry(if l.clamp { x.saturating_sub(y) } else { x.wrapping_sub(y) }, y > x),
        "v_subrev_co_u32" => carry(if l.clamp { y.saturating_sub(x) } else { y.wrapping_sub(x) }, x > y),
        "v_add_co_ci_u32" => {
            let s = u64::from(x) + u64::from(y) + u64::from(l.mask);
            carry(s as u32, s >> 32 != 0)
        }
        "v_sub_co_ci_u32" => {
            let c = u64::from(y) + u64::from(l.mask);
            carry(x.wrapping_sub(y).wrapping_sub(u32::from(l.mask)), c > u64::from(x))
        }
        "v_subrev_co_ci_u32" => {
            let c = u64::from(x) + u64::from(l.mask);
            carry(y.wrapping_sub(x).wrapping_sub(u32::from(l.mask)), c > u64::from(y))
        }
        "v_mul_lo_u32" => r(x.wrapping_mul(y)),
        "v_mul_hi_u32" => r(((u64::from(x) * u64::from(y)) >> 32) as u32),
        "v_mul_hi_i32" => r(((i64::from(xi) * i64::from(yi)) >> 32) as u32),
        "v_mul_u32_u24" => r((u24(x) * u24(y)) as u32),
        "v_mul_hi_u32_u24" => r(((u24(x) * u24(y)) >> 32) as u32),
        "v_mul_i32_i24" => r((s24(x) * s24(y)) as u32),
        "v_mul_hi_i32_i24" => r(((s24(x) * s24(y)) >> 32) as u32),
        "v_mad_u32_u24" => r(((u24(x) * u24(y)) as u32).wrapping_add(z)),
        "v_mad_i32_i24" => r(((s24(x) * s24(y)) as u32).wrapping_add(z)),
        "v_mad_u64_u32" => {
            let s = u128::from(x) * u128::from(y) + u128::from(l.s[2]);
            Ok(Some(LaneOut { d: Some(s as u64), carry: Some(s >> 64 != 0), cmp: None }))
        }
        "v_sad_u32" => r(x.abs_diff(y).wrapping_add(z)),
        "v_min_u32" => r(x.min(y)),
        "v_max_u32" => r(x.max(y)),
        "v_min_i32" => r(xi.min(yi) as u32),
        "v_max_i32" => r(xi.max(yi) as u32),
        "v_min3_u32" => r(x.min(y).min(z)),
        "v_max3_u32" => r(x.max(y).max(z)),
        "v_min3_i32" => r(xi.min(yi).min(zi) as u32),
        "v_max3_i32" => r(xi.max(yi).max(zi) as u32),
        "v_med3_u32" => r(x.max(y).min(x.min(y).max(z))),
        "v_med3_i32" => r(xi.max(yi).min(xi.min(yi).max(zi)) as u32),
        _ => return Ok(None),
    };
    if l.clamp && !saturating {
        return Err(format!("{name}: the clamp bit on an integer opcode that does not saturate"));
    }
    result
}

/// `byte_permute` of `V_PERM_B32` (12.12).
fn perm(s0: u32, s1: u32, sel: u32) -> u32 {
    let bytes = (u64::from(s0) << 32 | u64::from(s1)).to_le_bytes();
    let pick = |sel: u32| -> u32 {
        match sel {
            13.. => 0xFF,
            12 => 0,
            11 => u32::from(bytes[7] >> 7) * 0xFF,
            10 => u32::from(bytes[5] >> 7) * 0xFF,
            9 => u32::from(bytes[3] >> 7) * 0xFF,
            8 => u32::from(bytes[1] >> 7) * 0xFF,
            n => u32::from(bytes[n as usize]),
        }
    };
    (0..4).map(|i| pick(sel >> (8 * i) & 0xFF) << (8 * i)).fold(0, |a, b| a | b)
}

/// `V_MIN_F32` and `V_MAX_F32` of 12.7, case by case.
fn minmax32(mode: Mode, a: u32, b: u32, max: bool) -> u32 {
    let ieee = mode.ieee();
    if ieee && snan32(a) {
        return quiet32(a);
    }
    if ieee && snan32(b) {
        return quiet32(b);
    }
    if nan32(a) {
        return b;
    }
    if nan32(b) {
        return a;
    }
    let (x, y) = (in32(mode, a), in32(mode, b));
    match (a, b) {
        (0, 0x8000_0000) => return if max { a } else { b },
        (0x8000_0000, 0) => return if max { b } else { a },
        _ => {}
    }
    let pick_a = if max {
        if ieee { x >= y } else { x > y }
    } else {
        x < y
    };
    let r = if pick_a { x } else { y };
    out32(mode, r)
}

/// The floats of `V_FRACT_F32` and friends: `D.f = trunc(S0.f)` and its adjustments (12.8).
fn floor32(x: f32) -> f32 {
    let t = x.trunc();
    if x < 0.0 && x != t { t - 1.0 } else { t }
}

/// The clamp bit and the output modifier on a float result (6.2.2, 6.5): the modifier applies
/// only where output denormals are not allowed, flushes -0 to +0 and is ignored in IEEE mode;
/// clamp limits the result to [0, 1] (NaN to 0 with DX10_CLAMP).
fn finish32(l: &Lane, b: u32) -> u32 {
    let mut b = b;
    if l.omod != 0 && l.mode.denorm(false) & 2 == 0 && !l.mode.ieee() {
        let f = f32::from_bits(b);
        let f = match l.omod {
            1 => f * 2.0,
            2 => f * 4.0,
            _ => f * 0.5,
        };
        b = out32(l.mode, f);
        if b == 0x8000_0000 {
            b = 0;
        }
    }
    if l.clamp {
        let f = f32::from_bits(b);
        b = if f.is_nan() {
            if l.mode.dx10_clamp() { 0 } else { b }
        } else {
            f.clamp(0.0, 1.0).to_bits()
        };
    }
    b
}

/// Single-precision operations of 12.7, 12.8 and 12.12, under the denormal mode (6.4) for those
/// whose entries say "denormals are supported" or give no rule, always flushed for those whose
/// entries say "denormals are flushed".
fn valu_f32(name: &str, l: &Lane) -> Result<LaneOut, String> {
    l.mode.nearest(false)?;
    let [a, b, c] = l.s.map(|v| v as u32);
    let m = l.mode;
    let (x, y, z) = (in32(m, a), in32(m, b), in32(m, c));
    let arith = |f: f32| out32(m, f);
    // Past a NaN a minimum or maximum gives the other operand (12.7, 12.12); whether that
    // operand is flushed where the mode flushes denormals the entries do not say.
    if matches!(name, "v_min_f32" | "v_max_f32" | "v_min3_f32" | "v_max3_f32" | "v_med3_f32")
        && m.denorm(false) != 3
        && [a, b, c].into_iter().any(nan32)
        && [a, b, c].into_iter().any(denormal32)
    {
        return Err(format!("{name}: a NaN beside a denormal where the mode flushes denormals: whether the \
                            operand given back is flushed the reference does not say"));
    }
    let v: u32 = match name {
        "v_add_f32" => arith(x + y),
        "v_sub_f32" => arith(x - y),
        "v_subrev_f32" => arith(y - x),
        "v_mul_f32" => arith(x * y),
        // DX9 rules: 0.0 times anything (infinity and NaN too) is 0.0.
        "v_mul_legacy_f32" => {
            if x == 0.0 || y == 0.0 { 0 } else { arith(x * y) }
        }
        "v_fma_f32" => arith(x.mul_add(y, z)),
        "v_fmac_f32" => arith(x.mul_add(y, in32(m, l.d as u32))),
        // the literal K is source 1 of fmamk and source 2 of fmaak, as the decoder places it
        "v_fmamk_f32" | "v_fmaak_f32" => arith(x.mul_add(y, z)),
        "v_min_f32" => minmax32(m, a, b, false),
        "v_max_f32" => minmax32(m, a, b, true),
        "v_min3_f32" => minmax32(m, minmax32(m, a, b, false), c, false),
        "v_max3_f32" => minmax32(m, minmax32(m, a, b, true), c, true),
        "v_med3_f32" => {
            if nan32(a) || nan32(b) || nan32(c) {
                minmax32(m, minmax32(m, a, b, false), c, false)
            } else {
                let top = minmax32(m, minmax32(m, a, b, true), c, true);
                if top == minmax32(m, a, a, true) {
                    minmax32(m, b, c, true)
                } else if top == minmax32(m, b, b, true) {
                    minmax32(m, a, c, true)
                } else {
                    minmax32(m, a, b, true)
                }
            }
        }
        "v_floor_f32" => arith(floor32(x)),
        "v_trunc_f32" => arith(x.trunc()),
        "v_ceil_f32" => {
            let t = x.trunc();
            arith(if x > 0.0 && x != t { t + 1.0 } else { t })
        }
        "v_rndne_f32" => arith(x.round_ties_even()),
        "v_fract_f32" => {
            // D.f = S0.f + -floor(S0.f), as one single-precision addition
            if x.is_infinite() || x.is_nan() {
                return Err("v_fract_f32 of an infinity or NaN: the entry gives no case".into());
            }
            arith(x + -floor32(x))
        }
        "v_ldexp_f32" => {
            let e = b as i32;
            arith((f64::from(x) * 2f64.powi(e.clamp(-400, 400))) as f32)
        }
        "v_rcp_f32" => flush32(((1.0 / f64::from(f32::from_bits(flush32(a)))) as f32).to_bits()),
        "v_rcp_iflag_f32" => {
            // its own entry (12.8) has no "denormals are flushed": a denormal in or out is
            // refused, the rest is the reciprocal
            let r = ((1.0 / f64::from(f32::from_bits(a))) as f32).to_bits();
            if denormal32(a) || denormal32(r) {
                return Err("v_rcp_iflag_f32 of or to a denormal: its entry gives no denormal rule".into());
            }
            r
        }
        "v_rsq_f32" => {
            let v = f64::from(f32::from_bits(flush32(a)));
            flush32(((1.0 / v.sqrt()) as f32).to_bits())
        }
        "v_sqrt_f32" => flush32((f64::from(f32::from_bits(flush32(a))).sqrt() as f32).to_bits()),
        "v_exp_f32" => flush32((f64::from(f32::from_bits(flush32(a))).exp2() as f32).to_bits()),
        "v_log_f32" => {
            let v = f32::from_bits(flush32(a));
            if v == 0.0 { 0xFF80_0000 } else { flush32((f64::from(v).log2() as f32).to_bits()) }
        }
        "v_sin_f32" | "v_cos_f32" => {
            // D.f = sin(S0.f * 2 * PI): the input in turns; its whole turns change nothing
            if x.is_infinite() || x.is_nan() {
                0xFFC0_0000
            } else {
                let turns = f64::from(x) - f64::from(x).floor();
                let r = 2.0 * std::f64::consts::PI * turns;
                let v = if name == "v_sin_f32" { r.sin() } else { r.cos() };
                // whole quarter turns are exact (12.8's example: cos(0.25) is 0x0)
                let quarter = [(0.0, 0.0, 1.0), (0.25, 1.0, 0.0), (0.5, 0.0, -1.0), (0.75, -1.0, 0.0)]
                    .into_iter()
                    .find(|q| q.0 == turns)
                    .map(|(_, s, c)| if name == "v_sin_f32" { s } else { c });
                let v = quarter.unwrap_or(v);
                let v = if name == "v_sin_f32" && x == 0.0 { f64::from(x) } else { v };
                out32(m, v as f32)
            }
        }
        "v_cvt_f32_u32" => arith(a as f32),
        "v_cvt_f32_i32" => arith(a as i32 as f32),
        "v_cvt_u32_f32" => return Ok(out(u64::from(if x.is_nan() { 0 } else { x as u32 }))),
        "v_cvt_i32_f32" => return Ok(out(u64::from(if x.is_nan() { 0 } else { x as i32 as u32 }))),
        "v_cvt_flr_i32_f32" => {
            return Ok(out(u64::from(if x.is_nan() { 0 } else { floor32(x) as i32 as u32 })));
        }
        "v_cvt_rpi_i32_f32" => {
            // floor(S0.f + 0.5) of the value: the sum in double is exact, a single sum would round
            return Ok(out(u64::from(if x.is_nan() { 0 } else { (f64::from(x) + 0.5).floor() as i32 as u32 })));
        }
        "v_cvt_f32_ubyte0" => arith((a & 0xFF) as f32),
        "v_cvt_f32_ubyte1" => arith((a >> 8 & 0xFF) as f32),
        "v_cvt_f32_ubyte2" => arith((a >> 16 & 0xFF) as f32),
        "v_cvt_f32_ubyte3" => arith((a >> 24) as f32),
        "v_cvt_pk_u8_f32" => {
            return Err("v_cvt_pk_u8_f32: flt32_to_uint8's rounding is not defined in the reference".into());
        }
        "v_fma_mix_f32" => return Err("v_fma_mix_f32 goes through the machine's mix sources".into()),
        _ => return Err(format!("{name}: not in the oracle's vector ALU")),
    };
    Ok(out(u64::from(finish32(l, v))))
}

/// `V_FMA_MIX_F32` (12.10): each source a single (`opsel_hi` clear) or the low or high half
/// (`opsel`) as a half, widened exactly; `neg_hi` is `abs` here; then a fused multiply-add.
pub fn fma_mix(l: &Lane, opsel: u8, opsel_hi: u8, neg: u8, abs: u8) -> Result<u32, String> {
    l.mode.nearest(false)?;
    let src = |i: usize| -> u32 {
        let v = l.s[i] as u32;
        let wide = if opsel_hi >> i & 1 == 0 {
            v
        } else {
            f16_to_f32(if opsel >> i & 1 == 1 { (v >> 16) as u16 } else { v as u16 })
        };
        modify(u64::from(wide), Kind::F32, abs >> i & 1 == 1, neg >> i & 1 == 1) as u32
    };
    let m = l.mode;
    let r = in32(m, src(0)).mul_add(in32(m, src(1)), in32(m, src(2)));
    Ok(finish32(l, out32(m, r)))
}

/// Double-precision operations (12.8, 12.12).
fn valu_f64(name: &str, l: &Lane) -> Result<LaneOut, String> {
    l.mode.nearest(true)?;
    let m = l.mode;
    let [a, b, c] = l.s;
    let (x, y, z) = (in64(m, a), in64(m, b), in64(m, c));
    let v: u64 = match name {
        "v_add_f64" => out64(m, x + y),
        "v_mul_f64" => out64(m, x * y),
        "v_fma_f64" => out64(m, x.mul_add(y, z)),
        "v_min_f64" | "v_max_f64" => {
            let max = name == "v_max_f64";
            if m.denorm(true) != 3 && (nan64(a) || nan64(b)) && (denormal64(a) || denormal64(b)) {
                return Err(format!("{name}: a NaN beside a denormal where the mode flushes denormals: whether \
                                    the operand given back is flushed the reference does not say"));
            }
            let snan = |v: u64| nan64(v) && v & 0x0008_0000_0000_0000 == 0;
            if m.ieee() && snan(a) {
                a | 0x0008_0000_0000_0000
            } else if m.ieee() && snan(b) {
                b | 0x0008_0000_0000_0000
            } else if nan64(a) {
                b
            } else if nan64(b) {
                a
            } else if (a, b) == (0, 1 << 63) {
                if max { a } else { b }
            } else if (a, b) == (1 << 63, 0) {
                if max { b } else { a }
            } else if max {
                if (m.ieee() && x >= y) || x > y { out64(m, x) } else { out64(m, y) }
            } else if x < y {
                out64(m, x)
            } else {
                out64(m, y)
            }
        }
        "v_cvt_f64_f32" => out64(m, f64::from(in32(m, a as u32))),
        "v_cvt_f64_i32" => (a as u32 as i32 as f64).to_bits(),
        "v_cvt_f64_u32" => (a as u32 as f64).to_bits(),
        "v_cvt_f32_f64" => {
            l.mode.nearest(false)?;
            return Ok(out(u64::from(finish32(l, out32(m, x as f32)))));
        }
        "v_cvt_i32_f64" => return Ok(out(u64::from(if x.is_nan() { 0 } else { x as i32 as u32 }))),
        "v_cvt_u32_f64" => return Ok(out(u64::from(if x.is_nan() { 0 } else { x as u32 }))),
        _ => return Err(format!("{name}: not in the oracle's vector ALU")),
    };
    if l.clamp || l.omod != 0 {
        return Err(format!("{name}: clamp or an output modifier on a double"));
    }
    Ok(out(v))
}

/// The half-precision conversions and packs (12.8, 12.12). Half arithmetic is not modelled yet.
fn valu_f16(name: &str, l: &Lane) -> Result<LaneOut, String> {
    let [a, b, _] = l.s.map(|v| v as u32);
    match name {
        "v_cvt_f16_f32" => {
            l.mode.nearest(true)?;
            if l.clamp || l.omod != 0 {
                return Err("v_cvt_f16_f32 with clamp or an output modifier".into());
            }
            // the low half; the high half of the destination is left as it was
            let h = f32_to_f16(l.mode, flush_in(l.mode, a));
            Ok(out((l.d & 0xFFFF_0000) | u64::from(h)))
        }
        "v_cvt_f32_f16" => Ok(out(u64::from(finish32(l, f16_to_f32(a as u16))))),
        "v_pack_b32_f16" => Ok(out(u64::from((b & 0xFFFF) << 16 | (a & 0xFFFF)))),
        _ => Err(format!("{name}: half arithmetic is not in the oracle yet")),
    }
}

fn flush_in(mode: Mode, b: u32) -> u32 {
    in32(mode, b).to_bits()
}

/// Vector compares (12.9): `v_cmp[x]_<op>_<type>`, D[threadId] = (S0 <op> S1). Ordered float
/// compares are false with a NaN; the `n` forms are their negations ("With NAN inputs this is
/// not the same operation"); `o` and `u` ask for neither or either NaN; `class` per its list.
pub fn compare(name: &str, l: &Lane) -> Result<bool, String> {
    let rest = name.trim_start_matches("v_cmpx_").trim_start_matches("v_cmp_");
    let (op, ty) = rest.rsplit_once('_').ok_or_else(|| format!("{name}: no type"))?;
    let [a, b, _] = l.s;
    if op == "class" {
        let class = b as u32;
        let (neg, bits) = match ty {
            "f32" => {
                let v = a as u32;
                (v >> 31 == 1, classify(f64::from(f32::from_bits(v)), snan32(v), denormal32(v)))
            }
            "f64" => {
                let snan = nan64(a) && a & 0x0008_0000_0000_0000 == 0;
                (a >> 63 == 1, classify(f64::from_bits(a), snan, denormal64(a)))
            }
            _ => return Err(format!("{name}: not in the oracle's compares")),
        };
        let bit = match bits {
            Class::Snan => 0,
            Class::Qnan => 1,
            Class::Inf => if neg { 2 } else { 9 },
            Class::Normal => if neg { 3 } else { 8 },
            Class::Denormal => if neg { 4 } else { 7 },
            Class::Zero => if neg { 5 } else { 6 },
        };
        return Ok(class >> bit & 1 == 1);
    }
    let int = |x: i128, y: i128| -> Result<bool, String> {
        Ok(match op {
            "f" => false,
            "t" | "tru" => true,
            "lt" => x < y,
            "eq" => x == y,
            "le" => x <= y,
            "gt" => x > y,
            "ne" | "lg" => x != y,
            "ge" => x >= y,
            _ => return Err(format!("{name}: no such compare")),
        })
    };
    let float = |x: f64, y: f64| -> Result<bool, String> {
        use std::cmp::Ordering::{Equal, Greater, Less};
        // `None`: unordered (a NaN)
        let o = x.partial_cmp(&y);
        Ok(match op {
            "f" => false,
            "tru" => true,
            "lt" => o == Some(Less),
            "eq" => o == Some(Equal),
            "le" => matches!(o, Some(Less | Equal)),
            "gt" => o == Some(Greater),
            "lg" => matches!(o, Some(Less | Greater)),
            "ge" => matches!(o, Some(Greater | Equal)),
            "o" => o.is_some(),
            "u" => o.is_none(),
            "nge" => !matches!(o, Some(Greater | Equal)),
            "nlg" => !matches!(o, Some(Less | Greater)),
            "ngt" => o != Some(Greater),
            "nle" => !matches!(o, Some(Less | Equal)),
            "neq" => o != Some(Equal),
            "nlt" => o != Some(Less),
            _ => return Err(format!("{name}: no such compare")),
        })
    };
    match ty {
        "u32" => int(i128::from(a as u32), i128::from(b as u32)),
        "i32" => int(i128::from(a as u32 as i32), i128::from(b as u32 as i32)),
        "u64" => int(i128::from(a), i128::from(b)),
        "i64" => int(i128::from(a as i64), i128::from(b as i64)),
        "u16" => int(i128::from(a as u16), i128::from(b as u16)),
        "i16" => int(i128::from(a as u16 as i16), i128::from(b as u16 as i16)),
        "f32" => float(f64::from(in32(l.mode, a as u32)), f64::from(in32(l.mode, b as u32))),
        "f64" => float(in64(l.mode, a), in64(l.mode, b)),
        "f16" => float(
            f64::from(f32::from_bits(f16_to_f32(a as u16))),
            f64::from(f32::from_bits(f16_to_f32(b as u16))),
        ),
        _ => Err(format!("{name}: not in the oracle's compares")),
    }
}

enum Class {
    Snan,
    Qnan,
    Inf,
    Normal,
    Denormal,
    Zero,
}

fn classify(v: f64, snan: bool, denormal: bool) -> Class {
    if v.is_nan() {
        if snan { Class::Snan } else { Class::Qnan }
    } else if v.is_infinite() {
        Class::Inf
    } else if v == 0.0 {
        Class::Zero
    } else if denormal {
        Class::Denormal
    } else {
        Class::Normal
    }
}

// ---------------------------------------------------------------------------------------------
// Which lane a cross-lane operation reads (6.9 DPP, 13.3.9 DPP16, 13.3.10 DPP8,
// 12.12 V_PERMLANE*, 12.13.1 DS_SWIZZLE_B32, 10.4.4 and 12.13 DS_[B]PERMUTE_B32)
// ---------------------------------------------------------------------------------------------

/// The lane DPP16's `DPP_CTRL` reads for `lane` (13.3.9, Table 91), `None` where the source is
/// out of range (row shifts past the row's end). The table defines quad_perm (0x00-0xFF),
/// row_shl (0x101-0x10F), row_shr (0x111-0x11F), row_ror (0x121-0x12F), row_mirror (0x140) and
/// row_half_mirror (0x141); any other control is refused.
pub fn dpp16_source(ctrl: u16, lane: usize) -> Result<Option<usize>, String> {
    let n = lane;
    let row = n & !0xF;
    let c = usize::from(ctrl & 0xF);
    Ok(match ctrl {
        0x000..=0x0FF => Some((n & 0x3C) + usize::from(ctrl >> ((n % 4) * 2) & 3)),
        0x101..=0x10F => ((n & 0xF) < 16 - c).then_some(n + c),
        0x111..=0x11F => ((n & 0xF) >= c).then(|| n - c),
        0x121..=0x12F => Some(if (n & 0xF) >= c { n - c } else { n + 16 - c }),
        0x140 => Some(row + 15 - (n & 0xF)),
        0x141 => Some((n & !7) + 7 - (n & 7)),
        _ => return Err(format!("DPP control {ctrl:#x} is not in the reference's Table 91")),
    })
}

/// The lane DPP8 reads for `lane` (13.3.10, Table 92): the select of its position in its group
/// of eight.
pub fn dpp8_source(sel: u32, lane: usize) -> usize {
    (lane & !7) | (sel >> (3 * (lane & 7)) & 7) as usize
}

/// The lane `V_PERMLANE16_B32` (or, `cross`, `V_PERMLANEX16_B32`) reads for `lane` (12.12):
/// `lanesel = {S2.u, S1.u}`, the row's own (or the other of its pair, `altrow = {row[1],
/// ~row[0]}`).
pub fn permlane_source(s1: u32, s2: u32, lane: usize, cross: bool) -> usize {
    let sel = u64::from(s1) | u64::from(s2) << 32;
    let row = lane / 16;
    let row = if cross { (row & 2) | (!row & 1) } else { row };
    row * 16 + (sel >> (4 * (lane % 16)) & 0xF) as usize
}

/// The lane `DS_SWIZZLE_B32` reads for `lane` (12.13.1's pseudocode, over `offset1:offset0`).
/// In the rotate mode with a mask, the section's worked example ("0x01, mask=1, rotate left:
/// {2,1,4,7,...}") is not what its pseudocode gives (thread 0 reads thread 0 there): refused.
pub fn swizzle_source(offset: u16, lane: usize) -> Result<usize, String> {
    let i = lane;
    let o = usize::from(offset);
    if (0xC000..0xE000).contains(&o) && o & 0x1F != 0 {
        return Err("ds_swizzle_b32's rotate mode with a mask: 12.13.1's example and pseudocode disagree".into());
    }
    Ok(if o >= 0xE000 {
        let mask = o & 0x1F;
        let mut j = ((i & 0x1F) as u32).reverse_bits() as usize >> 27;
        j >>= mask.count_ones();
        j |= i & mask;
        j | (i & 0x20)
    } else if o >= 0xC000 {
        let rotate = (o >> 5 & 0x1F) as isize;
        let rotate = if o >> 10 & 1 == 1 { -rotate } else { rotate };
        let mask = o & 0x1F;
        let j = (i & mask) | ((i as isize + rotate) as usize & !mask);
        // (the pseudocode keeps j in the lane's group of 32)
        (j & 0x1F) | (i & 0x20)
    } else if o >> 15 & 1 == 1 {
        (i & !3) + (o >> (2 * (i & 3)) & 3)
    } else {
        let (xor, or, and) = (o >> 10 & 0x1F, o >> 5 & 0x1F, o & 0x1F);
        ((((i & 0x1F) & and) | or) ^ xor) | (i & 0x20)
    })
}

/// How a wave64 permute picks its lane: the reference gives two readings. 10.4.4 says "in wave64
/// mode the permute operates only across 32 lanes at a time of each half of a wave64 ... Each
/// half-wave can use indices in the range 0-31"; the entries of 12.13 say `src_lane =
/// floor((VGPR[ADDR][i] + OFFSET) / 4) mod 64`. The caller chooses; with no choice made the
/// oracle refuses a wave64 permute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Wave64Permute {
    #[default]
    Unchosen,
    /// 10.4.4: within the lane's own half.
    Halves,
    /// 12.13: across the 64 lanes.
    Whole,
}

/// The lane a `DS_BPERMUTE_B32` reads, or a `DS_PERMUTE_B32` writes, for an address `addr`
/// (byte index plus the instruction's offset).
pub fn permute_lane(addr: u32, lane: usize, lanes: usize, reading: Wave64Permute) -> Result<usize, String> {
    let index = (addr / 4) as usize;
    if lanes == 32 {
        return Ok(index % 32);
    }
    match reading {
        Wave64Permute::Halves => Ok((lane & 32) | (index % 32)),
        Wave64Permute::Whole => Ok(index % 64),
        Wave64Permute::Unchosen => Err("a wave64 permute: the reference's 10.4.4 and 12.13 disagree on \
                                        its lanes (per half or across 64); none chosen"
            .into()),
    }
}

// ---------------------------------------------------------------------------------------------
// LDS atomics and memory atomics (12.13; 12.14 BUFFER_ATOMIC_*; 12.18 GLOBAL_ATOMIC_*)
// ---------------------------------------------------------------------------------------------

/// The new value of a 32-bit read-modify-write `op` (the name without its unit's prefix and
/// `_rtn`/type suffixes: `add`, `sub`, `rsub`, `inc`, `dec`, `min_i32`, ...) on `old` with the
/// instruction's data (`data2` for the two-operand forms). Pseudocode of 12.13 (DS_ADD_U32
/// "MEM[ADDR] += DATA" and its siblings) and 12.14.
pub fn atomic32(op: &str, old: u32, data: u32, data2: u32) -> Result<u32, String> {
    Ok(match op {
        "add" | "add_u32" => old.wrapping_add(data),
        "sub" | "sub_u32" => old.wrapping_sub(data),
        "rsub" | "rsub_u32" => data.wrapping_sub(old),
        "inc" | "inc_u32" => if old >= data { 0 } else { old + 1 },
        "dec" | "dec_u32" => if old == 0 || old > data { data } else { old - 1 },
        "min_i32" | "smin" => (old as i32).min(data as i32) as u32,
        "max_i32" | "smax" => (old as i32).max(data as i32) as u32,
        "min_u32" | "umin" => old.min(data),
        "max_u32" | "umax" => old.max(data),
        "and" | "and_b32" => old & data,
        "or" | "or_b32" => old | data,
        "xor" | "xor_b32" => old ^ data,
        "swap" | "wrxchg" | "wrxchg_b32" => data,
        "mskor" | "mskor_b32" => (old & !data) | data2,
        // DS_CMPST: "MEM[ADDR] = (tmp == cmp) ? src : tmp" with cmp = DATA, src = DATA2.
        "cmpst" | "cmpst_b32" => if old == data { data2 } else { old },
        // BUFFER_ATOMIC_CMPSWAP takes them the other way round (12.13's caution): src first.
        "cmpswap" => if old == data2 { data } else { old },
        _ => return Err(format!("atomic {op}: not in the oracle")),
    })
}

// ---------------------------------------------------------------------------------------------
// Buffers (8.1.4 Buffer Data, 8.1.5 Buffer Addressing and Range Checking, 8.1.7 Alignment,
// 8.1.8 Buffer Resource; 7.2.1 scalar loads through a buffer constant)
// ---------------------------------------------------------------------------------------------

/// A buffer resource (V#), decoded as 8.1.8's Table 37 lays it out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferResource {
    pub base: u64,
    pub stride: u32,
    pub swizzle: bool,
    pub num_records: u32,
    pub dst_sel: [u8; 4],
    pub format: u8,
    pub index_stride: u32,
    pub add_tid: bool,
    pub oob_select: u8,
    pub kind: u8,
    /// "A resource set to all zeros acts as an unbound texture or buffer (return 0,0,0,0)."
    pub unbound: bool,
}

impl BufferResource {
    pub fn decode(w: [u32; 4]) -> BufferResource {
        BufferResource {
            base: u64::from(w[0]) | u64::from(w[1] & 0xFFFF) << 32,
            stride: w[1] >> 16 & 0x3FFF,
            swizzle: w[1] >> 31 == 1,
            num_records: w[2],
            dst_sel: [0, 3, 6, 9].map(|s| (w[3] >> s & 7) as u8),
            format: (w[3] >> 12 & 0x7F) as u8,
            index_stride: 8 << (w[3] >> 21 & 3),
            add_tid: w[3] >> 23 & 1 == 1,
            oob_select: (w[3] >> 28 & 3) as u8,
            kind: (w[3] >> 30) as u8,
            unbound: w == [0; 4],
        }
    }

    /// The resource's state for a buffer access, or why the oracle will not take it: a non-zero
    /// resource of the INVALID format is "an unbound resource" by 8.1.4, and what the console
    /// does with one that is not all zeros the reference leaves open; a type other than buffer
    /// (8.1.8: "Value == 0 for buffer") is no buffer.
    pub fn check(&self) -> Result<(), String> {
        if self.unbound {
            return Ok(());
        }
        if self.kind != 0 {
            return Err(format!("a buffer access through a resource of type {} (not a buffer)", self.kind));
        }
        if self.format == 0 {
            return Err("a buffer resource of the INVALID format that is not all zeros: 8.1.4 calls the \
                        format an unbound resource"
                .into());
        }
        if self.swizzle && self.stride != 0 && !self.stride.is_multiple_of(4) {
            return Err(format!("a swizzled buffer of stride {}: 8.1.5's swizzle is of dwords, the stride \
                                no multiple of 4",
                self.stride));
        }
        if !self.swizzle && self.stride != 0 && self.oob_select == 3 {
            return Err("an unswizzled buffer with a stride and OOB_SELECT 3: Tables 33 and 37 disagree on \
                        the units of NUM_RECORDS there"
                .into());
        }
        Ok(())
    }
}

/// One lane's address of a buffer access: the byte address, and the `index` and `offset` the
/// range check of Table 35 is asked of (8.1.5; the swizzled form by its formula).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferAddress {
    pub address: u64,
    pub index: u32,
    pub offset: u32,
}

/// 8.1.5: `index` from a VGPR (or zero) plus the thread id with `add_tid`; `offset` from a VGPR
/// (or zero) plus the instruction's; the address `base + sgpr_offset + buffer_offset`, linear
/// (`index * stride + offset`) or swizzled.
pub fn buffer_address(r: &BufferResource, vindex: u32, voffset: u32, inst_offset: u32, soffset: u32, lane: u32) -> BufferAddress {
    let index = vindex.wrapping_add(if r.add_tid { lane & 63 } else { 0 });
    let offset = voffset.wrapping_add(inst_offset);
    let buffer_offset = if r.swizzle && r.stride != 0 {
        let element = 4u64;
        let (is, stride) = (u64::from(r.index_stride), u64::from(r.stride));
        let (index_msb, index_lsb) = (u64::from(index) / is, u64::from(index) % is);
        let (offset_msb, offset_lsb) = (u64::from(offset) / element, u64::from(offset) % element);
        (index_msb * stride + offset_msb * element) * is + index_lsb * element + offset_lsb
    } else {
        u64::from(index) * u64::from(r.stride) + u64::from(offset)
    };
    BufferAddress { address: r.base + u64::from(soffset) + buffer_offset, index, offset }
}

/// Table 35, the range check of OOB_SELECT, for an access of `payload` bytes at `offset` (one
/// component for the dword loads and stores, note 4; the whole element for format and atomic
/// accesses, note 3). In mode 3 "num_records is reduced by sgpr_offset".
pub fn out_of_range(r: &BufferResource, a: &BufferAddress, payload: u32, soffset: u32) -> bool {
    let records = r.num_records;
    let offset = u64::from(a.offset);
    match r.oob_select {
        0 => a.index >= records || offset >= u64::from(r.stride),
        1 => a.index >= records,
        2 => records == 0,
        _ => {
            if r.swizzle && r.stride != 0 {
                a.index >= records || offset + u64::from(payload) > u64::from(r.stride)
            } else {
                offset + u64::from(payload) > u64::from(records.saturating_sub(soffset))
            }
        }
    }
}

/// A component of a format load (`BUFFER_LOAD_FORMAT_*`, 8.1.4: format and dst_sel from the
/// resource), from the bytes of one element in `format` (8.2.8, Table 47): the channel's value
/// as a 32-bit register holds it, and whether its rounding is pinned (a 32-bit or 16-bit float or
/// integer channel is exact; the normalized and scaled conversions are not given by the
/// reference). `None` where the format is not modelled.
pub fn format_channel(format: u8, element: &[u8], channel: usize) -> Option<(u32, bool)> {
    let (bits, channels, kind) = layout(format)?;
    if channel >= channels {
        return None;
    }
    let bytes = bits / 8;
    let at = channel * bytes;
    let raw = element.get(at..at + bytes)?;
    let raw = raw.iter().rev().fold(0u32, |acc, &b| acc << 8 | u32::from(b));
    let max = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
    Some(match kind {
        'u' => (raw, true),
        's' => (((raw << (32 - bits)) as i32 >> (32 - bits)) as u32, true),
        'f' if bits == 16 => (f16_to_f32(raw as u16), true),
        'f' => (raw, true),
        'n' => (((f64::from(raw) / f64::from(max)) as f32).to_bits(), false),
        _ => {
            let v = ((raw << (32 - bits)) as i32 >> (32 - bits)) as f64;
            ((((v / f64::from(max >> 1)).max(-1.0)) as f32).to_bits(), false)
        }
    })
}

/// Bytes of one channel of a modelled format (Table 47): what a format access must be aligned
/// to (8.1.7).
pub fn format_channel_bytes(format: u8) -> Option<u32> {
    layout(format).map(|(bits, _, _)| bits as u32 / 8)
}

/// A modelled format (Table 47): bits per channel, channels, and the kind (u uint, s sint,
/// f float, n unorm, m snorm).
fn layout(format: u8) -> Option<(usize, usize, char)> {
    Some(match format {
        1 => (8, 1, 'n'),
        2 => (8, 1, 'm'),
        5 => (8, 1, 'u'),
        6 => (8, 1, 's'),
        7 => (16, 1, 'n'),
        8 => (16, 1, 'm'),
        11 => (16, 1, 'u'),
        12 => (16, 1, 's'),
        13 => (16, 1, 'f'),
        14 => (8, 2, 'n'),
        18 => (8, 2, 'u'),
        20 => (32, 1, 'u'),
        21 => (32, 1, 's'),
        22 => (32, 1, 'f'),
        23 => (16, 2, 'n'),
        27 => (16, 2, 'u'),
        28 => (16, 2, 's'),
        29 => (16, 2, 'f'),
        56 => (8, 4, 'n'),
        57 => (8, 4, 'm'),
        60 => (8, 4, 'u'),
        61 => (8, 4, 's'),
        62 => (32, 2, 'u'),
        63 => (32, 2, 's'),
        64 => (32, 2, 'f'),
        65 => (16, 4, 'n'),
        69 => (16, 4, 'u'),
        70 => (16, 4, 's'),
        71 => (16, 4, 'f'),
        72 => (32, 3, 'u'),
        73 => (32, 3, 's'),
        74 => (32, 3, 'f'),
        75 => (32, 4, 'u'),
        76 => (32, 4, 's'),
        77 => (32, 4, 'f'),
        _ => return None,
    })
}

/// Bytes of one element of a modelled format (Table 47), for the all-or-nothing range check.
pub fn format_bytes(format: u8) -> Option<u32> {
    Some(match format {
        1 | 2 | 5 | 6 => 1,
        7 | 8 | 11 | 12 | 13 | 14 | 18 => 2,
        20..=29 | 56..=61 => 4,
        62..=71 => 8,
        72..=74 => 12,
        75..=77 => 16,
        _ => return None,
    })
}

/// How much the oracle trusts an instruction's own rounding, against what the host does with the
/// same operation, for the comparison's second class: ULPs (and an absolute error) the oracle
/// allows a result of `name` to differ by. Zero for what the reference pins exactly (integer and
/// bit operations, moves, single additions, multiplications and conversions it gives as 0.5 ULP
/// or exact; the float-to-integer conversions, whose results are the integers their pseudocode
/// gives, are exact although 12.8 words them as "1ULP"). The others, with the reference's
/// words, and what is the oracle's own allowance for a host (not the reference's) said as such:
/// - `v_rcp_f32`, `v_rcp_iflag_f32`, `v_rsq_f32`, `v_sqrt_f32`, `v_exp_f32`, `v_log_f32`: "1ULP
///   accuracy" on the console (12.8); a host allowance of 1 more (3 more for exp and log),
///   not validated against any host's documented accuracy.
/// - `v_sin_f32`, `v_cos_f32`: the reference gives no accuracy; 2 ULPs or 2^-10 absolute is a
///   host allowance only.
/// - fused multiply-adds (`v_fma_f32`, `v_fmac_f32`, `v_fmaak_f32`, `v_fmamk_f32`,
///   `v_fma_mix_f32`, `v_fma_f64`): 0.5 ULP and fused on the console (12.7, 12.12); 1 ULP is a
///   host allowance for an unfused host, which does not hold under cancellation (a host that
///   does not fuse there fails the comparison, it is not passed). A 64-bit result is never
///   compared approximately: the oracle refuses it ([`super::machine`]).
pub fn own_tolerance(name: &str) -> (u32, f32) {
    match name {
        "v_rcp_f32" | "v_rcp_iflag_f32" | "v_rsq_f32" | "v_sqrt_f32" => (2, 0.0),
        "v_exp_f32" | "v_log_f32" => (4, 0.0),
        "v_sin_f32" | "v_cos_f32" => (2, 1.0 / 1024.0),
        "v_fma_f32" | "v_fmac_f32" | "v_fmaak_f32" | "v_fmamk_f32" | "v_fma_mix_f32" | "v_fma_f64" => (1, 0.0),
        _ => (0, 0.0),
    }
}

/// Whether `name`'s result is a single-precision float, whose NaNs' bits the reference does not
/// pin (an integer from a float conversion, a half, or a mask is not one).
pub fn float_result(name: &str) -> bool {
    let not_f32 = [
        "v_cvt_i32_", "v_cvt_u32_", "v_cvt_flr_i32_", "v_cvt_rpi_i32_", "v_cvt_pk", "v_cvt_i16_", "v_cvt_u16_",
        "v_cvt_f16_", "v_frexp_exp_", "v_cmp",
    ];
    name.starts_with("v_")
        && (name.contains("_f32") || name.contains("_f64") || name.contains("_f16"))
        && !not_f32.iter().any(|p| name.starts_with(p))
        && !matches!(name, "v_cndmask_b32" | "v_mov_b32")
}

/// Whether `name`'s result holds halves (a conversion to a half, a pack of halves, half
/// arithmetic), whose NaNs' bits the reference does not pin.
pub fn half_result(name: &str) -> bool {
    name.starts_with("v_") && name.contains("f16") && !name.starts_with("v_cmp")
        && !["v_cvt_f32_", "v_cvt_f64_", "v_cvt_i", "v_cvt_u", "v_frexp_exp_", "v_cvt_norm_"].iter().any(|p| name.starts_with(p))
}

/// How an instruction carries an approximate input (the comparison's second class) into its
/// result where a bound still holds; [`carries`] gives `None` for the rest (sums, which cancel;
/// conversions to integers, compares and bit operations, whose results have no ULPs; the
/// transcendentals, whose error grows with the input), and the oracle refuses those.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Carry {
    /// A copy of a source: any trust passes as it is.
    Copy,
    /// One of the sources by their order (minimum, maximum, median): the ULPs pass; a NaN of
    /// unpinned bits does not (IEEE mode looks at its quiet bit).
    Select,
    /// A product: the relative errors add and the result rounds once (2 max + 1).
    Product,
    /// A reciprocal or a square root (or both): the relative error is at most the input's, and
    /// the result rounds once (+1).
    Root,
}

pub fn carries(name: &str) -> Option<Carry> {
    Some(match name {
        "v_mov_b32" | "v_cndmask_b32" => Carry::Copy,
        "v_min_f32" | "v_max_f32" | "v_min3_f32" | "v_max3_f32" | "v_med3_f32" => Carry::Select,
        "v_mul_f32" => Carry::Product,
        "v_rcp_f32" | "v_rsq_f32" | "v_sqrt_f32" => Carry::Root,
        _ => return None,
    })
}

/// The class of atomic `op` (as [`atomic32`] names it) whose members leave the same value
/// whatever order the lanes take them in (additions with subtractions; each of and, or, xor,
/// and the signed and unsigned minimum and maximum on its own): the hardware's order then
/// changes what each lane gets back, not what is left. `None`: the order matters.
pub fn commute_class(op: &str) -> Option<u8> {
    Some(match op {
        "add" | "add_u32" | "sub" | "sub_u32" => 1,
        "and" | "and_b32" => 2,
        "or" | "or_b32" => 3,
        "xor" | "xor_b32" => 4,
        "min_i32" | "smin" => 5,
        "max_i32" | "smax" => 6,
        "min_u32" | "umin" => 7,
        "max_u32" | "umax" => 8,
        _ => return None,
    })
}
