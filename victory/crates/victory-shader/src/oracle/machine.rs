//! The interpreter's machine: waves, registers, LDS and memory, which lanes run, which
//! registers an instruction reads and writes, and the order of the group's waves. What an
//! instruction computes is [`super::semantics`]; the reference's sections on state (3.x),
//! operands (6.2) and memory addressing (7.2, 8.1, 9.3, 10.4) are cited where they decide
//! something here.

use std::collections::HashMap;

use super::semantics::{self as sem, BufferResource, Kind, Lane, Mode, SaluIn};
use super::{Group, Memory, Ran, Refusal, Seen, Tol, Who, WriteKind};
use crate::gfx10::{self, Dpp, Encoding, Fields, Instruction, Operand, Valu};

/// Where the program's code is, for `s_getpc_b64` and `s_setpc_b64` (any address does: the
/// program only ever adds offsets to what `s_getpc_b64` gave it).
const CODE_BASE: u64 = 0x0000_7F00_0000_0000;
const VCC: usize = 106;
const M0: usize = 124;
const NULL: u8 = 125;
const EXEC: usize = 126;

type R<T> = Result<T, String>;

struct Wave {
    s: [u32; 128],
    st: [Tol; 128],
    v: Vec<[u32; 64]>,
    vt: Vec<[Tol; 64]>,
    scc: bool,
    scc_t: Tol,
    pc: usize,
    done: bool,
    at_barrier: bool,
    mode: Mode,
    m0_written: bool,
    steps: u64,
}

struct Machine<'a> {
    g: &'a Group,
    mem: &'a mut Memory,
    lanes: usize,
    waves: Vec<Wave>,
    lds: Vec<u8>,
    /// Per byte.
    lds_t: Vec<Tol>,
    /// Per dword, the waves that accessed it between the current two barriers.
    lds_seen: Vec<Seen>,
    /// Barriers the group has passed.
    epoch: u32,
    fetched: HashMap<usize, Instruction>,
    ran: Ran,
}

pub(super) fn run(g: &Group, mem: &mut Memory) -> Result<Ran, Refusal> {
    let refuse = |reason: &str| Refusal { offset: 0, instruction: "(dispatch)".into(), reason: reason.into() };
    if g.rsrc2 & 1 != 0 {
        return Err(refuse("scratch is enabled (COMPUTE_PGM_RSRC2.scratch_en): scratch is not modelled"));
    }
    let lanes = if g.wave64 { 64 } else { 32 };
    let [tx, ty, tz] = g.threads;
    let total = (tx * ty * tz) as usize;
    if total == 0 {
        return Err(refuse("a group of no threads"));
    }
    let user = (g.rsrc2 >> 1 & 0x1F) as usize;
    if g.user_sgprs.len() > user {
        return Err(refuse("more user SGPR values than COMPUTE_PGM_RSRC2 enables"));
    }
    let count = total.div_ceil(lanes);
    if count > 64 {
        return Err(refuse("a group of more than 64 waves"));
    }
    let vgprs = usize::from(g.vgprs.max(1));
    // LDS_SIZE in units of 128 dwords (LLVM's AMDGPUUsage, COMPUTE_PGM_RSRC2 for GFX7-GFX10)
    let lds_bytes = (g.rsrc2 >> 15 & 0x1FF) as usize * 512;
    let mut waves = Vec::new();
    for w in 0..count {
        let mut wave = Wave {
            s: [0; 128],
            st: [Tol::UNDEFINED; 128],
            v: vec![[0; 64]; vgprs],
            vt: vec![[Tol::UNDEFINED; 64]; vgprs],
            scc: false,
            scc_t: Tol::UNDEFINED,
            pc: 0,
            done: false,
            at_barrier: false,
            mode: g.mode,
            m0_written: false,
            steps: 0,
        };
        // 3.12.2, compute: the user SGPRs, then the enabled group ids, then the group info.
        let mut next = 0;
        for i in 0..user {
            if let Some(&value) = g.user_sgprs.get(i) {
                wave.s[i] = value;
                wave.st[i] = Tol::EXACT;
            }
            next += 1;
        }
        for (bit, id) in [(7, g.group_id[0]), (8, g.group_id[1]), (9, g.group_id[2])] {
            if g.rsrc2 >> bit & 1 == 1 {
                wave.s[next] = id;
                wave.st[next] = Tol::EXACT;
                next += 1;
            }
        }
        if g.rsrc2 >> 10 & 1 == 1 {
            // {first_wave, 6'h00, wave_id_in_group[4:0], 2'h0, ordered_append_term[11:0],
            // threadgroup_size_in_waves[5:0]}; the ordered-append term is not modelled
            wave.s[next] = u32::from(w == 0) << 31 | (w as u32 & 0x1F) << 20 | (count as u32 & 0x3F);
            wave.st[next] = Tol::EXACT;
        }
        // 3.12.2: VGPR0..2 the thread's id in the group, as many as TIDIG_COMP_CNT asks.
        let ids = (g.rsrc2 >> 11 & 3) as usize + 1;
        let mut exec = 0u64;
        for lane in 0..lanes {
            let t = w * lanes + lane;
            if t >= total {
                continue;
            }
            exec |= 1 << lane;
            let t = t as u32;
            let id = [t % tx, t / tx % ty, t / (tx * ty)];
            for (c, &value) in id.iter().enumerate().take(ids.min(vgprs)) {
                wave.v[c][lane] = value;
                wave.vt[c][lane] = Tol::EXACT;
            }
        }
        wave.s[EXEC] = exec as u32;
        wave.s[EXEC + 1] = (exec >> 32) as u32;
        wave.st[EXEC] = Tol::EXACT;
        wave.st[EXEC + 1] = Tol::EXACT;
        waves.push(wave);
    }
    let mut m = Machine {
        g,
        mem,
        lanes,
        waves,
        lds: vec![0; lds_bytes],
        lds_t: vec![Tol::UNDEFINED; lds_bytes],
        lds_seen: vec![Seen::default(); lds_bytes.div_ceil(4)],
        epoch: 0,
        fetched: HashMap::new(),
        ran: Ran::default(),
    };
    loop {
        let mut progressed = false;
        for w in 0..m.waves.len() {
            while !m.waves[w].done && !m.waves[w].at_barrier {
                m.step(w)?;
                progressed = true;
            }
        }
        if m.waves.iter().all(|w| w.done) {
            return Ok(m.ran);
        }
        // every wave still running waits at a barrier: release them
        if m.waves.iter().filter(|w| !w.done).all(|w| w.at_barrier) {
            for w in &mut m.waves {
                w.at_barrier = false;
            }
            m.epoch += 1;
            continue;
        }
        if !progressed {
            return Err(refuse("no wave can run"));
        }
    }
}

/// The opcodes the machine runs itself, besides those [`sem::salu`] and [`sem::valu`] answer.
const OWN: &[&str] = &[
    "s_endpgm", "s_branch", "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz", "s_cbranch_vccnz",
    "s_cbranch_execz", "s_cbranch_execnz", "s_barrier", "s_nop", "s_waitcnt", "s_inst_prefetch",
    "s_clause", "s_icache_inv", "s_sleep", "s_setprio", "s_round_mode", "s_denorm_mode",
    "s_waitcnt_vscnt", "s_waitcnt_vmcnt", "s_waitcnt_expcnt", "s_waitcnt_lgkmcnt", "s_getpc_b64",
    "s_setpc_b64", "s_swappc_b64", "s_call_b64", "s_setreg_b32", "s_setreg_imm32_b32", "s_getreg_b32",
    "s_dcache_inv", "s_gl1_inv", "s_dcache_wb", "v_nop", "v_pipeflush", "v_readlane_b32",
    "v_readfirstlane_b32", "v_writelane_b32", "v_permlane16_b32", "v_permlanex16_b32", "v_swap_b32",
    "v_movrels_b32", "v_movreld_b32", "v_movrelsd_b32", "v_fma_mix_f32", "buffer_gl0_inv",
    "buffer_gl1_inv",
];

pub(super) fn implements(name: &str) -> bool {
    if OWN.contains(&name) || smem(name).is_some() || ds(name).is_some() || buffer(name).is_some() || global(name).is_some() {
        return true;
    }
    if name.starts_with("s_") {
        return sem::salu(name, SaluIn::default()).is_ok();
    }
    if name.starts_with("v_") {
        let lane = Lane { s: [0; 3], mask: false, d: 0, lane: 0, mode: Mode(0), clamp: false, omod: 0 };
        return sem::valu(name, &lane).is_ok();
    }
    false
}

/// Scalar memory reads (7.2.1): dwords, and whether through a buffer constant.
fn smem(name: &str) -> Option<(usize, bool)> {
    let (rest, buffer) = match name.strip_prefix("s_buffer_load_dword") {
        Some(rest) => (rest, true),
        None => (name.strip_prefix("s_load_dword")?, false),
    };
    let n = match rest {
        "" => 1,
        "x2" => 2,
        "x4" => 4,
        "x8" => 8,
        "x16" => 16,
        _ => return None,
    };
    Some((n, buffer))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DsOp {
    Swizzle,
    Permute { back: bool },
    /// Bytes per access, sign extension, and for the two-address forms the scale of the offsets.
    Read { bytes: usize, signed: bool, pair: Option<usize> },
    Write { bytes: usize, pair: Option<usize> },
    Atomic { rtn: bool },
}

/// The LDS operations the machine runs (12.13).
fn ds(name: &str) -> Option<DsOp> {
    let rest = name.strip_prefix("ds_")?;
    Some(match rest {
        "swizzle_b32" => DsOp::Swizzle,
        "permute_b32" => DsOp::Permute { back: false },
        "bpermute_b32" => DsOp::Permute { back: true },
        "read_b32" => DsOp::Read { bytes: 4, signed: false, pair: None },
        "read_b64" => DsOp::Read { bytes: 8, signed: false, pair: None },
        "read_b96" => DsOp::Read { bytes: 12, signed: false, pair: None },
        "read_b128" => DsOp::Read { bytes: 16, signed: false, pair: None },
        "read_u8" => DsOp::Read { bytes: 1, signed: false, pair: None },
        "read_i8" => DsOp::Read { bytes: 1, signed: true, pair: None },
        "read_u16" => DsOp::Read { bytes: 2, signed: false, pair: None },
        "read_i16" => DsOp::Read { bytes: 2, signed: true, pair: None },
        "read2_b32" => DsOp::Read { bytes: 4, signed: false, pair: Some(4) },
        "read2_b64" => DsOp::Read { bytes: 8, signed: false, pair: Some(8) },
        "read2st64_b32" => DsOp::Read { bytes: 4, signed: false, pair: Some(4 * 64) },
        "read2st64_b64" => DsOp::Read { bytes: 8, signed: false, pair: Some(8 * 64) },
        "write_b8" => DsOp::Write { bytes: 1, pair: None },
        "write_b16" => DsOp::Write { bytes: 2, pair: None },
        "write_b32" => DsOp::Write { bytes: 4, pair: None },
        "write_b64" => DsOp::Write { bytes: 8, pair: None },
        "write_b96" => DsOp::Write { bytes: 12, pair: None },
        "write_b128" => DsOp::Write { bytes: 16, pair: None },
        "write2_b32" => DsOp::Write { bytes: 4, pair: Some(4) },
        "write2_b64" => DsOp::Write { bytes: 8, pair: Some(8) },
        "write2st64_b32" => DsOp::Write { bytes: 4, pair: Some(4 * 64) },
        "write2st64_b64" => DsOp::Write { bytes: 8, pair: Some(8 * 64) },
        _ => {
            let rtn = rest.contains("_rtn_");
            let op = rest.replace("_rtn", "");
            sem::atomic32(&op, 0, 0, 0).ok()?;
            if !(op.ends_with("_u32") || op.ends_with("_i32") || op.ends_with("_b32")) {
                return None;
            }
            DsOp::Atomic { rtn }
        }
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BufOp {
    /// Dwords, or bytes of a sub-dword load (with sign extension).
    Load { dwords: usize },
    LoadSmall { bytes: usize, signed: bool },
    Store { dwords: usize },
    StoreSmall { bytes: usize },
    LoadFormat { components: usize },
    StoreFormat { components: usize },
    Atomic,
    Invalidate,
}

fn components(s: &str) -> Option<usize> {
    Some(match s {
        "x" => 1,
        "xy" => 2,
        "xyz" => 3,
        "xyzw" => 4,
        _ => return None,
    })
}

fn dwords(s: &str) -> Option<usize> {
    Some(match s {
        "dword" => 1,
        "dwordx2" => 2,
        "dwordx3" => 3,
        "dwordx4" => 4,
        _ => return None,
    })
}

/// The buffer operations the machine runs (12.14).
fn buffer(name: &str) -> Option<BufOp> {
    let rest = name.strip_prefix("buffer_")?;
    if let Some(c) = rest.strip_prefix("load_format_") {
        return components(c).map(|components| BufOp::LoadFormat { components });
    }
    if let Some(c) = rest.strip_prefix("store_format_") {
        return components(c).map(|components| BufOp::StoreFormat { components });
    }
    if let Some(op) = rest.strip_prefix("atomic_") {
        return sem::atomic32(op, 0, 0, 0).ok().map(|_| BufOp::Atomic);
    }
    Some(match rest {
        "gl0_inv" | "gl1_inv" => BufOp::Invalidate,
        "load_ubyte" => BufOp::LoadSmall { bytes: 1, signed: false },
        "load_sbyte" => BufOp::LoadSmall { bytes: 1, signed: true },
        "load_ushort" => BufOp::LoadSmall { bytes: 2, signed: false },
        "load_sshort" => BufOp::LoadSmall { bytes: 2, signed: true },
        "store_byte" => BufOp::StoreSmall { bytes: 1 },
        "store_short" => BufOp::StoreSmall { bytes: 2 },
        _ => {
            if let Some(d) = rest.strip_prefix("load_") {
                BufOp::Load { dwords: dwords(d)? }
            } else {
                BufOp::Store { dwords: dwords(rest.strip_prefix("store_")?)? }
            }
        }
    })
}

/// The global-segment operations the machine runs (12.18.3).
fn global(name: &str) -> Option<BufOp> {
    let rest = name.strip_prefix("global_")?;
    if let Some(op) = rest.strip_prefix("atomic_") {
        return sem::atomic32(op, 0, 0, 0).ok().map(|_| BufOp::Atomic);
    }
    Some(match rest {
        "load_ubyte" => BufOp::LoadSmall { bytes: 1, signed: false },
        "load_sbyte" => BufOp::LoadSmall { bytes: 1, signed: true },
        "load_ushort" => BufOp::LoadSmall { bytes: 2, signed: false },
        "load_sshort" => BufOp::LoadSmall { bytes: 2, signed: true },
        "store_byte" => BufOp::StoreSmall { bytes: 1 },
        "store_short" => BufOp::StoreSmall { bytes: 2 },
        _ => {
            if let Some(d) = rest.strip_prefix("load_") {
                BufOp::Load { dwords: dwords(d)? }
            } else {
                BufOp::Store { dwords: dwords(rest.strip_prefix("store_")?)? }
            }
        }
    })
}

/// A source value read for a lane, with its trust.
type Val = (u64, Tol);

/// 3.6.3: a 64-bit SGPR operand starts at an even register.
fn pair(code: u8, n: u8) -> R<()> {
    if n >= 2 && code < 128 && code % 2 == 1 {
        return Err(format!("a 64-bit SGPR operand at s{code}: SGPR pairs start at an even register (3.6.3)"));
    }
    Ok(())
}

impl Machine<'_> {
    fn fetch(&mut self, pc: usize) -> Result<Instruction, Refusal> {
        if let Some(i) = self.fetched.get(&pc) {
            return Ok(i.clone());
        }
        let inst = gfx10::instruction(&self.g.code, pc).map_err(|e| Refusal {
            offset: pc,
            instruction: "(fetch)".into(),
            reason: e.to_string(),
        })?;
        self.fetched.insert(pc, inst.clone());
        Ok(inst)
    }

    fn step(&mut self, w: usize) -> Result<(), Refusal> {
        let pc = self.waves[w].pc;
        let inst = self.fetch(pc)?;
        let name = inst.name().unwrap_or("?");
        let refuse = |reason: String| Refusal { offset: pc, instruction: name.to_owned(), reason };
        self.waves[w].steps += 1;
        if self.waves[w].steps > self.g.step_limit {
            return Err(refuse(format!(
                "a wave ran past {} instructions without ending (a loop the program never leaves?)",
                self.g.step_limit
            )));
        }
        self.ran.instructions += 1;
        let next = pc + 4 * inst.words.len();
        self.waves[w].pc = next;
        if name == "?" {
            return Err(refuse("an opcode the decoder does not name".into()));
        }
        let r = match inst.encoding {
            Encoding::Sop1 | Encoding::Sop2 | Encoding::Sopc | Encoding::Sopk => self.scalar(w, &inst, name),
            Encoding::Sopp => self.control(w, &inst, name, next),
            Encoding::Smem => self.smem(w, &inst, name),
            Encoding::Vop1 | Encoding::Vop2 | Encoding::Vopc | Encoding::Vop3 | Encoding::Vop3p => {
                self.valu(w, &inst, name)
            }
            Encoding::Ds => self.ds(w, &inst, name),
            Encoding::Mubuf | Encoding::Mtbuf => self.buffer(w, &inst, name),
            Encoding::Global => self.global(w, &inst, name),
            Encoding::Mimg => Err("image instructions: the reference gives no surface layout (swizzle \
                                   modes, pitch) to address an image's memory by"
                .into()),
            other => Err(format!("{other:?} instructions are not in the oracle")),
        };
        r.map_err(refuse)
    }

    // -----------------------------------------------------------------------------------------
    // registers
    // -----------------------------------------------------------------------------------------

    /// Which lanes run must be known exactly before a vector instruction (an approximate,
    /// order-dependent or undefined EXEC would decide which writes happen).
    fn exec_known(&self, w: usize) -> R<()> {
        let st = &self.waves[w].st;
        if !st[EXEC].exact() || (self.lanes == 64 && !st[EXEC + 1].exact()) {
            return Err("EXEC holds a value the oracle does not know exactly: which lanes run is not known".into());
        }
        Ok(())
    }

    fn who(&self, w: usize, lane: usize) -> Who {
        Who { wave: w, lane, epoch: self.epoch }
    }

    /// The trust of the lane mask at `code` (VCC or EXEC): both halves on a wave of 64.
    fn mask_t(&self, w: usize, code: usize) -> Tol {
        let st = &self.waves[w].st;
        if self.lanes == 64 { st[code].join(st[code + 1]) } else { st[code] }
    }

    fn lane_mask(&self) -> u64 {
        if self.lanes == 64 { u64::MAX } else { 0xFFFF_FFFF }
    }

    fn exec(&self, w: usize) -> u64 {
        let s = &self.waves[w].s;
        (u64::from(s[EXEC]) | u64::from(s[EXEC + 1]) << 32) & self.lane_mask()
    }

    fn vcc(&self, w: usize) -> u64 {
        let s = &self.waves[w].s;
        (u64::from(s[VCC]) | u64::from(s[VCC + 1]) << 32) & self.lane_mask()
    }

    fn sget(&self, w: usize, code: u8, n: u8) -> Val {
        if code == NULL {
            return (0, Tol::EXACT);
        }
        let wave = &self.waves[w];
        let c = usize::from(code);
        let (mut v, mut t) = (u64::from(wave.s[c]), wave.st[c]);
        if n >= 2 && c + 1 < 128 {
            v |= u64::from(wave.s[c + 1]) << 32;
            t = t.join(wave.st[c + 1]);
        }
        (v, t)
    }

    fn sset(&mut self, w: usize, code: u8, n: u8, v: u64, t: Tol) {
        if code == NULL {
            return;
        }
        let wave = &mut self.waves[w];
        let c = usize::from(code);
        wave.s[c] = v as u32;
        wave.st[c] = t;
        if n >= 2 && c + 1 < 128 {
            wave.s[c + 1] = (v >> 32) as u32;
            wave.st[c + 1] = t;
        }
        if c == M0 {
            wave.m0_written = true;
        }
    }

    /// A scalar source (5.2): an SGPR, an inline constant, a special source or the literal.
    fn sread(&self, w: usize, op: Operand, wide: bool) -> R<Val> {
        Ok(match op {
            Operand::Scalar { code, dwords } => {
                let n = if wide { 2 } else { dwords };
                pair(code, n)?;
                self.sget(w, code, n)
            }
            // VCCZ and EXECZ are of the whole mask: both halves on a wave of 64
            Operand::Const(251) => (u64::from(self.vcc(w) == 0), self.mask_t(w, VCC)),
            Operand::Const(252) => (u64::from(self.exec(w) == 0), self.mask_t(w, EXEC)),
            Operand::Const(253) => (u64::from(self.waves[w].scc), self.waves[w].scc_t),
            Operand::Const(c @ 240..=248) if wide => {
                return Err(format!("inline float constant {c} as a 64-bit scalar source: the reference \
                                    names no precision for it"));
            }
            Operand::Const(c) => {
                let kind = if wide { Kind::I64 } else { Kind::B32 };
                (sem::inline_constant(c, kind).ok_or_else(|| format!("source code {c} is not modelled"))?, Tol::EXACT)
            }
            Operand::Literal(v) if wide => {
                return Err(format!("literal {v:#x} as a 64-bit scalar source: the reference gives its \
                                    expansion for vector sources only (6.2.1)"));
            }
            Operand::Literal(v) => (u64::from(v), Tol::EXACT),
            other => return Err(format!("{other:?} as a scalar source")),
        })
    }

    /// A VGPR of a lane, wide or not; a register beyond the program's reads v0 (6.2.6).
    fn vget(&self, w: usize, reg: u8, lane: usize, wide: bool) -> Val {
        let wave = &self.waves[w];
        let r = |i: usize| if i < wave.v.len() { i } else { 0 };
        let lo = r(usize::from(reg));
        let (mut v, mut t) = (u64::from(wave.v[lo][lane]), wave.vt[lo][lane]);
        if wide {
            let hi = r(usize::from(reg) + 1);
            v |= u64::from(wave.v[hi][lane]) << 32;
            t = t.join(wave.vt[hi][lane]);
        }
        (v, t)
    }

    /// Writes a lane's VGPR(s); a destination beyond the program's is not written (6.2.6).
    fn vset(&mut self, w: usize, reg: u8, lane: usize, n: usize, v: u64, t: Tol) {
        let wave = &mut self.waves[w];
        let reg = usize::from(reg);
        if reg + n > wave.v.len() {
            return;
        }
        for i in 0..n {
            wave.v[reg + i][lane] = (v >> (32 * i)) as u32;
            wave.vt[reg + i][lane] = t;
        }
    }

    /// A vector source of a lane (6.2): VGPR, SGPR, inline constant in the kind's precision, or
    /// the literal expanded for the kind.
    fn vread(&self, w: usize, op: Operand, kind: Kind, lane: usize) -> R<Val> {
        Ok(match op {
            Operand::Vector { reg, .. } => self.vget(w, reg, lane, kind.wide()),
            Operand::Scalar { code, .. } => {
                let n = if kind.wide() { 2 } else { 1 };
                pair(code, n)?;
                self.sget(w, code, n)
            }
            Operand::Const(c @ 251..=253) => self.sread(w, Operand::Const(c), false)?,
            Operand::Const(254) => return Err("LDS direct reads are not modelled".into()),
            Operand::Const(c) => {
                (sem::inline_constant(c, kind).ok_or_else(|| format!("source code {c} is not modelled"))?, Tol::EXACT)
            }
            Operand::Literal(v) => (sem::literal(v, kind), Tol::EXACT),
            Operand::Attr { .. } => return Err("interpolation attributes are not modelled".into()),
        })
    }

    fn vreg(op: Option<Operand>) -> R<(u8, usize)> {
        match op {
            Some(Operand::Vector { reg, dwords }) => Ok((reg, usize::from(dwords))),
            other => Err(format!("a VGPR operand expected, {other:?}")),
        }
    }

    // -----------------------------------------------------------------------------------------
    // scalar ALU and program control
    // -----------------------------------------------------------------------------------------

    fn scalar(&mut self, w: usize, inst: &Instruction, name: &str) -> R<()> {
        let Fields::Sop { dst, src, imm } = inst.fields(self.g.wave64) else { unreachable!("a SOP") };
        let pc = self.waves[w].pc - 4 * inst.words.len();
        let dst_n = match dst {
            Some(Operand::Scalar { dwords, .. }) => dwords,
            _ => 1,
        };
        if let Some(Operand::Scalar { code, dwords }) = dst {
            pair(code, dwords)?;
        }
        match name {
            "s_getpc_b64" | "s_call_b64" => {
                let Some(Operand::Scalar { code, .. }) = dst else { return Err("no destination".into()) };
                // "D.u64 = PC + 4": the address of the next instruction
                self.sset(w, code, 2, CODE_BASE + (pc + 4 * inst.words.len()) as u64, Tol::EXACT);
                if name == "s_call_b64" {
                    let next = pc + 4;
                    self.waves[w].pc = (next as i64 + 4 * i64::from(imm.unwrap_or(0) as i16)) as usize;
                }
                return Ok(());
            }
            "s_setpc_b64" | "s_swappc_b64" => {
                let (target, t) = self.sread(w, src[0], true)?;
                if t.undefined || !t.exact() {
                    return Err("a jump to an address the oracle does not know exactly".into());
                }
                let offset = target.checked_sub(CODE_BASE).filter(|&o| (o as usize) < self.g.code.len());
                let Some(offset) = offset else {
                    return Err(format!("a jump to {target:#x}, outside the program"));
                };
                if let (true, Some(Operand::Scalar { code, .. })) = (name == "s_swappc_b64", dst) {
                    self.sset(w, code, 2, CODE_BASE + (pc + 4) as u64, Tol::EXACT);
                }
                self.waves[w].pc = offset as usize;
                return Ok(());
            }
            "s_setreg_b32" | "s_setreg_imm32_b32" => {
                let (value, t) = self.sread(w, src[0], false)?;
                if !t.exact() {
                    return Err("a hardware register written from a value the oracle does not know exactly".into());
                }
                return self.setreg(w, imm.unwrap_or(0), value as u32);
            }
            "s_getreg_b32" => {
                let imm = imm.unwrap_or(0);
                if imm & 0x3F != 1 {
                    return Err(format!("hardware register {} is not modelled (MODE only)", imm & 0x3F));
                }
                return Err("s_getreg_b32: the reference's pseudocode reads SIMM16[15:11] as the size \
                            where its text says 1..32: two readings"
                    .into());
            }
            n if n.starts_with("s_waitcnt_") || n == "s_version" => return Ok(()),
            _ => {}
        }
        let wide = dst_n == 2
            || src.iter().any(|s| matches!(s, Operand::Scalar { dwords: 2, .. }))
            || name.ends_with("_b64") && !name.contains("_i32_");
        let mut t = Tol::EXACT;
        let mut read = |i: usize, wide: bool| -> R<u64> {
            match src.get(i) {
                Some(&op) => {
                    let (v, vt) = self.sread(w, op, wide)?;
                    t = t.join(vt);
                    Ok(v)
                }
                None => Ok(0),
            }
        };
        let src_wide = |i: usize| match src.get(i) {
            Some(Operand::Scalar { dwords, .. }) => *dwords == 2,
            Some(Operand::Const(_) | Operand::Literal(_)) => {
                wide && !matches!(name, "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" | "s_bfm_b64" | "s_bitset0_b64" | "s_bitset1_b64")
                    || (i == 0 && matches!(name, "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64"))
            }
            _ => false,
        };
        let (w0, w1) = (src_wide(0), src_wide(1));
        let s0 = read(0, w0)?;
        let s1 = read(1, w1)?;
        let d = match dst {
            Some(Operand::Scalar { code, dwords }) => {
                let (v, dt) = self.sget(w, code, dwords);
                if matches!(name, "s_cmov_b32" | "s_cmov_b64" | "s_cmovk_i32" | "s_addk_i32" | "s_mulk_i32")
                    || name.starts_with("s_bitset")
                {
                    t = t.join(dt);
                }
                v
            }
            _ => 0,
        };
        let uses_scc = name.contains("cselect") || name.contains("addc") || name.contains("subb") || name.contains("cmov");
        if uses_scc {
            t = t.join(self.waves[w].scc_t);
        }
        if t.approximate() && !matches!(name, "s_mov_b32" | "s_mov_b64") {
            return Err(format!("{name} of an approximate value: its result has no tolerance"));
        }
        let exec = self.exec(w);
        let r = sem::salu(name, SaluIn { s0, s1, scc: self.waves[w].scc, exec, imm: imm.unwrap_or(0), d })?;
        if name.contains("saveexec") || name.contains("wrexec") {
            // the old EXEC: EXEC_LO for the _B32 forms, the whole mask for the _B64 forms
            t = t.join(if name.ends_with("_b32") { self.waves[w].st[EXEC] } else { self.mask_t(w, EXEC) });
        }
        if let (Some(v), Some(Operand::Scalar { code, dwords })) = (r.d, dst) {
            self.sset(w, code, dwords, v, t);
        }
        if let Some(c) = r.scc {
            self.waves[w].scc = c;
            self.waves[w].scc_t = t;
        }
        if let Some(e) = r.exec {
            // 12.3: the _B32 forms write EXEC_LO, on a wave of 64 too
            let n = if self.lanes == 64 && !name.ends_with("_b32") { 2 } else { 1 };
            self.sset(w, EXEC as u8, n, e, t);
        }
        Ok(())
    }

    /// `S_SETREG*` of MODE (5.8, Table 16 code 1). The reference gives SIMM16 as
    /// `{size[4:0], offset[4:0], hwRegId[5:0]}` with "size is 1..32", and S_GETREG's pseudocode
    /// reads the field itself as the size: both readings are taken, and a write on which they
    /// disagree (the bit only one of them writes would change) is refused.
    fn setreg(&mut self, w: usize, imm: u16, value: u32) -> R<()> {
        if imm & 0x3F != 1 {
            return Err(format!("hardware register {} is not modelled (MODE only)", imm & 0x3F));
        }
        let offset = u32::from(imm >> 6 & 0x1F);
        let field = u32::from(imm >> 11 & 0x1F);
        let write = |old: u32, size: u32| -> u32 {
            if size == 0 {
                return old;
            }
            let mask = if size >= 32 { u32::MAX } else { (1 << size) - 1 };
            let mask = mask.checked_shl(offset).unwrap_or(0);
            (old & !mask) | (value.checked_shl(offset).unwrap_or(0) & mask)
        };
        let old = self.waves[w].mode.0;
        let (a, b) = (write(old, field + 1), write(old, field));
        if a != b {
            return Err(format!("s_setreg of MODE: SIMM16[15:11] = {field} as size-1 and as the size give \
                                {a:#x} and {b:#x}"));
        }
        self.waves[w].mode = Mode(a);
        Ok(())
    }

    fn control(&mut self, w: usize, inst: &Instruction, name: &str, next: usize) -> R<()> {
        let Fields::Sopp { imm } = inst.fields(self.g.wave64) else { unreachable!("a SOPP") };
        // "PC = PC + signext(SIMM16 * 4) + 4"
        let target = (next as i64 + 4 * i64::from(imm as i16)) as usize;
        let cond = |t: Tol, taken: bool| -> R<bool> {
            if t.undefined {
                return Err("a branch on a value nothing wrote".into());
            }
            if !t.exact() {
                return Err("a branch on a value the oracle holds approximate or order-dependent: the console and a host may part ways there"
                    .into());
            }
            Ok(taken)
        };
        let taken = match name {
            "s_endpgm" => {
                self.waves[w].done = true;
                return Ok(());
            }
            "s_barrier" => {
                self.waves[w].at_barrier = true;
                return Ok(());
            }
            "s_branch" => true,
            "s_cbranch_scc0" | "s_cbranch_scc1" => {
                cond(self.waves[w].scc_t, self.waves[w].scc == (name == "s_cbranch_scc1"))?
            }
            "s_cbranch_vccz" | "s_cbranch_vccnz" => {
                let t = self.waves[w].st[VCC];
                let t = if self.lanes == 64 { t.join(self.waves[w].st[VCC + 1]) } else { t };
                cond(t, (self.vcc(w) == 0) == (name == "s_cbranch_vccz"))?
            }
            "s_cbranch_execz" | "s_cbranch_execnz" => {
                let t = self.waves[w].st[EXEC];
                let t = if self.lanes == 64 { t.join(self.waves[w].st[EXEC + 1]) } else { t };
                cond(t, (self.exec(w) == 0) == (name == "s_cbranch_execz"))?
            }
            "s_round_mode" => {
                let m = self.waves[w].mode.0;
                self.waves[w].mode = Mode((m & !0xF) | u32::from(imm & 0xF));
                false
            }
            "s_denorm_mode" => {
                let m = self.waves[w].mode.0;
                self.waves[w].mode = Mode((m & !0xF0) | u32::from(imm & 0xF) << 4);
                false
            }
            "s_nop" | "s_waitcnt" | "s_inst_prefetch" | "s_clause" | "s_icache_inv" | "s_sleep" | "s_setprio" => false,
            _ => return Err(format!("{name}: not in the oracle's program control")),
        };
        if taken {
            self.waves[w].pc = target;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------------------------
    // scalar memory (7.2.1, 7.5)
    // -----------------------------------------------------------------------------------------

    fn smem(&mut self, w: usize, inst: &Instruction, name: &str) -> R<()> {
        let Fields::Smem { data, base, offset, soffset, .. } = inst.fields(self.g.wave64) else {
            unreachable!("an SMEM")
        };
        if matches!(name, "s_dcache_inv" | "s_gl1_inv" | "s_dcache_wb") {
            return Ok(());
        }
        let Some((n, through_buffer)) = smem(name) else {
            return Err(format!("{name}: not in the oracle's scalar memory"));
        };
        let Operand::Scalar { code: base, .. } = base else { return Err("no base".into()) };
        let Some(Operand::Scalar { code: sdst, .. }) = data else { return Err("no destination".into()) };
        pair(sdst, n as u8)?;
        if n >= 4 && !sdst.is_multiple_of(4) {
            return Err(format!("a {n}-dword scalar load into s{sdst}: 7.5 aligns it to a multiple of 4"));
        }
        // "If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset"
        let (so, so_t) = match soffset {
            Operand::Scalar { code, .. } => self.sget(w, code, 1),
            _ => (0, Tol::EXACT),
        };
        let mut t = so_t;
        let (address, limit) = if through_buffer {
            let mut words = [0u32; 4];
            for (i, word) in words.iter_mut().enumerate() {
                let (v, vt) = self.sget(w, base + i as u8, 1);
                *word = v as u32;
                t = t.join(vt);
            }
            let m_base = u64::from(words[0]) | u64::from(words[1] & 0xFFFF) << 32;
            if words[1] >> 16 != 0 {
                return Err("s_buffer_load through a constant with a stride: the reference's bounds \
                            check (7.2.1, m_size) does not say in what units"
                    .into());
            }
            let off = (i64::from(offset) + so as i64) as u64;
            (m_base.wrapping_add(off), Some((off, u64::from(words[2]))))
        } else {
            let (b, bt) = self.sget(w, base, 2);
            t = t.join(bt);
            (b.wrapping_add(i64::from(offset) as u64).wrapping_add(so), None)
        };
        if !t.exact() {
            return Err("a scalar load from an address the oracle does not know exactly".into());
        }
        let address = address & !3;
        for i in 0..n {
            if let Some((off, records)) = limit {
                // 7.5: "the operation is not performed for any Dwords that are out-of-range"
                if (off & !3) + 4 * i as u64 + 4 > records {
                    continue;
                }
            }
            let a = address + 4 * i as u64;
            let (v, vt) = self.mem.read_dword(a, self.who(w, 0), true)?;
            self.sset(w, sdst + i as u8, 1, u64::from(v), vt);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------------------------
    // vector ALU (6.2-6.10)
    // -----------------------------------------------------------------------------------------

    fn valu(&mut self, w: usize, inst: &Instruction, name: &str) -> R<()> {
        let Fields::Valu(v) = inst.fields(self.g.wave64) else { unreachable!("a VALU") };
        if matches!(name, "v_nop" | "v_pipeflush") {
            return Ok(());
        }
        self.exec_known(w)?;
        if let Some(r) = self.cross_lane(w, inst, name, &v)? {
            return Ok(r);
        }
        if v.dpp.is_some() && (name.starts_with("v_cmp") || v.carry.is_some()) {
            return Err(format!("{name} through DPP: 13.3.9 gives the row and bank masks for VGPR writes, not for a lane-mask result"));
        }
        if inst.encoding == Encoding::Vop3p && name != "v_fma_mix_f32" {
            return Err(format!("{name}: packed math is not in the oracle yet"));
        }
        if inst.encoding == Encoding::Vop3 && v.opsel != 0 {
            return Err(format!("{name}: OPSEL (6.2.5) is not in the oracle yet"));
        }
        let mask_src = matches!(name, "v_cndmask_b32") || name.ends_with("_co_ci_u32");
        let n = if mask_src { 2 } else { v.src.len() };
        let kinds: Vec<Kind> = (0..n).map(|i| sem::source_kind(name, i)).collect();
        let float = kinds.iter().any(|k| k.float());
        if v.sdwa.is_some() && float {
            return Err(format!("{name}: SDWA on a float opcode is not in the oracle"));
        }
        if !float && (v.abs != 0 || v.neg != 0) && inst.encoding != Encoding::Vop3p {
            return Err(format!("{name}: input modifiers on an integer opcode (\"not supported\")"));
        }
        let exec = self.exec(w);
        let lanes = self.lanes;
        let mode = self.waves[w].mode;
        let dst = v.dst;
        let (mut cmp_bits, mut carry_bits) = (0u64, 0u64);
        let mut mask_t = Tol::EXACT;
        let mask = match (mask_src, v.src.get(2)) {
            (true, Some(&op)) => {
                let (m, t) = self.sread(w, op, self.lanes == 64)?;
                mask_t = t;
                m
            }
            _ => 0,
        };
        if mask_src && mask_t.approximate() {
            return Err(format!("{name}: a lane mask the oracle holds approximate"));
        }
        // 3.6.1: a destination past the program's VGPRs makes the instruction write nothing, its
        // SGPR results (a VOP3B carry-out) too
        if let Some(Operand::Vector { reg, dwords }) = dst
            && usize::from(reg) + usize::from(dwords) > self.waves[w].v.len()
        {
            return Ok(());
        }
        let mut writes: Vec<(usize, u64, Tol)> = Vec::new();
        let own = sem::own_tolerance(name);
        for lane in 0..lanes {
            if exec >> lane & 1 == 0 {
                continue;
            }
            // DPP: source 0 from another lane, or not at all
            let mut from = [lane; 3];
            let mut zero0 = false;
            let mut writes_lane = true;
            if let Some(dpp) = v.dpp {
                let (src, fi, bound) = match dpp {
                    Dpp::Row { ctrl, row_mask, bank_mask, bound_ctrl, fi } => {
                        if row_mask >> (lane / 16) & 1 == 0 || bank_mask >> (lane % 16 / 4) & 1 == 0 {
                            writes_lane = false;
                        }
                        (sem::dpp16_source(ctrl, lane)?, fi, bound_ctrl)
                    }
                    Dpp::Lanes { sel, fi } => (Some(sem::dpp8_source(sel, lane)), fi, false),
                };
                match src {
                    // 13.3.9 BC: "0 = do not write when source is out of range, 1 = write"
                    None if bound => zero0 = true,
                    None => writes_lane = false,
                    // FI: "0 = read zero for any inactive lanes"
                    Some(s) if exec >> s & 1 == 0 && !fi => zero0 = true,
                    Some(s) => from[0] = s,
                }
            }
            if !writes_lane {
                continue;
            }
            let mut t = Tol::EXACT;
            let mut s = [0u64; 3];
            let mut src_t = [Tol::EXACT; 3];
            for i in 0..n {
                let (mut val, vt) = if i == 0 && zero0 {
                    (0, Tol::EXACT)
                } else {
                    self.vread(w, v.src[i], kinds[i], from[i])?
                };
                t = t.join(vt);
                src_t[i] = vt;
                if let Some(sdwa) = v.sdwa
                    && i < 2
                {
                    val = sdwa_select(val as u32, sdwa.src_sel[i], sdwa.sext >> i & 1 == 1)?;
                }
                if kinds[i].float() && inst.encoding != Encoding::Vop3p {
                    val = sem::modify(val, kinds[i], v.abs >> i & 1 == 1, v.neg >> i & 1 == 1);
                }
                s[i] = val;
            }
            let (d_old, d_t) = match dst {
                Some(Operand::Vector { reg, dwords }) => self.vget(w, reg, lane, dwords == 2),
                _ => (0, Tol::EXACT),
            };
            if matches!(name, "v_fmac_f32" | "v_cvt_f16_f32") {
                t = t.join(d_t);
            }
            if mask_src {
                t = t.join(mask_t);
            }
            let l = Lane { s, mask: mask >> lane & 1 == 1, d: d_old, lane: lane as u32, mode, clamp: v.clamp, omod: v.omod };
            let r = if name == "v_fma_mix_f32" {
                sem::LaneOut { d: Some(u64::from(sem::fma_mix(&l, v.opsel, v.opsel_hi, v.neg, v.neg_hi)?)), carry: None, cmp: None }
            } else {
                sem::valu(name, &l)?
            };
            // the trust of the result: an approximate input only where a bound still holds
            if t.approximate() {
                match sem::carries(name) {
                    Some(sem::Carry::Copy) => {}
                    Some(sem::Carry::Select) if !t.nan => {}
                    Some(c @ (sem::Carry::Product | sem::Carry::Root)) if !t.nan && t.abs == 0.0 => {
                        // each approximate input's interval inside the normal range: across its
                        // edge a host may flush, and a reciprocal turn infinite
                        if (0..n).any(|i| src_t[i].approximate() && !normal_within(s[i] as u32, src_t[i].ulps)) {
                            return Err(format!("{name} of an approximate value at the edge of the normal range"));
                        }
                        t.ulps = if c == sem::Carry::Product { 2 * t.ulps + 1 } else { t.ulps + 1 };
                    }
                    _ => return Err(format!("{name} of an approximate value: no tolerance bounds its result")),
                }
            }
            t.ulps += own.0;
            t.abs += own.1;
            if let Some(d) = r.d {
                if sem::float_result(name) && dst_dwords(dst) == 1 && f32::from_bits(d as u32).is_nan() {
                    t.nan = true;
                }
                // a NaN a double or a half operation makes has bits the reference does not pin,
                // and the comparison has no class for it there
                let half_nan = |h: u64| h & 0x7C00 == 0x7C00 && h & 0x03FF != 0;
                if (dst_dwords(dst) == 2 && name.contains("_f64") && f64::from_bits(d).is_nan())
                    || (sem::half_result(name) && (half_nan(d & 0xFFFF) || half_nan(d >> 16 & 0xFFFF)))
                {
                    return Err(format!("{name}: a NaN of a double or a half, whose bits the reference does not pin"));
                }
                if t.ulps > 0 && t.abs == 0.0 && !t.nan && !normal_within(d as u32, t.ulps) {
                    return Err(format!("{name}: an approximate result at the edge of the normal range"));
                }
            }
            if t.approximate() && (dst_dwords(dst) == 2 || v.sdwa.is_some_and(|s| s.dst_sel != 6)) {
                return Err(format!("{name}: an approximate result of other than one whole dword, which the comparison takes per dword as a single"));
            }
            // SDWA's PRESERVE keeps the old bits, and so their trust
            if v.sdwa.is_some_and(|s| s.dst_sel != 6 && s.dst_unused == 2) {
                t = t.join(d_t);
            }
            if let Some(c) = r.cmp {
                cmp_bits |= u64::from(c) << lane;
                mask_t = mask_t.join(t);
            }
            if let Some(c) = r.carry {
                carry_bits |= u64::from(c) << lane;
                mask_t = mask_t.join(t);
            }
            if let Some(d) = r.d {
                let d = match v.sdwa {
                    Some(sdwa) if sdwa.dst_sel != 6 => sdwa_place(d as u32, d_old as u32, sdwa.dst_sel, sdwa.dst_unused)?.into(),
                    _ => d,
                };
                writes.push((lane, d, t));
            }
        }
        if let Some(Operand::Vector { reg, dwords }) = dst {
            for (lane, d, t) in writes {
                self.vset(w, reg, lane, usize::from(dwords), d, t);
            }
        }
        let n = if self.lanes == 64 { 2 } else { 1 };
        for op in [dst, v.carry] {
            if let Some(Operand::Scalar { code, .. }) = op {
                pair(code, n)?;
            }
        }
        if name.starts_with("v_cmp") {
            // 3.9: "VCC[n] = EXEC[n] & (test passed for thread[n])", fully written
            if let Some(Operand::Scalar { code, .. }) = dst {
                self.sset(w, code, n, cmp_bits, mask_t);
            }
        }
        if let Some(Operand::Scalar { code, .. }) = v.carry {
            self.sset(w, code, n, carry_bits, mask_t);
        }
        Ok(())
    }

    /// The lane select of `V_READLANE_B32` and `V_WRITELANE_B32`: "SGPR or M0" (12.12), and in
    /// AMD's machine-readable RDNA 2 ISA (`OPR_SSRC_LANESEL`, an independent source) also the
    /// inline integers 0..63. No VGPR, literal, negative or float constant.
    fn lane_select(&self, w: usize, op: Operand) -> R<Val> {
        match op {
            Operand::Scalar { .. } | Operand::Const(128..=191) => self.sread(w, op, false),
            other => Err(format!("a lane select of {other:?}: the entry allows an SGPR, M0 or an inline 0..63")),
        }
    }

    /// The VALU instructions that read other lanes than their own or registers by number, and
    /// write a lane or a scalar regardless of EXEC: `v_readlane_b32`, `v_readfirstlane_b32`,
    /// `v_writelane_b32` (12.12, 12.8), `v_permlane16_b32`, `v_permlanex16_b32` (12.12),
    /// `v_swap_b32` and `v_movrel*` (12.8, 6.6). `None` for any other opcode.
    fn cross_lane(&mut self, w: usize, inst: &Instruction, name: &str, v: &Valu) -> R<Option<()>> {
        let exec = self.exec(w);
        let lanes = self.lanes;
        if matches!(name, "v_readlane_b32" | "v_readfirstlane_b32" | "v_writelane_b32")
            && (v.abs != 0 || v.neg != 0 || v.omod != 0 || v.clamp || v.opsel != 0 || v.dpp.is_some() || v.sdwa.is_some())
        {
            return Err(format!("{name} with an input or output modifier, DPP or SDWA: its entry (12.12) gives none"));
        }
        match name {
            "v_readlane_b32" | "v_readfirstlane_b32" => {
                let Some(Operand::Vector { reg, .. }) = v.src.first().copied() else {
                    return Err("an LDS-direct source is not modelled".into());
                };
                let lane = if name == "v_readlane_b32" {
                    let (sel, t) = self.lane_select(w, v.src[1])?;
                    if !t.exact() {
                        return Err("a lane number the oracle does not know exactly".into());
                    }
                    // "Lane is S1 % (32 if wave32, 64 if wave64)"
                    sel as usize % lanes
                } else if exec == 0 {
                    0 // "Lane# = 0 if exec is zero"
                } else {
                    exec.trailing_zeros() as usize
                };
                let (val, t) = self.vget(w, reg, lane, false);
                let Some(Operand::Scalar { code, .. }) = v.dst else { return Err("no scalar destination".into()) };
                self.sset(w, code, 1, val, t);
                Ok(Some(()))
            }
            "v_writelane_b32" => {
                let (val, t) = self.sread(w, v.src[0], false)?;
                let (sel, st) = self.lane_select(w, v.src[1])?;
                if !st.exact() {
                    return Err("a lane number the oracle does not know exactly".into());
                }
                let (reg, _) = Self::vreg(v.dst)?;
                self.vset(w, reg, sel as usize % lanes, 1, val, t);
                Ok(Some(()))
            }
            "v_permlane16_b32" | "v_permlanex16_b32" => {
                if v.abs != 0 || v.neg != 0 || v.omod != 0 {
                    return Err("ABS, NEG and OMOD \"should all be zeroed for this instruction\"".into());
                }
                let Some(Operand::Vector { reg: src, .. }) = v.src.first().copied() else {
                    return Err("source 0 must be a VGPR".into());
                };
                let (s1, t1) = self.sread(w, v.src[1], false)?;
                let (s2, t2) = self.sread(w, v.src[2], false)?;
                if !t1.join(t2).exact() {
                    return Err("lane selects the oracle does not know exactly".into());
                }
                // OP_SEL[0] is FI, OP_SEL[1] BOUND_CTRL (no source is out of range here)
                let fi = v.opsel & 1 == 1;
                let (reg, _) = Self::vreg(v.dst)?;
                let mut out = Vec::new();
                for lane in (0..lanes).filter(|&l| exec >> l & 1 == 1) {
                    let from = sem::permlane_source(s1 as u32, s2 as u32, lane, name == "v_permlanex16_b32");
                    let val = if exec >> from & 1 == 0 && !fi { (0, Tol::EXACT) } else { self.vget(w, src, from, false) };
                    out.push((lane, val));
                }
                for (lane, (val, t)) in out {
                    self.vset(w, reg, lane, 1, val, t);
                }
                Ok(Some(()))
            }
            "v_swap_b32" => {
                let (a, _) = Self::vreg(v.dst)?;
                let Some(Operand::Vector { reg: b, .. }) = v.src.first().copied() else {
                    return Err("source 0 must be a VGPR".into());
                };
                for lane in (0..lanes).filter(|&l| exec >> l & 1 == 1) {
                    let (x, xt) = self.vget(w, a, lane, false);
                    let (y, yt) = self.vget(w, b, lane, false);
                    self.vset(w, a, lane, 1, y, yt);
                    self.vset(w, b, lane, 1, x, xt);
                }
                Ok(Some(()))
            }
            "v_movrels_b32" | "v_movreld_b32" | "v_movrelsd_b32" => {
                let (m0, t) = self.sget(w, M0 as u8, 1);
                if !t.exact() {
                    return Err("an index in M0 the oracle does not know exactly".into());
                }
                let (dst, _) = Self::vreg(v.dst)?;
                let Some(Operand::Vector { reg: src, .. }) = v.src.first().copied() else {
                    return Err("source 0 must be a VGPR".into());
                };
                let shift = |r: u8, by: bool| -> usize { usize::from(r) + if by { m0 as usize } else { 0 } };
                let (d, s) = match name {
                    "v_movrels_b32" => (shift(dst, false), shift(src, true)),
                    "v_movreld_b32" => (shift(dst, true), shift(src, false)),
                    _ => (shift(dst, true), shift(src, true)),
                };
                let s = if s < self.waves[w].v.len() { s } else { 0 };
                let _ = inst;
                for lane in (0..lanes).filter(|&l| exec >> l & 1 == 1) {
                    if d < self.waves[w].v.len() {
                        let (val, vt) = (self.waves[w].v[s][lane], self.waves[w].vt[s][lane]);
                        self.waves[w].v[d][lane] = val;
                        self.waves[w].vt[d][lane] = vt;
                    }
                }
                Ok(Some(()))
            }
            _ => Ok(None),
        }
    }

    // -----------------------------------------------------------------------------------------
    // LDS (10.4, 12.13; bounds 3.6.1)
    // -----------------------------------------------------------------------------------------

    /// Whether `len` bytes at `address` are inside the group's LDS. 3.6.1 bounds an access by
    /// `MIN(lds_size, m0)`; this generation's compilers do not set M0 for LDS, so where the
    /// program has written M0 below the allocation and an access lies between the two, the two
    /// readings disagree and the access is refused.
    fn lds_in(&self, w: usize, address: u64, len: usize) -> R<bool> {
        let end = address + len as u64;
        let size = self.lds.len() as u64;
        let wave = &self.waves[w];
        if wave.m0_written {
            let m0 = u64::from(wave.s[M0]);
            if m0 < size && end > m0 && address < size {
                return Err("an LDS access past M0 but inside the allocation: 3.6.1 bounds LDS by \
                            MIN(lds_size, m0), this generation's compilers do not set M0 for LDS"
                    .into());
            }
        }
        if !(address as usize).is_multiple_of(len.min(4)) {
            return Err(format!("an LDS access at {address:#x} not aligned to {} bytes", len.min(4)));
        }
        Ok(end <= size)
    }

    /// `len` bytes of LDS at `address`, read by lane `lane` of wave `w` (`note`: a read whose
    /// order with other waves' writes matters; an atomic that gives nothing back reads
    /// without one). Each byte keeps its own trust.
    fn lds_read(&mut self, w: usize, lane: usize, address: u64, len: usize, note: bool) -> R<(Vec<u8>, Tol)> {
        let a = address as usize;
        if note {
            let who = self.who(w, lane);
            for i in a / 4..(a + len).div_ceil(4) {
                let defined = (4 * i..4 * i + 4).any(|b| (a..a + len).contains(&b) && !self.lds_t[b].undefined);
                self.lds_seen[i].read(who, defined)?;
            }
        }
        let t = self.lds_t[a..a + len].iter().fold(Tol::EXACT, |t, &b| t.join(b));
        Ok((self.lds[a..a + len].to_vec(), t))
    }

    /// Writes `bytes` to LDS at `address` for lane `lane` of wave `w`. Where another wave wrote
    /// the dword between the same two barriers (in an order that matters), a byte that changes
    /// is order-dependent.
    fn lds_write(&mut self, w: usize, lane: usize, address: u64, bytes: &[u8], t: Tol, class: Option<u8>) -> R<()> {
        if !bytes.len().is_multiple_of(4) && t.approximate() {
            return Err("a part of a dword written from an approximate value".into());
        }
        let a = address as usize;
        let who = self.who(w, lane);
        for i in a / 4..(a + bytes.len()).div_ceil(4) {
            let clash = self.lds_seen[i].write(who, class)?;
            for b in (4 * i..4 * i + 4).filter(|b| (a..a + bytes.len()).contains(b)) {
                let new = bytes[b - a];
                let mut bt = t;
                if clash {
                    if self.lds[b] == new {
                        bt = bt.join(self.lds_t[b]);
                    } else {
                        bt.unordered = true;
                    }
                }
                self.lds[b] = new;
                self.lds_t[b] = bt;
            }
        }
        Ok(())
    }

    /// 3.6.1: a destination range past the program's VGPRs is checked before any access; the
    /// oracle refuses it rather than write a part of it (or change memory for an atomic).
    fn dst_fits(&self, w: usize, reg: u8, dwords: usize) -> R<()> {
        if usize::from(reg) + dwords > self.waves[w].v.len() {
            return Err(format!(
                "a destination v{reg}..v{} past the program's {} VGPRs (3.6.1)",
                usize::from(reg) + dwords - 1,
                self.waves[w].v.len()
            ));
        }
        Ok(())
    }

    /// 3.6.1: a memory instruction's VGPR source range past the program's VGPRs has no defined
    /// value (the VALU's v0 of 6.2.6 is not given for memory instructions): refused.
    fn src_fits(&self, w: usize, reg: u8, dwords: usize) -> R<()> {
        if dwords > 0 && usize::from(reg) + dwords > self.waves[w].v.len() {
            return Err(format!("a source v{reg}..v{} past the program's {} VGPRs (3.6.1)", usize::from(reg) + dwords - 1, self.waves[w].v.len()));
        }
        Ok(())
    }

    fn ds(&mut self, w: usize, inst: &Instruction, name: &str) -> R<()> {
        let Fields::Ds { dst, addr, data, offset0, offset1, gds } = inst.fields(self.g.wave64) else {
            unreachable!("a DS")
        };
        if gds {
            return Err("GDS is not modelled".into());
        }
        let Some(op) = ds(name) else { return Err(format!("{name}: not in the oracle's LDS operations")) };
        self.exec_known(w)?;
        let exec = self.exec(w);
        let lanes = self.lanes;
        let active: Vec<usize> = (0..lanes).filter(|&l| exec >> l & 1 == 1).collect();
        let offset = u64::from(offset1) << 8 | u64::from(offset0);
        let areg = match addr {
            Some(Operand::Vector { reg, .. }) => reg,
            _ => return Err("no address VGPR".into()),
        };
        let addr_of = |m: &Self, lane: usize| -> R<u64> {
            let (a, t) = m.vget(w, areg, lane, false);
            if !t.exact() {
                return Err("an LDS address the oracle does not know exactly".into());
            }
            Ok(a)
        };
        // the whole destination, before any access
        let dst_dwords = match op {
            DsOp::Swizzle | DsOp::Permute { .. } | DsOp::Atomic { rtn: true } => 1,
            DsOp::Read { bytes, pair, .. } => bytes.div_ceil(4) * if pair.is_some() { 2 } else { 1 },
            _ => 0,
        };
        if dst_dwords > 0 {
            let (reg, _) = Self::vreg(dst)?;
            self.dst_fits(w, reg, dst_dwords)?;
        }
        // and every source, which past the VGPRs has no defined value (3.6.1)
        self.src_fits(w, areg, 1)?;
        let per_data = match op {
            DsOp::Write { bytes, .. } => bytes.div_ceil(4),
            DsOp::Atomic { .. } | DsOp::Permute { .. } => 1,
            _ => 0,
        };
        for d in &data {
            if let Operand::Vector { reg, .. } = *d {
                self.src_fits(w, reg, per_data)?;
            }
        }
        match op {
            DsOp::Swizzle => {
                // 12.13.1: reads the ADDR VGPR's data of the lane the offset picks
                let (reg, _) = Self::vreg(dst)?;
                let mut out = Vec::new();
                for &lane in &active {
                    let j = sem::swizzle_source(offset as u16, lane)?;
                    // "reading from an invalid thread results in 0x0"
                    let val = if j < lanes && exec >> j & 1 == 1 { self.vget(w, areg, j, false) } else { (0, Tol::EXACT) };
                    out.push((lane, val));
                }
                for (lane, (v, t)) in out {
                    self.vset(w, reg, lane, 1, v, t);
                }
            }
            DsOp::Permute { back } => {
                if offset1 != 0 {
                    return Err("a permute with OFFSET1 set: 10.4.4 adds offset0 alone, 12.13 \"OFFSET\"".into());
                }
                let (reg, _) = Self::vreg(dst)?;
                let Some(Operand::Vector { reg: src, .. }) = data.first().copied() else {
                    return Err("no data VGPR".into());
                };
                let mut tmp = vec![(0u64, Tol::EXACT); lanes];
                let mut hit = vec![0u32; lanes];
                for &lane in &active {
                    let index = addr_of(self, lane)?.wrapping_add(offset) as u32;
                    let other = sem::permute_lane(index, lane, lanes, self.g.permute)?;
                    if back {
                        // "next if !EXEC[src_lane]": a disabled lane gives zero
                        if exec >> other & 1 == 1 {
                            tmp[lane] = self.vget(w, src, other, false);
                        }
                    } else {
                        tmp[other] = self.vget(w, src, lane, false);
                        hit[other] += 1;
                    }
                }
                for &lane in &active {
                    let (v, mut t) = tmp[lane];
                    if hit[lane] > 1 {
                        // "If multiple sources map to the same destination lane, the final value
                        // is not predictable" (the pseudocode's comment says the highest wins)
                        t.unordered = true;
                    }
                    self.vset(w, reg, lane, 1, v, t);
                }
            }
            DsOp::Read { bytes, signed, pair } => {
                let (reg, _) = Self::vreg(dst)?;
                let places: Vec<u64> = match pair {
                    Some(scale) => vec![u64::from(offset0) * scale as u64, u64::from(offset1) * scale as u64],
                    None => vec![offset],
                };
                for &lane in &active {
                    let a = addr_of(self, lane)?;
                    let mut words: Vec<(u32, Tol)> = Vec::new();
                    for &p in &places {
                        let at = a.wrapping_add(p) & 0xFFFF_FFFF;
                        let (raw, t) = if self.lds_in(w, at, bytes)? {
                            self.lds_read(w, lane, at, bytes, true)?
                        } else {
                            (vec![0; bytes], Tol::EXACT) // "Reads return the value zero"
                        };
                        if bytes < 4 {
                            if t.approximate() {
                                return Err("a part of a dword the oracle holds approximate: an integer with no ULPs".into());
                            }
                            let v = raw.iter().rev().fold(0u32, |x, &b| x << 8 | u32::from(b));
                            let bits = 8 * bytes as u32;
                            let v = if signed { ((v << (32 - bits)) as i32 >> (32 - bits)) as u32 } else { v };
                            words.push((v, t));
                        } else {
                            words.extend(raw.chunks(4).map(|c| (u32::from_le_bytes(c.try_into().expect("4")), t)));
                        }
                    }
                    for (i, (v, t)) in words.into_iter().enumerate() {
                        self.vset(w, reg + i as u8, lane, 1, u64::from(v), t);
                    }
                }
            }
            DsOp::Write { bytes, pair } => {
                let places: Vec<(u64, usize)> = match pair {
                    // 10.4.3: two offsets of one place make one access, of DATA0
                    Some(scale) if offset0 == offset1 => vec![(u64::from(offset0) * scale as u64, 0)],
                    Some(scale) => vec![(u64::from(offset0) * scale as u64, 0), (u64::from(offset1) * scale as u64, 1)],
                    None => vec![(offset, 0)],
                };
                // per byte, the lane of this instruction that last wrote it, and what
                let mut written: HashMap<u64, (usize, u8)> = HashMap::new();
                let mut raced: Vec<u64> = Vec::new();
                for &lane in &active {
                    let a = addr_of(self, lane)?;
                    for &(p, which) in &places {
                        let Some(Operand::Vector { reg, .. }) = data.get(which).copied() else {
                            return Err("no data VGPR".into());
                        };
                        let mut raw = Vec::new();
                        let mut t = Tol::EXACT;
                        for i in 0..bytes.div_ceil(4) {
                            let (v, vt) = self.vget(w, reg + i as u8, lane, false);
                            raw.extend((v as u32).to_le_bytes());
                            t = t.join(vt);
                        }
                        raw.truncate(bytes);
                        let at = a.wrapping_add(p) & 0xFFFF_FFFF;
                        if !self.lds_in(w, at, bytes)? {
                            continue; // "Writes out-of-range are discarded"
                        }
                        for (k, &byte) in raw.iter().enumerate() {
                            let b = at + k as u64;
                            // two lanes of one instruction write one byte differently: which
                            // lands last is the hardware's
                            if let Some(&(other, before)) = written.get(&b)
                                && other != lane
                                && before != byte
                            {
                                raced.push(b);
                            }
                            written.insert(b, (lane, byte));
                        }
                        self.lds_write(w, lane, at, &raw, t, None)?;
                    }
                }
                for b in raced {
                    self.lds_t[b as usize].unordered = true;
                }
            }
            DsOp::Atomic { rtn } => {
                let rest = name.trim_start_matches("ds_").replace("_rtn", "");
                let class = sem::commute_class(&rest);
                let commutes = class.is_some();
                // lanes per dword inside the LDS: past it no dword is raced for
                let mut seen: HashMap<u64, u32> = HashMap::new();
                for &lane in &active {
                    let at = addr_of(self, lane)?.wrapping_add(offset) & 0xFFFF_FFFF;
                    if self.lds_in(w, at, 4)? {
                        *seen.entry(at).or_default() += 1;
                    }
                }
                for &lane in &active {
                    let at = addr_of(self, lane)?.wrapping_add(offset) & 0xFFFF_FFFF;
                    let read = |m: &Self, i: usize| -> Val {
                        match data.get(i) {
                            Some(&Operand::Vector { reg, .. }) => m.vget(w, reg, lane, false),
                            _ => (0, Tol::EXACT),
                        }
                    };
                    let ((d0, t0), (d1, t1)) = (read(self, 0), read(self, 1));
                    let inside = self.lds_in(w, at, 4)?;
                    let (old, ot) = if inside {
                        let (raw, t) = self.lds_read(w, lane, at, 4, rtn || !commutes)?;
                        (u32::from_le_bytes(raw.try_into().expect("4")), t)
                    } else {
                        (0, Tol::EXACT)
                    };
                    let new = sem::atomic32(&rest, old, d0 as u32, d1 as u32)?;
                    let contended = seen.get(&at).copied().unwrap_or(0) > 1;
                    if inside {
                        let mut t = t0.join(t1).join(ot);
                        // lanes racing for a dword with an atomic whose order matters leave
                        // whichever came last
                        t.unordered |= contended && !commutes;
                        // (a return is a read, noted above: another wave's write then refuses)
                        self.lds_write(w, lane, at, &new.to_le_bytes(), t, class)?;
                    }
                    if rtn {
                        let (reg, _) = Self::vreg(dst)?;
                        let mut rt = ot;
                        rt.unordered |= contended; // the lanes' order is the hardware's
                        self.vset(w, reg, lane, 1, u64::from(old), rt);
                    }
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------------------------
    // buffers (8.1) and the global segment (9.4)
    // -----------------------------------------------------------------------------------------

    fn buffer(&mut self, w: usize, inst: &Instruction, name: &str) -> R<()> {
        let Fields::Buffer { data, addr, rsrc, soffset, offset, offen, idxen, format, lds, tfe, cache } =
            inst.fields(self.g.wave64)
        else {
            unreachable!("a MUBUF or MTBUF")
        };
        if format.is_some() {
            return Err("typed buffer instructions (MTBUF) are not in the oracle yet".into());
        }
        if lds || tfe {
            return Err("buffer loads to LDS and TFE are not modelled".into());
        }
        let Some(op) = buffer(name) else { return Err(format!("{name}: not in the oracle's buffer operations")) };
        if op == BufOp::Invalidate {
            return Ok(());
        }
        self.exec_known(w)?;
        let Operand::Scalar { code: rbase, .. } = rsrc else { return Err("no resource".into()) };
        let mut words = [0u32; 4];
        let mut rt = Tol::EXACT;
        for (i, word) in words.iter_mut().enumerate() {
            let (v, t) = self.sget(w, rbase + i as u8, 1);
            *word = v as u32;
            rt = rt.join(t);
        }
        if !rt.exact() {
            return Err("a buffer resource the oracle does not know exactly".into());
        }
        let r = BufferResource::decode(words);
        r.check()?;
        let (so, st) = self.sread(w, soffset, false)?;
        if !st.exact() {
            return Err("an SGPR offset the oracle does not know exactly".into());
        }
        let so = so as u32;
        let exec = self.exec(w);
        let Operand::Vector { reg: areg, .. } = addr else { return Err("no address VGPR".into()) };
        let active: Vec<usize> = (0..self.lanes).filter(|&l| exec >> l & 1 == 1).collect();
        let mut contended: HashMap<u64, u32> = HashMap::new();
        let mut places = Vec::new();
        for &lane in &active {
            let mut at = 0;
            let mut part = |want: bool| -> R<u32> {
                if !want {
                    return Ok(0);
                }
                let (v, t) = self.vget(w, areg.wrapping_add(at), lane, false);
                at += 1;
                if !t.exact() {
                    return Err("a buffer index or offset the oracle does not know exactly".into());
                }
                Ok(v as u32)
            };
            // with both, the index is VADDR and the offset VADDR+1
            let vindex = part(idxen)?;
            let voffset = part(offen)?;
            places.push((lane, vindex, voffset));
            let a = sem::buffer_address(&r, vindex, voffset, u32::from(offset), so, lane as u32);
            // lanes past the range race for nothing
            if !(r.unbound || sem::out_of_range(&r, &a, 4, so)) {
                *contended.entry(a.address & !3).or_default() += 1;
            }
        }
        let data_reg = match data {
            Some(Operand::Vector { reg, .. }) => reg,
            _ => return Err("no data VGPR".into()),
        };
        let dst_dwords = match op {
            BufOp::Load { dwords } => dwords,
            BufOp::LoadSmall { .. } => 1,
            BufOp::LoadFormat { components } => components,
            BufOp::Atomic if cache.glc => 1,
            _ => 0,
        };
        if dst_dwords > 0 {
            self.dst_fits(w, data_reg, dst_dwords)?;
        }
        self.src_fits(w, areg, usize::from(idxen) + usize::from(offen))?;
        let data_dwords = match op {
            BufOp::Store { dwords } => dwords,
            BufOp::StoreSmall { .. } => 1,
            BufOp::StoreFormat { components } => components,
            BufOp::Atomic if name.ends_with("cmpswap") => 2,
            BufOp::Atomic => 1,
            _ => 0,
        };
        self.src_fits(w, data_reg, data_dwords)?;
        if matches!(op, BufOp::LoadFormat { .. } | BufOp::StoreFormat { .. }) && !r.unbound {
            let size = sem::format_bytes(r.format).ok_or_else(|| format!("buffer format {} is not modelled", r.format))?;
            if r.swizzle && r.stride != 0 && size > 4 {
                return Err(format!("a swizzled format element of {size} bytes: 8.1.5's swizzle formula is per dword, and the reference gives no layout of the element's dwords"));
            }
        }
        // 8.1.7: a format access is aligned to its channels, and one of several channels to its
        // element (where the two differ, refused)
        let aligned = |a: &sem::BufferAddress| -> R<()> {
            let channel = sem::format_channel_bytes(r.format).unwrap_or(1);
            let element = sem::format_bytes(r.format).unwrap_or(channel);
            if !a.address.is_multiple_of(u64::from(channel)) || (element > channel && !a.address.is_multiple_of(u64::from(element))) {
                return Err(format!("a format access at {:#x}, not aligned to its {channel}-byte channels and {element}-byte element (8.1.7)", a.address));
            }
            Ok(())
        };
        for (lane, vindex, voffset) in places {
            let at = |i: u32| sem::buffer_address(&r, vindex, voffset, u32::from(offset) + i, so, lane as u32);
            match op {
                BufOp::Load { dwords } => {
                    for i in 0..dwords as u32 {
                        let a = at(4 * i);
                        let (v, t) = if r.unbound || sem::out_of_range(&r, &a, 4, so) {
                            (0, Tol::EXACT)
                        } else {
                            // 8.1.7: "the two LSBs of the byte-address are ignored"
                            self.load(w, lane, a.address & !3)?
                        };
                        self.vset(w, data_reg + i as u8, lane, 1, u64::from(v), t);
                    }
                }
                BufOp::Store { dwords } => {
                    for i in 0..dwords as u32 {
                        let a = at(4 * i);
                        if r.unbound || sem::out_of_range(&r, &a, 4, so) {
                            continue;
                        }
                        let (v, t) = self.vget(w, data_reg + i as u8, lane, false);
                        self.store(w, lane, a.address & !3, &(v as u32).to_le_bytes(), t, WriteKind::Store)?;
                    }
                }
                BufOp::LoadSmall { bytes, signed } => {
                    let a = at(0);
                    let (v, t) = if r.unbound || sem::out_of_range(&r, &a, bytes as u32, so) {
                        (0, Tol::EXACT)
                    } else {
                        self.load_small(w, lane, a.address, bytes, signed)?
                    };
                    self.vset(w, data_reg, lane, 1, u64::from(v), t);
                }
                BufOp::StoreSmall { bytes } => {
                    let a = at(0);
                    if !(r.unbound || sem::out_of_range(&r, &a, bytes as u32, so)) {
                        let (v, t) = self.vget(w, data_reg, lane, false);
                        self.store(w, lane, a.address, &(v as u32).to_le_bytes()[..bytes], t, WriteKind::Store)?;
                    }
                }
                BufOp::LoadFormat { components } => {
                    if r.unbound {
                        // 8.1.8: "return 0,0,0,0"
                        for c in 0..components {
                            self.vset(w, data_reg + c as u8, lane, 1, 0, Tol::EXACT);
                        }
                        continue;
                    }
                    let size = sem::format_bytes(r.format)
                        .ok_or_else(|| format!("buffer format {} is not modelled", r.format))?;
                    let a = at(0);
                    aligned(&a)?;
                    let oob = sem::out_of_range(&r, &a, size, so);
                    let element = if oob { None } else { Some(self.mem_bytes(w, lane, a.address, size as usize)?) };
                    for c in 0..components {
                        let (v, t) = match (&element, r.dst_sel[c]) {
                            (_, 0) => (0, Tol::EXACT),
                            (_, 1) => return Err("dst_sel 1: the reference does not say 1 of which kind".into()),
                            (None, _) => (0, Tol::EXACT),
                            (Some((_, t)), _) if t.approximate() && !matches!(r.format, 22 | 64 | 74 | 77) => {
                                return Err("a format channel of bytes the oracle holds approximate".into());
                            }
                            (Some((bytes, t)), sel @ 4..=7) => {
                                let (v, exact) = sem::format_channel(r.format, bytes, usize::from(sel - 4))
                                    .ok_or_else(|| format!("channel {} of buffer format {}", sel - 4, r.format))?;
                                (v, if exact { *t } else { t.join(Tol { ulps: 1, ..Tol::EXACT }) })
                            }
                            (_, sel) => return Err(format!("dst_sel {sel} is reserved")),
                        };
                        self.vset(w, data_reg + c as u8, lane, 1, u64::from(v), t);
                    }
                }
                BufOp::StoreFormat { components } => {
                    if r.unbound {
                        continue;
                    }
                    if !matches!(r.format, 20..=22 | 62..=64 | 72..=77) || r.dst_sel != [4, 5, 6, 7] {
                        return Err(format!("a format store to format {} (dst_sel {:?}): only 32-bit channels \
                                            are modelled",
                            r.format, r.dst_sel));
                    }
                    let size = sem::format_bytes(r.format).expect("a 32-bit format");
                    let a = at(0);
                    aligned(&a)?;
                    if sem::out_of_range(&r, &a, size, so) {
                        continue;
                    }
                    for c in 0..components.min(size as usize / 4) {
                        let (v, t) = self.vget(w, data_reg + c as u8, lane, false);
                        self.store(w, lane, a.address + 4 * c as u64, &(v as u32).to_le_bytes(), t, WriteKind::Store)?;
                    }
                }
                BufOp::Atomic => {
                    let a = at(0);
                    let op = name.trim_start_matches("buffer_atomic_");
                    let (d0, t0) = self.vget(w, data_reg, lane, false);
                    let (d1, t1) = if op == "cmpswap" { self.vget(w, data_reg + 1, lane, false) } else { (0, Tol::EXACT) };
                    let (old, ot) = if r.unbound || sem::out_of_range(&r, &a, 4, so) {
                        (0, Tol::EXACT)
                    } else {
                        let class = sem::commute_class(op);
                        let (old, ot) = self.mem.read_dword(a.address & !3, self.who(w, lane), cache.glc || class.is_none())?;
                        let new = sem::atomic32(op, old, d0 as u32, d1 as u32)?;
                        let kind = class.map_or(WriteKind::Atomic, WriteKind::Commutes);
                        self.store(w, lane, a.address & !3, &new.to_le_bytes(), t0.join(t1).join(ot), kind)?;
                        (old, ot)
                    };
                    if cache.glc {
                        let mut t = ot;
                        if contended.get(&(a.address & !3)).copied().unwrap_or(0) > 1 {
                            t.unordered = true;
                        }
                        self.vset(w, data_reg, lane, 1, u64::from(old), t);
                    }
                }
                BufOp::Invalidate => {}
            }
        }
        Ok(())
    }

    fn global(&mut self, w: usize, inst: &Instruction, name: &str) -> R<()> {
        let Fields::Flat { dst, addr, data, saddr, offset, lds, cache } = inst.fields(self.g.wave64) else {
            unreachable!("a GLOBAL")
        };
        if lds {
            return Err("global loads to LDS are not modelled".into());
        }
        let Some(op) = global(name) else { return Err(format!("{name}: not in the oracle's global operations")) };
        self.exec_known(w)?;
        let dst_dwords = match op {
            BufOp::Load { dwords } => dwords,
            BufOp::LoadSmall { .. } => 1,
            BufOp::Atomic if cache.glc => 1,
            _ => 0,
        };
        if dst_dwords > 0 {
            let (reg, _) = Self::vreg(dst)?;
            self.dst_fits(w, reg, dst_dwords)?;
        }
        if let Some(Operand::Vector { reg, .. }) = addr {
            self.src_fits(w, reg, if matches!(saddr, Some(Operand::Scalar { code, .. }) if code != NULL) { 1 } else { 2 })?;
        }
        if let Some(Operand::Vector { reg, .. }) = data {
            let n = match op {
                BufOp::Store { dwords } => dwords,
                _ if name.ends_with("cmpswap") => 2,
                _ => 1,
            };
            self.src_fits(w, reg, n)?;
        }
        let exec = self.exec(w);
        let Some(Operand::Vector { reg: areg, .. }) = addr else { return Err("no address VGPR".into()) };
        let base = match saddr {
            Some(Operand::Scalar { code, .. }) if code != NULL => {
                let (b, t) = self.sget(w, code, 2);
                if !t.exact() {
                    return Err("an SGPR address the oracle does not know exactly".into());
                }
                Some(b)
            }
            _ => None,
        };
        let active: Vec<usize> = (0..self.lanes).filter(|&l| exec >> l & 1 == 1).collect();
        let mut addresses = Vec::new();
        for &lane in &active {
            // 9.3: with SADDR, SGPR[SADDR] + VGPR[ADDR] (32 bits, unsigned) + offset; without,
            // the 64-bit VGPR pair + offset
            let (v, t) = self.vget(w, areg, lane, base.is_none());
            if !t.exact() {
                return Err("an address the oracle does not know exactly".into());
            }
            let a = base.map_or(v, |b| b.wrapping_add(v & 0xFFFF_FFFF)).wrapping_add(i64::from(offset) as u64);
            addresses.push((lane, a));
        }
        let mut contended: HashMap<u64, u32> = HashMap::new();
        for &(_, a) in &addresses {
            *contended.entry(a).or_default() += 1;
        }
        let data_reg = match (data, dst) {
            (Some(Operand::Vector { reg, .. }), _) => Some(reg),
            _ => None,
        };
        for (lane, a) in addresses {
            match op {
                BufOp::Load { dwords } => {
                    let (reg, _) = Self::vreg(dst)?;
                    for i in 0..dwords {
                        let (v, t) = self.load(w, lane, a + 4 * i as u64)?;
                        self.vset(w, reg + i as u8, lane, 1, u64::from(v), t);
                    }
                }
                BufOp::LoadSmall { bytes, signed } => {
                    let (reg, _) = Self::vreg(dst)?;
                    let (v, t) = self.load_small(w, lane, a, bytes, signed)?;
                    self.vset(w, reg, lane, 1, u64::from(v), t);
                }
                BufOp::Store { dwords } => {
                    let reg = data_reg.ok_or("no data VGPR")?;
                    for i in 0..dwords {
                        let (v, t) = self.vget(w, reg + i as u8, lane, false);
                        self.store(w, lane, a + 4 * i as u64, &(v as u32).to_le_bytes(), t, WriteKind::Store)?;
                    }
                }
                BufOp::StoreSmall { bytes } => {
                    let reg = data_reg.ok_or("no data VGPR")?;
                    let (v, t) = self.vget(w, reg, lane, false);
                    self.store(w, lane, a, &(v as u32).to_le_bytes()[..bytes], t, WriteKind::Store)?;
                }
                BufOp::Atomic => {
                    let op = name.trim_start_matches("global_atomic_");
                    let reg = data_reg.ok_or("no data VGPR")?;
                    let (d0, t0) = self.vget(w, reg, lane, false);
                    let (d1, t1) = if op == "cmpswap" { self.vget(w, reg + 1, lane, false) } else { (0, Tol::EXACT) };
                    let class = sem::commute_class(op);
                    let (old, ot) = self.mem.read_dword(a, self.who(w, lane), cache.glc || class.is_none())?;
                    let new = sem::atomic32(op, old, d0 as u32, d1 as u32)?;
                    let kind = class.map_or(WriteKind::Atomic, WriteKind::Commutes);
                    self.store(w, lane, a, &new.to_le_bytes(), t0.join(t1).join(ot), kind)?;
                    if cache.glc {
                        let (reg, _) = Self::vreg(dst)?;
                        let mut t = ot;
                        if contended[&a] > 1 {
                            t.unordered = true;
                        }
                        self.vset(w, reg, lane, 1, u64::from(old), t);
                    }
                }
                _ => return Err(format!("{name}: not in the oracle's global operations")),
            }
        }
        Ok(())
    }

    fn load(&mut self, w: usize, lane: usize, a: u64) -> R<(u32, Tol)> {
        self.mem.read_dword(a, self.who(w, lane), true)
    }

    fn mem_bytes(&mut self, w: usize, lane: usize, a: u64, len: usize) -> R<(Vec<u8>, Tol)> {
        self.mem.read(a, len, self.who(w, lane), true)
    }

    fn load_small(&mut self, w: usize, lane: usize, a: u64, bytes: usize, signed: bool) -> R<(u32, Tol)> {
        let (raw, t) = self.mem_bytes(w, lane, a, bytes)?;
        if t.approximate() {
            return Err("a part of a dword the oracle holds approximate: an integer with no ULPs".into());
        }
        let v = raw.iter().rev().fold(0u32, |x, &b| x << 8 | u32::from(b));
        let bits = 8 * bytes as u32;
        Ok((if signed { ((v << (32 - bits)) as i32 >> (32 - bits)) as u32 } else { v }, t))
    }

    fn store(&mut self, w: usize, lane: usize, a: u64, bytes: &[u8], t: Tol, kind: WriteKind) -> R<()> {
        if bytes.len() != 4 && t.approximate() {
            return Err("a part of a dword stored from an approximate value".into());
        }
        let who = self.who(w, lane);
        self.mem.write(a, bytes, t, who, kind)
    }
}

/// Whether single `b` and every value within `ulps` of it are normal: an approximate value
/// whose interval reaches a denormal (which a host may flush) or past the largest finite one
/// has no bound a comparison can take.
fn normal_within(b: u32, ulps: u32) -> bool {
    let m = b & 0x7FFF_FFFF;
    (0x0080_0000..=0x7F7F_FFFF).contains(&m) && m - 0x0080_0000 >= ulps && 0x7F7F_FFFF - m >= ulps
}

fn dst_dwords(dst: Option<Operand>) -> u8 {
    match dst {
        Some(Operand::Vector { dwords, .. }) => dwords,
        _ => 0,
    }
}

/// SDWA's source select (6.8): a byte (0-3), a word (4-5) or the dword (6), zero- or
/// sign-extended.
fn sdwa_select(v: u32, sel: u8, sext: bool) -> R<u64> {
    let (shift, bits) = match sel {
        0..=3 => (8 * u32::from(sel), 8),
        4 | 5 => (16 * u32::from(sel - 4), 16),
        6 => (0, 32),
        _ => return Err(format!("SDWA select {sel} is reserved")),
    };
    let x = v >> shift;
    Ok(u64::from(if bits == 32 {
        x
    } else if sext {
        ((x << (32 - bits)) as i32 >> (32 - bits)) as u32
    } else {
        x & ((1 << bits) - 1)
    }))
}

/// SDWA's destination select (6.8) with DST_UNUSED 0 (pad with zeros) or 2 (preserve); the
/// sign-extending form is refused (the reference does not say which bits it fills).
fn sdwa_place(r: u32, old: u32, sel: u8, unused: u8) -> R<u32> {
    let (shift, bits) = match sel {
        0..=3 => (8 * u32::from(sel), 8),
        4 | 5 => (16 * u32::from(sel - 4), 16),
        _ => return Err(format!("SDWA select {sel} is reserved")),
    };
    let mask = ((1u32 << bits) - 1) << shift;
    let placed = (r << shift) & mask;
    Ok(match unused {
        0 => placed,
        2 => (old & !mask) | placed,
        _ => return Err("SDWA's sign-extending DST_UNUSED is not in the oracle".into()),
    })
}
