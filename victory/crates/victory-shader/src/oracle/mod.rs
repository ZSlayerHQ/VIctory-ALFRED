//! A second way of running a console program: an interpreter of GFX10 machine code on the CPU,
//! one lane at a time, as the reference for the translator's fixtures and for local checks of
//! dispatches. Test and diagnostics use only: nothing in the translation path calls it.
//!
//! It runs one thread group: its waves of 64 (or 32) lanes, each with its own SGPRs, VGPRs per
//! lane, EXEC, VCC, SCC and M0, sharing the group's LDS; `s_barrier` holds a wave until every
//! wave still running has come (the reference's `S_BARRIER`: "If some waves in the threadgroup
//! have already terminated, this waits on only the surviving waves"). Waves run one at a time
//! to their next barrier or end, one order the hardware may take. Memory is plain byte arrays
//! at 64-bit addresses ([`Memory`]); descriptors are read from it as the reference decodes them.
//!
//! What each instruction does is [`semantics`], written from AMD's "RDNA 2" Instruction Set
//! Architecture reference alone. Anything it does not implement, it refuses ([`Refusal`]) with
//! the instruction and the reason: it never guesses. Every value carries how far the oracle
//! trusts it ([`Tol`]): exact, approximate (from an instruction whose rounding the reference
//! does not pin, or that the console fuses where a host may not), order-dependent (what lanes or
//! waves raced for: an atomic's return, a dword several of them stored to), or undefined (read
//! from a register or LDS nothing wrote); [`compare`] keeps those classes apart.
//!
//! Order between waves: the oracle runs one wave to its barrier before the next, one order of
//! many. A dword one wave reads and another writes between the same two barriers has no one
//! answer, and the oracle refuses it; a dword two waves (or two lanes) write is order-dependent
//! (in memory, with the values its writers left: [`Memory::choices`]). An approximate value is
//! never a branch's condition, an EXEC mask, an address or an integer: the oracle refuses
//! where one would become one.

pub mod compare;
#[cfg(test)]
pub(crate) mod fixture;
mod machine;
pub mod semantics;
#[cfg(test)]
mod tests;

use std::collections::HashMap;

pub use semantics::{Mode, Wave64Permute};

/// How far the oracle trusts a value (one dword of a register lane, LDS or memory).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Tol {
    /// ULPs a single-precision value may differ by.
    pub ulps: u32,
    /// An absolute error it may differ by.
    pub abs: f32,
    /// A NaN whose bits the reference does not give: any NaN matches.
    pub nan: bool,
    /// Returned by an atomic other lanes raced for: any order of them is the hardware's.
    pub unordered: bool,
    /// Read from a register or LDS dword nothing wrote: the console's value is unknown.
    pub undefined: bool,
}

impl Tol {
    pub const EXACT: Tol = Tol { ulps: 0, abs: 0.0, nan: false, unordered: false, undefined: false };
    pub const UNDEFINED: Tol = Tol { undefined: true, ..Tol::EXACT };

    pub fn exact(&self) -> bool {
        *self == Tol::EXACT
    }

    /// Within ULPs, an absolute error, or a NaN of unpinned bits.
    pub fn approximate(&self) -> bool {
        self.ulps > 0 || self.abs > 0.0 || self.nan
    }

    /// The trust of a value made from values of these two trusts.
    pub fn join(self, o: Tol) -> Tol {
        Tol {
            ulps: self.ulps.max(o.ulps),
            abs: self.abs.max(o.abs),
            nan: self.nan || o.nan,
            unordered: self.unordered || o.unordered,
            undefined: self.undefined || o.undefined,
        }
    }
}

/// Memory as plain byte arrays at 64-bit addresses.
#[derive(Debug, Clone, Default)]
pub struct Memory {
    regions: Vec<Region>,
}

#[derive(Debug, Clone)]
struct Region {
    base: u64,
    bytes: Vec<u8>,
    /// Per dword.
    tol: Vec<Tol>,
    /// Per dword written, each lane that wrote it ([`race`]).
    writers: HashMap<usize, Vec<Writer>>,
    /// Per dword accessed, the waves that did between the current two barriers.
    seen: HashMap<usize, Seen>,
}

/// Who makes an access: the wave, its lane, and the barriers the group has passed (`epoch`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Who {
    pub wave: usize,
    pub lane: usize,
    pub epoch: u32,
}

/// What a write is, for the order of its writers: a store, an atomic whose result does not
/// depend on the order ([`semantics::commutes`]), or another atomic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WriteKind {
    Store,
    /// Of this commuting class ([`semantics::commute_class`]).
    Commutes(u8),
    Atomic,
}

/// One lane's writes of a dword: the bytes it wrote, whether any write of it was of a part of
/// the dword, the dword after its last write and that dword's trust, and the barriers the group
/// had passed then.
#[derive(Debug, Clone)]
struct Writer {
    who: (usize, usize),
    bytes: u8,
    partial: bool,
    value: u32,
    tol: Tol,
    kind: WriteKind,
    epoch: u32,
}

/// What the lanes that wrote one dword leave there, in an order the hardware chooses.
#[derive(Debug, Clone, PartialEq)]
enum Race {
    /// One answer: one writer, atomics of one commuting class, or stores of disjoint bytes or
    /// of one value.
    None,
    /// Whole-dword stores of exact values: one of these, whichever landed last.
    Choices(Vec<u32>),
    /// Whole-dword stores of values not all exact: order-dependent, no value checked.
    Unordered,
}

/// The writers of one dword, as [`admitted`] lets them be.
fn race(writers: Option<&Vec<Writer>>) -> Race {
    let Some(ws) = writers.filter(|ws| ws.len() > 1) else { return Race::None };
    if ws.iter().any(|w| w.partial) || ws.iter().all(|w| matches!(w.kind, WriteKind::Commutes(_))) {
        return Race::None; // disjoint bytes, or one commuting class
    }
    let mut values: Vec<u32> = ws.iter().map(|w| w.value).collect();
    values.sort_unstable();
    values.dedup();
    match (values.len(), ws.iter().all(|w| w.tol.exact())) {
        (1, _) => Race::None,
        (_, true) => Race::Choices(values),
        (_, false) => Race::Unordered,
    }
}

/// Whether the writers of one dword leave what the oracle can establish: one writer; or, all
/// between the same two barriers, atomics of one commuting class, whole-dword stores (one of
/// their values is left), or stores of disjoint bytes. Anything else is refused: mixed atomics,
/// an atomic whose order matters, stores that overlap in part, and writers on either side of a
/// barrier (which orders their stores only with the waits before it, not modelled).
fn admitted(ws: &[Writer]) -> Result<(), String> {
    if ws.len() <= 1 {
        return Ok(());
    }
    if ws.iter().any(|w| w.epoch != ws[0].epoch) {
        return Err("a dword two lanes write on either side of a barrier: whether the barrier orders \
                    the stores (the waits before it) is not modelled"
            .into());
    }
    let one_class = match ws[0].kind {
        WriteKind::Commutes(c) => ws.iter().all(|w| w.kind == WriteKind::Commutes(c)),
        _ => false,
    };
    let stores = ws.iter().all(|w| w.kind == WriteKind::Store);
    let whole = stores && ws.iter().all(|w| !w.partial);
    let disjoint = stores
        && ws.iter().all(|w| w.partial && w.bytes != 0xF)
        && ws.iter().map(|w| w.bytes).try_fold(0u8, |all, b| (all & b == 0).then_some(all | b)).is_some();
    if one_class || whole || disjoint {
        return Ok(());
    }
    Err("a dword lanes or waves race for with mixed atomics, an atomic whose order matters, or \
         stores that overlap in part: what is left the oracle cannot establish"
        .into())
}

/// The waves that read and wrote one dword between the same two barriers.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Seen {
    epoch: u32,
    read: u64,
    wrote: u64,
    commuted: u64,
    /// The commuting class of those atomics (0 none yet, 255 more than one).
    class: u8,
}

pub(crate) const WAVES_RACE: &str =
    "a dword one wave reads and another writes between the same two barriers: which comes first is the hardware's";

impl Seen {
    fn at(&mut self, who: Who) -> u64 {
        if self.epoch != who.epoch {
            *self = Seen { epoch: who.epoch, ..Seen::default() };
        }
        1u64 << who.wave
    }

    /// A read by `who`, of a value something wrote (`defined`) or not.
    pub(crate) fn read(&mut self, who: Who, defined: bool) -> Result<(), String> {
        let me = self.at(who);
        if (self.wrote | self.commuted) & !me != 0 {
            return Err(WAVES_RACE.into());
        }
        if defined {
            self.read |= me;
        }
        Ok(())
    }

    /// A write by `who` (an atomic of commuting class `class`, or not): whether another wave
    /// wrote the dword between these barriers in a way whose order with this write matters.
    pub(crate) fn write(&mut self, who: Who, class: Option<u8>) -> Result<bool, String> {
        let me = self.at(who);
        if self.read & !me != 0 {
            return Err(WAVES_RACE.into());
        }
        let others = self.commuted & !me != 0;
        let clash = self.wrote & !me != 0 || (others && class.is_none_or(|c| self.class != c));
        match class {
            Some(c) => {
                self.commuted |= me;
                self.class = if self.class == 0 || self.class == c { c } else { 255 };
            }
            None => self.wrote |= me,
        }
        Ok(clash)
    }
}

impl Memory {
    pub fn new() -> Memory {
        Memory::default()
    }

    /// Maps `bytes` at `base` (a multiple of 4; no overlap with another region).
    pub fn map(&mut self, base: u64, bytes: Vec<u8>) {
        assert!(base.is_multiple_of(4) && bytes.len().is_multiple_of(4), "regions are whole dwords");
        let end = base + bytes.len() as u64;
        assert!(
            self.regions.iter().all(|r| end <= r.base || base >= r.base + r.bytes.len() as u64),
            "regions do not overlap"
        );
        let tol = vec![Tol::EXACT; bytes.len().div_ceil(4)];
        self.regions.push(Region { base, bytes, tol, writers: HashMap::new(), seen: HashMap::new() });
    }

    /// Maps `dwords` at `base`.
    pub fn map_dwords(&mut self, base: u64, dwords: &[u32]) {
        self.map(base, dwords.iter().flat_map(|d| d.to_le_bytes()).collect());
    }

    /// The dwords of the region mapped at `base`, and the trust of each: order-dependent where
    /// several lanes or waves wrote a dword and the order they land in changes what is left.
    pub fn dwords(&self, base: u64) -> Option<(Vec<u32>, Vec<Tol>)> {
        let r = self.regions.iter().find(|r| r.base == base)?;
        let words = r.bytes.chunks(4).map(|c| c.iter().rev().fold(0, |a, &b| a << 8 | u32::from(b))).collect();
        let mut tol = r.tol.clone();
        for (&i, ws) in &r.writers {
            if race(Some(ws)) != Race::None {
                tol[i].unordered = true;
            }
        }
        Some((words, tol))
    }

    /// Per dword of the region at `base`, the values one of which it holds where several lanes
    /// or waves stored exact whole dwords to it (whichever landed last); empty elsewhere.
    pub fn choices(&self, base: u64) -> Option<Vec<Vec<u32>>> {
        let r = self.regions.iter().find(|r| r.base == base)?;
        let mut out = vec![Vec::new(); r.tol.len()];
        for (&i, ws) in &r.writers {
            if let Race::Choices(values) = race(Some(ws)) {
                out[i] = values;
            }
        }
        Some(out)
    }

    fn find(&mut self, address: u64, len: u64) -> Result<(&mut Region, usize), String> {
        self.regions
            .iter_mut()
            .find(|r| address >= r.base && address + len <= r.base + r.bytes.len() as u64)
            .map(|r| {
                let at = (address - r.base) as usize;
                (r, at)
            })
            .ok_or_else(|| format!("an access at {address:#x}, outside memory"))
    }

    /// `len` bytes at `address`, read by `who` (`note`: a read whose order with other waves'
    /// writes matters; an atomic that gives nothing back reads without one).
    pub(crate) fn read(&mut self, address: u64, len: usize, who: Who, note: bool) -> Result<(Vec<u8>, Tol), String> {
        let (r, at) = self.find(address, len as u64)?;
        let mut tol = Tol::EXACT;
        for i in at / 4..(at + len).div_ceil(4) {
            if note {
                r.seen.entry(i).or_default().read(who, !r.tol[i].undefined)?;
            }
            let mut t = r.tol[i];
            if race(r.writers.get(&i)) != Race::None {
                t.unordered = true;
            }
            tol = tol.join(t);
        }
        Ok((r.bytes[at..at + len].to_vec(), tol))
    }

    pub(crate) fn read_dword(&mut self, address: u64, who: Who, note: bool) -> Result<(u32, Tol), String> {
        let (b, t) = self.read(address, 4, who, note)?;
        Ok((u32::from_le_bytes(b.try_into().expect("4 bytes")), t))
    }

    /// Writes `bytes` at `address` for `who`: a dword written whole takes `tol`, one written in
    /// part joins it with what it kept.
    pub(crate) fn write(&mut self, address: u64, bytes: &[u8], tol: Tol, who: Who, kind: WriteKind) -> Result<(), String> {
        let (r, at) = self.find(address, bytes.len() as u64)?;
        let end = at + bytes.len();
        r.bytes[at..end].copy_from_slice(bytes);
        for i in at / 4..end.div_ceil(4) {
            // (the order of two waves' writes is the writers' to settle, below)
            let class = match kind {
                WriteKind::Commutes(c) => Some(c),
                _ => None,
            };
            r.seen.entry(i).or_default().write(who, class)?;
            let mask = (0..4).filter(|k| (at..end).contains(&(4 * i + k))).fold(0u8, |m, k| m | 1 << k);
            r.tol[i] = if mask == 0xF { tol } else { r.tol[i].join(tol) };
            let value = u32::from_le_bytes(r.bytes[4 * i..4 * i + 4].try_into().expect("a dword"));
            let ws = r.writers.entry(i).or_default();
            match ws.iter_mut().find(|w| w.who == (who.wave, who.lane)) {
                Some(w) => {
                    w.bytes |= mask;
                    w.partial |= mask != 0xF;
                    (w.value, w.tol, w.epoch) = (value, r.tol[i], who.epoch);
                    if w.kind != kind {
                        w.kind = WriteKind::Atomic;
                    }
                }
                None => ws.push(Writer {
                    who: (who.wave, who.lane),
                    bytes: mask,
                    partial: mask != 0xF,
                    value,
                    tol: r.tol[i],
                    kind,
                    epoch: who.epoch,
                }),
            }
            admitted(ws)?;
        }
        Ok(())
    }
}

/// One thread group of a compute program, as the hardware starts it (3.12, "Initial Wave
/// State").
#[derive(Debug, Clone)]
pub struct Group {
    /// The program's GFX10 code, offset 0 its entry.
    pub code: Vec<u8>,
    pub wave64: bool,
    /// Threads of the group in x, y, z.
    pub threads: [u32; 3],
    /// The group's id, for the SGPRs `COMPUTE_PGM_RSRC2` enables.
    pub group_id: [u32; 3],
    /// `COMPUTE_PGM_RSRC2`: user SGPRs, the system SGPRs after them, the thread id VGPRs, LDS.
    pub rsrc2: u32,
    /// The user SGPRs' values, from s0 on.
    pub user_sgprs: Vec<u32>,
    /// VGPRs the program has (6.2.6: a source beyond them reads v0, a destination beyond them
    /// is not written).
    pub vgprs: u16,
    /// The MODE register at the start (from `SPI_SHADER_PGM_RSRC1`, which a program record does
    /// not keep: the caller's choice).
    pub mode: Mode,
    /// Which reading of a wave64 permute to take ([`Wave64Permute`]).
    pub permute: Wave64Permute,
    /// Instructions a wave may run before the oracle gives up on it.
    pub step_limit: u64,
}

/// Why the oracle did not run a program to its end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    /// Byte offset of the instruction in the code.
    pub offset: usize,
    /// Its name (or what stood there).
    pub instruction: String,
    pub reason: String,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#x} {}: {}", self.offset, self.instruction, self.reason)
    }
}

/// What a run did.
#[derive(Debug, Clone, Default)]
pub struct Ran {
    /// Instructions run, over all waves.
    pub instructions: u64,
}

/// Runs `group` against `memory` to the end of every wave.
pub fn run(group: &Group, memory: &mut Memory) -> Result<Ran, Refusal> {
    machine::run(group, memory)
}

/// Whether the oracle implements `name` (an opcode name of [`crate::gfx10`]), for coverage
/// lists: the same test its dispatch makes, without running anything.
pub fn implements(name: &str) -> bool {
    machine::implements(name)
}
