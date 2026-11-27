//! The paired form ([`super::with_paired_lanes`]): a console wave of 64 lanes run by one host
//! wave of 32, host lane n holding console lanes n and n + 32. What a lane runs is lowered
//! twice, once for each lane of the thread ([`Lower::swap_lanes`]); what the wave has once is
//! lowered once: its scalar code and memory ([`Lower::paired_inst`]), the reads of other
//! lanes ([`Lower::paired_cross`]) and the values of whole-wave sections
//! ([`Lower::paired_shape`]).
//!
//! Ported from fxceless pull request #854 by PiloteCode (commit 26600c61e), with three
//! changes: scalar code is lowered once, not once per lane; LDS accesses one lane may share
//! with another meet at a group barrier instead of a fence; and a float sum is added in the
//! console's order, or refused.

use std::collections::HashMap;

use super::{
    Attr, Bin, Bool, Fields, Inst, Lower, Operand, Reg, Site, Stop, V, Val, Value, Why, flags, pred, unsupported,
};
use crate::emit::wave::{Op, Shape};

thread_local! {
    /// See [`with_paired_lanes`].
    static PAIRED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` with console wave64 compute programs on 32-lane hosts lowered in the paired form:
/// each console wave is one host wave of 32 lanes, and host lane n runs console lanes n and
/// n + 32. The group has half the console's threads; each thread runs what a lane does twice,
/// once for each of its lanes, in the order of the console's instructions. Cross-lane
/// operations reach any of the 64 lanes through the host wave's own operations on either half
/// (lane n + 32 is lane n's upper half), and a lane mask is the two ballots. The caller makes
/// every per-lane branch one the wave takes together, as the console does
/// (`structure::structure_wide`): the wave goes where some lane has to, and EXEC stays each
/// lane's guard. So control flow is the same in every thread of a host wave, and the
/// program's barriers are reached as on the console.
///
/// What one lane writes to LDS another reads at a later instruction, with no barrier on the
/// console (a wave's lanes run in step). On the host the two lanes are two threads, and D3D12
/// orders no groupshared accesses of two threads without a group barrier: a fence orders one
/// thread's own accesses only, and NVIDIA's threads of a wave need not run in step. So
/// between two LDS accesses of which one writes, with no barrier of the program between them,
/// the host wave meets (a group barrier with its shared-memory fence). That is a barrier every
/// thread of the group reaches only where the group is one console wave (control flow is the
/// wave's); a group of several console waves whose waves walk apart cannot meet there, and
/// such a program is refused at the first access that would need it
/// ([`PAIRED_LDS_APART`]).
// Ported from fxceless pull request #854 by PiloteCode (commit 26600c61e); the meetings
// between LDS accesses replace that form's fences, which relied on a host wave's threads
// running in step.
pub(in crate::emit) fn with_paired_lanes<R>(f: impl FnOnce() -> R) -> R {
    let was = PAIRED.replace(true);
    let r = f();
    PAIRED.set(was);
    r
}

/// Whether this thread lowers in the paired form ([`with_paired_lanes`]).
pub(super) fn on_this_thread() -> bool {
    PAIRED.get()
}

/// A paired group of several console waves whose LDS accesses need the console's order between
/// lanes of one wave where the program has no barrier ([`with_paired_lanes`]).
pub(in crate::emit) const PAIRED_LDS_APART: Why =
    "LDS shared between lanes of a wave without a barrier, in a paired group of several waves";

/// What the lowering keeps per console lane, for the paired form ([`with_paired_lanes`]): each
/// host thread runs two lanes of one console wave, and what a lane runs is lowered once for
/// each with this state exchanged between them ([`Lower::swap_lanes`]): the values (scalar ones
/// too, the same in both), the lane booleans and their joins, the handles of descriptors a lane
/// may have its own of, the phis still to fill.
#[derive(Default)]
pub(super) struct LaneState {
    vals: HashMap<Value, Val>,
    floats: HashMap<Value, V>,
    bools: HashMap<(usize, Bool), V>,
    lane_bits: HashMap<Value, V>,
    lane_fail: HashMap<Value, Why>,
    mask_phis: HashMap<Value, V>,
    pub(super) pending_masks: Vec<(V, usize, usize)>,
    data_phis: Vec<(V, usize, usize)>,
    handles: HashMap<(usize, Site, u8, bool, (u32, u32)), V>,
    strides: HashMap<(usize, Site, u8), V>,
    clane: Option<V>,
}

impl<'a> Lower<'a> {
    /// The paired form: makes the other console lane of this thread the one being lowered.
    pub(super) fn swap_lanes(&mut self) {
        let Some(t) = self.twin.as_deref_mut() else {
            return;
        };
        std::mem::swap(&mut self.vals, &mut t.vals);
        std::mem::swap(&mut self.floats, &mut t.floats);
        std::mem::swap(&mut self.bools, &mut t.bools);
        std::mem::swap(&mut self.lane_bits, &mut t.lane_bits);
        std::mem::swap(&mut self.lane_fail, &mut t.lane_fail);
        std::mem::swap(&mut self.mask_phis, &mut t.mask_phis);
        std::mem::swap(&mut self.pending_masks, &mut t.pending_masks);
        std::mem::swap(&mut self.data_phis, &mut t.data_phis);
        std::mem::swap(&mut self.handles, &mut t.handles);
        std::mem::swap(&mut self.strides, &mut t.strides);
        std::mem::swap(&mut self.clane, &mut t.clane);
        self.upper = !self.upper;
    }

    /// The paired form: `f` for each console lane of this thread, the lower one first; the lane
    /// being lowered stays the same.
    pub(super) fn both<T>(&mut self, mut f: impl FnMut(&mut Self) -> Result<T, Why>) -> Result<[T; 2], Why> {
        let was = self.upper;
        if was {
            self.swap_lanes();
        }
        let lo = f(self);
        self.swap_lanes();
        let hi = f(self);
        if !was {
            self.swap_lanes();
        }
        Ok([lo?, hi?])
    }

    /// The paired form: `lo` as the lower lane's value of `v`, `hi` as the upper lane's.
    fn put_lanes(&mut self, v: Value, lo: Val, hi: Val) {
        let (mine, other) = if self.upper { (hi, lo) } else { (lo, hi) };
        self.vals.insert(v, mine);
        if let Some(t) = self.twin.as_deref_mut() {
            t.vals.insert(v, other);
        }
    }

    /// The console thread of the lane being lowered: the thread with group index `g` holds
    /// lane `g % 32` of console wave `g / 32`, and that lane plus 32 (`upper`). The group index
    /// is the one number D3D12 gives each thread; the wave operations the form uses address the
    /// wave's lanes, so the form is right only where lane `g % 32` of a wave is the thread of
    /// group index `g`, which every dispatch checks ([`Self::paired_entry_check`]).
    pub(super) fn paired_thread(&mut self) -> V {
        let (e, i32t) = (self.s.entry, self.s.t.i32);
        let g = self.s.op(e, "flattenedThreadIdInGroup.i32", Attr::ReadNone, i32t, 96, &[]);
        let (k5, k31) = (self.s.m.i32(5), self.s.m.i32(31));
        let wave = self.s.m.bin(e, Bin::LShr, g, k5);
        let k6 = self.s.m.i32(6);
        let base = self.s.m.bin(e, Bin::Shl, wave, k6);
        let lane = self.s.m.bin(e, Bin::And, g, k31);
        let half = self.s.m.i32(if self.upper { 32 } else { 0 });
        let lane = self.s.m.bin(e, Bin::Or, lane, half);
        self.s.m.bin(e, Bin::Or, base, lane)
    }

    /// The paired form's check at the program's entry (`at`), on every dispatch: whether the
    /// group runs as the form takes it, the threads of group index `32w .. 32w + 31` one host
    /// wave of 32 lanes in that order. D3D12 fixes the wave's size (`WaveSize(32)`) but promises
    /// neither that a wave holds 32 threads of the group in a row nor that a thread's lane is its
    /// index. Each thread asks: its wave has 32 lanes (`WaveGetLaneCount`), all of them here
    /// (a ballot of every lane counts 32), its lane is its group index modulo 32, and every lane
    /// of its wave has the same group index divided by 32; each wave asks whether all its lanes
    /// say yes. The group then agrees on one answer through an LDS dword (`halves`): every
    /// thread zeroes it, the group meets, a wave that says no sets it, the group meets, every
    /// thread reads it (and meets once more where that dword is the program's own). Where the
    /// answer is no the whole group leaves at once, before any effect and any barrier of the
    /// program: no stores rather than wrong ones, and no hang.
    /// Costs, each dispatch: one LDS dword, two group barriers (three where the program takes
    /// all 32 KiB of LDS), about twenty instructions a thread.
    // shortcut: a group that fails leaves without a word to the host; a flag in a diagnostics
    // buffer the host reads (to log it and turn the form off for that program) needs a binding
    // the host does not give yet.
    pub(super) fn paired_entry_check(&mut self, at: usize) -> V {
        let i32t = self.s.t.i32;
        self.s.flags |= flags::WAVE_OPS;
        let (k0, k1, k5, k31, k32) = (self.s.m.i32(0), self.s.m.i32(1), self.s.m.i32(5), self.s.m.i32(31), self.s.m.i32(32));
        let lanes = self.s.op(at, "waveGetLaneCount", Attr::ReadNone, i32t, 112, &[]);
        let mut ok = self.s.m.cmp(at, pred::EQ, lanes, k32);
        let t = self.s.m.bool(true);
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self.s.op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[t]);
        let lo = self.s.m.extract(at, r, 0);
        let n = self.s.op(at, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[lo]);
        let full = self.s.m.cmp(at, pred::EQ, n, k32);
        ok = self.s.m.bin(at, Bin::And, ok, full);
        let g = self.s.op(at, "flattenedThreadIdInGroup.i32", Attr::ReadNone, i32t, 96, &[]);
        let lane = self.s.op(at, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let own = self.s.m.bin(at, Bin::And, g, k31);
        let here = self.s.m.cmp(at, pred::EQ, lane, own);
        ok = self.s.m.bin(at, Bin::And, ok, here);
        let wave = self.s.m.bin(at, Bin::LShr, g, k5);
        let first = self.s.op(at, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[wave]);
        let alike = self.s.m.cmp(at, pred::EQ, wave, first);
        ok = self.s.m.bin(at, Bin::And, ok, alike);
        let ok = self.s.op(at, "waveAllTrue", Attr::NoUnwind, self.s.t.i1, 114, &[ok]);
        // the group's one answer
        let slot = self.halves.expect("the paired form's entry dword");
        let lds = self.s.lds();
        let k = self.s.m.i32(slot);
        let ptr = self.s.m.element(at, lds, k);
        self.s.m.store(at, ptr, k0);
        self.group_sync(at);
        let no = self.s.m.select(at, ok, k0, k1);
        let ptr = self.s.m.element(at, lds, k);
        self.s.m.atomic(at, super::Rmw::Or, ptr, no);
        self.group_sync(at);
        let ptr = self.s.m.element(at, lds, k);
        let said = self.s.m.load(at, ptr);
        if matches!(self.x.native.io, super::StageIo::Cs(cs) if slot < cs.lds_bytes() / 4) {
            self.group_sync(at);
        }
        self.s.m.cmp(at, pred::EQ, said, k0)
    }

    /// The paired form's answer to "does some lane of the wave (`any`), or every lane, hold
    /// `when`", at DXIL block `at`: both lanes of every thread; a lane outside EXEC says what
    /// it says on the console (no set bit for "some lane", no objection to "every lane").
    pub(super) fn paired_wave_test(&mut self, when: Bool, at: usize, any: bool) -> Result<V, Why> {
        let [lo, hi] = self.both(|l| l.boolean(when, at))?;
        let pair = self.s.m.bin(at, if any { Bin::Or } else { Bin::And }, lo, hi);
        self.s.flags |= flags::WAVE_OPS;
        let (name, code) = if any { ("waveAnyTrue", 113) } else { ("waveAllTrue", 114) };
        Ok(self.s.op(at, name, Attr::NoUnwind, self.s.t.i1, code, &[pair]))
    }

    /// Instruction `i` of IR block `b` in the paired form, for both console lanes of the
    /// thread. What a lane does is lowered for the lower lane, then for the upper one, which
    /// keeps the console's order of the wave's instructions. Scalar code and scalar memory
    /// once: the wave's one scalar state, the same in both lanes (`mirror`). A read of other
    /// lanes once, with what both lanes hold ([`Self::paired_cross`]).
    ///
    /// LDS: what one lane writes, another may read at a later instruction, and what one reads
    /// another may overwrite. The two are two threads of the host wave, which D3D12 orders
    /// only at a group barrier; so the host wave meets before an access that follows another
    /// of which one writes, unless the program's own barrier stands between them. In a group
    /// of several console waves a meeting is no barrier every thread reaches: refused there
    /// ([`super::PAIRED_LDS_APART`]).
    pub(super) fn paired_inst(&mut self, b: usize, i: usize) -> Result<(), Stop> {
        let inst: &'a Inst = &self.x.p.blocks[b].insts[i];
        let name = inst.name();
        if matches!(inst.fields, Fields::Sop { .. } | Fields::Sopp { .. } | Fields::Smem { .. }) {
            self.inst(b, i)?;
            self.mirror(inst);
            // (a barrier that waits for the group orders every LDS access before it)
            if name == "s_barrier" && !self.fence_only.contains(&b) && super::super::group_sync() != 0 {
                self.lds_since = Some((false, false));
            }
            return Ok(());
        }
        let cross = matches!(name, "v_readlane_b32" | "v_readfirstlane_b32" | "ds_bpermute_b32");
        if cross {
            if let Err(why) = self.paired_cross(inst, b, i) {
                if matches!(inst.fields, Fields::Ds { .. }) {
                    return Err(unsupported(inst.machine.offset, why));
                }
                for &(_, w) in &inst.writes {
                    self.put_lanes(w, Val::Missing(why), Val::Missing(why));
                }
            }
            return Ok(());
        }
        if matches!(inst.fields, Fields::Ds { .. }) && !name.starts_with("ds_swizzle") && !name.contains("permute") {
            let write = !name.starts_with("ds_read");
            let meet = match self.lds_since {
                None => true,
                Some((any, wrote)) => write && any || wrote,
            };
            if meet {
                if !self.one_wave {
                    return Err(unsupported(inst.machine.offset, super::PAIRED_LDS_APART));
                }
                let c = self.cur;
                self.group_sync(c);
                self.lds_since = Some((false, false));
            }
            let (_, wrote) = self.lds_since.unwrap_or_default();
            self.lds_since = Some((true, wrote || write));
        }
        self.inst(b, i)?;
        self.swap_lanes();
        let r = self.inst(b, i);
        self.swap_lanes();
        r
    }

    /// The paired form: what `inst` wrote in the lane being lowered, for the other lane too (a
    /// scalar result, the same in both).
    fn mirror(&mut self, inst: &Inst) {
        let Some(t) = self.twin.as_deref_mut() else {
            return;
        };
        for &(_, w) in &inst.writes {
            if let Some(&v) = self.vals.get(&w) {
                t.vals.insert(w, v);
            }
        }
    }

    /// The paired form: an instruction that reads other lanes of the console wave, any of its
    /// 64: `v_readlane_b32` and `v_readfirstlane_b32` (scalar results, the same in both lanes of
    /// the thread) and `ds_bpermute_b32`. Lane n and lane n + 32 are the lower and the upper
    /// lane of host lane n. Every thread of the host wave is here (its control flow is the
    /// wave's), so every thread reads from both banks, the lower lanes' and the upper lanes',
    /// at the host lane the number names, and keeps the one its bit 5 picks: a lane outside
    /// EXEC is still there to be read.
    fn paired_cross(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        self.cross_lane()?;
        let c = self.cur;
        let (k31, k32, z) = (self.s.m.i32(31), self.s.m.i32(32), self.s.m.i32(0));
        // a lane's value from the lower or upper value of the host lane `at` (of 0..63)
        let pick = |l: &mut Self, lo: V, hi: V, at: V| {
            let host = l.s.m.bin(c, Bin::And, at, k31);
            let upper = l.s.m.bin(c, Bin::And, at, k32);
            let upper = l.s.m.cmp(c, pred::NE, upper, z);
            let (lo, hi) = (l.read_lane(lo, host), l.read_lane(hi, host));
            l.s.m.select(c, upper, hi, lo)
        };
        match inst.name() {
            "v_readlane_b32" => {
                // the lane: a scalar or a constant, the wave's
                let lane = self.src(inst, 1)?;
                let [lo, hi] = self.both(|l| l.src(inst, 0))?;
                let r = pick(self, lo, hi, lane);
                self.write_i(inst, r, None)?;
            }
            "v_readfirstlane_b32" => {
                let [(lo, in_lo), (hi, in_hi)] = self.both(|l| {
                    let x = l.src(inst, 0)?;
                    let guard = l.guard(b, i)?;
                    Ok(l.first_lane(x, guard))
                })?;
                // the first lane in EXEC is a lower one if there is one; with none, lane 0
                let hi = self.s.m.select(c, in_hi, hi, lo);
                let r = self.s.m.select(c, in_lo, lo, hi);
                self.write_i(inst, r, None)?;
            }
            _ => {
                // `ds_bpermute_b32`: each lane reads the lane its address names, which gives
                // its data, or 0 outside EXEC
                let Fields::Ds {
                    dst: Some(Operand::Vector { reg, .. }),
                    addr: Some(addr),
                    data,
                    offset0,
                    offset1,
                    ..
                } = &inst.fields
                else {
                    return Err("a permute's operands");
                };
                let data = data.first().ok_or("a permute source")?;
                let offset = u32::from(*offset0) | u32::from(*offset1) << 8;
                let [(from_lo, x_lo, g_lo), (from_hi, x_hi, g_hi)] = self.both(|l| {
                    let at = l.vgprs(inst, addr, 1)?[0];
                    let x = l.vgprs(inst, data, 1)?[0];
                    let guard = l.guard(b, i)?;
                    let (o, two) = (l.s.m.i32(offset), l.s.m.i32(2));
                    let at = l.s.m.bin(c, Bin::Add, at, o);
                    let from = l.s.m.bin(c, Bin::LShr, at, two);
                    let x = match guard {
                        Some(g) => l.s.m.select(c, g, x, z),
                        None => x,
                    };
                    Ok((from, x, guard))
                })?;
                let read = [pick(self, x_lo, x_hi, from_lo), pick(self, x_lo, x_hi, from_hi)];
                let guards = [g_lo, g_hi];
                let reg = *reg;
                self.both(|l| {
                    let n = usize::from(l.upper);
                    l.write(inst, Reg::V(reg), read[n], guards[n]);
                    Ok(())
                })?;
                return Ok(());
            }
        }
        self.mirror(inst);
        Ok(())
    }

    /// [`Self::section_inst`] in the paired form: a value over the wave's lanes takes both
    /// lanes of every thread ([`Self::paired_shape`]), a lane's own value each lane's.
    pub(super) fn paired_section_inst(&mut self, b: usize, i: usize, start: usize) {
        const UNKNOWN: Why = "a whole-wave section step the simulation does not follow";
        let inst = &self.x.p.blocks[b].insts[i];
        for &(_, w) in &inst.writes {
            let (shape, order) = match self.section_out.contains(&w) {
                true => {
                    let sec = &self.sections[&b][&start];
                    (sec.shapes.get(&w).copied(), sec.butterflies.get(&w).cloned())
                }
                false => (None, None),
            };
            let made = match shape {
                None => Err(UNKNOWN),
                Some(sh @ (Shape::Active(..) | Shape::Prefix(..) | Shape::ActiveF64(..))) => {
                    self.paired_shape(sh, order.as_deref())
                }
                Some(sh) => self.both(|l| l.shape(sh, false)),
            };
            match made {
                Ok([lo, hi]) => self.put_lanes(w, Val::I32(lo), Val::I32(hi)),
                Err(why) => self.put_lanes(w, Val::Missing(why), Val::Missing(why)),
            }
        }
    }

    /// [`Self::shape`] of a reduction or prefix in the paired form, over the console wave's 64
    /// lanes: for the lower and the upper lane of this thread. Integer and bit operations, and
    /// a float minimum or maximum, give the same in any order: a host wave operation over each
    /// half, the two results joined. A float sum rounds at every addition, so it is added in
    /// the console's order, the butterfly `order` the simulation found (lane bits 0..4 a host
    /// lane read apart, bit 5 the thread's two lanes); a sum in any other order, and a prefix
    /// of floats, are refused.
    fn paired_shape(&mut self, shape: Shape, order: Option<&[u32]>) -> Result<[V; 2], Why> {
        const ORDER: Why = "a wave sum of floats in an order the paired form does not replay";
        let (op, src, mask, ident, prefix) = match shape {
            Shape::ActiveF64(lo, hi, mask, high) => {
                let r = self.paired_sum64(lo, hi, mask, order.ok_or(ORDER)?)?;
                let r = r[usize::from(high)];
                return Ok([r, r]);
            }
            Shape::Active(op, src, mask, ident) => (op, src, mask, ident, None),
            Shape::Prefix(op, src, mask, ident, inclusive) => (op, src, mask, ident, Some(inclusive)),
            _ => return Err("a whole-wave section value of another shape"),
        };
        let &m = self.x.st.masks.get(&mask).ok_or("a wave reduction's untracked mask")?;
        let c = self.cur;
        let [lo, hi] = self.both(|l| {
            let on = l.boolean(m, c)?;
            let x = l.i32_of(src)?;
            let id = l.s.m.i32(ident);
            Ok(l.s.m.select(c, on, x, id))
        })?;
        self.s.flags |= flags::WAVE_OPS;
        match (op, prefix) {
            (Op::AddH, None) => {
                // the balanced tree over the lanes in their order (`wave::buddies`), with the
                // high half of lane 0's term
                let sum = self.butterfly(lo, hi, &[0, 1, 2, 3, 4, 5], |l, a, b| l.add_halves(a, b));
                let (high, low, zero) = (self.s.m.i32(0xFFFF_0000), self.s.m.i32(0xFFFF), self.s.m.i32(0));
                let first = self.read_lane(lo, zero);
                let first = self.s.m.bin(c, Bin::And, first, high);
                let sum = self.s.m.bin(c, Bin::And, sum, low);
                let r = self.s.m.bin(c, Bin::Or, first, sum);
                Ok([r, r])
            }
            (Op::AddF, None) => {
                let order = order.ok_or(ORDER)?;
                let add = |l: &mut Self, a: V, b: V| {
                    let (fa, fb) = (l.as_float(a), l.as_float(b));
                    let f = l.s.m.bin(l.cur, Bin::Add, fa, fb);
                    l.as_int(f)
                };
                let r = self.butterfly(lo, hi, order, add);
                Ok([r, r])
            }
            (Op::AddU, Some(inclusive)) => {
                // the upper lanes' prefix after the lower lanes' total
                let mine = self.host_prefix(lo, inclusive);
                let theirs = self.host_prefix(hi, inclusive);
                let below = self.host_op(Op::AddU, lo);
                let r = self.s.m.bin(c, Bin::Add, theirs, below);
                Ok([mine, r])
            }
            (_, Some(_)) => Err("a wave prefix of an operation other than an integer sum, in paired lanes"),
            (op, None) => {
                let float = matches!(op, Op::MinF | Op::MaxF);
                let (a, b) = if float { (self.as_float(lo), self.as_float(hi)) } else { (lo, hi) };
                let (a, b) = (self.host_op(op, a), self.host_op(op, b));
                let r = self.join_halves(op, a, b, float);
                let r = if float { self.as_int(r) } else { r };
                Ok([r, r])
            }
        }
    }

    /// The sum over the console wave of `lo` (the lower lanes' terms) and `hi` (the upper
    /// lanes'), added as the butterfly over the lane bits `order`: at each step every lane adds
    /// the partial sum of the lane whose number differs in that bit; bit 5 is the other lane
    /// of the same thread. The result, the same in every lane.
    fn butterfly(&mut self, lo: V, hi: V, order: &[u32], add: impl Fn(&mut Self, V, V) -> V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let lane = self.s.op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let (mut lo, mut hi) = (lo, hi);
        for &d in order {
            if d == 5 {
                let s = add(self, lo, hi);
                (lo, hi) = (s, s);
                continue;
            }
            let bit = self.s.m.i32(1 << d);
            let partner = self.s.m.bin(c, Bin::Xor, lane, bit);
            let (pl, ph) = (self.read_lane(lo, partner), self.read_lane(hi, partner));
            (lo, hi) = (add(self, lo, pl), add(self, hi, ph));
        }
        lo
    }

    /// `op` of `x` (a float for a float operation, else bits) over this host wave.
    fn host_op(&mut self, op: Op, x: V) -> V {
        let c = self.cur;
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        let ty = if float { self.s.t.f32 } else { self.s.t.i32 };
        let sfx = if float { "f32" } else { "i32" };
        match op {
            Op::Or | Op::And | Op::Xor => {
                let k = self.s.m.i8(match op {
                    Op::And => 0,
                    Op::Or => 1,
                    _ => 2,
                });
                self.s.op(c, "waveActiveBit.i32", Attr::NoUnwind, ty, 120, &[x, k])
            }
            _ => {
                let (code, unsigned) = match op {
                    Op::AddU => (0, 1),
                    Op::AddF | Op::AddH => (0, 0),
                    Op::MinU => (2, 1),
                    Op::MaxU => (3, 1),
                    Op::MinI | Op::MinF => (2, 0),
                    _ => (3, 0),
                };
                let (k, u) = (self.s.m.i8(code), self.s.m.i8(unsigned));
                self.s.op(c, &format!("waveActiveOp.{sfx}"), Attr::NoUnwind, ty, 119, &[x, k, u])
            }
        }
    }

    /// The integer sum of `x` over the host lanes below this one (`inclusive`: and this one).
    fn host_prefix(&mut self, x: V, inclusive: bool) -> V {
        let c = self.cur;
        let (k0, k1) = (self.s.m.i8(0), self.s.m.i8(1));
        let r = self.s.op(c, "wavePrefixOp.i32", Attr::NoUnwind, self.s.t.i32, 121, &[x, k0, k1]);
        if inclusive { self.s.m.bin(c, Bin::Add, r, x) } else { r }
    }

    /// Two results of `op` over a console wave's halves, joined into the wave's.
    fn join_halves(&mut self, op: Op, lo: V, hi: V, float: bool) -> V {
        let c = self.cur;
        let ty = if float { self.s.t.f32 } else { self.s.t.i32 };
        let sfx = if float { "f32" } else { "i32" };
        match op {
            Op::AddU => self.s.m.bin(c, Bin::Add, lo, hi),
            Op::Or => self.s.m.bin(c, Bin::Or, lo, hi),
            Op::And => self.s.m.bin(c, Bin::And, lo, hi),
            Op::Xor => self.s.m.bin(c, Bin::Xor, lo, hi),
            _ => {
                // FMax 35, FMin 36, IMax 37, IMin 38, UMax 39, UMin 40
                let code = match op {
                    Op::MaxF => 35,
                    Op::MinF => 36,
                    Op::MinU => 40,
                    Op::MaxU => 39,
                    Op::MinI => 38,
                    _ => 37,
                };
                self.s.op(c, &format!("binary.{sfx}"), Attr::ReadNone, ty, code, &[lo, hi])
            }
        }
    }

    /// The 64-bit float sum over the console wave of `f64(lo, hi)` in the lanes of `mask`
    /// (+0.0 elsewhere), added as the butterfly `order` ([`Self::butterfly`]): its low and high
    /// dwords.
    fn paired_sum64(&mut self, lo: crate::ir::Value, hi: crate::ir::Value, mask: crate::ir::Value, order: &[u32]) -> Result<[V; 2], Why> {
        let &m = self.x.st.masks.get(&mask).ok_or("a wave reduction's untracked mask")?;
        let c = self.cur;
        self.s.flags |= flags::WAVE_OPS | flags::DOUBLES;
        let f64t = self.s.m.double();
        let i32t = self.s.t.i32;
        let split_ty = self.s.m.named_struct("dx.types.splitdouble", vec![i32t, i32t]);
        // each lane's term as a double
        let [d_lo, d_hi] = self.both(|l| {
            let on = l.boolean(m, c)?;
            let (x, y, z) = (l.i32_of(lo)?, l.i32_of(hi)?, l.s.m.i32(0));
            let (x, y) = (l.s.m.select(c, on, x, z), l.s.m.select(c, on, y, z));
            Ok(l.s.op(c, "makeDouble.f64", Attr::ReadNone, f64t, 101, &[x, y]))
        })?;
        let lane = self.s.op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        // a double read from another host lane, dword by dword
        let read = |l: &mut Self, d: V, at: V| {
            let s = l.s.op(c, "splitDouble.f64", Attr::ReadNone, split_ty, 102, &[d]);
            let (x, y) = (l.s.m.extract(c, s, 0), l.s.m.extract(c, s, 1));
            let (x, y) = (l.read_lane(x, at), l.read_lane(y, at));
            l.s.op(c, "makeDouble.f64", Attr::ReadNone, f64t, 101, &[x, y])
        };
        let (mut a, mut b) = (d_lo, d_hi);
        for &d in order {
            if d == 5 {
                let s = self.s.m.bin(c, Bin::Add, a, b);
                (a, b) = (s, s);
                continue;
            }
            let bit = self.s.m.i32(1 << d);
            let partner = self.s.m.bin(c, Bin::Xor, lane, bit);
            let (pa, pb) = (read(self, a, partner), read(self, b, partner));
            (a, b) = (self.s.m.bin(c, Bin::Add, a, pa), self.s.m.bin(c, Bin::Add, b, pb));
        }
        let s = self.s.op(c, "splitDouble.f64", Attr::ReadNone, split_ty, 102, &[a]);
        Ok([self.s.m.extract(c, s, 0), self.s.m.extract(c, s, 1)])
    }
}
