//! A sampler that control flow chooses between two immediate ones.
//!
//! A conditional choice between two immediate samplers can reach the translator as a scalar
//! compare and a branch around two moves of the samplers' first word, joined at a block that
//! sets the words they share and samples. The descriptor is then a
//! [`Source::Phi`] whose every source is a [`Source::Constant`].
//!
//! An immediate sampler is a static sampler of the host, one binding each: no heap index
//! names it, so there is no number a phi could pick (`Lower::descriptor_phi`), and DXIL
//! takes no phi or select of handles. So the join picks a number per edge, and the
//! sample is made once with each sampler where the console's one sample stands, in the
//! same block with the same operands; each channel is the result of the sampler whose
//! number the lane brought. No sample moves across a branch. An implicit one takes
//! explicit gradients, made where its coordinates are written (below).
//!
//! Only one form, whose choice can be proved the console's, is lowered so; every other is
//! refused, each by its own reason. The form:
//!
//! - **The choice is the wave's.** SGPRs hold one descriptor for the whole wave, the one
//!   of the edge the wave came by, and a lane brings the number of its own edge: the two
//!   are the same only where every lane takes the choosing branch alike. That is proved
//!   for one selector ([`Lower::flag_branch`]): `s_cbranch_scc1` on the SCC of
//!   `s_cmp_lg_u32` of a constant and a word an `s_buffer_load_dword*` reads at a fixed
//!   offset of a constant buffer whose descriptor comes from a fixed table entry. Each
//!   of those values is written by one instruction with no input that depends on the
//!   path a lane took or on the pass of a loop it is in, so every lane computes the same
//!   SCC. An SCC of anything else is not taken as the wave's on its word: a scalar an
//!   earlier branch on a lane mask left differs between the lanes as lowered.
//! - **The buffer's table address is a root parameter's own value** as the lowering
//!   reads it ([`Lower::own`]). The descriptor tracker follows an address through a lane
//!   of a VGPR and back (`v_writelane_b32`, `v_readlane_b32`) and names the parameter;
//!   the heap index is made of the value read back, by a lane read of a lane that may
//!   not run there. Only an address the hardware set up, moved by `s_mov_b32` if at
//!   all, is the same number in every lane as emitted.
//! - **One branch chooses.** The join's two predecessors are the branch's two targets,
//!   each entered from the branch alone and left by a jump to the join.
//! - **The sample stands in the join's block**: a lane that samples has just come in by
//!   its edge (after a loop its lanes leave one by one, a lane would still hold the
//!   number of its own last pass).
//! - **The operation's lowering takes the sampler** (`operation` below): `image_sample` of
//!   a pixel program and `image_sample_lz`, of a 2D view, with 32-bit normalised
//!   coordinates and 32-bit results, under no lane mask. A gather at a named level
//!   reads texels without the sampler; outside a pixel program `image_sample` reads
//!   level 0 by a shortcut; no other operation is admitted.
//! - **The two samplers are one clamped and one wrapped, alike otherwise**
//!   ([`Lower::pair`]). Both samples run in every lane that reaches them, so the one a
//!   lane does not take must be as legal for the view as its own (the same filters)
//!   and select the same level (the same anisotropy, bias and level clamps). Neither
//!   forces unnormalised coordinates, which the host's sampler cannot say and which
//!   would want coordinates of their own; neither is a sampler a comparison sample of
//!   the program uses too, which makes the one static sampler of those words a
//!   comparison sampler.
//! - **The lanes that reach the sample are the lanes the console samples in**
//!   ([`Lower::lanes_sampled`]). Structurisation lowers a branch on a lane mask as each
//!   lane's own, which is the console's only for some forms; the choice admits those
//!   forms alone and refuses every other, without asking why it would differ:
//!   - the sample's EXEC is one of the list ([`Lower::run_of`]): the program's own, its
//!     whole quads (`s_wqm` of it), the program's again after whole-quad mode
//!     (`s_and` of the two), every lane (`s_mov` of -1, as mesh programs start), or a
//!     section of one of these (`v_cmpx_*`, or
//!     `s_and_saveexec` of a `v_cmp_*` mask), through plain copies (`s_mov`, the EXEC an
//!     `s_and_saveexec` saves). A join (a phi of EXEC) only of copies of one and the same
//!     mask on every way, which is that mask ([`Lower::copied`]); not of two masks, though
//!     they be equal as the program runs. No other write of EXEC;
//!   - every branch the sample's block hangs on ([`Lower::hangs_on`]) is a flag of the one
//!     proved form ([`Lower::flag_branch`]), or the `s_cbranch_execz` of a mask the
//!     sample's EXEC was made from, entered by its "some lane" side, in no loop with the
//!     sample. Each lane of that EXEC goes in by its own bit, as the console's wave does
//!     when it has a lane. What a block hangs on is asked of the program as lowered, with
//!     the lowering's own ways out: a plain jump back to a loop header also leaves the
//!     program at the iteration limit, and decides like a branch; a program in which a
//!     block has no way to the end at all is refused.
//!
//!   A plain sample behind another branch has the same lanes wrong: that is the
//!   lowering of the branch, which this module leaves as it is and keeps out of.
//! - **An implicit sample takes explicit gradients made where the quad is whole**
//!   ([`Lower::coordinates`]). The console differences the coordinate registers of the
//!   quad's lanes (lane 1 less lane 0 across, lane 2 less lane 0 down, for all four),
//!   whichever lanes run the sample. The host's implicit derivatives are of the lanes that
//!   run it, which a branch around the choice changes. So the gradients are not proved
//!   alike, they are made: each coordinate is differenced with `QuadReadLaneAt` (which
//!   keeps the helper lanes; `derivCoarse*` may take either pair of a quad) right after
//!   the instruction that writes it, and the sample is `SampleGrad` with those
//!   differences. That is exact where every lane of the quad holds the coordinate as the
//!   console's does when the instruction runs, which is proved by construction, for a
//!   short list of values, and refused for anything else:
//!   - interpolated attributes, `v_interp_p2*` completing one `v_interp_p1*` with the
//!     hardware's barycentrics, of a row neither flat nor of halves;
//!   - literals and inline constants, scalars the hardware set up (user data), words of
//!     a constant buffer as [`Lower::flag_branch`] proves its flag's, and the scalar
//!     arithmetic in `SALU` of them;
//!   - the per-lane arithmetic in `VALU` (no DPP, no SDWA) of values of the list, and
//!     `v_cndmask_b32` by a mask that is the whole result of one `v_cmp_*` of values of
//!     the list ([`Lower::compared`]): each lane selects by its own compare. That is a
//!     per-lane value like any other here: no lane of a quad decides by it whether it
//!     runs or which sampler it has, and every lane computes it, so its differences at
//!     the write are what the console's implicit derivatives see. A mask is of the list
//!     only so, never as a number or a branch.
//!
//!   Each instruction of these runs in a block every lane runs (it hangs on no branch
//!   but proved flags, and is in no loop), and each vector one under whole quads: EXEC
//!   is `s_wqm` of the program's own through plain copies, so the console's helper lanes
//!   compute it as the host's do. No lane read, mask, VCC, SCC or join is of the list,
//!   whatever it would come to; nor EXEC as an operand, either dword, `exec_hi` of a wave32
//!   program too: a vector instruction reads EXEC only as its lanes, and a select's mask
//!   is in VCC or an SGPR below it. The differences are calls of a wave operation, which the
//!   clean-up passes reuse only within their block (`bitcode::opt`'s `Effect::Lanes`).
//!   (`llvm/lib/Target/AMDGPU/SIWholeQuadMode.cpp` at `TII->isWQM`: samples "just require all inputs of a quad to have been computed for derivatives".)
//!
//!   What it is exact for: `image_sample` of a 2D view with normalised coordinates. The
//!   host's sampler scales gradients by the view's size, applies the sampler's bias,
//!   level clamps and anisotropy to them as it does to implicit ones, so one gradient
//!   pair gives one level and footprint. Not admitted (`operation` refuses them):
//!   `image_sample_b`, whose bias the gradient form has no operand for (scaling the
//!   gradients by 2^bias moves the level by a rounded amount, not by the bias);
//!   `image_sample_c`, whose gradient form needs shader model 6.8; cubes, whose
//!   console gradients are of face coordinates; arrays and 3D views, and the `_l`,
//!   `_cl`, `_o` and `_d` forms, which are refused as unproved.
//!
//! What it costs: one more sample at each such site, and three `QuadReadLaneAt` and two
//! subtractions per coordinate at its write; no scratch memory, no barrier.

use std::collections::{HashMap, HashSet};

use super::{Atom, Attr, Bin, LaneExit, Lower, Site, Source, V, Why, flags, pred};
use crate::gfx10::{Fields, NULL, Operand};
use crate::io::Interp;
use crate::ir::{EXEC, Exit, Inst, Program, Reg, VCC, Value};
use crate::native::{ResourceType, RootType, StageIo};

/// A sampler chosen by a branch the console's wave takes as one and its lanes each for
/// themselves.
const LANES: Why = "an immediate sampler chosen by a per-lane branch";

/// A sampler chosen by a scalar branch whose condition is not the compare of a constant
/// buffer's word with a constant ([`Lower::flag_branch`]).
const UNPROVED: Why =
    "an immediate sampler chosen by a scalar condition not proved the same in every lane";

/// A sampler chosen by more than one branch around two moves.
const SHAPE: Why = "an immediate sampler chosen by other than one branch between two blocks";

/// A sampler chosen at a join whose block the sample is not in.
const ELSEWHERE: Why = "an immediate sampler chosen in another block than its sample";

const GATHER: Why = "a gather with an immediate sampler chosen by control flow";
const LEVEL: Why = "a level query with an immediate sampler chosen by control flow";
const OPERATION: Why = "an immediate sampler chosen by control flow for other than image_sample of a pixel program or image_sample_lz";
const VIEW: Why = "an immediate sampler chosen by control flow for other than a 2D view with 32-bit normalised coordinates and results";
const MASKED: Why = "an immediate sampler chosen by control flow for a sample under a lane mask";
const PAIR: Why = "immediate samplers chosen by control flow that are not one clamped and one wrapped and alike otherwise";
const UNNORMALISED: Why =
    "an immediate sampler chosen by control flow that forces unnormalised coordinates";
const COMPARED: Why =
    "an immediate sampler chosen by control flow that a comparison sample uses too";

/// A flag of a constant buffer whose table address the lowering reads otherwise than as
/// a root parameter's own value ([`Lower::own`]).
const ADDRESS: Why = "an immediate sampler chosen by a flag of a constant buffer whose table address is not a root parameter's own value";

/// A sample behind a branch that is neither a proved flag nor a section of its EXEC
/// ([`Lower::lanes_sampled`]).
const BEHIND: Why = "an immediate sampler chosen by control flow behind a branch neither a proved flag nor the section of its sample's EXEC";
/// A program with a block from which no way leads to its end as lowered ([`Lower::certain`]).
const ENDLESS: Why =
    "an immediate sampler chosen by control flow in a program with a loop that has no way out";
/// A sample under an EXEC not of the list ([`Lower::run_of`]).
const LANE_SET: Why = "an immediate sampler chosen by control flow for a sample under an EXEC not proved the program's, its whole quads' or a section of them";

/// An implicit sample at coordinates not of the list ([`Lower::coordinates`]).
const UNWHOLE: Why = "an immediate sampler chosen by control flow for an implicit sample at coordinates computed outside whole-quad mode";
const COMPUTED: Why = "an immediate sampler chosen by control flow for an implicit sample at coordinates not every lane of the quad computes";
const JOINED: Why = "an immediate sampler chosen by control flow for an implicit sample at coordinates joined from different paths";
const UNLISTED: Why = "an immediate sampler chosen by control flow for an implicit sample at coordinates made of other than interpolated attributes, constants, constant buffer words and per-lane arithmetic";
const UNINTERPOLATED: Why = "an immediate sampler chosen by control flow for an implicit sample at an attribute flat, of halves, or not interpolated by one pair";
const COORDINATES: Why = "an immediate sampler chosen by control flow for an implicit sample of other than two coordinate registers";
/// The differences of an admitted coordinate were not made where it is written (its value
/// had no host value there).
const UNDIFFERENCED: Why = "an immediate sampler chosen by control flow for an implicit sample whose coordinates were not differenced where they are written";

/// The scalar arithmetic of the list: of values of the list, the same in every lane.
const SALU: [&str; 18] = [
    "s_mov_b32", "s_movk_i32", "s_add_u32", "s_add_i32", "s_sub_u32", "s_sub_i32", "s_mul_i32",
    "s_lshl_b32", "s_lshr_b32", "s_ashr_i32", "s_and_b32", "s_or_b32", "s_xor_b32", "s_min_u32",
    "s_max_u32", "s_min_i32", "s_max_i32", "s_bfe_u32",
];

/// The per-lane arithmetic of the list: each lane's result is of its own operands alone.
const VALU: [&str; 52] = [
    "v_mov_b32", "v_add_f32", "v_sub_f32", "v_subrev_f32", "v_mul_f32", "v_mul_legacy_f32",
    "v_fma_f32", "v_fmac_f32", "v_mac_f32", "v_mad_f32", "v_fmaak_f32", "v_fmamk_f32",
    "v_madak_f32", "v_madmk_f32", "v_min_f32", "v_max_f32", "v_med3_f32", "v_min3_f32",
    "v_max3_f32", "v_fract_f32", "v_floor_f32", "v_ceil_f32", "v_trunc_f32", "v_rndne_f32",
    "v_rcp_f32", "v_rsq_f32", "v_sqrt_f32", "v_exp_f32", "v_log_f32", "v_sin_f32", "v_cos_f32",
    "v_ldexp_f32", "v_cvt_f32_i32", "v_cvt_f32_u32", "v_cvt_i32_f32", "v_cvt_u32_f32",
    "v_add_nc_u32", "v_sub_nc_u32", "v_subrev_nc_u32", "v_and_b32", "v_or_b32", "v_xor_b32",
    "v_not_b32", "v_lshlrev_b32", "v_lshrrev_b32", "v_ashrrev_i32", "v_bfe_u32", "v_bfe_i32",
    "v_bfi_b32", "v_mul_lo_u32", "v_mul_u32_u24", "v_mad_u32_u24",
];

/// The three address modes of a sampler's first word, and all three clamped (wrapped: 0).
const MODES: u32 = 0x1FF;
const CLAMPED: u32 = 0x92;
/// The bit of a sampler's first word that forces unnormalised coordinates.
const FORCE_UNNORMALISED: u32 = 1 << 15;

/// The immediate samplers a sample's sampler is chosen among.
pub(super) struct Chosen {
    /// The handle of the first, which the sample is made with.
    pub first: V,
    /// Of each other one: whether it is the lane's, and its handle.
    others: Vec<(V, V)>,
    /// Of an implicit sample: the differences of each coordinate across and down its
    /// quad, the gradients it is made with (`SampleGrad`).
    pub gradients: Option<Vec<[V; 2]>>,
}

/// The differences of the coordinates of implicit samples with a chosen sampler.
#[derive(Default)]
pub(super) struct Differences {
    /// The coordinates, by the instruction writing them: differenced right after it.
    at: HashMap<Site, Vec<Value>>,
    /// Their differences, across and down the quad, by coordinate.
    made: HashMap<Value, [V; 2]>,
}

#[cfg(test)]
thread_local! {
    /// On this thread: the coordinates differenced, and the chosen samples given gradients.
    pub(crate) static MADE: std::cell::Cell<(usize, usize)> =
        const { std::cell::Cell::new((0, 0)) };
}

/// The form of a choice [`Lower::admit`] proved: its join and the two blocks before it,
/// and the coordinates of an implicit sample (none for a named level).
struct Admitted {
    join: usize,
    arms: [usize; 2],
    coordinates: Vec<Value>,
}

/// What the lanes of an EXEC of the list are ([`Lower::run_of`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Run {
    /// The program's own (helper lanes out).
    Entry,
    /// Its whole quads: `s_wqm` of it.
    Whole,
    /// A section of one of these: the lanes of a compare.
    Section,
    /// Every lane of the wave: `s_mov` of -1, as mesh programs start.
    All,
}

/// Whether the lowering of the image instruction `inst`, of a pixel program or not, makes
/// its one sample with the sampler handle and can so be made with each of two.
fn operation(inst: &Inst, pixel: bool) -> Result<(), Why> {
    let name = inst.name();
    if name == "image_get_lod" {
        return Err(LEVEL);
    }
    if name.starts_with("image_gather4") {
        return Err(GATHER);
    }
    if !(name == "image_sample" && pixel || name == "image_sample_lz") {
        return Err(OPERATION);
    }
    match inst.fields {
        Fields::Image { dim: 1, unorm: false, a16: false, d16: false, .. } => Ok(()),
        _ => Err(VIEW),
    }
}

/// The blocks reached from `from` (themselves too) by the steps `next` gives, `wall`
/// never entered.
fn reached(
    n: usize,
    from: impl IntoIterator<Item = usize>,
    wall: Option<usize>,
    next: impl Fn(usize) -> Vec<usize>,
) -> Vec<bool> {
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = from.into_iter().collect();
    while let Some(b) = stack.pop() {
        if Some(b) == wall || std::mem::replace(&mut seen[b], true) {
            continue;
        }
        stack.extend(next(b));
    }
    seen
}

fn has(bits: &[u64], b: usize) -> bool {
    bits[b / 64] >> (b % 64) & 1 == 1
}

/// Whether each of the explicit sources `src` of a vector instruction is one a value of the
/// list is read from: a VGPR, an SGPR below VCC, a literal, or an inline constant that is
/// no special source (251-254: vccz, execz, scc, lds_direct). Never a mask as a number:
/// VCC, either dword of EXEC (`exec_hi` of a wave32 program too, which the lowering takes
/// as a plain scalar), M0, `null` and the trap registers are none (Neon's seventh review).
fn sources(src: &[Operand]) -> bool {
    src.iter().all(|s| match *s {
        Operand::Vector { .. } | Operand::Literal(_) => true,
        Operand::Scalar { code, dwords } => u16::from(code) + u16::from(dwords) <= u16::from(VCC),
        Operand::Const(c) => !(251..=254).contains(&c),
        _ => false,
    })
}

/// What a scalar is as the lowering reads it ([`Lower::own`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Own {
    /// A register the hardware set up: a root parameter's.
    Parameter,
    Constant,
}

impl<'a> Lower<'a> {
    /// The samplers the sampler `src` of the sample at `site` is chosen among, if control
    /// flow chooses it among immediate ones: their handles (with `props`) in the current
    /// block, which of them is the lane's, and an implicit sample's gradients. Refused
    /// unless the choice has the one form the module's text describes.
    pub(super) fn immediate_choice(
        &mut self,
        site: Site,
        src: &Source,
        props: (u32, u32),
    ) -> Result<Option<Chosen>, Why> {
        const WHY: Why = "a descriptor chosen by control flow";
        let Some(Admitted { join, arms, coordinates }) = self.admit(site, src)? else {
            return Ok(None);
        };
        let Source::Phi { args, .. } = src else {
            return Ok(None);
        };
        let gradients = if coordinates.is_empty() {
            None
        } else {
            let made: Option<Vec<[V; 2]>> =
                coordinates.iter().map(|v| self.differences.made.get(v).copied()).collect();
            Some(made.ok_or(UNDIFFERENCED)?)
        };
        #[cfg(test)]
        if gradients.is_some() {
            MADE.with(|m| m.set((m.get().0, m.get().1 + 1)));
        }
        // The number of its sampler on each edge, picked at the join.
        let at_join = self.block[join].ok_or(WHY)?;
        let mut incoming = Vec::new();
        for (k, &q) in arms.iter().enumerate() {
            let tail = self.tail[q].ok_or(WHY)?;
            let at = self.edges.get(&(q, join, false)).copied().unwrap_or(tail);
            incoming.push((self.s.m.i32(k as u32), at));
        }
        let first = self.handle(site, &args[0], true, props)?;
        let number = self.s.m.phi(at_join, self.s.t.i32);
        self.s.m.set_phi(number, incoming);
        let second = self.handle(site, &args[1], true, props)?;
        let k = self.s.m.i32(1);
        let is = self.s.m.cmp(self.cur, pred::EQ, number, k);
        Ok(Some(Chosen { first, others: vec![(is, second)], gradients }))
    }

    /// The form of the choice of the sampler `src` of the sample at `site`, if control flow
    /// chooses it among immediate ones and the choice has the one form the module's text
    /// describes; refused otherwise. Emits nothing.
    fn admit(&self, site: Site, src: &Source) -> Result<Option<Admitted>, Why> {
        let Source::Phi { block, args } = src else {
            return Ok(None);
        };
        if !args.iter().all(|a| matches!(a, Source::Constant(_))) {
            return Ok(None);
        }
        let (join, p, st) = (*block, self.x.p, self.x.st);
        let inst = &p.blocks[site.0].insts[site.1];
        operation(inst, self.s.pixel)?;
        if st.guards[site.0][site.1].is_some() {
            return Err(MASKED);
        }
        if site.0 != join {
            return Err(ELSEWHERE);
        }
        // One branch between two blocks, each of which brings one sampler.
        let &[one, other] = &p.blocks[join].preds[..] else {
            return Err(SHAPE);
        };
        let &[top] = &p.blocks[one].preds[..] else {
            return Err(SHAPE);
        };
        let arm = |q: usize| p.blocks[q].preds[..] == [top] && p.blocks[q].exit == Exit::Jump(join);
        let branches = matches!(
            p.blocks[top].exit,
            Exit::Branch { taken, not_taken } if [taken, not_taken] == [one, other] || [taken, not_taken] == [other, one]
        );
        if one == other || !arm(one) || !arm(other) || !branches {
            return Err(SHAPE);
        }
        self.pair(&args[0], &args[1])?;
        self.flag_branch(top)?;
        let certain = self.certain()?;
        self.lanes_sampled(site, &certain)?;
        let coordinates = if inst.name() == "image_sample" {
            self.coordinates(site, &certain)?
        } else {
            Vec::new() // a named level: no derivatives
        };
        Ok(Some(Admitted { join, arms: [one, other], coordinates }))
    }

    /// Before the program is lowered: where each coordinate of an admitted implicit sample
    /// is written, to be differenced there ([`Self::differentiate`]).
    pub(super) fn plan_differences(&mut self) {
        if !self.s.pixel {
            return;
        }
        let mut at: HashMap<Site, Vec<Value>> = HashMap::new();
        for (&(site, sampler), &src) in &self.descriptors {
            if !sampler {
                continue;
            }
            let Ok(Some(admitted)) = self.admit(site, src) else {
                continue;
            };
            for v in admitted.coordinates {
                if let Some((written, _)) = self.writer(v) {
                    let list = at.entry(written).or_default();
                    if !list.contains(&v) {
                        list.push(v);
                    }
                }
            }
        }
        // In the order the instruction writes them, whatever order the map gave.
        for list in at.values_mut() {
            list.sort_unstable();
        }
        self.differences.at = at;
    }

    /// After the instruction at `(b, i)` is lowered: the differences across and down each
    /// quad of each coordinate it writes ([`Self::plan_differences`]), as the console's
    /// sampler takes them (lane 1 less lane 0, lane 2 less lane 0), by quad reads that
    /// keep the helper lanes. Every lane of the quad runs here (the module's text).
    pub(super) fn differentiate(&mut self, b: usize, i: usize) {
        let Some(values) = self.differences.at.get(&(b, i)).cloned() else {
            return;
        };
        for v in values {
            let Ok(x) = self.float_of(v) else {
                continue; // no host value: the sample is refused (`UNDIFFERENCED`)
            };
            self.s.flags |= flags::WAVE_OPS;
            let (c, f32t) = (self.cur, self.s.t.f32);
            let mut lane = [x; 3];
            for (n, slot) in lane.iter_mut().enumerate() {
                let k = self.s.m.i32(n as u32);
                *slot = self.s.op(c, "quadReadLaneAt.f32", Attr::NoUnwind, f32t, 122, &[x, k]);
            }
            let across = self.s.m.bin(c, Bin::Sub, lane[1], lane[0]);
            let down = self.s.m.bin(c, Bin::Sub, lane[2], lane[0]);
            self.differences.made.insert(v, [across, down]);
            #[cfg(test)]
            MADE.with(|m| m.set((m.get().0 + 1, m.get().1)));
        }
    }

    /// Whether the two immediate samplers `a` and `b` are one clamped and one wrapped and
    /// alike otherwise, neither forcing unnormalised coordinates, neither the sampler of
    /// a comparison sample anywhere in the program.
    fn pair(&self, a: &Source, b: &Source) -> Result<(), Why> {
        let words = |s: &Source| match s {
            Source::Constant(w) => w.get(..4).and_then(|w| <[u32; 4]>::try_from(w).ok()),
            _ => None,
        };
        let (Some(a), Some(b)) = (words(a), words(b)) else {
            return Err("a sampler constant");
        };
        if (a[0] | b[0]) & FORCE_UNNORMALISED != 0 {
            return Err(UNNORMALISED);
        }
        let modes = [a[0] & MODES, b[0] & MODES];
        if !(modes == [CLAMPED, 0] || modes == [0, CLAMPED])
            || (a[0] ^ b[0]) & !MODES != 0
            || a[1..] != b[1..]
        {
            return Err(PAIR);
        }
        let p = self.x.p;
        let compared = self.descriptors.iter().any(|(&((block, i), sampler), src)| {
            let name = p.blocks[block].insts[i].name();
            sampler
                && (name.starts_with("image_sample_c") || name.starts_with("image_gather4_c"))
                && src.leaves().into_iter().any(|leaf| [Some(a), Some(b)].contains(&words(leaf)))
        });
        if compared {
            return Err(COMPARED);
        }
        Ok(())
    }

    /// Whether every lane takes the branch that closes `top` alike, by the one proof
    /// there is: the branch is `s_cbranch_scc1` on the SCC of an `s_cmp_lg_u32` of a
    /// constant and an SGPR an `s_buffer_load_dword*` wrote, which reads at a fixed
    /// offset (no register offset) a constant buffer whose descriptor one load of a
    /// fixed entry of a descriptor table gave. Each value is one instruction's (no join
    /// of paths), and the address has no computed term: whichever lane runs them, on
    /// whichever pass, reads the same word of the draw's buffer.
    fn flag_branch(&self, top: usize) -> Result<(), Why> {
        let (p, st) = (self.x.p, self.x.st);
        let written = |v: Value| {
            p.blocks.iter().enumerate().find_map(|(b, block)| {
                let i = block.insts.iter().position(|x| x.writes.iter().any(|w| w.1 == v))?;
                Some(((b, i), &block.insts[i]))
            })
        };
        let branch = p.blocks[top].insts.last().ok_or(SHAPE)?;
        let scc = match branch.name() {
            "s_cbranch_scc1" => branch.read(Reg::Scc).ok_or(UNPROVED)?,
            "s_cbranch_scc0" => return Err(UNPROVED),
            _ => return Err(LANES),
        };
        // As one lane runs it: that SCC and nothing else.
        let LaneExit::Branch { when, .. } = st.exits[top] else {
            return Err(UNPROVED);
        };
        if st.wide.contains(&top) || st.bools.support(when)[..] != [Atom::Uniform(scc)] {
            return Err(UNPROVED);
        }
        let (_, compare) = written(scc).ok_or(UNPROVED)?;
        let Fields::Sop { src, .. } = &compare.fields else {
            return Err(UNPROVED);
        };
        let (&[Operand::Scalar { code, dwords: 1 }, constant], "s_cmp_lg_u32") =
            (&src[..], compare.name())
        else {
            return Err(UNPROVED);
        };
        let constant = match constant {
            Operand::Const(c) => crate::resource::inline(c).is_some(),
            Operand::Literal(_) => true,
            _ => false,
        };
        let flag = compare.read(Reg::S(code)).filter(|_| constant).ok_or(UNPROVED)?;
        let (site, load) = written(flag).ok_or(UNPROVED)?;
        self.buffer_word(site, load)
    }

    /// Whether the instruction `load` at `site` reads a word of the draw's one constant
    /// buffer at a fixed place: `s_buffer_load_dword*` at a fixed offset (no register
    /// offset) of a constant buffer whose descriptor one load of a fixed entry of a
    /// descriptor table gave, at a root parameter's own address ([`Self::own`]).
    fn buffer_word(&self, site: Site, load: &Inst) -> Result<(), Why> {
        let p = self.x.p;
        let fixed = matches!(
            load.fields,
            Fields::Smem { soffset: Operand::Scalar { code: NULL, .. }, .. }
        );
        if !load.name().starts_with("s_buffer_load_dword") || !fixed {
            return Err(UNPROVED);
        }
        // The buffer: a constant buffer of a fixed table entry.
        let Some(&buffer) = self.descriptors.get(&(site, false)) else {
            return Err(UNPROVED);
        };
        let Source::Loaded { load: entry, .. } = *buffer else {
            return Err(UNPROVED);
        };
        let constant_buffer = self
            .used_kind(buffer)
            .is_some_and(|u| u.kind == ResourceType::Cbv && u.root == RootType::Table);
        let at = self.scalar.get(&entry).ok_or(UNPROVED)?;
        if !constant_buffer
            || at.dynamic.is_some()
            || at.lane.is_some()
            || self.x.res.waterfall.loads.contains(&entry)
        {
            return Err(UNPROVED);
        }
        // The address the heap index is made of (`Lower::heap_index`: the low dword of
        // the table load's base, and its register offset if it has one).
        let table = &p.blocks[entry.0].insts[entry.1];
        let Fields::Smem { base: Operand::Scalar { code, .. }, soffset, .. } = table.fields else {
            return Err(ADDRESS);
        };
        let own = |code: u8| table.read(Reg::S(code)).and_then(|v| self.own(v));
        let offset = match soffset {
            Operand::Scalar { code: NULL, .. } => true,
            Operand::Scalar { code, .. } => own(code) == Some(Own::Constant),
            _ => false,
        };
        if own(code) != Some(Own::Parameter) || !offset {
            return Err(ADDRESS);
        }
        Ok(())
    }

    /// What the scalar `v` is as the lowering reads it, if every lane reads the same
    /// number whatever runs it: a register the hardware set up, or a constant, moved by
    /// `s_mov_b32` if at all. Not a value a lane read wrote (`v_readlane_b32`, which the
    /// descriptor tracker follows back to what was parked), nor a join.
    fn own(&self, mut v: Value) -> Option<Own> {
        let p = self.x.p;
        loop {
            if p.inputs.iter().any(|&(reg, x)| x == v && matches!(reg, Reg::S(code) if code < VCC)) {
                return Some(Own::Parameter);
            }
            let block = &p.blocks[*self.def_block.get(&v)?];
            let inst = block.insts.iter().find(|i| i.writes.iter().any(|w| w.1 == v))?;
            let Fields::Sop { src, .. } = &inst.fields else {
                return None;
            };
            match (inst.name(), &src[..]) {
                ("s_movk_i32", _) | ("s_mov_b32", [Operand::Literal(_)]) => {
                    return Some(Own::Constant);
                }
                ("s_mov_b32", [Operand::Const(c)]) => {
                    return crate::resource::inline(*c).map(|_| Own::Constant);
                }
                ("s_mov_b32", [Operand::Scalar { code, .. }]) => v = inst.read(Reg::S(*code))?,
                _ => return None,
            }
        }
    }

    /// The instruction that writes `v`, and where: none for a phi or a register the
    /// hardware set up.
    fn writer(&self, v: Value) -> Option<(Site, &'a Inst)> {
        if self.phi_at.contains_key(&v) {
            return None;
        }
        let p: &'a Program = self.x.p;
        let b = *self.def_block.get(&v)?;
        let i = p.blocks[b].insts.iter().position(|x| x.writes.iter().any(|w| w.1 == v))?;
        Some(((b, i), &p.blocks[b].insts[i]))
    }

    /// EXEC as `inst` reads it: the low dword, and the high one of a wave64 program.
    fn exec_read(&self, inst: &Inst) -> Option<(Value, Option<Value>)> {
        let lo = inst.read(Reg::S(EXEC))?;
        let hi = if self.x.p.wave64 { Some(inst.read(Reg::S(EXEC + 1))?) } else { None };
        Some((lo, hi))
    }

    /// What the lanes of the mask `(lo, hi)` are, if it is an EXEC of the list (the
    /// module's text); each mask it is made of, itself first, is pushed to `made_of`.
    /// Nothing else is followed: a phi, a write of EXEC by any other instruction, an
    /// SGPR the hardware set up, all end it.
    fn run_of(
        &self,
        (lo, hi): (Value, Option<Value>),
        made_of: &mut Vec<Value>,
        depth: u32,
    ) -> Option<Run> {
        let p = self.x.p;
        if depth > 64 {
            return None;
        }
        made_of.push(lo);
        if self.phi_at.contains_key(&lo) {
            // A join of copies of one mask on every way is that mask; any other is not.
            let root = |v: Value| self.copied(v, &mut Vec::new(), 0).filter(|&r| r != v);
            let low = root(lo)?;
            let high = match hi {
                Some(h) => Some(root(h)?),
                None => None,
            };
            return self.run_of((low, high), made_of, depth + 1);
        }
        let input = |v: Value, code: u8| p.inputs.iter().any(|&(r, x)| x == v && r == Reg::S(code));
        if input(lo, EXEC) && hi.is_none_or(|h| input(h, EXEC + 1)) {
            return Some(Run::Entry);
        }
        let (_, inst) = self.writer(lo)?;
        if hi.is_some_and(|h| !inst.writes.iter().any(|w| w.1 == h)) {
            return None;
        }
        let &(reg, _) = inst.writes.iter().find(|w| w.1 == lo)?;
        let to_exec = reg == Reg::S(EXEC);
        let source = |k: usize| {
            let Fields::Sop { src, .. } = &inst.fields else {
                return None;
            };
            let Operand::Scalar { code, .. } = *src.get(k)? else {
                return None;
            };
            let low = inst.read(Reg::S(code))?;
            let high = if p.wave64 { Some(inst.read(Reg::S(code + 1))?) } else { None };
            Some((low, high))
        };
        let next = |m: (Value, Option<Value>), made_of: &mut Vec<Value>| {
            self.run_of(m, made_of, depth + 1)
        };
        let name = inst.name();
        if name.starts_with("v_cmpx_") && to_exec {
            next(self.exec_read(inst)?, made_of)?;
            return Some(Run::Section);
        }
        let width = if p.wave64 { "_b64" } else { "_b32" };
        let every = || match &inst.fields {
            Fields::Sop { src, .. } => matches!(src[..], [Operand::Const(193)]),
            _ => false,
        };
        match name.strip_suffix(width)? {
            "s_mov" if every() => Some(Run::All),
            "s_mov" => next(source(0)?, made_of),
            "s_wqm" => match next(source(0)?, made_of)? {
                Run::Entry | Run::Whole => Some(Run::Whole),
                Run::Section | Run::All => None,
            },
            // The EXEC it saves: the one it reads.
            "s_and_saveexec" if !to_exec => next(self.exec_read(inst)?, made_of),
            "s_and_saveexec" => {
                let (mask, _) = source(0)?;
                let (_, compare) = self.writer(mask)?;
                if !compare.name().starts_with("v_cmp_") {
                    return None;
                }
                next(self.exec_read(inst)?, made_of)?;
                Some(Run::Section)
            }
            // Whole-quad mode ended: the whole quads and the program's own lanes.
            "s_and" => match (next(source(0)?, made_of)?, next(source(1)?, made_of)?) {
                (Run::Whole, Run::Whole) => Some(Run::Whole),
                (Run::Entry | Run::Whole, Run::Entry | Run::Whole) => Some(Run::Entry),
                _ => None,
            },
            _ => None,
        }
    }

    /// The one value the mask dword `v` is a copy of: through `s_mov` and the EXEC an
    /// `s_and_saveexec` saves, and through a phi whose every input (but the phi itself,
    /// round a loop) is a copy of that same value. `None` for a phi of two values, though
    /// they be equal as the program runs. `resolving` holds the phis being followed.
    fn copied(&self, v: Value, resolving: &mut Vec<Value>, depth: u32) -> Option<Value> {
        if depth > 64 {
            return None;
        }
        if let Some(&(b, k)) = self.phi_at.get(&v) {
            if resolving.contains(&v) {
                return Some(v); // round a loop to itself: no other value
            }
            resolving.push(v);
            let mut root = None;
            for &a in &self.x.p.blocks[b].phis[k].args {
                let r = self.copied(a, resolving, depth + 1)?;
                if resolving.contains(&r) {
                    continue;
                }
                match root {
                    None => root = Some(r),
                    Some(x) if x == r => {}
                    Some(_) => return None,
                }
            }
            resolving.retain(|&x| x != v);
            return root;
        }
        let Some((_, inst)) = self.writer(v) else {
            return Some(v); // a register the hardware set up
        };
        let Some(&(Reg::S(at), _)) = inst.writes.iter().find(|w| w.1 == v) else {
            return Some(v);
        };
        let Fields::Sop { dst: Some(Operand::Scalar { code: dst, .. }), src, .. } = &inst.fields
        else {
            return Some(v);
        };
        let name = inst.name();
        let from = match (name, &src[..]) {
            ("s_mov_b32" | "s_mov_b64", [Operand::Scalar { code, .. }]) => code + (at - dst),
            ("s_and_saveexec_b32" | "s_and_saveexec_b64", _) if at != EXEC && at != EXEC + 1 => {
                EXEC + (at - dst)
            }
            _ => return Some(v),
        };
        self.copied(inst.read(Reg::S(from))?, resolving, depth + 1)
    }

    /// Whether block `b` ends in a plain jump back to a loop header. As lowered a lane goes
    /// round only below the iteration limit and leaves the program at it (`Lower::exit`,
    /// which asks this; `abort_block`).
    pub(super) fn jumps_back(&self, b: usize) -> bool {
        let back = |t: usize| self.counters.contains_key(&t) && self.rpo[t] <= self.rpo[b];
        matches!(self.x.st.exits[b], LaneExit::Jump(t) if back(t))
    }

    /// The two ways out of block `q` as lowered, if it has two: a branch's targets; a jump
    /// back's header and out of the program (`None`, [`Self::jumps_back`]).
    fn ways(&self, q: usize) -> Option<[Option<usize>; 2]> {
        match self.x.p.blocks[q].exit {
            Exit::Branch { taken, not_taken } => Some([Some(taken), Some(not_taken)]),
            Exit::Jump(t) if self.jumps_back(q) => Some([Some(t), None]),
            _ => None,
        }
    }

    /// Per block, as bits: the blocks every way from it to the program's end passes as
    /// lowered, itself too (its post-dominators). A jump back is an end too: from it a lane
    /// may leave the program ([`Self::jumps_back`]). The sets are exact where every block
    /// that runs has a way to an end, so a program with a block that has none (a loop that
    /// only branches back into itself) is refused (Neon's sixth and seventh reviews).
    fn certain(&self) -> Result<Vec<Vec<u64>>, Why> {
        let p = self.x.p;
        let n = p.blocks.len();
        let own = |b: usize| {
            let mut bits = vec![0u64; n.div_ceil(64)];
            bits[b / 64] |= 1 << (b % 64);
            bits
        };
        let ends = |b: usize| p.blocks[b].exit.successors().is_empty() || self.jumps_back(b);
        let to_end = reached(n, (0..n).filter(|&b| ends(b)), None, |b| p.blocks[b].preds.clone());
        if (0..n).any(|b| self.rpo[b] != usize::MAX && !to_end[b]) {
            return Err(ENDLESS);
        }
        let mut sets: Vec<Vec<u64>> =
            (0..n).map(|b| if ends(b) { own(b) } else { vec![u64::MAX; n.div_ceil(64)] }).collect();
        loop {
            let mut changed = false;
            for b in (0..n).rev().filter(|&b| !ends(b)) {
                let (mut bits, next) = (own(b), p.blocks[b].exit.successors());
                for (w, word) in bits.iter_mut().enumerate() {
                    *word |= next.iter().map(|&t| sets[t][w]).fold(u64::MAX, |x, y| x & y);
                }
                changed |= std::mem::replace(&mut sets[b], bits.clone()) != bits;
            }
            if !changed {
                return Ok(sets);
            }
        }
    }

    /// The blocks `block` hangs on: those with two ways ([`Self::ways`]) one of which comes
    /// to it for certain while the block itself does not (or is `block`, in a loop), and
    /// the blocks those hang on in turn. Which way each goes decides whether `block` runs;
    /// nothing else does.
    fn hangs_on(&self, certain: &[Vec<u64>], block: usize) -> Vec<usize> {
        let mut found: Vec<usize> = Vec::new();
        let mut todo = vec![block];
        while let Some(x) = todo.pop() {
            for q in 0..self.x.p.blocks.len() {
                let Some(ways) = self.ways(q) else {
                    continue;
                };
                let [one, other] = ways.map(|t| t.is_some_and(|t| t == x || has(&certain[t], x)));
                if (q == x || !has(&certain[q], x)) && one != other && !found.contains(&q) {
                    found.push(q);
                    todo.push(q);
                }
            }
        }
        found
    }

    /// Whether every lane runs block `b` or none does: it is in no loop, and every block it
    /// hangs on ([`Self::hangs_on`]) ends in a flag of the proved form ([`Self::flag_branch`]).
    fn everyone(&self, b: usize, certain: &[Vec<u64>]) -> bool {
        let p = self.x.p;
        let after = |x: usize| p.blocks[x].exit.successors();
        !reached(p.blocks.len(), after(b), None, after)[b]
            && self.hangs_on(certain, b).iter().all(|&q| self.flag_branch(q).is_ok())
    }

    /// Whether the lanes that reach the sample at `site` are the lanes the console
    /// samples in there, by the forms of the module's text: its EXEC is of the list, and
    /// every branch its block hangs on is a proved flag or the `s_cbranch_execz` of a mask
    /// that EXEC is made of, entered on its fall-through side, in no loop with the sample.
    fn lanes_sampled(&self, site: Site, certain: &[Vec<u64>]) -> Result<(), Why> {
        let p = self.x.p;
        let inst = &p.blocks[site.0].insts[site.1];
        let mut made_of = Vec::new();
        self.run_of(self.exec_read(inst).ok_or(LANE_SET)?, &mut made_of, 0).ok_or(LANE_SET)?;
        let n = p.blocks.len();
        let after = |b: usize| p.blocks[b].exit.successors();
        let later = reached(n, after(site.0), None, after);
        let leads = |b: usize| reached(n, [b], None, after)[site.0];
        for q in self.hangs_on(certain, site.0) {
            if self.flag_branch(q).is_ok() {
                continue;
            }
            let Some(branch) = p.blocks[q].insts.last() else {
                return Err(BEHIND);
            };
            let Exit::Branch { taken, not_taken } = p.blocks[q].exit else {
                return Err(BEHIND);
            };
            let tested = self.exec_read(branch).map(|e| e.0);
            if branch.name() != "s_cbranch_execz"
                || !tested.is_some_and(|e| made_of.contains(&e))
                || !leads(not_taken)
                || leads(taken)
                || later[q]
            {
                return Err(BEHIND);
            }
        }
        Ok(())
    }

    /// The coordinates of the implicit sample at `site`, if each is a value of the list
    /// (the module's text), computed where every lane of its quad runs; refused otherwise.
    fn coordinates(&self, site: Site, certain: &[Vec<u64>]) -> Result<Vec<Value>, Why> {
        let inst = &self.x.p.blocks[site.0].insts[site.1];
        let Fields::Image { addr, .. } = &inst.fields else {
            return Err(OPERATION);
        };
        let mut regs = Vec::new();
        for a in addr {
            let Operand::Vector { reg, dwords } = *a else {
                return Err(COORDINATES);
            };
            regs.extend((0..dwords).map(|k| reg + k));
        }
        if regs.len() != 2 {
            return Err(COORDINATES);
        }
        let mut seen = HashSet::new();
        let mut coordinates = Vec::new();
        for r in regs {
            let v = inst.read(Reg::V(r)).ok_or(COORDINATES)?;
            self.listed(v, certain, &mut seen)?;
            coordinates.push(v);
        }
        Ok(coordinates)
    }

    /// Whether `v` is a value of the list (the module's text), and every value it is made
    /// of; `seen` holds those already found so.
    fn listed(&self, v: Value, certain: &[Vec<u64>], seen: &mut HashSet<Value>) -> Result<(), Why> {
        if !seen.insert(v) {
            return Ok(());
        }
        let p = self.x.p;
        if let Some(&(reg, _)) = p.inputs.iter().find(|i| i.1 == v) {
            // User data: scalars the hardware set up, the same in every lane.
            return match reg {
                Reg::S(code) if code < VCC => Ok(()),
                _ => Err(UNLISTED),
            };
        }
        if self.phi_at.contains_key(&v) {
            return Err(JOINED);
        }
        let ((b, i), inst) = self.writer(v).ok_or(UNLISTED)?;
        if !self.everyone(b, certain) {
            return Err(COMPUTED);
        }
        let &(reg, _) = inst.writes.iter().find(|w| w.1 == v).ok_or(UNLISTED)?;
        let name = inst.name();
        match (reg, &inst.fields) {
            (Reg::S(_), Fields::Smem { .. }) => {
                self.buffer_word((b, i), inst).map_err(|_| UNLISTED)
            }
            (Reg::S(_), Fields::Sop { .. }) if SALU.contains(&name) => {
                for &(r, x) in &inst.reads {
                    match r {
                        Reg::S(code) if code < VCC => self.listed(x, certain, seen)?,
                        _ => return Err(UNLISTED),
                    }
                }
                Ok(())
            }
            (Reg::V(_), Fields::Vintrp { .. }) => {
                self.interpolated_attribute((b, i), inst, certain)
            }
            // A select by the lane's own compare ([`Self::compared`]).
            (Reg::V(_), Fields::Valu(valu))
                if name == "v_cndmask_b32"
                    && valu.dpp.is_none()
                    && valu.sdwa.is_none()
                    && valu.src.get(..2).is_some_and(sources) =>
            {
                self.whole_quads((b, i), inst)?;
                // The mask in VCC or in SGPRs below it: never EXEC.
                let width = 1 + u8::from(p.wave64);
                let [_, _, Operand::Scalar { code, .. }] = valu.src[..] else {
                    return Err(UNLISTED);
                };
                if code != VCC && u16::from(code) + u16::from(width) > u16::from(VCC) {
                    return Err(UNLISTED);
                }
                let mask: Vec<Reg> = (0..width).map(|k| Reg::S(code + k)).collect();
                let bits: Option<Vec<Value>> = mask.iter().map(|&r| inst.read(r)).collect();
                self.compared(&bits.ok_or(UNLISTED)?, certain, seen)?;
                // The two values it selects between, each a value of the list: the mask
                // among them is a number, none (Neon's sixth review).
                for s in &valu.src[..2] {
                    let r = match *s {
                        Operand::Vector { reg, .. } => Reg::V(reg),
                        Operand::Scalar { code, .. } if code < VCC => Reg::S(code),
                        Operand::Const(_) | Operand::Literal(_) => continue,
                        _ => return Err(UNLISTED),
                    };
                    self.listed(inst.read(r).ok_or(UNLISTED)?, certain, seen)?;
                }
                Ok(())
            }
            (Reg::V(_), Fields::Valu(valu))
                if VALU.contains(&name)
                    && valu.dpp.is_none()
                    && valu.sdwa.is_none()
                    && sources(&valu.src) =>
            {
                self.whole_quads((b, i), inst)?;
                self.operands(inst, certain, seen)
            }
            _ => Err(UNLISTED),
        }
    }

    /// Whether every register the vector instruction `inst` reads (its explicit sources
    /// [`sources`] first, a destination it accumulates into too) holds a value of the list.
    /// EXEC is read only as every vector instruction reads it, the low dword and the high
    /// one of a wave64 program: its lanes, which [`Self::whole_quads`] proves.
    fn operands(
        &self,
        inst: &Inst,
        certain: &[Vec<u64>],
        seen: &mut HashSet<Value>,
    ) -> Result<(), Why> {
        let p = self.x.p;
        let lanes = |r: Reg| r == Reg::S(EXEC) || p.wave64 && r == Reg::S(EXEC + 1);
        for r in crate::ir::operand_reads(p, inst) {
            match r {
                _ if lanes(r) => {}
                Reg::V(_) => self.listed(inst.read(r).ok_or(UNLISTED)?, certain, seen)?,
                Reg::S(code) if code < VCC => {
                    self.listed(inst.read(r).ok_or(UNLISTED)?, certain, seen)?;
                }
                _ => return Err(UNLISTED),
            }
        }
        Ok(())
    }

    /// Whether the mask `bits` (its dwords) is the whole result of one `v_cmp_*` (not
    /// `v_cmpx`, no DPP or SDWA) of values of the list, in a block every lane runs, under
    /// whole quads: each lane's bit is its own compare, which every lane of its quad made
    /// on the console as on the host. A mask of anything else (EXEC, a mask op, a constant,
    /// a join) is not one.
    fn compared(
        &self,
        bits: &[Value],
        certain: &[Vec<u64>],
        seen: &mut HashSet<Value>,
    ) -> Result<(), Why> {
        let &[first, ..] = bits else {
            return Err(UNLISTED);
        };
        if self.phi_at.contains_key(&first) {
            return Err(JOINED);
        }
        let ((b, i), cmp) = self.writer(first).ok_or(UNLISTED)?;
        let Fields::Valu(valu) = &cmp.fields else {
            return Err(UNLISTED);
        };
        let whole = bits.iter().all(|v| cmp.writes.iter().any(|w| w.1 == *v));
        if !cmp.name().starts_with("v_cmp_")
            || !whole
            || valu.dpp.is_some()
            || valu.sdwa.is_some()
            || !sources(&valu.src)
        {
            return Err(UNLISTED);
        }
        if !self.everyone(b, certain) {
            return Err(COMPUTED);
        }
        self.whole_quads((b, i), cmp)?;
        self.operands(cmp, certain, seen)
    }

    /// Whether the vector instruction at `site` runs under whole quads: unguarded, with
    /// EXEC `s_wqm` of the program's own through plain copies ([`Self::run_of`]).
    fn whole_quads(&self, (b, i): Site, inst: &Inst) -> Result<(), Why> {
        if self.x.st.guards[b][i].is_some() {
            return Err(UNWHOLE);
        }
        let exec = self.exec_read(inst).ok_or(UNWHOLE)?;
        match self.run_of(exec, &mut Vec::new(), 0) {
            Some(Run::Whole) => Ok(()),
            _ => Err(UNWHOLE),
        }
    }

    /// Whether the `v_interp*` at `site` is an interpolated attribute of the list: the
    /// second half of one pair with the hardware's barycentrics ([`Interp::Interpolated`]),
    /// both halves where every lane runs under whole quads, of a row the stage data marks
    /// neither flat nor of halves, read as a whole channel.
    fn interpolated_attribute(
        &self,
        site: Site,
        inst: &Inst,
        certain: &[Vec<u64>],
    ) -> Result<(), Why> {
        if !inst.name().starts_with("v_interp_p2") {
            return Err(UNINTERPOLATED);
        }
        let Some(Interp::Interpolated { attribute, first, .. }) = self.interps.get(&site).copied()
        else {
            return Err(UNINTERPOLATED);
        };
        let &[half] = &first[..] else {
            return Err(UNINTERPOLATED);
        };
        // Flat: the console's flat attribute is the provoking vertex's value whatever the
        // interpolation reads.
        let other = match &self.x.native.io {
            StageIo::Ps { inputs, .. } => {
                inputs.iter().any(|i| i.row == attribute.row && (i.fp16 || i.flat))
            }
            _ => true,
        };
        if other || attribute.high || self.per_vertex(attribute.row) {
            return Err(UNINTERPOLATED);
        }
        self.whole_quads(site, inst)?;
        let p1 = &self.x.p.blocks[half.0].insts[half.1];
        if !p1.name().starts_with("v_interp_p1") {
            return Err(UNINTERPOLATED);
        }
        if !self.everyone(half.0, certain) {
            return Err(COMPUTED);
        }
        self.whole_quads(half, p1)
    }

    /// `sampleGrad` in `block` of the handles and coordinates `args`, texel offsets `offsets`
    /// and gradients `made` (across and down, by coordinate; the rest `undef`), with the
    /// level clamp `clamp`: the call of an implicit sample with a chosen sampler
    /// ([`Chosen::gradients`]) and of `image_sample_d`.
    pub(super) fn sample_grad(
        &mut self,
        block: usize,
        args: &[V],
        offsets: &[V],
        made: Vec<[V; 2]>,
        [undef, clamp]: [V; 2],
    ) -> V {
        let mut g = vec![undef; 6];
        for (k, [across, down]) in made.into_iter().enumerate().take(3) {
            g[k] = across;
            g[3 + k] = down;
        }
        let mut args = args.to_vec();
        args.extend(offsets);
        args.extend(g);
        args.push(clamp);
        self.s.op(block, "sampleGrad.f32", Attr::ReadOnly, self.s.t.res_ret, 63, &args)
    }

    /// `sample`, the call made with the first sampler of `chosen`, made again at the end
    /// of `block` with each other one: whether that one is the lane's, and its result.
    pub(super) fn sampled_again(
        &mut self,
        block: usize,
        sample: V,
        chosen: &Option<Chosen>,
    ) -> Vec<(V, V)> {
        let Some(chosen) = chosen else {
            return Vec::new();
        };
        let mut again = Vec::new();
        for &(is, handle) in &chosen.others {
            again.push((is, self.s.m.call_with(block, sample, chosen.first, handle)));
        }
        again
    }

    /// Channel `ch` of `sample`, or of the one among `again` whose sampler is the lane's.
    pub(super) fn chosen_channel(
        &mut self,
        block: usize,
        sample: V,
        ch: u32,
        again: &[(V, V)],
    ) -> V {
        let mut x = self.s.m.extract(block, sample, ch);
        for &(is, other) in again {
            let y = self.s.m.extract(block, other, ch);
            x = self.s.m.select(block, is, y, x);
        }
        x
    }
}
