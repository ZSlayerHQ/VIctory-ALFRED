//! The meeting form: a console wave of 64 lanes on a host whose waves are narrower, exact.
//!
//! The console runs the 64 lanes of a wave in step, so one lane can ask the others: a sum or
//! a count over the wave, the value its first active lane holds, the value of lane 37, a
//! lane mask as a number. A host of 32-lane waves runs those lanes as two waves (WARP as
//! sixteen), and a host wave operation only sees its own. The forms before this one either
//! let each host wave answer for itself (approximate: half a sum, a first lane per half) or
//! exchange a host wave's own answer through two dwords (`Lower::exchange`, exact, but it
//! declares 32-lane waves and reads the first lane and a named lane with host wave
//! operations). This form uses no host wave operation at all and declares no wave size:
//!
//! * **Where the lanes meet.** At the operation itself, in a block every thread of the group
//!   runs. The threads store what they hold, wait for the group (one group barrier with a
//!   shared-memory fence), read what they need, and wait once more before the slots are
//!   written again.
//! * **What they exchange, and through what.** Group-shared memory after the program's own:
//!   one dword per thread (its value, or the operation's identity where the lane is not in
//!   EXEC) and two dwords per console wave (the lanes' bits, set with an atomic OR and
//!   cleared by each lane after the second wait). A thread's console lane is its index in
//!   the group modulo 64, as on the console, so nothing depends on which host wave holds it.
//!   A named lane is one load; the first active lane is the lowest set bit of the wave's two
//!   dwords, then one load (lane 0 with no lane in EXEC, as the hardware reads it); a mask as
//!   a number is the two dwords; a reduction is the 64 slots folded pairwise in lane order
//!   (lanes 2k and 2k+1, then pairs, up to the two halves); a prefix is the fold of the
//!   lanes below. A sum of floats rounds at each addition, so its terms are added in the
//!   order the program's own instructions add them (`wave::Order`, which the simulation of
//!   the section keeps), reduction and prefix alike: the console's bits, whatever expansion
//!   the compiler chose. One whose order is not kept is refused ([`FLOAT_ORDER`]).
//! * **Under control flow only some lanes reach.** A barrier needs every thread, so a
//!   meeting may only stand where all of them come. Three things bring them there. Branches
//!   every lane takes alike are told from per-lane ones ([`together`]): the compiler keeps
//!   values that are one per wave in VGPRs (a load at a scalar address, arithmetic on it, a
//!   compare of it) and branches on masks made of them, which the structure sees as per-lane
//!   branches. The skip-free forms of the program (`ir::without_skips`) let lanes outside
//!   EXEC walk a skipped region with their effects masked, and they then take part with the
//!   identity. And, only when asked for (`VICTORY_WAVE_MEET_WIDE=1`, not run on a card yet),
//!   the per-lane branches around a meeting are taken as the console takes them, by the whole
//!   wave at once: the lanes vote at a meeting and walk the branch or the loop together
//!   ([`Lower::meet_wide`], `emit::widen`), in a group of one console wave. What is left (a
//!   meeting inside a loop the lanes leave one by one, or under a branch on values that
//!   differ between lanes that no form removes) is refused here with the place and the idiom
//!   named, and the forms after this one take the program as before.
//! * **Order of shared memory.** The program's own LDS accesses keep the console's order
//!   between host waves by a group barrier before each (as with two host waves before).
//! * **Cost, per thread group and per time the operation runs.** Two group barriers per
//!   meeting. A lane read or a first-lane read: one store, one or three loads, and for the
//!   first lane two atomics. A mask: two atomics and two loads. A reduction or prefix: one
//!   store and 64 loads per thread. Scratch: `threads + 2 * ceil(threads / 64) + 2` dwords
//!   of group-shared memory (68 for one wave), and one more barrier at the program's start
//!   where masks are asked (the pairs are cleared there). The last two dwords take the
//!   program's own LDS accesses past the end of its LDS, which the console reads as 0 and
//!   drops, so that no address of the program's reaches the meeting's dwords
//!   ([`Lower::meet_bound`]: a compare and a select per LDS dword accessed). An atomic past
//!   the end is not run, and returns 0 ([`Lower::meet_atomic`]).
//! * **What is refused, each with a reason of its own.** A group barrier some thread does
//!   not reach stops the group on the host, so the form does not take: a meeting in the
//!   block of a branch each lane takes for itself that a way leads back to ([`APART`]: a
//!   loop the lanes leave one by one; from the second round on only some run it); a barrier
//!   of the program's own (`s_barrier`) in a block not every thread runs ([`OWN_BARRIER`]);
//!   two LDS writes, or one a way leads back to, in the blocks of one branch only some
//!   lanes take ([`LDS_WRITES`]: the waves meet around those blocks, not between their
//!   instructions); a group barrier in a loop, or past one, that the threads of the group
//!   do not walk together ([`ASTRAY`], the rule of the loops, [`walked`]); and any group
//!   of several console waves ([`SEVERAL_WAVES`]): each wave answers its own masks and
//!   lane reads, and the proof that the threads come together is one for a single wave.
//! * **Loops.** The lowering gives every loop a turn limit (`Lower::exit`): a thread at
//!   the limit takes the other arm of a branch back, and returns at a plain jump back.
//!   Threads whose turn counts differ part there, and a later barrier waits for the ones
//!   that left. So a group barrier stands only where every loop that holds it, or that a
//!   way leads from to it, is one all the threads enter together, leave and go round by
//!   the same decisions, and count alike ([`walked`]): their counters then never differ,
//!   and the limit is everyone's or no one's. Per-lane branches inside such a loop are
//!   fine where both arms join again before a block that leaves it or goes back.
//! * **Not covered.** Meetings where not every thread comes (above); groups of several
//!   console waves (above); sums of doubles and of 16-bit floats; a barrier beside or past a loop
//!   a lane can be cut out of (a plain jump back, or a branch back whose other arm stays in
//!   the loop, `crate::bounds::cut_at_limit`) which the threads do not go round together (a
//!   lane cut at the iteration limit returns, and would be missed there: that return counts as an end of
//!   the program where [`together`] finds the blocks every thread runs, and the way back of
//!   a loop no way leads out of counts as one whoever goes round it, [`closed_ways_back`]);
//!   lane permutes and DPP outside whole-wave sections are lane reads here (two barriers
//!   each), which is exact and slow.
//!
//! Learned from Leonida Continued's wave64 emulation (`leonida/docs/design.md`, the row
//! "Wave64 emulation", and `leonida/tools/rdna2spirv.py`, `x_round` and `x_memory_order`):
//! every cross-lane value through a workgroup buffer and a barrier in uniform control flow.
//! This is our own form of it: Leonida cuts the program into scheduler rounds, this keeps
//! the per-lane structure and only meets where that structure already has every lane.

use std::collections::{HashMap, HashSet};

use super::{Lower, Parts, Why, lane_regions_by, need};
use crate::emit::bitcode::{Attr, Bin, Cast, Rmw, V, pred};
use crate::emit::wave::{Op, Order, Shape, Step};
use crate::gfx10::Fields;
use crate::ir::{EXEC, Reg, Value};
use crate::resource::{Role, Source};
use crate::structure::{Atom, Bool, LaneExit};

thread_local! {
    /// See [`with_meeting`].
    static MEETING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` with console wave64 compute programs lowered in the meeting form (this module)
/// where the host's waves are narrower than the console's.
pub(in crate::emit) fn with_meeting<R>(f: impl FnOnce() -> R) -> R {
    let was = MEETING.replace(true);
    let r = f();
    MEETING.set(was);
    r
}

thread_local! {
    /// See [`take_first_lane_kept`].
    static FIRST_LANE_KEPT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A lowering took a first-lane offset per host wave (the shortcut `Lower::new` counts).
pub(super) fn kept_first_lane() {
    FIRST_LANE_KEPT.set(true);
}

/// Whether a lowering on this thread since the last call read a first-lane offset per host
/// wave instead of for the console's wave: such a program is tried in the meeting form.
pub(in crate::emit) fn take_first_lane_kept() -> bool {
    FIRST_LANE_KEPT.replace(false)
}

/// Whether [`with_meeting`] is on.
pub(super) fn meeting() -> bool {
    MEETING.get()
}

/// What stops a meeting where some thread of the group does not come.
pub(in crate::emit) const NOT_EVERY: Why = "a meeting of the wave's lanes where not every lane runs";

/// What stops a sum of floats the meeting cannot add up as the program does: the steps
/// were not kept, or are more than [`FLOAT_STEPS`].
pub(in crate::emit) const FLOAT_ORDER: Why = "a float sum whose order of additions the meeting form cannot follow";

/// The most steps (terms and additions) of a float sum a meeting adds up: every thread
/// runs them all. The compiler's sum over a wave has 127, its prefix under 500.
const FLOAT_STEPS: usize = 1024;

/// What stops a meeting in the block of a branch each lane takes for itself that a way
/// leads back to: every thread runs it the first time, and only the lanes still in the loop
/// after that.
pub(in crate::emit) const APART: Why = "a meeting in a loop the lanes leave one by one, where not every lane runs";

/// What stops the form for a program whose own barrier stands in a block some thread does
/// not run: it waits for the group as a meeting's barrier does.
pub(in crate::emit) const OWN_BARRIER: Why = "a barrier of the program's own where not every lane runs";

/// What stops the form for LDS writes in the blocks of one lane branch whose order the
/// meetings around those blocks do not keep.
pub(in crate::emit) const LDS_WRITES: Why = "LDS writes where not every lane runs that no meeting orders";

/// What stops the form for a group of several console waves. Each console wave answers its
/// own lane masks, ballots, first lanes and lane reads, so a value one for every lane of a
/// wave may differ between the waves of the group, and a branch on it send them different
/// ways before a barrier the whole group must reach. The proof that a branch is taken alike
/// ([`together`]) is a proof for one wave; the form takes groups of one console wave only.
pub(in crate::emit) const SEVERAL_WAVES: Why = "a group of several console waves in the meeting form";

/// What stops the form for a group barrier (a meeting, the program's own, one before an LDS
/// access) in a loop, or past one, that the threads of the group do not walk together
/// ([`walked`]): their turn counts can differ there, and with them who is left at the limit.
pub(in crate::emit) const ASTRAY: Why = "a group barrier in or past a loop the threads do not walk together";

/// What [`together`] finds.
pub(super) struct Together {
    /// Per IR block: every thread of the group runs it, whichever way the program goes.
    pub all: Vec<bool>,
    /// Per IR block: it ends in a branch each lane takes for itself that a way leads back
    /// to (not in `all`: [`APART`]).
    pub apart: Vec<bool>,
    /// Per IR block: its exit is not a branch each lane takes for itself (a jump, an end, a
    /// branch every thread that comes takes alike).
    pub alike: Vec<bool>,
    /// What a `v_readfirstlane_b32` writes where the value it reads is the same in every
    /// lane and every arriving lane is in EXEC: the lane's own, with no meeting.
    pub own: HashSet<Value>,
}

/// Whether `inst` reads VGPR `r` only as the old value of a register it writes (what the
/// lanes outside EXEC keep), not as an operand. Not for results that keep part of their
/// register (16-bit and SDWA results) nor for the ones that add to it (`v_fmac`), and an
/// address register a load writes its result over is an operand. Nor for a swap, whose two
/// registers each take what the other held, nor for the moves and swaps relative to M0,
/// which write every register they may reach and leave all but one as it was: there the
/// old value of a written register is what the instruction puts somewhere.
fn old_only(inst: &crate::ir::Inst, r: Reg) -> bool {
    use crate::gfx10::Operand;
    let Reg::V(d) = r else { return false };
    if !inst.writes.iter().any(|w| w.0 == r) {
        return false;
    }
    // (an operand may span registers: a 64-bit value, an index and an offset, coordinates)
    let spans = |op: &Operand, n: u8| matches!(*op, Operand::Vector { reg, .. } if d >= reg && d - reg < n);
    let name = inst.name();
    match &inst.fields {
        Fields::Valu(v) => {
            v.sdwa.is_none()
                && !name.contains("swap")
                && !name.contains("movrel")
                && !name.contains("fmac")
                && !name.contains("_mac_")
                && !super::keeps_part_of_dst(name)
                && !v.src.iter().any(|op| spans(op, 4))
        }
        Fields::Buffer { addr, .. } => !spans(addr, 2),
        Fields::Image { addr, .. } => !addr.iter().any(|op| spans(op, 12)),
        _ => false,
    }
}

/// A piece of memory a program stores to: a descriptor of a table (parameter and byte), or
/// whatever is built around a root address.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Written {
    Table(usize, u32),
    Root(usize),
}

/// The pieces of memory behind the descriptor the instruction at `site` uses; `None` where
/// that is not known (no descriptor traced, a computed table offset).
fn pieces(x: &Parts, site: (usize, usize)) -> Option<Vec<Written>> {
    let d = x.res.descriptors.iter().find(|d| d.site == site && d.role == Role::Resource)?;
    d.source
        .leaves()
        .into_iter()
        .map(|leaf| match leaf {
            Source::Loaded { load, first } => {
                let a = x.res.scalar.iter().find(|(s, _)| s == load).map(|(_, a)| *a)?;
                a.dynamic.is_none().then_some(Written::Table(a.param, a.offset.wrapping_add(4 * u32::from(*first))))
            }
            Source::Built { address, .. } => Some(Written::Root(address.param)),
            _ => None,
        })
        .collect()
}

/// The kinds of the descriptors behind the instruction at `site`, by the ranges the
/// program reads (`NativeProgram::used`: the range a table descriptor falls in, the root
/// descriptor's); `None` where one is not known.
fn kinds(x: &Parts, site: (usize, usize)) -> Option<Vec<crate::native::ResourceType>> {
    use crate::native::RootType;
    let d = x.res.descriptors.iter().find(|d| d.site == site && d.role == Role::Resource)?;
    let used = &x.native.used;
    d.source
        .leaves()
        .into_iter()
        .map(|leaf| match leaf {
            Source::Loaded { load, first } => {
                let a = x.res.scalar.iter().find(|(s, _)| s == load).map(|(_, a)| *a)?;
                let rp = x.native.layout.get(a.param)?;
                if rp.root != RootType::Table || a.dynamic.is_some() {
                    return None;
                }
                let dword = a.offset / 4 + u32::from(*first);
                used.iter()
                    .filter(|u| u.root == RootType::Table && u.user_data == rp.user_data && u.table_offset_dwords <= dword)
                    .max_by_key(|u| u.table_offset_dwords)
                    .map(|u| u.kind)
            }
            Source::Built { address, .. } => {
                let rp = x.native.layout.get(address.param)?;
                used.iter().find(|u| u.root == RootType::Descriptor && u.user_data == rp.user_data).map(|u| u.kind)
            }
            _ => None,
        })
        .collect()
}

/// Which branches every lane takes alike, and so which blocks every thread of the group
/// runs: [`super::all_lane_blocks`] with more branches known to be taken together.
///
/// The structure calls a branch per-lane when its mask comes from a VALU compare. Compiled
/// code also keeps values that are one per wave in VGPRs: there is no scalar float unit and
/// no scalar buffer load of a typed element, so a probe's record loaded at a scalar index,
/// the arithmetic on it and the compares of it all run in every lane with the same result,
/// and `v_readfirstlane_b32` brings them back. A branch on a mask made only of such compares
/// (and of scalar tests) goes one way for the whole wave.
///
/// The group is one console wave ([`plan`] refuses groups of several): a value the same in
/// every lane of the wave is the same in every thread of the group.
///
/// A value is the same in every lane that holds it when:
/// - a scalar on entry;
/// - written by an instruction without cross-lane meaning whose inputs all are, masks
///   included (a mask is, when every atom of its lane bit is: a scalar test, a compare of
///   such values, a join of such masks in a block entered together), and which either every
///   arriving lane runs or runs under a guard that is the same in every lane over an old
///   value that is too. Cross-lane meaning: whatever the lowering computes from the lane's
///   number and not from the operands' values alone. So not a select (`v_cndmask_b32`) on
///   a scalar pair no mask is known for (the lane takes its own bit of the number,
///   `Lower::lane_bit`);
/// - loaded (scalar, buffer, image) at such an address from memory the program does not
///   store to through any descriptor it can be told from; where it stores anything, only
///   through a CBV or an SRV (two UAV descriptors in other places may view one buffer, and
///   a resource is not a UAV and a CBV or SRV of one dispatch); a scalar load from title
///   memory by address: where the program stores nothing at all, title memory being behind
///   every descriptor too;
/// - left by a whole-wave section as a constant, such a value, or such a value under such a
///   mask;
/// - joined by a phi of values that are so outright (not only under a guard's atoms) in a
///   block all of whose predecessors every thread runs (the lanes then arrive over the same
///   edge), or whose inputs are one value;
/// - what the wave answers as one, where the meeting form computes it for the console's
///   wave (a first-lane read into a scalar; a lane read where the lane's number is such a
///   value, each thread reading the lane it names, a VGPR as the number included, and the
///   register read as the number too where it is the same one; a scalar made from a lane
///   mask read as a number, and a mask joined with numbers or read as one by a vector
///   instruction; SCC of a mask operation; a reduction a section leaves; a sum of floats
///   only where every lane adds it up in the same order, [`Order::results`]). Not a
///   waterfall loop's first-lane read, which is lowered as each lane's own value before any
///   meeting (`Lower::valu`): that is one value only where its operand is.
///
/// Values and blocks depend on each other (a loop counter kept in a VGPR is the same in
/// every lane as long as every lane goes round together, which its own compare decides), so
/// both start as "all" and lose what breaks a rule until nothing changes: what is left holds
/// step by step along any run, the lanes starting together with equal scalars.
/// `None` with an indirect exit.
pub(super) fn together(x: &Parts) -> Option<Together> {
    let (p, st) = (x.p, x.st);
    let n = p.blocks.len();
    let order = crate::ir::reverse_postorder(&p.blocks);
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let mask_of = |v: Value| st.masks.get(&st.highs.get(&v).copied().unwrap_or(v)).copied();
    let exec = |r: Reg| matches!(r, Reg::S(c) if c == EXEC || c == EXEC + 1);
    // what the program stores to
    let mut written: HashSet<Written> = HashSet::new();
    let mut any_write = false;
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            let name = inst.name();
            let stores = name.contains("store") || name.contains("atomic");
            match &inst.fields {
                Fields::Buffer { .. } | Fields::Image { .. } if stores => match pieces(x, (b, i)) {
                    Some(w) => written.extend(w),
                    None => any_write = true,
                },
                Fields::Flat { .. } if !name.starts_with("global_load_") => any_write = true,
                _ => {}
            }
        }
    }
    // (where the program stores anything: only through a CBV or an SRV, which no UAV of the
    // dispatch views; two UAV descriptors in other places of the tables or a root UAV and a
    // table's may view one buffer)
    let viewed = |site: (usize, usize)| {
        use crate::native::ResourceType;
        written.is_empty() || kinds(x, site).is_some_and(|ks| ks.iter().all(|k| matches!(k, ResourceType::Cbv | ResourceType::Srv)))
    };
    let read_only = |site: (usize, usize)| {
        !any_write
            && viewed(site)
            && pieces(x, site).is_some_and(|ps| {
                ps.iter().all(|piece| match *piece {
                    Written::Root(param) => !written.iter().any(|w| matches!(*w, Written::Root(q) if q == param)),
                    table => !written.contains(&table),
                })
            })
    };
    let sections: Vec<_> = (0..n).map(|b| super::super::wave::sections(p, b)).collect();
    let phi_block: HashMap<Value, usize> =
        p.blocks.iter().enumerate().flat_map(|(b, blk)| blk.phis.iter().map(move |phi| (phi.value, b))).collect();
    // Joins of masks that hold in every lane: every edge brings "all lanes" or the join
    // itself (EXEC saved and restored around a loop's body). Start with all of them.
    let mut always: HashSet<Value> = st.phi_masks.keys().copied().collect();
    loop {
        let holds = |b: Bool, always: &HashSet<Value>| {
            let mut b = b;
            loop {
                match st.bools.node(b) {
                    None => return b == Bool::TRUE,
                    Some((Atom::Phi(id), _, hi)) if always.contains(&id) => b = hi,
                    Some(_) => return false,
                }
            }
        };
        let gone: Vec<Value> = always
            .iter()
            .copied()
            .filter(|id| !st.phi_masks[id].iter().all(|&e| holds(e, &always)))
            .collect();
        if gone.is_empty() {
            break;
        }
        for id in gone {
            always.remove(&id);
        }
    }
    // A value with the atoms under which it is the same in every lane (none: always).
    type Under = Vec<Atom>;
    fn join(into: &mut Under, more: &[Atom]) {
        for a in more {
            if !into.contains(a) {
                into.push(*a);
            }
        }
    }
    // The state: start with everything, always; take away or weaken what breaks a rule.
    // (On entry: the scalars.)
    let mut same: HashMap<Value, Under> = p
        .inputs
        .iter()
        .filter(|&&(r, _)| matches!(r, Reg::S(_)) && !exec(r))
        .map(|&(_, v)| (v, Vec::new()))
        .collect();
    // (the lane bits of VALU-written masks, before EXEC: `Atom::Lane`)
    let mut plain: HashMap<Value, Under> = HashMap::new();
    for block in &p.blocks {
        same.extend(block.phis.iter().map(|phi| (phi.value, Vec::new())));
        for inst in &block.insts {
            same.extend(inst.writes.iter().map(|w| (w.1, Vec::new())));
            if matches!(inst.fields, Fields::Valu(_)) {
                plain.extend(inst.writes.iter().filter(|w| matches!(w.0, Reg::S(_))).map(|w| (w.1, Vec::new())));
            }
        }
    }
    struct State<'s> {
        st: &'s crate::structure::Structured,
        same: &'s HashMap<Value, Under>,
        plain: &'s HashMap<Value, Under>,
        always: &'s HashSet<Value>,
        joined: &'s dyn Fn(Value) -> bool,
    }
    impl State<'_> {
        /// Whether `b` is the same in every lane, on the path that made `held` true.
        fn alike(&self, b: Bool, held: &mut Vec<Atom>, seen: &mut Vec<Value>, budget: &mut u32) -> bool {
            let Some((a, lo, hi)) = self.st.bools.node(b) else { return true };
            if *budget == 0 {
                return false;
            }
            *budget -= 1;
            let under = |u: &Under, held: &[Atom]| {
                u.iter().all(|c| held.contains(c) || matches!(c, Atom::Phi(id) if self.always.contains(id)))
            };
            let ok = match a {
                Atom::Phi(id) if self.always.contains(&id) => return self.alike(hi, held, seen, budget),
                Atom::Uniform(v) => self.same.get(&v).is_some_and(|u| under(u, held)),
                Atom::Lane(v) => self.plain.get(&v).is_some_and(|u| under(u, held)),
                Atom::Phi(id) => {
                    // (a join around a loop: as its other edges say)
                    seen.contains(&id) || {
                        seen.push(id);
                        (self.joined)(id)
                            && self.st.phi_masks.get(&id).is_some_and(|edges| {
                                edges.iter().all(|&e| self.alike(e, &mut Vec::new(), seen, budget))
                            })
                    }
                }
                Atom::Quad(_) | Atom::Opaque(_) | Atom::Constant(_) => false,
            };
            if !ok || !self.alike(lo, held, seen, budget) {
                return false;
            }
            held.push(a);
            let r = self.alike(hi, held, seen, budget);
            held.pop();
            r
        }

        fn mask(&self, b: Bool) -> bool {
            self.alike(b, &mut Vec::new(), &mut Vec::new(), &mut 4096)
        }

        /// Whether `b` holds in every lane, always.
        fn holds(&self, b: Bool) -> bool {
            let mut b = b;
            loop {
                match self.st.bools.node(b) {
                    None => return b == Bool::TRUE,
                    Some((Atom::Phi(id), _, hi)) if self.always.contains(&id) => b = hi,
                    Some(_) => return false,
                }
            }
        }

        /// The atoms of `b` where it is all of them at once (`a && b && ...`).
        fn all_of(&self, b: Bool) -> Option<Under> {
            let (mut b, mut atoms) = (b, Vec::new());
            loop {
                match self.st.bools.node(b) {
                    None => return (b == Bool::TRUE).then_some(atoms),
                    Some((a, Bool::FALSE, hi)) => {
                        if !matches!(a, Atom::Phi(id) if self.always.contains(&id)) {
                            atoms.push(a);
                        }
                        b = hi;
                    }
                    Some(_) => return None,
                }
            }
        }
    }
    let mut all = vec![true; n];
    let mut apart = vec![false; n];
    let mut alike = vec![true; n];
    let mut own: HashSet<Value> = HashSet::new();
    // The iteration limit: a way back to a loop's header with no other way on for a thread
    // that reaches the limit (a plain jump back, or a branch whose other arm stays in the
    // loop) is lowered with a return for it (`abort_block`), an end of the program the
    // program does not have. Per block: the header of such a way back
    // (`crate::bounds::cut_at_limit`).
    let mut place = vec![usize::MAX; n];
    for (i, &b) in order.iter().enumerate() {
        place[b] = i;
    }
    let jumps_back: Vec<Option<usize>> = (0..n).map(|b| crate::bounds::cut_at_limit(p, st, &place, b, true)).collect();
    let closed = closed_ways_back(st, &place);
    loop {
        let mut changed = false;
        own.clear();
        let entered_together =
            |b: usize, all: &[bool]| p.blocks[b].preds.iter().all(|&q| all.get(q).copied().unwrap_or(false));
        for &b in &order {
            let block = &p.blocks[b];
            for phi in &block.phis {
                if !same.contains_key(&phi.value) {
                    continue;
                }
                let one = phi.args.first().is_some_and(|a| phi.args.iter().all(|x| x == a));
                // Only what is alike outright passes a join: an atom a value was written
                // under names one run of its block, and around a loop the join would carry
                // it into the next, where the same atom is another round's.
                // (around a loop the join's own value comes back: as the others say)
                let ok = (one || entered_together(b, &all))
                    && phi.args.iter().all(|a| *a == phi.value || same.get(a).is_some_and(|u| u.is_empty()));
                if !ok {
                    same.remove(&phi.value);
                    changed = true;
                }
            }
            for (i, inst) in block.insts.iter().enumerate() {
                if inst.writes.is_empty() {
                    continue;
                }
                let name = inst.name();
                let joined = |id: Value| phi_block.get(&id).is_some_and(|&j| entered_together(j, &all));
                let state = State { st, same: &same, plain: &plain, always: &always, joined: &joined };
                // what an operand holds: a mask by its lane bit, anything else as it is
                let value = |v: Value| -> Option<Under> {
                    match mask_of(v) {
                        Some(m) => state.mask(m).then(Vec::new),
                        None if is_mask(v) => None,
                        None => same.get(&v).cloned(),
                    }
                };
                // every operand (not EXEC, and not `skip`): the atoms they need
                let reads = |skip: &dyn Fn(Reg, Value) -> bool| -> Option<Under> {
                    let mut under = Vec::new();
                    for &(r, v) in &inst.reads {
                        if exec(r) || skip(r, v) {
                            continue;
                        }
                        join(&mut under, &value(v)?);
                    }
                    Some(under)
                };
                // (every operand; the old value of a written register is none)
                let every = |r: Reg, _: Value| old_only(inst, r);
                // (found: per written value, whether it stays and under what)
                let mut found: Vec<(Value, Option<Under>)> = Vec::new();
                let mut bits: Option<Under> = None;
                // inside a whole-wave section: by what the simulation found each value to be
                if let Some((_, sec)) = sections[b].iter().find(|(s, sec)| i > **s && i < sec.end) {
                    for &(_, w) in &inst.writes {
                        let under = match sec.shapes.get(&w) {
                            Some(&Shape::K(_)) => Some(Vec::new()),
                            Some(&Shape::Value(v)) => value(v),
                            Some(&Shape::Masked(src, mask, _)) => {
                                value(src).filter(|_| st.masks.get(&mask).is_some_and(|&m| state.mask(m)))
                            }
                            // A sum of floats rounds at each addition: the lanes hold one
                            // value only where they add it up the same way, which the
                            // section's order says (one result: a scalar, or a vector
                            // whose lanes all hold the same steps). "The sum over the
                            // wave" in every lane (the shape) is true of lanes that add
                            // the terms in different orders too, and those can differ:
                            // a branch on such a vector is each lane's own.
                            Some(&Shape::Active(Op::AddF, ..)) => {
                                let one = sec.orders.get(&w).is_some_and(|o| o.results.len() == 1);
                                one.then(Vec::new)
                            }
                            Some(&Shape::Active(..) | &Shape::ActiveF64(..)) => Some(Vec::new()),
                            Some(&Shape::Prefix(..)) | None => None,
                        };
                        found.push((w, under));
                    }
                } else {
                    // A guarded write leaves the old value in the lanes outside the guard:
                    // alike if the guard holds in every lane; or if the guard is alike and
                    // the old value too; or, the guard being all of some atoms, under those.
                    let guard = st.guards[b][i].filter(|&g| !state.holds(g));
                    let kept = &every;
                    let guarded = |mut under: Under| -> Option<Under> {
                        let Some(g) = guard else { return Some(under) };
                        if !state.mask(g) {
                            return None;
                        }
                        let mut old = Vec::new();
                        // (what is read as an operand too is in `under` already)
                        let old_alike = inst.writes.iter().filter(|w| old_only(inst, w.0)).all(|w| {
                            inst.read(w.0).and_then(&value).map(|u| join(&mut old, &u)).is_some()
                        });
                        if old_alike {
                            join(&mut under, &old);
                        } else {
                            join(&mut under, &state.all_of(g)?);
                        }
                        Some(under)
                    };
                    let under = match &inst.fields {
                        Fields::Valu(v) => {
                            // a select on a scalar pair no mask is known for: the lane's own
                            // bit of the number
                            let own_bit = name == "v_cndmask_b32"
                                && match v.src.get(2) {
                                    Some(&crate::gfx10::Operand::Scalar { code, .. }) => {
                                        inst.read(Reg::S(code)).is_none_or(|m| !st.masks.contains_key(&m))
                                    }
                                    _ => true,
                                };
                            let lanes = v.dpp.is_some()
                                || name.starts_with("v_mbcnt")
                                || name.starts_with("v_permlane")
                                || name == "v_writelane_b32"
                                || own_bit;
                            let asks = matches!(name, "v_readfirstlane_b32" | "v_readlane_b32");
                            if lanes {
                                None
                            } else if asks && inst.writes.iter().any(|w| x.res.waterfall.reads.contains(&w.1)) {
                                // a waterfall loop's read is lowered as the lane's own value
                                reads(&every)
                            } else if asks {
                                let operand = reads(&every);
                                // this lane's own where every lane holds it and is in EXEC
                                if name == "v_readfirstlane_b32" && guard.is_none() && operand.as_ref().is_some_and(|u| u.is_empty()) {
                                    own.extend(inst.writes.iter().map(|w| w.1));
                                }
                                if name == "v_readlane_b32" {
                                    // each thread reads the lane it names: one answer where
                                    // the lane's number (every operand but the register
                                    // read, a VGPR as the number too) is one; where the
                                    // number is in the register read, that register too
                                    let vector = |op: Option<&crate::gfx10::Operand>| match op {
                                        Some(&crate::gfx10::Operand::Vector { reg, .. }) => Some(Reg::V(reg)),
                                        _ => None,
                                    };
                                    let lane = vector(v.src.get(1));
                                    let read = vector(v.src.first()).filter(|&r| Some(r) != lane);
                                    reads(&|r, _| Some(r) == read)
                                } else {
                                    // (the wave answers as one at a meeting)
                                    Some(Vec::new())
                                }
                            } else {
                                bits = reads(&every);
                                reads(kept).and_then(guarded)
                            }
                        }
                        // a scalar made from a lane mask read as a number is the wave's answer
                        Fields::Sop { .. } if super::reads_mask_as_data(st, inst) => reads(&|_, v| is_mask(v)),
                        Fields::Sop { .. } => reads(&every),
                        Fields::Smem { .. } => {
                            // (title memory by address is what the program's stores by
                            // address change, between one host wave's load and another's)
                            let memory = x.res.scalar.iter().any(|(s, a)| *s == (b, i) && a.param == crate::resource::MEMORY);
                            // (and title memory is behind every descriptor too: a store
                            // through one the proof knows the piece of may be where it reads)
                            let constant = if name.starts_with("s_buffer_load") {
                                read_only((b, i)) || pieces(x, (b, i)).is_none() && !any_write && written.is_empty()
                            } else {
                                !(memory && (any_write || !written.is_empty()))
                            };
                            if constant { reads(&every) } else { None }
                        }
                        Fields::Buffer { .. } | Fields::Image { .. }
                            if !(name.contains("store") || name.contains("atomic") || name.starts_with("image_bvh")) =>
                        {
                            if read_only((b, i)) { reads(kept).and_then(guarded) } else { None }
                        }
                        _ => None,
                    };
                    for &(_, w) in &inst.writes {
                        // a mask an instruction writes is alike by its lane bit, asked where read
                        found.push((w, if is_mask(w) { same.get(&w).cloned() } else { under.clone() }));
                    }
                }
                for (w, under) in found {
                    match (same.get(&w), under) {
                        (None, _) => {}
                        (Some(_), None) => {
                            same.remove(&w);
                            changed = true;
                        }
                        (Some(old), Some(mut new)) => {
                            join(&mut new, &old.clone());
                            if new.len() != old.len() {
                                same.insert(w, new);
                                changed = true;
                            }
                        }
                    }
                }
                for &(r, w) in &inst.writes {
                    if !matches!(r, Reg::S(_)) {
                        continue;
                    }
                    match (plain.get(&w), &bits) {
                        (None, _) => {}
                        (Some(_), None) => {
                            plain.remove(&w);
                            changed = true;
                        }
                        (Some(old), Some(new)) => {
                            let mut new = new.clone();
                            join(&mut new, &old.clone());
                            if new.len() != old.len() {
                                plain.insert(w, new);
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
        // the branches taken together, and the blocks no other branch keeps a lane from
        let joined = |id: Value| phi_block.get(&id).is_some_and(|&j| entered_together(j, &all));
        let state = State { st, same: &same, plain: &plain, always: &always, joined: &joined };
        // Threads that go round a loop together reach its limit together and all return
        // there: no one is left to wait. Where they do not (the jump back or its header is
        // not a block every thread runs), a thread may return at that jump while others go
        // on, so the jump is an end of the program like `s_endpgm`: a block past it on
        // another way no longer comes after a branch whichever way a thread goes. (Without
        // that end a loop with no way out has every block after it, as nothing leaves it:
        // a meeting beside it stood where the threads in the loop never come.) The blocks
        // found so depend on the ends and the ends on the blocks; both only ever grow.
        // That alone is circular where no way leads to an end at all: the ends wait for a
        // block not every thread runs, and such a block for an end. The ways back of those
        // loops are ends from the start ([`closed_ways_back`]).
        let ends = |b: usize| closed[b] || jumps_back[b].is_some_and(|to| !all[b] || !all[to]);
        let regions =
            lane_regions_by(st, &|d, when| st.wide.contains(&d) || state.mask(when), &ends)?;
        let mut now = all.clone();
        alike = vec![true; n];
        for &(d, ref region) in &regions {
            alike[d] = false;
            for b in (0..n).filter(|&b| region[b]) {
                now[b] = false;
            }
            // The branch's own block, where a way leads back to it (the branch closes a loop
            // or leaves one): its region stops at the block itself, which post-dominates
            // itself, but only the first time does every thread run it. After that the lanes
            // that left are gone, and a barrier there would wait for them.
            let back = |b: usize| match st.exits[b] {
                LaneExit::Jump(t) => t == d,
                LaneExit::Branch { taken, not_taken, .. } => taken == d || not_taken == d,
                LaneExit::End | LaneExit::Indirect => false,
            };
            if back(d) || (0..n).any(|b| region[b] && back(b)) {
                now[d] = false;
                apart[d] = true;
            }
        }
        // (blocks only ever leave: a branch once per-lane stays so)
        if now != all {
            all = now;
            changed = true;
        }
        if !changed {
            break;
        }
    }
    // (A thread that left early would be missed at a barrier: `all` already says so, the
    // region of a per-lane branch reaching to the program's end where one side ends it.)
    // `VICTORY_MEET_WHY=1`: each branch left per-lane, and the first values that keep it so.
    if std::env::var_os("VICTORY_MEET_WHY").is_some() {
        // the guards that are not known to hold in every lane, once each
        let mut guards: Vec<Bool> = Vec::new();
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, inst) in block.insts.iter().enumerate() {
                let Some(g) = st.guards[b][i] else { continue };
                if guards.contains(&g) {
                    continue;
                }
                guards.push(g);
                let atoms: Vec<String> = st
                    .bools
                    .support(g)
                    .into_iter()
                    .map(|a| match a {
                        Atom::Phi(id) => format!(
                            "a join of masks (in every lane: {}; {} ways in, their atoms: {:?})",
                            always.contains(&id),
                            st.phi_masks.get(&id).map_or(0, Vec::len),
                            st.phi_masks.get(&id).map(|e| e.iter().map(|&x| st.bools.support(x).len()).collect::<Vec<_>>())
                        ),
                        Atom::Lane(v) => format!("a compare (alike: {})", plain.contains_key(&v)),
                        Atom::Uniform(v) => format!("a scalar test (alike: {})", same.contains_key(&v)),
                        other => format!("{other:?}"),
                    })
                    .collect();
                eprintln!("MEET-WHY guard first at {:#x} {}: {}", inst.machine.offset, inst.name(), atoms.join(" & "));
            }
        }
        let def: HashMap<Value, &crate::ir::Inst> =
            p.blocks.iter().flat_map(|b| &b.insts).flat_map(|i| i.writes.iter().map(move |w| (w.1, i))).collect();
        for (d, e) in st.exits.iter().enumerate() {
            let LaneExit::Branch { when, .. } = *e else { continue };
            let at = p.blocks[d].insts.last().map_or(p.blocks[d].offset, |i| i.machine.offset);
            let mut line = format!("MEET-WHY branch at {at:#x} (every thread there: {}):", all[d]);
            let mut per_lane = false;
            for a in st.bools.support(when) {
                let what = match a {
                    Atom::Uniform(v) if !same.contains_key(&v) => format!("a scalar test {v:?}"),
                    Atom::Lane(v) if !plain.contains_key(&v) => {
                        let mut s = "a compare".to_owned();
                        // follow the first operand that differs between lanes to where it starts
                        let (mut v, mut steps) = (v, 0);
                        while let Some(i) = def.get(&v)
                            && steps < 12
                        {
                            s.push_str(&format!(" <- {:#x} {}", i.machine.offset, i.name()));
                            let next = i.reads.iter().find(|&&(r, x)| {
                                !exec(r) && x != v && !same.contains_key(&x) && mask_of(x).is_none()
                            });
                            match next {
                                Some(&(r, x)) => {
                                    if !def.contains_key(&x) {
                                        // a join: the first input that differs, or the way in
                                        // that only some lanes take
                                        let join = p.blocks.iter().enumerate().find_map(|(b, blk)| {
                                            blk.phis.iter().find(|phi| phi.value == x).map(|phi| (b, phi))
                                        });
                                        match join {
                                            None => s.push_str(&format!(" <- {r:?} on entry")),
                                            Some((b, phi)) => {
                                                let at = p.blocks[b].offset;
                                                match phi.args.iter().find(|a| !same.contains_key(a)) {
                                                    Some(&a) => {
                                                        s.push_str(&format!(" <- {r:?} joined at {at:#x}"));
                                                        v = a;
                                                        steps += 1;
                                                        continue;
                                                    }
                                                    None => {
                                                        let ways: Vec<String> = p.blocks[b]
                                                            .preds
                                                            .iter()
                                                            .filter(|&&q| !all[q])
                                                            .map(|&q| format!("{:#x}", p.blocks[q].offset))
                                                            .collect();
                                                        s.push_str(&format!(
                                                            " <- {r:?} joined at {at:#x} from blocks not every thread runs: {}",
                                                            ways.join(", ")
                                                        ));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    v = x;
                                }
                                None => break,
                            }
                            steps += 1;
                        }
                        s
                    }
                    Atom::Phi(_) => "a join of masks".to_owned(),
                    Atom::Quad(_) | Atom::Opaque(_) | Atom::Constant(_) => "an opaque or constant mask".to_owned(),
                    _ => continue,
                };
                per_lane = true;
                line.push_str(&format!(" [{what}]"));
            }
            if per_lane {
                eprintln!("{line}");
            }
        }
    }
    Some(Together { all, apart, alike, own })
}

/// A loop the threads of the group do not walk together: its header, and what it fails.
pub(super) type Astray = (usize, &'static str);

/// The rule of the loops. Per block: a loop that holds the block, or that a way leads from
/// to it, which the threads of the group do not walk together (the first found, outermost
/// first); `None` where they walk every such loop together, and only there may a group
/// barrier stand.
///
/// A loop is a block ways lead back to (its header: the target of an edge to a block not
/// after its source in `order`, the lowering's own test in `Lower::run` and `Lower::exit`)
/// with the blocks a way leads from to such an edge without passing the header. The group
/// walks it together when:
/// 1. **entered together**: every thread runs its header (`all`: no branch each lane takes
///    for itself stands between; so all enter it or none);
/// 2. **decided together**: every block of it with an edge out of the loop or back to its
///    header is one every thread runs, and that exit is not a branch each lane takes for
///    itself (`alike`, [`together`]'s proof). An edge back to an outer loop's header is an
///    edge out of this one; an inner loop's are asked of the inner loop, whose blocks this
///    one holds;
/// 3. **counted as the lowering counts**: its header comes before every way back to it on
///    every way from the program's start (else the counter is read where it was not made).
///    (A branch both of whose arms go back is counted on each, `Lower::branch_end`.)
///
/// Why that is enough. Take the threads together at a block every thread runs, their
/// counters equal, in no loop that fails. A jump or a branch taken alike moves them on
/// together, and a counted edge counts the same in each, so the limit sends all or none.
/// At a branch each lane takes for itself they part into its region (the blocks before
/// the first one that comes after the branch on every way), where no edge goes back or
/// leaves a loop without the loop failing (2) and no loop begins without failing (1): a
/// thread there only goes forward, counts nothing, cannot return at a limit, and comes
/// to the region's end, where all are together again with the counters they had. A
/// barrier stands in a block every thread runs, so outside those regions, where the
/// threads are together. And a thread that entered a failing loop is, with every block it
/// can still reach, where no barrier is admitted.
pub(super) fn walked(st: &crate::structure::Structured, order: &[usize], all: &[bool], alike: &[bool]) -> Vec<Option<Astray>> {
    let n = st.exits.len();
    let mut place = vec![usize::MAX; n];
    for (i, &b) in order.iter().enumerate() {
        place[b] = i;
    }
    let succ = |b: usize| -> Vec<usize> {
        match st.exits[b] {
            LaneExit::Jump(t) => vec![t],
            LaneExit::Branch { taken, not_taken, .. } => vec![taken, not_taken],
            LaneExit::End | LaneExit::Indirect => Vec::new(),
        }
    };
    let back = |b: usize, t: usize| place[t] <= place[b];
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &b in order {
        for t in succ(b) {
            if !preds[t].contains(&b) {
                preds[t].push(b);
            }
        }
    }
    // the blocks before a block on every way from the start (bitsets)
    let words = n.div_ceil(64);
    let mut dom: Vec<Vec<u64>> = vec![vec![u64::MAX; words]; n];
    if let Some(&entry) = order.first() {
        dom[entry] = vec![0; words];
        dom[entry][entry / 64] |= 1 << (entry % 64);
    }
    let mut changed = true;
    while changed {
        changed = false;
        for &b in order.iter().skip(1) {
            let mut v = vec![u64::MAX; words];
            for &q in &preds[b] {
                for w in 0..words {
                    v[w] &= dom[q][w];
                }
            }
            v[b / 64] |= 1 << (b % 64);
            if v != dom[b] {
                dom[b] = v;
                changed = true;
            }
        }
    }
    let mut astray: Vec<Option<Astray>> = vec![None; n];
    for &h in order {
        let latches: Vec<usize> = preds[h].iter().copied().filter(|&q| back(q, h)).collect();
        if latches.is_empty() {
            continue;
        }
        let mut body = vec![false; n];
        body[h] = true;
        let mut walk = latches.clone();
        while let Some(b) = walk.pop() {
            if !std::mem::replace(&mut body[b], true) {
                walk.extend(preds[b].iter().copied());
            }
        }
        let decides = |b: usize| latches.contains(&b) || succ(b).into_iter().any(|t| !body[t]);
        let deciding = || (0..n).filter(|&b| body[b] && decides(b));
        let why = if !all[h] {
            "entered by only some lanes"
        } else if deciding().any(|b| !all[b]) {
            "left or gone round from a block only some lanes run"
        } else if deciding().any(|b| !alike[b]) {
            "left or gone round by a branch each lane takes for itself"
        } else if latches.iter().any(|&l| dom[l][h / 64] >> (h % 64) & 1 == 0) {
            "entered past its header"
        } else {
            continue;
        };
        // its blocks, and every block a way leads to from them
        let mut stack: Vec<usize> = (0..n).filter(|&b| body[b]).collect();
        for &b in &stack {
            astray[b].get_or_insert((h, why));
        }
        while let Some(b) = stack.pop() {
            for t in succ(b) {
                if astray[t].is_none() {
                    astray[t] = Some((h, why));
                    stack.push(t);
                }
            }
        }
    }
    astray
}

/// Per block: a way back starts from it (a jump or a branch to a block not after it in
/// `place`, the reverse postorder) and no way leads from it to an end of the program. A
/// thread in such a loop only ever leaves the program at the iteration limit (`Lower::exit`,
/// `abort_block`), so [`together`] takes these blocks for ends whoever goes round the loop.
///
/// Why from the start, and not as the other limit returns, once a block of the loop is one
/// not every thread runs: "comes after a branch whichever way a thread goes" is asked of
/// the ways from the branch to an end, and of a block no way leads from to an end it holds
/// for every block, there being no such way. A branch each lane takes for itself with a
/// loop of this kind on both sides then had every block after it, a meeting on one side
/// included; no block lost "every thread runs it", so no limit became an end, and the
/// threads of the other side were waited for at that meeting after they had returned. With
/// these blocks as ends a way leads from every block to an end (one that never reaches the
/// program's own goes round for ever, so passes a way back from a block that does not
/// either), and what comes after a branch is then asked of ways that exist. The same rule
/// as `first_lane::deciders`, which met the same emptiness.
///
/// Threads that go round such a loop together still reach its limit together, and a
/// meeting inside it or before it stands as before: the end only says that nothing past
/// the way back is reached for certain.
fn closed_ways_back(st: &crate::structure::Structured, place: &[usize]) -> Vec<bool> {
    let n = st.exits.len();
    let succ = |b: usize| -> Vec<usize> {
        match st.exits[b] {
            LaneExit::Jump(t) => vec![t],
            LaneExit::Branch { taken, not_taken, .. } => vec![taken, not_taken],
            LaneExit::End | LaneExit::Indirect => Vec::new(),
        }
    };
    // the blocks a way leads from to an end
    let mut leaves: Vec<bool> = (0..n).map(|b| succ(b).is_empty()).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            if !leaves[b] && succ(b).into_iter().any(|t| leaves[t]) {
                leaves[b] = true;
                changed = true;
            }
        }
    }
    let at = |b: usize| place.get(b).copied().unwrap_or(usize::MAX);
    (0..n).map(|b| !leaves[b] && at(b) != usize::MAX && succ(b).into_iter().any(|t| at(t) <= at(b))).collect()
}

/// A block whose LDS writes the meetings around the blocks only some lanes run do not keep
/// in the console's order, if there is one. Those blocks, joined by the edges between them,
/// are regions the host waves meet before and after ([`super::lds_meetings`]) and not
/// inside: a region with one writing instruction that runs once is in order, whichever
/// thread gets there first. Two writing instructions can run the other way round between
/// two host waves (a thread of one at the second before a thread of another at the first),
/// and so can one that a way inside the region leads back to (two rounds of it).
fn lds_writes_apart(p: &crate::ir::Program, st: &crate::structure::Structured, all: &[bool]) -> Option<usize> {
    let n = p.blocks.len();
    let succ = |b: usize| -> Vec<usize> {
        match st.exits[b] {
            LaneExit::Jump(t) => vec![t],
            LaneExit::Branch { taken, not_taken, .. } => vec![taken, not_taken],
            LaneExit::End | LaneExit::Indirect => Vec::new(),
        }
    };
    let writes = |b: usize| {
        p.blocks[b]
            .insts
            .iter()
            .filter(|i| {
                let name = i.name();
                matches!(i.fields, Fields::Ds { .. })
                    && !(name.starts_with("ds_read") || name.starts_with("ds_swizzle") || name.contains("permute"))
            })
            .count()
    };
    // the regions: blocks not every lane runs, joined by the edges between them
    let mut region: Vec<usize> = (0..n).collect();
    let mut joined = true;
    while joined {
        joined = false;
        for b in (0..n).filter(|&b| !all[b]) {
            for t in succ(b).into_iter().filter(|&t| !all[t]) {
                let low = region[b].min(region[t]);
                if region[b] != low || region[t] != low {
                    (region[b], region[t]) = (low, low);
                    joined = true;
                }
            }
        }
    }
    for w in (0..n).filter(|&b| !all[b] && writes(b) > 0) {
        let others: usize = (0..n).filter(|&b| !all[b] && b != w && region[b] == region[w]).map(writes).sum();
        if writes(w) > 1 || others > 0 {
            return Some(w);
        }
        // a way back to the one write, inside the region
        let mut seen = vec![false; n];
        let mut stack = succ(w);
        while let Some(b) = stack.pop() {
            if b == w {
                return Some(w);
            }
            if !all[b] && !std::mem::replace(&mut seen[b], true) {
                stack.extend(succ(b));
            }
        }
    }
    None
}

impl Meet {
    /// The program's own LDS, in dwords: the form's dwords follow it.
    pub(super) fn program_lds(&self) -> u32 {
        self.slots
    }
}

/// The meeting form's state in a lowering ([`Lower::meet`]).
pub(super) struct Meet {
    /// The first of the threads' dwords in group-shared memory (one per thread, whole waves).
    slots: u32,
    /// The first of the console waves' pairs of dwords (the lanes' bits).
    bits: u32,
    /// Threads in the group.
    threads: u32,
    /// Per IR block: every thread of the group runs it ([`together`]).
    pub all: Vec<bool>,
    /// Per IR block: the lanes leave its loop one by one ([`Together::apart`]).
    apart: Vec<bool>,
    /// Per IR block: a loop on a way to it that the threads do not walk together ([`walked`]).
    astray: Vec<Option<Astray>>,
    /// The two dwords after the meeting's own: what the program reads past the end of its
    /// LDS (kept 0) and what it writes there ([`Lower::meet_bound`]).
    spare: u32,
    /// The first of them has been cleared in the entry block.
    cleared: bool,
    /// First-lane reads that are the lane's own ([`Together::own`]).
    pub own: HashSet<Value>,
    /// The headers of the loops closed by a plain jump that the lanes do not go round
    /// together: a lane still in one at the iteration limit returns (`abort_block`).
    pub cut: Vec<usize>,
    /// The IR blocks a meeting stands in.
    pub met: HashSet<usize>,
    /// A meeting asked for where not every thread comes: what the instruction being
    /// lowered stops with.
    pub fail: Option<Why>,
    /// The value a whole-wave section leaves that is being lowered: its block, the
    /// section's start and the value ([`Lower::meet_leaving`]).
    leaving: Option<(usize, usize, crate::ir::Value)>,
    /// Made in the entry block: the thread's index in the group, its console lane, its
    /// dword, its wave's first dword, its wave's pair, the dword of the pair that holds its
    /// bit and the bit.
    flat: Option<V>,
    lane: Option<V>,
    slot: Option<V>,
    wave: Option<V>,
    pair: Option<V>,
    bit: Option<(V, V)>,
}

impl Meet {
    /// The scratch a group of `threads` threads needs after `lds_dwords` dwords of the
    /// program's own shared memory: the form's state and the dwords it adds. `None` where
    /// group-shared memory has no room for it (32 KiB a group on the host).
    pub(super) fn new(
        lds_dwords: u32,
        threads: u32,
        found: Together,
        astray: Vec<Option<Astray>>,
        cut: Vec<usize>,
    ) -> Option<(Meet, u32)> {
        let Together { all, apart, own, .. } = found;
        let waves = threads.div_ceil(64);
        let added = 64 * waves + 2 * waves + 2;
        if (lds_dwords + added) * 4 > 32768 {
            return None;
        }
        let meet = Meet {
            slots: lds_dwords,
            bits: lds_dwords + 64 * waves,
            threads,
            all,
            apart,
            astray,
            spare: lds_dwords + 66 * waves,
            cleared: false,
            own,
            cut,
            met: HashSet::new(),
            fail: None,
            leaving: None,
            flat: None,
            lane: None,
            slot: None,
            wave: None,
            pair: None,
            bit: None,
        };
        Some((meet, added))
    }
}

/// The meeting form for a program of `group` threads with `lds_dwords` dwords of LDS: its
/// state, the dwords of group-shared memory it adds, and where the host waves meet around
/// lane branches with LDS accesses (`lds_meetings`); or why the form cannot take the program.
#[allow(clippy::type_complexity)]
pub(super) fn plan(
    x: &Parts,
    one_wave: bool,
    group: u32,
    lds_dwords: u32,
) -> Result<(Meet, u32, Option<(Vec<bool>, Vec<(bool, bool)>)>), Why> {
    let (p, st) = (x.p, x.st);
    let lds = |i: &crate::ir::Inst| {
        matches!(i.fields, Fields::Ds { .. }) && !(i.name().starts_with("ds_swizzle") || i.name().contains("permute"))
    };
    // (a group of several console waves: their waves decide apart, see `SEVERAL_WAVES`)
    if !one_wave {
        return Err(SEVERAL_WAVES);
    }
    let t = together(x).ok_or("an indirect exit in the meeting form")?;
    let order = crate::ir::reverse_postorder(&p.blocks);
    // The rule of the loops ([`walked`]), asked of every group barrier of the form before
    // anything else is: here of the program's own and of the ones before its LDS accesses,
    // of a meeting where it is lowered ([`Lower::meet_in`]). What it finds is said in the
    // form's record whichever check refuses the program in the end.
    let astray = walked(st, &order, &t.all, &t.alike);
    if std::env::var_os("VICTORY_MEET_WHY").is_some() {
        let mut said: Vec<usize> = Vec::new();
        for &(h, why) in astray.iter().flatten() {
            if !said.contains(&h) {
                said.push(h);
                eprintln!("MEET-WHY loop at {:#x} (block {h}) not walked together: {why}", p.blocks[h].offset);
            }
        }
    }
    let strays = |b: usize, i: &crate::ir::Inst, what: &str| -> bool {
        let Some((h, why)) = astray[b] else { return false };
        let (offset, name, at) = (i.machine.offset, i.name(), p.blocks[h].offset);
        crate::emit::note_form(|| format!("not walked together: {offset:#x} {name}: {what}: the loop at {at:#x} is {why}"));
        true
    };
    let mut stray = false;
    for (b, blk) in p.blocks.iter().enumerate() {
        if let Some(i) = blk.insts.iter().find(|i| i.name() == "s_barrier") {
            stray |= strays(b, i, "a barrier of the program's own");
        }
        // (a barrier before each LDS access in a block every thread runs, `Lower::inst`)
        if let Some(i) = blk.insts.iter().find(|i| lds(i)).filter(|_| lds_dwords > 0 && t.all[b]) {
            stray |= strays(b, i, "a barrier before an LDS access");
        }
    }
    // The program's own barrier waits for the group as a meeting's does (`s_barrier` with
    // the default group sync): only where every thread comes.
    for (b, blk) in p.blocks.iter().enumerate().filter(|&(b, _)| !t.all[b]) {
        if let Some(i) = blk.insts.iter().find(|i| i.name() == "s_barrier") {
            need(b);
            let offset = i.machine.offset;
            crate::emit::note_form(|| format!("needs every lane: {offset:#x} s_barrier: a barrier of the program's own"));
            return Err(OWN_BARRIER);
        }
    }
    // A lane cut out of a loop at the iteration limit returns (`abort_block`) and would be
    // missed at a later barrier: the loops it is cut out of (`crate::bounds::cut_at_limit`)
    // that the threads do not go round together ([`Lower::meet_missed`] asks once the
    // meetings are known).
    let mut place = vec![usize::MAX; p.blocks.len()];
    for (n, &b) in order.iter().enumerate() {
        place[b] = n;
    }
    let cut: Vec<usize> =
        (0..p.blocks.len()).filter(|&b| !t.all[b]).filter_map(|b| crate::bounds::cut_at_limit(p, st, &place, b, true)).collect();
    // LDS the console shares within its wave without barriers: a group barrier before every
    // access keeps the order between host waves where every thread comes, and around the
    // lane branches whose regions only read or only write (`lds_meetings`).
    let inside = lds_dwords > 0 && p.blocks.iter().enumerate().any(|(b, blk)| !t.all[b] && blk.insts.iter().any(lds));
    let meetings = if inside { super::lds_meetings(p, st, &t.all) } else { None };
    if inside && meetings.is_none() {
        for (b, blk) in p.blocks.iter().enumerate() {
            if !t.all[b] && blk.insts.iter().any(lds) {
                need(b);
            }
        }
        crate::emit::note_form(|| "needs every lane: LDS read and write in one lane branch".to_owned());
        return Err("an LDS read and write where not every lane runs");
    }
    if inside && let Some(b) = lds_writes_apart(p, st, &t.all) {
        need(b);
        let at = p.blocks[b].insts.iter().find(|i| lds(i) && !i.name().starts_with("ds_read"));
        let (offset, name) = at.map_or((p.blocks[b].offset, "?"), |i| (i.machine.offset, i.name()));
        crate::emit::note_form(|| format!("needs every lane: {offset:#x} {name}: LDS writes in one lane branch"));
        return Err(LDS_WRITES);
    }
    // (the barriers around a lane branch with LDS accesses, `Lower::lds_meeting`)
    if let Some(around) = &meetings {
        for (b, blk) in p.blocks.iter().enumerate().filter(|&(b, _)| around[b].0 || around[b].1) {
            stray |= match blk.insts.first() {
                Some(i) => strays(b, i, "a barrier around a lane branch with LDS accesses"),
                None => astray[b].is_some(),
            };
        }
    }
    if stray {
        return Err(ASTRAY);
    }
    let all = t.all.clone();
    let (m, added) = Meet::new(lds_dwords, group, t, astray, cut).ok_or("no room in group-shared memory for a meeting")?;
    Ok((m, added, meetings.map(|m| (all, m))))
}

impl Lower<'_> {
    /// The end of a lowering in the meeting form: no meeting was left without its threads,
    /// no thread can be missed at a barrier, and no host wave is asked (an operation that
    /// slipped through would answer for its own host wave only).
    pub(super) fn meet_done(&mut self) -> Result<(), Why> {
        let Some(m) = &self.meet else { return Ok(()) };
        if let Some(why) = m.fail {
            return Err(why);
        }
        if self.meet_missed() {
            return Err("a loop a lane can be cut out of before a meeting");
        }
        if self.s.wave_calls != 0 {
            return Err("a host wave operation in the meeting form");
        }
        self.s.flags &= !crate::emit::shader::flags::WAVE_OPS;
        Ok(())
    }

    /// The console's decision of the branch closing IR block `b`, which the wave takes
    /// together (`Lower::wave_wide`), asked of its whole wave at a meeting: what the lanes
    /// raise is "some lane speaks against" either way (`any`: some lane's `when` holds;
    /// else: some lane's does not).
    pub(super) fn meet_wide(&mut self, b: usize, when: Bool, any: bool) -> Result<V, Why> {
        if !self.meet_in(b, "a wave-wide branch") {
            return Err(self.meet_why());
        }
        let c = self.cur;
        let mine = self.boolean(when, c)?;
        let set = if any {
            mine
        } else {
            let t = self.s.m.bool(true);
            self.s.m.bin(c, Bin::Xor, mine, t)
        };
        let (lo, hi) = self.meet_mask(c, set);
        let both = self.s.m.bin(c, Bin::Or, lo, hi);
        let z = self.s.m.i32(0);
        Ok(self.s.m.cmp(c, if any { pred::NE } else { pred::EQ }, both, z))
    }

    /// Whether some thread could be missed at a group barrier of this program: a lane cut
    /// out of a loop at the iteration limit returns, and a barrier that can still be reached
    /// from that loop (a meeting, a barrier before an LDS access, the title's own) would
    /// wait for it. Asked once the program is lowered and its meetings are known.
    fn meet_missed(&self) -> bool {
        let Some(m) = &self.meet else { return false };
        if m.cut.is_empty() {
            return false;
        }
        let (p, st) = (self.x.p, self.x.st);
        let lds = |i: &crate::ir::Inst| {
            matches!(i.fields, Fields::Ds { .. }) && !(i.name().starts_with("ds_swizzle") || i.name().contains("permute"))
        };
        let waits = |b: usize| {
            m.met.contains(&b)
                || self.lds_meet.as_ref().is_some_and(|(_, meet)| meet[b].0 || meet[b].1)
                || p.blocks[b].insts.iter().any(|i| i.name() == "s_barrier" || self.split_lds && lds(i))
        };
        let mut seen = vec![false; p.blocks.len()];
        let mut stack = m.cut.clone();
        while let Some(b) = stack.pop() {
            if std::mem::replace(&mut seen[b], true) {
                continue;
            }
            if waits(b) {
                return true;
            }
            match st.exits[b] {
                LaneExit::Jump(t) => stack.push(t),
                LaneExit::Branch { taken, not_taken, .. } => stack.extend([taken, not_taken]),
                LaneExit::End | LaneExit::Indirect => {}
            }
        }
        false
    }

    /// What a meeting [`Lower::meet_in`] just refused stops with, for the callers that name
    /// the refusal themselves: the rule of the loops' own reason, and for a block not every
    /// thread runs the one reason they gave before. Takes [`Meet::fail`].
    pub(super) fn meet_why(&mut self) -> Why {
        match self.meet.as_mut().and_then(|m| m.fail.take()) {
            Some(ASTRAY) => ASTRAY,
            _ => NOT_EVERY,
        }
    }

    /// Whether a meeting may stand in IR block `b`: every thread of the group runs it, and
    /// walks together every loop on a way to it ([`walked`]). Else
    /// the block is named for the forms that bring the lanes there, the census is told what
    /// stood there (`idiom`), and the instruction being lowered stops ([`Meet::fail`]).
    pub(super) fn meet_in(&mut self, b: usize, idiom: &str) -> bool {
        let Some(m) = self.meet.as_mut() else { return false };
        let (offset, name) = self.at;
        // The rule of the loops first ([`walked`]): said in the form's record whether or not
        // the block is one every thread runs, which names the refusal where it is not.
        let stray = m.astray.get(b).copied().flatten();
        if let Some((h, why)) = stray {
            let at = self.x.p.blocks[h].offset;
            crate::emit::note_form(|| format!("not walked together: {offset:#x} {name}: {idiom}: the loop at {at:#x} is {why}"));
        }
        let every = m.all.get(b).copied().unwrap_or(false);
        if every && stray.is_none() {
            m.met.insert(b);
            crate::emit::note_form(|| format!("met: {offset:#x} {name}: {idiom}"));
            return true;
        }
        if every {
            m.fail = Some(ASTRAY);
            return false;
        }
        let apart = m.apart.get(b).copied().unwrap_or(false);
        m.fail = Some(if apart { APART } else { NOT_EVERY });
        need(b);
        let how = if apart { ", in a loop the lanes leave one by one" } else { "" };
        crate::emit::note_form(|| format!("needs every lane: {offset:#x} {name}: {idiom}{how}"));
        false
    }

    /// The thread's index in the group (the console's thread index: no host wave is asked).
    fn meet_flat(&mut self) -> V {
        if let Some(v) = self.meet.as_ref().and_then(|m| m.flat) {
            return v;
        }
        let (e, i32t) = (self.s.entry, self.s.t.i32);
        let v = self.s.op(e, "flattenedThreadIdInGroup.i32", Attr::ReadNone, i32t, 96, &[]);
        self.meet.as_mut().expect("meeting form").flat = Some(v);
        v
    }

    /// The thread's console lane (0..63): its index in the group modulo 64.
    pub(super) fn meet_lane(&mut self) -> V {
        if let Some(v) = self.meet.as_ref().and_then(|m| m.lane) {
            return v;
        }
        let e = self.s.entry;
        let flat = self.meet_flat();
        let k63 = self.s.m.i32(63);
        let v = self.s.m.bin(e, Bin::And, flat, k63);
        self.meet.as_mut().expect("meeting form").lane = Some(v);
        v
    }

    /// The index of the thread's own dword, and of its wave's first.
    fn meet_slots(&mut self) -> (V, V) {
        if let Some(m) = self.meet.as_ref()
            && let (Some(s), Some(w)) = (m.slot, m.wave)
        {
            return (s, w);
        }
        let e = self.s.entry;
        let flat = self.meet_flat();
        let base = self.s.m.i32(self.meet.as_ref().expect("meeting form").slots);
        let slot = self.s.m.bin(e, Bin::Add, base, flat);
        let k = self.s.m.i32(!63);
        let first = self.s.m.bin(e, Bin::And, flat, k);
        let wave = self.s.m.bin(e, Bin::Add, base, first);
        let m = self.meet.as_mut().expect("meeting form");
        (m.slot, m.wave) = (Some(slot), Some(wave));
        (slot, wave)
    }

    /// The index of the wave's pair of dwords, the dword of the two that holds this lane's
    /// bit, and the bit. The first call clears the pair in the entry block, and the group
    /// waits there: shared memory starts with what the host left in it.
    fn meet_pair(&mut self) -> (V, V, V) {
        if let Some(m) = self.meet.as_ref()
            && let (Some(p), Some((w, b))) = (m.pair, m.bit)
        {
            return (p, w, b);
        }
        let e = self.s.entry;
        let (flat, lane) = (self.meet_flat(), self.meet_lane());
        let [k1, k5, k6, k31, z] = [1, 5, 6, 31, 0].map(|k| self.s.m.i32(k));
        let base = self.s.m.i32(self.meet.as_ref().expect("meeting form").bits);
        let wave = self.s.m.bin(e, Bin::LShr, flat, k6);
        let twice = self.s.m.bin(e, Bin::Shl, wave, k1);
        let pair = self.s.m.bin(e, Bin::Add, base, twice);
        let half = self.s.m.bin(e, Bin::LShr, lane, k5);
        let word = self.s.m.bin(e, Bin::Add, pair, half);
        let sh = self.s.m.bin(e, Bin::And, lane, k31);
        let bit = self.s.m.bin(e, Bin::Shl, k1, sh);
        let lds = self.s.lds();
        let high = self.s.m.bin(e, Bin::Add, pair, k1);
        for index in [pair, high] {
            let ptr = self.s.m.element(e, lds, index);
            self.s.m.store(e, ptr, z);
        }
        self.group_sync(e);
        let m = self.meet.as_mut().expect("meeting form");
        (m.pair, m.bit) = (Some(pair), Some((word, bit)));
        (pair, word, bit)
    }

    /// The dword of group-shared memory an LDS access of the program's at dword `index`
    /// goes to: that dword inside the program's own LDS. Past its end the console reads 0
    /// and drops what is written, and the meeting's dwords lie there: a read goes to a dword
    /// kept 0 (every thread clears it in the entry block, and nothing else is stored to
    /// it), anything else to a dword no one reads. Only in the meeting form.
    pub(super) fn meet_bound(&mut self, at: usize, index: V, read: bool) -> V {
        let Some(m) = self.meet.as_mut() else { return index };
        let (own, spare) = (m.slots, m.spare + u32::from(!read));
        let clear = read && !std::mem::replace(&mut m.cleared, true);
        let (end, spare) = (self.s.m.i32(own), self.s.m.i32(spare));
        if clear {
            let (e, z, lds) = (self.s.entry, self.s.m.i32(0), self.s.lds());
            let ptr = self.s.m.element(e, lds, spare);
            self.s.m.store(e, ptr, z);
        }
        let inside = self.s.m.cmp(at, pred::IULT, index, end);
        self.s.m.select(at, inside, index, spare)
    }

    /// An LDS atomic of the program's at byte `offset` from the address `a`, which runs
    /// under `guard` and gives the lanes outside it `outside`: the guard it runs under in
    /// the meeting form and what the lanes that do not run it get. Past the end of the
    /// program's LDS the console reads 0 and drops the write, so there the atomic is not
    /// run at all and a lane in EXEC gets 0. ([`Lower::meet_bound`] alone sent it to the
    /// dword the writes past the end go to, and the value it returned was what another
    /// lane or an earlier instruction had left there.) A lane outside EXEC keeps `outside`
    /// as before. Cost: a compare, an `and` and a select per atomic. Other forms: as given.
    pub(super) fn meet_atomic(&mut self, a: V, offset: u32, guard: Option<V>, outside: V) -> (Option<V>, V) {
        let Some(m) = self.meet.as_ref() else { return (guard, outside) };
        let (at, end) = (self.cur, self.s.m.i32(m.slots));
        let (o, two, z) = (self.s.m.i32(offset), self.s.m.i32(2), self.s.m.i32(0));
        let byte = self.s.m.bin(at, Bin::Add, a, o);
        let index = self.s.m.bin(at, Bin::LShr, byte, two);
        let inside = self.s.m.cmp(at, pred::IULT, index, end);
        match guard {
            None => (Some(inside), z),
            Some(g) => (Some(self.s.m.bin(at, Bin::And, g, inside)), self.s.m.select(at, g, z, outside)),
        }
    }

    /// This thread's dword takes `x`.
    fn meet_put(&mut self, at: usize, x: V) {
        let (slot, _) = self.meet_slots();
        let lds = self.s.lds();
        let ptr = self.s.m.element(at, lds, slot);
        self.s.m.store(at, ptr, x);
    }

    /// The dword of console lane `lane` (0..63) of this thread's wave; `absent` for a lane
    /// the group has no thread for (a last wave that is not full).
    fn meet_get(&mut self, at: usize, lane: V, absent: V) -> V {
        let (_, wave) = self.meet_slots();
        let lds = self.s.lds();
        let index = self.s.m.bin(at, Bin::Add, wave, lane);
        let ptr = self.s.m.element(at, lds, index);
        let x = self.s.m.load(at, ptr);
        let (first, threads) = {
            let m = self.meet.as_ref().expect("meeting form");
            (m.slots, m.threads)
        };
        if threads.is_multiple_of(64) {
            return x;
        }
        let end = self.s.m.i32(first + threads);
        let there = self.s.m.cmp(at, pred::IULT, index, end);
        self.s.m.select(at, there, x, absent)
    }

    /// This lane's bit goes into its wave's pair where `on` holds (before the wait).
    fn meet_raise(&mut self, at: usize, on: V) {
        let (_, word, bit) = self.meet_pair();
        let z = self.s.m.i32(0);
        let add = self.s.m.select(at, on, bit, z);
        let lds = self.s.lds();
        let ptr = self.s.m.element(at, lds, word);
        self.s.m.atomic(at, Rmw::Or, ptr, add);
    }

    /// The wave's pair: the bits of lanes 0..31 and of lanes 32..63 (between the waits).
    fn meet_raised(&mut self, at: usize) -> (V, V) {
        let (pair, _, _) = self.meet_pair();
        let lds = self.s.lds();
        let one = self.s.m.i32(1);
        let high = self.s.m.bin(at, Bin::Add, pair, one);
        let (p0, p1) = (self.s.m.element(at, lds, pair), self.s.m.element(at, lds, high));
        (self.s.m.load(at, p0), self.s.m.load(at, p1))
    }

    /// This lane takes its bit back (after the second wait: no one reads the pair then, and
    /// the other lanes' bits are not touched), so the pair is clear for the next meeting.
    fn meet_lower(&mut self, at: usize) {
        let (_, word, bit) = self.meet_pair();
        let ones = self.s.m.i32(u32::MAX);
        let keep = self.s.m.bin(at, Bin::Xor, bit, ones);
        let lds = self.s.lds();
        let ptr = self.s.m.element(at, lds, word);
        self.s.m.atomic(at, Rmw::And, ptr, keep);
    }

    /// The console wave's mask of the lanes where `on` holds, as its two dwords: every
    /// thread of the group must run `at`.
    pub(super) fn meet_mask(&mut self, at: usize, on: V) -> (V, V) {
        self.meet_raise(at, on);
        self.group_sync(at);
        let xy = self.meet_raised(at);
        self.group_sync(at);
        self.meet_lower(at);
        xy
    }

    /// `x` as console lane `lane` of this thread's wave holds it.
    pub(super) fn meet_read(&mut self, at: usize, x: V, lane: V) -> V {
        self.meet_put(at, x);
        self.group_sync(at);
        let k63 = self.s.m.i32(63);
        let lane = self.s.m.bin(at, Bin::And, lane, k63);
        let z = self.s.m.i32(0);
        let r = self.meet_get(at, lane, z);
        self.group_sync(at);
        r
    }

    /// `x` as the console wave's first lane where `on` holds has it (lane 0 with no such
    /// lane, as the hardware reads it), and whether there is such a lane.
    pub(super) fn meet_first(&mut self, at: usize, x: V, on: Option<V>) -> (V, V) {
        let on = on.unwrap_or_else(|| self.s.m.bool(true));
        self.meet_put(at, x);
        self.meet_raise(at, on);
        self.group_sync(at);
        let (lo, hi) = self.meet_raised(at);
        let i32t = self.s.t.i32;
        let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
        // FirstbitLo (dx.op 32): the lowest set bit
        let flo = self.s.op(at, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[lo]);
        let fhi = self.s.op(at, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[hi]);
        let fhi = self.s.m.bin(at, Bin::Add, fhi, k32);
        let in_lo = self.s.m.cmp(at, pred::NE, lo, z);
        let in_hi = self.s.m.cmp(at, pred::NE, hi, z);
        let any = self.s.m.bin(at, Bin::Or, in_lo, in_hi);
        let lane = self.s.m.select(at, in_hi, fhi, z);
        let lane = self.s.m.select(at, in_lo, flo, lane);
        let r = self.meet_get(at, lane, z);
        self.group_sync(at);
        self.meet_lower(at);
        (r, any)
    }

    /// `op` of two values as the console's VALU instruction of that name gives it.
    fn meet_op(&mut self, at: usize, op: Op, a: V, b: V) -> V {
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let (a, b) = if float {
            (self.s.m.cast(at, Cast::Bitcast, a, f32t), self.s.m.cast(at, Cast::Bitcast, b, f32t))
        } else {
            (a, b)
        };
        let r = match op {
            Op::AddU | Op::AddF | Op::AddH => self.s.m.bin(at, Bin::Add, a, b),
            Op::Or => self.s.m.bin(at, Bin::Or, a, b),
            Op::And => self.s.m.bin(at, Bin::And, a, b),
            Op::Xor => self.s.m.bin(at, Bin::Xor, a, b),
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
                let (ty, sfx) = if float { (f32t, "f32") } else { (i32t, "i32") };
                self.s.op(at, &format!("binary.{sfx}"), Attr::ReadNone, ty, code, &[a, b])
            }
        };
        if float { self.s.m.cast(at, Cast::Bitcast, r, i32t) } else { r }
    }

    /// The value a whole-wave section leaves that is lowered next: `w`, of the section
    /// starting at instruction `start` of IR block `b`.
    pub(super) fn meet_leaving(&mut self, b: usize, start: usize, w: crate::ir::Value) {
        if let Some(m) = self.meet.as_mut() {
            m.leaving = Some((b, start, w));
        }
    }

    /// A sum of floats added up by `order`, the steps the program's instructions add it by:
    /// the terms are the wave's dwords (`id` for a lane the group has no thread for), and
    /// where the lanes hold different sums (a prefix) each thread takes its lane's.
    fn meet_sum(&mut self, at: usize, order: &Order, id: V) -> V {
        let mut made: Vec<V> = Vec::with_capacity(order.steps.len());
        for step in &order.steps {
            let v = match *step {
                Step::Lane(lane) => {
                    let k = self.s.m.i32(u32::from(lane));
                    self.meet_get(at, k, id)
                }
                Step::K(k) => self.s.m.i32(k),
                Step::Add(a, b) => self.meet_op(at, Op::AddF, made[a as usize], made[b as usize]),
            };
            made.push(v);
        }
        let mut r = made[order.results[0] as usize];
        if order.results.len() > 1 {
            let me = self.meet_lane();
            for (lane, &step) in order.results.iter().enumerate().skip(1) {
                let k = self.s.m.i32(lane as u32);
                let mine = self.s.m.cmp(at, pred::EQ, me, k);
                r = self.s.m.select(at, mine, made[step as usize], r);
            }
        }
        r
    }

    /// A value a whole-wave section leaves, over the console's wave: `x` is each lane's term
    /// (the operation's identity, `ident`, where the lane is not in the section's mask).
    /// A reduction folds the wave's 64 dwords pairwise in lane order; a prefix (`Some`:
    /// whether the lane's own term counts) folds the lanes below in their order. A sum of
    /// floats, where the order of the additions is part of the result, is added up as the
    /// section adds it ([`Lower::meet_sum`]).
    pub(super) fn meet_shape(&mut self, op: Op, prefix: Option<bool>, x: V, ident: u32) -> Result<V, Why> {
        if op == Op::AddH {
            return Err("a wave sum of 16-bit floats in the meeting form");
        }
        if prefix.is_some() && !matches!(op, Op::AddU | Op::AddF) {
            return Err("a wave prefix of an operation other than a sum");
        }
        let order = if op == Op::AddF {
            let kept = self.meet.as_ref().and_then(|m| m.leaving).and_then(|(b, start, w)| {
                self.sections.get(&b)?.get(&start)?.orders.get(&w).cloned()
            });
            Some(kept.filter(|o| o.steps.len() <= FLOAT_STEPS).ok_or(FLOAT_ORDER)?)
        } else {
            None
        };
        let b = self.cur_ir;
        let mut what = format!("{} {op:?}", if prefix.is_some() { "prefix" } else { "reduction" });
        if let Some(order) = &order {
            // (for the census: how the program adds it up)
            let how = if order.neighbours_first() { "neighbours first" } else { "another order" };
            what.push_str(&format!(" in the program's order, {} steps, {how}", order.steps.len()));
        }
        if !self.meet_in(b, &what) {
            return Err(self.meet_why());
        }
        let at = self.cur;
        self.meet_put(at, x);
        self.group_sync(at);
        let id = self.s.m.i32(ident);
        if let Some(order) = &order {
            let r = self.meet_sum(at, order, id);
            self.group_sync(at);
            return Ok(r);
        }
        let mut terms: Vec<V> = (0..64)
            .map(|lane| {
                let k = self.s.m.i32(lane);
                self.meet_get(at, k, id)
            })
            .collect();
        let r = match prefix {
            None => {
                while terms.len() > 1 {
                    terms = terms.chunks(2).map(|c| self.meet_op(at, op, c[0], c[1])).collect();
                }
                terms[0]
            }
            Some(inclusive) => {
                let me = self.meet_lane();
                let mut sum = id;
                for (lane, &term) in terms.iter().enumerate() {
                    let k = self.s.m.i32(lane as u32);
                    let below = self.s.m.cmp(at, pred::IULT, k, me);
                    let term = self.s.m.select(at, below, term, id);
                    sum = self.meet_op(at, op, sum, term);
                }
                if inclusive { self.meet_op(at, op, sum, x) } else { sum }
            }
        };
        self.group_sync(at);
        Ok(r)
    }
}
