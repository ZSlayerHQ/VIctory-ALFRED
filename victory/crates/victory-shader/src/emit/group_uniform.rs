//! Which blocks of a console compute program every thread of its thread group reaches, and
//! the same number of times (the census of waits). A group barrier is the only
//! cross-wave wait D3D12 defines, and only where every thread of the group arrives: the
//! translation's waits ([`super::lower`]'s meetings of the two host waves of a console wave,
//! and the title's own `s_barrier`s) may stand only in such blocks when the meetings are
//! group barriers ([`super::pair_meetings_barrier`]). Without pair meetings, a varying
//! first-lane read's exchange (its group barriers) is emitted only in such blocks too.
//!
//! A block is group-uniform when no branch whose region holds it (the blocks reachable from
//! the branch that do not post-dominate it) has a condition shown alike in every thread of the
//! group. A loop whose latch is such a branch holds its whole body in the latch's region, so a
//! group-uniform block is reached the same number of times too. A block no path leads from to
//! an End is not group-uniform, and for the regions it has an exit of its own. Alike: literals, user SGPRs,
//! the thread-group id, and scalar arithmetic and compares on these; a phi only where its
//! block is reached alike and so are all its predecessors (which edge brings the value is then
//! the same everywhere), or where every edge brings one value; a scalar load only from a
//! read-only view (a constant or shader-resource buffer, a descriptor table, the code) at an
//! address made of such values. Not alike: anything a lane holds or that names lanes (VGPRs,
//! EXEC, VCC, lane masks, `v_readfirstlane`, ballots), LDS and writable memory, the wave's own
//! index in its group, and anything else. Two console waves can compare different scalar loads
//! and set opposite SCCs, so a branch on SCC alone is not taken alike by the group.
//!
//! Optimistic: every candidate value starts alike and loses that when a source, its block or a
//! predecessor of its phi is not (a divergence analysis), until nothing changes.

use std::collections::HashSet;

use crate::gfx10::Fields;
use crate::ir::{EXEC, Inst, Program, Reg, VCC, Value};
use crate::native::{NativeProgram, ResourceType, RootType, StageIo};
use crate::resource::{CODE, MEMORY, Resources, Role, Site, Source};
use crate::structure::{Atom, LaneExit, Structured};

/// Per IR block: `None` where every thread of the group reaches it alike; else the branch
/// (its block) whose region holds it, and why that branch's condition is not shown alike.
pub(super) type Census = Vec<Option<(usize, &'static str)>>;

pub(super) const WIDE: &str = "a branch each console wave takes as a whole";
pub(super) const LANE: &str = "a per-lane condition";
pub(super) const SCALAR: &str = "a scalar condition not shown alike in every console wave";
/// (the block itself in place of a branch)
pub(super) const NO_END: &str = "a block no path leads from to the program's end";

/// [`Census`] of `p`; None with an indirect exit or outside compute.
pub(super) fn blocks(p: &Program, st: &Structured, native: &NativeProgram, res: &Resources) -> Option<Census> {
    let StageIo::Cs(cs) = native.io else { return None };
    let regions = super::lower::branch_regions(st, true, |_, _| false)?;
    let live = super::lower::reaches_end(st);
    let n = st.exits.len().min(p.blocks.len());
    let (ids, _) = cs.system_sgprs();
    let lane_reg = |r: &Reg| matches!(*r, Reg::S(k) if [EXEC, EXEC + 1, VCC, VCC + 1].contains(&k));
    let mut alike: HashSet<Value> = p
        .inputs
        .iter()
        .filter(|(r, _)| matches!(*r, Reg::S(k) if k < cs.user_sgprs() || ids.contains(&Some(k))))
        .map(|&(_, v)| v)
        .collect();
    for (b, blk) in p.blocks.iter().enumerate().take(n) {
        alike.extend(blk.phis.iter().filter(|phi| !matches!(phi.reg, Reg::V(_))).map(|phi| phi.value));
        for (i, inst) in blk.insts.iter().enumerate() {
            let lanes = inst.reads.iter().chain(&inst.writes).any(|(r, _)| lane_reg(r));
            if !lanes && scalar_alike(inst, (b, i), native, res) {
                alike.extend(inst.writes.iter().map(|w| w.1));
            }
        }
    }
    alike.retain(|v| !st.masks.contains_key(v) && !st.highs.contains_key(v));
    loop {
        // A loop with no way out is left only at the iteration limit, by threads apart.
        let mut why: Census = (0..n).map(|b| (!live[b]).then_some((b, NO_END))).collect();
        for (d, region) in &regions {
            let LaneExit::Branch { when, .. } = st.exits[*d] else { continue };
            let not = if st.wide.contains(d) {
                Some(WIDE)
            } else {
                st.bools.support(when).into_iter().find_map(|a| match a {
                    Atom::Uniform(v) if alike.contains(&v) => None,
                    Atom::Uniform(_) => Some(SCALAR),
                    _ => Some(LANE),
                })
            };
            if let Some(r) = not {
                for (b, w) in why.iter_mut().enumerate() {
                    if region[b] && w.is_none() {
                        *w = Some((*d, r));
                    }
                }
            }
        }
        let before = alike.len();
        for b in 0..n {
            let blk = &p.blocks[b];
            let bad = why[b].is_some();
            for phi in &blk.phis {
                let one = phi.args.windows(2).all(|w| w[0] == w[1]);
                let preds = blk.preds.iter().all(|&q| why.get(q).is_some_and(Option::is_none));
                if bad || !(one || preds) || !phi.args.iter().all(|a| alike.contains(a)) {
                    alike.remove(&phi.value);
                }
            }
            for inst in &blk.insts {
                if bad || inst.reads.iter().any(|r| !alike.contains(&r.1)) {
                    for w in &inst.writes {
                        alike.remove(&w.1);
                    }
                }
            }
        }
        if alike.len() == before {
            return Some(why);
        }
    }
}

/// Whether `inst` (at `site`) computes only from what it reads: a scalar ALU instruction (not
/// a hardware register read), or a scalar load from a view the program cannot write.
fn scalar_alike(inst: &Inst, site: Site, native: &NativeProgram, res: &Resources) -> bool {
    let name = inst.name();
    match inst.fields {
        Fields::Sop { .. } => !name.starts_with("s_getreg"),
        Fields::Smem { .. } if name.starts_with("s_buffer_load_dword") => res
            .descriptors
            .iter()
            .find(|d| d.site == site && d.role == Role::Resource)
            .is_some_and(|d| read_only(&d.source, native, res)),
        Fields::Smem { .. } if name.starts_with("s_load_dword") => {
            let Some(a) = res.scalar.iter().find(|s| s.0 == site).map(|s| s.1) else { return false };
            match a.param {
                CODE => true,
                MEMORY => false,
                k => match native.layout.get(k) {
                    // (descriptors and root constants: what the host wrote before the dispatch)
                    Some(rp) if matches!(rp.root, RootType::Table | RootType::Constants) => true,
                    Some(rp) if rp.root == RootType::Descriptor => native
                        .used
                        .iter()
                        .find(|u| u.root == RootType::Descriptor && u.user_data == rp.user_data)
                        .is_some_and(|u| matches!(u.kind, ResourceType::Cbv | ResourceType::Srv)),
                    _ => false,
                },
            }
        }
        _ => false,
    }
}

/// Whether the buffer `src` describes is a constant or shader-resource view (as `Lower`'s
/// `used_kind` finds its range); one source only.
fn read_only(src: &Source, native: &NativeProgram, res: &Resources) -> bool {
    let used = match src {
        Source::Built { address, .. } => native.layout.get(address.param).and_then(|rp| {
            native.used.iter().find(|u| u.root == RootType::Descriptor && u.user_data == rp.user_data)
        }),
        Source::Loaded { load, first } => {
            let Some(a) = res.scalar.iter().find(|s| s.0 == *load).map(|s| s.1) else { return false };
            // A computed term in the table's offset may pick any later range of the table, a
            // writable one too: unproven (a review's finding).
            // shortcut: no bound on that term is sought.
            if a.dynamic.is_some() {
                return false;
            }
            let Some(rp) = native.layout.get(a.param).filter(|rp| rp.root == RootType::Table) else { return false };
            let dword = a.offset / 4 + u32::from(*first);
            native
                .used
                .iter()
                .filter(|u| u.root == RootType::Table && u.user_data == rp.user_data && u.table_offset_dwords <= dword)
                .max_by_key(|u| u.table_offset_dwords)
        }
        _ => None,
    };
    used.is_some_and(|u| matches!(u.kind, ResourceType::Cbv | ResourceType::Srv))
}
