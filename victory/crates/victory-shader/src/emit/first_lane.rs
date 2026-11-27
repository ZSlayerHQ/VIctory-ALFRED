//! Which values depend on a first-lane read whose lanes may differ.
//!
//! `v_readfirstlane_b32` gives the first lane of EXEC across the console's wave. Where the
//! host's waves are not the console's (a console wave of 64 split over two host waves of 32,
//! or lanes the host orders otherwise) each host wave reads its own first lane, so whatever
//! the program computes from such a read may differ from the console: an address, a permute's
//! selector, the descriptor a twin borrows. This finds every such value once, for the
//! lowering to ask one question wherever it needs the console's first lane (the fourth review
//! of job/land-2c-shader: following only directly tagged reads missed scalar arithmetic and
//! branches that choose a phi's input).

use std::collections::HashSet;

use super::{all_lane_blocks, lane_uniform};
use crate::gfx10::Fields;
use crate::ir::{Exit, Inst, Program, Reg, Value};
use crate::structure::Structured;

/// The values that depend on a first-lane read whose lanes may differ (empty in a group of one
/// thread, where every read is of the one lane):
///
/// - such a read: a `v_readfirstlane_b32` of a VGPR value that is not proved alike in every
///   lane ([`lane_uniform`], defined in a block every lane runs), other than a waterfall's
///   read (`waterfall`: lowered as each lane's own);
/// - what any instruction computes from such a value (each register it writes, whatever it
///   reads, EXEC included);
/// - a phi any of whose inputs depends on one;
/// - a phi whose choice of input a branch makes on a condition that depends on one: every
///   conditional branch on a path from the phi block's immediate dominator to the phi;
/// - what memory may carry (the fifth review: a store and a scalar reload): once a store or
///   atomic reads such a value, every load of that memory is marked. LDS and the rest
///   (buffers, images, flat and scalar memory, which may alias) are told apart; which store
///   reaches which load is not bounded;
/// - likewise once a store or atomic stands where a branch on such a condition decides
///   whether it runs, whatever it stores (the sixth review, 3.1: `s =
///   readfirstlane(thread >> 5)`, a constant stored where `s != 0`, the word read back), in a
///   loop with no way out but the iteration limit too ([`deciders`]).
///
/// Conservative: with an indirect exit (no lane analysis) every first-lane read counts.
pub(super) fn dependents(p: &Program, st: &Structured, waterfall: &[Value], one_thread: bool) -> HashSet<Value> {
    let mut dep = HashSet::new();
    if one_thread {
        return dep;
    }
    let n = p.blocks.len();
    let all = all_lane_blocks(st);
    let same = all.as_ref().map(|a| lane_uniform(p, st, a)).unwrap_or_default();
    let block_of = |v: Value| -> Option<usize> {
        if p.inputs.iter().any(|&(_, i)| i == v) {
            return Some(0);
        }
        p.blocks.iter().position(|b| {
            b.phis.iter().any(|f| f.value == v) || b.insts.iter().any(|i| i.writes.iter().any(|w| w.1 == v))
        })
    };
    // The reads whose lanes may differ.
    for b in &p.blocks {
        for i in &b.insts {
            if i.name() != "v_readfirstlane_b32" {
                continue;
            }
            let Some(&(_, out)) = i.writes.first() else { continue };
            if waterfall.contains(&out) {
                continue;
            }
            let src = i.reads.iter().find(|r| matches!(r.0, Reg::V(_))).map(|r| r.1);
            let alike = match (&all, src) {
                (Some(a), Some(v)) => same.contains(&v) && block_of(v).is_some_and(|b| a[b]),
                (_, None) => true, // a scalar source: one value
                (None, Some(_)) => false,
            };
            if !alike {
                dep.insert(out);
            }
        }
    }
    if dep.is_empty() {
        return dep;
    }
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut index = vec![usize::MAX; n];
    for (k, &b) in order.iter().enumerate() {
        index[b] = k;
    }
    let idom = crate::ir::idoms(&p.blocks, &order, &index);
    // Per block with phis: the blocks whose branch chooses among its inputs (every
    // conditional branch between the block's immediate dominator and its predecessors).
    let choosers: Vec<Vec<usize>> = (0..n)
        .map(|j| {
            let blk = &p.blocks[j];
            if blk.phis.is_empty() || blk.preds.len() < 2 || idom[j] == usize::MAX {
                return Vec::new();
            }
            let d = idom[j];
            let mut seen = vec![false; n];
            let mut stack: Vec<usize> = blk.preds.clone();
            let mut out = Vec::new();
            while let Some(b) = stack.pop() {
                if std::mem::replace(&mut seen[b], true) {
                    continue;
                }
                if matches!(p.blocks[b].exit, Exit::Branch { .. }) {
                    out.push(b);
                }
                if b != d {
                    stack.extend(p.blocks[b].preds.iter().copied());
                }
            }
            out
        })
        .collect();
    // The condition a block's branch reads: what its last instruction reads.
    let condition = |b: usize| p.blocks[b].insts.last().map(|i| i.reads.as_slice()).unwrap_or(&[]);
    // Per block: the blocks whose branch decides whether it runs.
    let deciders = deciders(p, &index);
    // Memory a marked value may have been written to: [LDS, the rest].
    let mut written = [false; 2];
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &order {
            let blk = &p.blocks[b];
            let chosen = choosers[b].iter().any(|&c| condition(c).iter().any(|r| dep.contains(&r.1)));
            // (a store here is one a marked condition chooses)
            let decided = deciders[b].iter().any(|&c| condition(c).iter().any(|r| dep.contains(&r.1)));
            for f in &blk.phis {
                if !dep.contains(&f.value) && (chosen || f.args.iter().any(|a| dep.contains(a))) {
                    dep.insert(f.value);
                    changed = true;
                }
            }
            for i in &blk.insts {
                let memory = memory(i);
                let marked = i.reads.iter().any(|r| dep.contains(&r.1));
                if let Some((m, writes)) = memory
                    && writes
                    && (marked || decided)
                    && !written[m]
                {
                    written[m] = true;
                    changed = true;
                }
                if marked || memory.is_some_and(|(m, _)| written[m]) {
                    for w in &i.writes {
                        changed |= dep.insert(w.1);
                    }
                }
            }
        }
    }
    dep
}

/// Per block: the blocks whose conditional branch decides whether it runs, those it is
/// reachable from without post-dominating them (as [`super::lane_regions`] has a lane
/// branch's region, here for every branch, scalar ones too). An indirect exit ends a path as
/// `s_endpgm` does.
///
/// So does the iteration limit, in a loop with no other way out (the recheck of
/// job/land-2e): from a block no path leads from to the end of the program, a back edge
/// (`index`: each block's place in reverse postorder) may also leave, as the lowering's
/// counted jump does (`Lower::exit`, `abort_block`). Without it nothing in such a loop
/// post-dominates anything (every set keeps its first value, every block) and the walk from
/// a branch stopped at both its successors: no block of the loop was one its branch decides.
/// Blocks a path leads from to the end keep the ends they had.
fn deciders(p: &Program, index: &[usize]) -> Vec<Vec<usize>> {
    let n = p.blocks.len();
    let succ: Vec<Vec<usize>> = p.blocks.iter().map(|b| b.exit.successors()).collect();
    // the blocks a path leads from to a block without a successor
    let mut ends: Vec<bool> = succ.iter().map(Vec::is_empty).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            if !ends[b] && succ[b].iter().any(|&t| ends[t]) {
                ends[b] = true;
                changed = true;
            }
        }
    }
    // the blocks a path ends after: those without a successor, and in a loop with no way out
    // those a back edge leaves
    let last: Vec<bool> = (0..n)
        .map(|b| {
            succ[b].is_empty()
                || !ends[b] && index[b] != usize::MAX && succ[b].iter().any(|&t| index[t] <= index[b])
        })
        .collect();
    // post-dominators (bitsets; a virtual exit after every such block)
    let words = n.div_ceil(64);
    let own = |b: usize| {
        let mut v = vec![0u64; words];
        v[b / 64] |= 1 << (b % 64);
        v
    };
    let mut pdom: Vec<Vec<u64>> = (0..n).map(|b| if last[b] { own(b) } else { vec![u64::MAX; words] }).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            if last[b] {
                continue;
            }
            let mut v = vec![u64::MAX; words];
            for &t in &succ[b] {
                for w in 0..words {
                    v[w] &= pdom[t][w];
                }
            }
            v[b / 64] |= 1 << (b % 64);
            if v != pdom[b] {
                pdom[b] = v;
                changed = true;
            }
        }
    }
    let mut out = vec![Vec::new(); n];
    for (d, blk) in p.blocks.iter().enumerate() {
        if !matches!(blk.exit, Exit::Branch { .. }) {
            continue;
        }
        let mut seen = vec![false; n];
        let mut stack = succ[d].clone();
        while let Some(b) = stack.pop() {
            if seen[b] || pdom[d][b / 64] >> (b % 64) & 1 == 1 {
                continue;
            }
            seen[b] = true;
            out[b].push(d);
            stack.extend(succ[b].iter().copied());
        }
    }
    out
}

/// The memory an instruction reads or writes (0: LDS, 1: the rest), and whether it may write
/// it. Not the lane exchanges that are DS instructions.
fn memory(i: &Inst) -> Option<(usize, bool)> {
    let n = i.name();
    match i.fields {
        Fields::Ds { .. } if n.starts_with("ds_swizzle") || n.contains("permute") => None,
        Fields::Ds { .. } => Some((0, !n.starts_with("ds_read"))),
        Fields::Smem { .. } | Fields::Buffer { .. } | Fields::Image { .. } | Fields::Flat { .. } => {
            Some((1, n.contains("store") || n.contains("atomic") || n.contains("write")))
        }
        _ => None,
    }
}
