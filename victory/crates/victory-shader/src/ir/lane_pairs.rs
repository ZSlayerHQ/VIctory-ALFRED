//! Scalar pairs parked in two VGPR lanes that also hold other scalars elsewhere in the program
//! (the compiler reuses lanes), for [`super::unspill`]: such a lane pair is no pair as a whole,
//! but each save of a pair into it and the restores that read exactly that save are one, and
//! move as one 64-bit register, so a parked lane mask comes back as the mask it is.
//!
//! The rule is web renaming. Sites (the saves and restores of the two lanes) are joined into
//! webs: a save with the save of the next lane that completes it (`partner`), a restore
//! likewise, and a save with every restore it can reach. A web every site of which has its
//! partner in it, whose restores all read saves of the web and never the program's entry, and
//! whose restore pairs each read the partners of what their first halves read, is closed:
//! nothing outside it reads what it saves, and it reads nothing saved outside it. So it can
//! have registers of its own, which is what makes the move of its pair one move.

use crate::gfx10::{self, Flow};

/// `v_writelane_b32` / `v_readlane_b32` of a parked scalar, as `unspill` reads it:
/// (save, VGPR, scalar code, lane).
pub(super) type Park = (bool, u8, u8, u32);

/// A pair of scalars saved into two lanes at some sites and restored from them at others,
/// and nothing else of those lanes' content touches (see [`site_pairs`]). Its sites, as
/// instruction indices: each save or restore of the first lane with the one of the second.
#[derive(Debug)]
pub(super) struct WebPair {
    saves: Vec<(usize, usize)>,
    restores: Vec<(usize, usize)>,
    /// The scalar code the pair is parked in, with the one after it.
    pub(super) code: u8,
}

impl WebPair {
    pub(super) fn sites(&self) -> impl Iterator<Item = usize> + '_ {
        self.saves.iter().chain(&self.restores).flat_map(|&(a, b)| [a, b])
    }

    /// The pair moves that stand for its sites, and the no-ops that take the other halves'
    /// places: a save moves the pair where its second half is saved, a restore where its
    /// first half is restored (as `unspill` moves the pairs it finds over the whole program).
    pub(super) fn edits(&self, parks: &[Option<Park>]) -> Vec<(usize, Option<(bool, u8, u8)>)> {
        let code_of = |i: usize| parks[i].map_or(0, |p| p.2);
        let mut edits = Vec::new();
        for &(first, second) in &self.saves {
            edits.push((second, Some((true, self.code, code_of(first)))));
            edits.push((first, None));
        }
        for &(first, second) in &self.restores {
            edits.push((first, Some((true, code_of(first), self.code))));
            edits.push((second, None));
        }
        edits
    }
}

/// The blocks of the decoded code, as `build` cuts it: each instruction's block (an index into
/// the leaders) and each block's successors.
pub(super) struct BlockGraph {
    block_of: Vec<usize>,
    succ: Vec<Vec<usize>>,
}

impl BlockGraph {
    pub(super) fn new(list: &[gfx10::Instruction], leaders: &[usize], flows: &[Flow]) -> Self {
        let block_of: Vec<usize> = list.iter().map(|i| leaders.partition_point(|&l| l <= i.offset) - 1).collect();
        let mut succ = vec![Vec::new(); leaders.len()];
        let block_at = |offset: usize| leaders.binary_search(&offset).ok();
        for (i, flow) in flows.iter().enumerate() {
            let b = block_of[i];
            let fall = block_of.get(i + 1).copied();
            if fall == Some(b) {
                continue;
            }
            match *flow {
                Flow::Next => succ[b].extend(fall),
                Flow::Jump(t) => succ[b].extend(block_at(t)),
                Flow::Branch(t) => {
                    succ[b].extend(block_at(t));
                    succ[b].extend(fall);
                }
                Flow::Stop => {}
            }
        }
        Self { block_of, succ }
    }
}

/// The saves (bit k for the k-th, `entry` for none) that can reach the start of each block,
/// when the block `b` leaves only `last_save[b]` if it saves at all.
fn reach_blocks(graph: &BlockGraph, last_save: &[Option<u64>], entry: u64) -> Vec<u64> {
    let mut reach = vec![0u64; graph.succ.len()];
    reach[0] = entry;
    let mut changed = true;
    while changed {
        changed = false;
        for (b, succ) in graph.succ.iter().enumerate() {
            let out = last_save[b].unwrap_or(reach[b]);
            for &s in succ {
                let seen = reach[s];
                reach[s] |= out;
                changed |= reach[s] != seen;
            }
        }
    }
    reach
}

/// For each restore of the parked `slot` (VGPR, lane), the saves that can reach it: `None`
/// for the program's entry, where the lane holds nothing. Gives up (`None`) on a lane saved
/// too often for the sets to be bit masks.
fn reaching_saves(parks: &[Option<Park>], graph: &BlockGraph, slot: (u8, u32)) -> Option<Vec<(usize, Vec<Option<usize>>)>> {
    let here = |i: usize| parks[i].filter(|p| (p.1, p.3) == slot);
    let saves: Vec<usize> = (0..parks.len()).filter(|&i| here(i).is_some_and(|p| p.0)).collect();
    if saves.len() >= 63 {
        return None;
    }
    let mut last_save: Vec<Option<u64>> = vec![None; graph.succ.len()];
    for (k, &i) in saves.iter().enumerate() {
        last_save[graph.block_of[i]] = Some(1 << k);
    }
    let reach = reach_blocks(graph, &last_save, 1 << saves.len());
    let restores = (0..parks.len()).filter(|&i| here(i).is_some_and(|p| !p.0));
    Some(
        restores
            .map(|r| {
                let b = graph.block_of[r];
                let mut before = saves.iter().enumerate().filter(|&(_, &i)| graph.block_of[i] == b && i < r);
                let bits = before.next_back().map_or(reach[b], |(k, _)| 1 << k);
                let defs = (0..=saves.len()).filter(|k| bits >> k & 1 != 0);
                (r, defs.map(|k| saves.get(k).copied()).collect())
            })
            .collect(),
    )
}

fn root_of(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// The pairs of every candidate lane pair (VGPR, lower lane), none sharing a site with an
/// earlier one.
pub(super) fn site_webs(
    candidates: &[(u8, u32)],
    parks: &[Option<Park>],
    partner: &dyn Fn(usize) -> Option<usize>,
    graph: &BlockGraph,
) -> Vec<WebPair> {
    let mut webs: Vec<WebPair> = Vec::new();
    for &candidate in candidates {
        let claimed: Vec<usize> = webs.iter().flat_map(WebPair::sites).collect();
        let fresh = site_pairs(parks, partner, graph, candidate)
            .into_iter()
            .filter(|web| !web.sites().any(|i| claimed.contains(&i)));
        webs.extend(fresh);
    }
    webs
}

/// Whether each restore pair (first, second) reads exactly the saves that go with the saves
/// its first restore reads: `reaching` gives the saves each restore can see.
fn restores_read_their_pair(
    restores: &[(usize, usize)],
    reaching: &[(usize, Vec<Option<usize>>)],
    partner: &dyn Fn(usize) -> Option<usize>,
) -> bool {
    let reads = |i: usize| -> Vec<Option<usize>> {
        let mut saves = reaching.iter().find(|r| r.0 == i).map_or_else(Vec::new, |r| r.1.clone());
        saves.sort_unstable();
        saves
    };
    restores.iter().all(|&(first, second)| {
        let mut want: Vec<Option<usize>> = reads(first).iter().map(|d| d.and_then(partner)).collect();
        want.sort_unstable();
        want == reads(second)
    })
}

/// The pairs (see [`WebPair`]) parked in the lanes `lane` and `lane + 1` of a VGPR that
/// `unspill` can move as one 64-bit register though the lanes hold other scalars at other
/// places: the closed webs of the module's rule.
fn site_pairs(
    parks: &[Option<Park>],
    partner: &dyn Fn(usize) -> Option<usize>,
    graph: &BlockGraph,
    (reg, lane): (u8, u32),
) -> Vec<WebPair> {
    let in_lane = |i: usize, l: u32| parks[i].is_some_and(|p| p.1 == reg && p.3 == l);
    let sites: Vec<usize> = (0..parks.len()).filter(|&i| in_lane(i, lane) || in_lane(i, lane + 1)).collect();
    let at = |i: usize| sites.binary_search(&i).expect("a site of the lanes");
    let is_save = |i: usize| parks[i].is_some_and(|p| p.0);
    let mut parent: Vec<usize> = (0..sites.len()).collect();
    let mut broken = vec![false; sites.len()];
    let join = |parent: &mut [usize], a: usize, b: usize| {
        let (a, b) = (root_of(parent, at(a)), root_of(parent, at(b)));
        parent[a] = b;
    };
    let mut partners = Vec::new();
    for &i in sites.iter().filter(|&&i| in_lane(i, lane)) {
        match partner(i) {
            Some(j) => {
                join(&mut parent, i, j);
                partners.push(j);
            }
            None => broken[at(i)] = true,
        }
    }
    for &j in sites.iter().filter(|&&j| in_lane(j, lane + 1) && !partners.contains(&j)) {
        broken[at(j)] = true;
    }
    let mut reaching: Vec<(usize, Vec<Option<usize>>)> = Vec::new();
    for l in [lane, lane + 1] {
        let Some(found) = reaching_saves(parks, graph, (reg, l)) else { return Vec::new() };
        reaching.extend(found);
    }
    for (r, defs) in &reaching {
        for def in defs {
            match def {
                Some(d) => join(&mut parent, *r, *d),
                None => broken[at(*r)] = true,
            }
        }
    }
    let mut components: Vec<usize> = (0..sites.len()).map(|k| root_of(&mut parent, k)).collect();
    components.sort_unstable();
    components.dedup();
    let mut webs = Vec::new();
    for component in components {
        let members: Vec<usize> = sites.iter().copied().filter(|&i| root_of(&mut parent, at(i)) == component).collect();
        if members.iter().any(|&i| broken[at(i)]) {
            continue;
        }
        let (saves, restores): (Vec<usize>, Vec<usize>) =
            members.iter().copied().filter(|&i| in_lane(i, lane)).partition(|&i| is_save(i));
        let (upper_saves, upper_restores): (Vec<usize>, Vec<usize>) =
            members.iter().copied().filter(|&i| in_lane(i, lane + 1)).partition(|&i| is_save(i));
        if saves.is_empty()
            || restores.is_empty()
            || saves.len() != upper_saves.len()
            || restores.len() != upper_restores.len()
        {
            continue;
        }
        let pairs = |sites: &[usize]| -> Vec<(usize, usize)> { sites.iter().filter_map(|&i| Some((i, partner(i)?))).collect() };
        let restores = pairs(&restores);
        if restores_read_their_pair(&restores, &reaching, partner) {
            webs.push(WebPair { saves: pairs(&saves), restores, code: 0 });
        }
    }
    webs
}
