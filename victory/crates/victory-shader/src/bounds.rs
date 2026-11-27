//! Loop bounds the console's own code proves (the plan's loop-bounds pass).
//!
//! The translation counts every loop's turns and leaves a loop at a cap per
//! stage (`emit::lower`, `count_iteration`): a count read from data the host
//! has not mirrored yet can make a loop run four billion times. Where the
//! loop's own counter bounds it, the console stops there whatever its data
//! say, and so may the host: such a loop's cap is its proved turn count where
//! that is above the stage's.
//!
//! The proof ([`loops`]): a block every turn passes (one that dominates each
//! of the loop's back edges) ends in a branch with one arm in the loop and
//! one out of it; a lane takes the arm in only where a compare holds (or
//! fails) whose one side is the counter and whose other is a constant. The
//! counter is a phi of the loop's header that starts at constants on every
//! edge into the loop and moves by the same constant step on every back edge
//! (adds and moves of constants, nothing else). The compare is then a
//! function of the turn alone, evaluated turn by turn in the registers'
//! 32-bit wrapping arithmetic until it sends the lane out. A counter the
//! body also changes reaches the back edge through another phi and is not
//! proved; nor is a compare with a value the program reads or computes, nor
//! a branch the whole wave takes together (`Structured::wide`: there a lane
//! goes by the other lanes' tests too), nor a branch whose way out is another
//! loop's back edge the lowering would leave by the other arm at that loop's
//! cap (on into this loop, [`cut_at_limit`]). The proof is of the lane form the
//! lowering emits (`Structured::exits`): a host lane takes a back edge only
//! where its own condition holds, so with the proved cap the cap never acts.

use std::collections::{HashMap, HashSet};

use crate::gfx10::{Fields, NULL, Operand, Valu};
use crate::ir::{self, Inst, Program, Reg, Value};
use crate::structure::{Atom, Bool, Bools, LaneExit, Structured};

/// Turns a proof looks at: a counter that has not left the loop by then is
/// not proved ([`Bound::Beyond`]). The translation's old cap for every stage
/// (`emit::DEFAULT_LOOP_LIMIT`).
pub const MAX_TURNS: u32 = 1 << 20;

/// What bounds one loop's turns per entry into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    /// Proved: at most this many turns (entries into the header, one more
    /// than the back edges a lane takes), whatever the data. What the
    /// lowering's cap counts: with this cap no lane reaches it.
    Turns(u32),
    /// A counter from constants by a constant step, tested against a
    /// constant, still in the loop after [`MAX_TURNS`] turns.
    Beyond,
    /// A counter tested against a value read from memory or the program's
    /// inputs (a constant buffer, user data).
    Data(Source),
    /// A compare of the header's counter where something else is computed:
    /// its start, its step, or the other side.
    Computed,
    /// A float counter (a header phi from constants, one constant `v_add_f32`
    /// each turn) tested against a constant: the turns an f32 run of the
    /// counter takes. A count for the census only, never a cap: the host's
    /// float arithmetic is not proved to be the console's.
    Float(u32),
    /// A float counter tested against a value read from memory or inputs.
    FloatData(Source),
    /// No counter test the translator can see on every turn (or none whose
    /// way out the host keeps: see [`loops`]).
    Unseen,
}

/// Where the other side of a counter's test comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// A scalar or vector load, an LDS read.
    Memory,
    /// A register the program reads on entry (user data, system values).
    Input,
}

impl Bound {
    /// Proofs first, then what is more visible.
    fn rank(self) -> u64 {
        match self {
            Bound::Turns(n) => u64::from(n),
            Bound::Float(n) => (1 << 32) + u64::from(n),
            Bound::Beyond => 1 << 33,
            Bound::Data(_) => 1 << 34,
            Bound::FloatData(_) => (1 << 34) + 1,
            Bound::Computed => 1 << 35,
            Bound::Unseen => 1 << 36,
        }
    }
}

/// One loop: its header block, the header's byte offset, its bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Loop {
    pub header: usize,
    pub offset: usize,
    pub bound: Bound,
}

/// Whether the translation uses the proofs: `VICTORY_SHADER_LOOP_BOUNDS=0`
/// switches it off (every loop then has its stage's cap, as before the pass).
/// Part of the translation: the shader cache keys on it (`victory-gpu`'s
/// does, as on every `VICTORY_*` switch named in this crate).
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("VICTORY_SHADER_LOOP_BOUNDS").map_or(true, |v| v.trim() != "0")
    })
}

/// The proved turn counts by loop header, for the lowering's caps; empty
/// with the pass switched off ([`enabled`]). `strict`: the lowering's rule at
/// a lane's limit ([`cuts`]).
pub(crate) fn caps(p: &Program, st: &Structured, strict: bool) -> HashMap<usize, u32> {
    if !enabled() {
        return HashMap::new();
    }
    loops_of(p, st, false, strict)
        .into_iter()
        .filter_map(|l| match l.bound {
            Bound::Turns(n) => Some((l.header, n)),
            _ => None,
        })
        .collect()
}

/// Whether block `x` is in the loop of header `h` (`place`: each block's
/// place in reverse postorder, `usize::MAX` where unreachable): `h`, or a
/// block a way leads from to a back edge into `h` without passing `h`.
pub(crate) fn in_loop(p: &Program, place: &[usize], h: usize, x: usize) -> bool {
    let at = |b: usize| place.get(b).copied().unwrap_or(usize::MAX);
    let mut seen = vec![false; p.blocks.len()];
    seen[h] = true;
    let mut work: Vec<usize> = p.blocks[h]
        .preds
        .iter()
        .copied()
        .filter(|&q| at(q) != usize::MAX && at(h) <= at(q))
        .collect();
    while let Some(b) = work.pop() {
        if !std::mem::replace(&mut seen[b], true) {
            work.extend(p.blocks[b].preds.iter().copied());
        }
    }
    seen[x]
}

/// Whether a lane at the iteration limit of header `to`, on the arm of block
/// `b`'s branch back to it, leaves the program (else it goes down the other
/// arm, to `other`): where `other` stays in the loop of `to`, and where both
/// arms go back and the other's limit would send the lane down this arm
/// again (then each cap leaves the program; never so in reducible flow, one
/// loop holding the other).
///
/// `strict` false, for a program with a group barrier outside the meeting
/// form (an `s_barrier`, the exchanges of a split wave): a lane that
/// left the program there would be waited for at the barrier (NVIDIA's wait
/// is for every thread, and the card is reset), so it leaves only where there
/// is no other arm to go down, both arms going back; else it goes down the
/// other arm, as before the loop-bounds pass, still in the loop.
pub(crate) fn cuts(p: &Program, place: &[usize], b: usize, to: usize, other: usize, strict: bool) -> bool {
    let at = |x: usize| place.get(x).copied().unwrap_or(usize::MAX);
    let back = |t: usize| at(b) != usize::MAX && at(t) <= at(b);
    back(to)
        && (strict || back(other))
        && (in_loop(p, place, to, other) || back(other) && !in_loop(p, place, other, to))
}

/// The header whose iteration limit makes a lane leave the program at the
/// end of block `b` (`emit::lower`'s `abort_block`), if any. Every back edge
/// is counted; a lane at the limit goes down the branch's other arm where
/// that arm leaves the loop, and leaves the program where there is no other
/// arm (a plain jump back) or the other arm stays in the loop (both arms
/// `continue`; with `strict` false, only both arms back, [`cuts`]). Of two
/// such headers, the outer one.
pub(crate) fn cut_at_limit(
    p: &Program,
    st: &Structured,
    place: &[usize],
    b: usize,
    strict: bool,
) -> Option<usize> {
    let at = |x: usize| place.get(x).copied().unwrap_or(usize::MAX);
    if at(b) == usize::MAX {
        return None;
    }
    let back = |to: usize| at(to) <= at(b);
    match st.exits[b] {
        LaneExit::Jump(to) if back(to) => Some(to),
        LaneExit::Branch {
            taken, not_taken, ..
        } => [(taken, not_taken), (not_taken, taken)]
            .into_iter()
            .filter(|&(to, other)| cuts(p, place, b, to, other, strict))
            .map(|(to, _)| to)
            .min_by_key(|&to| at(to)),
        _ => None,
    }
}

/// Every loop of `p` (a header: a block with a back edge into it, as the
/// lowering counts them) with what bounds it, in reverse postorder.
/// The float counters' census ([`Bound::Float`]) included.
/// (Of the lowering where a lane at its limit may leave the program: no
/// group barrier, [`cuts`].)
pub fn loops(p: &Program, st: &Structured) -> Vec<Loop> {
    loops_of(p, st, true, true)
}

/// [`loops`], with `floats` whether float counters are run (a census's
/// cost: the translation never caps by them).
fn loops_of(p: &Program, st: &Structured, floats: bool, strict: bool) -> Vec<Loop> {
    let order = ir::reverse_postorder(&p.blocks);
    let mut index = vec![usize::MAX; p.blocks.len()];
    for (n, &b) in order.iter().enumerate() {
        index[b] = n;
    }
    let idom = ir::idoms(&p.blocks, &order, &index);
    let defs = Defs::new(p, st, floats);
    let mut out = Vec::new();
    for &h in &order {
        let latches: Vec<usize> = p.blocks[h]
            .preds
            .iter()
            .copied()
            .filter(|&q| index[q] != usize::MAX && index[h] <= index[q])
            .collect();
        if latches.is_empty() {
            continue;
        }
        // The blocks every turn passes: on each back edge's dominator chain.
        let mut every: Option<Vec<usize>> = None;
        for &l in &latches {
            let mut chain = vec![l];
            let mut d = l;
            while d != h && idom[d] != d && idom[d] != usize::MAX {
                d = idom[d];
                chain.push(d);
            }
            if d != h {
                chain.clear();
            }
            every = Some(match every {
                None => chain,
                Some(e) => e.into_iter().filter(|b| chain.contains(b)).collect(),
            });
        }
        let body = body(p, h, &latches);
        let mut bound = Bound::Unseen;
        for x in every.unwrap_or_default() {
            let b = defs.exit_bound(h, x, &body, &latches, (&index, strict));
            if b.rank() < bound.rank() {
                bound = b;
            }
        }
        out.push(Loop {
            header: h,
            offset: p.blocks[h].offset,
            bound,
        });
    }
    out
}

/// The loop of header `h`: the blocks that reach one of its back edges
/// without passing `h`, and `h`.
fn body(p: &Program, h: usize, latches: &[usize]) -> HashSet<usize> {
    let mut body = HashSet::from([h]);
    let mut work: Vec<usize> = latches.to_vec();
    while let Some(b) = work.pop() {
        if body.insert(b) {
            work.extend(p.blocks[b].preds.iter().copied());
        }
    }
    body
}

/// An operand: a constant, or the value of a register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Val {
    K(u32),
    V(Value),
}

/// One side of a compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    /// The header's phi (by value) plus a constant.
    Counter(Value, u32),
    K(u32),
    Data(Source),
    Computed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pred {
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
}

/// A compare of two 32-bit operands.
#[derive(Debug, Clone, Copy)]
struct Test {
    pred: Pred,
    signed: bool,
    a: Val,
    b: Val,
}

impl Test {
    fn eval(&self, a: u32, b: u32) -> bool {
        let ord = if self.signed {
            (a as i32).cmp(&(b as i32))
        } else {
            a.cmp(&b)
        };
        match self.pred {
            Pred::Eq => ord.is_eq(),
            Pred::Ne => ord.is_ne(),
            Pred::Gt => ord.is_gt(),
            Pred::Ge => ord.is_ge(),
            Pred::Lt => ord.is_lt(),
            Pred::Le => ord.is_le(),
        }
    }
}

/// `s_cmp_gt_i32`, `s_cmpk_lg_u32`, `v_cmp_lt_u32`, `v_cmpx_ne_i32`...: the
/// predicate and signedness.
fn compare(name: &str) -> Option<(Pred, bool)> {
    let rest = ["s_cmpk_", "s_cmp_", "v_cmpx_", "v_cmp_"]
        .iter()
        .find_map(|p| name.strip_prefix(p))?;
    let (op, ty) = rest.split_once('_')?;
    let signed = match ty {
        "i32" => true,
        "u32" => false,
        _ => return None,
    };
    let pred = match op {
        "eq" => Pred::Eq,
        "lg" | "ne" => Pred::Ne,
        "gt" => Pred::Gt,
        "ge" => Pred::Ge,
        "lt" => Pred::Lt,
        "le" => Pred::Le,
        _ => return None,
    };
    Some((pred, signed))
}

/// A vector instruction without modifiers: its operands are its values.
fn plain(f: &Valu) -> bool {
    f.sdwa.is_none() && f.dpp.is_none() && !f.clamp && f.omod == 0 && f.neg == 0 && f.abs == 0
}

/// How deep [`Defs`] follows moves and adds.
const DEPTH: u32 = 16;
/// How many phis [`Defs::constant`] follows within each other.
const PHI_DEPTH: u32 = 256;

/// Where each value is written.
struct Defs<'a> {
    p: &'a Program,
    st: &'a Structured,
    insts: HashMap<Value, (usize, usize)>,
    phis: HashMap<Value, (usize, usize)>,
    inputs: HashSet<Value>,
    /// Whether float counters are looked at ([`Bound::Float`]).
    floats: bool,
    /// [`Defs::constant`]'s answers for phis (a phi on the way: `None`), so
    /// that webs of phis are walked once.
    phi_constants: std::cell::RefCell<HashMap<Value, Option<u32>>>,
    /// Phis [`Defs::constant`] is inside of (a bound on its stack).
    phi_depth: std::cell::Cell<u32>,
}

impl<'a> Defs<'a> {
    fn new(p: &'a Program, st: &'a Structured, floats: bool) -> Self {
        let mut insts = HashMap::new();
        let mut phis = HashMap::new();
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, v) in &inst.writes {
                    insts.insert(v, (b, i));
                }
            }
            for (i, phi) in block.phis.iter().enumerate() {
                phis.insert(phi.value, (b, i));
            }
        }
        let inputs = p.inputs.iter().map(|&(_, v)| v).collect();
        Defs {
            p,
            st,
            insts,
            phis,
            inputs,
            floats,
            phi_constants: std::cell::RefCell::default(),
            phi_depth: std::cell::Cell::new(0),
        }
    }

    /// The instruction writing `v`, where every lane that reaches it runs it
    /// (no guard: a guarded vector write keeps some lanes' old value).
    fn def(&self, v: Value) -> Option<&'a Inst> {
        let &(b, i) = self.insts.get(&v)?;
        let unguarded = self
            .st
            .guards
            .get(b)
            .and_then(|g| g.get(i))
            .is_none_or(|g| g.is_none());
        unguarded.then(|| &self.p.blocks[b].insts[i])
    }

    fn operand(inst: &Inst, op: &Operand) -> Option<Val> {
        match *op {
            Operand::Const(c @ 128..=192) => Some(Val::K(u32::from(c - 128))),
            Operand::Const(c @ 193..=208) => Some(Val::K((-(i32::from(c) - 192)) as u32)),
            // Float inline constants: their f32 bits (32-bit operations).
            Operand::Const(c @ 240..=248) => Some(Val::K(
                [
                    0.5f32,
                    -0.5,
                    1.0,
                    -1.0,
                    2.0,
                    -2.0,
                    4.0,
                    -4.0,
                ]
                .get(usize::from(c - 240))
                .map_or(0x3E22_F983, |f| f.to_bits()),
            )),
            Operand::Literal(k) => Some(Val::K(k)),
            Operand::Scalar { code: NULL, .. } => Some(Val::K(0)),
            Operand::Scalar { code, dwords: 1 } => inst.read(Reg::S(code)).map(Val::V),
            Operand::Vector { reg, dwords: 1 } => inst.read(Reg::V(reg)).map(Val::V),
            _ => None,
        }
    }

    /// A move's or an add's operands (`None`: neither).
    fn sources(inst: &Inst) -> Option<Vec<Val>> {
        let ops = match &inst.fields {
            Fields::Sop { src, .. } => src.clone(),
            Fields::Valu(f) if plain(f) => f.src.clone(),
            _ => return None,
        };
        ops.iter().map(|o| Self::operand(inst, o)).collect()
    }

    /// `v` as a constant, through moves and phis whose arguments agree.
    fn constant(&self, v: Val, depth: u32) -> Option<u32> {
        let v = match v {
            Val::K(k) => return Some(k),
            Val::V(v) => v,
        };
        if depth == 0 {
            return None;
        }
        if let Some(&(b, i)) = self.phis.get(&v) {
            if let Some(&known) = self.phi_constants.borrow().get(&v) {
                return known;
            }
            if self.phi_depth.get() >= PHI_DEPTH {
                return None;
            }
            self.phi_depth.set(self.phi_depth.get() + 1);
            self.phi_constants.borrow_mut().insert(v, None);
            let args = &self.p.blocks[b].phis[i].args;
            // Each phi is walked once, whatever the depth it is met at: the
            // answer does not depend on the order of the questions (a phi on
            // a cycle meets itself on the way and is no constant).
            let first = args.first().and_then(|&a| self.constant(Val::V(a), DEPTH));
            let all = first.filter(|&f| {
                args[1..].iter().all(|&a| self.constant(Val::V(a), DEPTH) == Some(f))
            });
            self.phi_constants.borrow_mut().insert(v, all);
            self.phi_depth.set(self.phi_depth.get() - 1);
            return all;
        }
        let inst = self.def(v)?;
        match (inst.name(), &inst.fields) {
            ("s_movk_i32", Fields::Sop { imm: Some(k), .. }) => Some(*k as i16 as i32 as u32),
            ("s_mov_b32" | "v_mov_b32", _) => match *Self::sources(inst)?.as_slice() {
                [s] => self.constant(s, depth - 1),
                _ => None,
            },
            _ => None,
        }
    }

    /// `v` as a root value plus a constant, through moves and adds or
    /// subtractions of constants (in 32-bit wrapping arithmetic, as the
    /// registers have it). The root is a phi or what no such step wrote.
    fn linear(&self, v: Value, depth: u32) -> (Value, u32) {
        if depth == 0 || self.phis.contains_key(&v) {
            return (v, 0);
        }
        let Some(inst) = self.def(v) else {
            return (v, 0);
        };
        let name = inst.name();
        let step = |x: Val, k: u32, neg: bool| match x {
            Val::V(x) => {
                let (root, c) = self.linear(x, depth - 1);
                Some((
                    root,
                    if neg {
                        c.wrapping_sub(k)
                    } else {
                        c.wrapping_add(k)
                    },
                ))
            }
            Val::K(_) => None,
        };
        let found = match (name, &inst.fields) {
            (
                "s_addk_i32",
                Fields::Sop {
                    dst: Some(Operand::Scalar { code, .. }),
                    imm: Some(k),
                    ..
                },
            ) => inst
                .read(Reg::S(*code))
                .and_then(|x| step(Val::V(x), *k as i16 as i32 as u32, false)),
            _ => match (name, Self::sources(inst).as_deref()) {
                ("s_mov_b32" | "v_mov_b32", Some(&[x])) => step(x, 0, false),
                ("s_add_u32" | "s_add_i32" | "v_add_nc_u32" | "v_add_co_u32", Some(&[x, y])) => {
                    match (x, y) {
                        (x, Val::K(k)) | (Val::K(k), x) => step(x, k, false),
                        _ => None,
                    }
                }
                (
                    "s_sub_u32" | "s_sub_i32" | "v_sub_nc_u32" | "v_sub_co_u32",
                    Some(&[x, Val::K(k)]),
                ) => step(x, k, true),
                ("v_subrev_nc_u32" | "v_subrev_co_u32", Some(&[Val::K(k), x])) => step(x, k, true),
                _ => None,
            },
        };
        found.unwrap_or((v, 0))
    }

    /// Where a value no step leads back from comes from.
    fn source(&self, v: Value) -> Option<Source> {
        if self.inputs.contains(&v) {
            return Some(Source::Input);
        }
        let &(b, i) = self.insts.get(&v)?;
        let name = self.p.blocks[b].insts[i].name();
        let memory = [
            "s_load_",
            "s_buffer_load_",
            "buffer_load_",
            "global_load_",
            "flat_load_",
            "ds_read",
            "tbuffer_load_",
        ];
        memory
            .iter()
            .any(|m| name.starts_with(m))
            .then_some(Source::Memory)
    }

    /// One side of a compare in a loop of header `h`.
    fn side(&self, v: Val, h: usize) -> Side {
        if let Some(k) = self.constant(v, DEPTH) {
            return Side::K(k);
        }
        let Val::V(v) = v else {
            unreachable!("constants are constant")
        };
        let (root, c) = self.linear(v, DEPTH);
        match self.phis.get(&root) {
            Some(&(b, _)) if b == h => Side::Counter(root, c),
            _ => self.source(root).map_or(Side::Computed, Side::Data),
        }
    }

    /// The compare that sets `atom`: a scalar compare's SCC (or an
    /// unsigned add's or subtraction's carry, as a compare), or a vector
    /// compare's lane bit.
    fn test(&self, atom: Atom) -> Option<Test> {
        let (v, uniform) = match atom {
            Atom::Uniform(v) => (v, true),
            Atom::Lane(v) => (v, false),
            _ => return None,
        };
        let inst = self.def(v)?;
        if uniform && inst.write(Reg::Scc) != Some(v) {
            return None;
        }
        let name = inst.name();
        if let Some((pred, signed)) = compare(name) {
            if name.starts_with("s_cmpk_") {
                let Fields::Sop {
                    src, imm: Some(k), ..
                } = &inst.fields
                else {
                    return None;
                };
                let k = if signed {
                    *k as i16 as i32 as u32
                } else {
                    u32::from(*k)
                };
                return Some(Test {
                    pred,
                    signed,
                    a: Self::operand(inst, src.first()?)?,
                    b: Val::K(k),
                });
            }
            if !uniform && !matches!(&inst.fields, Fields::Valu(f) if plain(f)) {
                return None;
            }
            let [a, b] = *Self::sources(inst)?.as_slice() else {
                return None;
            };
            return Some(Test { pred, signed, a, b });
        }
        let [a, b] = *Self::sources(inst)?.as_slice() else {
            return None;
        };
        match name {
            // Borrow: a < b.
            "s_sub_u32" if uniform => Some(Test {
                pred: Pred::Lt,
                signed: false,
                a,
                b,
            }),
            // Carry: a + k past 2^32 - 1, that is a > !k.
            "s_add_u32" if uniform => match (a, b) {
                (x, Val::K(k)) | (Val::K(k), x) => Some(Test {
                    pred: Pred::Gt,
                    signed: false,
                    a: x,
                    b: Val::K(!k),
                }),
                _ => None,
            },
            _ => None,
        }
    }

    /// What block `x` of the loop of `h` proves at its closing branch
    /// (`index`: each block's place in reverse postorder; `strict`, [`cuts`]).
    fn exit_bound(
        &self,
        h: usize,
        x: usize,
        body: &HashSet<usize>,
        latches: &[usize],
        (index, strict): (&[usize], bool),
    ) -> Bound {
        let LaneExit::Branch {
            when,
            taken,
            not_taken,
        } = self.st.exits[x]
        else {
            return Bound::Unseen;
        };
        // A branch the whole wave takes together goes by other lanes too.
        if self.st.wide.contains(&x) {
            return Bound::Unseen;
        }
        let inside = |a: usize| a == h || body.contains(&a);
        // Staying is `when` (taken arm inside) or its negation.
        let negate = match (inside(taken), inside(not_taken)) {
            (true, false) => false,
            (false, true) => true,
            _ => return Bound::Unseen,
        };
        // A way out that is another loop's back edge is one the host keeps
        // only where that loop's cap makes the lane leave the program: where
        // it sends the lane down the other arm instead, that arm is a way on
        // into this loop ([`cuts`]).
        let (out, stay) = if negate { (taken, not_taken) } else { (not_taken, taken) };
        if index[out] <= index[x] && !cuts(self.p, index, x, out, stay, strict) {
            return Bound::Unseen;
        }
        let mut best = Bound::Unseen;
        for atom in self.st.bools.support(when) {
            let Some(stay) = stays_only_if(&self.st.bools, when, negate, atom) else {
                continue;
            };
            let b = match self.test(atom) {
                Some(test) => self.bound(h, latches, test, stay),
                None if self.floats => self.float_bound(h, latches, atom, stay),
                None => continue,
            };
            if b.rank() < best.rank() {
                best = b;
            }
        }
        best
    }

    /// The bound a lane stays in the loop of `h` under: only while `test`
    /// gives `stay`.
    fn bound(&self, h: usize, latches: &[usize], test: Test, stay: bool) -> Bound {
        let (a, b) = (self.side(test.a, h), self.side(test.b, h));
        let (phi, c, k, counter_first) = match (a, b) {
            (Side::Counter(phi, c), Side::K(k)) => (phi, c, k, true),
            (Side::K(k), Side::Counter(phi, c)) => (phi, c, k, false),
            (Side::Counter(..), Side::Data(s)) | (Side::Data(s), Side::Counter(..)) => {
                return Bound::Data(s);
            }
            (Side::Counter(..), _) | (_, Side::Counter(..)) => return Bound::Computed,
            _ => return Bound::Unseen,
        };
        let block = &self.p.blocks[h];
        let Some(phi_at) = block.phis.iter().position(|p| p.value == phi) else {
            return Bound::Computed;
        };
        let args = &block.phis[phi_at].args;
        let (mut starts, mut step) = (Vec::new(), None);
        for (&q, &arg) in block.preds.iter().zip(args) {
            if latches.contains(&q) {
                match self.linear(arg, DEPTH) {
                    (root, d) if root == phi && d != 0 && step.is_none_or(|s| s == d) => {
                        step = Some(d)
                    }
                    _ => return Bound::Computed,
                }
            } else {
                match self.constant(Val::V(arg), DEPTH) {
                    Some(s) => starts.push(s),
                    None => return Bound::Computed,
                }
            }
        }
        let Some(d) = step else {
            return Bound::Computed;
        };
        let mut most = 0;
        for s in starts {
            // Turn t (from 0) tests s + t·d + c.
            let mut value = s.wrapping_add(c);
            let mut left = None;
            for t in 0..MAX_TURNS {
                let r = if counter_first {
                    test.eval(value, k)
                } else {
                    test.eval(k, value)
                };
                if r != stay {
                    left = Some(t + 1);
                    break;
                }
                value = value.wrapping_add(d);
            }
            match left {
                Some(turns) => most = most.max(turns),
                None => return Bound::Beyond,
            }
        }
        if most == 0 {
            Bound::Computed
        } else {
            Bound::Turns(most)
        }
    }
}

impl Defs<'_> {
    /// A float side: the header's phi, or one constant add to it (`Some`:
    /// the constant, as bits); a constant; or where it comes from.
    fn float_side(&self, v: Val, h: usize) -> Result<(Value, Option<u32>), Side> {
        if let Some(k) = self.constant(v, DEPTH) {
            return Err(Side::K(k));
        }
        let Val::V(v) = v else {
            unreachable!("constants are constant")
        };
        let header_phi = |x: Value| self.phis.get(&x).is_some_and(|&(b, _)| b == h);
        if header_phi(v) {
            return Ok((v, None));
        }
        if let Some((x, k)) = self.float_step(v)
            && header_phi(x)
        {
            return Ok((x, Some(k)));
        }
        let (root, _) = self.linear(v, DEPTH);
        Err(self.source(root).map_or(Side::Computed, Side::Data))
    }

    /// `v` as `x + k` by one `v_add_f32` (or subtraction) of a constant.
    fn float_step(&self, v: Value) -> Option<(Value, u32)> {
        let inst = self.def(v)?;
        let neg = |k: u32| k ^ 0x8000_0000;
        match (inst.name(), Self::sources(inst)?.as_slice()) {
            ("v_add_f32", &[Val::V(x), Val::K(k)] | &[Val::K(k), Val::V(x)]) => Some((x, k)),
            ("v_sub_f32", &[Val::V(x), Val::K(k)]) | ("v_subrev_f32", &[Val::K(k), Val::V(x)]) => {
                Some((x, neg(k)))
            }
            _ => None,
        }
    }

    /// [`Defs::bound`] for a float compare's lane bit (census only).
    fn float_bound(&self, h: usize, latches: &[usize], atom: Atom, stay: bool) -> Bound {
        let Atom::Lane(v) = atom else {
            return Bound::Unseen;
        };
        let Some(inst) = self.def(v) else {
            return Bound::Unseen;
        };
        let Some(op) = ["v_cmp_", "v_cmpx_"]
            .iter()
            .find_map(|p| inst.name().strip_prefix(p))
            .and_then(|r| r.strip_suffix("_f32"))
        else {
            return Bound::Unseen;
        };
        let Some(&[a, b]) = Self::sources(inst).as_deref() else {
            return Bound::Unseen;
        };
        let (side, k, counter_first) = match (self.float_side(a, h), self.float_side(b, h)) {
            (Ok(c), Err(Side::K(k))) => (c, k, true),
            (Err(Side::K(k)), Ok(c)) => (c, k, false),
            (Ok(_), Err(Side::Data(s))) | (Err(Side::Data(s)), Ok(_)) => {
                return Bound::FloatData(s);
            }
            (Ok(_), _) | (_, Ok(_)) => return Bound::Computed,
            _ => return Bound::Unseen,
        };
        let cmp = |x: f32, y: f32| -> Option<bool> {
            let o = x.partial_cmp(&y);
            let ordered = |f: fn(std::cmp::Ordering) -> bool| o.is_some_and(f);
            Some(match op {
                "lt" => ordered(|o| o.is_lt()),
                "le" => ordered(|o| o.is_le()),
                "gt" => ordered(|o| o.is_gt()),
                "ge" => ordered(|o| o.is_ge()),
                "eq" => ordered(|o| o.is_eq()),
                "lg" => ordered(|o| o.is_ne()),
                "nlt" => !ordered(|o| o.is_lt()),
                "nle" => !ordered(|o| o.is_le()),
                "ngt" => !ordered(|o| o.is_gt()),
                "nge" => !ordered(|o| o.is_ge()),
                "neq" => !ordered(|o| o.is_eq()),
                "nlg" => !ordered(|o| o.is_ne()),
                _ => return None,
            })
        };
        let ((phi, offset), k) = (side, f32::from_bits(k));
        let block = &self.p.blocks[h];
        let Some(at) = block.phis.iter().position(|p| p.value == phi) else {
            return Bound::Computed;
        };
        let (mut starts, mut step) = (Vec::new(), None);
        for (&q, &arg) in block.preds.iter().zip(&block.phis[at].args) {
            if latches.contains(&q) {
                match self.float_step(arg) {
                    Some((x, d)) if x == phi && step.is_none_or(|s| s == d) => step = Some(d),
                    _ => return Bound::Computed,
                }
            } else {
                match self.constant(Val::V(arg), DEPTH) {
                    Some(s) => starts.push(f32::from_bits(s)),
                    None => return Bound::Computed,
                }
            }
        }
        let Some(d) = step.map(f32::from_bits) else {
            return Bound::Computed;
        };
        let mut most = 0;
        for mut x in starts {
            let mut left = None;
            for t in 0..MAX_TURNS {
                let value = offset.map_or(x, |o| x + f32::from_bits(o));
                let Some(r) = (if counter_first {
                    cmp(value, k)
                } else {
                    cmp(k, value)
                }) else {
                    return Bound::Unseen;
                };
                if r != stay {
                    left = Some(t + 1);
                    break;
                }
                // A step too small to move the counter: it stays for good.
                if x + d == x || !x.is_finite() {
                    break;
                }
                x += d;
            }
            match left {
                Some(turns) => most = most.max(turns),
                None => return Bound::Beyond,
            }
        }
        if most == 0 {
            Bound::Computed
        } else {
            Bound::Float(most)
        }
    }
}

/// Whether a lane can stay (`when`, or its negation with `negate`) only
/// where `atom` has one value: that value.
fn stays_only_if(b: &Bools, when: Bool, negate: bool, atom: Atom) -> Option<bool> {
    match (
        reaches(b, when, negate, atom, false),
        reaches(b, when, negate, atom, true),
    ) {
        (true, false) => Some(false),
        (false, true) => Some(true),
        _ => None,
    }
}

/// Whether `when` (negated with `negate`) holds for some lane where `atom`
/// is `value`: a path to the true leaf with that atom fixed (every path of a
/// reduced ordered diagram tests an atom once, so a path is an assignment).
fn reaches(b: &Bools, when: Bool, negate: bool, atom: Atom, value: bool) -> bool {
    let leaf = if negate { Bool::FALSE } else { Bool::TRUE };
    let mut seen = HashSet::new();
    let mut work = vec![when];
    while let Some(n) = work.pop() {
        if !seen.insert(n) {
            continue;
        }
        match b.node(n) {
            None if n == leaf => return true,
            None => {}
            Some((a, lo, hi)) if a == atom => work.push(if value { hi } else { lo }),
            Some((_, lo, hi)) => work.extend([lo, hi]),
        }
    }
    false
}
