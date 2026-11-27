//! Whole-wave sections (decision 0003): code between `s_or_saveexec -1`
//! (or `s_nand_saveexec 0`) and the restore of the saved mask runs in every lane of the console's
//! wave, the inactive ones included, which is how the compiler expands
//! HLSL wave reductions and prefix sums (an identity put into the inactive
//! lanes with `v_cndmask`, then DPP, `v_permlanex16` and `v_readlane`
//! steps). A host lane that is elsewhere cannot run that code, so each
//! section is simulated on the console's lanes with symbolic values, and a
//! value it leaves is lowered by what it turned out to be: the reduction or
//! prefix of one operation over the lanes' (masked) inputs, which DXIL's
//! wave operations compute over the host's active lanes. No instruction
//! sequence is matched; any step the simulation does not model leaves the
//! values it writes unknown.

use std::collections::HashMap;

use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::ir::{EXEC, Inst, Program, Reg, Value};

/// An associative operation and its identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Op {
    AddU,
    AddF,
    /// A sum of 16-bit floats in the low halves (`v_add_f16`), each addition rounded to 16
    /// bits: not associative, so only the balanced tree over the lanes in their order is
    /// followed (see [`buddies`]), which is how the compiler sums a wave.
    AddH,
    MinU,
    MaxU,
    MinI,
    MaxI,
    MinF,
    MaxF,
    Or,
    And,
    Xor,
}

impl Op {
    fn of(name: &str) -> Option<Op> {
        Some(match name {
            // the low dword of a 64-bit sum (`v_add_co_ci_u32` carries into
            // the high one) is the 32-bit sum of the low dwords
            "v_add_nc_u32" | "v_add_co_u32" => Op::AddU,
            "v_add_f32" => Op::AddF,
            "v_add_f16" => Op::AddH,
            "v_min_u32" => Op::MinU,
            "v_max_u32" => Op::MaxU,
            "v_min_i32" => Op::MinI,
            "v_max_i32" => Op::MaxI,
            "v_min_f32" => Op::MinF,
            "v_max_f32" => Op::MaxF,
            "v_or_b32" => Op::Or,
            "v_and_b32" => Op::And,
            "v_xor_b32" => Op::Xor,
            _ => return None,
        })
    }

    /// Whether `k` leaves the other operand unchanged.
    fn identity(self, k: u32) -> bool {
        match self {
            Op::AddU | Op::AddH | Op::Or | Op::Xor | Op::MaxU => k == 0,
            Op::AddF => k == 0 || k == 0x8000_0000,
            Op::MinU | Op::And => k == u32::MAX,
            Op::MinI => k == 0x7FFF_FFFF,
            Op::MaxI => k == 0x8000_0000,
            Op::MinF => k == 0x7F80_0000,
            Op::MaxF => k == 0xFF80_0000,
        }
    }

    /// Counting a lane twice changes the result.
    fn counts(self) -> bool {
        matches!(self, Op::AddU | Op::AddF | Op::AddH | Op::Xor)
    }
}

/// Whether the lane sets `a` and `b` are the two halves of one aligned run of lanes: lanes
/// 2k and 2k+1, then two such pairs side by side, and so on up to the two halves of the
/// wave. Sums joined only that way form the balanced tree over the lanes in their order,
/// whichever of the two a lane adds to the other (the addition is commutative).
fn buddies(a: u64, b: u64) -> bool {
    let n = a.count_ones();
    if n != b.count_ones() || !n.is_power_of_two() || n == 64 {
        return false;
    }
    let run = |s: u64| {
        let first = s.trailing_zeros();
        (first.is_multiple_of(n) && s >> first == (1u64 << n) - 1).then_some(first)
    };
    match (run(a), run(b)) {
        (Some(x), Some(y)) => x.min(y) % (2 * n) == 0 && x.abs_diff(y) == n,
        _ => false,
    }
}

/// The order in which a sum of floats was added up, where every addition joined two sets of
/// lanes that are each other's image under one lane bit (`l ^ (1 << d)`), both added in the
/// same order: the bits `d` in the order they were joined (three bits each above a count in
/// the low three), or [`Butterfly::UNKNOWN`]. Such a sum is the butterfly over those bits, which the
/// paired form replays addition by addition ([`Section::butterflies`]). Two symbols that differ only
/// in it are the same value to the rest of the simulation (equality ignores it), so following
/// it changes nothing else the simulation finds.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Butterfly(pub(crate) u32);

impl Butterfly {
    /// One lane's term: nothing added yet.
    pub(crate) const LEAF: Butterfly = Butterfly(0);
    /// An order that is no butterfly, or not followed.
    pub(crate) const UNKNOWN: Butterfly = Butterfly(u32::MAX);

    /// The lane bits, in the order they were joined.
    pub(crate) fn bits(self) -> Option<Vec<u32>> {
        (self.0 != u32::MAX).then(|| (0..self.0 & 7).map(|j| self.0 >> (3 + 3 * j) & 7).collect())
    }

    /// `a` (lanes `la`) joined with `b` (lanes `lb`): the butterfly one bit further, or unknown.
    fn join(a: Butterfly, la: u64, b: Butterfly, lb: u64) -> Butterfly {
        let (Some(bits), true) = (a.bits(), a.0 == b.0) else {
            return Butterfly::UNKNOWN;
        };
        let n = bits.len() as u32;
        match (0..6u32).find(|&d| !bits.contains(&d) && image(la, d) == lb) {
            Some(d) if n < 6 => Butterfly((a.0 & !7) | (n + 1) | d << (3 + 3 * n)),
            _ => Butterfly::UNKNOWN,
        }
    }
}

impl PartialEq for Butterfly {
    fn eq(&self, _: &Butterfly) -> bool {
        true
    }
}

impl Eq for Butterfly {}

/// The lanes `l ^ (1 << d)` of the lanes in `set`.
fn image(set: u64, d: u32) -> u64 {
    const LOW: [u64; 6] = [
        0x5555_5555_5555_5555,
        0x3333_3333_3333_3333,
        0x0F0F_0F0F_0F0F_0F0F,
        0x00FF_00FF_00FF_00FF,
        0x0000_FFFF_0000_FFFF,
        0x0000_0000_FFFF_FFFF,
    ];
    let (low, k) = (LOW[d as usize], 1u32 << d);
    ((set & low) << k) | ((set & !low) >> k)
}

/// One step of how a section adds up a sum ([`Order`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Step {
    /// A lane's term (`mask ? src : ident`).
    Lane(u8),
    /// A constant (the 0 a DPP shift reads past the end of its row).
    K(u32),
    /// The sum of two earlier steps.
    Add(u32, u32),
}

/// How a section adds up a sum of floats it leaves: which terms it adds to which, as the
/// instructions do. Each addition rounds, so the same terms in another order give another
/// sum (2^25, 1, -2^25 and 0 in four lanes: 0 added as neighbours, 1 with lanes two apart
/// first). A lowering that adds the terms up itself follows this.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Order {
    /// Each step after the steps it adds.
    pub steps: Vec<Step>,
    /// The step that is the value: one where every lane holds the same sum added up the
    /// same way, else one per lane (a prefix; a sum each lane adds up in its own order).
    pub results: Vec<u32>,
}

impl Order {
    /// Whether this is one sum of all 64 lanes made as the compiler's expansion of a wave
    /// sum makes it: lanes 2k and 2k+1, then those pairs side by side, and so on up to the
    /// two halves ([`buddies`]), with nothing else added in.
    pub(crate) fn neighbours_first(&self) -> bool {
        let mut lanes: Vec<u64> = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
            lanes.push(match *step {
                Step::Lane(l) => 1 << l,
                Step::Add(a, b) if buddies(lanes[a as usize], lanes[b as usize]) => {
                    lanes[a as usize] | lanes[b as usize]
                }
                _ => return false,
            });
        }
        self.results.len() == 1 && lanes[self.results[0] as usize] == u64::MAX
    }
}

/// The step ([`Steps`]) a lane's sum is; `None`: not followed. Not part of what a lane's
/// value is (two lanes hold the same sum whichever way each added it up), so every `Tree`
/// is equal to every other.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Tree(Option<u32>);

impl Tree {
    pub(crate) const NONE: Tree = Tree(None);
}

impl PartialEq for Tree {
    fn eq(&self, _: &Tree) -> bool {
        true
    }
}

impl Eq for Tree {}

/// The steps of a section's sums, each kept once (an addition with its terms in either
/// order: it is commutative).
#[derive(Default)]
struct Steps {
    all: Vec<Step>,
    index: HashMap<Step, u32>,
}

impl Steps {
    fn of(&mut self, step: Step) -> Tree {
        let n = *self.index.entry(step).or_insert_with(|| {
            self.all.push(step);
            self.all.len() as u32 - 1
        });
        Tree(Some(n))
    }

    fn add(&mut self, a: Tree, b: Tree) -> Tree {
        match (a.0, b.0) {
            (Some(a), Some(b)) => self.of(Step::Add(a.min(b), a.max(b))),
            _ => Tree::NONE,
        }
    }

    /// The steps `results` are made of, renumbered; `None` where one is not followed.
    fn order(&self, results: &[Tree]) -> Option<Order> {
        let results: Vec<u32> = results.iter().map(|t| t.0).collect::<Option<_>>()?;
        let mut used = vec![false; self.all.len()];
        let mut stack = results.clone();
        while let Some(n) = stack.pop() {
            if !std::mem::replace(&mut used[n as usize], true)
                && let Step::Add(a, b) = self.all[n as usize]
            {
                stack.extend([a, b]);
            }
        }
        // (a step is made after the steps it adds: their order is kept)
        let mut place = vec![0u32; self.all.len()];
        let mut steps = Vec::new();
        for (n, step) in self.all.iter().enumerate().filter(|&(n, _)| used[n]) {
            place[n] = steps.len() as u32;
            steps.push(match *step {
                Step::Add(a, b) => Step::Add(place[a as usize], place[b as usize]),
                other => other,
            });
        }
        let mut results: Vec<u32> = results.iter().map(|&n| place[n as usize]).collect();
        if results.iter().all(|&r| r == results[0]) {
            results.truncate(1);
        }
        Some(Order { steps, results })
    }
}

/// A lane's value inside a section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sym {
    Unknown,
    K(u32),
    /// The lane's own value of a VGPR value from before the section.
    Own(Value),
    /// A uniform value from before the section.
    Uni(Value),
    /// `op` over `lanes` of `mask ? src : ident` (`op` none: one lane), a float sum added in
    /// the order `tree` (and, where that order is a butterfly, over the bits `butterfly`).
    Red {
        op: Option<Op>,
        src: Value,
        mask: Value,
        ident: u32,
        lanes: u64,
        /// How the lane added it up.
        tree: Tree,
        /// The same order as lane bits, for the paired form.
        butterfly: Butterfly,
    },
    /// Low (`!high`) or high (`high`) 32 bits of the 64-bit float sum over
    /// `lanes` of `mask ? f64(lo, hi) : 0.0`, added in the order `butterfly`.
    Red64F {
        lo: Value,
        hi: Value,
        mask: Value,
        lanes: u64,
        high: bool,
        butterfly: Butterfly,
    },
}

/// What a value a section leaves holds in every lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shape {
    K(u32),
    Value(Value),
    /// The lane's own `mask ? src : ident`.
    Masked(Value, Value, u32),
    /// `op` over every lane.
    Active(Op, Value, Value, u32),
    /// Low (`!high`) or high (`high`) 32 bits of the 64-bit float sum over every lane.
    ActiveF64(Value, Value, Value, bool),
    /// `op` over the lanes below (`inclusive`: and including) each lane.
    Prefix(Op, Value, Value, u32, bool),
}

/// The section starting at instruction `start` of a block: its restore
/// index and the shapes of the values it writes.
pub(crate) struct Section {
    pub end: usize,
    pub shapes: HashMap<Value, Shape>,
    /// How each sum of floats among them (`Op::AddF`, over the wave or a prefix) is added up.
    pub orders: HashMap<Value, Order>,
    /// For a float sum over every lane (`Shape::Active` of [`Op::AddF`], `Shape::ActiveF64`)
    /// added as a butterfly: its lane bits in the order the console joins them
    /// ([`Butterfly`]). The paired form's record; the meeting form reads [`Section::orders`].
    pub butterflies: HashMap<Value, Vec<u32>>,
}

/// Every whole-wave section of `block`, by start index.
pub(crate) fn sections(p: &Program, block: usize) -> HashMap<usize, Section> {
    let insts = &p.blocks[block].insts;
    let wave: u32 = if p.wave64 { 64 } else { 32 };
    let mut out = HashMap::new();
    let mut i = 0;
    while i < insts.len() {
        let inst = &insts[i];
        // EXEC becomes every lane: `s_or_saveexec -1` or `s_nand_saveexec 0`.
        let all = match inst.name() {
            "s_or_saveexec_b64" | "s_or_saveexec_b32" => Some(Operand::Const(193)),
            "s_nand_saveexec_b64" | "s_nand_saveexec_b32" => Some(Operand::Const(128)),
            _ => None,
        };
        let start = all.is_some_and(
            |k| matches!(&inst.fields, Fields::Sop { src, .. } if src.first() == Some(&k)),
        );
        let saved = match &inst.fields {
            Fields::Sop {
                dst: Some(Operand::Scalar { code, .. }),
                ..
            } if start => inst.write(Reg::S(*code)),
            _ => None,
        };
        let Some(saved) = saved else {
            i += 1;
            continue;
        };
        let end = (i + 1..insts.len()).find(|&j| {
            insts[j].write(Reg::S(EXEC)).is_some()
                && insts[j].read(Reg::S(saved_code(inst))) == Some(saved)
        });
        let Some(end) = end else {
            break;
        };
        let (shapes, orders, butterflies) = simulate(&insts[i + 1..end], wave, saved);
        out.insert(
            i,
            Section {
                end,
                shapes,
                orders,
                butterflies,
            },
        );
        i = end + 1;
    }
    out
}

fn saved_code(inst: &Inst) -> u8 {
    match &inst.fields {
        Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            ..
        } => *code,
        _ => NULL,
    }
}

/// The lane `l` reads with DPP, whether it reads 0 instead, and whether it
/// writes; `None` for patterns not modelled.
fn dpp_lane(d: Dpp, l: u32) -> Option<(u32, bool, bool)> {
    match d {
        Dpp::Lanes { sel, .. } => Some(((l & !7) | (sel >> (3 * (l & 7)) & 7), false, true)),
        Dpp::Row {
            ctrl,
            row_mask,
            bank_mask,
            bound_ctrl,
            ..
        } => {
            let (row, pos, n) = (l & !15, l & 15, u32::from(ctrl & 15));
            let enabled = row_mask >> (l >> 4 & 3) & 1 != 0 && bank_mask >> (l >> 2 & 3) & 1 != 0;
            let (src, inside) = match ctrl {
                0..=0xFF => ((l & !3) | (u32::from(ctrl) >> (2 * (l & 3)) & 3), true),
                0x101..=0x10F => (l + n, pos + n < 16),
                0x111..=0x11F => (l.wrapping_sub(n), pos >= n),
                0x121..=0x12F => (row | (pos.wrapping_sub(n) & 15), true),
                0x140 => (l ^ 15, true),
                0x141 => (l ^ 7, true),
                0x150..=0x15F => (row | n, true),
                0x160..=0x16F => (l ^ n, true),
                _ => return None,
            };
            match (inside, bound_ctrl) {
                (true, _) => Some((src, false, enabled)),
                (false, true) => Some((l, true, enabled)),
                (false, false) => Some((l, false, false)),
            }
        }
    }
}

fn combine(op: Op, a: Sym, b: Sym, steps: &mut Steps) -> Sym {
    match (a, b) {
        // Of floats, only -0.0 leaves every other operand as it is: +0.0 added to -0.0
        // gives +0.0. So two constants of which one is a zero are added here (the other
        // constant; of two zeros -0.0 only if both are, whichever comes first), ...
        (Sym::K(x), Sym::K(y)) if op == Op::AddF && (op.identity(x) || op.identity(y)) => {
            Sym::K(if op.identity(x) && op.identity(y) {
                x & y
            } else if op.identity(x) {
                y
            } else {
                x
            })
        }
        // ... a sum of floats, or one lane's term of one, keeps the +0.0 as a step of its
        // own and is a sum of floats from then on (not the lane's term as it was; no
        // butterfly over the lanes, which has no such step), ...
        (Sym::K(0), Sym::Red { op: o, src, mask, ident, lanes, tree, .. })
        | (Sym::Red { op: o, src, mask, ident, lanes, tree, .. }, Sym::K(0))
            if op == Op::AddF && o.is_none_or(|o| o == Op::AddF) && op.identity(ident) =>
        {
            let zero = steps.of(Step::K(0));
            Sym::Red {
                op: Some(Op::AddF),
                src,
                mask,
                ident,
                lanes,
                tree: steps.add(zero, tree),
                butterfly: Butterfly::UNKNOWN,
            }
        }
        // ... and anything else +0.0 is added to (a minimum or a sum of integers over the
        // lanes, a value from before the section) is not followed: it is no longer what
        // it was, where it was -0.0.
        (Sym::K(0), _) | (_, Sym::K(0)) if op == Op::AddF => Sym::Unknown,
        (Sym::K(k), x) | (x, Sym::K(k)) if op.identity(k) => x,
        (
            Sym::Red {
                op: oa,
                src,
                mask,
                ident,
                lanes: la,
                tree: ta,
                butterfly: ba,
            },
            Sym::Red {
                op: ob,
                src: sb,
                mask: mb,
                ident: ib,
                lanes: lb,
                tree: tb,
                butterfly: bb,
            },
        ) if (src, mask, ident) == (sb, mb, ib)
            && op.identity(ident)
            && oa.is_none_or(|o| o == op)
            && ob.is_none_or(|o| o == op)
            && (!op.counts() || la & lb == 0)
            && (op != Op::AddH || buddies(la, lb)) =>
        {
            Sym::Red {
                op: Some(op),
                src,
                mask,
                ident,
                lanes: la | lb,
                tree: steps.add(ta, tb),
                butterfly: match op {
                    Op::AddF => Butterfly::join(ba, la, bb, lb),
                    _ => Butterfly::UNKNOWN,
                },
            }
        }
        _ => Sym::Unknown,
    }
}

/// A section's values: their shapes, how each float sum is added up ([`Order`]) and, of
/// those that are butterflies, the lane bits ([`Butterfly`]).
type Simulated = (HashMap<Value, Shape>, HashMap<Value, Order>, HashMap<Value, Vec<u32>>);

fn simulate(insts: &[Inst], wave: u32, saved: Value) -> Simulated {
    let all = if wave == 64 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    };
    let mut exec = all;
    let mut vec: HashMap<Value, Vec<Sym>> = HashMap::new();
    let mut uni: HashMap<Value, Sym> = HashMap::new();
    let mut steps = Steps::default();
    // Masks the section derives from the saved EXEC (`s_and` with it).
    let mut masked = vec![saved];
    for inst in insts {
        let name = inst.name();
        if inst.write(Reg::S(EXEC)).is_some() {
            // Constant lane sets (prefix sums); anything else ends what
            // the simulation can follow.
            let lo = name
                .starts_with("s_mov_b")
                .then(|| constant(inst))
                .flatten();
            let Some(k) = lo else {
                return shapes(vec, uni, wave, &mut steps);
            };
            exec = (if name == "s_mov_b64" {
                k
            } else {
                (exec & !0xFFFF_FFFF) | (k & 0xFFFF_FFFF)
            }) & all;
            continue;
        }
        if inst.write(Reg::S(EXEC + 1)).is_some() {
            let Some(k) = name
                .starts_with("s_mov_b")
                .then(|| constant(inst))
                .flatten()
            else {
                return shapes(vec, uni, wave, &mut steps);
            };
            exec = ((exec & 0xFFFF_FFFF) | ((k & 0xFFFF_FFFF) << 32)) & all;
            continue;
        }
        if let Fields::Sop { dst, src, .. } = &inst.fields {
            if name.starts_with("s_and_b")
                && src.iter().any(|o| matches!(*o, Operand::Scalar { code, .. } if inst.read(Reg::S(code)).is_some_and(|v| masked.contains(&v))))
                && let Some(&(_, w)) = inst.writes.iter().find(|w| matches!(w.0, Reg::S(_)))
            {
                masked.push(w);
            }
            if name == "s_mov_b32"
                && let (Some(Operand::Scalar { code: dc, .. }), Some(s0)) = (dst, src.first())
                && *dc != NULL
                && let Some(w) = inst.write(Reg::S(*dc))
            {
                let sym = match *s0 {
                    Operand::Scalar { code, .. } if code != NULL => inst
                        .read(Reg::S(code))
                        .map_or(Sym::Unknown, |x| uni.get(&x).copied().unwrap_or(Sym::Uni(x))),
                    Operand::Const(c) => crate::resource::inline(c).map_or(Sym::Unknown, Sym::K),
                    Operand::Literal(x) => Sym::K(x),
                    _ => Sym::Unknown,
                };
                uni.insert(w, sym);
            }
            for &(_, w) in &inst.writes {
                uni.entry(w).or_insert(Sym::Unknown);
            }
            continue;
        }
        let Fields::Valu(v) = &inst.fields else {
            for &(_, w) in &inst.writes {
                vec.insert(w, vec![Sym::Unknown; wave as usize]);
            }
            continue;
        };
        let lanes = |op: &Operand,
                     vec: &HashMap<Value, Vec<Sym>>,
                     uni: &HashMap<Value, Sym>|
         -> Vec<Sym> {
            let one = |s: Sym| vec![s; wave as usize];
            match *op {
                Operand::Vector { reg, .. } => match inst.read(Reg::V(reg)) {
                    Some(x) => vec
                        .get(&x)
                        .cloned()
                        .unwrap_or_else(|| (0..wave).map(|_| Sym::Own(x)).collect()),
                    None => one(Sym::Unknown),
                },
                Operand::Scalar { code, .. } if code != NULL => match inst.read(Reg::S(code)) {
                    Some(x) => one(uni.get(&x).copied().unwrap_or(Sym::Uni(x))),
                    None => one(Sym::Unknown),
                },
                Operand::Const(c) => one(crate::resource::inline(c).map_or(Sym::Unknown, Sym::K)),
                Operand::Literal(x) => one(Sym::K(x)),
                _ => one(Sym::Unknown),
            }
        };
        let modifiers = v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0 || v.sdwa.is_some();
        if name == "v_readlane_b32" {
            let (Some(Operand::Scalar { code, .. }), Some(src), Some(lane)) =
                (v.dst, v.src.first(), v.src.get(1))
            else {
                continue;
            };
            let l = match *lane {
                Operand::Const(c) => crate::resource::inline(c),
                Operand::Literal(x) => Some(x),
                _ => None,
            };
            let x = lanes(src, &vec, &uni);
            if let Some(w) = inst.write(Reg::S(code)) {
                let s = l
                    .and_then(|l| x.get(l as usize).copied())
                    .unwrap_or(Sym::Unknown);
                uni.insert(w, s);
            }
            continue;
        }
        if name == "v_add_f64"
            && !modifiers
            && v.dpp.is_none()
            && let Some(Operand::Vector { reg: dreg, .. }) = v.dst
            && let (Some(dst_lo), Some(dst_hi)) =
                (inst.write(Reg::V(dreg)), inst.write(Reg::V(dreg + 1)))
        {
            let pair_lanes = |op: Option<&Operand>| -> Option<(Vec<Sym>, Vec<Sym>)> {
                match *op? {
                    Operand::Vector { reg, .. } => Some((
                        lanes(&Operand::Vector { reg, dwords: 1 }, &vec, &uni),
                        lanes(
                            &Operand::Vector {
                                reg: reg + 1,
                                dwords: 1,
                            },
                            &vec,
                            &uni,
                        ),
                    )),
                    Operand::Scalar { code, .. } if code != NULL => Some((
                        lanes(&Operand::Scalar { code, dwords: 1 }, &vec, &uni),
                        lanes(
                            &Operand::Scalar {
                                code: code + 1,
                                dwords: 1,
                            },
                            &vec,
                            &uni,
                        ),
                    )),
                    _ => None,
                }
            };
            let old_lo = lanes(
                &Operand::Vector {
                    reg: dreg,
                    dwords: 1,
                },
                &vec,
                &uni,
            );
            let old_hi = lanes(
                &Operand::Vector {
                    reg: dreg + 1,
                    dwords: 1,
                },
                &vec,
                &uni,
            );
            let f64_term = |lo_s: Sym, hi_s: Sym| -> Option<(Value, Value, Value, u64, Butterfly)> {
                match (lo_s, hi_s) {
                    (
                        Sym::Red {
                            op: None,
                            src: lo,
                            mask: ml,
                            ident: 0,
                            lanes: ll,
                            ..
                        },
                        Sym::Red {
                            op: None,
                            src: hi,
                            mask: mh,
                            ident: 0,
                            lanes: lh,
                            ..
                        },
                    ) if ml == mh && ll == lh => Some((lo, hi, ml, ll, Butterfly::LEAF)),
                    (
                        Sym::Red64F {
                            lo: lo1,
                            hi: hi1,
                            mask: m1,
                            lanes: l1,
                            high: false,
                            butterfly: t1,
                        },
                        Sym::Red64F {
                            lo: lo2,
                            hi: hi2,
                            mask: m2,
                            lanes: l2,
                            high: true,
                            butterfly: t2,
                        },
                    ) if (lo1, hi1, m1, l1) == (lo2, hi2, m2, l2) => {
                        // (halves added in two orders: the same sum, but no one butterfly)
                        Some((lo1, hi1, m1, l1, if t1.0 == t2.0 { t1 } else { Butterfly::UNKNOWN }))
                    }
                    _ => None,
                }
            };
            let mut out_lo = vec![Sym::Unknown; wave as usize];
            let mut out_hi = vec![Sym::Unknown; wave as usize];
            if let (Some((alo, ahi)), Some((blo, bhi))) =
                (pair_lanes(v.src.first()), pair_lanes(v.src.get(1)))
            {
                for l in 0..wave as usize {
                    let (rl, rh) = match (f64_term(alo[l], ahi[l]), f64_term(blo[l], bhi[l])) {
                        (Some((lo1, hi1, m1, la, ta)), Some((lo2, hi2, m2, lb, tb)))
                            if (lo1, hi1, m1) == (lo2, hi2, m2) && la & lb == 0 =>
                        {
                            let butterfly = Butterfly::join(ta, la, tb, lb);
                            (
                                Sym::Red64F {
                                    lo: lo1,
                                    hi: hi1,
                                    mask: m1,
                                    lanes: la | lb,
                                    high: false,
                                    butterfly,
                                },
                                Sym::Red64F {
                                    lo: lo1,
                                    hi: hi1,
                                    mask: m1,
                                    lanes: la | lb,
                                    high: true,
                                    butterfly,
                                },
                            )
                        }
                        _ => (Sym::Unknown, Sym::Unknown),
                    };
                    out_lo[l] = if exec >> l & 1 != 0 { rl } else { old_lo[l] };
                    out_hi[l] = if exec >> l & 1 != 0 { rh } else { old_hi[l] };
                }
            }
            vec.insert(dst_lo, out_lo);
            vec.insert(dst_hi, out_hi);
            continue;
        }
        let Some(Operand::Vector { reg: dreg, .. }) = v.dst else {
            for &(_, w) in &inst.writes {
                uni.insert(w, Sym::Unknown);
            }
            continue;
        };
        let Some(dst) = inst.write(Reg::V(dreg)) else {
            continue;
        };
        let old = lanes(
            &Operand::Vector {
                reg: dreg,
                dwords: 1,
            },
            &vec,
            &uni,
        );
        // src0 as each lane fetches it (DPP), with the lanes that write.
        let src0 = v.src.first().map(|o| lanes(o, &vec, &uni));
        let mut writes = exec;
        let fetched: Option<Vec<Sym>> = match (v.dpp, &src0) {
            (Some(d), Some(s)) => {
                if exec != all {
                    vec.insert(dst, vec![Sym::Unknown; wave as usize]);
                    continue;
                }
                let mut f = Vec::with_capacity(wave as usize);
                let mut ok = true;
                for l in 0..wave {
                    match dpp_lane(d, l) {
                        Some((src, zero, on)) if src < wave => {
                            f.push(if zero { Sym::K(0) } else { s[src as usize] });
                            if !on {
                                writes &= !(1u64 << l);
                            }
                        }
                        _ => ok = false,
                    }
                }
                ok.then_some(f)
            }
            (None, Some(s)) => Some(s.clone()),
            _ => None,
        };
        let src = |k: usize| v.src.get(k).map(|o| lanes(o, &vec, &uni));
        let result: Vec<Sym> = match (name, fetched) {
            _ if modifiers => vec![Sym::Unknown; wave as usize],
            ("v_mov_b32", Some(a)) => a,
            ("v_bfrev_b32", Some(a)) => a
                .into_iter()
                .map(|s| match s {
                    Sym::K(k) => Sym::K(k.reverse_bits()),
                    _ => Sym::Unknown,
                })
                .collect(),
            ("v_cndmask_b32", Some(a)) => {
                let mask = match v.src.get(2) {
                    Some(&Operand::Scalar { code, .. }) => inst.read(Reg::S(code)),
                    _ => None,
                };
                let b = src(1).unwrap_or_default();
                (0..wave as usize)
                    .map(|l| match (a[l], b.get(l).copied(), mask) {
                        (Sym::K(ident), Some(Sym::Own(x)), Some(m)) if masked.contains(&m) => {
                            Sym::Red {
                                op: None,
                                src: x,
                                mask: m,
                                ident,
                                lanes: 1 << l,
                                tree: steps.of(Step::Lane(l as u8)),
                                butterfly: Butterfly::LEAF,
                            }
                        }
                        _ => Sym::Unknown,
                    })
                    .collect()
            }
            ("v_permlanex16_b32", Some(a)) => {
                // Lanes of each row read the other row of their half; lane
                // i's pick is nibble i of {src2, src1}.
                let sel = |k: usize| match v.src.get(k) {
                    Some(&Operand::Const(c)) => crate::resource::inline(c),
                    Some(&Operand::Literal(x)) => Some(x),
                    _ => None,
                };
                match (sel(1), sel(2)) {
                    (Some(s1), Some(s2)) => {
                        let s = u64::from(s1) | u64::from(s2) << 32;
                        (0..wave)
                            .map(|l| {
                                let other = (l & !31) | ((l & 16) ^ 16);
                                let pick = (s >> (4 * (l & 15)) & 15) as u32;
                                a[(other | pick) as usize]
                            })
                            .collect()
                    }
                    _ => vec![Sym::Unknown; wave as usize],
                }
            }
            ("v_sub_nc_u32", Some(a)) => {
                // A prefix sum less the lane's own term: the exclusive one.
                let b = src(1).unwrap_or_default();
                (0..wave as usize)
                    .map(|l| match (a[l], b.get(l).copied()) {
                        (
                            Sym::Red {
                                op: None | Some(Op::AddU),
                                src,
                                mask,
                                ident: 0,
                                lanes,
                                ..
                            },
                            Some(Sym::Red {
                                op: None | Some(Op::AddU),
                                src: s2,
                                mask: m2,
                                ident: 0,
                                lanes: l2,
                                ..
                            }),
                        ) if (src, mask) == (s2, m2) && lanes & l2 == l2 => {
                            let rest = lanes & !l2;
                            if rest == 0 {
                                Sym::K(0)
                            } else {
                                Sym::Red {
                                    op: Some(Op::AddU),
                                    src,
                                    mask,
                                    ident: 0,
                                    lanes: rest,
                                    tree: Tree::NONE,
                                    butterfly: Butterfly::UNKNOWN,
                                }
                            }
                        }
                        _ => Sym::Unknown,
                    })
                    .collect()
            }
            (n, Some(a)) => match (Op::of(n), src(1)) {
                (Some(op), Some(b)) => (0..wave as usize)
                    .map(|l| combine(op, a[l], b[l], &mut steps))
                    .collect(),
                _ => vec![Sym::Unknown; wave as usize],
            },
            _ => vec![Sym::Unknown; wave as usize],
        };
        let merged = (0..wave as usize)
            .map(|l| {
                if writes >> l & 1 != 0 {
                    result[l]
                } else {
                    old[l]
                }
            })
            .collect();
        vec.insert(dst, merged);
    }
    shapes(vec, uni, wave, &mut steps)
}

fn constant(inst: &Inst) -> Option<u64> {
    let Fields::Sop { src, .. } = &inst.fields else {
        return None;
    };
    let wide = inst.name() == "s_mov_b64";
    match *src.first()? {
        Operand::Scalar { code: NULL, .. } => Some(0),
        Operand::Const(c @ 128..=208) => {
            let x = crate::resource::inline(c)?;
            Some(if wide {
                i64::from(x as i32) as u64
            } else {
                u64::from(x)
            })
        }
        Operand::Const(c) if !wide => crate::resource::inline(c).map(u64::from),
        Operand::Literal(x) => Some(u64::from(x)),
        _ => None,
    }
}

/// The shape of each value, where every lane agrees on one; how each sum of floats among
/// them is added up; and of those that are butterflies over every lane, the lane bits.
fn shapes(vec: HashMap<Value, Vec<Sym>>, uni: HashMap<Value, Sym>, wave: u32, steps: &mut Steps) -> Simulated {
    let all = if wave == 64 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    };
    let mut out = HashMap::new();
    let mut orders = HashMap::new();
    let mut butterflies = HashMap::new();
    // what a lane of a sum of floats holds, as a step
    let mut tree = |s: Sym| match s {
        Sym::Red { tree, .. } => tree,
        Sym::K(k) => steps.of(Step::K(k)),
        _ => Tree::NONE,
    };
    let float_sum = |sh: Shape| matches!(sh, Shape::Active(Op::AddF, ..) | Shape::Prefix(Op::AddF, ..));
    let single = |s: Sym| match s {
        Sym::K(k) => Some(Shape::K(k)),
        Sym::Own(x) | Sym::Uni(x) => Some(Shape::Value(x)),
        Sym::Red {
            op: Some(op),
            src,
            mask,
            ident,
            lanes,
            ..
        } if lanes == all => Some(Shape::Active(op, src, mask, ident)),
        Sym::Red64F {
            lo,
            hi,
            mask,
            lanes,
            high,
            ..
        } if lanes == all => Some(Shape::ActiveF64(lo, hi, mask, high)),
        _ => None,
    };
    // a float sum's butterfly over every lane bit ([`Butterfly`])
    let butterfly = |s: Sym| match s {
        Sym::Red {
            op: Some(Op::AddF),
            butterfly,
            ..
        }
        | Sym::Red64F { butterfly, .. } => butterfly.bits().filter(|b| b.len() as u32 == wave.trailing_zeros()),
        _ => None,
    };
    for (v, s) in uni {
        if let Some(sh) = single(s) {
            out.insert(v, sh);
            if float_sum(sh) {
                orders.insert(v, vec![tree(s)]);
            }
            if let Some(b) = butterfly(s) {
                butterflies.insert(v, b);
            }
        }
    }
    for (v, lanes) in vec {
        if let Some(sh) = per_lane(&lanes, all) {
            out.insert(v, sh);
            if float_sum(sh) {
                orders.insert(v, lanes.iter().map(|&s| tree(s)).collect());
            }
            // (every lane the same sum: added in the same order in every lane)
            let bits = |s: &Sym| match *s {
                Sym::Red { butterfly, .. } | Sym::Red64F { butterfly, .. } => Some(butterfly.0),
                _ => None,
            };
            if matches!(sh, Shape::Active(..) | Shape::ActiveF64(..))
                && lanes.iter().all(|s| bits(s) == bits(&lanes[0]))
                && let Some(b) = butterfly(lanes[0])
            {
                butterflies.insert(v, b);
            }
        }
    }
    let orders = orders.into_iter().filter_map(|(v, results)| Some((v, steps.order(&results)?))).collect();
    (out, orders, butterflies)
}

fn per_lane(lanes: &[Sym], all: u64) -> Option<Shape> {
    let first = lanes[0];
    let own = |l: usize, s: Sym| match s {
        Sym::Red {
            op: None,
            src,
            mask,
            ident,
            lanes,
            ..
        } if lanes == 1 << l => Some((src, mask, ident)),
        _ => None,
    };
    if let Some(m) = own(0, first)
        && lanes.iter().enumerate().all(|(l, &s)| own(l, s) == Some(m))
    {
        return Some(Shape::Masked(m.0, m.1, m.2));
    }
    if lanes.iter().all(|&s| s == first) {
        return match first {
            Sym::K(k) => Some(Shape::K(k)),
            Sym::Uni(x) | Sym::Own(x) => Some(Shape::Value(x)),
            Sym::Red {
                op: Some(op),
                src,
                mask,
                ident,
                lanes,
                ..
            } if lanes == all => Some(Shape::Active(op, src, mask, ident)),
            Sym::Red64F {
                lo,
                hi,
                mask,
                lanes,
                high,
                ..
            } if lanes == all => Some(Shape::ActiveF64(lo, hi, mask, high)),
            _ => None,
        };
    }
    // Prefixes: lane l holds lanes 0..l (exclusive) or 0..=l (inclusive).
    let mut meta: Option<(Value, Value, u32, bool)> = None;
    let mut op_found: Option<Op> = None;
    for (l, &s) in lanes.iter().enumerate() {
        let below = (1u64 << l) - 1;
        let (op, src, mask, ident, set) = match s {
            Sym::Red {
                op,
                src,
                mask,
                ident,
                lanes,
                ..
            } => (op, src, mask, ident, lanes),
            // Lane 0 of an exclusive prefix holds the identity.
            Sym::K(_) if l == 0 => continue,
            _ => return None,
        };
        let inclusive = set == below | (1u64 << l);
        if !inclusive && set != below {
            return None;
        }
        if l == 0 && !inclusive {
            return None;
        }
        if op.is_none() && set.count_ones() != 1 {
            return None;
        }
        let lane_meta = (src, mask, ident, inclusive);
        match meta {
            None => meta = Some(lane_meta),
            Some(prev) if prev == lane_meta => {}
            _ => return None,
        }
        if let Some(o) = op {
            match op_found {
                None => op_found = Some(o),
                Some(prev) if prev == o => {}
                _ => return None,
            }
        }
    }
    let (src, mask, ident, inclusive) = meta?;
    let op = op_found?;
    if !op.identity(ident) {
        return None;
    }
    match lanes[0] {
        Sym::K(k) if !inclusive && k == ident => {}
        Sym::Red { .. } if inclusive => {}
        _ => return None,
    }
    Some(Shape::Prefix(op, src, mask, ident, inclusive))
}

/// Test programs: a reduction over a whole console wave of 64 lanes in the steps the console
/// compiler expands one to, put together here field by field from the instruction set's
/// formats. The steps are the case the tests stand for (neighbours, pairs, fours, the two
/// halves of a row of 16, the two rows of a half, the two halves of the wave); the registers
/// are the fixture's own choice.
#[cfg(test)]
pub(super) mod reduction {
    /// The VGPR the reduction is made in, the VGPR for the other row's lanes, and the first
    /// SGPR of the pair EXEC is saved in.
    pub(crate) const WORK: u8 = 14;
    pub(crate) const TMP: u8 = 15;
    pub(crate) const SAVED: u8 = 26;
    /// VOP2 opcodes of the operations reduced.
    pub(crate) const V_ADD_F32: u8 = 3;
    pub(crate) const V_OR_B32: u8 = 28;
    pub(crate) const V_ADD_F16: u8 = 50;
    /// Each lane with its neighbour, with the lane two away (twice: as the compiler has it,
    /// and with every lane named), and with the lane four away.
    pub(crate) const NEIGHBOURS: [u32; 8] = [1, 0, 3, 2, 5, 4, 7, 6];
    pub(crate) const PAIRS: [u32; 8] = [2, 2, 0, 0, 6, 6, 4, 4];
    pub(crate) const TWO_APART: [u32; 8] = [2, 3, 0, 1, 6, 7, 4, 5];
    const FOURS: [u32; 8] = [4, 4, 4, 4, 0, 0, 0, 0];
    const EXEC: u32 = 126;
    const MINUS_ONE: u32 = 193;
    const ZERO: u32 = 128;
    const DPP8: u32 = 0xE9;
    const DPP16: u32 = 0xFA;

    const fn sop1(op: u32, sdst: u32, ssrc0: u32) -> u32 {
        0xBE80_0000 | sdst << 16 | op << 8 | ssrc0
    }

    const fn vop2(op: u8, vdst: u8, src0: u32, vsrc1: u8) -> u32 {
        (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | src0
    }

    const fn vop3(op: u32, dst: u8, src: [u32; 3]) -> [u32; 2] {
        [0xD400_0000 | op << 16 | dst as u32, src[0] | src[1] << 9 | src[2] << 18]
    }

    /// The second word of a `dpp8` operation: `v<vsrc>` of the lanes `lanes` name, in eights.
    pub(crate) const fn dpp8(lanes: [u32; 8], vsrc: u8) -> u32 {
        let mut w = vsrc as u32;
        let mut i = 0;
        while i < 8 {
            w |= lanes[i] << (8 + 3 * i);
            i += 1;
        }
        w
    }

    /// `s_or_saveexec_b64 s[SAVED:SAVED+1], -1` and `s_mov_b64 exec, s[SAVED:SAVED+1]`.
    pub(crate) const SAVE_EXEC: u32 = sop1(0x25, SAVED as u32, MINUS_ONE);
    pub(crate) const RESTORE_EXEC: u32 = sop1(0x04, EXEC, SAVED as u32);

    /// `v_cndmask_b32 v<dst>, <outside>, v<src>, s[SAVED:SAVED+1]`: the terms, `outside`
    /// (an operand code) in the lanes outside the saved EXEC.
    pub(crate) const fn terms(dst: u8, outside: u32, src: u8) -> [u32; 2] {
        vop3(0x101, dst, [outside, 256 + src as u32, SAVED as u32])
    }

    /// The reduction by VOP2 `op` of `v<src>` over the wave into `s<out>`: every lane takes
    /// part, those outside EXEC with 0.
    pub(crate) const fn words(op: u8, src: u8, out: u8) -> [u32; 20] {
        let own = vop2(op, WORK, DPP8, WORK);
        let t = terms(WORK, ZERO, src);
        let other_row = vop3(0x378, TMP, [256 + WORK as u32, ZERO, ZERO]);
        let half = vop3(0x360, out, [256 + WORK as u32, ZERO + 32, 0]);
        let first = vop3(0x360, out, [256 + WORK as u32, ZERO, 0]);
        [
            SAVE_EXEC, // s_or_saveexec_b64 saved, -1
            t[0],
            t[1], // v_cndmask_b32 work, 0, src, saved
            own,
            dpp8(NEIGHBOURS, WORK), // op work, work, work dpp8:[1,0,3,2,5,4,7,6]
            own,
            dpp8(PAIRS, WORK), // op work, work, work dpp8:[2,2,0,0,6,6,4,4]
            own,
            dpp8(FOURS, WORK), // op work, work, work dpp8:[4,4,4,4,0,0,0,0]
            vop2(op, WORK, DPP16, WORK),
            0xFF << 24 | 1 << 19 | 0x140 << 8 | WORK as u32, // op work, work, work row_mirror bound_ctrl:1
            other_row[0],
            other_row[1],                         // v_permlanex16_b32 tmp, work, 0, 0
            vop2(op, WORK, 256 + TMP as u32, WORK), // op work, tmp, work
            half[0],
            half[1],                          // v_readlane_b32 out, work, 32
            vop2(op, WORK, out as u32, WORK), // op work, out, work
            first[0],
            first[1],     // v_readlane_b32 out, work, 0
            RESTORE_EXEC, // s_mov_b64 exec, saved
        ]
    }
}

/// `WaveActiveBitOr(v7)` into s7 ([`reduction::words`]): every lane takes part, those
/// outside the saved EXEC with 0.
#[cfg(test)]
pub(super) const WAVE_OR_V7_TO_S7: [u32; 20] = reduction::words(reduction::V_OR_B32, 7, 7);

#[cfg(test)]
mod tests {
    use super::*;

    const S_ENDPGM: u32 = 0xBF81_0000;

    const fn sop1(sdst: u8, op: u8, ssrc0: u8) -> u32 {
        0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | ssrc0 as u32
    }

    const fn vop1(vdst: u8, op: u8, src0: u16) -> u32 {
        (0x3F << 25) | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
    }

    const fn vop2(vdst: u8, op: u8, src0: u16, vsrc1: u8) -> u32 {
        (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
    }

    const fn dpp16(vsrc0: u8, ctrl: u16, row_mask: u8, bank_mask: u8, bound_ctrl: bool) -> u32 {
        (row_mask as u32) << 28
            | (bank_mask as u32) << 24
            | (bound_ctrl as u32) << 19
            | (ctrl as u32) << 8
            | vsrc0 as u32
    }

    const fn vop3(vdst: u8, op: u16, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
        [
            0xD400_0000 | (op as u32) << 16 | vdst as u32,
            (src0 as u32 & 0x1FF) | ((src1 as u32 & 0x1FF) << 9) | ((src2 as u32 & 0x1FF) << 18),
        ]
    }

    fn build(words: &[u32], wave64: bool) -> Program {
        let code: Vec<u8> = words
            .iter()
            .copied()
            .chain(std::iter::once(S_ENDPGM))
            .flat_map(u32::to_le_bytes)
            .collect();
        crate::ir::build(&code, wave64, 256).unwrap()
    }

    #[test]
    fn dpp_lanes_follow_the_isa() {
        // row_shr:1 with bound_ctrl: lane 16 reads 0, lane 17 lane 16.
        let shr = Dpp::Row {
            ctrl: 0x111,
            row_mask: 0xF,
            bank_mask: 0xF,
            bound_ctrl: true,
            fi: false,
        };
        assert_eq!(dpp_lane(shr, 16), Some((16, true, true)));
        assert_eq!(dpp_lane(shr, 17), Some((16, false, true)));
        // row_mirror: lane 3 reads lane 12.
        let mirror = Dpp::Row {
            ctrl: 0x140,
            row_mask: 0xF,
            bank_mask: 0xF,
            bound_ctrl: true,
            fi: false,
        };
        assert_eq!(dpp_lane(mirror, 3), Some((12, false, true)));
        // dpp8 [1,0,3,2,5,4,7,6] swaps neighbours.
        let swap = Dpp::Lanes {
            sel: 1 | 3 << 6 | 2 << 9 | 5 << 12 | 4 << 15 | 7 << 18 | 6 << 21,
            fi: false,
        };
        assert_eq!(dpp_lane(swap, 8), Some((9, false, true)));
    }

    /// What `combine` makes of +0.0 in a sum of floats (the third review of the meeting
    /// form, Question 2 item 3): it is not an operand that leaves the other as it is.
    #[test]
    fn a_positive_zero_is_added_in_a_float_sum() {
        const NEG: u32 = 0x8000_0000;
        let mut steps = Steps::default();
        let mut add = |a: Sym, b: Sym| combine(Op::AddF, a, b, &mut steps);
        // constants: +0.0 + -0.0 is +0.0 either way round, -0.0 + -0.0 is -0.0
        assert_eq!(add(Sym::K(0), Sym::K(NEG)), Sym::K(0));
        assert_eq!(add(Sym::K(NEG), Sym::K(0)), Sym::K(0));
        assert_eq!(add(Sym::K(NEG), Sym::K(NEG)), Sym::K(NEG));
        assert_eq!(add(Sym::K(0), Sym::K(0)), Sym::K(0));
        let one = 1.0f32.to_bits();
        assert_eq!((add(Sym::K(0), Sym::K(one)), add(Sym::K(one), Sym::K(NEG))), (Sym::K(one), Sym::K(one)));
        // a value from before the section: -0.0 leaves it, +0.0 does not
        let own = Sym::Own(Value(1));
        assert_eq!((add(Sym::K(NEG), own), add(own, Sym::K(0)), add(Sym::K(0), own)), (own, Sym::Unknown, Sym::Unknown));
        // a minimum or an integer sum over the lanes is not left in place under + 0.0
        let red = |op: Option<Op>, ident: u32, tree: Tree| Sym::Red {
            op,
            src: Value(1),
            mask: Value(2),
            ident,
            lanes: 0xF,
            tree,
            butterfly: Butterfly::UNKNOWN,
        };
        for (op, ident) in [(Op::MinF, 0x7F80_0000), (Op::AddU, 0), (Op::MaxU, 0)] {
            assert_eq!(add(Sym::K(0), red(Some(op), ident, Tree::NONE)), Sym::Unknown, "{op:?}");
            assert_eq!(add(red(Some(op), ident, Tree::NONE), Sym::K(0)), Sym::Unknown, "{op:?}");
        }
        // one lane's term whose other lanes hold +inf is no term of a float sum
        assert_eq!(add(Sym::K(0), red(None, 0x7F80_0000, Tree::NONE)), Sym::Unknown);
        // a float sum, or a lane's term of one, keeps the +0.0 as a step and is a sum
        let leaf = steps.of(Step::Lane(3));
        let zero = steps.of(Step::K(0));
        let want = steps.add(zero, leaf);
        for term in [red(None, 0, leaf), red(Some(Op::AddF), 0, leaf)] {
            let Sym::Red { op, tree, .. } = combine(Op::AddF, Sym::K(0), term, &mut steps) else { panic!("not followed") };
            assert_eq!((op, tree.0), (Some(Op::AddF), want.0));
        }
    }

    #[test]
    fn prefixes_are_recognised() {
        let (x, m) = (Value(1), Value(2));
        let red = |lanes: u64| Sym::Red {
            op: Some(Op::AddU),
            src: x,
            mask: m,
            ident: 0,
            lanes,
            tree: Tree::NONE,
            butterfly: Butterfly::UNKNOWN,
        };
        let inclusive: Vec<Sym> = (0..64).map(|l| red(u64::MAX >> (63 - l))).collect();
        assert_eq!(
            per_lane(&inclusive, u64::MAX),
            Some(Shape::Prefix(Op::AddU, x, m, 0, true))
        );
        let mut exclusive: Vec<Sym> = (0..64).map(|l| red((1u64 << l) - 1)).collect();
        exclusive[0] = Sym::K(0);
        assert_eq!(
            per_lane(&exclusive, u64::MAX),
            Some(Shape::Prefix(Op::AddU, x, m, 0, false))
        );
        assert_eq!(
            per_lane(&vec![red(u64::MAX); 64], u64::MAX),
            Some(Shape::Active(Op::AddU, x, m, 0))
        );
    }

    #[test]
    fn constant_decodes_width_and_inline_kinds() {
        let b64 = 0x04;
        let b32 = 0x03;
        let check = |words: &[u32], wave64: bool| -> Option<u64> {
            let p = build(words, wave64);
            constant(&p.blocks[0].insts[0])
        };

        assert_eq!(
            check(&[sop1(EXEC, b64, 194)], true),
            Some(0xFFFF_FFFF_FFFF_FFFE)
        );
        assert_eq!(
            check(&[sop1(EXEC, b64, 196)], true),
            Some(0xFFFF_FFFF_FFFF_FFFC)
        );
        assert_eq!(
            check(&[sop1(EXEC, b64, 208)], true),
            Some(0xFFFF_FFFF_FFFF_FFF0)
        );
        assert_eq!(check(&[sop1(EXEC, b64, 193)], true), Some(u64::MAX));
        assert_eq!(check(&[sop1(EXEC, b64, NULL)], true), Some(0));
        assert_eq!(check(&[sop1(EXEC, b64, 243)], true), None);
        assert_eq!(
            check(&[sop1(EXEC, b64, 255), 0x1234_5678], true),
            Some(0x1234_5678)
        );

        assert_eq!(check(&[sop1(EXEC, b32, 194)], false), Some(0xFFFF_FFFE));
        assert_eq!(check(&[sop1(EXEC, b32, 243)], false), Some(0xBF80_0000));
        assert_eq!(check(&[sop1(EXEC, b32, NULL)], false), Some(0));
        assert_eq!(
            check(&[sop1(EXEC, b32, 255), 0x1234_5678], false),
            Some(0x1234_5678)
        );
        assert_eq!(check(&[sop1(EXEC, b32, 0)], false), None);
    }

    #[test]
    fn exec_minus_two_masks_lane_zero_and_enforces_dpp_guard() {
        for wave64 in [false, true] {
            let mut w = Vec::new();
            let save_op = if wave64 { 0x25 } else { 0x3D };
            let mov_exec = if wave64 { 0x04 } else { 0x03 };
            let push_exec = |w: &mut Vec<u32>, k: u8| w.push(sop1(EXEC, mov_exec, k));
            let push_exec_lit = |w: &mut Vec<u32>, lit: u32| {
                w.push(sop1(EXEC, mov_exec, 255));
                w.push(lit);
            };

            w.push(sop1(0, save_op, 193));
            w.extend(vop3(1, 0x101, 128, 256, 0));
            for shift in [1u16, 2, 4, 8] {
                w.push(vop2(1, 0x25, 0xFA, 1));
                w.push(dpp16(1, 0x110 + shift, 0xF, 0xF, false));
            }
            w.extend(vop3(2, 0x360, 257, 128 + 15, 0));
            push_exec_lit(&mut w, 0xFFFF_0000);
            w.push(vop2(1, 0x25, 2, 1));

            if wave64 {
                w.extend(vop3(2, 0x360, 257, 128 + 31, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 255));
                w.push(0x0000_FFFF);
                w.push(vop2(1, 0x25, 2, 1));

                w.extend(vop3(2, 0x360, 257, 128 + 47, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 255));
                w.push(0xFFFF_0000);
                w.push(vop2(1, 0x25, 2, 1));
            }

            push_exec(&mut w, 193);
            w.push(vop1(3, 0x01, 257));
            w.push(vop1(3, 0x01, 0xFA));
            w.push(dpp16(1, 0x111, 0xF, 0xF, false));

            w.extend(vop3(2, 0x360, 257, 128 + 15, 0));
            push_exec_lit(&mut w, 0x0001_0000);
            w.push(vop1(3, 0x01, 2));

            if wave64 {
                w.extend(vop3(2, 0x360, 257, 128 + 31, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 129));
                w.push(vop1(3, 0x01, 2));

                w.extend(vop3(2, 0x360, 257, 128 + 47, 0));
                w.push(sop1(EXEC, 0x03, 128));
                w.push(sop1(EXEC + 1, 0x03, 255));
                w.push(0x0001_0000);
                w.push(vop1(3, 0x01, 2));
            }

            push_exec(&mut w, 193);
            w.push(vop1(2, 0x01, 128));
            w.push(vop1(5, 0x01, 135));
            push_exec(&mut w, 194);
            w.push(vop1(2, 0x01, 259));
            w.push(vop1(4, 0x01, 0xFA));
            w.push(dpp16(5, 0xE4, 0xF, 0xF, false));
            w.push(sop1(EXEC, mov_exec, 0));

            let p = build(&w, wave64);
            let insts = &p.blocks[0].insts;
            let saved = insts[0].write(Reg::S(0)).unwrap();
            let v0_in = p
                .inputs
                .iter()
                .find_map(|&(r, v)| (r == Reg::V(0)).then_some(v))
                .unwrap();
            let v2_val = insts[insts.len() - 4].write(Reg::V(2)).unwrap();
            let v4_val = insts[insts.len() - 3].write(Reg::V(4)).unwrap();
            let sec = &sections(&p, 0)[&0];
            assert_eq!(
                sec.shapes.get(&v2_val),
                Some(&Shape::Prefix(Op::AddU, v0_in, saved, 0, false))
            );
            assert_eq!(sec.shapes.get(&v4_val), None);
        }
    }

    /// `WaveActiveSum` of 16-bit floats (v11 into s22) in the steps of [`reduction::words`]:
    /// the tree the lowering reproduces. A tree in another order is not followed.
    #[test]
    fn a_sum_of_halves_over_the_wave_is_followed_in_the_order_of_its_tree() {
        let tree = reduction::words(reduction::V_ADD_F16, 11, 22);
        let shape = |w: &[u32]| {
            let p = build(w, true);
            let insts = &p.blocks[0].insts;
            let saved = insts[0].write(Reg::S(reduction::SAVED)).unwrap();
            let v11 = p.inputs.iter().find_map(|&(r, v)| (r == Reg::V(11)).then_some(v)).unwrap();
            // ..., v_readlane_b32 s22, work, 0; s_mov_b64 exec, saved; s_endpgm
            let total = insts[insts.len() - 3].write(Reg::S(22)).unwrap();
            (sections(&p, 0)[&0].shapes.get(&total).copied(), Shape::Active(Op::AddH, v11, saved, 0))
        };
        let (found, want) = shape(&tree);
        assert_eq!(found, Some(want));
        // lanes two apart first, neighbours second: every lane counted once, but
        // (x0 + x2) + (x1 + x3) rounds differently from (x0 + x1) + (x2 + x3)
        let mut other = tree;
        other[4] = reduction::dpp8(reduction::TWO_APART, reduction::WORK);
        other[6] = reduction::dpp8(reduction::NEIGHBOURS, reduction::WORK);
        let (found, _) = shape(&other);
        assert_eq!(found, None);
        // the first two steps as they are, the other way round: lanes 1 and 2 would be
        // counted twice, which is no sum either
        let mut twice = tree;
        twice.swap(4, 6);
        let (found, _) = shape(&twice);
        assert_eq!(found, None);
    }

    /// A 64-bit float sum whose low dword comes from one partial sum and whose high dword
    /// from the same lanes added in another order is still the sum over those lanes, as
    /// `world` takes it: its shape does not depend on the paired form's record
    /// ([`Butterfly`]), which knows no order for it and so leaves it out of
    /// `Section::butterflies` (the merge's review, 1: the paired record's check had made it
    /// unknown with the paired form off).
    #[test]
    fn an_f64_sum_of_halves_added_in_two_orders_keeps_its_shape() {
        const XOR1: u32 = 14_570_689; // dpp8:[1,0,3,2,5,4,7,6]
        const XOR2: u32 = 11_788_826; // dpp8:[2,3,0,1,6,7,4,5]
        const XOR4: u32 = 0o32107654; // dpp8:[4,5,6,7,0,1,2,3]
        let v = |n: u8| 256 + u16::from(n);
        let dpp8 = |w: &mut Vec<u32>, dst: u8, src: u8, sel: u32| w.extend([vop1(dst, 1, 0xE9), u32::from(src) | sel << 8]);
        // v[lo:lo+1] += the lanes `sel` picks of it (v[4:5] the copy)
        let step = |w: &mut Vec<u32>, lo: u8, sel: u32| {
            dpp8(w, 4, lo, sel);
            dpp8(w, 5, lo + 1, sel);
            w.extend(vop3(lo, 0x164, v(4), v(lo), 0)); // v_add_f64
        };
        let words = |other_order: bool| {
            let mut w = vec![sop1(12, 0x3D, 0xC1)]; // s_or_saveexec_b32 s12, -1
            w.extend(vop3(2, 0x101, 0x80, v(6), 12)); // v_cndmask_b32 v2, 0, v6, s12
            w.extend(vop3(3, 0x101, 0x80, v(7), 12)); // v_cndmask_b32 v3, 0, v7, s12
            w.extend([vop1(10, 1, v(2)), vop1(11, 1, v(3))]); // v[10:11] = v[2:3]
            step(&mut w, 2, XOR1);
            step(&mut w, 2, XOR2);
            let (first, second) = if other_order { (XOR2, XOR1) } else { (XOR1, XOR2) };
            step(&mut w, 10, first);
            step(&mut w, 10, second);
            // v[8:9]: the low dword of the one, the high of the other
            w.extend([vop1(8, 1, v(2)), vop1(9, 1, v(11))]);
            step(&mut w, 8, XOR4);
            w.extend([vop1(4, 1, 0xFA), dpp16(8, 0x140, 0xF, 0xF, true)]); // row_mirror
            w.extend([vop1(5, 1, 0xFA), dpp16(9, 0x140, 0xF, 0xF, true)]);
            w.extend(vop3(8, 0x164, v(4), v(8), 0));
            w.extend(vop3(4, 0x378, v(8), 0x80, 0x80)); // v_permlanex16_b32 v4, v8, 0, 0
            w.extend(vop3(5, 0x378, v(9), 0x80, 0x80));
            w.extend(vop3(8, 0x164, v(4), v(8), 0));
            w.extend(vop3(14, 0x360, v(8), 0x80, 0)); // v_readlane_b32 s14, v8, 0
            w.extend(vop3(15, 0x360, v(9), 0x80, 0)); // v_readlane_b32 s15, v9, 0
            w.push(sop1(126, 0x03, 12)); // s_mov_b32 exec_lo, s12
            w
        };
        for other_order in [false, true] {
            let p = build(&words(other_order), false);
            let insts = &p.blocks[0].insts;
            let saved = insts[0].write(Reg::S(12)).unwrap();
            let input = |r: u8| p.inputs.iter().find_map(|&(g, x)| (g == Reg::V(r)).then_some(x)).unwrap();
            let low = insts[insts.len() - 4].write(Reg::S(14)).unwrap();
            let sec = &sections(&p, 0)[&0];
            assert_eq!(
                sec.shapes.get(&low),
                Some(&Shape::ActiveF64(input(6), input(7), saved, false)),
                "the sum over the wave (two orders: {other_order})"
            );
            assert_eq!(sec.butterflies.contains_key(&low), !other_order, "a butterfly only where both halves are one");
        }
    }

    /// A sum of floats over the wave keeps the additions it was made by: the same lanes
    /// summed with neighbours first or with lanes two apart first are one shape and two
    /// orders. A wave OR, where the order changes nothing, keeps none.
    #[test]
    fn a_float_sum_keeps_the_order_of_its_additions() {
        let left = |words: Vec<u32>| {
            let p = build(&words, true);
            let insts = &p.blocks[0].insts;
            let saved = insts[0].write(Reg::S(reduction::SAVED)).unwrap();
            let v7 = p.inputs.iter().find_map(|&(r, v)| (r == Reg::V(7)).then_some(v)).unwrap();
            let total = insts[insts.len() - 3].write(Reg::S(7)).unwrap();
            let sec = &sections(&p, 0)[&0];
            (sec.shapes.get(&total).copied(), sec.orders.get(&total).cloned(), v7, saved)
        };
        let float = |w: u32| if w >> 25 == 28 { w & 0x01FF_FFFF | 3 << 25 } else { w }; // v_add_f32
        let (shape, order, ..) = left(WAVE_OR_V7_TO_S7.to_vec());
        assert!(matches!(shape, Some(Shape::Active(Op::Or, ..))));
        assert_eq!(order, None);
        let neighbours: Vec<u32> = WAVE_OR_V7_TO_S7.iter().map(|&w| float(w)).collect();
        let mut two_apart = neighbours.clone();
        // dpp8:[2,3,0,1,6,7,4,5], then [1,0,3,2,5,4,7,6]
        two_apart[4] = reduction::dpp8(reduction::TWO_APART, reduction::WORK);
        two_apart[6] = reduction::dpp8(reduction::NEIGHBOURS, reduction::WORK);
        for (words, first) in [(neighbours, Step::Add(0, 1)), (two_apart, Step::Add(0, 2))] {
            let (shape, order, v7, saved) = left(words);
            assert_eq!(shape, Some(Shape::Active(Op::AddF, v7, saved, 0)));
            let order = order.expect("the additions of a float sum");
            // 64 terms and the 63 additions that make one sum of them, the same in every lane
            assert_eq!((order.steps.len(), order.results.len()), (127, 1));
            assert!((0..64).all(|l| order.steps[l] == Step::Lane(l as u8)));
            assert_eq!(order.steps[64], first, "lane 0's first addition");
            assert_eq!(order.neighbours_first(), first == Step::Add(0, 1));
            assert_eq!(order.results[0], 126);
        }
    }

    #[test]
    fn buddies_are_the_two_halves_of_an_aligned_run() {
        assert!(buddies(0b01, 0b10));
        assert!(buddies(0b1100, 0b0011));
        assert!(buddies(0xF0, 0x0F));
        assert!(buddies(u64::from(u32::MAX), u64::from(u32::MAX) << 32));
        assert!(!buddies(0b010, 0b100)); // lanes 1 and 2: not one pair
        assert!(!buddies(0b0011, 0b110000)); // pairs 0 and 2
        assert!(!buddies(0b0101, 0b1010)); // not runs
        assert!(!buddies(0b01, 0b1100)); // unequal
        assert!(!buddies(0b11, 0b11));
    }

    #[test]
    fn wave32_mov_b64_all_lanes_keeps_exec_masked_for_dpp() {
        let w = [
            sop1(0, 0x3D, 193),
            vop1(1, 0x01, 135),
            sop1(EXEC, 0x04, 193),
            vop1(2, 0x01, 0xFA),
            dpp16(1, 0xE4, 0xF, 0xF, false),
            sop1(EXEC + 1, 0x03, 193),
            vop1(3, 0x01, 0xFA),
            dpp16(1, 0xE4, 0xF, 0xF, false),
            sop1(EXEC, 0x03, 0),
        ];
        let p = build(&w, false);
        let insts = &p.blocks[0].insts;
        let v2_val = insts[3].write(Reg::V(2)).unwrap();
        let v3_val = insts[5].write(Reg::V(3)).unwrap();
        let sec = &sections(&p, 0)[&0];
        assert_eq!(sec.shapes.get(&v2_val), Some(&Shape::K(7)));
        assert_eq!(sec.shapes.get(&v3_val), Some(&Shape::K(7)));
    }

    #[test]
    fn per_lane_prefix_soundness() {
        let (v0, v_other, m0, m_other) = (Value(1), Value(2), Value(3), Value(4));
        let red = |op: Option<Op>, src: Value, mask: Value, ident: u32, lanes: u64| Sym::Red {
            op,
            src,
            mask,
            ident,
            lanes,
            tree: Tree::NONE,
            butterfly: Butterfly::UNKNOWN,
        };

        for (wave, all) in [(32usize, u64::from(u32::MAX)), (64usize, u64::MAX)] {
            let below_mask = |l: usize| if l == 0 { 0 } else { (1u64 << l) - 1 };

            let mut valid_inc: Vec<Sym> = (0..wave)
                .map(|l| red(Some(Op::AddU), v0, m0, 0, below_mask(l) | (1u64 << l)))
                .collect();
            valid_inc[0] = red(None, v0, m0, 0, 1);
            assert_eq!(
                per_lane(&valid_inc, all),
                Some(Shape::Prefix(Op::AddU, v0, m0, 0, true))
            );

            let mut valid_exc: Vec<Sym> = (0..wave)
                .map(|l| red(Some(Op::AddU), v0, m0, 0, below_mask(l)))
                .collect();
            valid_exc[0] = Sym::K(0);
            valid_exc[1] = red(None, v0, m0, 0, 1);
            assert_eq!(
                per_lane(&valid_exc, all),
                Some(Shape::Prefix(Op::AddU, v0, m0, 0, false))
            );

            let mut mixed_other_src = valid_exc.clone();
            mixed_other_src[0] = red(None, v_other, m0, 0, 1);
            assert_eq!(per_lane(&mixed_other_src, all), None);

            let mut mixed_same_src = valid_exc.clone();
            mixed_same_src[0] = red(None, v0, m0, 0, 1);
            assert_eq!(per_lane(&mixed_same_src, all), None);

            let mut exc_bad_lane1_src = valid_exc.clone();
            exc_bad_lane1_src[1] = red(None, v_other, m0, 0, 1);
            assert_eq!(per_lane(&exc_bad_lane1_src, all), None);

            let mut exc_bad_lane1_set = valid_exc.clone();
            exc_bad_lane1_set[1] = red(None, v0, m0, 0, 0b11);
            assert_eq!(per_lane(&exc_bad_lane1_set, all), None);

            for bad0 in [
                red(None, v0, m0, 1, 1),
                red(None, v_other, m0, 0, 1),
                red(None, v0, m_other, 0, 1),
                red(None, v0, m0, 0, 0),
                Sym::K(0),
            ] {
                let mut bad_inc = valid_inc.clone();
                bad_inc[0] = bad0;
                assert_eq!(per_lane(&bad_inc, all), None);
            }

            let mut bad_ident: Vec<Sym> = (0..wave)
                .map(|l| red(Some(Op::AddU), v0, m0, 1, below_mask(l) | (1u64 << l)))
                .collect();
            bad_ident[0] = red(None, v0, m0, 1, 1);
            assert_eq!(per_lane(&bad_ident, all), None);
        }
    }
}
