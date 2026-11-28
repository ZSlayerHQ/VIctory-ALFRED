//! Lane masks the ballot cannot give as numbers, given by the scalar instruction that writes
//! them ([`with_mask_numbers`]).
//!
//! A mask read as a number is the ballot of its lanes' bits where it is written
//! ([`Lower::mask_dword`]), which misses the bits of lanes that do not run there: a mask with
//! lanes outside EXEC is refused ([`super::MASK_ELSEWHERE`]). Compiled code makes such masks
//! when it reuses a register: the shadow-ray tracer's packed key is
//! `s_or_b32 key, high, s_and_b32(low, 0x1fffff)` with `low` a join of a number on one path and
//! a saved EXEC on another, compared lane by lane with a VGPR. The structure pass tracks the key
//! as a mask (it derives from EXEC on one path), and its bits from the number are no lane's.
//!
//! What the console's register holds is the instruction's operation on its operands' values.
//! Where one host wave is the console wave (console lanes, no exchange, no halves, no meetings:
//! a wave32 program, or wave64 on a host of 64), each operand's number is exact: a number, a
//! join's number, or a mask the ballot gives (one whose lanes all run where it is written), or
//! again one of these masks. So the mask is that operation, made in the block that writes it.
//! No barrier, no LDS: nothing is exchanged.
//!
//! A join's number is the wave's only where every lane reaches the join the same way: the
//! console takes each branch with the whole wave, the lowering a branch on a lane mask per
//! lane. A join decided by such a branch keeps the refusal
//! ([`Lower::joins_taken_together`]).

use std::collections::HashSet;

use super::{Atom, Bin, Fields, LaneExit, Lower, MaskData, Operand, Reg, V, Val, Why};
use crate::gfx10::NULL;
use crate::ir::{EXEC, Value};

thread_local! {
    /// Whether masks are given by their writing instruction ([`with_mask_numbers`]).
    static MASK_NUMBERS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `translate` with lane masks the ballot cannot give made from their writing
/// instruction's operands: the translation of a program refused with
/// [`super::MASK_ELSEWHERE`], and only such a program, so every program that translated before
/// is emitted as before.
pub(in crate::emit) fn with_mask_numbers<R>(translate: impl FnOnce() -> R) -> R {
    super::super::with_flag(&MASK_NUMBERS, true, translate)
}

impl Lower<'_> {
    /// Mask `lo` (a low dword's value) as numbers (low, high dword), made by the 32- or 64-bit
    /// scalar bitwise instruction or move that writes it from its operands' numbers; None
    /// where this form does not apply (off, the host waves are not the console's, another
    /// writer).
    pub(super) fn mask_number(&mut self, lo: Value) -> Result<Option<(V, V)>, Why> {
        let own_waves = self.console_lanes
            && self.mask_data == MaskData::Console
            && self.exchange.is_none()
            && self.halves.is_none()
            && self.pairs.is_none()
            && !self.local;
        if !MASK_NUMBERS.get() || !own_waves {
            return Ok(None);
        }
        let Some(&b) = self.def_block.get(&lo) else {
            return Ok(None);
        };
        let p = self.x.p;
        let Some(inst) = p.blocks[b].insts.iter().find(|i| i.writes.iter().any(|w| w.1 == lo)) else {
            return Ok(None);
        };
        let Fields::Sop { dst: Some(Operand::Scalar { code: d, .. }), src, .. } = &inst.fields else {
            return Ok(None);
        };
        let name = inst.name();
        let (op, wide) = match name.strip_suffix("_b64") {
            Some(op) => (op, true),
            None => (name.strip_suffix("_b32").unwrap_or(""), false),
        };
        let known = ["s_mov", "s_not", "s_and", "s_or", "s_xor", "s_andn2", "s_orn2", "s_nand", "s_nor", "s_xnor"];
        if !known.contains(&op) || *d == EXEC || inst.write(Reg::S(*d)) != Some(lo) || wide != self.x.native.wave64 {
            return Ok(None);
        }
        let at = if b == self.cur_ir { self.cur } else { self.tail[b].ok_or("a lane mask written after its use")? };
        if !self.balloting.insert(lo) {
            return Err("a lane mask whose lanes depend on its own value");
        }
        let r = self.mask_number_at(inst, src, op, wide, at);
        self.balloting.remove(&lo);
        r.map(Some)
    }

    fn mask_number_at(&mut self, inst: &crate::ir::Inst, src: &[Operand], op: &str, wide: bool, at: usize) -> Result<(V, V), Why> {
        // dword `h` of source `k`: a register's number (made where it is written: a mask's
        // ballot in its own block, a join's phi, user data in the entry), an inline constant
        // (sign-extended) or a literal (the low dword). Nothing that would be made in the block
        // being lowered, which `at` need not be.
        let dword = |l: &mut Self, k: usize, h: u8| -> Result<V, Why> {
            match *src.get(k).ok_or("a missing operand")? {
                Operand::Scalar { code, .. } if code != NULL => {
                    let v = inst.read(Reg::S(code + h)).ok_or("an unread operand")?;
                    l.joins_taken_together(v)?;
                    match l.get(v)? {
                        Val::I32(x) => Ok(x),
                        Val::Missing(w) => Err(w),
                        Val::Bit(_) => Err("a lane mask made of a condition"),
                    }
                }
                Operand::Scalar { .. } => Ok(l.s.m.i32(0)),
                Operand::Const(c) => {
                    let x = crate::resource::inline(c).ok_or("a lane mask made of a special operand")?;
                    Ok(l.s.m.i32(if h == 0 { x } else if (x as i32) < 0 { u32::MAX } else { 0 }))
                }
                Operand::Literal(x) => Ok(l.s.m.i32(if h == 0 { x } else { 0 })),
                _ => Err("a lane mask made of a special operand"),
            }
        };
        let ones = self.s.m.i32(u32::MAX);
        let mut r = [self.s.m.i32(0); 2];
        for h in 0..1 + u8::from(wide) {
            let a = dword(self, 0, h)?;
            r[usize::from(h)] = match op {
                "s_mov" => a,
                "s_not" => self.s.m.bin(at, Bin::Xor, a, ones),
                _ => {
                    let mut c = dword(self, 1, h)?;
                    if op.ends_with("n2") {
                        c = self.s.m.bin(at, Bin::Xor, c, ones);
                    }
                    let bin = match op {
                        "s_and" | "s_andn2" | "s_nand" => Bin::And,
                        "s_or" | "s_orn2" | "s_nor" => Bin::Or,
                        _ => Bin::Xor,
                    };
                    let x = self.s.m.bin(at, bin, a, c);
                    if matches!(op, "s_nand" | "s_nor" | "s_xnor") { self.s.m.bin(at, Bin::Xor, x, ones) } else { x }
                }
            };
        }
        Ok((r[0], r[1]))
    }

    /// Whether the joins behind `v` (it, if it is a join, and the joins its arguments are) are
    /// each reached the same way by every lane, marking each read. A scalar register is the wave's, and on the
    /// console the wave takes every branch together, while the lowering takes a branch on a
    /// lane mask per lane (the lanes whose bit is clear go the other way and bring the other
    /// arm's value). So every branch deciding which way the lanes reach the join must be on a
    /// scalar condition or taken by the whole wave ([`Structured::wide`]).
    ///
    /// [`Structured::wide`]: crate::structure::Structured::wide
    fn joins_taken_together(&mut self, v: Value) -> Result<(), Why> {
        let p = self.x.p;
        let mut work = vec![v];
        let mut seen = HashSet::new();
        while let Some(w) = work.pop() {
            let Some(&(b, i)) = self.phi_at.get(&w) else { continue };
            if !seen.insert(w) {
                continue;
            }
            if !self.decided_together(b) {
                return Err("a lane mask joined after a branch the wave does not take together");
            }
            // read, so that its arguments come as their numbers or the join is refused: a join
            // only another join reads is otherwise filled with 0 where an argument has no
            // number (`fill_phis` without `VICTORY_STRICT_JOINS`)
            self.read_phis.insert(w);
            work.extend(p.blocks[b].phis[i].args.iter().copied());
        }
        Ok(())
    }

    /// Whether every branch deciding which way the lanes reach block `b` is on a scalar
    /// condition or taken by the whole wave: the branches of the blocks between `b`'s immediate
    /// dominator (it included) and `b` with both ways leading on to `b`.
    fn decided_together(&self, b: usize) -> bool {
        let (p, st) = (self.x.p, self.x.st);
        let order = crate::ir::reverse_postorder(&p.blocks);
        let mut index = vec![usize::MAX; p.blocks.len()];
        for (n, &x) in order.iter().enumerate() {
            index[x] = n;
        }
        let top = crate::ir::idoms(&p.blocks, &order, &index)[b];
        if top == usize::MAX {
            return false;
        }
        // the blocks from which `b` is reached without passing `top` (or `b`), and `top`
        let mut region: HashSet<usize> = HashSet::from([top]);
        let mut work = p.blocks[b].preds.clone();
        while let Some(x) = work.pop() {
            if x != b && index[x] != usize::MAX && region.insert(x) {
                work.extend(p.blocks[x].preds.iter().copied());
            }
        }
        let on = |x: usize| x == b || region.contains(&x);
        region.iter().all(|&x| match st.exits[x] {
            LaneExit::Branch { when, taken, not_taken } if on(taken) && on(not_taken) => {
                st.wide.contains(&x) || st.bools.support(when).iter().all(|a| matches!(a, Atom::Uniform(_)))
            }
            _ => true,
        })
    }
}

/// A translation that panics in the mask-number retry leaves the retry off for the next
/// program on its thread.
#[cfg(test)]
#[test]
fn a_panic_in_the_retry_leaves_mask_numbers_off() {
    super::super::caught(|| with_mask_numbers(|| panic!("a translation's panic")));
    assert!(!MASK_NUMBERS.get());
}
