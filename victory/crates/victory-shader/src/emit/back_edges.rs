//! A branch's end in the lowering, with the iteration limit on its back edges.
//!
//! Every back edge carries the cap check (`Lower::count_iteration`). A branch with one arm
//! back to a loop's header and the other out of that loop goes back only below the limit
//! and leaves the loop at it: `cond && ok` (or `cond || stop`) on the one branch, the lane at
//! the limit going down the other arm. Where no arm leads out ([`crate::bounds::cuts`]: both
//! arms back to one header, an outer header's arm whose other arm is in the inner loop, a
//! `continue`), each back arm gets a block of its own that checks its count: below the limit
//! on to the header, at it out of the program (`abort_block`), or down the other arm where
//! that one leaves the loop (and is checked there in turn, if it goes back too). The count's
//! and every join's value then come in from that block (`edges`, as a critical edge's).
//!
//! The lowering left a lane at the limit to the other arm whatever it was: with both arms back
//! the other went back unchecked (a loop the console's data run four billion times ran so on
//! the host), and with the other arm in the loop the lane went round it again. In a program
//! with a group barrier outside the meeting form it still does where there is another arm
//! ([`Lower::strict_cuts`]): a lane that left the program would be waited for there.
//!
//! The edges are the console's: a negated condition branches with its arms swapped, so where
//! both arms go to one block the join takes the swapped arm's edge (`second`) for each.

use super::{Bin, Lower, Term, V, pred};

impl Lower<'_> {
    /// The end of IR block `b`'s branch: to `taken` where `cond` holds, else to `not_taken`
    /// (`swapped`: the console's not-taken arm, the condition negated).
    pub(super) fn branch_end(&mut self, b: usize, cond: V, arms: (usize, usize), swapped: bool) -> Term {
        let (taken, not_taken) = arms;
        // Which of two edges to one block each arm is: the console's taken one first.
        let same = taken == not_taken;
        let second = [same && swapped, same && !swapped];
        let p = self.x.p;
        if crate::bounds::cut_at_limit(p, self.x.st, &self.rpo, b, self.strict_cuts()).is_some() {
            return self.counted_end(b, cond, arms, second);
        }
        // A back edge is taken only below the iteration limit.
        let back =
            |l: &Self, to: usize| l.counters.contains_key(&to) && l.rpo[to] <= l.rpo[b];
        // (No `xor` of a compare: a loop exit testing one hung the
        // AMD driver.)
        let cond = if back(self, taken) {
            let ok = self.count_iteration(taken, pred::IULT);
            self.s.m.bin(self.cur, Bin::And, cond, ok)
        } else if back(self, not_taken) {
            let stop = self.count_iteration(not_taken, pred::IUGE);
            self.s.m.bin(self.cur, Bin::Or, cond, stop)
        } else {
            cond
        };
        // Critical edges get a block of their own, so loops have
        // dedicated exits and preheaders as DXC emits them.
        let t = self.edge(b, taken, second[0]);
        let f = self.edge(b, not_taken, second[1]);
        for (to, via) in [(taken, t), (not_taken, f)] {
            if !back(self, to) {
                // From an edge block, or from here.
                let from = if Some(via) == self.block[to] {
                    self.cur
                } else {
                    via
                };
                self.enter_loop(b, to, from);
            }
        }
        Term::CondBr(cond, t, f)
    }

    /// [`Lower::branch_end`] where a lane at a back arm's limit has no way out down the
    /// other arm: each back arm through a block of its own that checks its count.
    fn counted_end(&mut self, b: usize, cond: V, (taken, not_taken): (usize, usize), second: [bool; 2]) -> Term {
        let c = self.cur;
        let back = |l: &Self, to: usize| l.counters.contains_key(&to) && l.rpo[to] <= l.rpo[b];
        let arms = [(taken, not_taken, second[0]), (not_taken, taken, second[1])];
        let strict = self.strict_cuts();
        let mut via = [0; 2];
        for (i, &(to, _, second)) in arms.iter().enumerate() {
            via[i] = if back(self, to) {
                let e = self.s.m.new_block();
                self.edges.insert((b, to, second), e);
                e
            } else {
                let e = self.edge(b, to, second);
                let from = if Some(e) == self.block[to] { c } else { e };
                self.enter_loop(b, to, from);
                e
            };
        }
        for (i, &(to, other, _)) in arms.iter().enumerate() {
            if !back(self, to) {
                continue;
            }
            // Counted here, coming in to the header from the arm's block.
            let ok = self.count_iteration(to, pred::IULT);
            let counted = self.counters.get_mut(&to).expect("a header");
            counted.1.last_mut().expect("just counted").1 = via[i];
            // The other arm is one out of this loop here, so also a back arm (a way on that
            // is not would have taken the branch's plain form).
            let stopped = if crate::bounds::cuts(self.x.p, &self.rpo, b, to, other, strict) {
                self.abort_block()
            } else {
                via[1 - i]
            };
            let header = self.block[to].expect("reachable");
            self.s.m.terminate(via[i], Term::CondBr(ok, header, stopped));
        }
        Term::CondBr(cond, via[0], via[1])
    }

    /// Whether a lane at a back arm's limit may leave the program where the other arm stays
    /// in the loop ([`crate::bounds::cuts`]): in the meeting form, which proves no barrier
    /// waits for such a lane, and in a program without a group barrier (the title's own, or
    /// a split wave's exchanges).
    pub(super) fn strict_cuts(&self) -> bool {
        let title = || self.x.p.blocks.iter().any(|b| b.insts.iter().any(|i| i.name() == "s_barrier"));
        self.meet.is_some() || !(self.exchange.is_some() || self.split_lds || self.lds_meet.is_some() || title())
    }
}
