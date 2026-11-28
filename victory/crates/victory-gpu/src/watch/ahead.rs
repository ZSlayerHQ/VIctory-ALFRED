//! The page ranges a submission's asking asked the kernel about ahead of it
//! ([`super::ask_ahead`], [`super::ask_once`]), and the question `writes`,
//! `asked` and `nothing_new` ask of them: "does one range contain
//! `[first, last)`?".

/// Page ranges `(first, last)`, `last` exclusive, sorted by `first`, with the
/// running maximum of `last` over that order.
///
/// One range `(a, b)` contains `[first, last)` exactly when
/// `a <= first && last <= b`, the walk's answer. Sorted by `a`, that holds
/// exactly when the largest `b` among the ranges with `a <= first` is at
/// least `last`: a binary search finds how many ranges have `a <= first`,
/// and the running maximum gives their largest `b`. Overlap, nesting,
/// duplicates, empty ranges and the order of pushes do not matter, and
/// ranges are never merged: the question asks for one range containing the
/// caller's, not for a union.
#[derive(Default)]
pub(super) struct Ahead {
    /// The ranges, sorted by `first`.
    sorted: Vec<(usize, usize)>,
    /// `max_last[i]` is the largest `last` over `sorted[..=i]`.
    max_last: Vec<usize>,
}

impl Ahead {
    pub(super) const fn new() -> Self {
        Self {
            sorted: Vec::new(),
            max_last: Vec::new(),
        }
    }

    pub(super) fn clear(&mut self) {
        self.sorted.clear();
        self.max_last.clear();
    }

    /// Built on each push, not on the first lookup after one: lookups read
    /// the journal under its read lock, where a lazy build would need
    /// interior mutability, while a push happens once per asking and a
    /// lookup thousands of times a frame. A push inserts into the sorted
    /// copy and recomputes the running maximum from there.
    pub(super) fn push(&mut self, range: (usize, usize)) {
        let at = self.sorted.partition_point(|&(first, _)| first <= range.0);
        self.sorted.insert(at, range);
        // 0 is the identity of `max` over `usize`, and every `last` is at least 0.
        let mut max = at.checked_sub(1).map_or(0, |i| self.max_last[i]);
        self.max_last.insert(at, max);
        for i in at..self.sorted.len() {
            max = max.max(self.sorted[i].1);
            self.max_last[i] = max;
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.sorted.is_empty()
    }

    /// Whether one pushed range contains `[first, last)`: the walk's answer
    /// for any input.
    pub(super) fn covers(&self, first: usize, last: usize) -> bool {
        // Ranges with `a <= first` are the sorted copy's prefix.
        let ends = self.sorted.partition_point(|&(a, _)| a <= first);
        ends > 0 && self.max_last[ends - 1] >= last
    }
}

#[cfg(test)]
mod tests {
    use super::Ahead;

    /// The walk the journal answered with before this file.
    fn walk(ranges: &[(usize, usize)], first: usize, last: usize) -> bool {
        ranges.iter().any(|&(a, b)| a <= first && last <= b)
    }

    fn of(ranges: &[(usize, usize)]) -> Ahead {
        let mut ahead = Ahead::new();
        for &range in ranges {
            ahead.push(range);
        }
        ahead
    }

    /// xorshift64*, seeded here: no crate for a generator.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    #[test]
    fn the_search_answers_as_the_walk() {
        // Empty, new or taken (`ask_rest` takes the journal's list).
        let mut taken = of(&[(1, 4)]);
        let taken = std::mem::take(&mut taken);
        for empty in [Ahead::new(), Ahead::default()] {
            assert!(empty.is_empty());
            for (first, last) in [(0, 0), (0, 1), (7, 7), (0, usize::MAX), (usize::MAX, usize::MAX)] {
                assert!(!empty.covers(first, last), "the empty set contains [{first}, {last})");
            }
        }
        assert!(taken.covers(1, 4) && !taken.is_empty(), "what was taken keeps its ranges");

        // Ranges at both ends of the page space, nested and overlapping ones
        // among them, against every query over a set of interesting pages.
        let max = usize::MAX;
        let sets: [&[(usize, usize)]; 6] = [
            &[(0, 0)],
            &[(max, max)],
            &[(0, max)],
            &[(0, 3), (max - 3, max)],
            &[(2, 20), (4, 8), (5, 6), (0, 1), (max - 1, max)],
            &[(10, 30), (5, 15), (25, 40), (12, 13), (0, 2), (1, 3), (max - 5, max - 2), (max - 3, max)],
        ];
        let points = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 13, 15, 20, 25, 30, 40, 41, max - 5, max - 3, max - 2, max - 1, max];
        for set in sets {
            let ahead = of(set);
            for first in points {
                for last in points {
                    assert_eq!(ahead.covers(first, last), walk(set, first, last), "{set:?}, query [{first}, {last})");
                }
            }
        }

        // Many sets, random and seeded: 0 to 11 ranges over a small page
        // space (so they overlap, nest, touch and duplicate), pushed in a
        // shuffled order, then 64 queries each, some empty and some reversed.
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for case in 0..4_000 {
            let count = rng.below(12);
            let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(count);
            for _ in 0..count {
                let a = rng.below(33);
                let range = match rng.below(4) {
                    0 => (a, a),
                    1 => (a, a + rng.below(5)),
                    2 => (a, a + rng.below(33)),
                    _ => (a, rng.below(33)),
                };
                // Sometimes a range already pushed, for duplicates.
                ranges.push(if !ranges.is_empty() && rng.below(4) == 0 { ranges[rng.below(ranges.len())] } else { range });
            }
            let mut shuffled = ranges.clone();
            for i in (1..shuffled.len()).rev() {
                shuffled.swap(i, rng.below(i + 1));
            }
            let mut ahead = of(&shuffled);
            assert_eq!(ahead.is_empty(), ranges.is_empty(), "case {case}: {ranges:?}");
            for _ in 0..64 {
                let first = rng.below(40);
                let last = match rng.below(4) {
                    0 => first,
                    1 => first + rng.below(10),
                    _ => rng.below(40),
                };
                assert_eq!(
                    ahead.covers(first, last),
                    walk(&ranges, first, last),
                    "case {case}: ranges {ranges:?}, query [{first}, {last})"
                );
            }
            // A clear empties it, and a push after it answers as a fresh list.
            if rng.below(8) == 0 {
                ahead.clear();
                assert!(ahead.is_empty(), "case {case}: cleared");
                let range = (rng.below(40), rng.below(40));
                ahead.push(range);
                for (first, last) in [range, (range.0, range.0), (0, 40), (range.0.saturating_sub(1), range.1)] {
                    assert_eq!(
                        ahead.covers(first, last),
                        walk(&[range], first, last),
                        "case {case}: {range:?} alone after a clear, query [{first}, {last})"
                    );
                }
            }
        }
    }
}
