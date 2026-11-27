//! A cheap 128-bit digest to tell whether a texture's bytes in title memory
//! changed since the last look (`VICTORY_TEXTURE_DIGEST=1`, off by default:
//! `crate::texture` keeps blake3 then).
//!
//! The construction is the CBJ fork's (its commit `2dfba4dbd`, `texture.rs`,
//! `digest`), written again here with tests the fork had not. A block is 64
//! bytes, four lanes of 16 bytes, each lane read as two little-endian words
//! `a` and `b` and carried as its own 64-bit chain:
//!
//! `lane = (fold(a ^ K[4 + i], b ^ lane) + (rotl(a, 23) ^ rotl(lane, 41))) ^ b`
//!
//! where `fold` is the xor of a 64 by 64 bit product's two halves. The tail
//! is zero-padded into one more block, stepped always (also when empty), and
//! the length enters two of the four final folds, so that inputs which
//! differ only by trailing zeros differ. The lanes meet only at the end.
//!
//! What it is for: "did these bytes change", over memory nobody picks to
//! fool us, compared within one run. On the development PC, one thread,
//! bytes in cache, a release build: 0.031 ms a megabyte against blake3's
//! 0.08 (`digest_against_blake3_ms_per_mb`).
//!
//! What it must never be used for: anything keyed, anything persisted or
//! shared between builds (the pipeline and shader caches' keys, a disk
//! cache), anything where someone chooses the inputs. A collision is easy to
//! construct (`a_constructed_collision_collides`): where one factor of the
//! product is zero the step is linear and `a` can be traded against `b`.
//! Bytes do not meet that by chance (a 64-bit word equal to a constant or to
//! the running lane state), but no patterned-change test finds it either.

/// The multipliers and seeds: lanes start at `K[0..4]`, lane `i`'s word `a`
/// is xored with `K[4 + i]`, the final folds take the rest.
const K: [u64; 8] = [
    0xa076_1d64_78bd_642f,
    0xe703_7ed1_a0b4_28db,
    0x8ebc_6af0_9c88_c6e3,
    0x5899_65cc_7537_4cc3,
    0x1d8e_4e27_c47d_124f,
    0xeb44_acca_b455_d165,
    0x9e37_79b9_7f4a_7c15,
    0xc2b2_ae3d_27d4_eb4f,
];

/// The two halves of `a * b` (128 bits) xored.
#[inline(always)]
fn fold(a: u64, b: u64) -> u64 {
    let p = u128::from(a) * u128::from(b);
    p as u64 ^ (p >> 64) as u64
}

/// The little-endian word at `block[at..at + 8]`.
#[inline(always)]
fn word(block: &[u8; 64], at: usize) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&block[at..at + 8]);
    u64::from_le_bytes(w)
}

/// One 64-byte block into the four lanes.
#[inline(always)]
fn step(lanes: &mut [u64; 4], block: &[u8; 64]) {
    for (i, lane) in lanes.iter_mut().enumerate() {
        let a = word(block, 16 * i);
        let b = word(block, 16 * i + 8);
        // What went in is added again outside the product: a factor of zero
        // forgets neither the lane nor the block (but makes the step linear,
        // the constructible collision of the module note).
        *lane = fold(a ^ K[4 + i], b ^ *lane).wrapping_add(a.rotate_left(23) ^ lane.rotate_left(41)) ^ b;
    }
}

/// The 128-bit digest of `bytes`. Equal bytes give equal digests; see the
/// module note for what else it promises and what it must not be used for.
pub(crate) fn digest(bytes: &[u8]) -> u128 {
    let mut lanes = [K[0], K[1], K[2], K[3]];
    let mut blocks = bytes.chunks_exact(64);
    for block in &mut blocks {
        let block: &[u8; 64] = block.try_into().unwrap_or(&[0; 64]);
        step(&mut lanes, block);
    }
    let rest = blocks.remainder();
    let mut last = [0u8; 64];
    last[..rest.len()].copy_from_slice(rest);
    step(&mut lanes, &last);
    let len = bytes.len() as u64;
    let low = fold(lanes[0] ^ K[1], lanes[1] ^ len) ^ fold(lanes[2] ^ K[3], lanes[3] ^ K[5]);
    let high = fold(lanes[0].rotate_left(32) ^ K[6], lanes[2] ^ len) ^ fold(lanes[1].rotate_left(32) ^ K[7], lanes[3] ^ K[2]);
    u128::from(low) << 64 | u128::from(high)
}

#[cfg(test)]
mod tests {
    use super::{K, digest};
    use std::collections::HashSet;

    /// Synthetic contents, made here: no runs of zeros.
    fn contents(len: usize) -> Vec<u8> {
        (0..len).map(|i| ((i as u32).wrapping_mul(2_654_435_761) >> 13) as u8 ^ (i >> 8) as u8).collect()
    }

    /// Every digest in `inputs` differs from `base`'s and from each other's.
    fn all_apart(base: &[u8], inputs: impl IntoIterator<Item = Vec<u8>>, what: &str) -> usize {
        let mut seen = HashSet::new();
        assert!(seen.insert(digest(base)));
        let mut n = 0;
        for input in inputs {
            assert_ne!(input, base, "{what}: the case changes nothing");
            assert!(seen.insert(digest(&input)), "{what}: case {n} collides");
            n += 1;
        }
        n
    }

    /// The fork's own test: every length of zeros differs, so do single
    /// bytes at every place and adjacent swaps.
    #[test]
    fn a_digest_tells_bytes_apart() {
        let zeros = [0u8; 400];
        let mut seen: Vec<u128> = (0..400).map(|n| digest(&zeros[..n])).collect();
        for at in 0..400 {
            for value in [1u8, 0x80, 0xff] {
                let mut bytes = zeros;
                bytes[at] = value;
                seen.push(digest(&bytes));
            }
        }
        let text: Vec<u8> = (0..400u32).map(|i| ((i * 37) >> 2) as u8).collect();
        for at in 0..399 {
            let mut bytes = text.clone();
            bytes.swap(at, at + 1);
            if bytes != text {
                seen.push(digest(&bytes));
            }
        }
        seen.push(digest(&text));
        let n = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), n);
        assert_eq!(digest(&text), digest(&text.clone()));
    }

    /// One flipped bit, at every bit of a whole 64-byte block (the first,
    /// the middle and the last of three: the last full block before the
    /// empty padding block) and of every tail of 1 to 63 bytes (after
    /// no block and after one), over zeros and over contents.
    #[test]
    fn every_single_bit_flip_changes_the_digest() {
        for base in [vec![0u8; 192], contents(192)] {
            for block in [0usize, 1, 2] {
                let flips = (0..512).map(|bit| {
                    let mut b = base.clone();
                    b[64 * block + bit / 8] ^= 1 << (bit % 8);
                    b
                });
                assert_eq!(all_apart(&base, flips, "a block's bit"), 512);
            }
        }
        for whole in [0usize, 64] {
            for tail in 1..64 {
                for base in [vec![0u8; whole + tail], contents(whole + tail)] {
                    let flips = (0..8 * tail).map(|bit| {
                        let mut b = base.clone();
                        b[whole + bit / 8] ^= 1 << (bit % 8);
                        b
                    });
                    assert_eq!(all_apart(&base, flips, "a tail's bit"), 8 * tail);
                }
            }
        }
    }

    /// The pattern our card hash missed (`host/tests/hash_model.rs`, the
    /// reviewer's cancelling pair): the top bit of a byte, of a 32-bit word
    /// and of a 64-bit word flipped in two neighbouring blocks, at the same
    /// place of the same lane and at a place of another lane, over zeros
    /// and over contents.
    #[test]
    fn sign_bit_flips_in_neighbouring_blocks_change_the_digest() {
        const BLOCKS: usize = 6;
        let mut cases = 0;
        for base in [vec![0u8; 64 * BLOCKS], contents(64 * BLOCKS)] {
            // The byte whose bit 7 is the top bit, for each width.
            for width in [1usize, 4, 8] {
                let tops: Vec<usize> = (0..64 / width).map(|w| w * width + width - 1).collect();
                for first in 0..BLOCKS - 1 {
                    let mut flips = Vec::new();
                    for &at in &tops {
                        // Same place: the same lane. Then a quarter, a half
                        // and three quarters of a block on: the others.
                        for lanes_on in 0..4 {
                            let there = (at + 16 * lanes_on) % 64;
                            let mut b = base.clone();
                            b[64 * first + at] ^= 0x80;
                            b[64 * (first + 1) + there] ^= 0x80;
                            flips.push(b);
                        }
                    }
                    cases += all_apart(&base, flips, "two neighbouring sign bits");
                }
            }
        }
        assert_eq!(cases, 2 * (BLOCKS - 1) * 4 * (64 + 16 + 8));
    }

    /// Exchanged 8-byte words: the two of one lane, words of two lanes of
    /// one block, and whole blocks.
    #[test]
    fn swapped_words_and_blocks_change_the_digest() {
        const BLOCKS: usize = 6;
        let base = contents(64 * BLOCKS + 24);
        let swapped = |x: usize, y: usize, n: usize| {
            let mut b = base.clone();
            let (lo, hi) = b.split_at_mut(y);
            lo[x..x + n].swap_with_slice(&mut hi[..n]);
            b
        };
        let mut within = Vec::new();
        let mut across = Vec::new();
        for block in 0..BLOCKS {
            for lane in 0..4 {
                let at = 64 * block + 16 * lane;
                within.push(swapped(at, at + 8, 8));
                for other in lane + 1..4 {
                    for (x, y) in [(0, 0), (0, 8), (8, 0), (8, 8)] {
                        across.push(swapped(at + x, 64 * block + 16 * other + y, 8));
                    }
                }
            }
        }
        assert_eq!(all_apart(&base, within, "a lane's two words"), 4 * BLOCKS);
        assert_eq!(all_apart(&base, across, "words of two lanes"), 6 * 4 * BLOCKS);
        let mut blocks = Vec::new();
        for x in 0..BLOCKS {
            for y in x + 1..BLOCKS {
                blocks.push(swapped(64 * x, 64 * y, 64));
            }
        }
        assert_eq!(all_apart(&base, blocks, "two blocks"), BLOCKS * (BLOCKS - 1) / 2);
    }

    /// A deterministic pseudo-random source (xorshift), so a failure names
    /// the same bytes every run.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// Small correlated edits, 250,000 of them (50,000 of each kind), as
    /// `small_correlated_edits_never_keep_the_wide_hash` makes them for the
    /// card hash: one to four flipped bits; the same bit in two to four
    /// neighbouring blocks; the sign bit of the first word of two to four
    /// neighbouring 16-byte lanes; two blocks exchanged; a block copied over
    /// its neighbour. None keeps the digest. At a whole number of blocks,
    /// and again with a tail of 37 bytes after them (the bit flips reach
    /// into it).
    #[test]
    fn small_correlated_edits_never_keep_the_digest() {
        const BLOCKS: usize = 32;
        for len in [64 * BLOCKS, 64 * BLOCKS + 37] {
            correlated_edits_at(len);
        }
    }

    /// [`small_correlated_edits_never_keep_the_digest`] over `len` bytes of
    /// contents, the block edits within its first 32 whole blocks.
    fn correlated_edits_at(len: usize) {
        const BLOCKS: usize = 32;
        const EACH: u64 = 50_000;
        let base = contents(len);
        let kept = digest(&base);
        let mut edited = base.clone();
        let mut rng = Rng(0x243f_6a88_85a3_08d3);
        let mut cases = 0u64;
        for kind in 0..5u32 {
            let mut done = 0u64;
            let mut attempts = 0u64;
            while done < EACH {
                attempts += 1;
                assert!(attempts < 10_000_000, "kind {kind} cannot make a change");
                edited.copy_from_slice(&base);
                match kind {
                    0 => {
                        for _ in 0..1 + rng.below(4) {
                            let bit = rng.below(8 * base.len());
                            edited[bit / 8] ^= 1 << (bit % 8);
                        }
                    }
                    1 => {
                        let n = 2 + rng.below(3);
                        let from = rng.below(BLOCKS - n + 1);
                        let bit = rng.below(512);
                        for i in 0..n {
                            edited[64 * (from + i) + bit / 8] ^= 1 << (bit % 8);
                        }
                    }
                    2 => {
                        let n = 2 + rng.below(3);
                        let from = rng.below(4 * BLOCKS - n + 1);
                        for i in 0..n {
                            edited[16 * (from + i) + 7] ^= 0x80;
                        }
                    }
                    3 => {
                        let (x, y) = (rng.below(BLOCKS), rng.below(BLOCKS));
                        if x == y {
                            continue;
                        }
                        let (x, y) = (x.min(y), x.max(y));
                        let (lo, hi) = edited.split_at_mut(64 * y);
                        lo[64 * x..64 * x + 64].swap_with_slice(&mut hi[..64]);
                    }
                    _ => {
                        let x = rng.below(BLOCKS - 1);
                        edited.copy_within(64 * (x + 1)..64 * (x + 2), 64 * x);
                    }
                }
                if edited == base {
                    continue;
                }
                assert_ne!(digest(&edited), kept, "length {len}, edit kind {kind}, case {done}, kept the digest");
                done += 1;
                cases += 1;
            }
        }
        assert_eq!(cases, 5 * EACH);
    }

    /// The module note's constructed collision, kept so that nobody takes
    /// the digest for more than it is: with lane 0's first-block state
    /// `K[0]`, `a = K[4]` makes the product's first factor zero and
    /// `a = 0, b = K[0]` its second, and both steps leave
    /// `rotl(K[0], 41) ^ K[0]`. Two different 64-byte inputs, one digest.
    #[test]
    fn a_constructed_collision_collides() {
        let input = |a: u64, b: u64| {
            let mut bytes = vec![0u8; 64];
            bytes[..8].copy_from_slice(&a.to_le_bytes());
            bytes[8..16].copy_from_slice(&b.to_le_bytes());
            bytes
        };
        let x = input(K[4], K[4].rotate_left(23) ^ K[0]);
        let y = input(0, K[0]);
        assert_ne!(x, y);
        assert_eq!(digest(&x), digest(&y), "the constructed pair collides (if not, the module note is out of date)");
        // The same with anything after it, the rest being equal.
        let tail = contents(1000);
        let (mut x, mut y) = (x, y);
        x.extend_from_slice(&tail);
        y.extend_from_slice(&tail);
        assert_eq!(digest(&x), digest(&y));
    }

    /// Milliseconds a megabyte, one thread, for the digest and for blake3
    /// at 1, 4 and 16 MB. A measurement for the report, no check: run with
    /// `-- --ignored --nocapture` (and `--release` for numbers that mean
    /// something: this crate's own code is not optimised in a test build,
    /// blake3 is).
    #[test]
    #[ignore = "a measurement, not a check"]
    fn digest_against_blake3_ms_per_mb() {
        let mut line = String::from("ms per MB, digest / blake3:");
        for mb in [1usize, 4, 16] {
            let bytes = contents(mb << 20);
            let runs = (64 / mb).max(4) as u32;
            let time = |f: &dyn Fn()| {
                f();
                let started = std::time::Instant::now();
                (0..runs).for_each(|_| f());
                started.elapsed().as_secs_f64() * 1e3 / f64::from(runs) / mb as f64
            };
            let d = time(&|| {
                std::hint::black_box(digest(std::hint::black_box(&bytes)));
            });
            let b = time(&|| {
                std::hint::black_box(blake3::hash(std::hint::black_box(&bytes)));
            });
            line += &format!(" {mb} MB {d:.3} / {b:.3};");
        }
        println!("{line}");
    }
}
