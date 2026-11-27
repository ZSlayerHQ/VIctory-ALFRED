//! CPU models of the two card hash programs (`host::hash`): the narrow chain
//! `CARD_COMPARE` computes, one thread a piece, and the wide form `WIDE`
//! computes, 64 lanes a piece and a fold in lane order. The GPU tests in
//! `wide.rs` hold the card to the wide model on the software adapter; these
//! models pin the arithmetic and the properties the wide form takes from its
//! seeds, its per-lane chain and its fold, without a device.

const MUL: [u32; 4] = [0x0100_0193, 0x9E37_79B1, 0x85EB_CA77, 0xC2B2_AE3D];
const SEED: [u32; 4] = [0x9E37_79B9, 0x85EB_CA6B, 0xC2B2_AE35, 0x27D4_EB2F];

/// `WIDE`'s second multiply, after the narrow step's xor-shift (`Step`).
const MUL2: [u32; 4] = [0x7FEB_352D, 0x846C_A68B, 0x297A_2D39, 0x5BD1_E995];

/// One 16-byte step of `CARD_COMPARE`: `h = (h ^ v) * M; h = h.yzwx ^ (h >> 15)`.
fn step(h: [u32; 4], v: [u32; 4]) -> [u32; 4] {
    let x: [u32; 4] = std::array::from_fn(|i| (h[i] ^ v[i]).wrapping_mul(MUL[i]));
    std::array::from_fn(|i| x[(i + 1) % 4] ^ (x[i] >> 15))
}

/// One 16-byte step of `WIDE` (its `Step`), in its lanes and in its fold, operation for
/// operation: `h = (h ^ v) * M; h = h.yzwx ^ (h >> 15); h = h * M2`.
fn wide_step(h: [u32; 4], v: [u32; 4]) -> [u32; 4] {
    let x = step(h, v);
    std::array::from_fn(|i| x[i].wrapping_mul(MUL2[i]))
}

/// `FinaliseLane`: the lane number is mixed in, then three rounds each with
/// two xor-shift and odd-multiply steps a word, the words rotated into each
/// other between rounds.
fn finalise_lane(mut h: [u32; 4], lane: u32) -> [u32; 4] {
    h = std::array::from_fn(|i| {
        h[i] ^ match i {
            0 => 0x9E37_79B9 ^ lane,
            1 => 0x85EB_CA6B_u32.wrapping_add(lane.wrapping_mul(0x9E37_79B1)),
            2 => 0xC2B2_AE35 ^ lane.wrapping_mul(0x85EB_CA77),
            _ => 0x27D4_EB2F_u32.wrapping_add(lane.wrapping_mul(0xC2B2_AE3D)),
        }
    });
    for _ in 0..3 {
        h = std::array::from_fn(|i| h[i] ^ (h[i] >> 16));
        h = std::array::from_fn(|i| {
            h[i].wrapping_mul([0x7FEB_352D, 0x846C_A68B, 0x9E37_79B1, 0x85EB_CA77][i])
        });
        h = std::array::from_fn(|i| h[i] ^ (h[i] >> 15));
        h = std::array::from_fn(|i| {
            h[i].wrapping_mul([0x0100_0193, 0xC2B2_AE3D, 0x27D4_EB2F, 0x9E37_79B9][i])
        });
        h = std::array::from_fn(|i| h[i] ^ (h[i] >> 13));
        h = std::array::from_fn(|i| h[(i + 1) % 4]);
    }
    h
}

/// `Finalise`: the same shape of rounds after the fold, other odd multipliers.
fn finalise(mut h: [u32; 4]) -> [u32; 4] {
    h = std::array::from_fn(|i| h[i] ^ SEED[i]);
    for _ in 0..3 {
        h = std::array::from_fn(|i| h[i] ^ (h[i] >> 16));
        h = std::array::from_fn(|i| {
            h[i].wrapping_mul([0x85EB_CA77, 0xC2B2_AE3D, 0x27D4_EB2F, 0x9E37_79B9][i])
        });
        h = std::array::from_fn(|i| h[i] ^ (h[i] >> 13));
        h = std::array::from_fn(|i| {
            h[i].wrapping_mul([0x9E37_79B1, 0x0100_0193, 0x7FEB_352D, 0x846C_A68B][i])
        });
        h = std::array::from_fn(|i| h[i] ^ (h[i] >> 16));
        h = std::array::from_fn(|i| h[(i + 1) % 4]);
    }
    h
}

/// `Seed(len)`: the fold's seed, with the piece's covered byte length mixed
/// in four ways.
fn seed(len: usize) -> [u32; 4] {
    let len = len as u32;
    [
        0x9E37_79B9 ^ len,
        0x85EB_CA6B_u32.wrapping_add(len),
        0xC2B2_AE35 ^ len.wrapping_mul(0x0100_0193),
        0x27D4_EB2F_u32.wrapping_add(len.wrapping_mul(0x9E37_79B1)),
    ]
}

/// `CARD_COMPARE`'s hash, one `Load4` after the other.
pub(crate) fn narrow(blocks: &[[u32; 4]]) -> [u32; 4] {
    let mut h = SEED;
    for &v in blocks {
        h = step(h, v);
    }
    h
}

/// `WIDE`'s hash: lane t hashes blocks t, t+64, t+128, ..., each from the
/// seed plus t, then through the lane finaliser; the fold starts from
/// `Seed(len)` (the covered byte length) and steps the 64 final lane states
/// in lane order, then through the final finaliser.
pub(crate) fn wide(blocks: &[[u32; 4]]) -> [u32; 4] {
    let mut parts = [[0u32; 4]; 64];
    for (t, part) in parts.iter_mut().enumerate() {
        let mut h = SEED.map(|s| s.wrapping_add(t as u32));
        for &v in blocks.iter().skip(t).step_by(64) {
            h = wide_step(h, v);
        }
        *part = finalise_lane(h, t as u32);
    }
    let mut h = seed(blocks.len() * 16);
    for &part in &parts {
        h = wide_step(h, part);
    }
    finalise(h)
}

/// The bytes a piece's hash covers, as both shaders cut them: the piece
/// clamped to the copy (`skew` in piece 0), rounded out to 16 bytes.
pub(crate) fn piece_range(piece: u32, skew: u32, size: u32) -> (usize, usize) {
    let mut lo = piece.wrapping_mul(65_536);
    lo = if lo > skew { lo - skew } else { 0 };
    let mut hi = (piece + 1)
        .wrapping_mul(65_536)
        .wrapping_sub(skew)
        .min(size);
    lo &= !15;
    hi = (hi + 15) & !15;
    (lo as usize, hi as usize)
}

/// The 16-byte `Load4` values of `bytes[lo..hi]`, which must lie inside it.
pub(crate) fn blocks(bytes: &[u8], lo: usize, hi: usize) -> Vec<[u32; 4]> {
    (lo..hi)
        .step_by(16)
        .map(|at| {
            std::array::from_fn(|i| {
                u32::from_le_bytes(bytes[at + i * 4..at + i * 4 + 4].try_into().unwrap())
            })
        })
        .collect()
}

/// A piece's words: a deterministic sequence, so a failure names the same
/// bytes every run.
fn words(i: u32) -> [u32; 4] {
    [
        i.wrapping_mul(0x9E37_79B1),
        i ^ 0x85EB_CA6B,
        i.rotate_left(7),
        !i,
    ]
}

/// The narrow model is the shader's chain: 4,096 zero steps from the seed,
/// pinned to the value the shader's constants give.
#[test]
fn the_model_is_the_shaders_chain() {
    let zero = vec![[0u32; 4]; 4096];
    assert_eq!(
        narrow(&zero),
        [4_004_657_588, 3_111_047_755, 522_970_509, 391_886_411]
    );
    assert_eq!(narrow(&[]), SEED);
}

/// The wide split is another function at every size tried, including one
/// block: the lane seeds and the fold change the value even when no block
/// is reordered. That is why the switch is chosen once per process (see
/// `host::hash::wide`), not flipped while one runs.
#[test]
fn the_wide_split_is_another_hash() {
    let blocks: Vec<[u32; 4]> = (0..4096u32).map(words).collect();
    for n in [0usize, 1, 2, 63, 64, 65, 127, 128, 4095, 4096] {
        assert_ne!(narrow(&blocks[..n]), wide(&blocks[..n]), "{n} blocks");
    }
}

/// A piece's covered range follows the shaders' arithmetic: the copy's
/// offset (`skew`) moves the first piece's start, `size` clamps the last
/// end, and both are rounded out to 16 bytes.
#[test]
fn the_covered_range_is_clamped_and_rounded() {
    assert_eq!(piece_range(0, 0, 65_536), (0, 65_536));
    assert_eq!(piece_range(1, 0, 3 * 65_536), (65_536, 131_072));
    // A copy that starts 0x1234 bytes into its first piece: piece 0 covers
    // from byte 0 (the rounding reaches back before the start), piece 1
    // from 0x10000 - 0x1234 rounded down.
    assert_eq!(
        piece_range(0, 0x1234, 2 * 65_536),
        (0, (65_536 - 0x1234 + 15) & !15)
    );
    assert_eq!(
        piece_range(1, 0x1234, 2 * 65_536),
        ((65_536 - 0x1234) & !15, 126_416)
    );
    // One byte short: the end rounds up to the whole last block, as the
    // shader's `Load4` reads it.
    assert_eq!(piece_range(0, 0, 65_535), (0, 65_536));
    // A ragged tail is covered to the next 16-byte boundary.
    assert_eq!(piece_range(2, 0, 2 * 65_536 + 13), (131_072, 131_088));
}

/// The per-lane chain and the fold are order-sensitive: swapping two
/// unequal blocks changes the hash, whether they are in one lane or in two
/// (a lane-wise XOR-like combine would not see the second).
#[test]
fn position_matters_in_a_lane_and_between_lanes() {
    let mut blocks: Vec<[u32; 4]> = (0..4096u32).map(words).collect();
    let base = wide(&blocks);
    // Two blocks of lane 0: steps 0 and 64, the same lane, different places
    // in its chain.
    blocks.swap(0, 64);
    assert_ne!(wide(&blocks), base, "a swap within lane 0");
    blocks.swap(0, 64);
    // Two lanes' first blocks: lane 0's step 0 and lane 1's step 0.
    blocks.swap(0, 1);
    assert_ne!(wide(&blocks), base, "a swap between lane 0 and lane 1");
    blocks.swap(0, 1);
    // And the lane index enters each lane, in its seed and its finaliser:
    // the same blocks hashed with the lane index taken out give another
    // value.
    let blocks: Vec<[u32; 4]> = (0..65u32).map(words).collect();
    assert_ne!(
        wide(&blocks),
        wide_common_seed(&blocks),
        "the lane index taken out of the seeds and finalisers"
    );
}

/// [`wide`] with the lane index taken out of both the lane seed and the
/// lane finaliser: what the lane index is worth.
fn wide_common_seed(blocks: &[[u32; 4]]) -> [u32; 4] {
    let mut parts = [[0u32; 4]; 64];
    for (t, part) in parts.iter_mut().enumerate() {
        let mut h = SEED;
        for &v in blocks.iter().skip(t).step_by(64) {
            h = wide_step(h, v);
        }
        *part = finalise_lane(h, 0);
    }
    let mut h = seed(blocks.len() * 16);
    for &part in &parts {
        h = wide_step(h, part);
    }
    finalise(h)
}

/// The wide fold `WIDE` computes after the group barrier, from the final
/// lane states and the covered byte length.
fn fold(parts: &[[u32; 4]; 64], len: usize) -> [u32; 4] {
    let mut h = seed(len);
    for &part in parts {
        h = wide_step(h, part);
    }
    finalise(h)
}

/// A lane's last two blocks and the chain states before them: enough to
/// re-evaluate a lane after a bit flip in either, without walking the piece
/// again.
struct LaneEnds {
    before_last: [[u32; 4]; 64],
    last: [[u32; 4]; 64],
    before_second: [[u32; 4]; 64],
    second: [[u32; 4]; 64],
    /// The lane states after the lane finaliser, before the fold.
    parts: [[u32; 4]; 64],
}

/// The chain of every lane of `blocks` (at least two blocks a lane), with
/// the last two blocks' values and the states before them.
fn lane_ends(blocks: &[[u32; 4]]) -> LaneEnds {
    let mut out = LaneEnds {
        before_last: [[0; 4]; 64],
        last: [[0; 4]; 64],
        before_second: [[0; 4]; 64],
        second: [[0; 4]; 64],
        parts: [[0; 4]; 64],
    };
    for t in 0..64usize {
        let indices: Vec<usize> = (t..blocks.len()).step_by(64).collect();
        assert!(indices.len() >= 2, "every lane has two blocks");
        let (second, last) = (indices[indices.len() - 2], indices[indices.len() - 1]);
        let mut h = SEED.map(|s| s.wrapping_add(t as u32));
        for &i in &indices {
            if i == second {
                out.before_second[t] = h;
                out.second[t] = blocks[i];
            }
            if i == last {
                out.before_last[t] = h;
                out.last[t] = blocks[i];
            }
            h = wide_step(h, blocks[i]);
        }
        out.parts[t] = finalise_lane(h, t as u32);
    }
    out
}

/// `block` with bit `bit` (0..127, word 0 first) flipped.
fn flipped(block: [u32; 4], bit: usize) -> [u32; 4] {
    let mut v = block;
    v[bit / 32] ^= 1 << (bit % 32);
    v
}

/// A piece's words: zero, or a deterministic non-zero sequence.
fn contents(zero: bool) -> Vec<[u32; 4]> {
    if zero {
        vec![[0u32; 4]; 4096]
    } else {
        (0..4096u32).map(words).collect()
    }
}

/// The same single bit flipped in the LAST block of every pair of lanes, for
/// all 128 bit positions and all 2,016 lane pairs, over a zero piece and one
/// with contents: 516,096 cases, the hash always changes. The reviewer's
/// finding 1 was the old fold's first step cancelling the two equal deltas.
#[test]
fn a_bit_in_every_lane_pairs_last_block_changes_the_hash() {
    let mut cases = 0u64;
    for zero in [true, false] {
        let blocks = contents(zero);
        let base = wide(&blocks);
        let ends = lane_ends(&blocks);
        for a in 0..64usize {
            for b in a + 1..64 {
                for bit in 0..128 {
                    let mut parts = ends.parts;
                    parts[a] = finalise_lane(wide_step(ends.before_last[a], flipped(ends.last[a], bit)), a as u32);
                    parts[b] = finalise_lane(wide_step(ends.before_last[b], flipped(ends.last[b], bit)), b as u32);
                    assert_ne!(
                        fold(&parts, blocks.len() * 16),
                        base,
                        "lanes {a} and {b}, bit {bit} (zero: {zero})"
                    );
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 2 * 2_016 * 128);
    println!("last/last lane pairs: {cases} flipped-bit cases, no collision");
}

/// The same single bit flipped in the LAST block of one lane and the
/// SECOND-TO-LAST block of another (every ordered pair, 4,032), all 128 bit
/// positions, zero and contents: 1,032,192 cases, the hash always changes.
#[test]
fn a_bit_in_one_lanes_last_and_anothers_second_last_changes_the_hash() {
    let mut cases = 0u64;
    for zero in [true, false] {
        let blocks = contents(zero);
        let base = wide(&blocks);
        let ends = lane_ends(&blocks);
        for a in 0..64usize {
            for b in 0..64usize {
                if b == a {
                    continue;
                }
                for bit in 0..128 {
                    let mut parts = ends.parts;
                    parts[a] = finalise_lane(wide_step(ends.before_last[a], flipped(ends.last[a], bit)), a as u32);
                    parts[b] = finalise_lane(wide_step(ends.before_second[b], flipped(ends.second[b], bit)), b as u32);
                    assert_ne!(
                        fold(&parts, blocks.len() * 16),
                        base,
                        "lane {a}'s last and lane {b}'s second-to-last, bit {bit} (zero: {zero})"
                    );
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 2 * 4_032 * 128);
    println!("last/second-to-last lane pairs: {cases} swapped-bit cases, no collision");
}

/// The same single bit flipped in two consecutive blocks (the last two) of
/// the same lane, all 64 lanes, all 128 bit positions, zero and contents:
/// 16,384 cases, the hash always changes.
#[test]
fn a_bit_in_two_consecutive_blocks_of_one_lane_changes_the_hash() {
    let mut cases = 0u64;
    for zero in [true, false] {
        let blocks = contents(zero);
        let base = wide(&blocks);
        let ends = lane_ends(&blocks);
        for lane in 0..64usize {
            for bit in 0..128 {
                let mut parts = ends.parts;
                let second = wide_step(ends.before_second[lane], flipped(ends.second[lane], bit));
                parts[lane] = finalise_lane(wide_step(second, flipped(ends.last[lane], bit)), lane as u32);
                assert_ne!(
                    fold(&parts, blocks.len() * 16),
                    base,
                    "lane {lane}'s last two blocks, bit {bit} (zero: {zero})"
                );
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 2 * 64 * 128);
    println!("consecutive blocks of one lane: {cases} flipped-bit cases, no collision");
}

/// A deterministic pseudo-random source (xorshift), so a failure names the
/// same bytes every run.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
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

/// Small correlated edits, 250,000 of them (50,000 of each kind): one to
/// four flipped bits; the same bit in two to four neighbouring blocks;
/// the sign bit of the first word of two to four neighbouring 16-byte
/// records; two blocks exchanged; a block copied over its neighbour. Every
/// edit that changes a byte in range changes the hash.
#[test]
fn small_correlated_edits_never_keep_the_wide_hash() {
    const BLOCKS: usize = 128;
    const EACH: u64 = 50_000;
    let base_blocks: Vec<[u32; 4]> = (0..BLOCKS as u32).map(words).collect();
    let base = wide(&base_blocks);
    let mut edited = base_blocks.clone();
    let mut rng = Rng::new(0x243f_6a88_85a3_08d3);
    let mut cases = 0u64;
    for kind in 0..5u32 {
        let mut done = 0u64;
        let mut attempts = 0u64;
        while done < EACH {
            attempts += 1;
            assert!(attempts < 10_000_000, "kind {kind} cannot make a change");
            edited.copy_from_slice(&base_blocks);
            let changed = match kind {
                // One to four bits anywhere.
                0 => {
                    let n = 1 + rng.below(4);
                    for _ in 0..n {
                        let block = rng.below(BLOCKS);
                        let word = rng.below(4);
                        let bit = rng.below(32);
                        edited[block][word] ^= 1 << bit;
                    }
                    edited != base_blocks
                }
                // The same bit in two to four neighbouring blocks.
                1 => {
                    let n = 2 + rng.below(3);
                    let from = rng.below(BLOCKS - n + 1);
                    let word = rng.below(4);
                    let bit = rng.below(32);
                    for i in 0..n {
                        edited[from + i][word] ^= 1 << bit;
                    }
                    true
                }
                // The sign bit of the first word of two to four neighbouring
                // 16-byte records.
                2 => {
                    let n = 2 + rng.below(3);
                    let from = rng.below(BLOCKS - n + 1);
                    for i in 0..n {
                        edited[from + i][0] ^= 1 << 31;
                    }
                    true
                }
                // Two blocks exchanged.
                3 => {
                    let a = rng.below(BLOCKS);
                    let b = rng.below(BLOCKS);
                    if a == b || base_blocks[a] == base_blocks[b] {
                        continue;
                    }
                    edited.swap(a, b);
                    true
                }
                // A block copied over its neighbour.
                _ => {
                    let a = rng.below(BLOCKS - 1);
                    if base_blocks[a] == base_blocks[a + 1] {
                        continue;
                    }
                    edited[a] = base_blocks[a + 1];
                    true
                }
            };
            if !changed {
                continue;
            }
            assert_ne!(wide(&edited), base, "edit kind {kind} kept the hash");
            done += 1;
            cases += 1;
        }
    }
    assert_eq!(cases, 5 * EACH);
    println!("random correlated edits: {cases} cases, no collision");
}

/// An appended block, a shortened range and a moved skew each change the
/// hash: the covered byte length enters the fold's seed before any block
/// is stepped.
#[test]
fn length_and_range_enter_the_wide_hash() {
    let piece: Vec<[u32; 4]> = (0..256u32).map(words).collect();
    let base = wide(&piece);
    let mut appended = piece.clone();
    appended.push([0u32; 4]);
    assert_ne!(wide(&appended), base, "an appended zero block");
    assert_ne!(wide(&piece[..piece.len() - 1]), base, "a shortened range");

    // A moved skew: piece 1 of the same copy with the piece start 16 bytes
    // earlier or later covers the same byte count but another window.
    let bytes: Vec<u8> = (0..3 * 65_536u32).map(|i| words(i)[0] as u8).collect();
    let (lo0, hi0) = piece_range(1, 0, 2 * 65_536);
    let (lo1, hi1) = piece_range(1, 16, 2 * 65_536);
    assert_eq!(hi0 - lo0, hi1 - lo1, "the same covered length");
    assert_ne!(lo0, lo1, "the range moved");
    assert_ne!(
        wide(&blocks(&bytes, lo0, hi0)),
        wide(&blocks(&bytes, lo1, hi1)),
        "a moved skew"
    );
}

/// The reviewer's finding 1: bit 31 of word 0 flipped in blocks 4032 and
/// 4033, the last blocks of lanes 0 and 1 of a full zero piece. Odd
/// multiplication turns that bit into the same fixed delta in both lanes,
/// and the old fold's first step (`parts[0] ^ parts[1]`) cancels the two.
#[test]
fn the_reviewers_cancelling_pair_does_not_cancel() {
    let zero = vec![[0u32; 4]; 4096];
    let base = wide(&zero);
    let mut flipped = zero.clone();
    flipped[4032][0] ^= 0x8000_0000;
    flipped[4033][0] ^= 0x8000_0000;
    assert_ne!(
        wide(&flipped),
        base,
        "the zero piece's two last-block flips read the same"
    );
    assert_ne!(
        narrow(&flipped),
        narrow(&zero),
        "the narrow form sees the same flips"
    );
}

/// The same flip with non-zero contents: the fixed delta does not depend on
/// the lane's state, so the old fold lost this too.
#[test]
fn the_reviewers_cancelling_pair_does_not_cancel_with_contents() {
    let blocks: Vec<[u32; 4]> = (0..4096u32).map(words).collect();
    let base = wide(&blocks);
    let mut flipped = blocks.clone();
    flipped[4032][0] ^= 0x8000_0000;
    flipped[4033][0] ^= 0x8000_0000;
    assert_ne!(wide(&flipped), base, "a piece with contents");
    assert_ne!(narrow(&flipped), narrow(&blocks), "the narrow form");
}

/// The swap the reviewer names: two blocks that differ only in that bit,
/// exchanged between lanes 0 and 1 (the last block of each), over a zero
/// piece and over contents.
#[test]
fn the_reviewers_cancelling_swap_does_not_cancel() {
    for contents in [false, true] {
        let mut blocks: Vec<[u32; 4]> = if contents {
            (0..4096u32).map(words).collect()
        } else {
            vec![[0u32; 4]; 4096]
        };
        // Lane 0's last block and lane 1's differ only in bit 31 of word 0.
        blocks[4033][0] ^= 0x8000_0000;
        let base = wide(&blocks);
        blocks.swap(4032, 4033);
        assert_ne!(
            wide(&blocks),
            base,
            "the last blocks of lanes 0 and 1 swapped (contents: {contents})"
        );
    }
}

// Opus's review of `job/hash-wide-fix`, 2: for a top-bit change an odd multiply is linear,
// `(a ^ 2^31) * M = (a * M) ^ 2^31`, and the xor-shift after it is linear, so in the narrow step
// bit 31 of word k leaves exactly bit 31 of word k-1 and bit 16 of word k changed, whatever the
// state. Flipping those two in the lane's next block puts the state back.

/// A step function of the model (`step` or `wide_step`).
type Step = fn([u32; 4], [u32; 4]) -> [u32; 4];

/// Every lane's chain over `blocks` under `step`: `states[lane][i]` is the state before the
/// lane's `i`-th block (block `lane + 64 i`); the last entry is the state after its last block.
fn lane_states(blocks: &[[u32; 4]], step: Step) -> Vec<Vec<[u32; 4]>> {
    (0..64usize)
        .map(|lane| {
            let mut h = SEED.map(|s| s.wrapping_add(lane as u32));
            let mut states = vec![h];
            for &v in blocks.iter().skip(lane).step_by(64) {
                h = step(h, v);
                states.push(h);
            }
            states
        })
        .collect()
}

/// The lane parts (after the lane finaliser) of `states`.
fn parts_of(states: &[Vec<[u32; 4]>]) -> [[u32; 4]; 64] {
    std::array::from_fn(|lane| finalise_lane(*states[lane].last().unwrap(), lane as u32))
}

/// The wide hash of `blocks` with `edits` (block, new value), all in the lane of block `from`
/// and none before it: that lane's chain is run again from `from` on, the others are kept.
fn wide_edited(
    blocks: &[[u32; 4]],
    states: &[Vec<[u32; 4]>],
    parts: &[[u32; 4]; 64],
    from: usize,
    edits: &[(usize, [u32; 4])],
) -> [u32; 4] {
    let lane = from % 64;
    let mut h = states[lane][from / 64];
    for at in (from..blocks.len()).step_by(64) {
        let v = edits.iter().find(|e| e.0 == at).map_or(blocks[at], |e| e.1);
        h = wide_step(h, v);
    }
    let mut parts = *parts;
    parts[lane] = finalise_lane(h, lane as u32);
    fold(&parts, blocks.len() * 16)
}

/// `block` with the review's pair for word `k`: bit 31 of word k-1 and bit 16 of word k.
fn review_pair(block: [u32; 4], k: usize) -> [u32; 4] {
    let mut v = block;
    v[(k + 3) % 4] ^= 1 << 31;
    v[k] ^= 1 << 16;
    v
}

/// A piece of 4,096 pseudo-random blocks.
fn random_piece(rng: &mut Rng) -> Vec<[u32; 4]> {
    (0..4096).map(|_| std::array::from_fn(|_| rng.next() as u32)).collect()
}

/// The review's exact pattern, for every word k and every pair of blocks 64 apart (the same
/// lane's consecutive blocks in the wide form), over a zero piece, the `words` piece and a
/// random one: bit 31 of word k in block t, then bit 31 of word k-1 and bit 16 of word k in
/// block t+64. 48,384 cases, each the whole wide hash (the lane run again, then the fold); the
/// wide hash always changes. The narrow form, kept as it was, loses the same pattern between
/// consecutive blocks: a known weakness (`docs/card-hash-wide.md`), pinned here.
#[test]
fn the_reviews_top_bit_pattern_changes_the_wide_hash() {
    let mut rng = Rng::new(0x1319_8a2e_0370_7344);
    let pieces = [contents(true), contents(false), random_piece(&mut rng)];
    let mut cases = 0u64;
    for blocks in &pieces {
        let states = lane_states(blocks, wide_step);
        let parts = parts_of(&states);
        let base = fold(&parts, blocks.len() * 16);
        assert_eq!(base, wide(blocks), "the lane states are the model's");
        for k in 0..4 {
            for t in 0..blocks.len() - 64 {
                let mut first = blocks[t];
                first[k] ^= 1 << 31;
                let edits = [(t, first), (t + 64, review_pair(blocks[t + 64], k))];
                assert_ne!(
                    wide_edited(blocks, &states, &parts, t, &edits),
                    base,
                    "word {k}, blocks {t} and {}",
                    t + 64
                );
                cases += 1;
            }
            // The whole model too, at the first and the last pair of a lane.
            for t in [0usize, 4031] {
                let mut edited = blocks.clone();
                edited[t][k] ^= 1 << 31;
                edited[t + 64] = review_pair(edited[t + 64], k);
                assert_ne!(wide(&edited), base, "word {k}, blocks {t} and {} (whole model)", t + 64);
                // The narrow form between consecutive blocks: known, and kept.
                let mut narrow_edit = blocks.clone();
                narrow_edit[t][k] ^= 1 << 31;
                narrow_edit[t + 1] = review_pair(narrow_edit[t + 1], k);
                assert_eq!(narrow(&narrow_edit), narrow(blocks), "the narrow form's known weakness, word {k}, block {t}");
            }
        }
    }
    assert_eq!(cases, 3 * 4 * 4032);
    println!("the review's top-bit pattern, blocks 64 apart: {cases} cases, no collision");
}

/// Of every single-bit flip of block `x0` and every one- and two-bit flip of the lane's next
/// block `x1` (128 x 8,256 = 1,056,768 pairs), with `h0` the lane's state before `x0`: how many
/// pairs leave the state after `x1` as it was, and how many there were. Every step after `x1`
/// is a bijection, so a kept state is a kept hash and a changed one a changed hash.
fn kept_by_compensation(step: Step, h0: [u32; 4], x0: [u32; 4], x1: [u32; 4]) -> (u64, u64) {
    let base = step(step(h0, x0), x1);
    let (mut cases, mut kept) = (0u64, 0u64);
    for bit in 0..128 {
        let mid = step(h0, flipped(x0, bit));
        for a in 0..128 {
            let one = flipped(x1, a);
            kept += u64::from(step(mid, one) == base);
            cases += 1;
            for b in a + 1..128 {
                kept += u64::from(step(mid, flipped(one, b)) == base);
                cases += 1;
            }
        }
    }
    (cases, kept)
}

/// The state difference entering block `t + d` of a lane, after `bit` flipped in block `t`
/// (`states` the lanes' chains under `step`). A flip of block `t + d` puts the state back
/// exactly when it is this difference (the step takes `h ^ v`), so the difference's weight
/// says, for every one- and two-bit flip there at once, whether one cancels.
fn difference_before(step: Step, blocks: &[[u32; 4]], states: &[Vec<[u32; 4]>], t: usize, d: usize, bit: usize) -> [u32; 4] {
    let (lane, i) = (t % 64, t / 64);
    let mut h = step(states[lane][i], flipped(blocks[t], bit));
    for at in (t + 64..t + d).step_by(64) {
        h = step(h, blocks[at]);
    }
    let base = states[lane][(t + d) / 64];
    std::array::from_fn(|w| h[w] ^ base[w])
}

/// A difference's weight in bits.
fn weight(d: [u32; 4]) -> u32 {
    d.iter().map(|w| w.count_ones()).sum()
}

/// For every single-bit flip in one block, every single-bit and two-bit flip in the same lane's
/// next block: exhaustive (all 1,056,768 pairs, each stepped) over one pair of blocks at three
/// places of a lane in each of three pieces (zero, `words`, random); then, over every pair of
/// consecutive lane blocks (4,032) of five more random pieces, the same question asked of the
/// difference itself (a flip cancels exactly when it is the difference), 128 flips each. No
/// pair keeps the wide state. The narrow step, given the same blocks, keeps it for at least the
/// four top-bit flips at every pair: the sweep finds the review's pattern where it is.
#[test]
fn no_one_or_two_bit_flip_of_the_next_block_cancels_a_flip() {
    let mut rng = Rng::new(0xa409_3822_299f_31d0);
    let pieces = [contents(true), contents(false), random_piece(&mut rng)];
    let (mut stepped, mut narrow_kept) = (0u64, 0u64);
    for (p, blocks) in pieces.iter().enumerate() {
        let states = lane_states(blocks, wide_step);
        let narrow_states = lane_states(blocks, step);
        // A lane's first pair, one in the middle of another, the last pair of a third.
        for t in [0usize, 17 + 64 * 30, 4031 - 64] {
            let (lane, i) = (t % 64, t / 64);
            let (cases, kept) = kept_by_compensation(wide_step, states[lane][i], blocks[t], blocks[t + 64]);
            assert_eq!(kept, 0, "piece {p}, blocks {t} and {}: a flip was cancelled", t + 64);
            stepped += cases;
            let (_, kept) = kept_by_compensation(step, narrow_states[lane][i], blocks[t], blocks[t + 64]);
            assert!(kept >= 4, "piece {p}, block {t}: the narrow step kept {kept}, not the four top-bit flips");
            narrow_kept += kept;
        }
    }
    assert_eq!(stepped, 9 * 128 * 8256);
    let mut by_difference = 0u64;
    for _ in 0..5 {
        let blocks = random_piece(&mut rng);
        let states = lane_states(&blocks, wide_step);
        for t in 0..4032 {
            for bit in 0..128 {
                let w = weight(difference_before(wide_step, &blocks, &states, t, 64, bit));
                assert!(w > 2, "block {t}, bit {bit}: a flip of {w} bits in block {} cancels it", t + 64);
                by_difference += 8256;
            }
        }
    }
    println!(
        "one-bit flip, then every one- and two-bit flip 64 blocks on: {stepped} pairs stepped, {by_difference} more by the \
         difference, no collision (the narrow step on the same blocks: {narrow_kept} kept)"
    );
}

/// Same-lane pairs 64, 128 and 4,032 blocks apart (the lane's next block, the one after it,
/// and a lane's first and last): a single-bit flip in block t against every one- and two-bit
/// flip in block t + d, asked of the difference entering block t + d. Sampled: 40,000 (t, bit)
/// cases a distance over eight random pieces, each standing for 8,256 flips. None cancels.
#[test]
fn same_lane_pairs_64_128_and_4032_apart_do_not_cancel() {
    let mut rng = Rng::new(0x0828_2ef6_ee3a_3c2a);
    let mut counted = Vec::new();
    for d in [64usize, 128, 4032] {
        let mut cases = 0u64;
        for _ in 0..8 {
            let blocks = random_piece(&mut rng);
            let states = lane_states(&blocks, wide_step);
            for _ in 0..5000 {
                let t = rng.below(4096 - d);
                let bit = rng.below(128);
                let w = weight(difference_before(wide_step, &blocks, &states, t, d, bit));
                assert!(w > 2, "{d} apart: block {t}, bit {bit}: a flip of {w} bits in block {} cancels it", t + d);
                cases += 1;
            }
        }
        counted.push((d, cases));
    }
    println!("same-lane pairs (distance, cases of 8,256 flips each): {counted:?}, no collision");
}

/// For every one of the 128 single-bit changes of the state, the change it leaves after one
/// wide step depends on the state: of 64 states, two leave different changes. The narrow step
/// passes exactly the four top bits through whatever the state (the review's finding).
#[test]
fn no_single_bit_change_passes_the_wide_step_independent_of_the_data() {
    let mut rng = Rng::new(0x4528_21e6_38d0_1377);
    let states: Vec<[u32; 4]> = (0..64).map(|_| std::array::from_fn(|_| rng.next() as u32)).collect();
    let fixed = |step: Step| -> Vec<usize> {
        (0..128)
            .filter(|&bit| {
                let change = |h: [u32; 4]| -> [u32; 4] {
                    let (a, b) = (step(h, [0; 4]), step(flipped(h, bit), [0; 4]));
                    std::array::from_fn(|w| a[w] ^ b[w])
                };
                let first = change(states[0]);
                states.iter().all(|&h| change(h) == first)
            })
            .collect()
    };
    assert_eq!(fixed(step), vec![31, 63, 95, 127], "the narrow step's top bits");
    assert_eq!(fixed(wide_step), Vec::<usize>::new(), "the wide step");
}

/// The inverse of an odd `m` modulo 2^32 (Newton's iteration: `m` is its own inverse to three
/// bits, and each round doubles them).
fn inverse(m: u32) -> u32 {
    let mut x = m;
    for _ in 0..5 {
        x = x.wrapping_mul(2u32.wrapping_sub(m.wrapping_mul(x)));
    }
    x
}

/// [`wide_step`] undone for the block `v`: each part of the step has an inverse. The
/// rotate-and-shift `y[i] = x[i+1] ^ (x[i] >> 15)` is undone by
/// `x[i] = y[i-1] ^ (y[i-2] >> 15) ^ (y[i-3] >> 30)` (the shift part is nilpotent: a third
/// shift by 15 leaves nothing of a 32-bit word).
fn wide_step_undone(y: [u32; 4], v: [u32; 4]) -> [u32; 4] {
    let z: [u32; 4] = std::array::from_fn(|i| y[i].wrapping_mul(inverse(MUL2[i])));
    let x: [u32; 4] = std::array::from_fn(|i| z[(i + 3) % 4] ^ (z[(i + 2) % 4] >> 15) ^ (z[(i + 1) % 4] >> 30));
    std::array::from_fn(|i| x[i].wrapping_mul(inverse(MUL[i])) ^ v[i])
}

/// The wide step is a bijection of the state for a fixed block, shown by its inverse: 100,000
/// random states and blocks come back. So a change confined to one block always changes the
/// lane's state after it, and every later step keeps it changed.
#[test]
fn the_wide_step_is_a_bijection_for_a_fixed_block() {
    for m in MUL.iter().chain(&MUL2) {
        assert_eq!(m.wrapping_mul(inverse(*m)), 1, "{m:#x}");
    }
    let mut rng = Rng::new(0xbe54_66cf_34e9_0c6c);
    for _ in 0..100_000 {
        let h: [u32; 4] = std::array::from_fn(|_| rng.next() as u32);
        let v: [u32; 4] = std::array::from_fn(|_| rng.next() as u32);
        assert_eq!(wide_step_undone(wide_step(h, v), v), h);
    }
}

/// The arithmetic weight of `d` modulo 2^32: the fewest non-zero digits of a signed binary
/// form of it. `x ^ (x + d)` has at least that many bits for every `x` (the bits that differ,
/// each with its sign, are such a form of `d`).
fn arithmetic_weight(d: u32) -> u32 {
    // The non-adjacent form, of `d` and of `d - 2^32`; the lighter one.
    let naf = |mut n: i64| -> u32 {
        let mut w = 0;
        while n != 0 {
            if n & 1 != 0 {
                n -= 2 - n.rem_euclid(4);
                w += 1;
            }
            n /= 2;
        }
        w
    };
    naf(i64::from(d)).min(naf(i64::from(d) - (1i64 << 32)))
}

/// Whether some state turns an input difference `e` of one word into a single-bit output
/// difference through an odd multiply by `m`. The additive difference is a sum of `±2^b` over
/// the bits of `e`; a single bit out needs it to be `±2^b0 * m^-1`, with `b0` the lowest bit of
/// `e`. That value is such a sum exactly when `(value + e) / 2` (or it plus 2^31) has its bits
/// among `e`'s.
fn one_bit_out(e: u32, m: u32) -> bool {
    if e == 0 {
        return false;
    }
    let low = e.trailing_zeros();
    [inverse(m) << low, (inverse(m) << low).wrapping_neg()].into_iter().any(|value| {
        let t = value.wrapping_add(e);
        t & 1 == 0 && [t >> 1, (t >> 1) | 1 << 31].iter().any(|s| s & !e == 0)
    })
}

/// Proved, not sampled, for every state and block: a flip of any one of bits 12 to 31 of any
/// word leaves at least three bits of the state changed after one wide step, so no one- or
/// two-bit flip of the lane's next block can put it back; and a flip of bit 31 leaves at least
/// 9, 9, 8 and 8 bits (words 0 to 3). How: the first multiply turns the flip of bit j into a
/// difference `E = z ^ (z ± 2^j * M)`, which depends only on the bits of `z` from j up (all
/// 2^(33-j) of them enumerated); the rotate-and-shift puts `E` in word k-1 and `E >> 15` in
/// word k; each then goes through its own second multiply, and a single bit can come out of
/// both only if [`one_bit_out`] allows it for both (itself held to every sign pattern of 2,000
/// small differences first). 16,777,200 differences, none passes. Bits 0 to 11 are only
/// sampled (the sweeps above).
#[test]
fn a_flip_of_bits_12_to_31_leaves_at_least_three_bits_after_one_wide_step() {
    // `one_bit_out` against every sign pattern of small differences, and two plain cases.
    let mut rng = Rng::new(0x3f84_d5b5_b547_0917);
    for _ in 0..2000 {
        let e = (0..1 + rng.below(6)).fold(0u32, |e, _| e | 1 << rng.below(32));
        let m = rng.next() as u32 | 1;
        let bits: Vec<u32> = (0..32).filter(|b| e >> b & 1 != 0).collect();
        let by_signs = (0..1u32 << bits.len()).any(|signs| {
            let a = bits.iter().enumerate().fold(0u32, |a, (i, &b)| {
                if signs >> i & 1 != 0 { a.wrapping_add(1 << b) } else { a.wrapping_sub(1 << b) }
            });
            a.wrapping_mul(m).count_ones() == 1 || a.wrapping_mul(m).wrapping_neg().count_ones() == 1
        });
        assert_eq!(one_bit_out(e, m), by_signs, "{e:#x} times {m:#x}");
    }
    assert!(one_bit_out(1 << 16, 1) && !one_bit_out(1 << 16, MUL2[0]));
    let mut differences = 0u64;
    for k in 0..4 {
        for j in 12..32u32 {
            let d = (1u32 << j).wrapping_mul(MUL[k]);
            for t in 0..1u64 << (32 - j) {
                let z = (t as u32) << j;
                for s in [d, d.wrapping_neg()] {
                    let e = z ^ z.wrapping_add(s);
                    assert_ne!(e >> 15, 0, "word {k}, bit {j}: nothing reaches word {k} itself");
                    assert!(
                        !(one_bit_out(e, MUL2[(k + 3) % 4]) && one_bit_out(e >> 15, MUL2[k])),
                        "word {k}, bit {j}: {e:#x} may leave two bits"
                    );
                    differences += 1;
                }
            }
        }
        // Bit 31: E is bit 31 alone, which the second multiply of word k-1 passes as it is;
        // bit 16 in word k comes out with at least the arithmetic weight of 2^16 * M2.
        let top = 1 + arithmetic_weight(MUL2[k] << 16);
        assert!(top >= [9, 9, 8, 8][k], "word {k}: {top}");
    }
    assert_eq!(differences, 4 * 2 * ((1u64 << 21) - 2));
    println!("bits 12 to 31 of every word: {differences} differences, none leaves fewer than three bits");
}
