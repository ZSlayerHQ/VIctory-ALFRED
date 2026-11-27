//! Synthetic BCPack streams for tests and benchmarks (no game data): any
//! tables, every format's lane layout, lanes of many fragments. A lane's
//! symbols come in segments of one model each, drawn from that model's own
//! distribution; a segment is one rANS stream, cut into fragments (short
//! and long) where a cut keeps the state it carries, with a reset between
//! segments. The lanes' bytes are interleaved by running the decoder's own
//! schedule, which also checks that the symbols come back.

use super::*;

/// xorshift64: the same numbers on every run.
pub(super) struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

/// The lane widths and symbols per block of BC format `bc` (as `decode`).
pub(super) fn layout(bc: u8) -> (&'static [u8], usize, usize) {
    match bc {
        1 => (&[7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7], 1, 8),
        3 => (&[3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7], 2, 16),
        4 => (&[7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 1, 8),
        5 => (&[3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 2, 16),
        6 => (&[3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4], 2, 16),
        _ => (&[3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4], 2, 16),
    }
}

/// A symbol's frequency and first slot in `model`, as the encoder needs
/// them; `None` if the model cannot code it (symbol 0 of frequency 1: slot
/// 0 is never used).
fn coded(t: &Tables, model: u8, s: u8) -> Option<(u32, u32)> {
    let m = usize::from(model);
    if s == 0 {
        let f = t.freq[m][0] - 1;
        return (f > 0).then_some((f, 1));
    }
    let k = (8 - s.leading_zeros()) as usize;
    let f = *t.freq[m].get(k)?;
    (f > 0 && t.start[m][k] < 1 << 16)
        .then(|| (f, t.start[m][k] + (u32::from(s) - (1 << (k - 1))) * f))
}

/// One decode of `model` at state `x` (above 2^24): the symbol and the
/// state after it.
fn decode_one(t: &Tables, model: u8, x: u32) -> (u8, u32) {
    let m = usize::from(model);
    let slot = x & 0xffff;
    let b = t.start[m][1..BUCKETS]
        .iter()
        .filter(|&&s| s <= slot)
        .count();
    if b == 0 {
        return (
            0,
            (x >> 16)
                .wrapping_mul(t.freq[m][0] - 1)
                .wrapping_add(slot.wrapping_sub(1)),
        );
    }
    let (offset, f) = (slot - t.start[m][b], t.freq[m][b]);
    let q = offset / f;
    (
        ((1u32 << (b - 1)) + q) as u8,
        (x >> 16).wrapping_mul(f).wrapping_add(offset - q * f),
    )
}

/// rANS from 2^24: the payload bytes, in the order the decoder takes them.
fn encode(t: &Tables, model: u8, symbols: &[u8]) -> Vec<u8> {
    let mut x = LOW;
    let mut bytes = Vec::new();
    for &s in symbols.iter().rev() {
        let (freq, start) = coded(t, model, s).expect("a symbol the model codes");
        while x >= (freq << 16) {
            bytes.push(x as u8);
            x >>= 8;
        }
        x = ((x / freq) << 16) + x % freq + start;
    }
    bytes.extend_from_slice(&x.to_le_bytes());
    bytes.reverse();
    bytes
}

/// Where a payload may end a fragment and the next go on with the state
/// carried: after a byte whose read leaves the state above 2^24 (a lane
/// that runs out of bytes below it ends the fragment without that read).
fn cuts(t: &Tables, model: u8, payload: &[u8]) -> Vec<usize> {
    let (mut x, mut out) = (0u32, Vec::new());
    for (i, &b) in payload.iter().enumerate() {
        x = x << 8 | u32::from(b);
        if x > LOW {
            if i + 1 < payload.len() {
                out.push(i + 1);
            }
            while x > LOW {
                x = decode_one(t, model, x).1;
            }
        }
    }
    out
}

/// A symbol drawn from `model`'s distribution.
fn draw(t: &Tables, model: u8, rng: &mut Rng) -> u8 {
    loop {
        let slot = 1 + rng.below(0xffff) as u32;
        let s = decode_one(t, model, 0x100 << 16 | slot).0;
        if coded(t, model, s).is_some() {
            return s;
        }
    }
}

/// One lane's bytes for `count` symbols: its own (`given`) or drawn ones.
/// Returns the symbols and the bytes, a parking filler last.
fn lane(
    t: &Tables,
    width: u8,
    count: usize,
    given: Option<Vec<u8>>,
    rng: &mut Rng,
) -> (Vec<u8>, Vec<u8>) {
    let mut symbols = Vec::with_capacity(count);
    let mut bytes = Vec::new();
    while symbols.len() < count {
        let left = count - symbols.len();
        // Mostly short segments, some of thousands of symbols (long fragments).
        let len = (if rng.below(4) == 0 {
            1 + rng.below(6000)
        } else {
            1 + rng.below(300)
        } as usize)
            .min(left);
        let last = len == left;
        let (group, shape, part) = loop {
            // Given symbols need the lane's whole alphabet.
            let group = match given {
                Some(_) => width,
                None => width - rng.below(u64::from(width.min(3)) + 1) as u8,
            };
            let shape = rng.below(16) as u8;
            let model = group * 16 + shape;
            let part: Vec<u8> = match &given {
                Some(g) => g[symbols.len()..symbols.len() + len].to_vec(),
                None => (0..len).map(|_| draw(t, model, rng)).collect(),
            };
            if part.iter().all(|&s| coded(t, model, s).is_some()) {
                break (group, shape, part);
            }
        };
        let model = group * 16 + shape;
        let payload = encode(t, model, &part);
        let cuts = cuts(t, model, &payload);
        let mut from = 0;
        while from < payload.len() {
            // A short or a long fragment, ending at a cut (or the payload's end).
            const MOST: usize = 65792;
            let want = if rng.below(3) == 0 {
                257 + rng.below(4000)
            } else {
                1 + rng.below(256)
            } as usize;
            let end = if payload.len() - from <= want.min(MOST) {
                payload.len()
            } else {
                let inside = |c: &usize| *c > from && *c - from <= MOST;
                cuts.iter()
                    .copied()
                    .filter(|c| inside(c) && c - from <= want)
                    .max()
                    .or_else(|| cuts.iter().copied().find(inside))
                    .unwrap_or(payload.len())
            };
            assert!(end - from <= MOST, "a fragment holds at most {MOST} bytes");
            let n = end - from;
            let reset = end == payload.len() && !last;
            let mut header = (width - group) << 6 | shape;
            if reset {
                header |= 0x10;
            }
            if n > 256 {
                header |= 0x20;
                bytes.extend([header, (n - 257) as u8, ((n - 257) >> 8) as u8]);
            } else {
                bytes.extend([header, (n - 1) as u8]);
            }
            bytes.extend_from_slice(&payload[from..end]);
            from = end;
        }
        symbols.extend(part);
    }
    // A finished lane still asks: two payload bytes it never takes park it.
    bytes.extend([0, 1, 0, 0]);
    (symbols, bytes)
}

/// The symbols of lane 15 that make every block valid, if the format
/// needs any (a BC7 control code the tables have, a BC6H mode 1-14), but
/// block `bad`, which gets an unused code or mode 15.
fn valid_lane(
    t: &Tables,
    bc: u8,
    blocks: usize,
    bad: Option<usize>,
    rng: &mut Rng,
) -> Option<Vec<u8>> {
    match bc {
        7 => {
            let codes: Vec<usize> = (0..512).filter(|&c| t.ctrl7[c] != 0).collect();
            let unused = (0..512)
                .find(|&c| t.ctrl7[c] == 0)
                .expect("an unused BC7 code");
            Some(
                (0..blocks)
                    .flat_map(|i| {
                        let c = if Some(i) == bad {
                            unused
                        } else {
                            codes[rng.below(codes.len() as u64) as usize]
                        };
                        [(c & 31) as u8, (c >> 5) as u8 | (rng.below(2) as u8) << 4]
                    })
                    .collect(),
            )
        }
        6 => Some(
            (0..blocks)
                .flat_map(|i| {
                    [
                        if Some(i) == bad {
                            14
                        } else {
                            rng.below(14) as u8
                        },
                        rng.below(32) as u8,
                    ]
                })
                .collect(),
        ),
        _ => None,
    }
}

/// A packed stream of `blocks` blocks of BC format `bc` against `t`, and
/// its records (as `lanes` writes them).
pub(super) fn stream(t: &Tables, bc: u8, blocks: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    stream_with(t, bc, blocks, seed, None)
}

/// [`stream`], with block `bad` (BC6H, BC7) one the decoder refuses.
pub(super) fn stream_with(
    t: &Tables,
    bc: u8,
    blocks: usize,
    seed: u64,
    bad: Option<usize>,
) -> (Vec<u8>, Vec<u8>) {
    let (widths, per, _) = layout(bc);
    let mut rng = Rng(seed | 1);
    let mut special = valid_lane(t, bc, blocks, bad, &mut rng);
    let lanes_out: Vec<(Vec<u8>, Vec<u8>)> = widths
        .iter()
        .enumerate()
        .map(|(j, &w)| {
            let given = if j == 15 { special.take() } else { None };
            lane(t, w, blocks * per, given, &mut rng)
        })
        .collect();
    let stride = widths.len() * per;
    let mut records = vec![0u8; blocks * stride];
    for (j, (symbols, _)) in lanes_out.iter().enumerate() {
        for (k, &s) in symbols.iter().enumerate() {
            records[(k / per) * stride + j * per + k % per] = s;
        }
    }
    let mut queues: Vec<std::collections::VecDeque<u8>> =
        lanes_out.into_iter().map(|(_, b)| b.into()).collect();
    let mut src = Vec::new();
    let mut check = vec![0u8; records.len()];
    let fetch = |j: usize| {
        let b = queues[j]
            .pop_front()
            .expect("a parked lane asked for input");
        src.push(b);
        Some(b)
    };
    lanes(t, widths, per, blocks, fetch, &mut check).unwrap();
    assert!(
        check == records,
        "the schedule gives the lanes' symbols back"
    );
    (src, records)
}

/// Uniform tables: every model codes its alphabet evenly, identity ranks,
/// BC7 control code 0 (mode 6) only, empty dictionaries.
pub(super) fn uniform_tables() -> Tables {
    Tables::parse(&uniform_bytes()).unwrap()
}

/// Where a table file's 512 BC7 control words begin.
pub(super) const CTRL7_AT: usize = MODELS * BUCKETS * 2 + 256 + 64 + 256 + 64 + 64;

/// The file of [`uniform_tables`], to change before it is parsed.
pub(super) fn uniform_bytes() -> Vec<u8> {
    let mut t = Vec::new();
    for m in 0..MODELS {
        let g = m / 16;
        for b in 0..BUCKETS {
            let f: u16 = if b < g + 2 { 1 << (15 - g) } else { 0 };
            t.extend_from_slice(&f.to_le_bytes());
        }
    }
    t.extend((0..=255u8).map(|v| v & 0x77));
    t.extend(0..64u8);
    t.extend(0..=255u8);
    t.extend((0..64u8).map(|v| (v & 7) | (v >> 3) << 4));
    t.extend(0..64u8);
    for code in 0..512u16 {
        let ctrl: u16 = if code == 0 { 0x46 } else { 0 };
        t.extend_from_slice(&ctrl.to_le_bytes());
    }
    t.resize(TABLES_SIZE, 0);
    t
}
