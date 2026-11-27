//! The lanes of [`super::lanes`], every lane of a round at once: a lane's
//! step within a round depends on its own state only, and the schedule
//! between them (which lanes take the round's 8 bytes) is a prefix over the
//! lanes that ask. That is how the console's unit runs them; here the 12 or
//! 16 lanes are the 16 32-bit elements of an AVX-512 register. A lane's
//! header and length bytes (a few steps a fragment) are taken one lane at a
//! time.
//!
//! The same rounds as [`super::lanes`], step for step: the same records,
//! the same bytes taken, the same error at the same round.

use std::arch::x86_64::*;

use super::{BUCKETS, Error, LOW, ROUND_BYTES, Tables};

/// Whether this processor has what [`lanes`] needs.
pub(super) fn available() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512vbmi2")
            && is_x86_feature_detected!("bmi2")
            && is_x86_feature_detected!("popcnt")
    })
}

/// Per lane, its model's buckets as columns (element `j` is lane `j`'s):
/// bucket k + 1's first slot (65536 for a bucket the model lacks), its
/// symbols' frequency, and its reciprocal in two 16-bit halves; frequency
/// 0 is symbol 0's, less one.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Columns {
    start: [[u32; 16]; BUCKETS - 1],
    freq: [[u32; 16]; BUCKETS],
    high: [[u32; 16]; BUCKETS - 1],
    low: [[u32; 16]; BUCKETS - 1],
}

impl Columns {
    fn set(&mut self, j: usize, t: &Tables, model: u8) {
        let m = usize::from(model);
        self.freq[0][j] = t.freq[m][0].wrapping_sub(1);
        for k in 1..BUCKETS {
            self.start[k - 1][j] = t.start[m][k];
            self.freq[k][j] = t.freq[m][k];
            // At most 2^32 (frequency 1): the high half fits.
            self.high[k - 1][j] = (t.recip[m][k] >> 16) as u32;
            self.low[k - 1][j] = (t.recip[m][k] & 0xffff) as u32;
        }
    }
}

/// [`super::lanes`] with input `src`, if this processor has AVX-512.
pub(super) fn lanes(
    t: &Tables,
    widths: &[u8],
    per: usize,
    blocks: usize,
    src: &[u8],
    records: &mut [u8],
) -> Option<Result<(), Error>> {
    if !available()
        || widths.len() > 16
        || !matches!(per, 1 | 2)
        || records.len() != blocks * widths.len() * per
    {
        return None;
    }
    // SAFETY: the processor has the features (`available`).
    Some(unsafe { run(t, widths, per, blocks, src, records) })
}

/// Lanes in a mask, lowest first.
fn each(mut mask: u16) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        (mask != 0).then(|| {
            let j = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            j
        })
    })
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2,bmi2,popcnt")]
fn run(
    t: &Tables,
    widths: &[u8],
    per: usize,
    blocks: usize,
    src: &[u8],
    records: &mut [u8],
) -> Result<(), Error> {
    let n = widths.len();
    let valid = ((1u32 << n) - 1) as u16;
    let stride = n * per;
    // Header state, per lane.
    let mut phase = [0u8; 16];
    let mut width = [0u8; 16];
    width[..n].copy_from_slice(widths);
    let (mut long, mut reset, mut payload) = (0u16, 0u16, 0u16);
    let mut columns = Columns {
        start: [[0; 16]; BUCKETS - 1],
        freq: [[0; 16]; BUCKETS],
        high: [[0; 16]; BUCKETS - 1],
        low: [[0; 16]; BUCKETS - 1],
    };
    for j in 0..16 {
        columns.set(j, t, 0);
    }
    // The rest, as vectors and masks (a bit a lane).
    let zero = _mm512_setzero_si512();
    let one = _mm512_set1_epi32(1);
    let low = _mm512_set1_epi32(LOW as i32);
    let slots = _mm512_set1_epi32(0xffff);
    let (mut x, mut left, mut latch, mut out) = (zero, zero, zero, zero);
    let (mut latched, mut held, mut need, mut stalled, mut done) = (0u16, 0u16, valid, 0u16, 0u16);
    let (mut taken_all, mut pos, mut at, mut idle) = (0usize, 0usize, 0usize, 0u32);
    let load = |row: &[u32; 16]| {
        // SAFETY: 16 elements.
        unsafe { _mm512_loadu_epi32(row.as_ptr().cast()) }
    };
    while pos < blocks * per {
        let active = !stalled & valid;
        // Lanes in a fragment's payload; the others take header bytes.
        let stepping = active & payload;
        let headers = active & !payload & latched;
        if headers != 0 {
            let mut bytes = [0u32; 16];
            let mut lefts = [0u32; 16];
            // SAFETY: 16 elements each.
            unsafe {
                _mm512_storeu_epi32(bytes.as_mut_ptr().cast(), latch);
                _mm512_storeu_epi32(lefts.as_mut_ptr().cast(), left);
            }
            for j in each(headers) {
                let bit = 1u16 << j;
                let b = bytes[j] as u8;
                latched &= !bit;
                need |= bit;
                match phase[j] {
                    0 => {
                        // The top two bits shrink the alphabet; width >= 3.
                        let group = width[j] - (b >> 6);
                        columns.set(j, t, group * 16 + (b & 15));
                        reset = if b & 0x10 != 0 {
                            reset | bit
                        } else {
                            reset & !bit
                        };
                        long = if b & 0x20 != 0 {
                            long | bit
                        } else {
                            long & !bit
                        };
                        lefts[j] = 0;
                        phase[j] = 1;
                    }
                    1 if long & bit != 0 => {
                        lefts[j] = b.into();
                        phase[j] = 2;
                    }
                    1 => {
                        lefts[j] = u32::from(b) + 1;
                        phase[j] = 3;
                        payload |= bit;
                    }
                    _ => {
                        lefts[j] = (u32::from(b) << 8 | lefts[j]) + 257;
                        phase[j] = 3;
                        payload |= bit;
                    }
                }
            }
            left = load(&lefts);
        }
        // Payload steps: read the latched byte below 2^24, decode above it.
        let read = stepping & _mm512_cmplt_epu32_mask(x, low) & latched;
        let xr = _mm512_mask_or_epi32(x, read, _mm512_slli_epi32::<8>(x), latch);
        left = _mm512_mask_sub_epi32(left, read, left, one);
        latched &= !read;
        let more = _mm512_test_epi32_mask(left, left);
        need = (need & !stepping) | (stepping & more & !latched);
        let decoding = stepping & _mm512_cmpgt_epu32_mask(xr, low);
        if decoding != 0 {
            // The bucket: the last whose first slot is at or below the
            // slot (the starts ascend); none: bucket 0, symbol 0.
            let slot = _mm512_and_si512(xr, slots);
            let (mut f, mut first, mut high, mut lowh, mut base) =
                (load(&columns.freq[0]), zero, zero, zero, zero);
            let mut any = 0u16;
            for k in 0..BUCKETS - 1 {
                let start = load(&columns.start[k]);
                let m = _mm512_cmple_epu32_mask(start, slot);
                if k == 0 {
                    any = m;
                }
                f = _mm512_mask_mov_epi32(f, m, load(&columns.freq[k + 1]));
                first = _mm512_mask_mov_epi32(first, m, start);
                high = _mm512_mask_mov_epi32(high, m, load(&columns.high[k]));
                lowh = _mm512_mask_mov_epi32(lowh, m, load(&columns.low[k]));
                base = _mm512_mask_mov_epi32(base, m, _mm512_set1_epi32(1 << k));
            }
            // offset * reciprocal >> 32, exactly: offset < 2^16.
            let offset = _mm512_sub_epi32(slot, first);
            let q = _mm512_srli_epi32::<16>(_mm512_add_epi32(
                _mm512_mullo_epi32(offset, high),
                _mm512_srli_epi32::<16>(_mm512_mullo_epi32(offset, lowh)),
            ));
            let rem = _mm512_sub_epi32(offset, _mm512_mullo_epi32(q, f));
            let rem = _mm512_mask_mov_epi32(_mm512_sub_epi32(slot, one), any, rem);
            let sym = _mm512_maskz_mov_epi32(any, _mm512_add_epi32(base, q));
            let next = _mm512_add_epi32(_mm512_mullo_epi32(_mm512_srli_epi32::<16>(xr), f), rem);
            x = _mm512_mask_mov_epi32(x, decoding, next);
            out = _mm512_mask_mov_epi32(out, decoding, sym);
            held |= decoding;
        }
        let idle_lanes = stepping & !decoding;
        // At exactly 2^24 with bytes left the lane neither reads nor decodes.
        x = _mm512_mask_mov_epi32(x, idle_lanes & more, xr);
        let ended = idle_lanes & !more;
        if ended != 0 {
            // The fragment ends; a byte read in this step does not reach
            // the carried state.
            payload &= !ended;
            for j in each(ended) {
                phase[j] = 0;
            }
            need = (need & !ended) | (ended & !latched);
            x = _mm512_mask_mov_epi32(x, ended & reset, zero);
        }
        // The round's bytes, to the lanes that ask, in lane order.
        let waiting = held & done;
        let asking = need & !waiting & valid;
        let room = (src.len() - taken_all).min(ROUND_BYTES) as u32;
        let got = _pdep_u32((1 << room) - 1, asking.into()) as u16;
        let taken = got.count_ones() as usize;
        if taken != 0 {
            // SAFETY: `taken` bytes from `taken_all` on, inside `src`.
            let bytes =
                unsafe { _mm_maskz_expandloadu_epi8(got, src.as_ptr().add(taken_all).cast()) };
            latch = _mm512_mask_mov_epi32(latch, got, _mm512_cvtepu8_epi32(bytes));
            latched |= got;
            taken_all += taken;
        }
        // A lane that asked after the round's 8 bytes were gone waits.
        let late = if taken == ROUND_BYTES {
            !((2u32 << (15 - got.leading_zeros())) - 1) as u16
        } else {
            0
        };
        stalled = ((need & late) | waiting) & valid;
        let deliver = !stalled & held & valid;
        let mut moved = false;
        if deliver != 0 {
            // SAFETY: lane j writes `at + j * per`, inside the records while
            // `pos` is below `blocks * per`.
            unsafe {
                let to = records.as_mut_ptr().add(at).cast();
                if per == 1 {
                    _mm512_mask_cvtepi32_storeu_epi8(to, deliver, out);
                } else {
                    _mm256_mask_storeu_epi8(
                        to,
                        _pdep_u32(deliver.into(), 0x5555_5555),
                        _mm512_cvtepi32_epi16(out),
                    );
                }
            }
            held &= !deliver;
            done |= deliver;
            moved = true;
        }
        if done == valid {
            pos += 1;
            at += if pos.is_multiple_of(per) {
                stride - per + 1
            } else {
                1
            };
            done = 0;
            moved = true;
        }
        // Nothing read and nothing delivered for a few rounds: no lane can go on.
        idle = if taken > 0 || moved { 0 } else { idle + 1 };
        if idle > 3 {
            return Err(Error("the stream ends early or is corrupt"));
        }
    }
    Ok(())
}
