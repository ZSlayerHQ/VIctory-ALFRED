//! Upload fills at submission, on a pool with `VICTORY_UPLOAD_POOL=1` (off by default): the
//! copies from title memory into upload memory that are most of the preparation's copy time,
//! split over workers where the fills of a step are independent.
//!
//! The join: [`run_fills`] does not return until every fill of a step has finished, and the
//! step loop calls it in place — the submitting thread cannot go on to the next step, the
//! send of the list to the card, or a release of title memory before that. The reads of
//! title memory are the same kernel reads (`ReadProcessMemory`) the serial path makes, in
//! the same interval of the submission: nothing here holds a pointer past the join.

use super::{Deferred, Step, copy_mapped};

/// Whether upload fills run on a pool at submission (`VICTORY_UPLOAD_POOL=1`; off by
/// default).
fn upload_pool() -> bool {
    crate::live::upload_pool()
}

/// How many workers the fills of a step may go to: one (serially) unless the pool is on, and
/// otherwise one fewer than the machine's processors, at most six.
pub(crate) fn fill_workers() -> usize {
    if !upload_pool() {
        return 1;
    }
    std::thread::available_parallelism()
        .map_or(1, |n| n.get().saturating_sub(1))
        .clamp(1, 6)
}

/// One upload fill: `len` bytes of title memory at `src` into upload memory at `dst`.
#[derive(Clone, Copy)]
pub(crate) struct Fill {
    pub(crate) dst: usize,
    pub(crate) src: usize,
    pub(crate) len: usize,
}

impl Fill {
    /// Runs it and returns the bytes moved ([`copy_mapped`]'s count: none with
    /// `VICTORY_COPY_MODE=none`).
    fn run(self) -> usize {
        copy_mapped(self.dst as *mut u8, self.src as *const u8, self.len, false)
    }
}

impl Step {
    /// The fills a step runs: one, a group's, or none.
    pub(crate) fn fills(&self) -> &[Fill] {
        match self {
            Self::Fill(fill) => std::slice::from_ref(fill),
            Self::Fills(group) => group,
            _ => &[],
        }
    }
}

/// A group is made once this many independent fills run together, and holds at most
/// [`FILLS_PER_GROUP`].
const FILLS_PER_JOB: usize = 128;
const FILLS_PER_GROUP: usize = 4096;

impl Deferred {
    /// Adds a fill to the list's steps: with more than one worker, a run of fills with no
    /// other step between them is grouped so [`run_fills`] can split it; with one, each is
    /// its own step, exactly as before the pool existed.
    pub(crate) fn append_fill(&mut self, fill: Fill, workers: usize) {
        if workers > 1 {
            if let Some(Step::Fills(group)) = self.copies.last_mut() {
                if group.len() < FILLS_PER_GROUP {
                    group.push(fill);
                    return;
                }
            } else {
                let run = self
                    .copies
                    .iter()
                    .rev()
                    .take_while(|step| matches!(step, Step::Fill(_)))
                    .count();
                if run + 1 == FILLS_PER_JOB {
                    let mut group = Vec::with_capacity(FILLS_PER_JOB);
                    for step in self.copies.drain(self.copies.len() - run..) {
                        let Step::Fill(f) = step else { unreachable!() };
                        group.push(f);
                    }
                    group.push(fill);
                    self.copies.push(Step::Fills(group));
                    return;
                }
            }
        }
        self.copies.push(Step::Fill(fill));
    }
}

/// Whether the fills of a group may run at once: the destinations strictly increasing (no
/// two fills write the same bytes) and every source outside the whole destination span (no
/// fill reads bytes another of them writes). An overflowing range is not independent.
fn independent_fills(fills: &[Fill]) -> bool {
    let Some(first) = fills.first() else {
        return false;
    };
    let mut end = first.dst;
    for fill in fills {
        let Some(next) = fill.dst.checked_add(fill.len) else {
            return false;
        };
        if fill.dst < end {
            return false;
        }
        end = next;
    }
    fills
        .iter()
        .all(|f| f.src.checked_add(f.len).is_some_and(|src_end| f.src >= end || src_end <= first.dst))
}

#[cfg(test)]
thread_local! {
    /// A test hook: each job sleeps this long before its fills, to prove the join waits.
    static SLOW_FILL: std::cell::Cell<std::time::Duration> = const { std::cell::Cell::new(std::time::Duration::ZERO) };
}

/// Runs the fills of one step and returns the bytes they moved. Every fill has finished when
/// this returns — the join: the submitting thread does not go on to the next step, the send
/// of the list to the card, or a release of title memory until the pool has run them all
/// (the [`crate::shadow::wide_run`] call waits; there is no detached work).
pub(crate) fn run_fills(fills: &[Fill], workers: usize) -> usize {
    let jobs = workers.min(fills.len() / FILLS_PER_JOB);
    if jobs > 1 && independent_fills(fills) {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let moved = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let queued = fills
            .chunks(fills.len().div_ceil(jobs))
            .map(|chunk| {
                let owned = chunk.to_vec();
                let moved = moved.clone();
                #[cfg(test)]
                let slow = SLOW_FILL.with(std::cell::Cell::get);
                Box::new(move || {
                    #[cfg(test)]
                    std::thread::sleep(slow);
                    let sum: usize = owned.into_iter().map(Fill::run).sum();
                    moved.fetch_add(sum as u64, std::sync::atomic::Ordering::Relaxed);
                }) as Box<dyn FnOnce() + Send>
            })
            .collect();
        crate::shadow::wide_run(queued);
        let moved = moved.load(std::sync::atomic::Ordering::Relaxed) as usize;
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::UploadPoolBytes, moved as u64);
            add(C::UploadPoolJoinUs, started.elapsed().as_micros() as u64);
        }
        moved
    } else {
        fills.iter().map(|fill| fill.run()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `count` fills of `bytes` bytes each down a buffer: one destination chunk after
    /// another, sources from `source`.
    fn buffer_fills(source: &[u8], dst: &mut [u8], count: usize, bytes: usize) -> Vec<Fill> {
        assert_eq!(dst.len(), count * bytes, "the destination fits the fills");
        dst.chunks_exact_mut(bytes)
            .enumerate()
            .map(|(i, out)| Fill {
                dst: out.as_mut_ptr() as usize,
                src: source[i * bytes..].as_ptr() as usize,
                len: bytes,
            })
            .collect()
    }

    /// `count` fills of `bytes` bytes each down a mipped texture: rows of a 320-byte pitch
    /// (64 bytes of padding a row), as an upload's rows are, each from its own row of
    /// `source`.
    fn texture_fills(source: &[u8], dst: &mut [u8], count: usize, bytes: usize) -> Vec<Fill> {
        const PITCH: usize = 320;
        // SAFETY: every row stands inside `dst` (the caller sizes it count * PITCH) and
        // only addresses are taken; the copy itself is the tested code.
        (0..count)
            .map(|i| Fill {
                dst: unsafe { dst.as_mut_ptr().add(i * PITCH) } as usize,
                src: source[i * PITCH..].as_ptr() as usize,
                len: bytes,
            })
            .collect()
    }

    /// A buffer's fills: the pool lands the same bytes as serial, and a later edit of the
    /// source is seen by both (the fills copy at run time, not at recording).
    #[test]
    fn pooled_fills_land_the_same_bytes_for_a_buffer() {
        let (count, bytes) = (4 * FILLS_PER_JOB, 448);
        let mut source: Vec<u8> = (0..count * bytes).map(|i| (i % 251) as u8).collect();
        for turn in [0u8, 0x5a] {
            for b in &mut source {
                *b ^= turn;
            }
            let mut serial = vec![0u8; count * bytes];
            let mut pooled = vec![0u8; count * bytes];
            let serial_fills = buffer_fills(&source, &mut serial, count, bytes);
            let pooled_fills = buffer_fills(&source, &mut pooled, count, bytes);
            assert_eq!(run_fills(&serial_fills, 1), count * bytes);
            assert_eq!(run_fills(&pooled_fills, 4), count * bytes);
            assert_eq!(serial, pooled, "the pool lands the same bytes");
            assert_eq!(&serial[..], &source[..]);
        }
    }

    /// A mipped texture's fills (rows of a 320-byte pitch): the same bytes, serial and on
    /// the pool, padding untouched.
    #[test]
    fn pooled_fills_land_the_same_bytes_for_a_mipped_texture() {
        const PITCH: usize = 320;
        let (count, bytes) = (4 * FILLS_PER_JOB, 256);
        let source: Vec<u8> = (0..count * PITCH).map(|i| (i % 251) as u8).collect();
        let mut serial = vec![0xeeu8; count * PITCH];
        let mut pooled = vec![0xeeu8; count * PITCH];
        let serial_fills = texture_fills(&source, &mut serial, count, bytes);
        let pooled_fills = texture_fills(&source, &mut pooled, count, bytes);
        run_fills(&serial_fills, 1);
        run_fills(&pooled_fills, 4);
        assert_eq!(serial, pooled, "the pool lands the same bytes");
        for row in 0..count {
            assert_eq!(&serial[row * PITCH..row * PITCH + bytes], &source[row * PITCH..row * PITCH + bytes]);
            assert!(serial[row * PITCH + bytes..(row + 1) * PITCH].iter().all(|&b| b == 0xee), "padding untouched");
        }
    }

    /// The join: with a job made slow, `run_fills` does not return until its fill is done,
    /// so nothing after the step can read the upload or release its memory early.
    #[test]
    fn the_join_waits_for_a_slow_fill() {
        let (count, bytes) = (4 * FILLS_PER_JOB, 64);
        let source: Vec<u8> = (0..count * bytes).map(|i| (i % 251) as u8).collect();
        let mut dst = vec![0u8; count * bytes];
        let fills = buffer_fills(&source, &mut dst, count, bytes);
        let hook = SLOW_FILL.with(|s| s.replace(std::time::Duration::from_millis(80)));
        let started = std::time::Instant::now();
        let moved = run_fills(&fills, 4);
        let took = started.elapsed();
        SLOW_FILL.with(|s| s.set(hook));
        assert!(took >= std::time::Duration::from_millis(75), "the join waited for the slow job: {took:?}");
        assert_eq!(moved, count * bytes);
        assert_eq!(&dst[..], &source[..]);
    }

    /// Overlapping destinations are not independent: such a run stays serial and still
    /// lands the same bytes.
    #[test]
    fn dependent_fills_stay_serial() {
        let source = vec![7u8; 256];
        let mut dst = vec![0u8; 256];
        let base = dst.as_mut_ptr() as usize;
        let fills = vec![
            Fill { dst: base, src: source.as_ptr() as usize, len: 128 },
            Fill { dst: base + 64, src: source.as_ptr() as usize, len: 128 },
        ];
        assert!(!independent_fills(&fills));
        run_fills(&fills, 4);
        assert_eq!(&dst[..128], &source[..128]);
        assert_eq!(&dst[64..192], &source[..128]);
    }
}
