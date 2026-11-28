//! [`super::decode`], faster, with the same bytes out: [`unpack`].
//!
//! The lanes stage is serial by the format (one schedule interleaves every
//! lane's bytes, and which lane takes the next one depends on every lane's
//! state); it runs a round's lanes at once where the processor allows
//! ([`super::wide`]). The blocks stage is two: resolving each block's
//! references against the blocks before it (serial: a block refers to any
//! of the last 256), then packing the resolved values into BC bits, block
//! by block apart. A stream's blocks are resolved a chunk at a time into
//! one of two buffers; while the next chunk resolves, the last one is
//! packed by a [`Pool`]'s helpers (and by this thread when it has resolved
//! its own: no thread waits while there is packing to do).
//!
//! What a failure leaves is what [`super::decode`] leaves: nothing at all
//! when the lanes fail (they all run before the first block is written),
//! every block before the failing one when a block fails.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use super::*;

/// Blocks resolved into one buffer before they are handed out for packing.
const CHUNK: usize = 4096;
/// Blocks a thread packs at a time.
const PIECE: usize = 256;

/// Helper threads that pack resolved blocks, shared by every decode that
/// is given the pool: a fixed number, each running `setup` first (the
/// runtime lowers their priority there, below the title's threads).
pub struct Pool {
    shared: Arc<Shared>,
    threads: usize,
}

struct Shared {
    queue: Mutex<Queue>,
    work: Condvar,
    /// Pieces the helpers ran (not the threads that handed them out).
    helped: std::sync::atomic::AtomicU64,
}

struct Queue {
    jobs: VecDeque<Arc<Job>>,
    closing: bool,
}

/// A chunk's packing: `pieces` calls of `work`, taken one at a time by
/// whoever comes (helpers, and the thread that handed it out).
struct Job {
    work: Box<dyn Fn(usize) + Send + Sync>,
    pieces: usize,
    next: AtomicUsize,
    done: Mutex<usize>,
    finished: Condvar,
    panicked: AtomicBool,
}

impl Job {
    /// Takes and runs one piece; false when none is left to take.
    fn help(&self) -> bool {
        let piece = self.next.fetch_add(1, Ordering::Relaxed);
        if piece >= self.pieces {
            return false;
        }
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (self.work)(piece))).is_err() {
            self.panicked.store(true, Ordering::Relaxed);
        }
        let mut done = self.done.lock().unwrap_or_else(|e| e.into_inner());
        *done += 1;
        if *done == self.pieces {
            self.finished.notify_all();
        }
        true
    }

    fn taken(&self) -> bool {
        self.next.load(Ordering::Relaxed) >= self.pieces
    }
}

impl Pool {
    /// A pool of `threads` helpers (0: every decode packs on its own thread).
    pub fn new(threads: usize, name: &str, setup: fn()) -> Pool {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                jobs: VecDeque::new(),
                closing: false,
            }),
            work: Condvar::new(),
            helped: std::sync::atomic::AtomicU64::new(0),
        });
        let threads = (0..threads)
            .filter(|i| {
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("{name} {i}"))
                    .spawn(move || {
                        setup();
                        helper(&shared);
                    })
                    .is_ok()
            })
            .count();
        Pool { shared, threads }
    }

    /// The helpers that started.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Pieces of work the helpers ran since the pool started (the threads
    /// that handed work out run the rest).
    pub fn helped(&self) -> u64 {
        self.shared.helped.load(Ordering::Relaxed)
    }

    /// Runs `work(0)` to `work(pieces - 1)` on the helpers and this thread,
    /// and returns once all have run. Panics if one did.
    pub fn run(&self, pieces: usize, work: impl Fn(usize) + Send + Sync + 'static) {
        self.submit(pieces, Box::new(work)).finish();
    }

    /// Hands out `pieces` calls of `work`. The returned [`Pending`] helps
    /// and waits for them when finished or dropped.
    fn submit(&self, pieces: usize, work: Box<dyn Fn(usize) + Send + Sync>) -> Pending {
        let job = Arc::new(Job {
            work,
            pieces,
            next: AtomicUsize::new(0),
            done: Mutex::new(0),
            finished: Condvar::new(),
            panicked: AtomicBool::new(false),
        });
        if self.threads > 0 {
            let mut queue = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            queue.jobs.push_back(job.clone());
            drop(queue);
            self.shared.work.notify_all();
        }
        Pending { job, waited: false }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .closing = true;
        self.shared.work.notify_all();
    }
}

fn helper(shared: &Shared) {
    let mut queue = shared.queue.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        while queue.jobs.front().is_some_and(|j| j.taken()) {
            queue.jobs.pop_front();
        }
        if let Some(job) = queue.jobs.front().cloned() {
            drop(queue);
            while job.help() {
                shared.helped.fetch_add(1, Ordering::Relaxed);
            }
            queue = shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            continue;
        }
        if queue.closing {
            return;
        }
        queue = shared.work.wait(queue).unwrap_or_else(|e| e.into_inner());
    }
}

/// A job handed out: finishing (or dropping) it takes what pieces are left
/// on this thread, then waits for those the helpers took.
struct Pending {
    job: Arc<Job>,
    waited: bool,
}

impl Pending {
    fn finish(mut self) {
        self.waited = true;
        assert!(!self.wait(), "BCPack: a block failed to pack");
    }

    fn wait(&self) -> bool {
        while self.job.help() {}
        let mut done = self.job.done.lock().unwrap_or_else(|e| e.into_inner());
        while *done < self.job.pieces {
            done = self
                .job
                .finished
                .wait(done)
                .unwrap_or_else(|e| e.into_inner());
        }
        self.job.panicked.load(Ordering::Relaxed)
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        // Unwinding past a chunk in flight: its buffers must outlive it.
        if !self.waited {
            self.wait();
        }
    }
}

/// A buffer's address, for a job's pieces (which touch disjoint parts).
#[derive(Clone, Copy)]
struct At<T>(*mut T);
// SAFETY: see `blocks`: each piece reads its own values and writes its own
// blocks, and the buffers outlive the job.
unsafe impl<T> Send for At<T> {}
// SAFETY: as above.
unsafe impl<T> Sync for At<T> {}

impl<T> At<T> {
    /// The address `k` elements on (a method: a closure then takes the
    /// whole `At`, not its pointer).
    fn add(self, k: usize) -> *mut T {
        self.0.wrapping_add(k)
    }
}

/// Resolves every block in order and packs them, `block` bytes each, into
/// `dst`: one after another without helpers, else as the module says.
fn blocks<
    V: Copy + Default + Send + Sync + 'static,
    P: Fn(&V, &mut [u8]) + Copy + Send + Sync + 'static,
>(
    records: &[u8],
    stride: usize,
    dst: &mut [u8],
    block: usize,
    mut resolve: impl FnMut(usize, &[u8]) -> Result<V, Error>,
    pack: P,
    pool: Option<&Pool>,
) -> Result<(), Error> {
    let count = dst.len() / block;
    let Some(pool) = pool.filter(|p| p.threads() > 0 && count > CHUNK / 2) else {
        for (i, (s, out)) in records
            .chunks_exact(stride)
            .zip(dst.chunks_exact_mut(block))
            .enumerate()
        {
            pack(&resolve(i, s)?, out);
        }
        return Ok(());
    };
    let mut buffers = [vec![V::default(); CHUNK], vec![V::default(); CHUNK]];
    // From here on `dst` is written through this only.
    let dst = At(dst.as_mut_ptr());
    let mut pending: Option<Pending> = None;
    let mut result = Ok(());
    let mut first = 0;
    for which in (0..2).cycle() {
        if first >= count {
            break;
        }
        let n = CHUNK.min(count - first);
        // The other buffer may be packing; this one is free (its job
        // finished before the other's was handed out).
        let values = &mut buffers[which][..n];
        let mut made = 0usize;
        for (k, v) in values.iter_mut().enumerate() {
            let i = first + k;
            match resolve(i, &records[i * stride..][..stride]) {
                Ok(r) => *v = r,
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
            made += 1;
        }
        if let Some(p) = pending.take() {
            p.finish();
        }
        let values = At(buffers[which].as_mut_ptr());
        // Block `first` on, inside `dst` (`first` < `count`).
        let out = At(dst.add(first * block));
        pending = Some(pool.submit(
            made.div_ceil(PIECE),
            Box::new(move |piece| {
                let from = piece * PIECE;
                let to = (from + PIECE).min(made);
                for k in from..to {
                    // SAFETY: value `k` of this chunk's buffer and its block
                    // in `dst`: no other piece touches them, nothing else
                    // writes the buffer or reads the blocks until the job
                    // has finished (`Pending`), and both outlive it.
                    let (v, out) = unsafe {
                        (
                            &*values.add(k),
                            std::slice::from_raw_parts_mut(out.add(k * block), block),
                        )
                    };
                    pack(v, out);
                }
            }),
        ));
        first += n;
        if result.is_err() {
            break;
        }
    }
    if let Some(p) = pending.take() {
        p.finish();
    }
    result
}

/// Unpacks as [`decode`] does, into the same bytes (and on a failure, the
/// same of them), faster: see the module. `pool`: helpers for the packing
/// of BC6H and BC7 blocks (`None`: all on this thread).
pub fn unpack(
    bc: u8,
    src: &[u8],
    dst: &mut [u8],
    tables: &Tables,
    pool: Option<&Pool>,
) -> Result<(), Error> {
    unpack_with(bc, src, dst, tables, pool, true)
}

/// [`unpack`]; `wide`: the lanes a round at once where the processor
/// allows, else always one at a time (the tests take both).
pub(super) fn unpack_with(
    bc: u8,
    src: &[u8],
    dst: &mut [u8],
    tables: &Tables,
    pool: Option<&Pool>,
    wide: bool,
) -> Result<(), Error> {
    let block = match bc {
        1 | 4 => 8,
        3 | 5 | 6 | 7 => 16,
        _ => return Err(Error("unsupported BC format")),
    };
    if dst.is_empty() || !dst.len().is_multiple_of(block) {
        return Err(Error("destination is not a whole number of blocks"));
    }
    let count = dst.len() / block;
    let (widths, per): (&[u8], usize) = match bc {
        1 => (&[7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7], 1),
        3 => (&[3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7], 2),
        4 => (&[7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 1),
        5 => (&[3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 2),
        6 => (&[3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4], 2),
        _ => (&[3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4], 2),
    };
    let stride = widths.len() * per;
    let mut records = vec![0u8; count * stride];
    #[cfg(target_arch = "x86_64")]
    let wide = wide
        .then(|| super::wide::lanes(tables, widths, per, count, src, &mut records))
        .flatten();
    #[cfg(not(target_arch = "x86_64"))]
    let wide: Option<Result<(), Error>> = wide.then_some(None).flatten();
    match wide {
        Some(result) => result?,
        None => {
            let mut next = src.iter().copied();
            lanes(tables, widths, per, count, |_| next.next(), &mut records)?;
        }
    }
    let t = tables;
    // BC1-BC5 blocks pack in a few nanoseconds: handing chunks out costs
    // more than it saves (BC1 181 MB/s alone, 132 with four helpers).
    let pool = pool.filter(|_| matches!(bc, 6 | 7));
    match bc {
        1 => blocks(
            &records,
            stride,
            dst,
            block,
            Plains::new(t, &t.bc1, &BC1, bc1_residual).resolver(),
            bc1_pack,
            pool,
        ),
        3 => blocks(
            &records,
            stride,
            dst,
            block,
            Plains::new(t, &t.bc3, &BC3, bc3_residual).resolver(),
            bc3_pack,
            pool,
        ),
        4 => blocks(
            &records,
            stride,
            dst,
            block,
            Plains::new(t, &t.bc4, &BC4, bc4_residual).resolver(),
            bc4_pack,
            pool,
        ),
        5 => blocks(
            &records,
            stride,
            dst,
            block,
            Plains::new(t, &t.bc5, &BC5, bc5_residual).resolver(),
            bc5_pack,
            pool,
        ),
        6 => blocks(
            &records,
            stride,
            dst,
            block,
            bc6_resolver(&t.bc6),
            bc6_pack,
            pool,
        ),
        _ => blocks(
            &records,
            stride,
            dst,
            block,
            bc7_resolver(t),
            |v: &Bc7Value, out: &mut [u8]| bc7_pack(&v.ep, &v.ix, v.prec, out),
            pool,
        ),
    }
}

// Resolving: [`super::plain`], [`super::bc6`] and [`super::bc7`] up to
// their packing, block by block.

struct Plains<'a, R> {
    t: &'a Tables,
    dictionary: &'a [Plain],
    paired: bool,
    /// Each element's width as a mask: the sums then vectorise.
    mask: Plain,
    residual: R,
    history: Vec<Plain>,
}

impl<'a, R: Fn(&Tables, &[u8]) -> Plain> Plains<'a, R> {
    fn new(t: &'a Tables, dictionary: &'a [Plain], w: &'static Widths, residual: R) -> Self {
        Plains {
            t,
            dictionary,
            paired: w.paired,
            mask: Plain {
                ep: w.ep.map(|bits| ((1u16 << bits) - 1) as u8),
                ix: w.ix.map(|bits| (1u8 << bits) - 1),
            },
            residual,
            history: vec![Plain::default(); HISTORY],
        }
    }

    fn resolver(mut self) -> impl FnMut(usize, &[u8]) -> Result<Plain, Error> + use<'a, R> {
        move |i, s| Ok(self.resolve(i, s))
    }

    fn resolve(&mut self, i: usize, s: &[u8]) -> Plain {
        let zero = Plain::default();
        let (ep_ref, ix_ref) = references(s, self.paired);
        let ep_ref = pick(i, ep_ref, &self.history, self.dictionary, &zero).ep;
        let ix_ref = pick(i, ix_ref, &self.history, self.dictionary, &zero).ix;
        let mut v = (self.residual)(self.t, s);
        for k in 0..8 {
            v.ep[k] = v.ep[k].wrapping_add(ep_ref[k]) & self.mask.ep[k];
        }
        for k in 0..32 {
            v.ix[k] = v.ix[k].wrapping_add(ix_ref[k]) & self.mask.ix[k];
        }
        self.history[i % HISTORY] = v;
        v
    }
}

/// A BC7 block resolved: what [`bc7_pack`] takes.
#[derive(Clone, Copy, Default)]
struct Bc7Value {
    ep: Bc7Ep,
    ix: Bc7Ix,
    prec: [u8; 4],
}

fn bc7_resolver(t: &Tables) -> impl FnMut(usize, &[u8]) -> Result<Bc7Value, Error> + '_ {
    let mut history = vec![(Bc7Ep::default(), Bc7Ix::default()); HISTORY];
    let zero = (Bc7Ep::default(), Bc7Ix::default());
    move |i, s| {
        let (ep_ref, ix_ref) = references(s, true);
        let ctrl = t.ctrl7[usize::from(s[31] & 15) << 5 | usize::from(s[30] & 31)];
        if ctrl == 0 {
            return Err(Error("unused BC7 control code"));
        }
        let mode = usize::from(ctrl & 7);
        let subsets = usize::from((ctrl & 0xff) >> 6);
        let prec = bc7_precision(ctrl);
        let three = subsets == 3;
        let or0 = |v: u8| if three { 0 } else { v };
        let if3 = |v: u8| if three { v } else { 0 };
        let stored = [
            [s[4], s[5], s[6], s[7], s[8], s[9], or0(s[10]), or0(s[11])],
            [
                s[12],
                s[13],
                s[14],
                s[15],
                s[16],
                s[17],
                or0(s[18]),
                or0(s[19]),
            ],
            [
                s[20],
                s[21],
                if3(s[10]),
                if3(s[11]),
                if3(s[18]),
                if3(s[19]),
                0,
                0,
            ],
        ];
        let r = &pick(i, ep_ref, &history, &t.bc7, &zero).0;
        let mut ep = Bc7Ep {
            ctrl,
            ..Default::default()
        };
        // Rows the reference lacks build on its first row.
        let common = subsets.min(usize::from((r.ctrl & 0xff) >> 6));
        let columns = if prec[3] > 0 { 8 } else { 6 };
        for c in 0..columns {
            let p = prec[c / 2];
            if p == 0 {
                continue;
            }
            let mask = 0xffffu16 << (16 - p);
            for (row, st) in stored.iter().enumerate().take(subsets) {
                let v = st[c];
                let residual =
                    (u16::from(v >> 1) ^ (v & 1).wrapping_neg() as i8 as u16) << (16 - p);
                let base = r.rows[if row < common { row } else { 0 }][c] & mask;
                ep.rows[row][c] = base.wrapping_add(residual);
            }
        }
        let bits = BC7_INDEX[mode];
        let rx = &pick(i, ix_ref, &history, &t.bc7, &zero).1;
        let mut ix = Bc7Ix {
            bits,
            ..Default::default()
        };
        for (set, offsets) in [[22, 24, 26, 28], [12, 14, 16, 20]].into_iter().enumerate() {
            if bits[set] == 0 {
                continue;
            }
            for (k, o) in offsets.into_iter().enumerate() {
                for (b, &byte) in s[o..o + 2].iter().enumerate() {
                    let pair = nibbles(byte, k < 3);
                    for (h, &v) in pair.iter().enumerate() {
                        let p = 4 * k + 2 * b + h;
                        ix.ix[set][p] = add_index(v, bits[set], rx.ix[set][p], rx.bits[set]);
                    }
                }
            }
        }
        history[i % HISTORY] = (ep, ix);
        Ok(Bc7Value { ep, ix, prec })
    }
}

/// A BC6H block resolved: its endpoints, indices and mode, and partition.
#[derive(Clone, Copy, Default)]
struct Bc6Value {
    ep: Bc6Ep,
    ix: Bc6Ix,
    part: u8,
}

fn bc6_resolver(
    dictionary: &[(Bc6Ep, Bc6Ix)],
) -> impl FnMut(usize, &[u8]) -> Result<Bc6Value, Error> + '_ {
    let mut history = vec![(Bc6Ep::default(), Bc6Ix::default()); HISTORY];
    let zero = (
        Bc6Ep::default(),
        Bc6Ix {
            mode: 11,
            ..Default::default()
        },
    );
    let z = |v: u16| (v >> 1) ^ (v & 1).wrapping_neg();
    let wide = |a: u8, b: u8| {
        (0..8).fold(0u16, |acc, k| {
            acc | u16::from(a >> k & 1) << (2 * k) | u16::from(b >> k & 1) << (2 * k + 1)
        })
    };
    move |i, s| {
        let (ep_ref, ix_ref) = references(s, true);
        let mode = s[30].wrapping_add(1);
        if !(1..=14).contains(&mode) {
            return Err(Error("BC6H mode out of range"));
        }
        let two = mode < 11;
        let shift = BC6_BITS[usize::from(mode)].map(|b| 16 - u32::from(b));
        let e0 = [wide(s[4], s[5]), wide(s[6], s[7]), wide(s[8], s[9])].map(z);
        let e1 = [wide(s[10], s[11]), wide(s[12], s[13]), wide(s[14], s[15])].map(z);
        let (e2, e3) = if two {
            (
                [s[16], s[18], s[20]].map(|v| z(v.into())),
                [s[17], s[19], s[21]].map(|v| z(v.into())),
            )
        } else {
            (e0, e1)
        };
        let r = &pick(i, ep_ref, &history, dictionary, &zero).0;
        let mut ep = Bc6Ep::default();
        // Endpoint 0 at the base precision, the others per channel; the
        // reference is cut by the mask after the sum.
        let mask = |sh: u32| 0xffffu16 << sh;
        for ((e, &r), res) in ep.e[0].iter_mut().zip(&r.e[0]).zip(e0) {
            *e = r.wrapping_add(res << shift[0]) & mask(shift[0]);
        }
        for (k, res) in [e1, e2, e3]
            .iter()
            .enumerate()
            .take(if two { 3 } else { 1 })
        {
            for c in 0..3 {
                ep.e[k + 1][c] =
                    r.e[k + 1][c].wrapping_add(res[c] << shift[1 + c]) & mask(shift[1 + c]);
            }
        }
        if !two {
            for c in 0..3 {
                ep.e[2][c] = (ep.e[0][c] >> shift[0]) << shift[1];
            }
            ep.e[3] = ep.e[1];
        }
        let rx = &pick(i, ix_ref, &history, dictionary, &zero).1;
        let bits = bc6_index_bits(mode);
        let mut ix = Bc6Ix {
            mode,
            ..Default::default()
        };
        for k in 0..8 {
            for (h, v) in nibbles(s[22 + k], k < 6).into_iter().enumerate() {
                let p = 2 * k + h;
                ix.ix[p] = add_index(v, bits, rx.ix[p], bc6_index_bits(rx.mode));
            }
        }
        history[i % HISTORY] = (ep, ix);
        Ok(Bc6Value {
            ep,
            ix,
            part: s[31] & 31,
        })
    }
}

/// [`super::bc6`]'s packing of a resolved block.
fn bc6_pack(v: &Bc6Value, block: &mut [u8]) {
    let (ep, ix, mode) = (&v.ep, &v.ix, v.ix.mode);
    let two = mode < 11;
    let shift = BC6_BITS[usize::from(mode)].map(|b| 16 - u32::from(b));
    let bits = bc6_index_bits(mode);
    // Fields: the working value has its top bit flipped.
    let field = |k: usize, c: usize| {
        let sh = if k == 0 { shift[0] } else { shift[1 + c] };
        u64::from((ep.e[k][c] ^ 0x8000) >> sh)
    };
    let mut w = Bits::default();
    let (code, n) = BC6_CODE[usize::from(mode)];
    w.put(code, n);
    let part = v.part;
    for run in BC6_LAYOUT[usize::from(mode)] {
        if run.ch == 3 {
            w.put(part.into(), 5);
        } else {
            w.put(
                field(run.ep.into(), run.ch.into()) >> run.first,
                run.len.into(),
            );
        }
    }
    let anchor = if two { ANCHOR2[usize::from(part)] } else { 0 };
    for (p, &v) in ix.ix.iter().enumerate() {
        let short = p == 0 || p == usize::from(anchor);
        let v = (v as u16).wrapping_add(1 << (bits - 1));
        w.put(v.into(), u32::from(bits) - u32::from(short));
    }
    block.copy_from_slice(&w.v.to_le_bytes());
}
