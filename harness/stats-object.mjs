// One object of the runtime's live stats file (`VICTORY_LIVE_STATS`, victory-gpu `host/stats.rs` `json`), read into one
// record so no reader keeps its own copy of the names. Two shapes are read:
//
//   interval  (world f3ce91562 on, job/timing-truth-2): `seq`, `t_ms`, `unix_ms`, `interval` {first_frame, last_frame,
//             frames, wall_ms}; every mean is over `interval.frames`, the frames since the previous object;
//             `frame_ms_p50`, `frame_ms_worst` (the object's longest frame: a few frames have no 95th percentile),
//             `frame_ms_list`; `budget` (with `calls`, `largest_title_gap_ms`, `title_and_calls_cpu_ms`),
//             `other_threads`, `summed_across_threads`, `gpu_stamps`, `dispatched` (per frame), `*_total` counters,
//             `segments_closed_total`, `segment_open` (transition, light and long frames left out, each with its mean;
//             `begun` start, change or mark), `kernel_calls`, then the optional parts while their switches are on.
//   sample    (before job/timing-truth, the pre-rebuild shape): one second's sample about once in six; `parts`,
//             `dispatches` (a count since the previous object, not per frame), `stores_failed`, `pipelines_created`,
//             `segment` (the closed segments); no seq, no clock, no window.
//
// The first rebuilt shape (world b1714baf6 to f3ce91562: `frame_ms_p95`, `segment`, `segment_open.excluded`) is read
// as the interval shape, and the fields it named otherwise are null: no run log of that build has a stats history. A
// field the shape lacks is null, never zero, and `shapeName` names the shape for a reader's "n/a". The shapes are fixed
// by fixtures/runtime-stats.mjs, copied from the runtime's own tests.
import { RUNTIME_PHASES } from './fixtures/runtime-stats.mjs';

const num = (value) => (typeof value === 'number' && Number.isFinite(value) ? value : null);
const obj = (value) => (value && typeof value === 'object' && !Array.isArray(value) ? value : null);

/// The budget's phases in the runtime's order (budget.rs `NAMES`), from the fixture the drift test checks.
export { RUNTIME_PHASES };

/// The shapes' names, for a reader's "n/a (...)".
export const SHAPE_NAMES = Object.freeze({
  interval: 'the interval shape (seq, unix_ms, interval)',
  sample: 'the pre-rebuild sample shape (no seq, clock or window)',
});

const OPTIONAL_HASH = ['hash_missed_pieces_total', 'hash_checked_pieces_total', 'hash_check_skipped_total'];
/// The optional parts in the order the runtime appends them (stats.rs `optional`).
export const OPTIONAL_PARTS = Object.freeze(['hash_check', 'texture_receipts', 'landed_classes', 'texture_recheck', 'textures_made']);

function phasesOf(raw) {
  const phases = {};
  for (const [name, value] of Object.entries(obj(raw) ?? {})) {
    const entry = obj(value);
    if (!entry || !('ms' in entry)) continue;
    phases[name] = { ms: num(entry.ms), cpuMs: num(entry.cpu_ms) };
  }
  return phases;
}

function budgetOf(raw) {
  const budget = obj(raw);
  if (!budget) return null;
  return {
    thread: typeof budget.thread === 'string' ? budget.thread : null,
    frames: num(budget.frames),
    frameMs: num(budget.frame_ms),
    sumMs: num(budget.sum_ms),
    scopes: num(budget.scopes),
    calls: num(budget.calls),
    largestTitleGapMs: num(budget.largest_title_gap_ms),
    // The processor time of title and the title's calls together: none of the four has one of its own (null there).
    titleAndCallsCpuMs: num(budget.title_and_calls_cpu_ms),
    phases: phasesOf(budget.phases),
  };
}

function segmentOpenOf(raw) {
  const open = obj(raw);
  if (!open) return null;
  const counted = num(open.counted);
  // A segment with nothing counted prints 0.00 for its figures, and a kind with no frame left out 0.00 for its mean:
  // they are no figures.
  const figure = (value) => (counted ? num(value) : null);
  const left = (count, mean) => ({ count: num(count), meanMs: num(count) ? num(mean) : null });
  const transition = left(open.transition_excluded, open.transition_mean_ms);
  const light = left(open.light_excluded, open.light_mean_ms);
  const long = left(open.long_excluded, open.long_mean_ms);
  return {
    segment: num(open.segment),
    firstFrame: num(open.first_frame),
    lastFrame: num(open.last_frame),
    counted,
    transitionExcluded: transition.count,
    transitionMeanMs: transition.meanMs,
    lightExcluded: light.count,
    lightMeanMs: light.meanMs,
    longExcluded: long.count,
    longMeanMs: long.meanMs,
    meanMs: figure(open.mean_ms),
    p50Ms: figure(open.p50_ms),
    p95Ms: figure(open.p95_ms),
    begun: ['start', 'change', 'mark'].includes(open.begun) ? open.begun : null,
  };
}

/// One parsed stats object as a record, or null when it is not one (no object, or no finite `frame`).
export function readStatsObject(raw) {
  const object = obj(raw);
  if (!object || num(object.frame) === null) return null;
  const interval = obj(object.interval);
  const shape = num(object.seq) !== null || interval ? 'interval' : 'sample';
  const list = Array.isArray(object.frame_ms_list) ? object.frame_ms_list.filter((value) => num(value) !== null) : null;
  const total = (name, old) => (shape === 'interval' ? num(object[`${name}_total`]) : num(object[old ?? name]));
  const hashOn = OPTIONAL_HASH.every((key) => key in object);
  return {
    shape,
    shapeName: SHAPE_NAMES[shape],
    seq: num(object.seq),
    tMs: num(object.t_ms),
    unixMs: num(object.unix_ms),
    interval: interval
      ? {
          firstFrame: num(interval.first_frame),
          lastFrame: num(interval.last_frame),
          frames: num(interval.frames),
          wallMs: num(interval.wall_ms),
        }
      : null,
    frame: num(object.frame),
    fps: num(object.fps),
    frameMs: num(object.frame_ms),
    frameMsP50: num(object.frame_ms_p50),
    // The longest of the object's frames (not a percentile); an arm's percentiles come from the joined lists.
    frameMsWorst: num(object.frame_ms_worst),
    frameMsList: list,
    frameMsListMissing: num(object.frame_ms_list_missing),
    budget: budgetOf(object.budget),
    otherThreads: obj(object.other_threads)
      ? { scopes: num(object.other_threads.scopes), phases: phasesOf(object.other_threads.phases) }
      : null,
    // The figures summed across threads: `summed_across_threads`, called `parts` before. They overlap each other and
    // the frame; they are never parts of it.
    summed: obj(object.summed_across_threads) ?? obj(object.parts),
    // The card's queues by their timestamps (`host/queue_stamps.rs`): `copy_queue`, `not_stamped`, and `idle`, a
    // text saying idle time is not derivable (there is no idle figure). Null in a build without diagnostics.
    gpuStamps: obj(object.gpu_stamps),
    draws: num(object.draws),
    // Per frame in the interval shape; the sample shape's `dispatches` was a count since the previous object.
    dispatched: shape === 'interval' ? num(object.dispatched) : null,
    dispatchesSincePrevious: shape === 'sample' ? num(object.dispatches) : null,
    dropped: num(object.dropped),
    storesFailed: total('stores_failed'),
    storesFailedBytes: total('stores_failed_bytes'),
    pipelinesCreated: total('pipelines_created'),
    pipelinesPerMinute: num(object.pipelines_per_minute),
    vramMb: num(object.vram_mb),
    live: obj(object.live),
    // The segments closed so far: `segments_closed_total`, called `segment` in the pre-rebuild sample.
    segmentsClosed: shape === 'interval' ? num(object.segments_closed_total) : num(object.segment),
    segmentOpen: segmentOpenOf(object.segment_open),
    // Always written: `per_frame` over interval.frames (null in the first object) and counts since the start (`_total`).
    kernelCalls: obj(object.kernel_calls)
      ? { perFrame: obj(object.kernel_calls.per_frame), raw: object.kernel_calls }
      : null,
    // The optional parts (stats.rs `optional`): present only while their switch is on, so null here means the switch is
    // off (or, in an older build, that it had none), never zeros. Their counts are named `_total`. `optionalOn` lists
    // the parts present, so `landed_classes` on before its first window (its value `null`) is told from off.
    hashCheck: hashOn
      ? {
          missedPiecesTotal: num(object.hash_missed_pieces_total),
          checkedPiecesTotal: num(object.hash_checked_pieces_total),
          skippedTotal: num(object.hash_check_skipped_total),
        }
      : null,
    textureReceipts: obj(object.texture_receipts),
    landedClasses: obj(object.landed_classes),
    textureRecheck: obj(object.texture_recheck),
    texturesMade: obj(object.textures_made),
    optionalOn: OPTIONAL_PARTS.filter((name) => (name === 'hash_check' ? hashOn : name in object)),
    raw: object,
  };
}

/// The text of the stats file as a record, or null when it does not parse or is not a stats object.
export function parseStatsText(text) {
  try {
    return readStatsObject(JSON.parse(text));
  } catch {
    return null;
  }
}

/// How old the object is by its own wall clock (`unix_ms`), in ms, or null where the shape has no clock (the caller
/// then falls back to the file's modification time).
export function objectAgeMs(record, nowMs = Date.now()) {
  return record?.unixMs == null ? null : nowMs - record.unixMs;
}

/// How `next` follows `previous` by `seq`: 'first', 'next', 'repeat' (the same object read again), 'missed' (a jump:
/// objects were written and not read; `missed` says how many), 'older' (a lower seq: a new run or a reset), or null
/// where either has no seq.
export function sequenceOf(previous, next) {
  if (!next || next.seq == null) return null;
  if (!previous || previous.seq == null) return { kind: 'first', missed: 0 };
  if (next.seq === previous.seq) return { kind: 'repeat', missed: 0 };
  if (next.seq === previous.seq + 1) return { kind: 'next', missed: 0 };
  if (next.seq > previous.seq) return { kind: 'missed', missed: next.seq - previous.seq - 1 };
  return { kind: 'older', missed: 0 };
}

function nearestRank(sorted, p) {
  if (!sorted.length) return null;
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1))];
}

/// Figures over several objects (an arm): the mean weighted by each object's `interval.frames`, and the median and
/// 95th percentile (nearest rank) of the joined `frame_ms_list`s. Objects of the sample shape have no frame count or
/// list: their mean is the plain mean of their `frame_ms`, and the percentiles are null. `frames` is the frames
/// covered (null for the sample shape); `listFrames` how many frame times the percentiles stand on.
export function objectsFigures(records) {
  const usable = records.filter((record) => record && record.frameMs != null);
  if (!usable.length) return { shape: null, objects: 0, frames: null, meanMs: null, p50Ms: null, p95Ms: null, listFrames: 0 };
  const interval = usable.every((record) => record.shape === 'interval' && record.interval?.frames > 0);
  if (!interval) {
    const mean = usable.reduce((sum, record) => sum + record.frameMs, 0) / usable.length;
    return { shape: 'sample', objects: usable.length, frames: null, meanMs: mean, p50Ms: null, p95Ms: null, listFrames: 0 };
  }
  const frames = usable.reduce((sum, record) => sum + record.interval.frames, 0);
  const mean = usable.reduce((sum, record) => sum + record.frameMs * record.interval.frames, 0) / frames;
  const list = usable.flatMap((record) => record.frameMsList ?? []).sort((a, b) => a - b);
  return {
    shape: 'interval',
    objects: usable.length,
    frames,
    meanMs: mean,
    p50Ms: nearestRank(list, 0.5),
    p95Ms: nearestRank(list, 0.95),
    listFrames: list.length,
  };
}
