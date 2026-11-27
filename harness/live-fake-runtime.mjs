// A stand-in for the runtime's live side, for the tests of live-drive and the MCP server. It keeps the contract of
// `victory-gpu/src/live.rs`: the control file is read when it changes; a file with a non-comment line that has no
// `=` is ignored whole; only the names PRESENT in the file are applied (`apply`), so a name removed from the file
// keeps its last value; a value already in force is no change. A change closes the stretch that just ended (a
// segment line, its mean the mean of the stretch's frame times) and then logs one `live:` marker per name. Frame
// lines keep the log growing, and a stats file like `host/stats.rs` writes is rewritten atomically, the `live` map
// with the HUD switch left out while it is off. A change of `LIVE_MARK` alone closes the segment as a mark does
// (ended by a mark; the next begun at a mark, no frame left out as transition). `shape` picks the runtime's two shapes
// (stats-object.mjs): 'interval' (the default, as `world` writes it: every object, segment line and frame line in
// the shapes of fixtures/runtime-stats.mjs and runtime-lines.mjs, which are copied from the runtime's own tests) or
// 'sample' (the pre-rebuild build: one second's sample, the older segment line). It writes only at a present (a
// tick): `pause(ms)` presents nothing for that long (no frame line, no control read, no stats), and the next present
// carries the whole gap as one frame, so one object holds it, as the runtime's does.
//
// `makeSim` is the same contract on a virtual clock, for tests that need many seeds or the runtime's own timing
// (written from the round-2 check's sim-runtime.mjs, itself written from live.rs, host.rs and host/stats.rs): see its
// comment. Neither ever touches a game.
import fs from 'node:fs';
import path from 'node:path';

import { FRAME_LINE } from './fixtures/runtime-lines.mjs';
import { STATS_LIVE } from './fixtures/runtime-stats.mjs';
import { mergeControl, writeAtomic } from './live.mjs';
import { LIVE_NAMES, UNLISTED_WHEN_OFF, isLiveName, normalizeLiveValue } from './live-names.mjs';
import { virtualClock } from './virtual-clock.mjs';

// The tests drive the fake on it.
export { virtualClock };

// ----------------------------------------------------------------------------------------------------------------
// The runtime's shapes, written as `world` writes them

/// The switch that changes nothing (live.rs `Kind::Mark`): a new value only ends the segment under way.
export const MARK = 'LIVE_MARK';
const TRANSITION_FRAMES = 3;
const LIGHT_DRAWS = 1000;
const LONG_MS = 1000;
const BEGUN_TEXT = { start: 'the start of the run', change: 'a change', mark: 'a mark' };
const ENDED_TEXT = { change: 'a change', mark: 'a mark', end: 'the end of the run' };

const nearestOf = (sorted, percent) => sorted[Math.max(1, Math.ceil((sorted.length * percent) / 100)) - 1];

/// A segment's figures from its frames ({ms, draws}) as live.rs `Segment::frame` sorts them: each frame in one of
/// four, asked in this order: the first three after a change (transition), under 1000 draws (light), over 1000 ms
/// (long), or counted; the mean and nearest-rank p50 and p95 of the counted, each kind left out with its mean.
export function segmentFigures(frames, begun) {
  const kinds = { transition: [], light: [], long: [], counted: [] };
  for (const { ms, draws = 7000 } of frames) {
    const seen = kinds.transition.length + kinds.light.length + kinds.long.length + kinds.counted.length;
    if (begun === 'change' && seen < TRANSITION_FRAMES) kinds.transition.push(ms);
    else if (draws < LIGHT_DRAWS) kinds.light.push(ms);
    else if (ms > LONG_MS) kinds.long.push(ms);
    else kinds.counted.push(ms);
  }
  const mean = (list) => (list.length ? list.reduce((a, b) => a + b, 0) / list.length : 0);
  const sorted = [...kinds.counted].sort((a, b) => a - b);
  return {
    counted: kinds.counted.length,
    transition: [kinds.transition.length, mean(kinds.transition)],
    light: [kinds.light.length, mean(kinds.light)],
    long: [kinds.long.length, mean(kinds.long)],
    mean: mean(kinds.counted),
    p50: sorted.length ? nearestOf(sorted, 50) : 0,
    p95: sorted.length ? nearestOf(sorted, 95) : 0,
  };
}

/// live.rs `segment_line`: a closed segment (`ended` 'change', 'mark' or 'end') or the one under way (`ended` null).
export function segmentLine({ k, first, last, figures: f, begun, ended = null, skipped = [0, 0], inForce }) {
  return (
    `live segment ${k}${ended ? '' : ' open'}: frames ${first}..${last}, ${f.counted} counted, ${f.transition[0]} excluded as ` +
    `transition (the first ${TRANSITION_FRAMES} after a change; their mean ${f.transition[1].toFixed(1)}), ${f.light[0]} excluded ` +
    `as light (under ${LIGHT_DRAWS} draws: a load, a menu or a pause, the runtime cannot tell which; their mean ` +
    `${f.light[1].toFixed(1)}), ${f.long[0]} excluded as long (over ${LONG_MS} ms; their mean ${f.long[1].toFixed(1)}), mean ` +
    `${f.mean.toFixed(1)}, p50 ${f.p50.toFixed(1)}, p95 ${f.p95.toFixed(1)} of the counted, begun at ${BEGUN_TEXT[begun]}, ` +
    `${ended ? `ended by ${ENDED_TEXT[ended]}` : 'not ended (the segment under way, as of this line)'}, skipped ${skipped[0]} ` +
    `draws and ${skipped[1]} dispatches, in force: ${inForce}`
  );
}

/// live.rs `segment_json`: the stats object's `segment_open`.
export function segmentOpenObject({ k, first, last, figures: f, begun }) {
  const two = (x) => Number(x.toFixed(2));
  return {
    segment: k,
    first_frame: first,
    last_frame: last,
    counted: f.counted,
    transition_excluded: f.transition[0],
    transition_mean_ms: two(f.transition[1]),
    light_excluded: f.light[0],
    light_mean_ms: two(f.light[1]),
    long_excluded: f.long[0],
    long_mean_ms: two(f.long[1]),
    mean_ms: two(f.mean),
    p50_ms: two(f.p50),
    p95_ms: two(f.p95),
    begun,
  };
}

/// A stats object of the runtime's shape (fixtures/runtime-stats.mjs STATS_LIVE: every key, in its order) over the
/// frames `times` (ms), as a build without diagnostics writes it (`budget`, `other_threads` and `gpu_stamps` null).
export function statsObjectOver({ seq, tMs, unixMs, firstFrame, times, live, closed, open, draws = 1500 }) {
  const object = JSON.parse(STATS_LIVE);
  const frames = times.length;
  const wall = times.reduce((a, b) => a + b, 0);
  const sorted = [...times].sort((a, b) => a - b);
  const two = (x) => Number(x.toFixed(2));
  Object.assign(object, {
    seq,
    t_ms: Number(tMs.toFixed(1)),
    unix_ms: unixMs,
    interval: { first_frame: firstFrame, last_frame: firstFrame + frames - 1, frames, wall_ms: two(wall) },
    frame: firstFrame + frames - 1,
    fps: two(wall > 0 ? (frames * 1000) / wall : 0),
    frame_ms: two(wall / Math.max(1, frames)),
    frame_ms_p50: two(nearestOf(sorted, 50)),
    frame_ms_worst: two(sorted.at(-1)),
    frame_ms_list: times.slice(0, 240).map(two),
    frame_ms_list_missing: Math.max(0, frames - 240),
    budget: null,
    other_threads: null,
    gpu_stamps: null,
    draws,
    dispatched: 20,
    dropped: 0,
    pipelines_created_total: 10,
    pipelines_per_minute: 0,
    vram_mb: 1000,
    live,
    segments_closed_total: closed,
    segment_open: open,
  });
  return object;
}

/// stats.rs `frame_line` (FRAME_LINE) for `frames` frames up to `last` in `wallMs`, with their p50 and p95.
export function frameLineOver({ first, last, wallMs, p50, p95 }) {
  const frames = last - first + 1;
  const fps = wallMs > 0 ? (frames * 1000) / wallMs : 0;
  return FRAME_LINE.replace('frame 6, 3.0 fps, 333.3 ms/frame', `frame ${last}, ${fps.toFixed(1)} fps, ${(wallMs / Math.max(1, frames)).toFixed(1)} ms/frame`).replace(
    'means over frames 1..6 (6 frames in 2000 ms, p50 250.0, p95 500.0 ms/frame)',
    `means over frames ${first}..${last} (${frames} frames in ${wallMs.toFixed(0)} ms, p50 ${p50.toFixed(1)}, p95 ${p95.toFixed(1)} ms/frame)`,
  );
}

/// What ended a segment when `changes` ([name, old, new]) applied: a mark, if nothing but the mark changed.
const endedBy = (changes) => (changes.every(([name]) => name === MARK) ? 'mark' : 'change');

/// The start values a run with an empty environment has (`live.rs`, `SPECS`).
export const DEFAULT_START = Object.freeze({
  VICTORY_KEEP_TEXTURES: 'on',
  VICTORY_TEXTURE_LOW: '70',
  VICTORY_TEXTURE_HIGH: '85',
  VICTORY_LATE_WAITS: 'off',
  LIVE_SKIP_DISPATCH: '',
  LIVE_SKIP_DRAW: '',
  LIVE_KEYS: '',
  LIVE_MARK: '0',
  VICTORY_LAND_SCAN: 'off',
  VICTORY_EXACT_BACK: 'off',
  VICTORY_UPLOAD_POOL: 'off',
  VICTORY_TEXTURE_RECHECK: 'off',
  VICTORY_FETCH_QUEUE: 'off',
  VICTORY_LIVE_HUD: 'off',
  VICTORY_TEXTURE_MADE: 'off',
});

/// The assignments a control file's text asks for, or null while the runtime would ignore the whole file.
export function runtimeParse(text) {
  const given = [];
  for (const raw of String(text).split(/\r?\n/)) {
    const line = raw.split('#')[0].trim();
    if (!line) continue;
    const eq = line.indexOf('=');
    if (eq < 0) return null;
    const name = line.slice(0, eq).trim();
    if (!name) return null;
    if (!isLiveName(name)) continue;
    const value = normalizeLiveValue(name, line.slice(eq + 1));
    if (value !== null) given.push([name, value]);
  }
  return given;
}

/// The registry's order (live.rs `SPECS`, alphabetical: a runtime test says so), which the `live` map and the `in force`
/// list follow; LIVE_NAMES's own order means nothing.
export const REGISTRY_ORDER = Object.freeze([...LIVE_NAMES].sort());

/// The `live` map as the stats file writes it.
export function listedLive(state) {
  const out = {};
  for (const name of REGISTRY_ORDER) {
    if (name === UNLISTED_WHEN_OFF && state[name] === 'off') continue;
    out[name] = state[name];
  }
  return out;
}


/// Starts the fake, on a virtual clock (`clock`, else a new one; the handle returns it): its presents run only as the
/// clock is moved on, so a runner must be given the clock's `now` and `sleep`. Options:
/// - `start`: start values over DEFAULT_START; `baseline`: the plan's baseline, which with `start` is the state
///   whose frames take `baselineMean` (any other state takes `stepMean`); `meanOf(state, tick)` overrides both.
/// - `answer: false`: never read the control file; `ignoreAfterChanges: n`: stop reading it after n reads that
///   changed something, frames and stats going on (a runtime that stopped applying the file).
/// - `applyAs: {NAME: (value) => value}`: the value the runtime applies instead of the one asked.
/// - `frames: false`: no frame lines; `stopAfterChanges: n`: after n reads that changed something everything stops
///   (log and stats). `framesPerTick`: frames presented per tick.
/// - `failureLine`, `prelude`: a line logged on the first tick, text written before it.
/// - `shape`: 'interval' or 'sample'; `seqStep(tick, state)`: how far seq moves at a write (over 1: objects missed).
/// - `stats: false`: no stats file; `statsLive: false`: the stats object has no `live` map; `statsEvery`: ticks
///   between stats writes; `statsAt`: where it is written; `statsAge`: the first write's age in ms (a stale file);
///   `statsLagMs`: the `live` map shows a change only this long after it was applied; `statsFrozen(tick, state)`:
///   true while the stats frame stops advancing (the older shape still rewrites the object; the interval shape writes
///   none, as nothing presents); `statsGarbage(tick, state)`: true
///   while the stats file is written unparsable; `statsPaused(tick, state)`: true while it is not written at all.
export function fakeRuntime(t, options) {
  const {
    log,
    control,
    statsAt = null,
    start = {},
    baseline = {},
    answer = true,
    ignoreAfterChanges = Infinity,
    frames = true,
    stopAfterChanges = Infinity,
    failureLine = null,
    prelude = '',
    baselineMean = 20,
    stepMean = 10,
    meanOf = null,
    stats = true,
    statsLive = true,
    statsEvery = 1,
    statsAge = 0,
    statsLagMs = 0,
    statsFrozen = () => false,
    statsGarbage = () => false,
    statsPaused = () => false,
    applyAs = {},
    shape = 'interval',
    seqStep = () => 1,
    framesPerTick = 10,
    tickMs = 10,
    clock = virtualClock(),
  } = options;
  const now = clock.now;
  fs.writeFileSync(log, `fake title log\n${prelude}`);
  let state = { ...DEFAULT_START };
  for (const [name, value] of Object.entries(start)) state[name] = normalizeLiveValue(name, value);
  const reference = { ...state };
  for (const [name, value] of Object.entries(baseline)) reference[name] = normalizeLiveValue(name, value);
  const frameMs = (current, tick) => {
    if (meanOf) return meanOf(current, tick);
    return LIVE_NAMES.every((name) => name === MARK || current[name] === reference[name]) ? baselineMean : stepMean;
  };
  // What the stats file shows: the state, `statsLagMs` behind.
  const history = [{ at: now(), state: { ...state } }];
  const shown = () => {
    const cutoff = now() - statsLagMs;
    let pick = history[0].state;
    for (const entry of history) if (entry.at <= cutoff) pick = entry.state;
    return pick;
  };
  let lastText = null;
  let segment = 0;
  let begun = 'start';
  let segmentStart = 1;
  let frame = 1;
  let statsFrame = 1;
  let lastStatsFrame = 0;
  let seq = 0;
  let firstWrite = true;
  const startedAt = now();
  let stretch = [];
  let applied = 0;
  let tick = 0;
  let stopped = false;
  let failureWritten = false;
  // A pause: no present until `pausedUntil`; the gap frames waiting for the next object (`gaps`, their times).
  let pausedUntil = 0;
  let pausedFrom = 0;
  let gaps = [];
  const writeStats = () => {
    if (!stats || !statsAt || statsPaused(tick, state)) return;
    if (statsGarbage(tick, state)) {
      writeAtomic(statsAt, `{"frame": ${tick}, "fps": `);
      return;
    }
    const frozen = statsFrozen(tick, state);
    if (shape === 'interval' && frozen) return;
    if (!frozen) statsFrame = Math.max(1, frame - 1);
    const steady = frameMs(state, tick);
    const covered = Math.max(1, statsFrame - lastStatsFrame);
    const gapList = gaps.slice(0, covered);
    gaps = [];
    const times = [...gapList, ...Array.from({ length: covered - gapList.length }, () => steady)];
    const wall = times.reduce((sum, value) => sum + value, 0);
    const ms = gapList.length ? wall / covered : steady;
    let object;
    if (shape === 'sample') {
      lastStatsFrame = statsFrame;
      object = {
        frame: statsFrame,
        fps: 1000 / ms,
        frame_ms: ms,
        parts: { host_wait_ms: 1 },
        draws: 1500,
        dispatches: 20,
        written: tick,
        segment,
      };
      if (statsLive) object.live = listedLive(shown());
    } else {
      seq += seqStep(tick, state);
      const at = now();
      const figures = segmentFigures(stretch.map((ms) => ({ ms })), begun);
      object = statsObjectOver({
        seq,
        tMs: at - startedAt,
        unixMs: firstWrite ? at - statsAge : at,
        firstFrame: lastStatsFrame + 1,
        times,
        live: statsLive ? listedLive(shown()) : null,
        closed: segment,
        open: segmentOpenObject({ k: segment + 1, first: segmentStart, last: statsFrame, figures, begun }),
      });
      if (!statsLive) delete object.live;
      lastStatsFrame = statsFrame;
    }
    firstWrite = false;
    writeAtomic(statsAt, `${JSON.stringify(object)}\n`);
  };
  writeStats();
  if (stats && statsAt && statsAge) {
    const old = new Date(Date.now() - statsAge);
    fs.utimesSync(statsAt, old, old);
  }
  /// A segment line of the shape: closed (`ended`: 'change', 'mark' or 'end') or open (`ended` null).
  const segmentText = (k, first, last, inForce, ended) => {
    if (shape === 'sample') {
      const mean = stretch.reduce((sum, value) => sum + value, 0) / Math.max(1, stretch.length);
      return (
        `live segment ${k}: frames ${first}..${last}, mean ${mean.toFixed(1)}, p50 ${mean.toFixed(1)}, ` +
        `p95 ${(mean + 2).toFixed(1)}, skipped 0 draws and 0 dispatches, in force: ${inForce}`
      );
    }
    const figures = segmentFigures(stretch.map((ms) => ({ ms })), begun);
    return segmentLine({ k, first, last, figures, begun, ended, inForce });
  };
  const inForceText = (values) => REGISTRY_ORDER.filter((name) => name !== UNLISTED_WHEN_OFF || values[name] !== 'off').map((name) => `${name}=${values[name]}`).join(' ');
  const readControl = () => {
    try {
      return fs.readFileSync(control, 'utf8');
    } catch {
      return '';
    }
  };
  lastText = readControl();
  {
    const given = runtimeParse(lastText);
    for (const [name, value] of given ?? []) state[name] = value;
    history.push({ at: now() - statsLagMs, state: { ...state } });
  }
  const cancel = clock.every(tickMs, () => {
    if (stopped) return;
    // Nothing presents during a pause: no frame, no line, no control read, no stats.
    if (now() < pausedUntil) return;
    tick += 1;
    if (failureLine && !failureWritten) {
      fs.appendFileSync(log, `${failureLine}\n`);
      failureWritten = true;
    }
    if (pausedFrom) {
      // The first present after the pause: one frame whose time is the whole gap.
      const gap = now() - pausedFrom;
      pausedFrom = 0;
      gaps.push(gap);
      stretch.push(gap);
      frame += 1;
    }
    const tickMsNow = frameMs(state, tick);
    for (let i = 0; i < framesPerTick; i += 1) stretch.push(tickMsNow);
    frame += framesPerTick;
    if (frames && shape === 'sample') fs.appendFileSync(log, `frame ${frame - 1}, 4.0 fps, 250.0 ms/frame\n`);
    if (frames && shape === 'interval') {
      fs.appendFileSync(log, `${frameLineOver({ first: frame - framesPerTick, last: frame - 1, wallMs: tickMsNow * framesPerTick, p50: tickMsNow, p95: tickMsNow })}\n`);
      if (tick % 5 === 0 && stretch.length) fs.appendFileSync(log, `${segmentText(segment + 1, segmentStart, frame - 1, inForceText(state), null)}\n`);
    }
    if (tick % statsEvery === 0) writeStats();
    if (!answer || applied >= ignoreAfterChanges) return;
    const text = readControl();
    if (text === lastText) return;
    lastText = text;
    const given = runtimeParse(text);
    if (!given) return;
    const changes = [];
    const next = { ...state };
    for (const [name, asked] of given) {
      const value = applyAs[name] ? applyAs[name](asked) : asked;
      if (next[name] !== value) {
        changes.push([name, next[name], value]);
        next[name] = value;
      }
    }
    if (changes.length) {
      applied += 1;
      segment += 1;
      const last = frame - 1;
      const ended = endedBy(changes);
      fs.appendFileSync(log, `${segmentText(segment, segmentStart, last, inForceText(state), ended)}\n`);
      segmentStart = frame;
      begun = ended;
      stretch = [];
      for (const [name, old, value] of changes) {
        // At the frame whose present applied it: the last of the segment it closed (live.rs `poll`).
        fs.appendFileSync(log, `live: ${name} ${old} -> ${value} at frame ${last}\n`);
      }
      state = next;
      history.push({ at: now(), state: { ...state } });
    }
    if (applied >= stopAfterChanges) stopped = true;
  });
  /// A switch flipped by hand: the control file edited by someone other than the runner.
  const handSet = (name, value) => {
    let text = '';
    try {
      text = fs.readFileSync(control, 'utf8');
    } catch {
      // No file yet.
    }
    writeAtomic(control, mergeControl(text, { set: { [name]: value } }));
  };
  const stop = () => {
    stopped = true;
    cancel();
  };
  t.after(stop);
  /// A load or pause: nothing presents for `ms`; the next present's frame holds the whole gap.
  const pause = (ms) => {
    pausedFrom = now();
    pausedUntil = pausedFrom + ms;
  };
  return {
    clock,
    stop,
    handSet,
    pause,
    state: () => ({ ...state }),
    segmentCount: () => segment,
    /// The segment line the runtime writes at shutdown, for the stretch still open.
    shutdown() {
      stop();
      fs.appendFileSync(log, `${segmentText(segment + 1, segmentStart, frame - 1, inForceText(state), 'end')}\n`);
    },
  };
}

// ----------------------------------------------------------------------------------------------------------------
// The simulated title on a virtual clock

/// A seeded generator (mulberry32) and a standard normal draw from it, so a test's seeds give the same frames.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gauss(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/// Frame-time models for makeSim. `noise`: 'none', 'low' (3% lognormal jitter) or 'real', fitted by the round-2 check
/// to a run log's frame lines (one-second windows with a median near 230 ms and a 95th percentile over twice that):
/// 12% lognormal jitter plus a hitch on `hitchP` of frames, exponential with mean `hitchMean` ms. `effect(ctx)` is
/// the switch's true effect as a fraction (-0.1 is 10% faster).
export function frameModel({ base = 230, effect = () => 0, noise = 'real', hitchP = 0.03, hitchMean = 500 } = {}) {
  return (ctx) => {
    const mean = base * (1 + effect(ctx));
    if (noise === 'none') return mean;
    if (noise === 'low') return mean * Math.exp(0.03 * gauss(ctx.rng));
    let ms = mean * Math.exp(0.12 * gauss(ctx.rng) - 0.0072);
    if (ctx.rng() < hitchP) ms += -hitchMean * Math.log(1 - ctx.rng());
    return ms;
  };
}

const nearest = (sorted, p) => sorted[Math.max(0, Math.ceil(p * sorted.length) - 1)];

/// A title that keeps the runtime's live contract on a virtual clock (`sim.t`, ms), advanced by `advanceTo(t)`:
/// - live.rs: the control file is looked at only at a present, at most once per 500 ms, and read when it changed (by
///   its text: see `poll`); parsed as runtimeParse; only the names present are applied; a reading that changes something
///   closes the segment (its line carries the values in force during it; ended by a mark when only `LIVE_MARK`
///   changed), then logs one `live:` marker per name, at that frame.
/// - host.rs and stats.rs: the frame line every 5 s or more. `shape` 'interval' (the default, `world`'s shapes): a
///   stats object at the first present a second or more after the previous one, covering every frame since
///   (statsObjectOver: the fixture's keys in their order), the segment line with transition, light and long frames
///   left out (segmentLine), and an open line after each frame line. 'sample' (the pre-rebuild build): a one-second
///   window's figures, written with the frame line, so one sample about every 5 to 6 s, and the older segment line.
/// - No present, no anything: `frameMs` returning {pause: ms} presents nothing for that long; the present after it
///   carries the gap, as the runtime's `now - previous` does.
/// Options: dir, name, start (values over DEFAULT_START), frameMs(ctx) with ctx {values, t, since (ms since each
/// name's last change), rng, frame, sim}, events [{at, do(sim)}], seed, readsControl, statsMode(t) -> 'ok' | 'skip' |
/// 'torn'. `sim.samples` lists every stats write {t, frame, frame_ms}; `sim.markers` every change applied.
export function makeSim(options) {
  const { dir, name = 'sim', start = {}, frameMs, events = [], seed = 1, readsControl = true, shape = 'interval' } = options;
  const statsMode = options.statsMode ?? (() => 'ok');
  fs.mkdirSync(dir, { recursive: true });
  const log = path.join(dir, `${name}.log`);
  const control = path.join(dir, `${name}.control`);
  const stats = path.join(dir, `${name}.stats.json`);
  fs.writeFileSync(log, 'sim title log\n');
  const values = { ...DEFAULT_START };
  for (const [key, value] of Object.entries(start)) values[key] = normalizeLiveValue(key, value);
  const sim = {
    log,
    control,
    stats,
    t: 0,
    frame: 0,
    values,
    rng: mulberry32(seed),
    watcher: { text: null, checked: null },
    seg: { first: null, last: 0, times: [], closed: 0, begun: 'start' },
    line: { since: 0, frames: 0, logged: 0, times: [] },
    object: { since: 0, first: 1, times: [], seq: 0 },
    changedAt: {},
    markers: [],
    samples: [],
    gap: 0,
  };
  const append = (text) => fs.appendFileSync(log, text);
  const inForce = () =>
    REGISTRY_ORDER.filter((key) => key !== UNLISTED_WHEN_OFF || sim.values[key] !== 'off')
      .map((key) => `${key}=${sim.values[key]}`)
      .join(' ');
  const figuresNow = () => segmentFigures(sim.seg.times.map((ms) => ({ ms })), sim.seg.begun);
  const lineOf = (k, open, ended) => {
    const s = sim.seg;
    const all = s.times;
    if (shape === 'sample') {
      const sorted = [...all].sort((a, b) => a - b);
      const mean = all.reduce((x, y) => x + y, 0) / all.length;
      return (
        `live segment ${k}: frames ${s.first}..${s.last}, mean ${mean.toFixed(1)}, p50 ${nearest(sorted, 0.5).toFixed(1)}, ` +
        `p95 ${nearest(sorted, 0.95).toFixed(1)}, skipped 0 draws and 0 dispatches, in force: ${inForce()}`
      );
    }
    return segmentLine({ k, first: s.first, last: s.last, figures: figuresNow(), begun: s.begun, ended: open ? null : ended, inForce: inForce() });
  };
  function close(ended) {
    const s = sim.seg;
    if (s.first !== null && s.times.length) {
      s.closed += 1;
      append(`INFO victory_gpu::live: ${lineOf(s.closed, false, ended)}\n`);
    }
    s.first = null;
    s.times = [];
    s.begun = ended === 'mark' ? 'mark' : ended === 'end' ? 'start' : 'change';
  }
  function poll(frame) {
    if (!readsControl) return;
    const w = sim.watcher;
    if (w.checked !== null && sim.t - w.checked < 500) return;
    w.checked = sim.t;
    // The runtime reads the file when its modified time moves. On the virtual clock a minute passes in a few real
    // milliseconds, inside one step of the file system's coarse timestamp on a slow machine, so two writes the
    // runtime would see at two times can carry one here (the hosted runner's failure: a write never read). The sim
    // reads the file at every look and acts when its text moved, which is the runtime's rule whenever writes are a
    // real second or more apart, as a runner's and a person's are.
    let text;
    try {
      text = fs.readFileSync(control, 'utf8');
    } catch {
      return;
    }
    if (w.text === text) return;
    w.text = text;
    const given = runtimeParse(text);
    if (given === null) {
      w.text = null;
      return;
    }
    const next = { ...sim.values };
    for (const [key, value] of given) next[key] = value;
    const changes = LIVE_NAMES.filter((key) => next[key] !== sim.values[key]).map((key) => [key, sim.values[key], next[key]]);
    if (!changes.length) return;
    close(endedBy(changes));
    for (const [key, old, value] of changes) {
      sim.values[key] = value;
      sim.changedAt[key] = sim.t;
      sim.markers.push({ t: sim.t, name: key, old, new: value, frame });
      append(`INFO victory_gpu::live: live: ${key} ${old} -> ${value} at frame ${frame}\n`);
    }
  }
  function write(object) {
    const mode = statsMode(sim.t);
    if (mode === 'skip') return;
    const text = JSON.stringify(object);
    writeAtomic(stats, mode === 'torn' ? text.slice(0, Math.floor(text.length / 2)) : text);
    sim.samples.push({ t: sim.t, frame: sim.frame, frame_ms: object.frame_ms, mode });
  }
  function present() {
    for (const e of events) {
      if (!e.done && sim.t >= e.at) {
        e.done = true;
        e.do(sim);
      }
    }
    const since = Object.fromEntries(Object.entries(sim.changedAt).map(([key, at]) => [key, sim.t - at]));
    const got = frameMs({ values: sim.values, t: sim.t, since, rng: sim.rng, frame: sim.frame, sim });
    if (got && typeof got === 'object') {
      sim.t += got.pause;
      sim.gap += got.pause;
      return;
    }
    sim.t += got;
    const dt = got + sim.gap;
    sim.gap = 0;
    sim.frame += 1;
    const s = sim.seg;
    s.last = sim.frame;
    if (s.first === null) s.first = sim.frame;
    s.times.push(dt);
    poll(sim.frame);
    const o = sim.object;
    o.times.push(dt);
    const l = sim.line;
    l.frames += 1;
    if (shape === 'interval') l.times.push(dt);
    if (shape === 'interval') {
      if (sim.t - o.since >= 1000) {
        o.seq += 1;
        write(
          statsObjectOver({
            seq: o.seq,
            tMs: sim.t,
            unixMs: Date.now(),
            firstFrame: o.first,
            times: o.times,
            live: listedLive(sim.values),
            closed: sim.seg.closed,
            open: segmentOpenObject({ k: sim.seg.closed + 1, first: sim.seg.first, last: sim.seg.last, figures: figuresNow(), begun: sim.seg.begun }),
          }),
        );
        o.since = sim.t;
        o.first = sim.frame + 1;
        o.times = [];
      }
      if (sim.t - l.logged >= 5000) {
        l.logged = sim.t;
        const line = sim.line;
        const times = line.times.splice(0);
        const sorted = [...times].sort((x, y) => x - y);
        append(
          `INFO victory_gpu::host: ${frameLineOver({ first: sim.frame - times.length + 1, last: sim.frame, wallMs: times.reduce((x, y) => x + y, 0), p50: nearestOf(sorted, 50), p95: nearestOf(sorted, 95) })}\n`,
        );
        if (sim.seg.first !== null) append(`INFO victory_gpu::live: ${lineOf(sim.seg.closed + 1, true)}\n`);
      }
      return;
    }
    const elapsed = sim.t - l.since;
    if (elapsed < 1000) return;
    const fps = l.frames / (elapsed / 1000);
    const ms = elapsed / l.frames;
    l.since = sim.t;
    l.frames = 0;
    if (sim.t - l.logged >= 5000) {
      l.logged = sim.t;
      append(`INFO victory_gpu::host: frame ${sim.frame}, ${fps.toFixed(1)} fps, ${ms.toFixed(1)} ms/frame\n`);
      write({
        frame: sim.frame,
        fps: Number(fps.toFixed(2)),
        frame_ms: Number(ms.toFixed(2)),
        parts: { host_wait_ms: 2 },
        draws: 1500.0,
        dispatches: 10,
        dropped: 0,
        live: listedLive(sim.values),
        segment: sim.seg.closed,
      });
    }
  }
  sim.advanceTo = (t) => {
    while (sim.t < t) present();
  };
  sim.handSet = (set) =>
    writeAtomic(control, mergeControl(fs.existsSync(control) ? fs.readFileSync(control, 'utf8') : '', { set }));
  sim.shutdown = () => close('end');
  return sim;
}

/// A scheduler that lets runners share a sim's clock: `sleep(ms)` parks the caller and the pump advances the title to
/// the earliest wake. A runner made with `killAt` is dropped for good at that time (a process killed with no handler
/// run: its promise never settles) and its `killed` promise resolves.
export function makeClock(sim) {
  const waiting = [];
  let pumping = false;
  const kick = () => {
    if (!pumping && waiting.length) {
      pumping = true;
      setImmediate(pump);
    }
  };
  function pump() {
    pumping = false;
    if (!waiting.length) return;
    waiting.sort((x, y) => x.at - y.at);
    const next = waiting.shift();
    sim.advanceTo(next.at);
    if (next.runner.killAt != null && sim.t >= next.runner.killAt) {
      next.runner.dead = true;
      next.runner.onKilled();
      kick();
      return;
    }
    next.resolve();
    kick();
  }
  const runner = ({ killAt = null } = {}) => {
    const r = { killAt, dead: false };
    r.killed = new Promise((resolve) => {
      r.onKilled = resolve;
    });
    r.sleep = (ms) =>
      new Promise((resolve) => {
        if (r.dead) return;
        waiting.push({ at: sim.t + ms, resolve, runner: r });
        kick();
      });
    r.now = () => sim.t;
    return r;
  };
  return { runner };
}
