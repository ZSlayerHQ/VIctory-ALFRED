#!/usr/bin/env node
// Ends a VIctory world run N seconds after its scene is settled.
//
// What "settled" requires: three frame lines in a row (the runtime writes one about every 5 to 6 s), each with
//   - at least 1000 draws, dropped draws included;
//   - a frame rate at or above --floor-fps, 1.5 by default (cutscene loading runs near 0.5 fps);
//   - quiet pipelines, on a build that reports them: "0 being made" on the frame line, and the recording threads'
//     pipeline wait under --pipeline-quiet-ms (5) on its last three samples (a build that prints neither is not
//     gated, and the settle line says "pipeline quiet not checked");
//   - at most --waiting K textures waiting, only when --waiting is given ("textures picked by index: ... , K
//     waiting" is otherwise only reported: on the 2026-11-25 runs it never fell below a few hundred in play);
//   - at least --settle-after-s N seconds after the first heavy line, when given;
// and the three lines' frame rates steady: (highest - lowest) / mean at most --max-fps-spread, 0.25 by default
// (`off` drops it). The window slides, so a run of good lines settles at the first three that are steady. Replayed
// over the 150 newest runs of 2026-11-25 and 2026-11-26, 125 of the 126 that settled by the rule without it still
// settle, 80 at the same line, the median at the same line and the latest 79 s later; the one that no longer does
// "settled" 12 s in at a 49% spread.
// Named limit: it cannot tell the world from a cutscene: a heavy, steady cutscene settles.
//
//   node harness/settled-watch.mjs <log> <seconds> [--timeout 600] [--floor-fps 1.5] [--waiting K] [--poll-ms 250]
//                                  [--pipeline-quiet-ms 5] [--max-fps-spread 0.25|off] [--after-file <path>]
//
// It appends "harness: settled at <s> s (waiting K, <fps> fps); waited: …" when that happens, the tail naming
// what it waited for and the seconds each took, creates --after-file (the
// runtime's VICTORY_KEYS_AFTER sentinel, which starts the relative key entries), and N seconds later ends this
// run's victory.exe and appends "harness: ended by --settled-seconds after N s of settled scene". A scene that
// has not settled within --timeout seconds of the first heavy line ends the run with
// "harness: ended by --settled-timeout after N s: the scene never settled" — the only end for a build that keeps a
// pipeline being made or never quiets its wait; such a run has no settle line, so compare-runs measures heavy frames
// and its pipeline-wait row marks the window compiling. While unsettled, a line every minute (UNSETTLED_NOTE_MS)
// names the condition still unmet, so the run explains itself in its own log. A game that exits by itself is not
// touched, as in heavy-watch.mjs.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { endSelected } from './end-run.mjs';

import { stripAnsi } from './parse-run.mjs';
import { parsePipelineWait } from './run-lines.mjs';

export const DEFAULT_POLL_MS = 250;
export const DEFAULT_FLOOR_FPS = 1.5;
export const DEFAULT_WAITING = null;
export const SETTLED_LINES = 3;
export const DEFAULT_TIMEOUT_S = 600;
// A run that never reaches the world never has a heavy line, so the settled timeout above never starts. This
// second limit runs from the watcher's start and stops applying at the first heavy line (2026-11-27: a run sat
// on the game's first-boot screen for minutes).
export const DEFAULT_WORLD_TIMEOUT_S = 600;
// A scene on its first visit builds pipelines while it draws: the recording threads wait hundreds of ms a frame
// for them, and the frame rate and texture queue can both look settled through it (the boat run, 09:12:56). A settle
// also needs the pipeline work quiet: the last sample's wait under this, and none being made.
export const PIPELINE_QUIET_MS = 5;
// A scene still streaming in swings its frame rate while every other test passes: the settled lines' rates must
// be within this fraction of their mean, (highest - lowest) / mean. See the header for what it costs today's runs.
export const MAX_FPS_SPREAD = 0.25;

/// (highest - lowest) / mean of the frame rates, or null for none.
export function fpsSpread(rates) {
  if (!rates.length) return null;
  const mean = rates.reduce((sum, fps) => sum + fps, 0) / rates.length;
  return mean > 0 ? (Math.max(...rates) - Math.min(...rates)) / mean : null;
}

const percent = (fraction) => `${Math.round(fraction * 100)}%`;
// An unsettled run says why once a minute (see `unmetText`).
export const UNSETTLED_NOTE_MS = 60_000;

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FRAME_RE =
  /^(\S+Z)\s+.*?([\d.]+) fps, ([\d.]+) ms\/frame.*?([\d.]+) draws\/frame, dropped ([\d.]+)(?: without a host pipeline \((\d+) being made\))?/;
const WAITING_RE = /textures picked by index: \d+ uploaded so far \(\d+ MB\), (\d+) waiting/;

/// One frame line's figures, or null when the line is not the frame line. `making` is null on a build that does not
/// say how many pipelines are being made.
export function frameLine(raw) {
  const line = stripAnsi(String(raw));
  const m = line.match(FRAME_RE);
  if (!m) return null;
  const at = Date.parse(m[1]);
  if (Number.isNaN(at)) return null;
  const waiting = line.match(WAITING_RE);
  return {
    at,
    fps: Number(m[2]),
    draws: Number(m[4]),
    dropped: Number(m[5]),
    making: m[6] === undefined ? null : Number(m[6]),
    waiting: waiting ? Number(waiting[1]) : null,
  };
}

/// The recording threads' wait for pipelines, from the copystat line:
/// "recording per frame (105 frames): … ms; 0.0 ms waiting for pipelines; …". Null for any other line.
export function pipelineLine(raw) {
  const line = stripAnsi(String(raw));
  const ms = parsePipelineWait(line);
  if (ms === null) return null;
  const at = Date.parse((line.match(/^(\S+Z)/) || [])[1] ?? '');
  return Number.isNaN(at) ? null : { at, ms };
}

/// The steadiness bound an option set asks for: MAX_FPS_SPREAD unless given, and null (off) when given as null.
const spreadOption = (options) => (options.maxFpsSpread === undefined ? MAX_FPS_SPREAD : options.maxFpsSpread);

/// A fresh state: no heavy line seen, no run of settled lines, no settle.
export function initialState() {
  return {
    heavyAt: null,
    consecutive: 0,
    settledAt: null,
    settledWaiting: null,
    settledFps: null,
    makingSeen: false,
    pipelineSeen: false,
    waitMs: null,
    pipelineRun: 0,
    metFpsAt: null,
    metPipelinesAt: null,
    metTexturesAt: null,
    settledDetail: null,
    lastFps: null,
    lastDraws: null,
    lastWaiting: null,
    lastMaking: null,
    // The frame rates of the current run of lines meeting the other tests, newest last, at most SETTLED_LINES.
    runFps: [],
    settledSpread: null,
  };
}

/// The state after one more line: the first heavy line's time, the run of consecutive frame lines meeting the rule,
/// and the settle time once enough of them have. A `waiting` ceiling is optional; without one the rate is the test,
/// and the count is only reported.
///
/// When the build prints either the recording threads' pipeline wait or how many pipelines are being made, the
/// settle also waits for the pipeline work to be quiet: "being made" at 0 and the wait's run of consecutive
/// samples under `pipelineQuietMs` as long as the frame run. A build that prints neither settles as it always did.
export function observe(state, raw, options = {}) {
  const floorFps = options.floorFps ?? DEFAULT_FLOOR_FPS;
  const waiting = options.waiting ?? DEFAULT_WAITING;
  const quietMs = options.pipelineQuietMs ?? PIPELINE_QUIET_MS;
  const settleAfterS = options.settleAfterS ?? null;
  const maxSpread = spreadOption(options);
  const next = { ...state };

  const pipeline = pipelineLine(raw);
  if (pipeline) {
    next.pipelineSeen = true;
    next.waitMs = pipeline.ms;
    next.pipelineRun = pipeline.ms < quietMs ? next.pipelineRun + 1 : 0;
    if (next.metPipelinesAt === null && next.pipelineRun >= SETTLED_LINES) next.metPipelinesAt = pipeline.at;
    return next;
  }

  const frame = frameLine(raw);
  if (!frame) return next;
  next.lastFps = frame.fps;
  next.lastDraws = frame.draws;
  next.lastWaiting = frame.waiting;
  if (frame.making !== null) {
    next.makingSeen = true;
    next.lastMaking = frame.making;
  }
  const heavy = frame.draws + frame.dropped >= 1000;
  if (heavy && next.heavyAt === null) next.heavyAt = frame.at;
  if (heavy && frame.fps >= floorFps && next.metFpsAt === null) next.metFpsAt = frame.at;
  if (heavy && waiting !== null && frame.waiting !== null && frame.waiting <= waiting && next.metTexturesAt === null) {
    next.metTexturesAt = frame.at;
  }
  const belowCeiling = waiting === null || (frame.waiting !== null && frame.waiting <= waiting);
  // The pipeline part: a build that says how many are being made must say 0; one that reports the recording wait
  // must have shown its run of quiet samples. A build with neither is not gated (and the marker says so).
  const makingQuiet = !next.makingSeen || frame.making === 0;
  // The first time pipelines are quiet: from the wait samples when the build prints them (their run of three),
  // otherwise from the first frame that says none are being made.
  if (next.makingSeen && frame.making === 0 && next.metPipelinesAt === null && !next.pipelineSeen) {
    next.metPipelinesAt = frame.at;
  }
  const waitingQuiet = !next.pipelineSeen || next.pipelineRun >= SETTLED_LINES;
  const oldEnough = settleAfterS === null || (next.heavyAt !== null && frame.at - next.heavyAt >= settleAfterS * 1000.0);
  const meets = heavy && frame.fps >= floorFps && belowCeiling && makingQuiet && waitingQuiet && oldEnough;
  next.consecutive = meets ? next.consecutive + 1 : 0;
  next.runFps = meets ? [...(state.runFps ?? []), frame.fps].slice(-SETTLED_LINES) : [];
  const spread = next.runFps.length >= SETTLED_LINES ? fpsSpread(next.runFps) : null;
  const steady = maxSpread === null || (spread !== null && spread <= maxSpread);
  if (next.settledAt === null && next.consecutive >= SETTLED_LINES && steady) {
    next.settledAt = frame.at;
    next.settledWaiting = frame.waiting;
    next.settledFps = frame.fps;
    next.settledSpread = spread;
    next.settledDetail = detailFrom(next, quietMs, maxSpread);
  }
  return next;
}

/// Which of the settle's conditions is unmet now, for the once-a-minute line an unsettled run writes: a build that
/// keeps one pipeline being made, or never brings its pipeline wait under the threshold, would otherwise hold the
/// settle off until the timeout with nothing in the log saying why.
export function unmetText(state, options = {}) {
  if (state.heavyAt === null && options.worldTimeout != null && options.startedAt != null && options.now != null) {
    const left = Math.max(0, Math.round((options.worldTimeout * 1000 - (options.now - options.startedAt)) / 1000));
    return `waiting for the first heavy frame line; ${left} s left of the ${options.worldTimeout} s world timeout`;
  }
  const floorFps = options.floorFps ?? DEFAULT_FLOOR_FPS;
  const waiting = options.waiting ?? DEFAULT_WAITING;
  const quietMs = options.pipelineQuietMs ?? PIPELINE_QUIET_MS;
  const parts = [];
  if (state.lastFps === null || state.lastFps < floorFps) {
    parts.push(`frame rate ${state.lastFps === null ? 'not seen yet' : `${state.lastFps} fps under ${floorFps}`}`);
  }
  if (waiting !== null && state.lastWaiting !== null && state.lastWaiting > waiting) {
    parts.push(`${state.lastWaiting} textures waiting over ${waiting}`);
  }
  if (state.pipelineSeen && (state.pipelineRun < SETTLED_LINES || (state.waitMs ?? 0) >= quietMs)) {
    parts.push(`${state.pipelineRun} of ${SETTLED_LINES} quiet pipeline samples (last ${state.waitMs === null ? '?' : state.waitMs.toFixed(1)} ms, under ${quietMs.toFixed(1)} needed)`);
  }
  if (options.settleAfterS !== null && options.settleAfterS !== undefined && state.heavyAt !== null) {
    parts.push(`settle held until ${options.settleAfterS} s after the first heavy line`);
  }
  if (state.makingSeen && state.lastMaking !== 0) {
    parts.push(`${state.lastMaking === null ? '?' : state.lastMaking} pipelines being made`);
  }
  const maxSpread = spreadOption(options);
  const spread = (state.runFps ?? []).length >= SETTLED_LINES ? fpsSpread(state.runFps) : null;
  if (maxSpread !== null && spread !== null && !(spread <= maxSpread)) {
    parts.push(`frame rate not steady: spread ${percent(spread)} over the last ${SETTLED_LINES} settled lines (${percent(maxSpread)} allowed)`);
  }
  if (!parts.length) parts.push(`${state.consecutive} of ${SETTLED_LINES} settled frame lines so far`);
  return parts.join('; ');
}

export const unsettledMarker = (seconds, unmet) => `harness: not settled after ${seconds} s: ${unmet}`;

/// What the settle waited for, and how many seconds each took from the first heavy line.
function detailFrom(state, quietMs, maxSpread) {
  const since = (at) =>
    at === null || state.heavyAt === null ? null : Math.max(0, Math.round((at - state.heavyAt) / 1000));
  const parts = [];
  parts.push(since(state.metFpsAt) === null ? 'frame rate not met' : `frame rate from ${since(state.metFpsAt)} s`);
  if (!state.makingSeen && !state.pipelineSeen) {
    parts.push('pipeline quiet not checked');
  } else {
    const says = [];
    if (state.pipelineSeen) says.push(`${SETTLED_LINES} samples under ${quietMs.toFixed(1)} ms`);
    if (state.makingSeen) says.push('none being made');
    const from = since(state.metPipelinesAt);
    parts.push(`pipelines ${from === null ? 'quiet at the settle' : `from ${from} s`} (${says.join(', ')})`);
  }
  parts.push(
    state.metTexturesAt === null
      ? `textures not gated (${state.settledWaiting ?? '?'} waiting)`
      : `textures from ${since(state.metTexturesAt)} s`,
  );
  if (maxSpread === null) parts.push('frame rate steadiness not checked');
  else if (state.settledSpread !== null) {
    parts.push(`frame rate steady (spread ${percent(state.settledSpread)}, ${percent(maxSpread)} allowed)`);
  }
  return parts.join(', ');
}

/// What the watcher does now: end N seconds after the settle; time out when the scene has not settled within the
/// cap after the first heavy line; otherwise wait.
export function due(state, { now, seconds, timeout, worldTimeout = null, startedAt = null }) {
  if (state.settledAt !== null) {
    return now - state.settledAt >= seconds * 1000 ? 'end' : 'wait';
  }
  // A run that never reaches the world never has a heavy line, so the settled timeout below never starts: this
  // second limit runs from the watcher's start until the first heavy line is seen.
  if (state.heavyAt === null && worldTimeout !== null && startedAt !== null && now - startedAt >= worldTimeout * 1000) {
    return 'world-timeout';
  }
  if (state.heavyAt !== null && now - state.heavyAt >= timeout * 1000) return 'timeout';
  return 'wait';
}

export const settledMarker = (seconds, waiting, fps, detail = null) =>
  `harness: settled at ${seconds} s (waiting ${waiting ?? '?'}, ${fps} fps)` +
  (detail ? `; waited: ${detail}` : '');
export const settleEndMarker = (seconds) =>
  `harness: ended by --settled-seconds after ${seconds} s of settled scene`;
export const timeoutMarker = (seconds) =>
  `harness: ended by --settled-timeout after ${seconds} s: the scene never settled`;
export const worldTimeoutMarker = (seconds, last) =>
  `harness: ended by --world-timeout after ${seconds} s: the world never came (` +
  (last ? `last frame line: ${last.fps} fps, ${last.draws} draws/frame` : 'no frame line') +
  ')';

// run-process.ps1's strict selection, one pid per line. null means PowerShell failed, which is not evidence that
// the game is gone. Exported so the watchdog ends a run by this route too, not a new one.
export function selectionPids(psFile) {
  const result = spawnSync(
    'powershell.exe',
    ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', psFile, '-Pids', '-Strict'],
    { encoding: 'utf8', timeout: 15_000, windowsHide: true },
  );
  if (result.error || result.status !== 0) return null;
  return (result.stdout || '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => /^\d+$/.test(line))
    .map(Number);
}

export function isAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return err.code === 'EPERM';
  }
}

// taskkill needs /F: a plain taskkill asks the window to close, and this title ignores that.
export function endPid(pid) {
  const result = spawnSync('taskkill', ['/F', '/PID', String(pid)], {
    encoding: 'utf8',
    windowsHide: true,
  });
  return result.status === 0;
}

export function watch(logPath, seconds, options = {}) {
  const pollMs = options.pollMs ?? DEFAULT_POLL_MS;
  const timeout = options.timeout ?? DEFAULT_TIMEOUT_S;
  const worldTimeout = options.worldTimeout === null ? null : (options.worldTimeout ?? DEFAULT_WORLD_TIMEOUT_S);
  const startedAt = Date.now();
  const floorFps = options.floorFps ?? DEFAULT_FLOOR_FPS;
  const waitingCeiling = options.waiting ?? DEFAULT_WAITING;
  const pipelineQuietMs = options.pipelineQuietMs ?? PIPELINE_QUIET_MS;
  const settleAfterS = options.settleAfterS ?? null;
  const maxFpsSpread = spreadOption(options);
  const afterFile = options.afterFile ?? null;
  const psFile = options.psFile ?? path.join(HERE, 'run-process.ps1');
  return new Promise((resolve) => {
    let pos = 0;
    let pending = '';
    let state = initialState();
    let announced = false;
    let pids = null;
    let seenGame = false;
    let lastSelection = 0;
    let stopping = false;
    let timer = null;
    let lastNoteAt = null;

    const append = (line) => {
      try {
        fs.appendFileSync(logPath, `${line}\n`);
      } catch {
        // The console line below still says what happened; never take the watcher down over the log.
      }
      console.log(line);
    };

    const stop = (code) => {
      if (stopping) return;
      stopping = true;
      if (timer) clearInterval(timer);
      resolve(code);
    };

    const refreshPids = () => {
      lastSelection = Date.now();
      const found = selectionPids(psFile);
      if (found === null) return;
      pids = found;
      if (found.length) seenGame = true;
    };

    const gameAlive = () => {
      if (!seenGame) return true;
      if (!pids) return true;
      if (pids.some(isAlive)) return true;
      refreshPids();
      return pids ? pids.some(isAlive) : true;
    };

    const readNew = () => {
      let size;
      try {
        size = fs.statSync(logPath).size;
      } catch {
        return;
      }
      if (size < pos) {
        pos = 0;
        pending = '';
      }
      if (size === pos) return;
      let data;
      try {
        const fd = fs.openSync(logPath, 'r');
        try {
          const buffer = Buffer.alloc(size - pos);
          fs.readSync(fd, buffer, 0, buffer.length, pos);
          data = buffer.toString('utf8');
        } finally {
          fs.closeSync(fd);
        }
      } catch {
        return;
      }
      pos = size;
      pending += data;
      const lines = pending.split('\n');
      pending = lines.pop();
      for (const line of lines) {
        state = observe(state, line, { floorFps, waiting: waitingCeiling, pipelineQuietMs, settleAfterS, maxFpsSpread });
        if (state.settledAt !== null && !announced) {
          announced = true;
          const since = Math.max(0, Math.round((state.settledAt - state.heavyAt) / 1000));
          append(settledMarker(since, state.settledWaiting, state.settledFps, state.settledDetail));
          if (afterFile) {
            try {
              fs.writeFileSync(afterFile, '');
            } catch {
              append(`harness: settled, but the keys-after file ${afterFile} could not be made`);
            }
          }
        }
      }
    };

    // A fresh selection at the moment of the kill, never the cached ids (end-run.mjs says why).
    const killThisRun = () => {
      lastSelection = Date.now();
      const result = endSelected(() => selectionPids(psFile), endPid);
      if (result.fresh) pids = result.fresh;
      if (result.outcome === 'gone') {
        stop(0); // Already gone: it ended by itself.
        return false;
      }
      return result.outcome === 'ended';
    };

    const tick = () => {
      if (stopping) return;
      readNew();
      if (!seenGame && Date.now() - lastSelection >= 1000) refreshPids();

      if (state.heavyAt === null) {
        if (seenGame && !gameAlive()) stop(0);
        const now = Date.now();
        if (lastNoteAt === null) lastNoteAt = startedAt;
        if (now - lastNoteAt >= UNSETTLED_NOTE_MS) {
          append(unsettledMarker(Math.round((now - startedAt) / 1000), unmetText(state, { worldTimeout, startedAt, now })));
          lastNoteAt = now;
        }
        if (due(state, { now, seconds, timeout, worldTimeout, startedAt }) === 'world-timeout') {
          if (!killThisRun()) return; // Gone by itself (stop(0) inside), or taskkill failed: retry next poll.
          append(
            worldTimeoutMarker(
              Math.round((now - startedAt) / 1000),
              state.lastFps === null ? null : { fps: state.lastFps, draws: state.lastDraws },
            ),
          );
          stop(0);
        }
        return;
      }
      if (!gameAlive()) {
        stop(0);
        return;
      }
      const now = Date.now();
      if (state.settledAt === null) {
        if (lastNoteAt === null) lastNoteAt = state.heavyAt;
        if (now - lastNoteAt >= UNSETTLED_NOTE_MS) {
          append(unsettledMarker(Math.round((now - state.heavyAt) / 1000), unmetText(state, { floorFps, waiting: waitingCeiling, pipelineQuietMs, maxFpsSpread })));
          lastNoteAt = now;
        }
      }
      const action = due(state, { now, seconds, timeout });
      if (action === 'wait') return;
      if (action === 'end') {
        if (!killThisRun()) return; // Gone by itself (stop(0) inside), or taskkill failed: retry next poll.
        append(settleEndMarker(seconds));
        stop(0);
        return;
      }
      if (!killThisRun()) return;
      append(timeoutMarker(Math.round((now - state.heavyAt) / 1000)));
      stop(0);
    };

    process.once('SIGINT', () => stop(0));
    process.once('SIGTERM', () => stop(0));
    refreshPids();
    timer = setInterval(tick, pollMs);
    tick();
  });
}

function usage() {
  console.error(
    'usage: node settled-watch.mjs <log> <seconds> [--timeout 600] [--world-timeout 600|off] [--floor-fps 1.5] [--waiting K] [--pipeline-quiet-ms 5] [--max-fps-spread 0.25|off] [--settle-after-s N] [--poll-ms 250] [--after-file <path>]',
  );
}

async function main(argv) {
  const positional = [];
  let pollMs = DEFAULT_POLL_MS;
  let timeout = DEFAULT_TIMEOUT_S;
  let worldTimeout = DEFAULT_WORLD_TIMEOUT_S;
  let floorFps = DEFAULT_FLOOR_FPS;
  let waiting = DEFAULT_WAITING;
  let pipelineQuietMs = PIPELINE_QUIET_MS;
  let settleAfterS = null;
  let maxFpsSpread = MAX_FPS_SPREAD;
  let afterFile = null;
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--poll-ms') pollMs = Number(argv[++i]);
    else if (arg === '--timeout') timeout = Number(argv[++i]);
    else if (arg === '--world-timeout') {
      const value = argv[++i];
      worldTimeout = value === 'off' ? null : Number(value);
    }
    else if (arg === '--floor-fps') floorFps = Number(argv[++i]);
    else if (arg === '--waiting') waiting = Number(argv[++i]);
    else if (arg === '--pipeline-quiet-ms') pipelineQuietMs = Number(argv[++i]);
    else if (arg === '--settle-after-s') settleAfterS = Number(argv[++i]);
    else if (arg === '--max-fps-spread') {
      const value = argv[++i];
      maxFpsSpread = value === 'off' ? null : Number(value);
    }
    else if (arg === '--after-file') afterFile = argv[++i];
    else if (arg.startsWith('--')) {
      console.error(`settled-watch: unknown argument ${arg}`);
      usage();
      return 2;
    } else {
      positional.push(arg);
    }
  }
  const seconds = Number(positional[1]);
  const bad =
    positional.length !== 2 ||
    !Number.isFinite(seconds) ||
    seconds < 0 ||
    !Number.isFinite(timeout) ||
    timeout <= 0 ||
    !Number.isFinite(floorFps) ||
    floorFps <= 0 ||
    !(waiting === null || (Number.isFinite(waiting) && waiting >= 0)) ||
    !Number.isFinite(pollMs) ||
    pollMs <= 0 ||
    !Number.isFinite(pipelineQuietMs) ||
    pipelineQuietMs <= 0 ||
    !(settleAfterS === null || (Number.isFinite(settleAfterS) && settleAfterS >= 0)) ||
    !(worldTimeout === null || (Number.isFinite(worldTimeout) && worldTimeout > 0)) ||
    !(maxFpsSpread === null || (Number.isFinite(maxFpsSpread) && maxFpsSpread > 0));
  if (bad) {
    usage();
    return 2;
  }
  return watch(positional[0], seconds, {
    pollMs,
    timeout,
    worldTimeout,
    floorFps,
    waiting,
    pipelineQuietMs,
    settleAfterS,
    maxFpsSpread,
    afterFile,
  });
}

const isMain =
  process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isMain) {
  main(process.argv.slice(2))
    .then((code) => process.exit(code))
    .catch((err) => {
      console.error(err);
      process.exit(1);
    });
}
