#!/usr/bin/env node
// Watch a run's frame counter: end a frozen timed run, or warn in an open, untimed session.
//
//   node harness/watchdog.mjs <log> [--stall-seconds 45] [--poll-ms 1000] [--stats <stats.json>] [--warn]
//                              [--settled-fps F] [--settled-waiting K] [--settled-after-s N]
//
// The counter is the `frame` field of the runtime's stats file when --stats names one (a --live run), and the newest
// frame number in the run's log otherwise. A build whose frame lines carry no `frame N` has no counter, and the
// watcher says so: "not-armed: no frame counter". The runtime writes both the stats file and the log's frame line
// from a present, inside its 5-second gate: about once every 6 s, and not at all while nothing presents. The window
// (--stall-seconds, 45 by default) must stay well over that cadence.
//
// What a stall is. Once armed, the watcher reads the stats file (or, without one, the log) every poll. Progress is
// the stats counter changing or a new frame line arriving in the log; "changed" means any change, so a counter that
// goes backwards or flips between two values counts as progress. The stats file has two shapes (stats-object.mjs).
// In the interval shape the counter is the object's `seq`: a new seq is progress and a write, dated by the object's
// own clock (`unix_ms`); the same seq read again is neither, whatever the file's time says; a jump is objects missed,
// counted. The runtime writes an object only at a present, so a pause is no new seq and then one object holding the
// whole gap, and the watcher's warnings give the newest object's age by its own clock. In the older shape the counter
// is the frame, and a rewrite is a changed file time or contents. A stall is declared only when nothing has
// progressed for the whole window AND there is positive evidence that frames stopped:
//   - the stats file is still being rewritten (older shape: its modification time or its contents changed within the
//     window) while its frame counter did not move; or
//   - the log was read successfully at least a window after its last frame line (no frame line for the window).
// Without either, the watcher cannot tell a frozen game from a stats file it cannot read or that stopped being
// written, and it writes one `harness: watchdog: warning: cannot tell ...; NOT ending the run` line instead: a
// warning never ends a run. A stats counter that stands still for the window while the log keeps showing frame lines
// is a healthy run whose stats writes are failing: `harness: watchdog: warning: the stats counter has not moved ...
// but the log shows frames ...`. Once armed, every failed read of the stats file (missing, unreadable, torn or
// without a frame number) and of the log is counted; the evidence line before a stall, each warning, and the closing
// `harness: watchdog: done: ...` line (written when anything failed or warned) carry the counts.
// Named limit: a load after the settle (a fast travel) presents nothing, so a load longer than the window is a stall
// by this rule; that is why an open, untimed session uses warn mode. A freeze in which presents continue is not seen.
//
// It arms only when the run has settled in the world: the same three settled frame lines settled-watch.mjs uses
// (frame rate, pipeline quiet, optional texture ceiling), or that watcher's `harness: settled at ...` line. Loading
// and shader compilation may stop the counter for a long time before that, so nothing is counted until it arms.
// An ending line in the log disarms it.
//
// In the default, ending mode, on a stall it appends
//   harness: watchdog: evidence: <which of the two above>; stats file: K failed reads of M[, log: L failed reads]
//   harness: watchdog: the frame counter stopped at frame N for S s; ending the run
// to the log, writes the log's last 300 lines beside it as <log without .log>-stall-tail.log, and ends this run's
// victory.exe by the route settled-watch.mjs uses (run-process.ps1's strict selection, then taskkill /F). A game
// that has already gone is not touched.
//
// With --warn it never ends the run. On a stall it appends the same line with
// `; NOT ending the run (warn mode)`, saves the tail file once for that stall, and keeps watching; when the counter
// moves again it appends `harness: watchdog: frames resumed at frame N after S s`. summarize-run.mjs reports the
// stalls it finds in the log. An open session's counter can stand still while the window is minimised or covered,
// during a long load after a fast travel, or in the title's own pause, so a false end is worse than the freeze.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { endSelected } from './end-run.mjs';

import {
  FRAME_RE,
  HARNESS_DISK_RE,
  HARNESS_ENDED_RE,
  HARNESS_SETTLE_TIMEOUT_RE,
  HARNESS_SETTLED_RE,
  HARNESS_VICTORY_EXIT_RE,
  stripAnsi,
} from './parse-run.mjs';
import {
  endPid,
  initialState as settledInitialState,
  isAlive,
  observe as observeSettled,
  selectionPids,
} from './settled-watch.mjs';
import { readStatsObject, sequenceOf } from './stats-object.mjs';

export const DEFAULT_STALL_S = 45;
export const DEFAULT_POLL_MS = 1000;
export const TAIL_LINES = 300;
// The newest frame line is always in the tail; a long run's log is many megabytes. A frame line is about 1.3 KB and
// comes every few seconds, so this covers minutes of ordinary lines.
export const LOG_TAIL_BYTES = 262144;
// A live run whose stats file has not appeared yet is not declared counterless on its first armed second.
export const NO_COUNTER_NOTE_MS = 10_000;

const HERE = path.dirname(fileURLToPath(import.meta.url));

/// The stats file's frame counter, or null for a missing file, a half-written one, or one without a numeric frame.
/// The runtime rewrites it atomically, so a half-written file should not happen; the guard is what keeps a torn read
/// from looking like a stopped counter.
export function statsFrame(text) {
  if (typeof text !== 'string' || text.trim() === '') return null;
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    return null;
  }
  if (!data || typeof data.frame !== 'number' || !Number.isFinite(data.frame)) return null;
  return data.frame;
}

/// One read of the stats file: `{ frame, seq, unixMs, shape, mtimeMs, text }` when it holds a numeric frame, else
/// `{ frame: null, error }` with error `missing`, `unreadable`, `unparsable` (empty or torn) or `no frame`. A file
/// without `frame` is a failed read, never frame 0. Either shape is read (stats-object.mjs): `seq` and `unixMs` are
/// the interval shape's identity and own clock, null for the older sample.
export function readStats(file) {
  if (!file) return { frame: null, error: 'missing' };
  let text;
  let mtimeMs;
  try {
    mtimeMs = fs.statSync(file).mtimeMs;
    text = fs.readFileSync(file, 'utf8');
  } catch (err) {
    return { frame: null, error: err && err.code === 'ENOENT' ? 'missing' : 'unreadable' };
  }
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    return { frame: null, error: 'unparsable' };
  }
  const record = readStatsObject(data);
  if (!record) return { frame: null, error: 'no frame' };
  return { frame: record.frame, seq: record.seq, unixMs: record.unixMs, shape: record.shape, mtimeMs, text };
}

export function readStatsFrame(file) {
  return readStats(file).frame;
}

/// The newest frame line's own frame number, and whether any frame line was found. A frame line without a number
/// means this build keeps no usable counter in the log.
export function logFrame(text) {
  if (typeof text !== 'string' || text === '') return { frame: null, sawFrameLine: false };
  const lines = stripAnsi(text).split(/\r?\n/);
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const m = lines[i].match(FRAME_RE);
    if (!m) continue;
    if (m.groups && m.groups.frame !== undefined) {
      const frame = Number(m.groups.frame);
      if (Number.isFinite(frame)) return { frame, sawFrameLine: true };
    }
    return { frame: null, sawFrameLine: true };
  }
  return { frame: null, sawFrameLine: false };
}

/// The log's last LOG_TAIL_BYTES, or null when it cannot be read (the harness creates the log before starting us).
export function readLogTail(file, bytes = LOG_TAIL_BYTES) {
  let fd;
  try {
    fd = fs.openSync(file, 'r');
  } catch {
    return null;
  }
  try {
    const size = fs.fstatSync(fd).size;
    const start = Math.max(0, size - bytes);
    const buffer = Buffer.alloc(size - start);
    if (buffer.length) fs.readSync(fd, buffer, 0, buffer.length, start);
    return buffer.toString('utf8');
  } catch {
    return null;
  } finally {
    fs.closeSync(fd);
  }
}

/// A fresh watcher. `source` is 'stats' when a stats file was named, else 'log'. `settle` is the settled watcher's
/// own state, so arming means the same thing this harness already means by settled. `warn` keeps the watcher from
/// ever ending the run; it records each stall and the resume that follows it.
export function initialState({ source = 'log', warn = false } = {}) {
  return {
    source,
    warn: Boolean(warn),
    armed: false,
    armedAt: null,
    ended: false,
    frame: null,
    advancedAt: null,
    sawFrameLine: false,
    noFrameNumber: false,
    settle: settledInitialState(),
    // Warn mode's tally: one entry per stall, the longest seen, the stall in progress, and the resume the next
    // sample has just found (the caller prints it, then clears it).
    stalls: 0,
    longestS: 0,
    stall: null,
    resume: null,
    // The evidence: when the log was last read whole and when a frame line last arrived in it; when the stats file
    // was last read and when it was last seen rewritten (its modification time changed). Counts are from the arm.
    logReadAt: null,
    logFrameLineAt: null,
    logFailures: 0,
    statsReadAt: null,
    statsMtimeMs: null,
    statsText: null,
    statsWrittenAt: null,
    statsReads: 0,
    statsFailures: 0,
    warnings: 0,
    // The interval shape's identity: the newest object's seq, its own write time (unix_ms), and the objects written
    // and never read (seq jumps) since the arm.
    seq: null,
    statsObjectAt: null,
    statsMissed: 0,
  };
}

/// One poll's read of the log: `ok` false (it could not be read) is counted once armed.
export function noteLogRead(state, { at = Date.now(), ok = true } = {}) {
  if (state.ended) return state;
  if (!ok) return state.armed ? { ...state, logFailures: state.logFailures + 1 } : state;
  return { ...state, logReadAt: at };
}

/// One poll's read of the stats file (`readStats`). A failed read is counted once armed and is no sample; a good one
/// notes whether the file was rewritten since the last read, then samples its counter.
export function sampleStats(state, { at = Date.now(), read } = {}) {
  if (state.ended || !read) return state;
  if (!Number.isFinite(read.frame)) {
    return state.armed ? { ...state, statsReads: state.statsReads + 1, statsFailures: state.statsFailures + 1 } : state;
  }
  let next = state.armed ? { ...state, statsReads: state.statsReads + 1 } : { ...state };
  const mtimeMs = Number.isFinite(read.mtimeMs) ? read.mtimeMs : null;
  const text = typeof read.text === 'string' ? read.text : null;
  if (Number.isFinite(read.seq)) {
    // The interval shape: the runtime writes an object only at a present, and its seq is its identity. A new seq is
    // a write and progress, dated by the object's own clock (unix_ms; the read's time if it has none, or if that is
    // later); the same seq read again is neither, whatever the file's time or contents say. A jump is objects missed.
    const seen = sequenceOf(next.seq === null ? null : { seq: next.seq }, { seq: read.seq });
    const fresh = seen.kind !== 'repeat';
    const writtenAt = fresh ? Math.min(at, Number.isFinite(read.unixMs) ? read.unixMs : at) : null;
    next = {
      ...next,
      statsMtimeMs: mtimeMs ?? next.statsMtimeMs,
      statsText: text ?? next.statsText,
      ...(fresh ? { statsWrittenAt: writtenAt, statsObjectAt: Number.isFinite(read.unixMs) ? read.unixMs : null } : {}),
      ...(fresh && next.armed ? { statsMissed: next.statsMissed + seen.missed } : {}),
    };
    return sampleFrame(next, { at, frame: read.frame, seq: read.seq, writtenAt });
  }
  // The older shape has no seq. The first sight of the file is no evidence of a write; a changed time, or changed
  // contents (a timestamp field, whatever the runtime names it), is.
  const moved = mtimeMs !== null && next.statsMtimeMs !== null && mtimeMs !== next.statsMtimeMs;
  const changed = text !== null && next.statsText !== null && text !== next.statsText;
  if (moved || changed) next = { ...next, statsWrittenAt: at };
  next = { ...next, statsMtimeMs: mtimeMs ?? next.statsMtimeMs, statsText: text ?? next.statsText };
  return sampleFrame(next, { at, frame: read.frame });
}

/// One more log line. The settled watcher's rule arms the state; a `harness: settled at ...` line arms it too (an
/// untimed session prints no such line, but its frame lines reach the same rule). An ending line disarms it for
/// good. Frame lines are noted only for whether they carry a frame number: the counter itself is sampled from the
/// stats file or the log tail, as `sampleFrame` below.
export function noteLine(state, raw, options = {}) {
  const now = options.now ?? Date.now();
  const line = stripAnsi(String(raw)).replace(/\r$/, '');
  if (isEndingLine(line)) return { ...state, ended: true, armed: false, advancedAt: null };
  const settle = observeSettled(state.settle, line, {
    floorFps: options.floorFps,
    waiting: options.waiting,
    settleAfterS: options.settleAfterS,
    pipelineQuietMs: options.pipelineQuietMs,
  });
  // A line read at `now` is a successful read of the log at `now`.
  let next = { ...state, settle, logReadAt: now };
  const settled = HARNESS_SETTLED_RE.test(line) || settle.settledAt !== null;
  if (settled && !state.armed) next = { ...next, armed: true, armedAt: now, advancedAt: null };
  const m = line.match(FRAME_RE);
  if (m) {
    next = { ...next, sawFrameLine: true, logFrameLineAt: now };
    if (!m.groups || m.groups.frame === undefined) next = { ...next, noFrameNumber: true };
  }
  return next;
}

/// The log line that means the run is ending. This watcher's own lines are not here: its no-counter note is not an
/// end, and its stall line is followed by the end it starts itself.
export function isEndingLine(raw) {
  const line = stripAnsi(String(raw)).replace(/\r$/, '');
  return (
    HARNESS_ENDED_RE.test(line) ||
    HARNESS_SETTLE_TIMEOUT_RE.test(line) ||
    HARNESS_DISK_RE.test(line) ||
    HARNESS_VICTORY_EXIT_RE.test(line)
  );
}

/// One frame-counter sample, read at `at` from the stats file (stats source) or the log (log source), so it is also
/// a successful read of that source. Before arming the value is only remembered (so a counterless run can say so);
/// after arming, the first sample starts the clock, a changed value restarts it, and an unchanged value leaves it
/// running. A lower value is a restart of the counter, not a stall. In warn mode a change while a stall is in
/// progress also records the resume (frame and seconds stood still) for the caller to print; it is never an end.
/// With `seq` (the interval shape's object identity) the counter is the seq, not the frame: a new seq is progress and
/// a repeated one is not. `writtenAt` dates an advance by the object's own clock instead of the read's.
export function sampleFrame(state, { at = Date.now(), frame, seq = null, writtenAt = null } = {}) {
  if (!Number.isFinite(frame) || state.ended) return state;
  const read = state.source === 'stats' ? { statsReadAt: at } : { logReadAt: at };
  const keyed = Number.isFinite(seq);
  if (!state.armed) return { ...state, ...read, frame, ...(keyed ? { seq } : {}) };
  const moved = keyed ? state.seq !== seq : state.frame !== frame;
  if (state.advancedAt === null || moved) {
    const advancedAt = Number.isFinite(writtenAt) ? writtenAt : at;
    const next = { ...state, ...read, frame, ...(keyed ? { seq } : {}), advancedAt };
    if (state.warn && state.stall) {
      // The stall began at the last advance, so this is the whole time the counter stood still.
      const seconds = Math.max(1, Math.round((advancedAt - state.stall.since) / 1000));
      return {
        ...next,
        stall: null,
        longestS: Math.max(state.longestS, seconds),
        resume: { frame, seconds },
      };
    }
    return next;
  }
  return { ...state, ...read };
}

const secondsOf = (ms) => Math.max(0, Math.round(ms / 1000));

/// What the watcher says now (the header's rule): `ok`; `stalled` with its `evidence` (`stats` or `log`), the frame
/// and the seconds the counter has stood still; `warning` with its `kind` (`stats-not-advancing`: the log shows
/// frames the stats file does not; `cannot-tell`: no evidence either way) and `reason`; or `not-armed` with the
/// reason. A counterless armed run says `not-armed: no frame counter`. Every result carries the read counts.
export function statusOf(state, { now = Date.now(), stallS = DEFAULT_STALL_S } = {}) {
  const counts = {
    source: state.source,
    statsReads: state.statsReads ?? 0,
    statsFailures: state.statsFailures ?? 0,
    statsMissed: state.statsMissed ?? 0,
    logFailures: state.logFailures ?? 0,
  };
  // The newest interval-shape object's age by its own clock, for the reasons below (null for the older shape).
  const objectAge =
    state.statsObjectAt == null
      ? null
      : `the newest stats object (seq ${state.seq}) was written ${secondsOf(now - state.statsObjectAt)} s ago by its own clock`;
  if (state.ended) return { status: 'not-armed', reason: 'run ending', frame: state.frame, seconds: null, ...counts };
  if (!state.armed) return { status: 'not-armed', reason: 'not settled', frame: state.frame, seconds: null, ...counts };
  if (state.frame === null) return { status: 'not-armed', reason: 'no frame counter', frame: null, seconds: null, ...counts };
  const windowMs = stallS * 1000;
  // No sample since the arm counts from the arm: the counter was known then, and nothing has advanced it.
  const since = state.advancedAt ?? state.armedAt;
  const elapsed = since === null ? 0 : Math.max(0, now - since);
  const base = { frame: state.frame, seconds: secondsOf(elapsed), ...counts };
  if (since === null) return { status: 'ok', ...base };
  const stats = state.source === 'stats';
  const logLast = state.logFrameLineAt ?? null;
  // Progress from either instrument: the counter changing, or, on a stats run, a frame line arriving in the log.
  const lastProgress = stats && logLast !== null ? Math.max(since, logLast) : since;
  if (now - lastProgress < windowMs) {
    if (stats && elapsed >= windowMs && logLast !== null && logLast - since >= windowMs) {
      // The log kept its frame lines for a whole window after the stats counter last moved: the run presents, and
      // the stats file is what stopped.
      return {
        status: 'warning',
        kind: 'stats-not-advancing',
        reason:
          `the stats counter has not moved from frame ${state.frame} for ${base.seconds} s, but the log shows frames ` +
          `(the stats file is not being written, or cannot be read)${objectAge ? `; ${objectAge}` : ''}`,
        ...base,
      };
    }
    return { status: 'ok', ...base };
  }
  // Nothing has progressed for the window. A stall needs positive evidence that frames stopped.
  if (stats && state.statsWrittenAt !== null && now - state.statsWrittenAt < windowMs) {
    return { status: 'stalled', evidence: 'stats', quietS: base.seconds, ...base };
  }
  const logQuietFrom = stats ? logLast : since;
  if (logQuietFrom !== null && state.logReadAt !== null && state.logReadAt - logQuietFrom >= windowMs) {
    return { status: 'stalled', evidence: 'log', quietS: secondsOf(state.logReadAt - logQuietFrom), ...base };
  }
  const parts = [];
  if (stats) {
    const goodRead = state.statsReadAt ?? null;
    parts.push(
      goodRead === null || now - goodRead >= windowMs
        ? `no good read of the stats file for ${secondsOf(now - (goodRead ?? since))} s`
        : (objectAge ?? `the stats file has not been rewritten for ${base.seconds} s`),
    );
  }
  if (logQuietFrom === null) parts.push('the log has no frame lines to judge by');
  else parts.push(`the log has not been read for ${secondsOf(now - (state.logReadAt ?? since))} s`);
  return {
    status: 'warning',
    kind: 'cannot-tell',
    reason: `cannot tell whether frames stopped at frame ${state.frame}: ${parts.join(', and ')}`,
    ...base,
  };
}

export function statusText(result) {
  if (result.status === 'not-armed') return `not-armed${result.reason ? `: ${result.reason}` : ''}`;
  if (result.status === 'warning') return `warning: ${result.reason}`;
  return result.status;
}

/// The read counts a verdict carries.
export function verdictText(result) {
  const log = `log: ${result.logFailures ?? 0} failed reads`;
  if (result.source !== 'stats') return log;
  const missed = result.statsMissed ? `, ${result.statsMissed} objects missed` : '';
  const stats = `stats file: ${result.statsFailures ?? 0} failed reads of ${result.statsReads ?? 0}${missed}`;
  return result.logFailures ? `${stats}, ${log}` : stats;
}

/// The line written before a stall line: what showed that frames stopped, and the counts.
export function evidenceMarker(result) {
  const what =
    result.evidence === 'stats'
      ? `the stats file is still being rewritten and its frame counter stayed at ${result.frame} for ${result.seconds} s`
      : `the log was read with no frame line in it for ${result.quietS} s`;
  return `harness: watchdog: evidence: ${what}; ${verdictText(result)}`;
}

export const warningMarker = (result) =>
  `harness: watchdog: warning: ${result.reason}; ${verdictText(result)}; NOT ending the run`;

/// The closing tally, written when the run ends and anything failed or warned.
export function doneMarker(state) {
  const result = statusOf({ ...state, ended: true });
  return `harness: watchdog: done: ${state.stalls} stalls, ${state.warnings} warnings; ${verdictText(result)}`;
}

export const stallMarker = (frame, seconds, { warn = false } = {}) =>
  `harness: watchdog: the frame counter stopped at frame ${frame} for ${seconds} s; ` +
  (warn ? 'NOT ending the run (warn mode)' : 'ending the run');

export const warnStallMarker = (frame, seconds) => stallMarker(frame, seconds, { warn: true });

export const resumeMarker = (frame, seconds) =>
  `harness: watchdog: frames resumed at frame ${frame} after ${seconds} s`;

export const noCounterMarker = (status) => `harness: watchdog: ${statusText(status)}`;

/// <log without .log>-stall-tail.log, beside the run's log.
export function tailPathFor(logPath) {
  return `${String(logPath).replace(/\.log$/i, '')}-stall-tail.log`;
}

/// The last `count` lines of a log, read backwards so a long log is not loaded whole. Returns text with one final
/// newline, or '' for an unreadable or empty file.
export function tailLines(file, count = TAIL_LINES) {
  let fd;
  try {
    fd = fs.openSync(file, 'r');
  } catch {
    return '';
  }
  try {
    const size = fs.fstatSync(fd).size;
    const chunks = [];
    let end = size;
    let newlines = 0;
    const chunkSize = 64 * 1024;
    while (end > 0 && newlines <= count) {
      const start = Math.max(0, end - chunkSize);
      const buffer = Buffer.alloc(end - start);
      fs.readSync(fd, buffer, 0, buffer.length, start);
      chunks.unshift(buffer);
      for (const byte of buffer) if (byte === 0x0a) newlines += 1;
      end = start;
    }
    const lines = Buffer.concat(chunks).toString('utf8').split(/\r?\n/);
    if (lines.length && lines[lines.length - 1] === '') lines.pop();
    return lines.length ? `${lines.slice(-count).join('\n')}\n` : '';
  } catch {
    return '';
  } finally {
    fs.closeSync(fd);
  }
}

/// The real clock; a test passes its own (virtual-clock.mjs) as `options.clock`, so that no stall or warning in it
/// depends on how fast the machine is.
const REAL_CLOCK = Object.freeze({
  now: () => Date.now(),
  every(ms, fn) {
    const timer = setInterval(fn, ms);
    return () => clearInterval(timer);
  },
});

export function watch(logPath, options = {}) {
  const clock = options.clock ?? REAL_CLOCK;
  const pollMs = options.pollMs ?? DEFAULT_POLL_MS;
  const stallS = options.stallSeconds ?? DEFAULT_STALL_S;
  const statsFile = options.statsFile ?? null;
  const source = statsFile ? 'stats' : 'log';
  const warn = options.warn ?? options.mode === 'warn';
  const psFile = options.psFile ?? path.join(HERE, 'run-process.ps1');
  const tailCount = options.tailLines ?? TAIL_LINES;
  const settleOptions = {
    floorFps: options.floorFps,
    waiting: options.waiting,
    settleAfterS: options.settleAfterS,
    pipelineQuietMs: options.pipelineQuietMs,
  };
  return new Promise((resolve) => {
    let pos = 0;
    let pending = '';
    let state = initialState({ source, warn });
    let pids = null;
    let seenGame = false;
    let lastSelection = 0;
    let stopping = false;
    let cancel = null;
    let announced = false;
    let notedNoCounter = false;
    let warnedKind = null;

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
      if (cancel) cancel();
      resolve(code);
    };

    // The run has ended: the closing tally, when anything failed or warned.
    const finish = () => {
      if (state.statsFailures || state.logFailures || state.warnings) append(doneMarker(state));
      stop(0);
    };

    const refreshPids = () => {
      lastSelection = clock.now();
      const found = selectionPids(psFile);
      if (found === null) return;
      pids = found;
      if (found.length) seenGame = true;
    };

    const gameAlive = () => {
      if (!seenGame) return true; // Still starting: not yet evidence of a game, let alone a gone one.
      if (!pids) return true;
      if (pids.some(isAlive)) return true;
      refreshPids();
      return pids ? pids.some(isAlive) : true;
    };

    /// Reads what the log gained since the last poll; false when it could not be read.
    const readNew = () => {
      let size;
      try {
        size = fs.statSync(logPath).size;
      } catch {
        return false; // The harness creates the log before it starts us; give it a moment.
      }
      if (size < pos) {
        pos = 0;
        pending = '';
      }
      if (size === pos) return true;
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
        return false;
      }
      pos = size;
      pending += data;
      const lines = pending.split('\n');
      pending = lines.pop();
      for (const line of lines) {
        state = noteLine(state, line, { now: clock.now(), ...settleOptions });
        if (state.ended) {
          finish();
          return true;
        }
      }
      return true;
    };

    const noteNoCounter = (status) => {
      if (notedNoCounter) return;
      notedNoCounter = true;
      append(noCounterMarker(status));
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

    const writeTail = () => {
      try {
        fs.writeFileSync(tailPathFor(logPath), tailLines(logPath, tailCount));
      } catch {
        // The line in the log still says what happened; a missing tail is not a reason to keep a frozen run.
      }
    };

    const tick = () => {
      if (stopping) return;
      let logOk = readNew();
      if (stopping) return;
      if (!seenGame && clock.now() - lastSelection >= 1000) refreshPids();

      const now = clock.now();
      const beforeSample = state;
      if (source === 'stats') {
        state = sampleStats(state, { at: now, read: readStats(statsFile) });
      } else {
        const tail = readLogTail(logPath);
        if (tail === null) logOk = false;
        else {
          const found = logFrame(tail);
          if (found.sawFrameLine) {
            state = { ...state, sawFrameLine: true };
            if (found.frame === null) state = { ...state, noFrameNumber: true };
            else state = sampleFrame(state, { at: now, frame: found.frame });
          }
        }
      }
      state = noteLogRead(state, { at: now, ok: logOk });
      if (warn && state.resume && state.resume !== beforeSample.resume) {
        // The counter moved after a stall we reported: say when, and keep watching.
        append(resumeMarker(state.resume.frame, state.resume.seconds));
        state = { ...state, resume: null };
        announced = false;
      }

      if (seenGame && !gameAlive()) {
        finish();
        return;
      }

      const status = statusOf(state, { now, stallS });
      if (status.status === 'warning') {
        // Not evidence of a freeze: said once an episode, and never an end.
        if (warnedKind !== status.kind) {
          warnedKind = status.kind;
          state = { ...state, warnings: state.warnings + 1 };
          append(warningMarker(status));
        }
        return;
      }
      if (status.status === 'ok') warnedKind = null;
      if (status.status === 'stalled') {
        // A counter that stopped because the game is gone is no frozen run: when a fresh selection finds no game, the
        // run ended by itself, and the watchdog stops without a stall line (which the summary would read as its ending).
        if (!warn && !announced) {
          const fresh = selectionPids(psFile);
          if (fresh !== null && !fresh.length) {
            stop(0);
            return;
          }
        }
        if (!announced) {
          announced = true;
          append(evidenceMarker(status));
          append(stallMarker(status.frame, status.seconds, { warn }));
          writeTail();
          if (warn) {
            state = {
              ...state,
              stalls: state.stalls + 1,
              longestS: Math.max(state.longestS, status.seconds),
              stall: {
                frame: status.frame,
                seconds: status.seconds,
                since: state.advancedAt ?? state.armedAt ?? now,
              },
            };
          }
        }
        if (warn) return; // Warn mode never ends the run; the tally is in the log and the stall lasts as long as it does.
        if (!killThisRun()) return; // Gone by itself (stop(0) inside), or taskkill failed: retry next poll.
        stop(0);
        return;
      }
      if (status.status === 'not-armed' && status.reason === 'no frame counter') {
        const waited = state.armedAt === null ? 0 : now - state.armedAt;
        const grace = source === 'log' ? 0 : Math.min(NO_COUNTER_NOTE_MS, stallS * 1000);
        if (state.noFrameNumber || (source === 'stats' && waited >= grace)) noteNoCounter(status);
      }
    };

    process.once('SIGINT', () => stop(0));
    process.once('SIGTERM', () => stop(0));
    refreshPids();
    cancel = clock.every(pollMs, tick);
    tick();
  });
}

function usage() {
  console.error(
    'usage: node watchdog.mjs <log> [--stall-seconds 45] [--poll-ms 1000] [--stats <stats.json>] [--warn] [--settled-fps F] [--settled-waiting K] [--settled-after-s N]',
  );
}

async function main(argv) {
  const positional = [];
  let stallSeconds = DEFAULT_STALL_S;
  let pollMs = DEFAULT_POLL_MS;
  let statsFile = null;
  let warn = false;
  let floorFps;
  let waiting;
  let settleAfterS;
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--stall-seconds') stallSeconds = Number(argv[++i]);
    else if (arg === '--poll-ms') pollMs = Number(argv[++i]);
    else if (arg === '--warn' || arg === '--warn-only') warn = true;
    else if (arg === '--mode') {
      const mode = argv[++i];
      if (mode === 'warn') warn = true;
      else if (mode === 'end' || mode === '1') warn = false;
      else {
        usage();
        return 2;
      }
    } else if (arg === '--stats') {
      statsFile = argv[++i] ?? null;
      if (!statsFile) {
        usage();
        return 2;
      }
    }
    else if (arg === '--settled-fps') floorFps = Number(argv[++i]);
    else if (arg === '--settled-waiting') waiting = Number(argv[++i]);
    else if (arg === '--settled-after-s') settleAfterS = Number(argv[++i]);
    else if (arg.startsWith('--')) {
      console.error(`watchdog: unknown argument ${arg}`);
      usage();
      return 2;
    } else {
      positional.push(arg);
    }
  }
  const bad =
    positional.length !== 1 ||
    !Number.isFinite(stallSeconds) ||
    stallSeconds <= 0 ||
    !Number.isFinite(pollMs) ||
    pollMs <= 0 ||
    !(floorFps === undefined || (Number.isFinite(floorFps) && floorFps > 0)) ||
    !(waiting === undefined || (Number.isFinite(waiting) && waiting >= 0)) ||
    !(settleAfterS === undefined || (Number.isFinite(settleAfterS) && settleAfterS >= 0));
  if (bad) {
    usage();
    return 2;
  }
  return watch(positional[0], { stallSeconds, pollMs, statsFile, warn, floorFps, waiting, settleAfterS });
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isMain) {
  main(process.argv.slice(2))
    .then((code) => process.exit(code))
    .catch((err) => {
      console.error(err);
      process.exit(1);
    });
}
